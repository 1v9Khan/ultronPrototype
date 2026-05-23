"""History processors for LLM message-list compression.

Direct port of patterns from SWE-Agent's
``sweagent/agent/history_processors.py`` (MIT, Yang et al. 2024).
Three load-bearing processors land here, all pure (no I/O, no
globals, no model calls):

* **T2 :class:`ClosedWindowHistoryProcessor`.** Walks the history
  in reverse. When the same file's view appears multiple times,
  keep the most recent verbatim and replace older snapshots with
  an ``Outdated window with N lines omitted...`` summary. Catches
  the common "I opened foo.py, then later read foo.py again, now
  the conversation has two copies of the same 100-line block"
  redundancy.

* **T9 :class:`LastNObservations`.** Keep the last N observations
  verbatim; elide older ones to
  ``Old environment output: (M lines omitted)``. The ``polling``
  parameter slows the elision-window update so the cache stays
  warm for multiple turns at a time -- the naive sliding window
  flips the prompt prefix every turn and nukes Anthropic prompt
  caching. Per-tag ``always_keep_output`` / ``always_remove_output``
  overrides let specific tool outputs survive (e.g. the final
  ``submit`` diff) or always be elided (e.g. ``view_image``
  base64 blobs).

* **T9 companion :class:`TagToolCallObservations`.** Walks the
  history once and tags observations produced by specific tool
  names. Used together with :class:`LastNObservations` to
  selectively keep/elide observations by their source tool without
  the model having to know about tags.

The processors operate on a generic history shape:

.. code-block:: python

    HistoryItem = TypedDict("HistoryItem", {
        "role": str,                  # "user" | "assistant" | "tool" | "system"
        "content": str | list[dict],  # text OR multimodal segments
        "message_type": NotRequired[str],   # "observation" | "action" | "thought" | ...
        "tool_calls": NotRequired[list[dict]],
        "tags": NotRequired[list[str]],
        "is_demo": NotRequired[bool],
    })

The processors never raise on malformed items -- they preserve
unrecognised entries unchanged. This is critical because ultron's
history items don't always carry ``message_type`` (the legacy
:class:`LLMEngine._build_messages` path emits bare role/content
dicts); the processors treat such items as "user" messages and
the file-pattern regex either matches them or doesn't.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases (PEP-484 only; no runtime overhead)
# ---------------------------------------------------------------------------

HistoryItem = MutableMapping[str, Any]
History = list[HistoryItem]


# ---------------------------------------------------------------------------
# Regex patterns (verbatim from SWE-Agent where the pattern is load-bearing)
# ---------------------------------------------------------------------------

#: Matches SWE-Agent's file-view header
#: ``[File: /path/to/file.py (123 lines total)]``. Captures the file
#: identity so multiple snapshots of the same file collapse.
_FILE_PATTERN = re.compile(r"\[File:\s+(.*?)\s+\((\d+)\s+lines total\)\]")

#: Matches the ``<line_number>:<content>`` line blocks SWE-Agent's
#: file viewer emits. Used to detect AND find the start/end of the
#: block to replace with the summary text.
_LINE_BLOCK_PATTERN = re.compile(r"^(\d+):.*?(?:\n|$)", re.MULTILINE)

#: Default summary template -- verbatim from SWE-Agent's
#: ``ClosedWindowHistoryProcessor``. ``{n_lines}`` is the line count
#: of the elided block.
DEFAULT_CLOSED_WINDOW_TEMPLATE: str = (
    "Outdated window with {n_lines} lines omitted...\n"
)

#: Default elision template for :class:`LastNObservations`. Mirrors
#: SWE-Agent's ``Old environment output: (M lines omitted)``. When
#: image content was elided too the ``image_suffix`` is appended.
DEFAULT_OBSERVATION_ELISION_TEMPLATE: str = (
    "Old environment output: ({n_lines} lines omitted)"
)

#: Suffix appended when an elided observation also dropped images.
DEFAULT_IMAGE_ELISION_SUFFIX: str = " ({n_images} images omitted)"


# ---------------------------------------------------------------------------
# Stats records
# ---------------------------------------------------------------------------


@dataclass
class CompressionStats:
    """Diagnostic counters returned from compressor calls."""

    items_processed: int = 0
    items_compressed: int = 0
    items_skipped: int = 0
    chars_dropped: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _item_role(item: Mapping[str, Any]) -> str:
    """Return the role of ``item`` ("user" if absent)."""
    role = item.get("role")
    if isinstance(role, str):
        return role
    return "user"


def _item_is_observation(item: Mapping[str, Any]) -> bool:
    """True if ``item`` looks like a tool observation.

    Two signals are accepted, matching SWE-Agent's loose typing:

    1. ``message_type == "observation"``.
    2. ``role`` in {"tool", "function"} (legacy / function-calling shape).
    """
    mt = item.get("message_type")
    if isinstance(mt, str) and mt == "observation":
        return True
    role = _item_role(item)
    return role in ("tool", "function")


def _item_is_demo(item: Mapping[str, Any]) -> bool:
    """True if ``item`` is part of a few-shot demonstration."""
    return bool(item.get("is_demo", False))


def _item_text_content(item: Mapping[str, Any]) -> str:
    """Return the textual content of ``item`` (multi-modal segments
    flattened to their text parts only)."""
    content = item.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for seg in content:
            if not isinstance(seg, Mapping):
                continue
            stype = seg.get("type")
            if stype == "text":
                txt = seg.get("text", "")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(parts)
    return ""


def _set_item_content(item: HistoryItem, new_content: str) -> None:
    """Mutate ``item['content']`` to ``new_content`` (string form)."""
    item["content"] = new_content


def _content_image_count(item: Mapping[str, Any]) -> int:
    """Return the number of image segments in a multi-modal content list."""
    content = item.get("content")
    if not isinstance(content, list):
        return 0
    count = 0
    for seg in content:
        if isinstance(seg, Mapping) and seg.get("type") == "image_url":
            count += 1
    return count


def _item_tags(item: Mapping[str, Any]) -> set[str]:
    """Return the tag set on ``item`` (empty if absent)."""
    raw = item.get("tags")
    if isinstance(raw, (list, tuple, set, frozenset)):
        return {str(t) for t in raw}
    return set()


def _add_tag(item: HistoryItem, tag: str) -> None:
    """Append ``tag`` to ``item['tags']`` (creating the list if absent)."""
    existing = item.get("tags")
    if isinstance(existing, list):
        if tag not in existing:
            existing.append(tag)
    elif isinstance(existing, (set, frozenset)):
        item["tags"] = sorted({*existing, tag})
    else:
        item["tags"] = [tag]


# ---------------------------------------------------------------------------
# T2: ClosedWindowHistoryProcessor
# ---------------------------------------------------------------------------


@dataclass
class ClosedWindowHistoryProcessor:
    """Collapse repeated file-view snapshots into a one-line summary.

    Walks ``history`` in REVERSE chronological order. The first time a
    file path appears it is preserved verbatim. Any earlier appearance
    of the same file gets its line-block region replaced with
    :data:`DEFAULT_CLOSED_WINDOW_TEMPLATE` (or ``template``). Content
    OUTSIDE the line-block region (the model's surrounding chatter) is
    preserved.

    The processor never modifies in place: it returns a new list of
    new dicts. Items that don't carry a file-view header (system
    prompts, plain user turns, action turns) pass through unchanged.
    Demo items are also passed through (the ``is_demo`` flag flags
    few-shot examples whose canonical shape must survive).
    """

    template: str = DEFAULT_CLOSED_WINDOW_TEMPLATE
    file_pattern: re.Pattern[str] = _FILE_PATTERN
    line_block_pattern: re.Pattern[str] = _LINE_BLOCK_PATTERN
    enabled: bool = True

    def __call__(self, history: Iterable[Mapping[str, Any]]) -> History:
        items = [self._copy_item(h) for h in history]
        if not self.enabled:
            return items
        seen_files: set[str] = set()
        compressed: list[HistoryItem] = []
        for item in reversed(items):
            if not self._eligible(item):
                compressed.append(item)
                continue
            content = _item_text_content(item)
            file_match = self.file_pattern.search(content)
            if file_match is None:
                compressed.append(item)
                continue
            line_matches = list(self.line_block_pattern.finditer(content))
            if not line_matches:
                compressed.append(item)
                continue
            file_id = file_match.group(1)
            if file_id in seen_files:
                start = line_matches[0].start()
                end = line_matches[-1].end()
                n_lines = len(line_matches)
                replacement = self.template.format(n_lines=n_lines)
                new_content = content[:start] + replacement + content[end:]
                _set_item_content(item, new_content)
            seen_files.add(file_id)
            compressed.append(item)
        return list(reversed(compressed))

    @staticmethod
    def _copy_item(item: Mapping[str, Any]) -> HistoryItem:
        """Shallow-copy ``item`` so the processor never mutates input."""
        result: HistoryItem = dict(item)
        return result

    @staticmethod
    def _eligible(item: Mapping[str, Any]) -> bool:
        """True if ``item`` is a candidate for file-view collapse.

        We only collapse user / tool / function messages -- system
        messages and assistant turns are left untouched. Demo items
        are excluded so few-shot prompts survive intact.
        """
        if _item_is_demo(item):
            return False
        role = _item_role(item)
        # Files typically show up in the user message that holds the
        # tool observation OR directly in a tool/function message.
        return role in ("user", "tool", "function")


# ---------------------------------------------------------------------------
# T9: LastNObservations + TagToolCallObservations
# ---------------------------------------------------------------------------


_DEFAULT_REMOVE_TAGS: frozenset[str] = frozenset({"remove_output"})
_DEFAULT_KEEP_TAGS: frozenset[str] = frozenset({"keep_output"})


@dataclass
class LastNObservations:
    """Elide all but the last N observations from the LLM history.

    Algorithm (verbatim from SWE-Agent, slightly adapted to ultron's
    history shape):

    1. Collect the indices of every observation in ``history`` that
       isn't a demo.
    2. Compute ``last_removed_idx = max(0, (len(obs_idx) // polling)
       * polling - n)``. This is the polling trick: with ``n=5`` and
       ``polling=5`` the keep-window stays the same for 5 turns at
       a time, so Anthropic's prompt cache stays warm. ``polling=1``
       collapses to the naive sliding window.
    3. NEVER elide the first observation (typically the instance
       template / problem statement -- losing it confuses every
       downstream model).
    4. Observations at indices in ``[1:last_removed_idx]`` are
       replaced with :data:`DEFAULT_OBSERVATION_ELISION_TEMPLATE`
       unless they carry a tag in ``always_keep_output_for_tags``;
       observations with a tag in ``always_remove_output_for_tags``
       are elided regardless of position.
    """

    n: int = 5
    polling: int = 1
    always_remove_output_for_tags: frozenset[str] = _DEFAULT_REMOVE_TAGS
    always_keep_output_for_tags: frozenset[str] = _DEFAULT_KEEP_TAGS
    elision_template: str = DEFAULT_OBSERVATION_ELISION_TEMPLATE
    image_suffix_template: str = DEFAULT_IMAGE_ELISION_SUFFIX
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError(f"n must be > 0 (got {self.n})")
        if self.polling <= 0:
            raise ValueError(f"polling must be > 0 (got {self.polling})")
        # Normalise tag containers so callers can pass list/tuple/set.
        if not isinstance(self.always_remove_output_for_tags, frozenset):
            self.always_remove_output_for_tags = frozenset(
                self.always_remove_output_for_tags
            )
        if not isinstance(self.always_keep_output_for_tags, frozenset):
            self.always_keep_output_for_tags = frozenset(
                self.always_keep_output_for_tags
            )

    def _compute_omit_indices(self, history: Sequence[Mapping[str, Any]]) -> list[int]:
        """Return the indices in ``history`` whose observations get elided."""
        obs_indices = [
            idx
            for idx, item in enumerate(history)
            if _item_is_observation(item) and not _item_is_demo(item)
        ]
        if not obs_indices:
            return []
        # Polling: round-down to the nearest multiple of polling, then
        # subtract n to find the cut-off.
        last_removed_idx = max(
            0, (len(obs_indices) // self.polling) * self.polling - self.n
        )
        # Skip the first observation always (instance template).
        return obs_indices[1:last_removed_idx]

    def __call__(self, history: Iterable[Mapping[str, Any]]) -> History:
        items = [dict(h) for h in history]
        if not self.enabled:
            return items
        omit_indices = set(self._compute_omit_indices(items))
        out: History = []
        for idx, item in enumerate(items):
            tags = _item_tags(item)
            should_remove = bool(tags & self.always_remove_output_for_tags)
            should_keep = bool(tags & self.always_keep_output_for_tags)
            if should_remove or (idx in omit_indices and not should_keep):
                if not _item_is_observation(item):
                    # Sanity guard: if a non-observation accidentally
                    # landed in the omit set, leave it alone.
                    out.append(item)
                    continue
                text = _item_text_content(item)
                n_lines = len(text.splitlines()) if text else 0
                replacement = self.elision_template.format(n_lines=n_lines)
                image_count = _content_image_count(item)
                if image_count:
                    replacement += self.image_suffix_template.format(
                        n_images=image_count
                    )
                _set_item_content(item, replacement)
                _add_tag(item, "elided")
            out.append(item)
        return out


@dataclass
class TagToolCallObservations:
    """Tag observations produced by specific tool names.

    Walks the history once. For each ASSISTANT action that invoked a
    tool whose name appears in ``function_names``, finds the NEXT
    observation in chronological order and adds every tag in
    ``tags`` to it. Useful with :class:`LastNObservations` to
    selectively keep / elide observations by source tool.

    Tool-name matching is exact. Multiple tool calls in one assistant
    turn each apply to the same next observation (one observation
    gets all matching tags).
    """

    tags: frozenset[str] = frozenset({"keep_output"})
    function_names: frozenset[str] = frozenset()
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.tags, frozenset):
            self.tags = frozenset(self.tags)
        if not isinstance(self.function_names, frozenset):
            self.function_names = frozenset(self.function_names)

    def __call__(self, history: Iterable[Mapping[str, Any]]) -> History:
        items = [dict(h) for h in history]
        if not self.enabled or not self.function_names or not self.tags:
            return items
        pending_tags: set[str] = set()
        for item in items:
            if self._action_matches(item):
                pending_tags |= self.tags
                continue
            if pending_tags and _item_is_observation(item):
                for tag in sorted(pending_tags):
                    _add_tag(item, tag)
                pending_tags.clear()
        return items

    def _action_matches(self, item: Mapping[str, Any]) -> bool:
        """True if ``item`` is an assistant action whose tool_calls
        include any name in :attr:`function_names`."""
        role = _item_role(item)
        if role != "assistant":
            return False
        calls = item.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            return False
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            fn = call.get("function")
            if not isinstance(fn, Mapping):
                # Some shapes use `{"name": "..."}` directly.
                name = call.get("name")
            else:
                name = fn.get("name")
            if isinstance(name, str) and name in self.function_names:
                return True
        return False


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------


HistoryProcessor = Callable[[Iterable[Mapping[str, Any]]], History]


def apply_history_processors(
    history: Iterable[Mapping[str, Any]],
    processors: Sequence[HistoryProcessor],
) -> History:
    """Run each processor in order against ``history``.

    Each processor receives the OUTPUT of the previous processor.
    Returns the final list. Empty processor list short-circuits to
    ``list(history)``.

    The composer catches per-processor exceptions and logs WARN
    rather than re-raising -- one broken processor must never break
    the whole history-forwarding path. The failing processor's
    output is skipped (the previous step's output flows forward).
    """
    state: History = [dict(h) for h in history]
    for proc in processors:
        try:
            state = proc(state)
        except Exception:
            logger.warning(
                "history processor %s raised; skipping",
                getattr(proc, "__class__", type(proc)).__name__,
                exc_info=True,
            )
    return state


def build_default_processors(
    *,
    closed_window_enabled: bool = True,
    last_n: Optional[int] = None,
    polling: int = 1,
    keep_for_tools: Optional[Sequence[str]] = None,
    remove_for_tools: Optional[Sequence[str]] = None,
) -> list[HistoryProcessor]:
    """Build the canonical processor chain used by ultron's coding +
    architect-supervisor paths.

    Default chain:

    1. :class:`TagToolCallObservations` -- tags observations from the
       configured tools.
    2. :class:`ClosedWindowHistoryProcessor` -- collapses redundant
       file snapshots.
    3. :class:`LastNObservations` -- elides old observations, with
       ``polling`` for cache stability.

    Order matters: tagging must run BEFORE last-N so the keep/remove
    overrides apply. Closed-window runs between them so the latest
    snapshot of a file always survives last-N elision (an elided
    observation has no file-block to compress).
    """
    chain: list[HistoryProcessor] = []
    if keep_for_tools or remove_for_tools:
        tags_to_keep = frozenset(keep_for_tools or ())
        tags_to_remove = frozenset(remove_for_tools or ())
        if tags_to_keep:
            chain.append(
                TagToolCallObservations(
                    tags=frozenset({"keep_output"}),
                    function_names=tags_to_keep,
                )
            )
        if tags_to_remove:
            chain.append(
                TagToolCallObservations(
                    tags=frozenset({"remove_output"}),
                    function_names=tags_to_remove,
                )
            )
    if closed_window_enabled:
        chain.append(ClosedWindowHistoryProcessor())
    if last_n is not None and last_n > 0:
        chain.append(LastNObservations(n=int(last_n), polling=int(polling)))
    return chain


__all__ = [
    "ClosedWindowHistoryProcessor",
    "CompressionStats",
    "DEFAULT_CLOSED_WINDOW_TEMPLATE",
    "DEFAULT_IMAGE_ELISION_SUFFIX",
    "DEFAULT_OBSERVATION_ELISION_TEMPLATE",
    "History",
    "HistoryItem",
    "HistoryProcessor",
    "LastNObservations",
    "TagToolCallObservations",
    "apply_history_processors",
    "build_default_processors",
]
