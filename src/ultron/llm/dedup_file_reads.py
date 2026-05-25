"""In-place dedup of duplicate file-read tool results in API history.

Adapted from cline's ``attemptFileReadOptimizationCore`` pattern
(Apache 2.0; see ``THIRD_PARTY_NOTICES.md``). The dedup pass walks a
recent slice of API conversation history, groups tool-result blocks
by ``(tool_name, file_path)``, and replaces every duplicate except the
LATEST occurrence with a short ``[Note]`` elision marker. The latest
read is kept so the model always sees current content.

The implementation is intentionally read-only — it returns a NEW
history list plus a per-step token-savings estimate. Callers decide
whether the savings warrant suppressing a separate (more expensive)
compaction pass. The catalog suggests skipping compaction when dedup
saves >= 30 %.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from ultron.llm.response_format import (
    duplicate_file_read_notice,
    duplicate_payload_notice,
)

LOGGER = logging.getLogger(__name__)

#: Estimated characters-per-token ratio for the savings heuristic.
#: 4 is the conventional cheap default (matches char_count_tokens in
#: ``utils/token_budget.py``).
DEFAULT_CHARS_PER_TOKEN: int = 4

#: Default minimum savings ratio that should suppress full compaction
#: (cline cites ~30 %; ultron keeps the same threshold).
DEFAULT_SAVINGS_SUPPRESS_THRESHOLD: float = 0.30


@dataclass(frozen=True)
class DedupResult:
    """Outcome of a single dedup pass.

    Attributes:
        history: dedup'd history; same length as input, with duplicate
            tool-result bodies replaced by short elision notices.
        bytes_saved: estimated bytes saved by the rewrite (sum of
            elided body lengths minus notice lengths).
        tokens_saved_estimate: ``bytes_saved // chars_per_token``;
            useful for the compaction-skip heuristic.
        bytes_before: total payload size before dedup (used by the
            savings ratio).
        savings_ratio: ``bytes_saved / max(bytes_before, 1)``.
        notes: human-readable list of which entries were elided
            (file path + timestamp pairs); useful for audit logs.
        rewritten_indices: indices in the input history that were
            rewritten by the pass.
    """

    history: list[Mapping[str, Any]]
    bytes_saved: int
    tokens_saved_estimate: int
    bytes_before: int
    savings_ratio: float
    notes: tuple[str, ...] = field(default_factory=tuple)
    rewritten_indices: tuple[int, ...] = field(default_factory=tuple)


def _stringify_block(block: Any) -> str:
    """Render an arbitrary content block to a UTF-8 string for sizing."""
    if isinstance(block, str):
        return block
    if isinstance(block, Mapping):
        # Concatenate the common cline/anthropic shape: ``{"type": ..., "text": ...}``.
        text = block.get("text")
        if isinstance(text, str):
            return text
    return ""


def _payload_size(value: Any) -> int:
    """Best-effort character-count of a message ``content`` payload.

    Handles the common typed-block shapes:
    * ``str`` → length directly.
    * ``Sequence`` → sum of element sizes (recursive).
    * ``Mapping`` with ``text`` field → text length.
    * ``Mapping`` with ``content`` field (Anthropic tool_result shape) →
      recurse into the content (may be string OR sub-list).
    """
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, Mapping):
        text = value.get("text")
        if isinstance(text, str):
            return len(text)
        content = value.get("content")
        if content is not None:
            return _payload_size(content)
        return 0
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return sum(_payload_size(item) for item in value)
    return 0


def _extract_file_paths(parameters: Mapping[str, Any]) -> tuple[str, ...]:
    """Pull file-path-like parameters from a tool-call parameters dict."""
    paths: list[str] = []
    for key in ("path", "file", "file_path", "filepath", "filename"):
        value = parameters.get(key)
        if isinstance(value, str) and value:
            paths.append(value)
    for key in ("paths", "files", "file_paths"):
        value = parameters.get(key)
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            for entry in value:
                if isinstance(entry, str) and entry:
                    paths.append(entry)
    return tuple(paths)


def _timestamp_label(message: Mapping[str, Any], fallback_index: int) -> str:
    """Render a timestamp label for an elision notice."""
    for key in ("ts", "timestamp", "time"):
        value = message.get(key)
        if value:
            return str(value)
    return f"turn-{fallback_index}"


def _is_tool_use_block(block: Any) -> bool:
    """Heuristic: is this an Anthropic-style tool_use content block?"""
    if isinstance(block, Mapping):
        return block.get("type") == "tool_use"
    return False


def _is_tool_result_block(block: Any) -> bool:
    """Heuristic: is this an Anthropic-style tool_result content block?"""
    if isinstance(block, Mapping):
        return block.get("type") == "tool_result"
    return False


def dedup_duplicate_file_reads(
    history: Sequence[Mapping[str, Any]],
    *,
    read_tool_names: Iterable[str] = ("read_file",),
    chars_per_token: int = DEFAULT_CHARS_PER_TOKEN,
    keep_latest: bool = True,
) -> DedupResult:
    """Replace duplicate file-read tool results with elision notices.

    The walk pairs each ``tool_use`` block with the IMMEDIATELY-FOLLOWING
    ``tool_result`` block in the message after it (the cline shape). For
    every ``(tool_name, file_path)`` pair seen more than once, every
    occurrence EXCEPT the latest (when ``keep_latest=True``) is rewritten
    in place with a short bracket notice.

    Args:
        history: API conversation history (list of ``{role, content}``
            mappings). Content may be a string OR a list of typed
            content blocks (Anthropic/OpenAI shape).
        read_tool_names: tool names whose results should be deduped.
            Default ``("read_file",)``; pass extras like
            ``("read_file", "fetch_url")`` to broaden coverage.
        chars_per_token: char-count-to-token approximation for the
            tokens-saved estimate.
        keep_latest: when True (default), the LATEST read of each path
            is preserved verbatim; older reads are elided. When False,
            the FIRST read is preserved and subsequent ones elided
            (useful when the underlying file is known immutable).

    Returns:
        :class:`DedupResult` with the rewritten history and savings
        bookkeeping. The original history is not mutated.
    """
    read_set = frozenset(read_tool_names)
    rewritten: list[Mapping[str, Any]] = [dict(m) for m in history]
    bytes_before = sum(_payload_size(m.get("content")) for m in rewritten)

    # Build a list of (history_index, block_index, path, timestamp_label,
    # body_string) for each tool_result that resolves a read tool.
    candidates: list[tuple[int, int, str, str, str, str]] = []
    for hist_idx, message in enumerate(rewritten):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        # Walk the message: pair each tool_use(name in read_set, params)
        # with the next tool_result block whose tool_use_id matches.
        tool_use_paths: dict[str, tuple[str, tuple[str, ...]]] = {}
        for block in content:
            if _is_tool_use_block(block):
                name = block.get("name", "")
                if name in read_set:
                    params = block.get("input") or block.get("parameters") or {}
                    if isinstance(params, Mapping):
                        for path in _extract_file_paths(params):
                            tool_use_paths[block.get("id", "")] = (name, (path,))
        # Resolve tool_result blocks in this same message or the next.
        all_messages_to_scan: Iterable[tuple[int, Mapping[str, Any]]] = (
            (hist_idx, rewritten[hist_idx]),
            (hist_idx + 1, rewritten[hist_idx + 1])
            if hist_idx + 1 < len(rewritten)
            else None,
        )
        for entry in all_messages_to_scan:
            if entry is None:
                continue
            target_idx, target_msg = entry
            target_content = target_msg.get("content")
            if not isinstance(target_content, list):
                continue
            for block_idx, block in enumerate(target_content):
                if not _is_tool_result_block(block):
                    continue
                tool_use_id = block.get("tool_use_id", "")
                pair = tool_use_paths.get(tool_use_id)
                if pair is None:
                    continue
                name, paths = pair
                if not paths:
                    continue
                body_string = _stringify_block(
                    block.get("content"),
                ) or "\n".join(
                    _stringify_block(b) for b in (block.get("content") or [])
                    if isinstance(b, (str, Mapping))
                )
                ts_label = _timestamp_label(target_msg, target_idx)
                candidates.append(
                    (
                        target_idx,
                        block_idx,
                        name,
                        paths[0],
                        ts_label,
                        body_string,
                    )
                )

    # Group candidates by (name, path) and decide which to elide.
    groups: dict[tuple[str, str], list[int]] = {}
    for cand_idx, (_target_idx, _block_idx, name, path, _ts, _body) in enumerate(
        candidates,
    ):
        groups.setdefault((name, path), []).append(cand_idx)

    to_elide: dict[int, tuple[str, str]] = {}
    for key, idxs in groups.items():
        if len(idxs) < 2:
            continue
        if keep_latest:
            keep = idxs[-1]
        else:
            keep = idxs[0]
        for idx in idxs:
            if idx == keep:
                continue
            _path = key[1]
            ts_label = candidates[idx][4]
            to_elide[idx] = (_path, ts_label)

    if not to_elide:
        return DedupResult(
            history=rewritten,
            bytes_saved=0,
            tokens_saved_estimate=0,
            bytes_before=bytes_before,
            savings_ratio=0.0,
        )

    bytes_saved = 0
    notes: list[str] = []
    rewritten_indices: set[int] = set()
    # Apply the elisions in reverse so block_idx values stay valid.
    for cand_idx in sorted(to_elide.keys(), reverse=True):
        target_idx, block_idx, _name, path, ts_label, body = candidates[cand_idx]
        notice = duplicate_file_read_notice(path, ts_label)
        original_len = len(body)
        replaced_len = len(notice)
        bytes_saved += max(0, original_len - replaced_len)
        target_message = rewritten[target_idx]
        content_list = list(target_message.get("content") or [])
        if 0 <= block_idx < len(content_list):
            existing_block = content_list[block_idx]
            if isinstance(existing_block, Mapping):
                new_block = dict(existing_block)
                # Anthropic-shape tool_result content can be string OR
                # list[block]; we collapse to a single text element.
                new_block["content"] = notice
                content_list[block_idx] = new_block
        target_message["content"] = content_list
        notes.append(f"{path} @ {ts_label}")
        rewritten_indices.add(target_idx)

    savings_ratio = (
        bytes_saved / max(bytes_before, 1) if bytes_before else 0.0
    )
    return DedupResult(
        history=rewritten,
        bytes_saved=bytes_saved,
        tokens_saved_estimate=bytes_saved // max(1, chars_per_token),
        bytes_before=bytes_before,
        savings_ratio=savings_ratio,
        notes=tuple(notes),
        rewritten_indices=tuple(sorted(rewritten_indices)),
    )


def should_skip_compaction(
    result: DedupResult,
    *,
    threshold: float = DEFAULT_SAVINGS_SUPPRESS_THRESHOLD,
) -> bool:
    """True when the dedup savings are large enough to skip compaction.

    Args:
        result: outcome of :func:`dedup_duplicate_file_reads`.
        threshold: minimum ``savings_ratio`` that suppresses compaction
            (default 0.30, matching the cline heuristic).
    """
    return result.savings_ratio >= threshold


def dedup_payload_duplicates(
    payloads: Sequence[tuple[str, str, str]],
    *,
    keep_latest: bool = True,
) -> list[tuple[str, str]]:
    """Generic dedup for non-file-read payload streams.

    Useful for collapsing repeated nvidia-smi outputs, repeated
    web-search snippet bodies, etc. The caller provides a sequence of
    ``(label, timestamp, body)`` tuples and receives the deduped
    output as a sequence of ``(label_or_notice, body_or_notice)``
    suitable for joining back into a prompt.

    Args:
        payloads: items to dedup.
        keep_latest: keep the latest occurrence of each label
            (default) or the first.

    Returns:
        List of ``(label, body)`` tuples where duplicates have been
        rewritten to ``(label, notice)`` pairs.
    """
    groups: dict[str, list[int]] = {}
    for idx, (label, _ts, _body) in enumerate(payloads):
        groups.setdefault(label, []).append(idx)
    to_elide: set[int] = set()
    for label, idxs in groups.items():
        if len(idxs) < 2:
            continue
        keep = idxs[-1] if keep_latest else idxs[0]
        for idx in idxs:
            if idx != keep:
                to_elide.add(idx)
    out: list[tuple[str, str]] = []
    for idx, (label, ts, body) in enumerate(payloads):
        if idx in to_elide:
            out.append((label, duplicate_payload_notice(label, ts)))
        else:
            out.append((label, body))
    return out


__all__ = [
    "DEFAULT_CHARS_PER_TOKEN",
    "DEFAULT_SAVINGS_SUPPRESS_THRESHOLD",
    "DedupResult",
    "dedup_duplicate_file_reads",
    "dedup_payload_duplicates",
    "should_skip_compaction",
]
