"""Edit-tool snapshot-and-recheck recovery wrapper.

T14 (OpenClaw catalog port; see ``THIRD_PARTY_NOTICES.md``). Wraps a
SEARCH/REPLACE-style edit tool with a recovery layer:

1. Snapshot the file BEFORE invoking the edit.
2. Run the edit.
3. On error, RE-READ the file and check whether the edit actually
   landed (via the :func:`did_edit_likely_apply` heuristic).
4. If the heuristic says yes, the error is spurious (post-write
   validation race, filesystem hiccup) — recover to success.
5. If the heuristic says no AND the error looks like a search-mismatch,
   append the current file snippet so the next LLM attempt sees what
   actually lives in the file (rather than just "Could not find the
   exact text").

The :func:`did_edit_likely_apply` heuristic is conservative: it only
returns ``True`` when ALL ``new_text`` substrings appear in the
current file AND ALL ``old_text`` substrings are absent (after
removing each ``new_text`` from a working copy to avoid double-counting
overlap). LF normalisation handles CRLF differences. Edits with empty
input or unchanged file content are explicitly rejected.

Use cases:

* AI coding agent edit retries — when the AI coding agent's
  ``str_replace_editor`` raises a spurious post-write validation
  error but the file actually changed correctly, recover to success
  instead of forcing the model to retry the entire write.
* SEARCH/REPLACE whitespace mismatches — the recovery layer cannot
  fix these directly but it CAN enrich the error message with the
  current file body so the next attempt has the info it needs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

LOGGER = logging.getLogger(__name__)

#: Maximum file-body snippet appended to a mismatch error before truncation.
DEFAULT_MISMATCH_SNIPPET_CHARS: int = 800

#: Search-mismatch error pattern. When the underlying edit tool surfaces
#: this exact substring (or a localised variant), the recovery layer
#: appends the current file body to the error. Mirrors OpenClaw's
#: ``Could not find the exact text in`` recognition.
MISMATCH_ERROR_MARKERS: tuple[str, ...] = (
    "Could not find the exact text",
    "no exact match",
    "search text not found",
)


@dataclass(frozen=True)
class EditSpec:
    """One ``(old_text, new_text)`` replacement spec.

    ``old_text`` is matched verbatim against the file body. ``new_text``
    is inserted in its place. The recovery heuristic checks that every
    ``new_text`` appears in the post-edit file AND every ``old_text``
    is absent.

    Attributes:
        old_text: text the edit tool is asked to find + replace.
        new_text: replacement.
        path: file the edit targets (used by the recovery layer to
            read post-edit content).
    """

    old_text: str
    new_text: str
    path: str = ""


def _normalise_lf(text: str) -> str:
    """Convert CRLF to LF for boundary-agnostic substring search."""
    if not text:
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def did_edit_likely_apply(
    *,
    original_content: str,
    current_content: str,
    edits: Sequence[EditSpec],
) -> bool:
    """Heuristic: did the edit batch land successfully?

    Both conditions must hold:

    1. Every ``edit.new_text`` (non-empty after LF normalisation)
       appears in ``current_content``.
    2. Every ``edit.old_text`` (non-empty) is ABSENT from a working
       copy of ``current_content`` where each ``new_text`` has been
       removed (prevents a ``new_text`` that contains the ``old_text``
       from causing a false negative).

    Plus a sanity guard: if ``original_content == current_content``,
    the file didn't change at all — heuristic returns ``False`` even
    if both substring conditions accidentally pass.

    Args:
        original_content: file body before the edit attempt.
        current_content: file body after the edit raised.
        edits: list of edit specs that were attempted.

    Returns:
        ``True`` only when the heuristic is confident the edit landed.
    """
    if not edits:
        return False
    original = _normalise_lf(original_content)
    current = _normalise_lf(current_content)
    if not current:
        return False
    if original == current:
        return False
    # Condition 1: every non-empty new_text appears in current.
    for edit in edits:
        new = _normalise_lf(edit.new_text)
        if not new:
            continue
        if new not in current:
            return False
    # Condition 2: build a working copy by removing each new_text once,
    # then check that no old_text remains. Removing absorbs the case
    # where a new_text legitimately contains the old text fragment.
    working = current
    for edit in edits:
        new = _normalise_lf(edit.new_text)
        if new and new in working:
            working = working.replace(new, "", 1)
    for edit in edits:
        old = _normalise_lf(edit.old_text)
        if not old:
            continue
        if old in working:
            return False
    return True


def is_search_mismatch_error(error: BaseException) -> bool:
    """``True`` when ``error`` looks like a search-mismatch raise."""
    text = str(error).lower()
    return any(marker.lower() in text for marker in MISMATCH_ERROR_MARKERS)


def enrich_mismatch_error(
    error: BaseException,
    *,
    current_content: str,
    max_chars: int = DEFAULT_MISMATCH_SNIPPET_CHARS,
) -> str:
    """Format an enriched error message including ``current_content`` snippet.

    Returns the formatted message string; the wrapper raises a new
    error with this as the payload OR attaches it as a chained note.
    """
    snippet = current_content[:max_chars]
    truncated_note = " (truncated)" if len(current_content) > max_chars else ""
    return (
        f"{error}\n\n"
        f"Current file contents{truncated_note}:\n{snippet}"
    )


@dataclass
class EditRecoveryResult:
    """Outcome of one wrapped-edit invocation."""

    succeeded: bool
    recovered: bool = False
    enriched_error: Optional[str] = None
    raw_error: Optional[BaseException] = None
    tool_result: Any = None
    original_content: Optional[str] = None
    current_content: Optional[str] = None


ReadFileFn = Callable[[str], str]
EditToolFn = Callable[[Sequence[EditSpec]], Any]


def _safe_read(path: str, reader: ReadFileFn) -> Optional[str]:
    """Best-effort read; returns ``None`` on any error."""
    if not path:
        return None
    try:
        return reader(path)
    except Exception:  # noqa: BLE001 -- best-effort snapshot
        return None


def run_edit_with_recovery(
    edits: Sequence[EditSpec],
    *,
    edit_tool: EditToolFn,
    read_file: ReadFileFn,
    enrich_mismatch: bool = True,
    mismatch_snippet_chars: int = DEFAULT_MISMATCH_SNIPPET_CHARS,
) -> EditRecoveryResult:
    """Run ``edit_tool(edits)``; recover from spurious errors.

    Args:
        edits: edit batch to apply (all targeting the same path).
        edit_tool: callable that actually performs the edit; returns
            whatever the underlying tool returns; raises on failure.
        read_file: reader callable used to snapshot before + after.
        enrich_mismatch: when ``True``, search-mismatch errors get a
            ``Current file contents:`` snippet appended.
        mismatch_snippet_chars: cap on the appended snippet.

    Returns:
        :class:`EditRecoveryResult` with the outcome.
    """
    target_path = edits[0].path if edits else ""
    original = _safe_read(target_path, read_file)
    try:
        tool_result = edit_tool(edits)
        return EditRecoveryResult(
            succeeded=True,
            recovered=False,
            tool_result=tool_result,
            original_content=original,
        )
    except BaseException as raw_error:  # noqa: BLE001 -- we will re-raise / wrap
        current = _safe_read(target_path, read_file)
        if (
            original is not None
            and current is not None
            and did_edit_likely_apply(
                original_content=original,
                current_content=current,
                edits=edits,
            )
        ):
            LOGGER.info(
                "edit reported failure but file changed in the expected "
                "way; recovering to success (path=%s)",
                target_path,
            )
            return EditRecoveryResult(
                succeeded=True,
                recovered=True,
                raw_error=raw_error,
                original_content=original,
                current_content=current,
            )
        enriched: Optional[str] = None
        if enrich_mismatch and current is not None and is_search_mismatch_error(raw_error):
            enriched = enrich_mismatch_error(
                raw_error,
                current_content=current,
                max_chars=mismatch_snippet_chars,
            )
        return EditRecoveryResult(
            succeeded=False,
            recovered=False,
            raw_error=raw_error,
            enriched_error=enriched,
            original_content=original,
            current_content=current,
        )


def wrap_edit_tool_with_recovery(
    edit_tool: EditToolFn,
    *,
    read_file: ReadFileFn,
    enrich_mismatch: bool = True,
    mismatch_snippet_chars: int = DEFAULT_MISMATCH_SNIPPET_CHARS,
) -> Callable[[Sequence[EditSpec]], EditRecoveryResult]:
    """Curry :func:`run_edit_with_recovery` against a fixed tool + reader.

    Convenience for call sites that perform many edits against the
    same tool / reader pair.
    """
    def _wrapped(edits: Sequence[EditSpec]) -> EditRecoveryResult:
        return run_edit_with_recovery(
            edits,
            edit_tool=edit_tool,
            read_file=read_file,
            enrich_mismatch=enrich_mismatch,
            mismatch_snippet_chars=mismatch_snippet_chars,
        )
    return _wrapped


__all__ = [
    "DEFAULT_MISMATCH_SNIPPET_CHARS",
    "EditRecoveryResult",
    "EditSpec",
    "EditToolFn",
    "MISMATCH_ERROR_MARKERS",
    "ReadFileFn",
    "did_edit_likely_apply",
    "enrich_mismatch_error",
    "is_search_mismatch_error",
    "run_edit_with_recovery",
    "wrap_edit_tool_with_recovery",
]
