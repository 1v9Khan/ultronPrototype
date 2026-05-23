"""Structured diagnostics for failed file edits.

Direct port of the error-template structure in SWE-Agent's
``tools/windowed_edit_replace/bin/edit`` (MIT, Yang et al. 2024).
The pattern: when a search/replace edit fails, the harness emits a
SPECIFIC error message per failure mode rather than a generic
"didn't apply." Each message names the reason and suggests the
remediation. The "found elsewhere -- here are the line numbers"
hint is the single highest-information-density signal you can give
a stuck model.

Five distinct failure modes mirroring SWE-Agent's templates:

* :data:`EditDiagnostic.NOT_FOUND` -- search string isn't in the
  file at all. Message hints at whitespace / indentation drift +
  suggests re-opening the file.
* :data:`EditDiagnostic.NOT_FOUND_IN_WINDOW` -- search string isn't
  in the currently-displayed window but DOES appear elsewhere in
  the file. Message lists EVERY line where it appears so the model
  can ``goto`` first.
* :data:`EditDiagnostic.MULTIPLE_OCCURRENCES_IN_WINDOW` -- the
  single-replace mode found the search in the window more than
  once. Message asks for a more-specific search string.
* :data:`EditDiagnostic.NO_CHANGES_MADE` -- search and replace
  strings are identical. Early exit without wasting an apply cycle.
* :data:`EditDiagnostic.AMBIGUOUS_CROSS_FILE` -- the catalog's
  "creative extension": the search isn't in the open file but DOES
  appear in another session-touched file. Hint: "Did you mean to
  edit <other_file>?"

The diagnoser is pure -- no I/O, no side effects, no LLM call. It
runs in <5 ms even on large files because Python's substring search
is C-optimised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Optional, Sequence


class EditDiagnostic(Enum):
    """Categorical edit-failure label."""

    NOT_FOUND = "not_found"
    NOT_FOUND_IN_WINDOW = "not_found_in_window"
    MULTIPLE_OCCURRENCES_IN_WINDOW = "multiple_occurrences_in_window"
    NO_CHANGES_MADE = "no_changes_made"
    AMBIGUOUS_CROSS_FILE = "ambiguous_cross_file"
    OK = "ok"


@dataclass(frozen=True)
class CrossFileHit:
    """One match of the search string in a different file.

    :param path: relative or absolute path of the OTHER file
        (whatever the caller passed in).
    :param line_numbers: 1-indexed line numbers where the search
        appears. Empty list = the file was inspected but had no
        match (shouldn't normally appear in results).
    """

    path: str
    line_numbers: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class EditDiagnosticResult:
    """Structured outcome of :func:`diagnose_edit_failure`.

    :param diagnostic: one of :class:`EditDiagnostic` values.
    :param line_numbers: 1-indexed line numbers where the search
        appears in the OPEN file (for NOT_FOUND_IN_WINDOW), or empty
        for diagnostics where the location info doesn't apply.
    :param cross_file_hits: list of :class:`CrossFileHit` records
        for AMBIGUOUS_CROSS_FILE.
    :param message: human-readable rendered message ready to
        forward to the model. Matches SWE-Agent's template shape so
        models trained on the SWE-Agent ecosystem recognise it.
    """

    diagnostic: EditDiagnostic
    line_numbers: list[int] = field(default_factory=list)
    cross_file_hits: list[CrossFileHit] = field(default_factory=list)
    message: str = ""


# ---------------------------------------------------------------------------
# Templates (verbatim shape from SWE-Agent; ultron-friendly phrasing tweaks)
# ---------------------------------------------------------------------------

_NOT_FOUND = (
    "Your edit was not applied (file not modified): Text {search!r} not "
    "found in displayed lines (or anywhere in the file).\n"
    "Please modify your search string. Did you forget to properly handle "
    "whitespace/indentation?\n"
    "You can also call `open` again to re-display the file with the correct "
    "context."
)

_NOT_FOUND_IN_WINDOW = (
    "Your edit was not applied (file not modified): Text {search!r} not "
    "found in displayed lines.\n\n"
    "However, we found the following occurrences of your search string in "
    "the file:\n\n"
    "{occurrences}\n\n"
    "You can use the `goto` command to navigate to these locations before "
    "running the edit command again."
)

_MULTIPLE_OCCURRENCES = (
    "Your edit was not applied (file not modified): Found more than one "
    "occurrence of {search!r} in the currently displayed lines.\n"
    "Please make your search string more specific (for example, by including "
    "more lines of context)."
)

_NO_CHANGES_MADE = (
    "Your search and replace strings are the same. No changes were made. "
    "Please modify your search or replace strings."
)

_AMBIGUOUS_CROSS_FILE = (
    "Your edit was not applied: Text {search!r} not found in the open file, "
    "but it does appear in other files we've touched this session:\n\n"
    "{occurrences}\n\n"
    "Did you mean to edit one of those? Use `open <path>` to switch the "
    "active file before retrying the edit."
)


def _find_all_line_numbers(text: str, search: str) -> list[int]:
    """Return 1-indexed line numbers where ``search`` starts in ``text``.

    Multi-line search strings count by the line of their FIRST
    character. Empty search returns ``[]`` (callers shouldn't pass
    empty -- the no-changes-made path catches that earlier).
    """
    if not search:
        return []
    out: list[int] = []
    start = 0
    while True:
        idx = text.find(search, start)
        if idx == -1:
            break
        # Count newlines before the index to derive the 1-indexed line.
        line = text.count("\n", 0, idx) + 1
        out.append(line)
        start = idx + 1
    return out


def _format_line_list(line_numbers: Iterable[int]) -> str:
    """Render a line-number list as ``- line N`` bullets."""
    return "\n".join(f"- line {n}" for n in line_numbers)


def _format_cross_file(hits: Iterable[CrossFileHit]) -> str:
    """Render cross-file hits as ``- <path>: lines N, M`` bullets."""
    parts = []
    for hit in hits:
        if not hit.line_numbers:
            continue
        nums = ", ".join(str(n) for n in hit.line_numbers)
        parts.append(f"- {hit.path}: lines {nums}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def diagnose_edit_failure(
    file_text: str,
    search: str,
    *,
    replace: str = "",
    in_window: Optional[str] = None,
    other_files: Optional[Mapping[str, str]] = None,
    single_replace_required: bool = True,
) -> EditDiagnosticResult:
    """Diagnose a failed search/replace edit and return a structured
    result with a rendered message.

    :param file_text: the full content of the file the edit targets.
    :param search: the search string the model proposed.
    :param replace: the replace string (used only to detect the
        no-changes-made early exit).
    :param in_window: the currently-displayed window content
        (substring of ``file_text``). ``None`` means "no window
        context" -- caller treats the whole file as the window.
    :param other_files: optional ``{path: content}`` map of OTHER
        files in the current session. When the search isn't in the
        open file but appears in one or more others, the diagnostic
        is AMBIGUOUS_CROSS_FILE and the message lists each other
        file with its line numbers.
    :param single_replace_required: when True, finding multiple
        occurrences of ``search`` in the window is treated as an
        error (mirrors SWE-Agent's default single-replace mode).
        When False, multiple matches are allowed (replace-all mode).
    """
    if replace == search and search != "":
        return EditDiagnosticResult(
            diagnostic=EditDiagnostic.NO_CHANGES_MADE,
            message=_NO_CHANGES_MADE,
        )

    file_hits = _find_all_line_numbers(file_text, search)

    if not file_hits:
        # Not in the open file. Check other session files for a
        # cross-file ambiguity hint.
        cross_hits: list[CrossFileHit] = []
        if other_files:
            for path, content in other_files.items():
                lines = _find_all_line_numbers(content, search)
                if lines:
                    cross_hits.append(CrossFileHit(path=path, line_numbers=lines))
        if cross_hits:
            return EditDiagnosticResult(
                diagnostic=EditDiagnostic.AMBIGUOUS_CROSS_FILE,
                cross_file_hits=cross_hits,
                message=_AMBIGUOUS_CROSS_FILE.format(
                    search=search,
                    occurrences=_format_cross_file(cross_hits),
                ),
            )
        return EditDiagnosticResult(
            diagnostic=EditDiagnostic.NOT_FOUND,
            message=_NOT_FOUND.format(search=search),
        )

    # The search IS in the file. Check whether it's in the window.
    if in_window is None or in_window == file_text:
        in_window_count = len(file_hits)
    else:
        in_window_count = in_window.count(search)

    if in_window_count == 0:
        # In the file but not the window: list every line in the file.
        return EditDiagnosticResult(
            diagnostic=EditDiagnostic.NOT_FOUND_IN_WINDOW,
            line_numbers=file_hits,
            message=_NOT_FOUND_IN_WINDOW.format(
                search=search,
                occurrences=_format_line_list(file_hits),
            ),
        )

    if single_replace_required and in_window_count > 1:
        return EditDiagnosticResult(
            diagnostic=EditDiagnostic.MULTIPLE_OCCURRENCES_IN_WINDOW,
            line_numbers=file_hits,
            message=_MULTIPLE_OCCURRENCES.format(search=search),
        )

    return EditDiagnosticResult(
        diagnostic=EditDiagnostic.OK,
        line_numbers=file_hits,
        message="",
    )


def find_all_in_file(file_text: str, search: str) -> list[int]:
    """Public helper: return 1-indexed line numbers where ``search``
    starts in ``file_text``.

    Exposed because the supervisor + voice path may want the raw
    line list independently of the templated message.
    """
    return _find_all_line_numbers(file_text, search)


__all__ = [
    "CrossFileHit",
    "EditDiagnostic",
    "EditDiagnosticResult",
    "diagnose_edit_failure",
    "find_all_in_file",
]
