"""Pre/post lint diff with line-shift arithmetic + revert template.

Direct port of the load-bearing algorithm from SWE-Agent's
``tools/windowed/lib/flake8_utils.py`` (MIT, Yang et al. 2024).
The pattern: snapshot the lint output BEFORE the edit. Apply the
edit. Snapshot the lint output AFTER. Subtract: only NEW errors
caused by the edit count. The subtraction is LINE-NUMBER-AWARE --
every pre-edit error's line number is shifted forward by
``(replacement_n_lines - (replacement_window[1] - replacement_window[0] + 1))``
so an error that was on line 50 before still maps to its post-edit
line. Errors inside the replacement window are NEVER carried over
as pre-existing (the edit owns that region).

When new errors survive the subtraction, the edit gets REVERTED
(in callers that wire it up) and the model is shown:

* The would-be window with line numbers (annotated with the new errors)
* The original window with line numbers
* A "DO NOT re-run the same failed edit" hint

This module ships only the pure functions. The runner-side wiring
(snapshot pre, run edit, snapshot post, compute diff, revert via
:class:`FileHistory`, narrate via supervisor) lives in the existing
:mod:`ultron.coding.runner` integration points and an opt-in
config flag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence


# ---------------------------------------------------------------------------
# Flake8 line shape
# ---------------------------------------------------------------------------

#: Matches a flake8 error line like
#: ``path/to/file.py:42:5: F821 undefined name 'x'``.
_FLAKE8_LINE = re.compile(
    r"^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):\s*(?P<msg>.+?)\s*$"
)


@dataclass(frozen=True)
class LintError:
    """One parsed flake8 error."""

    filename: str
    line_number: int
    col_number: int
    problem: str


def parse_flake8_output(text: str) -> list[LintError]:
    """Parse flake8 stdout into a list of :class:`LintError`."""
    out: list[LintError] = []
    if not text:
        return out
    for raw in text.splitlines():
        m = _FLAKE8_LINE.match(raw.strip())
        if not m:
            continue
        try:
            out.append(
                LintError(
                    filename=m.group("file"),
                    line_number=int(m.group("line")),
                    col_number=int(m.group("col")),
                    problem=m.group("msg"),
                )
            )
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------
# Line-shift arithmetic (verbatim port of SWE-Agent's _update_previous_errors)
# ---------------------------------------------------------------------------


def shift_pre_edit_errors(
    pre_errors: Sequence[LintError],
    *,
    replacement_window: tuple[int, int],
    replacement_n_lines: int,
) -> list[LintError]:
    """Update each pre-edit error's line number to its POST-edit value.

    Errors INSIDE the replacement window are dropped (the edit owns
    that region; whatever the pre-edit lint said about it is no
    longer relevant). Errors BEFORE the window are kept unchanged.
    Errors AFTER the window are shifted by
    ``replacement_n_lines - window_height``.

    ``replacement_window`` is a 1-indexed inclusive ``(start, stop)``
    tuple matching SWE-Agent's convention.
    """
    win_start, win_stop = replacement_window
    if win_stop < win_start:
        raise ValueError(
            f"replacement_window stop must be >= start "
            f"(got start={win_start}, stop={win_stop})"
        )
    window_height = win_stop - win_start + 1
    lines_added = replacement_n_lines - window_height
    shifted: list[LintError] = []
    for err in pre_errors:
        if err.line_number < win_start:
            shifted.append(err)
        elif err.line_number <= win_stop:
            # Inside the replaced region -- drop.
            continue
        else:
            shifted.append(
                LintError(
                    filename=err.filename,
                    line_number=err.line_number + lines_added,
                    col_number=err.col_number,
                    problem=err.problem,
                )
            )
    return shifted


def compute_new_errors(
    pre_errors: Sequence[LintError],
    post_errors: Sequence[LintError],
    *,
    replacement_window: tuple[int, int],
    replacement_n_lines: int,
) -> list[LintError]:
    """Return the errors introduced by an edit.

    Algorithm:

    1. Shift ``pre_errors`` so each carries its POST-edit line number
       (via :func:`shift_pre_edit_errors`).
    2. Drop any post-edit error that exactly matches a shifted
       pre-edit error (same file + line + col + problem) -- those
       are PRE-existing and not the edit's fault.
    3. Per SWE-Agent's additional filter, drop any post-edit error
       whose line number is BEFORE the replacement window (the
       edit's effect is bounded by the window; earlier-line errors
       must be pre-existing the line-shift missed).
    """
    shifted = shift_pre_edit_errors(
        pre_errors,
        replacement_window=replacement_window,
        replacement_n_lines=replacement_n_lines,
    )
    pre_keys = {
        (e.filename, e.line_number, e.col_number, e.problem) for e in shifted
    }
    win_start, _ = replacement_window
    new_errors: list[LintError] = []
    for err in post_errors:
        key = (err.filename, err.line_number, err.col_number, err.problem)
        if key in pre_keys:
            continue
        if err.line_number < win_start:
            # Lower-line errors that the shift didn't catch -- treat
            # as pre-existing per SWE-Agent's filter.
            continue
        new_errors.append(err)
    return new_errors


# ---------------------------------------------------------------------------
# Revert message template (twin window with errors annotated)
# ---------------------------------------------------------------------------


_REVERT_TEMPLATE = (
    "Your proposed edit has introduced new syntax error(s). Please read this "
    "error message carefully and then retry editing the file.\n\n"
    "ERRORS:\n"
    "\n"
    "{errors}\n"
    "\n"
    "This is how your edit would have looked if applied\n"
    "------------------------------------------------\n"
    "{window_applied}\n"
    "------------------------------------------------\n"
    "\n"
    "This is the original code before your edit\n"
    "------------------------------------------------\n"
    "{window_original}\n"
    "------------------------------------------------\n"
    "\n"
    "Your changes have NOT been applied. Please fix your edit command and "
    "try again.\n"
    "DO NOT re-run the same failed edit command. Running it again will lead "
    "to the same error."
)


def render_window_with_line_numbers(
    lines: Sequence[str],
    *,
    first_line: int,
) -> str:
    """Render a window of source as ``<n>:<content>`` per line.

    ``first_line`` is the 1-indexed line number of ``lines[0]``.
    """
    return "\n".join(f"{first_line + i}:{lines[i]}" for i in range(len(lines)))


def format_revert_message(
    *,
    errors: Sequence[LintError],
    window_applied: str,
    window_original: str,
) -> str:
    """Render the twin-window revert message ready to forward to the model.

    ``window_applied`` and ``window_original`` are pre-rendered text
    (e.g. via :func:`render_window_with_line_numbers`).
    """
    if not errors:
        error_block = "(no specific errors -- check your indentation and syntax)"
    else:
        error_block = "\n".join(_format_error(e) for e in errors)
    return _REVERT_TEMPLATE.format(
        errors=error_block,
        window_applied=window_applied,
        window_original=window_original,
    )


def _format_error(err: LintError) -> str:
    """Format one :class:`LintError` as a single bullet line."""
    return f"- line {err.line_number} col {err.col_number}: {err.problem}"


# ---------------------------------------------------------------------------
# End-to-end convenience
# ---------------------------------------------------------------------------


@dataclass
class LintDiffResult:
    """Output of :func:`evaluate_edit_lint`.

    :param ok: True if the edit introduced no new errors.
    :param new_errors: list of NEW errors the edit introduced (empty
        when ok is True).
    :param shifted_pre_errors: pre-edit errors after line-shift
        application (for diagnostic logging).
    :param message: rendered revert message ready to forward to the
        model (empty string when ok is True).
    """

    ok: bool
    new_errors: list[LintError] = field(default_factory=list)
    shifted_pre_errors: list[LintError] = field(default_factory=list)
    message: str = ""


def evaluate_edit_lint(
    *,
    pre_lint_output: str,
    post_lint_output: str,
    replacement_window: tuple[int, int],
    replacement_n_lines: int,
    window_applied: Optional[str] = None,
    window_original: Optional[str] = None,
) -> LintDiffResult:
    """End-to-end helper that parses both lint outputs, computes the
    new errors, and renders the revert message.

    When ``window_applied`` / ``window_original`` are passed, the
    rendered message includes the twin-window template; otherwise
    only the error list is rendered.
    """
    pre_errors = parse_flake8_output(pre_lint_output)
    post_errors = parse_flake8_output(post_lint_output)
    shifted = shift_pre_edit_errors(
        pre_errors,
        replacement_window=replacement_window,
        replacement_n_lines=replacement_n_lines,
    )
    new_errors = compute_new_errors(
        pre_errors,
        post_errors,
        replacement_window=replacement_window,
        replacement_n_lines=replacement_n_lines,
    )
    if not new_errors:
        return LintDiffResult(ok=True, shifted_pre_errors=shifted)
    if window_applied is not None and window_original is not None:
        msg = format_revert_message(
            errors=new_errors,
            window_applied=window_applied,
            window_original=window_original,
        )
    else:
        msg = "\n".join(_format_error(e) for e in new_errors)
    return LintDiffResult(
        ok=False,
        new_errors=new_errors,
        shifted_pre_errors=shifted,
        message=msg,
    )


__all__ = [
    "LintDiffResult",
    "LintError",
    "compute_new_errors",
    "evaluate_edit_lint",
    "format_revert_message",
    "parse_flake8_output",
    "render_window_with_line_numbers",
    "shift_pre_edit_errors",
]
