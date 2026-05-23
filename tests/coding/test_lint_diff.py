"""Tests for the pre/post lint diff + revert template (catalog T1)."""

from __future__ import annotations

import pytest

from ultron.coding.lint_diff import (
    LintDiffResult,
    LintError,
    compute_new_errors,
    evaluate_edit_lint,
    format_revert_message,
    parse_flake8_output,
    render_window_with_line_numbers,
    shift_pre_edit_errors,
)


# ---------------------------------------------------------------------------
# parse_flake8_output
# ---------------------------------------------------------------------------


def test_parse_basic_line():
    text = "x.py:5:10: F821 undefined name 'foo'"
    errs = parse_flake8_output(text)
    assert len(errs) == 1
    assert errs[0] == LintError(
        filename="x.py", line_number=5, col_number=10, problem="F821 undefined name 'foo'"
    )


def test_parse_multiple_lines():
    text = "x.py:1:1: E111 indent\nx.py:5:10: F821 undefined"
    errs = parse_flake8_output(text)
    assert len(errs) == 2


def test_parse_skips_non_matching_lines():
    text = "x.py:5:10: F821 oops\n--- garbage ---\nx.py:9:1: E999 syntax"
    errs = parse_flake8_output(text)
    assert len(errs) == 2


def test_parse_empty_input():
    assert parse_flake8_output("") == []


# ---------------------------------------------------------------------------
# shift_pre_edit_errors
# ---------------------------------------------------------------------------


def test_shift_keeps_errors_before_window():
    errs = [LintError("x.py", 3, 1, "F821 a")]
    out = shift_pre_edit_errors(
        errs, replacement_window=(5, 10), replacement_n_lines=4
    )
    assert out == errs


def test_shift_drops_errors_inside_window():
    errs = [LintError("x.py", 7, 1, "F821 a")]
    out = shift_pre_edit_errors(
        errs, replacement_window=(5, 10), replacement_n_lines=4
    )
    assert out == []


def test_shift_adjusts_errors_after_window_by_lines_added():
    # Replacement window is 5..10 (6 lines). Replacement is 4 lines.
    # Lines added = 4 - 6 = -2. So an error at line 20 shifts to 18.
    errs = [LintError("x.py", 20, 5, "F821 a")]
    out = shift_pre_edit_errors(
        errs, replacement_window=(5, 10), replacement_n_lines=4
    )
    assert out[0].line_number == 18


def test_shift_adjusts_for_growing_edit():
    # Replacement window 5..6 (2 lines). Replacement 10 lines. Added 8.
    errs = [LintError("x.py", 20, 5, "F821 a")]
    out = shift_pre_edit_errors(
        errs, replacement_window=(5, 6), replacement_n_lines=10
    )
    assert out[0].line_number == 28


def test_shift_invalid_window_raises():
    with pytest.raises(ValueError):
        shift_pre_edit_errors(
            [], replacement_window=(10, 5), replacement_n_lines=5
        )


# ---------------------------------------------------------------------------
# compute_new_errors
# ---------------------------------------------------------------------------


def test_compute_new_errors_identical_returns_empty():
    pre = [LintError("x.py", 5, 1, "F821 a")]
    post = [LintError("x.py", 5, 1, "F821 a")]
    new = compute_new_errors(
        pre, post,
        replacement_window=(5, 5),
        replacement_n_lines=1,
    )
    # The pre-error is inside the window -> dropped from shifted set;
    # the post-error is INSIDE the window so it counts.
    assert len(new) == 1


def test_compute_new_errors_pre_existing_after_window_filtered():
    # Edit at lines 5..7 (3 lines), replaces with 3 lines. Pre-existing
    # error at line 15 shifts to 15 (lines_added=0). Post-edit lint
    # also reports it -- it must be filtered.
    pre = [LintError("x.py", 15, 1, "F821 oldname")]
    post = [LintError("x.py", 15, 1, "F821 oldname")]
    new = compute_new_errors(
        pre, post,
        replacement_window=(5, 7),
        replacement_n_lines=3,
    )
    assert new == []


def test_compute_new_errors_filters_lines_before_window():
    # SWE-Agent's additional filter: post-edit errors BEFORE the
    # replacement window are treated as pre-existing.
    pre: list[LintError] = []
    post = [LintError("x.py", 1, 1, "E111 bad indent")]
    new = compute_new_errors(
        pre, post,
        replacement_window=(10, 15),
        replacement_n_lines=5,
    )
    assert new == []


def test_compute_new_errors_surfaces_genuinely_new():
    pre: list[LintError] = []
    post = [LintError("x.py", 12, 5, "F821 undefined_new_name")]
    new = compute_new_errors(
        pre, post,
        replacement_window=(10, 15),
        replacement_n_lines=5,
    )
    assert len(new) == 1
    assert new[0].problem == "F821 undefined_new_name"


# ---------------------------------------------------------------------------
# render_window_with_line_numbers
# ---------------------------------------------------------------------------


def test_render_window_basic():
    lines = ["import os", "def foo():", "    return 1"]
    out = render_window_with_line_numbers(lines, first_line=10)
    assert "10:import os" in out
    assert "11:def foo():" in out
    assert "12:    return 1" in out


def test_render_window_empty():
    assert render_window_with_line_numbers([], first_line=1) == ""


# ---------------------------------------------------------------------------
# format_revert_message
# ---------------------------------------------------------------------------


def test_format_revert_message_includes_twin_windows():
    msg = format_revert_message(
        errors=[LintError("x.py", 5, 1, "F821 oops")],
        window_applied="5:def broken():\n6:    no_close_paren(",
        window_original="5:def working():\n6:    pass",
    )
    assert "would have looked if applied" in msg
    assert "original code before your edit" in msg
    assert "DO NOT re-run" in msg
    assert "def broken" in msg
    assert "def working" in msg
    assert "F821 oops" in msg


def test_format_revert_message_no_errors_uses_placeholder():
    msg = format_revert_message(
        errors=[],
        window_applied="x",
        window_original="y",
    )
    assert "(no specific errors" in msg


# ---------------------------------------------------------------------------
# evaluate_edit_lint (end-to-end)
# ---------------------------------------------------------------------------


def test_evaluate_edit_lint_no_new_errors():
    pre = "x.py:5:1: F821 a"
    post = "x.py:5:1: F821 a"  # same as pre
    r = evaluate_edit_lint(
        pre_lint_output=pre,
        post_lint_output=post,
        replacement_window=(10, 12),
        replacement_n_lines=3,
    )
    assert isinstance(r, LintDiffResult)
    assert r.ok is True
    assert r.new_errors == []


def test_evaluate_edit_lint_introduces_error():
    pre = ""
    post = "x.py:12:5: F821 undefined name 'newvar'"
    r = evaluate_edit_lint(
        pre_lint_output=pre,
        post_lint_output=post,
        replacement_window=(10, 15),
        replacement_n_lines=5,
        window_applied="10:def foo():\n11:    pass",
        window_original="10:def foo():\n11:    pass",
    )
    assert r.ok is False
    assert len(r.new_errors) == 1
    assert "would have looked if applied" in r.message


def test_evaluate_edit_lint_message_falls_back_when_windows_missing():
    pre = ""
    post = "x.py:12:5: F821 undefined name 'newvar'"
    r = evaluate_edit_lint(
        pre_lint_output=pre,
        post_lint_output=post,
        replacement_window=(10, 15),
        replacement_n_lines=5,
    )
    # No twin-window template -- just the error list.
    assert r.ok is False
    assert "would have looked if applied" not in r.message
    assert "F821 undefined name" in r.message


# ---------------------------------------------------------------------------
# Dataclass invariants
# ---------------------------------------------------------------------------


def test_lint_error_is_frozen():
    e = LintError("x.py", 1, 1, "msg")
    with pytest.raises(Exception):
        e.line_number = 2  # type: ignore[misc]
