"""Tests for the edit-failure diagnostics (catalog T12)."""

from __future__ import annotations

import pytest

from ultron.coding.edit_diagnostics import (
    CrossFileHit,
    EditDiagnostic,
    EditDiagnosticResult,
    diagnose_edit_failure,
    find_all_in_file,
)


# ---------------------------------------------------------------------------
# find_all_in_file
# ---------------------------------------------------------------------------


def test_find_all_empty_search_returns_empty():
    assert find_all_in_file("hello world", "") == []


def test_find_all_basic_hits():
    text = "alpha\nbeta\nalpha\ngamma\nalpha\n"
    hits = find_all_in_file(text, "alpha")
    assert hits == [1, 3, 5]


def test_find_all_multiline_search_counts_first_line():
    text = "alpha\nbeta\ngamma\n"
    hits = find_all_in_file(text, "beta\ngamma")
    assert hits == [2]


# ---------------------------------------------------------------------------
# NOT_FOUND
# ---------------------------------------------------------------------------


def test_not_found_returns_correct_diagnostic():
    r = diagnose_edit_failure("alpha\nbeta\n", search="zeta")
    assert r.diagnostic == EditDiagnostic.NOT_FOUND
    assert "not found" in r.message.lower()
    assert "zeta" in r.message
    assert "whitespace" in r.message.lower()


def test_not_found_includes_re_open_hint():
    r = diagnose_edit_failure("alpha\nbeta\n", search="zeta")
    assert "open" in r.message.lower()


# ---------------------------------------------------------------------------
# NOT_FOUND_IN_WINDOW
# ---------------------------------------------------------------------------


def test_not_found_in_window_lists_line_numbers():
    text = "alpha\nbeta\nalpha\ngamma\nalpha\n"
    # The window contains only "beta" + "gamma".
    window = "beta\ngamma\n"
    r = diagnose_edit_failure(text, search="alpha", in_window=window)
    assert r.diagnostic == EditDiagnostic.NOT_FOUND_IN_WINDOW
    assert r.line_numbers == [1, 3, 5]
    assert "- line 1" in r.message
    assert "- line 3" in r.message
    assert "- line 5" in r.message
    assert "goto" in r.message.lower()


def test_not_found_in_window_no_other_files_falls_through_to_in_window_diag():
    # The search appears in the file but not the window AND no other
    # files are present. Result is NOT_FOUND_IN_WINDOW.
    text = "alpha\nbeta\nalpha\n"
    window = "beta\n"
    r = diagnose_edit_failure(text, search="alpha", in_window=window, other_files={})
    assert r.diagnostic == EditDiagnostic.NOT_FOUND_IN_WINDOW


# ---------------------------------------------------------------------------
# MULTIPLE_OCCURRENCES_IN_WINDOW
# ---------------------------------------------------------------------------


def test_multiple_in_window_triggered_in_single_replace_mode():
    text = "alpha\nalpha\nalpha\n"
    window = text  # full file is the window
    r = diagnose_edit_failure(
        text, search="alpha", in_window=window, single_replace_required=True
    )
    assert r.diagnostic == EditDiagnostic.MULTIPLE_OCCURRENCES_IN_WINDOW
    assert "more specific" in r.message.lower()


def test_multiple_in_window_not_triggered_in_replace_all_mode():
    text = "alpha\nalpha\nalpha\n"
    window = text
    r = diagnose_edit_failure(
        text, search="alpha", in_window=window, single_replace_required=False
    )
    assert r.diagnostic == EditDiagnostic.OK


# ---------------------------------------------------------------------------
# NO_CHANGES_MADE
# ---------------------------------------------------------------------------


def test_no_changes_made_when_search_equals_replace():
    r = diagnose_edit_failure("alpha\n", search="alpha", replace="alpha")
    assert r.diagnostic == EditDiagnostic.NO_CHANGES_MADE
    assert "no changes" in r.message.lower()


def test_no_changes_skipped_when_search_empty():
    # Empty search isn't NO_CHANGES even if equal; the caller never
    # passes empty searches.
    r = diagnose_edit_failure("alpha\n", search="", replace="")
    assert r.diagnostic != EditDiagnostic.NO_CHANGES_MADE


# ---------------------------------------------------------------------------
# AMBIGUOUS_CROSS_FILE
# ---------------------------------------------------------------------------


def test_ambiguous_cross_file_when_only_in_other_files():
    open_file = "alpha\nbeta\n"
    other = {"/tmp/other.py": "gamma\nzeta\n", "/tmp/another.py": "zeta\n"}
    r = diagnose_edit_failure(open_file, search="zeta", other_files=other)
    assert r.diagnostic == EditDiagnostic.AMBIGUOUS_CROSS_FILE
    paths = {hit.path for hit in r.cross_file_hits}
    assert "/tmp/other.py" in paths
    assert "/tmp/another.py" in paths
    assert "lines" in r.message.lower()


def test_ambiguous_cross_file_reports_correct_line_numbers():
    other = {"/tmp/x.py": "a\nb\nzeta\nzeta\n"}
    r = diagnose_edit_failure("nothing\n", search="zeta", other_files=other)
    hit = r.cross_file_hits[0]
    assert hit.line_numbers == [3, 4]


def test_ambiguous_cross_file_no_match_in_others_falls_back_to_not_found():
    other = {"/tmp/x.py": "alpha\n"}
    r = diagnose_edit_failure("nothing\n", search="zeta", other_files=other)
    assert r.diagnostic == EditDiagnostic.NOT_FOUND


# ---------------------------------------------------------------------------
# OK
# ---------------------------------------------------------------------------


def test_ok_when_single_match_in_window():
    r = diagnose_edit_failure("alpha\nbeta\n", search="beta", in_window="alpha\nbeta\n")
    assert r.diagnostic == EditDiagnostic.OK
    assert r.message == ""


def test_ok_when_no_window_specified_and_unique_match():
    r = diagnose_edit_failure("alpha\nbeta\n", search="beta")
    assert r.diagnostic == EditDiagnostic.OK


# ---------------------------------------------------------------------------
# Dataclass invariants
# ---------------------------------------------------------------------------


def test_result_dataclass_frozen():
    r = EditDiagnosticResult(diagnostic=EditDiagnostic.OK)
    with pytest.raises(Exception):
        r.diagnostic = EditDiagnostic.NOT_FOUND  # type: ignore[misc]


def test_cross_file_hit_dataclass_frozen():
    h = CrossFileHit(path="/x", line_numbers=[1, 2])
    with pytest.raises(Exception):
        h.path = "/y"  # type: ignore[misc]


def test_enum_values_stable():
    # Stable string values so log-grep + audit-log filters work.
    assert EditDiagnostic.NOT_FOUND.value == "not_found"
    assert EditDiagnostic.NOT_FOUND_IN_WINDOW.value == "not_found_in_window"
    assert EditDiagnostic.MULTIPLE_OCCURRENCES_IN_WINDOW.value == (
        "multiple_occurrences_in_window"
    )
    assert EditDiagnostic.NO_CHANGES_MADE.value == "no_changes_made"
    assert EditDiagnostic.AMBIGUOUS_CROSS_FILE.value == "ambiguous_cross_file"
    assert EditDiagnostic.OK.value == "ok"
