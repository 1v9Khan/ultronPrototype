"""Tests for the T14 edit-snapshot-and-recheck recovery wrapper."""

from __future__ import annotations

from typing import Sequence

import pytest

from ultron.coding.edit_recovery import (
    DEFAULT_MISMATCH_SNIPPET_CHARS,
    EditRecoveryResult,
    EditSpec,
    MISMATCH_ERROR_MARKERS,
    did_edit_likely_apply,
    enrich_mismatch_error,
    is_search_mismatch_error,
    run_edit_with_recovery,
    wrap_edit_tool_with_recovery,
)


# ----------------------------------------------------------------------
# did_edit_likely_apply


def test_heuristic_true_when_both_conditions_pass() -> None:
    original = "def foo():\n    pass\n"
    current = "def foo():\n    return 1\n"
    edits = [EditSpec(old_text="    pass", new_text="    return 1", path="x.py")]
    assert did_edit_likely_apply(
        original_content=original,
        current_content=current,
        edits=edits,
    ) is True


def test_heuristic_false_when_unchanged() -> None:
    edits = [EditSpec(old_text="A", new_text="B")]
    assert did_edit_likely_apply(
        original_content="same",
        current_content="same",
        edits=edits,
    ) is False


def test_heuristic_false_when_new_text_missing() -> None:
    edits = [EditSpec(old_text="pass", new_text="return 1")]
    assert did_edit_likely_apply(
        original_content="def f(): pass",
        current_content="def f(): pass  # touched",  # changed but new_text not present
        edits=edits,
    ) is False


def test_heuristic_false_when_old_text_still_present() -> None:
    edits = [EditSpec(old_text="pass", new_text="return 1")]
    assert did_edit_likely_apply(
        original_content="def f(): pass",
        current_content="def f(): pass  # return 1 (note)",  # both present
        edits=edits,
    ) is False


def test_heuristic_normalises_crlf() -> None:
    original = "line1\r\nline2\r\n"
    current = "line1\nline2_changed\n"
    edits = [EditSpec(old_text="line2", new_text="line2_changed")]
    assert did_edit_likely_apply(
        original_content=original,
        current_content=current,
        edits=edits,
    ) is True


def test_heuristic_empty_edits_returns_false() -> None:
    assert did_edit_likely_apply(
        original_content="a",
        current_content="b",
        edits=[],
    ) is False


def test_heuristic_empty_current_returns_false() -> None:
    edits = [EditSpec(old_text="x", new_text="y")]
    assert did_edit_likely_apply(
        original_content="x",
        current_content="",
        edits=edits,
    ) is False


def test_heuristic_handles_new_text_containing_old() -> None:
    # new_text contains old_text as substring; removing new from
    # working should still leave old absent.
    original = "FOO"
    current = "FOO_PLUS"
    edits = [EditSpec(old_text="FOO", new_text="FOO_PLUS")]
    assert did_edit_likely_apply(
        original_content=original,
        current_content=current,
        edits=edits,
    ) is True


def test_heuristic_empty_new_text_skipped() -> None:
    # An edit with empty new_text means deletion; old must be absent
    # in the current content.
    original = "before\nDELETE_ME\nafter"
    current = "before\nafter"
    edits = [EditSpec(old_text="DELETE_ME\n", new_text="")]
    assert did_edit_likely_apply(
        original_content=original,
        current_content=current,
        edits=edits,
    ) is True


# ----------------------------------------------------------------------
# is_search_mismatch_error


def test_is_search_mismatch_recognises_first_marker() -> None:
    err = RuntimeError("Could not find the exact text in foo.py")
    assert is_search_mismatch_error(err)


def test_is_search_mismatch_recognises_alt_marker() -> None:
    err = ValueError("no exact match for the pattern")
    assert is_search_mismatch_error(err)


def test_is_search_mismatch_returns_false_for_unrelated() -> None:
    err = OSError("permission denied")
    assert is_search_mismatch_error(err) is False


# ----------------------------------------------------------------------
# enrich_mismatch_error


def test_enrich_includes_snippet() -> None:
    err = RuntimeError("Could not find text")
    out = enrich_mismatch_error(err, current_content="ABCDEF")
    assert "Current file contents" in out
    assert "ABCDEF" in out


def test_enrich_truncates_long_snippet() -> None:
    err = RuntimeError("Could not find text")
    out = enrich_mismatch_error(err, current_content="A" * 2000, max_chars=10)
    assert "truncated" in out
    assert out.count("A") == 10


# ----------------------------------------------------------------------
# run_edit_with_recovery


def test_run_edit_succeeds_when_tool_succeeds() -> None:
    edits = [EditSpec(old_text="x", new_text="y", path="f.py")]
    reads: list[str] = []

    def reader(p: str) -> str:
        reads.append(p)
        return "snapshot"

    def tool(es: Sequence[EditSpec]):
        return "tool_ok"

    result = run_edit_with_recovery(edits, edit_tool=tool, read_file=reader)
    assert result.succeeded
    assert result.recovered is False
    assert result.tool_result == "tool_ok"
    # Reader was called once for the pre-edit snapshot.
    assert reads.count("f.py") == 1


def test_run_edit_recovers_when_heuristic_says_yes() -> None:
    edits = [EditSpec(old_text="pass", new_text="return 1", path="f.py")]
    state = {"phase": "pre"}

    def reader(p: str) -> str:
        if state["phase"] == "pre":
            return "def f(): pass"
        return "def f(): return 1"

    def tool(es: Sequence[EditSpec]):
        # Simulate the edit landing then a spurious post-write error.
        state["phase"] = "post"
        raise RuntimeError("post-write validation error")

    result = run_edit_with_recovery(edits, edit_tool=tool, read_file=reader)
    assert result.succeeded
    assert result.recovered is True
    assert result.raw_error is not None  # preserved for debug


def test_run_edit_does_not_recover_when_file_unchanged() -> None:
    edits = [EditSpec(old_text="pass", new_text="return 1", path="f.py")]

    def reader(p: str) -> str:
        return "def f(): pass"

    def tool(es: Sequence[EditSpec]):
        raise RuntimeError("could not find the exact text in f.py")

    result = run_edit_with_recovery(edits, edit_tool=tool, read_file=reader)
    assert result.succeeded is False
    assert result.enriched_error is not None
    assert "Current file contents" in result.enriched_error


def test_run_edit_unrelated_error_not_enriched() -> None:
    edits = [EditSpec(old_text="x", new_text="y", path="f.py")]

    def reader(p: str) -> str:
        return "snapshot"

    def tool(es: Sequence[EditSpec]):
        raise OSError("permission denied")

    result = run_edit_with_recovery(edits, edit_tool=tool, read_file=reader)
    assert result.succeeded is False
    assert result.enriched_error is None


def test_run_edit_no_path_skips_snapshot() -> None:
    edits = [EditSpec(old_text="x", new_text="y")]  # no path
    reads: list[str] = []

    def reader(p: str) -> str:
        reads.append(p)
        return ""

    def tool(es: Sequence[EditSpec]):
        return "ok"

    result = run_edit_with_recovery(edits, edit_tool=tool, read_file=reader)
    assert result.succeeded
    # No path -> no read.
    assert reads == []


def test_run_edit_reader_exception_swallowed() -> None:
    edits = [EditSpec(old_text="x", new_text="y", path="f.py")]

    def reader(p: str) -> str:
        raise OSError("read failed")

    def tool(es: Sequence[EditSpec]):
        return "ok"

    # Reader fails -> snapshot None -> tool still runs.
    result = run_edit_with_recovery(edits, edit_tool=tool, read_file=reader)
    assert result.succeeded
    assert result.original_content is None


def test_wrap_edit_tool_returns_callable() -> None:
    def reader(p: str) -> str:
        return ""

    def tool(es: Sequence[EditSpec]):
        return "ok"

    wrapped = wrap_edit_tool_with_recovery(tool, read_file=reader)
    result = wrapped([EditSpec(old_text="x", new_text="y", path="f.py")])
    assert isinstance(result, EditRecoveryResult)
    assert result.succeeded


def test_disable_enrich_mismatch() -> None:
    edits = [EditSpec(old_text="pass", new_text="return", path="f.py")]

    def reader(p: str) -> str:
        return "def f(): pass"

    def tool(es: Sequence[EditSpec]):
        raise RuntimeError("could not find the exact text")

    result = run_edit_with_recovery(
        edits, edit_tool=tool, read_file=reader, enrich_mismatch=False,
    )
    assert result.enriched_error is None
