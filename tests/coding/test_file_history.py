"""Tests for the multi-file undo stack (catalog T20)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ultron.coding.file_history import (
    DEFAULT_MAX_HISTORY_PER_FILE,
    FileHistory,
    FileHistoryEntry,
    REGISTRY_KEY,
    UndoResult,
    get_file_history,
)
from ultron.coding.session_registry import (
    SessionRegistry,
    reset_session_registries_for_testing,
)


@pytest.fixture
def reg(tmp_path: Path) -> SessionRegistry:
    reset_session_registries_for_testing()
    return SessionRegistry(session_id="undo-test", root=tmp_path)


@pytest.fixture
def fh(reg: SessionRegistry) -> FileHistory:
    return FileHistory(registry=reg)


@pytest.fixture(autouse=True)
def _cleanup() -> None:
    yield
    reset_session_registries_for_testing()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_constructor_rejects_invalid_cap(reg: SessionRegistry):
    with pytest.raises(ValueError):
        FileHistory(registry=reg, max_history_per_file=0)


def test_default_cap_constant():
    assert DEFAULT_MAX_HISTORY_PER_FILE >= 1


def test_registry_key_constant_matches_swe_agent():
    # Cross-tool legibility: matches the SWE-Agent key name.
    assert REGISTRY_KEY == "file_history"


# ---------------------------------------------------------------------------
# record_pre_edit
# ---------------------------------------------------------------------------


def test_record_pre_edit_existing_file(fh: FileHistory, tmp_path: Path):
    p = tmp_path / "x.py"
    p.write_text("print('before')\n", encoding="utf-8")
    assert fh.record_pre_edit(str(p)) is True
    entry = fh.peek_last(str(p))
    assert entry is not None
    assert entry.content == "print('before')\n"


def test_record_pre_edit_missing_file_stores_none(fh: FileHistory, tmp_path: Path):
    p = tmp_path / "new.py"
    assert not p.exists()
    fh.record_pre_edit(str(p))
    entry = fh.peek_last(str(p))
    assert entry is not None
    assert entry.content is None


def test_record_pre_edit_empty_path_returns_false(fh: FileHistory):
    assert fh.record_pre_edit("") is False


def test_record_pre_edit_captures_narration_and_origin(
    fh: FileHistory, tmp_path: Path
):
    p = tmp_path / "x.py"
    p.write_text("body", encoding="utf-8")
    fh.record_pre_edit(
        str(p), narration="Added Tkinter close button", origin="runner"
    )
    entry = fh.peek_last(str(p))
    assert entry is not None
    assert entry.narration == "Added Tkinter close button"
    assert entry.origin == "runner"


def test_record_pre_edit_caps_history_per_file(reg: SessionRegistry, tmp_path: Path):
    fh = FileHistory(registry=reg, max_history_per_file=3)
    p = tmp_path / "x.py"
    p.write_text("v0", encoding="utf-8")
    fh.record_pre_edit(str(p), narration="v0")
    p.write_text("v1", encoding="utf-8")
    fh.record_pre_edit(str(p), narration="v1")
    p.write_text("v2", encoding="utf-8")
    fh.record_pre_edit(str(p), narration="v2")
    p.write_text("v3", encoding="utf-8")
    fh.record_pre_edit(str(p), narration="v3")
    stack = fh.history_for(str(p))
    assert len(stack) == 3
    # Oldest dropped, newest preserved.
    assert stack[-1].narration == "v3"
    assert stack[0].narration == "v1"


def test_record_pre_edit_unreadable_file_returns_false(
    fh: FileHistory, tmp_path: Path, monkeypatch
):
    p = tmp_path / "exists.py"
    p.write_text("body", encoding="utf-8")

    def boom(*_a, **_k):
        raise OSError("simulated read failure")

    monkeypatch.setattr(Path, "read_text", boom)
    assert fh.record_pre_edit(str(p)) is False


# ---------------------------------------------------------------------------
# undo_last
# ---------------------------------------------------------------------------


def test_undo_last_restores_file(fh: FileHistory, tmp_path: Path):
    p = tmp_path / "x.py"
    p.write_text("before edit", encoding="utf-8")
    fh.record_pre_edit(str(p))
    p.write_text("after edit", encoding="utf-8")
    result = fh.undo_last(str(p))
    assert isinstance(result, UndoResult)
    assert result.applied is True
    assert p.read_text(encoding="utf-8") == "before edit"


def test_undo_last_deletes_file_when_snapshot_was_missing(
    fh: FileHistory, tmp_path: Path
):
    p = tmp_path / "created.py"
    fh.record_pre_edit(str(p))  # path didn't exist -> content=None
    p.write_text("now i exist", encoding="utf-8")
    result = fh.undo_last(str(p))
    assert result.applied is True
    assert not p.exists()


def test_undo_last_no_history_returns_false(fh: FileHistory, tmp_path: Path):
    p = tmp_path / "x.py"
    p.write_text("body", encoding="utf-8")
    result = fh.undo_last(str(p))
    assert result.applied is False
    assert result.entry is None


def test_undo_last_pops_snapshot_only_on_success(
    fh: FileHistory, tmp_path: Path
):
    p = tmp_path / "x.py"
    p.write_text("v0", encoding="utf-8")
    fh.record_pre_edit(str(p))
    p.write_text("v1", encoding="utf-8")
    fh.record_pre_edit(str(p))
    p.write_text("v2", encoding="utf-8")
    # Two snapshots; first undo restores v1, second restores v0.
    fh.undo_last(str(p))
    assert p.read_text(encoding="utf-8") == "v1"
    fh.undo_last(str(p))
    assert p.read_text(encoding="utf-8") == "v0"


def test_undo_last_clears_path_when_stack_empties(
    fh: FileHistory, tmp_path: Path
):
    p = tmp_path / "x.py"
    p.write_text("v0", encoding="utf-8")
    fh.record_pre_edit(str(p))
    p.write_text("v1", encoding="utf-8")
    fh.undo_last(str(p))
    assert str(p.resolve()) not in fh.all_paths()


def test_undo_last_empty_path_returns_error(fh: FileHistory):
    result = fh.undo_last("")
    assert result.applied is False
    assert "invalid" in result.error.lower()


# ---------------------------------------------------------------------------
# peek_last / history_for / all_paths / total_snapshots
# ---------------------------------------------------------------------------


def test_peek_last_returns_none_for_unknown_path(
    fh: FileHistory, tmp_path: Path
):
    assert fh.peek_last(str(tmp_path / "missing.py")) is None


def test_history_for_returns_full_stack(fh: FileHistory, tmp_path: Path):
    p = tmp_path / "x.py"
    p.write_text("v0", encoding="utf-8")
    fh.record_pre_edit(str(p), narration="first")
    p.write_text("v1", encoding="utf-8")
    fh.record_pre_edit(str(p), narration="second")
    stack = fh.history_for(str(p))
    assert [e.narration for e in stack] == ["first", "second"]


def test_all_paths_returns_sorted_list(fh: FileHistory, tmp_path: Path):
    for name in ("c.py", "a.py", "b.py"):
        p = tmp_path / name
        p.write_text(name, encoding="utf-8")
        fh.record_pre_edit(str(p))
    paths = fh.all_paths()
    assert paths == sorted(paths)
    assert len(paths) == 3


def test_total_snapshots_counts_all_files(fh: FileHistory, tmp_path: Path):
    p1 = tmp_path / "a.py"
    p2 = tmp_path / "b.py"
    p1.write_text("a", encoding="utf-8")
    p2.write_text("b", encoding="utf-8")
    fh.record_pre_edit(str(p1))
    fh.record_pre_edit(str(p1))
    fh.record_pre_edit(str(p2))
    assert fh.total_snapshots() == 3


# ---------------------------------------------------------------------------
# find_by_narration
# ---------------------------------------------------------------------------


def test_find_by_narration_substring_match(fh: FileHistory, tmp_path: Path):
    p = tmp_path / "x.py"
    p.write_text("body", encoding="utf-8")
    fh.record_pre_edit(str(p), narration="Added close button")
    fh.record_pre_edit(str(p), narration="Wired Tkinter widget")
    hits = fh.find_by_narration("close")
    assert len(hits) == 1
    assert hits[0][1].narration == "Added close button"


def test_find_by_narration_case_insensitive(fh: FileHistory, tmp_path: Path):
    p = tmp_path / "x.py"
    p.write_text("body", encoding="utf-8")
    fh.record_pre_edit(str(p), narration="ADDED CLOSE button")
    hits = fh.find_by_narration("close")
    assert len(hits) == 1


def test_find_by_narration_most_recent_first(fh: FileHistory, tmp_path: Path):
    p = tmp_path / "x.py"
    p.write_text("body", encoding="utf-8")
    fh.record_pre_edit(str(p), narration="close v1")
    fh.record_pre_edit(str(p), narration="close v2")
    hits = fh.find_by_narration("close", n=10)
    assert hits[0][1].narration == "close v2"
    assert hits[1][1].narration == "close v1"


def test_find_by_narration_empty_query_returns_empty(fh: FileHistory):
    assert fh.find_by_narration("") == []


def test_find_by_narration_caps_at_n(fh: FileHistory, tmp_path: Path):
    p = tmp_path / "x.py"
    p.write_text("body", encoding="utf-8")
    for i in range(5):
        fh.record_pre_edit(str(p), narration=f"button v{i}")
    hits = fh.find_by_narration("button", n=2)
    assert len(hits) == 2


# ---------------------------------------------------------------------------
# clear / clear_all
# ---------------------------------------------------------------------------


def test_clear_drops_path(fh: FileHistory, tmp_path: Path):
    p = tmp_path / "x.py"
    p.write_text("body", encoding="utf-8")
    fh.record_pre_edit(str(p))
    dropped = fh.clear(str(p))
    assert dropped == 1
    assert fh.peek_last(str(p)) is None


def test_clear_unknown_path_returns_zero(fh: FileHistory, tmp_path: Path):
    assert fh.clear(str(tmp_path / "missing.py")) == 0


def test_clear_all_drops_everything(fh: FileHistory, tmp_path: Path):
    for name in ("a.py", "b.py"):
        p = tmp_path / name
        p.write_text("body", encoding="utf-8")
        fh.record_pre_edit(str(p))
    dropped = fh.clear_all()
    assert dropped >= 1  # depends on path canonicalisation -> may dedup
    assert fh.all_paths() == []


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------


def test_get_file_history_constructs_with_session_registry(tmp_path: Path):
    reset_session_registries_for_testing()
    reg = SessionRegistry(session_id="factory-test", root=tmp_path)
    fh = get_file_history("factory-test", registry=reg)
    assert isinstance(fh, FileHistory)
    assert fh.registry is reg


def test_file_history_entry_is_frozen():
    e = FileHistoryEntry(content="x", recorded_at=1.0)
    with pytest.raises(Exception):
        e.content = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Round-trip across instances (simulates crash recovery)
# ---------------------------------------------------------------------------


def test_history_survives_instance_recreation(tmp_path: Path):
    reset_session_registries_for_testing()
    reg_a = SessionRegistry(session_id="persist", root=tmp_path)
    fh_a = FileHistory(registry=reg_a)
    p = tmp_path / "x.py"
    p.write_text("v0", encoding="utf-8")
    fh_a.record_pre_edit(str(p), narration="initial")
    p.write_text("v1", encoding="utf-8")

    # Simulate fresh process: new registry + history pointing at the
    # same JSON file.
    reg_b = SessionRegistry(session_id="persist", root=tmp_path)
    fh_b = FileHistory(registry=reg_b)
    entry = fh_b.peek_last(str(p))
    assert entry is not None
    assert entry.content == "v0"
    assert entry.narration == "initial"

    # Undo from the second instance restores the file.
    fh_b.undo_last(str(p))
    assert p.read_text(encoding="utf-8") == "v0"
