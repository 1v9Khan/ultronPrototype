"""Tests for the bridge-layer helpers (no subprocess, no LLM).

The DirectClaudeCodeBridge end-to-end test lives in
``test_coding_e2e.py`` (slow tier). Here we verify the pure utilities:
  * directory snapshot + diff
  * render_prompt preamble injection
  * TaskState mutation under lock (via DirectTaskHandle's helper)
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ultron.coding.bridge import (
    EventKind,
    FileChangeKind,
    TaskEvent,
    TaskRequest,
    TaskState,
    _StateMutex,
    diff_snapshots,
    directory_snapshot,
    render_prompt,
)


# ---------------------------------------------------------------------------
# directory_snapshot + diff_snapshots
# ---------------------------------------------------------------------------


def test_snapshot_returns_empty_for_missing_dir(tmp_path: Path):
    out = directory_snapshot(tmp_path / "does-not-exist")
    assert out == {}


def test_snapshot_finds_files(tmp_path: Path):
    (tmp_path / "a.py").write_text("hello", encoding="utf-8")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "c.txt").write_text("x", encoding="utf-8")
    snap = directory_snapshot(tmp_path)
    rels = sorted(str(p).replace("\\", "/") for p in snap)
    assert rels == ["a.py", "b/c.txt"]


def test_snapshot_skips_well_known_dirs(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_text("", encoding="utf-8")
    (tmp_path / "src.py").write_text("hello", encoding="utf-8")
    snap = directory_snapshot(tmp_path)
    rels = {str(p).replace("\\", "/") for p in snap}
    assert rels == {"src.py"}


def test_diff_detects_create_modify_delete(tmp_path: Path):
    (tmp_path / "kept.py").write_text("hello", encoding="utf-8")
    (tmp_path / "removed.py").write_text("bye", encoding="utf-8")
    before = directory_snapshot(tmp_path)

    # Wait so mtimes differ; on Windows mtime resolution is generally 1s.
    time.sleep(1.05)
    (tmp_path / "kept.py").write_text("hello again", encoding="utf-8")
    (tmp_path / "added.py").write_text("new", encoding="utf-8")
    (tmp_path / "removed.py").unlink()

    after = directory_snapshot(tmp_path)
    created, modified, deleted = diff_snapshots(before, after)
    assert [str(p).replace("\\", "/") for p in created] == ["added.py"]
    assert [str(p).replace("\\", "/") for p in modified] == ["kept.py"]
    assert [str(p).replace("\\", "/") for p in deleted] == ["removed.py"]


def test_diff_empty_when_unchanged(tmp_path: Path):
    (tmp_path / "x.py").write_text("hi", encoding="utf-8")
    snap = directory_snapshot(tmp_path)
    created, modified, deleted = diff_snapshots(snap, snap)
    assert created == [] and modified == [] and deleted == []


# ---------------------------------------------------------------------------
# render_prompt (discipline preamble)
# ---------------------------------------------------------------------------


def test_render_prompt_prepends_discipline_preamble(tmp_path: Path):
    req = TaskRequest(task_prompt="Add a hello world script.", cwd=tmp_path)
    out = render_prompt(req)
    assert "Write tests for each component" in out
    assert "Add a hello world script." in out
    # Preamble before user task.
    assert out.index("Write tests for each component") < out.index("Add a hello world script.")


def test_render_prompt_skips_preamble_when_disabled(tmp_path: Path):
    req = TaskRequest(
        task_prompt="Add a hello world script.",
        cwd=tmp_path,
        require_testing=False,
    )
    out = render_prompt(req)
    # No testing-discipline preamble when require_testing=False...
    assert "Write tests for each component" not in out
    # ...but the always-on code-quality preamble + the body are still present.
    assert "best practices" in out and "pyproject.toml" in out
    assert out.endswith("Add a hello world script.")


def test_render_prompt_strips_outer_whitespace(tmp_path: Path):
    req = TaskRequest(task_prompt="\n  hello task  \n", cwd=tmp_path,
                       require_testing=False)
    out = render_prompt(req)
    assert out.endswith("hello task") and "  hello task  " not in out


def test_render_prompt_always_includes_quality_preamble(tmp_path: Path):
    """The code-quality preamble (type hints + docstrings + pyproject for new
    projects) is prepended regardless of require_testing -- so voice-dispatched
    tasks (require_testing=False) still get best-practices guidance."""
    for rt in (True, False):
        req = TaskRequest(task_prompt="build a thing", cwd=tmp_path, require_testing=rt)
        out = render_prompt(req)
        assert "type hints" in out and "pyproject.toml" in out
        assert out.endswith("build a thing")


# ---------------------------------------------------------------------------
# _StateMutex snapshot + mutate
# ---------------------------------------------------------------------------


def test_state_mutex_snapshot_is_a_copy(tmp_path: Path):
    state = TaskState(
        label="t", task_prompt="p", cwd=tmp_path, started_at=time.time(),
    )
    mutex = _StateMutex(state)
    snap = mutex.snapshot()
    snap.completed_steps.append("write a.py")
    # Mutating the snapshot doesn't leak into the protected state.
    assert mutex.snapshot().completed_steps == []


def test_state_mutex_mutate_persists(tmp_path: Path):
    state = TaskState(
        label="t", task_prompt="p", cwd=tmp_path, started_at=time.time(),
    )
    mutex = _StateMutex(state)
    mutex.mutate(lambda s: s.completed_steps.append("step 1"))
    assert mutex.snapshot().completed_steps == ["step 1"]


def test_task_event_has_kind_field():
    e = TaskEvent(kind=EventKind.STATUS, stage="running")
    assert e.kind == EventKind.STATUS
    assert e.stage == "running"
    # Optional fields default to None.
    assert e.text is None and e.tool_name is None
