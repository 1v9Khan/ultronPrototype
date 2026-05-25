"""Tests for ultron.checkpoints.registry (uses test-doubles for the tracker)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest

from ultron.checkpoints import registry as reg
from ultron.checkpoints.restore import RestoreAxis
from ultron.checkpoints.shadow_repo import CheckpointCommit, ShadowRepoTracker


class _StubTracker:
    """Test double that records calls without spawning git."""

    def __init__(self) -> None:
        self.commits: list[str] = []
        self.resets: list[str] = []
        self.workspace = Path("/fake")
        self.session_id = "stub"

    def commit(self, *, extra_message: str = "", **_kwargs) -> CheckpointCommit:  # type: ignore[no-untyped-def]
        self.commits.append(extra_message)
        return CheckpointCommit(
            commit_hash="abc" + str(len(self.commits)).zfill(2),
            message=extra_message,
            timestamp=0.0,
        )

    def hard_reset(self, commit_hash: str) -> bool:
        self.resets.append(commit_hash)
        return True


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    reg.reset_checkpoint_registry_for_testing()
    yield
    reg.reset_checkpoint_registry_for_testing()


# ---------------------------------------------------------------------------
# SessionCheckpointManager
# ---------------------------------------------------------------------------

class TestSessionManager:
    def test_event_kind_filter(self) -> None:
        tracker = _StubTracker()
        manager = reg.SessionCheckpointManager(
            tracker=tracker,  # type: ignore[arg-type]
            triggered_event_kinds={"CodingFileChangedEvent"},
        )
        # Matching event: commits.
        out = manager.on_event("CodingFileChangedEvent")
        assert out is not None
        # Non-matching event: skipped.
        none_out = manager.on_event("UnrelatedEvent")
        assert none_out is None
        assert len(tracker.commits) == 1

    def test_force_bypasses_filter(self) -> None:
        tracker = _StubTracker()
        manager = reg.SessionCheckpointManager(
            tracker=tracker,  # type: ignore[arg-type]
            triggered_event_kinds=set(),
        )
        out = manager.on_event("Anything", force=True)
        assert out is not None
        assert len(tracker.commits) == 1

    def test_register_unregister(self) -> None:
        tracker = _StubTracker()
        manager = reg.SessionCheckpointManager(
            tracker=tracker,  # type: ignore[arg-type]
            triggered_event_kinds=set(),
        )
        manager.register_event_kind("NewKind")
        assert "NewKind" in manager.configured_event_kinds()
        manager.unregister_event_kind("NewKind")
        assert "NewKind" not in manager.configured_event_kinds()

    def test_commits_list_and_head(self) -> None:
        tracker = _StubTracker()
        manager = reg.SessionCheckpointManager(
            tracker=tracker,  # type: ignore[arg-type]
            triggered_event_kinds={"X"},
        )
        manager.on_event("X")
        manager.on_event("X")
        commits = manager.commits()
        assert len(commits) == 2
        assert manager.head_commit() is commits[-1]

    def test_plan_voice_history_undo(self) -> None:
        tracker = _StubTracker()
        manager = reg.SessionCheckpointManager(
            tracker=tracker,  # type: ignore[arg-type]
        )
        plan = manager.plan_voice_history_undo(offset=3, after_turn_id="t-7")
        assert plan.axis is RestoreAxis.VOICE_HISTORY
        assert plan.will_drop_turn_count == 3

    def test_plan_full_rewind_uses_head_when_not_specified(self) -> None:
        tracker = _StubTracker()
        manager = reg.SessionCheckpointManager(
            tracker=tracker,  # type: ignore[arg-type]
            triggered_event_kinds={"X"},
        )
        manager.on_event("X")
        plan = manager.plan_full_rewind(offset=2)
        assert plan.axis is RestoreAxis.BOTH
        assert plan.target_commit_hash.startswith("abc")

    def test_restore_invokes_tracker_hard_reset(self) -> None:
        tracker = _StubTracker()
        manager = reg.SessionCheckpointManager(
            tracker=tracker,  # type: ignore[arg-type]
            triggered_event_kinds={"X"},
        )
        manager.voice_history_truncate = lambda _t: 1  # type: ignore[assignment]
        manager.on_event("X")
        plan = manager.plan_full_rewind(offset=1)
        outcome = manager.restore(plan)
        assert outcome.workspace_reset_succeeded is True
        assert outcome.voice_history_truncated == 1
        assert tracker.resets  # hard_reset called

    def test_restore_with_explicit_workspace_reset(self) -> None:
        tracker = _StubTracker()
        captured: list[str] = []

        manager = reg.SessionCheckpointManager(
            tracker=tracker,  # type: ignore[arg-type]
            workspace_reset=lambda c: captured.append(c) or True,  # type: ignore[arg-type]
            voice_history_truncate=lambda _: 0,  # type: ignore[arg-type]
        )
        plan = manager.plan_workspace_rewind(target_commit_hash="xyz")
        outcome = manager.restore(plan)
        assert outcome.workspace_reset_succeeded is True
        assert captured == ["xyz"]
        # Explicit workspace_reset means the tracker's hard_reset is NOT used.
        assert tracker.resets == []


# ---------------------------------------------------------------------------
# CheckpointRegistry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_get_or_create_returns_stable(self, tmp_path: Path) -> None:
        registry = reg.CheckpointRegistry(checkpoints_root=tmp_path / "cp")
        a = registry.get_or_create("s1", tmp_path / "ws")
        b = registry.get_or_create("s1", tmp_path / "ws")
        assert a is b

    def test_distinct_sessions(self, tmp_path: Path) -> None:
        registry = reg.CheckpointRegistry(checkpoints_root=tmp_path / "cp")
        a = registry.get_or_create("s1", tmp_path / "ws")
        b = registry.get_or_create("s2", tmp_path / "ws")
        assert a is not b

    def test_drop_removes(self, tmp_path: Path) -> None:
        registry = reg.CheckpointRegistry(checkpoints_root=tmp_path / "cp")
        registry.get_or_create("s1", tmp_path / "ws")
        assert registry.drop("s1") is True
        assert registry.manager_for("s1") is None

    def test_list_sessions(self, tmp_path: Path) -> None:
        registry = reg.CheckpointRegistry(checkpoints_root=tmp_path / "cp")
        registry.get_or_create("s1", tmp_path / "ws")
        registry.get_or_create("s2", tmp_path / "ws")
        assert set(registry.list_sessions()) == {"s1", "s2"}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_lazy_construction(self) -> None:
        a = reg.get_checkpoint_registry()
        b = reg.get_checkpoint_registry()
        assert a is b

    def test_rebuild(self, tmp_path: Path) -> None:
        a = reg.get_checkpoint_registry()
        b = reg.get_checkpoint_registry(
            checkpoints_root=tmp_path / "cp",
            rebuild=True,
        )
        assert a is not b
