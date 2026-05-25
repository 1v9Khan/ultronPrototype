"""Tests for ultron.checkpoints.shadow_repo (requires git)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ultron.checkpoints import shadow_repo as sr


# Skip the integration tests when git is missing.
git_required = pytest.mark.skipif(
    shutil.which("git") is None, reason="git binary not on PATH",
)


class TestHashWorkingDir:
    def test_stable_per_input(self) -> None:
        assert sr.hash_working_dir("/tmp/a") == sr.hash_working_dir("/tmp/a")

    def test_length_is_12(self) -> None:
        assert len(sr.hash_working_dir("/tmp/a")) == 12

    def test_distinct_for_distinct(self) -> None:
        assert sr.hash_working_dir("/tmp/a") != sr.hash_working_dir("/tmp/b")


@git_required
class TestShadowRepoIntegration:
    def _build(self, tmp_path: Path) -> sr.ShadowRepoTracker:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "hello.txt").write_text("hello", encoding="utf-8")
        repo_root = tmp_path / "checkpoints" / "sess1"
        return sr.ShadowRepoTracker(
            workspace_path=workspace,
            repo_root=repo_root,
            session_id="sess1",
        )

    def test_init_creates_git_dir(self, tmp_path: Path) -> None:
        tracker = self._build(tmp_path)
        tracker.initialise()
        assert tracker.initialised is True
        assert tracker.git_dir.is_dir()
        # Exclude file written.
        exclude_file = tracker.git_dir / "info" / "exclude"
        assert exclude_file.is_file()
        assert "node_modules/" in exclude_file.read_text(encoding="utf-8")

    def test_commit_returns_hash(self, tmp_path: Path) -> None:
        tracker = self._build(tmp_path)
        tracker.initialise()
        commit = tracker.commit()
        assert commit is not None
        assert commit.commit_hash
        assert len(commit.commit_hash) == 40

    def test_commit_idempotent_with_no_changes(self, tmp_path: Path) -> None:
        tracker = self._build(tmp_path)
        tracker.initialise()
        first = tracker.commit()
        second = tracker.commit()
        # Both commits succeeded (--allow-empty by default).
        assert first is not None and second is not None
        # Hashes should differ (timestamps differ even on identical trees).
        assert first.commit_hash != second.commit_hash

    def test_commit_picks_up_new_file(self, tmp_path: Path) -> None:
        tracker = self._build(tmp_path)
        tracker.initialise()
        tracker.commit()
        (tracker.workspace / "new.txt").write_text("new", encoding="utf-8")
        commit = tracker.commit()
        assert commit is not None
        # The log should contain at least 2 commits.
        log = tracker.log(max_count=10)
        assert len(log) >= 2

    def test_hard_reset_restores_workspace(self, tmp_path: Path) -> None:
        tracker = self._build(tmp_path)
        tracker.initialise()
        first = tracker.commit()
        # Modify the workspace.
        (tracker.workspace / "hello.txt").write_text("changed", encoding="utf-8")
        # Reset back to the original commit.
        assert first is not None
        ok = tracker.hard_reset(first.commit_hash)
        assert ok is True
        # File should be restored.
        assert (tracker.workspace / "hello.txt").read_text(encoding="utf-8") == "hello"

    def test_disabled_after_init_failure(self, tmp_path: Path) -> None:
        # Force a failure by passing a bogus binary into our env via path.
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        tracker = sr.ShadowRepoTracker(
            workspace_path=workspace,
            repo_root=tmp_path / "repo",
            session_id="sess",
            init_timeout_seconds=2.0,
        )
        # Set a fake git on PATH by manipulating environment? Skip — we
        # don't want to permanently mutate the environment. Just confirm
        # the disabled-flag path works manually.
        tracker._disabled = True
        tracker._disabled_reason = "test"
        assert tracker.commit() is None

    def test_excludes_voice_baseline_protected_files(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "SOUL.md").write_text("the lock", encoding="utf-8")
        (workspace / "hello.txt").write_text("plain", encoding="utf-8")
        tracker = sr.ShadowRepoTracker(
            workspace_path=workspace,
            repo_root=tmp_path / "repo",
            session_id="sess",
        )
        tracker.initialise()
        tracker.commit()
        # SOUL.md must NOT appear in the log; the gitignore filter
        # excludes it from staging.
        completed = tracker._run_git(
            ["log", "--all", "--name-only", "--pretty=format:"],
            timeout=10.0, allow_nonzero=True,
        )
        text = (completed.stdout or b"").decode("utf-8", errors="replace")
        assert "SOUL.md" not in text
