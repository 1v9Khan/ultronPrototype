"""Tests for the voice "scrap it" cancel + revert (production-hardening #4).

Covers the strict matcher (ordinary cancels / conversation never trip),
the FileHistory-backed revert (a real round-trip under tmp_path restores
pre-task content and deletes created files), the TTS summary, the voice
controller handler, and the orchestrator short-circuit. All hermetic.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

import ultron.coding.scrap as scrap_mod
from ultron.coding.file_history import FileHistory
from ultron.coding.scrap import (
    ScrapRevertResult,
    match_scrap_command,
    revert_session_edits,
    summarize_scrap,
)
from ultron.coding.session_registry import SessionRegistry
from ultron.coding.voice import CapabilityVoiceController, VoiceResponse
from ultron.pipeline.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# matcher
# ---------------------------------------------------------------------------


class TestMatcher:
    @pytest.mark.parametrize(
        "text",
        [
            "scrap it",
            "Scrap that.",
            "ultron, scrap the project",
            "just scrap it",
            "scrap the whole thing",
            "scrap everything",
            "throw it away",
            "throw that out",
            "trash it",
            "trash the project",
            "undo everything you just did",
            "revert all the changes",
            "undo all of that",
            "revert everything",
            "cancel it and revert",
            "cancel the task and undo the changes",
        ],
    )
    def test_positive(self, text: str) -> None:
        assert match_scrap_command(text)

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "cancel",
            "cancel the task",
            "stop",
            "undo that",  # dual-history territory; deliberately NOT scrap
            "scrap metal prices are up",
            "what's the scrap value of copper",
            "revert the last commit in git",
            "tell me about the project",
            "throw a party",
        ],
    )
    def test_negative(self, text: str) -> None:
        assert not match_scrap_command(text)


# ---------------------------------------------------------------------------
# revert round-trip (real FileHistory under tmp_path)
# ---------------------------------------------------------------------------


def _history(tmp_path: Path) -> FileHistory:
    return FileHistory(
        registry=SessionRegistry(session_id="scrap-test", root=tmp_path / "reg")
    )


class TestRevert:
    def test_restores_modified_and_deletes_created(self, tmp_path: Path) -> None:
        history = _history(tmp_path)
        edited = tmp_path / "app.py"
        edited.write_text("original content\n", encoding="utf-8")
        history.record_pre_edit(str(edited))
        edited.write_text("claude's first edit\n", encoding="utf-8")
        history.record_pre_edit(str(edited))
        edited.write_text("claude's second edit\n", encoding="utf-8")

        created = tmp_path / "helper.py"
        history.record_pre_edit(str(created))  # did not exist -> creation marker
        created.write_text("brand new file\n", encoding="utf-8")

        result = revert_session_edits("scrap-test", history=history)
        assert result.had_history
        assert result.files_restored == 1
        assert result.files_deleted == 1
        assert result.errors == 0
        assert edited.read_text(encoding="utf-8") == "original content\n"
        assert not created.exists()
        # History is cleared so a second scrap can't double-revert.
        assert history.all_paths() == []

    def test_no_history_reports_empty(self, tmp_path: Path) -> None:
        history = _history(tmp_path)
        result = revert_session_edits("scrap-test", history=history)
        assert not result.had_history
        assert result.files_reverted == 0
        assert result.errors == 0

    def test_broken_history_is_fail_open(self) -> None:
        class _Boom:
            def all_paths(self) -> list:
                raise RuntimeError("store boom")

        result = revert_session_edits("x", history=_Boom())
        assert result.errors == 1
        assert result.files_reverted == 0


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_cancel_and_revert(self) -> None:
        msg = summarize_scrap(
            cancelled=True,
            result=ScrapRevertResult(files_restored=2, files_deleted=1, had_history=True),
        )
        assert "cancelled the task" in msg
        assert "reverted 3 files" in msg

    def test_revert_only_singular(self) -> None:
        msg = summarize_scrap(
            cancelled=False,
            result=ScrapRevertResult(files_restored=1, had_history=True),
        )
        assert "cancelled" not in msg
        assert "reverted 1 file." in msg

    def test_nothing_to_revert(self) -> None:
        msg = summarize_scrap(cancelled=True, result=ScrapRevertResult())
        assert "no recorded edits" in msg

    def test_errors_surfaced(self) -> None:
        msg = summarize_scrap(
            cancelled=False,
            result=ScrapRevertResult(files_restored=2, errors=1, had_history=True),
        )
        assert "could not be reverted" in msg


# ---------------------------------------------------------------------------
# voice controller handler
# ---------------------------------------------------------------------------


def _controller(runner: Any) -> Any:
    c = CapabilityVoiceController.__new__(CapabilityVoiceController)
    c.runner = runner
    return c


def _runner(
    *, state: Any = None, running: bool = False, claude_session_id: Optional[str] = None
) -> Any:
    cancels: list[int] = []
    runner = SimpleNamespace(
        active_state=lambda: state,
        has_active_task=lambda: running,
        cancel_active=lambda: cancels.append(1),
        _handle=SimpleNamespace(claude_session_id=claude_session_id),
        cancels=cancels,
    )
    return runner


class TestControllerHandler:
    def test_non_scrap_text_falls_through(self) -> None:
        c = _controller(_runner())
        assert c.maybe_handle_scrap_command("tell me a joke") is None

    def test_no_recent_task_is_honest(self) -> None:
        c = _controller(_runner(state=None))
        resp = c.maybe_handle_scrap_command("scrap it")
        assert resp is not None
        assert "no recent coding task" in resp.text

    def test_active_task_cancelled_and_reverted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reverted: list[str] = []

        def _fake_revert(session_id: str, *, history=None) -> ScrapRevertResult:
            reverted.append(session_id)
            return ScrapRevertResult(files_restored=2, had_history=True)

        monkeypatch.setattr(scrap_mod, "revert_session_edits", _fake_revert)
        runner = _runner(
            state=SimpleNamespace(cwd=Path("C:/sandbox/calc")),
            running=True,
            claude_session_id="sess-abc",
        )
        c = _controller(runner)
        resp = c.maybe_handle_scrap_command("scrap it")
        assert resp is not None and resp.cancelled
        assert runner.cancels == [1]
        # Both session keys tried: the claude session id + the cwd hash.
        assert "sess-abc" in reverted
        assert any(s.startswith("cwd-") for s in reverted)
        assert "cancelled the task" in resp.text

    def test_finished_task_reverts_without_cancel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            scrap_mod,
            "revert_session_edits",
            lambda sid, *, history=None: ScrapRevertResult(
                files_deleted=1, had_history=True
            ),
        )
        runner = _runner(
            state=SimpleNamespace(cwd=Path("C:/sandbox/calc")), running=False
        )
        c = _controller(runner)
        resp = c.maybe_handle_scrap_command("throw it away")
        assert resp is not None and not resp.cancelled
        assert runner.cancels == []
        assert "reverted 1 file" in resp.text


# ---------------------------------------------------------------------------
# orchestrator short-circuit
# ---------------------------------------------------------------------------


class TestOrchestratorShortCircuit:
    @staticmethod
    def _orch(cv: Any) -> Any:
        o = Orchestrator.__new__(Orchestrator)
        o.coding_voice = cv
        o._spoken = []
        o._speak = lambda text: o._spoken.append(text)  # type: ignore[attr-defined]
        return o

    def test_handled_speaks_and_returns_true(self) -> None:
        cv = SimpleNamespace(
            maybe_handle_scrap_command=lambda text: VoiceResponse(text="Scrapped.")
        )
        o = self._orch(cv)
        assert o._maybe_handle_scrap_command("scrap it") is True
        assert o._spoken == ["Scrapped."]

    def test_unhandled_returns_false(self) -> None:
        cv = SimpleNamespace(maybe_handle_scrap_command=lambda text: None)
        o = self._orch(cv)
        assert o._maybe_handle_scrap_command("hello") is False
        assert o._spoken == []

    def test_no_coding_voice_returns_false(self) -> None:
        o = self._orch(None)
        assert o._maybe_handle_scrap_command("scrap it") is False

    def test_fail_open_on_controller_raise(self) -> None:
        def _boom(text: str) -> None:
            raise RuntimeError("controller boom")

        cv = SimpleNamespace(maybe_handle_scrap_command=_boom)
        o = self._orch(cv)
        assert o._maybe_handle_scrap_command("scrap it") is False
