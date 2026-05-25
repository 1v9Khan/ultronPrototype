"""Tests for ultron.checkpoints.restore."""

from __future__ import annotations

import pytest

from ultron.checkpoints import restore as rt


class TestPlanRestore:
    def test_voice_history_plan(self) -> None:
        plan = rt.plan_restore(
            axis=rt.RestoreAxis.VOICE_HISTORY,
            truncate_after_turn_id="t-42",
            will_drop_turn_count=3,
        )
        assert plan.axis is rt.RestoreAxis.VOICE_HISTORY
        assert "drop the last 3" in plan.narration

    def test_workspace_plan(self) -> None:
        plan = rt.plan_restore(
            axis=rt.RestoreAxis.WORKSPACE,
            target_commit_hash="abc123def456",
        )
        assert "abc123de" in plan.narration

    def test_both_axes_plan(self) -> None:
        plan = rt.plan_restore(
            axis=rt.RestoreAxis.BOTH,
            target_commit_hash="abc123",
            will_drop_turn_count=2,
        )
        # Multiple actions are joined with "then".
        assert ", then " in plan.narration or "reset workspace" in plan.narration

    def test_empty_workspace_target_uses_generic_narration(self) -> None:
        plan = rt.plan_restore(axis=rt.RestoreAxis.WORKSPACE)
        assert "latest checkpoint" in plan.narration


class TestExecuteRestore:
    def test_voice_history_axis_calls_truncator(self) -> None:
        seen: list[str] = []

        def truncator(turn_id: str) -> int:
            seen.append(turn_id)
            return 3

        plan = rt.plan_restore(
            axis=rt.RestoreAxis.VOICE_HISTORY,
            truncate_after_turn_id="t-5",
            will_drop_turn_count=3,
        )
        outcome = rt.execute_restore(plan, voice_history_truncate=truncator)
        assert outcome.voice_history_truncated == 3
        assert outcome.workspace_reset_succeeded is True
        assert outcome.error_message == ""
        assert seen == ["t-5"]

    def test_workspace_axis_calls_reset(self) -> None:
        captured: list[str] = []

        def reset(commit: str) -> bool:
            captured.append(commit)
            return True

        plan = rt.plan_restore(
            axis=rt.RestoreAxis.WORKSPACE,
            target_commit_hash="abc",
        )
        outcome = rt.execute_restore(plan, workspace_reset=reset)
        assert outcome.workspace_reset_succeeded is True
        assert captured == ["abc"]

    def test_workspace_reset_failure_reported(self) -> None:
        plan = rt.plan_restore(
            axis=rt.RestoreAxis.WORKSPACE,
            target_commit_hash="abc",
        )
        outcome = rt.execute_restore(plan, workspace_reset=lambda _: False)
        assert outcome.workspace_reset_succeeded is False

    def test_both_axis_invokes_each(self) -> None:
        ws_calls: list[str] = []
        hist_calls: list[str] = []
        plan = rt.plan_restore(
            axis=rt.RestoreAxis.BOTH,
            target_commit_hash="abc",
            truncate_after_turn_id="t-1",
            will_drop_turn_count=2,
        )
        outcome = rt.execute_restore(
            plan,
            workspace_reset=lambda c: ws_calls.append(c) or True,
            voice_history_truncate=lambda t: hist_calls.append(t) or 2,
        )
        assert ws_calls == ["abc"]
        assert hist_calls == ["t-1"]
        assert outcome.voice_history_truncated == 2

    def test_missing_callable_reports_error(self) -> None:
        plan = rt.plan_restore(
            axis=rt.RestoreAxis.BOTH,
            target_commit_hash="abc",
            truncate_after_turn_id="t-1",
        )
        outcome = rt.execute_restore(plan)
        # Both callables missing → workspace_reset failed + voice error.
        assert outcome.workspace_reset_succeeded is False
        assert "workspace_reset" in outcome.error_message
        assert "voice_history_truncate" in outcome.error_message

    def test_callable_raises_swallowed(self) -> None:
        def boom(_: str) -> bool:
            raise RuntimeError("nope")
        plan = rt.plan_restore(
            axis=rt.RestoreAxis.WORKSPACE,
            target_commit_hash="abc",
        )
        outcome = rt.execute_restore(plan, workspace_reset=boom)
        assert outcome.workspace_reset_succeeded is False
        assert "RuntimeError" in outcome.error_message

    def test_event_log_truncate_called(self) -> None:
        plan = rt.plan_restore(
            axis=rt.RestoreAxis.VOICE_HISTORY,
            truncate_after_turn_id="t-1",
        )
        outcome = rt.execute_restore(
            plan,
            voice_history_truncate=lambda _: 1,
            event_log_truncate=lambda _: 7,
        )
        assert outcome.events_truncated == 7
