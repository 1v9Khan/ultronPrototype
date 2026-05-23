"""Tests for SupervisorDispatchController's architect-narrator wiring.

Catalog T5 Phase 2 (batch 14). The dispatch controller accepts an
optional ``architect_narrator`` callable; when set AND
``decision.architect_plan_text`` is non-empty, the narrator runs
between the existing decision-narration and the dispatch handoff.
A True return from the narrator short-circuits the dispatch with
``BARGED_IN``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pytest

from ultron.coding.project_supervisor import (
    SupervisorAction,
    SupervisorDecision,
    SupervisorInputs,
)
from ultron.coding.supervisor_dispatch import (
    DispatchActionKind,
    SupervisorDispatchController,
)


class _FakeSupervisor:
    def __init__(self, decision: SupervisorDecision) -> None:
        self.decision = decision

    def decide(self, inputs: SupervisorInputs) -> SupervisorDecision:
        return self.decision


def _edit_decision_with_plan(plan: Optional[str], tmp_path: Path) -> SupervisorDecision:
    return SupervisorDecision(
        action=SupervisorAction.EDIT,
        target_project_id="p1",
        target_project_name="myproj",
        target_project_path=str(tmp_path),
        architect_plan_text=plan,
    )


def test_architect_narrator_not_called_when_plan_text_empty(tmp_path: Path):
    decision = _edit_decision_with_plan(None, tmp_path)
    called_with: List[str] = []

    def narrator(text):
        called_with.append(text)
        return False

    controller = SupervisorDispatchController(
        supervisor=_FakeSupervisor(decision),
        architect_narrator=narrator,
    )
    outcome = controller.dispatch(SupervisorInputs(user_text="add a route"))
    assert called_with == []
    assert outcome.kind == DispatchActionKind.EDIT_DISPATCH


def test_architect_narrator_called_when_plan_text_present(tmp_path: Path):
    decision = _edit_decision_with_plan(
        "Modify foo. Then update bar.", tmp_path,
    )
    called_with: List[str] = []

    def narrator(text):
        called_with.append(text)
        return False  # no interrupt

    controller = SupervisorDispatchController(
        supervisor=_FakeSupervisor(decision),
        architect_narrator=narrator,
    )
    outcome = controller.dispatch(SupervisorInputs(user_text="please apply"))
    assert called_with == ["Modify foo. Then update bar."]
    assert outcome.kind == DispatchActionKind.EDIT_DISPATCH


def test_architect_narrator_barge_in_returns_BARGED_IN(tmp_path: Path):
    decision = _edit_decision_with_plan(
        "First we'll modify foo. Then bar.", tmp_path,
    )

    def narrator(text):
        return True  # user interrupted

    controller = SupervisorDispatchController(
        supervisor=_FakeSupervisor(decision),
        architect_narrator=narrator,
    )
    outcome = controller.dispatch(SupervisorInputs(user_text="apply it"))
    assert outcome.kind == DispatchActionKind.BARGED_IN
    assert outcome.already_narrated is True
    assert outcome.voice_message == ""


def test_architect_narrator_exception_is_fail_open(tmp_path: Path):
    """An exception in the narrator must NOT abort the dispatch -- the
    plan still flows through the prompt and the editor LLM proceeds."""
    decision = _edit_decision_with_plan(
        "Modify foo. Then bar.", tmp_path,
    )

    def narrator(text):
        raise RuntimeError("simulated TTS hiccup")

    controller = SupervisorDispatchController(
        supervisor=_FakeSupervisor(decision),
        architect_narrator=narrator,
    )
    outcome = controller.dispatch(SupervisorInputs(user_text="apply it"))
    assert outcome.kind == DispatchActionKind.EDIT_DISPATCH


def test_architect_narrator_only_runs_after_decision_narration_passes(tmp_path: Path):
    """When narrate_enabled is on AND the digest barge-in fires, the
    architect narrator must NOT be reached -- we've already aborted
    the dispatch."""
    decision = _edit_decision_with_plan(
        "Lots of stuff to do.", tmp_path,
    )
    narrator_calls: List[str] = []

    def narrator(text):
        narrator_calls.append(text)
        return False

    def barge_in_speak(text):
        return True  # decision narration triggered a barge-in immediately

    controller = SupervisorDispatchController(
        supervisor=_FakeSupervisor(decision),
        barge_in_speak=barge_in_speak,
        narrate_enabled=True,
        architect_narrator=narrator,
    )
    outcome = controller.dispatch(SupervisorInputs(user_text="apply it"))
    assert outcome.kind == DispatchActionKind.BARGED_IN
    # Architect narrator must not have run.
    assert narrator_calls == []


def test_architect_narrator_not_called_when_unset(tmp_path: Path):
    """``architect_narrator=None`` is the default: dispatch proceeds
    straight to outcome construction even when plan text is set.
    Existing tests rely on this -- if architect_narrator defaulted to
    a non-None value, every existing test that injects plan text
    would suddenly start hitting a narrator stub."""
    decision = _edit_decision_with_plan(
        "Modify foo.", tmp_path,
    )
    controller = SupervisorDispatchController(
        supervisor=_FakeSupervisor(decision),
    )
    outcome = controller.dispatch(SupervisorInputs(user_text="apply it"))
    # No barge-in / abort -- regular EDIT_DISPATCH.
    assert outcome.kind == DispatchActionKind.EDIT_DISPATCH
