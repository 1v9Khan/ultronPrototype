"""Orchestrator-level drains for coding-runner queues.

Two voice-loop drains surface runner-side queues the user needs to hear:

* ``_drain_coding_dialog_narrations`` -- speaks dialog-appearance lines the
  catalog 08/09 dialog auto-handler queues (previously queued but never
  surfaced).
* ``_drain_coding_loop_alerts`` -- speaks the T1 loop-detection heads-up.

Both are exercised via an Orchestrator __new__ stub (the drains only touch
``coding_voice`` + ``_speak`` + ``_last_response_finished_monotonic``).
"""

from __future__ import annotations

from types import SimpleNamespace

from ultron.pipeline.orchestrator import Orchestrator


def _orch_with_runner(runner):
    o = Orchestrator.__new__(Orchestrator)
    o.coding_voice = SimpleNamespace(runner=runner)
    o._last_response_finished_monotonic = 0.0
    spoken: list[str] = []
    o._speak = lambda text: spoken.append(text)  # type: ignore[assignment]
    return o, spoken


def test_dialog_narrations_are_drained_and_spoken():
    lines = ["A 'Save As' dialog appeared in notepad.exe -- shall I confirm?"]
    runner = SimpleNamespace(
        pop_dialog_narration=lambda: lines.pop(0) if lines else None,
    )
    o, spoken = _orch_with_runner(runner)
    o._drain_coding_dialog_narrations()
    assert spoken == ["A 'Save As' dialog appeared in notepad.exe -- shall I confirm?"]


def test_dialog_drain_noop_when_queue_empty():
    runner = SimpleNamespace(pop_dialog_narration=lambda: None)
    o, spoken = _orch_with_runner(runner)
    o._drain_coding_dialog_narrations()
    assert spoken == []


def test_dialog_drain_noop_when_no_coding_voice():
    o = Orchestrator.__new__(Orchestrator)
    o.coding_voice = None
    # Must not raise even though _speak is never bound.
    o._drain_coding_dialog_narrations()


def test_loop_alerts_are_drained_and_spoken():
    alerts = ["Heads up -- the coding agent has repeated the same step."]
    runner = SimpleNamespace(
        pop_loop_alert=lambda: alerts.pop(0) if alerts else None,
    )
    o, spoken = _orch_with_runner(runner)
    o._drain_coding_loop_alerts()
    assert spoken == ["Heads up -- the coding agent has repeated the same step."]


def test_loop_alert_drain_fail_open_on_runner_error():
    def _boom():
        raise RuntimeError("runner exploded")

    runner = SimpleNamespace(pop_loop_alert=_boom)
    o, spoken = _orch_with_runner(runner)
    # Swallowed -- never raises into the voice loop.
    o._drain_coding_loop_alerts()
    assert spoken == []
