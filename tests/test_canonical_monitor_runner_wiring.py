"""4B optimization plan Item 7 — runner-listener integration tests.

Verifies the runner attaches a canonical-path-monitor listener when the
config flag is on, that the listener cancels the active task on the
first abort verdict, and that the voice loop polls + clears the
abort narration. Mocks the bridge + handle so no Claude subprocess
runs.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ultron.coding.bridge import EventKind, TaskEvent


def _enabled_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.coding.canonical_monitor.enabled = True
    cfg.coding.canonical_monitor.off_canonical_threshold = 3
    cfg.coding.canonical_monitor.early_window_calls = 10
    return cfg


def _disabled_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.coding.canonical_monitor.enabled = False
    return cfg


def _make_runner_with_mocks():
    """Construct a CodingTaskRunner with the bridge mocked out.

    ``_log_path`` is force-nulled after construction so the JSONL log
    listener doesn't pollute the listener list — keeps tests focused
    on the canonical-monitor listener specifically.
    """
    from ultron.coding.runner import CodingTaskRunner

    bridge = MagicMock()
    bridge.name.return_value = "mock"

    handle = MagicMock()
    handle.task_id.return_value = "tid-123"
    handle.is_running.return_value = True
    handle.claude_session_id = None
    handle._listeners = []  # noqa: SLF001
    handle.add_listener.side_effect = lambda fn: handle._listeners.append(fn)
    bridge.submit.return_value = handle

    runner = CodingTaskRunner(bridge=bridge, log_path=None)
    runner._log_path = None  # noqa: SLF001 — suppress JSONL log listener
    return runner, handle


def _drive_event(handle, event):
    """Send an event to every listener that the runner attached."""
    for listener in handle._listeners:  # noqa: SLF001
        listener(event)


# ---------------------------------------------------------------------------
# Listener attachment is gated by config
# ---------------------------------------------------------------------------


def test_listener_not_attached_when_disabled() -> None:
    from ultron.coding.bridge import TaskRequest

    runner, handle = _make_runner_with_mocks()
    cfg = _disabled_cfg()
    with patch("ultron.coding.canonical_monitor.get_config", return_value=cfg):
        request = TaskRequest(
            task_prompt="x", cwd="C:/tmp", model="haiku", label="t",
        )
        runner.start_task(request)

    # When the canonical monitor is disabled the only listener
    # attached is the 2026-05-12 safety validator FILE_CHANGE
    # listener (always-on; degrades to no-op if the safety subsystem
    # is unavailable). Default runner has log_path=None + no bound
    # session, so no log listener / usage listener attach.
    assert len(handle._listeners) == 1


def test_listener_attached_when_enabled() -> None:
    from ultron.coding.bridge import TaskRequest

    runner, handle = _make_runner_with_mocks()
    cfg = _enabled_cfg()
    with patch("ultron.coding.canonical_monitor.get_config", return_value=cfg):
        request = TaskRequest(
            task_prompt="x", cwd="C:/tmp", model="haiku", label="t",
        )
        runner.start_task(request)
    # Canonical-monitor listener + safety-validator FILE_CHANGE
    # listener (always-on as of 2026-05-12 Phase 2).
    assert len(handle._listeners) == 2


# ---------------------------------------------------------------------------
# Abort flow
# ---------------------------------------------------------------------------


def test_listener_cancels_handle_on_abort_verdict() -> None:
    from ultron.coding.bridge import TaskRequest

    runner, handle = _make_runner_with_mocks()
    cfg = _enabled_cfg()
    with patch("ultron.coding.canonical_monitor.get_config", return_value=cfg):
        request = TaskRequest(
            task_prompt="x", cwd="C:/tmp", model="haiku", label="t",
        )
        runner.start_task(request)

        # Drive 3 off-canonical tool_use events — should trigger abort.
        for tname in ["weird_a", "weird_b", "weird_c"]:
            _drive_event(handle, TaskEvent(
                kind=EventKind.TOOL_USE, tool_name=tname,
            ))

    # Handle was cancelled exactly once
    assert handle.cancel.call_count == 1
    # Voice narration is queued and polls cleanly
    msg = runner.pop_canonical_abort_warning()
    assert msg is not None
    assert "stopping that task" in msg
    assert "unexpected tool calls" in msg
    # Polling again clears (consumed-once)
    assert runner.pop_canonical_abort_warning() is None


def test_listener_does_not_cancel_on_canonical_calls() -> None:
    from ultron.coding.bridge import TaskRequest

    runner, handle = _make_runner_with_mocks()
    cfg = _enabled_cfg()
    with patch("ultron.coding.canonical_monitor.get_config", return_value=cfg):
        request = TaskRequest(
            task_prompt="x", cwd="C:/tmp", model="haiku", label="t",
        )
        runner.start_task(request)
        for tname in ["Read", "Edit", "Bash", "Write", "Grep"]:
            _drive_event(handle, TaskEvent(
                kind=EventKind.TOOL_USE, tool_name=tname,
            ))

    assert handle.cancel.call_count == 0
    assert runner.pop_canonical_abort_warning() is None


def test_listener_latches_after_first_abort() -> None:
    """Once the listener has cancelled, subsequent off-canonical calls
    must NOT trigger a second cancel — bridge events keep flowing
    until the handle's own cancel propagation completes."""
    from ultron.coding.bridge import TaskRequest

    runner, handle = _make_runner_with_mocks()
    cfg = _enabled_cfg()
    with patch("ultron.coding.canonical_monitor.get_config", return_value=cfg):
        request = TaskRequest(
            task_prompt="x", cwd="C:/tmp", model="haiku", label="t",
        )
        runner.start_task(request)
        for tname in ["weird_a", "weird_b", "weird_c", "weird_d", "weird_e"]:
            _drive_event(handle, TaskEvent(
                kind=EventKind.TOOL_USE, tool_name=tname,
            ))

    assert handle.cancel.call_count == 1


def test_listener_swallows_exceptions() -> None:
    """A listener exception must NEVER raise back to the bridge —
    that would break event delivery."""
    from ultron.coding.bridge import TaskRequest

    runner, handle = _make_runner_with_mocks()
    cfg = _enabled_cfg()
    handle.cancel.side_effect = RuntimeError("cancel boom")
    with patch("ultron.coding.canonical_monitor.get_config", return_value=cfg):
        request = TaskRequest(
            task_prompt="x", cwd="C:/tmp", model="haiku", label="t",
        )
        runner.start_task(request)
        # Should not raise — listener catches and swallows
        for tname in ["weird_a", "weird_b", "weird_c"]:
            _drive_event(handle, TaskEvent(
                kind=EventKind.TOOL_USE, tool_name=tname,
            ))


# ---------------------------------------------------------------------------
# Voice controller poll path
# ---------------------------------------------------------------------------


def test_voice_controller_polls_canonical_abort_warning(tmp_path) -> None:
    """``CapabilityVoiceController.pending_canonical_abort`` returns the
    runner's queued narration once and clears."""
    from ultron.coding.voice import CapabilityVoiceController

    runner = MagicMock()
    runner.pop_canonical_abort_warning.return_value = "I'm stopping that task — XYZ."
    ctrl = CapabilityVoiceController(
        runner=runner, registry=MagicMock(), resolver=MagicMock(),
        sandbox_root=tmp_path / "sandbox",
    )
    msg = ctrl.pending_canonical_abort()
    assert msg == "I'm stopping that task — XYZ."
    runner.pop_canonical_abort_warning.assert_called_once()


def test_voice_controller_canonical_abort_swallows_runner_exception(tmp_path) -> None:
    from ultron.coding.voice import CapabilityVoiceController

    runner = MagicMock()
    runner.pop_canonical_abort_warning.side_effect = RuntimeError("boom")
    ctrl = CapabilityVoiceController(
        runner=runner, registry=MagicMock(), resolver=MagicMock(),
        sandbox_root=tmp_path / "sandbox",
    )
    msg = ctrl.pending_canonical_abort()
    assert msg is None
