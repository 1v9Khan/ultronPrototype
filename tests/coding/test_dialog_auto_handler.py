"""Tests for the dialog auto-handler wiring in CodingTaskRunner
(catalog 08 + 09 batch B).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ultron.bus import publish, reset_bus_for_testing
from ultron.bus.events import DialogAppearedEvent


@pytest.fixture(autouse=True)
def fresh_bus():
    reset_bus_for_testing()
    yield
    reset_bus_for_testing()


def _build_runner(monkeypatch):
    from ultron.coding.runner import CodingTaskRunner

    runner = CodingTaskRunner.__new__(CodingTaskRunner)
    # Minimal init -- we only test the dialog-handler subset.
    import threading
    runner._handle = None
    runner._handle_lock = threading.Lock()
    runner._dialog_lock = threading.Lock()
    runner._pending_dialog_narrations = []
    runner._dialog_unsubscribe = None
    runner._log_path = None
    runner._session_audit_log_path = None

    # Stub out _session_audit so we don't write to disk.
    runner._session_audit = lambda *a, **kw: None
    return runner


# ---------------------------------------------------------------------------
# Dialog narration queue
# ---------------------------------------------------------------------------


def test_pop_dialog_narration_returns_none_when_empty(monkeypatch):
    runner = _build_runner(monkeypatch)
    assert runner.pop_dialog_narration() is None


def test_pop_dialog_narration_drains_oldest_first(monkeypatch):
    runner = _build_runner(monkeypatch)
    runner._pending_dialog_narrations.extend(["first", "second", "third"])
    assert runner.pop_dialog_narration() == "first"
    assert runner.pop_dialog_narration() == "second"
    assert runner.pop_dialog_narration() == "third"
    assert runner.pop_dialog_narration() is None


# ---------------------------------------------------------------------------
# Bus subscription
# ---------------------------------------------------------------------------


def test_dialog_handler_queues_narration_on_event(monkeypatch):
    runner = _build_runner(monkeypatch)

    # Attach the handler with a stub TaskHandle (only used for the
    # COMPLETE-listener wiring we don't exercise here).
    handle = MagicMock()
    handle.task_id.return_value = "task-1"

    listener = runner._attach_dialog_auto_handler(handle)
    assert listener is not None  # subscription created

    # Publish a dialog event on the real bus.
    publish(DialogAppearedEvent, {
        "hwnd": 101,
        "title": "Save As",
        "class_name": "#32770",
        "matched_by": "class",
        "process_name": "notepad.exe",
        "monitor_index": 1,
        "first_seen_at": 1234567.0,
    })

    line = runner.pop_dialog_narration()
    assert line is not None
    assert "Save As" in line
    assert "notepad.exe" in line


def test_dialog_handler_falls_back_for_untitled_dialog(monkeypatch):
    runner = _build_runner(monkeypatch)
    handle = MagicMock()
    handle.task_id.return_value = "task-2"
    runner._attach_dialog_auto_handler(handle)

    publish(DialogAppearedEvent, {
        "hwnd": 202,
        "title": "",  # empty title triggers fallback phrasing
        "class_name": "Dialog",
        "matched_by": "class",
        "process_name": "installer.exe",
        "monitor_index": 0,
        "first_seen_at": 0.0,
    })

    line = runner.pop_dialog_narration()
    assert line is not None
    assert "installer.exe" in line
    assert "yes" in line.lower()


def test_dialog_handler_disabled_via_config(monkeypatch):
    """When coding.dialog_auto_handler.enabled=False, no listener
    is registered."""
    from ultron.config import get_config
    cfg = get_config().coding
    monkeypatch.setattr(cfg.dialog_auto_handler, "enabled", False)

    runner = _build_runner(monkeypatch)
    handle = MagicMock()
    listener = runner._attach_dialog_auto_handler(handle)
    assert listener is None
    assert runner._dialog_unsubscribe is None

    publish(DialogAppearedEvent, {
        "hwnd": 303,
        "title": "Disabled",
        "class_name": "Dialog",
        "matched_by": "class",
        "process_name": "test.exe",
        "monitor_index": 0,
        "first_seen_at": 0.0,
    })
    # No narration queued because no subscriber was registered.
    assert runner.pop_dialog_narration() is None


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


def test_dialog_teardown_unsubscribes_on_complete(monkeypatch):
    from ultron.coding.bridge import EventKind, TaskEvent

    runner = _build_runner(monkeypatch)
    handle = MagicMock()
    handle.task_id.return_value = "task-x"
    listener = runner._attach_dialog_auto_handler(handle)
    assert runner._dialog_unsubscribe is not None

    # Simulate a COMPLETE event -- the runner's listener should
    # unsubscribe.
    listener(TaskEvent(kind=EventKind.COMPLETE))
    assert runner._dialog_unsubscribe is None

    # Subsequent dialog events should not surface to this runner's queue.
    publish(DialogAppearedEvent, {
        "hwnd": 404,
        "title": "After Complete",
        "class_name": "Dialog",
        "matched_by": "class",
        "process_name": "x.exe",
        "monitor_index": 0,
        "first_seen_at": 0.0,
    })
    assert runner.pop_dialog_narration() is None


def test_dialog_teardown_ignores_non_complete_events(monkeypatch):
    from ultron.coding.bridge import EventKind, TaskEvent

    runner = _build_runner(monkeypatch)
    handle = MagicMock()
    handle.task_id.return_value = "task-x"
    listener = runner._attach_dialog_auto_handler(handle)

    # TEXT, TOOL_USE, FILE_CHANGE events must NOT tear down the
    # subscription -- only COMPLETE does.
    listener(TaskEvent(kind=EventKind.TEXT, text="hi"))
    listener(TaskEvent(kind=EventKind.TOOL_USE, tool_name="Read"))
    assert runner._dialog_unsubscribe is not None


def test_new_task_tears_down_old_subscription(monkeypatch):
    """When a runner starts a new task while a prior subscription is
    still alive (rare but possible), the new attach replaces the old."""
    runner = _build_runner(monkeypatch)

    handle1 = MagicMock()
    handle1.task_id.return_value = "task-1"
    listener1 = runner._attach_dialog_auto_handler(handle1)
    first_unsub = runner._dialog_unsubscribe

    handle2 = MagicMock()
    handle2.task_id.return_value = "task-2"
    listener2 = runner._attach_dialog_auto_handler(handle2)
    second_unsub = runner._dialog_unsubscribe

    assert first_unsub is not None
    assert second_unsub is not None
    assert first_unsub is not second_unsub  # different subscription


# ---------------------------------------------------------------------------
# Audit / observability
# ---------------------------------------------------------------------------


def test_dialog_audit_called_with_event_fields(monkeypatch):
    runner = _build_runner(monkeypatch)
    captured = []

    def _audit(event, **kw):
        captured.append((event, kw))

    runner._session_audit = _audit
    handle = MagicMock()
    handle.task_id.return_value = "task-audit"
    runner._attach_dialog_auto_handler(handle)

    publish(DialogAppearedEvent, {
        "hwnd": 505,
        "title": "Overwrite?",
        "class_name": "Confirm",
        "matched_by": "class",
        "process_name": "audit.exe",
        "monitor_index": 1,
        "first_seen_at": 0.0,
    })

    assert any(
        event == "dialog_appeared" and kw["title"] == "Overwrite?"
        for (event, kw) in captured
    )
