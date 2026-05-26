"""Tests for the two-phase approval voice yes/no on close_window
(catalog 09 batch E).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import pytest

from ultron.openclaw_routing.intents import (
    RoutingIntent,
    RoutingIntentKind,
    WindowCloseConfirmationIntent,
    WindowCloseIntent,
)


@pytest.fixture
def fresh_approval_registry(monkeypatch):
    from ultron.safety.two_phase_approval import (
        ApprovalRegistry, set_approval_registry,
    )
    registry = ApprovalRegistry()
    monkeypatch.setattr(
        "ultron.safety.two_phase_approval.get_approval_registry",
        lambda: registry,
    )
    set_approval_registry(registry)
    yield registry
    set_approval_registry(None)


def _make_controller(monkeypatch):
    from ultron.coding.voice import CapabilityVoiceController

    return CapabilityVoiceController(
        runner=MagicMock(),
        registry=MagicMock(),
        resolver=MagicMock(),
        coordinator=MagicMock(),
    )


@dataclass
class _FakeCloseResult:
    success: bool
    voice_message: str
    error: Optional[str] = None
    suspected_unsaved: bool = False


# ---------------------------------------------------------------------------
# Suspected-unsaved triggers approval
# ---------------------------------------------------------------------------


def test_close_with_suspected_unsaved_registers_approval(
    monkeypatch, fresh_approval_registry,
):
    """When the first close attempt reports suspected_unsaved=True,
    the controller registers a two-phase approval and speaks the
    prompt INSTEAD OF force-closing."""
    monkeypatch.setattr(
        "ultron.desktop.voice.handle_window_close",
        lambda intent: _FakeCloseResult(
            success=False,
            voice_message="VS Code may have unsaved work.",
            suspected_unsaved=True,
        ),
    )
    controller = _make_controller(monkeypatch)

    routing_intent = RoutingIntent(
        kind=RoutingIntentKind.WINDOW_CLOSE,
        raw_text="close VS Code",
        window_close_intent=WindowCloseIntent(
            window_query="VS Code",
            raw_text="close VS Code",
        ),
    )
    response = controller._handle_window_close(routing_intent)
    assert response.handled is True
    assert "VS Code" in response.text
    assert "yes or no" in response.text.lower()
    # Approval registered.
    assert controller._pending_close_approval is not None
    assert controller._pending_close_approval["window_query"] == "VS Code"


def test_close_without_suspected_unsaved_no_approval(
    monkeypatch, fresh_approval_registry,
):
    """A clean graceful close path doesn't register approval."""
    monkeypatch.setattr(
        "ultron.desktop.voice.handle_window_close",
        lambda intent: _FakeCloseResult(
            success=True,
            voice_message="Closed Discord.",
            suspected_unsaved=False,
        ),
    )
    controller = _make_controller(monkeypatch)

    routing_intent = RoutingIntent(
        kind=RoutingIntentKind.WINDOW_CLOSE,
        raw_text="close Discord",
        window_close_intent=WindowCloseIntent(
            window_query="Discord",
            raw_text="close Discord",
        ),
    )
    response = controller._handle_window_close(routing_intent)
    assert response.handled is True
    assert response.text == "Closed Discord."
    assert controller._pending_close_approval is None


# ---------------------------------------------------------------------------
# Yes/no consumption
# ---------------------------------------------------------------------------


def test_yes_reply_force_closes(monkeypatch, fresh_approval_registry):
    """After the prompt is spoken, the user's 'yes' reply triggers
    a force close and clears the pending approval."""
    # First close: suspected_unsaved=True.
    monkeypatch.setattr(
        "ultron.desktop.voice.handle_window_close",
        lambda intent: _FakeCloseResult(
            success=False,
            voice_message="Notepad may have unsaved work.",
            suspected_unsaved=True,
        ),
    )

    # Force close called via the low-level close_window primitive.
    force_calls = []

    def _fake_close_window(*, partial_title, force, user_text):
        force_calls.append((partial_title, force, user_text))
        return _FakeCloseResult(
            success=True, voice_message="closed", suspected_unsaved=False,
        )

    monkeypatch.setattr(
        "ultron.desktop.windows.close_window", _fake_close_window,
    )
    controller = _make_controller(monkeypatch)

    # Trigger the approval.
    controller._handle_window_close(RoutingIntent(
        kind=RoutingIntentKind.WINDOW_CLOSE,
        raw_text="close Notepad",
        window_close_intent=WindowCloseIntent(
            window_query="Notepad", raw_text="close Notepad",
        ),
    ))
    assert controller._pending_close_approval is not None

    # Spoken 'yes' reply.
    conf_intent = RoutingIntent(
        kind=RoutingIntentKind.WINDOW_CLOSE_CONFIRMATION,
        raw_text="yes",
        window_close_confirmation_intent=WindowCloseConfirmationIntent(
            decision="yes", raw_text="yes",
        ),
    )
    response = controller._handle_window_close_confirmation(conf_intent)
    assert response.handled is True
    assert "Notepad" in response.text
    # Approval cleared.
    assert controller._pending_close_approval is None
    # force-close was called with force=True.
    assert len(force_calls) == 1
    title, force, _user_text = force_calls[0]
    assert title == "Notepad"
    assert force is True


def test_no_reply_aborts_close(monkeypatch, fresh_approval_registry):
    """The user's 'no' reply cancels the close without calling
    force_close."""
    monkeypatch.setattr(
        "ultron.desktop.voice.handle_window_close",
        lambda intent: _FakeCloseResult(
            success=False,
            voice_message="x may have unsaved work.",
            suspected_unsaved=True,
        ),
    )
    force_calls = []
    monkeypatch.setattr(
        "ultron.desktop.windows.close_window",
        lambda **kw: force_calls.append(kw) or _FakeCloseResult(
            success=True, voice_message="closed",
        ),
    )
    controller = _make_controller(monkeypatch)

    controller._handle_window_close(RoutingIntent(
        kind=RoutingIntentKind.WINDOW_CLOSE,
        raw_text="close x",
        window_close_intent=WindowCloseIntent(window_query="x", raw_text="x"),
    ))

    response = controller._handle_window_close_confirmation(RoutingIntent(
        kind=RoutingIntentKind.WINDOW_CLOSE_CONFIRMATION,
        raw_text="no",
        window_close_confirmation_intent=WindowCloseConfirmationIntent(
            decision="no", raw_text="no",
        ),
    ))
    assert response.handled is True
    assert "leaving" in response.text.lower() or "okay" in response.text.lower()
    assert controller._pending_close_approval is None
    # force_close NOT called.
    assert force_calls == []


def test_confirmation_without_pending_approval_acks_neutrally(
    monkeypatch, fresh_approval_registry,
):
    """A bare 'yes' / 'no' with no pending approval surfaces a
    neutral ack (controller-level handler doesn't trigger any
    side effects)."""
    controller = _make_controller(monkeypatch)
    response = controller._handle_window_close_confirmation(RoutingIntent(
        kind=RoutingIntentKind.WINDOW_CLOSE_CONFIRMATION,
        raw_text="yes",
        window_close_confirmation_intent=WindowCloseConfirmationIntent(
            decision="yes", raw_text="yes",
        ),
    ))
    assert response.handled is True
    assert response.text  # non-empty ack


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_force_close_failure_surfaces_voice_error(
    monkeypatch, fresh_approval_registry,
):
    """When the force-close itself fails after approval, the user
    hears the failure message."""
    monkeypatch.setattr(
        "ultron.desktop.voice.handle_window_close",
        lambda intent: _FakeCloseResult(
            success=False,
            voice_message="x may have unsaved work.",
            suspected_unsaved=True,
        ),
    )
    monkeypatch.setattr(
        "ultron.desktop.windows.close_window",
        lambda **kw: _FakeCloseResult(
            success=False,
            voice_message="",
            error="access denied",
        ),
    )
    controller = _make_controller(monkeypatch)

    controller._handle_window_close(RoutingIntent(
        kind=RoutingIntentKind.WINDOW_CLOSE,
        raw_text="close x",
        window_close_intent=WindowCloseIntent(window_query="x"),
    ))
    response = controller._handle_window_close_confirmation(RoutingIntent(
        kind=RoutingIntentKind.WINDOW_CLOSE_CONFIRMATION,
        raw_text="yes",
        window_close_confirmation_intent=WindowCloseConfirmationIntent(
            decision="yes",
        ),
    ))
    assert response.handled is True
    assert "failed" in response.text.lower() or "couldn" in response.text.lower()


def test_force_close_exception_swallowed(
    monkeypatch, fresh_approval_registry,
):
    monkeypatch.setattr(
        "ultron.desktop.voice.handle_window_close",
        lambda intent: _FakeCloseResult(
            success=False, voice_message="x", suspected_unsaved=True,
        ),
    )

    def _boom(**kw):
        raise RuntimeError("display gone")
    monkeypatch.setattr("ultron.desktop.windows.close_window", _boom)
    controller = _make_controller(monkeypatch)

    controller._handle_window_close(RoutingIntent(
        kind=RoutingIntentKind.WINDOW_CLOSE,
        raw_text="close x",
        window_close_intent=WindowCloseIntent(window_query="x"),
    ))
    response = controller._handle_window_close_confirmation(RoutingIntent(
        kind=RoutingIntentKind.WINDOW_CLOSE_CONFIRMATION,
        raw_text="yes",
        window_close_confirmation_intent=WindowCloseConfirmationIntent(
            decision="yes",
        ),
    ))
    assert response.handled is True
    assert "couldn" in response.text.lower()


def test_second_close_supersedes_first(
    monkeypatch, fresh_approval_registry,
):
    """When the user fires a second close before answering the first,
    the latest one supersedes."""
    monkeypatch.setattr(
        "ultron.desktop.voice.handle_window_close",
        lambda intent: _FakeCloseResult(
            success=False,
            voice_message=f"{intent.window_query} may have unsaved.",
            suspected_unsaved=True,
        ),
    )
    controller = _make_controller(monkeypatch)

    controller._handle_window_close(RoutingIntent(
        kind=RoutingIntentKind.WINDOW_CLOSE,
        raw_text="close A",
        window_close_intent=WindowCloseIntent(window_query="A"),
    ))
    first_approval_id = controller._pending_close_approval["approval_id"]

    controller._handle_window_close(RoutingIntent(
        kind=RoutingIntentKind.WINDOW_CLOSE,
        raw_text="close B",
        window_close_intent=WindowCloseIntent(window_query="B"),
    ))
    second_approval_id = controller._pending_close_approval["approval_id"]

    assert first_approval_id != second_approval_id
    assert controller._pending_close_approval["window_query"] == "B"


def test_approval_registry_unavailable_falls_through(
    monkeypatch,
):
    """When the safety.two_phase_approval registry can't be imported,
    the controller degrades to the legacy synchronous-close path
    (returns the original voice_message)."""
    # Force the import to fail by replacing the module-level
    # get_approval_registry with one that raises.
    def _boom():
        raise RuntimeError("registry gone")

    monkeypatch.setattr(
        "ultron.safety.two_phase_approval.get_approval_registry", _boom,
    )
    monkeypatch.setattr(
        "ultron.desktop.voice.handle_window_close",
        lambda intent: _FakeCloseResult(
            success=False,
            voice_message="x may have unsaved.",
            suspected_unsaved=True,
        ),
    )
    controller = _make_controller(monkeypatch)

    response = controller._handle_window_close(RoutingIntent(
        kind=RoutingIntentKind.WINDOW_CLOSE,
        raw_text="close x",
        window_close_intent=WindowCloseIntent(window_query="x"),
    ))
    # The controller surfaces the original voice_message (legacy path).
    assert response.handled is True
    assert "x may have unsaved" in response.text
    assert controller._pending_close_approval is None
