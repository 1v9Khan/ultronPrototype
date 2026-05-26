"""Tests for catalog 09 voice-intent wiring: ACTIVE_WINDOW_QUERY,
SEMANTIC_CLICK, WINDOW_CLOSE_CONFIRMATION.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ultron.openclaw_routing.classifier import classify_routing
from ultron.openclaw_routing.intents import (
    ActiveWindowQueryIntent,
    RoutingIntentKind,
    SemanticClickIntent,
    WindowCloseConfirmationIntent,
)


# ---------------------------------------------------------------------------
# ACTIVE_WINDOW_QUERY classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "utterance",
    [
        "what's my active window?",
        "what's my active window",
        "What is my current window?",
        "what window am I on?",
        "What window am I using right now?",
        "which window am I on?",
        "Which window is active?",
        "Name my active window.",
        "name my current window",
        "Tell me the title of my active window.",
        "what's my foreground window?",
        "what's the foreground window",
    ],
)
def test_active_window_query_classifies(utterance):
    result = classify_routing(utterance)
    assert result.kind is RoutingIntentKind.ACTIVE_WINDOW_QUERY
    assert result.active_window_query_intent is not None
    assert result.active_window_query_intent.raw_text == utterance


@pytest.mark.parametrize(
    "utterance",
    [
        "what's on my screen?",  # SCREEN_CONTEXT_QUERY, broader
        "explain what I'm looking at",  # SCREEN_CONTEXT_QUERY
        "hey ultron",  # CONVERSATIONAL
        "open Chrome",  # APP_LAUNCH
    ],
)
def test_active_window_query_does_not_overshadow_other_intents(utterance):
    result = classify_routing(utterance)
    assert result.kind is not RoutingIntentKind.ACTIVE_WINDOW_QUERY


# ---------------------------------------------------------------------------
# SEMANTIC_CLICK classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "utterance,expected_name",
    [
        ("click the Submit button", "Submit"),
        ("press the OK button", "OK"),
        ("Tap on the Sign In link", "Sign In"),
        ("activate the File menu", "File"),
        ("hit the Cancel button", "Cancel"),
        ("click the Send Message button", "Send Message"),
        ("press the Save button", "Save"),
        ("click the OK button", "OK"),
        ("activate the Edit menu", "Edit"),
    ],
)
def test_semantic_click_classifies(utterance, expected_name):
    result = classify_routing(utterance)
    assert result.kind is RoutingIntentKind.SEMANTIC_CLICK
    intent = result.semantic_click_intent
    assert intent is not None
    assert expected_name.lower() in intent.element_name.lower()


def test_semantic_click_requires_explicit_control_noun():
    """Bare "click Save" / "click X" without a trailing button/menu/link
    noun falls through to other routes (BROWSER_AUTOMATION etc.).
    Catalog 09 wiring deliberately requires the explicit noun to
    avoid hijacking the existing browser-interact behaviour."""
    result = classify_routing("Click Save")
    assert result.kind is not RoutingIntentKind.SEMANTIC_CLICK


@pytest.mark.parametrize(
    "utterance",
    [
        "click here",  # generic referent
        "click that",  # generic referent
        "click anywhere",  # generic referent
        "press it",  # generic referent
        "tap that",  # generic referent
    ],
)
def test_semantic_click_rejects_generic_referents(utterance):
    result = classify_routing(utterance)
    # Generic referents fall through to the next branch (typically
    # CODE_TASK / CONVERSATIONAL).
    assert result.kind is not RoutingIntentKind.SEMANTIC_CLICK


def test_semantic_click_picks_up_button_control_type():
    result = classify_routing("click the Submit button")
    assert result.kind is RoutingIntentKind.SEMANTIC_CLICK
    assert result.semantic_click_intent.control_type == "Button"


def test_semantic_click_picks_up_menu_control_type():
    result = classify_routing("activate the File menu")
    assert result.kind is RoutingIntentKind.SEMANTIC_CLICK
    assert result.semantic_click_intent.control_type == "MenuItem"


def test_semantic_click_picks_up_window_scope():
    result = classify_routing("click the OK button in the Save dialog window")
    assert result.kind is RoutingIntentKind.SEMANTIC_CLICK
    assert "Save dialog" in result.semantic_click_intent.window_title


# ---------------------------------------------------------------------------
# WINDOW_CLOSE_CONFIRMATION classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "utterance",
    [
        "yes",
        "Yes",
        "yes.",
        "yeah",
        "yep",
        "yup",
        "Sure",
        "Confirm",
        "confirm it",
        "do it",
        "go ahead",
        "Proceed.",
        "go for it",
    ],
)
def test_window_close_confirmation_yes(utterance):
    result = classify_routing(utterance)
    # Note: yes/no replies during a pending clarification flow get
    # different handling (has_pending_clarification skips this rule).
    # In a fresh state (default args), the bare yes/no should match
    # WINDOW_CLOSE_CONFIRMATION.
    assert result.kind is RoutingIntentKind.WINDOW_CLOSE_CONFIRMATION
    assert result.window_close_confirmation_intent.decision == "yes"


@pytest.mark.parametrize(
    "utterance",
    [
        "no",
        "No",
        "no.",
        "nope",
        "nah",
        "Cancel",
        "cancel it",
        "stop",
        "abort",
        "don't",
        "Never mind",
        "nvm",
        "keep it open",
        "leave it open",
    ],
)
def test_window_close_confirmation_no(utterance):
    result = classify_routing(utterance)
    assert result.kind is RoutingIntentKind.WINDOW_CLOSE_CONFIRMATION
    assert result.window_close_confirmation_intent.decision == "no"


def test_window_close_confirmation_skipped_during_pending_clarification():
    """During a coordinator clarification, bare 'yes' must NOT hijack
    the response -- it routes to CLARIFICATION_RESPONSE per the
    existing precedence."""
    result = classify_routing("yes", has_pending_clarification=True)
    assert result.kind is not RoutingIntentKind.WINDOW_CLOSE_CONFIRMATION


def test_yes_with_content_does_not_match_confirmation():
    """'yes, please send the email' isn't a bare reply -- it has
    content that other intents may want."""
    # The yes-only regex requires the utterance to be ONLY yes (+
    # optional punctuation), so longer sentences fall through.
    result = classify_routing("yes please send the email")
    assert result.kind is not RoutingIntentKind.WINDOW_CLOSE_CONFIRMATION


# ---------------------------------------------------------------------------
# Voice handler dispatch
# ---------------------------------------------------------------------------


def test_handle_active_window_query_returns_title(monkeypatch):
    from ultron.coding.voice import CapabilityVoiceController
    from ultron.openclaw_routing.intents import RoutingIntent

    monkeypatch.setattr(
        "ultron.desktop.windows.get_active_window_title",
        lambda: "Visual Studio Code",
    )
    controller = CapabilityVoiceController(
        runner=MagicMock(),
        registry=MagicMock(),
        resolver=MagicMock(),
        coordinator=MagicMock(),
    )
    routing_intent = RoutingIntent(
        kind=RoutingIntentKind.ACTIVE_WINDOW_QUERY,
        raw_text="what's my active window",
        active_window_query_intent=ActiveWindowQueryIntent(
            raw_text="what's my active window",
        ),
    )
    response = controller._handle_active_window_query(routing_intent)
    assert response.handled is True
    assert "Visual Studio Code" in response.text


def test_handle_active_window_query_empty_title(monkeypatch):
    from ultron.coding.voice import CapabilityVoiceController
    from ultron.openclaw_routing.intents import RoutingIntent

    monkeypatch.setattr(
        "ultron.desktop.windows.get_active_window_title",
        lambda: None,
    )
    controller = CapabilityVoiceController(
        runner=MagicMock(),
        registry=MagicMock(),
        resolver=MagicMock(),
        coordinator=MagicMock(),
    )
    routing_intent = RoutingIntent(
        kind=RoutingIntentKind.ACTIVE_WINDOW_QUERY,
        raw_text="x",
        active_window_query_intent=ActiveWindowQueryIntent(raw_text="x"),
    )
    response = controller._handle_active_window_query(routing_intent)
    assert response.handled is True
    assert "no window" in response.text.lower()


def test_handle_active_window_query_exception(monkeypatch):
    from ultron.coding.voice import CapabilityVoiceController
    from ultron.openclaw_routing.intents import RoutingIntent

    def _boom():
        raise RuntimeError("pywin32 broke")

    monkeypatch.setattr(
        "ultron.desktop.windows.get_active_window_title", _boom,
    )
    controller = CapabilityVoiceController(
        runner=MagicMock(),
        registry=MagicMock(),
        resolver=MagicMock(),
        coordinator=MagicMock(),
    )
    routing_intent = RoutingIntent(
        kind=RoutingIntentKind.ACTIVE_WINDOW_QUERY,
        raw_text="x",
        active_window_query_intent=ActiveWindowQueryIntent(raw_text="x"),
    )
    response = controller._handle_active_window_query(routing_intent)
    assert response.handled is True
    assert "couldn't read" in response.text.lower() or "couldn't" in response.text.lower()


def test_handle_semantic_click_success(monkeypatch):
    from ultron.coding.voice import CapabilityVoiceController
    from ultron.desktop.element_click import ClickResult
    from ultron.openclaw_routing.intents import RoutingIntent

    fake_result = ClickResult(
        success=True,
        element_name="Submit",
        window_title="My Form",
        control_type="Button",
        center=(100, 200),
        method="invoke",
        candidates=1,
        is_exact=True,
    )
    monkeypatch.setattr(
        "ultron.desktop.element_click.click_element_by_name",
        lambda **kw: fake_result,
    )
    controller = CapabilityVoiceController(
        runner=MagicMock(),
        registry=MagicMock(),
        resolver=MagicMock(),
        coordinator=MagicMock(),
    )
    routing_intent = RoutingIntent(
        kind=RoutingIntentKind.SEMANTIC_CLICK,
        raw_text="click the Submit button",
        semantic_click_intent=SemanticClickIntent(
            element_name="Submit",
            control_type="Button",
            raw_text="click the Submit button",
        ),
    )
    response = controller._handle_semantic_click(routing_intent)
    assert response.handled is True
    assert "Submit" in response.text
    assert "My Form" in response.text


def test_handle_semantic_click_not_found(monkeypatch):
    from ultron.coding.voice import CapabilityVoiceController
    from ultron.desktop.element_click import ClickResult
    from ultron.openclaw_routing.intents import RoutingIntent

    fake_result = ClickResult(
        success=False,
        element_name="",
        window_title="",
        control_type="",
        center=(0, 0),
        method="",
        candidates=0,
        is_exact=False,
        error="no candidate found",
    )
    monkeypatch.setattr(
        "ultron.desktop.element_click.click_element_by_name",
        lambda **kw: fake_result,
    )
    controller = CapabilityVoiceController(
        runner=MagicMock(),
        registry=MagicMock(),
        resolver=MagicMock(),
        coordinator=MagicMock(),
    )
    routing_intent = RoutingIntent(
        kind=RoutingIntentKind.SEMANTIC_CLICK,
        raw_text="click the Submit button",
        semantic_click_intent=SemanticClickIntent(
            element_name="Submit",
            raw_text="click the Submit button",
        ),
    )
    response = controller._handle_semantic_click(routing_intent)
    assert response.handled is True
    assert "couldn't find" in response.text.lower() or "Submit" in response.text


def test_handle_semantic_click_safety_blocked(monkeypatch):
    from ultron.coding.voice import CapabilityVoiceController
    from ultron.desktop.element_click import ClickResult
    from ultron.openclaw_routing.intents import RoutingIntent

    fake_result = ClickResult(
        success=False,
        element_name="Buy",
        window_title="Payment",
        control_type="Button",
        center=(0, 0),
        method="",
        candidates=1,
        is_exact=True,
        error="safety: Cap-3 explicit intent missing",
    )
    monkeypatch.setattr(
        "ultron.desktop.element_click.click_element_by_name",
        lambda **kw: fake_result,
    )
    controller = CapabilityVoiceController(
        runner=MagicMock(),
        registry=MagicMock(),
        resolver=MagicMock(),
        coordinator=MagicMock(),
    )
    routing_intent = RoutingIntent(
        kind=RoutingIntentKind.SEMANTIC_CLICK,
        raw_text="click the Buy button",
        semantic_click_intent=SemanticClickIntent(
            element_name="Buy",
            raw_text="click the Buy button",
        ),
    )
    response = controller._handle_semantic_click(routing_intent)
    assert response.handled is True
    assert "held off" in response.text.lower() or "safety" in response.text.lower() or "Buy" in response.text


def test_handle_semantic_click_missing_name():
    from ultron.coding.voice import CapabilityVoiceController
    from ultron.openclaw_routing.intents import RoutingIntent

    controller = CapabilityVoiceController(
        runner=MagicMock(),
        registry=MagicMock(),
        resolver=MagicMock(),
        coordinator=MagicMock(),
    )
    routing_intent = RoutingIntent(
        kind=RoutingIntentKind.SEMANTIC_CLICK,
        raw_text="x",
        semantic_click_intent=None,
    )
    response = controller._handle_semantic_click(routing_intent)
    assert response.handled is True
    assert "didn't catch" in response.text.lower()


def test_handle_window_close_confirmation_yes():
    from ultron.coding.voice import CapabilityVoiceController
    from ultron.openclaw_routing.intents import RoutingIntent

    controller = CapabilityVoiceController(
        runner=MagicMock(),
        registry=MagicMock(),
        resolver=MagicMock(),
        coordinator=MagicMock(),
    )
    routing_intent = RoutingIntent(
        kind=RoutingIntentKind.WINDOW_CLOSE_CONFIRMATION,
        raw_text="yes",
        window_close_confirmation_intent=WindowCloseConfirmationIntent(
            decision="yes", raw_text="yes",
        ),
    )
    response = controller._handle_window_close_confirmation(routing_intent)
    assert response.handled is True
    # The bare controller (no orchestrator intercept) should just ack.
    assert response.text


def test_handle_window_close_confirmation_no():
    from ultron.coding.voice import CapabilityVoiceController
    from ultron.openclaw_routing.intents import RoutingIntent

    controller = CapabilityVoiceController(
        runner=MagicMock(),
        registry=MagicMock(),
        resolver=MagicMock(),
        coordinator=MagicMock(),
    )
    routing_intent = RoutingIntent(
        kind=RoutingIntentKind.WINDOW_CLOSE_CONFIRMATION,
        raw_text="no",
        window_close_confirmation_intent=WindowCloseConfirmationIntent(
            decision="no", raw_text="no",
        ),
    )
    response = controller._handle_window_close_confirmation(routing_intent)
    assert response.handled is True
    assert response.text


def test_capability_dispatch_routes_active_window_query(monkeypatch):
    from ultron.coding.voice import CapabilityVoiceController
    from ultron.openclaw_routing.intents import RoutingIntent

    monkeypatch.setattr(
        "ultron.desktop.windows.get_active_window_title",
        lambda: "Chrome",
    )
    controller = CapabilityVoiceController(
        runner=MagicMock(),
        registry=MagicMock(),
        resolver=MagicMock(),
        coordinator=MagicMock(),
    )
    routing_intent = RoutingIntent(
        kind=RoutingIntentKind.ACTIVE_WINDOW_QUERY,
        raw_text="what's my active window",
        active_window_query_intent=ActiveWindowQueryIntent(
            raw_text="what's my active window",
        ),
    )
    response = controller.handle_capability_intent(routing_intent)
    assert response is not None
    assert response.handled is True
    assert "Chrome" in response.text


def test_capability_dispatch_routes_semantic_click(monkeypatch):
    from ultron.coding.voice import CapabilityVoiceController
    from ultron.desktop.element_click import ClickResult
    from ultron.openclaw_routing.intents import RoutingIntent

    fake = ClickResult(
        success=True, element_name="OK", window_title="",
        control_type="Button", center=(0, 0), method="invoke",
        candidates=1, is_exact=True,
    )
    monkeypatch.setattr(
        "ultron.desktop.element_click.click_element_by_name",
        lambda **kw: fake,
    )
    controller = CapabilityVoiceController(
        runner=MagicMock(),
        registry=MagicMock(),
        resolver=MagicMock(),
        coordinator=MagicMock(),
    )
    routing_intent = RoutingIntent(
        kind=RoutingIntentKind.SEMANTIC_CLICK,
        raw_text="click OK",
        semantic_click_intent=SemanticClickIntent(
            element_name="OK", raw_text="click OK",
        ),
    )
    response = controller.handle_capability_intent(routing_intent)
    assert response is not None
    assert response.handled is True
