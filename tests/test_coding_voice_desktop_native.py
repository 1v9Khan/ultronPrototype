"""Tests for the Phase 8 native desktop handlers in CapabilityVoiceController.

Covers:

- _handle_app_launch: dispatch to ultron.desktop.voice + record routing
  outcome + use preferences for default placement.
- _handle_screen_context_query: capture + LLM call + voice response.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ultron.coding.voice import CapabilityVoiceController, VoiceResponse
from ultron.openclaw_routing.intents import (
    AppLaunchIntent,
    RoutingIntent,
    RoutingIntentKind,
    ScreenContextIntent,
)


# ---------------------------------------------------------------------------
# Build a CapabilityVoiceController with minimal mocks for these tests.
# ---------------------------------------------------------------------------


def _build_controller(*, llm_engine=None, tmp_path=None) -> CapabilityVoiceController:
    """Build a controller that survives the constructor without touching
    real Claude / OpenClaw / project registry state.
    """
    runner = MagicMock()
    runner.has_active_task.return_value = False
    registry = MagicMock()
    resolver = MagicMock()
    return CapabilityVoiceController(
        runner=runner,
        registry=registry,
        resolver=resolver,
        sandbox_root=(tmp_path or "/tmp/test_sandbox"),
        coordinator=None,
        llm_engine=llm_engine,
        openclaw_bridge=None,
        gaming_mode_manager=None,
    )


def _routing_intent_for_app_launch(intent: AppLaunchIntent) -> RoutingIntent:
    return RoutingIntent(
        kind=RoutingIntentKind.APP_LAUNCH,
        raw_text=intent.raw_text,
        confidence=0.9,
        source="rule",
        reason="app-launch pattern matched",
        app_launch_intent=intent,
    )


def _routing_intent_for_screen_context(intent: ScreenContextIntent) -> RoutingIntent:
    return RoutingIntent(
        kind=RoutingIntentKind.SCREEN_CONTEXT_QUERY,
        raw_text=intent.raw_text or "what's on my screen",
        confidence=0.9,
        source="rule",
        reason="screen-context query pattern matched",
        screen_context_intent=intent,
    )


# ---------------------------------------------------------------------------
# _handle_app_launch
# ---------------------------------------------------------------------------


def test_handle_app_launch_missing_intent_returns_voice_error(tmp_path):
    controller = _build_controller(tmp_path=tmp_path)
    routing = RoutingIntent(
        kind=RoutingIntentKind.APP_LAUNCH,
        raw_text="open chrome",
        app_launch_intent=None,
    )
    response = controller._handle_app_launch(routing)
    assert isinstance(response, VoiceResponse)
    assert response.handled is True
    assert "didn't catch" in response.text.lower()


def test_handle_app_launch_dispatches_to_voice_handler(monkeypatch, tmp_path):
    from ultron.desktop.voice import AppLaunchVoiceResult

    captured = []

    def fake_handler(intent):
        captured.append(intent)
        return AppLaunchVoiceResult(
            success=True,
            voice_message="Opening chrome on monitor 2.",
            app_name="chrome",
            monitor_index=1,
            hwnd=42,
        )

    monkeypatch.setattr(
        "ultron.desktop.voice.handle_app_launch", fake_handler,
    )
    controller = _build_controller(tmp_path=tmp_path)
    intent = AppLaunchIntent(
        app_name="chrome",
        url="https://youtube.com",
        monitor_index=1,
        raw_text="open youtube on my second monitor",
    )
    response = controller._handle_app_launch(
        _routing_intent_for_app_launch(intent),
    )
    assert response.text == "Opening chrome on monitor 2."
    assert response.handled is True
    assert len(captured) == 1
    assert captured[0].monitor_index == 1


def test_handle_app_launch_failure_returns_voice_message(monkeypatch, tmp_path):
    from ultron.desktop.voice import AppLaunchVoiceResult

    monkeypatch.setattr(
        "ultron.desktop.voice.handle_app_launch",
        lambda intent: AppLaunchVoiceResult(
            success=False,
            voice_message="I couldn't open chrome. Chrome is not installed on this system",
            app_name="chrome",
            error="Chrome is not installed on this system",
        ),
    )
    controller = _build_controller(tmp_path=tmp_path)
    intent = AppLaunchIntent(app_name="chrome", raw_text="open chrome")
    response = controller._handle_app_launch(
        _routing_intent_for_app_launch(intent),
    )
    assert response.handled is True
    assert "couldn't open" in response.text.lower()


def test_handle_app_launch_exception_fails_open(monkeypatch, tmp_path):
    def boom(intent):
        raise RuntimeError("simulated desktop module failure")

    monkeypatch.setattr("ultron.desktop.voice.handle_app_launch", boom)
    controller = _build_controller(tmp_path=tmp_path)
    intent = AppLaunchIntent(app_name="chrome", raw_text="open chrome")
    response = controller._handle_app_launch(
        _routing_intent_for_app_launch(intent),
    )
    assert response.handled is True
    assert "isn't available" in response.text.lower() or "not available" in response.text.lower()


def test_handle_app_launch_uses_preference_default_monitor(monkeypatch, tmp_path):
    """When the utterance has no explicit monitor target AND there's a
    matching prior preference, the prior monitor should be used.
    """
    from ultron.desktop.preferences import DesktopPreference
    from ultron.desktop.voice import AppLaunchVoiceResult

    prior = DesktopPreference(
        user_phrase="open chrome",
        app_name="chrome",
        monitor_index=2,
        maximize=True,
        success=True,
        timestamp=1234567890.0,
    )
    monkeypatch.setattr(
        "ultron.desktop.preferences.find_preference_for_phrase",
        lambda *a, **kw: prior,
    )
    captured = []

    def fake_handler(intent):
        captured.append(intent)
        return AppLaunchVoiceResult(
            success=True, voice_message="ok",
            app_name="chrome", monitor_index=2,
        )

    monkeypatch.setattr(
        "ultron.desktop.voice.handle_app_launch", fake_handler,
    )
    controller = _build_controller(tmp_path=tmp_path)
    intent = AppLaunchIntent(app_name="chrome", raw_text="open chrome")
    controller._handle_app_launch(_routing_intent_for_app_launch(intent))
    # The intent passed to the handler should have prior's monitor + flags.
    assert captured[0].monitor_index == 2
    assert captured[0].maximize is True


def test_handle_app_launch_explicit_monitor_overrides_preference(monkeypatch, tmp_path):
    """When the utterance has an explicit monitor target, ignore the
    prior preference's monitor."""
    from ultron.desktop.preferences import DesktopPreference
    from ultron.desktop.voice import AppLaunchVoiceResult

    prior = DesktopPreference(
        user_phrase="open chrome",
        app_name="chrome",
        monitor_index=2,
        success=True,
        timestamp=1234567890.0,
    )
    pref_lookups = []

    def find_pref(*a, **kw):
        pref_lookups.append(a)
        return prior

    monkeypatch.setattr(
        "ultron.desktop.preferences.find_preference_for_phrase", find_pref,
    )
    captured = []

    def fake_handler(intent):
        captured.append(intent)
        return AppLaunchVoiceResult(
            success=True, voice_message="ok", app_name="chrome",
            monitor_index=intent.monitor_index,
        )

    monkeypatch.setattr(
        "ultron.desktop.voice.handle_app_launch", fake_handler,
    )
    controller = _build_controller(tmp_path=tmp_path)
    intent = AppLaunchIntent(
        app_name="chrome", monitor_index=0,  # explicit
        raw_text="open chrome on monitor 1",
    )
    controller._handle_app_launch(_routing_intent_for_app_launch(intent))
    # Explicit monitor wins; preference lookup should NOT have fired.
    assert captured[0].monitor_index == 0
    assert pref_lookups == []  # short-circuit -- explicit target present


# ---------------------------------------------------------------------------
# _handle_screen_context_query
# ---------------------------------------------------------------------------


def test_handle_screen_context_missing_intent_returns_voice_error(tmp_path):
    controller = _build_controller(
        llm_engine=MagicMock(), tmp_path=tmp_path,
    )
    routing = RoutingIntent(
        kind=RoutingIntentKind.SCREEN_CONTEXT_QUERY,
        raw_text="explain this",
        screen_context_intent=None,
    )
    response = controller._handle_screen_context_query(routing)
    assert response.handled is True
    assert "didn't catch" in response.text.lower()


def test_handle_screen_context_no_llm_returns_voice_error(tmp_path):
    """Without a wired LLM engine, the handler must not attempt to
    generate a response."""
    controller = _build_controller(llm_engine=None, tmp_path=tmp_path)
    intent = ScreenContextIntent(question="what's on my screen", raw_text="x")
    response = controller._handle_screen_context_query(
        _routing_intent_for_screen_context(intent),
    )
    assert response.handled is True
    assert "not wired" in response.text.lower() or "language model" in response.text.lower()


def test_handle_screen_context_happy_path(monkeypatch, tmp_path):
    from ultron.desktop.voice import ScreenContextVoiceResult

    captured_prompts = []
    captured_kwargs = []

    def fake_handle(intent):
        return ScreenContextVoiceResult(
            success=True,
            injection_text="[Visual context: Cursor showing main.py]",
            elapsed_ms=42.0,
            used_vlm=True,
        )

    monkeypatch.setattr(
        "ultron.desktop.voice.handle_screen_context_query", fake_handle,
    )
    fake_llm = MagicMock()

    def _fake_generate(prompt, **kwargs):
        captured_prompts.append(prompt)
        captured_kwargs.append(kwargs)
        return "It's a Python file."

    fake_llm.generate.side_effect = _fake_generate

    controller = _build_controller(
        llm_engine=fake_llm, tmp_path=tmp_path,
    )
    intent = ScreenContextIntent(
        question="what is this", raw_text="what is this",
    )
    response = controller._handle_screen_context_query(
        _routing_intent_for_screen_context(intent),
    )
    assert response.handled is True
    assert response.text == "It's a Python file."
    # The augmented prompt should include both the screen context AND
    # the user's question.
    assert len(captured_prompts) == 1
    assert "Cursor showing main.py" in captured_prompts[0]
    assert "what is this" in captured_prompts[0]
    # 2026-05-14: brevity hint baked into the prompt so the response
    # stays 1-2 sentences instead of an essay.
    assert "1-2 short sentences" in captured_prompts[0]
    # 2026-05-14: thinking-mode disabled so the model doesn't burn
    # tokens on <think>...</think> chains AND there's no way for a
    # thought trace to leak to TTS even if the strip filter regressed.
    assert captured_kwargs[0].get("enable_thinking") is False


def test_handle_screen_context_snapshot_failure(monkeypatch, tmp_path):
    from ultron.desktop.voice import ScreenContextVoiceResult

    monkeypatch.setattr(
        "ultron.desktop.voice.handle_screen_context_query",
        lambda intent: ScreenContextVoiceResult(
            success=False, error="capture failed",
        ),
    )
    controller = _build_controller(
        llm_engine=MagicMock(), tmp_path=tmp_path,
    )
    intent = ScreenContextIntent(
        question="explain this", raw_text="explain this",
    )
    response = controller._handle_screen_context_query(
        _routing_intent_for_screen_context(intent),
    )
    assert response.handled is True
    assert "couldn't see" in response.text.lower()


def test_handle_screen_context_llm_exception_fails_open(monkeypatch, tmp_path):
    from ultron.desktop.voice import ScreenContextVoiceResult

    monkeypatch.setattr(
        "ultron.desktop.voice.handle_screen_context_query",
        lambda intent: ScreenContextVoiceResult(
            success=True,
            injection_text="[Visual context]",
            elapsed_ms=10.0,
        ),
    )
    fake_llm = MagicMock()
    fake_llm.generate.side_effect = RuntimeError("simulated LLM failure")
    controller = _build_controller(
        llm_engine=fake_llm, tmp_path=tmp_path,
    )
    intent = ScreenContextIntent(question="x", raw_text="x")
    response = controller._handle_screen_context_query(
        _routing_intent_for_screen_context(intent),
    )
    assert response.handled is True
    assert "can't put it into words" in response.text.lower()


# ---------------------------------------------------------------------------
# handle_capability_intent dispatch
# ---------------------------------------------------------------------------


def test_capability_dispatch_routes_app_launch_to_native(monkeypatch, tmp_path):
    from ultron.desktop.voice import AppLaunchVoiceResult

    monkeypatch.setattr(
        "ultron.desktop.voice.handle_app_launch",
        lambda intent: AppLaunchVoiceResult(
            success=True, voice_message="ok", app_name="chrome",
        ),
    )
    # The handler also calls find_preference_for_phrase; mock to None.
    monkeypatch.setattr(
        "ultron.desktop.preferences.find_preference_for_phrase",
        lambda *a, **kw: None,
    )
    controller = _build_controller(tmp_path=tmp_path)
    intent = AppLaunchIntent(
        app_name="chrome", raw_text="open chrome",
        monitor_index=None,
    )
    routing = _routing_intent_for_app_launch(intent)
    response = controller.handle_capability_intent(routing)
    assert response is not None
    assert response.text == "ok"


def test_capability_dispatch_routes_screen_context_to_native(monkeypatch, tmp_path):
    from ultron.desktop.voice import ScreenContextVoiceResult

    monkeypatch.setattr(
        "ultron.desktop.voice.handle_screen_context_query",
        lambda intent: ScreenContextVoiceResult(
            success=True,
            injection_text="[Visual]",
            elapsed_ms=5.0,
        ),
    )
    fake_llm = MagicMock()
    fake_llm.generate.return_value = "A Python file."
    controller = _build_controller(
        llm_engine=fake_llm, tmp_path=tmp_path,
    )
    intent = ScreenContextIntent(question="what is this", raw_text="what is this")
    routing = _routing_intent_for_screen_context(intent)
    response = controller.handle_capability_intent(routing)
    assert response is not None
    assert response.text == "A Python file."
