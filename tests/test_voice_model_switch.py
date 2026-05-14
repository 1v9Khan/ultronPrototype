"""4B optimization plan — voice controller MODEL_SWITCH dispatch tests.

Verifies that ``CapabilityVoiceController.handle_capability_intent``
routes MODEL_SWITCH intents to ``llm_engine.reload_for_preset`` and
shapes the result into a sensible VoiceResponse. Mocks the engine so
no GPU is needed.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ultron.coding.voice import CapabilityVoiceController, VoiceResponse
from ultron.openclaw_routing.classifier import classify_routing
from ultron.openclaw_routing.intents import (
    ModelSwitchIntent,
    RoutingIntent,
    RoutingIntentKind,
)


def _make_controller(llm_engine=None, tmp_sandbox: Path = None) -> CapabilityVoiceController:
    if tmp_sandbox is None:
        tmp_sandbox = Path("test_sandbox_voice_model_switch")
        tmp_sandbox.mkdir(parents=True, exist_ok=True)
    return CapabilityVoiceController(
        runner=MagicMock(),
        registry=MagicMock(),
        resolver=MagicMock(),
        sandbox_root=tmp_sandbox,
        coordinator=None,
        llm_engine=llm_engine,
    )


def _model_switch_intent(target: str, raw: str = "switch to 4B") -> RoutingIntent:
    return RoutingIntent(
        kind=RoutingIntentKind.MODEL_SWITCH,
        raw_text=raw,
        confidence=0.95,
        source="rule",
        reason="model-switch pattern matched",
        model_switch_intent=ModelSwitchIntent(target_preset=target, raw_text=raw),
    )


# ---------------------------------------------------------------------------
# Engine wired — happy path
# ---------------------------------------------------------------------------


def test_model_switch_calls_reload_and_speaks_success(tmp_path: Path) -> None:
    engine = MagicMock()
    engine.reload_for_preset.return_value = (True, "loaded qwen3.5-4b")
    ctrl = _make_controller(llm_engine=engine, tmp_sandbox=tmp_path)
    intent = _model_switch_intent("qwen3.5-4b", "switch to 4B")

    response = ctrl.handle_capability_intent(intent)

    assert isinstance(response, VoiceResponse)
    assert response.handled is True
    assert response.text == "Switched to the 4B."
    engine.reload_for_preset.assert_called_once_with("qwen3.5-4b")


def test_model_switch_to_9b_speaks_correct_label(tmp_path: Path) -> None:
    engine = MagicMock()
    engine.reload_for_preset.return_value = (True, "loaded qwen3.5-9b")
    ctrl = _make_controller(llm_engine=engine, tmp_sandbox=tmp_path)
    intent = _model_switch_intent("qwen3.5-9b", "switch to 9B")

    response = ctrl.handle_capability_intent(intent)
    assert response.text == "Switched to the 9B."


def test_model_switch_already_on_target_says_already(tmp_path: Path) -> None:
    """Idempotent — engine returns (True, 'already on ...')."""
    engine = MagicMock()
    engine.reload_for_preset.return_value = (True, "already on qwen3.5-4b")
    ctrl = _make_controller(llm_engine=engine, tmp_sandbox=tmp_path)
    intent = _model_switch_intent("qwen3.5-4b")

    response = ctrl.handle_capability_intent(intent)
    assert response.text == "I'm already running the 4B."


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_model_switch_engine_failure_speaks_reason(tmp_path: Path) -> None:
    engine = MagicMock()
    engine.reload_for_preset.return_value = (False, "GGUF not found")
    ctrl = _make_controller(llm_engine=engine, tmp_sandbox=tmp_path)
    intent = _model_switch_intent("qwen3.5-4b")

    response = ctrl.handle_capability_intent(intent)
    assert "couldn't switch to the 4B" in response.text
    assert "GGUF not found" in response.text


def test_model_switch_no_engine_says_misconfigured(tmp_path: Path) -> None:
    """When llm_engine isn't wired (e.g. tests, partial init), the
    handler must NOT crash — it returns a clear voice error."""
    ctrl = _make_controller(llm_engine=None, tmp_sandbox=tmp_path)
    intent = _model_switch_intent("qwen3.5-4b")

    response = ctrl.handle_capability_intent(intent)
    assert "can't switch models" in response.text or "isn't wired" in response.text
    assert response.handled is True


def test_model_switch_missing_intent_payload_speaks_clarification(tmp_path: Path) -> None:
    """A MODEL_SWITCH kind with no model_switch_intent payload is a
    classifier bug — handler still returns a clean voice error."""
    engine = MagicMock()
    ctrl = _make_controller(llm_engine=engine, tmp_sandbox=tmp_path)
    intent = RoutingIntent(
        kind=RoutingIntentKind.MODEL_SWITCH,
        raw_text="switch to something",
        model_switch_intent=None,
    )

    response = ctrl.handle_capability_intent(intent)
    assert "couldn't tell which model" in response.text
    engine.reload_for_preset.assert_not_called()


# ---------------------------------------------------------------------------
# End-to-end through classifier — utterance → controller → reload call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("utterance,expected_target", [
    # 2026-05-14: "switch to 4B" now resolves to josiefied-qwen3-4b
    # (the abliterated variant; new default). "9B" stays plain qwen3.5-9b.
    # "8B" added as a switch target for the abliterated 8B swap-back.
    ("switch to the 4B", "josiefied-qwen3-4b"),
    ("use the 9B model", "qwen3.5-9b"),
    ("load 4B", "josiefied-qwen3-4b"),
    ("swap over to the nine B", "qwen3.5-9b"),
    ("switch to model 4B", "josiefied-qwen3-4b"),
    ("switch to the 8B", "josiefied-qwen3-8b"),
    ("use the 8B model", "josiefied-qwen3-8b"),
])
def test_classifier_then_voice_controller_end_to_end(
    utterance: str, expected_target: str, tmp_path: Path,
) -> None:
    engine = MagicMock()
    engine.reload_for_preset.return_value = (True, f"loaded {expected_target}")
    ctrl = _make_controller(llm_engine=engine, tmp_sandbox=tmp_path)

    routing = classify_routing(utterance)
    assert routing.kind == RoutingIntentKind.MODEL_SWITCH

    response = ctrl.handle_capability_intent(routing)
    engine.reload_for_preset.assert_called_once_with(expected_target)
    assert response.handled is True
    assert "Switched to" in response.text


# ---------------------------------------------------------------------------
# Conversational utterances must NOT trigger model_switch
# ---------------------------------------------------------------------------


def test_conversational_utterance_does_not_call_reload(tmp_path: Path) -> None:
    engine = MagicMock()
    ctrl = _make_controller(llm_engine=engine, tmp_sandbox=tmp_path)
    routing = classify_routing("the 4B is faster than the 9B")
    response = ctrl.handle_capability_intent(routing)
    # CONVERSATIONAL → returns None (orchestrator handles via LLM normally)
    assert response is None
    engine.reload_for_preset.assert_not_called()
