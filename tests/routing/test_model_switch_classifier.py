"""4B optimization plan — voice-driven MODEL_SWITCH classifier tests.

Verifies that the routing classifier maps "switch to 4B" / "use the 9B
model" / etc. to ``RoutingIntentKind.MODEL_SWITCH`` with the right
``target_preset``, while NOT misfiring on conversational utterances
that mention model names without a clear command verb.

The patterns must be both robust (Whisper homophones / spacing
variants — "four B", "for B", "4 B", "4-B") and conservative (false
positives mid-conversation = unwanted swap = bad UX).
"""
from __future__ import annotations

import pytest

from ultron.openclaw_routing.classifier import classify_routing
from ultron.openclaw_routing.intents import RoutingIntentKind


# ---------------------------------------------------------------------------
# Positive cases — must classify as MODEL_SWITCH
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    # Canonical "switch to" + identifier
    ("switch to 4B", "josiefied-qwen3-4b"),
    ("switch to 9B", "qwen3.5-9b"),
    ("switch to the 4B", "josiefied-qwen3-4b"),
    ("switch to the 9B", "qwen3.5-9b"),
    ("switch to the 4B model", "josiefied-qwen3-4b"),
    ("switch to the 9B model", "qwen3.5-9b"),
    ("Switch to 4B.", "josiefied-qwen3-4b"),
    ("Switch to 9b model", "qwen3.5-9b"),
    # Synonyms — swap / change / load / use / go to / move to
    ("swap to 4B", "josiefied-qwen3-4b"),
    ("swap to the 9B model", "qwen3.5-9b"),
    ("change to the 4B", "josiefied-qwen3-4b"),
    ("load the 9B", "qwen3.5-9b"),
    ("load 4B model", "josiefied-qwen3-4b"),
    ("use the 4B", "josiefied-qwen3-4b"),
    ("use the 9B model", "qwen3.5-9b"),
    ("go to 4B", "josiefied-qwen3-4b"),
    ("move to 9B", "qwen3.5-9b"),
    ("activate 4B", "josiefied-qwen3-4b"),
    ("engage the 9B", "qwen3.5-9b"),
    ("run the 4B model", "josiefied-qwen3-4b"),
    ("select 9B", "qwen3.5-9b"),
    # Whisper homophones / number words
    ("switch to four B", "josiefied-qwen3-4b"),
    ("switch to nine B", "qwen3.5-9b"),
    ("use the four B model", "josiefied-qwen3-4b"),
    ("switch to for B", "josiefied-qwen3-4b"),  # "for" homophone of "four"
    # Spacing / punctuation variants
    ("switch to 4 B", "josiefied-qwen3-4b"),
    ("switch to 9 b", "qwen3.5-9b"),
    ("switch to 4-B", "josiefied-qwen3-4b"),
    ("switch to 9-b model", "qwen3.5-9b"),
    # Verb variants with prepositions
    ("switch over to 4B", "josiefied-qwen3-4b"),
    ("switch over to the 9B", "qwen3.5-9b"),
    ("change over to 4B", "josiefied-qwen3-4b"),
    # 2026-05-14: noun BEFORE the token ("switch to model 4B").
    # Whisper transcribes the user's actual phrasing this way; the old
    # regex only allowed the noun AFTER the token, so this leaked to
    # the conversational LLM (which hallucinated "Model 4B engaged.").
    ("switch to model 4B", "josiefied-qwen3-4b"),
    ("switch to model 9B", "qwen3.5-9b"),
    ("switch to the model 4B", "josiefied-qwen3-4b"),
    ("change to model 4B", "josiefied-qwen3-4b"),
    ("use the model 4B", "josiefied-qwen3-4b"),
    ("load the model 9B", "qwen3.5-9b"),
    ("switch to llm 4B", "josiefied-qwen3-4b"),
    ("switch to preset 9B", "qwen3.5-9b"),
    ("switch to qwen 4B", "josiefied-qwen3-4b"),
    # 2026-05-14: 8B added as a switch target (swap-back from new 4B default).
    ("switch to 8B", "josiefied-qwen3-8b"),
    ("switch to the 8B", "josiefied-qwen3-8b"),
    ("switch to the 8B model", "josiefied-qwen3-8b"),
    ("switch to model 8B", "josiefied-qwen3-8b"),
    ("use the 8B model", "josiefied-qwen3-8b"),
    ("switch to eight B", "josiefied-qwen3-8b"),
    ("load the 8B", "josiefied-qwen3-8b"),
    ("activate 8B", "josiefied-qwen3-8b"),
    ("switch to 8 B", "josiefied-qwen3-8b"),
    ("switch to 8-B", "josiefied-qwen3-8b"),
])
def test_classify_routes_to_model_switch(text: str, expected: str) -> None:
    intent = classify_routing(text)
    assert intent.kind == RoutingIntentKind.MODEL_SWITCH, (
        f"Expected MODEL_SWITCH for {text!r}, got {intent.kind}"
    )
    assert intent.model_switch_intent is not None
    assert intent.model_switch_intent.target_preset == expected
    assert intent.model_switch_intent.raw_text == text
    assert intent.confidence >= 0.9
    assert intent.source == "rule"


# ---------------------------------------------------------------------------
# Negative cases — must NOT trigger MODEL_SWITCH
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    # No model identifier
    "switch to chrome",
    "switch on the lights",
    "switch to dark mode",
    "switch to the next song",
    # Model name mentioned without a command verb
    "the 4B should be faster than the 9B",
    "I think 4B will work fine",
    "the 9B model gave a wrong answer earlier",
    "tell me about the 4B model's architecture",
    "what's the difference between 4B and 9B",
    "9B is bigger than 4B",
    # Other categories
    "open hacker news",
    "send a message to bob",
    "delete the file at temp.txt",
    "what's the boiling point of water",
    "are you afraid of death",
    # Empty / whitespace
    "",
    "   ",
])
def test_classify_does_not_trigger_model_switch(text: str) -> None:
    intent = classify_routing(text)
    assert intent.kind != RoutingIntentKind.MODEL_SWITCH, (
        f"False MODEL_SWITCH on {text!r} (got {intent.kind})"
    )


# ---------------------------------------------------------------------------
# Mid-task safety — pending clarification must suppress MODEL_SWITCH
# so we don't tear down the LLM in the middle of an active dialogue
# turn.
# ---------------------------------------------------------------------------


def test_pending_clarification_suppresses_model_switch() -> None:
    intent = classify_routing(
        "switch to 4B", has_pending_clarification=True,
    )
    assert intent.kind != RoutingIntentKind.MODEL_SWITCH
    # Falls through to coding-classifier handling for the clarification
    # flow (or to CONVERSATIONAL if none matches).
    assert intent.source != "rule" or intent.kind != RoutingIntentKind.MODEL_SWITCH


def test_active_coding_task_does_not_block_model_switch() -> None:
    """An active coding task is fine to interrupt — the user asked for
    a model swap, the pipeline should honor it. Pending clarification
    is the ONLY suppressor (mid-dialogue safety)."""
    intent = classify_routing(
        "switch to the 4B model", has_active_coding_task=True,
    )
    assert intent.kind == RoutingIntentKind.MODEL_SWITCH
    assert intent.model_switch_intent.target_preset == "josiefied-qwen3-4b"


# ---------------------------------------------------------------------------
# Resolver helper — direct unit test
# ---------------------------------------------------------------------------


def test_resolve_model_switch_target_canonical() -> None:
    from ultron.openclaw_routing.classifier import _resolve_model_switch_target
    assert _resolve_model_switch_target("4B") == "josiefied-qwen3-4b"
    assert _resolve_model_switch_target("9B") == "qwen3.5-9b"


def test_resolve_model_switch_target_variants() -> None:
    from ultron.openclaw_routing.classifier import _resolve_model_switch_target
    assert _resolve_model_switch_target("four B") == "josiefied-qwen3-4b"
    assert _resolve_model_switch_target("for B") == "josiefied-qwen3-4b"
    assert _resolve_model_switch_target("nine B") == "qwen3.5-9b"
    assert _resolve_model_switch_target("9 b") == "qwen3.5-9b"
    assert _resolve_model_switch_target("4-B") == "josiefied-qwen3-4b"


def test_resolve_model_switch_target_unknown_raises() -> None:
    from ultron.openclaw_routing.classifier import _resolve_model_switch_target
    with pytest.raises(ValueError):
        _resolve_model_switch_target("XYZ")
