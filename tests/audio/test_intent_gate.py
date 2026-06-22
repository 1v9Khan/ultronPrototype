"""Tests for the Ultron 1.0 always-listening 3-way (4-class) intent gate (src/kenning/audio/intent_gate.py).

Hermetic: uses relay cases caught by the strict matcher / complete-tactical-callout (no embedder
sidecar needed) and name-agnostic addressing NO-rules. Validates the cost-asymmetric, fail-closed
classification + the ASR pre-reject + the 8B band escalation (with a stub llm).
"""
import pytest

from kenning.audio import intent_gate as ig
from kenning.audio.intent_gate import Scenario


@pytest.mark.parametrize("text", [
    "tell my team to rush B",
    "sova hit 84 on A main",
    "two on A site one rotating",
])
def test_relay_to_team(text):
    v = ig.classify_scenario(text)
    assert v.scenario is Scenario.RELAY_TO_TEAM, (text, v)


@pytest.mark.parametrize("text", [
    "flavor off",
    "no flavor",
    "thinking mode on",
    "switch to the GPU",
    "stop",
    "ultron stop",
])
def test_command_local(text):
    v = ig.classify_scenario(text)
    assert v.scenario is Scenario.COMMAND_LOCAL, (text, v)


def test_private_reply_with_wake():
    v = ig.classify_scenario("ultron, what map is this")
    assert v.scenario is Scenario.PRIVATE_REPLY


def test_private_reply_factual_question_no_wake():
    # A clear factual question addressed to the assistant (addressing rule YES >= tau).
    v = ig.classify_scenario("what time is it right now")
    assert v.scenario is Scenario.PRIVATE_REPLY


@pytest.mark.parametrize("text", [
    "hey mom how are you doing today",   # phone opener -> NO
    "oh shit",                            # interjection -> NO
    "I'm talking to him right now",       # third-party narrative -> NO
])
def test_ignore_addressing_no(text):
    v = ig.classify_scenario(text)
    assert v.scenario is Scenario.IGNORE, (text, v)


def test_asr_pre_reject_no_speech():
    v = ig.classify_scenario("let's go team", no_speech_prob=0.9)
    assert v.scenario is Scenario.IGNORE and "no_speech" in v.reason


def test_asr_pre_reject_low_logprob():
    v = ig.classify_scenario("garbled audio here", avg_logprob=-2.5)
    assert v.scenario is Scenario.IGNORE and "avg_logprob" in v.reason


def test_undecided_is_failclosed_ignore_with_llm_flag():
    # An ambiguous statement with no relay/command/addressing signal -> fail-closed IGNORE, needs_llm.
    v = ig.classify_scenario("the rotations feel pretty clean this map")
    assert v.scenario is Scenario.IGNORE
    assert v.needs_llm is True


def test_empty():
    assert ig.classify_scenario("").scenario is Scenario.IGNORE
    assert ig.classify_scenario("   ").scenario is Scenario.IGNORE


class _StubLLM:
    def __init__(self, reply):
        self._reply = reply

    def generate_stream(self, *a, **k):
        return [self._reply]


def test_resolve_with_llm_private():
    v = ig.classify_scenario("the rotations feel pretty clean this map")  # needs_llm
    out = ig.resolve_with_llm(v, "the rotations feel pretty clean this map", _StubLLM("PRIVATE"))
    assert out.scenario is Scenario.PRIVATE_REPLY


def test_resolve_with_llm_failclosed_on_garbage():
    v = ig.classify_scenario("the rotations feel pretty clean this map")
    out = ig.resolve_with_llm(v, "...", _StubLLM("uhh I think maybe"))
    assert out.scenario is Scenario.IGNORE  # non-PRIVATE token -> fail closed


def test_resolve_with_llm_noop_when_not_needed():
    v = ig.classify_scenario("tell my team to rush B")   # RELAY, needs_llm False
    out = ig.resolve_with_llm(v, "tell my team to rush B", _StubLLM("PRIVATE"))
    assert out.scenario is Scenario.RELAY_TO_TEAM        # unchanged


# --- friend-chatter filter (2026-06-21): reaction openers IGNORE'd without the 8B ---


@pytest.mark.parametrize("text", [
    "Yeah, I can.",
    "It's okay, it's okay.",
    "nice shot dude",
    "sure thing",
    "lol no way",
    "oh damn",
])
def test_reaction_openers_ignored_without_8b(text):
    v = ig.classify_scenario(text, seconds_since_response=5.0)
    assert v.scenario is Scenario.IGNORE, (text, v)
    assert v.needs_llm is False, (text, v)  # dropped cheaply, no 8B spend


@pytest.mark.parametrize("text", [
    "Ultron, what is their economy?",
    "machine, mute yourself",
    "hey ai are you there",
])
def test_addressed_line_not_reaction_filtered(text):
    # A line that names Ultron must NOT be swallowed by the reaction filter.
    v = ig.classify_scenario(text, seconds_since_response=5.0)
    assert "reaction opener" not in v.reason, (text, v)


class _StubGateLLM:
    def __init__(self, token):
        self._token = token

    def generate_stream(self, *a, **k):
        return iter([self._token])


@pytest.mark.parametrize("token,scenario,conf", [
    ("IGNORE", Scenario.IGNORE, 0.75),
    ("PRIVATE", Scenario.PRIVATE_REPLY, 0.65),
    ("PRIVATELY yours", Scenario.IGNORE, 0.75),   # not exactly PRIVATE -> fail-closed
    ("Sure, PRIVATE", Scenario.IGNORE, 0.75),     # leading non-PRIVATE token -> fail-closed
])
def test_resolve_with_llm_fail_closed_exact_private(token, scenario, conf):
    base = ig.ScenarioVerdict(Scenario.IGNORE, 0.55, "undecided", needs_llm=True)
    v = ig.resolve_with_llm(base, "some ambiguous line", _StubGateLLM(token))
    assert v.scenario is scenario, (token, v)
    assert abs(v.confidence - conf) < 1e-6, (token, v)


# --- opinion-with-location-word guard (2026-06-21): "that's not even that long" ---


@pytest.mark.parametrize("text", [
    "that's not even that long",
    "that's pretty long honestly",
    "it was so long",
    "i think that's too long",
])
def test_opinion_with_location_word_not_relayed(text: str) -> None:
    # A conversational opinion that merely contains a location word ("long") must
    # NOT be classified RELAY_TO_TEAM (the live 'long' false-positive).
    v = ig.classify_scenario(text, seconds_since_response=5.0)
    assert v.scenario is not Scenario.RELAY_TO_TEAM, (text, v)


@pytest.mark.parametrize("text", [
    "they are pushing long",
    "one long",
    "tell my team to rush B",
])
def test_real_callout_still_relays(text: str) -> None:
    # Strong relay signals (complete tactical callout / strict matcher) are
    # unaffected by the opinion guard -- these must still relay.
    v = ig.classify_scenario(text, seconds_since_response=5.0)
    assert v.scenario is Scenario.RELAY_TO_TEAM, (text, v)
