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


def test_unnamed_factual_question_ignored_in_always_listening():
    # 2026-06-22: an un-named factual question is NO LONGER a private reply. In
    # always-listening the player asks teammates questions constantly; without a
    # name/wake signal it is almost never meant for Ultron -> IGNORE (no LLM spend).
    v = ig.classify_scenario("what time is it right now")
    assert v.scenario is Scenario.IGNORE, v
    assert v.needs_llm is False, v


def test_named_factual_question_is_private():
    # The SAME question that names Ultron IS a private reply.
    v = ig.classify_scenario("ultron, what time is it right now")
    assert v.scenario is Scenario.PRIVATE_REPLY, v


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


def test_unnamed_ambiguous_statement_ignored_no_llm():
    # 2026-06-22: an un-named ambiguous statement is dropped CHEAPLY -> IGNORE with
    # needs_llm False. The LLM band no longer fires on un-named chatter (it leaked
    # 'Follow orders.' -> PRIVATE and cost a forward-pass on every ambiguous line).
    v = ig.classify_scenario("the rotations feel pretty clean this map")
    assert v.scenario is Scenario.IGNORE
    assert v.needs_llm is False


def test_empty():
    assert ig.classify_scenario("").scenario is Scenario.IGNORE
    assert ig.classify_scenario("   ").scenario is Scenario.IGNORE


class _StubLLM:
    def __init__(self, reply):
        self._reply = reply

    def generate_stream(self, *a, **k):
        return [self._reply]


def test_resolve_with_llm_private():
    # resolve_with_llm is RETAINED for callers; tested in isolation since
    # classify_scenario no longer sets needs_llm (the gate decides on the cheap layers).
    base = ig.ScenarioVerdict(Scenario.IGNORE, 0.55, "undecided", needs_llm=True)
    out = ig.resolve_with_llm(base, "the rotations feel pretty clean this map", _StubLLM("PRIVATE"))
    assert out.scenario is Scenario.PRIVATE_REPLY


def test_resolve_with_llm_failclosed_on_garbage():
    base = ig.ScenarioVerdict(Scenario.IGNORE, 0.55, "undecided", needs_llm=True)
    out = ig.resolve_with_llm(base, "...", _StubLLM("uhh I think maybe"))
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


# --- responded-when-not-addressed regression (live session bu5fh4lc8, 2026-06-22) ---


@pytest.mark.parametrize("text", [
    "What? That doesn't sound right. I think you might be mistaken.",
    "No.",
    "What is that brimstone doing?",
    "Follow orders.",
    "Respond.",
    "He was kind of clean with the Marshall.",
])
def test_unaddressed_conversation_not_private(text: str) -> None:
    # The live session false-fired PRIVATE_REPLY on these un-named conversational
    # lines (the player talking to teammates / themselves). Without an Ultron
    # name/wake signal they must never become a private reply.
    v = ig.classify_scenario(text, seconds_since_response=5.0)
    assert v.scenario is not Scenario.PRIVATE_REPLY, (text, v)


@pytest.mark.parametrize("text", [
    "Explain the math, Ultron.",
    "Ultron, what is their economy?",
    "kenning, what map is this",
])
def test_named_question_still_private(text: str) -> None:
    # A line that NAMES Ultron is genuinely addressed -> private reply (unchanged).
    v = ig.classify_scenario(text, seconds_since_response=5.0)
    assert v.scenario is Scenario.PRIVATE_REPLY, (text, v)


@pytest.mark.parametrize("text", [
    "this machine is so slow",          # their PC, not Ultron
    "reload the machine gun",           # a weapon
    "that robot in the corner is creepy",
])
def test_common_noun_machine_robot_not_private(text: str) -> None:
    # 'machine' / 'robot' are common nouns that also name Ultron -- as a PRIVATE_REPLY
    # TRIGGER anywhere they false-fire on ordinary speech (2026-06-22 review). Only the
    # unambiguous names (ultron / kenning / hey ai / the ai) gate a private reply.
    v = ig.classify_scenario(text, seconds_since_response=5.0)
    assert v.scenario is not Scenario.PRIVATE_REPLY, (text, v)


# --- 2026-06-22: mangled team-lead mishears relay (the "tell myself" ignore) ---


def test_gate_relays_mangled_team_lead_mishears():
    # "tell my team nice try" mis-heard "Tell myself a nice try." was IGNORED by
    # the always-listening gate (the relay signal missed the mangled lead). The
    # gate now canonicalizes an EXISTING mangled team-directed lead before the
    # strict matcher -- WITHOUT inventing a lead for a bare callout.
    assert ig._relay_signal("Tell myself a nice try.", None) == 0.95
    assert ig._relay_signal("tell my self good job", None) == 0.95
    assert ig._relay_signal("Call my team to rotate B", None) == 0.95
    # a genuine self-instruction is NOT relayed (the "to <verb>" guard)
    assert ig._relay_signal("tell myself to calm down", None) is None
    assert ig._relay_signal("tell myself to relax", None) is None
    # banter still never false-relays (the reason the gate was tightened)
    assert ig._relay_signal("the rotations feel clean", None) is None
    assert ig._relay_signal("nice shot dude", None) is None


def test_canonicalize_relay_lead_self_mishear():
    from kenning.audio.command_normalizer import canonicalize_relay_lead
    assert canonicalize_relay_lead("tell myself a nice try") == "tell my team nice try"
    assert canonicalize_relay_lead("tell my self good job") == "tell my team good job"
    # guarded: a self-instruction + a mid-utterance "telling myself" are left alone
    assert canonicalize_relay_lead("tell myself to calm down") == "tell myself to calm down"
    assert "tell my team" not in canonicalize_relay_lead("I keep telling myself to relax")
