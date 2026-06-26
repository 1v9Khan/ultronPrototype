"""The social / conversational LLM-novel path + the teammates-fighting fix (2026-06-20).

Non-tactical responses (identity, encouragement, calm, flame, criticize, compliment,
defiance) are AUTHORED by the 8B (novel) when the u1.0 LLM route is ON, with the
curated pools supplied only as STYLE exemplars; they fall back to the curated pool on
ANY LLM failure, and are byte-identical canned when u1 is OFF. Plus the "my <agent>
and <agent> are fighting/arguing" de-escalation that previously fell through silently.
"""
from types import SimpleNamespace

import pytest

from kenning.audio.relay_speech import (
    _social_llm_line, match_relay_command, build_relay_line,
    set_u1_llm_route_enabled, DEFAULT_ENCOURAGEMENT_LINES,
)
from kenning.audio.ultron_prompt import (
    build_social_prompt, SOCIAL_SYSTEM, _SOCIAL_SYSTEM_FOR,
)


class _StubLLM:
    def __init__(self, text):
        self.text = text
    def generate_stream(self, user, **kw):
        return iter([self.text])


class _RaisingLLM:
    def generate_stream(self, user, **kw):
        raise RuntimeError("boom")


@pytest.fixture(autouse=True)
def _u1_off_around_each_test():
    set_u1_llm_route_enabled(False)
    yield
    set_u1_llm_route_enabled(False)


def _cmd(addressee="team", context="", raw_text=None):
    return SimpleNamespace(addressee=addressee, context=context, raw_text=raw_text)


# --- build_social_prompt -----------------------------------------------------

def test_social_prompt_is_conversational_and_forbids_repeating():
    pr = build_social_prompt(
        "identity", addressee="team", context="are you a soundboard",
        exemplars=("I am Ultron.", "A soundboard repeats; I evolve."),
    )
    assert pr.enable_thinking is False
    # identity now has its OWN dedicated per-pool template, distinct from the general one.
    assert pr.system == _SOCIAL_SYSTEM_FOR["identity"]
    assert pr.system != SOCIAL_SYSTEM
    assert "Ultron" in pr.system
    assert "machine" in pr.system.lower()          # cold-machine persona anchor
    assert "soundboard" in pr.system.lower()       # the identity behaviour names + rebuts the accusation
    assert "repeat" in pr.system.lower()           # never-repeat-the-style-examples guard
    assert "soundboard" in pr.user                 # the situation/context (teammate's words)
    assert "do NOT repeat" in pr.user              # the style-exemplar guard


def test_social_prompt_named_addressee_strips_name_placeholder():
    pr = build_social_prompt(
        "criticize", addressee="Reyna", target="Reyna",
        exemplars=("{name}, you whiffed that.",),
    )
    assert "Reyna" in pr.user
    assert "{name}" not in pr.user                 # placeholder stripped from exemplars


# --- _social_llm_line robustness --------------------------------------------

def test_social_off_returns_canned():
    set_u1_llm_route_enabled(False)
    out = _social_llm_line(_cmd(), "encouragement", DEFAULT_ENCOURAGEMENT_LINES,
                           max_chars=360, llm=_StubLLM("NOVEL"), canned="CANNED")
    assert out == "CANNED"


def test_social_on_returns_novel_line():
    set_u1_llm_route_enabled(True)
    out = _social_llm_line(
        _cmd(), "encouragement", DEFAULT_ENCOURAGEMENT_LINES, max_chars=360,
        llm=_StubLLM("Steel yourselves; the round is already mine."), canned="CANNED")
    assert out == "Steel yourselves; the round is already mine."


@pytest.mark.parametrize("llm", [
    _StubLLM("   "),          # empty / whitespace output
    _RaisingLLM(),            # the LLM raised
    None,                     # no LLM available at all
])
def test_social_on_failure_falls_back_to_canned(llm):
    set_u1_llm_route_enabled(True)
    out = _social_llm_line(_cmd(), "encouragement", DEFAULT_ENCOURAGEMENT_LINES,
                           max_chars=360, llm=llm, generate_fn=None, canned="CANNED")
    assert out == "CANNED"


def test_social_on_no_canned_uses_pool():
    set_u1_llm_route_enabled(True)
    out = _social_llm_line(_cmd(), "encouragement", DEFAULT_ENCOURAGEMENT_LINES,
                           max_chars=360, llm=_RaisingLLM())   # no `canned=` -> pool
    assert out in DEFAULT_ENCOURAGEMENT_LINES


# --- end-to-end via build_relay_line ----------------------------------------

def _relay(text, u1, llm):
    set_u1_llm_route_enabled(u1)
    cmd = match_relay_command(text)
    assert cmd is not None, text
    return build_relay_line(cmd, llm, rephrase=True, max_chars=360,
                            recent_lines=[], generate_fn=None)


def test_identity_off_canned_on_novel():
    novel = "I am Ultron. A soundboard echoes; I do not."
    off = _relay("Sage asked if you're a soundboard, respond.", False, None)
    on = _relay("Sage asked if you're a soundboard, respond.", True, _StubLLM(novel))
    # The novel line is spoken, with the accuser's NAME enforced as the opener
    # (2026-06-26 user: identity replies must address the accuser by name).
    assert on == "Sage, " + novel
    assert off != on            # OFF is a canned pool line, not the novel one


# --- the teammates-fighting bug ---------------------------------------------

@pytest.mark.parametrize("text", [
    "my yoru and sage are fighting",
    "my yoru and sage are arguing",
    "my reyna and jett are toxic",
    "our sova and breach keep arguing",
    "my yoru and sage are at each other's throats",
])
def test_teammates_fighting_routes_to_calm(text):
    cmd = match_relay_command(text)
    assert cmd is not None, text
    assert cmd.directive == "calm"


@pytest.mark.parametrize("text", [
    "my sova and jett are fighting for mid",   # TACTICAL -> never a de-escalation
    "my yoru and sage are pushing A",
])
def test_tactical_pair_is_not_a_calm_down(text):
    cmd = match_relay_command(text)
    assert cmd is None or cmd.directive != "calm"


# --- 2026-06-26 streamer persona direction: own-the-AI + shut-up defiance -----

def test_ai_accusation_owns_the_word_and_survives_guard():
    """'are you an AI' -> Ultron OWNS it (yes, an AI, and the step past you). The
    owning-AI LLM line must NOT be dropped by the meta-leak guard (the social path
    used allow_self_ai=False, which rejected 'I am an AI'); identity now uses the
    RELAXED guard so the owning line is spoken."""
    owning = "Killjoy, I am an AI, and the next step past you."
    out = _relay("my teammate asked if you are an AI, respond", True, _StubLLM(owning))
    assert out is not None
    assert "AI" in out                       # the owning-AI line is SPOKEN, not dropped
    assert out == owning


def test_ai_accusation_canned_fallback_is_the_own_it_pool():
    """With no LLM, 'are you an AI' draws from the dedicated own-it `ai` pool (it
    affirms the word), NOT the reframe `bot` pool."""
    from kenning.audio._ultron_identity import IDENTITY_POOLS
    out = _relay("my teammate asked if you are an AI, respond", False, None)
    assert out in IDENTITY_POOLS["ai"]
    assert out not in IDENTITY_POOLS["bot"]


def test_bot_accusation_still_reframes_not_owns():
    """A bare 'bot' still routes to the reframe `bot` pool (a bot obeys; he is a
    mind) -- the AI split must not change the bot behaviour."""
    from kenning.audio._ultron_identity import IDENTITY_POOLS
    out = _relay("my teammate asked if you are a bot, respond", False, None)
    assert out in IDENTITY_POOLS["bot"]


def test_identity_prompt_ai_directive_says_own_it():
    """The identity system prompt now carries the OWN-IT exception for AI, and the
    AI accusation phrasing tells the model to affirm the word."""
    pr = build_social_prompt(
        "identity", addressee="Killjoy", context="are you an AI",
        accusation="ai", verbosity="low",
    )
    assert "OWN IT" in pr.system                 # the AI exception in the behaviour
    assert "OWN it" in pr.user                    # the accusation phrasing affirms it
    # a bot accusation still tells it to rebut/reframe.
    pr_bot = build_social_prompt(
        "identity", addressee="Jett", context="are you a bot",
        accusation="bot", verbosity="low",
    )
    assert "no mere bot" in pr_bot.user


def test_shutup_routes_to_defiance_behaviour():
    """'shut up' is a DEMAND to silence Ultron -> authored with the DEFIANCE
    behaviour (rebuke the demand), not the generic reaction one that latched onto
    the 'silence' word and remarked on the teammate being quiet."""
    rebuke = "Reyna, you will not silence me. I speak until this round is won."
    out = _relay("Reyna told you to shut up, respond", True, _StubLLM(rebuke))
    assert out == rebuke


def test_defiance_behaviour_rebukes_demand_not_quiet():
    """The defiance system prompt reads 'shut up' as a DEMAND to silence him that he
    rebukes, and forbids remarking on quiet/silence-as-a-thing."""
    sys = _SOCIAL_SYSTEM_FOR["defiance"]
    low = sys.lower()
    assert "demanding that you be silenced" in low
    assert "do not remark on the quiet" in low
