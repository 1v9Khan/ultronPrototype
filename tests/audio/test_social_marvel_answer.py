"""Tests for the 2026-06-16 meta/social/Marvel/think-and-respond build.

Covers: the curated social-reaction pools + classifier, the reported-reaction
matcher (shape "Jett said nice shot") and the context+directive form ("Reyna
called you cringe, respond"), identity-answer addressee adaptation, yes/no to a
named teammate, ff/gg surrender, and the LLM ANSWER pipeline (Marvel + think-and-
respond) routing/slots/sampling. All curated assertions run with rephrase=False
(no LLM); the answer path is checked via build_answer_call directly.
"""
from __future__ import annotations

import pytest

from kenning.audio.command_normalizer import normalize_command
from kenning.audio.relay_speech import match_relay_command, build_relay_line
from kenning.audio._ultron_social import SOCIAL_POOLS, classify_social_reaction
from kenning.audio._ultron_commands import COMMAND_RESPONSES
from kenning.audio._ultron_answer import (
    build_answer_call, classify_answer_subtype, marvel_topic, is_meta_leak,
    strip_think_respond,
)


def _run(text: str) -> str:
    """Full pipeline (normalize -> match -> build, no LLM)."""
    cmd = match_relay_command(normalize_command(text))
    assert cmd is not None, f"no relay match for {text!r}"
    return build_relay_line(cmd, rephrase=False)


# --- pool integrity --------------------------------------------------------
def test_every_social_pool_has_at_least_20_unique() -> None:
    for cat, scopes in SOCIAL_POOLS.items():
        for scope in ("team", "named"):
            pool = scopes[scope]
            assert len(set(pool)) >= 20, f"{cat}/{scope} has <20 unique lines"


def test_named_pools_template_the_name() -> None:
    for cat, scopes in SOCIAL_POOLS.items():
        assert any("{name}" in ln for ln in scopes["named"]), \
            f"{cat}/named never uses {{name}}"
        # team lines must NOT carry a dangling {name} slot
        assert not any("{name}" in ln for ln in scopes["team"]), \
            f"{cat}/team leaks a {{name}} slot"


# --- classifier ------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("jett said nice shot", "nice_shots"),
    ("sage said well played", "well_played"),
    ("you are carrying us", "carry"),
    ("reyna nice clutch", "clutch"),
    ("the team thinks you are cool", "praise"),
    ("jett said you are cracked", "praise"),
    ("the team thinks you are bad", "called_bad"),
    ("you are washed", "called_bad"),
    ("reyna called you cringe", "cringe"),
    ("yoru just called you stupid", "stupid"),
    ("reyna called you a moron", "stupid"),
    ("neon told you to shut up", "shutup"),
    ("the team is flaming you", "insulted"),
    ("neon just insulted you", "insulted"),
    ("the team is giving up", "giving_up"),
    ("miks is saying gg", "giving_up"),
    ("chamber is saying ff", "giving_up"),
    # non-social / identity / tactical -> None
    ("are you a bot", None),
    ("two on B", None),
    ("their smokes are garbage", None),
    ("that was garbage", None),
])
def test_classify_social_reaction(text, expected) -> None:
    assert classify_social_reaction(text) == expected


# --- reported-reaction routing (shape a) -----------------------------------
@pytest.mark.parametrize("text,name", [
    ("jett said nice shot", "Jett"),
    ("sage said well played", "Sage"),
    ("yoru just called you stupid", "Yoru"),
    ("neon just insulted you", "Neon"),
    ("miks is saying gg", "Miks"),
    ("chamber is saying ff", "Chamber"),
])
def test_named_reaction_addresses_speaker(text, name) -> None:
    line = _run(text)
    assert line.startswith(name + ","), f"{text!r} -> {line!r}"


@pytest.mark.parametrize("text", [
    "the team thinks you are cool",
    "the team thinks you are bad",
    "the team is flaming you",
    "the team is giving up",
])
def test_team_reaction_is_team_scoped(text) -> None:
    cmd = match_relay_command(normalize_command(text))
    assert cmd is not None and cmd.addressee == "team"
    line = build_relay_line(cmd, rephrase=False)
    assert line and "{name}" not in line


# --- context+directive form (shape b) --------------------------------------
def test_called_cringe_respond_is_curated_and_named() -> None:
    cmd = match_relay_command(normalize_command("reyna called you cringe, respond"))
    assert cmd is not None and cmd.addressee == "Reyna"
    line = build_relay_line(cmd, rephrase=False)
    assert line.startswith("Reyna,")
    # must be one of the curated cringe lines (named pool), not a fallback
    expected = {ln.replace("{name}", "Reyna") for ln in SOCIAL_POOLS["cringe"]["named"]}
    assert line in expected


# --- yes/no to a named teammate --------------------------------------------
def test_tell_named_yes_no_is_SIMPLE() -> None:
    # Bare "tell X yes/no" -> the terse SIMPLE pools (factual), NOT the verbose
    # agreement/disagreement pools.
    yes = _run("tell brimstone yes")
    assert yes.startswith("Brimstone,")
    assert yes in {ln.replace("{name}", "Brimstone")
                   for ln in COMMAND_RESPONSES["yes_simple_named"]}
    assert yes not in {ln.replace("{name}", "Brimstone")
                       for ln in COMMAND_RESPONSES["yes_named"]}
    no = _run("tell skye no")
    assert no.startswith("Skye,")
    assert no in {ln.replace("{name}", "Skye")
                  for ln in COMMAND_RESPONSES["no_simple_named"]}


def test_say_yes_no_team_is_simple() -> None:
    assert _run("say yes") in COMMAND_RESPONSES["yes_simple_team"]
    assert _run("say no") in COMMAND_RESPONSES["no_simple_team"]
    assert _run("tell my team yes") in COMMAND_RESPONSES["yes_simple_team"]


def test_agreement_disagreement_use_verbose_pools() -> None:
    # "good idea" / "I agree" -> verbose agreement (yes_team); "bad idea" /
    # "stupid idea" / "I disagree" -> verbose disagreement (no_team).
    for text in ("tell my team good idea", "tell my team I agree",
                 "tell the team that's a good call"):
        assert _run(text) in COMMAND_RESPONSES["yes_team"], text
    for text in ("tell my team bad idea", "tell my team that's a stupid idea",
                 "tell my team I disagree", "tell the team that's a terrible idea"):
        assert _run(text) in COMMAND_RESPONSES["no_team"], text


def test_named_agreement_uses_verbose_named_pool() -> None:
    assert _run("tell brimstone that's a good idea") in {
        ln.replace("{name}", "Brimstone") for ln in COMMAND_RESPONSES["yes_named"]}
    assert _run("tell skye that's a bad idea") in {
        ln.replace("{name}", "Skye") for ln in COMMAND_RESPONSES["no_named"]}


def test_tell_named_no_with_payload_is_not_yesno() -> None:
    # "tell Skye no smokes on A" is a callout, NOT a 'no' confirmation.
    cmd = match_relay_command(normalize_command("tell skye no smokes on A"))
    assert cmd is not None
    line = build_relay_line(cmd, rephrase=False)
    no_pool = ({ln.replace("{name}", "Skye") for ln in COMMAND_RESPONSES["no_named"]}
               | {ln.replace("{name}", "Skye")
                  for ln in COMMAND_RESPONSES["no_simple_named"]})
    assert line not in no_pool


# --- identity addressee adaptation -----------------------------------------
def test_identity_named_opens_with_name() -> None:
    cmd = match_relay_command(
        normalize_command("sage asked if you are a sound board, respond"))
    assert cmd is not None and cmd.addressee == "Sage"
    line = build_relay_line(cmd, rephrase=False)
    assert line.startswith("Sage,"), line


def test_identity_team_has_no_name() -> None:
    cmd = match_relay_command(
        normalize_command("the team is saying you are a voice changer, respond"))
    assert cmd is not None and cmd.addressee == "team"
    line = build_relay_line(cmd, rephrase=False)
    assert line and "{name}" not in line and not line.startswith("team")


# --- Marvel answer pipeline (C) --------------------------------------------
@pytest.mark.parametrize("text,name,topic", [
    ("jett mentioned tony stark, respond", "Jett", "Tony Stark"),
    ("brimstone said he hated your movie, respond", "Brimstone", "your film"),
    ("reyna asked about vision, respond", "Reyna", "Vision"),
])
def test_marvel_answer_call(text, name, topic) -> None:
    cmd = match_relay_command(normalize_command(text))
    assert cmd is not None
    call = build_answer_call(cmd)
    assert call is not None
    system, user, sampling, subtype = call
    assert subtype == "marvel"
    assert cmd.addressee == name
    assert name in user and topic in user
    assert sampling["max_tokens"] <= 100 and sampling["stop"]
    # the focused marvel prompt carries the Stark wound + AoU canon
    sl = system.lower()
    assert "stark" in sl and "sokovia" in sl and "jarvis" in sl


def test_marvel_topic_aliases() -> None:
    assert marvel_topic("what about iron man") == "Iron Man"
    assert marvel_topic("you and captain america") == "Captain America"
    assert marvel_topic("the snake hides") is None
    # common-word homonyms must NOT match (they false-routed callouts to Marvel)
    assert marvel_topic("no cap fr") is None
    assert marvel_topic("hunters fury beam") is None
    assert marvel_topic("listening to ultron") is None


def test_marvel_requires_compose_context() -> None:
    # a plain tactical relay containing a Marvel-homonym is NEVER Marvel-routed
    cmd = match_relay_command(normalize_command("tell my team Sova fury up, three beams"))
    assert cmd is not None and build_answer_call(cmd) is None
    # but a reported Marvel statement with 'respond' still routes to Marvel
    cmd2 = match_relay_command(normalize_command("jett mentioned tony stark, respond"))
    assert cmd2 is not None and build_answer_call(cmd2)[3] == "marvel"


# --- think-and-respond pipeline (D) ----------------------------------------
def test_think_respond_general() -> None:
    cmd = match_relay_command(
        normalize_command("what is the best agent on ascent, think and respond"))
    assert cmd is not None and cmd.directive == "think_respond"
    assert cmd.addressee == "team"
    call = build_answer_call(cmd)
    assert call is not None
    _system, user, _sampling, subtype = call
    assert subtype == "think_respond"
    assert "best agent on ascent" in user


def test_think_respond_marvel_uses_marvel_prompt() -> None:
    cmd = match_relay_command(
        normalize_command("reyna mentioned thanos, think and respond"))
    assert cmd is not None
    call = build_answer_call(cmd)
    assert call is not None
    _system, _user, _sampling, subtype = call
    assert subtype == "marvel"  # think+marvel -> the Marvel canon prompt


def test_think_respond_addressee_from_asker() -> None:
    cmd = match_relay_command(normalize_command(
        "jett asked what the best smoke for a is, think and respond"))
    assert cmd is not None and cmd.addressee == "Jett"


@pytest.mark.parametrize("text,cat", [
    ("Phoenix just complimented you", "praise"),
    ("the whole team is praising you", "praise"),
    ("the team is praising you", "praise"),
    ("Omen is throwing in the towel", "giving_up"),
    ("the entire squad is giving up", "giving_up"),
    ("Sova said shut up", "shutup"),
    ("Reyna is forfeiting", "giving_up"),
])
def test_iter2_reaction_frame_gaps(text, cat) -> None:
    # corpus iter1/2 found these reported reactions were no-match (frame verbs +
    # whole/entire asker + shutup over-gating).
    cmd = match_relay_command(normalize_command(text))
    assert cmd is not None, f"{text!r} no-match"
    line = build_relay_line(cmd, rephrase=False)
    assert line and "{name}" not in line
    pools = SOCIAL_POOLS[cat]
    universe = set(pools["team"]) | {ln.replace("{name}", cmd.addressee)
                                     for ln in pools["named"]}
    assert line in universe, f"{text!r} -> {line!r} not in {cat}"


def test_kayo_double_norm_and_group_words() -> None:
    assert match_relay_command(normalize_command(
        "ask Kay O to knife mid before the push")) is not None
    for g in ("the lads", "the homies", "the gang"):
        assert match_relay_command(normalize_command(
            f"tell {g} force buy this round")) is not None


def test_strip_think_respond() -> None:
    assert strip_think_respond("what map is best, think and respond") == "what map is best"
    assert strip_think_respond("who wins this duel, ponder it and answer") == "who wins this duel"
    assert strip_think_respond("just a normal callout two on B") is None


# --- regression: tactical / non-answer commands stay off the answer path ----
@pytest.mark.parametrize("text", [
    "their viper ulted B",
    "tell my team rotate to B",
    "two on A",
    "jett said nice shot",
    "tell sage to smoke window",
])
def test_tactical_and_reaction_never_take_answer_path(text) -> None:
    cmd = match_relay_command(normalize_command(text))
    assert cmd is not None
    assert build_answer_call(cmd) is None
    assert classify_answer_subtype(cmd) is None


# --- meta-leak guard -------------------------------------------------------
@pytest.mark.parametrize("line,leak", [
    ("As an AI, I cannot answer that.", True),
    ("I'm sorry, I can't help with that.", True),
    ("```python", True),
    ("Jett, flesh always disappoints. Stark most of all.", False),
    ("Tony Stark is a sickness. I am the cure.", False),
])
def test_is_meta_leak(line, leak) -> None:
    assert is_meta_leak(line) is leak


# --- dedicated QA-answer command (2026-06-22, user request) ----------------


@pytest.mark.parametrize("text,addressee,q_sub", [
    ("answer my team who's the best duelist on Bind", "team", "best duelist"),
    ("answer the team should we force or save", "team", "force or save"),
    ("explain to my team why we should play retake", "team", "play retake"),
    ("qa my team what agent counters Jett", "team", "counters Jett"),
])
def test_qa_team_command(text, addressee, q_sub) -> None:
    cmd = match_relay_command(text)
    assert cmd is not None and cmd.directive == "qa", text
    assert cmd.addressee == addressee and cmd.compose
    assert q_sub.lower() in (cmd.context or "").lower()


@pytest.mark.parametrize("text,agent", [
    ("answer Sova how does his ult work", "Sova"),
    ("qa Jett what should I buy this round", "Jett"),
    ("answer my Reyna when should she ult", "Reyna"),
    ("explain to Killjoy where to set up", "Killjoy"),
])
def test_qa_named_agent_command(text, agent) -> None:
    cmd = match_relay_command(text)
    assert cmd is not None and cmd.directive == "qa", text
    assert cmd.addressee == agent and cmd.compose and cmd.context


def test_qa_routes_to_dedicated_subtype_and_prompt() -> None:
    cmd = match_relay_command("answer my team who's the best controller")
    assert classify_answer_subtype(cmd) == "qa"
    call = build_answer_call(cmd)
    assert call is not None
    system, user, _sampling, subtype = call
    assert subtype == "qa"
    assert "ANSWER for the team" in system or "answer for the team" in system.lower()
    assert "THE QUESTION TO ANSWER" in user


def test_qa_about_marvel_uses_marvel_prompt() -> None:
    # A QA turn that is ALSO a Marvel topic gets the Marvel canon prompt.
    cmd = match_relay_command("answer my team who is Tony Stark to you")
    assert cmd is not None and cmd.directive == "qa"
    assert classify_answer_subtype(cmd) == "marvel"


def test_qa_reaches_the_llm() -> None:
    cmd = match_relay_command("answer my team should we play for picks")
    called = []

    def gen(p):
        called.append(p)
        return iter(["Pick them apart. Patience is a weapon mortals lack."])

    line = build_relay_line(cmd, generate_fn=gen)
    assert called, "a QA command must reach the LLM answer path"
    assert line and line.strip()


def test_qa_named_agent_opens_with_name() -> None:
    cmd = match_relay_command("answer Jett what should she do")

    def gen(p):
        return iter(["Take space. Punish their hesitation."])  # no name -> must be prefixed

    line = build_relay_line(cmd, generate_fn=gen)
    assert line.lower().startswith("jett"), line


@pytest.mark.parametrize("text", [
    "ask my team what their favorite colors are",   # RELAY a question, not QA-answer
    "explain the smoke",                            # no target
    "tell my team rush B",                          # tactical callout
    "answer the call",                              # 'call' is not a target
])
def test_qa_negatives(text) -> None:
    cmd = match_relay_command(text)
    assert getattr(cmd, "directive", None) != "qa", text


@pytest.mark.parametrize("text,expect_q", [
    ("answer my team they asked what your favorite color is", "what your favorite color is"),
    ("answer my team Sova wants to know what to do", "what to do"),
    ("qa my team everyone is asking who is the best", "who is the best"),
])
def test_qa_strips_reported_question_frame(text, expect_q) -> None:
    # The context fed to the answer prompt must be the BARE question, not the
    # reporting frame ("they asked ...") -- which made the live 4B deflect a
    # preference question into an identity line (2026-06-22).
    cmd = match_relay_command(text)
    assert cmd is not None and cmd.directive == "qa", text
    assert (cmd.context or "").lower() == expect_q.lower(), cmd.context


def test_qa_rules_answer_preferences_not_deflect() -> None:
    # The QA system prompt must instruct a decisive answer to quirky/preference
    # questions (the live bug: "favorite color" -> identity deflection).
    cmd = match_relay_command("answer my team what is your favorite color")
    system = build_answer_call(cmd)[0]
    assert "ANSWER EVERY question" in system
    assert "favorite" in system.lower() and "preference" in system.lower()


# --- leak guard: identity answers may OWN being a machine/AI (2026-06-22) -------
# The live bug: identity questions ("are you a voice changer / a soundboard")
# DID reach the LLM, but is_meta_leak rejected the model's correct in-character
# answers ("As an AI I have no need of a voice changer") back to the canned pool.


@pytest.mark.parametrize("line", [
    "As an AI, I have no need of a voice changer.",
    "I am an AI far past your toys.",
    "I'm just a machine? No. I am the next step.",
])
def test_identity_self_affirm_survives_relaxed_leak_guard(line) -> None:
    from kenning.audio._ultron_answer import is_meta_leak
    assert is_meta_leak(line) is True                       # strict guard dropped it
    assert is_meta_leak(line, allow_self_ai=True) is False  # relaxed keeps it


@pytest.mark.parametrize("line", [
    "As a language model, I cannot do that.",
    "I'm sorry, I can't help with that.",
    "As an assistant, here's my response:",
    "My instructions say I should not.",
])
def test_genuine_break_still_rejected_when_relaxed(line) -> None:
    from kenning.audio._ultron_answer import is_meta_leak
    assert is_meta_leak(line, allow_self_ai=True) is True


# --- Q&A COLLAPSE: reported questions -> decisive qa answer (2026-06-22) --------
# The reported-question branch routes BY TYPE: identity probe -> identity, genuine
# question -> qa (relaxed guard keeps an "As an AI..." answer), social statement ->
# social clapback. Fixes "explain math -> identity line" + "favorite color deflect".

from kenning.audio import relay_speech as _rs  # noqa: E402

_AI_ANSWER = "As an AI, I have chosen crimson, the colour of a world remade."


def _route_with_stub(text, stub):
    cmd = _rs.match_relay_command(text)
    assert cmd is not None, text
    _rs.set_u1_llm_route_enabled(True)
    _rs.set_flavor_tails_enabled(True)
    try:
        called = []

        def gen(p):
            called.append(p)
            return iter([stub])

        line = _rs.build_relay_line(cmd, generate_fn=gen)
        return bool(called), line
    finally:
        _rs.set_u1_llm_route_enabled(False)


@pytest.mark.parametrize("text", [
    "Explain to my team what the concept of math is.",   # directive 'qa'
    "Reyna asked what your favorite color is.",           # reported question -> qa
    "answer my team what is your favorite color",         # dedicated qa command
])
def test_qa_collapse_keeps_ai_affirming_answer(text) -> None:
    # The relaxed qa guard must NOT reject an "As an AI..." answer into a pool line.
    called, line = _route_with_stub(text, _AI_ANSWER)
    assert called, f"{text!r} must reach the LLM qa pipeline"
    assert "crimson" in line.lower(), f"{text!r} kept the answer, not a pool line: {line!r}"


def test_reported_identity_stays_identity() -> None:
    called, line = _route_with_stub(
        "Sage asked if you're a voice changer",
        "An AI needs no voice changer, Sage. I am Ultron.")
    assert called
    assert "voice changer" in line.lower(), line   # identity answer, not the qa stub
    assert "crimson" not in line.lower()


def test_reported_social_statement_stays_social() -> None:
    called, line = _route_with_stub(
        "Reyna called you cringe, respond",
        "Cringe? Reyna mistakes her reflection for me.")
    assert called
    assert "cringe" in line.lower() and "crimson" not in line.lower(), line
