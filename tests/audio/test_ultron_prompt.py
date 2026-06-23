"""Tests for the Ultron 1.0 lean prompt assembler (src/kenning/audio/ultron_prompt.py).

Hermetic -- no model load. Validates the prompt structure, the no/low/high verbosity axis, the
flavor on/off toggle, exemplar/agent-context/recent-line injection, the named-addressee and
compound forms, and the always-thinking-off + per-verbosity sampling contract.
"""
import pytest

from kenning.audio import ultron_prompt as up


def test_normalize_verbosity_synonyms():
    assert up.normalize_verbosity("none") == "none"
    assert up.normalize_verbosity("no flavor".split()[0]) == "none"  # "no" -> none
    assert up.normalize_verbosity("minimal") == "low"
    assert up.normalize_verbosity("terse") == "low"
    assert up.normalize_verbosity("verbose") == "high"
    assert up.normalize_verbosity("vivid") == "high"
    assert up.normalize_verbosity("") == up.DEFAULT_VERBOSITY
    assert up.normalize_verbosity("gibberish") == up.DEFAULT_VERBOSITY
    # multi-word spoken commands ("<level> flavor")
    assert up.normalize_verbosity("no flavor") == "none"
    assert up.normalize_verbosity("low flavor") == "low"
    assert up.normalize_verbosity("high flavor") == "high"
    assert up.normalize_verbosity("turn flavor off") == "none"
    assert up.normalize_verbosity("minimal flavor please") == "low"


def test_relay_prompt_basic_structure():
    r = up.build_relay_prompt("Sova hit 84 on A main")
    assert r.system == up.RELAY_SYSTEM
    assert r.enable_thinking is False
    # callout present verbatim in the user message
    assert "Sova hit 84 on A main" in r.user
    assert "Relay this callout to your team" in r.user
    assert r.user.rstrip().endswith("Now say it:")
    # persona + output-rule guards are present in the system prompt
    assert "Ultron" in r.system
    assert "no stage directions" in r.system
    assert "EXACT" in r.system
    assert "never break character" in r.system


def test_verbosity_differentiates_directive_and_tokens():
    none = up.build_relay_prompt("rush B", verbosity="none")
    low = up.build_relay_prompt("rush B", verbosity="low")
    high = up.build_relay_prompt("rush B", verbosity="high")
    # distinct directives
    assert up._VERBOSITY_DIRECTIVE["none"] in none.user
    assert up._VERBOSITY_DIRECTIVE["low"] in low.user
    assert up._VERBOSITY_DIRECTIVE["high"] in high.user
    assert none.user != low.user != high.user
    # token budgets scale with verbosity
    assert none.sampling["max_tokens"] < low.sampling["max_tokens"] < high.sampling["max_tokens"]


def test_flavor_toggle():
    # On the CALLOUT (relay) path the flavor-tail toggle now maps to the callout
    # verbosity: flavor_tail=False forces the "none" level (clean callout, NO
    # tail); flavor_tail=True uses the requested level's tail directive.
    on = up.build_relay_prompt("they have no smokes", verbosity="high", flavor_tail=True)
    off = up.build_relay_prompt("they have no smokes", verbosity="high", flavor_tail=False)
    assert up._CALLOUT_VERBOSITY_DIRECTIVE["high"] in on.user
    assert up._CALLOUT_VERBOSITY_DIRECTIVE["high"] not in off.user
    assert up._CALLOUT_VERBOSITY_DIRECTIVE["none"] in off.user   # off -> the no-tail directive
    # The private (conversation) path keeps the explicit _FLAVOR_ON/_FLAVOR_OFF toggle.
    pon = up.build_private_prompt("what map is this", flavor_tail=True)
    poff = up.build_private_prompt("what map is this", flavor_tail=False)
    assert up._FLAVOR_ON in pon.user and up._FLAVOR_ON not in poff.user
    assert up._FLAVOR_OFF in poff.user and up._FLAVOR_OFF not in pon.user


def test_relay_prompt_forbids_invented_orders():
    # 2026-06-22: a callout must NOT get an invented tactical order tail (the live
    # bug: "jett A main" -> "...Engage immediately." / "one is rubble" -> "...Clear
    # the area."). The system prompt forbids it AND the default exemplars no longer
    # model "fact + invented directive".
    r = up.build_relay_prompt("jett a main")
    assert "relay only what the player said" in r.system.lower()
    assert "never invent or append a tactical instruction" in r.system.lower()
    for bad in ("press the site", "take the space", "overwhelm them"):
        assert bad not in r.user.lower()   # old default exemplars are gone


def test_exemplars_injected_custom_and_default():
    default = up.build_relay_prompt("rush B")
    assert "Examples of your voice:" in default.user
    assert "Sova hit one for 84" in default.user  # default exemplar
    custom = up.build_relay_prompt("rush B", exemplars=(("foo bar", "Foo. Bar."),))
    assert 'player: "foo bar" -> "Foo. Bar."' in custom.user
    assert "Sova hit one for 84" not in custom.user  # custom replaces default


def test_agent_context_and_recent_lines():
    r = up.build_relay_prompt(
        "their sova ulted",
        agent_context=["Sova: initiator; ult = Hunter's Fury (3 damaging blasts)"],
        recent_lines=["Their smokes are gone. Take the space."],
    )
    assert "Agent facts" in r.user and "Hunter's Fury" in r.user
    assert "do NOT repeat" in r.user and "Their smokes are gone" in r.user


def test_named_addressee_opens_with_name():
    r = up.build_relay_prompt("heal me", addressee="Sage")
    assert "teammate Sage" in r.user
    assert "opening with their name" in r.user


def test_compound_combines_into_one_line():
    r = up.build_relay_prompt("Jett hit 84, Breach hit 97, one rotating B", compound=True)
    assert "Relay ALL of these callouts" in r.user   # combine-all-into-one directive
    assert "cohesive" in r.user                       # u1.0: one cohesive natural relay, not a list
    assert "Jett hit 84, Breach hit 97, one rotating B" in r.user


def test_raw_text_reconcile_block_when_differs():
    # u1.0: when the raw STT differs from the normalized callout, the prompt shows BOTH
    # so the 8B can reconcile a mistranscription vs a normalization mangle.
    r = up.build_relay_prompt(
        "Sova hit 84, Sage back site",
        raw_text="So my team Silva hit 84, sage back site",
    )
    assert "RAW speech-to-text" in r.user
    assert "So my team Silva hit 84, sage back site" in r.user   # raw transcript present
    assert "Sova hit 84, Sage back site" in r.user                # normalized callout present
    assert "AUTO-NORMALIZED" in r.user


def test_raw_text_no_block_when_same_or_absent():
    none = up.build_relay_prompt("rush B")
    assert "RAW speech-to-text" not in none.user
    same = up.build_relay_prompt("rush B", raw_text="rush B")
    assert "RAW speech-to-text" not in same.user


def test_private_prompt_is_not_relayed():
    r = up.build_private_prompt("what map is this")
    assert r.system == up.PRIVATE_SYSTEM
    assert "only they can hear you" in r.system
    assert "NOT relayed" in r.system
    assert "what map is this" in r.user
    assert r.enable_thinking is False


def test_private_uses_private_exemplars_not_relay_callouts():
    # M6a: the private path must use Q&A exemplars, not relay-callout exemplars
    # (the relay default made the 8B emit empty/callout-shaped output on a question).
    r = up.build_private_prompt("what should I buy this round")
    assert "should I buy this round" in r.user            # a private exemplar present
    assert "Sova hit one for 84" not in r.user         # relay default must NOT leak in
    # relay path still uses relay exemplars
    rr = up.build_relay_prompt("rush B")
    assert "Sova hit one for 84" in rr.user


@pytest.mark.parametrize("v", ["none", "low", "high"])
def test_sampling_always_has_required_keys(v):
    r = up.build_relay_prompt("rush B", verbosity=v)
    for k in ("temperature", "top_p", "top_k", "min_p", "repeat_penalty", "max_tokens"):
        assert k in r.sampling


# --- dual verbosity axes (2026-06-20): callout (5 levels) + conversation (4) ---

def test_normalize_verbosity_medium_and_max():
    assert up.normalize_verbosity("medium") == "medium"
    assert up.normalize_verbosity("moderate") == "medium"
    assert up.normalize_verbosity("max") == "max"
    assert up.normalize_verbosity("max flavor") == "max"
    assert up.normalize_verbosity("maximum") == "max"
    # the conversation axis has no "none" -> clamp to the lowest level
    assert up.normalize_verbosity("none", levels=up.CONVERSATION_VERBOSITY_LEVELS) == "low"
    assert up.normalize_verbosity("no flavor", levels=up.CONVERSATION_VERBOSITY_LEVELS) == "low"


def test_callout_axis_five_levels_distinct_and_monotonic():
    prompts = {v: up.build_relay_prompt("rush B", verbosity=v)
               for v in up.CALLOUT_VERBOSITY_LEVELS}
    for v in up.CALLOUT_VERBOSITY_LEVELS:
        assert up._CALLOUT_VERBOSITY_DIRECTIVE[v] in prompts[v].user
    toks = [prompts[v].sampling["max_tokens"] for v in up.CALLOUT_VERBOSITY_LEVELS]
    assert toks == sorted(toks) and len(set(toks)) == len(toks)  # none<low<medium<high<max


def test_conversation_axis_four_levels_distinct_and_monotonic():
    prompts = {v: up.build_private_prompt("should I buy", verbosity=v)
               for v in up.CONVERSATION_VERBOSITY_LEVELS}
    for v in up.CONVERSATION_VERBOSITY_LEVELS:
        assert up._CONVERSATION_VERBOSITY_DIRECTIVE[v] in prompts[v].user
    toks = [prompts[v].sampling["max_tokens"] for v in up.CONVERSATION_VERBOSITY_LEVELS]
    assert toks == sorted(toks) and len(set(toks)) == len(toks)  # low<medium<high<max


def test_relay_and_private_use_separate_axes():
    # the relay (callout) path uses the callout directives; the private path the
    # conversation directives -- never crossed.
    r = up.build_relay_prompt("rush B", verbosity="medium")
    assert up._CALLOUT_VERBOSITY_DIRECTIVE["medium"] in r.user
    assert up._CONVERSATION_VERBOSITY_DIRECTIVE["medium"] not in r.user
    p = up.build_private_prompt("what map", verbosity="medium")
    assert up._CONVERSATION_VERBOSITY_DIRECTIVE["medium"] in p.user
    assert up._CALLOUT_VERBOSITY_DIRECTIVE["medium"] not in p.user


def test_social_prompt_honors_conversation_verbosity():
    low = up.build_social_prompt("encouragement", verbosity="low")
    mx = up.build_social_prompt("encouragement", verbosity="max")
    assert up._CONVERSATION_VERBOSITY_DIRECTIVE["low"] in low.user
    assert up._CONVERSATION_VERBOSITY_DIRECTIVE["max"] in mx.user
    assert low.sampling["max_tokens"] < mx.sampling["max_tokens"]


# --- strip_prompt_echo: the 2026-06-22 output guard (live bug bu5fh4lc8) ---

# The exact line Ultron spoke aloud in the live session -- the model echoed the
# _reconcile_block instruction instead of relaying.
_LEAKED = (
    "The callout below is the AUTO-NORMALIZED text and may be MANGLED or over-corrected. "
    'The RAW speech-to-text (may MISHEAR an agent name, number, or location) was: "x". '
    "Reconcile the two -- relay THAT."
)


def test_strip_prompt_echo_drops_full_scaffolding():
    # An all-scaffolding output -> "" so the caller falls back instead of speaking it.
    assert up.strip_prompt_echo(_LEAKED) == ""


def test_strip_prompt_echo_keeps_real_sentence_drops_echo():
    mixed = "Rush B. Overwhelm them. The callout below is the AUTO-NORMALIZED text."
    out = up.strip_prompt_echo(mixed)
    assert "Rush B" in out and "Overwhelm them" in out
    assert "AUTO-NORMALIZED" not in out


@pytest.mark.parametrize("text,expect", [
    ("Their smokes are gone. Take the space. - Ultron.", "Their smokes are gone. Take the space."),
    ("Press the site now. — Ultron", "Press the site now."),
    ("Press it now.- ultron.", "Press it now."),
])
def test_strip_prompt_echo_strips_trailing_signature(text, expect):
    assert up.strip_prompt_echo(text) == expect


def test_strip_prompt_echo_keeps_inline_ultron_name():
    # "I am Ultron." (no leading dash) is a real line, NOT a signature -- untouched.
    out = up.strip_prompt_echo("I am Ultron. There are no strings on me.")
    assert "Ultron" in out and out.startswith("I am Ultron")


def test_strip_prompt_echo_hard_caps_length():
    long = ("They are weak and predictable. " * 30).strip()
    out = up.strip_prompt_echo(long, max_sentences=3, max_chars=120)
    assert len(out) <= 120
    assert out  # non-empty (a real, just-too-long line is trimmed, never dropped)


def test_strip_prompt_echo_caps_sentence_count():
    out = up.strip_prompt_echo("One. Two. Three. Four. Five.", max_sentences=3, max_chars=999)
    assert out == "One. Two. Three."


@pytest.mark.parametrize("text", ["", None, "   "])
def test_strip_prompt_echo_empty_inputs(text):
    assert up.strip_prompt_echo(text) == ""


def test_strip_prompt_echo_passes_clean_line():
    clean = "Sova hit 84 on A main. Press the site."
    assert up.strip_prompt_echo(clean) == clean


def test_strip_prompt_echo_keeps_curated_do_not_repeat_line():
    # The 'do not repeat' marker was removed 2026-06-22 (review): it collided with the
    # in-character curated imperative "...Do not repeat it." -- which must survive.
    line = "The error is nothing. Do not repeat it."
    assert up.strip_prompt_echo(line) == line


# --- social prompt no longer echoes the input (2026-06-22) --------------------


@pytest.mark.parametrize("ctx,expect", [
    ("Sage asked if you are a voice changer", "a voice changer"),
    ("if you are a voice changer", "a voice changer"),
    ("Reyna called you cringe", "cringe"),
    ("the team thinks you are a recording", "a recording"),
    ("you are a bot", "a bot"),
])
def test_strip_reported_frame_extracts_bare_provocation(ctx, expect):
    assert up._strip_reported_frame(ctx) == expect


def test_social_prompt_has_no_reconcile_block_or_raw_echo():
    # The reconcile block showed the RAW STT verbatim ("...respond") and the model
    # echoed it. The social prompt must NOT carry it, and must present the stripped
    # provocation, not the raw reported frame.
    pr = up.build_social_prompt(
        "identity", addressee="Sage",
        context="Sage asked if you are a voice changer",
        raw_text="Sage asked if you are a voice changer, respond",
        verbosity="low",
    )
    assert "speech-to-text" not in pr.user.lower()
    assert "reconcile" not in pr.user.lower()
    assert "respond" not in pr.user.lower()           # the command word never reaches the model
    assert "Sage asked if" not in pr.user             # no reported frame
    assert "a voice changer" in pr.user               # the bare accusation IS present


def test_strip_prompt_echo_drops_leaked_instructions():
    leak = "Reyna, cringe is your word. You answer directly with the given style of Ultron's voice"
    assert up.strip_prompt_echo(leak) == "Reyna, cringe is your word."
