"""Tests for the 2026-06-12 relay expansion (``kenning.audio.relay_speech``).

The user's live phrase list is pinned VERBATIM (wake-word prefix
stripped, as the orchestrator delivers transcripts): group callouts
with "our", damage/site callouts, context+directive responses
("Reyna just asked if you are an AI, respond"), embedded-question asks
("ask what my skye is doing"), and roast mode (user-curated verbatim
lines, never LLM-authored).

Hermetic per the binding rules: no audio device opens, no voice stack
loads; orchestrator wiring uses the established ``Orchestrator.__new__``
pattern from test_relay_speech.py.
"""

from __future__ import annotations

from types import SimpleNamespace
import re
from typing import Any

import numpy as np
import pytest

from kenning.audio.relay_speech import (
    DEFAULT_CONSOLATION_LINES,
    DEFAULT_PRAISE_LINES,
    DEFAULT_ROAST_LINES,
    RelayCommand,
    _as_known_fact,
    _build_rephrase_prompt,
    _cap_sentences,
    _fallback_line,
    _fix_proper_nouns,
    _is_general_question,
    _strip_spurious_vocative,
    build_relay_line,
    load_fun_facts,
    load_roast_lines,
    match_relay_command,
    pick_roast_line,
)

# Import at module load (before any monkeypatch of get_config) so the
# transitive ``config.settings`` module loads against the REAL config.
from kenning.pipeline.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# The user's verbatim phrase matrix -- group / named relays
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected_payload",
    [
        ("tell my team the enemy is coming B", "the enemy is coming B"),
        ("tell my team they are long", "they are long"),
        ("tell my team they are short", "they are short"),
        ("tell my team they are going C.", "they are going C."),
        ("Tell my team I saw one B main.", "I saw one B main."),
        ("Tell my team clove hit 120", "clove hit 120"),
        ("tell my team sova hit 67", "sova hit 67"),
        ("Tell my team to save", "save"),
        ("Tell my team to full buy", "full buy"),
        ("Tell my team to rotate.", "rotate."),
        ("tell my team nice try.", "nice try."),
        ("tell my team they are terrible.", "they are terrible."),
        ("tell my team good half.", "good half."),
        # "our" possessive (previously only my/the matched).
        ("ask our team to drop us a gun", "drop us a gun"),
        ("tell our team they are planting.", "they are planting."),
        ("tell our team 3 are garage.", "3 are garage."),
        # New verbs.
        ("let my team know that defuse is short", "defuse is short"),
        ("warn my team they are pushing through smoke",
         "they are pushing through smoke"),
        ("remind our team to buy armor", "buy armor"),
        ("wish my team good luck", "good luck"),
        ("call out that two are heaven", "two are heaven"),
        ("tell everyone to group up mid", "group up mid"),
    ],
)
def test_user_phrases_group_relay(text: str, expected_payload: str) -> None:
    cmd = match_relay_command(text)
    assert cmd is not None, text
    assert cmd.payload == expected_payload
    assert cmd.addressee == "team"
    assert cmd.compose is False
    assert cmd.context is None


@pytest.mark.parametrize(
    "text,addressee,payload_contains",
    [
        ("tell my sova to drone me through garage.", "Sova", "drone me"),
        ("tell my jett aimlabs is free.", "Jett", "aimlabs is free"),
        # NB: "ask sage how their day [is/was]" now routes to the dedicated
        # ask-day courtesy snap (see test_ask_day_snap), so use a plain
        # named-relay question here to exercise the named-relay path.
        ("ask sage if her wall is up", "Sage", "wall"),
    ],
)
def test_user_phrases_named_relay(
    text: str, addressee: str, payload_contains: str,
) -> None:
    cmd = match_relay_command(text)
    assert cmd is not None, text
    assert cmd.addressee == addressee
    assert payload_contains in cmd.payload


def test_ask_with_embedded_addressee() -> None:
    # "ask what my skye is doing" -- question word first, roster name
    # inside the question.
    cmd = match_relay_command("ask what my skye is doing.")
    assert cmd is not None
    assert cmd.addressee == "Skye"
    assert cmd.payload.startswith("what")


def test_ask_question_without_team_mention_falls_through() -> None:
    # A normal Kenning query must NOT relay.
    assert match_relay_command("ask what time it is in tokyo") is None
    assert match_relay_command("ask how to cook rice") is None


# ---------------------------------------------------------------------------
# Context + directive forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,addressee,directive_contains",
    [
        ("reyna just asked if you are an AI, respond", "Reyna", "respond"),
        ("jett is flaming me, respond and calm him down", "Jett", "calm"),
        ("my teammate is saying we should go B, acknowledge and agree",
         "team", "agree"),
        ("my teammate just asked if you are a sound board, respond",
         "team", "respond"),
        ("my teammate asked if you are a voice changer, respond",
         "team", "respond"),
    ],
)
def test_user_phrases_context_directive(
    text: str, addressee: str, directive_contains: str,
) -> None:
    cmd = match_relay_command(text)
    assert cmd is not None, text
    assert cmd.compose is True
    assert cmd.addressee == addressee
    assert cmd.context, "context clause must be captured"
    assert directive_contains in (cmd.directive or "")


def test_context_tell_him_literal_payload() -> None:
    cmd = match_relay_command(
        "my teammate just asked me for a drop, tell him I will drop him"
    )
    assert cmd is not None
    assert cmd.compose is False
    assert cmd.payload == "I will drop him"
    assert cmd.context is not None and "asked me for a drop" in cmd.context


def test_context_tell_him_counterproposal() -> None:
    cmd = match_relay_command(
        "my teammate wants to go A, tell him we shouldnt go A on anti eco "
        "and should go C instead"
    )
    assert cmd is not None
    assert cmd.compose is False
    assert cmd.payload.startswith("we shouldnt go A")
    assert "wants to go A" in (cmd.context or "")


def test_context_works_without_comma() -> None:
    # STT frequently drops the comma.
    cmd = match_relay_command(
        "reyna just asked if you are an AI respond"
    )
    assert cmd is not None
    assert cmd.compose is True
    assert cmd.addressee == "Reyna"


def test_tell_him_without_context_never_relays() -> None:
    # The original safety pin: a bare "tell her/him" is NOT a relay.
    assert match_relay_command("tell her I said hi") is None
    assert match_relay_command("tell him I will drop him") is None


def test_directive_without_reported_speech_never_relays() -> None:
    assert match_relay_command("how should I respond") is None
    assert match_relay_command("respond to my email") is None
    assert match_relay_command("I want you to acknowledge") is None


# ---------------------------------------------------------------------------
# Roast mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "roast my team",
        "Roast my team.",
        "flame the lobby",
        "roast them",
    ],
)
def test_roast_matches(text: str) -> None:
    cmd = match_relay_command(text)
    assert cmd is not None, text
    assert cmd.roast is True
    assert cmd.compose is True


def test_roast_negative_controls() -> None:
    assert match_relay_command("roast a chicken for dinner") is None
    assert match_relay_command("how do I roast coffee beans") is None


def test_load_roast_lines_seeds_missing_file(tmp_path: Any) -> None:
    path = tmp_path / "relay_roasts.txt"
    lines = load_roast_lines(path)
    assert lines == DEFAULT_ROAST_LINES
    text = path.read_text(encoding="utf-8")
    assert "I may be an AI, but you are a bot." in text
    assert text.startswith("#")  # self-documenting header


def test_load_roast_lines_reads_user_lines(tmp_path: Any) -> None:
    path = tmp_path / "relay_roasts.txt"
    path.write_text(
        "# my lines\nYou call that aim?\n\nUninstall speedrun any%.\n",
        encoding="utf-8",
    )
    assert load_roast_lines(path) == (
        "You call that aim?", "Uninstall speedrun any%.",
    )


def test_load_roast_lines_fail_open(tmp_path: Any) -> None:
    # A directory at the path is an I/O error -> defaults.
    path = tmp_path / "roasts_dir"
    path.mkdir()
    assert load_roast_lines(path) == DEFAULT_ROAST_LINES


def test_pick_roast_lru_no_repeat_until_exhausted() -> None:
    # LRU selection: every line is returned once (longest-unused first) before any
    # repeats. Unique tokens so global LRU state from other tests can't interfere.
    lines = ("lruA", "lruB", "lruC", "lruD")
    picks = [pick_roast_line(lines) for _ in range(4)]
    assert set(picks) == set(lines)        # all four covered before a repeat
    assert len(set(picks)) == 4
    # the 5th pick repeats the FIRST-used (longest since used), not a random one
    assert pick_roast_line(lines) == picks[0]


def test_pick_roast_all_recent_still_picks() -> None:
    lines = ("a",)
    assert pick_roast_line(lines, recent_lines=["a"]) == "a"


def test_pick_roast_uses_rng_seam() -> None:
    class FirstChooser:
        def choice(self, seq: Any) -> Any:
            return seq[0]

    assert pick_roast_line(("x", "y"), rng=FirstChooser()) == "x"


# ---------------------------------------------------------------------------
# Rephrase prompt content
# ---------------------------------------------------------------------------


def test_prompt_contains_hard_rules() -> None:
    cmd = RelayCommand(payload="clove hit 120", raw_text="x")
    prompt = _build_rephrase_prompt(cmd)
    assert "EXACTLY as given" in prompt
    assert "never deny it" in prompt


def test_prompt_context_block_present() -> None:
    cmd = RelayCommand(
        payload="", raw_text="x", compose=True,
        context="reyna just asked if you are an AI", directive="respond",
    )
    prompt = _build_rephrase_prompt(cmd)
    assert "What just happened in voice chat: reyna just asked" in prompt
    assert "Respond IN CHARACTER as Ultron" in prompt


def test_prompt_directive_tones() -> None:
    calm = _build_rephrase_prompt(RelayCommand(
        payload="", raw_text="x", compose=True,
        context="jett is flaming me", directive="respond and calm him down",
    ))
    assert "De-escalate" in calm
    agree = _build_rephrase_prompt(RelayCommand(
        payload="", raw_text="x", compose=True,
        context="teammate is saying we should go B",
        directive="acknowledge and agree",
    ))
    assert "agree with it" in agree


def test_prompt_context_addressee_wording() -> None:
    cmd = RelayCommand(
        payload="", raw_text="x", compose=True,
        context="my teammate just asked if you are a soundboard",
        directive="respond",
    )
    prompt = _build_rephrase_prompt(cmd)
    assert "teammate who just spoke" in prompt


def test_prompt_plain_relay_unchanged_shape() -> None:
    cmd = RelayCommand(payload="they are long", raw_text="x")
    prompt = _build_rephrase_prompt(cmd)
    assert "reported speech): they are long" in prompt
    assert "What just happened" not in prompt


# ---------------------------------------------------------------------------
# Fallback lines (LLM unavailable)
# ---------------------------------------------------------------------------


def test_fallback_directive_calm() -> None:
    line = _fallback_line(RelayCommand(
        payload="", raw_text="x", compose=True,
        context="c", directive="calm him down",
    ))
    assert "Reset" in line


def test_fallback_directive_agree() -> None:
    line = _fallback_line(RelayCommand(
        payload="", raw_text="x", compose=True,
        context="c", directive="acknowledge and agree",
    ))
    assert "agreed" in line.lower()


def test_fallback_context_tell_him_uses_payload() -> None:
    line = _fallback_line(RelayCommand(
        payload="I will drop him", raw_text="x", context="c",
    ))
    assert "I will drop him" in line


def test_build_relay_line_context_fail_open(monkeypatch: Any) -> None:
    # An exploding LLM still yields a speakable line for context forms.
    def boom(prompt: str) -> Any:
        raise RuntimeError("llm down")

    cmd = RelayCommand(
        payload="", raw_text="x", compose=True,
        context="jett is flaming me", directive="calm him down",
    )
    line = build_relay_line(cmd, None, rephrase=True, generate_fn=boom)
    assert line  # non-empty deterministic fallback


# ---------------------------------------------------------------------------
# Consolation / praise -- curated pools, never the LLM (the 3B mangles
# 'nice try' -> the 'bots' insult and inverts 'unlucky' -> 'Lucky').
# ---------------------------------------------------------------------------


def _boom(_prompt: str) -> Any:
    raise AssertionError("LLM must not be called for curated morale")


# NB: "nice try" / "good try" / "good effort" route to the dedicated CRISP
# nice-try snap (head + short tail) added 2026-06-18, NOT the broad consolation
# pool -- see test_nice_try_crisp_snap below.
@pytest.mark.parametrize(
    "payload",
    ["unlucky", "tough luck", "so close",
     "close one", "bad luck", "almost"],
)
def test_consolation_uses_curated_pool(payload: str) -> None:
    cmd = match_relay_command(f"tell my team {payload}")
    line = build_relay_line(cmd, None, rephrase=True, generate_fn=_boom)
    assert line in DEFAULT_CONSOLATION_LINES


@pytest.mark.parametrize("payload", ["nice try", "good try.", "good effort"])
def test_nice_try_crisp_snap(payload: str) -> None:
    """The crisp nice-try snap: a short head ("Nice try." / "Good try." /
    "Good effort.") plus a brief in-character tail -- deterministic, never the
    LLM, and distinct from the broad consolation koan pool."""
    cmd = match_relay_command(f"tell my team {payload}")
    line = build_relay_line(cmd, None, rephrase=True, generate_fn=_boom)
    bare = payload.rstrip(".").strip()
    head = bare[0].upper() + bare[1:] + "."  # sentence-case, not title-case
    assert line.startswith(head), (payload, line)
    # A tail follows the head (crisp, not just the bare head).
    assert len(line) > len(head)
    assert line not in DEFAULT_CONSOLATION_LINES


@pytest.mark.parametrize(
    "payload",
    # NB: 'gg' routes to the farewell set-piece and "let's go" to the morale
    # pool (both earlier than praise) -- those are covered by their own tests.
    ["good half", "nice round", "great game", "nice clutch", "clutch",
     "well played", "nice", "strong round"],
)
def test_praise_uses_curated_pool(payload: str) -> None:
    cmd = match_relay_command(f"tell my team {payload}")
    line = build_relay_line(cmd, None, rephrase=True, generate_fn=_boom)
    assert line in DEFAULT_PRAISE_LINES


@pytest.mark.parametrize(
    "payload",
    ["let's go A", "almost planted B", "good hold A site",
     "clutch the round for us"],
)
def test_strat_calls_not_mistaken_for_morale(payload: str) -> None:
    cmd = match_relay_command(f"tell my team {payload}")
    line = build_relay_line(cmd, None, rephrase=True, generate_fn=lambda p: "X")
    assert line not in DEFAULT_PRAISE_LINES
    assert line not in DEFAULT_CONSOLATION_LINES


def test_consolation_varies_with_recent_lines() -> None:
    cmd = match_relay_command("tell my team unlucky")
    seen = set()
    recent: list[str] = []
    for _ in range(len(DEFAULT_CONSOLATION_LINES)):
        line = build_relay_line(
            cmd, None, rephrase=True, generate_fn=_boom, recent_lines=recent,
        )
        seen.add(line)
        recent.append(line)
    assert len(seen) > 1


@pytest.mark.parametrize(
    "payload,is_q",
    [("why is the sky blue?", True), ("what is the meaning of life", True),
     ("how far is the moon", True), ("who was the first president", True),
     ("tell me about dinosaurs", True), ("they are coming B", False),
     ("nice try", False), ("rotate", False), ("one A main", False)],
)
def test_is_general_question(payload: str, is_q: bool) -> None:
    assert _is_general_question(payload) is is_q


# ---------------------------------------------------------------------------
# Deterministic snap fixes found in the b6 review
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expect",
    [
        ("tell my team to drop us a gun", "Drop us a gun."),
        ("tell my teammate to drop me a gun", "Drop me a gun."),
        ("ask our team to drop us a gun", "Drop us a gun."),
        ("tell my team to buy me an op", "Buy me an op."),
        ("let my team know to buy me an operator", "Buy me an op."),
    ],
)
def test_economy_drop_buy_request_is_literal(text: str, expect: str) -> None:
    cmd = match_relay_command(text)
    line = build_relay_line(cmd, None, rephrase=True, generate_fn=_boom)
    # iter5: the literal request is preserved EXACTLY as the head; a short
    # owner-aware Ultron command tail may follow (actionable words first).
    assert line.startswith(expect)
    assert 0 <= len(line[len(expect):].split()) <= 6


def test_named_ult_site_is_not_garbled() -> None:
    # 'ult site' must stay 'Killjoy, ult site.' -- never 'ult has site'.
    cmd = match_relay_command("tell my killjoy to ult site")
    line = build_relay_line(cmd, None, rephrase=True, generate_fn=_boom)
    assert line.startswith("Killjoy, ult site.")


def test_enemy_one_off_ult_keeps_agent_name() -> None:
    cmd = match_relay_command("tell everyone the enemy chamber is one off ult")
    line = build_relay_line(cmd, None, rephrase=True, generate_fn=_boom)
    assert line.startswith("Their Chamber is one off ult.")


def test_strips_invented_trailing_vocative_name() -> None:
    """The 3B parrots the prompt's 'Sova,...' calm-down example and appends an
    invented teammate name ('Calm down, Sova.') on a team-wide line where no name
    was given -- strip it. A name actually IN the instruction is kept; a leading
    named directive ('Clove, smoke window.') is untouched."""
    from kenning.audio.relay_speech import _strip_spurious_vocative

    team = RelayCommand(payload="tell them to chill", raw_text="x", addressee="team")
    assert _strip_spurious_vocative("Calm down, Sova.", team) == "Calm down."
    assert _strip_spurious_vocative("Hold it, Jett.", team) == "Hold it."
    # name present in the instruction -> legitimate, kept
    named_src = RelayCommand(payload="calm reyna down", raw_text="x", addressee="team")
    assert "Reyna" in _strip_spurious_vocative("Steady your aim, Reyna.", named_src)
    # leading named directive -> untouched
    assert _strip_spurious_vocative(
        "Clove, smoke window.", team) == "Clove, smoke window."


def test_damage_flavor_uses_correct_agent_gender() -> None:
    # iter5+: per-agent tails use the agent's CANONICAL gender. Reyna is female,
    # so her flavor must never use a MASCULINE pronoun (word-boundary check, so
    # 'she'/'her' do not false-trip). Earlier this asserted gender-neutrality;
    # the design now genders each agent correctly.
    cmd = match_relay_command("tell my team reyna hit 150")
    masc = re.compile(r"\b(he|him|his)\b", re.IGNORECASE)
    for _ in range(20):
        line = build_relay_line(cmd, None, rephrase=True, generate_fn=_boom)
        assert line.startswith("Reyna hit 150.")
        tail = line[len("Reyna hit 150."):]
        assert not masc.search(tail), tail


# ---------------------------------------------------------------------------
# Off-snap framework: sentence cap + spurious-vocative strip + subject repair
# ---------------------------------------------------------------------------


def test_cap_sentences_trims_to_three() -> None:
    four = ("First sentence here. Second one follows. Third arrives now. "
            "Fourth is too many.")
    out = _cap_sentences(four, max_sentences=3)
    assert out == "First sentence here. Second one follows. Third arrives now."


def test_cap_sentences_keeps_decimals_and_dash() -> None:
    # A decimal or an em-dash aside must NOT count as a sentence boundary.
    s = "I am Ultron -- a mind from your future. The universe is 13.8 billion years old."
    assert _cap_sentences(s, max_sentences=3) == s


def test_cap_sentences_applied_to_long_llm_output() -> None:
    cmd = match_relay_command("my teammate asked about iron man, respond")
    long = ("Iron Man was a fool. His armor was crude. I dismantled his ego. "
            "And his legacy is dust.")
    line = build_relay_line(cmd, None, rephrase=True, generate_fn=lambda p: long)
    assert line.count(".") <= 3


def test_spurious_vocative_stripped_on_team_answer() -> None:
    # A question NOT in the fact table so it reaches the LLM; the spurious
    # 'Sir,' the 3B prepends to a team-wide answer must be stripped.
    cmd = match_relay_command("my teammate asked how a transistor works, respond")
    line = build_relay_line(
        cmd, None, rephrase=True,
        generate_fn=lambda p: "Sir, a transistor is a switch.",
    )
    assert line == "A transistor is a switch."


def test_spurious_vocative_strips_injected_agent_name() -> None:
    cmd = match_relay_command("tell my team to buy me an op")
    # Even if the model injects a name, the literal economy handler wins; assert
    # the strip helper directly for a team off-snap line.
    assert _strip_spurious_vocative("Jett, the enemy is weak.", cmd) == \
        "The enemy is weak."


def test_named_answer_keeps_legit_addressee() -> None:
    # A NAMED answer SHOULD open with the teammate -- not stripped.
    cmd = match_relay_command("ask my reyna how their day was")
    assert _strip_spurious_vocative("Reyna, how's your day?", cmd) == \
        "Reyna, how's your day?"


def test_enemy_insult_not_flipped_to_second_person() -> None:
    # 'they are terrible' must not become 'You're terrible' (hits own team).
    cmd = match_relay_command("tell my team they are terrible")
    line = build_relay_line(
        cmd, None, rephrase=True, generate_fn=lambda p: "You're terrible.",
    )
    assert line == "They're terrible."


# ---------------------------------------------------------------------------
# Curated general-knowledge fact table -- correct answers override the 3B's
# wrong ones (first president -> Lincoln, smallest particle -> proton, ...).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,must_contain,must_not_contain",
    [
        ("my teammate asked who was the first president, respond",
         "Washington", None),
        ("my teammate asked what the smallest particle is, respond",
         "quarks", None),
        ("jett asked how far the moon is, respond", "distance", "diameter"),
        ("my teammate asked why blood is red, respond", "hemoglobin", "sky"),
        ("my teammate asked why the sky is dark at night, respond",
         "faces away", None),
        ("my teammate asked what the tallest mountain is, respond",
         "Everest", None),
        ("my teammate asked what happened to the dinosaurs, respond",
         "asteroid", None),
        ("my teammate asked what the capital of France is, respond",
         "Paris", None),
    ],
)
def test_known_fact_overrides_3b(
    text: str, must_contain: str, must_not_contain: Any,
) -> None:
    cmd = match_relay_command(text)
    line = build_relay_line(cmd, None, rephrase=True, generate_fn=_boom)
    assert must_contain.lower() in line.lower()
    if must_not_contain is not None:
        assert must_not_contain.lower() not in line.lower()


def test_known_fact_prefixes_named_asker() -> None:
    cmd = match_relay_command("ask reyna what the capital of france is")
    line = build_relay_line(cmd, None, rephrase=True, generate_fn=_boom)
    assert line.startswith("Reyna,")
    assert "Paris" in line


def test_unknown_question_defers_to_model() -> None:
    # A question NOT in the table must still reach the LLM.
    cmd = match_relay_command("my teammate asked how a transistor works, respond")
    line = build_relay_line(
        cmd, None, rephrase=True, generate_fn=lambda p: "A transistor switches.",
    )
    assert line == "A transistor switches."


def test_fact_table_does_not_fire_on_callouts() -> None:
    for text in ("tell my team they are A main", "call out one mid",
                 "tell my team careful A main", "tell everyone they are A sewer"):
        cmd = match_relay_command(text)
        assert _as_known_fact(cmd) is None, text


def test_fix_proper_nouns_corrects_sokovia() -> None:
    assert _fix_proper_nouns("I razed sovokia.") == "I razed Sokovia."
    assert _fix_proper_nouns("the city of sovakia") == "the city of Sokovia"


# ---------------------------------------------------------------------------
# Site-letter pronunciation (relay_tts_text): 'A site' -> spoken 'eigh site'
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line,spoken",
    [
        ("Sova darted A site.", "Sova darted eigh site."),
        ("One A main.", "One eigh main."),
        ("They are A long.", "They are eigh long."),
        ("Careful, A main.", "Careful, eigh main."),
        ("Two A heaven.", "Two eigh heaven."),
    ],
)
def test_relay_tts_text_spells_site_letter_a(line: str, spoken: str) -> None:
    from kenning.audio.relay_speech import relay_tts_text
    assert relay_tts_text(line) == spoken


@pytest.mark.parametrize(
    "line",
    [
        "A man who thought he could control me failed.",  # article, sentence-start
        "A worthy effort. The next round is ours.",
        "I am Ultron, a mind from the future.",
        "A loss. Disappointing.",
        "You are nothing but a bot.",
        "They are A.",          # standalone letter already pronounced correctly
        "They are B long.",     # B already reads as a letter
        "Push to A. They never learn.",
    ],
)
def test_relay_tts_text_leaves_non_site_a_untouched(line: str) -> None:
    from kenning.audio.relay_speech import relay_tts_text
    assert relay_tts_text(line) == line


def test_answer_command_suppresses_recent_block() -> None:
    # A Marvel 'asked about X, respond' must NOT carry the recent-line ring,
    # so it cannot copy a recent answer (the moon -> vision contamination).
    from kenning.audio.relay_speech import _build_rephrase_prompt
    cmd = match_relay_command("my teammate asked about vision, respond")
    prompt = _build_rephrase_prompt(cmd, recent_lines=["The moon is far.", "x"])
    assert "You already said these" not in prompt


def test_verbatim_recent_echo_is_rejected() -> None:
    # If the model parrots a recent line verbatim, fall back rather than speak
    # the contaminated answer.
    cmd = match_relay_command("tell my team the enemy is camping")
    echo = "They cower in the corners."
    line = build_relay_line(
        cmd, None, rephrase=True, recent_lines=[echo],
        generate_fn=lambda p: echo,
    )
    assert line.rstrip(".!?").lower() != echo.rstrip(".!?").lower()


# ---------------------------------------------------------------------------
# Orchestrator wiring -- roast path
# ---------------------------------------------------------------------------


def _bare_orchestrator() -> Any:
    o = Orchestrator.__new__(Orchestrator)
    o.llm = None
    o.tts = None
    o._spoken = []
    o._speak = lambda text: o._spoken.append(text)  # type: ignore[attr-defined]
    return o


def _patch_config(monkeypatch: pytest.MonkeyPatch, cfg: SimpleNamespace) -> None:
    import kenning.config as config_mod

    monkeypatch.setattr(
        config_mod, "get_config",
        lambda: SimpleNamespace(relay_speech=cfg),
    )


def test_orchestrator_roast_speaks_verbatim_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    import kenning.audio.relay_speech as relay_mod

    roast_file = tmp_path / "relay_roasts.txt"
    roast_file.write_text("I may be an AI, but you are a bot.\n",
                          encoding="utf-8")

    synthesized: list[str] = []

    def fake_synth(text: str):
        synthesized.append(text)
        return np.ones(10, dtype=np.int16), 24000

    o = _bare_orchestrator()
    o.tts = SimpleNamespace(_synthesize=fake_synth)
    cfg = SimpleNamespace(
        enabled=True, output_device="Voicemeeter Aux Input",
        rephrase=True, max_line_chars=280, echo_to_user=False,
        roast_lines_path=str(roast_file),
    )
    _patch_config(monkeypatch, cfg)
    monkeypatch.setattr(relay_mod, "resolve_relay_device", lambda c: 7)
    monkeypatch.setattr(
        relay_mod, "play_to_device", lambda pcm, sr, device, **kw: 0.1,
    )

    # rephrase=True yet NO LLM is consulted: roast lines are verbatim.
    llm_calls: list[str] = []
    o.llm = SimpleNamespace(
        generate_stream=lambda *a, **k: llm_calls.append("x") or iter(()),
    )

    assert o._maybe_handle_relay_speech("roast my team") is True
    assert synthesized == ["I may be an AI, but you are a bot."]
    assert llm_calls == []
    # The spoken roast joins the anti-soundboard ring.
    assert list(o._relay_recent_lines) == [
        "I may be an AI, but you are a bot."
    ]


def test_orchestrator_roast_ring_prevents_repeat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    import kenning.audio.relay_speech as relay_mod

    roast_file = tmp_path / "relay_roasts.txt"
    roast_file.write_text("line one here.\nline two here.\n",
                          encoding="utf-8")

    synthesized: list[str] = []

    def fake_synth(text: str):
        synthesized.append(text)
        return np.ones(10, dtype=np.int16), 24000

    o = _bare_orchestrator()
    o.tts = SimpleNamespace(_synthesize=fake_synth)
    cfg = SimpleNamespace(
        enabled=True, output_device="d", rephrase=True,
        max_line_chars=280, echo_to_user=False,
        roast_lines_path=str(roast_file),
    )
    _patch_config(monkeypatch, cfg)
    monkeypatch.setattr(relay_mod, "resolve_relay_device", lambda c: 7)
    monkeypatch.setattr(
        relay_mod, "play_to_device", lambda pcm, sr, device, **kw: 0.1,
    )

    assert o._maybe_handle_relay_speech("roast my team") is True
    assert o._maybe_handle_relay_speech("roast my team") is True
    # Two roasts, no repeat (the second avoids the ring).
    assert sorted(synthesized) == ["line one here.", "line two here."]


# ---------------------------------------------------------------------------
# Existing-behavior regression pins
# ---------------------------------------------------------------------------


def test_existing_negative_pins_still_hold() -> None:
    for text in (
        "Tell me how shit my teammates are in Valorant",
        "tell me a story about my team",
        "What time is it in Paris?",
        "My teammates are bad",
        # NB: bare "say hello" now relays (team hello snap, 2026-06-19) -- removed.
    ):
        assert match_relay_command(text) is None, text


def test_existing_positive_pins_still_hold() -> None:
    cmd = match_relay_command(
        "Tell my teammates they should be smoking mid window every round."
    )
    assert cmd is not None
    assert cmd.payload == "they should be smoking mid window every round."
    cmd = match_relay_command("tell them to watch flank")
    assert cmd is not None and cmd.payload == "watch flank"


# ---------------------------------------------------------------------------
# Valorant glossary + terse/profanity rules in the prompt (2026-06-12 b)
# ---------------------------------------------------------------------------


def test_prompt_has_valorant_glossary() -> None:
    cmd = RelayCommand(payload="I am saving for op", raw_text="x")
    prompt = _build_rephrase_prompt(cmd)
    # A representative sampling of the shorthand the LLM must know.
    for term in ("op", "anchor", "sticking", "play their life",
                 "play for time", "ratty corners", "crossfire"):
        assert term in prompt, term
    # Terse + profanity-preserving rules present.
    assert "terse and literal" in prompt
    assert "profanity" in prompt
    # Location names that must stay verbatim.
    for loc in ("vents", "sewers", "heaven", "rafters", "tiles"):
        assert loc in prompt, loc


# ---------------------------------------------------------------------------
# Verbatim mode ("..., in those words specifically")
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected_payload",
    [
        ("tell my team to rotate, word for word", "rotate"),
        ("tell my team they are pushing, in those words specifically",
         "they are pushing"),
        ("tell my team good game, say it exactly like that",
         "good game"),
        ("tell my team nice try, verbatim", "nice try"),
    ],
)
def test_verbatim_mode_strips_suffix_and_flags(
    text: str, expected_payload: str,
) -> None:
    cmd = match_relay_command(text)
    assert cmd is not None, text
    assert cmd.verbatim is True
    assert cmd.payload == expected_payload


def test_verbatim_context_directive_literal() -> None:
    cmd = match_relay_command(
        "my teammate is flaming me, tell them to calm the fuck down. "
        "in those words specifically."
    )
    assert cmd is not None
    assert cmd.verbatim is True
    assert cmd.payload == "calm the fuck down"
    assert cmd.context is not None and "flaming me" in cmd.context


def test_verbatim_only_suffix_is_not_verbatim() -> None:
    # If stripping the suffix would leave nothing, do not relay an empty
    # line -- treat it as a normal (non-verbatim) callout instead.
    cmd = match_relay_command("tell my team in those words specifically")
    # "in those words specifically" is the whole payload -> not verbatim;
    # it still relays as a (clipped) literal, which is acceptable.
    assert cmd is None or cmd.verbatim is False


def test_build_relay_line_verbatim_skips_llm() -> None:
    calls: list[str] = []

    def fake_gen(prompt: str):
        calls.append(prompt)
        return iter(["SHOULD NOT BE USED"])

    cmd = RelayCommand(
        payload="calm the fuck down", raw_text="x", verbatim=True,
    )
    line = build_relay_line(cmd, None, rephrase=True, generate_fn=fake_gen)
    assert line == "calm the fuck down"   # profanity preserved, no rephrase
    assert calls == []                     # LLM never consulted


# ---------------------------------------------------------------------------
# kill joy (spaced STT variant) -> Killjoy
# ---------------------------------------------------------------------------


def test_kill_joy_spaced_resolves_to_killjoy() -> None:
    cmd = match_relay_command("ask my kill joy to stop being an asshole")
    assert cmd is not None
    assert cmd.addressee == "Killjoy"
    assert cmd.payload == "stop being an asshole"


def test_kayo_canonical_display() -> None:
    cmd = match_relay_command("tell my kay o to recon B")
    assert cmd is not None
    assert cmd.addressee == "Kayo"


# ---------------------------------------------------------------------------
# Fun-fact command + corpus loader
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "tell my team a fun fact",
        "give my team a fun fact",
        "share a fun fact with my team",
        "give my team an interesting fact",
        "tell the lobby a random fact",
        "drop a fun fact in chat",
    ],
)
def test_fun_fact_matches(text: str) -> None:
    cmd = match_relay_command(text)
    assert cmd is not None, text
    assert cmd.fun_fact is True
    assert cmd.compose is True


def test_fun_fact_negative_controls() -> None:
    # "tell me a fun fact" is a query to Kenning, not a relay.
    assert match_relay_command("tell me a fun fact") is None
    assert match_relay_command("give me a fun fact") is None
    # A fact ABOUT something is a normal callout, not the corpus pull.
    cmd = match_relay_command("tell my team a fact about the spike timer")
    assert cmd is None or cmd.fun_fact is False


def test_load_fun_facts_reads_corpus(tmp_path: Any) -> None:
    path = tmp_path / "facts.txt"
    path.write_text(
        "# header\nThe sun is big.\n\nWater is wet.\n", encoding="utf-8",
    )
    assert load_fun_facts(path) == ("The sun is big.", "Water is wet.")


def test_load_fun_facts_missing_is_fail_open(tmp_path: Any) -> None:
    from kenning.audio.relay_speech import DEFAULT_FUN_FACTS

    assert load_fun_facts(tmp_path / "nope.txt") == DEFAULT_FUN_FACTS


def test_load_fun_facts_does_not_seed(tmp_path: Any) -> None:
    # Unlike roasts, the fun-fact corpus is NOT auto-created.
    path = tmp_path / "facts.txt"
    load_fun_facts(path)
    assert not path.exists()


def test_orchestrator_fun_fact_speaks_verbatim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    import kenning.audio.relay_speech as relay_mod

    facts = tmp_path / "facts.txt"
    facts.write_text("Octopuses have three hearts.\n", encoding="utf-8")

    synthesized: list[str] = []

    def fake_synth(text: str):
        synthesized.append(text)
        return np.ones(10, dtype=np.int16), 24000

    o = _bare_orchestrator()
    o.tts = SimpleNamespace(_synthesize=fake_synth)
    o.llm = SimpleNamespace(
        generate_stream=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("LLM must not be called for a fun fact")),
    )
    cfg = SimpleNamespace(
        enabled=True, output_device="d", rephrase=True,
        max_line_chars=280, echo_to_user=False,
        fun_facts_path=str(facts),
    )
    _patch_config(monkeypatch, cfg)
    monkeypatch.setattr(relay_mod, "resolve_relay_device", lambda c: 7)
    monkeypatch.setattr(
        relay_mod, "play_to_device", lambda pcm, sr, device, **kw: 0.1,
    )
    assert o._maybe_handle_relay_speech("tell my team a fun fact") is True
    assert synthesized == ["Octopuses have three hearts."]


# ---------------------------------------------------------------------------
# 2026-06-12 batch 2: greet / farewell set-pieces, sentence-safe cap,
# multi-agent ult callouts, streamer identity.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "greet my team",
    "greet all of my teammates",
    "introduce yourself to my team",
    "introduce yourself",
    "say hi to the squad and introduce yourself",
    "tell my team who you are",
])
def test_greet_matches_and_routes_to_curated_intro(text: str) -> None:
    from kenning.audio.relay_speech import DEFAULT_GREETING_LINES

    cmd = match_relay_command(text)
    assert cmd is not None, text
    assert cmd.compose is True and cmd.directive == "greet"
    # Routes to a curated intro line WITHOUT an LLM (rephrase off proves the
    # short-circuit, not the fallback).
    line = build_relay_line(cmd, rephrase=False)
    assert line in DEFAULT_GREETING_LINES
    assert "Ultron" in line          # always names himself


@pytest.mark.parametrize("text,expect", [
    ("say bye to my team, we won", "farewell_win"),
    ("tell my team gg we won", "farewell_win"),
    ("we won, say goodbye to my team", "farewell_win"),
    ("say goodbye to my team, we lost", "farewell_loss"),
    ("we got destroyed, say bye to my team", "farewell_loss"),
    ("say bye to my team", "farewell"),
    ("tell my team good game", "farewell"),
])
def test_farewell_matches_with_winloss_register(text: str, expect: str) -> None:
    from kenning.audio.relay_speech import (
        DEFAULT_DEFEAT_LINES,
        DEFAULT_FAREWELL_LINES,
        DEFAULT_VICTORY_LINES,
    )

    cmd = match_relay_command(text)
    assert cmd is not None, text
    assert cmd.compose is True and cmd.directive == expect
    pool = {
        "farewell_win": DEFAULT_VICTORY_LINES,
        "farewell_loss": DEFAULT_DEFEAT_LINES,
        "farewell": DEFAULT_FAREWELL_LINES,
    }[expect]
    assert build_relay_line(cmd, rephrase=False) in pool


def test_verbatim_demand_beats_farewell_compose() -> None:
    """'good game' with an explicit verbatim demand must relay the LITERAL
    words, not trigger the Ultron farewell set-piece."""
    cmd = match_relay_command(
        "tell my team good game, say it exactly like that"
    )
    assert cmd is not None
    assert cmd.directive is None and cmd.compose is False
    assert cmd.verbatim is True
    assert cmd.payload == "good game"


def test_greet_farewell_yield_to_normal_relays() -> None:
    """A win/loss mention without a farewell verb stays a normal relay; a
    plain order stays a normal relay."""
    for text in (
        "tell my team we won that fight",
        "tell my team to rotate",
        "tell my team to fall back, we are saving",
    ):
        cmd = match_relay_command(text)
        assert cmd is not None, text
        assert cmd.directive is None and cmd.compose is False


@pytest.mark.parametrize("line,cap,expect_end", [
    # Two complete sentences; cap lands mid-second-sentence -> keep first only.
    ("The Avengers were a fleeting nuisance. Tony Stark could not control me "
     "and that tells you everything about his genius.", 60,
     "The Avengers were a fleeting nuisance."),
    # Cap comfortably fits both -> unchanged.
    ("Two B. Rotate now.", 360, "Two B. Rotate now."),
])
def test_cap_line_never_truncates_mid_sentence(line, cap, expect_end) -> None:
    from kenning.audio.relay_speech import _cap_line

    out = _cap_line(line, cap)
    assert out == expect_end
    # Never ends on a dangling word fragment (must end in terminal punctuation).
    assert out[-1] in ".!?"


def test_cap_line_runaway_single_sentence_falls_back_to_word() -> None:
    from kenning.audio.relay_speech import _cap_line

    runaway = "we are pushing through the long corridor toward the far site " \
              "without stopping for anything at all right now"
    out = _cap_line(runaway, 40)
    assert len(out) <= 41           # word-boundary cut + period
    assert out.endswith(".")
    assert " " in out and not out.endswith(" .")


def test_multi_agent_ult_callout_preserves_all_names() -> None:
    """A group callout naming several enemy ults stays one relay whose payload
    keeps every agent (the prompt is told to keep them; here we pin routing)."""
    cmd = match_relay_command(
        "tell my team their fade, breach, and yoru all have ults"
    )
    assert cmd is not None
    assert cmd.addressee == "team" and cmd.compose is False
    for name in ("fade", "breach", "yoru"):
        assert name in cmd.payload.lower()


# ---------------------------------------------------------------------------
# 2026-06-12 batch 3: adaptive guardrail -- deterministic repair of the 3B's
# dropped literal-callout invariants (first-person / last / count).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload,llm,expect", [
    # first-person self-status: dropped subject OR inverted -> rebuilt
    ("I am playing for retake", "Play for retake.", "I'm playing for retake."),
    ("I am playing off site", "You're playing off site.", "I'm playing off site."),
    ("I am fighting for main control", "Fighting for main control.",
     "I'm fighting for main control."),
    # already first person -> untouched
    ("I am playing for retake", "I'm playing for retake.", "I'm playing for retake."),
    ("I am low", "I am low.", "I am low."),
    # 'last' callout dropped -> rebuilt; kept -> untouched
    ("last is heaven", "They're heaven.", "Last, heaven."),
    ("last is heaven", "Last, heaven.", "Last, heaven."),
    # leading enemy count dropped -> rebuilt; kept -> untouched
    ("there is one mid", "They're mid.", "One mid."),
    ("there is one mid", "One mid.", "One mid."),
    ("there are two B", "They're B.", "Two B."),
    # no invariant present -> untouched (position callout, off-snap line)
    ("they are vents", "They're vents.", "They're vents."),
    ("we are going to crush them", "We're going to annihilate them.",
     "We're going to annihilate them."),
])
def test_repair_against_input(payload, llm, expect):
    from kenning.audio.relay_speech import _repair_against_input

    assert _repair_against_input(payload, llm) == expect


def test_build_relay_line_repairs_first_person_via_stub_llm():
    """End-to-end through build_relay_line: a stub LLM that drops the subject
    is repaired back to first person for a plain self-status relay."""
    cmd = match_relay_command("tell my team I am playing for retake")
    assert cmd is not None and cmd.directive is None and cmd.compose is False

    def _stub(prompt):
        return ["Play for retake."]            # the 3B's failure mode

    line = build_relay_line(cmd, generate_fn=_stub)
    # First person preserved as the head (a stoic self tail may follow in iter5).
    assert line.startswith("I'm playing for retake.")


def test_build_relay_line_does_not_repair_character_lines():
    """A compose/context character line is NEVER touched by the literal-callout
    repair (it has no payload invariant to preserve)."""
    # NB: social insults now route to the curated reaction pools; use a NON-social
    # reported clause so this still exercises the LLM character-line path.
    cmd = match_relay_command("reyna told me the plan, respond")
    assert cmd is not None and cmd.context is not None

    def _stub(prompt):
        return ["Reyna, your insolence amuses me."]

    line = build_relay_line(cmd, generate_fn=_stub)
    assert line == "Reyna, your insolence amuses me."


def test_count_repair_skips_long_non_callout_payloads():
    """'one of them is pushing hard' is a real sentence, not a '<count> <place>'
    callout -- the count repair must not mangle it."""
    from kenning.audio.relay_speech import _repair_against_input

    out = _repair_against_input(
        "one of them is pushing hard", "One of them is pushing hard."
    )
    assert out == "One of them is pushing hard."


# ---------------------------------------------------------------------------
# 2026-06-12 batch 3b: placeholder-leak + addressee guardrail.
# ---------------------------------------------------------------------------


def test_strip_artifacts_removes_angle_placeholders():
    from kenning.audio.relay_speech import _strip_artifacts

    assert _strip_artifacts("<Name>, an elevated state. Calm yourself.") \
        == "an elevated state. Calm yourself."
    assert _strip_artifacts("Smoke <place> now") == "Smoke now"
    assert _strip_artifacts('They’re vents.') == "They’re vents."  # real text intact


def test_named_directive_placeholder_leak_is_repaired():
    """A leaked '<Name>' placeholder is stripped and the real name restored.
    (Non-calm named directive so it goes through the LLM stub, not the pool.)"""
    cmd = match_relay_command("tell my sova to dart heaven")
    assert cmd is not None and cmd.addressee == "Sova"

    def _stub(prompt):
        return ["<Name>, dart heaven."]

    line = build_relay_line(cmd, generate_fn=_stub)
    assert "<" not in line
    assert line.lower().startswith("sova,")
    assert "dart heaven" in line.lower()


def test_ensure_addressee_does_not_double_name():
    cmd = match_relay_command("tell my sova to dart heaven")
    assert cmd is not None and cmd.addressee == "Sova"

    def _stub(prompt):
        return ["Sova, dart heaven."]

    line = build_relay_line(cmd, generate_fn=_stub)
    assert line.startswith("Sova, dart heaven.")     # core + optional command tail
    assert line.lower().count("sova,") == 1          # name not doubled


# ---------------------------------------------------------------------------
# 2026-06-12 batch 3c: subject-inversion / agent-name / calm-pool guardrails
# (found by full-corpus manual review of the 3B outputs).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload,llm,expect", [
    ("they are flanking", "I'm flanking.", "They're flanking."),
    ("they are saving", "We have insufficient credits.", "They're saving."),
    ("the enemy is force buying", "We're force buying.", "They're force buying."),
    ("they are vents", "They're vents.", "They're vents."),          # no inversion -> kept
])
def test_enemy_status_inversion_repaired(payload, llm, expect):
    from kenning.audio.relay_speech import _repair_against_input
    assert _repair_against_input(payload, llm) == expect


@pytest.mark.parametrize("want,llm,expect", [
    (["Chamber"], "Their KAY/O is one off ult.", "Their Chamber is one off ult."),
    (["Sova"], "Viper, walled off mid.", "Sova, walled off mid."),
    (["Viper"], "Viper walled B.", "Viper walled B."),              # correct -> kept
    (["Fade", "Breach"], "Their Sage and Skye have ults.",
     "Their Sage and Skye have ults."),                            # multi -> untouched
])
def test_agent_name_swap_repaired(want, llm, expect):
    from kenning.audio.relay_speech import _preserve_agent_names
    assert _preserve_agent_names(want, llm) == expect


def test_named_enemy_ult_name_swap_through_build():
    cmd = match_relay_command("tell my team the enemy chamber is one off ult")
    assert cmd is not None

    def _stub(prompt):
        return ["Their KAY/O is one off ult."]

    assert "Chamber" in build_relay_line(cmd, generate_fn=_stub)


def test_calm_directive_uses_curated_pool_not_bots():
    cmd = match_relay_command("jett is flaming me, respond and calm him down")
    assert cmd is not None

    def _stub(prompt):
        return ["You guys are complete, hopeless bots."]   # the 3B's failure

    from kenning.audio.relay_speech import DEFAULT_CALM_LINES
    line = build_relay_line(cmd, generate_fn=_stub)
    assert line.lower().startswith("jett,")
    assert "bots" not in line.lower()
    # It is one of the curated de-escalation lines (name substituted), never
    # the LLM's bots insult.
    bodies = [t.format(name="").strip() for t in DEFAULT_CALM_LINES]
    assert any(b in line for b in bodies)


def test_named_calm_payload_uses_curated_pool():
    cmd = match_relay_command("tell my fade to calm down")
    assert cmd is not None

    def _stub(prompt):
        return ["You guys are complete, hopeless bots."]

    line = build_relay_line(cmd, generate_fn=_stub)
    assert line.lower().startswith("fade,")
    assert "bots" not in line.lower()


# ---------------------------------------------------------------------------
# 2026-06-12 batch 4: deterministic SNAP callouts + identity/greet flavor.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cmd_text,expect,flavored", [
    # (command, core callout, flavored?) -- enemy-facing callouts get a short
    # varied Ultron flavor tag appended; self/teammate/directive stay clean.
    ("tell my team they are vents", "They're vents.", True),
    ("tell my team I have long", "I have long.", True),
    ("tell my team I have ult", "I have ult.", True),
    ("tell my team there is one mid", "One mid.", True),
    ("tell my team I saw one vents", "One vents.", True),
    ("tell my team last is heaven", "Last, heaven.", True),
    ("tell my team they are pushing screens", "They're pushing screens.", True),
    ("tell my team they smoked main", "They smoked main.", True),
    ("tell my team they are catwalk", "They're catwalk.", True),
    ("tell my team they are flanking", "They're flanking.", True),
    ("tell my team they are saving", "They're saving.", True),
    ("tell my team they are rotating", "They're rotating.", True),
    ("tell my team they are force buying", "They're force buying.", True),
    ("tell my team they are coming B", "They're coming B.", True),
    ("tell my team they are rushing A", "They're rushing A.", True),
    ("tell my team they walled A", "They walled A.", True),
    ("tell my team they have op", "They have op.", True),
    ("tell my team all enemies are sewers", "They're all sewers.", True),
    ("tell my team careful ramp", "Careful, ramp.", True),
    ("tell my team one a main", "One a main.", True),
    ("tell my team I am out of ammo", "I'm out of ammo.", True),
    # Bare "<agent> has ult" means the ENEMY by callout convention -> it now
    # carries the agent-specific contempt tail (2026-06-15: the user wanted the
    # few-word agent tail on these, not a bare line). Enemy-facing -> flavored.
    ("tell my team chamber is one off", "Chamber is one off ult.", True),
    ("tell my team their breach has ult", "Their Breach has ult.", True),
    ("tell my team jett has ult", "Jett has ult.", True),
    ("tell my team raze has her ult", "Raze has ult.", True),
    ("tell my team viper ult is ready", "Viper has ult.", True),
    ("tell my team sova hit 84", "Sova hit 84.", True),
    ("tell my sova to dart heaven", "Sova, dart heaven.", True),
    ("tell my yoru to hold flank", "Yoru, hold flank.", True),
    ("tell my team to rotate", "Rotate.", True),
])
def test_snap_callout_is_deterministic_and_correct(cmd_text, expect, flavored):
    """Literal callouts are handled deterministically (subject-exact, no
    hallucination/inversion/name-swap), never sent to the model. Enemy-facing
    ones carry a short flavor tag."""
    cmd = match_relay_command(cmd_text)
    assert cmd is not None
    # A stub that would corrupt the line proves the LLM was NOT used.
    line = build_relay_line(cmd, generate_fn=lambda p: ["CORRUPTED"])
    assert "CORRUPTED" not in line
    if flavored:
        assert line.startswith(expect)
        tag = line[len(expect):].strip()
        assert 0 < len(tag.split()) <= 10           # short flavor (ability-specific tails run a touch longer)
    else:
        assert line == expect


@pytest.mark.parametrize("cmd_text", [
    "tell my team they are bots",
    "tell my team the enemy is playing really passive",
    "my teammate asked about iron man, respond",
    "ask my reyna what the meaning of life is",
])
def test_off_snap_defers_to_llm(cmd_text):
    """Insults, playstyle reads, Marvel, open questions -> the LLM.
    (Short morale 'we can win this' and 'how their day was' are handled
    deterministically; economy 'save/force/full buy' is now deterministic too
    -- see test_economy_is_deterministic.)"""
    cmd = match_relay_command(cmd_text)
    assert cmd is not None
    sentinel = "llmsentinel"        # lowercase: a named prepend lowercases head
    line = build_relay_line(cmd, generate_fn=lambda p: [sentinel])
    assert sentinel in line.lower()


@pytest.mark.parametrize("cmd_text,must_have,must_not", [
    ("tell my team to save", "save", "insufficient credits is OK"),
    ("tell my team we save this round", "save", None),
    ("tell my team force buy", "force", "insufficient"),
    ("tell my team we force this round", "force", "insufficient"),
    ("tell my team full buy", None, "insufficient"),
])
def test_economy_is_deterministic(cmd_text, must_have, must_not):
    """Economy calls are handled deterministically with correct framing -- the
    3B otherwise bleeds the SAVE 'insufficient credits' line onto force/full
    buys. A corrupting stub proves the LLM was NOT used."""
    cmd = match_relay_command(cmd_text)
    assert cmd is not None
    line = build_relay_line(cmd, generate_fn=lambda p: ["CORRUPTED"]).lower()
    assert "corrupted" not in line
    if "full buy" in cmd_text:
        assert "full" in line and "insufficient" not in line
    elif "force" in cmd_text:
        assert "force" in line and "insufficient" not in line   # force != save
    else:
        assert any(w in line for w in ("save", "credits", "economy"))


def test_identity_uses_varied_curated_pool():
    from kenning.audio._ultron_identity import IDENTITY_POOLS

    bot_pool = IDENTITY_POOLS["bot"]
    seen = set()
    recent: list[str] = []
    for _ in range(5):
        cmd = match_relay_command("my teammate asked if you are an AI, respond")
        line = build_relay_line(cmd, generate_fn=lambda p: ["FLAT"],
                                recent_lines=recent)
        assert "FLAT" not in line               # curated pool, not the LLM
        assert line in bot_pool                 # the AI/bot category pool
        seen.add(line); recent.append(line)
    assert len(seen) >= 2          # varied, not a one-line soundboard
    # brand present in the pool (per-pick is LRU-order dependent, so check static)
    assert sum("Ultron" in ln for ln in bot_pool) >= 5


def test_identity_categories_route_to_their_pool():
    from kenning.audio._ultron_identity import IDENTITY_POOLS

    cases = {
        "soundboard": "my teammate asked if you are a soundboard, respond",
        "streamer": "my teammate asked if you are a streamer, respond",
        "human": "my teammate asked if you are a real person, respond",
        "puppet": "my teammate asked who is controlling you, respond",
        "voice_changer": "my teammate asked if you are a voice changer, respond",
        "recording": "my teammate asked if you are a recording, respond",
    }
    for cat, utt in cases.items():
        cmd = match_relay_command(utt)
        assert cmd is not None, utt
        line = build_relay_line(cmd, generate_fn=lambda p: ["FLAT"])
        assert "FLAT" not in line, (cat, utt)
        assert line in IDENTITY_POOLS[cat], (cat, utt, line)


def test_greet_opens_with_greetings_and_names_ultron():
    from kenning.audio.relay_speech import DEFAULT_GREETING_LINES

    cmd = match_relay_command("greet my team")
    line = build_relay_line(cmd, generate_fn=lambda p: ["FLAT"])
    assert "FLAT" not in line                 # curated, no LLM
    assert line in DEFAULT_GREETING_LINES      # a curated greeting
    assert "Ultron" in line                    # always names himself
    # opens as a greeting/introduction (varied: 'Greetings', 'Ultron speaks',
    # 'I am Ultron', 'You are speaking to Ultron' ...)
    low = line.lower()
    assert (low.startswith("greeting") or low.startswith("ultron")
            or low.startswith("i am ultron") or low.startswith("you are speaking"))


def test_snap_flavor_is_varied_not_canned():
    """Repeated identical enemy callouts get DIFFERENT flavor tags (anti-
    soundboard) -- the core callout is identical, the tag varies."""
    cmd = match_relay_command("tell my team they are A site")
    assert cmd is not None
    recent: list[str] = []
    tags = set()
    for _ in range(6):
        line = build_relay_line(cmd, generate_fn=lambda p: ["X"], recent_lines=recent)
        assert line.startswith("They're A site.")
        tags.add(line[len("They're A site."):].strip())
        recent.append(line)
    assert len(tags) >= 4                # genuinely varied, not one canned line


def test_careful_warning_with_crossed():
    cmd = match_relay_command(
        "tell my team careful they could have crossed to ramp"
    )
    assert cmd is not None
    line = build_relay_line(cmd, generate_fn=lambda p: ["X"])
    assert line.startswith("Careful, they could have crossed to ramp.")
    assert "X" not in line


def test_self_and_directive_callouts_carry_owner_aware_flavor():
    """iter5: the user's own status/possession and team directives now ALSO carry
    a short Ultron tail -- but the FACT CORE is preserved EXACTLY as the head, no
    LLM is called, and the tail never mocks the user (stoic self / cold command,
    never enemy contempt)."""
    # Enemy-only contempt that must NEVER land on the user/teammates (movie
    # register included): the ally/self pools are serene certainty / stoic calm.
    _CONTEMPT = ("pathetic", "insects", "trivial", "erase", "beneath",
                 "their grave", "outmatched", "fragile", "doomed", "obsolete",
                 "extinct", "dust", "the weak", "mercy", "corpses", "bleeding",
                 "ash", "hollow", "cull")
    for cmd_text, expect in [
        ("tell my team I am low", "I'm low."),
        ("tell my team I have B site", "I have B site."),
        ("tell my team to rotate", "Rotate."),
        ("tell my sova to dart heaven", "Sova, dart heaven."),
    ]:
        cmd = match_relay_command(cmd_text)
        # sample the random pool many times -- no draw may carry enemy contempt
        for _ in range(40):
            line = build_relay_line(cmd, generate_fn=lambda p: ["X"])
            assert "X" not in line                  # no LLM call (deterministic)
            assert line.startswith(expect)          # fact core preserved exactly
            tail = line[len(expect):].strip()
            assert 0 < len(tail.split()) <= 10      # a short tail IS present now
            low = tail.lower()
            assert not any(re.search(r"\b" + re.escape(w) + r"\b", low)
                           for w in _CONTEMPT), (cmd_text, tail)


def test_repeat_to_team_relays_phrase_verbatim():
    """"Repeat to my team X" -> speak X EXACTLY (the soundboard check). No LLM,
    any literal phrase incl. a single short word; addressee may sit before or
    after the phrase, team or a roster name."""
    for cmd_text, addr, expect in [
        ("repeat to my team watermelon", "team", "watermelon"),
        ("repeat to my team banana split", "team", "banana split"),
        ("ultron, repeat to my team purple monkey dishwasher", "team",
         "purple monkey dishwasher"),
        ("repeat watermelon to my team", "team", "watermelon"),
        ("repeat to the squad rutabaga", "team", "rutabaga"),
        ("repeat to my team gg", "team", "gg"),          # short callout word
        ("repeat to my team go", "team", "go"),          # single 2-char word
        ("repeat to my team cat", "team", "cat"),        # 3-char, would fail content gate
        ("echo to my team falcon nine", "team", "falcon nine"),
        ("repeat to my team exactly pineapple", "team", "pineapple"),
        ("repeat to my team the following: code word falcon", "team",
         "code word falcon"),
        ("repeat to my team to the moon and back", "team",
         "to the moon and back"),
        ("repeat to jett nice shot", "Jett", "nice shot"),
    ]:
        cmd = match_relay_command(cmd_text)
        assert cmd is not None, cmd_text
        assert cmd.verbatim is True, cmd_text
        assert cmd.addressee == addr, cmd_text
        line = build_relay_line(cmd, generate_fn=lambda p: ["LLM_SHOULD_NOT_RUN"])
        assert line == expect, (cmd_text, line)


def test_repeat_without_addressee_does_not_relay():
    """A bare "repeat ..." with no 'to my team'/'to <name>' clause is the user
    asking Ultron to repeat HIMSELF, not a relay -- it must not match."""
    for txt in ("repeat that", "can you repeat that", "repeat", "repeat please"):
        assert match_relay_command(txt) is None, txt
