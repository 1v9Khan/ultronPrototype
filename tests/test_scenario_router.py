"""Pins the scenario taxonomy + one-shot router (2026-07-26).

Routing was an ordered chain of ~24 ``_maybe_handle_*`` matchers where order
IS semantics -- whichever regex fires first wins, so a phrasing that trips an
earlier handler is silently stolen from its real one. The router asks a small
model which of 29 scenarios the utterance is, once, and dispatches directly.

These tests cover the pure logic with NO model: the taxonomy's internal
consistency, the tolerant label parser, and -- most importantly -- the
FAIL-SAFE posture, because the whole design rests on the claim that a wrong or
missing classifier costs latency and never correctness.

Measured accuracy (``scripts/relay_test/scenario_scorecard.py`` over the
144-case labelled corpus): Qwen3-4B heretic Q5 = 140/144 (97.2%), warm median
125 ms. Llama-3.2-3B = 82.6%, the 1Bs 25-38% -- a 29-way choice is past what a
1B does reliably, and the 3B saves only 12 ms for 15 accuracy points.
"""

from __future__ import annotations

import pytest

from kenning.audio import scenario_router as sr
from kenning.audio.scenario_taxonomy import (
    CONTROL_SCENARIOS,
    CONVERSATIONAL_SCENARIOS,
    DESTRUCTIVE_SCENARIOS,
    SCENARIOS,
    Scenario,
    all_labels,
    handler_for,
    scenario_by_value,
)


# ---------------------------------------------------------------------------
# Taxonomy integrity
# ---------------------------------------------------------------------------

def test_every_scenario_has_a_spec() -> None:
    missing = [s.value for s in Scenario if s not in SCENARIOS]
    assert not missing, f"scenarios with no spec: {missing}"


def test_every_spec_has_description_examples_and_handler() -> None:
    for s, spec in SCENARIOS.items():
        assert spec.description.strip(), f"{s.value}: empty description"
        assert spec.examples, f"{s.value}: no examples"
        assert spec.handler.startswith("_maybe_handle"), (
            f"{s.value}: handler {spec.handler!r} is not a dispatch method")


def test_labels_are_unique_and_prompt_safe() -> None:
    """Labels go into a grammar and a prompt -- duplicates or whitespace would
    make the model's output ambiguous to parse."""
    labels = all_labels()
    assert len(labels) == len(set(labels))
    for lab in labels:
        assert lab == lab.strip().lower()
        assert " " not in lab and '"' not in lab


def test_destructive_set_is_derived_not_hardcoded() -> None:
    expected = {s for s, spec in SCENARIOS.items() if spec.destructive}
    assert DESTRUCTIVE_SCENARIOS == expected
    # The two that must never be auto-dispatched on a classifier's say-so.
    assert Scenario.TWITCH_MODERATION in DESTRUCTIVE_SCENARIOS
    assert Scenario.SCRAP_COMMAND in DESTRUCTIVE_SCENARIOS


def test_conversational_scenarios_share_the_private_reply_handler() -> None:
    for s in CONVERSATIONAL_SCENARIOS:
        assert handler_for(s) == "_maybe_handle_private_reply"


def test_control_scenarios_are_all_real() -> None:
    for s in CONTROL_SCENARIOS:
        assert s in SCENARIOS


# ---------------------------------------------------------------------------
# Label parsing -- must tolerate what small models actually emit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,want", [
    ("relay_team", Scenario.RELAY_TEAM),
    ("  RELAY_TEAM  ", Scenario.RELAY_TEAM),
    ('"tell_chat"', Scenario.TELL_CHAT),
    ("spotify.", Scenario.SPOTIFY),
    ("answer-question", Scenario.ANSWER_QUESTION),
    ("answer question", Scenario.ANSWER_QUESTION),
    ("`ignore`", Scenario.IGNORE),
    ("nonsense", None),
    ("", None),
    (None, None),
])
def test_scenario_by_value_is_tolerant_but_not_guessy(raw, want) -> None:
    assert scenario_by_value(raw) is want


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def test_system_prompt_names_every_label() -> None:
    p = sr.build_classifier_system_prompt()
    for lab in all_labels():
        assert lab in p, f"{lab} missing from the classifier prompt"


def test_system_prompt_is_byte_stable() -> None:
    """It is the prefix-cached half. If it varies between calls the cache
    misses every turn and the router costs a full re-prefill."""
    assert sr.build_classifier_system_prompt() == \
        sr.build_classifier_system_prompt()


def test_user_prompt_is_small() -> None:
    """The per-turn cost is this string; the taxonomy rides the cache."""
    u = sr.build_classifier_user_prompt("tell my team to push A now")
    assert len(u) < 120
    assert "tell my team to push A now" in u


# ---------------------------------------------------------------------------
# FAIL-SAFE posture -- the claim the whole design rests on
# ---------------------------------------------------------------------------

def _router(reply, **kw):
    def _gen(*, system, user, max_tokens):
        if isinstance(reply, Exception):
            raise reply
        return reply
    return sr.ScenarioRouter(_gen, **kw)


def test_clean_label_is_actionable() -> None:
    v = _router("relay_team", shadow=False).classify("tell the team to push")
    assert v.scenario is Scenario.RELAY_TEAM
    assert v.actionable is True


def test_scaffolded_label_still_parses() -> None:
    """Small models prefix their own scaffolding; take the last valid token."""
    v = _router("Label: tell_chat", shadow=False).classify("tell chat hi")
    assert v.scenario is Scenario.TELL_CHAT
    assert v.actionable is True


def test_unparsed_label_is_not_actionable() -> None:
    v = _router("I think they want to talk to their team",
                shadow=False).classify("x")
    assert v.scenario is None
    assert v.actionable is False


def test_generator_error_never_raises_and_is_not_actionable() -> None:
    v = _router(RuntimeError("model died"), shadow=False).classify("x")
    assert v.scenario is None
    assert v.actionable is False
    assert "error" in v.reason


def test_destructive_scenario_is_never_actionable() -> None:
    """Ban / scrap must go through the chain, which has the two-phase
    confirm -- a classifier must not be able to trigger them directly."""
    for s in DESTRUCTIVE_SCENARIOS:
        v = _router(s.value, shadow=False).classify("ban that guy")
        assert v.scenario is s
        assert v.actionable is False, f"{s.value} was actionable"
        assert "destructive" in v.reason


def test_shadow_mode_classifies_but_never_acts() -> None:
    v = _router("relay_team", shadow=True).classify("tell the team to push")
    assert v.scenario is Scenario.RELAY_TEAM
    assert v.actionable is False
    assert "shadow" in v.reason


def test_env_gates(monkeypatch) -> None:
    monkeypatch.delenv("KENNING_SCENARIO_ROUTER", raising=False)
    assert sr.router_enabled() is False, "router must default OFF"
    monkeypatch.setenv("KENNING_SCENARIO_ROUTER", "1")
    assert sr.router_enabled() is True

    monkeypatch.delenv("KENNING_SCENARIO_ROUTER_SHADOW", raising=False)
    assert sr.router_shadow_mode() is True, "shadow must default ON"
    monkeypatch.setenv("KENNING_SCENARIO_ROUTER_SHADOW", "0")
    assert sr.router_shadow_mode() is False


def test_stats_track_parse_rate() -> None:
    r = _router("relay_team", shadow=False)
    for _ in range(3):
        r.classify("x")
    st = r.stats()
    assert st["calls"] == 3 and st["parsed"] == 3 and st["unparsed"] == 0
    assert st["mean_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Anticheat (BR-P1): this module sits on the voice path.
# ---------------------------------------------------------------------------

def test_router_modules_import_no_heavy_deps() -> None:
    """The LLM is INJECTED precisely so importing the router never pulls
    llama-cpp (or torch) onto the voice path."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "src" / "kenning"
    banned = ("llama_cpp", "torch", "sentence_transformers", "transformers")
    for name in ("audio/scenario_router.py", "audio/scenario_taxonomy.py"):
        src = (root / name).read_text(encoding="utf-8")
        for line in src.splitlines():
            st = line.strip()
            if st.startswith("import ") or st.startswith("from "):
                for b in banned:
                    assert b not in st, f"{name} imports {b}: {line!r}"
