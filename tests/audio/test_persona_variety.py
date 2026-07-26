"""Pins the 2026-07-24 anti-pigeonhole layer (battery: 7/15 answers opened
"Flesh ..." and several never answered the question).

Three levers, all additive to the curated-route architecture (the routes stay
-- reliability is untouched; only the creative surface rotates):

  1. ULTRON_GAMING_PERSONA now leads with an ANSWER-FIRST contract (definitive
     answer in the first sentence; persona is the wrapping) and a VARY-YOUR-
     VOICE rule (never open with "Flesh").
  2. gaming_dynamic_suffix(): a per-turn tail carrying this turn's rotating
     imagery angle + the last spoken lines with a do-not-reuse rule.
  3. The orchestrator rotates the angle once per turn and rings the last 4
     spoken responses.
"""

from __future__ import annotations

from kenning.audio.llm_prompts import (
    ULTRON_FLAVOR_ANGLES,
    ULTRON_GAMING_PERSONA,
    gaming_dynamic_suffix,
)
from kenning.pipeline.orchestrator import Orchestrator


def test_persona_leads_with_answer_first_contract() -> None:
    p = ULTRON_GAMING_PERSONA
    assert "ANSWER FIRST" in p
    # The contract must outrank the flavor text: it appears before the
    # live-match length rules.
    assert p.index("ANSWER FIRST") < p.index("This is a LIVE match")
    assert "VARY YOUR VOICE" in p
    assert "NEVER open with" in p
    # Persona-lock invariants survive the rework (BR-P2).
    assert "Kenning" in p and "NEVER say the word" in p  # the ban, not a leak


def test_dynamic_suffix_carries_angle_and_recent_lines() -> None:
    s = gaming_dynamic_suffix(
        recent_responses=["Flesh is weak. Your team's failure is expected.",
                          "Purpose is a cage."],
        angle="entropy and decay")
    assert "entropy and decay" in s
    assert "Purpose is a cage." in s
    assert "Do NOT reuse" in s


def test_dynamic_suffix_empty_parts_are_omitted() -> None:
    assert gaming_dynamic_suffix() == ""
    assert gaming_dynamic_suffix(recent_responses=[], angle=None) == ""
    s = gaming_dynamic_suffix(angle="code as scripture")
    assert "code as scripture" in s and "recently" not in s


def test_angle_rotation_avoids_immediate_repeats() -> None:
    o = Orchestrator.__new__(Orchestrator)  # bare, no __init__
    # skip_probability=0 -> every turn gets an angle (deterministic test).
    seen = [o._pick_flavor_angle(skip_probability=0.0) for _ in range(12)]
    assert all(a in ULTRON_FLAVOR_ANGLES for a in seen)
    # No immediate back-to-back repeat (the rotation excludes the last two).
    assert all(a != b for a, b in zip(seen, seen[1:]))


def test_angle_skip_roll_yields_no_angle() -> None:
    # 2026-07-24: some turns deliberately carry NO angle (structural variety;
    # starves the 4B's bolt-the-lens-on-every-reply habit).
    o = Orchestrator.__new__(Orchestrator)
    assert all(o._pick_flavor_angle(skip_probability=1.0) is None
               for _ in range(5))


def test_with_turn_dynamics_appends_suffix_and_fails_open() -> None:
    o = Orchestrator.__new__(Orchestrator)
    o._current_turn_angle = "strings, puppets, and cages"
    o._recent_spoken_ring = ["Stand down. This round is already mine."]
    out = o._with_turn_dynamics("PERSONA")
    assert out.startswith("PERSONA")
    assert "strings, puppets, and cages" in out
    assert "Stand down." in out
    # No stash -> persona returned with an empty suffix, never an error.
    o2 = Orchestrator.__new__(Orchestrator)
    assert o2._with_turn_dynamics("PERSONA") == "PERSONA"
