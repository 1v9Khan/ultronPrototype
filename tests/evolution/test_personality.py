"""Tests for ultron.evolution.personality -- adaptive response temperament.
All hermetic."""

from __future__ import annotations

from ultron.evolution.models import PersonalityState
from ultron.evolution.personality import (
    PersonalityFeedback,
    PersonalityTuner,
    apply_temperament,
    temperament_hint,
)


# --- feedback ---------------------------------------------------------------


def test_feedback_satisfied():
    assert PersonalityFeedback().satisfied is True
    assert PersonalityFeedback(corrected=True).satisfied is False
    assert PersonalityFeedback(barged_in=True).satisfied is False


# --- hint -------------------------------------------------------------------


def test_temperament_hint_balanced_is_empty():
    assert temperament_hint(PersonalityState.balanced()) == ""


def test_temperament_hint_concise():
    h = temperament_hint(PersonalityState(verbosity=0.2))
    assert "concise" in h
    assert h.startswith("[Tone:")


def test_temperament_hint_thorough_and_precise():
    h = temperament_hint(PersonalityState(verbosity=0.8, rigor=0.8))
    assert "thorough" in h
    assert "precise" in h


def test_temperament_hint_creative():
    assert "creativity" in temperament_hint(PersonalityState(creativity=0.9))


def test_apply_temperament_prepends_and_is_idempotent():
    state = PersonalityState(verbosity=0.2)
    out = apply_temperament("what is the capital of france", state)
    assert out.startswith("[Tone:")
    assert "capital of france" in out
    # idempotent
    assert apply_temperament(out, state) == out


def test_apply_temperament_noop_balanced_or_empty():
    assert apply_temperament("hello", PersonalityState.balanced()) == "hello"
    assert apply_temperament("", PersonalityState(verbosity=0.1)) == ""


# --- record_feedback nudges -------------------------------------------------


def test_correction_raises_rigor_lowers_risk():
    t = PersonalityTuner()
    new = t.record_feedback(PersonalityFeedback(corrected=True))
    assert new.rigor > 0.5
    assert new.risk_tolerance < 0.5


def test_barge_in_lowers_verbosity():
    t = PersonalityTuner()
    new = t.record_feedback(PersonalityFeedback(barged_in=True))
    assert new.verbosity < 0.5


def test_re_ask_raises_verbosity():
    t = PersonalityTuner()
    new = t.record_feedback(PersonalityFeedback(re_asked=True))
    assert new.verbosity > 0.5


def test_satisfied_regresses_toward_balanced():
    t = PersonalityTuner(state=PersonalityState(rigor=0.9, verbosity=0.1))
    new = t.record_feedback(PersonalityFeedback())  # satisfied
    assert new.rigor < 0.9  # moved toward 0.5
    assert new.verbosity > 0.1  # moved toward 0.5


def test_max_drift_bounds_a_single_step():
    t = PersonalityTuner(learning_rate=0.5, max_drift=0.05)
    new = t.record_feedback(PersonalityFeedback(corrected=True))
    assert new.rigor == 0.55  # bounded by max_drift, not 1.0


def test_traits_stay_clamped():
    t = PersonalityTuner(state=PersonalityState(risk_tolerance=0.02), learning_rate=0.5, max_drift=0.5)
    new = t.record_feedback(PersonalityFeedback(corrected=True))
    assert 0.0 <= new.risk_tolerance <= 1.0


def test_current_hint_reflects_state():
    t = PersonalityTuner(state=PersonalityState(verbosity=0.1))
    assert "concise" in t.current_hint()


# --- outcome ranking --------------------------------------------------------


def test_best_personality_picks_highest_mean():
    t = PersonalityTuner()
    good = PersonalityState(verbosity=0.8)
    bad = PersonalityState(verbosity=0.2)
    for _ in range(3):
        t.record_outcome(0.9, state=good)
        t.record_outcome(0.2, state=bad)
    best = t.best_personality()
    assert best is not None
    assert best.verbosity == 0.8


def test_best_personality_requires_min_samples():
    t = PersonalityTuner()
    t.record_outcome(0.9, state=PersonalityState(verbosity=0.8))
    assert t.best_personality() is None  # only 1 sample, need >= 3


def test_report():
    t = PersonalityTuner()
    assert "no outcome data" in t.report()
    for _ in range(3):
        t.record_outcome(0.8, state=PersonalityState(rigor=0.7))
    r = t.report()
    assert "rigor=" in r


# --- persistence ------------------------------------------------------------


def test_to_from_dict_round_trip():
    t = PersonalityTuner(state=PersonalityState(rigor=0.7, verbosity=0.3))
    d = t.to_dict()
    t2 = PersonalityTuner.from_dict(d)
    assert t2.state.rigor == 0.7
    assert t2.state.verbosity == 0.3


def test_from_dict_malformed_falls_back():
    t = PersonalityTuner.from_dict({"rigor": "bad"})
    # clamp01 turns "bad" into 0.0 rather than raising
    assert 0.0 <= t.state.rigor <= 1.0
