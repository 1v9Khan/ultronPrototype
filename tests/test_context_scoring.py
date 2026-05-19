"""Tests for :mod:`ultron.llm.context_scoring`."""

from __future__ import annotations

import pytest

from ultron.llm.context_scoring import (
    ContextRecommendation,
    score_context,
)


# ---------------------------------------------------------------------------
# Empty + degenerate input
# ---------------------------------------------------------------------------


def test_empty_text_returns_defaults() -> None:
    rec = score_context("")
    assert rec.history_turns == 4
    assert rec.retrieval_k == 5
    assert rec.suppress_rag is False
    assert rec.reason == "empty utterance"


def test_whitespace_only_returns_defaults() -> None:
    rec = score_context("   \t\n")
    assert rec.history_turns == 4
    assert rec.retrieval_k == 5


# ---------------------------------------------------------------------------
# Short factual queries
# ---------------------------------------------------------------------------


def test_short_factual_stem_suppresses_retrieval() -> None:
    rec = score_context("what time is it")
    assert rec.retrieval_k == 0
    assert rec.suppress_rag is True
    assert rec.history_turns <= 2
    assert "factual stem" in rec.reason


def test_short_factual_who_question_suppresses_retrieval() -> None:
    rec = score_context("who wrote Hamlet")
    assert rec.retrieval_k == 0
    assert rec.history_turns <= 2


# ---------------------------------------------------------------------------
# Long technical questions
# ---------------------------------------------------------------------------


def test_long_utterance_raises_budget() -> None:
    text = (
        "I want to refactor the entire authentication subsystem so that "
        "instead of passing the user token through the middleware chain "
        "manually we have a dedicated session context object that "
        "subsystems request explicitly via dependency injection. Walk "
        "me through how you'd plan that change."
    )
    rec = score_context(text)
    assert rec.history_turns >= 5
    assert rec.retrieval_k >= 6
    assert (
        "long utterance" in rec.reason
        or "depth marker" in rec.reason
    )


def test_depth_marker_alone_lifts_budget() -> None:
    # "Walk me through" without much length still lifts defaults.
    rec = score_context("walk me through promises")
    assert rec.history_turns >= 5
    assert rec.retrieval_k >= 6


# ---------------------------------------------------------------------------
# Reference / pronoun-heavy queries
# ---------------------------------------------------------------------------


def test_pronoun_reference_boosts_history() -> None:
    rec = score_context("open it now")
    assert rec.history_turns >= 6


def test_explicit_back_reference_boosts_history() -> None:
    rec = score_context("Go back to the one we discussed earlier")
    assert rec.history_turns >= 6


# ---------------------------------------------------------------------------
# Topic-shift markers
# ---------------------------------------------------------------------------


def test_topic_shift_zeroes_history() -> None:
    rec = score_context("by the way what is the capital of france")
    assert rec.history_turns == 0
    assert "topic-shift" in rec.reason


def test_topic_shift_overrides_other_signals() -> None:
    # Even with a "remember when" marker, topic-shift wins -- the user
    # explicitly wants to abandon the prior thread.
    rec = score_context("different question - do you remember when we set up postgres")
    assert rec.history_turns == 0


# ---------------------------------------------------------------------------
# Personal recall
# ---------------------------------------------------------------------------


def test_personal_recall_boosts_history() -> None:
    rec = score_context("do you remember what we decided about the deployment")
    assert rec.history_turns >= 6


def test_personal_recall_preserves_retrieval() -> None:
    # Personal recall != short factual; retrieval should still fire.
    rec = score_context("what did i say earlier about the schema")
    assert rec.retrieval_k > 0


# ---------------------------------------------------------------------------
# Active coding task
# ---------------------------------------------------------------------------


def test_active_task_boosts_history() -> None:
    rec = score_context("how is it going", has_active_task=True)
    # Active task signal raises history above the short-utterance floor.
    assert rec.history_turns >= 5


# ---------------------------------------------------------------------------
# Clamp behaviour
# ---------------------------------------------------------------------------


def test_clamping_respects_max_history_turns() -> None:
    rec = score_context(
        "walk me through the architecture, the one we discussed",
        max_history_turns=3,
    )
    assert rec.history_turns <= 3


def test_clamping_respects_min_retrieval_k() -> None:
    rec = score_context(
        "what time is it",
        min_retrieval_k=2,
    )
    assert rec.retrieval_k >= 2
    assert rec.suppress_rag is False  # min_retrieval_k=2 => suppress=False


def test_custom_defaults_propagate() -> None:
    # Use a depth-marker query so the depth branch fires and lifts both
    # budgets above the caller-supplied defaults.
    rec = score_context(
        "explain in detail how quantum entanglement actually works",
        default_history_turns=2,
        default_retrieval_k=3,
    )
    assert rec.history_turns >= 3  # default + 1 from depth
    assert rec.retrieval_k >= 4    # default + 1 from depth


# ---------------------------------------------------------------------------
# Recommendation API
# ---------------------------------------------------------------------------


def test_recommendation_fixed_factory() -> None:
    rec = ContextRecommendation.fixed(history_turns=7, retrieval_k=0)
    assert rec.history_turns == 7
    assert rec.retrieval_k == 0
    assert rec.suppress_rag is True


def test_recommendation_suppress_flag_matches_retrieval_zero() -> None:
    rec = score_context("what time is it")
    assert (rec.retrieval_k == 0) == (rec.suppress_rag is True)


def test_recommendation_is_frozen() -> None:
    rec = score_context("hello")
    with pytest.raises(Exception):  # FrozenInstanceError
        rec.history_turns = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Smoke: no exception on a battery of varied inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "utterance",
    [
        "yes",
        "no",
        "stop",
        "ok",
        "what time is it",
        "tell me a joke about cats",
        "remember when we configured nginx",
        "write a python function to compute fibonacci numbers please",
        "actually let's talk about something else now",
        "those buttons are broken",
        "explain how transformers work step by step",
        "by the way is the weather okay today",
    ],
)
def test_smoke_no_exception(utterance: str) -> None:
    rec = score_context(utterance)
    assert 0 <= rec.history_turns <= 8
    assert 0 <= rec.retrieval_k <= 10
    assert isinstance(rec.reason, str)
