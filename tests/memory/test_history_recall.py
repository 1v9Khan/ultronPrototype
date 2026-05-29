"""Tests for conversation-history recall ("what did I say earlier?").

Two layers:
* match_history_recall -- the strict matcher (subject + recall-verb gate).
* Orchestrator._maybe_handle_history_recall -- speaks the matching verbatim
  turn from the in-memory dual-history store, excluding the current query.
"""

from __future__ import annotations

import pytest

from ultron.memory.history_recall import (
    ROLE_ASSISTANT,
    ROLE_USER,
    match_history_recall,
)


# --- matcher ----------------------------------------------------------------


def test_match_user_recall_with_topic():
    m = match_history_recall("what did I say earlier about the database?")
    assert m is not None
    assert m.role == ROLE_USER
    assert m.topic == "database"


def test_match_assistant_recall_with_topic():
    m = match_history_recall("what did you tell me about the API?")
    assert m is not None
    assert m.role == ROLE_ASSISTANT
    assert m.topic == "API"


def test_match_remind_me_form():
    m = match_history_recall("remind me what I asked about postgres")
    assert m is not None
    assert m.role == ROLE_USER
    assert m.topic == "postgres"


def test_match_did_i_mention_form():
    m = match_history_recall("did I mention the meeting?")
    assert m is not None
    assert m.role == ROLE_USER
    assert m.topic == "meeting"


def test_match_topicless():
    m = match_history_recall("what did I say earlier?")
    assert m is not None
    assert m.role == ROLE_USER
    assert m.topic == ""


@pytest.mark.parametrize(
    "text",
    [
        "what should I do about this?",
        "what did you mean?",
        "what did I miss?",
        "tell me about the weather",
        "how do I make pasta?",
        "",
        "   ",
    ],
)
def test_non_recall_returns_none(text):
    assert match_history_recall(text) is None


# --- handler ----------------------------------------------------------------


def _orch():
    from ultron.memory.dual_history import DualHistoryStore
    from ultron.pipeline.orchestrator import Orchestrator

    o = Orchestrator.__new__(Orchestrator)
    o._dual_history = DualHistoryStore(verbatim_cap=300)
    spoken: list[str] = []
    o._speak = lambda text: spoken.append(text)  # type: ignore[assignment]
    return o, spoken


def test_handler_speaks_matching_user_turn():
    o, spoken = _orch()
    o._record_dialogue_turn("user", "we should use postgres for the database")
    o._record_dialogue_turn("assistant", "Good choice.")
    # The loop records the query turn before the recall check; simulate that.
    query = "what did I say earlier about the database"
    o._record_dialogue_turn("user", query)

    handled = o._maybe_handle_history_recall(query)
    assert handled is True
    assert len(spoken) == 1
    assert spoken[0].startswith("Earlier you said:")
    assert "postgres" in spoken[0].lower()


def test_handler_speaks_matching_assistant_turn():
    o, spoken = _orch()
    o._record_dialogue_turn("assistant", "The capital of France is Paris.")
    query = "what did you say about france"
    o._record_dialogue_turn("user", query)

    handled = o._maybe_handle_history_recall(query)
    assert handled is True
    assert spoken[0].startswith("Earlier I said:")
    assert "Paris" in spoken[0]


def test_handler_no_record_speaks_apology():
    o, spoken = _orch()
    o._record_dialogue_turn("user", "hello there")
    query = "what did I say about quantum physics"
    o._record_dialogue_turn("user", query)

    handled = o._maybe_handle_history_recall(query)
    assert handled is True
    assert "don't have a record" in spoken[0].lower()


def test_handler_excludes_the_current_query_turn():
    """The query itself contains the topic word; it must not match itself."""
    o, spoken = _orch()
    query = "what did I say earlier about database"
    o._record_dialogue_turn("user", query)  # ONLY the query is recorded

    handled = o._maybe_handle_history_recall(query)
    assert handled is True
    # No prior turn -> apology, NOT an echo of the query.
    assert "don't have a record" in spoken[0].lower()


def test_handler_returns_false_for_non_recall():
    o, spoken = _orch()
    assert o._maybe_handle_history_recall("what should I cook for dinner") is False
    assert spoken == []


def test_handler_disabled_via_config_returns_false(monkeypatch):
    from ultron.config import get_config

    monkeypatch.setattr(get_config().memory, "history_recall_enabled", False)
    o, spoken = _orch()
    o._record_dialogue_turn("user", "we discussed the database")
    assert o._maybe_handle_history_recall("what did I say about the database") is False
    assert spoken == []
