"""2026-05-19 round 4: brevity-hint prefix must not bypass the
short-query memory suppressor.

Live session 'Thanks.' -> Berlin weather. The orchestrator wraps the
user text in ``apply_brevity_hint`` BEFORE handing it to the LLM, so
the LLM sees::

    [Style: respond in 1-3 short sentences. ...]

    Thanks.

The pre-existing ``_is_short_conversational_query`` checked the
hinted text and saw a long string ("[Style:..." is > 4 tokens) -> the
gate stayed off -> recent-turn history + RAG flooded the prompt ->
the model replayed an old assistant turn about Berlin weather.

Fix: strip the ``[Style: ...]`` prefix before evaluating.
"""

from __future__ import annotations

import pytest

from ultron.llm.inference import (
    _is_short_conversational_query,
    _strip_brevity_hint,
)
from ultron.response_style import apply_brevity_hint


# ---------------------------------------------------------------------------
# _strip_brevity_hint
# ---------------------------------------------------------------------------


def test_strip_brevity_hint_removes_prefix():
    text = "[Style: respond in 1-3 short sentences.]\n\nThanks."
    assert _strip_brevity_hint(text) == "Thanks."


def test_strip_brevity_hint_removes_multiline_directive():
    text = (
        "[Style: respond with detailed numbered steps. "
        "Do not summarise. Include specific measurements.]"
        "\n\nGive me a cake recipe."
    )
    out = _strip_brevity_hint(text)
    assert out == "Give me a cake recipe."


def test_strip_brevity_hint_passthrough_on_unhinted_text():
    assert _strip_brevity_hint("Thanks.") == "Thanks."
    assert _strip_brevity_hint("what time is it?") == "what time is it?"


def test_strip_brevity_hint_handles_empty_and_none():
    assert _strip_brevity_hint("") == ""
    assert _strip_brevity_hint(None) is None


def test_strip_brevity_hint_idempotent():
    text = "[Style: brief.]\n\nHi."
    once = _strip_brevity_hint(text)
    twice = _strip_brevity_hint(once)
    assert once == twice == "Hi."


# ---------------------------------------------------------------------------
# End-to-end: a hinted greeting still triggers the short-query gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("greeting", [
    "hello",
    "Hello.",
    "hi",
    "Hi!",
    "thanks",
    "Thanks.",
    "ok",
    "good morning",
    "say hello",
    "and say hello",
    "ok cool",
])
def test_hinted_greeting_still_classifies_as_short(greeting):
    hinted = apply_brevity_hint(greeting)
    # Whether the hint actually fires depends on is_brief_question;
    # the gate must work either way.
    assert _is_short_conversational_query(hinted) is True, (
        f"hinted greeting {hinted!r} should classify as short "
        f"(strip should remove the prefix)"
    )


def test_explicitly_hinted_greeting_still_short():
    """Construct the exact prefix shape and verify."""
    text = "[Style: respond in 1-3 short sentences.]\n\nThanks."
    assert _is_short_conversational_query(text) is True


def test_explicitly_hinted_factual_question_not_short():
    """Hinted factual questions stay non-short (factual stem wins)."""
    text = "[Style: respond in 1-3 short sentences.]\n\nwhat is the meaning of life?"
    assert _is_short_conversational_query(text) is False


def test_explicitly_hinted_long_query_not_short():
    text = (
        "[Style: respond with detailed numbered steps.]\n\n"
        "Tell me everything about the history of jazz music."
    )
    assert _is_short_conversational_query(text) is False


# ---------------------------------------------------------------------------
# Regression: un-hinted classification is unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("greeting", [
    "hello",
    "thanks",
    "ok",
    "say hello",
])
def test_unhinted_greeting_classifies_as_short(greeting):
    assert _is_short_conversational_query(greeting) is True


@pytest.mark.parametrize("factual", [
    "what time is it",
    "how much does a duck weigh",
    "what's the meaning of life",
])
def test_unhinted_factual_not_short(factual):
    assert _is_short_conversational_query(factual) is False
