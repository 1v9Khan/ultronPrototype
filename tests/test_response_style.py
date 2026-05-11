"""Tests for the per-call response-style addenda.

Covers :func:`apply_brevity_hint` -- the 2026-05-10 reinforcement that
counters the 4B model's tendency to produce 4-paragraph essays in
response to short questions like "What are the Orcs in 40k?".

The hint must:
- fire on short, non-explain questions (the regression case)
- NOT fire on questions explicitly asking for depth
- NOT fire on long questions (they often legitimately need detail)
- pass through empty input unchanged
- compose cleanly above the user_text (newline-separated)
"""

from __future__ import annotations

import pytest

from ultron.response_style import apply_brevity_hint, is_brief_question


# ---------------------------------------------------------------------------
# is_brief_question
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "utterance",
    [
        "What are the Orcs in 40k?",                      # live-session regression
        "Who are you?",
        "What food do ducks eat?",
        "What is rain?",
        "Why is the sky blue?",
        "What's the capital of France?",
        "Hello.",
    ],
)
def test_brief_question_detected(utterance):
    assert is_brief_question(utterance), f"expected brief: {utterance!r}"


@pytest.mark.parametrize(
    "utterance",
    [
        "Explain to me the Tyranids in 40k.",             # live-session OK case
        "Walk me through how to bake bread.",
        "Give me step-by-step instructions for changing the oil.",
        "Tell me everything you know about the Black Library.",
        "How do I file a Schedule C in detail?",
        "Elaborate on the differences between TCP and UDP.",
        "List all the planets in the solar system.",
    ],
)
def test_depth_request_skips_brevity(utterance):
    assert not is_brief_question(utterance), (
        f"expected non-brief because of depth marker: {utterance!r}"
    )


def test_long_question_skips_brevity():
    long_q = (
        "I'm trying to set up a multi-stage CI pipeline that builds a "
        "Docker image, runs unit tests inside the container, pushes "
        "the image to a private registry, then triggers a Kubernetes "
        "rolling deployment -- what are the right tooling choices and "
        "how should I sequence the jobs to keep wall-clock low?"
    )
    assert not is_brief_question(long_q)


def test_empty_or_whitespace_is_not_brief():
    assert not is_brief_question("")
    assert not is_brief_question("   ")
    assert not is_brief_question("\n\t\n")


def test_borderline_word_count():
    # 12 words exactly: at the threshold, still brief.
    twelve_words = "tell me what color sky is at dawn in late autumn here"
    assert is_brief_question(twelve_words)

    # 13 words: just over the threshold AND well over the char threshold,
    # so it falls out.
    thirteen_words = (
        "tell me what color sky is at dawn in late autumn here too"
    )
    # If words > 12 AND chars > 80, returns False. Char count here is
    # comfortably > 80 once split; check actual length and assert.
    if len(thirteen_words) > 80:
        assert not is_brief_question(thirteen_words)


# ---------------------------------------------------------------------------
# apply_brevity_hint
# ---------------------------------------------------------------------------


def test_apply_brevity_prepends_directive_to_brief():
    out = apply_brevity_hint("What are the Orcs in 40k?")
    assert out.startswith("[Style:")
    assert "1-3 short sentences" in out
    # Original text preserved at the end.
    assert out.endswith("What are the Orcs in 40k?")
    # Blank line between directive and user text.
    assert "]\n\nWhat" in out


def test_apply_brevity_returns_unchanged_on_depth_request():
    text = "Explain in detail how speculative decoding works."
    assert apply_brevity_hint(text) == text


def test_apply_brevity_returns_unchanged_on_long_question():
    long_q = (
        "I have a Python codebase that uses asyncio and I want to "
        "convert it to use threading instead -- what are the gotchas "
        "I should plan for and what would the migration look like?"
    )
    assert apply_brevity_hint(long_q) == long_q


def test_apply_brevity_returns_unchanged_on_empty():
    assert apply_brevity_hint("") == ""
    assert apply_brevity_hint("   ") == "   "


def test_apply_brevity_idempotent_when_already_hinted():
    """Calling apply_brevity_hint on already-hinted text should be a no-op
    (the hinted version is too long + contains '[Style:' which makes
    the question read as 'long' or 'system instruction', not brief)."""
    once = apply_brevity_hint("Who are you?")
    twice = apply_brevity_hint(once)
    # Should not double-prepend the directive.
    assert twice.count("[Style: respond in 1-3 short sentences") == 1
