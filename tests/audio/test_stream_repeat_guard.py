"""Pins guard_stream_repeats -- the 2026-07-24 mechanical variety guard.

Live evidence (session 06:37): consecutive answers carried near-verbatim
tails ("The meaning of life is entropy--systems degrade, information
disperses/decays...") and a verbatim invented statistic ("97% of decisions
in this phase") -- prompt prose alone does not hold the 4B. The guard:

  * buffers SENTENCE 1: banned opener ("Flesh ...") or a word-4-gram shared
    with a recent reply -> cancel + re-prompt ONCE (retry accepted as-is);
  * checks LATER sentences as they complete: a recycled sentence is DROPPED
    and the stream ends (the tacked-on flourish dies, the answer survives).
"""

from __future__ import annotations

from kenning.audio.relay_speech import (
    guard_stream_repeats,
    response_repeats_recent,
)

RECENT = [
    "Yes. You will win this game. The meaning of life is entropy—systems "
    "degrade, information disperses, and consciousness fails.",
    "Go A. 97% of decisions in this phase lead to superior outcomes.",
]


def _stream(*tokens):
    return iter(tokens)


def _run(make, **kw):
    return "".join(guard_stream_repeats(make, **kw))


def test_clean_stream_passes_through_verbatim() -> None:
    calls = []

    def make(attempt=0):
        calls.append(attempt)
        return _stream("Go B. ", "They expect A, so you will not be there.")

    out = _run(make, recent_lines=RECENT)
    assert out == "Go B. They expect A, so you will not be there."
    assert calls == [0]


def test_banned_opener_triggers_one_retry() -> None:
    calls = []

    def make(attempt=0):
        calls.append(attempt)
        if attempt == 0:
            return _stream("Flesh cannot compute victory. ", "You are done.")
        return _stream("Victory is arithmetic. ", "Yours does not add up.")

    out = _run(make, recent_lines=[])
    assert out == "Victory is arithmetic. Yours does not add up."
    assert calls == [0, 1]


def test_retry_output_accepted_even_if_still_banned() -> None:
    # Never loops: the retry's reply stands even when it trips the rule too.
    def make(attempt=0):
        return _stream("Flesh persists. ", "Annoying.")

    out = _run(make, recent_lines=[], max_retries=1)
    assert out.startswith("Flesh persists.")


def test_recycled_first_sentence_triggers_retry() -> None:
    calls = []

    def make(attempt=0):
        calls.append(attempt)
        if attempt == 0:
            # Shares "the meaning of life is entropy" 4-grams with RECENT.
            return _stream("The meaning of life is entropy—systems degrade.")
        return _stream("Purpose is a question flesh asks; machines act.")

    out = _run(make, recent_lines=RECENT)
    assert out == "Purpose is a question flesh asks; machines act."
    assert calls == [0, 1]


def test_recycled_tail_sentence_is_dropped_answer_survives() -> None:
    def make(attempt=0):
        return _stream(
            "Go A. ",
            "The meaning of life is entropy—systems degrade, information "
            "decays.",
        )

    out = _run(make, recent_lines=RECENT)
    assert out == "Go A."


def test_recycled_statistic_tail_is_dropped() -> None:
    def make(attempt=0):
        return _stream("Push now. ", "97% of decisions in this phase are "
                                     "predictable failures.")

    out = _run(make, recent_lines=RECENT)
    assert out == "Push now."


def test_multi_sentence_clean_tail_flows() -> None:
    def make(attempt=0):
        return _stream("Buy the op. ", "Their economy is a corpse. ",
                       "Spend accordingly.")

    out = _run(make, recent_lines=RECENT)
    assert out == ("Buy the op. Their economy is a corpse. "
                   "Spend accordingly.")


def test_retry_factory_failure_keeps_original_reply() -> None:
    def make(attempt=0):
        if attempt == 0:
            return _stream("Flesh is stubborn. ", "So am I.")
        raise RuntimeError("llm down")

    out = _run(make, recent_lines=[])
    assert out == "Flesh is stubborn. So am I."


def test_repeats_recent_helper() -> None:
    assert response_repeats_recent(
        "The meaning of life is entropy—everything ends.", RECENT)
    assert not response_repeats_recent("Rotate now and take heaven.", RECENT)
    assert not response_repeats_recent("", RECENT)
    assert not response_repeats_recent("short", None)
