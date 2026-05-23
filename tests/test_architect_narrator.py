"""Tests for the architect-plan TTS narrator (catalog T5 Phase 2).

Sentence splitter, narrator loop, barge-in handoff, char cap, and the
fail-open behaviour on TTS exceptions are all covered. No real TTS
engine is invoked -- a recording stub stands in.
"""

from __future__ import annotations

from typing import List

import pytest

from ultron.coding.architect_narrator import (
    ArchitectNarrator,
    NarrationResult,
    narrate_plan,
    split_into_sentences,
)


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------


def test_splitter_handles_simple_sentences():
    out = split_into_sentences("First sentence. Second one! Third?")
    assert len(out) == 3
    assert out[0].startswith("First")
    assert out[2].endswith("?")


def test_splitter_handles_trailing_fragment_without_terminator():
    out = split_into_sentences("First sentence. And then a fragment")
    assert out == ["First sentence.", "And then a fragment"]


def test_splitter_does_not_split_decimals():
    out = split_into_sentences("Pi is 3.14. That's all.")
    assert out == ["Pi is 3.14.", "That's all."]


def test_splitter_handles_empty_input():
    assert split_into_sentences("") == []
    assert split_into_sentences("   \n\t") == []


def test_splitter_keeps_quotes_with_sentence():
    out = split_into_sentences('She said "go now." We went.')
    # The split should occur after the closing quote, not before.
    assert any("now." in s for s in out)
    assert any("went." in s for s in out)


def test_splitter_handles_only_one_sentence():
    out = split_into_sentences("Just one.")
    assert out == ["Just one."]


# ---------------------------------------------------------------------------
# Fake TTS
# ---------------------------------------------------------------------------


class _RecordingTTS:
    def __init__(self, raise_on_call: int | None = None):
        self.spoken: List[str] = []
        self.stop_count = 0
        self._raise_on_call = raise_on_call

    def speak(self, text: str) -> None:
        if (
            self._raise_on_call is not None
            and len(self.spoken) == self._raise_on_call
        ):
            raise RuntimeError("simulated TTS failure")
        self.spoken.append(text)

    def stop(self) -> None:
        self.stop_count += 1


# ---------------------------------------------------------------------------
# Narrator behaviour
# ---------------------------------------------------------------------------


def test_narrate_completes_with_no_barge_in():
    tts = _RecordingTTS()
    result = ArchitectNarrator(tts, inter_sentence_pause_seconds=0).narrate(
        "First. Second. Third.",
    )
    assert isinstance(result, NarrationResult)
    assert result.completed is True
    assert result.interrupted is False
    assert result.sentences_spoken == 3
    assert tts.spoken == ["First.", "Second.", "Third."]


def test_narrate_interrupts_when_should_stop_returns_true():
    tts = _RecordingTTS()
    calls = []

    def should_stop():
        calls.append(1)
        # Interrupt before the SECOND sentence (i.e. after one is spoken).
        return len(calls) >= 2

    result = ArchitectNarrator(tts, inter_sentence_pause_seconds=0).narrate(
        "Sentence one. Sentence two. Sentence three.",
        should_stop=should_stop,
    )
    assert result.completed is False
    assert result.interrupted is True
    assert result.sentences_spoken == 1
    assert tts.stop_count == 1


def test_narrate_returns_error_when_plan_is_empty():
    tts = _RecordingTTS()
    result = ArchitectNarrator(tts).narrate("")
    assert result.completed is False
    assert result.interrupted is False
    assert result.error == "empty plan"
    assert tts.spoken == []


def test_narrate_returns_error_when_no_sentences_after_split():
    tts = _RecordingTTS()
    # whitespace-only -> splitter returns [] -> narrator surfaces error.
    result = ArchitectNarrator(tts).narrate("\n\t   ")
    assert result.completed is False
    assert "empty plan" in result.error  # whitespace caught by the empty-check


def test_narrate_respects_max_chars_cap():
    tts = _RecordingTTS()
    # max_chars=20 -> first sentence (~13 chars) fits; second pushes over.
    result = ArchitectNarrator(
        tts, max_chars=20, inter_sentence_pause_seconds=0,
    ).narrate("First short. Second slightly longer one. Tail.")
    assert result.completed is False
    assert result.error == "max_chars_exceeded"
    assert result.sentences_spoken == 1


def test_narrate_with_max_chars_zero_means_no_cap():
    tts = _RecordingTTS()
    # Use capitalised "Sentence" so the splitter (which requires
    # whitespace-then-capital after the terminator) fires correctly.
    long_plan = ". ".join(f"Sentence {i}" for i in range(20)) + "."
    result = ArchitectNarrator(
        tts, max_chars=0, inter_sentence_pause_seconds=0,
    ).narrate(long_plan)
    assert result.completed is True
    assert result.sentences_spoken == 20


def test_narrate_fail_open_on_tts_exception():
    tts = _RecordingTTS(raise_on_call=1)  # raise on 2nd speak() call
    result = ArchitectNarrator(tts, inter_sentence_pause_seconds=0).narrate(
        "First. Second. Third.",
    )
    assert result.completed is False
    assert result.interrupted is False
    assert "tts.speak raised" in result.error
    assert result.sentences_spoken == 1  # only the first succeeded


def test_narrate_handles_should_stop_exception_gracefully():
    tts = _RecordingTTS()

    def bad_callback():
        raise RuntimeError("boom")

    result = ArchitectNarrator(tts, inter_sentence_pause_seconds=0).narrate(
        "First. Second.",
        should_stop=bad_callback,
    )
    # Exception in should_stop is treated as "no interrupt"; narration runs.
    assert result.completed is True
    assert result.sentences_spoken == 2


def test_narrate_one_shot_convenience_wrapper():
    tts = _RecordingTTS()
    result = narrate_plan(
        "Hello there. General Kenobi.",
        tts,
        inter_sentence_pause_seconds=0,
    )
    assert result.completed is True
    assert result.sentences_spoken == 2


def test_narrator_constructor_rejects_negative_pause():
    tts = _RecordingTTS()
    with pytest.raises(ValueError):
        ArchitectNarrator(tts, inter_sentence_pause_seconds=-1)


def test_narrator_constructor_rejects_negative_max_chars():
    tts = _RecordingTTS()
    with pytest.raises(ValueError):
        ArchitectNarrator(tts, max_chars=-5)
