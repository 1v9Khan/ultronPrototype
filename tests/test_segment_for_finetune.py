"""Unit tests for scripts/segment_for_finetune.py pure helpers.

The segmenter has audio-IO + transcript-IO entry points (``main``)
and pure helpers (``plan_segments``, ``select_split_index``). This
file covers the pure helpers only -- the IO path is exercised
manually during a Path B run.

Boundary priority being verified:

1. Hard-break gap (silence >= hard_break_gap_s) AT or after target_s
2. Sentence terminator (.!?) at or after target_s
3. Clause terminator (,;:) at or after target_s
4. Word boundary nearest target_s
5. Hard cut at max_s
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add scripts/ to sys.path so we can import segment_for_finetune.
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from segment_for_finetune import (  # noqa: E402
    PlannedSegment,
    _Word,
    plan_segments,
    select_split_index,
    _is_sentence_end,
    _is_clause_end,
    _word_gap_after,
)


# ---------------------------------------------------------------------------
# Predicate helpers
# ---------------------------------------------------------------------------


def test_is_sentence_end_recognises_period():
    assert _is_sentence_end("done.")


def test_is_sentence_end_recognises_question_mark():
    assert _is_sentence_end("really?")


def test_is_sentence_end_recognises_exclamation():
    assert _is_sentence_end("now!")


def test_is_sentence_end_rejects_plain_word():
    assert not _is_sentence_end("hello")


def test_is_sentence_end_rejects_comma():
    assert not _is_sentence_end("though,")


def test_is_sentence_end_handles_trailing_whitespace():
    assert _is_sentence_end("ok.  ")


def test_is_sentence_end_handles_empty_string():
    assert not _is_sentence_end("")


def test_is_clause_end_recognises_comma():
    assert _is_clause_end("first,")


def test_is_clause_end_recognises_semicolon():
    assert _is_clause_end("first;")


def test_is_clause_end_recognises_colon():
    assert _is_clause_end("first:")


def test_is_clause_end_rejects_period():
    """Clause vs sentence are distinct categories -- the boundary
    selector tries sentence-end first, then clause-end, then nearest
    word. If a period collapsed into _is_clause_end, sentence-priority
    would silently fail."""
    assert not _is_clause_end("done.")


# ---------------------------------------------------------------------------
# _word_gap_after
# ---------------------------------------------------------------------------


def test_word_gap_after_returns_silence_between_words():
    words = [
        _Word(start=0.0, end=0.5, text="hello"),
        _Word(start=1.0, end=1.5, text="world"),
    ]
    assert _word_gap_after(words, 0) == pytest.approx(0.5)


def test_word_gap_after_returns_zero_for_last_word():
    words = [_Word(start=0.0, end=0.5, text="hello")]
    assert _word_gap_after(words, 0) == 0.0


def test_word_gap_after_clamps_negative_to_zero():
    """Word timestamps from Whisper can occasionally overlap by a
    few ms when the model is unsure -- the helper must not return
    negative."""
    words = [
        _Word(start=0.0, end=1.0, text="hello"),
        _Word(start=0.9, end=1.5, text="world"),
    ]
    assert _word_gap_after(words, 0) == 0.0


# ---------------------------------------------------------------------------
# select_split_index
# ---------------------------------------------------------------------------


def _uniform_words(n: int, *, word_dur=0.4, gap=0.1, suffix="") -> list[_Word]:
    """Build ``n`` uniform-spaced words; word N takes the slot
    ``[N*(word_dur+gap), N*(word_dur+gap)+word_dur]``."""
    out = []
    for i in range(n):
        start = i * (word_dur + gap)
        out.append(_Word(start=start, end=start + word_dur, text=f"w{i}{suffix}"))
    return out


def test_select_split_index_returns_start_when_no_words_remain():
    words: list[_Word] = []
    assert select_split_index(
        words, 0, target_s=7, min_s=3, max_s=15, hard_break_gap_s=0.6
    ) == 0


def test_select_split_index_one_word_too_short_still_returned():
    """Even if the first word alone is below min_s, the loop must
    still return a valid index so callers don't infinite-loop."""
    words = [_Word(start=0.0, end=0.4, text="hi")]
    j = select_split_index(
        words, 0, target_s=7, min_s=3, max_s=15, hard_break_gap_s=0.6
    )
    assert j == 0  # the one and only word


def test_select_split_index_prefers_sentence_end_over_target_word():
    # 14 words of 0.5s each, gap=0.1. Word 13 ends at 13*0.6+0.5=8.3s,
    # over target. Mark word 11 with a period (sentence end at ~7.1s).
    words = _uniform_words(14)
    words[11] = _Word(start=words[11].start, end=words[11].end, text="w11.")
    j = select_split_index(
        words, 0, target_s=7, min_s=3, max_s=15, hard_break_gap_s=999.0
    )
    assert j == 11  # sentence-end takes precedence


def test_select_split_index_prefers_clause_end_when_no_sentence_end():
    words = _uniform_words(14)
    words[12] = _Word(start=words[12].start, end=words[12].end, text="w12,")
    j = select_split_index(
        words, 0, target_s=7, min_s=3, max_s=15, hard_break_gap_s=999.0
    )
    assert j == 12  # clause-end picked because no sentence-end in range


def test_select_split_index_prefers_hard_gap_over_sentence_end():
    """The priority is: hard-break gap > sentence > clause > nearest
    target. A long silence is the strongest signal that the speaker
    paused."""
    # Word 11 has a period; word 12 has a 1.0s gap following it.
    words = _uniform_words(14)
    words[11] = _Word(start=words[11].start, end=words[11].end, text="w11.")
    # Shift word 13's start to leave a 1.0s gap after word 12.
    gap_word_end = words[12].end
    words[13] = _Word(start=gap_word_end + 1.0, end=gap_word_end + 1.4, text="w13")
    j = select_split_index(
        words, 0, target_s=7, min_s=3, max_s=15, hard_break_gap_s=0.6
    )
    assert j == 12  # gap break wins


def test_select_split_index_falls_back_to_nearest_word_to_target():
    # No punctuation, no big gaps. Should land near target=7s.
    words = _uniform_words(20)
    j = select_split_index(
        words, 0, target_s=7, min_s=3, max_s=15, hard_break_gap_s=999.0
    )
    end_s = words[j].end
    # target_s=7; legal window after min_s=3 starts at index ~5 and
    # before max_s=15 ends at index ~24. With 0.6s/word, the word
    # whose end is closest to 7s is the 11th word (end = 11*0.6+0.5 = 7.1s).
    assert 6.0 <= end_s <= 8.0


def test_select_split_index_respects_max_s_hard_ceiling():
    """No matter what, no candidate may exceed max_s."""
    words = _uniform_words(50)  # plenty of words to span
    j = select_split_index(
        words, 0, target_s=7, min_s=3, max_s=15, hard_break_gap_s=999.0
    )
    assert words[j].end - words[0].start <= 15.0


def test_select_split_index_respects_min_s_floor_when_sentence_too_early():
    """A sentence terminator BEFORE min_s shouldn't be picked --
    otherwise we'd produce sub-3-second segments on a transcript full
    of acks. The selector must keep going past that period until
    seg_dur >= min_s."""
    # Word 1 ends at 0.9s but has a period -- segment would be < min_s.
    words = _uniform_words(20)
    words[1] = _Word(start=words[1].start, end=words[1].end, text="w1.")
    j = select_split_index(
        words, 0, target_s=7, min_s=3, max_s=15, hard_break_gap_s=999.0
    )
    # The early period shouldn't win; selector should reach min_s.
    assert words[j].end - words[0].start >= 3.0


# ---------------------------------------------------------------------------
# plan_segments (sequencing the splitter end-to-end)
# ---------------------------------------------------------------------------


def test_plan_segments_consumes_all_words():
    """Every word from the input must appear in exactly one segment."""
    words = _uniform_words(40)
    segments = plan_segments(words, target_s=5, min_s=2, max_s=10, hard_break_gap_s=999.0)
    all_indices = sorted(idx for s in segments for idx in s.word_indices)
    assert all_indices == list(range(len(words)))


def test_plan_segments_returns_PlannedSegment_dataclass_fields():
    words = _uniform_words(20)
    out = plan_segments(words, target_s=5, min_s=2, max_s=10, hard_break_gap_s=999.0)
    assert all(isinstance(s, PlannedSegment) for s in out)
    for s in out:
        assert isinstance(s.start, float)
        assert isinstance(s.end, float)
        assert isinstance(s.text, str)
        assert isinstance(s.word_indices, tuple)


def test_plan_segments_text_joins_words_with_single_space():
    words = [
        _Word(start=0.0, end=0.5, text="hello"),
        _Word(start=0.6, end=1.1, text="world."),
        _Word(start=1.3, end=1.8, text="how"),
        _Word(start=1.9, end=2.4, text="are"),
        _Word(start=2.5, end=3.0, text="you?"),
    ]
    out = plan_segments(words, target_s=2, min_s=1, max_s=4, hard_break_gap_s=999.0)
    full_text = " ".join(s.text for s in out)
    assert full_text == "hello world. how are you?"


def test_plan_segments_breaks_at_sentence_endings_under_target():
    """When a transcript naturally has frequent sentence endings near
    the target, the segmenter should produce one segment per
    sentence."""
    words = []
    t = 0.0
    for i in range(6):
        # Each sentence is 5 words of 0.5s each (2.5s wall) + period.
        for k in range(5):
            text = f"w{i}{k}"
            if k == 4:
                text = text + "."
            words.append(_Word(start=t, end=t + 0.5, text=text))
            t += 0.7  # 0.5s word + 0.2s gap
        # Add a small gap between sentences.
        t += 0.3
    out = plan_segments(words, target_s=2.5, min_s=1.5, max_s=5, hard_break_gap_s=999.0)
    # Expect 6 sentences -> 6 segments, each ending in '.'
    assert len(out) == 6
    for seg in out:
        assert seg.text.rstrip().endswith(".")


def test_plan_segments_handles_empty_input():
    assert plan_segments([], target_s=7, min_s=3, max_s=15, hard_break_gap_s=0.6) == []


def test_plan_segments_emits_durations_within_max_when_possible():
    """When the transcript has plenty of breakable points, no segment
    should exceed max_s. This is the contract the XTTS-v2 fine-tune
    recipe relies on."""
    words = _uniform_words(60)
    out = plan_segments(words, target_s=5, min_s=2, max_s=8, hard_break_gap_s=999.0)
    for seg in out:
        assert (seg.end - seg.start) <= 8.0 + 1e-6


def test_plan_segments_first_segment_starts_at_first_word():
    words = _uniform_words(30)
    # Shift the entire timeline so segment-relative timing matters.
    shifted = [_Word(start=w.start + 5.0, end=w.end + 5.0, text=w.text) for w in words]
    out = plan_segments(shifted, target_s=5, min_s=2, max_s=10, hard_break_gap_s=999.0)
    assert out[0].start == pytest.approx(5.0)


def test_plan_segments_advances_when_word_alone_exceeds_max():
    """A pathological single-word duration longer than max_s would
    cause an infinite loop if the selector returned start_idx-1. The
    contract: the selector returns at least start_idx so plan_segments
    always advances."""
    long_word = _Word(start=0.0, end=20.0, text="aaaaa")
    words = [long_word, _Word(start=21.0, end=21.5, text="b")]
    out = plan_segments(words, target_s=7, min_s=3, max_s=15, hard_break_gap_s=0.6)
    # The pathological word ends up in its own oversize segment.
    # The second word gets its own segment.
    assert len(out) == 2
    assert out[0].text.strip() == "aaaaa"
    assert out[1].text.strip() == "b"
