"""Slice the Ultron reference clip into LJSpeech-format training data.

Round-9 Path B step 2 (2026-05-20). Consumes the word-level transcript
from ``transcribe_ultron_reference.py`` and writes a Coqui XTTS-v2
fine-tune-ready dataset:

* ``wavs/0001.wav`` ... ``wavs/NNNN.wav`` -- mono 22050 Hz PCM clips,
  each 5-15 s, sliced at word boundaries near natural prosodic
  breaks (sentence-end > clause-end > long word gap > forced cut).
* ``metadata.csv`` -- LJSpeech-format: ``id|text|normalized_text``.
* ``manifest.json`` -- machine-readable summary (id, source-time,
  duration, text) for downstream tooling.

The segmenter is split into pure helpers (``plan_segments``,
``select_split_index``) and an I/O entry-point (``main``), so the
boundary logic is unit-testable without loading audio.

Run from main venv at project root::

    C:\\STC\\ultronPrototype\\.venv\\Scripts\\python.exe ^
        C:\\STC\\ultronPrototype\\scripts\\segment_for_finetune.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

DEFAULT_AUDIO_SOURCE = PROJECT / "ultronVoiceAudio" / "kokoro training audio" / "Ultron_vocals_mono_v1.wav"
DEFAULT_TRANSCRIPT_DIR = PROJECT / "ultronVoiceAudio" / "transcript_large_v3"
DEFAULT_OUT_DIR = PROJECT / "ultronVoiceAudio" / "xtts_finetune_dataset"

# Target segment length (seconds). XTTS-v2 fine-tune sweet spot is
# 5-15 s; longer hurts batching, shorter starves the GPT context.
DEFAULT_TARGET_S = 7.0
DEFAULT_MIN_S = 3.0
DEFAULT_MAX_S = 15.0
# Inter-word gap that signals a hard break (likely sentence/clause
# silence; do not span across it).
DEFAULT_HARD_BREAK_GAP_S = 0.6
# Sample rate for the output clips. XTTS-v2 fine-tune accepts other
# rates but 22050 mono is the LJSpeech canonical and what Coqui's
# trainer assumes by default.
DEFAULT_OUTPUT_SR = 22050

SENTENCE_TERMINATORS = {".", "!", "?"}
CLAUSE_TERMINATORS = {",", ";", ":"}


@dataclass(frozen=True)
class _Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class PlannedSegment:
    """A planned slice of the source audio. ``start`` and ``end`` are
    in seconds relative to the source clip. ``text`` is the
    concatenated word text. ``word_indices`` records which words from
    the input list went into this segment (useful for debugging)."""
    start: float
    end: float
    text: str
    word_indices: tuple[int, ...]


def _is_sentence_end(word_text: str) -> bool:
    """True if the word text ends with a hard sentence terminator."""
    stripped = word_text.strip()
    return bool(stripped) and stripped[-1] in SENTENCE_TERMINATORS


def _is_clause_end(word_text: str) -> bool:
    """True if the word text ends with a soft clause terminator
    (comma, semicolon, colon)."""
    stripped = word_text.strip()
    return bool(stripped) and stripped[-1] in CLAUSE_TERMINATORS


def _word_gap_after(words: Sequence[_Word], i: int) -> float:
    """Inter-word silence between ``words[i]`` and ``words[i+1]``,
    in seconds. Returns 0.0 if ``i`` is the last word."""
    if i + 1 >= len(words):
        return 0.0
    return max(0.0, words[i + 1].start - words[i].end)


def select_split_index(
    words: Sequence[_Word],
    start_idx: int,
    *,
    target_s: float,
    min_s: float,
    max_s: float,
    hard_break_gap_s: float,
) -> int:
    """Pick the index ``j`` such that ``words[start_idx:j+1]`` is the
    next segment. The chosen ``j`` is the LAST word in the segment.

    Boundary preference (highest priority first), all considered
    within the legal window ``[min_s, max_s]``:

    1. Hard break gap >= ``hard_break_gap_s`` after the word -- take
       the FIRST such gap (silence is the strongest signal).
    2. Sentence terminator (``.``, ``!``, ``?``) -- take the
       candidate nearest to ``target_s`` (preferring full natural
       sentences even if shorter than target).
    3. Clause terminator (``,``, ``;``, ``:``) -- nearest to
       ``target_s``.
    4. Word boundary nearest to ``target_s``.
    5. If even ``min_s`` is unreachable (single very long word at
       start), return the last word that fits inside ``max_s``, or
       at minimum ``start_idx`` itself so the segmenter always
       advances.

    Pure helper -- no audio, no IO.
    """
    if start_idx >= len(words):
        return start_idx
    base = words[start_idx].start
    last_legal: int = start_idx  # at minimum the segment is one word

    candidate_hard_gap: Optional[int] = None
    candidate_sentence: Optional[int] = None
    candidate_sentence_dist = float("inf")
    candidate_clause: Optional[int] = None
    candidate_clause_dist = float("inf")
    candidate_target: Optional[int] = None
    candidate_target_dist = float("inf")

    for j in range(start_idx, len(words)):
        seg_dur = words[j].end - base
        if seg_dur > max_s:
            break
        last_legal = j
        if seg_dur < min_s:
            continue
        # Inside legal [min_s, max_s] window.
        dist = abs(seg_dur - target_s)
        gap = _word_gap_after(words, j)
        if gap >= hard_break_gap_s and candidate_hard_gap is None:
            candidate_hard_gap = j
        if _is_sentence_end(words[j].text) and dist < candidate_sentence_dist:
            candidate_sentence = j
            candidate_sentence_dist = dist
        if _is_clause_end(words[j].text) and dist < candidate_clause_dist:
            candidate_clause = j
            candidate_clause_dist = dist
        if dist < candidate_target_dist:
            candidate_target_dist = dist
            candidate_target = j

    if candidate_hard_gap is not None:
        return candidate_hard_gap
    if candidate_sentence is not None:
        return candidate_sentence
    if candidate_clause is not None:
        return candidate_clause
    if candidate_target is not None:
        return candidate_target
    return last_legal


def plan_segments(
    words: Sequence[_Word],
    *,
    target_s: float = DEFAULT_TARGET_S,
    min_s: float = DEFAULT_MIN_S,
    max_s: float = DEFAULT_MAX_S,
    hard_break_gap_s: float = DEFAULT_HARD_BREAK_GAP_S,
) -> list[PlannedSegment]:
    """Slice the word sequence into a list of segments using
    ``select_split_index`` repeatedly. Each segment's ``start`` is
    the first word's start time; ``end`` is the last word's end
    time; ``text`` joins the words with a single space.

    The pure function -- no audio, no IO.
    """
    out: list[PlannedSegment] = []
    i = 0
    while i < len(words):
        j = select_split_index(
            words, i,
            target_s=target_s,
            min_s=min_s,
            max_s=max_s,
            hard_break_gap_s=hard_break_gap_s,
        )
        chosen = words[i : j + 1]
        text = " ".join(w.text.strip() for w in chosen).strip()
        out.append(PlannedSegment(
            start=chosen[0].start,
            end=chosen[-1].end,
            text=text,
            word_indices=tuple(range(i, j + 1)),
        ))
        i = j + 1
    return out


def _load_words_from_segments_json(segments_path: Path) -> list[_Word]:
    """Read the segments.json file written by
    ``transcribe_ultron_reference.py`` and flatten to a single
    word list (sorted by start time)."""
    data = json.loads(segments_path.read_text(encoding="utf-8"))
    words: list[_Word] = []
    for seg in data:
        for w in seg.get("words") or []:
            words.append(_Word(
                start=float(w["start"]),
                end=float(w["end"]),
                text=str(w["word"]),
            ))
    words.sort(key=lambda w: w.start)
    return words


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Slice the Ultron reference clip into LJSpeech training data."
    )
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO_SOURCE)
    parser.add_argument("--transcript-dir", type=Path, default=DEFAULT_TRANSCRIPT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-s", type=float, default=DEFAULT_TARGET_S)
    parser.add_argument("--min-s", type=float, default=DEFAULT_MIN_S)
    parser.add_argument("--max-s", type=float, default=DEFAULT_MAX_S)
    parser.add_argument("--hard-break-gap-s", type=float, default=DEFAULT_HARD_BREAK_GAP_S)
    parser.add_argument("--output-sr", type=int, default=DEFAULT_OUTPUT_SR)
    args = parser.parse_args(argv)

    if not args.audio.is_file():
        print(f"ERROR: audio missing: {args.audio}")
        return 1
    segments_json = args.transcript_dir / "segments.json"
    if not segments_json.is_file():
        print(f"ERROR: transcript missing: {segments_json}. "
              f"Run transcribe_ultron_reference.py first.")
        return 1

    import numpy as np
    import soundfile as sf
    import librosa

    print(f"audio:      {args.audio}")
    print(f"transcript: {segments_json}")
    print(f"out:        {args.out_dir}")
    print(f"target/min/max seconds: {args.target_s}/{args.min_s}/{args.max_s}")
    print(f"hard-break gap: {args.hard_break_gap_s}s")
    print(f"output sample rate: {args.output_sr} Hz mono")

    words = _load_words_from_segments_json(segments_json)
    print(f"\nloaded {len(words)} words from transcript")

    segments = plan_segments(
        words,
        target_s=args.target_s,
        min_s=args.min_s,
        max_s=args.max_s,
        hard_break_gap_s=args.hard_break_gap_s,
    )
    print(f"planned {len(segments)} segments")
    durations = [s.end - s.start for s in segments]
    if durations:
        print(f"  min:  {min(durations):.2f}s")
        print(f"  mean: {sum(durations) / len(durations):.2f}s")
        print(f"  max:  {max(durations):.2f}s")

    print("\nloading + resampling source audio...")
    audio, src_sr = sf.read(str(args.audio), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype("float32")
    if src_sr != args.output_sr:
        # ``kaiser_best`` is librosa's built-in high-quality resampler --
        # avoids the optional ``soxr`` dep that isn't pinned in the
        # main venv. The 44.1/48k -> 22.05k downsample is gentle
        # enough that the quality difference vs soxr is inaudible.
        audio = librosa.resample(
            audio, orig_sr=src_sr, target_sr=args.output_sr, res_type="kaiser_best"
        ).astype("float32")
    sr = args.output_sr

    wavs_dir = args.out_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nwriting clips to {wavs_dir}/...")
    manifest: list[dict] = []
    metadata_lines: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        clip_id = f"{idx:04d}"
        a = int(round(seg.start * sr))
        b = int(round(seg.end * sr))
        clip = audio[a:b]
        out = wavs_dir / f"{clip_id}.wav"
        sf.write(str(out), clip, sr, subtype="PCM_16")
        manifest.append({
            "id": clip_id,
            "wav": f"wavs/{clip_id}.wav",
            "source_start_s": seg.start,
            "source_end_s": seg.end,
            "duration_s": float(len(clip)) / sr,
            "text": seg.text,
        })
        metadata_lines.append(f"{clip_id}|{seg.text}|{seg.text}")

    (args.out_dir / "metadata.csv").write_text(
        "\n".join(metadata_lines) + "\n", encoding="utf-8"
    )
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"wrote {len(segments)} clips + metadata.csv + manifest.json")
    print(f"  {args.out_dir / 'metadata.csv'}")
    print(f"  {args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
