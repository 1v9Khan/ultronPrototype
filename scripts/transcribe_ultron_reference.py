"""Transcribe the 3-min Ultron reference clip with Whisper large-v3.

Round-9 Path B step 1 (2026-05-20). Generates the (audio, text) pairs
that Path B step 2 (``segment_for_finetune.py``) needs to slice the
3-minute ``Ultron_vocals_mono_v1.wav`` into 5-15 s training clips at
word/sentence boundaries.

Uses faster-whisper large-v3 with word-level timestamps and beam search
for maximum accuracy. The source is a single clean studio clip of one
speaker so we expect near-perfect transcription -- if you spot any
errors in the output, hand-edit ``transcript.txt`` BEFORE running the
segmenter; the segmenter trusts the words array.

Output goes to ``ultronVoiceAudio/transcript_large_v3/``:

* ``segments.json`` -- list of Whisper segments with their words
  (each word has ``start``, ``end``, ``word`` for sub-second
  alignment). The segmenter consumes this.
* ``transcript.txt`` -- plain concatenated text for human review.
* ``info.json`` -- model + decode params + measured timings.

The script loads large-v3 (~3 GB VRAM), runs the transcription, and
exits. No long-lived process; the orchestrator is unaffected.

Run from the main venv at the project root::

    C:\\STC\\ultronPrototype\\.venv\\Scripts\\python.exe ^
        C:\\STC\\ultronPrototype\\scripts\\transcribe_ultron_reference.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent  # C:/STC/ultronPrototype

DEFAULT_SOURCE = PROJECT / "ultronVoiceAudio" / "kokoro training audio" / "Ultron_vocals_mono_v1.wav"
DEFAULT_OUT = PROJECT / "ultronVoiceAudio" / "transcript_large_v3"

DEFAULT_MODEL = "large-v3"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE = "float16"  # large-v3 at float16 = ~3 GB VRAM
DEFAULT_BEAM = 5             # quality > speed for this one-shot
DEFAULT_LANGUAGE = "en"


def transcribe(
    source: Path,
    out_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE,
    beam_size: int = DEFAULT_BEAM,
    language: Optional[str] = DEFAULT_LANGUAGE,
) -> dict:
    """Transcribe ``source`` and write segments.json, transcript.txt,
    info.json under ``out_dir``. Returns a summary dict."""
    from faster_whisper import WhisperModel

    if not source.is_file():
        raise FileNotFoundError(f"audio source missing: {source}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"source: {source} ({source.stat().st_size / 1e6:.1f} MB)")
    print(f"output: {out_dir}/")
    print(f"model: {model_name} / device={device} / compute={compute_type} / beam={beam_size}")

    print("\nLoading Whisper large-v3 (~3 GB VRAM)...")
    t0 = time.monotonic()
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    load_s = time.monotonic() - t0
    print(f"loaded in {load_s:.2f}s")

    print("\nTranscribing (word-level timestamps enabled)...")
    t0 = time.monotonic()
    segments_iter, info = model.transcribe(
        str(source),
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=False,    # source is already cleaned; let Whisper see the full audio
        condition_on_previous_text=True,
    )

    # segments_iter is a generator; collect.
    segments_out: list[dict] = []
    transcript_chunks: list[str] = []
    for seg in segments_iter:
        words: list[dict] = []
        if seg.words:
            for w in seg.words:
                words.append({
                    "start": float(w.start),
                    "end": float(w.end),
                    "word": w.word,
                    "probability": float(getattr(w, "probability", 0.0)),
                })
        segments_out.append({
            "id": seg.id,
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text,
            "words": words,
            "avg_logprob": float(seg.avg_logprob),
            "no_speech_prob": float(seg.no_speech_prob),
        })
        transcript_chunks.append(seg.text.strip())

    transcribe_s = time.monotonic() - t0
    print(f"transcribed in {transcribe_s:.2f}s")
    print(f"detected language: {info.language} (prob {info.language_probability:.2f})")
    print(f"audio duration: {info.duration:.2f}s")
    print(f"segments produced: {len(segments_out)}")
    word_count = sum(len(s["words"]) for s in segments_out)
    print(f"words produced: {word_count}")

    transcript = " ".join(transcript_chunks).strip()

    (out_dir / "segments.json").write_text(
        json.dumps(segments_out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "transcript.txt").write_text(transcript + "\n", encoding="utf-8")
    (out_dir / "info.json").write_text(
        json.dumps({
            "source": str(source),
            "model_name": model_name,
            "device": device,
            "compute_type": compute_type,
            "beam_size": beam_size,
            "language_detected": info.language,
            "language_probability": float(info.language_probability),
            "duration_seconds": float(info.duration),
            "load_seconds": load_s,
            "transcribe_seconds": transcribe_s,
            "n_segments": len(segments_out),
            "n_words": word_count,
        }, indent=2),
        encoding="utf-8",
    )

    print(f"\nWrote:")
    print(f"  {out_dir / 'segments.json'}")
    print(f"  {out_dir / 'transcript.txt'}")
    print(f"  {out_dir / 'info.json'}")

    return {
        "n_segments": len(segments_out),
        "n_words": word_count,
        "duration_seconds": float(info.duration),
        "transcribe_seconds": transcribe_s,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Whisper large-v3 transcription of the Ultron reference clip."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Input WAV (default: {DEFAULT_SOURCE.relative_to(PROJECT)}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output dir (default: {DEFAULT_OUT.relative_to(PROJECT)}).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--compute", default=DEFAULT_COMPUTE)
    parser.add_argument("--beam-size", type=int, default=DEFAULT_BEAM)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    args = parser.parse_args(argv)

    transcribe(
        args.source,
        args.out_dir,
        model_name=args.model,
        device=args.device,
        compute_type=args.compute,
        beam_size=args.beam_size,
        language=args.language,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
