"""Package the bulk-eval output as a Kokoro fine-tune training corpus.

Produces an LJSpeech-style dataset directory consumable by StyleTTS2
fine-tune tooling (Kokoro is StyleTTS2 + ISTFTNet):

    kokoro_training_corpus_<ts>/
      wavs/
        short_response_0001.wav
        short_response_0002.wav
        ...
        long_response_0050.wav
      metadata.csv          (LJSpeech format: id|raw|normalized)
      train.csv             (95% of entries, shuffled deterministically)
      val.csv               (5% of entries)
      README.md             (reproducibility info + lineage)
      stats.json            (summary: count, duration, per-category)

Source: a completed
``ultronVoiceAudio/bulk_eval/bulk_eval_<ts>/`` run directory with its
manifest.json carrying ``id``, ``category``, ``text``, ``wav``,
``duration_s``, and the four trim-flag fields.

Pipeline:

1. Read manifest.json.
2. Filter for clips in the ``[min_duration_s, max_duration_s]`` window
   (default 1.0-12.0 s -- standard Kokoro / StyleTTS2 training bounds).
3. Open each WAV, validate sample rate + mono + non-empty.
4. Copy validated WAVs into ``wavs/`` with deterministic IDs.
5. Emit metadata.csv (pipe-delimited).
6. Deterministic shuffle + train/val split.
7. Write README.md + stats.json with the lineage and reproducibility info.

Usage::

    python ultronVoiceAudio/scripts/package_kokoro_corpus.py \\
        --source ultronVoiceAudio/bulk_eval/bulk_eval_20260521_123515 \\
        [--output ultronVoiceAudio/kokoro_training_corpus_<ts>] \\
        [--min-duration 1.0] [--max-duration 12.0] \\
        [--val-fraction 0.05] [--seed 42] \\
        [--dry-run]

Idempotent: ``--dry-run`` shows counts without writing. Output dir
must not exist (script bails to avoid accidental clobber).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent  # ultronVoiceAudio/ -> project root


def _resolve(p: str) -> Path:
    """Resolve relative paths against the project root."""
    pp = Path(p)
    return pp if pp.is_absolute() else PROJECT_ROOT / pp


# ---------------------------------------------------------------------------
# Normalisation for the LJSpeech "normalized_transcription" column.
#
# Kokoro / StyleTTS2 trainers typically take the normalised column as
# the actual training input. We don't need full TTS normalisation here
# (that's what runtime ``normalize_text_for_tts`` does at inference)
# but we DO want to strip XTTS-control artifacts and standardise some
# obvious patterns so the training text matches the audio.
# ---------------------------------------------------------------------------


def _normalize_for_training(text: str) -> str:
    """Light normalisation for the LJSpeech 'normalized' column."""
    # Strip leading/trailing whitespace.
    t = text.strip()
    # Collapse internal whitespace runs.
    t = re.sub(r"\s+", " ", t)
    # Strip surrounding quotes.
    t = t.strip('"“”\'‘’')
    # Ensure terminal punctuation -- StyleTTS2 trains better with
    # consistent sentence endings.
    if t and t[-1] not in ".!?":
        t = t + "."
    return t


# ---------------------------------------------------------------------------
# WAV validation
# ---------------------------------------------------------------------------


def _validate_wav(path: Path, expected_sr: int = 24000) -> tuple[bool, str]:
    """Open the WAV and check basic invariants. Returns
    ``(is_valid, reason_if_not)``."""
    try:
        with wave.open(str(path), "rb") as w:
            if w.getnchannels() != 1:
                return False, f"channels={w.getnchannels()} (need mono)"
            if w.getsampwidth() != 2:
                return False, f"sampwidth={w.getsampwidth()} (need 16-bit)"
            sr = w.getframerate()
            if sr != expected_sr:
                return False, f"sample_rate={sr} (expected {expected_sr})"
            n = w.getnframes()
            if n < 1:
                return False, "empty (no frames)"
            return True, ""
    except Exception as e:
        return False, f"open failed: {e}"


# ---------------------------------------------------------------------------
# Main packaging
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--source", required=True, action="append",
        help="Path to a completed bulk_eval_<ts>/ run directory. "
             "Repeat for multi-pass corpora (e.g., 3 passes at different "
             "temperatures): --source pass1/ --source pass2/ --source pass3/. "
             "Output IDs are prefixed with the source dir name to keep "
             "them unique across passes.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output dir. Default: ultronVoiceAudio/kokoro_training_corpus_<ts>/",
    )
    parser.add_argument(
        "--min-duration", type=float, default=1.0,
        help="Drop clips shorter than this many seconds (default 1.0).",
    )
    parser.add_argument(
        "--max-duration", type=float, default=12.0,
        help="Drop clips longer than this many seconds (default 12.0).",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.05,
        help="Fraction held out for val.csv (default 0.05 = 5%%).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Deterministic shuffle seed (default 42).",
    )
    parser.add_argument(
        "--expected-sr", type=int, default=24000,
        help="Required sample rate (default 24000 -- Kokoro native).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show counts without writing anything.",
    )
    args = parser.parse_args(argv)

    sources = [_resolve(s) for s in args.source]
    for s in sources:
        if not s.is_dir():
            print(f"ERROR: source {s} not a directory")
            return 1
        if not (s / "manifest.json").is_file():
            print(f"ERROR: manifest not found at {s / 'manifest.json'}")
            return 1

    if args.output:
        output = _resolve(args.output)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output = PROJECT_ROOT / "ultronVoiceAudio" / f"kokoro_training_corpus_{ts}"

    print("Kokoro corpus packaging")
    print("-" * 60)
    print(f"  sources:     {len(sources)} pass(es)")
    for s in sources:
        print(f"    - {s.name}")
    print(f"  output:      {output}")
    print(f"  duration:    {args.min_duration}-{args.max_duration} s")
    print(f"  val_fraction:{args.val_fraction}")
    print(f"  seed:        {args.seed}")
    print(f"  expected sr: {args.expected_sr}")
    print(f"  dry_run:     {args.dry_run}")
    print()

    if not args.dry_run:
        if output.exists():
            print(f"ERROR: output dir {output} already exists. "
                  f"Delete it or pass a different --output.")
            return 1

    # Load manifests from all source dirs, prefixing each entry's id
    # with the source dir name so multi-pass corpora don't collide.
    manifest: list[dict] = []
    for source in sources:
        mlist = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        for e in mlist:
            # Disambiguate id across passes by prepending the source
            # dir name. wav stays a path *relative to its source* --
            # we resolve it to an absolute path in the accept loop
            # below so the multi-source case works seamlessly.
            e["_source_dir"] = str(source)
            e["_source_name"] = source.name
            e["_original_id"] = e.get("id")
            e["id"] = f"{source.name}__{e.get('id')}"
            manifest.append(e)
    print(f"  manifest entries (union): {len(manifest)}")

    # Filter + validate.
    accepted = []
    rejected: list[tuple[str, str]] = []
    for entry in manifest:
        eid = entry.get("id")
        cat = entry.get("category", "unknown")
        text = entry.get("text", "")
        rel_wav = entry.get("wav")
        if not rel_wav:
            rejected.append((eid, f"synth error: {entry.get('error', 'unknown')}"))
            continue
        # Resolve wav path against THIS entry's source dir (multi-pass safe).
        wav_path = Path(entry["_source_dir"]) / rel_wav
        if not wav_path.is_file():
            rejected.append((eid, f"wav missing on disk: {rel_wav}"))
            continue
        ok, reason = _validate_wav(wav_path, expected_sr=args.expected_sr)
        if not ok:
            rejected.append((eid, reason))
            continue
        duration = float(entry.get("duration_s", 0.0))
        if duration < args.min_duration:
            rejected.append((eid, f"too short ({duration:.2f}s)"))
            continue
        if duration > args.max_duration:
            rejected.append((eid, f"too long ({duration:.2f}s)"))
            continue
        accepted.append({
            "id": eid,
            "category": cat,
            "text": text,
            "normalized": _normalize_for_training(text),
            "wav": wav_path,
            "rel_wav": rel_wav,
            "duration_s": duration,
        })

    print(f"  accepted:    {len(accepted)}")
    print(f"  rejected:    {len(rejected)}")
    if rejected:
        # Group rejection reasons for the operator's eye.
        reason_counts = Counter(reason.split(":")[0] for _, reason in rejected)
        for r, n in reason_counts.most_common():
            print(f"    {r}: {n}")

    if not accepted:
        print("ERROR: no entries accepted; aborting.")
        return 1

    # Stats
    by_cat = Counter(e["category"] for e in accepted)
    durations = [e["duration_s"] for e in accepted]
    total_dur = sum(durations)
    print()
    print(f"  total audio: {total_dur/60:.1f} min ({total_dur:.0f} sec)")
    print(f"  mean clip:   {total_dur/len(accepted):.2f} sec")
    print(f"  shortest:    {min(durations):.2f} sec")
    print(f"  longest:     {max(durations):.2f} sec")
    print(f"  per-category:")
    for cat, n in by_cat.most_common():
        print(f"    {cat:<22} {n}")

    if args.dry_run:
        print()
        print("DRY RUN -- nothing written. Drop --dry-run to actually package.")
        return 0

    # Create output structure.
    output.mkdir(parents=True, exist_ok=False)
    wavs_dir = output / "wavs"
    wavs_dir.mkdir()

    print()
    print("  copying WAVs + writing CSVs...")
    metadata_lines = []
    for entry in accepted:
        dest_name = f"{entry['id']}.wav"
        dest_path = wavs_dir / dest_name
        shutil.copyfile(entry["wav"], dest_path)
        # LJSpeech: id|raw|normalized
        metadata_lines.append(
            "|".join((entry["id"], entry["text"], entry["normalized"]))
        )

    # Deterministic shuffle + split.
    rng = random.Random(args.seed)
    indices = list(range(len(metadata_lines)))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(indices) * args.val_fraction)))
    val_idx = set(indices[:val_count])
    train_lines = [metadata_lines[i] for i in indices[val_count:]]
    val_lines = [metadata_lines[i] for i in indices[:val_count]]

    # Write CSVs (LJSpeech-style pipe-delimited; UTF-8; no header per spec).
    (output / "metadata.csv").write_text(
        "\n".join(metadata_lines) + "\n", encoding="utf-8",
    )
    (output / "train.csv").write_text(
        "\n".join(train_lines) + "\n", encoding="utf-8",
    )
    (output / "val.csv").write_text(
        "\n".join(val_lines) + "\n", encoding="utf-8",
    )

    # Write stats.json.
    stats = {
        "source_runs": [s.name for s in sources],
        "manifest_entries_total": len(manifest),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejection_reasons": dict(Counter(
            r.split(":")[0] for _, r in rejected
        )),
        "audio_duration_seconds": total_dur,
        "audio_duration_minutes": total_dur / 60,
        "mean_clip_seconds": total_dur / len(accepted),
        "shortest_seconds": min(durations),
        "longest_seconds": max(durations),
        "per_category": dict(by_cat),
        "train_count": len(train_lines),
        "val_count": len(val_lines),
        "split": {
            "val_fraction": args.val_fraction,
            "seed": args.seed,
        },
        "filters": {
            "min_duration_seconds": args.min_duration,
            "max_duration_seconds": args.max_duration,
            "expected_sample_rate": args.expected_sr,
        },
    }
    (output / "stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8",
    )

    # Write README.
    backslash = "\\"
    source_args_block = "\n".join(
        f"    --source {s} {backslash}" for s in sources
    )
    readme = f"""# Ultron Kokoro fine-tune training corpus

Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Lineage

This corpus was synthesised by XTTS-v2 fine-tuned on the Ultron voice
reference at `best_model_102.pth` (epoch 17 of the
2026-05-21 retrain). Each clip went through the production
post-processing pipeline:

1. **Phantom trims** -- four-stage cleanup of XTTS lead/trail artifacts:
   - Gapped lead trim (silence -> blip -> silence -> word)
   - Soft lead trim (silence -> soft ramp into word)
   - Gapped tail trim (word -> silence -> blip -> silence)
   - Soft trail trim (word -> continuous decay into silence)
2. **Spectral magnitude median filter** (window=3) -- smooths
   frame-to-frame harmonic micro-variations (the "shakiness" the
   model sometimes produces).
3. **v2custom Pedalboard chain** -- pitch shift -0.8 semitones,
   chorus, 0.16 reverb wet, 5 dB distortion, EQ.

Source bulk-eval run(s): {", ".join(f"`{s.name}`" for s in sources)}

## Stats

- Accepted: **{len(accepted)}** / {len(manifest)} entries
- Rejected: {len(rejected)} (duration outside [{args.min_duration}, {args.max_duration}] s window, or WAV invalid)
- Total audio: **{total_dur/60:.1f} min** ({total_dur:.0f} s)
- Mean clip: {total_dur/len(accepted):.2f} s
- Shortest: {min(durations):.2f} s
- Longest: {max(durations):.2f} s

Per-category breakdown:

| Category | Count |
|---|---|
{chr(10).join(f"| {cat} | {n} |" for cat, n in by_cat.most_common())}

## Format

LJSpeech-style. Compatible with StyleTTS2 fine-tune tooling (Kokoro
is StyleTTS2 + ISTFTNet).

- `wavs/<id>.wav` -- mono PCM_16 at {args.expected_sr} Hz.
- `metadata.csv` -- pipe-delimited `id|raw_text|normalized_text`, UTF-8,
  no header.
- `train.csv` / `val.csv` -- deterministic 95/5 split with
  `seed={args.seed}`. {len(train_lines)} train rows, {len(val_lines)} val rows.

## Reproducibility

```
python ultronVoiceAudio/scripts/package_kokoro_corpus.py \\
{source_args_block}
    --output {output} \\
    --min-duration {args.min_duration} \\
    --max-duration {args.max_duration} \\
    --val-fraction {args.val_fraction} \\
    --seed {args.seed}
```
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    print()
    print("=" * 60)
    print(f"SUCCESS: corpus packaged at {output}")
    print(f"  {len(accepted)} clips ({total_dur/60:.1f} min)")
    print(f"  train: {len(train_lines)} | val: {len(val_lines)}")
    print(f"  metadata: {output / 'metadata.csv'}")
    print(f"  stats:    {output / 'stats.json'}")
    print(f"  readme:   {output / 'README.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
