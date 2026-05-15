"""STT engine latency benchmark.

Loads the configured Whisper engine and measures transcription latency
on synthetic 16 kHz audio of varying lengths (1s, 3s, 5s, 8s). Reports
median / p95 / RTF for each length.

Used to right-size the Phase 4 STT swap decision: if current Whisper
is fast enough at typical voice-query lengths, the swap is a smaller
win than the research suggested.

Run from main checkout (or set ULTRON_LLM_MODEL_PATH=...) so models
resolve.

Usage:

    python scripts/bench_stt_latency.py
    python scripts/bench_stt_latency.py --lengths 1,3,5,8 --trials 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import ultron  # noqa: F401


def _make_speech_like(seconds: float, *, sr: int = 16000) -> np.ndarray:
    """Synthesise audio with speech-like spectral envelope.

    Not real speech (Whisper will hallucinate or emit empty text on
    pure synth) but realistic enough to exercise the inference path
    end-to-end. Latency is dominated by model FLOPs + audio shape,
    so content doesn't materially affect the number we care about.
    """
    n = int(seconds * sr)
    rng = np.random.default_rng(seed=42)
    # Pink-ish noise with 100 Hz fundamental + harmonics, ducked at edges.
    t = np.arange(n, dtype=np.float32) / sr
    voice_band = 0.3 * np.sin(2 * np.pi * 220 * t)
    voice_band += 0.15 * np.sin(2 * np.pi * 440 * t)
    voice_band += 0.08 * np.sin(2 * np.pi * 880 * t)
    # Add formant-like modulation
    voice_band *= 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)
    # Pink noise overlay
    noise = rng.standard_normal(n).astype(np.float32) * 0.02
    audio = voice_band + noise
    # 50 ms edge ramps
    ramp_samples = int(0.05 * sr)
    if ramp_samples * 2 < n:
        ramp = np.linspace(0, 1, ramp_samples, dtype=np.float32)
        audio[:ramp_samples] *= ramp
        audio[-ramp_samples:] *= ramp[::-1]
    return audio


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths", type=str, default="1,3,5,8",
        help="Comma-separated audio durations in seconds.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args(argv)

    lengths_s = [float(x) for x in args.lengths.split(",") if x.strip()]

    from ultron.transcription import WhisperEngine

    print("  loading WhisperEngine...", flush=True)
    t_load = time.monotonic()
    engine = WhisperEngine()
    print(f"  loaded in {time.monotonic() - t_load:.2f}s", flush=True)
    print(
        f"  model={engine.model_name} device={engine.device} "
        f"compute_type={engine.compute_type} beam={engine.beam_size}",
        flush=True,
    )

    # Warmup at each length so first-shot cost (CUDA kernel JIT) is
    # paid before measurement.
    print("\n  warming up...", flush=True)
    for w in range(args.warmup):
        for length in lengths_s:
            audio = _make_speech_like(length)
            engine.transcribe(audio)

    # Trials
    print("\n  benchmark...", flush=True)
    results: dict = {}
    for length in lengths_s:
        durs = []
        for trial in range(args.trials):
            audio = _make_speech_like(length)
            t0 = time.monotonic()
            _text = engine.transcribe(audio)
            ms = (time.monotonic() - t0) * 1000
            durs.append(ms)
            print(f"  length={length}s trial={trial + 1}/{args.trials}: {ms:.0f} ms",
                  flush=True)
        # Aggregate
        durs_sorted = sorted(durs)
        median = durs_sorted[len(durs_sorted) // 2]
        p95_idx = max(0, int(len(durs_sorted) * 0.95) - 1)
        p95 = durs_sorted[p95_idx]
        results[length] = {
            "median_ms": round(median, 1),
            "p95_ms": round(p95, 1),
            "min_ms": round(min(durs), 1),
            "max_ms": round(max(durs), 1),
            "rtf": round(median / 1000 / length, 3),
            "samples": [round(d, 1) for d in durs],
        }

    print("\n## STT latency by audio length\n")
    print("| length (s) | median (ms) | p95 (ms) | min (ms) | max (ms) | RTF |")
    print("|---|---|---|---|---|---|")
    for length in lengths_s:
        r = results[length]
        print(
            f"| {length} | {r['median_ms']} | {r['p95_ms']} | "
            f"{r['min_ms']} | {r['max_ms']} | {r['rtf']} |"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
