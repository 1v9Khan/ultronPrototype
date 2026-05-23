"""End-to-end latency benchmark.

Measures cold and warm timings for each pipeline stage on a fixed audio
sample. Useful for verifying the under-2-second budget after install.

Usage:
    python scripts/benchmark.py [path/to/sample.wav]

If no WAV is given, a synthetic 3-second 440 Hz tone is used (transcription
will be empty, but the timings are still meaningful).
"""

from __future__ import annotations

import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import settings  # noqa: E402
from ultron.utils.logging import configure_logging  # noqa: E402


def _load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != settings.SAMPLE_RATE or wav.getnchannels() != 1:
            raise ValueError(
                f"Expected mono {settings.SAMPLE_RATE} Hz, got "
                f"{wav.getnchannels()}ch @ {wav.getframerate()} Hz"
            )
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _synthetic_tone(seconds: float = 3.0) -> np.ndarray:
    t = np.arange(int(seconds * settings.SAMPLE_RATE)) / settings.SAMPLE_RATE
    return (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _stat(label: str, samples_ms: list[float]) -> None:
    if not samples_ms:
        return
    print(
        f"  {label:<22s} "
        f"min={min(samples_ms):>7.1f} ms  "
        f"median={statistics.median(samples_ms):>7.1f} ms  "
        f"max={max(samples_ms):>7.1f} ms  "
        f"(n={len(samples_ms)})"
    )


def main() -> int:
    configure_logging(level="WARNING")

    audio_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    audio = _load_wav(audio_path) if audio_path else _synthetic_tone()
    print(f"\nBenchmarking with {len(audio) / settings.SAMPLE_RATE:.2f}s of audio\n")

    print("Loading components...")
    from ultron.llm import LLMEngine
    from ultron.transcription import make_stt_engine
    from ultron.tts import make_tts_engine

    t0 = time.monotonic()
    stt = make_stt_engine()
    print(f"  STT loaded in {time.monotonic() - t0:.1f}s ({type(stt).__name__})")

    t0 = time.monotonic()
    llm = LLMEngine()
    print(f"  LLM loaded in {time.monotonic() - t0:.1f}s")

    t0 = time.monotonic()
    _rvc, tts = make_tts_engine()
    print(f"  TTS loaded in {time.monotonic() - t0:.1f}s ({type(tts).__name__})")

    print("\nWarming up...")
    stt.transcribe(audio)
    llm.generate("Say 'ready' and nothing else.")

    print("\nMeasuring (5 runs each):")
    stt_times: list[float] = []
    llm_ttft: list[float] = []
    tts_synth: list[float] = []

    for _ in range(5):
        t = time.monotonic()
        text = stt.transcribe(audio)
        stt_times.append((time.monotonic() - t) * 1000)

        t = time.monotonic()
        first = True
        for _tok in llm.generate_stream(text or "Tell me a one-sentence joke."):
            if first:
                llm_ttft.append((time.monotonic() - t) * 1000)
                first = False
            # consume the rest without timing
        if first:
            llm_ttft.append((time.monotonic() - t) * 1000)

        t = time.monotonic()
        tts._synthesize("This is a benchmark sentence for synthesis timing.")  # noqa: SLF001
        tts_synth.append((time.monotonic() - t) * 1000)

    _stat(f"{type(stt).__name__} transcribe", stt_times)
    _stat("LLM time-to-first-token", llm_ttft)
    _stat("TTS sentence synth", tts_synth)

    budget = 2000
    end_to_end = (
        statistics.median(stt_times)
        + statistics.median(llm_ttft)
        + statistics.median(tts_synth)
    )
    print(f"\n  estimated end-to-end (median): {end_to_end:.0f} ms "
          f"(budget {budget} ms)")
    print(f"  {'PASS' if end_to_end <= budget else 'OVER BUDGET'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
