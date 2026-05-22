"""Identity-preserving augmentation of Ultron_vocals_only_clean_v1.wav.

Generates 12 variants that preserve voice character (no large pitch shifts, no
heavy effects, no noise injection). Each variant adds prosodic or temporal
variance suitable for Kokoro style-vector training data.

Outputs land alongside the source in ultronVoiceAudio/ as
Ultron_vocals_aug_<type>_<value>_v1.wav. The original is never modified.
"""
from __future__ import annotations

import pathlib
import time

import librosa
import numpy as np
import scipy.signal
import soundfile as sf
from pedalboard import HighShelfFilter, LowShelfFilter, Pedalboard, PitchShift

SRC = pathlib.Path(
    "C:/STC/ultronPrototype/ultronVoiceAudio/Ultron_vocals_only_clean_v1.wav"
)
OUT_DIR = SRC.parent
SUBTYPE = "PCM_16"


def load_source() -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(SRC), dtype="float32")
    return audio, sr


def save(name: str, data: np.ndarray, sample_rate: int) -> pathlib.Path:
    out = OUT_DIR / name
    # clip to avoid wrap-around on int16 conversion
    data = np.clip(data, -1.0, 1.0)
    sf.write(str(out), data, sample_rate, subtype=SUBTYPE)
    dur = len(data) / sample_rate
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"  wrote {name:55s} {dur:6.2f}s  {size_mb:5.1f} MB")
    return out


def time_stretch_stereo(audio: np.ndarray, rate: float) -> np.ndarray:
    """Time-stretch preserving pitch (librosa phase vocoder)."""
    if audio.ndim == 1:
        return librosa.effects.time_stretch(y=audio, rate=rate)
    chans = [
        librosa.effects.time_stretch(y=audio[:, c], rate=rate)
        for c in range(audio.shape[1])
    ]
    min_len = min(c.shape[0] for c in chans)
    return np.stack([c[:min_len] for c in chans], axis=1)


def pitch_shift_cents(audio: np.ndarray, sr: int, cents: float) -> np.ndarray:
    """Pitch-shift preserving duration (pedalboard / Rubber Band)."""
    semitones = cents / 100.0
    board = Pedalboard([PitchShift(semitones=semitones)])
    return board(audio, sr)


def speed_perturb(audio: np.ndarray, rate: float) -> np.ndarray:
    """Resample-based speed change. Affects both pitch and duration.

    Natural slow/fast speech effect (think Kaldi speed perturbation).
    """
    new_len = int(len(audio) / rate)
    if audio.ndim == 1:
        return scipy.signal.resample(audio, new_len).astype(np.float32)
    chans = [
        scipy.signal.resample(audio[:, c], new_len).astype(np.float32)
        for c in range(audio.shape[1])
    ]
    return np.stack(chans, axis=1)


def eq_tilt(
    audio: np.ndarray,
    sr: int,
    low_gain_db: float = 0.0,
    high_gain_db: float = 0.0,
) -> np.ndarray:
    """Subtle EQ tilt for spectral variance."""
    filters = []
    if low_gain_db != 0.0:
        filters.append(LowShelfFilter(cutoff_frequency_hz=200.0, gain_db=low_gain_db))
    if high_gain_db != 0.0:
        filters.append(
            HighShelfFilter(cutoff_frequency_hz=5000.0, gain_db=high_gain_db)
        )
    if not filters:
        return audio
    board = Pedalboard(filters)
    return board(audio, sr)


def main() -> None:
    print(f"loading {SRC.name}...")
    audio, sr = load_source()
    print(
        f"  source: {len(audio) / sr:.2f}s, {sr} Hz, "
        f"{audio.shape[1] if audio.ndim > 1 else 1} ch, "
        f"peak={np.abs(audio).max():.3f}"
    )
    print()

    plan = [
        # time-stretch (preserves pitch) -- 4 variants
        ("stretch_092", lambda a: time_stretch_stereo(a, 0.92)),
        ("stretch_096", lambda a: time_stretch_stereo(a, 0.96)),
        ("stretch_104", lambda a: time_stretch_stereo(a, 1.04)),
        ("stretch_108", lambda a: time_stretch_stereo(a, 1.08)),
        # pitch-shift (preserves duration) -- 4 variants, all sub-semitone
        ("pitch_down30c", lambda a: pitch_shift_cents(a, sr, -30.0)),
        ("pitch_up30c", lambda a: pitch_shift_cents(a, sr, +30.0)),
        ("pitch_down50c", lambda a: pitch_shift_cents(a, sr, -50.0)),
        ("pitch_up50c", lambda a: pitch_shift_cents(a, sr, +50.0)),
        # speed perturbation (changes both pitch and duration) -- 2 variants
        ("speed_096", lambda a: speed_perturb(a, 0.96)),
        ("speed_104", lambda a: speed_perturb(a, 1.04)),
        # subtle EQ tilt -- 2 variants
        ("eq_warm", lambda a: eq_tilt(a, sr, low_gain_db=1.5)),
        ("eq_bright", lambda a: eq_tilt(a, sr, high_gain_db=1.5)),
    ]

    total_duration_s = 0.0
    for label, fn in plan:
        t0 = time.perf_counter()
        out = fn(audio)
        elapsed = time.perf_counter() - t0
        name = f"Ultron_vocals_aug_{label}_v1.wav"
        save(name, out, sr)
        total_duration_s += len(out) / sr
        print(f"    ({elapsed:.1f}s wall)")

    print()
    print(
        f"original duration: {len(audio) / sr / 60:.2f} min  "
        f"({len(audio) / sr:.1f} s)"
    )
    print(
        f"augmented total:   {total_duration_s / 60:.2f} min  "
        f"({total_duration_s:.1f} s) across {len(plan)} variants"
    )
    print(
        f"combined corpus:   {(len(audio) / sr + total_duration_s) / 60:.2f} min"
    )


if __name__ == "__main__":
    main()
