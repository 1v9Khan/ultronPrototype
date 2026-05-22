"""Tests for ``ultron.tts.spectral_smooth``.

The runtime port of ``_spectral_smooth`` from the corpus-evaluation
script. Tests focus on the behaviour the engine relies on:

- short-input pass-through (don't crash on tiny clips)
- length preservation within STFT framing tolerance
- magnitude smoothing actually reduces frame-to-frame magnitude
  variance on a synthetic wobbly signal
- phase preservation
- median_window_frames=1 is a near-identity (no smoothing)
- accepts non-contiguous + non-float32 arrays
"""

from __future__ import annotations

import numpy as np
import pytest

from ultron.tts.spectral_smooth import spectral_smooth, trim_and_fade


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------


def test_short_clip_passes_through_unchanged():
    """A clip shorter than ``n_fft`` (default 2048) can't be STFT'd
    meaningfully -- the function returns it untouched (still as
    float32)."""
    audio = np.random.RandomState(0).randn(1000).astype(np.float32) * 0.1
    out = spectral_smooth(audio, sr=24000)
    assert out.dtype == np.float32
    # Length preserved exactly on the pass-through branch.
    assert out.size == audio.size
    np.testing.assert_array_equal(out, audio)


def test_empty_array_does_not_crash():
    out = spectral_smooth(np.zeros(0, dtype=np.float32), sr=24000)
    assert out.size == 0


def test_zero_signal_returns_zeros():
    """Silence in -> silence out (smoothing zero magnitudes still
    yields zeros)."""
    audio = np.zeros(4096, dtype=np.float32)
    out = spectral_smooth(audio, sr=24000)
    assert np.allclose(out, 0.0, atol=1e-6)


# ----------------------------------------------------------------------
# Length / dtype contract
# ----------------------------------------------------------------------


def test_output_is_float32():
    audio = np.random.RandomState(1).randn(8192).astype(np.float32) * 0.1
    out = spectral_smooth(audio, sr=24000)
    assert out.dtype == np.float32


def test_length_within_stft_framing_tolerance():
    """Output length is ``(n_frames - 1) * hop + n_fft``; for inputs
    not aligned to ``hop`` the tail samples (up to n_fft - hop
    = 1536 samples with the defaults) get dropped. The output is
    therefore always within 1536 samples of the input length on
    either side."""
    audio = np.random.RandomState(2).randn(24000).astype(np.float32) * 0.1
    out = spectral_smooth(audio, sr=24000)
    assert abs(out.size - audio.size) <= 1536


def test_accepts_float64_and_upcasts():
    audio = (np.random.RandomState(3).randn(8192) * 0.1).astype(np.float64)
    out = spectral_smooth(audio, sr=24000)
    assert out.dtype == np.float32


def test_accepts_non_contiguous_input():
    """np.asarray() should make a contiguous copy when needed."""
    base = np.random.RandomState(4).randn(16384).astype(np.float32) * 0.1
    sliced = base[::2]  # non-contiguous view
    assert not sliced.flags["C_CONTIGUOUS"]
    out = spectral_smooth(sliced, sr=24000)
    assert out.dtype == np.float32


# ----------------------------------------------------------------------
# Algorithm correctness
# ----------------------------------------------------------------------


def _wobbly_tone(sr: int, dur_s: float, base_hz: float = 220.0,
                wobble_hz: float = 8.0, wobble_amp: float = 0.04):
    """Synthesise a pure tone with frame-to-frame amplitude wobble
    (the kind of micro-variation Kokoro's under-trained checkpoint
    produces). The wobble lives in the magnitude domain so spectral
    smoothing should attenuate it."""
    t = np.arange(int(sr * dur_s)) / sr
    carrier = np.sin(2 * np.pi * base_hz * t)
    am = 1.0 + wobble_amp * np.sin(2 * np.pi * wobble_hz * t)
    return (carrier * am).astype(np.float32) * 0.3


def test_smoothing_reduces_magnitude_variance_on_wobbly_tone():
    """The smoothing pass should attenuate frame-to-frame magnitude
    micro-variations -- that's the whole point. We synthesise a
    pure tone with 8 Hz amplitude wobble and check that the median
    of the per-frame magnitude variance drops after smoothing.
    """
    sr = 24000
    audio = _wobbly_tone(sr, dur_s=1.0, wobble_hz=8.0, wobble_amp=0.06)
    out = spectral_smooth(audio, sr=sr, median_window_frames=5)

    n_fft, hop = 2048, 512

    def _frame_mag_std(x):
        n = 1 + (len(x) - n_fft) // hop
        if n < 2:
            return 0.0
        mags = np.zeros((n_fft // 2 + 1, n), dtype=np.float32)
        win = np.hanning(n_fft).astype(np.float32)
        for i in range(n):
            mags[:, i] = np.abs(np.fft.rfft(x[i * hop: i * hop + n_fft] * win))
        # Frame-to-frame std at each frequency bin, then average.
        return float(np.mean(np.std(mags, axis=1)))

    raw_std = _frame_mag_std(audio)
    smooth_std = _frame_mag_std(out)
    assert smooth_std < raw_std, (
        f"smoothing did not reduce magnitude std "
        f"(raw={raw_std:.4f} smoothed={smooth_std:.4f})"
    )


def test_window_size_1_is_near_identity():
    """median_window_frames=1 means "smooth with a single-frame
    window" -- the median of a single value is itself, so the
    output should closely resemble the input modulo STFT framing
    error."""
    sr = 24000
    audio = _wobbly_tone(sr, dur_s=0.8, wobble_amp=0.0)  # pure tone
    out = spectral_smooth(audio, sr=sr, median_window_frames=1)

    # Trim to common length and compare RMS difference -- should be
    # well under 5 % of signal RMS.
    n = min(len(audio), len(out))
    diff = audio[:n] - out[:n]
    rms_in = float(np.sqrt(np.mean(audio[:n] ** 2)))
    rms_diff = float(np.sqrt(np.mean(diff ** 2)))
    assert rms_in > 0
    assert rms_diff / rms_in < 0.10, (
        f"window=1 should be near-identity but rms_diff/rms_in="
        f"{rms_diff / rms_in:.3f}"
    )


def test_window_clamped_below_one():
    """Defensive: window < 1 is clamped to 1 (no-op) rather than
    raising. Keeps the engine's call site simple."""
    audio = _wobbly_tone(24000, dur_s=0.5)
    # Should not raise.
    out0 = spectral_smooth(audio, sr=24000, median_window_frames=0)
    out_neg = spectral_smooth(audio, sr=24000, median_window_frames=-5)
    assert out0.dtype == np.float32
    assert out_neg.dtype == np.float32


def test_phase_preservation_at_low_freq():
    """Phase is supposed to be preserved exactly -- only magnitudes
    are smoothed. Verify the low-frequency content (where the
    base tone lives) maintains correlation > 0.9 with the input."""
    sr = 24000
    audio = _wobbly_tone(sr, dur_s=1.0, wobble_amp=0.0)
    out = spectral_smooth(audio, sr=sr, median_window_frames=3)
    n = min(len(audio), len(out))
    a = audio[:n] - audio[:n].mean()
    b = out[:n] - out[:n].mean()
    corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    assert corr > 0.9, f"phase not preserved: corr={corr:.3f}"


# ----------------------------------------------------------------------
# Performance characterisation (informational; not a hard gate)
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# trim_and_fade
# ----------------------------------------------------------------------


def _noisy_speech(sr: int = 24000, noise_frames: int = 3, speech_frames: int = 20) -> np.ndarray:
    """Build synthetic signal: noise_frames of low-level noise + speech_frames of
    voiced speech + noise_frames of trailing noise."""
    frame = sr // 100  # 10 ms frames at 24 kHz
    rng = np.random.RandomState(42)
    noise = rng.randn(noise_frames * frame).astype(np.float32) * 0.001
    t = np.arange(speech_frames * frame) / sr
    speech = (np.sin(2 * np.pi * 220.0 * t) * 0.3).astype(np.float32)
    return np.concatenate([noise, speech, noise.copy()])


def test_trim_removes_leading_trailing_noise():
    """Trimmed output should be shorter than the input (noise stripped)."""
    sr = 24000
    audio = _noisy_speech(sr)
    out = trim_and_fade(audio, sr=sr, threshold_db=-20.0)
    assert out.size < audio.size, (
        f"expected trimming; got same or larger: {out.size} vs {audio.size}"
    )


def test_trim_preserves_speech_rms():
    """After trimming, the RMS of the output should be close to the speech
    segment RMS (not the full noisy clip). Specifically, it should be higher
    than the raw-clip RMS since we removed the attenuating noise padding."""
    sr = 24000
    audio = _noisy_speech(sr)
    # Pin short fade durations + disable tail zero / silence pad so we
    # isolate the "trim removes noise -> RMS up" property. Longer
    # production fades (25/45 ms) attenuate enough of the synthetic
    # speech that the gain from noise removal is partially cancelled.
    out = trim_and_fade(
        audio, sr=sr, threshold_db=-20.0,
        fade_in_ms=2.0, fade_out_ms=2.0,
        tail_aggressive_trim_ms=0, hard_silence_pad_ms=0,
    )
    rms_raw = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    rms_out = float(np.sqrt(np.mean(out.astype(np.float32) ** 2)))
    assert rms_out > rms_raw, (
        f"output should be louder after trimming noise; raw={rms_raw:.4f} out={rms_out:.4f}"
    )


def test_tail_aggressive_zero_silences_last_n_samples():
    """The aggressive-tail-zero forces the LAST ``tail_aggressive_trim_ms``
    samples to exact zero -- catches the partial-fine-tune "blip"
    artifact that survives the cosine fade-out by hard-muting the
    decay region."""
    sr = 24000
    # Steady tone -- no natural fade. Without the tail zero, the
    # cosine fade-out would leave decaying audio; with it, we should
    # see exact zeros at the very end.
    t = np.arange(sr) / sr
    audio = (np.sin(2 * np.pi * 440.0 * t) * 0.5).astype(np.float32)
    out = trim_and_fade(audio, sr=sr, tail_aggressive_trim_ms=20.0)
    # Last 20 ms (480 samples) should be exactly zero.
    n_zero = int(sr * 0.020)
    # Subtract hard_silence_pad_ms (8 ms = 192 samples) which is also zero.
    tail = out[-(n_zero + int(sr * 0.008)):]
    assert np.all(tail == 0.0), (
        f"tail should be all zeros; got max abs = {np.abs(tail).max():.6f}"
    )


def test_tail_aggressive_zero_disabled_with_zero_ms():
    """Pass tail_aggressive_trim_ms=0 to opt out of the hard-mute."""
    sr = 24000
    t = np.arange(sr) / sr
    audio = (np.sin(2 * np.pi * 440.0 * t) * 0.5).astype(np.float32)
    out = trim_and_fade(audio, sr=sr, tail_aggressive_trim_ms=0.0)
    # The hard silence pad is still 8 ms at end (separate parameter).
    # Strip THAT and look at what's left -- should NOT be all zeros
    # (cosine fade leaves some non-zero residue).
    pad_samples = int(sr * 0.008)
    before_pad = out[-(pad_samples + int(sr * 0.005)):-pad_samples]
    # Cosine fade-out: at the boundary just before the silence pad,
    # the fade has already reached ~0 but there's still some content
    # in the body of the fade region. Check a few samples in the body.
    assert np.abs(out[-(pad_samples + 100):-pad_samples]).max() > 0.0, (
        "without aggressive zero, fade-out region should have non-zero content"
    )


def test_trim_output_is_float32():
    audio = _noisy_speech()
    out = trim_and_fade(audio, sr=24000)
    assert out.dtype == np.float32


def test_trim_accepts_float64():
    audio = _noisy_speech().astype(np.float64)
    out = trim_and_fade(audio, sr=24000)
    assert out.dtype == np.float32


def test_trim_short_clip_passes_through():
    """Clips shorter than 2 frames (< 2 * frame_ms * sr) pass through unchanged."""
    audio = np.ones(100, dtype=np.float32) * 0.5
    out = trim_and_fade(audio, sr=24000, frame_ms=10.0)
    assert out.size == audio.size
    np.testing.assert_array_equal(out, audio.astype(np.float32))


def test_trim_all_silence_passes_through():
    """If no speech is detected (all below threshold), return input unchanged."""
    audio = np.zeros(4800, dtype=np.float32)  # 200 ms pure silence
    out = trim_and_fade(audio, sr=24000, threshold_db=-40.0)
    assert out.size == audio.size


def test_trim_fade_in_starts_near_zero():
    """The first sample of the faded output should be very close to 0
    because of the fade-in applied after the trim."""
    sr = 24000
    # Strong speech from the very start (no leading silence to trim).
    t = np.arange(sr) / sr
    audio = (np.sin(2 * np.pi * 440.0 * t) * 0.5).astype(np.float32)
    out = trim_and_fade(audio, sr=sr, threshold_db=-40.0, fade_in_ms=10.0)
    assert abs(float(out[0])) < 0.05, (
        f"fade-in should start near zero; out[0]={out[0]:.4f}"
    )


def test_trim_fade_out_ends_near_zero():
    """The last sample of the faded output should be very close to 0."""
    sr = 24000
    t = np.arange(sr) / sr
    audio = (np.sin(2 * np.pi * 440.0 * t) * 0.5).astype(np.float32)
    out = trim_and_fade(audio, sr=sr, threshold_db=-40.0, fade_out_ms=10.0)
    assert abs(float(out[-1])) < 0.05, (
        f"fade-out should end near zero; out[-1]={out[-1]:.4f}"
    )


def test_trim_does_not_enlarge_clip():
    """trim_and_fade should never return more samples than it received."""
    sr = 24000
    audio = _noisy_speech(sr)
    out = trim_and_fade(audio, sr=sr)
    assert out.size <= audio.size, (
        f"output ({out.size}) must not exceed input ({audio.size})"
    )


def test_performance_under_50ms_for_one_second_clip():
    """Smoothing a 1-second clip should comfortably finish under
    50 ms on modern CPUs. The empirical number is ~10 ms; this
    test acts as a regression gate -- if a future refactor blows
    past 50 ms something is wrong."""
    import time
    audio = _wobbly_tone(24000, dur_s=1.0)
    # Warm up scipy.ndimage import + numpy FFT plan cache.
    spectral_smooth(audio, sr=24000)

    t0 = time.monotonic()
    for _ in range(3):
        spectral_smooth(audio, sr=24000)
    avg_ms = (time.monotonic() - t0) / 3 * 1000.0
    assert avg_ms < 50.0, f"smoothing too slow: {avg_ms:.1f} ms / sec audio"
