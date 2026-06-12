"""Spectral magnitude smoothing for Kokoro fine-tune output.

The partially-trained Ultron Kokoro checkpoint (Stage 1 complete +
Stage 2 epoch 0 only -- SLM joint adversarial training at epoch 3+
never ran) produces audible **pitch wobble / shakiness**. The
right long-term fix is more training; the short-term fix is a
lightweight DSP smoothing pass that masks frame-to-frame harmonic
micro-variations without smearing consonants.

Algorithm: STFT -> median-filter magnitudes across time (NOT
frequency) -> ISTFT with original phase. The production window
is 5 frames at ``hop=512``, ``sr=24 kHz`` (~107 ms of audio) --
the post-A/B sweet spot on the partial-fine-tune corpus (2026-05-22
user pick after comparing windows 3 / 5 / 7 / 9 on the 16-sentence
Ultron test set). 3 frames (~64 ms) leaves audible wobble; 7+
frames (~150 ms+) starts softening fricatives.

Origin: this is the runtime port of ``_spectral_smooth`` in
``ultronVoiceAudio/scripts/bulk_evaluate_finetune.py``. The
bulk-evaluate version was proven on the 1654-clip training corpus
(used to A/B different smoothing intensities); the algorithm here
is bit-for-bit identical, just wrapped for runtime use with the
fail-open semantics the orchestrator expects.

**Cost:** ~10 ms per second of audio on CPU (measured against the
16-sentence Ultron test corpus on 2026-05-22):

  ============  ==========
  Clip length  Latency
  ============  ==========
  1.7 s         ~15-16 ms
  3.5 s         ~16-32 ms
  5-6 s         ~31-46 ms
  10.4 s        ~63 ms
  ============  ==========

The Kokoro engine's round-8c producer-consumer pipeline hides
this cost on every clip after the first -- synth N+1 (including
the smoothing pass) runs while playback N is still draining. The
ack cache pre-renders the 16 cached phrases at cache-build time
so cached acks pay zero smoothing cost at runtime.

**Net runtime impact:**

- Ack cache hit (most turns): 0 ms
- First clip of cache-miss reply (~1-3 s typical): +15-30 ms TTFT
- Clips 2, 3, ... of multi-sentence reply: 0 ms (overlap)
"""
from __future__ import annotations

import numpy as np

__all__ = ["spectral_smooth", "trim_and_fade"]


def spectral_smooth(
    audio: np.ndarray,
    sr: int = 24000,
    *,
    n_fft: int = 2048,
    hop: int = 512,
    median_window_frames: int = 5,
) -> np.ndarray:
    """Smooth ``audio`` by median-filtering STFT magnitudes across time.

    Args:
        audio: 1-D float numpy array, expected normalized to [-1, 1].
            Other dtypes are upcast to float32. Multi-channel input
            is not supported -- pass mono.
        sr: sample rate in Hz. Default 24000 (Kokoro's native rate).
        n_fft: FFT size in samples. 2048 = ~85 ms window at 24 kHz.
        hop: STFT hop size in samples. 512 = 25 % overlap with the
            default ``n_fft``.
        median_window_frames: width of the magnitude median filter
            across time, in STFT frames. Default 5 frames at
            hop=512, sr=24 kHz covers ~107 ms -- the post-A/B sweet
            spot on the partial-fine-tune corpus (2026-05-22 user
            pick after comparing 3 / 5 / 7 / 9 on the 16-sentence
            Ultron test set). 3 (~64 ms) leaves audible wobble; 7+
            (~150 ms+) starts softening fricatives. Pass 1 to
            disable smoothing (no-op) without removing the call
            site. Values < 1 are clamped to 1.

    Returns:
        Float32 numpy array, same shape rank as input but possibly
        slightly different length at the tail due to STFT framing
        (always within ``n_fft`` samples of the input length).

    Raises:
        ImportError: scipy not installed. Callers using this in a
            hot path should wrap in try/except and fail open.

    Notes:
        - Phase is preserved exactly; only magnitudes are smoothed.
          This is what keeps consonants from blurring while killing
          pitch micro-wobble in vowels.
        - For very short clips (< n_fft samples = ~85 ms at 24 kHz)
          the function returns the input unchanged (a single STFT
          frame can't be median-filtered across time).
        - Implementation uses Python loops for STFT/ISTFT rather
          than scipy.signal.stft for byte-for-byte parity with the
          corpus-evaluation version. The cost dominates at clip
          length; the loop overhead is negligible. Vectorization
          via stride_tricks is an obvious optimization if a future
          session needs sub-10 ms smoothing.
    """
    from scipy.ndimage import median_filter

    audio_f32 = np.asarray(audio, dtype=np.float32)
    if median_window_frames < 1:
        median_window_frames = 1

    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(audio_f32) - n_fft) // hop
    if n_frames < 1:
        # Clip too short to STFT meaningfully -- pass through.
        return audio_f32

    frames = np.zeros((n_fft, n_frames), dtype=np.float32)
    for i in range(n_frames):
        frames[:, i] = audio_f32[i * hop: i * hop + n_fft] * window
    spec = np.fft.rfft(frames, axis=0)
    mag = np.abs(spec)
    phase = np.angle(spec)
    mag_smooth = median_filter(mag, size=(1, median_window_frames))
    spec_smooth = mag_smooth * np.exp(1j * phase)
    frames_out = np.fft.irfft(spec_smooth, n=n_fft, axis=0).astype(np.float32)

    out_len = (n_frames - 1) * hop + n_fft
    out = np.zeros(out_len, dtype=np.float32)
    weight = np.zeros(out_len, dtype=np.float32)
    for i in range(n_frames):
        out[i * hop: i * hop + n_fft] += frames_out[:, i] * window
        weight[i * hop: i * hop + n_fft] += window * window
    weight[weight < 1e-8] = 1.0
    return out / weight


def trim_and_fade(
    audio: np.ndarray,
    sr: int = 24000,
    *,
    threshold_db: float = -40.0,
    frame_ms: float = 10.0,
    fade_in_ms: float = 25.0,
    fade_out_ms: float = 45.0,
    pad_ms: float = 5.0,
    hard_silence_pad_ms: float = 8.0,
    tail_aggressive_trim_ms: float = 25.0,
) -> np.ndarray:
    """Trim boundary noise, apply fades, prepend/append hard silence.

    Designed for the partial Kokoro fine-tune: the undertrained model
    (Stage 1 + Stage 2 epoch 0 only; no SLM joint) generates noise
    bursts before speech starts and after speech ends -- ranging from
    short clicks (<5 ms) to medium bursts (up to ~40 ms). This applies
    three layers of defense:

    1. **RMS trim** removes low-level boundary noise (below threshold).
    2. **Cosine fades** attenuate medium-level artifacts within the
       fade-in / fade-out region (raised-cosine curve quieter early
       than a linear ramp).
    3. **Hard silence pad** guarantees the first/last few samples are
       byte-exact zeros, eliminating any DC-step or sub-frame artifact
       that survives the trim+fade.

    Args:
        audio: 1-D float numpy array, expected normalized to [-1, 1].
            Other dtypes are upcast to float32.
        sr: Sample rate in Hz. Default 24000 (Kokoro native).
        threshold_db: RMS frames below this level (dB relative to
            full scale) are treated as silence/noise and may be
            trimmed from the boundaries. Default -40 dB (1% of full
            scale).
        frame_ms: Frame size in ms for RMS energy analysis. Default
            10 ms = 240 samples at 24 kHz.
        fade_in_ms: Duration of the raised-cosine fade-in applied
            after leading-noise trim. Default 25 ms. Long enough to
            attenuate burst artifacts up to ~20 ms in length.
        fade_out_ms: Duration of the raised-cosine fade-out applied
            after trailing-noise trim. Default 30 ms. Slightly longer
            than fade-in because natural speech offset is gentler and
            the partial fine-tune tends to leave slightly longer tail
            artifacts than leading ones.
        pad_ms: Silence buffer to keep around the detected speech
            region before trimming. Default 5 ms. Smaller than before
            since the longer fades absorb consonant onsets natively.
        hard_silence_pad_ms: Pure-silence buffer prepended and
            appended to the trimmed+faded audio. Default 4 ms = 96
            samples at 24 kHz. Guarantees the very first and very
            last samples played are byte-exact zeros so stream
            transitions are clean regardless of any residual artifact
            inside the audio body. Pass 0 to disable padding.

    Returns:
        Trimmed, faded, padded float32 array. If no speech region is
        found (all silence/noise) or the clip is too short for
        analysis, returns the input unchanged as float32.
    """
    audio_f32 = np.asarray(audio, dtype=np.float32)
    frame_samples = max(1, int(sr * frame_ms / 1000))
    n_frames = len(audio_f32) // frame_samples
    if n_frames < 2:
        return audio_f32

    rms_linear = 10.0 ** (threshold_db / 20.0)
    rms = np.array([
        np.sqrt(np.mean(audio_f32[i * frame_samples:(i + 1) * frame_samples] ** 2))
        for i in range(n_frames)
    ])

    speech_frames = np.where(rms > rms_linear)[0]
    if len(speech_frames) == 0:
        return audio_f32

    # 2026-06-11 live fix (user-audible "blip after the sentence"):
    # the partial fine-tune emits an isolated noise burst well AFTER
    # speech ends (watcher-measured live: a ~70 ms burst ~440 ms past
    # the body). A loud burst counts as a "speech frame" above the
    # threshold, so ``speech_frames[-1]`` used to point at the BURST --
    # the trim kept the dead air + blip and faded the blip's tail
    # instead of the speech's. Group loud frames into runs and discard
    # edge runs that are SHORT (<= burst_max) and ISOLATED from the
    # body by a large gap (>= burst_gap): real words are longer than
    # 120 ms, and intra-sentence pauses at a clip edge rarely exceed
    # 200 ms of sub-threshold silence, so speech is never clipped.
    burst_max_frames = max(1, int(np.ceil(120.0 / frame_ms)))
    burst_gap_frames = max(1, int(np.ceil(200.0 / frame_ms)))
    runs: list[tuple[int, int]] = []
    run_start = prev = int(speech_frames[0])
    for f in speech_frames[1:]:
        f = int(f)
        if f == prev + 1:
            prev = f
            continue
        runs.append((run_start, prev))
        run_start = prev = f
    runs.append((run_start, prev))
    while len(runs) > 1:
        s, e = runs[-1]
        gap = s - runs[-2][1] - 1
        if (e - s + 1) <= burst_max_frames and gap >= burst_gap_frames:
            runs.pop()                      # trailing isolated burst
        else:
            break
    while len(runs) > 1:
        s, e = runs[0]
        gap = runs[1][0] - e - 1
        if (e - s + 1) <= burst_max_frames and gap >= burst_gap_frames:
            runs.pop(0)                     # leading isolated burst
        else:
            break

    pad_frames = max(0, int(np.ceil(pad_ms / frame_ms)))
    first_frame = max(0, runs[0][0] - pad_frames)
    last_frame = min(n_frames - 1, runs[-1][1] + pad_frames)

    start = first_frame * frame_samples
    end = min(len(audio_f32), (last_frame + 1) * frame_samples)
    trimmed = audio_f32[start:end].copy()

    if len(trimmed) == 0:
        return audio_f32

    # Raised-cosine fade is quieter in the first ~30% of the ramp
    # than a linear fade -- better at hiding burst artifacts that
    # sit close to the boundary.
    fi = min(int(sr * fade_in_ms / 1000), len(trimmed) // 4)
    if fi > 1:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fi, dtype=np.float32))
        trimmed[:fi] *= ramp

    fo = min(int(sr * fade_out_ms / 1000), len(trimmed) // 4)
    if fo > 1:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(np.pi, 0.0, fo, dtype=np.float32))
        trimmed[-fo:] *= ramp

    # 2026-05-22 -- aggressive last-N-samples mute. The partial fine-
    # tune produces an audible "blip" right at the very end of clips
    # (after the fade-out completes the audio decays but a residual
    # decoder artifact is still above zero). Force the LAST N samples
    # to byte-exact zero so the speaker sees clean silence regardless
    # of any sample-level artifact the cosine fade left in. Cost: at
    # most 25 ms of clipped speech tail, which on natural speech is
    # already in the breath-decay zone.
    if tail_aggressive_trim_ms > 0:
        ta = min(int(sr * tail_aggressive_trim_ms / 1000), len(trimmed) // 4)
        if ta > 0:
            trimmed[-ta:] = 0.0

    if hard_silence_pad_ms > 0:
        pad_samples = max(1, int(sr * hard_silence_pad_ms / 1000))
        silence = np.zeros(pad_samples, dtype=np.float32)
        trimmed = np.concatenate([silence, trimmed, silence])

    return trimmed
