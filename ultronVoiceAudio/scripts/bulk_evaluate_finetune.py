"""Bulk-synthesize the 602-entry corpus through fine-tune + post-process.

End-to-end pipeline for extensive listening evaluation:

    corpus.json entries
      -> XTTS-v2 fine-tune (best_model_102.pth + extended reference)
      -> spectral magnitude median filter (3-frame smoothing pre-process)
      -> v2custom Pedalboard chain (pitch shift -0.8, chorus, 0.16 reverb)
      -> output WAV organized by category

Goal: produce the comprehensive sample set the user wants to audition
before committing to using this as the Kokoro fine-tune training corpus
(deferred round 7c/7d).

Runs inside the XTTS isolated venv::

    C:\\STC\\ultronPrototype\\ultronVoiceAudio\\.venv-xtts\\Scripts\\python.exe ^
        C:\\STC\\ultronPrototype\\ultronVoiceAudio\\scripts\\bulk_evaluate_finetune.py ^
            --checkpoint <path-to-best_model_102.pth>

Output structure::

    ultronVoiceAudio/bulk_eval/<run_name>/
      manifest.json
      short_response/0001.wav ...
      medium_technical/0001.wav ...
      theatrical_ultron/0001.wav ...
      ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from collections import defaultdict
from pathlib import Path
from typing import Optional


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent  # ultronVoiceAudio
os.environ.setdefault("TORCH_HOME", str(PROJECT / ".torch_cache"))
os.environ.setdefault("HF_HOME", str(PROJECT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT / ".hf_cache"))
os.environ.setdefault("COQUI_TOS_AGREED", "1")

REFERENCE_WAV = PROJECT / "kokoro training audio" / "Ultron_vocals_mono_v1.wav"
CORPUS_JSON = PROJECT / "scripts" / "corpus.json"
OUTPUT_ROOT = PROJECT / "bulk_eval"

# Synthesis parameters (round-9 production defaults + user-chosen).
GPT_COND_LEN = 30
GPT_COND_CHUNK_LEN = 6
MAX_REF_LENGTH = 60
DEFAULT_TEMPERATURE = 0.65
DEFAULT_SPEED = 1.0

# Post-process parameters (user-chosen via iteration).
DEFAULT_PITCH_SHIFT = -0.8
DEFAULT_SMOOTH_WINDOW = 3       # spectral magnitude median filter, frames
TAIL_SILENCE_S = 0.5            # padded before filter for reverb decay


def _trim_phantom_tail(audio_f32, sample_rate, *,
                       silence_threshold=0.005,
                       max_event_ms=200.0,
                       min_lead_silence_ms=150.0,
                       trailing_grace_ms=80.0,
                       window_ms=20.0,
                       min_clip_duration_ms=800.0):
    """Detect + trim the XTTS phantom-token tail (port of the
    production helper at src/ultron/tts/xtts_v3.py:trim_phantom_tail).

    Pattern (walking RMS from end backward):
        sustained_speech -> silence(>=min_lead_silence_ms) ->
        short_event(<max_event_ms) -> silence_to_end
    """
    import numpy as np
    if audio_f32.ndim != 1:
        audio_f32 = audio_f32.reshape(-1)
    n = audio_f32.shape[0]
    if n == 0:
        return audio_f32, False
    if sample_rate > 0 and (n / sample_rate) * 1000.0 < min_clip_duration_ms:
        return audio_f32, False
    win = max(1, int(sample_rate * window_ms / 1000.0))
    n_win = n // win
    if n_win < 4:
        return audio_f32, False
    buf = audio_f32[: n_win * win].reshape(n_win, win)
    rms = np.sqrt(np.mean(buf.astype(np.float64) ** 2, axis=1)).astype(np.float32)
    speech_mask = rms >= silence_threshold
    if not speech_mask.any():
        return audio_f32, False
    speech_indices = np.where(speech_mask)[0]
    last_idx = int(speech_indices[-1])
    if last_idx == 0:
        return audio_f32, False
    trailing_start = last_idx
    while trailing_start > 0 and speech_mask[trailing_start - 1]:
        trailing_start -= 1
    trailing_event_ms = (last_idx - trailing_start + 1) * window_ms
    if trailing_event_ms > max_event_ms:
        return audio_f32, False
    prior_indices = np.where(speech_mask[:trailing_start])[0]
    if prior_indices.size == 0:
        return audio_f32, False
    prior_end = int(prior_indices[-1])
    gap_ms = (trailing_start - prior_end - 1) * window_ms
    if gap_ms < min_lead_silence_ms:
        return audio_f32, False
    grace_windows = max(1, int(trailing_grace_ms / window_ms))
    cut_window = prior_end + 1 + grace_windows
    cut_samples = min(cut_window * win, n)
    if cut_samples <= 0 or cut_samples >= n:
        cut_samples = min((prior_end + 1) * win, n)
        if cut_samples <= 0:
            return audio_f32, False
    return audio_f32[:cut_samples], True


def _trim_phantom_lead(audio_f32, sample_rate, *,
                       silence_threshold=0.005,
                       max_event_ms=200.0,
                       min_lag_silence_ms=120.0,
                       leading_grace_ms=40.0,
                       window_ms=20.0,
                       min_clip_duration_ms=800.0):
    """Mirror of trim_phantom_tail: detect + trim a phantom AT THE
    START of the clip (the "breath in" the user reported).

    Pattern (walking RMS from start forward):
        silence_from_start -> short_event(<max_event_ms) ->
        silence(>=min_lag_silence_ms) -> sustained_speech...

    XTTS occasionally emits a fragmentary aspirated syllable BEFORE
    the first real word, with a brief silent gap between it and the
    speech onset. This trims the audio to start ``leading_grace_ms``
    before the sustained speech.
    """
    import numpy as np
    if audio_f32.ndim != 1:
        audio_f32 = audio_f32.reshape(-1)
    n = audio_f32.shape[0]
    if n == 0:
        return audio_f32, False
    if sample_rate > 0 and (n / sample_rate) * 1000.0 < min_clip_duration_ms:
        return audio_f32, False
    win = max(1, int(sample_rate * window_ms / 1000.0))
    n_win = n // win
    if n_win < 4:
        return audio_f32, False
    buf = audio_f32[: n_win * win].reshape(n_win, win)
    rms = np.sqrt(np.mean(buf.astype(np.float64) ** 2, axis=1)).astype(np.float32)
    speech_mask = rms >= silence_threshold
    if not speech_mask.any():
        return audio_f32, False

    speech_indices = np.where(speech_mask)[0]
    first_idx = int(speech_indices[0])

    # Walk forward from first_idx to find the end of this contiguous
    # speech region.
    leading_event_end = first_idx
    while leading_event_end + 1 < n_win and speech_mask[leading_event_end + 1]:
        leading_event_end += 1
    leading_event_ms = (leading_event_end - first_idx + 1) * window_ms

    # If the first speech region is long, this is real speech onset --
    # no phantom to trim.
    if leading_event_ms > max_event_ms:
        return audio_f32, False

    # Look AFTER the short leading event for silence then more speech.
    after_indices = np.where(speech_mask[leading_event_end + 1:])[0]
    if after_indices.size == 0:
        # The short event is the only speech in the clip; not a phantom
        # situation (single word).
        return audio_f32, False

    next_speech_window = leading_event_end + 1 + int(after_indices[0])
    gap_ms = (next_speech_window - leading_event_end - 1) * window_ms
    if gap_ms < min_lag_silence_ms:
        # Gap too small -- this is probably a brief held-breath or
        # micro-pause inside real speech, not a phantom + new word.
        return audio_f32, False

    # Phantom signature matched. Cut everything before the sustained-
    # speech onset minus a small grace cushion (so we don't clip the
    # actual word's attack).
    grace_windows = max(1, int(leading_grace_ms / window_ms))
    cut_window = max(0, next_speech_window - grace_windows)
    cut_samples = cut_window * win
    return audio_f32[cut_samples:], True


def _trim_soft_lead(audio_f32, sample_rate, *,
                    real_speech_threshold=0.030,
                    sustained_windows=3,
                    soft_lead_max_ms=400.0,
                    leading_grace_ms=20.0,
                    window_ms=20.0,
                    min_clip_duration_ms=800.0):
    """Second-stage lead trimmer: catches the breath-in -> word
    ramp pattern that ``_trim_phantom_lead`` misses (no silent gap
    between the soft pre-speech energy and the real word).

    Algorithm: walk forward from start until RMS crosses the
    ``real_speech_threshold`` (~6x silence threshold) AND stays
    above for ``sustained_windows`` consecutive windows -- that's
    the real word onset. Trim everything before, with a small
    grace cushion so we don't clip the word's attack transient.

    Capped at ``soft_lead_max_ms`` (400 ms by default) so we don't
    over-trim a clip whose first word happens to start quietly
    after a longer natural pause.
    """
    import numpy as np
    if audio_f32.ndim != 1:
        audio_f32 = audio_f32.reshape(-1)
    n = audio_f32.shape[0]
    if n == 0:
        return audio_f32, False
    if sample_rate > 0 and (n / sample_rate) * 1000.0 < min_clip_duration_ms:
        return audio_f32, False
    win = max(1, int(sample_rate * window_ms / 1000.0))
    n_win = n // win
    if n_win < sustained_windows + 1:
        return audio_f32, False
    buf = audio_f32[: n_win * win].reshape(n_win, win)
    rms = np.sqrt(np.mean(buf.astype(np.float64) ** 2, axis=1)).astype(np.float32)
    above_real = rms >= real_speech_threshold
    # Skip if clip is very quiet overall (no window crosses threshold).
    if not above_real.any():
        return audio_f32, False
    # Find earliest onset_idx where N consecutive windows are above.
    onset_idx = None
    for i in range(n_win - sustained_windows + 1):
        if bool(above_real[i:i + sustained_windows].all()):
            onset_idx = i
            break
    if onset_idx is None or onset_idx == 0:
        return audio_f32, False
    max_trim_windows = int(soft_lead_max_ms / window_ms)
    if onset_idx > max_trim_windows:
        # Pre-speech is too long to be a phantom -- legit pause /
        # natural start. Leave alone.
        return audio_f32, False
    grace_windows = max(1, int(leading_grace_ms / window_ms))
    cut_window = max(0, onset_idx - grace_windows)
    cut_samples = cut_window * win
    return audio_f32[cut_samples:], True


def _trim_soft_trail(audio_f32, sample_rate, *,
                     real_speech_threshold=0.030,
                     sustained_windows=3,
                     trailing_grace_ms=160.0,
                     window_ms=20.0,
                     min_clip_duration_ms=800.0):
    """Mirror of ``_trim_soft_lead`` applied at the END of the clip.

    Catches the trail-off pattern that ``_trim_phantom_tail`` misses
    -- a continuous soft decay extending past natural word-end into
    artifact territory (no clean silent gap separating it from real
    speech).

    Algorithm: walk BACKWARD to find the latest sustained_windows
    run above real_speech_threshold (the last real speech), then
    preserve audio through that point + trailing_grace_ms and trim
    everything beyond. The generous 160 ms grace cushion preserves
    natural consonant decay (e.g., trailing ``/sh/`` or ``/s/``)
    while still cutting longer artifact trails.
    """
    import numpy as np
    if audio_f32.ndim != 1:
        audio_f32 = audio_f32.reshape(-1)
    n = audio_f32.shape[0]
    if n == 0:
        return audio_f32, False
    if sample_rate > 0 and (n / sample_rate) * 1000.0 < min_clip_duration_ms:
        return audio_f32, False
    win = max(1, int(sample_rate * window_ms / 1000.0))
    n_win = n // win
    if n_win < sustained_windows + 1:
        return audio_f32, False
    buf = audio_f32[: n_win * win].reshape(n_win, win)
    rms = np.sqrt(np.mean(buf.astype(np.float64) ** 2, axis=1)).astype(np.float32)
    above_real = rms >= real_speech_threshold
    if not above_real.any():
        return audio_f32, False

    # Find the LATEST window that starts a sustained_windows-long run
    # above real_speech_threshold (i.e., the last reliable real-speech
    # onset).
    last_sustained_start = None
    for i in range(n_win - sustained_windows, -1, -1):
        if bool(above_real[i:i + sustained_windows].all()):
            last_sustained_start = i
            break
    if last_sustained_start is None:
        return audio_f32, False

    # Walk FORWARD from there to find the last contiguous above-threshold
    # window (the actual end of real speech).
    speech_end = last_sustained_start + sustained_windows - 1
    while speech_end + 1 < n_win and bool(above_real[speech_end + 1]):
        speech_end += 1

    # Anything after speech_end + grace is the soft trail we want to cut.
    grace_windows = max(1, int(trailing_grace_ms / window_ms))
    cut_window = speech_end + 1 + grace_windows
    cut_samples = min(cut_window * win, n)
    if cut_samples >= n:
        return audio_f32, False  # nothing meaningful past speech-end-plus-grace
    return audio_f32[:cut_samples], True


def _spectral_smooth(audio, sr, n_fft=2048, hop=512, median_window_frames=3):
    """STFT -> median-filter magnitude across time -> ISTFT with
    original phase. Smooths frame-to-frame harmonic micro-variations
    (where pitch wobble lives) while preserving consonants.

    median_window_frames=3 = ~3 frames of magnitude smoothing
    (~96 ms window at hop=512, sr=24 kHz). Light enough to keep
    fricatives intact."""
    import numpy as np
    from scipy.ndimage import median_filter

    window = np.hanning(n_fft).astype("float32")
    n_frames = 1 + (len(audio) - n_fft) // hop
    if n_frames < 1:
        return audio
    frames = np.zeros((n_fft, n_frames), dtype="float32")
    for i in range(n_frames):
        frames[:, i] = audio[i * hop : i * hop + n_fft] * window
    spec = np.fft.rfft(frames, axis=0)
    mag = np.abs(spec)
    phase = np.angle(spec)
    mag_smooth = median_filter(mag, size=(1, median_window_frames))
    spec_smooth = mag_smooth * np.exp(1j * phase)
    frames_out = np.fft.irfft(spec_smooth, n=n_fft, axis=0).astype("float32")
    out_len = (n_frames - 1) * hop + n_fft
    out = np.zeros(out_len, dtype="float32")
    weight = np.zeros(out_len, dtype="float32")
    for i in range(n_frames):
        out[i * hop : i * hop + n_fft] += frames_out[:, i] * window
        weight[i * hop : i * hop + n_fft] += window * window
    weight[weight < 1e-8] = 1.0
    return out / weight


def _build_filter_chain(pitch_shift_semitones: float):
    """Construct the v2custom chain. Pulled inline so the script is
    self-contained -- ultron_filter.py preset definitions don't
    expose a chorus + 0.16-wet-reverb variant yet."""
    from pedalboard import (
        Pedalboard, HighpassFilter, PitchShift, Compressor, LowShelfFilter,
        Delay, Chorus, Distortion, PeakFilter, HighShelfFilter, Reverb,
        LowpassFilter,
    )
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80),
        PitchShift(semitones=pitch_shift_semitones),
        Compressor(threshold_db=-20.0, ratio=3.0, attack_ms=4.0, release_ms=70.0),
        LowShelfFilter(cutoff_frequency_hz=140, gain_db=3.5, q=0.7),
        Delay(delay_seconds=0.005, feedback=0.18, mix=0.13),
        Chorus(rate_hz=0.9, depth=0.18, centre_delay_ms=4.5,
               feedback=0.10, mix=0.10),
        Distortion(drive_db=5.0),
        PeakFilter(cutoff_frequency_hz=2400, gain_db=2.5, q=1.4),
        HighShelfFilter(cutoff_frequency_hz=6500, gain_db=-3.0, q=0.7),
        Reverb(room_size=0.16, damping=0.62, wet_level=0.16,
               dry_level=0.84, width=1.0),
        LowpassFilter(cutoff_frequency_hz=9000),
    ])


def _resolve_base_checkpoint_dir():
    from TTS.utils.manage import ModelManager
    manager = ModelManager()
    model_path, _, _ = manager.download_model(
        "tts_models/multilingual/multi-dataset/xtts_v2"
    )
    return Path(model_path)


def _synthesize(model, text, gpt_latent, speaker_emb, temperature, speed):
    """Drain inference_stream into a single float32 numpy array."""
    import numpy as np
    import torch
    chunks = []
    for chunk in model.inference_stream(
        text=text,
        language="en",
        gpt_cond_latent=gpt_latent,
        speaker_embedding=speaker_emb,
        stream_chunk_size=20,
        temperature=temperature,
        speed=speed,
    ):
        if isinstance(chunk, torch.Tensor):
            chunks.append(chunk.detach().cpu().numpy().astype("float32"))
        else:
            chunks.append(chunk.astype("float32"))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")


def _save_wav(path, audio, sr):
    import numpy as np
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype("int16")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bulk-synthesize corpus.json through fine-tune + v2custom post-process."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=CORPUS_JSON)
    parser.add_argument("--reference-wav", type=Path, default=REFERENCE_WAV)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-name", default=time.strftime("bulk_eval_%Y%m%d_%H%M%S"))
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--pitch-shift", type=float, default=DEFAULT_PITCH_SHIFT)
    parser.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW,
                        help="Spectral median filter window in frames; 0 disables.")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Restrict to specific categories from corpus.json.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max sentences to process (debug / quick run).")
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        print(f"ERROR: checkpoint missing: {args.checkpoint}")
        return 1
    if not args.corpus.is_file():
        print(f"ERROR: corpus missing: {args.corpus}")
        return 1
    if not args.reference_wav.is_file():
        print(f"ERROR: reference WAV missing: {args.reference_wav}")
        return 1

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    if args.categories:
        corpus = [e for e in corpus if e.get("category") in args.categories]
    if args.limit:
        corpus = corpus[: args.limit]
    print(f"corpus:      {args.corpus} ({len(corpus)} entries after filter)")

    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"output:      {run_dir}")
    print(f"params:      temp={args.temperature} speed={args.speed} "
          f"pitch_shift={args.pitch_shift} smooth_window={args.smooth_window}")

    by_cat = defaultdict(int)
    for e in corpus:
        by_cat[e.get("category", "unknown")] += 1
    print(f"categories:  {dict(by_cat)}")

    import torch
    import numpy as np
    print(f"\ntorch: {torch.__version__}  cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")

    base_dir = _resolve_base_checkpoint_dir()
    vocab_path = base_dir / "vocab.json"
    speakers_path = base_dir / "speakers_xtts.pth"
    config_path = base_dir / "config.json"

    print("\n=== Loading XTTS-v2 with fine-tune checkpoint ===")
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    t0 = time.monotonic()
    config = XttsConfig()
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=str(args.checkpoint),
        vocab_path=str(vocab_path),
        speaker_file_path=str(speakers_path),
        use_deepspeed=False,
        eval=True,
    )
    if torch.cuda.is_available():
        model.cuda()
    sample_rate = config.audio.output_sample_rate
    print(f"loaded in {time.monotonic() - t0:.1f}s, sample_rate={sample_rate}")

    print("\n=== Computing conditioning latents ===")
    t0 = time.monotonic()
    gpt_latent, speaker_emb = model.get_conditioning_latents(
        audio_path=str(args.reference_wav),
        gpt_cond_len=GPT_COND_LEN,
        gpt_cond_chunk_len=GPT_COND_CHUNK_LEN,
        max_ref_length=MAX_REF_LENGTH,
    )
    print(f"computed in {time.monotonic() - t0:.2f}s")

    print("\n=== Warmup pass ===")
    _ = _synthesize(model, "Hello.", gpt_latent, speaker_emb, args.temperature, args.speed)

    board = _build_filter_chain(args.pitch_shift)

    print(f"\n=== Generating {len(corpus)} sentences ===")
    manifest = []
    t_start = time.monotonic()
    n_failed = 0
    n_lead_trimmed = 0
    n_soft_lead_trimmed = 0
    n_tail_trimmed = 0
    n_soft_trail_trimmed = 0
    for i, entry in enumerate(corpus, 1):
        cat = entry.get("category", "unknown")
        eid = entry["id"]
        text = entry["text"]
        cat_dir = run_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        out_path = cat_dir / f"{eid}.wav"
        try:
            t0 = time.monotonic()
            audio = _synthesize(model, text, gpt_latent, speaker_emb,
                                args.temperature, args.speed)
            synth_s = time.monotonic() - t0

            # Phantom trims: applied FIRST on the raw XTTS output so
            # we don't carry the leading "breath" or trailing blip into
            # the smoothing/filter stages where they'd get re-shaped
            # by reverb and look harder to remove. Two-stage lead trim
            # because XTTS produces two distinct lead-artifact patterns:
            #   1. silence -> short blip -> silence -> word  (gapped phantom)
            #   2. silence -> soft ramp continuously into word  (breath-in)
            # Stage 1 (gapped) requires the silent gap; stage 2 (ramp)
            # uses an absolute "real speech" threshold sustained over
            # several windows.
            audio, lead_trimmed = _trim_phantom_lead(audio, sample_rate)
            audio, soft_lead_trimmed = _trim_soft_lead(audio, sample_rate)
            audio, tail_trimmed = _trim_phantom_tail(audio, sample_rate)
            audio, soft_trail_trimmed = _trim_soft_trail(audio, sample_rate)
            if lead_trimmed:
                n_lead_trimmed += 1
            if soft_lead_trimmed:
                n_soft_lead_trimmed += 1
            if tail_trimmed:
                n_tail_trimmed += 1
            if soft_trail_trimmed:
                n_soft_trail_trimmed += 1

            t0 = time.monotonic()
            if args.smooth_window > 0:
                audio = _spectral_smooth(audio, sample_rate,
                                         median_window_frames=args.smooth_window)
            smooth_s = time.monotonic() - t0

            t0 = time.monotonic()
            tail_n = int(TAIL_SILENCE_S * sample_rate)
            audio_padded = np.concatenate([audio, np.zeros(tail_n, dtype="float32")])
            audio_filtered = board(audio_padded, sample_rate)
            filter_s = time.monotonic() - t0

            _save_wav(out_path, audio_filtered, sample_rate)
            duration_s = len(audio_filtered) / sample_rate
            manifest.append({
                "id": eid,
                "category": cat,
                "text": text,
                "wav": f"{cat}/{eid}.wav",
                "duration_s": duration_s,
                "synth_s": synth_s,
                "smooth_s": smooth_s,
                "filter_s": filter_s,
                "lead_trimmed": lead_trimmed,
                "soft_lead_trimmed": soft_lead_trimmed,
                "tail_trimmed": tail_trimmed,
                "soft_trail_trimmed": soft_trail_trimmed,
            })
            if i % 25 == 0 or i == len(corpus):
                elapsed = time.monotonic() - t_start
                rate = i / elapsed
                eta = (len(corpus) - i) / rate
                print(f"  [{i:>3}/{len(corpus)}] {cat:<20} synth={synth_s:.2f}s "
                      f"smooth={smooth_s*1000:.0f}ms filter={filter_s*1000:.0f}ms "
                      f"(rate {rate:.2f}/s, ETA {eta/60:.1f}min)")
        except Exception as ex:
            n_failed += 1
            print(f"  [{i:>3}/{len(corpus)}] FAILED {eid}: {ex}")
            manifest.append({"id": eid, "category": cat, "text": text,
                             "wav": None, "error": str(ex)})

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total_elapsed = time.monotonic() - t_start
    print("\n" + "=" * 60)
    print(f"Completed {len(corpus)} sentences in {total_elapsed/60:.1f} min "
          f"({n_failed} failed)")
    print(f"Phantom trims: lead={n_lead_trimmed} soft_lead={n_soft_lead_trimmed} "
          f"tail={n_tail_trimmed} soft_trail={n_soft_trail_trimmed}")
    print(f"Output: {run_dir}")
    print(f"Manifest: {run_dir / 'manifest.json'}")
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
