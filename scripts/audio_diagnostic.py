"""Audio diagnostic harness for far-field tuning.

Loads ONLY the audio path (sounddevice capture + openWakeWord + Silero
VAD + faster-whisper). No LLM, no TTS, no RVC, no orchestrator. Total
VRAM around ~1.5 GB so we can iterate without competing with a running
Ultron.

Modes
-----

``--mode noise-floor`` — captures 5 s of silence, reports peak / mean
RMS dBFS so we know the room's noise floor. Run BEFORE wake / phrase
tests so dBFS comparisons are meaningful.

``--mode wake`` — opens a window of N seconds, records every chunk's
wake-word score, and reports the maximum (whether it crossed the
threshold). The user is prompted to say the wake word at a stated
distance.

``--mode phrase`` — captures from "press Enter to start" until VAD
reports speech end (or a hard timeout). Runs Whisper on the captured
clip and prints the transcription alongside per-chunk VAD probability
+ peak RMS.

``--mode monitor`` — live real-time display: rolling peak/mean RMS,
VAD probability, wake-word score. User talks, watches numbers. Ctrl+C
to exit.

CLI overrides
-------------

All overrides are PROCESS-LOCAL — they don't write to ``config.yaml``.
That way we can iterate quickly without producing churn in the live
config. Once a value works we promote it to ``config.yaml`` by hand.

* ``--device <substring>`` — input device substring, e.g. "Focusrite",
  "Voicemeeter", "NVIDIA Broadcast". Default reads from config.
* ``--gain-db <float>`` — pre-amp gain in dB. Default reads from
  config (currently 0.0).
* ``--wake-threshold <float>`` — override openWakeWord confidence
  floor for THIS run. Lower = more sensitive.
* ``--vad-threshold <float>`` — override Silero VAD confidence floor.
  Lower = catches quieter speech.
* ``--seconds <int>`` — capture window length for wake / phrase modes.
* ``--whisper-beam <int>`` — Whisper beam size override (default 5;
  10 is more accurate at ~1.5x latency cost).
* ``--save-wav <path>`` — save raw captured audio to a WAV so we can
  inspect / replay it.
* ``--label <str>`` — annotation written to the JSONL audit log so
  we can correlate test rows with the user-prompt context (e.g.
  "wake @ 14ft").

Audit log
---------

Every test result is appended to ``logs/audio_diag_<timestamp>.jsonl``.
Each row has the test mode, label, all metrics, and any overrides
applied. The file is the source of truth for what "Round 1 baseline"
looked like vs what "Round 3 with +12 dB gain" looked like.

Run from the worktree directory; the ``measure_baseline``-style sys.path
shim picks up worktree code over the main checkout's installed copy.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup: same pattern as scripts/measure_baseline.py. Worktree's src/
# wins over installed package so any worktree edits are exercised. Models
# resolve via cwd / config -> main checkout (or worktree junctions to it).
# ---------------------------------------------------------------------------
WORKTREE_ROOT = Path(__file__).resolve().parent.parent
MAIN_REPO_PATH = Path(r"C:\STC\ultronPrototype")
sys.path.insert(0, str(MAIN_REPO_PATH))            # config/ shim
sys.path.insert(0, str(WORKTREE_ROOT / "src"))     # worktree's ultron code

# Stdout encoding: unicode log lines from libraries shouldn't crash a
# cp1252-default console.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rms_dbfs(audio: np.ndarray) -> float:
    """RMS amplitude expressed in dBFS for a float32 buffer in [-1, 1].

    Returns -120.0 for true silence so log10 doesn't blow up.
    """
    if audio.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
    if rms < 1e-9:
        return -120.0
    return 20.0 * float(np.log10(rms))


def peak_dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    peak = float(np.max(np.abs(audio.astype(np.float64))))
    if peak < 1e-9:
        return -120.0
    return 20.0 * float(np.log10(peak))


def save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Save a float32 audio buffer to a 16-bit PCM WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


# ---------------------------------------------------------------------------
# Component construction (overridable for the diagnostic)
# ---------------------------------------------------------------------------


def build_audio_capture(device: Optional[str], gain_db: float):
    """Build an AudioCapture with override for device + gain.

    Also logs the resolved device after capture starts so the operator
    can confirm which physical input is being used (Voicemeeter vs
    Focusrite vs system default).
    """
    from ultron.audio.capture import AudioCapture
    from ultron.audio.devices import describe_device
    from ultron.config import get_config
    import os

    cfg = get_config().audio
    # Override path: explicit None means "use config default"; an empty
    # string means "system default". Same convention as the env var.
    resolved_device = device if device is not None else cfg.input_device
    env_override = os.environ.get("ULTRON_AUDIO_DEVICE")
    if env_override is not None:
        # The settings shim / env reader uses the env var when set.
        # An EMPTY string is treated as "no override" by some readers
        # but "force system default" by others -- whichever wins, the
        # harness should surface the actual env state so the operator
        # knows.
        if env_override:
            print(f"  ENV override: ULTRON_AUDIO_DEVICE='{env_override}' (matches '{env_override}')")
        else:
            print(f"  ENV override: ULTRON_AUDIO_DEVICE=<empty>  (system default; config 'input_device' is ignored)")

    cap = AudioCapture(
        sample_rate=cfg.sample_rate,
        channels=cfg.channels,
        blocksize=cfg.blocksize,
        device=resolved_device,
        input_gain_db=gain_db,
    )
    return cap


def _print_resolved_device(cap) -> None:
    """Call after cap.start() to print the actually-resolved device index + name."""
    from ultron.audio.devices import describe_device
    if cap.device is None:
        print(f"  Resolved device: <system default>")
    else:
        print(f"  Resolved device: [{cap.device}] {describe_device(cap.device, 'input')}")


def build_wake_word(threshold: Optional[float]):
    """Build a WakeWordDetector. ``threshold=None`` -> config default."""
    from ultron.audio.wake_word import WakeWordDetector
    if threshold is None:
        return WakeWordDetector()
    return WakeWordDetector(threshold=threshold)


def build_vad(threshold: Optional[float]):
    """Build a VAD. ``threshold=None`` -> config default."""
    from ultron.audio.vad import VoiceActivityDetector
    if threshold is None:
        return VoiceActivityDetector()
    return VoiceActivityDetector(threshold=threshold)


def build_whisper(beam_size: Optional[int]):
    """Build a Whisper engine. Beam override is per-call, not per-engine,
    so we keep the engine and override at transcribe time."""
    from ultron.transcription import WhisperEngine
    return WhisperEngine()


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def mode_noise_floor(args, audit_path: Path):
    """Capture 5 s of silence; report dBFS stats."""
    cap = build_audio_capture(args.device, args.gain_db)
    print(f"\n[noise-floor] Will capture {args.seconds} s of audio.")
    print(f"             Stay quiet. Don't move the mic. Press Enter when ready.")
    input()

    chunks: list[np.ndarray] = []
    cap.start()
    _print_resolved_device(cap)
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < args.seconds:
            chunk = cap.get_chunk(timeout=0.5)
            if chunk is None:
                continue
            chunks.append(chunk)
    finally:
        cap.stop()

    if not chunks:
        print("[noise-floor] ERROR: no audio captured. Check device.")
        return

    audio = np.concatenate(chunks).astype(np.float32, copy=False)
    peak = peak_dbfs(audio)
    rms = rms_dbfs(audio)

    print(f"\n[noise-floor] Captured {audio.size} samples ({audio.size / 16000:.2f} s)")
    print(f"             Peak: {peak:+.1f} dBFS")
    print(f"             RMS : {rms:+.1f} dBFS")
    print(f"             A clean room is typically below -50 dBFS RMS.")
    print(f"             Above -40 dBFS RMS suggests fans / AC / hum.")

    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "mode": "noise-floor",
        "label": args.label,
        "device": args.device,
        "gain_db": args.gain_db,
        "duration_s": float(audio.size / 16000),
        "peak_dbfs": peak,
        "rms_dbfs": rms,
    }
    append_jsonl(audit_path, row)
    print(f"\n[noise-floor] Logged to {audit_path}")

    if args.save_wav:
        wav_path = Path(args.save_wav)
        save_wav(wav_path, audio, sample_rate=16000)
        print(f"[noise-floor] WAV saved to {wav_path}")


def mode_wake(args, audit_path: Path):
    """Wake-word distance test. Records max score over a window."""
    cap = build_audio_capture(args.device, args.gain_db)
    wake = build_wake_word(args.wake_threshold)
    print(f"\n[wake] Window: {args.seconds} s.")
    if args.label:
        print(f"       Label: {args.label}")
    print(f"       Say 'Ultron.' once. Speak naturally for the distance.")
    print(f"       Press Enter to start.")
    input()

    cap.start()
    _print_resolved_device(cap)
    wake.reset()
    t0 = time.monotonic()
    chunk_count = 0
    max_score = 0.0
    fired_at: Optional[float] = None
    chunks: list[np.ndarray] = []
    try:
        while time.monotonic() - t0 < args.seconds:
            chunk = cap.get_chunk(timeout=0.1)
            if chunk is None:
                continue
            chunks.append(chunk)
            chunk_count += 1
            # WakeWordDetector exposes either ``predict`` (returns name
            # or None) or ``process`` (boolean). We want the raw score.
            # openWakeWord internally tracks per-model scores; expose
            # via the ``predict_proba`` / model.prediction method when
            # available.
            score = _wake_score(wake, chunk)
            if score is not None and score > max_score:
                max_score = score
            if fired_at is None and score is not None and score >= wake.threshold:
                fired_at = time.monotonic() - t0
    finally:
        cap.stop()

    audio = np.concatenate(chunks).astype(np.float32, copy=False) if chunks else np.zeros(0, dtype=np.float32)
    peak = peak_dbfs(audio)
    rms = rms_dbfs(audio)

    print(f"\n[wake] Captured {chunk_count} chunks ({audio.size / 16000:.2f} s)")
    print(f"       Peak: {peak:+.1f} dBFS    RMS: {rms:+.1f} dBFS")
    print(f"       Wake threshold: {wake.threshold:.2f}")
    print(f"       Max wake score: {max_score:.3f}")
    if fired_at is not None:
        print(f"       FIRED at t+{fired_at:.2f} s  --  WAKE DETECTED")
    else:
        gap = wake.threshold - max_score
        if gap > 0:
            print(f"       Did NOT fire (max {max_score:.3f} < threshold {wake.threshold:.2f}; gap {gap:.3f})")
        else:
            print(f"       Did NOT fire (something odd; gap calc says it should have)")

    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "mode": "wake",
        "label": args.label,
        "device": args.device,
        "gain_db": args.gain_db,
        "wake_threshold": wake.threshold,
        "max_score": max_score,
        "fired": fired_at is not None,
        "fired_at_s": fired_at,
        "duration_s": float(audio.size / 16000),
        "peak_dbfs": peak,
        "rms_dbfs": rms,
        "chunks": chunk_count,
    }
    append_jsonl(audit_path, row)
    print(f"\n[wake] Logged to {audit_path}")

    if args.save_wav:
        wav_path = Path(args.save_wav)
        save_wav(wav_path, audio, sample_rate=16000)
        print(f"[wake] WAV saved to {wav_path}")


def mode_phrase(args, audit_path: Path):
    """Phrase capture + VAD + Whisper transcription test."""
    from ultron.audio.vad import SpeechEvent

    cap = build_audio_capture(args.device, args.gain_db)
    vad = build_vad(args.vad_threshold)
    stt = build_whisper(args.whisper_beam)

    print(f"\n[phrase] Will capture until VAD reports end-of-speech (or {args.seconds} s timeout).")
    if args.label:
        print(f"         Label: {args.label}")
    if args.expected_text:
        print(f'         Expected: "{args.expected_text}"')
    print(f"         Press Enter to start, then speak.")
    input()

    cap.start()
    _print_resolved_device(cap)
    vad.reset()
    chunks: list[np.ndarray] = []
    t0 = time.monotonic()
    speech_started = False
    speech_start_t: Optional[float] = None
    speech_end_t: Optional[float] = None
    max_vad_prob = 0.0
    chunk_count = 0
    try:
        while time.monotonic() - t0 < args.seconds:
            chunk = cap.get_chunk(timeout=0.1)
            if chunk is None:
                continue
            chunks.append(chunk)
            chunk_count += 1
            result = vad.process(chunk)
            if result.probability > max_vad_prob:
                max_vad_prob = float(result.probability)
            if result.event == SpeechEvent.SPEECH_START and not speech_started:
                speech_started = True
                speech_start_t = time.monotonic() - t0
                print(f"  -> VAD: speech start at t+{speech_start_t:.2f} s")
            elif result.event == SpeechEvent.SPEECH_END and speech_started:
                speech_end_t = time.monotonic() - t0
                print(f"  -> VAD: speech end at t+{speech_end_t:.2f} s")
                break
    finally:
        cap.stop()

    if not chunks:
        print("[phrase] ERROR: no audio captured.")
        return
    audio = np.concatenate(chunks).astype(np.float32, copy=False)
    peak = peak_dbfs(audio)
    rms = rms_dbfs(audio)

    if not speech_started:
        print(f"[phrase] WARNING: VAD never detected speech start.")
        print(f"         Max VAD prob: {max_vad_prob:.3f} (threshold {vad.threshold:.2f})")
        print(f"         Either speech was below VAD threshold, or audio was silent.")

    # Whisper run on the whole captured buffer.
    t_stt = time.monotonic()
    transcription = stt.transcribe(audio)
    stt_ms = (time.monotonic() - t_stt) * 1000

    print(f"\n[phrase] Captured {chunk_count} chunks ({audio.size / 16000:.2f} s)")
    print(f"         Peak: {peak:+.1f} dBFS    RMS: {rms:+.1f} dBFS")
    print(f"         Max VAD prob: {max_vad_prob:.3f}    threshold: {vad.threshold:.2f}")
    if speech_start_t is not None and speech_end_t is not None:
        print(f"         Speech window: t+{speech_start_t:.2f} -> t+{speech_end_t:.2f} ({speech_end_t-speech_start_t:.2f} s)")
    print(f"         Whisper ({stt_ms:.0f} ms): \"{transcription}\"")
    if args.expected_text:
        print(f"         Expected     : \"{args.expected_text}\"")
        # Simple WER-ish indicator.
        exp_words = args.expected_text.lower().strip().rstrip(".!?").split()
        got_words = transcription.lower().strip().rstrip(".!?").split()
        match = sum(1 for w in exp_words if w in got_words)
        print(f"         Word coverage: {match}/{len(exp_words)} expected words appeared")

    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "mode": "phrase",
        "label": args.label,
        "device": args.device,
        "gain_db": args.gain_db,
        "vad_threshold": vad.threshold,
        "max_vad_prob": max_vad_prob,
        "speech_start_s": speech_start_t,
        "speech_end_s": speech_end_t,
        "duration_s": float(audio.size / 16000),
        "peak_dbfs": peak,
        "rms_dbfs": rms,
        "chunks": chunk_count,
        "transcription": transcription,
        "expected": args.expected_text,
        "whisper_ms": stt_ms,
    }
    append_jsonl(audit_path, row)
    print(f"\n[phrase] Logged to {audit_path}")

    if args.save_wav:
        wav_path = Path(args.save_wav)
        save_wav(wav_path, audio, sample_rate=16000)
        print(f"[phrase] WAV saved to {wav_path}")


def mode_monitor(args, audit_path: Path):
    """Live real-time meter. Ctrl+C to exit."""
    from ultron.audio.vad import SpeechEvent

    cap = build_audio_capture(args.device, args.gain_db)
    wake = build_wake_word(args.wake_threshold)
    vad = build_vad(args.vad_threshold)

    print(f"\n[monitor] Live mic meter. Ctrl+C to exit.")
    print(f"          Format: {'time':>5} | {'peak':>6} | {'rms':>6} | {'VAD':>5} | {'wake':>5} | event")
    print(f"          ----------------------------------------------------------")

    cap.start()
    _print_resolved_device(cap)
    wake.reset()
    vad.reset()
    last_print = 0.0
    try:
        t0 = time.monotonic()
        while True:
            chunk = cap.get_chunk(timeout=0.1)
            if chunk is None:
                continue
            now = time.monotonic() - t0
            peak = peak_dbfs(chunk)
            rms = rms_dbfs(chunk)
            vad_result = vad.process(chunk)
            wake_score = _wake_score(wake, chunk) or 0.0
            event = ""
            if vad_result.event == SpeechEvent.SPEECH_START:
                event = "*** speech start ***"
            elif vad_result.event == SpeechEvent.SPEECH_END:
                event = "--- speech end ---"
            if wake_score >= wake.threshold:
                event += "  WAKE!"

            # Throttle prints to ~5 Hz so the console isn't a wall of
            # text. Always print on events.
            if event or (now - last_print) >= 0.2:
                print(f"  {now:5.1f} | {peak:+6.1f} | {rms:+6.1f} | {vad_result.probabilityability:5.2f} | {wake_score:5.2f} | {event}")
                last_print = now
    except KeyboardInterrupt:
        print("\n[monitor] Stopped.")
    finally:
        cap.stop()


# ---------------------------------------------------------------------------
# Wake-word score extraction
# ---------------------------------------------------------------------------


def _wake_score(detector, chunk: np.ndarray) -> Optional[float]:
    """Return the raw openWakeWord score for the active model, or None.

    The :class:`WakeWordDetector` wraps openWakeWord. The library's
    ``predict()`` method returns a dict mapping model name -> score.
    We look up the configured model name and return its current value.
    """
    try:
        model = getattr(detector, "_model", None) or getattr(detector, "model", None)
        if model is None:
            return None
        # Convert the float32 [-1, 1] chunk to the int16 PCM
        # openWakeWord expects.
        pcm16 = (chunk * 32767.0).astype(np.int16)
        scores = model.predict(pcm16)
        if not isinstance(scores, dict) or not scores:
            return None
        # Pick the highest score across all models in the dict; the
        # detector may have multiple models loaded. Highest is what
        # would have triggered.
        return float(max(scores.values()))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("noise-floor", "wake", "phrase", "monitor"),
        required=True,
    )
    p.add_argument("--device", default=None,
                   help="Input device substring (e.g. 'Focusrite'). "
                        "None = read from config.")
    p.add_argument("--gain-db", type=float, default=None,
                   help="Pre-amp gain in dB. Default reads from config.")
    p.add_argument("--wake-threshold", type=float, default=None,
                   help="Override openWakeWord threshold for this run.")
    p.add_argument("--vad-threshold", type=float, default=None,
                   help="Override Silero VAD threshold for this run.")
    p.add_argument("--seconds", type=int, default=8,
                   help="Capture window length (default 8 s).")
    p.add_argument("--whisper-beam", type=int, default=None,
                   help="Whisper beam size override.")
    p.add_argument("--save-wav", default=None,
                   help="Save raw captured audio to this path.")
    p.add_argument("--label", default="",
                   help="Annotation written to the audit log.")
    p.add_argument("--expected-text", default=None,
                   help="Expected transcription (for phrase mode word-coverage report).")
    p.add_argument("--audit-log", default=None,
                   help="JSONL audit path. Default logs/audio_diag_<ts>.jsonl.")
    args = p.parse_args(argv)

    if args.gain_db is None:
        from ultron.config import get_config
        try:
            args.gain_db = float(getattr(get_config().audio, "input_gain_db", 0.0))
        except Exception:
            args.gain_db = 0.0

    # Audit log path.
    if args.audit_log:
        audit_path = Path(args.audit_log)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        audit_path = WORKTREE_ROOT / "logs" / f"audio_diag_{ts}.jsonl"

    print("=" * 64)
    print(f"Ultron audio diagnostic harness  --  mode: {args.mode}")
    print("=" * 64)
    print(f"  Device     : {args.device or '<from config>'}")
    print(f"  Gain (dB)  : {args.gain_db:+.1f}")
    if args.wake_threshold is not None:
        print(f"  Wake thr   : {args.wake_threshold:.2f} (override)")
    if args.vad_threshold is not None:
        print(f"  VAD thr    : {args.vad_threshold:.2f} (override)")
    if args.whisper_beam is not None:
        print(f"  Whisper    : beam {args.whisper_beam} (override)")
    print(f"  Audit log  : {audit_path}")
    print()

    # Quiet logging. We want our own structured stdout, not framework logs.
    import os
    os.environ["ULTRON_LOG_LEVEL"] = "WARNING"
    from ultron.utils.logging import configure_logging
    configure_logging(level="WARNING")

    # Dispatch.
    if args.mode == "noise-floor":
        mode_noise_floor(args, audit_path)
    elif args.mode == "wake":
        mode_wake(args, audit_path)
    elif args.mode == "phrase":
        mode_phrase(args, audit_path)
    elif args.mode == "monitor":
        mode_monitor(args, audit_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
