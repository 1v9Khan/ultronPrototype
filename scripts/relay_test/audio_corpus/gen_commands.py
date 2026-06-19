"""Generate per-command composite audio for the live audio-injection corpus.

Each clip simulates the user speaking a battery command into the mic:

  [0.5s silence] + [trained "Ultron" wake sample] + [gap] + [stock-Kokoro command
  body, fast cadence] + [1.3s trailing silence]

WHY the spliced wake: the custom openWakeWord "ultron" model is trained on real
speech + Piper synthetic samples and scores ~0.27 (< 0.65 threshold) on stock
Kokoro "Ultron", so a Kokoro wake word would NEVER fire. We therefore prepend a
trained-distribution wake sample (training/crosscheck_ultron/*.wav, fires at
~0.94) so the REAL wake detector triggers, then the stock-Kokoro COMMAND BODY
(the "Ultron," prefix stripped) tests the pre-roll + audio-domain wake-drop +
no-clipping + STT exactly as live. Trailing silence is mandatory so the VAD
closes the utterance (else capture hangs to max_utterance_seconds).

Output: 16 kHz mono WAV (injection source) + MP3 (deliverable) named by the full
command, plus manifest.json with {command, body, expected wav/mp3, gap_s}.

Usage:
  python scripts/relay_test/audio_corpus/gen_commands.py \
      [battery.txt] [outdir] [--voice am_michael] [--speed 1.18] [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from math import gcd
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

CAPTURE_SR = 16000
LEAD_SILENCE_S = 0.5
TAIL_SILENCE_S = 1.3
GAP_COMMA_S = 0.25      # "ultron, X" -> a natural pause after the wake word
GAP_RUNON_S = 0.06      # "ultron X"  -> near-continuous (the hard wake-drop case)
WAKE_RE = re.compile(r"^\s*ultron\b[\s,]*", re.IGNORECASE)


def _slug(text: str, idx: int) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:70]
    return f"{idx:03d}_{s}"


def _to_16k_f32(x: np.ndarray, sr: int) -> np.ndarray:
    if x.ndim > 1:
        x = x[:, 0]
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    else:
        x = x.astype(np.float32)
    if sr != CAPTURE_SR:
        from scipy.signal import resample_poly
        g = gcd(CAPTURE_SR, sr)
        x = resample_poly(x, CAPTURE_SR // g, sr // g).astype(np.float32)
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("battery", nargs="?",
                    default=str(_ROOT / "scripts/relay_test/battery_cmds.txt"))
    ap.add_argument("outdir", nargs="?", default=str(_HERE / "out"))
    ap.add_argument("--voice", default="am_michael")
    ap.add_argument("--speed", type=float, default=1.18)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cmds = [ln.strip() for ln in Path(args.battery).read_text(encoding="utf-8").splitlines()]
    cmds = [c for c in cmds if c and not c.lstrip().startswith("#")]
    if args.limit:
        cmds = cmds[: args.limit]

    outdir = Path(args.outdir)
    wav_dir, mp3_dir = outdir / "wav", outdir / "mp3"
    wav_dir.mkdir(parents=True, exist_ok=True)
    mp3_dir.mkdir(parents=True, exist_ok=True)

    import soundfile as sf
    from pydub import AudioSegment
    from kenning.tts.kokoro_engine import KokoroSpeech

    # Pre-load the trained wake-word samples (fire at ~0.94), rotated for variety.
    wake_files = sorted(glob.glob(str(_ROOT / "training/crosscheck_ultron/*.wav")))[:40]
    wakes = []
    for wf in wake_files:
        x, sr = sf.read(wf, dtype="float32")
        wakes.append(_to_16k_f32(np.asarray(x), int(sr)))
    if not wakes:
        print("!! no crosscheck_ultron wake samples found")
        return 2

    eng = KokoroSpeech(voice=args.voice, speed=args.speed, device="cpu",
                       apply_runtime_filter=False)

    lead = np.zeros(int(LEAD_SILENCE_S * CAPTURE_SR), dtype=np.float32)
    tail = np.zeros(int(TAIL_SILENCE_S * CAPTURE_SR), dtype=np.float32)

    manifest = []
    for i, cmd in enumerate(cmds):
        body = WAKE_RE.sub("", cmd).strip()           # command without "Ultron,"
        had_comma = bool(re.match(r"^\s*ultron\s*,", cmd, re.IGNORECASE))
        gap_s = GAP_COMMA_S if had_comma else GAP_RUNON_S
        gap = np.zeros(int(gap_s * CAPTURE_SR), dtype=np.float32)

        pcm, sr = eng._synthesize(body)
        if pcm is None or len(pcm) == 0:
            print(f"  !! empty synth #{i}: {body!r}")
            continue
        body_f32 = _to_16k_f32(np.asarray(pcm), int(sr))
        wake = wakes[i % len(wakes)]
        composite = np.concatenate([lead, wake, gap, body_f32, tail]).astype(np.float32)

        slug = _slug(cmd, i)
        wav_path = wav_dir / f"{slug}.wav"
        mp3_path = mp3_dir / f"{slug}.mp3"
        sf.write(str(wav_path), composite, CAPTURE_SR, subtype="PCM_16")
        i16 = np.clip(composite * 32767, -32768, 32767).astype(np.int16)
        AudioSegment(i16.tobytes(), frame_rate=CAPTURE_SR, sample_width=2,
                     channels=1).export(str(mp3_path), format="mp3", bitrate="96k")

        manifest.append({"i": i, "command": cmd, "body": body, "slug": slug,
                         "wav": str(wav_path), "mp3": str(mp3_path),
                         "gap_s": gap_s, "wake_sample": Path(wake_files[i % len(wakes)]).name,
                         "duration_s": round(len(composite) / CAPTURE_SR, 2)})
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(cmds)}")

    (outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"generated {len(manifest)} composite clips (voice={args.voice}, "
          f"speed={args.speed}) -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
