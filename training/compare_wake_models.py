"""Compare wake-word ONNX candidates on a recall/false-accept threshold sweep.

Scores each model against its own positive_test / negative_test clip dirs
through the RUNTIME inference path (the same 1280-sample frame chunking +
trailing-silence padding that ``validate_wake_model.py`` and the live
``WakeWordDetector`` use), then prints recall and false-accept rate across a
threshold sweep. This is the apples-to-apples way to compare "ultron model on
ultron clips" vs "kenning candidate on kenning clips" and to find the
threshold where a candidate matches ultron's recall, and what FAR that costs.

Usage (from repo root, runtime venv):
    .venv\\Scripts\\python.exe training\\compare_wake_models.py --limit 600

Add --full to score every clip (slower, definitive).
"""

from __future__ import annotations

import argparse
import glob
import os
import wave

import numpy as np

FRAME = 1280  # 80 ms @ 16 kHz -- matches WakeWordDetector chunking
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

ROOT = os.path.dirname(os.path.abspath(__file__))
MCM = os.path.join(ROOT, "my_custom_model")

# (label, onnx path, positive_test dir, negative_test dir)
CANDIDATES = [
    ("ultron",    os.path.join("models", "openwakeword", "ultron.onnx"),
        os.path.join(MCM, "ultron", "positive_test"),
        os.path.join(MCM, "ultron", "negative_test")),
]
for _p in sorted(glob.glob(os.path.join(MCM, "kenning_v*.onnx"))):
    _label = os.path.splitext(os.path.basename(_p))[0]
    CANDIDATES.append((
        _label, _p,
        os.path.join(MCM, "kenning", "positive_test"),
        os.path.join(MCM, "kenning", "negative_test"),
    ))
# Also score a bare my_custom_model/kenning.onnx (the just-trained model
# before it's renamed) under the label "kenning_latest".
_latest = os.path.join(MCM, "kenning.onnx")
if os.path.isfile(_latest):
    CANDIDATES.append((
        "kenning_latest", _latest,
        os.path.join(MCM, "kenning", "positive_test"),
        os.path.join(MCM, "kenning", "negative_test"),
    ))


def _read_wav_16k_mono(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got width={width}")
    pcm = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
    if rate != 16000:
        n_out = int(len(pcm) * 16000 / rate)
        pcm = np.interp(
            np.linspace(0, len(pcm) - 1, n_out),
            np.arange(len(pcm)),
            pcm.astype(np.float32),
        ).astype(np.int16)
    return pcm


def _score_clip(model, pcm: np.ndarray) -> float:
    pcm = np.concatenate([pcm, np.zeros(FRAME * 12, dtype=np.int16)])
    model.reset()
    best = 0.0
    for i in range(0, len(pcm) - FRAME + 1, FRAME):
        scores = model.predict(pcm[i : i + FRAME])
        frame_best = max(scores.values())
        if frame_best > best:
            best = float(frame_best)
    return best


def _score_dir(model, path: str, limit: int | None) -> np.ndarray:
    wavs = sorted(glob.glob(os.path.join(path, "*.wav")))
    if limit:
        wavs = wavs[:limit]
    if not wavs:
        raise SystemExit(f"no WAVs under {path}")
    return np.array([_score_clip(model, _read_wav_16k_mono(w)) for w in wavs],
                    dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=600,
                    help="clips per dir (default 600; use --full for all)")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these labels (e.g. ultron kenning_v2)")
    args = ap.parse_args()
    limit = None if args.full else args.limit

    from openwakeword.model import Model

    os.chdir(os.path.dirname(ROOT))  # repo root so relative paths resolve

    rows = []
    for label, model_path, pos_dir, neg_dir in CANDIDATES:
        if args.only and label not in args.only:
            continue
        if not os.path.isfile(model_path):
            print(f"skip {label}: model missing {model_path}")
            continue
        print(f"\n=== {label}  ({os.path.basename(model_path)}) ===", flush=True)
        model = Model(wakeword_models=[model_path], inference_framework="onnx")
        pos = _score_dir(model, pos_dir, limit)
        neg = _score_dir(model, neg_dir, limit)
        print(f"  positives n={len(pos)} mean={pos.mean():.3f} "
              f"p50={np.percentile(pos,50):.3f} p90={np.percentile(pos,90):.3f}")
        print(f"  negatives n={len(neg)} mean={neg.mean():.3f} "
              f"p90={np.percentile(neg,90):.3f} p99={np.percentile(neg,99):.3f}")
        print(f"  {'thr':>5} {'recall':>8} {'FAR':>8}")
        for t in THRESHOLDS:
            recall = float((pos >= t).mean())
            far = float((neg >= t).mean())
            print(f"  {t:>5.2f} {recall:>7.1%} {far:>7.1%}")
        rows.append((label, pos, neg))

    # Side-by-side: recall at the threshold where FAR <= 2% (a practical
    # operating point) for each model.
    print("\n=== operating point: highest recall with FAR <= 2% ===")
    for label, pos, neg in rows:
        best = None
        for t in np.arange(0.30, 0.96, 0.01):
            far = float((neg >= t).mean())
            if far <= 0.02:
                best = (float((pos >= t).mean()), t, far)
                break
        if best:
            print(f"  {label:>12}: recall={best[0]:.1%} @ thr={best[1]:.2f} "
                  f"(FAR={best[2]:.1%})")
        else:
            print(f"  {label:>12}: never reaches FAR<=2% in [0.30,0.95]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
