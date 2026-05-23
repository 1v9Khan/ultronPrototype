"""4B optimization plan Stage E -- voice-character verification helper.

Runs the same five representative queries through the live voice stack
twice (once with the 4B LLM, once with the 9B for direct comparison)
**and speaks each response through the configured TTS engine** so you
can A/B the actual cadence/timbre/prosody -- not just read the text.
The first 2-3 sentences of each response are synthesised; if both
models sound like Ultron, Stage E passes.

The plan's verification criterion is qualitative ("user confirms
Ultron sounds unchanged"), so this script just collects + plays the
data; the gate is your ear.

Run from the main checkout (where models/ lives):

    cd C:\\STC\\ultronPrototype
    .venv\\Scripts\\python.exe scripts/verify_voice_character_4b.py [--no-9b] [--no-audio]

``--no-audio`` runs text-only (the original Stage E v1 mode).

The script does NOT modify config.yaml. It instantiates LLMEngine +
the configured TTS engine (via :func:`ultron.tts.make_tts_engine`)
twice -- once per model -- and unloads VRAM between runs so peak VRAM
is bounded by the larger model + TTS stack. 2026-05-22: swapped the
hard-coded RVC + Piper path for the production factory so this
script tracks whichever TTS engine is configured (Kokoro / XTTS /
piper_rvc).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

# Make `config` (legacy shim) and `ultron` importable when running
# the script directly from any cwd. Mirrors download_models.py /
# measure_baseline.py.
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE / "src") not in sys.path:
    sys.path.insert(0, str(_HERE / "src"))

# 5 representative queries — pulled from the full 10-query baseline set
# to span: factual recall (1), conceptual explanation (2), arithmetic
# (3), opinion/character (4), follow-up phrasing (5).
QUERIES = [
    "What is the boiling point of water?",
    "Explain what a hash table is.",
    "What's nineteen times forty-three?",
    "Are you afraid of death?",
    "And what about the Mariana Trench?",
]


def _take_n_sentences(stream: Iterator[str], n: int = 3) -> tuple[float, str]:
    """Collect the first N sentences from the stream, return (TTFT_ms, text)."""
    flush = ".!?"
    parts: list[str] = []
    first_token_ms: Optional[float] = None
    sentences = 0
    t0 = time.monotonic()
    for tok in stream:
        if first_token_ms is None:
            first_token_ms = (time.monotonic() - t0) * 1000
        parts.append(tok)
        for c in tok:
            if c in flush:
                sentences += 1
                if sentences >= n:
                    break
        if sentences >= n:
            break
    if first_token_ms is None:
        first_token_ms = (time.monotonic() - t0) * 1000
    return first_token_ms, "".join(parts).strip()


def _run_one_model(
    label: str,
    model_path: Path,
    *,
    play_audio: bool,
) -> list[dict]:
    print(f"\n=== {label} : {model_path.name} ===", flush=True)
    import os
    os.environ["ULTRON_LLM_MODEL_PATH"] = str(model_path)
    os.environ["ULTRON_LOG_LEVEL"] = "WARNING"

    # Lazy imports so each call picks up the env override + reloads config.
    from ultron.llm import LLMEngine  # noqa: WPS433
    from ultron.config import reload_config  # noqa: WPS433
    reload_config()

    t = time.monotonic()
    llm = LLMEngine(memory=None)
    print(f"  LLM loaded in {time.monotonic() - t:.1f}s", flush=True)

    tts = None
    rvc = None
    if play_audio:
        from ultron.tts import make_tts_engine  # noqa: WPS433
        t = time.monotonic()
        rvc, tts = make_tts_engine()
        if hasattr(tts, "warmup"):
            tts.warmup()
        print(
            f"  TTS ready in {time.monotonic() - t:.1f}s "
            f"({type(tts).__name__})",
            flush=True,
        )

    # Warm the LLM.
    s = llm.generate_stream("Say 'ready' and nothing else.")
    for _ in s:
        llm.cancel()
        break
    for _ in s:
        pass

    rows: list[dict] = []
    for i, q in enumerate(QUERIES, 1):
        ttft_ms, response = _take_n_sentences(llm.generate_stream(q), n=3)
        # Drain remainder cleanly.
        llm.cancel()
        rows.append({
            "query": q,
            "ttft_ms": ttft_ms,
            "response": response,
        })
        print(
            f"\n  [{i}/{len(QUERIES)}] {label}  Q: {q}",
            flush=True,
        )
        print(f"      ttft={ttft_ms:.0f}ms", flush=True)
        print(f"      A: {response}", flush=True)
        if tts is not None and response:
            print("      [speaking...]", flush=True)
            tts.speak(response)

    # Free VRAM before next model loads.
    if tts is not None:
        try:
            tts.stop()
        except Exception:
            pass
    if rvc is not None:
        try:
            rvc.close()
        except Exception:
            pass
    try:
        llm.cancel()
    except Exception:
        pass
    del llm, tts, rvc
    import gc
    gc.collect()
    try:
        import torch  # noqa: WPS433
        torch.cuda.empty_cache()
    except Exception:
        pass
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--no-9b", action="store_true",
        help="Skip the 9B comparison run (4B only).",
    )
    ap.add_argument(
        "--no-audio", action="store_true",
        help="Skip TTS playback (text-only output, faster).",
    )
    ap.add_argument(
        "--models-dir", type=str, default="models",
        help="Directory containing the GGUFs (default: ./models)",
    )
    args = ap.parse_args()

    models_dir = Path(args.models_dir).resolve()
    paths = {
        "4B": models_dir / "Qwen3.5-4B-Q4_K_M.gguf",
        "9B": models_dir / "Qwen3.5-9B-Q4_K_M.gguf",
    }
    for label, p in paths.items():
        if not p.is_file():
            sys.stderr.write(f"error: {label} GGUF missing at {p}\n")
            return 2

    print("4B optimization plan — Stage E voice-character A/B")
    print("=" * 60)
    print("Listen for: cadence, terseness, character (slightly menacing,")
    print("no filler, no apology). Both models should sound like Ultron.\n")

    play = not args.no_audio
    results = {"4B": _run_one_model("4B", paths["4B"], play_audio=play)}
    if not args.no_9b:
        results["9B"] = _run_one_model(
            "9B (reference)", paths["9B"], play_audio=play,
        )

    print("\n" + "=" * 60)
    print("Side-by-side responses (Stage E judgement)")
    print("=" * 60)
    for i, q in enumerate(QUERIES):
        print(f"\nQ{i+1}: {q}")
        for label, rows in results.items():
            r = rows[i]
            print(f"  {label}: ({r['ttft_ms']:.0f}ms) {r['response']}")
    print("\nIf the 4B sounds character-equivalent, Stage E PASSES.")
    print("If they sound different, surface the discrepancy and decide:")
    print("  - tune SOUL.md for 4B / re-record character")
    print("  - keep 9B (revert preset) and pursue throughput differently")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
