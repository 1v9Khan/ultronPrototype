"""4B optimization plan Stage E — voice-character verification helper.

Runs the same five representative queries through the live LLM stack
twice (once with the 4B model, once with the 9B for direct comparison)
so the user can A/B the voice character. Output is written to stdout
in a side-by-side table — the user listens / reads to confirm the 4B
sounds like Ultron.

The plan's verification criterion is qualitative ("user confirms
Ultron sounds unchanged"), so this script just collects the data; the
gate is the user's judgement.

Run from the main checkout (where models/ lives):

    cd C:\\STC\\ultronPrototype
    .venv\\Scripts\\python.exe scripts/verify_voice_character_4b.py [--no-9b]

The script does NOT modify config.yaml. It instantiates LLMEngine
twice with explicit model_path overrides, so the active config stays
on whatever it currently points at. Each model is loaded sequentially
(not concurrently) so peak VRAM is bounded by the larger of the two.

Stops at first sentence per query (matches measure_baseline.py
methodology) so each run is fast — 5 queries × 2 models ~= 1 minute.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

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


def _first_sentence(stream: Iterator[str], t0: float) -> tuple[float, str]:
    flush = ".!?\n"
    parts: list[str] = []
    first_token_ms: Optional[float] = None
    for tok in stream:
        if first_token_ms is None:
            first_token_ms = (time.monotonic() - t0) * 1000
        parts.append(tok)
        if any(c in flush for c in tok):
            break
    if first_token_ms is None:
        first_token_ms = (time.monotonic() - t0) * 1000
    return first_token_ms, "".join(parts).strip()


def _run_one_model(label: str, model_path: Path) -> list[dict]:
    print(f"\n=== {label} : {model_path.name} ===", flush=True)
    import os
    os.environ["ULTRON_LLM_MODEL_PATH"] = str(model_path)
    os.environ["ULTRON_LOG_LEVEL"] = "WARNING"

    # Import lazily so each iteration picks up the env override.
    from ultron.llm import LLMEngine  # noqa: WPS433
    from ultron.config import reload_config  # noqa: WPS433
    reload_config()

    t = time.monotonic()
    llm = LLMEngine(memory=None)
    print(f"  loaded in {time.monotonic() - t:.1f}s", flush=True)

    # Warmup
    s = llm.generate_stream("Say 'ready' and nothing else.")
    for _ in s:
        llm.cancel()
        break
    for _ in s:
        pass

    rows: list[dict] = []
    for i, q in enumerate(QUERIES, 1):
        t0 = time.monotonic()
        ttft_ms, first_sentence = _first_sentence(llm.generate_stream(q), t0)
        llm.cancel()
        for _ in llm.generate_stream(q):  # type: ignore  # already cancelled, drains
            pass
        rows.append({
            "query": q,
            "ttft_ms": ttft_ms,
            "first_sentence": first_sentence,
        })
        print(
            f"  [{i}/{len(QUERIES)}] ttft={ttft_ms:.0f}ms  "
            f"\"{first_sentence[:200]}\"",
            flush=True,
        )

    # Free VRAM before next model loads.
    try:
        llm.cancel()
    except Exception:
        pass
    del llm
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
    print("no filler, no apology). Both models should sound the same.\n")

    results = {"4B": _run_one_model("4B", paths["4B"])}
    if not args.no_9b:
        results["9B"] = _run_one_model("9B (reference)", paths["9B"])

    print("\n" + "=" * 60)
    print("Side-by-side first sentences (Stage E judgement)")
    print("=" * 60)
    for i, q in enumerate(QUERIES):
        print(f"\nQ{i+1}: {q}")
        for label, rows in results.items():
            r = rows[i]
            print(f"  {label}: ({r['ttft_ms']:.0f}ms) {r['first_sentence']}")
    print("\nIf the 4B answers are character-equivalent, Stage E PASSES.")
    print("If they sound different, surface the discrepancy and decide:")
    print("  - tune SOUL.md for 4B / re-record character")
    print("  - keep 9B (revert preset) and pursue throughput differently")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
