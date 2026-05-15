"""LLM n_batch / n_ubatch tuning sweep.

Loads ``LLMEngine`` repeatedly with different ``(n_batch, n_ubatch)``
combinations and measures TTFT on a fixed voice-length prompt. The
goal is to find the combination that minimises TTFT for the
production prompt size (~1.5 KB system prompt + ~0.2 KB user message)
on the active hardware.

This script REPLACES the running orchestrator's voice stack while it
runs -- per the voice-stack-concurrency rule, the user must confirm
Ultron is NOT running before invoking it.

Usage:

    python scripts/bench_llm_ubatch.py
    python scripts/bench_llm_ubatch.py --sweep "128,256,512,1024"
    python scripts/bench_llm_ubatch.py --warmup 2 --trials 5

Each combination is loaded fresh (full Llama() construction) so the
sweep takes ~30-60 s per combination. The default sweep is
(n_batch, n_ubatch) in {(2048, 128), (2048, 256), (2048, 512),
(2048, 1024), (4096, 256), (4096, 512)} -- six combinations, ~3-6 min
total wall time.

Output: a Markdown summary table on stdout AND a JSON snapshot at
``baselines.json:llm_n_ubatch_sweep`` so successive runs can be
compared.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Make ``ultron`` importable when running from the repo root or a
# worktree. The script intentionally NOT a package member.
HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Import after path setup. CUDA DLL discovery runs in ultron/__init__.
import ultron  # noqa: F401  -- registers CUDA DLL search path on Windows


# Representative voice-length prompts. Match the production
# ``measure_baseline.py`` set so deltas are comparable across passes.
PROMPTS: List[str] = [
    "What's the weather today.",
    "Tell me a haiku about silicon.",
    "Who is the current president.",
    "Set a timer for fifteen minutes.",
    "Explain why the sky is blue in one sentence.",
]


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _measure_ttft_once(engine, prompt: str) -> float:
    """One TTFT measurement against the engine. Returns ms.

    Pulls the first token from ``generate_stream``, then cancels the
    rest so we don't pay decode time we don't need.
    """
    t0 = time.monotonic()
    it = engine.generate_stream(prompt, enable_thinking=False)
    try:
        next(it)
    except StopIteration:
        ttft_ms = (time.monotonic() - t0) * 1000
        return ttft_ms
    ttft_ms = (time.monotonic() - t0) * 1000
    # Drain + cancel the rest so the engine's history stays sane.
    engine.cancel()
    try:
        for _ in it:
            pass
    except Exception:
        pass
    return ttft_ms


def _run_combination(
    n_batch: Optional[int],
    n_ubatch: Optional[int],
    *,
    warmup: int,
    trials: int,
    prompts: List[str],
) -> dict:
    """Load LLMEngine with the given batch sizes, measure TTFT, tear down.

    Returns a dict with median / p95 / per-prompt timings.
    """
    # Tweak config in-process so each combination gets a fresh load.
    from ultron.config import get_config, set_config

    cfg = get_config()
    cfg.llm.n_batch = n_batch
    cfg.llm.n_ubatch = n_ubatch
    set_config(cfg)

    from ultron.llm import LLMEngine

    print(
        f"\n  --- (n_batch={n_batch}, n_ubatch={n_ubatch}) ---",
        flush=True,
    )
    print("  loading engine...", flush=True)
    t_load = time.monotonic()
    engine = LLMEngine()
    load_s = time.monotonic() - t_load
    print(f"  loaded in {load_s:.2f}s", flush=True)

    # Warmup so the first measured call doesn't pay one-shot CUDA
    # kernel JIT etc.
    for i in range(warmup):
        _measure_ttft_once(engine, prompts[i % len(prompts)])
        print(f"  warmup {i + 1}/{warmup} done", flush=True)

    # Trials
    ttfts: List[float] = []
    for trial in range(trials):
        prompt = prompts[trial % len(prompts)]
        engine.reset_history()
        ttft_ms = _measure_ttft_once(engine, prompt)
        ttfts.append(ttft_ms)
        print(f"  trial {trial + 1}/{trials}: {ttft_ms:.0f} ms [{prompt[:40]!r}]",
              flush=True)

    # Tear down engine.
    engine._llm = None
    del engine
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "n_batch": n_batch,
        "n_ubatch": n_ubatch,
        "load_s": round(load_s, 2),
        "ttft_ms": {
            "median": round(statistics.median(ttfts), 1),
            "mean": round(statistics.mean(ttfts), 1),
            "p95": round(_percentile(ttfts, 0.95), 1),
            "min": round(min(ttfts), 1),
            "max": round(max(ttfts), 1),
            "samples": [round(t, 1) for t in ttfts],
        },
    }


def _default_combinations() -> List[Tuple[Optional[int], Optional[int]]]:
    """Six combinations covering the typical n_ubatch sweep space."""
    return [
        (None, None),       # baseline: llama.cpp defaults
        (2048, 128),
        (2048, 256),
        (2048, 512),
        (2048, 1024),
        (4096, 512),
    ]


def _parse_sweep(raw: str) -> List[Tuple[Optional[int], Optional[int]]]:
    """``--sweep "128,256,512"`` => six combinations of (n_batch, n_ubatch).

    For now sweeps ubatch only with n_batch=2048; richer parsing
    later if needed.
    """
    sizes = [int(x.strip()) for x in raw.split(",") if x.strip()]
    return [(2048, s) for s in sizes]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep", type=str, default="",
        help="Comma-separated n_ubatch sizes to test (e.g. '128,256,512'). "
             "Empty = use default 6-combination matrix.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--baselines-path", type=str, default=str(ROOT / "baselines.json"),
        help="Where to merge the result (under llm_n_ubatch_sweep). "
             "Pass /dev/null or empty to skip the merge.",
    )
    args = parser.parse_args(argv)

    combinations = (
        _parse_sweep(args.sweep) if args.sweep else _default_combinations()
    )

    print(f"  sweep: {combinations}", flush=True)
    print(f"  warmup: {args.warmup}  trials: {args.trials}", flush=True)

    results: List[dict] = []
    for n_batch, n_ubatch in combinations:
        try:
            r = _run_combination(
                n_batch, n_ubatch,
                warmup=args.warmup, trials=args.trials, prompts=PROMPTS,
            )
            results.append(r)
        except Exception as e:                                       # noqa: BLE001
            print(f"  combination ({n_batch}, {n_ubatch}) failed: {e}",
                  flush=True)
            results.append({
                "n_batch": n_batch,
                "n_ubatch": n_ubatch,
                "error": str(e),
            })

    # Markdown table on stdout
    print("\n## TTFT sweep results\n")
    print("| n_batch | n_ubatch | median (ms) | mean (ms) | p95 (ms) | min (ms) | max (ms) |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        ttft = r.get("ttft_ms")
        if ttft is None:
            print(f"| {r['n_batch']} | {r['n_ubatch']} | ERROR: {r.get('error')} |")
            continue
        print(
            f"| {r['n_batch']} | {r['n_ubatch']} | "
            f"{ttft['median']} | {ttft['mean']} | {ttft['p95']} | "
            f"{ttft['min']} | {ttft['max']} |"
        )

    # Pick the winner: lowest median.
    valid = [r for r in results if r.get("ttft_ms") is not None]
    if valid:
        best = min(valid, key=lambda r: r["ttft_ms"]["median"])
        print(
            f"\n  winner: n_batch={best['n_batch']} n_ubatch={best['n_ubatch']} "
            f"-> median {best['ttft_ms']['median']} ms",
            flush=True,
        )

    # Merge into baselines.json
    bp = args.baselines_path
    if bp and bp not in ("/dev/null", "NUL"):
        try:
            baselines_path = Path(bp)
            baselines = {}
            if baselines_path.is_file():
                baselines = json.loads(baselines_path.read_text(encoding="utf-8"))
            baselines["llm_n_ubatch_sweep"] = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "warmup": args.warmup,
                "trials": args.trials,
                "results": results,
            }
            baselines_path.write_text(
                json.dumps(baselines, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"  results merged into {baselines_path}", flush=True)
        except Exception as e:
            print(f"  failed to merge into baselines.json: {e}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
