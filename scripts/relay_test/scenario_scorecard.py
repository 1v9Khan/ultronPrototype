#!/usr/bin/env python
"""Scenario-router accuracy + latency scorecard (2026-07-26).

Runs the labelled corpus (``tests/data/routing_corpus.py``, 144 cases across
all 29 scenarios) through the 1B classifier and reports:

  * overall accuracy, and accuracy PER SCENARIO (a flat number hides the fact
    that the confusable toggle family is where routing actually breaks);
  * accuracy per DIFFICULTY TAG -- plain / disfluent / embedded / confusable /
    compound / negative -- because the live failures were all in the non-plain
    buckets;
  * a confusion list, so a systematic swap (e.g. tell_chat -> relay_team) is
    visible as a pattern instead of scattered misses;
  * per-call latency, including the first call separately: the ~1.5k-token
    taxonomy prefill is paid ONCE and reused from the prefix cache, so mixing
    it into the mean would badly misreport steady-state cost.

This is the gate for "reliably ensure perfect routing" -- the claim has to be
measured, and this is what measures it.

Run:
  $env:PYTHONPATH="E:\\ultronPrototype\\src;E:\\ultronPrototype"
  .venv\\Scripts\\python.exe scripts\\relay_test\\scenario_scorecard.py
        [--model <gguf>] [--gpu 0] [--jsonl logs/scenario_scorecard.jsonl]
        [--limit N] [--tag confusable]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tests" / "data"))

DEFAULT_MODEL = _ROOT / "models" / "google_gemma-3-1b-it-Q4_K_M.gguf"


def _build_generate(model_path: Path, gpu_index: int, n_ctx: int):
    """A ``(system, user, max_tokens) -> str`` backed by llama-cpp.

    Deliberately constructed HERE rather than inside ``ScenarioRouter`` so the
    router module never imports llama-cpp (BR-P1: it sits on the voice path).
    """
    import kenning                       # noqa: F401 -- registers CUDA dll paths
    from llama_cpp import Llama, LlamaGrammar
    from kenning.audio.scenario_taxonomy import all_labels

    # GRAMMAR-CONSTRAINED DECODING. Without it the 1B does two things that
    # make a 29-way choice impossible to score: it echoes the utterance back
    # instead of emitting a label (28% of the first run), and it collapses
    # onto one attractor label (tell_chat took 72/144). Restricting the output
    # alphabet to exactly the label strings removes the first failure by
    # construction and forces a genuine choice for the second.
    _alts = " | ".join(f'"{lab}"' for lab in all_labels())
    grammar = LlamaGrammar.from_string(f"root ::= {_alts}", verbose=False)

    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=-1,
        main_gpu=gpu_index,
        split_mode=0,                 # LLAMA_SPLIT_MODE_NONE -- pin to one card
        logits_all=False,
        verbose=False,
    )

    def _generate(*, system: str, user: str, max_tokens: int) -> str:
        out = llm.create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            grammar=grammar,
        )
        return (out["choices"][0]["message"]["content"] or "").strip()

    return _generate, llm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from routing_corpus import CASES                            # noqa: E402
    from kenning.audio.scenario_router import ScenarioRouter    # noqa: E402

    cases = list(CASES)
    if args.tag:
        cases = [c for c in cases if args.tag in c[2]]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("no cases selected")
        return 2

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"model not found: {model_path}")
        return 2

    print(f"loading {model_path.name} on GPU {args.gpu} ...", flush=True)
    t0 = time.perf_counter()
    generate, llm = _build_generate(model_path, args.gpu, args.n_ctx)
    print(f"loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    # shadow=False so the verdict reflects what the router WOULD dispatch.
    router = ScenarioRouter(generate, shadow=False)
    print(f"taxonomy prompt: {len(router.system_prompt)} chars "
          f"(~{len(router.system_prompt)//4} tokens, prefix-cached)\n",
          flush=True)

    rows = []
    per_scenario = defaultdict(lambda: [0, 0])      # label -> [correct, total]
    per_tag = defaultdict(lambda: [0, 0])
    confusion: Counter = Counter()
    latencies = []

    for i, (utterance, expected, tags) in enumerate(cases):
        v = router.classify(utterance)
        ok = v.scenario is expected
        latencies.append(v.elapsed_ms)
        per_scenario[expected.value][1] += 1
        if ok:
            per_scenario[expected.value][0] += 1
        else:
            confusion[(expected.value, v.label)] += 1
        for t in tags:
            per_tag[t][1] += 1
            if ok:
                per_tag[t][0] += 1
        rows.append({
            "utterance": utterance, "expected": expected.value,
            "got": v.label, "ok": ok, "ms": round(v.elapsed_ms, 1),
            "tags": list(tags), "raw": v.raw,
        })
        mark = "ok " if ok else "MISS"
        print(f"  [{i + 1:3d}/{len(cases)}] {mark} {expected.value:>22s} -> "
              f"{v.label:<22s} {v.elapsed_ms:6.0f}ms  {utterance[:52]}",
              flush=True)

    total = len(rows)
    correct = sum(1 for r in rows if r["ok"])
    print("\n" + "=" * 72)
    print(f"ACCURACY  {correct}/{total} = {100.0 * correct / total:.1f}%")
    print("=" * 72)

    print("\nper scenario (worst first):")
    for label, (c, n) in sorted(per_scenario.items(),
                                key=lambda kv: kv[1][0] / max(kv[1][1], 1)):
        pct = 100.0 * c / n if n else 0.0
        flag = "  <-- " if c < n else ""
        print(f"  {label:24s} {c:3d}/{n:<3d} {pct:5.1f}%{flag}")

    print("\nper difficulty tag:")
    for tag, (c, n) in sorted(per_tag.items(),
                              key=lambda kv: kv[1][0] / max(kv[1][1], 1)):
        print(f"  {tag:14s} {c:3d}/{n:<3d} {100.0 * c / n:5.1f}%")

    if confusion:
        print("\nconfusions (expected -> got), most common first:")
        for (exp, got), n in confusion.most_common(15):
            print(f"  {n:3d}x  {exp:24s} -> {got}")

    # First call carries the one-off taxonomy prefill; the rest ride the
    # prefix cache. Reporting one mean would misstate steady-state cost.
    print("\nlatency:")
    print(f"  first call (cold prefill) : {latencies[0]:8.0f} ms")
    warm = latencies[1:] or latencies
    warm_sorted = sorted(warm)
    print(f"  warm mean                 : {statistics.fmean(warm):8.1f} ms")
    print(f"  warm median               : "
          f"{warm_sorted[len(warm_sorted) // 2]:8.1f} ms")
    print(f"  warm p90                  : "
          f"{warm_sorted[int(len(warm_sorted) * 0.9)]:8.1f} ms")
    print(f"  warm max                  : {max(warm):8.1f} ms")
    print(f"\nrouter stats: {router.stats()}")

    if args.jsonl:
        out = Path(args.jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"trace -> {out}")

    try:
        llm.close()
    except Exception:                                            # noqa: BLE001
        pass
    return 0 if correct == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
