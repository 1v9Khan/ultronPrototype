"""LLM HTTP-mode TTFT benchmark.

Measures first-token latency for the same 10 representative queries
``measure_baseline.py`` uses, but against the HTTP-server runtime.
Compare medians/P95 to the in-process baseline (in baselines.json:
``latency_ms.per_query[*].first_token_ms``) to decide whether the
HTTP path is fast enough to be the default.

Pre-flight: llama-cpp-server must already be running on the URL in
``llm.server.base_url`` (typically http://127.0.0.1:8765/v1).

Run from anywhere:

    python scripts/_bench_llm_http.py

Writes the result block (median, p95, per-query) to
``baselines.json`` under ``llm_http_runtime`` so it's diff-able
against the in-process baseline. Does NOT touch other top-level
keys.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Resolve src/ relative to this script (the worktree has the latest
# LLMEngine; main checkout is older until the branch is pulled).
WORKTREE_ROOT = Path(__file__).resolve().parent.parent
MAIN_REPO = Path(r"C:\STC\ultronPrototype")
sys.path.insert(0, str(WORKTREE_ROOT / "src"))   # newest code
sys.path.insert(0, str(MAIN_REPO))                # config/ shim

from ultron.llm import LLMEngine  # noqa: E402

REPRESENTATIVE_QUERIES = [
    "What is the boiling point of water?",
    "Walk me through how a transistor works.",
    "Who was Nikola Tesla?",
    "What's nineteen times forty-three?",
    "Explain what a hash table is.",
    "Are you afraid of death?",
    "What's a good book to read on a flight?",
    "What do you think about meditation?",
    "And what about the Mariana Trench?",
    "Tell me something interesting about black holes.",
]


def main() -> int:
    print("[bench] constructing LLMEngine in http_server mode...")
    llm = LLMEngine(memory=None, runtime="http_server")

    print("[bench] warming up (cancel after first token)...")
    warm = llm.generate_stream("Say 'ready' and nothing else.")
    for _tok in warm:
        llm.cancel()
        break
    for _ in warm:  # drain
        pass

    print("[bench] running 10 queries...")
    per_query = []
    for i, q in enumerate(REPRESENTATIVE_QUERIES, 1):
        llm.reset_history()
        t0 = time.monotonic()
        first_token_ms = None
        first_text = ""
        char_count = 0
        for tok in llm.generate_stream(q):
            if first_token_ms is None:
                first_token_ms = (time.monotonic() - t0) * 1000.0
                first_text = tok
            char_count += len(tok)
            if char_count >= 80:
                # stop early — we just want TTFT, not full TTFA
                llm.cancel()
                break
        for _ in llm.generate_stream(""):  # drain a no-op call to settle state
            break
        per_query.append({
            "query": q,
            "first_token_ms": round(first_token_ms or -1, 2),
            "first_text": first_text,
        })
        print(f"  [{i:>2}/10] ttft={first_token_ms:.0f}ms  {q[:60]}")

    ttft_values = [
        p["first_token_ms"] for p in per_query
        if p["first_token_ms"] >= 0
    ]
    median = statistics.median(ttft_values)
    p95 = sorted(ttft_values)[int(0.95 * (len(ttft_values) - 1))]
    minv = min(ttft_values)
    maxv = max(ttft_values)

    block = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime": "http_server",
        "base_url": llm._http_base_url,
        "model_alias": llm._http_model_alias,
        "per_query": per_query,
        "ttft_ms": {
            "min": round(minv, 2),
            "median": round(median, 2),
            "p95": round(p95, 2),
            "max": round(maxv, 2),
        },
    }

    out_path = Path(__file__).resolve().parent.parent / "baselines.json"
    if out_path.is_file():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        existing = {}
    existing["llm_http_runtime"] = block
    out_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print(f"[bench] HTTP TTFT median={median:.0f}ms p95={p95:.0f}ms (n={len(ttft_values)})")
    print(f"[bench] wrote llm_http_runtime block to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
