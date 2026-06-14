r"""Split a rephrase JSONL into per-agent audit chunks for the line-by-line review.

Keeps only the LLM-routed lines (deterministic snap/compound/curated lines are
correct by construction and verified by the scorecard's fact-retention), since
those are where output quality actually varies. Writes N chunk_NN.txt files.

    python scripts/relay_test/make_audit_chunks.py logs/relay_test/rephrase_iter4.jsonl \
        logs/relay_test/analysis/iter4 16
"""
from __future__ import annotations

import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "scripts", "relay_test"),
                os.path.join(ROOT, "src"), ROOT]

from kenning.audio.relay_speech import match_relay_command  # noqa: E402
from scorecard import classify_route  # noqa: E402


def main() -> int:
    jsonl = sys.argv[1]
    outdir = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    os.makedirs(outdir, exist_ok=True)

    rows = [json.loads(l) for l in open(jsonl, encoding="utf-8")]
    matched = [r for r in rows if r.get("matched") and r.get("line")]
    llm = []
    for r in matched:
        cmd = match_relay_command(r["text"])
        if cmd is None:
            continue
        route, _ = classify_route(cmd)
        if route in ("llm", "partial"):
            r["_route"] = route
            llm.append(r)

    per = math.ceil(len(llm) / n)
    for i in range(n):
        chunk = llm[i * per:(i + 1) * per]
        if not chunk:
            continue
        with open(os.path.join(outdir, f"chunk_{i:02d}.txt"), "w",
                  encoding="utf-8") as f:
            for j, r in enumerate(chunk):
                gi = i * per + j
                f.write(f"#{gi:04d} [{r['category'].replace('pack_', '')}] "
                        f"({r['_route']})\n")
                f.write(f"  IN : {r['text']}\n")
                f.write(f"  OUT: {r['line']}\n\n")
    print(f"matched-with-line={len(matched)}  LLM/partial-routed (audited)={len(llm)}  "
          f"chunks={n} (~{per}/chunk) -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
