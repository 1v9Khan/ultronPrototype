#!/usr/bin/env python
"""Full-pipeline TRACE emitter for by-hand corpus reading.

For every corpus case, dump each pipeline stage so the whole 20k can be read
case-by-case for COHERENCE / character / tail<->callout pairing:

    transcription (raw heard text)
      -> normalize_command            (norm-1 in/out)
      -> match_relay_command          (routing in/out: payload/addressee/flags)
      -> build_relay_line             (compose in/out: the spoken line = snap+tail)

Deterministic by default (llm=None) so the FULL corpus traces in seconds and the
snap+tail PAIRING is visible; pass --llm to also run the 3B rephrase path on a
sample. Output is a readable text file, chunked, that Read can page through.

Usage:
  python scripts/relay_test/pipeline_trace.py [--seed N] [--limit N]
         [--filter agents|multi|ult|damage|all] [--compact|--verbose]
         [--cat <category-substr>] [--out logs/relay_test/trace.txt]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parents[1]
sys.path.insert(0, str(_HERE))               # corpus, corpus_packs
sys.path.insert(0, str(ROOT / "src"))        # kenning
sys.path.insert(0, str(ROOT))                # top-level `config` package at repo root

from corpus_packs import build_corpus_10k            # noqa: E402
from kenning.audio.command_normalizer import normalize_command  # noqa: E402
from kenning.audio.relay_speech import (             # noqa: E402
    match_relay_command, build_relay_line, _roster_agents,
)


def _cmd_brief(cmd) -> str:
    if cmd is None:
        return "NONE (no relay)"
    g = lambda k, d=None: getattr(cmd, k, d)  # noqa: E731
    bits = [f"payload={g('payload','')!r}", f"addr={g('addressee','team')!r}"]
    for f in ("compose", "context", "directive", "verbatim"):
        v = g(f)
        if v:
            bits.append(f"{f}={v!r}")
    return "  ".join(bits)


def _want(case, cmd, filt: str) -> bool:
    if filt == "all":
        return True
    payload = (getattr(cmd, "payload", "") or case.text) if cmd else case.text
    ags = _roster_agents(payload)
    if filt == "agents":
        return len(ags) >= 1
    if filt == "multi":
        return len(ags) >= 2
    if filt == "ult":
        return "ult" in payload.lower() or "ultimate" in payload.lower()
    if filt == "damage":
        return any(w in payload.lower() for w in
                   ("one shot", "one-shot", "low", "dead", "damage", "hit", "hp"))
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int,
                    default=int(os.environ.get("RELAY_CORPUS_SEED", "0") or "0"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--filter", default="all",
                    choices=["all", "agents", "multi", "ult", "damage"])
    ap.add_argument("--cat", default=None, help="only categories containing this substring")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--llm", action="store_true", help="also run the 3B rephrase path")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cases = build_corpus_10k(args.seed)
    if args.cat:
        cases = [c for c in cases if args.cat in c.category]
    if args.limit:
        cases = cases[: args.limit]

    llm = None
    if args.llm:
        from harness import _load_llm  # noqa: E402
        llm = _load_llm()

    out_path = Path(args.out) if args.out else (
        _HERE.parents[1] / "logs" / "relay_test" / f"trace_seed{args.seed}_{args.filter}.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, case in enumerate(cases):
            raw = case.text
            norm = normalize_command(raw)
            cmd = match_relay_command(norm)
            if not _want(case, cmd, args.filter):
                continue
            try:
                line = build_relay_line(
                    cmd, llm=llm, rephrase=bool(args.llm)) if cmd else "(conversational / no relay)"
            except Exception as e:  # noqa: BLE001
                line = f"(EXC: {e})"
            n += 1
            exp = "T" if getattr(case, "expect_match", True) else "F"
            if args.verbose:
                f.write(f"\n#{i} [{case.category}] expect_relay={exp}\n")
                f.write(f"  STT/raw : {raw!r}\n")
                f.write(f"  norm-1  : {norm!r}\n")
                f.write(f"  route   : {_cmd_brief(cmd)}\n")
                f.write(f"  SPOKEN  : {line!r}\n")
            else:
                f.write(f"#{i} [{case.category}] e={exp} | RAW {raw!r} | "
                        f"N1 {norm!r} | RT {_cmd_brief(cmd)} | OUT {line!r}\n")
    print(f"wrote {n} traced cases -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
