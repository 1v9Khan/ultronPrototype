#!/usr/bin/env python
"""Full-pipeline tracer for the relay corpus -- regenerates the corpus through the
CURRENT system and logs every stage of every case so the whole corpus can be read
case-by-case:

  transcription (raw)  ->  norm1 (normalize_command: STT-correct + relay-lead
  recovery + relay-intent gate)  ->  routing (match_relay_command -> RelayCommand:
  payload / addressee / register flags)  ->  norm2 (the deterministic SNAP, payload
  -> snap string)  ->  final (snap + flavor TAIL, the spoken line).

Deterministic (no LLM): snap callouts + curated pools produce their REAL spoken
line; off-snap tactical / conversational cases that need the 3B are flagged
``llm_path=True`` and show the deterministic fallback (the LLM sample run covers
their real wording). This keeps a 20k trace fast while showing every callout+tail
pairing -- the focus of the expansion.

Usage: RELAY_CORPUS_SEED=0 .venv/Scripts/python.exe scripts/relay_test/trace_corpus.py [out.jsonl]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_HERE))                 # corpus_packs / corpus
sys.path.insert(0, str(_ROOT / "src"))         # kenning
sys.path.insert(0, str(_ROOT))                 # top-level `config` package

from corpus_packs import build_corpus_10k       # noqa: E402
from kenning.audio.command_normalizer import normalize_command  # noqa: E402
from kenning.audio.relay_speech import (        # noqa: E402
    match_relay_command, build_relay_line,
)
try:
    from kenning.audio.relay_speech import _as_snap_callout  # noqa: E402
except Exception:                                            # noqa: BLE001
    _as_snap_callout = None


def _snap_only(payload: str):
    """Best-effort bare snap (norm2 output) WITHOUT the flavor tail."""
    if _as_snap_callout is None or not payload:
        return None
    for kwargs in ({"flavor": False}, {}):
        try:
            return _as_snap_callout(payload, None, **kwargs)
        except TypeError:
            continue
        except Exception:                                    # noqa: BLE001
            return None
    return None


def main() -> int:
    seed = int(os.environ.get("RELAY_CORPUS_SEED", "0") or "0")
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        _ROOT / "logs" / "relay_test" / "trace_full.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    cases = build_corpus_10k(seed)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for c in cases:
            rec = {"text": c.text, "category": c.category,
                   "expect_match": c.expect_match}
            try:
                n1 = normalize_command(c.text)
            except Exception as e:                           # noqa: BLE001
                n1 = f"<NORM_ERR {e}>"
            rec["norm1"] = n1
            cmd = None
            try:
                cmd = match_relay_command(n1)
            except Exception as e:                           # noqa: BLE001
                rec["route_err"] = str(e)
            rec["matched"] = cmd is not None
            if cmd is not None:
                rec["payload"] = getattr(cmd, "payload", None)
                rec["addressee"] = getattr(cmd, "addressee", None)
                rec["compose"] = getattr(cmd, "compose", None)
                rec["verbatim"] = getattr(cmd, "verbatim", None)
                rec["directive"] = getattr(cmd, "directive", None)
                rec["context"] = getattr(cmd, "context", None)
                snap = _snap_only(getattr(cmd, "payload", "") or "")
                rec["snap"] = snap
                try:
                    line = build_relay_line(cmd, llm=None, rephrase=False,
                                            recent_lines=None)
                except Exception as e:                       # noqa: BLE001
                    line = f"<LINE_ERR {e}>"
                rec["final"] = line
                # off-snap: needs the 3B (deterministic fallback shows "Team:"/name:)
                rec["llm_path"] = bool(
                    snap is None and not getattr(cmd, "verbatim", False)
                    and not getattr(cmd, "compose", False))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 2000 == 0:
                print(f"  ...{n}", flush=True)
    pos = sum(1 for c in cases if c.expect_match)
    print(f"traced {n} cases (seed {seed}, expect_match={pos}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
