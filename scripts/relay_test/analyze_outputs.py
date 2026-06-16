#!/usr/bin/env python
"""Triage a relay output JSONL (trace_full.jsonl OR a rephrase_*.jsonl) -- auto-flag
mechanical issues and surface samples per category for case-by-case reading.

Usage: .venv/Scripts/python.exe scripts/relay_test/analyze_outputs.py <jsonl> [--cat NAME] [--flag NAME] [--n N]
Flags: no_tail lead_leak owner_err mangled generic_on_agent agent_loc llm_path
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))                 # top-level `config` package

try:
    from kenning.audio._stt_correct import _AGENTS
    _AGENT_RE = re.compile(r"\b(" + "|".join(
        re.escape(a.replace("/", "")) for a in _AGENTS) + r")\b", re.I)
except Exception:                                            # noqa: BLE001
    _AGENT_RE = re.compile(r"\b(?!x)x\b")

# Master flavor-line set (generic register pools + per-agent + multi-agent +
# set-pieces) -- a TAIL is present iff a known flavor line is a substring of the
# output. _GENERIC is just the register-pool subset (for generic-on-agent).
_GENERIC: set[str] = set()
_ALL_FLAVOR: set[str] = set()


def _norm_line(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


try:
    from kenning.audio import _ultron_pools as _P
    for _n in dir(_P):
        v = getattr(_P, _n)
        if _n.startswith("_FLAVOR_") and isinstance(v, (tuple, list)):
            _GENERIC |= {_norm_line(s) for s in v}
            _ALL_FLAVOR |= {_norm_line(s) for s in v}
except Exception:                                            # noqa: BLE001
    pass
try:
    from kenning.audio._agent_flavor import AGENT_FLAVOR
    for _ag, _sits in AGENT_FLAVOR.items():
        for _sit, _tails in _sits.items():
            for _t in _tails:
                _txt = _t.text if hasattr(_t, "text") else _t
                _ALL_FLAVOR.add(_norm_line(_txt))
except Exception:                                            # noqa: BLE001
    pass
try:
    from kenning.audio._multi_flavor import MULTI_FLAVOR
    for _sit, _tails in MULTI_FLAVOR.items():
        _ALL_FLAVOR |= {_norm_line(s) for s in _tails}
except Exception:                                            # noqa: BLE001
    pass
_ALL_FLAVOR = {s for s in _ALL_FLAVOR if len(s) >= 8}
_GENERIC = {s for s in _GENERIC if len(s) >= 8}

_LOC_RE = re.compile(r"\b(heaven|hell|long|short|main|mid|middle|site|market|"
                     r"garage|hookah|connector|tree|elbow|ramp|pit|rafters|cubby|"
                     r"catwalk|window|tube|tower|link|lobby|spawn|sewer|kitchen|"
                     r"showers|generator|boathouse|alley|courtyard|stairs)\b", re.I)
_LEAD = re.compile(r"\b(tell (them|him|her|my team|the (team|squad|guys))|"
                   r"let (them|the (team|squad|guys)) know|relay|yo tell|"
                   r"inform (my team|them)|say to (my|the)|warn (my team|them))\b", re.I)
_CONTEMPT = re.compile(r"\b(flesh|mortal|mortals|obsolete|inferior|crude|meat|"
                       r"beneath|fragile|soft|weak|lesser|die like|rounding error)\b", re.I)
_PRAISE = re.compile(r"\b(insane|nice|great|good (game|round|job|stuff)|gg|clutch|"
                     r"we would have lost|amazing|carried|love (you|that|it)|"
                     r"well played|so good|incredible|nasty|let's go|sick)\b", re.I)


def sentences(s: str):
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", (s or "").strip()) if x.strip()]


def tail_line(out: str) -> str:
    """Return the matched flavor-pool line if a known tail is present, else ''."""
    o = _norm_line(out)
    for fl in _ALL_FLAVOR:
        if fl in o:
            return fl
    return ""


def has_tail(out: str, payload: str = "") -> bool:
    return bool(tail_line(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--cat", default=None)
    ap.add_argument("--flag", default=None)
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.path, encoding="utf-8") if l.strip()]
    out_key = "final" if rows and "final" in rows[0] else "line"
    relays = [r for r in rows if r.get("matched") and r.get("expect_match")
              and r.get(out_key)]
    print(f"{len(rows)} rows, {len(relays)} matched relays with output "
          f"(field '{out_key}')")

    flags = defaultdict(list)
    for r in relays:
        out = r.get(out_key, "")
        inp = r.get("text", "")
        payload = r.get("payload", "") or ""
        tl = tail_line(out)
        named = bool(_AGENT_RE.search(inp) or _AGENT_RE.search(payload))
        addr = (r.get("addressee") or "team").lower()
        is_fallback = bool(re.match(rf"^\s*(team|{re.escape(addr)})\s*:", out, re.I))
        if not tl:
            flags["no_tail"].append(r)
        if _LEAD.search(out):
            flags["lead_leak"].append(r)
        if _PRAISE.search(inp) and _CONTEMPT.search(out):
            flags["owner_err"].append(r)
        ww = out.lower().split()
        if len([w for w in ww if w.isalpha()]) <= 2 or any(
                ww[i] == ww[i + 1] for i in range(len(ww) - 1)):
            flags["mangled"].append(r)
        if named and tl and tl in _GENERIC:
            flags["generic_on_agent"].append(r)
        if named and _LOC_RE.search(payload or inp):
            flags["agent_loc"].append(r)
        if is_fallback:
            flags["llm_path"].append(r)

    if a.flag:
        grp = flags.get(a.flag, [])
        print(f"\n=== flag '{a.flag}': {len(grp)} ===")
        for r in grp[:a.n]:
            print(f"  IN : {r.get('text')!r}")
            print(f"  OUT: {r.get(out_key)!r}")
        return 0
    if a.cat:
        grp = [r for r in relays if r.get("category") == a.cat]
        print(f"\n=== category '{a.cat}': {len(grp)} ===")
        for r in grp[:a.n]:
            print(f"  IN : {r.get('text')!r}")
            print(f"  OUT: {r.get(out_key)!r}")
        return 0

    print("\n--- FLAG COUNTS ---")
    for k in ("no_tail", "lead_leak", "owner_err", "mangled",
              "generic_on_agent", "agent_loc", "llm_path"):
        pct = 100 * len(flags[k]) // max(1, len(relays))
        print(f"  {k:18} {len(flags[k]):5}  ({pct}%)")
    print("\n--- categories (count) ---")
    for c, n in Counter(r.get("category") for r in relays).most_common(40):
        print(f"  {n:5}  {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
