#!/usr/bin/env python
"""Apply the voice/coherence audit board's cuts to ``_agent_flavor.py``.

Reads the audit output (results:[{agent, cuts:[{text,reason}]}]), removes those
exact tail texts, re-emits the module. Preserves a deterministic floor: never
lets a (agent, situation) cell drop below MIN_CELL entries (keeps the flagged
ones if doing so would starve the cell -- the tier filter + LRU need a base).
  .venv/Scripts/python.exe scripts/flavor_gen/apply_cuts.py <audit_output.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
from kenning.audio._tail_schema import AGENT_GENDER  # noqa: E402

OUT = _ROOT / "src" / "kenning" / "audio" / "_agent_flavor.py"
MIN_CELL = 4
_SIT_ORDER = ["spotted", "damaged", "ult", "utility", "moving", "planting",
              "defusing", "rotating", "saving", "falling_back", "peeking",
              "holding", "lurking", "trading", "last_alive", "near_death"]


def _load_cuts(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    res = raw.get("result", raw) if isinstance(raw, dict) else raw
    if isinstance(res, str):
        res = json.loads(res)
    results = res["results"] if isinstance(res, dict) and "results" in res else res
    cuts: dict[str, set] = {}
    for r in results:
        ag = r.get("agent")
        if not ag:
            continue
        cuts.setdefault(ag, set())
        for c in r.get("cuts", []):
            t = (c.get("text") or "").strip().strip('"').strip("'").strip().lower()
            if t:
                cuts[ag].add(t)
    return cuts


def main() -> int:
    cuts = _load_cuts(Path(sys.argv[1]))
    from kenning.audio._agent_flavor import AGENT_FLAVOR as AF

    applied = kept_floor = 0
    out: dict = {}
    for ag, sits in AF.items():
        cset = cuts.get(ag, set())
        out[ag] = {}
        for sit, ents in sits.items():
            survivors, cut_here = [], []
            for e in ents:
                txt = e.text if hasattr(e, "text") else str(e)
                tags = getattr(e, "tags", frozenset())
                (cut_here if txt.strip().lower() in cset else survivors).append((txt, tags))
            # floor: if cutting starved the cell, add back cut TAGLESS base lines
            if len(survivors) < MIN_CELL and cut_here:
                back = [x for x in cut_here if not x[1]] or cut_here
                while len(survivors) < MIN_CELL and back:
                    survivors.append(back.pop(0))
                    kept_floor += 1
            applied += len(ents) - len(survivors)
            if survivors:
                out[ag][sit] = survivors

    # emit (same format as integrate_tails)
    lines = ['"""Per-agent CONTEXTUAL movie-Ultron flavor (Age of Ultron).',
             "",
             "GENERATED + MERGED + AUDITED (board 2026-06-16). Each entry is a",
             "TailEntry(text, tags). Selection: coarse route (agent+situation) ->",
             "4-tier tag filter -> semantic fine-select. Regenerate via scripts/flavor_gen/.",
             '"""',
             "from __future__ import annotations",
             "",
             "from kenning.audio._tail_schema import TailEntry",
             "",
             "AGENT_FLAVOR: dict[str, dict[str, list[TailEntry]]] = {"]
    for ag in sorted(out):
        lines.append(f"    {ag!r}: {{   # {AGENT_GENDER.get(ag, '')}")
        cells = out[ag]
        ordered = [s for s in _SIT_ORDER if s in cells] + \
                  [s for s in sorted(cells) if s not in _SIT_ORDER]
        for sit in ordered:
            lines.append(f"        {sit!r}: [")
            for txt, tags in sorted(cells[sit]):
                tg = "frozenset()" if not tags else \
                    "frozenset({%s})" % ", ".join(repr(t) for t in sorted(tags))
                lines.append(f"            TailEntry({txt!r}, {tg}),")
            lines.append("        ],")
        lines.append("    },")
    lines.append("}")
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")

    n = sum(len(c) for a in out.values() for c in a.values())
    print(f"cuts applied: {applied} | kept-for-floor: {kept_floor} | remaining entries: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
