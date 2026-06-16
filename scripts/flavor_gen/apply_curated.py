#!/usr/bin/env python
"""Apply CURATED (scripts/flavor_gen/curated_overrides.py) onto _agent_flavor.py.

For each agent/situation present in CURATED, REPLACE that cell with the curated
TailEntry list (text + tags). Cells not in CURATED keep their existing content.
Re-emits the module. Idempotent. Run lint after.
  .venv/Scripts/python.exe scripts/flavor_gen/apply_curated.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts" / "flavor_gen"))
from kenning.audio._tail_schema import AGENT_GENDER         # noqa: E402
from curated_overrides import CURATED                       # noqa: E402

OUT = _ROOT / "src" / "kenning" / "audio" / "_agent_flavor.py"
_SIT_ORDER = ["spotted", "damaged", "ult", "utility", "moving", "planting",
              "defusing", "rotating", "saving", "falling_back", "peeking",
              "holding", "lurking", "trading", "last_alive", "near_death"]


def main() -> int:
    from kenning.audio._agent_flavor import AGENT_FLAVOR as AF
    out: dict = {}
    replaced = 0
    for ag, sits in AF.items():
        out[ag] = {}
        cur = CURATED.get(ag, {})
        for sit, ents in sits.items():
            if sit in cur:
                out[ag][sit] = [(t, tuple(tags)) for (t, tags) in cur[sit]]
                replaced += 1
            else:
                out[ag][sit] = [(e.text if hasattr(e, "text") else str(e),
                                 tuple(getattr(e, "tags", ()) or ())) for e in ents]
        # add any NEW situations introduced by CURATED for this agent
        for sit, lst in cur.items():
            if sit not in out[ag]:
                out[ag][sit] = [(t, tuple(tags)) for (t, tags) in lst]
                replaced += 1

    lines = ['"""Per-agent CONTEXTUAL movie-Ultron flavor (Age of Ultron).',
             "",
             "GENERATED + MERGED + AUDITED + HAND-CURATED (2026-06-16). Each entry is a",
             "TailEntry(text, tags). Curated cells come from scripts/flavor_gen/",
             "curated_overrides.py. Selection: coarse route (agent+side+situation) ->",
             "tag filter -> fine-select. Regenerate/curate via scripts/flavor_gen/.",
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
            if not cells[sit]:
                continue
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
    print(f"replaced {replaced} cells from CURATED ({len(CURATED)} agents) | "
          f"total entries: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
