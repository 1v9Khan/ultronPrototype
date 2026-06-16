#!/usr/bin/env python
"""Integrate generated Ultron tails into ``src/kenning/audio/_agent_flavor.py``.

Reads the tail-generation board output (sets of {agent, pools:[{situation, subtag,
tails}]}), LINTS (length, gender, quotes), DEDUPS (exact within-cell + cross-agent),
MERGES with the existing audited tails (kept as the tagless base), and EMITS the
module as ``TailEntry`` literals keyed dict[agent][situation] = list[TailEntry].

loc:* and dmg:* subtags are KEPT (they match the runtime tag filter); ability:*
and "" become tagless (ability-specificity is recovered by the semantic selector
via the query). Usage:
  .venv/Scripts/python.exe scripts/flavor_gen/integrate_tails.py <board_output.json>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from kenning.audio._tail_schema import AGENT_GENDER  # noqa: E402

OUT = _ROOT / "src" / "kenning" / "audio" / "_agent_flavor.py"

_WRONG_PRONOUN = {
    "she": {"he", "him", "his", "himself"},
    "he": {"she", "her", "hers", "herself"},
    "it": {"he", "him", "his", "himself", "she", "her", "hers", "herself"},
    "they": {"he", "him", "his", "himself", "she", "her", "hers", "herself"},
}
_BASE_SITUATIONS = ("spotted", "damaged", "ult", "utility")
_MAX_WORDS = 8          # terse 2-short-sentence tails; drop the genuinely long


def _canon_situation(s: str) -> str:
    s = (s or "").strip().lower()
    for base in _BASE_SITUATIONS:
        if s == base or s.startswith(base + "_") or s.startswith(base):
            return base
    return s          # the new situations (planting/defusing/...) stand as-is


def _keep_tag(subtag: str) -> str:
    t = (subtag or "").strip()
    return t if (t.startswith("loc:") or t.startswith("dmg:")) else ""


def _clean(text: str) -> str:
    t = (text or "").strip().strip('"').strip("'").strip()
    t = re.sub(r"\s+", " ", t)
    if t and t[-1] not in ".!?":
        t += "."
    return t


def _wordcount(t: str) -> int:
    return len([w for w in re.findall(r"[A-Za-z0-9'/-]+", t)])


def _bad_gender(text: str, gender: str) -> bool:
    bad = _WRONG_PRONOUN.get(gender, set())
    words = {w.lower() for w in re.findall(r"[A-Za-z']+", text)}
    return bool(words & bad)


def _load_sets(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    res = raw.get("result", raw) if isinstance(raw, dict) else raw
    if isinstance(res, str):
        res = json.loads(res)
    return res["sets"] if isinstance(res, dict) and "sets" in res else res


def main() -> int:
    src = Path(sys.argv[1])
    sets = _load_sets(src)
    from kenning.audio._agent_flavor import AGENT_FLAVOR as EXISTING

    # agent -> situation -> dict[text_lower] = (text, tag)
    merged: dict[str, dict[str, dict]] = {}
    stats = {"in": 0, "kept": 0, "drop_len": 0, "drop_gender": 0, "dup": 0}

    # 1) seed with the EXISTING audited tails as the tagless base (coerce tuples).
    for ag, sits in EXISTING.items():
        merged.setdefault(ag, {})
        for sit, tails in sits.items():
            cell = merged[ag].setdefault(sit, {})
            for x in tails:
                txt = x.text if hasattr(x, "text") else str(x)
                cell.setdefault(txt.lower(), (txt, ""))

    # 2) fold in the generated tails (linted + deduped).
    seen_global: dict[str, str] = {}        # text_lower -> owner agent (cross-agent dup guard)
    for s in sets:
        ag = s.get("agent")
        gender = AGENT_GENDER.get(ag, "")
        if not ag or ag not in AGENT_GENDER:
            continue
        for pool in s.get("pools", []):
            sit = _canon_situation(pool.get("situation", ""))
            tag = _keep_tag(pool.get("subtag", ""))
            for raw in pool.get("tails", []):
                stats["in"] += 1
                txt = _clean(raw)
                if not txt:
                    continue
                if _wordcount(txt) > _MAX_WORDS:
                    stats["drop_len"] += 1
                    continue
                if _bad_gender(txt, gender):
                    stats["drop_gender"] += 1
                    continue
                low = txt.lower()
                # cross-agent dedup: a generic line owned by another agent is dropped
                # so the same tail never proliferates across characters.
                owner = seen_global.get(low)
                if owner and owner != ag:
                    stats["dup"] += 1
                    continue
                cell = merged[ag].setdefault(sit, {})
                if low in cell:
                    # keep the more specific (tagged) variant
                    if tag and not cell[low][1]:
                        cell[low] = (txt, tag)
                    else:
                        stats["dup"] += 1
                    continue
                cell[low] = (txt, tag)
                seen_global[low] = ag
                stats["kept"] += 1

    # 3) emit
    lines = ['"""Per-agent CONTEXTUAL movie-Ultron flavor (Age of Ultron).',
             "",
             "GENERATED + MERGED (board 2026-06-16): existing audited tails kept as the",
             "tagless base; generated agent x situation x sub-context tails added with",
             "loc:/dmg: tags. Each entry is a TailEntry(text, tags). Selection: coarse",
             "route (agent+situation) -> 4-tier tag filter -> semantic fine-select.",
             "Regenerate via scripts/flavor_gen/.",
             '"""',
             "from __future__ import annotations",
             "",
             "from kenning.audio._tail_schema import TailEntry",
             "",
             "AGENT_FLAVOR: dict[str, dict[str, list[TailEntry]]] = {"]
    sit_order = list(_BASE_SITUATIONS) + [
        "moving", "planting", "defusing", "rotating", "saving", "falling_back",
        "peeking", "holding", "lurking", "trading", "last_alive", "near_death"]
    for ag in sorted(merged):
        g = AGENT_GENDER.get(ag, "")
        lines.append(f"    {ag!r}: {{   # {g}")
        cells = merged[ag]
        ordered = [s for s in sit_order if s in cells] + \
                  [s for s in sorted(cells) if s not in sit_order]
        for sit in ordered:
            entriesd = cells[sit]
            if not entriesd:
                continue
            lines.append(f"        {sit!r}: [")
            for txt, tag in sorted(entriesd.values()):
                tg = "frozenset()" if not tag else f"frozenset({{{tag!r}}})"
                lines.append(f"            TailEntry({txt!r}, {tg}),")
            lines.append("        ],")
        lines.append("    },")
    lines.append("}")
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")

    n_tails = sum(len(c) for a in merged.values() for c in a.values())
    print(f"stats: {stats}")
    print(f"wrote {OUT} | {len(merged)} agents | {n_tails} total tail entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
