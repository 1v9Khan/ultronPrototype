#!/usr/bin/env python
"""Deterministic lint gate for the Ultron tail library (_agent_flavor.py).

Pure-python, zero-cost, pre-commit-suitable. HARD failures (exit 1): a tail over
the word cap, a wrong-gender pronoun (vs AGENT_GENDER), surrounding quotes, or a
(agent, situation) cell below the deterministic floor. SOFT warnings: starts with
a tactical verb (reads as a second callout), missing terminal punctuation.

  .venv/Scripts/python.exe scripts/flavor_audit/lint_tails.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
from kenning.audio._agent_flavor import AGENT_FLAVOR        # noqa: E402
from kenning.audio._tail_schema import AGENT_GENDER         # noqa: E402

_MAX_WORDS = 10         # existing 2-sentence lines; generated set is <=8
_MIN_BASE = 4
_WRONG = {
    "she": {"he", "him", "his", "himself"},
    "he": {"she", "her", "hers", "herself"},
    "it": {"he", "him", "his", "himself", "she", "her", "hers", "herself"},
    "they": {"he", "him", "his", "himself", "she", "her", "hers", "herself"},
}
_BAD_START = re.compile(r"^(rotate|push|plant|defuse|save|hold|fall back|"
                        r"site|go|peek|swing|cross)\b", re.I)


def main() -> int:
    hard: list[str] = []
    soft: list[str] = []
    thin: list[str] = []
    n = 0
    for ag, sits in AGENT_FLAVOR.items():
        gender = AGENT_GENDER.get(ag, "")
        bad = _WRONG.get(gender, set())
        for sit, ents in sits.items():
            base = 0
            for e in ents:
                n += 1
                txt = e.text if hasattr(e, "text") else str(e)
                tags = getattr(e, "tags", frozenset())
                if not tags:
                    base += 1
                if len(re.findall(r"[A-Za-z0-9'/-]+", txt)) > _MAX_WORDS:
                    hard.append(f"[{ag}/{sit}] >{_MAX_WORDS}w: {txt!r}")
                words = {w.lower() for w in re.findall(r"[A-Za-z']+", txt)}
                if words & bad:
                    hard.append(f"[{ag}/{sit}] wrong-gender ({gender}): {txt!r}")
                if txt[:1] in "\"'" or txt[-1:] in "\"'":
                    hard.append(f"[{ag}/{sit}] quoted: {txt!r}")
                if txt and txt[-1] not in ".!?":
                    soft.append(f"[{ag}/{sit}] no end punct: {txt!r}")
                if _BAD_START.match(txt):
                    soft.append(f"[{ag}/{sit}] tactical-start: {txt!r}")
            if sit == "spotted" and base < _MIN_BASE:
                thin.append(f"[{ag}/spotted] only {base} base tails (<{_MIN_BASE})")

    print(f"linted {n} tails across {len(AGENT_FLAVOR)} agents")
    for t in thin:
        print("  FLOOR:", t)
    for s in soft[:30]:
        print("  WARN :", s)
    if len(soft) > 30:
        print(f"  ... +{len(soft) - 30} more warnings")
    for h in hard:
        print("  FAIL :", h)
    print(f"summary: {len(hard)} hard, {len(soft)} soft, {len(thin)} thin cells")
    return 1 if (hard or thin) else 0


if __name__ == "__main__":
    raise SystemExit(main())
