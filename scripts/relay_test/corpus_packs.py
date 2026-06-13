"""Expanded relay test corpus: the original build_corpus() plus the generated
vocab packs (scripts/relay_test/vocab_packs/*.py), wrapped into ~10k unique
relay/conversation cases.

Pack handling:
  * RELAY-CONTENT packs (callouts, agents, directives, opinions, conversation,
    natural phrasing) -> a relay command is expected. Items that are already a
    command ("tell my team ...", "ask Sage ...") are used as-is; raw callouts
    ("two A main") are wrapped with rotating relay prefixes (and multiplied a
    little to widen phrasing coverage + stress the matcher).
  * QUESTION pack (questions_to_ultron) -> direct questions to Ultron that must
    NOT be relayed; they fall through to the conversational pipeline. We assert
    the matcher correctly ignores them (expect_match=False).
  * persona_flavor is Ultron OUTPUT (flavor pool for the framework), NOT a test
    input -- excluded here.

``build_corpus_10k(seed)`` returns the combined, de-duplicated case list. The
seed reshuffles prefix assignment so successive regenerations phrase the same
scenarios differently (per the user's regenerate-and-reshuffle loop).
"""
from __future__ import annotations

import importlib.util
import os
import random
import re

from corpus import Case, _GROUP_PREFIXES, build_corpus

_PACK_DIR = os.path.join(os.path.dirname(__file__), "vocab_packs")

_RELAY_PACKS = (
    "callouts_maps", "agents_abilities", "directives_tactics_eco",
    "opinions_maps_meta", "natural_phrasing_edge", "conversation_natural",
)
_QUESTION_PACKS = ("questions_to_ultron",)
# persona_flavor intentionally excluded (Ultron output, not a relay input).

# Items already phrased as a relay / direct-address command -> use verbatim.
_CMD_LEAD_RE = re.compile(
    r"^\s*(tell|ask|let|inform|say|warn|remind|call\s+out|have|get|relay)\b",
    re.IGNORECASE,
)
# Strip a leading wake word (the corpus convention is POST-wake-word text).
_WAKE_LEAD_RE = re.compile(r"^\s*(ultron|kenning)\b[\s,:-]*", re.IGNORECASE)

# Interrupted / clipped phrasings ("tell my team --", "ask my teammates",
# "tell them the") carry no real payload and SHOULD fall through to the
# conversational pipeline rather than relay a fragment -- so for these the
# matcher correctly returns None (expect_match=False).
_INCOMPLETE_RE = re.compile(
    r"(?:--|—)\s*$"                         # trailing dash (cut off)
    r"|\b(?:the|a|an|that|to|of|uh|um|like)\s*$",  # trailing function word
    re.IGNORECASE,
)


def _expect_match(text: str) -> bool:
    """A relay-content item should match UNLESS it is an interrupted fragment
    or has no addressable payload after the trigger."""
    if _INCOMPLETE_RE.search(text.strip()):
        return False
    return True


def _load_pack(name: str) -> list[str]:
    path = os.path.join(_PACK_DIR, name + ".py")
    if not os.path.exists(path):
        return []
    try:
        spec = importlib.util.spec_from_file_location("vp_" + name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        items = list(getattr(mod, "ITEMS", []))
    except Exception:
        return []
    out, seen = [], set()
    for s in items:
        s = _WAKE_LEAD_RE.sub("", str(s)).strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


def _pack_cases(seed: int = 0) -> list[Case]:
    rng = random.Random(seed)
    cases: list[Case] = []
    np_ = len(_GROUP_PREFIXES)
    for pi, name in enumerate(_RELAY_PACKS):
        items = _load_pack(name)
        wide = name in ("callouts_maps", "agents_abilities")
        for ii, item in enumerate(items):
            if _CMD_LEAD_RE.match(item):
                cases.append(Case(item, "pack_" + name,
                                  expect_match=_expect_match(item)))
            else:
                k = 7 if wide else 4
                used = set()
                for j in range(k):
                    pre = _GROUP_PREFIXES[(ii + j + pi + seed) % np_]
                    if pre in used:
                        continue
                    used.add(pre)
                    cases.append(Case(f"{pre} {item}", "pack_" + name,
                                      expect_match=_expect_match(item)))
    for name in _QUESTION_PACKS:
        for item in _load_pack(name):
            cases.append(Case(item, "pack_" + name, expect_match=False))
    rng.shuffle(cases)
    return cases


def _compound_cases(seed: int = 0, target: int = 2000) -> list[Case]:
    """Realistic multi-piece comms: a raw callout + a tactical/ult/directive
    tail ("two B and their Killjoy has ult", "one mid, we save this round").
    Stresses the rephrase on compound info the way real call-outs combine it."""
    rng = random.Random(1000 + seed)
    heads = [s for s in _load_pack("callouts_maps") if not _CMD_LEAD_RE.match(s)]
    tails = []
    for nm in ("agents_abilities", "directives_tactics_eco"):
        tails += [s for s in _load_pack(nm) if not _CMD_LEAD_RE.match(s)]
    if not heads or not tails:
        return []
    joiners = (" and ", ", ", " -- ", ", also ", " plus ")
    out, seen = [], set()
    attempts = 0
    while len(out) < target and attempts < target * 4:
        attempts += 1
        h = rng.choice(heads)
        t = rng.choice(tails)
        j = joiners[rng.randrange(len(joiners))]
        pre = _GROUP_PREFIXES[rng.randrange(len(_GROUP_PREFIXES))]
        text = f"{pre} {h}{j}{t}"
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(Case(text, "pack_compound", expect_match=True))
    return out


def build_corpus_10k(seed: int = 0) -> list[Case]:
    base = build_corpus()
    packs = _pack_cases(seed) + _compound_cases(seed)
    seen = set()
    out: list[Case] = []
    for c in base + packs:
        key = (c.text.strip().lower(), c.category)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


if __name__ == "__main__":
    cs = build_corpus_10k()
    pos = sum(1 for c in cs if c.expect_match)
    print(f"total={len(cs)}  expect_match={pos}  no_match={len(cs) - pos}")
    from collections import Counter
    for cat, n in Counter(c.category for c in cs).most_common(40):
        print(f"  {n:>5}  {cat}")
