"""Expanded relay test corpus: the original build_corpus() plus the generated
vocab packs (scripts/relay_test/vocab_packs/*.py), built into a 20,000-case
relay/conversation/negative corpus.

Pack kinds (auto-discovered; persona_flavor excluded as it is Ultron OUTPUT):
  * RELAY    -> a relay command is expected (expect_match=True). Items already
    phrased as a command ("tell my team ...") are used verbatim; raw callouts
    ("two A main") are wrapped with a rotating relay prefix.
  * QUESTION -> a teammate talking TO Ultron; must NOT relay (expect_match=False).
  * NEGATIVE -> stream narration / private thought / out-of-roster addressee that
    is relay-SHAPED but must NOT relay (expect_match=False) -- the false-relay gate.

There are ~29k unique pack payloads; ``build_corpus`` caps the corpus at 20,000
via a category-stratified sample, and the ``seed`` reshuffles BOTH the prefix
assignment and which subset is sampled -- so each iteration's regeneration
exercises a different 20k slice of the full pool (full coverage over the loop).
"""
from __future__ import annotations

import importlib.util
import os
import random
import re
from collections import defaultdict

from corpus import Case, _GROUP_PREFIXES, build_corpus as _orig_build_corpus

_PACK_DIR = os.path.join(os.path.dirname(__file__), "vocab_packs")
_TARGET = 20000

# Teammate-talking-TO-Ultron packs -> answered, never relayed (expect_match=False).
_QUESTION_PACKS = frozenset((
    "questions_to_ultron", "var_teammate_to_ultron", "var_identity_questions",
    "var_marvel_banter", "var_banter_at_ultron", "stress_banter_mock",
    "stress_marvel_identity_edge",
))
# Relay-SHAPED but must NOT relay (false-relay gate): narration + out-of-roster.
_NEGATIVE_PACKS = frozenset(("stress_false_relay_hard", "stress_oov_safety"))
# Ultron OUTPUT pools / non-inputs -> never a test input.
_EXCLUDE_PACKS = frozenset(("persona_flavor", "__init__"))


def _all_pack_names() -> list[str]:
    try:
        names = [f[:-3] for f in os.listdir(_PACK_DIR)
                 if f.endswith(".py") and f != "__init__.py"]
    except FileNotFoundError:
        return []
    return sorted(names)


def _relay_pack_names() -> list[str]:
    return [n for n in _all_pack_names()
            if n not in _QUESTION_PACKS and n not in _NEGATIVE_PACKS
            and n not in _EXCLUDE_PACKS]


# Items already phrased as a relay / direct-address command -> use verbatim.
# 'repeat'/'echo' lead the verbatim soundboard-check command ("repeat to my team
# X"), which carries its own addressee and must NOT be re-wrapped with a prefix.
_CMD_LEAD_RE = re.compile(
    r"^\s*(tell|ask|let|inform|say|warn|remind|call\s+out|have|get|relay"
    r"|repeat|echo)\b",
    re.IGNORECASE,
)
# Strip a leading wake word (the corpus convention is POST-wake-word text).
_WAKE_LEAD_RE = re.compile(r"^\s*(ultron|kenning)\b[\s,:-]*", re.IGNORECASE)

# A whole-utterance bare trigger skeleton with no real payload ("tell my team --",
# "ask my teammates") -> interrupted fragment, correctly falls through.
_INCOMPLETE_RE = re.compile(
    r"^\W*(?:please\s+|hey\s+|just\s+|ok(?:ay)?\s+)?"
    r"(?:tell|say|ask|let|warn|inform|remind|relay)\b"
    r"(?:\s+(?:to|my|our|the|whole|guys?|team\s?mates?|teams?|squad|others?|"
    r"lobby|chat|them|everyone|everybody|know|about|that|the|a|an|uh|um))*"
    r"\s*(?:--|—)?\s*$",
    re.IGNORECASE,
)


def _expect_match(text: str) -> bool:
    return not _INCOMPLETE_RE.search(text.strip())


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
    """One relay prefix per raw callout (varied by seed) -- with ~29k unique
    payloads we maximize distinct-scenario coverage rather than multiplying
    near-duplicate prefixes; the stratified cap then trims to 20k."""
    cases: list[Case] = []
    np_ = len(_GROUP_PREFIXES)
    for pi, name in enumerate(_relay_pack_names()):
        for ii, item in enumerate(_load_pack(name)):
            cat = "pack_" + name
            if _CMD_LEAD_RE.match(item):
                cases.append(Case(item, cat, expect_match=_expect_match(item)))
            else:
                pre = _GROUP_PREFIXES[(ii + pi + seed) % np_]
                cases.append(Case(f"{pre} {item}", cat,
                                  expect_match=_expect_match(item)))
    for name in _QUESTION_PACKS:
        for item in _load_pack(name):
            cases.append(Case(item, "pack_" + name, expect_match=False))
    for name in _NEGATIVE_PACKS:
        for item in _load_pack(name):
            cases.append(Case(item, "neg_" + name, expect_match=False))
    return cases


def _compound_cases(seed: int = 0, target: int = 2000) -> list[Case]:
    """Procedural multi-piece comms (callout head + tactical tail) on top of the
    hand-written stress_compounds packs -- extra compound pressure."""
    rng = random.Random(1000 + seed)
    heads = [s for s in _load_pack("callouts_maps") if not _CMD_LEAD_RE.match(s)]
    heads += [s for s in _load_pack("var_positions_counts")
              if not _CMD_LEAD_RE.match(s)][:400]
    tails = []
    for nm in ("agents_abilities", "directives_tactics_eco",
               "var_utility_reports", "var_ult_states"):
        tails += [s for s in _load_pack(nm) if not _CMD_LEAD_RE.match(s)][:300]
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


def _cap_stratified(cases: list[Case], target: int, seed: int) -> list[Case]:
    """Trim to `target` keeping every category proportionally represented."""
    if len(cases) <= target:
        random.Random(seed).shuffle(cases)
        return cases
    rng = random.Random(seed)
    by_cat: dict[str, list[Case]] = defaultdict(list)
    for c in cases:
        by_cat[c.category].append(c)
    for v in by_cat.values():
        rng.shuffle(v)
    total = len(cases)
    out: list[Case] = []
    for cat, group in by_cat.items():
        share = max(1, round(target * len(group) / total))
        out.extend(group[:share])
    rng.shuffle(out)
    return out[:target]


def build_corpus(seed: int = 0, target: int = _TARGET) -> list[Case]:
    base = _orig_build_corpus()
    packs = _pack_cases(seed) + _compound_cases(seed)
    seen = set()
    deduped: list[Case] = []
    for c in base + packs:
        key = (c.text.strip().lower(), c.category)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return _cap_stratified(deduped, target, seed)


# Back-compat aliases (harness.py / scorecard.py import build_corpus_10k).
def build_corpus_10k(seed: int = 0) -> list[Case]:
    return build_corpus(seed, _TARGET)


build_corpus_20k = build_corpus_10k


if __name__ == "__main__":
    import sys
    sd = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cs = build_corpus(sd)
    pos = sum(1 for c in cs if c.expect_match)
    print(f"total={len(cs)}  expect_match={pos}  no_match={len(cs) - pos}")
    from collections import Counter
    for cat, n in Counter(c.category for c in cs).most_common(60):
        print(f"  {n:>5}  {cat}")
