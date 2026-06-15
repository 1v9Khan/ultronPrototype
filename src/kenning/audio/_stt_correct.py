"""Valorant-aware STT correction for the live callout path.

The CPU-side Moonshine STT mishears domain words it has never been biased toward
-- agent names especially ("Sova"->"Silva", "Reyna"->"Royal", "Jett"->"jet")
and tactical terms ("ult"->"sold/old"). This layer fixes them with (1) a curated
mishear map for the big phonetic misses a generic fuzzy match can't reach, and
(2) a fuzzy fallback (difflib) that snaps any token CLOSE to a real agent name
onto the canonical spelling. Applied ONLY as a fallback in the relay path (after
the clean text fails to match), so clear speech is never over-corrected.
"""
from __future__ import annotations

import difflib
import re

# Canonical Valorant agent roster (display spelling the flavor library keys on).
_AGENTS = (
    "Astra", "Breach", "Brimstone", "Chamber", "Clove", "Cypher", "Deadlock",
    "Fade", "Gekko", "Harbor", "Iso", "Jett", "KAY/O", "Killjoy", "Neon",
    "Omen", "Phoenix", "Raze", "Reyna", "Sage", "Skye", "Sova", "Viper",
    "Vyse", "Yoru", "Tejo", "Waylay", "Miks", "Veto",
)
_AGENT_LOWER = {a.lower().replace("/", ""): a for a in _AGENTS}

# Curated mishears the fuzzy pass cannot reach (the STT produced a genuinely
# different word). lower-case single tokens -> canonical replacement.
_MISHEARS = {
    # agents
    "silva": "Sova", "selva": "Sova", "sofa": "Sova", "soda": "Sova",
    "soever": "Sova", "sovereign": "Sova",
    "royal": "Reyna", "raina": "Reyna", "rayna": "Reyna", "reina": "Reyna",
    "rena": "Reyna", "regina": "Reyna",
    "jet": "Jett", "jed": "Jett", "jett's": "Jett",
    "cipher": "Cypher", "sypher": "Cypher", "cyphus": "Cypher",
    "kayo": "KAY/O", "cale": "KAY/O", "kayle": "KAY/O", "cайо": "KAY/O",
    "oman": "Omen", "open": "Omen", "omens": "Omen",
    "vyper": "Viper", "wiper": "Viper",
    "race": "Raze", "rays": "Raze", "raisa": "Raze", "raise": "Raze",
    "felix": "Phoenix", "phoenix's": "Phoenix",
    "gecko": "Gekko", "geko": "Gekko",
    "breech": "Breach", "bridge": "Breach",
    "fate": "Fade", "faded": "Fade",
    "sky": "Skye", "ski": "Skye",
    "aster": "Astra", "astro": "Astra",
    "arbor": "Harbor", "harbour": "Harbor",
    "clive": "Clove", "clove's": "Clove",
    "brimston": "Brimstone", "grimstone": "Brimstone", "brimstine": "Brimstone",
    "deadlocke": "Deadlock", "dead lock": "Deadlock",
    "neon's": "Neon", "nian": "Neon",
    "yору": "Yoru", "euro": "Yoru", "yoda": "Yoru",
    # tactical terms
    "sold": "ult", "old": "ult", "ulta": "ult", "alt": "ult", "oat": "ult",
    "ulti": "ult", "ultes": "ult", "ulted": "ulted", "ultимейт": "ultimate",
    "diffuse": "defuse", "diffusing": "defusing",
}

# Words we must NOT fuzzy-snap onto an agent (common English that is close to a
# short agent name -- "is/iso", "sky/Skye", "neon" sign, etc.). Only the curated
# map touches these.
_FUZZY_BLOCK = {"is", "in", "it", "i", "a", "an", "the", "of", "on", "no",
                "so", "go", "to", "up", "we", "he", "me", "be", "by", "raze",
                "sage", "fade", "neon", "iso", "omen", "clove", "viper"}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'/]*")


def _fix_token(tok: str) -> str:
    low = tok.lower()
    if low in _MISHEARS:
        return _MISHEARS[low]
    if low in _AGENT_LOWER:                      # already a canonical agent
        return _AGENT_LOWER[low]
    if len(low) >= 4 and low not in _FUZZY_BLOCK:
        m = difflib.get_close_matches(low, list(_AGENT_LOWER), n=1, cutoff=0.82)
        if m:
            return _AGENT_LOWER[m[0]]
    return tok


def correct_callout_stt(text: str) -> str:
    """Snap mis-transcribed agent names + tactical terms back to canon.

    Token-wise; punctuation/spacing preserved. Returns ``text`` unchanged when
    nothing matches. Safe to call on already-correct text (canonical words pass
    through), but the caller uses it as a fallback so clear speech is untouched.
    """
    if not text:
        return text
    return _WORD_RE.sub(lambda m: _fix_token(m.group(0)), text)
