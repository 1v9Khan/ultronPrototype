"""Valorant-aware STT correction for the live callout path.

The CPU-side Moonshine STT has no hotword/biasing hook, so it mishears domain
vocabulary it was never trained to favour -- agent names ("Sova"->"Silva"),
abilities ("ult"->"sold/old"), and callout terms. This layer fixes them with
pure string work (microseconds -> negligible latency), in three stages:

  1. CONTEXT rules -- regexes that disambiguate words which are ALSO normal
     English ("old" -> "ult" only after "has/their/popped ...", never on its
     own), so we recover slang without corrupting real words.
  2. CURATED maps -- the big phonetic misses a fuzzy match can't reach
     (Silva->Sova, Royal->Reyna), plus an unambiguous Valorant term lexicon.
  3. FUZZY agent-snap -- difflib snaps any leftover token CLOSE to a real agent
     onto the canonical spelling (catches misses we didn't enumerate).

Used ONLY as a relay fallback (after the clean text fails to match), and clean
callouts pass through unchanged, so well-transcribed speech is never altered.
"""
from __future__ import annotations

import difflib
import re

# --- Canonical Valorant agent roster (the flavor library keys on these) ------
_AGENTS = (
    "Astra", "Breach", "Brimstone", "Chamber", "Clove", "Cypher", "Deadlock",
    "Fade", "Gekko", "Harbor", "Iso", "Jett", "KAY/O", "Killjoy", "Neon",
    "Omen", "Phoenix", "Raze", "Reyna", "Sage", "Skye", "Sova", "Viper",
    "Vyse", "Yoru", "Tejo", "Waylay", "Miks", "Veto",
)
_AGENT_LOWER = {a.lower().replace("/", ""): a for a in _AGENTS}

# --- 1. Context rules: disambiguate words that are ALSO real English ---------
# "ult" is the worst offender -- the STT writes old/sold/alt/oat/vault/halt for
# it. Only rewrite to "ult" when the grammar makes it a callout (after a
# possessive/verb, or before "is up / ready / coming"), so a literal "old" /
# "fall back" is never touched.
_ULTISH = r"(?:old|sold|alt|oat|halt|vault|ault|ulta|ulte|ulti|olt)"
_CONTEXT_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # "<has/have/his/her/their/got/using/popped/pop> [a|an|the] <ultish>" -> ult
    (re.compile(
        r"\b(has|have|his|her|their|got|using|use|pop(?:ped|ping)?|"
        r"is\s+on|on)\s+(?:an?\s+|the\s+)?" + _ULTISH + r"\b", re.I),
     lambda m: m.group(1) + " ult"),
    # "<ultish> is up / ready / coming / is ready" -> "ult ..."
    (re.compile(r"\b" + _ULTISH + r"\s+(is\s+up|up|ready|coming|is\s+ready)\b",
                re.I),
     lambda m: "ult " + m.group(1)),
    # "site a/b/c" mis-segmentations -> "A site" / "B site" / "C site"
    (re.compile(r"\bsite\s+([abc])\b", re.I),
     lambda m: m.group(1).upper() + " site"),
    # "a main / b main" stays, but "amen" (heard for "A main") -> "A main"
    (re.compile(r"\bamen\b", re.I), "A main"),
)

# --- 2a. Curated agent mishears (genuine wrong words the fuzzy pass misses) ---
_AGENT_MISHEARS = {
    "silva": "Sova", "selva": "Sova", "sofa": "Sova", "soda": "Sova",
    "soever": "Sova", "sovereign": "Sova", "saba": "Sova", "sova's": "Sova",
    "royal": "Reyna", "raina": "Reyna", "rayna": "Reyna", "reina": "Reyna",
    "rena": "Reyna", "regina": "Reyna", "reyna's": "Reyna",
    "jet": "Jett", "jed": "Jett", "jett's": "Jett", "jett": "Jett",
    "cipher": "Cypher", "sypher": "Cypher", "cyphus": "Cypher", "sifa": "Cypher",
    "kayo": "KAY/O", "cale": "KAY/O", "kayle": "KAY/O", "kio": "KAY/O",
    "kayoh": "KAY/O", "kay-o": "KAY/O", "que": "KAY/O",
    "oman": "Omen", "open": "Omen", "omens": "Omen", "amen2": "Omen",
    "vyper": "Viper", "wiper": "Viper", "viper's": "Viper",
    "race": "Raze", "rays": "Raze", "raisa": "Raze", "raise": "Raze",
    "raz": "Raze", "raze's": "Raze",
    "felix": "Phoenix", "phoenix's": "Phoenix", "fenix": "Phoenix",
    "gecko": "Gekko", "geko": "Gekko", "gekko's": "Gekko",
    "breech": "Breach", "bridge": "Breach", "breeze": "Breach",
    "fate": "Fade", "faded": "Fade", "fad": "Fade",
    "sky": "Skye", "ski": "Skye", "skype": "Skye",
    "aster": "Astra", "astro": "Astra", "extra": "Astra",
    "arbor": "Harbor", "harbour": "Harbor", "harper": "Harbor",
    "clive": "Clove", "clove's": "Clove", "clo": "Clove", "claw": "Clove",
    "brimston": "Brimstone", "grimstone": "Brimstone", "brimstine": "Brimstone",
    "brim": "Brimstone", "brimstone's": "Brimstone",
    "deadlocke": "Deadlock", "dead lock": "Deadlock", "deadbolt": "Deadlock",
    "neon's": "Neon", "nian": "Neon", "neyon": "Neon",
    "euro": "Yoru", "yoda": "Yoru", "yoru's": "Yoru", "yo-ru": "Yoru",
    "kill joy": "Killjoy", "killjoy's": "Killjoy", "kiljoy": "Killjoy",
    "chambers": "Chamber", "chamber's": "Chamber",
    "teacho": "Tejo", "taho": "Tejo", "tejo's": "Tejo",
    "way lay": "Waylay", "weigh lay": "Waylay",
    "mix": "Miks", "meeks": "Miks",
    "vice": "Vyse", "vise": "Vyse", "vis": "Vyse",
    "fage": "Fade", "geco": "Gekko", "deadshot": "Deadlock",
}

# --- 2b. Unambiguous Valorant TERM lexicon (safe -> never a normal-word miss) -
_TERM_MISHEARS = {
    "ulted": "ulted", "ulting": "ulting", "ultimate": "ult", "ultima": "ult",
    "diffuse": "defuse", "diffusing": "defusing", "diffused": "defused",
    "molotov": "molly", "incendiary": "molly", "mollie": "molly",
    "tripwire": "tripwire", "trip wire": "tripwire",
    "alarm bot": "alarmbot", "nano swarm": "nanoswarm", "nanoswarm": "nanoswarm",
    "recon dart": "recon dart", "owl drone": "drone",
    "smokes": "smokes", "spike": "spike", "the spike": "the spike",
}

_MISHEARS = {**_AGENT_MISHEARS, **_TERM_MISHEARS}

# Common English that is phonetically close to a short agent name -- only the
# curated map may touch these, never the fuzzy pass.
_FUZZY_BLOCK = {
    "is", "in", "it", "i", "a", "an", "the", "of", "on", "no", "so", "go",
    "to", "up", "we", "he", "me", "be", "by", "see", "say", "way", "play",
    "they", "them", "their", "there", "here", "have", "has", "had", "site",
    "main", "mid", "left", "right", "back", "push", "hold", "rotate", "save",
    # short agent names that ARE common words -> curated-map-only
    "raze", "sage", "fade", "neon", "iso", "omen", "clove", "viper", "skye",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'/]*")


def _fix_token(tok: str) -> str:
    low = tok.lower()
    if low in _MISHEARS:
        return _MISHEARS[low]
    if low in _AGENT_LOWER:                       # already canonical
        return _AGENT_LOWER[low]
    if len(low) >= 4 and low not in _FUZZY_BLOCK:
        m = difflib.get_close_matches(low, list(_AGENT_LOWER), n=1, cutoff=0.82)
        if m:
            return _AGENT_LOWER[m[0]]
    return tok


def correct_callout_stt(text: str) -> str:
    """Snap mis-transcribed agents + tactical vocab back to canon.

    Stage 1 context rules (disambiguate "ult" etc.) -> stage 2/3 token fixes.
    Returns ``text`` unchanged when nothing matches; idempotent on already-clean
    callouts. Negligible cost (regex + short-string difflib on ~10 tokens).
    """
    if not text:
        return text
    for pat, rep in _CONTEXT_RULES:
        text = pat.sub(rep, text)
    return _WORD_RE.sub(lambda m: _fix_token(m.group(0)), text)
