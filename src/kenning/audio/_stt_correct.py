"""Valorant-aware STT correction for the live callout path.

Whisper (and the prior Moonshine) have no hotword/biasing hook, so they mishear
domain vocabulary they were never trained to favour -- agent names
("Sova"->"Silva"), abilities ("ult"->"sold/old"), maps, weapons, callout
locations -- and they BLEND adjacent words ("Raze on B" -> "ray zombie",
"our Sova" -> "Arsova"). This layer repairs them at the text level (no model
retraining), using the frontier post-ASR-correction recipe: a domain GAZETTEER
(bias list) matched by PHONETIC encoding (Metaphone) + EDIT distance
(Jaro-Winkler / Levenshtein, via rapidfuzz) over an n-gram window, with
CONTEXT re-evaluation and conservative thresholds so real words are never
corrupted. Pure string work -> microseconds, deterministic, no hallucination.

Stages (each conservative + idempotent on already-clean text):
  0. PHRASE mishears  -- multi-word blends a token pass can't see across spaces.
  1. CONTEXT rules    -- disambiguate words that are ALSO English ("old"->"ult"
                         only in callout grammar; "site a"->"A site").
  2. TOKEN fixes      -- curated map -> already-canonical -> PHONETIC+FUZZY snap
                         onto the gazetteer (Metaphone match AND/OR high
                         Jaro-Winkler), gated by a common-word block list.
  3. N-GRAM spans     -- 2-word spans snapped onto multi-word gazetteer terms
                         ("recon dart", "owl drone", "alarm bot") by phonetics.

``correct_callout_stt`` is the entry point. The command normalizer only calls
it on NON-conversational (callout-bound) text, so questions / Spotify / chatter
are never touched -- aggressive correction stays where it belongs.
"""
from __future__ import annotations

import difflib
import re
from collections import defaultdict

try:                                                              # phonetic
    import jellyfish as _jf
except Exception:                                                 # noqa: BLE001
    _jf = None
try:                                                              # fast fuzzy
    from rapidfuzz import process as _rf_process
    from rapidfuzz.distance import JaroWinkler as _rf_jw
except Exception:                                                 # noqa: BLE001
    _rf_process = None
    _rf_jw = None

# Baked frequency-ranked common-English-word set (pure-python data, no heavy
# import -> anticheat-safe). Protects real words from being rewritten by the
# phonetic/fuzzy snapper: a closed-domain corrector must only ever touch
# genuinely OOV/misheard tokens, never an in-vocabulary common word. This
# replaces the (necessarily incomplete) hand-curated _FUZZY_BLOCK denylist as
# the primary protection -- e.g. "let"->"lit", "mean"->"main" are killed here.
# Regenerate via scripts/build_common_words.py.
try:
    from ._common_words import COMMON_WORDS as _COMMON_WORDS
except Exception:                                                 # noqa: BLE001
    _COMMON_WORDS = frozenset()


# ===========================================================================
# 1. DOMAIN GAZETTEER (the bias list)
# ===========================================================================
# --- Agents (canonical display incl. the KAY/O slash) ----------------------
# 2026-06-18 Part B (routing aggregate): the STT-correction vocabulary +
# mishear maps + protection sets are the editable NORMALIZATION-LAYER-1 RULES,
# relocated to kenning.audio.routing_rules (Section 1) -- edit them THERE.
# Imported here (aliased to the existing private names); the derivation below
# (phonetic index, lower-case maps, merged mishears) is UNCHANGED. Behaviour is
# byte-for-byte identical (proven by scripts/_voice_lines_verify.py).
from kenning.audio.routing_rules import (  # noqa: E402
    AGENTS as _AGENTS, MAPS as _MAPS, WEAPONS as _WEAPONS,
    ABILITIES as _ABILITIES, LOCATIONS as _LOCATIONS, TERMS as _TERMS,
    AGENT_MISHEARS as _AGENT_MISHEARS, TERM_MISHEARS as _TERM_MISHEARS,
    MISHEAR_FORCE as _MISHEAR_FORCE, FUZZY_BLOCK as _FUZZY_BLOCK,
    PROTECT_EXTRA as _PROTECT_EXTRA, MULTI_TERMS as _MULTI_TERMS,
)

# Single-token canonical set (multi-word terms handled by phrase/n-gram).
_GAZ_LOWER: dict[str, str] = {}
for _grp in (_AGENTS, _MAPS, _WEAPONS, _ABILITIES, _LOCATIONS, _TERMS):
    for _t in _grp:
        _w = _t.replace("/", "")
        if " " not in _w:
            _GAZ_LOWER.setdefault(_w.lower(), _t if _t in _AGENTS or _t in _MAPS
                                  or _t in _WEAPONS else _w)
# Agents keep the canonical display ("KAY/O", "Sova"); the rest map to their
# lowercase token (the flavor/relay path expects lowercase tactical words).
_AGENT_LOWER = {a.lower().replace("/", ""): a for a in _AGENTS}
_GAZ_LOWER.update(_AGENT_LOWER)

# --- Phonetic index: Metaphone code -> canonical(s) (single tokens) ---------
_PHONETIC_INDEX: dict[str, set] = defaultdict(set)
if _jf is not None:
    for _low, _canon in _GAZ_LOWER.items():
        try:
            _code = _jf.metaphone(_low)
        except Exception:                                         # noqa: BLE001
            _code = ""
        if _code:
            _PHONETIC_INDEX[_code].add(_canon)

_GAZ_KEYS = tuple(_GAZ_LOWER.keys())


# ===========================================================================
# 2. CONTEXT RULES (disambiguate words that are ALSO real English)
# ===========================================================================
_ULTISH = r"(?:old|sold|alt|oat|halt|vault|ault|ulta|ulte|ulti|olt|ault|aunt|volt|volts)"
_CONTEXT_RULES: tuple[tuple[re.Pattern[str], object], ...] = (
    (re.compile(
        r"\b(has|have|his|her|their|got|using|use|pop(?:ped|ping)?|"
        r"is\s+on|on)\s+(?:an?\s+|the\s+)?" + _ULTISH + r"\b", re.I),
     lambda m: m.group(1) + " ult"),
    (re.compile(r"\b" + _ULTISH + r"\s+(is\s+up|up|ready|coming|is\s+ready|now)\b",
                re.I),
     lambda m: "ult " + m.group(1)),
    # site/main letter mis-segmentation
    (re.compile(r"\bsite\s+([abc])\b", re.I), lambda m: m.group(1).upper() + " site"),
    (re.compile(r"\b([abc])\s+site\b", re.I), lambda m: m.group(1).upper() + " site"),
    (re.compile(r"\bamen\b", re.I), "A main"),
    (re.compile(r"\bay\s+main\b", re.I), "A main"),
    # "on a / on b / on c" as a site reference -> capitalise the letter
    (re.compile(r"\bon\s+([abc])\b(?!\w)", re.I), lambda m: "on " + m.group(1).upper()),
)


# _AGENT_MISHEARS / _TERM_MISHEARS -> kenning.audio.routing_rules Section 1
# (imported above). The merge below stays here (derivation, not a rule).
_MISHEARS = {**_AGENT_MISHEARS, **_TERM_MISHEARS}

# _MISHEAR_FORCE -> kenning.audio.routing_rules Section 1 (imported above).


# ===========================================================================
# 3b. MULTI-WORD phrase mishears (blends the token pass can't see) -----------
# ===========================================================================
# The STT collapses "<agent> <prep> <site>" or "<adj> <agent>" into a single
# garbled phrase. Recover the WHOLE callout. More get added as the
# routing:normalized logs (raw + normalized, every turn) surface them.
_PHRASE_MISHEARS: tuple[tuple[re.Pattern[str], object], ...] = (
    # "Raze on B" -> "ray-zom-bie" ("ray zombie") -- recover agent + site.
    (re.compile(r"\b(?:ray|raise[ds]?|race|res)\s+zombie\b", re.I), "Raze on B"),
    (re.compile(r"\brays?\s+own\b", re.I), "Raze"),
    # owner-aware "our Sova" -> "Arsova" / "our selva".
    (re.compile(r"\bar+\s*sova\b", re.I), "our Sova"),
    (re.compile(r"\bour\s+(?:silva|selva|sofa|soda|saba|sovah)\b", re.I), "our Sova"),
    # "roast my team" mis-heard as "toast my team" (STT swaps r->t before the
    # rounded vowel). Only when a team/group reference follows, so a literal
    # "toast" (rare) is untouched. Live: "Toast, my team" mis-routed to the long
    # greeting because the un-corrected "toast" matched no relay verb.
    (re.compile(
        r"\btoast(?=[\s,]+(?:my\s+|our\s+|the\s+)?"
        r"(?:team|teammates?|squad|guys|boys|mates|crew|them|everyone|chat)\b)",
        re.I), "roast"),
    # "mic check" heard as "Mike check" / "my check".
    (re.compile(r"\b(?:mike|my)\s+check\b", re.I), "mic check"),
    # "Reyna" mis-heard as "rain a" / "rain uh" (the "-eyna" tail -> "ain a").
    # Live: "tell my Reyna nice try" -> "tell my rain a nice try" -> LLM. Gated
    # to an agent-reference lead so a literal "rain" elsewhere is untouched.
    (re.compile(r"\b(my|our|the|tell|ask|told|for)\s+rain\s+(?:a|uh)\b", re.I),
     lambda m: m.group(1) + " Reyna"),
    # Joined multi-word LOCATIONS (STT drops the space): "back site" -> "Backsite",
    # "top mid" -> "topmid", etc. _LOC_TOKENS holds the words SEPARATELY, so the
    # joined form is not a valid location and a multi-agent callout SILENTLY DROPS
    # that segment ("Cypher backsite, Sova heaven" -> just "Sova heaven" -- the
    # live "Sova deletes the rest" report). Re-split them. (\s*-? also matches the
    # already-correct spaced form, so this is idempotent.)
    (re.compile(r"\bback\s*-?site\b", re.I), "back site"),
    (re.compile(r"\bfront\s*-?site\b", re.I), "front site"),
    (re.compile(r"\bmid\s*-?site\b", re.I), "mid site"),
    (re.compile(r"\btop\s*-?mid\b", re.I), "top mid"),
    (re.compile(r"\bmid\s*-?top\b", re.I), "mid top"),
    # site letters heard as words.
    (re.compile(r"\b(?:bee?|be)\s+(main|site|long|short)\b", re.I),
     lambda m: "B " + m.group(1).lower()),
    (re.compile(r"\b(?:see|sea|cee)\s+(main|site|long|short)\b", re.I),
     lambda m: "C " + m.group(1).lower()),
    (re.compile(r"\ba\s+main[e]?\b", re.I), "A main"),
    (re.compile(r"\bhey\s+main\b", re.I), "A main"),
    # Site letter at the END of a movement order heard as a word: "rotate to be"
    # -> "rotate to B", "push to see" -> "push to C". Gated to a movement verb so
    # a conversational "going to be there" (which never reaches this callout-only
    # corrector anyway) stays safe.
    (re.compile(
        r"\b(rotate|rotating|push|pushing|fall\s*back|falling\s*back|go|going|"
        r"move|moving|rush|rushing|split|swing|swinging|cross|crossing|head|"
        r"heading|hit|hitting|take|taking|send\s+it|over)\s+to\s+be\b", re.I),
     lambda m: m.group(1) + " to B"),
    (re.compile(
        r"\b(rotate|rotating|push|pushing|fall\s*back|falling\s*back|go|going|"
        r"move|moving|rush|rushing|split|swing|swinging|cross|crossing|head|"
        r"heading|hit|hitting|take|taking|send\s+it|over)\s+to\s+(?:see|sea|cee)\b",
        re.I),
     lambda m: m.group(1) + " to C"),
    # "hey <agent>" blends into "Hell<agent>" ("hey Sage" -> "Hellsage", "hey
    # Jett" -> "Helljet") -- the greeting glues to the name and the location
    # "hell" surfaces. Drop the glued prefix so only the agent (then praised)
    # remains. Requires NO space (a real "hell" location keeps its space).
    (re.compile(
        r"\bhell(?=(?:sage|jett|jet|reyna|raze|sova|omen|neon|viper|cypher|"
        r"killjoy|phoenix|breach|fade|skye|astra|harbor|clove|chamber|"
        r"brimstone|gekko|yoru|iso|deadlock|gekko|tejo|waylay|vyse)\b)",
        re.I), ""),
    # Number-word mishears in a COUNT context only: "three pushing" -> heard
    # "tree pushing"; "one mid" -> "won mid". Gated to a following push/site
    # token so the location "tree" ("split through tree") and "we won" are safe.
    (re.compile(
        r"\btree\b(?=\s+(?:pushing|rushing|coming|going|rotating|on\b|in\b|"
        r"at\b|enemies|enemy|guys|of\s+them|men))", re.I), "three"),
    (re.compile(
        r"\bwon\b(?=\s+(?:mid|middle|a\b|b\b|c\b|on\b|in\b|at\b|heaven|hell|"
        r"main|long|short|site|pushing|rushing|rotating|enemy|enemies))",
        re.I), "one"),
    # multi-word abilities the STT splits.
    (re.compile(r"\b(?:cale|kayo|kayle|kio)\s+knife\b", re.I), "KAY/O knife"),
    (re.compile(r"\bspikes\s+down\b", re.I), "spike is down"),
    # Sova's real ability name is "Recon Bolt" (the flavor/TailEntry kit data uses
    # "recon bolt") -- never mangle it back to the legacy "recon dart".
    (re.compile(r"\brecon\s+(?:dart|bolt)\b", re.I), "recon bolt"),
    (re.compile(r"\bowl\s+drone\b", re.I), "drone"),
    (re.compile(r"\bnano\s*swarm\b", re.I), "nanoswarm"),
    (re.compile(r"\balarm\s*bot\b", re.I), "alarmbot"),
    (re.compile(r"\btrip\s*wire\b", re.I), "tripwire"),
    # --- 2026-06-17 battery STT repairs (context-anchored; common words survive) ---
    # "black window" -> the Marvel hero "Black Widow".
    (re.compile(r"\bblack\s+window\b", re.I), "Black Widow"),
    # "playoff site" / "playoff" <- "play off (site)" (post-plant off-site hold).
    (re.compile(r"\bplay[\s-]?off\s+site\b", re.I), "play off site"),
    (re.compile(r"\bplayoff\b", re.I), "play off"),
    # "<agent>-Walt" / "<agent> Walt" <- "<agent> walled" (used their wall).
    (re.compile(r"\b(sage|viper|harbor|clove|omen|astra|brimstone|phoenix)[\s-]+"
                r"walt\b", re.I), lambda m: m.group(1).capitalize() + " walled"),
    # "Mysawa" / "my sava" <- "my Sova".
    (re.compile(r"\bmy\s*sawa\b|\bmysava\b|\bmysawa\b", re.I), "my Sova"),
    # "flame jet" / "flamejet" <- "flame Jett" (roast a teammate).
    (re.compile(r"\bflame[\s-]?jet\b", re.I), "flame Jett"),
    # "Ruse said/asked/told/gg ..." <- the agent "Yoru".
    (re.compile(r"\bruse\b(?=\s+(?:said|says|asked|told|tells|called|is|gg))", re.I),
     "Yoru"),
    # dropped first-person "I" on a sound callout: "Here's some <loc>" / "hear
    # <count> <loc>" <- "I hear ...".
    (re.compile(r"\bhere'?s\s+some\b", re.I), "I hear some"),
    (re.compile(r"^\s*hear\s+(?=(?:one|two|three|four|five|lots|some|footsteps)\b)",
                re.I), "I hear "),
    # "raise ult(s)" <- the agent "Raze" + her ult.
    (re.compile(r"\braise[ds]?\s+(ult|ulted|ults)\b", re.I),
     lambda m: "Raze " + m.group(1)),
    # "drop/buy me a share" <- the Sheriff pistol.
    (re.compile(r"\b(drop|buy|get|need|grab)\s+(me\s+)?an?\s+share\b", re.I),
     lambda m: m.group(1) + " " + (m.group(2) or "") + "a Sheriff"),
    # "(could) be rapping" <- "wrapping" (the enemy wrapping around).
    (re.compile(r"\bbe\s+rapping\b", re.I), "be wrapping"),
    # "Tukat" / "too cat" <- "two cat" (two enemies at catwalk).
    (re.compile(r"\btukat\b|\btoo\s*cat\b", re.I), "two cat"),
)


# ===========================================================================
# 4. Common English close to a short agent name -> curated/phonetic only -----
# ===========================================================================
# _FUZZY_BLOCK -> kenning.audio.routing_rules Section 1 (imported above).

# 2026-06-16 (C2): real English / kit words the snapper or the _GAZ_LOWER-direct
# branch corrupted (live-confirmed). Consulted ONLY for the gaz-direct branch and
# the snap guard -- NOT folded into the gaz-branch as _FUZZY_BLOCK (that would
# decap clean agents raze/sage/neon which are in _FUZZY_BLOCK but must stay
# canonical) and NOT _COMMON_WORDS (that would decap Chamber/Ghost/Judge). The
# _MISHEAR_FORCE escape hatch still overrides these. Deliberately EXCLUDES "ego"
# (ego->eco is a useful STT fix) and "incendiary" (incendiary->molly is the
# colloquial kit name) -- both are live-verify trade-offs, kept as corrections.
# _PROTECT_EXTRA -> kenning.audio.routing_rules Section 1 (imported above).

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'/]*")


# ===========================================================================
# 5. Token-level fix: curated -> canonical -> phonetic+fuzzy snap
# ===========================================================================
def _phonetic_fuzzy_snap(low: str) -> str | None:
    """Snap a token onto the gazetteer by Metaphone + Jaro-Winkler. Conservative:
    requires a phonetic-code match OR a very high edit-similarity, so real words
    are not corrupted. Returns the canonical form or None."""
    if (len(low) < 3 or low in _FUZZY_BLOCK or low in _COMMON_WORDS
            or low in _PROTECT_EXTRA):
        return None
    # An INFLECTED real word is a genuine usage, not a mishear of a base gazetteer
    # noun: snapping "walled"->wall, "haunted"->haunt, "darted"->dart, "prowlers"
    # ->prowler, "orbs"->orb breaks the grammar of the relayed callout. A -ed/-ing
    # verb is never a noun-ability mishear; a plural of a real/gazetteer word keeps
    # its plural. (A true mishear of "wall" looks like "waul"/"wal", not "walled".)
    if len(low) >= 5 and low.endswith(("ed", "ing", "ers")):
        return None
    if len(low) >= 4 and low.endswith("s"):
        _stem = low[:-2] if low.endswith("es") else low[:-1]
        if _stem in _COMMON_WORDS or _stem in _GAZ_LOWER:
            return None
    # Phonetic candidates (same Metaphone code).
    phon_hit = None
    if _jf is not None:
        try:
            code = _jf.metaphone(low)
        except Exception:                                         # noqa: BLE001
            code = ""
        cands = _PHONETIC_INDEX.get(code) if code else None
        if cands and len(cands) == 1:
            phon_hit = next(iter(cands))
    # Edit-distance best candidate (rapidfuzz Jaro-Winkler, else difflib).
    fuzzy_hit = None
    fuzzy_score = 0.0
    if _rf_process is not None and _rf_jw is not None:
        best = _rf_process.extractOne(
            low, _GAZ_KEYS, scorer=_rf_jw.normalized_similarity)
        if best is not None:
            fuzzy_hit = _GAZ_LOWER[best[0]]
            fuzzy_score = float(best[1])
    else:
        m = difflib.get_close_matches(low, _GAZ_KEYS, n=1, cutoff=0.84)
        if m:
            fuzzy_hit = _GAZ_LOWER[m[0]]
            fuzzy_score = 0.86
    # Decision: phonetic match confirmed by decent edit-sim, OR a very strong
    # edit-sim alone. (Frontier recipe: phonetic similarity AND edit distance.)
    # Phonetic match must be corroborated by a HIGH edit-similarity (0.88), so
    # same-Metaphone-but-different-word pairs ("set"/"site", both "ST") never
    # snap. A fuzzy-only snap needs to be near-identical (0.92).
    #
    # 2026-06-15 fix: the corroborating edit-sim must be between the input and
    # the PHONETIC CANDIDATE ITSELF -- not the GLOBAL best fuzzy score. The old
    # code let an UNRELATED close word lend its score to a distant phonetic
    # collision ("greet": metaphone KRT == "corrode", and "greet"~="green" at
    # >=0.88 -> wrongly snapped "greet"->"Corrode"). Score phon_hit directly.
    def _is_oov_superstring(cand: str) -> bool:
        # An OOV name that simply EXTENDS an agent ("omenix"->Omen, "clover"->
        # Clove, "reynard"->Reyna) is a different person, not a mishear -- the
        # agent is a strict prefix and the token has extra trailing syllables.
        # (A genuine mishear is same-length-ish, not a superstring: "jet"->Jett
        # is SHORTER, "silver"->Sova shares no prefix.)
        cl = cand.lower().replace("/", "")
        return cand in _AGENTS and len(low) > len(cl) and low.startswith(cl)

    if phon_hit is not None and not _is_oov_superstring(phon_hit):
        phon_key = phon_hit.lower().replace("/", "")
        if _rf_jw is not None:
            phon_sim = float(_rf_jw.normalized_similarity(low, phon_key))
        else:
            phon_sim = difflib.SequenceMatcher(None, low, phon_key).ratio()
        if fuzzy_hit == phon_hit or phon_sim >= 0.88:
            return phon_hit
    if (fuzzy_hit is not None and fuzzy_score >= 0.92
            and not _is_oov_superstring(fuzzy_hit)):
        return fuzzy_hit
    return None


def _fix_token(tok: str) -> str:
    low = tok.lower()
    # CONTRACTION guard (C2 FIX-C1): a token carrying an apostrophe that is NOT a
    # curated possessive mishear (sova's->Sova, handled by _MISHEARS below) is a
    # contraction (he'll/she'll/let's/we're) -- never a gazetteer term. Keep it
    # literal so it is never snapped (let's->Lotus, he'll->hell, she'll->shells).
    # No gazetteer canonical contains an apostrophe, so this is strictly safe.
    if "'" in low and low not in _MISHEARS:
        return tok
    forced = low in _MISHEAR_FORCE
    if low in _MISHEARS and (low not in _COMMON_WORDS or forced):
        return _MISHEARS[low]
    if low in _GAZ_LOWER:                         # already canonical (any group)
        # A PROTECTED verb/payload word (veto/split/dash/drift/...) that is ALSO a
        # gazetteer canonical must stay LITERAL -- the map/agent reading mangles
        # the relayed message. Gated on _PROTECT_EXTRA ONLY (not _FUZZY_BLOCK,
        # which holds clean agents that must keep canonical case); _MISHEAR_FORCE
        # still overrides.
        if low in _PROTECT_EXTRA and not forced:
            return tok
        return _GAZ_LOWER[low]
    snap = _phonetic_fuzzy_snap(low)
    if snap is not None:
        return snap
    return tok


# --- N-gram span phonetic match for 2-word gazetteer terms ------------------
# _MULTI_TERMS -> kenning.audio.routing_rules Section 1 (imported above).


# ===========================================================================
# 5b. CONTEXT SLOT confirmation -- pure-python, ~microseconds, additive.
# ===========================================================================
# An agent name sits in characteristic SLOTS: subject of a damage report
# ("<agent> hit 18"), object of one ("hit the <agent> for 18"), or after a side
# word before a state/ability verb ("their <agent> ulted"). In those slots a
# token that is a common English word but PHONETICALLY an agent is almost
# certainly the agent -- context confirms a correction the common-word guard
# would otherwise (correctly, in isolation) block. "raze hit 18" is never "raise
# hit 18"; but "raise your crosshair" / "raise the volume" have NO agent slot and
# are left untouched. This is the only place the common-word protection is
# overridden, and only when the slot grammar supplies the confidence.
_AGENT_KEY_TO_CANON = {a.lower().replace("/", ""): a for a in _AGENTS}
_AGENT_KEYS = tuple(_AGENT_KEY_TO_CANON.keys())


def _closest_agent(low: str, thresh: float = 0.82) -> "str | None":
    """The canonical agent a token most resembles, or None. Skips tokens that are
    already a known gazetteer term/ability (e.g. 'cage', 'wall') so a real ability
    word is never mistaken for an agent."""
    if low in _AGENT_KEY_TO_CANON:
        return _AGENT_KEY_TO_CANON[low]
    if low in _GAZ_LOWER:                 # a known non-agent term -> never an agent
        return None
    if _rf_process is not None and _rf_jw is not None:
        best = _rf_process.extractOne(low, _AGENT_KEYS,
                                      scorer=_rf_jw.normalized_similarity)
        if best is not None and best[1] >= thresh:
            return _AGENT_KEY_TO_CANON[best[0]]
    elif _jf is not None:                 # phonetic fallback
        try:
            code = _jf.metaphone(low)
        except Exception:                                         # noqa: BLE001
            code = ""
        for k in _AGENT_KEYS:
            try:
                if code and _jf.metaphone(k) == code:
                    return _AGENT_KEY_TO_CANON[k]
            except Exception:                                     # noqa: BLE001
                pass
    return None


_SLOT_HIT_RE = re.compile(
    r"\b([a-z]{3,})(\s+(?:hit|tagged|chunked|dinked|cracked|wiped|clipped)\s+"
    r"(?:the\s+\w+\s+(?:for\s+)?)?\d)", re.IGNORECASE)
_SLOT_HIT_OBJ_RE = re.compile(
    r"\b(hit|tagged|chunked|cracked|clipped)\s+the\s+([a-z]{3,})\b"
    r"(?=\s+(?:for\s+)?\d)", re.IGNORECASE)
_SLOT_SIDE_RE = re.compile(
    r"\b((?:their|enemy|our|the)\s+)([a-z]{3,})\b"
    r"(?=\s+(?:ulted|ulting|mollied|walled|smoked|darted|flashed|caged|stunned|"
    r"droned|recon|reviving|rez|res|is\s+(?:low|one|lit|dead)|has\s+ult))",
    re.IGNORECASE)


def _slot_agent_correct(text: str) -> str:
    def _hit(m):
        c = _closest_agent(m.group(1).lower())
        return (c if c else m.group(1)) + m.group(2)

    def _hit_obj(m):
        c = _closest_agent(m.group(2).lower())
        return m.group(1) + " the " + (c if c else m.group(2))

    def _side(m):
        c = _closest_agent(m.group(2).lower())
        return m.group(1) + (c if c else m.group(2))

    text = _SLOT_HIT_RE.sub(_hit, text)
    text = _SLOT_HIT_OBJ_RE.sub(_hit_obj, text)
    text = _SLOT_SIDE_RE.sub(_side, text)
    return text


def correct_callout_stt(text: str) -> str:
    """Snap mis-transcribed agents + tactical vocab back to canon (phrase ->
    context slot -> context rules -> token phonetic/fuzzy). Idempotent on
    already-clean callouts; negligible cost. Intended for CALLOUT-bound text only
    (the normalizer gates conversational / Spotify text out, so this never
    corrupts non-callouts)."""
    if not text:
        return text
    # Stage 0: multi-word phrase mishears (before tokenisation can split them).
    for pat, rep in _PHRASE_MISHEARS:
        text = pat.sub(rep, text)
    # Stage 1: context rules.
    for pat, rep in _CONTEXT_RULES:
        text = pat.sub(rep, text)
    # Stage 1.5: context SLOT confirmation (agent-slot common-word override).
    text = _slot_agent_correct(text)
    # Stage 2: token-level curated + phonetic + fuzzy.
    return _WORD_RE.sub(lambda m: _fix_token(m.group(0)), text)
