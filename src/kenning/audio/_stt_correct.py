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


# ===========================================================================
# 1. DOMAIN GAZETTEER (the bias list)
# ===========================================================================
# --- Agents (canonical display incl. the KAY/O slash) ----------------------
_AGENTS = (
    "Astra", "Breach", "Brimstone", "Chamber", "Clove", "Cypher", "Deadlock",
    "Fade", "Gekko", "Harbor", "Iso", "Jett", "KAY/O", "Killjoy", "Neon",
    "Omen", "Phoenix", "Raze", "Reyna", "Sage", "Skye", "Sova", "Viper",
    "Vyse", "Yoru", "Tejo", "Waylay", "Miks", "Veto",
)
# --- Maps -------------------------------------------------------------------
_MAPS = (
    "Ascent", "Bind", "Breeze", "Fracture", "Haven", "Icebox", "Lotus",
    "Pearl", "Split", "Sunset", "Abyss", "Corrode",
)
# --- Weapons ----------------------------------------------------------------
_WEAPONS = (
    "Classic", "Shorty", "Frenzy", "Ghost", "Sheriff", "Stinger", "Spectre",
    "Bucky", "Judge", "Bulldog", "Guardian", "Phantom", "Vandal", "Marshal",
    "Outlaw", "Operator", "Ares", "Odin",
)
# --- Single-word abilities / utility callout terms --------------------------
_ABILITIES = (
    "turret", "alarmbot", "nanoswarm", "lockdown", "drone", "dart", "smoke",
    "smokes", "flash", "flashes", "molly", "wall", "cage", "cages", "trip",
    "tripwire", "stun", "slow", "recon", "knife", "blade", "orb", "beacon",
    "ult", "ulted", "ulting", "ultimate", "gatecrash", "showstopper",
    "rolling", "thunder", "blade", "paint", "shells", "satchel", "boombot",
    "blast", "pack", "snowball", "tailwind", "updraft", "cloudburst",
    "barrier", "slow", "heal", "resurrect", "rez", "fragment", "seize",
    "wingman", "trailblazer", "dizzy", "haunt", "shroud", "trademark",
    "cove", "headhunter", "tour", "rendezvous", "prowler", "nightfall",
    "paranoia", "blindside", "fault", "aftershock", "rift",
)
# --- Single-word callout locations (cross-map) ------------------------------
_LOCATIONS = (
    "heaven", "hell", "hookah", "garage", "market", "tree", "elbow", "ramp",
    "ramps", "connector", "window", "rafters", "generator", "pit", "link",
    "showers", "lamps", "mid", "middle", "main", "site", "long", "short",
    "spawn", "lobby", "default", "tower", "boathouse", "kitchen", "dish",
    "snowman", "yellow", "green", "tube", "tubes", "vents", "vent", "cubby",
    "stairs", "catwalk", "alley", "courtyard", "logs", "double",
)
# --- Tactical verbs / nouns (callout vocabulary) ----------------------------
_TERMS = (
    "spike", "plant", "planted", "planting", "defuse", "defusing", "defused",
    "retake", "eco", "force", "save", "rotate", "rotating", "flank",
    "flanking", "lurk", "lurking", "peek", "peeking", "push", "pushing",
    "hold", "holding", "trade", "clutch", "entry", "anchor", "refrag",
    "wallbang", "headshot", "dink", "spray", "crosshair", "swing", "jiggle",
    "molotov", "incendiary", "defuser", "carrier", "lit", "tagged",
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
_ULTISH = r"(?:old|sold|alt|oat|halt|vault|ault|ulta|ulte|ulti|olt|ault|aunt)"
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


# ===========================================================================
# 3a. CURATED single-word mishears (genuine wrong words) ---------------------
# ===========================================================================
_AGENT_MISHEARS = {
    "silva": "Sova", "selva": "Sova", "sofa": "Sova", "soda": "Sova",
    "soever": "Sova", "sovereign": "Sova", "saba": "Sova", "sova's": "Sova",
    "sobi": "Sova", "sovah": "Sova", "soba": "Sova", "silver": "Sova",
    "silvers": "Sova",
    "royal": "Reyna", "raina": "Reyna", "rayna": "Reyna", "reina": "Reyna",
    "rena": "Reyna", "regina": "Reyna", "reyna's": "Reyna", "ray nuh": "Reyna",
    "jet": "Jett", "jed": "Jett", "jett's": "Jett", "jette": "Jett",
    "yet": "Jett", "jett": "Jett",
    "cipher": "Cypher", "sypher": "Cypher", "cyphus": "Cypher", "sifa": "Cypher",
    "sina": "Cypher", "saifer": "Cypher",
    "kayo": "KAY/O", "cale": "KAY/O", "kayle": "KAY/O", "kio": "KAY/O",
    "kayoh": "KAY/O", "kay-o": "KAY/O", "que": "KAY/O", "kayode": "KAY/O",
    "oman": "Omen", "omens": "Omen", "omar": "Omen",
    "vyper": "Viper", "wiper": "Viper", "viper's": "Viper", "hyper": "Viper",
    "race": "Raze", "rays": "Raze", "raisa": "Raze",
    "raz": "Raze", "raze's": "Raze", "rage": "Raze",
    "felix": "Phoenix", "phoenix's": "Phoenix", "fenix": "Phoenix",
    "venix": "Phoenix", "phoenixs": "Phoenix",
    "gecko": "Gekko", "geko": "Gekko", "gekko's": "Gekko", "geco": "Gekko",
    "breech": "Breach", "breach's": "Breach",
    "fate": "Fade", "faded": "Fade", "fad": "Fade", "fage": "Fade",
    "sky": "Skye", "ski": "Skye", "skype": "Skye", "sky's": "Skye",
    "aster": "Astra", "astro": "Astra", "astra's": "Astra",
    "arbor": "Harbor", "harbour": "Harbor", "harper": "Harbor", "arbour": "Harbor",
    "clive": "Clove", "clove's": "Clove", "clo": "Clove", "claw": "Clove",
    "cloves": "Clove", "clive's": "Clove",
    "brimston": "Brimstone", "grimstone": "Brimstone", "brimstine": "Brimstone",
    "brim": "Brimstone", "brimstone's": "Brimstone", "brimstown": "Brimstone",
    "deadlocke": "Deadlock", "dead lock": "Deadlock", "deadbolt": "Deadlock",
    "deadshot": "Deadlock", "deadlog": "Deadlock",
    "neon's": "Neon", "nian": "Neon", "neyon": "Neon", "leon": "Neon",
    "euro": "Yoru", "yoda": "Yoru", "yoru's": "Yoru", "yo-ru": "Yoru",
    "yoroo": "Yoru", "yору": "Yoru",
    "kill joy": "Killjoy", "killjoy's": "Killjoy", "kiljoy": "Killjoy",
    "kj": "Killjoy", "kill-joy": "Killjoy",
    "chambers": "Chamber", "chamber's": "Chamber", "jaber": "Chamber",
    "teacho": "Tejo", "taho": "Tejo", "tejo's": "Tejo", "techno": "Tejo",
    "way lay": "Waylay", "weigh lay": "Waylay", "waylaid": "Waylay",
    "mix": "Miks", "meeks": "Miks", "mics": "Miks",
    "vice": "Vyse", "vise": "Vyse", "vis": "Vyse", "vyce": "Vyse", "wise": "Vyse",
    "veto's": "Veto", "vito": "Veto",
    "iso's": "Iso", "isa": "Iso", "ito": "Iso",
}

# --- Unambiguous Valorant TERM lexicon (single-word) ------------------------
_TERM_MISHEARS = {
    "ultimate": "ult", "ultima": "ult", "ulta": "ult",
    "diffuse": "defuse", "diffusing": "defusing", "diffused": "defused",
    "molotov": "molly", "incendiary": "molly", "mollie": "molly", "mali": "molly",
    "trip wire": "tripwire", "nano swarm": "nanoswarm", "alarm bot": "alarmbot",
    "recon dart": "recon dart", "owl drone": "drone",
    "the spike": "the spike", "operator": "Operator", "op": "Operator",
}

_MISHEARS = {**_AGENT_MISHEARS, **_TERM_MISHEARS}


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
    (re.compile(r"\brecon\s+(?:dart|bolt)\b", re.I), "recon dart"),
    (re.compile(r"\bowl\s+drone\b", re.I), "drone"),
    (re.compile(r"\bnano\s*swarm\b", re.I), "nanoswarm"),
    (re.compile(r"\balarm\s*bot\b", re.I), "alarmbot"),
    (re.compile(r"\btrip\s*wire\b", re.I), "tripwire"),
)


# ===========================================================================
# 4. Common English close to a short agent name -> curated/phonetic only -----
# ===========================================================================
_FUZZY_BLOCK = {
    "is", "in", "it", "i", "a", "an", "the", "of", "on", "no", "so", "go",
    "to", "up", "we", "he", "me", "be", "by", "see", "say", "way", "play",
    "they", "them", "their", "there", "here", "have", "has", "had", "get",
    "got", "out", "now", "one", "two", "three", "four", "five", "six", "won",
    "our", "all", "for", "and", "but",
    "site", "main", "mid", "left", "right", "back", "push", "hold", "rotate",
    "save", "down", "dead", "kill", "team", "good", "nice", "this", "that",
    "with", "from", "just", "like", "what", "when", "where", "who", "why",
    "hey", "yes", "not", "can", "will", "you", "your",
    # common words that phonetically collide with gazetteer terms (never snap)
    "set", "sit", "seat", "sight", "made", "make", "makes", "said", "send",
    "sent", "sort", "shirt", "fight", "light", "night", "side", "time",
    "line", "mine", "point", "place", "thing", "lower", "raise", "volume",
    "music", "song", "track", "turn", "put", "throw", "song", "want", "need",
    "should", "would", "could", "about", "going", "coming",
    # common words that JW snapped onto gazetteer terms in live logs:
    #   are->Ares(weapon), you're->Yoru(agent), shot->short(loc),
    #   greet->Corrode(map, both metaphone "KRT")
    "are", "art", "you're", "youre", "yours", "shot", "shots", "shoot",
    "sort", "war", "core", "more", "store", "wore", "ore", "her", "his",
    "greet", "greets", "greeting", "greetings",
    # greetings that phonetically collide with locations (hello/hellos metaphone
    # "HL" == "hell") -- a greeting must never snap onto a callout location.
    "hello", "hellos", "howdy", "hiya", "heya", "yo", "sup", "wassup",
    "last", "fast", "past", "blast",   # "last guy"->"blast" (last is a callout)
    # short agent names that ARE common words -> curated/phonetic only
    "raze", "sage", "fade", "neon", "iso", "omen", "clove", "viper", "skye",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'/]*")


# ===========================================================================
# 5. Token-level fix: curated -> canonical -> phonetic+fuzzy snap
# ===========================================================================
def _phonetic_fuzzy_snap(low: str) -> str | None:
    """Snap a token onto the gazetteer by Metaphone + Jaro-Winkler. Conservative:
    requires a phonetic-code match OR a very high edit-similarity, so real words
    are not corrupted. Returns the canonical form or None."""
    if len(low) < 3 or low in _FUZZY_BLOCK:
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
    if phon_hit is not None:
        phon_key = phon_hit.lower().replace("/", "")
        if _rf_jw is not None:
            phon_sim = float(_rf_jw.normalized_similarity(low, phon_key))
        else:
            phon_sim = difflib.SequenceMatcher(None, low, phon_key).ratio()
        if fuzzy_hit == phon_hit or phon_sim >= 0.88:
            return phon_hit
    if fuzzy_hit is not None and fuzzy_score >= 0.92:
        return fuzzy_hit
    return None


def _fix_token(tok: str) -> str:
    low = tok.lower()
    if low in _MISHEARS:
        return _MISHEARS[low]
    if low in _GAZ_LOWER:                         # already canonical (any group)
        return _GAZ_LOWER[low]
    snap = _phonetic_fuzzy_snap(low)
    if snap is not None:
        return snap
    return tok


# --- N-gram span phonetic match for 2-word gazetteer terms ------------------
_MULTI_TERMS = {
    "recon dart": "recon dart", "owl drone": "drone", "alarm bot": "alarmbot",
    "nano swarm": "nanoswarm", "trip wire": "tripwire",
}


def correct_callout_stt(text: str) -> str:
    """Snap mis-transcribed agents + tactical vocab back to canon (phrase ->
    context -> token phonetic/fuzzy). Idempotent on already-clean callouts;
    negligible cost. Intended for CALLOUT-bound text only (the normalizer gates
    conversational / Spotify text out, so this never corrupts non-callouts)."""
    if not text:
        return text
    # Stage 0: multi-word phrase mishears (before tokenisation can split them).
    for pat, rep in _PHRASE_MISHEARS:
        text = pat.sub(rep, text)
    # Stage 1: context rules.
    for pat, rep in _CONTEXT_RULES:
        text = pat.sub(rep, text)
    # Stage 2: token-level curated + phonetic + fuzzy.
    return _WORD_RE.sub(lambda m: _fix_token(m.group(0)), text)
