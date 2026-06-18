"""AGGREGATE of Kenning's NORMALIZATION + ROUTING rules, in ONE editable place.

Companion to ``voice_lines.py`` (which holds WHAT Ultron says). This file holds
HOW raw speech is cleaned and routed -- so every normalization rule and routing
knob can be reviewed/edited/added here without touching pipeline code. The
pipeline imports these names; behaviour is byte-for-byte identical to before this
file existed (proven by ``scripts/_voice_lines_verify.py`` baseline/check, run
with PYTHONHASHSEED=0). DATA only -- the matching/derivation FUNCTIONS stay in
their modules and consume these names.

THREE SECTIONS (the two normalization layers + routing):

  SECTION 1 -- NORMALIZATION LAYER 1: STT vocab correction
      consumed by kenning.audio._stt_correct.correct_callout_stt
      gazetteers (the canonical Valorant vocabulary the snapper targets) +
      single-word mishear maps (wrong-word -> right-word) + the protection sets
      that stop common English words from being corrupted.

  SECTION 2 -- NORMALIZATION LAYER 2: pre-routing command normalization
      consumed by kenning.audio.command_normalizer.normalize_command
      (added below.)

  SECTION 3 -- ROUTING / SEMANTICS
      consumed by command_router / _router_backends / _command_exemplars /
      _relay_intent.  (added below.)

OVERLAP NOTE: the agent gazetteer (``AGENTS``) is the single source of truth for
agent names; ``_stt_correct`` derives its lower-case map from it and
``voice_lines`` resolves "say hello to <agent>" through that derived map -- so
the command aggregate transitively pulls agent names from HERE (one source).

(regex, replacement) RULE TUPLES that mix compiled patterns with lambdas
(``_stt_correct._CONTEXT_RULES`` / ``_PHRASE_MISHEARS``; the command_normalizer
lead/scaffold/disfluency regexes) are NOT relocated in this first pass -- they
are documented at their site so they stay hand-tunable; relocating them needs a
behavioural (not value) diff and is a marked follow-up.
"""
from __future__ import annotations

# ============================================================================
# SECTION 1 -- NORMALIZATION LAYER 1: STT VOCAB CORRECTION
# (consumed by kenning.audio._stt_correct, which builds the phonetic index +
#  lower-case maps FROM these; edit the vocabulary / mishears / protections here)
# ============================================================================

# --- Canonical Valorant vocabulary (the snapper's targets) ------------------
AGENTS = (
    "Astra", "Breach", "Brimstone", "Chamber", "Clove", "Cypher", "Deadlock",
    "Fade", "Gekko", "Harbor", "Iso", "Jett", "KAY/O", "Killjoy", "Neon",
    "Omen", "Phoenix", "Raze", "Reyna", "Sage", "Skye", "Sova", "Viper",
    "Vyse", "Yoru", "Tejo", "Waylay", "Miks", "Veto",
)
MAPS = (
    "Ascent", "Bind", "Breeze", "Fracture", "Haven", "Icebox", "Lotus",
    "Pearl", "Split", "Sunset", "Abyss", "Corrode",
)
WEAPONS = (
    "Classic", "Shorty", "Frenzy", "Ghost", "Sheriff", "Stinger", "Spectre",
    "Bucky", "Judge", "Bulldog", "Guardian", "Phantom", "Vandal", "Marshal",
    "Outlaw", "Operator", "Ares", "Odin",
)
ABILITIES = (
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
LOCATIONS = (
    "heaven", "hell", "hookah", "garage", "market", "tree", "elbow", "ramp",
    "ramps", "connector", "window", "rafters", "generator", "pit", "link",
    "showers", "lamps", "mid", "middle", "main", "site", "long", "short",
    "spawn", "lobby", "default", "tower", "boathouse", "kitchen", "dish",
    "snowman", "yellow", "green", "tube", "tubes", "vents", "vent", "cubby",
    "stairs", "catwalk", "alley", "courtyard", "logs", "double",
)
TERMS = (
    "spike", "plant", "planted", "planting", "defuse", "defusing", "defused",
    "retake", "eco", "force", "save", "rotate", "rotating", "flank",
    "flanking", "lurk", "lurking", "peek", "peeking", "push", "pushing",
    "hold", "holding", "trade", "clutch", "entry", "anchor", "refrag",
    "wallbang", "headshot", "dink", "spray", "crosshair", "swing", "jiggle",
    "molotov", "incendiary", "defuser", "carrier", "lit", "tagged",
)

# --- Curated single-word mishears (wrong word -> right word) -----------------
AGENT_MISHEARS = {
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
TERM_MISHEARS = {
    "ultimate": "ult", "ultima": "ult", "ulta": "ult",
    "diffuse": "defuse", "diffusing": "defusing", "diffused": "defused",
    "molotov": "molly", "incendiary": "molly", "mollie": "molly", "mali": "molly",
    "trip wire": "tripwire", "nano swarm": "nanoswarm", "alarm bot": "alarmbot",
    "recon dart": "recon bolt", "owl drone": "drone",
    "meddle": "Meddle",
    "the spike": "the spike", "operator": "Operator", "op": "Operator",
}

# A curated mishear whose SOURCE token is a common English word fires ONLY when
# on this allow-list (else the common-word reading wins).
MISHEAR_FORCE = frozenset({
    "jet", "race", "sky", "ski", "silver", "euro", "que", "mix",
    "operator", "op", "ultimate", "ultima", "ulta", "meddle",
    "royal", "wise", "vice",
})

# Common English words the fuzzy snapper must NEVER corrupt into a gazetteer term.
FUZZY_BLOCK = {
    "is", "in", "it", "i", "a", "an", "the", "of", "on", "no", "so", "go",
    "to", "up", "we", "he", "me", "be", "by", "see", "say", "way", "play",
    "they", "them", "their", "there", "here", "have", "has", "had", "get",
    "got", "out", "now", "one", "two", "three", "four", "five", "six", "won",
    "our", "all", "for", "and", "but",
    "site", "main", "mid", "left", "right", "back", "push", "hold", "rotate",
    "save", "down", "dead", "kill", "team", "good", "nice", "this", "that",
    "with", "from", "just", "like", "what", "when", "where", "who", "why",
    "hey", "yes", "not", "can", "will", "you", "your",
    "set", "sit", "seat", "sight", "made", "make", "makes", "said", "send",
    "sent", "sort", "shirt", "fight", "light", "night", "side", "time",
    "line", "mine", "point", "place", "thing", "lower", "raise", "volume",
    "music", "song", "track", "turn", "put", "throw", "song", "want", "need",
    "should", "would", "could", "about", "going", "coming",
    "are", "art", "you're", "youre", "yours", "shot", "shots", "shoot",
    "sort", "war", "core", "more", "store", "wore", "ore", "her", "his",
    "greet", "greets", "greeting", "greetings",
    "hello", "hellos", "howdy", "hiya", "heya", "yo", "sup", "wassup",
    "last", "fast", "past", "blast",
    "raze", "sage", "fade", "neon", "iso", "omen", "clove", "viper", "skye",
}

# Real English / kit words consulted ONLY for the gaz-direct branch + snap guard.
PROTECT_EXTRA = frozenset({
    "veto", "flush", "dash", "smack", "plat", "lurker", "rotation", "rotates",
    "doable", "marker", "ascend", "drift", "breath", "drain", "trap", "split",
})

# Multi-word term normalizations.
MULTI_TERMS = {
    "recon dart": "recon dart", "owl drone": "drone", "alarm bot": "alarmbot",
    "nano swarm": "nanoswarm", "trip wire": "tripwire",
}
