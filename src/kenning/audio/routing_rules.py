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


# ============================================================================
# SECTION 2 -- NORMALIZATION LAYER 2: PRE-ROUTING COMMAND NORMALIZATION
# (consumed by kenning.audio.command_normalizer.normalize_command /
#  recover_relay_lead; edit the "tell my team" RECOGNITION rules here)
# ============================================================================
# The lead-recognition rules: which leading words/verbs + team-noun mean "this is
# a relay to my team". TEAM_NOUN / MANGLED_TELL / TELL_CLASS_VERB are the editable
# WORD LISTS; the *_RE regexes are built from them. To add a new STT mishear of
# "tell" (e.g. it keeps hearing "told"), add it to MANGLED_TELL. To accept a new
# team synonym, add it to TEAM_NOUN.
#
# NOTE (this pass): the rest of command_normalizer's rule groups -- scaffold/
# wrapper strip, disfluency resolver, STT phrase repairs, want/need-team veto --
# remain in command_normalizer.py (dense, fragment-interwoven; relocating them
# needs a behavioural diff). They are catalogued in command_normalizer's section
# headers; this section holds the highest-traffic, most-edited lead rules.

import re

NORM2_TEAM_NOUN = r"(?:team|teammates?|squad|boys|guys|mates|crew|fellas|homies)"
# Known mis-hears + casual variants of "tell" when a TEAM addressee follows.
# "call out", "give", "share", "drop", "ask", "relay" are real relay verbs with
# their own downstream handling and are NOT here -- only the wrong-word leads
# that otherwise leak or fall to desktop. A team noun MUST follow.
NORM2_MANGLED_TELL = (
    r"calls?|called|holds?|help|helps|helped|builds?|build|follows?|kills?|"
    r"while|how|puts?|don'?t|without|all|tale|tales|fell|filled|hail|paul|"
    r"y'?all|told|sell|tal|tel|kel|whilst|hauled|valorant|tellin'?|telling|"
    # "hope"/"hoped" observed as STT mishears of "tell" before a team addressee
    # ("hope my team nice try" == "tell my team nice try"); "I hope my team wins"
    # opens with "I" so the ^-anchored lead never fires. NOT "give" (real verb).
    r"hope|hopes|hoped"
)
# A canonical, already-correct team lead.
NORM2_TELL_CLASS_VERB = (
    r"tell|say|let|warn|inform|remind|announce|broadcast|yell|shout")

# The determiner (my/the/our) is OPTIONAL: a 2-word mishear of "tell my" often
# absorbs the determiner -- live, "tell my team to fall back" was heard as
# "Valorant team to fall back" (no "my"), so the lead went unrecognized and the
# whole phrase relayed literally ("Valorant team to fall back."). The mishear set
# is curated to words that MEAN "tell", so "<tell-mishear> team X" == "tell my
# team X" with or without the determiner.
NORM2_MANGLED_TEAM_LEAD_RE = re.compile(
    rf"^\s*(?:{NORM2_MANGLED_TELL})\s+(?:(?:my|the|a|our)\s+)?{NORM2_TEAM_NOUN}\b[\s,:.]*",
    re.IGNORECASE,
)
NORM2_IRREGULAR_TEAM_LEAD_RE = re.compile(
    rf"^\s*(?:"
    # NB: the first-person PAST "I (just/already) told my team ..." branch was
    # REMOVED (2026-06-18 corpus audit F5): it canonicalized recounts ("I told
    # my team to save and watched them buy rifles", "I told my squad to rotate
    # but they stayed A") into a LIVE relay. A past first-person recount is
    # narration, not a relay -- now left for the narration/musing gate to keep
    # conversational. The bare STT-mishear "told my team X" (no "I") is still
    # handled as a relay by NORM2_MANGLED_TELL ("told" is a mishear of "tell").
    rf"(?:that'?s|this\s+is)\s+(?:the\s+)?team(?:\s+that)?"
    rf")\b[\s,:.]*",
    re.IGNORECASE,
)
NORM2_TELL_TEAM_LEAD_RE = re.compile(
    rf"^\s*(?:please\s+)?(?:{NORM2_TELL_CLASS_VERB})\s+(?:to\s+)?"
    rf"(?:my\s+|our\s+|the\s+)?{NORM2_TEAM_NOUN}\b(?:\s+know)?[\s,:.]*",
    re.IGNORECASE,
)


# ============================================================================
# SECTION 3 -- ROUTING / SEMANTICS
# (the semantic command router commits an utterance to a FAMILY when the top
#  family clears a threshold AND beats the runner-up by a margin AND isn't the
#  conversational anchor)
# ============================================================================
# EDITABLE TUNING KNOBS (relocated here from command_router.py, where they were
# hardcoded). Lower a family's threshold to make it commit more eagerly; raise it
# to require more confidence. The margin is the min lead over the runner-up.
ROUTE_DEFAULT_THRESHOLD = 0.50     # min top-family score to commit
ROUTE_DEFAULT_MARGIN = 0.06        # min (top - runner_up) to commit
ROUTE_FAMILY_THRESHOLDS = {        # per-family overrides of the default
    "identity": 0.55,
    "spotify": 0.50,
    "team_callout": 0.48,
    "desktop_refuse": 0.50,
}

# The ROUTING EXEMPLARS (phrases that define each family) + the relay-intent gate
# exemplars are large, already-clean DATA modules; to keep this aggregate's import
# graph minimal (it loads very early via _stt_correct) they are NOT imported here.
# Edit them in their dedicated files -- this map is the index:
#   * family -> exemplar phrases ........ kenning/audio/_command_exemplars.py
#       (FAMILIES, ABSTAIN_FAMILIES, DETERMINISTIC_FAMILIES). To ADD a routable
#        command family or grow one, add exemplar phrases there.
#   * relay-intent gate exemplars ....... kenning/audio/_relay_intent.py
#       (RELAY_POSITIVE_EXEMPLARS / RELAY_NEGATIVE_EXEMPLARS + gate threshold).
