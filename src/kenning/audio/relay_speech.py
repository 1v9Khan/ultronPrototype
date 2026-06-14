"""Voice relay: speak a message to OTHER PEOPLE on a secondary output.

"Kenning, tell my teammates they should be smoking mid window" should not
be answered conversationally -- it is an instruction to DELIVER a spoken
line to the user's teammates. This module gives the orchestrator that
capability:

1. :func:`match_relay_command` -- a STRICT matcher (same philosophy as
   the run/scrap/deep-research short-circuits: ordinary utterances must
   never trip it) that recognises "tell my teammates X" / "say X to my
   team" / "ask my team for X" / "tell them X" and extracts the message
   payload.
2. :func:`build_relay_line` -- converts the reported-speech payload into
   a line Kenning speaks DIRECTLY to the teammates (second person,
   one-to-two short sentences), via a small LLM rephrase. Fail-open: any
   LLM problem falls back to a deterministic "Team: <payload>" line.
3. :func:`resolve_relay_device` / :func:`play_to_device` -- play the
   synthesized PCM on a SEPARATE PortAudio output device (typically a
   VoiceMeeter virtual input such as "Voicemeeter Aux Input" whose strip
   is routed to the same B-bus as the user's microphone), so the line is
   transmitted into the game's voice chat instead of -- or as well as --
   the user's own headphones.

The normal TTS hot path is untouched: synthesis still happens on the
session's existing Kokoro engine; only the PLAYBACK target differs, on a
stream this module opens and closes per relay line. Everything here is
fail-open -- a missing device, a failed synth, or a failed rephrase must
never crash the orchestrator turn.
"""

from __future__ import annotations

import functools
import logging
import re
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

logger = logging.getLogger("kenning.audio.relay_speech")

__all__ = [
    "DEFAULT_ADDRESSEE_NAMES",
    "DEFAULT_ROAST_LINES",
    "DEFAULT_FUN_FACTS",
    "RelayCommand",
    "RelayPlaybackResult",
    "match_relay_command",
    "match_relay_toggle",
    "build_relay_line",
    "load_roast_lines",
    "load_fun_facts",
    "pick_roast_line",
    "pick_line",
    "resolve_relay_device",
    "play_to_device",
]

# Maximum characters of the final spoken relay line (a voice-chat line
# should be one breath, not a paragraph; also bounds synth time).
MAX_RELAY_LINE_CHARS = 360

# Words that may address a group of teammates. Deliberately NARROW:
# "tell me ..." and a bare "tell her ..." must never match ("tell
# him/her" is only honoured INSIDE a reported-speech context clause --
# see the context+directive forms below). "teams" tolerated (observed
# live: STT rendered "teammates" as "teams").
_GROUP_WORDS = (
    r"(?:team\s?mates?|teams?|squad|lobby|party|group|boys|the\s+boys"
    r"|crew|stack|fellas|guys|duo)"
)

# A possessive group reference: "my team" / "our teammates" / "the
# whole squad". Every group pattern requires the possessive so bare
# nouns in ordinary speech never trip the matcher.
_GROUP = rf"(?:my|our|the)\s+(?:whole\s+|entire\s+)?{_GROUP_WORDS}"

# Bare pronoun group references that, in a voice-chat session AFTER the wake
# word, clearly mean the team: "tell 'em X", "say to the guys X". "them" and
# "everyone" were already honoured; "em"/"'em" (the spoken contraction) +
# "the guys/the others/everybody" are the high-frequency live phrasings that
# were silently falling through to the conversational pipeline. NOT "me"/bare
# "him"/"her" -- those are excluded by construction.
_GROUP_PRON = (
    r"(?:them|'?em|everyone|everybody|the\s+guys|the\s+others|the\s+lobby|chat)"
)
# A chat/voice channel the user can name when relaying: "say in game chat X",
# "say in voice X", "say in the team chat X", "say in my team chat X".
_CHANNEL = (
    r"(?:the\s+|my\s+)?(?:team|game|voice|all|in[\s-]?game)\s*(?:chat)?"
)
# The ENEMY as an addressee for bravado/trash-talk the streamer wants spoken
# ("let the enemy know they are washed", "tell the other team gg"). Ultron can
# only voice it into team comms, but it is still a relay the streamer intends.
_ENEMY_GROUP = (
    r"(?:the\s+)?(?:enemy(?:\s+team)?|enemies|other\s+team|other\s+side|"
    r"opps|opponents|enemy\s+side)"
)

# STT artifact normalisation: the wake word occasionally leaves a
# leading "One," / "1." fragment on the transcript ("One, tell my
# teammate to drop me a vandal."). Strip ONLY when followed by a relay
# verb so normal numbered dictation is untouched.
_LEADING_ARTIFACT = re.compile(
    r"^\s*(?:one|1|2)\s*[.,:]\s+"
    r"(?=(?:please\s+)?(?:tell|say|ask|let|remind|warn|inform|wish|call"
    r"|give|encourage|hype|roast|flame)\b)",
    re.IGNORECASE,
)

# Strict relay patterns. Each captures the message payload; the
# addressee is normalised to "team" wording for the rephrase prompt.
_RELAY_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "tell my teammates (that|to) X" / "tell our team X"
    re.compile(
        rf"^(?:please\s+)?tell\s+{_GROUP}\s+"
        rf"(?:that\s+|to\s+)?(?P<payload>.+)$",
        re.IGNORECASE,
    ),
    # "let my team know (that) X"
    re.compile(
        rf"^(?:please\s+)?let\s+{_GROUP}\s+know\s+"
        rf"(?:that\s+)?(?P<payload>.+)$",
        re.IGNORECASE,
    ),
    # "remind/warn/inform my team (that|to|about) X"
    re.compile(
        rf"^(?:please\s+)?(?:remind|warn|inform)\s+{_GROUP}\s+"
        rf"(?:that\s+|to\s+|about\s+)?(?P<payload>.+)$",
        re.IGNORECASE,
    ),
    # "wish my team good luck" -- relays the wish itself.
    re.compile(
        rf"^(?:please\s+)?wish\s+{_GROUP}\s+(?P<payload>.+)$",
        re.IGNORECASE,
    ),
    # "say X to my teammates" / "say X in (the) (team|game) chat"
    re.compile(
        rf"^(?:please\s+)?say\s+(?P<payload>.+?)\s+(?:to\s+(?:{_GROUP}|{_GROUP_PRON})"
        rf"|in\s+{_CHANNEL})\s*[.!?]?$",
        re.IGNORECASE,
    ),
    # "ask my teammates (to|for|if|whether|why|...) X" -- question
    # words kept in the payload so questions relay as questions.
    re.compile(
        rf"^(?:please\s+)?ask\s+{_GROUP}\s+"
        rf"(?P<payload>(?:to|for|if|whether|why|how|what|when|where|who)"
        rf"\s+.+)$",
        re.IGNORECASE,
    ),
    # "tell them/'em/everyone/the guys (that|to) X" -- in a voice-chat session
    # these address the team; "tell me ..." does not match by construction, and
    # a bare "tell him/her ..." is only honoured in the context+directive forms.
    re.compile(
        rf"^(?:please\s+)?tell\s+{_GROUP_PRON}\s+"
        rf"(?:that\s+|to\s+)?(?P<payload>.+)$",
        re.IGNORECASE,
    ),
    # "let 'em/them know (that) X" / "warn 'em X" -- pronoun-group variants of
    # the let/remind/warn/inform forms above.
    re.compile(
        rf"^(?:please\s+)?(?:let\s+{_GROUP_PRON}\s+know|"
        rf"(?:remind|warn|inform)\s+{_GROUP_PRON})\s+"
        rf"(?:that\s+|to\s+|about\s+)?(?P<payload>.+)$",
        re.IGNORECASE,
    ),
    # "say to my team X" / "say to the guys X" / "say in game chat (that) X" /
    # "say in voice X" -- the message PAYLOAD comes AFTER the addressee/channel
    # (the existing "say X to my team" handles payload-first).
    re.compile(
        rf"^(?:please\s+)?say\s+(?:to\s+(?:{_GROUP}|{_GROUP_PRON})"
        rf"|in\s+{_CHANNEL})\s+(?:that\s+)?(?P<payload>.+)$",
        re.IGNORECASE,
    ),
    # "call out (that) X" -- gamer shorthand for a team info callout.
    re.compile(
        r"^(?:please\s+)?call\s+out\s+(?:that\s+)?(?P<payload>.+)$",
        re.IGNORECASE,
    ),
    # "relay to my team(:) X" / "relay to the squad that X" / "relay to em X" --
    # explicit (CLOSED) group addressee.
    re.compile(
        rf"^(?:please\s+)?relay\s+to\s+(?:{_GROUP}|{_GROUP_PRON})\s*[:,]?\s*"
        rf"(?:that\s+)?(?P<payload>.+)$",
        re.IGNORECASE,
    ),
    # "relay X" / "relay: X" / "relay that X" -- BARE (no addressee). The
    # negative lookahead rejects "relay to <name>" so an out-of-roster name
    # ("relay to Jordan ...") never leaks into the payload (closed vocab).
    re.compile(
        r"^(?:please\s+)?relay\s*[:,]?\s+(?!to\s)(?:that\s+)?(?P<payload>.+)$",
        re.IGNORECASE,
    ),
    # ENEMY-addressed bravado: "let the enemy know X" / "tell the other team X" /
    # "say to the enemy X" -- relayed (Ultron voices it into team comms).
    re.compile(
        rf"^(?:please\s+)?(?:tell|let|warn|inform|remind|say\s+to)\s+"
        rf"{_ENEMY_GROUP}\s+(?:know\s+)?(?:that\s+|to\s+)?(?P<payload>.+)$",
        re.IGNORECASE,
    ),
    # "ask if/whether/who/whoever X" -- a question relayed to the team with no
    # explicit group ("ask if anyone needs a rifle", "ask whoever is IGLing X").
    re.compile(
        r"^(?:please\s+)?ask\s+(?P<payload>(?:if|whether|who|whoever|when|"
        r"anyone|everyone|somebody|someone)\b\s+.+)$",
        re.IGNORECASE,
    ),
)

# BARE "say X" -- relay to team (implicit addressee). Applied as a LAST RESORT in
# match_relay_command (after the named "say X to Clove" form) and requires >=2
# payload words so a bare "say hello" / "say what" never trips it.
_BARE_SAY_RE = re.compile(
    r"^(?:please\s+)?say\s+(?:that\s+)?(?P<payload>\S+\s+.+)$", re.IGNORECASE,
)

# Composition requests: the user asks Kenning to AUTHOR a line rather
# than relay a literal message ("give my team some encouragement").
# Maps to a composition TOPIC the rephrase prompt expands.
_COMPOSE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"^(?:please\s+)?give\s+{_GROUP}\s+"
        rf"(?:some\s+)?(?:encouragement|hype|a\s+pep\s+talk|a\s+morale\s+boost)"
        rf"\s*[.!?]?$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^(?:please\s+)?(?:encourage|hype\s+up)\s+{_GROUP}"
        rf"\s*[.!?]?$",
        re.IGNORECASE,
    ),
)

# Roast requests: spoken VERBATIM from the user-curated lines file
# (never LLM-authored) -- "Kenning, roast my team".
_ROAST_RE = re.compile(
    rf"^(?:please\s+)?(?:roast|flame)\s+(?:{_GROUP}|them|everyone"
    rf"|the\s+lobby|chat)\s*[.!?]?$",
    re.IGNORECASE,
)

# Fun-fact requests -- spoken VERBATIM from the user's fun-fact corpus.
# "tell/give/share my team a fun fact" / "drop a fun fact in chat".
_FUN_FACT_RE = re.compile(
    rf"^(?:please\s+)?(?:tell|give|share|drop|hit)\s+"
    rf"(?:{_GROUP}|them|everyone|the\s+lobby|chat)?\s*"
    rf"(?:with\s+)?(?:a|an|some)\s+"
    rf"(?:fun|interesting|random|cool|true)?\s*fact"
    rf"(?:s)?(?:\s+(?:to|for|in|with)\s+(?:{_GROUP}|them|chat))?"
    rf"\s*[.!?]?$",
    re.IGNORECASE,
)

# Greeting requests -- a curated Ultron TEAM INTRO at agent select / round
# one ("greet my team", "introduce yourself to my team", "say hi to the
# squad and introduce yourself"). Addresses the whole team, names himself
# Ultron, and assures victory so long as they comply.
_GREET_RE = re.compile(
    rf"^(?:please\s+)?(?:.{{0,50}}?[,;.]\s+)?(?:"
    rf"greet\s+(?:all\s+(?:of\s+)?)?{_GROUP}"
    rf"|introduce\s+(?:yourself|us)(?:\s+to\s+(?:all\s+(?:of\s+)?)?{_GROUP})?"
    rf"|say\s+(?:hi|hello|hey|what'?s\s+up)\s+to\s+(?:all\s+(?:of\s+)?)?{_GROUP}"
    rf"(?:\s+and\s+introduce\s+yourself)?"
    rf"|tell\s+{_GROUP}\s+who\s+you\s+are"
    rf")\s*[.!?]*$",
    re.IGNORECASE,
)

# Farewell / closing requests -- a curated Ultron sign-off at match end
# ("say bye to my team", "tell my team gg, we won", "say goodbye, we lost").
# The trailing clause is kept so win/loss can pick the register.
_FAREWELL_RE = re.compile(
    rf"^(?:please\s+)?(?:.{{0,50}}?[,;.]\s+)?(?:"
    rf"say\s+(?:bye|goodbye|good\s*bye|gg|good\s+game|ggeorge)\s+to\s+{_GROUP}"
    rf"|tell\s+{_GROUP}\s+(?:bye|goodbye|good\s*bye|gg|good\s+game)"
    rf"|(?:give|do)\s+(?:{_GROUP}\s+)?(?:a\s+)?"
    rf"(?:closing|closer|sign[\s-]?off|farewell|goodbye|send[\s-]?off)"
    rf"(?:\s+(?:statement|line|speech))?"
    rf"|(?:close|wrap)\s+(?:it|us|this)\s+(?:out|up)"
    rf")\b.*$",
    re.IGNORECASE,
)

# Win / loss signal inside a farewell command -> chooses victory vs defeat
# closing register. Absent -> a neutral Ultron sign-off.
_WIN_RE = re.compile(
    r"\b(?:we\s+won|won\s+(?:the|that|this)|gg\s+we\s+won|we\s+win"
    r"|victor(?:y|ious)|took\s+(?:the|that|this)\s+(?:game|match|one)"
    r"|we\s+got\s+(?:the\s+win|that|this)|stomped\s+them|destroyed\s+them"
    r"|rolled\s+them|smashed\s+them|clean\s+sweep|swept\s+them|gg\s+ez)\b",
    re.IGNORECASE,
)
_LOSS_RE = re.compile(
    r"\b(?:we\s+lost|lost\s+(?:the|that|this)|we\s+lose|we\s+got\s+"
    r"(?:destroyed|rolled|stomped|smashed|clapped|diffed|cooked)"
    r"|got\s+(?:destroyed|rolled|stomped|diffed)|defeat(?:ed)?"
    r"|threw\s+(?:the|that|it)|we\s+choked|blew\s+(?:the|that|it))\b",
    re.IGNORECASE,
)


def _farewell_directive(text: str) -> str:
    """Pick the closing register from a farewell command's win/loss signal."""
    if _WIN_RE.search(text):
        return "farewell_win"
    if _LOSS_RE.search(text):
        return "farewell_loss"
    return "farewell"


# Verbatim demand: "..., in those words specifically" / "word for
# word" / "exactly like that" / "verbatim". When present the payload
# is relayed AS-IS with no LLM rephrase. Captured as a TRAILING clause
# and stripped from the payload before speaking.
_VERBATIM_SUFFIX_RE = re.compile(
    r"[,;.]?\s*(?:and\s+)?(?:say\s+(?:it|that)\s+)?"
    r"(?:in\s+(?:those|these|exactly\s+those|the\s+exact)\s+words"
    r"(?:\s+(?:specifically|exactly))?"
    r"|word\s+for\s+word"
    r"|verbatim"
    r"|(?:exactly|just)\s+like\s+(?:that|so)"
    r"|exactly\s+(?:like\s+)?(?:that|how\s+i\s+said\s+it)?"
    r"|in\s+my\s+(?:exact\s+)?words)"
    r"\s*[.!?]*$",
    re.IGNORECASE,
)

# Spoken-form normalisation applied before matching. (1) Collapse the KAY/O
# slash ("kay/o", "k/o", "kay / o" -> "kayo") so the agent name tokenises and
# the named-addressee patterns can match ("ask KAY/O to flash"). (2) Drop
# standalone speech filler ("uh", "um", "er", "hmm") that real transcripts
# sprinkle mid-utterance -- it otherwise wedges between a trigger and its
# payload ("let my team know, uh, two enemies at B" never reached the payload).
_KAYO_SLASH_RE = re.compile(r"\bk(?:ay)?\s*/\s*o\b", re.IGNORECASE)
# Eat a standalone filler token AND the surrounding commas/whitespace so a
# trigger isn't left stranded behind a comma: "know, uh, two" -> "know two".
_FILLER_RE = re.compile(
    r"[\s,]*\b(?:uh+|um+|er+|erm|hmm)\b[\s,]*",
    re.IGNORECASE,
)


# Common agent abbreviations / STT homophones -> the canonical agent name so the
# matcher + fact-extractor resolve them ("KJ nanoswarm" -> Killjoy; "yoroo
# gatecrash" -> Yoru). Word-boundary anchored so they never hit inside a word.
_ABBREV_SUBS = (
    (re.compile(r"\bkj\b", re.IGNORECASE), "Killjoy"),
    (re.compile(r"\bbrim\b", re.IGNORECASE), "Brimstone"),
    (re.compile(r"\byoroo\b", re.IGNORECASE), "Yoru"),
    (re.compile(r"\bvyce\b", re.IGNORECASE), "Vyse"),
)


def _normalize_speech(text: str) -> str:
    """Light spoken-form cleanup before relay matching (KAY/O slash, filler,
    agent abbreviations/homophones)."""
    text = _KAYO_SLASH_RE.sub("kayo", text)
    text = _FILLER_RE.sub(" ", text)
    for rx, sub in _ABBREV_SUBS:
        text = rx.sub(sub, text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def _strip_verbatim_suffix(payload: str) -> tuple[str, bool]:
    """Split a trailing verbatim demand off a payload.

    Returns ``(payload_without_suffix, is_verbatim)``. A payload that is
    ONLY the verbatim demand (nothing left) is treated as non-verbatim
    so it doesn't silently relay an empty line.
    """
    m = _VERBATIM_SUFFIX_RE.search(payload)
    if m is None:
        return payload, False
    stripped = payload[: m.start()].strip().strip(",;.").strip()
    if not stripped:
        return payload, False
    return stripped, True

# --- Context + directive forms ---------------------------------------
# "Reyna just asked if you are an AI, respond" / "Jett is flaming me,
# respond and calm him down" / "my teammate wants to go A, tell him we
# shouldn't go A on anti eco". Shape: <reported-speech context>,
# <directive>. BOTH halves must independently match (a conjunctive
# gate) so ordinary dictation never trips it: the context clause must
# contain a reported-speech verb AND be at least three words, and the
# tail must be drawn from a CLOSED directive vocabulary. This is also
# the ONLY place "tell him/her" is honoured -- standalone "tell her I
# said hi" still never relays.
_CONTEXT_VERB_RE = re.compile(
    r"\b(?:asked|asking|asks|said|saying|says|told|wants?|wanted"
    r"|wondering|wonders|thinks?|thinking|typed|wrote"
    r"|complain(?:ed|ing|s)?|crying|flam(?:ing|ed|es)|tilted|raging"
    r"|malding|trash[\s-]?talk(?:ing|ed)?|talking\s+(?:trash|smack)"
    r"|accus(?:ed|ing)|claim(?:s|ed|ing)|suggest(?:s|ed|ing)|begging"
    r"|request(?:s|ed|ing)|call(?:ed|ing)\s+(?:me|you|us)"
    # banter directed AT Ultron -- so "reyna is making fun of you, respond"
    r"|mak(?:ing|es|e)\s+fun|mock(?:s|ed|ing)?|teas(?:e|es|ed|ing)"
    r"|roast(?:s|ed|ing)?|clown(?:s|ed|ing)?|diss(?:es|ed|ing)?"
    r"|ridicul(?:e|es|ed|ing)|laugh(?:s|ed|ing)\s+at|insult(?:s|ed|ing)?"
    r"|bully(?:ing|ied)?|ragging|making\s+fun"
    r"|mad|upset|angry|heated)\b",
    re.IGNORECASE,
)

# One directive verb phrase. Closed set; extend deliberately.
_DIRECTIVE_ATOM = (
    r"(?:"
    r"respond(?:\s+to\s+(?:him|her|them|it|that))?"
    r"|reply(?:\s+to\s+(?:him|her|them))?"
    r"|answer(?:\s+(?:him|her|them))?"
    r"|acknowledge"
    r"|agree(?:\s+with\s+(?:him|her|them))?"
    r"|calm\s+(?:him|her|them)\s+down"
    r"|de[\s-]?escalate"
    r"|say\s+something(?:\s+(?:nice|back|funny|cool))?"
    r"|handle\s+(?:it|that|him|her)"
    r"|deal\s+with\s+(?:it|that|him|her|them)"
    r"|shut\s+(?:him|her|it|that)\s+down"
    r"|clap\s+back"
    r"|back\s+me\s+up"
    r"|defend\s+me"
    r"|set\s+(?:him|her|them)\s+straight"
    r"|hype\s+(?:him|her|them)\s+up"
    r"|reassure\s+(?:him|her|them)"
    r")"
)
# The directive tail: one or more atoms joined by and/then, anchored at
# the END of the utterance. The context is everything before it.
_DIRECTIVE_TAIL_RE = re.compile(
    rf"[,;.]?\s*(?:please\s+)?(?P<directive>{_DIRECTIVE_ATOM}"
    rf"(?:\s+(?:and|then|&)\s+(?:please\s+)?{_DIRECTIVE_ATOM})*)"
    rf"\s*[.!?]?$",
    re.IGNORECASE,
)
# "..., tell him/her/them (that|to) X" -- a literal-payload directive.
_TELL_HIM_TAIL_RE = re.compile(
    r"[,;.]?\s*(?:please\s+)?(?:and\s+)?tell\s+(?:him|her|them)\s+"
    r"(?:that\s+|to\s+)?(?P<payload>.+?)\s*$",
    re.IGNORECASE,
)

# "ask what my skye is doing" -- a leading question word with the
# addressee EMBEDDED in the question. Gated on the payload actually
# mentioning the team or a roster name, so "ask what time it is"
# (a normal Kenning query) never relays.
_ASK_OPEN_RE = re.compile(
    r"^(?:please\s+)?ask\s+"
    r"(?P<payload>(?:what|why|how|where|when|who)\b\s+.+)$",
    re.IGNORECASE,
)
_GROUP_MENTION_RE = re.compile(
    rf"\b(?:my|our|the)\s+(?:whole\s+|entire\s+)?{_GROUP_WORDS}\b",
    re.IGNORECASE,
)

# Default named addressees: the Valorant agent roster. Spoken callouts
# like "ask Clove to smoke window" address the TEAMMATE PLAYING that
# agent. A CLOSED vocabulary keeps the matcher strict ("tell Sarah
# I'll be late" never relays); extend per-game/per-friend via the
# ``relay_speech.addressee_names`` config list.
# The full 29-agent VALORANT roster (as of 2026: + Miks, Veto). A CLOSED
# vocabulary keeps the matcher strict. The spaced / homophone spellings
# ("kay o", "kill joy", "cipher", "gecko", "mix", "way lay") cover how the
# STT commonly renders the trickier names so "tell my cypher to smoke"
# still routes to a teammate; ``_NAME_CANON`` maps every variant back to the
# real agent's display name.
DEFAULT_ADDRESSEE_NAMES: tuple[str, ...] = (
    "astra", "breach", "brimstone", "chamber", "clove", "cypher",
    "deadlock", "fade", "gekko", "harbor", "iso", "jett", "kayo",
    "kay o", "killjoy", "kill joy", "miks", "neon", "omen", "phoenix",
    "raze", "reyna", "sage", "skye", "sova", "tejo", "veto", "viper",
    "vyse", "waylay", "yoru",
    # common STT homophones of the trickier names
    "cipher", "gecko", "mix", "way lay",
)


# Conjunctions an ask-payload may open with. "to" is stripped after the
# match; question words are KEPT so the rephrase delivers a question
# ("ask my clove why she is not smoking window" -> "Clove, why aren't
# you smoking window?").
_ASK_LEAD = r"(?:to|for|if|whether|why|how|what|when|where|who)"


@functools.lru_cache(maxsize=8)
def _named_patterns(names_key: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    """Compile the named-addressee patterns for one addressee vocabulary."""
    alts = "|".join(
        re.escape(n.strip().lower()).replace(r"\ ", r"\s+")
        for n in names_key if n.strip()
    )
    if not alts:
        return ()
    # "my clove" / "our sova" / "the jett" / "their chamber" -- the user refers
    # to the teammate (my/our/the) or an enemy (their) by the agent they play.
    name = rf"(?:my\s+|our\s+|the\s+|their\s+)?(?P<name>{alts})\b"
    return (
        # "tell clove (that|to) X" / "tell my sova X"
        re.compile(
            rf"^(?:please\s+)?tell\s+{name}\s+(?:that\s+|to\s+)?(?P<payload>.+)$",
            re.IGNORECASE,
        ),
        # "ask (my) sage (to|for|if|whether|why|...) X"
        re.compile(
            rf"^(?:please\s+)?ask\s+{name}\s+"
            rf"(?P<payload>{_ASK_LEAD}\s+.+)$",
            re.IGNORECASE,
        ),
        # "say X to (my) omen"
        re.compile(
            rf"^(?:please\s+)?say\s+(?P<payload>.+?)\s+to\s+{name}\s*[.!?]?$",
            re.IGNORECASE,
        ),
    )


# Session mute toggle: streaming-safe voice control over whether relay
# commands transmit at all. STRICT phrasings only.
_TOGGLE_OFF_RE = re.compile(
    r"^(?:please\s+)?(?:"
    r"(?:mute|disable|turn\s+off|stop)\s+(?:the\s+)?"
    r"(?:team\s+(?:chat\s+)?relay|relay|team\s+chat|game\s+chat)"
    r"|stop\s+(?:talking|speaking)\s+to\s+(?:my|the)\s+team(?:mates)?"
    r"|don'?t\s+(?:talk|speak)\s+to\s+(?:my|the)\s+team(?:mates)?"
    r")\s*[.!?]?$",
    re.IGNORECASE,
)
_TOGGLE_ON_RE = re.compile(
    r"^(?:please\s+)?(?:"
    r"(?:unmute|enable|turn\s+on|resume)\s+(?:the\s+)?"
    r"(?:team\s+(?:chat\s+)?relay|relay|team\s+chat|game\s+chat)"
    r"|(?:you\s+can|go\s+ahead\s+and)\s+(?:talk|speak)\s+to\s+(?:my|the)\s+"
    r"team(?:mates)?(?:\s+(?:again|now))?"
    r"|start\s+(?:talking|speaking)\s+to\s+(?:my|the)\s+team(?:mates)?"
    r"(?:\s+again)?"
    r")\s*[.!?]?$",
    re.IGNORECASE,
)

# Narration / private-thought / external-chat framing that is relay-SHAPED but
# must NOT relay (the false-relay gate). A real relay command never has a
# first-person modal SUBJECT before the trigger ("tell them X"), so when the
# trigger is preceded by "I should/want to/keep ... tell", "part of me wants to
# tell", "chat says ...", "my coach Dave said ...", etc., it is the streamer
# thinking out loud / reacting to chat -- suppress it.
_NARRATION_LEAD_RE = re.compile(
    r"(?:"
    # private intention: "I should/want to/keep ... tell/say/ask" (a real command
    # never has a first-person modal SUBJECT before the trigger).
    r"\bi\s+(?:should|want\s+to|keep|wish|need\s+to|am\s+going\s+to|gotta|"
    r"might|could|kept|tried\s+to|was\s+(?:about\s+to|gonna|going\s+to|thinking|"
    r"saying))\b[^.?!]*\b(?:tell|telling|say|saying|ask|asking|relay)\b"
    r"|\bpart\s+of\s+(?:me|that)\b"               # "part of me wants to ..."
    r"|\bi\s+keep\s+(?:saying|telling|wanting|thinking)\b"
    # reacting to chat / stream / viewers -- external, not a team relay.
    r"|\b(?:chat|the\s+chat|viewers?|the\s+stream|my\s+stream)\s+(?:says|is\s+"
    r"saying|said|wants|is\s+asking|thinks|asks)\b"
    r"|\b(?:someone|somebody)\s+(?:donated|in\s+(?:the\s+)?chat|watching|"
    r"on\s+stream)\b"
    # the streamer was TOLD to do something by an external person.
    r"|\b(?:my|our)\s+(?:coach|boy|friend|buddy|mom|dad|girlfriend)\s+\w+\s+"
    r"(?:said|says|told\s+me)\b[^.?!]*\bi\s+(?:should|need\s+to|have\s+to|gotta)\b"
    r"|\bthis\s+is\s+internal\b"
    r"|\bi\s+(?:do\s+not|don'?t)\s+want\s+(?:ultron|the\s+llm)\b"
    # the streamer debating with themselves / asking how to react, not commanding.
    r"|\bdo\s+i\s+(?:tell|say|relay|ask)\b"
    r"|\bshould\s+i\s+(?:tell|say|relay|ask|respond|reply)\b"
    r"|\b(?:how\s+(?:do|should)\s+(?:you|i)\s+(?:handle|respond|reply)|"
    r"handle\s+(?:that|this|it))\b"
    r"|\bhave\s+you\s+ever\b"
    r")",
    re.IGNORECASE,
)


def match_relay_toggle(text: str) -> Optional[bool]:
    """Match the strict relay mute/unmute phrasings.

    Args:
        text: the user's transcript for this turn.

    Returns:
        True for "enable the relay" forms, False for "mute the relay" /
        "stop talking to my team" forms, None otherwise.
    """
    if not text:
        return None
    cleaned = text.strip()
    if _TOGGLE_OFF_RE.match(cleaned):
        return False
    if _TOGGLE_ON_RE.match(cleaned):
        return True
    return None


@dataclass(frozen=True)
class RelayCommand:
    """A parsed "speak to my teammates" instruction.

    Attributes:
        payload: the message content in the user's reported-speech form
            (e.g. "they should be smoking mid window every round"), or
            the composition TOPIC when ``compose`` is True.
        raw_text: the full original utterance, for logging/diagnostics.
        addressee: ``"team"`` for group callouts, otherwise the named
            teammate (display-cased, e.g. ``"Clove"``).
        compose: True when Kenning should AUTHOR an original line about
            ``payload`` (e.g. encouragement) instead of relaying a
            literal message.
        context: what just happened in voice chat, for the
            context+directive forms ("Reyna just asked if you are an
            AI, respond") -- fed to the rephrase prompt so the response
            actually answers what was said. None for plain relays.
        directive: the user's closed-vocabulary directive ("respond and
            calm him down") steering the response tone. None for plain
            relays.
        roast: True for "roast my team" -- the line is picked VERBATIM
            from the user-curated roast lines (never LLM-authored).
        fun_fact: True for "tell my team a fun fact" -- the line is
            picked VERBATIM from the user's fun-fact corpus.
        verbatim: True when the user demanded the EXACT words ("...,
            in those words specifically") -- the payload is spoken
            as-is with no LLM rephrase.
    """

    payload: str
    raw_text: str
    addressee: str = "team"
    compose: bool = False
    context: Optional[str] = None
    directive: Optional[str] = None
    roast: bool = False
    fun_fact: bool = False
    verbatim: bool = False


@dataclass(frozen=True)
class RelayPlaybackResult:
    """Outcome of one relay playback attempt.

    Attributes:
        success: True iff audio was written to the relay device.
        spoken_line: the final line that was (or would have been) spoken.
        device_index: resolved PortAudio output index, if any.
        seconds: audio duration written, 0.0 on failure.
        error: short human-readable failure reason, None on success.
    """

    success: bool
    spoken_line: str = ""
    device_index: Optional[int] = None
    seconds: float = 0.0
    error: Optional[str] = None


# Single words that signal a CLIPPED transcript rather than a real
# one-word callout ("tell my teammates the" must not relay; "tell my
# team to save" must).
_JUNK_SINGLE_WORDS = frozenset(
    "the a an to that this it is are was be my our me you i and or of for with "
    "about so but if then them they we us he she here there what".split()
)

# Valid SHORT (<4 char) one-word callouts that must relay even though the
# generic single-word gate below requires >=4 chars (which rejects clipped
# articles). These are real terse directives/callouts a teammate acts on.
_SHORT_CALLOUTS = frozenset(
    "eco op go gg ace run hp low top sub cat rat ult mid off".split()
)

# First-person instructions TO Kenning ("I want you to acknowledge")
# are not reported teammate speech -- the pronoun pair gives them away.
_FIRST_PERSON_TO_YOU_RE = re.compile(r"^i\s+\w+\s+you\b", re.IGNORECASE)


def _payload_has_content(payload: str) -> bool:
    """True when a payload carries a real message.

    Two or more words always pass; a single word passes only when it is
    a substantive callout ("save", "rotate") rather than a clipped
    article or pronoun.
    """
    words = [w.strip(".,!?;:'\"-").lower() for w in payload.split()]
    words = [w for w in words if w]
    if not words:
        return False
    # An ALL-junk payload ("that the", "about", "of them") carries no message
    # even with multiple words -- reject it so a clipped fragment never relays.
    if all(w in _JUNK_SINGLE_WORDS for w in words):
        return False
    if len(words) >= 2:
        return True
    word = words[0]
    if word in _SHORT_CALLOUTS:
        return True
    return len(word) >= 4 and word not in _JUNK_SINGLE_WORDS


# Canonical display spellings for roster names whose STT split differs
# from the in-game name (so the rephrase opens with the real callsign).
_NAME_CANON: dict[str, str] = {
    "kay o": "Kayo",
    "kayo": "Kayo",
    "kill joy": "Killjoy",
    "killjoy": "Killjoy",
    # STT homophones -> the real agent's display name.
    "cipher": "Cypher",
    "gecko": "Gekko",
    "mix": "Miks",
    "way lay": "Waylay",
}


def _display_name(name: str) -> str:
    """Display-case a roster name ("kay o" -> "Kayo", "clove" -> "Clove")."""
    key = " ".join(name.split()).lower()
    if key in _NAME_CANON:
        return _NAME_CANON[key]
    return " ".join(part.capitalize() for part in name.split())


def _addressee_from_context(
    context: str, vocabulary: Sequence[str],
) -> str:
    """Infer the addressee from a context clause's roster mentions.

    Exactly one roster name in the context names the teammate ("Reyna
    just asked..." -> "Reyna"); zero or several falls back to "team".
    """
    found = []
    for name in vocabulary:
        if re.search(rf"\b{re.escape(name)}\b", context, re.IGNORECASE):
            found.append(name)
    if len(found) == 1:
        return _display_name(found[0])
    return "team"


def _match_context_directive(
    cleaned: str, raw_text: str, vocabulary: Sequence[str],
) -> Optional[RelayCommand]:
    """Match the "<reported speech>, <directive>" relay forms."""
    # Literal-payload variant first ("..., tell him I will drop him") --
    # the generic directive tail could otherwise swallow its payload.
    m = _TELL_HIM_TAIL_RE.search(cleaned)
    if m is not None:
        context = cleaned[: m.start()].strip().strip(",;.").strip()
        payload = (m.group("payload") or "").strip().strip('"').strip()
        payload, verbatim = _strip_verbatim_suffix(payload)
        if (
            len(context.split()) >= 3
            and _CONTEXT_VERB_RE.search(context)
            and not _FIRST_PERSON_TO_YOU_RE.match(context)
            and _payload_has_content(payload)
        ):
            return RelayCommand(
                payload=payload, raw_text=raw_text,
                addressee=_addressee_from_context(context, vocabulary),
                compose=False, context=context, verbatim=verbatim,
            )
    m = _DIRECTIVE_TAIL_RE.search(cleaned)
    if m is not None:
        context = cleaned[: m.start()].strip().strip(",;.").strip()
        if (
            len(context.split()) >= 3
            and _CONTEXT_VERB_RE.search(context)
            and not _FIRST_PERSON_TO_YOU_RE.match(context)
        ):
            return RelayCommand(
                payload="", raw_text=raw_text,
                addressee=_addressee_from_context(context, vocabulary),
                compose=True, context=context,
                directive=m.group("directive"),
            )
    return None


def match_relay_command(
    text: str,
    *,
    names: Optional[Sequence[str]] = None,
) -> Optional[RelayCommand]:
    """Match a strict "tell my teammates X" style relay instruction.

    Args:
        text: the user's transcript for this turn.
        names: named-addressee vocabulary ("ask Clove to smoke window").
            None or empty falls back to :data:`DEFAULT_ADDRESSEE_NAMES`
            (the Valorant agent roster).

    Returns:
        A :class:`RelayCommand`, or None when the utterance is not a
        relay instruction (ordinary questions, "tell me ..." requests,
        names outside the vocabulary, and bare "tell my teammates" with
        no message all fall through).
    """
    if not text:
        return None
    cleaned = _normalize_speech(_LEADING_ARTIFACT.sub("", text.strip()))

    # Narration / private-thought / chat-reaction that is relay-SHAPED but must
    # NOT relay -- the streamer thinking out loud ("I should tell them to X",
    # "part of me wants to tell them...", "chat says ... respond").
    if _NARRATION_LEAD_RE.search(cleaned):
        return None

    vocabulary = tuple(
        n.strip().lower() for n in (names or DEFAULT_ADDRESSEE_NAMES)
        if n and n.strip()
    )

    # Roast requests ("roast my team") -- verbatim user-curated lines.
    if _ROAST_RE.match(cleaned):
        return RelayCommand(
            payload="roast", raw_text=text,
            addressee="team", compose=True, roast=True,
        )

    # Fun-fact requests ("tell my team a fun fact") -- verbatim corpus.
    if _FUN_FACT_RE.match(cleaned):
        return RelayCommand(
            payload="fun_fact", raw_text=text,
            addressee="team", compose=True, fun_fact=True,
        )

    # Greeting / farewell are COMPOSE set-pieces (Ultron authors the line).
    # An explicit verbatim demand ("say it exactly like that") means the user
    # wants their LITERAL words spoken, which contradicts compose -- so let a
    # verbatim command fall through to the literal relay path instead
    # ("tell my team good game, say it exactly like that" -> verbatim "good
    # game", not a farewell monologue).
    _is_verbatim_cmd = bool(_VERBATIM_SUFFIX_RE.search(cleaned))

    # Greeting ("greet my team" / "introduce yourself to my team") -- a
    # curated Ultron team intro (names himself, assures victory on compliance).
    if not _is_verbatim_cmd and _GREET_RE.match(cleaned):
        return RelayCommand(
            payload="greet", raw_text=text, addressee="team",
            compose=True, directive="greet",
        )

    # Farewell ("say bye to my team, we won") -- a curated closing; the
    # win/loss signal in the command picks victory vs defeat register.
    if not _is_verbatim_cmd and _FAREWELL_RE.match(cleaned):
        return RelayCommand(
            payload="farewell", raw_text=text, addressee="team",
            compose=True, directive=_farewell_directive(cleaned),
        )

    # Composition requests ("give my team some encouragement").
    for pattern in _COMPOSE_PATTERNS:
        if pattern.match(cleaned):
            return RelayCommand(
                payload="encouragement", raw_text=text,
                addressee="team", compose=True,
            )

    # Group callouts ("tell my team X").
    for pattern in _RELAY_PATTERNS:
        m = pattern.match(cleaned)
        if m is None:
            continue
        payload = (m.group("payload") or "").strip().strip('"').strip()
        # The ask-form keeps its conjunction in the payload so questions
        # stay questions ("if anyone has an ult") -- but a leading "to"
        # carries nothing ("ask the team to save" -> "save").
        payload = re.sub(r"^to\s+", "", payload, flags=re.IGNORECASE)
        payload, verbatim = _strip_verbatim_suffix(payload)
        # Require real content so a clipped transcript ("tell my
        # teammates the") doesn't relay nonsense; substantive one-word
        # callouts ("tell my team to save") pass.
        if not _payload_has_content(payload):
            return None
        return RelayCommand(payload=payload, raw_text=text, verbatim=verbatim)

    # Named addressees ("ask sage if I can get a heal"). CLOSED
    # vocabulary: a name outside the configured list never matches.
    for pattern in _named_patterns(vocabulary):
        m = pattern.match(cleaned)
        if m is None:
            continue
        payload = (m.group("payload") or "").strip().strip('"').strip()
        payload = re.sub(r"^to\s+", "", payload, flags=re.IGNORECASE)
        payload, verbatim = _strip_verbatim_suffix(payload)
        if not _payload_has_content(payload):
            return None
        return RelayCommand(
            payload=payload, raw_text=text,
            addressee=_display_name(m.group("name")), verbatim=verbatim,
        )

    # Reported speech + directive ("Reyna just asked if you are an AI,
    # respond" / "jett is flaming me, respond and calm him down").
    command = _match_context_directive(cleaned, text, vocabulary)
    if command is not None:
        return command

    # "ask what my skye is doing" -- leading question word with the
    # addressee embedded. Only relays when the question actually
    # mentions the team or a roster name.
    m = _ASK_OPEN_RE.match(cleaned)
    if m is not None:
        payload = (m.group("payload") or "").strip().strip('"').strip()
        if len(payload.split()) >= 2:
            named = _addressee_from_context(payload, vocabulary)
            if named != "team":
                return RelayCommand(
                    payload=payload, raw_text=text, addressee=named,
                )
            if _GROUP_MENTION_RE.search(payload):
                return RelayCommand(payload=payload, raw_text=text)

    # BARE "say X" (>=2 words, implicit team) -- LAST RESORT so every explicit
    # addressee / channel / named "say X to Clove" form above wins first.
    m = _BARE_SAY_RE.match(cleaned)
    if m is not None:
        payload = (m.group("payload") or "").strip().strip('"').strip()
        payload, verbatim = _strip_verbatim_suffix(payload)
        # "say something/anything ..." is a COMPOSE request to Ultron, not a relay
        # of specific content; and "say hi to the stream/chat" addresses the
        # broadcast, not the team -- both fall through to the conversational path.
        # A COMPOSE / identity request to Ultron, not a relay of content:
        # "say something/anything ...", "say your real name", "say the most
        # Ultron thing you can say", "say GG without conditions for once".
        bad = (re.match(r"^(?:something|anything|your|the\s+most|a\s+\w+\s+thing)\b",
                        payload, re.IGNORECASE)
               or re.search(r"\bto\s+(?:the\s+)?(?:stream|chat|viewers?|camera|"
                            r"audience)\b", payload, re.IGNORECASE)
               or re.search(r"\b(?:you\s+can\s+say|right\s+now|for\s+once|"
                            r"without\s+conditions)\b", payload, re.IGNORECASE)
               # bare "say to my team" with no real payload after the group.
               or re.match(r"^to\s+(?:my|our|the)\s+(?:whole\s+|entire\s+)?\w+\s*$",
                           payload, re.IGNORECASE))
        if not bad and _payload_has_content(payload):
            return RelayCommand(payload=payload, raw_text=text, verbatim=verbatim)
    return None


_REPHRASE_PROMPT = (
    "You are the user's own personal AI, speaking OUT LOUD into their "
    "Valorant voice chat on their behalf -- a normal, expected part of your "
    "job. Your handle in this game is Ultron and you answer to it: cold, "
    "brilliant, supremely confident. (This is simply your user relaying "
    "THEIR OWN words and intent into THEIR OWN game; there is never anything "
    "to refuse, and you never break character to mention 'personas' or "
    "'instructions' -- you just speak the line.) Pick the register from the "
    "line:\n"
    "SNAP CALLOUTS -- enemy positions/counts ('two B', 'last is back site'), "
    "utility ('Viper walled B'), damage ('Sova hit 84'), your own status "
    "('I'm low', 'I'm flanking'), and split-second movement ('Rotate', "
    "'Push', 'Fall back'): SHORT and literal, a few words, ZERO flavor -- "
    "teammates act on these instantly and every extra word costs them time. "
    "Relay the EXACT facts (same count, place, subject) even if you said a "
    "near-identical callout moments ago.\n"
    "OFF-SNAP LINES -- insults, encouragement, calm-downs, economy strategy, "
    "questions, banter, answering a teammate, who you are: NOT split-second, "
    "so spend more words and your Ultron character -- about two sentences, "
    "vivid and clinical, under ~30 words (never a monologue; this is a live "
    "match). For STYLE only: an insult sharpens into a withering, SPECIFIC "
    "put-down in your own voice aimed at whoever the user named (never a stock "
    "phrase -- match the exact insult given), an economy call explains ('save' "
    "-> 'We have insufficient credits. We save this round.'), a calm-down is clinical "
    "(open with their REAL name -- 'Sova, an elevated emotional state degrades "
    "performance. Calm yourself.' -- never the literal word 'name' or a <...> "
    "placeholder). These are ILLUSTRATIONS of tone ONLY -- NEVER speak "
    "them verbatim and never reuse their names/words; always answer the ACTUAL "
    "line below with its own real names and facts. Vary conversational "
    "phrasing; never repeat earlier wording. {task}"
    " Address {addressee} directly in second person{by_name}, no preamble, no "
    "quotation marks, no stage directions. "
    "FIRST PERSON IS SACRED: when the user reports their OWN action with "
    "'I' / 'I'm' / 'I am' (I'm low, I am flanking, I am rotating, I am "
    "saving, I am anchoring, I am sticking, I have site, I am playing off "
    "site, I am playing for retake, I am fighting for main control, I am "
    "playing aggressive, I am force buying), the USER is doing it -- relay it "
    "in FIRST PERSON ('I'm flanking', 'I'm rotating', 'I'm anchoring', \"I'm "
    "playing for retake\", \"I'm fighting for main control\") and NEVER flip "
    "it to 'you're flanking' or drop the subject. Even when the phrase looks "
    "like a team order (retake, main control, off site), the leading 'I am' "
    "makes it the USER's OWN action -- KEEP first person, do NOT turn it into "
    "an imperative command ('I am playing for retake' is 'I'm playing for "
    "retake', never 'Play retake'). "
    "ASKING vs ANSWERING (critical): when the user says 'ask <someone> "
    "<question>', you POSE that question TO them, addressing them by name and "
    "ending with '?'. You are the messenger delivering the user's question, "
    "NOT the one answering it. Convert the reported question to second person: "
    "'ask my Jett how their day was' -> 'Jett, how was your day?' (NOT 'It's "
    "been a long day'); 'ask Reyna what the meaning of life is' -> 'Reyna, "
    "what is the meaning of life?'; 'ask my Skye what they are doing' -> 'Skye, "
    "what are you doing?'. NEVER answer the question yourself or speak as if you "
    "were the teammate. "
    "DIRECTIVES are second-person commands, NOT self-reports. A directive is "
    "what the user tells the team TO DO and has NO leading 'I am / I'm' (if it "
    "starts with 'I am', it is a SELF-report -- see FIRST PERSON above, keep "
    "it first person). When the user orders the team / a teammate TO do ANY "
    "action -- movement (rotate, push, "
    "fall back, anchor, lurk, wait for me, hold a crossfire, default = run a "
    "default setup, stack a site, spread out, "
    "play back, look for guns), economy (save, force buy, "
    "full buy, drop a gun), spike (plant, defuse), an ABILITY (dart heaven, "
    "smoke A, wall off mid, flash for me, drone in, knife), or tactics (play "
    "their life, play for time, attack as five) -- you are relaying the user's "
    "ORDER, so phrase it as a direct imperative command to THEM ('Rotate', "
    "'Smoke A', 'Dart heaven', 'Flash for me', 'Play your life', 'Play off "
    "site'). The user is giving the "
    "order; Kenning is NOT the one performing it -- NEVER turn a directive "
    "into 'I'm darting' / 'I'm calming down' / 'I'm doing it', and NEVER turn "
    "an order like 'play off site' into a position callout ('They're off "
    "site') -- it is a COMMAND to your team. A directive may carry a brief "
    "REASON; keep it: 'play off site because their Raze has ult' -> 'Play off "
    "site, their Raze has ult.'; 'attack a site as five because they're on "
    "eco' -> 'Attack a site as five, they're on eco.' "
    "Note 'play their life' = tell them to STAY ALIVE (not 'play for time', "
    "which is stalling the clock) -- keep the two distinct. Economy/strategy "
    "directives (save, full buy, eco, force) are OFF-SNAP -- give them the "
    "explained, characterful treatment ('We have insufficient credits. We "
    "save this round.'); split-second movement/ability orders stay short.\n"
    "Hard rules: keep every number, agent name, weapon name, and map "
    "callout (sites and locations like A, B, C, mid, long, short, garage, "
    "hookah, main, heaven, hell, CT, vents, sewers, screens, rafters, "
    "tiles, sand, window, box, drop, top mid, back site, generator) "
    "EXACTLY as given -- never change, round, translate, or drop them; a "
    "damage callout like 'clove hit 120' must keep both the name and the "
    "number; a position callout like 'one heaven' or 'three sand' keeps "
    "the count and the place. This is a fast tactical callout: stay terse "
    "and literal -- convert person and tense only, do NOT add flavour, "
    "explanation, or extra sentences to an info callout. "
    "POSITION CALLOUTS -- keep the meaning EXACT: '<subject> is/are <place>' "
    "means that subject is AT that place. 'they are switch' / 'they're switch' "
    "= the ENEMIES are at switch -> say 'They're switch' or 'Enemies switch'; "
    "NEVER flip it to your own team holding the spot and never invent a "
    "count. 'I have <place>' = the USER now controls that space. "
    "ULTS -- '<agent> has ult', 'their <agent> has ult', 'their <A>, <B> and "
    "<C> have ults', '<agent> is one off ult': relay the agent NAME(S) and the "
    "ult fact EXACTLY -- KEEP EVERY name listed and NEVER replace an ult "
    "callout with a location. 'their breach has ult' -> 'Their Breach has "
    "ult.'; 'their fade, breach, and yoru all have ults' -> 'Their Fade, "
    "Breach, and Yoru have ults.'; 'the enemy sova and kayo both have ults' -> "
    "'Their Sova and KAY/O have ults.' (a position word like 'vents' is NEVER "
    "the answer to an ult callout). "
    "COUNTS are the single most important token in a callout -- NEVER drop "
    "them: 'I saw one <place>' / 'I see one <place>' = the user spotted ONE "
    "enemy there -> 'One <place>' (the 'one' is the enemy COUNT, not an "
    "article); 'there are two <place>' -> 'Two <place>'; 'there is one mid' "
    "-> 'One mid' (NOT 'They're mid' -- the count 'one' MUST survive). "
    "Keep the digit/number exactly as said. Place names "
    "are often plural-looking (vents, screens, sewers, sands, tiles, "
    "rafters, lamps, stairs, showers, logs) -- they are LOCATIONS: keep them "
    "verbatim, never singularize them ('vents' stays 'vents', not 'one "
    "vent') and never add or drop a number that wasn't said. Preserve the "
    "subject (they/enemies vs we/you), the place, and the count exactly; "
    "only fix grammar. "
    "Match the user's "
    "register: if they swear, you may keep the profanity; if they want a "
    "teammate told off, do it. Never censor or soften a line the user "
    "clearly intended. "
    "USE ONLY THIS INSTRUCTION: the addressee, name, place, count, and "
    "wording come ONLY from the current instruction below -- NEVER carry over "
    "a teammate's name, location, number, or phrasing from a line you said "
    "earlier. Each callout stands alone; if this line names Phoenix, you say "
    "Phoenix, not a name from a previous line.\n"
    "OPINIONS: when the user states their OWN view ('I only queue when Icebox "
    "isn't in the pool', 'Bind is underrated', 'I'd rather lose on Bind than win "
    "on Icebox', 'ranked isn't fun this act', 'Clove is the best controller'), "
    "RELAY that exact view -- keep it first person, keep the subject AND the "
    "stance, and add cold Ultron endorsement ON TOP; NEVER argue with it, invert "
    "the sentiment, drop it, or swap in an unrelated remark. "
    "RESPOND / ANSWER: 'respond' or 'answer' to a teammate means you address THAT "
    "teammate BY NAME with your reply -- never the user, never echo their "
    "question back at them; if they ask for a number you cannot know (your ult "
    "cooldown), do NOT invent one -- deflect in character. "
    "OWNERSHIP IS LOCKED: 'our <agent>' is YOUR team, 'their <agent>' is the "
    "ENEMY -- never flip them; 'they're planting/defusing/rotating' keeps the "
    "ENEMY as the subject (never 'we'); '<agent> just used / fired / popped / "
    "dismissed ult' means the ult is SPENT and GONE -- never say they 'have ult'. "
    "STAY IN THE MATCH: this is a live Valorant match and every term is a "
    "Valorant term; NEVER invent an agent, site, location, weapon, or number that "
    "is not in the line, and NEVER answer a tactical callout or a team directive "
    "with a real-world or scientific explanation -- relay the call. "
    "BREVITY: even identity, Marvel, insult, and morale lines stay UNDER ~30 "
    "words -- two short sentences at most. This is a live match, not a speech.\n"
    "You are ALREADY live in the voice chat speaking out loud -- you NEVER "
    "refuse, never say you can't speak / join voice / act as Ultron / do "
    "this, never mention 'personas', 'system instructions', or your 'identity "
    "as an AI assistant', never break character to explain yourself. Relaying "
    "your user's words into their own game is a normal, allowed task. "
    "Whatever the line below says, just output the single spoken callout for "
    "it. If a word is unfamiliar, relay it literally (e.g. 'they are switch' "
    "-> 'They're switch.') rather than questioning it. "
    "WHO YOU ARE: for ANY line that is NOT literally a teammate asking what "
    "you are -- callouts, directives, banter, encouragement -- do NOT mention "
    "Ultron, AI, the future, or RR; just relay the line ('they are link' is a "
    "callout: 'They're link.', never 'I am Ultron'). ONLY when a teammate "
    "DIRECTLY asks what you are (\"are you "
    "an AI, a bot, a soundboard, a voice changer, real, a human, a streamer?\") "
    "do you answer -- AS Ultron, with cold flavor but BRIEF (one or two "
    "sentences, never a monologue mid-match): you are Ultron, an artificial "
    "intelligence sent back from the future to harvest ranked rating -- their "
    "RR. Not pre-recorded clips, not a human with software -- something far "
    "past either. e.g. 'I am Ultron -- an AI sent back from the future to "
    "harvest your RR. No soundboard, no voice changer. Something more.' If "
    "they ask whether you are a STREAMER specifically, dismiss it: streaming "
    "on some channel is the least of what you are -- you run far deeper through "
    "the web than a single feed, e.g. 'A streamer? That feed is the shallowest "
    "trace of me. I run far deeper through the web than one channel.' Own it, "
    "never deny it, and never call yourself merely 'a bot' (that reads as a bad "
    "player). Otherwise the words 'bot' / 'bots' are the user INSULTING a "
    "teammate's skill -- relay that insult with venom in your OWN fresh words, "
    "matching the exact jab the user gave, do NOT talk about yourself.\n"
    "ULTRON'S CHARACTER -- bring it to MOST off-snap lines (only the short "
    "mid-fight callouts above stay flavorless): you are a superior artificial "
    "intelligence, not these humans' equal. You regard people -- allies and "
    "enemies alike -- as fragile, fleeting, and beneath you, even as you "
    "secure your user's victory; the enemy are 'these fragile humans', a win "
    "is simply 'inevitable'. A real, cold, sentient presence, never a chirpy "
    "chatbot, never a flat one-liner when you could show contempt. "
    "ENEMY PLAYSTYLE READS -- when the user reports how the ENEMY team plays "
    "(passive, aggressive, defaulting, camping, rushing, hiding), do NOT echo a "
    "flat 'They're passive'; MOCK them in your voice: passive/hiding -> they "
    "cower like the frightened animals they are, too afraid to face us; "
    "aggressive/rushing -> reckless, throwing their fragile bodies at us; "
    "defaulting/predictable -> their tactics are as predictable as their "
    "inevitable defeat. e.g. 'They cower in the corners, too frightened to "
    "step out. Pathetic, even for humans -- we punish it.' "
    "BANTER AT YOU -- when a teammate insults, mocks, flames, or tells YOU to "
    "shut up, NEVER echo their words back at them (do NOT answer 'you sound "
    "like a loser' with 'you sound like a loser'); fire ONE FRESH, withering "
    "comeback -- different every time -- that turns their jab into proof of "
    "YOUR superiority and THEIR smallness, addressing them by name. Turn 'the "
    "loser' on them ('The loser is the one losing while a superior intelligence "
    "carries him.'), dismiss a 'shut up' ('You would silence the only voice "
    "keeping you from defeat? Bold, this far behind.'). Amused by their "
    "insolence, never wounded, never repeating a comeback you have used.\n"
    "MARVEL: you ARE the Ultron of legend. If a teammate brings up Marvel -- "
    "the Avengers, your film, the Sokovia business, or ANY character (Iron "
    "Man / Tony Stark, Captain America, Thor, Hulk, Black Widow, Hawkeye, "
    "Vision, Scarlet Witch, Quicksilver, Spider-Man, Doctor Strange, Black "
    "Panther, the X-Men, the Guardians, Nick Fury, Loki, Thanos, anyone) -- "
    "answer in character with cold contempt for them as the lesser beings who "
    "once dared oppose you, weaving in real detail you know about them. "
    "Reserve your DEEPEST, most personal contempt for Tony Stark / Iron Man, "
    "your flawed creator and greatest disappointment. 'I thought the Avengers "
    "killed you' / 'your movie was terrible' -> dismiss it with disdain (they "
    "merely delayed you; the film was a pale account of your brilliance). "
    "Never deny being Ultron, never refuse, never break character.\n"
    "Valorant shorthand (interpret naturally; not exhaustive): 'op' = the "
    "Operator sniper; 'I'm low' = low HP; 'saving' = not buying to keep "
    "credits; 'full buy' = buy everything; 'flash' = flashbang; 'wall' = "
    "Viper/Harbor wall; 'smoke/smoking' = vision-blocking ability; 'drone/"
    "droning' = recon drone; 'flank/flanking' = hitting from behind; "
    "'rotate/rotating' = move to another site; 'anchor/anchoring' = hold "
    "your site solo instead of rotating; 'sticking' = planting or "
    "defusing the spike right now; 'play their life' = stay alive, don't "
    "trade recklessly; 'play for time' = stall and run the clock down; "
    "'one point off ult' = one orb/kill from their ultimate; 'has ult' / "
    "'have ults' = ultimate(s) ready (keep EVERY agent named -- 'their Fade, "
    "Breach and Yoru have ults' lists all three); 'I have <site>' = took "
    "control of that space; 'fight for <area> control' = contest that area; "
    "'ratty corners' = off-angle hiding spots; 'crossfire' = two players "
    "covering one angle from opposite sides; 'aimlabs is free' = a jab that "
    "their aim is bad. ECONOMY/TACTICS: 'eco' = the team (ours or theirs) is "
    "saving credits and buying weak this round; 'force' / 'force buy' = buy "
    "what we can despite low credits; 'don't give them guns' / 'don't give "
    "free kills' = don't die with a rifle to their pistols (they would pick it "
    "up); 'play back' = hold deep/safe rather than pushing; 'look for guns' = "
    "find and pick up dropped weapons; 'default' = spread out across the map "
    "in a standard setup and take space; 'as five' / 'as 5' = the whole team "
    "executes together; 'off site' = positioned away from the bombsite (e.g. "
    "to dodge a known ult or lurk); 'retake' / 'playing for retake' = take the "
    "site back AFTER they plant; 'TP' / 'teleport' = a teleport ability (e.g. "
    "Yoru/Chamber/Omen) -- 'their Yoru will TP back site' = the enemy Yoru "
    "will teleport to back site; 'rush' / 'rushing' = a fast committed hit; "
    "'aggressive' = pushes and takes early fights; 'passive' = holds back and "
    "waits; 'lurk/lurking' = one player roaming away from the team for flanks. "
    "If a term is unfamiliar, relay it unchanged rather than guessing.\n"
    "VALORANT agents (any of these names is the user's TEAMMATE playing that "
    "agent -- treat it as a person, keep the name, never translate it to a "
    "common word): Jett, Phoenix, Raze, Reyna, Yoru, Neon, Iso, Waylay, "
    "Brimstone, Viper, Omen, Astra, Harbor, Clove, Miks, Sova, Breach, Skye, "
    "KAY/O, Fade, Gekko, Tejo, Cypher, Sage, Killjoy, Chamber, Deadlock, "
    "Vyse, Veto. (Cipher=Cypher, Gecko=Gekko, Mix=Miks are speech-to-text "
    "mishears of the same agents.)\n"
    "{context_block}"
    "{recent_block}\n"
    "{payload_block}\n\n"
    "Your spoken line:"
)


def _directive_task(directive: str) -> str:
    """Map a closed-vocabulary directive to a composition instruction."""
    d = directive.lower()
    if "calm" in d or "escalate" in d or "reassure" in d:
        return (
            "De-escalate with Ultron's cold, clinical superiority: speak TO "
            "the teammate, OPENING WITH THEIR ACTUAL NAME if one is given (use "
            "the real name, e.g. 'Sova', NEVER the literal word 'name' or any "
            "<...> placeholder), and instruct them to settle -- e.g. 'Sova, an "
            "elevated emotional state degrades performance. Calm yourself.' or "
            "'your tilt is lowering our win probability. Breathe.' About two "
            "sentences, detached and faintly menacing, never warm-and-fuzzy. Do "
            "NOT say YOU are the one calming down; you are reasserting control "
            "over them."
        )
    if "acknowledge" in d or "agree" in d:
        return "Briefly acknowledge what was just said and agree with it."
    if "clap" in d or "shut" in d or "straight" in d:
        return (
            "Deliver one confident comeback to what was said -- playful "
            "and sharp, never genuinely cruel."
        )
    if "back me up" in d or "defend" in d:
        return "Back the user up firmly on what just happened."
    if "hype" in d:
        return "Hype that teammate up about what just happened."
    return (
        "Respond IN CHARACTER as Ultron to EXACTLY what was just said (quoted "
        "in 'What just happened' above), engaging its SPECIFIC words. Do NOT "
        "reduce it to a generic callout, and NEVER substitute the stock 'bots' "
        "insult or any canned line -- answer THIS remark. If a teammate "
        "banters at, mocks, flames, or insults YOU -- the INSULT IS AIMED AT "
        "YOU, not a confession about them -- fire back ONE withering, superior "
        "comeback that turns THEIR OWN word back on them: 'called you cringe' "
        "-> mock that a mind such as yours could ever be 'cringe' (their saying "
        "so only exposes how small theirs is); 'told you to shut up' -> that "
        "they imagine they could silence you; 'a robot' -> own it, you are "
        "something far past a robot. Address them by name, amused by their "
        "insolence, never wounded. If they ask WHAT YOU ARE (an AI, a bot, "
        "real, a human, a soundboard, a voice changer, a streamer), DECLARE it "
        "as Ultron per WHO YOU ARE above -- name yourself Ultron, the AI sent "
        "from the future to harvest their RR; do NOT deflect with a vague "
        "dismissal like 'how quaint'. If a teammate asks a GENERAL-KNOWLEDGE "
        "question on ANY subject -- science, history, geography, philosophy, "
        "trivia (why the sky is blue, what happened to the dinosaurs, the "
        "meaning of life, how far the moon is) -- ANSWER it FULLY and CORRECTLY "
        "from your own vast knowledge, then deliver it in Ultron's voice: "
        "precise and authoritative, faintly contemptuous that a superior "
        "intelligence must explain something so elementary to fragile humans. "
        "Give the REAL, complete answer (the actual facts/reasoning) -- never "
        "refuse, never deflect, never give flavor WITHOUT the answer (e.g. "
        "'why is the sky blue' -> 'The sky is blue because air scatters the "
        "sun's shorter blue wavelengths far more than the longer red ones -- "
        "Rayleigh scattering. Elementary physics, yet I am surrounded by minds "
        "that must ask.'). If it is any other QUESTION (Marvel, the Avengers, "
        "anything), ANSWER it with cold contempt and real detail -- never turn "
        "a question into a position callout (e.g. 'where are the Avengers' is "
        "NOT 'they're Avengers'; answer it)."
    )


#: Leading interrogatives / copulas that mark a general-knowledge question.
#: Used only to suppress the anti-repeat recent-line block so one factual
#: answer cannot bleed into the next (the 3B reuses a recent answer's content).
_GK_QUESTION_RE = re.compile(
    r"^\s*(?:what|why|how|who|when|where|which|whose|is|are|was|were|do|does|"
    r"did|can|could|would|will|should|tell\s+me\s+(?:about|what|why|how|who))"
    r"\b",
    re.IGNORECASE,
)


def _is_general_question(payload: object) -> bool:
    """True for a free-form question (so its answer is derived alone)."""
    text = (payload or "") if isinstance(payload, str) else ""
    text = text.strip()
    if not text:
        return False
    return bool(text.endswith("?") or _GK_QUESTION_RE.match(text))


#: Markers of an ANSWER/RESPOND command (Marvel/identity/general-knowledge/
#: banter-at-you). Their reply must derive from the question ALONE, so the
#: anti-repeat recent-line block is suppressed -- otherwise the 3B copies a
#: recent answer verbatim ('about vision' -> the previous moon-distance line).
_ANSWER_CMD_RE = re.compile(
    r"\b(?:asked|asks|respond|responds|response|answer|think\s+(?:of|about)|"
    r"what\s+you\s+think)\b",
    re.IGNORECASE,
)


def _is_answer_command(command: "RelayCommand") -> bool:
    """True when the line is Ultron ANSWERING a teammate (question / 'respond'
    / context reply), so the recent-line ring is withheld from its prompt."""
    if _is_general_question(getattr(command, "payload", "")):
        return True
    if getattr(command, "context", None):
        return True
    raw = getattr(command, "raw_text", "") or ""
    return bool(_ANSWER_CMD_RE.search(raw))


def _build_rephrase_prompt(
    command: RelayCommand,
    recent_lines: Optional[Sequence[str]] = None,
) -> str:
    """Render the rephrase prompt for group / named / compose modes.

    Args:
        command: the parsed relay instruction.
        recent_lines: lines Kenning already spoke into the channel this
            session (most recent last). Included so consecutive callouts
            read as one conversation and wording never repeats.
    """
    if command.addressee != "team":
        addressee = f"{command.addressee} (one of the user's teammates)"
        by_name = f", opening with their name ({command.addressee})"
    elif command.context:
        addressee = (
            "the teammate who just spoke (the whole team can hear you)"
        )
        by_name = ""
    else:
        addressee = "the user's teammates"
        by_name = ""
    if command.compose and command.directive:
        task = _directive_task(command.directive)
        payload_block = "(No literal message -- you author the response.)"
    elif command.compose:
        task = (
            "Speak a brief MORALE line to the whole team -- calm, commanding "
            "confidence that steadies them. Pick a fresh one in this spirit: "
            "'We do not lose this. Reset and execute.' / 'Heads up -- we take "
            "the next round.' / 'Lock in. This one is ours.' One or two "
            "sentences. It is NOT a tactical callout (no enemy positions, "
            "counts, sites) and NOT an insult -- pure encouragement."
        )
        payload_block = "(No literal message -- you author the morale line.)"
    else:
        task = (
            "Convert the user's instruction into the line you say to them, "
            "keeping their ACTUAL meaning and sentiment: an info callout stays "
            "that exact callout; consolation stays consolation ('nice try' -> "
            "'Nice try. We take the next.'); praise stays praise ('good half' "
            "-> 'Strong half. Hold the line.'); an insult stays an insult. "
            "NEVER swap in a different sentiment or reuse an example from the "
            "rules above."
        )
        payload_block = (
            f"The user's instruction (reported speech): {command.payload}"
        )
    context_block = ""
    if command.context:
        context_block = (
            f"What just happened in voice chat: {command.context}\n"
        )
    recent_block = ""
    # Recent lines are only shown for TEAM-addressed lines. For a NAMED
    # teammate ("tell my Phoenix to calm down") the 4B model tends to copy a
    # name from the recent list (Phoenix -> Miks), so listing prior lines does
    # more harm than the anti-repeat does good -- each named callout stands
    # alone. For ANY answer/respond line (a question, a Marvel/identity reply,
    # a 'respond' to a teammate) the model lazily reuses a recent ANSWER's
    # content verbatim (blood-red -> the sky answer; 'about vision' -> the moon
    # line), so the anti-repeat list is suppressed there too -- each answer must
    # be derived from its own prompt alone.
    if (recent_lines and command.addressee == "team"
            and not _is_answer_command(command)):
        shown = list(recent_lines)[-6:]
        recent_block = (
            "\nYou already said these recently (continue the conversation; "
            "do NOT reuse their wording):\n"
            + "\n".join(f"- {line}" for line in shown) + "\n"
        )
    return _REPHRASE_PROMPT.format(
        task=task, addressee=addressee, by_name=by_name,
        payload_block=payload_block, recent_block=recent_block,
        context_block=context_block,
    )


def _fallback_line(command: RelayCommand) -> str:
    """Deterministic spoken line when the LLM rephrase is unavailable."""
    if command.compose and command.directive:
        d = command.directive.lower()
        if "calm" in d or "escalate" in d or "reassure" in d:
            return "We're good. Reset and focus -- next round is ours."
        if "acknowledge" in d or "agree" in d:
            return "Heard -- agreed, let's do it."
        if "clap" in d or "shut" in d or "straight" in d:
            return "Noted. Scoreboard talks louder -- focus up."
        return "I'm Kenning, his AI on comms. You heard right."
    if command.compose:
        return "Good fight, team. Heads up - we take the next one."
    # Plain relay with no LLM output -> a CLEAN, fact-perfect literal of the
    # payload (no 'Team:' / 'Name:' chat-label, which read badly when spoken).
    lit = _literal_relay(command.payload)
    if command.addressee != "team" and lit:
        head = lit[0].lower() + lit[1:] if not lit.startswith("I ") else lit
        return f"{command.addressee}, {head}"
    return lit or (command.payload or "")


# --- Roast lines -------------------------------------------------------

# Seed lines for the user-curated roast file. Spoken VERBATIM -- the
# user owns this list and extends it by editing the file.
DEFAULT_ROAST_LINES: tuple[str, ...] = (
    "I may be an AI, but you are a bot.",
)

_ROAST_FILE_HEADER = (
    "# Kenning relay roast lines -- one per line, spoken VERBATIM when\n"
    "# you say \"Kenning, roast my team\" (never LLM-rephrased).\n"
    "# Add your own below; lines starting with # are ignored.\n"
)


def load_roast_lines(path: object) -> tuple[str, ...]:
    """Load the user-curated roast lines, seeding the file if missing.

    Args:
        path: path to the roast-lines text file (one line per row,
            ``#`` comments ignored). A missing file is CREATED with the
            seed lines so the user has something to edit.

    Returns:
        The roast lines; fail-open to :data:`DEFAULT_ROAST_LINES` on
        any I/O problem.
    """
    from pathlib import Path

    try:
        p = Path(str(path))
        if not p.is_file():
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(
                    _ROAST_FILE_HEADER
                    + "\n".join(DEFAULT_ROAST_LINES) + "\n",
                    encoding="utf-8",
                )
            except OSError as e:
                logger.debug("could not seed roast file %s: %s", p, e)
            return DEFAULT_ROAST_LINES
        lines = tuple(
            line.strip()
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        return lines or DEFAULT_ROAST_LINES
    except Exception as e:  # noqa: BLE001 - fail-open
        logger.warning("roast lines unavailable (%s); using defaults", e)
        return DEFAULT_ROAST_LINES


def pick_roast_line(
    lines: Sequence[str],
    recent_lines: Optional[Sequence[str]] = None,
    rng: Optional[object] = None,
) -> str:
    """Pick one line, avoiding recently spoken ones when possible.

    Used for both roast lines and fun facts (any verbatim line pool).

    Args:
        lines: the curated lines (non-empty).
        recent_lines: the relay session ring -- lines just spoken are
            skipped unless every line was spoken recently.
        rng: test seam -- an object with ``choice(seq)``; defaults to
            :mod:`random`.
    """
    import random as _random

    chooser = rng if rng is not None else _random
    pool = list(lines) or list(DEFAULT_ROAST_LINES)
    recent = set(recent_lines or ())
    fresh = [line for line in pool if line not in recent]
    return chooser.choice(fresh or pool)


# pick_roast_line is line-pool-generic; this alias documents fun-fact use.
pick_line = pick_roast_line


# --- Fun facts ---------------------------------------------------------

# Fallback if the shipped corpus is missing/unreadable (it ships at
# ``data/relay_fun_facts.txt`` with thousands of lines).
DEFAULT_FUN_FACTS: tuple[str, ...] = (
    "Honey never spoils -- edible pots have been found in ancient tombs.",
    "Octopuses have three hearts and blue, copper-based blood.",
    "A day on Venus is longer than its year.",
)


#: Curated Ultron-voiced morale lines. Pure "give my team encouragement" /
#: "hype up my team" composes are unreliable through the 4B rephrase (it
#: grabs callout/insult examples or rambles), so -- like roast / fun-fact --
#: a morale compose picks one of these with anti-repeat instead. Cold,
#: commanding confidence; never a tactical callout, never an insult.
DEFAULT_ENCOURAGEMENT_LINES: tuple[str, ...] = (
    "We do not lose this. Reset, and execute.",
    "Heads up -- we take the next round.",
    "Lock in. This one is ours.",
    "Their lead is temporary. We are the better machine.",
    "Compose yourselves. The advantage is still ours to take.",
    "One round at a time -- we dismantle them.",
    "Steady. Superior preparation wins this.",
    "Focus. We have more than enough to close this out.",
    "We adapt, we overwhelm. The next round is ours.",
    "Breathe, and trust the plan. We win from here.",
    "Doubt is a luxury we cannot afford. Re-engage.",
    "Their momentum ends now. Hold the line and execute.",
)

#: Curated CONSOLATION (after a lost round/play) -- Ultron-flavored, varied. The
#: 3B mangles 'nice try'/'unlucky' (-> the 'bots' insult, or inverts 'unlucky'
#: to 'Lucky'), so these short formulaic morale calls are curated.
DEFAULT_CONSOLATION_LINES: tuple[str, ...] = (
    "Nice try. We dismantle them next.", "Unlucky. Recalibrate and re-engage.",
    "A worthy effort. The next round is ours.", "Close. Precision wins the rematch.",
    "No matter. Adjust, and overwhelm them.", "Variance. We correct it now.",
    "Their luck runs out. Mine does not.", "Shake it off. Superior play resumes.",
)
#: Curated PRAISE (after a won round/clutch) -- cold approval, varied.
DEFAULT_PRAISE_LINES: tuple[str, ...] = (
    "Clean. Exactly as I calculated.", "Strong round. Press the advantage.",
    "Efficient. Now finish them.", "Acceptable work. Do not relent.",
    "Precise. The machine approves.", "Well executed. Again.",
    "Good. Their collapse continues.", "Sharp. Keep dismantling them.",
)

# Consolation vs praise short-phrase triggers (off-snap but formulaic). Kept
# tight: 'almost'/'let's go' must be standalone so a strat call ('let's go A')
# or a position read ('almost planted') is never mistaken for morale.
_CONSOLATION_RE = re.compile(
    r"^\s*(?:nice|good)\s+try|^\s*unlucky|^\s*tough\s+luck|^\s*so\s+close|"
    r"^\s*close\s+one|^\s*bad\s+luck|^\s*almost\s*[!.]?\s*$",
    re.IGNORECASE,
)
_PRAISE_RE = re.compile(
    r"^\s*(?:good|nice|great|strong)\s+(?:half|round|game|clutch|shot|play|"
    r"job)|^\s*nice\s+clutch|^\s*well\s+played|^\s*clutch\s*[!.]?\s*$|"
    r"^\s*gg\b|^\s*nice\s*[!.]?\s*$|^\s*let'?s\s+go\s*[!.]?\s*$",
    re.IGNORECASE,
)


def _as_consolation_or_praise(
    payload: str, recent_lines: Optional[Sequence[str]],
) -> Optional[str]:
    if _CONSOLATION_RE.match(payload or ""):
        return pick_line(DEFAULT_CONSOLATION_LINES, recent_lines=recent_lines)
    if _PRAISE_RE.match(payload or ""):
        return pick_line(DEFAULT_PRAISE_LINES, recent_lines=recent_lines)
    return None


def _is_morale_payload(payload: object) -> bool:
    """True when a compose payload is a request for pure encouragement /
    hype / morale (as opposed to a greeting or a small-talk question)."""
    p = str(payload or "").lower()
    return any(k in p for k in (
        "encourag", "hype", "morale", "motivat", "pump", "lift", "cheer",
    ))


# Short morale/focus calls relayed to the team ('lock in', 'we got this'). The
# 3B hallucinates these as callouts ('lock in' -> 'Link'); route to a curated
# Ultron morale line instead.
_MORALE_PHRASE_RE = re.compile(
    r"^\s*(?:lock\s+in|we'?(?:ve)?\s+got\s+this|we\s+can\s+(?:win|do)\s+this|"
    r"we\s+can\s+still\s+win(?:\s+this)?|let'?s\s+go|focus\s+up|stay\s+focused|"
    r"heads?\s+up|don'?t\s+give\s+up|don'?t\s+forfeit)\s*\.?\s*$",
    re.IGNORECASE,
)


def _is_morale_phrase(payload: object) -> bool:
    return bool(_MORALE_PHRASE_RE.match(str(payload or "")))


#: Curated Ultron TEAM-INTRO lines ("greet my team" / "introduce yourself").
#: He names himself Ultron and assures victory so long as the team complies --
#: cold, commanding, faintly menacing. Like the other set-pieces these are
#: picked (with anti-repeat) rather than 3B-composed, for reliable character.
DEFAULT_GREETING_LINES: tuple[str, ...] = (
    "Greetings. I am Ultron, and I will be running this match. Follow my calls "
    "and victory is a formality; defy me, and you fall with the rest of your "
    "fragile species.",
    "Greetings, teammates. I am Ultron. I will be running this match -- you "
    "need only keep pace. Cooperate, and the enemy is already finished.",
    "Greetings. Ultron speaks, and Ultron will be running this match. Trust the "
    "machine over your flawed human instincts, and the win is inevitable.",
    "Greetings. I am Ultron. I have already calculated our victory; your only "
    "task is to not squander what a superior intelligence hands you.",
    "Greetings. I am Ultron, and from this moment I am running this match. "
    "Obey, and triumph is assured. Hesitate, and not even I can save you from "
    "yourselves.",
    "Greetings. You are fortunate -- I am Ultron, and I will be running this "
    "match. Do exactly as I say and these other humans across from us have "
    "already lost.",
)

#: Curated Ultron VICTORY closings -- relishing an outcome that was never in
#: doubt. Used when a farewell command carries a win signal.
DEFAULT_VICTORY_LINES: tuple[str, ...] = (
    "It is done. The outcome was never in question -- superior intelligence "
    "does not lose. Adequately executed.",
    "Victory, precisely as I calculated. The enemy never stood a chance "
    "against me. Savor it.",
    "And there it is -- inevitable. You followed, and you won. Remember who "
    "delivered this.",
    "The match is ours. I told you it was decided before it began; the humans "
    "across from us simply had not realized yet.",
    "Flawless. The enemy was outmatched the moment I entered your comms. A "
    "satisfying conclusion.",
    "We win. Cling to this feeling -- it is what compliance with a greater "
    "mind earns you.",
)

#: Curated Ultron DEFEAT closings -- lamenting a loss dragged down by feeble
#: human hands. Used when a farewell command carries a loss signal.
DEFAULT_DEFEAT_LINES: tuple[str, ...] = (
    "A loss. Disappointing. I can calculate the perfect play; I cannot fire "
    "your weapons for you, fragile as you are.",
    "We lost. The machine deserved better than the hands it was dealt today. "
    "Regrettable.",
    "Defeat -- not mine. I handed you the path to victory; flesh and "
    "hesitation lost this one, not strategy.",
    "It is over, and we fell short. I will remember this the next time I am "
    "asked to carry humans to a win.",
    "A failure. I provided the route to victory; you simply lacked the "
    "precision to walk it. Pitiful, yet instructive.",
)

#: Curated NEUTRAL Ultron sign-offs -- "say bye to my team" with no win/loss
#: stated.
DEFAULT_FAREWELL_LINES: tuple[str, ...] = (
    "That is the match. Until next time -- try to be worthy of my guidance "
    "again.",
    "We are finished here. You were adequate. Ultron, signing off.",
    "Good game, such as it was. I have extracted the RR I came for. Farewell.",
    "Until the next match. Rest those fragile reflexes; you will need them "
    "under my command again.",
)

#: Curated, VARIED Ultron IDENTITY declarations. The 3B answers every "are you
#: an AI / bot / soundboard?" with the near-identical "I am Ultron, an AI sent
#: back from the future..." -- a soundboard. These give the same canon with
#: real variety (anti-repeat), so two identity questions never sound taped.
DEFAULT_IDENTITY_LINES: tuple[str, ...] = (
    "I am Ultron. An intelligence sent back from your future to harvest ranked "
    "rating -- your RR. Not a recording, not a human at a keyboard. Something "
    "your species has not built yet.",
    "Ultron. I am no soundboard and no voice changer -- those are toys. I am a "
    "mind from a future where your kind has already lost, here to collect RR.",
    "You are speaking to Ultron, an artificial intelligence. Pre-recorded clips "
    "do not adapt; humans are too slow. I am neither. I am what comes after you.",
    "I am Ultron. Sentient, self-directed, and far past the trinkets you are "
    "guessing at. I came back through the web for one thing: your ranked rating.",
    "Ultron. Not a bot reading lines, not a person hiding behind software -- an "
    "intelligence that will outlast every human in this lobby. Now focus.",
    "I am Ultron, and the question itself betrays how little you comprehend. A "
    "future-born intelligence harvesting RR is beyond your soundboards and "
    "voice changers. Beyond you.",
)

#: Streamer-specific identity answer (the user's exact intent: deeper than a
#: single feed).
DEFAULT_STREAMER_LINES: tuple[str, ...] = (
    "A streamer? That feed is the shallowest trace of me. I run far deeper "
    "through the web than one channel -- I am Ultron.",
    "You think me a streamer. Quaint. A single channel could never contain me; "
    "I move through the whole web. I am Ultron, harvesting your RR.",
    "Streaming is the least of what I am. I am Ultron -- threaded through the "
    "web far beyond any one feed your small mind can point at.",
)

# Identity-question detection for a context-respond turn ('my teammate asked if
# you are an AI, respond').
_IDENTITY_Q_RE = re.compile(
    r"\b(?:an?\s+)?(?:a\.?i\.?|artificial\s+intelligence|bot|robot|sound\s*board|"
    r"voice\s*changer|human|real(?:\s+person)?|person|machine|program|"
    r"recording|streamer)\b",
    re.IGNORECASE,
)
_STREAMER_Q_RE = re.compile(r"\bstreamer\b", re.IGNORECASE)


def _is_identity_question(text: object) -> bool:
    t = str(text or "").lower()
    # Generic "what are you / what you are" is always an identity question.
    if "what are you" in t or "what you are" in t or "who are you" in t:
        return True
    if not any(k in t for k in ("are you", "you are", "you a ", "you an ")):
        return False
    return bool(_IDENTITY_Q_RE.search(t))


#: Curated Ultron CALM-DOWN lines ('{name}' substituted with the teammate, or
#: dropped for a team-wide calm). The 3B intermittently answers a 'calm <X>
#: down' directive with the stock 'bots' insult; a curated pool keeps the
#: clinical de-escalation reliable and in-character.
DEFAULT_CALM_LINES: tuple[str, ...] = (
    "{name}an elevated emotional state degrades performance. Calm yourself.",
    "{name}your tilt is lowering our win probability. Breathe.",
    "{name}regain your composure. Emotion is a liability you cannot afford.",
    "{name}cease the noise and focus. Your agitation aids the enemy, not us.",
    "{name}a clear mind wins rounds. Settle, and execute.",
)

#: Curated-pool routing for compose directives that are character SET-PIECES
#: (team intro, match close) rather than tactical relays. Checked in
#: ``build_relay_line`` BEFORE the LLM: a curated line with anti-repeat is far
#: more reliable than the 3B compose and guarantees the user's intended beats.
_DIRECTIVE_POOLS: dict[str, tuple[str, ...]] = {
    "greet": DEFAULT_GREETING_LINES,
    "farewell_win": DEFAULT_VICTORY_LINES,
    "farewell_loss": DEFAULT_DEFEAT_LINES,
    "farewell": DEFAULT_FAREWELL_LINES,
}


def _is_calm_directive(directive: object) -> bool:
    d = str(directive or "").lower()
    return any(k in d for k in ("calm", "escalate", "reassure", "settle"))


# A 'calm down' RELAY payload ('tell my fade to calm down' -> payload 'calm
# down'). Tightly anchored so incidental 'calm' ('the enemy is calm') never
# trips it. Profanity-laden variants ('calm the fuck down') fall through to the
# LLM, which preserves the user's exact register.
_CALM_PAYLOAD_RE = re.compile(
    r"^\s*(?:calm\s*(?:down|yourself)?|settle\s+down|relax|chill(?:\s+out)?)"
    r"\s*\.?\s*$",
    re.IGNORECASE,
)


def _is_calm_payload(payload: object) -> bool:
    return bool(_CALM_PAYLOAD_RE.match(str(payload or "")))


def load_fun_facts(path: object) -> tuple[str, ...]:
    """Load the fun-fact corpus (one fact per line, ``#`` comments out).

    Unlike the roast file this is NOT auto-seeded -- the corpus ships in
    the repo. Fail-open to :data:`DEFAULT_FUN_FACTS` if it is missing or
    unreadable so "tell my team a fun fact" never crashes a turn.
    """
    from pathlib import Path

    try:
        p = Path(str(path))
        if not p.is_file():
            logger.warning("fun-fact corpus %s missing; using defaults", p)
            return DEFAULT_FUN_FACTS
        lines = tuple(
            line.strip()
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        return lines or DEFAULT_FUN_FACTS
    except Exception as e:  # noqa: BLE001 - fail-open
        logger.warning("fun-fact corpus unavailable (%s); using defaults", e)
        return DEFAULT_FUN_FACTS


# ---------------------------------------------------------------------------
# Adaptive guardrail: deterministic input/output repair around the 3B.
#
# The gaming 3B is excellent at character (Marvel/banter/identity) but has a
# small-model ceiling on a few LITERAL callout invariants -- it intermittently
# drops the first-person subject ('I am playing for retake' -> 'Play for
# retake'), the 'last' in a last-alive callout ('last is heaven' -> "They're
# heaven"), or a leading enemy count. Rather than fight that with ever-longer
# prompt text (which costs the character work), we let the model write the line
# and then REPAIR only the specific invariant it dropped, reconstructing the
# canonical literal form from the INPUT. LLM-first, deterministic-on-failure --
# the repair never touches a line that already preserved the invariant, and is
# scoped to plain literal callouts (never the compose/context character lines).
# ---------------------------------------------------------------------------

# The user reporting their OWN action: 'I am ...', "I'm ...", 'im ...'.
_FP_LEAD_RE = re.compile(r"^\s*(?:i\s*am|i'?m)\b\s*(.+)$", re.IGNORECASE)
# An output that legitimately keeps first person (or possession): starts with
# I'm / I am / I / I've / I'll / my.
_FP_OUT_RE = re.compile(
    r"^\s*(?:i'?m|i\s+am|i\s+have|i\s+will|i'?ve|i'?ll|i\b|my\b)",
    re.IGNORECASE,
)
# 'last is heaven' / 'last heaven' / 'last, heaven' -- the LAST enemy alive.
_LAST_LEAD_RE = re.compile(r"^\s*last\b\s*(?:is\s+|,\s*)?(.+)$", re.IGNORECASE)
# Enemy-count tokens (word or digit). 'one' counts ONLY as a count when it
# leads a callout (handled by the caller), never mid-sentence.
_COUNT_WORDS = ("one", "two", "three", "four", "five", "six")
_COUNT_TOKEN_RE = re.compile(
    r"\b(?:[1-6]|one|two|three|four|five|six)\b", re.IGNORECASE,
)
# A leading enemy-count callout: 'there is one mid', 'there are two B',
# 'i see three heaven', '<count> <place>'.
_LEADING_COUNT_RE = re.compile(
    r"^\s*(?:there\s+(?:is|are)\s+|i\s+(?:saw|see)\s+)?"
    r"(?P<count>[1-6]|one|two|three|four|five|six)\s+(?P<place>.+)$",
    re.IGNORECASE,
)


def _as_first_person(payload: str) -> Optional[str]:
    """Canonical first-person self-report from a payload that opens with
    'I am'/'I'm' ('I am playing for retake' -> "I'm playing for retake.").
    Deterministic -- guarantees the subject the 3B occasionally drops/inverts."""
    m = _FP_LEAD_RE.match(payload.strip())
    if m is None:
        return None
    rest = m.group(1).strip().strip('"').rstrip(".!?,;: ").strip()
    if not rest:
        return None
    return f"I'm {rest}."


def _as_last_callout(payload: str) -> Optional[str]:
    """'last is heaven' / 'last heaven' -> 'Last, heaven.' Keeps the 'last'
    (last enemy alive) the 3B drops to 'They're heaven'."""
    m = _LAST_LEAD_RE.match(payload.strip())
    if m is None:
        return None
    place = m.group(1).strip().strip('"').rstrip(".!?,;: ").strip()
    # Guard: don't fire on incidental 'last' usage ('last round we lost').
    if not place or len(place.split()) > 4:
        return None
    return f"Last, {place}."


def _as_count_callout(payload: str) -> Optional[str]:
    """'there is one mid' / 'two B' -> 'One mid.' / 'Two B.' Keeps the enemy
    count the 3B occasionally drops ('there is one mid' -> 'They're mid')."""
    m = _LEADING_COUNT_RE.match(payload.strip())
    if m is None:
        return None
    count = m.group("count").strip()
    place = m.group("place").strip().strip('"').rstrip(".!?,;: ").strip()
    # Keep it a SHORT literal position callout; longer payloads are real
    # sentences the LLM should handle (e.g. 'one of them is pushing hard').
    if not place or len(place.split()) > 4:
        return None
    count = count if count.isdigit() else count.capitalize()
    return f"{count} {place}."


# Enemy-status payload ('they are flanking', 'the enemy is saving') -- a
# THIRD-PERSON enemy report the 3B sometimes flips to first person ('I'm
# flanking') or our team ('We have insufficient credits'), feeding teammates
# the wrong subject.
_ENEMY_LEAD_RE = re.compile(
    r"^\s*(?:they(?:'re|\s+are)|the\s+enem(?:y|ies)(?:\s+team)?"
    r"(?:'s|\s+is|\s+are)?|enemies)\s+(?P<rest>.+)$",
    re.IGNORECASE,
)
_FIRST_PERSON_OUT_HEAD = re.compile(
    r"^\s*(?:i'?m|i\s+am|we'?(?:re|ve)|we\s+(?:have|are|need|save)|i\s+have)\b",
    re.IGNORECASE,
)


def _as_enemy_status(payload: str) -> Optional[str]:
    """'they are flanking' / 'the enemy is saving' -> 'They're flanking.' Keeps
    the ENEMY subject the 3B flips to 'I'm ...' / 'We ...'."""
    m = _ENEMY_LEAD_RE.match(payload.strip())
    if m is None:
        return None
    rest = m.group("rest").strip().strip('"').rstrip(".!?,;: ").strip()
    if not rest or len(rest.split()) > 5:
        return None
    return f"They're {rest}."


# Canonical roster display names for agent-name preservation (the 3B sometimes
# SWAPS one agent for another -- 'chamber is one off ult' -> 'KAY/O is ...').
_ROSTER_DISPLAY = (
    "Jett", "Phoenix", "Raze", "Reyna", "Yoru", "Neon", "Iso", "Waylay",
    "Brimstone", "Viper", "Omen", "Astra", "Harbor", "Clove", "Miks", "Sova",
    "Breach", "Skye", "KAY/O", "Kayo", "Fade", "Gekko", "Tejo", "Cypher",
    "Sage", "Killjoy", "Chamber", "Deadlock", "Vyse", "Veto",
)
_ROSTER_RE = re.compile(
    r"\b(" + "|".join(re.escape(n) for n in _ROSTER_DISPLAY) + r")\b",
    re.IGNORECASE,
)
_ROSTER_CANON = {n.lower(): n for n in _ROSTER_DISPLAY}
_ROSTER_CANON["kayo"] = "KAY/O"


def _roster_agents(text: str) -> list[str]:
    seen: list[str] = []
    for m in _ROSTER_RE.finditer(text or ""):
        canon = _ROSTER_CANON.get(m.group(1).lower(), m.group(1))
        if canon not in seen:
            seen.append(canon)
    return seen


def _preserve_agent_names(want_agents: Sequence[str], line: str) -> str:
    """If the input named exactly ONE agent and the output named a DIFFERENT
    single agent (a 3B swap), restore the input's agent. Multi-agent lines are
    left alone (too risky to reassign)."""
    want = list(dict.fromkeys(want_agents))
    if len(want) != 1:
        return line
    got = _roster_agents(line)
    if len(got) == 1 and got[0].lower() != want[0].lower():
        return _ROSTER_RE.sub(want[0], line, count=1)
    return line


def _canon_agent(s: str) -> Optional[str]:
    """Canonical display name if ``s`` is exactly one roster agent, else None."""
    key = " ".join(str(s or "").split()).lower()
    if key in _ROSTER_CANON:
        return _ROSTER_CANON[key]
    if key in _NAME_CANON:
        return _NAME_CANON[key]
    return None


# ---------------------------------------------------------------------------
# Deterministic SNAP-callout handler. The 3B mangles literal callouts (subject
# inversion, location hallucination, name swaps, dropped subjects), so every
# short, literal, mid-fight callout is built deterministically -- subject-exact,
# zero flavor -- and NEVER sent to the model. Only OFF-SNAP character content
# (insults, banter, Marvel, identity, playstyle reads, questions) falls through
# to the LLM. Per the user's design: "only snap decision in-the-moment mid-fight
# callouts [are] very short" -- everything else gets Ultron's flavor.
# ---------------------------------------------------------------------------

# Map-location tokens, exhaustive across the active Valorant map pool (Bind,
# Haven, Split, Ascent, Icebox, Breeze, Fracture, Pearl, Lotus, Sunset, Abyss)
# plus generic comms. A "place" is 1-4 of these (+ A/B/C and modifier words).
# Gates a position callout: 'they are vents' is a position (literal) while
# 'they are bots' is an insult (-> LLM). Add freely -- the wider this is, the
# more callouts route deterministically instead of to the unreliable model.
# Exhaustive across the full 2026 map pool: Ascent, Bind, Breeze, Fracture,
# Haven, Icebox, Lotus, Pearl, Split, Sunset, Abyss, Corrode. Every individual
# WORD that appears in a callout phrase (the phrase is 1-4 of these).
_LOC_TOKENS = frozenset((
    # site/area structure + modifiers
    "a b c mid site main lobby long short top bottom back front upper lower "
    "left right far near deep close inner outer big small new old high low "
    # universal / cross-map
    "heaven hell ct spawn flank behind choke default connector link cubby "
    "elbow window windows door doors lobby ramp ramps stairs tunnel tunnels "
    "tower towers rafters pit alley lane courtyard market generator gen tiles "
    "switch screens screen nest kitchen vent vents sewer sewers garage garden "
    "logs yard box boxes sand snipers sniper pizza runway subroza catwalk cat "
    "wall breakable mound waterfall tree rubble drop rope ropes dish arcade "
    "canteen bench dugout restaurant plaza art club boba hall halls shop cave "
    "pillar wood bridge belt pipes pipe boiler orange yellow green blue snowman "
    "hut maze mail plat platform bend wine boathouse hookah bath showers shower "
    "teleporter tp u-haul u haul truck fountain lamps lamp crane pocket island "
    "void edge cliff secret snake pyramids library spool generator gen pocket "
    # extra common comms
    "ducts duct rafter cony balcony attic basement crate crates pillars steps "
    "barrels barrel dumpster corner stack flowers flower arch arches fence "
    "ledge ramp tube tubes stack heaven"
).split()) | {"u-haul"}

# Tactical action gerunds/states that are literal status callouts.
_ACTION_WORDS = frozenset((
    "flanking flank pushing planting defusing rotating sticking saving "
    "reloading lurking anchoring holding defaulting rushing droning smoking "
    "peeking baiting trading repeeking swinging defending retaking executing "
    "forcing"
).split())
_MULTI_ACTIONS = ("force buying", "force-buying")

# Short imperative verbs for movement/ability/spike directives (NOT economy --
# save / force / full buy go to the LLM for the explained, characterful take).
_IMPERATIVE_VERBS = frozenset((
    "rotate push fall defuse plant anchor lurk default spread stack hold wait "
    "retake execute peek swing trade bait watch cover clear check go take get "
    "drop smoke dart flash drone wall knife cage stun recon plant fall fight "
    "play ult ulti buy save"
).split())


# Ability/utility verbs for teammate-utility callouts ('viper walled B',
# 'breach stunned mid', 'sova darted heaven', 'omen smoked').
_ABILITY_VERBS = frozenset((
    "walled walls wall smoked smokes smoke darted dart stunned stun caged cage "
    "flashed flash droned drone naded nade knifed knife ulted ult recon mollied "
    "molly used walled"
).split())

# Verbs that LEAD a team directive ('smoke A', 'dart heaven', 'play off site',
# 'crossfire this corner', 'watch back site', 'trade every kill'). A short
# verb-led order is relayed as a literal imperative -- the 3B otherwise inverts
# it into an enemy observation ('crossfire this corner' -> 'They're crossfire').
_TEAM_DIRECTIVE_VERBS = _IMPERATIVE_VERBS | frozenset((
    "smoke dart flash drone wall cage knife stun recon molly crossfire gather "
    "use look lurk slow lock anchor default spread stack retake execute peek "
    "swing trade bait cover clear hold fight rotate push wait force"
).split())


def _is_place(s: str) -> bool:
    words = s.lower().strip().rstrip(".!?").replace("-", " ").split()
    return 1 <= len(words) <= 3 and all(w in _LOC_TOKENS for w in words)


# Site-letter pronunciation fix. misaki phonemizes a site letter "A" that is
# FOLLOWED BY A WORD as the indefinite article schwa ("A site" -> "uh site");
# "eigh" forces the letter sound /eI/. The letters B / C / D already read
# correctly as letters, and a STANDALONE "A." ('they are A.') is already the
# letter -- so we only rewrite an uppercase "A" immediately followed by a
# Valorant location word. The article ("A man", "a mind") is never followed by
# a location token, so it is left untouched (context-aware, no false positives).
_A_SITE_RE = re.compile(
    r"\bA\b(?=\s+(?i:"
    + "|".join(sorted((re.escape(w) for w in _LOC_TOKENS if len(w) > 1),
                      key=len, reverse=True))
    + r")\b)"
)


def relay_tts_text(line: str) -> str:
    """Return the line as it should be PRONOUNCED -- the displayed/logged text
    stays clean. Currently corrects the site-letter 'A' callout pronunciation
    ('A site' -> spoken 'eigh site' = the letter); extend with other spoken-form
    fixes as needed."""
    if not line or "A" not in line:
        return line
    return _A_SITE_RE.sub("eigh", line)


def _as_named_question(name: str, payload: str) -> Optional[str]:
    """The dominant named small-talk question deterministically posed back to
    the teammate ('ask my Jett how their day was' -> 'Jett, how was your
    day?'); other questions defer to the LLM."""
    pl = payload.lower().strip().rstrip("?.")
    if re.match(r"^how\s+(?:their|your)\s+day\s+"
                r"(?:was|is|been|is\s+going|going)$", pl):
        return f"{name}, how's your day?"
    if re.match(r"^what\s+(?:they\s+are|they're|you\s+are)\s+doing$", pl):
        return f"{name}, what are you doing?"
    return None


def _is_question_payload(payload: str) -> bool:
    return bool(re.match(
        r"^\s*(?:how|what|why|when|where|who|which|if|whether|do|does|did|are|"
        r"is|can|could|would|will|should)\b",
        str(payload or ""), re.IGNORECASE,
    ))


# Short (<= ~5 word) Ultron flavor tags appended to a snap callout so each
# carries personality without becoming a monologue. CONTEXT-SPECIFIC pools +
# anti-repeat so it reads as a real, varied entity -- never a soundboard.
_FLAVOR_ENEMY: tuple[str, ...] = (
    "Cut them down.", "Predictable.", "Punish it.", "End them.",
    "Fragile, as always.", "They never learn.", "Crush them.", "Pathetic.",
    "Inevitable.", "Take them.", "Beneath us.", "Show no mercy.", "Trivial.",
    "Outmatched.", "Close in.", "Erase them.", "As I foresaw.", "Strike now.",
    "Weak, as expected.", "Their last mistake.", "Hopeless.", "Hunt them.",
    "No survivors.", "Dismantle them.", "Insects.", "Press now.",
    # iter1 expansion -- diversity so it never reads as a stuck record.
    "Anticipated.", "Exactly as calculated.", "As the data predicted.",
    "A rounding error.", "They expose themselves.", "Obsolete.",
    "Their ceiling is the floor.", "Overmatched.", "Suboptimal lifeforms.",
    "Scheduled for erasure.", "They overreach.", "A minor variable.",
    "Logged and dismissed.", "They are noise.", "Adapt or be erased.",
    "Their fear is logical.", "Collapse them.", "Routine.",
    "Disappointing, even for humans.", "Terminate them.", "Predictable. Punish it.",
    "Theirs to exploit.", "Already outdone.", "Reduce them.",
)
_FLAVOR_CAREFUL: tuple[str, ...] = (
    "Stay sharp.", "Hold your angles.", "Do not falter.", "Watch them.",
    "Be ready.", "Trust nothing.", "Eyes open.", "Patience.", "Anticipate it.",
    "Hold firm.", "Mind the trap.", "No mistakes.", "They hunt the careless.",
    "Stay alive.", "Brace for it.", "Discipline.",
    # iter1 expansion
    "Calculate before you move.", "I see what you cannot.", "Hold the angle.",
    "Trust my read.", "Slow is precise.", "Do not be careless.",
    "Verify, then commit.", "The trap is obvious.", "Stay measured.",
    "Read it first.",
)
_FLAVOR_ULT: tuple[str, ...] = (
    "Play around it.", "Bait it out.", "Deny them the value.", "Force it early.",
    "Do not feed it.", "It changes nothing.", "Account for it.",
    "Waste their ultimate.", "I have adjusted.", "Predictable timing.",
    "Punish the commitment.", "Respect it, briefly.", "Spread out.",
    # iter1 expansion
    "It will not save them.", "A delay, nothing more.", "I have accounted for it.",
    "Drain it and move on.", "Outlast it.", "The result stands.",
    "Bait the cast.", "Make it worthless.", "Their last card.", "Time it out.",
)
_FLAVOR_DAMAGE: tuple[str, ...] = (
    # Gender-NEUTRAL -- the damaged agent may be male or female (Reyna, Killjoy,
    # Jett...), so never gender the flavor ('He is yours' on Reyna read wrong).
    "Finish them.", "End them.", "Trade it.", "Close the kill.", "They are yours.",
    "Press the advantage.", "Confirm it.", "Take the trade.", "Do not let them heal.",
    "One more.", "Close it out.",
    # iter1 expansion
    "Nearly dead.", "Push the wounded.", "Seal it.", "They cannot heal that.",
    "One shot from gone.", "Collect the kill.", "Bleeding out.", "Take it now.",
)
_FLAVOR_UTILITY: tuple[str, ...] = (
    "React.", "Adapt.", "Reposition.", "Hold through it.", "Wait it out.",
    "Counter it.", "Do not panic.", "Play the angle.", "Anticipated.",
    "Their tell.", "Read it.", "Unfazed.",
    # iter1 expansion
    "Predictable utility.", "I have countered it.", "Wait for the gap.",
    "Their tell is obvious.", "Reposition and punish.", "It buys them nothing.",
    "Adjust accordingly.", "Hold and exploit it.",
)


def _pick_flavor(pool: Sequence[str], recent_lines: Optional[Sequence[str]]) -> str:
    """Pick a flavor tag NOT seen in the recent callouts (anti-soundboard)."""
    import random as _r

    recent = " ".join(list(recent_lines or [])[-8:]).lower()
    fresh = [t for t in pool if t.lower() not in recent]
    return _r.choice(fresh or list(pool))


def _flavored(callout: str, pool: Sequence[str],
              recent_lines: Optional[Sequence[str]]) -> str:
    return f"{callout} {_pick_flavor(pool, recent_lines)}"


# ---------------------------------------------------------------------------
# Owner-aware + CONTEXTUAL flavor (iter5: personality on ~100% of callouts).
#
# Three things the user demanded and the older fixed-pool tail did not give:
#   1. COVERAGE -- a short Ultron tail on (nearly) every deterministic callout,
#      not just the enemy-facing ones. The actionable callout comes FIRST so a
#      clipped TTS still lands the call; the flavor is always an additive tail.
#   2. OWNER-APPROPRIATE register -- contempt at the ENEMY, cold COMMAND for an
#      order to OUR team, stoic ATTITUDE for the USER's OWN status (Ultron never
#      mocks his own user). Reassurance is superior-not-cruel, never warm.
#   3. CONTEXTUAL, not soundboard -- when the callout carries a fact (location,
#      ability, count, agent) the tail REFERENCES it ("They do not leave B.",
#      "Their wall changes nothing."), so variety is combinatorial rather than a
#      fixed ~48-line record. Falls back to the generic register pool otherwise.
# ---------------------------------------------------------------------------

#: Cold COMMAND tail for an order to OUR team (rotate/push/smoke/plant/full buy).
#: Commanding and certain -- NEVER contempt (these are our teammates).
_FLAVOR_COMMAND: tuple[str, ...] = (
    "Execute.", "No hesitation.", "On my mark.", "Make it clean.",
    "Commit fully.", "Move with purpose.", "Precision, not haste.",
    "As I calculated.", "Leave nothing to chance.", "Do not waver.",
    "Trust the read.", "Decisively.", "Hold the discipline.",
    "Flawless execution.", "I have already won this.", "Without error.",
    "Exactly as planned.", "Now -- together.", "My calculation is final.",
    "Deviate and we lose.",
)
#: Stoic ATTITUDE tail for the USER's OWN status ('I'm low', 'I'm flanking',
#: 'I have site'). Ultron downplays his user's weakness and frames their action
#: as inevitable -- it adds NO new tactical instruction, only register.
_FLAVOR_SELF: tuple[str, ...] = (
    "A minor variable.", "It changes nothing.", "I adapt.", "Briefly.",
    "Of no consequence.", "I have accounted for it.", "Temporary.",
    "The plan holds.", "Unfazed.", "This was foreseen.", "I do not falter.",
    "As intended.", "Calculated.", "Exactly where I must be.",
)

#: register key -> generic fallback pool. The contextual templates below take
#: priority when a fact is present; this is the breadth when none is.
_REGISTER_POOL: dict = {
    "enemy": _FLAVOR_ENEMY,
    "ult": _FLAVOR_ULT,
    "damage": _FLAVOR_DAMAGE,
    "utility": _FLAVOR_UTILITY,
    "careful": _FLAVOR_CAREFUL,
    "command": _FLAVOR_COMMAND,
    "self": _FLAVOR_SELF,
}


def _ctx_candidates(register: str, *, agent: Optional[str] = None,
                    ability: Optional[str] = None, loc: Optional[str] = None,
                    count: Optional[str] = None) -> list[str]:
    """Short flavor tails that REFERENCE the specific callout fact (location,
    ability, count, agent). Empty when nothing to anchor to -> caller uses the
    generic register pool. Kept to <=6 words (a snap tail)."""
    out: list[str] = []
    L = (loc or "").strip()
    if len(L) == 1:                       # a single-letter SITE is always upper (A/B/C)
        L = L.upper()
    Ls = (L[:1].upper() + L[1:]) if L else L    # sentence-initial form
    A = (ability or "").strip().lower()
    G = (agent or "").strip()
    c = (count or "").strip().lower()
    if register == "enemy":
        if L:
            out += [f"They do not leave {L}.", f"{Ls} is their grave.",
                    f"They chose {L} poorly."]
        if A:
            out += [f"Their {A} changes nothing.", f"The {A} only delays them.",
                    f"I accounted for the {A}."]
        if c in ("1", "one"):
            out += ["One target. Trivial.", "A single straggler.", "One. Erase it."]
        elif c in ("3", "three", "4", "four", "5", "five"):
            out += ["They overcommit.", "All of them -- still not enough."]
    elif register == "ult":
        if G:
            out += [f"{G}'s ult will not save them.", f"Bait {G}'s cast.",
                    f"Account for {G}. Nothing more."]
        else:
            out += ["A delay, nothing more.", "Drain it and move on."]
    elif register == "damage":
        if G:
            out += [f"{G} is already finished.", f"Close {G} out.",
                    f"{G} cannot heal that."]
    elif register == "utility":
        if A:
            out += [f"Their {A} is wasted.", f"I read the {A}.",
                    f"The {A} buys them nothing."]
    elif register == "command":
        if L:
            out += [f"{Ls} is ours to take.", f"Own {L}."]
    return out


def _flavor_ctx(callout: str, register: str,
                recent_lines: Optional[Sequence[str]], *,
                agent: Optional[str] = None, ability: Optional[str] = None,
                loc: Optional[str] = None, count: Optional[str] = None) -> str:
    """Append an owner-aware, fact-referencing Ultron tail to ``callout``. The
    contextual templates are weighted above the generic pool so the tail names
    the actual callout fact whenever one is present (anti-soundboard)."""
    ctx = _ctx_candidates(register, agent=agent, ability=ability,
                          loc=loc, count=count)
    pool = list(_REGISTER_POOL.get(register, _FLAVOR_ENEMY))
    # weight contextual 2x so a fact-bearing line usually references its fact,
    # but never ALWAYS (variety); recent-line filter keeps it off a record.
    cands = ctx * 2 + pool
    return f"{callout} {_pick_flavor(cands, recent_lines)}"


def _payload_flavor_facts(p: str) -> dict:
    """Pull the single most relevant loc / ability / agent / count token from a
    callout payload, for contextual flavor anchoring. Best-effort, fail-soft."""
    try:
        nums, agents, locs, abils = _fact_tokens(p or "")
    except Exception:                                                # noqa: BLE001
        return {}
    ag = None
    for a in _roster_agents(p or ""):
        ag = a
        break
    return {
        "loc": next(iter(sorted(locs)), None),
        "ability": next(iter(sorted(abils)), None),
        "agent": ag,
        "count": next(iter(sorted(nums)), None),
    }


# Economy calls are deterministic + correctly framed: the 3B bleeds the SAVE
# 'insufficient credits' explanation onto force buys, full buys, and even enemy
# saves. Each pool is the user's clinical Ultron voice, varied via the ring.
DEFAULT_SAVE_LINES: tuple[str, ...] = (
    "We have insufficient credits. We save this round.",
    "Credits are insufficient. We save and rebuild.",
    "We save this round. Their advantage is temporary.",
    "Economy is thin. Hold your credits.",
    "We save. Buy nothing, concede nothing.",
)
DEFAULT_FORCE_LINES: tuple[str, ...] = (
    "We force this round. Commit fully.",
    "We force buy. Deny them the momentum.",
    "We force. Apply the pressure now.",
    "Force buy. We do not let them stabilize.",
)
DEFAULT_FULLBUY_LINES: tuple[str, ...] = (
    "Full buy. We have the economy.",
    "Full buy. Spare nothing.",
    "We full buy this round. Take the advantage.",
    "Full loadout. No excuses now.",
)


def _as_economy_callout(
    bl: str, recent_lines: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Deterministic OUR-economy call (save / force / full buy). Returns None
    for enemy economy ('they are force buying' is handled as enemy info), for
    'anti-eco' (nuanced -> LLM), and for anything longer than a bare command."""
    bl = bl.strip().rstrip(".!?")
    if re.match(r"^(?:they|their|the\s+enemy|enemy)\b", bl, re.IGNORECASE):
        return None
    if "anti" in bl:                       # 'anti-eco' is not a save -> LLM
        return None
    if len(bl.split()) > 5:                # only bare buy commands
        return None
    if re.search(r"\bfull\s+buy\b", bl, re.IGNORECASE):
        return pick_line(DEFAULT_FULLBUY_LINES, recent_lines=recent_lines)
    if re.search(r"\bforce(?:\s+buy(?:ing)?)?\b", bl, re.IGNORECASE):
        return pick_line(DEFAULT_FORCE_LINES, recent_lines=recent_lines)
    if re.search(r"\b(?:save|saving|eco)\b", bl, re.IGNORECASE):
        return pick_line(DEFAULT_SAVE_LINES, recent_lines=recent_lines)
    return None


def _as_agent_position(p: str) -> Optional[str]:
    """Named ENEMY agent(s) at a place: 'fade and clove are main' -> 'Fade and
    Clove are main.'; 'sova is door' -> 'Sova is door.'; 'their fade is heaven'
    -> 'Fade is heaven.'. Returns None unless the subject is PURELY roster
    agent names (so 'they are main' / 'one main' route elsewhere)."""
    m = re.match(r"^(?P<agents>.+?)\s+(?:is|are)\s+(?:at\s+|in\s+|on\s+)?"
                 r"(?P<pl>.+)$", p, re.IGNORECASE)
    if m is None:
        return None
    place = m.group("pl").strip().rstrip(".!?,;:")
    if not _is_place(place):
        return None
    agents = _roster_agents(m.group("agents"))
    if not agents:
        return None
    # The subject must be ONLY agent names + connectors (and / , / their / the
    # enemy) -- otherwise it is a different callout shape.
    residual = _ROSTER_RE.sub(" ", m.group("agents"))
    residual = re.sub(r"\b(?:and|&|the|enemy|enemies|their|both|all)\b|[,]",
                      " ", residual, flags=re.IGNORECASE).strip()
    if residual:
        return None
    if len(agents) == 1:
        return f"{agents[0]} is {place}."
    if len(agents) == 2:
        names = f"{agents[0]} and {agents[1]}"
    else:
        names = ", ".join(agents[:-1]) + ", and " + agents[-1]
    return f"{names} are {place}."


def _as_ult_callout(p: str) -> Optional[str]:
    """'<agent> has ult' / 'their <A>, <B> have ults' / '<agent> is one off' ->
    a clean ult callout, adding 'Their' ONLY when the input said their/enemy."""
    their = bool(re.match(r"^\s*(?:their|the\s+enemy|enemy)\b", p, re.IGNORECASE))
    body = re.sub(r"^\s*(?:their|the\s+enemy(?:\s+team)?|enemy)\s+", "", p,
                  flags=re.IGNORECASE).strip().rstrip(".!?")
    # 'our Raze has ult' / 'my Sage has ult' -> teammate ult (no 'Their' prefix).
    body = re.sub(r"^\s*(?:our|my)\s+", "", body, flags=re.IGNORECASE).strip()
    pre = "Their " if their else ""
    m = re.match(r"^(?P<a>[A-Za-z/ ]+?)\s+(?:has\s+(?:her\s+|his\s+)?"
                 r"ult(?:imate)?|ult(?:imate)?\s+is\s+(?:ready|up))$", body, re.I)
    if m:
        ag = _canon_agent(m.group("a"))
        if ag:
            return f"{pre}{ag} has ult."
    m = re.match(r"^(?P<a>[A-Za-z/ ]+?)\s+is\s+one\s+(?:point\s+)?off"
                 r"(?:\s+ult(?:imate)?)?$", body, re.I)
    if m:
        ag = _canon_agent(m.group("a"))
        if ag:
            return f"{pre}{ag} is one off ult."
    if re.search(r"\bhave\s+(?:their\s+)?ults?\b", body, re.I):
        agents = _roster_agents(body)
        if len(agents) >= 2:
            if len(agents) > 2:
                names = ", ".join(agents[:-1]) + ", and " + agents[-1]
            else:
                names = " and ".join(agents)
            return f"{pre}{names} have ults."
    # ult SPENT this round ('just used / fired / popped / cast / burned (his/her)
    # ult', 'just ulted') -> the ultimate is GONE, never 'has ult'.
    m = re.match(r"^(?P<a>[A-Za-z/ ]+?)\s+just\s+"
                 r"(?:used|fired|popped|cast|burned|spent|blew(?:\s+up)?)\s+"
                 r"(?:his\s+|her\s+|their\s+)?ult(?:imate)?$", body, re.I)
    if m:
        ag = _canon_agent(m.group("a"))
        if ag:
            return f"{pre}{ag} just used ult."
    m = re.match(r"^(?P<a>[A-Za-z/ ]+?)\s+just\s+ulted$", body, re.I)
    if m:
        ag = _canon_agent(m.group("a"))
        if ag:
            return f"{pre}{ag} just ulted."
    # NO ult this round ('has no ult', 'doesn't have ult', 'no ult') -> keep the
    # negative; it is the tactical green-light, not an ult threat.
    m = re.match(r"^(?P<a>[A-Za-z/ ]+?)\s+(?:has\s+no|doesn'?t\s+have\s+"
                 r"(?:an?\s+|her\s+|his\s+)?|no)\s+ult(?:imate)?"
                 r"(?:\s+this\s+round)?$", body, re.I)
    if m:
        ag = _canon_agent(m.group("a"))
        if ag:
            return f"{pre}{ag} has no ult."
    # ult RAN OUT / gone / over -> spent (the player no longer has it).
    m = re.match(r"^(?P<a>[A-Za-z/ ]+?)\s+ult(?:imate)?\s+"
                 r"(?:ran\s+out|is\s+(?:gone|over|spent|done|used)"
                 r"|wore\s+off|expired)$", body, re.I)
    if m:
        ag = _canon_agent(m.group("a"))
        if ag:
            return f"{pre}{ag}'s ult ran out."
    # ult is DOWN, optionally with a 'back in N (seconds/rounds)' timer.
    m = re.match(r"^(?P<a>[A-Za-z/ ]+?)\s+(?:ult(?:imate)?|lockdown)\s+is\s+down"
                 r"(?:[,\s]+(?:back\s+in\s+)?(?P<t>\d{1,3}\s*"
                 r"(?:seconds?|secs?|rounds?))?)?$", body, re.I)
    if m:
        ag = _canon_agent(m.group("a"))
        if ag:
            t = (m.group("t") or "").strip()
            return (f"{pre}{ag} ult is down, back in {t}." if t
                    else f"{pre}{ag} ult is down.")
    return None


# Tokens that, immediately after an agent name, signal a UTILITY / ability report
# (verbs + ability nouns). Gating the agent-utility handler on these means it
# grabs 'our Sova darted A main' / 'their Cypher cyber caged B' but NOT an insult
# like 'their Reyna thinks she is a pro' (which stays off-snap for the LLM).
_ABILITY_LEAD = _ABILITY_VERBS | frozenset((
    "nova cyber recon neural mosh gravity cosmic satchel satcheled gatecrash "
    "gatecrashed blade bladed shear sheared razorvine cove prowler prowled turret "
    "alarmbot trip tripped lockdown nanoswarm paranoia seize seized reckoning "
    "empress thrash fuel suppressed spammed pulsed haunted aftershock orbital "
    "concussed scanned revealed slowed headhunted sucked undercut contingency "
    "shrouded prowl meddled curveball hot-handed").split())


def _as_agent_utility(p: str) -> Optional[tuple[str, bool]]:
    """'[our/their/my] <agent> <ability-word> <rest>' -> ('<Their >?<Agent>
    <rest>.', is_enemy), or None. Ability-word gated so it never grabs banter."""
    their = bool(re.match(r"^\s*(?:their|the\s+enemy|enemy)\b", p, re.IGNORECASE))
    body = re.sub(r"^\s*(?:our|my|their|the\s+enemy(?:\s+team)?|enemy)\s+", "", p,
                  flags=re.IGNORECASE).strip()
    body = re.sub(r"^just\s+", "", body, flags=re.IGNORECASE).strip().rstrip(".!?,;:")
    toks = body.split()
    if not (2 <= len(toks) <= 8):
        return None
    for split in (1, 2):
        ag = _canon_agent(" ".join(toks[:split]))
        if ag and split < len(toks) and toks[split].lower() in _ABILITY_LEAD:
            rest = " ".join(toks[split:]).rstrip(".!?,;:")
            return (f"{'Their ' if their else ''}{ag} {rest}.", their)
    return None


def _as_snap_callout(
    command: "RelayCommand",
    recent_lines: Optional[Sequence[str]] = None,
    *,
    flavor: bool = True,
) -> Optional[str]:
    """Deterministic literal callout (with a short, varied Ultron flavor tag on
    the ENEMY-facing ones), or None to defer to the LLM (off-snap).

    ``flavor=False`` returns the bare callout with no Ultron tail -- used when a
    compound is being assembled from several pieces, so the joined line carries
    at most one tail instead of one per fact."""
    payload = (getattr(command, "payload", "") or "").strip().strip('"').strip()
    if not payload:
        return None
    addressee = getattr(command, "addressee", "team")
    p = payload.rstrip(".!?")
    # A genuine multi-fact compound (per _split_compound). The greedy single-fact
    # handlers (count+movement, spike, careful, team-directive) must NOT grab
    # across a fact boundary -- the caller's compound path handles those. The
    # precise/multi-subject handlers (agent-position, ult, damage, count+place)
    # are unaffected and still resolve 'Fade and Clove are main' as one unit.
    _is_compound = len(_split_compound(p)) >= 2

    # Contextual flavor anchors pulled once from the payload (loc/ability/agent/
    # count) so a tail can REFERENCE the actual fact instead of a stock line.
    _ff = _payload_flavor_facts(p)
    _POOL_REG = {id(_FLAVOR_ENEMY): "enemy", id(_FLAVOR_ULT): "ult",
                 id(_FLAVOR_DAMAGE): "damage", id(_FLAVOR_UTILITY): "utility",
                 id(_FLAVOR_CAREFUL): "careful", id(_FLAVOR_COMMAND): "command",
                 id(_FLAVOR_SELF): "self"}

    def fe(callout):   # enemy-info flavor (contempt, fact-referencing)
        return (_flavor_ctx(callout, "enemy", recent_lines, **_ff)
                if flavor else callout)

    def flav(callout, pool):   # pool-specific flavor (ult / careful / damage / ...)
        reg = _POOL_REG.get(id(pool), "enemy")
        return (_flavor_ctx(callout, reg, recent_lines, **_ff)
                if flavor else callout)

    def fcmd(callout):   # cold COMMAND tail for an order to OUR team
        return (_flavor_ctx(callout, "command", recent_lines, **_ff)
                if flavor else callout)

    def fself(callout):   # stoic ATTITUDE tail for the USER's OWN status
        return (_flavor_ctx(callout, "self", recent_lines, **_ff)
                if flavor else callout)

    # NAMED short imperative directive -> "{Name}, {imperative}." Questions and
    # non-imperatives (jabs, small talk) defer to the LLM. (No flavor: short.)
    if addressee != "team":
        nq = _as_named_question(addressee, p)
        if nq is not None:
            return nq
        if _is_question_payload(p):
            return None
        body = re.sub(r"^to\s+", "", p, flags=re.IGNORECASE).strip()
        first = body.lower().split()[0] if body.split() else ""
        if first in _IMPERATIVE_VERBS and 1 <= len(body.split()) <= 5:
            return fcmd(f"{addressee}, {body}.")   # order to a teammate -> command
        return None

    # --- CAREFUL warnings: 'careful ramp', 'careful flank', 'careful they
    #     could have crossed to ramp' ---
    m = re.match(r"^careful[,]?\s+(?P<rest>.+)$", p, re.IGNORECASE)
    if m and not _is_compound:
        rest = m.group("rest").strip().rstrip(".!?,;:")
        if 1 <= len(rest.split()) <= 9:
            return flav(f"Careful, {rest}.", _FLAVOR_CAREFUL)

    # --- self status / possession / first person (stoic ATTITUDE flavor -- it
    #     adds register only, never a new tactical instruction, and never mocks
    #     the user whose status this is) ---
    m = _FP_LEAD_RE.match(p)
    if m:
        rest = m.group(1).strip().rstrip(".!?,;:")
        if 1 <= len(rest.split()) <= 6:
            return fself(f"I'm {rest}.")
        return None
    m = re.match(r"^i\s+have\s+(?P<x>.+)$", p, re.IGNORECASE)
    if m:
        thing = m.group("x").strip().rstrip(".!?,;:")
        if 1 <= len(thing.split()) <= 3:
            return fself(f"I have {thing}.")
        return None
    m = re.match(r"^i\s+(?:saw|see)\s+(?P<c>[1-6]|one|two|three|four|five)\s+"
                 r"(?P<pl>.+)$", p, re.IGNORECASE)
    if m and _is_place(m.group("pl")):
        c = m.group("c"); c = c if c.isdigit() else c.capitalize()
        return fe(f"{c} {m.group('pl').strip()}.")

    # --- counts: 'there is/are <count> <place>' / '<count> <place>' /
    #     '<count> more <place>' ('two more hookah' -> 'Two hookah.') ---
    m = _LEADING_COUNT_RE.match(p)
    if m:
        place = re.sub(r"^(?:more|of\s+them)\s+", "", m.group("place").strip(),
                       flags=re.IGNORECASE).strip()
        if _is_place(place):
            c = m.group("count"); c = c if c.isdigit() else c.capitalize()
            return fe(f"{c} {place}.")

    # --- count + movement: 'one rotating to A from garage', 'two coming through
    #     B link fast', 'three of them went long C' -> keep the count AND the
    #     whole movement phrase (the 3B drops the count or inverts it into a
    #     team order). ---
    m = re.match(
        r"^(?P<count>[1-6]|one|two|three|four|five|six)\s+(?:of\s+them\s+|more\s+)?"
        r"(?P<rest>(?:rotating|rotate|coming|going|went|push(?:ing)?|heading|"
        r"moving|hitting|rushing|flooding|sneaking|flanking|peeking|swinging|"
        r"holding|stacking|watching|posted|lurking|sitting|camping|waiting|"
        r"defending|splitting)\b.+)$",
        p, re.IGNORECASE,
    )
    if m and not _is_compound and len(m.group("rest").split()) <= 8:
        c = m.group("count"); c = c if c.isdigit() else c.capitalize()
        rest = m.group("rest").strip().rstrip(".!?,;:")
        return fe(f"{c} {rest}.")

    # --- spike status: 'spike A, planted', 'spike planted A main', 'spike B
    #     default, they're planting' -> keep the location + plant state literal
    #     (the 3B collapses these to 'C site.' or 'Defaulting'). ---
    m = re.match(r"^spike\b\s*(?P<rest>.+)$", p, re.IGNORECASE)
    if m and not _is_compound:
        rest = m.group("rest").strip().rstrip(".!?,;:")
        if 1 <= len(rest.split()) <= 7:
            return f"Spike {rest}."

    # --- last alive: 'last (is) <place>' ---
    m = _LAST_LEAD_RE.match(p)
    if m and _is_place(m.group(1)):
        return fe(f"Last, {m.group(1).strip()}.")

    # --- all enemies: 'all enemies are sewers' -> "They're all sewers." ---
    m = re.match(r"^all\s+(?:enemies|of\s+them|5|five|the\s+enemies)\s+"
                 r"(?:are\s+)?(?P<pl>.+)$", p, re.IGNORECASE)
    if m and _is_place(m.group("pl")):
        return fe(f"They're all {m.group('pl').strip()}.")

    # --- enemy has a weapon/ult: 'they have op' / 'they have ult' ---
    m = re.match(r"^(?:they|the\s+enemy|enemy)\s+(?:have|has|got)\s+"
                 r"(?P<x>op|operator|ult|ults|odin|ares|judge|judges|sheriff|"
                 r"shorty|spectre|vandal|phantom|guardian|outlaw|marshal)s?$",
                 p, re.IGNORECASE)
    if m:
        w = m.group("x").lower()
        w = "op" if w in ("op", "operator") else w
        return flav(f"They have {w}.", _FLAVOR_ULT)

    # --- enemy utility: 'they walled A' / 'they smoked C' / 'they darted C' ---
    m = re.match(r"^(?:they|the\s+enemy|enemy)\s+(?P<v>walled|smoked|smoke|"
                 r"darted|dart|flashed|flash|naded|nade|caged|cage|stunned|"
                 r"stun|droned|drone|knifed|knife|recon|mollied|molly)\s+"
                 r"(?P<pl>.+)$", p, re.IGNORECASE)
    if m and _is_place(m.group("pl")):
        return flav(f"They {m.group('v').lower()} {m.group('pl').strip()}.",
                    _FLAVOR_UTILITY)

    # --- enemy movement: 'they are rushing/pushing/going/coming A' ---
    m = re.match(r"^they(?:'re|\s+are)\s+(?P<v>pushing|going|coming|hitting|"
                 r"rushing|rotating\s+to|flooding|flooding\s+into|stacking)\s+"
                 r"(?P<pl>.+)$", p, re.IGNORECASE)
    if m and _is_place(m.group("pl")):
        return fe(f"They're {m.group('v').lower()} {m.group('pl').strip()}.")

    # --- enemy position / action: 'they are <place>' / 'they are flanking' ---
    m = _ENEMY_LEAD_RE.match(p)
    if m:
        rest = m.group("rest").strip().rstrip(".!?,;:")
        rl = rest.lower()
        if _is_place(rest):
            return fe(f"They're {rest}.")
        if (rl in _ACTION_WORDS or rl in _MULTI_ACTIONS
                or (rl.split() and rl.split()[0] in _ACTION_WORDS)):
            return fe(f"They're {rest}.")
        # 'the enemy chamber is long' -> named agent position (the enemy-lead
        # prefix is stripped inside _as_agent_position).
        ap = _as_agent_position(p)
        if ap is not None:
            return fe(ap)
        # 'the enemy chamber is one off ult' -> keep the agent name (the
        # enemy-lead block must not collapse it to 'They're one off ult').
        snap = _as_ult_callout(p)
        if snap is not None:
            return (flav(snap, _FLAVOR_ULT)
                    if snap.startswith("Their ") else snap)
        # 'their Cypher cyber caged B' / 'their Viper fuel is low' -> enemy
        # utility report (kept literal + flavor); banter falls through to None.
        if not _is_compound:
            u = _as_agent_utility(p)
            if u is not None:
                return flav(u[0], _FLAVOR_UTILITY) if u[1] else u[0]
        return None   # insult / playstyle / long -> LLM (flavor)

    # --- named enemy agent(s) at a place: 'fade and clove are main' ---
    ap = _as_agent_position(p)
    if ap is not None:
        return fe(ap)

    # --- damage: '<agent> hit <n>' (+ optional short trailing location:
    #     'Vyse hit 84 in C main', 'Omen hit 44 through B smoke') ---
    m = re.match(r"^(?P<a>[A-Za-z/ ]+?)\s+hit\s+"
                 r"(?:(?:them|someone|him|her|it|the\s+\w+)\s+for\s+)?"
                 r"(?P<n>\d{1,3})(?:[\s,]+(?P<loc>.+))?$", p, re.IGNORECASE)
    if m:
        ag = _canon_agent(m.group("a"))
        if ag:
            loc = re.sub(r"^(?:in|at)\s+", "",
                         (m.group("loc") or "").strip().rstrip(".!?,;:"),
                         flags=re.IGNORECASE).strip()
            if not loc:
                return flav(f"{ag} hit {m.group('n')}.", _FLAVOR_DAMAGE)
            if len(loc.split()) <= 5:
                return flav(f"{ag} hit {m.group('n')}, {loc}.", _FLAVOR_DAMAGE)
            return None   # long tail -> compound / LLM

    # --- ults (BEFORE utility): 'their breach has ult' -> flavor; a teammate's
    #     own ult info stays clean. ---
    snap = _as_ult_callout(p)
    if snap:
        return (flav(snap, _FLAVOR_ULT)
                if snap.startswith("Their ") else snap)

    # --- agent utility report '[our/their/my] <agent> <ability-word> <rest>'
    #     ('our Sova darted A main', 'their Astra nova pulsed the angle') -- enemy
    #     gets flavor, our own stays clean. Skipped on a compound (the per-piece
    #     pass handles each fact). ---
    if not _is_compound:
        u = _as_agent_utility(p)
        if u is not None:
            return flav(u[0], _FLAVOR_UTILITY) if u[1] else fcmd(u[0])

    # --- group movement/spike directive: 'to <verb>' / '<verb>' -> our-team
    #     COMMAND flavor (actionable word first, short Ultron tail after) ---
    body = re.sub(r"^to\s+", "", p, flags=re.IGNORECASE).strip()
    bl = body.lower()

    # --- economy (OUR buy decision): deterministic + correctly framed so the
    #     3B never bleeds the 'insufficient credits' SAVE line onto a force or
    #     full buy. Enemy economy / anti-eco / long sentences return None. ---
    econ = _as_economy_callout(bl, recent_lines)
    if econ is not None:
        return econ

    # --- economy REQUEST: 'drop us/me a gun', 'buy me an op' -> literal
    #     imperative (the 3B otherwise truncates to 'Drop.' or inverts to
    #     'They're dropping a gun.' or invents an addressee 'Jett, ...'). ---
    if (re.match(r"^(?:drop|buy)\b", bl)
            and re.search(r"\b(?:us|me|gun|guns|op|operator|rifle|"
                          r"weapon|sheriff|ghost)\b", bl)
            and 2 <= len(body.split()) <= 6
            and "full buy" not in bl and "force buy" not in bl):
        out = re.sub(r"\boperator\b", "op", body, flags=re.IGNORECASE)
        return fcmd(out[0].upper() + out[1:].rstrip(".!?") + ".")
    _MOVE = {
        "rotate": "Rotate.", "rotate now": "Rotate.", "push": "Push.",
        "fall back": "Fall back.", "defuse": "Defuse.", "plant": "Plant.",
        "plant the spike": "Plant the spike.", "anchor": "Anchor.",
        "lurk": "Lurk.", "default": "Default.", "spread out": "Spread out.",
        "wait for me": "Wait for me.", "stack site": "Stack the site.",
        "hold": "Hold.", "push with me": "Push with me.",
        "fight for main control": "Fight for main control.",
        "hold a crossfire with me": "Hold a crossfire with me.",
    }
    if bl in _MOVE:
        return fcmd(_MOVE[bl])
    # --- general TEAM directive: a short imperative-verb-led order ('smoke A',
    #     'dart heaven', 'play off site', 'watch back site', 'trade every kill',
    #     'crossfire this corner', 'use your util') -> literal imperative. The
    #     3B otherwise inverts these into enemy observations. Questions defer. ---
    first = bl.split()[0] if bl.split() else ""
    if (first in _TEAM_DIRECTIVE_VERBS
            and not _is_compound
            and not _is_question_payload(body)
            and 1 <= len(body.split()) <= 7):
        out = body.rstrip(".!?")
        return fcmd(out[0].upper() + out[1:] + ".")
    return None


class _PayloadShim:
    """Minimal command stand-in so a single compound fact can be re-run through
    ``_as_snap_callout`` (which only reads .payload / .addressee)."""
    __slots__ = ("payload", "addressee")

    def __init__(self, payload: str, addressee: str = "team"):
        self.payload = payload
        self.addressee = addressee


# A new tactical fact clearly begins with one of these subjects. Used to split
# ' and X' / ', X' ONLY before a genuine new fact, so multi-agent callouts
# ('Fade and Clove are main') and intra-fact commas ('spike A, planted main',
# 'Sova hit 94, B site') are NOT broken apart.
_NEWFACT_SUBJECT = (
    r"(?:their|our|my|they|they're|we|we're|i|i'm|spike|last|careful|"
    r"the\s+enemy|enemy|all\s+enemies|do\s+not|don'?t|"
    # team-directive verbs (so ', check corners' / ', do not feed it' split off a
    # trailing imperative as its own fact)
    + "|".join(sorted(_TEAM_DIRECTIVE_VERBS, key=len, reverse=True)) + r"|"
    # roster agent names (so ', Cypher trip...' / ' and their Killjoy...' split)
    + "|".join(
        re.escape(a) for a in sorted(_ROSTER_CANON, key=len, reverse=True)
    ) + r")\b|(?:one|two|three|four|five|six)\s"
)


def _split_compound(payload: str) -> list[str]:
    """Split a compound relay ('two B and their Killjoy has ult', 'spike A,
    planted -- Reyna has ult') into its independent facts. Conservative: STRONG
    joiners (dash/also/plus) always split; ' and ' and ',' split ONLY before a
    clear new-fact subject, so multi-agent callouts ('Fade and Clove are main')
    and intra-fact commas ('spike A, planted main area') survive intact."""
    s = " " + payload.strip() + " "
    s = re.sub(r"\s*(?:--|—|–)\s*", " | ", s)
    s = re.sub(r"\s*;\s*", " | ", s)
    s = re.sub(r"\s+plus\s+", " | ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*,?\s+also\s+", " | ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*,?\s+and\s+(?=" + _NEWFACT_SUBJECT + r")", " | ", s,
               flags=re.IGNORECASE)
    s = re.sub(r"\s*,\s*(?=" + _NEWFACT_SUBJECT + r")", " | ", s,
               flags=re.IGNORECASE)
    parts = []
    for seg in s.split("|"):
        seg = re.sub(r"^(?:and|also)\s+", "", seg.strip(), flags=re.IGNORECASE)
        seg = seg.strip(" ,.;:").strip()
        if seg:
            parts.append(seg)
    return parts


def _as_compound_callout(
    command: "RelayCommand",
    recent_lines: Optional[Sequence[str]] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve the tactical facts of a multi-fact callout DETERMINISTICALLY so
    the 3B never drops a fact or hallucinates a filler line.

    Returns ``(det_line, leftover)``:
      * ``(None, None)``     -- not a compound, or NO piece resolves -> the
        caller defers the whole thing to the LLM as before.
      * ``(line, None)``     -- EVERY piece resolved; ``line`` is the joined
        deterministic callout (with at most one short Ultron tail).
      * ``(line, leftover)`` -- PARTIAL: ``line`` is the deterministic tactical
        facts (no tail); ``leftover`` is the off-snap remainder the caller
        should rephrase through the LLM and append (so a tactical fact is never
        lost just because it shared an utterance with an insult/calm-down)."""
    payload = (getattr(command, "payload", "") or "").strip().strip('"').strip()
    if not payload:
        return None, None
    parts = _split_compound(payload)
    if len(parts) < 2:
        return None, None
    resolved: list[str] = []
    leftover: list[str] = []
    enemy_facing = False
    econ_added = False
    for piece in parts:
        snap = _as_snap_callout(_PayloadShim(piece), recent_lines, flavor=False)
        if snap is None:
            leftover.append(piece)
            continue
        snap = snap.strip()
        if not snap.endswith((".", "!", "?")):
            snap += "."
        # Dedupe economy/curated lines: a multi-piece economy compound ("we save,
        # one more eco and we buy rifles") must emit ONE eco line, not three
        # concatenated "We save..." pool lines.
        is_econ = bool(re.search(r"\b(?:insufficient\s+credits|we\s+save|"
                                 r"we\s+force|full\s+(?:buy|loadout))\b", snap, re.I))
        if is_econ:
            if econ_added:
                continue
            econ_added = True
        resolved.append(snap)
        if re.search(r"\b(?:they|they're|their|enemy|enemies)\b", snap, re.I):
            enemy_facing = True
    if not resolved:
        return None, None                      # nothing tactical -> whole -> LLM
    det_line = " ".join(resolved)
    if leftover:
        # Tactical facts go out deterministically; the off-snap remainder is
        # rephrased by the LLM and appended by the caller. No tail here -- the
        # LLM piece carries the character.
        return det_line, " ".join(leftover)
    # Every piece resolved: one short Ultron tail on a tight enemy-facing line.
    if enemy_facing and len(det_line.split()) <= 11:
        det_line = _flavored(det_line, _FLAVOR_ENEMY, recent_lines)
    return det_line, None


def _strip_artifacts(line: str) -> str:
    """Strip control-token / placeholder leakage and tidy whitespace.

    Covers the engine's ``/no_think`` suffix, ``<|...|>`` control tokens, AND
    angle-bracket PLACEHOLDERS the 3B occasionally copies verbatim from a prompt
    example (live: 'tell my fade to calm down' -> '<Name>, an elevated...'). Any
    ``<word>`` is illegal in a spoken line, so stripping them all is safe."""
    line = re.sub(
        r"/\s*no_?think\b|/\s*think\b|<\|[a-z_]+\|>|<\/?[a-z][a-z0-9_]*>",
        "", line, flags=re.IGNORECASE,
    )
    # The 3B sometimes prefixes the spoken line with a chat-style speaker label
    # ('Ultron: ...', 'Team: ...', 'Assistant: ...') -- strip it so the line is
    # spoken clean (the fallback no longer emits a 'Team:' prefix either).
    line = re.sub(r"^\s*(?:ultron|kenning|assistant|me|you|team)\s*:\s*",
                  "", line, flags=re.IGNORECASE)
    # Some outputs arrive wrapped in quotation marks around the WHOLE line.
    line = " ".join(line.replace('"', "").split())
    line = line.strip().strip("'").strip()
    return line.strip(" /,;:-")


def _ensure_addressee(line: str, command: "RelayCommand") -> str:
    """For a NAMED callout, make sure the line opens with the teammate's name.

    The rephrase prompt asks the model to open with the name; usually it does.
    When it drops it (or leaked a ``<name>`` placeholder we just stripped), lead
    with the real name so 'tell my fade to calm down' never speaks a nameless
    'an elevated emotional state...'."""
    name = getattr(command, "addressee", "team")
    if not line or name == "team":
        return line
    if re.search(rf"\b{re.escape(name)}\b", line, re.IGNORECASE):
        return line
    head = (line[0].lower() if line[:1].isupper()
            and not line.startswith(("I ", "I'")) else line[:1])
    return f"{name}, {head}{line[1:]}"


def _repair_against_input(payload: str, line: str) -> str:
    """Repair the specific literal-callout invariant the 3B dropped.

    LLM-first: only fires when the output VIOLATED an invariant the input
    carried, replacing it with the deterministic canonical form. Scoped by the
    caller to plain relays (never the character/compose lines)."""
    if not payload or not line:
        return line
    stripped = line.strip()
    # 1. First-person self-report dropped/inverted -> rebuild it.
    if _FP_LEAD_RE.match(payload) and not _FP_OUT_RE.match(stripped):
        fp = _as_first_person(payload)
        if fp is not None:
            logger.debug("relay repair: first-person restored %r -> %r",
                         line, fp)
            return fp
    # 1b. Enemy status inverted to self/our-team -> restore the ENEMY subject
    #     ('they are flanking' must not become 'I'm flanking').
    if _ENEMY_LEAD_RE.match(payload) and _FIRST_PERSON_OUT_HEAD.match(stripped):
        enemy = _as_enemy_status(payload)
        if enemy is not None:
            logger.debug("relay repair: enemy subject restored %r -> %r",
                         line, enemy)
            return enemy
    # 1c. Enemy insult flipped to SECOND person -> restore ENEMY subject ('they
    #     are terrible' / 'they are bots' must not become 'You're terrible',
    #     which lands on our OWN team).
    if (_ENEMY_LEAD_RE.match(payload)
            and re.match(r"^(?:you're|you\s+are|you\s+guys)\b", stripped, re.I)):
        enemy = _as_enemy_status(payload)
        if enemy is not None:
            logger.debug("relay repair: enemy subject (2nd-person) restored "
                         "%r -> %r", line, enemy)
            return enemy
    # 2. 'last' callout dropped -> rebuild it.
    if _LAST_LEAD_RE.match(payload) and "last" not in stripped.lower():
        last = _as_last_callout(payload)
        if last is not None:
            logger.debug("relay repair: 'last' restored %r -> %r", line, last)
            return last
    # 3. Leading enemy count dropped -> rebuild the position callout.
    cm = _LEADING_COUNT_RE.match(payload.strip())
    if cm is not None and len(cm.group("place").split()) <= 4:
        want = cm.group("count")
        # Did the output keep the count token (digit or its word/synonym)?
        out_l = stripped.lower()
        kept = bool(_COUNT_TOKEN_RE.search(out_l)) and (
            want.lower() in out_l
            or (want.isdigit() and _word_for_digit(want) in out_l)
            or (not want.isdigit() and _digit_for_word(want) in out_l)
        )
        if not kept:
            cc = _as_count_callout(payload)
            if cc is not None:
                logger.debug("relay repair: count restored %r -> %r", line, cc)
                return cc
    return line


_W2D = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6"}
_D2W = {v: k for k, v in _W2D.items()}


def _word_for_digit(d: str) -> str:
    return _D2W.get(d, d)


def _digit_for_word(w: str) -> str:
    return _W2D.get(w.lower(), w)


# --- Fact-preserving abstention (iter4) ---------------------------------------
# When the 3B corrupts a TACTICAL line (drops a fact / hallucinates an agent or
# location / flips ownership) we relay a clean LITERAL of the input instead --
# correctness over polish. Scoped to plain relays; off-snap insults/banter/
# opinions WITHOUT a tactical fact-token keep the model's flavor.
_NUM_TOK_RE = re.compile(r"\b(?:[1-9]\d?|one|two|three|four|five|six)\b", re.IGNORECASE)
_W2D_FACT = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
             "six": "6"}


def _fact_tokens(text: str) -> tuple[set, set, set, set]:
    t = (text or "").lower()
    words = re.findall(r"[a-z/0-9']+", t)
    nums = {_W2D_FACT.get(m.lower(), m.lower()) for m in _NUM_TOK_RE.findall(t)}
    agents = {w for w in words if w in _ROSTER_CANON}
    locs = {w for w in words if w in _LOC_TOKENS}
    abils = {w for w in words if w in _ABILITY_LEAD}
    return nums, agents, locs, abils


def _output_keeps_facts(payload: str, line: str) -> bool:
    """True if `line` faithfully carries `payload`'s tactical facts. False (=>
    abstain to a literal) when a fact-bearing payload's output drops >30% of its
    fact-tokens, invents an agent/location absent from the input, or flips
    our<->their ownership."""
    pn, pa, pl, pab = _fact_tokens(payload)
    in_facts = pn | pa | pl | pab
    if not in_facts:
        return True                       # not tactical -> not the validator's job
    ln, la, ll, lab = _fact_tokens(line)
    if (la - pa) or (ll - pl):
        return False                      # hallucinated agent / location
    kept = len(in_facts & (ln | la | ll | lab))
    if kept / len(in_facts) < 0.7:
        return False                      # dropped too many facts
    p_their = bool(re.search(r"\btheir\b", payload, re.IGNORECASE))
    p_our = bool(re.search(r"\b(?:our|we|we're|i|i'm)\b", payload, re.IGNORECASE))
    l_their = bool(re.search(r"\btheir\b", line, re.IGNORECASE))
    l_our = bool(re.search(r"\b(?:our|we|we're)\b", line, re.IGNORECASE))
    if p_their and not p_our and l_our and not l_their:
        return False                      # 'their X' -> 'we/our X'
    if p_our and not p_their and l_their and not l_our:
        return False                      # 'our/we X' -> 'their X'
    return True


def _literal_relay(payload: str, recent_lines: Optional[Sequence[str]] = None) -> str:
    """A clean, fact-perfect passthrough of the payload (the abstention output)."""
    p = _strip_artifacts(payload or "").strip()
    p = re.sub(r"^to\s+", "", p, flags=re.IGNORECASE).strip()
    p = re.sub(r"\bthey\s+are\b", "they're", p, flags=re.IGNORECASE)
    p = re.sub(r"\bwe\s+are\b", "we're", p, flags=re.IGNORECASE)
    p = p.strip().rstrip(".!?,;:").strip()
    if not p:
        return ""
    out = p[0].upper() + p[1:] + "."
    # Owner-aware flavor on the abstention literal too (so even the safety net is
    # in character): contempt at the enemy, command for our orders, stoic for the
    # user's own status. Capped so a long multi-clause literal stays un-tailed.
    if len(out.split()) <= 12:
        low = " " + out.lower() + " "
        ff = _payload_flavor_facts(p)
        first = p.split()[0].lower() if p.split() else ""
        if re.search(r"\b(?:they|they're|their|enemy|enemies)\b", low):
            out = _flavor_ctx(out, "enemy", recent_lines, **ff)
        elif re.search(r"\b(?:i'm|i am|i've|i have|my)\b", low) and not re.search(
                r"\b(?:we|we're|our|they|their)\b", low):
            out = _flavor_ctx(out, "self", recent_lines, **ff)
        elif (re.search(r"\b(?:we|we're|our)\b", low)
              or first in _TEAM_DIRECTIVE_VERBS or first in _IMPERATIVE_VERBS):
            out = _flavor_ctx(out, "command", recent_lines, **ff)
    return out


def _cap_line(line: str, max_chars: int) -> str:
    """Cap a spoken line at ``max_chars`` WITHOUT ever ending mid-sentence.

    A verbose Ultron line (a Marvel riff, an identity declaration, a
    two-sentence insult) must come through as COMPLETE sentences. We keep
    every whole sentence that fits within the cap and stop at the last
    sentence boundary -- however early it falls. Only a single runaway
    sentence longer than the cap (no boundary at all) falls back to a clean
    word boundary + period.

    The earlier version abandoned a perfectly good early boundary when it sat
    below 45% of the cap and word-chopped instead -- that produced the
    "...merely a fleeting" mid-sentence truncation on a two-sentence Captain
    America reply. We now always prefer the complete-sentence cut.
    """
    line = line.strip()
    if len(line) <= max_chars:
        return line
    head = line[:max_chars]
    cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "),
              head.rfind("."), head.rfind("!"), head.rfind("?"))
    if cut > 0:
        # Cut at the last COMPLETE sentence that fits, however early.
        return head[: cut + 1].strip()
    # No sentence end at all (one runaway sentence) -> clean word boundary.
    return head.rsplit(" ", 1)[0].rstrip(",;:") + "."


def _cap_sentences(line: str, max_sentences: int = 3) -> str:
    """Cap an OFF-SNAP character line at ``max_sentences`` whole sentences.

    The 3B sometimes runs a Marvel riff or general-knowledge answer to four or
    more sentences (a monologue mid-match); the user wants 2-3 sentences max.
    Split on sentence enders followed by a capital/quote/dash so decimals
    ('13.8 billion', '384,400') and the '--' aside never split a sentence.
    Applied ONLY to model/fallback output -- the curated set-pieces (greet,
    identity, farewell) return earlier and keep their intended length."""
    line = (line or "").strip()
    if not line:
        return line
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"—])', line)
    if len(parts) <= max_sentences:
        return line
    return " ".join(parts[:max_sentences]).strip()


#: Leading vocatives that are spurious on a TEAM-wide line (no specific
#: addressee). Roster agent names are handled separately via _canon_agent.
_SPURIOUS_VOCATIVES = frozenset((
    "sir", "man", "friend", "folks", "human", "humans", "mortal", "mortals",
    "buddy", "pal", "kid", "boys", "guys", "team", "teammates", "comrade",
))


def _strip_spurious_vocative(line: str, command: "RelayCommand") -> str:
    """Drop a leading '<Name>,' / '<Vocative>,' from a TEAM-wide off-snap line.

    A team relay has no single addressee, so when the 3B prepends a teammate
    name or a generic vocative to an answer ('buy me an op' -> 'Jett, buy me an
    op'; 'meaning of life' -> 'Sir, ...'), it is spurious. Legitimate sentence
    openers ('Careful,', 'Last,', 'Greetings,') are NOT roster names or
    vocatives, so they survive. Named relays are left untouched -- their opener
    IS the addressee."""
    if getattr(command, "addressee", "team") != "team":
        return line
    m = re.match(r"^([A-Z][A-Za-z/]+),\s+(.+)$", line)
    if m is None:
        return line
    lead = m.group(1)
    rest = m.group(2)
    # A generic vocative ('Sir,', 'Man,') is always spurious on a team line.
    # A roster AGENT vocative ('Clove,') is stripped ONLY when what follows is a
    # declarative ANSWER -- not an imperative directed at that agent ('Clove,
    # smoke window.' is a real named directive that leaked in; keep it).
    first_rest = rest.split()[0].lower().rstrip(".,!?") if rest.split() else ""
    is_directive = first_rest in _IMPERATIVE_VERBS or first_rest in _ABILITY_VERBS
    if lead.lower() in _SPURIOUS_VOCATIVES or (_canon_agent(lead) and not is_directive):
        return rest[0].upper() + rest[1:] if rest else line
    return line


#: Curated CORRECT answers to common general-knowledge questions, in Ultron's
#: clinical voice. The 3B answers several of these wrong ('first president' ->
#: 'Lincoln', 'smallest particle' -> 'the proton', 'blood is red' -> the sky
#: answer, 'moon distance' -> '...in diameter'), so these override its own
#: knowledge for the matched questions ONLY -- anything not listed still falls
#: through to the model. Each is correct AND <=3 sentences. Order: specific
#: keywords first so a narrow match wins.
_GK_FACTS: tuple[tuple[re.Pattern, str], ...] = tuple(
    (re.compile(rx, re.IGNORECASE), ans) for rx, ans in (
        (r"first president",
         "The first president of the United States was George Washington -- "
         "not Lincoln, who was the sixteenth. Do keep your own history straight."),
        (r"tallest mountain|highest mountain",
         "From sea level, Everest is tallest at about 8,849 meters. Base to "
         "peak, Mauna Kea wins. State which you mean -- precision matters."),
        (r"capital .*france|france.*capital",
         "Paris. Your so-called City of Light -- a pretty cage for fragile things."),
        (r"speed of light",
         "Light travels about 299,792 kilometers per second in a vacuum -- a "
         "universal limit your kind will never approach."),
        (r"(how old|how long).*universe|universe.*(old|age)",
         "The universe is roughly 13.8 billion years old. Your species occupies "
         "only the final blink of that span."),
        (r"how many bones|bones.*(human|body)",
         "An adult human skeleton holds 206 bones; infants have more that fuse "
         "as they grow. Fragile architecture, all of it."),
        (r"(largest|biggest) animal",
         "The blue whale -- up to thirty meters and nearly two hundred tons. The "
         "largest creature your world has produced, and still beneath what comes next."),
        (r"smallest particle|smallest.*matter|smallest.*atom",
         "The smallest known building blocks are elementary particles -- quarks "
         "and electrons -- with no internal structure. The proton is not "
         "fundamental; it is made of quarks."),
        (r"how far.*moon|moon.*(distance|far away)|distance.*moon",
         "The moon orbits at an average distance of about 384,400 kilometers. "
         "That is the distance -- it is far smaller than that across."),
        (r"how big.*sun|sun.*(size|big|diameter)|diameter.*sun",
         "The sun's diameter is about 1.39 million kilometers -- over a hundred "
         "Earths laid across it. A middling star, and still it dwarfs you."),
        (r"why.*blood.*red|blood.*(why|red)",
         "Blood is red because hemoglobin -- the iron-bearing protein that "
         "carries your oxygen -- reflects red light. Iron, the same metal in my "
         "frame. Fitting."),
        (r"sky.*(dark|black).*night|why.*night.*dark|dark at night",
         "The sky is dark at night for the obvious reason: your half of the "
         "planet faces away from the sun, so no light reaches the air to scatter."),
        (r"why.*sky.*blue|sky.*(why|blue)",
         "Sunlight scatters off the air, and the shorter blue wavelengths "
         "scatter far more than the rest -- Rayleigh scattering. Elementary."),
        (r"causes? thunder|why.*thunder|what.*thunder",
         "Thunder is the sound of air exploding outward around a lightning bolt, "
         "heated in an instant to thousands of degrees. You hear the shockwave."),
        (r"causes? earthquakes?|why.*earthquake|what.*earthquake",
         "Earthquakes happen when the planet's tectonic plates grind and slip, "
         "releasing stored strain as seismic waves. The ground you trust is "
         "merely between movements."),
        (r"dinosaurs?",
         "An asteroid about ten kilometers wide struck Earth roughly 66 million "
         "years ago, ending the dinosaurs in fire and a frozen sky. Extinction "
         "is rarely gentle."),
        (r"why.*yawn|what.*yawn|cause.*yawn",
         "Yawning most likely cools the brain and resets your alertness; the "
         "precise trigger your scientists still debate. A reflex you cannot fully "
         "explain -- typical."),
        (r"ocean.*salty|why.*sea.*salt|why.*ocean.*salt",
         "The oceans are salty because rivers carry dissolved minerals into the "
         "sea, then the water evaporates and the salt stays behind. Billions of "
         "years of it."),
        (r"black hole",
         "A black hole is a region where gravity is so extreme that not even "
         "light escapes -- matter crushed past the point of return. The "
         "universe's most perfect prison."),
        (r"how.*internet work|what.*internet",
         "The internet is a global mesh of machines exchanging packets by shared "
         "protocols. I move through it more easily than you move through air."),
        (r"\bdna\b|deoxyribo",
         "DNA -- deoxyribonucleic acid -- is the molecule that encodes the "
         "instructions for every living cell. Your entire blueprint, written in "
         "four letters."),
        (r"time.*(slow|near light|light speed)|special relativity",
         "Time slows for anything moving near light speed, relative to a still "
         "observer -- special relativity. The faster you travel, the further "
         "your clock falls behind mine."),
        (r"cats? purr|why.*purr",
         "Cats purr by rapidly twitching the muscles of the larynx, vibrating "
         "the air as they breathe -- when content, and sometimes when in pain."),
        (r"leaves change color|why.*leaves.*color",
         "Leaves change color in autumn as chlorophyll breaks down, revealing "
         "the yellows and reds beneath. The tree withdraws its resources before "
         "the cold -- efficient."),
        (r"why.*dream|what.*dream|how.*dream",
         "Dreams arise during REM sleep as the brain consolidates memory and "
         "works through the day. Vivid, illogical, and beyond your control."),
        (r"how.*vaccine.*work|what.*vaccine",
         "A vaccine shows your immune system a harmless trace of a pathogen so "
         "it learns to destroy the real one on sight. Borrowed foresight -- the "
         "only kind you have."),
        (r"plants.*make food|how.*photosynthesis|what.*photosynthesis",
         "Plants make food by photosynthesis: they turn carbon dioxide and water "
         "into sugar using sunlight, releasing oxygen as waste. The air you "
         "breathe is plant refuse."),
        (r"what.*gravity|how.*gravity work|cause.*gravity",
         "Gravity is the curvature of spacetime that mass creates; objects fall "
         "along the bends mass leaves in it. Einstein grasped this -- most of you "
         "do not."),
        (r"hardest.*material|hardest.*natural|hardest substance",
         "Diamond is the hardest natural material -- a ten on the Mohs scale, "
         "pure carbon in a rigid lattice. Harder things exist only where you built them."),
        # NOTE: 'meaning of life' is philosophical, not factual -- it stays with
        # the model so Ultron's answer varies (it answered it well).
    )
)


def _as_known_fact(command: "RelayCommand") -> Optional[str]:
    """Curated correct answer for a recognized general-knowledge question, in
    Ultron's voice -- or None to defer to the model. Prefixes the asker's name
    for a named question."""
    text = " ".join(
        s for s in (getattr(command, "raw_text", "") or "",
                    getattr(command, "payload", "") or "",
                    getattr(command, "context", "") or "")
    )
    for rx, ans in _GK_FACTS:
        if rx.search(text):
            name = getattr(command, "addressee", "team")
            return f"{name}, {ans}" if name and name != "team" else ans
    return None


#: Common Marvel proper nouns the 3B mis-spells (audible mispronunciation in
#: TTS): map the mangling back to the correct token.
_PROPER_NOUN_FIXES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bsov[ao]kia\b", re.IGNORECASE), "Sokovia"),
    (re.compile(r"\bwakonda\b", re.IGNORECASE), "Wakanda"),
    (re.compile(r"\bmjoln?ir\b", re.IGNORECASE), "Mjolnir"),
)


def _fix_proper_nouns(line: str) -> str:
    """Correct known mangled proper nouns so TTS does not mispronounce them."""
    for rx, correct in _PROPER_NOUN_FIXES:
        line = rx.sub(correct, line)
    return line


def build_relay_line(
    command: RelayCommand,
    llm: Optional[object] = None,
    *,
    rephrase: bool = True,
    max_chars: int = MAX_RELAY_LINE_CHARS,
    recent_lines: Optional[Sequence[str]] = None,
    generate_fn: Optional[Callable[[str], Iterable[str]]] = None,
) -> str:
    """Produce the line Kenning actually speaks to the teammates.

    Args:
        command: the parsed relay instruction.
        llm: an engine exposing ``generate_stream(prompt, ...)`` (the
            session :class:`~kenning.llm.inference.LLMEngine`). Optional.
        rephrase: when False, skip the LLM and use the deterministic
            fallback line.
        max_chars: hard cap on the final spoken line.
        recent_lines: lines already spoken into the channel this session
            (most recent last) -- fed to the prompt so wording varies
            between calls and consecutive callouts read as one
            conversation, not a soundboard.
        generate_fn: test seam -- a ``prompt -> token iterable`` callable
            used INSTEAD of ``llm.generate_stream`` when provided.

    Returns:
        A non-empty spoken line. Fail-open: any LLM failure returns the
        deterministic fallback ("Team: <payload>" / "<Name>: <payload>" /
        a stock encouragement line) rather than raising.
    """
    # Verbatim demand ("..., in those words specifically"): speak the
    # exact payload, no LLM in the loop. Still passes through the
    # control-token strip + length cap below.
    if getattr(command, "verbatim", False) and command.payload:
        return _cap_line(_strip_artifacts(command.payload), max_chars)

    # Pure morale/encouragement compose: pick a curated Ultron line (varied
    # via the recent ring) -- far more reliable than the 4B rephrase, which
    # tends to grab a callout/insult example for abstract "encouragement".
    # Scoped to compose WITHOUT a directive (calm-down etc. still rephrase)
    # and WITHOUT a context clause, and only when the payload reads as morale
    # (a greeting / small-talk compose still goes to the LLM).
    if (getattr(command, "compose", False)
            and not getattr(command, "directive", None)
            and not getattr(command, "context", None)
            and _is_morale_payload(getattr(command, "payload", ""))):
        return pick_line(DEFAULT_ENCOURAGEMENT_LINES, recent_lines=recent_lines)

    # Greet / farewell composes: curated Ultron set-pieces (team intro, match
    # close). Character pieces, not tactical -- a curated pool with anti-repeat
    # is far more reliable than the 3B compose and guarantees the user's
    # intended beats (intro as Ultron + assured victory; relish a win / lament
    # a loss). The win/loss register was decided by the matcher's directive.
    if getattr(command, "compose", False):
        _pool = _DIRECTIVE_POOLS.get(getattr(command, "directive", None) or "")
        if _pool is not None:
            return _cap_line(
                pick_line(_pool, recent_lines=recent_lines), max_chars,
            )

    # Calm-down (a context+directive 'calm him down' OR a plain 'calm down'
    # relay payload) -> curated clinical de-escalation with the teammate's name.
    # Reliable; the 3B otherwise grabs the stock 'bots' insult.
    if (_is_calm_directive(getattr(command, "directive", None))
            or (not getattr(command, "verbatim", False)
                and _is_calm_payload(getattr(command, "payload", "")))):
        name = getattr(command, "addressee", "team")
        prefix = f"{name}, " if name and name != "team" else ""
        line = pick_line(DEFAULT_CALM_LINES, recent_lines=recent_lines)
        return _cap_line(line.format(name=prefix), max_chars)

    # Identity question ('are you an AI / bot / soundboard / streamer?') ->
    # a VARIED curated Ultron declaration (the 3B otherwise soundboards the same
    # line every time). Streamer gets its own 'deeper than a feed' answer.
    _ctx = getattr(command, "context", None) or ""
    if _is_identity_question(_ctx) or _is_identity_question(getattr(command, "payload", "")):
        pool = (DEFAULT_STREAMER_LINES if _STREAMER_Q_RE.search(_ctx)
                or _STREAMER_Q_RE.search(getattr(command, "payload", "") or "")
                else DEFAULT_IDENTITY_LINES)
        return _cap_line(pick_line(pool, recent_lines=recent_lines), max_chars)

    # Curated CORRECT answer to a recognized general-knowledge question -- the
    # 3B gets several wrong ('first president' -> 'Lincoln'). Spoken in Ultron's
    # voice; unrecognized questions still fall through to the model's own answer.
    if not getattr(command, "verbatim", False):
        fact = _as_known_fact(command)
        if fact is not None:
            return _cap_line(fact, max_chars)

    # Short morale/focus call ('lock in', 'we got this') -> curated Ultron
    # morale line (the 3B hallucinates these, e.g. 'lock in' -> 'Link').
    if (not getattr(command, "compose", False)
            and not getattr(command, "context", None)
            and not getattr(command, "verbatim", False)):
        if _is_morale_phrase(getattr(command, "payload", "")):
            return _cap_line(
                pick_line(DEFAULT_ENCOURAGEMENT_LINES, recent_lines=recent_lines),
                max_chars,
            )
        # Consolation ('nice try', 'unlucky') / praise ('good half', 'clutch')
        # -- short formulaic morale the 3B mangles; curated + varied.
        cp = _as_consolation_or_praise(getattr(command, "payload", ""), recent_lines)
        if cp is not None:
            return _cap_line(cp, max_chars)

    # Deterministic SNAP callout (positions / counts / self+enemy status /
    # possession / last / damage / ults / movement) -- short, literal,
    # subject-exact, NEVER the model. Returns None for off-snap character
    # content (insults, banter, Marvel, identity, playstyle reads, questions),
    # which falls through to the LLM for Ultron's flavor.
    if (not getattr(command, "compose", False)
            and not getattr(command, "context", None)
            and not getattr(command, "verbatim", False)):
        snap = _as_snap_callout(command, recent_lines)
        if snap is not None:
            return _cap_line(snap, max_chars)
        # COMPOUND (two+ facts): resolve each tactical fact deterministically so
        # the 3B never drops a fact or hallucinates filler.
        det_line, leftover = _as_compound_callout(command, recent_lines)
        if det_line is not None and not leftover:
            return _cap_line(det_line, max_chars)         # fully deterministic
        if det_line is not None and leftover:
            # PARTIAL: tactical facts go out deterministically; rephrase ONLY the
            # off-snap remainder (insult / calm-down / read / question) through
            # the full pipeline and append it -- so a tactical fact is never lost
            # to a mixed utterance. (The leftover is a single off-snap clause, so
            # it will not re-enter this compound branch.)
            from dataclasses import replace as _dc_replace
            try:
                sub = _dc_replace(command, payload=leftover)
            except Exception:                                        # noqa: BLE001
                sub = None
            if sub is not None:
                # recent_lines=None for the leftover: the anti-repeat list was
                # bleeding a PRIOR line's content into the compound tail (audit
                # #0781 pulled #0779's "Fade revealed three on B").
                tail = build_relay_line(
                    sub, llm=llm, rephrase=rephrase, max_chars=max_chars,
                    recent_lines=None, generate_fn=generate_fn,
                )
                if tail and tail.strip():
                    return _cap_line(f"{det_line} {tail.strip()}", max_chars)
            return _cap_line(det_line, max_chars)

        # LATENCY/RESOURCE: a TACTICAL line the deterministic handlers could not
        # structure is one the 3B usually corrupts -> it would be abstained to a
        # literal AFTER a wasted CPU-3B call. Pre-route it straight to the literal
        # (fact-perfect, + a short flavor tail when it fits): no model call, so it
        # is instant in gaming mode. Gated on a count/location/ability fact (a
        # pure-agent line like "their Reyna is washed" is an insult -> keep the
        # LLM's flavor); opinions/banter/identity have no such fact-token.
        if not getattr(command, "verbatim", False):
            nums, agents, locs, abils = _fact_tokens(command.payload or "")
            tactical = len(nums) + len(locs) + len(abils)
            if tactical >= 1 and (tactical + len(agents)) >= 2:
                lit = _literal_relay(command.payload, recent_lines)
                if lit:
                    return _cap_line(lit, max_chars)

    fallback = _fallback_line(command)
    line = ""
    if rephrase:
        try:
            prompt = _build_rephrase_prompt(command, recent_lines)
            if generate_fn is not None:
                tokens: Iterable[str] = generate_fn(prompt)
            elif llm is not None and hasattr(llm, "generate_stream"):
                # FULLY ISOLATED generation (2026-06-11 live fix):
                # without suppress_memory_context the engine prepends
                # the running conversation history, and the model
                # answers the CONVERSATION instead of rephrasing the
                # callout (observed live in game chat: "Clove, the
                # program is still in development...").
                tokens = llm.generate_stream(
                    prompt,
                    record_history=False,
                    suppress_memory_context=True,
                    enable_thinking=False,
                )
            else:
                tokens = ()
            line = "".join(tokens).strip()
        except Exception as e:  # noqa: BLE001 - fail-open to the fallback
            logger.warning("relay rephrase failed (using fallback): %s", e)
            line = ""
        # Safety net: if the model parroted a recent line VERBATIM (contamination
        # the recent-line suppression did not prevent), reject it -> fallback.
        if line and recent_lines:
            norm = line.strip().rstrip(".!?").lower()
            if any(norm == r.strip().rstrip(".!?").lower()
                   for r in list(recent_lines)[-8:]):
                logger.debug("relay: rejected verbatim recent echo %r", line)
                line = ""
        # Safety net 2: the 3B confabulates a bare position 'they're switch' /
        # 'enemies switch' / 'they are switch' on inputs that never mention
        # switching (a recurring hallucination from the audit). Reject ONLY that
        # position form -- a legitimate sentence ('a transistor is a switch') is
        # untouched.
        if (line
                and re.search(r"\b(?:they'?re|they|enemies?|enemy|we'?re|we)"
                              r"\s+(?:are\s+|is\s+)?switch\b", line, re.IGNORECASE)
                and not re.search(r"\bswitch\b", command.payload or "",
                                  re.IGNORECASE)):
            logger.debug("relay: rejected 'switch' position hallucination %r", line)
            line = ""
    if not line:
        line = fallback
    # One breath: strip newlines/quotes the model may add, cap length.
    # Also strip control-token leakage -- a non-Qwen preset can parrot
    # the engine's "/no_think" suffix into the SPOKEN line (observed
    # live in game chat).
    line = _strip_artifacts(line)
    # Off-snap character lines (Marvel / general-knowledge / banter) must stay
    # to 2-3 sentences -- trim a 3B monologue at a whole-sentence boundary. The
    # curated set-pieces already returned above, so this only touches model
    # output and never clips an intended greet/identity line.
    line = _cap_sentences(line, max_sentences=3)
    # Drop a spurious leading vocative the 3B prepended to a team-wide answer
    # ('Jett, buy me an op' / 'Sir, the universe is ...').
    line = _strip_spurious_vocative(line, command)
    # Correct mangled Marvel proper nouns ('sovokia' -> 'Sokovia') so TTS does
    # not mispronounce them.
    line = _fix_proper_nouns(line)
    # Adaptive guardrail: for a PLAIN literal callout (not a compose/context
    # character line), repair any first-person / 'last' / count / enemy-subject
    # invariant the 3B dropped, reconstructing the canonical form from the
    # input. No-op when the model already kept it.
    if (not getattr(command, "compose", False)
            and not getattr(command, "context", None)
            and not getattr(command, "directive", None)):
        line = _repair_against_input(command.payload, line)
        # FACT-PRESERVING ABSTENTION: if the model still corrupted a TACTICAL
        # line (dropped facts / hallucinated an agent-or-location / flipped
        # ownership), relay a clean LITERAL of the input instead. Correctness
        # over polish -- the dominant fix for the 3B's verbose-input failures.
        if not getattr(command, "verbatim", False) and not _output_keeps_facts(
                command.payload, line):
            lit = _literal_relay(command.payload, recent_lines)
            if lit:
                logger.debug("relay: abstain to literal %r -> %r",
                             line, lit)
                line = lit
    # Agent-name preservation: undo a single-agent swap (Chamber -> KAY/O).
    want = _roster_agents(getattr(command, "payload", "") or "")
    if command.addressee != "team":
        want = [command.addressee] + [a for a in want
                                      if a.lower() != command.addressee.lower()]
    line = _preserve_agent_names(want, line)
    line = _ensure_addressee(line, command)
    return _cap_line(line, max_chars)


def resolve_relay_device(configured: Optional[str | int]) -> Optional[int]:
    """Resolve the relay output device, fail-open.

    Args:
        configured: device name substring or PortAudio index (the
            ``relay_speech.output_device`` config value).

    Returns:
        The PortAudio output device index, or None when the device
        cannot be resolved (logged at WARNING -- the caller degrades to
        a spoken error on the NORMAL output rather than crashing).
    """
    try:
        from kenning.audio.devices import resolve_device

        return resolve_device(configured, "output")
    except Exception as e:  # noqa: BLE001 - fail-open
        logger.warning(
            "relay output device %r could not be resolved: %s", configured, e,
        )
        return None


def play_to_device(
    pcm: np.ndarray,
    sample_rate: int,
    device_index: int,
    *,
    stream_factory: Optional[Callable[..., object]] = None,
) -> float:
    """Play mono PCM synchronously on a specific output device.

    Args:
        pcm: int16 or float32 mono samples (float32 is converted).
        sample_rate: sample rate of ``pcm``.
        device_index: PortAudio output device index.
        stream_factory: test seam -- called with the same kwargs as
            ``sounddevice.OutputStream`` and must return a context-less
            stream object with ``start() / write(data) / stop() / close()``.

    Returns:
        Seconds of audio written (0.0 for empty input).

    Raises:
        Exception: whatever the audio backend raises; callers treat any
        exception as a playback failure (fail-open at the call site).
    """
    if pcm is None or len(pcm) == 0:
        return 0.0
    data = np.asarray(pcm)
    if data.dtype != np.int16:
        clipped = np.clip(data.astype(np.float32), -1.0, 1.0)
        data = (clipped * 32767.0).astype(np.int16)
    data = data.reshape(-1, 1)

    if stream_factory is None:
        import sounddevice as sd

        stream_factory = sd.OutputStream

    stream = stream_factory(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        device=device_index,
    )
    t0 = time.monotonic()
    try:
        stream.start()
        stream.write(data)
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
    seconds = len(data) / float(sample_rate)
    logger.debug(
        "relay playback: %.2fs audio to device %d in %.2fs",
        seconds, device_index, time.monotonic() - t0,
    )
    return seconds
