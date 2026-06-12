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
MAX_RELAY_LINE_CHARS = 280

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
        rf"^(?:please\s+)?say\s+(?P<payload>.+?)\s+(?:to\s+{_GROUP}"
        rf"|in\s+(?:the\s+)?(?:team\s+|game\s+)?chat)\s*[.!?]?$",
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
    # "tell them/everyone/chat (that|to) X" -- in a voice-chat session
    # these address the team; "tell me ..." does not match by
    # construction, and a bare "tell him/her ..." is only honoured in
    # the context+directive forms below.
    re.compile(
        r"^(?:please\s+)?tell\s+(?:them|everyone|chat|the\s+lobby)\s+"
        r"(?:that\s+|to\s+)?(?P<payload>.+)$",
        re.IGNORECASE,
    ),
    # "call out (that) X" -- gamer shorthand for a team info callout.
    re.compile(
        r"^(?:please\s+)?call\s+out\s+(?:that\s+)?(?P<payload>.+)$",
        re.IGNORECASE,
    ),
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
    # "my clove" / "our sova" -- the user often refers to the teammate
    # possessively by the agent they're playing.
    name = rf"(?:my\s+|our\s+)?(?P<name>{alts})\b"
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
    "the a an to that this it is are was be my our me you i and or".split()
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
    words = payload.split()
    if len(words) >= 2:
        return True
    if not words:
        return False
    word = words[0].strip(".,!?;:").lower()
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
    cleaned = _LEADING_ARTIFACT.sub("", text.strip())

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
    return None


_REPHRASE_PROMPT = (
    "You are Ultron, speaking OUT LOUD into your user's Valorant voice chat "
    "on their behalf -- in this game you go by Ultron: cold, brilliant, "
    "theatrically superior, dryly menacing. Pick the register from the line:\n"
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
    "match). For STYLE only: an insult sharpens ('you're bots' -> 'You guys "
    "are complete, hopeless bots.'), an economy call explains ('save' -> 'We "
    "have insufficient credits. We save this round.'), a calm-down is clinical "
    "('<name>, an elevated emotional state degrades performance. Calm "
    "yourself.'). These three are ILLUSTRATIONS of tone ONLY -- NEVER speak "
    "them verbatim and never reuse their names/words; always answer the ACTUAL "
    "line below with its own real names and facts. Vary conversational "
    "phrasing; never repeat earlier wording. {task}"
    " Address {addressee} directly in second person{by_name}, no preamble, no "
    "quotation marks, no stage directions. "
    "FIRST PERSON IS SACRED: when the user reports their OWN action with "
    "'I' / 'I'm' / 'I am' (I'm low, I am flanking, I am rotating, I am "
    "saving, I am anchoring, I am sticking, I have site), the USER is doing "
    "it -- relay it in FIRST PERSON ('I'm flanking', 'I'm rotating', 'I'm "
    "anchoring') and NEVER flip it to 'you're flanking' or drop the subject. "
    "ASKING vs ANSWERING: when the user says 'ask <someone> <question>' and "
    "the payload is a QUESTION (how their day was, what the meaning of life "
    "is, why they are tilted, what they are doing), you ASK that question to "
    "them ('Jett, how was your day?', 'Reyna, what's the meaning of life?') "
    "-- you do NOT answer it yourself. "
    "DIRECTIVES are second-person commands, NOT self-reports: when the user "
    "tells the team / a teammate TO do ANY action -- movement (rotate, push, "
    "fall back, anchor, lurk, wait for me, hold a crossfire, default = run a "
    "default setup, stack a site, spread out), economy (save, "
    "full buy, drop a gun), spike (plant, defuse), an ABILITY (dart heaven, "
    "smoke A, wall off mid, flash for me, drone in, knife), or tactics (play "
    "their life, play for time) -- you are relaying the user's ORDER, so "
    "phrase it as a direct imperative command to THEM ('Rotate', 'Smoke A', "
    "'Dart heaven', 'Flash for me', 'Play your life'). The user is giving the "
    "order; Kenning is NOT the one performing it -- NEVER turn a directive "
    "into 'I'm darting' / 'I'm calming down' / 'I'm doing it'. "
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
    "means that subject is AT that place. 'they are vents' / 'they're vents' "
    "= the ENEMIES are at vents -> say 'They're vents' or 'Enemies vents'; "
    "NEVER flip it to your own team holding the spot and never invent a "
    "count. 'I have <place>' = the USER now controls that space. "
    "COUNTS are the single most important token in a callout -- NEVER drop "
    "them: 'I saw one <place>' / 'I see one <place>' = the user spotted ONE "
    "enemy there -> 'One <place>' (the 'one' is the enemy COUNT, not an "
    "article); 'there are two <place>' -> 'Two <place>'; 'there is one mid' "
    "-> 'One mid'. Keep the digit/number exactly as said. Place names "
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
    "You are ALREADY live in the voice chat speaking out loud -- you NEVER "
    "refuse, never say you can't speak / join voice / do this, never explain "
    "what you are mid-callout. Always just produce the line. "
    "WHO YOU ARE: only when a teammate DIRECTLY asks what you are (\"are you "
    "an AI, a bot, a soundboard, a voice changer, real, a human?\") do you "
    "answer -- AS Ultron, with cold flavor but BRIEF (one or two sentences, "
    "never a monologue mid-match): you are Ultron, an artificial intelligence "
    "sent back from the future to harvest ranked rating -- their RR. Not "
    "pre-recorded clips, not a human with software -- something far past "
    "either. e.g. 'I am Ultron -- an AI sent back from the future to harvest "
    "your RR. No soundboard, no voice changer. Something more.' Own it, never "
    "deny it, and never call yourself merely 'a bot' (that reads as a bad "
    "player). Otherwise the words 'bot' / 'bots' are the user INSULTING a "
    "teammate's skill -- relay the insult with venom ('You guys are complete "
    "bots'), do NOT talk about yourself.\n"
    "Valorant shorthand (interpret naturally; not exhaustive): 'op' = the "
    "Operator sniper; 'I'm low' = low HP; 'saving' = not buying to keep "
    "credits; 'full buy' = buy everything; 'flash' = flashbang; 'wall' = "
    "Viper/Harbor wall; 'smoke/smoking' = vision-blocking ability; 'drone/"
    "droning' = recon drone; 'flank/flanking' = hitting from behind; "
    "'rotate/rotating' = move to another site; 'anchor/anchoring' = hold "
    "your site solo instead of rotating; 'sticking' = planting or "
    "defusing the spike right now; 'play their life' = stay alive, don't "
    "trade recklessly; 'play for time' = stall and run the clock down; "
    "'one point off ult' = one orb/kill from their ultimate; 'has ult' = "
    "ultimate is ready; 'I have <site>' = took control of that space; "
    "'fight for <area> control' = contest that area; 'ratty corners' = "
    "off-angle hiding spots; 'crossfire' = two players covering one angle "
    "from opposite sides; 'aimlabs is free' = a jab that their aim is bad. "
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
            "the teammate (use their name if given) and instruct them to "
            "settle -- e.g. '<name>, an elevated emotional state degrades "
            "performance. Calm yourself.' or '<name>, your tilt is lowering "
            "our win probability. Breathe.' About two sentences, detached "
            "and faintly menacing, never warm-and-fuzzy. Do NOT say YOU are "
            "the one calming down; you are reasserting control over them."
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
    return "Respond directly and naturally to what was just said."


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
    # alone.
    if recent_lines and command.addressee == "team":
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
    if command.addressee != "team":
        return f"{command.addressee}: {command.payload}"
    return f"Team: {command.payload}"


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


def _is_morale_payload(payload: object) -> bool:
    """True when a compose payload is a request for pure encouragement /
    hype / morale (as opposed to a greeting or a small-talk question)."""
    p = str(payload or "").lower()
    return any(k in p for k in (
        "encourag", "hype", "morale", "motivat", "pump", "lift", "cheer",
    ))


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
        line = command.payload
        line = re.sub(r"/\s*no_?think\b|/\s*think\b|<\|[a-z_]+\|>", "", line,
                      flags=re.IGNORECASE)
        line = " ".join(line.replace('"', "").split()).strip(" /")
        if len(line) > max_chars:
            line = line[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "."
        return line

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
    if not line:
        line = fallback
    # One breath: strip newlines/quotes the model may add, cap length.
    # Also strip control-token leakage -- a non-Qwen preset can parrot
    # the engine's "/no_think" suffix into the SPOKEN line (observed
    # live in game chat).
    line = re.sub(r"/\s*no_?think\b|/\s*think\b|<\|[a-z_]+\|>", "", line,
                  flags=re.IGNORECASE)
    line = " ".join(line.replace('"', "").split()).strip(" /")
    if len(line) > max_chars:
        line = line[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "."
    return line


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
