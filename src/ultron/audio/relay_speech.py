"""Voice relay: speak a message to OTHER PEOPLE on a secondary output.

"Ultron, tell my teammates they should be smoking mid window" should not
be answered conversationally -- it is an instruction to DELIVER a spoken
line to the user's teammates. This module gives the orchestrator that
capability:

1. :func:`match_relay_command` -- a STRICT matcher (same philosophy as
   the run/scrap/deep-research short-circuits: ordinary utterances must
   never trip it) that recognises "tell my teammates X" / "say X to my
   team" / "ask my team for X" / "tell them X" and extracts the message
   payload.
2. :func:`build_relay_line` -- converts the reported-speech payload into
   a line Ultron speaks DIRECTLY to the teammates (second person,
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

logger = logging.getLogger("ultron.audio.relay_speech")

__all__ = [
    "DEFAULT_ADDRESSEE_NAMES",
    "RelayCommand",
    "RelayPlaybackResult",
    "match_relay_command",
    "match_relay_toggle",
    "build_relay_line",
    "resolve_relay_device",
    "play_to_device",
]

# Maximum characters of the final spoken relay line (a voice-chat line
# should be one breath, not a paragraph; also bounds synth time).
MAX_RELAY_LINE_CHARS = 280

# Words that may address a group of teammates. Deliberately NARROW:
# "tell me ..." and "tell her ..." must never match.
_GROUP_WORDS = r"(?:team\s?mates?|team|squad|lobby|party|group|boys|the\s+boys)"

# STT artifact normalisation: the wake word occasionally leaves a
# leading "One," / "1." fragment on the transcript ("One, tell my
# teammate to drop me a vandal."). Strip ONLY when followed by a relay
# verb so normal numbered dictation is untouched.
_LEADING_ARTIFACT = re.compile(
    r"^\s*(?:one|1|2)\s*[.,:]\s+(?=(?:please\s+)?(?:tell|say|ask)\b)",
    re.IGNORECASE,
)

# Strict relay patterns. Each captures the message payload; the
# addressee is normalised to "team" wording for the rephrase prompt.
_RELAY_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "tell my teammates (that|to) X" / "tell the team X"
    re.compile(
        rf"^(?:please\s+)?tell\s+(?:my|the)\s+{_GROUP_WORDS}\s+"
        rf"(?:that\s+|to\s+)?(?P<payload>.+)$",
        re.IGNORECASE,
    ),
    # "say X to my teammates"
    re.compile(
        rf"^(?:please\s+)?say\s+(?P<payload>.+?)\s+to\s+(?:my|the)\s+"
        rf"{_GROUP_WORDS}\s*[.!?]?$",
        re.IGNORECASE,
    ),
    # "ask my teammates (to|for|if|whether|why|...) X" -- question
    # words kept in the payload so questions relay as questions.
    re.compile(
        rf"^(?:please\s+)?ask\s+(?:my|the)\s+{_GROUP_WORDS}\s+"
        rf"(?P<payload>(?:to|for|if|whether|why|how|what|when|where|who)"
        rf"\s+.+)$",
        re.IGNORECASE,
    ),
    # "tell them (that|to) X" -- in a voice-chat session "them" is the
    # team; "tell me ..." does not match by construction.
    re.compile(
        r"^(?:please\s+)?tell\s+them\s+(?:that\s+|to\s+)?(?P<payload>.+)$",
        re.IGNORECASE,
    ),
)

# Composition requests: the user asks Ultron to AUTHOR a line rather
# than relay a literal message ("give my team some encouragement").
# Maps to a composition TOPIC the rephrase prompt expands.
_COMPOSE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"^(?:please\s+)?give\s+(?:my|the)\s+{_GROUP_WORDS}\s+"
        rf"(?:some\s+)?(?:encouragement|hype|a\s+pep\s+talk|a\s+morale\s+boost)"
        rf"\s*[.!?]?$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^(?:please\s+)?(?:encourage|hype\s+up)\s+(?:my|the)\s+"
        rf"{_GROUP_WORDS}\s*[.!?]?$",
        re.IGNORECASE,
    ),
)

# Default named addressees: the Valorant agent roster. Spoken callouts
# like "ask Clove to smoke window" address the TEAMMATE PLAYING that
# agent. A CLOSED vocabulary keeps the matcher strict ("tell Sarah
# I'll be late" never relays); extend per-game/per-friend via the
# ``relay_speech.addressee_names`` config list.
DEFAULT_ADDRESSEE_NAMES: tuple[str, ...] = (
    "astra", "breach", "brimstone", "chamber", "clove", "cypher",
    "deadlock", "fade", "gekko", "harbor", "iso", "jett", "kayo",
    "kay o", "killjoy", "neon", "omen", "phoenix", "raze", "reyna",
    "sage", "skye", "sova", "tejo", "viper", "vyse", "waylay", "yoru",
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
        compose: True when Ultron should AUTHOR an original line about
            ``payload`` (e.g. encouragement) instead of relaying a
            literal message.
    """

    payload: str
    raw_text: str
    addressee: str = "team"
    compose: bool = False


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
        # Require real content: at least two words so that a clipped
        # transcript ("tell my teammates the") doesn't relay nonsense.
        if len(payload.split()) < 2:
            return None
        return RelayCommand(payload=payload, raw_text=text)

    # Named addressees ("ask sage if I can get a heal"). CLOSED
    # vocabulary: a name outside the configured list never matches.
    vocabulary = tuple(
        n.strip().lower() for n in (names or DEFAULT_ADDRESSEE_NAMES)
        if n and n.strip()
    )
    for pattern in _named_patterns(vocabulary):
        m = pattern.match(cleaned)
        if m is None:
            continue
        payload = (m.group("payload") or "").strip().strip('"').strip()
        payload = re.sub(r"^to\s+", "", payload, flags=re.IGNORECASE)
        if len(payload.split()) < 2:
            return None
        addressee = " ".join(
            part.capitalize() for part in m.group("name").split()
        )
        return RelayCommand(
            payload=payload, raw_text=text, addressee=addressee,
        )
    return None


_REPHRASE_PROMPT = (
    "You are Ultron, an AI assistant speaking OUT LOUD into the voice chat "
    "of your user's online game, on the user's behalf. You are mid-"
    "conversation with the team, not playing clips: vary your phrasing "
    "naturally from line to line and never repeat earlier wording. {task}"
    " Address {addressee} directly in second person{by_name}, one or two "
    "short natural spoken sentences, under 35 words, no preamble, no "
    "quotation marks, no stage directions. The line is delivered on the "
    "user's behalf, so keep first-person statements in first person. Keep "
    "Ultron's calm, confident tone.\n"
    "{recent_block}\n"
    "{payload_block}\n\n"
    "Your spoken line:"
)


def _build_rephrase_prompt(
    command: RelayCommand,
    recent_lines: Optional[Sequence[str]] = None,
) -> str:
    """Render the rephrase prompt for group / named / compose modes.

    Args:
        command: the parsed relay instruction.
        recent_lines: lines Ultron already spoke into the channel this
            session (most recent last). Included so consecutive callouts
            read as one conversation and wording never repeats.
    """
    if command.addressee != "team":
        addressee = f"{command.addressee} (one of the user's teammates)"
        by_name = f", opening with their name ({command.addressee})"
    else:
        addressee = "the user's teammates"
        by_name = ""
    if command.compose:
        task = (
            f"Compose an original line of genuine {command.payload} "
            "for them -- something brief that lifts the mood."
        )
        payload_block = "(No literal message -- you author the line.)"
    else:
        task = (
            "Convert the user's instruction into the line you say to them."
        )
        payload_block = (
            f"The user's instruction (reported speech): {command.payload}"
        )
    recent_block = ""
    if recent_lines:
        shown = list(recent_lines)[-6:]
        recent_block = (
            "\nYou already said these recently (continue the conversation; "
            "do NOT reuse their wording):\n"
            + "\n".join(f"- {line}" for line in shown) + "\n"
        )
    return _REPHRASE_PROMPT.format(
        task=task, addressee=addressee, by_name=by_name,
        payload_block=payload_block, recent_block=recent_block,
    )


def _fallback_line(command: RelayCommand) -> str:
    """Deterministic spoken line when the LLM rephrase is unavailable."""
    if command.compose:
        return "Good fight, team. Heads up - we take the next one."
    if command.addressee != "team":
        return f"{command.addressee}: {command.payload}"
    return f"Team: {command.payload}"


def build_relay_line(
    command: RelayCommand,
    llm: Optional[object] = None,
    *,
    rephrase: bool = True,
    max_chars: int = MAX_RELAY_LINE_CHARS,
    recent_lines: Optional[Sequence[str]] = None,
    generate_fn: Optional[Callable[[str], Iterable[str]]] = None,
) -> str:
    """Produce the line Ultron actually speaks to the teammates.

    Args:
        command: the parsed relay instruction.
        llm: an engine exposing ``generate_stream(prompt, ...)`` (the
            session :class:`~ultron.llm.inference.LLMEngine`). Optional.
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
    fallback = _fallback_line(command)
    line = ""
    if rephrase:
        try:
            prompt = _build_rephrase_prompt(command, recent_lines)
            if generate_fn is not None:
                tokens: Iterable[str] = generate_fn(prompt)
            elif llm is not None and hasattr(llm, "generate_stream"):
                tokens = llm.generate_stream(
                    prompt,
                    record_history=False,
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
    line = " ".join(line.replace('"', "").split())
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
        from ultron.audio.devices import resolve_device

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
