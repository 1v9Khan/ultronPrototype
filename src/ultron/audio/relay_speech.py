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

import logging
import re
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np

logger = logging.getLogger("ultron.audio.relay_speech")

__all__ = [
    "RelayCommand",
    "RelayPlaybackResult",
    "match_relay_command",
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
    # "ask my teammates (to|for|if|whether) X"
    re.compile(
        rf"^(?:please\s+)?ask\s+(?:my|the)\s+{_GROUP_WORDS}\s+"
        rf"(?P<payload>(?:to|for|if|whether)\s+.+)$",
        re.IGNORECASE,
    ),
    # "tell them (that|to) X" -- in a voice-chat session "them" is the
    # team; "tell me ..." does not match by construction.
    re.compile(
        r"^(?:please\s+)?tell\s+them\s+(?:that\s+|to\s+)?(?P<payload>.+)$",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class RelayCommand:
    """A parsed "speak to my teammates" instruction.

    Attributes:
        payload: the message content in the user's reported-speech form
            (e.g. "they should be smoking mid window every round").
        raw_text: the full original utterance, for logging/diagnostics.
    """

    payload: str
    raw_text: str


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


def match_relay_command(text: str) -> Optional[RelayCommand]:
    """Match a strict "tell my teammates X" style relay instruction.

    Args:
        text: the user's transcript for this turn.

    Returns:
        A :class:`RelayCommand` with the extracted payload, or None when
        the utterance is not a relay instruction (ordinary questions,
        "tell me ..." requests, and bare "tell my teammates" with no
        message all fall through).
    """
    if not text:
        return None
    cleaned = _LEADING_ARTIFACT.sub("", text.strip())
    for pattern in _RELAY_PATTERNS:
        m = pattern.match(cleaned)
        if m is None:
            continue
        payload = (m.group("payload") or "").strip().strip('"').strip()
        # Require real content: at least two words so that a clipped
        # transcript ("tell my teammates the") doesn't relay nonsense.
        if len(payload.split()) < 2:
            return None
        return RelayCommand(payload=payload, raw_text=text)
    return None


_REPHRASE_PROMPT = (
    "You are Ultron, an AI assistant speaking OUT LOUD into the voice chat "
    "of your user's online game, addressing the user's teammates on his "
    "behalf. Convert the user's instruction into the line you say to the "
    "teammates: address THEM directly in second person, one or two short "
    "natural spoken sentences, under 35 words, no preamble, no quotation "
    "marks, no stage directions. Keep Ultron's calm, confident tone.\n\n"
    "The user's instruction (reported speech): {payload}\n\n"
    "Your spoken line:"
)


def build_relay_line(
    command: RelayCommand,
    llm: Optional[object] = None,
    *,
    rephrase: bool = True,
    max_chars: int = MAX_RELAY_LINE_CHARS,
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
        generate_fn: test seam -- a ``prompt -> token iterable`` callable
            used INSTEAD of ``llm.generate_stream`` when provided.

    Returns:
        A non-empty spoken line. Fail-open: any LLM failure returns the
        deterministic "Team: <payload>" fallback rather than raising.
    """
    fallback = f"Team: {command.payload}"
    line = ""
    if rephrase:
        try:
            prompt = _REPHRASE_PROMPT.format(payload=command.payload)
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
