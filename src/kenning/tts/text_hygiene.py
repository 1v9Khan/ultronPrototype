"""Pre-synthesis text hygiene: never SPEAK model artifacts.

2026-06-11 live incident (3B gaming preset): the TTS spoke a roleplay
stage direction out loud ("*repositions window on monitor to show a
blank, dark screen*"), verbalised a parroted control token ("No
think."), and rendered a bare quote character as its own clip. The
model side is also being fixed (preset-conditional ``/no_think``), but
the SPEAKER must defend independently: whatever model is active,
asterisk-wrapped stage directions, control tokens, and
punctuation-only fragments never reach the voice.

One pure function, applied at the synthesis choke point
(:meth:`KokoroSpeech._synthesize`), so every spoken surface -- normal
responses, acks, the team relay -- is covered.
"""

from __future__ import annotations

import re

__all__ = ["sanitize_spoken_text"]

# Asterisk- or bracket-wrapped stage directions: *nods slowly*,
# [sighs]. Bounded span so a legitimate lone asterisk in dictated text
# can't swallow a whole sentence.
_STAGE_DIRECTION = re.compile(r"\*[^*\n]{1,200}\*|\[[^\]\n]{1,80}\]")
# Chat-template / thinking-control tokens a non-native model may parrot.
_CONTROL_TOKENS = re.compile(
    r"<think>.{0,400}?</think>"
    r"|/\s*no_?think\b|/\s*think\b|\bno_think\b"  # marker, with or without slash
    r"|<\|[a-zA-Z0-9_]+\|>|</?think>",
    re.IGNORECASE | re.DOTALL,
)
_HAS_SPEAKABLE = re.compile(r"[A-Za-z0-9]")

# Proper nouns the Kokoro G2P mis-reads as INITIALISMS: an all-caps token (or a
# dotted form) is spelled out letter-by-letter ("JARVIS" -> "J. A. R. V. I. S").
# Map them to a spoken-word spelling so they're read as a name. Case-insensitive
# and dot-tolerant so "JARVIS", "Jarvis", and "J.A.R.V.I.S." all normalise.
_PRONUNCIATION = (
    (re.compile(r"\bJ\.?A\.?R\.?V\.?I\.?S\b", re.IGNORECASE), "Jarvis"),
)


def sanitize_spoken_text(text: str) -> str:
    """Strip unspeakable artifacts; return "" when nothing remains.

    Args:
        text: a sentence/clip about to be synthesized.

    Returns:
        The cleaned text, or ``""`` when the input contains no
        speakable content (callers skip synthesis entirely).
    """
    if not text:
        return ""
    cleaned = _STAGE_DIRECTION.sub(" ", text)
    cleaned = _CONTROL_TOKENS.sub(" ", cleaned)
    for _pat, _rep in _PRONUNCIATION:
        cleaned = _pat.sub(_rep, cleaned)
    cleaned = " ".join(cleaned.split())
    # Drop leading/trailing orphaned quote marks left by stripped spans.
    cleaned = cleaned.strip(' "“”/')
    if not _HAS_SPEAKABLE.search(cleaned):
        return ""
    return cleaned
