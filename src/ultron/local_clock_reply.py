"""2026-05-19 round 4 fix: local clock / date short-circuit.

The live session showed Ultron asking "what time is it?" triggering a
SEARCH gate, hitting brave.com + NIST, then crashing XTTS with a
4595-token overflow error -- the user got nothing. Even when search
succeeds, asking the network for the wall-clock time is absurd: the
computer has a clock, and reading it is sub-millisecond. This module
recognises bare time / date queries and produces a TTS-safe spoken
form directly, bypassing the gate and the LLM entirely.

The handler is intentionally conservative -- only fires when the
utterance is unambiguously a time/date ask (no follow-on clauses, no
mixed intent). Anything richer than "what time is it?" or "what day
is it today?" falls through to the normal LLM path.

Empty / None input returns ``None`` (no handling). The reply is
already normalised in TTS-friendly form (digit-by-digit clock,
no AM/PM ligatures, weekday spelled out) so it can be passed
directly to ``speak`` / streamed.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Pattern matchers
# ---------------------------------------------------------------------------


# Strict: the WHOLE utterance is a time question. Optional leading
# "ultron," / "hey ultron," / "and " / "so " is tolerated. Trailing
# punctuation only. No mixed intent ("what time is it and the
# weather"). Matches: "what time is it" / "what is the time" /
# "tell me the time" / "do you know what time it is" / "the time
# please" / "current time" / "got the time".
_TIME_QUERY_RE = re.compile(
    r"""
    ^\s*
    (?:(?:hey\s+|hi\s+|ok\s+|okay\s+)?ultron[,\s]+)?
    (?:and\s+|so\s+|then\s+|but\s+)?
    (?:
        # "what time is it" / "what's the time" / "what is the time"
        what(?:'s|s|\s+is)?\s+(?:the\s+)?(?:current\s+|local\s+)?time\b
      | what\s+time\s+(?:is\s+it|do\s+you\s+have)\b
      | (?:please\s+)?(?:tell|give|show)\s+me\s+the\s+(?:current\s+|local\s+)?time\b
      | do\s+you\s+(?:know|have)\s+(?:what\s+time\s+(?:it\s+is|is\s+it))\b
      | do\s+you\s+(?:know|have)\s+(?:the\s+)?time\b
      | got\s+(?:the\s+|a\s+)?time\b
      | the\s+(?:current\s+|local\s+)?time(?:\s+please)?\b
      | (?:could|can)\s+you\s+(?:tell|give|read)\s+me\s+the\s+time\b
      | current\s+time(?:\s+please)?\b
    )
    \s*[.!?]?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


_DATE_QUERY_RE = re.compile(
    r"""
    ^\s*
    (?:(?:hey\s+|hi\s+|ok\s+|okay\s+)?ultron[,\s]+)?
    (?:and\s+|so\s+|then\s+|but\s+)?
    (?:
        # Day of week
        what\s+day\s+(?:is\s+(?:it|today)|of\s+the\s+week(?:\s+is\s+it)?)\b
      | what(?:'s|s|\s+is)?\s+today(?:'s\s+date)?\b
      | what(?:'s|s|\s+is)?\s+the\s+(?:current\s+)?date(?:\s+today)?\b
      | what(?:'s|s|\s+is)?\s+the\s+day\b
      | (?:tell|give|show)\s+me\s+(?:the\s+date|today's\s+date|the\s+day)\b
      | today's\s+date\b
    )
    \s*[.!?]?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Spoken-form renderers (TTS-safe -- no AM/PM ligature, no "Mon."
# abbreviations, no digit-clumping).
# ---------------------------------------------------------------------------


def _render_time_for_tts(now: datetime) -> str:
    """Render the wall-clock time as a TTS-friendly spoken string.

    Format: "It's 2:16 PM." -- emitted with leading article + period
    so XTTS treats it as a complete sentence. The hour-minute split
    relies on :func:`ultron.tts.xtts_v3.normalize_text_for_tts`'s
    AM/PM rewriter to convert ``PM`` -> ``P M`` on the TTS side, so
    the user hears clearly-spoken letters instead of a slurred
    ligature.
    """
    hour = now.hour
    minute = now.minute
    am_pm = "AM" if hour < 12 else "PM"
    hour_12 = hour % 12
    if hour_12 == 0:
        hour_12 = 12
    if minute == 0:
        # "It's 2 PM." reads more naturally than "It's 2:00 PM."
        return f"It's {hour_12} {am_pm}."
    return f"It's {hour_12}:{minute:02d} {am_pm}."


def _render_date_for_tts(now: datetime) -> str:
    """Render today's date as a TTS-friendly spoken string.

    Format: "Today is Tuesday, May 19." -- weekday spelled out,
    month spelled out, day as a plain number (the year is omitted
    because the user is asking about today; year-explicit asks fall
    through to the LLM).
    """
    weekday = now.strftime("%A")
    month = now.strftime("%B")
    day = now.day
    return f"Today is {weekday}, {month} {day}."


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def maybe_local_clock_reply(
    user_text: str,
    *,
    now: Optional[datetime] = None,
) -> Optional[str]:
    """Return a spoken reply if ``user_text`` is a bare time / date ask.

    Returns ``None`` for anything else so the caller falls through to
    the normal LLM path. The detector is deliberately strict -- mixed
    intents ("what time is it and the weather in Paris") DO NOT match,
    because answering only the time would drop the rest of the query.

    Args:
        user_text: the transcribed user utterance.
        now: injectable clock for tests. Defaults to ``datetime.now()``
            in the caller's local timezone (system wall clock).

    Returns:
        A complete sentence (with trailing period) ready to be passed
        to TTS, or ``None`` when the query isn't a bare time/date ask.
    """
    if not user_text:
        return None
    text = user_text.strip()
    if not text:
        return None
    clock = now if now is not None else datetime.now()
    if _TIME_QUERY_RE.match(text):
        return _render_time_for_tts(clock)
    if _DATE_QUERY_RE.match(text):
        return _render_date_for_tts(clock)
    return None


__all__ = [
    "maybe_local_clock_reply",
]
