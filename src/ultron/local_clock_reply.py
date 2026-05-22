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

try:
    from zoneinfo import ZoneInfo
    _ZONEINFO_AVAILABLE = True
except ImportError:                                          # pragma: no cover
    _ZONEINFO_AVAILABLE = False
    ZoneInfo = None  # type: ignore


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
    # STT-artifact prefixes that Moonshine sometimes prepends to short
    # utterances ("you", "yeah", "uh", "um"). Tolerating them here so
    # "you What time is it in Paris?" still routes to local clock.
    (?:and\s+|so\s+|then\s+|but\s+|you\s+|yeah\s+|uh\s+|um\s+|hmm\s+)?
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
    # STT-artifact prefixes that Moonshine sometimes prepends to short
    # utterances ("you", "yeah", "uh", "um"). Tolerating them here so
    # "you What time is it in Paris?" still routes to local clock.
    (?:and\s+|so\s+|then\s+|but\s+|you\s+|yeah\s+|uh\s+|um\s+|hmm\s+)?
    (?:
        # Day of week
        what\s+day\s+(?:is\s+(?:it|today)|of\s+the\s+week(?:\s+is\s+it)?)\b
      | what(?:'s|s|\s+is)?\s+today(?:'s\s+date)?\b
      | what(?:'s|s|\s+is)?\s+the\s+(?:current\s+)?date(?:\s+today)?\b
      | what(?:'s|s|\s+is)?\s+the\s+day\b
      | (?:tell|give|show)\s+me\s+(?:the\s+date|today's\s+date|the\s+day)\b
      | today's\s+date\b
      # 2026-05-19 round 5: Whisper-mangled variants. Live session
      # gave "That's today's date." for "What's today's date?" -- the
      # detector below treats the declarative form as a question too.
      | that(?:'s|s|\s+is)?\s+today(?:'s\s+date)?\b
      | that(?:'s|s|\s+is)?\s+the\s+date(?:\s+today)?\b
      | (?:do|d')\s+you\s+know\s+(?:the\s+date|today's\s+date|what\s+(?:the\s+)?date\s+(?:it\s+)?is)\b
    )
    \s*[.!?]?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Timezone-aware "what time is it in X" (2026-05-22)
# ---------------------------------------------------------------------------
#
# Live session showed "What time is it in Paris?" hit the preflight LLM
# gate, which decided "no search needed" (incorrect -- the wall-clock
# time changes every second), and the LLM then regurgitated a stale time
# from RAG. The fix below short-circuits the common case to a local
# zoneinfo lookup, returning a fresh time. Unknown cities fall through
# to the gate (which the new ``_TIME_IN_LOCATION_RE`` rule forces to
# SEARCH so the LLM gets fresh data instead of stale RAG).

# Map of normalized city names (lowercase, no punctuation) to IANA
# timezone identifiers. Limited to widely-recognised cities so the
# regex doesn't false-match arbitrary nouns. Extend as needed -- new
# entries must use the IANA name (verifiable via `python -c "import
# zoneinfo; print(sorted(zoneinfo.available_timezones())[:5])"`).
_CITY_TIMEZONES = {
    # North America
    "new york": "America/New_York",
    "nyc": "America/New_York",
    "boston": "America/New_York",
    "atlanta": "America/New_York",
    "miami": "America/New_York",
    "toronto": "America/Toronto",
    "chicago": "America/Chicago",
    "dallas": "America/Chicago",
    "houston": "America/Chicago",
    "denver": "America/Denver",
    "phoenix": "America/Phoenix",
    "los angeles": "America/Los_Angeles",
    "la": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles",
    "seattle": "America/Los_Angeles",
    "vancouver": "America/Vancouver",
    "anchorage": "America/Anchorage",
    "honolulu": "Pacific/Honolulu",
    "mexico city": "America/Mexico_City",
    # Europe
    "london": "Europe/London",
    "dublin": "Europe/Dublin",
    "paris": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "frankfurt": "Europe/Berlin",
    "munich": "Europe/Berlin",
    "hamburg": "Europe/Berlin",
    "cologne": "Europe/Berlin",
    "amsterdam": "Europe/Amsterdam",
    "brussels": "Europe/Brussels",
    "vienna": "Europe/Vienna",
    "zurich": "Europe/Zurich",
    "geneva": "Europe/Zurich",
    "madrid": "Europe/Madrid",
    "barcelona": "Europe/Madrid",
    "lisbon": "Europe/Lisbon",
    "rome": "Europe/Rome",
    "milan": "Europe/Rome",
    "athens": "Europe/Athens",
    "warsaw": "Europe/Warsaw",
    "prague": "Europe/Prague",
    "budapest": "Europe/Budapest",
    "copenhagen": "Europe/Copenhagen",
    "stockholm": "Europe/Stockholm",
    "oslo": "Europe/Oslo",
    "helsinki": "Europe/Helsinki",
    "reykjavik": "Atlantic/Reykjavik",
    "moscow": "Europe/Moscow",
    "saint petersburg": "Europe/Moscow",
    "kiev": "Europe/Kiev",
    "kyiv": "Europe/Kiev",
    "istanbul": "Europe/Istanbul",
    # Middle East
    "dubai": "Asia/Dubai",
    "abu dhabi": "Asia/Dubai",
    "riyadh": "Asia/Riyadh",
    "doha": "Asia/Qatar",
    "tel aviv": "Asia/Jerusalem",
    "jerusalem": "Asia/Jerusalem",
    "tehran": "Asia/Tehran",
    # South Asia
    "mumbai": "Asia/Kolkata",
    "delhi": "Asia/Kolkata",
    "bangalore": "Asia/Kolkata",
    "karachi": "Asia/Karachi",
    "dhaka": "Asia/Dhaka",
    # Southeast Asia
    "bangkok": "Asia/Bangkok",
    "ho chi minh city": "Asia/Ho_Chi_Minh",
    "saigon": "Asia/Ho_Chi_Minh",
    "hanoi": "Asia/Ho_Chi_Minh",
    "jakarta": "Asia/Jakarta",
    "manila": "Asia/Manila",
    "singapore": "Asia/Singapore",
    "kuala lumpur": "Asia/Kuala_Lumpur",
    # East Asia
    "hong kong": "Asia/Hong_Kong",
    "taipei": "Asia/Taipei",
    "shanghai": "Asia/Shanghai",
    "beijing": "Asia/Shanghai",
    "tokyo": "Asia/Tokyo",
    "osaka": "Asia/Tokyo",
    "seoul": "Asia/Seoul",
    # Oceania
    "sydney": "Australia/Sydney",
    "melbourne": "Australia/Melbourne",
    "brisbane": "Australia/Brisbane",
    "perth": "Australia/Perth",
    "auckland": "Pacific/Auckland",
    "wellington": "Pacific/Auckland",
    # South America
    "sao paulo": "America/Sao_Paulo",
    "rio de janeiro": "America/Sao_Paulo",
    "buenos aires": "America/Argentina/Buenos_Aires",
    "santiago": "America/Santiago",
    "lima": "America/Lima",
    "bogota": "America/Bogota",
    "caracas": "America/Caracas",
    # Africa
    "cairo": "Africa/Cairo",
    "lagos": "Africa/Lagos",
    "nairobi": "Africa/Nairobi",
    "johannesburg": "Africa/Johannesburg",
    "cape town": "Africa/Johannesburg",
    "addis ababa": "Africa/Addis_Ababa",
    "casablanca": "Africa/Casablanca",
}


# Same lead-in tolerance as _TIME_QUERY_RE but trailing "in <city>".
# The city group captures everything up to the trailing punctuation.
_TIME_IN_LOCATION_RE = re.compile(
    r"""
    ^\s*
    (?:(?:hey\s+|hi\s+|ok\s+|okay\s+)?ultron[,\s]+)?
    # STT-artifact prefixes that Moonshine sometimes prepends to short
    # utterances ("you", "yeah", "uh", "um"). Tolerating them here so
    # "you What time is it in Paris?" still routes to local clock.
    (?:and\s+|so\s+|then\s+|but\s+|you\s+|yeah\s+|uh\s+|um\s+|hmm\s+)?
    (?:
        what(?:'s|s|\s+is)?\s+(?:the\s+)?(?:current\s+|local\s+)?time
      | what\s+time\s+(?:is\s+it|do\s+they\s+have)
      | (?:please\s+)?(?:tell|give|show)\s+me\s+the\s+(?:current\s+|local\s+)?time
      | current\s+time
      | the\s+(?:current\s+|local\s+)?time
    )
    \s+in\s+(?P<city>[A-Za-z][A-Za-z\s.]*?)
    \s*[.!?]?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _maybe_city_time_reply(text: str, now: datetime) -> Optional[str]:
    """Return a spoken city-time reply when ``text`` is a "what time
    is it in X" ask AND ``X`` is in the timezone map. Returns ``None``
    when the regex doesn't match OR the city is unknown -- the caller
    falls through to the SEARCH path so the user still gets an answer."""
    if not _ZONEINFO_AVAILABLE:
        return None
    m = _TIME_IN_LOCATION_RE.match(text)
    if m is None:
        return None
    city_raw = (m.group("city") or "").strip().rstrip(".!?").lower()
    if not city_raw:
        return None
    tz_name = _CITY_TIMEZONES.get(city_raw)
    if tz_name is None:
        return None
    try:
        zone = ZoneInfo(tz_name)
    except Exception:                                        # pragma: no cover
        return None
    # Use the system clock anchored to UTC, then convert to the
    # target zone. ``now`` is assumed to be in local time; if it's
    # naive we treat it as the system wall clock by converting via
    # astimezone() (which Python interprets as local time when naive).
    try:
        local_now = (
            now.astimezone(zone) if now.tzinfo is not None
            else now.astimezone().astimezone(zone)
        )
    except Exception:                                        # pragma: no cover
        return None
    # Render "It's 9:25 PM in Paris."
    spoken = _render_time_for_tts(local_now).rstrip(".")
    # Replace "It's" with "In <City>, it's" for natural phrasing.
    city_display = m.group("city").strip().rstrip(".!?")
    # Title-case for spoken form ("paris" -> "Paris").
    city_display = " ".join(w.capitalize() for w in city_display.split())
    return f"In {city_display}, {spoken[0].lower()}{spoken[1:]}."


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
    # 2026-05-22: "what time is it in <city>" via zoneinfo lookup.
    # Falls through to None when the city isn't in the map -- the
    # gate's _TIME_IN_LOCATION_RE rule then forces SEARCH so the
    # LLM gets fresh data instead of stale RAG.
    city_reply = _maybe_city_time_reply(text, clock)
    if city_reply:
        return city_reply
    return None


__all__ = [
    "maybe_local_clock_reply",
]
