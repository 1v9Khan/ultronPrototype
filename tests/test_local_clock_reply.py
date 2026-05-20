"""2026-05-19 round 4: local clock / date short-circuit tests.

Live session: 'what time is it' triggered SEARCH gate + Brave +
crashed XTTS at 4595 tokens. The clock is on the user's computer;
consulting the network for it was absurd. ``maybe_local_clock_reply``
handles bare time/date asks directly.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from ultron.local_clock_reply import maybe_local_clock_reply


_FIXED_PM = datetime(2026, 5, 19, 14, 16, 0)  # Tuesday 2:16 PM
_FIXED_AM = datetime(2026, 5, 19, 8, 5, 0)    # Tuesday 8:05 AM
_FIXED_NOON = datetime(2026, 5, 19, 12, 0, 0)  # noon
_FIXED_MIDNIGHT = datetime(2026, 5, 19, 0, 0, 0)  # midnight


# ---------------------------------------------------------------------------
# Time queries -> spoken time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("utterance", [
    "what time is it",
    "What time is it?",
    "what's the time",
    "what is the time",
    "tell me the time",
    "please tell me the time",
    "do you know the time",
    "do you know what time it is",
    "got the time?",
    "current time",
    "current time please",
    "the time please",
    "can you tell me the time",
    "could you give me the time",
    "what time do you have",
    "ultron, what time is it",
    "hey ultron, what time is it",
    "and what time is it",
    "so what time is it",
    "What's the current time?",
    "What's the local time?",
])
def test_time_query_returns_spoken_time(utterance):
    reply = maybe_local_clock_reply(utterance, now=_FIXED_PM)
    assert reply is not None
    assert "2:16 PM" in reply
    assert reply.endswith(".")
    # TTS-safe: no AM/PM ligature, no colon ambiguity beyond the time
    # itself (xtts_v3.normalize_text_for_tts rewrites the colon).
    assert "2:16 PM" in reply


def test_time_query_at_noon_uses_12_pm():
    reply = maybe_local_clock_reply("what time is it", now=_FIXED_NOON)
    assert reply == "It's 12 PM."


def test_time_query_at_midnight_uses_12_am():
    reply = maybe_local_clock_reply("what time is it", now=_FIXED_MIDNIGHT)
    assert reply == "It's 12 AM."


def test_time_query_morning_format():
    reply = maybe_local_clock_reply("what time is it", now=_FIXED_AM)
    assert reply == "It's 8:05 AM."


def test_time_on_the_hour_omits_zero_minutes():
    """3 PM should read 'It's 3 PM.', not 'It's 3:00 PM.'"""
    three_pm = datetime(2026, 5, 19, 15, 0, 0)
    reply = maybe_local_clock_reply("what time is it", now=three_pm)
    assert reply == "It's 3 PM."


# ---------------------------------------------------------------------------
# Date queries -> spoken date
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("utterance", [
    "what day is it",
    "What day is today?",
    "what day of the week is it",
    "what's today",
    "what is today",
    "what's today's date",
    "what's the date",
    "what's the date today",
    "what is the date",
    "what is the current date",
    "what's the day",
    "tell me the date",
    "tell me today's date",
    "give me the date",
    "today's date",
    "ultron, what day is it",
    "and what's today's date",
])
def test_date_query_returns_spoken_date(utterance):
    reply = maybe_local_clock_reply(utterance, now=_FIXED_PM)
    assert reply is not None
    assert "Tuesday" in reply
    assert "May" in reply
    assert "19" in reply
    assert reply.endswith(".")


# ---------------------------------------------------------------------------
# Mixed-intent / richer queries DO NOT short-circuit (fall through to LLM)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("utterance", [
    "what time is it in Paris",                   # other timezone
    "what time is it and the weather",            # mixed intent
    "what time should I leave",                   # different question
    "what time does the store open",              # external knowledge
    "tell me about time",                         # not a time ask
    "how do I save time",                         # different
    "is it time for lunch",                       # different
    "what was the time of the meeting",           # historical
    "what's the date of the meeting",             # event date, not today
    "when did World War 2 end",                   # historical date
    "say hello",                                  # unrelated
    "what's the weather",                         # unrelated
    "open Chrome",                                # unrelated
])
def test_mixed_or_richer_queries_fall_through(utterance):
    assert maybe_local_clock_reply(utterance, now=_FIXED_PM) is None


# ---------------------------------------------------------------------------
# Empty / invalid input
# ---------------------------------------------------------------------------


def test_empty_input_returns_none():
    assert maybe_local_clock_reply("", now=_FIXED_PM) is None
    assert maybe_local_clock_reply(None, now=_FIXED_PM) is None
    assert maybe_local_clock_reply("   ", now=_FIXED_PM) is None


def test_default_clock_uses_datetime_now():
    """When ``now`` is omitted the helper uses datetime.now(). Smoke
    test that it returns SOMETHING (we can't pin the exact value)."""
    reply = maybe_local_clock_reply("what time is it")
    assert reply is not None
    assert reply.startswith("It's ")
    assert reply.endswith(".")
