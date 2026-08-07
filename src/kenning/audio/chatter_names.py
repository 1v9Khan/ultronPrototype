"""Resolve a SPOKEN name to a known Twitch chatter (2026-07-26).

THE PROBLEM
-----------
"say hi to Izumi" is ambiguous to a regex and always will be: Izumi could be a
teammate or a viewer, and the sentence does not say which. The tell-chat
matcher needs a chat marker ("in chat"), so without one the utterance falls
through to the relay matcher, which defaults ``addressee='team'`` and parrots
the words onto the TEAM MIC:

    raw='Say hi to Izumi.'  ->  route='relay_llm'  channel='team_mic'

Meanwhile ``data/twitch/welcomed.db`` has been recording every chatter the
stream has welcomed -- 28 of them, including ``izumiikiryo``. The information
needed to disambiguate already exists; nothing was asking for it.

WHY SPOKEN NAMES NEED FUZZY MATCHING
------------------------------------
Twitch logins are not what people say out loud. The streamer says "Izumi"; the
login is ``izumiikiryo``. "IceMapple" is ``icemapple14``. So an exact lookup
finds nothing useful -- resolution has to handle a spoken PREFIX of a longer
login, and STT spelling drift on top of that.

Matching is deliberately conservative, because a false positive sends a team
callout into public chat:

* a spoken name must be >= ``_MIN_LEN`` characters (so "hi"/"yo"/"em" can never
  resolve);
* it must PREFIX a login, or clear a high fuzzy ratio -- not merely be
  "similar";
* anything that is a Valorant agent name is rejected outright, before any
  lookup, so "say hi to Sage" stays a team callout forever;
* an ambiguous spoken name matching two different logins resolves to NEITHER.

Anticheat (BR-P1): stdlib + rapidfuzz only, both on the allowlist. sqlite3 is
stdlib. No network.
"""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Iterable, Sequence

from kenning.utils.logging import get_logger

logger = get_logger("audio.chatter_names")

__all__ = [
    "known_chatters",
    "resolve_chatter",
    "invalidate_cache",
]

#: A spoken name shorter than this can never resolve. "hi", "yo", "em", "sup"
#: are greeting words that would otherwise prefix-match a login.
_MIN_LEN = 4

#: Fuzzy floor for a non-prefix match. High on purpose: a wrong resolution
#: publishes a team callout to public chat, so "close" is not good enough.
_FUZZY_FLOOR = 88

_CACHE_TTL_SECONDS = 60.0
_cache: tuple[float, frozenset[str]] | None = None


def _agent_names() -> set[str]:
    """Valorant agent names, lowercased. These are NEVER chatters.

    Read from the relay roster so a future agent addition is picked up here
    automatically rather than needing a second list to be maintained.
    """
    try:
        from kenning.audio._stt_correct import _AGENTS
        return {a.lower().replace("/", "") for a in _AGENTS}
    except Exception:                                            # noqa: BLE001
        return set()


def invalidate_cache() -> None:
    """Drop the cached chatter set (used by tests, and after a new welcome)."""
    global _cache
    _cache = None


def known_chatters(db_path: str | os.PathLike | None = None) -> frozenset[str]:
    """Logins the stream has welcomed, lowercased. Empty set on any problem.

    Cached for ``_CACHE_TTL_SECONDS`` -- this is consulted on the voice hot
    path and the underlying table changes at most once per new viewer.
    """
    global _cache
    now = time.monotonic()
    if _cache is not None and (now - _cache[0]) < _CACHE_TTL_SECONDS:
        return _cache[1]
    path = db_path
    if path is None:
        try:
            from kenning.config import PROJECT_ROOT
            path = PROJECT_ROOT / "data" / "twitch" / "welcomed.db"
        except Exception:                                        # noqa: BLE001
            return frozenset()
    names: set[str] = set()
    try:
        if os.path.isfile(str(path)):
            # read-only + short timeout: never block a spoken turn on a lock.
            con = sqlite3.connect(
                f"file:{path}?mode=ro", uri=True, timeout=0.5)
            try:
                for (login,) in con.execute("SELECT login FROM welcomed"):
                    if login:
                        names.add(str(login).strip().lower())
            finally:
                con.close()
    except Exception as e:                                       # noqa: BLE001
        logger.debug("known_chatters read failed (%s) -- treating as empty", e)
        names = set()
    result = frozenset(names)
    _cache = (now, result)
    return result


def resolve_chatter(
    spoken: str,
    *,
    chatters: Iterable[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> str | None:
    """Resolve a spoken name to a chatter login, or None.

    Returns None whenever the answer is not clear-cut -- too short, an agent
    name, no candidate, or MORE THAN ONE candidate. The caller should treat
    None as "not a chat greeting" and leave the turn on its existing path.
    """
    if not spoken:
        return None
    name = spoken.strip().lower().strip(".,!?;:'\"")
    if len(name) < _MIN_LEN:
        return None
    # An agent name is a teammate, never a viewer -- reject before any lookup
    # so "say hi to Sage" can never become a chat message.
    if name in _agent_names():
        return None
    if exclude and name in {e.strip().lower() for e in exclude}:
        return None

    pool = (frozenset(c.strip().lower() for c in chatters if c)
            if chatters is not None else known_chatters())
    if not pool:
        return None

    if name in pool:
        return name

    # Spoken names are usually a PREFIX of the login ("Izumi" -> izumiikiryo).
    prefix = [c for c in pool if c.startswith(name)]
    if len(prefix) == 1:
        return prefix[0]
    if len(prefix) > 1:
        return None            # ambiguous -- refuse rather than guess

    # Fall back to a high fuzzy ratio for STT spelling drift.
    try:
        from rapidfuzz import fuzz
    except Exception:                                            # noqa: BLE001
        return None
    scored = [(c, fuzz.ratio(name, c)) for c in pool]
    scored = [(c, sc) for c, sc in scored if sc >= _FUZZY_FLOOR]
    if len(scored) != 1:
        return None            # zero, or ambiguous -- refuse
    return scored[0][0]
