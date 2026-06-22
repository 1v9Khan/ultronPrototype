"""Provenance taint for STRUCTURAL team-mic isolation (BR-P1-adjacent).

The single catastrophic-and-silent risk of the Twitch capability is a path by
which viewer/abliterated-model content reaches the TEAM voice channel (the mic
bus teammates hear in a ranked Valorant game). The board's verdict: make team
isolation a TESTED CODE-CAPABILITY BOUNDARY, not prose.

Every utterance carries a :class:`Provenance` tag from its origin. ONLY
``LOCAL_VOICE`` (the streamer's own microphone) is eligible to key the team mic /
drive the relay/PTT path. Chat replies, channel-point redeems, and system audio
are physically refused at the relay boundary by :func:`assert_team_eligible`.

This lives in ``kenning.audio`` (not ``kenning.twitch``) so the import-pinned
voice/relay process can enforce it WITHOUT importing the Twitch package — the
flags-OFF "no kenning.twitch imported" invariant is preserved. Pure stdlib.
"""
from __future__ import annotations

from enum import Enum

__all__ = [
    "Provenance", "TeamIsolationViolation", "TEAM_ELIGIBLE",
    "is_team_eligible", "assert_team_eligible", "relay_allowed",
]


class Provenance(str, Enum):
    """Where an utterance came from. The ONLY team-mic-eligible source is
    ``LOCAL_VOICE`` — the streamer speaking into their own microphone."""
    LOCAL_VOICE = "local_voice"
    TWITCH_CHAT = "twitch_chat"
    REDEEM = "redeem"
    SYSTEM = "system"


#: The allowlist of provenances permitted to reach the team mic / relay / PTT.
TEAM_ELIGIBLE: frozenset[Provenance] = frozenset({Provenance.LOCAL_VOICE})


class TeamIsolationViolation(RuntimeError):
    """Raised when non-LOCAL_VOICE content attempts to reach the team path.

    This is a HARD failure, not a fail-open: a chat string reaching the relay is
    a competitive-integrity catastrophe, so the relay boundary raises rather than
    silently routing. Callers treat it as 'drop the relay attempt'."""


def _coerce(p: "Provenance | str") -> Provenance:
    if isinstance(p, Provenance):
        return p
    try:
        return Provenance(str(p))
    except ValueError:
        # Unknown provenance is treated as NOT local voice (fail-closed).
        return Provenance.SYSTEM


def is_team_eligible(provenance: "Provenance | str") -> bool:
    """True iff this provenance may reach the team mic. Unknown -> False."""
    return _coerce(provenance) in TEAM_ELIGIBLE


def assert_team_eligible(provenance: "Provenance | str", *, where: str = "") -> None:
    """Raise :class:`TeamIsolationViolation` unless the provenance is LOCAL_VOICE.

    The relay/PTT entry point calls this at the TOP, before building any team
    callout, so chat/redeem/system-sourced audio is structurally refused."""
    if not is_team_eligible(provenance):
        raise TeamIsolationViolation(
            f"non-LOCAL_VOICE provenance {_coerce(provenance).value!r} attempted to "
            f"reach the team mic{(' at ' + where) if where else ''}"
        )


def relay_allowed(
    provenance: "Provenance | str",
    *,
    relay_runtime_enabled: bool,
    chat_mode_active: bool,
) -> bool:
    """The full team-relay precondition: LOCAL_VOICE provenance AND the relay is
    armed AND chat-mode is NOT active (chat-mode running means we must not also be
    keying the team mic from any path). Fail-CLOSED on any unmet condition."""
    return (
        is_team_eligible(provenance)
        and bool(relay_runtime_enabled)
        and not bool(chat_mode_active)
    )
