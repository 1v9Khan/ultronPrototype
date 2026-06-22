"""S11 — voice-commanded moderation: Helix write client + the ModerationGuard.

This package is the ACTION channel for moderation (ban / timeout / delete-message /
chat-settings). Two cooperating pieces:

  * :mod:`kenning.twitch.moderation.helix` — :class:`HelixClient`, a thin Helix
    write shim over an INJECTED transport callable (so the real network is never
    touched in tests). It is SELF-IDEMPOTENT (a duplicate ban / already-applied
    action resolves to success, never a raise), keyed by ``(action, target_id,
    message_id)``; carries a token-bucket rate governor that stays far below the
    Helix ~800/min bucket and an exponential-backoff retry for HTTP 429 ONLY.
    A non-GET write is NEVER blind-retried on a generic error.

  * :mod:`kenning.twitch.moderation.guard` — :class:`ModerationGuard`, the
    server-authoritative gate in FRONT of the client. It resolves a spoken name
    to a ``user_id`` against the LIVE roster (exact-login first, then RapidFuzz +
    a phonetic key), refuses to auto-pick on ambiguity/homoglyph, authorizes the
    action (refuse self / moderator / broadcaster; trip a mass-action circuit
    breaker at <=N actions / 60s on a monotonic clock), and records every
    attempted/applied action to an append-only audit (reusing
    :class:`kenning.safety.audit.AuditLog` when importable, else a minimal JSONL
    writer).

ANTICHEAT (BR-P1): pure stdlib + urllib + rapidfuzz. No desktop-automation /
screen-capture / input-injection libs; no ``requests``/``aiohttp``/``websockets``;
``urllib`` + stdlib only. Importable in the voice process behind the flag (same
class as the EmbeddingGemma router client). The abliterated 8B is NEVER consulted
on a moderation decision — the guard is purely deterministic.

Threat model + slice spec: docs/twitch_integration/02_board/{MASTER,S_report}.md
(SLICE 11). All Twitch behavior is flag-gated default-OFF (BR-P1).
"""
from __future__ import annotations

from kenning.twitch.moderation.guard import (
    AuditWriter,
    JsonlAuditWriter,
    ModerationGuard,
    ResolveResult,
    RosterEntry,
)
from kenning.twitch.moderation.helix import (
    HelixClient,
    HelixError,
    HelixResult,
    RateGovernor,
    TokenProvider,
    Transport,
    TransportResponse,
)

__all__ = [
    # helix
    "HelixClient",
    "HelixError",
    "HelixResult",
    "RateGovernor",
    "TokenProvider",
    "Transport",
    "TransportResponse",
    # guard
    "AuditWriter",
    "JsonlAuditWriter",
    "ModerationGuard",
    "ResolveResult",
    "RosterEntry",
]
