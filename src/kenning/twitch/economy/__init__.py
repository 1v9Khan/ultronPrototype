"""Ultron Twitch economy — append-only ledger + provably-fair RNG + games.

SLICE 9 (DESIGN / MASTER.md). Games / commands / economy are a DETERMINISTIC
trust class: a fully-local second currency whose authority is a single
append-only, event-sourced SQLite WAL ledger (``synchronous=FULL`` on money
commits), with an idempotency key per mutation so the same Twitch event replayed
never double-applies, and balances that are a rebuildable projection of the
event log.

Provably-fair RNG is commit-reveal HMAC-SHA256: ``new_round`` publishes
``sha256(server_seed)`` BEFORE the round; the outcome of every game is decided
SERVER-SIDE (``outcome`` / ``weighted_choice``) from
``HMAC-SHA256(server_seed, f"{client_seed}:{nonce}")`` BEFORE any overlay
animation. The OBS overlay is a dumb renderer that animates to a server-supplied
``target_angle``; ``!verify`` re-derives every outcome from the revealed seed.

ANTICHEAT POSTURE (BR-P1). Pure stdlib only — ``sqlite3`` + ``hashlib`` +
``hmac`` + ``secrets`` + ``threading`` + ``time`` + ``logging``. No network, no
third-party deps, no desktop/screen/input libs. Importable in either the
voice-process (behind the master flag) or a sidecar; the heavy attack surface
(EventSub, Helix outbox) lives in the sidecar and is NOT imported here.

The ``lose ALL points`` wheel consequence is AT-4-class: it is OFF unless the
caller explicitly opts in per :class:`~kenning.twitch.economy.games.SpinTheWheel`.
"""
from __future__ import annotations

from kenning.twitch.economy.games import (
    GameResult,
    SegmentResult,
    Slots,
    SlotsResult,
    SpinTheWheel,
    WheelSegment,
)
from kenning.twitch.economy.ledger import (
    InsufficientFunds,
    Ledger,
    LedgerError,
    LedgerEvent,
)
from kenning.twitch.economy.rng import (
    ProvablyFairRNG,
    RngError,
    RoundCommit,
)

__all__ = [
    # ledger
    "Ledger",
    "LedgerEvent",
    "LedgerError",
    "InsufficientFunds",
    # rng
    "ProvablyFairRNG",
    "RoundCommit",
    "RngError",
    # games
    "SpinTheWheel",
    "WheelSegment",
    "SegmentResult",
    "Slots",
    "SlotsResult",
    "GameResult",
]
