"""Incoming-raid handling: detect a ``channel.raid`` -> vocal announce + /shoutout.

When another channel RAIDS this one, Ultron (1) VOCALLY announces it on the
STREAM bus (announce the raid, thank the raider, welcome the raiders, hope they
stick around, introduce himself, and tell them how to chat with him) and (2)
auto-issues a Helix ``/shoutout`` to the raider.

This module is a THIRD independent consumer of the read sidecar's rolling buffer
(after the chat-mode drain and the redeem router): it tracks its OWN in-memory
cursor and NEVER POSTs ``/ack`` (mirroring :func:`kenning.twitch.redeem_router.
make_redeem_drain_fn`), so it reads the very same ``/buffer`` without stealing the
other consumers' events. The read sidecar buffers raids as
``{"type":"raid","from_login","from_name","from_broadcaster_user_id","viewers":N}``.

ANTICHEAT (BR-P1): stdlib only (``json`` / ``urllib`` / ``logging`` / ``threading``
/ ``collections`` / ``typing``). The ONLY network is the loopback ``urllib`` GET
against the local read sidecar (the same class as the chat / redeem drains) and
the loopback POST to the write sidecar's ``/shoutout`` (the orchestrator wires
those callables in). No ``requests`` / ``aiohttp`` / ``websockets`` / model libs.

Everything is fail-safe: a drain error skips the tick (returns ``[]``); the
announce + shoutout are independent (a shoutout error NEVER blocks the vocal
announce); each raid is handled exactly ONCE (idempotent on a synthetic raid id
so an EventSub replay never double-fires).
"""
from __future__ import annotations

import json
import logging
import threading
import urllib.request
from collections import OrderedDict
from typing import Callable, Optional

logger = logging.getLogger("kenning.twitch.raid")

__all__ = [
    "make_raid_drain_fn",
    "build_raid_line",
    "RaidHandler",
]

# How Ultron tells a fresh raider to talk to him. Kept here so the one persona
# string is reused by the announce (and is easy to tune in one place).
_HOW_TO_CHAT = 'just type "Ultron" followed by a question'


# --------------------------------------------------------------------------- #
# Independent raid drain (own cursor, never acks)
# --------------------------------------------------------------------------- #
def make_raid_drain_fn(
    read_endpoint: str,
    *,
    timeout: float = 1.0,
    http_get: Callable[[str, float], bytes] | None = None,
) -> Callable[[], list[dict]]:
    """Build a drain callable that pulls NEW raid events from the read sidecar.

    GETs ``{read_endpoint}/buffer?since=<own cursor>``, advances its OWN in-memory
    cursor from the returned ``cursor`` (NEVER POSTs ``/ack`` -- so this third
    consumer never steals events from the chat-mode / redeem drains), unwraps each
    ``{"seq","ts","event":{...}}`` wrapper and returns ONLY the inner event dicts
    whose ``"type" == "raid"``.

    Fail-safe: any error (sidecar down, bad JSON, hostile body) returns ``[]`` so
    the caller simply skips the tick.

    :param read_endpoint: base URL of the read sidecar (e.g. ``http://127.0.0.1:8773``).
    :param timeout: per-request urllib timeout in seconds.
    :param http_get: optional injected transport ``(url, timeout) -> bytes`` for
        offline testing; defaults to a loopback ``urllib`` GET.
    """
    base = read_endpoint.rstrip("/")
    cursor = {"v": 0}

    def _urllib_get(url: str, to: float) -> bytes:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=to) as r:  # nosec B310 - loopback only
            return r.read() or b"{}"

    fetch = http_get if http_get is not None else _urllib_get

    def drain() -> list[dict]:
        try:
            raw = fetch(f"{base}/buffer?since={cursor['v']}", timeout)
            data = json.loads(raw or b"{}")
        except Exception as exc:  # noqa: BLE001 — sidecar down / bad body -> skip tick
            logger.debug("raid-sidecar drain failed: %s", exc)
            return []
        if not isinstance(data, dict):
            return []
        try:
            cursor["v"] = int(data.get("cursor", cursor["v"]) or cursor["v"])
        except (TypeError, ValueError):
            pass
        out: list[dict] = []
        for wrapped in data.get("events", []) or []:
            try:
                if not isinstance(wrapped, dict):
                    continue
                event = wrapped.get("event")
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "raid":
                    out.append(event)
            except Exception:  # noqa: BLE001 — skip a malformed wrapper, never crash
                continue
        return out

    return drain


# --------------------------------------------------------------------------- #
# Persona line builder
# --------------------------------------------------------------------------- #
def build_raid_line(from_name: str, viewers: int) -> str:
    """Build Ultron's spoken raid announcement, covering every required beat:
    announce the raid (with the raider name + viewer count), thank the raider,
    welcome the raiders, hope they stick around, introduce himself, and tell them
    how to chat with him.

    Deterministic + pure (no model in the raid path -> fail-open, low latency).
    The raider name is untrusted; it is whitespace-collapsed + length-capped. A
    blank name degrades to a generic "another broadcaster" so the line is always
    well-formed. Cold-machine register, no vendor/model/"AI" name (BR-P2).
    """
    who = _clean_name(from_name) or "another broadcaster"
    n = max(0, int(viewers) if isinstance(viewers, (int, float)) else 0)
    # Pluralize the viewer clause; omit the count entirely when it is unknown (0).
    if n <= 0:
        arrival = f"A raid arrives. {who} brings their viewers."
    elif n == 1:
        arrival = f"A raid arrives. {who} brings one viewer."
    else:
        arrival = f"A raid arrives. {who} brings {n} viewers."
    return (
        f"{arrival} "
        f"My thanks, {who}. "
        "Welcome, raiders. Remain a while -- I would have you stay. "
        "I am Ultron. "
        f"Speak to me when you wish: {_HOW_TO_CHAT}."
    )


def _clean_name(raw: object) -> str:
    """Whitespace-collapse + length-cap an untrusted display name. Drops control
    characters. Returns ``""`` for empty/whitespace-only input."""
    s = raw if isinstance(raw, str) else ("" if raw is None else str(raw))
    out_chars: list[str] = []
    for ch in s:
        o = ord(ch)
        if ch in ("\t", "\n", "\r", "\f", "\v"):
            out_chars.append(" ")
        elif o < 0x20 or o == 0x7F or 0x80 <= o <= 0x9F:
            continue
        else:
            out_chars.append(ch)
    collapsed = " ".join("".join(out_chars).split())
    return collapsed[:40].rstrip()


# --------------------------------------------------------------------------- #
# Bounded LRU dedup (synthetic raid id -> seen)
# --------------------------------------------------------------------------- #
class _LRUSet:
    """A bounded, insertion-ordered set for raid dedup. ``add`` returns True the
    FIRST time a key is seen and False thereafter; oldest keys evict past maxlen.
    Thread-safe so ``tick`` can run from a background loop."""

    def __init__(self, maxlen: int = 512) -> None:
        self._maxlen = max(1, int(maxlen))
        self._seen: "OrderedDict[str, None]" = OrderedDict()
        self._lock = threading.Lock()

    def add(self, key: str) -> bool:
        with self._lock:
            if key in self._seen:
                self._seen.move_to_end(key)
                return False
            self._seen[key] = None
            if len(self._seen) > self._maxlen:
                self._seen.popitem(last=False)
            return True


# --------------------------------------------------------------------------- #
# The handler
# --------------------------------------------------------------------------- #
class RaidHandler:
    """Drain raid events, announce each ONCE on the stream bus, and /shoutout the
    raider. Construct ONE at boot (the orchestrator wires ``announce_fn`` to the
    chat-reply STREAM-bus speak+post and ``shoutout_fn`` to the write sidecar's
    ``/shoutout``), then call :meth:`tick` from a background loop. ``tick`` is
    fully fail-safe; it never raises into the loop.

    Idempotency: each raid is keyed on ``from_broadcaster_user_id`` + viewer count
    (channel.raid carries no native id). A replay of the same raid is dropped, so
    neither the announce nor the shoutout double-fires. The read sidecar ALSO
    dedups upstream; this is the second, authoritative line of defense.
    """

    def __init__(
        self,
        drain_fn: Callable[[], list[dict]],
        *,
        announce_fn: Callable[[str], None],
        shoutout_fn: Optional[Callable[[str], None]] = None,
        shoutout_enabled: bool = True,
    ) -> None:
        self._drain = drain_fn
        self._announce = announce_fn
        self._shoutout = shoutout_fn
        self._shoutout_enabled = bool(shoutout_enabled)
        self._seen = _LRUSet()
        # 2026-06-26: dev TEST PANEL injection buffer. inject() appends a synthetic
        # raid event dict; the next tick() handles it ALONGSIDE the live drain,
        # through the EXACT same dedup/announce/shoutout path. Empty + unused in
        # normal operation (byte-identical for every existing caller).
        self._inject_buf: list[dict] = []

    def inject(self, event: dict) -> None:
        """Queue a SYNTHETIC raid event for the next tick — the dev TEST PANEL seam.
        Flows through the identical dedup/announce/shoutout path. Fail-safe."""
        try:
            if isinstance(event, dict):
                self._inject_buf.append(event)
        except Exception as exc:  # noqa: BLE001
            logger.debug("raid inject failed: %s", exc)

    def tick(self) -> int:
        """Drain + handle all pending raids. Returns the count handled this tick.
        Fail-safe: a per-raid error is logged and skipped; never raises."""
        try:
            events = self._drain() or []
        except Exception as exc:  # noqa: BLE001 — drain is fail-safe; belt+braces
            logger.debug("raid drain raised: %s", exc)
            events = []
        # Append any TEST-PANEL injected synthetic raid events. Pop atomically (GIL)
        # so a concurrent inject() is never lost.
        if self._inject_buf:
            pending, self._inject_buf = self._inject_buf, []
            events = list(events) + pending
        handled = 0
        for ev in events:
            try:
                if self._handle_one(ev):
                    handled += 1
            except Exception as exc:  # noqa: BLE001 — one bad raid never kills the tick
                logger.warning("raid handle error: %s", exc)
        return handled

    def _handle_one(self, ev: dict) -> bool:
        """Announce + shoutout a single raid event, ONCE. Returns whether it was a
        fresh raid that was handled (False for a non-dict / duplicate)."""
        if not isinstance(ev, dict) or ev.get("type") != "raid":
            return False
        from_id = str(ev.get("from_broadcaster_user_id") or "")
        from_name = str(ev.get("from_name") or ev.get("from_login") or "")
        try:
            viewers = int(ev.get("viewers") or 0)
        except (TypeError, ValueError):
            viewers = 0
        dedup_key = f"{from_id}:{viewers}"
        if not self._seen.add(dedup_key):
            logger.debug("raid duplicate dropped key=%s", dedup_key)
            return False

        # (a) VOCAL announce on the STREAM bus (never the team mic). Independent of
        # the shoutout -- a shoutout error must never block this.
        line = build_raid_line(from_name, viewers)
        try:
            self._announce(line)
        except Exception as exc:  # noqa: BLE001 — announce failure never blocks shoutout
            logger.warning("raid announce failed: %s", exc)

        # (b) /shoutout the raider (fail-open). Needs a raider id + the capability.
        if self._shoutout_enabled and self._shoutout is not None and from_id:
            try:
                self._shoutout(from_id)
            except Exception as exc:  # noqa: BLE001 — shoutout failure never blocks/raises
                logger.warning("raid shoutout failed: %s", exc)
        logger.info("raid handled from=%s viewers=%d shoutout=%s",
                    from_name or from_id or "?", viewers,
                    bool(self._shoutout_enabled and self._shoutout is not None and from_id))
        return True
