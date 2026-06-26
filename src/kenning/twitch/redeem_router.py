"""S12 — channel-point REDEEM ROUTER (redeem -> game -> announce + overlay).

The read sidecar (``scripts/twitch_read_sidecar.py``) subscribes to channel-point
redemptions and buffers them as ``{"type":"redeem",...}`` events alongside the
``{"type":"chat",...}`` chat events, each wrapped as ``{"seq","ts","event":{...}}``
in the rolling buffer (drained over loopback via ``GET /buffer?since=N``). This
module is the SECOND, independent consumer of that same buffer: it tracks its OWN
in-memory cursor and NEVER POSTs ``/ack`` (mirroring
``kenning.twitch.service.make_read_drain_fn``), so it can read the very same buffer
the chat-mode drain reads without either consumer stealing the other's events.

When a redeemed reward's title maps to a game (spin the wheel / slots / heist /
duel / trivia / raffle) the router RUNS that game from a freshly-minted
provably-fair round and surfaces the outcome two ways:

  * ``announce_fn(line)``     -- a short in-character spoken line (the orchestrator
                                 passes its Kokoro TTS speak), and
  * ``overlay_emit(event)``   -- a JSON-serializable overlay event the dumb overlay
                                 renderer shows.

SPEAK REDEEMS (2026-06-26): two NON-game reward titles let a VIEWER make Ultron
SPEAK their own typed message via TTS. The viewer text is UNTRUSTED, so every
speak is gated by the SAME Llama-Guard sidecar that gates chat-reply (injected
``guard_classify_fn``; FAIL-CLOSED — a guard error / unreachable sidecar BLOCKS
the speak), then control-char-stripped + length-capped + framed (prefixed with
the viewer name) before TTS:

  * the ``say`` title  -> ``say_speak_fn(framed)``  (streamer speakers + the
    stream/broadcast bus + chat post; NEVER the team mic), and
  * the ``team`` title -> ``team_speak_fn(framed)`` (the team voice bus). Wired
    only when the streamer opted the team redeem in (the chat->team boundary), so
    a missing ``team_speak_fn`` simply blocks that title.

Each speak redeem is dedup-idempotent on the redemption id (the same LRU dedup
the games use), so an EventSub replay never re-speaks.

ANTICHEAT (BR-P1): stdlib only (``json`` / ``urllib`` / ``logging`` / ``threading``
/ ``collections`` / ``dataclasses`` / ``typing``) + ``kenning.twitch.economy.*``.
The ONLY network is :func:`make_redeem_drain_fn`'s loopback ``urllib`` GET against
the local read sidecar -- the same class as the existing chat drain. No
``requests`` / ``aiohttp`` / ``websockets`` / model libs ever load here.

Everything is fail-safe: a drain error skips the tick (returns ``[]``); a single
bad redeem is logged and skipped without breaking the rest of the tick; a game
that raises is caught and that one redeem is dropped. Outcomes are deterministic
for a given injected RNG.
"""
from __future__ import annotations

import json
import logging
import threading
import urllib.request
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from kenning.twitch.economy.games import (
    Duel,
    Heist,
    Raffle,
    Slots,
    SpinTheWheel,
    Trivia,
    WheelSegment,
)
from kenning.twitch.economy.ledger import Ledger
from kenning.twitch.economy.rng import ProvablyFairRNG

logger = logging.getLogger("kenning.twitch.redeem_router")

__all__ = [
    "make_redeem_drain_fn",
    "DEFAULT_REWARD_MAP",
    "REDEEM_SLOTS_WIN",
    "REDEEM_DUEL_WAGER",
    "REDEEM_HEIST_POT",
    "SPEAK_SAY",
    "SPEAK_TEAM",
    "sanitize_speak_text",
    "frame_speak_line",
    "say_name_enabled",
    "set_say_name_enabled",
    "RedeemRouter",
]

# Speak-redeem action keys (NON-game; map a reward title to one of these).
SPEAK_SAY = "speak_say"     # speak on the streamer/stream-broadcast bus + chat
SPEAK_TEAM = "speak_team"   # speak onto the team voice bus

# --------------------------------------------------------------------------- #
# SAY-NAME runtime toggle (2026-06-26) — governs whether the TEAM speak redeem
# announces the viewer name as a prefix ("<viewer> says: <msg>") or speaks ONLY
# the message. Default ON (name announced). Flipped live by the stop-window
# SAY-NAME toggle via ``set_say_name_enabled`` (the orchestrator wires the
# stop_button callback to it), exactly like the CHAT / HEAR-CHAT runtime toggles.
# Module-level so the framing in ``frame_speak_line`` (which runs in the redeem
# tick) sees the current state without a config reload. Anticheat-clean (a plain
# bool; no new imports). Only the TEAM variant consults it — the SAY redeem keeps
# its "<viewer> says:" framing so the stream always knows who spoke.
_SAY_NAME_ENABLED = True


def say_name_enabled() -> bool:
    """True iff the TEAM speak redeem should prefix the viewer name. Default ON."""
    return bool(_SAY_NAME_ENABLED)


def set_say_name_enabled(on: bool) -> None:
    """Flip the SAY-NAME runtime toggle (stop-window button / future voice cmd)."""
    global _SAY_NAME_ENABLED
    _SAY_NAME_ENABLED = bool(on)
    logger.info("redeem team-speak SAY-NAME -> %s", "ON" if _SAY_NAME_ENABLED else "OFF")

# Hard ceiling on a viewer's spoken text regardless of the configured cap (the
# config cap is clamped to this; defends against a misconfigured huge max_chars).
_SPEAK_HARD_MAX = 500

# House-funded payout amounts for the SINGLE-redeem games. A channel-point redeem
# already cost the viewer Twitch's native points, so the economy game pays out
# (credit only -- never a ledger debit). Keyed on the redemption id so an EventSub
# replay never double-pays.
REDEEM_SLOTS_WIN = 500     # paid on a redeem-slots triple
REDEEM_HEIST_POT = 100     # the lone-redeemer heist pot (WIN pays it back, PARTIAL half)
REDEEM_DUEL_WAGER = 100    # the duel-vs-house wager (paid on a redeemer win)


# --------------------------------------------------------------------------- #
# Reward-title -> game action map
# --------------------------------------------------------------------------- #
# Lowercased reward titles -> a game action key. The router lowercases + strips
# the incoming reward title before lookup, so the keys here are all lowercase.
DEFAULT_REWARD_MAP: dict[str, str] = {
    "spin the wheel": "wheel",
    "spin": "wheel",
    "wheel": "wheel",
    "slots": "slots",
    "slot machine": "slots",
    "heist": "heist",
    "duel": "duel",
    "trivia": "trivia",
    "raffle": "raffle",
}


# --------------------------------------------------------------------------- #
# Independent redeem drain (own cursor, never acks)
# --------------------------------------------------------------------------- #
def make_redeem_drain_fn(
    read_endpoint: str,
    *,
    timeout: float = 1.0,
    http_get: Callable[[str, float], bytes] | None = None,
) -> Callable[[], list[dict]]:
    """Build a drain callable that pulls NEW redeem events from the read sidecar.

    GETs ``{read_endpoint}/buffer?since=<own cursor>``, advances its OWN in-memory
    cursor from the returned ``cursor`` (NEVER POSTs ``/ack`` -- so this second
    consumer never steals events from the chat-mode drain), unwraps each
    ``{"seq","ts","event":{...}}`` wrapper and returns ONLY the inner event dicts
    whose ``"type" == "redeem"``.

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
            logger.debug("redeem-sidecar drain failed: %s", exc)
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
                if event.get("type") == "redeem":
                    out.append(event)
            except Exception:  # noqa: BLE001 — skip a malformed wrapper, never crash
                continue
        return out

    return drain


# --------------------------------------------------------------------------- #
# Default game-segment libraries
# --------------------------------------------------------------------------- #
def _default_wheel_segments() -> list[WheelSegment]:
    """Six fun, all-positive wheel segments (no LOSE_ALL -> safe by default)."""
    return [
        WheelSegment("DOUBLE", weight=1.0, payout=200),
        WheelSegment("TRIPLE", weight=0.5, payout=300),
        WheelSegment("NOTHING", weight=2.0, payout=0),
        WheelSegment("SMALL WIN", weight=2.0, payout=50),
        WheelSegment("JACKPOT", weight=0.25, payout=1000),
        WheelSegment("REFUND", weight=1.5, payout=100),
    ]


_DEFAULT_SLOT_SYMBOLS = ("cherry", "lemon", "bell", "star", "seven", "skull")


# --------------------------------------------------------------------------- #
# Bounded LRU dedup set (redemption_id -> seen)
# --------------------------------------------------------------------------- #
class _LRUSet:
    """A bounded, insertion-ordered set used for redemption_id dedup.

    ``add`` returns True the FIRST time an id is seen and False thereafter; the
    oldest ids are evicted once ``maxlen`` is exceeded. Thread-safe so ``tick``
    can be called from a background loop while another caller introspects."""

    def __init__(self, maxlen: int) -> None:
        self._maxlen = max(1, int(maxlen))
        self._seen: OrderedDict[str, None] = OrderedDict()
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

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._seen

    def __len__(self) -> int:
        with self._lock:
            return len(self._seen)


# --------------------------------------------------------------------------- #
# Speak-redeem text hygiene (UNTRUSTED viewer input)
# --------------------------------------------------------------------------- #
def sanitize_speak_text(raw: object, *, max_chars: int) -> str:
    """Sanitize an UNTRUSTED viewer message for TTS: drop control characters,
    collapse all whitespace runs to single spaces, strip the ends, and cap the
    length to ``min(max_chars, _SPEAK_HARD_MAX)`` (trimmed back to a word
    boundary when possible so a cut never splits a word mid-token).

    Pure + stdlib-free (no ``re``): keeps the module's import surface within the
    anticheat-pinned allowlist. Returns ``""`` for empty/whitespace-only input.
    """
    s = raw if isinstance(raw, str) else ("" if raw is None else str(raw))
    # Drop control characters (C0 + DEL + C1) but turn tab/newline into spaces so
    # words don't fuse; everything else printable is kept (incl. unicode letters).
    out_chars: list[str] = []
    for ch in s:
        o = ord(ch)
        if ch in ("\t", "\n", "\r", "\f", "\v"):
            out_chars.append(" ")
        elif o < 0x20 or o == 0x7F or 0x80 <= o <= 0x9F:
            continue  # control char -> drop
        else:
            out_chars.append(ch)
    # Collapse whitespace runs -> single spaces, strip ends.
    collapsed = " ".join("".join(out_chars).split())
    if not collapsed:
        return ""
    cap = max(1, min(int(max_chars), _SPEAK_HARD_MAX))
    if len(collapsed) <= cap:
        return collapsed
    head = collapsed[:cap]
    # Trim back to the last space so we don't cut a word in half (unless the very
    # first token is already longer than the cap).
    sp = head.rfind(" ")
    if sp >= cap // 2:
        head = head[:sp]
    return head.rstrip()


def frame_speak_line(viewer: str, text: str, *, to_team: bool = False) -> str:
    """Frame a SAFE, sanitized viewer message in Ultron's cold register so the
    listener knows it came from chat. The viewer name is itself sanitized (it is
    also untrusted) and prefixed. Returns ``""`` if there is nothing safe to say.

    TEAM variant (2026-06-26): the leading "Relaying from chat." was dropped (the
    team callout is tighter without it). The "<viewer> says:" name prefix is
    governed by the SAY-NAME runtime toggle (:func:`say_name_enabled`, default
    ON); when OFF the team line is the bare message. The SAY (broadcast) variant
    ALWAYS keeps the name prefix so the stream knows who spoke."""
    body = (text or "").strip()
    if not body:
        return ""
    who = sanitize_speak_text(viewer, max_chars=40).strip() or "A viewer"
    if to_team:
        if not say_name_enabled():
            return body
        return f"{who} says: {body}"
    return f"{who} says: {body}"


# --------------------------------------------------------------------------- #
# The router
# --------------------------------------------------------------------------- #
class RedeemRouter:
    """Drain redeem events, run the mapped game, announce + emit the outcome.

    Construct ONE at boot (the orchestrator wires ``announce_fn`` to its TTS speak
    and ``overlay_emit`` to the overlay sidecar publish), then call :meth:`tick`
    from the idle/background loop alongside the chat-mode tick. ``tick`` is fully
    fail-safe; it never raises into the loop.
    """

    def __init__(
        self,
        drain_fn: Callable[[], list[dict]],
        *,
        rng: ProvablyFairRNG | None = None,
        reward_map: dict[str, str] | None = None,
        announce_fn: Callable[[str], Any] | None = None,
        overlay_emit: Callable[[dict], Any] | None = None,
        games: dict[str, Any] | None = None,
        ledger: Ledger | None = None,
        dedup_max: int = 2048,
        speak_reward_map: dict[str, str] | None = None,
        guard_classify_fn: Callable[[str], Any] | None = None,
        say_speak_fn: Callable[[str], Any] | None = None,
        team_speak_fn: Callable[[str], Any] | None = None,
        blocked_chat_fn: Callable[[str], Any] | None = None,
        speak_max_chars: int = 200,
    ) -> None:
        self._drain = drain_fn
        # 2026-06-26: dev TEST PANEL injection buffer. inject() appends a synthetic
        # redeem event dict; the next tick() processes it ALONGSIDE the live drain,
        # through the EXACT same dedup/dispatch path. Empty + unused in normal
        # operation (byte-identical for every existing caller).
        self._inject_buf: list[dict] = []
        # --- SPEAK-redeem wiring (UNTRUSTED viewer text -> guard -> TTS) ------ #
        # ``speak_reward_map``: lowercased reward title -> SPEAK_SAY / SPEAK_TEAM.
        # ``guard_classify_fn(text) -> result`` (result.unsafe truthy => blocked);
        # FAIL-CLOSED: any exception OR a missing guard BLOCKS the speak. The say
        # path speaks to the streamer/broadcast bus + chat; the team path speaks
        # onto the team voice bus (wired only when the streamer opted it in -- a
        # missing ``team_speak_fn`` blocks the team title). ``blocked_chat_fn`` is
        # an optional short chat note when a message is refused as unsafe.
        smap = speak_reward_map or {}
        self._speak_map = {str(k).strip().lower(): str(v) for k, v in smap.items()}
        self._guard_classify = guard_classify_fn
        self._say_speak = say_speak_fn
        self._team_speak = team_speak_fn
        self._blocked_chat = blocked_chat_fn
        self._speak_max_chars = max(1, int(speak_max_chars))
        # Optional ledger: when present, a redeem game's outcome credits the
        # redeemer's balance (house-funded, keyed on the redemption id). None ->
        # the games still run + announce + overlay, just without a currency move
        # (byte-identical to the pre-ledger router for every existing caller).
        self._ledger = ledger
        self._rng = rng if rng is not None else ProvablyFairRNG()
        # Lowercase the reward map keys defensively (callers may pass mixed case).
        rmap = reward_map if reward_map is not None else DEFAULT_REWARD_MAP
        self._reward_map = {str(k).strip().lower(): str(v) for k, v in rmap.items()}
        self._announce = announce_fn
        self._overlay = overlay_emit
        self._games: dict[str, Any] = dict(games) if games else {}
        self._dedup = _LRUSet(dedup_max)
        # Per-action nonce so every redeem of the same game advances the round
        # (still deterministic for a fixed rng + sequence).
        self._nonce: dict[str, int] = {}

    # -- game accessors (lazy defaults) ---------------------------------- #
    def _next_nonce(self, action: str) -> int:
        n = self._nonce.get(action, 0)
        self._nonce[action] = n + 1
        return n

    def _wheel(self) -> SpinTheWheel:
        g = self._games.get("wheel")
        if g is None:
            g = SpinTheWheel(_default_wheel_segments(), rng=self._rng)
            self._games["wheel"] = g
        return g

    def _slots(self) -> Slots:
        g = self._games.get("slots")
        if g is None:
            g = Slots(_DEFAULT_SLOT_SYMBOLS, reels=3, rng=self._rng)
            self._games["slots"] = g
        return g

    def _heist(self) -> Heist:
        g = self._games.get("heist")
        if g is None:
            g = Heist(rng=self._rng)
            self._games["heist"] = g
        return g

    def _duel(self) -> Duel:
        g = self._games.get("duel")
        if g is None:
            g = Duel(rng=self._rng)
            self._games["duel"] = g
        return g

    def _trivia(self) -> Trivia:
        g = self._games.get("trivia")
        if g is None:
            g = Trivia(rng=self._rng)
            self._games["trivia"] = g
        return g

    def _raffle(self) -> Raffle:
        g = self._games.get("raffle")
        if g is None:
            g = Raffle(rng=self._rng)
            self._games["raffle"] = g
        return g

    # -- public tick ----------------------------------------------------- #
    def inject(self, event: dict) -> None:
        """Queue a SYNTHETIC redeem event for the next tick — the dev TEST PANEL
        seam. The event flows through the identical dedup/dispatch path as a live
        redemption. Fail-safe."""
        try:
            if isinstance(event, dict):
                self._inject_buf.append(event)
        except Exception as exc:  # noqa: BLE001
            logger.debug("redeem inject failed: %s", exc)

    def tick(self) -> list[dict]:
        """Drain + process every NEW redeem this cycle. Returns the list of
        outcome dicts processed (for tests + logging). Fail-safe end-to-end."""
        try:
            events = self._drain() or []
        except Exception as exc:  # noqa: BLE001 — drain must never crash the loop
            logger.warning("redeem drain raised: %s", exc)
            events = []
        # Append any TEST-PANEL injected synthetic redeem events (processed through
        # the same per-redeem path). Pop atomically (GIL) so a concurrent inject()
        # is never lost.
        if self._inject_buf:
            pending, self._inject_buf = self._inject_buf, []
            events = list(events) + pending
        if not events:
            return []
        outcomes: list[dict] = []
        for ev in events:
            try:
                outcome = self._process_one(ev)
            except Exception as exc:  # noqa: BLE001 — one bad redeem never breaks the tick
                logger.warning(
                    "redeem processing failed redemption_id=%r: %s",
                    (ev or {}).get("redemption_id") if isinstance(ev, dict) else None,
                    exc,
                )
                continue
            if outcome is not None:
                outcomes.append(outcome)
        return outcomes

    # -- per-redeem -------------------------------------------------------- #
    def _process_one(self, ev: dict) -> dict | None:
        if not isinstance(ev, dict):
            return None
        redemption_id = str(ev.get("redemption_id") or "")
        # Dedup on the redemption id (bounded LRU). A redeem with no id is still
        # processed (the sidecar always supplies one for a real redemption; a
        # missing id just bypasses dedup rather than blocking the game).
        if redemption_id and not self._dedup.add(redemption_id):
            logger.debug("redeem dedup skip redemption_id=%s", redemption_id)
            return None

        title = str(ev.get("reward_title") or "")
        viewer = str(ev.get("chatter_login") or ev.get("chatter_name") or "")
        uid = str(ev.get("chatter_user_id") or ev.get("user_id") or "")
        user_input = str(ev.get("user_input") or "")
        # SE-points backend: map uid -> login so a redeem payout can address the
        # StreamElements API (keyed by login). No-op for the SQLite ledger.
        if uid and viewer:
            _reg = getattr(self._ledger, "register", None)
            if _reg is not None:
                try:
                    _reg(uid, viewer)
                except Exception:  # noqa: BLE001 — must never block a redeem
                    pass
        title_key = title.strip().lower()

        # SPEAK redeems (checked BEFORE the game map; dedup above already ran so
        # an EventSub replay never re-speaks). A viewer's typed message is guard-
        # screened, sanitized, framed, then spoken on the say/team bus.
        speak_action = self._speak_map.get(title_key)
        if speak_action is not None:
            return self._process_speak(speak_action, title, viewer, user_input)

        action = self._reward_map.get(title_key)

        if action is None:
            # Not a game reward -> still emit a generic overlay event so the
            # overlay can show the redemption; no game, no spoken line.
            generic = {"type": "redeem", "reward": title, "viewer": viewer}
            self._emit(generic)
            logger.info("redeem (non-game) reward=%r viewer=%r", title, viewer)
            return None

        runner = self._RUNNERS.get(action)
        if runner is None:
            # A mapped action with no runner (a misconfigured map) -> treat as a
            # generic redeem rather than crashing.
            generic = {"type": "redeem", "reward": title, "viewer": viewer}
            self._emit(generic)
            logger.warning("redeem action %r has no runner; emitted generic", action)
            return None

        line, event = runner(self, viewer or "someone", uid, user_input, redemption_id)
        if line:
            self._announce_safe(line)
        self._emit(event)
        logger.info(
            "redeem game=%s viewer=%r outcome=%r", action, viewer, event.get("outcome")
        )
        return event

    # -- speak redeems (UNTRUSTED viewer text -> guard -> TTS) ------------- #
    def _guard_blocks(self, text: str) -> tuple[bool, str]:
        """Run the injected guard over ``text``. Returns ``(blocked, reason)``.
        FAIL-CLOSED: no guard configured OR any guard error -> blocked. A truthy
        ``result.unsafe`` (the GuardResult / GuardModelClient contract) -> blocked.
        """
        if self._guard_classify is None:
            return True, "no guard configured (fail-closed)"
        try:
            result = self._guard_classify(text)
        except Exception as exc:  # noqa: BLE001 — guard unreachable/error -> fail CLOSED
            logger.warning("speak redeem guard error -> BLOCKED: %s", exc)
            return True, f"guard error (fail-closed): {exc}"
        unsafe = bool(getattr(result, "unsafe", result))
        if unsafe:
            cat = str(getattr(result, "category", "") or "")
            return True, f"guard flagged unsafe{(': ' + cat) if cat else ''}"
        return False, "safe"

    def _process_speak(
        self, speak_action: str, title: str, viewer: str, user_input: str,
    ) -> dict | None:
        """Handle a SPEAK redeem: sanitize the UNTRUSTED viewer message, guard-
        screen it (FAIL-CLOSED), frame it, then speak via the say/team callback.
        Returns an outcome dict (for tests + logging) or ``None`` when nothing was
        spoken. Fully fail-safe: never raises into the tick."""
        to_team = speak_action == SPEAK_TEAM
        speak_fn = self._team_speak if to_team else self._say_speak
        bus = "team" if to_team else "say"
        viewer = viewer or "someone"
        text = sanitize_speak_text(user_input, max_chars=self._speak_max_chars)
        if not text:
            logger.info("speak redeem (%s) empty after sanitize viewer=%r", bus, viewer)
            return {"type": "redeem_speak", "bus": bus, "viewer": viewer,
                    "spoken": False, "reason": "empty input"}
        if speak_fn is None:
            # The team title can land here when the streamer hasn't opted the
            # team redeem in (no team callback wired) -> refuse to speak.
            logger.info("speak redeem (%s) no speak callback wired viewer=%r", bus, viewer)
            return {"type": "redeem_speak", "bus": bus, "viewer": viewer,
                    "spoken": False, "reason": "speak path not enabled"}
        blocked, reason = self._guard_blocks(text)
        if blocked:
            logger.info("speak redeem (%s) BLOCKED viewer=%r reason=%s", bus, viewer, reason)
            self._note_blocked(viewer)
            return {"type": "redeem_speak", "bus": bus, "viewer": viewer,
                    "spoken": False, "reason": reason}
        framed = frame_speak_line(viewer, text, to_team=to_team)
        if not framed:
            return {"type": "redeem_speak", "bus": bus, "viewer": viewer,
                    "spoken": False, "reason": "empty after framing"}
        try:
            speak_fn(framed)
        except Exception as exc:  # noqa: BLE001 — a TTS hiccup never breaks the tick
            logger.warning("speak redeem (%s) speak failed: %s", bus, exc)
            return {"type": "redeem_speak", "bus": bus, "viewer": viewer,
                    "spoken": False, "reason": f"speak error: {exc}"}
        logger.info("speak redeem (%s) spoke viewer=%r reward=%r", bus, viewer, title)
        result = {"type": "redeem_speak", "bus": bus, "viewer": viewer,
                  "spoken": True, "text": framed, "reason": "spoken"}
        # Surface a unified bottom-left "speech" card on the overlay (fail-safe via
        # _emit). The framed text is the SAME guard-screened + sanitized string that
        # was spoken; the overlay re-length-checks + renders it as inert text.
        self._emit(result)
        return result

    def _note_blocked(self, viewer: str) -> None:
        """Optional brief chat note that a viewer's message was blocked. Fail-safe;
        never reveals what tripped the guard (anti-probe)."""
        if self._blocked_chat is None:
            return
        try:
            self._blocked_chat(
                f"@{viewer} your Ultron message was held back by the safety filter."
            )
        except Exception as exc:  # noqa: BLE001 — a chat-post hiccup never breaks the tick
            logger.debug("speak redeem blocked-note failed: %s", exc)

    # -- per-game runners ------------------------------------------------- #
    # Each runner returns (spoken_line, overlay_event). The overlay event shape is
    # {"type":"redeem_result","game":<action>,"viewer":<login>,"outcome":<label>,
    #  "detail":{...provably-fair provenance + game specifics...}}.
    def _round(self) -> Any:
        """Mint a fresh provably-fair round (server_seed + commit)."""
        return self._rng.new_round()

    def _run_wheel(self, viewer: str, uid: str, user_input: str, rid: str) -> tuple[str, dict]:
        rnd = self._round()
        nonce = self._next_nonce("wheel")
        res = self._wheel().spin(rnd.server_seed, nonce=nonce)
        label = res.segment.label
        credited = self._credit(uid, res.segment.payout, "wheel", rid)
        line = f"Wheel landed on {label} for {viewer}." + (f" +{credited}." if credited else "")
        event = {
            "type": "redeem_result",
            "game": "wheel",
            "viewer": viewer,
            "outcome": label,
            "detail": {
                "index": res.index,
                "payout": res.segment.payout,
                "credited": credited,
                "target_angle": res.target_angle,
                "commit": rnd.commit,
                "server_seed": rnd.server_seed,
                "nonce": nonce,
            },
        }
        return line, event

    def _run_slots(self, viewer: str, uid: str, user_input: str, rid: str) -> tuple[str, dict]:
        rnd = self._round()
        nonce = self._next_nonce("slots")
        res = self._slots().pull(rnd.server_seed, nonce=nonce)
        reels = " | ".join(res.reels)
        credited = self._credit(uid, REDEEM_SLOTS_WIN if res.is_win else 0, "slots", rid)
        if res.is_win:
            line = f"Slots hit triple {res.win_symbol} for {viewer}. Jackpot +{credited}."
            outcome = f"WIN:{res.win_symbol}"
        else:
            line = f"Slots landed {reels} for {viewer}. No match."
            outcome = "LOSS"
        event = {
            "type": "redeem_result",
            "game": "slots",
            "viewer": viewer,
            "outcome": outcome,
            "detail": {
                "reels": list(res.reels),
                "is_win": res.is_win,
                "win_symbol": res.win_symbol,
                "credited": credited,
                "commit": rnd.commit,
                "server_seed": rnd.server_seed,
                "nonce": nonce,
            },
        }
        return line, event

    def _run_heist(self, viewer: str, uid: str, user_input: str, rid: str) -> tuple[str, dict]:
        rnd = self._round()
        nonce = self._next_nonce("heist")
        # Single-redeem heist: the redeemer is the lone participant with a fixed
        # token pot, resolved immediately (the simplest meaningful behaviour).
        pot = REDEEM_HEIST_POT
        res = self._heist().resolve(rnd.server_seed, [viewer], pot, nonce=nonce)
        credited = self._credit(uid, res.payout_per_head, "heist", rid)
        line = (
            f"Heist {res.outcome} for {viewer}." + (f" +{credited}." if credited else " No payout.")
        )
        event = {
            "type": "redeem_result",
            "game": "heist",
            "viewer": viewer,
            "outcome": res.outcome,
            "detail": {
                "participants": list(res.participants),
                "pot": res.pot,
                "payout_per_head": res.payout_per_head,
                "credited": credited,
                "commit": rnd.commit,
                "server_seed": rnd.server_seed,
                "nonce": nonce,
            },
        }
        return line, event

    def _run_duel(self, viewer: str, uid: str, user_input: str, rid: str) -> tuple[str, dict]:
        rnd = self._round()
        nonce = self._next_nonce("duel")
        # Single-redeem duel: the redeemer challenges "the house". A distinct,
        # non-equal target keeps Duel.resolve happy and is deterministic.
        target = "the_house" if viewer != "the_house" else "the_challenger"
        wager = REDEEM_DUEL_WAGER
        res = self._duel().resolve(
            rnd.server_seed, viewer, target, wager, nonce=nonce
        )
        won = res.winner == viewer
        credited = self._credit(uid, wager if won else 0, "duel", rid)
        line = (
            f"{viewer} won the duel against the house. +{credited}."
            if won
            else f"{viewer} lost the duel to the house."
        )
        event = {
            "type": "redeem_result",
            "game": "duel",
            "viewer": viewer,
            "outcome": "WIN" if won else "LOSS",
            "detail": {
                "winner": res.winner,
                "loser": res.loser,
                "wager": res.wager,
                "credited": credited,
                "challenger": res.challenger,
                "target": res.target,
                "commit": rnd.commit,
                "server_seed": rnd.server_seed,
                "nonce": nonce,
            },
        }
        return line, event

    def _run_trivia(self, viewer: str, uid: str, user_input: str, rid: str) -> tuple[str, dict]:
        rnd = self._round()
        nonce = self._next_nonce("trivia")
        # Single-redeem trivia: draw + announce a question (chat answers it later;
        # the router's job is to surface the prompt deterministically).
        question, idx, prov = self._trivia().draw_question(rnd.server_seed, nonce=nonce)
        line = f"Trivia for {viewer}: {question.question}"
        event = {
            "type": "redeem_result",
            "game": "trivia",
            "viewer": viewer,
            "outcome": question.question,
            "detail": {
                "question": question.question,
                "question_index": idx,
                "commit": prov.commit,
                "server_seed": prov.server_seed,
                "nonce": nonce,
            },
        }
        return line, event

    def _run_raffle(self, viewer: str, uid: str, user_input: str, rid: str) -> tuple[str, dict]:
        rnd = self._round()
        nonce = self._next_nonce("raffle")
        raffle = self._raffle()
        # Single-redeem raffle: open a window if none is live, then enter the
        # viewer. Entry is the meaningful single-redeem behaviour; the streamer
        # draws the winner separately when the window closes.
        if not raffle.is_open:
            raffle.open()
        entered = raffle.enter(viewer)
        outcome = "entered" if entered else "already_entered"
        line = (
            f"{viewer} is in the raffle."
            if entered
            else f"{viewer} is already in the raffle."
        )
        event = {
            "type": "redeem_result",
            "game": "raffle",
            "viewer": viewer,
            "outcome": outcome,
            "detail": {
                "entered": entered,
                "entrants": list(raffle.entrants),
                "commit": rnd.commit,
                "server_seed": rnd.server_seed,
                "nonce": nonce,
            },
        }
        return line, event

    # Action key -> bound runner. Defined once at class scope.
    _RUNNERS: dict[str, Callable[[RedeemRouter, str, str, str, str], tuple[str, dict]]] = {
        "wheel": _run_wheel,
        "slots": _run_slots,
        "heist": _run_heist,
        "duel": _run_duel,
        "trivia": _run_trivia,
        "raffle": _run_raffle,
    }

    # -- ledger (house-funded payout, redemption-id keyed) ----------------- #
    def _credit(self, uid: str, amount: object, game: str, rid: str) -> int:
        """Credit a house-funded payout to the redeemer's ledger balance, keyed on
        the redemption id so an EventSub replay never double-pays. No-op without a
        ledger / uid / positive amount. Returns the amount actually credited."""
        if self._ledger is None or not uid:
            return 0
        try:
            amt = int(amount)
        except (TypeError, ValueError):
            return 0
        if amt <= 0:
            return 0
        key = f"redeem:{game}:{rid}" if rid else f"redeem:{game}:{uid}:{self._next_nonce(game + ':credit')}"
        try:
            self._ledger.credit(uid, amt, f"{game} redeem", key)
            return amt
        except Exception as exc:  # noqa: BLE001 — a credit fault never breaks the tick
            logger.warning("redeem credit failed game=%s uid=%s: %s", game, uid, exc)
            return 0

    # -- sinks (fail-safe) ------------------------------------------------- #
    def _announce_safe(self, line: str) -> None:
        if self._announce is None:
            return
        try:
            self._announce(line)
        except Exception as exc:  # noqa: BLE001 — a TTS hiccup never breaks the tick
            logger.warning("redeem announce failed: %s", exc)

    @staticmethod
    def _to_overlay_event(event: dict) -> dict | None:
        """Translate an internal redeem / redeem_result / redeem_speak event into a
        UNIFIED overlay card so a redeemed game looks IDENTICAL to the same typed
        chat-command game (one visual language; the only difference is a REDEEM tag
        instead of CHAT). Game outcomes -> a ``chat_game`` card (source="redeem");
        a SPEAK redeem -> a ``speech`` card; the generic non-game redeem -> a small
        ``chat_game``-less fallback card is intentionally NOT emitted (no game ->
        nothing to render). Returns None when there is nothing to show (the overlay
        then renders nothing rather than erroring). The router's own event shape is
        kept for the spoken line + the outcomes log."""
        etype = str(event.get("type") or "")
        viewer = str(event.get("viewer") or "someone")

        if etype == "redeem_result":
            return RedeemRouter._game_card(event, viewer)

        if etype == "redeem_speak":
            # Only the speaks that actually SPOKE get a card (a blocked/empty speak
            # shows nothing). Render in the same bottom-left style as a speech card.
            if not event.get("spoken"):
                return None
            return {
                "type": "speech",
                "bus": "team" if event.get("bus") == "team" else "say",
                "viewer": str(viewer)[:80],
                "text": str(event.get("text") or "")[:300],
            }

        # A generic (non-game) redeem carries no game outcome -> nothing to render
        # as a game card. (Previously an 'alert' banner; retired with the old style.)
        return None

    @staticmethod
    def _game_card(event: dict, viewer: str) -> dict | None:
        """Map a redeem GAME result to the unified ``chat_game`` card schema
        (source="redeem"). Mirrors the chat-game router's card so a redeemed Slots
        and a typed ``!slots`` are byte-shape identical to the renderer."""
        game = str(event.get("game") or "game")
        outcome = str(event.get("outcome") or "")
        detail = event.get("detail") or {}
        try:
            credited = int(detail.get("credited", 0) or 0)
        except (TypeError, ValueError):
            credited = 0

        card: dict = {
            "type": "chat_game",
            "game": game,
            "source": "redeem",
            "viewer": str(viewer)[:80],
            "title": game.upper()[:120],
            "outcome": outcome[:120],
            "won": False,
            "amount": credited,
            "detail": {},
        }

        if game == "wheel":
            try:
                payout = int(detail.get("payout", 0) or 0)
            except (TypeError, ValueError):
                payout = 0
            card["outcome"] = outcome[:120]
            card["won"] = payout > 0
            card["amount"] = credited or payout
            card["detail"] = {"segment": outcome[:60], "payout": payout}
        elif game == "slots":
            is_win = bool(detail.get("is_win"))
            reels = detail.get("reels") or []
            card["outcome"] = "WIN" if is_win else "LOSS"
            card["won"] = is_win
            card["amount"] = credited
            card["detail"] = {
                "reels": [str(s)[:40] for s in reels][:8],
                "win_symbol": str(detail.get("win_symbol") or "")[:40] or None,
                "payout": credited,
            }
        elif game == "heist":
            up = outcome.upper()
            won = up in ("WIN", "PARTIAL")
            try:
                pot = int(detail.get("pot", 0) or 0)
            except (TypeError, ValueError):
                pot = 0
            crew = len(detail.get("participants") or []) or 1
            card["outcome"] = up
            card["won"] = won
            card["amount"] = credited
            card["detail"] = {"pot": pot, "crew": crew, "payout": credited}
        elif game == "duel":
            won = str(outcome).upper() == "WIN"
            try:
                wager = int(detail.get("wager", 0) or 0)
            except (TypeError, ValueError):
                wager = 0
            card["outcome"] = "WIN" if won else "LOSS"
            card["won"] = won
            card["amount"] = credited
            card["detail"] = {
                "winner": str(detail.get("winner") or viewer)[:80],
                "loser": str(detail.get("loser") or "the house")[:80],
                "wager": wager,
            }
        elif game == "trivia":
            # A redeem trivia surfaces the QUESTION (chat answers later) -> an
            # open-phase card; no winner/answer yet.
            card["outcome"] = "QUESTION"
            card["won"] = False
            card["amount"] = 0
            card["detail"] = {"phase": "open", "answer": outcome[:200]}
        elif game == "raffle":
            entered = outcome == "entered"
            card["outcome"] = "ENTERED" if entered else outcome.upper()[:120]
            card["won"] = entered
            card["amount"] = 0
            card["detail"] = {
                "phase": "open",
                "entrants": len(detail.get("entrants") or []),
            }
        else:
            return None
        return card

    def _emit(self, event: dict) -> None:
        if self._overlay is None:
            return
        overlay_event = self._to_overlay_event(event)
        if overlay_event is None:
            return
        try:
            self._overlay(overlay_event)
        except Exception as exc:  # noqa: BLE001 — overlay down never breaks the tick
            logger.warning("redeem overlay emit failed: %s", exc)
