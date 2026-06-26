"""Gap-c — chat-command economy games (the dispatcher the parser was missing).

A NEW own-cursor drain + dispatcher over the EXISTING closed-grammar parser
(:func:`kenning.twitch.commands.parse_command`), the SQLite-WAL
:class:`~kenning.twitch.economy.ledger.Ledger`, the pure provably-fair games, and
:class:`~kenning.twitch.economy.rng.ProvablyFairRNG`. Mirrors the redeem router's
own-cursor daemon-drain pattern (a SECOND consumer of the read sidecar's /buffer
that never acks, so it never steals events from the chat-reply drain).

THIS PASS (foundation + single-shot bet games): ``!points``/``!balance`` (read),
``!gamble <amount|all>`` (coinflip), ``!slots <amount|all>`` (slot machine),
``!leaderboard``, ``!help`` — each STAKE is debited first, the game is resolved
provably-fairly, and the multiplier payout is credited, with the payout tables
derived so EV == ``gamble_rtp`` (a net-negative house edge -> a sink). Viewers
EARN currency by watch-time (``earn_per_minute`` per active chatter, once per
minute). A per-viewer ``per_stream_loss_cap`` refuses a bet past the ceiling, and
a per-user command cooldown throttles spam.

The multi-viewer / transfer games are ALSO wired (gap-c next pass, 2026-06-24):
``!trivia`` (mod-started, first-correct-wins), ``!heist <amount>`` (join-window
group bet with a house bonus so a win profits), ``!duel @user <amount>`` +
``!accept`` (1v1 escrow, winner takes 2x), ``!raffle``/``!enter`` (mod-opened
entry window, house-funded prize), ``!wheel`` (a per-stream free spin), and
``!give @user <amount>`` (viewer->viewer transfer, gated ``transfers_enabled``).
Every stake/payout is a keyed-leg ledger write so an EventSub replay never
double-applies; each timed game resolves in :meth:`tick` at its deadline.

The abliterated model is NEVER in this path — every action is a deterministic
closed-grammar parse + ledger arithmetic. Chat-command replies route through the
injected ``announce_fn`` (the stream bus), never the team mic.

ANTICHEAT (BR-P1): stdlib only (json/logging/time/urllib/collections) + the
economy/commands siblings; no new deps. Flag-gated default-OFF (the orchestrator
only constructs this when ``twitch.economy.enabled`` AND
``twitch.economy.chat_commands_enabled``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.request
from collections import OrderedDict
from typing import Callable, Optional

from kenning.twitch.clients.eventsub import ChatEvent
from kenning.twitch.commands import ALL_SENTINEL, Command, CommandKind, parse_command
from kenning.twitch.economy.games import (
    Duel,
    Heist,
    Raffle,
    Slots,
    SpinTheWheel,
    Trivia,
    WheelSegment,
)
from kenning.twitch.economy.ledger import InsufficientFunds, Ledger
from kenning.twitch.economy.rng import ProvablyFairRNG
from kenning.twitch.redeem_router import _LRUSet

logger = logging.getLogger("kenning.twitch.economy.chat_games")

__all__ = [
    "ChatGameRouter",
    "make_chat_command_drain_fn",
    "chat_event_from_buffer",
    "DEFAULT_SLOT_SYMBOLS",
    "DEFAULT_WHEEL_SEGMENTS",
]

# 6 symbols -> P(win)=1/36 on 3 reels; the win multiplier is floor(rtp * 36).
DEFAULT_SLOT_SYMBOLS: tuple[str, ...] = ("cherry", "lemon", "bell", "star", "seven", "skull")


def _default_wheel_segments() -> list[WheelSegment]:
    """The free-spin wheel: all-positive point payouts (no LOSE_ALL -> AT-4-safe
    by default). EV is positive by design -- !wheel is a capped free reward, not a
    bet (the per-stream cap is what bounds the faucet)."""
    return [
        WheelSegment("100", weight=3.0, payout=100),
        WheelSegment("250", weight=2.0, payout=250),
        WheelSegment("50", weight=3.0, payout=50),
        WheelSegment("500", weight=1.0, payout=500),
        WheelSegment("JACKPOT 1000", weight=0.25, payout=1000),
        WheelSegment("10", weight=2.5, payout=10),
    ]


DEFAULT_WHEEL_SEGMENTS = _default_wheel_segments

_PRESENCE_WINDOW_S = 90.0   # a viewer counts as "active" (earning) for this long after a message
_MSG_INDEX_MAX = 4096       # bound on the {login -> last message_id} map (delete-moderation)
_GAMBLE_WIN_P = 0.5         # coinflip win probability


def make_chat_command_drain_fn(
    read_endpoint: str,
    *,
    timeout: float = 1.0,
    http_get: Callable[[str, float], bytes] | None = None,
) -> Callable[[], list[ChatEvent]]:
    """Own-cursor drain returning EVERY chat ``ChatEvent`` from the read sidecar.

    The router needs every message (presence/earn + the delete message-id index),
    then filters commands itself via ``parse_command``. GETs ``{endpoint}/buffer?
    since=<own cursor>`` and NEVER POSTs ``/ack`` (so this third consumer never
    steals events from the chat-reply or redeem drains). Fail-safe: any error
    (sidecar down / bad body) returns ``[]`` so the caller skips the tick.
    """
    base = read_endpoint.rstrip("/")
    cursor = {"v": 0}

    def _urllib_get(url: str, to: float) -> bytes:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=to) as r:  # nosec B310 - loopback only
            return r.read() or b"{}"

    fetch = http_get if http_get is not None else _urllib_get

    def drain() -> list[ChatEvent]:
        try:
            raw = fetch(f"{base}/buffer?since={cursor['v']}", timeout)
            data = json.loads(raw or b"{}")
        except Exception as exc:  # noqa: BLE001 — sidecar down / bad body -> skip tick
            logger.debug("chat-command drain failed: %s", exc)
            return []
        if not isinstance(data, dict):
            return []
        try:
            cursor["v"] = int(data.get("cursor", cursor["v"]) or cursor["v"])
        except (TypeError, ValueError):
            pass
        out: list[ChatEvent] = []
        for wrapped in data.get("events", []) or []:
            try:
                if not isinstance(wrapped, dict):
                    continue
                event = wrapped.get("event")
                if not isinstance(event, dict) or event.get("type") != "chat":
                    continue
                ce = chat_event_from_buffer(event)
                if ce is not None:
                    out.append(ce)
            except Exception:  # noqa: BLE001 — skip a malformed wrapper, never crash
                continue
        return out

    return drain


def chat_event_from_buffer(event: dict) -> Optional[ChatEvent]:
    """Build a :class:`ChatEvent` from the read sidecar's FLAT buffered chat dict
    ``{"type":"chat","message_id","chatter_login","chatter_name","chatter_user_id",
    "text"[,"badges"]}`` (twitch_read_sidecar._map_notification). NOTE: this is NOT
    the raw EventSub envelope ``ChatEvent.from_eventsub`` parses (which reads the
    NESTED ``message.text`` / ``chatter_user_login``) — the sidecar already flattened
    it, so we map the flat fields directly. Returns None on a non-chat / unusable dict.

    Delegates to the canonical :meth:`ChatEvent.from_buffer` so the flat-shape mapping
    has a SINGLE source of truth shared with the chat-reply drain (they drifted apart
    once — the reply drain used ``from_eventsub`` and silently dropped every line)."""
    return ChatEvent.from_buffer(event)


class ChatGameRouter:
    """Drains chat, runs the economy commands, persists balances to the ledger.

    Construct with an injected ``drain_fn`` (a ``make_chat_command_drain_fn`` or a
    canned list for tests), a long-lived ``ledger``, an ``rng``, the
    ``cfg`` (``TwitchEconomyConfig``), and an ``announce_fn(text)`` that speaks on
    the STREAM bus. ``now_fn``/``epoch_fn`` are injectable for deterministic tests.
    """

    def __init__(
        self,
        drain_fn: Callable[[], list],
        *,
        ledger: Ledger,
        cfg: object,
        rng: ProvablyFairRNG | None = None,
        announce_fn: Callable[[str], object] | None = None,
        overlay_emit: Callable[[dict], object] | None = None,
        now_fn: Callable[[], float] = time.monotonic,
        epoch_fn: Callable[[], float] = time.time,
        dedup_max: int = _MSG_INDEX_MAX,
        chat_cfg: object | None = None,
    ) -> None:
        self._drain = drain_fn
        self._ledger = ledger
        self._cfg = cfg
        # The CHAT config (TwitchChatConfig) — only ``commands_panel_doc_url`` is
        # read, to build the on-demand !ultron commands-panel reply. None (or a
        # config lacking the field) just omits the guide link; the panel still
        # posts. Kept separate from the ECONOMY cfg in ``self._cfg``.
        self._chat_cfg = chat_cfg
        self._rng = rng or ProvablyFairRNG()
        self._announce = announce_fn
        # Optional overlay sink (the orchestrator wires this to the OBS overlay
        # server's ``emit``, the SAME sink the redeem router uses). None => the
        # games still run + announce, just without an on-stream card (byte-identical
        # to the pre-overlay router for every existing caller / the overlay-OFF
        # boot). Every emit is fail-safe: an overlay error never breaks a game/tick.
        self._overlay = overlay_emit
        self._now = now_fn
        self._epoch = epoch_fn
        self._slots = Slots(DEFAULT_SLOT_SYMBOLS, reels=3, rng=self._rng)
        self._trivia_game = Trivia(rng=self._rng)
        self._heist_game = Heist(
            rng=self._rng,
            house_bonus_pct=_as_float(getattr(cfg, "heist_house_bonus_pct", 0.5), 0.5),
        )
        self._duel_game = Duel(rng=self._rng)
        self._raffle_game = Raffle(rng=self._rng)
        self._wheel = SpinTheWheel(_default_wheel_segments(), rng=self._rng)
        self._trivia: Optional[dict] = None      # {question, deadline (monotonic), prize}
        # Multi-viewer state machines (gap-c next pass, 2026-06-24). Each opens a
        # deadline (monotonic, via self._now) and resolves in tick() at expiry.
        self._heist: Optional[dict] = None       # {round_id, deadline, pot, participants:{uid:(login,stake)}}
        self._duels: dict[str, dict] = {}        # target_login_lower -> pending challenge
        self._raffle: Optional[dict] = None      # {round_id, deadline, prize}
        self._wheel_spins: dict[str, int] = {}   # user_id -> free !wheel spins used this session
        self._login_to_uid: dict[str, str] = {}  # login_lower -> user_id (from presence, for !give/!duel)
        self._uid_to_login: dict[str, str] = {}   # user_id -> display login (for the leaderboard)
        self._round_seq = 0                      # monotonically-increasing round id source
        self._dedup = _LRUSet(dedup_max)
        self._presence: dict[str, tuple[str, float]] = {}    # user_id -> (login, last_seen)
        self._last_msg: "OrderedDict[str, str]" = OrderedDict()  # login_lower -> message_id
        self._net_loss: dict[str, int] = {}                  # user_id -> net loss this session
        self._cooldown: dict[str, float] = {}                # user_id -> last command monotonic
        self._last_earn_minute: Optional[int] = None
        self._last_auto_trivia_epoch: Optional[float] = None   # arms on first tick
        self._nonce = 0

    # -- public surface ---------------------------------------------------- #
    def last_message_id(self, login: str) -> Optional[str]:
        """The most-recent chat message_id seen for ``login`` (for voice
        delete-moderation), or None. Login is matched case-insensitively."""
        return self._last_msg.get((login or "").strip().lower())

    def _event_key(self, ev: ChatEvent) -> str:
        """A STABLE per-event idempotency key: the Twitch message_id when present,
        else a content-hash surrogate of (user_id, text). Stable across an EventSub
        replay so the command dedup AND every ledger leg key match on re-delivery --
        a wall-clock fallback would mint a fresh key each time and double-apply the
        stake/payout."""
        mid = getattr(ev, "message_id", "") or ""
        if mid:
            return mid
        uid = getattr(ev, "chatter_user_id", "") or ""
        text = getattr(ev, "text", "") or ""
        return "syn:" + hashlib.sha1(  # noqa: S324 — an idempotency key, not security
            f"{uid}:{text}".encode("utf-8")).hexdigest()[:16]

    def tick(self) -> int:
        """Drain + dispatch one batch; accrue watch-time earnings. Returns the
        number of commands handled. Fail-safe per event (never raises)."""
        try:
            events = self._drain() or []
        except Exception as exc:  # noqa: BLE001
            logger.debug("chat-command drain raised: %s", exc)
            events = []
        handled = 0
        for ev in events:
            try:
                self._observe(ev)
                cmd = parse_command(ev)
                if cmd is None:
                    # An ordinary (non-command) message — the only thing it can
                    # drive is a trivia answer during an active round.
                    self._maybe_trivia_answer(ev)
                    continue
                if self._deferred_to_streamelements(cmd):
                    # !points/!balance and !gamble are answered by StreamElements'
                    # own chat bot — Ultron stays silent so viewers don't get a
                    # double reply. Drop BEFORE dedup so the event is a true no-op.
                    continue
                mid = self._event_key(ev)
                if not self._dedup.add(mid):
                    continue  # EventSub replay of an already-processed command
                if self._dispatch(ev, cmd):
                    handled += 1
            except Exception as exc:  # noqa: BLE001 — one bad event never kills the loop
                logger.warning("chat-command tick error: %s", exc)
        self._expire_trivia()
        self._expire_heist()
        self._expire_duels()
        self._expire_raffle()
        self._accrue_earnings()
        self._maybe_auto_trivia()
        return handled

    # -- presence / message-id / earn ------------------------------------- #
    def _observe(self, ev: ChatEvent) -> None:
        uid = getattr(ev, "chatter_user_id", "") or ""
        login = (getattr(ev, "chatter_login", "") or "").strip().lower()
        if uid:
            self._presence[uid] = (login, self._now())
            if login:
                self._login_to_uid[login] = uid   # for !give / !duel target resolution
                self._uid_to_login[uid] = login   # for the leaderboard display
                # SE-points backend: map uid -> login so the ledger can address the
                # StreamElements API (keyed by login). No-op for the SQLite ledger.
                _reg = getattr(self._ledger, "register", None)
                if _reg is not None:
                    try:
                        _reg(uid, login)
                    except Exception:  # noqa: BLE001 — must never break ingest
                        pass
        if login:
            mid = getattr(ev, "message_id", "") or ""
            if mid:
                self._last_msg[login] = mid
                self._last_msg.move_to_end(login)
                while len(self._last_msg) > _MSG_INDEX_MAX:
                    self._last_msg.popitem(last=False)

    def _accrue_earnings(self) -> None:
        per_min = _as_int(getattr(self._cfg, "earn_per_minute", 0))
        if per_min <= 0:
            return
        minute = int(self._epoch() // 60)
        if self._last_earn_minute is None:
            self._last_earn_minute = minute     # arm on first tick; pay from the next minute
            return
        if minute <= self._last_earn_minute:
            return
        self._last_earn_minute = minute
        cutoff = self._now() - _PRESENCE_WINDOW_S
        for uid, (login, seen) in list(self._presence.items()):
            if seen < cutoff:
                self._presence.pop(uid, None)   # drop stale presence (bounded memory)
                continue
            try:
                self._ledger.credit(uid, per_min, "watch-time", f"earn:{uid}:{minute}")
            except Exception as exc:  # noqa: BLE001 — earn must never break the tick
                logger.debug("earn credit skipped for %s: %s", login or uid, exc)

    # -- StreamElements deferral ------------------------------------------- #
    def _deferred_to_streamelements(self, cmd: Command) -> bool:
        """True iff this command must be left to StreamElements' chat bot.

        The channel runs ONE economy on StreamElements, whose bot already answers
        ``!points``/``!balance`` (POINTS) and ``!gamble`` (GAMBLE). When
        ``economy.defer_points_gamble_to_streamelements`` is set (default True),
        Ultron does NOT handle those two kinds — every other command still runs.
        """
        if not bool(getattr(self._cfg, "defer_points_gamble_to_streamelements", True)):
            return False
        return cmd.kind in (CommandKind.POINTS, CommandKind.GAMBLE)

    # -- dispatch ---------------------------------------------------------- #
    def _dispatch(self, ev: ChatEvent, cmd: Command) -> bool:
        k = cmd.kind
        if k is CommandKind.POINTS:
            return self._cmd_points(cmd)
        if k is CommandKind.GAMBLE:
            return self._cmd_bet(ev, cmd, "gamble")
        if k is CommandKind.SLOTS:
            return self._cmd_bet(ev, cmd, "slots")
        if k is CommandKind.TRIVIA:
            return self._cmd_trivia(cmd)
        if k is CommandKind.LEADERBOARD:
            return self._cmd_leaderboard(cmd)
        if k is CommandKind.HELP:
            return self._cmd_help(cmd)
        if k is CommandKind.ULTRON:
            return self._cmd_ultron(cmd)
        if k is CommandKind.HEIST:
            return self._cmd_heist(ev, cmd)
        if k is CommandKind.DUEL:
            return self._cmd_duel(ev, cmd)
        if k is CommandKind.ACCEPT:
            return self._cmd_accept(ev, cmd)
        if k is CommandKind.RAFFLE:
            return self._cmd_raffle(ev, cmd)
        if k is CommandKind.GIVE:
            return self._cmd_give(ev, cmd)
        if k is CommandKind.WHEEL:
            return self._cmd_wheel(ev, cmd)
        # UNKNOWN
        self._reply(f"@{cmd.user_login} unknown command. Try !help.")
        return False

    def _cmd_points(self, cmd: Command) -> bool:
        if not cmd.user_id:
            return False
        bal = self._ledger.balance(cmd.user_id)
        self._reply(f"@{cmd.user_login} you have {bal} {self._currency()}.")
        return True

    def _cmd_leaderboard(self, cmd: Command) -> bool:
        try:
            balances = self._ledger.rebuild_balances()
        except Exception as exc:  # noqa: BLE001
            logger.debug("leaderboard rebuild failed: %s", exc)
            return False
        top = sorted(balances.items(), key=lambda kv: kv[1], reverse=True)[:5]
        if not top:
            self._reply(f"No one has any {self._currency()} yet. Start chatting!")
            return True
        # ONE message — Twitch collapses a chat message to a single line, so true
        # line breaks aren't possible; ranks go inline, separated for readability.
        parts = [
            f"{i + 1}. {self._uid_to_login.get(uid, uid)} ({bal})"
            for i, (uid, bal) in enumerate(top)
        ]
        self._reply("Top " + self._currency() + ": " + " · ".join(parts))
        return True

    def _cmd_help(self, cmd: Command) -> bool:
        deferred = bool(getattr(self._cfg, "defer_points_gamble_to_streamelements", True))
        # Only advertise !points/!gamble as Ultron commands when Ultron actually
        # handles them; otherwise StreamElements' bot owns those two.
        head = "Commands: " if deferred else "Commands: !points, !gamble <amount|all>, "
        self._reply(
            head + "!slots <amount|all>, "
            "!wheel (free spin), !heist <amount>, !duel @user <amount> + !accept, "
            "!raffle, !give @user <amount>, !trivia (mods), !leaderboard. Earn "
            + self._currency() + " by watching."
        )
        return True

    def _cmd_ultron(self, cmd: Command) -> bool:
        """``!ultron`` — post the SAME condensed commands-panel message viewers see
        on the periodic auto-post, on demand. Reads the chat config's
        ``commands_panel_doc_url`` (threaded in at construction) so the guide link is
        appended. Per-user cooldown (the shared throttle the bet games use) so a
        spammer can't flood the panel; fail-safe (a missing/empty config just omits
        the link, an import/build error is swallowed and the tick continues)."""
        uid = cmd.user_id or ""
        if uid and self._cooldown_active(uid):
            return False
        try:
            from kenning.twitch.panel import build_commands_panel_text
            text = build_commands_panel_text(self._chat_cfg)
        except Exception as exc:  # noqa: BLE001 — never break the tick on a panel build
            logger.debug("ultron panel build failed: %s", exc)
            return False
        if uid:
            self._mark_cooldown(uid)
        self._reply(text)
        return True

    # -- shared helpers for the multi-viewer / transfer games ------------- #
    def _new_round_id(self, game: str) -> str:
        self._round_seq += 1
        return f"{game}:{self._round_seq}"

    def _uid_for_login(self, login: str) -> Optional[str]:
        """Resolve a chat login to a user_id from this session's presence (the
        ledger keys on user_id, never a login). None when the login has not been
        seen chatting this session (an unknown recipient cannot receive)."""
        return self._login_to_uid.get((login or "").strip().lower())

    def _cooldown_active(self, uid: str) -> bool:
        cd = _as_float(getattr(self._cfg, "command_cooldown_seconds", 5))
        if cd <= 0:
            return False
        return (self._now() - self._cooldown.get(uid, -1.0e9)) < cd

    def _mark_cooldown(self, uid: str) -> None:
        self._cooldown[uid] = self._now()

    def _stake_from_args(self, cmd: Command, bal: int) -> tuple[Optional[int], Optional[str]]:
        """Resolve a bet amount (handling the 'all' sentinel) + validate min/max/
        balance. Returns ``(stake, None)`` or ``(None, error_text)``."""
        amt = cmd.args.get("amount")
        if amt is None:
            return None, cmd.args.get("error", "bad amount")
        stake = bal if amt == ALL_SENTINEL else _as_int(amt)
        min_bet = max(1, _as_int(getattr(self._cfg, "min_bet", 1)))
        max_bet = _as_int(getattr(self._cfg, "max_bet", 0))
        if stake < min_bet:
            return None, f"minimum bet is {min_bet}"
        if max_bet and stake > max_bet:
            return None, f"maximum bet is {max_bet} {self._currency()}"
        if stake > bal:
            return None, f"you only have {bal} {self._currency()}"
        return stake, None

    # -- !give (viewer -> viewer transfer, gated transfers_enabled) -------- #
    def _cmd_give(self, ev: ChatEvent, cmd: Command) -> bool:
        login = cmd.user_login or "viewer"
        uid = cmd.user_id or ""
        if not uid:
            return False
        if not bool(getattr(self._cfg, "transfers_enabled", False)):
            self._reply(f"@{login} transfers are disabled.")
            return False
        target_login = str(cmd.args.get("target") or "")
        amount = cmd.args.get("amount")
        if not target_login or amount is None:
            self._reply(f"@{login} {cmd.args.get('error', 'usage: !give @user <amount>')}.")
            return False
        target_uid = self._uid_for_login(target_login)
        if target_uid is None:
            self._reply(f"@{login} I don't know @{target_login} yet -- they must chat first.")
            return False
        if target_uid == uid:
            self._reply(f"@{login} you can't give to yourself.")
            return False
        amt = _as_int(amount)
        if amt < 1:
            self._reply(f"@{login} amount must be positive.")
            return False
        if self._cooldown_active(uid):
            return False
        bal = self._ledger.balance(uid)
        if amt > bal:
            self._reply(f"@{login} you only have {bal} {self._currency()}.")
            return False
        mid = self._event_key(ev)
        self._mark_cooldown(uid)
        try:
            self._ledger.debit(uid, amt, "give", f"give:{mid}:from")
        except InsufficientFunds:
            self._reply(f"@{login} you only have {self._ledger.balance(uid)} {self._currency()}.")
            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("give debit failed for %s: %s", uid, exc)
            return False
        try:
            self._ledger.credit(target_uid, amt, "gift", f"give:{mid}:to")
        except Exception as exc:  # noqa: BLE001 — replay-safe; recovers on resume
            logger.warning("give credit failed for %s: %s", target_uid, exc)
        self._reply(
            f"@{login} gave {amt} {self._currency()} to @{target_login}. "
            f"Balance {self._ledger.balance(uid)}."
        )
        return True

    # -- !wheel (free spin, per-stream cap, house-funded payout) ----------- #
    def _cmd_wheel(self, ev: ChatEvent, cmd: Command) -> bool:
        login = cmd.user_login or "viewer"
        uid = cmd.user_id or ""
        if not uid:
            return False
        cap = _as_int(getattr(self._cfg, "wheel_free_per_stream", 1))
        if cap <= 0:
            self._reply(f"@{login} the free wheel is disabled.")
            return False
        if self._wheel_spins.get(uid, 0) >= cap:
            self._reply(f"@{login} no free spins left this stream.")
            return False
        if self._cooldown_active(uid):
            return False
        self._mark_cooldown(uid)
        self._wheel_spins[uid] = self._wheel_spins.get(uid, 0) + 1
        rnd = self._rng.new_round()
        self._nonce += 1
        res = self._wheel.spin(rnd.server_seed, nonce=self._nonce)
        payout = max(0, _as_int(res.segment.payout))
        mid = self._event_key(ev)
        if payout > 0:
            try:
                self._ledger.credit(uid, payout, "wheel", f"wheel:{mid}:win")
            except Exception as exc:  # noqa: BLE001
                logger.warning("wheel credit failed for %s: %s", uid, exc)
        self._reply(
            f"@{login} spun the wheel -> {res.segment.label}. +{payout} "
            f"{self._currency()}. Balance {self._ledger.balance(uid)}."
        )
        # On-stream wheel card: reveal the landed segment + the payout (always a
        # win for the free spin -> green accent unless the segment paid 0).
        self._emit_overlay(
            "wheel", login,
            outcome=str(res.segment.label),
            title="WHEEL",
            won=payout > 0,
            amount=payout,
            detail={"segment": str(res.segment.label), "payout": payout},
        )
        return True

    # -- !heist (group pooled bet, join window, house bonus) --------------- #
    def _cmd_heist(self, ev: ChatEvent, cmd: Command) -> bool:
        login = cmd.user_login or "viewer"
        uid = cmd.user_id or ""
        if not uid:
            return False
        bal = self._ledger.balance(uid)
        stake, err = self._stake_from_args(cmd, bal)
        if stake is None:
            self._reply(f"@{login} {err}.")
            return False
        if self._heist is not None and uid in self._heist["participants"]:
            self._reply(f"@{login} you're already in the heist.")
            return False
        if self._cooldown_active(uid):
            return False
        mid = self._event_key(ev)
        opened = False
        if self._heist is None:
            window = _as_float(getattr(self._cfg, "heist_window_seconds", 30)) or 30.0
            self._heist = {
                "round_id": self._new_round_id("heist"),
                "deadline": self._now() + window,
                "pot": 0,
                "participants": {},
                "window": window,
            }
            opened = True
        h = self._heist
        try:
            self._ledger.debit(uid, stake, "heist bet", f"heist:{mid}:bet")
        except InsufficientFunds:
            if opened and not h["participants"]:
                self._heist = None     # roll back an empty just-opened round
            self._reply(f"@{login} you only have {self._ledger.balance(uid)} {self._currency()}.")
            return False
        except Exception as exc:  # noqa: BLE001
            if opened and not h["participants"]:
                self._heist = None
            logger.warning("heist debit failed for %s: %s", uid, exc)
            return False
        self._mark_cooldown(uid)
        h["participants"][uid] = (login, int(stake))
        h["pot"] += int(stake)
        if opened:
            self._reply(
                f"@{login} started a HEIST for {stake}! Type !heist <amount> in the "
                f"next {int(h['window'])}s to join the crew. Pot {h['pot']}."
            )
        else:
            self._reply(
                f"@{login} joined the heist for {stake}. Pot {h['pot']} "
                f"({len(h['participants'])} in)."
            )
        return True

    def _expire_heist(self) -> None:
        h = self._heist
        if h is None or self._now() < h["deadline"]:
            return
        self._heist = None     # close FIRST so a slow tick can't double-resolve
        participants = h["participants"]
        round_id = h["round_id"]
        min_players = max(1, _as_int(getattr(self._cfg, "heist_min_players", 1)))
        if len(participants) < min_players:
            for puid, (_pl, pstake) in participants.items():
                self._safe_credit(puid, pstake, "heist refund", f"{round_id}:{puid}:refund")
            self._reply(
                f"The heist needed {min_players} to run -- not enough joined. "
                "Everyone was refunded."
            )
            return
        logins = [pl for (pl, _s) in participants.values()]
        rnd = self._rng.new_round()
        self._nonce += 1
        try:
            res = self._heist_game.resolve(rnd.server_seed, logins, h["pot"], nonce=self._nonce)
        except Exception as exc:  # noqa: BLE001 — never strand stakes
            logger.warning("heist resolve failed (%s); refunding", exc)
            for puid, (_pl, pstake) in participants.items():
                self._safe_credit(puid, pstake, "heist refund", f"{round_id}:{puid}:refund")
            return
        per_head = int(res.payout_per_head)
        if per_head > 0:
            for puid in participants:
                self._safe_credit(puid, per_head, "heist win", f"{round_id}:{puid}:win")
        crew = ", ".join(f"@{pl}" for pl in logins)
        lead = logins[0] if logins else "the crew"
        if res.outcome in ("win", "partial"):
            self._reply(
                f"HEIST {res.outcome.upper()}! The crew ({crew}) each take "
                f"{per_head} {self._currency()}."
            )
            self._emit_overlay(
                "heist", lead,
                outcome=str(res.outcome).upper(),
                title="HEIST",
                won=True,
                amount=per_head,
                detail={"pot": int(h["pot"]), "crew": len(logins), "payout": per_head},
            )
        else:
            self._reply(f"HEIST FAILED. The crew ({crew}) lost their stakes.")
            self._emit_overlay(
                "heist", lead,
                outcome="FAIL",
                title="HEIST",
                won=False,
                amount=int(h["pot"]),
                detail={"pot": int(h["pot"]), "crew": len(logins), "payout": 0},
            )

    # -- !duel + !accept (1v1 escrow challenge) ---------------------------- #
    def _cmd_duel(self, ev: ChatEvent, cmd: Command) -> bool:
        login = cmd.user_login or "viewer"
        uid = cmd.user_id or ""
        if not uid:
            return False
        target_login = str(cmd.args.get("target") or "").strip().lower()
        amount = cmd.args.get("amount")
        if not target_login or amount is None:
            self._reply(f"@{login} {cmd.args.get('error', 'usage: !duel @user <amount>')}.")
            return False
        if target_login == (cmd.user_login or "").strip().lower():
            self._reply(f"@{login} you can't duel yourself.")
            return False
        target_uid = self._uid_for_login(target_login)
        if target_uid is None:
            self._reply(f"@{login} I don't know @{target_login} yet -- they must chat first.")
            return False
        if target_login in self._duels:
            self._reply(f"@{login} @{target_login} already has a pending duel.")
            return False
        bal = self._ledger.balance(uid)
        wager, err = self._stake_from_args(cmd, bal)
        if wager is None:
            self._reply(f"@{login} {err}.")
            return False
        if self._cooldown_active(uid):
            return False
        round_id = self._new_round_id("duel")
        try:
            self._ledger.debit(uid, wager, "duel stake", f"{round_id}:chal:bet")
        except InsufficientFunds:
            self._reply(f"@{login} you only have {self._ledger.balance(uid)} {self._currency()}.")
            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("duel debit failed for %s: %s", uid, exc)
            return False
        self._mark_cooldown(uid)
        window = _as_float(getattr(self._cfg, "duel_window_seconds", 60)) or 60.0
        self._duels[target_login] = {
            "round_id": round_id,
            "challenger_uid": uid,
            "challenger_login": cmd.user_login or login,
            "target_login": target_login,
            "target_uid": target_uid,
            "wager": int(wager),
            "deadline": self._now() + window,
        }
        self._reply(
            f"@{login} challenges @{target_login} to a duel for {wager} "
            f"{self._currency()}! @{target_login}, type !accept in {int(window)}s."
        )
        return True

    def _cmd_accept(self, ev: ChatEvent, cmd: Command) -> bool:
        login = cmd.user_login or "viewer"
        uid = cmd.user_id or ""
        if not uid:
            return False
        my_login = (cmd.user_login or "").strip().lower()
        duel = self._duels.get(my_login)
        if duel is None:
            self._reply(f"@{login} you have no duel to accept.")
            return False
        wager = int(duel["wager"])
        bal = self._ledger.balance(uid)
        if wager > bal:
            self._reply(f"@{login} you need {wager} {self._currency()} to accept (you have {bal}).")
            return False
        del self._duels[my_login]
        round_id = duel["round_id"]
        try:
            self._ledger.debit(uid, wager, "duel stake", f"{round_id}:tgt:bet")
        except InsufficientFunds:
            self._reply(f"@{login} you only have {self._ledger.balance(uid)} {self._currency()}.")
            self._refund_duel(duel)
            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("duel accept debit failed for %s: %s", uid, exc)
            self._refund_duel(duel)
            return False
        challenger_login = duel["challenger_login"]
        rnd = self._rng.new_round()
        self._nonce += 1
        try:
            res = self._duel_game.resolve(
                rnd.server_seed, challenger_login, duel["target_login"], wager, nonce=self._nonce,
            )
        except Exception as exc:  # noqa: BLE001 — refund BOTH stakes, never strand
            logger.warning("duel resolve failed (%s); refunding both", exc)
            self._refund_duel(duel)
            self._safe_credit(uid, wager, "duel refund", f"{round_id}:tgt:refund")
            return False
        winner_is_challenger = str(res.winner).lower() == challenger_login.strip().lower()
        winner_uid = duel["challenger_uid"] if winner_is_challenger else uid
        winner_login = challenger_login if winner_is_challenger else (cmd.user_login or login)
        loser_login = (cmd.user_login or login) if winner_is_challenger else challenger_login
        pot = wager * 2
        self._safe_credit(winner_uid, pot, "duel win", f"{round_id}:win")
        self._reply(f"DUEL: @{winner_login} beat @{loser_login} and takes {pot} {self._currency()}.")
        self._emit_overlay(
            "duel", winner_login,
            outcome="WIN",
            title="DUEL",
            won=True,
            amount=pot,
            detail={"winner": winner_login, "loser": loser_login, "wager": int(wager)},
        )
        return True

    def _refund_duel(self, duel: dict) -> None:
        """Return the challenger's escrowed stake (the target never matched it)."""
        self._safe_credit(
            duel["challenger_uid"], int(duel["wager"]), "duel refund",
            f"{duel['round_id']}:chal:refund",
        )

    def _expire_duels(self) -> None:
        now = self._now()
        for target_login in [k for k, d in self._duels.items() if now >= d["deadline"]]:
            duel = self._duels.pop(target_login, None)
            if duel is None:
                continue
            self._refund_duel(duel)
            self._reply(
                f"@{duel['challenger_login']}'s duel challenge to @{target_login} "
                "expired. Stake refunded."
            )

    # -- !raffle (mod-opened entry window, house prize) -------------------- #
    def _cmd_raffle(self, ev: ChatEvent, cmd: Command) -> bool:
        login = cmd.user_login or "viewer"
        uid = cmd.user_id or ""
        if not uid:
            return False
        if self._raffle is None:
            if not cmd.is_mod:
                self._reply(f"@{login} no raffle is running.")
                return False
            window = _as_float(getattr(self._cfg, "raffle_window_seconds", 60)) or 60.0
            prize = max(0, _as_int(getattr(self._cfg, "raffle_prize", 500)))
            if not self._open_raffle_game():
                return False
            self._raffle = {
                "round_id": self._new_round_id("raffle"),
                "deadline": self._now() + window,
                "prize": prize,
            }
            self._reply(
                f"RAFFLE open for {int(window)}s! Type !raffle (or !enter) to join. "
                f"Prize: {prize} {self._currency()}."
            )
            return True
        if self._raffle_game.enter(cmd.user_login or login):
            self._reply(f"@{login} entered the raffle.")
            return True
        return False   # duplicate / closed entry -> silent

    def _open_raffle_game(self) -> bool:
        """Open the stateful Raffle with an effectively-infinite internal window so
        OUR deadline (self._now-driven, test-controllable) owns the close. Recovers
        a stuck-open raffle by drawing it down first. Fail-safe."""
        try:
            self._raffle_game.open(window_s=1.0e9)
            return True
        except Exception:  # noqa: BLE001 — already open: draw it down + reopen
            try:
                self._raffle_game.draw(self._rng.new_round().server_seed)
                self._raffle_game.open(window_s=1.0e9)
                return True
            except Exception as exc:  # noqa: BLE001
                logger.warning("raffle open failed: %s", exc)
                return False

    def _expire_raffle(self) -> None:
        r = self._raffle
        if r is None or self._now() < r["deadline"]:
            return
        self._raffle = None    # close FIRST
        rnd = self._rng.new_round()
        self._nonce += 1
        try:
            res = self._raffle_game.draw(rnd.server_seed, nonce=self._nonce)
        except Exception as exc:  # noqa: BLE001
            logger.warning("raffle draw failed: %s", exc)
            return
        if res.winner is None:
            self._reply("The raffle closed with no entrants.")
            return
        prize = int(r["prize"])
        winner_uid = self._uid_for_login(res.winner)
        if prize > 0 and winner_uid:
            self._safe_credit(winner_uid, prize, "raffle win", f"{r['round_id']}:win")
        self._reply(
            f"RAFFLE winner: @{res.winner}! +{prize} {self._currency()} "
            f"({len(res.entrants)} entered)."
        )
        self._emit_overlay(
            "raffle", str(res.winner),
            outcome="WINNER",
            title="RAFFLE",
            won=True,
            amount=int(prize),
            detail={
                "winner": str(res.winner),
                "entrants": len(res.entrants),
                "phase": "result",
            },
        )

    def _safe_credit(self, uid: str, amount: int, reason: str, key: str) -> None:
        """Ledger credit that never raises into a game-resolution path (a credit is
        replay-safe; a failure here recovers on the next ledger rebuild)."""
        if amount <= 0 or not uid:
            return
        try:
            self._ledger.credit(uid, int(amount), reason, key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("credit failed (%s) key=%s: %s", reason, key, exc)

    # -- trivia (multi-viewer first-correct round) ------------------------- #
    def _cmd_trivia(self, cmd: Command) -> bool:
        """Mod-started trivia round: draw a provably-fair question, open a window;
        the FIRST correct chat answer wins a house-funded prize."""
        if not cmd.is_mod:
            self._reply(f"@{cmd.user_login} only mods can start trivia.")
            return False
        if self._trivia_active():
            self._reply("A trivia round is already running.")
            return False
        self._start_trivia()
        return True

    def _trivia_active(self) -> bool:
        """True iff a trivia round is currently open (not past its deadline)."""
        return self._trivia is not None and self._now() < self._trivia["deadline"]

    def _start_trivia(self) -> None:
        """Draw a provably-fair question, open the window, and announce it. Shared
        by the mod ``!trivia`` command and the periodic auto-trivia trigger."""
        rnd = self._rng.new_round()
        self._nonce += 1
        question, _idx, _prov = self._trivia_game.draw_question(rnd.server_seed, nonce=self._nonce)
        window = _as_float(getattr(self._cfg, "trivia_window_seconds", 30)) or 30.0
        prize = max(0, _as_int(getattr(self._cfg, "trivia_prize", 100)))
        self._trivia = {"question": question, "deadline": self._now() + window, "prize": prize}
        self._reply(
            f"TRIVIA for {prize} {self._currency()} — first correct answer wins: "
            f"{question.question}"
        )

    def _maybe_auto_trivia(self) -> None:
        """Auto-start a trivia round about every ``trivia_auto_interval_minutes``
        with NO mod command (0 disables). Uses the SAME injectable epoch clock as
        watch-time earning so it is unit-testable with a fake clock. Never starts
        a round while one is already active (the mod command and auto share the
        single ``self._trivia`` slot)."""
        interval_min = _as_int(getattr(self._cfg, "trivia_auto_interval_minutes", 0))
        if interval_min <= 0:
            return
        now = self._epoch()
        if self._last_auto_trivia_epoch is None:
            # Arm on the first tick; fire from the next interval so a fresh boot
            # doesn't immediately blast a round.
            self._last_auto_trivia_epoch = now
            return
        if (now - self._last_auto_trivia_epoch) < interval_min * 60.0:
            return
        if self._trivia_active():
            return   # a round (mod- or auto-started) is live -> don't stack
        self._last_auto_trivia_epoch = now
        self._start_trivia()

    def _maybe_trivia_answer(self, ev: ChatEvent) -> None:
        """Scan one ordinary chat message for the trivia answer. The FIRST correct
        answerer wins (the round closes atomically before crediting, so a replay or
        a second correct answer can't double-award)."""
        t = self._trivia
        if t is None or self._now() >= t["deadline"]:
            return
        uid = getattr(ev, "chatter_user_id", "") or ""
        login = getattr(ev, "chatter_login", "") or "viewer"
        text = getattr(ev, "text", "") or ""
        if not uid or not text:
            return
        if not self._trivia_game.check_answer(t["question"], text):
            return
        prize = int(t["prize"])
        answer = t["question"].answer
        self._trivia = None   # close the round FIRST -> first-correct-wins, no double-award
        mid = self._event_key(ev)
        if prize > 0:
            try:
                self._ledger.credit(uid, prize, "trivia win", f"trivia:{mid}:win")
            except Exception as exc:  # noqa: BLE001
                logger.warning("trivia credit failed for %s: %s", uid, exc)
        self._reply(f"@{login} got it! The answer was '{answer}'. +{prize} {self._currency()}.")
        self._emit_overlay(
            "trivia", login,
            outcome="WINNER",
            title="TRIVIA",
            won=True,
            amount=int(prize),
            detail={"winner": str(login), "answer": str(answer), "phase": "result"},
        )

    def _expire_trivia(self) -> None:
        t = self._trivia
        if t is not None and self._now() >= t["deadline"]:
            self._trivia = None
            self._reply(f"Trivia timed out. The answer was '{t['question'].answer}'.")

    def _cmd_bet(self, ev: ChatEvent, cmd: Command, game: str) -> bool:
        uid = cmd.user_id or ""
        login = cmd.user_login or "viewer"
        if not uid:
            return False
        # per-user cooldown (silent throttle so a spammer can't farm reply spam).
        cd = _as_float(getattr(self._cfg, "command_cooldown_seconds", 5))
        now = self._now()
        if cd > 0 and (now - self._cooldown.get(uid, -1.0e9)) < cd:
            return False

        amt = cmd.args.get("amount")
        if amt is None:
            self._reply(f"@{login} {cmd.args.get('error', 'bad amount')}.")
            return False
        bal = self._ledger.balance(uid)
        stake = bal if amt == ALL_SENTINEL else _as_int(amt)

        min_bet = max(1, _as_int(getattr(self._cfg, "min_bet", 1)))
        max_bet = _as_int(getattr(self._cfg, "max_bet", 0))
        if stake < min_bet:
            self._reply(f"@{login} minimum bet is {min_bet}.")
            return False
        if max_bet and stake > max_bet:
            self._reply(f"@{login} maximum bet is {max_bet} {self._currency()}.")
            return False
        if stake > bal:
            self._reply(f"@{login} you only have {bal} {self._currency()}.")
            return False
        cap = _as_int(getattr(self._cfg, "per_stream_loss_cap", 0))
        if cap and self._net_loss.get(uid, 0) + stake > cap:
            self._reply(f"@{login} you've hit the per-stream loss cap of {cap}. Take a break.")
            return False

        mid = self._event_key(ev)
        self._cooldown[uid] = now
        try:
            self._ledger.debit(uid, stake, f"{game} bet", f"{game}:{mid}:bet")
        except InsufficientFunds:
            self._reply(f"@{login} you only have {self._ledger.balance(uid)} {self._currency()}.")
            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s debit failed for %s: %s", game, uid, exc)
            return False

        payout, detail = self._resolve_payout(game, stake, mid)
        if payout > 0:
            try:
                self._ledger.credit(uid, payout, f"{game} win", f"{game}:{mid}:win")
            except Exception as exc:  # noqa: BLE001 — replay-safe; a crash here recovers on resume
                logger.warning("%s credit failed for %s: %s", game, uid, exc)

        net = payout - stake
        if net < 0:
            self._net_loss[uid] = self._net_loss.get(uid, 0) + (-net)
        self._reply(self._bet_line(login, game, stake, payout, net, self._ledger.balance(uid)))
        if game == "slots":
            # On-stream slots card: the reels settle on the ACTUAL pulled symbols;
            # color-coded WIN (triple) vs loss; amount = net (won) / stake (lost).
            won = payout > 0
            self._emit_overlay(
                "slots", login,
                outcome=("WIN" if won else "LOSS"),
                title="SLOTS",
                won=won,
                amount=(net if won else stake),
                detail={
                    "reels": detail.get("reels"),
                    "win_symbol": detail.get("win_symbol"),
                    "stake": stake,
                    "payout": payout,
                    "net": net,
                },
            )
        return True

    # -- game payouts (EV == gamble_rtp) ---------------------------------- #
    def _resolve_payout(self, game: str, stake: int, mid: str) -> tuple[int, dict]:
        """Resolve a single-shot bet payout. Returns ``(payout, detail)`` where
        ``detail`` carries the on-stream specifics (the slot reels + win symbol).
        ``detail`` is ``{}`` for games without extra overlay data."""
        rtp = _as_float(getattr(self._cfg, "gamble_rtp", 0.90)) or 0.90
        rnd = self._rng.new_round()
        self._nonce += 1
        nonce = self._nonce
        if game == "gamble":
            # Coinflip: P(win)=0.5; a win pays floor(stake*rtp/0.5) GROSS, so
            # EV = 0.5*payout - stake = stake*(rtp-1) (net-negative house edge).
            draw = self._rng.uniform_unit(rnd.server_seed, str(mid), nonce)
            if draw < _GAMBLE_WIN_P:
                return int(stake * rtp / _GAMBLE_WIN_P), {}
            return 0, {}
        if game == "slots":
            res = self._slots.pull(rnd.server_seed, client_seed=str(mid), nonce=nonce)
            detail = {"reels": list(res.reels), "win_symbol": res.win_symbol}
            if res.is_win:
                s = len(DEFAULT_SLOT_SYMBOLS)
                mult = int(rtp * s * s)   # P(win)=1/s^2 -> EV = (1/s^2)*stake*mult = stake*rtp
                return stake * mult, detail
            return 0, detail
        return 0, {}

    # -- replies ----------------------------------------------------------- #
    def _bet_line(self, login: str, game: str, stake: int, payout: int, net: int, bal: int) -> str:
        cur = self._currency()
        if payout > 0:
            return (f"@{login} {game}: WON {payout} {cur} (net +{net}). "
                    f"Balance {bal}.")
        return f"@{login} {game}: lost {stake} {cur}. Balance {bal}."

    def _currency(self) -> str:
        c = getattr(self._cfg, "currency_name", "") or "one taps"
        return str(c)

    def _reply(self, text: str) -> None:
        if self._announce is None:
            return
        try:
            self._announce(text)
        except Exception as exc:  # noqa: BLE001 — a dead announce channel never breaks the loop
            logger.debug("chat-game reply failed: %s", exc)

    # -- overlay (on-stream chat-game card) -------------------------------- #
    def _emit_overlay(
        self,
        game: str,
        viewer: str,
        *,
        outcome: str,
        title: str,
        won: bool = False,
        amount: int = 0,
        detail: dict | None = None,
    ) -> None:
        """Emit ONE on-stream overlay card for a chat-game OUTCOME (fail-safe).

        Builds a ``{"type":"chat_game", ...}`` event matching the overlay
        validator's schema (``game`` discriminator + ``source:"chat"`` so the card
        is distinguishable from a channel-point redeem). ``amount`` is the "one
        taps" value (payout / wager / prize); ``won`` color-codes the card. Any
        exception is swallowed so a stalled/erroring overlay NEVER breaks the game
        or the tick (BR-2.3 error handling)."""
        sink = self._overlay
        if sink is None:
            return
        try:
            event = {
                "type": "chat_game",
                "game": str(game),
                "source": "chat",
                "viewer": str(viewer or "someone")[:80],
                "outcome": str(outcome or "")[:120],
                "title": str(title or "")[:120],
                "won": bool(won),
                "amount": int(amount),
                "detail": {k: v for k, v in (detail or {}).items() if v is not None},
            }
            sink(event)
        except Exception as exc:  # noqa: BLE001 — overlay down never breaks a game/tick
            logger.debug("chat-game overlay emit failed (game=%s): %s", game, exc)


def _as_int(value: object, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return default
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
