"""S1 — Twitch chat READ sidecar (loopback HTTP buffer over a pluggable source).

Runs as a SEPARATE process so NO Twitch transport ever loads into Ultron's
anticheat-pinned main process (BR-P1). The main/voice process keeps only a thin
``urllib`` client; this sidecar owns the EventSub WebSocket and exposes a tiny
loopback JSON HTTP surface that the consumer drains. Mirrors the proven
``scripts/embedder_server.py`` precedent: a loopback-ONLY ``ThreadingHTTPServer``
(127.0.0.1), a parent-death deadman thread (``os._exit`` when the parent pid is
gone), and fail-quiet logging.

ANTICHEAT POSTURE: pure compute + loopback networking only (poll a chat source,
buffer events, serve them over a 127.0.0.1 HTTP socket). NO input injection, NO
screen/window capture, NO foreign-process memory, NO hooks, never touches the
game -- the same class as OBS/Discord. Imports are stdlib only
(``http.server``/``socket``/``threading``/``json``/``collections.deque`` + the
optional EventSub transport which itself is stdlib ``socket``/``ssl``). Binds
127.0.0.1 ONLY.

SOURCE abstraction
------------------
The chat source is a pluggable object exposing ``poll() -> list[dict]`` (and an
optional ``close()``). The default real source (:class:`EventSubChatSource`)
drives the receive-only EventSub WebSocket client in
``kenning.twitch.clients.eventsub`` and maps each ``channel.chat.message``
notification to a small JSON-serializable dict. Tests inject a ``FakeSource``
instead, so the whole sidecar is exercisable WITHOUT a live Twitch connection,
creds, or models.

Protocol (JSON over loopback HTTP)
----------------------------------
  GET  /healthz          -> {"ok":true, "buffered":N, "cursor":M, "running":bool,
                             "dropped":D, "source":NAME}
  GET  /buffer?since=N   -> {"events":[...], "cursor":M}
                            Drains buffered events whose sequence id is > N
                            (``since`` defaults to the persisted consumer cursor).
  POST /ack {"cursor":N}  -> {"ok":true, "cursor":N}
                            Advances the consumer cursor and prunes acked events.

Each buffered event carries a monotonically increasing integer ``seq`` (the
cursor space) plus a ``ts`` ingest timestamp; the buffer is a thread-safe rolling
``deque`` with a ``maxlen`` cap AND a TTL so an idle consumer can never make the
sidecar grow without bound -- the oldest events are evicted first.

Master-flag wiring
------------------
With the Twitch master flag OFF the orchestrator MUST NOT spawn this process (the
``KENNING_TWITCH_*`` switches all default OFF; the flag-off-zero-sidecars
invariant is asserted by ``tests/twitch/test_config_anticheat_invariant.py``).
Run directly with the flag off and no source configured, the sidecar simply
serves an EMPTY buffer -- it is harmless on its own.

Run:  python scripts/twitch_read_sidecar.py [PORT]
Env:  KENNING_TWITCH_READ_PORT (default 8773; 8775 is the overlay sidecar),
      KENNING_TWITCH_READ_BUFFER_MAX (deque maxlen, default 2000),
      KENNING_TWITCH_READ_TTL_SECONDS (event TTL, default 900),
      KENNING_TWITCH_READ_POLL_SECONDS (source poll cadence, default 0.5),
      KENNING_TWITCH_PARENT_PID (parent pid for the deadman watchdog),
      KENNING_TWITCH_EVENTSUB_URL (override the wss endpoint for the real source).

Live EventSub subscription (the real source only subscribes when creds are set):
      KENNING_TWITCH_CLIENT_ID (the Twitch app client id; REQUIRED to subscribe —
        with it unset the source connects but never subscribes and serves empty),
      KENNING_TWITCH_BROADCASTER_LOGIN (the channel whose chat to read),
      KENNING_TWITCH_BOT_LOGIN (the bot identity reading chat),
      KENNING_TWITCH_BOT_TOKEN_PATH (default ~/.kenning/twitch_bot.json),
      KENNING_TWITCH_BROADCASTER_TOKEN_PATH (default ~/.kenning/twitch.json),
      KENNING_TWITCH_SUBSCRIBE_REDEEMS ("1"/"0", default "0"),
      KENNING_TWITCH_SUBSCRIBE_RAIDS ("1"/"0", default "0"; channel.raid rides the
        SAME isolated broadcaster-token session as redeems),
      KENNING_TWITCH_HELIX_BASE (default https://api.twitch.tv/helix).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections import OrderedDict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Deque, Optional, Protocol, runtime_checkable
from urllib.parse import parse_qs, urlsplit

logger = logging.getLogger("kenning.twitch.read_sidecar")

# --------------------------------------------------------------------------- #
# Configuration (env-overridable; safe defaults)
# --------------------------------------------------------------------------- #
DEFAULT_PORT = int(os.environ.get("KENNING_TWITCH_READ_PORT", "8773"))  # 8775 = overlay; read = 8773
DEFAULT_BUFFER_MAX = max(1, int(os.environ.get("KENNING_TWITCH_READ_BUFFER_MAX", "2000")))
DEFAULT_TTL_SECONDS = float(os.environ.get("KENNING_TWITCH_READ_TTL_SECONDS", "900"))
DEFAULT_POLL_SECONDS = max(0.01, float(os.environ.get("KENNING_TWITCH_READ_POLL_SECONDS", "0.5")))


# --------------------------------------------------------------------------- #
# Source abstraction
# --------------------------------------------------------------------------- #
@runtime_checkable
class ChatSource(Protocol):
    """Pluggable chat source. ``poll()`` returns a (possibly empty) list of
    JSON-serializable event dicts ingested since the last call; it MUST NOT raise
    (the buffer's poll loop tolerates a raise, but a well-behaved source fails
    quiet and returns ``[]``). ``close()`` is optional best-effort teardown."""

    name: str

    def poll(self) -> list[dict]:  # pragma: no cover - structural protocol
        ...


class FakeSource:
    """In-memory chat source for tests and the flag-off no-op default.

    Events queued via :meth:`push` are returned by the next :meth:`poll`. With
    nothing queued, ``poll()`` returns ``[]`` so the sidecar serves an empty
    buffer (the documented flag-off behaviour). Thread-safe so a test can push
    from one thread while the poll loop drains from another.
    """

    name = "fake"

    def __init__(self, events: Optional[list[dict]] = None) -> None:
        self._pending: Deque[dict] = deque(events or [])
        self._lock = threading.Lock()
        self.closed = False

    def push(self, *events: dict) -> None:
        with self._lock:
            for ev in events:
                self._pending.append(dict(ev))

    def poll(self) -> list[dict]:
        with self._lock:
            out = list(self._pending)
            self._pending.clear()
        return out

    def close(self) -> None:
        self.closed = True


class EventSubChatSource:
    """The default real source: drives the receive-only EventSub WebSocket client,
    creates the live subscriptions over Helix, and maps each notification to a
    small event dict (a ``channel.chat.message`` -> a ``{"type":"chat",...}`` dict
    and, when enabled, a redemption-add -> a ``{"type":"redeem",...}`` dict).

    Lazily connects on the first :meth:`poll`; every failure path is fail-quiet
    and returns ``[]`` (a transient socket error must never raise into the poll
    loop). The EventSub transport is itself stdlib-only (``socket``/``ssl``) and
    the Helix client is stdlib ``urllib`` only -- no third-party ``websockets`` /
    ``requests`` -- so importing them keeps the anticheat posture (BR-P1).

    Subscription lifecycle (per session):
      1. connect the wss socket;
      2. on the first ``session_welcome`` of a session, capture ``session_id``,
         load the stored OAuth tokens, resolve the broadcaster + bot logins to
         numeric user ids over Helix, then create the ``channel.chat.message``
         subscription (and, when ``subscribe_redeems`` is set, the redemption-add
         subscription) -- ONCE per session;
      3. drain notifications each poll via the client's non-blocking
         ``recv_json_ready``;
      4. on ``session_reconnect`` dial the new url; on a closed/stale socket the
         client is nulled so the next poll reconnects and re-subscribes.

    When ``client_id`` is empty (no creds) the source connects but NEVER
    subscribes -- Twitch then sends only the welcome + keepalives and the buffer
    stays empty (the documented flag-off behaviour). The sidecar's full behaviour
    is covered offline via injected fakes (``connect_factory`` + ``helix_factory``)
    and via ``FakeSource``.
    """

    name = "eventsub"

    def __init__(
        self,
        url: Optional[str] = None,
        *,
        connect_factory: Optional[Callable[[str], Any]] = None,
        redeem_connect_factory: Optional[Callable[[str], Any]] = None,
        helix_factory: Optional[Callable[[], Any]] = None,
        client_id: Optional[str] = None,
        broadcaster_login: Optional[str] = None,
        bot_login: Optional[str] = None,
        bot_token_path: Optional[str] = None,
        broadcaster_token_path: Optional[str] = None,
        subscribe_redeems: Optional[bool] = None,
        subscribe_raids: Optional[bool] = None,
        helix_base: Optional[str] = None,
        recv_timeout: float = 0.25,
    ) -> None:
        self._url = url or os.environ.get(
            "KENNING_TWITCH_EVENTSUB_URL", "wss://eventsub.wss.twitch.tv/ws"
        )
        self._connect_factory = connect_factory
        self._redeem_connect_factory = redeem_connect_factory
        self._helix_factory = helix_factory
        self._recv_timeout = recv_timeout

        # Subscription config (env-overridable; explicit args win for tests).
        self._client_id = (
            client_id if client_id is not None else os.environ.get("KENNING_TWITCH_CLIENT_ID", "")
        )
        self._broadcaster_login = (
            broadcaster_login
            if broadcaster_login is not None
            else os.environ.get("KENNING_TWITCH_BROADCASTER_LOGIN", "")
        )
        self._bot_login = (
            bot_login if bot_login is not None else os.environ.get("KENNING_TWITCH_BOT_LOGIN", "")
        )
        self._bot_token_path = (
            bot_token_path
            if bot_token_path is not None
            else os.environ.get("KENNING_TWITCH_BOT_TOKEN_PATH", "~/.kenning/twitch_bot.json")
        )
        self._broadcaster_token_path = (
            broadcaster_token_path
            if broadcaster_token_path is not None
            else os.environ.get("KENNING_TWITCH_BROADCASTER_TOKEN_PATH", "~/.kenning/twitch.json")
        )
        if subscribe_redeems is None:
            subscribe_redeems = os.environ.get("KENNING_TWITCH_SUBSCRIBE_REDEEMS", "0") == "1"
        self._subscribe_redeems = bool(subscribe_redeems)
        if subscribe_raids is None:
            subscribe_raids = os.environ.get("KENNING_TWITCH_SUBSCRIBE_RAIDS", "0") == "1"
        self._subscribe_raids = bool(subscribe_raids)
        self._helix_base = (
            helix_base
            if helix_base is not None
            else os.environ.get("KENNING_TWITCH_HELIX_BASE", "https://api.twitch.tv/helix")
        )

        self._client: Any = None
        self._session: Any = None
        self._dedup: Any = None
        self._helix: Any = None
        # True once subscriptions are created for the CURRENT session; reset on
        # every (re)connect so a reconnect re-subscribes against the new session.
        self._subscribed = False
        self._lock = threading.Lock()

        # Redeems + raids run on a SEPARATE EventSub session. Twitch REJECTS a
        # websocket session whose subscriptions are created by DIFFERENT users (400
        # "subscriptions created by different users"): the chat sub uses the BOT
        # token, but the redeem AND raid subs both need the BROADCASTER token. So a
        # second, isolated connection carries BOTH the redeem and raid
        # subscriptions (same token -> one session is fine). Fully additive +
        # fail-quiet -- a fault on this connection never touches the chat path.
        self._redeem_client: Any = None
        self._redeem_session: Any = None
        self._redeem_subscribed = False
        # Raids ride the SAME isolated broadcaster-token session as redeems
        # (channel.raid also subscribes with the broadcaster token, so co-locating
        # avoids a third websocket AND avoids the cross-user 400). Tracked with its
        # own flag so each can subscribe independently per session.
        self._raid_subscribed = False
        self._redeem_url = self._url

    # ---- credentials ---------------------------------------------------- #
    def _subscribe_enabled(self) -> bool:
        """Subscribe only when a client id is configured (else serve empty)."""
        return bool(self._client_id)

    def _load_token(self, path: str) -> str:
        """Load a live OAuth access token, proactively refreshing if near expiry."""
        try:
            from kenning.twitch.auth import TokenStore, TwitchAuth
        except Exception as exc:  # noqa: BLE001
            logger.warning("twitch auth import failed: %s", exc)
            return ""
        store = TokenStore(path)
        client_id = os.environ.get("KENNING_TWITCH_CLIENT_ID", "").strip()
        if client_id and store.is_expired(margin_seconds=300.0):
            try:
                access = TwitchAuth(client_id, store).ensure_valid(margin_seconds=300.0)
                if access:
                    return access
            except Exception as exc:  # noqa: BLE001
                logger.warning("proactive token refresh skipped (%s)", type(exc).__name__)
        try:
            tokens = store.load()
        except Exception as exc:  # noqa: BLE001
            logger.warning("twitch token load failed path=%s: %s", path, type(exc).__name__)
            return ""
        if not isinstance(tokens, dict):
            return ""
        access = tokens.get("access_token")
        return access if isinstance(access, str) else ""

    def _ensure_helix(self) -> Any:
        """Build (and cache) the Helix client. ``None`` if it cannot be built."""
        if self._helix is not None:
            return self._helix
        if self._helix_factory is not None:
            try:
                self._helix = self._helix_factory()
            except Exception as exc:  # noqa: BLE001
                logger.warning("helix factory failed: %s", type(exc).__name__)
                return None
            return self._helix
        try:
            from kenning.twitch.clients.helix_eventsub import HelixEventSubClient
        except Exception as exc:  # noqa: BLE001
            logger.warning("helix client import failed: %s", exc)
            return None
        try:
            self._helix = HelixEventSubClient(self._client_id, base_url=self._helix_base)
        except Exception as exc:  # noqa: BLE001
            logger.warning("helix client construct failed: %s", type(exc).__name__)
            return None
        return self._helix

    # ---- connection ----------------------------------------------------- #
    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        try:
            from kenning.twitch.clients.eventsub import (
                DedupLRU,
                EventSubSession,
                RFC6455Client,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("eventsub transport unavailable: %s", exc)
            return False
        try:
            if self._connect_factory is not None:
                self._client = self._connect_factory(self._url)
            else:
                client = RFC6455Client(timeout=30.0)
                client.connect(self._url)
                self._client = client
            self._session = EventSubSession()
            self._dedup = DedupLRU()
            self._subscribed = False  # a fresh socket has a fresh session
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("eventsub connect failed: %s", exc)
            self._reset_client()
            return False

    def _reset_client(self) -> None:
        """Drop the current client/session so the next poll reconnects + re-subscribes."""
        try:
            if self._client is not None and hasattr(self._client, "close"):
                self._client.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("eventsub client close (reset) failed: %s", exc)
        self._client = None
        self._session = None
        self._subscribed = False

    def _reconnect(self, url: str) -> bool:
        """Dial ``url`` (a session_reconnect target) on a fresh client."""
        self._reset_client()
        if url:
            self._url = url
        return self._ensure_client()

    # ---- subscription bootstrap ---------------------------------------- #
    def _subscribe(self, session_id: str) -> None:
        """Create the CHAT subscription for ``session_id`` (BOT token).

        ONCE per session. Resolves logins -> ids over Helix using the stored
        tokens. Fail-quiet: a Helix hiccup logs + leaves ``_subscribed`` False so
        the next poll retries (we never raise into the poll loop). Redeems are NOT
        created here -- they live on a SEPARATE session/connection (see
        :meth:`_subscribe_redeem_only`) because the redeem sub needs the
        BROADCASTER token and Twitch forbids two users' subs on one session.
        """
        if self._subscribed or not session_id or not self._subscribe_enabled():
            return
        helix = self._ensure_helix()
        if helix is None:
            return
        bot_token = self._load_token(self._bot_token_path)
        if not bot_token:
            logger.warning("eventsub subscribe skipped: no bot access token")
            return
        broadcaster_id = helix.get_user_id(self._broadcaster_login, token=bot_token)
        bot_id = helix.get_user_id(self._bot_login, token=bot_token)
        if not broadcaster_id or not bot_id:
            logger.warning(
                "eventsub subscribe skipped: unresolved ids broadcaster=%s bot=%s",
                bool(broadcaster_id), bool(bot_id),
            )
            return
        ok = helix.create_chat_subscription(
            broadcaster_id=broadcaster_id,
            bot_user_id=bot_id,
            session_id=session_id,
            token=bot_token,
        )
        if not ok:
            logger.warning("eventsub chat subscription create failed; will retry next poll")
            return
        # Mark subscribed only after the (required) chat subscription succeeded.
        self._subscribed = True
        logger.info("eventsub chat subscription established for session=%s", session_id)

    # ---- redeem connection (SEPARATE session, broadcaster token) -------- #
    def _ensure_redeem_client(self) -> bool:
        """Connect the SECOND websocket (redeems-only). Mirrors :meth:`_ensure_client`
        but is fully isolated: a failure here never affects the chat connection."""
        if self._redeem_client is not None:
            return True
        try:
            from kenning.twitch.clients.eventsub import EventSubSession, RFC6455Client
        except Exception as exc:  # noqa: BLE001
            logger.warning("eventsub redeem transport unavailable: %s", exc)
            return False
        try:
            if self._redeem_connect_factory is not None:
                self._redeem_client = self._redeem_connect_factory(self._redeem_url)
            else:
                client = RFC6455Client(timeout=30.0)
                client.connect(self._redeem_url)
                self._redeem_client = client
            self._redeem_session = EventSubSession()
            self._redeem_subscribed = False
            self._raid_subscribed = False
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("eventsub redeem connect failed: %s", exc)
            self._reset_redeem_client()
            return False

    def _reset_redeem_client(self) -> None:
        try:
            if self._redeem_client is not None and hasattr(self._redeem_client, "close"):
                self._redeem_client.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("eventsub redeem client close failed: %s", exc)
        self._redeem_client = None
        self._redeem_session = None
        self._redeem_subscribed = False
        self._raid_subscribed = False

    def _subscribe_redeem_only(self, session_id: str) -> None:
        """Create ONLY the channel-points redeem subscription for ``session_id``,
        with the BROADCASTER token (its own session -> no cross-user 400)."""
        if self._redeem_subscribed or not session_id or not self._subscribe_enabled():
            return
        helix = self._ensure_helix()
        if helix is None:
            return
        broadcaster_token = self._load_token(self._broadcaster_token_path)
        if not broadcaster_token:
            logger.warning("eventsub redeem subscribe skipped: no broadcaster access token")
            return
        broadcaster_id = helix.get_user_id(self._broadcaster_login, token=broadcaster_token)
        if not broadcaster_id:
            logger.warning("eventsub redeem subscribe skipped: unresolved broadcaster id")
            return
        ok = helix.create_redeem_subscription(
            broadcaster_id=broadcaster_id,
            session_id=session_id,
            token=broadcaster_token,
        )
        if ok:
            self._redeem_subscribed = True
            logger.info("eventsub redeem subscription established for session=%s", session_id)
        else:
            logger.warning("eventsub redeem subscription create failed; will retry next poll")

    def _subscribe_raid_only(self, session_id: str) -> None:
        """Create ONLY the ``channel.raid`` subscription for ``session_id``, with
        the BROADCASTER token (rides the SAME isolated session as the redeem sub --
        both use the broadcaster token so there is no cross-user 400). The condition
        binds the 'to' side (the channel being raided), which needs no special scope."""
        if self._raid_subscribed or not session_id or not self._subscribe_enabled():
            return
        helix = self._ensure_helix()
        if helix is None:
            return
        # Some injected/fakes implement only the chat/redeem subs; degrade gracefully.
        create = getattr(helix, "create_raid_subscription", None)
        if not callable(create):
            logger.debug("eventsub raid subscribe skipped: helix has no create_raid_subscription")
            return
        broadcaster_token = self._load_token(self._broadcaster_token_path)
        if not broadcaster_token:
            logger.warning("eventsub raid subscribe skipped: no broadcaster access token")
            return
        broadcaster_id = helix.get_user_id(self._broadcaster_login, token=broadcaster_token)
        if not broadcaster_id:
            logger.warning("eventsub raid subscribe skipped: unresolved broadcaster id")
            return
        ok = create(
            broadcaster_id=broadcaster_id,
            session_id=session_id,
            token=broadcaster_token,
        )
        if ok:
            self._raid_subscribed = True
            logger.info("eventsub raid subscription established for session=%s", session_id)
        else:
            logger.warning("eventsub raid subscription create failed; will retry next poll")

    def _subscribe_broadcaster_session(self, session_id: str) -> None:
        """Create whichever broadcaster-token subscriptions are enabled (redeems
        and/or raids) for ``session_id``. Both ride one isolated session."""
        if self._subscribe_redeems:
            self._subscribe_redeem_only(session_id)
        if self._subscribe_raids:
            self._subscribe_raid_only(session_id)

    def _poll_redeems(self, out: list[dict]) -> None:
        """Drain the BROADCASTER-token connection: connect, subscribe on welcome,
        and map any redemption / raid notifications into ``out``. Fully fail-quiet +
        isolated. Named ``_poll_redeems`` for history; it now also carries raids."""
        if not (self._subscribe_redeems or self._subscribe_raids) or not self._subscribe_enabled():
            return
        if not self._ensure_redeem_client():
            return
        try:
            from kenning.twitch.clients.eventsub import WebSocketClosed
        except Exception:  # noqa: BLE001
            return
        try:
            for _ in range(64):
                msg = self._redeem_client.recv_json_ready(self._recv_timeout)
                if msg is None:
                    break
                if not isinstance(msg, dict) or self._redeem_session is None:
                    continue
                try:
                    self._redeem_session.note_keepalive()
                except Exception:  # noqa: BLE001
                    pass
                cls = self._redeem_session.classify_message(msg)
                if cls == "welcome":
                    sid = self._redeem_session.parse_welcome(msg)
                    if sid:
                        self._subscribe_broadcaster_session(sid)
                elif cls == "reconnect":
                    new_url = self._redeem_session.handle_reconnect(msg)
                    self._reset_redeem_client()
                    if new_url:
                        self._redeem_url = new_url
                    break
                elif cls == "revocation":
                    logger.warning("eventsub redeem subscription revoked; reconnecting")
                    self._reset_redeem_client()
                    break
                elif cls == "notification":
                    self._map_broadcaster_notification(msg, out)
        except WebSocketClosed as exc:
            logger.info("eventsub redeem socket closed (%s); will reconnect", exc)
            self._reset_redeem_client()
        except Exception as exc:  # noqa: BLE001
            logger.warning("eventsub redeem poll error: %s", exc)
            self._reset_redeem_client()

    # ---- poll ----------------------------------------------------------- #
    def poll(self) -> list[dict]:
        with self._lock:
            out: list[dict] = []
            try:
                from kenning.twitch.clients.eventsub import (
                    ChatEvent,
                    WebSocketClosed,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("eventsub ChatEvent import failed: %s", exc)
                return out
            # --- CHAT connection (bot token) ---
            if self._ensure_client():
                try:
                    for _ in range(64):  # bounded per-poll so we never spin forever
                        msg = self._client.recv_json_ready(self._recv_timeout)
                        if msg is None:
                            break  # no more data ready this cycle
                        if not isinstance(msg, dict):
                            continue
                        handled = self._handle_message(msg, ChatEvent, out)
                        if handled == "reconnect":
                            break  # dialed a new url; resume draining next poll
                except WebSocketClosed as exc:
                    # Clean/abnormal close or stale socket -> reconnect + re-subscribe.
                    logger.info("eventsub socket closed (%s); will reconnect", exc)
                    self._reset_client()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("eventsub poll error: %s", exc)
                    self._reset_client()  # force a reconnect + re-subscribe next poll
            # --- REDEEM connection (broadcaster token, SEPARATE session) ---
            # Independent of chat: a fault on either side never affects the other.
            try:
                self._poll_redeems(out)
            except Exception as exc:  # noqa: BLE001 — redeems never break the chat path
                logger.warning("eventsub redeem drain error: %s", exc)
            return out

    def _handle_message(self, msg: dict, chat_event_cls: Any, out: list[dict]) -> str:
        """Classify + map one EventSub message; append any event dicts to ``out``.

        Returns ``"reconnect"`` if a ``session_reconnect`` was handled (the caller
        stops draining this cycle), else ``"ok"``. Never raises.
        """
        if self._session is None:
            return "ok"
        # Any received message is liveness for the staleness clock.
        try:
            self._session.note_keepalive()
        except Exception as exc:  # noqa: BLE001
            logger.debug("eventsub note_keepalive failed: %s", exc)
        cls = self._session.classify_message(msg)
        if cls == "welcome":
            sid = self._session.parse_welcome(msg)
            if sid:
                self._subscribe(sid)
            return "ok"
        if cls == "reconnect":
            new_url = self._session.handle_reconnect(msg)
            if new_url:
                self._reconnect(new_url)
            return "reconnect"
        if cls == "revocation":
            logger.warning("eventsub subscription revoked; dropping session to re-subscribe")
            self._reset_client()
            return "reconnect"
        if cls != "notification":
            return "ok"  # keepalive / unknown -> nothing to emit
        self._map_notification(msg, chat_event_cls, out)
        return "ok"

    def _map_notification(self, msg: dict, chat_event_cls: Any, out: list[dict]) -> None:
        """Map a notification to a chat or redeem event dict (fail-quiet)."""
        sub_type = ""
        meta = msg.get("metadata")
        if isinstance(meta, dict):
            st = meta.get("subscription_type")
            if isinstance(st, str):
                sub_type = st

        # channel.chat.message -> {"type":"chat",...} (the unchanged consumer shape).
        if sub_type == "" or sub_type == "channel.chat.message":
            ev = chat_event_cls.from_eventsub(msg)
            if ev is not None:
                if self._dedup is not None and self._dedup.seen(ev.message_id):
                    return
                out.append(
                    {
                        "type": "chat",
                        "message_id": ev.message_id,
                        "chatter_login": ev.chatter_login,
                        "chatter_name": ev.chatter_name,
                        "chatter_user_id": ev.chatter_user_id,
                        "text": ev.text,
                        # badges carry mod/broadcaster provenance for the chat-command
                        # authz (commands.parse_command -> is_mod); a list of badge dicts.
                        "badges": ev.badges,
                    }
                )
                return
            # Not a chat message under an unknown sub_type -> fall through to redeem.

        if sub_type == "channel.channel_points_custom_reward_redemption.add":
            self._map_redeem(msg, out)
            return

        if sub_type == "channel.raid":
            self._map_raid(msg, out)
            return

        # Unknown/unhandled subscription type -> ignore (fail-safe).
        if sub_type:
            logger.debug("eventsub notification ignored sub_type=%s", sub_type)

    def _map_broadcaster_notification(self, msg: dict, out: list[dict]) -> None:
        """Dispatch a notification on the BROADCASTER-token session (redeems +
        raids ride one session) by its subscription type. Fail-safe."""
        sub_type = ""
        meta = msg.get("metadata")
        if isinstance(meta, dict):
            st = meta.get("subscription_type")
            if isinstance(st, str):
                sub_type = st
        if sub_type == "channel.raid":
            self._map_raid(msg, out)
            return
        # Default (incl. the redemption-add type and an empty/legacy sub_type) ->
        # redeem mapping, preserving the prior behaviour for the redeem session.
        self._map_redeem(msg, out)

    def _map_redeem(self, msg: dict, out: list[dict]) -> None:
        """Map a redemption-add notification to a ``{"type":"redeem",...}`` dict.

        Parses defensively from ``payload.event`` (the redemption-add shape). Dedup
        is keyed on the redemption id (its own LRU, independent of the chat dedup).
        """
        event = self._locate_event(msg)
        if not isinstance(event, dict):
            return
        redemption_id = self._coerce_str(event.get("id"))
        if redemption_id and self._redeem_seen(redemption_id):
            return
        reward = event.get("reward")
        if not isinstance(reward, dict):
            reward = {}
        out.append(
            {
                "type": "redeem",
                "redemption_id": redemption_id,
                "reward_id": self._coerce_str(reward.get("id")),
                "reward_title": self._coerce_str(reward.get("title")),
                "user_input": self._coerce_str(event.get("user_input")),
                "chatter_login": self._coerce_str(event.get("user_login")),
                "chatter_name": self._coerce_str(event.get("user_name")),
                "chatter_user_id": self._coerce_str(event.get("user_id")),
                "status": self._coerce_str(event.get("status")),
            }
        )

    def _map_raid(self, msg: dict, out: list[dict]) -> None:
        """Map a ``channel.raid`` notification to a ``{"type":"raid",...}`` dict.

        Parses defensively from ``payload.event`` (the channel.raid shape:
        ``from_broadcaster_user_{id,login,name}`` + ``viewers``). Dedup is keyed on
        a synthetic id (the raider id + viewer count) so an EventSub replay of the
        same raid never re-fires; raids carry no native id. Independent LRU."""
        event = self._locate_event(msg)
        if not isinstance(event, dict):
            return
        from_id = self._coerce_str(event.get("from_broadcaster_user_id"))
        from_login = self._coerce_str(event.get("from_broadcaster_user_login"))
        from_name = self._coerce_str(event.get("from_broadcaster_user_name"))
        try:
            viewers = int(event.get("viewers") or 0)
        except (TypeError, ValueError):
            viewers = 0
        # channel.raid has no native id -> synthesize a stable dedup key from the
        # raider + viewer count (a single raid notifies once; a transport replay
        # repeats the same triple).
        dedup_key = f"{from_id}:{from_login}:{viewers}"
        if self._raid_seen(dedup_key):
            return
        out.append(
            {
                "type": "raid",
                "from_login": from_login,
                "from_name": from_name,
                "from_broadcaster_user_id": from_id,
                "viewers": viewers,
            }
        )

    # ---- raid dedup (lazily created; independent of the chat/redeem dedup) -- #
    def _raid_seen(self, dedup_key: str) -> bool:
        dedup = getattr(self, "_raid_dedup", None)
        if dedup is None:
            try:
                from kenning.twitch.clients.eventsub import DedupLRU

                dedup = DedupLRU()
            except Exception as exc:  # noqa: BLE001
                logger.debug("raid dedup unavailable: %s", exc)
                return False
            self._raid_dedup = dedup
        return bool(dedup.seen(dedup_key))

    # ---- redeem dedup (lazily created; independent of the chat dedup) --- #
    def _redeem_seen(self, redemption_id: str) -> bool:
        dedup = getattr(self, "_redeem_dedup", None)
        if dedup is None:
            try:
                from kenning.twitch.clients.eventsub import DedupLRU

                dedup = DedupLRU()
            except Exception as exc:  # noqa: BLE001
                logger.debug("redeem dedup unavailable: %s", exc)
                return False
            self._redeem_dedup = dedup
        return bool(dedup.seen(redemption_id))

    @staticmethod
    def _locate_event(msg: dict) -> Optional[dict]:
        """Pull the ``event`` dict out of a full envelope or a bare payload."""
        payload = msg.get("payload")
        if isinstance(payload, dict):
            ev = payload.get("event")
            if isinstance(ev, dict):
                return ev
        ev = msg.get("event")
        return ev if isinstance(ev, dict) else None

    @staticmethod
    def _coerce_str(value: Any) -> str:
        if value is None:
            return ""
        return value if isinstance(value, str) else str(value)

    def close(self) -> None:
        try:
            if self._client is not None and hasattr(self._client, "close"):
                self._client.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("eventsub source close failed: %s", exc)
        self._reset_redeem_client()


# --------------------------------------------------------------------------- #
# Rolling buffer
# --------------------------------------------------------------------------- #
class RollingBuffer:
    """Thread-safe rolling buffer of chat events with a ``maxlen`` AND a TTL cap.

    Each appended event is wrapped with a monotonically increasing integer
    ``seq`` (the cursor space) and an ingest ``ts``. Eviction is oldest-first by
    the deque ``maxlen``; on every read/append, events older than ``ttl_seconds``
    are also pruned. ``drained_total`` and ``dropped_total`` are kept for
    observability. All public methods take the lock, so concurrent appends (poll
    loop) and reads (HTTP handler threads) are safe.
    """

    def __init__(self, maxlen: int = DEFAULT_BUFFER_MAX, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        if maxlen < 1:
            raise ValueError("RollingBuffer maxlen must be >= 1")
        self._maxlen = maxlen
        self._ttl = max(0.0, ttl_seconds)
        self._events: Deque[dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._seq = 0          # last assigned seq (cursor high-water mark)
        self._cursor = 0       # consumer-acked cursor
        self.dropped_total = 0  # events evicted before being acked (maxlen/TTL)
        self.appended_total = 0
        # login_lower -> last chat message_id, for cross-process voice DELETE
        # moderation (the write sidecar resolves a target's last message via
        # GET /last_message). Bounded LRU, independent of the rolling buffer's TTL
        # so a delete still works after the message scrolled out of /buffer.
        self._last_msg: "OrderedDict[str, str]" = OrderedDict()
        self._last_msg_max = 4096

    # -- mutation -------------------------------------------------------- #
    def append(self, event: dict, *, now: Optional[float] = None) -> int:
        """Wrap ``event`` with a fresh ``seq``/``ts`` and append it. Returns the
        assigned seq. Evicts the oldest entry when ``maxlen`` is exceeded (deque
        does this for us; we count the drop for observability)."""
        now = time.time() if now is None else now
        with self._lock:
            self._seq += 1
            wrapped = {"seq": self._seq, "ts": now, "event": event}
            at_cap = len(self._events) >= self._maxlen
            if at_cap:
                # The about-to-be-evicted head was never acked -> a real drop.
                self.dropped_total += 1
            self._events.append(wrapped)
            self.appended_total += 1
            self._index_last_message(event)
            self._prune_ttl(now)
            return self._seq

    def _index_last_message(self, event: dict) -> None:
        """Record a chat event's ``{login -> message_id}`` for voice DELETE
        moderation (caller holds the lock). No-op for non-chat / id-less events."""
        if not isinstance(event, dict) or event.get("type") != "chat":
            return
        login = str(event.get("chatter_login") or "").strip().lower()
        mid = str(event.get("message_id") or "")
        if not login or not mid:
            return
        self._last_msg[login] = mid
        self._last_msg.move_to_end(login)
        while len(self._last_msg) > self._last_msg_max:
            self._last_msg.popitem(last=False)

    def last_message_id(self, login: str) -> Optional[str]:
        """The most-recent chat message_id seen for ``login`` (case-insensitive),
        or ``None``. Backs the GET /last_message route for cross-process delete."""
        key = (login or "").strip().lower()
        if not key:
            return None
        with self._lock:
            return self._last_msg.get(key)

    def _prune_ttl(self, now: float) -> None:
        """Drop events older than the TTL. Caller holds the lock."""
        if self._ttl <= 0:
            return
        cutoff = now - self._ttl
        while self._events and self._events[0]["ts"] < cutoff:
            evicted = self._events.popleft()
            # Only count it as a drop if the consumer never acked past it.
            if evicted["seq"] > self._cursor:
                self.dropped_total += 1

    # -- read ------------------------------------------------------------ #
    def drain(self, since: Optional[int] = None, *, now: Optional[float] = None) -> tuple[list[dict], int]:
        """Return (events, cursor): all buffered events whose ``seq`` > ``since``.

        ``since`` defaults to the consumer cursor. The returned ``cursor`` is the
        seq high-water mark (the largest seq returned, or ``since`` if nothing is
        newer) -- the consumer ACKs that value to advance. Does NOT mutate the
        consumer cursor (that is :meth:`ack`'s job), so a crash mid-consume safely
        redelivers (at-least-once). Prunes TTL-expired events first."""
        now = time.time() if now is None else now
        with self._lock:
            self._prune_ttl(now)
            floor = self._cursor if since is None else since
            out = [dict(w) for w in self._events if w["seq"] > floor]
            cursor = out[-1]["seq"] if out else floor
            return out, cursor

    def ack(self, cursor: int) -> int:
        """Advance the consumer cursor to ``cursor`` (monotonic; never regresses)
        and prune acked events from the front of the buffer. Returns the effective
        cursor."""
        with self._lock:
            if cursor > self._cursor:
                self._cursor = cursor
            while self._events and self._events[0]["seq"] <= self._cursor:
                self._events.popleft()
            return self._cursor

    # -- introspection --------------------------------------------------- #
    @property
    def cursor(self) -> int:
        with self._lock:
            return self._cursor

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def stats(self, *, now: Optional[float] = None) -> dict:
        now = time.time() if now is None else now
        with self._lock:
            self._prune_ttl(now)
            return {
                "buffered": len(self._events),
                "cursor": self._cursor,
                "seq": self._seq,
                "dropped": self.dropped_total,
                "appended": self.appended_total,
                "maxlen": self._maxlen,
                "ttl_seconds": self._ttl,
            }


# --------------------------------------------------------------------------- #
# Poll loop
# --------------------------------------------------------------------------- #
class PollLoop:
    """Background thread that pumps ``source.poll()`` into the buffer on a fixed
    cadence. Fail-quiet: a source that raises is logged and retried next tick
    (never crashes the sidecar). ``run_once`` is exposed so tests can drive a
    single deterministic pump without a thread."""

    def __init__(
        self,
        source: ChatSource,
        buffer: RollingBuffer,
        *,
        interval: float = DEFAULT_POLL_SECONDS,
    ) -> None:
        self._source = source
        self._buffer = buffer
        self._interval = max(0.01, interval)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def run_once(self) -> int:
        """Pump one poll() into the buffer. Returns the count appended."""
        try:
            events = self._source.poll()
        except Exception as exc:  # noqa: BLE001 — never let a source raise out
            logger.warning("chat source poll raised: %s", exc)
            return 0
        if not events:
            return 0
        count = 0
        for ev in events:
            if not isinstance(ev, dict):
                logger.debug("dropping non-dict source event: %r", type(ev))
                continue
            self._buffer.append(ev)
            count += 1
        return count

    def _run(self) -> None:
        logger.info("chat read poll loop started interval=%.3fs source=%s",
                    self._interval, getattr(self._source, "name", "?"))
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(self._interval)
        logger.info("chat read poll loop stopped")

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="twitch-read-poll")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            close = getattr(self._source, "close", None)
            if callable(close):
                close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("source close on stop failed: %s", exc)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
def make_handler(buffer: RollingBuffer, poll_loop: Optional[PollLoop], source_name: str):
    """Build a ``BaseHTTPRequestHandler`` subclass bound to this sidecar's state.

    A factory (not module globals) so a test can stand up an isolated server with
    its own buffer/source on an ephemeral port without cross-test state."""

    class _Handler(BaseHTTPRequestHandler):
        server_version = "KenningTwitchRead/1.0"

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode("utf-8")
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionError) as exc:  # client hung up
                logger.debug("client disconnected mid-response: %s", exc)

        # -- GET ---------------------------------------------------------- #
        def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            parts = urlsplit(self.path)
            path = parts.path
            if path == "/healthz":
                stats = buffer.stats()
                running = bool(poll_loop.running) if poll_loop is not None else False
                self._send(200, {
                    "ok": True,
                    "buffered": stats["buffered"],
                    "cursor": stats["cursor"],
                    "dropped": stats["dropped"],
                    "running": running,
                    "source": source_name,
                })
                return
            if path == "/buffer":
                since = self._parse_since(parts.query)
                events, cursor = buffer.drain(since=since)
                self._send(200, {"events": events, "cursor": cursor})
                return
            if path == "/last_message":
                # Voice DELETE moderation: the write sidecar resolves a target's
                # last chat message_id here. Fail-safe -> {"message_id": null}.
                login = ""
                mid = None
                try:
                    login = parse_qs(parts.query or "").get("login", [""])[0] or ""
                    mid = buffer.last_message_id(login)
                except Exception:  # noqa: BLE001 - never raise on a hostile query
                    mid = None
                self._send(200, {"login": login, "message_id": mid})
                return
            self._send(404, {"error": "not found"})

        @staticmethod
        def _parse_since(query: str) -> Optional[int]:
            """Parse ?since=N. Absent/garbage -> None (use the consumer cursor).
            Fail-safe: never raises on a hostile query string."""
            try:
                qs = parse_qs(query or "")
                raw = qs.get("since", [None])[0]
                if raw is None or raw == "":
                    return None
                return max(0, int(raw))
            except (ValueError, TypeError):
                return None

        # -- POST --------------------------------------------------------- #
        def do_POST(self) -> None:  # noqa: N802
            parts = urlsplit(self.path)
            if parts.path != "/ack":
                self._send(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", "0") or "0")
                if n < 0 or n > (1 << 20):
                    self._send(400, {"error": "bad content-length"})
                    return
                payload = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, TypeError) as exc:
                self._send(400, {"error": f"bad request: {exc}"})
                return
            if not isinstance(payload, dict):
                self._send(400, {"error": "body must be a JSON object"})
                return
            try:
                cursor = int(payload.get("cursor", 0))
            except (ValueError, TypeError):
                self._send(400, {"error": "cursor must be an integer"})
                return
            effective = buffer.ack(cursor)
            self._send(200, {"ok": True, "cursor": effective})

        def log_message(self, *args: Any) -> None:  # noqa: ARG002 — fail-quiet
            return

    return _Handler


# --------------------------------------------------------------------------- #
# Parent-death deadman (clone of the embedder_server precedent)
# --------------------------------------------------------------------------- #
def _pid_alive(pid: int) -> bool:
    """True iff process ``pid`` is still running. psutil if present, else a
    ctypes OpenProcess+GetExitCodeProcess check on Windows / os.kill(0) on POSIX.
    Fail-SAFE: an indeterminate result returns True (never self-kill on doubt)."""
    if pid <= 0:
        return True
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:  # noqa: BLE001
        pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            k = ctypes.windll.kernel32
            h = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not h:
                return False  # cannot open -> gone
            code = wintypes.DWORD()
            ok = k.GetExitCodeProcess(h, ctypes.byref(code))
            k.CloseHandle(h)
            return (not ok) or code.value == STILL_ACTIVE
        except Exception:  # noqa: BLE001
            return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except Exception:  # noqa: BLE001
        return True


def parent_watchdog_check(pid: int) -> str:
    """Single-shot watchdog decision for ``pid``. Returns ``"alive"`` if the
    parent is still running (or the pid is unset/invalid -> do not self-kill) and
    ``"dead"`` if the parent is gone (the caller should ``os._exit``). Split out
    from the loop so it is unit-testable without spawning a process."""
    if pid <= 0:
        return "alive"
    return "alive" if _pid_alive(pid) else "dead"


def _parent_watchdog(poll_seconds: float = 3.0) -> None:
    """Self-exit when the parent (Ultron orchestrator) dies, so a force-killed or
    crashed parent NEVER leaves this sidecar as a runaway orphan holding a live
    Twitch socket. Parent pid via ``KENNING_TWITCH_PARENT_PID`` (fallback: the
    spawn-time parent). ``os._exit`` skips atexit/locks so the socket is freed
    immediately by the OS."""
    try:
        ppid = int(os.environ.get("KENNING_TWITCH_PARENT_PID", "0") or "0")
    except Exception:  # noqa: BLE001
        ppid = 0
    if ppid <= 0:
        ppid = os.getppid()
    if ppid <= 0:
        return
    sys.stderr.write(f"[twitch-read] parent-watchdog armed on pid {ppid}\n")
    sys.stderr.flush()
    while True:
        time.sleep(poll_seconds)
        if parent_watchdog_check(ppid) == "dead":
            sys.stderr.write(
                f"[twitch-read] parent pid {ppid} gone -> self-terminating "
                "(orphan guard)\n")
            sys.stderr.flush()
            os._exit(0)


# --------------------------------------------------------------------------- #
# Server assembly
# --------------------------------------------------------------------------- #
def build_server(
    source: Optional[ChatSource] = None,
    *,
    port: int = 0,
    buffer_max: int = DEFAULT_BUFFER_MAX,
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    poll_interval: float = DEFAULT_POLL_SECONDS,
    start_poll: bool = True,
) -> tuple[ThreadingHTTPServer, RollingBuffer, PollLoop]:
    """Assemble (server, buffer, poll_loop) bound to 127.0.0.1 ONLY.

    ``port=0`` binds an ephemeral port (read it back from ``server.server_address``
    in a test). With no ``source`` an empty :class:`FakeSource` is used, so a
    bare run serves an empty buffer (the documented flag-off behaviour). The
    server is returned NOT yet serving -- the caller runs ``serve_forever`` (or, in
    a test, drives requests against the bound address on a thread)."""
    src = source if source is not None else FakeSource()
    buffer = RollingBuffer(maxlen=buffer_max, ttl_seconds=ttl_seconds)
    poll_loop = PollLoop(src, buffer, interval=poll_interval)
    handler = make_handler(buffer, poll_loop, getattr(src, "name", "fake"))
    # EXCLUSIVE bind (anti-stale-sidecar): a second instance on a live port FAILS
    # rather than co-serving. port=0 (tests) binds a fresh ephemeral port.
    from kenning.subprocess.sidecar_server import SingletonThreadingHTTPServer
    server = SingletonThreadingHTTPServer(("127.0.0.1", port), handler)
    if start_poll:
        poll_loop.start()
    return server, buffer, poll_loop


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [twitch-read] %(levelname)s %(message)s",
    )
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    # Anti-stale-sidecar guard: reap same-role strays + reclaim the port BEFORE
    # binding (which build_server does EXCLUSIVELY) -> exactly one live instance.
    import atexit

    from kenning.subprocess import sidecar_lock
    sidecar_lock.guard_singleton("127.0.0.1", port, "twitch_read")
    # The real source connects to EventSub; if its transport/creds are absent it
    # fails quiet and the sidecar serves an empty buffer (harmless on its own).
    source: ChatSource = EventSubChatSource()
    server, _buffer, poll_loop = build_server(source, port=port)
    sidecar_lock.write_role("twitch_read", os.getpid(), port)
    atexit.register(sidecar_lock.clear_role, "twitch_read")
    # Parent-death deadman: the strongest orphan guard -- the child cleans itself
    # up on ANY parent death (crash, taskkill /F, TerminateProcess).
    threading.Thread(target=_parent_watchdog, daemon=True,
                     name="twitch-read-parent-watchdog").start()
    host, bound_port = server.server_address[:2]
    logger.info("twitch read sidecar serving on http://%s:%s", host, bound_port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        poll_loop.stop()
        server.server_close()
        sidecar_lock.clear_role("twitch_read")


if __name__ == "__main__":
    main()
