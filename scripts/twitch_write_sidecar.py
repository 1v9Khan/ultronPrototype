"""S11 — Twitch WRITE / Helix moderation sidecar (loopback HTTP on 127.0.0.1:8777).

Runs as a SEPARATE process so NO Twitch network transport ever loads into
Ultron's anticheat-pinned main/voice process (BR-P1). The in-process
:class:`~kenning.twitch.moderation.service.ModerationService` (+ its
:class:`HelixClient` and :class:`ModerationGuard`) live HERE, behind a tiny
loopback JSON HTTP surface; the orchestrator keeps only a thin ``urllib`` client
(:class:`kenning.twitch.moderation.remote.ModerationRemote`). Mirrors the proven
``scripts/twitch_read_sidecar.py`` precedent: a loopback-ONLY exclusive-bind
``SingletonThreadingHTTPServer`` (127.0.0.1), the ``sidecar_lock`` singleton
guard + role pidfile, a parent-death deadman thread (``os._exit`` when the parent
pid is gone), and fail-quiet logging.

ANTICHEAT POSTURE: pure logic + loopback networking only. Moderation is pure
regex/keyword parsing + a server-authoritative resolve/authorize gate + a single
``urllib`` Helix write — NO model / GGUF is loaded (the abliterated LLM is never
in the moderation decision path). Imports are stdlib only
(``http.server``/``socket``/``json``/``urllib``/``threading``/``secrets`` + the
``kenning.twitch.*`` / ``kenning.subprocess.*`` libraries which are themselves
``urllib``-only). NO ``requests`` / ``aiohttp`` / ``websockets`` /
``transformers`` / ``torch`` / desktop-input/screen libs. Binds 127.0.0.1 ONLY.

Fail-safe startup
-----------------
Every credential / id-resolution step is best-effort. If creds are absent (no
token, no client id, unresolved logins) the sidecar still SERVES — ``/healthz``
reports ``ready=false`` and every action route refuses cleanly. It NEVER crashes
on missing creds, so a bare run (the documented master-flag-OFF behaviour) is
harmless on its own.

Protocol (JSON over loopback HTTP)
----------------------------------
  GET  /healthz                 -> {"ok":true, "ready":bool, "broadcaster_id":ID}
  POST /prepare  {"text":...}    -> parse+resolve+authorize -> a confirmable token
                                    {"ok":bool, "token":TOK, "readback":...,
                                     "reason_blocked":..., "candidates":[...],
                                     "action":..., "target":...}
                                    or {"ok":false, "not_a_command":true} when the
                                    text is not a moderation command.
  POST /confirm  {"token":...}   -> look up the prepared proposal + execute the
                                    single Helix write -> the confirm() result dict
                                    ({"ok":..., "action":..., "target":..., ...}),
                                    or {"ok":false, "error":"expired"}.
  POST /cancel   {"token":...}   -> drop the pending proposal -> {"ok":true}.
  POST /shoutout {"to_broadcaster_id":ID} -> Helix POST /chat/shoutouts (the raid
                                    handler promotes a raider). Fail-open: always
                                    200; {"ok":false,...} when unavailable/error so
                                    the raid VOCAL announce is never blocked.

A ``/prepare`` mints a fresh ``secrets.token_urlsafe(16)`` token, stores the
proposal in a bounded TTL map (cap 64, ~120 s), and returns it; ``/confirm`` /
``/cancel`` consume it. A misheard "ban" therefore never fires without the
streamer's explicit second-phase confirm.

Run:  python scripts/twitch_write_sidecar.py [PORT]
Env:  KENNING_TWITCH_WRITE_PORT (default 8777),
      KENNING_TWITCH_PARENT_PID (parent pid for the deadman watchdog),
      KENNING_TWITCH_CLIENT_ID (the Twitch app client id),
      KENNING_TWITCH_BROADCASTER_LOGIN (the channel owner login),
      KENNING_TWITCH_BOT_LOGIN (the bot identity login),
      KENNING_TWITCH_BROADCASTER_TOKEN_PATH (default ~/.kenning/twitch.json;
        the broadcaster's write-scope token — moderator:manage:banned_users),
      KENNING_TWITCH_READ_ENDPOINT (default http://127.0.0.1:8773; the read
        sidecar whose /buffer feeds the chatter roster),
      KENNING_TWITCH_HELIX_BASE (default https://api.twitch.tv/helix),
      KENNING_TWITCH_MOD_REQUIRE_CONFIRM ("1"/"0", default "1").
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

logger = logging.getLogger("kenning.twitch.write_sidecar")

# --------------------------------------------------------------------------- #
# Configuration (env-overridable; safe defaults)
# --------------------------------------------------------------------------- #
DEFAULT_PORT = int(os.environ.get("KENNING_TWITCH_WRITE_PORT", "8777"))
DEFAULT_READ_ENDPOINT = os.environ.get(
    "KENNING_TWITCH_READ_ENDPOINT", "http://127.0.0.1:8773"
)
DEFAULT_HELIX_BASE = os.environ.get(
    "KENNING_TWITCH_HELIX_BASE", "https://api.twitch.tv/helix"
)

# Bounded pending-proposal map: at most this many in flight, each living this long.
_PROPOSAL_CAP = 64
_PROPOSAL_TTL_S = 120.0
# Bounded roster cache: most-recent chatter logins -> user ids.
_ROSTER_CAP = 500
# Bound on an accepted request body (defense against a hostile/oversized POST).
_MAX_BODY_BYTES = 1 << 20  # 1 MiB
_ROSTER_FETCH_TIMEOUT_S = 2.0


# --------------------------------------------------------------------------- #
# Pending-proposal store (token -> proposal, bounded + TTL)
# --------------------------------------------------------------------------- #
class ProposalStore:
    """A thread-safe, bounded, TTL'd map of ``token -> (proposal, expires_at)``.

    A ``/prepare`` that produced an OK proposal mints a token here; ``/confirm``
    and ``/cancel`` pop it. The cap (oldest-first eviction) and TTL bound memory
    so a caller that prepares-but-never-confirms can't grow the sidecar without
    bound. All methods take the lock and never raise into the caller.
    """

    def __init__(
        self,
        *,
        cap: int = _PROPOSAL_CAP,
        ttl_s: float = _PROPOSAL_TTL_S,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cap = max(1, int(cap))
        self._ttl = max(1.0, float(ttl_s))
        self._monotonic = monotonic
        self._items: "dict[str, tuple[Any, float]]" = {}
        self._lock = threading.Lock()

    def put(self, proposal: Any) -> str:
        """Store ``proposal`` under a fresh unguessable token; return the token."""
        token = secrets.token_urlsafe(16)
        now = self._monotonic()
        with self._lock:
            self._prune_locked(now)
            # Evict the oldest while at capacity (insertion order == dict order).
            while len(self._items) >= self._cap:
                oldest = next(iter(self._items))
                self._items.pop(oldest, None)
            self._items[token] = (proposal, now + self._ttl)
        return token

    def pop(self, token: str) -> Optional[Any]:
        """Remove + return the proposal for ``token``, or ``None`` if missing/expired."""
        if not token or not isinstance(token, str):
            return None
        now = self._monotonic()
        with self._lock:
            self._prune_locked(now)
            entry = self._items.pop(token, None)
        if entry is None:
            return None
        proposal, expires_at = entry
        if expires_at < now:
            return None
        return proposal

    def discard(self, token: str) -> bool:
        """Drop ``token`` if present; return whether it existed."""
        if not token or not isinstance(token, str):
            return False
        with self._lock:
            return self._items.pop(token, None) is not None

    def _prune_locked(self, now: float) -> None:
        expired = [t for t, (_, exp) in self._items.items() if exp < now]
        for t in expired:
            self._items.pop(t, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


# --------------------------------------------------------------------------- #
# Roster provider (read sidecar /buffer -> {login: user_id})
# --------------------------------------------------------------------------- #
class RosterCache:
    """Accumulates a recent-chatter roster by GETting the read sidecar's
    ``/buffer?since=0`` (read-only — it never POSTs ``/ack``) and harvesting
    ``{chatter_login.lower(): chatter_user_id}`` from every ``type=="chat"``
    event. Bounded (cap, most-recent wins) and fail-safe: a read-sidecar that is
    down/garbage leaves the last-known roster intact (or ``{}`` if never seen).

    The read sidecar wraps each event as ``{"seq","ts","event":{...}}``; the inner
    event carries ``chatter_login`` / ``chatter_user_id`` for a chat message.
    """

    def __init__(
        self,
        read_endpoint: str,
        *,
        cap: int = _ROSTER_CAP,
        opener: Optional[Callable[[str], bytes]] = None,
    ) -> None:
        self._endpoint = (read_endpoint or "").rstrip("/")
        self._cap = max(1, int(cap))
        self._opener = opener or self._urllib_get
        self._roster: "dict[str, str]" = {}
        self._lock = threading.Lock()

    @staticmethod
    def _urllib_get(url: str) -> bytes:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=_ROSTER_FETCH_TIMEOUT_S) as resp:  # noqa: S310 — loopback only
            return resp.read()

    def __call__(self) -> "dict[str, str]":
        """Refresh from the read sidecar (best-effort) and return the roster snapshot."""
        self._refresh()
        with self._lock:
            return dict(self._roster)

    def _refresh(self) -> None:
        if not self._endpoint:
            return
        url = f"{self._endpoint}/buffer?since=0"
        try:
            raw = self._opener(url)
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
            logger.debug("roster refresh: read sidecar unreachable (%s)", type(exc).__name__)
            return
        except Exception as exc:  # noqa: BLE001 — a roster miss is never fatal
            logger.debug("roster refresh failed: %s", type(exc).__name__)
            return
        try:
            body = json.loads(raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw))
        except (ValueError, TypeError) as exc:
            logger.debug("roster refresh: bad JSON from read sidecar (%s)", exc)
            return
        events = body.get("events") if isinstance(body, dict) else None
        if not isinstance(events, list):
            return
        self._ingest(events)

    def _ingest(self, events: list) -> None:
        with self._lock:
            for wrapped in events:
                if not isinstance(wrapped, dict):
                    continue
                ev = wrapped.get("event")
                if not isinstance(ev, dict) or ev.get("type") != "chat":
                    continue
                login = ev.get("chatter_login")
                uid = ev.get("chatter_user_id")
                if not isinstance(login, str) or not login:
                    continue
                if not isinstance(uid, str) or not uid:
                    continue
                key = login.lower()
                # Most-recent wins: re-insert at the end (refresh insertion order).
                self._roster.pop(key, None)
                self._roster[key] = uid
            # Evict the oldest while over the cap.
            while len(self._roster) > self._cap:
                oldest = next(iter(self._roster))
                self._roster.pop(oldest, None)


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
def make_handler(service: Any, store: ProposalStore, *, ready_fn: Callable[[], bool],
                 broadcaster_id_fn: Callable[[], str],
                 chat_send_fn: "Optional[Callable[[str], bool]]" = None,
                 shoutout_fn: "Optional[Callable[[str], bool]]" = None):
    """Build a ``BaseHTTPRequestHandler`` subclass bound to this sidecar's state.

    A factory (not module globals) so a test can stand up an isolated server with
    an injected fake :class:`ModerationService` on an ephemeral port without
    cross-test state. Every handler is fail-safe: it never raises out (an internal
    fault becomes a 500 JSON body, never a stack trace into the socket)."""

    class _Handler(BaseHTTPRequestHandler):
        server_version = "KenningTwitchWrite/1.0"

        # -- response helper --------------------------------------------- #
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

        def _read_json_body(self) -> "Optional[dict]":
            """Read + parse a JSON object body. Returns ``None`` (after sending a
            400) on a bad/oversized/non-object body. Never raises."""
            try:
                n = int(self.headers.get("Content-Length", "0") or "0")
            except (ValueError, TypeError):
                self._send(400, {"ok": False, "error": "bad content-length"})
                return None
            if n < 0 or n > _MAX_BODY_BYTES:
                self._send(400, {"ok": False, "error": "bad content-length"})
                return None
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, TypeError) as exc:
                self._send(400, {"ok": False, "error": f"bad request: {exc}"})
                return None
            if not isinstance(payload, dict):
                self._send(400, {"ok": False, "error": "body must be a JSON object"})
                return None
            return payload

        # -- GET --------------------------------------------------------- #
        def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            path = urlsplit(self.path).path
            if path == "/healthz":
                try:
                    ready = bool(ready_fn())
                    bid = str(broadcaster_id_fn() or "")
                except Exception as exc:  # noqa: BLE001 — healthz never raises
                    logger.debug("healthz state read failed: %s", exc)
                    ready, bid = False, ""
                self._send(200, {"ok": True, "ready": ready, "broadcaster_id": bid})
                return
            self._send(404, {"ok": False, "error": "not found"})

        # -- POST -------------------------------------------------------- #
        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path == "/prepare":
                self._handle_prepare()
                return
            if path == "/confirm":
                self._handle_confirm()
                return
            if path == "/cancel":
                self._handle_cancel()
                return
            if path == "/say":
                self._handle_say()
                return
            if path == "/chat_settings":
                self._handle_chat_settings()
                return
            if path == "/shoutout":
                self._handle_shoutout()
                return
            self._send(404, {"ok": False, "error": "not found"})

        def _handle_shoutout(self) -> None:
            """Issue a Helix /shoutout to a raider (POST /chat/shoutouts). The
            orchestrator's raid handler POSTs ``{"to_broadcaster_id": "<raider id>"}``.
            Fail-safe: a missing capability or a shoutout error never raises out --
            the raid VOCAL announce is independent and is NOT gated on this."""
            payload = self._read_json_body()
            if payload is None:
                return
            tid = payload.get("to_broadcaster_id")
            if not isinstance(tid, str) or not tid.strip():
                self._send(400, {"ok": False, "error": "to_broadcaster_id must be a non-empty string"})
                return
            if shoutout_fn is None:
                self._send(200, {"ok": False, "error": "shoutout_unavailable"})
                return
            try:
                done = bool(shoutout_fn(tid.strip()))
            except Exception as exc:  # noqa: BLE001 — fail-safe; never block the announce
                logger.warning("shoutout failed unexpectedly: %s", type(exc).__name__)
                self._send(200, {"ok": False, "error": "shoutout_error"})
                return
            self._send(200, {"ok": done})

        def _handle_chat_settings(self) -> None:
            """Apply a chat-settings voice command (slow/follower/sub/emote/unique/
            clear). Parsed here (deterministic, no model) -> the service applies it."""
            payload = self._read_json_body()
            if payload is None:
                return
            text = payload.get("text")
            if not isinstance(text, str):
                self._send(400, {"ok": False, "error": "text must be a string"})
                return
            if service is None:
                self._send(200, {"ok": False, "error": "not_ready"})
                return
            from kenning.twitch.moderation.chat_settings import parse_chat_settings
            cmd = parse_chat_settings(text)
            if cmd is None:
                self._send(200, {"ok": False, "not_a_command": True})
                return
            try:
                result = service.apply_chat_settings(cmd)
            except Exception as exc:  # noqa: BLE001 — service is fail-safe; belt+braces
                logger.warning("chat-settings apply failed: %s", exc)
                self._send(500, {"ok": False, "error": "apply_failed"})
                return
            self._send(200, result)

        def _handle_say(self) -> None:
            """Post a chat message AS THE BOT (the periodic commands-panel poster).
            Loopback-only; the orchestrator builds the text and POSTs it here."""
            payload = self._read_json_body()
            if payload is None:
                return
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                self._send(400, {"ok": False, "error": "text must be a non-empty string"})
                return
            if chat_send_fn is None:
                self._send(200, {"ok": False, "error": "chat_send_unavailable"})
                return
            try:
                sent = bool(chat_send_fn(text))
            except Exception as exc:  # noqa: BLE001 — fail-safe
                logger.warning("chat-send failed unexpectedly: %s", exc)
                self._send(500, {"ok": False, "error": "send_failed"})
                return
            self._send(200, {"ok": sent})

        def _handle_prepare(self) -> None:
            payload = self._read_json_body()
            if payload is None:
                return
            text = payload.get("text")
            if not isinstance(text, str):
                self._send(400, {"ok": False, "error": "text must be a string"})
                return
            try:
                proposal = service.prepare(text)
            except Exception as exc:  # noqa: BLE001 — service is fail-safe; belt+braces
                logger.warning("prepare failed unexpectedly: %s", exc)
                self._send(500, {"ok": False, "error": "prepare_failed"})
                return
            if proposal is None:
                # Not a moderation command at all.
                self._send(200, {"ok": False, "not_a_command": True})
                return
            ok = bool(getattr(proposal, "ok", False))
            token = store.put(proposal) if ok else ""
            command = getattr(proposal, "command", None)
            action = getattr(command, "action", "") if command is not None else ""
            self._send(200, {
                "ok": ok,
                "token": token,
                "readback": getattr(proposal, "readback", ""),
                "reason_blocked": getattr(proposal, "reason_blocked", ""),
                "candidates": list(getattr(proposal, "candidates", []) or []),
                "action": action,
                "target": getattr(proposal, "resolved_name", ""),
            })

        def _handle_confirm(self) -> None:
            payload = self._read_json_body()
            if payload is None:
                return
            token = payload.get("token")
            if not isinstance(token, str) or not token:
                self._send(400, {"ok": False, "error": "token must be a string"})
                return
            proposal = store.pop(token)
            if proposal is None:
                self._send(200, {"ok": False, "error": "expired"})
                return
            try:
                result = service.confirm(proposal)
            except Exception as exc:  # noqa: BLE001 — service is fail-safe; belt+braces
                logger.warning("confirm failed unexpectedly: %s", exc)
                self._send(500, {"ok": False, "error": "confirm_failed"})
                return
            if not isinstance(result, dict):
                result = {"ok": False, "error": "bad_confirm_result"}
            self._send(200, result)

        def _handle_cancel(self) -> None:
            payload = self._read_json_body()
            if payload is None:
                return
            token = payload.get("token")
            if isinstance(token, str) and token:
                store.discard(token)
            self._send(200, {"ok": True})

        def log_message(self, *args: Any) -> None:  # noqa: ARG002 — fail-quiet
            return

    return _Handler


# --------------------------------------------------------------------------- #
# Service assembly (fail-safe: serves even with creds absent)
# --------------------------------------------------------------------------- #
class _ServiceState:
    """Holds the (possibly not-ready) ModerationService + the resolved ids.

    Build never raises: if creds/ids are missing, ``service`` is ``None`` and the
    action routes refuse cleanly while ``/healthz`` reports ``ready=false``."""

    def __init__(self) -> None:
        self.service: Any = None
        self.broadcaster_id: str = ""
        # Optional bot chat-SEND callable (text -> bool); wired when a bot token +
        # bot id resolve. Used by the periodic commands-panel poster via POST /say.
        self.chat_send: Optional[Callable[[str], bool]] = None
        # Optional /shoutout callable (to_broadcaster_id -> bool); wired from the
        # SAME HelixClient + broadcaster id the moderation service uses. Used by the
        # raid handler via POST /shoutout. Needs moderator:manage:shoutouts scope.
        self.shoutout: Optional[Callable[[str], bool]] = None

    @property
    def ready(self) -> bool:
        return self.service is not None and bool(self.broadcaster_id)


def _load_access_token(token_path: str) -> str:
    """Load the broadcaster's stored OAuth access token (fail-quiet -> "").

    Fast path — reads from disk only.  Call _proactive_token_refresh() ONCE
    at startup to rotate an expired token; never do the HTTP refresh here
    because this function is called on every Helix request via get_token().
    """
    try:
        from kenning.twitch.auth import TokenStore
    except Exception as exc:  # noqa: BLE001
        logger.warning("twitch auth import failed: %s", type(exc).__name__)
        return ""
    try:
        tokens = TokenStore(token_path).load()
    except Exception as exc:  # noqa: BLE001
        logger.warning("twitch token load failed path=%s: %s", token_path, type(exc).__name__)
        return ""
    if not isinstance(tokens, dict):
        return ""
    access = tokens.get("access_token")
    return access if isinstance(access, str) else ""


def _proactive_token_refresh(token_path: str, client_id: str) -> None:
    """One-shot proactive refresh at sidecar startup (fail-quiet).

    If the stored access token is expired or within 5 minutes of expiry,
    rotates it using the stored refresh_token so the very first Helix call
    uses a live token.  Never called inside get_token() — that path is
    fast (disk-only) so per-request latency is unaffected.
    """
    if not client_id:
        return
    try:
        from kenning.twitch.auth import TokenStore, TwitchAuth
        store = TokenStore(token_path)
        if store.is_expired(margin_seconds=300.0):
            TwitchAuth(client_id, store).ensure_valid(margin_seconds=300.0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("proactive token refresh skipped (%s)", type(exc).__name__)


def _make_message_id_lookup(read_endpoint: str) -> Callable[[str], Optional[str]]:
    """A ``login -> last message_id`` resolver backed by the read sidecar's
    ``GET /last_message`` route (the cross-process delete plumb). Fail-safe: an
    unreachable sidecar / bad body / no recent message -> ``None`` (the
    ModerationService then reports ``no_message`` rather than deleting)."""
    base = (read_endpoint or "").rstrip("/")

    def _lookup(login: str) -> Optional[str]:
        if not base or not login:
            return None
        url = f"{base}/last_message?login={urllib.parse.quote(str(login), safe='')}"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=_ROSTER_FETCH_TIMEOUT_S) as resp:  # noqa: S310 — loopback only
                raw = resp.read()
            body = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception as exc:  # noqa: BLE001 — a lookup miss is never fatal
            logger.debug("message_id lookup failed for %r: %s", login, type(exc).__name__)
            return None
        mid = body.get("message_id") if isinstance(body, dict) else None
        return str(mid) if mid else None

    return _lookup


def build_service_state() -> _ServiceState:
    """Assemble the ModerationService from the environment, fail-safe.

    Loads the broadcaster token, resolves the broadcaster + bot logins to numeric
    ids over Helix, and wires HelixClient + ModerationGuard + ModerationService.
    ANY missing piece (no token, no client id, unresolved id, an import/construct
    fault) leaves a not-ready state — the sidecar still serves and refuses actions.
    """
    state = _ServiceState()
    client_id = os.environ.get("KENNING_TWITCH_CLIENT_ID", "")
    broadcaster_login = os.environ.get("KENNING_TWITCH_BROADCASTER_LOGIN", "")
    bot_login = os.environ.get("KENNING_TWITCH_BOT_LOGIN", "")
    token_path = os.environ.get(
        "KENNING_TWITCH_BROADCASTER_TOKEN_PATH", "~/.kenning/twitch.json"
    )
    read_endpoint = DEFAULT_READ_ENDPOINT
    helix_base = DEFAULT_HELIX_BASE
    require_confirm = os.environ.get("KENNING_TWITCH_MOD_REQUIRE_CONFIRM", "1") == "1"

    if not client_id or not broadcaster_login:
        logger.warning(
            "write sidecar: creds absent (client_id=%s broadcaster_login=%s); "
            "serving not-ready (actions refused)",
            bool(client_id), bool(broadcaster_login),
        )
        return state

    # Proactive one-shot refresh at startup (fail-quiet).
    _proactive_token_refresh(token_path, client_id)

    # get_token re-reads the store each call so a rotated token is picked up.
    def get_token() -> str:
        return _load_access_token(token_path)

    access_token = get_token()
    if not access_token:
        logger.warning(
            "write sidecar: no broadcaster access token at %s; serving not-ready",
            token_path,
        )
        return state

    # Resolve broadcaster + bot logins -> numeric ids over Helix.
    try:
        from kenning.twitch.clients.helix_eventsub import HelixEventSubClient

        resolver = HelixEventSubClient(client_id, base_url=helix_base)
        broadcaster_id = resolver.get_user_id(broadcaster_login, token=access_token)
        bot_id = (
            resolver.get_user_id(bot_login, token=access_token) if bot_login else ""
        )
    except Exception as exc:  # noqa: BLE001 — id resolution failure => not-ready
        logger.warning("write sidecar: id resolution failed (%s); not-ready", type(exc).__name__)
        return state

    if not broadcaster_id:
        logger.warning(
            "write sidecar: broadcaster login %r did not resolve; not-ready",
            broadcaster_login,
        )
        return state

    # The broadcaster is a moderator of their own channel.
    moderator_id = broadcaster_id
    protected_ids = {broadcaster_id}
    if bot_id:
        protected_ids.add(bot_id)

    try:
        from kenning.twitch.moderation.guard import ModerationGuard
        from kenning.twitch.moderation.helix import HelixClient
        from kenning.twitch.moderation.service import ModerationService

        roster = RosterCache(read_endpoint)

        # The guard's roster_provider yields RosterEntry-shaped mappings (it maps
        # user_id/login itself); our cache is {login: user_id} so we adapt it.
        def guard_roster() -> list:
            return [
                {"user_id": uid, "login": login, "display_name": login}
                for login, uid in roster().items()
            ]

        helix = HelixClient(client_id, get_token=get_token)
        # Honour the configured mass-action breaker limit (the orchestrator passes
        # twitch.moderation.mass_action_limit_per_60s as this env). 0/absent -> the
        # guard's own default.
        try:
            breaker_limit = int(os.environ.get("KENNING_TWITCH_MOD_BREAKER_LIMIT", "0") or "0")
        except (TypeError, ValueError):
            breaker_limit = 0
        guard = (
            ModerationGuard(guard_roster, protected_ids=protected_ids, breaker_limit=breaker_limit)
            if breaker_limit > 0
            else ModerationGuard(guard_roster, protected_ids=protected_ids)
        )
        service = ModerationService(
            helix,
            guard,
            broadcaster_id=broadcaster_id,
            moderator_id=moderator_id,
            roster_provider=lambda: roster(),
            require_readback_confirm=require_confirm,
            message_id_lookup=_make_message_id_lookup(read_endpoint),
        )
    except Exception as exc:  # noqa: BLE001 — any wiring fault => not-ready
        logger.warning("write sidecar: service wiring failed (%s); not-ready", type(exc).__name__)
        return state

    state.service = service
    state.broadcaster_id = broadcaster_id
    logger.info(
        "write sidecar service ready: broadcaster_id=%s protected=%d require_confirm=%s",
        broadcaster_id, len(protected_ids), require_confirm,
    )
    # /shoutout (Helix POST /chat/shoutouts, moderator:manage:shoutouts) for the
    # raid handler. Reuses the SAME HelixClient + broadcaster id the moderation
    # service uses (the broadcaster moderates their own channel -> moderator_id ==
    # broadcaster_id). Returns ok on a 2xx OR an idempotent already/cooldown. Wired
    # only when the helix client built; never blocks the moderation service.
    try:
        _bid = broadcaster_id
        _mid = moderator_id

        def _do_shoutout(to_broadcaster_id: str) -> bool:
            tid = str(to_broadcaster_id or "").strip()
            if not tid:
                return False
            result = helix.send_shoutout(_bid, tid, _mid)
            return bool(getattr(result, "ok", False))

        state.shoutout = _do_shoutout
        logger.info("write sidecar: shoutout ready (from_broadcaster_id=%s)", _bid)
    except Exception as exc:  # noqa: BLE001 — shoutout is optional
        logger.warning("write sidecar: shoutout wiring failed (%s)", type(exc).__name__)
    # Bot chat-SEND (Helix POST /chat/messages, user:write:chat) for the periodic
    # commands-panel poster. Wired only when a bot id resolved AND a bot token is on
    # disk; optional, never blocks the moderation service.
    try:
        bot_token_path = os.environ.get(
            "KENNING_TWITCH_BOT_TOKEN_PATH", "~/.kenning/twitch_bot.json")
        if bot_id and _load_access_token(bot_token_path):
            from kenning.twitch.clients.chat_send import ChatSendClient
            sender = ChatSendClient(
                client_id, get_token=lambda: _load_access_token(bot_token_path))
            _bid, _sid = broadcaster_id, bot_id
            state.chat_send = lambda text: sender.send(_bid, _sid, text)
            logger.info("write sidecar: chat-send ready (bot sender_id=%s)", bot_id)
    except Exception as exc:  # noqa: BLE001 — chat-send is optional
        logger.warning("write sidecar: chat-send wiring failed (%s)", type(exc).__name__)
    return state


# --------------------------------------------------------------------------- #
# Server assembly
# --------------------------------------------------------------------------- #
def build_server(service: Any, *, port: int = 0, ready: Optional[bool] = None,
                 broadcaster_id: str = "",
                 chat_send: "Optional[Callable[[str], bool]]" = None,
                 shoutout: "Optional[Callable[[str], bool]]" = None):
    """Assemble a write-sidecar server bound to 127.0.0.1 ONLY.

    ``port=0`` binds an ephemeral port (read it back from ``server.server_address``
    in a test). ``service`` is injected (tests pass a fake ModerationService); a
    ``None`` service serves a not-ready surface that refuses actions. ``ready`` and
    ``broadcaster_id`` populate ``/healthz`` (``ready`` defaults to
    ``service is not None``). Returns ``(server, store)``; the caller runs
    ``serve_forever`` (or, in a test, drives requests against the bound address)."""
    store = ProposalStore()
    is_ready = (service is not None) if ready is None else bool(ready)

    handler = make_handler(
        service,
        store,
        ready_fn=lambda: is_ready,
        broadcaster_id_fn=lambda: broadcaster_id,
        chat_send_fn=chat_send,
        shoutout_fn=shoutout,
    )
    from kenning.subprocess.sidecar_server import SingletonThreadingHTTPServer

    server = SingletonThreadingHTTPServer(("127.0.0.1", port), handler)
    return server, store


# --------------------------------------------------------------------------- #
# Parent-death deadman (clone of the read-sidecar precedent)
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
    """Single-shot watchdog decision for ``pid``: ``"alive"`` if the parent is
    still running (or the pid is unset/invalid -> do not self-kill) and ``"dead"``
    if the parent is gone (the caller should ``os._exit``). Split out so it is
    unit-testable without spawning a process."""
    if pid <= 0:
        return "alive"
    return "alive" if _pid_alive(pid) else "dead"


def _parent_watchdog(poll_seconds: float = 3.0) -> None:
    """Self-exit when the parent (Ultron orchestrator) dies, so a force-killed or
    crashed parent NEVER leaves this sidecar as a runaway orphan holding a live
    Twitch write capability. Parent pid via ``KENNING_TWITCH_PARENT_PID`` (fallback:
    the spawn-time parent). ``os._exit`` skips atexit/locks so the socket is freed
    immediately by the OS."""
    try:
        ppid = int(os.environ.get("KENNING_TWITCH_PARENT_PID", "0") or "0")
    except Exception:  # noqa: BLE001
        ppid = 0
    if ppid <= 0:
        ppid = os.getppid()
    if ppid <= 0:
        return
    sys.stderr.write(f"[twitch-write] parent-watchdog armed on pid {ppid}\n")
    sys.stderr.flush()
    while True:
        time.sleep(poll_seconds)
        if parent_watchdog_check(ppid) == "dead":
            sys.stderr.write(
                f"[twitch-write] parent pid {ppid} gone -> self-terminating "
                "(orphan guard)\n")
            sys.stderr.flush()
            os._exit(0)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [twitch-write] %(levelname)s %(message)s",
    )
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    import atexit

    from kenning.subprocess import sidecar_lock
    # Anti-stale-sidecar guard: reap same-role strays + reclaim the port BEFORE
    # binding (which build_server does EXCLUSIVELY) -> exactly one live instance.
    sidecar_lock.guard_singleton("127.0.0.1", port, "twitch_write")

    # Fail-safe assembly: a missing-creds state still serves (actions refused).
    state = build_service_state()
    server, _store = build_server(
        state.service, port=port, ready=state.ready, broadcaster_id=state.broadcaster_id,
        chat_send=state.chat_send, shoutout=state.shoutout,
    )
    sidecar_lock.write_role("twitch_write", os.getpid(), port)
    atexit.register(sidecar_lock.clear_role, "twitch_write")
    # Parent-death deadman: the child cleans itself up on ANY parent death.
    threading.Thread(target=_parent_watchdog, daemon=True,
                     name="twitch-write-parent-watchdog").start()
    host, bound_port = server.server_address[:2]
    logger.info(
        "twitch write sidecar serving on http://%s:%s (ready=%s)",
        host, bound_port, state.ready,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        sidecar_lock.clear_role("twitch_write")


if __name__ == "__main__":
    main()
