"""Minimal Helix client for EventSub WebSocket subscription bootstrap (urllib).

The read sidecar opens an EventSub ``wss`` socket, receives the
``session_welcome`` (carrying a ``session_id``), and then must create the
subscriptions that bind that session to a live channel's chat (and, optionally,
its channel-point redeems). Those subscriptions are created OUT-OF-BAND over the
Twitch Helix REST API; this module is the thin, fail-safe Helix client the
sidecar uses for exactly that bootstrap plus the ``/users`` login->id lookup.

Scope (deliberately tiny — the moderation Helix write client lives elsewhere):
  * ``get_user_id(login)``          -> GET  /helix/users?login=...
  * ``create_chat_subscription``    -> POST /helix/eventsub/subscriptions
  * ``create_redeem_subscription``  -> POST /helix/eventsub/subscriptions

ANTICHEAT (BR-P1): stdlib + ``urllib`` only. No ``requests`` / ``aiohttp`` /
``websockets`` / desktop-automation. The HTTP transport is an INJECTED callable
so the unit tests run fully offline with a mock::

    transport(method, url, headers: dict, body: Optional[bytes]) -> (status: int, body_bytes: bytes)

The default transport is a small ``urllib.request`` implementation that NEVER
raises for ordinary HTTP error statuses (it returns the (status, body) so the
caller can branch); genuine connection-level failures (DNS/TLS/timeout) surface
as a transport exception which every public method catches and converts to a
fail-safe ``None`` / ``False`` (logged) — a Helix hiccup must never raise into
the sidecar's poll loop.

Token discipline: a bearer token is sent only in the ``Authorization`` header
and is NEVER logged. The chat subscription uses the BOT's token (the bot is the
``user_id`` reading chat); the redeem subscription uses the BROADCASTER's token
(``channel:read:redemptions`` is a broadcaster scope). The caller passes the
right token per method (S3 / docs/twitch_integration EventSub bootstrap).
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Optional, Tuple

logger = logging.getLogger("kenning.twitch.clients.helix_eventsub")

__all__ = [
    "HelixEventSubError",
    "HelixEventSubClient",
    "HELIX_BASE",
    "CHAT_SUBSCRIPTION_TYPE",
    "REDEEM_SUBSCRIPTION_TYPE",
    "RAID_SUBSCRIPTION_TYPE",
]

HELIX_BASE = "https://api.twitch.tv/helix"

# EventSub subscription type names + versions (dev.twitch.tv/docs/eventsub,
# verified live 2026-06-23): channel.chat.message v1 needs broadcaster_user_id +
# user_id; the redemption-add v1 needs only broadcaster_user_id; channel.raid v1
# binds via to_broadcaster_user_id (the channel BEING raided -> the 'to' side
# needs NO special scope, only that the subscribing user can read the channel).
CHAT_SUBSCRIPTION_TYPE = "channel.chat.message"
REDEEM_SUBSCRIPTION_TYPE = "channel.channel_points_custom_reward_redemption.add"
RAID_SUBSCRIPTION_TYPE = "channel.raid"
_SUBSCRIPTION_VERSION = "1"

# A successful POST /eventsub/subscriptions returns 202 Accepted (it may also
# surface 200 on some replays); GET /users returns 200. We accept any 2xx.
_DEFAULT_TIMEOUT = 10.0

# The injected transport: (method, url, headers, body) -> (status, body_bytes).
Transport = Callable[[str, str, dict, Optional[bytes]], Tuple[int, bytes]]


class HelixEventSubError(Exception):
    """A Helix EventSub bootstrap fault.

    Raised only by the default transport for genuine connection-level failures
    (DNS / TLS / timeout / socket reset). Every PUBLIC method on
    :class:`HelixEventSubClient` CATCHES this (and anything else) and converts it
    to a fail-safe ``None`` / ``False`` so a Helix hiccup never raises into the
    sidecar's poll loop (BR-2.3 error-handling on every external call).
    """


def _default_transport(timeout: float = _DEFAULT_TIMEOUT) -> Transport:
    """The real ``urllib`` transport: no third-party deps (BR-P1).

    Returns ``(status, body_bytes)``. A non-2xx HTTP status does NOT raise — the
    body is read so the caller can log Twitch's error envelope. Genuine
    transport failures (DNS / TLS / timeout / connection reset) raise
    :class:`HelixEventSubError`.
    """

    def _transport(
        method: str,
        url: str,
        headers: dict,
        body: Optional[bytes],
    ) -> Tuple[int, bytes]:
        req = urllib.request.Request(
            url, data=body, headers=dict(headers or {}), method=(method or "GET").upper()
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — Twitch API over HTTPS
                status = int(getattr(resp, "status", resp.getcode()) or 0)
                raw = resp.read()
                return status, raw
        except urllib.error.HTTPError as exc:
            # Non-2xx — read the body so the caller can log the Twitch error.
            status = int(exc.code)
            try:
                raw = exc.read()
            except Exception:  # noqa: BLE001
                raw = b""
            return status, raw
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise HelixEventSubError(f"helix transport error contacting Twitch: {exc}") from exc

    return _transport


class HelixEventSubClient:
    """A minimal, fail-safe Helix client for the EventSub subscription bootstrap.

    All three public methods are fail-safe: any error (transport raise, non-2xx
    status, malformed body) is logged and converted to ``None`` (``get_user_id``)
    or ``False`` (the ``create_*`` methods) — they NEVER raise into the sidecar's
    poll loop. The HTTP transport is injectable for offline tests.
    """

    def __init__(
        self,
        client_id: str,
        *,
        base_url: str = HELIX_BASE,
        transport: Optional[Transport] = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        if not client_id or not isinstance(client_id, str):
            raise ValueError("client_id is required")
        self._client_id = client_id
        self._base = (base_url or HELIX_BASE).rstrip("/")
        self._timeout = float(timeout)
        self._transport: Transport = transport or _default_transport(self._timeout)

    # ------------------------------------------------------------------ #
    # Header builders (a bearer token is NEVER logged)
    # ------------------------------------------------------------------ #
    def _auth_headers(self, token: str, *, json_body: bool) -> dict:
        headers = {
            "Authorization": f"Bearer {token}",
            "Client-Id": self._client_id,
        }
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    @staticmethod
    def _parse_json(raw: bytes) -> dict:
        """Best-effort JSON-object parse of a response body; ``{}`` on anything else."""
        if not raw:
            return {}
        try:
            obj = json.loads(raw.decode("utf-8", "replace"))
        except (ValueError, TypeError):
            return {}
        return obj if isinstance(obj, dict) else {}

    @staticmethod
    def _is_2xx(status: int) -> bool:
        return 200 <= status < 300

    @staticmethod
    def _preview(raw: bytes, limit: int = 300) -> str:
        """A short, token-free body preview for an error log (never carries a token —
        Helix error envelopes echo the request type/condition, not the bearer)."""
        try:
            text = raw.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            text = repr(raw[:limit])
        return text[:limit]

    # ------------------------------------------------------------------ #
    # GET /users?login=<login>
    # ------------------------------------------------------------------ #
    def get_user_id(self, login: str, *, token: str) -> Optional[str]:
        """Resolve a Twitch login to its numeric user id. ``None`` on any failure.

        GET ``{base}/users?login={login}`` with ``Authorization: Bearer <token>``
        + ``Client-Id``. Returns ``data[0]["id"]`` or ``None`` (fail-safe: logs +
        returns ``None`` on a transport error, a non-2xx, an empty ``data`` list,
        or a missing id).
        """
        if not login or not isinstance(login, str):
            logger.warning("helix get_user_id: empty/invalid login")
            return None
        if not token:
            logger.warning("helix get_user_id: missing token for login=%s", login)
            return None
        url = f"{self._base}/users?login={urllib.parse.quote(login, safe='')}"
        try:
            status, raw = self._transport("GET", url, self._auth_headers(token, json_body=False), None)
        except Exception as exc:  # noqa: BLE001 — fail-safe, never raise into the poll loop
            logger.warning("helix get_user_id transport failed login=%s: %s", login, type(exc).__name__)
            return None
        if not self._is_2xx(status):
            logger.warning(
                "helix get_user_id non-2xx login=%s status=%s body=%s",
                login, status, self._preview(raw),
            )
            return None
        body = self._parse_json(raw)
        data = body.get("data")
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            logger.warning("helix get_user_id empty/!data for login=%s", login)
            return None
        uid = data[0].get("id")
        if not isinstance(uid, str) or not uid:
            logger.warning("helix get_user_id missing id for login=%s", login)
            return None
        logger.info("helix resolved login=%s -> user_id=%s", login, uid)
        return uid

    # ------------------------------------------------------------------ #
    # POST /eventsub/subscriptions
    # ------------------------------------------------------------------ #
    def _create_subscription(self, body_obj: dict, *, token: str, what: str) -> bool:
        """POST a subscription create request. ``True`` on 2xx, else ``False`` (logged)."""
        if not token:
            logger.warning("helix %s: missing token", what)
            return False
        url = f"{self._base}/eventsub/subscriptions"
        try:
            payload = json.dumps(body_obj).encode("utf-8")
        except (TypeError, ValueError) as exc:
            logger.warning("helix %s: payload encode failed: %s", what, exc)
            return False
        try:
            status, raw = self._transport(
                "POST", url, self._auth_headers(token, json_body=True), payload
            )
        except Exception as exc:  # noqa: BLE001 — fail-safe
            logger.warning("helix %s transport failed: %s", what, type(exc).__name__)
            return False
        if self._is_2xx(status):
            logger.info("helix %s created (status=%s)", what, status)
            return True
        logger.warning("helix %s failed status=%s body=%s", what, status, self._preview(raw))
        return False

    def create_chat_subscription(
        self,
        *,
        broadcaster_id: str,
        bot_user_id: str,
        session_id: str,
        token: str,
    ) -> bool:
        """Create the ``channel.chat.message`` v1 EventSub subscription.

        The condition binds the BROADCASTER's chat to the BOT identity reading it
        (both ids required, per the live spec). The transport is ``websocket``
        bound to the ``session_id`` from the welcome. ``token`` must be the BOT's
        user access token (``user:read:chat``). Returns ``True`` on 2xx (incl.
        202 Accepted), ``False`` otherwise.
        """
        if not broadcaster_id or not bot_user_id or not session_id:
            logger.warning(
                "helix chat-sub: missing id(s) broadcaster=%r bot=%r session=%r",
                bool(broadcaster_id), bool(bot_user_id), bool(session_id),
            )
            return False
        body = {
            "type": CHAT_SUBSCRIPTION_TYPE,
            "version": _SUBSCRIPTION_VERSION,
            "condition": {
                "broadcaster_user_id": str(broadcaster_id),
                "user_id": str(bot_user_id),
            },
            "transport": {"method": "websocket", "session_id": str(session_id)},
        }
        return self._create_subscription(body, token=token, what="chat-subscription")

    def create_redeem_subscription(
        self,
        *,
        broadcaster_id: str,
        session_id: str,
        token: str,
    ) -> bool:
        """Create the ``channel.channel_points_custom_reward_redemption.add`` v1 sub.

        The condition needs only ``broadcaster_user_id``. ``token`` must be the
        BROADCASTER's user access token (``channel:read:redemptions`` is a
        broadcaster scope). Returns ``True`` on 2xx, ``False`` otherwise.
        """
        if not broadcaster_id or not session_id:
            logger.warning(
                "helix redeem-sub: missing id(s) broadcaster=%r session=%r",
                bool(broadcaster_id), bool(session_id),
            )
            return False
        body = {
            "type": REDEEM_SUBSCRIPTION_TYPE,
            "version": _SUBSCRIPTION_VERSION,
            "condition": {"broadcaster_user_id": str(broadcaster_id)},
            "transport": {"method": "websocket", "session_id": str(session_id)},
        }
        return self._create_subscription(body, token=token, what="redeem-subscription")

    def create_raid_subscription(
        self,
        *,
        broadcaster_id: str,
        session_id: str,
        token: str,
    ) -> bool:
        """Create the ``channel.raid`` v1 EventSub subscription for INCOMING raids.

        The condition binds the channel BEING raided via ``to_broadcaster_user_id``
        (the 'to' side needs NO special scope — only that the subscribing user can
        read the channel). ``token`` is the BROADCASTER's user access token (it
        rides the SAME isolated session as the redeem sub, which also uses the
        broadcaster token, so there is no cross-user 400). Returns ``True`` on 2xx,
        ``False`` otherwise.
        """
        if not broadcaster_id or not session_id:
            logger.warning(
                "helix raid-sub: missing id(s) broadcaster=%r session=%r",
                bool(broadcaster_id), bool(session_id),
            )
            return False
        body = {
            "type": RAID_SUBSCRIPTION_TYPE,
            "version": _SUBSCRIPTION_VERSION,
            "condition": {"to_broadcaster_user_id": str(broadcaster_id)},
            "transport": {"method": "websocket", "session_id": str(session_id)},
        }
        return self._create_subscription(body, token=token, what="raid-subscription")
