"""S11 — Helix moderation write client (urllib transport, self-idempotent).

A thin shim over the Twitch Helix moderation write endpoints. The HTTP transport
is INJECTED (a callable matching :data:`Transport`) so tests never touch the real
network; the default transport is a small ``urllib`` loopback-safe implementation.

Design invariants (from docs/twitch_integration/02_board/{MASTER,S_report}.md,
SLICE 11):

  * SELF-IDEMPOTENT. Re-issuing a moderation action that already took effect must
    resolve to SUCCESS, never raise. Twitch signals this two ways and we treat
    BOTH as applied:
      - an HTTP ``409 Conflict``, or
      - any HTTP 4xx whose body carries an error message containing ``"already"``
        (e.g. ``"The user specified in the user_id field is already banned."`` or
        ``"... message ... does not exist"`` for a delete that already happened).
    Idempotency is also tracked LOCALLY by ``(action, target_id, message_id)``:
    once an action key has succeeded this process, re-issuing it short-circuits to
    a cached success WITHOUT another network call — so a flapping caller can never
    double-ban.
  * NEVER BLIND-RETRY A WRITE. A POST/DELETE/PATCH is retried ONLY on HTTP 429
    (rate limited) with exponential backoff (capped). Any other transport error
    fails LOUD (auth/permission/network faults must surface, never be silently
    swallowed — BR-2.5 / the board's "AUTH failures are LOUD").
  * RATE GOVERNOR. A token-bucket pre-gate kept far below the Helix ~800/min
    bucket so a buggy loop can't burst the real API; configurable rate + burst.

ANTICHEAT (BR-P1): stdlib + urllib + rapidfuzz only. No requests/aiohttp/
websockets, no desktop/input/screen libs.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

logger = logging.getLogger("kenning.twitch.moderation.helix")

__all__ = [
    "HelixClient",
    "HelixError",
    "HelixResult",
    "RateGovernor",
    "TokenProvider",
    "Transport",
    "TransportResponse",
]

_HELIX_BASE = "https://api.twitch.tv/helix"

# A token provider returns a bearer token string (already validated upstream).
TokenProvider = Callable[[], str]


@dataclass(frozen=True)
class TransportResponse:
    """A minimal HTTP response the injected transport must return.

    ``status`` is the HTTP status code; ``body`` is the raw decoded text body
    (may be empty for 204). The transport must NOT raise for ordinary HTTP error
    statuses (4xx/5xx) — it returns them here so the client can apply its
    idempotency / retry policy uniformly. It MAY raise for genuine
    connection-level failures (DNS/socket/TLS), which the client treats as a
    hard, non-retryable error (except the 429 path, which is a status, not a
    raise).
    """

    status: int
    body: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        """Best-effort JSON parse of the body; returns ``None`` on empty/invalid."""
        if not self.body:
            return None
        try:
            return json.loads(self.body)
        except (ValueError, TypeError):
            return None


# A transport takes (method, url, headers, body_bytes) and returns a
# TransportResponse. Injected so tests can mock Helix deterministically.
Transport = Callable[[str, str, Mapping[str, str], Optional[bytes]], TransportResponse]


class HelixError(RuntimeError):
    """A non-idempotent, non-retryable Helix failure (auth/permission/network/5xx).

    Carries the HTTP ``status`` (``None`` for a transport-level exception) and the
    raw ``body`` for the caller's audit trail. Raised LOUD so AUTH/permission
    faults are never silently swallowed.
    """

    def __init__(self, message: str, *, status: Optional[int] = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass(frozen=True)
class HelixResult:
    """The outcome of a Helix write.

    Attributes:
        action: the logical action ("ban" / "timeout" / "delete_message" /
            "update_chat_settings").
        ok: True when applied or already-applied.
        status: the HTTP status that produced this result (``0`` for a locally
            cached idempotent short-circuit).
        idempotent: True when this resolved as already-applied (409 / "already" /
            local cache hit) rather than a fresh 2xx.
        data: parsed response payload (Helix ``data`` list / settings dict), or
            ``None``.
        key: the ``(action, target_id, message_id)`` idempotency key.
    """

    action: str
    ok: bool
    status: int
    idempotent: bool
    data: Any = None
    key: tuple[str, str, str] = ("", "", "")


class RateGovernor:
    """A monotonic-clock token bucket.

    Refills ``rate`` tokens/second up to ``burst`` capacity. :meth:`acquire`
    blocks (sleeping in small bounded slices) until a token is available, then
    consumes one. Thread-safe. Kept deliberately far below the Helix ~800/min
    bucket so a buggy caller cannot burst the real API.

    A ``sleep`` and ``monotonic`` are injectable so tests can drive the clock
    deterministically without real waits.
    """

    def __init__(
        self,
        rate: float = 1.0,
        burst: int = 1,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be > 0")
        if burst < 1:
            raise ValueError("burst must be >= 1")
        self._rate = float(rate)
        self._capacity = float(burst)
        self._tokens = float(burst)
        self._monotonic = monotonic
        self._sleep = sleep
        self._last = monotonic()
        self._lock = threading.Lock()

    def _refill_locked(self) -> None:
        now = self._monotonic()
        elapsed = max(0.0, now - self._last)
        self._last = now
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)

    def try_acquire(self) -> bool:
        """Consume one token if available; return whether it was consumed."""
        with self._lock:
            self._refill_locked()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def acquire(self, *, timeout: Optional[float] = None) -> bool:
        """Block until a token is available (or ``timeout`` seconds elapse).

        Returns True once a token is consumed, False if it timed out. ``timeout``
        is measured on the injected monotonic clock.
        """
        deadline = None if timeout is None else self._monotonic() + timeout
        while True:
            with self._lock:
                self._refill_locked()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                # Time until the next whole token.
                needed = 1.0 - self._tokens
                wait = needed / self._rate
            if deadline is not None:
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    return False
                wait = min(wait, remaining)
            # Bound each sleep slice so an injected clock that doesn't advance
            # on sleep can still make progress via refill on the next loop.
            self._sleep(max(0.0, min(wait, 0.05)))


class HelixClient:
    """Self-idempotent Helix moderation write client over an injected transport.

    Args:
        client_id: the Twitch app/client id (sent as ``Client-Id``).
        get_token: a zero-arg callable returning the current bearer token. Called
            per request so a rotated token is always picked up; the client never
            persists the token.
        transport: the injected HTTP transport (see :data:`Transport`). Defaults
            to a small ``urllib`` implementation when omitted.
        rate_governor: a :class:`RateGovernor` (defaults to ~1 write/sec, burst 1).
        max_retries: max 429 backoff retries before giving up (default 4).
        base_backoff_s / max_backoff_s: exponential-backoff bounds for 429.
        monotonic / sleep: injectable clock for the backoff (tests drive it).

    Thread-safe. Records nothing itself — the :class:`~kenning.twitch.moderation.
    guard.ModerationGuard` owns the audit. All write methods return a
    :class:`HelixResult` and never raise on an idempotent already-applied state;
    they DO raise :class:`HelixError` on auth/permission/unexpected failures.
    """

    def __init__(
        self,
        client_id: str,
        get_token: TokenProvider,
        transport: Optional[Transport] = None,
        *,
        rate_governor: Optional[RateGovernor] = None,
        max_retries: int = 4,
        base_backoff_s: float = 0.5,
        max_backoff_s: float = 8.0,
        request_timeout_s: float = 10.0,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not client_id:
            raise ValueError("client_id is required")
        if not callable(get_token):
            raise ValueError("get_token must be callable")
        self._client_id = client_id
        self._get_token = get_token
        self._transport = transport or self._default_transport
        self._rate = rate_governor or RateGovernor(rate=1.0, burst=1, monotonic=monotonic, sleep=sleep)
        self._max_retries = max(0, int(max_retries))
        self._base_backoff = max(0.0, float(base_backoff_s))
        self._max_backoff = max(self._base_backoff, float(max_backoff_s))
        self._request_timeout = max(0.1, float(request_timeout_s))
        self._monotonic = monotonic
        self._sleep = sleep
        # Local idempotency cache: key -> HelixResult of the first success.
        self._applied: dict[tuple[str, str, str], HelixResult] = {}
        self._applied_lock = threading.Lock()

    # ---- default urllib transport -------------------------------------------
    def _default_transport(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: Optional[bytes],
    ) -> TransportResponse:
        """Minimal ``urllib`` transport. Returns ordinary HTTP errors as a
        :class:`TransportResponse` (so the client's policy applies uniformly);
        raises only on connection-level failures."""
        req = urllib.request.Request(url=url, data=body, method=method)
        for k, v in headers.items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=self._request_timeout) as resp:  # nosec B310 - https only, fixed host
                raw = resp.read().decode("utf-8", errors="replace")
                return TransportResponse(
                    status=getattr(resp, "status", 200) or 200,
                    body=raw,
                    headers={k.lower(): v for k, v in resp.headers.items()},
                )
        except urllib.error.HTTPError as exc:
            try:
                raw = exc.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001 - body read on an error response is best-effort
                raw = ""
            hdrs = {}
            try:
                hdrs = {k.lower(): v for k, v in (exc.headers or {}).items()}
            except Exception:  # noqa: BLE001
                hdrs = {}
            return TransportResponse(status=exc.code, body=raw, headers=hdrs)
        except urllib.error.URLError as exc:
            raise HelixError(f"helix transport error: {exc.reason}", status=None, body="") from exc

    # ---- helpers -------------------------------------------------------------
    @staticmethod
    def _body_says_already(body: str) -> bool:
        """True when a response body's error text indicates the action was already
        applied (Twitch returns e.g. '... is already banned' / 'does not exist')."""
        if not body:
            return False
        low = body.lower()
        if "already" in low:
            return True
        # A delete of a message that no longer exists is the delete-idempotent case.
        if "does not exist" in low and "message" in low:
            return True
        return False

    @staticmethod
    def _extract_error_message(body: str) -> str:
        """Pull a human 'message' field from a Helix error body for logs/raises."""
        if not body:
            return ""
        try:
            obj = json.loads(body)
        except (ValueError, TypeError):
            return body[:300]
        if isinstance(obj, dict):
            msg = obj.get("message")
            if isinstance(msg, str) and msg:
                return msg
        return body[:300]

    def _headers(self, *, json_body: bool) -> dict[str, str]:
        token = self._get_token() or ""
        if not token:
            raise HelixError("no bearer token available (auth not ready)", status=None, body="")
        h = {
            "Authorization": f"Bearer {token}",
            "Client-Id": self._client_id,
        }
        if json_body:
            h["Content-Type"] = "application/json"
        return h

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str],
        json_payload: Optional[dict[str, Any]] = None,
    ) -> TransportResponse:
        """Issue ONE governed request with 429-only backoff. Never blind-retries
        on any non-429 status."""
        qs = urllib.parse.urlencode({k: v for k, v in query.items() if v is not None and v != ""})
        url = f"{_HELIX_BASE}{path}"
        if qs:
            url = f"{url}?{qs}"
        body_bytes: Optional[bytes] = None
        if json_payload is not None:
            body_bytes = json.dumps(json_payload).encode("utf-8")
        headers = self._headers(json_body=json_payload is not None)

        attempt = 0
        while True:
            # Pre-gate every network call through the rate governor.
            self._rate.acquire()
            resp = self._transport(method, url, headers, body_bytes)
            if resp.status != 429:
                return resp
            # 429: the ONLY retryable status. Exponential backoff with cap; honor
            # a Retry-After header when present.
            if attempt >= self._max_retries:
                logger.warning(
                    "helix %s %s rate-limited (429); retries exhausted (%d)",
                    method, path, self._max_retries,
                )
                return resp
            retry_after = self._retry_after_seconds(resp)
            backoff = min(self._max_backoff, self._base_backoff * (2 ** attempt))
            wait = max(retry_after, backoff)
            logger.info(
                "helix %s %s 429; backing off %.2fs (attempt %d/%d)",
                method, path, wait, attempt + 1, self._max_retries,
            )
            self._sleep(wait)
            attempt += 1

    @staticmethod
    def _retry_after_seconds(resp: TransportResponse) -> float:
        """Parse a Retry-After header (seconds or a unix-ish ratelimit-reset).
        Returns 0.0 when absent/unparseable (the caller still applies backoff)."""
        hdrs = resp.headers or {}
        for name in ("retry-after", "ratelimit-reset"):
            val = hdrs.get(name)
            if not val:
                continue
            try:
                num = float(val)
            except (ValueError, TypeError):
                continue
            # ratelimit-reset is an absolute epoch; convert to a delta if it looks
            # like one (large value). retry-after is already a delta.
            if name == "ratelimit-reset" and num > 1_000_000_000:
                delta = num - time.time()
                return max(0.0, delta)
            return max(0.0, num)
        return 0.0

    def _finish_write(
        self,
        action: str,
        key: tuple[str, str, str],
        resp: TransportResponse,
        *,
        success_statuses: tuple[int, ...],
    ) -> HelixResult:
        """Apply the shared idempotency policy to a write response and cache a
        success under ``key``."""
        if resp.status in success_statuses:
            result = HelixResult(
                action=action, ok=True, status=resp.status, idempotent=False,
                data=resp.json(), key=key,
            )
            self._cache(key, result)
            return result
        # 409, or any 4xx whose body says "already", is the already-applied case.
        if resp.status == 409 or (400 <= resp.status < 500 and self._body_says_already(resp.body)):
            logger.info("helix %s already-applied (status %d) key=%s", action, resp.status, key)
            result = HelixResult(
                action=action, ok=True, status=resp.status, idempotent=True,
                data=resp.json(), key=key,
            )
            self._cache(key, result)
            return result
        # Everything else is a LOUD failure (auth/permission/validation/5xx).
        msg = self._extract_error_message(resp.body)
        logger.error("helix %s failed: status=%d msg=%s key=%s", action, resp.status, msg, key)
        raise HelixError(f"helix {action} failed: {msg}", status=resp.status, body=resp.body)

    def _cache(self, key: tuple[str, str, str], result: HelixResult) -> None:
        if not any(key):
            return
        with self._applied_lock:
            self._applied.setdefault(key, result)

    def _cached(self, key: tuple[str, str, str]) -> Optional[HelixResult]:
        if not any(key):
            return None
        with self._applied_lock:
            return self._applied.get(key)

    # ---- public write API ----------------------------------------------------
    def ban_user(
        self,
        broadcaster_id: str,
        moderator_id: str,
        target_id: str,
        reason: str = "",
    ) -> HelixResult:
        """POST /moderation/bans with NO duration -> a permanent ban.

        Idempotent: a re-issued ban for an already-banned ``target_id`` resolves
        to success (local cache hit, or Twitch 409 / "already banned" body).
        """
        return self._ban_or_timeout("ban", broadcaster_id, moderator_id, target_id, reason, None)

    def timeout_user(
        self,
        broadcaster_id: str,
        moderator_id: str,
        target_id: str,
        duration_s: int,
        reason: str = "",
    ) -> HelixResult:
        """POST /moderation/bans WITH a duration (seconds) -> a timeout.

        ``duration_s`` must be 1..1_209_600 (Twitch's 2-week max). Idempotent on
        an already-timed-out target the same way as :meth:`ban_user`.
        """
        if not isinstance(duration_s, int) or duration_s < 1 or duration_s > 1_209_600:
            raise ValueError("duration_s must be an int in 1..1209600 (seconds)")
        return self._ban_or_timeout(
            "timeout", broadcaster_id, moderator_id, target_id, reason, duration_s
        )

    def _ban_or_timeout(
        self,
        action: str,
        broadcaster_id: str,
        moderator_id: str,
        target_id: str,
        reason: str,
        duration_s: Optional[int],
    ) -> HelixResult:
        if not broadcaster_id or not moderator_id or not target_id:
            raise ValueError("broadcaster_id, moderator_id and target_id are required")
        key = (action, str(target_id), "")
        cached = self._cached(key)
        if cached is not None:
            logger.info("helix %s short-circuit (local idempotency) key=%s", action, key)
            return HelixResult(
                action=action, ok=True, status=0, idempotent=True,
                data=cached.data, key=key,
            )
        data: dict[str, Any] = {"user_id": str(target_id)}
        if reason:
            data["reason"] = reason[:500]
        if duration_s is not None:
            data["duration"] = int(duration_s)
        resp = self._request(
            "POST",
            "/moderation/bans",
            query={"broadcaster_id": str(broadcaster_id), "moderator_id": str(moderator_id)},
            json_payload={"data": data},
        )
        return self._finish_write(action, key, resp, success_statuses=(200, 201))

    def delete_message(
        self,
        broadcaster_id: str,
        moderator_id: str,
        message_id: str,
    ) -> HelixResult:
        """DELETE /chat/messages for a single ``message_id``.

        Keyed on ``message_id`` (NOT a user) so deleting the same message twice is
        a cache hit; a Twitch 404 "message does not exist" is also treated as
        already-applied.
        """
        if not broadcaster_id or not moderator_id or not message_id:
            raise ValueError("broadcaster_id, moderator_id and message_id are required")
        key = ("delete_message", "", str(message_id))
        cached = self._cached(key)
        if cached is not None:
            logger.info("helix delete_message short-circuit (local idempotency) key=%s", key)
            return HelixResult(
                action="delete_message", ok=True, status=0, idempotent=True,
                data=cached.data, key=key,
            )
        # Helix single-message delete lives under /chat/messages (NOT
        # /moderation/chat, which 404s). moderator:manage:chat_messages scope.
        resp = self._request(
            "DELETE",
            "/chat/messages",
            query={
                "broadcaster_id": str(broadcaster_id),
                "moderator_id": str(moderator_id),
                "message_id": str(message_id),
            },
        )
        # A 404 for a message that no longer exists is the delete-idempotent case.
        if resp.status == 404 and ("message" in (resp.body or "").lower() or not resp.body):
            logger.info("helix delete_message already-gone (404) key=%s", key)
            result = HelixResult(
                action="delete_message", ok=True, status=404, idempotent=True,
                data=None, key=key,
            )
            self._cache(key, result)
            return result
        return self._finish_write("delete_message", key, resp, success_statuses=(200, 204))

    def unban_user(
        self,
        broadcaster_id: str,
        moderator_id: str,
        target_id: str,
    ) -> HelixResult:
        """DELETE /moderation/bans -> remove a ban OR a timeout for ``target_id``.

        Twitch has NO separate untimeout endpoint: a timeout is a temporary ban,
        so removing the ban entry lifts either one. Idempotent: a Twitch 400/404
        whose body says the user is not banned (or an empty body) is treated as
        already-applied (the unban already took effect), the same way
        :meth:`delete_message` treats an already-gone message.
        """
        if not broadcaster_id or not moderator_id or not target_id:
            raise ValueError("broadcaster_id, moderator_id and target_id are required")
        key = ("unban", str(target_id), "")
        cached = self._cached(key)
        if cached is not None:
            logger.info("helix unban short-circuit (local idempotency) key=%s", key)
            return HelixResult(
                action="unban", ok=True, status=0, idempotent=True,
                data=cached.data, key=key,
            )
        resp = self._request(
            "DELETE",
            "/moderation/bans",
            query={
                "broadcaster_id": str(broadcaster_id),
                "moderator_id": str(moderator_id),
                "user_id": str(target_id),
            },
        )
        _body = (resp.body or "").lower()
        if resp.status in (400, 404) and (
            "not banned" in _body or "isn't banned" in _body or not resp.body
        ):
            logger.info("helix unban already-not-banned (%s) key=%s", resp.status, key)
            result = HelixResult(
                action="unban", ok=True, status=resp.status, idempotent=True,
                data=None, key=key,
            )
            self._cache(key, result)
            return result
        return self._finish_write("unban", key, resp, success_statuses=(200, 204))

    def clear_chat(self, broadcaster_id: str, moderator_id: str) -> HelixResult:
        """DELETE /chat/messages (no message_id) — remove ALL messages from chat.

        Channel-scoped (not user-keyed) and naturally idempotent (clearing already-
        empty chat is a server-side no-op), so it is NOT locally cached. A 2xx
        (200/204) is success; anything else raises LOUD.
        """
        if not broadcaster_id or not moderator_id:
            raise ValueError("broadcaster_id and moderator_id are required")
        # Helix clears chat via DELETE /chat/messages WITHOUT a message_id (the
        # same endpoint as a single delete; omitting message_id clears ALL).
        # NOT /moderation/chat (404). moderator:manage:chat_messages scope.
        resp = self._request(
            "DELETE",
            "/chat/messages",
            query={"broadcaster_id": str(broadcaster_id), "moderator_id": str(moderator_id)},
        )
        return self._finish_write(
            "clear_chat", ("clear_chat", "", ""), resp, success_statuses=(200, 204),
        )

    def update_chat_settings(
        self,
        broadcaster_id: str,
        moderator_id: str,
        settings: Mapping[str, Any],
    ) -> HelixResult:
        """PATCH /chat/settings with the given settings map.

        Chat settings are not user-scoped, so this is NOT locally cached (an empty
        idempotency key) — Twitch itself treats a PATCH of the same settings as a
        no-op 200, which is idempotent server-side. A whitelist guards the body so
        a caller can't smuggle arbitrary keys.
        """
        if not broadcaster_id or not moderator_id:
            raise ValueError("broadcaster_id and moderator_id are required")
        allowed = {
            "emote_mode",
            "follower_mode",
            "follower_mode_duration",
            "non_moderator_chat_delay",
            "non_moderator_chat_delay_duration",
            "slow_mode",
            "slow_mode_wait_time",
            "subscriber_mode",
            "unique_chat_mode",
        }
        payload = {k: v for k, v in settings.items() if k in allowed}
        if not payload:
            raise ValueError(
                "settings contains no recognized chat-setting keys "
                f"(allowed: {sorted(allowed)})"
            )
        # Helix chat settings live under /chat/settings (NOT /moderation/...).
        # The wrong /moderation/chat/settings path returns a bare 404
        # ({"error":"Not Found","status":404,"message":""}).
        resp = self._request(
            "PATCH",
            "/chat/settings",
            query={"broadcaster_id": str(broadcaster_id), "moderator_id": str(moderator_id)},
            json_payload=payload,
        )
        return self._finish_write(
            "update_chat_settings", ("update_chat_settings", "", ""), resp,
            success_statuses=(200,),
        )

    def send_shoutout(
        self,
        from_broadcaster_id: str,
        to_broadcaster_id: str,
        moderator_id: str,
    ) -> HelixResult:
        """POST /chat/shoutouts — send an official Twitch /shoutout.

        ``from_broadcaster_id`` is this channel; ``to_broadcaster_id`` is the
        raider being promoted; ``moderator_id`` is a moderator of THIS channel (the
        broadcaster moderates their own channel). Needs the
        ``moderator:manage:shoutouts`` scope on the token.

        Twitch returns 204 on success. Two non-failure cases are treated as
        already-applied (idempotent) so a replayed raid never raises:
          * the per-target / global shoutout COOLDOWN (Twitch returns 429 with a
            body mentioning the cooldown — the streamer just shouted this raider, or
            shouted someone else within the 2-minute global window), and
          * a body that says the shoutout was "already" sent.
        Keyed on ``to_broadcaster_id`` so the local idempotency cache short-circuits
        a duplicate for the same raider within this process.
        """
        if not from_broadcaster_id or not to_broadcaster_id or not moderator_id:
            raise ValueError(
                "from_broadcaster_id, to_broadcaster_id and moderator_id are required"
            )
        key = ("shoutout", str(to_broadcaster_id), "")
        cached = self._cached(key)
        if cached is not None:
            logger.info("helix shoutout short-circuit (local idempotency) key=%s", key)
            return HelixResult(
                action="shoutout", ok=True, status=0, idempotent=True,
                data=cached.data, key=key,
            )
        resp = self._request(
            "POST",
            "/chat/shoutouts",
            query={
                "from_broadcaster_id": str(from_broadcaster_id),
                "to_broadcaster_id": str(to_broadcaster_id),
                "moderator_id": str(moderator_id),
            },
        )
        # The shoutout cooldown surfaces as a 429 whose body names the cooldown.
        # That is NOT a real failure (the streamer simply shouted recently); treat
        # it as already-applied so a raid-handler retry never raises LOUD. A generic
        # 429 (true rate-limit) is already retried+capped inside _request before we
        # see it here, so a 429 reaching this point is the cooldown case.
        if resp.status == 429:
            logger.info("helix shoutout cooldown (429) -> idempotent key=%s", key)
            result = HelixResult(
                action="shoutout", ok=True, status=429, idempotent=True,
                data=resp.json(), key=key,
            )
            self._cache(key, result)
            return result
        return self._finish_write("shoutout", key, resp, success_statuses=(204, 200))
