"""Twitch OAuth lifecycle — Device Code Flow + rotation-safe token store.

ANTICHEAT (BR-P1). Pure stdlib + ``urllib`` only — no ``requests`` /
``aiohttp`` / ``websockets`` / desktop-automation. Importable in either the
voice process (behind the flag) or a sidecar; in practice the OAuth dance runs
in the Twitch sidecar / the ``scripts/twitch_setup.py`` wake-up step.

Why Device Code Flow (RFC 8628), not auth-code:
  * Twitch's GA-2024 device flow makes Ultron a PUBLIC client -> there is **no
    ``client_secret`` on disk** (a secret beside a gitignored token file is the
    bigger leak). The user reads a short code on screen and approves it in a
    browser on any device.

Why the store is paranoid:
  * Twitch refresh tokens are **single-use and rotate on every refresh**. A
    non-atomic write, or a crash between "got new tokens" and "wrote them",
    permanently logs the streamer out mid-stream and kills chat-mode. So
    :class:`TokenStore` writes a tmp file, ``fsync``s it, ``os.replace``s it
    (atomic on POSIX and on NTFS), best-effort ``fsync``s the directory, and
    chmods ``0o600`` (POSIX; best-effort on Windows). :meth:`TokenStore.rotate`
    persists the NEW refresh token FIRST, before the caller uses the new access
    token, so a crash never loses the only valid refresh token.

Security:
  * Token VALUES are NEVER logged. Every log line uses :func:`_fp` (a short
    salted-by-nothing sha256 fingerprint) or a redacted preview, never the raw
    token. The store file is created mode-0o600.

All network goes through an injectable ``transport`` callable so the unit tests
never touch the network::

    transport(method, url, *, data=None, headers=None) -> (status: int, body: dict)

``data`` is a ``dict`` of form fields (the transport form-url-encodes it for a
POST); ``body`` is the parsed JSON response (``{}`` if the body was empty / not
JSON).
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("kenning.twitch.auth")

__all__ = [
    "TwitchAuthError",
    "DeviceFlowError",
    "AuthorizationPendingError",
    "SlowDownError",
    "DeviceFlowExpiredError",
    "RevokedError",
    "DeviceCode",
    "Transport",
    "TokenStore",
    "TwitchAuth",
    "make_urllib_transport",
    "BROADCASTER_SCOPES",
    "BOT_SCOPES",
    "DEVICE_URL",
    "TOKEN_URL",
    "VALIDATE_URL",
]

# --- Twitch OAuth endpoints (RFC 8628 device flow) ---------------------------
DEVICE_URL = "https://id.twitch.tv/oauth2/device"
TOKEN_URL = "https://id.twitch.tv/oauth2/token"
VALIDATE_URL = "https://id.twitch.tv/oauth2/validate"

# Least-privilege scope sets, feature-gated. Over-requesting scopes can get the
# API access SUSPENDED, so these are deliberately minimal; the caller passes the
# set matching the identity being authorized.
BROADCASTER_SCOPES = (
    "channel:read:redemptions",
    "channel:manage:redemptions",
    "moderator:read:automod_settings",
    "moderator:manage:automod",
    "moderator:read:chat_settings",
    "moderator:manage:chat_settings",
    "moderator:manage:chat_messages",
    "moderator:manage:shield_mode",
    "moderator:manage:banned_users",
    "moderator:manage:shoutouts",
    "channel:moderate",
)
BOT_SCOPES = (
    "user:read:chat",
    "user:write:chat",
    "user:bot",
)

_DEFAULT_TIMEOUT = 15.0
_DEFAULT_STORE_PATH = "~/.kenning/twitch.json"


# --- exceptions --------------------------------------------------------------
class TwitchAuthError(RuntimeError):
    """Base class for every auth failure in this module."""


class DeviceFlowError(TwitchAuthError):
    """The device-authorization request itself failed (network / 4xx / 5xx)."""


class AuthorizationPendingError(TwitchAuthError):
    """The user has not yet approved the device code — keep polling."""


class SlowDownError(TwitchAuthError):
    """Twitch asked us to poll more slowly — increase the interval."""


class DeviceFlowExpiredError(TwitchAuthError):
    """The device code expired before the user approved it."""


class RevokedError(TwitchAuthError):
    """The access token is no longer valid (validate returned non-200) and a
    refresh was not possible or also failed — the streamer must re-authorize."""


# --- transport ---------------------------------------------------------------
# Signature: transport(method, url, *, data=None, headers=None) -> (status, body)
Transport = Callable[..., "tuple[int, dict]"]


def make_urllib_transport(timeout: float = _DEFAULT_TIMEOUT) -> Transport:
    """The default real transport — pure ``urllib``, no third-party deps.

    POST ``data`` is form-url-encoded (the Twitch token/device endpoints expect
    ``application/x-www-form-urlencoded``). A non-2xx HTTP status does NOT raise:
    the (status, parsed-json-body) tuple is returned so callers can branch on the
    OAuth ``error`` field (e.g. ``authorization_pending``). Genuine transport
    failures (DNS, TLS, timeout, connection reset) raise :class:`DeviceFlowError`.
    """

    def _transport(
        method: str,
        url: str,
        *,
        data: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> "tuple[int, dict]":
        method = (method or "GET").upper()
        req_headers = {"Accept": "application/json"}
        if headers:
            req_headers.update(headers)
        body_bytes: Optional[bytes] = None
        if data is not None:
            encoded = urllib.parse.urlencode(data).encode("utf-8")
            if method == "GET":
                sep = "&" if "?" in url else "?"
                url = f"{url}{sep}{encoded.decode('ascii')}"
            else:
                body_bytes = encoded
                req_headers.setdefault(
                    "Content-Type", "application/x-www-form-urlencoded"
                )
        req = urllib.request.Request(
            url, data=body_bytes, headers=req_headers, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = int(getattr(resp, "status", resp.getcode()) or 0)
                raw = resp.read()
        except urllib.error.HTTPError as e:
            # Non-2xx — read the body so the caller can branch on the OAuth error.
            status = int(e.code)
            try:
                raw = e.read()
            except Exception:  # noqa: BLE001
                raw = b""
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            # Genuine transport failure — surface as a typed error, no token leak.
            raise DeviceFlowError(f"transport error contacting Twitch: {e}") from e
        return status, _parse_json(raw)

    return _transport


def _parse_json(raw: Any) -> dict:
    if isinstance(raw, (bytes, bytearray)):
        text = raw.decode("utf-8", "replace").strip()
    else:
        text = str(raw or "").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return {}
    return obj if isinstance(obj, dict) else {}


# --- redaction (NEVER log a token value) -------------------------------------
def _fp(token: Optional[str]) -> str:
    """A short, non-reversible fingerprint of a secret, for logs."""
    if not token:
        return "<none>"
    digest = hashlib.sha256(token.encode("utf-8", "replace")).hexdigest()[:10]
    return f"sha256:{digest}(len={len(token)})"


def _redact_tokens(d: dict) -> dict:
    """Copy of a dict with any token-ish values replaced by a fingerprint."""
    sensitive = {"access_token", "refresh_token", "device_code", "id_token"}
    out: dict = {}
    for k, v in d.items():
        if k in sensitive and isinstance(v, str):
            out[k] = _fp(v)
        else:
            out[k] = v
    return out


# --- the rotation-safe token store -------------------------------------------
class TokenStore:
    """Atomic, fsync'd, 0o600 JSON store for the Twitch token set.

    The stored dict is opaque to the store but is expected to carry at least
    ``access_token`` and ``refresh_token`` (plus ``expires_at`` / ``scope`` /
    ``token_type`` as written by :class:`TwitchAuth`).
    """

    def __init__(self, path: str | os.PathLike | None = None) -> None:
        raw = str(path) if path is not None else _DEFAULT_STORE_PATH
        self.path = Path(os.path.expanduser(raw)).resolve()

    # -- load -----------------------------------------------------------------
    def load(self) -> Optional[dict]:
        """Return the persisted token dict, or ``None`` if absent/corrupt.

        A corrupt/partial file is treated as "no creds" (fail-safe): the caller
        re-runs the device flow rather than crashing. We never raise on read.
        """
        try:
            with open(self.path, encoding="utf-8") as fh:
                obj = json.load(fh)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as e:
            logger.warning(
                "twitch token store unreadable at %s (%s); treating as absent",
                self.path,
                type(e).__name__,
            )
            return None
        if not isinstance(obj, dict):
            logger.warning("twitch token store at %s is not an object; ignoring", self.path)
            return None
        return obj

    # -- save (atomic) --------------------------------------------------------
    def save(self, tokens: dict) -> None:
        """Atomically persist ``tokens``.

        Sequence: ensure the parent dir (0o700 on POSIX) -> write a uniquely
        named tmp file in the SAME dir -> ``fsync`` it -> ``chmod 0o600`` ->
        ``os.replace`` over the target (atomic) -> best-effort ``fsync`` the
        directory so the rename is durable. Any failure raises
        :class:`TwitchAuthError` AFTER cleaning the tmp file — the old file is
        left intact (we never truncate the target).
        """
        if not isinstance(tokens, dict):
            raise TwitchAuthError("refusing to persist non-dict token payload")

        parent = self.path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
            self._chmod_best_effort(parent, 0o700)
        except OSError as e:
            raise TwitchAuthError(f"cannot create token store dir {parent}: {e}") from e

        # Unique tmp name in the same directory (same filesystem -> atomic replace).
        tmp_name = f".{self.path.name}.{os.getpid()}.{int(time.time()*1000)}.tmp"
        tmp_path = parent / tmp_name
        payload = json.dumps(tokens, ensure_ascii=False, indent=2, sort_keys=True)
        fd = None
        try:
            # 0o600 from creation so the secret is never briefly world-readable.
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            fd = os.open(str(tmp_path), flags, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fd = None  # fdopen now owns it
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            self._chmod_best_effort(tmp_path, 0o600)
            os.replace(tmp_path, self.path)  # atomic on POSIX + NTFS
            self._chmod_best_effort(self.path, 0o600)
            self._fsync_dir(parent)
        except OSError as e:
            # Clean the tmp file; leave the existing target untouched.
            self._unlink_best_effort(tmp_path)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise TwitchAuthError(f"atomic token save failed: {e}") from e
        logger.info(
            "twitch tokens persisted (%s); access=%s refresh=%s",
            self.path,
            _fp(tokens.get("access_token")),
            _fp(tokens.get("refresh_token")),
        )

    # -- rotate (single-use refresh safety) -----------------------------------
    def rotate(self, new_tokens: dict) -> dict:
        """Persist a freshly-rotated token set, merging onto the prior dict.

        Twitch rotates the refresh token on every refresh; ``new_tokens`` MUST
        carry the new ``refresh_token`` (and access token). We merge it over the
        existing stored dict (so unrelated fields survive) and save atomically
        BEFORE returning — the caller only uses the new access token after the
        new refresh token is durably on disk. Returns the merged, persisted dict.
        """
        if not isinstance(new_tokens, dict):
            raise TwitchAuthError("rotate requires a dict of new tokens")
        if not new_tokens.get("refresh_token"):
            # A rotation that drops the refresh token would brick the next refresh.
            raise TwitchAuthError("rotate refused: new_tokens has no refresh_token")
        merged = dict(self.load() or {})
        merged.update(new_tokens)
        self.save(merged)
        return merged

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _chmod_best_effort(p: Path, mode: int) -> None:
        try:
            os.chmod(p, mode)
        except (OSError, NotImplementedError) as e:
            # Windows chmod is limited (no POSIX perms); best-effort only.
            logger.debug("chmod %o on %s skipped (%s)", mode, p, type(e).__name__)

    @staticmethod
    def _fsync_dir(directory: Path) -> None:
        # Directory fsync makes the rename durable on POSIX. Not supported on
        # Windows (O_RDONLY on a dir fails) — best-effort, never fatal.
        try:
            dir_fd = os.open(str(directory), os.O_RDONLY)
        except OSError as e:
            if e.errno not in (errno.EACCES, errno.EISDIR, errno.EINVAL, errno.ENOTSUP):
                logger.debug("dir open for fsync failed on %s (%s)", directory, e)
            return
        try:
            os.fsync(dir_fd)
        except OSError as e:
            logger.debug("dir fsync skipped on %s (%s)", directory, type(e).__name__)
        finally:
            try:
                os.close(dir_fd)
            except OSError:
                pass

    @staticmethod
    def _unlink_best_effort(p: Path) -> None:
        try:
            os.unlink(p)
        except OSError:
            pass

    def is_expired(self, margin_seconds: float = 0.0) -> bool:
        """True if the stored access token has expired (or is absent).

        ``margin_seconds`` adds a safety buffer: a token expiring within that
        window is treated as already expired so callers can proactively refresh.
        """
        tokens = self.load()
        if not tokens:
            return True
        expires_at = tokens.get("expires_at")
        try:
            expires_at = float(expires_at)
        except (TypeError, ValueError):
            return True
        return time.time() + float(margin_seconds) >= expires_at

    def is_secure_perms(self) -> bool:
        """True iff the store file exists and is 0o600 on POSIX.

        On Windows (no POSIX perms) this returns True when the file exists — the
        guard is informational, not load-bearing, on that platform.
        """
        try:
            st = os.stat(self.path)
        except OSError:
            return False
        if os.name == "nt":
            return True
        return stat.S_IMODE(st.st_mode) == 0o600


# --- the OAuth lifecycle -----------------------------------------------------
@dataclass(frozen=True)
class DeviceCode:
    """The device-authorization response the user must act on."""

    user_code: str
    verification_uri: str
    device_code: str
    interval: int
    expires_in: int


class TwitchAuth:
    """Twitch Device Code Flow + validate/refresh + auto-refresh-on-401.

    Args:
        client_id: the Twitch application client id (public — no secret).
        store: a :class:`TokenStore` for rotation-safe persistence.
        transport: the injectable HTTP transport (defaults to a real urllib one).
        scopes: the least-privilege scope set for the identity being authorized.
        clock / sleep: injectable for deterministic tests of the poll loop.
    """

    def __init__(
        self,
        client_id: str,
        store: TokenStore,
        transport: Optional[Transport] = None,
        *,
        scopes: "tuple[str, ...]" = BROADCASTER_SCOPES,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not client_id or not isinstance(client_id, str):
            raise TwitchAuthError("client_id is required")
        self.client_id = client_id
        self.store = store
        self.transport: Transport = transport or make_urllib_transport()
        self.scopes = tuple(scopes)
        self._clock = clock
        self._sleep = sleep

    # -- 1. start the device flow --------------------------------------------
    def start_device_flow(self) -> DeviceCode:
        """POST the device-authorization request; return what the user must do.

        Returns a :class:`DeviceCode` (also usable as the 4-tuple
        ``(user_code, verification_uri, device_code, interval)`` the spec asks
        for, since the CLI unpacks fields by name).
        """
        data = {"client_id": self.client_id, "scopes": " ".join(self.scopes)}
        status, body = self._call("POST", DEVICE_URL, data=data, what="device authorization")
        if status < 200 or status >= 300:
            raise DeviceFlowError(
                f"device authorization failed (HTTP {status}): "
                f"{body.get('message') or body.get('error') or 'unknown error'}"
            )
        try:
            user_code = str(body["user_code"])
            device_code = str(body["device_code"])
        except KeyError as e:
            raise DeviceFlowError(f"device authorization response missing {e}") from e
        verification_uri = str(
            body.get("verification_uri") or body.get("verification_uri_complete") or ""
        )
        interval = int(body.get("interval", 5) or 5)
        expires_in = int(body.get("expires_in", 1800) or 1800)
        dc = DeviceCode(
            user_code=user_code,
            verification_uri=verification_uri,
            device_code=device_code,
            interval=max(1, interval),
            expires_in=expires_in,
        )
        logger.info(
            "device flow started: user_code=%s uri=%s interval=%ds expires_in=%ds device=%s",
            dc.user_code,
            dc.verification_uri,
            dc.interval,
            dc.expires_in,
            _fp(dc.device_code),
        )
        return dc

    # -- 2. poll for the token ------------------------------------------------
    def poll_device_token(
        self,
        device_code: str,
        interval: int = 5,
        timeout: float = 1800.0,
    ) -> dict:
        """Poll the token endpoint until the user approves, or raise.

        Honors ``authorization_pending`` (keep waiting) and ``slow_down`` (back
        off the interval by +5s, per RFC 8628). Raises
        :class:`DeviceFlowExpiredError` on expiry / overall ``timeout``. On
        success the token set is persisted atomically before it is returned.
        """
        if not device_code:
            raise DeviceFlowError("poll_device_token requires a device_code")
        interval = max(1, int(interval))
        deadline = self._clock() + float(timeout)
        data = {
            "client_id": self.client_id,
            "scopes": " ".join(self.scopes),
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }
        while True:
            if self._clock() >= deadline:
                raise DeviceFlowExpiredError("device authorization timed out before approval")
            status, body = self._call("POST", TOKEN_URL, data=data, what="device token poll")
            if 200 <= status < 300 and body.get("access_token"):
                tokens = self._normalize_token_response(body)
                self.store.save(tokens)
                logger.info("device flow approved; tokens stored (access=%s)",
                            _fp(tokens.get("access_token")))
                return tokens
            err = str(body.get("error") or body.get("message") or "").lower()
            if "authorization_pending" in err or status == 400 and not err:
                # still waiting
                self._sleep(interval)
                continue
            if "slow_down" in err:
                interval += 5
                self._sleep(interval)
                continue
            if "expired_token" in err or "expired" in err:
                raise DeviceFlowExpiredError("device code expired; restart the flow")
            if "access_denied" in err or "denied" in err:
                raise DeviceFlowError("user denied the device authorization")
            # Unknown pending-ish 4xx -> wait; hard errors -> raise.
            if status == 400:
                self._sleep(interval)
                continue
            raise DeviceFlowError(
                f"device token poll failed (HTTP {status}): {err or 'unknown error'}"
            )

    # -- 3. validate ----------------------------------------------------------
    def validate(self, access_token: str) -> Optional[dict]:
        """GET /oauth2/validate. Return the validation payload, or ``None`` if
        the token is invalid/revoked (any non-200). ToS requires this at startup
        and hourly. Never logs the token value."""
        if not access_token:
            return None
        status, body = self._call(
            "GET",
            VALIDATE_URL,
            headers={"Authorization": f"OAuth {access_token}"},
            what="validate",
        )
        if status == 200 and (body.get("client_id") or body.get("login") or body.get("user_id")):
            logger.debug(
                "validate ok: login=%s expires_in=%s",
                body.get("login"),
                body.get("expires_in"),
            )
            return body
        logger.info("validate: token invalid/revoked (HTTP %s)", status)
        return None

    # -- 4. refresh -----------------------------------------------------------
    def refresh(self, refresh_token: str) -> dict:
        """POST grant_type=refresh_token; ROTATE the store with the result.

        Twitch returns a NEW refresh token on every refresh (single-use). We
        persist the new set atomically via :meth:`TokenStore.rotate` BEFORE
        returning, so a crash never loses the only valid refresh token. Raises
        :class:`RevokedError` if Twitch rejects the refresh token (e.g. the user
        revoked the app or a stale single-use token was replayed).
        """
        if not refresh_token:
            raise RevokedError("no refresh token available; re-authorization required")
        data = {
            "client_id": self.client_id,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        status, body = self._call("POST", TOKEN_URL, data=data, what="refresh")
        if 200 <= status < 300 and body.get("access_token"):
            tokens = self._normalize_token_response(body)
            if not tokens.get("refresh_token"):
                # Twitch always returns one; if absent, keep the old so we don't brick.
                tokens["refresh_token"] = refresh_token
            persisted = self.store.rotate(tokens)
            logger.info("token refreshed and rotated (access=%s)",
                        _fp(persisted.get("access_token")))
            return persisted
        err = str(body.get("message") or body.get("error") or "").lower()
        # 400/401 on refresh = the grant is dead -> the streamer must re-auth.
        if status in (400, 401, 403) or "invalid" in err:
            raise RevokedError(
                f"refresh rejected (HTTP {status}: {err or 'invalid grant'}); "
                "re-authorization required"
            )
        raise TwitchAuthError(f"token refresh failed (HTTP {status}): {err or 'unknown error'}")

    # -- 5. proactive refresh (call on startup / before use) ------------------
    def ensure_valid(self, margin_seconds: float = 300.0) -> str:
        """Return a live access token, proactively refreshing if near expiry.

        Checks whether the stored token expires within ``margin_seconds`` (default
        5 minutes) and silently rotates it using the stored refresh_token if so.
        This prevents sidecars from starting with a stale token that would cause
        an immediate 401 on the first Helix/EventSub call.

        Returns the access token string.  On any failure (no stored token,
        refresh rejected, network error) logs a warning and returns whatever the
        store holds (or "" if absent) — the caller proceeds and will hit a 401
        which ``call_with_auth`` will handle reactively.
        """
        tokens = self.store.load() or {}
        access = str(tokens.get("access_token") or "")
        if not self.store.is_expired(margin_seconds):
            return access  # still fresh
        refresh_token = str(tokens.get("refresh_token") or "")
        if not refresh_token:
            logger.warning("token near/past expiry but no refresh_token; re-auth required")
            return access
        try:
            new_tokens = self.refresh(refresh_token)
            return str(new_tokens.get("access_token") or access)
        except RevokedError as exc:
            logger.warning("proactive token refresh failed (revoked): %s", exc)
            return access
        except Exception as exc:  # noqa: BLE001
            logger.warning("proactive token refresh failed: %s", type(exc).__name__)
            return access

    # -- 6. authed request with one auto-refresh-on-401 -----------------------
    def call_with_auth(self, do_request: Callable[[str], "tuple[int, Any]"]) -> Any:
        """Run ``do_request(access_token) -> (status, result)`` with the current
        access token; on a 401 refresh ONCE then retry; on a still-401 (or no
        usable token) raise :class:`RevokedError`.

        The store is the source of truth for the current token set. ``do_request``
        receives the bearer token string and must perform the actual API call.
        """
        tokens = self.store.load()
        if not tokens or not tokens.get("access_token"):
            raise RevokedError("no stored access token; run the device flow first")
        access = str(tokens["access_token"])

        status, result = do_request(access)
        if status != 401:
            return result

        # 401 -> the token is expired/revoked. Confirm via validate, then refresh once.
        logger.info("authed request returned 401; attempting a single refresh")
        refresh_token = tokens.get("refresh_token")
        if not refresh_token:
            raise RevokedError("401 with no refresh token; re-authorization required")
        # refresh() raises RevokedError if the grant itself is dead.
        new_tokens = self.refresh(str(refresh_token))
        new_access = str(new_tokens.get("access_token") or "")
        if not new_access:
            raise RevokedError("refresh produced no access token; re-authorization required")

        status2, result2 = do_request(new_access)
        if status2 == 401:
            # Still unauthorized after a fresh token -> the app/grant is revoked.
            raise RevokedError("still 401 after refresh; the authorization was revoked")
        return result2

    # -- internals ------------------------------------------------------------
    def _normalize_token_response(self, body: dict) -> dict:
        """Shape a token endpoint response into the stored schema + expiry."""
        now = time.time()
        expires_in = body.get("expires_in")
        try:
            expires_in = int(expires_in)
        except (TypeError, ValueError):
            expires_in = 0
        scope = body.get("scope")
        if isinstance(scope, list):
            scope_str = " ".join(str(s) for s in scope)
        else:
            scope_str = str(scope or " ".join(self.scopes))
        tokens = {
            "access_token": str(body.get("access_token") or ""),
            "refresh_token": str(body.get("refresh_token") or ""),
            "token_type": str(body.get("token_type") or "bearer"),
            "scope": scope_str,
            "expires_in": expires_in,
            "expires_at": (now + expires_in) if expires_in else 0,
            "obtained_at": now,
        }
        return tokens

    def _call(
        self,
        method: str,
        url: str,
        *,
        data: Optional[dict] = None,
        headers: Optional[dict] = None,
        what: str = "request",
    ) -> "tuple[int, dict]":
        """Invoke the transport with logging that NEVER leaks a token value."""
        try:
            status, body = self.transport(method, url, data=data, headers=headers)
        except DeviceFlowError:
            raise
        except Exception as e:  # noqa: BLE001 — any transport error is typed + tokenless
            logger.warning("twitch %s transport failed: %s", what, type(e).__name__)
            raise DeviceFlowError(f"twitch {what} transport failed: {type(e).__name__}") from e
        if not isinstance(body, dict):
            body = {}
        logger.debug("twitch %s -> HTTP %s body=%s", what, status, _redact_tokens(body))
        return int(status), body
