"""Spotify OAuth: credential loading + authorization-code flow + refresh.

Credentials live in a gitignored JSON file OUTSIDE the repo (default
``~/.ultron/spotify.json``) -- never tracked. Playback control needs the
authorization-code grant (a one-time browser consent that yields a
long-lived refresh token, persisted back to the same file); from then
on :meth:`SpotifyAuth.access_token` silently refreshes short-lived
access tokens as needed.

All network calls go through an injectable ``post_fn`` so the unit
tests never hit Spotify.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("ultron.spotify.auth")

__all__ = [
    "SpotifyCredentials",
    "SpotifyAuthError",
    "SpotifyAuth",
    "DEFAULT_SCOPES",
    "build_authorize_url",
    "load_credentials",
    "save_refresh_token",
]

TOKEN_URL = "https://accounts.spotify.com/api/token"
AUTHORIZE_URL = "https://accounts.spotify.com/authorize"

# Scopes for full playback control + read state (Premium required for
# the modify-playback ones).
DEFAULT_SCOPES = (
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
    "playlist-read-private",
    "playlist-read-collaborative",
    "user-library-read",
    "user-library-modify",
    "user-read-recently-played",
)


class SpotifyAuthError(RuntimeError):
    """Raised when credentials are missing or a token request fails."""


@dataclass
class SpotifyCredentials:
    """App credentials + the persisted refresh token.

    Attributes:
        client_id: Spotify app client id.
        client_secret: Spotify app client secret (sensitive).
        redirect_uri: redirect URI registered in the Spotify dashboard.
        refresh_token: long-lived token from the auth-code grant (None
            until ``scripts/spotify_setup.py`` runs once).
        path: the gitignored file these were loaded from (for saving).
    """

    client_id: str
    client_secret: str
    redirect_uri: str
    refresh_token: Optional[str] = None
    path: Optional[Path] = None


def load_credentials(path: str | Path) -> SpotifyCredentials:
    """Load credentials from the gitignored JSON file.

    Raises:
        SpotifyAuthError: file missing / unreadable / lacks client id.
    """
    p = Path(path).expanduser()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise SpotifyAuthError(
            f"Spotify credentials not found at {p}. Create it with your "
            "client_id / client_secret / redirect_uri."
        ) from e
    except (OSError, ValueError) as e:
        raise SpotifyAuthError(f"Spotify credentials unreadable: {e}") from e
    cid = str(data.get("client_id", "")).strip()
    if not cid:
        raise SpotifyAuthError(f"Spotify credentials at {p} have no client_id")
    return SpotifyCredentials(
        client_id=cid,
        client_secret=str(data.get("client_secret", "")).strip(),
        redirect_uri=str(
            data.get("redirect_uri", "http://127.0.0.1:8899/callback")
        ).strip(),
        refresh_token=(data.get("refresh_token") or None),
        path=p,
    )


def save_refresh_token(path: str | Path, refresh_token: str) -> None:
    """Persist the refresh token back into the gitignored file,
    preserving every other field + comment."""
    p = Path(path).expanduser()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data["refresh_token"] = refresh_token
    p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _basic_auth_header(creds: SpotifyCredentials) -> str:
    raw = f"{creds.client_id}:{creds.client_secret}".encode("ascii")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def build_authorize_url(
    creds: SpotifyCredentials,
    *,
    scopes: tuple[str, ...] = DEFAULT_SCOPES,
    state: str = "ultron",
) -> str:
    """Build the consent URL the user opens once in a browser."""
    params = {
        "client_id": creds.client_id,
        "response_type": "code",
        "redirect_uri": creds.redirect_uri,
        "scope": " ".join(scopes),
        "state": state,
    }
    return AUTHORIZE_URL + "?" + urllib.parse.urlencode(params)


# A ``post_fn`` matches ``requests.post``: (url, data=..., headers=...)
# -> object with ``.status_code`` and ``.json()``.
PostFn = Callable[..., Any]


def _default_post() -> PostFn:
    import requests

    return requests.post


def exchange_code(
    creds: SpotifyCredentials,
    code: str,
    *,
    post_fn: Optional[PostFn] = None,
) -> dict:
    """Exchange an authorization code for tokens (one-time setup).

    Returns the token payload (contains ``refresh_token`` +
    ``access_token``). Raises :class:`SpotifyAuthError` on failure.
    """
    post = post_fn or _default_post()
    resp = post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": creds.redirect_uri,
        },
        headers={"Authorization": _basic_auth_header(creds)},
        timeout=15,
    )
    if getattr(resp, "status_code", 500) != 200:
        raise SpotifyAuthError(
            f"token exchange failed ({getattr(resp, 'status_code', '?')}): "
            f"{_safe_body(resp)}"
        )
    return resp.json()


def _safe_body(resp: Any) -> str:
    try:
        return json.dumps(resp.json())[:200]
    except Exception:  # noqa: BLE001
        return str(getattr(resp, "text", ""))[:200]


@dataclass
class SpotifyAuth:
    """Holds credentials + caches a short-lived access token.

    Attributes:
        creds: the loaded credentials (must have a refresh_token).
        post_fn: injectable HTTP POST (defaults to ``requests.post``).
        clock: injectable monotonic clock (tests).
    """

    creds: SpotifyCredentials
    post_fn: Optional[PostFn] = None
    clock: Callable[[], float] = time.monotonic
    _access_token: Optional[str] = field(default=None, init=False)
    _expires_at: float = field(default=0.0, init=False)

    @property
    def authorized(self) -> bool:
        """True iff a refresh token is present (setup has run)."""
        return bool(self.creds.refresh_token)

    def access_token(self) -> str:
        """Return a valid access token, refreshing when expired.

        Raises:
            SpotifyAuthError: no refresh token, or the refresh failed.
        """
        if not self.creds.refresh_token:
            raise SpotifyAuthError(
                "Spotify is not authorized yet. Run "
                "scripts/spotify_setup.py once to grant access."
            )
        # 30 s safety margin before the real expiry.
        if self._access_token and self.clock() < self._expires_at - 30.0:
            return self._access_token
        post = self.post_fn or _default_post()
        resp = post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": self.creds.refresh_token,
            },
            headers={"Authorization": _basic_auth_header(self.creds)},
            timeout=15,
        )
        if getattr(resp, "status_code", 500) != 200:
            raise SpotifyAuthError(
                f"token refresh failed ({getattr(resp, 'status_code', '?')}): "
                f"{_safe_body(resp)}"
            )
        payload = resp.json()
        token = payload.get("access_token")
        if not token:
            raise SpotifyAuthError("token refresh returned no access_token")
        self._access_token = token
        self._expires_at = self.clock() + float(payload.get("expires_in", 3600))
        # Spotify may rotate the refresh token; persist if so.
        new_refresh = payload.get("refresh_token")
        if new_refresh and new_refresh != self.creds.refresh_token:
            self.creds.refresh_token = new_refresh
            if self.creds.path is not None:
                try:
                    save_refresh_token(self.creds.path, new_refresh)
                except Exception as e:  # noqa: BLE001
                    logger.debug("refresh-token persist failed: %s", e)
        return token
