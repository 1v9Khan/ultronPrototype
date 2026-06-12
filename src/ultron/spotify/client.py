"""Spotify Web API client -- playback control + search.

Thin wrapper over the REST API. Every call attaches a fresh access
token from :class:`~ultron.spotify.auth.SpotifyAuth` and goes through an
injectable ``request_fn`` (matches ``requests.request``) so unit tests
never hit the network. Methods return small result objects / plain
dicts; the voice layer turns those into spoken lines.

Playback-control endpoints (play/pause/next/volume/...) require Spotify
Premium and an ACTIVE device. When nothing is playing and no device is
active, Spotify returns 404 "NO_ACTIVE_DEVICE"; :meth:`ensure_device`
transfers playback to the user's last/most-available device first.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ultron.spotify.auth import SpotifyAuth, SpotifyAuthError

logger = logging.getLogger("ultron.spotify.client")

__all__ = ["SpotifyClient", "SpotifyAPIError", "NowPlaying", "Device"]

API = "https://api.spotify.com/v1"

RequestFn = Callable[..., Any]


class SpotifyAPIError(RuntimeError):
    """A Web API call returned a non-success status."""


@dataclass(frozen=True)
class NowPlaying:
    """The currently-playing item (or a paused/empty state)."""

    is_playing: bool
    track: str = ""
    artist: str = ""
    device: str = ""

    def spoken(self) -> str:
        if not self.track:
            return "Nothing is playing right now."
        verb = "Playing" if self.is_playing else "Paused on"
        who = f" by {self.artist}" if self.artist else ""
        return f"{verb} {self.track}{who}."


@dataclass(frozen=True)
class Device:
    """A Spotify Connect playback device."""

    id: str
    name: str
    is_active: bool
    type: str = ""


def _default_request() -> RequestFn:
    import requests

    return requests.request


class SpotifyClient:
    """Spotify Web API client bound to one authorized user."""

    def __init__(
        self,
        auth: SpotifyAuth,
        *,
        request_fn: Optional[RequestFn] = None,
        default_device: str = "",
    ) -> None:
        """Args:
        auth: the token provider.
        request_fn: injectable HTTP (defaults to ``requests.request``).
        default_device: preferred device NAME to fall back to when no
            device is active (empty = first available).
        """
        self._auth = auth
        self._request = request_fn or _default_request()
        self._default_device = default_device

    # -- low-level -----------------------------------------------------

    def _call(
        self, method: str, path: str, *,
        params: Optional[dict] = None, json_body: Optional[dict] = None,
        allow_404: bool = False,
    ) -> Any:
        token = self._auth.access_token()
        resp = self._request(
            method, API + path,
            headers={"Authorization": f"Bearer {token}"},
            params=params, json=json_body, timeout=15,
        )
        code = getattr(resp, "status_code", 500)
        if code == 204 or code == 202:
            return None
        if allow_404 and code == 404:
            return None
        if code in (401, 403):
            raise SpotifyAPIError(
                "Spotify rejected the request -- re-authorize "
                "(run scripts/spotify_setup.py). "
                f"({code})"
            )
        if code >= 400:
            raise SpotifyAPIError(f"Spotify API {method} {path} -> {code}")
        try:
            return resp.json()
        except Exception:  # noqa: BLE001 - empty body on a 200/PUT
            return None

    # -- state ---------------------------------------------------------

    def now_playing(self) -> NowPlaying:
        """What's playing right now (or an empty state)."""
        data = self._call("GET", "/me/player", allow_404=True)
        if not data or not data.get("item"):
            return NowPlaying(is_playing=False)
        item = data["item"]
        artists = ", ".join(a.get("name", "") for a in item.get("artists", []))
        return NowPlaying(
            is_playing=bool(data.get("is_playing")),
            track=item.get("name", ""),
            artist=artists,
            device=(data.get("device") or {}).get("name", ""),
        )

    def devices(self) -> list[Device]:
        data = self._call("GET", "/me/player/devices") or {}
        out: list[Device] = []
        for d in data.get("devices", []):
            out.append(Device(
                id=d.get("id", ""), name=d.get("name", ""),
                is_active=bool(d.get("is_active")), type=d.get("type", ""),
            ))
        return out

    def ensure_device(self) -> Optional[str]:
        """Make sure SOME device is active; transfer to one if not.

        Returns the active device id, or None when the user has no
        Spotify Connect device available at all (nothing open)."""
        devs = self.devices()
        if not devs:
            return None
        active = next((d for d in devs if d.is_active), None)
        if active:
            return active.id
        pick = None
        if self._default_device:
            pick = next(
                (d for d in devs
                 if d.name.lower() == self._default_device.lower()), None,
            )
        pick = pick or devs[0]
        self._call("PUT", "/me/player",
                   json_body={"device_ids": [pick.id], "play": False})
        return pick.id

    # -- transport -----------------------------------------------------

    def resume(self) -> None:
        self.ensure_device()
        self._call("PUT", "/me/player/play")

    def pause(self) -> None:
        self._call("PUT", "/me/player/pause", allow_404=True)

    def next_track(self) -> None:
        self.ensure_device()
        self._call("POST", "/me/player/next")

    def previous_track(self) -> None:
        self.ensure_device()
        self._call("POST", "/me/player/previous")

    def set_volume(self, percent: int) -> None:
        percent = max(0, min(100, int(percent)))
        self._call("PUT", "/me/player/volume",
                   params={"volume_percent": percent}, allow_404=True)

    def current_volume(self) -> Optional[int]:
        data = self._call("GET", "/me/player", allow_404=True)
        if not data:
            return None
        return (data.get("device") or {}).get("volume_percent")

    def set_shuffle(self, state: bool) -> None:
        self._call("PUT", "/me/player/shuffle",
                   params={"state": str(bool(state)).lower()}, allow_404=True)

    def set_repeat(self, mode: str) -> None:
        if mode not in ("track", "context", "off"):
            mode = "off"
        self._call("PUT", "/me/player/repeat",
                   params={"state": mode}, allow_404=True)

    # -- search + play -------------------------------------------------

    def search_first(self, query: str, kind: str) -> Optional[dict]:
        """First search hit of ``kind`` (track/artist/album/playlist)."""
        data = self._call("GET", "/search", params={
            "q": query, "type": kind, "limit": 5,
        }) or {}
        items = (data.get(kind + "s") or {}).get("items") or []
        items = [i for i in items if i]
        return items[0] if items else None

    def play_query(self, query: str, kind: str = "track") -> str:
        """Search for ``query`` and start playing the first match.

        ``kind`` = track plays that one song; artist/album/playlist
        plays the whole context. Returns a spoken confirmation, or a
        clear "couldn't find it" message.
        """
        hit = self.search_first(query, kind)
        if hit is None:
            return f"I couldn't find {query} on Spotify."
        self.ensure_device()
        name = hit.get("name", query)
        if kind == "track":
            artists = ", ".join(
                a.get("name", "") for a in hit.get("artists", []))
            self._call("PUT", "/me/player/play",
                       json_body={"uris": [hit["uri"]]})
            who = f" by {artists}" if artists else ""
            return f"Playing {name}{who}."
        # artist / album / playlist -> play the whole context.
        self._call("PUT", "/me/player/play",
                   json_body={"context_uri": hit["uri"]})
        label = {"artist": "", "album": "the album ",
                 "playlist": "the playlist "}.get(kind, "")
        return f"Playing {label}{name}."

    def queue_query(self, query: str) -> str:
        """Add the first matching track to the up-next queue."""
        hit = self.search_first(query, "track")
        if hit is None:
            return f"I couldn't find {query} to queue."
        self.ensure_device()
        self._call("POST", "/me/player/queue", params={"uri": hit["uri"]})
        artists = ", ".join(a.get("name", "") for a in hit.get("artists", []))
        who = f" by {artists}" if artists else ""
        return f"Queued {hit.get('name', query)}{who}."
