"""Tests for Spotify control (auth + client + voice + wiring).

Fully hermetic: every HTTP call is an injected fake; no network, no
real credentials, no Spotify.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ultron.spotify.auth import (
    SpotifyAuth,
    SpotifyAuthError,
    SpotifyCredentials,
    build_authorize_url,
    load_credentials,
    save_refresh_token,
)
from ultron.spotify.client import NowPlaying, SpotifyAPIError, SpotifyClient
from ultron.spotify.voice import (
    SpotifyCommand,
    handle_spotify_command,
    match_spotify_command,
)

from ultron.pipeline.orchestrator import Orchestrator


class _Resp:
    def __init__(self, status: int, body=None):
        self.status_code = status
        self._body = body if body is not None else {}

    def json(self):
        return self._body


# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------


def _creds(refresh="rt") -> SpotifyCredentials:
    return SpotifyCredentials(
        client_id="cid", client_secret="sec",
        redirect_uri="http://127.0.0.1:8899/callback", refresh_token=refresh,
    )


def test_load_credentials(tmp_path: Path) -> None:
    p = tmp_path / "spotify.json"
    p.write_text(json.dumps({
        "client_id": "abc", "client_secret": "xyz",
        "redirect_uri": "http://127.0.0.1:8899/callback",
        "refresh_token": "tok",
    }), encoding="utf-8")
    creds = load_credentials(p)
    assert creds.client_id == "abc" and creds.refresh_token == "tok"
    assert creds.path == p


def test_load_credentials_missing(tmp_path: Path) -> None:
    with pytest.raises(SpotifyAuthError):
        load_credentials(tmp_path / "nope.json")


def test_load_credentials_no_client_id(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"client_secret": "x"}), encoding="utf-8")
    with pytest.raises(SpotifyAuthError):
        load_credentials(p)


def test_save_refresh_token_preserves_fields(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"client_id": "a", "_note": "keep"}),
                 encoding="utf-8")
    save_refresh_token(p, "newtok")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["refresh_token"] == "newtok"
    assert data["client_id"] == "a" and data["_note"] == "keep"


def test_build_authorize_url_has_scopes_and_redirect() -> None:
    url = build_authorize_url(_creds())
    assert "accounts.spotify.com/authorize" in url
    assert "client_id=cid" in url
    assert "user-modify-playback-state" in url
    assert "redirect_uri=http" in url


def test_access_token_refreshes_and_caches() -> None:
    calls = []

    def post(url, **kw):
        calls.append(kw["data"]["grant_type"])
        return _Resp(200, {"access_token": "AT", "expires_in": 3600})

    t = {"v": 1000.0}
    auth = SpotifyAuth(_creds(), post_fn=post, clock=lambda: t["v"])
    assert auth.access_token() == "AT"
    assert auth.access_token() == "AT"  # cached, no second call
    assert calls == ["refresh_token"]
    # After expiry it refreshes again.
    t["v"] += 4000
    assert auth.access_token() == "AT"
    assert calls == ["refresh_token", "refresh_token"]


def test_access_token_without_refresh_raises() -> None:
    auth = SpotifyAuth(_creds(refresh=None))
    assert auth.authorized is False
    with pytest.raises(SpotifyAuthError):
        auth.access_token()


def test_access_token_refresh_failure_raises() -> None:
    auth = SpotifyAuth(_creds(), post_fn=lambda u, **k: _Resp(400, {"e": "x"}))
    with pytest.raises(SpotifyAuthError):
        auth.access_token()


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------


class _FakeAuth:
    authorized = True

    def access_token(self) -> str:
        return "AT"


def _client(routes):
    """routes: (method, path_substr) -> _Resp or callable(kw)->_Resp."""
    seen = []

    def request(method, url, **kw):
        seen.append((method, url, kw))
        for (m, sub), resp in routes.items():
            if m == method and sub in url:
                return resp(kw) if callable(resp) else resp
        return _Resp(204)

    c = SpotifyClient(_FakeAuth(), request_fn=request)
    return c, seen


def test_now_playing() -> None:
    body = {"is_playing": True, "device": {"name": "Echo"},
            "item": {"name": "Song", "artists": [{"name": "Artist"}]}}
    c, _ = _client({("GET", "/me/player"): _Resp(200, body)})
    np = c.now_playing()
    assert np.is_playing and np.track == "Song" and np.artist == "Artist"
    assert "Playing Song by Artist" in np.spoken()


def test_now_playing_empty() -> None:
    c, _ = _client({("GET", "/me/player"): _Resp(204)})
    assert c.now_playing().spoken() == "Nothing is playing right now."


def test_play_query_track_searches_then_plays() -> None:
    search = _Resp(200, {"tracks": {"items": [
        {"uri": "spotify:track:1", "name": "Despacito",
         "artists": [{"name": "Luis Fonsi"}]}]}})
    routes = {
        ("GET", "/search"): search,
        ("GET", "/me/player/devices"): _Resp(200, {"devices": [
            {"id": "d1", "name": "PC", "is_active": True}]}),
        ("PUT", "/me/player/play"): _Resp(204),
    }
    c, seen = _client(routes)
    line = c.play_query("despacito", "track")
    assert "Playing Despacito by Luis Fonsi" in line
    assert any(m == "PUT" and "/play" in u for m, u, _ in seen)


def test_play_query_not_found() -> None:
    c, _ = _client({("GET", "/search"): _Resp(200, {"tracks": {"items": []}})})
    assert "couldn't find" in c.play_query("zzzz", "track").lower()


def test_play_query_playlist_uses_context() -> None:
    routes = {
        ("GET", "/search"): _Resp(200, {"playlists": {"items": [
            {"uri": "spotify:playlist:9", "name": "Focus"}]}}),
        ("GET", "/me/player/devices"): _Resp(200, {"devices": [
            {"id": "d1", "name": "PC", "is_active": True}]}),
        ("PUT", "/me/player/play"): lambda kw: (
            _Resp(204) if kw["json"].get("context_uri") else _Resp(400)),
    }
    c, _ = _client(routes)
    assert "Focus" in c.play_query("focus", "playlist")


def test_ensure_device_transfers_when_none_active() -> None:
    routes = {
        ("GET", "/me/player/devices"): _Resp(200, {"devices": [
            {"id": "d1", "name": "Phone", "is_active": False}]}),
        ("PUT", "/me/player"): _Resp(204),
    }
    c, seen = _client(routes)
    assert c.ensure_device() == "d1"
    assert any(m == "PUT" and u.endswith("/me/player") for m, u, _ in seen)


def test_set_volume_clamps() -> None:
    captured = {}

    def vol(kw):
        captured.update(kw["params"])
        return _Resp(204)

    c, _ = _client({("PUT", "/me/player/volume"): vol})
    c.set_volume(150)
    assert captured["volume_percent"] == 100


def test_auth_rejected_maps_to_clear_error() -> None:
    c, _ = _client({("GET", "/me/player"): _Resp(403)})
    with pytest.raises(SpotifyAPIError):
        c.now_playing()


# ---------------------------------------------------------------------------
# voice matcher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,action,arg,kind", [
    ("play despacito", "play", "despacito", "track"),
    ("play the song bohemian rhapsody", "play", "bohemian rhapsody", "track"),
    ("play some daft punk", "play", "daft punk", "artist"),
    ("play the album discovery", "play", "discovery", "album"),
    ("play my focus playlist", "play", "focus", "playlist"),
    ("play despacito on spotify", "play", "despacito", "track"),
    ("queue blinding lights", "queue", "blinding lights", "track"),
])
def test_match_play_and_queue(text, action, arg, kind) -> None:
    cmd = match_spotify_command(text)
    assert cmd is not None, text
    assert cmd.action == action
    assert cmd.argument == arg
    if action == "play":
        assert cmd.kind == kind


@pytest.mark.parametrize("text,action", [
    ("pause", "pause"),
    ("pause the music", "pause"),
    ("stop the music", "pause"),
    ("resume", "resume"),
    ("keep playing", "resume"),
    ("next", "next"),
    ("skip", "next"),
    ("skip this song", "next"),
    ("next track", "next"),
    ("previous song", "previous"),
    ("go back", "previous"),
    ("what's playing", "now_playing"),
    ("what song is this", "now_playing"),
    ("turn it up", "volume_up"),
    ("louder", "volume_up"),
    ("turn it down", "volume_down"),
    ("shuffle on", "shuffle"),
    ("turn off shuffle", "shuffle"),
    ("repeat this song", "repeat"),
])
def test_match_transport(text, action) -> None:
    cmd = match_spotify_command(text)
    assert cmd is not None, text
    assert cmd.action == action


def test_match_volume_set() -> None:
    cmd = match_spotify_command("set the volume to 40 percent")
    assert cmd is not None and cmd.action == "volume_set" and cmd.value == 40


@pytest.mark.parametrize("text", [
    "what's the weather today",
    "open chrome",
    "tell my team to rotate",
    "write me a program",
    "how are you",
    "",
])
def test_match_negatives(text) -> None:
    assert match_spotify_command(text) is None


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


class _SpyClient:
    def __init__(self):
        self.calls = []

    def play_query(self, q, kind):
        self.calls.append(("play", q, kind))
        return f"Playing {q}."

    def queue_query(self, q):
        self.calls.append(("queue", q))
        return f"Queued {q}."

    def pause(self):
        self.calls.append(("pause",))

    def resume(self):
        self.calls.append(("resume",))

    def next_track(self):
        self.calls.append(("next",))

    def previous_track(self):
        self.calls.append(("prev",))

    def now_playing(self):
        return NowPlaying(is_playing=True, track="X", artist="Y")

    def current_volume(self):
        return 50

    def set_volume(self, v):
        self.calls.append(("vol", v))

    def set_shuffle(self, s):
        self.calls.append(("shuffle", s))

    def set_repeat(self, m):
        self.calls.append(("repeat", m))


def test_handle_play() -> None:
    c = _SpyClient()
    line = handle_spotify_command(SpotifyCommand("play", "despacito"), c)
    assert line == "Playing despacito." and c.calls == [
        ("play", "despacito", "track")]


def test_handle_volume_up_from_current() -> None:
    c = _SpyClient()
    line = handle_spotify_command(SpotifyCommand("volume_up"), c)
    assert c.calls == [("vol", 65)] and "65" in line


def test_handle_now_playing() -> None:
    line = handle_spotify_command(SpotifyCommand("now_playing"), _SpyClient())
    assert "Playing X by Y" in line


def test_handle_auth_error_speaks_setup_hint() -> None:
    class _Boom:
        def pause(self):
            raise SpotifyAuthError("no token")

    line = handle_spotify_command(SpotifyCommand("pause"), _Boom())
    assert "setup" in line.lower() or "authorize" in line.lower()


def test_handle_no_device_error() -> None:
    class _Boom:
        def next_track(self):
            raise SpotifyAPIError("Spotify API POST /next -> 404")

    line = handle_spotify_command(SpotifyCommand("next"), _Boom())
    assert "device" in line.lower()


# ---------------------------------------------------------------------------
# orchestrator wiring
# ---------------------------------------------------------------------------


def _orch():
    o = Orchestrator.__new__(Orchestrator)
    o._spoken = []
    o._speak = lambda t: o._spoken.append(t)  # type: ignore[attr-defined]
    return o


def test_orchestrator_spotify_disabled(monkeypatch) -> None:
    import ultron.config as config_mod

    monkeypatch.setattr(config_mod, "get_config",
                        lambda: SimpleNamespace(
                            spotify=SimpleNamespace(enabled=False)))
    o = _orch()
    assert o._maybe_handle_spotify("play despacito") is False
    assert o._spoken == []


def test_orchestrator_spotify_no_match(monkeypatch) -> None:
    import ultron.config as config_mod

    monkeypatch.setattr(config_mod, "get_config",
                        lambda: SimpleNamespace(
                            spotify=SimpleNamespace(enabled=True)))
    o = _orch()
    assert o._maybe_handle_spotify("what's the weather") is False


def test_orchestrator_spotify_not_set_up(monkeypatch) -> None:
    import ultron.config as config_mod

    monkeypatch.setattr(config_mod, "get_config",
                        lambda: SimpleNamespace(
                            spotify=SimpleNamespace(enabled=True)))
    o = _orch()
    o._spotify_client = None  # simulate missing credentials
    assert o._maybe_handle_spotify("play despacito") is True
    assert "set up" in o._spoken[0].lower()


def test_orchestrator_spotify_happy(monkeypatch) -> None:
    import ultron.config as config_mod

    monkeypatch.setattr(config_mod, "get_config",
                        lambda: SimpleNamespace(
                            spotify=SimpleNamespace(enabled=True)))
    o = _orch()
    o._spotify_client = _SpyClient()
    assert o._maybe_handle_spotify("play despacito") is True
    assert o._spoken == ["Playing despacito."]
    assert o._spotify_client.calls == [("play", "despacito", "track")]
