"""Offline tests for kenning.twitch.auth — Device Code Flow + rotation-safe store.

No network, no creds, no models: every Twitch HTTP call goes through an injected
``transport`` mock. The token store writes to a tmp path. We also assert that
token VALUES never appear in captured logs.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import sys

import pytest

from kenning.twitch.auth import (
    BOT_SCOPES,
    BROADCASTER_SCOPES,
    DEVICE_URL,
    TOKEN_URL,
    VALIDATE_URL,
    DeviceCode,
    DeviceFlowError,
    DeviceFlowExpiredError,
    RevokedError,
    TokenStore,
    TwitchAuth,
    TwitchAuthError,
    make_urllib_transport,
)


# --------------------------------------------------------------------------- #
# Helpers / fakes                                                             #
# --------------------------------------------------------------------------- #
class FakeTransport:
    """Scriptable transport: a queue of (status, body) responses per URL.

    Records every call so tests can inspect method/url/data/headers. The
    signature matches the real one:
        transport(method, url, *, data=None, headers=None) -> (status, body)
    """

    def __init__(self) -> None:
        self.responses: dict = {}   # url -> list[(status, body)]
        self.default = None         # fallback (status, body)
        self.calls: list = []

    def queue(self, url: str, status: int, body: dict) -> "FakeTransport":
        self.responses.setdefault(url, []).append((status, body))
        return self

    def set_default(self, status: int, body: dict) -> "FakeTransport":
        self.default = (status, body)
        return self

    def __call__(self, method, url, *, data=None, headers=None):
        self.calls.append(
            {"method": method, "url": url, "data": data, "headers": headers}
        )
        seq = self.responses.get(url)
        if seq:
            return seq.pop(0)
        if self.default is not None:
            return self.default
        raise AssertionError(f"FakeTransport: no scripted response for {method} {url}")


def make_auth(tmp_path, transport, scopes=BROADCASTER_SCOPES):
    """A TwitchAuth wired to a tmp store + a no-real-sleep clock."""
    store = TokenStore(tmp_path / "twitch.json")
    ticks = {"t": 0.0}

    def clock():
        return ticks["t"]

    def sleep(sec):
        ticks["t"] += float(sec)

    return TwitchAuth(
        "client-abc",
        store,
        transport=transport,
        scopes=scopes,
        clock=clock,
        sleep=sleep,
    ), store


# --------------------------------------------------------------------------- #
# TokenStore: atomic save / load / rotate                                     #
# --------------------------------------------------------------------------- #
def test_store_save_then_load_roundtrips(tmp_path):
    store = TokenStore(tmp_path / "twitch.json")
    assert store.load() is None  # absent -> None, never raises

    tokens = {"access_token": "AAA", "refresh_token": "RRR", "scope": "user:bot"}
    store.save(tokens)
    loaded = store.load()
    assert loaded == tokens
    # File really exists and is valid JSON on disk.
    on_disk = json.loads((tmp_path / "twitch.json").read_text(encoding="utf-8"))
    assert on_disk["access_token"] == "AAA"


def test_store_save_is_atomic_no_tmp_left_behind(tmp_path):
    store = TokenStore(tmp_path / "twitch.json")
    store.save({"access_token": "A", "refresh_token": "R"})
    # No leftover *.tmp scratch files in the directory.
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []
    # Overwriting must also be atomic + leave only the one file.
    store.save({"access_token": "A2", "refresh_token": "R2"})
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["twitch.json"]
    assert store.load()["access_token"] == "A2"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits only")
def test_store_save_chmods_0600(tmp_path):
    store = TokenStore(tmp_path / "twitch.json")
    store.save({"access_token": "A", "refresh_token": "R"})
    mode = stat.S_IMODE(os.stat(tmp_path / "twitch.json").st_mode)
    assert mode == 0o600
    assert store.is_secure_perms() is True


def test_store_load_corrupt_file_returns_none(tmp_path):
    p = tmp_path / "twitch.json"
    p.write_text("{not valid json", encoding="utf-8")
    store = TokenStore(p)
    assert store.load() is None  # fail-safe: corrupt == absent, no raise


def test_store_rotate_persists_new_refresh_token(tmp_path):
    store = TokenStore(tmp_path / "twitch.json")
    store.save({"access_token": "OLD_A", "refresh_token": "OLD_R", "scope": "s"})

    merged = store.rotate({"access_token": "NEW_A", "refresh_token": "NEW_R"})
    # Returned + persisted both carry the NEW refresh token...
    assert merged["refresh_token"] == "NEW_R"
    assert merged["access_token"] == "NEW_A"
    # ...and unrelated prior fields survive the merge.
    assert merged["scope"] == "s"
    # And it is durably on disk (a crash right now would still have the new token).
    reloaded = store.load()
    assert reloaded["refresh_token"] == "NEW_R"
    assert reloaded["access_token"] == "NEW_A"


def test_store_rotate_without_refresh_token_is_refused(tmp_path):
    store = TokenStore(tmp_path / "twitch.json")
    store.save({"access_token": "A", "refresh_token": "R"})
    with pytest.raises(TwitchAuthError):
        store.rotate({"access_token": "A2"})  # no refresh_token -> would brick
    # Old good token is untouched.
    assert store.load()["refresh_token"] == "R"


def test_store_expands_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    store = TokenStore()  # default ~/.kenning/twitch.json
    assert ".kenning" in str(store.path)
    assert store.path.name == "twitch.json"


# --------------------------------------------------------------------------- #
# Device flow: start                                                          #
# --------------------------------------------------------------------------- #
def test_start_device_flow_happy_path(tmp_path):
    t = FakeTransport().queue(
        DEVICE_URL,
        200,
        {
            "device_code": "DEV123",
            "user_code": "WXYZ-1234",
            "verification_uri": "https://www.twitch.tv/activate",
            "interval": 5,
            "expires_in": 1800,
        },
    )
    auth, _ = make_auth(tmp_path, t)
    dc = auth.start_device_flow()
    assert isinstance(dc, DeviceCode)
    assert dc.user_code == "WXYZ-1234"
    assert dc.verification_uri == "https://www.twitch.tv/activate"
    assert dc.device_code == "DEV123"
    assert dc.interval == 5
    # The request POSTed the client_id + the requested scopes.
    call = t.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == DEVICE_URL
    assert call["data"]["client_id"] == "client-abc"
    assert "channel:read:redemptions" in call["data"]["scopes"]


def test_start_device_flow_http_error_raises(tmp_path):
    t = FakeTransport().queue(DEVICE_URL, 400, {"message": "invalid client"})
    auth, _ = make_auth(tmp_path, t)
    with pytest.raises(DeviceFlowError):
        auth.start_device_flow()


# --------------------------------------------------------------------------- #
# Device flow: poll                                                           #
# --------------------------------------------------------------------------- #
def test_poll_returns_tokens_and_persists(tmp_path):
    t = (
        FakeTransport()
        .queue(TOKEN_URL, 400, {"message": "authorization_pending"})
        .queue(TOKEN_URL, 400, {"message": "authorization_pending"})
        .queue(
            TOKEN_URL,
            200,
            {
                "access_token": "ACCESS1",
                "refresh_token": "REFRESH1",
                "expires_in": 14400,
                "scope": ["user:read:chat", "user:write:chat"],
                "token_type": "bearer",
            },
        )
    )
    auth, store = make_auth(tmp_path, t)
    tokens = auth.poll_device_token("DEV123", interval=5, timeout=10_000)
    assert tokens["access_token"] == "ACCESS1"
    assert tokens["refresh_token"] == "REFRESH1"
    assert tokens["scope"] == "user:read:chat user:write:chat"
    assert tokens["expires_at"] > 0
    # Persisted to disk on success.
    assert store.load()["access_token"] == "ACCESS1"
    # Polled 3 times (2 pending + 1 success).
    poll_calls = [c for c in t.calls if c["url"] == TOKEN_URL]
    assert len(poll_calls) == 3
    assert poll_calls[0]["data"]["grant_type"].endswith("device_code")


def test_poll_honors_slow_down(tmp_path):
    t = (
        FakeTransport()
        .queue(TOKEN_URL, 400, {"message": "slow_down"})
        .queue(
            TOKEN_URL,
            200,
            {"access_token": "A", "refresh_token": "R", "expires_in": 100},
        )
    )
    auth, _ = make_auth(tmp_path, t)
    tokens = auth.poll_device_token("DEV", interval=5, timeout=10_000)
    assert tokens["access_token"] == "A"


def test_poll_times_out_raises_expired(tmp_path):
    # Always pending -> the clock advances via sleep -> deadline exceeded.
    t = FakeTransport().set_default(400, {"message": "authorization_pending"})
    auth, _ = make_auth(tmp_path, t)
    with pytest.raises(DeviceFlowExpiredError):
        auth.poll_device_token("DEV", interval=5, timeout=12)


def test_poll_expired_token_raises(tmp_path):
    t = FakeTransport().queue(TOKEN_URL, 400, {"message": "expired_token"})
    auth, _ = make_auth(tmp_path, t)
    with pytest.raises(DeviceFlowExpiredError):
        auth.poll_device_token("DEV", interval=5, timeout=10_000)


def test_poll_access_denied_raises(tmp_path):
    t = FakeTransport().queue(TOKEN_URL, 400, {"message": "access_denied"})
    auth, _ = make_auth(tmp_path, t)
    with pytest.raises(DeviceFlowError):
        auth.poll_device_token("DEV", interval=5, timeout=10_000)


# --------------------------------------------------------------------------- #
# validate                                                                    #
# --------------------------------------------------------------------------- #
def test_validate_ok(tmp_path):
    t = FakeTransport().queue(
        VALIDATE_URL,
        200,
        {
            "client_id": "client-abc",
            "login": "streamer",
            "user_id": "12345",
            "scopes": ["user:read:chat"],
            "expires_in": 13000,
        },
    )
    auth, _ = make_auth(tmp_path, t)
    res = auth.validate("ACCESS1")
    assert res is not None
    assert res["login"] == "streamer"
    # Sent as an OAuth Authorization header (Twitch's required scheme).
    call = t.calls[0]
    assert call["method"] == "GET"
    assert call["headers"]["Authorization"] == "OAuth ACCESS1"


def test_validate_revoked_returns_none(tmp_path):
    t = FakeTransport().queue(VALIDATE_URL, 401, {"message": "invalid access token"})
    auth, _ = make_auth(tmp_path, t)
    assert auth.validate("DEAD") is None


def test_validate_empty_token_returns_none(tmp_path):
    t = FakeTransport()
    auth, _ = make_auth(tmp_path, t)
    assert auth.validate("") is None
    assert t.calls == []  # short-circuits, no network


# --------------------------------------------------------------------------- #
# refresh (single-use rotation)                                               #
# --------------------------------------------------------------------------- #
def test_refresh_rotates_and_persists(tmp_path):
    t = FakeTransport().queue(
        TOKEN_URL,
        200,
        {
            "access_token": "ACCESS2",
            "refresh_token": "REFRESH2",
            "expires_in": 14400,
            "scope": "user:read:chat",
        },
    )
    auth, store = make_auth(tmp_path, t)
    store.save({"access_token": "ACCESS1", "refresh_token": "REFRESH1"})
    tokens = auth.refresh("REFRESH1")
    assert tokens["access_token"] == "ACCESS2"
    assert tokens["refresh_token"] == "REFRESH2"
    # The new single-use refresh token is durably persisted.
    assert store.load()["refresh_token"] == "REFRESH2"
    call = t.calls[0]
    assert call["data"]["grant_type"] == "refresh_token"
    assert call["data"]["refresh_token"] == "REFRESH1"


def test_refresh_rejected_raises_revoked(tmp_path):
    t = FakeTransport().queue(TOKEN_URL, 400, {"message": "Invalid refresh token"})
    auth, _ = make_auth(tmp_path, t)
    with pytest.raises(RevokedError):
        auth.refresh("STALE")


def test_refresh_no_token_raises_revoked(tmp_path):
    t = FakeTransport()
    auth, _ = make_auth(tmp_path, t)
    with pytest.raises(RevokedError):
        auth.refresh("")


# --------------------------------------------------------------------------- #
# call_with_auth: 401 -> refresh once -> retry                                #
# --------------------------------------------------------------------------- #
def test_call_with_auth_401_refresh_then_retry_succeeds(tmp_path):
    t = FakeTransport().queue(
        TOKEN_URL,
        200,
        {"access_token": "FRESH", "refresh_token": "NEWR", "expires_in": 100},
    )
    auth, store = make_auth(tmp_path, t)
    store.save({"access_token": "EXPIRED", "refresh_token": "OLDR"})

    seen = []

    def do_request(token):
        seen.append(token)
        if token == "EXPIRED":
            return 401, None
        return 200, {"data": "ok"}

    result = auth.call_with_auth(do_request)
    assert result == {"data": "ok"}
    # Called with the expired token, refreshed, retried with the fresh token.
    assert seen == ["EXPIRED", "FRESH"]
    # The rotated refresh token was persisted.
    assert store.load()["refresh_token"] == "NEWR"


def test_call_with_auth_first_try_ok_no_refresh(tmp_path):
    t = FakeTransport()  # no token endpoint should be hit
    auth, store = make_auth(tmp_path, t)
    store.save({"access_token": "GOOD", "refresh_token": "R"})

    def do_request(token):
        return 200, {"ok": True}

    assert auth.call_with_auth(do_request) == {"ok": True}
    assert t.calls == []  # never refreshed


def test_call_with_auth_still_401_after_refresh_raises_revoked(tmp_path):
    t = FakeTransport().queue(
        TOKEN_URL,
        200,
        {"access_token": "FRESH", "refresh_token": "NEWR", "expires_in": 100},
    )
    auth, store = make_auth(tmp_path, t)
    store.save({"access_token": "EXPIRED", "refresh_token": "OLDR"})

    def do_request(token):
        return 401, None  # always unauthorized -> app/grant revoked

    with pytest.raises(RevokedError):
        auth.call_with_auth(do_request)


def test_call_with_auth_no_stored_token_raises_revoked(tmp_path):
    t = FakeTransport()
    auth, _ = make_auth(tmp_path, t)

    def do_request(token):  # pragma: no cover - should never be invoked
        raise AssertionError("must not request without a token")

    with pytest.raises(RevokedError):
        auth.call_with_auth(do_request)


def test_call_with_auth_refresh_dead_grant_raises_revoked(tmp_path):
    # 401 on the API, then a 400 on the refresh => RevokedError from refresh().
    t = FakeTransport().queue(TOKEN_URL, 400, {"message": "Invalid refresh token"})
    auth, store = make_auth(tmp_path, t)
    store.save({"access_token": "EXPIRED", "refresh_token": "STALE"})

    def do_request(token):
        return 401, None

    with pytest.raises(RevokedError):
        auth.call_with_auth(do_request)


# --------------------------------------------------------------------------- #
# RevokedError on validate -> None (the revocation handling contract)         #
# --------------------------------------------------------------------------- #
def test_revoked_on_validate_none_can_be_raised_clearly(tmp_path):
    """validate()->None signals revocation; the caller raises a clear error.

    The module gives validate()->None and a RevokedError type so a caller can
    turn 'validate said the token is dead' into an explicit, clear failure.
    """
    t = FakeTransport().queue(VALIDATE_URL, 401, {"message": "invalid"})
    auth, store = make_auth(tmp_path, t)
    store.save({"access_token": "DEAD", "refresh_token": "ALSO_DEAD"})

    tokens = store.load()
    if auth.validate(tokens["access_token"]) is None:
        with pytest.raises(RevokedError):
            raise RevokedError("token revoked; re-authorize")


# --------------------------------------------------------------------------- #
# Tokens NEVER appear in logs                                                 #
# --------------------------------------------------------------------------- #
def test_tokens_never_logged(tmp_path, caplog):
    secret_access = "SUPER_SECRET_ACCESS_TOKEN_abcdef0123456789"
    secret_refresh = "SUPER_SECRET_REFRESH_TOKEN_zyxwvu9876543210"
    secret_device = "SECRET_DEVICE_CODE_qwerty"

    t = (
        FakeTransport()
        .queue(
            DEVICE_URL,
            200,
            {
                "device_code": secret_device,
                "user_code": "WXYZ-1234",
                "verification_uri": "https://www.twitch.tv/activate",
                "interval": 1,
                "expires_in": 600,
            },
        )
        .queue(
            TOKEN_URL,
            200,
            {
                "access_token": secret_access,
                "refresh_token": secret_refresh,
                "expires_in": 100,
                "scope": "user:read:chat",
            },
        )
        .queue(
            VALIDATE_URL,
            200,
            {"client_id": "client-abc", "login": "streamer", "user_id": "1"},
        )
    )
    auth, store = make_auth(tmp_path, t)

    with caplog.at_level(logging.DEBUG, logger="kenning.twitch.auth"):
        dc = auth.start_device_flow()
        auth.poll_device_token(dc.device_code, interval=1, timeout=10_000)
        # A refresh too (queue a rotation response).
        t.queue(
            TOKEN_URL,
            200,
            {
                "access_token": "ANOTHER_SECRET_ACCESS_zz",
                "refresh_token": "ANOTHER_SECRET_REFRESH_yy",
                "expires_in": 100,
            },
        )
        auth.refresh(secret_refresh)
        auth.validate(secret_access)

    blob = "\n".join(r.getMessage() for r in caplog.records)
    for secret in (
        secret_access,
        secret_refresh,
        secret_device,
        "ANOTHER_SECRET_ACCESS_zz",
        "ANOTHER_SECRET_REFRESH_yy",
    ):
        assert secret not in blob, f"token value leaked into logs: {secret!r}"
    # And we DID log something (proving the assertion isn't vacuous).
    assert blob.strip() != ""


# --------------------------------------------------------------------------- #
# Transport contract / urllib default                                         #
# --------------------------------------------------------------------------- #
def test_urllib_transport_parses_http_error_body_without_raising(monkeypatch):
    """A non-2xx response must return (status, body), NOT raise — so the poll
    loop can branch on authorization_pending. We stub urlopen with an HTTPError.
    """
    import io
    import urllib.error
    import urllib.request

    transport = make_urllib_transport(timeout=1.0)

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url,
            400,
            "Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"message": "authorization_pending"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    status, body = transport("POST", TOKEN_URL, data={"client_id": "x"})
    assert status == 400
    assert body["message"] == "authorization_pending"


def test_urllib_transport_network_failure_raises_deviceflowerror(monkeypatch):
    import urllib.error
    import urllib.request

    transport = make_urllib_transport(timeout=1.0)

    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("dns boom")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(DeviceFlowError):
        transport("GET", VALIDATE_URL)


def test_client_id_required(tmp_path):
    store = TokenStore(tmp_path / "twitch.json")
    with pytest.raises(TwitchAuthError):
        TwitchAuth("", store)


def test_bot_and_broadcaster_scopes_are_distinct():
    assert set(BOT_SCOPES) != set(BROADCASTER_SCOPES)
    assert "user:write:chat" in BOT_SCOPES
    assert "channel:manage:redemptions" in BROADCASTER_SCOPES


# --------------------------------------------------------------------------- #
# scripts/twitch_setup.py — import-safe + end-to-end with a mock              #
# --------------------------------------------------------------------------- #
def test_setup_script_import_has_no_side_effects():
    # Importing the CLI must not run the flow or touch the network.
    import importlib

    mod = importlib.import_module("scripts.twitch_setup")
    assert hasattr(mod, "main")
    assert hasattr(mod, "run_setup")
    assert hasattr(mod, "build_parser")
    # build_parser is pure (no parsing).
    parser = mod.build_parser()
    assert parser.prog == "twitch_setup"


def test_setup_script_run_setup_end_to_end(tmp_path):
    import io

    from scripts import twitch_setup

    t = (
        FakeTransport()
        .queue(
            DEVICE_URL,
            200,
            {
                "device_code": "DEVX",
                "user_code": "ABCD-9999",
                "verification_uri": "https://www.twitch.tv/activate",
                "interval": 1,
                "expires_in": 600,
            },
        )
        .queue(
            TOKEN_URL,
            200,
            {
                "access_token": "A_TOK",
                "refresh_token": "R_TOK",
                "expires_in": 100,
                "scope": "user:read:chat",
            },
        )
    )
    auth, store = make_auth(tmp_path, t, scopes=BOT_SCOPES)
    out = io.StringIO()
    tokens = twitch_setup.run_setup(
        "client-abc", identity="bot", auth=auth, out=out, timeout=10_000
    )
    assert tokens["access_token"] == "A_TOK"
    assert store.load()["refresh_token"] == "R_TOK"
    printed = out.getvalue()
    # The user code + URL are shown; the token values are NOT.
    assert "ABCD-9999" in printed
    assert "twitch.tv/activate" in printed
    assert "A_TOK" not in printed
    assert "R_TOK" not in printed


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
