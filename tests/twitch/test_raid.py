"""Tests for the incoming-RAID feature: detect channel.raid -> vocal announce on
the STREAM bus + auto Helix /shoutout to the raider.

Fully offline (mocked): no live Twitch connection, no creds, no models, no ports
beyond ephemeral loopback test servers. Covered:

  * make_raid_drain_fn: parses raid events out of the read-sidecar /buffer shape
    on its OWN cursor, never acks, ignores non-raid events, fail-safe on a down
    sidecar / bad body.
  * build_raid_line: covers every required beat (announce + raider name + viewer
    count, thank, welcome, hope they stick around, introduce himself, how to chat)
    and stays in the Ultron persona (no vendor/model/"AI" name).
  * RaidHandler.tick: announces on the (stream-bus) announce_fn, calls the
    shoutout_fn with the raider id, is IDEMPOTENT (a replay never double-fires),
    and is FAIL-OPEN (an announce error never blocks the shoutout; a shoutout
    error never raises / never blocks the announce; shoutout disabled -> no call).
  * HelixClient.send_shoutout: POSTs /chat/shoutouts with the three ids, 204 ok,
    idempotent on the cooldown (429) + an "already" body + a local cache hit.
  * the read sidecar subscribes to channel.raid on the broadcaster-token session
    and maps a raid notification to {"type":"raid",...} (+ dedups a replay).
  * the write sidecar /shoutout endpoint calls the injected shoutout_fn, validates
    the body, and is fail-open (always 200; ok=false when unavailable/error).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from kenning.twitch.moderation.helix import HelixClient, TransportResponse
from kenning.twitch.raid import RaidHandler, build_raid_line, make_raid_drain_fn

_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Helpers to mimic the read sidecar /buffer wire shape
# --------------------------------------------------------------------------- #
def _wrap(seq: int, event: dict) -> dict:
    return {"seq": seq, "ts": 0.0, "event": event}


def _raid_event(from_id="R-1", from_login="raider", from_name="Raider", viewers=42) -> dict:
    return {
        "type": "raid",
        "from_login": from_login,
        "from_name": from_name,
        "from_broadcaster_user_id": from_id,
        "viewers": viewers,
    }


def _buffer_body(events: list, cursor: int) -> bytes:
    return json.dumps({"events": events, "cursor": cursor}).encode("utf-8")


# --------------------------------------------------------------------------- #
# make_raid_drain_fn
# --------------------------------------------------------------------------- #
def test_drain_parses_raid_events_and_advances_own_cursor() -> None:
    calls = {"urls": []}

    def http_get(url: str, to: float) -> bytes:
        calls["urls"].append(url)
        # First call returns one raid (+ a chat event that must be ignored).
        if "since=0" in url:
            return _buffer_body(
                [_wrap(1, {"type": "chat", "text": "hi"}),
                 _wrap(2, _raid_event(viewers=7))],
                cursor=2,
            )
        # Second call (since=2) returns nothing new.
        return _buffer_body([], cursor=2)

    drain = make_raid_drain_fn("http://127.0.0.1:8773", http_get=http_get)
    first = drain()
    assert [e["type"] for e in first] == ["raid"]
    assert first[0]["viewers"] == 7
    # Own cursor advanced to 2; the next call asks since=2 and NEVER posts /ack.
    second = drain()
    assert second == []
    assert calls["urls"] == [
        "http://127.0.0.1:8773/buffer?since=0",
        "http://127.0.0.1:8773/buffer?since=2",
    ]


def test_drain_fail_safe_on_down_sidecar_and_bad_body() -> None:
    def boom(url: str, to: float) -> bytes:
        raise OSError("connection refused")

    assert make_raid_drain_fn("http://127.0.0.1:8773", http_get=boom)() == []

    def garbage(url: str, to: float) -> bytes:
        return b"not json"

    assert make_raid_drain_fn("http://127.0.0.1:8773", http_get=garbage)() == []


# --------------------------------------------------------------------------- #
# build_raid_line — every beat
# --------------------------------------------------------------------------- #
def test_build_raid_line_covers_all_beats() -> None:
    line = build_raid_line("CoolStreamer", 42)
    low = line.lower()
    # (1) announce the raid (+ raider name + viewer count)
    assert "raid" in low and "CoolStreamer" in line and "42" in line
    # (2) thank the raider
    assert "thanks" in low or "thank" in low
    # (3) welcome the raiders
    assert "welcome" in low
    # (4) hope they stick around
    assert "stay" in low or "remain" in low
    # (5) introduce himself
    assert "i am ultron" in low
    # (6) how to chat with him
    assert "ultron" in low and "question" in low
    # Persona: no vendor/model/"AI"/"bot"/"assistant" leak (BR-P2).
    for banned in ("kenning", "assistant", " ai ", "model", "language model", "bot"):
        assert banned not in low


def test_build_raid_line_pluralizes_and_handles_blank_name() -> None:
    assert "one viewer" in build_raid_line("Solo", 1)
    assert "viewers" in build_raid_line("Crowd", 5)
    blank = build_raid_line("", 0)
    assert "another broadcaster" in blank
    # An unknown viewer count (0) omits a bogus "0 viewers".
    assert "0 viewer" not in blank


# --------------------------------------------------------------------------- #
# RaidHandler.tick — announce + shoutout + idempotency + fail-open
# --------------------------------------------------------------------------- #
class _Recorder:
    def __init__(self) -> None:
        self.announces: list[str] = []
        self.shoutouts: list[str] = []

    def announce(self, text: str) -> None:
        self.announces.append(text)

    def shoutout(self, tid: str) -> None:
        self.shoutouts.append(tid)


def test_handler_inject_processes_synthetic_raid_on_next_tick() -> None:
    # 2026-06-26 dev TEST PANEL seam: inject() queues a synthetic raid that the
    # next tick announces through the SAME path as a live drain.
    rec = _Recorder()
    h = RaidHandler(lambda: [], announce_fn=rec.announce,
                    shoutout_fn=rec.shoutout, shoutout_enabled=True)
    h.inject({"type": "raid", "from_login": "tester", "from_name": "Tester",
              "from_broadcaster_user_id": "T-1", "viewers": 7})
    handled = h.tick()
    assert handled == 1
    assert rec.announces and "Tester" in rec.announces[0] and "7" in rec.announces[0]
    assert rec.shoutouts == ["T-1"]
    # The buffer is consumed -> a second tick is a no-op.
    assert h.tick() == 0


def test_handler_announces_on_stream_bus_and_shoutouts() -> None:
    rec = _Recorder()
    events = [_raid_event(from_id="R-9", from_name="Nova", viewers=12)]
    h = RaidHandler(lambda: list(events), announce_fn=rec.announce,
                    shoutout_fn=rec.shoutout, shoutout_enabled=True)
    handled = h.tick()
    assert handled == 1
    assert len(rec.announces) == 1
    assert "Nova" in rec.announces[0] and "12" in rec.announces[0]
    # The shoutout targets the raider's broadcaster id.
    assert rec.shoutouts == ["R-9"]


def test_handler_is_idempotent_across_replays() -> None:
    rec = _Recorder()
    ev = _raid_event(from_id="R-1", viewers=3)
    # Both ticks return the SAME raid (an EventSub replay).
    h = RaidHandler(lambda: [dict(ev)], announce_fn=rec.announce,
                    shoutout_fn=rec.shoutout)
    assert h.tick() == 1
    assert h.tick() == 0   # replay deduped -> not handled again
    assert len(rec.announces) == 1
    assert rec.shoutouts == ["R-1"]


def test_handler_shoutout_disabled_skips_shoutout_but_still_announces() -> None:
    rec = _Recorder()
    h = RaidHandler(lambda: [_raid_event()], announce_fn=rec.announce,
                    shoutout_fn=rec.shoutout, shoutout_enabled=False)
    h.tick()
    assert len(rec.announces) == 1
    assert rec.shoutouts == []   # disabled -> never called


def test_handler_announce_error_never_blocks_shoutout() -> None:
    rec = _Recorder()

    def bad_announce(_text: str) -> None:
        raise RuntimeError("tts down")

    h = RaidHandler(lambda: [_raid_event(from_id="R-7")], announce_fn=bad_announce,
                    shoutout_fn=rec.shoutout)
    # The announce raises internally but the handler swallows it and STILL shouts.
    assert h.tick() == 1
    assert rec.shoutouts == ["R-7"]


def test_handler_shoutout_error_never_raises_or_blocks_announce() -> None:
    rec = _Recorder()

    def bad_shoutout(_tid: str) -> None:
        raise RuntimeError("helix 500")

    h = RaidHandler(lambda: [_raid_event()], announce_fn=rec.announce,
                    shoutout_fn=bad_shoutout)
    # The shoutout raises but the announce already happened and tick never raises.
    assert h.tick() == 1
    assert len(rec.announces) == 1


def test_handler_drain_error_is_fail_safe() -> None:
    rec = _Recorder()

    def bad_drain() -> list:
        raise RuntimeError("drain boom")

    h = RaidHandler(bad_drain, announce_fn=rec.announce, shoutout_fn=rec.shoutout)
    assert h.tick() == 0
    assert rec.announces == [] and rec.shoutouts == []


def test_handler_ignores_non_raid_events() -> None:
    rec = _Recorder()
    h = RaidHandler(lambda: [{"type": "redeem"}, {"not": "a dict-shape raid"}],
                    announce_fn=rec.announce, shoutout_fn=rec.shoutout)
    assert h.tick() == 0
    assert rec.announces == []


def test_handler_no_shoutout_when_raider_id_missing() -> None:
    rec = _Recorder()
    ev = _raid_event(from_id="", from_name="Anon", viewers=4)
    h = RaidHandler(lambda: [ev], announce_fn=rec.announce, shoutout_fn=rec.shoutout)
    assert h.tick() == 1
    assert len(rec.announces) == 1     # still announces
    assert rec.shoutouts == []         # but no id -> no shoutout


# --------------------------------------------------------------------------- #
# HelixClient.send_shoutout
# --------------------------------------------------------------------------- #
class _ScriptedTransport:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._queue: list[TransportResponse] = []
        self.default = TransportResponse(status=204, body="")

    def queue(self, *responses: TransportResponse) -> "_ScriptedTransport":
        self._queue.extend(responses)
        return self

    def __call__(self, method, url, headers, body):
        self.calls.append({"method": method, "url": url})
        if self._queue:
            return self._queue.pop(0)
        return self.default


def _client(transport) -> HelixClient:
    return HelixClient("cid", get_token=lambda: "tok", transport=transport)


def test_send_shoutout_204_ok_with_three_ids() -> None:
    t = _ScriptedTransport().queue(TransportResponse(status=204, body=""))
    res = _client(t).send_shoutout("B-1", "R-2", "M-1")
    assert res.ok and not res.idempotent and res.action == "shoutout"
    call = t.calls[0]
    assert call["method"] == "POST" and "/chat/shoutouts" in call["url"]
    assert "from_broadcaster_id=B-1" in call["url"]
    assert "to_broadcaster_id=R-2" in call["url"]
    assert "moderator_id=M-1" in call["url"]


def test_send_shoutout_cooldown_429_is_idempotent_not_loud() -> None:
    # The shoutout cooldown surfaces as a 429 AFTER the retry path is exhausted;
    # the client treats it as already-applied (ok + idempotent), never raising.
    t = _ScriptedTransport()
    # Queue enough 429s to exhaust retries (max_retries default 4 -> 5 calls).
    t.queue(*[TransportResponse(status=429, body="shoutout cooldown") for _ in range(6)])
    # Inject a no-op sleep + a manual clock so the backoff never actually waits.
    clock = {"t": 0.0}
    client = HelixClient(
        "cid", get_token=lambda: "tok", transport=t,
        base_backoff_s=0.0, max_backoff_s=0.0,
        monotonic=lambda: clock["t"],
        sleep=lambda dt: clock.__setitem__("t", clock["t"] + dt),
    )
    res = client.send_shoutout("B-1", "R-2", "M-1")
    assert res.ok and res.idempotent and res.status == 429


def test_send_shoutout_already_body_is_idempotent() -> None:
    t = _ScriptedTransport().queue(
        TransportResponse(status=400, body=json.dumps(
            {"message": "shoutout already sent to this broadcaster"})))
    res = _client(t).send_shoutout("B-1", "R-2", "M-1")
    assert res.ok and res.idempotent


def test_send_shoutout_local_cache_short_circuits() -> None:
    t = _ScriptedTransport().queue(TransportResponse(status=204, body=""))
    client = _client(t)
    r1 = client.send_shoutout("B-1", "R-2", "M-1")
    r2 = client.send_shoutout("B-1", "R-2", "M-1")  # same raider -> cache hit
    assert r1.ok and r2.ok and r2.idempotent and r2.status == 0
    assert len(t.calls) == 1   # the second call never hit the network


def test_send_shoutout_requires_all_ids() -> None:
    client = _client(_ScriptedTransport())
    for args in (("", "R", "M"), ("B", "", "M"), ("B", "R", "")):
        with pytest.raises(ValueError):
            client.send_shoutout(*args)


# --------------------------------------------------------------------------- #
# helix_eventsub.create_raid_subscription
# --------------------------------------------------------------------------- #
def test_create_raid_subscription_builds_to_broadcaster_condition() -> None:
    from kenning.twitch.clients.helix_eventsub import (
        HelixEventSubClient,
        RAID_SUBSCRIPTION_TYPE,
    )

    seen = {}

    def transport(method, url, headers, body):
        seen["method"] = method
        seen["url"] = url
        seen["body"] = json.loads(body.decode()) if body else None
        return (202, b"{}")

    client = HelixEventSubClient("cid", transport=transport)
    ok = client.create_raid_subscription(
        broadcaster_id="B-100", session_id="sess-1", token="btok")
    assert ok is True
    assert seen["body"]["type"] == RAID_SUBSCRIPTION_TYPE
    # The 'to' side binds via to_broadcaster_user_id (needs no special scope).
    assert seen["body"]["condition"] == {"to_broadcaster_user_id": "B-100"}
    assert seen["body"]["transport"] == {"method": "websocket", "session_id": "sess-1"}


def test_create_raid_subscription_missing_ids_is_false() -> None:
    from kenning.twitch.clients.helix_eventsub import HelixEventSubClient

    client = HelixEventSubClient("cid", transport=lambda *a: (202, b"{}"))
    assert client.create_raid_subscription(
        broadcaster_id="", session_id="s", token="t") is False
    assert client.create_raid_subscription(
        broadcaster_id="B", session_id="", token="t") is False


# --------------------------------------------------------------------------- #
# Read sidecar: subscribe + map channel.raid on the broadcaster-token session
# --------------------------------------------------------------------------- #
def _load_read_sidecar():
    path = _ROOT / "scripts" / "twitch_read_sidecar.py"
    spec = importlib.util.spec_from_file_location("twitch_read_sidecar", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("twitch_read_sidecar", mod)
    spec.loader.exec_module(mod)
    return mod


read_sidecar = _load_read_sidecar()


class _FakeWSClient:
    def __init__(self, messages):
        self._queue = list(messages)
        self.closed = False

    def recv_json_ready(self, timeout):
        if not self._queue:
            return None
        return self._queue.pop(0)

    def close(self):
        self.closed = True


class _FakeHelix:
    def __init__(self):
        self._ids = {"streamer": "B-100", "ultronbot": "U-200"}
        self.chat_subs = []
        self.redeem_subs = []
        self.raid_subs = []

    def get_user_id(self, login, *, token):
        return self._ids.get(login)

    def create_chat_subscription(self, *, broadcaster_id, bot_user_id, session_id, token):
        self.chat_subs.append((broadcaster_id, bot_user_id, session_id, token))
        return True

    def create_redeem_subscription(self, *, broadcaster_id, session_id, token):
        self.redeem_subs.append((broadcaster_id, session_id, token))
        return True

    def create_raid_subscription(self, *, broadcaster_id, session_id, token):
        self.raid_subs.append((broadcaster_id, session_id, token))
        return True


def _welcome(session_id="sess-1") -> dict:
    return {
        "metadata": {"message_type": "session_welcome"},
        "payload": {"session": {"id": session_id, "status": "connected",
                                "keepalive_timeout_seconds": 10}},
    }


def _raid_notification(from_id="R-1", login="raider", name="Raider", viewers=88) -> dict:
    return {
        "metadata": {"message_type": "notification", "subscription_type": "channel.raid"},
        "payload": {
            "subscription": {"type": "channel.raid"},
            "event": {
                "from_broadcaster_user_id": from_id,
                "from_broadcaster_user_login": login,
                "from_broadcaster_user_name": name,
                "to_broadcaster_user_id": "B-100",
                "to_broadcaster_user_login": "streamer",
                "viewers": viewers,
            },
        },
    }


def _patch_tokens(monkeypatch) -> None:
    import kenning.twitch.auth as auth_mod

    class _Store:
        def __init__(self, path=None):
            self.path = path

        def load(self):
            return {"access_token": "tok-" + str(self.path)}

    monkeypatch.setattr(auth_mod, "TokenStore", _Store)


def test_read_sidecar_subscribes_and_maps_raid(monkeypatch) -> None:
    _patch_tokens(monkeypatch)
    helix = _FakeHelix()
    # Raids ride the broadcaster-token session (the redeem session). With raids ON
    # the broadcaster-session connects, subscribes on welcome, and maps the raid.
    raid_ws = _FakeWSClient([_welcome("sess-raid"),
                             _raid_notification("R-5", "nova", "Nova", 55)])
    src = read_sidecar.EventSubChatSource(
        url="wss://test/ws",
        client_id="cid",
        broadcaster_login="streamer",
        bot_login="ultronbot",
        subscribe_redeems=False,
        subscribe_raids=True,
        # The chat session has nothing to do; only the broadcaster session matters.
        connect_factory=lambda url: _FakeWSClient([]),
        redeem_connect_factory=lambda url: raid_ws,
        helix_factory=lambda: helix,
    )
    out = src.poll()
    raids = [e for e in out if e["type"] == "raid"]
    assert raids == [
        {
            "type": "raid",
            "from_login": "nova",
            "from_name": "Nova",
            "from_broadcaster_user_id": "R-5",
            "viewers": 55,
        }
    ]
    # The raid sub was created on the broadcaster session with the broadcaster token.
    assert helix.raid_subs == [("B-100", "sess-raid", "tok-~/.kenning/twitch.json")]


def test_read_sidecar_dedups_replayed_raid(monkeypatch) -> None:
    _patch_tokens(monkeypatch)
    helix = _FakeHelix()
    raid_ws = _FakeWSClient([
        _welcome("sess-raid"),
        _raid_notification("Rdup", "dup", "Dup", 9),
        _raid_notification("Rdup", "dup", "Dup", 9),  # same raid -> dropped
    ])
    src = read_sidecar.EventSubChatSource(
        url="wss://test/ws",
        client_id="cid",
        broadcaster_login="streamer",
        bot_login="ultronbot",
        subscribe_raids=True,
        connect_factory=lambda url: _FakeWSClient([]),
        redeem_connect_factory=lambda url: raid_ws,
        helix_factory=lambda: helix,
    )
    out = src.poll()
    assert [e["from_broadcaster_user_id"] for e in out if e["type"] == "raid"] == ["Rdup"]


# --------------------------------------------------------------------------- #
# Write sidecar /shoutout endpoint
# --------------------------------------------------------------------------- #
def _load_write_sidecar():
    path = _ROOT / "scripts" / "twitch_write_sidecar.py"
    spec = importlib.util.spec_from_file_location("twitch_write_sidecar", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("twitch_write_sidecar", mod)
    spec.loader.exec_module(mod)
    return mod


write_sidecar = _load_write_sidecar()


class _Served:
    def __init__(self, service=None, **kw):
        self.server, self.store = write_sidecar.build_server(service, port=0, **kw)
        self.host, self.port = self.server.server_address[:2]
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(timeout=3.0)

    @property
    def base(self) -> str:
        return f"http://{self.host}:{self.port}"


def _post(url: str, body) -> tuple[int, object]:
    data = json.dumps(body).encode("utf-8")
    req = Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=5) as resp:  # noqa: S310 — loopback only
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        payload = exc.read().decode("utf-8")
        try:
            return exc.code, json.loads(payload)
        except (ValueError, TypeError):
            return exc.code, payload


def test_write_sidecar_shoutout_calls_fn() -> None:
    seen = []

    def shout(tid: str) -> bool:
        seen.append(tid)
        return True

    with _Served(None, shoutout=shout) as s:
        status, body = _post(f"{s.base}/shoutout", {"to_broadcaster_id": "R-42"})
    assert status == 200 and body == {"ok": True}
    assert seen == ["R-42"]


def test_write_sidecar_shoutout_unavailable_is_fail_open() -> None:
    # No shoutout callable wired (creds absent) -> 200 ok=false, never an error code.
    with _Served(None) as s:
        status, body = _post(f"{s.base}/shoutout", {"to_broadcaster_id": "R-1"})
    assert status == 200
    assert body["ok"] is False and body["error"] == "shoutout_unavailable"


def test_write_sidecar_shoutout_error_is_fail_open() -> None:
    def shout(_tid: str) -> bool:
        raise RuntimeError("helix down")

    with _Served(None, shoutout=shout) as s:
        status, body = _post(f"{s.base}/shoutout", {"to_broadcaster_id": "R-1"})
    # Fail-open: the endpoint swallows the error and returns 200 ok=false.
    assert status == 200 and body["ok"] is False and body["error"] == "shoutout_error"


def test_write_sidecar_shoutout_rejects_empty_target() -> None:
    with _Served(None, shoutout=lambda t: True) as s:
        status, body = _post(f"{s.base}/shoutout", {"to_broadcaster_id": "  "})
    assert status == 400 and body["ok"] is False
