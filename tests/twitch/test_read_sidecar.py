"""S1 — tests for the Twitch chat READ sidecar (scripts/twitch_read_sidecar.py).

Fully offline: a ``FakeSource`` feeds the rolling buffer and an ephemeral-port
(``port=0``) ``ThreadingHTTPServer`` is driven with ``urllib`` over loopback. No
live Twitch connection, no creds, no models.

Covered:
  * GET /healthz reports ok + buffered + cursor + source name.
  * /buffer drains injected events and the returned cursor advances.
  * ?since=N filters to events strictly after N.
  * POST /ack advances the consumer cursor and prunes acked events.
  * the rolling buffer respects maxlen (oldest-first eviction, dropped count).
  * the rolling buffer respects the TTL (time-expired events pruned).
  * the parent-watchdog helper returns 'dead' for a bogus pid (mocked) and
    'alive' for a live/unset pid (so it would NOT self-kill on doubt).
  * the server binds 127.0.0.1 ONLY (not 0.0.0.0).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

# --------------------------------------------------------------------------- #
# Load the sidecar module by path (scripts/ is not an importable package).
# --------------------------------------------------------------------------- #
_ROOT = Path(__file__).resolve().parents[2]
_SIDECAR_PATH = _ROOT / "scripts" / "twitch_read_sidecar.py"


def _load_sidecar():
    spec = importlib.util.spec_from_file_location("twitch_read_sidecar", _SIDECAR_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("twitch_read_sidecar", mod)
    spec.loader.exec_module(mod)
    return mod


sidecar = _load_sidecar()


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
class _Served:
    """Context manager: a built sidecar server serving on a background thread."""

    def __init__(self, source=None, **kw):
        # Tests pump the poll loop manually -> start_poll=False for determinism,
        # except where a test explicitly wants the live thread.
        kw.setdefault("start_poll", False)
        self.server, self.buffer, self.poll_loop = sidecar.build_server(source, port=0, **kw)
        self.host, self.port = self.server.server_address[:2]
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.poll_loop.stop()
        self._thread.join(timeout=3.0)

    @property
    def base(self) -> str:
        return f"http://{self.host}:{self.port}"


def _get(url: str) -> dict:
    with urlopen(url, timeout=5) as resp:  # noqa: S310 — loopback only
        assert resp.status == 200
        return json.loads(resp.read().decode("utf-8"))


def _post(url: str, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=5) as resp:  # noqa: S310 — loopback only
        assert resp.status == 200
        return json.loads(resp.read().decode("utf-8"))


def _chat(login: str, text: str, mid: str) -> dict:
    return {"type": "chat", "message_id": mid, "chatter_login": login, "text": text}


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_healthz_reports_state() -> None:
    src = sidecar.FakeSource()
    with _Served(src) as s:
        body = _get(f"{s.base}/healthz")
    assert body["ok"] is True
    assert body["buffered"] == 0
    assert body["cursor"] == 0
    assert body["source"] == "fake"
    # poll loop not started in this fixture -> running False
    assert body["running"] is False


def test_buffer_drain_returns_injected_events_and_advances_cursor() -> None:
    src = sidecar.FakeSource()
    with _Served(src) as s:
        src.push(_chat("alice", "hello", "m1"), _chat("bob", "gg", "m2"))
        appended = s.poll_loop.run_once()
        assert appended == 2

        body = _get(f"{s.base}/buffer")
        events = body["events"]
        assert [e["event"]["message_id"] for e in events] == ["m1", "m2"]
        assert [e["event"]["text"] for e in events] == ["hello", "gg"]
        # every event carries a monotonically increasing seq + ts
        assert [e["seq"] for e in events] == [1, 2]
        assert all(isinstance(e["ts"], (int, float)) for e in events)
        # cursor returned is the high-water mark of what was drained
        assert body["cursor"] == 2

        # healthz now reflects the 2 buffered events
        h = _get(f"{s.base}/healthz")
        assert h["buffered"] == 2
        assert h["cursor"] == 0  # not acked yet


def test_since_cursor_filters() -> None:
    src = sidecar.FakeSource()
    with _Served(src) as s:
        src.push(_chat("a", "1", "m1"), _chat("b", "2", "m2"), _chat("c", "3", "m3"))
        s.poll_loop.run_once()

        # ?since=1 -> only seq 2 and 3
        body = _get(f"{s.base}/buffer?since=1")
        assert [e["seq"] for e in body["events"]] == [2, 3]
        assert body["cursor"] == 3

        # ?since=3 -> nothing newer; cursor stays at the floor
        body = _get(f"{s.base}/buffer?since=3")
        assert body["events"] == []
        assert body["cursor"] == 3

        # garbage ?since -> falls back to the consumer cursor (0) -> all events
        body = _get(f"{s.base}/buffer?since=not_a_number")
        assert [e["seq"] for e in body["events"]] == [1, 2, 3]


def test_ack_advances_cursor_and_prunes() -> None:
    src = sidecar.FakeSource()
    with _Served(src) as s:
        src.push(_chat("a", "1", "m1"), _chat("b", "2", "m2"), _chat("c", "3", "m3"))
        s.poll_loop.run_once()

        # ack up to seq 2 -> events m1,m2 pruned, only m3 remains
        ack = _post(f"{s.base}/ack", {"cursor": 2})
        assert ack["ok"] is True
        assert ack["cursor"] == 2

        h = _get(f"{s.base}/healthz")
        assert h["cursor"] == 2
        assert h["buffered"] == 1  # only m3 left

        # /buffer with no ?since now uses the consumer cursor (2) -> only m3
        body = _get(f"{s.base}/buffer")
        assert [e["seq"] for e in body["events"]] == [3]

        # ack cannot regress the cursor
        ack2 = _post(f"{s.base}/ack", {"cursor": 1})
        assert ack2["cursor"] == 2


def test_rolling_buffer_respects_maxlen() -> None:
    buf = sidecar.RollingBuffer(maxlen=3, ttl_seconds=0)  # ttl off -> isolate maxlen
    for i in range(5):
        buf.append({"message_id": f"m{i}"})
    # only the last 3 survive (oldest-first eviction)
    events, cursor = buf.drain(since=0)
    assert [e["event"]["message_id"] for e in events] == ["m2", "m3", "m4"]
    assert [e["seq"] for e in events] == [3, 4, 5]
    assert cursor == 5
    # two events (m0, m1) were evicted unacked -> counted as drops
    assert buf.dropped_total == 2
    assert len(buf) == 3


def test_rolling_buffer_respects_ttl() -> None:
    buf = sidecar.RollingBuffer(maxlen=100, ttl_seconds=10.0)
    base = 1000.0
    buf.append({"message_id": "old"}, now=base)
    buf.append({"message_id": "fresh"}, now=base + 9.0)
    # at base+11: "old" (age 11s) is expired, "fresh" (age 2s) survives
    events, _cursor = buf.drain(since=0, now=base + 11.0)
    assert [e["event"]["message_id"] for e in events] == ["fresh"]
    assert len(buf) == 1
    # the expired, unacked event counts as a drop
    assert buf.dropped_total == 1


def test_parent_watchdog_dead_for_bogus_pid(monkeypatch) -> None:
    # A bogus pid that _pid_alive reports as gone -> watchdog says 'dead'
    # (the loop would then os._exit). We mock _pid_alive so we never depend on a
    # real OS pid table and never actually exit.
    monkeypatch.setattr(sidecar, "_pid_alive", lambda pid: False)
    assert sidecar.parent_watchdog_check(999_999_999) == "dead"

    # A pid reported alive -> 'alive' (never self-kill).
    monkeypatch.setattr(sidecar, "_pid_alive", lambda pid: True)
    assert sidecar.parent_watchdog_check(4321) == "alive"

    # An unset/invalid pid -> 'alive' (fail-safe: never self-kill on doubt),
    # WITHOUT consulting _pid_alive.
    def _boom(pid):  # pragma: no cover - must not be called
        raise AssertionError("_pid_alive should not be consulted for pid<=0")

    monkeypatch.setattr(sidecar, "_pid_alive", _boom)
    assert sidecar.parent_watchdog_check(0) == "alive"
    assert sidecar.parent_watchdog_check(-1) == "alive"


def test_pid_alive_failsafe_on_indeterminate(monkeypatch) -> None:
    # If psutil is absent and the OS probe raises, _pid_alive must fail SAFE (True)
    # so the watchdog never self-kills on an indeterminate result.
    import builtins

    real_import = builtins.__import__

    def _no_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("psutil intentionally absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_psutil)
    # pid<=0 short-circuits to True before any probe.
    assert sidecar._pid_alive(0) is True


def test_server_binds_loopback_only() -> None:
    with _Served(sidecar.FakeSource()) as s:
        # bound host must be the loopback address, never 0.0.0.0 / a routable iface
        assert s.host == "127.0.0.1"
        assert s.host != "0.0.0.0"
        # and it actually answers on loopback
        body = _get(f"{s.base}/healthz")
        assert body["ok"] is True


def test_unknown_route_404() -> None:
    import urllib.error

    with _Served(sidecar.FakeSource()) as s:
        with pytest.raises(urllib.error.HTTPError) as ei:
            _get(f"{s.base}/nope")
        assert ei.value.code == 404


def test_default_source_serves_empty_buffer() -> None:
    # build_server with no source -> empty FakeSource -> empty buffer (the
    # documented flag-off behaviour: run directly, serve nothing).
    with _Served() as s:
        s.poll_loop.run_once()  # nothing queued
        body = _get(f"{s.base}/buffer")
        assert body["events"] == []
        h = _get(f"{s.base}/healthz")
        assert h["buffered"] == 0
        assert h["source"] == "fake"


def test_poll_loop_thread_pumps_live(monkeypatch) -> None:
    # Exercise the real background thread (start_poll=True) end-to-end.
    src = sidecar.FakeSource()
    server, buffer, poll_loop = sidecar.build_server(
        src, port=0, poll_interval=0.01, start_poll=True
    )
    host, port = server.server_address[:2]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        assert poll_loop.running is True
        src.push(_chat("live", "from the thread", "live1"))
        # wait for the poll thread to pick it up
        deadline = time.time() + 3.0
        events = []
        while time.time() < deadline:
            body = _get(f"http://{host}:{port}/buffer?since=0")
            events = body["events"]
            if events:
                break
            time.sleep(0.02)
        assert [e["event"]["message_id"] for e in events] == ["live1"]
    finally:
        server.shutdown()
        server.server_close()
        poll_loop.stop()
        t.join(timeout=3.0)
    assert poll_loop.running is False
    assert src.closed is True  # stop() closed the source


def test_source_poll_raise_is_swallowed() -> None:
    # A source that raises on poll() must not crash the poll loop.
    class _Boom:
        name = "boom"

        def poll(self):
            raise RuntimeError("source exploded")

    buffer = sidecar.RollingBuffer()
    loop = sidecar.PollLoop(_Boom(), buffer, interval=0.01)
    assert loop.run_once() == 0  # swallowed, zero appended
    assert len(buffer) == 0
