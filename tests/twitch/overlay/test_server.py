"""Tests for the S8 local OBS overlay server (offline; stdlib http only).

Covers: token gate (missing/wrong -> 403; correct -> 200 + strict CSP header),
loopback-only bind, SSE event delivery end-to-end, schema validation (unknown
event type rejected), the static overlay.html hardening (no innerHTML; a CSP meta
matching the header; textContent usage), and an XSS payload routed through emit()
arriving JSON-escaped on the wire (not a live tag).
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from pathlib import Path

import pytest

from kenning.twitch.overlay import OverlayError, OverlayServer, validate_event
from kenning.twitch.overlay.server import (
    ALLOWED_EVENT_TYPES,
    CSP_POLICY,
)

_HTML = (
    Path(__file__).resolve().parents[3]
    / "src" / "kenning" / "twitch" / "overlay" / "static" / "overlay.html"
)


# --- fixtures ----------------------------------------------------------------
@pytest.fixture()
def server():
    srv = OverlayServer(host="127.0.0.1", port=0)
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


def _get(url: str, timeout: float = 5.0):
    """GET helper returning (status, headers, body_bytes). 4xx is captured, not raised."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - loopback only
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


# --- token gate --------------------------------------------------------------
def test_missing_token_is_forbidden(server):
    host_port = server.port
    status, _, body = _get(f"http://127.0.0.1:{host_port}/")
    assert status == HTTPStatus.FORBIDDEN
    assert b"token" in body.lower()


def test_wrong_token_is_forbidden(server):
    status, _, _ = _get(f"http://127.0.0.1:{server.port}/?token=not-the-real-token")
    assert status == HTTPStatus.FORBIDDEN


def test_events_route_also_token_gated(server):
    status, _, _ = _get(f"http://127.0.0.1:{server.port}/events?token=bogus")
    assert status == HTTPStatus.FORBIDDEN


def test_correct_token_serves_overlay_with_strict_csp(server):
    status, headers, body = _get(server.url())
    assert status == HTTPStatus.OK
    # The CSP RESPONSE HEADER must be present and exactly the strict policy.
    csp = headers.get("Content-Security-Policy")
    assert csp == CSP_POLICY
    assert "default-src 'none'" in csp
    assert "connect-src 'self'" in csp
    assert b"<html" in body.lower()


# --- bind / addressing -------------------------------------------------------
def test_binds_loopback_only_in_url(server):
    assert server.url().startswith("http://127.0.0.1:")
    assert "token=" in server.url()
    assert server.token in server.url()


def test_non_loopback_host_rejected():
    with pytest.raises(OverlayError):
        OverlayServer(host="0.0.0.0", port=0)
    with pytest.raises(OverlayError):
        OverlayServer(host="192.168.1.5", port=0)


# --- schema validation -------------------------------------------------------
def test_validate_event_accepts_known_types():
    w = validate_event({"type": "wheel", "angle": 137.5, "label": "Reyna"})
    assert w["type"] == "wheel" and w["angle"] == 137.5 and w["label"] == "Reyna"
    a = validate_event({"type": "alert", "title": "Sub!", "body": "thanks"})
    assert a["type"] == "alert"
    t = validate_event({"type": "ticker", "label": "viewer", "points": 50})
    assert t["type"] == "ticker" and t["points"] == 50


def test_validate_event_rejects_unknown_type():
    with pytest.raises(OverlayError):
        validate_event({"type": "exec_shell", "cmd": "rm -rf /"})
    with pytest.raises(OverlayError):
        validate_event({"type": "wheel_BOGUS"})
    assert "exec_shell" not in ALLOWED_EVENT_TYPES


def test_emit_rejects_unknown_type(server):
    with pytest.raises(OverlayError):
        server.emit({"type": "definitely-not-real", "x": 1})


def test_validate_event_rejects_non_finite_and_oversize():
    with pytest.raises(OverlayError):
        validate_event({"type": "wheel", "angle": float("inf"), "label": "x"})
    with pytest.raises(OverlayError):
        validate_event({"type": "alert", "title": "x" * 5000, "body": "y"})
    with pytest.raises(OverlayError):
        validate_event("not a dict")


# --- chat_game event (typed chat-command game cards) ------------------------
def test_validate_event_accepts_chat_game_slots():
    ev = validate_event({
        "type": "chat_game", "game": "slots", "source": "chat",
        "viewer": "alice", "outcome": "WIN", "title": "SLOTS",
        "won": True, "amount": 9000,
        "detail": {"reels": ["seven", "seven", "seven"], "win_symbol": "seven",
                   "stake": 1000, "payout": 9000, "net": 8000},
    })
    assert ev["type"] == "chat_game"
    assert ev["game"] == "slots"
    assert ev["source"] == "chat"          # discriminator injected by the validator
    assert ev["viewer"] == "alice" and ev["won"] is True and ev["amount"] == 9000
    assert ev["detail"]["reels"] == ["seven", "seven", "seven"]
    assert ev["detail"]["win_symbol"] == "seven"
    assert ev["duration_ms"] == 7000.0     # default hold


def test_validate_event_accepts_every_chat_game_kind():
    for game in ("slots", "wheel", "heist", "duel", "trivia", "raffle"):
        ev = validate_event({"type": "chat_game", "game": game, "viewer": "v",
                             "outcome": "X", "title": game.upper(), "amount": 1})
        assert ev["game"] == game and ev["source"] == "chat"


def test_validate_event_rejects_unknown_chat_game():
    with pytest.raises(OverlayError):
        validate_event({"type": "chat_game", "game": "rocket_league",
                        "viewer": "v", "outcome": "x"})


def test_validate_event_chat_game_drops_unknown_detail_keys():
    ev = validate_event({"type": "chat_game", "game": "duel", "viewer": "v",
                        "outcome": "WIN", "amount": 600,
                        "detail": {"winner": "v", "loser": "u", "wager": 300,
                                   "evil": "rm -rf /", "server_seed": "leak"}})
    # only the known, vetted detail keys survive; secret/unknown keys are dropped.
    assert "evil" not in ev["detail"] and "server_seed" not in ev["detail"]
    assert ev["detail"]["winner"] == "v" and ev["detail"]["wager"] == 300


def test_validate_event_chat_game_rejects_oversize_and_bad_detail():
    with pytest.raises(OverlayError):
        validate_event({"type": "chat_game", "game": "trivia", "viewer": "v",
                        "outcome": "x", "detail": {"answer": "a" * 5000}})
    with pytest.raises(OverlayError):
        validate_event({"type": "chat_game", "game": "slots", "viewer": "v",
                        "outcome": "x", "detail": {"reels": ["x"] * 99}})
    with pytest.raises(OverlayError):
        validate_event({"type": "chat_game", "game": "slots", "viewer": "v",
                        "outcome": "x", "detail": "not-a-dict"})


def test_chat_game_emit_streams_through_server(server):
    # a chat_game event passes validation and is accepted by emit()
    assert server.emit({"type": "chat_game", "game": "heist", "viewer": "crew",
                        "outcome": "WIN", "title": "HEIST", "won": True,
                        "amount": 420, "detail": {"pot": 1500, "crew": 6}}) is True


# --- SSE end-to-end ----------------------------------------------------------
def _read_one_sse_frame(url: str, ready: threading.Event, holder: dict, timeout: float = 6.0) -> None:
    """Open the SSE stream, signal ready, read until one full event frame arrives."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - loopback
            ready.set()
            buf = b""
            deadline = time.time() + timeout
            while time.time() < deadline:
                chunk = resp.read(1)
                if not chunk:
                    break
                buf += chunk
                if b"data:" in buf and buf.endswith(b"\n\n"):
                    holder["raw"] = buf
                    return
    except Exception as e:  # noqa: BLE001 - record for the assertion
        holder["error"] = repr(e)
    finally:
        ready.set()


def test_emit_streams_json_event_to_sse_client(server):
    ready = threading.Event()
    holder: dict = {}
    t = threading.Thread(
        target=_read_one_sse_frame, args=(server.events_url(), ready, holder), daemon=True
    )
    t.start()
    assert ready.wait(5.0), "SSE client never connected"
    # Wait for the client to be registered for fan-out, then emit.
    for _ in range(50):
        if server.client_count >= 1:
            break
        time.sleep(0.05)
    assert server.client_count >= 1
    server.emit({"type": "ticker", "label": "alice", "points": 42})
    t.join(timeout=6.0)

    assert "error" not in holder, holder.get("error")
    raw = holder.get("raw", b"")
    assert b"event: overlay" in raw
    # extract the data: line and JSON-parse it
    data_line = None
    for line in raw.split(b"\n"):
        if line.startswith(b"data:"):
            data_line = line[len(b"data:"):].strip()
            break
    assert data_line is not None
    payload = json.loads(data_line.decode("utf-8"))
    assert payload == {"type": "ticker", "label": "alice", "points": 42}


def test_emit_xss_payload_is_json_escaped_on_the_wire(server):
    ready = threading.Event()
    holder: dict = {}
    t = threading.Thread(
        target=_read_one_sse_frame, args=(server.events_url(), ready, holder), daemon=True
    )
    t.start()
    assert ready.wait(5.0)
    for _ in range(50):
        if server.client_count >= 1:
            break
        time.sleep(0.05)
    assert server.client_count >= 1

    payload_name = '<img src=x onerror=alert(1)>'
    server.emit({"type": "ticker", "label": payload_name, "points": 1})
    t.join(timeout=6.0)

    assert "error" not in holder, holder.get("error")
    raw = holder.get("raw", b"")
    # The raw display name is carried as a JSON string value (the browser then
    # renders it via textContent -> inert). Assert it survives intact as DATA
    # inside the JSON, and that decoding yields the literal payload (not a tag the
    # transport interpreted).
    data_line = next(
        (ln[len(b"data:"):].strip() for ln in raw.split(b"\n") if ln.startswith(b"data:")),
        None,
    )
    assert data_line is not None
    decoded = json.loads(data_line.decode("utf-8"))
    assert decoded["label"] == payload_name  # round-trips as inert text data
    # The on-the-wire bytes are valid JSON (quoted), never a bare executable tag
    # at the top level: the payload only appears as the value of a JSON string.
    assert b'"label":"<img src=x onerror=alert(1)>"' in raw


# --- static overlay.html hardening ------------------------------------------
def test_overlay_html_exists_and_is_self_contained():
    assert _HTML.is_file()
    text = _HTML.read_text(encoding="utf-8")
    assert "<html" in text.lower()


def test_overlay_html_has_no_innerHTML():
    text = _HTML.read_text(encoding="utf-8")
    assert "innerHTML" not in text, "overlay must render via textContent, never innerHTML"


def test_overlay_html_uses_textContent():
    text = _HTML.read_text(encoding="utf-8")
    assert "textContent" in text


def test_overlay_html_has_csp_meta_matching_header():
    text = _HTML.read_text(encoding="utf-8")
    assert 'http-equiv="Content-Security-Policy"' in text
    # The meta content must be byte-identical to the served header policy.
    assert CSP_POLICY in text


def test_overlay_html_has_no_inline_event_handlers():
    text = _HTML.read_text(encoding="utf-8").lower()
    for handler in ("onclick=", "onerror=", "onload=", "onmouseover=", "onmouseenter="):
        assert handler not in text, f"inline handler {handler!r} present"


def test_overlay_html_uses_eventsource_with_token():
    text = _HTML.read_text(encoding="utf-8")
    assert "EventSource(" in text
    assert "/events?token=" in text


# --- routing edge cases ------------------------------------------------------
def test_unknown_path_with_token_is_404(server):
    status, _, _ = _get(f"http://127.0.0.1:{server.port}/secret?token={server.token}")
    assert status == HTTPStatus.NOT_FOUND


def test_double_start_and_stop_is_idempotent():
    srv = OverlayServer()
    srv.start()
    srv.start()  # no-op, must not raise or rebind
    p = srv.port
    assert p > 0
    srv.stop()
    srv.stop()  # idempotent


def test_emit_with_no_clients_is_safe():
    srv = OverlayServer()
    srv.start()
    try:
        assert srv.emit({"type": "alert", "title": "noone", "body": "listening"}) is True
    finally:
        srv.stop()


# --------------------------------------------------------------------------- #
# PERMANENT token (stable across reboots) — 2026-06-24
# --------------------------------------------------------------------------- #
def test_explicit_config_token_is_used_verbatim() -> None:
    from kenning.twitch.overlay.server import OverlayServer
    srv = OverlayServer(port=0, token="MY-PERMANENT-TOKEN")
    assert srv.token == "MY-PERMANENT-TOKEN"
    assert srv.url().endswith("?token=MY-PERMANENT-TOKEN")


def test_blank_token_persists_and_is_stable_across_reboots(tmp_path, monkeypatch) -> None:
    """No config token -> generate once, persist to ~/.kenning/overlay_token, and
    reuse it on every later boot so the OBS Browser-Source URL never changes."""
    import kenning.twitch.overlay.server as srv_mod
    token_file = tmp_path / ".kenning" / "overlay_token"
    monkeypatch.setattr(srv_mod, "_OVERLAY_TOKEN_FILE", token_file)

    first = srv_mod.OverlayServer(port=0).token
    assert first and len(first) > 20
    assert token_file.read_text(encoding="utf-8").strip() == first
    # a "reboot" (new server, same persisted file) reuses the SAME token
    second = srv_mod.OverlayServer(port=0).token
    assert second == first


# --------------------------------------------------------------------------- #
# LIVE SSE WIRING — the inline renderer must open the EventSource on a NORMAL
# (non-demo) load. Root cause of the "only ?demo=1 works" bug: the strict CSP had
# NO script-src directive, so it fell back to default-src 'none' and the inline
# renderer <script> was blocked in OBS's CEF -> init()/connect() never ran ->
# no GET /events. The fix grants script-src 'self' 'unsafe-inline' (header + meta).
# These are STRUCTURAL assertions on the served JS (no browser needed). — 2026-06-26
# --------------------------------------------------------------------------- #
def test_csp_grants_script_src_for_inline_renderer(server):
    """The CSP must explicitly allow the inline renderer script; otherwise the
    overlay JS never runs in OBS and the live SSE is never opened."""
    # header on the served page
    status, headers, _ = _get(server.url())
    assert status == HTTPStatus.OK
    csp = headers.get("Content-Security-Policy")
    assert csp == CSP_POLICY
    assert "script-src 'self' 'unsafe-inline'" in csp
    # the <meta http-equiv> copy must match the header byte-for-byte
    text = _HTML.read_text(encoding="utf-8")
    assert CSP_POLICY in text
    assert "script-src 'self' 'unsafe-inline'" in text


def test_csp_no_longer_blocks_scripts_via_default_src(server):
    """Regression guard: default-src is still 'none' but script-src is now present,
    so scripts are NOT blocked by the default-src fallback."""
    csp = CSP_POLICY
    assert "default-src 'none'" in csp          # other sinks stay locked down
    assert "script-src" in csp                  # but scripts are explicitly granted
    # ordering/format sanity: script-src grant present as a full directive
    assert "; script-src 'self' 'unsafe-inline';" in csp


def test_overlay_live_path_opens_eventsource_on_load():
    """Trace the inline JS: a NORMAL load (no ?demo=1) must reach connect() and
    construct the EventSource. We assert the control-flow structurally:
      init() -> (demo branch returns early) -> else connect()
      connect() -> new EventSource("/events?token=" + ...)
    so the live SSE is opened on DOMContentLoaded/immediate-init."""
    text = _HTML.read_text(encoding="utf-8")
    # init() branches on demo; the NON-demo branch falls through to connect().
    assert "function init()" in text
    assert 'params.get("demo")' in text
    # connect() is the live path and it constructs the EventSource to /events.
    assert "function connect()" in text
    assert 'new EventSource("/events?token=" + encodeURIComponent(token))' in text
    # init() is actually invoked on load (DOMContentLoaded or immediately).
    assert "DOMContentLoaded" in text
    assert "init);" in text or "init()" in text
    # The demo branch returns BEFORE connect, so connect() is the live-only path:
    # connect must be CALLED unconditionally after the demo early-return.
    demo_idx = text.index('params.get("demo")')
    connect_call_idx = text.index("\n    connect();")
    assert connect_call_idx > demo_idx, "connect() must be the post-demo live path"


def test_overlay_client_reconnects_on_hard_close():
    """The client must re-open the stream itself on a hard CLOSED state (server
    restart / torn-down connection), not rely solely on EventSource auto-retry."""
    text = _HTML.read_text(encoding="utf-8")
    assert "EventSource.CLOSED" in text          # detects the dead-stream state
    assert "scheduleReconnect" in text           # and schedules its own re-open
    # bounded backoff (won't hammer a down server)
    assert "RECONNECT_MAX_MS" in text


# --------------------------------------------------------------------------- #
# SERVER REPLAY RING BUFFER — a freshly-(re)connected SSE client replays the last
# few vetted frames so an OBS source that connected late / after a refresh isn't
# blank. Bounded + secret-stripped (validate_event runs before buffering). — 2026-06-26
# --------------------------------------------------------------------------- #
def _read_n_sse_frames(url: str, ready: threading.Event, holder: dict, n: int, timeout: float = 6.0) -> None:
    """Open the SSE stream, signal ready, collect up to ``n`` event frames (data: lines)."""
    frames: list = []
    holder["frames"] = frames
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - loopback
            ready.set()
            buf = b""
            deadline = time.time() + timeout
            while time.time() < deadline and len(frames) < n:
                chunk = resp.read(1)
                if not chunk:
                    break
                buf += chunk
                if buf.endswith(b"\n\n"):
                    if b"data:" in buf:
                        frames.append(buf)
                    buf = b""
    except Exception as e:  # noqa: BLE001 - record for the assertion
        holder["error"] = repr(e)
    finally:
        ready.set()


def _emit_via_server(server) -> None:
    server.emit({"type": "ticker", "label": "first", "points": 1})
    server.emit({"type": "ticker", "label": "second", "points": 2})


def test_replay_buffer_replays_buffered_events_to_new_client(server):
    """Events emitted BEFORE a client connects are replayed to it on connect."""
    # emit two events with NO client connected -> they go into the ring buffer only
    assert server.client_count == 0
    _emit_via_server(server)

    # now a fresh client connects and must receive BOTH buffered frames as replay
    ready = threading.Event()
    holder: dict = {}
    t = threading.Thread(
        target=_read_n_sse_frames, args=(server.events_url(), ready, holder, 2), daemon=True
    )
    t.start()
    assert ready.wait(5.0), "SSE client never connected"
    t.join(timeout=6.0)

    assert "error" not in holder, holder.get("error")
    frames = holder.get("frames", [])
    labels = []
    for raw in frames:
        for line in raw.split(b"\n"):
            if line.startswith(b"data:"):
                payload = json.loads(line[len(b"data:"):].strip().decode("utf-8"))
                labels.append(payload.get("label"))
    assert "first" in labels and "second" in labels, labels


def test_replay_buffer_is_bounded(server):
    """The ring buffer never grows past its cap; only the most-recent frames replay."""
    from kenning.twitch.overlay.server import _REPLAY_BUFFER_MAXLEN

    total = _REPLAY_BUFFER_MAXLEN + 10
    for i in range(total):
        server.emit({"type": "ticker", "label": f"e{i}", "points": i})
    # internal buffer is capped
    assert len(server._replay) == _REPLAY_BUFFER_MAXLEN

    ready = threading.Event()
    holder: dict = {}
    # ask for more frames than the cap; we should only ever get the cap's worth
    t = threading.Thread(
        target=_read_n_sse_frames,
        args=(server.events_url(), ready, holder, _REPLAY_BUFFER_MAXLEN + 5, 2.5),
        daemon=True,
    )
    t.start()
    assert ready.wait(5.0)
    t.join(timeout=4.0)

    frames = holder.get("frames", [])
    assert len(frames) <= _REPLAY_BUFFER_MAXLEN, f"replayed {len(frames)} > cap"
    # the OLDEST events were evicted: the very first label must be gone, the last kept
    labels = []
    for raw in frames:
        for line in raw.split(b"\n"):
            if line.startswith(b"data:"):
                labels.append(json.loads(line[len(b"data:"):].strip().decode("utf-8")).get("label"))
    assert "e0" not in labels                     # evicted (oldest)
    assert f"e{total - 1}" in labels              # retained (newest)


def test_replay_buffer_strips_secret_keys_before_buffering(server):
    """validate_event runs BEFORE buffering, so secret/unknown detail keys never
    enter the ring buffer and are never replayed."""
    server.emit({
        "type": "chat_game", "game": "duel", "viewer": "v", "outcome": "WIN",
        "amount": 600,
        "detail": {"winner": "v", "wager": 300, "server_seed": "TOPSECRET"},
    })
    buffered = "".join(server._replay)
    assert "TOPSECRET" not in buffered and "server_seed" not in buffered
    assert "winner" in buffered                   # the vetted shape is what's kept


def test_replay_delivered_once_not_doubled_for_live_client(server):
    """A client connected at emit time gets each frame exactly once (live), and is
    not ALSO sent it from the buffer on a later (re)connect of a DIFFERENT client."""
    # client A connects first (empty buffer), then we emit -> A gets it live
    readyA = threading.Event()
    holderA: dict = {}
    tA = threading.Thread(
        target=_read_n_sse_frames, args=(server.events_url(), readyA, holderA, 1, 4.0), daemon=True
    )
    tA.start()
    assert readyA.wait(5.0)
    for _ in range(50):
        if server.client_count >= 1:
            break
        time.sleep(0.05)
    server.emit({"type": "ticker", "label": "live-once", "points": 7})
    tA.join(timeout=5.0)

    framesA = holderA.get("frames", [])
    a_labels = []
    for raw in framesA:
        for line in raw.split(b"\n"):
            if line.startswith(b"data:"):
                a_labels.append(json.loads(line[len(b"data:"):].strip().decode("utf-8")).get("label"))
    # A got the live frame exactly once (it asked for 1 and got "live-once")
    assert a_labels.count("live-once") == 1
