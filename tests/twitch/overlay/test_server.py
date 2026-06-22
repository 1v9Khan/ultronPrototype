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
