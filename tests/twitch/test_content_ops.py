"""S12 — content-ops tests (Stream Markers / clips + highlight scorer).

Fully OFFLINE: the Helix transport is a deterministic mock callable; the
HighlightScorer is driven by injected timestamps. No network, no creds, no models.

ANTICHEAT (BR-P1): the no-capture scan asserts the module's source imports no
screen/video-capture or desktop-automation library.
"""
from __future__ import annotations

import ast
import inspect

import pytest

from kenning.twitch import content_ops
from kenning.twitch.content_ops import (
    MARKER_DESCRIPTION_MAX,
    HighlightScorer,
    create_clip,
    create_stream_marker,
)


# --------------------------------------------------------------------------- #
# A tiny recording mock for the injected helix(method, path, body) -> dict.
# --------------------------------------------------------------------------- #
class _MockHelix:
    """Records every call and returns a scripted response (or raises)."""

    def __init__(self, response=None, *, raises: Exception | None = None):
        self.response = response
        self.raises = raises
        self.calls: list[tuple[str, str, object]] = []

    def __call__(self, method, path, body):
        self.calls.append((method, path, body))
        if self.raises is not None:
            raise self.raises
        return self.response


# --------------------------------------------------------------------------- #
# Stream markers
# --------------------------------------------------------------------------- #
def test_create_stream_marker_success():
    helix = _MockHelix({"data": [{"id": "marker-123", "position_seconds": 244}]})
    result = create_stream_marker("9001", description="ace clutch", helix=helix)

    assert result["ok"] is True
    assert result["id"] == "marker-123"
    assert result["marker"]["position_seconds"] == 244
    assert result["description_truncated"] is False
    # Correct verb/endpoint and the description rode through in the body.
    method, path, body = helix.calls[0]
    assert method == "POST"
    assert path == "/streams/markers"
    assert body["user_id"] == "9001"
    assert body["description"] == "ace clutch"


def test_create_stream_marker_truncates_long_description():
    helix = _MockHelix({"data": [{"id": "m1"}]})
    long_desc = "x" * 300
    result = create_stream_marker("9001", description=long_desc, helix=helix)

    assert result["ok"] is True
    assert result["description_truncated"] is True
    _, _, body = helix.calls[0]
    assert len(body["description"]) == MARKER_DESCRIPTION_MAX == 140
    assert body["description"] == "x" * 140


def test_create_stream_marker_omits_empty_description():
    helix = _MockHelix({"data": [{"id": "m2"}]})
    result = create_stream_marker("9001", helix=helix)

    assert result["ok"] is True
    _, _, body = helix.calls[0]
    assert "description" not in body  # empty desc is not sent


def test_create_stream_marker_helix_error_is_structured_not_raised():
    helix = _MockHelix(raises=RuntimeError("helix 404 channel offline"))
    # Must NOT raise — content-ops is best-effort and returns a structured failure.
    result = create_stream_marker("9001", description="d", helix=helix)

    assert result["ok"] is False
    assert result["action"] == "create_stream_marker"
    assert "helix 404 channel offline" in result["error"]
    assert result["error_type"] == "RuntimeError"


def test_create_stream_marker_empty_response_is_structured_failure():
    helix = _MockHelix({"data": []})  # channel offline -> no marker
    result = create_stream_marker("9001", helix=helix)

    assert result["ok"] is False
    assert "empty response" in result["error"]


def test_create_stream_marker_requires_broadcaster_id():
    helix = _MockHelix({"data": [{"id": "m"}]})
    result = create_stream_marker("", helix=helix)

    assert result["ok"] is False
    assert "broadcaster_id" in result["error"]
    assert helix.calls == []  # never hit the transport on bad input


# --------------------------------------------------------------------------- #
# Clips
# --------------------------------------------------------------------------- #
def test_create_clip_success():
    helix = _MockHelix(
        {"data": [{"id": "AbCdEf", "edit_url": "https://clips.twitch.tv/AbCdEf/edit"}]}
    )
    result = create_clip("9001", helix=helix)

    assert result["ok"] is True
    assert result["id"] == "AbCdEf"
    assert result["edit_url"] == "https://clips.twitch.tv/AbCdEf/edit"
    # vod_offset is populated server-side later, so it is flagged pending now.
    assert result["vod_offset_pending"] is True
    method, path, body = helix.calls[0]
    assert method == "POST"
    assert path.startswith("/clips")
    assert "broadcaster_id=9001" in path
    assert body is None


def test_create_clip_helix_error_is_structured_not_raised():
    helix = _MockHelix(raises=ConnectionError("network down"))
    result = create_clip("9001", helix=helix)

    assert result["ok"] is False
    assert result["action"] == "create_clip"
    assert "network down" in result["error"]
    assert result["error_type"] == "ConnectionError"


def test_create_clip_missing_id_is_structured_failure():
    helix = _MockHelix({"data": [{"edit_url": "https://x/edit"}]})  # no id
    result = create_clip("9001", helix=helix)

    assert result["ok"] is False
    assert "missing clip id" in result["error"]


def test_create_clip_requires_broadcaster_id():
    helix = _MockHelix({"data": [{"id": "x"}]})
    result = create_clip("", helix=helix)

    assert result["ok"] is False
    assert helix.calls == []


# --------------------------------------------------------------------------- #
# HighlightScorer — deterministic chat-rate spike detection
# --------------------------------------------------------------------------- #
def test_highlight_scorer_steady_chat_is_neutral():
    scorer = HighlightScorer(window_seconds=10.0)
    # One message every second for 10s -> steady, no spike.
    for t in range(0, 10):
        scorer.note_message(float(t))
    s = scorer.score(now=10.0)
    assert 0.5 <= s <= 1.6  # near 1.0, definitely not a spike
    assert scorer.should_mark(now=10.0, threshold=2.0) is False


def test_highlight_scorer_detects_spike():
    scorer = HighlightScorer(window_seconds=10.0)
    # Sparse baseline across the first 8s (one msg/sec for 5 messages)...
    for t in (0.0, 2.0, 4.0, 6.0, 7.5):
        scorer.note_message(t)
    # ...then a burst in the last 2s (the recent sub-window of a 10s window).
    for _ in range(20):
        scorer.note_message(9.5)
    assert scorer.score(now=10.0) >= 2.0
    assert scorer.should_mark(now=10.0, threshold=2.0) is True


def test_highlight_scorer_cold_start_does_not_fire():
    scorer = HighlightScorer(window_seconds=10.0)
    # A couple of messages with no real baseline must not read as a spike.
    scorer.note_message(9.0)
    scorer.note_message(9.5)
    assert scorer.score(now=10.0) == 1.0
    assert scorer.should_mark(now=10.0, threshold=2.0) is False


def test_highlight_scorer_evicts_old_events():
    scorer = HighlightScorer(window_seconds=5.0)
    scorer.note_message(0.0)  # will fall out of a 5s window by t=100
    scorer.note_message(1.0)
    # Far in the future: everything is evicted -> neutral, no crash.
    assert scorer.score(now=100.0) == 1.0


def test_highlight_scorer_out_of_order_timestamps_do_not_crash():
    scorer = HighlightScorer(window_seconds=10.0)
    scorer.note_message(5.0)
    scorer.note_message(3.0)  # out of order -> clamped, no exception
    scorer.note_message("bad")  # non-numeric -> ignored
    # Still produces a finite, deterministic score.
    s = scorer.score(now=5.0)
    assert isinstance(s, float)


def test_highlight_scorer_rejects_bad_window():
    with pytest.raises(ValueError):
        HighlightScorer(window_seconds=0)
    with pytest.raises(ValueError):
        HighlightScorer(window_seconds=-3.0)


def test_highlight_scorer_rejects_bad_threshold():
    scorer = HighlightScorer(window_seconds=10.0)
    with pytest.raises(ValueError):
        scorer.should_mark(now=1.0, threshold=0)


# --------------------------------------------------------------------------- #
# Anticheat: no-capture import scan (BR-P1)
# --------------------------------------------------------------------------- #
_FORBIDDEN_IMPORTS = {
    "mss", "pyautogui", "pywinauto", "pynput", "cv2", "PIL", "Pillow",
    "d3dshot", "dxcam", "win32gui", "win32api", "win32con", "ctypes",
    "requests", "aiohttp", "websockets", "websocket", "torch", "transformers",
    "obswebsocket", "obsws_python", "sounddevice", "pyaudio",
}


def test_no_capture_imports():
    """The content-ops module imports NO screen/video-capture or desktop-automation
    library, and no banned network/ML stack — the SCAN-PROOF anticheat guarantee."""
    src = inspect.getsource(content_ops)
    tree = ast.parse(src)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])

    leaked = imported & _FORBIDDEN_IMPORTS
    assert not leaked, f"content_ops imported forbidden capture/automation libs: {leaked}"
    # Positive assertion: only the expected stdlib names are imported.
    assert imported <= {"__future__", "logging", "collections", "typing"}, (
        f"unexpected imports in content_ops: {imported}"
    )
