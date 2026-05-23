"""Tests for the visual crosshair click preview (catalog T16)."""

from __future__ import annotations

import io
import time
from typing import Callable, Optional

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from ultron.desktop.click_preview import (
    AUTO_PASS_RADIUS_PX,
    ConfirmationHistory,
    ConfirmedClick,
    DEFAULT_CROSSHAIR_COLOR,
    DEFAULT_CROSSHAIR_SIZE,
    DEFAULT_CROSSHAIR_THICKNESS,
    DEFAULT_HISTORY_DEPTH,
    PreviewDecision,
    PreviewResult,
    draw_crosshair_on_image,
    preview_click,
)


def _png(w: int = 200, h: int = 200, color: tuple = (40, 40, 40)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (w, h), color).save(out, format="PNG")
    return out.getvalue()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_constants_sane():
    assert DEFAULT_CROSSHAIR_SIZE > 0
    assert DEFAULT_CROSSHAIR_THICKNESS > 0
    assert AUTO_PASS_RADIUS_PX > 0
    assert DEFAULT_HISTORY_DEPTH >= 1


def test_default_color_is_red():
    assert DEFAULT_CROSSHAIR_COLOR == (255, 0, 0)


# ---------------------------------------------------------------------------
# draw_crosshair_on_image
# ---------------------------------------------------------------------------


def test_draw_crosshair_rejects_empty_bytes():
    with pytest.raises(ValueError):
        draw_crosshair_on_image(b"", x=10, y=10)


def test_draw_crosshair_returns_png_bytes():
    out = draw_crosshair_on_image(_png(), x=100, y=100)
    assert isinstance(out, bytes)
    assert len(out) > 0
    # Round-trip through PIL to confirm it's a valid PNG.
    with Image.open(io.BytesIO(out)) as im:
        assert im.size == (200, 200)


def test_draw_crosshair_marks_center_red():
    out = draw_crosshair_on_image(
        _png(color=(0, 0, 0)),  # black background
        x=100,
        y=100,
    )
    with Image.open(io.BytesIO(out)) as im:
        # Centre pixel should be red.
        r, g, b = im.getpixel((100, 100))
        assert (r, g, b) == (255, 0, 0)


def test_draw_crosshair_does_not_alter_far_pixels():
    bg = (50, 50, 50)
    out = draw_crosshair_on_image(_png(color=bg), x=100, y=100)
    with Image.open(io.BytesIO(out)) as im:
        far = im.getpixel((5, 5))
        assert tuple(far) == bg


def test_draw_crosshair_custom_size_and_color():
    out = draw_crosshair_on_image(
        _png(color=(0, 0, 0)),
        x=100,
        y=100,
        size=40,
        thickness=5,
        color=(0, 255, 0),
    )
    with Image.open(io.BytesIO(out)) as im:
        r, g, b = im.getpixel((100, 100))
        assert (r, g, b) == (0, 255, 0)


# ---------------------------------------------------------------------------
# ConfirmationHistory
# ---------------------------------------------------------------------------


def test_history_invalid_max_entries_raises():
    with pytest.raises(ValueError):
        ConfirmationHistory(max_entries=0)


def test_history_record_and_near_hit():
    h = ConfirmationHistory()
    h.record(100, 100, description="OK button")
    near = h.near(105, 105, radius=20)
    assert near is not None
    assert near.x == 100
    assert near.y == 100


def test_history_near_miss():
    h = ConfirmationHistory()
    h.record(100, 100)
    assert h.near(300, 300, radius=20) is None


def test_history_returns_most_recent():
    h = ConfirmationHistory()
    h.record(100, 100, description="first")
    h.record(110, 110, description="second")
    near = h.near(105, 105, radius=50)
    assert near is not None
    assert near.description == "second"  # most-recent wins


def test_history_caps_at_max_entries():
    h = ConfirmationHistory(max_entries=3)
    for i in range(10):
        h.record(i, i)
    assert len(h) == 3
    # Oldest dropped -- (0, 0) shouldn't be findable.
    assert h.near(0, 0, radius=1) is None
    # Most-recent retained.
    assert h.near(9, 9, radius=1) is not None


def test_history_clear():
    h = ConfirmationHistory()
    h.record(100, 100)
    h.clear()
    assert len(h) == 0
    assert h.near(100, 100, radius=50) is None


def test_history_negative_radius_returns_none():
    h = ConfirmationHistory()
    h.record(100, 100)
    assert h.near(100, 100, radius=-1) is None


def test_confirmed_click_is_frozen():
    c = ConfirmedClick(x=1, y=2, confirmed_at=time.time())
    with pytest.raises(Exception):
        c.x = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# preview_click -- auto-pass tier
# ---------------------------------------------------------------------------


def _allow_capture() -> bytes:
    return _png()


def _vlm_yes(_img: bytes, _prompt: str) -> str:
    return "Yes, the crosshair is on the OK button."


def _vlm_no(_img: bytes, _prompt: str) -> str:
    return "No -- the crosshair is on a banner ad, not the OK button."


def test_auto_pass_within_radius():
    h = ConfirmationHistory()
    h.record(100, 100)
    result = preview_click(
        x=110, y=110,
        capture_screen=_allow_capture,
        vlm_describe=_vlm_yes,
        history=h,
    )
    assert result.decision == PreviewDecision.AUTO_PASS


def test_outside_radius_triggers_vlm():
    h = ConfirmationHistory()
    h.record(100, 100)
    result = preview_click(
        x=300, y=300,
        capture_screen=_allow_capture,
        vlm_describe=_vlm_yes,
        history=h,
    )
    assert result.decision == PreviewDecision.ALLOW


# ---------------------------------------------------------------------------
# preview_click -- VLM round-trip
# ---------------------------------------------------------------------------


def test_vlm_yes_returns_allow_and_records_history():
    h = ConfirmationHistory()
    result = preview_click(
        x=100, y=100,
        capture_screen=_allow_capture,
        vlm_describe=_vlm_yes,
        history=h,
        intent_description="click the OK button",
    )
    assert result.decision == PreviewDecision.ALLOW
    assert "OK" in result.vlm_response or "yes" in result.vlm_response.lower()
    # History should now have this confirmation.
    assert len(h) == 1


def test_vlm_no_returns_block_and_does_not_record():
    h = ConfirmationHistory()
    result = preview_click(
        x=100, y=100,
        capture_screen=_allow_capture,
        vlm_describe=_vlm_no,
        history=h,
        intent_description="click the OK button",
    )
    assert result.decision == PreviewDecision.BLOCK
    assert len(h) == 0


def test_no_vlm_returns_degraded():
    h = ConfirmationHistory()
    result = preview_click(
        x=100, y=100,
        capture_screen=_allow_capture,
        vlm_describe=None,
        history=h,
    )
    assert result.decision == PreviewDecision.DEGRADED
    assert "vlm_describe" in result.reason.lower()


def test_vlm_exception_returns_degraded():
    h = ConfirmationHistory()

    def boom(_a: bytes, _b: str) -> str:
        raise RuntimeError("vlm down")

    result = preview_click(
        x=100, y=100,
        capture_screen=_allow_capture,
        vlm_describe=boom,
        history=h,
    )
    assert result.decision == PreviewDecision.DEGRADED


def test_capture_exception_returns_degraded():
    h = ConfirmationHistory()

    def boom() -> bytes:
        raise OSError("screen unavailable")

    result = preview_click(
        x=100, y=100,
        capture_screen=boom,
        vlm_describe=_vlm_yes,
        history=h,
    )
    assert result.decision == PreviewDecision.DEGRADED


def test_capture_empty_returns_degraded():
    h = ConfirmationHistory()
    result = preview_click(
        x=100, y=100,
        capture_screen=lambda: b"",
        vlm_describe=_vlm_yes,
        history=h,
    )
    assert result.decision == PreviewDecision.DEGRADED


def test_custom_confirmation_keyword():
    h = ConfirmationHistory()

    def vlm(_a: bytes, _b: str) -> str:
        return "AFFIRMATIVE -- target confirmed"

    result = preview_click(
        x=100, y=100,
        capture_screen=_allow_capture,
        vlm_describe=vlm,
        history=h,
        require_confirmation_keyword="affirmative",
    )
    assert result.decision == PreviewDecision.ALLOW
