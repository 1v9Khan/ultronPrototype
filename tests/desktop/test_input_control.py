"""Tests for ultron.desktop.input_control."""

from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock

import pytest

from ultron.desktop.input_control import (
    InputController,
    InputControlResult,
    _foreground_is_security_window,
    get_input_controller,
    set_input_controller,
)


# ---------------------------------------------------------------------------
# Result dataclass + singleton
# ---------------------------------------------------------------------------


def test_input_control_result_is_frozen():
    r = InputControlResult(success=True, action="click")
    with pytest.raises(Exception):
        r.success = False


def test_get_input_controller_singleton_caches():
    set_input_controller(None)
    try:
        a = get_input_controller()
        b = get_input_controller()
        assert a is b
    finally:
        set_input_controller(None)


def test_set_input_controller_swaps():
    set_input_controller(None)
    custom = InputController()
    try:
        set_input_controller(custom)
        assert get_input_controller() is custom
    finally:
        set_input_controller(None)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def _no_block_controller():
    """Build a controller whose security check + validator always allow."""
    return InputController(
        max_actions_per_second=999.0,
        enforce_security_window_block=False,
    )


def test_click_rejects_unknown_button(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.input_control.pyautogui",
        MagicMock(),
    )
    c = _no_block_controller()
    r = c.click(button="weird")
    assert r.success is False
    assert "unknown button" in (r.error or "")


def test_click_rejects_out_of_range_clicks(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.input_control.pyautogui",
        MagicMock(),
    )
    c = _no_block_controller()
    assert c.click(clicks=0).success is False
    assert c.click(clicks=10).success is False


def test_type_text_rejects_non_string(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.input_control.pyautogui",
        MagicMock(),
    )
    c = _no_block_controller()
    r = c.type_text(12345)  # type: ignore[arg-type]
    assert r.success is False


def test_type_text_empty_string_is_noop(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    c = _no_block_controller()
    r = c.type_text("")
    assert r.success is True
    pa.write.assert_not_called()


def test_press_key_rejects_empty(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.input_control.pyautogui",
        MagicMock(),
    )
    c = _no_block_controller()
    r = c.press_key("")
    assert r.success is False
    r = c.press_key("   ")
    assert r.success is False


def test_press_hotkey_rejects_empty(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.input_control.pyautogui",
        MagicMock(),
    )
    c = _no_block_controller()
    r = c.press_hotkey()
    assert r.success is False


# ---------------------------------------------------------------------------
# Rate limit
# ---------------------------------------------------------------------------


def test_rate_limit_blocks_after_threshold(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    # Disable validator + security-window check so rate-limit is the only gate.
    c = InputController(
        max_actions_per_second=3.0,
        enforce_security_window_block=False,
    )
    # First 3 calls succeed, fourth fails.
    r1 = c.click(0, 0)
    r2 = c.click(0, 0)
    r3 = c.click(0, 0)
    r4 = c.click(0, 0)
    assert r1.success is True
    assert r2.success is True
    assert r3.success is True
    assert r4.success is False
    assert "rate limit" in (r4.error or "")


def test_rate_limit_resets_after_window(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    c = InputController(
        max_actions_per_second=2.0,
        enforce_security_window_block=False,
    )
    c.click(0, 0)
    c.click(0, 0)
    blocked = c.click(0, 0)
    assert blocked.success is False
    # Wait for the 1s window to roll over.
    time.sleep(1.05)
    r = c.click(0, 0)
    assert r.success is True


# ---------------------------------------------------------------------------
# Security-window gate
# ---------------------------------------------------------------------------


def test_security_foreground_blocks_input(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    monkeypatch.setattr(
        "ultron.desktop.input_control._foreground_is_security_window",
        lambda: True,
    )
    c = InputController(max_actions_per_second=999.0)
    r = c.click(100, 100)
    assert r.success is False
    assert "security window" in (r.error or "")
    pa.click.assert_not_called()


def test_security_check_false_when_no_foreground(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.windows.get_foreground_window", lambda: None,
    )
    assert _foreground_is_security_window() is False


def test_security_check_recognises_uac_class(monkeypatch):
    from ultron.desktop.windows import WindowInfo
    fake_fg = WindowInfo(
        hwnd=1, title="User Account Control", class_name="ConsentUI",
        process_name="consent.exe", pid=0,
        rect=(0, 0, 100, 100), monitor_index=0,
        is_minimized=False, is_foreground=True,
    )
    monkeypatch.setattr(
        "ultron.desktop.windows.get_foreground_window", lambda: fake_fg,
    )
    assert _foreground_is_security_window() is True


def test_security_check_corewindow_only_when_title_matches(monkeypatch):
    """CoreWindow class is too broad; require the title to match a security keyword."""
    from ultron.desktop.windows import WindowInfo

    fake_fg = WindowInfo(
        hwnd=1, title="Calculator", class_name="Windows.UI.Core.CoreWindow",
        process_name="CalculatorApp.exe", pid=0,
        rect=(0, 0, 100, 100), monitor_index=0,
        is_minimized=False, is_foreground=True,
    )
    monkeypatch.setattr(
        "ultron.desktop.windows.get_foreground_window", lambda: fake_fg,
    )
    assert _foreground_is_security_window() is False

    # Now the same class with a security-relevant title.
    security_fg = WindowInfo(
        hwnd=2, title="Windows Security: sign in", class_name="Windows.UI.Core.CoreWindow",
        process_name="systemsettings.exe", pid=0,
        rect=(0, 0, 100, 100), monitor_index=0,
        is_minimized=False, is_foreground=True,
    )
    monkeypatch.setattr(
        "ultron.desktop.windows.get_foreground_window", lambda: security_fg,
    )
    assert _foreground_is_security_window() is True


# ---------------------------------------------------------------------------
# Validator hook
# ---------------------------------------------------------------------------


def test_validator_block_short_circuits_click(monkeypatch):
    from ultron.safety.validator import ValidatorVerdict, Verdict

    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    monkeypatch.setattr(
        "ultron.desktop.input_control._foreground_is_security_window",
        lambda: False,
    )
    monkeypatch.setattr(
        "ultron.desktop.input_control._validate_input_action",
        lambda **kw: ValidatorVerdict(
            verdict=Verdict.BLOCK_HARD, reason="policy block",
            triggered_rule_id="test", user_message="refused",
        ),
    )
    c = InputController(max_actions_per_second=999.0)
    r = c.click(100, 100)
    assert r.success is False
    assert "safety" in (r.error or "")
    pa.click.assert_not_called()


def test_validator_allow_lets_click_through(monkeypatch):
    from ultron.safety.validator import ValidatorVerdict, Verdict

    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    monkeypatch.setattr(
        "ultron.desktop.input_control._foreground_is_security_window",
        lambda: False,
    )
    monkeypatch.setattr(
        "ultron.desktop.input_control._validate_input_action",
        lambda **kw: ValidatorVerdict(verdict=Verdict.ALLOW, reason="ok"),
    )
    c = InputController(max_actions_per_second=999.0)
    r = c.click(100, 100)
    assert r.success is True
    pa.click.assert_called_once()


def test_pyautogui_exception_returns_failure(monkeypatch):
    from ultron.safety.validator import ValidatorVerdict, Verdict
    pa = MagicMock()
    pa.click.side_effect = RuntimeError("simulated input failure")
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    monkeypatch.setattr(
        "ultron.desktop.input_control._foreground_is_security_window",
        lambda: False,
    )
    monkeypatch.setattr(
        "ultron.desktop.input_control._validate_input_action",
        lambda **kw: ValidatorVerdict(verdict=Verdict.ALLOW, reason="ok"),
    )
    c = InputController(max_actions_per_second=999.0)
    r = c.click(100, 100)
    assert r.success is False
    assert "simulated input failure" in (r.error or "")


# ---------------------------------------------------------------------------
# Move / scroll / hotkey happy paths
# ---------------------------------------------------------------------------


def _set_allow_all(monkeypatch):
    from ultron.safety.validator import ValidatorVerdict, Verdict
    monkeypatch.setattr(
        "ultron.desktop.input_control._foreground_is_security_window",
        lambda: False,
    )
    monkeypatch.setattr(
        "ultron.desktop.input_control._validate_input_action",
        lambda **kw: ValidatorVerdict(verdict=Verdict.ALLOW, reason="ok"),
    )


def test_move_mouse_invokes_pyautogui(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    _set_allow_all(monkeypatch)
    c = InputController(max_actions_per_second=999.0)
    r = c.move_mouse(500, 600, duration_s=0.05)
    assert r.success is True
    pa.moveTo.assert_called_once_with(500, 600, duration=0.05)


def test_press_hotkey_invokes_pyautogui(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    _set_allow_all(monkeypatch)
    c = InputController(max_actions_per_second=999.0)
    r = c.press_hotkey("ctrl", "s")
    assert r.success is True
    pa.hotkey.assert_called_once_with("ctrl", "s")


def test_scroll_invokes_pyautogui(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    _set_allow_all(monkeypatch)
    c = InputController(max_actions_per_second=999.0)
    r = c.scroll(120, x=500, y=400)
    assert r.success is True
    pa.scroll.assert_called_once_with(120, x=500, y=400)


# ---------------------------------------------------------------------------
# Click preview gate (SWE-Agent T16)
# ---------------------------------------------------------------------------


def _build_test_png(*, width: int = 100, height: int = 100) -> bytes:
    """Build a tiny PNG so the preview's PIL pipeline has real bytes."""
    from io import BytesIO

    from PIL import Image  # type: ignore[import-not-found]

    img = Image.new("RGB", (width, height), (0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_click_preview_default_disabled_skips_vlm(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    _set_allow_all(monkeypatch)
    # No capture / vlm callables set -> preview path inert.
    c = InputController(max_actions_per_second=999.0)
    r = c.click(100, 200)
    assert r.success is True
    pa.click.assert_called_once()


def test_click_preview_blocks_when_vlm_disagrees(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    _set_allow_all(monkeypatch)
    png = _build_test_png()
    captures: list[None] = []

    def capture():
        captures.append(None)
        return png

    def vlm(_image, _prompt):
        return "actually that is a Submit button, not what you wanted"

    c = InputController(
        max_actions_per_second=999.0,
        click_preview_enabled=True,
        click_preview_capture_screen=capture,
        click_preview_vlm_describe=vlm,
    )
    r = c.click(50, 50, user_text="open the Files menu")
    assert r.success is False
    assert "click_preview" in (r.error or "")
    assert pa.click.call_count == 0
    assert len(captures) == 1


def test_click_preview_allows_when_vlm_confirms(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    _set_allow_all(monkeypatch)
    png = _build_test_png()
    c = InputController(
        max_actions_per_second=999.0,
        click_preview_enabled=True,
        click_preview_capture_screen=lambda: png,
        click_preview_vlm_describe=lambda _img, _p: "yes that's the Files menu",
    )
    r = c.click(50, 50, user_text="open the Files menu")
    assert r.success is True
    pa.click.assert_called_once()


def test_click_preview_auto_pass_skips_second_vlm_round(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    _set_allow_all(monkeypatch)
    png = _build_test_png()
    vlm_calls = []

    def vlm(_img, _p):
        vlm_calls.append(None)
        return "yes"

    c = InputController(
        max_actions_per_second=999.0,
        click_preview_enabled=True,
        click_preview_capture_screen=lambda: png,
        click_preview_vlm_describe=vlm,
        click_preview_auto_pass_radius_px=100,
    )
    c.click(50, 50, user_text="open the Files menu")
    c.click(60, 60, user_text="click near the same spot")
    assert len(vlm_calls) == 1  # second click is auto-pass


def test_click_preview_degraded_default_allows(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    _set_allow_all(monkeypatch)
    # vlm_describe=None forces DEGRADED -> default policy is allow.
    png = _build_test_png()
    c = InputController(
        max_actions_per_second=999.0,
        click_preview_enabled=True,
        click_preview_capture_screen=lambda: png,
        click_preview_vlm_describe=None,
    )
    r = c.click(50, 50)
    assert r.success is True
    pa.click.assert_called_once()


def test_click_preview_degraded_blocks_when_strict(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    _set_allow_all(monkeypatch)
    png = _build_test_png()
    c = InputController(
        max_actions_per_second=999.0,
        click_preview_enabled=True,
        click_preview_capture_screen=lambda: png,
        click_preview_vlm_describe=None,
        click_preview_block_on_degraded=True,
    )
    r = c.click(50, 50)
    assert r.success is False
    assert "DEGRADED" in (r.error or "")
    assert pa.click.call_count == 0


def test_click_preview_skips_when_no_coordinates(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    _set_allow_all(monkeypatch)
    captures = []
    c = InputController(
        max_actions_per_second=999.0,
        click_preview_enabled=True,
        click_preview_capture_screen=lambda: captures.append(None) or b"",
        click_preview_vlm_describe=lambda _i, _p: "yes",
    )
    r = c.click()  # no x/y -> click at current cursor; preview not invoked
    assert r.success is True
    assert pa.click.call_count == 1
    assert captures == []


# ---------------------------------------------------------------------------
# Catalog 08 T8: drag_to
# ---------------------------------------------------------------------------


def test_drag_to_rejects_unknown_button(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.input_control.pyautogui",
        MagicMock(),
    )
    c = _no_block_controller()
    r = c.drag_to(10, 20, 30, 40, button="weird")
    assert r.success is False
    assert "unknown button" in (r.error or "")


def test_drag_to_rejects_negative_duration(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.input_control.pyautogui",
        MagicMock(),
    )
    c = _no_block_controller()
    r = c.drag_to(10, 20, 30, 40, duration_s=-0.1)
    assert r.success is False
    assert "non-negative" in (r.error or "")


def test_drag_to_returns_success_when_pyautogui_succeeds(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    c = _no_block_controller()
    r = c.drag_to(100, 200, 300, 400, duration_s=0.05)
    assert r.success is True
    assert r.action == "drag_to"
    pa.moveTo.assert_called_once_with(100, 200)
    pa.dragTo.assert_called_once()
    # dragTo args: (x, y, duration=..., button=...)
    args, kwargs = pa.dragTo.call_args
    assert args == (300, 400)
    assert kwargs["duration"] == 0.05
    assert kwargs["button"] == "left"


def test_drag_to_supports_right_button(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    c = _no_block_controller()
    r = c.drag_to(10, 20, 30, 40, button="right")
    assert r.success is True
    _, kwargs = pa.dragTo.call_args
    assert kwargs["button"] == "right"


def test_drag_to_rate_limit_enforced(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    c = InputController(
        max_actions_per_second=1.0,
        enforce_security_window_block=False,
    )
    # First call passes; second within the 1-second window must fail.
    r1 = c.drag_to(0, 0, 10, 10, duration_s=0.0)
    r2 = c.drag_to(0, 0, 10, 10, duration_s=0.0)
    assert r1.success is True
    assert r2.success is False
    assert "rate limit" in (r2.error or "")


def test_drag_to_blocked_by_security_window(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.input_control.pyautogui",
        MagicMock(),
    )
    monkeypatch.setattr(
        "ultron.desktop.input_control._foreground_is_security_window",
        lambda: True,
    )
    c = InputController(max_actions_per_second=999.0)
    r = c.drag_to(0, 0, 10, 10)
    assert r.success is False
    assert "security window" in (r.error or "")


def test_drag_to_blocked_by_safety_validator(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    from ultron.safety.validator import ValidatorVerdict, Verdict
    blocked = ValidatorVerdict(
        verdict=Verdict.BLOCK_HARD, reason="cap-3 hardstop",
        triggered_rule_id="test", user_message="refused",
    )
    monkeypatch.setattr(
        "ultron.safety.validator.get_validator",
        lambda: type("V", (), {"check": lambda self, ctx: blocked})(),
    )
    c = _no_block_controller()
    r = c.drag_to(0, 0, 10, 10, user_text="drag the icon")
    assert r.success is False
    assert "safety" in (r.error or "")
    assert pa.dragTo.call_count == 0


def test_drag_to_returns_failure_when_pyautogui_raises(monkeypatch):
    pa = MagicMock()
    pa.dragTo.side_effect = RuntimeError("pyautogui blew up")
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    c = _no_block_controller()
    r = c.drag_to(0, 0, 10, 10)
    assert r.success is False
    assert "blew up" in (r.error or "")


def test_drag_to_click_preview_block_short_circuits(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    _set_allow_all(monkeypatch)
    png = _build_test_png()

    # VLM response without the "yes" confirmation keyword should produce
    # a BLOCK decision via the existing preview_click logic.
    c = InputController(
        max_actions_per_second=999.0,
        enforce_security_window_block=False,
        click_preview_enabled=True,
        click_preview_capture_screen=lambda: png,
        click_preview_vlm_describe=lambda _i, _p: "no, this is the wrong target",
        click_preview_require_confirmation_keyword="yes",
    )
    r = c.drag_to(50, 60, 100, 120)
    assert r.success is False
    assert r.action == "drag_to"
    assert "click_preview" in (r.error or "") or "BLOCK" in (r.error or "")
    # pyautogui should NOT have fired.
    assert pa.dragTo.call_count == 0


def test_drag_to_works_at_zero_duration(monkeypatch):
    """duration_s=0 is valid (snap drag, no animation)."""
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    c = _no_block_controller()
    r = c.drag_to(0, 0, 50, 50, duration_s=0.0)
    assert r.success is True
    _, kwargs = pa.dragTo.call_args
    assert kwargs["duration"] == 0.0


# ---------------------------------------------------------------------------
# Catalog 09 T1 -- scroll direction (vertical default + horizontal axis)
# ---------------------------------------------------------------------------


def test_scroll_default_direction_is_vertical_back_compat(monkeypatch):
    """Existing scroll callers without ``direction`` should keep using
    pyautogui.scroll (vertical axis). Back-compat invariant."""
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    _set_allow_all(monkeypatch)
    c = InputController(max_actions_per_second=999.0)
    r = c.scroll(120, x=200, y=300)
    assert r.success is True
    pa.scroll.assert_called_once_with(120, x=200, y=300)
    pa.hscroll.assert_not_called()


def test_scroll_horizontal_dispatches_to_hscroll(monkeypatch):
    """direction='horizontal' must route to pyautogui.hscroll, not scroll."""
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    _set_allow_all(monkeypatch)
    c = InputController(max_actions_per_second=999.0)
    r = c.scroll(80, direction="horizontal", x=400, y=500)
    assert r.success is True
    pa.hscroll.assert_called_once_with(80, x=400, y=500)
    pa.scroll.assert_not_called()


def test_scroll_vertical_explicit_still_uses_scroll(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    _set_allow_all(monkeypatch)
    c = InputController(max_actions_per_second=999.0)
    r = c.scroll(-60, direction="vertical")
    assert r.success is True
    pa.scroll.assert_called_once_with(-60, x=None, y=None)


def test_scroll_rejects_unknown_direction(monkeypatch):
    """Unlike the upstream plugin (silent fallthrough), ultron rejects
    invalid ``direction`` with a structured error."""
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    c = _no_block_controller()
    r = c.scroll(50, direction="diagonal")
    assert r.success is False
    assert "unknown direction" in (r.error or "")
    pa.scroll.assert_not_called()
    pa.hscroll.assert_not_called()


def test_scroll_horizontal_with_no_coords_passes_none(monkeypatch):
    """When x/y are not supplied, both are passed as None to pyautogui
    so the scroll fires at the current cursor location."""
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    _set_allow_all(monkeypatch)
    c = InputController(max_actions_per_second=999.0)
    r = c.scroll(40, direction="horizontal")
    assert r.success is True
    pa.hscroll.assert_called_once_with(40, x=None, y=None)


def test_scroll_validator_arguments_carry_direction(monkeypatch):
    """The validator must see the direction so audit log entries
    distinguish vertical vs horizontal scroll attempts."""
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    seen: dict = {}

    def _spy(**kw):
        seen.update(kw)
        from ultron.safety.validator import ValidatorVerdict, Verdict
        return ValidatorVerdict(verdict=Verdict.ALLOW, reason="ok")

    monkeypatch.setattr(
        "ultron.desktop.input_control._validate_input_action", _spy,
    )
    monkeypatch.setattr(
        "ultron.desktop.input_control._foreground_is_security_window",
        lambda: False,
    )
    c = InputController(max_actions_per_second=999.0)
    c.scroll(30, direction="horizontal")
    args = seen.get("arguments") or {}
    assert args.get("direction") == "horizontal"
    assert args.get("amount") == 30


# ---------------------------------------------------------------------------
# Catalog 09 T3 -- WPM-cadence typing
# ---------------------------------------------------------------------------


def test_type_text_wpm_overrides_interval_s(monkeypatch):
    """When ``wpm`` is set, ``interval_s`` is overridden by the standard
    5-chars-per-word formula: chars_per_second = wpm * 5 / 60;
    interval = 1.0 / chars_per_second.
    """
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    _set_allow_all(monkeypatch)
    c = InputController(max_actions_per_second=999.0)
    # 60 WPM = 300 chars/min = 5 chars/s -> interval = 0.2 s
    r = c.type_text("hello", interval_s=0.001, wpm=60)
    assert r.success is True
    _, kwargs = pa.write.call_args
    assert kwargs["interval"] == pytest.approx(0.2, abs=1e-6)


def test_type_text_wpm_120_yields_short_interval(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    _set_allow_all(monkeypatch)
    c = InputController(max_actions_per_second=999.0)
    # 120 WPM = 600 chars/min = 10 chars/s -> interval = 0.1 s
    r = c.type_text("fast", wpm=120)
    assert r.success is True
    _, kwargs = pa.write.call_args
    assert kwargs["interval"] == pytest.approx(0.1, abs=1e-6)


def test_type_text_wpm_zero_rejected(monkeypatch):
    """The upstream plugin's bare ``1.0 / ((wpm * 5) / 60)`` would raise
    ZeroDivisionError on wpm=0. Ultron returns a structured error."""
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    c = _no_block_controller()
    r = c.type_text("hi", wpm=0)
    assert r.success is False
    assert "wpm must be positive" in (r.error or "")
    pa.write.assert_not_called()


def test_type_text_wpm_negative_rejected(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    c = _no_block_controller()
    r = c.type_text("hi", wpm=-30)
    assert r.success is False
    assert "wpm must be positive" in (r.error or "")
    pa.write.assert_not_called()


def test_type_text_wpm_none_falls_back_to_interval_s(monkeypatch):
    """Back-compat: wpm=None preserves the legacy interval_s contract."""
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    _set_allow_all(monkeypatch)
    c = InputController(max_actions_per_second=999.0)
    r = c.type_text("hello", interval_s=0.025)
    assert r.success is True
    _, kwargs = pa.write.call_args
    assert kwargs["interval"] == pytest.approx(0.025, abs=1e-9)


def test_type_text_validator_arguments_carry_wpm(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    seen: dict = {}

    def _spy(**kw):
        seen.update(kw)
        from ultron.safety.validator import ValidatorVerdict, Verdict
        return ValidatorVerdict(verdict=Verdict.ALLOW, reason="ok")

    monkeypatch.setattr(
        "ultron.desktop.input_control._validate_input_action", _spy,
    )
    monkeypatch.setattr(
        "ultron.desktop.input_control._foreground_is_security_window",
        lambda: False,
    )
    c = InputController(max_actions_per_second=999.0)
    c.type_text("hi", wpm=80)
    args = seen.get("arguments") or {}
    assert args.get("wpm") == 80
    # Computed interval is reflected in audit args:
    assert args.get("interval_s") == pytest.approx(60.0 / (80.0 * 5.0))


# ---------------------------------------------------------------------------
# Catalog 09 T7 -- bezier-smooth mouse movement
# ---------------------------------------------------------------------------


def test_move_mouse_smooth_off_uses_linear(monkeypatch):
    """Default smooth=False matches the legacy back-compat call shape:
    no ``tween`` kwarg to pyautogui.moveTo."""
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    _set_allow_all(monkeypatch)
    c = InputController(max_actions_per_second=999.0)
    r = c.move_mouse(100, 200, duration_s=0.2)
    assert r.success is True
    _, kwargs = pa.moveTo.call_args
    assert kwargs["duration"] == pytest.approx(0.2, abs=1e-9)
    assert "tween" not in kwargs


def test_move_mouse_smooth_with_duration_uses_easeInOutQuad(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    _set_allow_all(monkeypatch)
    c = InputController(max_actions_per_second=999.0)
    r = c.move_mouse(300, 400, duration_s=0.5, smooth=True)
    assert r.success is True
    _, kwargs = pa.moveTo.call_args
    assert kwargs["duration"] == pytest.approx(0.5, abs=1e-9)
    assert kwargs["tween"] is pa.easeInOutQuad


def test_move_mouse_smooth_true_but_zero_duration_no_tween(monkeypatch):
    """smooth=True with duration_s=0 yields an instant non-tweened
    move (matches the upstream plugin's branching)."""
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    _set_allow_all(monkeypatch)
    c = InputController(max_actions_per_second=999.0)
    r = c.move_mouse(50, 50, duration_s=0.0, smooth=True)
    assert r.success is True
    _, kwargs = pa.moveTo.call_args
    assert kwargs["duration"] == 0.0
    assert "tween" not in kwargs


def test_move_mouse_validator_arguments_carry_smooth(monkeypatch):
    pa = MagicMock()
    monkeypatch.setattr("ultron.desktop.input_control.pyautogui", pa)
    seen: dict = {}

    def _spy(**kw):
        seen.update(kw)
        from ultron.safety.validator import ValidatorVerdict, Verdict
        return ValidatorVerdict(verdict=Verdict.ALLOW, reason="ok")

    monkeypatch.setattr(
        "ultron.desktop.input_control._validate_input_action", _spy,
    )
    monkeypatch.setattr(
        "ultron.desktop.input_control._foreground_is_security_window",
        lambda: False,
    )
    c = InputController(max_actions_per_second=999.0)
    c.move_mouse(10, 20, duration_s=0.4, smooth=True)
    args = seen.get("arguments") or {}
    assert args.get("smooth") is True
    assert args.get("duration_s") == pytest.approx(0.4)
