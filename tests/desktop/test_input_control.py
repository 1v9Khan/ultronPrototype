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
