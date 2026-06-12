"""Tests for ultron.desktop.voice handlers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ultron.desktop.launcher import LaunchResult
from ultron.desktop.monitors import Monitor
from ultron.desktop.screen_context import ScreenContextSnapshot
from ultron.desktop.voice import (
    AppLaunchVoiceResult,
    ScreenContextVoiceResult,
    handle_app_launch,
    handle_screen_context_query,
)
from ultron.openclaw_routing.intents import (
    AppLaunchIntent,
    ScreenContextIntent,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _mon(idx=0) -> Monitor:
    return Monitor(
        index=idx, name=f"D{idx}",
        x=idx * 1920, y=0, width=1920, height=1080,
        work_x=idx * 1920, work_y=0,
        work_width=1920, work_height=1040,
        is_primary=(idx == 0),
    )


def _patch_monitors(monkeypatch, mons):
    monkeypatch.setattr(
        "ultron.desktop.monitors.enumerate_monitors", lambda: mons,
    )


def _patch_launcher(monkeypatch, *, result):
    fake = MagicMock()
    fake.launch_app.return_value = result
    fake.launch_chrome.return_value = result
    monkeypatch.setattr(
        "ultron.desktop.launcher.get_app_launcher", lambda: fake,
    )
    return fake


def _ok_launch_result(*, app="chrome", monitor_index=None, hwnd=42):
    return LaunchResult(
        success=True, app_name=app,
        exe_path=Path("C:/ghost.exe"),
        pid=100, hwnd=hwnd, monitor_index=monitor_index,
    )


def _bad_launch_result(error="something broke"):
    return LaunchResult(success=False, app_name="chrome", error=error)


# ---------------------------------------------------------------------------
# handle_app_launch
# ---------------------------------------------------------------------------


def test_handle_app_launch_empty_app_name():
    intent = AppLaunchIntent(app_name="")
    result = handle_app_launch(intent)
    assert result.success is False
    assert "didn't catch" in result.voice_message.lower()


def test_handle_app_launch_simple_app(monkeypatch):
    _patch_monitors(monkeypatch, [_mon()])
    fake = _patch_launcher(monkeypatch, result=_ok_launch_result(app="cursor"))
    intent = AppLaunchIntent(app_name="cursor")
    result = handle_app_launch(intent)
    assert result.success is True
    assert "Opening cursor" in result.voice_message
    fake.launch_app.assert_called_once()
    fake.launch_chrome.assert_not_called()


def test_handle_app_launch_chrome_with_url(monkeypatch):
    _patch_monitors(monkeypatch, [_mon()])
    fake = _patch_launcher(
        monkeypatch,
        result=_ok_launch_result(app="chrome", monitor_index=1),
    )
    intent = AppLaunchIntent(
        app_name="chrome", url="https://youtube.com",
        monitor_index=1,
    )
    result = handle_app_launch(intent)
    assert result.success is True
    fake.launch_chrome.assert_called_once()
    fake.launch_app.assert_not_called()
    kwargs = fake.launch_chrome.call_args.kwargs
    assert kwargs["url"] == "https://youtube.com"


def test_handle_app_launch_resolves_monitor_index(monkeypatch):
    _patch_monitors(monkeypatch, [_mon(0), _mon(1), _mon(2)])
    fake = _patch_launcher(
        monkeypatch,
        result=_ok_launch_result(app="cursor", monitor_index=1),
    )
    intent = AppLaunchIntent(app_name="cursor", monitor_index=1)
    handle_app_launch(intent)
    kwargs = fake.launch_app.call_args.kwargs
    assert kwargs["monitor"] is not None
    assert kwargs["monitor"].index == 1


def test_handle_app_launch_resolves_directional_monitor(monkeypatch):
    """When monitor_query is 'left', resolve via find_monitor."""
    _patch_monitors(monkeypatch, [_mon(0), _mon(1)])
    fake = _patch_launcher(
        monkeypatch,
        result=_ok_launch_result(app="cursor", monitor_index=0),
    )
    intent = AppLaunchIntent(
        app_name="cursor", monitor_query="left",
    )
    handle_app_launch(intent)
    kwargs = fake.launch_app.call_args.kwargs
    # find_monitor("left") returns the leftmost = idx 0 (D0 at x=0)
    assert kwargs["monitor"] is not None


def test_handle_app_launch_out_of_range_monitor_falls_back(monkeypatch):
    """Out-of-range monitor index should not crash; monitor=None passed."""
    _patch_monitors(monkeypatch, [_mon()])
    fake = _patch_launcher(monkeypatch, result=_ok_launch_result())
    intent = AppLaunchIntent(app_name="cursor", monitor_index=99)
    result = handle_app_launch(intent)
    assert result.success is True  # launcher still called
    kwargs = fake.launch_app.call_args.kwargs
    assert kwargs["monitor"] is None


def test_handle_app_launch_launcher_failure(monkeypatch):
    _patch_monitors(monkeypatch, [_mon()])
    _patch_launcher(
        monkeypatch,
        result=_bad_launch_result("Chrome not installed"),
    )
    intent = AppLaunchIntent(app_name="chrome", url="https://x.com")
    result = handle_app_launch(intent)
    assert result.success is False
    assert "couldn't open" in result.voice_message.lower()
    assert "Chrome not installed" in result.voice_message


def test_handle_app_launch_window_timeout_voice_is_honest(monkeypatch):
    # 2026-06-12 honesty fix: pre-fix the timeout path spoke "Opening
    # that on monitor 2." while no window ever appeared.
    _patch_monitors(monkeypatch, [_mon(0), _mon(1)])
    _patch_launcher(
        monkeypatch,
        result=LaunchResult(
            success=True, app_name="chrome",
            exe_path=Path("C:/ghost.exe"), pid=1, hwnd=None,
            monitor_index=None,
            error="window did not appear within timeout",
            window_appeared=False,
        ),
    )
    intent = AppLaunchIntent(
        app_name="chrome", url="https://x.com", monitor_index=1,
    )
    result = handle_app_launch(intent)
    assert result.success is True  # the spawn itself succeeded
    assert result.window_appeared is False
    assert "didn't appear" in result.voice_message
    assert "Opening" not in result.voice_message
    assert "monitor 2" in result.voice_message


def test_handle_app_launch_window_timeout_no_monitor_phrase_lie(monkeypatch):
    # Out-of-range monitor index resolves to None -> the honest line
    # must not mention any monitor at all.
    _patch_monitors(monkeypatch, [_mon()])
    _patch_launcher(
        monkeypatch,
        result=LaunchResult(
            success=True, app_name="cursor",
            exe_path=Path("C:/ghost.exe"), pid=1, hwnd=None,
            monitor_index=None,
            error="window did not appear within timeout",
            window_appeared=False,
        ),
    )
    intent = AppLaunchIntent(app_name="cursor", monitor_index=99)
    result = handle_app_launch(intent)
    assert result.success is True
    assert result.window_appeared is False
    assert "didn't appear" in result.voice_message
    assert "monitor" not in result.voice_message


def test_handle_app_launch_threads_window_appeared_true(monkeypatch):
    _patch_monitors(monkeypatch, [_mon()])
    _patch_launcher(
        monkeypatch,
        result=LaunchResult(
            success=True, app_name="cursor",
            exe_path=Path("C:/ghost.exe"), pid=1, hwnd=42,
            monitor_index=0, window_appeared=True,
        ),
    )
    intent = AppLaunchIntent(app_name="cursor")
    result = handle_app_launch(intent)
    assert result.success is True
    assert result.window_appeared is True
    assert "Opening" in result.voice_message


def test_handle_app_launch_voice_message_mentions_monitor(monkeypatch):
    _patch_monitors(monkeypatch, [_mon(0), _mon(1)])
    _patch_launcher(
        monkeypatch,
        result=_ok_launch_result(app="chrome", monitor_index=1),
    )
    intent = AppLaunchIntent(
        app_name="chrome", url="https://youtube.com",
        monitor_index=1,
    )
    result = handle_app_launch(intent)
    assert "monitor 2" in result.voice_message  # 1-indexed in narration


# ---------------------------------------------------------------------------
# handle_screen_context_query
# ---------------------------------------------------------------------------


def test_handle_screen_context_query_returns_injection(monkeypatch):
    snap = ScreenContextSnapshot(
        timestamp=0.0,
        monitors=(_mon(),),
        foreground=None,
        windows=(),
        ui_text=("some text",),
        screenshot=None,
        vlm_description=None,
        elapsed_ms=10.0,
    )
    monkeypatch.setattr(
        "ultron.desktop.screen_context.build_screen_context",
        lambda **kw: snap,
    )
    intent = ScreenContextIntent(question="what's this", include_vlm=False)
    result = handle_screen_context_query(intent)
    assert result.success is True
    assert "Visual context" in result.injection_text
    assert result.used_vlm is False
    assert result.elapsed_ms > 0


def test_handle_screen_context_query_with_vlm(monkeypatch):
    snap = ScreenContextSnapshot(
        timestamp=0.0,
        monitors=(_mon(),),
        foreground=None,
        windows=(),
        ui_text=(),
        screenshot=None,
        vlm_description="A code editor.",
        elapsed_ms=15.0,
    )
    captured_kwargs = {}

    def _build(**kw):
        captured_kwargs.update(kw)
        return snap

    monkeypatch.setattr(
        "ultron.desktop.screen_context.build_screen_context", _build,
    )
    intent = ScreenContextIntent(include_vlm=True)
    result = handle_screen_context_query(intent)
    assert result.success is True
    assert result.used_vlm is True
    assert "A code editor." in result.injection_text
    assert captured_kwargs.get("include_vlm") is True


def test_handle_screen_context_query_snapshot_failure(monkeypatch):
    def boom(**kw):
        raise RuntimeError("simulated snapshot failure")

    monkeypatch.setattr(
        "ultron.desktop.screen_context.build_screen_context", boom,
    )
    intent = ScreenContextIntent()
    result = handle_screen_context_query(intent)
    assert result.success is False
    assert "snapshot build failed" in (result.error or "")
