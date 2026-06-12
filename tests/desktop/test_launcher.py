"""Tests for ultron.desktop.launcher.

These avoid actually spawning processes -- subprocess.Popen is patched
out and the safety validator is mocked. The real-Chrome integration
test is skipped by default (requires a live Chrome and visible monitor).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from ultron.desktop.launcher import (
    AppEntry,
    AppLauncher,
    LaunchResult,
    _default_registry,
    _validate_launch,
    get_app_launcher,
    set_app_launcher,
)
from ultron.desktop.monitors import Monitor


def _mon(idx=0) -> Monitor:
    return Monitor(
        index=idx, name=f"D{idx}",
        x=0, y=0, width=1920, height=1080,
        work_x=0, work_y=0, work_width=1920, work_height=1040,
        is_primary=True,
    )


# ---------------------------------------------------------------------------
# Registry lookup
# ---------------------------------------------------------------------------


def test_default_registry_has_known_apps():
    names = {e.name for e in _default_registry()}
    assert "chrome" in names
    assert "cursor" in names
    assert "discord" in names
    assert "explorer" in names


def test_find_app_by_canonical_name():
    launcher = AppLauncher()
    assert launcher.find_app("chrome").name == "chrome"
    assert launcher.find_app("Chrome").name == "chrome"  # case-insensitive
    assert launcher.find_app("CHROME").name == "chrome"


def test_find_app_by_alias():
    launcher = AppLauncher()
    assert launcher.find_app("google chrome").name == "chrome"
    assert launcher.find_app("vs code").name == "vscode"
    assert launcher.find_app("code").name == "vscode"


def test_find_app_substring_fallback():
    launcher = AppLauncher()
    # "chr" is a substring of "chrome"
    assert launcher.find_app("chr").name == "chrome"


def test_find_app_unknown_returns_none():
    launcher = AppLauncher()
    assert launcher.find_app("nonexistent_app_xyz") is None
    assert launcher.find_app("") is None
    assert launcher.find_app(None) is None


# ---------------------------------------------------------------------------
# Executable resolution
# ---------------------------------------------------------------------------


def test_resolve_executable_picks_first_existing(tmp_path):
    launcher = AppLauncher()
    real = tmp_path / "fake.exe"
    real.write_text("dummy")
    entry = AppEntry(
        name="fake_app",
        candidate_paths=[
            tmp_path / "nonexistent1.exe",
            real,
            tmp_path / "nonexistent2.exe",
        ],
    )
    assert launcher.resolve_executable(entry) == real


def test_resolve_executable_none_when_no_candidate_exists(tmp_path):
    launcher = AppLauncher()
    entry = AppEntry(
        name="fake_app",
        candidate_paths=[tmp_path / "ghost.exe"],
    )
    assert launcher.resolve_executable(entry) is None


def test_resolve_chrome_picks_first_existing(tmp_path):
    """Chrome's resolver looks for ``Application/chrome.exe`` directly."""
    launcher = AppLauncher()
    chrome_path = tmp_path / "chrome.exe"
    chrome_path.write_text("")
    entry = AppEntry(
        name="chrome",
        candidate_paths=[tmp_path / "missing.exe", chrome_path],
        process_name="chrome.exe",
    )
    assert launcher.resolve_executable(entry) == chrome_path


# ---------------------------------------------------------------------------
# launch_app safety gate
# ---------------------------------------------------------------------------


def _allow_all_validator():
    from ultron.safety.validator import ValidatorVerdict, Verdict
    return ValidatorVerdict(verdict=Verdict.ALLOW, reason="test allow-all")


def _block_validator(reason="test block"):
    from ultron.safety.validator import ValidatorVerdict, Verdict
    return ValidatorVerdict(
        verdict=Verdict.BLOCK_HARD, reason=reason,
        triggered_rule_id="test", user_message=f"refused: {reason}",
    )


def test_launch_app_unknown_app_returns_failure():
    launcher = AppLauncher()
    r = launcher.launch_app("never_existed_app")
    assert r.success is False
    assert "registry" in (r.error or "")


def test_launch_app_no_exe_on_disk_returns_failure(tmp_path):
    launcher = AppLauncher(registry=[
        AppEntry(name="fake", candidate_paths=[tmp_path / "ghost.exe"]),
    ])
    r = launcher.launch_app("fake")
    assert r.success is False
    assert "no candidate path exists" in (r.error or "")


def test_launch_app_validator_block_short_circuits(tmp_path, monkeypatch):
    real = tmp_path / "harmless.exe"
    real.write_text("")
    launcher = AppLauncher(registry=[
        AppEntry(name="fake", candidate_paths=[real]),
    ])
    monkeypatch.setattr(
        "ultron.desktop.launcher._validate_launch",
        lambda *a, **kw: _block_validator("blocked by test rule"),
    )
    spawned = []
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda *a, **kw: spawned.append((a, kw)) or MagicMock(pid=1),
    )
    r = launcher.launch_app("fake")
    assert r.success is False
    assert "safety" in (r.error or "")
    assert spawned == [], "Popen must not be called when validator blocks"


def test_launch_app_spawn_failure_returns_failure(tmp_path, monkeypatch):
    real = tmp_path / "harmless.exe"
    real.write_text("")
    launcher = AppLauncher(registry=[
        AppEntry(name="fake", candidate_paths=[real]),
    ])
    monkeypatch.setattr(
        "ultron.desktop.launcher._validate_launch",
        lambda *a, **kw: _allow_all_validator(),
    )
    def boom(*a, **kw):
        raise OSError("simulated spawn failure")
    monkeypatch.setattr("subprocess.Popen", boom)
    r = launcher.launch_app("fake", wait_for_window=False)
    assert r.success is False
    assert "spawn failed" in (r.error or "")


def test_launch_app_success_no_window_wait(tmp_path, monkeypatch):
    real = tmp_path / "harmless.exe"
    real.write_text("")
    launcher = AppLauncher(registry=[
        AppEntry(name="fake", candidate_paths=[real], process_name="harmless.exe"),
    ])
    monkeypatch.setattr(
        "ultron.desktop.launcher._validate_launch",
        lambda *a, **kw: _allow_all_validator(),
    )
    fake_proc = MagicMock(pid=12345)
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: fake_proc)
    r = launcher.launch_app("fake", wait_for_window=False)
    assert r.success is True
    assert r.pid == 12345
    assert r.app_name == "fake"
    assert r.exe_path == real
    assert r.window_appeared is None  # no window wait requested


def test_launch_app_window_appears_and_moves_to_monitor(tmp_path, monkeypatch):
    real = tmp_path / "harmless.exe"
    real.write_text("")
    launcher = AppLauncher(
        registry=[AppEntry(
            name="fake",
            candidate_paths=[real],
            process_name="harmless.exe",
        )],
        window_wait_seconds=1.0,
        poll_interval_seconds=0.01,
    )
    monkeypatch.setattr(
        "ultron.desktop.launcher._validate_launch",
        lambda *a, **kw: _allow_all_validator(),
    )
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: MagicMock(pid=12345))

    # Initially no harmless.exe windows; after one poll, one appears.
    from ultron.desktop.windows import WindowInfo
    call_counter = {"n": 0}

    def fake_enum_windows(**kwargs):
        call_counter["n"] += 1
        if call_counter["n"] <= 1:
            return []  # no windows yet
        return [WindowInfo(
            hwnd=42, title="Harmless App", class_name="Cls",
            process_name="harmless.exe", pid=12345,
            rect=(0, 0, 800, 600),
            monitor_index=0, is_minimized=False, is_foreground=True,
        )]

    monkeypatch.setattr(
        "ultron.desktop.launcher.enumerate_windows", fake_enum_windows,
    )
    placement_called = []
    monkeypatch.setattr(
        "ultron.desktop.launcher.move_window_to_monitor",
        lambda hwnd, monitor, **kw: placement_called.append((hwnd, monitor, kw))
        or __import__("ultron.desktop.placement", fromlist=["PlacementResult"])
            .PlacementResult(success=True, hwnd=hwnd, monitor_index=monitor.index),
    )
    focused = []
    monkeypatch.setattr(
        "ultron.desktop.launcher.focus_window",
        lambda hwnd: focused.append(hwnd),
    )

    r = launcher.launch_app(
        "fake", monitor=_mon(idx=1), wait_for_window=True, fullscreen=True,
    )
    assert r.success is True
    assert r.hwnd == 42
    assert r.monitor_index == 1
    assert r.window_appeared is True
    assert len(placement_called) == 1
    assert placement_called[0][0] == 42
    assert placement_called[0][2]["fullscreen"] is True
    # 2026-06-12 bring-to-front fix: placement is followed by a focus.
    assert focused == [42]


def test_launch_app_window_doesnt_appear_within_timeout(tmp_path, monkeypatch):
    real = tmp_path / "harmless.exe"
    real.write_text("")
    launcher = AppLauncher(
        registry=[AppEntry(
            name="fake", candidate_paths=[real], process_name="harmless.exe",
        )],
        window_wait_seconds=0.1, poll_interval_seconds=0.02,
    )
    monkeypatch.setattr(
        "ultron.desktop.launcher._validate_launch",
        lambda *a, **kw: _allow_all_validator(),
    )
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: MagicMock(pid=1))
    monkeypatch.setattr(
        "ultron.desktop.launcher.enumerate_windows", lambda **kw: [],
    )
    monkeypatch.setattr(
        "ultron.desktop.launcher.focus_window",
        lambda hwnd: pytest.fail("focus must not run on window timeout"),
    )
    r = launcher.launch_app("fake", monitor=_mon(), wait_for_window=True)
    assert r.success is True  # process spawned
    assert r.hwnd is None
    assert r.window_appeared is False
    assert "window did not appear" in (r.error or "")


def test_launch_app_focuses_unplaced_window(tmp_path, monkeypatch):
    real = tmp_path / "harmless.exe"
    real.write_text("")
    launcher = AppLauncher(
        registry=[AppEntry(
            name="fake", candidate_paths=[real], process_name="harmless.exe",
        )],
        window_wait_seconds=1.0, poll_interval_seconds=0.01,
    )
    monkeypatch.setattr(
        "ultron.desktop.launcher._validate_launch",
        lambda *a, **kw: _allow_all_validator(),
    )
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: MagicMock(pid=7))

    from ultron.desktop.windows import WindowInfo
    calls = {"n": 0}

    def fake_enum_windows(**kwargs):
        calls["n"] += 1
        if calls["n"] <= 1:
            return []
        return [WindowInfo(
            hwnd=77, title="Harmless App", class_name="Cls",
            process_name="harmless.exe", pid=7, rect=(0, 0, 800, 600),
            monitor_index=0, is_minimized=False, is_foreground=False,
        )]

    monkeypatch.setattr(
        "ultron.desktop.launcher.enumerate_windows", fake_enum_windows,
    )
    focused = []
    monkeypatch.setattr(
        "ultron.desktop.launcher.focus_window",
        lambda hwnd: focused.append(hwnd),
    )
    r = launcher.launch_app("fake", monitor=None, wait_for_window=True)
    assert r.success is True
    assert r.window_appeared is True
    assert r.monitor_index is None
    assert focused == [77]


def test_launch_app_focus_failure_is_fail_open(tmp_path, monkeypatch):
    real = tmp_path / "harmless.exe"
    real.write_text("")
    launcher = AppLauncher(
        registry=[AppEntry(
            name="fake", candidate_paths=[real], process_name="harmless.exe",
        )],
        window_wait_seconds=1.0, poll_interval_seconds=0.01,
    )
    monkeypatch.setattr(
        "ultron.desktop.launcher._validate_launch",
        lambda *a, **kw: _allow_all_validator(),
    )
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: MagicMock(pid=7))

    from ultron.desktop.windows import WindowInfo
    calls = {"n": 0}

    def fake_enum_windows(**kwargs):
        calls["n"] += 1
        if calls["n"] <= 1:
            return []  # pre-spawn snapshot: no windows yet
        return [WindowInfo(
            hwnd=88, title="App", class_name="Cls",
            process_name="harmless.exe", pid=7, rect=(0, 0, 800, 600),
            monitor_index=0, is_minimized=False, is_foreground=False,
        )]

    monkeypatch.setattr(
        "ultron.desktop.launcher.enumerate_windows", fake_enum_windows,
    )

    def boom_focus(hwnd):
        raise RuntimeError("anticheat engaged mid-launch")

    monkeypatch.setattr("ultron.desktop.launcher.focus_window", boom_focus)
    placement_mod = __import__(
        "ultron.desktop.placement", fromlist=["PlacementResult"]
    )
    monkeypatch.setattr(
        "ultron.desktop.launcher.move_window_to_monitor",
        lambda hwnd, monitor, **kw: placement_mod.PlacementResult(
            success=True, hwnd=hwnd, monitor_index=monitor.index,
        ),
    )
    r = launcher.launch_app("fake", monitor=_mon(), wait_for_window=True)
    # Focus failure never degrades a successful launch.
    assert r.success is True
    assert r.hwnd == 88
    assert r.window_appeared is True


def test_launch_app_focus_called_even_when_placement_fails(
    tmp_path, monkeypatch
):
    real = tmp_path / "harmless.exe"
    real.write_text("")
    launcher = AppLauncher(
        registry=[AppEntry(
            name="fake", candidate_paths=[real], process_name="harmless.exe",
        )],
        window_wait_seconds=1.0, poll_interval_seconds=0.01,
    )
    monkeypatch.setattr(
        "ultron.desktop.launcher._validate_launch",
        lambda *a, **kw: _allow_all_validator(),
    )
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: MagicMock(pid=7))

    from ultron.desktop.windows import WindowInfo
    calls = {"n": 0}

    def fake_enum_windows(**kwargs):
        calls["n"] += 1
        if calls["n"] <= 1:
            return []  # pre-spawn snapshot: no windows yet
        return [WindowInfo(
            hwnd=99, title="App", class_name="Cls",
            process_name="harmless.exe", pid=7, rect=(0, 0, 800, 600),
            monitor_index=0, is_minimized=False, is_foreground=False,
        )]

    monkeypatch.setattr(
        "ultron.desktop.launcher.enumerate_windows", fake_enum_windows,
    )
    placement_mod = __import__(
        "ultron.desktop.placement", fromlist=["PlacementResult"]
    )
    monkeypatch.setattr(
        "ultron.desktop.launcher.move_window_to_monitor",
        lambda hwnd, monitor, **kw: placement_mod.PlacementResult(
            success=False, hwnd=hwnd, error="move failed",
        ),
    )
    focused = []
    monkeypatch.setattr(
        "ultron.desktop.launcher.focus_window",
        lambda hwnd: focused.append(hwnd),
    )
    r = launcher.launch_app("fake", monitor=_mon(), wait_for_window=True)
    assert r.success is True
    assert focused == [99]


# ---------------------------------------------------------------------------
# Chrome convenience
# ---------------------------------------------------------------------------


def test_launch_chrome_not_installed_returns_failure(tmp_path):
    launcher = AppLauncher(registry=[
        AppEntry(
            name="chrome",
            candidate_paths=[tmp_path / "ghost.exe"],
            process_name="chrome.exe",
        ),
    ])
    r = launcher.launch_chrome(url="https://example.com")
    assert r.success is False
    assert "not installed" in (r.error or "")


def test_launch_chrome_passes_new_window_and_url(monkeypatch, tmp_path):
    chrome_path = tmp_path / "chrome.exe"
    chrome_path.write_text("")
    launcher = AppLauncher(
        registry=[AppEntry(
            name="chrome",
            candidate_paths=[chrome_path],
            process_name="chrome.exe",
        )],
        window_wait_seconds=0.1, poll_interval_seconds=0.02,
    )
    monkeypatch.setattr(
        "ultron.desktop.launcher._validate_launch",
        lambda *a, **kw: _allow_all_validator(),
    )
    monkeypatch.setattr(
        "ultron.desktop.launcher.enumerate_windows", lambda **kw: [],
    )
    spawned: list = []
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda cmd, **kw: spawned.append(cmd) or MagicMock(pid=1),
    )

    r = launcher.launch_chrome(url="https://youtube.com", new_window=True)
    assert r.success is True
    cmd = spawned[0]
    assert str(chrome_path) in cmd[0]
    assert "--new-window" in cmd
    assert "https://youtube.com" in cmd


def test_launch_chrome_no_user_data_dir_or_debug_flags(monkeypatch, tmp_path):
    """The launcher must never set --remote-debugging-port or
    --user-data-dir. Those are the exact flags Cap-3 blocks.
    """
    chrome_path = tmp_path / "chrome.exe"
    chrome_path.write_text("")
    launcher = AppLauncher(
        registry=[AppEntry(
            name="chrome", candidate_paths=[chrome_path],
            process_name="chrome.exe",
        )],
        window_wait_seconds=0.05, poll_interval_seconds=0.02,
    )
    monkeypatch.setattr(
        "ultron.desktop.launcher._validate_launch",
        lambda *a, **kw: _allow_all_validator(),
    )
    monkeypatch.setattr(
        "ultron.desktop.launcher.enumerate_windows", lambda **kw: [],
    )
    spawned = []
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda cmd, **kw: spawned.append(cmd) or MagicMock(pid=1),
    )
    launcher.launch_chrome(url="https://example.com")
    cmd = spawned[0]
    cmdstr = " ".join(cmd)
    assert "--remote-debugging-port" not in cmdstr
    assert "--remote-debugging-pipe" not in cmdstr
    assert "--user-data-dir" not in cmdstr
    assert "--load-extension" not in cmdstr
    assert "--disable-web-security" not in cmdstr


def test_open_image_search_builds_google_images_url(monkeypatch, tmp_path):
    chrome_path = tmp_path / "chrome.exe"
    chrome_path.write_text("")
    launcher = AppLauncher(
        registry=[AppEntry(
            name="chrome", candidate_paths=[chrome_path],
            process_name="chrome.exe",
        )],
        window_wait_seconds=0.05, poll_interval_seconds=0.02,
    )
    monkeypatch.setattr(
        "ultron.desktop.launcher._validate_launch",
        lambda *a, **kw: _allow_all_validator(),
    )
    monkeypatch.setattr(
        "ultron.desktop.launcher.enumerate_windows", lambda **kw: [],
    )
    spawned = []
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda cmd, **kw: spawned.append(cmd) or MagicMock(pid=1),
    )
    launcher.open_image_search("golden retriever")
    cmdstr = " ".join(spawned[0])
    assert "google.com/search" in cmdstr
    assert "tbm=isch" in cmdstr
    assert "golden" in cmdstr  # URL-encoded query


def test_open_image_search_empty_query_returns_failure():
    launcher = AppLauncher()
    r = launcher.open_image_search("")
    assert r.success is False


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


def test_get_app_launcher_singleton_caches():
    set_app_launcher(None)
    try:
        a = get_app_launcher()
        b = get_app_launcher()
        assert a is b
    finally:
        set_app_launcher(None)


def test_set_app_launcher_swaps():
    custom = AppLauncher(registry=[])
    try:
        set_app_launcher(custom)
        assert get_app_launcher() is custom
    finally:
        set_app_launcher(None)
