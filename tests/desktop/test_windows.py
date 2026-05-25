"""Tests for ultron.desktop.windows."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from ultron.desktop.monitors import Monitor
from ultron.desktop.windows import (
    DEFAULT_WAIT_INTERVAL_S,
    DEFAULT_WAIT_TIMEOUT_S,
    FocusResult,
    WindowInfo,
    _appactivate_via_powershell,
    _appactivate_via_wscript_shell,
    _monitor_index_for_rect,
    _set_foreground_window,
    enumerate_windows,
    find_window,
    focus_by_title,
    get_foreground_window,
    wait_for_window,
)


# ---------------------------------------------------------------------------
# WindowInfo dataclass shape
# ---------------------------------------------------------------------------


def test_window_info_helpers():
    w = WindowInfo(
        hwnd=12345, title="My App", class_name="Cls",
        process_name="app.exe", pid=999,
        rect=(100, 200, 300, 500),
        monitor_index=0, is_minimized=False, is_foreground=True,
    )
    assert w.width == 200
    assert w.height == 300
    assert w.center == (200, 350)


def test_window_info_handles_inverted_rect():
    """Rects from minimized windows can have right<left. width/height clamp to 0."""
    w = WindowInfo(
        hwnd=1, title="t", class_name="c", process_name="p", pid=1,
        rect=(500, 500, 100, 100),
        monitor_index=None, is_minimized=True, is_foreground=False,
    )
    assert w.width == 0
    assert w.height == 0


# ---------------------------------------------------------------------------
# _monitor_index_for_rect logic
# ---------------------------------------------------------------------------


def _three_mons() -> list[Monitor]:
    return [
        Monitor(  # 0 = primary, (0,0)..(2048,1152)
            index=0, name="D1",
            x=0, y=0, width=2048, height=1152,
            work_x=0, work_y=0, work_width=2048, work_height=1112,
            is_primary=True,
        ),
        Monitor(  # 1 = left, (-1920,186)..(0,1266)
            index=1, name="D3",
            x=-1920, y=186, width=1920, height=1080,
            work_x=-1920, work_y=186, work_width=1920, work_height=1040,
            is_primary=False,
        ),
        Monitor(  # 2 = right, (2560,106)..(4480,1186)
            index=2, name="D2",
            x=2560, y=106, width=1920, height=1080,
            work_x=2560, work_y=106, work_width=1920, work_height=1040,
            is_primary=False,
        ),
    ]


def test_monitor_index_for_rect_fully_inside_primary():
    mons = _three_mons()
    assert _monitor_index_for_rect((100, 100, 500, 500), mons) == 0


def test_monitor_index_for_rect_left_monitor():
    mons = _three_mons()
    assert _monitor_index_for_rect((-1500, 300, -100, 800), mons) == 1


def test_monitor_index_for_rect_right_monitor():
    mons = _three_mons()
    assert _monitor_index_for_rect((3000, 200, 4000, 800), mons) == 2


def test_monitor_index_for_rect_straddles_picks_greatest_overlap():
    mons = _three_mons()
    # Window straddles primary (mostly) and right monitor (a sliver).
    # Primary overlap: x ∈ (1000..2048) = 1048 wide × y 100..500 = 400 tall = 419200
    # Right overlap:   x ∈ (2560..3100) = 540  wide × y 100..500 = 400 tall = 216000  (no — y on right starts at 106)
    # Right overlap actually: x∈(2560..3100)=540 × y∈(106..500)=394 = 212760
    # So primary wins.
    assert _monitor_index_for_rect((1000, 100, 3100, 500), mons) == 0


def test_monitor_index_for_rect_no_overlap_returns_none():
    mons = _three_mons()
    # Above all monitors
    assert _monitor_index_for_rect((100, -1000, 500, -500), mons) is None


def test_monitor_index_for_rect_degenerate_rect_returns_none():
    mons = _three_mons()
    assert _monitor_index_for_rect((100, 100, 100, 100), mons) is None
    assert _monitor_index_for_rect((500, 500, 400, 400), mons) is None


def test_monitor_index_for_rect_empty_monitors():
    assert _monitor_index_for_rect((0, 0, 100, 100), []) is None


# ---------------------------------------------------------------------------
# find_window scoring with synthetic candidate list
# ---------------------------------------------------------------------------


def _mk(title, proc, *, fg=False, mon=0, hwnd=0):
    return WindowInfo(
        hwnd=hwnd or hash((title, proc)) & 0xFFFFFFFF,
        title=title,
        class_name="Cls",
        process_name=proc,
        pid=0,
        rect=(0, 0, 100, 100),
        monitor_index=mon,
        is_minimized=False,
        is_foreground=fg,
    )


def test_find_window_substring_match(monkeypatch):
    candidates = [
        _mk("Cursor - main.py", "Cursor.exe"),
        _mk("Discord", "Discord.exe"),
    ]
    monkeypatch.setattr("ultron.desktop.windows.enumerate_windows", lambda **_: candidates)
    w = find_window("cursor")
    assert w is not None
    assert w.process_name == "Cursor.exe"


def test_find_window_by_process_name(monkeypatch):
    candidates = [
        _mk("My Document - Word", "WINWORD.EXE"),
    ]
    monkeypatch.setattr("ultron.desktop.windows.enumerate_windows", lambda **_: candidates)
    w = find_window("winword")
    assert w is not None
    assert w.process_name == "WINWORD.EXE"


def test_find_window_exact_title_outranks_substring(monkeypatch):
    candidates = [
        _mk("My Project - Cursor", "Cursor.exe"),
        _mk("cursor", "Cursor.exe"),  # exact-match (case-insensitive)
    ]
    monkeypatch.setattr("ultron.desktop.windows.enumerate_windows", lambda **_: candidates)
    w = find_window("cursor")
    assert w is not None
    # exact title match (after lower()) ranks ahead of partial title
    assert w.title == "cursor"


def test_find_window_foreground_breaks_tie(monkeypatch):
    candidates = [
        _mk("chrome.exe", "chrome.exe", fg=False, hwnd=1),
        _mk("chrome.exe", "chrome.exe", fg=True, hwnd=2),
    ]
    monkeypatch.setattr("ultron.desktop.windows.enumerate_windows", lambda **_: candidates)
    w = find_window("chrome", prefer_foreground=True)
    assert w.is_foreground


def test_find_window_monitor_preference_breaks_tie(monkeypatch):
    candidates = [
        _mk("chrome - tab 1", "chrome.exe", mon=0, hwnd=1),
        _mk("chrome - tab 2", "chrome.exe", mon=2, hwnd=2),
    ]
    monkeypatch.setattr("ultron.desktop.windows.enumerate_windows", lambda **_: candidates)
    w = find_window("chrome", prefer_monitor=2)
    assert w.monitor_index == 2


def test_find_window_empty_query(monkeypatch):
    monkeypatch.setattr("ultron.desktop.windows.enumerate_windows", lambda **_: [])
    assert find_window("") is None
    assert find_window("   ") is None


def test_find_window_no_match(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.windows.enumerate_windows",
        lambda **_: [_mk("Cursor", "Cursor.exe")],
    )
    assert find_window("nonexistent") is None


def test_find_window_disable_process_match(monkeypatch):
    candidates = [_mk("My Document", "WINWORD.EXE")]
    monkeypatch.setattr(
        "ultron.desktop.windows.enumerate_windows", lambda **_: candidates,
    )
    # With by_process=False, "winword" doesn't match title.
    assert find_window("winword", by_process=False) is None
    # ... but matches when by_process=True (the default).
    assert find_window("winword", by_process=True) is not None


# ---------------------------------------------------------------------------
# Live integration (Windows only)
# ---------------------------------------------------------------------------


pytestmark_windows = pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows-only (pywin32 window enumeration)",
)


@pytestmark_windows
def test_enumerate_windows_live_returns_some():
    wins = enumerate_windows()
    assert len(wins) >= 1, "expected at least one visible window"


@pytestmark_windows
def test_enumerate_windows_live_all_have_titles_by_default():
    wins = enumerate_windows()
    assert all(w.title.strip() for w in wins), "require_title=True drops empty"


@pytestmark_windows
def test_enumerate_windows_live_monitor_indices_valid():
    from ultron.desktop.monitors import enumerate_monitors

    mon_count = len(enumerate_monitors())
    if mon_count == 0:
        pytest.skip("no monitors detected")
    wins = enumerate_windows()
    for w in wins:
        if w.monitor_index is not None:
            assert 0 <= w.monitor_index < mon_count


@pytestmark_windows
def test_get_foreground_window_live():
    fg = get_foreground_window()
    # In an interactive session there's always SOME foreground window; in
    # weird CI states there may not be. Skip if so.
    if fg is None:
        pytest.skip("no foreground window in current session")
    assert fg.is_foreground
    assert fg.hwnd > 0


# ---------------------------------------------------------------------------
# T6 focus_by_title
# ---------------------------------------------------------------------------


def _focus_target() -> WindowInfo:
    return _mk("Cursor - main.py", "Cursor.exe", hwnd=42, mon=1)


class TestFocusByTitle:

    def test_empty_title_returns_failure(self):
        result = focus_by_title("")
        assert isinstance(result, FocusResult)
        assert result.success is False
        assert "empty" in (result.error or "").lower()

    def test_whitespace_only_returns_failure(self):
        result = focus_by_title("    ")
        assert result.success is False

    def test_primary_path_success(self, monkeypatch):
        target = _focus_target()
        monkeypatch.setattr(
            "ultron.desktop.windows.find_window",
            lambda *a, **k: target,
        )
        monkeypatch.setattr(
            "ultron.desktop.windows._set_foreground_window",
            lambda hwnd: True,
        )
        result = focus_by_title("cursor")
        assert result.success is True
        assert result.window is target
        assert result.method == "set_foreground_window"

    def test_primary_path_failure_falls_through_to_com(self, monkeypatch):
        target = _focus_target()
        focused_after = _mk("Cursor - main.py", "Cursor.exe", fg=True, hwnd=42)
        monkeypatch.setattr(
            "ultron.desktop.windows.find_window",
            lambda *a, **k: target,
        )
        monkeypatch.setattr(
            "ultron.desktop.windows._set_foreground_window",
            lambda hwnd: False,
        )
        monkeypatch.setattr(
            "ultron.desktop.windows._appactivate_via_wscript_shell",
            lambda title: True,
        )
        monkeypatch.setattr(
            "ultron.desktop.windows.get_foreground_window",
            lambda: focused_after,
        )
        result = focus_by_title("cursor")
        assert result.success is True
        assert result.method == "app_activate_com"
        assert result.window is focused_after

    def test_no_match_falls_through_to_com(self, monkeypatch):
        monkeypatch.setattr(
            "ultron.desktop.windows.find_window",
            lambda *a, **k: None,
        )
        monkeypatch.setattr(
            "ultron.desktop.windows._appactivate_via_wscript_shell",
            lambda title: True,
        )
        monkeypatch.setattr(
            "ultron.desktop.windows.get_foreground_window",
            lambda: None,
        )
        result = focus_by_title("nothing-here")
        assert result.success is True
        assert result.method == "app_activate_com"

    def test_com_failure_falls_through_to_powershell(self, monkeypatch):
        target = _focus_target()
        monkeypatch.setattr(
            "ultron.desktop.windows.find_window",
            lambda *a, **k: target,
        )
        monkeypatch.setattr(
            "ultron.desktop.windows._set_foreground_window",
            lambda hwnd: False,
        )
        monkeypatch.setattr(
            "ultron.desktop.windows._appactivate_via_wscript_shell",
            lambda title: False,
        )
        monkeypatch.setattr(
            "ultron.desktop.windows._appactivate_via_powershell",
            lambda title, *, timeout_s=2.0: True,
        )
        monkeypatch.setattr(
            "ultron.desktop.windows.get_foreground_window",
            lambda: None,
        )
        result = focus_by_title("cursor")
        assert result.success is True
        assert result.method == "app_activate_powershell"

    def test_all_paths_fail(self, monkeypatch):
        target = _focus_target()
        monkeypatch.setattr(
            "ultron.desktop.windows.find_window",
            lambda *a, **k: target,
        )
        monkeypatch.setattr(
            "ultron.desktop.windows._set_foreground_window",
            lambda hwnd: False,
        )
        monkeypatch.setattr(
            "ultron.desktop.windows._appactivate_via_wscript_shell",
            lambda title: False,
        )
        monkeypatch.setattr(
            "ultron.desktop.windows._appactivate_via_powershell",
            lambda title, *, timeout_s=2.0: False,
        )
        result = focus_by_title("cursor")
        assert result.success is False
        assert "no AppActivate fallback" in (result.error or "")
        # The candidate window is reported in the result so caller can
        # inspect what we tried.
        assert result.window is target

    def test_fallback_disabled_returns_after_primary_failure(self, monkeypatch):
        target = _focus_target()
        monkeypatch.setattr(
            "ultron.desktop.windows.find_window",
            lambda *a, **k: target,
        )
        monkeypatch.setattr(
            "ultron.desktop.windows._set_foreground_window",
            lambda hwnd: False,
        )
        # Sentinels: if we DO reach a fallback, the test fails loudly.
        monkeypatch.setattr(
            "ultron.desktop.windows._appactivate_via_wscript_shell",
            lambda title: pytest.fail("fallback should not have run"),
        )
        result = focus_by_title("cursor", fall_back_to_app_activate=False)
        assert result.success is False
        assert "SetForegroundWindow refused" in (result.error or "")
        assert result.window is target

    def test_fallback_disabled_with_no_candidate(self, monkeypatch):
        monkeypatch.setattr(
            "ultron.desktop.windows.find_window",
            lambda *a, **k: None,
        )
        result = focus_by_title("nothing", fall_back_to_app_activate=False)
        assert result.success is False
        assert result.window is None
        assert "no window matching" in (result.error or "")

    def test_prefer_monitor_threaded_to_find_window(self, monkeypatch):
        captured: dict[str, Any] = {}

        def _fake_find_window(query, *, prefer_monitor=None, **kwargs):
            captured["query"] = query
            captured["prefer_monitor"] = prefer_monitor
            return None

        monkeypatch.setattr(
            "ultron.desktop.windows.find_window", _fake_find_window
        )
        monkeypatch.setattr(
            "ultron.desktop.windows._appactivate_via_wscript_shell",
            lambda title: False,
        )
        monkeypatch.setattr(
            "ultron.desktop.windows._appactivate_via_powershell",
            lambda title, *, timeout_s=2.0: False,
        )
        focus_by_title("cursor", prefer_monitor=2)
        assert captured["query"] == "cursor"
        assert captured["prefer_monitor"] == 2


# ---------------------------------------------------------------------------
# Helper exposure: _set_foreground_window / AppActivate paths
# ---------------------------------------------------------------------------


class TestSetForegroundWindow:

    def test_success(self, monkeypatch):
        monkeypatch.setattr(
            "ultron.desktop.windows.win32gui.SetForegroundWindow",
            lambda hwnd: 1,
        )
        assert _set_foreground_window(12345) is True

    def test_failure_returns_false(self, monkeypatch):
        monkeypatch.setattr(
            "ultron.desktop.windows.win32gui.SetForegroundWindow",
            lambda hwnd: 0,
        )
        assert _set_foreground_window(12345) is False

    def test_exception_returns_false(self, monkeypatch):
        def _boom(hwnd):
            raise RuntimeError("foreground lock")

        monkeypatch.setattr(
            "ultron.desktop.windows.win32gui.SetForegroundWindow", _boom,
        )
        assert _set_foreground_window(12345) is False


class TestAppActivateWscriptShell:

    def test_returns_true_when_com_returns_true(self, monkeypatch):
        class _FakeShell:
            def AppActivate(self, title):  # noqa: N802
                return True

        class _FakeClient:
            @staticmethod
            def Dispatch(name):  # noqa: N802
                return _FakeShell()

        class _FakeModule:
            client = _FakeClient

        monkeypatch.setitem(sys.modules, "win32com", _FakeModule)
        monkeypatch.setitem(sys.modules, "win32com.client", _FakeClient)
        assert _appactivate_via_wscript_shell("cursor") is True

    def test_returns_false_when_com_returns_false(self, monkeypatch):
        class _FakeShell:
            def AppActivate(self, title):  # noqa: N802
                return False

        class _FakeClient:
            @staticmethod
            def Dispatch(name):  # noqa: N802
                return _FakeShell()

        class _FakeModule:
            client = _FakeClient

        monkeypatch.setitem(sys.modules, "win32com", _FakeModule)
        monkeypatch.setitem(sys.modules, "win32com.client", _FakeClient)
        assert _appactivate_via_wscript_shell("cursor") is False

    def test_returns_false_when_dispatch_raises(self, monkeypatch):
        class _FakeClient:
            @staticmethod
            def Dispatch(name):  # noqa: N802
                raise RuntimeError("COM error")

        class _FakeModule:
            client = _FakeClient

        monkeypatch.setitem(sys.modules, "win32com", _FakeModule)
        monkeypatch.setitem(sys.modules, "win32com.client", _FakeClient)
        assert _appactivate_via_wscript_shell("cursor") is False

    def test_returns_false_when_module_unavailable(self, monkeypatch):
        # Force the import to fail by stashing a sentinel non-module.
        monkeypatch.setitem(sys.modules, "win32com", None)
        assert _appactivate_via_wscript_shell("cursor") is False


class TestAppActivateViaPowershell:

    def _fake_completed(self, *, stdout: str, returncode: int = 0):
        cp = subprocess.CompletedProcess(args=[], returncode=returncode)
        cp.stdout = stdout
        cp.stderr = ""
        return cp

    def test_success_returns_true(self, monkeypatch):
        if sys.platform != "win32":
            pytest.skip("platform guard short-circuits off-Windows")
        monkeypatch.setattr(
            "ultron.desktop.windows.subprocess.run",
            lambda *a, **k: self._fake_completed(stdout="True"),
        )
        assert _appactivate_via_powershell("cursor") is True

    def test_false_output_returns_false(self, monkeypatch):
        if sys.platform != "win32":
            pytest.skip("platform guard short-circuits off-Windows")
        monkeypatch.setattr(
            "ultron.desktop.windows.subprocess.run",
            lambda *a, **k: self._fake_completed(stdout="False"),
        )
        assert _appactivate_via_powershell("cursor") is False

    def test_off_windows_returns_false_without_subprocess(self, monkeypatch):
        monkeypatch.setattr("ultron.desktop.windows.sys.platform", "linux")

        def _boom(*a, **k):
            raise AssertionError("subprocess should not run off-Windows")

        monkeypatch.setattr(
            "ultron.desktop.windows.subprocess.run", _boom,
        )
        assert _appactivate_via_powershell("cursor") is False

    def test_timeout_returns_false(self, monkeypatch):
        if sys.platform != "win32":
            pytest.skip("platform guard short-circuits off-Windows")

        def _timeout(*a, **k):
            raise subprocess.TimeoutExpired(cmd="powershell.exe", timeout=2.0)

        monkeypatch.setattr(
            "ultron.desktop.windows.subprocess.run", _timeout,
        )
        assert _appactivate_via_powershell("cursor", timeout_s=0.1) is False

    def test_filenotfound_returns_false(self, monkeypatch):
        if sys.platform != "win32":
            pytest.skip("platform guard short-circuits off-Windows")

        def _missing(*a, **k):
            raise FileNotFoundError("powershell.exe not found")

        monkeypatch.setattr(
            "ultron.desktop.windows.subprocess.run", _missing,
        )
        assert _appactivate_via_powershell("cursor") is False

    def test_escapes_single_quotes_in_title(self, monkeypatch):
        if sys.platform != "win32":
            pytest.skip("platform guard short-circuits off-Windows")
        captured: dict[str, Any] = {}

        def _capture(args, *a, **k):
            captured["args"] = list(args)
            return self._fake_completed(stdout="True")

        monkeypatch.setattr(
            "ultron.desktop.windows.subprocess.run", _capture,
        )
        assert _appactivate_via_powershell("It's a Title") is True
        cmd_arg = captured["args"][-1]
        # Single quotes must be doubled-up so PowerShell parses the
        # literal apostrophe instead of closing the string early.
        assert "It''s a Title" in cmd_arg

    def test_creation_flags_suppress_console(self, monkeypatch):
        if sys.platform != "win32":
            pytest.skip("platform guard short-circuits off-Windows")
        captured: dict[str, Any] = {}

        def _capture(args, *a, **k):
            captured["kwargs"] = k
            return self._fake_completed(stdout="True")

        monkeypatch.setattr(
            "ultron.desktop.windows.subprocess.run", _capture,
        )
        _appactivate_via_powershell("anything")
        # CREATE_NO_WINDOW = 0x08000000 must be set so no console flashes.
        assert captured["kwargs"].get("creationflags", 0) == 0x08000000


# ---------------------------------------------------------------------------
# enumerate_windows -- exclude_cloaked plumbing
# ---------------------------------------------------------------------------


class TestEnumerateWindowsExcludeCloaked:

    def _patch_enumeration(
        self,
        monkeypatch,
        *,
        hwnds: list[int],
        cloaked_hwnds: set[int],
        visible: bool = True,
    ) -> None:
        """Stub win32gui.EnumWindows + is_window_cloaked + per-window
        inspection so the filter logic is testable off Windows too."""

        def _fake_enum_windows(callback, _arg):
            for hwnd in hwnds:
                callback(hwnd, None)

        monkeypatch.setattr(
            "ultron.desktop.windows.win32gui.EnumWindows",
            _fake_enum_windows,
        )
        monkeypatch.setattr(
            "ultron.desktop.windows.win32gui.IsWindowVisible",
            lambda hwnd: visible,
        )
        monkeypatch.setattr(
            "ultron.desktop.windows.win32gui.GetForegroundWindow",
            lambda: 0,
        )
        monkeypatch.setattr(
            "ultron.desktop.windows.enumerate_monitors",
            lambda: [],
        )

        def _fake_build(hwnd, monitors, fg_hwnd):
            return WindowInfo(
                hwnd=int(hwnd),
                title=f"win-{hwnd}",
                class_name="Cls",
                process_name="proc.exe",
                pid=0,
                rect=(0, 0, 100, 100),
                monitor_index=None,
                is_minimized=False,
                is_foreground=False,
            )

        monkeypatch.setattr(
            "ultron.desktop.windows._build_window_info", _fake_build,
        )

        def _fake_is_cloaked(hwnd):
            return True if hwnd in cloaked_hwnds else False

        monkeypatch.setattr(
            "ultron.desktop.win32_helpers.is_window_cloaked",
            _fake_is_cloaked,
        )

    def test_default_excludes_cloaked(self, monkeypatch):
        self._patch_enumeration(
            monkeypatch,
            hwnds=[1, 2, 3, 4],
            cloaked_hwnds={2, 4},
        )
        wins = enumerate_windows()
        hwnds = sorted(w.hwnd for w in wins)
        assert hwnds == [1, 3]

    def test_explicit_false_includes_cloaked(self, monkeypatch):
        self._patch_enumeration(
            monkeypatch,
            hwnds=[1, 2, 3, 4],
            cloaked_hwnds={2, 4},
        )
        wins = enumerate_windows(exclude_cloaked=False)
        hwnds = sorted(w.hwnd for w in wins)
        assert hwnds == [1, 2, 3, 4]

    def test_unknown_cloaked_state_includes(self, monkeypatch):
        # is_window_cloaked returning None must NOT exclude (legacy behaviour).
        def _patch_unknown(hwnd):  # noqa: ARG001
            return None

        self._patch_enumeration(
            monkeypatch,
            hwnds=[7, 8],
            cloaked_hwnds=set(),
        )
        monkeypatch.setattr(
            "ultron.desktop.win32_helpers.is_window_cloaked",
            _patch_unknown,
        )
        wins = enumerate_windows()
        hwnds = sorted(w.hwnd for w in wins)
        assert hwnds == [7, 8]

    def test_cloaked_probe_exception_includes_legacy(self, monkeypatch):
        def _boom(hwnd):  # noqa: ARG001
            raise RuntimeError("dwmapi unavailable")

        self._patch_enumeration(
            monkeypatch,
            hwnds=[10, 11],
            cloaked_hwnds=set(),
        )
        monkeypatch.setattr(
            "ultron.desktop.win32_helpers.is_window_cloaked", _boom,
        )
        wins = enumerate_windows()
        hwnds = sorted(w.hwnd for w in wins)
        # Exception means "couldn't tell" -> include both.
        assert hwnds == [10, 11]


def test_find_window_forwards_exclude_cloaked(monkeypatch):
    captured: dict[str, Any] = {}

    def _fake_enum(*, exclude_cloaked=True, **kwargs):
        captured["exclude_cloaked"] = exclude_cloaked
        return []

    monkeypatch.setattr(
        "ultron.desktop.windows.enumerate_windows", _fake_enum,
    )
    find_window("anything", exclude_cloaked=False)
    assert captured["exclude_cloaked"] is False
    find_window("anything", exclude_cloaked=True)
    assert captured["exclude_cloaked"] is True


# ---------------------------------------------------------------------------
# Catalog 08 T4: wait_for_window
# ---------------------------------------------------------------------------


class _FakeClock:
    def __init__(self) -> None:
        self._t = 0.0

    def __call__(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += dt


def _make_target(hwnd: int = 1, title: str = "Notepad") -> WindowInfo:
    return WindowInfo(
        hwnd=hwnd, title=title, class_name="Notepad",
        process_name="notepad.exe", pid=4242,
        rect=(0, 0, 600, 400), monitor_index=0,
        is_minimized=False, is_foreground=False,
    )


def test_wait_for_window_constants_match_upstream():
    assert DEFAULT_WAIT_TIMEOUT_S == 30.0
    assert DEFAULT_WAIT_INTERVAL_S == 0.5


def test_wait_for_window_returns_none_on_empty_title():
    assert wait_for_window("") is None
    assert wait_for_window("   ") is None


def test_wait_for_window_returns_none_on_zero_timeout(monkeypatch):
    # find_window must not even be consulted when timeout is 0.
    sentinel = {"called": False}

    def _no_call(**kw):
        sentinel["called"] = True
        return _make_target()

    monkeypatch.setattr("ultron.desktop.windows.find_window", _no_call)
    result = wait_for_window("notepad", timeout_s=0.0)
    assert result is None
    assert sentinel["called"] is False


def test_wait_for_window_found_on_first_poll(monkeypatch):
    target = _make_target(hwnd=77, title="Notepad - Untitled")
    monkeypatch.setattr(
        "ultron.desktop.windows.find_window", lambda **kw: target,
    )
    clock = _FakeClock()
    slept = []
    result = wait_for_window(
        "notepad", timeout_s=10.0, interval_s=0.5,
        sleep_fn=lambda s: slept.append(s),
        clock_fn=clock,
    )
    assert result is target
    assert slept == []


def test_wait_for_window_polls_until_appears(monkeypatch):
    target = _make_target(hwnd=88, title="Save As")
    calls = [0]

    def _appear_on_third_poll(**kw):
        calls[0] += 1
        return target if calls[0] >= 3 else None

    monkeypatch.setattr(
        "ultron.desktop.windows.find_window", _appear_on_third_poll,
    )
    clock = _FakeClock()

    def _sleep(dt: float) -> None:
        clock.advance(dt)

    result = wait_for_window(
        "save", timeout_s=5.0, interval_s=0.1,
        sleep_fn=_sleep,
        clock_fn=clock,
    )
    assert result is target
    assert calls[0] == 3


def test_wait_for_window_returns_none_on_timeout(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.windows.find_window", lambda **kw: None,
    )
    clock = _FakeClock()
    sleeps: list[float] = []

    def _sleep(dt: float) -> None:
        sleeps.append(dt)
        clock.advance(dt)

    result = wait_for_window(
        "missing", timeout_s=1.0, interval_s=0.25,
        sleep_fn=_sleep,
        clock_fn=clock,
    )
    assert result is None
    # 1.0s timeout / 0.25s interval -> roughly 4 sleeps.
    assert 3 <= len(sleeps) <= 5


def test_wait_for_window_forwards_keyword_args_to_find_window(monkeypatch):
    captured: dict[str, Any] = {}

    def _capture(**kw):
        captured.update(kw)
        return _make_target()

    monkeypatch.setattr(
        "ultron.desktop.windows.find_window", _capture,
    )
    wait_for_window(
        "notepad",
        timeout_s=1.0, interval_s=0.1,
        by_process=False,
        exclude_cloaked=False,
        prefer_monitor=2,
        sleep_fn=lambda s: None,
        clock_fn=_FakeClock(),
    )
    assert captured["by_process"] is False
    assert captured["exclude_cloaked"] is False
    assert captured["prefer_monitor"] == 2
    # Foreground preference is suppressed for "appears" polling.
    assert captured["prefer_foreground"] is False


def test_wait_for_window_fail_open_on_find_exception(monkeypatch):
    """find_window raising should NOT abort the loop -- the next poll
    iteration retries. After timeout the function returns None."""
    raised = [0]

    def _raise(**kw):
        raised[0] += 1
        raise RuntimeError("simulated UIA failure")

    monkeypatch.setattr(
        "ultron.desktop.windows.find_window", _raise,
    )
    clock = _FakeClock()

    def _sleep(dt: float) -> None:
        clock.advance(dt)

    result = wait_for_window(
        "notepad", timeout_s=1.0, interval_s=0.25,
        sleep_fn=_sleep,
        clock_fn=clock,
    )
    assert result is None
    assert raised[0] >= 1


def test_wait_for_window_caps_sleep_to_deadline(monkeypatch):
    """If interval > remaining time, the final sleep should be clamped
    to the remaining deadline (no overshoot)."""
    monkeypatch.setattr(
        "ultron.desktop.windows.find_window", lambda **kw: None,
    )
    clock = _FakeClock()
    sleeps: list[float] = []

    def _sleep(dt: float) -> None:
        sleeps.append(dt)
        clock.advance(dt)

    wait_for_window(
        "missing", timeout_s=0.3, interval_s=0.5,
        sleep_fn=_sleep,
        clock_fn=clock,
    )
    # The first sleep should be clamped down from 0.5 to <= 0.3.
    assert sleeps[0] <= 0.3
