"""Tests for ultron.desktop.monitors."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from ultron.desktop.monitors import (
    Monitor,
    enumerate_monitors,
    find_monitor,
    point_to_monitor,
)


# ---------------------------------------------------------------------------
# Monitor dataclass shape
# ---------------------------------------------------------------------------


def test_monitor_geometry_helpers():
    m = Monitor(
        index=0, name="\\\\.\\DISPLAY1",
        x=0, y=0, width=1920, height=1080,
        work_x=0, work_y=0, work_width=1920, work_height=1040,
        is_primary=True,
    )
    assert m.right == 1920
    assert m.bottom == 1080
    assert m.center == (960, 540)


def test_monitor_is_frozen():
    m = Monitor(
        index=0, name="d", x=0, y=0, width=100, height=100,
        work_x=0, work_y=0, work_width=100, work_height=100,
        is_primary=True,
    )
    with pytest.raises(Exception):
        m.x = 1  # frozen dataclass — assignment must raise


# ---------------------------------------------------------------------------
# find_monitor with hand-crafted monitor list (mocks the Win32 call)
# ---------------------------------------------------------------------------


def _fake_monitors() -> list[Monitor]:
    """Three-monitor layout: primary center, secondary left, tertiary right."""
    return [
        Monitor(  # 0 = primary, center
            index=0, name="\\\\.\\DISPLAY1",
            x=0, y=0, width=2048, height=1152,
            work_x=0, work_y=0, work_width=2048, work_height=1112,
            is_primary=True,
        ),
        Monitor(  # 1 = left of primary
            index=1, name="\\\\.\\DISPLAY3",
            x=-1920, y=186, width=1920, height=1080,
            work_x=-1920, work_y=186, work_width=1920, work_height=1040,
            is_primary=False,
        ),
        Monitor(  # 2 = right of primary
            index=2, name="\\\\.\\DISPLAY2",
            x=2560, y=106, width=1920, height=1080,
            work_x=2560, work_y=106, work_width=1920, work_height=1040,
            is_primary=False,
        ),
    ]


@pytest.fixture
def fake_monitors(monkeypatch):
    fakes = _fake_monitors()
    monkeypatch.setattr(
        "ultron.desktop.monitors.enumerate_monitors", lambda: fakes,
    )
    return fakes


def test_find_monitor_by_int_index(fake_monitors):
    assert find_monitor(0).index == 0
    assert find_monitor(2).index == 2


def test_find_monitor_by_numeric_string(fake_monitors):
    assert find_monitor("0").index == 0
    assert find_monitor("2").index == 2


def test_find_monitor_out_of_range_returns_none(fake_monitors):
    assert find_monitor(5) is None
    assert find_monitor("9") is None
    assert find_monitor(-1) is None


def test_find_monitor_primary_aliases(fake_monitors):
    # In this fixture, primary IS the center monitor (DISPLAY1 at x=0..2048
    # spanning the virtual-screen midpoint), so "primary"/"main"/"default"
    # all happen to return the same monitor. The semantic split is
    # exercised by test_find_monitor_main_is_center_not_win32_primary
    # below using a fixture where they diverge.
    assert find_monitor("primary").is_primary
    assert find_monitor("main").is_primary
    assert find_monitor("default").is_primary


def test_find_monitor_main_is_center_not_win32_primary(monkeypatch):
    """2026-05-14: 'main' resolves to physical center, not Win32 primary.

    User's setup has primary = right monitor; calling that "main" doesn't
    match how they think of the displays. Build a fixture mirroring the
    user's layout and assert the new semantics.
    """
    fakes = [
        # Index 0 in our enumeration order: Win32 primary -- the RIGHT one.
        # (Primary sorts to index 0 per enumerate_monitors' sort key.)
        Monitor(
            index=0, name="\\\\.\\DISPLAY2",
            x=3840, y=0, width=1920, height=1080,
            work_x=3840, work_y=0, work_width=1920, work_height=1040,
            is_primary=True,
        ),
        Monitor(  # Index 1 = leftmost
            index=1, name="\\\\.\\DISPLAY3",
            x=0, y=0, width=1920, height=1080,
            work_x=0, work_y=0, work_width=1920, work_height=1040,
            is_primary=False,
        ),
        Monitor(  # Index 2 = center
            index=2, name="\\\\.\\DISPLAY4",
            x=1920, y=0, width=1920, height=1080,
            work_x=1920, work_y=0, work_width=1920, work_height=1040,
            is_primary=False,
        ),
    ]
    monkeypatch.setattr(
        "ultron.desktop.monitors.enumerate_monitors", lambda: fakes,
    )
    # "primary" still maps to Win32 primary (right).
    assert find_monitor("primary").name == "\\\\.\\DISPLAY2"
    # "main" / "default" / "center" / "middle" map to physical center.
    assert find_monitor("main").name == "\\\\.\\DISPLAY4"
    assert find_monitor("default").name == "\\\\.\\DISPLAY4"
    assert find_monitor("center").name == "\\\\.\\DISPLAY4"
    assert find_monitor("middle").name == "\\\\.\\DISPLAY4"
    # Left / right by virtual-screen position.
    assert find_monitor("left").name == "\\\\.\\DISPLAY3"
    assert find_monitor("right").name == "\\\\.\\DISPLAY2"


def test_find_monitor_main_single_monitor(monkeypatch):
    """'main' on a 1-monitor setup returns that monitor."""
    fakes = [
        Monitor(
            index=0, name="\\\\.\\DISPLAY1",
            x=0, y=0, width=1920, height=1080,
            work_x=0, work_y=0, work_width=1920, work_height=1040,
            is_primary=True,
        ),
    ]
    monkeypatch.setattr(
        "ultron.desktop.monitors.enumerate_monitors", lambda: fakes,
    )
    assert find_monitor("main").index == 0
    assert find_monitor("center").index == 0


def test_find_monitor_main_two_monitor_falls_back_left(monkeypatch):
    """'main' on a 2-monitor setup collapses to leftmost (deterministic)."""
    fakes = [
        Monitor(  # primary, right
            index=0, name="\\\\.\\DISPLAY1",
            x=1920, y=0, width=1920, height=1080,
            work_x=1920, work_y=0, work_width=1920, work_height=1040,
            is_primary=True,
        ),
        Monitor(  # left
            index=1, name="\\\\.\\DISPLAY2",
            x=0, y=0, width=1920, height=1080,
            work_x=0, work_y=0, work_width=1920, work_height=1040,
            is_primary=False,
        ),
    ]
    monkeypatch.setattr(
        "ultron.desktop.monitors.enumerate_monitors", lambda: fakes,
    )
    assert find_monitor("main").name == "\\\\.\\DISPLAY2"  # leftmost


def test_find_monitor_ordinal_words(fake_monitors):
    assert find_monitor("first").index == 0
    assert find_monitor("second").index == 1
    assert find_monitor("third").index == 2
    assert find_monitor("1st").index == 0
    assert find_monitor("2nd").index == 1
    assert find_monitor("3rd").index == 2


def test_find_monitor_directional(fake_monitors):
    # Left = smallest x (DISPLAY3 at x=-1920)
    assert find_monitor("left").index == 1
    # Right = largest right edge (DISPLAY2 right edge at 2560+1920=4480)
    assert find_monitor("right").index == 2
    # Top = smallest y (DISPLAY1 at y=0)
    assert find_monitor("top").index == 0
    # Center: virtual screen left=-1920, right=4480, center ~1280 → DISPLAY1 center 1024
    assert find_monitor("center").index == 0
    assert find_monitor("middle").index == 0


def test_find_monitor_device_name_substring(fake_monitors):
    assert find_monitor("DISPLAY1").index == 0
    assert find_monitor("display3").index == 1  # case-insensitive


def test_find_monitor_empty_or_none(fake_monitors):
    assert find_monitor("") is None
    assert find_monitor(None) is None
    assert find_monitor("   ") is None


def test_find_monitor_unknown(fake_monitors):
    assert find_monitor("nonexistent") is None


def test_find_monitor_empty_list(monkeypatch):
    monkeypatch.setattr(
        "ultron.desktop.monitors.enumerate_monitors", lambda: [],
    )
    assert find_monitor(0) is None
    assert find_monitor("primary") is None
    assert find_monitor("left") is None


def test_point_to_monitor_inside(fake_monitors):
    # Point (1000, 500) is inside DISPLAY1 (primary, 0..2048 x 0..1152)
    m = point_to_monitor(1000, 500)
    assert m is not None and m.index == 0


def test_point_to_monitor_left_monitor(fake_monitors):
    # Point (-500, 500) is inside DISPLAY3 (-1920..0 x 186..1266)
    m = point_to_monitor(-500, 500)
    assert m is not None and m.index == 1


def test_point_to_monitor_outside(fake_monitors):
    # Point above all monitors
    assert point_to_monitor(0, -1000) is None
    # Point right of all monitors
    assert point_to_monitor(99999, 500) is None


# ---------------------------------------------------------------------------
# enumerate_monitors fail-open path
# ---------------------------------------------------------------------------


def test_enumerate_monitors_returns_empty_on_pywin32_failure(monkeypatch):
    def boom():
        raise RuntimeError("pywin32 unavailable")

    monkeypatch.setattr("win32api.EnumDisplayMonitors", boom)
    assert enumerate_monitors() == []


def test_enumerate_monitors_skips_failing_handle(monkeypatch):
    """One bad GetMonitorInfo per-handle failure shouldn't kill the whole call."""
    calls = {"n": 0}

    def _fake_get_info(hmon):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("bad handle")
        return {
            "Monitor": (0, 0, 1920, 1080),
            "Work": (0, 0, 1920, 1040),
            "Flags": 1,  # primary
            "Device": "\\\\.\\DISPLAY2",
        }

    monkeypatch.setattr(
        "win32api.EnumDisplayMonitors",
        lambda: [(1, 2, (0, 0, 1, 1)), (3, 4, (0, 0, 1, 1))],
    )
    monkeypatch.setattr("win32api.GetMonitorInfo", _fake_get_info)

    out = enumerate_monitors()
    assert len(out) == 1
    assert out[0].is_primary


# ---------------------------------------------------------------------------
# Live integration (Windows only, real hardware)
# ---------------------------------------------------------------------------


pytestmark_windows = pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows-only (Win32 monitor enumeration)",
)


@pytestmark_windows
def test_enumerate_monitors_live_returns_at_least_one():
    mons = enumerate_monitors()
    assert len(mons) >= 1, "expected at least one monitor on a desktop session"


@pytestmark_windows
def test_enumerate_monitors_live_has_exactly_one_primary():
    mons = enumerate_monitors()
    if not mons:
        pytest.skip("no monitors detected (headless session?)")
    primaries = [m for m in mons if m.is_primary]
    assert len(primaries) == 1, "exactly one primary monitor expected"
    assert primaries[0].index == 0, "primary must sort to index 0"


@pytestmark_windows
def test_enumerate_monitors_live_indices_sequential():
    mons = enumerate_monitors()
    if not mons:
        pytest.skip("no monitors detected")
    indices = [m.index for m in mons]
    assert indices == list(range(len(mons))), "indices must be sequential 0..N"
