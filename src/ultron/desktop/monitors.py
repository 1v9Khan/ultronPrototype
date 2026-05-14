"""Win32 monitor enumeration.

Backed by ``win32api.EnumDisplayMonitors`` + ``GetMonitorInfo``.
No external native binaries; ``pywin32`` is already in the stack.

The :class:`Monitor` dataclass is the single shape callers consume.
Monitor indexing is stable across calls: primary monitor is always
index 0; the rest are ordered left-to-right by x-coordinate. This
matches how users count ("my second monitor", "my third monitor")
and avoids depending on Windows' internal monitor handle ordering
(which can change after display configuration changes).

User-facing labels (2026-05-14): the ``find_monitor`` helper resolves
``"main"`` to the *physical center* monitor by virtual-screen
coordinate, not the Windows-designated primary. On the user's setup
the primary is the right monitor; calling that "main" doesn't match
how the user thinks of the displays. ``"left"`` and ``"right"`` use
physical position (leftmost / rightmost by x). With three monitors
this gives the natural mapping; with two, ``"main"`` collapses to
``"left"`` (the only non-right monitor). When the user wants the
Windows-primary specifically, use ``"primary"`` or the explicit
display index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import win32api  # type: ignore[import]
import win32con  # type: ignore[import]

from ultron.utils.logging import get_logger

logger = get_logger("desktop.monitors")


@dataclass(frozen=True)
class Monitor:
    """A connected display.

    Attributes:
        index: 0-based ordinal. 0 = primary; 1, 2, ... ordered
            left-to-right by x-coordinate.
        name: Win32 display name (``\\\\.\\DISPLAY1`` etc.).
        x: leftmost pixel coordinate of the monitor in virtual-screen space.
        y: topmost pixel coordinate.
        width: width in pixels.
        height: height in pixels.
        work_x: x of the work area (excludes taskbar).
        work_y: y of the work area.
        work_width: width of the work area.
        work_height: height of the work area.
        is_primary: True for the primary display.
    """

    index: int
    name: str
    x: int
    y: int
    width: int
    height: int
    work_x: int
    work_y: int
    work_width: int
    work_height: int
    is_primary: bool

    @property
    def right(self) -> int:
        """Rightmost pixel coordinate (exclusive)."""
        return self.x + self.width

    @property
    def bottom(self) -> int:
        """Bottommost pixel coordinate (exclusive)."""
        return self.y + self.height

    @property
    def center(self) -> tuple[int, int]:
        """(x, y) of the geometric center."""
        return (self.x + self.width // 2, self.y + self.height // 2)


# Ordinal words callers may use to address monitors.
_ORDINAL_WORDS: dict[str, int] = {
    "first": 0, "1st": 0, "one": 0,
    "second": 1, "2nd": 1, "two": 1, "secondary": 1,
    "third": 2, "3rd": 2, "three": 2,
    "fourth": 3, "4th": 3, "four": 3,
    "fifth": 4, "5th": 4, "five": 4,
}


def enumerate_monitors() -> list[Monitor]:
    """List all connected monitors.

    Returns monitors with primary first, then left-to-right by x.
    On enumeration failure, returns an empty list (fail-open).
    """
    try:
        raw = win32api.EnumDisplayMonitors()
    except Exception as e:  # noqa: BLE001 -- pywin32 raises generic errors
        logger.warning("EnumDisplayMonitors failed: %s", e)
        return []

    rows: list[dict] = []
    for hmon, _hdc, _rect in raw:
        try:
            info = win32api.GetMonitorInfo(hmon)
        except Exception as e:  # noqa: BLE001
            logger.warning("GetMonitorInfo failed for handle: %s", e)
            continue
        mon_rect = info.get("Monitor")
        work_rect = info.get("Work") or mon_rect
        flags = info.get("Flags", 0)
        device = info.get("Device", "")
        if mon_rect is None:
            continue
        rows.append({
            "rect": mon_rect,
            "work": work_rect,
            "name": str(device),
            "is_primary": bool(flags & win32con.MONITORINFOF_PRIMARY),
        })

    if not rows:
        return []

    rows.sort(key=lambda m: (0 if m["is_primary"] else 1, m["rect"][0]))

    monitors: list[Monitor] = []
    for idx, m in enumerate(rows):
        left, top, right, bottom = m["rect"]
        wl, wt, wr, wb = m["work"]
        monitors.append(
            Monitor(
                index=idx,
                name=m["name"],
                x=int(left),
                y=int(top),
                width=int(right - left),
                height=int(bottom - top),
                work_x=int(wl),
                work_y=int(wt),
                work_width=int(wr - wl),
                work_height=int(wb - wt),
                is_primary=m["is_primary"],
            )
        )
    return monitors


def find_monitor(query: Union[str, int, None]) -> Optional[Monitor]:
    """Resolve a user-friendly monitor reference.

    Accepted forms:

    - ``int`` -- direct index.
    - numeric string (``"0"``, ``"2"``) -- direct index.
    - ``"main"`` / ``"default"`` -- the *physical center* monitor
      (matches user intuition for "main" on a 3-monitor setup).
      Collapses to ``"left"`` on a 2-monitor setup, or the sole
      monitor on a 1-monitor setup. 2026-05-14: this was changed
      from "Windows-designated primary" because the user's primary
      is their right monitor physically, not their center one.
    - ``"primary"`` -- the Windows-designated primary monitor (kept
      separate from "main" so callers who specifically want the
      Win32 primary still have a way to ask for it).
    - ordinal words (``"first"``, ``"second"``, etc.).
    - directional (``"left"``, ``"right"``, ``"center"``,
      ``"top"``, ``"bottom"``).
    - device name substring (``"DISPLAY2"``).

    Returns None when the query doesn't resolve.
    """
    monitors = enumerate_monitors()
    if not monitors:
        return None

    if isinstance(query, int):
        return monitors[query] if 0 <= query < len(monitors) else None

    if query is None:
        return None

    q = str(query).strip().lower()
    if not q:
        return None

    # Numeric index in string form.
    try:
        idx = int(q)
        return monitors[idx] if 0 <= idx < len(monitors) else None
    except ValueError:
        pass

    # "main" / "default" -- physical center monitor (user-facing semantic).
    # "primary" -- Windows-designated primary (Win32 semantic).
    if q in {"main", "default"}:
        return _center_monitor(monitors)
    if q == "primary":
        for m in monitors:
            if m.is_primary:
                return m
        return monitors[0]

    # Ordinal words.
    if q in _ORDINAL_WORDS:
        idx = _ORDINAL_WORDS[q]
        return monitors[idx] if 0 <= idx < len(monitors) else None

    # Directional.
    if q == "left":
        return min(monitors, key=lambda m: m.x)
    if q == "right":
        return max(monitors, key=lambda m: m.x + m.width)
    if q in {"top", "upper", "above"}:
        return min(monitors, key=lambda m: m.y)
    if q in {"bottom", "lower", "below"}:
        return max(monitors, key=lambda m: m.y + m.height)
    if q in {"center", "middle", "centre"}:
        return _center_monitor(monitors)

    # Device name substring match (last-ditch).
    for m in monitors:
        if q in m.name.lower():
            return m

    return None


def _center_monitor(monitors: list[Monitor]) -> Monitor:
    """Pick the monitor whose center x is closest to the virtual-screen
    midpoint -- the user's "main" monitor on a multi-display setup.

    For 1 monitor: returns it. For 2 monitors: returns the leftmost
    (arbitrary but stable; users with 2 monitors typically use "left"
    / "right" or "primary" / "secondary" rather than "main"). For 3+
    monitors with a clear physical layout: returns the middle one.
    """
    if len(monitors) <= 1:
        return monitors[0]
    if len(monitors) == 2:
        return min(monitors, key=lambda m: m.x)
    all_left = min(m.x for m in monitors)
    all_right = max(m.x + m.width for m in monitors)
    target_cx = (all_left + all_right) // 2
    return min(monitors, key=lambda m: abs(m.center[0] - target_cx))


def point_to_monitor(x: int, y: int) -> Optional[Monitor]:
    """Return the monitor containing point (x, y), or None if outside all monitors."""
    for m in enumerate_monitors():
        if m.x <= x < m.right and m.y <= y < m.bottom:
            return m
    return None
