"""Window placement primitives -- move / resize / maximize on a target monitor.

Backed by pywin32. Per-monitor DPI awareness is left to whatever the
calling process declared; on this codebase the orchestrator hasn't
called ``SetProcessDpiAwareness`` explicitly, so window coordinates
land in legacy virtual-screen space (consistent with what
:mod:`ultron.desktop.monitors` returns).

Fail-open at every layer: pywin32 exceptions degrade to a
:class:`PlacementResult` with ``success=False`` rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import win32con  # type: ignore[import]
import win32gui  # type: ignore[import]

from ultron.desktop.monitors import Monitor
from ultron.utils.logging import get_logger

logger = get_logger("desktop.placement")


@dataclass(frozen=True)
class PlacementResult:
    """Outcome of a placement operation."""

    success: bool
    hwnd: int
    monitor_index: Optional[int] = None
    error: Optional[str] = None


def _restore_if_minimized(hwnd: int) -> None:
    """Restore a minimized window so subsequent move calls take effect.

    Move-window APIs on a minimized window get queued / ignored on
    some Windows versions. Restore first, then place.
    """
    try:
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    except Exception as e:  # noqa: BLE001
        logger.debug("restore failed hwnd=%d: %s", hwnd, e)


def move_window_to_monitor(
    hwnd: int,
    monitor: Monitor,
    *,
    fullscreen: bool = False,
    maximize: bool = False,
    size: Optional[tuple[int, int]] = None,
    offset: tuple[int, int] = (0, 0),
) -> PlacementResult:
    """Move and size a window to fit a target monitor.

    Args:
        hwnd: Win32 window handle.
        monitor: target :class:`Monitor`.
        fullscreen: place the window to fully cover the monitor and
            keep it as a regular window (no SW_MAXIMIZE). Useful when
            an app's own fullscreen mode is preferred.
        maximize: after moving, also call ``ShowWindow(SW_MAXIMIZE)``.
            Cannot combine with ``fullscreen``.
        size: optional explicit ``(width, height)``. Ignored when
            ``fullscreen`` or ``maximize`` is set.
        offset: ``(x, y)`` offset within the monitor's work area when
            ``size`` is set (e.g. ``(50, 50)`` shifts the window 50 px
            in from the work-area top-left). Ignored when ``fullscreen``
            or ``maximize`` is set.

    Returns:
        :class:`PlacementResult` with ``success=True`` on best-effort
        completion. Note that on Windows, ``SetForegroundWindow`` /
        ``ShowWindow`` may be partially honored depending on focus
        lock state -- a ``True`` here means the call succeeded, not
        that the user-perceived result is guaranteed.
    """
    # Anticheat-safe mode: hard-blocked while the user is in game.
    from ultron.safety.anticheat import guard as _anticheat_guard
    _anticheat_guard('window_move')
    if fullscreen and maximize:
        return PlacementResult(
            success=False, hwnd=hwnd, monitor_index=monitor.index,
            error="fullscreen and maximize are mutually exclusive",
        )

    _restore_if_minimized(hwnd)

    try:
        if fullscreen:
            win32gui.MoveWindow(
                hwnd, monitor.x, monitor.y,
                monitor.width, monitor.height, True,
            )
        elif maximize:
            # Move to the target monitor first so SW_MAXIMIZE maximises
            # to the monitor we want, then maximise.
            win32gui.MoveWindow(
                hwnd, monitor.work_x, monitor.work_y,
                max(monitor.work_width // 2, 400),
                max(monitor.work_height // 2, 300),
                True,
            )
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
        elif size is not None:
            w, h = max(int(size[0]), 100), max(int(size[1]), 100)
            ox, oy = int(offset[0]), int(offset[1])
            # Clamp within the work area so the title bar stays accessible.
            x = monitor.work_x + max(0, min(ox, monitor.work_width - w))
            y = monitor.work_y + max(0, min(oy, monitor.work_height - h))
            win32gui.MoveWindow(hwnd, x, y, w, h, True)
        else:
            # No size + no fullscreen + no maximize: just shift the
            # window to the monitor's work-area origin, preserve size.
            l, t, r, b = win32gui.GetWindowRect(hwnd)
            cur_w, cur_h = max(100, r - l), max(100, b - t)
            x = monitor.work_x + 32
            y = monitor.work_y + 32
            # Clamp so window isn't entirely off-screen.
            x = min(x, monitor.right - 100)
            y = min(y, monitor.bottom - 100)
            win32gui.MoveWindow(hwnd, x, y, cur_w, cur_h, True)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "move_window_to_monitor hwnd=%d monitor=%d failed: %s",
            hwnd, monitor.index, e,
        )
        return PlacementResult(
            success=False, hwnd=hwnd, monitor_index=monitor.index,
            error=str(e)[:200],
        )

    return PlacementResult(
        success=True, hwnd=hwnd, monitor_index=monitor.index,
    )


def maximize_window(hwnd: int) -> PlacementResult:
    """Maximize on whichever monitor the window currently sits."""
    # Anticheat-safe mode: hard-blocked while the user is in game.
    from ultron.safety.anticheat import guard as _anticheat_guard
    _anticheat_guard('window_move')
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
    except Exception as e:  # noqa: BLE001
        return PlacementResult(success=False, hwnd=hwnd, error=str(e)[:200])
    return PlacementResult(success=True, hwnd=hwnd)


def minimize_window(hwnd: int) -> PlacementResult:
    """Minimize the window to the taskbar."""
    # Anticheat-safe mode: hard-blocked while the user is in game.
    from ultron.safety.anticheat import guard as _anticheat_guard
    _anticheat_guard('window_move')
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
    except Exception as e:  # noqa: BLE001
        return PlacementResult(success=False, hwnd=hwnd, error=str(e)[:200])
    return PlacementResult(success=True, hwnd=hwnd)


def restore_window(hwnd: int) -> PlacementResult:
    """Restore (un-minimize / un-maximize) the window."""
    # Anticheat-safe mode: hard-blocked while the user is in game.
    from ultron.safety.anticheat import guard as _anticheat_guard
    _anticheat_guard('window_move')
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    except Exception as e:  # noqa: BLE001
        return PlacementResult(success=False, hwnd=hwnd, error=str(e)[:200])
    return PlacementResult(success=True, hwnd=hwnd)


def focus_window(hwnd: int) -> PlacementResult:
    """Bring a window to the foreground.

    Windows' ``SetForegroundWindow`` is subject to the foreground-lock
    rules and may silently no-op when called from a background thread
    while the user is interacting with another window. We do a best
    effort: restore-if-minimized, then SetForegroundWindow, then
    BringWindowToTop as a fallback. The result reports success of the
    call, not the user-perceived outcome.
    """
    # Anticheat-safe mode: hard-blocked while the user is in game.
    from ultron.safety.anticheat import guard as _anticheat_guard
    _anticheat_guard('window_focus')
    _restore_if_minimized(hwnd)
    try:
        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "SetForegroundWindow hwnd=%d returned: %s; trying BringWindowToTop",
                hwnd, e,
            )
            win32gui.BringWindowToTop(hwnd)
    except Exception as e:  # noqa: BLE001
        return PlacementResult(success=False, hwnd=hwnd, error=str(e)[:200])
    return PlacementResult(success=True, hwnd=hwnd)


def _is_minimized(hwnd: int) -> Optional[bool]:
    """Return True iff iconic; None when the probe fails."""
    try:
        return bool(win32gui.IsIconic(hwnd))
    except Exception as exc:  # noqa: BLE001
        logger.debug("IsIconic hwnd=%d failed: %s", hwnd, exc)
        return None


def _is_maximized(hwnd: int) -> Optional[bool]:
    """Return True iff zoomed (maximised); None when the probe fails."""
    try:
        return bool(win32gui.IsZoomed(hwnd))
    except Exception as exc:  # noqa: BLE001
        logger.debug("IsZoomed hwnd=%d failed: %s", hwnd, exc)
        return None


def minimize_window_idempotent(hwnd: int) -> PlacementResult:
    """Minimize ``hwnd`` only if it is not already minimized.

    Catalog 08 T6 creative extension. Mirrors the upstream
    "idempotent state transition" discipline: check
    :func:`win32gui.IsIconic` before acting, return a result whose
    ``error`` field carries an explanatory string when the call was a
    no-op. The ``success=True`` flag covers BOTH the no-op and the
    actual transition; callers that want to distinguish should inspect
    ``error`` for the ``"already minimized"`` sentinel.

    Returns:
        :class:`PlacementResult` with ``success=True`` on both already-
        minimized AND post-minimize success. ``error="already minimized"``
        when the call was a no-op.
    """
    state = _is_minimized(hwnd)
    if state is True:
        return PlacementResult(
            success=True, hwnd=hwnd, error="already minimized",
        )
    return minimize_window(hwnd)


def maximize_window_idempotent(hwnd: int) -> PlacementResult:
    """Maximize ``hwnd`` only if it is not already maximized.

    Same idempotent-state pattern as :func:`minimize_window_idempotent`.
    """
    state = _is_maximized(hwnd)
    if state is True:
        return PlacementResult(
            success=True, hwnd=hwnd, error="already maximized",
        )
    return maximize_window(hwnd)


def restore_window_idempotent(hwnd: int) -> PlacementResult:
    """Restore ``hwnd`` only when it is in a non-restored state.

    Same idempotent-state pattern: when the window is neither minimized
    nor maximized (i.e., already in the NORMAL state), the call is a
    no-op with ``error="already restored"``.
    """
    minimized = _is_minimized(hwnd)
    maximized = _is_maximized(hwnd)
    if minimized is False and maximized is False:
        return PlacementResult(
            success=True, hwnd=hwnd, error="already restored",
        )
    return restore_window(hwnd)


__all__ = [
    "PlacementResult",
    "move_window_to_monitor",
    "maximize_window",
    "minimize_window",
    "restore_window",
    "focus_window",
    "minimize_window_idempotent",
    "maximize_window_idempotent",
    "restore_window_idempotent",
]
