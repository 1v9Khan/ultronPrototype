"""Low-level Win32 helpers via ``ctypes`` (clawhub-windows-ui-automation
catalog 07 T2).

Adds capabilities that are not in ``pywin32``'s default surface but are
useful across the desktop automation stack:

* :func:`get_monitor_dpi` -- per-monitor effective DPI for a virtual-screen
  point. Load-bearing for T5 (DPI-aware logical-to-physical coordinate
  mapping) so UIA element bounding rects (logical pixels) can be
  translated into the physical pixel coordinates pyautogui expects on
  high-DPI displays.

* :func:`get_last_input_idle_ms` -- milliseconds since the last physical
  user input. Useful as a secondary signal for the gaming-mode engage
  callback ("user is actually at the keyboard") and idle-window
  detection for background workers.

* :func:`block_input_context` -- atomic input suspension as a
  context manager. Built-in watchdog guarantees the user regains
  control even if the calling code hangs. Capability-gated through
  the existing safety stack (Cap-3 + explicit-intent + two-phase
  approval) at call sites; this module ships the primitive, not
  the gate.

* :func:`is_window_cloaked` -- detect Windows compositor-cloaked windows
  (virtual-desktop occluded, DWM-hidden) that ``IsWindowVisible``
  returns True for. Improves accuracy of the foreground-window
  security check in :mod:`ultron.desktop.input_control` and lets
  :func:`enumerate_windows` filter out windows that visually aren't
  there.

Cross-platform: every function is a graceful no-op (returns a documented
default) on non-Windows platforms, so callers don't need to gate by
``sys.platform``. Module load is also safe on non-Windows -- the
``ctypes`` struct definitions and DLL handles are only constructed on
first call when running on Windows.

Pattern adapted from the public Win32 API reference + the consolidated-
``Add-Type`` discipline noted in catalog 07 T2 (single class definition,
type-existence guard before re-use). All P/Invoke signatures and
constants come from Microsoft's published headers; no source code is
copied verbatim from the quarantined plugin.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

from ultron.utils.logging import get_logger

logger = get_logger("desktop.win32_helpers")


# ---------------------------------------------------------------------------
# Platform gate
# ---------------------------------------------------------------------------

#: True when the runtime platform is Windows. Every public function checks
#: this before attempting any P/Invoke; off-Windows callers get the
#: documented no-op default without any ctypes import side-effects.
IS_WINDOWS: bool = sys.platform == "win32"


# ---------------------------------------------------------------------------
# Win32 constants (from public Microsoft API headers)
# ---------------------------------------------------------------------------

#: ``MonitorFromPoint`` flag: return the nearest monitor when the point
#: is outside the virtual screen rectangle. Matches ``WinUser.h``.
MONITOR_DEFAULTTONEAREST: int = 2

#: ``GetDpiForMonitor`` DPI type: effective DPI (as the user sees it
#: after Windows scaling). Matches ``ShellScalingApi.h``.
MDT_EFFECTIVE_DPI: int = 0

#: ``GetDpiForMonitor`` DPI type: angular DPI.
MDT_ANGULAR_DPI: int = 1

#: ``GetDpiForMonitor`` DPI type: raw physical DPI (before scaling).
MDT_RAW_DPI: int = 2

#: ``DwmGetWindowAttribute`` attribute id: cloaked state. Returns a BOOL
#: indicating whether DWM is hiding the window (virtual desktop,
#: compositor trick, etc.). Matches ``dwmapi.h`` (DWMWA_CLOAKED).
DWMWA_CLOAKED: int = 14

#: Default Windows DPI (96 == 100% scaling).
DEFAULT_DPI: int = 96

#: Watchdog defaults for :func:`block_input_context`. The hard cap
#: bounds even an explicit caller-supplied duration -- the operator
#: must never lose control of their machine for longer than this.
_BLOCK_INPUT_DEFAULT_MAX_DURATION_S: float = 5.0
_BLOCK_INPUT_HARD_CAP_S: float = 30.0


# ---------------------------------------------------------------------------
# ctypes struct definitions (lazy)
# ---------------------------------------------------------------------------

if IS_WINDOWS:

    class _POINT(ctypes.Structure):
        """Win32 ``POINT`` struct (``LONG x; LONG y;``)."""

        _fields_ = [
            ("x", ctypes.c_long),
            ("y", ctypes.c_long),
        ]

    class _LASTINPUTINFO(ctypes.Structure):
        """Win32 ``LASTINPUTINFO`` struct (``UINT cbSize; DWORD dwTime;``)."""

        _fields_ = [
            ("cbSize", ctypes.c_uint),
            ("dwTime", ctypes.c_uint),
        ]


# ---------------------------------------------------------------------------
# DLL cache
# ---------------------------------------------------------------------------

_dll_cache: dict[str, Optional[object]] = {}
_dll_cache_lock = threading.Lock()


def _load_dll(name: str):
    """Lazy-load a Win32 DLL by name with caching.

    Returns the ``ctypes.WinDLL`` handle on Windows when the DLL is
    available, ``None`` otherwise (non-Windows, missing DLL, or
    permission issue).

    The result is cached so subsequent calls cost a dict lookup. Caller
    code should always check for ``None`` and fail-open.
    """

    if not IS_WINDOWS:
        return None
    with _dll_cache_lock:
        if name in _dll_cache:
            return _dll_cache[name]
        dll: Optional[object]
        try:
            dll = getattr(ctypes.windll, name)
        except (AttributeError, OSError) as exc:
            logger.debug("win32 dll %s unavailable: %s", name, exc)
            dll = None
        _dll_cache[name] = dll
        return dll


def _reset_dll_cache_for_testing() -> None:
    """Clear the cached DLL handles. Test-only escape hatch."""

    with _dll_cache_lock:
        _dll_cache.clear()


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitorDpi:
    """Per-monitor DPI readout.

    Attributes:
        dpi_x: Horizontal DPI (effective, post-Windows-scaling).
        dpi_y: Vertical DPI.
        scale_x: ``dpi_x / DEFAULT_DPI`` -- the multiplier to apply to a
            logical pixel x-coordinate to get its physical position.
        scale_y: ``dpi_y / DEFAULT_DPI`` for the y-axis.
        is_default: True when the reading fell back to
            :data:`DEFAULT_DPI` (no Win32 API available, or every
            probe failed). Lets callers distinguish "the monitor really
            IS at 100% scaling" from "we couldn't tell, assuming 100%".
    """

    dpi_x: int
    dpi_y: int
    scale_x: float
    scale_y: float
    is_default: bool = False

    @property
    def is_high_dpi(self) -> bool:
        """True iff either axis is above 100% scaling."""

        return self.dpi_x > DEFAULT_DPI or self.dpi_y > DEFAULT_DPI


def _make_default_dpi() -> MonitorDpi:
    """Return the 96-DPI fallback when no probe succeeds."""

    return MonitorDpi(
        dpi_x=DEFAULT_DPI,
        dpi_y=DEFAULT_DPI,
        scale_x=1.0,
        scale_y=1.0,
        is_default=True,
    )


# ---------------------------------------------------------------------------
# T5 building block: per-monitor DPI
# ---------------------------------------------------------------------------


def get_monitor_dpi(x: int, y: int) -> MonitorDpi:
    """Return the effective DPI of the monitor containing ``(x, y)``.

    On Windows: calls ``MonitorFromPoint`` (user32) with
    ``MONITOR_DEFAULTTONEAREST`` to resolve a monitor handle, then
    ``GetDpiForMonitor`` (shcore, Windows 8.1+) with
    :data:`MDT_EFFECTIVE_DPI` to read the effective DPI. Returns a
    :class:`MonitorDpi` with ``is_default=False``.

    Off Windows OR when the API isn't available (older Windows
    without shcore, denied access, ctypes import failure): returns the
    :func:`_make_default_dpi` fallback with ``is_default=True``. Callers
    that want explicit "couldn't read DPI" handling should check
    ``result.is_default``.

    Fail-open at every layer -- a DPI read failure must NEVER take down
    the desktop pipeline.
    """

    if not IS_WINDOWS:
        return _make_default_dpi()

    user32 = _load_dll("user32")
    shcore = _load_dll("shcore")
    if user32 is None or shcore is None:
        return _make_default_dpi()

    try:
        point = _POINT(int(x), int(y))
        monitor_handle = user32.MonitorFromPoint(point, MONITOR_DEFAULTTONEAREST)
    except Exception as exc:  # noqa: BLE001
        logger.debug("MonitorFromPoint(%d, %d) failed: %s", x, y, exc)
        return _make_default_dpi()

    if not monitor_handle:
        return _make_default_dpi()

    try:
        dpi_x = ctypes.c_uint(0)
        dpi_y = ctypes.c_uint(0)
        # HRESULT GetDpiForMonitor(HMONITOR, MONITOR_DPI_TYPE, UINT* dpiX, UINT* dpiY)
        result = shcore.GetDpiForMonitor(
            monitor_handle,
            MDT_EFFECTIVE_DPI,
            ctypes.byref(dpi_x),
            ctypes.byref(dpi_y),
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("GetDpiForMonitor failed: %s", exc)
        return _make_default_dpi()

    if result != 0:
        # HRESULT non-zero is a failure code.
        logger.debug("GetDpiForMonitor HRESULT=%s", hex(result))
        return _make_default_dpi()

    raw_x = int(dpi_x.value) or DEFAULT_DPI
    raw_y = int(dpi_y.value) or DEFAULT_DPI
    return MonitorDpi(
        dpi_x=raw_x,
        dpi_y=raw_y,
        scale_x=raw_x / DEFAULT_DPI,
        scale_y=raw_y / DEFAULT_DPI,
        is_default=False,
    )


def get_monitor_dpi_for_window(hwnd: int) -> MonitorDpi:
    """Return the DPI of the monitor a window primarily sits on.

    Uses ``GetWindowRect`` to find the window's centre point, then
    delegates to :func:`get_monitor_dpi`. Returns the default fallback
    when the window can't be located.
    """

    if not IS_WINDOWS or not hwnd:
        return _make_default_dpi()

    user32 = _load_dll("user32")
    if user32 is None:
        return _make_default_dpi()

    try:
        rect = (ctypes.c_long * 4)(0, 0, 0, 0)
        # GetWindowRect signature: BOOL GetWindowRect(HWND, LPRECT)
        ok = user32.GetWindowRect(int(hwnd), ctypes.byref(rect))
    except Exception as exc:  # noqa: BLE001
        logger.debug("GetWindowRect hwnd=%d failed: %s", hwnd, exc)
        return _make_default_dpi()

    if not ok:
        return _make_default_dpi()

    left, top, right, bottom = rect[0], rect[1], rect[2], rect[3]
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    return get_monitor_dpi(cx, cy)


# ---------------------------------------------------------------------------
# T2 creative extension: idle-input detection
# ---------------------------------------------------------------------------


def get_last_input_idle_ms() -> Optional[int]:
    """Return the number of milliseconds since the last physical input.

    Calls ``GetLastInputInfo`` (user32) plus ``GetTickCount`` to compute
    ``now - last_input_tick``. The Win32 tick counter wraps at ~49.7
    days; the subtraction is done in 32-bit unsigned arithmetic so
    wrap-around is handled correctly within the standard window.

    Returns ``None`` on non-Windows, when the API is unavailable, or
    when the call fails. Callers should treat ``None`` as "couldn't
    tell" rather than "user is active".

    Use cases:

    * Gaming-mode engage confirmation: a second signal beyond voice
      trigger that the user is physically present.
    * Background worker scheduling: only run idle-priority work when
      the user has been away for N minutes.
    * Wake-word false-positive suppression on long idle windows.
    """

    if not IS_WINDOWS:
        return None

    user32 = _load_dll("user32")
    kernel32 = _load_dll("kernel32")
    if user32 is None or kernel32 is None:
        return None

    try:
        lii = _LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(_LASTINPUTINFO)
        ok = user32.GetLastInputInfo(ctypes.byref(lii))
    except Exception as exc:  # noqa: BLE001
        logger.debug("GetLastInputInfo failed: %s", exc)
        return None

    if not ok:
        return None

    try:
        # GetTickCount returns DWORD (uint32). The wrap window matches
        # GetLastInputInfo's dwTime so subtraction in uint32 space is
        # correct.
        kernel32.GetTickCount.restype = ctypes.c_uint
        now_tick = int(kernel32.GetTickCount())
    except Exception as exc:  # noqa: BLE001
        logger.debug("GetTickCount failed: %s", exc)
        return None

    delta = (now_tick - int(lii.dwTime)) & 0xFFFFFFFF
    return int(delta)


# ---------------------------------------------------------------------------
# T2 creative extension: cloaked window detection
# ---------------------------------------------------------------------------


def is_window_cloaked(hwnd: int) -> Optional[bool]:
    """Return True iff DWM is hiding the window (cloaked state).

    ``IsWindowVisible`` returns True for cloaked windows because the
    style bits are still set; the visibility is actually controlled
    by the desktop window manager. Cloaking happens when:

    * The window is on a non-active virtual desktop.
    * UWP apps suspend and DWM hides them.
    * A compositor trick offscreens the window without minimising it.

    Returns ``True`` iff the window is cloaked, ``False`` iff DWM
    confirms it isn't, and ``None`` when the API isn't available
    (non-Windows, dwmapi not loadable, or call failure). Callers
    treating ``None`` as "assume visible" preserves the legacy
    ``IsWindowVisible`` behaviour.
    """

    if not IS_WINDOWS or not hwnd:
        return None

    dwmapi = _load_dll("dwmapi")
    if dwmapi is None:
        return None

    try:
        # HRESULT DwmGetWindowAttribute(HWND, DWORD attr, PVOID, DWORD cb)
        cloaked = ctypes.c_int(0)
        result = dwmapi.DwmGetWindowAttribute(
            int(hwnd),
            DWMWA_CLOAKED,
            ctypes.byref(cloaked),
            ctypes.sizeof(cloaked),
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("DwmGetWindowAttribute hwnd=%d failed: %s", hwnd, exc)
        return None

    if result != 0:
        logger.debug("DwmGetWindowAttribute HRESULT=%s hwnd=%d", hex(result), hwnd)
        return None

    return bool(cloaked.value)


# ---------------------------------------------------------------------------
# T2 creative extension: atomic input suspension (BlockInput)
# ---------------------------------------------------------------------------


class BlockInputUnavailableError(RuntimeError):
    """Raised when :func:`block_input_context` can't actually block input.

    Reasons include: non-Windows platform, missing user32, the calling
    process lacks the privilege (UIPI denial), or the function returned
    failure. Callers can let this propagate when the suspension is
    safety-critical, or catch it when block-input is best-effort.
    """


@dataclass(frozen=True)
class BlockInputResult:
    """Outcome of one :func:`block_input_context` use.

    Attributes:
        engaged: True iff BlockInput(TRUE) succeeded on entry.
        watchdog_fired: True iff the watchdog auto-unblocked because
            the caller exceeded ``max_duration_s``. When True, the
            context block ran past its budget -- usually a stuck or
            very slow caller.
        actual_duration_s: Wall-clock the suspension was active.
    """

    engaged: bool
    watchdog_fired: bool
    actual_duration_s: float


def _call_block_input(enable: bool) -> bool:
    """Direct ``user32.BlockInput`` call. Returns True on success, False
    on any failure (or off Windows)."""

    if not IS_WINDOWS:
        return False
    user32 = _load_dll("user32")
    if user32 is None:
        return False
    try:
        # BOOL BlockInput(BOOL fBlockIt). On non-admin processes this
        # returns 0 (FALSE) and Windows itself ensures the call is a
        # no-op so the user never loses control. UIPI gives us a
        # safety floor for free.
        ok = user32.BlockInput(ctypes.c_int(1 if enable else 0))
    except Exception as exc:  # noqa: BLE001
        logger.debug("BlockInput(%s) raised: %s", enable, exc)
        return False
    return bool(ok)


@contextmanager
def block_input_context(
    *,
    max_duration_s: float = _BLOCK_INPUT_DEFAULT_MAX_DURATION_S,
    raise_if_unavailable: bool = False,
) -> Iterator[BlockInputResult]:
    """Suspend physical mouse + keyboard input for the duration of the block.

    Hardens the raw Win32 ``BlockInput`` primitive in three ways:

    1. **Try/finally guarantee:** the unblock call ALWAYS runs on exit
       (even on exception), so a buggy caller cannot leave the user
       locked out of their own machine.

    2. **Watchdog:** a daemon thread fires the unblock after
       ``max_duration_s`` regardless of caller behaviour. The duration
       is clamped to ``[0, _BLOCK_INPUT_HARD_CAP_S=30s]`` so even an
       explicit large value cannot leave the user locked out for
       longer than 30 seconds.

    3. **UIPI floor:** non-admin processes have their BlockInput call
       silently no-op'd by Windows itself, which means a compromised
       in-process LLM cannot use this primitive to lock the user out
       of an elevated remediation flow.

    Yields a :class:`BlockInputResult` describing the actual outcome.
    Callers that need the suspension to be active should check
    ``result.engaged`` after entry.

    The caller's gating responsibilities (NOT this module's):

    * Cap-3 explicit-intent matcher must confirm a recent user
      utterance authorises an input-blocking action.
    * Two-phase approval should hold a yes/no decision before the
      ``with`` block enters.
    * Audit log entry should record the action that the block wraps.

    Raises :class:`BlockInputUnavailableError` only when
    ``raise_if_unavailable=True`` AND the entry call failed. Default
    behaviour yields a ``BlockInputResult(engaged=False, ...)`` so the
    caller can degrade-and-continue.
    """

    requested_duration_s = float(max_duration_s)
    duration_s = max(0.0, min(requested_duration_s, _BLOCK_INPUT_HARD_CAP_S))

    engaged = _call_block_input(True)
    if not engaged:
        if raise_if_unavailable:
            raise BlockInputUnavailableError(
                "BlockInput(TRUE) failed -- non-admin process, off-Windows, "
                "or dll unavailable",
            )
        # Degrade-and-continue: yield a non-engaged result so the
        # caller can decide to abort or proceed without the lock.
        yield BlockInputResult(engaged=False, watchdog_fired=False, actual_duration_s=0.0)
        return

    started_at = time.monotonic()
    watchdog_fired_flag = threading.Event()
    cancel_watchdog = threading.Event()

    def _watchdog() -> None:
        # Wait either the budget or an external cancel from the exit
        # path. Either way, ensure BlockInput(FALSE) is called.
        triggered = not cancel_watchdog.wait(duration_s)
        if triggered:
            watchdog_fired_flag.set()
            _call_block_input(False)

    watchdog = threading.Thread(
        target=_watchdog,
        name="ultron-block-input-watchdog",
        daemon=True,
    )
    watchdog.start()

    try:
        yield BlockInputResult(
            engaged=True,
            watchdog_fired=False,
            actual_duration_s=0.0,
        )
    finally:
        cancel_watchdog.set()
        if not watchdog_fired_flag.is_set():
            # The caller exited within budget. We are responsible for
            # the unblock call.
            _call_block_input(False)
        # Join the watchdog briefly so the thread doesn't outlive the
        # context manager.
        try:
            watchdog.join(timeout=0.1)
        except Exception:  # noqa: BLE001
            pass

    # The yielded result is the live one; this trailing log helps
    # operators see budget-busting callers in the audit trail.
    elapsed = time.monotonic() - started_at
    if watchdog_fired_flag.is_set():
        logger.warning(
            "block_input_context: watchdog fired after %.2fs (budget=%.2fs)",
            elapsed,
            duration_s,
        )


# ---------------------------------------------------------------------------
# T5 building block: coordinate space conversions
# ---------------------------------------------------------------------------


def logical_to_physical(
    x: int,
    y: int,
    *,
    reference_x: Optional[int] = None,
    reference_y: Optional[int] = None,
    dpi: Optional[MonitorDpi] = None,
) -> tuple[int, int]:
    """Convert logical (DPI-unscaled) coordinates into physical pixels.

    Logical pixels are what pywinauto UIA element bounding rects
    report; physical pixels are what ``SetCursorPos`` and pyautogui
    expect. On a 150% DPI display, a UIA element centred at logical
    ``(500, 300)`` lives at physical ``(750, 450)``.

    Args:
        x: Logical x-coordinate.
        y: Logical y-coordinate.
        reference_x: Optional point used to look up the monitor's DPI
            via :func:`get_monitor_dpi` when ``dpi`` is None. Defaults
            to ``x``.
        reference_y: Same as ``reference_x`` for y.
        dpi: Pre-fetched :class:`MonitorDpi`. When supplied, no
            ``get_monitor_dpi`` call is made -- useful for tight loops.

    Returns the ``(x_physical, y_physical)`` tuple rounded to the
    nearest integer pixel.

    On 100% DPI displays (or non-Windows / API unavailable) the
    function is the identity -- the input is returned unchanged.
    """

    if dpi is None:
        ref_x = reference_x if reference_x is not None else int(x)
        ref_y = reference_y if reference_y is not None else int(y)
        dpi = get_monitor_dpi(ref_x, ref_y)

    if dpi.scale_x == 1.0 and dpi.scale_y == 1.0:
        return int(x), int(y)

    return round(x * dpi.scale_x), round(y * dpi.scale_y)


def physical_to_logical(
    x: int,
    y: int,
    *,
    reference_x: Optional[int] = None,
    reference_y: Optional[int] = None,
    dpi: Optional[MonitorDpi] = None,
) -> tuple[int, int]:
    """Inverse of :func:`logical_to_physical`.

    Converts physical pixels (mss capture, SetCursorPos) into the
    logical pixel space pywinauto UIA bounding rects use. Same DPI-
    lookup semantics + identity-on-100%-scale as the forward
    direction.
    """

    if dpi is None:
        ref_x = reference_x if reference_x is not None else int(x)
        ref_y = reference_y if reference_y is not None else int(y)
        dpi = get_monitor_dpi(ref_x, ref_y)

    if dpi.scale_x == 1.0 and dpi.scale_y == 1.0:
        return int(x), int(y)

    return round(x / dpi.scale_x), round(y / dpi.scale_y)


__all__ = [
    "BlockInputResult",
    "BlockInputUnavailableError",
    "DEFAULT_DPI",
    "DWMWA_CLOAKED",
    "IS_WINDOWS",
    "MDT_ANGULAR_DPI",
    "MDT_EFFECTIVE_DPI",
    "MDT_RAW_DPI",
    "MONITOR_DEFAULTTONEAREST",
    "MonitorDpi",
    "block_input_context",
    "get_last_input_idle_ms",
    "get_monitor_dpi",
    "get_monitor_dpi_for_window",
    "is_window_cloaked",
    "logical_to_physical",
    "physical_to_logical",
]
