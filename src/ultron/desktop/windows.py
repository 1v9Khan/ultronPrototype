"""Window enumeration + foreground detection via ``pywin32`` + ``psutil``.

Returns a :class:`WindowInfo` per visible top-level window with the
title, owning process name, and the index of the monitor it primarily
sits on. Use cases:

- "what app is the user looking at right now" -> :func:`get_foreground_window`
- "find the Chrome window on monitor 2" -> :func:`find_window`
- "list everything visible" -> :func:`enumerate_windows`
- "focus the window titled X, fall back to AppActivate when HWND is
  stale" -> :func:`focus_by_title` (catalog 07 T6)

Fail-open: any pywin32 / psutil exception per window degrades to skipping
that window rather than raising up to the caller.

2026 catalog 07 additions:

* ``enumerate_windows`` and ``find_window`` now filter out DWM-cloaked
  windows by default (``exclude_cloaked=True``). Cloaked windows are
  on inactive virtual desktops, hidden by UWP suspend, or offscreened
  by a compositor trick -- ``IsWindowVisible`` returns True for them,
  but the user can't see or interact with them. The new flag bridges
  to :func:`ultron.desktop.win32_helpers.is_window_cloaked`.

* :func:`focus_by_title` provides a title-substring focus with a
  primary ``SetForegroundWindow`` path and an AppActivate fallback for
  stale-HWND recovery. Uses pywin32's WScript.Shell COM dispatch when
  available, falls back to a PowerShell subprocess (with
  ``CREATE_NO_WINDOW``) when the COM dispatcher fails.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

import psutil  # type: ignore[import]
import win32con  # type: ignore[import]
import win32gui  # type: ignore[import]
import win32process  # type: ignore[import]

from ultron.desktop.monitors import Monitor, enumerate_monitors
from ultron.utils.logging import get_logger

logger = get_logger("desktop.windows")

# PowerShell subprocess fallback for AppActivate. CREATE_NO_WINDOW
# suppresses the brief console flash; we time the subprocess out so a
# wedged PowerShell never holds the orchestrator.
_APP_ACTIVATE_TIMEOUT_S: float = 2.0
_CREATE_NO_WINDOW = (
    0x08000000 if sys.platform == "win32" else 0
)  # subprocess.CREATE_NO_WINDOW


@dataclass(frozen=True)
class WindowInfo:
    """One top-level window.

    Attributes:
        hwnd: Win32 window handle.
        title: window title (post-Unicode normalisation; may be empty).
        class_name: Win32 window class name.
        process_name: owning process exe name (``chrome.exe``, ``Cursor.exe``);
            empty string when lookup fails.
        pid: owning process id; 0 when lookup fails.
        rect: (left, top, right, bottom) in virtual-screen coordinates.
        monitor_index: index of the monitor the window primarily sits on
            (greatest-overlap rule). None when the window is fully offscreen.
        is_minimized: True when iconic (minimized to taskbar).
        is_foreground: True when this window is currently the focused window.
    """

    hwnd: int
    title: str
    class_name: str
    process_name: str
    pid: int
    rect: tuple[int, int, int, int]
    monitor_index: Optional[int]
    is_minimized: bool
    is_foreground: bool

    @property
    def width(self) -> int:
        return max(0, self.rect[2] - self.rect[0])

    @property
    def height(self) -> int:
        return max(0, self.rect[3] - self.rect[1])

    @property
    def center(self) -> tuple[int, int]:
        l, t, r, b = self.rect
        return ((l + r) // 2, (t + b) // 2)


def _monitor_index_for_rect(
    rect: tuple[int, int, int, int],
    monitors: list[Monitor],
) -> Optional[int]:
    """Index of the monitor with the most overlap with rect.

    Returns None when there's no overlap with any monitor.
    """
    l, t, r, b = rect
    if r <= l or b <= t:
        return None
    best_idx: Optional[int] = None
    best_overlap = 0
    for m in monitors:
        ox = max(0, min(r, m.right) - max(l, m.x))
        oy = max(0, min(b, m.bottom) - max(t, m.y))
        overlap = ox * oy
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = m.index
    return best_idx


def _process_name_for_pid(pid: int) -> str:
    """Fetch the exe name for ``pid``; empty string on any failure."""
    if pid <= 0:
        return ""
    try:
        return psutil.Process(pid).name() or ""
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return ""
    except Exception as e:  # noqa: BLE001
        logger.debug("psutil lookup failed for pid %d: %s", pid, e)
        return ""


def _build_window_info(
    hwnd: int,
    monitors: list[Monitor],
    fg_hwnd: int,
) -> Optional[WindowInfo]:
    """Build a :class:`WindowInfo` for ``hwnd``; None when the window can't be inspected."""
    try:
        title = win32gui.GetWindowText(hwnd) or ""
        class_name = win32gui.GetClassName(hwnd) or ""
        rect = win32gui.GetWindowRect(hwnd)  # (l, t, r, b)
        is_min = bool(win32gui.IsIconic(hwnd))
    except Exception as e:  # noqa: BLE001
        logger.debug("window inspect failed hwnd=%d: %s", hwnd, e)
        return None

    try:
        _tid, pid = win32process.GetWindowThreadProcessId(hwnd)
    except Exception:  # noqa: BLE001
        pid = 0

    return WindowInfo(
        hwnd=int(hwnd),
        title=title,
        class_name=class_name,
        process_name=_process_name_for_pid(int(pid)),
        pid=int(pid),
        rect=(int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])),
        monitor_index=_monitor_index_for_rect(rect, monitors),
        is_minimized=is_min,
        is_foreground=(int(hwnd) == int(fg_hwnd)),
    )


def enumerate_windows(
    *,
    include_minimized: bool = False,
    include_invisible: bool = False,
    require_title: bool = True,
    exclude_cloaked: bool = True,
) -> list[WindowInfo]:
    """List top-level windows.

    Args:
        include_minimized: include windows iconic to the taskbar.
        include_invisible: include windows hidden via ``ShowWindow(SW_HIDE)``
            and tool windows that don't appear in the alt-tab list.
        require_title: skip windows with empty title text (Explorer's
            shell windows, hidden helper windows, etc.).
        exclude_cloaked: skip DWM-cloaked windows (default True, catalog 07
            T2 creative extension). ``IsWindowVisible`` returns True for
            cloaked windows because the style bits stay set; the desktop
            window manager hides them. Filtering them out improves the
            accuracy of "what's actually visible" lookups for
            foreground-window security checks and ``find_window``
            tiebreakers. Pass ``False`` to include cloaked windows (e.g.
            when explicitly enumerating windows across virtual desktops).

    Returns the visible windows, in arbitrary order. Sort externally
    if a particular order is needed (e.g. foreground first).
    """
    monitors = enumerate_monitors()
    try:
        fg = win32gui.GetForegroundWindow()
    except Exception:  # noqa: BLE001
        fg = 0

    results: list[WindowInfo] = []

    def _enum_cb(hwnd: int, _) -> bool:
        try:
            visible = bool(win32gui.IsWindowVisible(hwnd))
        except Exception:  # noqa: BLE001
            return True
        if not include_invisible and not visible:
            return True

        if exclude_cloaked:
            # Lazy-import so the win32_helpers module load (and its
            # ctypes setup) only happens when the cloak filter is
            # actually consulted.
            try:
                from ultron.desktop.win32_helpers import is_window_cloaked
                cloaked = is_window_cloaked(int(hwnd))
            except Exception:  # noqa: BLE001
                cloaked = None
            # Only EXCLUDE on a positive True; None / False keep the
            # window so the legacy IsWindowVisible behaviour is the
            # fallback when the cloak probe is unavailable.
            if cloaked is True:
                return True

        info = _build_window_info(hwnd, monitors, fg)
        if info is None:
            return True
        if require_title and not info.title.strip():
            return True
        if not include_minimized and info.is_minimized:
            return True
        results.append(info)
        return True

    try:
        win32gui.EnumWindows(_enum_cb, None)
    except Exception as e:  # noqa: BLE001
        logger.warning("EnumWindows failed: %s", e)

    return results


def get_foreground_window() -> Optional[WindowInfo]:
    """Return the currently focused window, or None when there isn't one."""
    monitors = enumerate_monitors()
    try:
        hwnd = win32gui.GetForegroundWindow()
    except Exception as e:  # noqa: BLE001
        logger.warning("GetForegroundWindow failed: %s", e)
        return None
    if not hwnd:
        return None
    info = _build_window_info(int(hwnd), monitors, int(hwnd))
    if info is None:
        return None
    return info


def find_window(
    query: str,
    *,
    prefer_foreground: bool = True,
    prefer_monitor: Optional[int] = None,
    by_process: bool = True,
    exclude_cloaked: bool = True,
) -> Optional[WindowInfo]:
    """Find a window whose title (and optionally process name) matches ``query``.

    Matching:

    - Case-insensitive substring against title.
    - When ``by_process`` is True, also matches case-insensitive
      substring against process name (so ``"chrome"`` finds
      ``chrome.exe``'s window even when the title is the page name).

    Tiebreakers, in order:

    1. Exact title match wins over substring.
    2. ``prefer_foreground=True`` prefers the foreground window.
    3. ``prefer_monitor`` (when set) prefers windows whose
       ``monitor_index`` matches.
    4. Most recently enumerated (z-order) wins last.

    ``exclude_cloaked`` (default True): forwarded to
    :func:`enumerate_windows` so cloaked / virtual-desktop-occluded
    windows are skipped at the source. Pass ``False`` to include them.
    """
    q = (query or "").strip().lower()
    if not q:
        return None

    candidates: list[WindowInfo] = []
    for w in enumerate_windows(exclude_cloaked=exclude_cloaked):
        title_lower = w.title.lower()
        proc_lower = w.process_name.lower()
        title_match = q in title_lower
        proc_match = by_process and q in proc_lower
        if title_match or proc_match:
            candidates.append(w)

    if not candidates:
        return None

    def _score(w: WindowInfo) -> tuple[int, int, int]:
        title_lower = w.title.lower()
        exact = 1 if title_lower == q else 0
        fg = 1 if (prefer_foreground and w.is_foreground) else 0
        mon = 1 if (prefer_monitor is not None and w.monitor_index == prefer_monitor) else 0
        return (exact, fg, mon)

    candidates.sort(key=_score, reverse=True)
    return candidates[0]


# ---------------------------------------------------------------------------
# T6: title-based focus with AppActivate fallback
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FocusResult:
    """Outcome of :func:`focus_by_title`.

    Attributes:
        success: True iff the target window now holds the foreground.
        window: :class:`WindowInfo` of the focused window when known
            (always set on success unless the focus changed via
            AppActivate and the foreground window can't be re-resolved).
        method: which path actually focused the window:
            ``"set_foreground_window"`` (primary path, HWND-based) or
            ``"app_activate_com"`` (WScript.Shell COM via pywin32) or
            ``"app_activate_powershell"`` (PowerShell subprocess
            fallback). Empty string when no method succeeded.
        error: short human-readable error string when ``success`` is
            False, otherwise None.
    """

    success: bool
    window: Optional[WindowInfo] = None
    method: str = ""
    error: Optional[str] = None


def _set_foreground_window(hwnd: int) -> bool:
    """Try ``SetForegroundWindow(hwnd)``. Returns True on success.

    Windows' foreground-lock rules can make this return False even with
    a valid HWND (e.g., another process has the foreground lock). The
    AppActivate path is the documented workaround for that case.
    """

    try:
        # SetForegroundWindow returns BOOL; 0 means failure.
        result = win32gui.SetForegroundWindow(int(hwnd))
        return bool(result)
    except Exception as exc:  # noqa: BLE001
        logger.debug("SetForegroundWindow(%d) failed: %s", hwnd, exc)
        return False


def _appactivate_via_wscript_shell(partial_title: str) -> bool:
    """Try the WScript.Shell COM ``AppActivate(title)`` path.

    Returns True iff a window was activated. Fail-open on import error
    or COM dispatch failure -- caller can fall through to the
    PowerShell subprocess path.
    """

    try:
        import win32com.client  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        logger.debug("win32com.client unavailable: %s", exc)
        return False

    try:
        shell = win32com.client.Dispatch("WScript.Shell")
        # AppActivate returns True / False; can also raise on COM
        # errors (com_error subclass).
        return bool(shell.AppActivate(partial_title))
    except Exception as exc:  # noqa: BLE001
        logger.debug("WScript.Shell AppActivate(%r) failed: %s", partial_title, exc)
        return False


def _appactivate_via_powershell(
    partial_title: str,
    *,
    timeout_s: float = _APP_ACTIVATE_TIMEOUT_S,
) -> bool:
    """Last-resort fallback: spawn PowerShell to invoke AppActivate.

    Used when ``win32com.client`` isn't available (e.g., minimal pywin32
    install). The subprocess is suppressed with ``CREATE_NO_WINDOW``
    and time-bounded so a wedged PowerShell never holds the
    orchestrator. Returns True on successful activation, False
    otherwise.
    """

    if sys.platform != "win32":
        return False

    # PowerShell single-quoted strings escape ' as ''.
    title_escaped = partial_title.replace("'", "''")
    ps_command = (
        "$ws = New-Object -ComObject WScript.Shell; "
        f"$result = $ws.AppActivate('{title_escaped}'); "
        "[Console]::Out.Write($result.ToString())"
    )
    try:
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                ps_command,
            ],
            capture_output=True,
            text=True,
            timeout=max(0.1, float(timeout_s)),
            creationflags=_CREATE_NO_WINDOW,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.debug("PowerShell AppActivate(%r) failed: %s", partial_title, exc)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.debug("PowerShell AppActivate(%r) raised: %s", partial_title, exc)
        return False

    stdout = (completed.stdout or "").strip().lower()
    return stdout == "true"


def focus_by_title(
    partial_title: str,
    *,
    prefer_monitor: Optional[int] = None,
    fall_back_to_app_activate: bool = True,
    timeout_s: float = _APP_ACTIVATE_TIMEOUT_S,
) -> FocusResult:
    """Focus a window by title substring (catalog 07 T6).

    Two-tier resolution:

    1. **Primary path** -- :func:`find_window` resolves the HWND, then
       :func:`_set_foreground_window` attempts ``SetForegroundWindow``.
       This is the fast path and works for almost every case.

    2. **AppActivate fallback** -- when the HWND is stale (window was
       recreated at the same title), or Windows' foreground-lock
       prevents ``SetForegroundWindow`` from succeeding, the title-
       substring AppActivate path takes over. Two sub-fallbacks:

       * ``win32com.client.Dispatch("WScript.Shell").AppActivate(title)``
         (in-process, no subprocess).
       * PowerShell subprocess invoking the same COM (used when
         ``win32com.client`` isn't available; CREATE_NO_WINDOW +
         time-bounded).

    Args:
        partial_title: Case-insensitive substring against the target
            window title. AppActivate matches the full title
            substring; the primary path also matches process name.
        prefer_monitor: Forward to :func:`find_window` as a tiebreaker
            when multiple windows match the substring.
        fall_back_to_app_activate: When False, the AppActivate
            fallbacks are skipped (caller wants HWND-only semantics).
        timeout_s: Wall-clock timeout for the PowerShell subprocess
            fallback. The COM path has no separate timeout knob.

    Returns:
        :class:`FocusResult` describing which method succeeded.
    """
    # Anticheat-safe mode: hard-blocked while the user is in game.
    from ultron.safety.anticheat import guard as _anticheat_guard
    _anticheat_guard('window_focus')

    title = (partial_title or "").strip()
    if not title:
        return FocusResult(
            success=False,
            error="empty title",
        )

    # ---- Primary path ----
    candidate = find_window(title, prefer_monitor=prefer_monitor)
    if candidate is not None:
        if _set_foreground_window(candidate.hwnd):
            return FocusResult(
                success=True,
                window=candidate,
                method="set_foreground_window",
            )
        logger.debug(
            "primary SetForegroundWindow path failed for hwnd=%d title=%r; "
            "attempting AppActivate fallback",
            candidate.hwnd,
            candidate.title,
        )

    if not fall_back_to_app_activate:
        return FocusResult(
            success=False,
            window=candidate,
            error=(
                f"no window matching '{title}'"
                if candidate is None
                else f"SetForegroundWindow refused (hwnd={candidate.hwnd})"
            ),
        )

    # ---- AppActivate fallbacks ----
    if _appactivate_via_wscript_shell(title):
        focused = get_foreground_window()
        return FocusResult(
            success=True,
            window=focused,
            method="app_activate_com",
        )

    if _appactivate_via_powershell(title, timeout_s=timeout_s):
        focused = get_foreground_window()
        return FocusResult(
            success=True,
            window=focused,
            method="app_activate_powershell",
        )

    return FocusResult(
        success=False,
        window=candidate,
        error=(
            f"no AppActivate fallback succeeded for '{title}'"
        ),
    )


# ---------------------------------------------------------------------------
# Catalog 08 T4: wait-for-window primitive
# ---------------------------------------------------------------------------


# Defaults mirror the upstream clawhub-windows-control ``wait_for_window``
# script: 30 s total timeout, 500 ms poll interval. The two-tuple of
# constants matches the equivalents in :mod:`ultron.desktop.uia` so the
# wait family has consistent defaults across the desktop surface.
DEFAULT_WAIT_TIMEOUT_S: float = 30.0
DEFAULT_WAIT_INTERVAL_S: float = 0.5


def wait_for_window(
    partial_title: str,
    *,
    timeout_s: float = DEFAULT_WAIT_TIMEOUT_S,
    interval_s: float = DEFAULT_WAIT_INTERVAL_S,
    by_process: bool = True,
    exclude_cloaked: bool = True,
    prefer_monitor: Optional[int] = None,
    sleep_fn: Optional[object] = None,
    clock_fn: Optional[object] = None,
) -> Optional[WindowInfo]:
    """Poll until a window matching ``partial_title`` appears.

    Catalog 08 T4 (GREEN, read-only). The synchronous companion to
    :func:`find_window` for "wait until this window opens" automation
    barriers (post-app-launch settle, post-dialog-trigger wait, etc.).
    Each poll iteration delegates to :func:`find_window` so the
    matching semantics (substring against title and optionally process
    name, exact-match tiebreaker, foreground / monitor preference) are
    identical.

    Args:
        partial_title: case-insensitive substring match against window
            title. Forwarded to :func:`find_window`.
        timeout_s: wall-clock timeout in seconds.
        interval_s: poll interval in seconds.
        by_process: forwarded to :func:`find_window` for substring
            match against process name as well.
        exclude_cloaked: forwarded to :func:`find_window` so cloaked /
            virtual-desktop-occluded windows are skipped.
        prefer_monitor: tiebreaker forwarded to :func:`find_window`
            when multiple windows match.
        sleep_fn: optional ``(float) -> None`` injection for tests so
            the polling loop doesn't actually sleep. Defaults to
            :func:`time.sleep`.
        clock_fn: optional ``() -> float`` injection for tests so the
            deadline computation is deterministic. Defaults to
            :func:`time.monotonic`.

    Returns:
        The matched :class:`WindowInfo` on first hit, or ``None`` on
        timeout. Empty ``partial_title`` returns None without polling
        (matches the upstream guard).

    Fail-open: any :func:`find_window` exception logs DEBUG and the
    loop retries on the next iteration.
    """

    title = (partial_title or "").strip()
    if not title:
        return None
    if timeout_s <= 0:
        return None

    sleeper = sleep_fn if callable(sleep_fn) else time.sleep
    clock = clock_fn if callable(clock_fn) else time.monotonic

    deadline = clock() + float(timeout_s)
    poll_interval = max(0.01, float(interval_s))

    while True:
        try:
            match = find_window(
                query=title,
                prefer_foreground=False,
                prefer_monitor=prefer_monitor,
                by_process=by_process,
                exclude_cloaked=exclude_cloaked,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("wait_for_window find_window raised: %s", exc)
            match = None
        if match is not None:
            return match

        now = clock()
        if now >= deadline:
            return None
        remaining = deadline - now
        sleeper(min(poll_interval, remaining))


# ---------------------------------------------------------------------------
# Catalog 08 T6: get_active_window_title + graceful close_window
# ---------------------------------------------------------------------------


# Editor convention: many editors prepend ``*`` to the title bar when
# the document has unsaved changes (Notepad, VS Code, Sublime, GIMP,
# Office apps). The upstream catalog flags this as the "are you about
# to lose data" heuristic for the close_window safety gate.
UNSAVED_CHANGES_TITLE_HINTS: tuple[str, ...] = (
    "*",
    "[modified]",
    "(modified)",
    "● ",  # VS Code dot
    " — modified",
)


@dataclass(frozen=True)
class CloseWindowResult:
    """Outcome of :func:`close_window`."""

    success: bool
    window: Optional[WindowInfo] = None
    method: str = ""  # "wm_close" / "pywinauto_close" / "kill_tree" / ""
    suspected_unsaved: bool = False
    error: Optional[str] = None


def get_active_window_title() -> Optional[str]:
    """Return the title of the currently focused window.

    Catalog 08 T6 (GREEN, read-only). The lightweight companion to
    :func:`get_foreground_window` -- returns only the title string,
    or ``None`` when no window is focused or the lookup fails.

    Useful as a cheap probe for "what is the user looking at" voice
    queries without paying for the full :class:`WindowInfo`
    construction (no psutil process-name lookup, no monitor-index
    computation, no rect enumeration).
    """

    try:
        hwnd = win32gui.GetForegroundWindow()
    except Exception as exc:  # noqa: BLE001
        logger.debug("GetForegroundWindow failed: %s", exc)
        return None
    if not hwnd:
        return None
    try:
        title = win32gui.GetWindowText(int(hwnd)) or ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("GetWindowText hwnd=%d failed: %s", hwnd, exc)
        return None
    return title or None


def _title_suggests_unsaved_changes(title: str) -> bool:
    """True iff ``title`` matches an "unsaved document" editor convention."""

    if not title:
        return False
    title_lower = title.lower()
    for hint in UNSAVED_CHANGES_TITLE_HINTS:
        if hint.lower() in title_lower:
            return True
    return False


def _validate_close_window(
    *,
    window: WindowInfo,
    user_text: str,
    suspected_unsaved: bool,
) -> object:
    """Run the runtime tool-call validator against a close-window action."""

    try:
        from ultron.safety.validator import RuleContext, get_validator

        ctx = RuleContext(
            tool_name="desktop.window.close",
            arguments={
                "window_title": window.title,
                "process_name": window.process_name,
                "hwnd": int(window.hwnd),
                "suspected_unsaved": bool(suspected_unsaved),
            },
            capability="desktop_window_close",
            user_text=user_text,
        )
        return get_validator().check(ctx)
    except Exception as exc:  # noqa: BLE001
        logger.debug("close_window validator skipped: %s", exc)
        from ultron.safety.validator import ValidatorVerdict, Verdict
        return ValidatorVerdict(
            verdict=Verdict.ALLOW, reason="validator unavailable",
        )


def _post_wm_close(hwnd: int) -> bool:
    """Send a graceful ``WM_CLOSE`` to ``hwnd``. True on success."""

    try:
        win32gui.PostMessage(int(hwnd), win32con.WM_CLOSE, 0, 0)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("WM_CLOSE post hwnd=%d failed: %s", hwnd, exc)
        return False


def _force_kill_window_process(window: WindowInfo) -> tuple[bool, str]:
    """Force-terminate the process owning ``window`` via kill_process_tree."""

    pid = int(window.pid or 0)
    if pid <= 0:
        return False, "no pid recorded for window"
    try:
        from ultron.subprocess.kill_tree import kill_process_tree
    except Exception as exc:  # noqa: BLE001
        return False, f"kill_process_tree unavailable: {exc}"
    try:
        result = kill_process_tree(pid)
    except Exception as exc:  # noqa: BLE001
        return False, f"kill_process_tree raised: {exc}"
    ok = bool(getattr(result, "terminated", 0)) or bool(
        getattr(result, "force_killed", 0)
    )
    if not ok:
        return False, "kill_process_tree reported no terminations"
    return True, ""


def close_window(
    partial_title: str,
    *,
    force: bool = False,
    user_text: str = "",
    prefer_monitor: Optional[int] = None,
    exclude_cloaked: bool = True,
) -> CloseWindowResult:
    """Close a window by title, gracefully by default.

    Catalog 08 T6 (YELLOW). The "graceful" path sends ``WM_CLOSE`` via
    :func:`win32gui.PostMessage` -- this triggers the app's own close
    hook, so apps with unsaved changes will prompt the user. The
    "force" path falls through to
    :func:`ultron.subprocess.kill_tree.kill_process_tree` which is the
    same primitive ultron uses for Parakeet / XTTS shutdown.

    Args:
        partial_title: case-insensitive substring match against the
            target window's title. Resolves via :func:`find_window`.
        force: when True, skip the graceful ``WM_CLOSE`` step and
            terminate the owning process tree. When False (default),
            only the graceful path is tried; apps with save-prompts
            will surface them.
        user_text: forwarded to the safety validator so the Cap-3
            explicit-intent matcher can verify the user actually
            asked for the close.
        prefer_monitor: tiebreaker forwarded to :func:`find_window`.
        exclude_cloaked: forwarded to :func:`find_window`.

    Returns:
        :class:`CloseWindowResult`. ``suspected_unsaved`` reflects
        whether the title matched :data:`UNSAVED_CHANGES_TITLE_HINTS`;
        callers can use this to decide whether to gate the close
        behind a two-phase voice confirmation.
    """
    # Anticheat-safe mode: hard-blocked while the user is in game.
    from ultron.safety.anticheat import guard as _anticheat_guard
    _anticheat_guard('window_close')

    title = (partial_title or "").strip()
    if not title:
        return CloseWindowResult(
            success=False, error="empty title",
        )

    candidate = find_window(
        query=title,
        prefer_foreground=False,
        prefer_monitor=prefer_monitor,
        exclude_cloaked=exclude_cloaked,
    )
    if candidate is None:
        return CloseWindowResult(
            success=False,
            error=f"no window matching '{title}'",
        )

    suspected_unsaved = _title_suggests_unsaved_changes(candidate.title)

    verdict = _validate_close_window(
        window=candidate,
        user_text=user_text,
        suspected_unsaved=suspected_unsaved,
    )
    if not verdict.is_allowed:
        return CloseWindowResult(
            success=False,
            window=candidate,
            suspected_unsaved=suspected_unsaved,
            error=f"safety: {verdict.reason}",
        )

    if force:
        ok, err = _force_kill_window_process(candidate)
        if ok:
            return CloseWindowResult(
                success=True,
                window=candidate,
                method="kill_tree",
                suspected_unsaved=suspected_unsaved,
            )
        return CloseWindowResult(
            success=False,
            window=candidate,
            suspected_unsaved=suspected_unsaved,
            error=err or "force-kill failed",
        )

    if _post_wm_close(int(candidate.hwnd)):
        return CloseWindowResult(
            success=True,
            window=candidate,
            method="wm_close",
            suspected_unsaved=suspected_unsaved,
        )

    return CloseWindowResult(
        success=False,
        window=candidate,
        suspected_unsaved=suspected_unsaved,
        error="WM_CLOSE post failed; pass force=True to escalate",
    )
