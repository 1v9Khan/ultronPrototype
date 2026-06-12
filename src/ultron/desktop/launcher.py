"""Application launcher -- launch known apps with monitor targeting.

Distinct from the OpenClaw ``browser`` plugin (Playwright-based,
isolated profile) -- this module launches the user's *actual* Chrome
with their *actual* profile. ``chrome.exe --new-window <URL>`` against
a running Chrome process simply opens a new window in the existing
process, preserving sign-ins, cookies, extensions, and history. If
Chrome isn't running, it starts fresh in the user's default profile.

Safety: every launch goes through :class:`ToolCallValidator`. The
existing Cap-2 rules already block:

- ``--remote-debugging-port=`` (CDP attach exposure)
- ``--load-extension=`` (extension injection)
- ``--user-data-dir=`` pointing outside standard profile dirs
- launches from ``Temp/`` ``Downloads/`` ``AppData/Local/Temp/``

The launcher's own contract: never spawn from an unverified path,
never pass debug flags by default, never set a custom user-data-dir.

Monitor targeting works by:

1. Snapshotting existing windows of the target process before launch.
2. Spawning the executable.
3. Polling for a new window owned by the spawned process (or its
   children -- Chrome's launched ``.exe`` exits immediately when
   the running instance handles ``--new-window``).
4. Calling :func:`move_window_to_monitor` on the new HWND.
"""

from __future__ import annotations

import os
import subprocess
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ultron.desktop.monitors import Monitor
from ultron.desktop.placement import (
    PlacementResult,
    focus_window,
    move_window_to_monitor,
)
from ultron.desktop.windows import enumerate_windows
from ultron.utils.logging import get_logger

logger = get_logger("desktop.launcher")


# ---------------------------------------------------------------------------
# Result + app registry types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaunchResult:
    """Outcome of a launch attempt.

    Attributes:
        success: True iff the executable spawned (Windows-visible
            window may or may not have appeared in time -- see ``hwnd``).
        app_name: name from the registry (``"chrome"`` etc.); empty
            for ad-hoc launches.
        exe_path: resolved executable path.
        pid: PID of the spawned process (may exit quickly for apps that
            relay to an existing instance, e.g. Chrome).
        hwnd: handle of the new window when it appeared in time. None
            when the launch succeeded but no window was detected within
            the wait timeout.
        monitor_index: monitor the window was placed on (when placement
            ran).
        placement: result of the placement call, when applicable.
        error: failure reason. Present iff ``success=False`` OR when
            ``success=True`` and follow-up steps (window-wait, placement)
            partially failed.
        window_appeared: None when no window wait was requested; True
            when the launched window was detected within the timeout;
            False when the wait timed out (placement and focus are
            skipped -- callers should voice this honestly instead of
            claiming the window is "on monitor N").
    """

    success: bool
    app_name: str = ""
    exe_path: Optional[Path] = None
    pid: Optional[int] = None
    hwnd: Optional[int] = None
    monitor_index: Optional[int] = None
    placement: Optional[PlacementResult] = None
    error: Optional[str] = None
    window_appeared: Optional[bool] = None


@dataclass(frozen=True)
class AppEntry:
    """One known application.

    Attributes:
        name: canonical name; matches case-insensitively against user
            requests.
        candidate_paths: paths to try in order. First existing path wins.
        args_prefix: arguments inserted before user-supplied args.
            Used for launchers that wrap the real exe (Discord's
            Update.exe ``--processStart``).
        aliases: alternate user-friendly names.
        process_name: exe name as it appears in ``psutil.Process.name()``
            (used to find the launched window when the original PID
            exits immediately -- Chrome / Discord pattern).
    """

    name: str
    candidate_paths: list[Path]
    args_prefix: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    process_name: str = ""


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------


def _home(*parts: str) -> Path:
    """Build a path under the user's home directory."""
    return Path(os.path.expandvars("%USERPROFILE%")).joinpath(*parts)


def _program_files(*parts: str) -> Path:
    pf = os.environ.get("ProgramFiles") or r"C:\Program Files"
    return Path(pf).joinpath(*parts)


def _program_files_x86(*parts: str) -> Path:
    pf = os.environ.get("ProgramFiles(x86)") or r"C:\Program Files (x86)"
    return Path(pf).joinpath(*parts)


def _localappdata(*parts: str) -> Path:
    lad = os.environ.get("LOCALAPPDATA") or str(_home("AppData", "Local"))
    return Path(lad).joinpath(*parts)


def _default_registry() -> list[AppEntry]:
    """The built-in app registry.

    Operators can extend via the desktop config (Phase 5 wiring); this
    list covers the common cases the user mentioned.
    """
    return [
        AppEntry(
            name="chrome",
            candidate_paths=[
                _program_files("Google", "Chrome", "Application", "chrome.exe"),
                _program_files_x86("Google", "Chrome", "Application", "chrome.exe"),
                _localappdata("Google", "Chrome", "Application", "chrome.exe"),
            ],
            aliases=("google chrome", "google-chrome"),
            process_name="chrome.exe",
        ),
        AppEntry(
            name="edge",
            candidate_paths=[
                _program_files("Microsoft", "Edge", "Application", "msedge.exe"),
                _program_files_x86("Microsoft", "Edge", "Application", "msedge.exe"),
            ],
            aliases=("microsoft edge", "msedge"),
            process_name="msedge.exe",
        ),
        AppEntry(
            name="firefox",
            candidate_paths=[
                _program_files("Mozilla Firefox", "firefox.exe"),
                _program_files_x86("Mozilla Firefox", "firefox.exe"),
            ],
            aliases=("mozilla", "mozilla firefox"),
            process_name="firefox.exe",
        ),
        AppEntry(
            name="cursor",
            candidate_paths=[
                _localappdata("Programs", "cursor", "Cursor.exe"),
                _program_files("Cursor", "Cursor.exe"),
            ],
            process_name="Cursor.exe",
        ),
        AppEntry(
            name="vscode",
            candidate_paths=[
                _localappdata("Programs", "Microsoft VS Code", "Code.exe"),
                _program_files("Microsoft VS Code", "Code.exe"),
            ],
            aliases=("vs code", "code", "visual studio code"),
            process_name="Code.exe",
        ),
        AppEntry(
            name="discord",
            candidate_paths=[
                # Discord's "stable" launcher; we resolve the latest
                # app-<version> below via _resolve_discord.
                _localappdata("Discord", "Update.exe"),
            ],
            args_prefix=("--processStart", "Discord.exe"),
            process_name="Discord.exe",
        ),
        AppEntry(
            name="notepad",
            candidate_paths=[
                Path(r"C:\Windows\System32\notepad.exe"),
                Path(r"C:\Windows\notepad.exe"),
            ],
            process_name="notepad.exe",
        ),
        AppEntry(
            name="explorer",
            candidate_paths=[Path(r"C:\Windows\explorer.exe")],
            aliases=("file explorer", "files", "windows explorer"),
            process_name="explorer.exe",
        ),
        AppEntry(
            name="terminal",
            candidate_paths=[
                _localappdata("Microsoft", "WindowsApps", "wt.exe"),
            ],
            aliases=("windows terminal", "wt"),
            process_name="WindowsTerminal.exe",
        ),
        AppEntry(
            name="spotify",
            candidate_paths=[
                # Microsoft Store install shim (most common on Win 11) --
                # 0-byte alias under %LOCALAPPDATA%\Microsoft\WindowsApps\
                # that the OS resolves to the real Spotify package under
                # %ProgramFiles%\WindowsApps\SpotifyAB.SpotifyMusic_*\.
                # Added 2026-05-14 after a live-session "Open Spotify"
                # failed with "no candidate path exists on disk" because
                # only the legacy install paths below were checked.
                _localappdata("Microsoft", "WindowsApps", "Spotify.exe"),
                # Legacy direct-install paths (older Spotify versions
                # and corporate installs).
                _home("AppData", "Roaming", "Spotify", "Spotify.exe"),
                _localappdata("Spotify", "Spotify.exe"),
            ],
            process_name="Spotify.exe",
        ),
        AppEntry(
            name="slack",
            candidate_paths=[
                _localappdata("slack", "slack.exe"),
            ],
            process_name="slack.exe",
        ),
        AppEntry(
            name="obs",
            candidate_paths=[
                _program_files("obs-studio", "bin", "64bit", "obs64.exe"),
            ],
            aliases=("obs studio",),
            process_name="obs64.exe",
        ),
    ]


def _resolve_chrome_exe(candidates: list[Path]) -> Optional[Path]:
    """Pick the first existing Chrome path, preferring the highest-version subdir
    when present (Chrome auto-updates and leaves multiple `Application/<ver>/`
    folders behind, but the canonical launcher is always
    ``Application/chrome.exe`` which Chrome itself updates to point at the
    latest binary).
    """
    for p in candidates:
        if p.exists():
            return p
    return None


def _resolve_discord_exe(update_exe: Path) -> Optional[Path]:
    """Resolve Discord's actual exe path under ``app-<version>/``.

    Returns the Update.exe (launcher) when present; the args_prefix
    field tells subprocess to relay to Discord.exe via the launcher.
    """
    if update_exe.exists():
        return update_exe
    # Fallback: pick the highest-versioned app-* directory and use its
    # Discord.exe directly.
    parent = update_exe.parent
    if parent.exists():
        candidates = sorted(
            (p for p in parent.glob("app-*") if p.is_dir()),
            key=lambda p: p.name,
            reverse=True,
        )
        for cand in candidates:
            exe = cand / "Discord.exe"
            if exe.exists():
                return exe
    return None


# ---------------------------------------------------------------------------
# Safety validator hook
# ---------------------------------------------------------------------------


def _validate_launch(
    exe_path: Path,
    args: list[str],
    *,
    user_text: str = "",
):
    """Run the safety validator on a launch attempt.

    Returns the :class:`ValidatorVerdict`. Fail-open on validator errors
    (errors-during-validation never block a launch; an unconfigured
    validator returns ALLOW via the no-op fallback).
    """
    try:
        from ultron.safety.validator import RuleContext, get_validator

        ctx = RuleContext(
            tool_name="desktop.launch_app",
            arguments={"path": str(exe_path), "argv": list(args)},
            capability="desktop_launcher",
            paths=(exe_path,) if exe_path else (),
            user_text=user_text,
        )
        return get_validator().check(ctx)
    except Exception as e:  # noqa: BLE001
        logger.debug("validator skipped: %s", e)
        # Use a minimal allow-shaped object via the actual class so the
        # rest of the launcher can treat the return value uniformly.
        from ultron.safety.validator import ValidatorVerdict, Verdict
        return ValidatorVerdict(
            verdict=Verdict.ALLOW, reason="validator unavailable",
        )


# ---------------------------------------------------------------------------
# AppLauncher
# ---------------------------------------------------------------------------


class AppLauncher:
    """Resolve app names + spawn processes + place windows on a chosen monitor."""

    def __init__(
        self,
        *,
        registry: Optional[list[AppEntry]] = None,
        window_wait_seconds: float = 5.0,
        poll_interval_seconds: float = 0.1,
    ) -> None:
        self.registry = registry if registry is not None else _default_registry()
        self.window_wait_seconds = float(window_wait_seconds)
        self.poll_interval_seconds = float(poll_interval_seconds)

    # -------- registry lookup --------

    def find_app(self, query: str) -> Optional[AppEntry]:
        """Look up an :class:`AppEntry` by name or alias (case-insensitive)."""
        q = (query or "").strip().lower()
        if not q:
            return None
        for entry in self.registry:
            if entry.name.lower() == q:
                return entry
            for alias in entry.aliases:
                if alias.lower() == q:
                    return entry
        # Substring fallback
        for entry in self.registry:
            if q in entry.name.lower():
                return entry
            for alias in entry.aliases:
                if q in alias.lower():
                    return entry
        return None

    def resolve_executable(self, entry: AppEntry) -> Optional[Path]:
        """Pick the first existing candidate path for an :class:`AppEntry`."""
        if entry.name == "discord":
            return _resolve_discord_exe(entry.candidate_paths[0])
        if entry.name == "chrome":
            return _resolve_chrome_exe(entry.candidate_paths)
        for p in entry.candidate_paths:
            if p.exists():
                return p
        return None

    # -------- spawning --------

    def launch_app(
        self,
        app_name: str,
        *,
        monitor: Optional[Monitor] = None,
        extra_args: Optional[list[str]] = None,
        fullscreen: bool = False,
        maximize: bool = False,
        wait_for_window: bool = True,
        user_text: str = "",
    ) -> LaunchResult:
        """Launch a registered app, optionally targeting a monitor.

        Args:
            app_name: registry name or alias.
            monitor: target monitor; when provided AND ``wait_for_window``,
                the launcher polls for the new window and moves it to
                this monitor.
            extra_args: additional CLI args (appended after ``args_prefix``).
            fullscreen: place the window to fully cover the monitor
                (preserving the window's chrome).
            maximize: ``ShowWindow(SW_MAXIMIZE)`` after placement.
                Mutually exclusive with ``fullscreen``.
            wait_for_window: poll for the launched window. False when
                the caller only needs the process to start.
            user_text: most recent user utterance, threaded into the
                safety validator's :class:`RuleContext`.
        """
        # Anticheat-safe mode: hard-blocked while the user is in game.
        from ultron.safety.anticheat import guard as _anticheat_guard
        _anticheat_guard('app_launch')
        entry = self.find_app(app_name)
        if entry is None:
            return LaunchResult(
                success=False, app_name=app_name,
                error=f"app '{app_name}' is not in the registry",
            )
        exe = self.resolve_executable(entry)
        if exe is None:
            return LaunchResult(
                success=False, app_name=entry.name,
                error=f"{entry.name}: no candidate path exists on disk",
            )

        args = list(entry.args_prefix) + list(extra_args or [])
        verdict = _validate_launch(exe, args, user_text=user_text)
        if not verdict.is_allowed:
            return LaunchResult(
                success=False, app_name=entry.name, exe_path=exe,
                error=f"safety: {verdict.reason}",
            )

        return self._spawn_and_place(
            exe=exe,
            args=args,
            app_entry=entry,
            monitor=monitor,
            fullscreen=fullscreen,
            maximize=maximize,
            wait_for_window=wait_for_window,
        )

    # -------- Chrome convenience --------

    def launch_chrome(
        self,
        *,
        url: str,
        monitor: Optional[Monitor] = None,
        fullscreen: bool = False,
        maximize: bool = False,
        window_size: Optional[tuple[int, int]] = None,
        new_window: bool = True,
        user_text: str = "",
    ) -> LaunchResult:
        """Launch Chrome (or open a new window in the running Chrome).

        ``new_window=True`` (default) passes ``--new-window`` so the URL
        opens in its own window (which we can then move to a monitor).
        Reuses the user's default Chrome profile -- no custom
        ``--user-data-dir`` -- so cookies, sign-ins, and extensions
        are preserved.

        For Chrome's actual fullscreen mode (kiosk-style, no chrome
        UI), set ``fullscreen=True`` AND pass ``--start-fullscreen``
        via :meth:`launch_app`'s ``extra_args``. The ``fullscreen`` arg
        here only controls window placement (fill the monitor as a
        regular window).
        """
        # Anticheat-safe mode: hard-blocked while the user is in game.
        from ultron.safety.anticheat import guard as _anticheat_guard
        _anticheat_guard('app_launch')
        entry = self.find_app("chrome")
        if entry is None:
            return LaunchResult(success=False, error="chrome not in registry")
        exe = self.resolve_executable(entry)
        if exe is None:
            return LaunchResult(
                success=False, app_name="chrome", exe_path=None,
                error="Chrome is not installed on this system",
            )

        args: list[str] = []
        if new_window:
            args.append("--new-window")
        if url:
            args.append(url)

        verdict = _validate_launch(exe, args, user_text=user_text)
        if not verdict.is_allowed:
            return LaunchResult(
                success=False, app_name="chrome", exe_path=exe,
                error=f"safety: {verdict.reason}",
            )

        return self._spawn_and_place(
            exe=exe,
            args=args,
            app_entry=entry,
            monitor=monitor,
            fullscreen=fullscreen,
            maximize=maximize,
            window_size=window_size,
            wait_for_window=monitor is not None,
        )

    def open_image_search(
        self,
        query: str,
        *,
        monitor: Optional[Monitor] = None,
        small_window: bool = True,
        user_text: str = "",
    ) -> LaunchResult:
        """Open Google Images for ``query`` in a new Chrome window.

        Convenience: "show me a picture of a golden retriever" routes
        here. Small-window default: 1024x768; set ``small_window=False``
        for a fullscreen-style window on the target monitor.
        """
        # Anticheat-safe mode: hard-blocked while the user is in game.
        from ultron.safety.anticheat import guard as _anticheat_guard
        _anticheat_guard('app_launch')
        q = (query or "").strip()
        if not q:
            return LaunchResult(success=False, error="empty image query")
        search_url = (
            "https://www.google.com/search?tbm=isch&q="
            + urllib.parse.quote_plus(q)
        )
        size = (1024, 768) if small_window else None
        return self.launch_chrome(
            url=search_url,
            monitor=monitor,
            fullscreen=False,
            maximize=False if small_window else True,
            window_size=size,
            user_text=user_text,
        )

    # -------- internal --------

    def _spawn_and_place(
        self,
        *,
        exe: Path,
        args: list[str],
        app_entry: AppEntry,
        monitor: Optional[Monitor],
        fullscreen: bool,
        maximize: bool,
        window_size: Optional[tuple[int, int]] = None,
        wait_for_window: bool,
    ) -> LaunchResult:
        # Snapshot existing windows of this process so we can detect the
        # new one when the original spawned process exits (Chrome /
        # Discord relay pattern).
        proc_name = (app_entry.process_name or exe.name).lower()
        pre_hwnds = {
            w.hwnd for w in enumerate_windows()
            if w.process_name.lower() == proc_name
        }

        try:
            # DETACHED_PROCESS keeps the spawned process independent of
            # Ultron's lifetime; CREATE_NEW_PROCESS_GROUP lets it be a
            # process-group root.
            creationflags = 0
            if hasattr(subprocess, "DETACHED_PROCESS"):
                creationflags |= subprocess.DETACHED_PROCESS
            if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
                creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP
            proc = subprocess.Popen(
                [str(exe)] + args,
                creationflags=creationflags,
                close_fds=True,
            )
        except Exception as e:  # noqa: BLE001
            return LaunchResult(
                success=False, app_name=app_entry.name, exe_path=exe,
                error=f"spawn failed: {e}",
            )

        pid = proc.pid

        if not wait_for_window:
            return LaunchResult(
                success=True, app_name=app_entry.name, exe_path=exe,
                pid=pid, window_appeared=None,
            )

        new_hwnd = self._wait_for_new_window(
            pre_hwnds=pre_hwnds, proc_name=proc_name,
        )
        if new_hwnd is None:
            return LaunchResult(
                success=True, app_name=app_entry.name, exe_path=exe,
                pid=pid,
                error="window did not appear within timeout",
                window_appeared=False,
            )

        if monitor is None:
            self._focus_fail_open(new_hwnd)
            return LaunchResult(
                success=True, app_name=app_entry.name, exe_path=exe,
                pid=pid, hwnd=new_hwnd, window_appeared=True,
            )

        placement = move_window_to_monitor(
            new_hwnd, monitor,
            fullscreen=fullscreen, maximize=maximize, size=window_size,
        )
        # 2026-06-12 bring-to-front fix: MoveWindow/ShowWindow do not
        # change Z-order, so a window opened by an already-running
        # process (the Chrome relay pattern) stayed BEHIND the current
        # foreground window. Focus regardless of placement success --
        # bringing the window forward is still desirable when only the
        # move failed.
        self._focus_fail_open(new_hwnd)
        return LaunchResult(
            success=True, app_name=app_entry.name, exe_path=exe,
            pid=pid, hwnd=new_hwnd,
            monitor_index=monitor.index,
            placement=placement,
            window_appeared=True,
        )

    def _focus_fail_open(self, hwnd: int) -> None:
        """Best-effort bring-to-front after launch/placement.

        Windows foreground-lock rules mean SetForegroundWindow from a
        background process may be refused; ``focus_window`` already
        falls back to BringWindowToTop. Never raises -- a focus
        failure (or an anticheat engage mid-launch raising
        AnticheatBlockedError inside focus_window's own guard) must
        not fail a successful launch.
        """
        try:
            focus_window(hwnd)
        except Exception as e:  # noqa: BLE001
            logger.debug("post-launch focus skipped hwnd=%d: %s", hwnd, e)

    def _wait_for_new_window(
        self,
        *,
        pre_hwnds: set[int],
        proc_name: str,
    ) -> Optional[int]:
        """Poll EnumWindows until a window with ``proc_name`` appears whose
        hwnd is not in ``pre_hwnds``. Returns hwnd or None on timeout.
        """
        deadline = time.monotonic() + self.window_wait_seconds
        while time.monotonic() < deadline:
            time.sleep(self.poll_interval_seconds)
            wins = [
                w for w in enumerate_windows()
                if w.process_name.lower() == proc_name
                and w.hwnd not in pre_hwnds
            ]
            if wins:
                # Prefer foreground if multiple appeared; else newest hwnd.
                wins.sort(key=lambda w: (w.is_foreground, w.hwnd), reverse=True)
                return wins[0].hwnd
        return None


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


_launcher_singleton: Optional[AppLauncher] = None


def get_app_launcher() -> AppLauncher:
    """Module-level singleton accessor."""
    global _launcher_singleton
    if _launcher_singleton is None:
        _launcher_singleton = AppLauncher()
    return _launcher_singleton


def set_app_launcher(launcher: Optional[AppLauncher]) -> None:
    """Test / orchestrator hook -- swap the singleton."""
    global _launcher_singleton
    _launcher_singleton = launcher


__all__ = [
    "AppEntry",
    "AppLauncher",
    "LaunchResult",
    "get_app_launcher",
    "set_app_launcher",
]
