"""Gaming mode (V1-spec gap A1).

Voice-triggered shutdown of OpenClaw plugins that anticheat systems
might flag (Vanguard, Easy Anti-Cheat, etc.) before the user launches a
protected game.

Engage flow:

  1. For each configured plugin slug, call
     :meth:`OpenClawClient.disable_plugin`. Best-effort: a single
     plugin's failure does not prevent the others from being processed.
  2. Optionally stop Docker Desktop (frees system RAM and removes
     any container-side processes that anticheat may flag). Disabled
     by default.
  3. Append a row to ``logs/gaming_mode.jsonl`` recording the engage
     timestamp + the plugin states.

Disengage reverses the cycle: enable each plugin, restart Docker if
configured, log the event.

The dispatcher's voice messages match the V1-spec phrasing:
``"Shutting down desktop control. Have fun."`` /
``"Full control restored."``

Critical contract: this module never raises into the orchestrator's
hot path. Plugin toggles are async; failures degrade to clear voice
messages so the user always knows what happened.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ultron.utils.logging import get_logger

logger = get_logger("openclaw_routing.gaming_mode")


class GamingModeError(RuntimeError):
    """Internal failure in the manager (audit-log write, etc.). Always
    caught at the dispatcher boundary so the voice loop never sees it."""


class GamingModeStatus(str, Enum):
    IDLE = "idle"
    ENGAGED = "engaged"
    TRANSITIONING = "transitioning"


@dataclass
class _PluginState:
    plugin_id: str
    success: bool
    error: Optional[str] = None


@dataclass
class GamingModeReport:
    """Outcome of an engage / disengage cycle."""

    status: GamingModeStatus
    action: str  # "engage" | "disengage" | "status"
    plugin_states: List[_PluginState] = field(default_factory=list)
    docker_acted: bool = False
    docker_error: Optional[str] = None
    note: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def all_plugin_actions_succeeded(self) -> bool:
        return all(p.success for p in self.plugin_states) if self.plugin_states else True


_GAMING_MODE_ACTIVE_LOCK = threading.Lock()
_GAMING_MODE_ACTIVE: bool = False


def is_gaming_mode_active() -> bool:
    """Process-global query for "is gaming mode currently engaged".

    2026-05-19 Track 6: the desktop automation primitives (Moondream2
    VLM, mss capture, pyautogui input, pywinauto UIA) consult this
    flag at their public-API entry points and short-circuit when
    True. The rationale (per the 2026-05-19 design conversation): a
    Python process with pyautogui resident is functionally
    indistinguishable from Discord + Voicemod from Vanguard's
    perspective, so the baseline is safe -- but the actual MITIGATION
    against the kernel-level behavioural fingerprinting is to ensure
    SendInput / SetCursorPos / etc. are NOT exercised while a
    Riot-protected game is running. This flag gates the call sites,
    not the imports (the modules stay importable; only their
    side-effects are blocked).

    Thread-safe read; the underlying flag is a module-level bool
    guarded by a lock. Fail-open: when in doubt, the flag returns
    False -- the gating is meant to PROTECT during gameplay, not to
    block desktop automation outside it.
    """
    with _GAMING_MODE_ACTIVE_LOCK:
        return _GAMING_MODE_ACTIVE


def set_gaming_mode_active(active: bool) -> None:
    """Update the process-global gaming-mode flag.

    Called by :class:`GamingModeManager` on engage / disengage. Test
    fixtures can call this directly to simulate the flag state
    without instantiating the full manager.
    """
    global _GAMING_MODE_ACTIVE
    with _GAMING_MODE_ACTIVE_LOCK:
        _GAMING_MODE_ACTIVE = bool(active)


class GamingModeManager:
    """Owns the engage/disengage state machine.

    Args:
        client: live :class:`OpenClawClient` for ``plugins enable/disable``.
            ``None`` short-circuits to "no client" voice messages so
            unit tests can construct without the OpenClaw stack.
        plugins_to_disable: ordered list of plugin slugs to toggle.
        toggle_docker: whether to stop/start Docker Desktop on
            engage/disengage.
        docker_executable_path: explicit path to ``Docker Desktop.exe``
            for restart. ``None`` falls back to PATH lookup.
        docker_process_name: the process name to terminate on engage
            (default ``"Docker Desktop"`` — Windows Task Manager name).
        log_path: where to write engage/disengage rows.
    """

    def __init__(
        self,
        *,
        client: Optional[Any] = None,
        plugins_to_disable: Optional[List[str]] = None,
        toggle_docker: bool = False,
        docker_executable_path: Optional[str] = None,
        docker_process_name: str = "Docker Desktop",
        log_path: Optional[Path] = None,
        on_engaged: Optional[Any] = None,
        on_disengaged: Optional[Any] = None,
    ) -> None:
        self.client = client
        self.plugins_to_disable = list(plugins_to_disable or [
            "desktop-control", "windows-control",
        ])
        self.toggle_docker = bool(toggle_docker)
        self.docker_executable_path = docker_executable_path
        self.docker_process_name = docker_process_name
        self.log_path = Path(log_path) if log_path else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._status: GamingModeStatus = GamingModeStatus.IDLE
        # Track which plugins we successfully disabled so disengage
        # doesn't try to enable plugins that weren't in scope.
        self._disabled_during_engage: List[str] = []
        self._docker_was_killed = False
        # Optional callables fired AFTER engage/disengage completes.
        # Used to flip per-component VRAM state (Kokoro device, etc.)
        # in sync with gaming mode. Both run in a try/except so a
        # callback failure cannot break the engage/disengage cycle.
        self._on_engaged = on_engaged
        self._on_disengaged = on_disengaged

    # --- public API ---------------------------------------------------------

    def status(self) -> GamingModeStatus:
        with self._lock:
            return self._status

    def _set_anticheat(self, active: bool) -> None:
        """Anticheat-safe mode is 100% TIED to gaming mode.

        Engaging gaming mode turns anticheat ON; disengaging turns it
        OFF. BOTH directions are unconditional -- anticheat is purely a
        function of gaming-mode state: it never defaults on and never
        turns on unless gaming mode is on, and it never lingers after
        gaming mode goes off. The enable direction is fail-safe (a
        kernel-anticheat game must never launch unprotected)."""
        try:
            from ultron.safety.anticheat import set_anticheat_active

            set_anticheat_active(
                active,
                "gaming mode engaged" if active else "gaming mode disengaged",
            )
        except Exception as e:                                       # noqa: BLE001
            logger.warning("anticheat tie-in failed: %s", e)

    async def engage(self) -> GamingModeReport:
        """Disable each configured plugin (and optionally Docker)."""
        with self._lock:
            if self._status == GamingModeStatus.ENGAGED:
                return GamingModeReport(
                    status=GamingModeStatus.ENGAGED,
                    action="engage",
                    note="already engaged",
                )
            self._status = GamingModeStatus.TRANSITIONING

        report = GamingModeReport(
            status=GamingModeStatus.TRANSITIONING, action="engage",
        )
        try:
            for slug in self.plugins_to_disable:
                state = await self._toggle_plugin(slug, "disable")
                report.plugin_states.append(state)
                if state.success:
                    self._disabled_during_engage.append(slug)
            if self.toggle_docker:
                acted, err = self._stop_docker()
                report.docker_acted = acted
                report.docker_error = err
                if acted:
                    self._docker_was_killed = True
        finally:
            with self._lock:
                self._status = GamingModeStatus.ENGAGED
            report.status = GamingModeStatus.ENGAGED
            # 2026-05-19 Track 6: flip the process-global flag so the
            # desktop primitives short-circuit. Done in ``finally`` so
            # even a partial-failure engage still gates the surface.
            set_gaming_mode_active(True)
            # 2026-06-11: anticheat-safe mode rides along with gaming
            # mode (config-gated, default ON) -- hard-blocks every
            # desktop-interaction surface while the game runs.
            self._set_anticheat(True)
            if self._on_engaged is not None:
                try:
                    self._on_engaged()
                except Exception as e:                            # noqa: BLE001
                    logger.warning(
                        "gaming_mode on_engaged callback failed: %s", e,
                    )

        self._write_log_row(report)
        return report

    async def disengage(self) -> GamingModeReport:
        """Re-enable each previously-disabled plugin."""
        with self._lock:
            if self._status == GamingModeStatus.IDLE:
                return GamingModeReport(
                    status=GamingModeStatus.IDLE,
                    action="disengage",
                    note="already idle",
                )
            self._status = GamingModeStatus.TRANSITIONING

        report = GamingModeReport(
            status=GamingModeStatus.TRANSITIONING, action="disengage",
        )
        try:
            # Restore everything we disabled, in the order we disabled.
            slugs = list(self._disabled_during_engage)
            for slug in slugs:
                state = await self._toggle_plugin(slug, "enable")
                report.plugin_states.append(state)
            self._disabled_during_engage = []
            if self.toggle_docker and self._docker_was_killed:
                acted, err = self._start_docker()
                report.docker_acted = acted
                report.docker_error = err
                if acted:
                    self._docker_was_killed = False
        finally:
            with self._lock:
                self._status = GamingModeStatus.IDLE
            report.status = GamingModeStatus.IDLE
            # Track 6: clear the process-global flag so the desktop
            # surface re-engages immediately on disengage.
            set_gaming_mode_active(False)
            self._set_anticheat(False)
            if self._on_disengaged is not None:
                try:
                    self._on_disengaged()
                except Exception as e:                            # noqa: BLE001
                    logger.warning(
                        "gaming_mode on_disengaged callback failed: %s", e,
                    )

        self._write_log_row(report)
        return report

    # --- helpers ------------------------------------------------------------

    async def _toggle_plugin(
        self, plugin_id: str, action: str,
    ) -> _PluginState:
        """Run enable/disable on a single slug. Always returns a state
        object -- never raises into the caller."""
        if self.client is None:
            return _PluginState(
                plugin_id=plugin_id, success=False,
                error="no OpenClaw client wired",
            )
        try:
            if action == "enable":
                result = await self.client.enable_plugin(plugin_id)
            elif action == "disable":
                result = await self.client.disable_plugin(plugin_id)
            else:
                return _PluginState(
                    plugin_id=plugin_id, success=False,
                    error=f"invalid action: {action!r}",
                )
        except Exception as e:                                       # noqa: BLE001
            logger.warning(
                "gaming_mode: plugins %s %s raised: %s",
                action, plugin_id, e,
            )
            return _PluginState(
                plugin_id=plugin_id, success=False, error=str(e)[:300],
            )
        return _PluginState(
            plugin_id=plugin_id,
            success=bool(getattr(result, "success", False)),
            error=getattr(result, "error", None),
        )

    def _stop_docker(self) -> tuple[bool, Optional[str]]:
        """Best-effort Docker Desktop kill. Windows-only; safe no-op
        elsewhere."""
        if sys.platform != "win32":
            return False, "docker toggle is windows-only"
        try:
            # taskkill /F /IM "Docker Desktop.exe"
            result = subprocess.run(  # noqa: S603 -- explicit args
                [
                    "taskkill", "/F", "/IM",
                    f"{self.docker_process_name}.exe",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=(subprocess.CREATE_NO_WINDOW
                               if hasattr(subprocess, "CREATE_NO_WINDOW") else 0),
            )
        except (OSError, subprocess.SubprocessError) as e:
            return False, str(e)[:200]
        if result.returncode == 0:
            return True, None
        # Returncode 128 = process not running; that's fine.
        if "not found" in (result.stderr or "").lower():
            return False, "docker was not running"
        return False, (result.stderr or "").strip()[:200] or "taskkill failed"

    def _start_docker(self) -> tuple[bool, Optional[str]]:
        if sys.platform != "win32":
            return False, "docker toggle is windows-only"
        executable = (
            self.docker_executable_path
            or r"C:\Program Files\Docker\Docker\Docker Desktop.exe"
        )
        if not Path(executable).exists():
            return False, f"docker executable not found at {executable}"
        try:
            # Spawn detached so we don't block on Docker boot.
            subprocess.Popen(  # noqa: S603 -- explicit args
                [executable],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
            )
        except (OSError, subprocess.SubprocessError) as e:
            return False, str(e)[:200]
        return True, None

    def _write_log_row(self, report: GamingModeReport) -> None:
        if self.log_path is None:
            return
        try:
            row = {
                "ts": datetime.fromtimestamp(
                    report.timestamp, timezone.utc,
                ).isoformat(),
                "action": report.action,
                "status": report.status.value,
                "plugin_states": [
                    {"id": p.plugin_id, "ok": p.success, "error": p.error}
                    for p in report.plugin_states
                ],
                "docker_acted": report.docker_acted,
                "docker_error": report.docker_error,
                "note": report.note,
            }
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            logger.debug("gaming_mode log write failed: %s", e)


__all__ = [
    "GamingModeError",
    "GamingModeStatus",
    "GamingModeReport",
    "GamingModeManager",
    "_PluginState",
]
