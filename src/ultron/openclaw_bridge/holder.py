"""Bridge holder owned by the orchestrator (Phase 3.5).

Encapsulates all Phase 3 bridge components — lifecycle, client,
workspace writer, event receiver, MCP registrar — plus the fail-open
startup sequence and a background retry thread.

Wiring contract: the orchestrator constructs an :class:`OpenClawBridge`
in its ``__init__`` and calls :meth:`start` once, then calls
:meth:`shutdown` from its existing ``shutdown()`` method. The voice
pipeline NEVER touches this holder — bridge calls go through
``self.openclaw_bridge.client`` only when an OpenClaw-bound intent
fires.

Fail-open semantics:

- Construction never blocks on the Gateway. CLI discovery failure
  produces ``client=None`` rather than raising.
- :meth:`start` runs a fast (≤2 s) Gateway probe. When reachable,
  attempts synchronous MCP registration. When unreachable or
  registration fails, schedules a daemon-thread retry loop.
- :meth:`shutdown` stops the retry thread and the event receiver.
  We deliberately do NOT unregister the MCP entry — leaving it lets
  OpenClaw spawn Ultron's MCP on demand across Ultron restarts.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ultron.errors import OpenClawGatewayError
from ultron.openclaw_bridge.client import OpenClawClient
from ultron.openclaw_bridge.events import OpenClawEventReceiver
from ultron.openclaw_bridge.heartbeat_alerts import (
    HeartbeatAlert,
    HeartbeatAlertLog,
)
from ultron.openclaw_bridge.lifecycle import OpenClawLifecycle
from ultron.openclaw_bridge.mcp_registration import (
    RegistrationResult,
    UltronMcpRegistrar,
)
from ultron.openclaw_bridge.notifications import NotificationDispatcher
from ultron.openclaw_bridge.persona import default_workspace_dir
from ultron.openclaw_bridge.workspace import WorkspaceWriter
from ultron.utils.logging import get_logger

logger = get_logger("openclaw_bridge.holder")


@dataclass
class OpenClawBridge:
    """Container for Phase 3 bridge components.

    Constructed once by the orchestrator. Public attributes are read
    by callers (the dispatcher, the coding-completion narrator, etc.)
    on demand. Lifecycle is owned by :meth:`start` / :meth:`shutdown`.
    """

    lifecycle: OpenClawLifecycle
    client: Optional[OpenClawClient]                 # None when CLI cannot be discovered
    workspace: WorkspaceWriter
    events: OpenClawEventReceiver
    registrar: Optional[UltronMcpRegistrar]          # None when client is None or registration disabled
    notifications: NotificationDispatcher            # Phase 4 — proactive Telegram pings
    heartbeat_alerts: HeartbeatAlertLog              # Phase 5 — local alert log

    # Internal startup/lifecycle state
    _retry_interval_s: float = 60.0
    _auto_notify_heartbeat: bool = True              # Phase 5 config knob
    _retry_thread: Optional[threading.Thread] = field(default=None, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _started: bool = field(default=False, init=False)

    @classmethod
    def from_config(
        cls,
        openclaw_cfg,
        *,
        notifications_cfg=None,
        heartbeat_cfg=None,
    ) -> Optional["OpenClawBridge"]:
        """Build a bridge from the loaded config, or return None when
        ``openclaw.enabled`` is False.

        Args:
            openclaw_cfg: the loaded :class:`OpenClawConfig`.
            notifications_cfg: the loaded :class:`NotificationsConfig`.
                Optional — when omitted, a default-disabled instance
                is created (preserves backward compat with callers
                from before Phase 4).
            heartbeat_cfg: the loaded :class:`HeartbeatConfig`. Optional
                — defaults to a fresh instance with the standard
                ``logs/heartbeat_alerts.jsonl`` path (Phase 5).

        Construction is forgiving: CLI discovery failure produces
        ``client=None`` (callers degrade), but the rest of the bridge
        (workspace writer, event receiver, lifecycle, alert log) is
        always built.
        """
        if not openclaw_cfg.enabled:
            return None
        bridge_cfg = openclaw_cfg.bridge
        if notifications_cfg is None:
            from ultron.config import NotificationsConfig
            notifications_cfg = NotificationsConfig()
        if heartbeat_cfg is None:
            from ultron.config import HeartbeatConfig
            heartbeat_cfg = HeartbeatConfig()

        # Lifecycle — never raises; uses HTTP probe to existing canvas path.
        lifecycle = OpenClawLifecycle(
            gateway_url=openclaw_cfg.gateway_url,
        )

        # Client — depends on locating the openclaw CLI executable.
        client: Optional[OpenClawClient]
        try:
            client = OpenClawClient(
                cli_path=bridge_cfg.cli_path,
                default_timeout_s=bridge_cfg.cli_timeout_seconds,
                default_agent_id=openclaw_cfg.required_agent_id or "ultron-main",
            )
        except OpenClawGatewayError as e:
            logger.warning(
                "OpenClaw CLI not found (%s); bridge will operate with "
                "client=None — outbound tool/message ops disabled.",
                e,
            )
            client = None

        # Workspace writer — pure file IO; always succeeds.
        workspace_dir = (
            Path(bridge_cfg.workspace_dir)
            if bridge_cfg.workspace_dir
            else default_workspace_dir()
        )
        workspace = WorkspaceWriter(
            workspace_dir,
            lock_timeout_s=bridge_cfg.workspace_lock_timeout_seconds,
        )

        # Event receiver — gated off by default. start() is a no-op
        # when disabled, so wiring is safe even when the user hasn't
        # opted in to voice handoff.
        events = OpenClawEventReceiver(
            prefix=bridge_cfg.inbound_voice_handoff_prefix,
            enabled=bridge_cfg.inbound_voice_handoff_enabled,
        )

        # MCP registrar — conditional on (client present) AND
        # (mcp_server_command set). The "auto" sentinel resolves to
        # the canonical stdio entry script so out-of-the-box the
        # registrar registers Ultron's MCP without operator action.
        registrar: Optional[UltronMcpRegistrar] = None
        if client is not None:
            command, args = cls._resolve_mcp_command(
                bridge_cfg.mcp_server_command,
                list(bridge_cfg.mcp_server_args),
            )
            registrar = UltronMcpRegistrar(
                client,
                name=bridge_cfg.mcp_server_name,
                command=command,
                args=args,
            )

        # Notifications dispatcher (Phase 4). Built unconditionally —
        # the dispatcher itself fails open if client is None or the
        # config flags are off, so passing it through is harmless.
        notifications = NotificationDispatcher(
            client,
            notifications_cfg,
            timeout_s=bridge_cfg.message_send_timeout_seconds,
        )

        # Heartbeat alert log (Phase 5). Path resolves against the
        # project root for relative values. The log is created lazily
        # on first record.
        from ultron.config import resolve_path
        alert_log = HeartbeatAlertLog(
            resolve_path(heartbeat_cfg.alert_log_path),
            retention_days=heartbeat_cfg.alert_retention_days,
        )

        return cls(
            lifecycle=lifecycle,
            client=client,
            workspace=workspace,
            events=events,
            registrar=registrar,
            notifications=notifications,
            heartbeat_alerts=alert_log,
            _retry_interval_s=bridge_cfg.retry_registration_interval_seconds,
            _auto_notify_heartbeat=heartbeat_cfg.auto_notify_telegram,
        )

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def start(self) -> None:
        """Probe the Gateway and attempt one-shot MCP registration.

        Sync wrapper around the async :meth:`registrar.register`. When
        the Gateway is unreachable or registration fails, a daemon
        retry thread is launched. Subsequent calls are no-ops.
        """
        if self._started:
            return
        self._started = True
        reachable = False
        try:
            reachable = self.lifecycle.is_reachable()
        except Exception as exc:                                # noqa: BLE001
            logger.warning(
                "OpenClaw lifecycle probe raised (%s); treating as unreachable",
                exc,
            )
        if reachable:
            logger.info(
                "OpenClaw Gateway reachable at %s",
                self.lifecycle.gateway_url,
            )
        else:
            logger.warning(
                "OpenClaw Gateway not reachable at %s — bridge running in "
                "fail-open mode (voice path unaffected). Will retry MCP "
                "registration every %.0fs in the background.",
                self.lifecycle.gateway_url,
                self._retry_interval_s,
            )

        # Bring up the event receiver. start() is a no-op when disabled;
        # safe regardless of Gateway reachability since Phase 3 has no
        # transport hooked up yet.
        try:
            self._run_async(self.events.start())
        except Exception as exc:                                # noqa: BLE001
            logger.warning(
                "voice-handoff receiver failed to start (%s) — disabling",
                exc,
            )

        if self.registrar is None or not self.registrar.is_configured:
            logger.info(
                "MCP registration deferred (no stdio command configured)",
            )
            return

        if reachable:
            result = self._run_async(self.registrar.register())
            if not result.registered:
                logger.warning(
                    "initial MCP registration failed (%s); scheduling retries",
                    result.error or result.skipped_reason,
                )
                self._launch_retry_thread()
            else:
                logger.info(
                    "MCP entry %r registered (idempotent=%s)",
                    result.name, result.already_registered,
                )
        else:
            self._launch_retry_thread()

    def shutdown(self) -> None:
        """Stop the retry thread and the event receiver. Idempotent.

        We keep the MCP registration in place so OpenClaw can re-spawn
        Ultron's MCP across restarts; explicit unregistration is the
        caller's job (e.g. uninstall script)."""
        self._stop_event.set()
        if self._retry_thread is not None and self._retry_thread.is_alive():
            self._retry_thread.join(timeout=2.0)
        try:
            self._run_async(self.events.stop())
        except Exception:
            # Stopping the receiver should never raise; swallow defensively.
            pass

    def record_heartbeat_alert(
        self,
        text: str,
        *,
        source: str = "heartbeat",
        severity: str = "info",
        metadata: Optional[dict] = None,
    ) -> HeartbeatAlert:
        """Record a heartbeat alert and (optionally) push it to
        Telegram.

        The Telegram push respects:

        1. ``heartbeat.auto_notify_telegram`` (this flag) — master.
        2. ``notifications.telegram.enabled`` (NotificationDispatcher) — channel master.
        3. ``notifications.telegram.notify_on.heartbeat_alerts`` — per-event.

        Returns the recorded alert regardless of Telegram outcome.
        Telegram failures are swallowed inside the dispatcher.

        Phase 5 entry point. Called by the OpenClaw-side heartbeat
        agent via the (future) MCP tool, or directly by Ultron-side
        diagnostics that want to surface something to the user.
        """
        alert = self.heartbeat_alerts.record(
            text, source=source, severity=severity, metadata=metadata,
        )
        if self._auto_notify_heartbeat and self.notifications is not None:
            self.fire_and_forget(
                lambda: self.notifications.notify_heartbeat_alert(alert.text),
            )
        return alert

    def fire_and_forget(self, coro_factory) -> None:
        """Schedule an async coroutine to run on a daemon thread,
        not blocking the caller.

        Used by the orchestrator's voice-loop hooks (e.g. coding-task
        completion) to fire a Telegram notification without waiting on
        Telegram's round-trip. ``coro_factory`` is a zero-arg callable
        that returns a fresh coroutine — we accept the factory rather
        than the coroutine itself so creating the coroutine and
        running it both happen on the worker thread (safer when the
        underlying client uses async primitives that bind to a
        specific loop).

        Failures inside the coroutine are logged at WARN and never
        propagate — this is fire-and-forget by design.
        """
        def _runner() -> None:
            try:
                asyncio.run(coro_factory())
            except Exception as exc:                            # noqa: BLE001
                logger.warning(
                    "fire_and_forget coroutine raised: %s", exc,
                )

        thread = threading.Thread(
            target=_runner, name="openclaw-notify", daemon=True,
        )
        thread.start()

    # -------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------

    def _launch_retry_thread(self) -> None:
        """Start a daemon thread that retries MCP registration on a loop.

        The thread runs until either (a) registration succeeds, or
        (b) :attr:`_stop_event` is set (shutdown)."""
        if self.registrar is None:
            return
        if self._retry_thread is not None and self._retry_thread.is_alive():
            return

        thread = threading.Thread(
            target=self._retry_loop_thread,
            name="openclaw-mcp-retry",
            daemon=True,
        )
        thread.start()
        self._retry_thread = thread
        logger.info(
            "MCP registration retry thread started "
            "(interval=%.0fs)",
            self._retry_interval_s,
        )

    def _retry_loop_thread(self) -> None:
        """Daemon-thread body. Retries registration until success or stop."""
        registrar = self.registrar
        if registrar is None:
            return
        attempt = 0
        while not self._stop_event.is_set():
            attempt += 1
            try:
                result = self._run_async(registrar.register())
            except Exception as exc:                            # noqa: BLE001
                logger.warning(
                    "MCP retry attempt %d raised (%s); will retry in %.0fs",
                    attempt, exc, self._retry_interval_s,
                )
            else:
                if result.registered:
                    logger.info(
                        "MCP registration succeeded on retry attempt %d "
                        "(idempotent=%s)",
                        attempt, result.already_registered,
                    )
                    return
                logger.debug(
                    "MCP retry attempt %d still failing (%s)",
                    attempt, result.error or result.skipped_reason,
                )
            # Wake early on shutdown — Event.wait() returns True when set.
            if self._stop_event.wait(timeout=self._retry_interval_s):
                return

    @staticmethod
    def _resolve_mcp_command(
        configured: Optional[str],
        configured_args: list,
    ) -> tuple[Optional[str], list]:
        """Translate the config's ``mcp_server_command`` into the
        concrete (command, args) the registrar should pass to
        ``openclaw mcp set``.

        Three cases:

        - ``None``: registration is explicitly disabled. Returns
          ``(None, [])`` — the registrar's ``is_configured`` will
          report False.
        - ``"auto"`` (default): resolve to the canonical entry
          script. Uses the project's ``.venv`` Python so OpenClaw
          spawns with the right interpreter, and prepends the
          entry script path before any caller-provided extra args.
        - explicit string: use as-is. ``configured_args`` is
          appended verbatim.

        When ``"auto"`` is used but the canonical script can't be
        located (unusual layout / missing checkout), falls back to
        ``(None, [])`` so the registrar disables itself rather than
        registering a broken command.
        """
        if configured is None:
            return None, []
        if configured == "auto":
            from ultron.config import PROJECT_ROOT
            # The .venv Python that OpenClaw should spawn.
            venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
            if not venv_python.exists():
                # POSIX fallback (the prototype's reference setup is
                # Windows, but support reproducible installs elsewhere).
                venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
            entry = (
                PROJECT_ROOT / "scripts" / "run_ultron_mcp_for_openclaw.py"
            )
            if not entry.exists():
                logger.warning(
                    "openclaw.bridge.mcp_server_command='auto' but the "
                    "entry script %s does not exist; MCP registration "
                    "will be disabled.", entry,
                )
                return None, []
            if not venv_python.exists():
                # No project venv — fall back to the current interpreter.
                # OpenClaw will spawn from the same env, so the import
                # path resolves the same way.
                import sys as _sys
                interpreter = _sys.executable
            else:
                interpreter = str(venv_python)
            return interpreter, [str(entry), "--stdio", *configured_args]
        return configured, configured_args

    @staticmethod
    def _run_async(coro):
        """Run a coroutine to completion using the running loop when
        possible, falling back to a one-shot loop. Caller must pass
        a coroutine, not a task or future."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            # Inside a live event loop — schedule on it and block on the
            # result. This path is rare for the bridge holder (orchestrator
            # is sync), but keeps the helper usable from async contexts.
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
        return asyncio.run(coro)


__all__ = [
    "OpenClawBridge",
]
