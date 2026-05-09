"""AutomationTaskRunner — stubbed-but-functional mirror of CodingTaskRunner.

Phase 5 ships this fully wired up to the OpenClawDispatcher (which
returns stubs). Each ``submit_task`` allocates a TaskInfo, fires the
dispatcher's handler, and stores the stub :class:`DispatchResult` so
``progress_narration`` / ``completion_narration`` have something to say.

After OpenClaw integration the dispatcher returns real results; this
runner needs no changes.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ultron.config import UltronConfig, get_config, resolve_path
from ultron.openclaw_routing.dispatcher import OpenClawDispatcher
from ultron.openclaw_routing.intents import (
    BrowserIntent,
    DesktopIntent,
    DispatchResult,
    FileOpIntent,
    GamingModeIntent,
    MediaGenIntent,
    MessagingIntent,
    RoutingIntent,
    RoutingIntentKind,
    ShellOpIntent,
    TaskInfo,
    WindowIntent,
)
from ultron.utils.logging import get_logger

logger = get_logger("openclaw_routing.runner")


class AutomationTaskRunner:
    """Tracks OpenClaw-dispatched tasks and serves voice narration about them.

    Args:
        config: full :class:`UltronConfig`; defaults to ``get_config()``.
        dispatcher: a configured :class:`OpenClawDispatcher`.
        audit_log_path: optional override for the JSONL audit log path.
            Defaults to ``logs/automation_tasks.jsonl``.
    """

    def __init__(
        self,
        config: Optional[UltronConfig] = None,
        dispatcher: Optional[OpenClawDispatcher] = None,
        audit_log_path: Optional[Path] = None,
    ) -> None:
        cfg = config if config is not None else get_config()
        self._dispatcher = dispatcher if dispatcher is not None else OpenClawDispatcher(cfg)
        self._tasks: Dict[str, TaskInfo] = {}
        self._results: Dict[str, DispatchResult] = {}
        self._lock = threading.Lock()
        self._audit_log_path = (
            Path(audit_log_path) if audit_log_path is not None
            else resolve_path("logs/automation_tasks.jsonl")
        )
        self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_lock = threading.Lock()

    # --- task lifecycle ----------------------------------------------------

    async def submit_task(self, intent: RoutingIntent) -> str:
        """Submit an automation intent. Returns a task_id.

        The dispatcher is awaited inline. In Phase 5 every dispatch returns
        a stub immediately; once OpenClaw is integrated this becomes a
        long-lived HTTP call and we'll either keep awaiting inline (if
        the response is fast) or move to a background task pattern.
        """
        task_id = uuid.uuid4().hex[:12]
        info = TaskInfo(
            task_id=task_id,
            kind=intent.kind,
            description=intent.raw_text[:200],
            started_at=time.time(),
        )
        with self._lock:
            self._tasks[task_id] = info

        result = await self._dispatch(intent)

        with self._lock:
            info.completed_at = time.time()
            info.success = bool(result.success)
            info.voice_summary = result.voice_message
            self._results[task_id] = result

        self._audit({
            "task_id": task_id,
            "kind": intent.kind.value,
            "description": info.description,
            "started_at": info.started_at,
            "completed_at": info.completed_at,
            "success": info.success,
            "voice_summary": info.voice_summary,
            "stub": result.metadata.get("stub", False),
            "error": result.error,
        })
        return task_id

    async def _dispatch(self, intent: RoutingIntent) -> DispatchResult:
        """Route the intent to the right dispatcher method.

        V1-gap A1 / C3 — gaming-mode, desktop, and window intents live
        on dedicated fields (``gaming_mode_intent``, ``desktop_intent``,
        ``window_intent``) rather than ``automation_intent``. They get
        priority routing here so the legacy ``automation_intent`` path
        still works for browser / media / messaging / file / shell.
        """
        # V1-gap A1 — gaming mode.
        if (
            intent.kind == RoutingIntentKind.GAMING_MODE
            and intent.gaming_mode_intent is not None
        ):
            return await self._dispatcher.handle_gaming_mode(
                intent.gaming_mode_intent,
            )
        # V1-gap C3 — desktop / window control.
        if (
            intent.kind == RoutingIntentKind.DESKTOP_AUTOMATION
            and intent.desktop_intent is not None
        ):
            return await self._dispatcher.handle_desktop_automation(
                intent.desktop_intent,
            )
        if (
            intent.kind == RoutingIntentKind.WINDOW_AUTOMATION
            and intent.window_intent is not None
        ):
            return await self._dispatcher.handle_window_automation(
                intent.window_intent,
            )

        a = intent.automation_intent
        if a is None:
            return DispatchResult(
                success=False,
                voice_message="I couldn't translate that into an action.",
                error="missing automation_intent",
            )
        if isinstance(a, BrowserIntent):
            return await self._dispatcher.handle_browser(a)
        if isinstance(a, MediaGenIntent):
            return await self._dispatcher.handle_media_generation(a)
        if isinstance(a, MessagingIntent):
            return await self._dispatcher.handle_messaging(a)
        if isinstance(a, FileOpIntent):
            return await self._dispatcher.handle_file_operation(a)
        if isinstance(a, ShellOpIntent):
            return await self._dispatcher.handle_shell_operation(a)
        return DispatchResult(
            success=False,
            voice_message="I don't know how to handle that intent yet.",
            error=f"unknown automation_intent type: {type(a).__name__}",
        )

    # --- introspection / narration ----------------------------------------

    async def progress_narration(self, task_id: str) -> Optional[str]:
        """Return a voice-friendly status string, or None if no such task.

        In Phase 5 every task completes synchronously inside ``submit_task``
        (the dispatcher returns immediately), so progress is always either
        "done" or "no such task". Once OpenClaw lands and dispatch is
        long-lived, this method returns intermediate narration.
        """
        with self._lock:
            info = self._tasks.get(task_id)
            result = self._results.get(task_id)
        if info is None:
            return None
        if info.completed_at is None:
            return "Working on it."
        # Already done — narrate the stub voice message.
        return result.voice_message if result is not None else "Done."

    async def completion_narration(self, task_id: str) -> Optional[str]:
        """Same idea as progress, but only returns once the task is complete."""
        with self._lock:
            info = self._tasks.get(task_id)
            result = self._results.get(task_id)
        if info is None or info.completed_at is None:
            return None
        return result.voice_message if result is not None else None

    async def cancel(self, task_id: str) -> bool:
        """Cancellation is a no-op for stubs (they complete instantly).
        Returns False if the task doesn't exist or is already done."""
        with self._lock:
            info = self._tasks.get(task_id)
        if info is None or info.completed_at is not None:
            return False
        # If we ever have long-lived async dispatch, this would set a
        # cancel flag and abort the in-flight call.
        return False

    def list_active(self) -> List[TaskInfo]:
        """Return every task that hasn't completed yet."""
        with self._lock:
            return [t for t in self._tasks.values() if t.completed_at is None]

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        with self._lock:
            return self._tasks.get(task_id)

    # --- internal ----------------------------------------------------------

    def _audit(self, record: Dict[str, Any]) -> None:
        try:
            with self._log_lock:
                with self._audit_log_path.open("a", encoding="utf-8") as f:
                    json.dump(record, f, default=str)
                    f.write("\n")
        except OSError as e:
            logger.warning("automation_tasks.jsonl write failed: %s", e)


__all__ = ["AutomationTaskRunner"]
