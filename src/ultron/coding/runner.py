"""Coding task runner: bridges -> tasks -> voice-friendly progress narration.

The runner owns at most one in-flight task at a time. The orchestrator
calls :meth:`start_task` to submit work, then -- whenever the user asks
"how's it going?" -- the orchestrator calls :meth:`progress_narration`
to get back a one-or-two-sentence Ultron-character status update.

Internally the runner listens for :class:`TaskEvent` instances from the
bridge and folds them into a small running summary so progress queries
are O(1) to answer.

The runner is bridge-agnostic: it accepts any concrete
:class:`CodingBridge` (today's :class:`DirectClaudeCodeBridge`, the
future ``OpenClawBridge``, or a test mock). The :func:`build_default_bridge`
helper picks one based on ``settings.CODING_BRIDGE``.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from config import settings
from ultron.coding.bridge import (
    CodingBridge,
    EventKind,
    TaskEvent,
    TaskHandle,
    TaskRequest,
    TaskResult,
    TaskState,
)
from ultron.coding.narration import StatusNarrator
from ultron.coding.session import ProjectSession, SessionStore
from ultron.utils.logging import get_logger

logger = get_logger("coding.runner")


# ---------------------------------------------------------------------------
# Bridge factory
# ---------------------------------------------------------------------------


def build_default_bridge() -> CodingBridge:
    """Construct the bridge selected by ``settings.CODING_BRIDGE``."""
    name = (settings.CODING_BRIDGE or "direct").strip().lower()
    if name == "direct":
        from ultron.coding.direct_bridge import DirectClaudeCodeBridge
        return DirectClaudeCodeBridge(
            claude_cli=settings.CODING_CLAUDE_CLI,
            log_path=settings.CODING_TASK_LOG_PATH,
        )
    if name == "openclaw":
        # Slot reserved for the OpenClaw HTTP bridge. When implemented,
        # importing will succeed and this raise will go away.
        try:
            from ultron.coding.openclaw_bridge import OpenClawBridge  # type: ignore
        except ImportError:
            raise NotImplementedError(
                "OpenClaw bridge not yet implemented. "
                "Set ULTRON_CODING_BRIDGE=direct."
            )
        return OpenClawBridge()
    raise ValueError(
        f"Unknown coding bridge: {name!r}. Expected 'direct' or 'openclaw'."
    )


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


@dataclass
class ProgressSinceLastQuery:
    """Delta-state used to answer 'what have you done since I last asked?'."""

    new_files_created: List[Path] = field(default_factory=list)
    new_files_modified: List[Path] = field(default_factory=list)
    new_steps: List[str] = field(default_factory=list)
    last_polled_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class CodingTaskRunner:
    """Owns one in-flight task and produces voice-friendly progress narration."""

    def __init__(
        self,
        bridge: Optional[CodingBridge] = None,
        log_path: Optional[Path] = None,
        narrator: Optional[StatusNarrator] = None,
        store: Optional[SessionStore] = None,
    ) -> None:
        self.bridge: CodingBridge = bridge or build_default_bridge()
        self._handle: Optional[TaskHandle] = None
        self._handle_lock = threading.Lock()
        self._log_path = log_path or settings.CODING_TASK_LOG_PATH

        # Multi-turn session tracking (Phase 2). When a task is submitted
        # we capture its claude_session_id + per-session metadata so a
        # later send_followup() can resume the same Claude conversation.
        self._claude_session_id: Optional[str] = None
        self._project_cwd: Optional[Path] = None
        self._project_model: str = "haiku"
        self._project_label: Optional[str] = None
        self._mcp_config_path: Optional[Path] = None

        # Tracking state for delta-narration ("since you last asked").
        # Used by the legacy bridge-only narration path. The Phase 5
        # session-aware path reads its delta off the ProjectSession's
        # last_user_status_query timestamp instead.
        self._last_seen_step_index = 0
        self._last_seen_files_created = 0
        self._last_seen_files_modified = 0
        self._delta_lock = threading.Lock()

        # Phase 5 wiring. When both a narrator and a store are provided,
        # progress_narration(session=...) routes through the rich
        # session-aware delta narration. Both are optional -- existing
        # callers / tests that construct the runner without these still
        # get the legacy bridge-only narration.
        self._narrator = narrator
        self._store = store

    # --- task lifecycle -----------------------------------------------------

    def has_active_task(self) -> bool:
        with self._handle_lock:
            h = self._handle
        return h is not None and h.is_running()

    def active_state(self) -> Optional[TaskState]:
        with self._handle_lock:
            h = self._handle
        if h is None:
            return None
        return h.state()

    def start_task(self, request: TaskRequest) -> TaskHandle:
        with self._handle_lock:
            if self._handle is not None and self._handle.is_running():
                raise RuntimeError(
                    "A coding task is already running. Wait for it to "
                    "complete or cancel it before starting another."
                )
            handle = self.bridge.submit(request)
            self._handle = handle
            # Capture multi-turn metadata so send_followup() can re-submit
            # to the same Claude session later.
            self._claude_session_id = getattr(handle, "claude_session_id", None)
            self._project_cwd = request.cwd
            self._project_model = request.model
            self._project_label = request.label
            self._mcp_config_path = request.mcp_config_path
            self._reset_delta_baseline()

        # Tee task lifecycle to a JSONL audit log -- one line per event.
        if self._log_path is not None:
            handle.add_listener(self._make_log_listener(handle.task_id()))
        # Also log a structured "start" record for offline inspection.
        self._log_record({
            "ts": time.time(),
            "task_id": handle.task_id(),
            "kind": "start",
            "label": request.label or "",
            "cwd": str(request.cwd),
            "model": request.model,
            "prompt_chars": len(request.task_prompt),
            "bridge": self.bridge.name(),
        })
        return handle

    def cancel_active(self) -> None:
        with self._handle_lock:
            h = self._handle
        if h is not None and h.is_running():
            h.cancel()

    def send_followup(self, prompt: str, kind: str = "adjustment") -> Optional[TaskHandle]:
        """Resume the active Claude session with a follow-up prompt.

        Used by:
          * the coordinator when translating a user adjustment
          * the verifier when sending a corrective prompt (Phase 4)
          * the coordinator when relaying a user clarification answer

        If a task is currently running we wait for it to complete before
        sending the follow-up -- Claude Code's ``--resume`` does not
        attach to a live subprocess, only to a persisted session.
        Returns the new TaskHandle on success, or ``None`` if there's no
        prior session to resume.
        """
        if not self._claude_session_id or not self._project_cwd:
            logger.warning("send_followup: no prior session to resume")
            return None

        with self._handle_lock:
            current = self._handle
        if current is not None and current.is_running():
            try:
                current.wait(timeout=settings.CODING_TASK_TIMEOUT_S)
            except TimeoutError:
                logger.warning("send_followup: prior task didn't finish in time")
                return None

        request = TaskRequest(
            task_prompt=prompt,
            cwd=self._project_cwd,
            model=self._project_model,
            label=f"{self._project_label or 'session'}-followup-{kind}",
            require_testing=False,        # the follow-up itself is the directive
            timeout_s=float(settings.CODING_TASK_TIMEOUT_S),
            claude_session_id=self._claude_session_id,
            mcp_config_path=self._mcp_config_path,
        )
        with self._handle_lock:
            handle = self.bridge.submit(request)
            self._handle = handle
            self._reset_delta_baseline()
        if self._log_path is not None:
            handle.add_listener(self._make_log_listener(handle.task_id()))
        self._log_record({
            "ts": time.time(),
            "task_id": handle.task_id(),
            "kind": "followup",
            "followup_kind": kind,
            "claude_session_id": self._claude_session_id,
            "prompt_chars": len(prompt),
        })
        return handle

    @property
    def claude_session_id(self) -> Optional[str]:
        return self._claude_session_id

    def wait_active(self, timeout: Optional[float] = None) -> Optional[TaskResult]:
        with self._handle_lock:
            h = self._handle
        if h is None:
            return None
        return h.wait(timeout=timeout)

    # --- voice-friendly narration ------------------------------------------

    def progress_narration(
        self, session: Optional[ProjectSession] = None,
    ) -> str:
        """One-or-two sentence Ultron-character status string.

        Phase 5: when a :class:`ProjectSession` is passed in, the
        narration is delta-aware (it reports only what's changed since
        the user's last query) and runs through the configured
        :class:`StatusNarrator` -- which voices the EXECUTING path via
        the LLM and handles every other status with deterministic
        edge-case lines.

        After a session-driven narration, the runner stamps
        ``session.last_user_status_query`` to ``now`` so the next call
        computes its delta against this point. The voice controller can
        also do this directly via ``store.touch_status_query()``;
        whichever path runs first wins (the timestamp is monotonic).

        When ``session`` is ``None`` the legacy bridge-state path runs:
        it inspects the active :class:`TaskState` and produces a
        bridge-derived line. Existing tests construct the runner without
        a narrator/store, so they exercise this path.

        Safe to call from any thread.
        """
        if session is not None:
            narrator = self._narrator or StatusNarrator(llm=None)
            text = narrator.narrate(session)
            # Stamp the timestamp through the store when one is wired so
            # the update is taken under the store's lock. Falling back
            # to direct mutation when the store is absent keeps the
            # narrator usable in pure unit tests.
            if self._store is not None:
                try:
                    self._store.touch_status_query(session.session_id)
                except Exception as e:
                    logger.debug("touch_status_query failed: %s", e)
            else:
                session.last_user_status_query = time.time()
            return text

        state = self.active_state()
        if state is None:
            return "No coding task is currently active."

        current = state.current_step or "starting"
        elapsed = max(0.0, time.time() - state.started_at)

        with self._delta_lock:
            new_steps = state.completed_steps[self._last_seen_step_index:]
            new_files_created = state.files_created[self._last_seen_files_created:]
            new_files_modified = state.files_modified[self._last_seen_files_modified:]
            self._last_seen_step_index = len(state.completed_steps)
            self._last_seen_files_created = len(state.files_created)
            self._last_seen_files_modified = len(state.files_modified)

        if state.is_complete:
            return self.completion_narration()

        # Build a concise narration. Keep it short -- TTS will read every
        # word and the user is interrupting their own conversation to ask.
        sentences: List[str] = []
        sentences.append(f"Currently {current}.")

        if new_files_created or new_files_modified:
            parts = []
            if new_files_created:
                parts.append(
                    f"{len(new_files_created)} new file"
                    + ("s" if len(new_files_created) != 1 else "")
                )
            if new_files_modified:
                parts.append(
                    f"{len(new_files_modified)} modification"
                    + ("s" if len(new_files_modified) != 1 else "")
                )
            sentences.append(f"Since you last asked: {', '.join(parts)}.")
        elif new_steps:
            sentences.append(f"Since you last asked: {len(new_steps)} steps.")
        else:
            sentences.append("No new completed steps since you last asked.")

        sentences.append(f"Total: {state.tool_use_count} tool calls in {int(elapsed)} seconds.")
        return " ".join(sentences)

    def completion_narration(self) -> str:
        """One-paragraph 'task is done' summary."""
        state = self.active_state()
        if state is None:
            return "No coding task has run."
        if not state.is_complete:
            return "Task still in progress."
        n_created = len(state.files_created)
        n_modified = len(state.files_modified)
        n_deleted = len(state.files_deleted)
        elapsed = int(state.duration_s)
        path = state.cwd

        if state.success:
            opener = "Done."
        elif state.is_cancelled:
            opener = "Cancelled."
        else:
            opener = "Task failed."

        parts: List[str] = [opener]
        change_bits: List[str] = []
        if n_created:
            change_bits.append(
                f"created {n_created} file" + ("s" if n_created != 1 else "")
            )
        if n_modified:
            change_bits.append(
                f"modified {n_modified}" + (
                    " files" if n_modified != 1 else " file"
                )
            )
        if n_deleted:
            change_bits.append(
                f"deleted {n_deleted}" + (
                    " files" if n_deleted != 1 else " file"
                )
            )
        if change_bits:
            parts.append(", ".join(change_bits).capitalize() + ".")
        parts.append(f"Project root: {path}.")
        if state.error and not state.success:
            parts.append(f"Error: {state.error}.")
        elif state.final_summary:
            tail = state.final_summary.strip().splitlines()
            tail_text = tail[-1] if tail else state.final_summary.strip()
            if tail_text:
                parts.append(tail_text[:300])
        parts.append(f"Elapsed: {elapsed} seconds.")
        return " ".join(parts)

    def _reset_delta_baseline(self) -> None:
        with self._delta_lock:
            self._last_seen_step_index = 0
            self._last_seen_files_created = 0
            self._last_seen_files_modified = 0

    # --- audit log ----------------------------------------------------------

    def _make_log_listener(self, task_id: str):
        """Return an event listener that writes one JSON line per event."""
        log_path = self._log_path

        def _listener(event: TaskEvent) -> None:
            try:
                self._log_record({
                    "ts": event.timestamp,
                    "task_id": task_id,
                    "kind": event.kind.value,
                    "stage": event.stage,
                    "tool_name": event.tool_name,
                    "tool_success": event.tool_success,
                    "file_path": str(event.file_path) if event.file_path else None,
                    "file_change_kind": (
                        event.file_change_kind.value
                        if event.file_change_kind else None
                    ),
                    "error": event.error,
                    "exit_status": event.exit_status,
                    "duration_s": event.duration_s,
                    "text_chars": len(event.text or ""),
                })
            except Exception as e:
                logger.debug("audit log write failed: %s", e)
        return _listener

    def _log_record(self, record: dict) -> None:
        if self._log_path is None:
            return
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass
