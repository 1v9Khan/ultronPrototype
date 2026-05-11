"""Coding task runner: bridges -> tasks -> voice-friendly progress narration.

The runner owns at most one in-flight task at a time. The orchestrator
calls :meth:`start_task` to submit work, then -- whenever the user asks
"how's it going?" -- the orchestrator calls :meth:`progress_narration`
to get back a one-or-two-sentence Ultron-character status update.

Internally the runner listens for :class:`TaskEvent` instances from the
bridge and folds them into a small running summary so progress queries
are O(1) to answer.

The runner is bridge-agnostic: it accepts any concrete
:class:`CodingBridge` (today's :class:`DirectClaudeCodeBridge` or a
test mock). The :func:`build_default_bridge` helper picks one based on
``settings.CODING_BRIDGE`` — only ``"direct"`` is supported.
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
from ultron.errors import FilesystemError
from ultron.resilience import get_error_log
from ultron.utils.logging import get_logger

logger = get_logger("coding.runner")

# True once we've already logged a coding-tasks audit-write failure to
# errors.jsonl during this process. Subsequent failures still skip the
# write but don't spam the log -- the first occurrence captures the
# actionable signal (path, errno, traceback).
_AUDIT_WRITE_FAILURE_LOGGED = False


# ---------------------------------------------------------------------------
# Bridge factory
# ---------------------------------------------------------------------------


def build_default_bridge() -> CodingBridge:
    """Construct the bridge selected by ``settings.CODING_BRIDGE``.

    Only ``"direct"`` is supported. The previous Phase A foundation
    reserved an ``"openclaw"`` slot here that pointed at a never-built
    ``ultron.coding.openclaw_bridge`` module — the new architecture
    treats OpenClaw as a peer Gateway via the routing layer
    (``ultron.openclaw_routing``), NOT as a Claude-Code bridge
    alternative. The reservation was removed in Foundation Part 5.
    """
    name = (settings.CODING_BRIDGE or "direct").strip().lower()
    if name == "direct":
        from ultron.coding.direct_bridge import DirectClaudeCodeBridge
        return DirectClaudeCodeBridge(
            claude_cli=settings.CODING_CLAUDE_CLI,
            log_path=settings.CODING_TASK_LOG_PATH,
        )
    raise ValueError(
        f"Unknown coding bridge: {name!r}. Only 'direct' is supported. "
        f"OpenClaw is a peer dispatcher (see ultron.openclaw_routing), "
        f"not a coding bridge alternative."
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

        # Phase 7: ProjectSession this runner is tracking, when known.
        # Set via :meth:`bind_session` so token-usage events from the
        # bridge can be forwarded to the right session in the store.
        self._bound_session_id: Optional[str] = None
        # Latest budget-warning text the orchestrator should surface to
        # the user (consumed once; voice loop polls + speaks).
        self._pending_budget_warning: Optional[str] = None
        self._budget_lock = threading.Lock()
        # 4B plan Item 7 — canonical-path-monitor abort message.
        # Mirrors the budget-warning pattern: queued once when the
        # monitor signals abort, consumed once by the voice loop.
        self._pending_canonical_abort: Optional[str] = None
        self._canonical_lock = threading.Lock()

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

    def bind_session(self, session_id: Optional[str]) -> None:
        """Phase 7: associate the runner with a :class:`ProjectSession`.

        When set, every USAGE event the bridge emits is forwarded to
        ``store.record_tokens(session_id, ...)`` so the session's
        ``tokens_used`` total stays current without the bridge needing
        a direct dependency on the store. Call this BEFORE
        :meth:`start_task` so the listener catches early events.
        """
        self._bound_session_id = session_id

    def _session_audit(self, event: str, **fields) -> None:
        """Phase 7: write to the bound session's per-session JSONL log.
        Used to record prompts sent to Claude (initial + followups) so
        the session log is the single retrospective view."""
        if self._bound_session_id is None or self._store is None:
            return
        writer = getattr(self._store, "audit_writer", None)
        if writer is None:
            return
        try:
            writer.write(self._bound_session_id, event, **fields)
        except Exception as e:
            logger.debug("session audit write failed: %s", e)

    def start_task(self, request: TaskRequest) -> TaskHandle:
        # Budget check: refuse to start a new task if the bound session's
        # budget is exhausted. The voice layer surfaces the same warning
        # via :meth:`pop_budget_warning`; this is a hard backstop.
        if self._is_session_halted():
            raise RuntimeError(
                "Coding session has hit its token budget. "
                "Cannot start another task without user approval."
            )
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
        # Phase 7: forward USAGE events to the bound session (if any) +
        # check the budget after each one.
        if self._bound_session_id is not None and self._store is not None:
            handle.add_listener(self._make_usage_listener())
        # 4B plan Item 7 — canonical-path-monitor listener (off when
        # ``coding.canonical_monitor.enabled`` is False; build_default_monitor
        # returns None so this is a cheap no-op in that case).
        canonical_listener = self._make_canonical_monitor_listener(handle)
        if canonical_listener is not None:
            handle.add_listener(canonical_listener)
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
        # Phase 7: persist the prompt text into the per-session log so the
        # JSONL is the single retrospective view of what Claude was told.
        # Truncate at 8000 chars -- enough to reconstruct intent + scope
        # without bloating the file when prompts include big templates.
        self._session_audit(
            "claude_prompt_sent",
            kind="initial",
            task_id=handle.task_id(),
            label=request.label or "",
            cwd=str(request.cwd),
            model=request.model,
            prompt_chars=len(request.task_prompt),
            prompt=request.task_prompt[:8000],
            claude_session_id=getattr(handle, "claude_session_id", None),
        )
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
        # Phase 7: refuse to spend more tokens once the budget is hit.
        if self._is_session_halted():
            logger.warning(
                "send_followup: session at token budget cap; refusing follow-up",
            )
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
        # Phase 7: persist the follow-up prompt to the per-session log too.
        self._session_audit(
            "claude_prompt_sent",
            kind=f"followup_{kind}",
            task_id=handle.task_id(),
            claude_session_id=self._claude_session_id,
            prompt_chars=len(prompt),
            prompt=prompt[:8000],
        )
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
        no_file_activity = (n_created + n_modified + n_deleted) == 0
        elapsed = int(state.duration_s)
        path = state.cwd

        # 2026-05-11 narration honesty: a success exit code with zero
        # file activity means Claude returned cleanly but did no work
        # -- usually budget exhaustion mid-exploration or a generic
        # "what should I build?" tail response. The legacy "Done."
        # opener was misleading the user (folder created, no scripts
        # inside). Surface the lack of file activity explicitly so the
        # user knows to say "continue" or rephrase. Failures and
        # cancellations keep their existing openers because the user
        # already knows the task didn't complete normally.
        if state.success and no_file_activity:
            opener = (
                "I finished without writing or modifying any files. "
                "The project may need more direction, or it may have "
                "run out of token budget mid-exploration -- say continue "
                "if you want me to keep going."
            )
        elif state.success:
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
        elif state.final_summary and not no_file_activity:
            # 2026-05-11: when no files were written, skip Claude's
            # tail summary -- it's usually a generic "what should I
            # build?" line that adds noise to the honest narration.
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

    # --- Phase 7: token budget --------------------------------------------

    def _is_session_halted(self) -> bool:
        if self._bound_session_id is None or self._store is None:
            return False
        try:
            session = self._store.get(self._bound_session_id)
        except Exception:
            return False
        return bool(session.budget_halted)

    def _make_usage_listener(self):
        """Return an event listener that forwards USAGE events to the
        store and checks the budget."""
        session_id = self._bound_session_id
        store = self._store

        def _listener(event: TaskEvent) -> None:
            if event.kind != EventKind.USAGE:
                return
            if session_id is None or store is None:
                return
            try:
                total = store.record_tokens(
                    session_id,
                    input_tokens=event.usage_input or 0,
                    output_tokens=event.usage_output or 0,
                    cache_creation_tokens=event.usage_cache_creation or 0,
                    cache_read_tokens=event.usage_cache_read or 0,
                )
            except Exception as e:
                logger.debug("record_tokens failed: %s", e)
                return
            self._check_budget(session_id, total)

        return _listener

    def _check_budget(self, session_id: str, tokens_used: int) -> None:
        """Compare ``tokens_used`` against the configured budget; flip
        the warning / halt fields on the session and queue a voice
        warning if a threshold has just been crossed."""
        if self._store is None:
            return
        try:
            session = self._store.get(session_id)
        except Exception:
            return
        budget = settings.CODING_TOKEN_BUDGET_PER_SESSION
        if budget <= 0:
            return
        ratio = tokens_used / budget
        warn_at = settings.CODING_TOKEN_WARNING_THRESHOLD
        # Halt at 100%.
        if ratio >= 1.0 and not session.budget_halted:
            session.budget_halted = True
            self._queue_warning(
                f"Token budget exhausted on {self._project_label or 'this session'} "
                f"({tokens_used} of {budget}). Pausing follow-ups; "
                f"say continue if you want him to keep going."
            )
            return
        # Warn at threshold.
        if ratio >= warn_at and not session.budget_warning_emitted:
            session.budget_warning_emitted = True
            pct = int(ratio * 100)
            self._queue_warning(
                f"Heads up: he's at {pct}% of the token budget on "
                f"{self._project_label or 'this session'}."
            )

    def _queue_warning(self, text: str) -> None:
        with self._budget_lock:
            self._pending_budget_warning = text

    def pop_budget_warning(self) -> Optional[str]:
        """Voice loop polls this each iteration to surface budget warnings.
        Returns the queued text once, then clears."""
        with self._budget_lock:
            text = self._pending_budget_warning
            self._pending_budget_warning = None
        return text

    # --- A4 pre-task confirmation audit -----------------------------------

    def record_pre_task_aborted(
        self,
        *,
        label: Optional[str],
        reason: str,
        intent_text: str = "",
    ) -> None:
        """A4: log a pre-task confirmation that was aborted before dispatch.

        Called by the orchestrator when its barge-in watcher detected a
        wake-word fire during the confirmation TTS, so the user's
        interrupt is durable in the coding task audit log. Best-effort:
        log-write failures degrade silently rather than crash the
        voice loop.
        """
        try:
            self._log_record({
                "event": "pre_task_aborted",
                "label": label or "(unset)",
                "reason": reason,
                "intent_text": (intent_text or "")[:300],
                "ts": time.time(),
            })
        except Exception as e:
            logger.debug("pre_task_aborted audit failed: %s", e)

    # --- 4B plan Item 7: canonical-path-monitor wiring --------------------

    def _make_canonical_monitor_listener(self, handle):
        """Build a per-task canonical-path-monitor listener.

        Returns ``None`` when the feature is disabled in config —
        callers treat that as "no listener to add". When enabled:

        - The listener observes each TaskEvent.
        - On the first abort verdict, it cancels the active handle,
          queues a voice narration via ``_pending_canonical_abort``
          (consumed by ``pop_canonical_abort_warning`` in the voice
          loop), and logs the reason.
        - Aborts latch — subsequent events on the same handle are
          ignored, so the listener doesn't re-cancel.
        """
        from ultron.coding.canonical_monitor import build_default_monitor

        monitor = build_default_monitor("CODE_TASK")
        if monitor is None:
            return None
        # Per-handle latch so a single monitor instance cancels at most
        # once. ``state["fired"]`` is checked + set inside the
        # listener; the lock guards the queued voice message.
        state = {"fired": False}

        def _listener(event):
            try:
                verdict = monitor.observe(event)
                if not verdict.should_abort or state["fired"]:
                    return
                state["fired"] = True
                logger.warning(
                    "CanonicalPathMonitor abort: %s", verdict.reason,
                )
                with self._canonical_lock:
                    self._pending_canonical_abort = (
                        f"I'm stopping that task — it was going off the rails. "
                        f"{verdict.off_canonical_count} unexpected tool calls in "
                        f"the first {verdict.total_tool_calls}. "
                        f"Ask me to try again with a clearer description."
                    )
                try:
                    handle.cancel()
                except Exception as e:
                    logger.warning("canonical-monitor cancel failed: %s", e)
            except Exception as e:
                # The listener must NEVER raise back into the bridge —
                # would break event delivery.
                logger.debug("canonical monitor listener error: %s", e)

        return _listener

    def pop_canonical_abort_warning(self) -> Optional[str]:
        """Voice loop polls this each iteration to surface canonical-
        monitor abort narration. Returns the queued text once, then
        clears. Mirrors :meth:`pop_budget_warning`."""
        with self._canonical_lock:
            text = self._pending_canonical_abort
            self._pending_canonical_abort = None
        return text

    # --- audit log ----------------------------------------------------------

    def _make_log_listener(self, task_id: str):
        """Return an event listener that writes one JSON line per event."""

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
        except OSError as e:
            global _AUDIT_WRITE_FAILURE_LOGGED
            if not _AUDIT_WRITE_FAILURE_LOGGED:
                _AUDIT_WRITE_FAILURE_LOGGED = True
                get_error_log().record(
                    FilesystemError(
                        f"coding tasks audit-log write failed: {e}",
                        context={"path": str(self._log_path)},
                        recovery="audit write skipped; system continues",
                    ),
                    dependency="filesystem",
                    include_traceback=False,
                )
