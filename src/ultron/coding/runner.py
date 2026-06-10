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
from typing import List, Optional, Tuple

from config import settings
from ultron.coding.anchors import (
    AnchorBudget,
    AnchorPlan,
    GoalAnchor,
    decompose_into_anchors,
    narration_for_anchor,
    narration_for_completion,
)
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
        # E2 goal-anchor planning (default-OFF in config). When the
        # feature is enabled, ``start_task`` builds an
        # :class:`AnchorPlan` from the task prompt + listener routes
        # USAGE events to the active anchor's budget.
        self._anchor_plan: Optional[AnchorPlan] = None
        self._anchor_lock = threading.Lock()
        self._pending_anchor_narration: Optional[str] = None
        # 2026-05-19 Track 1g voice-loop integration -- per-task
        # AST-syntax-failure tracker. The FILE_CHANGE listener appends
        # ``(path_leaf, error_message)`` pairs on every parse failure
        # so :meth:`completion_narration` can surface the count + first
        # few filenames to the user. Stays empty (and the narration
        # branch stays inert) when ``coding.ast_metadata.enabled`` is
        # False. Reset on every :meth:`start_task` so failures from a
        # prior task don't leak into the next narration.
        self._ast_failures_this_task: List[Tuple[str, str]] = []
        self._ast_lock = threading.Lock()
        # 2026 catalog 08/09 wiring: per-task dialog narration queue.
        # The dialog-auto-handler listener subscribes to the bus's
        # DialogAppearedEvent for the lifetime of the task and pushes
        # voice-friendly text into this queue. The orchestrator drains
        # the queue via :meth:`pending_dialog_narration` between bus
        # turn-completion events, the same way it drains
        # :meth:`pending_completion`.
        self._pending_dialog_narrations: List[str] = []
        self._dialog_lock = threading.Lock()
        # Unsubscribe callable returned by bus.subscribe; the runner
        # tears it down when the task COMPLETEs so subsequent dialogs
        # don't fire for the previous task's listener.
        self._dialog_unsubscribe: Optional[object] = None
        # 2026 catalog 14 (T1): command/tool failures observed on the bridge
        # event stream, queued here + drained by the orchestrator each loop
        # iteration into the EvolutionService. Decoupled -- the runner never
        # imports the evolution package; each entry is
        # ``(command, output, exit_code)``.
        self._pending_command_failures: List[tuple] = []
        self._command_failure_lock = threading.Lock()
        # Production-hardening (#66): successfully-completed coding tasks.
        # The success listener queues ``(label, summary)`` pairs here; the
        # orchestrator drains them into the EvolutionService as
        # ``coding_task_success`` opportunity capsules. The runner never
        # imports the evolution package.
        self._pending_task_successes: List[tuple] = []
        self._task_success_lock = threading.Lock()
        # 2026 catalog wiring (T1): loop-detection heads-up lines. The
        # per-task LoopDetectionManager listener queues a single line here
        # when a task's tool-call stream trips a hard escalation; the
        # orchestrator drains + speaks it each voice-loop iteration.
        self._pending_loop_alerts: List[str] = []
        self._loop_alert_lock = threading.Lock()

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
        # Production-hardening: a sandbox project must be its own git
        # root so the spawned coding CLI treats IT as the project
        # boundary instead of walking up into the ultron repo and
        # loading the (very large) repo orientation context into every
        # task. Idempotent + sandbox-scoped + fail-open; a user's own
        # project directory outside the sandbox is never touched.
        try:
            from ultron.coding.projects import ensure_sandbox_isolation

            ensure_sandbox_isolation(request.cwd)
        except Exception as e:  # noqa: BLE001
            logger.debug("sandbox isolation pre-spawn skipped: %s", e)
        # Budget check: refuse to start a new task if the bound session's
        # budget is exhausted. The voice layer surfaces the same warning
        # via :meth:`pop_budget_warning`; this is a hard backstop.
        if self._is_session_halted():
            raise RuntimeError(
                "Coding session has hit its token budget. "
                "Cannot start another task without user approval."
            )
        # Hooks lifecycle: fire TaskStart; a user hook may cancel the task
        # (raises RuntimeError, surfaced like any other start-task refusal).
        # Fail-open + a zero-cost no-op when hooks are disabled / none installed.
        self._fire_task_start_hook(request)
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
            # Track 1g voice-loop integration -- clear any AST syntax
            # failures left over from the previous task so the next
            # ``completion_narration`` starts from a clean slate.
            with self._ast_lock:
                self._ast_failures_this_task.clear()

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
        # 2026-05-12 Phase 2 -- runtime safety validator FILE_CHANGE
        # listener. When AI coding agent spawns a write to a K-protected
        # (or any other rule-protected) path, the listener cancels the
        # handle and audits the abort. Belt-and-braces on top of the
        # OpenClaw dispatcher gate -- coding-bridge file writes never
        # go through that dispatcher, so they need their own check.
        safety_listener = self._make_safety_validator_listener(handle)
        if safety_listener is not None:
            handle.add_listener(safety_listener)
        # E2 goal-anchor planning -- off by default. When enabled,
        # build a per-task AnchorPlan from the prompt + register a
        # USAGE-listener that attributes tokens to the active anchor.
        anchor_listener = self._build_anchor_plan_and_listener(handle, request)
        if anchor_listener is not None:
            handle.add_listener(anchor_listener)
        # 2026-05-19 Tracks 1f + 1g -- AST syntax verification on
        # FILE_CHANGE events. Off by default. When enabled, every
        # Python file the AI coding agent writes gets parsed via stdlib ast;
        # a syntax failure emits an ``ast_syntax_failure`` audit row
        # so completion narration can fact-check the success claim
        # instead of trusting the bridge's exit code alone.
        ast_listener = self._make_ast_syntax_listener(handle)
        if ast_listener is not None:
            handle.add_listener(ast_listener)
        # 2026-05-22 catalog batch 4: pre-write lint cascade. Runs
        # alongside the AST listener; the two are independent (AST is
        # Python-only stdlib, lint cascade adds tree-sitter for
        # non-Python + flake8 fatal-rule subset for Python). Off by
        # default; enable via ``coding.pre_write_lint.enabled``.
        lint_listener = self._make_pre_write_lint_listener(handle)
        if lint_listener is not None:
            handle.add_listener(lint_listener)
        # 2026 catalog 08 + 09 wiring: dialog auto-handler. Subscribes
        # to DialogAppearedEvent on the bus for the lifetime of this
        # task and surfaces dialogs via the runner's
        # ``_pending_dialog_narrations`` queue (drained by the
        # orchestrator like ``pending_completion``). On task COMPLETE
        # the subscription is torn down so subsequent dialogs only
        # surface to the next task's bridge. Default ON; flip
        # ``coding.dialog_auto_handler.enabled=False`` to disable.
        dialog_listener = self._attach_dialog_auto_handler(handle)
        if dialog_listener is not None:
            handle.add_listener(dialog_listener)
        # 2026 catalog 14 (T1): observe command/tool failures on this task's
        # event stream + queue them for the orchestrator to feed the
        # EvolutionService. Gated + fail-open + a no-op when evolution or its
        # command-failure capture is disabled.
        evo_failure_listener = self._make_evolution_failure_listener(handle)
        if evo_failure_listener is not None:
            handle.add_listener(evo_failure_listener)
        # Production-hardening (#66): mirror listener for SUCCESSFUL task
        # completion -- queues a coding_task_success observation the
        # orchestrator drains into the EvolutionService. Gated + fail-open.
        evo_success_listener = self._make_evolution_success_listener(
            handle, request.label or ""
        )
        if evo_success_listener is not None:
            handle.add_listener(evo_success_listener)
        # 2026 catalog wiring (T1): per-task 5-detector loop detection over
        # the tool-call stream. Surfaces a single spoken heads-up (never a
        # cancel) when a hard escalation fires. Gated + fail-open + a no-op
        # when ``coding.loop_detection_enabled`` is False.
        loop_listener = self._make_loop_detection_listener(handle)
        if loop_listener is not None:
            handle.add_listener(loop_listener)
        # Hooks lifecycle: fire TaskComplete when the task finishes
        # (observability; a finished task can't be cancelled). Gated +
        # fail-open + a no-op when hooks are disabled / none installed.
        hook_listener = self._make_hook_lifecycle_listener(handle)
        if hook_listener is not None:
            handle.add_listener(hook_listener)
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
        sending the follow-up -- AI coding agent's ``--resume`` does not
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

        # E2 goal-anchor resume: when the operator has enabled the
        # ``resume_prepend_next_anchor`` flag AND an unfinished anchor
        # plan is still in flight, prepend a one-line "Continue with..."
        # directive so AI coding agent picks up at the right milestone
        # instead of restarting from scratch.
        prompt = self._maybe_prepend_anchor_resume(prompt)

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
        # 2026-05-11 follow-up fix: speak ONLY the project folder name,
        # never the absolute path. The legacy ``f"Project root: {path}."``
        # forced XTTS to synthesise the full Windows path verbatim
        # ("C:\STC\ultronPrototype\data\sandbox\<slug>"). XTTS-v2 chokes
        # on backslash/colon/drive-letter sequences -- the live session
        # log shows ``XTTS server synth failed ... timed out`` on exactly
        # that string while the GPU pinned at 100 % trying to pronounce
        # it. The full path is still on disk in the per-session JSONL
        # audit log + the `coding_tasks.jsonl` start event; the voice
        # narration only needs the leaf for human context. ``Path.name``
        # is the trailing component for both ``Path`` (the typed
        # ``state.cwd``) and ``str`` (defensive). StatusNarrator already
        # speaks the leaf name only -- this brings completion_narration
        # in line.
        try:
            project_name = path.name if isinstance(path, Path) else Path(str(path)).name
        except Exception:
            project_name = ""
        if project_name:
            parts.append(f"Saved under {project_name}.")
        # B3: when the task produced files, surface that the user can run it
        # by voice (ties the completion report to the run/launch feature).
        if state.success and not no_file_activity and project_name:
            try:
                from ultron.coding.sandbox_runner import resolve_entry_point
                if resolve_entry_point(path) is not None:
                    parts.append(f"Say run {project_name} to try it.")
            except Exception:                                        # noqa: BLE001
                pass
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
        # Track 1g voice-loop integration -- surface AST syntax
        # failures discovered during the task. The opener stays as it
        # was (Claude reports success based on exit code); this adds
        # an honest "but" sentence so the user knows ground truth
        # diverged from the bridge's success signal. Speak only file
        # leaves (TTS-safe, never paths) and cap at three filenames
        # so a runaway broken-rewrite cycle doesn't produce a 30 s
        # narration. Anything past three collapses into "and N more".
        with self._ast_lock:
            ast_failures = list(self._ast_failures_this_task)
        if ast_failures:
            n_fail = len(ast_failures)
            if n_fail == 1:
                parts.append(
                    f"However, one file has syntax errors: {ast_failures[0][0]}."
                )
            else:
                shown = [p for (p, _e) in ast_failures[:3]]
                shown_str = ", ".join(shown)
                if n_fail > 3:
                    parts.append(
                        f"However, {n_fail} files have syntax errors, "
                        f"including {shown_str}, and {n_fail - 3} more."
                    )
                else:
                    parts.append(
                        f"However, {n_fail} files have syntax errors: {shown_str}."
                    )
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

    def _make_safety_validator_listener(self, handle):
        """Build the runtime-safety FILE_CHANGE listener.

        Whenever the coding bridge reports a FILE_CHANGE event, this
        listener runs the safety validator with capability=
        ``coding_bridge`` and the changed path. Block verdicts cancel
        the task handle and queue an in-character abort narration
        (consumed via :meth:`pop_canonical_abort_warning` -- we
        deliberately reuse that slot so the voice path doesn't need
        a separate poll for safety-validator aborts).

        Returns ``None`` when the safety subsystem is unavailable
        (import error / no rules registered). The listener itself
        never raises -- exceptions are logged at DEBUG.
        """
        try:
            from ultron.safety import (
                RuleContext, get_validator,
            )
            from ultron.safety.path_resolver import (
                PathResolveError, get_path_resolver,
            )
            from ultron.coding.bridge import EventKind
        except Exception as e:
            logger.debug(
                "safety validator listener unavailable (%s); skipping", e,
            )
            return None

        resolver = get_path_resolver()
        # Per-handle latch so a single block aborts at most once.
        state = {"fired": False}

        def _listener(event):
            try:
                if state["fired"]:
                    return
                # Only act on FILE_CHANGE events. STATUS / TEXT /
                # TOOL_USE / etc. aren't directly path-driven (paths
                # in TOOL_USE call arguments would need parsing per-
                # tool; FILE_CHANGE is the post-fact ground truth).
                if getattr(event, "kind", None) != EventKind.FILE_CHANGE:
                    return
                raw_path = getattr(event, "file_path", None)
                if not raw_path:
                    return
                # ``file_path`` is an Optional[Path]; coerce to str so the
                # downstream resolve / .lower() / arguments stay string-typed.
                path_str = str(raw_path)
                try:
                    canonical = resolver.resolve(path_str)
                except PathResolveError as e:
                    logger.warning(
                        "safety listener: unresolvable path in FILE_CHANGE "
                        "event (%s); treating as a block trigger", e,
                    )
                    canonical = None

                ctx = RuleContext(
                    tool_name="coding_bridge.file_change",
                    arguments={
                        "path": path_str,
                        "change_kind": (
                            event.file_change_kind.value
                            if event.file_change_kind else ""
                        ),
                        "write": True,
                    },
                    capability="coding_bridge",
                    paths=tuple([canonical] if canonical is not None else []),
                    user_text="",  # explicit-intent not consulted on
                    # file changes -- writes are already past the
                    # prompt-side check.
                    has_pending_clarification=False,
                )
                validator = get_validator()
                verdict = validator.check(ctx)
                if verdict.is_allowed:
                    return
                # Block: cancel the task and queue the narration.
                state["fired"] = True
                logger.warning(
                    "safety validator blocked FILE_CHANGE event "
                    "(rule=%s reason=%s); cancelling coding task",
                    verdict.triggered_rule_id, verdict.reason,
                )
                with self._canonical_lock:
                    self._pending_canonical_abort = (
                        f"I'm stopping that task. "
                        f"It tried to {('write' if 'write' in path_str.lower() else 'modify')} "
                        f"a protected file. {verdict.reason}"
                    )
                try:
                    handle.cancel()
                except Exception as e:
                    logger.warning(
                        "safety listener cancel failed: %s", e,
                    )
            except Exception as e:
                # The listener must NEVER raise back into the bridge --
                # would break event delivery.
                logger.debug("safety listener error: %s", e)

        return _listener

    def pop_canonical_abort_warning(self) -> Optional[str]:
        """Voice loop polls this each iteration to surface canonical-
        monitor abort narration. Returns the queued text once, then
        clears. Mirrors :meth:`pop_budget_warning`."""
        with self._canonical_lock:
            text = self._pending_canonical_abort
            self._pending_canonical_abort = None
        return text

    # --- 2026 catalog 14 (T1): command-failure observation ----------------

    def _make_evolution_failure_listener(self, handle):
        """Build a fail-open ``TaskEvent`` listener that captures command /
        tool failures into ``self._pending_command_failures`` (catalog 14, T1)
        for the orchestrator to feed the EvolutionService. Returns ``None``
        (a zero-cost no-op) when evolution or its command-failure capture is
        disabled. The runner never imports the evolution package -- it only
        queues ``(command, output, exit_code)`` tuples."""
        try:
            from ultron.coding.bridge import EventKind
            from ultron.config import get_config
        except Exception as e:  # noqa: BLE001
            logger.debug("evolution failure listener unavailable (%s); skipping", e)
            return None
        try:
            ev = getattr(get_config(), "evolution", None)
            if ev is None or not getattr(ev, "enabled", False):
                return None
            if not getattr(ev, "command_failure_capture_enabled", True):
                return None
        except Exception:  # noqa: BLE001
            return None

        def _queue(command: str, output: str, exit_code) -> None:
            with self._command_failure_lock:
                # Bound the queue so a pathological task can't grow it without
                # limit before the orchestrator next drains it.
                if len(self._pending_command_failures) < 200:
                    self._pending_command_failures.append((command, output, exit_code))

        def _listener(event) -> None:
            try:
                kind = getattr(event, "kind", None)
                if kind == EventKind.ERROR:
                    _queue("coding task", str(getattr(event, "error", "") or ""), None)
                elif kind == EventKind.TOOL_RESULT and getattr(event, "tool_success", None) is False:
                    _queue(
                        str(getattr(event, "tool_name", "") or "tool"),
                        str(getattr(event, "tool_brief", "") or ""),
                        None,
                    )
                elif kind == EventKind.COMPLETE:
                    code = getattr(event, "exit_status", None)
                    if code is not None and int(code) != 0:
                        _queue("coding task", str(getattr(event, "summary", "") or ""), int(code))
            except Exception as e:  # noqa: BLE001 -- never raise back into the bridge
                logger.debug("evolution failure listener error: %s", e)

        return _listener

    def drain_command_failures(self) -> List[tuple]:
        """Pop + clear all queued command/tool failures (catalog 14, T1). The
        orchestrator polls this each loop iteration and feeds each
        ``(command, output, exit_code)`` to the EvolutionService."""
        with self._command_failure_lock:
            if not self._pending_command_failures:
                return []
            out = list(self._pending_command_failures)
            self._pending_command_failures.clear()
            return out

    # --- production-hardening (#66): coding-success observation -----------

    def _make_evolution_success_listener(self, handle, label: str):
        """Build a fail-open ``TaskEvent`` listener that captures a
        SUCCESSFUL task completion into ``self._pending_task_successes``
        for the orchestrator to feed the EvolutionService as a
        ``coding_task_success`` opportunity capsule (#66). Returns ``None``
        (a zero-cost no-op) when evolution is disabled. The runner never
        imports the evolution package -- it only queues
        ``(label, summary)`` pairs."""
        try:
            from ultron.coding.bridge import EventKind
            from ultron.config import get_config
        except Exception as e:  # noqa: BLE001
            logger.debug("evolution success listener unavailable (%s); skipping", e)
            return None
        try:
            ev = getattr(get_config(), "evolution", None)
            if ev is None or not getattr(ev, "enabled", False):
                return None
        except Exception:  # noqa: BLE001
            return None

        state = {"queued": False}

        def _listener(event) -> None:
            if state["queued"]:
                return
            try:
                if getattr(event, "kind", None) != EventKind.COMPLETE:
                    return
                code = getattr(event, "exit_status", None)
                if code is not None and int(code) != 0:
                    return
                state["queued"] = True
                summary = str(getattr(event, "summary", "") or "")
                with self._task_success_lock:
                    if len(self._pending_task_successes) < 50:
                        self._pending_task_successes.append((label, summary))
            except Exception as e:  # noqa: BLE001 -- never raise into the bridge
                logger.debug("evolution success listener error: %s", e)

        return _listener

    def drain_task_successes(self) -> List[tuple]:
        """Pop + clear all queued successful-completion observations (#66).
        The orchestrator polls this each loop iteration and feeds each
        ``(label, summary)`` to the EvolutionService as a
        ``coding_task_success`` opportunity capsule."""
        with self._task_success_lock:
            if not self._pending_task_successes:
                return []
            out = list(self._pending_task_successes)
            self._pending_task_successes.clear()
            return out

    # --- 2026 catalog wiring (T1): per-task loop detection --------------

    def _make_loop_detection_listener(self, handle):
        """Build a fail-open ``TaskEvent`` listener that watches this task's
        TOOL_RESULT stream for pathological repetition (T1, the 5-detector
        :class:`LoopDetectionManager`) and queues a single spoken heads-up when
        a hard escalation fires (e.g. the same tool failing identically the
        circuit-breaker number of times). Returns ``None`` (a zero-cost no-op)
        when ``coding.loop_detection_enabled`` is False or the detector package
        can't be imported.

        This is a BACKSTOP: the coding subprocess and OpenClaw agents each
        enforce their own turn limits, so it only trips on a genuine
        stuck-in-place loop the inner limits missed. It LOGS + NARRATES only --
        it never cancels the handle (canceling the coding subprocess mid-flight
        could lose work). The user hears the heads-up and can say "stop", which
        routes to the existing cancel path. One manager per task (closure
        local) so history never bleeds across tasks; narrates at most once per
        task.
        """
        try:
            from ultron.agent_loop.loop_detection_extended import (
                LoopDetectionManager,
                OutcomeKind,
                ToolCallRecord,
            )
            from ultron.coding.bridge import EventKind
            from ultron.config import get_config
        except Exception as e:  # noqa: BLE001
            logger.debug("loop-detection listener unavailable (%s); skipping", e)
            return None
        try:
            if not getattr(get_config().coding, "loop_detection_enabled", True):
                return None
        except Exception:  # noqa: BLE001
            pass
        try:
            manager = LoopDetectionManager()
        except Exception as e:  # noqa: BLE001
            logger.debug("loop-detection manager construction failed: %s", e)
            return None

        state = {"narrated": False}

        def _listener(event) -> None:
            if state["narrated"]:
                return
            try:
                if getattr(event, "kind", None) is not EventKind.TOOL_RESULT:
                    return
                success = getattr(event, "tool_success", None)
                brief = str(getattr(event, "tool_brief", "") or "")[:200]
                record = ToolCallRecord(
                    tool_name=str(getattr(event, "tool_name", "") or "tool"),
                    params={},
                    outcome_kind=(
                        OutcomeKind.ERROR if success is False else OutcomeKind.SUCCESS
                    ),
                    result_summary={"brief": brief},
                    error_message=brief if success is False else "",
                )
                dominant, _ = manager.observe(record)
                if dominant.hard_escalation is not None:
                    state["narrated"] = True
                    logger.warning(
                        "loop-detection hard escalation on coding task: "
                        "signature=%s count=%d",
                        dominant.signature, dominant.count,
                    )
                    alert = (
                        "Heads up -- the coding agent has repeated the same "
                        "step many times without making progress. Say stop if "
                        "you'd like me to halt it."
                    )
                    with self._loop_alert_lock:
                        if len(self._pending_loop_alerts) < 8:
                            self._pending_loop_alerts.append(alert)
                elif dominant.soft_warning is not None:
                    logger.debug(
                        "loop-detection soft warning on coding task: "
                        "signature=%s count=%d",
                        dominant.signature, dominant.count,
                    )
            except Exception as e:  # noqa: BLE001 -- never raise into the bridge
                logger.debug("loop-detection listener error: %s", e)

        return _listener

    def pop_loop_alert(self) -> Optional[str]:
        """Pop the OLDEST queued loop-detection heads-up, or ``None`` (T1).
        Drained by the orchestrator each voice-loop iteration; mirrors
        :meth:`pop_dialog_narration`."""
        with self._loop_alert_lock:
            if not self._pending_loop_alerts:
                return None
            return self._pending_loop_alerts.pop(0)

    # --- hooks lifecycle (out-of-process user hook scripts) -------------

    def _fire_task_start_hook(self, request) -> None:
        """Fire the TaskStart lifecycle hook fan-out.

        A user hook may CANCEL the task (returns ``cancel: true``) -- we raise
        :class:`RuntimeError` with the hook's message so ``start_task``'s
        caller surfaces it like any other start-task refusal. Fail-open: an
        import / discovery / execution error never blocks a task; ONLY an
        explicit cancel does. A zero-cost no-op when hooks are disabled or no
        scripts are installed (cached discovery returns an empty fan-out)."""
        try:
            from ultron.config import get_config
            if not getattr(get_config().hooks, "enabled", True):
                return
            from ultron.hooks import HookKind, HookPayload, get_hook_registry
        except Exception as e:  # noqa: BLE001
            logger.debug("hooks unavailable (%s); skipping TaskStart", e)
            return
        try:
            registry = get_hook_registry()
            payload = HookPayload(
                kind=HookKind.TASK_START,
                session_id=str(self._bound_session_id or ""),
                actor="coding",
                extra={"prompt": (getattr(request, "prompt", "") or "")[:200]},
            )
            result = registry.fire(HookKind.TASK_START, payload)
        except Exception as e:  # noqa: BLE001
            logger.debug("TaskStart hook fire failed (%s); proceeding", e)
            return
        if result.cancelled:
            msg = ""
            for r in result.per_hook_results:
                if r.outcome.cancel and r.outcome.error_message:
                    msg = r.outcome.error_message
                    break
            raise RuntimeError(
                msg or "A TaskStart hook cancelled this coding task."
            )

    def _make_hook_lifecycle_listener(self, handle):
        """Build a fail-open ``TaskEvent`` listener that fires the TaskComplete
        lifecycle hook fan-out when the task COMPLETEs (observability -- a
        finished task can't be cancelled). Returns ``None`` (a zero-cost no-op)
        when hooks are disabled or the hooks package can't be imported."""
        try:
            from ultron.config import get_config
            if not getattr(get_config().hooks, "enabled", True):
                return None
            from ultron.coding.bridge import EventKind
            from ultron.hooks import HookKind, HookPayload, get_hook_registry
        except Exception as e:  # noqa: BLE001
            logger.debug("hooks unavailable (%s); skipping TaskComplete listener", e)
            return None

        def _listener(event) -> None:
            try:
                if getattr(event, "kind", None) is not EventKind.COMPLETE:
                    return
                registry = get_hook_registry()
                payload = HookPayload(
                    kind=HookKind.TASK_COMPLETE,
                    session_id=str(self._bound_session_id or ""),
                    actor="coding",
                    extra={
                        "exit_status": getattr(event, "exit_status", None),
                        "summary": (getattr(event, "summary", "") or "")[:200],
                    },
                )
                registry.fire(HookKind.TASK_COMPLETE, payload)
            except Exception as e:  # noqa: BLE001 -- never raise into the bridge
                logger.debug("TaskComplete hook listener error: %s", e)

        return _listener

    # --- 2026 catalog 08 + 09 wiring: dialog auto-handler ---------------

    def _attach_dialog_auto_handler(self, handle):
        """Subscribe to DialogAppearedEvent for the task's lifetime.

        Returns a listener callable that the runner attaches to the
        task handle so the bus subscription gets torn down at task
        COMPLETE (otherwise the subscription would survive across
        tasks and the wrong runner would receive future events).

        Returns ``None`` when:

        * ``coding.dialog_auto_handler.enabled`` is False (operator
          opt-out), OR
        * The bus module can't be imported (no event bus -> no
          subscription possible), OR
        * The DialogPoller's singleton hasn't been started by the
          orchestrator (no events would ever fire).

        On a real dialog appearance the handler:

        1. Builds a voice-friendly one-liner ("A 'Save As' dialog
           appeared in notepad.exe -- shall I confirm?").
        2. Pushes the line into ``self._pending_dialog_narrations``.

        The orchestrator drains the queue via
        :meth:`pop_dialog_narration` each voice-loop iteration. No
        click / type happens at the runner level -- the user-facing
        narration is the entire UX. Voice routing
        (WINDOW_CLOSE_CONFIRMATION batch E) handles the actual
        yes/no when the operator decides to act.
        """
        # Config gate. Default ON because the safety + UX value is
        # high; operators can opt out via config.yaml.
        try:
            from ultron.config import get_config
            cfg = get_config().coding
            dialog_cfg = getattr(cfg, "dialog_auto_handler", None)
            if dialog_cfg is not None and not getattr(dialog_cfg, "enabled", True):
                return None
        except Exception:  # noqa: BLE001
            # Missing config means the feature is on by default --
            # we want safety nets always present in production.
            pass

        try:
            from ultron.bus import subscribe
            from ultron.bus.events import DialogAppearedEvent
        except Exception as exc:  # noqa: BLE001
            logger.debug("dialog auto-handler bus unavailable: %s", exc)
            return None

        def _on_dialog_appeared(envelope):
            try:
                props = envelope.properties
            except Exception:  # noqa: BLE001
                props = {}
            try:
                title = str(props.get("title", "") or "untitled")
                process = str(props.get("process_name", "") or "an app")
                matched_by = str(props.get("matched_by", "") or "")
            except Exception:  # noqa: BLE001
                title, process, matched_by = "untitled", "an app", ""
            # Compose voice-friendly narration. Keep under ~25 words
            # so the TTS clip stays short.
            if title.lower() in ("", "untitled", "dialog"):
                line = (
                    f"A dialog appeared in {process}. "
                    "Say yes to confirm or no to dismiss."
                )
            else:
                line = (
                    f'A "{title}" dialog appeared in {process}. '
                    "Say yes to confirm or no to dismiss."
                )
            with self._dialog_lock:
                self._pending_dialog_narrations.append(line)
            self._session_audit(
                "dialog_appeared",
                title=title,
                process=process,
                matched_by=matched_by,
                hwnd=int(props.get("hwnd", 0) or 0),
            )

        try:
            unsubscribe = subscribe(DialogAppearedEvent, _on_dialog_appeared)
        except Exception as exc:  # noqa: BLE001
            logger.debug("dialog auto-handler subscribe raised: %s", exc)
            return None

        # Stash the unsubscribe so the COMPLETE listener can tear it
        # down. On the next task's start_task the previous
        # subscription is also explicitly cleaned up just in case.
        self._teardown_previous_dialog_subscription()
        self._dialog_unsubscribe = unsubscribe

        # Return a TaskEvent listener that fires on the bridge's
        # COMPLETE event and tears down the subscription.
        from ultron.coding.bridge import EventKind

        def _teardown_listener(event):
            try:
                if event.kind == EventKind.COMPLETE:
                    self._teardown_previous_dialog_subscription()
            except Exception as exc:  # noqa: BLE001
                logger.debug("dialog teardown listener raised: %s", exc)

        return _teardown_listener

    def _teardown_previous_dialog_subscription(self) -> None:
        """Unsubscribe the prior dialog listener (if any)."""
        unsub = self._dialog_unsubscribe
        self._dialog_unsubscribe = None
        if unsub is None:
            return
        try:
            unsub()
        except Exception as exc:  # noqa: BLE001
            logger.debug("dialog unsubscribe raised: %s", exc)

    def pop_dialog_narration(self) -> Optional[str]:
        """Voice loop polls this each iteration to surface dialog
        appearance narration. Returns the OLDEST queued line once,
        then clears it. Mirrors :meth:`pop_budget_warning` /
        :meth:`pop_canonical_abort_warning` shape."""
        with self._dialog_lock:
            if not self._pending_dialog_narrations:
                return None
            return self._pending_dialog_narrations.pop(0)

    # --- 2026-05-19 Tracks 1f + 1g: AST syntax verification ---------------

    def _make_ast_syntax_listener(self, handle):
        """Build the FILE_CHANGE listener that AST-parses written
        Python files.

        Returns ``None`` when:
        * ``coding.ast_metadata.enabled`` is False (the default), OR
        * ``syntax_check_on_file_change`` is False, OR
        * The :mod:`ultron.coding.ast_metadata` module is unavailable.

        The listener never cancels the task -- it only emits audit
        rows. Completion narration can later check the rolling syntax-
        failure count to fact-check a "Done." claim, but that wiring
        is intentionally separate so we can ship the audit signal
        first and decide on UX later.

        Non-Python files (``.txt``, ``.json``, etc.) are passed
        through without parsing -- the gate is name-based.
        """
        try:
            from ultron.config import get_config
            ast_cfg = get_config().coding.ast_metadata
        except Exception as e:                                # noqa: BLE001
            logger.debug("ast_metadata config read failed: %s", e)
            return None

        if not ast_cfg.enabled or not ast_cfg.syntax_check_on_file_change:
            return None

        try:
            from ultron.coding.ast_metadata import (
                extract_metadata_from_path,
                is_python_file,
            )
            from ultron.coding.bridge import EventKind
        except Exception as e:                                # noqa: BLE001
            logger.debug(
                "ast_metadata listener unavailable (%s); skipping", e,
            )
            return None

        attach_metadata = bool(ast_cfg.attach_metadata_to_audit)

        def _listener(event):
            try:
                if getattr(event, "kind", None) != EventKind.FILE_CHANGE:
                    return
                # The TaskEvent dataclass stores the changed path under
                # ``file_path`` -- the safety listener happens to use a
                # different attribute name today, but the actual bridge
                # emit shape is ``file_path``. We accept both for
                # defensive compatibility with any future bridge that
                # populates either field.
                file_path_obj = getattr(event, "file_path", None)
                if file_path_obj is None:
                    file_path_obj = getattr(event, "path", None)
                if not file_path_obj:
                    return
                from pathlib import Path as _Path
                path = file_path_obj if isinstance(file_path_obj, _Path) else _Path(str(file_path_obj))
                path_str = str(path)
                if not is_python_file(path):
                    return
                meta = extract_metadata_from_path(path)
                if meta.syntax_valid:
                    audit_kind = "ast_syntax_ok"
                else:
                    audit_kind = "ast_syntax_failure"
                    logger.warning(
                        "AST syntax check failed for %s: %s",
                        path_str, meta.error,
                    )
                    # Track 1g voice-loop integration -- append to the
                    # per-task tracker so ``completion_narration`` can
                    # surface the count + leaf filenames. Dedupe by
                    # leaf path (Claude often rewrites the same file
                    # multiple times mid-task; only the latest matters
                    # for the user-facing summary). Strict failure-only
                    # so the tracker never grows on healthy runs.
                    leaf = path.name
                    with self._ast_lock:
                        self._ast_failures_this_task = [
                            (p, e) for (p, e) in self._ast_failures_this_task
                            if p != leaf
                        ]
                        self._ast_failures_this_task.append((leaf, meta.error))
                fields = {
                    "ts": time.time(),
                    "task_id": handle.task_id(),
                    "kind": audit_kind,
                    "path": path_str,
                    "syntax_valid": meta.syntax_valid,
                    "error": meta.error,
                    "line_count": meta.line_count,
                }
                if attach_metadata and meta.syntax_valid:
                    fields["functions_defined"] = list(meta.functions_defined)
                    fields["classes_defined"] = list(meta.classes_defined)
                    fields["imports"] = list(meta.imports)
                    fields["has_main_guard"] = meta.has_main_guard
                self._log_record(fields)
                # Also tee to session audit for the project log -- the
                # signal is most useful when correlated with the prompt
                # that prompted the file write.
                self._session_audit(audit_kind, **{
                    k: v for k, v in fields.items() if k != "ts"
                })
            except Exception as e:                            # noqa: BLE001
                # Never raise back into the bridge event loop -- a
                # broken syntax check shouldn't bring down event
                # delivery.
                logger.debug("ast_metadata listener error: %s", e)

        return _listener

    # --- 2026-05-22 catalog batch 4: pre-write lint cascade ---------------

    def _make_pre_write_lint_listener(self, handle):
        """Build the FILE_CHANGE listener that runs the catalog
        batch-4 lint cascade (tree-sitter + Python compile + flake8
        FATAL-only).

        Returns ``None`` when ``coding.pre_write_lint.enabled`` is
        False (default) or when the lint modules are unavailable.

        Listener emits ``pre_write_lint_ok`` or
        ``pre_write_lint_fail`` audit rows + tees to the session
        audit log. Never cancels the task; the signal feeds back into
        completion narration honesty checks.
        """
        try:
            from ultron.config import get_config
            cfg = get_config().coding.pre_write_lint
        except Exception as e:                                # noqa: BLE001
            logger.debug("pre_write_lint config read failed: %s", e)
            return None

        if not cfg.enabled:
            return None

        try:
            from ultron.coding.bridge import EventKind
            from ultron.coding.python_lint import lint_python
            from ultron.coding.tree_sitter_lint import tree_sitter_lint
        except Exception as e:                                # noqa: BLE001
            logger.debug(
                "pre_write_lint listener unavailable (%s); skipping", e,
            )
            return None

        run_full_python_cascade = bool(cfg.python_full_cascade)
        multi_language = bool(cfg.multi_language)
        attach_summary = bool(cfg.attach_summary_to_audit)
        flake8_timeout = float(cfg.flake8_timeout_seconds)

        def _listener(event):
            try:
                if getattr(event, "kind", None) != EventKind.FILE_CHANGE:
                    return
                file_path_obj = getattr(event, "file_path", None)
                if file_path_obj is None:
                    file_path_obj = getattr(event, "path", None)
                if not file_path_obj:
                    return
                from pathlib import Path as _Path
                path = (
                    file_path_obj
                    if isinstance(file_path_obj, _Path)
                    else _Path(str(file_path_obj))
                )
                if not path.exists() or not path.is_file():
                    return
                suffix = path.suffix.lower()
                is_python = suffix in {".py", ".pyi"}
                if is_python:
                    report = lint_python(
                        path,
                        run_flake8=run_full_python_cascade,
                        flake8_timeout=flake8_timeout,
                    )
                elif multi_language:
                    report = tree_sitter_lint(path)
                else:
                    return

                if report.skipped_reason and not report.errors:
                    # Lint couldn't actually run (no parser for this
                    # language, file unreadable, etc.). Don't emit a
                    # false-positive failure; emit a "skipped" row
                    # only for visibility.
                    self._log_record({
                        "ts": time.time(),
                        "task_id": handle.task_id(),
                        "kind": "pre_write_lint_skipped",
                        "path": str(path),
                        "language": report.language,
                        "reason": report.skipped_reason,
                    })
                    return

                if report.ok:
                    audit_kind = "pre_write_lint_ok"
                else:
                    audit_kind = "pre_write_lint_fail"
                    logger.warning(
                        "pre_write_lint failed for %s: %s",
                        path, report.summary(),
                    )
                    leaf = path.name
                    with self._ast_lock:
                        # Reuse the same per-task tracker the AST
                        # listener uses so completion narration sees
                        # one unified list of files-with-issues.
                        self._ast_failures_this_task = [
                            (p, e) for (p, e) in self._ast_failures_this_task
                            if p != leaf
                        ]
                        self._ast_failures_this_task.append(
                            (leaf, report.summary()),
                        )

                fields = {
                    "ts": time.time(),
                    "task_id": handle.task_id(),
                    "kind": audit_kind,
                    "path": str(path),
                    "language": report.language,
                    "error_count": len(report.errors),
                    "truncated": report.truncated,
                }
                if attach_summary:
                    fields["summary"] = report.summary()
                    fields["errors"] = [
                        {
                            "line": e.line,
                            "column": e.column,
                            "kind": e.kind,
                            "source": e.source,
                            "message": e.message,
                        }
                        for e in report.errors[:20]  # cap at 20
                    ]
                self._log_record(fields)
                self._session_audit(
                    audit_kind,
                    **{k: v for k, v in fields.items() if k != "ts"},
                )
            except Exception as e:                            # noqa: BLE001
                logger.debug("pre_write_lint listener error: %s", e)

        return _listener

    # --- E2 goal-anchor planning -------------------------------------------

    def _goal_anchor_config(self):
        """Read live goal-anchor config. Returns ``None`` on failure.

        Fail-open: any pydantic / import / lookup hiccup short-circuits
        the feature for the rest of the call.
        """
        try:
            from ultron.config import get_config
            return get_config().coding.goal_anchors
        except Exception as e:
            logger.debug("goal_anchors config read failed: %s", e)
            return None

    def _build_anchor_plan_and_listener(self, handle, request):
        """Build the :class:`AnchorPlan` for this task + the listener.

        Returns ``None`` when the feature is disabled or when an
        unexpected failure prevents plan construction (fail-open --
        the task still runs without anchor narration).
        """
        cfg = self._goal_anchor_config()
        if cfg is None or not cfg.enabled:
            with self._anchor_lock:
                self._anchor_plan = None
            return None

        try:
            total_budget = int(settings.CODING_TOKEN_BUDGET_PER_SESSION)
        except Exception:
            total_budget = 100_000

        try:
            plan = decompose_into_anchors(
                request.task_prompt or "",
                total_budget_tokens=total_budget,
                min_anchors=int(cfg.min_anchors),
                max_anchors=int(cfg.max_anchors),
            )
        except Exception as e:
            logger.warning("anchor decomposition failed (%s); skipping", e)
            return None

        with self._anchor_lock:
            self._anchor_plan = plan
            # Queue the opening narration BEFORE any USAGE event lands,
            # so the orchestrator's next pop_anchor_narration() poll
            # surfaces "Starting anchor 1" before "60% of anchor 1".
            opener = ""
            if plan.active is not None:
                opener = narration_for_anchor(plan.active.anchor, verb="Starting")
            if opener:
                self._pending_anchor_narration = opener

        self._log_record({
            "ts": time.time(),
            "task_id": handle.task_id(),
            "kind": "anchor_plan_created",
            "anchor_count": len(plan),
            "anchors": [a.anchor.as_dict() for a in plan.anchors],
        })

        warn_threshold = float(cfg.warn_threshold)
        return self._make_anchor_listener(handle, warn_threshold)

    def _make_anchor_listener(self, handle, warn_threshold: float):
        """Listener: route USAGE events to the active anchor budget.

        On crossing ``warn_threshold`` queues a budget-warning
        narration. On exhaustion advances to the next anchor + queues
        the next-anchor narration. Idempotent at task boundary.
        """

        def _listener(event: TaskEvent) -> None:
            try:
                if event.kind != EventKind.USAGE:
                    return
                tokens = int(
                    (event.usage_input or 0)
                    + (event.usage_output or 0)
                    + (event.usage_cache_creation or 0)
                    + (event.usage_cache_read or 0)
                )
                if tokens <= 0:
                    return

                with self._anchor_lock:
                    plan = self._anchor_plan
                    if plan is None or plan.active is None:
                        return

                    # Cascade overflow: a single USAGE event can over-
                    # fill the active anchor, in which case the
                    # leftover spills into the next anchor (and so on).
                    # Production traffic almost never exhausts in one
                    # event, but it can happen on long blocking LLM
                    # calls that emit one big USAGE record.
                    remaining = tokens
                    last_narration: Optional[str] = None
                    while remaining > 0 and plan.active is not None:
                        active = plan.active
                        pre_warn_latched = active.warning_emitted_at is not None
                        budget_left = max(
                            0,
                            active.anchor.budget_tokens - active.tokens_spent,
                        )
                        if budget_left <= 0:
                            # Already-exhausted active anchor (defensive).
                            applied = remaining
                        else:
                            applied = min(remaining, budget_left)
                        active.update(applied)
                        remaining -= applied

                        crossed_warn = (
                            not pre_warn_latched
                            and active.should_warn(threshold=warn_threshold)
                        )
                        exhausted = active.is_exhausted

                        if crossed_warn and not exhausted:
                            pct = int(round(active.utilisation * 100))
                            anchor_desc = (
                                active.anchor.description.strip().rstrip(".")
                            )
                            last_narration = (
                                f"Heads up: anchor {active.anchor.order + 1} "
                                f"({anchor_desc}) is at {pct}% of its budget."
                            )
                            self._log_record({
                                "ts": time.time(),
                                "task_id": handle.task_id(),
                                "kind": "anchor_warning",
                                "anchor_name": active.anchor.name,
                                "anchor_order": active.anchor.order,
                                "tokens_spent": active.tokens_spent,
                                "budget_tokens": active.anchor.budget_tokens,
                                "utilisation": round(active.utilisation, 3),
                            })

                        if exhausted:
                            completed = active
                            new_active = plan.advance()
                            self._log_record({
                                "ts": time.time(),
                                "task_id": handle.task_id(),
                                "kind": "anchor_completed",
                                "anchor_name": completed.anchor.name,
                                "anchor_order": completed.anchor.order,
                                "tokens_spent": completed.tokens_spent,
                                "budget_tokens": completed.anchor.budget_tokens,
                                "utilisation": round(completed.utilisation, 3),
                            })
                            if new_active is not None:
                                last_narration = narration_for_anchor(
                                    new_active.anchor, verb="Moving to"
                                )
                                self._log_record({
                                    "ts": time.time(),
                                    "task_id": handle.task_id(),
                                    "kind": "anchor_started",
                                    "anchor_name": new_active.anchor.name,
                                    "anchor_order": new_active.anchor.order,
                                    "budget_tokens": new_active.anchor.budget_tokens,
                                })
                            else:
                                last_narration = narration_for_completion(plan)
                                self._log_record({
                                    "ts": time.time(),
                                    "task_id": handle.task_id(),
                                    "kind": "anchor_plan_completed",
                                    "total_anchors": len(plan),
                                })
                                # No more anchors; drop any overflow.
                                break
                        else:
                            # Not exhausted means we've consumed the
                            # full remaining budget of this anchor and
                            # there's nothing left to cascade.
                            break
                    if last_narration is not None:
                        self._pending_anchor_narration = last_narration
            except Exception as e:
                # The listener must NEVER raise back into the bridge.
                logger.debug("anchor listener error: %s", e)

        return _listener

    def _maybe_prepend_anchor_resume(self, prompt: str) -> str:
        """Conditionally prepend a "Continue with..." line to ``prompt``.

        No-op when:
        * goal-anchors are disabled in config
        * resume_prepend_next_anchor is False
        * no plan exists
        * all anchors are already completed
        Otherwise returns ``prompt`` with the next un-completed anchor's
        description prepended.
        """
        cfg = self._goal_anchor_config()
        if cfg is None or not cfg.enabled or not cfg.resume_prepend_next_anchor:
            return prompt
        nxt = self.next_unfinished_anchor()
        if nxt is None:
            return prompt
        description = nxt.description.strip().rstrip(".")
        if not description or description.lower() == "complete the task":
            # No useful resume signal; leave the prompt alone.
            return prompt
        header = f"Continue with anchor {nxt.order + 1}: {description}.\n\n"
        return header + prompt

    def pop_anchor_narration(self) -> Optional[str]:
        """Voice loop polls this each iteration to surface anchor
        narration (opening / warning / transition / completion).
        Returns the queued text once, then clears."""
        with self._anchor_lock:
            text = self._pending_anchor_narration
            self._pending_anchor_narration = None
        return text

    def current_anchor(self) -> Optional[GoalAnchor]:
        """Return the currently-active :class:`GoalAnchor` or ``None``.

        Useful for tests + the orchestrator's progress_narration to
        surface "anchor N of M" alongside the bridge-state delta.
        """
        with self._anchor_lock:
            if self._anchor_plan is None:
                return None
            active = self._anchor_plan.active
            return active.anchor if active is not None else None

    def anchor_plan_snapshot(self) -> Optional[dict]:
        """Return a JSON-shaped snapshot of the current plan + progress.

        ``None`` when no plan exists.
        """
        with self._anchor_lock:
            if self._anchor_plan is None:
                return None
            return self._anchor_plan.as_dict()

    def has_unfinished_anchors(self) -> bool:
        """Return True when an anchor plan exists with at least one
        anchor that has not yet been marked completed."""
        with self._anchor_lock:
            if self._anchor_plan is None:
                return False
            return not self._anchor_plan.all_completed

    def next_unfinished_anchor(self) -> Optional[GoalAnchor]:
        """Return the next un-completed :class:`GoalAnchor`, or ``None``.

        Used by the resume path to prepend "continue with: <desc>" to
        follow-up prompts so the LLM picks up at the right milestone.
        """
        with self._anchor_lock:
            if self._anchor_plan is None:
                return None
            for budget in self._anchor_plan.anchors:
                if not budget.completed:
                    return budget.anchor
        return None

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
