"""High-level voice controller for the coding pipeline.

The orchestrator's main loop calls a small, deliberately-narrow API:

  * :meth:`pending_completion` -- 'did a coding task just finish? if so,
    give me the narration to speak'.
  * :meth:`handle_utterance` -- 'here's a transcribed user utterance.
    Should I (the orchestrator) handle it, or do you (coding) want it?
    If you take it, give me the spoken response.'

Everything in between -- intent classification, project resolution,
sandbox creation, runner submission, completion-tracking -- lives in
this controller so the orchestrator stays simple.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from config import settings
from ultron.coding.bridge import TaskRequest
from ultron.coding.intent import (
    CodingIntent,
    CodingIntentKind,
    classify as classify_intent,
    derive_project_name,
)
from ultron.coding.projects import (
    ProjectRegistry,
    ProjectResolution,
    ProjectResolver,
    ResolutionKind,
    new_sandbox_project,
)
from ultron.coding.runner import CodingTaskRunner
from ultron.utils.logging import get_logger

logger = get_logger("coding.voice")


@dataclass
class VoiceResponse:
    """Result returned to the orchestrator from :meth:`handle_utterance`."""

    text: str  # what to speak
    handled: bool = True  # if False, orchestrator should fall through to LLM
    cancelled: bool = False  # set when we cancelled a running task


class CodingVoiceController:
    """Voice-side facade over the coding pipeline.

    Args:
        runner: the :class:`CodingTaskRunner` that owns the bridge.
        registry: the project registry (CRUD).
        resolver: the :class:`ProjectResolver` (lexical + optional
            semantic).
        sandbox_root: where new projects get scaffolded.
    """

    def __init__(
        self,
        runner: CodingTaskRunner,
        registry: ProjectRegistry,
        resolver: ProjectResolver,
        sandbox_root: Path = settings.CODING_SANDBOX_PATH,
        coordinator=None,
    ) -> None:
        self.runner = runner
        self.registry = registry
        self.resolver = resolver
        self.sandbox_root = Path(sandbox_root)
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        self.coordinator = coordinator
        self._lock = threading.Lock()
        # State machine for completion-push: when has_active_task() goes
        # from True to False, we capture the completion narration so the
        # orchestrator can deliver it the next time it polls.
        self._was_active = False
        self._pending_completion: Optional[str] = None
        # Track which pending clarifications we've already spoken to the
        # user so we don't repeat the same prompt every loop iteration.
        self._announced_clarifications: set[str] = set()

    # --- public API ---------------------------------------------------------

    def pending_completion(self) -> Optional[str]:
        """If a running task transitioned to complete since last poll,
        return the completion narration; otherwise None.

        The first call after a transition consumes the message; subsequent
        calls return None until another transition occurs.
        """
        with self._lock:
            currently_active = self.runner.has_active_task()
            if self._was_active and not currently_active:
                # Capture the narration once.
                self._pending_completion = self.runner.completion_narration()
            self._was_active = currently_active
            out = self._pending_completion
            self._pending_completion = None
            return out

    def pending_clarifications(self) -> List[str]:
        """One-line voice prompts for clarifications awaiting the user.

        Each entry is consumed -- the orchestrator's main loop calls this
        once per iteration; the controller returns each prompt at most
        once. Returns an empty list when no coordinator is wired or when
        no clarifications are pending.
        """
        if self.coordinator is None:
            return []
        try:
            pending = self.coordinator.pending_user_clarifications()
        except Exception as e:
            logger.debug("pending_user_clarifications failed: %s", e)
            return []
        out: List[str] = []
        with self._lock:
            for p in pending:
                if p.request_id in self._announced_clarifications:
                    continue
                self._announced_clarifications.add(p.request_id)
                out.append(p.voice_question)
        return out

    def has_pending_clarification(self) -> bool:
        """Cheap probe used by the intent classifier."""
        if self.coordinator is None:
            return False
        try:
            return bool(self.coordinator.pending_user_clarifications())
        except Exception:
            return False

    def handle_utterance(self, text: str) -> Optional[VoiceResponse]:
        """Classify and (if coding-related) act on an utterance.

        Returns:
          * ``None`` if the utterance is not coding-related; orchestrator
            should fall through to its normal LLM path.
          * :class:`VoiceResponse` with ``handled=True`` and the text to
            speak otherwise.
        """
        if not (text or "").strip():
            return None
        if not settings.CODING_ENABLED:
            return None
        intent = classify_intent(
            text,
            has_active_task=self._has_active_task_or_session(),
            has_pending_clarification=self.has_pending_clarification(),
        )
        logger.info(
            "coding intent: %s (conf=%.2f) -- %s",
            intent.kind.value, intent.confidence, intent.reason,
        )
        if intent.kind == CodingIntentKind.NONE:
            return None
        if intent.kind == CodingIntentKind.CANCEL:
            return self._handle_cancel()
        if intent.kind == CodingIntentKind.PROGRESS_QUERY:
            return self._handle_progress()
        if intent.kind == CodingIntentKind.CLARIFICATION_RESPONSE:
            return self._handle_clarification_response(text)
        if intent.kind == CodingIntentKind.MID_SESSION_ADJUSTMENT:
            return self._handle_adjustment(text)
        if intent.kind == CodingIntentKind.CODE_TASK:
            return self._handle_code_task(intent)
        return None

    def _handle_clarification_response(self, text: str) -> VoiceResponse:
        """The user just spoke an answer to a clarification Claude asked."""
        if self.coordinator is None:
            return VoiceResponse(text="No clarification is pending.")
        pending = self.coordinator.pending_user_clarifications()
        if not pending:
            return VoiceResponse(text="No clarification is pending.")
        # Take the oldest (FIFO) pending clarification.
        target = min(pending, key=lambda p: p.raised_at)
        delivered = self.coordinator.deliver_user_clarification_response(
            target.request_id, text,
        )
        if not delivered:
            return VoiceResponse(text="That clarification has already resolved.")
        return VoiceResponse(text="Got it. Passing that to Claude.")

    def _handle_adjustment(self, text: str) -> VoiceResponse:
        """The user is asking Claude to change direction mid-task."""
        if self.coordinator is None:
            return VoiceResponse(
                text="Coordinator not available; cannot route adjustment."
            )
        # Phase 6 wiring: resolve the actual ProjectSession from the
        # coordinator's store (replaces the Phase 2 label-as-id stand-in).
        # Falls back to the runner's bridge label only when no session is
        # registered.
        session = self._current_session()
        active = self.runner.active_state()
        if session is None and active is None:
            return VoiceResponse(
                text="There's no active coding task to adjust."
            )
        session_id = (
            session.session_id if session is not None
            else (active.label or "current")
        )
        # Bridge the async coordinator call into our sync caller.
        decision = self._await_coordinator_call(
            self.coordinator.decide_adjustment,
            session_id,
            text,
        )
        if decision is None:
            return VoiceResponse(
                text="I couldn't process that adjustment right now."
            )
        if decision.action == "ESCALATE_CONFLICT":
            return VoiceResponse(text=decision.voice_question or (
                "That adjustment conflicts with completed work. "
                "Should I have him pivot or finish what's in progress?"
            ))
        # FOLLOWUP: the runner is responsible for actually sending the prompt
        # to Claude (Phase 2e wires that). For now we just acknowledge.
        if decision.followup_prompt:
            self.runner.send_followup(
                decision.followup_prompt, kind="adjustment",
            )
        return VoiceResponse(text="Got it. Telling Claude.")

    @staticmethod
    def _await_coordinator_call(coro_fn, *args, **kwargs):
        """Run a coordinator coroutine to completion from a sync caller.

        The coordinator's public methods are async because they may run
        in the MCP server's asyncio loop. Voice-controller callers are
        on the orchestrator's main thread and need synchronous results,
        so we either schedule onto an existing loop (if the coordinator
        owns one) or spin up a one-shot loop here.
        """
        import asyncio
        try:
            return asyncio.run(coro_fn(*args, **kwargs))
        except Exception as e:
            logger.warning("coordinator call failed: %s", e)
            return None

    # --- handlers -----------------------------------------------------------

    def _handle_cancel(self) -> VoiceResponse:
        if not self.runner.has_active_task():
            return VoiceResponse(text="There's no active coding task to cancel.")
        self.runner.cancel_active()
        # Wait briefly so the bridge actually tears down -- improves UX
        # when the user immediately follows up with another request.
        try:
            self.runner.wait_active(timeout=5.0)
        except Exception:
            pass
        return VoiceResponse(text="Cancelled.", cancelled=True)

    def _handle_progress(self) -> VoiceResponse:
        # Phase 5: when the coordinator (and therefore the session store)
        # is wired, look up the most-recent active session and route the
        # rich session-aware narration through the runner. Falls back to
        # legacy bridge-state narration when no session is found -- which
        # is what the pre-Phase-5 tests rely on.
        session = self._current_session()
        narration = self.runner.progress_narration(session=session)
        return VoiceResponse(text=narration)

    def _current_session(self):
        """Resolve the most-recent active :class:`ProjectSession`, or None.

        Pulls the session from the coordinator's shared store. Returns
        ``None`` when no coordinator is wired (legacy / unit-test path)
        or no session is active. Mirrors the heuristic the MCP server
        uses on the Claude side -- pick the most-recently-started active
        session.
        """
        coordinator = self.coordinator
        if coordinator is None:
            return None
        store = getattr(coordinator, "store", None)
        if store is None:
            return None
        try:
            active = store.list_active()
        except Exception:
            return None
        if not active:
            return None
        return max(active, key=lambda s: s.started_at)

    def _has_active_task_or_session(self) -> bool:
        """True if the runner has an in-flight task OR the coordinator's
        store has an active :class:`ProjectSession`.

        Phase 5: an active session counts as "the project is running"
        from the user's perspective. This lets progress queries and
        cancel/adjustment intents fire even when the legacy bridge state
        is empty (e.g., when state lives in the session store, not the
        bridge handle).
        """
        if self.runner.has_active_task():
            return True
        return self._current_session() is not None

    def _handle_code_task(self, intent: CodingIntent) -> VoiceResponse:
        # Refuse to start a second task while one is running -- the user
        # has to cancel or wait. Tells them so explicitly.
        if self.runner.has_active_task():
            state = self.runner.active_state()
            current = state.current_step if state else "earlier task"
            return VoiceResponse(text=(
                f"A coding task is already running ({current}). "
                f"Say cancel to stop it, or wait for it to finish."
            ))

        # Resolve the project.
        resolution: Optional[ProjectResolution] = None
        for ref in intent.candidates_for_resolver or []:
            r = self.resolver.resolve(ref)
            if r.kind not in {ResolutionKind.NOT_FOUND, ResolutionKind.AMBIGUOUS}:
                resolution = r
                break

        # Existing-project flow: user mentioned a project, resolver found one.
        if resolution and resolution.project is not None:
            project_path = Path(resolution.project.path)
            if not project_path.is_dir():
                return VoiceResponse(text=(
                    f"Project {resolution.project.name} is registered but its "
                    f"folder is missing at {project_path}. Aborting."
                ))
            self.registry.touch(resolution.project.name)
            self._submit(project_path, intent, label=resolution.project.name)
            return VoiceResponse(text=(
                f"Working on {resolution.project.name}. "
                f"I'll let you know when it's done."
            ))

        # Ambiguous: ask the user to disambiguate.
        if resolution and resolution.kind == ResolutionKind.AMBIGUOUS:
            names = ", ".join(p.name for p in resolution.candidates[:4])
            return VoiceResponse(text=(
                f"More than one project matched: {names}. "
                f"Say which one you mean."
            ))

        # New-project flow.
        if intent.is_new_project or resolution is None or resolution.kind == ResolutionKind.NOT_FOUND:
            project_name = derive_project_name(intent)
            try:
                project = new_sandbox_project(
                    self.registry,
                    name=project_name,
                    aliases=[project_name.lower()],
                    description=intent.task_text[:200],
                    sandbox_root=self.sandbox_root,
                )
            except ValueError:
                # Name collision -- fall through with a uniqueness suffix.
                from uuid import uuid4
                project_name = f"{project_name}_{uuid4().hex[:4]}"
                project = new_sandbox_project(
                    self.registry,
                    name=project_name,
                    aliases=[project_name.lower()],
                    description=intent.task_text[:200],
                    sandbox_root=self.sandbox_root,
                )
            self._submit(Path(project.path), intent, label=project_name)
            return VoiceResponse(text=(
                f"Starting a new project, {project_name}, in the sandbox. "
                f"Working on it now."
            ))

        # Fallback (shouldn't reach here in practice).
        return VoiceResponse(text=(
            "I couldn't figure out which project you meant. "
            "Say create a new project or name an existing one."
        ))

    def _submit(self, project_path: Path, intent: CodingIntent, label: str) -> None:
        request = TaskRequest(
            task_prompt=intent.task_text,
            cwd=project_path,
            model=settings.CODING_CLAUDE_MODEL,
            skip_permissions=settings.CODING_SKIP_PERMISSIONS,
            require_testing=True,
            timeout_s=float(settings.CODING_TASK_TIMEOUT_S),
            label=label,
        )
        self.runner.start_task(request)
        with self._lock:
            self._was_active = True
            self._pending_completion = None
