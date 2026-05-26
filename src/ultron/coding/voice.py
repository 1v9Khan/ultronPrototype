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
from typing import Any, Callable, Dict, List, Optional

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
    """Result returned to the orchestrator from :meth:`handle_utterance`.

    A4 pre-task confirmation: when a CODE_TASK intent fires AND the
    feature is enabled, the controller defers the actual bridge spawn
    via :attr:`deferred_dispatch` and asks the orchestrator to speak
    :attr:`pre_task_confirmation` first. The orchestrator's barge-in
    watcher gives the user a window to interrupt before files start
    moving. On barge-in, ``deferred_dispatch`` is dropped without firing.
    """

    text: str  # what to speak
    handled: bool = True  # if False, orchestrator should fall through to LLM
    cancelled: bool = False  # set when we cancelled a running task
    # A4: optional confirmation phrase spoken BEFORE deferred_dispatch.
    pre_task_confirmation: Optional[str] = None
    # A4: deferred dispatch closure. Called by the orchestrator AFTER
    # the confirmation TTS completes without barge-in.
    deferred_dispatch: Optional[Callable[[], None]] = None
    # A4: tag used in the audit log for traceability.
    pre_task_label: Optional[str] = None


class CapabilityVoiceController:
    """Voice-side facade over the capability layer.

    Renamed from ``CodingVoiceController`` in Foundation Phase 5 — the
    legacy name is preserved as a module-level alias at the bottom of
    this file for backward compatibility, so existing imports
    ``from ultron.coding import CodingVoiceController`` keep working.

    The controller dispatches utterances across:
      * coding intents (existing path; routes to :class:`CodingTaskRunner`)
      * OpenClaw-bound capabilities (BROWSER / MEDIA / MESSAGING /
        FILE / SHELL) — routed to :class:`OpenClawDispatcher` via the
        :meth:`handle_capability_intent` method.
      * conversational utterances (return None; orchestrator handles
        normally)

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
        llm_engine=None,
        openclaw_bridge=None,
        gaming_mode_manager=None,
        supervisor_dispatch=None,
        project_index=None,
    ) -> None:
        self.runner = runner
        self.registry = registry
        self.resolver = resolver
        self.sandbox_root = Path(sandbox_root)
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        self.coordinator = coordinator
        # 4B plan voice-driven swap — when set, MODEL_SWITCH intents
        # call ``llm_engine.reload_for_preset(target)``. None disables
        # the feature (returns a clear voice error rather than crash).
        self.llm_engine = llm_engine
        # Phase 4 — when set, the OpenClawDispatcher used for MESSAGING
        # / BROWSER / etc. intents gets the bridge handle so it can
        # invoke real Gateway calls instead of returning the stub
        # voice messages. None disables the live path; stubs return
        # exactly as before.
        self.openclaw_bridge = openclaw_bridge
        # V1-gap A1 — gaming-mode manager. None disables gaming mode
        # with a clear voice message; otherwise GAMING_MODE intents
        # route to it via the OpenClawDispatcher.
        self.gaming_mode_manager = gaming_mode_manager
        # 2026-05-22 supervisor stack -- when set, the controller
        # routes CODE_TASK / MID_SESSION_ADJUSTMENT through the
        # supervisor BEFORE falling through to the legacy resolver
        # path. None disables the supervisor entirely (legacy path
        # is byte-for-byte unchanged).
        self.supervisor_dispatch = supervisor_dispatch
        self.project_index = project_index
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

    def pending_budget_warning(self) -> Optional[str]:
        """Phase 7: voice-loop poll for token-budget warnings raised by
        the runner. Returns once and clears, mirroring the
        ``pending_completion`` / ``pending_clarifications`` pattern."""
        try:
            return self.runner.pop_budget_warning()
        except Exception as e:
            logger.debug("pop_budget_warning failed: %s", e)
            return None

    def pending_canonical_abort(self) -> Optional[str]:
        """4B plan Item 7: voice-loop poll for canonical-path-monitor
        abort narration raised by the runner. Returns once and clears.
        Mirrors :meth:`pending_budget_warning`."""
        try:
            return self.runner.pop_canonical_abort_warning()
        except Exception as e:
            logger.debug("pop_canonical_abort_warning failed: %s", e)
            return None

    def pending_anchor_narration(self) -> Optional[str]:
        """E2 goal-anchor planning: voice-loop poll for anchor-lifecycle
        narration raised by the runner (opening / warning / transition /
        completion). Returns once and clears. Mirrors
        :meth:`pending_budget_warning`. No-op when goal-anchors are
        disabled in config -- the runner never queues anything in that
        case so the pop returns ``None`` cheaply.
        """
        try:
            return self.runner.pop_anchor_narration()
        except Exception as e:
            logger.debug("pop_anchor_narration failed: %s", e)
            return None

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

        # 2026-05-22 supervisor route: when the supervisor stack is
        # wired AND the master flag is on, intercept BEFORE the legacy
        # resolver path. The supervisor handles edit-vs-new
        # disambiguation, narration + barge-in, and enriched dispatch
        # context. On FALLBACK (or any supervisor error), drops
        # through to the legacy resolver path unchanged.
        if self.supervisor_dispatch is not None:
            supervisor_response = self._handle_code_task_via_supervisor(intent)
            if supervisor_response is not None:
                return supervisor_response

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
            return self._build_code_task_response(
                project_path=project_path,
                intent=intent,
                label=resolution.project.name,
                post_dispatch_text=(
                    f"Working on {resolution.project.name}. "
                    f"I'll let you know when it's done."
                ),
                project_phrase=resolution.project.name,
                is_new=False,
            )

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
            return self._build_code_task_response(
                project_path=Path(project.path),
                intent=intent,
                label=project_name,
                post_dispatch_text=(
                    f"Starting a new project, {project_name}, in the sandbox. "
                    f"Working on it now."
                ),
                project_phrase=project_name,
                is_new=True,
            )

        # Fallback (shouldn't reach here in practice).
        return VoiceResponse(text=(
            "I couldn't figure out which project you meant. "
            "Say create a new project or name an existing one."
        ))

    # --- 2026-05-22 supervisor route --------------------------------------

    def _handle_code_task_via_supervisor(
        self, intent: CodingIntent,
    ) -> Optional[VoiceResponse]:
        """Route a CODE_TASK through the supervisor when wired.

        Returns:
            * A :class:`VoiceResponse` to deliver to the user when the
              supervisor produced an actionable outcome (EDIT_DISPATCH,
              NEW_DISPATCH, RESUME_FORWARD, CLARIFY, BARGED_IN).
            * ``None`` when the supervisor returned FALLBACK (or any
              error). The caller (``_handle_code_task``) drops through
              to the legacy ProjectResolver path.

        Failure mode = always fail-open. Supervisor crashes never
        leave the user in a half-state.
        """
        from ultron.coding.project_supervisor import (
            SupervisorAction,
            SupervisorInputs,
        )
        from ultron.coding.supervisor_dispatch import DispatchActionKind

        try:
            inputs = SupervisorInputs(
                user_text=intent.task_text or "",
                coding_intent=intent,
                has_active_task=self.runner.has_active_task(),
                active_task_project_name=self._current_project_name(),
                active_task_session_id=self._current_session_id_or_label(),
            )
            outcome = self.supervisor_dispatch.dispatch(inputs)
        except Exception as e:                                      # noqa: BLE001
            logger.warning(
                "Supervisor dispatch raised (%s); falling back to "
                "legacy code-task path.", e,
            )
            return None

        if outcome.kind == DispatchActionKind.FALLBACK:
            return None
        if outcome.kind == DispatchActionKind.BARGED_IN:
            return VoiceResponse(
                text="Cancelled.", cancelled=True,
            )
        if outcome.kind == DispatchActionKind.CLARIFY:
            return VoiceResponse(
                text=outcome.clarification_question or
                outcome.voice_message or
                "I'm not sure which project you mean. Could you say its name?",
            )
        if outcome.kind == DispatchActionKind.RESUME_FORWARD:
            # The user said "now add error handling" -- forward as a
            # follow-up to the in-flight Claude session via the
            # runner's send_followup machinery (same path the legacy
            # _handle_adjustment uses).
            try:
                self.runner.send_followup(
                    intent.task_text or "",
                    kind="adjustment",
                )
            except Exception as e:                                  # noqa: BLE001
                logger.warning(
                    "Supervisor RESUME_FORWARD: send_followup failed (%s); "
                    "speaking the decision but not forwarding.", e,
                )
            return VoiceResponse(
                text=outcome.voice_message or "Got it. Telling Claude.",
            )
        if outcome.kind in (
            DispatchActionKind.EDIT_DISPATCH,
            DispatchActionKind.NEW_DISPATCH,
        ):
            if outcome.task_request is None:
                return None
            return self._dispatch_supervisor_task(intent, outcome)
        # Defensive: unknown outcome kind -> legacy fallback.
        return None

    def _dispatch_supervisor_task(
        self, intent: CodingIntent, outcome,
    ) -> VoiceResponse:
        """Dispatch a TaskRequest built by the supervisor.

        Differs from :meth:`_build_code_task_response` in that the
        TaskRequest is already enriched + narration already played.
        We just touch the registry (so the project's last_accessed
        updates) + register a digest listener (Phase A) + start the
        task.
        """
        request = outcome.task_request
        if request is None:
            return VoiceResponse(text="No task to dispatch.")

        # Ensure the cwd exists for NEW dispatch.
        try:
            request.cwd.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(
                "Supervisor: could not create cwd %s (%s)", request.cwd, e,
            )
            return VoiceResponse(
                text=f"I couldn't prepare the project folder. {e}",
            )

        decision = outcome.decision
        project_name = (
            decision.target_project_name if decision else None
        ) or request.cwd.name

        # Touch the registry when the project is registered. Mirrors
        # the legacy path.
        try:
            self.registry.touch(project_name)
        except Exception:                                           # noqa: BLE001
            pass

        # Start the task and attach a digest listener for COMPLETE.
        try:
            handle = self.runner.start_task(request)
        except Exception as e:                                      # noqa: BLE001
            logger.warning("Supervisor: runner.start_task failed (%s)", e)
            return VoiceResponse(
                text=f"I couldn't start the coding task. {e}",
            )

        self._attach_supervisor_digest_listener(
            handle=handle,
            project_name=project_name,
            project_path=request.cwd,
            user_goal_hint=intent.task_text or "",
        )
        # SWE-Agent T7: voice-lock review on COMPLETE.
        self._attach_submit_review_listener(
            handle=handle,
            project_name=project_name,
            project_path=request.cwd,
        )

        # Voice message: the supervisor already narrated (if narrate_enabled);
        # outcome.voice_message is "" in that case. Otherwise narrate now.
        if outcome.already_narrated:
            return VoiceResponse(text="")
        text = outcome.voice_message or (
            f"Working on {project_name}. I'll let you know when it's done."
        )
        return VoiceResponse(text=text)

    def _attach_submit_review_listener(
        self,
        handle,
        *,
        project_name: str,
        project_path: Path,
    ) -> None:
        """Register a COMPLETE listener that runs SubmitReviewLoop checks.

        Passive review: detects voice-lock hits in the session diff +
        records audit rows + queues a voice narration when a locked
        file was touched. Doesn't run the full interactive loop --
        that's for explicit-invocation paths. Fail-open: listener
        registration failures log WARN.

        2026-05-26 (production-wiring batch 6): the canonical
        wire-point for SWE-Agent T7 (SubmitReviewLoop).
        """
        try:
            from ultron.coding.bridge import EventKind
            from ultron.coding.submit_review import detect_voice_lock_hits
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "submit_review listener unavailable (%s); skipping", e,
            )
            return

        controller_ref = self

        def _review_listener(event) -> None:
            try:
                if event.kind != EventKind.COMPLETE:
                    return
                files_touched: list[str] = []
                for attr in ("files_created", "files_modified", "files_deleted"):
                    for path in (getattr(event, attr, None) or []):
                        try:
                            files_touched.append(str(path))
                        except Exception:  # noqa: BLE001
                            continue
                if not files_touched:
                    return
                hits = detect_voice_lock_hits(files_touched)
                summary = (
                    f"submit-review: {len(files_touched)} file(s); "
                    f"voice-lock hits={len(hits)}"
                )
                logger.info(summary)
                if hits:
                    msg = (
                        "Voice-baseline contract: the session touched "
                        + ", ".join(Path(h).name for h in hits[:3])
                        + (f" (+{len(hits)-3} more)" if len(hits) > 3 else "")
                        + ". Review before continuing."
                    )
                    logger.warning(
                        "submit_review voice-lock hits in %s: %s",
                        project_name, hits,
                    )
                    try:
                        # Queue a voice narration the orchestrator can
                        # drain on the next idle window.
                        controller_ref._pending_completion = (
                            controller_ref._pending_completion or ""
                        ) + ("\n" if controller_ref._pending_completion else "") + msg
                    except Exception as e:  # noqa: BLE001
                        logger.debug(
                            "submit_review narration queue failed: %s", e,
                        )
            except Exception as e:  # noqa: BLE001
                logger.debug("submit_review listener error: %s", e)

        try:
            handle.add_listener(_review_listener)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Could not attach submit_review listener (%s)", e,
            )

    def _attach_supervisor_digest_listener(
        self,
        handle,
        project_name: str,
        project_path: Path,
        user_goal_hint: str,
    ) -> None:
        """Register a COMPLETE listener that builds + upserts a digest.

        Fail-open: any exception during listener registration is
        logged WARN and ignored. The task still runs to completion;
        we just won't get a digest for it.
        """
        try:
            from ultron.config import get_config
            sup_cfg = get_config().coding.supervisor
        except Exception:                                           # noqa: BLE001
            return
        if not sup_cfg.digests_enabled:
            return
        if self.supervisor_dispatch is None or self.project_index is None:
            return

        from ultron.coding.bridge import EventKind

        index_ref = self.project_index
        dispatch_ref = self.supervisor_dispatch
        llm_engine = self.llm_engine

        prior_digest = ""
        try:
            existing = index_ref.get_by_path(project_path)
            if existing is not None:
                prior_digest = existing.digest_markdown
        except Exception:                                           # noqa: BLE001
            prior_digest = ""

        def _digest_listener(event) -> None:
            if event.kind != EventKind.COMPLETE:
                return
            try:
                files_created = list(event.files_created or [])
                files_modified = list(event.files_modified or [])
                summary = event.summary or ""

                # Build a tiny LLM-callable closure when an llm is wired.
                llm_call = _build_supervisor_llm_call(
                    llm_engine, sup_cfg,
                )

                digest = dispatch_ref.build_digest(
                    project_name=project_name,
                    project_path=project_path,
                    task_summary=summary,
                    files_created=files_created,
                    files_modified=files_modified,
                    files_deleted=[],
                    llm_call=llm_call,
                    prior_digest_markdown=prior_digest,
                    user_goal_hint=user_goal_hint,
                )
                index_ref.upsert(
                    digest,
                    language=digest.sections.get("Language", "") or "",
                    last_session_id=getattr(handle, "claude_session_id", None),
                )
            except Exception as e:                                  # noqa: BLE001
                logger.warning(
                    "supervisor digest listener failed (%s); skipping.", e,
                )

        try:
            handle.add_listener(_digest_listener)
        except Exception as e:                                      # noqa: BLE001
            logger.warning(
                "Could not attach supervisor digest listener (%s)", e,
            )

    def _current_project_name(self) -> Optional[str]:
        """Best-effort: what project the current Claude session is on."""
        try:
            state = self.runner.active_state()
        except Exception:                                           # noqa: BLE001
            return None
        if state is None:
            return None
        return getattr(state, "label", None) or None

    def _current_session_id_or_label(self) -> Optional[str]:
        try:
            state = self.runner.active_state()
        except Exception:                                           # noqa: BLE001
            return None
        if state is None:
            return None
        return getattr(state, "label", None) or None

    # --- A4 pre-task confirmation -----------------------------------------

    def _build_code_task_response(
        self,
        *,
        project_path: Path,
        intent: CodingIntent,
        label: str,
        post_dispatch_text: str,
        project_phrase: str,
        is_new: bool,
    ) -> VoiceResponse:
        """Either dispatch immediately (legacy path) or defer dispatch
        behind a spoken confirmation (A4 path).

        When ``coding.pre_task_confirmation_enabled`` is False (default),
        behaves exactly like the legacy path: ``_submit`` runs synchronously
        and the returned VoiceResponse carries only ``text``. When True,
        ``_submit`` is wrapped into a ``deferred_dispatch`` closure the
        orchestrator runs after the confirmation TTS clears its barge-in
        watch.
        """
        if not getattr(settings, "CODING_PRE_TASK_CONFIRMATION_ENABLED", False):
            self._submit(project_path, intent, label=label)
            return VoiceResponse(text=post_dispatch_text)

        confirmation = self._build_pre_task_confirmation(
            intent=intent,
            project_phrase=project_phrase,
            is_new=is_new,
        )

        # Capture the values _submit needs in a closure so the actual
        # bridge spawn can fire later without the controller reaching
        # back through `self`. Errors from _submit translate to a queued
        # warning the orchestrator can surface; we never crash the voice
        # loop on dispatch failure.
        def _dispatch() -> None:
            try:
                self._submit(project_path, intent, label=label)
            except Exception as e:                                        # noqa: BLE001
                logger.warning(
                    "deferred dispatch for %r failed: %s", label, e,
                )
        return VoiceResponse(
            text=post_dispatch_text,
            pre_task_confirmation=confirmation,
            deferred_dispatch=_dispatch,
            pre_task_label=label,
        )

    def _build_pre_task_confirmation(
        self,
        *,
        intent: CodingIntent,
        project_phrase: str,
        is_new: bool,
    ) -> str:
        """Render a short spoken confirmation in Ultron's voice.

        Format mirrors the V1 spec example:
            "I'll have AI coding agent <verb> on the <project> project. Going ahead."

        We extract a short verb/object phrase from the intent text. When
        the intent is for a brand-new project we say "scaffold a new"
        rather than "work on" so the user knows a new directory is about
        to be created.
        """
        max_words = int(getattr(settings, "CODING_PRE_TASK_MAX_WORDS", 30))
        action_phrase = self._summarise_intent_for_voice(
            intent_text=intent.task_text or "",
            max_words=max(8, max_words // 2),
        )
        if is_new:
            return (
                f"I'll have AI coding agent scaffold a new {project_phrase} "
                f"project: {action_phrase}. Going ahead."
            )
        return (
            f"I'll have AI coding agent {action_phrase} on the "
            f"{project_phrase} project. Going ahead."
        )

    @staticmethod
    def _summarise_intent_for_voice(*, intent_text: str, max_words: int) -> str:
        """Trim the intent text to a short phrase for the confirmation.

        Strips leading filler ("can you", "please", "i need you to"),
        clamps to ``max_words`` words, and ensures the result reads as
        a verb phrase (no trailing punctuation).
        """
        text = (intent_text or "").strip()
        if not text:
            return "make the requested change"
        # Strip filler.
        lowered = text.lower()
        for prefix in (
            "can you ", "could you ", "please ", "i need you to ",
            "i want you to ", "i'd like you to ", "go ahead and ",
            "go and ", "now ", "ok ", "okay ",
        ):
            if lowered.startswith(prefix):
                text = text[len(prefix):]
                lowered = text.lower()
                break
        # Clamp.
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words]) + "..."
        # Drop trailing terminator so it joins naturally.
        return text.rstrip(".!? ").rstrip(",;: ")

    def _submit(self, project_path: Path, intent: CodingIntent, label: str) -> None:
        # Phase 7: when the coordinator's session store is available,
        # register a ProjectSession so token usage + audit-log entries
        # have something to attach to. Falls back to the legacy
        # bridge-only path when no coordinator is wired.
        bound_session_id: Optional[str] = None
        if self.coordinator is not None:
            store = getattr(self.coordinator, "store", None)
            if store is not None:
                try:
                    session = store.create(
                        project_root=project_path,
                        user_intent=intent.task_text,
                        mode="edit" if not intent.is_new_project else "new",
                        model=settings.CODING_CLAUDE_MODEL,
                    )
                    bound_session_id = session.session_id
                    # Move from PLANNING -> EXECUTING so progress queries
                    # render the right status.
                    try:
                        from ultron.coding.session import SessionStatus
                        store.transition(
                            bound_session_id, SessionStatus.EXECUTING,
                        )
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning(
                        "session create failed (%s); proceeding without one", e,
                    )
        try:
            self.runner.bind_session(bound_session_id)
        except Exception:
            pass
        # 2026-05-11 token-efficiency: was hardcoded require_testing=True,
        # which prepended a heavy "MUST write tests, run, fix, re-run"
        # preamble that tripled-to-quintupled the token spend on small
        # voice-dispatched utilities. Now config-driven; default False
        # (config.coding.voice_task_require_testing). Users who want
        # the testing mandate can flip it; the orchestrator's
        # correction-loop path (runner.py) passes require_testing
        # explicitly and isn't affected.
        try:
            from ultron.config import get_config
            voice_require_testing = bool(
                get_config().coding.voice_task_require_testing
            )
        except Exception:
            voice_require_testing = False
        request = TaskRequest(
            task_prompt=intent.task_text,
            cwd=project_path,
            model=settings.CODING_CLAUDE_MODEL,
            skip_permissions=settings.CODING_SKIP_PERMISSIONS,
            require_testing=voice_require_testing,
            timeout_s=float(settings.CODING_TASK_TIMEOUT_S),
            label=label,
        )
        self.runner.start_task(request)
        with self._lock:
            self._was_active = True
            self._pending_completion = None

    # --- 4B plan: voice-driven LLM model switch ----------------------------

    def _handle_model_switch(self, routing_intent) -> "VoiceResponse":
        """Handle a MODEL_SWITCH routing intent.

        Calls ``self.llm_engine.reload_for_preset(target)`` and shapes
        the result into a single VoiceResponse. When ``llm_engine`` is
        None (e.g. tests that don't construct a real engine), reports
        the misconfiguration via voice rather than crashing.
        """
        from ultron.openclaw_routing import get_routing_log

        target = None
        if routing_intent.model_switch_intent is not None:
            target = routing_intent.model_switch_intent.target_preset

        if target is None:
            get_routing_log().record(
                routing_intent,
                handler="voice.model_switch",
                outcome="failed",
                extra={"error": "no target preset on intent"},
            )
            return VoiceResponse(
                text="I couldn't tell which model you meant.",
                handled=True,
            )

        if self.llm_engine is None:
            get_routing_log().record(
                routing_intent,
                handler="voice.model_switch",
                outcome="failed",
                extra={"error": "llm_engine not wired", "target": target},
            )
            return VoiceResponse(
                text=(
                    "I can't switch models — my engine isn't wired to "
                    "accept reloads. Restart Ultron with the new preset "
                    "instead."
                ),
                handled=True,
            )

        # Pretty label for the spoken response — "the 4B" / "the 9B".
        label = self._preset_voice_label(target)

        ok, msg = self.llm_engine.reload_for_preset(target)
        get_routing_log().record(
            routing_intent,
            handler="voice.model_switch",
            outcome="reloaded" if ok else "failed",
            extra={"target": target, "engine_message": msg},
        )

        if ok:
            if "already on" in msg:
                voice = f"I'm already running {label}."
            else:
                voice = f"Switched to {label}."
        else:
            voice = (
                f"I couldn't switch to {label}. "
                f"Reason: {msg}."
            )
        return VoiceResponse(text=voice, handled=True)

    def _dispatch_via_automation_runner(self, routing_intent) -> "VoiceResponse":
        """Shared dispatch path for all automation-bound kinds.

        Builds (or reuses) the singleton :class:`AutomationTaskRunner`,
        submits the intent, and awaits the completion narration. Threads
        the live :class:`LLMEngine`, :class:`OpenClawBridge`, and
        gaming-mode manager (V1-gap A1) through to the dispatcher so all
        of them apply uniformly.

        The runner is constructed lazily on first use so unit tests
        without these dependencies aren't forced to instantiate them.
        Routing-log outcome is derived from the dispatch result's
        ``stub`` / ``blocked`` metadata so existing tests (which assert
        ``outcome == "stub"`` for unwired automation kinds) keep
        working.
        """
        from ultron.openclaw_routing.intents import RoutingIntentKind
        from ultron.openclaw_routing import (
            AutomationTaskRunner,
            OpenClawDispatcher,
            get_routing_log,
        )
        import asyncio

        runner = getattr(self, "_automation_runner", None)
        if runner is None:
            runner = AutomationTaskRunner(
                dispatcher=OpenClawDispatcher(
                    llm=self.llm_engine,
                    bridge=self.openclaw_bridge,
                    gaming_mode_manager=getattr(
                        self, "gaming_mode_manager", None,
                    ),
                ),
            )
            self._automation_runner = runner

        async def _go():
            task_id = await runner.submit_task(routing_intent)
            return task_id, await runner.completion_narration(task_id)

        try:
            task_id, voice = asyncio.run(_go())
        except Exception as e:                                       # noqa: BLE001
            logger.warning(
                "automation runner raised on %s: %s",
                routing_intent.kind.value, e,
            )
            task_id = None
            voice = "Something went wrong dispatching that. Try again."

        # Derive outcome from the dispatch result's metadata so existing
        # routing-log assertions ("outcome=stub" for unwired automation
        # paths) stay correct without forcing the new V1-gap kinds to
        # masquerade as stubs.
        outcome = "dispatched"
        extra: Dict[str, Any] = {}
        if task_id is not None:
            try:
                result = runner._results.get(task_id)              # noqa: SLF001
            except Exception:
                result = None
            if result is not None:
                meta = result.metadata or {}
                if meta.get("blocked"):
                    outcome = "blocked"
                    extra["block_reason"] = meta.get("reason")
                elif meta.get("stub"):
                    outcome = "stub"
                    extra["stub_reason"] = (
                        "OpenClaw integration not yet complete"
                    )
        else:
            outcome = "failed"

        get_routing_log().record(
            routing_intent,
            handler=f"OpenClawDispatcher.handle_{routing_intent.kind.value}",
            outcome=outcome,
            extra=extra or None,
        )
        return VoiceResponse(
            text=voice or "I couldn't run that yet.", handled=True,
        )

    @staticmethod
    def _preset_voice_label(preset: str) -> str:
        return {
            "qwen3.5-9b": "the 9B",
            "qwen3.5-4b": "the 4B",
            "josiefied-qwen3-8b": "the 8B",
            "josiefied-qwen3-4b": "the 4B",
        }.get(preset, preset)

    # --- Phase 13: system-status voice queries -----------------------------

    def _handle_system_status(self, routing_intent) -> "VoiceResponse":
        """Resolve a SYSTEM_STATUS intent via the bridge's reporter.

        Reads heartbeat alerts + active coding session listing from
        disk. Returns a brief voice narration. Fail-open: any read
        failure yields a clear "no information" response rather than
        crashing the voice pipeline.
        """
        from ultron.openclaw_routing import get_routing_log
        from ultron.openclaw_routing.intents import SystemStatusIntent

        intent = (
            routing_intent.system_status_intent
            if routing_intent.system_status_intent is not None
            else SystemStatusIntent(focus="all", raw_text=routing_intent.raw_text)
        )

        # Find an alert log we can read. Prefer the bridge's instance
        # so the path matches what the OpenClaw side is using; fall
        # back to a fresh instance from config when no bridge is wired.
        alert_log = None
        if self.openclaw_bridge is not None:
            alert_log = getattr(self.openclaw_bridge, "heartbeat_alerts", None)
        if alert_log is None:
            try:
                from ultron.config import get_config, resolve_path
                from ultron.openclaw_bridge.heartbeat_alerts import (
                    HeartbeatAlertLog,
                )
                hb_cfg = get_config().heartbeat
                alert_log = HeartbeatAlertLog(
                    resolve_path(hb_cfg.alert_log_path),
                    retention_days=hb_cfg.alert_retention_days,
                )
            except Exception as exc:                            # noqa: BLE001
                get_routing_log().record(
                    routing_intent,
                    handler="voice.system_status",
                    outcome="failed",
                    extra={"error": f"alert log unreachable: {exc}"},
                )
                return VoiceResponse(
                    text="I can't read the alert log right now.",
                    handled=True,
                )

        try:
            from ultron.openclaw_bridge.system_status import SystemStatusReporter
            reporter = SystemStatusReporter(alert_log)
            report = reporter.report(intent)
        except Exception as exc:                                # noqa: BLE001
            get_routing_log().record(
                routing_intent,
                handler="voice.system_status",
                outcome="failed",
                extra={"error": str(exc)},
            )
            return VoiceResponse(
                text="I couldn't put together a status just now.",
                handled=True,
            )

        get_routing_log().record(
            routing_intent,
            handler="voice.system_status",
            outcome="dispatched",
            extra={
                "focus": report.focus,
                "alert_count": len(report.alerts),
                "session_count": len(report.active_sessions),
            },
        )
        return VoiceResponse(
            text=report.voice_message,
            handled=True,
        )

    # --- 2026-05-12 Phase 8: native desktop automation handlers ------------

    def _handle_app_launch(self, routing_intent) -> "VoiceResponse":
        """Native APP_LAUNCH: spawn an app via :class:`AppLauncher`.

        Routes through the safety validator's Cap-2 rules (launch path,
        debug-flag detection, Temp/Downloads block). Records a learned
        preference on success so next time the same phrase is said,
        the default placement matches what worked.

        Fail-open: missing launcher module, app-not-found, validator
        block all return a clear voice message rather than raising.
        """
        from ultron.openclaw_routing import get_routing_log

        intent = routing_intent.app_launch_intent
        if intent is None:
            return VoiceResponse(
                text="I didn't catch which app you wanted opened.",
                handled=True,
            )

        # Preference lookup: if we've handled a similar phrase before,
        # use the previous placement as a fallback default when the
        # current utterance has no explicit monitor target.
        if intent.monitor_index is None and not intent.monitor_query:
            try:
                from ultron.desktop.preferences import find_preference_for_phrase

                prior = find_preference_for_phrase(intent.raw_text)
                if prior is not None and prior.monitor_index is not None:
                    # Use the prior preference's monitor + flags.
                    intent = type(intent)(
                        app_name=intent.app_name,
                        url=intent.url,
                        monitor_index=prior.monitor_index,
                        monitor_query=intent.monitor_query,
                        fullscreen=intent.fullscreen or prior.fullscreen,
                        maximize=intent.maximize or prior.maximize,
                        raw_text=intent.raw_text,
                    )
            except Exception as e:                                # noqa: BLE001
                logger.debug("preference lookup failed: %s", e)

        try:
            from ultron.desktop.voice import handle_app_launch

            result = handle_app_launch(intent)
        except Exception as e:                                    # noqa: BLE001
            get_routing_log().record(
                routing_intent,
                handler="voice.app_launch",
                outcome="failed",
                extra={"error": str(e)},
            )
            return VoiceResponse(
                text="The desktop launcher isn't available right now.",
                handled=True,
            )

        get_routing_log().record(
            routing_intent,
            handler="voice.app_launch",
            outcome="dispatched" if result.success else "failed",
            extra={
                "app_name": result.app_name,
                "monitor_index": result.monitor_index,
                "success": result.success,
            },
        )
        return VoiceResponse(text=result.voice_message, handled=True)

    def _handle_screen_context_query(self, routing_intent) -> "VoiceResponse":
        """Native SCREEN_CONTEXT_QUERY: build a screen snapshot, fold it
        into an LLM prompt, return the LLM's response.

        The prompt is the user's original utterance prefixed with the
        snapshot's ``render_for_llm()`` text. Latency is dominated by
        the VLM call (when ``include_vlm=True``) -- typically 5-8 s on
        CPU. The orchestrator hears a single text block back; streaming
        would require deeper wiring.

        Fail-open: snapshot build failure, missing LLM, or LLM error
        all return a clear voice message.
        """
        from ultron.openclaw_routing import get_routing_log

        intent = routing_intent.screen_context_intent
        if intent is None:
            return VoiceResponse(
                text="I didn't catch what you wanted me to look at.",
                handled=True,
            )

        if self.llm_engine is None:
            return VoiceResponse(
                text="I can see your screen but my language model isn't wired.",
                handled=True,
            )

        try:
            from ultron.desktop.voice import handle_screen_context_query

            sc_result = handle_screen_context_query(intent)
        except Exception as e:                                    # noqa: BLE001
            get_routing_log().record(
                routing_intent,
                handler="voice.screen_context",
                outcome="failed",
                extra={"error": str(e)},
            )
            return VoiceResponse(
                text="I couldn't read your screen just now.",
                handled=True,
            )

        if not sc_result.success:
            get_routing_log().record(
                routing_intent,
                handler="voice.screen_context",
                outcome="failed",
                extra={"error": sc_result.error or "no injection"},
            )
            return VoiceResponse(
                text="I couldn't see your screen clearly.",
                handled=True,
            )

        # Compose the LLM prompt: screen context first, then the user's
        # actual question. Ultron's system prompt + persona apply
        # normally on top.
        # 2026-05-14: lead with a hard length cap so the screen-context
        # answer stays a 1-2 sentence voice line instead of a 1235-char
        # essay (the 2026-05-13 session log got "YouTube - Google Chrome.
        # Extensions like Dark Reader and uBlock Origin are active. Tabs
        # include YouTube videos..." -- correct, but 8+ s of TTS for
        # what could have been "YouTube in Chrome.").
        question = intent.question or routing_intent.raw_text
        augmented_prompt = (
            "[Style: respond in 1-2 short sentences. Identify the "
            "foreground app + what the user is doing. No lists, no "
            "preamble.]\n\n"
            f"{sc_result.injection_text}\n\n"
            f"User question: {question}\n\n"
            "Answer the user concisely, in your normal voice, "
            "grounded in the visual context above."
        )

        try:
            # 2026-05-14: disable the <think>...</think> chain so the
            # blocking generate() call doesn't burn tokens on reasoning
            # and doesn't risk leaking thought-traces to TTS even if the
            # strip helper had a bug. Screen-context Q&A is "simple
            # conversation" by the 4B-plan thinking-mode table.
            response_text = self.llm_engine.generate(
                augmented_prompt, enable_thinking=False,
            )
        except Exception as e:                                    # noqa: BLE001
            get_routing_log().record(
                routing_intent,
                handler="voice.screen_context",
                outcome="failed",
                extra={"error": f"llm failed: {e}"},
            )
            return VoiceResponse(
                text="I can see what you're working on but I can't put it into words right now.",
                handled=True,
            )

        get_routing_log().record(
            routing_intent,
            handler="voice.screen_context",
            outcome="dispatched",
            extra={
                "used_vlm": sc_result.used_vlm,
                "elapsed_ms": round(sc_result.elapsed_ms, 1),
            },
        )
        return VoiceResponse(
            text=(response_text or "").strip(),
            handled=True,
        )

    # --- WINDOW_MOVE / WINDOW_CLOSE (2026-05-14 second-pass) ---------------

    def _handle_window_move(self, routing_intent) -> "VoiceResponse":
        """Native WINDOW_MOVE: relocate an existing window to a target
        monitor. Bypasses OpenClaw entirely."""
        from ultron.openclaw_routing import get_routing_log

        intent = routing_intent.window_move_intent
        if intent is None:
            return VoiceResponse(
                text="I didn't catch which window to move.",
                handled=True,
            )
        try:
            from ultron.desktop.voice import handle_window_move
            result = handle_window_move(intent)
        except Exception as e:                                    # noqa: BLE001
            get_routing_log().record(
                routing_intent,
                handler="voice.window_move",
                outcome="failed",
                extra={"error": str(e)},
            )
            return VoiceResponse(
                text="I couldn't move that window right now.",
                handled=True,
            )

        get_routing_log().record(
            routing_intent,
            handler="voice.window_move",
            outcome="dispatched" if result.success else "failed",
            extra={
                "window_query": intent.window_query,
                "monitor_index": result.monitor_index,
                "error": result.error,
            },
        )
        return VoiceResponse(text=result.voice_message, handled=True)

    def _handle_window_close(self, routing_intent) -> "VoiceResponse":
        """Native WINDOW_CLOSE: find a window by name and send WM_CLOSE."""
        from ultron.openclaw_routing import get_routing_log

        intent = routing_intent.window_close_intent
        if intent is None:
            return VoiceResponse(
                text="I didn't catch which window to close.",
                handled=True,
            )
        try:
            from ultron.desktop.voice import handle_window_close
            result = handle_window_close(intent)
        except Exception as e:                                    # noqa: BLE001
            get_routing_log().record(
                routing_intent,
                handler="voice.window_close",
                outcome="failed",
                extra={"error": str(e)},
            )
            return VoiceResponse(
                text="I couldn't close that window right now.",
                handled=True,
            )

        get_routing_log().record(
            routing_intent,
            handler="voice.window_close",
            outcome="dispatched" if result.success else "failed",
            extra={
                "window_query": intent.window_query,
                "error": result.error,
            },
        )
        return VoiceResponse(text=result.voice_message, handled=True)

    # --- Catalog 09 wiring -------------------------------------------------

    def _handle_active_window_query(self, routing_intent) -> "VoiceResponse":
        """ACTIVE_WINDOW_QUERY: report the foreground window's title.

        Lighter than SCREEN_CONTEXT_QUERY -- a single pywin32 probe
        (1-2 ms). The user gets the title back in voice with no UIA
        walk, capture, or VLM cost.
        """
        from ultron.openclaw_routing import get_routing_log

        try:
            from ultron.desktop.windows import get_active_window_title
            title = get_active_window_title()
        except Exception as e:                                       # noqa: BLE001
            get_routing_log().record(
                routing_intent,
                handler="voice.active_window_query",
                outcome="failed",
                extra={"error": str(e)},
            )
            return VoiceResponse(
                text="I couldn't read the active window right now.",
                handled=True,
            )

        if not title:
            get_routing_log().record(
                routing_intent,
                handler="voice.active_window_query",
                outcome="empty",
            )
            return VoiceResponse(
                text="There's no window in the foreground right now.",
                handled=True,
            )

        get_routing_log().record(
            routing_intent,
            handler="voice.active_window_query",
            outcome="dispatched",
            extra={"title": title},
        )
        # Quote the title so the TTS engine reads it as one chunk.
        return VoiceResponse(
            text=f'Your active window is "{title}".',
            handled=True,
        )

    def _handle_semantic_click(self, routing_intent) -> "VoiceResponse":
        """SEMANTIC_CLICK: click a UI element by its accessible name.

        Routes through :func:`ultron.desktop.element_click.click_element_by_name`
        which walks the foreground UIA tree and clicks via the gated
        :class:`InputController` (click-preview VLM + foreground
        security + Cap-3 explicit-intent + rate limit all apply).
        """
        from ultron.openclaw_routing import get_routing_log

        intent = routing_intent.semantic_click_intent
        if intent is None or not (intent.element_name or "").strip():
            return VoiceResponse(
                text="I didn't catch which element to click.",
                handled=True,
            )

        # Try the click. The downstream click_element_by_name handles
        # missing pywinauto / missing elements / disabled targets via
        # a structured ClickResult; we surface the voice-friendly bits.
        try:
            from ultron.desktop.element_click import click_element_by_name
            result = click_element_by_name(
                name=intent.element_name,
                window_title=intent.window_title or None,
                control_type=intent.control_type or None,
                user_text=routing_intent.raw_text or "",
            )
        except Exception as e:                                       # noqa: BLE001
            get_routing_log().record(
                routing_intent,
                handler="voice.semantic_click",
                outcome="failed",
                extra={
                    "element_name": intent.element_name,
                    "error": str(e),
                },
            )
            return VoiceResponse(
                text=(
                    f'I couldn\'t click "{intent.element_name}" right now.'
                ),
                handled=True,
            )

        outcome = "dispatched" if result.success else "failed"
        get_routing_log().record(
            routing_intent,
            handler="voice.semantic_click",
            outcome=outcome,
            extra={
                "element_name": intent.element_name,
                "window_title": intent.window_title,
                "control_type": intent.control_type,
                "result_method": getattr(result, "method", ""),
                "candidates": getattr(result, "candidates", 0),
                "error": getattr(result, "error", None),
            },
        )

        if result.success:
            target_desc = result.element_name or intent.element_name
            if result.window_title:
                voice = (
                    f'I clicked "{target_desc}" in {result.window_title}.'
                )
            else:
                voice = f'I clicked "{target_desc}".'
            return VoiceResponse(text=voice, handled=True)

        # Failure paths: differentiate "not found" from "safety blocked".
        err = (result.error or "").lower()
        if "not found" in err or "no candidate" in err or result.candidates == 0:
            voice = (
                f'I couldn\'t find anything called '
                f'"{intent.element_name}" on screen.'
            )
        elif "safety" in err or "preview" in err:
            voice = (
                f'I held off on clicking "{intent.element_name}". '
                f'{result.error or "Something looked off about the target."}'
            )
        else:
            voice = (
                f'I tried to click "{intent.element_name}" but it didn\'t '
                f'land. {result.error or ""}'
            ).strip()
        return VoiceResponse(text=voice, handled=True)

    def _handle_window_close_confirmation(
        self,
        routing_intent,
    ) -> "VoiceResponse":
        """WINDOW_CLOSE_CONFIRMATION: bare yes/no reply.

        At the controller level we only acknowledge -- the
        orchestrator owns the pending-approval registry and consumes
        the decision via its own intercept. Returning ``handled=True``
        with a short ack keeps the user-perceptible response under
        100 ms; the orchestrator's intercept fires before this handler
        when an approval IS pending.
        """
        from ultron.openclaw_routing import get_routing_log

        intent = routing_intent.window_close_confirmation_intent
        decision = (intent.decision if intent is not None else "").lower()

        get_routing_log().record(
            routing_intent,
            handler="voice.window_close_confirmation",
            outcome="no_pending_approval",
            extra={"decision": decision},
        )

        # No approval was pending in the orchestrator's state when this
        # handler fires (otherwise the orchestrator's pre-dispatch
        # intercept would have consumed the intent). Surface a neutral
        # "noted" without taking action.
        if decision == "yes":
            voice = "Got it."
        elif decision == "no":
            voice = "Okay."
        else:
            voice = "Noted."
        return VoiceResponse(text=voice, handled=True)

    # --- Phase 5 capability dispatch ---------------------------------------

    def handle_capability_intent(self, routing_intent) -> Optional[VoiceResponse]:
        """Dispatch a top-level :class:`RoutingIntent`.

        For the openclaw-bound categories (BROWSER / MEDIA / MESSAGING /
        FILE / SHELL) and HYBRID_TASK, this method calls into the
        OpenClaw dispatcher (currently stubbed) and routes the resulting
        voice message back to the orchestrator. For coding kinds it
        delegates to :meth:`handle_utterance`. CONVERSATIONAL falls
        through (returns ``None``).

        Lazy-imported to avoid pulling in the openclaw_routing module
        when the controller is constructed in tests that don't need it.
        """
        from ultron.openclaw_routing.intents import RoutingIntentKind
        from ultron.openclaw_routing import (
            AutomationTaskRunner,
            OpenClawDispatcher,
            get_routing_log,
        )
        import asyncio

        kind = routing_intent.kind
        if kind == RoutingIntentKind.CONVERSATIONAL:
            get_routing_log().record(
                routing_intent,
                handler="voice.respond",
                outcome="passthrough",
            )
            return None

        # 4B plan — voice-driven model swap. Handled in-process by
        # calling reload_for_preset on the live LLMEngine. The reload
        # blocks (~1-3s for 4B, ~3-5s for 9B) so the user hears a brief
        # silence then a confirmation. The orchestrator's barge-in
        # semantics are unaffected: the wake word will still fire on
        # subsequent utterances even if the user changes their mind
        # mid-load.
        if kind == RoutingIntentKind.MODEL_SWITCH:
            return self._handle_model_switch(routing_intent)

        # Phase 13 — system-status voice queries ("what alerts did
        # you flag?", "what is Ultron working on?"). Read from the
        # heartbeat alert log + active session listing on disk via
        # the bridge's SystemStatusReporter. No OpenClaw call.
        if kind == RoutingIntentKind.SYSTEM_STATUS:
            return self._handle_system_status(routing_intent)

        # V1-gap A1 — gaming mode (anticheat-safe shutdown of OpenClaw
        # plugins). Routes through the dispatcher so the block-and-
        # revise validator + per-call audit log apply uniformly.
        if kind == RoutingIntentKind.GAMING_MODE:
            return self._dispatch_via_automation_runner(routing_intent)

        # V1-gap C3 — desktop / windows control (UI Automation +
        # screenshot via the OpenClaw desktop-control / windows-control
        # plugins). Routes through the same automation runner so the
        # audit log + block-and-revise apply identically to other
        # OpenClaw-bound kinds.
        if kind in {
            RoutingIntentKind.DESKTOP_AUTOMATION,
            RoutingIntentKind.WINDOW_AUTOMATION,
        }:
            return self._dispatch_via_automation_runner(routing_intent)

        # 2026-05-12 Phase 8 -- native desktop automation (NOT via
        # ClawHub plugins). APP_LAUNCH routes to the native
        # AppLauncher (Chrome with default profile, app registry,
        # monitor placement). SCREEN_CONTEXT_QUERY assembles a
        # snapshot of what the user is looking at and folds it into
        # the LLM context for the next response. Both bypass
        # OpenClaw entirely so they work without a Gateway online.
        if kind == RoutingIntentKind.APP_LAUNCH:
            return self._handle_app_launch(routing_intent)
        if kind == RoutingIntentKind.SCREEN_CONTEXT_QUERY:
            return self._handle_screen_context_query(routing_intent)
        # 2026-05-14 second-pass: WINDOW_MOVE / WINDOW_CLOSE operate on
        # already-open windows (find_window + move/close). Bypass
        # OpenClaw entirely (same as APP_LAUNCH).
        if kind == RoutingIntentKind.WINDOW_MOVE:
            return self._handle_window_move(routing_intent)
        if kind == RoutingIntentKind.WINDOW_CLOSE:
            return self._handle_window_close(routing_intent)
        # Catalog 09 wiring -- light foreground-title probe and the
        # semantic-click voice command. The window-close confirmation
        # intent is handled at the orchestrator level (it needs the
        # pending-approval registry) so we surface it as
        # ``handled=True`` here so the orchestrator can intercept.
        if kind == RoutingIntentKind.ACTIVE_WINDOW_QUERY:
            return self._handle_active_window_query(routing_intent)
        if kind == RoutingIntentKind.SEMANTIC_CLICK:
            return self._handle_semantic_click(routing_intent)
        if kind == RoutingIntentKind.WINDOW_CLOSE_CONFIRMATION:
            return self._handle_window_close_confirmation(routing_intent)

        # Coding kinds — delegate to the existing utterance pipeline.
        coding_kinds = {
            RoutingIntentKind.CODE_TASK,
            RoutingIntentKind.PROGRESS_QUERY,
            RoutingIntentKind.CANCEL,
            RoutingIntentKind.MID_SESSION_ADJUSTMENT,
            RoutingIntentKind.CLARIFICATION_RESPONSE,
        }
        if kind in coding_kinds:
            response = self.handle_utterance(routing_intent.raw_text)
            get_routing_log().record(
                routing_intent,
                handler="CodingTaskRunner.handle_utterance",
                outcome="dispatched" if response else "passthrough",
            )
            return response

        # Automation kinds — dispatch through the OpenClawDispatcher.
        # The dispatcher is currently stubbed; the spoken response tells
        # the user the gateway isn't connected yet.
        if kind in {
            RoutingIntentKind.BROWSER_AUTOMATION,
            RoutingIntentKind.MEDIA_GENERATION,
            RoutingIntentKind.MESSAGING,
            RoutingIntentKind.FILE_OPERATION,
            RoutingIntentKind.SHELL_OPERATION,
        }:
            return self._dispatch_via_automation_runner(routing_intent)

        # Hybrid — without a wired-up decomposer + Anthropic, we can only
        # tell the user we recognized it but can't run it yet.
        if kind == RoutingIntentKind.HYBRID_TASK:
            voice = (
                "I can see that's a mix of coding and automation. "
                "I'd split it up and run both, but the gateway isn't "
                "connected yet."
            )
            get_routing_log().record(
                routing_intent,
                handler="HybridTaskDecomposer",
                outcome="stub",
                extra={"stub_reason": "OpenClaw integration not yet complete"},
            )
            return VoiceResponse(text=voice, handled=True)

        # Unknown kind — log and fall through.
        get_routing_log().record(
            routing_intent,
            handler="voice.unknown",
            outcome="passthrough",
        )
        return None


# ---------------------------------------------------------------------------
# Backward-compatibility alias.
#
# Foundation Phase 5 renamed CodingVoiceController -> CapabilityVoiceController
# because the controller now dispatches across capabilities, not just coding.
# Existing imports `from ultron.coding import CodingVoiceController` keep
# working via this alias. New code should prefer CapabilityVoiceController.
# ---------------------------------------------------------------------------

CodingVoiceController = CapabilityVoiceController


# ---------------------------------------------------------------------------
# 2026-05-22 supervisor stack helper -- exposed at module scope so tests can
# stub it without monkey-patching the controller instance.
# ---------------------------------------------------------------------------


def _build_supervisor_llm_call(llm_engine, sup_cfg):
    """Build the LLM-call closure passed to
    :func:`ultron.coding.project_digest.generate_digest`.

    The digest generator expects a callable that takes a prompt
    string and returns the model completion text. We wrap
    ``llm_engine.generate_text`` (or similar) so the digest call
    inherits the in-process Qwen voice model + its current preset
    without spinning up a separate inference process.

    Returns ``None`` when no llm_engine is wired or the wrapper
    can't be built -- the digest generator's fail-open path then
    produces a deterministic-template digest.
    """
    if llm_engine is None:
        return None
    # The in-process LLMEngine exposes ``generate`` (full string) and
    # ``generate_stream`` (token iterator). Use ``generate`` so the
    # digest gets the full completion.
    fn = getattr(llm_engine, "generate", None)
    if fn is None or not callable(fn):
        return None

    max_chars = int(getattr(sup_cfg, "digest_max_summary_chars", 4000) or 4000)
    # The model often outputs ~1 token per ~4 chars; cap roughly so a
    # runaway model doesn't waste minutes.
    max_tokens = max(256, min(2048, max_chars // 2))

    def _call(prompt: str) -> str:
        try:
            result = fn(
                prompt,
                max_tokens=max_tokens,
                record_history=False,
            )
        except TypeError:
            # generate() signature varies across LLMEngine variants;
            # the conservative fallback drops the kwargs.
            try:
                result = fn(prompt)
            except Exception:                                       # noqa: BLE001
                return ""
        except Exception:                                           # noqa: BLE001
            return ""
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        # Some LLM wrappers return a tuple or dict; pull a text field.
        if isinstance(result, dict):
            return str(result.get("text") or result.get("content") or "")
        return str(result)

    return _call
