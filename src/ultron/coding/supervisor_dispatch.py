"""Supervisor dispatch -- narration + barge-in + enriched TaskRequest.

Sits on top of :class:`ultron.coding.project_supervisor.ProjectSupervisor`
and turns a :class:`SupervisorDecision` into the actual side effects:

  * Narrate the decision via TTS with a barge-in window (Phase D).
  * Build an enriched :class:`ultron.coding.bridge.TaskRequest` that
    includes the project digest + file-tree summary + entry-point
    hints in the prompt body (Phase E).
  * Return a :class:`DispatchOutcome` describing what the orchestrator
    should do next: dispatch the task, ask a clarification, resume
    the in-flight session, or abort (barge-in / error).

The orchestrator wires this controller once at startup and feeds it
the supervisor's decision per turn.

Threading: synchronous on the call site. Barge-in waits use the
orchestrator's existing wake-word detector via a callable passed in
at construction time.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional

from ultron.coding.bridge import TaskRequest
from ultron.coding.project_digest import (
    DigestRequest,
    LLMCallable,
    ProjectDigest,
    generate_digest,
)
from ultron.coding.project_index import ProjectIndex, ProjectIndexEntry
from ultron.coding.project_introspect import ProjectSnapshot, snapshot
from ultron.coding.project_supervisor import (
    ProjectSupervisor,
    SupervisorAction,
    SupervisorDecision,
    SupervisorInputs,
)

logger = logging.getLogger("ultron.coding.supervisor_dispatch")


# ---------------------------------------------------------------------------
# Outcome model
# ---------------------------------------------------------------------------


class DispatchActionKind(str, Enum):
    """What the orchestrator should do with the dispatch outcome."""

    # Spawn a fresh Claude session with the enriched TaskRequest.
    EDIT_DISPATCH = "edit_dispatch"

    # Spawn a fresh Claude session for a brand-new project scaffold.
    NEW_DISPATCH = "new_dispatch"

    # Forward as a follow-up to the in-flight Claude session.
    RESUME_FORWARD = "resume_forward"

    # Ask the user a disambiguation question; no Claude call yet.
    CLARIFY = "clarify"

    # Barge-in fired during narration -- drop everything, re-listen.
    BARGED_IN = "barged_in"

    # Supervisor / dispatch error -- caller should fall back to the
    # legacy non-supervisor path.
    FALLBACK = "fallback"


@dataclass
class DispatchOutcome:
    """Result of running the supervisor dispatch for one decision.

    Attributes:
        kind: high-level next action.
        voice_message: text the orchestrator should speak before
            acting (e.g. acknowledgment, clarification question). When
            the narration already happened (e.g. EDIT_DISPATCH with
            narrate_enabled), this is set to "" so the orchestrator
            doesn't double-speak.
        task_request: enriched :class:`TaskRequest` for the dispatch
            paths. ``None`` for RESUME / CLARIFY / BARGED_IN / FALLBACK.
        clarification_question: set when ``kind=CLARIFY``.
        resume_session_id: set when ``kind=RESUME_FORWARD``; this is
            the Claude session id to forward to.
        decision: the originating :class:`SupervisorDecision` for
            audit / logging.
        already_narrated: True when the supervisor already spoke its
            narration -- prevents the orchestrator from re-speaking
            voice_message.
    """

    kind: DispatchActionKind
    voice_message: str = ""
    task_request: Optional[TaskRequest] = None
    clarification_question: Optional[str] = None
    resume_session_id: Optional[str] = None
    decision: Optional[SupervisorDecision] = None
    already_narrated: bool = False


# ---------------------------------------------------------------------------
# Type aliases for injected dependencies
# ---------------------------------------------------------------------------


# Speak text + return True iff the wake-word fired during / shortly
# after playback ("barge-in"). Implemented by the orchestrator's
# existing :meth:`_speak_with_barge_in_check` -- the supervisor
# controller doesn't need to know about TTS internals.
BargeInCheckable = Callable[[str], bool]

# Plain non-barge-in TTS speak. Used to announce things that aren't
# barge-in-eligible (e.g. clarification questions go through this).
PlainSpeak = Callable[[str], None]


# ---------------------------------------------------------------------------
# SupervisorDispatchController
# ---------------------------------------------------------------------------


class SupervisorDispatchController:
    """Coordinator for the supervisor + narration + enriched dispatch.

    Constructor takes the supervisor + index + speak callbacks. The
    orchestrator constructs ONE instance at startup and calls
    :meth:`dispatch` per supervisor-eligible turn.
    """

    def __init__(
        self,
        supervisor: ProjectSupervisor,
        *,
        index: Optional[ProjectIndex] = None,
        barge_in_speak: Optional[BargeInCheckable] = None,
        plain_speak: Optional[PlainSpeak] = None,
        narrate_enabled: bool = False,
        narration_barge_in_window_seconds: float = 1.5,
        enriched_context_enabled: bool = False,
        sandbox_root: Optional[Path] = None,
        default_model: str = "haiku",
        # 2026-05-22 catalog batch 14 (T5 Phase 2). Optional callable that
        # narrates the architect's plan via TTS with sentence-boundary
        # barge-in. Signature: ``(plan_text) -> bool`` where True means
        # the user interrupted -- dispatch returns BARGED_IN. Wired by
        # the orchestrator when ``coding.architect.narrate_enabled`` is on.
        # Fail-open: any exception is treated as "no interrupt", so the
        # dispatch proceeds.
        architect_narrator: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.supervisor = supervisor
        self.index = index
        self._barge_in_speak = barge_in_speak
        self._plain_speak = plain_speak
        self.narrate_enabled = bool(narrate_enabled)
        self.narration_barge_in_window_seconds = float(
            max(0.0, narration_barge_in_window_seconds),
        )
        self.enriched_context_enabled = bool(enriched_context_enabled)
        self.sandbox_root = (
            Path(sandbox_root) if sandbox_root is not None else None
        )
        self.default_model = default_model
        self._architect_narrator = architect_narrator

    # --- public API ---------------------------------------------------------

    def dispatch(self, inputs: SupervisorInputs) -> DispatchOutcome:
        """Run the full supervisor dispatch pipeline.

        Step-by-step:
          1. supervisor.decide() -> SupervisorDecision
          2. Build a brief narration phrase from the decision.
          3. If narrate_enabled: speak it with barge-in check.
          4. On barge-in: return BARGED_IN.
          5. Otherwise build the appropriate DispatchOutcome.
        """
        try:
            decision = self.supervisor.decide(inputs)
        except Exception as e:                                      # noqa: BLE001
            logger.warning(
                "supervisor_dispatch: supervisor.decide raised (%s); "
                "falling back to legacy path.", e,
            )
            return DispatchOutcome(kind=DispatchActionKind.FALLBACK)

        narration = self._narration_for(decision)
        already_narrated = False
        if self.narrate_enabled and narration and self._barge_in_speak is not None:
            try:
                bargedin = self._barge_in_speak(narration)
            except Exception as e:                                  # noqa: BLE001
                logger.warning(
                    "supervisor_dispatch: narration speak raised (%s); "
                    "proceeding without barge-in check.", e,
                )
                bargedin = False
            already_narrated = True
            if bargedin:
                return DispatchOutcome(
                    kind=DispatchActionKind.BARGED_IN,
                    voice_message="",
                    decision=decision,
                    already_narrated=True,
                )

        # 2026-05-22 catalog batch 14 (T5 Phase 2): narrate the architect's
        # plan with sentence-boundary barge-in BEFORE dispatching the editor
        # LLM. The plan still flows into the editor's prompt regardless of
        # whether narration completed -- we never trim ``architect_plan_text``.
        if (
            self._architect_narrator is not None
            and decision.architect_plan_text
        ):
            try:
                plan_interrupted = bool(self._architect_narrator(
                    decision.architect_plan_text.strip(),
                ))
            except Exception as e:                                  # noqa: BLE001
                logger.warning(
                    "supervisor_dispatch: architect narrator raised (%s); "
                    "proceeding without architect narration.", e,
                )
                plan_interrupted = False
            if plan_interrupted:
                return DispatchOutcome(
                    kind=DispatchActionKind.BARGED_IN,
                    voice_message="",
                    decision=decision,
                    already_narrated=True,
                )

        return self._outcome_for(
            decision, inputs, narration, already_narrated,
        )

    # --- narration ----------------------------------------------------------

    def _narration_for(self, decision: SupervisorDecision) -> str:
        """Build a short, TTS-safe narration line for the decision.

        Naming follows the TTS-safety rules surfaced by
        :mod:`ultron.coding.voice_lock` and the lesson from
        :mod:`ultron.coding.runner` -- never speak absolute Windows
        paths with backslashes, drive letters, or colons.
        """
        if decision.action == SupervisorAction.RESUME:
            name = decision.target_project_name or "the current project"
            return (
                f"Continuing work on {_speakable(name)}. "
                f"Barge in to stop."
            )
        if decision.action == SupervisorAction.EDIT:
            name = decision.target_project_name or "that project"
            return (
                f"Editing your {_speakable(name)} project. "
                f"Barge in to stop."
            )
        if decision.action == SupervisorAction.NEW:
            return (
                "Starting a new project. Barge in to stop."
            )
        if decision.action == SupervisorAction.CLARIFY:
            # CLARIFY's voice line is the question itself; speak it
            # without barge-in (we're WAITING for the user, not
            # acting).
            return decision.clarification_question or (
                "I'm not sure which project you mean. Could you say its name?"
            )
        return ""

    # --- outcome construction ----------------------------------------------

    def _outcome_for(
        self,
        decision: SupervisorDecision,
        inputs: SupervisorInputs,
        narration: str,
        already_narrated: bool,
    ) -> DispatchOutcome:
        if decision.action == SupervisorAction.RESUME:
            return DispatchOutcome(
                kind=DispatchActionKind.RESUME_FORWARD,
                voice_message="" if already_narrated else narration,
                resume_session_id=decision.resume_session_id,
                decision=decision,
                already_narrated=already_narrated,
            )

        if decision.action == SupervisorAction.CLARIFY:
            # Clarify always needs to speak the question. If we already
            # narrated (which already includes the question), don't
            # re-speak.
            return DispatchOutcome(
                kind=DispatchActionKind.CLARIFY,
                voice_message="" if already_narrated else narration,
                clarification_question=decision.clarification_question,
                decision=decision,
                already_narrated=already_narrated,
            )

        if decision.action == SupervisorAction.EDIT:
            request = self._build_edit_task_request(decision, inputs)
            return DispatchOutcome(
                kind=DispatchActionKind.EDIT_DISPATCH,
                voice_message="" if already_narrated else narration,
                task_request=request,
                decision=decision,
                already_narrated=already_narrated,
            )

        if decision.action == SupervisorAction.NEW:
            request = self._build_new_task_request(decision, inputs)
            return DispatchOutcome(
                kind=DispatchActionKind.NEW_DISPATCH,
                voice_message="" if already_narrated else narration,
                task_request=request,
                decision=decision,
                already_narrated=already_narrated,
            )

        # Unreachable in practice; defensive.
        return DispatchOutcome(
            kind=DispatchActionKind.FALLBACK,
            decision=decision,
        )

    # --- TaskRequest builders -----------------------------------------------

    def _build_edit_task_request(
        self,
        decision: SupervisorDecision,
        inputs: SupervisorInputs,
    ) -> Optional[TaskRequest]:
        """Construct a TaskRequest pointed at an existing project.

        When ``enriched_context_enabled``, prepends the project digest +
        file tree snapshot to the prompt body so Claude doesn't
        re-explore.
        """
        if not decision.target_project_path:
            return None
        cwd = Path(decision.target_project_path)
        if not cwd.exists() or not cwd.is_dir():
            logger.warning(
                "supervisor_dispatch: EDIT target path %s does not exist; "
                "falling back to legacy.", cwd,
            )
            return None

        prompt = self._build_edit_prompt(decision, inputs, cwd)

        return TaskRequest(
            task_prompt=prompt,
            cwd=cwd,
            model=self.default_model,
            label=f"edit:{decision.target_project_name}",
            require_testing=False,
        )

    def _build_new_task_request(
        self,
        decision: SupervisorDecision,
        inputs: SupervisorInputs,
    ) -> Optional[TaskRequest]:
        """Construct a TaskRequest for a new project scaffold.

        The CWD lives under the configured sandbox root. The directory
        name is derived from the user utterance via a simple slugify.
        """
        if self.sandbox_root is None:
            logger.warning(
                "supervisor_dispatch: NEW dispatch needs sandbox_root; "
                "falling back to legacy.",
            )
            return None
        slug = _slugify_for_directory(inputs.user_text)
        if not slug:
            slug = f"project_{int(time.time())}"
        cwd = self.sandbox_root / slug
        # Caller is responsible for `mkdir` -- existing runner does
        # this. We don't pre-create here so we don't half-write state
        # if the dispatch never starts.
        # Catalog batches 6/7: prepend architect plan + repo map (if any)
        # so even NEW scaffolds benefit when the operator has the
        # architect provider on (rare but supported).
        prompt_parts: List[str] = []
        if decision.architect_plan_text:
            prompt_parts.extend([
                "Architect plan (follow these instructions):",
                decision.architect_plan_text.strip(),
                "",
            ])
        if decision.repo_map_text:
            prompt_parts.extend([
                "Repo map (PageRank-ranked symbols, for orientation):",
                decision.repo_map_text.strip(),
                "",
            ])
        prompt_parts.extend([
            "User request:",
            inputs.user_text.strip(),
        ])
        prompt = "\n".join(prompt_parts)
        return TaskRequest(
            task_prompt=prompt,
            cwd=cwd,
            model=self.default_model,
            label=f"new:{slug}",
            require_testing=False,
        )

    def _build_edit_prompt(
        self,
        decision: SupervisorDecision,
        inputs: SupervisorInputs,
        cwd: Path,
    ) -> str:
        """Assemble the enriched prompt body for an EDIT dispatch.

        Sections (omitted when empty):

          1. The user's request (verbatim).
          2. "Architect plan" -- 2026-05-22 catalog batch 6/7
             integration. When ``decision.architect_plan_text`` is set
             (the supervisor's ``architect_provider`` produced a
             prose plan), include it so the editor LLM has explicit
             prose direction. This section is included EVEN when
             ``enriched_context_enabled`` is off — the plan is the
             whole point of running the architect.
          3. "Repo map" -- catalog batch 2/7 integration. When
             ``decision.repo_map_text`` is set (PageRank-weighted
             symbol map), include it so the editor starts with
             structural awareness. Also included when enriched off.
          4. "What we know about this project" -- digest sections
             pulled from the index entry (enriched-only).
          5. "Project layout" -- file tree snapshot summary
             (enriched-only).
          6. "Likely relevant files" -- file hints from the digest
             (enriched-only).

        Capped at a reasonable size so we don't bloat Claude's prompt.
        """
        parts: List[str] = [
            "User request:",
            f"  {inputs.user_text.strip()}",
            "",
        ]

        # Catalog batches 6/7: architect plan + repo map injected even
        # when enriched_context is OFF — these come from the supervisor's
        # own optional providers and only exist if the operator already
        # opted into them via their respective config flags.
        if decision.architect_plan_text:
            parts.extend([
                "Architect plan (follow these instructions):",
                _indent_block(decision.architect_plan_text.strip(), "  "),
                "",
            ])
        if decision.repo_map_text:
            parts.extend([
                "Repo map (PageRank-ranked symbols, for orientation):",
                _indent_block(decision.repo_map_text.strip(), "  "),
                "",
            ])

        if not self.enriched_context_enabled:
            return "\n".join(parts).rstrip() + "\n"

        index_entry: Optional[ProjectIndexEntry] = None
        if self.index is not None and decision.target_project_id:
            try:
                index_entry = self.index.get(decision.target_project_id)
            except Exception as e:                                  # noqa: BLE001
                logger.debug(
                    "supervisor_dispatch: index.get failed (%s)", e,
                )

        if index_entry is not None and index_entry.digest_markdown:
            parts.extend([
                "What we know about this project (from prior sessions):",
                _indent_block(index_entry.digest_markdown.strip(), "  "),
                "",
            ])

        # Try to provide a fresh file-tree snapshot. snapshot() is
        # cached + cheap.
        try:
            snap: ProjectSnapshot = snapshot(cwd)
        except Exception as e:                                      # noqa: BLE001
            logger.debug(
                "supervisor_dispatch: snapshot failed (%s)", e,
            )
            snap = None  # type: ignore[assignment]

        if snap is not None and snap.file_count > 0:
            parts.extend([
                "Project layout (capped):",
                _indent_block(snap.render_tree_summary(max_lines=40), "  "),
                "",
            ])

        if decision.file_hints:
            parts.append("Likely-relevant files to consider first:")
            for f in decision.file_hints[:10]:
                parts.append(f"  - {f}")
            parts.append("")

        return "\n".join(parts).rstrip() + "\n"

    # --- digest helpers (used by orchestrator post-COMPLETE listener) -------

    def build_digest(
        self,
        project_name: str,
        project_path: Path,
        task_summary: str,
        files_created: List[Path],
        files_modified: List[Path],
        files_deleted: List[Path],
        *,
        llm_call: Optional[LLMCallable] = None,
        prior_digest_markdown: str = "",
        user_goal_hint: str = "",
    ) -> ProjectDigest:
        """Produce a :class:`ProjectDigest` for a completed session.

        Convenience wrapper that pulls language + entry-points from a
        fresh snapshot and forwards to
        :func:`ultron.coding.project_digest.generate_digest`. Used by
        the orchestrator's TaskHandle COMPLETE listener.
        """
        snap = snapshot(project_path, use_cache=False)
        request = DigestRequest(
            project_name=project_name,
            project_path=project_path,
            task_summary=task_summary,
            files_created=files_created,
            files_modified=files_modified,
            files_deleted=files_deleted,
            prior_digest_markdown=prior_digest_markdown,
            user_goal_hint=user_goal_hint,
            language=snap.dominant_language,
            entry_points=snap.entry_points,
        )
        return generate_digest(request, llm_call=llm_call)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _speakable(text: str) -> str:
    """Strip path-like noise so TTS doesn't choke on a project name.

    Mirrors the discipline in
    :mod:`ultron.coding.runner.completion_narration`: never speak
    backslashes, drive letters, or colons; just the leaf name.
    """
    if not text:
        return ""
    # Drop any path prefix.
    if "/" in text or "\\" in text:
        text = text.replace("\\", "/").rsplit("/", 1)[-1]
    # Strip surrounding quotes.
    text = text.strip().strip("'\"")
    return text


def _slugify_for_directory(text: str) -> str:
    """Build a filesystem-safe directory slug from a user utterance.

    Lowercase, replace runs of non-alphanumeric with underscore,
    trim to 50 chars.
    """
    if not text:
        return ""
    out = []
    last_was_underscore = False
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
            last_was_underscore = False
        else:
            if not last_was_underscore:
                out.append("_")
                last_was_underscore = True
    slug = "".join(out).strip("_")
    return slug[:50]


def _indent_block(block: str, prefix: str) -> str:
    """Prepend ``prefix`` to every line of ``block``."""
    return "\n".join(f"{prefix}{line}" for line in block.splitlines())


__all__ = [
    "BargeInCheckable",
    "DispatchActionKind",
    "DispatchOutcome",
    "PlainSpeak",
    "SupervisorDispatchController",
]
