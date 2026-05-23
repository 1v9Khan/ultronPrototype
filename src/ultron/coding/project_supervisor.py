"""Project supervisor -- the decision layer that sits between the
routing classifier and the AI coding agent dispatch.

Given a user utterance + classified intent (CODE_TASK /
MID_SESSION_ADJUSTMENT / CLARIFICATION_RESPONSE), the supervisor
decides:

  1. ``RESUME``  -- continue the current active Claude session
                    (typical for "now add error handling")
  2. ``EDIT``    -- open an existing known project and edit it
                    (typical for "edit the flask app")
  3. ``CLARIFY`` -- the reference is ambiguous; ask the user
                    ("did you mean X or Y?")
  4. ``NEW``     -- nothing matches; scaffold a new project

Decision strategy (priority order, first hit wins):

  * **Active-task + adjustment**: if a Claude session is currently
    running AND the utterance matches ``_ADJUSTMENT_PATTERNS``
    (the "now add error handling" pattern from
    :mod:`ultron.coding.intent`), RESUME.
  * **Strong semantic match**: top
    :class:`ultron.coding.project_index.ProjectMatch` score
    ``>= resolve_threshold`` (default 0.75) → EDIT that project.
  * **Lexical exact match** in the registry → EDIT.
  * **Ambiguous match band**: top score in ``[clarify_threshold,
    resolve_threshold)`` (default [0.55, 0.75)) → CLARIFY with the
    top 2-3 candidates.
  * **Else** → NEW.

Every decision is appended to ``logs/supervisor_decisions.jsonl``
for offline tuning of the cosine thresholds.

This module is deliberately stateless across calls; pass everything
the supervisor needs through ``decide()``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ultron.bus import SupervisorDecidedEvent, publish as bus_publish
from ultron.coding.intent import CodingIntent, CodingIntentKind, _ADJUSTMENT_PATTERNS
from ultron.coding.project_index import ProjectIndex, ProjectMatch
from ultron.coding.projects import (
    Project,
    ProjectRegistry,
    ProjectResolution,
    ProjectResolver,
    ResolutionKind,
)

logger = logging.getLogger("ultron.coding.project_supervisor")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class SupervisorAction(str, Enum):
    """The four actions the supervisor can decide on."""

    RESUME = "resume"
    EDIT = "edit"
    CLARIFY = "clarify"
    NEW = "new"


@dataclass
class SupervisorCandidate:
    """A candidate project considered during decision-making.

    Kept lightweight + JSON-serializable for the audit log.
    """

    project_id: str
    project_name: str
    project_path: str
    score: float = 0.0
    source: str = ""  # "semantic" | "registry_exact" | "registry_substring"


@dataclass
class SupervisorDecision:
    """The supervisor's verdict for a single utterance."""

    action: SupervisorAction
    target_project_id: Optional[str] = None
    target_project_name: Optional[str] = None
    target_project_path: Optional[str] = None
    resume_session_id: Optional[str] = None
    candidates: List[SupervisorCandidate] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    clarification_question: Optional[str] = None
    # File-path hints the supervisor wants Claude to focus on. Pulled
    # from the digest's "Relevant Files" section when EDIT-ing.
    file_hints: List[str] = field(default_factory=list)
    # Original utterance for log + audit.
    user_text: str = ""
    # PageRank-weighted repo map (2026-05-22 catalog batch 2). Set by
    # :meth:`ProjectSupervisor._attach_repo_map` when a
    # ``repo_map_provider`` was supplied at construction and the
    # decision resolves to a known project path. Downstream callers
    # (supervisor_dispatch) prepend this to the Claude prompt body so
    # the coding agent starts the session with structural awareness
    # of the project. Excluded from ``to_log_dict`` so the audit log
    # stays lean.
    repo_map_text: Optional[str] = None
    # 2026-05-22 catalog batch 6: optional architect plan, generated
    # by a local LLM via :class:`ArchitectSupervisor` when an
    # ``architect_provider`` is wired. Parallel to ``repo_map_text``;
    # downstream callers can prepend this prose plan to the Claude
    # dispatch prompt body so the editor LLM has explicit prose
    # direction. Excluded from ``to_log_dict`` so the audit log
    # stays lean (a boolean ``architect_plan_attached`` is emitted
    # instead).
    architect_plan_text: Optional[str] = None

    def to_log_dict(self) -> Dict[str, Any]:
        """JSON-serializable form for the audit log.

        The ``repo_map_text`` field is intentionally NOT included
        here — it can be multiple kilobytes per turn and would bloat
        the JSONL log without aiding offline tuning.
        """
        return {
            "action": self.action.value,
            "target_project_id": self.target_project_id,
            "target_project_name": self.target_project_name,
            "target_project_path": self.target_project_path,
            "resume_session_id": self.resume_session_id,
            "candidates": [asdict(c) for c in self.candidates],
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "clarification_question": self.clarification_question,
            "file_hints": self.file_hints,
            "user_text": self.user_text,
            "decided_at_unix": time.time(),
            "repo_map_attached": self.repo_map_text is not None,
            "architect_plan_attached": self.architect_plan_text is not None,
        }


@dataclass
class SupervisorInputs:
    """Bundle of context the supervisor needs for one decision.

    Kept as a dataclass so callers can build it once and pass it
    through several decisions if needed.
    """

    user_text: str
    coding_intent: Optional[CodingIntent] = None
    has_active_task: bool = False
    active_task_project_name: Optional[str] = None
    active_task_session_id: Optional[str] = None
    turn_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


class ProjectSupervisor:
    """Stateless decision layer over the project index + registry.

    Args:
        index: a :class:`ProjectIndex` for semantic search. None
            disables semantic lookup (falls back to registry only).
        registry: a :class:`ProjectRegistry` for lexical matches +
            scaffold targets.
        resolver: optional :class:`ProjectResolver` to reuse the
            existing semantic-resolver fallback logic. None means
            we use the registry directly.
        resolve_threshold: cosine score above which we EDIT
            without clarification. Default 0.75.
        clarify_threshold: cosine score above which we CLARIFY
            (band: clarify_threshold <= score < resolve_threshold).
            Default 0.55.
        decisions_log_path: JSONL log path for offline tuning. None
            disables logging. Logs are append-only.
        max_candidates_in_decision: how many candidates to retain
            in the decision (for CLARIFY presentation + audit).
    """

    def __init__(
        self,
        index: Optional[ProjectIndex],
        registry: ProjectRegistry,
        resolver: Optional[ProjectResolver] = None,
        *,
        resolve_threshold: float = 0.75,
        clarify_threshold: float = 0.55,
        decisions_log_path: Optional[Path] = None,
        max_candidates_in_decision: int = 5,
        repo_map_provider: Optional[
            "Callable[[str, str], Optional[str]]"  # type: ignore[name-defined]
        ] = None,
        architect_provider: Optional[
            "Callable[..., Optional[str]]"  # type: ignore[name-defined]
        ] = None,
    ) -> None:
        if not 0.0 <= clarify_threshold <= resolve_threshold <= 1.0:
            raise ValueError(
                f"clarify_threshold ({clarify_threshold}) must be "
                f"<= resolve_threshold ({resolve_threshold}); both in [0, 1]."
            )

        self.index = index
        self.registry = registry
        self.resolver = resolver
        self.resolve_threshold = float(resolve_threshold)
        self.clarify_threshold = float(clarify_threshold)
        self.decisions_log_path = (
            Path(decisions_log_path) if decisions_log_path else None
        )
        self.max_candidates_in_decision = max_candidates_in_decision
        # 2026-05-22 catalog batch 2: optional repo-map provider called
        # after a decision resolves to a known project path. The
        # callable takes (project_path, user_text) and returns either
        # a rendered map string or None. Always invoked fail-open —
        # provider errors are logged and swallowed.
        self.repo_map_provider = repo_map_provider
        # 2026-05-22 catalog batch 6: optional architect-plan provider.
        # Mirrors repo_map_provider but produces a prose plan from a
        # local LLM via :class:`ArchitectSupervisor`. Invoked AFTER
        # the repo-map attachment so the architect can be given the
        # rendered map as additional context.
        self.architect_provider = architect_provider
        self._log_lock = threading.RLock()

        if self.decisions_log_path is not None:
            try:
                self.decisions_log_path.parent.mkdir(
                    parents=True, exist_ok=True,
                )
            except OSError as e:
                logger.warning(
                    "Could not ensure decisions log dir %s (%s); "
                    "logging will fail-open.",
                    self.decisions_log_path.parent, e,
                )

    # --- public API ---------------------------------------------------------

    def decide(self, inputs: SupervisorInputs) -> SupervisorDecision:
        """Run the decision pipeline and return a verdict.

        Always returns a decision. Never raises. Emits an audit
        log entry + bus event before returning (best-effort).
        """
        text = (inputs.user_text or "").strip()
        decision: SupervisorDecision
        try:
            decision = self._decide_inner(text, inputs)
        except Exception as e:                                      # noqa: BLE001
            logger.warning(
                "ProjectSupervisor.decide failed (%s); defaulting to NEW.",
                e,
            )
            decision = SupervisorDecision(
                action=SupervisorAction.NEW,
                reasoning=f"supervisor error: {e}",
                user_text=text,
            )

        # Audit log + bus event are best-effort; they never affect the
        # returned decision.
        self._record_decision(decision, inputs)
        self._publish_bus_event(decision, inputs)
        # 2026-05-22 catalog batch 2: attach the repo map AFTER the
        # audit log entry has been written so the (potentially large)
        # rendered map doesn't bloat the JSONL.
        self._attach_repo_map(decision)
        # 2026-05-22 catalog batch 6: invoke architect AFTER repo_map
        # so the architect can receive the rendered map as additional
        # context. Same skip rules as repo_map.
        self._attach_architect_plan(decision)
        return decision

    def _attach_architect_plan(self, decision: SupervisorDecision) -> None:
        """Populate ``decision.architect_plan_text`` via ``architect_provider``.

        Skipped when:
          * no provider was supplied at construction time, or
          * the decision has no ``target_project_path``, or
          * the action is CLARIFY (no plan needed until the user
            disambiguates).

        Provider exceptions are logged + swallowed — like the repo
        map, the plan is a bonus, not a contract.

        The provider receives ``(project_path, user_text)`` positionally
        plus ``repo_map_text=`` as a keyword arg so providers built on
        :class:`ArchitectSupervisor` can fold the map into the
        architect prompt. Providers that don't accept the keyword
        (e.g. test stubs) get a fallback call without the kwarg.
        """
        if self.architect_provider is None:
            return
        if decision.action == SupervisorAction.CLARIFY:
            return
        if not decision.target_project_path:
            return
        try:
            try:
                plan = self.architect_provider(
                    decision.target_project_path,
                    decision.user_text,
                    repo_map_text=decision.repo_map_text,
                )
            except TypeError:
                # Provider doesn't accept the keyword — call positionally.
                plan = self.architect_provider(
                    decision.target_project_path,
                    decision.user_text,
                )
        except Exception as exc:                                    # noqa: BLE001
            logger.warning(
                "architect_provider raised for project_path=%s: %s",
                decision.target_project_path, exc,
            )
            return
        if plan:
            decision.architect_plan_text = plan

    def _attach_repo_map(self, decision: SupervisorDecision) -> None:
        """Populate ``decision.repo_map_text`` via ``repo_map_provider``.

        Skipped when:
          * no provider was supplied at construction time, or
          * the decision has no ``target_project_path`` (NEW with no
            scaffold, CLARIFY pending, decide() error path), or
          * the action is CLARIFY (we don't need the map until the
            user has resolved ambiguity).

        Provider exceptions are logged + swallowed — the supervisor
        decision is the contract; the repo map is a bonus.
        """
        if self.repo_map_provider is None:
            return
        if decision.action == SupervisorAction.CLARIFY:
            return
        if not decision.target_project_path:
            return
        try:
            rendered = self.repo_map_provider(
                decision.target_project_path,
                decision.user_text,
            )
        except Exception as exc:                                    # noqa: BLE001
            logger.warning(
                "repo_map_provider raised for project_path=%s: %s",
                decision.target_project_path, exc,
            )
            return
        if rendered:
            decision.repo_map_text = rendered

    # --- decision pipeline --------------------------------------------------

    def _decide_inner(
        self, text: str, inputs: SupervisorInputs,
    ) -> SupervisorDecision:
        if not text:
            return SupervisorDecision(
                action=SupervisorAction.NEW,
                reasoning="empty utterance; defaulting to NEW",
                user_text=text,
            )

        # 1) Active-task adjustment → RESUME current session.
        if self._is_resume_case(text, inputs):
            return SupervisorDecision(
                action=SupervisorAction.RESUME,
                target_project_name=inputs.active_task_project_name,
                resume_session_id=inputs.active_task_session_id,
                confidence=0.95,
                reasoning=(
                    "adjustment-style utterance with an active Claude "
                    "task -- routing to the in-flight session"
                ),
                user_text=text,
            )

        # 2) Gather candidates from the index (semantic) + registry
        # (lexical). Both sources are merged + ranked.
        semantic_candidates = self._semantic_candidates(text)
        registry_candidates = self._registry_candidates(text)
        all_candidates = _merge_candidates(
            semantic_candidates,
            registry_candidates,
            cap=self.max_candidates_in_decision,
        )

        # 3) Strong semantic match → EDIT.
        top = all_candidates[0] if all_candidates else None
        if top is not None and top.score >= self.resolve_threshold:
            return self._build_edit_decision(
                top, all_candidates, text,
                reasoning_extra="semantic match above resolve threshold",
            )

        # 4) Registry exact name / alias match → EDIT (highest-trust
        # lexical signal even when semantic is weak).
        exact = self._registry_exact_match(text)
        if exact is not None:
            cand = SupervisorCandidate(
                project_id=_project_id_for_registry(exact),
                project_name=exact.name,
                project_path=str(exact.path),
                score=1.0,
                source="registry_exact",
            )
            # Push exact-match candidate to the front of the list for
            # transparency in the audit log.
            cand_list = [cand] + [c for c in all_candidates if c.project_path != cand.project_path]
            return self._build_edit_decision(
                cand, cand_list[:self.max_candidates_in_decision], text,
                reasoning_extra="exact registry name / alias match",
            )

        # 5) Ambiguous band → CLARIFY.
        if (
            top is not None
            and self.clarify_threshold <= top.score < self.resolve_threshold
        ):
            return self._build_clarify_decision(all_candidates, text)

        # 6) Else NEW.
        return SupervisorDecision(
            action=SupervisorAction.NEW,
            candidates=all_candidates,
            confidence=1.0 - (top.score if top else 0.0),
            reasoning=(
                "no project matched above clarify threshold; "
                "treating as new scaffold target"
            ),
            user_text=text,
        )

    # --- candidate sources --------------------------------------------------

    def _semantic_candidates(self, text: str) -> List[SupervisorCandidate]:
        if self.index is None:
            return []
        try:
            matches = self.index.search(
                text,
                top_k=self.max_candidates_in_decision,
                min_score=0.0,  # rank everything; gate later
            )
        except Exception as e:                                      # noqa: BLE001
            logger.warning("supervisor: semantic search failed (%s)", e)
            return []
        out: List[SupervisorCandidate] = []
        for m in matches:
            out.append(SupervisorCandidate(
                project_id=m.entry.project_id,
                project_name=m.entry.project_name,
                project_path=m.entry.project_path,
                score=m.score,
                source="semantic",
            ))
        return out

    def _registry_candidates(
        self, text: str,
    ) -> List[SupervisorCandidate]:
        """Pull lexical candidates from the registry's resolver."""
        if self.resolver is None:
            return self._registry_substring_candidates(text)

        try:
            resolution = self.resolver.resolve(text)
        except Exception as e:                                      # noqa: BLE001
            logger.debug("supervisor: resolver.resolve raised (%s)", e)
            return self._registry_substring_candidates(text)

        out: List[SupervisorCandidate] = []
        if resolution.project is not None:
            out.append(_registry_to_candidate(
                resolution.project,
                score=float(resolution.confidence or 0.0),
                source=f"registry_{resolution.kind.value}",
            ))
        for cand in resolution.candidates or []:
            if any(c.project_path == str(cand.path) for c in out):
                continue
            out.append(_registry_to_candidate(
                cand,
                score=0.6,  # heuristic for substring-ambiguous candidates
                source="registry_substring",
            ))
        if not out:
            out = self._registry_substring_candidates(text)
        return out

    def _registry_substring_candidates(
        self, text: str,
    ) -> List[SupervisorCandidate]:
        """Fallback substring scan when no resolver is available."""
        try:
            projects = self.registry.list()
        except Exception as e:                                      # noqa: BLE001
            logger.debug("supervisor: registry.list raised (%s)", e)
            return []
        ref = text.lower()
        out: List[SupervisorCandidate] = []
        for p in projects:
            blob = (
                f"{p.name} {' '.join(p.aliases)} {p.description} "
                f"{' '.join(p.tags)}"
            ).lower()
            if (
                p.name.lower() in ref
                or ref in blob
                or any(a.lower() in ref for a in p.aliases)
            ):
                out.append(_registry_to_candidate(
                    p, score=0.55, source="registry_substring",
                ))
        return out

    def _registry_exact_match(self, text: str) -> Optional[Project]:
        """Exact name OR alias match in the registry."""
        ref = text.lower()
        try:
            projects = self.registry.list()
        except Exception:                                           # noqa: BLE001
            return None
        for p in projects:
            if p.name.lower() in ref.split():
                return p
            if any(a.lower() in ref.split() for a in p.aliases):
                return p
        return None

    # --- predicate helpers --------------------------------------------------

    def _is_resume_case(self, text: str, inputs: SupervisorInputs) -> bool:
        """Active-task + adjustment-style utterance.

        Uses the same ``_ADJUSTMENT_PATTERNS`` regex the existing
        intent classifier already uses, so behavior matches what
        the routing layer expects.
        """
        if not inputs.has_active_task:
            return False
        if inputs.coding_intent is not None and (
            inputs.coding_intent.kind == CodingIntentKind.MID_SESSION_ADJUSTMENT
            or inputs.coding_intent.kind == CodingIntentKind.CLARIFICATION_RESPONSE
        ):
            return True
        if _ADJUSTMENT_PATTERNS.search(text):
            return True
        return False

    # --- decision builders --------------------------------------------------

    def _build_edit_decision(
        self,
        top: SupervisorCandidate,
        candidates: List[SupervisorCandidate],
        text: str,
        *,
        reasoning_extra: str,
    ) -> SupervisorDecision:
        file_hints: List[str] = []
        if self.index is not None:
            entry = self.index.get(top.project_id)
            if entry is not None:
                # Pull the "Relevant Files" section out of the digest.
                from ultron.coding.project_digest import extract_files_from_digest
                file_hints = extract_files_from_digest(entry.digest_markdown)
        return SupervisorDecision(
            action=SupervisorAction.EDIT,
            target_project_id=top.project_id,
            target_project_name=top.project_name,
            target_project_path=top.project_path,
            candidates=candidates,
            confidence=float(top.score),
            reasoning=(
                f"{reasoning_extra}: matched {top.project_name!r} "
                f"({top.source}, score={top.score:.3f})"
            ),
            file_hints=file_hints,
            user_text=text,
        )

    def _build_clarify_decision(
        self,
        candidates: List[SupervisorCandidate],
        text: str,
    ) -> SupervisorDecision:
        top_two = candidates[:2]
        names = [c.project_name for c in top_two]
        if len(top_two) >= 2:
            question = (
                f"I'm not sure which project you mean -- "
                f"the {names[0]} one we worked on, or {names[1]}?"
            )
        elif len(top_two) == 1:
            question = (
                f"Did you mean the {names[0]} we worked on, "
                f"or are you starting something new?"
            )
        else:
            question = "I'm not sure which project you mean. Could you say its name?"
        top_score = candidates[0].score if candidates else 0.0
        return SupervisorDecision(
            action=SupervisorAction.CLARIFY,
            candidates=candidates,
            confidence=top_score,
            reasoning=(
                f"top match {candidates[0].project_name!r} "
                f"({candidates[0].score:.3f}) is in clarify band "
                f"[{self.clarify_threshold:.2f}, {self.resolve_threshold:.2f})"
                if candidates else "no candidates above clarify threshold"
            ),
            clarification_question=question,
            user_text=text,
        )

    # --- audit + bus --------------------------------------------------------

    def _record_decision(
        self, decision: SupervisorDecision, inputs: SupervisorInputs,
    ) -> None:
        if self.decisions_log_path is None:
            return
        line = decision.to_log_dict()
        line["turn_id"] = inputs.turn_id
        try:
            with self._log_lock:
                with self.decisions_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(line, default=str) + "\n")
        except OSError as e:
            logger.debug(
                "supervisor decision log write failed (%s); ignoring.", e,
            )

    def _publish_bus_event(
        self, decision: SupervisorDecision, inputs: SupervisorInputs,
    ) -> None:
        try:
            bus_publish(SupervisorDecidedEvent, {
                "turn_id": int(inputs.turn_id or 0),
                "action": decision.action.value,
                "target_project": decision.target_project_name or "",
                "confidence": float(decision.confidence),
                "reasoning": decision.reasoning,
                "candidates": [
                    {
                        "name": c.project_name,
                        "score": c.score,
                        "source": c.source,
                    }
                    for c in decision.candidates[:3]
                ],
            })
        except Exception as e:                                      # noqa: BLE001
            logger.debug("bus publish failed for supervisor.decided (%s)", e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _merge_candidates(
    semantic: Sequence[SupervisorCandidate],
    registry: Sequence[SupervisorCandidate],
    *,
    cap: int,
) -> List[SupervisorCandidate]:
    """Combine semantic + registry candidates without duplicates.

    Same project_path => same project. When a project appears in
    both lists, the higher score wins; sources are concatenated for
    transparency ("semantic+registry_alias").
    """
    seen: Dict[str, SupervisorCandidate] = {}
    for c in list(semantic) + list(registry):
        key = c.project_path or c.project_id
        existing = seen.get(key)
        if existing is None:
            seen[key] = SupervisorCandidate(
                project_id=c.project_id,
                project_name=c.project_name,
                project_path=c.project_path,
                score=c.score,
                source=c.source,
            )
        elif c.score > existing.score:
            merged_source = (
                f"{c.source}+{existing.source}"
                if existing.source != c.source
                else c.source
            )
            seen[key] = SupervisorCandidate(
                project_id=existing.project_id or c.project_id,
                project_name=existing.project_name or c.project_name,
                project_path=existing.project_path or c.project_path,
                score=c.score,
                source=merged_source,
            )
        elif existing.source != c.source:
            existing.source = f"{existing.source}+{c.source}"
    ranked = sorted(seen.values(), key=lambda c: -c.score)
    return ranked[:cap]


def _registry_to_candidate(
    project: Project, *, score: float, source: str,
) -> SupervisorCandidate:
    return SupervisorCandidate(
        project_id=_project_id_for_registry(project),
        project_name=project.name,
        project_path=str(project.path),
        score=score,
        source=source,
    )


def _project_id_for_registry(project: Project) -> str:
    """Map a registry :class:`Project` to a stable project_id.

    Mirrors :func:`ultron.coding.project_index._derive_project_id`
    so a registry project + an indexed project at the same path
    collapse to the same id during candidate merging.
    """
    from ultron.coding.project_index import _derive_project_id
    return _derive_project_id(Path(project.path))


__all__ = [
    "ProjectSupervisor",
    "SupervisorAction",
    "SupervisorCandidate",
    "SupervisorDecision",
    "SupervisorInputs",
]
