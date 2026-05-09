"""ConversationCoordinator -- the supervisor's decision policy engine.

Sits between :class:`UltronMCPServer` (which exposes Claude's tool calls
as events) and :class:`CodingTaskRunner` (which owns the bridge and the
session lifecycle). The coordinator is *only* consulted when a real
decision needs to be made:

  * Claude calls ``request_clarification`` -> coordinator decides
    answer-or-escalate.
  * Verification fails after ``declare_complete`` -> coordinator drafts
    a corrective prompt (Phase 4 hook).
  * User issues a mid-session adjustment -> coordinator wraps it as a
    follow-up prompt for Claude, with conflict detection.

The coordinator does not own state or subprocesses. It reads state via
:class:`SessionStore` and emits decisions; the runner acts on them.

LLM use is constrained: rule-based fast paths handle most clarifications
without touching Qwen. Only genuinely ambiguous cases (and the
voice-friendly question rendering on escalation) hit the main LLM.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from config import settings
from ultron.coding.session import (
    ClarificationRequest,
    CompletionClaim,
    ProjectSession,
    SessionStatus,
    SessionStore,
)
from ultron.coding.templates import (
    PromptTooLargeError,
    SchemaValidationError,
    TemplateRenderer,
)
from ultron.coding.verification import Verifier, VerificationReport
from ultron.utils.logging import get_logger

logger = get_logger("coding.coordinator")


# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------


class DecisionPath(str, Enum):
    """Why the coordinator answered the way it did. Stored on the
    ClarificationRequest so the per-clarification log records it."""
    RULE_ESCALATE = "rule_escalate"             # always-escalate keyword
    RULE_DEFAULT = "rule_default"               # urgency=preference + options
    RULE_ANSWER = "rule_answer"                 # always-answer keyword
    FACT_ANSWER = "fact_answer"                 # high-confidence stored fact
    LLM_ANSWER = "llm_answer"
    LLM_DEFAULT = "llm_default"
    LLM_ESCALATE = "llm_escalate"
    USER_ANSWER = "user_answer"                 # answer came from voice
    TIMEOUT_DEFAULT = "timeout_default"


@dataclass
class ClarificationDecision:
    """Output of :meth:`ConversationCoordinator.decide_clarification`."""
    action: str  # "ANSWER" | "USE_DEFAULT" | "ESCALATE"
    answer: Optional[str]
    reasoning: str
    decision_path: DecisionPath


@dataclass
class AdjustmentDecision:
    """Output of :meth:`ConversationCoordinator.decide_adjustment`."""
    action: str           # "FOLLOWUP" | "ESCALATE_CONFLICT" | "ESCALATE_AMBIGUOUS"
    followup_prompt: Optional[str]   # set when action=FOLLOWUP
    conflict_reason: Optional[str]   # set when action=ESCALATE_CONFLICT
    voice_question: Optional[str]    # set when escalation


@dataclass
class PendingUserClarification:
    """A clarification that's been escalated to the user and is awaiting
    their voice response. Surfaced by the voice controller's
    ``pending_clarifications()`` so the orchestrator main loop can speak
    the question."""
    request_id: str
    session_id: str
    voice_question: str
    options: List[str]
    raised_at: float


@dataclass
class _FactAnswer:
    """Internal: structured answer derived from a stored fact.

    Returned by :meth:`ConversationCoordinator._maybe_answer_from_facts`
    when a high-confidence fact directly addresses the question. The
    caller wraps this into a :class:`ClarificationDecision` with
    ``decision_path=FACT_ANSWER``.
    """
    answer_text: str
    reasoning: str


# ---------------------------------------------------------------------------
# Heuristic rules
# ---------------------------------------------------------------------------


# Keywords/phrases in Claude's question that always escalate to the user.
# These are decisions Qwen cannot make alone: scope, money, secrets,
# external services, irreversible architecture choices.
_ALWAYS_ESCALATE_PATTERNS = re.compile(
    r"\b(?:"
    # External services / accounts the user must own / pay for
    r"api\s+key|secret|credential|token|password|"
    r"openai|anthropic|stripe|twilio|sendgrid|mailgun|"
    r"aws|azure|gcp|google\s+cloud|cloudflare|vercel|netlify|render|"
    r"paid\s+(?:tier|plan|account|service|subscription)|"
    r"upgrade\s+plan|free\s+tier\s+limit|"
    # Scope / product decisions
    r"add(?:ing)?\s+(?:another|new|additional)\s+feature|"
    r"out\s+of\s+scope|beyond\s+scope|expand\s+scope|"
    r"(?:should\s+i|do\s+you\s+want\s+(?:me\s+)?to)\s+(?:also|additionally)|"
    # Breaking choices
    r"breaking\s+change|backward(?:s)?[-\s]incompat|"
    r"different\s+(?:database|architecture|approach)|"
    # Deployment / infra
    r"deploy(?:ment)?\s+(?:target|host|to|where)|"
    r"production\s+(?:host|server|database)"
    r")\b",
    re.IGNORECASE,
)


# Keywords/phrases that always get a "use your default" / sensible-default
# answer without bothering the user. These are pure implementation details.
_ALWAYS_ANSWER_PATTERNS = re.compile(
    r"\b(?:"
    # Code structure
    r"file\s+naming|naming\s+(?:convention|pattern)|"
    r"directory\s+structure|project\s+layout|module\s+layout|"
    r"function\s+name|class\s+name|variable\s+name|"
    # Style
    r"style|formatting|indentation|line\s+length|"
    r"docstring\s+(?:style|format)|comment\s+style|"
    # Testing details
    r"test\s+framework|testing\s+library|"
    r"test\s+file\s+location|test\s+naming|"
    # Logging / errors
    r"logging\s+(?:library|format|level)|"
    r"error\s+(?:handling|format)|exception\s+(?:naming|hierarchy)|"
    # Standard tools
    r"linter|formatter|type\s+checker|"
    # Default config
    r"default\s+(?:port|timeout|retry|config)"
    r")\b",
    re.IGNORECASE,
)


# Conventional defaults for common implementation choices. The coordinator
# returns the matching default text when the question matches both an
# always-answer rule AND a known category.
_CONVENTIONAL_DEFAULTS: List[tuple[re.Pattern, str]] = [
    (re.compile(r"\btest\s+framework\b", re.I),
     "Use the language's standard test framework (pytest for Python, jest for "
     "Node/TypeScript, cargo test for Rust)."),
    (re.compile(r"\b(?:linter|formatter)\b", re.I),
     "Use the standard formatter for the language (black + ruff for Python, "
     "prettier for JS/TS). Default config."),
    (re.compile(r"\b(?:directory|project)\s+(?:structure|layout)\b", re.I),
     "Use the language's conventional layout: src/ for source, tests/ for tests, "
     "single top-level package matching the project name."),
    (re.compile(r"\b(?:file|function|class|variable)\s+(?:naming|name)", re.I),
     "Use the language's conventional naming (snake_case for Python identifiers, "
     "PascalCase for classes, lowercase_with_underscores for files). Pick clear "
     "descriptive names."),
    (re.compile(r"\bdocstring", re.I),
     "Concise docstrings on public functions explaining behavior and any "
     "non-obvious behavior. No multi-paragraph essays."),
    (re.compile(r"\blogging\s+(?:library|format)", re.I),
     "Use the standard logging module with a module-level logger. Default format."),
    (re.compile(r"\berror\s+handling|exception", re.I),
     "Catch only what you handle, propagate the rest. No bare except."),
]


def _match_conventional_default(question: str) -> Optional[str]:
    for pattern, default in _CONVENTIONAL_DEFAULTS:
        if pattern.search(question):
            return default
    return None


# ---------------------------------------------------------------------------
# LLM prompts (kept as constants for inspectability)
# ---------------------------------------------------------------------------


_DECIDE_PROMPT = """\
You are deciding how to handle a clarification request from a coding agent (Claude Code) working on a project for the user. Your job: answer Claude's question yourself when possible, escalate to the user only when the decision is substantive.

{projected_context}

Decide ONE of:
- ANSWER: you have enough info from the user's goal or general engineering judgment to answer directly. Provide a concrete answer.
- USE_DEFAULT: this is a low-stakes implementation detail; tell Claude to use its default approach.
- ESCALATE: this needs the user's input. Things that ALWAYS escalate: paid services, credentials, scope additions beyond the original goal, breaking architectural choices, deployment targets, anything that costs money or commits the user to an external service.

Output ONLY a JSON object, no commentary, no markdown:
{{"action": "ANSWER" | "USE_DEFAULT" | "ESCALATE", "answer": "<concrete answer text or null>", "reasoning": "<one sentence>"}}
"""


_VOICE_QUESTION_PROMPT = """\
Translate this technical clarification request into a natural, spoken-style question for the user. Stay in Ultron's voice -- precise, weighted, brief, no filler. Lead with the project context.

{projected_context}

Output the spoken question only -- one or two sentences, no preamble. Don't explain what you're doing, just ask. If options exist, mention them naturally.
"""


_ADJUSTMENT_PROMPT = """\
The user has given a mid-session adjustment to the in-progress coding work. Translate it into a concrete follow-up prompt for Claude.

{projected_context}

Decide whether Claude should pivot immediately or finish what's in progress first, and write the follow-up. Be specific. The follow-up will be sent verbatim to Claude.

Output the follow-up prompt only -- no preamble, no commentary. Keep it under 200 words.
"""


_CONFLICT_PROMPT = """\
Decide whether a user's mid-session adjustment conflicts with work already completed in the project.

Stages completed (newest last):
{stages_listing}
Files created or modified: {files_summary}

User's adjustment: "{user_text}"

A conflict means the adjustment would require redoing or discarding meaningful completed work (e.g., changing the database engine after the data layer is done, swapping the framework after routes are written).

Output ONLY a JSON object:
{{"is_conflict": <true|false>, "reason": "<one sentence>", "completed_at_risk": [<file paths>]}}
"""


# ---------------------------------------------------------------------------
# Decision-log writer
# ---------------------------------------------------------------------------


class _ClarificationLog:
    def __init__(self, path: Optional[Path]) -> None:
        self.path = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, **fields: Any) -> None:
        if self.path is None:
            return
        record = {"ts": datetime.now(timezone.utc).isoformat(), **fields}
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass


# ---------------------------------------------------------------------------
# ConversationCoordinator
# ---------------------------------------------------------------------------


class ConversationCoordinator:
    """Decision policy for clarifications + adjustments.

    Args:
        store: shared :class:`SessionStore` (the MCP server's).
        llm: an :class:`LLMEngine` to consult for ambiguous cases. May be
            ``None`` -- in that case ambiguous cases default to escalation
            (which is the safer side: false escalations are fixable; bad
            auto-answers are not).
        log_path: clarification audit log (JSONL). Default uses
            ``logs/clarifications.jsonl``.
        clarification_user_timeout_s: how long an escalated clarification
            can wait on a voice answer before timing out and falling back
            to "use your default".
    """

    def __init__(
        self,
        store: SessionStore,
        llm=None,
        *,
        log_path: Optional[Path] = None,
        clarification_user_timeout_s: float = 600.0,
        renderer: Optional[TemplateRenderer] = None,
        verifier: Optional[Verifier] = None,
        on_failed_session: Optional[Any] = None,
        facts_lookup: Optional[Callable[..., List[Dict[str, Any]]]] = None,
    ) -> None:
        self.store = store
        self.llm = llm
        self._log = _ClarificationLog(
            log_path or (settings.LOGS_DIR / "clarifications.jsonl")
        )
        self.clarification_user_timeout_s = clarification_user_timeout_s
        # Phase 1 (A3 wiring) -- callable that queries the Qdrant facts
        # collection. Signature: ``facts_lookup(query: str, *, k=...,
        # min_confidence=..., max_age_days=...) -> List[Dict[str, Any]]``.
        # When ``None``, ``decide_clarification`` skips the stored-facts
        # fast-path. The orchestrator wires this to
        # ``UltronMCPServer.lookup_facts`` (which proxies to
        # ``ConversationMemory.search_facts``).
        self._facts_lookup = facts_lookup
        # Phase 3 hook: when set, follow-up prompts the coordinator hands
        # back to the runner pass through the rendered template (which
        # enforces schema + token-budget). When None we emit the LLM /
        # ad-hoc text directly -- safe fallback if the templates dir is
        # missing or misconfigured.
        self.renderer = renderer
        # Phase 4 hook: when set, declare_complete runs the verifier and
        # the coordinator drives the correction loop. When None,
        # declare_complete trusts Claude's claim (Phase 1 behavior).
        self.verifier = verifier
        # Optional callback for the runner: invoked when a session
        # transitions to FAILED so the runner can surface a final
        # narration and tear down. Signature: ``cb(session_id, reason)``.
        self.on_failed_session = on_failed_session
        # request_id -> (loop, future) waiting on a voice answer
        self._user_pending: Dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Future]] = {}
        # request_id -> PendingUserClarification (surfaced to voice loop)
        self._user_pending_meta: Dict[str, PendingUserClarification] = {}
        self._lock = threading.Lock()
        # Verifier audit log -- one line per declare_complete cycle.
        self._verify_log = _ClarificationLog(
            settings.LOGS_DIR / "verifications.jsonl"
        )

    # -----------------------------------------------------------------------
    # Public entry points
    # -----------------------------------------------------------------------

    async def decide_clarification(
        self,
        session_id: str,
        request: ClarificationRequest,
        session: ProjectSession,
    ) -> str:
        """Return the answer string Claude should receive.

        This is the responder installed on
        :meth:`UltronMCPServer.set_clarification_responder`. It runs in
        the MCP server's asyncio loop. Sync LLM calls dispatched via
        ``run_in_executor`` so we don't block the event loop thread.
        """
        question = (request.question or "").strip()
        options = list(request.options or [])
        urgency = request.urgency

        # Fast-path 1: always-escalate keyword.
        if _ALWAYS_ESCALATE_PATTERNS.search(question):
            answer = await self._escalate_to_user(
                session_id, request, session,
                decision_path=DecisionPath.RULE_ESCALATE,
                reasoning="matches always-escalate rule",
            )
            return answer

        # Fast-path 2: preference + Claude has options -> use default.
        if urgency == "preference" and options:
            return self._respond_use_default(
                session_id, request,
                decision_path=DecisionPath.RULE_DEFAULT,
                reasoning="urgency=preference with options; defaulting",
            )

        # Fast-path 2.5: stored facts. If a high-confidence fact directly
        # addresses the question, answer from it. Cheaper and more
        # consistent than burning an LLM call -- and skips an unnecessary
        # escalation when the user's stored preferences already cover the
        # question. Categories are restricted to ones whose facts map
        # cleanly to "answer Claude directly" (preference / decision /
        # constraint); 'person' / 'project' get logged but don't auto-
        # answer because their content is descriptive, not directive.
        fact_answer = self._maybe_answer_from_facts(question)
        if fact_answer is not None:
            return self._respond_with(
                session_id, request,
                answer=fact_answer.answer_text,
                decision_path=DecisionPath.FACT_ANSWER,
                reasoning=fact_answer.reasoning,
            )

        # Fast-path 3: known low-stakes implementation question.
        if _ALWAYS_ANSWER_PATTERNS.search(question):
            default = _match_conventional_default(question) or "Use your default approach."
            return self._respond_with(
                session_id, request,
                answer=default,
                decision_path=DecisionPath.RULE_ANSWER,
                reasoning="matches always-answer rule",
            )

        # Slow path: LLM decision.
        if self.llm is None:
            answer = await self._escalate_to_user(
                session_id, request, session,
                decision_path=DecisionPath.RULE_ESCALATE,
                reasoning="no LLM available; escalating",
            )
            return answer

        decision = await self._llm_decide(request, session)
        if decision.action == "ANSWER" and decision.answer:
            return self._respond_with(
                session_id, request,
                answer=decision.answer,
                decision_path=DecisionPath.LLM_ANSWER,
                reasoning=decision.reasoning,
            )
        if decision.action == "USE_DEFAULT":
            return self._respond_use_default(
                session_id, request,
                decision_path=DecisionPath.LLM_DEFAULT,
                reasoning=decision.reasoning,
            )
        # ESCALATE (default)
        return await self._escalate_to_user(
            session_id, request, session,
            decision_path=DecisionPath.LLM_ESCALATE,
            reasoning=decision.reasoning,
        )

    def deliver_user_clarification_response(
        self, request_id: str, answer: str,
    ) -> bool:
        """Voice controller calls this when the user has spoken a response
        to a previously-escalated clarification.

        Returns True if a waiter was actually resolved.
        """
        with self._lock:
            entry = self._user_pending.pop(request_id, None)
            self._user_pending_meta.pop(request_id, None)
        if entry is None:
            return False
        loop, future = entry

        def _set() -> None:
            if not future.done():
                future.set_result(answer)
        try:
            loop.call_soon_threadsafe(_set)
        except RuntimeError:
            # Loop closed; nothing we can do.
            return False
        return True

    def pending_user_clarifications(self) -> List[PendingUserClarification]:
        """Snapshot of clarifications awaiting voice answers."""
        with self._lock:
            return list(self._user_pending_meta.values())

    async def decide_adjustment(
        self,
        session_id: str,
        user_text: str,
    ) -> AdjustmentDecision:
        """Decide what to do with a user's mid-session adjustment."""
        session = self.store.get(session_id)

        # If there's a pending clarification awaiting the user, the
        # adjustment is most likely an answer to it -- resolve and return
        # an ESCALATE-resolved style.
        with self._lock:
            pending_meta = list(self._user_pending_meta.values())
            target = next(
                (m for m in pending_meta if m.session_id == session_id),
                None,
            )
        if target is not None:
            self.deliver_user_clarification_response(target.request_id, user_text)
            return AdjustmentDecision(
                action="FOLLOWUP",
                followup_prompt=None,
                conflict_reason=None,
                voice_question=None,
            )

        # Conflict detection (LLM-based).
        if self.llm is not None and session.stages_completed:
            try:
                conflict = await self._llm_detect_conflict(session, user_text)
            except Exception as e:
                logger.warning("conflict detection failed (%s); proceeding", e)
                conflict = {"is_conflict": False, "reason": "", "completed_at_risk": []}
            if conflict.get("is_conflict"):
                voice_question = (
                    f"He's already finished {self._stages_summary(session)}. "
                    f"This adjustment would mean redoing some of that. "
                    f"Should I have him pivot now or finish what's in progress first?"
                )
                return AdjustmentDecision(
                    action="ESCALATE_CONFLICT",
                    followup_prompt=None,
                    conflict_reason=str(conflict.get("reason", "")),
                    voice_question=voice_question,
                )

        # Translate to a follow-up prompt.
        if self.llm is None:
            # No LLM -- pass the user's text through verbatim.
            llm_followup = (
                f"The user has issued a mid-session adjustment: {user_text!r}. "
                f"Apply it to the in-progress work. Continue from where you are."
            )
        else:
            llm_followup = await self._llm_render_adjustment_prompt(session, user_text)

        # Wrap through the adjustment template when a renderer is wired,
        # so the prompt goes through schema + token-budget validation
        # before we hand it to the runner.
        prompt = self._render_adjustment_via_template(session, user_text, llm_followup)
        return AdjustmentDecision(
            action="FOLLOWUP",
            followup_prompt=prompt,
            conflict_reason=None,
            voice_question=None,
        )

    async def handle_declare_complete(self, session_id: str) -> str:
        """Drive the verification loop after Claude's declare_complete.

        Returns the text that should be sent back to Claude as the tool
        call response: success message on pass, correction prompt on
        fail, give-up text once we exhaust escalation.
        """
        session = self.store.get(session_id)
        if session.completion_claim is None:
            return (
                "Internal error: declare_complete recorded no claim. "
                "Please call declare_complete with a complete payload."
            )

        if self.verifier is None:
            # No verifier wired -- trust the claim. Phase 1 fallback.
            self._safe_transition(session_id, SessionStatus.COMPLETE)
            return (
                "Verification skipped (no verifier). Completion accepted."
            )

        # Run the verifier on a worker thread so the asyncio event loop
        # serving Claude doesn't stall on the subprocesses.
        loop = asyncio.get_running_loop()
        try:
            report: VerificationReport = await loop.run_in_executor(
                None, self.verifier.verify, session_id,
            )
        except Exception as e:
            logger.warning("verifier raised: %s", e)
            return (
                f"Verification could not run: {e}. The supervisor will "
                f"surface this to the user."
            )
        # Audit log every verification cycle.
        self._verify_log.write(
            session_id=session_id,
            passed=report.passed,
            failure_count_before=session.verification_failures,
            checks=[
                {
                    "check": c.check.value,
                    "passed": c.passed,
                    "skipped": c.skipped,
                    "duration_ms": c.duration_ms,
                    "detail": c.detail[:400],
                }
                for c in report.checks
            ],
            duration_s=report.duration_s,
        )

        # Phase 7: mirror the verification result into the per-session log.
        self._session_audit(
            session_id, "verification_completed",
            passed=report.passed,
            duration_s=report.duration_s,
            checks=[
                {
                    "check": c.check.value, "passed": c.passed,
                    "skipped": c.skipped, "duration_ms": c.duration_ms,
                } for c in report.checks
            ],
        )

        if report.passed:
            self._safe_transition(session_id, SessionStatus.COMPLETE)
            return (
                f"Verification passed ({report.duration_s:.1f}s, "
                f"{len(report.checks) - report.skipped_count} checks ran). "
                f"Project complete."
            )

        # Failure path: bump count, decide escalation.
        new_count = session.verification_failures + 1
        haiku_thresh = settings.CODING_ESCALATION_THRESHOLD_DEFAULT
        sonnet_thresh = settings.CODING_ESCALATION_THRESHOLD_ESCALATION
        total_threshold = haiku_thresh + sonnet_thresh

        with self._lock:
            session_now = self.store.get(session_id)
            session_now.verification_failures = new_count

        if new_count >= total_threshold:
            self._safe_transition(session_id, SessionStatus.FAILED)
            reason = (
                f"verification has failed {new_count} times across both "
                f"the default and escalation models"
            )
            if self.on_failed_session is not None:
                try:
                    self.on_failed_session(session_id, reason)
                except Exception as e:
                    logger.debug("on_failed_session callback error: %s", e)
            return (
                f"Verification has failed {new_count} times. The supervisor "
                f"is surfacing this to the user. Stop here and summarize "
                f"what's blocking you in your final message."
            )

        # Mark escalation if we just crossed the haiku threshold.
        if new_count >= haiku_thresh:
            with self._lock:
                self.store.get(session_id).model_escalation_count = max(
                    1, self.store.get(session_id).model_escalation_count,
                )

        # Render the correction prompt (uses the Phase 3 template if
        # available; falls back to plaintext otherwise).
        correction = self.render_correction_prompt(
            project_root=session.project_root,
            failures=report.to_correction_failures(),
            verification_failure_count=new_count - 1,
        )
        self._safe_transition(session_id, SessionStatus.CORRECTING)
        # Then back to EXECUTING -- Claude is going to keep working in
        # the same subprocess.
        self._safe_transition(session_id, SessionStatus.EXECUTING)
        return correction

    def _safe_transition(self, session_id: str, target: SessionStatus) -> None:
        try:
            self.store.transition(session_id, target)
        except Exception as e:
            logger.debug(
                "transition to %s skipped for %s: %s",
                target.value, session_id, e,
            )

    def render_correction_prompt(
        self,
        *,
        project_root: Path,
        failures: List[Dict[str, str]],
        verification_failure_count: int = 0,
    ) -> str:
        """Render the post-verification-failure correction prompt for the
        runner to send via send_followup(kind='correction').

        Used by the verifier in Phase 4. Falls back to a plain-text
        rendering if no template renderer is wired (the system still
        works, just without schema validation).
        """
        if self.renderer is None:
            return self._fallback_correction_text(failures, verification_failure_count)
        try:
            result = self.renderer.render_correction(
                project_root=project_root,
                failures=failures,
                verification_failure_count=verification_failure_count,
            )
            return result.text
        except (SchemaValidationError, PromptTooLargeError) as e:
            logger.warning(
                "correction template render failed (%s); using fallback", e,
            )
            return self._fallback_correction_text(failures, verification_failure_count)

    # --- internals: template wrapping --------------------------------------

    def _render_adjustment_via_template(
        self,
        session: ProjectSession,
        user_text: str,
        llm_followup: str,
    ) -> str:
        if self.renderer is None:
            return llm_followup
        try:
            result = self.renderer.render_adjustment(
                user_text=user_text,
                current_stage=session.current_stage,
                stages_summary=self._stages_summary(session),
                files_summary=self._files_summary(session),
                pivot_immediately=True,
                coordinator_followup=llm_followup,
            )
            return result.text
        except (SchemaValidationError, PromptTooLargeError) as e:
            logger.warning(
                "adjustment template render failed (%s); using LLM text directly", e,
            )
            return llm_followup

    @staticmethod
    def _fallback_correction_text(
        failures: List[Dict[str, str]],
        verification_failure_count: int,
    ) -> str:
        lines = ["# Verification failed", ""]
        lines.append(
            "You called declare_complete, but verification turned up these "
            "problems. Fix them and call declare_complete again when done.",
        )
        lines.append("")
        for i, f in enumerate(failures, 1):
            lines.append(f"## {i}. {f.get('check', '')}")
            lines.append(str(f.get("detail", "")).strip())
            hint = f.get("hint")
            if hint:
                lines.append(f"Hint: {hint}")
            lines.append("")
        lines.append("## What to do now")
        lines.append("- Fix each failure above.")
        lines.append("- Re-run tests; report counts via report_test_results.")
        lines.append("- Don't weaken or delete tests.")
        if verification_failure_count >= 1:
            lines.append("")
            lines.append(
                f"(This is the {verification_failure_count + 1}{'nd' if verification_failure_count == 1 else 'rd' if verification_failure_count == 2 else 'th'} verification cycle. Be deliberate.)"
            )
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Internals: stored-facts fast-path (A3 wiring)
    # -----------------------------------------------------------------------

    # Categories from extract_facts that we trust to auto-answer Claude
    # without escalation. 'person' and 'project' are descriptive, not
    # directive, so they don't qualify.
    _DIRECTIVE_FACT_CATEGORIES = {"preference", "decision", "constraint"}

    def _maybe_answer_from_facts(
        self, question: str,
    ) -> Optional[_FactAnswer]:
        """Return a structured answer from a high-confidence stored fact,
        or ``None`` to fall through to the next decision path.

        The fact must:
          * exist (lookup non-empty),
          * have ``confidence >= settings.CODING_FACTS_MIN_CONFIDENCE``,
          * have ``score >= settings.CODING_FACTS_MIN_SCORE``,
          * have ``category`` in :attr:`_DIRECTIVE_FACT_CATEGORIES`.

        Failures inside ``facts_lookup`` are swallowed -- the next
        decision path (always-answer / LLM) handles the request normally.
        """
        if self._facts_lookup is None:
            return None
        cfg = settings.CODING_FACTS
        try:
            rows = self._facts_lookup(
                question,
                k=cfg["top_k"],
                min_confidence=cfg["min_confidence"],
                max_age_days=cfg["max_age_days"],
            )
        except TypeError:
            try:
                rows = self._facts_lookup(question)
            except Exception as e:
                logger.debug("facts_lookup raised: %s", e)
                return None
        except Exception as e:
            logger.debug("facts_lookup raised: %s", e)
            return None
        if not rows:
            return None
        top = rows[0]
        try:
            confidence = float(top.get("confidence", 0.0))
            score = float(top.get("score", 0.0))
            category = str(top.get("category", "")).strip().lower()
            fact_text = str(top.get("fact", "")).strip()
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug("facts_lookup malformed top row: %s", e)
            return None
        if not fact_text:
            return None
        if confidence < cfg["min_confidence"]:
            return None
        if score < cfg["min_score"]:
            return None
        if category not in self._DIRECTIVE_FACT_CATEGORIES:
            return None
        # Wrap the fact so Claude knows the source. Keeping it explicit
        # lets Claude judge whether the fact is stale and re-call
        # request_clarification with rephrased context.
        answer_text = (
            f"From the user's stored preferences: {fact_text} Use that."
        )
        reasoning = (
            f"matched stored fact (category={category}, "
            f"confidence={confidence:.2f}, score={score:.2f})"
        )
        return _FactAnswer(answer_text=answer_text, reasoning=reasoning)

    # -----------------------------------------------------------------------
    # Internals: rule-based answers
    # -----------------------------------------------------------------------

    def _respond_use_default(
        self,
        session_id: str,
        request: ClarificationRequest,
        *,
        decision_path: DecisionPath,
        reasoning: str,
    ) -> str:
        return self._respond_with(
            session_id, request,
            answer="Use your default approach.",
            decision_path=decision_path,
            reasoning=reasoning,
        )

    def _respond_with(
        self,
        session_id: str,
        request: ClarificationRequest,
        *,
        answer: str,
        decision_path: DecisionPath,
        reasoning: str,
    ) -> str:
        self.store.resolve_clarification(session_id, answer, decision_path.value)
        self._log.write(
            session_id=session_id,
            request_id=request.request_id,
            question=request.question,
            urgency=request.urgency,
            options=request.options,
            decision_path=decision_path.value,
            answer=answer,
            reasoning=reasoning,
        )
        # Phase 7: mirror the decision into the per-session audit log.
        self._session_audit(
            session_id, "clarification_decided",
            request_id=request.request_id,
            decision_path=decision_path.value,
            answer=(answer or "")[:300],
            reasoning=(reasoning or "")[:300],
        )
        try:
            session = self.store.get(session_id)
            if session.status == SessionStatus.AWAITING_CLARIFICATION:
                self.store.transition(session_id, SessionStatus.EXECUTING)
        except Exception:
            pass
        return answer

    def _session_audit(self, session_id: str, event: str, **fields: Any) -> None:
        """Phase 7 helper: write to the per-session audit log when the
        store has one wired."""
        writer = getattr(self.store, "audit_writer", None)
        if writer is None:
            return
        try:
            writer.write(session_id, event, **fields)
        except Exception as e:
            logger.debug("session audit write failed: %s", e)

    # -----------------------------------------------------------------------
    # Internals: LLM-driven decisions
    # -----------------------------------------------------------------------

    async def _llm_decide(
        self, request: ClarificationRequest, session: ProjectSession,
    ) -> ClarificationDecision:
        # Phase C / Phase 1: build the bounded projection rather than
        # serializing the whole session. Prevents context-budget overflow
        # on long-running sessions.
        from ultron.coding.projections import project_clarification_context
        projection = project_clarification_context(
            session,
            clarification_question=request.question,
            options=list(request.options or []),
        )
        prompt = _DECIDE_PROMPT.format(projected_context=projection.text)
        raw = await self._llm_generate(prompt, max_tokens=512)
        parsed = _parse_json_object(raw) or {}
        action = str(parsed.get("action", "ESCALATE")).upper()
        if action not in ("ANSWER", "USE_DEFAULT", "ESCALATE"):
            action = "ESCALATE"
        answer = parsed.get("answer")
        if isinstance(answer, str):
            answer = answer.strip() or None
        else:
            answer = None
        reasoning = str(parsed.get("reasoning", "") or "")[:240]
        return ClarificationDecision(
            action=action, answer=answer, reasoning=reasoning,
            decision_path=DecisionPath.LLM_ANSWER,  # placeholder; caller sets
        )

    async def _llm_render_voice_question(
        self,
        session: ProjectSession,
        request: ClarificationRequest,
    ) -> str:
        if self.llm is None:
            opts = (
                f" Options he gave: {', '.join(request.options)}."
                if request.options else ""
            )
            return (
                f"Claude needs a clarification on the {session.refined_goal} project: "
                f"{request.question}.{opts}"
            )
        # Phase C / Phase 1: same bounded projection feeds the voice-rendering
        # prompt -- single source of truth for what Qwen sees.
        from ultron.coding.projections import project_clarification_context
        projection = project_clarification_context(
            session,
            clarification_question=request.question,
            options=list(request.options or []),
        )
        prompt = _VOICE_QUESTION_PROMPT.format(projected_context=projection.text)
        text = await self._llm_generate(prompt, max_tokens=200)
        text = _strip_thinking(text).strip()
        # First non-empty line.
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line[:400]
        return f"Claude needs a clarification on the project: {request.question}"

    async def _llm_render_adjustment_prompt(
        self,
        session: ProjectSession,
        user_text: str,
    ) -> str:
        # Phase C / Phase 1: bounded projection.
        from ultron.coding.projections import project_adjustment_context
        projection = project_adjustment_context(
            session, adjustment_text=user_text,
        )
        prompt = _ADJUSTMENT_PROMPT.format(projected_context=projection.text)
        text = await self._llm_generate(prompt, max_tokens=512)
        text = _strip_thinking(text).strip()
        return text or f"User adjustment: {user_text!r}. Continue from where you are."

    async def _llm_detect_conflict(
        self,
        session: ProjectSession,
        user_text: str,
    ) -> Dict[str, Any]:
        listing = "\n".join(
            f"- [{i+1}] {s.stage}: {s.summary} (files: {', '.join(s.files_touched) or '(none)'})"
            for i, s in enumerate(session.stages_completed)
        ) or "(none)"
        prompt = _CONFLICT_PROMPT.format(
            stages_listing=listing,
            files_summary=self._files_summary(session),
            user_text=user_text,
        )
        raw = await self._llm_generate(prompt, max_tokens=256)
        return _parse_json_object(raw) or {"is_conflict": False, "reason": "parse failed"}

    async def _llm_generate(self, prompt: str, *, max_tokens: int) -> str:
        """Run the sync LLM in a thread so we don't stall the asyncio loop."""
        if self.llm is None:
            return ""
        loop = asyncio.get_running_loop()

        def _call() -> str:
            old = settings.LLM_MAX_TOKENS
            try:
                settings.LLM_MAX_TOKENS = max_tokens
                return self.llm.generate(prompt)
            finally:
                settings.LLM_MAX_TOKENS = old

        try:
            return await loop.run_in_executor(None, _call)
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return ""

    # -----------------------------------------------------------------------
    # Internals: escalation / pending-user
    # -----------------------------------------------------------------------

    async def _escalate_to_user(
        self,
        session_id: str,
        request: ClarificationRequest,
        session: ProjectSession,
        *,
        decision_path: DecisionPath,
        reasoning: str,
    ) -> str:
        voice_question = await self._llm_render_voice_question(session, request)
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        meta = PendingUserClarification(
            request_id=request.request_id,
            session_id=session_id,
            voice_question=voice_question,
            options=list(request.options or []),
            raised_at=time.time(),
        )
        with self._lock:
            self._user_pending[request.request_id] = (loop, future)
            self._user_pending_meta[request.request_id] = meta
        try:
            self.store.transition(session_id, SessionStatus.AWAITING_USER)
        except Exception:
            pass

        self._log.write(
            session_id=session_id,
            request_id=request.request_id,
            question=request.question,
            urgency=request.urgency,
            options=request.options,
            decision_path=decision_path.value,
            voice_question=voice_question,
            reasoning=reasoning,
            escalated=True,
        )

        try:
            answer = await asyncio.wait_for(
                future, timeout=self.clarification_user_timeout_s,
            )
        except asyncio.TimeoutError:
            with self._lock:
                self._user_pending.pop(request.request_id, None)
                self._user_pending_meta.pop(request.request_id, None)
            answer = "Use your default approach -- the user did not respond in time."
            self._log.write(
                session_id=session_id, request_id=request.request_id,
                decision_path=DecisionPath.TIMEOUT_DEFAULT.value,
                answer=answer, escalated=False,
            )

        # Record the resolution + transition back to executing.
        self.store.resolve_clarification(
            session_id, answer, DecisionPath.USER_ANSWER.value,
        )
        try:
            session_now = self.store.get(session_id)
            if session_now.status in (
                SessionStatus.AWAITING_USER, SessionStatus.AWAITING_CLARIFICATION,
            ):
                self.store.transition(session_id, SessionStatus.EXECUTING)
        except Exception:
            pass
        return answer

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _stages_summary(session: ProjectSession) -> str:
        if not session.stages_completed:
            return "(nothing yet)"
        # Last 3 stages, newest first.
        recent = list(reversed(session.stages_completed[-3:]))
        return "; ".join(f"{s.stage}: {s.summary[:80]}" for s in recent)

    @staticmethod
    def _files_summary(session: ProjectSession) -> str:
        created = [f.path for f in session.files_created][:6]
        modified = [f.path for f in session.files_modified][:6]
        if not created and not modified:
            return "(none)"
        bits = []
        if created:
            bits.append("created: " + ", ".join(created))
        if modified:
            bits.append("modified: " + ", ".join(modified))
        return "; ".join(bits)


# ---------------------------------------------------------------------------
# Lightweight JSON / thinking-block parsers (mirrors maintenance.py)
# ---------------------------------------------------------------------------


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = _strip_thinking(text)
    if not text:
        return None
    candidates: List[str] = []
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1))
    i = text.find("{")
    if i != -1:
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[i: j + 1])
                    break
    candidates.append(text)
    for c in candidates:
        try:
            v = json.loads(c)
            if isinstance(v, dict):
                return v
        except (json.JSONDecodeError, TypeError):
            continue
    return None
