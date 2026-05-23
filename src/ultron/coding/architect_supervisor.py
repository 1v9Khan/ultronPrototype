"""Architect-supervisor: pre-dispatch planning via a local LLM.

Pattern lifted in spirit (not in source) from aider's
``architect_coder.py`` + ``architect_prompts.py`` (Apache 2.0; see
``THIRD_PARTY_NOTICES.md``). This is the catalog T5 Phase 1 ("Qwen
produces a plan") delivered in isolation; Phase 2 (TTS narration of
the plan with barge-in window) and Phase 3 (forward plan to the
editor coder) are intentionally deferred to a follow-up because they
touch the voice hot path and need a fresh measure_baseline.py pass.

What this module does (Phase 1 only):

  1. Caller hands the architect: user utterance + optional repo-map
     text (from batch 2) + optional project digest.
  2. Architect renders a prompt asking a *cheap* local LLM (Qwen 3.5
     4B by default in production) to describe — in prose, NOT in
     SEARCH/REPLACE blocks — how an editor LLM should modify the
     code. The architect's system prompt forbids producing full file
     contents or fenced code blocks: it's a planner, not an editor.
  3. The architect returns an :class:`ArchitectPlan` carrying the
     prose plan text + telemetry (token estimates, generation
     wall-time, fallback-cascade index).

The architect callable cascade mirrors :func:`generate_commit_message`
from batch 5: try the primary local LLM; on exception or empty
response, fall through to a secondary (typically the same model with
a different decoding config, or — in a future wiring — a remote LLM).

Public surface:

  * :class:`ArchitectPlan` — frozen output dataclass.
  * :class:`ArchitectRequest` — frozen input bundle.
  * :class:`ArchitectSupervisor` — instantiable class holding the LLM
    cascade + default system prompt.
  * :data:`DEFAULT_ARCHITECT_SYSTEM_PROMPT` — the catalog's exact
    architect prompt, customised slightly for ultron's hygiene rules.

Fail-open: every architect failure mode returns an
:class:`ArchitectPlan` with ``plan_text=None`` and ``error`` set.
Callers should treat this as "no plan available; proceed without
one" — never as "abort the dispatch".

Production wiring (when ``coding.architect.enabled`` is on AND the
voice baseline has been validated): the orchestrator constructs an
``ArchitectSupervisor`` with the in-process LLM as the primary
callable, then passes it to ``ProjectSupervisor`` as the
``architect_provider``. The supervisor invokes the provider AFTER
deciding on a target project (so the architect has a real
project_path) and attaches the resulting plan text to the decision's
``architect_plan_text`` field — parallel to ``repo_map_text``.
Downstream callers (``supervisor_dispatch``) prepend the plan to the
Claude prompt body.

The architect deliberately does NOT call any TTS or voice machinery
in this module. The catalog's Phase 2 (narrate the plan, open a
barge-in window) lives in a future PR.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence


logger = logging.getLogger("ultron.coding.architect_supervisor")


# Catalog T5 source-of-pattern: aider's architect_prompts.py
# main_system. Light ultron customisation:
#   * Explicit "no AI-assistant attribution" rule for hygiene parity
#     with the commit-message system prompt.
#   * Slightly tighter "don't produce code; describe the changes"
#     phrasing because Qwen 3.5 4B tends to over-eagerly emit code
#     blocks otherwise.
DEFAULT_ARCHITECT_SYSTEM_PROMPT = (
    "Act as an expert architect engineer and provide clear, complete "
    "direction to an editor engineer who will make the actual code "
    "changes. Study the change request and the current code carefully, "
    "then describe HOW to modify the code to complete the request.\n"
    "\n"
    "Rules:\n"
    "- The editor will rely SOLELY on your instructions, so make them "
    "unambiguous, specific, and complete.\n"
    "- Show only the CHANGES needed. Do NOT show entire updated "
    "functions, classes, or files.\n"
    "- Describe changes in prose — do NOT emit fenced code blocks, "
    "SEARCH/REPLACE blocks, or unified diffs. The editor's job is to "
    "translate your prose into edits.\n"
    "- Reference specific file paths, function names, and line ranges "
    "where helpful.\n"
    "- Concise but complete. Skip what the editor can infer; spell out "
    "what is non-obvious.\n"
    "- No author attribution, no AI assistant mentions, no Co-Authored-By "
    "trailers."
)


@dataclass(frozen=True)
class ArchitectRequest:
    """Inputs the architect needs to produce a plan."""

    user_text: str
    repo_map_text: Optional[str] = None
    project_digest: Optional[str] = None
    project_path: Optional[str] = None
    system_prompt: str = DEFAULT_ARCHITECT_SYSTEM_PROMPT
    # Per-LLM input budgets in characters. Cascade entries beyond the
    # last value reuse the last value.
    max_prompt_chars_per_llm: Sequence[int] = field(default_factory=lambda: (32000,))


@dataclass(frozen=True)
class ArchitectPlan:
    """Architect's prose plan + telemetry.

    Attributes:
        plan_text: The architect's prose direction to the editor.
            ``None`` on any failure path; ``error`` carries the cause.
        cascade_index: Which LLM in the cascade produced the plan
            (``-1`` when none did).
        prompt_chars: Length of the rendered prompt that produced the
            plan. Useful for budget tuning.
        generation_seconds: Wall-clock duration of the successful LLM
            call. 0.0 when no LLM succeeded.
        error: Empty when ``plan_text`` is non-None; otherwise a
            short failure reason.
        last_exception: Stringified last exception encountered in
            the cascade (when applicable).
    """

    plan_text: Optional[str]
    cascade_index: int = -1
    prompt_chars: int = 0
    generation_seconds: float = 0.0
    error: str = ""
    last_exception: Optional[str] = None


# Cascade entry: (prompt_text) -> generated_text.
ArchitectLLMCallable = Callable[[str], str]


class ArchitectSupervisor:
    """Pre-dispatch planning supervisor.

    Args:
        llm_cascade: Ordered LLM callables. First non-failing call
            wins. Each receives the fully rendered prompt and returns
            the model's output text.
        default_system_prompt: Override the architect system prompt.
            Caller-controlled because some operators want a tighter
            "describe in three bullet points" style while others want
            the full catalog prompt.
        strip_outer_quotes: Run the LLM output through
            :func:`ultron.coding.commit_message.strip_outer_quotes`
            so wrapped responses come back clean. Defaults to False
            because the architect's prose typically has internal
            quotes the user wants preserved.
    """

    def __init__(
        self,
        llm_cascade: Sequence[ArchitectLLMCallable],
        *,
        default_system_prompt: str = DEFAULT_ARCHITECT_SYSTEM_PROMPT,
        strip_outer_quotes: bool = False,
    ) -> None:
        if not llm_cascade:
            raise ValueError("ArchitectSupervisor: cascade must be non-empty")
        self._cascade = list(llm_cascade)
        self._default_system_prompt = default_system_prompt
        self._strip_quotes = strip_outer_quotes

    def generate_plan(self, request: ArchitectRequest) -> ArchitectPlan:
        """Run the cascade and return an :class:`ArchitectPlan`."""
        if not request.user_text or not request.user_text.strip():
            return ArchitectPlan(plan_text=None, error="empty user_text")

        prompt = self._render_prompt(request)
        prompt_chars = len(prompt)

        budgets = list(request.max_prompt_chars_per_llm) or [32000]
        while len(budgets) < len(self._cascade):
            budgets.append(budgets[-1])

        last_exception: Optional[BaseException] = None
        any_within_budget = False

        for idx, llm in enumerate(self._cascade):
            if prompt_chars > budgets[idx]:
                logger.debug(
                    "architect: cascade[%d] skipped — prompt %d chars > %d",
                    idx, prompt_chars, budgets[idx],
                )
                continue
            any_within_budget = True
            start = time.monotonic()
            try:
                raw = llm(prompt) or ""
            except Exception as exc:                          # noqa: BLE001
                last_exception = exc
                logger.debug("architect: cascade[%d] raised: %s", idx, exc)
                continue
            elapsed = time.monotonic() - start
            plan = self._post_process(raw)
            if not plan:
                logger.debug("architect: cascade[%d] returned empty", idx)
                continue
            return ArchitectPlan(
                plan_text=plan,
                cascade_index=idx,
                prompt_chars=prompt_chars,
                generation_seconds=elapsed,
            )

        if not any_within_budget:
            return ArchitectPlan(
                plan_text=None,
                prompt_chars=prompt_chars,
                error="prompt too large for every LLM in the cascade",
            )
        return ArchitectPlan(
            plan_text=None,
            prompt_chars=prompt_chars,
            error="all LLMs failed",
            last_exception=str(last_exception) if last_exception else None,
        )

    # ------------------------------------------------------------------
    # Provider contract — matches ProjectSupervisor.architect_provider
    # ------------------------------------------------------------------

    def __call__(
        self,
        project_path: str,
        user_text: str,
        *,
        repo_map_text: Optional[str] = None,
        project_digest: Optional[str] = None,
    ) -> Optional[str]:
        """Provider entry point. Returns the plan text or ``None``.

        Compatible with the optional kwargs (``repo_map_text``,
        ``project_digest``) so callers that have these can pass them
        through. ProjectSupervisor's basic ``(project_path, user_text)``
        signature still works — the kwargs default to ``None``.
        """
        try:
            result = self.generate_plan(ArchitectRequest(
                user_text=user_text,
                repo_map_text=repo_map_text,
                project_digest=project_digest,
                project_path=project_path,
            ))
        except Exception as exc:                              # noqa: BLE001
            logger.warning("architect provider raised: %s", exc)
            return None
        return result.plan_text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render_prompt(self, request: ArchitectRequest) -> str:
        parts: List[str] = []
        parts.append(request.system_prompt or self._default_system_prompt)
        if request.project_path:
            parts.append(f"Project path: {request.project_path}")
        if request.repo_map_text:
            parts.append(f"Repo map (PageRank-ranked):\n{request.repo_map_text}")
        if request.project_digest:
            parts.append(f"Project digest:\n{request.project_digest}")
        parts.append(f"User request:\n{request.user_text}")
        parts.append("Plan:")
        return "\n\n".join(parts)

    def _post_process(self, raw: str) -> str:
        text = (raw or "").strip()
        if not text:
            return ""
        if self._strip_quotes:
            from ultron.coding.commit_message import strip_outer_quotes
            text = strip_outer_quotes(text)
        return text


__all__ = [
    "ArchitectLLMCallable",
    "ArchitectPlan",
    "ArchitectRequest",
    "ArchitectSupervisor",
    "DEFAULT_ARCHITECT_SYSTEM_PROMPT",
]
