"""Goal-anchor planning (E2): decompose a coding task into named
milestones with per-anchor token budgets, then track per-anchor
progress against those budgets.

Motivation. The 2026-05-11 live session burned 134k tokens on a
PDF->DOCX task that produced zero files (the discipline preamble
prepended a "must write tests, run, fix, re-run" loop to a small
voice ask). With goal-anchors the same task would have hit anchor
"scaffold project" at ~5k tokens, anchor "implement converter" at
~15k tokens, anchor "wire GUI" at ~10k tokens -- and the orchestrator
could have surfaced budget-exhaustion narration mid-task, OR resumed
the next anchor on a follow-up "continue" without restarting.

V1 ships only the primitives:

* :func:`decompose_into_anchors` -- heuristic decomposition (no LLM
  call). Splits on imperative verbs + clause connectives; falls back
  to single "complete the task" anchor when no decomposition is found.
* :class:`AnchorBudget` -- per-anchor token-budget tracker.
* :class:`AnchorPlan` -- the assembled plan that consumers iterate.

Runner integration (per-anchor narration, resume-from-incomplete
support, budget-exhaustion handling) is a behaviour change deferred
to a follow-up so this commit stays no-regression.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional


# Imperative-verb stems that frequently mark the start of a new
# sub-goal in a multi-step coding task. The list is intentionally
# tight -- generic verbs ("do", "make") create false splits on
# narrative phrasing ("make sure", "do you also want").
_ANCHOR_VERBS = (
    "build",
    "implement",
    "write",
    "create",
    "add",
    "wire",
    "scaffold",
    "set up",
    "setup",
    "configure",
    "install",
    "deploy",
    "test",
    "verify",
    "document",
    "refactor",
    "fix",
    "migrate",
    "convert",
    "render",
)

# Clause connectives that signal the next clause starts a fresh
# sub-goal. Includes comma as a boundary so prompts in the natural
# "Build X, then test Y, finally deploy Z." form split cleanly.
_CONNECTIVES = re.compile(
    r"(?:^|[.;,!?\n])\s*(?:then|next|after\s+that|after\s+this|"
    r"finally|lastly|once\s+(?:that|this|done)|"
    r"and\s+then|and\s+also|and\s+finally)\b\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GoalAnchor:
    """One named milestone in a decomposed task."""

    name: str           # short slug ("scaffold_project", "implement_converter")
    description: str    # human-readable description for narration
    order: int          # 0-based position in the plan
    budget_tokens: int  # tokens budgeted to this anchor

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "order": self.order,
            "budget_tokens": self.budget_tokens,
        }


@dataclass
class AnchorBudget:
    """Mutable per-anchor token-spend tracker.

    Wraps a :class:`GoalAnchor` with running ``tokens_spent``,
    ``status``, and ``warnings``. Consumers ``update(tokens)``-then-
    inspect ``is_exhausted`` and ``utilisation`` to decide whether to
    advance to the next anchor or surface a warning.
    """

    anchor: GoalAnchor
    tokens_spent: int = 0
    completed: bool = False
    warning_emitted_at: Optional[float] = None  # utilisation level when warned

    def update(self, tokens: int) -> None:
        """Add ``tokens`` to the spend (clamped at >=0)."""
        if tokens <= 0:
            return
        self.tokens_spent += int(tokens)

    def mark_completed(self) -> None:
        self.completed = True

    @property
    def utilisation(self) -> float:
        if self.anchor.budget_tokens <= 0:
            return 1.0 if self.tokens_spent > 0 else 0.0
        return self.tokens_spent / float(self.anchor.budget_tokens)

    @property
    def is_exhausted(self) -> bool:
        return self.tokens_spent >= self.anchor.budget_tokens

    def should_warn(self, *, threshold: float = 0.8) -> bool:
        """Return True iff we just crossed ``threshold`` for the first time.

        Latches the warning -- subsequent calls return False until
        :meth:`reset_warning` clears the latch.
        """
        if self.warning_emitted_at is not None:
            return False
        if self.utilisation >= threshold:
            self.warning_emitted_at = self.utilisation
            return True
        return False

    def reset_warning(self) -> None:
        self.warning_emitted_at = None


@dataclass
class AnchorPlan:
    """An ordered list of :class:`GoalAnchor` budgets + the active index."""

    anchors: List[AnchorBudget] = field(default_factory=list)
    active_index: int = 0

    def __len__(self) -> int:
        return len(self.anchors)

    def __iter__(self):
        return iter(self.anchors)

    @property
    def active(self) -> Optional[AnchorBudget]:
        if not self.anchors:
            return None
        if 0 <= self.active_index < len(self.anchors):
            return self.anchors[self.active_index]
        return None

    @property
    def all_completed(self) -> bool:
        return bool(self.anchors) and all(a.completed for a in self.anchors)

    def advance(self) -> Optional[AnchorBudget]:
        """Mark the active anchor completed and advance.

        Returns the new active anchor (or ``None`` when the plan is
        finished).
        """
        if not self.anchors:
            return None
        cur = self.active
        if cur is not None:
            cur.mark_completed()
        self.active_index += 1
        return self.active

    def remaining_tokens(self) -> int:
        """Sum of budget remaining across not-yet-completed anchors."""
        out = 0
        for b in self.anchors:
            if b.completed:
                continue
            out += max(0, b.anchor.budget_tokens - b.tokens_spent)
        return out

    def as_dict(self) -> dict:
        return {
            "active_index": self.active_index,
            "anchors": [
                {
                    **a.anchor.as_dict(),
                    "tokens_spent": a.tokens_spent,
                    "completed": a.completed,
                    "utilisation": round(a.utilisation, 3),
                }
                for a in self.anchors
            ],
        }


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def _slugify(text: str, *, max_chars: int = 40) -> str:
    """Return a lowercase underscore slug suitable for an anchor name."""
    if not text:
        return "anchor"
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower()).strip("_")
    if not cleaned:
        return "anchor"
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip("_")
    return cleaned or "anchor"


def _starts_with_anchor_verb(clause: str) -> bool:
    lowered = clause.strip().lower()
    for verb in _ANCHOR_VERBS:
        if lowered.startswith(verb + " ") or lowered == verb:
            return True
    return False


def _split_on_connectives(prompt: str) -> List[str]:
    """Split ``prompt`` on connective markers; preserves clause order."""
    chunks = _CONNECTIVES.split(prompt or "")
    return [c.strip() for c in chunks if c and c.strip()]


def _split_on_anchor_verbs(clause: str) -> List[str]:
    """Within one clause, split further on imperative-verb sentence stems.

    Looks for ".  Build X..." / "; Implement Y..." style boundaries.
    """
    # Build the pattern dynamically from _ANCHOR_VERBS so adding a new
    # verb doesn't require touching this function. Comma is included
    # in the boundary set so "Build X, test Y" splits even without a
    # connective ("then" / "finally" / ...).
    verb_alt = "|".join(re.escape(v) for v in _ANCHOR_VERBS)
    pattern = re.compile(
        rf"(?:^|[.;,!?\n])\s*(?=(?:{verb_alt})\b)",
        re.IGNORECASE,
    )
    parts = pattern.split(clause)
    return [p.strip(" .;,") for p in parts if p and p.strip(" .;,")]


def decompose_into_anchors(
    prompt: str,
    *,
    total_budget_tokens: int = 100_000,
    min_anchors: int = 1,
    max_anchors: int = 6,
) -> AnchorPlan:
    """Decompose ``prompt`` into a :class:`AnchorPlan`.

    Heuristic (no LLM call):

    1. Split on connective markers ("then", "next", "finally", ...).
    2. Within each chunk, split further on imperative-verb stems
       (start-of-clause "build" / "implement" / "test" / ...).
    3. Drop chunks that don't begin with an anchor-verb (they're
       narrative, not actionable).
    4. Cap at ``max_anchors``; pad to ``min_anchors`` with a single
       "complete the task" anchor when no actionable chunks survive.
    5. Distribute ``total_budget_tokens`` evenly across the resulting
       anchors (any remainder goes to the first anchor).

    Returns a freshly-built :class:`AnchorPlan` ready for the consumer
    to ``update(tokens)`` against the active anchor.
    """
    text = (prompt or "").strip()

    actionable: List[str] = []
    if text:
        for chunk in _split_on_connectives(text):
            for piece in _split_on_anchor_verbs(chunk):
                if _starts_with_anchor_verb(piece):
                    actionable.append(piece)
        # Final dedupe preserving order.
        seen: set[str] = set()
        deduped: List[str] = []
        for piece in actionable:
            key = piece.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(piece)
        actionable = deduped

    if not actionable:
        actionable = ["complete the task"]
    if len(actionable) > max_anchors:
        actionable = actionable[:max_anchors]
    while len(actionable) < min_anchors:
        actionable.append("complete the task")

    n = len(actionable)
    per_anchor = max(1, int(total_budget_tokens // n))
    remainder = max(0, int(total_budget_tokens) - per_anchor * n)

    anchors: List[AnchorBudget] = []
    for i, description in enumerate(actionable):
        budget = per_anchor + (remainder if i == 0 else 0)
        anchor = GoalAnchor(
            name=_slugify(description),
            description=description,
            order=i,
            budget_tokens=budget,
        )
        anchors.append(AnchorBudget(anchor=anchor))

    return AnchorPlan(anchors=anchors, active_index=0)


# ---------------------------------------------------------------------------
# Helpers for future runner integration
# ---------------------------------------------------------------------------


def narration_for_anchor(anchor: GoalAnchor, *, verb: str = "Starting") -> str:
    """Compose a TTS-safe in-character voice line announcing an anchor.

    Used by the eventual runner integration to narrate per-anchor
    progress without ever speaking a raw file path.
    """
    description = anchor.description.strip().rstrip(".")
    if not description:
        return f"{verb} the next anchor."
    return f"{verb} anchor {anchor.order + 1}: {description}."


def narration_for_completion(plan: AnchorPlan) -> str:
    """Voice line for plan completion."""
    if not plan.anchors:
        return "There was nothing to do."
    return f"All {len(plan.anchors)} anchors completed."
