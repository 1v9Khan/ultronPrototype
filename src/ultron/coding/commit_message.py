"""LLM-generated git commit messages with model cascade.

Pattern lifted in spirit (not in source) from aider's
``repo.get_commit_message`` (Apache 2.0; see ``THIRD_PARTY_NOTICES.md``).

After a batch of edits, this module turns a unified diff + the user's
prompt context into a short commit-message-shaped string. The caller
walks a cascade of LLM callables (primary → fallback → ...) and the
first one to succeed wins. If the diff exceeds an estimated input
budget for a given LLM, that LLM is skipped (the catalog calls this
out specifically — we don't want to truncate the diff and lose the
"what actually changed" signal in the prompt).

Public surface:

  * :class:`CommitMessageRequest` — frozen dataclass of inputs.
  * :class:`CommitMessageResult` — frozen dataclass of outputs.
  * :func:`generate_commit_message` — primary entry point.
  * :data:`DEFAULT_COMMIT_SYSTEM_PROMPT` — the catalog's system prompt
    with light ultron customisation (one-line, no body, imperative).
  * :func:`strip_outer_quotes` — LLM-output sanitiser; some models
    wrap the message in quotes that need stripping.

Failure modes (all return ``message=None``):
  * Empty diff -> ``error="empty diff"``.
  * Every LLM in the cascade raised or returned empty ->
    ``error="all LLMs failed"`` with the last exception preserved.
  * Diff exceeds every cascade entry's input budget ->
    ``error="diff too large for any LLM"``.

The cascade callables are responsible for their own timeouts, model
selection, etc. This module just orchestrates the try-each-in-order
pattern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence


logger = logging.getLogger("ultron.coding.commit_message")


# Catalog T15 default system prompt. Light customisation: explicit
# "no Claude/AI attribution" rule because ultron's public-repo hygiene
# forbids brand-name AI-dev trailers in commit messages (per the
# pre-push hook). Keep the message imperative, present-tense, single-
# line. Body is OFF by default; callers who want a body can pass a
# custom prompt.
DEFAULT_COMMIT_SYSTEM_PROMPT = (
    "You are a git-commit-message author. Read the diff and the "
    "user's context, then write ONE imperative, present-tense, "
    "single-line commit message summarising the change.\n"
    "Rules:\n"
    "- 50 to 80 characters total.\n"
    "- Imperative mood: 'add', 'fix', 'remove', NOT 'added' / 'fixes'.\n"
    "- No file paths. No quotes. No closing punctuation.\n"
    "- No author attribution, no AI assistant mentions, no Co-Authored-By trailers.\n"
    "- Respond with the message ONLY. No preamble, no explanation."
)


@dataclass(frozen=True)
class CommitMessageRequest:
    """Inputs to :func:`generate_commit_message`."""

    diff_text: str
    user_context: str = ""
    system_prompt: str = DEFAULT_COMMIT_SYSTEM_PROMPT
    # Maximum prompt characters per LLM. The cascade entries are
    # checked against this budget (one entry per LLM in case different
    # models have different context windows).
    max_prompt_chars_per_llm: Sequence[int] = field(default_factory=lambda: (32000,))


@dataclass(frozen=True)
class CommitMessageResult:
    """Outputs of :func:`generate_commit_message`."""

    message: Optional[str]
    cascade_index: int = -1  # which LLM in the cascade produced it
    chars_in_prompt: int = 0
    error: str = ""
    last_exception: Optional[str] = None


# Cascade entry: a callable mapping (prompt_text) -> generated_text.
LLMCallable = Callable[[str], str]


def generate_commit_message(
    request: CommitMessageRequest,
    llm_cascade: Sequence[LLMCallable],
    *,
    strip_quotes: bool = True,
) -> CommitMessageResult:
    """Generate a commit message via the LLM cascade.

    Args:
        request: Input bundle (diff, context, system prompt, budgets).
        llm_cascade: Ordered sequence of LLM callables. First one to
            succeed wins. Each callable takes the full rendered
            prompt and returns the model's output.
        strip_quotes: When True (default), pass the LLM output through
            :func:`strip_outer_quotes` so wrapped-in-quotes messages
            come out clean.

    Returns:
        A :class:`CommitMessageResult`. ``message`` is None on total
        failure; check ``error`` for the reason.
    """
    if not request.diff_text or not request.diff_text.strip():
        return CommitMessageResult(message=None, error="empty diff")
    if not llm_cascade:
        return CommitMessageResult(message=None, error="empty cascade")

    last_exception: Optional[BaseException] = None
    chars_used = 0
    budgets = list(request.max_prompt_chars_per_llm) or [32000]
    # Extend the budget list to match the cascade length (last value repeated).
    while len(budgets) < len(llm_cascade):
        budgets.append(budgets[-1])

    any_within_budget = False

    for idx, llm in enumerate(llm_cascade):
        prompt = _render_prompt(request)
        chars_used = len(prompt)
        if chars_used > budgets[idx]:
            logger.debug(
                "commit_message: cascade[%d] skipped — prompt %d chars > budget %d",
                idx, chars_used, budgets[idx],
            )
            continue
        any_within_budget = True
        try:
            raw = llm(prompt) or ""
        except Exception as exc:                              # noqa: BLE001
            last_exception = exc
            logger.debug("commit_message: cascade[%d] raised: %s", idx, exc)
            continue
        cleaned = strip_outer_quotes(raw.strip()) if strip_quotes else raw.strip()
        if not cleaned:
            logger.debug("commit_message: cascade[%d] returned empty", idx)
            continue
        return CommitMessageResult(
            message=cleaned,
            cascade_index=idx,
            chars_in_prompt=chars_used,
        )

    if not any_within_budget:
        return CommitMessageResult(
            message=None,
            chars_in_prompt=chars_used,
            error="diff too large for any LLM in the cascade",
        )
    return CommitMessageResult(
        message=None,
        chars_in_prompt=chars_used,
        error="all LLMs failed",
        last_exception=str(last_exception) if last_exception else None,
    )


def strip_outer_quotes(text: str) -> str:
    """Strip an outer pair of matching quotes if present.

    Catalog says: 'some models wrap the message in quotes that need
    stripping'. Handles single-quotes, double-quotes, fancy quotes,
    and triple-backticks (some models render messages as code blocks).
    """
    if not text:
        return text
    s = text.strip()
    # Triple-backtick block.
    if s.startswith("```") and s.endswith("```") and len(s) >= 6:
        s = s[3:-3].strip()
        # Often the next character is a language tag we want to drop.
        if "\n" in s and s.split("\n", 1)[0].isalnum():
            s = s.split("\n", 1)[1].strip()
    # Paired ASCII quotes.
    for opening, closing in [("'", "'"), ('"', '"')]:
        if (
            len(s) >= 2
            and s.startswith(opening)
            and s.endswith(closing)
            # Don't strip when the quote is unbalanced (e.g. text containing
            # an apostrophe inside another single-quoted shell-style block).
            and s.count(opening) == 2
            and not s[1:-1].endswith(opening)
        ):
            s = s[1:-1].strip()
            break
    # Paired smart quotes.
    for opening, closing in [("‘", "’"), ("“", "”")]:
        if len(s) >= 2 and s.startswith(opening) and s.endswith(closing):
            s = s[1:-1].strip()
            break
    return s


def _render_prompt(request: CommitMessageRequest) -> str:
    parts: List[str] = []
    parts.append(request.system_prompt)
    if request.user_context:
        parts.append(f"User context:\n{request.user_context}")
    parts.append(f"Diff:\n{request.diff_text}")
    parts.append("Commit message:")
    return "\n\n".join(parts)


__all__ = [
    "CommitMessageRequest",
    "CommitMessageResult",
    "DEFAULT_COMMIT_SYSTEM_PROMPT",
    "LLMCallable",
    "generate_commit_message",
    "strip_outer_quotes",
]
