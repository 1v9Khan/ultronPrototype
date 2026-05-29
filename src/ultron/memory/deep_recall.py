"""Strict voice-intent matcher for explicit deep-memory recall.

Companion to :func:`ultron.web_search.deep_research.match_deep_research`,
but for the conversation-memory store instead of the web. It gates the
orchestrator's ``_maybe_handle_deep_recall`` short-circuit, which runs a
bounded :class:`ultron.agent_loop.deep_loops.DeepMemoryLoop` (iterative
RAG: decompose -> retrieve -> gap-fill -> retrieve, capped by ``max_steps``)
and synthesises an answer from the recalled turns.

The matcher is DELIBERATELY STRICT. Normal recall questions ("what do you
remember about my car?", "did I mention the dentist?") MUST stay on the
fast single-pass RAG path inside ``_respond`` -- the deep loop fires several
LLM + retrieve passes (a few seconds), so hijacking the fast path would be
a latency regression. We therefore require BOTH:

* an explicit *exhaustiveness / depth* marker ("in depth", "thoroughly",
  "everything we discussed", "dig deep", "exhaustively", ...), AND
* an explicit *memory / recall* referent ("remember", "recall", "your
  memory", "we discussed", "I told you", ...),

and we refuse anything that reads as a *web* research request (those belong
to :func:`match_deep_research`, which the orchestrator checks first).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

__all__ = ["DeepRecallMatch", "match_deep_recall"]


# Exhaustiveness / depth markers -- the signal that the user wants a
# thorough multi-pass recall rather than a one-line answer.
_DEPTH_RE = re.compile(
    r"\b("
    r"in[\s-]?depth|thoroughly|exhaustively|comprehensively|"
    r"dig\s+deep(?:er)?|deep[\s-]?dive|deep\s+recall|"
    r"everything\s+(?:you|we|i)|all\s+(?:of\s+)?(?:what|that)\s+(?:you|we|i)"
    r")\b",
    re.IGNORECASE,
)

# Explicit memory / recall referent -- the signal that the target is the
# conversation store, not the web or a general question.
_MEMORY_RE = re.compile(
    r"\b("
    r"remember|recall|your\s+memory|from\s+memory|"
    r"we\s+(?:discussed|talked\s+about|said|covered|went\s+over)|"
    r"you\s+know\s+about|i\s+(?:told|mentioned\s+to)\s+you|i\s+said"
    r")\b",
    re.IGNORECASE,
)

# If it reads as a web/online research request, defer to match_deep_research.
_WEB_RE = re.compile(
    r"\b("
    r"search\s+(?:the\s+)?(?:web|internet)|look\s+(?:it\s+)?up\s+online|"
    r"on\s+the\s+(?:web|internet)|latest\s+news|google"
    r")\b",
    re.IGNORECASE,
)

# Topic extraction: prefer the clause after an explicit recall pivot.
_TOPIC_RE = re.compile(
    r"\b(?:about|regarding|on\s+the\s+topic\s+of|concerning|"
    r"discussed|talked\s+about|said\s+about|remember\s+about|"
    r"recall\s+about|told\s+you\s+about|know\s+about)\s+(.+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DeepRecallMatch:
    """A matched deep-recall command.

    Attributes:
        topic: the recall subject (best-effort extracted; falls back to the
            whole utterance, which the loop's decomposer handles fine).
        raw_text: the original utterance.
    """

    topic: str
    raw_text: str


def _extract_topic(text: str) -> str:
    """Best-effort pull of the recall subject from the utterance."""
    m = _TOPIC_RE.search(text)
    candidate = m.group(1) if m else text
    candidate = candidate.strip().strip("?.!,").strip()
    # The regex may match an earlier pivot (e.g. "discussed") and leave a
    # trailing "about <topic>"; strip a leading pivot preposition so the
    # topic handed to the recall loop is clean.
    candidate = re.sub(
        r"^(?:about|regarding|concerning|on)\s+", "", candidate, flags=re.IGNORECASE,
    ).strip()
    return candidate


def match_deep_recall(text: str) -> Optional[DeepRecallMatch]:
    """Return a :class:`DeepRecallMatch` iff ``text`` is an explicit
    exhaustive-recall command, else ``None``.

    Strict: requires both a depth marker AND a memory referent, and is
    suppressed for web-research-shaped utterances. Empty / whitespace
    input returns ``None``.
    """
    if not text or not text.strip():
        return None
    if _WEB_RE.search(text):
        return None
    if not _DEPTH_RE.search(text):
        return None
    if not _MEMORY_RE.search(text):
        return None
    topic = _extract_topic(text)
    if not topic:
        topic = text.strip()
    return DeepRecallMatch(topic=topic, raw_text=text)
