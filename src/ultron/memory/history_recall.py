"""Match a "what did I say earlier?" conversation-recall request.

A strict matcher that fires only on explicit verbatim-recall questions about
the CURRENT conversation -- "what did I say earlier about the database?",
"what did you tell me about X?", "remind me what I asked". On a hit the
orchestrator searches the in-memory :class:`~ultron.memory.dual_history.
DualHistoryStore` for the matching verbatim turn(s) and speaks them.

This is distinct from:

* ``deep_recall`` -- iterative RAG over long-term Qdrant memory ("recall
  everything we discussed about X"); and
* web search -- external knowledge.

It needs no LLM and no Qdrant: the dual-history store is in-memory + always
available, so verbatim recall works even when ConversationMemory is disabled.
Normal questions ("what should I do", "what did you mean") never trip it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

#: Search side for the recall: which speaker's turns to look through.
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

#: Verbs that signal a recall of something *said* in conversation. A plain
#: "what did I do" / "what did you mean" carries none of these, so it falls
#: through to normal routing.
_VERB = (
    r"say|said|saying|ask|asked|asking|tell|told|"
    r"mention|mentioned|bring\s+up|brought\s+up|talk\s+about|talked\s+about"
)

#: "what did I/we/you <verb> ...", "remind me what I <verb> ...",
#: "did I/you <verb> ...". The subject + recall-verb pairing is the gate.
_RECALL_RE = re.compile(
    r"\b(?:what\s+did|remind\s+me\s+(?:what|of\s+what)|did)\s+"
    r"(?P<subject>i|we|you)\s+"
    r"(?:just\s+|earlier\s+|previously\s+|recently\s+|already\s+)?"
    r"(?P<verb>" + _VERB + r")\b",
    re.IGNORECASE,
)

#: Strip a leading recency adverb + object pronoun + preposition from the
#: text following the verb, leaving the bare topic ("me earlier about the X"
#: -> "X").
_LEAD_STRIP_RE = re.compile(
    r"^\s*(?:just|earlier|previously|recently|already|a\s+moment\s+ago|before)?\s*"
    r"(?:to\s+)?(?:me|you|us)?\s*"
    r"(?:about|regarding|concerning|on|re|with\s+respect\s+to)?\s*",
    re.IGNORECASE,
)

_ARTICLE_RE = re.compile(
    r"^(?:the|a|an|any|anything|something)\s+", re.IGNORECASE,
)


@dataclass(frozen=True)
class HistoryRecallMatch:
    """A matched conversation-recall request.

    Attributes:
        role: which side to search -- :data:`ROLE_USER` ("what did I say")
            or :data:`ROLE_ASSISTANT` ("what did you say").
        topic: extracted topic substring; ``""`` means "no specific topic"
            (the handler returns the most-recent turn for that role).
        raw: the original utterance (stripped).
    """

    role: str
    topic: str
    raw: str


def _extract_topic(tail: str) -> str:
    """Reduce the text after the recall verb to a bare topic phrase."""
    tail = _LEAD_STRIP_RE.sub("", tail, count=1).strip()
    tail = tail.rstrip("?.! ").strip()
    tail = _ARTICLE_RE.sub("", tail).strip()
    return tail


def match_history_recall(text: str) -> Optional[HistoryRecallMatch]:
    """Return a :class:`HistoryRecallMatch` for an explicit conversation-recall
    request, or ``None`` when the utterance isn't one.

    Args:
        text: the user utterance.
    """
    if not text or not text.strip():
        return None
    m = _RECALL_RE.search(text)
    if m is None:
        return None
    subject = m.group("subject").lower()
    role = ROLE_ASSISTANT if subject == "you" else ROLE_USER
    topic = _extract_topic(text[m.end():])
    return HistoryRecallMatch(role=role, topic=topic, raw=text.strip())


__all__ = [
    "HistoryRecallMatch",
    "ROLE_ASSISTANT",
    "ROLE_USER",
    "match_history_recall",
]
