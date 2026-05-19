"""Adaptive context-window scoring (cheap pre-flight heuristic).

Today the conversational LLM call uses a fixed
``memory.history_turns_for_llm`` history cap (default 4) and a fixed
``memory.rag_top_k`` retrieval size (default 5). That's wasteful for
short asks ("what time is it") and starves long technical asks that
need more conversation context.

This module returns a per-utterance recommendation in the form of a
:class:`ContextRecommendation` -- caller decides whether to honor it.
Pure-function, no IO, no config dependence (caller passes the static
defaults so they remain the per-deployment ceiling).

Default-OFF: callers wire this in only when the
``llm.adaptive_context.enabled`` flag is True. The helper itself is
always safe to call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Length thresholds (chars after strip).
_SHORT_CHAR_THRESHOLD = 30
_LONG_CHAR_THRESHOLD = 200

# Word-count thresholds.
_SHORT_WORD_THRESHOLD = 5
_LONG_WORD_THRESHOLD = 35

# Markers that indicate the user expects depth / elaboration. When
# present we bias toward keeping the full history + retrieval budget
# even on short utterances.
_DEPTH_MARKERS = re.compile(
    r"\b(?:explain|elaborate|in\s+detail|step\s+by\s+step|walk\s+me\s+through|"
    r"give\s+me\s+(?:an?\s+)?(?:overview|summary|background|breakdown)|"
    r"compare|contrast|why\s+does|how\s+does\s+\w+\s+work)\b",
    re.IGNORECASE,
)

# Reference-laden tokens: pronouns + "the one we discussed" style.
# When present we want MORE history (the resolution target is in past
# turns).
_REFERENCE_LADEN = re.compile(
    r"\b(?:it|that|this|those|these|him|her|them|the\s+one|"
    r"the\s+previous|the\s+last|earlier|before|just\s+now|"
    r"the\s+thing\s+(?:we|i)\s+(?:discussed|mentioned|talked\s+about))\b",
    re.IGNORECASE,
)

# Topic-shift markers: when present we want LESS history (the user
# wants to leave the prior topic behind).
_TOPIC_SHIFT = re.compile(
    r"\b(?:by\s+the\s+way|different\s+question|actually\s+let'?s|"
    r"moving\s+on|new\s+topic|new\s+question|forget\s+that|"
    r"on\s+a\s+different\s+note|switching\s+gears|change\s+(?:of\s+)?subject)\b",
    re.IGNORECASE,
)

# Personal-memory markers: definitely want history; skip RAG only when
# all other signals are also short.
_PERSONAL_RECALL = re.compile(
    r"\b(?:remember\s+when|what\s+did\s+(?:i|we)\s+|"
    r"do\s+you\s+(?:remember|recall)|"
    r"earlier\s+i\s+said|"
    r"what\s+were\s+we\s+(?:talking|discussing)\s+about)\b",
    re.IGNORECASE,
)

# Pure factual stems -- "what time is it", "who wrote X" -- benefit
# little from history.
_FACTUAL_STEM = re.compile(
    r"^\s*(?:"
    # "what's the time", "what is the date"
    r"what(?:'s|\s+is)\s+(?:the\s+)?(?:time|date|day|weather)|"
    # "what time is it" / "what day is today" (noun-first word order)
    r"what\s+(?:time|date|day|weather|year|month|hour)\s+(?:is|was)\s+(?:it|today|now)|"
    # "who is/wrote/invented X"
    r"who\s+(?:is|wrote|invented|founded|discovered|created)|"
    # "when is", "where is"
    r"when\s+(?:is|was)|where\s+is|"
    # "how tall / how long" measurement stems
    r"how\s+(?:tall|long|much|many|big|small|fast|old|far))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ContextRecommendation:
    """Caller's recommended context budget for this utterance.

    Fields
    ------
    history_turns:
        How many prior user/assistant turns to include in the prompt.
        Caller clamps to its own ``[0, history_turns_max]`` window.
    retrieval_k:
        Top-K to ask the retrieval store for. ``0`` disables retrieval
        entirely for this utterance.
    suppress_rag:
        Convenience equivalent of ``retrieval_k == 0`` -- callers that
        only branch on a bool can read this.
    reason:
        Short human-readable explanation of the recommendation, for
        the audit log + eval harness.
    """

    history_turns: int
    retrieval_k: int
    suppress_rag: bool
    reason: str

    @classmethod
    def fixed(
        cls,
        history_turns: int,
        retrieval_k: int,
        reason: str = "fixed default",
    ) -> "ContextRecommendation":
        return cls(
            history_turns=int(history_turns),
            retrieval_k=int(retrieval_k),
            suppress_rag=int(retrieval_k) == 0,
            reason=reason,
        )


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def score_context(
    user_text: str,
    *,
    default_history_turns: int = 4,
    default_retrieval_k: int = 5,
    has_active_task: bool = False,
    min_history_turns: int = 0,
    max_history_turns: int = 8,
    min_retrieval_k: int = 0,
    max_retrieval_k: int = 10,
) -> ContextRecommendation:
    """Return a :class:`ContextRecommendation` for ``user_text``.

    The heuristic blends six cheap signals (length, word-count,
    factual-stem, depth-marker, reference-laden, topic-shift, personal
    recall) into an adjustment delta applied to the caller-supplied
    defaults. All adjustments are clamped to the
    ``[min_*, max_*]`` ceilings the caller controls.

    Sub-millisecond on benign input; safe to call on every turn.
    """
    text = (user_text or "").strip()
    if not text:
        return ContextRecommendation.fixed(
            history_turns=default_history_turns,
            retrieval_k=default_retrieval_k,
            reason="empty utterance",
        )

    chars = len(text)
    words = _word_count(text)
    has_depth = bool(_DEPTH_MARKERS.search(text))
    has_reference = bool(_REFERENCE_LADEN.search(text))
    has_topic_shift = bool(_TOPIC_SHIFT.search(text))
    has_personal = bool(_PERSONAL_RECALL.search(text))
    is_factual_stem = bool(_FACTUAL_STEM.search(text))

    # Factual stems often carry a syntactic "it" / "that" / "this" that
    # the reference-laden regex would otherwise grab ("what time is it",
    # "who wrote that book"). Suppress the reference signal in those
    # cases so the factual-stem path wins.
    if is_factual_stem:
        has_reference = False

    reasons: list[str] = []

    history = default_history_turns
    retrieval = default_retrieval_k

    if has_topic_shift:
        history = min_history_turns
        retrieval = max(min_retrieval_k, default_retrieval_k - 2)
        reasons.append("topic-shift marker")
        return ContextRecommendation(
            history_turns=_clamp(history, min_history_turns, max_history_turns),
            retrieval_k=_clamp(retrieval, min_retrieval_k, max_retrieval_k),
            suppress_rag=(retrieval == 0),
            reason="; ".join(reasons) or "default",
        )

    # Length-based base adjustment.
    short = (chars <= _SHORT_CHAR_THRESHOLD) or (words <= _SHORT_WORD_THRESHOLD)
    long_ = (chars >= _LONG_CHAR_THRESHOLD) or (words >= _LONG_WORD_THRESHOLD)
    if short and not (has_depth or has_personal or has_reference):
        history = max(min_history_turns, 1)
        retrieval = max(min_retrieval_k, 2)
        reasons.append("short utterance")
    elif long_:
        history = min(max_history_turns, default_history_turns + 2)
        retrieval = min(max_retrieval_k, default_retrieval_k + 2)
        reasons.append("long utterance")

    if is_factual_stem and not (has_personal or has_reference):
        retrieval = min_retrieval_k
        reasons.append("factual stem (suppress retrieval)")

    if has_personal:
        history = min(max_history_turns, max(history, default_history_turns + 2))
        reasons.append("personal recall (boost history)")

    if has_reference:
        history = min(max_history_turns, max(history, default_history_turns + 2))
        reasons.append("reference-laden (boost history)")

    if has_depth:
        history = min(max_history_turns, max(history, default_history_turns + 1))
        retrieval = min(max_retrieval_k, max(retrieval, default_retrieval_k + 1))
        reasons.append("depth marker (raise both)")

    if has_active_task:
        history = min(max_history_turns, max(history, default_history_turns + 1))
        reasons.append("active coding task (preserve task context)")

    history = _clamp(history, min_history_turns, max_history_turns)
    retrieval = _clamp(retrieval, min_retrieval_k, max_retrieval_k)

    return ContextRecommendation(
        history_turns=history,
        retrieval_k=retrieval,
        suppress_rag=(retrieval == 0),
        reason="; ".join(reasons) or "default",
    )
