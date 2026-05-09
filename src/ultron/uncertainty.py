"""Phase 5: translate preflight uncertainty signals into response behavior.

The preflight pass (in :mod:`ultron.web_search.gating`) already produces
three signals on every gate verdict:

  * ``knowledge_confidence``: high / medium / low / None
  * ``knowledge_source``: weights / retrieved_memory / retrieved_facts /
    web_search_needed / unknown / None
  * ``has_temporal_dependency``: bool / None

This module turns those into two outputs the orchestrator can act on:

  1. **Search upgrade**: a NO_SEARCH verdict with low confidence on a
     temporally-dependent query is upgraded to SEARCH, since the LLM
     would otherwise guess at fresh facts.
  2. **Per-call user-message addendum**: a one-line hint prepended to
     the user's text that primes the LLM to match its answer style to
     the actual confidence level. The permanent system prompt already
     instructs Ultron to handle uncertainty correctly; the per-call
     addendum just nudges it on this specific query.

Hard-rule verdicts (no preflight) carry no uncertainty signals -- this
module is a no-op for them.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

from ultron.utils.logging import get_logger
from ultron.web_search.gating import GateDecision, GateVerdict

logger = get_logger("uncertainty")


# Tuned to be terse and instructional, not chatty -- Piper would otherwise
# read these out if they leaked into the response. They never do today
# (the LLM treats them as system instruction, not text to repeat) but
# we keep them short anyway.
_ADDENDUM_MEDIUM = (
    "[Confidence: medium. Qualify briefly if you're not fully sure.]"
)
_ADDENDUM_LOW_NON_TEMPORAL = (
    "[Confidence: low. If you don't know for certain, say so plainly. "
    "Do not guess as if certain.]"
)
_ADDENDUM_LOW_TEMPORAL = (
    "[Confidence: low and the answer may have changed. Acknowledge that "
    "you may not be current; do not fabricate fresh facts.]"
)

# B1: knowledge_source-aware source hints. Prepended ABOVE the confidence
# addendum so the LLM has both signals when it composes the response.
# The web-search-needed branch isn't here -- the search path adds its
# own ack and sources, and we don't want a confidence addendum on a
# verdict that's about to be replaced by fetched citations.
_SOURCE_HINT_RETRIEVED_MEMORY = (
    "[Source: prior conversation memory. Speak naturally; do not cite.]"
)
_SOURCE_HINT_RETRIEVED_FACTS = (
    "[Source: stored user preference. Speak naturally; do not cite.]"
)


def apply(verdict: GateVerdict, user_text: str) -> Tuple[GateVerdict, str]:
    """Apply Phase 5 transforms.

    Returns ``(possibly_upgraded_verdict, possibly_augmented_user_text)``.

    Behavior:
      * Verdict from a hard-rule firing (``source == "rule"``) is left
        alone -- rules don't carry confidence signals worth acting on,
        EXCEPT a rule-derived ``knowledge_source`` that points at
        retrieved memory/facts still gets its source hint so the LLM
        can match its tone.
      * Verdict with ``knowledge_confidence == "low"`` AND a temporal
        dependency AND a current NO_SEARCH decision is upgraded to
        SEARCH. The original user text becomes the search query.
      * The user text gets a leading source hint (B1) based on
        ``knowledge_source`` when it points at retrieved memory/facts,
        and a ``[Confidence: ...]`` addendum based on
        ``knowledge_confidence`` when present.
    """
    # B1: rule verdicts skip the confidence path but their source hint
    # (if memory/facts) still primes the LLM tone.
    if verdict.source == "rule":
        source_hint = _source_hint_for(verdict)
        if source_hint:
            return verdict, f"{source_hint}\n\n{user_text}"
        return verdict, user_text

    upgraded = verdict
    confidence = verdict.knowledge_confidence
    temporal = bool(verdict.has_temporal_dependency)

    # Upgrade rule: low confidence + temporal => search proactively.
    if (
        verdict.decision == GateDecision.NO_SEARCH
        and confidence == "low"
        and temporal
    ):
        upgraded = replace(
            verdict,
            decision=GateDecision.SEARCH,
            confidence="medium",
            source="phase5_low_temporal_upgrade",
            reason=(
                "low knowledge confidence on a temporal claim; "
                "searching rather than guessing"
            ),
            search_queries=verdict.search_queries or [user_text.strip()],
        )
        logger.info(
            "phase5: upgrading NO_SEARCH -> SEARCH (low confidence + temporal): %r",
            user_text[:60],
        )

    # Addendum based on the FINAL confidence + temporal signals.
    final_confidence = upgraded.knowledge_confidence
    final_temporal = bool(upgraded.has_temporal_dependency)
    confidence_addendum: str = ""
    if final_confidence == "medium":
        confidence_addendum = _ADDENDUM_MEDIUM
    elif final_confidence == "low":
        confidence_addendum = (
            _ADDENDUM_LOW_TEMPORAL if final_temporal else _ADDENDUM_LOW_NON_TEMPORAL
        )

    # B1: source hint from knowledge_source. Skipped on the search path
    # (sources will be cited inline) so we don't conflict with the
    # search-result formatter.
    source_hint = ""
    if upgraded.decision != GateDecision.SEARCH:
        source_hint = _source_hint_for(upgraded)

    parts = [p for p in (source_hint, confidence_addendum) if p]
    if not parts:
        return upgraded, user_text

    augmented = "\n".join(parts) + "\n\n" + user_text
    return upgraded, augmented


def _source_hint_for(verdict: GateVerdict) -> str:
    """B1: pick a leading source-hint based on ``knowledge_source``.

    Only fires on retrieved-memory / retrieved-facts. ``weights`` /
    ``unknown`` / ``web_search_needed`` get no hint -- the first two
    are the model's default mode, and the third has the search-result
    formatter handling source attribution.
    """
    src = (verdict.knowledge_source or "").lower()
    if src == "retrieved_memory":
        return _SOURCE_HINT_RETRIEVED_MEMORY
    if src == "retrieved_facts":
        return _SOURCE_HINT_RETRIEVED_FACTS
    return ""
