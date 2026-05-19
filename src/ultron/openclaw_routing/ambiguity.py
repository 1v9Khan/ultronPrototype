"""Confidence-band ambiguity helper for routing.

Today the classifier returns a single :class:`RoutingIntent` and the
orchestrator executes it. When the confidence sits in a soft band --
clearly above "no match" but below "obviously correct" -- a small UX
improvement is to ask one clarifying question instead of acting on a
guess.

This module ships only the pure predicate + a config-driven helper.
Wiring into the orchestrator's voice loop is a deliberate follow-up:
default behaviour stays "execute the verdict" until the user opts in
via ``routing.ambiguity_band_clarification.enabled``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .intents import RoutingIntent, RoutingIntentKind


# Intent kinds where confidence-band ambiguity matters in practice.
# CONVERSATIONAL is the default fall-through so its confidence is rarely
# meaningful in the same sense; CANCEL / CLARIFICATION_RESPONSE are
# in-flight commands that shouldn't be second-guessed.
_AMBIGUITY_RELEVANT_KINDS: frozenset[RoutingIntentKind] = frozenset(
    {
        RoutingIntentKind.CODE_TASK,
        RoutingIntentKind.BROWSER_AUTOMATION,
        RoutingIntentKind.MEDIA_GENERATION,
        RoutingIntentKind.MESSAGING,
        RoutingIntentKind.FILE_OPERATION,
        RoutingIntentKind.SHELL_OPERATION,
        RoutingIntentKind.HYBRID_TASK,
        RoutingIntentKind.MODEL_SWITCH,
        RoutingIntentKind.APP_LAUNCH,
        RoutingIntentKind.SCREEN_CONTEXT_QUERY,
        RoutingIntentKind.WINDOW_MOVE,
        RoutingIntentKind.WINDOW_CLOSE,
    }
)


@dataclass(frozen=True)
class AmbiguityVerdict:
    """Result of a single ambiguity check."""

    should_clarify: bool
    band_low: float
    band_high: float
    intent_kind: str
    confidence: float
    reason: str

    def as_dict(self) -> dict:
        return {
            "should_clarify": self.should_clarify,
            "band_low": self.band_low,
            "band_high": self.band_high,
            "intent_kind": self.intent_kind,
            "confidence": self.confidence,
            "reason": self.reason,
        }


def should_clarify(
    intent: RoutingIntent,
    *,
    band_low: float = 0.4,
    band_high: float = 0.65,
    enabled: bool = True,
    eligible_kinds: Optional[frozenset[RoutingIntentKind]] = None,
) -> AmbiguityVerdict:
    """Return a verdict indicating whether ``intent`` is ambiguous enough
    to warrant a clarifying question instead of immediate execution.

    The check fires only when:

    * ``enabled`` is True (caller-side master switch).
    * The intent kind is in the eligible set (mid-grade automation +
      coding kinds; the in-flight commands and CONVERSATIONAL are
      excluded).
    * The intent's already-flagged ``needs_user_clarification`` field
      is False (otherwise the classifier already wants clarification
      and the orchestrator handles that path directly).
    * The intent's ``confidence`` is in ``[band_low, band_high)``.

    Otherwise the verdict's ``should_clarify`` is False.
    """
    eligible = eligible_kinds or _AMBIGUITY_RELEVANT_KINDS
    if not enabled:
        return AmbiguityVerdict(
            should_clarify=False,
            band_low=band_low,
            band_high=band_high,
            intent_kind=intent.kind.value,
            confidence=float(intent.confidence or 0.0),
            reason="ambiguity-band clarification disabled",
        )
    if intent.needs_user_clarification:
        return AmbiguityVerdict(
            should_clarify=False,
            band_low=band_low,
            band_high=band_high,
            intent_kind=intent.kind.value,
            confidence=float(intent.confidence or 0.0),
            reason="classifier already requested clarification",
        )
    if intent.kind not in eligible:
        return AmbiguityVerdict(
            should_clarify=False,
            band_low=band_low,
            band_high=band_high,
            intent_kind=intent.kind.value,
            confidence=float(intent.confidence or 0.0),
            reason=f"intent kind {intent.kind.value} not in ambiguity-relevant set",
        )
    conf = float(intent.confidence or 0.0)
    if band_low <= conf < band_high:
        return AmbiguityVerdict(
            should_clarify=True,
            band_low=band_low,
            band_high=band_high,
            intent_kind=intent.kind.value,
            confidence=conf,
            reason=f"confidence {conf:.2f} in ambiguity band [{band_low:.2f}, {band_high:.2f})",
        )
    return AmbiguityVerdict(
        should_clarify=False,
        band_low=band_low,
        band_high=band_high,
        intent_kind=intent.kind.value,
        confidence=conf,
        reason=(
            "below band"
            if conf < band_low
            else "above band (acted on confidently)"
        ),
    )


def should_clarify_from_config(intent: RoutingIntent) -> AmbiguityVerdict:
    """Convenience: read the live config + check ``intent``.

    Fail-open: any config read failure returns ``should_clarify=False``
    with a reason that surfaces the failure.
    """
    try:
        from ultron.config import get_config

        cfg = get_config().routing.ambiguity_band_clarification
        return should_clarify(
            intent,
            band_low=cfg.band_low,
            band_high=cfg.band_high,
            enabled=cfg.enabled,
        )
    except Exception as exc:
        return AmbiguityVerdict(
            should_clarify=False,
            band_low=0.0,
            band_high=0.0,
            intent_kind=intent.kind.value if intent and intent.kind else "",
            confidence=0.0,
            reason=f"config read failed: {type(exc).__name__}",
        )
