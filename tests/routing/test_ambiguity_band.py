"""Tests for :mod:`ultron.openclaw_routing.ambiguity`."""

from __future__ import annotations

import pytest

from ultron.openclaw_routing.ambiguity import (
    AmbiguityVerdict,
    should_clarify,
    should_clarify_from_config,
)
from ultron.openclaw_routing.intents import RoutingIntent, RoutingIntentKind


def _intent(
    kind: RoutingIntentKind = RoutingIntentKind.CODE_TASK,
    *,
    confidence: float = 0.5,
    needs_user_clarification: bool = False,
) -> RoutingIntent:
    return RoutingIntent(
        kind=kind,
        raw_text="hello",
        confidence=confidence,
        source="rule",
        reason="test fixture",
        needs_user_clarification=needs_user_clarification,
    )


# ---------------------------------------------------------------------------
# Master switch
# ---------------------------------------------------------------------------


def test_disabled_returns_should_clarify_false() -> None:
    verdict = should_clarify(_intent(confidence=0.5), enabled=False)
    assert verdict.should_clarify is False
    assert "disabled" in verdict.reason


# ---------------------------------------------------------------------------
# Eligibility filter
# ---------------------------------------------------------------------------


def test_conversational_kind_never_clarified() -> None:
    intent = _intent(kind=RoutingIntentKind.CONVERSATIONAL, confidence=0.5)
    verdict = should_clarify(intent, enabled=True)
    assert verdict.should_clarify is False
    assert "not in ambiguity-relevant set" in verdict.reason


def test_cancel_kind_never_clarified() -> None:
    intent = _intent(kind=RoutingIntentKind.CANCEL, confidence=0.5)
    verdict = should_clarify(intent, enabled=True)
    assert verdict.should_clarify is False


@pytest.mark.parametrize(
    "kind",
    [
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
    ],
)
def test_relevant_kinds_can_clarify_in_band(kind: RoutingIntentKind) -> None:
    intent = _intent(kind=kind, confidence=0.5)
    verdict = should_clarify(intent, enabled=True)
    assert verdict.should_clarify is True


# ---------------------------------------------------------------------------
# Already-clarifying short-circuit
# ---------------------------------------------------------------------------


def test_already_flagged_clarification_short_circuits() -> None:
    intent = _intent(confidence=0.5, needs_user_clarification=True)
    verdict = should_clarify(intent, enabled=True)
    assert verdict.should_clarify is False
    assert "already requested clarification" in verdict.reason


# ---------------------------------------------------------------------------
# Band boundary semantics
# ---------------------------------------------------------------------------


def test_confidence_at_band_low_clarifies() -> None:
    intent = _intent(confidence=0.4)
    verdict = should_clarify(intent, enabled=True, band_low=0.4, band_high=0.65)
    assert verdict.should_clarify is True


def test_confidence_just_below_band_low_does_not_clarify() -> None:
    intent = _intent(confidence=0.39)
    verdict = should_clarify(intent, enabled=True, band_low=0.4, band_high=0.65)
    assert verdict.should_clarify is False
    assert "below band" in verdict.reason


def test_confidence_at_band_high_does_not_clarify() -> None:
    # Half-open interval: band_high is exclusive.
    intent = _intent(confidence=0.65)
    verdict = should_clarify(intent, enabled=True, band_low=0.4, band_high=0.65)
    assert verdict.should_clarify is False
    assert "above band" in verdict.reason


def test_confidence_above_band_high_does_not_clarify() -> None:
    intent = _intent(confidence=0.9)
    verdict = should_clarify(intent, enabled=True)
    assert verdict.should_clarify is False


# ---------------------------------------------------------------------------
# Verdict shape
# ---------------------------------------------------------------------------


def test_verdict_carries_diagnostics() -> None:
    intent = _intent(confidence=0.5)
    verdict = should_clarify(intent, enabled=True, band_low=0.4, band_high=0.65)
    payload = verdict.as_dict()
    assert payload["should_clarify"] is True
    assert payload["confidence"] == 0.5
    assert payload["band_low"] == 0.4
    assert payload["band_high"] == 0.65
    assert "[0.40, 0.65)" in payload["reason"]


def test_verdict_is_frozen() -> None:
    verdict = should_clarify(_intent(), enabled=True)
    with pytest.raises(Exception):  # FrozenInstanceError
        verdict.should_clarify = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Custom eligible set override
# ---------------------------------------------------------------------------


def test_custom_eligible_set_restricts_clarification() -> None:
    intent = _intent(kind=RoutingIntentKind.CODE_TASK, confidence=0.5)
    verdict = should_clarify(
        intent,
        enabled=True,
        eligible_kinds=frozenset({RoutingIntentKind.BROWSER_AUTOMATION}),
    )
    assert verdict.should_clarify is False
    assert "not in ambiguity-relevant set" in verdict.reason


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


def test_should_clarify_from_config_default_is_disabled() -> None:
    """Default config keeps the feature OFF; the predicate should
    surface that without raising."""
    intent = _intent(confidence=0.5)
    verdict = should_clarify_from_config(intent)
    assert verdict.should_clarify is False


def test_should_clarify_from_config_swallows_config_errors(monkeypatch) -> None:
    """If config.get_config raises, the helper must fail-open."""
    def _boom():
        raise RuntimeError("config gone")
    monkeypatch.setattr("ultron.config.get_config", _boom)
    intent = _intent(confidence=0.5)
    verdict = should_clarify_from_config(intent)
    assert verdict.should_clarify is False
    assert "config read failed" in verdict.reason
