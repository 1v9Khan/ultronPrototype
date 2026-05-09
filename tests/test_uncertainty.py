"""Phase 5 verification tests.

Pure-function tests for ``ultron.uncertainty.apply``: takes a
``GateVerdict`` and the original user text, returns a possibly-upgraded
verdict and possibly-augmented text. No LLM in the loop.

A live integration test (slow tier) verifies the LLM actually picks up
the addendum and adapts its response style.
"""

from __future__ import annotations

import os

import pytest

from ultron.uncertainty import apply as apply_uncertainty
from ultron.web_search.gating import GateDecision, GateVerdict


def _vmake(**overrides) -> GateVerdict:
    base = dict(
        decision=GateDecision.NO_SEARCH,
        confidence="medium",
        source="preflight",
        reason="test",
        search_queries=[],
        knowledge_confidence=None,
        knowledge_source=None,
        has_temporal_dependency=None,
        latency_ms=10.0,
    )
    base.update(overrides)
    return GateVerdict(**base)


# ---------------------------------------------------------------------------
# Rule-source verdicts pass through unchanged.
# ---------------------------------------------------------------------------


def test_rule_verdict_is_left_alone_when_source_is_weights():
    """Rule-fired verdicts don't carry preflight signals; Phase 5 must
    not second-guess them. When ``knowledge_source`` indicates weights
    (or unknown), the user text passes through unchanged."""
    v = _vmake(source="rule", knowledge_confidence="low",
               has_temporal_dependency=True, knowledge_source="weights")
    out_v, out_text = apply_uncertainty(v, "tell me about black holes")
    assert out_v is v  # no copy made
    assert out_text == "tell me about black holes"


def test_rule_verdict_with_retrieved_memory_source_gets_hint():
    """B1: a rule verdict whose knowledge_source indicates retrieved
    memory still gets the source hint -- the LLM benefits from knowing
    the answer is leaning on prior conversation."""
    v = _vmake(source="rule", knowledge_confidence="high",
               knowledge_source="retrieved_memory",
               decision=GateDecision.NO_SEARCH,
               reason="personal/memory question")
    _, out_text = apply_uncertainty(v, "what did I say earlier?")
    assert "prior conversation memory" in out_text
    assert "what did I say earlier?" in out_text


def test_rule_verdict_with_retrieved_facts_source_gets_hint():
    v = _vmake(source="rule", knowledge_confidence="high",
               knowledge_source="retrieved_facts",
               decision=GateDecision.NO_SEARCH,
               reason="matched stored fact")
    _, out_text = apply_uncertainty(v, "what's my favorite framework?")
    assert "stored user preference" in out_text


# ---------------------------------------------------------------------------
# Confidence -> addendum mapping.
# ---------------------------------------------------------------------------


def test_high_confidence_adds_no_addendum():
    v = _vmake(knowledge_confidence="high")
    out_v, out_text = apply_uncertainty(v, "explain photosynthesis")
    assert out_text == "explain photosynthesis"
    assert out_v.decision == GateDecision.NO_SEARCH


def test_medium_confidence_prepends_hedging_hint():
    v = _vmake(knowledge_confidence="medium")
    out_v, out_text = apply_uncertainty(v, "explain photosynthesis")
    assert out_text.startswith("[Confidence: medium")
    assert "explain photosynthesis" in out_text
    # Decision unchanged.
    assert out_v.decision == GateDecision.NO_SEARCH


def test_low_confidence_non_temporal_adds_dont_guess_hint():
    v = _vmake(knowledge_confidence="low", has_temporal_dependency=False,
               decision=GateDecision.NO_SEARCH)
    out_v, out_text = apply_uncertainty(v, "what's my favorite color")
    assert "Confidence: low" in out_text
    assert "say so plainly" in out_text
    # Without temporal dependency, decision stays NO_SEARCH (fabricating
    # a search result for "what's my favorite color" doesn't help).
    assert out_v.decision == GateDecision.NO_SEARCH


# ---------------------------------------------------------------------------
# LOW + temporal -> upgrade NO_SEARCH to SEARCH.
# ---------------------------------------------------------------------------


def test_low_confidence_plus_temporal_upgrades_to_search():
    """The classic 'I'm guessing about a fact that may have changed' case --
    Phase 5 escalates this to a real search."""
    v = _vmake(
        decision=GateDecision.NO_SEARCH,
        knowledge_confidence="low",
        has_temporal_dependency=True,
        search_queries=[],
    )
    out_v, out_text = apply_uncertainty(v, "What's the latest version of Python?")

    assert out_v.decision == GateDecision.SEARCH
    assert out_v.source == "phase5_low_temporal_upgrade"
    assert out_v.search_queries == ["What's the latest version of Python?"]
    # The search-augmented prompt also gets the temporal addendum so the
    # LLM still knows to caveat if the search comes back empty.
    assert "Confidence: low" in out_text
    assert "may have changed" in out_text


def test_low_confidence_plus_temporal_preserves_existing_search_queries():
    v = _vmake(
        decision=GateDecision.NO_SEARCH,
        knowledge_confidence="low",
        has_temporal_dependency=True,
        search_queries=["specific custom query"],
    )
    out_v, _ = apply_uncertainty(v, "what's that thing again")
    assert out_v.search_queries == ["specific custom query"]


def test_low_confidence_plus_temporal_with_already_search_no_change():
    """If the gate already said SEARCH, the upgrade is a no-op (decision
    stays SEARCH) but the addendum still applies."""
    v = _vmake(
        decision=GateDecision.SEARCH,
        source="preflight",
        knowledge_confidence="low",
        has_temporal_dependency=True,
        search_queries=["query"],
    )
    out_v, out_text = apply_uncertainty(v, "test query")
    assert out_v.decision == GateDecision.SEARCH
    # Source unchanged -- the upgrade only fires when decision was
    # NO_SEARCH.
    assert out_v.source == "preflight"
    assert "Confidence: low" in out_text


# ---------------------------------------------------------------------------
# No knowledge_confidence => no addendum.
# ---------------------------------------------------------------------------


def test_no_confidence_signal_means_no_addendum():
    v = _vmake(knowledge_confidence=None, has_temporal_dependency=None)
    out_v, out_text = apply_uncertainty(v, "Hello.")
    assert out_text == "Hello."
    assert out_v is v


# ---------------------------------------------------------------------------
# B1: knowledge_source-aware source hints.
# ---------------------------------------------------------------------------


def test_preflight_with_retrieved_memory_source_prepends_hint():
    """When the preflight identifies retrieved-memory as the source, the
    user text gets a source hint."""
    v = _vmake(
        source="preflight",
        knowledge_confidence="medium",
        knowledge_source="retrieved_memory",
    )
    _, out_text = apply_uncertainty(v, "tell me about that thing we discussed")
    assert "prior conversation memory" in out_text
    # Source hint comes BEFORE confidence addendum.
    assert out_text.index("conversation memory") < out_text.index("Confidence")


def test_preflight_with_weights_source_no_hint():
    v = _vmake(
        source="preflight",
        knowledge_confidence="high",
        knowledge_source="weights",
    )
    _, out_text = apply_uncertainty(v, "explain photosynthesis")
    assert out_text == "explain photosynthesis"


def test_preflight_with_unknown_source_no_hint():
    v = _vmake(
        source="preflight",
        knowledge_confidence="low",
        knowledge_source="unknown",
        has_temporal_dependency=False,
    )
    _, out_text = apply_uncertainty(v, "obscure trivia question")
    # Confidence addendum still fires; no source hint.
    assert "prior conversation memory" not in out_text
    assert "stored user preference" not in out_text
    assert "Confidence: low" in out_text


def test_search_decision_skips_source_hint():
    """The search path adds its own attribution; no source hint here."""
    v = _vmake(
        source="preflight",
        decision=GateDecision.SEARCH,
        knowledge_confidence="medium",
        knowledge_source="web_search_needed",
    )
    _, out_text = apply_uncertainty(v, "what is happening today")
    assert "prior conversation memory" not in out_text
    assert "stored user preference" not in out_text


def test_source_hint_for_retrieved_facts():
    v = _vmake(
        source="preflight",
        knowledge_confidence="high",
        knowledge_source="retrieved_facts",
    )
    _, out_text = apply_uncertainty(v, "what should I default to?")
    assert "stored user preference" in out_text


# ---------------------------------------------------------------------------
# Slow-tier: live LLM picks up the addendum.
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("PYTEST_RUN_GPU_TESTS") != "1",
    reason="set PYTEST_RUN_GPU_TESTS=1 to load the main LLM",
)
def test_low_confidence_response_acknowledges_uncertainty():
    """With a low-confidence addendum on a question the model probably
    can't verify, the response should explicitly admit uncertainty
    rather than fabricate."""
    from ultron.llm import LLMEngine

    llm = LLMEngine(memory=None)
    v = _vmake(knowledge_confidence="low", has_temporal_dependency=False)
    _, augmented = apply_uncertainty(
        v, "What was the closing share price of NVIDIA on April 17, 2026?"
    )
    response = llm.generate(augmented).lower()
    assert any(
        signal in response
        for signal in (
            "i don't know",
            "i'm not certain",
            "not sure",
            "unable to",
            "do not have",
            "cannot confirm",
            "verify",
            "would need",
            "uncertain",
        )
    ), f"low-confidence response didn't acknowledge uncertainty: {response[:300]!r}"
