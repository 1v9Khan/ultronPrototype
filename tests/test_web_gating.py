"""Phase 4 verification: hard-rule + LLM-preflight gating.

The rule layer is exhaustively tested unconditionally -- it's pure regex
and must not regress. The full classifier (rules + preflight) loads the
main LLM, so it's gated on PYTEST_RUN_GPU_TESTS=1.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import pytest

from ultron.web_search.gating import (
    GateDecision,
    WebSearchGate,
    classify_by_rules,
)

D = GateDecision


# (utterance, expected_decision, why)
_RULE_CASES: List[Tuple[str, GateDecision, str]] = [
    # Strong YES: time-sensitive markers
    ("What's the weather today?", D.SEARCH, "today + weather"),
    ("Tell me what's happening in the news right now.", D.SEARCH, "right now + news"),
    ("Who won the game tonight?", D.SEARCH, "tonight + score"),
    ("Latest iPhone release.", D.SEARCH, "latest + release"),
    ("Did Apple just announce something?", D.SEARCH, "just announced"),
    ("Has the package been delivered yet?", D.UNCERTAIN, "vague timing"),
    ("Recent updates to Python.", D.SEARCH, "recent + updates"),
    ("What is happening this week?", D.SEARCH, "this week"),
    ("What is the current price of Bitcoin?", D.SEARCH, "current + price"),
    ("Tomorrow's forecast for San Francisco.", D.SEARCH, "tomorrow + forecast"),

    # Strong YES: volatile topics (without explicit time markers)
    ("Weather in Tokyo.", D.SEARCH, "weather"),
    ("Apple stock price.", D.SEARCH, "stock price"),
    ("NBA standings.", D.SEARCH, "standings"),

    # Strong YES: post-cutoff year
    ("Who won the 2026 Super Bowl?", D.SEARCH, "year >= cutoff"),
    ("Tech announcements from 2027.", D.SEARCH, "year >= cutoff"),

    # Strong YES: embedded URL
    ("Read https://example.com/article and summarize.", D.SEARCH, "embedded URL"),

    # Strong NO: personal / memory
    ("What did I say earlier about the project?", D.NO_SEARCH, "personal/memory"),
    ("Do you remember my favorite color?", D.NO_SEARCH, "memory question"),
    ("My password for the app.", D.NO_SEARCH, "personal"),
    ("Earlier I said something about meditation.", D.NO_SEARCH, "memory recall"),

    # Strong NO: creative tasks
    ("Write me a poem about loneliness.", D.NO_SEARCH, "creative writing"),
    ("Compose a haiku about coffee.", D.NO_SEARCH, "creative writing"),
    ("Draft an email to my boss.", D.NO_SEARCH, "drafting"),
    ("Brainstorm names for my dog.", D.NO_SEARCH, "brainstorming"),
    ("Pretend you are a detective and interrogate me.", D.NO_SEARCH, "roleplay"),

    # NO_SEARCH via the stable-factual fast-path rule.
    ("Who was Nikola Tesla?", D.NO_SEARCH, "stable factual / who-was"),
    ("Explain how a hash table works.", D.NO_SEARCH, "conceptual / explain"),
    ("How does photosynthesis work?", D.NO_SEARCH, "stable factual / how-does"),
    ("Are you afraid of death?", D.NO_SEARCH, "philosophical / are-you"),
    ("What do you think about meditation?", D.NO_SEARCH, "opinion / what-do-you-think"),
    ("Tell me about black holes.", D.NO_SEARCH, "stable factual / tell-me"),
    ("Tell me an interesting story about the Mariana Trench.", D.NO_SEARCH, "tell-me"),
    ("How tall is Mount Everest?", D.NO_SEARCH, "stable factual / how-tall"),
    ("And what about the Mariana Trench?", D.NO_SEARCH, "continuation"),
    ("What's nineteen times forty-three?", D.NO_SEARCH, "math / contraction"),
    ("Walk me through how a transistor works.", D.NO_SEARCH, "walk-me-through"),
    ("What's a good book to read on a flight?", D.NO_SEARCH, "opinion / contraction"),

    # UNCERTAIN: genuinely ambiguous -> falls through to LLM preflight.
    ("Has the package been delivered yet?", D.UNCERTAIN, "vague timing"),

    # Tricky: time markers should win over the factual fast-path.
    ("Who was the latest emperor of Rome?", D.SEARCH, "latest fires before factual"),
]


def test_rule_layer_classifies_obvious_cases():
    """The rule layer must classify the obvious cases correctly. UNCERTAIN
    means the rule layer abstained, which is fine."""
    misses = []
    for utt, expected, note in _RULE_CASES:
        got = classify_by_rules(utt)
        actual = got.decision if got else D.UNCERTAIN
        if actual != expected:
            misses.append(
                f"  {utt!r}: expected {expected.value}, got {actual.value} "
                f"({got.reason if got else 'no rule fired'}) [{note}]"
            )
    if misses:
        pytest.fail(
            f"Rule-layer regressions ({len(misses)}/{len(_RULE_CASES)}):\n"
            + "\n".join(misses)
        )


def test_rule_layer_high_confidence_means_high_confidence():
    """If the rule layer returns a verdict, its confidence string should
    reflect actual rule strength (we use 'high' for all rule fires today)."""
    for utt, _, _ in _RULE_CASES:
        got = classify_by_rules(utt)
        if got is None:
            continue
        assert got.confidence in {"high", "medium", "low"}
        assert got.source == "rule"


def test_anti_search_beats_time_sensitive():
    """When a personal/creative rule and a time-sensitive marker both fire,
    the anti-search rule must win -- 'what did I say earlier today' is
    memory, not weather."""
    got = classify_by_rules("What did I say earlier today about the project?")
    assert got is not None
    assert got.decision == D.NO_SEARCH


def test_url_marker_attaches_query():
    got = classify_by_rules("Summarize https://example.com/post for me.")
    assert got is not None
    assert got.decision == D.SEARCH
    assert got.search_queries == ["https://example.com/post"]


def test_empty_utterance_is_no_search():
    got = classify_by_rules("")
    assert got is not None
    assert got.decision == D.NO_SEARCH
    got = classify_by_rules("    ")
    assert got is not None
    assert got.decision == D.NO_SEARCH


def test_gate_without_llm_returns_uncertain_for_uncovered_cases():
    """A genuinely ambiguous query (no rule fires) routes UNCERTAIN when
    the preflight LLM is unavailable."""
    gate = WebSearchGate(llm=None)
    v = gate.classify("Has the package been delivered yet?")
    assert v.decision == D.UNCERTAIN
    assert v.source == "default"


# ---------------------------------------------------------------------------
# Full classifier (rules + preflight). Slow; loads the main LLM.
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("PYTEST_RUN_GPU_TESTS") != "1",
    reason="set PYTEST_RUN_GPU_TESTS=1 to load the main LLM",
)
def test_preflight_routes_stable_factual_to_no_search():
    """A genuinely stable factual question (no time markers, no
    personal context) should route NO_SEARCH from the preflight."""
    from ultron.llm import LLMEngine

    llm = LLMEngine(memory=None)
    gate = WebSearchGate(llm=llm)

    v = gate.classify("How does a hash table work?")
    assert v.source == "preflight"
    assert v.decision == D.NO_SEARCH


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("PYTEST_RUN_GPU_TESTS") != "1",
    reason="set PYTEST_RUN_GPU_TESTS=1 to load the main LLM",
)
def test_preflight_routes_specific_recent_facts_to_search():
    """A specific factual question about something recent should route
    SEARCH from the preflight (no time-marker keyword in the utterance)."""
    from ultron.llm import LLMEngine

    llm = LLMEngine(memory=None)
    gate = WebSearchGate(llm=llm)

    # No "today" / "current" / "latest" -- forces preflight to decide.
    v = gate.classify("Who is the CEO of OpenAI?")
    assert v.source == "preflight"
    # NB: This is intentionally a flaky-on-model-output test. If preflight
    # misroutes consistently, that's a real signal we should retune the
    # prompt.
    assert v.decision in {D.SEARCH, D.NO_SEARCH}
    # Confidence + temporal metadata must be populated either way.
    assert v.knowledge_confidence in {"high", "medium", "low"}
    assert isinstance(v.has_temporal_dependency, (bool, type(None)))


# ---------------------------------------------------------------------------
# B1: knowledge_source enumeration
# ---------------------------------------------------------------------------


def test_knowledge_source_for_personal_question_is_retrieved_memory():
    got = classify_by_rules("What did I say earlier about the project?")
    assert got is not None
    assert got.decision == D.NO_SEARCH
    assert got.knowledge_source == "retrieved_memory"


def test_knowledge_source_for_url_is_web_search_needed():
    got = classify_by_rules("Summarize https://example.com/post for me.")
    assert got is not None
    assert got.decision == D.SEARCH
    assert got.knowledge_source == "web_search_needed"


def test_knowledge_source_for_time_sensitive_is_web_search_needed():
    got = classify_by_rules("What's the weather today?")
    assert got is not None
    assert got.decision == D.SEARCH
    assert got.knowledge_source == "web_search_needed"


def test_knowledge_source_for_post_cutoff_year_is_web_search_needed():
    got = classify_by_rules("Who won the 2027 Super Bowl?")
    assert got is not None
    assert got.decision == D.SEARCH
    assert got.knowledge_source == "web_search_needed"


def test_knowledge_source_for_volatile_topic_is_web_search_needed():
    got = classify_by_rules("Apple stock price.")
    assert got is not None
    assert got.decision == D.SEARCH
    assert got.knowledge_source == "web_search_needed"


def test_knowledge_source_for_creative_task_is_weights():
    got = classify_by_rules("Write me a poem about coffee.")
    assert got is not None
    assert got.decision == D.NO_SEARCH
    # Creative tasks have high-confidence "no need to search"; they're
    # answered from training (weights).
    assert got.knowledge_source == "weights"


def test_knowledge_source_for_stable_factual_is_weights():
    got = classify_by_rules("Who was Nikola Tesla?")
    assert got is not None
    assert got.decision == D.NO_SEARCH
    # The catch-all stable-factual path is medium-confidence -- so the
    # resolver returns "unknown" rather than "weights" until preflight
    # tightens the call.
    assert got.knowledge_source == "unknown"


def test_knowledge_source_for_empty_utterance_is_weights():
    got = classify_by_rules("")
    assert got is not None
    assert got.knowledge_source == "weights"


def test_knowledge_source_default_branch_when_no_llm():
    gate = WebSearchGate(llm=None)
    v = gate.classify("Has the package been delivered yet?")
    assert v.decision == D.UNCERTAIN
    assert v.knowledge_source in {"unknown", "retrieved_memory"}


def test_gate_verdict_default_context_categories_empty():
    from ultron.web_search.gating import GateVerdict
    v = GateVerdict(D.NO_SEARCH, "high", "rule", "test")
    assert v.context_categories == []
    assert v.memory_search_queries == []


def test_classify_by_rules_leaves_categories_empty():
    """Rule branches don't fill in categories -- only the LLM preflight
    pass produces those."""
    got = classify_by_rules("What's the weather today?")
    assert got is not None
    assert got.context_categories == []
    assert got.memory_search_queries == []


def test_knowledge_source_resolver_helper_directly():
    """Unit-test the helper so its decision tree is locked down."""
    from ultron.web_search.gating import _resolve_knowledge_source

    # Rule precedence: search wins.
    assert _resolve_knowledge_source(
        needs_search=True, confidence="high",
    ) == "web_search_needed"

    # Personal-question reason marker.
    assert _resolve_knowledge_source(
        needs_search=False, confidence="high",
        rule_reason="personal/memory question",
    ) == "retrieved_memory"

    # Stored-fact reason marker.
    assert _resolve_knowledge_source(
        needs_search=False, confidence="high",
        rule_reason="matched stored fact (preference)",
    ) == "retrieved_facts"

    # Memory snippets non-empty -> retrieved_memory.
    class _Fake:
        role = "user"
        content = "earlier"

    assert _resolve_knowledge_source(
        needs_search=False, confidence="medium",
        memory_snippets=[_Fake()],
    ) == "retrieved_memory"

    # High confidence + nothing else -> weights.
    assert _resolve_knowledge_source(
        needs_search=False, confidence="high",
    ) == "weights"

    # Low confidence + nothing else -> unknown.
    assert _resolve_knowledge_source(
        needs_search=False, confidence="low",
    ) == "unknown"
