"""4B optimization plan Item 6 — self-consistency machinery + decomposer integration.

Tests cover:
- The aggregator helpers (text mode, JSON mode, label mode)
- The ``run_self_consistency`` driver
- The config gate (``should_apply_self_consistency``)
- The decomposer's flag-OFF / flag-ON behaviour (back-compat + lift)

Mocked LLM throughout — no GPU.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from ultron.llm.self_consistency import (
    ConsistencyResult,
    majority_vote_json,
    majority_vote_label,
    majority_vote_text,
    run_self_consistency,
    should_apply_self_consistency,
)


# ---------------------------------------------------------------------------
# majority_vote_text
# ---------------------------------------------------------------------------


def test_majority_vote_text_winner() -> None:
    answer, votes = majority_vote_text(["yes", "yes", "no"])
    assert answer == "yes"
    assert votes == {"yes": 2, "no": 1}


def test_majority_vote_text_strips_whitespace() -> None:
    answer, votes = majority_vote_text(["  yes ", "yes\n", " no"])
    assert answer == "yes"
    assert votes["yes"] == 2


def test_majority_vote_text_tie_first_occurrence_wins() -> None:
    answer, _ = majority_vote_text(["a", "b"])
    assert answer == "a"


def test_majority_vote_text_empty_input() -> None:
    answer, votes = majority_vote_text([])
    assert answer == ""
    assert votes == {}


def test_majority_vote_text_filters_blank() -> None:
    answer, votes = majority_vote_text(["", "  ", "yes"])
    assert answer == "yes"


# ---------------------------------------------------------------------------
# majority_vote_json
# ---------------------------------------------------------------------------


def test_majority_vote_json_picks_consistent_winner() -> None:
    samples = [
        '{"verdict": "CODING", "n": 2}',
        '{"verdict": "CODING", "n": 2}',
        '{"verdict": "AUTOMATION", "n": 1}',
    ]
    parsed, votes = majority_vote_json(samples)
    assert parsed == {"verdict": "CODING", "n": 2}
    # The serialised key for the winner should have count 2
    assert max(votes.values()) == 2


def test_majority_vote_json_handles_unparseable() -> None:
    samples = [
        "not JSON at all",
        '{"verdict": "CODING"}',
        '{"verdict": "CODING"}',
    ]
    parsed, votes = majority_vote_json(samples)
    assert parsed == {"verdict": "CODING"}
    # Two parsed → counts of 2
    assert max(votes.values()) == 2


def test_majority_vote_json_strips_thinking_blocks() -> None:
    samples = [
        '<think>I need to think about this</think>{"verdict": "CODING"}',
        '<think>...</think>{"verdict": "CODING"}',
    ]
    parsed, _ = majority_vote_json(samples)
    assert parsed == {"verdict": "CODING"}


def test_majority_vote_json_picks_first_block_only() -> None:
    """Multiple JSON blocks in one sample: only the first counts."""
    samples = [
        '{"a": 1} ignore this {"a": 2}',
        '{"a": 1}',
    ]
    parsed, _ = majority_vote_json(samples)
    assert parsed == {"a": 1}


def test_majority_vote_json_all_unparseable_returns_none() -> None:
    parsed, votes = majority_vote_json(["nope", "still nope"])
    assert parsed is None
    assert votes == {}


def test_majority_vote_json_handles_arrays() -> None:
    samples = [
        '[{"x": 1}, {"x": 2}]',
        '[{"x": 1}, {"x": 2}]',
    ]
    parsed, votes = majority_vote_json(samples)
    assert parsed == [{"x": 1}, {"x": 2}]
    assert max(votes.values()) == 2


# ---------------------------------------------------------------------------
# majority_vote_label
# ---------------------------------------------------------------------------


def test_majority_vote_label_extracts_first_match() -> None:
    samples = [
        "Looking at this... CODING.",
        "I think it's CODING.",
        "AUTOMATION — open the file.",
    ]
    answer, votes = majority_vote_label(
        samples, ["CODING", "AUTOMATION", "HYBRID", "UNCLEAR"],
    )
    assert answer == "CODING"
    assert votes == {"CODING": 2, "AUTOMATION": 1}


def test_majority_vote_label_case_insensitive() -> None:
    samples = ["coding", "Coding", "CODING"]
    answer, _ = majority_vote_label(
        samples, ["CODING", "AUTOMATION"],
    )
    assert answer == "CODING"


def test_majority_vote_label_no_matches_returns_none() -> None:
    samples = ["I have no idea", "yeah dunno"]
    answer, votes = majority_vote_label(
        samples, ["CODING", "AUTOMATION"],
    )
    assert answer is None
    assert votes == {}


# ---------------------------------------------------------------------------
# run_self_consistency driver
# ---------------------------------------------------------------------------


def test_run_self_consistency_calls_sampler_n_times() -> None:
    calls = []

    def sampler(temp):
        calls.append(temp)
        return "yes"

    result = run_self_consistency(sampler, n=3, temperature=0.7)
    assert len(calls) == 3
    assert all(c == 0.7 for c in calls)
    assert isinstance(result, ConsistencyResult)
    assert result.answer == "yes"
    assert result.votes == {"yes": 3}


def test_run_self_consistency_default_aggregator_is_text() -> None:
    samples = iter(["yes", "yes", "no"])
    result = run_self_consistency(lambda t: next(samples), n=3)
    assert result.answer == "yes"


def test_run_self_consistency_handles_sampler_exception() -> None:
    """A failed sample is recorded as an empty string, not raised."""
    samples = iter(["yes", "yes"])

    def sampler(t):
        try:
            return next(samples)
        except StopIteration:
            raise RuntimeError("third call boom")

    result = run_self_consistency(sampler, n=3)
    assert result.answer == "yes"
    assert "" in result.samples


def test_run_self_consistency_fallback_to_first_when_unaggregatable() -> None:
    """If the aggregator returns ('', votes), the driver falls back to
    the first non-empty sample with fallback_used=True."""
    samples = iter(["sample-A", "sample-B", "sample-C"])

    def empty_aggregator(samples):
        return "", {}

    result = run_self_consistency(
        lambda t: next(samples), n=3, aggregator=empty_aggregator,
    )
    assert result.answer == "sample-A"
    assert result.fallback_used is True


def test_run_self_consistency_clamps_n_to_one_minimum() -> None:
    calls = []

    def sampler(t):
        calls.append(t)
        return "x"

    result = run_self_consistency(sampler, n=0)
    assert len(calls) == 1  # clamped to 1
    assert result.answer == "x"


# ---------------------------------------------------------------------------
# Config gate
# ---------------------------------------------------------------------------


def test_should_apply_self_consistency_default_off() -> None:
    cfg = MagicMock()
    cfg.llm.self_consistency.enabled = False
    cfg.llm.self_consistency.disabled_sites = []
    assert should_apply_self_consistency("decomposer", cfg) is False


def test_should_apply_self_consistency_global_on() -> None:
    cfg = MagicMock()
    cfg.llm.self_consistency.enabled = True
    cfg.llm.self_consistency.disabled_sites = []
    assert should_apply_self_consistency("decomposer", cfg) is True


def test_should_apply_self_consistency_per_site_disabled() -> None:
    cfg = MagicMock()
    cfg.llm.self_consistency.enabled = True
    cfg.llm.self_consistency.disabled_sites = ["decomposer"]
    assert should_apply_self_consistency("decomposer", cfg) is False
    assert should_apply_self_consistency("disambiguator", cfg) is True


# ---------------------------------------------------------------------------
# Decomposer integration
# ---------------------------------------------------------------------------


def _async_run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def cfg_mock():
    cfg = MagicMock()
    cfg.routing.hybrid_task_decomposition_enabled = True
    cfg.llm.self_consistency.enabled = False
    cfg.llm.self_consistency.disabled_sites = []
    cfg.llm.self_consistency.n = 3
    cfg.llm.self_consistency.temperature = 0.8
    return cfg


def test_decomposer_default_off_uses_single_llm_call(cfg_mock) -> None:
    """Self-consistency OFF (default) ⇒ one llm.generate call per
    decompose. Back-compat guarantee."""
    from ultron.openclaw_routing.decomposer import HybridTaskDecomposer

    llm = MagicMock()
    llm.generate.return_value = (
        '{"subtasks": [{"order": 1, "type": "coding", "description": "x"}]}'
    )
    d = HybridTaskDecomposer(llm)
    with patch("ultron.openclaw_routing.decomposer.get_config", return_value=cfg_mock):
        result = asyncio.new_event_loop().run_until_complete(d.decompose("u"))

    assert llm.generate.call_count == 1
    assert result.fallback_used is False
    assert result.subtasks[0].description == "x"


def test_decomposer_self_consistency_on_calls_n_times(cfg_mock) -> None:
    cfg_mock.llm.self_consistency.enabled = True
    cfg_mock.llm.self_consistency.n = 3

    from ultron.openclaw_routing.decomposer import HybridTaskDecomposer

    llm = MagicMock()
    # All three samples agree
    payload = '{"subtasks": [{"order": 1, "type": "coding", "description": "x"}]}'
    llm.generate.return_value = payload
    d = HybridTaskDecomposer(llm)
    with patch("ultron.openclaw_routing.decomposer.get_config", return_value=cfg_mock):
        result = asyncio.new_event_loop().run_until_complete(d.decompose("u"))

    assert llm.generate.call_count == 3
    assert result.fallback_used is False
    assert result.subtasks[0].description == "x"


def test_decomposer_self_consistency_majority_wins(cfg_mock) -> None:
    """Two samples agree on plan A, one diverges to plan B → A wins."""
    cfg_mock.llm.self_consistency.enabled = True
    cfg_mock.llm.self_consistency.n = 3

    from ultron.openclaw_routing.decomposer import HybridTaskDecomposer

    llm = MagicMock()
    plan_a = '{"subtasks": [{"order": 1, "type": "coding", "description": "alpha"}]}'
    plan_b = '{"subtasks": [{"order": 1, "type": "automation", "description": "beta"}]}'
    llm.generate.side_effect = [plan_a, plan_a, plan_b]
    d = HybridTaskDecomposer(llm)
    with patch("ultron.openclaw_routing.decomposer.get_config", return_value=cfg_mock):
        result = asyncio.new_event_loop().run_until_complete(d.decompose("u"))

    assert llm.generate.call_count == 3
    # plan_a is the majority (2 of 3) → "alpha" wins
    assert result.subtasks[0].description == "alpha"
    assert result.subtasks[0].type == "coding"


def test_decomposer_self_consistency_per_site_disabled(cfg_mock) -> None:
    """Even with global flag on, per-site disabled list bypasses
    self-consistency for that call site."""
    cfg_mock.llm.self_consistency.enabled = True
    cfg_mock.llm.self_consistency.disabled_sites = ["decomposer"]

    from ultron.openclaw_routing.decomposer import HybridTaskDecomposer

    llm = MagicMock()
    llm.generate.return_value = (
        '{"subtasks": [{"order": 1, "type": "coding", "description": "x"}]}'
    )
    d = HybridTaskDecomposer(llm)
    with patch("ultron.openclaw_routing.decomposer.get_config", return_value=cfg_mock):
        asyncio.new_event_loop().run_until_complete(d.decompose("u"))

    assert llm.generate.call_count == 1  # back to single call


def test_decomposer_self_consistency_all_unparseable_falls_back(cfg_mock) -> None:
    cfg_mock.llm.self_consistency.enabled = True

    from ultron.openclaw_routing.decomposer import HybridTaskDecomposer

    llm = MagicMock()
    llm.generate.return_value = "not JSON at all"
    d = HybridTaskDecomposer(llm)
    with patch("ultron.openclaw_routing.decomposer.get_config", return_value=cfg_mock):
        result = asyncio.new_event_loop().run_until_complete(d.decompose("the original utterance"))

    # Falls back to coding-only with the original utterance
    assert result.fallback_used is True
    assert result.subtasks[0].description == "the original utterance"
