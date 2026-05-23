"""Tests for the STT bias manager (catalog T12)."""

from __future__ import annotations

import pytest

from ultron.coding.stt_bias import (
    STTBiasManager,
    apply_bias_prompt,
    extract_identifiers,
)


def test_extract_identifiers_returns_distinct_tokens():
    out = extract_identifiers("def parse_json_input(payload): return None")
    assert "parse_json_input" in out
    assert "payload" in out
    # Python keywords filtered.
    assert "def" not in out
    assert "return" not in out


def test_extract_identifiers_filters_short_tokens():
    """Tokens under 3 chars don't make it into the prompt."""
    out = extract_identifiers("ab cd efg hij")
    assert out == ["efg", "hij"]


def test_extract_identifiers_handles_empty_input():
    assert extract_identifiers("") == []


def test_extract_identifiers_preserves_order():
    """The output should preserve first-seen order to support MRU ranking."""
    out = extract_identifiers("alpha beta alpha gamma")
    assert out == ["alpha", "beta", "gamma"]


def test_add_stores_term():
    mgr = STTBiasManager()
    mgr.add("calculate_total")
    assert "calculate_total" in mgr
    assert len(mgr) == 1


def test_add_dedupes_case_insensitively():
    mgr = STTBiasManager()
    mgr.add("Foo")
    mgr.add("foo")
    mgr.add("FOO")
    assert len(mgr) == 1


def test_add_short_terms_are_ignored():
    mgr = STTBiasManager()
    mgr.add("ab")  # too short
    mgr.add("")
    assert len(mgr) == 0


def test_add_many_inserts_in_order():
    mgr = STTBiasManager()
    mgr.add_many(["alpha", "beta", "gamma"])
    assert mgr.terms() == ["alpha", "beta", "gamma"]


def test_add_from_text_picks_up_identifiers():
    mgr = STTBiasManager()
    mgr.add_from_text("def parse_json(payload): return payload.text")
    terms = mgr.terms()
    assert "parse_json" in terms
    assert "payload" in terms
    assert "text" in terms


def test_max_terms_enforces_LRU_eviction():
    mgr = STTBiasManager(max_terms=3)
    for t in ["alpha", "beta", "gamma", "delta"]:
        mgr.add(t)
    # alpha was first in -> evicted.
    assert "alpha" not in mgr
    assert "delta" in mgr
    assert len(mgr) == 3


def test_re_adding_term_moves_it_to_MRU():
    """When a duplicate is re-added, it should be treated as fresh
    (move to MRU position so it survives the next eviction)."""
    mgr = STTBiasManager(max_terms=3)
    mgr.add("alpha")
    mgr.add("beta")
    mgr.add("gamma")
    # Re-add alpha -> it moves to MRU.
    mgr.add("alpha")
    # Now add delta -> evicts the LRU which is now beta.
    mgr.add("delta")
    assert "alpha" in mgr  # survived eviction
    assert "beta" not in mgr  # got evicted because it's now LRU


def test_render_prompt_orders_mru_first():
    mgr = STTBiasManager()
    mgr.add("first")
    mgr.add("second")
    mgr.add("third")
    rendered = mgr.render_prompt()
    # MRU-first: "third" should appear before "first".
    assert rendered.index("third") < rendered.index("first")


def test_render_prompt_respects_max_chars():
    mgr = STTBiasManager(max_chars=30)
    mgr.add("aaaa_first_term")  # 15 chars
    mgr.add("bbbb_second_term")  # 16 chars
    mgr.add("cccc_third_term")  # 15 chars
    rendered = mgr.render_prompt()
    # 30-char cap is tight enough to truncate after the first ~one and a half terms.
    assert len(rendered) <= 30


def test_render_prompt_with_zero_max_chars_returns_empty():
    mgr = STTBiasManager(max_chars=0)
    mgr.add("alpha")
    assert mgr.render_prompt() == ""


def test_render_prompt_empty_when_no_terms():
    mgr = STTBiasManager()
    assert mgr.render_prompt() == ""


def test_remove_drops_term():
    mgr = STTBiasManager()
    mgr.add("alpha")
    assert mgr.remove("alpha") is True
    assert "alpha" not in mgr
    # second removal is a no-op
    assert mgr.remove("alpha") is False


def test_clear_drops_all_terms():
    mgr = STTBiasManager()
    mgr.add_many(["a_one", "a_two", "a_three"])
    mgr.clear()
    assert len(mgr) == 0


def test_constructor_rejects_invalid_caps():
    with pytest.raises(ValueError):
        STTBiasManager(max_terms=0)
    with pytest.raises(ValueError):
        STTBiasManager(max_chars=-1)


# ---------------------------------------------------------------------------
# apply_bias_prompt
# ---------------------------------------------------------------------------


class _EngineWithInitialPrompt:
    initial_prompt = None


class _EngineWithoutBiasSupport:
    pass


class _EngineWithBiasPrompt:
    bias_prompt = None


def test_apply_bias_prompt_sets_initial_prompt():
    engine = _EngineWithInitialPrompt()
    assert apply_bias_prompt(engine, "context terms here") is True
    assert engine.initial_prompt == "context terms here"


def test_apply_bias_prompt_sets_bias_prompt_attr():
    engine = _EngineWithBiasPrompt()
    assert apply_bias_prompt(engine, "context") is True
    assert engine.bias_prompt == "context"


def test_apply_bias_prompt_unsupported_engine_returns_false():
    engine = _EngineWithoutBiasSupport()
    assert apply_bias_prompt(engine, "context") is False


def test_apply_bias_prompt_empty_prompt_returns_false():
    engine = _EngineWithInitialPrompt()
    assert apply_bias_prompt(engine, "") is False
    # And didn't mutate the engine.
    assert engine.initial_prompt is None
