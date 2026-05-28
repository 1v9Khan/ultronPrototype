"""Catalog 12 (felo-search T1): pre-search query reformulation.

Pure-function rule expansion + the LLM path (fake engine, no real model)
+ the executor-facing ``maybe_reformulate_queries`` helper (config
monkeypatched; log redirected to tmp_path per the binding test rules).
"""

from __future__ import annotations

import json
import types

import pytest

from ultron.web_search.query_rewrite import (
    DEFAULT_MAX_VARIANTS,
    MAX_TOTAL_QUERIES,
    QueryReformulation,
    _parse_queries_json,
    expand_query_llm,
    expand_query_rules,
    maybe_reformulate_queries,
    reformulate_query,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeInner:
    def __init__(self, content: str = "", raiser: bool = False) -> None:
        self.content = content
        self.raiser = raiser
        self.calls = 0

    def create_chat_completion(self, *, messages, temperature, max_tokens):
        self.calls += 1
        if self.raiser:
            raise RuntimeError("boom")
        return {"choices": [{"message": {"content": self.content}}]}


class _FakeLLM:
    """Mirrors the ``llm._llm.create_chat_completion`` seam the module uses."""

    def __init__(self, content: str = "", raiser: bool = False) -> None:
        self._llm = _FakeInner(content, raiser)


def _fake_cfg(*, enabled=True, use_llm=False, max_variants=2):
    return types.SimpleNamespace(
        web_search=types.SimpleNamespace(
            query_reformulation=types.SimpleNamespace(
                enabled=enabled, use_llm=use_llm, max_variants=max_variants,
            )
        )
    )


# ---------------------------------------------------------------------------
# expand_query_rules
# ---------------------------------------------------------------------------


def test_comparison_split_grafts_shared_tail():
    out = expand_query_rules("Python vs Go for backend services", max_variants=2)
    assert out == ["Python for backend services", "Go for backend services"]


def test_comparison_split_bare_subjects():
    out = expand_query_rules("cats vs dogs", max_variants=2)
    assert out == ["cats", "dogs"]


def test_comparison_split_versus_keyword():
    out = expand_query_rules("Vue 3 versus React for new projects", max_variants=2)
    assert out == ["Vue 3 for new projects", "React for new projects"]


def test_howto_expands_to_tutorial_and_guide():
    out = expand_query_rules("how to install Docker on Windows", max_variants=2)
    assert out == ["install Docker on Windows tutorial", "install Docker on Windows guide"]


def test_best_expands_to_review_and_comparison():
    out = expand_query_rules("best ramen in Osaka", max_variants=2)
    assert out == ["ramen in Osaka review", "ramen in Osaka comparison"]


def test_best_practice_is_not_expanded():
    # "best practices" is how-to territory, not a product/recommendation
    # query -- the review/comparison expansion would be noise.
    out = expand_query_rules("best practices for REST API design", max_variants=2)
    assert out == []


def test_leading_temporal_adds_bare_subject():
    out = expand_query_rules("latest iPhone release", max_variants=2)
    assert "iPhone release" in out


def test_no_rule_match_returns_empty():
    assert expand_query_rules("what is photosynthesis", max_variants=2) == []
    assert expand_query_rules("tell me about black holes", max_variants=2) == []


def test_max_variants_caps_output():
    out = expand_query_rules("how to install Docker", max_variants=1)
    assert out == ["install Docker tutorial"]


def test_max_variants_zero_and_empty_input():
    assert expand_query_rules("how to install Docker", max_variants=0) == []
    assert expand_query_rules("", max_variants=2) == []
    assert expand_query_rules("   ", max_variants=2) == []


def test_rule_variants_never_include_original():
    # A query whose expansion would echo the original is filtered.
    out = expand_query_rules("cats vs dogs", max_variants=4)
    assert "cats vs dogs" not in out


# ---------------------------------------------------------------------------
# QueryReformulation.all_queries
# ---------------------------------------------------------------------------


def test_all_queries_dedup_order_preserving():
    r = QueryReformulation(
        original="Python", variants=("python", "Go", "Python", "Rust"), method="rules"
    )
    # Original first; case-insensitive dedup ("python" == "Python");
    # order preserved.
    assert r.all_queries == ["Python", "Go", "Rust"]


def test_all_queries_strips_blanks():
    r = QueryReformulation(original="  X  ", variants=("", "  ", "Y"), method="rules")
    assert r.all_queries == ["X", "Y"]


# ---------------------------------------------------------------------------
# _parse_queries_json
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ('{"queries": ["a", "b"]}', ["a", "b"]),
    ('{"search_queries": ["x"]}', ["x"]),
    ('["a", "b"]', ["a", "b"]),
    ('```json\n{"queries": ["fenced"]}\n```', ["fenced"]),
    ('<think>reasoning</think>{"queries": ["c"]}', ["c"]),
    ('here you go: {"queries": ["d", "e"]} done', ["d", "e"]),
    ("not json at all", []),
    ("", []),
    ('{"queries": [1, 2, "ok"]}', ["ok"]),  # non-strings dropped
])
def test_parse_queries_json(raw, expected):
    assert _parse_queries_json(raw) == expected


# ---------------------------------------------------------------------------
# expand_query_llm
# ---------------------------------------------------------------------------


def test_expand_query_llm_parses_and_dedupes():
    llm = _FakeLLM('{"queries": ["A backend", "B backend", "A backend"]}')
    out = expand_query_llm("A vs B backend", llm, max_variants=3)
    assert out == ["A backend", "B backend"]
    assert llm._llm.calls == 1


def test_expand_query_llm_excludes_original():
    llm = _FakeLLM('{"queries": ["A vs B", "fresh take"]}')
    out = expand_query_llm("A vs B", llm, max_variants=3)
    assert out == ["fresh take"]


def test_expand_query_llm_caps_variants():
    llm = _FakeLLM('{"queries": ["one", "two", "three", "four"]}')
    out = expand_query_llm("q", llm, max_variants=2)
    assert out == ["one", "two"]


def test_expand_query_llm_fail_open_on_exception():
    llm = _FakeLLM(raiser=True)
    assert expand_query_llm("q", llm, max_variants=2) == []


def test_expand_query_llm_none_llm_and_empty():
    assert expand_query_llm("q", None, max_variants=2) == []
    assert expand_query_llm("", _FakeLLM('{"queries":["a"]}'), max_variants=2) == []


# ---------------------------------------------------------------------------
# reformulate_query
# ---------------------------------------------------------------------------


def test_reformulate_disabled_returns_none_method():
    r = reformulate_query("how to install Docker", enabled=False)
    assert r.method == "none" and r.variants == ()


def test_reformulate_rules_path():
    r = reformulate_query("how to install nginx", use_llm=False, max_variants=2)
    assert r.method == "rules"
    assert r.variants == ("install nginx tutorial", "install nginx guide")


def test_reformulate_llm_path():
    llm = _FakeLLM('{"queries": ["alpha", "beta"]}')
    r = reformulate_query("complex question", use_llm=True, llm=llm, max_variants=2)
    assert r.method == "llm"
    assert r.variants == ("alpha", "beta")


def test_reformulate_llm_empty_falls_back_to_rules():
    # LLM returns nothing parseable; a recognised shape still yields rules.
    llm = _FakeLLM("garbage")
    r = reformulate_query("how to install nginx", use_llm=True, llm=llm, max_variants=2)
    assert r.method == "rules"
    assert r.variants == ("install nginx tutorial", "install nginx guide")


def test_reformulate_llm_empty_and_no_rule_is_none():
    llm = _FakeLLM("garbage")
    r = reformulate_query("explain photosynthesis", use_llm=True, llm=llm, max_variants=2)
    assert r.method == "none" and r.variants == ()


def test_reformulate_empty_query():
    r = reformulate_query("   ", use_llm=False)
    assert r.method == "none" and r.variants == ()


# ---------------------------------------------------------------------------
# maybe_reformulate_queries (executor-facing; config monkeypatched)
# ---------------------------------------------------------------------------


def test_maybe_reformulate_disabled_passthrough(monkeypatch):
    monkeypatch.setattr("ultron.config.get_config", lambda: _fake_cfg(enabled=False))
    base = ["how to install Docker"]
    assert maybe_reformulate_queries("how to install Docker", base) == base


def test_maybe_reformulate_merges_variants(monkeypatch, tmp_path):
    monkeypatch.setattr("ultron.config.get_config", lambda: _fake_cfg(enabled=True))
    monkeypatch.setattr("ultron.config.LOGS_DIR", tmp_path)
    out = maybe_reformulate_queries(
        "how to install Docker", ["how to install Docker"],
    )
    assert out == [
        "how to install Docker",
        "install Docker tutorial",
        "install Docker guide",
    ]
    # log row written to the redirected LOGS_DIR (R9: not the repo logs/)
    log = tmp_path / "search_reformulations.jsonl"
    assert log.exists()
    row = json.loads(log.read_text(encoding="utf-8").strip())
    assert row["method"] == "rules"
    assert row["query"] == "how to install Docker"


def test_maybe_reformulate_caps_total(monkeypatch, tmp_path):
    monkeypatch.setattr("ultron.config.get_config", lambda: _fake_cfg(enabled=True))
    monkeypatch.setattr("ultron.config.LOGS_DIR", tmp_path)
    # 4 base queries already; reformulation must not exceed MAX_TOTAL_QUERIES.
    base = ["how to install Docker", "q2", "q3", "q4"]
    out = maybe_reformulate_queries("how to install Docker", base)
    assert len(out) <= MAX_TOTAL_QUERIES
    assert out[:4] == base  # base preserved, order-first


def test_maybe_reformulate_no_variants_returns_base(monkeypatch):
    monkeypatch.setattr("ultron.config.get_config", lambda: _fake_cfg(enabled=True))
    base = ["explain photosynthesis"]
    # No rule matches -> no variants -> base returned unchanged.
    assert maybe_reformulate_queries("explain photosynthesis", base) == base


def test_maybe_reformulate_fail_open_on_config_error(monkeypatch):
    def _boom():
        raise RuntimeError("config blew up")

    monkeypatch.setattr("ultron.config.get_config", _boom)
    base = ["how to install Docker"]
    # Config failure must not break the search path.
    assert maybe_reformulate_queries("how to install Docker", base) == base


def test_maybe_reformulate_empty_base_falls_back_to_user_query(monkeypatch):
    monkeypatch.setattr("ultron.config.get_config", lambda: _fake_cfg(enabled=True))
    # Empty base + a query with no rule match -> [user_query].
    assert maybe_reformulate_queries("explain photosynthesis", []) == ["explain photosynthesis"]


def test_default_constants_sane():
    assert DEFAULT_MAX_VARIANTS == 2
    assert MAX_TOTAL_QUERIES >= 3
