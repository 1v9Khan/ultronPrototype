"""Catalog 12 (felo-search T3): bounded agentic deep-research loop.

Exercises the DeepResearchLoop with a fake WebSearchExecutor + a fake LLM
(no network, no real model), the step-limit / gap-fill / dedup / cap
behaviour, the fail-open paths, and the strict match_deep_research matcher.
"""

from __future__ import annotations

import json

import pytest

from ultron.web_search.deep_research import (
    DeepResearchLoop,
    DeepResearchMatch,
    DeepResearchResult,
    match_deep_research,
)
from ultron.web_search.search import SearchPayload, SearchSource


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeExecutor:
    """Stands in for WebSearchExecutor. ``results`` maps a query to the list
    of URLs it should return; unlisted queries return a single derived URL."""

    def __init__(self, results=None):
        self._results = results or {}
        self.calls = []

    def run(self, query, search_queries=None, top_n=3):
        self.calls.append(query)
        urls = self._results.get(query)
        if urls is None:
            urls = [f"https://ex/{query}"]
        srcs = [
            SearchSource(url=u, title=u, snippet=f"snip {u}", full_text=None, rank=i)
            for i, u in enumerate(urls)
        ]
        return SearchPayload(
            query=query, sources=srcs, cache_hit=False, elapsed_ms=1.0,
            queries=[query],
        )


class _FakeInner:
    def __init__(self, decompose, gap_scripts, raiser=False):
        self._decompose = decompose
        self._gaps = list(gap_scripts)
        self.raiser = raiser
        self.calls = 0

    def create_chat_completion(self, *, messages, temperature, max_tokens):
        self.calls += 1
        if self.raiser:
            raise RuntimeError("llm boom")
        content = messages[0]["content"]
        if "Findings so far" in content:  # gap-analysis call
            gaps = self._gaps.pop(0) if self._gaps else []
            return {"choices": [{"message": {"content": json.dumps({"gaps": gaps})}}]}
        return {
            "choices": [
                {"message": {"content": json.dumps({"sub_questions": self._decompose})}}
            ]
        }


class _FakeLLM:
    def __init__(self, decompose=None, gap_scripts=(), raiser=False):
        self._llm = _FakeInner(decompose or [], list(gap_scripts), raiser)


def _loop(executor, llm, **kw):
    return DeepResearchLoop(executor=executor, llm=llm, **kw)


# ---------------------------------------------------------------------------
# Core loop behaviour
# ---------------------------------------------------------------------------


def test_decompose_then_gap_fill_then_done():
    ex = _FakeExecutor()
    llm = _FakeLLM(decompose=["q1", "q2"], gap_scripts=[["q3"], []])
    res = _loop(ex, llm, max_steps=3).research("big question")
    assert isinstance(res, DeepResearchResult)
    assert res.loop_status == "completed"
    assert ex.calls == ["q1", "q2", "q3"]
    assert res.sub_queries == ["q1", "q2", "q3"]
    # one source per query (default URL), all distinct
    assert {s.url for s in res.sources} == {
        "https://ex/q1", "https://ex/q2", "https://ex/q3"
    }


def test_no_llm_falls_back_to_single_search():
    ex = _FakeExecutor()
    res = _loop(ex, None, max_steps=3).research("just this")
    # No LLM -> decompose returns nothing -> search the question verbatim;
    # gap analysis returns nothing -> done after one round.
    assert ex.calls == ["just this"]
    assert res.sub_queries == ["just this"]
    assert res.loop_status == "completed"


def test_llm_raising_is_fail_open():
    ex = _FakeExecutor()
    llm = _FakeLLM(raiser=True)
    res = _loop(ex, llm, max_steps=3).research("topic")
    # Decompose call raises -> [] -> search the question verbatim; gap call
    # raises -> [] -> done. Never propagates.
    assert ex.calls == ["topic"]
    assert res.loop_status == "completed"


def test_max_steps_cap_enforced():
    ex = _FakeExecutor()
    # Gap analysis always returns a fresh sub-question -> would loop forever
    # without the cap.
    llm = _FakeLLM(decompose=["s1"], gap_scripts=[["s2"], ["s3"], ["s4"], ["s5"]])
    res = _loop(ex, llm, max_steps=3).research("endless")
    assert res.loop_status == "max_steps_exhausted"
    assert res.steps == 3
    assert ex.calls == ["s1", "s2", "s3"]


def test_zero_progress_round_stops_loop():
    # q2 returns no sources -> a round that adds nothing finishes the loop.
    ex = _FakeExecutor(results={"q1": ["https://ex/a"], "q2": []})
    llm = _FakeLLM(decompose=["q1"], gap_scripts=[["q2"], ["q3"]])
    res = _loop(ex, llm, max_steps=4).research("q")
    assert res.loop_status == "completed"
    assert ex.calls == ["q1", "q2"]  # stopped after the zero-progress round
    assert {s.url for s in res.sources} == {"https://ex/a"}


def test_already_searched_gap_is_dropped():
    ex = _FakeExecutor()
    # The gap repeats q1 (already searched) -> filtered -> plan returns None.
    llm = _FakeLLM(decompose=["q1"], gap_scripts=[["q1"]])
    res = _loop(ex, llm, max_steps=4).research("q")
    assert ex.calls == ["q1"]
    assert res.loop_status == "completed"


def test_sources_deduped_by_url():
    ex = _FakeExecutor(results={
        "q1": ["https://ex/a", "https://ex/b"],
        "q2": ["https://ex/b", "https://ex/c"],  # b overlaps q1
    })
    llm = _FakeLLM(decompose=["q1", "q2"], gap_scripts=[[]])
    res = _loop(ex, llm, max_steps=2).research("q")
    assert [s.url for s in res.sources] == [
        "https://ex/a", "https://ex/b", "https://ex/c"
    ]


def test_max_accumulated_sources_cap():
    ex = _FakeExecutor(results={"q1": [f"https://ex/{i}" for i in range(10)]})
    llm = _FakeLLM(decompose=["q1"], gap_scripts=[["q2"], ["q3"]])
    res = _loop(ex, llm, max_steps=3, max_accumulated_sources=3).research("q")
    assert len(res.sources) == 3
    # Cap reached on the first round -> plan short-circuits to done; q2/q3
    # never searched.
    assert ex.calls == ["q1"]
    assert res.loop_status == "completed"


def test_empty_question_is_noop():
    ex = _FakeExecutor()
    res = _loop(ex, _FakeLLM(decompose=["x"]), max_steps=3).research("   ")
    assert res.question == ""
    assert res.loop_status == "empty"
    assert ex.calls == []


def test_executor_none_returns_no_sources():
    # Defensive: a loop with no executor must not crash.
    res = _loop(None, _FakeLLM(decompose=["q1"], gap_scripts=[[]]), max_steps=2).research("q")
    assert res.sources == []
    assert res.loop_status == "completed"


# ---------------------------------------------------------------------------
# DeepResearchResult.to_payload
# ---------------------------------------------------------------------------


def test_to_payload_shape():
    ex = _FakeExecutor()
    llm = _FakeLLM(decompose=["q1", "q2"], gap_scripts=[[]])
    res = _loop(ex, llm, max_steps=2).research("q")
    payload = res.to_payload()
    assert isinstance(payload, SearchPayload)
    assert payload.query == "q"
    assert payload.queries == ["q1", "q2"]            # strategy (T4)
    assert {s.url for s in payload.sources} == {"https://ex/q1", "https://ex/q2"}
    assert any("deep_research" in n for n in payload.notes)


# ---------------------------------------------------------------------------
# match_deep_research (strict matcher)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,topic", [
    ("research the fall of the Roman empire in depth", "fall of the Roman empire"),
    ("research quantum error correction thoroughly", "quantum error correction"),
    ("do a deep dive on rust async runtimes", "rust async runtimes"),
    ("deep dive into the history of jazz", "history of jazz"),
    ("give me a deep dive on transformer architectures", "transformer architectures"),
    ("do thorough research on electric vehicle batteries", "electric vehicle batteries"),
    ("do an in-depth investigation of the 2008 financial crisis", "2008 financial crisis"),
    ("dig deeper into CRISPR gene editing", "CRISPR gene editing"),
    ("deeply research the causes of inflation", "causes of inflation"),
])
def test_match_deep_research_positive(text, topic):
    m = match_deep_research(text)
    assert m is not None, f"expected a match for {text!r}"
    assert isinstance(m, DeepResearchMatch)
    assert m.topic == topic


@pytest.mark.parametrize("text", [
    "what is quantum computing",
    "search for the best ramen in Osaka",
    "look up the weather in Tokyo",
    "tell me about black holes",
    "how do transformers work",
    "research",                       # no topic
    "do some research",               # no topic
    "",
    "play the next track",
])
def test_match_deep_research_negative(text):
    assert match_deep_research(text) is None


# ---------------------------------------------------------------------------
# Orchestrator handler regression: _maybe_handle_deep_research references the
# lazy-imported ``trace`` module (NOT a module global), so the handler needs
# its own ``from ultron import trace`` -- without it, trace.tlog raised
# NameError before the try-block and crashed every deep-research command.
# The loop tests above never exercised the handler, so this locks the fix.
# ---------------------------------------------------------------------------


def test_orchestrator_deep_research_handler_runs_without_trace_nameerror(monkeypatch):
    import threading
    from types import SimpleNamespace

    from ultron.pipeline import orchestrator as orch_mod
    from ultron.pipeline.orchestrator import Orchestrator
    import ultron.web_search.deep_research as dr_mod

    monkeypatch.setattr(orch_mod.settings, "BARGE_IN_ENABLED", False, raising=False)

    class _FakeResult:
        loop_status = "completed"
        steps = 1

        def to_payload(self):
            return SimpleNamespace(sources=[], queries=[])

    class _FakeLoop:
        def __init__(self, **_kwargs):
            pass

        def research(self, _topic):
            return _FakeResult()

    # The handler imports DeepResearchLoop by name inside the method, so
    # patching the module attribute is picked up on the next call.
    monkeypatch.setattr(dr_mod, "DeepResearchLoop", _FakeLoop)

    spoken = []
    o = Orchestrator.__new__(Orchestrator)
    o.web_executor = object()  # non-None
    o.llm = object()           # non-None
    o._interrupt = threading.Event()
    o._shutdown = threading.Event()
    o._last_search_payload = None
    o._last_response_text = ""
    o._speak = lambda text: spoken.append(text)

    # Must reach + complete the handler (both trace.tlog calls run) without
    # NameError, and take the empty-sources branch.
    handled = o._maybe_handle_deep_research("research quantum computing in depth")
    assert handled is True
    assert any("couldn't surface" in s for s in spoken)
