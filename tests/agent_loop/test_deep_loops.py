"""Catalog 12 (felo-search T3 extensions): cross-system deep-gather loops.

Exercises the generic DeepGatherLoop + the three domain subclasses
(DeepMemoryLoop / DeepExplorationLoop / DeepUIDiscoveryLoop) with injected
fake gather callables + a fake LLM. No Qdrant / ripgrep / live desktop.
"""

from __future__ import annotations

import json
import types

import pytest

from ultron.agent_loop.deep_loops import (
    DeepExplorationLoop,
    DeepGatherLoop,
    DeepGatherResult,
    DeepMemoryLoop,
    DeepUIDiscoveryLoop,
)


# ---------------------------------------------------------------------------
# Fake LLM (decompose + gap scripting, same shape as test_deep_research)
# ---------------------------------------------------------------------------


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
        # The gap prompts all contain a "... so far:" findings section.
        if "so far:" in content:
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


# ---------------------------------------------------------------------------
# Generic DeepGatherLoop
# ---------------------------------------------------------------------------


def _generic(gather, llm, **kw):
    return DeepGatherLoop(
        llm=llm,
        kind="test",
        gather=gather,
        item_key=lambda x: x,
        item_summary=lambda x: str(x),
        decompose_prompt="decompose {n} {query}",
        gap_prompt="gaps {n} {query} so far:\n{findings}",
        **kw,
    )


def test_generic_decompose_gather_gapfill_done():
    seen = []

    def gather(q):
        seen.append(q)
        return [f"{q}:0", f"{q}:1"]

    llm = _FakeLLM(decompose=["a", "b"], gap_scripts=[["c"], []])
    res = _generic(gather, llm, max_steps=3).gather_for("topic")
    assert isinstance(res, DeepGatherResult)
    assert res.loop_status == "completed"
    assert seen == ["a", "b", "c"]
    assert res.sub_queries == ["a", "b", "c"]
    assert set(res.items) == {"a:0", "a:1", "b:0", "b:1", "c:0", "c:1"}


def test_generic_no_llm_single_gather():
    seen = []
    res = _generic(lambda q: seen.append(q) or ["x"], None, max_steps=3).gather_for("q")
    assert seen == ["q"]
    assert res.sub_queries == ["q"]


def test_generic_llm_raising_fail_open():
    seen = []
    llm = _FakeLLM(raiser=True)
    res = _generic(lambda q: seen.append(q) or ["x"], llm, max_steps=3).gather_for("q")
    assert seen == ["q"]
    assert res.loop_status == "completed"


def test_generic_gather_raising_is_skipped():
    def gather(q):
        raise RuntimeError("gather boom")

    llm = _FakeLLM(decompose=["a"], gap_scripts=[[]])
    res = _generic(gather, llm, max_steps=2).gather_for("q")
    # gather raised -> 0 items, but the loop completes without propagating.
    assert res.items == []
    assert res.loop_status == "completed"


def test_generic_max_steps_cap():
    llm = _FakeLLM(decompose=["s1"], gap_scripts=[["s2"], ["s3"], ["s4"]])
    res = _generic(lambda q: [f"{q}.item"], llm, max_steps=3).gather_for("q")
    assert res.loop_status == "max_steps_exhausted"
    assert res.steps == 3


def test_generic_dedup_by_key():
    def gather(q):
        return {"a": ["u1", "u2"], "b": ["u2", "u3"]}.get(q, [])

    llm = _FakeLLM(decompose=["a", "b"], gap_scripts=[[]])
    res = _generic(gather, llm, max_steps=2).gather_for("q")
    assert res.items == ["u1", "u2", "u3"]  # u2 deduped, order preserved


def test_generic_accumulated_cap():
    llm = _FakeLLM(decompose=["a"], gap_scripts=[["b"], ["c"]])
    res = _generic(
        lambda q: [f"{q}{i}" for i in range(10)], llm,
        max_steps=3, max_accumulated=3,
    ).gather_for("q")
    assert len(res.items) == 3
    assert res.loop_status == "completed"


def test_generic_empty_question_noop():
    seen = []
    res = _generic(lambda q: seen.append(q) or ["x"], _FakeLLM(decompose=["a"])).gather_for("  ")
    assert res.loop_status == "empty"
    assert seen == []


def test_generic_zero_progress_stops():
    def gather(q):
        return {"a": ["u1"]}.get(q, [])  # "b" returns nothing

    llm = _FakeLLM(decompose=["a"], gap_scripts=[["b"], ["c"]])
    res = _generic(gather, llm, max_steps=4).gather_for("q")
    assert res.loop_status == "completed"
    assert res.items == ["u1"]


# ---------------------------------------------------------------------------
# DeepMemoryLoop
# ---------------------------------------------------------------------------


def _turn(tid, role, content):
    return types.SimpleNamespace(id=tid, role=role, content=content)


def test_deep_memory_recall_dedups_by_id():
    store = {
        "decisions about the db": [_turn("t1", "user", "use postgres"), _turn("t2", "assistant", "ok")],
        "preferences for the db": [_turn("t2", "assistant", "ok"), _turn("t3", "user", "no orm")],
    }
    llm = _FakeLLM(decompose=["decisions about the db", "preferences for the db"], gap_scripts=[[]])
    res = DeepMemoryLoop(retrieve=lambda q, k: store.get(q, []), llm=llm, max_steps=2).recall("the db")
    assert res.kind == "memory"
    ids = [t.id for t in res.items]
    assert ids == ["t1", "t2", "t3"]  # t2 deduped


def test_deep_memory_passes_k_to_retrieve():
    captured = {}

    def retrieve(q, k):
        captured["k"] = k
        return [_turn("t1", "user", "x")]

    llm = _FakeLLM(decompose=["q1"], gap_scripts=[[]])
    DeepMemoryLoop(retrieve=retrieve, llm=llm, k_per_query=5, max_steps=2).recall("q")
    assert captured["k"] == 5


# ---------------------------------------------------------------------------
# DeepExplorationLoop
# ---------------------------------------------------------------------------


def _match(path, line, text):
    return types.SimpleNamespace(file_path=path, line_number=line, text=text)


def test_deep_exploration_dedups_by_file_line():
    store = {
        "def foo": [_match("a.py", 10, "def foo():"), _match("b.py", 5, "foo()")],
        "foo callers": [_match("b.py", 5, "foo()"), _match("c.py", 1, "import foo")],
    }
    llm = _FakeLLM(decompose=["def foo", "foo callers"], gap_scripts=[[]])
    res = DeepExplorationLoop(search=lambda q: store.get(q, []), llm=llm, max_steps=2).explore("how foo works")
    assert res.kind == "code"
    keys = [(m.file_path, m.line_number) for m in res.items]
    assert keys == [("a.py", 10), ("b.py", 5), ("c.py", 1)]  # (b.py,5) deduped


# ---------------------------------------------------------------------------
# DeepUIDiscoveryLoop
# ---------------------------------------------------------------------------


def _element(name, ctype, window):
    return types.SimpleNamespace(name=name, control_type=ctype, window=window)


def test_deep_ui_discovery_dedups_by_window_name_type():
    store = {
        "Submit": [_element("Submit", "Button", "Form")],
        "OK": [_element("Submit", "Button", "Form"), _element("OK", "Button", "Dialog")],
    }
    llm = _FakeLLM(decompose=["Submit", "OK"], gap_scripts=[[]])
    res = DeepUIDiscoveryLoop(find=lambda q: store.get(q, []), llm=llm, max_steps=2).discover("the submit button")
    assert res.kind == "ui"
    names = [e.name for e in res.items]
    assert names == ["Submit", "OK"]  # duplicate Submit@Form deduped
