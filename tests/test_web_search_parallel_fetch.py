"""Regression tests for the WebSearchExecutor parallel-Jina-fetch + collective-deadline pathway.

Replaces the pre-2026-05-09 sequential fetch loop. The executor now
fans Jina fetches out across a thread pool and waits at most
``collective_deadline_seconds`` for ALL of them. Anything still in
flight at deadline is abandoned (source falls back to snippet-only)
so a single pathological page can't block the entire voice path.

These tests mock the Jina client to control timing precisely without
real network calls.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from ultron.web_search import (
    BraveResult,
    BraveSearchClient,
    JinaReaderClient,
    WebSearchExecutor,
)


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class _StubLLM:
    """Snippet-ranking stub that returns indices [1, 2, 3] in order."""

    @property
    def _llm(self):
        out = MagicMock()
        out.create_chat_completion = MagicMock(return_value={
            "choices": [{"message": {"content": '{"ranked_indices":[1,2,3]}'}}],
        })
        return out

    def generate(self, prompt: str) -> str:
        return '{"ranked_indices":[1,2,3]}'


class _MockBrave(BraveSearchClient):
    """Bypasses __init__ so we don't need an API key."""

    def __init__(self, fixture=None):
        self.api_key = "mock"
        self.endpoint = "mock://"
        self.rate_limit_s = 0.0
        self.timeout_s = 0.0
        self._last_call = 0.0
        self._lock = threading.Lock()
        if fixture is None:
            fixture = [
                BraveResult(url="https://example.com/a", title="A", snippet="snippet A", rank=0),
                BraveResult(url="https://example.com/b", title="B", snippet="snippet B", rank=1),
                BraveResult(url="https://example.com/c", title="C", snippet="snippet C", rank=2),
            ]
        self._fixture = list(fixture)

    def search(self, query, count=5):
        return list(self._fixture[:count])


class _TimedJina(JinaReaderClient):
    """Jina mock whose fetch sleeps a configurable per-URL duration.

    Lets us simulate slow / fast / pathological pages and verify the
    executor's collective-deadline behaviour without real HTTP.
    """

    def __init__(self, durations: dict[str, float], body_template: str = "# Page {url}"):
        self.endpoint = "mock://"
        self.timeout_s = 0.0
        self.max_bytes = 200_000
        self._durations = durations
        self._template = body_template

    def fetch(self, url):
        delay = self._durations.get(url, 0.0)
        if delay > 0:
            time.sleep(delay)
        return self._template.format(url=url)


# ---------------------------------------------------------------------------
# Parallel fetch wall-clock
# ---------------------------------------------------------------------------


def test_parallel_fetch_walltime_dominated_by_slowest_url():
    """Three 0.4 s fetches should complete in ~0.4 s wall time, not 1.2 s.

    Pre-fix this loop was sequential -- wall = sum(durations). Post-fix
    it runs in parallel -- wall = max(durations) (plus a small thread
    setup overhead).
    """
    fixture = [
        BraveResult(url="https://example.com/a", title="A", snippet="a", rank=0),
        BraveResult(url="https://example.com/b", title="B", snippet="b", rank=1),
        BraveResult(url="https://example.com/c", title="C", snippet="c", rank=2),
    ]
    durations = {
        "https://example.com/a": 0.4,
        "https://example.com/b": 0.4,
        "https://example.com/c": 0.4,
    }
    executor = WebSearchExecutor(
        brave=_MockBrave(fixture=fixture),
        jina=_TimedJina(durations),
        llm=_StubLLM(),
        cache=None,
        max_fetch=3,
        collective_deadline_seconds=5.0,
    )
    t0 = time.monotonic()
    payload = executor.run("anything", top_n=3)
    wall = time.monotonic() - t0

    assert len(payload.sources) == 3
    assert all(s.full_text is not None for s in payload.sources)
    # Generous bound: 0.4 s (parallel max) + thread-setup slack.
    # Sequential would be ~1.2 s.
    assert wall < 1.0, f"parallel fetch took {wall:.2f}s; expected <1.0s"


def test_collective_deadline_abandons_slow_fetches():
    """A 5 s fetch is abandoned when the collective deadline is 0.5 s.

    The executor must return promptly; the slow source degrades to
    snippet-only via ``full_text=None``. Notes record which URL hit
    the deadline.
    """
    fixture = [
        BraveResult(url="https://example.com/fast", title="Fast", snippet="fast", rank=0),
        BraveResult(url="https://example.com/slow", title="Slow", snippet="slow", rank=1),
    ]
    durations = {
        "https://example.com/fast": 0.05,
        "https://example.com/slow": 5.0,    # never returns within deadline
    }
    executor = WebSearchExecutor(
        brave=_MockBrave(fixture=fixture),
        jina=_TimedJina(durations),
        llm=_StubLLM(),
        cache=None,
        max_fetch=2,
        collective_deadline_seconds=0.5,
    )
    t0 = time.monotonic()
    payload = executor.run("anything", top_n=2)
    wall = time.monotonic() - t0

    # Wall must be bounded by the collective deadline (plus slack).
    assert wall < 1.5, f"deadline expired in {wall:.2f}s; expected <1.5s"
    # Both sources are present (snippet-only fallback for the slow one).
    assert len(payload.sources) == 2
    fast = next(s for s in payload.sources if s.url.endswith("/fast"))
    slow = next(s for s in payload.sources if s.url.endswith("/slow"))
    assert fast.full_text is not None  # came back in time
    assert slow.full_text is None       # abandoned at deadline
    # Notes record the deadline expiry.
    assert any("jina_deadline:https://example.com/slow" in n for n in payload.notes)
    assert any("snippet_only:https://example.com/slow" in n for n in payload.notes)


def test_partial_results_when_one_fetch_completes_in_time():
    """Two URLs, deadline 0.5 s, one fetches in 0.1 s and one in 5 s.

    The fast result has full_text; the slow result is snippet-only.
    Mirrors the production case where one Jina page returns quickly
    and another would otherwise dominate the search-path latency.
    """
    fixture = [
        BraveResult(url="https://example.com/fast", title="Fast", snippet="fast", rank=0),
        BraveResult(url="https://example.com/slow", title="Slow", snippet="slow", rank=1),
    ]
    durations = {
        "https://example.com/fast": 0.1,
        "https://example.com/slow": 5.0,
    }
    executor = WebSearchExecutor(
        brave=_MockBrave(fixture=fixture),
        jina=_TimedJina(durations),
        llm=_StubLLM(),
        cache=None,
        max_fetch=2,
        collective_deadline_seconds=0.5,
    )
    payload = executor.run("anything", top_n=2)
    fast = next(s for s in payload.sources if s.url.endswith("/fast"))
    slow = next(s for s in payload.sources if s.url.endswith("/slow"))
    assert "Page" in (fast.full_text or "")
    assert slow.full_text is None


def test_jina_exception_in_one_fetch_doesnt_block_others():
    """A raising fetch must not break the parallel pool.

    Other fetches must still complete and the failed one degrades to
    snippet-only with a ``jina_error`` note.
    """
    fixture = [
        BraveResult(url="https://example.com/ok", title="OK", snippet="ok", rank=0),
        BraveResult(url="https://example.com/raise", title="Raise", snippet="raise", rank=1),
    ]

    class _MixedJina(JinaReaderClient):
        def __init__(self):
            self.endpoint = "mock://"
            self.timeout_s = 0.0
            self.max_bytes = 200_000

        def fetch(self, url):
            if "raise" in url:
                raise RuntimeError("simulated transient failure")
            return "# good page"

    executor = WebSearchExecutor(
        brave=_MockBrave(fixture=fixture),
        jina=_MixedJina(),
        llm=_StubLLM(),
        cache=None,
        max_fetch=2,
        collective_deadline_seconds=5.0,
    )
    payload = executor.run("anything", top_n=2)
    ok = next(s for s in payload.sources if s.url.endswith("/ok"))
    raised = next(s for s in payload.sources if s.url.endswith("/raise"))
    assert ok.full_text == "# good page"
    assert raised.full_text is None
    assert any("jina_error:https://example.com/raise" in n for n in payload.notes)


def test_zero_collective_deadline_disables_cap():
    """``collective_deadline_seconds=0`` waits indefinitely (per-fetch timeout only)."""
    fixture = [
        BraveResult(url="https://example.com/slowish", title="SlowIsh", snippet="x", rank=0),
    ]
    durations = {
        "https://example.com/slowish": 0.3,
    }
    executor = WebSearchExecutor(
        brave=_MockBrave(fixture=fixture),
        jina=_TimedJina(durations),
        llm=_StubLLM(),
        cache=None,
        max_fetch=1,
        collective_deadline_seconds=0.0,   # no cap
    )
    payload = executor.run("anything", top_n=1)
    assert len(payload.sources) == 1
    # Even though the page took 0.3 s and the deadline is 0, we waited
    # because 0 disables the cap.
    assert payload.sources[0].full_text is not None


def test_max_fetch_zero_skips_jina_entirely():
    """``max_fetch=0`` -- no parallel fetches are spawned; sources are snippet-only."""
    fixture = [
        BraveResult(url="https://example.com/a", title="A", snippet="a", rank=0),
    ]
    fetch_calls = []

    class _CountingJina(JinaReaderClient):
        def __init__(self):
            self.endpoint = "mock://"
            self.timeout_s = 0.0
            self.max_bytes = 200_000

        def fetch(self, url):
            fetch_calls.append(url)
            return "should not be called"

    executor = WebSearchExecutor(
        brave=_MockBrave(fixture=fixture),
        jina=_CountingJina(),
        llm=_StubLLM(),
        cache=None,
        max_fetch=0,
        collective_deadline_seconds=1.0,
    )
    payload = executor.run("anything", top_n=1)
    assert fetch_calls == []
    assert len(payload.sources) == 1
    assert payload.sources[0].full_text is None
