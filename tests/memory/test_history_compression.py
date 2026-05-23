"""Tests for :mod:`ultron.memory.history_compression`."""

from __future__ import annotations

import threading

import pytest

from ultron.memory.history_compression import (
    DEFAULT_SUMMARY_PREAMBLE,
    CompressionResult,
    compress_history,
    compress_history_recursive,
    compress_history_with_guard,
    find_split_point,
    messages_to_dicts,
    messages_to_text,
)
from ultron.utils.snapshot_guard import SnapshotGuard
from ultron.utils.token_budget import char_count_tokens


# ---------------------------------------------------------------------------
# find_split_point
# ---------------------------------------------------------------------------


def _msgs(*pairs):
    """Tiny constructor: _msgs(("user", "hi"), ("assistant", "hello")) -> list."""
    return [{"role": r, "content": c} for r, c in pairs]


def test_find_split_empty_input():
    assert find_split_point([], 100, char_count_tokens) == 0


def test_find_split_single_message():
    msgs = _msgs(("user", "hi"))
    assert find_split_point(msgs, 100, char_count_tokens) == 0


def test_find_split_returns_zero_when_all_fits():
    msgs = _msgs(("user", "short"), ("assistant", "ok"))
    # Budget is huge so everything fits in the tail; no head to summarise.
    assert find_split_point(msgs, 10_000, char_count_tokens) == 0


def test_find_split_returns_assistant_boundary():
    msgs = _msgs(
        ("user", "u1" * 100),
        ("assistant", "a1" * 100),
        ("user", "u2" * 100),
        ("assistant", "a2" * 100),
        ("user", "u3" * 50),
        ("assistant", "a3" * 50),
    )
    # Force a tail half budget that will split somewhere in the middle.
    split = find_split_point(msgs, 80, char_count_tokens)
    assert split > 0
    # The message just before split must be an assistant.
    assert msgs[split - 1]["role"] == "assistant"


def test_find_split_without_boundary_pref():
    msgs = _msgs(
        ("user", "u1" * 100),
        ("user", "u2" * 100),
        ("assistant", "a1" * 100),
        ("user", "u3" * 100),
    )
    split = find_split_point(
        msgs, 80, char_count_tokens, prefer_assistant_boundary=False
    )
    assert split > 0


# ---------------------------------------------------------------------------
# compress_history (single pass)
# ---------------------------------------------------------------------------


def test_compress_history_empty_returns_empty():
    result = compress_history([], lambda _: "x", max_tokens=100, token_counter=char_count_tokens)
    assert result.compressed == []
    assert result.head_summarised == 0


def test_compress_history_skip_when_fits():
    msgs = _msgs(("user", "hi"), ("assistant", "hello"))
    captured = {"calls": 0}

    def summarize(_):
        captured["calls"] += 1
        return "summary"

    result = compress_history(
        msgs, summarize, max_tokens=10_000, token_counter=char_count_tokens
    )
    # No summarisation should fire when everything already fits.
    assert captured["calls"] == 0
    assert result.compressed == list(msgs)


def test_compress_history_summarises_head():
    msgs = _msgs(
        ("user", "alpha" * 200),
        ("assistant", "ack alpha" * 200),
        ("user", "beta" * 50),
        ("assistant", "ack beta"),
    )

    def summarize(blob: str) -> str:
        return f"<SUMMARY:{len(blob)}>"

    result = compress_history(
        msgs, summarize, max_tokens=100, token_counter=char_count_tokens
    )
    assert result.compressed is not None
    assert result.head_summarised > 0
    # First message in compressed list is the summary.
    assert result.compressed[0]["role"] == "user"
    assert "<SUMMARY:" in result.compressed[0]["content"]
    assert DEFAULT_SUMMARY_PREAMBLE.strip() in result.compressed[0]["content"]


def test_compress_history_summarise_fn_failure_returns_none():
    msgs = _msgs(
        ("user", "long" * 500),
        ("assistant", "ack" * 500),
        ("user", "current"),
    )

    def summarize(_):
        raise RuntimeError("LLM down")

    result = compress_history(
        msgs, summarize, max_tokens=50, token_counter=char_count_tokens
    )
    assert result.compressed is None
    assert "LLM down" in result.error


def test_compress_history_empty_summary_returns_none():
    msgs = _msgs(
        ("user", "long" * 500),
        ("assistant", "ack" * 500),
        ("user", "current"),
    )

    def summarize(_):
        return ""

    result = compress_history(
        msgs, summarize, max_tokens=50, token_counter=char_count_tokens
    )
    assert result.compressed is None
    assert "empty" in result.error.lower()


# ---------------------------------------------------------------------------
# compress_history_recursive
# ---------------------------------------------------------------------------


def test_compress_history_recursive_returns_when_fits():
    msgs = _msgs(("user", "ok"))

    def summarize(_):
        return "should not be called"

    result = compress_history_recursive(
        msgs, summarize, max_tokens=10_000, token_counter=char_count_tokens
    )
    assert result.depth == 0


def test_compress_history_recursive_iterates_when_overflow():
    msgs = _msgs(
        ("user", "x" * 1000),
        ("assistant", "y" * 1000),
        ("user", "z" * 1000),
        ("assistant", "w" * 1000),
        ("user", "v" * 1000),
        ("assistant", "u" * 1000),
        ("user", "t" * 1000),
        ("assistant", "s" * 1000),
    )
    call_log = {"texts": []}

    def summarize(text: str) -> str:
        call_log["texts"].append(text)
        # Return a summary about as long as the input divided by 10.
        return "summary " * max(1, len(text) // 100)

    result = compress_history_recursive(
        msgs, summarize, max_tokens=400, token_counter=char_count_tokens, max_depth=3
    )
    # Either we converged to a fitting result or we hit max_depth and fell
    # back to summarising everything.
    assert result.compressed is not None
    assert call_log["texts"]
    # depth >= 0; if iterating did happen, depth > 0.
    assert result.depth >= 0


def test_compress_history_recursive_deep_fallback_uses_everything():
    """At max_depth the algorithm summarises EVERYTHING in one go."""
    msgs = _msgs(
        ("user", "alpha" * 5000),
        ("assistant", "beta" * 5000),
        ("user", "gamma" * 5000),
        ("assistant", "delta" * 5000),
    )
    counts = {"calls": 0}

    def stubborn_summarize(text: str) -> str:
        counts["calls"] += 1
        # Return a summary that's still longer than the budget.
        return "VERY LONG SUMMARY " * 1000

    result = compress_history_recursive(
        msgs,
        stubborn_summarize,
        max_tokens=50,
        token_counter=char_count_tokens,
        max_depth=2,
    )
    assert result.compressed is not None
    assert result.depth == 2
    # The deep fallback should have been called at least once.
    assert counts["calls"] >= 1


# ---------------------------------------------------------------------------
# compress_history_with_guard (race protection)
# ---------------------------------------------------------------------------


def test_compress_with_guard_no_mutation_succeeds():
    msgs = _msgs(
        ("user", "x" * 500),
        ("assistant", "y" * 500),
        ("user", "tail"),
        ("assistant", "ok"),
    )

    def summarize(_):
        return "summary"

    guard = SnapshotGuard()
    result = compress_history_with_guard(
        msgs,
        summarize,
        max_tokens=100,
        token_counter=char_count_tokens,
        guard=guard,
    )
    assert result.race_detected is False
    assert result.compressed is not None


def test_compress_with_guard_detects_mutation():
    """The race is real: mutate the live list during summarize_fn and
    the result must be discarded."""
    msgs = _msgs(
        ("user", "x" * 500),
        ("assistant", "y" * 500),
        ("user", "tail"),
        ("assistant", "ok"),
    )

    def mutating_summarize(_):
        msgs.append({"role": "user", "content": "intruder"})
        return "summary"

    result = compress_history_with_guard(
        msgs,
        mutating_summarize,
        max_tokens=100,
        token_counter=char_count_tokens,
    )
    assert result.race_detected is True
    assert result.compressed is None


def test_compress_with_guard_internal_snapshot_when_no_guard_supplied():
    msgs = _msgs(
        ("user", "x" * 500),
        ("assistant", "y" * 500),
        ("user", "tail"),
        ("assistant", "ok"),
    )

    def summarize(_):
        return "summary"

    # No guard arg -> internal one-shot snapshot.
    result = compress_history_with_guard(
        msgs,
        summarize,
        max_tokens=100,
        token_counter=char_count_tokens,
    )
    assert result.race_detected is False
    assert result.compressed is not None


def test_compress_with_guard_threaded_mutation():
    """Race during a background-thread compression."""
    msgs = _msgs(
        ("user", "x" * 500),
        ("assistant", "y" * 500),
        ("user", "tail"),
        ("assistant", "ok"),
    )

    started = threading.Event()
    proceed = threading.Event()

    def slow_summarize(_):
        started.set()
        proceed.wait(timeout=2)
        return "summary"

    result_container = {}

    def worker():
        result_container["r"] = compress_history_with_guard(
            msgs,
            slow_summarize,
            max_tokens=100,
            token_counter=char_count_tokens,
        )

    t = threading.Thread(target=worker)
    t.start()
    started.wait(timeout=2)
    # Foreground mutates the live list mid-compression.
    msgs.append({"role": "user", "content": "new turn"})
    proceed.set()
    t.join(timeout=3)
    assert "r" in result_container
    assert result_container["r"].race_detected is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_messages_to_text_format():
    msgs = _msgs(("user", "hi"), ("assistant", "hello"))
    text = messages_to_text(msgs)
    assert "# USER" in text
    assert "# ASSISTANT" in text
    assert "hi" in text
    assert "hello" in text


def test_messages_to_dicts_normalises_dataclass_like():
    class Dummy:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    msgs = [Dummy("user", "hi"), Dummy("assistant", "hello")]
    out = messages_to_dicts(msgs)
    assert out == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_compression_result_is_frozen():
    r = CompressionResult(
        compressed=None,
        original_tokens=0,
        final_tokens=0,
        head_summarised=0,
        tail_preserved=0,
    )
    with pytest.raises(Exception):
        r.compressed = []  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BackgroundSummarizer integration
# ---------------------------------------------------------------------------


def test_background_summarizer_compress_history_method(tmp_path):
    from ultron.memory.background_summarizer import BackgroundSummarizer

    def turns_fn():
        return []

    summarize_calls = {"count": 0}

    def fact_extract_fn(_):
        return "{}"  # legitimate JSON for the fact-extract path

    def free_form_summarize(blob: str) -> str:
        summarize_calls["count"] += 1
        return f"<summary of {len(blob)} chars>"

    bs = BackgroundSummarizer(
        generate_fn=fact_extract_fn,
        recent_turns_fn=turns_fn,
        compress_summarize_fn=free_form_summarize,
    )

    msgs = _msgs(
        ("user", "long" * 500),
        ("assistant", "ack" * 500),
        ("user", "fresh"),
        ("assistant", "ack"),
    )
    result = bs.compress_history_for_llm(
        msgs, max_tokens=100, token_counter=char_count_tokens
    )
    assert result.compressed is not None
    assert summarize_calls["count"] >= 1


def test_background_summarizer_compress_falls_back_to_generate_fn():
    """When no separate compress_summarize_fn is wired, generate_fn is reused."""
    from ultron.memory.background_summarizer import BackgroundSummarizer

    def turns_fn():
        return []

    calls = {"count": 0}

    def generate(_):
        calls["count"] += 1
        return "summary"

    bs = BackgroundSummarizer(
        generate_fn=generate,
        recent_turns_fn=turns_fn,
    )
    msgs = _msgs(
        ("user", "long" * 500),
        ("assistant", "ack" * 500),
        ("user", "fresh"),
        ("assistant", "ack"),
    )
    result = bs.compress_history_for_llm(
        msgs, max_tokens=100, token_counter=char_count_tokens
    )
    assert result.compressed is not None
    assert calls["count"] >= 1
