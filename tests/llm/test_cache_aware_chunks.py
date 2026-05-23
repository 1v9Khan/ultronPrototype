"""Tests for :mod:`ultron.llm.cache_aware_chunks`."""

from __future__ import annotations

import pytest

from ultron.llm.cache_aware_chunks import (
    CacheableChunk,
    ChunkedPrompt,
    DEFAULT_CHUNK_ORDER,
    DEFAULT_CHUNK_STABILITY,
    count_cacheable_chars,
    to_anthropic_messages,
    to_plain_messages,
)


def test_default_stability_marks_system_cacheable():
    assert DEFAULT_CHUNK_STABILITY["system"] is True
    assert DEFAULT_CHUNK_STABILITY["repo_map"] is True


def test_default_stability_marks_history_and_current_dynamic():
    assert DEFAULT_CHUNK_STABILITY["history"] is False
    assert DEFAULT_CHUNK_STABILITY["current"] is False


def test_chunked_prompt_convenience_helpers():
    p = ChunkedPrompt()
    p.add_system("You are an assistant.")
    p.add_repo_map("rendered map content")
    p.add_history_turn("user", "hi")
    p.add_current("today's question")
    assert len(p.system) == 1
    assert p.system[0].label == "system"
    assert p.repo_map[0].label == "repo_map"
    assert p.history[0].label == "history"
    assert p.current[0].label == "current"


def test_slot_lookup_raises_on_unknown_name():
    p = ChunkedPrompt()
    with pytest.raises(KeyError):
        p.slot("nonexistent")


def test_to_plain_messages_round_trip():
    p = ChunkedPrompt()
    p.add_system("S")
    p.add_history_turn("user", "Q1")
    p.add_history_turn("assistant", "A1")
    p.add_current("Q2")
    out = to_plain_messages(p)
    assert out == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]


def test_to_anthropic_messages_injects_cache_control_on_last_cacheable():
    p = ChunkedPrompt()
    p.add_system("S")
    p.add_repo_map("MAP")
    p.add_history_turn("user", "Q1")
    p.add_current("Q2")
    out = to_anthropic_messages(p)
    # Last cacheable block is repo_map (history + current are non-cacheable).
    # Should be the SECOND block (index 1 in S, MAP, Q1, Q2).
    cache_blocks = [
        idx for idx, msg in enumerate(out)
        if msg["content"][0].get("cache_control") is not None
    ]
    assert cache_blocks == [1], (
        f"expected cache_control on repo_map block only, got: "
        f"{[(i, out[i]) for i in cache_blocks]}"
    )
    # And the markers are the correct shape.
    assert out[1]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_to_anthropic_messages_no_cacheable_means_no_marker():
    p = ChunkedPrompt()
    p.add_current("Q")  # only the dynamic slot
    out = to_anthropic_messages(p)
    assert len(out) == 1
    assert "cache_control" not in out[0]["content"][0]


def test_to_anthropic_messages_only_one_marker_emitted():
    """We mark the LAST cacheable block; markers on earlier blocks
    are wasted metadata, so we don't emit them."""
    p = ChunkedPrompt()
    p.add_system("S")
    p.add_repo_map("M")
    # readonly_files slot also cacheable
    p.readonly_files.append(CacheableChunk(
        role="user", content="readonly", label="readonly_files",
    ))
    out = to_anthropic_messages(p)
    count = sum(
        1 for msg in out
        if msg["content"][0].get("cache_control") is not None
    )
    assert count == 1


def test_per_block_cacheable_override():
    """A per-block cacheable=True/False overrides slot stability."""
    p = ChunkedPrompt()
    p.history.append(CacheableChunk(
        role="user", content="cacheable history",
        label="history", cacheable=True,
    ))
    out = to_anthropic_messages(p)
    assert out[0]["content"][0].get("cache_control") is not None


def test_count_cacheable_chars():
    p = ChunkedPrompt()
    p.add_system("AAA")          # 3 chars, cacheable
    p.add_repo_map("BBBB")       # 4 chars, cacheable
    p.add_history_turn("user", "CCCCC")  # 5 chars, NOT cacheable
    p.add_current("DD")          # 2 chars, NOT cacheable
    assert count_cacheable_chars(p) == 7


def test_count_cacheable_chars_zero_when_nothing_cacheable():
    p = ChunkedPrompt()
    p.add_current("Q")
    assert count_cacheable_chars(p) == 0


def test_custom_slot_order_changes_serialisation():
    p = ChunkedPrompt()
    p.add_system("S")
    p.add_current("C")
    custom = ("current", "system")
    out = to_plain_messages(p, slot_order=custom)
    # current first now.
    assert out[0]["content"] == "C"
    assert out[1]["content"] == "S"


def test_custom_stability_disables_caching():
    p = ChunkedPrompt()
    p.add_system("S")
    # Force system non-cacheable.
    custom = {**DEFAULT_CHUNK_STABILITY, "system": False}
    out = to_anthropic_messages(p, stability=custom)
    assert out[0]["content"][0].get("cache_control") is None


def test_cacheable_chunk_is_frozen():
    c = CacheableChunk(role="system", content="x", label="system")
    with pytest.raises(Exception):
        c.content = "y"  # type: ignore[misc]
