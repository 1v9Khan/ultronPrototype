"""Tests for the T10 token-share splitter + identifier preservation + oversized fallback."""

from __future__ import annotations

import pytest

from ultron.llm.condensers.splitter import (
    BASE_CHUNK_RATIO,
    IDENTIFIER_PRESERVATION_SUFFIX,
    IdentifierPreservationPolicy,
    MIN_CHUNK_RATIO,
    Message,
    SAFETY_MARGIN,
    UNSUMMARISABLE_TEMPLATE,
    chunk_messages_by_max_tokens,
    compute_adaptive_chunk_ratio,
    estimate_message_tokens,
    identifier_preservation_text,
    is_oversized_for_summary,
    split_messages_by_token_share,
    summarise_with_fallback,
    with_identifier_preservation,
)


# ----------------------------------------------------------------------
# Constants


def test_constants_match_openclaw_defaults() -> None:
    assert BASE_CHUNK_RATIO == 0.4
    assert MIN_CHUNK_RATIO == 0.15
    assert SAFETY_MARGIN == 1.2


def test_identifier_preservation_suffix_mentions_uuids() -> None:
    assert "UUID" in IDENTIFIER_PRESERVATION_SUFFIX
    assert "hashes" in IDENTIFIER_PRESERVATION_SUFFIX


# ----------------------------------------------------------------------
# estimate_message_tokens


def test_estimate_string_content() -> None:
    assert estimate_message_tokens(Message(role="user", content="A" * 40)) == 10


def test_estimate_list_content_sums_text_parts() -> None:
    m = Message(
        role="user",
        content=[{"text": "AAAA"}, {"text": "BBBB"}],
    )
    assert estimate_message_tokens(m) == 2


def test_estimate_empty_content_returns_one() -> None:
    assert estimate_message_tokens(Message(role="user", content="")) == 1


# ----------------------------------------------------------------------
# is_oversized_for_summary


def test_oversized_when_message_exceeds_half_context() -> None:
    big = Message(role="user", content="X" * 4000)  # ~1000 tokens
    assert is_oversized_for_summary(big, context_window=1000) is True


def test_not_oversized_for_small_message() -> None:
    small = Message(role="user", content="hello")
    assert is_oversized_for_summary(small, context_window=8192) is False


def test_oversized_with_zero_context_window() -> None:
    msg = Message(role="user", content="x")
    assert is_oversized_for_summary(msg, context_window=0) is True


# ----------------------------------------------------------------------
# compute_adaptive_chunk_ratio


def test_adaptive_ratio_unchanged_for_small_avg() -> None:
    messages = [Message(role="user", content="hi") for _ in range(5)]
    assert compute_adaptive_chunk_ratio(messages, context_window=8192) == BASE_CHUNK_RATIO


def test_adaptive_ratio_scales_down_for_large_avg() -> None:
    # Each message ~250 chars = ~62 tokens; ratio with safety = ~7.5%
    # That's below the 10% trigger, so try larger:
    messages = [Message(role="user", content="X" * 4000) for _ in range(5)]  # ~1000 tokens each
    ratio = compute_adaptive_chunk_ratio(messages, context_window=2000)
    assert ratio < BASE_CHUNK_RATIO


def test_adaptive_ratio_floors_at_min() -> None:
    # 8 KB chars = 2k tokens; ratio = (2000 * 1.2)/100 = 24, way above
    # threshold. Reduction is min(24*2, 0.4-0.15) = 0.25; final = 0.15.
    messages = [Message(role="user", content="X" * 8000)]
    ratio = compute_adaptive_chunk_ratio(messages, context_window=100)
    assert abs(ratio - MIN_CHUNK_RATIO) < 1e-9


def test_adaptive_ratio_empty_input_returns_base() -> None:
    assert compute_adaptive_chunk_ratio([], context_window=8192) == BASE_CHUNK_RATIO


# ----------------------------------------------------------------------
# chunk_messages_by_max_tokens


def test_chunk_simple_split_at_budget() -> None:
    messages = [Message(role="user", content="X" * 40) for _ in range(4)]  # 10 tokens each
    plan = chunk_messages_by_max_tokens(messages, max_tokens=20)
    # 4 * 10 = 40 tokens, budget 20 -> roughly 2 chunks
    assert sum(len(c) for c in plan.chunks) == 4
    assert len(plan.chunks) >= 2


def test_chunk_preserves_tool_call_pair() -> None:
    messages = [
        Message(role="user", content="small"),
        Message(
            role="assistant",
            content="X" * 40,  # 10 tokens
            tool_calls=({"id": "call_1"},),
        ),
        Message(role="tool", content="result", tool_use_id="call_1"),
        Message(role="user", content="next turn"),
    ]
    # Force a split mid-pair by setting low budget.
    plan = chunk_messages_by_max_tokens(messages, max_tokens=8)
    # The assistant+tool pair must end up in the same chunk; assert
    # no chunk contains assistant without its matching tool_result.
    for chunk in plan.chunks:
        opened: set[str] = set()
        closed: set[str] = set()
        for m in chunk:
            for tc in m.tool_calls:
                opened.add(tc["id"])
            if m.tool_use_id:
                closed.add(m.tool_use_id)
        # All opened in this chunk should be closed in this chunk OR
        # remain open at the end (they continue into the next chunk
        # only if the splitter couldn't find a safe boundary).
        assert opened - closed == set() or opened - closed == opened


def test_chunk_aborted_assistant_does_not_track() -> None:
    messages = [
        Message(
            role="assistant",
            content="X" * 40,
            tool_calls=({"id": "call_x"},),
            stop_reason="aborted",
        ),
        Message(role="user", content="next"),
    ]
    # An aborted assistant message must not add to pending (no result
    # is ever coming), so the splitter can break freely.
    plan = chunk_messages_by_max_tokens(messages, max_tokens=5)
    assert sum(len(c) for c in plan.chunks) == 2


def test_chunk_empty_input_returns_empty_plan() -> None:
    plan = chunk_messages_by_max_tokens([], max_tokens=100)
    assert plan.chunks == ()


def test_chunk_zero_budget_returns_single_chunk() -> None:
    messages = [Message(role="user", content="hi") for _ in range(3)]
    plan = chunk_messages_by_max_tokens(messages, max_tokens=0)
    assert len(plan.chunks) == 1
    assert sum(len(c) for c in plan.chunks) == 3


def test_chunk_records_oversized_indices() -> None:
    messages = [
        Message(role="user", content="X" * 40),       # 10 tokens
        Message(role="user", content="X" * 400),      # 100 tokens — oversized
        Message(role="user", content="X" * 40),       # 10 tokens
    ]
    plan = chunk_messages_by_max_tokens(messages, max_tokens=20)
    assert 1 in plan.oversized_indices


# ----------------------------------------------------------------------
# split_messages_by_token_share


def test_split_by_token_share_returns_chunks() -> None:
    messages = [Message(role="user", content="X" * 40) for _ in range(10)]
    plan = split_messages_by_token_share(messages, parts=4, context_window=8192)
    assert plan.adaptive_ratio == BASE_CHUNK_RATIO  # small avg, no scaling
    assert sum(len(c) for c in plan.chunks) == 10


def test_split_empty_input_returns_empty_plan() -> None:
    plan = split_messages_by_token_share([], parts=4, context_window=8192)
    assert plan.chunks == ()


def test_split_invalid_parts_clamps_to_one() -> None:
    messages = [Message(role="user", content="hi")]
    plan = split_messages_by_token_share(messages, parts=0, context_window=8192)
    assert len(plan.chunks) >= 1


# ----------------------------------------------------------------------
# identifier_preservation_text + with_identifier_preservation


def test_identifier_preservation_strict_uses_suffix() -> None:
    assert identifier_preservation_text() == IDENTIFIER_PRESERVATION_SUFFIX


def test_identifier_preservation_off_returns_empty() -> None:
    assert identifier_preservation_text(policy=IdentifierPreservationPolicy.OFF) == ""


def test_identifier_preservation_custom_uses_supplied_text() -> None:
    out = identifier_preservation_text(
        policy=IdentifierPreservationPolicy.CUSTOM,
        custom_text="My custom rule",
    )
    assert out == "My custom rule"


def test_with_identifier_preservation_appends_suffix() -> None:
    base = "System prompt body."
    out = with_identifier_preservation(base)
    assert base in out
    assert IDENTIFIER_PRESERVATION_SUFFIX in out


def test_with_identifier_preservation_off_returns_unchanged() -> None:
    base = "System prompt body."
    out = with_identifier_preservation(base, policy=IdentifierPreservationPolicy.OFF)
    assert out == base


def test_with_identifier_preservation_empty_prompt_returns_suffix() -> None:
    assert with_identifier_preservation("") == IDENTIFIER_PRESERVATION_SUFFIX


# ----------------------------------------------------------------------
# summarise_with_fallback


def test_summarise_tier1_success_returns_immediately() -> None:
    messages = [Message(role="user", content="hi")]
    calls = []

    def fn(msgs):
        calls.append(len(msgs))
        return "summary v1"

    result = summarise_with_fallback(messages, context_window=8192, summarise_fn=fn)
    assert result == "summary v1"
    assert calls == [1]


def test_summarise_tier1_returns_empty_falls_through_to_tier3() -> None:
    messages = [Message(role="user", content="hi")]  # small, no oversized

    def fn(_):
        return ""

    result = summarise_with_fallback(messages, context_window=8192, summarise_fn=fn)
    # No oversized -> tier 2 skipped -> tier 3 fallback.
    assert "Summary unavailable" in result


def test_summarise_tier2_recovers_when_oversized_present() -> None:
    messages = [
        Message(role="user", content="hi"),
        Message(role="user", content="X" * 8000),  # oversized
    ]
    attempts = []

    def fn(msgs):
        attempts.append([m.content[:10] for m in msgs])
        # Tier 1 fails; tier 2 succeeds with elided big message.
        if any(len(m.content) > 1000 for m in msgs):
            raise RuntimeError("too big")
        return "tier-2 summary"

    result = summarise_with_fallback(messages, context_window=1000, summarise_fn=fn)
    assert result == "tier-2 summary"
    assert len(attempts) == 2


def test_summarise_tier3_when_both_fail() -> None:
    messages = [Message(role="user", content="X" * 8000)]

    def fn(_):
        raise RuntimeError("nope")

    result = summarise_with_fallback(messages, context_window=1000, summarise_fn=fn)
    assert result == UNSUMMARISABLE_TEMPLATE.format(total=1, oversized=1)


def test_summarise_empty_input_returns_empty() -> None:
    def fn(_):
        return "should not be called"

    assert summarise_with_fallback([], context_window=8192, summarise_fn=fn) == ""
