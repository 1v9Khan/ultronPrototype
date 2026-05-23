"""Tests for :mod:`ultron.coding.commit_message`."""

from __future__ import annotations

import pytest

from ultron.coding.commit_message import (
    CommitMessageRequest,
    CommitMessageResult,
    DEFAULT_COMMIT_SYSTEM_PROMPT,
    generate_commit_message,
    strip_outer_quotes,
)


def test_strip_outer_double_quotes():
    assert strip_outer_quotes('"hello world"') == "hello world"


def test_strip_outer_single_quotes():
    assert strip_outer_quotes("'hello world'") == "hello world"


def test_strip_smart_quotes():
    assert strip_outer_quotes("“hello”") == "hello"
    assert strip_outer_quotes("‘hello’") == "hello"


def test_strip_triple_backticks():
    assert strip_outer_quotes("```hello world```") == "hello world"


def test_strip_triple_backticks_with_lang_tag():
    assert strip_outer_quotes("```text\nhello world\n```") == "hello world"


def test_strip_preserves_inner_quotes():
    # Don't crack apart "fix don't crash" by mistake.
    assert strip_outer_quotes("fix don't crash") == "fix don't crash"


def test_strip_empty_string():
    assert strip_outer_quotes("") == ""


def test_generate_commit_message_empty_diff_returns_error():
    result = generate_commit_message(
        CommitMessageRequest(diff_text=""),
        [lambda p: "anything"],
    )
    assert result.message is None
    assert result.error == "empty diff"


def test_generate_commit_message_empty_cascade_returns_error():
    result = generate_commit_message(
        CommitMessageRequest(diff_text="+ x\n- y"),
        [],
    )
    assert result.message is None
    assert result.error == "empty cascade"


def test_generate_commit_message_first_llm_wins():
    calls = {"primary": 0, "fallback": 0}

    def primary(prompt: str) -> str:
        calls["primary"] += 1
        return "add user auth"

    def fallback(prompt: str) -> str:
        calls["fallback"] += 1
        return "should not be called"

    result = generate_commit_message(
        CommitMessageRequest(diff_text="+++ user.py", user_context="add login"),
        [primary, fallback],
    )
    assert result.message == "add user auth"
    assert result.cascade_index == 0
    assert calls["primary"] == 1
    assert calls["fallback"] == 0


def test_generate_commit_message_falls_through_on_exception():
    def broken(_):
        raise RuntimeError("LLM down")

    def working(_):
        return "fix authentication bug"

    result = generate_commit_message(
        CommitMessageRequest(diff_text="+++ auth.py"),
        [broken, working],
    )
    assert result.message == "fix authentication bug"
    assert result.cascade_index == 1


def test_generate_commit_message_falls_through_on_empty():
    def empty(_):
        return ""

    def working(_):
        return "improve performance"

    result = generate_commit_message(
        CommitMessageRequest(diff_text="+++ perf.py"),
        [empty, working],
    )
    assert result.message == "improve performance"
    assert result.cascade_index == 1


def test_generate_commit_message_all_fail():
    def broken(_):
        raise RuntimeError("down")

    def empty(_):
        return ""

    result = generate_commit_message(
        CommitMessageRequest(diff_text="+++ x"),
        [broken, empty],
    )
    assert result.message is None
    assert "all LLMs failed" in result.error
    assert result.last_exception is not None


def test_generate_commit_message_strips_quotes_by_default():
    result = generate_commit_message(
        CommitMessageRequest(diff_text="+++ x"),
        [lambda _: '"add feature X"'],
    )
    assert result.message == "add feature X"


def test_generate_commit_message_can_disable_strip():
    result = generate_commit_message(
        CommitMessageRequest(diff_text="+++ x"),
        [lambda _: '"keep quotes"'],
        strip_quotes=False,
    )
    assert result.message == '"keep quotes"'


def test_generate_commit_message_respects_budget():
    """A small budget should skip an LLM whose prompt exceeds it."""
    huge_diff = "+ x" * 50_000  # ~150k chars

    def small_model(_):
        return "should be skipped"

    def big_model(_):
        return "big model wins"

    result = generate_commit_message(
        CommitMessageRequest(
            diff_text=huge_diff,
            max_prompt_chars_per_llm=(1000, 1_000_000),
        ),
        [small_model, big_model],
    )
    assert result.message == "big model wins"
    assert result.cascade_index == 1


def test_generate_commit_message_all_exceed_budget():
    huge_diff = "+ x" * 50_000

    def small_a(_):
        return "skipped"

    def small_b(_):
        return "skipped"

    result = generate_commit_message(
        CommitMessageRequest(
            diff_text=huge_diff,
            max_prompt_chars_per_llm=(1000, 2000),
        ),
        [small_a, small_b],
    )
    assert result.message is None
    assert "too large" in result.error


def test_generate_commit_message_uses_system_prompt():
    captured_prompts = []

    def capture(p):
        captured_prompts.append(p)
        return "result"

    custom_prompt = "Custom system prompt."
    generate_commit_message(
        CommitMessageRequest(diff_text="+ x", system_prompt=custom_prompt),
        [capture],
    )
    assert captured_prompts
    assert custom_prompt in captured_prompts[0]


def test_default_system_prompt_has_no_brand_attribution():
    """The default prompt explicitly forbids AI-author trailers."""
    assert "Co-Authored-By" in DEFAULT_COMMIT_SYSTEM_PROMPT
    assert "AI assistant" in DEFAULT_COMMIT_SYSTEM_PROMPT


def test_commit_message_result_is_frozen():
    r = CommitMessageResult(message="x")
    with pytest.raises(Exception):
        r.message = "y"  # type: ignore[misc]
