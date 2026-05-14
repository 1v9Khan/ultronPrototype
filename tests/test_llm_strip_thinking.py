"""Tests for :func:`ultron.llm.inference.strip_thinking_text`.

The blocking ``LLMEngine.generate`` path now applies this filter before
returning so callers that take the whole response in one shot (the
screen-context handler, the disambiguator's JSON pass, etc.) can't leak
``<think>...</think>`` chains into TTS. The streaming path goes through
``_strip_thinking_blocks`` which handles tags split across token
boundaries; this filter is the materialised-string analogue.
"""

from __future__ import annotations

from ultron.llm.inference import strip_thinking_text


def test_passes_through_clean_text():
    assert strip_thinking_text("just an answer") == "just an answer"
    assert strip_thinking_text("") == ""


def test_strips_single_block_at_start():
    raw = (
        "<think>\nOkay, the user asked X. I'll answer Y.\n</think>\n\n"
        "The actual answer."
    )
    assert strip_thinking_text(raw) == "The actual answer."


def test_strips_block_with_surrounding_text():
    raw = "Hello <think>internal</think> world"
    assert strip_thinking_text(raw) == "Hello  world".strip()


def test_strips_multiple_blocks():
    raw = (
        "<think>first thought</think>"
        "first part. "
        "<think>second thought</think>"
        "second part."
    )
    # Trailing space after "first part." is preserved; the inner block
    # collapses to nothing without adding a separator.
    assert strip_thinking_text(raw) == "first part. second part."


def test_unterminated_block_drops_tail():
    """If a <think> is never closed (model cancelled / truncated), drop
    everything from the opening tag onward rather than risk leaking a
    chain-of-thought."""
    raw = (
        "Visible answer.\n"
        "<think>This is a chain of thought that never closes because the "
        "stream got cut."
    )
    assert strip_thinking_text(raw) == "Visible answer."


def test_handles_multiline_block():
    raw = (
        "<think>\nFirst, identify X.\n"
        "Then, weigh Y vs Z.\n"
        "Conclude with W.\n</think>\n\n"
        "The answer is W."
    )
    assert strip_thinking_text(raw) == "The answer is W."


def test_real_session_screen_context_block():
    """Reproduces the 2026-05-13 session log shape that leaked to TTS."""
    raw = (
        "<think>\n"
        "Okay, the user is asking, \"What's on my screen?\" I need to "
        "answer based on the visual context provided.\n"
        "</think>\n\n"
        "YouTube - Google Chrome. Extensions like Dark Reader and "
        "uBlock Origin are active."
    )
    out = strip_thinking_text(raw)
    assert "<think>" not in out
    assert "</think>" not in out
    assert "Okay, the user is asking" not in out
    assert out.startswith("YouTube - Google Chrome")


def test_strip_is_idempotent():
    raw = "<think>a</think>b"
    out1 = strip_thinking_text(raw)
    out2 = strip_thinking_text(out1)
    assert out1 == out2 == "b"


def test_short_input_passes_through():
    """The fast-path "if '<think>' not in text" branch."""
    assert strip_thinking_text("a") == "a"
    assert strip_thinking_text("hello") == "hello"
