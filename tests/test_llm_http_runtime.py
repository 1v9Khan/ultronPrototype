"""Tests for LLMEngine HTTP-client runtime.

The in-process runtime is exercised by the existing test suite (and
by ``scripts/measure_baseline.py``). These tests cover the new
``runtime="http_server"`` branch — construction without a local
model load, request-shape correctness for blocking calls, and SSE
streaming with cancel.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ultron.llm.inference import LLMEngine


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_http_runtime_constructs_without_loading_local_model(tmp_path):
    """Constructing with runtime='http_server' must NOT touch llama_cpp.Llama
    or look for a GGUF on disk. The server is what holds the weights."""
    engine = LLMEngine(
        system_prompt="test",
        history_turns=2,
        runtime="http_server",
    )
    assert engine._runtime == "http_server"
    assert engine._llm is None
    assert engine.model_path is None
    assert engine._http_base_url.endswith("/v1") or engine._http_base_url.endswith(":8765")
    assert engine._http_model_alias == "qwen3.5-9b-local"


def test_unknown_runtime_raises_value_error():
    with pytest.raises(ValueError, match="unknown llm.runtime"):
        LLMEngine(runtime="bogus", system_prompt="x")


# ---------------------------------------------------------------------------
# Blocking generate() against the HTTP endpoint
# ---------------------------------------------------------------------------


def _ok_chat_response(content: str) -> dict:
    return {
        "id": "test",
        "object": "chat.completion",
        "model": "qwen3.5-9b-local",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_http_generate_returns_completion_text():
    engine = LLMEngine(
        system_prompt="test sys",
        history_turns=2,
        runtime="http_server",
    )

    posted_payload = {}

    def fake_post(url, headers, json, timeout):  # noqa: A002 — mirror requests sig
        posted_payload["url"] = url
        posted_payload["headers"] = headers
        posted_payload["body"] = json
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = _ok_chat_response("hello")
        return resp

    with patch("requests.post", side_effect=fake_post):
        text = engine.generate("hi")

    assert text == "hello"
    # Request shape verification
    assert posted_payload["url"].endswith("/chat/completions")
    assert posted_payload["headers"]["Authorization"].startswith("Bearer ")
    assert posted_payload["body"]["model"] == "qwen3.5-9b-local"
    assert posted_payload["body"]["stream"] is False
    msgs = posted_payload["body"]["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "test sys"
    assert msgs[-1] == {"role": "user", "content": "hi"}


# ---------------------------------------------------------------------------
# Streaming generate_stream() against SSE endpoint
# ---------------------------------------------------------------------------


def _sse_lines(deltas: list[str]) -> list[str]:
    """Build a list of SSE lines that look like llama-cpp-server's stream."""
    lines = []
    for d in deltas:
        chunk = {
            "id": "x", "object": "chat.completion.chunk", "model": "qwen3.5-9b-local",
            "choices": [{
                "index": 0,
                "delta": {"content": d} if d else {"role": "assistant"},
                "finish_reason": None,
            }],
        }
        lines.append(f"data: {json.dumps(chunk)}")
        lines.append("")  # blank separator between SSE events
    # Final chunk + DONE marker
    final = {
        "id": "x", "object": "chat.completion.chunk", "model": "qwen3.5-9b-local",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    lines.append(f"data: {json.dumps(final)}")
    lines.append("data: [DONE]")
    return lines


class _FakeStreamingResponse:
    """Minimal stand-in for ``requests.Response`` in stream=True mode."""

    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=True):
        for line in self._lines:
            yield line


def test_http_generate_stream_yields_visible_deltas():
    engine = LLMEngine(
        system_prompt="test sys",
        history_turns=2,
        runtime="http_server",
    )

    sse_lines = _sse_lines(["", "Hello, ", "world", "!"])

    def fake_post(url, headers, json, timeout, stream):
        assert stream is True
        return _FakeStreamingResponse(sse_lines)

    with patch("requests.post", side_effect=fake_post):
        out = "".join(engine.generate_stream("greet me"))

    assert out == "Hello, world!"


def test_http_generate_stream_strips_thinking_blocks():
    """Same chain-of-thought stripping should apply to HTTP streams."""
    engine = LLMEngine(
        system_prompt="x",
        history_turns=2,
        runtime="http_server",
    )

    deltas = ["", "<think>", "internal reasoning", "</think>", "visible answer"]
    sse_lines = _sse_lines(deltas)

    def fake_post(url, headers, json, timeout, stream):
        return _FakeStreamingResponse(sse_lines)

    with patch("requests.post", side_effect=fake_post):
        out = "".join(engine.generate_stream("q"))

    assert out == "visible answer"
    assert "internal reasoning" not in out


def test_http_generate_stream_cancel_stops_iteration():
    engine = LLMEngine(
        system_prompt="x",
        history_turns=2,
        runtime="http_server",
    )

    sse_lines = _sse_lines(["", "first", "second", "third"])

    def fake_post(url, headers, json, timeout, stream):
        return _FakeStreamingResponse(sse_lines)

    with patch("requests.post", side_effect=fake_post):
        gen = engine.generate_stream("q")
        first = next(gen)
        engine.cancel()
        # After cancel, iterator should drain quickly.
        rest = "".join(gen)

    # The chain-of-thought stripper holds an 8-char tail buffer so partial
    # <think> tags can't slip through; the first yield is therefore a
    # prefix of the data we sent in. Just confirm we got SOMETHING.
    assert first
    # Cancel should land before "third" makes it through.
    assert "third" not in rest


def test_http_generate_skips_non_data_sse_lines():
    """SSE stream may include comment lines / heartbeats / empty lines.
    The reader should ignore them."""
    engine = LLMEngine(
        system_prompt="x",
        history_turns=2,
        runtime="http_server",
    )

    lines = [
        ":heartbeat",  # SSE comment
        "",
        "data: {malformed json",  # bad chunk
        "",
        f"data: {json.dumps({'choices': [{'delta': {'content': 'ok'}}]})}",
        "",
        "data: [DONE]",
    ]

    def fake_post(url, headers, json, timeout, stream):  # noqa: A002
        return _FakeStreamingResponse(lines)

    with patch("requests.post", side_effect=fake_post):
        out = "".join(engine.generate_stream("q"))

    assert out == "ok"


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


def test_config_default_runtime_is_in_process():
    """Without an explicit override, runtime defaults to in_process.

    This is a critical safety property: existing callers must not
    suddenly start hitting an HTTP endpoint that may not be running.
    """
    from ultron.config import get_config
    cfg = get_config().llm
    assert cfg.runtime == "in_process"


def test_config_server_block_has_expected_defaults():
    from ultron.config import get_config
    cfg = get_config().llm
    assert cfg.server.base_url.startswith("http://")
    assert cfg.server.api_key  # non-empty
    assert cfg.server.model_alias  # non-empty
    assert cfg.server.request_timeout_s > 0
    assert cfg.server.connect_timeout_s > 0
