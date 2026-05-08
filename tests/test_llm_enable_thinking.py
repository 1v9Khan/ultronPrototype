"""4B optimization plan Stage F — enable_thinking parameter tests.

Verifies that ``LLMEngine.generate`` / ``generate_stream`` plumb the
``enable_thinking`` parameter through to llama-cpp-python's
``chat_template_kwargs`` (in-process runtime) and to the HTTP payload
(http_server runtime).

Mocks the underlying Llama / requests so no GPU or network is needed.
The actual effect on Qwen3.5 output (suppressing the
``<think>...</think>`` block) is verified at Stage H against the live
model.
"""
from __future__ import annotations

from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest

from ultron.llm.inference import LLMEngine


# ---------------------------------------------------------------------------
# Pure helper test — _chat_completion_kwargs
# ---------------------------------------------------------------------------


class _LLMCfg:
    default_temperature = 0.7
    default_top_p = 0.9
    default_max_tokens = 512
    default_repeat_penalty = 1.1


def test_chat_completion_kwargs_default() -> None:
    """No enable_thinking ⇒ no chat_template_kwargs in the request
    (back-compat — every existing test must keep passing)."""
    kw = LLMEngine._chat_completion_kwargs(_LLMCfg(), None, stream=False)
    assert "chat_template_kwargs" not in kw
    assert kw["temperature"] == 0.7
    assert kw["top_p"] == 0.9
    assert kw["max_tokens"] == 512
    assert kw["repeat_penalty"] == 1.1
    assert "stream" not in kw  # only set when streaming


def test_chat_completion_kwargs_streaming_flag() -> None:
    kw = LLMEngine._chat_completion_kwargs(_LLMCfg(), None, stream=True)
    assert kw["stream"] is True


def test_chat_completion_kwargs_enable_thinking_false() -> None:
    kw = LLMEngine._chat_completion_kwargs(_LLMCfg(), False, stream=False)
    assert kw["chat_template_kwargs"] == {"enable_thinking": False}


def test_chat_completion_kwargs_enable_thinking_true() -> None:
    kw = LLMEngine._chat_completion_kwargs(_LLMCfg(), True, stream=True)
    assert kw["chat_template_kwargs"] == {"enable_thinking": True}
    assert kw["stream"] is True


# ---------------------------------------------------------------------------
# In-process runtime — verify create_chat_completion receives the kwargs
# ---------------------------------------------------------------------------


def _make_engine_with_mock_llm() -> LLMEngine:
    """Construct an LLMEngine with the in_process llama mocked out.

    Avoids actual GGUF loading; lets us assert what create_chat_completion
    is called with."""
    eng = LLMEngine.__new__(LLMEngine)
    eng._runtime = "in_process"
    eng._llm = MagicMock()
    eng._cancel = __import__("threading").Event()
    eng._history = __import__("collections").deque()
    eng._memory = None
    eng._history_turns = 6
    eng._cfg = MagicMock()
    eng._system_prompt = "test prompt"

    def _build(user_message):
        return [
            {"role": "system", "content": "test prompt"},
            {"role": "user", "content": user_message},
        ]
    eng._build_messages = _build  # type: ignore
    eng._record_turn = MagicMock()
    return eng


def test_in_process_generate_passes_enable_thinking_false() -> None:
    eng = _make_engine_with_mock_llm()
    eng._llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"completion_tokens": 1},
    }
    with patch("ultron.llm.inference.get_config") as gc:
        gc.return_value.llm = _LLMCfg()
        eng.generate("hello", enable_thinking=False)
    call_kwargs = eng._llm.create_chat_completion.call_args.kwargs
    assert call_kwargs["chat_template_kwargs"] == {"enable_thinking": False}


def test_in_process_generate_passes_enable_thinking_true() -> None:
    eng = _make_engine_with_mock_llm()
    eng._llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "hi"}}],
        "usage": {},
    }
    with patch("ultron.llm.inference.get_config") as gc:
        gc.return_value.llm = _LLMCfg()
        eng.generate("hello", enable_thinking=True)
    call_kwargs = eng._llm.create_chat_completion.call_args.kwargs
    assert call_kwargs["chat_template_kwargs"] == {"enable_thinking": True}


def test_in_process_generate_default_omits_kwarg() -> None:
    """Default call (no enable_thinking) must not introduce
    chat_template_kwargs — preserves bit-for-bit back-compat with the
    Stage A measurement."""
    eng = _make_engine_with_mock_llm()
    eng._llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "hi"}}],
        "usage": {},
    }
    with patch("ultron.llm.inference.get_config") as gc:
        gc.return_value.llm = _LLMCfg()
        eng.generate("hello")
    call_kwargs = eng._llm.create_chat_completion.call_args.kwargs
    assert "chat_template_kwargs" not in call_kwargs


def test_in_process_generate_stream_passes_enable_thinking_false() -> None:
    eng = _make_engine_with_mock_llm()

    def _fake_stream() -> Iterator[dict]:
        yield {"choices": [{"delta": {"content": "ok"}}]}
    eng._llm.create_chat_completion.return_value = _fake_stream()
    with patch("ultron.llm.inference.get_config") as gc:
        gc.return_value.llm = _LLMCfg()
        # Drain the stream
        list(eng.generate_stream("hi", enable_thinking=False))
    call_kwargs = eng._llm.create_chat_completion.call_args.kwargs
    assert call_kwargs["chat_template_kwargs"] == {"enable_thinking": False}
    assert call_kwargs["stream"] is True


def test_in_process_generate_stream_default_omits_kwarg() -> None:
    eng = _make_engine_with_mock_llm()

    def _fake_stream() -> Iterator[dict]:
        yield {"choices": [{"delta": {"content": "ok"}}]}
    eng._llm.create_chat_completion.return_value = _fake_stream()
    with patch("ultron.llm.inference.get_config") as gc:
        gc.return_value.llm = _LLMCfg()
        list(eng.generate_stream("hi"))
    call_kwargs = eng._llm.create_chat_completion.call_args.kwargs
    assert "chat_template_kwargs" not in call_kwargs


# ---------------------------------------------------------------------------
# HTTP runtime — verify the request payload carries chat_template_kwargs
# ---------------------------------------------------------------------------


def _make_http_engine() -> LLMEngine:
    eng = LLMEngine.__new__(LLMEngine)
    eng._runtime = "http_server"
    eng._http_base_url = "http://localhost:9999/v1"
    eng._http_api_key = "test-key"
    eng._http_model_alias = "qwen-test"
    eng._http_timeout = 5.0
    eng._cancel = __import__("threading").Event()
    eng._history = __import__("collections").deque()
    eng._memory = None
    eng._history_turns = 6
    eng._system_prompt = "test"
    eng._build_messages = lambda u: [
        {"role": "system", "content": "test"},
        {"role": "user", "content": u},
    ]
    eng._record_turn = MagicMock()
    return eng


def test_http_payload_includes_enable_thinking_false() -> None:
    eng = _make_http_engine()
    captured: dict[str, Any] = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["payload"] = json
        resp = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }
        resp.raise_for_status = MagicMock()
        return resp

    with patch("ultron.llm.inference.get_config") as gc, \
         patch("requests.post", side_effect=fake_post):
        gc.return_value.llm = _LLMCfg()
        eng.generate("hello", enable_thinking=False)
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_http_payload_omits_when_default() -> None:
    eng = _make_http_engine()
    captured: dict[str, Any] = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["payload"] = json
        resp = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }
        resp.raise_for_status = MagicMock()
        return resp

    with patch("ultron.llm.inference.get_config") as gc, \
         patch("requests.post", side_effect=fake_post):
        gc.return_value.llm = _LLMCfg()
        eng.generate("hello")
    assert "chat_template_kwargs" not in captured["payload"]
