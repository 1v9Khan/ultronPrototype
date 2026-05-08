"""Local LLM inference.

The Ultron system prompt is baked in at construction. Conversation history
comes from one of two sources:

- **memory mode** (default when a :class:`ConversationMemory` is supplied):
  the recent N turns + top-K RAG-retrieved older snippets are injected into
  every request. History is persisted on disk by the memory module itself.
- **legacy deque mode** (no memory passed): the engine keeps a small in-memory
  ``deque`` of recent turns. Used for tests / minimal setups.

Two runtimes:

- ``in_process`` (default): loads the GGUF via llama-cpp-python in this
  process. The current voice-pipeline mode. ~5.7 GB VRAM.
- ``http_server``: talks to a separately-run llama-cpp-server over OpenAI-
  compatible HTTP. Used to share one model load with OpenClaw. The voice
  path can opt in via ``llm.runtime: http_server`` in config.yaml; both
  consumers share VRAM.

Both runtimes expose the same :meth:`generate` / :meth:`generate_stream`
surface. Same params, same chat history composition, same cancel
behaviour, same system prompt. The branching is internal.

Addressee classification used to live here as ``should_respond``; that path
was retired in Phase 2 in favor of a dedicated CPU classifier in
:mod:`ultron.addressing`, which keeps the main 9 B LLM off the WARM-mode hot
path entirely.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from threading import Event
from typing import Deque, Iterator, List, Optional, Tuple

import os

from ultron.config import get_config, resolve_path
from ultron.utils.logging import get_logger

logger = get_logger("llm.inference")

Turn = Tuple[str, str]  # (role, content)


def _strip_thinking_blocks(stream: Iterator[str]) -> Iterator[str]:
    """Yield tokens from ``stream`` with ``<think>...</think>`` blocks removed.

    Qwen3 / Qwen3.5 models emit a chain-of-thought block before the actual
    answer when reasoning mode is on. That block is part of the streamed
    content, so it would otherwise reach Piper and be spoken. We hold back a
    small tail buffer so partial tags split across token boundaries are
    handled correctly.
    """
    HOLD = 8  # longer than "</think>"
    buf = ""
    in_think = False
    for delta in stream:
        if not delta:
            continue
        buf += delta
        while True:
            if in_think:
                idx = buf.find("</think>")
                if idx == -1:
                    if len(buf) > HOLD:
                        buf = buf[-HOLD:]
                    break
                buf = buf[idx + len("</think>"):]
                in_think = False
            else:
                idx = buf.find("<think>")
                if idx == -1:
                    if len(buf) > HOLD:
                        emit = buf[:-HOLD]
                        buf = buf[-HOLD:]
                        if emit:
                            yield emit
                    break
                if idx > 0:
                    yield buf[:idx]
                buf = buf[idx + len("<think>"):]
                in_think = True
    if not in_think and buf:
        yield buf


class LLMEngine:
    """LLM client with chat history.

    Two backends, selected by ``llm.runtime`` in config:

    - ``in_process`` (default): loads the GGUF via llama-cpp-python.
      Same VRAM-resident model used directly. Today's voice-pipeline
      mode.
    - ``http_server``: talks to a separately-run llama-cpp-server
      (``scripts/start_llamacpp_server.py``) via OpenAI-compat HTTP.
      Lets the voice pipeline share the same model load with OpenClaw.

    Both backends expose identical ``generate()`` /
    ``generate_stream()`` surfaces with the same params, history
    composition, cancel behaviour, and chain-of-thought stripping.

    Args:
        model_path: Path to a GGUF file. Only used for in_process mode.
        n_ctx: Context window in tokens. Only used for in_process mode.
        n_gpu_layers: -1 for full offload to GPU, 0 for CPU-only.
            Only used for in_process mode.
        system_prompt: Persistent system message.
        history_turns: Legacy max user/assistant turn pairs to retain
            when no ``memory`` is supplied.
        memory: Optional :class:`ConversationMemory`. When provided,
            history is sourced from it (recent + RAG) and turns are
            persisted there instead of in the local deque.
        runtime: Optional override of ``llm.runtime``. Useful for tests
            that want to exercise the HTTP path without flipping global
            config.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        n_ctx: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        system_prompt: Optional[str] = None,
        history_turns: Optional[int] = None,
        memory=None,
        runtime: Optional[str] = None,
    ) -> None:
        cfg = get_config().llm
        if system_prompt is None:
            system_prompt = cfg.system_prompt
        if history_turns is None:
            history_turns = cfg.history_turns
        runtime = runtime or cfg.runtime

        self.system_prompt = system_prompt
        self.history_turns = history_turns
        self._history: Deque[Turn] = deque(maxlen=history_turns * 2)
        self._memory = memory
        self._cancel = Event()
        self._runtime = runtime

        if runtime == "in_process":
            self._init_in_process(cfg, model_path, n_ctx, n_gpu_layers)
        elif runtime == "http_server":
            self._init_http_server(cfg)
        else:
            raise ValueError(
                f"unknown llm.runtime {runtime!r}; "
                f"expected 'in_process' or 'http_server'"
            )

    # --- runtime selectors -------------------------------------------------

    def _init_in_process(
        self,
        cfg,
        model_path: Optional[Path],
        n_ctx: Optional[int],
        n_gpu_layers: Optional[int],
    ) -> None:
        from llama_cpp import Llama

        if model_path is None:
            # Env var override remains as an opt-in for swapping models without
            # editing config.yaml; falls through to the configured path.
            env_path = os.getenv("ULTRON_LLM_MODEL_PATH")
            model_path = resolve_path(env_path or cfg.model_path)
        if n_ctx is None:
            n_ctx = cfg.n_ctx
        if n_gpu_layers is None:
            n_gpu_layers = cfg.gpu_layers

        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"LLM model not found at {model_path}. "
                f"Run `python scripts/download_models.py` first."
            )

        self.model_path = Path(model_path)
        flash_attn = cfg.flash_attn
        kv_cache_type = cfg.kv_cache_type
        logger.info(
            "Loading LLM (in_process): %s (n_ctx=%d, n_gpu_layers=%d, "
            "flash_attn=%s, kv_cache_type=%d)...",
            model_path, n_ctx, n_gpu_layers, flash_attn, kv_cache_type,
        )
        t0 = time.monotonic()
        try:
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                # Flash attention + quantized KV cache cut KV memory ~30 %
                # each (combined ~50 %) at quality parity for inference.
                # Flash attn is required for non-F16 KV cache types.
                flash_attn=flash_attn,
                type_k=kv_cache_type,
                type_v=kv_cache_type,
                verbose=False,
            )
        except Exception as e:
            logger.error("LLM load failed: %s", e)
            raise
        logger.info("LLM ready in %.2fs (memory=%s)",
                    time.monotonic() - t0,
                    "on" if self._memory is not None else "off")

    def _init_http_server(self, cfg) -> None:
        """Configure the HTTP-client path. No model load happens here —
        the server (started separately) holds the weights."""
        server = cfg.server
        # Normalise the base URL to end without a trailing slash; we
        # always construct ``<base>/chat/completions`` etc.
        base = server.base_url.rstrip("/")
        self.model_path = None  # not applicable for HTTP runtime
        self._llm = None
        self._http_base_url = base
        self._http_api_key = server.api_key
        self._http_model_alias = server.model_alias
        self._http_timeout = (server.connect_timeout_s, server.request_timeout_s)
        logger.info(
            "LLM in http_server runtime: base=%s model_alias=%s",
            base, server.model_alias,
        )

    # --- context manager -----------------------------------------------------

    def __enter__(self) -> "LLMEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._llm = None  # release GPU memory at GC time

    # --- history management --------------------------------------------------

    def reset_history(self) -> None:
        self._history.clear()

    def _record_turn(self, user_message: str, assistant_message: str) -> None:
        """Persist a completed user/assistant exchange."""
        if self._memory is not None:
            self._memory.add("user", user_message)
            self._memory.add("assistant", assistant_message)
        else:
            self._history.append(("user", user_message))
            self._history.append(("assistant", assistant_message))

    def _build_messages(self, user_message: str) -> List[dict]:
        # RAG snippets are folded into the leading system message rather than
        # emitted as a second `system`-role entry: Qwen3's chat template
        # rejects a second system message with "System message must be at
        # the beginning."
        system_content = self.system_prompt

        if self._memory is not None:
            mem_cfg = get_config().memory
            try:
                snippets = self._memory.retrieve(
                    user_message,
                    k=mem_cfg.rag_top_k,
                    exclude_recent=mem_cfg.rag_exclude_recent,
                )
            except Exception as e:
                logger.warning("memory.retrieve failed: %s", e)
                snippets = []
            if snippets:
                lines = ["", "Relevant earlier context from prior conversations:"]
                for s in snippets:
                    lines.append(f"- {s.role}: {s.content}")
                system_content = system_content + "\n".join(lines)

        msgs: List[dict] = [{"role": "system", "content": system_content}]

        if self._memory is not None:
            for turn in self._memory.recent(get_config().memory.recent_turns):
                msgs.append({"role": turn.role, "content": turn.content})
        else:
            for role, content in self._history:
                msgs.append({"role": role, "content": content})

        msgs.append({"role": "user", "content": user_message})
        return msgs

    # --- generation ----------------------------------------------------------

    def cancel(self) -> None:
        """Signal :meth:`generate_stream` to stop emitting tokens.

        The underlying llama-cpp call will continue until its current token
        finishes — but the iterator will exit immediately afterward.
        """
        self._cancel.set()

    def generate(self, user_message: str) -> str:
        """Blocking generation. Returns the full response string."""
        messages = self._build_messages(user_message)
        _llm_cfg = get_config().llm
        t0 = time.monotonic()
        if self._runtime == "in_process":
            out = self._llm.create_chat_completion(
                messages=messages,
                temperature=_llm_cfg.default_temperature,
                top_p=_llm_cfg.default_top_p,
                max_tokens=_llm_cfg.default_max_tokens,
                repeat_penalty=_llm_cfg.default_repeat_penalty,
            )
        else:
            out = self._http_chat_completion(messages, _llm_cfg, stream=False)
        text = out["choices"][0]["message"]["content"].strip()
        logger.info(
            "LLM: %d chars in %.2fs (%d tokens)",
            len(text),
            time.monotonic() - t0,
            out.get("usage", {}).get("completion_tokens", -1),
        )
        self._record_turn(user_message, text)
        return text

    def generate_stream(self, user_message: str) -> Iterator[str]:
        """Yield response tokens as they arrive.

        The full response is appended to history once the stream completes
        normally; on cancel, partial output is recorded so the model
        remembers what it had said.
        """
        self._cancel.clear()
        messages = self._build_messages(user_message)
        _llm_cfg = get_config().llm
        t0 = time.monotonic()
        first_token_time: Optional[float] = None
        accumulated: List[str] = []
        completed = False
        canceled = False

        if self._runtime == "in_process":
            stream = self._llm.create_chat_completion(
                messages=messages,
                temperature=_llm_cfg.default_temperature,
                top_p=_llm_cfg.default_top_p,
                max_tokens=_llm_cfg.default_max_tokens,
                repeat_penalty=_llm_cfg.default_repeat_penalty,
                stream=True,
            )
            stream_iter = stream
        else:
            stream_iter = self._http_chat_completion(messages, _llm_cfg, stream=True)

        def _raw_deltas():
            nonlocal canceled, first_token_time, completed
            for chunk in stream_iter:
                if self._cancel.is_set():
                    canceled = True
                    logger.info("LLM stream canceled by caller")
                    return
                delta = chunk["choices"][0].get("delta", {}).get("content")
                if not delta:
                    continue
                if first_token_time is None:
                    first_token_time = time.monotonic()
                    logger.info("LLM TTFT: %.0fms",
                                (first_token_time - t0) * 1000)
                yield delta
            completed = True

        try:
            for visible in _strip_thinking_blocks(_raw_deltas()):
                accumulated.append(visible)
                yield visible
        finally:
            full = "".join(accumulated).strip()
            if full and completed and not canceled:
                self._record_turn(user_message, full)
            elif full:
                logger.info("Skipping interrupted LLM stream in chat history")
            logger.info(
                "LLM stream: %d chars in %.2fs",
                len(full),
                time.monotonic() - t0,
            )

    # --- HTTP runtime helpers ----------------------------------------------

    def _http_chat_completion(self, messages, _llm_cfg, *, stream: bool):
        """OpenAI-compat chat-completion request to llama-cpp-server.

        Returns either a single response dict (``stream=False``) or an
        iterator of streaming chunk dicts (``stream=True``). The chunk
        shape mirrors llama-cpp-python's
        ``create_chat_completion(stream=True)`` output, so the
        surrounding code in :meth:`generate_stream` is identical.
        """
        import json as _json
        import requests

        url = f"{self._http_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._http_api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }
        payload = {
            "model": self._http_model_alias,
            "messages": messages,
            "temperature": _llm_cfg.default_temperature,
            "top_p": _llm_cfg.default_top_p,
            "max_tokens": _llm_cfg.default_max_tokens,
            "repeat_penalty": _llm_cfg.default_repeat_penalty,
            "stream": stream,
        }
        if not stream:
            resp = requests.post(
                url, headers=headers, json=payload,
                timeout=self._http_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        # Streaming path. Yield chunk dicts as they arrive.
        return self._http_stream(url, headers, payload)

    def _http_stream(self, url, headers, payload):
        """Stream OpenAI-compat SSE chunks. Cancel-aware via
        ``self._cancel``; closes the response cleanly on cancel."""
        import json as _json
        import requests

        with requests.post(
            url, headers=headers, json=payload,
            timeout=self._http_timeout, stream=True,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines(decode_unicode=True):
                if self._cancel.is_set():
                    # Caller will record the cancel; we just stop reading.
                    break
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    # Heartbeat/comments — ignore.
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = _json.loads(data)
                except _json.JSONDecodeError:
                    logger.debug("dropping non-JSON SSE chunk: %s", data[:120])
                    continue
                yield chunk

