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
        if history_turns is None:
            history_turns = cfg.history_turns
        runtime = runtime or cfg.runtime

        # Phase 1: persona source can be the shared workspace files
        # (loaded fresh each turn so SOUL.md edits hot-reload) OR the
        # legacy hardcoded ``llm.system_prompt`` string.
        #
        # When ``system_prompt=`` is passed explicitly to the
        # constructor we honor it as-is — that's the test path and the
        # explicit override path. Otherwise we resolve per
        # ``llm.persona.source``.
        self._explicit_system_prompt: Optional[str] = system_prompt
        self._persona_loader = self._maybe_build_persona_loader(cfg)
        # Cached static prompt for ``persona.source == "config"``.
        self._static_system_prompt: str = (
            system_prompt if system_prompt is not None else cfg.system_prompt
        )
        # ``self.system_prompt`` is kept for backward compat (existing
        # tests read it). It reflects the most recently resolved value.
        self.system_prompt = self._resolve_system_prompt()
        self.history_turns = history_turns
        self._history: Deque[Turn] = deque(maxlen=history_turns * 2)
        self._memory = memory
        self._cancel = Event()
        self._runtime = runtime
        self._logged_initial_persona = False

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
        llama, resolved_path = self._build_llama(cfg, model_path, n_ctx, n_gpu_layers)
        self._llm = llama
        self.model_path = resolved_path

    def _build_llama(
        self,
        cfg,
        model_path: Optional[Path],
        n_ctx: Optional[int],
        n_gpu_layers: Optional[int],
    ) -> "tuple":
        """Construct + return a fresh ``Llama`` instance per ``cfg``.

        Returns ``(llama, resolved_model_path)``. Does NOT mutate
        ``self`` — used by both ``_init_in_process`` (sets ``self._llm``
        from the result) and ``reload_for_preset`` (constructs the new
        instance before releasing the old one so VRAM is recoverable
        on failure).
        """
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

        flash_attn = cfg.flash_attn
        kv_cache_type = cfg.kv_cache_type
        logger.info(
            "Loading LLM (in_process): %s (n_ctx=%d, n_gpu_layers=%d, "
            "flash_attn=%s, kv_cache_type=%d)...",
            model_path, n_ctx, n_gpu_layers, flash_attn, kv_cache_type,
        )
        t0 = time.monotonic()
        try:
            llama = Llama(
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
        logger.info(
            "LLM ready in %.2fs (memory=%s)",
            time.monotonic() - t0,
            "on" if self._memory is not None else "off",
        )
        return llama, Path(model_path)

    @staticmethod
    def _maybe_build_persona_loader(cfg):
        """Construct a PersonaLoader if config asks for the workspace
        source. Returns ``None`` for the legacy ``config`` source.

        Importing PersonaLoader is deferred so test environments that
        don't need it never pay the import cost.
        """
        persona_cfg = getattr(cfg, "persona", None)
        if persona_cfg is None or persona_cfg.source != "workspace":
            return None
        # Lazy import: PersonaLoader is in the openclaw_bridge package
        # which has no runtime deps, but we still avoid importing it
        # when the config doesn't ask for it.
        from ultron.openclaw_bridge.persona import (
            PersonaLoader, default_workspace_dir,
        )
        ws = persona_cfg.workspace_dir
        return PersonaLoader(
            Path(ws) if ws else default_workspace_dir()
        )

    def _resolve_system_prompt(self) -> str:
        """Resolve the system prompt for this turn.

        Order:
        1. Explicit constructor override (``system_prompt=`` arg).
        2. Workspace persona via PersonaLoader (``persona.source == "workspace"``).
           Hot-reloads via ``refresh_if_stale``.
        3. Fallback to ``cfg.system_prompt`` (the legacy hardcoded string)
           when workspace returned empty AND
           ``persona.fallback_to_config_on_empty`` is True.
        4. Otherwise the static prompt captured at construction.
        """
        if self._explicit_system_prompt is not None:
            return self._explicit_system_prompt
        loader = self._persona_loader
        if loader is None:
            return self._static_system_prompt
        try:
            prompt = loader.get_system_prompt("user_facing")
        except Exception as e:
            logger.warning(
                "PersonaLoader failed (%s); falling back to config", e,
            )
            return self._static_system_prompt
        if prompt:
            return prompt
        # Workspace was empty / unset.
        cfg = get_config().llm
        if cfg.persona.fallback_to_config_on_empty:
            logger.warning(
                "Persona workspace empty; falling back to "
                "llm.system_prompt config value"
            )
            return self._static_system_prompt
        return ""

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

    def _build_messages(
        self, user_message: str, *, gate_verdict=None,
    ) -> List[dict]:
        # Resolve the system prompt fresh each turn. When the persona
        # source is the workspace, this is what makes hot reload work:
        # PersonaLoader's refresh_if_stale catches mtime/size changes
        # so a SOUL.md edit takes effect on the next user turn without
        # restart. Cost is ~6 stat() calls (sub-millisecond).
        system_content = self._resolve_system_prompt()
        # Keep ``self.system_prompt`` in sync with the resolved value
        # so external readers (tests, debug log dumps) see the live
        # prompt, not the construction-time snapshot.
        self.system_prompt = system_content
        if not self._logged_initial_persona:
            self._logged_initial_persona = True
            logger.debug(
                "system prompt (%d chars, source=%s):\n%s",
                len(system_content),
                "explicit" if self._explicit_system_prompt is not None
                else ("workspace" if self._persona_loader is not None
                      else "config"),
                system_content,
            )

        # 4B plan Stage G — RAG injection position is config-driven.
        # Qwen3's chat template rejects a second system-role message, so
        # the only two viable positions are:
        #   "system": fold the RAG block into the leading system message
        #   "recency": prepend the RAG block to the final user message
        # The second is the default at Stage G — it puts retrieved
        # context in the strongest-attention zone (right before the
        # user query) and recovers +10-20% recall on the 4B.
        rag_block = self._format_rag_block(
            self._retrieve_rag_snippets(
                user_message, gate_verdict=gate_verdict,
            ),
        )
        rag_position = get_config().llm.rag.position

        if rag_block and rag_position == "system":
            system_content = system_content + rag_block

        msgs: List[dict] = [{"role": "system", "content": system_content}]

        if self._memory is not None:
            for turn in self._memory.recent(get_config().memory.recent_turns):
                msgs.append({"role": turn.role, "content": turn.content})
        else:
            for role, content in self._history:
                msgs.append({"role": role, "content": content})

        if rag_block and rag_position == "recency":
            user_content = rag_block.lstrip("\n") + "\n\n" + user_message
        else:
            user_content = user_message
        msgs.append({"role": "user", "content": user_content})
        return msgs

    # --- 4B plan Stage G: RAG retrieval + formatting helpers ---------------

    def _retrieve_rag_snippets(
        self, user_message: str, *, gate_verdict=None,
    ) -> List:
        """Best-effort fetch of RAG snippets from the memory module.

        V1-gap A2: when ``gate_verdict`` is provided AND
        ``memory.retrieval.multi_pass_enabled`` is True, routes through
        :meth:`ConversationMemory.retrieve_for_query` so the gate's
        category sub-queries fan out into a multi-pass retrieval. With
        no verdict (or the flag off), falls back to the original
        single-pass ``retrieve`` -- byte-for-byte identical to today.

        Returns ``[]`` on failure or when memory is disabled. Logs a
        warning on retrieval failure but never raises.
        """
        if self._memory is None:
            return []
        mem_cfg = get_config().memory
        try:
            if gate_verdict is not None and hasattr(
                self._memory, "retrieve_for_query",
            ):
                return list(self._memory.retrieve_for_query(
                    user_message,
                    gate_verdict,
                    k=mem_cfg.rag_top_k,
                    exclude_recent=mem_cfg.rag_exclude_recent,
                ))
            return list(self._memory.retrieve(
                user_message,
                k=mem_cfg.rag_top_k,
                exclude_recent=mem_cfg.rag_exclude_recent,
            ))
        except Exception as e:
            logger.warning("memory.retrieve failed: %s", e)
            return []

    @staticmethod
    def _format_rag_block(snippets: List) -> str:
        """Render the retrieved snippets as a labelled text block.

        Returns ``""`` when there are no snippets so the caller can do
        a simple truthiness check. Same content shape as before Stage G
        for back-compat with anything inspecting the rendered prompt.

        4B plan Item 4: optionally compresses the rendered block when
        ``llm.compression.enabled`` AND ``llm.compression.compress_rag``
        are both True. Pass-through otherwise (default).
        """
        if not snippets:
            return ""
        lines = ["", "Relevant earlier context from prior conversations:"]
        for s in snippets:
            lines.append(f"- {s.role}: {s.content}")
        block = "\n".join(lines)
        # Late import + best-effort: never break the hot path.
        try:
            from ultron.llm.compression import maybe_compress
            return maybe_compress(block, surface="rag")
        except Exception:
            return block

    # --- generation ----------------------------------------------------------

    def cancel(self) -> None:
        """Signal :meth:`generate_stream` to stop emitting tokens.

        The underlying llama-cpp call will continue until its current token
        finishes — but the iterator will exit immediately afterward.
        """
        self._cancel.set()

    # --- 4B plan: voice-driven on-the-fly model reload ---------------------

    def reload_for_preset(self, preset: str) -> "tuple[bool, str]":
        """Hot-swap the loaded LLM to ``preset`` without restarting Ultron.

        Implementation strategy: load the NEW ``Llama`` instance FIRST,
        then release the old one only on success. This means a failed
        swap (missing GGUF, invalid preset) leaves the engine in its
        original working state — no broken-pipeline window.

        Cost: peak VRAM during the swap is roughly ``old + new`` GGUF
        size, briefly. For 4B (2.5 GB) ↔ 9B (5.3 GB) on a 12 GB card,
        7.8 GB peak is comfortably under the 11.5 GB hard cap.

        Returns ``(success, message)``. On failure, ``self._llm`` and
        ``self.model_path`` are unchanged. On success, history is
        reset (different model = different context budget; carrying
        over recent turns risks exceeding the new ``n_ctx``).

        Only supports ``runtime == "in_process"``. The HTTP-server
        path requires restarting llama-cpp-server with the new ``--from-config``
        flags — that's a separate orchestrator-level concern.
        """
        from ultron.config import LLM_PRESETS, get_config, reload_config

        if self._runtime != "in_process":
            return False, "reload_for_preset only supports in_process runtime"
        if preset not in LLM_PRESETS and preset != "custom":
            return False, f"unknown preset {preset!r}"

        current = get_config().llm.preset
        if current == preset:
            return True, f"already on {preset}"

        # Make the env override authoritative for the upcoming reload —
        # this is the same path the user would take from the shell.
        # Save originals so we can restore on failure.
        prior_env_preset = os.environ.get("ULTRON_LLM_PRESET")
        prior_env_model = os.environ.get("ULTRON_LLM_MODEL_PATH")
        os.environ["ULTRON_LLM_PRESET"] = preset
        # A stale model-path override would clobber the preset's table.
        os.environ.pop("ULTRON_LLM_MODEL_PATH", None)

        # Cancel any in-flight stream so the old generator's clean-up
        # finishes before we drop the Llama instance.
        self._cancel.set()

        try:
            new_cfg = reload_config().llm
            new_llm, new_path = self._build_llama(
                new_cfg, model_path=None, n_ctx=None, n_gpu_layers=None,
            )
        except Exception as e:
            # Restore env (so a subsequent get_config() doesn't drift)
            # and reload to recover the prior config.
            if prior_env_preset is None:
                os.environ.pop("ULTRON_LLM_PRESET", None)
            else:
                os.environ["ULTRON_LLM_PRESET"] = prior_env_preset
            if prior_env_model is not None:
                os.environ["ULTRON_LLM_MODEL_PATH"] = prior_env_model
            try:
                reload_config()
            except Exception:
                pass  # don't compound failures
            self._cancel.clear()
            logger.error("reload_for_preset(%s) failed: %s", preset, e)
            return False, f"failed to load {preset}: {e}"

        # Success — release old, swap in new.
        old_llm = self._llm
        self._llm = new_llm
        self.model_path = new_path
        del old_llm
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        try:  # pragma: no cover — torch import may fail in CPU-only test envs
            import torch  # noqa: WPS433
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Reset history — different n_ctx + different tokenizer state.
        # Memory turns persist on disk; only the in-memory deque clears.
        self._history.clear()
        self._cancel.clear()
        logger.info("reload_for_preset(%s) succeeded; model=%s", preset, new_path)
        return True, f"loaded {preset}"

    def generate(
        self,
        user_message: str,
        *,
        enable_thinking: Optional[bool] = None,
        gate_verdict=None,
    ) -> str:
        """Blocking generation. Returns the full response string.

        ``enable_thinking`` (4B optimization plan Stage F):
        - ``None`` (default): inherit the chat template's default. Today
          that's "thinking on" for Qwen3.5 — the model emits a
          ``<think>...</think>`` block before the answer, which
          :func:`_strip_thinking_blocks` filters out before tokens reach
          TTS.
        - ``False``: disable thinking via Qwen3.5's
          ``chat_template_kwargs={"enable_thinking": False}``. Recovers
          the 2-5x token-output overhead the thinking block adds. Use
          for: simple conversation, voice path on 4B, acknowledgment
          phrases, pre-flight uncertainty pass.
        - ``True``: explicitly request thinking on. Use for: tool-routing
          decisions, clarification, correction-prompt generation,
          HYBRID_TASK decomposition, adjustment context processing.

        See [docs/4b_optimization_plan.md](../../docs/4b_optimization_plan.md)
        for the per-intent thinking-mode table.

        ``gate_verdict`` (V1-gap A2): when set AND
        ``memory.retrieval.multi_pass_enabled`` is True, the RAG block
        is built via the multi-pass per-category retrieval path. ``None``
        preserves the original single-pass behaviour.
        """
        messages = self._build_messages(user_message, gate_verdict=gate_verdict)
        _llm_cfg = get_config().llm
        t0 = time.monotonic()
        if self._runtime == "in_process":
            kwargs = self._chat_completion_kwargs(_llm_cfg, enable_thinking, stream=False)
            out = self._llm.create_chat_completion(messages=messages, **kwargs)
        else:
            out = self._http_chat_completion(
                messages, _llm_cfg, stream=False, enable_thinking=enable_thinking,
            )
        text = out["choices"][0]["message"]["content"].strip()
        logger.info(
            "LLM: %d chars in %.2fs (%d tokens)",
            len(text),
            time.monotonic() - t0,
            out.get("usage", {}).get("completion_tokens", -1),
        )
        self._record_turn(user_message, text)
        return text

    def generate_stream(
        self,
        user_message: str,
        *,
        enable_thinking: Optional[bool] = None,
        gate_verdict=None,
    ) -> Iterator[str]:
        """Yield response tokens as they arrive.

        See :meth:`generate` for the ``enable_thinking`` and
        ``gate_verdict`` (V1-gap A2) semantics.

        The full response is appended to history once the stream completes
        normally; on cancel, partial output is recorded so the model
        remembers what it had said.
        """
        self._cancel.clear()
        messages = self._build_messages(user_message, gate_verdict=gate_verdict)
        _llm_cfg = get_config().llm
        t0 = time.monotonic()
        first_token_time: Optional[float] = None
        accumulated: List[str] = []
        completed = False
        canceled = False

        if self._runtime == "in_process":
            kwargs = self._chat_completion_kwargs(_llm_cfg, enable_thinking, stream=True)
            stream = self._llm.create_chat_completion(messages=messages, **kwargs)
            stream_iter = stream
        else:
            stream_iter = self._http_chat_completion(
                messages, _llm_cfg, stream=True, enable_thinking=enable_thinking,
            )

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

    # --- 4B plan Stage F: selective thinking mode ---------------------------

    @staticmethod
    def _chat_completion_kwargs(
        _llm_cfg, enable_thinking: Optional[bool], *, stream: bool,
    ) -> dict:
        """Build the kwargs dict for ``Llama.create_chat_completion``.

        Centralised so both ``generate`` and ``generate_stream`` produce
        identical request shape (only ``stream`` differs), and so the
        Stage F ``enable_thinking`` toggle is set in exactly one place.

        Returns a fresh dict — the caller is free to mutate without
        affecting other calls.
        """
        kwargs: dict = {
            "temperature": _llm_cfg.default_temperature,
            "top_p": _llm_cfg.default_top_p,
            "max_tokens": _llm_cfg.default_max_tokens,
            "repeat_penalty": _llm_cfg.default_repeat_penalty,
        }
        if stream:
            kwargs["stream"] = True
        if enable_thinking is not None:
            # Qwen3.5's chat template reads ``enable_thinking`` to decide
            # whether to emit a ``<think>...</think>`` block. ``False``
            # cuts the 2-5x token-output overhead the thinking block
            # adds — meaningful on the 4B (smaller model = relatively
            # bigger latency cost from thinking).
            kwargs["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        return kwargs

    # --- HTTP runtime helpers ----------------------------------------------

    def _http_chat_completion(
        self, messages, _llm_cfg, *, stream: bool,
        enable_thinking: Optional[bool] = None,
    ):
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
        if enable_thinking is not None:
            # llama-cpp-server passes chat_template_kwargs through to its
            # underlying create_chat_completion call. Same Qwen3.5 toggle
            # as the in-process path.
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
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

