"""Contextual retrieval (Anthropic technique) for conversational memory
(frontier item 4, 2026-05-21).

For each memory turn, generate a 5-15 word context phrase summarizing
the TOPIC of the turn (not its literal words). The context is
prepended to the content before embedding -- the same content is
preserved unmodified in the payload. At retrieval time, queries that
share the topic with a stored turn match via the embedded context
prefix even when the literal text is sparse (e.g., user said
"yes" + context says "agreeing to launch ChatGPT plugin tomorrow").

Why this matters for conversational memory: short utterances ("OK",
"yes", "later") have almost no embeddable signal on their own. The
LLM-generated context restores their topical meaning so retrieval
can find them when the user circles back ("what did we decide about
the plugin?").

Performance:
- The generator is a small LLM (default: the spec-decoding draft
  GGUF, e.g., Qwen3.5-0.8B Q4_K_M). Loaded lazily on first call.
- Inference is ~50-200 ms per turn on CPU; ~20-80 ms on GPU.
- Runs in the background-writer thread (off the voice hot path), so
  latency adds no perceived delay to the conversational loop.

Default device is CPU so we don't compete with the main 4B LLM
for VRAM. Set ``memory.contextual_retrieval.generator_device:
"cuda"`` to move it to GPU if VRAM headroom allows (~0.6 GB for
Qwen3.5-0.8B Q4_K_M).

Fail-open at every layer:
- Model file missing or load error -> empty context (no prefix added).
- Inference error mid-turn -> empty context for that turn.
- Voice path never blocked / never crashes on contextualizer issues.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from ultron.config import get_config
from ultron.utils.logging import get_logger

if TYPE_CHECKING:
    from ultron.config import LLMConfig, MemoryContextualRetrievalConfig

logger = get_logger("memory.contextualizer")


_DEFAULT_PROMPT_TEMPLATE = (
    "Summarize the TOPIC of this turn in 5-15 words. "
    "Just the topic phrase, no quotes, no preamble.\n\n"
    "{role}: {content}\n\n"
    "Topic:"
)


class ContextGenerator:
    """LLM-driven contextual retrieval helper.

    Wraps a small ``llama_cpp.Llama`` instance dedicated to producing
    per-turn context phrases. Stays off the voice hot path by living
    in the background memory writer thread; the loaded model never
    competes with the main voice-path 4B LLM (CPU device by default).

    Args:
        model_path: Optional override for the generator GGUF. When
            None, falls back to ``llm.draft_model_path`` from the
            unified config -- typically the spec-decoding draft.
        device: ``"cpu"`` (default) or ``"cuda"``. CPU is correct
            for the typical write rate; switch to CUDA only if you
            measure write-queue backlog.
        max_tokens: Cap on generated tokens. Defaults to 40
            (sufficient for 5-15 word phrases with headroom).
        temperature: Sampling temperature. Defaults to 0.2 -- low
            so the generator is consistent (same turn -> very
            similar context). Higher = more lexical variation
            (rarely useful for this task).
        eager: Load the model at construction. Default False --
            first ``generate_context`` call triggers the load.

    Thread-safety: ``_ensure_model`` is guarded by an internal lock
    so concurrent first-calls don't double-load. After load, the
    llama_cpp.Llama instance handles concurrency via its own state.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        eager: bool = False,
    ) -> None:
        cfg = get_config()
        ctx_cfg = cfg.memory.contextual_retrieval
        # Resolve generator model path -- explicit override, else
        # config field, else fall back to the LLM draft GGUF.
        if model_path is None:
            model_path = ctx_cfg.generator_model_path or cfg.llm.draft_model_path
        if model_path is None:
            self._model_path = None
        else:
            self._model_path = self._resolve_path(model_path)
        self.device = device or ctx_cfg.generator_device
        self.max_tokens = int(
            max_tokens if max_tokens is not None else ctx_cfg.max_context_tokens
        )
        self.temperature = float(
            temperature
            if temperature is not None
            else ctx_cfg.generator_temperature
        )
        self._llama = None
        self._load_failed = False
        self._lock = threading.Lock()
        if eager:
            self._ensure_model()

    @staticmethod
    def _resolve_path(p: str) -> Path:
        """Resolve a model path relative to project root when relative,
        else absolute. Mirrors the pattern in
        :mod:`ultron.utils.paths`."""
        path = Path(p)
        if not path.is_absolute():
            # ROOT = the project root inferred from this file's
            # location. ``src/ultron/memory/contextualizer.py`` ->
            # project root is three parents up.
            root = Path(__file__).resolve().parents[3]
            path = root / path
        return path

    def _ensure_model(self) -> bool:
        """Lazy-load the generator GGUF. Returns True on success,
        False if load failed (caller falls back to empty context)."""
        if self._llama is not None:
            return True
        if self._load_failed:
            return False
        if self._model_path is None or not Path(self._model_path).is_file():
            self._load_failed = True
            logger.warning(
                "Context generator model not found at %s; contextual "
                "retrieval will be a no-op.", self._model_path,
            )
            return False
        with self._lock:
            if self._llama is not None:
                return True
            if self._load_failed:
                return False
            t0 = time.monotonic()
            try:
                from llama_cpp import Llama
                # n_gpu_layers=0 forces CPU; -1 = all to GPU.
                n_gpu_layers = -1 if self.device == "cuda" else 0
                self._llama = Llama(
                    model_path=str(self._model_path),
                    n_ctx=2048,        # plenty for one-turn summarisation
                    n_gpu_layers=n_gpu_layers,
                    verbose=False,
                )
                logger.info(
                    "Context generator loaded: %s (device=%s) in %.2fs",
                    self._model_path.name, self.device,
                    time.monotonic() - t0,
                )
                return True
            except Exception as e:                                     # noqa: BLE001
                self._load_failed = True
                logger.warning(
                    "Context generator load failed (%s); contextual "
                    "retrieval will be a no-op.", e,
                )
                return False

    def generate_context(
        self,
        content: str,
        role: str = "user",
        *,
        prompt_template: Optional[str] = None,
    ) -> str:
        """Generate a short topic phrase for ``content``.

        Returns the empty string on any failure (model not loaded,
        generation error, empty input). Callers should treat empty
        as "no context available; embed the original content alone".

        Args:
            content: The turn's literal text.
            role: ``"user"``, ``"assistant"``, etc. Threaded into the
                prompt so the LLM knows whose perspective.
            prompt_template: Override the default prompt. Useful for
                testing OR for swapping in domain-specific framing.
                Must contain ``{role}`` + ``{content}`` placeholders.
        """
        if not content or not content.strip():
            return ""
        if not self._ensure_model():
            return ""
        template = prompt_template or _DEFAULT_PROMPT_TEMPLATE
        try:
            prompt = template.format(role=role, content=content.strip())
        except Exception as e:                                         # noqa: BLE001
            logger.warning(
                "Context prompt formatting failed (%s); returning empty.", e,
            )
            return ""
        try:
            result = self._llama(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["\n\n", "User:", "Assistant:", "Topic:"],
                echo=False,
            )
            choices = result.get("choices", []) if isinstance(result, dict) else []
            if not choices:
                return ""
            text = str(choices[0].get("text", "")).strip()
            # Strip surrounding quotes the LLM sometimes adds.
            text = text.strip("\"'`")
            # Strip leading "Topic:" if the LLM echoed.
            for prefix in ("Topic:", "topic:", "TOPIC:"):
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()
            return text
        except Exception as e:                                         # noqa: BLE001
            logger.warning(
                "Context generation failed (%s); returning empty.", e,
            )
            return ""

    def close(self) -> None:
        """Release the generator's resources. Idempotent."""
        if self._llama is not None:
            try:
                self._llama.close()
            except Exception:
                pass
            self._llama = None


__all__ = ["ContextGenerator"]
