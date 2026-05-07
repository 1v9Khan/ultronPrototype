"""Local LLM inference via llama-cpp-python.

The Ultron system prompt is baked in at construction. Conversation history
comes from one of two sources:

- **memory mode** (default when a :class:`ConversationMemory` is supplied):
  the recent N turns + top-K RAG-retrieved older snippets are injected into
  every request. History is persisted on disk by the memory module itself.
- **legacy deque mode** (no memory passed): the engine keeps a small in-memory
  ``deque`` of recent turns. Used for tests / minimal setups.

The engine also exposes :meth:`should_respond` — a fast addressee classifier
used by the orchestrator's follow-up listening state to decide whether a
non-wake-word-gated utterance is meant for Ultron.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from threading import Event
from typing import Deque, Iterator, List, Optional, Tuple

from config import settings
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


_ADDRESSEE_SYSTEM_PROMPT = """You decide whether a user's spoken utterance is \
addressed to you (Ultron) or to someone or something else (another person, \
themselves, the room, ambient speech).

You ONLY classify. You do NOT reply, explain, or continue the conversation.

Return exactly one token: YES, NO, or UNCLEAR.

Use YES only when the utterance is clearly a continuation of the conversation \
with you, a follow-up question to your last response, or directly addressed at \
you. Use NO when the utterance is clearly directed elsewhere — talking to \
another person, thinking aloud about an unrelated topic, asides. \
When uncertain, return UNCLEAR.

Default toward UNCLEAR rather than YES when the signal is weak."""


class LLMEngine:
    """Wraps a llama-cpp-python ``Llama`` instance with chat history.

    Args:
        model_path: Path to a GGUF file.
        n_ctx: Context window in tokens.
        n_gpu_layers: -1 for full offload to GPU, 0 for CPU-only.
        system_prompt: Persistent system message.
        history_turns: Legacy max user/assistant turn pairs to retain when
            no ``memory`` is supplied.
        memory: Optional :class:`ConversationMemory`. When provided, history
            is sourced from it (recent + RAG) and turns are persisted there
            instead of in the local deque.
    """

    def __init__(
        self,
        model_path: Path = settings.LLM_MODEL_PATH,
        n_ctx: int = settings.LLM_CONTEXT_LENGTH,
        n_gpu_layers: int = settings.LLM_GPU_LAYERS,
        system_prompt: str = settings.ULTRON_SYSTEM_PROMPT,
        history_turns: int = settings.LLM_HISTORY_TURNS,
        memory=None,
    ) -> None:
        from llama_cpp import Llama

        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"LLM model not found at {model_path}. "
                f"Run `python scripts/download_models.py` first."
            )

        self.model_path = Path(model_path)
        self.system_prompt = system_prompt
        self.history_turns = history_turns
        self._history: Deque[Turn] = deque(maxlen=history_turns * 2)
        self._memory = memory
        self._cancel = Event()

        logger.info("Loading LLM: %s (n_ctx=%d, n_gpu_layers=%d)…",
                    model_path, n_ctx, n_gpu_layers)
        t0 = time.monotonic()
        try:
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
        except Exception as e:
            logger.error("LLM load failed: %s", e)
            raise
        logger.info("LLM ready in %.2fs (memory=%s)",
                    time.monotonic() - t0,
                    "on" if memory is not None else "off")

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
        msgs: List[dict] = [{"role": "system", "content": self.system_prompt}]

        if self._memory is not None:
            # RAG: surface semantically-relevant turns from before the
            # recent window. Skipped if memory holds nothing yet.
            try:
                snippets = self._memory.retrieve(
                    user_message,
                    k=settings.MEMORY_RAG_TOP_K,
                    exclude_recent=settings.MEMORY_RAG_EXCLUDE_RECENT,
                )
            except Exception as e:
                logger.warning("memory.retrieve failed: %s", e)
                snippets = []
            if snippets:
                lines = ["Relevant earlier context from prior conversations:"]
                for s in snippets:
                    lines.append(f"- {s.role}: {s.content}")
                msgs.append({"role": "system", "content": "\n".join(lines)})

            for turn in self._memory.recent(settings.MEMORY_RECENT_TURNS):
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
        t0 = time.monotonic()
        out = self._llm.create_chat_completion(
            messages=messages,
            temperature=settings.LLM_TEMPERATURE,
            top_p=settings.LLM_TOP_P,
            max_tokens=settings.LLM_MAX_TOKENS,
            repeat_penalty=settings.LLM_REPEAT_PENALTY,
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

    def generate_stream(self, user_message: str) -> Iterator[str]:
        """Yield response tokens as they arrive.

        The full response is appended to history once the stream completes
        normally; on cancel, partial output is recorded so the model
        remembers what it had said.
        """
        self._cancel.clear()
        messages = self._build_messages(user_message)
        t0 = time.monotonic()
        first_token_time: Optional[float] = None
        accumulated: List[str] = []
        completed = False
        canceled = False

        stream = self._llm.create_chat_completion(
            messages=messages,
            temperature=settings.LLM_TEMPERATURE,
            top_p=settings.LLM_TOP_P,
            max_tokens=settings.LLM_MAX_TOKENS,
            repeat_penalty=settings.LLM_REPEAT_PENALTY,
            stream=True,
        )

        def _raw_deltas():
            nonlocal canceled, first_token_time, completed
            for chunk in stream:
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

    # --- addressee classification --------------------------------------------

    def should_respond(self, utterance: str) -> bool:
        """Decide whether ``utterance`` is addressed to Ultron.

        Used by the orchestrator's follow-up state to gate non-wake-word-gated
        speech. Defaults to ``False`` on parse failure, on ambiguous classifier
        output (when ``ADDRESSEE_DEFAULT_SILENT`` is set), and on errors —
        false-positive responses to unrelated speech are worse than missing a
        continuation that the user can re-trigger with the wake word.
        """
        utterance = utterance.strip()
        if not utterance:
            return False

        # Build a tight prompt: addressee system prompt + last ~3 turns + the
        # candidate utterance. Don't pollute conversation history with these.
        msgs: List[dict] = [
            {"role": "system", "content": _ADDRESSEE_SYSTEM_PROMPT},
        ]
        if self._memory is not None:
            recent = self._memory.recent(3)
            for turn in recent:
                msgs.append({"role": turn.role, "content": turn.content})
        else:
            for role, content in list(self._history)[-6:]:
                msgs.append({"role": role, "content": content})
        msgs.append(
            {
                "role": "user",
                "content": (
                    f"Utterance just heard: {utterance!r}\n"
                    "Was this addressed to you? Reply with one token only: "
                    "YES, NO, or UNCLEAR."
                ),
            }
        )

        t0 = time.monotonic()
        try:
            out = self._llm.create_chat_completion(
                messages=msgs,
                temperature=settings.ADDRESSEE_CLASSIFIER_TEMPERATURE,
                max_tokens=settings.ADDRESSEE_CLASSIFIER_MAX_TOKENS,
            )
        except Exception as e:
            logger.warning("Addressee classifier call failed: %s", e)
            return False

        raw = out["choices"][0]["message"]["content"].strip().upper()
        # Allow "YES.", "YES — ...", etc; first word is the verdict.
        verdict = raw.split()[0].rstrip(".,!?:;") if raw else "UNCLEAR"
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "Addressee classifier: %r → %s (%.0fms)", utterance[:60], verdict, elapsed_ms
        )

        if verdict == "YES":
            return True
        if verdict == "NO":
            return False
        # UNCLEAR or anything weird
        return not settings.ADDRESSEE_DEFAULT_SILENT
