"""Reasoning-vs-text chunk demultiplexer with first-text-finalises semantics.

Adapted from cline's reasoning-block handling in
``src/core/task/index.ts:2872-2900`` (Apache 2.0; see
``THIRD_PARTY_NOTICES.md``). The discipline:

* Reasoning chunks ARE appended to a pending "reasoning" block as
  partial output.
* As soon as the FIRST non-reasoning text chunk arrives, the pending
  reasoning block is finalised (emit a :class:`ReasoningFinalisedEvent`,
  reset the pending state).
* This guarantees mixed reasoning + text never lands in the same
  emitted block, and prevents reasoning text from leaking into TTS
  via the text channel.

For ultron specifically, reasoning chunks fire ONLY to a separate
sink — never to the TTS pipeline. The orchestrator can subscribe to
:class:`ReasoningChunkEvent` to log to an audit file without piping
it through the speech path.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReasoningChunkEvent:
    """One reasoning chunk emitted by the LLM stream.

    Attributes:
        content: textual reasoning content.
        signature: optional reasoning signature (when the provider
            exposes a per-block signature for verification on the next
            turn).
        timestamp: monotonic seconds when the chunk landed.
    """

    content: str
    signature: str = ""
    timestamp: float = 0.0


@dataclass(frozen=True)
class ReasoningFinalisedEvent:
    """Emitted when a reasoning block is finalised by a text chunk.

    Attributes:
        full_text: concatenated reasoning content for the block.
        signature: optional reasoning signature preserved across turns.
        chunk_count: number of reasoning chunks accumulated.
        elapsed_seconds: time between the first reasoning chunk and the
            finalising text chunk.
    """

    full_text: str
    signature: str = ""
    chunk_count: int = 0
    elapsed_seconds: float = 0.0


class ReasoningDemultiplexer:
    """Demultiplex an LLM stream's reasoning vs text chunks.

    Args:
        on_reasoning_chunk: callback fired per reasoning chunk
            (informational; for the audit sink).
        on_reasoning_finalised: callback fired when the pending
            reasoning block is finalised by the first text chunk.
        on_text_chunk: callback fired for every text chunk (this is
            the channel the orchestrator routes to TTS).
        drop_reasoning: when True, suppress reasoning chunks entirely
            (matches the ``enable_thinking=False`` voice-path default).
        clock: optional monotonic-time clock (test hook).
    """

    def __init__(
        self,
        *,
        on_reasoning_chunk: Optional[Callable[[ReasoningChunkEvent], None]] = None,
        on_reasoning_finalised: Optional[
            Callable[[ReasoningFinalisedEvent], None]
        ] = None,
        on_text_chunk: Optional[Callable[[str], None]] = None,
        drop_reasoning: bool = False,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._on_reasoning_chunk = on_reasoning_chunk
        self._on_reasoning_finalised = on_reasoning_finalised
        self._on_text_chunk = on_text_chunk
        self._drop_reasoning = bool(drop_reasoning)
        self._clock = clock or time.monotonic
        self._lock = threading.RLock()
        self._pending_reasoning: list[str] = []
        self._pending_signature: str = ""
        self._pending_started_at: float = 0.0
        self._reasoning_blocks_finalised: int = 0

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def feed_reasoning(self, content: str, *, signature: str = "") -> None:
        """Append a reasoning chunk to the pending block.

        Args:
            content: chunk content.
            signature: optional signature (preserved across the block).
        """
        with self._lock:
            if self._drop_reasoning:
                return
            if not self._pending_reasoning:
                self._pending_started_at = self._clock()
            self._pending_reasoning.append(content)
            if signature:
                self._pending_signature = signature
        if self._on_reasoning_chunk is not None and not self._drop_reasoning:
            event = ReasoningChunkEvent(
                content=content,
                signature=signature,
                timestamp=self._clock(),
            )
            self._safe(self._on_reasoning_chunk, event)

    def feed_text(self, content: str) -> None:
        """Finalise any pending reasoning, then emit ``content`` as text."""
        finalised: Optional[ReasoningFinalisedEvent] = None
        with self._lock:
            if self._pending_reasoning:
                full_text = "".join(self._pending_reasoning)
                finalised = ReasoningFinalisedEvent(
                    full_text=full_text,
                    signature=self._pending_signature,
                    chunk_count=len(self._pending_reasoning),
                    elapsed_seconds=self._clock() - self._pending_started_at,
                )
                self._pending_reasoning = []
                self._pending_signature = ""
                self._pending_started_at = 0.0
                self._reasoning_blocks_finalised += 1
        if finalised is not None and self._on_reasoning_finalised is not None:
            self._safe(self._on_reasoning_finalised, finalised)
        if content and self._on_text_chunk is not None:
            self._safe(self._on_text_chunk, content)

    def finalise(self) -> Optional[ReasoningFinalisedEvent]:
        """Finalise any pending reasoning block (e.g. at stream end).

        Returns:
            The finalisation event (also dispatched to the callback)
            or None when nothing was pending.
        """
        with self._lock:
            if not self._pending_reasoning:
                return None
            full_text = "".join(self._pending_reasoning)
            event = ReasoningFinalisedEvent(
                full_text=full_text,
                signature=self._pending_signature,
                chunk_count=len(self._pending_reasoning),
                elapsed_seconds=self._clock() - self._pending_started_at,
            )
            self._pending_reasoning = []
            self._pending_signature = ""
            self._pending_started_at = 0.0
            self._reasoning_blocks_finalised += 1
        if self._on_reasoning_finalised is not None:
            self._safe(self._on_reasoning_finalised, event)
        return event

    def set_drop_reasoning(self, drop: bool) -> None:
        """Toggle reasoning suppression."""
        with self._lock:
            self._drop_reasoning = bool(drop)

    def reasoning_blocks_finalised(self) -> int:
        with self._lock:
            return self._reasoning_blocks_finalised

    def has_pending_reasoning(self) -> bool:
        with self._lock:
            return bool(self._pending_reasoning)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _safe(callback: Callable[..., object], *args: object) -> None:
        try:
            callback(*args)
        except Exception:  # noqa: BLE001
            LOGGER.warning(
                "reasoning demultiplexer callback raised",
                exc_info=True,
            )


__all__ = [
    "ReasoningChunkEvent",
    "ReasoningDemultiplexer",
    "ReasoningFinalisedEvent",
]
