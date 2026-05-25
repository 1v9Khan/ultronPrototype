"""Stream coordinator with retry-status surface for invisible auto-retries.

Adapted from cline's ``StreamChunkCoordinator`` + ``Task.onRetryAttempt``
pattern (Apache 2.0; see ``THIRD_PARTY_NOTICES.md``). Cline updates an
existing ``api_req_started`` UI message in-place with a
``retryStatus`` block; ultron's equivalent publishes a structured
:class:`RetryStatus` payload that the orchestrator can render at
configurable verbosity ("silent" / "narrate" / "interrupt").

The coordinator wraps any iterable / iterator of chunks (textual or
typed) and exposes a small state machine: ``next_chunk``, ``stop``,
``wait_for_completion``. Per-chunk type filters route ``usage`` chunks
to a separate callback so token meters update LIVE during the stream.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

LOGGER = logging.getLogger(__name__)


class StreamState(str, Enum):
    """States the coordinator passes through."""

    IDLE = "idle"
    STREAMING = "streaming"
    RETRYING = "retrying"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class StreamChunk:
    """Typed view of one chunk produced by the underlying stream.

    Attributes:
        kind: chunk type marker (``"text"`` / ``"reasoning"`` /
            ``"tool_use"`` / ``"usage"`` / other provider-specific).
        content: chunk payload (caller defines shape).
        sequence: monotonic per-stream index (0-based).
    """

    kind: str
    content: Any
    sequence: int = 0


@dataclass(frozen=True)
class RetryStatus:
    """Retry-attempt status published mid-stream.

    Attributes:
        attempt: 1-based retry attempt number.
        max_attempts: total attempts the wrapper will make.
        delay_seconds: time the wrapper is sleeping before the next try.
        error_snippet: short rendering of the error that triggered retry.
    """

    attempt: int
    max_attempts: int
    delay_seconds: float
    error_snippet: str = ""


class StreamCoordinator:
    """Wrap an iterable of chunks with state + retry-status semantics.

    Args:
        source: iterable producing :class:`StreamChunk` (or arbitrary
            objects the caller is happy to receive verbatim).
        on_usage: optional callback fired for chunks whose ``kind``
            matches ``"usage"`` (cline's pattern: token meters update
            live during the stream).
        on_retry: optional callback receiving :class:`RetryStatus`
            events. Wire this into the bus so the UI / TTS can decide
            whether to narrate or stay silent.
        on_state_change: optional callback receiving the new
            :class:`StreamState` after every transition.
        clock: optional monotonic clock (test hook).
    """

    def __init__(
        self,
        source: Iterable[Any],
        *,
        on_usage: Optional[Callable[[Any], None]] = None,
        on_retry: Optional[Callable[[RetryStatus], None]] = None,
        on_state_change: Optional[Callable[[StreamState], None]] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._source = source
        self._iter: Optional[Iterator[Any]] = None
        self._on_usage = on_usage
        self._on_retry = on_retry
        self._on_state_change = on_state_change
        self._clock = clock or time.monotonic
        self._lock = threading.RLock()
        self._state: StreamState = StreamState.IDLE
        self._sequence: int = 0
        self._chunks_emitted: int = 0
        self._cancel_event = threading.Event()
        self._completion_event = threading.Event()
        self._last_error: Optional[BaseException] = None
        self._last_retry_status: Optional[RetryStatus] = None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def state(self) -> StreamState:
        with self._lock:
            return self._state

    def chunks_emitted(self) -> int:
        with self._lock:
            return self._chunks_emitted

    def last_retry_status(self) -> Optional[RetryStatus]:
        with self._lock:
            return self._last_retry_status

    def last_error(self) -> Optional[BaseException]:
        with self._lock:
            return self._last_error

    # ------------------------------------------------------------------
    # Stream lifecycle
    # ------------------------------------------------------------------

    def next_chunk(self) -> Optional[Any]:
        """Pull the next chunk from the source.

        Returns:
            The next chunk, or None when the stream is exhausted /
            cancelled.

        Notes:
            On the first call, the iterator is materialised and the
            state transitions to ``STREAMING``. Subsequent calls
            advance the iterator. When exhaustion is detected the
            state moves to ``COMPLETE`` and ``wait_for_completion``
            unblocks.
        """
        if self._cancel_event.is_set():
            self._transition(StreamState.CANCELLED)
            self._completion_event.set()
            return None
        with self._lock:
            if self._iter is None:
                try:
                    self._iter = iter(self._source)
                except Exception as exc:  # noqa: BLE001
                    self._last_error = exc
                    self._transition(StreamState.FAILED)
                    self._completion_event.set()
                    return None
                self._transition(StreamState.STREAMING)
        try:
            chunk = next(self._iter)
        except StopIteration:
            self._transition(StreamState.COMPLETE)
            self._completion_event.set()
            return None
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._last_error = exc
            self._transition(StreamState.FAILED)
            self._completion_event.set()
            return None
        with self._lock:
            self._sequence += 1
            self._chunks_emitted += 1
        if self._is_usage_chunk(chunk):
            self._safe_call(self._on_usage, chunk)
        return chunk

    def iterate(self) -> Iterator[Any]:
        """Convenience generator wrapping :meth:`next_chunk`.

        Yields:
            Each chunk until exhaustion / cancellation.
        """
        while True:
            chunk = self.next_chunk()
            if chunk is None:
                return
            yield chunk

    def stop(self) -> None:
        """Signal cancellation. The next :meth:`next_chunk` returns None."""
        self._cancel_event.set()
        with self._lock:
            if self._state in (StreamState.IDLE, StreamState.STREAMING, StreamState.RETRYING):
                self._transition(StreamState.CANCELLED)
        self._completion_event.set()

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Block until the stream reaches a terminal state.

        Args:
            timeout: optional wall-clock timeout (seconds). None waits
                indefinitely.

        Returns:
            True when the stream terminated within the timeout,
            False otherwise.
        """
        return self._completion_event.wait(timeout=timeout)

    # ------------------------------------------------------------------
    # Retry status
    # ------------------------------------------------------------------

    def publish_retry_attempt(
        self,
        *,
        attempt: int,
        max_attempts: int,
        delay_seconds: float,
        error_snippet: str = "",
    ) -> RetryStatus:
        """Surface a retry attempt without re-narrating the request.

        Args:
            attempt: 1-based attempt number.
            max_attempts: total attempts the wrapper will make.
            delay_seconds: pre-retry sleep duration.
            error_snippet: short error string to display.

        Returns:
            The :class:`RetryStatus` that was published.
        """
        status = RetryStatus(
            attempt=attempt,
            max_attempts=max_attempts,
            delay_seconds=max(0.0, float(delay_seconds)),
            error_snippet=(error_snippet or "")[:200],
        )
        with self._lock:
            self._last_retry_status = status
            self._transition(StreamState.RETRYING)
        self._safe_call(self._on_retry, status)
        return status

    def clear_retry_status(self) -> None:
        """Clear the retry-status field once the next chunk arrives."""
        with self._lock:
            self._last_retry_status = None
            if self._state is StreamState.RETRYING:
                self._transition(StreamState.STREAMING)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transition(self, new_state: StreamState) -> None:
        with self._lock:
            if self._state is new_state:
                return
            self._state = new_state
        self._safe_call(self._on_state_change, new_state)

    @staticmethod
    def _is_usage_chunk(chunk: Any) -> bool:
        if isinstance(chunk, StreamChunk):
            return chunk.kind == "usage"
        if isinstance(chunk, dict):
            return chunk.get("type") == "usage" or chunk.get("kind") == "usage"
        return False

    @staticmethod
    def _safe_call(callback: Optional[Callable[..., Any]], *args: Any) -> None:
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:  # noqa: BLE001
            LOGGER.warning("StreamCoordinator callback raised", exc_info=True)


__all__ = [
    "RetryStatus",
    "StreamChunk",
    "StreamCoordinator",
    "StreamState",
]
