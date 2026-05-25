"""Bounded sliding-window writer with debounce + disk spillover.

Adapted from cline's ``CommandOrchestrator`` line-by-line buffer +
``MAX_LINES_BEFORE_FILE`` / ``MAX_BYTES_BEFORE_FILE`` spillover
pattern (Apache 2.0; see ``THIRD_PARTY_NOTICES.md``). Ultron's
variant generalises beyond terminal output to any streaming source:

* ``WindowedOutputWriter.feed_line(line)`` accumulates into a bounded
  in-memory buffer (default 20 lines / 2 KB) and flushes on
  configurable debounce.
* When the total grows past :data:`DEFAULT_SPILL_LINE_THRESHOLD` lines
  OR :data:`DEFAULT_SPILL_BYTE_THRESHOLD` bytes, subsequent writes go
  to a temp file under ``data/streaming-overflow/<session-id>.txt``
  and the in-memory view collapses to "(first N lines) ... (X lines
  written to <path>) ... (last N lines)".
* :func:`is_compiling_output` mirrors cline's COMPILING_MARKERS check
  so callers can double their "hot timeout" when a long-running tool
  is legitimately producing no output (build / bundler).

The writer is intentionally callback-driven (callers register
``on_flush`` to receive each batched chunk) so it composes with the
:class:`PresentationScheduler` without hard-coupling.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

LOGGER = logging.getLogger(__name__)

#: Default number of lines buffered before a forced flush.
DEFAULT_LINE_BUDGET: int = 20

#: Default byte size buffered before a forced flush.
DEFAULT_BYTE_BUDGET: int = 2 * 1024

#: Default debounce window between flushes (milliseconds).
DEFAULT_DEBOUNCE_MS: int = 100

#: Default head/tail width preserved when overflow spills to disk.
DEFAULT_HEAD_TAIL_LINES: int = 100

#: Default total-line threshold above which the writer spills to disk.
DEFAULT_SPILL_LINE_THRESHOLD: int = 1000

#: Default total-byte threshold above which the writer spills to disk.
DEFAULT_SPILL_BYTE_THRESHOLD: int = 512 * 1024

#: Sub-strings that indicate a long-running compile / bundle / build is
#: in progress (legitimately produces no output for many seconds).
#: Lowercase substring match; mirrors cline's heuristic.
COMPILING_MARKERS: tuple[str, ...] = (
    "compiling",
    "building",
    "bundling",
    "transpiling",
    "generating",
    "linking",
    "minifying",
    "packaging",
    "indexing",
    "loading model",
)


def is_compiling_output(line: str) -> bool:
    """Return True when ``line`` matches a long-running-build marker."""
    if not line:
        return False
    lowered = line.lower()
    return any(marker in lowered for marker in COMPILING_MARKERS)


@dataclass(frozen=True)
class WindowSnapshot:
    """Frozen view of the writer's current state.

    Attributes:
        head_lines: first ``head_tail_lines`` lines of the stream.
        tail_lines: most-recent ``head_tail_lines`` lines.
        spilled_line_count: number of lines elided to the overflow file.
        spilled_bytes: number of bytes elided to the overflow file.
        overflow_path: path to the spillover file (None when not spilled).
        total_lines: total lines fed to the writer over its lifetime.
        total_bytes: total bytes fed to the writer over its lifetime.
        spilled: True when overflow has begun.
    """

    head_lines: tuple[str, ...]
    tail_lines: tuple[str, ...]
    spilled_line_count: int
    spilled_bytes: int
    overflow_path: Optional[Path]
    total_lines: int
    total_bytes: int
    spilled: bool

    def render(self) -> str:
        """Render the snapshot as a head + elision-marker + tail string.

        Suitable for prompt injection or audit-log capture.
        """
        if not self.spilled:
            return "\n".join(self.head_lines + self.tail_lines)
        head_text = "\n".join(self.head_lines)
        tail_text = "\n".join(self.tail_lines)
        marker = (
            f"... ({self.spilled_line_count} lines, "
            f"{self.spilled_bytes} bytes elided; "
            f"see {self.overflow_path.name if self.overflow_path else 'overflow file'}) ..."
        )
        parts = [s for s in (head_text, marker, tail_text) if s]
        return "\n".join(parts)


class WindowedOutputWriter:
    """Bounded sliding-window writer with debounce + spillover.

    Args:
        on_flush: callback invoked with each flushed chunk (the chunk is
            a newline-joined string of the buffered lines). The writer
            never holds the lock while invoking the callback.
        line_budget: max lines buffered before a forced flush.
        byte_budget: max bytes buffered before a forced flush.
        debounce_ms: minimum milliseconds between flushes (debounce).
        head_tail_lines: number of lines preserved at head + tail when
            overflow spills.
        spill_line_threshold: total-line threshold for disk spillover.
        spill_byte_threshold: total-byte threshold for disk spillover.
        overflow_dir: directory where overflow files are written.
            Default ``data/streaming-overflow/``. The directory is
            created on first spillover.
        overflow_label: optional label included in the overflow file
            name (helps disambiguate concurrent windows).
        clock: optional monotonic clock callable (test hook).
    """

    def __init__(
        self,
        on_flush: Optional[Callable[[str], None]] = None,
        *,
        line_budget: int = DEFAULT_LINE_BUDGET,
        byte_budget: int = DEFAULT_BYTE_BUDGET,
        debounce_ms: int = DEFAULT_DEBOUNCE_MS,
        head_tail_lines: int = DEFAULT_HEAD_TAIL_LINES,
        spill_line_threshold: int = DEFAULT_SPILL_LINE_THRESHOLD,
        spill_byte_threshold: int = DEFAULT_SPILL_BYTE_THRESHOLD,
        overflow_dir: Optional[Path] = None,
        overflow_label: str = "stream",
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        if line_budget < 1 or byte_budget < 1:
            raise ValueError("line_budget and byte_budget must be >= 1")
        if head_tail_lines < 1:
            raise ValueError("head_tail_lines must be >= 1")
        self._on_flush = on_flush
        self._line_budget = line_budget
        self._byte_budget = byte_budget
        self._debounce_ms = max(0, int(debounce_ms))
        self._head_tail_lines = head_tail_lines
        self._spill_line_threshold = spill_line_threshold
        self._spill_byte_threshold = spill_byte_threshold
        self._overflow_dir = overflow_dir
        self._overflow_label = overflow_label
        self._clock = clock or time.monotonic

        self._lock = threading.RLock()
        self._buffer: list[str] = []
        self._buffer_bytes: int = 0
        self._last_flush_at: float = 0.0
        self._has_flushed_once: bool = False
        self._total_lines: int = 0
        self._total_bytes: int = 0
        self._head: list[str] = []
        self._tail: list[str] = []
        self._spilled: bool = False
        self._spilled_lines: int = 0
        self._spilled_bytes: int = 0
        self._overflow_path: Optional[Path] = None
        self._overflow_handle = None

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def feed_line(self, line: str) -> bool:
        """Append ``line`` to the buffer.

        Args:
            line: textual line (newline terminator NOT required; the
                writer adds it on flush).

        Returns:
            True when this call triggered an immediate flush (line /
            byte budget exceeded), False when the line was simply
            appended.
        """
        if line is None:
            return False
        normalised = line.rstrip("\r\n")
        line_bytes = len(normalised.encode("utf-8")) + 1  # +1 for newline
        with self._lock:
            self._total_lines += 1
            self._total_bytes += line_bytes
            # Update head / tail summary windows.
            if len(self._head) < self._head_tail_lines:
                self._head.append(normalised)
            self._tail.append(normalised)
            if len(self._tail) > self._head_tail_lines:
                # Evict oldest tail line.
                evicted = self._tail.pop(0)
                if self._spilled:
                    self._spilled_lines += 1
                    self._spilled_bytes += len(evicted.encode("utf-8")) + 1
            # Append to the active buffer.
            self._buffer.append(normalised)
            self._buffer_bytes += line_bytes
            # Trigger spillover if total exceeds thresholds.
            if (
                not self._spilled
                and (
                    self._total_lines > self._spill_line_threshold
                    or self._total_bytes > self._spill_byte_threshold
                )
            ):
                self._begin_spillover()
            # Flush on budget exhaustion.
            if (
                len(self._buffer) >= self._line_budget
                or self._buffer_bytes >= self._byte_budget
            ):
                self._flush_locked(force=True)
                return True
            return False

    def maybe_flush(self) -> bool:
        """Flush IF the debounce window has elapsed and buffer is non-empty.

        Returns:
            True when a flush occurred, False otherwise.

        Notes:
            The first flush of a fresh writer always passes — the
            ``_last_flush_at`` sentinel of 0 is treated as "never
            flushed". Subsequent calls enforce the debounce window.
        """
        with self._lock:
            if not self._buffer:
                return False
            if self._has_flushed_once:
                now_ms = self._clock() * 1000
                if (now_ms - self._last_flush_at * 1000) < self._debounce_ms:
                    return False
            self._flush_locked(force=False)
            return True

    def flush(self) -> bool:
        """Force-flush the buffer regardless of debounce.

        Returns:
            True when a flush occurred (buffer was non-empty), False
            otherwise.
        """
        with self._lock:
            if not self._buffer:
                return False
            self._flush_locked(force=True)
            return True

    def close(self) -> None:
        """Flush remaining content + close the overflow file handle."""
        with self._lock:
            if self._buffer:
                self._flush_locked(force=True)
            if self._overflow_handle is not None:
                try:
                    self._overflow_handle.close()
                except Exception:  # noqa: BLE001
                    pass
                self._overflow_handle = None

    def snapshot(self) -> WindowSnapshot:
        """Return a frozen :class:`WindowSnapshot` of the current state."""
        with self._lock:
            return WindowSnapshot(
                head_lines=tuple(self._head),
                tail_lines=tuple(self._tail) if self._spilled else (),
                spilled_line_count=self._spilled_lines,
                spilled_bytes=self._spilled_bytes,
                overflow_path=self._overflow_path,
                total_lines=self._total_lines,
                total_bytes=self._total_bytes,
                spilled=self._spilled,
            )

    def total_lines(self) -> int:
        with self._lock:
            return self._total_lines

    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def spilled(self) -> bool:
        with self._lock:
            return self._spilled

    def overflow_path(self) -> Optional[Path]:
        with self._lock:
            return self._overflow_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_locked(self, *, force: bool) -> None:
        """Flush the buffer; invoke ``on_flush`` (lock released first).

        Must be called with ``self._lock`` held; will release the lock
        around the callback to prevent deadlock on re-entrant calls.
        """
        if not self._buffer:
            return
        rendered = "\n".join(self._buffer)
        self._buffer = []
        self._buffer_bytes = 0
        self._last_flush_at = self._clock()
        self._has_flushed_once = True
        # Also persist to the overflow file when spilled.
        if self._spilled and self._overflow_handle is not None:
            try:
                self._overflow_handle.write(rendered + "\n")
                self._overflow_handle.flush()
            except Exception:  # noqa: BLE001
                LOGGER.warning(
                    "overflow file write failed for %s", self._overflow_path,
                    exc_info=True,
                )
        callback = self._on_flush
        if callback is None:
            return
        # Release the lock around the callback so re-entrant calls are safe.
        self._lock.release()
        try:
            callback(rendered)
        except Exception:  # noqa: BLE001
            LOGGER.warning("on_flush callback raised", exc_info=True)
        finally:
            self._lock.acquire()

    def _begin_spillover(self) -> None:
        """Transition into spillover mode + open the overflow file."""
        self._spilled = True
        if self._overflow_dir is None:
            self._overflow_dir = (
                Path(os.environ.get("ULTRON_PROJECT_ROOT", "."))
                / "data" / "streaming-overflow"
            )
        try:
            self._overflow_dir.mkdir(parents=True, exist_ok=True)
            ts = int(self._clock() * 1000)
            self._overflow_path = (
                self._overflow_dir / f"{self._overflow_label}-{ts}.txt"
            )
            self._overflow_handle = self._overflow_path.open(
                "w", encoding="utf-8", newline="\n",
            )
            # Drop a header so the file is self-describing.
            self._overflow_handle.write(
                f"# ultron streaming overflow — label={self._overflow_label} "
                f"started={ts}\n",
            )
            self._overflow_handle.flush()
        except Exception:  # noqa: BLE001
            LOGGER.warning(
                "could not open overflow file under %s; continuing in-memory only",
                self._overflow_dir,
                exc_info=True,
            )
            self._overflow_handle = None
            self._overflow_path = None


__all__ = [
    "COMPILING_MARKERS",
    "DEFAULT_BYTE_BUDGET",
    "DEFAULT_DEBOUNCE_MS",
    "DEFAULT_HEAD_TAIL_LINES",
    "DEFAULT_LINE_BUDGET",
    "DEFAULT_SPILL_BYTE_THRESHOLD",
    "DEFAULT_SPILL_LINE_THRESHOLD",
    "WindowSnapshot",
    "WindowedOutputWriter",
    "is_compiling_output",
]
