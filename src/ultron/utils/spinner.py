"""Pre-rendered ASCII spinner with bounce + Unicode autodetect.

Pattern lifted in spirit (not in source) from aider's ``waiting.py``
(Apache 2.0; see ``THIRD_PARTY_NOTICES.md``).

A spinner for TTY output during long-running operations (an LLM
call, a test sweep, a model download). Three details mirror aider's
implementation because they're load-bearing for the UX:

  1. **Frame pre-rendering.** The 18 frames are computed once at
     construction; ``step()`` just writes the next one. No string
     building under display latency.

  2. **Bounce.** Frames 0-9 slide a "scanner" character left-to-right
     across a 10-cell track; frames 10-17 slide it back. The track
     length is intentionally a constant so the cursor positioning
     stays predictable.

  3. **Continuity across instances.** A class-level
     ``last_frame_index`` is preserved between spinner instances —
     when a complex flow re-instantiates the spinner several times,
     the visual stays a single continuous animation rather than
     restarting each time.

Other niceties:

  * **Unicode autodetect.** On construction we try writing the
    block-character / scanner pair (``░``, ``█``); on
    ``UnicodeEncodeError`` we fall back to ASCII (``=``, ``#``).
    Catalog says this catches the cp1252 console case on Windows.
  * **First-frame delay.** No output for the first 0.5 s of a
    spinner's life. Fast operations don't flicker a visible spinner
    at all.
  * **Update throttle.** 0.1 s minimum between visible updates so
    fast tight loops don't hammer the terminal.
  * **Context-manager support.** ``with Spinner("Loading") as s:
    ...`` cleans up cursor + final newline.
  * **Width truncation.** Output line truncated to ``terminal_width
    - 2`` so the spinner never wraps.
  * **Daemon thread wrapper** (:class:`WaitingSpinner`) for "spin
    while task X runs" — start it, do the work, ``stop()``.

This module is stdlib-only; no `tqdm`, no curses. Safe to call from
any context including subprocess wrappers.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from typing import List, Optional, TextIO


# Track length: the scanner moves across this many cells before
# bouncing back. Matches aider's value; chosen so the full frame is
# visually distinct without dominating the terminal width.
TRACK_LENGTH = 10


# Visible-frame delay. Operations completing within this window
# render no spinner at all (no flicker).
DEFAULT_FIRST_FRAME_DELAY_SECONDS = 0.5


# Minimum interval between visible frame updates. Tight loops calling
# step() many times per second collapse to one update per interval.
DEFAULT_UPDATE_INTERVAL_SECONDS = 0.1


_UNICODE_CHARS = ("░", "█")  # track, scanner
_ASCII_CHARS = ("=", "#")


class Spinner:
    """Pre-rendered ASCII spinner with bounce + Unicode autodetect.

    Args:
        message: Text shown to the left of the spinner.
        stream: Output stream (defaults to sys.stdout). Pass a
            io.StringIO in tests to capture output.
        first_frame_delay: Seconds before the first visible frame
            (fast operations stay invisible).
        update_interval: Minimum seconds between visible updates.
        track_length: Number of cells the scanner moves across.
        force_ascii: Skip Unicode autodetection.

    Class state ``last_frame_index`` persists across instances so
    re-instantiating the spinner mid-flow continues the animation
    seamlessly. Reset via :meth:`reset_continuity`.
    """

    # Catalog detail: visual continuity across short-lived instances.
    last_frame_index: int = 0

    def __init__(
        self,
        message: str = "",
        *,
        stream: Optional[TextIO] = None,
        first_frame_delay: float = DEFAULT_FIRST_FRAME_DELAY_SECONDS,
        update_interval: float = DEFAULT_UPDATE_INTERVAL_SECONDS,
        track_length: int = TRACK_LENGTH,
        force_ascii: bool = False,
    ) -> None:
        self._message = str(message)
        self._stream = stream if stream is not None else sys.stdout
        self._first_frame_delay = float(first_frame_delay)
        self._update_interval = float(update_interval)
        self._track_length = max(2, int(track_length))
        self._start_time: Optional[float] = None
        self._last_update_time: float = 0.0
        self._frame_index = Spinner.last_frame_index
        self._first_frame_rendered = False
        self._chars = self._detect_charset(force_ascii)
        self._frames = self._build_frames()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Mark the start time. Idempotent — re-starting is a no-op."""
        if self._start_time is None:
            self._start_time = time.monotonic()
            self._last_update_time = self._start_time

    def step(self, message: Optional[str] = None) -> None:
        """Advance one frame. Honors first-frame delay + update interval.

        Args:
            message: Optionally update the message text shown to the
                left of the spinner. When None, the previous message
                is kept.
        """
        if message is not None:
            self._message = str(message)
        if self._start_time is None:
            self.start()
        now = time.monotonic()
        if not self._first_frame_rendered:
            if (now - (self._start_time or now)) < self._first_frame_delay:
                return
            self._first_frame_rendered = True
        if (now - self._last_update_time) < self._update_interval:
            return
        self._render_current_frame()
        self._last_update_time = now
        self._frame_index = (self._frame_index + 1) % len(self._frames)
        Spinner.last_frame_index = self._frame_index

    def end(self) -> None:
        """Clear the spinner line + emit a newline."""
        if not self._first_frame_rendered:
            return
        try:
            width = self._terminal_width()
            self._stream.write("\r" + " " * width + "\r")
            self._stream.flush()
        except (OSError, ValueError):
            pass

    @classmethod
    def reset_continuity(cls) -> None:
        """Reset ``last_frame_index`` so the next instance starts fresh."""
        cls.last_frame_index = 0

    # Context-manager hooks ------------------------------------------------

    def __enter__(self) -> "Spinner":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _detect_charset(self, force_ascii: bool) -> tuple[str, str]:
        if force_ascii:
            return _ASCII_CHARS
        try:
            # Try to encode the Unicode pair against the stream's encoding.
            track, scanner = _UNICODE_CHARS
            test_text = track + scanner
            encoding = getattr(self._stream, "encoding", None) or "utf-8"
            test_text.encode(encoding)
            return _UNICODE_CHARS
        except (UnicodeEncodeError, LookupError, AttributeError):
            return _ASCII_CHARS

    def _build_frames(self) -> List[str]:
        track_char, scanner_char = self._chars
        frames: List[str] = []
        # Forward (positions 0..track_length-1)
        for pos in range(self._track_length):
            frames.append(
                track_char * pos + scanner_char
                + track_char * (self._track_length - pos - 1)
            )
        # Backward (positions track_length-2..1) — skip endpoints to
        # avoid stuttering at the bounce.
        for pos in range(self._track_length - 2, 0, -1):
            frames.append(
                track_char * pos + scanner_char
                + track_char * (self._track_length - pos - 1)
            )
        return frames

    def _render_current_frame(self) -> None:
        frame = self._frames[self._frame_index]
        prefix = f"{self._message} " if self._message else ""
        line = prefix + frame
        max_width = max(2, self._terminal_width() - 2)
        if len(line) > max_width:
            line = line[:max_width]
        try:
            self._stream.write("\r" + line)
            self._stream.flush()
        except (OSError, ValueError):
            pass

    def _terminal_width(self) -> int:
        try:
            return shutil.get_terminal_size((80, 24)).columns
        except (OSError, ValueError):
            return 80


class WaitingSpinner:
    """Daemon-thread wrapper that animates a :class:`Spinner` while a
    blocking operation runs.

    Usage::

        with WaitingSpinner("Loading model"):
            heavy_blocking_call()

    Args:
        message: Forwarded to the underlying :class:`Spinner`.
        tick_interval: Seconds between automatic ``step()`` calls.
        stream: Forwarded to the underlying :class:`Spinner`.
        first_frame_delay: Forwarded.
    """

    def __init__(
        self,
        message: str = "",
        *,
        tick_interval: float = 0.1,
        stream: Optional[TextIO] = None,
        first_frame_delay: float = DEFAULT_FIRST_FRAME_DELAY_SECONDS,
    ) -> None:
        self._spinner = Spinner(
            message,
            stream=stream,
            first_frame_delay=first_frame_delay,
        )
        self._tick_interval = float(tick_interval)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._spinner.start()
        self._thread = threading.Thread(
            target=self._loop, name="ultron-spinner", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop_event.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        self._thread = None
        self._spinner.end()

    def __enter__(self) -> "WaitingSpinner":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._spinner.step()
            if self._stop_event.wait(self._tick_interval):
                return


__all__ = [
    "DEFAULT_FIRST_FRAME_DELAY_SECONDS",
    "DEFAULT_UPDATE_INTERVAL_SECONDS",
    "Spinner",
    "TRACK_LENGTH",
    "WaitingSpinner",
]
