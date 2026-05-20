"""2026-05-20 round 6: structured trace context for the voice loop.

Adds a thread-local ``turn_id`` (per-utterance) and ``phase`` (which
stage of the pipeline) so every log line written via the helpers
below carries a ``turn=N phase=X`` prefix. The user can then::

    grep "turn=42" logs/ultron.log

and see the complete lifecycle of a single user utterance: wake
detect, capture, VAD, STT, addressing, routing, gate, memory
retrieve, LLM call, TTS synth, playback, memory write -- in order,
with timings and key/value details.

Design notes:

* **Thread-local state** so the orchestrator main loop sets the
  turn id once at the top of each iteration and every downstream
  call automatically inherits. Background threads (speculative STT
  / LLM / RAG prefetch) inherit by reading the parent thread's
  state via :func:`copy_to_thread`.
* **Helpers, not a class hierarchy** -- ``tlog`` formats a single
  line with ``key=value`` pairs and the current turn/phase. ``phase``
  is a context manager that bookmarks an entry / exit pair with
  elapsed milliseconds.
* **Cheap when disabled** -- the helpers degrade to no-op string
  formatting when the underlying logger is not enabled at the
  caller's level.

The helpers are deliberately small: most of the value comes from
*consistent placement* of log lines at every decision point in the
pipeline. The :mod:`ultron.trace` helpers just keep the format
uniform so the resulting log is grep-able and scannable.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

__all__ = [
    "set_turn",
    "get_turn",
    "next_turn",
    "set_phase",
    "get_phase",
    "fmt",
    "tlog",
    "phase",
    "snapshot",
    "restore",
]


_state = threading.local()
_turn_counter_lock = threading.Lock()
_turn_counter: int = 0


# ---------------------------------------------------------------------------
# Turn id (per-utterance identifier that flows through every log)
# ---------------------------------------------------------------------------


def set_turn(turn_id: Optional[int]) -> None:
    """Set the per-thread turn id used by every subsequent tlog call.

    Pass ``None`` to clear (start-up phase logs have no turn).
    """
    _state.turn = turn_id


def get_turn() -> Optional[int]:
    """Return the current turn id, or ``None`` when unset."""
    return getattr(_state, "turn", None)


def next_turn() -> int:
    """Allocate a fresh turn id and install it on the current thread.

    The counter is process-global and monotonic. Callers (typically
    the orchestrator main loop) invoke this at the top of every new
    voice-loop iteration so the turn id increments per user utterance.

    Returns the freshly-installed id so callers can include it in
    their own bookkeeping (e.g. observation rows).
    """
    global _turn_counter
    with _turn_counter_lock:
        _turn_counter += 1
        tid = _turn_counter
    set_turn(tid)
    return tid


# ---------------------------------------------------------------------------
# Phase (which pipeline stage are we in)
# ---------------------------------------------------------------------------


def set_phase(phase_name: Optional[str]) -> None:
    """Set the per-thread phase tag (e.g. ``"capture"``, ``"stt"``)."""
    _state.phase = phase_name


def get_phase() -> Optional[str]:
    """Return the current phase tag, or ``None`` when unset."""
    return getattr(_state, "phase", None)


# ---------------------------------------------------------------------------
# Cross-thread propagation
# ---------------------------------------------------------------------------


def snapshot() -> dict:
    """Return a dict capturing the current thread's trace state.

    Background threads (speculative STT / LLM / RAG prefetch /
    XTTS synth worker / etc.) capture the parent's state via this
    helper then call :func:`restore` on entry so their log lines
    carry the same turn id.
    """
    return {
        "turn": get_turn(),
        "phase": get_phase(),
    }


def restore(state: dict) -> None:
    """Install a previously-snapshotted state on the current thread."""
    set_turn(state.get("turn"))
    set_phase(state.get("phase"))


# ---------------------------------------------------------------------------
# Format helper -- builds the structured prefix
# ---------------------------------------------------------------------------


def _fmt_value(v: Any) -> str:
    """Render a value for inclusion in a structured log line.

    Strings are wrapped in single quotes (so spaces stay readable).
    Numerics + bools render bare. Containers are coerced to ``repr``
    and truncated to keep the log line scannable.
    """
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)
    if isinstance(v, str):
        # Truncate long strings; preserve newlines as a visible marker.
        s = v.replace("\n", "\\n").replace("\r", "\\r")
        if len(s) > 200:
            s = s[:200] + "..."
        return f"'{s}'"
    rep = repr(v)
    if len(rep) > 200:
        rep = rep[:200] + "..."
    return rep


def fmt(msg: str, **kwargs: Any) -> str:
    """Format a log line with the current turn/phase prefix + kwargs.

    Output shape::

        turn=42 phase=stt | msg=<message> | k1=v1 | k2=v2

    When no turn / phase is set, the prefix sections are omitted so
    pre-loop logs still read cleanly.
    """
    parts: list[str] = []
    tid = get_turn()
    if tid is not None:
        parts.append(f"turn={tid}")
    ph = get_phase()
    if ph is not None:
        parts.append(f"phase={ph}")
    if msg:
        parts.append(msg)
    for k, v in kwargs.items():
        parts.append(f"{k}={_fmt_value(v)}")
    return " | ".join(parts)


def tlog(
    log: logging.Logger,
    msg: str,
    *,
    level: int = logging.INFO,
    **kwargs: Any,
) -> None:
    """Emit a structured log line via the given logger.

    Skips the format work entirely when the logger is not enabled at
    ``level`` so callers can pile on detail without paying for it
    when the level is filtered out.
    """
    if not log.isEnabledFor(level):
        return
    log.log(level, fmt(msg, **kwargs))


# ---------------------------------------------------------------------------
# Phase context manager (bookmarks start / end with elapsed timing)
# ---------------------------------------------------------------------------


@contextmanager
def phase(
    name: str,
    *,
    log: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    **kwargs: Any,
) -> Iterator[dict]:
    """Bracket a pipeline phase with start / end log lines.

    Usage::

        with phase("stt", log=logger, audio_s=duration):
            transcript = stt.transcribe(audio)
            # phase tag is "stt" inside this block; any tlog() call
            # downstream picks it up via get_phase().

    Yields a mutable dict so the body can stash extra fields onto
    the eventual "end" log line::

        with phase("llm", log=logger) as ctx:
            text = llm.generate(prompt)
            ctx["chars"] = len(text)
        # END line will include chars=<n>

    The phase tag is installed for the duration of the block and
    restored to the prior value on exit (supports nesting).
    """
    prior_phase = get_phase()
    set_phase(name)
    t0 = time.monotonic()
    extra: dict = {}
    if log is not None and log.isEnabledFor(level):
        log.log(level, fmt(f"{name}:start", **kwargs))
    try:
        yield extra
    finally:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        if log is not None and log.isEnabledFor(level):
            log.log(
                level,
                fmt(f"{name}:end", elapsed_ms=elapsed_ms, **{**kwargs, **extra}),
            )
        set_phase(prior_phase)
