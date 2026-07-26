"""2026-05-20 round 6: structured trace context for the voice loop.

Adds a thread-local ``turn_id`` (per-utterance) and ``phase`` (which
stage of the pipeline) so every log line written via the helpers
below carries a ``turn=N phase=X`` prefix. The user can then::

    grep "turn=42" logs/kenning.log

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
pipeline. The :mod:`kenning.trace` helpers just keep the format
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
    "mark",
    "mark_at",
    "latency_stages",
    "latency_line",
    "reset_latency",
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
# Per-turn LATENCY marks (2026-07-26)
# ---------------------------------------------------------------------------
#
# Why this lives here and not on the Orchestrator: the stages that matter are
# spread across threads. Speculative STT and speculative LLM run on background
# daemons that inherit the turn id via ``restore()`` but get a FRESH
# thread-local store -- so a thread-local mark map would silently drop exactly
# the stages we most want to measure. Marks are therefore held in a
# process-global dict keyed by TURN ID, which every participating thread agrees
# on, under a lock.
#
# Bounded to the last few turns so a long stream cannot leak; the report is
# emitted at end-of-turn, well before eviction.
#
# Cost: one perf_counter() + a dict write per stage. Nanoseconds. Always on --
# the whole point is that the NEXT latency question is answerable from the log
# the streamer already has, instead of needing a re-instrumentation pass.

_MARKS_MAX_TURNS = 32
_marks_lock = threading.Lock()
_marks: "dict[int, list[tuple[str, float]]]" = {}


def mark(stage: str) -> None:
    """Timestamp ``stage`` against the current turn. Safe from any thread."""
    mark_at(stage, time.perf_counter())


def mark_at(stage: str, when: float) -> None:
    """Record ``stage`` at an explicit ``perf_counter`` value.

    Used for boundaries captured before the mark point exists -- notably
    speech-end, which the VAD observes inside the capture loop.
    """
    tid = get_turn()
    if tid is None:
        return
    with _marks_lock:
        seq = _marks.get(tid)
        if seq is None:
            if len(_marks) >= _MARKS_MAX_TURNS:
                for old in sorted(_marks)[:len(_marks) - _MARKS_MAX_TURNS + 1]:
                    _marks.pop(old, None)
            seq = _marks[tid] = []
        seq.append((stage, when))


def reset_latency(turn_id: Optional[int] = None) -> None:
    """Drop the marks for a turn (defaults to the current one)."""
    tid = get_turn() if turn_id is None else turn_id
    if tid is None:
        return
    with _marks_lock:
        _marks.pop(tid, None)


def latency_stages(turn_id: Optional[int] = None) -> "list[tuple[str, float, float]]":
    """Return ``(stage, since_first_ms, delta_ms)`` in recorded order.

    ``delta_ms`` is the gap from the PREVIOUS mark, which is what identifies
    the expensive stage; ``since_first_ms`` is cumulative from the first mark.
    """
    tid = get_turn() if turn_id is None else turn_id
    if tid is None:
        return []
    with _marks_lock:
        seq = list(_marks.get(tid, ()))
    if not seq:
        return []
    base = seq[0][1]
    out: "list[tuple[str, float, float]]" = []
    prev = base
    for name, when in seq:
        out.append((name, (when - base) * 1000.0, (when - prev) * 1000.0))
        prev = when
    return out


def latency_line(turn_id: Optional[int] = None) -> str:
    """One-line summary: total plus each stage's own cost, slowest first.

    Shape::

        total=1592ms | stt=812 | tts_synth=430 | llm=210 | route=9 ...

    Empty string when the turn recorded fewer than two marks (nothing to
    compare), so callers can skip the log line entirely.
    """
    stages = latency_stages(turn_id)
    if len(stages) < 2:
        return ""
    total = stages[-1][1]
    costs = [(name, delta) for name, _, delta in stages[1:]]
    costs.sort(key=lambda kv: kv[1], reverse=True)
    body = " | ".join(f"{name}={delta:.0f}" for name, delta in costs)
    return f"total={total:.0f}ms | {body}"


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
