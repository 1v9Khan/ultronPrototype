"""Streaming primitives — windowed writers, presentation scheduling, coordinators.

The package collects the cross-cutting "how do we emit chunks safely
under varying cadence + size pressure" primitives that ultron's voice
path, web-search reader, supervisor narration, and coding-bridge
stdout all need but had each implemented separately.

* :mod:`window` — bounded sliding-window writer with debounce + disk
  spillover.
* :mod:`presentation_scheduler` — priority-banded chunk scheduler that
  adapts cadence by environment (Bluetooth vs local PortAudio, etc.).
* :mod:`reasoning_stream` — reasoning-vs-text chunk demultiplexer
  with the "first text finalises reasoning" discipline.
* :mod:`coordinator` — stream coordinator + retry-status surface so
  retries surface as updates rather than re-narration.
"""

from __future__ import annotations

from .coordinator import (
    RetryStatus,
    StreamChunk,
    StreamCoordinator,
    StreamState,
)
from .presentation_scheduler import (
    Cadence,
    PresentationPriority,
    PresentationScheduler,
    detect_audio_profile,
)
from .reasoning_stream import (
    ReasoningChunkEvent,
    ReasoningDemultiplexer,
    ReasoningFinalisedEvent,
)
from .window import (
    DEFAULT_BYTE_BUDGET,
    DEFAULT_DEBOUNCE_MS,
    DEFAULT_HEAD_TAIL_LINES,
    DEFAULT_LINE_BUDGET,
    DEFAULT_SPILL_BYTE_THRESHOLD,
    DEFAULT_SPILL_LINE_THRESHOLD,
    COMPILING_MARKERS,
    WindowedOutputWriter,
    WindowSnapshot,
    is_compiling_output,
)

__all__ = [
    "COMPILING_MARKERS",
    "Cadence",
    "DEFAULT_BYTE_BUDGET",
    "DEFAULT_DEBOUNCE_MS",
    "DEFAULT_HEAD_TAIL_LINES",
    "DEFAULT_LINE_BUDGET",
    "DEFAULT_SPILL_BYTE_THRESHOLD",
    "DEFAULT_SPILL_LINE_THRESHOLD",
    "PresentationPriority",
    "PresentationScheduler",
    "ReasoningChunkEvent",
    "ReasoningDemultiplexer",
    "ReasoningFinalisedEvent",
    "RetryStatus",
    "StreamChunk",
    "StreamCoordinator",
    "StreamState",
    "WindowSnapshot",
    "WindowedOutputWriter",
    "detect_audio_profile",
    "is_compiling_output",
]
