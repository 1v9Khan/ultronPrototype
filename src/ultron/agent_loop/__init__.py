"""Agent-loop primitives shared across the voice / coding / supervisor paths.

This package collects small helpers that govern the *outer* agent loop
(loop-detection, tool-signature normalisation, fan-out coordinators) so
each subsystem can opt in without dragging the orchestrator's full
machinery along.
"""

from __future__ import annotations

from .loop_detection import (
    DEFAULT_HARD_THRESHOLD,
    DEFAULT_SOFT_THRESHOLD,
    LoopDetector,
    LoopVerdict,
    tool_call_signature,
)

__all__ = [
    "DEFAULT_HARD_THRESHOLD",
    "DEFAULT_SOFT_THRESHOLD",
    "LoopDetector",
    "LoopVerdict",
    "tool_call_signature",
]
