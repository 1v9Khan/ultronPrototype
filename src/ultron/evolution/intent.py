"""Strict voice-command matcher for the evolution subsystem.

Catalog 13 clean-room. Mirrors the established strict-regex short-circuit
matchers (``feedback.report_intent.match_report_concern`` /
``web_search.deep_research.match_deep_research``): a spoken command only
trips when it explicitly references self-improvement / evolution, so
ordinary conversation never routes here. Returns ``None`` -> the
orchestrator falls through to normal routing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EvolutionCommandKind(str, Enum):
    """The kind of evolution command recognised."""

    RUN_CYCLE = "run_cycle"  # "evolve now", "run evolution", "self-improve"
    STATUS = "status"  # "evolution status", "evolution digest"


@dataclass(frozen=True)
class EvolutionCommand:
    """A matched evolution voice command."""

    kind: EvolutionCommandKind
    raw_text: str


# Status patterns FIRST (a status query also contains "evolution").
_STATUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bevolution\s+(status|digest|report|summary)\b", re.IGNORECASE),
    re.compile(r"\bself[-\s]?improvement\s+(status|digest|report|summary)\b", re.IGNORECASE),
    re.compile(r"\bhow\s+(are|have)\s+you\s+(been\s+)?(evolv|improv)\w*", re.IGNORECASE),
    re.compile(r"\bwhat\s+(have|skills?\s+have)\s+you\s+(learned|distilled|evolved)\b", re.IGNORECASE),
)

_RUN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(run|start|trigger|kick\s*off|do)\s+(an?\s+|a\s+)?evolution(\s+cycle)?\b", re.IGNORECASE),
    re.compile(r"\bevolve\s+(your\s*self|yourself|your\s+skills?|now|a\s+skill|skills?)\b", re.IGNORECASE),
    re.compile(r"\bself[-\s]?improve(\s+now)?\b", re.IGNORECASE),
    re.compile(r"\b(distil|distill)\s+((a|an|new|another)\s+)*skills?\b", re.IGNORECASE),
    re.compile(r"\bimprove\s+yourself(\s+now)?\b", re.IGNORECASE),
)


def match_evolution_command(text: str) -> Optional[EvolutionCommand]:
    """Return an :class:`EvolutionCommand` when ``text`` is an explicit
    evolution command, else ``None``.

    Status is checked before run so "what's the evolution status" is a
    status query, not a run trigger.
    """
    if not text or not text.strip():
        return None
    for pattern in _STATUS_PATTERNS:
        if pattern.search(text):
            return EvolutionCommand(kind=EvolutionCommandKind.STATUS, raw_text=text)
    for pattern in _RUN_PATTERNS:
        if pattern.search(text):
            return EvolutionCommand(kind=EvolutionCommandKind.RUN_CYCLE, raw_text=text)
    return None


__all__ = [
    "EvolutionCommandKind",
    "EvolutionCommand",
    "match_evolution_command",
]
