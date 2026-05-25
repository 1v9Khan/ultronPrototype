"""Subprocess lifecycle helpers (zombie-killer + per-process registry).

This package collects the cross-cutting subprocess discipline ultron
needs but had scattered across `coding/direct_bridge.py`, the Parakeet
HTTP server spawn, the MCP entry script, and the gaming-mode plugin
toggles. The headline primitive is :class:`ZombieKiller`, a periodic
reaper that enforces a 10-minute hard cap on any subprocess that has
not been explicitly tagged ``persistent``.
"""

from __future__ import annotations

from .zombie_killer import (
    DEFAULT_HARD_TIMEOUT_S,
    DEFAULT_POLL_INTERVAL_S,
    DEFAULT_WARN_RSS_MB,
    DEFAULT_WARN_AGE_S,
    ZombieKiller,
    ZombieReport,
    TrackedProcess,
    get_zombie_killer,
    reset_zombie_killer_for_testing,
)

__all__ = [
    "DEFAULT_HARD_TIMEOUT_S",
    "DEFAULT_POLL_INTERVAL_S",
    "DEFAULT_WARN_AGE_S",
    "DEFAULT_WARN_RSS_MB",
    "TrackedProcess",
    "ZombieKiller",
    "ZombieReport",
    "get_zombie_killer",
    "reset_zombie_killer_for_testing",
]
