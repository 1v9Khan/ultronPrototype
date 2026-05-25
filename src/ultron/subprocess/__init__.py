"""Subprocess lifecycle helpers (zombie-killer + per-process registry).

This package collects the cross-cutting subprocess discipline ultron
needs but had scattered across `coding/direct_bridge.py`, the Parakeet
HTTP server spawn, the MCP entry script, and the gaming-mode plugin
toggles. The headline primitive is :class:`ZombieKiller`, a periodic
reaper that enforces a 10-minute hard cap on any subprocess that has
not been explicitly tagged ``persistent``.
"""

from __future__ import annotations

from .kill_tree import (
    DEFAULT_GRACE_SECONDS,
    KillTreeResult,
    MAX_GRACE_SECONDS,
    kill_pid_if_alive,
    kill_process_tree,
)
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
    "DEFAULT_GRACE_SECONDS",
    "DEFAULT_HARD_TIMEOUT_S",
    "DEFAULT_POLL_INTERVAL_S",
    "DEFAULT_WARN_AGE_S",
    "DEFAULT_WARN_RSS_MB",
    "KillTreeResult",
    "MAX_GRACE_SECONDS",
    "TrackedProcess",
    "ZombieKiller",
    "ZombieReport",
    "get_zombie_killer",
    "kill_pid_if_alive",
    "kill_process_tree",
    "reset_zombie_killer_for_testing",
]
