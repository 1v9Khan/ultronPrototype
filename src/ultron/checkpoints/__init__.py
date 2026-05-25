"""Shadow-repo checkpoint system with three-axis restore.

A per-session ``ShadowRepoTracker`` writes commits to a parallel git
repository whose ``core.worktree`` is configured to point AT the
user's workspace but whose ``.git`` directory lives under
``data/checkpoints/<hash>/.git``. After every TOOL-CALL-LIKE bus
event, the orchestrator calls :meth:`ShadowRepoTracker.commit` to
record a workspace snapshot, plus optional rider snapshots of the
voice-memory JSONL stream and any subsystem state the operator opts
in to track.

Restore has three axes:

* ``voice_history`` — truncate the conversation memory + bus event log
  back to a target turn. The workspace is left untouched.
* ``workspace`` — ``git reset --hard <hash>`` the shadow repo. The
  conversation memory is left untouched.
* ``both`` — does both atomically (per-session lock).

The package is intentionally module-level isolated from the safety
validator and the voice path; the orchestrator wires it via the bus.
"""

from __future__ import annotations

from .exclusions import (
    DEFAULT_CHECKPOINT_EXCLUSIONS,
    VOICE_BASELINE_PROTECTED_PATTERNS,
    compose_gitignore,
)
from .restore import (
    RestoreAxis,
    RestoreOutcome,
    RestorePlan,
    plan_restore,
)
from .registry import (
    CheckpointRegistry,
    SessionCheckpointManager,
    get_checkpoint_registry,
    reset_checkpoint_registry_for_testing,
)
from .shadow_repo import (
    CheckpointCommit,
    CheckpointInitError,
    ShadowRepoTracker,
    hash_working_dir,
)

__all__ = [
    "CheckpointCommit",
    "CheckpointInitError",
    "CheckpointRegistry",
    "DEFAULT_CHECKPOINT_EXCLUSIONS",
    "RestoreAxis",
    "RestoreOutcome",
    "RestorePlan",
    "SessionCheckpointManager",
    "ShadowRepoTracker",
    "VOICE_BASELINE_PROTECTED_PATTERNS",
    "compose_gitignore",
    "get_checkpoint_registry",
    "hash_working_dir",
    "plan_restore",
    "reset_checkpoint_registry_for_testing",
]
