"""Per-session checkpoint manager + module-level registry.

The :class:`SessionCheckpointManager` wraps one
:class:`ShadowRepoTracker` together with the bus subscriptions that
drive it. The :class:`CheckpointRegistry` is the module-level
singleton the orchestrator addresses: ``registry.get_or_create(...)``
returns a per-session manager, and the
``register_event_kind / unregister_event_kind`` knobs let the
operator choose which canonical event kinds trigger commits.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .exclusions import compose_gitignore
from .restore import (
    EventLogTruncator,
    RestoreAxis,
    RestoreOutcome,
    RestorePlan,
    VoiceHistoryTruncator,
    WorkspaceReset,
    execute_restore,
    plan_restore,
)
from .shadow_repo import CheckpointCommit, ShadowRepoTracker, hash_working_dir

LOGGER = logging.getLogger(__name__)

#: Canonical event-kind names whose firing triggers a commit when
#: registered. Mirrors the bus event vocabulary the rest of ultron
#: publishes.
DEFAULT_EVENT_KINDS: frozenset[str] = frozenset({
    "CodingFileChangedEvent",
    "ProjectIndexedEvent",
    "ProjectDigestGeneratedEvent",
    "SafetyViolatedEvent",
    "MemoryRetrievedEvent",
})


@dataclass
class SessionCheckpointManager:
    """Per-session checkpoint state (tracker + event-kind subscriptions).

    Args:
        tracker: the underlying :class:`ShadowRepoTracker`.
        triggered_event_kinds: set of event-kind names that produce
            commits when received via :meth:`on_event`.
        workspace_reset: optional callable used by :meth:`restore` for
            the workspace axis (typically wraps
            :meth:`ShadowRepoTracker.hard_reset`).
        voice_history_truncate: optional callable used by
            :meth:`restore` for the voice-history axis.
        event_log_truncate: optional callable used alongside the
            voice-history axis to drop bus events.
    """

    tracker: ShadowRepoTracker
    triggered_event_kinds: set[str] = field(default_factory=lambda: set(DEFAULT_EVENT_KINDS))
    workspace_reset: Optional[WorkspaceReset] = None
    voice_history_truncate: Optional[VoiceHistoryTruncator] = None
    event_log_truncate: Optional[EventLogTruncator] = None
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _commits: list[CheckpointCommit] = field(default_factory=list, init=False, repr=False)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def register_event_kind(self, kind: str) -> None:
        with self._lock:
            self.triggered_event_kinds.add(kind)

    def unregister_event_kind(self, kind: str) -> None:
        with self._lock:
            self.triggered_event_kinds.discard(kind)

    def configured_event_kinds(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self.triggered_event_kinds))

    # ------------------------------------------------------------------
    # Event ingest
    # ------------------------------------------------------------------

    def on_event(
        self,
        kind: str,
        *,
        extra: str = "",
        force: bool = False,
    ) -> Optional[CheckpointCommit]:
        """Commit if ``kind`` is in the trigger set (or ``force=True``).

        Args:
            kind: event kind name (as published on the bus).
            extra: optional body appended after the commit headline.
            force: bypass the trigger-kind filter (used by the operator's
                explicit "checkpoint now" voice intent).
        """
        with self._lock:
            if not force and kind not in self.triggered_event_kinds:
                return None
        commit = self.tracker.commit(extra_message=f"event={kind} {extra}".strip())
        if commit is not None:
            with self._lock:
                self._commits.append(commit)
        return commit

    def commits(self) -> tuple[CheckpointCommit, ...]:
        """Return the chronological list of commits seen this session."""
        with self._lock:
            return tuple(self._commits)

    def head_commit(self) -> Optional[CheckpointCommit]:
        with self._lock:
            if not self._commits:
                return None
            return self._commits[-1]

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    def restore(self, plan: RestorePlan) -> RestoreOutcome:
        """Apply ``plan`` using this manager's configured callables.

        The default workspace reset wraps
        :meth:`ShadowRepoTracker.hard_reset` when the caller did not
        wire an explicit ``workspace_reset``.
        """
        reset_fn = self.workspace_reset or (
            lambda commit_hash: self.tracker.hard_reset(commit_hash)
        )
        return execute_restore(
            plan,
            workspace_reset=reset_fn,
            voice_history_truncate=self.voice_history_truncate,
            event_log_truncate=self.event_log_truncate,
        )

    def plan_voice_history_undo(
        self,
        *,
        offset: int = 1,
        after_turn_id: str = "",
        will_drop_event_count: int = 0,
    ) -> RestorePlan:
        """Build the canonical "undo the last N voice turns" plan."""
        return plan_restore(
            axis=RestoreAxis.VOICE_HISTORY,
            truncate_after_turn_id=after_turn_id,
            will_drop_turn_count=max(0, int(offset)),
            will_drop_event_count=max(0, int(will_drop_event_count)),
        )

    def plan_workspace_rewind(
        self,
        *,
        target_commit_hash: str = "",
    ) -> RestorePlan:
        """Build the canonical "rewind workspace to commit X" plan."""
        return plan_restore(
            axis=RestoreAxis.WORKSPACE,
            target_commit_hash=(
                target_commit_hash
                or (self.head_commit().commit_hash if self.head_commit() else "")
            ),
        )

    def plan_full_rewind(
        self,
        *,
        target_commit_hash: str = "",
        offset: int = 1,
        after_turn_id: str = "",
        will_drop_event_count: int = 0,
    ) -> RestorePlan:
        """Build the canonical "rewind everything" plan."""
        return plan_restore(
            axis=RestoreAxis.BOTH,
            target_commit_hash=(
                target_commit_hash
                or (self.head_commit().commit_hash if self.head_commit() else "")
            ),
            truncate_after_turn_id=after_turn_id,
            will_drop_turn_count=max(0, int(offset)),
            will_drop_event_count=max(0, int(will_drop_event_count)),
        )


class CheckpointRegistry:
    """Module-level registry mapping session_id → :class:`SessionCheckpointManager`."""

    def __init__(
        self,
        *,
        checkpoints_root: Path,
    ) -> None:
        self._root = Path(checkpoints_root).resolve()
        self._lock = threading.RLock()
        self._managers: dict[str, SessionCheckpointManager] = {}

    def get_or_create(
        self,
        session_id: str,
        workspace_path: Path,
        *,
        triggered_event_kinds: Optional[Iterable[str]] = None,
        gitignore_body: Optional[str] = None,
    ) -> SessionCheckpointManager:
        """Return the manager for ``session_id``, creating one on first call."""
        key = session_id or "default"
        with self._lock:
            existing = self._managers.get(key)
            if existing is not None:
                return existing
            workspace = Path(workspace_path).resolve()
            hashed = hash_working_dir(f"{key}:{workspace}")
            repo_root = self._root / hashed
            tracker = ShadowRepoTracker(
                workspace_path=workspace,
                repo_root=repo_root,
                session_id=key,
                gitignore_body=(
                    gitignore_body if gitignore_body is not None else compose_gitignore()
                ),
            )
            kinds = (
                set(triggered_event_kinds)
                if triggered_event_kinds is not None
                else set(DEFAULT_EVENT_KINDS)
            )
            manager = SessionCheckpointManager(
                tracker=tracker,
                triggered_event_kinds=kinds,
            )
            self._managers[key] = manager
            return manager

    def drop(self, session_id: str) -> bool:
        with self._lock:
            return self._managers.pop(session_id, None) is not None

    def list_sessions(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._managers.keys()))

    def manager_for(self, session_id: str) -> Optional[SessionCheckpointManager]:
        with self._lock:
            return self._managers.get(session_id)


_DEFAULT_REGISTRY: Optional[CheckpointRegistry] = None
_DEFAULT_REGISTRY_LOCK = threading.RLock()


def get_checkpoint_registry(
    *,
    checkpoints_root: Optional[Path] = None,
    rebuild: bool = False,
) -> CheckpointRegistry:
    """Return (and lazily construct) the module-level checkpoint registry."""
    global _DEFAULT_REGISTRY
    with _DEFAULT_REGISTRY_LOCK:
        if _DEFAULT_REGISTRY is not None and not rebuild:
            return _DEFAULT_REGISTRY
        root = (
            Path(checkpoints_root).resolve()
            if checkpoints_root is not None
            else Path("data") / "checkpoints"
        )
        _DEFAULT_REGISTRY = CheckpointRegistry(checkpoints_root=root)
        return _DEFAULT_REGISTRY


def reset_checkpoint_registry_for_testing() -> None:
    """Drop the module-level registry (test-only)."""
    global _DEFAULT_REGISTRY
    with _DEFAULT_REGISTRY_LOCK:
        _DEFAULT_REGISTRY = None


__all__ = [
    "CheckpointRegistry",
    "DEFAULT_EVENT_KINDS",
    "SessionCheckpointManager",
    "get_checkpoint_registry",
    "reset_checkpoint_registry_for_testing",
]
