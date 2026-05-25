"""Three-axis restore coordinator (voice_history / workspace / both).

The coordinator is intentionally PLAN-then-APPLY: callers ask for a
:class:`RestorePlan` first, optionally confirm with the user via voice
("undo the last 3 turns OK?"), then call ``execute``. This keeps the
destructive operation behind an explicit confirmation gate and makes
the audit log richer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence

LOGGER = logging.getLogger(__name__)


class RestoreAxis(str, Enum):
    """The three restore axes the catalog T1 calls out."""

    VOICE_HISTORY = "voice_history"
    WORKSPACE = "workspace"
    BOTH = "both"


@dataclass(frozen=True)
class RestorePlan:
    """Pre-execution snapshot of a restore request.

    Attributes:
        axis: which axes the plan touches.
        target_commit_hash: commit to reset the workspace to (empty
            when ``axis`` is :attr:`RestoreAxis.VOICE_HISTORY` only).
        truncate_after_turn_id: voice-history turn id to truncate after
            (empty when ``axis`` is :attr:`RestoreAxis.WORKSPACE` only).
        will_drop_turn_count: number of voice memory turns the plan
            will drop.
        will_drop_event_count: number of bus events the plan will drop.
        narration: user-facing description suitable for the voice
            confirmation prompt.
    """

    axis: RestoreAxis
    target_commit_hash: str = ""
    truncate_after_turn_id: str = ""
    will_drop_turn_count: int = 0
    will_drop_event_count: int = 0
    narration: str = ""


@dataclass(frozen=True)
class RestoreOutcome:
    """Outcome of a single :func:`execute_restore` call.

    Attributes:
        axis: axes that were applied.
        workspace_reset_succeeded: True when the workspace hard-reset
            succeeded (or no workspace reset was attempted).
        voice_history_truncated: number of memory turns dropped.
        events_truncated: number of bus events dropped.
        error_message: short summary of any failure (empty on success).
    """

    axis: RestoreAxis
    workspace_reset_succeeded: bool = True
    voice_history_truncated: int = 0
    events_truncated: int = 0
    error_message: str = ""


WorkspaceReset = Callable[[str], bool]
"""Callable mapping ``commit_hash`` to True/False on success."""

VoiceHistoryTruncator = Callable[[str], int]
"""Callable mapping ``after_turn_id`` to the number of dropped turns."""

EventLogTruncator = Callable[[str], int]
"""Callable mapping ``after_turn_id`` to the number of dropped bus events."""


def plan_restore(
    *,
    axis: RestoreAxis,
    target_commit_hash: str = "",
    truncate_after_turn_id: str = "",
    will_drop_turn_count: int = 0,
    will_drop_event_count: int = 0,
) -> RestorePlan:
    """Build a :class:`RestorePlan` with a voice-friendly narration.

    Args:
        axis: axes the plan touches.
        target_commit_hash: workspace reset target (when relevant).
        truncate_after_turn_id: voice-history truncation point.
        will_drop_turn_count: turns dropped by this plan.
        will_drop_event_count: bus events dropped by this plan.

    Returns:
        :class:`RestorePlan` ready for ``execute_restore``.
    """
    parts: list[str] = []
    if axis in (RestoreAxis.WORKSPACE, RestoreAxis.BOTH):
        if target_commit_hash:
            parts.append(
                f"reset workspace files to commit {target_commit_hash[:8]}"
            )
        else:
            parts.append("reset workspace files to the latest checkpoint")
    if axis in (RestoreAxis.VOICE_HISTORY, RestoreAxis.BOTH):
        if will_drop_turn_count > 0:
            parts.append(
                f"drop the last {will_drop_turn_count} voice turn(s)"
            )
        else:
            parts.append("truncate the voice history")
        if will_drop_event_count > 0:
            parts.append(
                f"drop {will_drop_event_count} bus event(s)"
            )
    if not parts:
        narration = "Nothing to restore."
    elif len(parts) == 1:
        narration = f"I'll {parts[0]}."
    else:
        narration = (
            f"I'll {', '.join(parts[:-1])}, then {parts[-1]}."
        )
    return RestorePlan(
        axis=axis,
        target_commit_hash=target_commit_hash,
        truncate_after_turn_id=truncate_after_turn_id,
        will_drop_turn_count=will_drop_turn_count,
        will_drop_event_count=will_drop_event_count,
        narration=narration,
    )


def execute_restore(
    plan: RestorePlan,
    *,
    workspace_reset: Optional[WorkspaceReset] = None,
    voice_history_truncate: Optional[VoiceHistoryTruncator] = None,
    event_log_truncate: Optional[EventLogTruncator] = None,
) -> RestoreOutcome:
    """Apply ``plan`` via the injected operation callables.

    Args:
        plan: plan to execute (typically the result of :func:`plan_restore`).
        workspace_reset: callable invoked when the plan touches the
            workspace axis. Returns True on success.
        voice_history_truncate: callable invoked when the plan touches
            the voice-history axis. Returns the number of dropped turns.
        event_log_truncate: optional callable invoked alongside the
            voice-history axis to also truncate the bus event log.

    Returns:
        :class:`RestoreOutcome` describing the outcome.
    """
    workspace_ok = True
    truncated_turns = 0
    truncated_events = 0
    errors: list[str] = []

    if plan.axis in (RestoreAxis.WORKSPACE, RestoreAxis.BOTH):
        if workspace_reset is None:
            workspace_ok = False
            errors.append("no workspace_reset callable supplied")
        else:
            try:
                workspace_ok = bool(workspace_reset(plan.target_commit_hash))
            except Exception as exc:  # noqa: BLE001 - never raise from here
                workspace_ok = False
                errors.append(f"workspace_reset raised: {type(exc).__name__}: {exc}")
                LOGGER.warning(
                    "workspace_reset raised during restore", exc_info=True,
                )

    if plan.axis in (RestoreAxis.VOICE_HISTORY, RestoreAxis.BOTH):
        if voice_history_truncate is None:
            errors.append("no voice_history_truncate callable supplied")
        else:
            try:
                truncated_turns = int(
                    voice_history_truncate(plan.truncate_after_turn_id) or 0,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    f"voice_history_truncate raised: {type(exc).__name__}: {exc}"
                )
                LOGGER.warning(
                    "voice_history_truncate raised during restore",
                    exc_info=True,
                )
        if event_log_truncate is not None:
            try:
                truncated_events = int(
                    event_log_truncate(plan.truncate_after_turn_id) or 0,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    f"event_log_truncate raised: {type(exc).__name__}: {exc}"
                )
                LOGGER.warning(
                    "event_log_truncate raised during restore", exc_info=True,
                )

    return RestoreOutcome(
        axis=plan.axis,
        workspace_reset_succeeded=workspace_ok,
        voice_history_truncated=truncated_turns,
        events_truncated=truncated_events,
        error_message="; ".join(errors),
    )


__all__ = [
    "EventLogTruncator",
    "RestoreAxis",
    "RestoreOutcome",
    "RestorePlan",
    "VoiceHistoryTruncator",
    "WorkspaceReset",
    "execute_restore",
    "plan_restore",
]
