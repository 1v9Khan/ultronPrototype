"""Forfeit primitive -- give up gracefully without burning more tokens.

Direct port of SWE-Agent's ``tools/forfeit/bin/exit_forfeit`` +
the ``_ExitForfeit`` exception handling in ``agents.py`` (MIT,
Yang et al. 2024). The pattern: when a coding session is clearly
not converging (re-trying the same broken edit, hallucinating a
missing file, etc.), an explicit "give up" path lets the model
exit cleanly rather than wasting the rest of the token budget.

Two routes to forfeit:

1. **In-band sentinel.** The model emits :data:`ULTRON_EXIT_FORFEIT`
   in its output; the supervisor's observation scanner catches it
   via :mod:`ultron.coding.sentinels`. This is the SWE-Agent
   pattern -- a tool prints the sentinel + the harness picks it up.

2. **Out-of-band trigger.** The user says "scrap it" / "give up" /
   "cancel the task" -- a voice-intent classifier raises the
   forfeit. Implemented as a regular function call into
   :class:`ForfeitController`.

The forfeit handler:

* Records the forfeit reason + per-session metadata via :class:`SessionRegistry`.
* Triggers the salvage path from :mod:`ultron.coding.diff_snapshot`
  so partial work is preserved.
* Optionally calls a caller-supplied :class:`ForfeitListener` to
  emit a bus event / queue a narration line.

Tiered forfeit (creative extension from the catalog):

* :data:`ForfeitTier.SAFE` -- record + salvage; leaves the working
  tree untouched.
* :data:`ForfeitTier.REVERT` -- in addition, revert every file the
  supervisor knows it edited this session (via :class:`FileHistory`'s
  undo stack).
* :data:`ForfeitTier.FOLLOWUP` -- record a memory observation
  carrying WHY we forfeited so future sessions have prior-attempt
  context.

Per-session minimum-effort threshold prevents the model from over-
using forfeit: must have made ``min_actions_before_forfeit`` tool
calls AND been running for ``min_runtime_seconds`` before a forfeit
is allowed. Below the threshold the controller returns
:data:`ForfeitOutcome.DENIED_TOO_EARLY`.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from ultron.coding.diff_snapshot import (
    SalvageResult,
    salvage_on_error,
)
from ultron.coding.session_registry import (
    SessionRegistry,
    get_session_registry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums + records
# ---------------------------------------------------------------------------


class ForfeitTier(Enum):
    """Three forfeit handling tiers."""

    SAFE = "safe"
    REVERT = "revert"
    FOLLOWUP = "followup"


class ForfeitOutcome(Enum):
    """Result of attempting a forfeit."""

    GRANTED = "granted"
    DENIED_TOO_EARLY = "denied_too_early"
    DENIED_DISABLED = "denied_disabled"
    ALREADY_FORFEITED = "already_forfeited"


@dataclass(frozen=True)
class ForfeitResult:
    """Output of :meth:`ForfeitController.forfeit`."""

    outcome: ForfeitOutcome
    tier: ForfeitTier
    reason: str = ""
    salvage: Optional[SalvageResult] = None
    files_reverted: list[str] = field(default_factory=list)
    forfeited_at: float = 0.0
    actions_at_forfeit: int = 0
    runtime_seconds_at_forfeit: float = 0.0


# Registry keys
_REGISTRY_FORFEIT_KEY: str = "forfeit_state"
_REGISTRY_ACTIONS_KEY: str = "forfeit_action_count"
_REGISTRY_STARTED_KEY: str = "forfeit_session_started_at"

#: Default minimum-effort thresholds.
DEFAULT_MIN_ACTIONS: int = 3
DEFAULT_MIN_RUNTIME_SECONDS: float = 30.0


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class ForfeitController:
    """Single point of forfeit decision-making for one session.

    Construct via :func:`get_forfeit_controller` (preferred,
    singleton-per-session) or directly with a custom registry +
    file-history binding.

    Public API:

    * :meth:`record_action()` -- bump the per-session action counter
      so the minimum-effort gate has data to consult.
    * :meth:`forfeit(reason, tier, repo_root)` -> :class:`ForfeitResult`.
    * :meth:`is_forfeited()` -- True after a successful forfeit.
    * :meth:`current_state()` -> dict for diagnostics.
    """

    def __init__(
        self,
        *,
        registry: SessionRegistry,
        enabled: bool = True,
        min_actions_before_forfeit: int = DEFAULT_MIN_ACTIONS,
        min_runtime_seconds: float = DEFAULT_MIN_RUNTIME_SECONDS,
    ) -> None:
        if min_actions_before_forfeit < 0:
            raise ValueError("min_actions_before_forfeit must be >= 0")
        if min_runtime_seconds < 0:
            raise ValueError("min_runtime_seconds must be >= 0")
        self.registry = registry
        self.enabled = bool(enabled)
        self.min_actions_before_forfeit = int(min_actions_before_forfeit)
        self.min_runtime_seconds = float(min_runtime_seconds)
        # Seed the started-at timestamp lazily so test-injected
        # registries can pre-populate it.
        if _REGISTRY_STARTED_KEY not in self.registry:
            self.registry[_REGISTRY_STARTED_KEY] = time.time()

    # ----- counters ----------------------------------------------------

    def record_action(self) -> int:
        """Increment the action counter; returns the new value."""
        current = int(self.registry.get(_REGISTRY_ACTIONS_KEY, 0))
        new = current + 1
        self.registry[_REGISTRY_ACTIONS_KEY] = new
        return new

    def action_count(self) -> int:
        return int(self.registry.get(_REGISTRY_ACTIONS_KEY, 0))

    def runtime_seconds(self) -> float:
        started = float(self.registry.get(_REGISTRY_STARTED_KEY, time.time()))
        return max(0.0, time.time() - started)

    # ----- forfeit -----------------------------------------------------

    def forfeit(
        self,
        *,
        reason: str,
        tier: ForfeitTier = ForfeitTier.SAFE,
        repo_root: Optional[str | Path] = None,
        files_to_revert: Optional[list[str]] = None,
        followup_writer: Optional[Callable[[str], None]] = None,
        listener: Optional[Callable[["ForfeitResult"], None]] = None,
    ) -> ForfeitResult:
        """Attempt to forfeit the current session.

        :param reason: human-readable explanation; recorded in the
            registry + the salvage metadata.
        :param tier: chooses between SAFE (record + salvage only),
            REVERT (also revert files via FileHistory.undo_last on
            each ``files_to_revert`` entry), and FOLLOWUP (also
            invoke ``followup_writer(reason)`` so the session's
            why-we-failed message reaches the next session).
        :param repo_root: optional path used by the salvage step.
        :param files_to_revert: paths to undo when tier is REVERT.
            Caller supplies (typically from session-touched-files
            tracking).
        :param followup_writer: callable invoked when tier is FOLLOWUP.
            Receives ``reason`` as its only argument.
        :param listener: optional notification hook invoked with
            the :class:`ForfeitResult`.
        """
        now = time.time()
        if not self.enabled:
            result = ForfeitResult(
                outcome=ForfeitOutcome.DENIED_DISABLED,
                tier=tier,
                reason=reason,
                forfeited_at=now,
                actions_at_forfeit=self.action_count(),
                runtime_seconds_at_forfeit=self.runtime_seconds(),
            )
            self._notify(listener, result)
            return result

        if self.is_forfeited():
            existing = self.current_state()
            result = ForfeitResult(
                outcome=ForfeitOutcome.ALREADY_FORFEITED,
                tier=ForfeitTier(existing.get("tier", "safe")),
                reason=str(existing.get("reason", "")),
                forfeited_at=float(existing.get("forfeited_at", now)),
                actions_at_forfeit=int(existing.get("actions_at_forfeit", 0)),
                runtime_seconds_at_forfeit=float(
                    existing.get("runtime_seconds_at_forfeit", 0.0)
                ),
            )
            self._notify(listener, result)
            return result

        if (
            self.action_count() < self.min_actions_before_forfeit
            or self.runtime_seconds() < self.min_runtime_seconds
        ):
            result = ForfeitResult(
                outcome=ForfeitOutcome.DENIED_TOO_EARLY,
                tier=tier,
                reason=reason,
                forfeited_at=now,
                actions_at_forfeit=self.action_count(),
                runtime_seconds_at_forfeit=self.runtime_seconds(),
            )
            self._notify(listener, result)
            return result

        # Granted. Run the tier handlers.
        salvage: Optional[SalvageResult] = None
        reverted: list[str] = []
        if repo_root is not None:
            try:
                salvage = salvage_on_error(
                    repo_root,
                    session_id=self.registry.session_id,
                    exit_status="exit_forfeit",
                    exception=None,
                )
            except Exception as exc:
                logger.warning("forfeit salvage failed: %s", exc)
        if tier == ForfeitTier.REVERT and files_to_revert:
            from ultron.coding.file_history import get_file_history

            fh = get_file_history(self.registry.session_id, registry=self.registry)
            for path in files_to_revert:
                try:
                    r = fh.undo_last(path)
                    if r.applied:
                        reverted.append(path)
                except Exception as exc:
                    logger.warning(
                        "forfeit revert of %s failed: %s", path, exc
                    )
        if tier == ForfeitTier.FOLLOWUP and followup_writer is not None:
            try:
                followup_writer(reason)
            except Exception as exc:
                logger.warning(
                    "forfeit followup_writer raised: %s", exc
                )

        result = ForfeitResult(
            outcome=ForfeitOutcome.GRANTED,
            tier=tier,
            reason=reason,
            salvage=salvage,
            files_reverted=reverted,
            forfeited_at=now,
            actions_at_forfeit=self.action_count(),
            runtime_seconds_at_forfeit=self.runtime_seconds(),
        )
        self.registry[_REGISTRY_FORFEIT_KEY] = {
            "outcome": result.outcome.value,
            "tier": result.tier.value,
            "reason": result.reason,
            "files_reverted": list(result.files_reverted),
            "forfeited_at": result.forfeited_at,
            "actions_at_forfeit": result.actions_at_forfeit,
            "runtime_seconds_at_forfeit": result.runtime_seconds_at_forfeit,
        }
        self._notify(listener, result)
        return result

    # ----- inspection --------------------------------------------------

    def is_forfeited(self) -> bool:
        return _REGISTRY_FORFEIT_KEY in self.registry

    def current_state(self) -> dict:
        v = self.registry.get(_REGISTRY_FORFEIT_KEY)
        return dict(v) if isinstance(v, dict) else {}

    def reset(self) -> None:
        """Clear forfeit state -- used between sessions."""
        self.registry.pop(_REGISTRY_FORFEIT_KEY, default=None)
        self.registry.pop(_REGISTRY_ACTIONS_KEY, default=None)
        self.registry[_REGISTRY_STARTED_KEY] = time.time()

    # ----- helpers -----------------------------------------------------

    @staticmethod
    def _notify(
        listener: Optional[Callable[[ForfeitResult], None]],
        result: ForfeitResult,
    ) -> None:
        if listener is None:
            return
        try:
            listener(result)
        except Exception as exc:
            logger.warning(
                "ForfeitController listener raised: %s", exc
            )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_forfeit_controller(
    session_id: str,
    *,
    registry: Optional[SessionRegistry] = None,
    enabled: bool = True,
    min_actions_before_forfeit: int = DEFAULT_MIN_ACTIONS,
    min_runtime_seconds: float = DEFAULT_MIN_RUNTIME_SECONDS,
) -> ForfeitController:
    """Return a :class:`ForfeitController` for ``session_id``.

    Uses :func:`get_session_registry` unless ``registry`` is passed
    (for tests).
    """
    if registry is None:
        registry = get_session_registry(session_id)
    return ForfeitController(
        registry=registry,
        enabled=enabled,
        min_actions_before_forfeit=min_actions_before_forfeit,
        min_runtime_seconds=min_runtime_seconds,
    )


__all__ = [
    "DEFAULT_MIN_ACTIONS",
    "DEFAULT_MIN_RUNTIME_SECONDS",
    "ForfeitController",
    "ForfeitOutcome",
    "ForfeitResult",
    "ForfeitTier",
    "get_forfeit_controller",
]
