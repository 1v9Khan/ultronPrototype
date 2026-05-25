"""Per-rule auto-approval matrix decoupled from binary trust dial.

Adapted from cline's ``autoApprovalSettings.actions`` pattern (Apache
2.0; see ``THIRD_PARTY_NOTICES.md``). Ultron's safety validator
already returns a per-rule :class:`~ultron.safety.validator.Verdict`
(ALLOW / BLOCK_HARD / NEEDS_EXPLICIT_INTENT / LOG_ONLY); this module
layers a configurable AUTO-APPROVAL MODE on top so an operator can
say "permit reads inside the project workspace but require confirm
for reads outside it" without rewriting any rule.

The matrix has four modes per rule:

* ``always_ask`` — every dispatch requires interactive confirmation.
* ``allow_local`` — allow when the resolved tool target lives inside
  the workspace; require confirmation when it lives outside.
* ``allow_external`` — allow when the target is outside the workspace;
  require confirmation for inside-workspace targets. Useful for
  network-touching tools the user wants to gate per-domain.
* ``allow_all`` — auto-approve regardless of target.

Locality is determined by an injected ``LocalityProbe`` predicate
(typically wrapping the safety validator's ``PathResolver``); when no
probe is provided every dispatch is treated as local.

Two master overrides sit on top of the matrix:

* ``yolo_mode`` (process-global; respects user direction "let me move
  fast") auto-approves every rule.
* Per-session warming (``record_user_grant``) — after the user has
  approved a (rule, target) pair N times in a row, subsequent matches
  are auto-approved for the rest of the session.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Mapping, Optional, Sequence

LOGGER = logging.getLogger(__name__)


class AutoApprovalMode(str, Enum):
    """Per-rule auto-approval mode."""

    ALWAYS_ASK = "always_ask"
    ALLOW_LOCAL = "allow_local"
    ALLOW_EXTERNAL = "allow_external"
    ALLOW_ALL = "allow_all"


class AutoApprovalOutcome(str, Enum):
    """Resolution of one auto-approval check."""

    ALLOW = "allow"
    ASK_USER = "ask_user"
    DENY = "deny"


#: Default mode applied when a rule is not in the per-rule overrides.
DEFAULT_AUTO_APPROVAL_MODE: AutoApprovalMode = AutoApprovalMode.ALWAYS_ASK

#: Default number of consecutive user approvals that warms a
#: (rule, target) pair into the session allowlist.
DEFAULT_WARMING_THRESHOLD: int = 5

#: Default session allowlist TTL (seconds) — a warmed entry expires if
#: not exercised within this window.
DEFAULT_WARMING_TTL_SECONDS: float = 30 * 60.0


LocalityProbe = Callable[[str], bool]
"""Callable mapping a target identifier (path, URL, command) to True
when the target is considered LOCAL (inside the workspace), False
otherwise. Used to disambiguate ``allow_local`` vs ``allow_external``.
"""


@dataclass(frozen=True)
class AutoApprovalResult:
    """Outcome of a single :meth:`AutoApprovalMatrix.evaluate` call.

    Attributes:
        rule_id: rule identifier that produced the verdict.
        mode: mode that was applied (``always_ask`` / ``allow_local`` /
            ``allow_external`` / ``allow_all``).
        outcome: ``allow`` / ``ask_user`` / ``deny`` decision.
        locality: True when the target was probed as LOCAL, False
            EXTERNAL, None when no probe was attached.
        target: the target identifier the rule was evaluated against
            (for audit log + voice-friendly rendering).
        reason: short explanation suitable for telemetry.
        warmed: True when the outcome was driven by the session
            allowlist (the user previously granted N consecutive
            approvals and the matrix is now silently auto-allowing).
    """

    rule_id: str
    mode: AutoApprovalMode
    outcome: AutoApprovalOutcome
    locality: Optional[bool] = None
    target: str = ""
    reason: str = ""
    warmed: bool = False


@dataclass
class _WarmingEntry:
    """Internal counter for per-(rule, target) consecutive approvals."""

    count: int = 0
    last_seen: float = 0.0


class AutoApprovalMatrix:
    """Per-rule auto-approval policy with optional session warming.

    Args:
        rule_modes: mapping of rule-id → :class:`AutoApprovalMode`.
            Missing keys fall back to :data:`DEFAULT_AUTO_APPROVAL_MODE`.
        locality_probe: optional callable mapping target → True (local)
            / False (external).
        yolo_mode: when True, every check returns
            :attr:`AutoApprovalOutcome.ALLOW` regardless of mode.
        warming_threshold: consecutive-approval count that warms a
            (rule, target) pair into the session allowlist. Default
            :data:`DEFAULT_WARMING_THRESHOLD`. Pass 0 to disable.
        warming_ttl_seconds: TTL on warmed entries. Default
            :data:`DEFAULT_WARMING_TTL_SECONDS`. Pass 0 to disable.
        clock: optional callable returning monotonic seconds (test hook).
    """

    def __init__(
        self,
        rule_modes: Optional[Mapping[str, AutoApprovalMode | str]] = None,
        *,
        locality_probe: Optional[LocalityProbe] = None,
        yolo_mode: bool = False,
        warming_threshold: int = DEFAULT_WARMING_THRESHOLD,
        warming_ttl_seconds: float = DEFAULT_WARMING_TTL_SECONDS,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._rule_modes: dict[str, AutoApprovalMode] = {}
        if rule_modes:
            for rule_id, mode in rule_modes.items():
                self._rule_modes[rule_id] = _coerce_mode(mode)
        self._locality_probe = locality_probe
        self._yolo_mode = bool(yolo_mode)
        self._warming_threshold = max(0, int(warming_threshold))
        self._warming_ttl = max(0.0, float(warming_ttl_seconds))
        self._clock = clock or time.monotonic
        self._lock = threading.RLock()
        self._consecutive: dict[tuple[str, str], _WarmingEntry] = {}
        self._warmed: dict[tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # Mode configuration
    # ------------------------------------------------------------------

    def set_mode(self, rule_id: str, mode: AutoApprovalMode | str) -> None:
        """Override the mode for ``rule_id`` (or insert if absent)."""
        with self._lock:
            self._rule_modes[rule_id] = _coerce_mode(mode)

    def mode_for(self, rule_id: str) -> AutoApprovalMode:
        """Resolve the effective mode for ``rule_id`` (falling back to default)."""
        with self._lock:
            return self._rule_modes.get(rule_id, DEFAULT_AUTO_APPROVAL_MODE)

    def configured_modes(self) -> Mapping[str, AutoApprovalMode]:
        """Snapshot of every explicitly-configured (rule_id, mode) pair."""
        with self._lock:
            return dict(self._rule_modes)

    def set_yolo_mode(self, enabled: bool) -> None:
        """Master override — when True, every check auto-allows."""
        with self._lock:
            self._yolo_mode = bool(enabled)

    def yolo_mode(self) -> bool:
        """Return whether the master override is active."""
        with self._lock:
            return self._yolo_mode

    def set_locality_probe(self, probe: Optional[LocalityProbe]) -> None:
        """Replace (or clear) the locality probe."""
        with self._lock:
            self._locality_probe = probe

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, rule_id: str, target: str = "") -> AutoApprovalResult:
        """Resolve the auto-approval verdict for ``rule_id`` + ``target``.

        Args:
            rule_id: identifier from the safety policy (e.g. ``"K1"``).
            target: optional target identifier (path / URL / command)
                used by the locality probe and the warming key.

        Returns:
            :class:`AutoApprovalResult` describing the decision.
        """
        with self._lock:
            mode = self._rule_modes.get(rule_id, DEFAULT_AUTO_APPROVAL_MODE)
            if self._yolo_mode:
                return AutoApprovalResult(
                    rule_id=rule_id,
                    mode=mode,
                    outcome=AutoApprovalOutcome.ALLOW,
                    target=target,
                    reason="yolo_mode override",
                )
            key = (rule_id, target or "")
            warmed_at = self._warmed.get(key)
            now = self._clock()
            if warmed_at is not None:
                if self._warming_ttl == 0 or now - warmed_at <= self._warming_ttl:
                    self._warmed[key] = now
                    return AutoApprovalResult(
                        rule_id=rule_id,
                        mode=mode,
                        outcome=AutoApprovalOutcome.ALLOW,
                        target=target,
                        reason="session warming",
                        warmed=True,
                        locality=self._safe_locality(target),
                    )
                # TTL expired; drop the warmed entry.
                self._warmed.pop(key, None)
            locality = self._safe_locality(target)
            outcome, reason = self._apply_mode(mode, locality)
            return AutoApprovalResult(
                rule_id=rule_id,
                mode=mode,
                outcome=outcome,
                target=target,
                locality=locality,
                reason=reason,
            )

    def evaluate_many(
        self, items: Iterable[tuple[str, str]],
    ) -> list[AutoApprovalResult]:
        """Bulk wrapper around :meth:`evaluate`."""
        return [self.evaluate(rule_id, target) for rule_id, target in items]

    # ------------------------------------------------------------------
    # User grants (session warming)
    # ------------------------------------------------------------------

    def record_user_grant(self, rule_id: str, target: str = "") -> bool:
        """Record one user approval; return True if the pair is now warmed.

        Args:
            rule_id: rule identifier that was approved.
            target: target identifier the user approved against.

        Returns:
            True when this call promoted the pair into the warmed
            allowlist; False otherwise.
        """
        if self._warming_threshold == 0:
            return False
        key = (rule_id, target or "")
        with self._lock:
            entry = self._consecutive.setdefault(key, _WarmingEntry())
            entry.count += 1
            entry.last_seen = self._clock()
            if entry.count >= self._warming_threshold:
                self._warmed[key] = entry.last_seen
                # Reset the counter so a future revoke can re-warm cleanly.
                entry.count = 0
                return True
            return False

    def revoke_user_grant(self, rule_id: str, target: str = "") -> bool:
        """Reset the consecutive counter AND drop the warmed entry.

        Returns:
            True when a warmed entry was removed, False otherwise.
        """
        key = (rule_id, target or "")
        with self._lock:
            self._consecutive.pop(key, None)
            return self._warmed.pop(key, None) is not None

    def record_user_denial(self, rule_id: str, target: str = "") -> None:
        """Reset the counter without dropping warmed entries.

        Use this when the user rejected a request but you don't want to
        immediately revoke a long-standing warming for the same pair.
        """
        key = (rule_id, target or "")
        with self._lock:
            self._consecutive.pop(key, None)

    def warmed_pairs(self) -> Sequence[tuple[str, str]]:
        """Snapshot of the currently-warmed (rule_id, target) pairs."""
        with self._lock:
            now = self._clock()
            alive: list[tuple[str, str]] = []
            for key, when in self._warmed.items():
                if self._warming_ttl == 0 or now - when <= self._warming_ttl:
                    alive.append(key)
            return alive

    def clear_session(self) -> None:
        """Drop every warmed entry + consecutive-approval counter."""
        with self._lock:
            self._consecutive.clear()
            self._warmed.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_locality(self, target: str) -> Optional[bool]:
        """Run the locality probe with fail-open semantics."""
        if self._locality_probe is None or not target:
            return None
        try:
            return bool(self._locality_probe(target))
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _apply_mode(
        mode: AutoApprovalMode, locality: Optional[bool],
    ) -> tuple[AutoApprovalOutcome, str]:
        """Map a (mode, locality) pair to the user-facing outcome."""
        if mode is AutoApprovalMode.ALLOW_ALL:
            return AutoApprovalOutcome.ALLOW, "allow_all mode"
        if mode is AutoApprovalMode.ALWAYS_ASK:
            return AutoApprovalOutcome.ASK_USER, "always_ask mode"
        if mode is AutoApprovalMode.ALLOW_LOCAL:
            if locality is True:
                return AutoApprovalOutcome.ALLOW, "allow_local — target is local"
            if locality is False:
                return AutoApprovalOutcome.ASK_USER, "allow_local — target is external"
            return AutoApprovalOutcome.ASK_USER, "allow_local — locality unknown"
        if mode is AutoApprovalMode.ALLOW_EXTERNAL:
            if locality is False:
                return AutoApprovalOutcome.ALLOW, "allow_external — target is external"
            if locality is True:
                return AutoApprovalOutcome.ASK_USER, "allow_external — target is local"
            return AutoApprovalOutcome.ASK_USER, "allow_external — locality unknown"
        return AutoApprovalOutcome.ASK_USER, "unrecognised mode"


def _coerce_mode(value: AutoApprovalMode | str) -> AutoApprovalMode:
    """Coerce a string or enum into :class:`AutoApprovalMode`."""
    if isinstance(value, AutoApprovalMode):
        return value
    try:
        return AutoApprovalMode(str(value))
    except ValueError:
        LOGGER.warning("unknown auto-approval mode %r; falling back to always_ask", value)
        return AutoApprovalMode.ALWAYS_ASK


__all__ = [
    "DEFAULT_AUTO_APPROVAL_MODE",
    "DEFAULT_WARMING_THRESHOLD",
    "DEFAULT_WARMING_TTL_SECONDS",
    "AutoApprovalMatrix",
    "AutoApprovalMode",
    "AutoApprovalOutcome",
    "AutoApprovalResult",
    "LocalityProbe",
]
