"""Composable trusted-tool-policy chain.

T13 (OpenClaw catalog port; see ``THIRD_PARTY_NOTICES.md``). Lets
plugin-shaped policies attach to a ``before_tool_call`` chain that
runs BEFORE the generic safety validator. Each policy can:

* **Block** the call (``allow=False`` with ``reason``) — terminal;
  short-circuits the chain.
* **Rewrite parameters** (``params=<new>``) — stacks; later policies
  see the rewritten params. Useful for PII redaction, secret
  scrubbing, path canonicalisation.
* **Require approval** (``require_approval=<descriptor>``) — first
  one wins; later approvals do not override an earlier requirement.
  Used for cost-limit gating, voice-confirmation requests, etc.

The chain runs in registration order. Block terminates immediately;
params-rewrites stack; approval-first-wins. Each policy can read
session-scoped state via ``ctx.get_session_extension(namespace)``
for cross-call memory (warning counters, recent-approval cache,
per-mode toggles).

Use cases ultron derives:

* PII detector: rewrites email / phone / SSN tokens in tool params
  before they hit memory-write.
* Cost-limit policy: if estimated tokens > N, require approval.
* Per-mode policy: in GAMING mode, force-approve tool calls that
  would take > 100 ms (the user's gaming session must not stall).
* Per-channel policy: future Telegram channel can't play audio, so
  any tool producing audio output gets blocked there.
* Audit-only policy: never blocks, just emits richer logs with
  per-call tags.

The validator stack consults the policy chain BEFORE the rule
categories — a block from a trusted policy is the cleanest "stop
this here" path because it doesn't require the call to round-trip
through the audit log on a rule mismatch.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Protocol

LOGGER = logging.getLogger(__name__)

#: Slow-policy WARN threshold. Anything taking longer than this on a
#: pre-tool-call check is suspect — the chain is meant to be
#: latency-cheap (< 5 ms) per registered policy.
DEFAULT_SLOW_POLICY_WARN_MS: float = 50.0


class PolicyOutcome(str, Enum):
    """Aggregated outcome of a chain run.

    ``PASS_THROUGH`` — no policy returned anything actionable.
    ``REWRITE`` — at least one policy rewrote params; no block.
    ``REQUIRE_APPROVAL`` — at least one policy demanded approval; no
    block.
    ``BLOCK`` — a policy explicitly blocked the call.
    """

    PASS_THROUGH = "pass_through"
    REWRITE = "rewrite"
    REQUIRE_APPROVAL = "require_approval"
    BLOCK = "block"


@dataclass(frozen=True)
class ApprovalRequest:
    """One policy's approval demand.

    Attributes:
        policy_id: The registered policy that produced the demand.
        kind: A short string identifying the kind of approval
            (``"voice_confirmation"`` / ``"cost_limit"`` /
            ``"category_warn"``). Free-form; consumed by the
            approval-router.
        message: Human-readable explanation. Surfaced through the
            voice TTS or chat channel.
        metadata: Opaque per-policy data (e.g. cost estimate, matched
            pattern). Forwarded to the approval-router.
    """

    policy_id: str
    kind: str
    message: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDecision:
    """A single policy's verdict.

    Exactly one of ``block`` / ``params`` / ``require_approval`` is
    typically set, though setting both ``params`` and
    ``require_approval`` is permitted (the chain stacks params then
    records the approval). ``block=False`` is a no-op.

    Attributes:
        allow: ``True`` to pass through; ``False`` to block. ``None``
            means "no opinion on allow/block".
        reason: Required when ``allow=False``; supplies the audit-log
            reason and (when no ``message``) the user-facing message.
        message: Optional user-facing message that overrides the
            default-formatted block notification.
        category: Analytics label (e.g. ``"pii"`` / ``"cost_limit"``).
        params: Replacement parameters dict. ``None`` leaves params
            unchanged. Later policies see whatever this policy wrote.
        require_approval: Optional :class:`ApprovalRequest`. First
            policy to set this wins; subsequent settings are ignored.
        metadata: Opaque per-policy state.
    """

    allow: Optional[bool] = None
    reason: str = ""
    message: Optional[str] = None
    category: Optional[str] = None
    params: Optional[Mapping[str, Any]] = None
    require_approval: Optional[ApprovalRequest] = None
    metadata: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class ToolCallContext:
    """Frozen snapshot the chain receives for each tool call.

    Mirrors enough of the validator's RuleContext to let a policy
    make a decision without needing to depend on the full safety
    layer. Use :meth:`get_session_extension` to access session-scoped
    state.
    """

    tool_name: str
    arguments: Mapping[str, Any]
    capability: str = ""
    user_text: str = ""
    mode: str = "standby"
    session_id: str = "default"
    _session_state: Mapping[str, Any] = field(default_factory=dict)

    def get_session_extension(self, namespace: str) -> Any:
        """Read session-scoped state at ``namespace``. ``None`` when absent."""
        return self._session_state.get(namespace) if self._session_state else None


class Policy(Protocol):
    """Protocol every chain policy implements."""

    policy_id: str

    def evaluate(self, ctx: ToolCallContext) -> Optional[PolicyDecision]:
        """Return a :class:`PolicyDecision` or ``None`` for pass-through."""


@dataclass(frozen=True)
class ChainResult:
    """Aggregated outcome of running the policy chain."""

    outcome: PolicyOutcome
    final_params: Mapping[str, Any]
    block_decision: Optional[PolicyDecision] = None
    approval: Optional[ApprovalRequest] = None
    rewrites_by: tuple[str, ...] = field(default_factory=tuple)
    decisions: tuple[tuple[str, PolicyDecision], ...] = field(default_factory=tuple)

    @property
    def blocked(self) -> bool:
        return self.outcome == PolicyOutcome.BLOCK

    @property
    def requires_approval(self) -> bool:
        return self.outcome == PolicyOutcome.REQUIRE_APPROVAL


@dataclass(frozen=True)
class RegisteredPolicy:
    """A policy that's been added to the chain."""

    policy_id: str
    policy: Policy
    order: int


class TrustedToolPolicyChain:
    """Ordered chain of trusted ``before_tool_call`` policies.

    Thread-safe: registration + run can interleave without external
    locking. Slow policies (above
    :data:`DEFAULT_SLOW_POLICY_WARN_MS`) WARN-log per call so
    misbehaving policies surface during ops.
    """

    def __init__(self, *, slow_policy_warn_ms: float = DEFAULT_SLOW_POLICY_WARN_MS) -> None:
        self._policies: list[RegisteredPolicy] = []
        self._lock = threading.RLock()
        self._next_order = 0
        self._slow_policy_warn_ms = slow_policy_warn_ms

    def register(self, policy: Policy) -> RegisteredPolicy:
        """Add ``policy`` to the chain. Returns the registration.

        Policies must expose a ``policy_id`` attribute (used in
        decisions + audit logs). Re-registering a policy with the
        same id replaces the existing entry.
        """
        policy_id = getattr(policy, "policy_id", policy.__class__.__name__)
        with self._lock:
            self._policies = [p for p in self._policies if p.policy_id != policy_id]
            reg = RegisteredPolicy(policy_id=policy_id, policy=policy, order=self._next_order)
            self._next_order += 1
            self._policies.append(reg)
            self._policies.sort(key=lambda p: p.order)
            return reg

    def unregister(self, policy_id: str) -> bool:
        """Remove the policy with ``policy_id``. Returns ``True`` on hit."""
        with self._lock:
            before = len(self._policies)
            self._policies = [p for p in self._policies if p.policy_id != policy_id]
            return len(self._policies) != before

    def clear(self) -> None:
        """Drop every registered policy."""
        with self._lock:
            self._policies = []
            self._next_order = 0

    def policy_ids(self) -> tuple[str, ...]:
        """Snapshot of registered policy ids in registration order."""
        with self._lock:
            return tuple(p.policy_id for p in self._policies)

    def has_policies(self) -> bool:
        """``True`` when at least one policy is registered."""
        with self._lock:
            return bool(self._policies)

    def run(self, ctx: ToolCallContext) -> ChainResult:
        """Run every registered policy against ``ctx``.

        Semantics:

        * Block terminates the chain immediately.
        * Params-rewrites stack: each policy sees the latest params.
          When a policy rewrites params, it's recorded under
          ``rewrites_by``.
        * Approval first-wins: only the first ``require_approval``
          is honoured; later approvals are discarded with a debug log.

        Returns:
            :class:`ChainResult` summarising the outcome.
        """
        import time as _time
        with self._lock:
            snapshot = list(self._policies)
        current_params: Mapping[str, Any] = ctx.arguments
        rewrites: list[str] = []
        decisions: list[tuple[str, PolicyDecision]] = []
        approval: Optional[ApprovalRequest] = None
        block_decision: Optional[PolicyDecision] = None
        for reg in snapshot:
            # Build a per-policy context that carries the latest params.
            inner_ctx = ToolCallContext(
                tool_name=ctx.tool_name,
                arguments=current_params,
                capability=ctx.capability,
                user_text=ctx.user_text,
                mode=ctx.mode,
                session_id=ctx.session_id,
                _session_state=ctx._session_state,
            )
            start = _time.perf_counter()
            try:
                decision = reg.policy.evaluate(inner_ctx)
            except Exception:  # noqa: BLE001
                LOGGER.warning(
                    "trusted tool policy %s raised; treating as pass-through",
                    reg.policy_id, exc_info=True,
                )
                decision = None
            duration_ms = (_time.perf_counter() - start) * 1000.0
            if duration_ms > self._slow_policy_warn_ms:
                LOGGER.warning(
                    "trusted tool policy %s slow (%.1f ms > %.1f ms)",
                    reg.policy_id, duration_ms, self._slow_policy_warn_ms,
                )
            if decision is None:
                continue
            decisions.append((reg.policy_id, decision))
            # Block terminates.
            if decision.allow is False:
                block_decision = decision
                return ChainResult(
                    outcome=PolicyOutcome.BLOCK,
                    final_params=current_params,
                    block_decision=block_decision,
                    approval=approval,
                    rewrites_by=tuple(rewrites),
                    decisions=tuple(decisions),
                )
            # Rewrite stacks.
            if decision.params is not None:
                current_params = decision.params
                rewrites.append(reg.policy_id)
            # Approval: first wins.
            if decision.require_approval is not None and approval is None:
                approval = decision.require_approval

        if approval is not None:
            outcome = PolicyOutcome.REQUIRE_APPROVAL
        elif rewrites:
            outcome = PolicyOutcome.REWRITE
        else:
            outcome = PolicyOutcome.PASS_THROUGH

        return ChainResult(
            outcome=outcome,
            final_params=current_params,
            block_decision=None,
            approval=approval,
            rewrites_by=tuple(rewrites),
            decisions=tuple(decisions),
        )


@dataclass
class FunctionPolicy:
    """Adapter that wraps a plain callable as a chain policy.

    Convenience for inline registration without subclassing.

    Example::

        chain.register(FunctionPolicy(
            policy_id="strip_pii",
            evaluate=lambda ctx: PolicyDecision(params=_redact(ctx.arguments)),
        ))
    """

    policy_id: str
    evaluate: Callable[[ToolCallContext], Optional[PolicyDecision]]


# ----------------------------------------------------------------------
# Module-level singleton accessors


_chain_singleton: Optional[TrustedToolPolicyChain] = None
_chain_lock = threading.Lock()


def get_policy_chain() -> TrustedToolPolicyChain:
    """Module-level :class:`TrustedToolPolicyChain` singleton.

    Returns the previously-set chain or lazily-constructs an empty
    one. Use :func:`set_policy_chain` to inject a customised
    instance during tests.
    """
    global _chain_singleton
    with _chain_lock:
        if _chain_singleton is None:
            _chain_singleton = TrustedToolPolicyChain()
        return _chain_singleton


def set_policy_chain(chain: TrustedToolPolicyChain) -> None:
    """Replace the module-level chain. Use during init / tests."""
    global _chain_singleton
    with _chain_lock:
        _chain_singleton = chain


def reset_policy_chain_for_testing() -> None:
    """Drop the singleton so the next :func:`get_policy_chain` returns fresh."""
    global _chain_singleton
    with _chain_lock:
        _chain_singleton = None


__all__ = [
    "ApprovalRequest",
    "ChainResult",
    "DEFAULT_SLOW_POLICY_WARN_MS",
    "FunctionPolicy",
    "Policy",
    "PolicyDecision",
    "PolicyOutcome",
    "RegisteredPolicy",
    "ToolCallContext",
    "TrustedToolPolicyChain",
    "get_policy_chain",
    "reset_policy_chain_for_testing",
    "set_policy_chain",
]
