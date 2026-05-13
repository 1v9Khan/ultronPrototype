"""Runtime tool-call validator for Ultron.

Pairs with the abliterated default LLM (Josiefied-Qwen3-8B,
`llm.preset = "josiefied-qwen3-8b"`) to gate the actual capability
surface even when the model is willing to attempt anything at the
content level. The user's restriction list (see the 2026-05-12
conversation transcript) defines the policy; this package
implements the engine that enforces it.

Top-level public API:

* :class:`Verdict` -- one of ``ALLOW``, ``BLOCK_HARD``,
  ``NEEDS_EXPLICIT_INTENT``, ``LOG_ONLY``. Returned by every rule.
* :class:`RuleResult` -- a single rule's decision: verdict + reason
  + rule id + matched context fields.
* :class:`RuleContext` -- frozen snapshot of what the model is
  trying to do: the tool name, the arguments, the originating
  capability (coding bridge, OpenClaw, MCP, etc.), the resolved
  path(s), the user's most recent utterance for explicit-intent
  matching.
* :class:`ToolCallValidator` -- the dispatcher. Runs every rule in
  priority order; the most-restrictive verdict wins. Fail-closed:
  any rule exception is treated as ``BLOCK_HARD``.
* :data:`get_validator()` -- module-level singleton accessor used
  by call sites (orchestrator, dispatcher, coding runner). Returns
  the configured validator, or a permissive no-op validator when
  ``safety.enabled = false``.

Design principles:

* **Fail-closed.** Unknown tools, unknown command shapes,
  unparseable arguments -- deny. The validator is the last line
  for the abliterated model, so the default for the unrecognised
  case is deny, not allow.
* **Canonicalize first.** Paths are resolved (symlinks, junctions,
  8.3 short names) BEFORE allowlist/denylist comparison. The
  reverse direction is a common bypass.
* **Audit everything.** Every block, every allow-with-warn, every
  explicit-intent override goes to ``logs/safety_audit.jsonl``.
  In Phase 5 the audit log gains a tamper-evident hash chain; for
  Phase 2 it's an append-only JSONL with fsync per entry.
* **Layered defense.** Rules are organised by category. Category K
  rules protect the validator itself from being disabled by the
  model. Categories A-S enforce the user-approved restrictions.
* **Config-driven.** Each rule has an entry in
  ``config.yaml:safety.rules`` -- the user can toggle individual
  rules without code changes.

Phase 2 (this module's first land) ships:

* The core types (Verdict / RuleResult / RuleContext / Rule).
* Windows-aware path canonicalization (`PathResolver`).
* Append-only audit log (`AuditLog`).
* Policy loader (`Policy`).
* The validator dispatcher (`ToolCallValidator`).
* Category K rules (self-protection -- airtight first so the model
  can't disable the validator before later phases land).

Phases 3-5 extend with the rest of the categories + cross-cutting
concerns (taint tracking, hash-chain audit, explicit-intent
matcher).
"""

from __future__ import annotations

from ultron.safety.audit import AuditLog, get_audit_log
from ultron.safety.path_resolver import PathResolver
from ultron.safety.policy import Policy, load_policy
from ultron.safety.rules.base import Rule
from ultron.safety.validator import (
    RuleContext,
    RuleResult,
    ToolCallValidator,
    Verdict,
    get_validator,
    set_validator,
)

__all__ = [
    "AuditLog",
    "PathResolver",
    "Policy",
    "Rule",
    "RuleContext",
    "RuleResult",
    "ToolCallValidator",
    "Verdict",
    "get_audit_log",
    "get_validator",
    "load_policy",
    "set_validator",
]
