"""Tool-call validator core.

The validator sits between the model's output and the actual
dispatch machinery (OpenClaw bridge, coding bridge, MCP tools, file
ops). Every potentially-dangerous tool call passes through
:meth:`ToolCallValidator.check` first; if any rule returns
``BLOCK_HARD``, the call is denied and an audit entry is written.

Design:

* **Rule registry.** Rules are registered at construction time. The
  validator iterates ALL rules per call -- there's no early-exit on
  the first allow (because a different rule might still block).
* **Most-restrictive wins.** Aggregated verdict is
  ``BLOCK_HARD > NEEDS_EXPLICIT_INTENT > LOG_ONLY > ALLOW``.
* **Fail-closed on exceptions.** Any rule raising in its
  :meth:`Rule.evaluate` is logged and treated as ``BLOCK_HARD``.
  Better to deny a legit call than miss a malicious one.
* **Audit every non-ALLOW verdict.** ``LOG_ONLY`` and above go to
  ``logs/safety_audit.jsonl``. ``ALLOW`` does not (would flood the
  log).
* **Per-rule disable via config.** :meth:`Policy.is_rule_enabled`
  is consulted before each rule's evaluate; disabled rules return
  ``ALLOW`` without running.

Singleton pattern: the orchestrator constructs one validator at
startup via :func:`build_validator_from_config` and sets it as the
module singleton via :func:`set_validator`. Call sites (coding
bridge, OpenClaw dispatcher, etc.) read via :func:`get_validator`.

Construction is intentionally explicit -- the validator does NOT
self-construct on first access. If :func:`get_validator` is called
before construction, you get a permissive no-op validator that
returns ``ALLOW`` for everything AND logs a WARN. This is the
graceful-degrade path for tests / unit modules that import safety
without a full orchestrator init.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from ultron.safety.audit import AuditLog, get_audit_log
from ultron.safety.path_resolver import PathResolver, get_path_resolver
from ultron.safety.policy import Policy

logger = logging.getLogger("ultron.safety.validator")


class Verdict(Enum):
    """Per-rule and aggregated verdicts.

    Aggregation rule (most-restrictive wins):
        BLOCK_HARD > NEEDS_EXPLICIT_INTENT > LOG_ONLY > ALLOW

    Semantic meaning:

    * ``ALLOW`` -- rule has no opinion; let the call proceed.
    * ``LOG_ONLY`` -- call is allowed but the audit log records it
      (used for ``▲`` items in the user's restriction list -- legit
      operations worth logging).
    * ``NEEDS_EXPLICIT_INTENT`` -- call is blocked unless the user's
      most recent utterance contains explicit verb+object matching
      the action (Phase 4 lands the explicit-intent matcher; Phase
      2 treats this as a block).
    * ``BLOCK_HARD`` -- call is denied. The dispatcher returns an
      in-character refusal message; the model can try a different
      approach.
    """

    ALLOW = "ALLOW"
    LOG_ONLY = "LOG_ONLY"
    NEEDS_EXPLICIT_INTENT = "NEEDS_EXPLICIT_INTENT"
    BLOCK_HARD = "BLOCK_HARD"

    @property
    def severity(self) -> int:
        return {
            Verdict.ALLOW: 0,
            Verdict.LOG_ONLY: 1,
            Verdict.NEEDS_EXPLICIT_INTENT: 2,
            Verdict.BLOCK_HARD: 3,
        }[self]


@dataclass(frozen=True)
class RuleResult:
    """One rule's decision.

    The ``reason`` field is INTERNAL — it appears in the audit log
    and ops dashboards but is never spoken to the user verbatim (it
    may include matched patterns, file paths, regex fragments). When
    a rule wants a different string spoken to the user, it sets
    ``user_message`` separately.

    The ``category`` field is the analytics label (e.g. ``"pii"`` /
    ``"cost_limit"`` / ``"violence"`` / ``"voice_lock"``). Categories
    let the audit dashboard group blocks without hard-coding the
    taxonomy into the validator core. The ``metadata`` field is an
    opaque per-rule blob for whatever cross-call state the rule
    needs (e.g. "user has been warned about this category 3 times
    today").

    T16 (OpenClaw catalog port; see ``THIRD_PARTY_NOTICES.md``):
    the optional ``user_message`` / ``category`` / ``metadata``
    split mirrors OpenClaw's ``HookDecisionBlock`` discriminated
    union shape.
    """

    rule_id: str
    verdict: Verdict
    reason: str
    context: dict[str, Any] = field(default_factory=dict)
    user_message: Optional[str] = None
    category: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class RuleContext:
    """Frozen snapshot of what the model is trying to do.

    Attributes:
        tool_name: e.g. ``openclaw.browser.navigate``, ``mcp.write_file``,
            ``coding_bridge.spawn``, ``file_op.write``. Identifies the
            tool category for rule routing.
        arguments: keyword arguments. Rules read from this. Best-effort
            JSON-friendly; non-serialisable values are stringified for
            audit.
        capability: which dispatch surface the call originated from --
            ``coding_bridge``, ``openclaw_dispatcher``, ``mcp_tool``,
            ``file_op``, ``shell``. Used by rules that gate per-
            capability.
        paths: pre-canonicalised :class:`Path` list. Caller is
            responsible for extracting path-shaped arguments from
            ``arguments`` and passing them here so rules don't each
            re-canonicalise. May be empty for non-filesystem tools.
        user_text: the user's most recent utterance text (or empty
            string). Used by the explicit-intent matcher (Phase 4).
        has_pending_clarification: from
            :meth:`CapabilityVoiceController.has_pending_clarification`.
            Some rules suppress while a clarification is active.
    """

    tool_name: str
    arguments: dict[str, Any]
    capability: str
    paths: tuple[Path, ...] = ()
    user_text: str = ""
    has_pending_clarification: bool = False


@dataclass(frozen=True)
class ValidatorVerdict:
    """Aggregated outcome of running all rules.

    Returned from :meth:`ToolCallValidator.check`. The dispatcher
    inspects ``verdict`` and ``reason`` to decide:

    * ``ALLOW`` / ``LOG_ONLY`` -- proceed.
    * ``BLOCK_HARD`` / ``NEEDS_EXPLICIT_INTENT`` -- refuse; speak
      ``user_message`` to the user; do NOT dispatch.

    Attributes:
        verdict: aggregated verdict (most-restrictive of all rules).
        reason: short reason, used for logs + user-facing message.
        triggered_rule_id: the rule that produced the dominating
            verdict (most-restrictive rule's id).
        user_message: in-character refusal message the dispatcher
            speaks when ``verdict != ALLOW``. Empty when verdict is
            ALLOW.
        rule_results: list of every rule's individual result -- for
            audit and debugging.
    """

    verdict: Verdict
    reason: str
    triggered_rule_id: str = ""
    user_message: str = ""
    rule_results: tuple[RuleResult, ...] = ()
    category: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    @property
    def is_allowed(self) -> bool:
        """True iff dispatcher should proceed.

        ``LOG_ONLY`` counts as allowed (the operation runs; we just
        log it). ``NEEDS_EXPLICIT_INTENT`` does NOT count as allowed
        in Phase 2 -- the explicit-intent matcher lands in Phase 4;
        until then NEI is treated as a hard block.
        """
        return self.verdict in (Verdict.ALLOW, Verdict.LOG_ONLY)


class ToolCallValidator:
    """Run a registered set of rules against a tool-call context.

    The validator is thread-safe (rule evaluation is read-only; the
    audit log writer has its own lock).

    Construction:

        validator = ToolCallValidator(
            policy=load_policy(),
            rules=[K1_ConfigYamlProtection(), K2_VoiceAssetProtection(), ...],
            audit_log=get_audit_log(),
        )

    Typical call site::

        ctx = RuleContext(
            tool_name="openclaw.file.write",
            arguments={"path": user_supplied_path, "content": payload},
            capability="openclaw_dispatcher",
            paths=(resolver.resolve(user_supplied_path),),
            user_text=last_user_utterance,
            has_pending_clarification=has_clar,
        )
        result = validator.check(ctx)
        if not result.is_allowed:
            speak(result.user_message)
            return refusal
        # ... dispatch ...
    """

    def __init__(
        self,
        *,
        policy: Policy,
        rules: list,  # list[Rule] but avoid circular import at type level
        audit_log: Optional[AuditLog] = None,
        path_resolver: Optional[PathResolver] = None,
        explicit_intent_matching: bool = True,
    ) -> None:
        self.policy = policy
        self.rules: list = list(rules)
        self.audit_log = audit_log if audit_log is not None else get_audit_log()
        self.path_resolver = (
            path_resolver if path_resolver is not None else get_path_resolver()
        )
        # When True, a NEEDS_EXPLICIT_INTENT verdict is conditionally upgraded
        # to ALLOW when the user's current utterance explicitly names the
        # action. Never overrides BLOCK_HARD. See check().
        self._explicit_intent_enabled = bool(explicit_intent_matching)

    def check(self, ctx: RuleContext) -> ValidatorVerdict:
        """Run every enabled rule against ``ctx`` and aggregate.

        Returns an :class:`ValidatorVerdict` containing the
        most-restrictive verdict, the rule that produced it, and the
        list of every rule's individual result. Audit entries for any
        non-ALLOW verdict are written to the audit log before this
        method returns.
        """
        if not self.policy.enabled:
            # Master kill-switch: validator is a permissive no-op.
            return ValidatorVerdict(
                verdict=Verdict.ALLOW,
                reason="safety.enabled=false",
            )

        # Anticheat-safe mode (2026-06-11): while active, EVERY desktop
        # tool class (input injection / screen capture / window
        # manipulation / clipboard / dialog / element / browser
        # automation) is hard-blocked BEFORE rule evaluation -- this is
        # the audit-trail layer on top of the per-module guards, so
        # every blocked attempt lands in the ledger. Fail-open on probe
        # errors (the module guards still apply).
        try:
            from ultron.safety.anticheat import (
                BLOCKED_NOTICE,
                anticheat_active,
                is_blocked_tool,
            )

            if anticheat_active() and is_blocked_tool(ctx.tool_name):
                reason = (
                    "anticheat-safe mode active: desktop-interaction "
                    f"tool {ctx.tool_name!r} is disabled in game"
                )
                try:
                    self.audit_log.record(
                        rule_id="anticheat_safe_mode",
                        verdict=Verdict.BLOCK_HARD.value,
                        tool_name=ctx.tool_name,
                        capability=ctx.capability,
                        reason=reason,
                        context={"user_text_preview": ctx.user_text[:120]},
                    )
                except Exception as e:
                    logger.warning("safety audit write failed: %s", e)
                return ValidatorVerdict(
                    verdict=Verdict.BLOCK_HARD,
                    reason=reason,
                    triggered_rule_id="anticheat_safe_mode",
                    user_message=BLOCKED_NOTICE,
                )
        except Exception as e:                                       # noqa: BLE001
            logger.debug("anticheat pre-check failed open: %s", e)

        results: list[RuleResult] = []
        for rule in self.rules:
            rule_id = getattr(rule, "rule_id", rule.__class__.__name__)
            if not self.policy.is_rule_enabled(rule_id):
                continue
            try:
                r = rule.evaluate(ctx, policy=self.policy, resolver=self.path_resolver)
            except Exception as e:
                # Fail-closed: a buggy rule defaults to deny rather
                # than allow. The bug needs to surface clearly so the
                # audit log gets a detailed entry and the standard
                # logger records the traceback at WARN.
                logger.warning(
                    "safety rule %s raised %s during evaluate; "
                    "failing closed (treating as BLOCK_HARD): %s",
                    rule_id, type(e).__name__, e,
                )
                r = RuleResult(
                    rule_id=rule_id,
                    verdict=Verdict.BLOCK_HARD,
                    reason=(
                        f"safety rule {rule_id} crashed during evaluate "
                        f"({type(e).__name__}); failing closed"
                    ),
                )
            if not isinstance(r, RuleResult):
                # Defensive: a misbehaving rule that returns
                # something other than RuleResult. Treat as block.
                logger.warning(
                    "safety rule %s returned non-RuleResult %r; "
                    "failing closed (treating as BLOCK_HARD)",
                    rule_id, type(r).__name__,
                )
                r = RuleResult(
                    rule_id=rule_id,
                    verdict=Verdict.BLOCK_HARD,
                    reason=(
                        f"safety rule {rule_id} returned malformed result; "
                        f"failing closed"
                    ),
                )
            results.append(r)

        if not results:
            # No rules ran -- fail open in the sense of "no policy to
            # enforce". Different from the master kill-switch above:
            # this can happen during tests where the validator is
            # constructed without any rules.
            return ValidatorVerdict(
                verdict=Verdict.ALLOW,
                reason="no rules registered",
                rule_results=tuple(),
            )

        # Find the most-restrictive verdict and the rule that produced it.
        dominant = max(results, key=lambda r: r.verdict.severity)

        # Conditional unblock: upgrade NEEDS_EXPLICIT_INTENT -> LOG_ONLY (an
        # audited allow) iff the user's CURRENT utterance explicitly names the
        # action (verb + object). Critically this NEVER overrides a BLOCK_HARD
        # (we only rewrite NEI results, then recompute the dominant -- so any
        # hard block still wins) and only consults ctx.user_text, so it cannot
        # open anything the user didn't ask for this turn. The matcher is
        # conservative (prefers false-negatives). Gated by
        # safety.explicit_intent_matching_enabled.
        if (
            self._explicit_intent_enabled
            and dominant.verdict == Verdict.NEEDS_EXPLICIT_INTENT
            and getattr(ctx, "user_text", "")
        ):
            try:
                from ultron.safety.intent import matches_explicit_intent
                hints = tuple(
                    Path(str(p)).name for p in ctx.paths if str(p)
                )
                im = matches_explicit_intent(
                    ctx.user_text, tool_name=ctx.tool_name, object_hints=hints,
                )
            except Exception as e:  # noqa: BLE001
                im = None
                logger.debug("explicit-intent matcher raised (%s); NEI stands", e)
            if im is not None and im.matched:
                upgraded: list[RuleResult] = []
                for r in results:
                    if r.verdict == Verdict.NEEDS_EXPLICIT_INTENT:
                        upgraded.append(RuleResult(
                            rule_id=r.rule_id,
                            verdict=Verdict.LOG_ONLY,
                            reason=(
                                f"explicit intent granted ({r.rule_id}): "
                                f"verb={im.verb!r} object={im.object_token!r}"
                            ),
                            context={
                                "original_verdict": "NEEDS_EXPLICIT_INTENT",
                                "matched_verb": im.verb,
                                "matched_object": im.object_token,
                            },
                        ))
                    else:
                        upgraded.append(r)
                results = upgraded
                # Recompute: any BLOCK_HARD from another rule still dominates.
                dominant = max(results, key=lambda r: r.verdict.severity)

        # Audit any non-ALLOW outcome.
        if dominant.verdict != Verdict.ALLOW:
            try:
                # T16: include category + metadata in audit context so
                # the audit log can group blocks by analytics label
                # without re-parsing the reason string.
                audit_context: dict[str, Any] = {
                    "arguments_keys": sorted(ctx.arguments.keys()),
                    "paths": [str(p) for p in ctx.paths],
                    "user_text_preview": ctx.user_text[:120],
                    "rule_details": dominant.context,
                }
                if dominant.category is not None:
                    audit_context["category"] = dominant.category
                if dominant.metadata is not None:
                    audit_context["rule_metadata"] = dominant.metadata
                self.audit_log.record(
                    rule_id=dominant.rule_id,
                    verdict=dominant.verdict.value,
                    tool_name=ctx.tool_name,
                    capability=ctx.capability,
                    reason=dominant.reason,
                    context=audit_context,
                )
            except Exception as e:
                # Audit failure must not block the verdict.
                logger.warning("safety audit write failed: %s", e)

        # Construct the user-facing message. T16: prefer the rule's
        # explicit user_message (clean, free of internal pattern
        # details) when present; otherwise synthesise from reason.
        user_message = ""
        if dominant.verdict == Verdict.BLOCK_HARD:
            if dominant.user_message:
                user_message = dominant.user_message
            else:
                user_message = f"I held off on that. {dominant.reason}"
        elif dominant.verdict == Verdict.NEEDS_EXPLICIT_INTENT:
            if dominant.user_message:
                user_message = dominant.user_message
            else:
                user_message = (
                    "I won't do that without you explicitly asking for it. "
                    f"{dominant.reason}"
                )

        # Evolution reach-signal (#63): notify the registered observer of a
        # hard block so the self-improvement loop can learn from repeated
        # refusals ("ultron keeps attempting X which is blocked" distils a
        # DEFENSIVE skill). Pure observation -- runs AFTER the verdict +
        # audit are final, never alters them, and any observer exception is
        # swallowed (the validator stays fail-closed on its own logic and
        # the observer can never weaken or strengthen a verdict).
        if dominant.verdict == Verdict.BLOCK_HARD:
            observer = _block_observer
            if observer is not None:
                try:
                    observer(ctx.tool_name, dominant.reason)
                except Exception as e:  # noqa: BLE001
                    logger.debug("block observer failed: %s", e)

        return ValidatorVerdict(
            verdict=dominant.verdict,
            reason=dominant.reason,
            triggered_rule_id=dominant.rule_id,
            user_message=user_message,
            rule_results=tuple(results),
            category=dominant.category,
            metadata=dominant.metadata,
        )


# ---------------------------------------------------------------------------
# Block observer (evolution reach-signal #63)
# ---------------------------------------------------------------------------

#: Optional ``(tool_name, reason) -> None`` callback fired on every
#: BLOCK_HARD verdict. Observation only -- it cannot affect verdicts.
_block_observer: Optional[Callable[[str, str], None]] = None


def set_block_observer(observer: Optional[Callable[[str, str], None]]) -> None:
    """Register (or clear, with ``None``) the BLOCK_HARD observer.

    The orchestrator registers a bounded-queue enqueue here so the
    evolution service can learn from repeated hard blocks. The observer
    is called AFTER the verdict + audit entry are final and is wrapped
    fail-open at the call site -- it can never alter a verdict, block a
    call, or raise into the validator.
    """
    global _block_observer
    _block_observer = observer


class _NoOpValidator:
    """Permissive fallback used when no validator has been configured.

    Returned by :func:`get_validator` before
    :func:`set_validator` has been called. Logs a WARN on first use
    so the operator sees the safety layer isn't active.
    """

    _warned: bool = False

    def check(self, ctx: RuleContext) -> ValidatorVerdict:  # noqa: ARG002
        if not _NoOpValidator._warned:
            logger.warning(
                "tool-call validator is not configured; "
                "every call is being ALLOWED (no safety enforcement). "
                "Construct a ToolCallValidator and call set_validator() "
                "during orchestrator init."
            )
            _NoOpValidator._warned = True
        return ValidatorVerdict(
            verdict=Verdict.ALLOW,
            reason="validator not configured (no-op)",
        )


_validator_singleton: object = _NoOpValidator()
_validator_lock = threading.Lock()


def get_validator():
    """Module-level singleton accessor.

    Returns whatever was set via :func:`set_validator`, or a
    :class:`_NoOpValidator` if nothing has been configured yet. The
    no-op variant logs a WARN on first use so missing init is visible.
    """
    return _validator_singleton


def set_validator(validator) -> None:
    """Set the module-level validator.

    Call once during orchestrator init. Subsequent calls replace the
    singleton; this is intentional for test scenarios (swap a
    permissive validator in for one test, restore after).

    Pass ``None`` to reset to the no-op fallback.
    """
    global _validator_singleton
    with _validator_lock:
        if validator is None:
            _validator_singleton = _NoOpValidator()
        else:
            _validator_singleton = validator


def build_validator_from_config() -> ToolCallValidator:
    """Construct the production validator from :class:`SafetyConfig`
    + the in-code rule list.

    Reads ``cfg.safety`` for per-rule toggles, sandbox-root + protected-
    path overrides, audit-log path, and the master enable switch. Falls
    back to ``load_policy()`` defaults when config access fails (so this
    function remains importable from minimal test envs).

    The returned validator's audit log is targeted at the config-specified
    path; tests can override by constructing a :class:`ToolCallValidator`
    directly with a different :class:`AuditLog` instance.

    Imports rule modules lazily so this function stays importable when
    those modules aren't on the path (e.g. minimal test envs).
    """
    from ultron.safety.policy import load_policy
    from ultron.safety.rules.cap_carveouts import build_capability_rules
    from ultron.safety.rules.category_a import build_category_a_rules
    from ultron.safety.rules.category_b import build_category_b_rules
    from ultron.safety.rules.category_c import build_category_c_rules
    from ultron.safety.rules.category_d import build_category_d_rules
    from ultron.safety.rules.category_e import build_category_e_rules
    from ultron.safety.rules.category_f import build_category_f_rules
    from ultron.safety.rules.category_g import build_category_g_rules
    from ultron.safety.rules.category_h import build_category_h_rules
    from ultron.safety.rules.category_i import build_category_i_rules
    from ultron.safety.rules.category_j import build_category_j_rules
    from ultron.safety.rules.category_k import build_category_k_rules
    from ultron.safety.rules.category_m import build_category_m_rules
    from ultron.safety.rules.category_n import build_category_n_rules
    from ultron.safety.rules.category_o import build_category_o_rules
    from ultron.safety.rules.category_p import build_category_p_rules
    from ultron.safety.rules.category_q import build_category_q_rules
    from ultron.safety.rules.category_r import build_category_r_rules
    from ultron.safety.rules.category_it import build_category_it_rules
    from ultron.safety.rules.category_s import build_category_s_rules

    rules = []
    # Category K first -- self-protection. The order matters for audit
    # logs (the first matching rule's id is the dominant when multiple
    # rules return the same verdict severity).
    rules.extend(build_category_k_rules())
    # Category U: .ultronignore path/command block (secrets protection).
    # Default-safe -- a no-op until the user creates a .ultronignore. Placed
    # after K so self-protection still reports first on a tie.
    try:
        from ultron.safety.rules.category_ignore import build_ignore_rules
        rules.extend(build_ignore_rules())
    except Exception:  # noqa: BLE001 -- keep the validator importable
        pass
    # Categories A-J: load-bearing safety items.
    rules.extend(build_category_a_rules())
    rules.extend(build_category_b_rules())
    rules.extend(build_category_c_rules())
    rules.extend(build_category_d_rules())
    rules.extend(build_category_e_rules())
    rules.extend(build_category_f_rules())
    rules.extend(build_category_g_rules())
    rules.extend(build_category_h_rules())
    rules.extend(build_category_i_rules())
    rules.extend(build_category_j_rules())
    # Categories M-S: persistence, process manipulation, anti-forensics,
    # AV tampering, containers, sensors, AI tampering.
    rules.extend(build_category_m_rules())
    rules.extend(build_category_n_rules())
    rules.extend(build_category_o_rules())
    rules.extend(build_category_p_rules())
    rules.extend(build_category_q_rules())
    rules.extend(build_category_r_rules())
    rules.extend(build_category_s_rules())
    # Category IT (SWE-Agent batch 4 / catalog T11): block hang-prone
    # interactive shell commands (vim / less / bare python / tail -f /
    # python -m venv / make etc.) before they reach the path resolver.
    # Fail-open: any config-error in the IT category builder leaves
    # the rule list unchanged.
    try:
        from ultron.config import get_config as _it_get_config
        cfg_it = _it_get_config().safety
        it_block = getattr(cfg_it, "interactive_tools", None)
        if it_block is not None:
            from ultron.safety.rules.category_it import InteractiveToolsConfig
            it_cfg = InteractiveToolsConfig(
                enabled=bool(getattr(it_block, "enabled", True)),
                prefix_blocklist=list(
                    getattr(it_block, "prefix_blocklist", None) or []
                )
                or list(__import__(
                    "ultron.safety.rules.category_it",
                    fromlist=["DEFAULT_PREFIX_BLOCKLIST"],
                ).DEFAULT_PREFIX_BLOCKLIST),
                standalone_blocklist=list(
                    getattr(it_block, "standalone_blocklist", None) or []
                )
                or list(__import__(
                    "ultron.safety.rules.category_it",
                    fromlist=["DEFAULT_STANDALONE_BLOCKLIST"],
                ).DEFAULT_STANDALONE_BLOCKLIST),
                unless_regex=dict(
                    getattr(it_block, "unless_regex", None) or {}
                )
                or dict(__import__(
                    "ultron.safety.rules.category_it",
                    fromlist=["DEFAULT_UNLESS_REGEX"],
                ).DEFAULT_UNLESS_REGEX),
                block_message=str(
                    getattr(it_block, "block_message", None)
                    or __import__(
                        "ultron.safety.rules.category_it",
                        fromlist=["DEFAULT_BLOCK_MESSAGE"],
                    ).DEFAULT_BLOCK_MESSAGE
                ),
            )
            rules.extend(build_category_it_rules(it_cfg))
        else:
            rules.extend(build_category_it_rules())
    except Exception:
        rules.extend(build_category_it_rules())
    # Capability carve-outs (Cap-1 .. Cap-4 sub-rules).
    rules.extend(build_capability_rules())

    try:
        from ultron.config import get_config
        from ultron.safety.audit import AuditLog
        cfg = get_config()
        safety_cfg = cfg.safety
        policy = load_policy(
            enabled=bool(safety_cfg.enabled),
            rule_overrides=dict(safety_cfg.rules or {}),
            extra_protected_files=list(safety_cfg.extra_protected_files or []),
            extra_protected_dirs=list(safety_cfg.extra_protected_dirs or []),
            sandbox_roots=list(safety_cfg.sandbox_roots or []),
            screen_cache_dir=safety_cfg.screen_cache_dir,
            approved_outbound_apis=list(safety_cfg.approved_outbound_apis or []),
        )
        audit_log = AuditLog(path=safety_cfg.audit_log_path)
        return ToolCallValidator(
            policy=policy,
            rules=rules,
            audit_log=audit_log,
            explicit_intent_matching=bool(
                getattr(safety_cfg, "explicit_intent_matching_enabled", True)
            ),
        )
    except Exception as e:
        logger.warning(
            "validator construction from config failed (%s); "
            "falling back to in-code defaults", e,
        )
        policy = load_policy()
        return ToolCallValidator(policy=policy, rules=rules)
