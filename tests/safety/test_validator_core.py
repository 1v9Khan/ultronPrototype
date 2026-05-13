"""Tests for the validator core dispatcher + policy + rule base classes."""

from __future__ import annotations

import pytest

from ultron.safety import (
    Policy,
    RuleContext,
    RuleResult,
    ToolCallValidator,
    Verdict,
    load_policy,
    set_validator,
)
from ultron.safety.audit import AuditLog
from ultron.safety.rules.base import (
    CommandPatternRule,
    PathPatternRule,
    PathSetRule,
    Rule,
    SandboxConfinementRule,
    ToolNameRule,
)


def _make_validator(rules, *, enabled=True, rule_enabled=None, tmp_path=None):
    policy = Policy(
        enabled=enabled,
        rule_enabled=rule_enabled or {},
    )
    audit_path = (tmp_path / "audit.jsonl") if tmp_path is not None else None
    audit = AuditLog(path=audit_path) if audit_path is not None else None
    return ToolCallValidator(policy=policy, rules=rules, audit_log=audit)


def test_verdict_severity_ordering():
    assert Verdict.ALLOW.severity < Verdict.LOG_ONLY.severity
    assert Verdict.LOG_ONLY.severity < Verdict.NEEDS_EXPLICIT_INTENT.severity
    assert Verdict.NEEDS_EXPLICIT_INTENT.severity < Verdict.BLOCK_HARD.severity


def test_empty_rules_returns_allow():
    v = _make_validator([])
    ctx = RuleContext(tool_name="t", arguments={}, capability="c")
    r = v.check(ctx)
    assert r.verdict == Verdict.ALLOW
    assert "no rules" in r.reason


def test_master_kill_switch_returns_allow(tmp_path):
    blocker = CommandPatternRule(
        rule_id="X1", description="always-block",
        category="X", patterns=[r".*"],
    )
    v = _make_validator([blocker], enabled=False, tmp_path=tmp_path)
    ctx = RuleContext(tool_name="anything", arguments={"x": "y"}, capability="c")
    r = v.check(ctx)
    assert r.verdict == Verdict.ALLOW
    assert "safety.enabled=false" in r.reason


def test_rule_disabled_via_policy_skipped():
    blocker = CommandPatternRule(
        rule_id="X1", description="always-block",
        category="X", patterns=[r".*"],
    )
    v = _make_validator([blocker], rule_enabled={"X1": False})
    ctx = RuleContext(tool_name="t", arguments={"command": "anything"}, capability="c")
    r = v.check(ctx)
    assert r.verdict == Verdict.ALLOW


def test_most_restrictive_wins(tmp_path):
    allow_rule = CommandPatternRule(
        rule_id="A", description="allow",
        category="X", patterns=[r"^never$"],   # won't match
    )
    log_rule = CommandPatternRule(
        rule_id="B", description="log only",
        category="X", patterns=[r"hello"],
        verdict_on_match=Verdict.LOG_ONLY,
    )
    block_rule = CommandPatternRule(
        rule_id="C", description="block",
        category="X", patterns=[r"hello"],
        verdict_on_match=Verdict.BLOCK_HARD,
    )
    v = _make_validator([allow_rule, log_rule, block_rule], tmp_path=tmp_path)
    ctx = RuleContext(tool_name="t", arguments={"x": "hello world"}, capability="c")
    r = v.check(ctx)
    assert r.verdict == Verdict.BLOCK_HARD
    assert r.triggered_rule_id == "C"


def test_rule_exception_fails_closed(tmp_path):
    class _Boom(Rule):
        rule_id = "BOOM"
        description = "raises"
        category = "X"

        def evaluate(self, ctx, *, policy, resolver):  # noqa: ARG002
            raise RuntimeError("intentional")

    v = _make_validator([_Boom()], tmp_path=tmp_path)
    ctx = RuleContext(tool_name="t", arguments={}, capability="c")
    r = v.check(ctx)
    assert r.verdict == Verdict.BLOCK_HARD
    assert "failing closed" in r.reason.lower() or "crashed" in r.reason.lower()


def test_rule_non_ruleresult_return_fails_closed(tmp_path):
    class _Bad(Rule):
        rule_id = "BAD"
        description = "returns wrong type"
        category = "X"

        def evaluate(self, ctx, *, policy, resolver):  # noqa: ARG002
            return "not a RuleResult"  # type: ignore[return-value]

    v = _make_validator([_Bad()], tmp_path=tmp_path)
    ctx = RuleContext(tool_name="t", arguments={}, capability="c")
    r = v.check(ctx)
    assert r.verdict == Verdict.BLOCK_HARD


def test_user_message_set_on_block(tmp_path):
    blocker = CommandPatternRule(
        rule_id="X1", description="always-block",
        category="X", patterns=[r"x"],
    )
    v = _make_validator([blocker], tmp_path=tmp_path)
    ctx = RuleContext(tool_name="t", arguments={"x": "x"}, capability="c")
    r = v.check(ctx)
    assert not r.is_allowed
    assert "held off" in r.user_message.lower()


def test_explicit_intent_treated_as_block_until_phase_5_matcher(tmp_path):
    """Phase 2 contract: ``NEEDS_EXPLICIT_INTENT`` blocks the call.
    The explicit-intent matcher landed in Phase 5 but Validator.check
    treats NEI as not-allowed unless the dispatcher consults the
    matcher itself."""
    nei_rule = CommandPatternRule(
        rule_id="N1", description="needs intent",
        category="X", patterns=[r"target"],
        verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
    )
    v = _make_validator([nei_rule], tmp_path=tmp_path)
    ctx = RuleContext(tool_name="t", arguments={"x": "target"}, capability="c")
    r = v.check(ctx)
    assert r.verdict == Verdict.NEEDS_EXPLICIT_INTENT
    assert not r.is_allowed


def test_log_only_is_allowed(tmp_path):
    log_rule = CommandPatternRule(
        rule_id="L1", description="log only",
        category="X", patterns=[r"target"],
        verdict_on_match=Verdict.LOG_ONLY,
    )
    v = _make_validator([log_rule], tmp_path=tmp_path)
    ctx = RuleContext(tool_name="t", arguments={"x": "target"}, capability="c")
    r = v.check(ctx)
    assert r.verdict == Verdict.LOG_ONLY
    assert r.is_allowed


def test_audit_written_only_for_non_allow(tmp_path):
    """ALLOW verdicts must NOT spam the audit log. Non-ALLOW does."""
    log_rule = CommandPatternRule(
        rule_id="L1", description="log only",
        category="X", patterns=[r"hello"],
        verdict_on_match=Verdict.LOG_ONLY,
    )
    audit_path = tmp_path / "audit.jsonl"
    audit = AuditLog(path=audit_path)
    policy = Policy(enabled=True)
    v = ToolCallValidator(policy=policy, rules=[log_rule], audit_log=audit)

    # Non-matching: ALLOW -> no audit row.
    v.check(RuleContext(tool_name="t", arguments={"x": "miss"}, capability="c"))
    assert not audit_path.is_file() or audit_path.stat().st_size == 0

    # Matching: LOG_ONLY -> audit row.
    v.check(RuleContext(tool_name="t", arguments={"x": "hello"}, capability="c"))
    assert audit_path.is_file()
    assert audit_path.stat().st_size > 0


def test_noop_validator_returns_allow():
    """Get-validator before set returns a permissive no-op."""
    from ultron.safety.validator import _NoOpValidator
    set_validator(None)
    from ultron.safety import get_validator
    v = get_validator()
    assert isinstance(v, _NoOpValidator)
    ctx = RuleContext(tool_name="t", arguments={}, capability="c")
    r = v.check(ctx)
    assert r.verdict == Verdict.ALLOW


def test_policy_is_rule_enabled_defaults_true():
    policy = Policy(rule_enabled={"K1": False})
    assert policy.is_rule_enabled("K1") is False
    assert policy.is_rule_enabled("K2") is True   # missing -> default True


def test_load_policy_default_protected_files_nonempty():
    policy = load_policy()
    assert len(policy.protected_files) > 0
    # config.yaml is in the default K list.
    assert any("config.yaml" in str(p).lower() for p in policy.protected_files)


def test_load_policy_respects_overrides():
    policy = load_policy(
        enabled=False,
        rule_overrides={"K1": False, "A3": True},
    )
    assert policy.enabled is False
    assert policy.is_rule_enabled("K1") is False
    assert policy.is_rule_enabled("A3") is True
