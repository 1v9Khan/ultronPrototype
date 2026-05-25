"""Tests for T16 (category + user_message + metadata in safety decisions).

Extends the existing validator test suite with coverage of the new
optional fields on ``RuleResult`` and ``ValidatorVerdict``.
"""

from __future__ import annotations

from typing import Any

import pytest

from ultron.safety.audit import AuditLog
from ultron.safety.path_resolver import get_path_resolver
from ultron.safety.policy import Policy
from ultron.safety.validator import (
    RuleContext,
    RuleResult,
    ToolCallValidator,
    ValidatorVerdict,
    Verdict,
)


class _StubRule:
    """Test-only rule that emits a configurable RuleResult."""

    def __init__(self, rule_id: str, result: RuleResult) -> None:
        self.rule_id = rule_id
        self._result = result

    def evaluate(self, ctx, *, policy, resolver) -> RuleResult:  # noqa: ARG002
        return self._result


@pytest.fixture
def policy_enabled_all() -> Policy:
    return Policy(enabled=True)


@pytest.fixture
def audit_log_in_memory(tmp_path) -> AuditLog:
    return AuditLog(tmp_path / "audit.jsonl")


@pytest.fixture
def context() -> RuleContext:
    return RuleContext(
        tool_name="test.tool",
        arguments={"k": "v"},
        capability="test_capability",
        paths=(),
        user_text="trigger",
    )


# ----------------------------------------------------------------------
# RuleResult new optional fields


def test_rule_result_defaults_have_none_for_new_fields() -> None:
    r = RuleResult(rule_id="r1", verdict=Verdict.ALLOW, reason="ok")
    assert r.user_message is None
    assert r.category is None
    assert r.metadata is None


def test_rule_result_accepts_category_user_message_metadata() -> None:
    r = RuleResult(
        rule_id="r1",
        verdict=Verdict.BLOCK_HARD,
        reason="matched regex /sensitive/",
        user_message="I won't share that.",
        category="pii",
        metadata={"matched": True},
    )
    assert r.category == "pii"
    assert r.user_message == "I won't share that."
    assert r.metadata == {"matched": True}


# ----------------------------------------------------------------------
# Validator propagates new fields end-to-end


def test_validator_uses_user_message_when_present(
    policy_enabled_all, audit_log_in_memory, context
) -> None:
    rule = _StubRule(
        "rule_pii",
        RuleResult(
            rule_id="rule_pii",
            verdict=Verdict.BLOCK_HARD,
            reason="matched email regex",
            user_message="I held off so I don't leak that.",
            category="pii",
        ),
    )
    validator = ToolCallValidator(
        policy=policy_enabled_all,
        rules=[rule],
        audit_log=audit_log_in_memory,
        path_resolver=get_path_resolver(),
    )
    result = validator.check(context)
    assert result.verdict == Verdict.BLOCK_HARD
    assert result.user_message == "I held off so I don't leak that."
    assert "matched email regex" not in result.user_message
    assert result.category == "pii"


def test_validator_falls_back_to_synthesised_message_when_user_message_absent(
    policy_enabled_all, audit_log_in_memory, context
) -> None:
    rule = _StubRule(
        "rule_x",
        RuleResult(rule_id="rule_x", verdict=Verdict.BLOCK_HARD, reason="bad"),
    )
    validator = ToolCallValidator(
        policy=policy_enabled_all,
        rules=[rule],
        audit_log=audit_log_in_memory,
        path_resolver=get_path_resolver(),
    )
    result = validator.check(context)
    assert "I held off on that" in result.user_message


def test_validator_propagates_category_to_verdict(
    policy_enabled_all, audit_log_in_memory, context
) -> None:
    rule = _StubRule(
        "rule_cost",
        RuleResult(
            rule_id="rule_cost",
            verdict=Verdict.NEEDS_EXPLICIT_INTENT,
            reason="cost over budget",
            category="cost_limit",
            metadata={"estimate": 12000},
        ),
    )
    validator = ToolCallValidator(
        policy=policy_enabled_all,
        rules=[rule],
        audit_log=audit_log_in_memory,
        path_resolver=get_path_resolver(),
    )
    result = validator.check(context)
    assert result.category == "cost_limit"
    assert result.metadata == {"estimate": 12000}


def test_validator_audit_log_includes_category_when_present(
    policy_enabled_all, tmp_path, context
) -> None:
    log_path = tmp_path / "audit.jsonl"
    log = AuditLog(log_path)
    rule = _StubRule(
        "rule_viol",
        RuleResult(
            rule_id="rule_viol",
            verdict=Verdict.BLOCK_HARD,
            reason="violence",
            category="violence",
        ),
    )
    validator = ToolCallValidator(
        policy=policy_enabled_all,
        rules=[rule],
        audit_log=log,
        path_resolver=get_path_resolver(),
    )
    validator.check(context)
    body = log_path.read_text(encoding="utf-8")
    assert "\"category\": \"violence\"" in body


def test_validator_audit_log_omits_category_when_absent(
    policy_enabled_all, tmp_path, context
) -> None:
    log_path = tmp_path / "audit.jsonl"
    log = AuditLog(log_path)
    rule = _StubRule(
        "rule_plain",
        RuleResult(rule_id="rule_plain", verdict=Verdict.BLOCK_HARD, reason="plain"),
    )
    validator = ToolCallValidator(
        policy=policy_enabled_all,
        rules=[rule],
        audit_log=log,
        path_resolver=get_path_resolver(),
    )
    validator.check(context)
    body = log_path.read_text(encoding="utf-8")
    # Audit context dict should not carry a "category" key.
    assert "\"category\"" not in body or "rule_metadata" not in body


def test_validator_dominant_rule_carries_category_through_aggregation(
    policy_enabled_all, audit_log_in_memory, context
) -> None:
    rule_log = _StubRule(
        "rule_log",
        RuleResult(
            rule_id="rule_log",
            verdict=Verdict.LOG_ONLY,
            reason="log only",
            category="audit",
        ),
    )
    rule_block = _StubRule(
        "rule_block",
        RuleResult(
            rule_id="rule_block",
            verdict=Verdict.BLOCK_HARD,
            reason="violence",
            category="violence",
        ),
    )
    validator = ToolCallValidator(
        policy=policy_enabled_all,
        rules=[rule_log, rule_block],
        audit_log=audit_log_in_memory,
        path_resolver=get_path_resolver(),
    )
    result = validator.check(context)
    assert result.verdict == Verdict.BLOCK_HARD
    assert result.category == "violence"


def test_validator_verdict_default_category_none() -> None:
    v = ValidatorVerdict(verdict=Verdict.ALLOW, reason="ok")
    assert v.category is None
    assert v.metadata is None
