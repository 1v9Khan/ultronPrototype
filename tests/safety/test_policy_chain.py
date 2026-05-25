"""Tests for the trusted-tool-policy chain (T13)."""

from __future__ import annotations

from typing import Optional

import pytest

from ultron.safety.policy_chain import (
    ApprovalRequest,
    ChainResult,
    FunctionPolicy,
    PolicyDecision,
    PolicyOutcome,
    ToolCallContext,
    TrustedToolPolicyChain,
    get_policy_chain,
    reset_policy_chain_for_testing,
    set_policy_chain,
)


@pytest.fixture(autouse=True)
def _isolate_singleton() -> None:
    reset_policy_chain_for_testing()
    yield
    reset_policy_chain_for_testing()


@pytest.fixture
def chain() -> TrustedToolPolicyChain:
    return TrustedToolPolicyChain()


@pytest.fixture
def basic_ctx() -> ToolCallContext:
    return ToolCallContext(
        tool_name="memory.write",
        arguments={"text": "hello", "session_id": "abc"},
        capability="memory_writer",
        user_text="remember this",
        mode="standby",
        session_id="abc",
    )


# ----------------------------------------------------------------------
# Registration


def test_register_returns_registration(chain: TrustedToolPolicyChain) -> None:
    policy = FunctionPolicy(policy_id="p1", evaluate=lambda _: None)
    reg = chain.register(policy)
    assert reg.policy_id == "p1"
    assert chain.has_policies()


def test_register_replaces_existing_id(chain: TrustedToolPolicyChain) -> None:
    p1a = FunctionPolicy(policy_id="dup", evaluate=lambda _: None)
    p1b = FunctionPolicy(policy_id="dup", evaluate=lambda _: PolicyDecision(allow=True))
    chain.register(p1a)
    chain.register(p1b)
    assert chain.policy_ids() == ("dup",)


def test_unregister_removes_known_id(chain: TrustedToolPolicyChain) -> None:
    chain.register(FunctionPolicy(policy_id="p1", evaluate=lambda _: None))
    assert chain.unregister("p1") is True
    assert chain.policy_ids() == ()


def test_unregister_unknown_returns_false(chain: TrustedToolPolicyChain) -> None:
    assert chain.unregister("nope") is False


def test_clear_removes_all(chain: TrustedToolPolicyChain) -> None:
    chain.register(FunctionPolicy(policy_id="p1", evaluate=lambda _: None))
    chain.register(FunctionPolicy(policy_id="p2", evaluate=lambda _: None))
    chain.clear()
    assert chain.policy_ids() == ()


def test_policy_ids_preserves_registration_order(chain: TrustedToolPolicyChain) -> None:
    for pid in ("a", "b", "c"):
        chain.register(FunctionPolicy(policy_id=pid, evaluate=lambda _: None))
    assert chain.policy_ids() == ("a", "b", "c")


# ----------------------------------------------------------------------
# Run semantics


def test_run_empty_chain_returns_pass_through(
    chain: TrustedToolPolicyChain, basic_ctx: ToolCallContext
) -> None:
    result = chain.run(basic_ctx)
    assert result.outcome == PolicyOutcome.PASS_THROUGH
    assert result.final_params == basic_ctx.arguments


def test_run_pass_through_when_all_policies_return_none(
    chain: TrustedToolPolicyChain, basic_ctx: ToolCallContext
) -> None:
    chain.register(FunctionPolicy(policy_id="noop", evaluate=lambda _: None))
    result = chain.run(basic_ctx)
    assert result.outcome == PolicyOutcome.PASS_THROUGH


def test_run_block_terminates_chain(
    chain: TrustedToolPolicyChain, basic_ctx: ToolCallContext
) -> None:
    blocked = []

    def blocking(ctx: ToolCallContext) -> Optional[PolicyDecision]:
        blocked.append(ctx.tool_name)
        return PolicyDecision(allow=False, reason="forbidden")

    def downstream(ctx: ToolCallContext) -> Optional[PolicyDecision]:
        blocked.append("downstream-ran")
        return PolicyDecision(params={"changed": True})

    chain.register(FunctionPolicy(policy_id="blocker", evaluate=blocking))
    chain.register(FunctionPolicy(policy_id="downstream", evaluate=downstream))
    result = chain.run(basic_ctx)
    assert result.outcome == PolicyOutcome.BLOCK
    assert "downstream-ran" not in blocked
    assert result.block_decision is not None
    assert result.block_decision.reason == "forbidden"


def test_run_params_rewrite_stacks(
    chain: TrustedToolPolicyChain, basic_ctx: ToolCallContext
) -> None:
    def add_one(ctx: ToolCallContext) -> PolicyDecision:
        return PolicyDecision(params={**ctx.arguments, "added_one": True})

    def add_two(ctx: ToolCallContext) -> PolicyDecision:
        return PolicyDecision(params={**ctx.arguments, "added_two": True})

    chain.register(FunctionPolicy(policy_id="p1", evaluate=add_one))
    chain.register(FunctionPolicy(policy_id="p2", evaluate=add_two))
    result = chain.run(basic_ctx)
    assert result.outcome == PolicyOutcome.REWRITE
    assert result.final_params.get("added_one") is True
    assert result.final_params.get("added_two") is True
    assert result.rewrites_by == ("p1", "p2")


def test_run_approval_first_wins(
    chain: TrustedToolPolicyChain, basic_ctx: ToolCallContext
) -> None:
    first_approval = ApprovalRequest(
        policy_id="p1", kind="cost_limit", message="approve cost?"
    )
    second_approval = ApprovalRequest(
        policy_id="p2", kind="voice_confirm", message="confirm?"
    )

    chain.register(
        FunctionPolicy(
            policy_id="p1",
            evaluate=lambda _: PolicyDecision(require_approval=first_approval),
        )
    )
    chain.register(
        FunctionPolicy(
            policy_id="p2",
            evaluate=lambda _: PolicyDecision(require_approval=second_approval),
        )
    )
    result = chain.run(basic_ctx)
    assert result.outcome == PolicyOutcome.REQUIRE_APPROVAL
    assert result.approval is first_approval


def test_run_rewrite_then_approval_records_both(
    chain: TrustedToolPolicyChain, basic_ctx: ToolCallContext
) -> None:
    def rewrite(ctx: ToolCallContext) -> PolicyDecision:
        return PolicyDecision(params={**ctx.arguments, "rewrote": 1})

    approval = ApprovalRequest(policy_id="p2", kind="cost_limit", message="approve?")

    chain.register(FunctionPolicy(policy_id="p1", evaluate=rewrite))
    chain.register(
        FunctionPolicy(policy_id="p2", evaluate=lambda _: PolicyDecision(require_approval=approval))
    )
    result = chain.run(basic_ctx)
    assert result.outcome == PolicyOutcome.REQUIRE_APPROVAL
    assert result.final_params.get("rewrote") == 1
    assert result.approval is approval
    assert "p1" in result.rewrites_by


def test_run_downstream_sees_rewritten_params(
    chain: TrustedToolPolicyChain, basic_ctx: ToolCallContext
) -> None:
    seen: list = []

    def rewrite(ctx: ToolCallContext) -> PolicyDecision:
        return PolicyDecision(params={**ctx.arguments, "redacted": True})

    def observer(ctx: ToolCallContext) -> Optional[PolicyDecision]:
        seen.append(dict(ctx.arguments))
        return None

    chain.register(FunctionPolicy(policy_id="rewrite", evaluate=rewrite))
    chain.register(FunctionPolicy(policy_id="observer", evaluate=observer))
    chain.run(basic_ctx)
    assert seen and seen[0].get("redacted") is True


def test_run_policy_exception_swallowed_treated_as_pass_through(
    chain: TrustedToolPolicyChain, basic_ctx: ToolCallContext
) -> None:
    def boom(_: ToolCallContext) -> PolicyDecision:
        raise RuntimeError("policy bug")

    chain.register(FunctionPolicy(policy_id="bug", evaluate=boom))
    # Should NOT raise even with a broken policy.
    result = chain.run(basic_ctx)
    assert result.outcome == PolicyOutcome.PASS_THROUGH


def test_run_block_false_is_no_op(
    chain: TrustedToolPolicyChain, basic_ctx: ToolCallContext
) -> None:
    # allow=True / None passes through without blocking.
    chain.register(
        FunctionPolicy(policy_id="allowed", evaluate=lambda _: PolicyDecision(allow=True))
    )
    result = chain.run(basic_ctx)
    assert result.outcome == PolicyOutcome.PASS_THROUGH


def test_run_records_every_decision(
    chain: TrustedToolPolicyChain, basic_ctx: ToolCallContext
) -> None:
    chain.register(
        FunctionPolicy(policy_id="p1", evaluate=lambda _: PolicyDecision(allow=True))
    )
    chain.register(
        FunctionPolicy(policy_id="p2", evaluate=lambda _: PolicyDecision(params={"new": 1}))
    )
    result = chain.run(basic_ctx)
    decision_ids = [pid for pid, _ in result.decisions]
    assert decision_ids == ["p1", "p2"]


# ----------------------------------------------------------------------
# Singletons


def test_get_policy_chain_returns_singleton() -> None:
    a = get_policy_chain()
    b = get_policy_chain()
    assert a is b


def test_set_policy_chain_replaces_singleton() -> None:
    custom = TrustedToolPolicyChain()
    set_policy_chain(custom)
    assert get_policy_chain() is custom


def test_reset_policy_chain_for_testing_drops() -> None:
    set_policy_chain(TrustedToolPolicyChain())
    reset_policy_chain_for_testing()
    assert get_policy_chain() is not None  # new singleton constructed


# ----------------------------------------------------------------------
# ChainResult convenience properties


def test_chain_result_blocked_property() -> None:
    r = ChainResult(outcome=PolicyOutcome.BLOCK, final_params={})
    assert r.blocked is True
    assert r.requires_approval is False


def test_chain_result_requires_approval_property() -> None:
    r = ChainResult(outcome=PolicyOutcome.REQUIRE_APPROVAL, final_params={})
    assert r.requires_approval is True
    assert r.blocked is False


# ----------------------------------------------------------------------
# ToolCallContext session extension


def test_session_extension_returns_none_when_absent() -> None:
    ctx = ToolCallContext(tool_name="t", arguments={})
    assert ctx.get_session_extension("plugin_a") is None


def test_session_extension_returns_state() -> None:
    ctx = ToolCallContext(
        tool_name="t",
        arguments={},
        _session_state={"plugin_a": {"warned": 3}},
    )
    state = ctx.get_session_extension("plugin_a")
    assert state == {"warned": 3}
