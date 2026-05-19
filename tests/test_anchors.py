"""Tests for :mod:`ultron.coding.anchors` (pure primitives).

The runner-integration tests live in :mod:`tests.test_coding_runner_anchors`.
This file exercises the decomposer + budget + plan dataclasses
without any bridge / runner involvement.
"""

from __future__ import annotations

import pytest

from ultron.coding.anchors import (
    AnchorBudget,
    AnchorPlan,
    GoalAnchor,
    decompose_into_anchors,
    narration_for_anchor,
    narration_for_completion,
)


# ---------------------------------------------------------------------------
# decompose_into_anchors
# ---------------------------------------------------------------------------


def test_empty_prompt_falls_back_to_single_anchor() -> None:
    plan = decompose_into_anchors("", total_budget_tokens=1000)
    assert len(plan) == 1
    assert plan.anchors[0].anchor.description == "complete the task"
    assert plan.anchors[0].anchor.budget_tokens == 1000


def test_whitespace_prompt_falls_back_to_single_anchor() -> None:
    plan = decompose_into_anchors("   \n", total_budget_tokens=500)
    assert len(plan) == 1


def test_prompt_with_three_clauses_decomposes_into_three_anchors() -> None:
    plan = decompose_into_anchors(
        "Build the parser, then test it, finally deploy.",
        total_budget_tokens=900,
    )
    assert len(plan) == 3
    descriptions = [a.anchor.description for a in plan.anchors]
    assert any("build" in d.lower() for d in descriptions)
    assert any("test" in d.lower() for d in descriptions)
    assert any("deploy" in d.lower() for d in descriptions)


def test_decompose_splits_on_period_boundary() -> None:
    plan = decompose_into_anchors(
        "Build the worker. Test the worker. Deploy the worker.",
        total_budget_tokens=600,
    )
    assert len(plan) == 3


def test_decompose_caps_at_max_anchors() -> None:
    plan = decompose_into_anchors(
        "Build A, test A, build B, test B, build C, test C, deploy.",
        total_budget_tokens=1000,
        max_anchors=3,
    )
    assert len(plan) <= 3


def test_decompose_pads_to_min_anchors() -> None:
    plan = decompose_into_anchors(
        "do something vague",
        total_budget_tokens=500,
        min_anchors=2,
    )
    assert len(plan) >= 2


def test_decompose_distributes_budget_with_remainder_to_first() -> None:
    plan = decompose_into_anchors(
        "Build X, then test Y, then deploy Z.",
        total_budget_tokens=1000,
    )
    budgets = [a.anchor.budget_tokens for a in plan.anchors]
    assert sum(budgets) == 1000
    # First anchor gets the +1 remainder.
    assert budgets[0] >= budgets[1]


def test_decompose_skips_non_actionable_clauses() -> None:
    plan = decompose_into_anchors(
        "Hmm, you know, build something, then test it.",
        total_budget_tokens=200,
    )
    # The "Hmm, you know" chunks don't start with anchor verbs and get
    # dropped; only "build" + "test" survive.
    descriptions = [a.anchor.description.lower() for a in plan.anchors]
    assert any("build" in d for d in descriptions)
    assert any("test" in d for d in descriptions)


def test_decompose_dedupes_repeated_clauses() -> None:
    plan = decompose_into_anchors(
        "Build the worker. Build the worker. Test the worker.",
        total_budget_tokens=300,
    )
    descriptions = [a.anchor.description.lower() for a in plan.anchors]
    # The two "Build the worker" clauses dedupe.
    assert sum("build" in d for d in descriptions) == 1


# ---------------------------------------------------------------------------
# AnchorBudget
# ---------------------------------------------------------------------------


def _budget(b: int = 100) -> AnchorBudget:
    return AnchorBudget(
        anchor=GoalAnchor(
            name="x", description="x", order=0, budget_tokens=b,
        )
    )


def test_budget_update_clamps_negatives() -> None:
    b = _budget(100)
    b.update(-5)
    assert b.tokens_spent == 0
    b.update(10)
    assert b.tokens_spent == 10


def test_budget_utilisation_handles_zero_budget() -> None:
    zero = AnchorBudget(
        anchor=GoalAnchor("x", "x", 0, 0),
    )
    assert zero.utilisation == 0.0
    zero.update(1)
    assert zero.utilisation == 1.0


def test_budget_is_exhausted_at_or_above_budget() -> None:
    b = _budget(100)
    b.update(99)
    assert not b.is_exhausted
    b.update(1)
    assert b.is_exhausted


def test_budget_should_warn_latches_once() -> None:
    b = _budget(100)
    b.update(50)
    assert not b.should_warn(threshold=0.8)
    b.update(40)  # at 90%
    assert b.should_warn(threshold=0.8)
    # Second call latches.
    assert not b.should_warn(threshold=0.8)


def test_budget_reset_warning_re_arms() -> None:
    b = _budget(100)
    b.update(90)
    assert b.should_warn(threshold=0.8)
    b.reset_warning()
    assert b.should_warn(threshold=0.8)


def test_budget_mark_completed_flips_flag() -> None:
    b = _budget(100)
    assert not b.completed
    b.mark_completed()
    assert b.completed


# ---------------------------------------------------------------------------
# AnchorPlan
# ---------------------------------------------------------------------------


def _plan(*budgets: int) -> AnchorPlan:
    return AnchorPlan(
        anchors=[
            AnchorBudget(
                anchor=GoalAnchor(
                    name=f"a{i}",
                    description=f"do step {i}",
                    order=i,
                    budget_tokens=b,
                )
            )
            for i, b in enumerate(budgets)
        ]
    )


def test_plan_active_starts_at_zero() -> None:
    plan = _plan(50, 50, 50)
    assert plan.active is not None
    assert plan.active.anchor.order == 0


def test_plan_advance_marks_completed_and_moves_forward() -> None:
    plan = _plan(50, 50)
    old_active = plan.active
    new_active = plan.advance()
    assert old_active.completed
    assert new_active is not None
    assert new_active.anchor.order == 1


def test_plan_advance_past_end_returns_none() -> None:
    plan = _plan(50, 50)
    plan.advance()
    plan.advance()
    assert plan.active is None
    assert plan.all_completed is True


def test_plan_remaining_tokens_excludes_completed() -> None:
    plan = _plan(50, 50, 50)
    plan.anchors[0].update(50)
    plan.advance()
    assert plan.remaining_tokens() == 50 + 50


def test_plan_iter_yields_budgets_in_order() -> None:
    plan = _plan(10, 20, 30)
    orders = [b.anchor.order for b in plan]
    assert orders == [0, 1, 2]


def test_plan_as_dict_carries_progress() -> None:
    plan = _plan(100, 100)
    plan.anchors[0].update(60)
    payload = plan.as_dict()
    assert payload["active_index"] == 0
    assert payload["anchors"][0]["tokens_spent"] == 60
    assert payload["anchors"][0]["utilisation"] == 0.6
    assert payload["anchors"][1]["tokens_spent"] == 0


def test_empty_plan_has_no_active() -> None:
    plan = AnchorPlan(anchors=[])
    assert plan.active is None
    assert plan.all_completed is False
    assert plan.advance() is None


# ---------------------------------------------------------------------------
# Narration helpers
# ---------------------------------------------------------------------------


def test_narration_for_anchor_includes_order_and_description() -> None:
    anchor = GoalAnchor("scaffold", "scaffold the project", 0, 1000)
    msg = narration_for_anchor(anchor, verb="Starting")
    assert "Starting" in msg
    assert "anchor 1" in msg
    assert "scaffold the project" in msg


def test_narration_handles_empty_description() -> None:
    anchor = GoalAnchor("empty", "", 1, 1000)
    msg = narration_for_anchor(anchor, verb="Moving to")
    assert "next anchor" in msg


def test_narration_for_completion_counts_anchors() -> None:
    plan = _plan(50, 50, 50)
    msg = narration_for_completion(plan)
    assert "3" in msg
    assert "completed" in msg.lower()


def test_narration_for_completion_empty_plan() -> None:
    msg = narration_for_completion(AnchorPlan(anchors=[]))
    assert "nothing" in msg.lower()


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_goal_anchor_is_frozen() -> None:
    anchor = GoalAnchor("n", "d", 0, 100)
    with pytest.raises(Exception):  # FrozenInstanceError
        anchor.name = "other"  # type: ignore[misc]
