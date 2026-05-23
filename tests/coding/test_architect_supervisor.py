"""Tests for :mod:`ultron.coding.architect_supervisor`."""

from __future__ import annotations

from pathlib import Path

import pytest

from ultron.coding.architect_supervisor import (
    DEFAULT_ARCHITECT_SYSTEM_PROMPT,
    ArchitectPlan,
    ArchitectRequest,
    ArchitectSupervisor,
)


def test_constructor_rejects_empty_cascade():
    with pytest.raises(ValueError):
        ArchitectSupervisor([])


def test_generate_plan_returns_text():
    arch = ArchitectSupervisor([lambda _: "Add validation to login function in auth.py."])
    plan = arch.generate_plan(ArchitectRequest(user_text="add login validation"))
    assert plan.plan_text == "Add validation to login function in auth.py."
    assert plan.cascade_index == 0
    assert plan.generation_seconds >= 0.0
    assert plan.error == ""


def test_generate_plan_empty_user_text():
    arch = ArchitectSupervisor([lambda _: "anything"])
    plan = arch.generate_plan(ArchitectRequest(user_text=""))
    assert plan.plan_text is None
    assert plan.error == "empty user_text"


def test_generate_plan_falls_through_on_exception():
    def broken(_):
        raise RuntimeError("model down")

    arch = ArchitectSupervisor([broken, lambda _: "fallback plan"])
    plan = arch.generate_plan(ArchitectRequest(user_text="x"))
    assert plan.plan_text == "fallback plan"
    assert plan.cascade_index == 1


def test_generate_plan_falls_through_on_empty():
    arch = ArchitectSupervisor([lambda _: "", lambda _: "real plan"])
    plan = arch.generate_plan(ArchitectRequest(user_text="x"))
    assert plan.plan_text == "real plan"
    assert plan.cascade_index == 1


def test_generate_plan_all_fail():
    def broken(_):
        raise RuntimeError("down")

    arch = ArchitectSupervisor([broken, lambda _: ""])
    plan = arch.generate_plan(ArchitectRequest(user_text="x"))
    assert plan.plan_text is None
    assert "all LLMs failed" in plan.error
    assert plan.last_exception is not None


def test_generate_plan_respects_budget():
    """Per-LLM budgets — too-big prompt skips the primary."""
    huge_repo_map = "x" * 100_000

    def primary(_):
        return "should be skipped"

    def fallback(_):
        return "big-budget plan"

    arch = ArchitectSupervisor([primary, fallback])
    plan = arch.generate_plan(ArchitectRequest(
        user_text="do thing",
        repo_map_text=huge_repo_map,
        max_prompt_chars_per_llm=(1000, 1_000_000),
    ))
    assert plan.plan_text == "big-budget plan"
    assert plan.cascade_index == 1


def test_generate_plan_all_exceed_budget():
    arch = ArchitectSupervisor([lambda _: "skipped", lambda _: "skipped"])
    plan = arch.generate_plan(ArchitectRequest(
        user_text="x" * 200_000,
        max_prompt_chars_per_llm=(1000, 2000),
    ))
    assert plan.plan_text is None
    assert "too large" in plan.error


def test_prompt_includes_repo_map_and_digest():
    captured = []

    def capture(p):
        captured.append(p)
        return "plan"

    arch = ArchitectSupervisor([capture])
    arch.generate_plan(ArchitectRequest(
        user_text="add login",
        repo_map_text="--FAKE REPO MAP--",
        project_digest="--FAKE DIGEST--",
        project_path="/x/y/proj",
    ))
    assert captured
    prompt = captured[0]
    assert "--FAKE REPO MAP--" in prompt
    assert "--FAKE DIGEST--" in prompt
    assert "/x/y/proj" in prompt
    assert "add login" in prompt


def test_default_system_prompt_forbids_code_blocks():
    """The architect prompt explicitly forbids emitting code blocks."""
    assert "fenced code blocks" in DEFAULT_ARCHITECT_SYSTEM_PROMPT
    # And no AI attribution per the hygiene rules.
    assert "AI assistant" in DEFAULT_ARCHITECT_SYSTEM_PROMPT


def test_call_provider_contract_basic():
    """ArchitectSupervisor itself is the provider callable for
    ProjectSupervisor.architect_provider."""
    arch = ArchitectSupervisor([lambda _: "plan text"])
    out = arch("/proj/path", "add feature")
    assert out == "plan text"


def test_call_provider_with_kwargs():
    captured = []

    def capture(p):
        captured.append(p)
        return "plan"

    arch = ArchitectSupervisor([capture])
    out = arch(
        "/proj/path",
        "add feature",
        repo_map_text="MAP",
        project_digest="DIGEST",
    )
    assert out == "plan"
    assert "MAP" in captured[0]
    assert "DIGEST" in captured[0]


def test_call_provider_returns_none_on_total_failure():
    def broken(_):
        raise RuntimeError("down")

    arch = ArchitectSupervisor([broken])
    out = arch("/proj/path", "add feature")
    assert out is None


def test_architect_plan_is_frozen():
    plan = ArchitectPlan(plan_text="x")
    with pytest.raises(Exception):
        plan.plan_text = "y"  # type: ignore[misc]


def test_strip_quotes_option():
    arch = ArchitectSupervisor(
        [lambda _: '"plan in quotes"'],
        strip_outer_quotes=True,
    )
    plan = arch.generate_plan(ArchitectRequest(user_text="x"))
    assert plan.plan_text == "plan in quotes"


def test_no_strip_quotes_default():
    """Default behaviour preserves quotes (architect prose often
    contains internal quotes the user wants preserved)."""
    arch = ArchitectSupervisor([lambda _: '"plan"'])
    plan = arch.generate_plan(ArchitectRequest(user_text="x"))
    assert plan.plan_text == '"plan"'


# ---------------------------------------------------------------------------
# ProjectSupervisor integration
# ---------------------------------------------------------------------------


def test_supervisor_attaches_architect_plan(tmp_path: Path):
    from ultron.coding.project_supervisor import (
        ProjectSupervisor,
        SupervisorAction,
        SupervisorDecision,
    )
    from ultron.coding.projects import ProjectRegistry

    registry = ProjectRegistry(tmp_path / "projects.json")
    captured = {"calls": 0, "last_args": None, "last_kwargs": None}

    def provider(*args, **kwargs):
        captured["calls"] += 1
        captured["last_args"] = args
        captured["last_kwargs"] = kwargs
        return "architect: do this"

    supervisor = ProjectSupervisor(
        index=None,
        registry=registry,
        architect_provider=provider,
    )
    decision = SupervisorDecision(
        action=SupervisorAction.EDIT,
        target_project_path="/proj",
        user_text="add login",
        repo_map_text="--MAP--",
    )
    supervisor._attach_architect_plan(decision)
    assert captured["calls"] == 1
    assert decision.architect_plan_text == "architect: do this"
    # Provider received the repo_map_text as a kwarg.
    assert captured["last_kwargs"].get("repo_map_text") == "--MAP--"


def test_supervisor_skips_architect_for_clarify(tmp_path: Path):
    from ultron.coding.project_supervisor import (
        ProjectSupervisor,
        SupervisorAction,
        SupervisorDecision,
    )
    from ultron.coding.projects import ProjectRegistry

    registry = ProjectRegistry(tmp_path / "projects.json")
    captured = {"calls": 0}

    def provider(*a, **k):
        captured["calls"] += 1
        return "should not be invoked"

    supervisor = ProjectSupervisor(
        index=None,
        registry=registry,
        architect_provider=provider,
    )
    decision = SupervisorDecision(
        action=SupervisorAction.CLARIFY,
        target_project_path="/proj",
        user_text="which?",
    )
    supervisor._attach_architect_plan(decision)
    assert captured["calls"] == 0
    assert decision.architect_plan_text is None


def test_supervisor_skips_architect_when_no_path(tmp_path: Path):
    from ultron.coding.project_supervisor import (
        ProjectSupervisor,
        SupervisorAction,
        SupervisorDecision,
    )
    from ultron.coding.projects import ProjectRegistry

    registry = ProjectRegistry(tmp_path / "projects.json")
    captured = {"calls": 0}

    def provider(*a, **k):
        captured["calls"] += 1
        return "x"

    supervisor = ProjectSupervisor(
        index=None,
        registry=registry,
        architect_provider=provider,
    )
    decision = SupervisorDecision(
        action=SupervisorAction.NEW,
        target_project_path=None,
        user_text="scaffold",
    )
    supervisor._attach_architect_plan(decision)
    assert captured["calls"] == 0


def test_supervisor_swallows_architect_errors(tmp_path: Path):
    from ultron.coding.project_supervisor import (
        ProjectSupervisor,
        SupervisorAction,
        SupervisorDecision,
    )
    from ultron.coding.projects import ProjectRegistry

    registry = ProjectRegistry(tmp_path / "projects.json")

    def provider(*a, **k):
        raise RuntimeError("boom")

    supervisor = ProjectSupervisor(
        index=None,
        registry=registry,
        architect_provider=provider,
    )
    decision = SupervisorDecision(
        action=SupervisorAction.EDIT,
        target_project_path="/proj",
        user_text="do thing",
    )
    # Must not raise.
    supervisor._attach_architect_plan(decision)
    assert decision.architect_plan_text is None


def test_supervisor_provider_without_kwarg_support(tmp_path: Path):
    """Providers that don't accept repo_map_text kwarg still work."""
    from ultron.coding.project_supervisor import (
        ProjectSupervisor,
        SupervisorAction,
        SupervisorDecision,
    )
    from ultron.coding.projects import ProjectRegistry

    registry = ProjectRegistry(tmp_path / "projects.json")

    def provider_no_kwargs(project_path, user_text):
        return f"plan for {project_path}"

    supervisor = ProjectSupervisor(
        index=None,
        registry=registry,
        architect_provider=provider_no_kwargs,
    )
    decision = SupervisorDecision(
        action=SupervisorAction.EDIT,
        target_project_path="/proj",
        user_text="x",
    )
    supervisor._attach_architect_plan(decision)
    assert decision.architect_plan_text == "plan for /proj"


def test_supervisor_log_dict_excludes_architect_plan_text(tmp_path: Path):
    from ultron.coding.project_supervisor import (
        SupervisorAction,
        SupervisorDecision,
    )

    decision = SupervisorDecision(
        action=SupervisorAction.EDIT,
        target_project_path="/proj",
        architect_plan_text="long architect prose " * 500,
    )
    log = decision.to_log_dict()
    assert "architect_plan_text" not in log
    assert log["architect_plan_attached"] is True
