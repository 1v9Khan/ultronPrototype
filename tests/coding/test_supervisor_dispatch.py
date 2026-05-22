"""Tests for ultron.coding.supervisor_dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pytest

from ultron.coding.project_index import ProjectIndexEntry
from ultron.coding.project_supervisor import (
    ProjectSupervisor,
    SupervisorAction,
    SupervisorCandidate,
    SupervisorDecision,
    SupervisorInputs,
)
from ultron.coding.projects import Project, ProjectRegistry
from ultron.coding.supervisor_dispatch import (
    DispatchActionKind,
    DispatchOutcome,
    SupervisorDispatchController,
    _indent_block,
    _slugify_for_directory,
    _speakable,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeSupervisor:
    """ProjectSupervisor stub returning a canned decision."""

    def __init__(self, decision: SupervisorDecision) -> None:
        self.decision = decision
        self.call_count = 0
        self.last_inputs: Optional[SupervisorInputs] = None

    def decide(self, inputs: SupervisorInputs) -> SupervisorDecision:
        self.call_count += 1
        self.last_inputs = inputs
        return self.decision


class FakeIndex:
    """ProjectIndex stub returning a canned entry."""

    def __init__(self, entry: Optional[ProjectIndexEntry] = None) -> None:
        self.entry = entry

    def get(self, project_id: str) -> Optional[ProjectIndexEntry]:
        return self.entry


# ---------------------------------------------------------------------------
# Slug / speakable helpers
# ---------------------------------------------------------------------------


def test_slugify_for_directory_basic() -> None:
    assert _slugify_for_directory("My Cool App!") == "my_cool_app"


def test_slugify_for_directory_truncates() -> None:
    long = "x" * 200
    assert len(_slugify_for_directory(long)) <= 50


def test_slugify_for_directory_empty() -> None:
    assert _slugify_for_directory("") == ""


def test_speakable_strips_backslashes() -> None:
    assert _speakable("C:\\Users\\alecf\\test") == "test"


def test_speakable_strips_forward_slashes() -> None:
    assert _speakable("/home/user/test") == "test"


def test_speakable_strips_quotes() -> None:
    assert _speakable("'project'") == "project"


def test_indent_block_indents_each_line() -> None:
    out = _indent_block("a\nb\nc", "  ")
    assert out == "  a\n  b\n  c"


# ---------------------------------------------------------------------------
# Dispatch -- per action kind
# ---------------------------------------------------------------------------


def _make_controller(supervisor, **kwargs):
    return SupervisorDispatchController(
        supervisor=supervisor,
        **kwargs,
    )


def test_dispatch_resume_returns_resume_forward() -> None:
    decision = SupervisorDecision(
        action=SupervisorAction.RESUME,
        target_project_name="myproj",
        resume_session_id="sess-1",
    )
    sup = FakeSupervisor(decision)
    controller = _make_controller(sup)
    outcome = controller.dispatch(SupervisorInputs(user_text="now add error handling"))

    assert outcome.kind == DispatchActionKind.RESUME_FORWARD
    assert outcome.resume_session_id == "sess-1"


def test_dispatch_edit_builds_task_request(tmp_path: Path) -> None:
    project_path = tmp_path / "flask_app"
    project_path.mkdir()
    decision = SupervisorDecision(
        action=SupervisorAction.EDIT,
        target_project_id="proj-1",
        target_project_name="flask_app",
        target_project_path=str(project_path),
    )
    sup = FakeSupervisor(decision)
    controller = _make_controller(sup)
    outcome = controller.dispatch(SupervisorInputs(user_text="add a route"))

    assert outcome.kind == DispatchActionKind.EDIT_DISPATCH
    assert outcome.task_request is not None
    assert outcome.task_request.cwd == project_path
    assert "add a route" in outcome.task_request.task_prompt


def test_dispatch_edit_missing_path_returns_fallback(tmp_path: Path) -> None:
    decision = SupervisorDecision(
        action=SupervisorAction.EDIT,
        target_project_name="missing",
        target_project_path="",
    )
    sup = FakeSupervisor(decision)
    controller = _make_controller(sup)
    outcome = controller.dispatch(SupervisorInputs(user_text="edit it"))

    # Missing target_project_path -> task_request is None -> EDIT_DISPATCH still
    # but with None task_request. The voice controller's _dispatch_supervisor_task
    # treats None task_request as a bail.
    assert outcome.kind == DispatchActionKind.EDIT_DISPATCH
    assert outcome.task_request is None


def test_dispatch_new_builds_task_request_with_sandbox(tmp_path: Path) -> None:
    decision = SupervisorDecision(
        action=SupervisorAction.NEW,
    )
    sup = FakeSupervisor(decision)
    controller = _make_controller(sup, sandbox_root=tmp_path)
    outcome = controller.dispatch(SupervisorInputs(
        user_text="build a brand-new pdf converter",
    ))

    assert outcome.kind == DispatchActionKind.NEW_DISPATCH
    assert outcome.task_request is not None
    assert outcome.task_request.cwd.parent == tmp_path.resolve() or outcome.task_request.cwd.parent == tmp_path
    assert "build_a_brand_new" in str(outcome.task_request.cwd)


def test_dispatch_new_without_sandbox_returns_no_request() -> None:
    decision = SupervisorDecision(
        action=SupervisorAction.NEW,
    )
    sup = FakeSupervisor(decision)
    controller = _make_controller(sup, sandbox_root=None)
    outcome = controller.dispatch(SupervisorInputs(user_text="build it"))
    assert outcome.task_request is None


def test_dispatch_clarify_returns_clarify_outcome() -> None:
    decision = SupervisorDecision(
        action=SupervisorAction.CLARIFY,
        clarification_question="Did you mean alpha or beta?",
    )
    sup = FakeSupervisor(decision)
    controller = _make_controller(sup)
    outcome = controller.dispatch(SupervisorInputs(user_text="edit the project"))

    assert outcome.kind == DispatchActionKind.CLARIFY
    assert outcome.clarification_question == "Did you mean alpha or beta?"


# ---------------------------------------------------------------------------
# Narration + barge-in
# ---------------------------------------------------------------------------


def test_narrate_disabled_does_not_call_barge_in(tmp_path: Path) -> None:
    decision = SupervisorDecision(
        action=SupervisorAction.EDIT,
        target_project_name="x",
        target_project_path=str(tmp_path),
    )
    sup = FakeSupervisor(decision)
    spoke = []
    controller = _make_controller(
        sup,
        barge_in_speak=lambda t: (spoke.append(t), False)[1],
        narrate_enabled=False,
    )
    outcome = controller.dispatch(SupervisorInputs(user_text="edit it"))
    assert spoke == []
    assert outcome.already_narrated is False
    # When narration is disabled, the outcome's voice_message carries
    # the narration so the caller can speak it after dispatch.
    assert outcome.voice_message != ""


def test_narrate_enabled_calls_barge_in(tmp_path: Path) -> None:
    decision = SupervisorDecision(
        action=SupervisorAction.EDIT,
        target_project_name="x",
        target_project_path=str(tmp_path),
    )
    sup = FakeSupervisor(decision)
    spoke = []
    controller = _make_controller(
        sup,
        barge_in_speak=lambda t: (spoke.append(t) or False),
        narrate_enabled=True,
    )
    outcome = controller.dispatch(SupervisorInputs(user_text="edit it"))
    assert len(spoke) == 1
    assert outcome.already_narrated is True
    assert outcome.voice_message == ""


def test_narrate_barge_in_returns_barged_in(tmp_path: Path) -> None:
    decision = SupervisorDecision(
        action=SupervisorAction.EDIT,
        target_project_name="x",
        target_project_path=str(tmp_path),
    )
    sup = FakeSupervisor(decision)
    controller = _make_controller(
        sup,
        barge_in_speak=lambda t: True,  # always barge in
        narrate_enabled=True,
    )
    outcome = controller.dispatch(SupervisorInputs(user_text="edit it"))
    assert outcome.kind == DispatchActionKind.BARGED_IN


def test_narration_text_for_resume() -> None:
    decision = SupervisorDecision(
        action=SupervisorAction.RESUME,
        target_project_name="flask_blog",
    )
    sup = FakeSupervisor(decision)
    controller = _make_controller(sup)
    outcome = controller.dispatch(SupervisorInputs(user_text="add error handling"))
    assert "flask" in outcome.voice_message.lower() or "barge" in outcome.voice_message.lower()


# ---------------------------------------------------------------------------
# Enriched context
# ---------------------------------------------------------------------------


def test_enriched_context_includes_digest(tmp_path: Path) -> None:
    project_path = tmp_path / "p1"
    project_path.mkdir()
    (project_path / "app.py").write_text("print('hi')\n")

    entry = ProjectIndexEntry(
        project_id="proj-1",
        project_name="p1",
        project_path=str(project_path),
        digest_markdown="## Goal\n- run a thing\n",
    )
    decision = SupervisorDecision(
        action=SupervisorAction.EDIT,
        target_project_id="proj-1",
        target_project_name="p1",
        target_project_path=str(project_path),
    )
    sup = FakeSupervisor(decision)
    controller = _make_controller(
        sup,
        index=FakeIndex(entry=entry),
        enriched_context_enabled=True,
    )
    outcome = controller.dispatch(SupervisorInputs(user_text="add a feature"))

    assert outcome.task_request is not None
    prompt = outcome.task_request.task_prompt
    assert "What we know about this project" in prompt
    assert "run a thing" in prompt
    assert "Project layout" in prompt
    assert "app.py" in prompt


def test_enriched_context_disabled_excludes_digest(tmp_path: Path) -> None:
    project_path = tmp_path / "p2"
    project_path.mkdir()
    decision = SupervisorDecision(
        action=SupervisorAction.EDIT,
        target_project_id="proj-2",
        target_project_name="p2",
        target_project_path=str(project_path),
    )
    sup = FakeSupervisor(decision)
    controller = _make_controller(
        sup,
        enriched_context_enabled=False,
    )
    outcome = controller.dispatch(SupervisorInputs(user_text="add a feature"))
    prompt = outcome.task_request.task_prompt
    assert "What we know about this project" not in prompt


def test_enriched_context_file_hints_listed(tmp_path: Path) -> None:
    project_path = tmp_path / "p3"
    project_path.mkdir()
    decision = SupervisorDecision(
        action=SupervisorAction.EDIT,
        target_project_id="proj-3",
        target_project_name="p3",
        target_project_path=str(project_path),
        file_hints=["src/app.py", "src/routes.py"],
    )
    sup = FakeSupervisor(decision)
    controller = _make_controller(
        sup, enriched_context_enabled=True,
    )
    outcome = controller.dispatch(SupervisorInputs(user_text="edit it"))
    prompt = outcome.task_request.task_prompt
    assert "Likely-relevant files" in prompt
    assert "src/app.py" in prompt


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


def test_dispatch_supervisor_raises_returns_fallback() -> None:
    class BadSupervisor:
        def decide(self, inputs):
            raise RuntimeError("supervisor exploded")

    controller = _make_controller(BadSupervisor())
    outcome = controller.dispatch(SupervisorInputs(user_text="anything"))
    assert outcome.kind == DispatchActionKind.FALLBACK


# ---------------------------------------------------------------------------
# build_digest convenience wrapper
# ---------------------------------------------------------------------------


def test_build_digest_uses_snapshot_for_language(tmp_path: Path) -> None:
    project_path = tmp_path / "py_proj"
    project_path.mkdir()
    (project_path / "main.py").write_text("def f(): pass\n")
    sup = FakeSupervisor(SupervisorDecision(action=SupervisorAction.NEW))
    controller = _make_controller(sup)

    digest = controller.build_digest(
        project_name="py_proj",
        project_path=project_path,
        task_summary="Built py_proj.",
        files_created=[project_path / "main.py"],
        files_modified=[],
        files_deleted=[],
        user_goal_hint="build a python script",
    )
    assert digest.project_name == "py_proj"
    assert "python" in digest.markdown.lower() or "main.py" in digest.markdown
