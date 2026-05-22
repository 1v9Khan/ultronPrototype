"""Tests for ultron.coding.project_supervisor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pytest

from ultron.coding.intent import CodingIntent, CodingIntentKind
from ultron.coding.project_digest import DigestRequest, ProjectDigest, parse_digest_sections, render_template
from ultron.coding.project_index import (
    ProjectIndex,
    ProjectIndexEntry,
    ProjectMatch,
)
from ultron.coding.project_supervisor import (
    ProjectSupervisor,
    SupervisorAction,
    SupervisorCandidate,
    SupervisorDecision,
    SupervisorInputs,
    _merge_candidates,
)
from ultron.coding.projects import (
    Project,
    ProjectRegistry,
    ProjectResolution,
    ProjectResolver,
    ResolutionKind,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeIndex:
    """Drop-in for ProjectIndex that returns canned search results.

    Avoids the embedder + qdrant startup cost in tests focused on the
    decision algorithm itself.
    """

    def __init__(self) -> None:
        self._matches: List[ProjectMatch] = []
        self._entries: dict = {}

    def add_match(
        self,
        project_id: str,
        project_name: str,
        project_path: str,
        score: float,
        *,
        digest_markdown: str = "",
    ) -> None:
        entry = ProjectIndexEntry(
            project_id=project_id,
            project_name=project_name,
            project_path=project_path,
            digest_markdown=digest_markdown,
        )
        self._entries[project_id] = entry
        self._matches.append(ProjectMatch(entry=entry, score=score))

    def clear(self) -> None:
        self._matches.clear()
        self._entries.clear()

    def search(self, query, top_k=5, min_score=0.0):
        return [m for m in self._matches if m.score >= min_score][:top_k]

    def get(self, project_id):
        return self._entries.get(project_id)


@pytest.fixture
def empty_registry(tmp_path: Path) -> ProjectRegistry:
    reg = ProjectRegistry(path=tmp_path / "registry.json")
    return reg


@pytest.fixture
def populated_registry(tmp_path: Path) -> ProjectRegistry:
    reg = ProjectRegistry(path=tmp_path / "registry.json")
    reg.add(Project(
        name="flask_blog",
        path=str(tmp_path / "flask_blog"),
        aliases=["the blog"],
        description="my flask blog",
    ))
    reg.add(Project(
        name="react_dashboard",
        path=str(tmp_path / "react_dashboard"),
        aliases=["the dashboard"],
        description="sales dashboard",
    ))
    return reg


def _make_supervisor(
    index=None,
    registry=None,
    resolver=None,
    log_path: Optional[Path] = None,
    resolve_threshold: float = 0.75,
    clarify_threshold: float = 0.55,
) -> ProjectSupervisor:
    if registry is None:
        # Use an in-memory-ish registry by pointing at a temp.
        registry = ProjectRegistry(path=Path("/tmp/_supervisor_test_reg.json"))
    return ProjectSupervisor(
        index=index,
        registry=registry,
        resolver=resolver,
        resolve_threshold=resolve_threshold,
        clarify_threshold=clarify_threshold,
        decisions_log_path=log_path,
    )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_invalid_thresholds(empty_registry) -> None:
    with pytest.raises(ValueError):
        ProjectSupervisor(
            index=None, registry=empty_registry,
            resolve_threshold=0.5, clarify_threshold=0.7,
        )
    with pytest.raises(ValueError):
        ProjectSupervisor(
            index=None, registry=empty_registry,
            resolve_threshold=1.5,
        )


# ---------------------------------------------------------------------------
# decide -- empty utterance
# ---------------------------------------------------------------------------


def test_decide_empty_text_returns_new(empty_registry) -> None:
    sup = _make_supervisor(registry=empty_registry)
    decision = sup.decide(SupervisorInputs(user_text=""))
    assert decision.action == SupervisorAction.NEW


# ---------------------------------------------------------------------------
# Resume path
# ---------------------------------------------------------------------------


def test_resume_when_active_task_and_adjustment(empty_registry) -> None:
    sup = _make_supervisor(registry=empty_registry)
    decision = sup.decide(SupervisorInputs(
        user_text="now add error handling",
        has_active_task=True,
        active_task_project_name="myproject",
        active_task_session_id="sess-1",
    ))
    assert decision.action == SupervisorAction.RESUME
    assert decision.target_project_name == "myproject"
    assert decision.resume_session_id == "sess-1"


def test_resume_when_intent_kind_is_mid_session_adjustment(empty_registry) -> None:
    sup = _make_supervisor(registry=empty_registry)
    intent = CodingIntent(
        kind=CodingIntentKind.MID_SESSION_ADJUSTMENT,
        task_text="add error handling",
        confidence=0.9,
    )
    decision = sup.decide(SupervisorInputs(
        user_text="add error handling",
        coding_intent=intent,
        has_active_task=True,
        active_task_session_id="sess-1",
    ))
    assert decision.action == SupervisorAction.RESUME


def test_no_resume_without_active_task(empty_registry) -> None:
    sup = _make_supervisor(registry=empty_registry)
    decision = sup.decide(SupervisorInputs(
        user_text="now add error handling",
        has_active_task=False,
    ))
    assert decision.action != SupervisorAction.RESUME


# ---------------------------------------------------------------------------
# EDIT path (semantic)
# ---------------------------------------------------------------------------


def test_edit_when_semantic_above_resolve_threshold(empty_registry) -> None:
    fake_index = FakeIndex()
    fake_index.add_match(
        project_id="proj-1",
        project_name="flask_app",
        project_path="/tmp/flask_app",
        score=0.85,
    )
    sup = _make_supervisor(index=fake_index, registry=empty_registry)
    decision = sup.decide(SupervisorInputs(user_text="edit the flask app"))
    assert decision.action == SupervisorAction.EDIT
    assert decision.target_project_id == "proj-1"
    assert decision.confidence == pytest.approx(0.85)


def test_edit_pulls_file_hints_from_digest(empty_registry) -> None:
    fake_index = FakeIndex()
    fake_index.add_match(
        project_id="proj-1",
        project_name="flask_app",
        project_path="/tmp/flask_app",
        score=0.85,
        digest_markdown=(
            "## Relevant Files\n"
            "- src/app.py: main entry\n"
            "- src/routes.py: HTTP routes\n"
        ),
    )
    sup = _make_supervisor(index=fake_index, registry=empty_registry)
    decision = sup.decide(SupervisorInputs(user_text="edit flask_app"))
    assert decision.action == SupervisorAction.EDIT
    assert "src/app.py" in decision.file_hints


# ---------------------------------------------------------------------------
# EDIT path (registry exact)
# ---------------------------------------------------------------------------


def test_edit_when_registry_exact_match(populated_registry) -> None:
    sup = _make_supervisor(registry=populated_registry)
    decision = sup.decide(SupervisorInputs(user_text="edit flask_blog now"))
    assert decision.action == SupervisorAction.EDIT
    assert decision.target_project_name == "flask_blog"


# ---------------------------------------------------------------------------
# CLARIFY path
# ---------------------------------------------------------------------------


def test_clarify_when_top_in_ambiguous_band(empty_registry) -> None:
    fake_index = FakeIndex()
    fake_index.add_match("p1", "alpha", "/tmp/alpha", score=0.65)
    fake_index.add_match("p2", "beta", "/tmp/beta", score=0.60)
    sup = _make_supervisor(index=fake_index, registry=empty_registry)
    decision = sup.decide(SupervisorInputs(user_text="edit the project"))
    assert decision.action == SupervisorAction.CLARIFY
    assert decision.clarification_question is not None
    # Question mentions both candidates.
    assert "alpha" in decision.clarification_question
    assert "beta" in decision.clarification_question


def test_clarify_single_candidate_phrasing(empty_registry) -> None:
    fake_index = FakeIndex()
    fake_index.add_match("p1", "lonely", "/tmp/lonely", score=0.6)
    sup = _make_supervisor(index=fake_index, registry=empty_registry)
    decision = sup.decide(SupervisorInputs(user_text="edit lonely"))
    assert decision.action == SupervisorAction.CLARIFY
    assert "lonely" in (decision.clarification_question or "").lower()


# ---------------------------------------------------------------------------
# NEW path
# ---------------------------------------------------------------------------


def test_new_when_no_matches_above_clarify_threshold(empty_registry) -> None:
    fake_index = FakeIndex()
    fake_index.add_match("p1", "unrelated", "/tmp/u", score=0.3)
    sup = _make_supervisor(index=fake_index, registry=empty_registry)
    decision = sup.decide(SupervisorInputs(
        user_text="build a brand-new pdf converter",
    ))
    assert decision.action == SupervisorAction.NEW


def test_new_when_no_index_no_matches(empty_registry) -> None:
    sup = _make_supervisor(registry=empty_registry)
    decision = sup.decide(SupervisorInputs(user_text="build something new"))
    assert decision.action == SupervisorAction.NEW


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def test_decision_logged_to_jsonl(tmp_path: Path, empty_registry) -> None:
    log_path = tmp_path / "decisions.jsonl"
    sup = _make_supervisor(
        registry=empty_registry, log_path=log_path,
    )
    sup.decide(SupervisorInputs(user_text="build a flask app"))
    sup.decide(SupervisorInputs(user_text="edit flask blog"))
    assert log_path.exists()
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
    row1 = json.loads(lines[0])
    assert "action" in row1
    assert "user_text" in row1


def test_audit_log_fails_open(tmp_path: Path, empty_registry) -> None:
    # Pointing the log at a nonexistent dir without create permission --
    # the supervisor should log a debug + return a decision regardless.
    bad_path = tmp_path / "nope" / "x" / "decisions.jsonl"
    sup = _make_supervisor(
        registry=empty_registry, log_path=bad_path,
    )
    # bad_path's parent gets mkdir'd in the constructor so this works
    # in normal flows; the failure case for actual write is when the
    # FS rejects mid-test, hard to simulate portably. We at least
    # verify the path that DOES exist after mkdir is OK.
    decision = sup.decide(SupervisorInputs(user_text="hello"))
    assert decision is not None


# ---------------------------------------------------------------------------
# _merge_candidates
# ---------------------------------------------------------------------------


def test_merge_candidates_dedupes_by_path() -> None:
    sem = [SupervisorCandidate("a", "alpha", "/tmp/a", 0.8, "semantic")]
    reg = [SupervisorCandidate("a", "alpha", "/tmp/a", 0.6, "registry_alias")]
    merged = _merge_candidates(sem, reg, cap=5)
    assert len(merged) == 1
    # Higher score wins.
    assert merged[0].score == 0.8


def test_merge_candidates_sorts_by_score() -> None:
    sem = [
        SupervisorCandidate("a", "alpha", "/tmp/a", 0.5, "semantic"),
        SupervisorCandidate("b", "beta", "/tmp/b", 0.9, "semantic"),
        SupervisorCandidate("c", "gamma", "/tmp/c", 0.7, "semantic"),
    ]
    merged = _merge_candidates(sem, [], cap=5)
    assert [c.project_name for c in merged] == ["beta", "gamma", "alpha"]


def test_merge_candidates_respects_cap() -> None:
    cands = [
        SupervisorCandidate(str(i), f"p{i}", f"/tmp/p{i}", 0.5, "semantic")
        for i in range(10)
    ]
    merged = _merge_candidates(cands, [], cap=3)
    assert len(merged) == 3


# ---------------------------------------------------------------------------
# Bus event integration
# ---------------------------------------------------------------------------


def test_decide_publishes_bus_event(empty_registry) -> None:
    from ultron.bus import SupervisorDecidedEvent, reset_bus_for_testing

    bus = reset_bus_for_testing()
    received = []
    bus.subscribe(SupervisorDecidedEvent, lambda p: received.append(p))

    sup = _make_supervisor(registry=empty_registry)
    sup.decide(SupervisorInputs(user_text="hello", turn_id=42))

    assert len(received) == 1
    assert received[0].properties["turn_id"] == 42
    assert "action" in received[0].properties
