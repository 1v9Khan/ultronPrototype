"""Phase 5 — status narration tests.

Coverage:
  * Edge-case renderers for every non-EXECUTING session status.
  * Delta tracking across multiple status queries (the spec's
    "since you last asked" behavior).
  * In-voice rendering for the EXECUTING path with a stub LLM.
  * 20 representative narration scenarios for manual / regression
    review of the voice output.
  * Long-running mock session with multiple status queries (the
    Phase 5 integration scenario).
  * Backward compat: the runner without a narrator / store still
    produces a legacy bridge-state narration.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

import pytest

from ultron.coding.bridge import (
    CodingBridge, EventListener, FileChangeKind, TaskEvent, TaskHandle,
    TaskRequest, TaskResult, TaskState,
)
from ultron.coding.narration import StatusNarrator, NarrationDelta
from ultron.coding.runner import CodingTaskRunner
from ultron.coding.session import (
    ClarificationRequest, CompletionClaim, FileRecord, ProjectSession,
    SessionStatus, SessionStore, StageRecord,
)
# Aliased to dodge pytest's "test_*" / Test* collection heuristics that
# would otherwise warn about the dataclass having an __init__.
from ultron.coding.session import TestStatus as _TestStatusDC


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubLLM:
    """Records prompts and returns scripted text."""

    def __init__(self, text: str = "Voice-rendered status."):
        self.text = text
        self.prompts: List[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.text


class _MultiResponseLLM:
    """Returns a different response per call."""

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.prompts: List[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            return ""
        return self.responses.pop(0)


def _build_session(
    *,
    status: SessionStatus = SessionStatus.EXECUTING,
    project_root: Path = Path("weather_cli"),
    user_intent: str = "Build a Python CLI that prints today's forecast",
    refined_goal: str = "",
    stages: Optional[List[StageRecord]] = None,
    files_created: Optional[List[FileRecord]] = None,
    files_modified: Optional[List[FileRecord]] = None,
    test_status: Optional[_TestStatusDC] = None,
    pending_clarification: Optional[ClarificationRequest] = None,
    last_user_status_query: Optional[float] = None,
    completion_claim: Optional[CompletionClaim] = None,
    verification_failures: int = 0,
    current_stage: Optional[str] = None,
) -> ProjectSession:
    s = ProjectSession(
        session_id="test-session",
        project_root=project_root,
        user_intent=user_intent,
        refined_goal=refined_goal,
        status=status,
        current_stage=current_stage,
        stages_completed=stages or [],
        files_created=files_created or [],
        files_modified=files_modified or [],
        test_status=test_status or _TestStatusDC(),
        pending_clarification=pending_clarification,
        verification_failures=verification_failures,
        last_user_status_query=last_user_status_query,
        completion_claim=completion_claim,
    )
    return s


# ---------------------------------------------------------------------------
# Edge-case renderings (every non-EXECUTING status)
# ---------------------------------------------------------------------------


def test_no_session_returns_no_project_running():
    n = StatusNarrator()
    assert n.narrate(None) == "No project running."


def test_planning_includes_project_label():
    s = _build_session(
        status=SessionStatus.PLANNING,
        project_root=Path("weather_cli"),
    )
    out = StatusNarrator().narrate(s)
    assert "weather_cli" in out
    assert "started" in out.lower() or "starting" in out.lower()


def test_awaiting_clarification_mentions_question():
    s = _build_session(
        status=SessionStatus.AWAITING_CLARIFICATION,
        pending_clarification=ClarificationRequest(
            request_id="r1",
            question="Should I use SQLite or Postgres for the cache?",
        ),
    )
    out = StatusNarrator().narrate(s)
    assert "stopped" in out.lower() or "asking" in out.lower() or "question" in out.lower()


def test_awaiting_clarification_without_question_text():
    s = _build_session(
        status=SessionStatus.AWAITING_CLARIFICATION,
        pending_clarification=None,
    )
    out = StatusNarrator().narrate(s)
    assert "question" in out.lower() or "stopped" in out.lower()


def test_awaiting_user_extracts_topic():
    s = _build_session(
        status=SessionStatus.AWAITING_USER,
        pending_clarification=ClarificationRequest(
            request_id="r1",
            question="Should I use SQLite or Postgres for storage?",
        ),
    )
    out = StatusNarrator().narrate(s)
    assert "waiting" in out.lower()
    # topic extraction should pick up "storage" or the tail of the question
    assert any(w in out.lower() for w in ("storage", "sqlite", "postgres", "question"))


def test_awaiting_user_without_clarification_text():
    s = _build_session(
        status=SessionStatus.AWAITING_USER,
        pending_clarification=None,
    )
    out = StatusNarrator().narrate(s)
    assert "waiting" in out.lower()


def test_verifying_says_running_verification():
    s = _build_session(status=SessionStatus.VERIFYING)
    out = StatusNarrator().narrate(s)
    assert "verification" in out.lower() or "verifying" in out.lower()


def test_correcting_after_first_failure():
    s = _build_session(
        status=SessionStatus.CORRECTING,
        verification_failures=1,
    )
    out = StatusNarrator().narrate(s)
    assert "fixing" in out.lower() or "verification" in out.lower()


def test_correcting_after_repeated_failures_warns():
    s = _build_session(
        status=SessionStatus.CORRECTING,
        verification_failures=3,
    )
    out = StatusNarrator().narrate(s)
    assert "3" in out
    assert "escalate" in out.lower() or "fix" in out.lower()


def test_complete_with_completion_claim():
    s = _build_session(
        status=SessionStatus.COMPLETE,
        completion_claim=CompletionClaim(summary="Forecast CLI; tests pass."),
    )
    out = StatusNarrator().narrate(s)
    assert "Done" in out
    assert "weather_cli" in out


def test_complete_falls_back_to_files():
    s = _build_session(
        status=SessionStatus.COMPLETE,
        files_created=[FileRecord(path="main.py"), FileRecord(path="test_main.py")],
    )
    out = StatusNarrator().narrate(s)
    assert "Done" in out
    assert "2 files" in out or "files" in out


def test_failed_surfaces_to_user():
    s = _build_session(status=SessionStatus.FAILED)
    out = StatusNarrator().narrate(s)
    assert "stopped" in out.lower() or "failed" in out.lower()
    assert "weather_cli" in out


def test_terminated_says_cancelled():
    s = _build_session(status=SessionStatus.TERMINATED)
    out = StatusNarrator().narrate(s)
    assert "Cancelled" in out


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------


def test_compute_delta_first_query_returns_full_state():
    now = time.time()
    s = _build_session(
        stages=[StageRecord(stage="scaffolding", summary="set up", timestamp=now - 30)],
        files_created=[FileRecord(path="main.py", first_seen=now - 30)],
        last_user_status_query=None,
    )
    delta = StatusNarrator().compute_delta(s)
    assert delta.is_first_query is True
    assert len(delta.new_stages) == 1
    assert len(delta.new_files_created) == 1


def test_compute_delta_subsequent_query_filters_by_timestamp():
    last_query = time.time() - 60
    s = _build_session(
        stages=[
            StageRecord(stage="early", summary="x", timestamp=last_query - 10),
            StageRecord(stage="later", summary="y", timestamp=last_query + 5),
        ],
        files_created=[
            FileRecord(path="old.py", first_seen=last_query - 10),
            FileRecord(path="new.py", first_seen=last_query + 5),
        ],
        last_user_status_query=last_query,
    )
    delta = StatusNarrator().compute_delta(s)
    assert delta.is_first_query is False
    assert [st.stage for st in delta.new_stages] == ["later"]
    assert [f.path for f in delta.new_files_created] == ["new.py"]


def test_compute_delta_test_status_change_detected():
    last_query = time.time() - 60
    s = _build_session(
        test_status=_TestStatusDC(passing=10, failing=0, last_updated=last_query + 5),
        last_user_status_query=last_query,
    )
    delta = StatusNarrator().compute_delta(s)
    assert delta.test_status_changed is True


def test_compute_delta_test_status_unchanged_when_no_update():
    last_query = time.time() - 60
    s = _build_session(
        test_status=_TestStatusDC(passing=10, failing=0, last_updated=last_query - 5),
        last_user_status_query=last_query,
    )
    delta = StatusNarrator().compute_delta(s)
    assert delta.test_status_changed is False


def test_compute_delta_pending_clarification_arrived_since_last_query():
    last_query = time.time() - 60
    s = _build_session(
        pending_clarification=ClarificationRequest(
            request_id="r1",
            question="?",
            asked_at=last_query + 5,
        ),
        last_user_status_query=last_query,
    )
    delta = StatusNarrator().compute_delta(s)
    assert delta.pending_clarification_arrived is True


# ---------------------------------------------------------------------------
# EXECUTING-path rendering (LLM stub + fallback)
# ---------------------------------------------------------------------------


def test_executing_with_llm_calls_generator_and_returns_text():
    llm = _StubLLM(text="He's writing the auth tests; two files added since you last asked.")
    s = _build_session(
        current_stage="writing tests",
        stages=[StageRecord(stage="auth_module", summary="implemented login")],
        files_created=[FileRecord(path="auth.py")],
    )
    out = StatusNarrator(llm=llm).narrate(s)
    assert "auth tests" in out
    assert len(llm.prompts) == 1
    assert "Build a Python CLI" in llm.prompts[0]
    assert "writing tests" in llm.prompts[0]


def test_executing_first_query_fallback_describes_state():
    s = _build_session(
        current_stage="building scaffolding",
        files_created=[FileRecord(path="main.py")],
    )
    out = StatusNarrator(llm=None).narrate(s)
    assert "scaffolding" in out.lower()
    assert "1 file" in out


def test_executing_subsequent_query_fallback_lists_deltas():
    last = time.time() - 60
    s = _build_session(
        current_stage="implementing routes",
        stages=[
            StageRecord(stage="auth", summary="done", timestamp=last - 10),
            StageRecord(stage="routes", summary="started", timestamp=last + 5),
        ],
        files_created=[
            FileRecord(path="auth.py", first_seen=last - 10),
            FileRecord(path="routes.py", first_seen=last + 5),
        ],
        files_modified=[
            FileRecord(path="config.py", first_seen=last + 10),
        ],
        last_user_status_query=last,
    )
    out = StatusNarrator(llm=None).narrate(s)
    assert "Since you last asked" in out
    assert "1 new file" in out
    assert "1 modification" in out


def test_executing_no_changes_since_last_says_so():
    last = time.time() - 60
    s = _build_session(
        current_stage="thinking",
        last_user_status_query=last,
    )
    out = StatusNarrator(llm=None).narrate(s)
    assert "no new" in out.lower() or "still" in out.lower()


def test_executing_pending_clarification_leads_with_it():
    last = time.time() - 60
    s = _build_session(
        current_stage="planning",
        pending_clarification=ClarificationRequest(
            request_id="r1",
            question="What database should I use for the cache layer?",
            asked_at=last + 5,
        ),
        last_user_status_query=last,
    )
    # Even with an LLM wired in, the clarification gets the lede.
    llm = _StubLLM(text="generic fallback")
    out = StatusNarrator(llm=llm).narrate(s)
    assert "stopped" in out.lower()
    # LLM should NOT be called when we lead with the clarification
    assert llm.prompts == []


def test_executing_llm_failure_falls_back():
    class _BoomLLM:
        def generate(self, prompt: str) -> str:
            raise RuntimeError("LLM crashed")
    s = _build_session(current_stage="working")
    out = StatusNarrator(llm=_BoomLLM()).narrate(s)
    assert out  # got something
    # Doesn't propagate the exception
    assert "He's" in out or "working" in out.lower()


def test_executing_llm_empty_response_falls_back():
    s = _build_session(current_stage="working")
    out = StatusNarrator(llm=_StubLLM(text="")).narrate(s)
    assert out  # got fallback
    assert "He's" in out or "working" in out.lower()


def test_executing_llm_thinking_block_stripped():
    s = _build_session(current_stage="working")
    out = StatusNarrator(
        llm=_StubLLM(text="<think>internal</think>He's working on routes."),
    ).narrate(s)
    assert "He's working" in out
    assert "internal" not in out


def test_executing_llm_response_trimmed_to_first_paragraph():
    s = _build_session(current_stage="working")
    out = StatusNarrator(
        llm=_StubLLM(text="He's writing tests.\n\nMore context after."),
    ).narrate(s)
    assert out == "He's writing tests."


# ---------------------------------------------------------------------------
# Manual-review scenarios (the spec's "20 narration scenarios").
# Each scenario constructs a realistic session and asserts the narration
# is non-empty + contains expected keywords. The actual text is recorded
# in scenario_outputs for the spec's manual review.
# ---------------------------------------------------------------------------


@pytest.fixture
def voice_scenarios():
    """Yields a dict that the parametrized scenarios populate. After all
    scenarios run, an at-exit fixture prints the captured outputs for
    manual review."""
    captured: List[tuple[str, str]] = []
    yield captured
    # On test teardown, print every scenario's output so a maintainer
    # can grep the test log to manually review the rendered text.
    for label, out in captured:
        print(f"\n[narration:{label}] {out}")


@pytest.mark.parametrize("scenario_label,build_session,expected_substrings", [
    # 1. Brand-new project, no progress yet
    ("planning_new",
     lambda: _build_session(status=SessionStatus.PLANNING, project_root=Path("weather_cli")),
     ["weather_cli", "started"]),
    # 2. Mid-execution, fresh start, no LLM
    ("executing_fresh_no_llm",
     lambda: _build_session(current_stage="scaffolding"),
     ["scaffolding"]),
    # 3. Mid-execution, second query, new files since last query
    ("executing_second_query_new_files",
     lambda: _build_session(
         current_stage="implementing",
         files_created=[FileRecord(path="auth.py", first_seen=time.time() + 1)],
         last_user_status_query=time.time() - 30,
     ),
     ["new file", "implementing"]),
    # 4. Mid-execution, second query, no new work
    ("executing_no_new_work",
     lambda: _build_session(
         current_stage="thinking",
         last_user_status_query=time.time() - 30,
     ),
     ["new", "thinking"]),
    # 5. Tests passed
    ("executing_tests_passed",
     lambda: _build_session(
         current_stage="testing",
         test_status=_TestStatusDC(passing=8, failing=0, last_updated=time.time()),
     ),
     ["8 passing"]),
    # 6. Tests failing (mid-correction)
    ("correcting_after_test_failures",
     lambda: _build_session(
         status=SessionStatus.CORRECTING,
         verification_failures=1,
     ),
     ["fix"]),
    # 7. Awaiting clarification
    ("awaiting_clarification_with_topic",
     lambda: _build_session(
         status=SessionStatus.AWAITING_CLARIFICATION,
         pending_clarification=ClarificationRequest(
             request_id="r1",
             question="Use SQLite or Postgres for storage?",
         ),
     ),
     ["stopped"]),
    # 8. Awaiting user response
    ("awaiting_user",
     lambda: _build_session(
         status=SessionStatus.AWAITING_USER,
         pending_clarification=ClarificationRequest(
             request_id="r1",
             question="Do you want SQLite or Postgres for storage?",
         ),
     ),
     ["waiting"]),
    # 9. Verifying
    ("verifying",
     lambda: _build_session(status=SessionStatus.VERIFYING),
     ["verification"]),
    # 10. Complete with summary
    ("complete_with_summary",
     lambda: _build_session(
         status=SessionStatus.COMPLETE,
         completion_claim=CompletionClaim(
             summary="Forecast CLI ready; 8 tests passing.",
         ),
     ),
     ["Done", "weather_cli"]),
    # 11. Complete without summary
    ("complete_no_summary",
     lambda: _build_session(
         status=SessionStatus.COMPLETE,
         files_created=[FileRecord(path="main.py")],
     ),
     ["Done", "weather_cli"]),
    # 12. Failed (gave up)
    ("failed",
     lambda: _build_session(status=SessionStatus.FAILED),
     ["stopped"]),
    # 13. Terminated by user
    ("terminated",
     lambda: _build_session(status=SessionStatus.TERMINATED),
     ["Cancelled"]),
    # 14. Long working session, multiple stages
    ("executing_multiple_stages_done",
     lambda: _build_session(
         current_stage="writing tests",
         stages=[
             StageRecord(stage="scaffolding", summary="dirs"),
             StageRecord(stage="auth_module", summary="implemented"),
             StageRecord(stage="routes", summary="implemented"),
         ],
         files_created=[
             FileRecord(path="auth.py"), FileRecord(path="routes.py"),
         ],
     ),
     ["tests"]),
    # 15. Pending clarification arrived since last query
    ("clarification_arrived_since_last",
     lambda: _build_session(
         current_stage="planning",
         pending_clarification=ClarificationRequest(
             request_id="r1",
             question="Use SQLite or Postgres for storage?",
             asked_at=time.time(),
         ),
         last_user_status_query=time.time() - 30,
     ),
     ["stopped", "ask"]),
    # 16. Edit-mode session, current stage is "modifying"
    ("edit_mode_executing",
     lambda: _build_session(
         current_stage="updating routes",
         project_root=Path("calculator"),
         files_modified=[FileRecord(path="ops.py")],
     ),
     ["updating routes"]),
    # 17. Many files modified
    ("many_files_modified",
     lambda: _build_session(
         current_stage="refactoring",
         files_modified=[
             FileRecord(path=f"file_{i}.py") for i in range(10)
         ],
     ),
     ["refactoring"]),
    # 18. Tests run, mix of pass/fail
    ("mixed_test_results",
     lambda: _build_session(
         current_stage="fixing tests",
         test_status=_TestStatusDC(
             passing=6, failing=2, last_updated=time.time(),
         ),
     ),
     ["6 passing", "2 failing"]),
    # 19. Refined goal differs from original intent
    ("refined_goal",
     lambda: _build_session(
         current_stage="thinking",
         user_intent="Make me a thing for forecasts",
         refined_goal="Build a Python CLI with --city flag",
     ),
     ["thinking"]),
    # 20. Long stage label gets cleaned up
    ("messy_stage_label",
     lambda: _build_session(
         current_stage="Implementing_The_Auth_Module:",
     ),
     ["auth"]),
])
def test_voice_scenarios(scenario_label, build_session, expected_substrings, voice_scenarios):
    """Run each manual-review scenario and stash its rendered text."""
    s = build_session()
    out = StatusNarrator(llm=None).narrate(s)
    voice_scenarios.append((scenario_label, out))
    assert out, f"{scenario_label} produced empty narration"
    for needle in expected_substrings:
        assert needle.lower() in out.lower(), (
            f"{scenario_label}: expected {needle!r} in {out!r}"
        )


# ---------------------------------------------------------------------------
# Long-running mock session with multiple status queries.
# ---------------------------------------------------------------------------


def test_long_running_session_three_status_queries():
    """Simulate the spec's "long-running session with multiple status
    queries" integration. We populate state in stages, advance the
    last_user_status_query in between, and verify each call narrates the
    delta correctly."""
    store = SessionStore()
    s = store.create(
        project_root=Path("data/sandbox/weather_cli"),
        user_intent="Build a Python CLI for weather forecasts",
    )
    store.transition(s.session_id, SessionStatus.EXECUTING)

    # Initial state: scaffolding done.
    store.record_stage(
        s.session_id, stage="scaffolding", summary="set up project layout",
        files_touched=["pyproject.toml", "main.py"],
    )

    narrator = StatusNarrator(llm=None)

    # First query.
    out_1 = narrator.narrate(store.get(s.session_id))
    assert "scaffolding" in out_1.lower()
    assert "2 files" in out_1
    store.touch_status_query(s.session_id)

    time.sleep(0.01)  # ensure subsequent timestamps strictly exceed t1

    # More work happens.
    store.record_stage(
        s.session_id, stage="auth_module", summary="login implemented",
        files_touched=["auth.py", "main.py"],  # main.py is now modified
    )
    store.record_test_results(
        s.session_id, passing=4, failing=0, skipped=0, details="auth tests pass",
    )

    out_2 = narrator.narrate(store.get(s.session_id))
    assert "Since you last asked" in out_2
    assert "auth_module" in out_2 or "auth" in out_2 or "1 new file" in out_2
    store.touch_status_query(s.session_id)

    time.sleep(0.01)

    # No new work yet.
    out_3 = narrator.narrate(store.get(s.session_id))
    # No new stages, files, or tests since last query.
    assert "no new" in out_3.lower() or "still" in out_3.lower()


def test_long_running_session_clarification_in_middle():
    """A clarification arrives between queries -- next narration leads with it."""
    store = SessionStore()
    s = store.create(
        project_root=Path("data/sandbox/weather_cli"),
        user_intent="Build a CLI",
    )
    store.transition(s.session_id, SessionStatus.EXECUTING)
    store.record_stage(
        s.session_id, stage="scaffolding", summary="dirs", files_touched=["main.py"],
    )
    narrator = StatusNarrator(llm=None)

    out_1 = narrator.narrate(store.get(s.session_id))
    assert out_1
    store.touch_status_query(s.session_id)

    time.sleep(0.01)
    # Claude requests a clarification.
    store.set_pending_clarification(s.session_id, ClarificationRequest(
        request_id="r1",
        question="Should we use SQLite or Postgres for the local cache?",
        asked_at=time.time(),
    ))

    out_2 = narrator.narrate(store.get(s.session_id))
    # Lead with the clarification (per spec).
    assert any(w in out_2.lower() for w in ("stopped", "ask", "needs"))


def test_status_query_during_completion_after_announcement():
    """The user 'forgot' they got the completion announcement and asks
    again. Narration should still produce a clean Done summary."""
    s = _build_session(
        status=SessionStatus.COMPLETE,
        completion_claim=CompletionClaim(summary="All tests pass."),
    )
    out = StatusNarrator().narrate(s)
    assert "Done" in out


# ---------------------------------------------------------------------------
# Runner integration -- session-aware narration through the runner API
# ---------------------------------------------------------------------------


class _FakeBridge(CodingBridge):
    def __init__(self):
        self.last_request: Optional[TaskRequest] = None

    def submit(self, request: TaskRequest):
        self.last_request = request

        class _H(TaskHandle):
            def __init__(self, req):
                self._state = TaskState(
                    label=req.label or "test",
                    task_prompt=req.task_prompt,
                    cwd=req.cwd,
                    started_at=time.time(),
                )
                self._task_id = "fake"

            def task_id(self): return self._task_id
            def state(self): return self._state
            def add_listener(self, listener): pass
            def cancel(self): pass
            def wait(self, timeout=None): return None
            def is_running(self): return True

        return _H(request)

    def name(self): return "fake"


def test_runner_progress_narration_with_session_uses_narrator():
    store = SessionStore()
    session = store.create(
        project_root=Path("data/sandbox/weather"),
        user_intent="Build a CLI",
    )
    store.transition(session.session_id, SessionStatus.EXECUTING)
    store.record_stage(
        session.session_id, stage="scaffold", summary="dirs",
        files_touched=["main.py"],
    )

    narrator = StatusNarrator(llm=None)
    runner = CodingTaskRunner(
        bridge=_FakeBridge(), log_path=None,
        narrator=narrator, store=store,
    )
    out = runner.progress_narration(session=store.get(session.session_id))
    assert "scaffold" in out.lower()
    # touch_status_query stamped via store.
    refreshed = store.get(session.session_id)
    assert refreshed.last_user_status_query is not None


def test_runner_progress_narration_without_session_uses_legacy_path():
    """Backward-compat: existing tests call progress_narration() with no
    session and expect the bridge-state narration."""
    runner = CodingTaskRunner(bridge=_FakeBridge(), log_path=None)
    # No active task -> the legacy path returns "no coding task active".
    out = runner.progress_narration()
    assert "No coding task" in out


def test_runner_session_aware_without_store_still_stamps_session():
    """When no store is wired, the runner mutates session.last_user_status_query
    directly so the next call computes its delta correctly."""
    s = _build_session(current_stage="working")
    runner = CodingTaskRunner(bridge=_FakeBridge(), log_path=None)
    assert s.last_user_status_query is None
    runner.progress_narration(session=s)
    assert s.last_user_status_query is not None


# ---------------------------------------------------------------------------
# Voice controller integration -- _handle_progress routes through runner
# with session lookup
# ---------------------------------------------------------------------------


class _StubCoordinator:
    def __init__(self, store: SessionStore):
        self.store = store


def test_voice_controller_progress_routes_through_session_when_coordinator_wired(tmp_path):
    """The voice controller should pull the active session from the
    coordinator's store and pass it to the runner, getting rich
    narration."""
    from ultron.coding.projects import ProjectRegistry, ProjectResolver
    from ultron.coding.voice import CodingVoiceController

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    registry = ProjectRegistry(path=tmp_path / "projects.json")
    resolver = ProjectResolver(registry, embedder=None)

    store = SessionStore()
    session = store.create(
        project_root=tmp_path / "weather",
        user_intent="Build a CLI",
    )
    store.transition(session.session_id, SessionStatus.EXECUTING)
    store.record_stage(
        session.session_id, stage="building",
        summary="implementing", files_touched=["main.py"],
    )

    runner = CodingTaskRunner(
        bridge=_FakeBridge(), log_path=None,
        narrator=StatusNarrator(llm=None), store=store,
    )
    coordinator = _StubCoordinator(store)
    controller = CodingVoiceController(
        runner=runner, registry=registry, resolver=resolver,
        sandbox_root=sandbox, coordinator=coordinator,
    )

    response = controller.handle_utterance("How's it going?")
    assert response is not None
    # Response was narrated from the session, not from the bridge state.
    assert "building" in response.text.lower() or "1 file" in response.text


def test_voice_controller_progress_falls_back_to_legacy_without_coordinator(tmp_path):
    """When no coordinator is wired but a bridge task is running, the
    voice controller falls back to the runner's legacy bridge-state
    narration path."""
    from ultron.coding.projects import ProjectRegistry, ProjectResolver
    from ultron.coding.voice import CodingVoiceController

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    registry = ProjectRegistry(path=tmp_path / "projects.json")
    resolver = ProjectResolver(registry, embedder=None)
    runner = CodingTaskRunner(bridge=_FakeBridge(), log_path=None)
    controller = CodingVoiceController(
        runner=runner, registry=registry, resolver=resolver,
        sandbox_root=sandbox, coordinator=None,  # legacy path
    )
    # Submit a coding task so the bridge has a running handle.
    started = controller.handle_utterance(
        "Create a python script that prints hello."
    )
    assert started is not None and started.handled
    # Now query progress -- legacy bridge-state path should fire.
    response = controller.handle_utterance("How's it going?")
    assert response is not None
    # Legacy path narrates from the bridge's TaskState; expected to
    # mention "currently" or "task" (matches the existing voice tests).
    assert "currently" in response.text.lower() or "task" in response.text.lower()


# ---------------------------------------------------------------------------
# Topic extraction (sanity)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("question,expected_substring", [
    ("Should I use SQLite or Postgres for storage?", "storage"),
    ("Which model should I use for the embedder?", "embedder"),
    ("Use 4-space or 2-space indentation?", "indentation"),
    ("", ""),  # no topic
])
def test_extract_topic(question, expected_substring):
    out = StatusNarrator._extract_topic(question)
    if expected_substring:
        assert expected_substring.lower() in out.lower()
    else:
        assert out == ""
