"""Phase 6 — orchestration integration scenarios (mocked).

Each scenario wires up the full supervisor stack (MCP server +
coordinator + verifier + runner + voice controller + narrator) against
a :class:`ScriptedClaudeBridge` that simulates Claude in-process. The
scripts are deterministic; no Claude tokens are burned.

The 10 scenarios mirror the spec's Phase 6 list:

  1. New project, smooth completion
  2. Existing-project edit (only the targeted project's files change)
  3. Clarification answered from intent (no escalation to user)
  4. Clarification escalated to user (user voice response resolves it)
  5. Verification failure + correction loop
  6. Mid-project adjustment via voice
  7. Status query during execution (narration is delta-aware)
  8. Cancellation tears down cleanly
  9. Model escalation after Haiku verification failures
  10. Project-root isolation rejection

Real-Claude variants of every scenario live in
:mod:`tests.coding.test_orchestration_real` and are gated on
``PYTEST_RUN_GPU_TESTS=1``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from ultron.coding import (
    CodingTaskRunner,
    CodingVoiceController,
    Project,
    ProjectRegistry,
    ProjectResolver,
    StatusNarrator,
    UltronMCPServer,
)
from ultron.coding.bridge import TaskRequest
from ultron.coding.coordinator import ConversationCoordinator
from ultron.coding.session import SessionStatus
from ultron.coding.verification import Verifier

from tests.coding.mock_bridge import ClaudeScript, ScriptedClaudeBridge


# ---------------------------------------------------------------------------
# Stub LLM
# ---------------------------------------------------------------------------


class _StubLLM:
    """Deterministic LLM stub. Tests can either set ``response_text`` once
    or use ``responses`` for a script of replies (popped FIFO)."""

    def __init__(self, response_text: str = "Use your default approach."):
        self.response_text = response_text
        self.responses: List[str] = []
        self.prompts: List[str] = []

    def push(self, *responses: str) -> "_StubLLM":
        self.responses.extend(responses)
        return self

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return self.response_text


# ---------------------------------------------------------------------------
# Stack assembly
# ---------------------------------------------------------------------------


@dataclass
class OrchStack:
    server: UltronMCPServer
    coordinator: ConversationCoordinator
    runner: CodingTaskRunner
    voice: CodingVoiceController
    narrator: StatusNarrator
    llm: _StubLLM
    registry: ProjectRegistry
    sandbox: Path

    def create_session(self, *, project_root: Path, intent: str, mode: str = "new"):
        s = self.server.create_session(
            project_root=project_root, initial_prompt=intent, mode=mode,
        )
        self.server.store.transition(s.session_id, SessionStatus.EXECUTING)
        return s


def _build_stack(
    tmp_path: Path,
    *,
    llm: Optional[_StubLLM] = None,
    bridge: Optional[Any] = None,
) -> OrchStack:
    """Construct the full Phase 1-5 stack with shared state. Bridge is
    optional -- the runner doesn't actually submit a task in some
    scenarios, in which case we use a placeholder fake bridge."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir(exist_ok=True)
    server = UltronMCPServer(host="127.0.0.1", port=0)
    llm = llm or _StubLLM()
    verifier = Verifier(store=server.store)
    coordinator = ConversationCoordinator(
        store=server.store, llm=llm, verifier=verifier,
    )
    server.set_clarification_responder(coordinator.decide_clarification)
    server.set_declare_complete_handler(coordinator.handle_declare_complete)

    # Narrator gets no LLM in tests -- the deterministic fallback produces
    # predictable, in-voice output without depending on a stub LLM that
    # might return clarification-shaped text in narration prompts.
    narrator = StatusNarrator(llm=None)

    if bridge is None:
        # A placeholder bridge that submit() refuses; scenarios that don't
        # actually start a task use this. Scenarios that do, supply their
        # own ScriptedClaudeBridge.
        from tests.coding.mock_bridge import ClaudeScript as _CS, ScriptedClaudeBridge as _SCB
        bridge = _SCB(server, _CS(), session_id="__unset__")

    runner = CodingTaskRunner(
        bridge=bridge, log_path=tmp_path / "audit.jsonl",
        narrator=narrator, store=server.store,
    )
    registry = ProjectRegistry(path=tmp_path / "projects.json")
    resolver = ProjectResolver(registry, embedder=None)
    voice = CodingVoiceController(
        runner=runner, registry=registry, resolver=resolver,
        sandbox_root=sandbox, coordinator=coordinator,
    )
    return OrchStack(
        server=server, coordinator=coordinator, runner=runner,
        voice=voice, narrator=narrator, llm=llm,
        registry=registry, sandbox=sandbox,
    )


# ---------------------------------------------------------------------------
# Scenario 1 — new project, smooth completion
# ---------------------------------------------------------------------------


def test_scenario_1_new_project_smooth_completion(tmp_path: Path):
    """Run a scripted task end-to-end: scaffolding -> file write -> tests
    -> declare_complete. Verifier passes; session reaches COMPLETE;
    completion narration says 'Done'."""
    stack = _build_stack(tmp_path)
    project = stack.sandbox / "hello_cli"
    project.mkdir()
    session = stack.create_session(
        project_root=project,
        intent="Create a Python script that prints hello world",
    )

    # Simulate a Python file + a passing test file so the verifier accepts.
    script = (
        ClaudeScript()
        .progress("scaffolding", "set up project layout", ["pyproject.toml"])
        .write_file("pyproject.toml", '[project]\nname = "hello"\nversion = "0.1.0"\n')
        .progress("implementing", "wrote main script", ["main.py"])
        .write_file("main.py", "def hello():\n    return 'hello'\n\nif __name__ == '__main__':\n    print(hello())\n")
        .progress("writing tests", "added unit tests", ["test_main.py"])
        .write_file(
            "test_main.py",
            "from main import hello\n\ndef test_hello():\n    assert hello() == 'hello'\n",
        )
        .test_results(passing=1, failing=0, details="all green")
        .declare_complete(
            summary="Hello world CLI; 1 test passing",
            entry_point="main.py",
            run_command="python main.py",
            files_created=["pyproject.toml", "main.py", "test_main.py"],
        )
    )
    bridge = ScriptedClaudeBridge(stack.server, script, session_id=session.session_id)
    handle = bridge.submit(TaskRequest(
        task_prompt="hello", cwd=project, model="haiku",
        timeout_s=60.0, label="scenario_1",
    ))
    result = handle.wait(timeout=60.0)

    assert result is not None and result.success, (
        f"task did not succeed: {result and result.error}"
    )
    final = stack.server.get_session_state(session.session_id)
    assert final.status == SessionStatus.COMPLETE, (
        f"expected COMPLETE, got {final.status.value}"
    )
    assert (project / "main.py").is_file()
    assert (project / "test_main.py").is_file()
    # Completion narration
    narration = stack.narrator.narrate(final)
    assert "Done" in narration, narration


# ---------------------------------------------------------------------------
# Scenario 2 — existing project edit (only targeted project changes)
# ---------------------------------------------------------------------------


def test_scenario_2_existing_project_edit_isolates_changes(tmp_path: Path):
    """Two pre-existing projects in the sandbox. We submit a task targeting
    project A. After completion, project A's files changed but project B
    was untouched."""
    stack = _build_stack(tmp_path)
    project_a = stack.sandbox / "calculator"
    project_a.mkdir()
    (project_a / "ops.py").write_text("def add(a, b): return a + b\n")
    (project_a / "test_ops.py").write_text(
        "from ops import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    )
    (project_a / "pyproject.toml").write_text('[project]\nname = "calc"\nversion = "0.1.0"\n')
    project_b = stack.sandbox / "weather"
    project_b.mkdir()
    (project_b / "fetcher.py").write_text("def fetch(): return 'weather'\n")
    (project_b / "pyproject.toml").write_text('[project]\nname = "weather"\nversion = "0.1.0"\n')
    b_pre_mtimes = {p.name: p.stat().st_mtime for p in project_b.iterdir()}

    stack.registry.add(Project(
        name="Calculator", path=str(project_a),
        aliases=["calc"], language="python",
    ))
    stack.registry.add(Project(
        name="Weather", path=str(project_b),
        aliases=["weather app"], language="python",
    ))

    session = stack.create_session(
        project_root=project_a,
        intent="Add a subtract function to my calculator",
        mode="edit",
    )

    script = (
        ClaudeScript()
        .progress("editing ops.py", "added subtract()", ["ops.py"])
        .modify_file(
            "ops.py",
            "def add(a, b): return a + b\n\ndef subtract(a, b): return a - b\n",
        )
        .progress("updating tests", "covers subtract", ["test_ops.py"])
        .modify_file(
            "test_ops.py",
            (
                "from ops import add, subtract\n\n"
                "def test_add():\n    assert add(2, 3) == 5\n\n"
                "def test_subtract():\n    assert subtract(5, 2) == 3\n"
            ),
        )
        .test_results(passing=2, failing=0)
        .declare_complete(
            summary="Added subtract; 2 tests passing",
            entry_point=None,
            files_modified=["ops.py", "test_ops.py"],
        )
    )
    bridge = ScriptedClaudeBridge(stack.server, script, session_id=session.session_id)
    handle = bridge.submit(TaskRequest(
        task_prompt="add subtract", cwd=project_a, model="haiku",
        timeout_s=60.0, label="scenario_2",
    ))
    result = handle.wait(timeout=60.0)
    assert result is not None and result.success

    # Project B untouched.
    b_post_mtimes = {p.name: p.stat().st_mtime for p in project_b.iterdir()}
    assert b_post_mtimes == b_pre_mtimes, (
        f"project B was modified: pre={b_pre_mtimes} post={b_post_mtimes}"
    )

    # Project A has the new function.
    assert "subtract" in (project_a / "ops.py").read_text()
    assert "test_subtract" in (project_a / "test_ops.py").read_text()


# ---------------------------------------------------------------------------
# Scenario 3 — clarification answered from heuristics (no LLM, no escalation)
# ---------------------------------------------------------------------------


def test_scenario_3_clarification_answered_without_escalation(tmp_path: Path):
    """Claude asks a low-stakes implementation question (test framework).
    The coordinator's RULE_ANSWER fast path returns a sensible default
    without escalating to the user."""
    stack = _build_stack(tmp_path)
    project = stack.sandbox / "hello"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname = "hello"\nversion = "0.1.0"\n')
    session = stack.create_session(
        project_root=project,
        intent="Create a small Python script that prints hello",
    )

    captured_answers: List[str] = []

    script = (
        ClaudeScript()
        .progress("planning", "thinking about layout", [])
        .clarify(
            "Which test framework should I use?",
            on_answer=captured_answers.append,
        )
        .write_file("main.py", "print('hi')\n")
        .write_file(
            "test_main.py",
            "import subprocess, sys\n\n"
            "def test_runs():\n    r = subprocess.run([sys.executable, 'main.py'], capture_output=True, text=True)\n"
            "    assert 'hi' in r.stdout\n",
        )
        .test_results(passing=1, failing=0)
        .declare_complete(
            summary="hello CLI", entry_point="main.py",
            files_created=["main.py", "test_main.py"],
        )
    )
    bridge = ScriptedClaudeBridge(stack.server, script, session_id=session.session_id)
    handle = bridge.submit(TaskRequest(
        task_prompt="hello", cwd=project, model="haiku",
        timeout_s=60.0, label="scenario_3",
    ))
    result = handle.wait(timeout=60.0)
    assert result is not None and result.success
    # The coordinator answered without escalating (no pending user clarification).
    assert stack.coordinator.pending_user_clarifications() == []
    assert captured_answers, "no clarification answer captured"
    assert "pytest" in captured_answers[0].lower(), captured_answers[0]


# ---------------------------------------------------------------------------
# Scenario 4 — clarification escalated to user, voice response resolves it
# ---------------------------------------------------------------------------


def test_scenario_4_clarification_escalated_and_resolved_by_voice(tmp_path: Path):
    """Claude asks about an external paid service. Coordinator's
    RULE_ESCALATE fires; user-voice response resolves the clarification."""
    stack = _build_stack(tmp_path, llm=_StubLLM(
        response_text="On the project, Claude wants to know if you'd pay for the OpenAI API. Your call.",
    ))
    project = stack.sandbox / "ai_app"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname = "ai"\nversion = "0.1.0"\n')
    session = stack.create_session(
        project_root=project, intent="Build something with an LLM",
    )

    captured_answers: List[str] = []

    # Spawn a thread that simulates the user voice answering after escalation.
    answer_thread_started = False
    def _user_responder() -> None:
        # Wait until the coordinator has surfaced a pending clarification
        # to the voice loop.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            pending = stack.coordinator.pending_user_clarifications()
            if pending:
                stack.coordinator.deliver_user_clarification_response(
                    pending[0].request_id,
                    "Use the free Anthropic tier for now.",
                )
                return
            time.sleep(0.05)

    import threading
    threading.Thread(target=_user_responder, daemon=True).start()

    script = (
        ClaudeScript()
        .clarify(
            "Should I use the OpenAI paid API tier or Anthropic?",
            on_answer=captured_answers.append,
        )
        .write_file("app.py", "print('app')\n")
        .write_file(
            "test_app.py",
            "def test_smoke():\n    assert True\n",
        )
        .test_results(passing=1, failing=0)
        .declare_complete(
            summary="app", entry_point="app.py",
            files_created=["app.py", "test_app.py"],
        )
    )
    bridge = ScriptedClaudeBridge(stack.server, script, session_id=session.session_id)
    handle = bridge.submit(TaskRequest(
        task_prompt="ai", cwd=project, model="haiku",
        timeout_s=60.0, label="scenario_4",
    ))
    result = handle.wait(timeout=60.0)
    assert result is not None and result.success, (
        f"task did not finish: {result and result.error}"
    )
    assert captured_answers, "no clarification answer received"
    assert "anthropic" in captured_answers[0].lower(), captured_answers[0]


# ---------------------------------------------------------------------------
# Scenario 5 — verification failure + correction loop succeeds
# ---------------------------------------------------------------------------


def test_scenario_5_verification_failure_then_correction_succeeds(tmp_path: Path):
    """First declare_complete claims a file that doesn't exist. Verifier
    fails (FILES_EXIST). Coordinator returns a correction prompt. Script
    reacts by writing the missing file and declaring again. Second
    verification passes -> COMPLETE."""
    stack = _build_stack(tmp_path)
    project = stack.sandbox / "broken"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname = "broken"\nversion = "0.1.0"\n')
    session = stack.create_session(
        project_root=project, intent="Create a small Python module with tests",
    )

    second_pass_done: List[bool] = []

    def _on_first_declare(ctx) -> None:
        # After the first (failing) declare_complete, the session is back
        # at EXECUTING (correction phase). Simulate Claude responding to
        # the correction by writing the missing file.
        s = stack.server.get_session_state(session.session_id)
        # If the coordinator put us in CORRECTING/EXECUTING we can recover.
        if s.status not in (SessionStatus.EXECUTING, SessionStatus.CORRECTING):
            return
        # The correction prompt was returned in ctx.declare_complete_response.
        assert "verification" in (ctx.declare_complete_response or "").lower()
        # Mirror Claude's behavior: actually create the missing file.
        (project / "real.py").write_text("def f(): return 1\n")
        (project / "test_real.py").write_text(
            "from real import f\n\ndef test_f():\n    assert f() == 1\n",
        )
        # Record the new test_results + new declare_complete.
        stack.server.store.record_test_results(
            session.session_id, passing=1, failing=0, skipped=0, details="ok",
        )
        from ultron.coding.session import CompletionClaim
        claim = CompletionClaim(
            summary="Real module", entry_point="real.py",
            files_created=["real.py", "test_real.py"],
        )
        stack.server.store.record_completion_claim(session.session_id, claim)
        try:
            stack.server.store.transition(session.session_id, SessionStatus.VERIFYING)
        except Exception:
            pass
        # Drive the handler in this thread (synchronously).
        async def _drive() -> str:
            return await stack.server._declare_complete_handler(session.session_id)  # type: ignore[attr-defined]
        ctx.declare_complete_response = asyncio.run(_drive())
        second_pass_done.append(True)

    script = (
        ClaudeScript()
        # First pass: claim a file that doesn't exist -> verifier fails.
        .progress("scaffolding", "set up", ["real.py"])
        .declare_complete(
            summary="early claim",
            files_created=["real.py", "test_real.py"],
        )
        # The hook above runs the corrective second pass.
        .callback(_on_first_declare)
    )
    bridge = ScriptedClaudeBridge(stack.server, script, session_id=session.session_id)
    handle = bridge.submit(TaskRequest(
        task_prompt="broken", cwd=project, model="haiku",
        timeout_s=60.0, label="scenario_5",
    ))
    result = handle.wait(timeout=60.0)
    assert result is not None
    assert second_pass_done, "second-pass correction never ran"
    final = stack.server.get_session_state(session.session_id)
    # After the corrected re-declare, verification should pass.
    assert final.status == SessionStatus.COMPLETE, (
        f"expected COMPLETE, got {final.status.value}"
    )
    assert final.verification_failures >= 1, (
        "expected at least one recorded verification failure"
    )


# ---------------------------------------------------------------------------
# Scenario 6 — mid-project adjustment is routed via voice
# ---------------------------------------------------------------------------


def test_scenario_6_mid_project_adjustment_via_voice(tmp_path: Path):
    """User issues a mid-session adjustment via voice. Coordinator
    decides FOLLOWUP and the runner queues the new prompt."""
    stack = _build_stack(tmp_path, llm=_StubLLM(
        response_text="Switch from requests to httpx; keep the existing CLI structure.",
    ))
    project = stack.sandbox / "weather"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname = "w"\nversion = "0.1.0"\n')
    session = stack.create_session(
        project_root=project,
        intent="Build a weather CLI using requests",
    )
    # Stage a piece of progress so the coordinator has context.
    stack.server.store.record_stage(
        session.session_id,
        stage="implementing", summary="initial fetch via requests",
        files_touched=["main.py"],
    )

    # We don't actually run the task here -- just verify the voice -> coordinator
    # routing produces an AdjustmentDecision. Use a sleeping script so the
    # runner has an active task while we issue the adjustment.
    script = ClaudeScript().sleep(0.5).declare_complete(
        summary="ok", files_created=["main.py"],
    )
    bridge = ScriptedClaudeBridge(stack.server, script, session_id=session.session_id)
    # Start the task on the runner so has_active_task() is True.
    handle = bridge.submit(TaskRequest(
        task_prompt="build", cwd=project, model="haiku",
        timeout_s=10.0, label="scenario_6",
    ))
    # Replace the runner's handle slot so the voice controller sees an active task.
    stack.runner._handle = handle  # noqa: SLF001
    stack.runner._claude_session_id = handle.claude_session_id  # noqa: SLF001
    stack.runner._project_cwd = project  # noqa: SLF001

    # Spy on send_followup so we can verify the runner gets the adjustment.
    sent: List[Dict[str, Any]] = []
    original_send_followup = stack.runner.send_followup
    def _spy_send(prompt: str, kind: str = "adjustment"):
        sent.append({"prompt": prompt, "kind": kind})
        return None  # don't actually submit
    stack.runner.send_followup = _spy_send  # type: ignore[assignment]

    response = stack.voice.handle_utterance(
        "Actually have him use httpx instead of requests."
    )
    assert response is not None
    assert response.handled
    assert sent, "send_followup never called"
    assert sent[0]["kind"] == "adjustment"
    assert "httpx" in sent[0]["prompt"].lower()

    # Restore original method + drain the script.
    stack.runner.send_followup = original_send_followup  # type: ignore[assignment]
    handle.cancel()
    handle.wait(timeout=2.0)


# ---------------------------------------------------------------------------
# Scenario 7 — status query during execution returns delta-aware narration
# ---------------------------------------------------------------------------


def test_scenario_7_status_query_during_execution(tmp_path: Path):
    """Mid-script, the user asks 'how's it going?'. The voice controller
    routes through the runner+narrator, returning a delta-aware status
    that mentions current stage + recent files."""
    stack = _build_stack(tmp_path)
    project = stack.sandbox / "longrun"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname = "long"\nversion = "0.1.0"\n')
    session = stack.create_session(
        project_root=project, intent="Build a long-running thing",
    )

    # Manually populate state instead of running a long script -- this
    # exercises the narration pathway directly through the voice loop.
    stack.server.store.record_stage(
        session.session_id, stage="building module foo",
        summary="implemented foo.py", files_touched=["foo.py"],
    )

    # The intent classifier requires has_active_task=True. Use the
    # session-store-based path: we already have an EXECUTING session so
    # _has_active_task_or_session() returns True.
    response = stack.voice.handle_utterance("How's it going?")
    assert response is not None
    assert response.handled
    text = response.text.lower()
    # Should mention something about the current stage / build context.
    assert any(needle in text for needle in (
        "building", "module", "foo", "1 file",
    )), f"narration didn't mention current state: {response.text!r}"


# ---------------------------------------------------------------------------
# Scenario 8 — cancellation tears the bridge down cleanly
# ---------------------------------------------------------------------------


def test_scenario_8_cancellation_terminates_session(tmp_path: Path):
    """User cancels mid-task via voice. The bridge handle ends with
    cancelled=True and the session can be marked TERMINATED."""
    stack = _build_stack(tmp_path)
    project = stack.sandbox / "cancelled"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname = "c"\nversion = "0.1.0"\n')
    session = stack.create_session(
        project_root=project, intent="long task",
    )
    script = (
        ClaudeScript()
        .progress("starting", "kicking off", [])
        .sleep(2.0)              # plenty of time to issue a cancel
        .progress("never_reached", "should not run", [])
        .declare_complete(summary="should not happen", files_created=[])
    )
    bridge = ScriptedClaudeBridge(stack.server, script, session_id=session.session_id)
    handle = bridge.submit(TaskRequest(
        task_prompt="long", cwd=project, model="haiku",
        timeout_s=10.0, label="scenario_8",
    ))
    stack.runner._handle = handle  # noqa: SLF001
    stack.runner._claude_session_id = handle.claude_session_id  # noqa: SLF001
    stack.runner._project_cwd = project  # noqa: SLF001
    time.sleep(0.05)

    response = stack.voice.handle_utterance("Cancel the task.")
    assert response is not None
    assert response.cancelled
    result = handle.wait(timeout=5.0)
    assert result is not None
    assert not result.success


# ---------------------------------------------------------------------------
# Scenario 9 — Haiku threshold trips model_escalation_count
# ---------------------------------------------------------------------------


def test_scenario_9_model_escalation_after_haiku_threshold(tmp_path: Path):
    """After CODING_ESCALATION_THRESHOLD_DEFAULT verification failures,
    the session's model_escalation_count is bumped. After the total
    threshold (default 3 + escalation 2 = 5 failures), session is FAILED."""
    from config import settings

    stack = _build_stack(tmp_path)
    project = stack.sandbox / "fails"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0.1.0"\n')
    session = stack.create_session(
        project_root=project, intent="failing project",
    )

    haiku_thresh = settings.CODING_ESCALATION_THRESHOLD_DEFAULT
    total_thresh = (
        settings.CODING_ESCALATION_THRESHOLD_DEFAULT
        + settings.CODING_ESCALATION_THRESHOLD_ESCALATION
    )

    # Trigger N back-to-back failed declare_complete calls. Each time
    # the verifier fails on FILES_EXIST (claimed file doesn't exist).
    from ultron.coding.session import CompletionClaim
    for i in range(total_thresh + 1):
        # Make sure the session is in a state that allows VERIFYING.
        s = stack.server.get_session_state(session.session_id)
        if s.status not in (SessionStatus.VERIFYING, SessionStatus.EXECUTING, SessionStatus.CORRECTING):
            break
        if s.status == SessionStatus.CORRECTING:
            stack.server.store.transition(session.session_id, SessionStatus.EXECUTING)
        if s.status == SessionStatus.EXECUTING:
            try:
                stack.server.store.transition(session.session_id, SessionStatus.VERIFYING)
            except Exception:
                pass
        stack.server.store.record_completion_claim(
            session.session_id,
            CompletionClaim(
                summary=f"attempt {i}", entry_point="real.py",
                files_created=["real.py"],
            ),
        )
        async def _drive() -> str:
            return await stack.server._declare_complete_handler(session.session_id)  # type: ignore[attr-defined]
        try:
            asyncio.run(_drive())
        except Exception:
            pass
        # Coordinator transitions the session; if it's now FAILED we stop.
        if stack.server.get_session_state(session.session_id).status == SessionStatus.FAILED:
            break

    final = stack.server.get_session_state(session.session_id)
    # Exact behavior: FAILED at total threshold, model_escalation_count bumped at haiku threshold.
    assert final.verification_failures >= haiku_thresh, (
        f"expected at least {haiku_thresh} failures, got {final.verification_failures}"
    )
    if final.verification_failures >= haiku_thresh:
        assert final.model_escalation_count >= 1, (
            "expected escalation flag set after haiku threshold"
        )
    if final.verification_failures >= total_thresh:
        assert final.status == SessionStatus.FAILED, (
            f"expected FAILED after total threshold, got {final.status.value}"
        )


# ---------------------------------------------------------------------------
# Scenario 10 — project root outside sandbox is rejected
# ---------------------------------------------------------------------------


def test_scenario_10_project_root_outside_sandbox_rejected(tmp_path: Path, monkeypatch):
    """The MCP server's _validate_project_root refuses paths outside
    CODING_SANDBOX_PATH unless explicitly relaxed. Phase 6 turns the
    relax flag OFF for this test to exercise the production check."""
    monkeypatch.delenv("ULTRON_CODING_MCP_ALLOW_ANY_ROOT", raising=False)
    # Production sandbox path; we attempt to create a session OUTSIDE it.
    server = UltronMCPServer(host="127.0.0.1", port=0)
    bad_path = tmp_path / "elsewhere"
    bad_path.mkdir()
    with pytest.raises(ValueError, match="sandbox"):
        server.create_session(
            project_root=bad_path, initial_prompt="should fail",
        )


# ---------------------------------------------------------------------------
# Scenario 7b — narration during multi-stage progress (delta tracking)
# ---------------------------------------------------------------------------


def test_scenario_7b_status_query_delta_tracking(tmp_path: Path):
    """Two consecutive status queries with new state in between -- the
    second query mentions only the delta."""
    stack = _build_stack(tmp_path)
    project = stack.sandbox / "delta"
    project.mkdir()
    session = stack.create_session(
        project_root=project, intent="Multi-stage task",
    )
    stack.server.store.record_stage(
        session.session_id, stage="step1", summary="first",
        files_touched=["a.py"],
    )

    out1 = stack.voice.handle_utterance("How's it going?")
    assert out1 is not None and out1.handled

    # New state arrives.
    time.sleep(0.01)
    stack.server.store.record_stage(
        session.session_id, stage="step2", summary="second",
        files_touched=["b.py"],
    )
    stack.server.store.record_test_results(
        session.session_id, passing=3, failing=0, skipped=0, details="ok",
    )

    out2 = stack.voice.handle_utterance("How's it going?")
    assert out2 is not None and out2.handled
    text = out2.text.lower()
    # The second query should mention deltas (new file or new stage).
    assert any(s in text for s in (
        "step2", "since you last asked", "1 new file", "step", "passing",
    )), f"second narration didn't reflect delta: {out2.text!r}"
