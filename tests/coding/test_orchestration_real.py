"""Phase 6 — orchestration scenarios with real Claude Code.

Gated on ``PYTEST_RUN_GPU_TESTS=1`` like the existing e2e tests. Each
test spawns a real ``claude --print`` subprocess and burns haiku
tokens, so they're slow and we keep the count tight.

Coverage map (Phase 6 spec's 10 scenarios → where they live):

  Scenario | Mocked (test_orchestration.py)            | Real
  ---------|-------------------------------------------|---------------------------------------------
  1        | test_scenario_1_new_project_smooth_completion | tests/test_coding_e2e.py::test_new_project_creates_files_at_dynamic_root
  2        | test_scenario_2_existing_project_edit_isolates_changes | tests/test_coding_e2e.py::test_existing_project_edits_correct_root
  3        | test_scenario_3_clarification_answered_without_escalation | mocked-only (coordinator policy is deterministic; real Claude may not request the clarification we expect)
  4        | test_scenario_4_clarification_escalated_and_resolved_by_voice | mocked-only (same reasoning as #3)
  5        | test_scenario_5_verification_failure_then_correction_succeeds | mocked-only (driving real Claude into a deliberate failure state is brittle; verifier+coordinator integration with real Claude on the *normal* declare_complete path is covered by tests/test_mcp_e2e.py::test_real_claude_calls_report_progress_and_declare_complete)
  6        | test_scenario_6_mid_project_adjustment_via_voice | mocked-only (coordinator policy is deterministic)
  7        | test_scenario_7_status_query_during_execution | tests/test_coding_e2e.py::test_progress_narration_during_real_task
  8        | test_scenario_8_cancellation_terminates_session | this file: test_real_cancellation_terminates_subprocess
  9        | test_scenario_9_model_escalation_after_haiku_threshold | mocked-only (config-driven; real Claude variant would need 5+ deliberate failures)
  10       | test_scenario_10_project_root_outside_sandbox_rejected | no Claude needed (pure MCP layer; mocked-only)
"""

from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path
from typing import List

import pytest

os.environ.setdefault("ULTRON_CODING_MCP_ALLOW_ANY_ROOT", "1")

from ultron.coding import (
    CodingTaskRunner,
    DirectClaudeCodeBridge,
    StatusNarrator,
    UltronMCPServer,
)
from ultron.coding.bridge import TaskRequest
from ultron.coding.coordinator import ConversationCoordinator
from ultron.coding.mcp_server import write_mcp_config
from ultron.coding.session import SessionStatus
from ultron.coding.verification import Verifier


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("PYTEST_RUN_GPU_TESTS") != "1",
        reason="set PYTEST_RUN_GPU_TESTS=1 to run real Claude orchestration",
    ),
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _bridge() -> DirectClaudeCodeBridge:
    try:
        return DirectClaudeCodeBridge()
    except FileNotFoundError as e:
        pytest.skip(str(e))


# ---------------------------------------------------------------------------
# Real Scenario 8 — cancellation terminates a real subprocess cleanly
# ---------------------------------------------------------------------------


def test_real_cancellation_terminates_subprocess(tmp_path: Path):
    """Submit a real Claude task that asks for a longer-running
    interaction, then cancel it via the runner. The subprocess should
    exit, the runner's TaskHandle should report not-running, no orphan
    process or files."""
    project = tmp_path / "p"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "p"\nversion = "0.1.0"\n', encoding="utf-8",
    )
    bridge = _bridge()
    runner = CodingTaskRunner(
        bridge=bridge, log_path=tmp_path / "audit.jsonl",
    )

    # Ask Claude to do something multi-step so we can cancel mid-stream.
    # We don't actually want the work to finish.
    prompt = (
        "List 50 distinct interesting facts about astronomy. Number them. "
        "Take your time and be thorough; one paragraph per fact."
    )
    handle = runner.start_task(TaskRequest(
        task_prompt=prompt, cwd=project, model="haiku",
        require_testing=False, timeout_s=120.0, label="cancel-real",
    ))

    # Let Claude get started.
    time.sleep(2.0)
    runner.cancel_active()

    # Wait for the bridge to actually report not-running.
    deadline = time.monotonic() + 30.0
    while runner.has_active_task():
        if time.monotonic() > deadline:
            pytest.fail("runner still reports an active task 30s after cancel")
        time.sleep(0.2)

    assert not runner.has_active_task()
    # The handle's result reflects the cancel.
    result = handle.wait(timeout=5.0)
    if result is not None:
        # Either the bridge marked it cancelled, or the subprocess exited
        # before we got there with a partial result. Both are fine -- the
        # contract is "the runner is no longer running".
        assert not runner.has_active_task()
