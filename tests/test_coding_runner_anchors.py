"""Tests for goal-anchor planning wired into :class:`CodingTaskRunner`.

The fixtures reuse the ``_FakeBridge`` / ``_FakeHandle`` pattern from
``tests/test_coding_runner.py`` (re-defined here so this file remains
self-contained -- the helpers there are private).
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterator, List, Optional

import pytest

from ultron.coding.anchors import AnchorPlan, GoalAnchor
from ultron.coding.bridge import (
    CodingBridge,
    EventKind,
    EventListener,
    TaskEvent,
    TaskHandle,
    TaskRequest,
    TaskResult,
    TaskState,
)
from ultron.coding.runner import CodingTaskRunner
from ultron.config import (
    UltronConfig,
    get_config,
    reload_config,
    set_config,
)


# ---------------------------------------------------------------------------
# Fake bridge fixtures (mirrors the private helpers in test_coding_runner.py).
# ---------------------------------------------------------------------------


class _FakeHandle(TaskHandle):
    def __init__(self, request: TaskRequest) -> None:
        self._request = request
        self._listeners: List[EventListener] = []
        self._state = TaskState(
            label=request.label or "test",
            task_prompt=request.task_prompt,
            cwd=request.cwd,
            started_at=time.time(),
        )
        self._done = threading.Event()
        self._result: Optional[TaskResult] = None
        self._task_id = "anchor-fake-001"

    def task_id(self) -> str:
        return self._task_id

    def state(self) -> TaskState:
        return replace(self._state)

    def add_listener(self, listener: EventListener) -> None:
        self._listeners.append(listener)

    def cancel(self) -> None:
        pass

    def wait(self, timeout: Optional[float] = None) -> TaskResult:
        self._done.wait(timeout=timeout)
        return self._result  # type: ignore[return-value]

    def is_running(self) -> bool:
        return not self._done.is_set()

    def fire(self, event: TaskEvent) -> None:
        for L in list(self._listeners):
            L(event)


class _FakeBridge(CodingBridge):
    def __init__(self) -> None:
        self.last_handle: Optional[_FakeHandle] = None

    def submit(self, request: TaskRequest) -> TaskHandle:
        h = _FakeHandle(request)
        self.last_handle = h
        return h

    def name(self) -> str:
        return "fake"


def _usage_event(*, total_tokens: int) -> TaskEvent:
    """Build a USAGE TaskEvent carrying ``total_tokens`` of input."""
    return TaskEvent(
        kind=EventKind.USAGE,
        usage_input=total_tokens,
        usage_output=0,
        usage_cache_creation=0,
        usage_cache_read=0,
    )


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def goal_anchors_enabled() -> Iterator[None]:
    """Install a fresh UltronConfig with goal_anchors enabled."""
    previous = get_config()
    cfg = previous.model_copy(deep=True)
    cfg.coding.goal_anchors.enabled = True
    # Make budget warnings predictable: warn at 50% so a small USAGE
    # event crosses the threshold cleanly in tests.
    cfg.coding.goal_anchors.warn_threshold = 0.5
    # Keep min/max anchors at sensible defaults; the decomposer caps at
    # max_anchors which lets us drive multi-anchor exhaustion.
    cfg.coding.goal_anchors.max_anchors = 4
    set_config(cfg)
    yield
    set_config(previous)


@pytest.fixture
def runner_with_anchors(
    tmp_path: Path,
    goal_anchors_enabled,
) -> Iterator[tuple[CodingTaskRunner, _FakeBridge]]:
    """A runner + bridge pair with goal anchors enabled."""
    bridge = _FakeBridge()
    runner = CodingTaskRunner(bridge=bridge, log_path=tmp_path / "audit.jsonl")
    yield runner, bridge


@pytest.fixture
def runner_without_anchors(
    tmp_path: Path,
) -> Iterator[tuple[CodingTaskRunner, _FakeBridge]]:
    """Default-config runner: goal anchors disabled."""
    bridge = _FakeBridge()
    runner = CodingTaskRunner(bridge=bridge, log_path=tmp_path / "audit.jsonl")
    yield runner, bridge


# ---------------------------------------------------------------------------
# Default-OFF: no plan is built, no narration queued
# ---------------------------------------------------------------------------


def test_no_plan_built_when_feature_disabled(
    runner_without_anchors, tmp_path: Path
) -> None:
    runner, bridge = runner_without_anchors
    handle = runner.start_task(
        TaskRequest(
            task_prompt="Build a flask app then test it then deploy it.",
            cwd=tmp_path,
            label="t",
        )
    )
    assert runner.current_anchor() is None
    assert runner.anchor_plan_snapshot() is None
    assert runner.pop_anchor_narration() is None


# ---------------------------------------------------------------------------
# Enabled: plan is built at start_task
# ---------------------------------------------------------------------------


def test_plan_built_at_start_task(
    runner_with_anchors, tmp_path: Path
) -> None:
    runner, bridge = runner_with_anchors
    handle = runner.start_task(
        TaskRequest(
            task_prompt=(
                "Build a flask app, then test the routes, finally deploy it."
            ),
            cwd=tmp_path,
            label="task",
        )
    )
    snapshot = runner.anchor_plan_snapshot()
    assert snapshot is not None
    assert len(snapshot["anchors"]) >= 2
    active = runner.current_anchor()
    assert active is not None
    assert active.order == 0


def test_opening_anchor_narration_queued(
    runner_with_anchors, tmp_path: Path
) -> None:
    runner, _ = runner_with_anchors
    runner.start_task(
        TaskRequest(
            task_prompt="Build the converter, then test it, finally write docs.",
            cwd=tmp_path,
            label="opener",
        )
    )
    narration = runner.pop_anchor_narration()
    assert narration is not None
    assert "Starting" in narration
    assert "anchor 1" in narration
    # The voice-character-lock convention requires no Windows paths or
    # long backslash-slug strings in narration; the helpers should
    # surface only the natural-language description.
    assert "\\" not in narration
    # Pop is one-shot.
    assert runner.pop_anchor_narration() is None


# ---------------------------------------------------------------------------
# Token attribution + warning
# ---------------------------------------------------------------------------


def test_usage_event_updates_active_anchor(
    runner_with_anchors, tmp_path: Path
) -> None:
    runner, bridge = runner_with_anchors
    runner.start_task(
        TaskRequest(
            task_prompt="Build foo, then test foo, then deploy foo.",
            cwd=tmp_path,
            label="attr",
        )
    )
    # Drain the opening narration so subsequent pops surface only
    # event-driven narration.
    runner.pop_anchor_narration()

    handle = bridge.last_handle
    assert handle is not None

    active_before = runner.current_anchor()
    snapshot = runner.anchor_plan_snapshot()
    assert snapshot is not None
    budget = snapshot["anchors"][0]["budget_tokens"]
    # A tiny USAGE event leaves the first anchor active.
    handle.fire(_usage_event(total_tokens=max(1, budget // 10)))
    snapshot_after = runner.anchor_plan_snapshot()
    assert snapshot_after["anchors"][0]["tokens_spent"] >= 1
    assert runner.current_anchor() == active_before


def test_anchor_warning_fires_when_threshold_crossed(
    runner_with_anchors, tmp_path: Path
) -> None:
    runner, bridge = runner_with_anchors
    runner.start_task(
        TaskRequest(
            task_prompt="Build alpha, then test alpha, then deploy alpha.",
            cwd=tmp_path,
            label="warn",
        )
    )
    runner.pop_anchor_narration()  # drain opener

    snapshot = runner.anchor_plan_snapshot()
    budget = snapshot["anchors"][0]["budget_tokens"]
    # warn_threshold is fixture-set to 0.5 -- send 60% of the anchor's
    # budget in one event so the warning latches.
    handle = bridge.last_handle
    handle.fire(_usage_event(total_tokens=int(budget * 0.6)))

    narration = runner.pop_anchor_narration()
    assert narration is not None
    assert "Heads up" in narration
    assert "anchor 1" in narration


def test_anchor_warning_latches_once(
    runner_with_anchors, tmp_path: Path
) -> None:
    runner, bridge = runner_with_anchors
    runner.start_task(
        TaskRequest(
            task_prompt="Build foo, then test foo, then deploy foo.",
            cwd=tmp_path,
            label="latch",
        )
    )
    runner.pop_anchor_narration()

    snapshot = runner.anchor_plan_snapshot()
    budget = snapshot["anchors"][0]["budget_tokens"]
    handle = bridge.last_handle
    handle.fire(_usage_event(total_tokens=int(budget * 0.6)))
    runner.pop_anchor_narration()  # drain the first warning

    # Second USAGE event still within the same anchor -- no new warning.
    handle.fire(_usage_event(total_tokens=int(budget * 0.1)))
    second = runner.pop_anchor_narration()
    # Either None (no new warning, no exhaustion) or transition narration
    # (if 70% is past the cap and triggered exhaustion). Use the snapshot
    # to assert what happened.
    snap_after = runner.anchor_plan_snapshot()
    if snap_after["anchors"][0]["completed"]:
        # Exhausted: the queued narration must be a transition.
        assert second is None or "Moving to" in second
    else:
        # Still in flight: no new warning.
        assert second is None


# ---------------------------------------------------------------------------
# Exhaustion + advance
# ---------------------------------------------------------------------------


def test_anchor_advances_on_exhaustion(
    runner_with_anchors, tmp_path: Path
) -> None:
    runner, bridge = runner_with_anchors
    runner.start_task(
        TaskRequest(
            task_prompt="Build the worker, then test the worker, finally deploy.",
            cwd=tmp_path,
            label="advance",
        )
    )
    runner.pop_anchor_narration()  # drain opener

    snapshot = runner.anchor_plan_snapshot()
    budget = snapshot["anchors"][0]["budget_tokens"]
    handle = bridge.last_handle
    # Drop more than the first anchor's whole budget so it exhausts.
    handle.fire(_usage_event(total_tokens=budget + 100))

    new_active = runner.current_anchor()
    assert new_active is not None
    assert new_active.order == 1
    transition = runner.pop_anchor_narration()
    assert transition is not None
    assert "Moving to" in transition
    assert "anchor 2" in transition


def test_plan_completes_when_all_anchors_exhaust(
    runner_with_anchors, tmp_path: Path
) -> None:
    runner, bridge = runner_with_anchors
    runner.start_task(
        TaskRequest(
            task_prompt="Build it, then test it, then deploy it.",
            cwd=tmp_path,
            label="finish",
        )
    )
    runner.pop_anchor_narration()  # drain opener

    snapshot = runner.anchor_plan_snapshot()
    handle = bridge.last_handle
    total = sum(a["budget_tokens"] for a in snapshot["anchors"])

    # Send tokens equal to the entire plan budget in one go -- each
    # anchor exhausts in turn, the plan finishes.
    handle.fire(_usage_event(total_tokens=total + 100))

    # Active is past the end.
    assert runner.current_anchor() is None
    # The last queued narration should be the completion summary.
    narration = runner.pop_anchor_narration()
    assert narration is not None
    assert "completed" in narration.lower()


# ---------------------------------------------------------------------------
# Resume support via send_followup prefix
# ---------------------------------------------------------------------------


def test_send_followup_prepends_next_anchor_when_unfinished(
    runner_with_anchors, tmp_path: Path
) -> None:
    runner, bridge = runner_with_anchors
    runner.start_task(
        TaskRequest(
            task_prompt="Build the parser, then test the parser, finally deploy.",
            cwd=tmp_path,
            label="resume",
        )
    )
    runner.pop_anchor_narration()

    # Set up the resume preconditions: the initial task has a
    # claude_session_id + project_cwd already (assigned in start_task).
    # Mark the prior handle as done so send_followup doesn't block.
    bridge.last_handle._done.set()  # type: ignore[attr-defined]
    runner._claude_session_id = "stub-session"  # type: ignore[attr-defined]

    # Advance one anchor manually so anchor 2 is the next unfinished.
    snapshot = runner.anchor_plan_snapshot()
    handle = bridge.last_handle
    handle.fire(_usage_event(total_tokens=snapshot["anchors"][0]["budget_tokens"] + 1))
    runner.pop_anchor_narration()

    next_unfinished = runner.next_unfinished_anchor()
    assert next_unfinished is not None
    assert next_unfinished.order == 1

    # send_followup should prepend "Continue with anchor 2: ..." onto
    # the operator's follow-up prompt.
    new_handle = runner.send_followup("Keep going.", kind="adjustment")
    assert new_handle is not None
    submitted_prompt = bridge.last_handle._request.task_prompt  # type: ignore[attr-defined]
    assert "Continue with anchor 2" in submitted_prompt
    assert "Keep going." in submitted_prompt


def test_send_followup_no_prefix_when_plan_fully_consumed(
    runner_with_anchors, tmp_path: Path
) -> None:
    runner, bridge = runner_with_anchors
    runner.start_task(
        TaskRequest(
            task_prompt="Build A, then test A, then deploy A.",
            cwd=tmp_path,
            label="consumed",
        )
    )
    runner.pop_anchor_narration()
    bridge.last_handle._done.set()  # type: ignore[attr-defined]
    runner._claude_session_id = "sess"  # type: ignore[attr-defined]

    snapshot = runner.anchor_plan_snapshot()
    total = sum(a["budget_tokens"] for a in snapshot["anchors"])
    bridge.last_handle.fire(_usage_event(total_tokens=total + 100))
    # Drain any queued narration.
    while runner.pop_anchor_narration() is not None:
        pass

    runner.send_followup("Anything to add?", kind="adjustment")
    submitted = bridge.last_handle._request.task_prompt  # type: ignore[attr-defined]
    assert "Continue with anchor" not in submitted
    assert "Anything to add?" in submitted


# ---------------------------------------------------------------------------
# Audit log entries
# ---------------------------------------------------------------------------


def test_audit_log_records_anchor_lifecycle(
    runner_with_anchors, tmp_path: Path
) -> None:
    runner, bridge = runner_with_anchors
    log_path = tmp_path / "audit.jsonl"
    runner.start_task(
        TaskRequest(
            task_prompt="Build the bot, then test the bot, finally deploy.",
            cwd=tmp_path,
            label="audit",
        )
    )
    snapshot = runner.anchor_plan_snapshot()
    budget = snapshot["anchors"][0]["budget_tokens"]
    bridge.last_handle.fire(_usage_event(total_tokens=budget + 1))

    # Audit should now contain anchor_plan_created + anchor_completed
    # + anchor_started records.
    import json
    rows = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kinds = [r.get("kind") for r in rows]
    assert "anchor_plan_created" in kinds
    assert "anchor_completed" in kinds
    assert "anchor_started" in kinds


# ---------------------------------------------------------------------------
# Fail-open semantics
# ---------------------------------------------------------------------------


def test_disabled_config_means_no_listener(
    runner_without_anchors, tmp_path: Path
) -> None:
    runner, bridge = runner_without_anchors
    runner.start_task(
        TaskRequest(
            task_prompt="Build foo, then test foo, then deploy foo.",
            cwd=tmp_path,
            label="off",
        )
    )
    # A USAGE event should NOT produce any narration when disabled.
    bridge.last_handle.fire(_usage_event(total_tokens=100_000))
    assert runner.pop_anchor_narration() is None
    assert runner.current_anchor() is None


def test_resume_prepend_disabled_via_config(
    runner_with_anchors, tmp_path: Path
) -> None:
    # Disable the resume prefix selectively while keeping the rest of
    # the feature on.
    cfg = get_config().model_copy(deep=True)
    cfg.coding.goal_anchors.resume_prepend_next_anchor = False
    set_config(cfg)
    try:
        runner, bridge = runner_with_anchors
        runner.start_task(
            TaskRequest(
                task_prompt="Build, then test, then deploy.",
                cwd=tmp_path,
                label="no-prefix",
            )
        )
        bridge.last_handle._done.set()  # type: ignore[attr-defined]
        runner._claude_session_id = "s"  # type: ignore[attr-defined]
        runner.send_followup("Continue.", kind="adjustment")
        prompt = bridge.last_handle._request.task_prompt  # type: ignore[attr-defined]
        assert "Continue with anchor" not in prompt
    finally:
        # The runner_with_anchors fixture restores config at teardown,
        # so no extra cleanup needed here.
        pass
