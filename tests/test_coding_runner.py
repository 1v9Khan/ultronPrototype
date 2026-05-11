"""Tests for :class:`CodingTaskRunner`.

Uses a fake :class:`CodingBridge` so we can drive the runner through a
deterministic event sequence without spawning a subprocess. The fake
bridge is also a useful template for what the future OpenClaw bridge
needs to look like.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional

import pytest

from ultron.coding.bridge import (
    CodingBridge,
    EventKind,
    EventListener,
    FileChangeKind,
    TaskEvent,
    TaskHandle,
    TaskRequest,
    TaskResult,
    TaskState,
)
from ultron.coding.runner import CodingTaskRunner


# ---------------------------------------------------------------------------
# Fake bridge / handle: feed the runner a scripted event sequence.
# ---------------------------------------------------------------------------


class _FakeHandle(TaskHandle):
    def __init__(self, request: TaskRequest):
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
        self._cancelled = False
        self._task_id = "fake-001"

    def task_id(self) -> str:
        return self._task_id

    def state(self) -> TaskState:
        from dataclasses import replace
        return replace(
            self._state,
            completed_steps=list(self._state.completed_steps),
            files_created=list(self._state.files_created),
            files_modified=list(self._state.files_modified),
            files_deleted=list(self._state.files_deleted),
        )

    def add_listener(self, listener: EventListener) -> None:
        self._listeners.append(listener)

    def cancel(self) -> None:
        self._cancelled = True

    def wait(self, timeout: Optional[float] = None) -> TaskResult:
        self._done.wait(timeout=timeout)
        return self._result  # type: ignore[return-value]

    def is_running(self) -> bool:
        return not self._done.is_set()

    # --- test helpers ------------------------------------------------------

    def fire(self, event: TaskEvent) -> None:
        # Apply state update + fan out to listeners (mimics direct bridge).
        self._apply_to_state(event)
        for L in list(self._listeners):
            L(event)

    def finish(self, *, success: bool, summary: str = "ok") -> None:
        result = TaskResult(
            success=success,
            exit_status=0 if success else 1,
            summary=summary,
            duration_s=time.time() - self._state.started_at,
            files_created=list(self._state.files_created),
            files_modified=list(self._state.files_modified),
        )
        self._result = result
        self._state.is_complete = True
        self._state.success = success
        self._state.duration_s = result.duration_s
        self._state.final_summary = summary
        self._done.set()
        complete_event = TaskEvent(
            kind=EventKind.COMPLETE,
            summary=summary,
            exit_status=result.exit_status,
            files_created=result.files_created,
            files_modified=result.files_modified,
            duration_s=result.duration_s,
        )
        for L in list(self._listeners):
            L(complete_event)

    def _apply_to_state(self, event: TaskEvent) -> None:
        if event.kind == EventKind.STATUS and event.stage:
            self._state.current_step = event.stage
        elif event.kind == EventKind.TOOL_USE:
            self._state.tool_use_count += 1
            if event.tool_name:
                self._state.last_tool_use = event.tool_name
                self._state.completed_steps.append(event.tool_name.lower())
                self._state.current_step = event.tool_name.lower()
        elif event.kind == EventKind.FILE_CHANGE and event.file_path:
            target = (
                self._state.files_created
                if event.file_change_kind == FileChangeKind.CREATED
                else self._state.files_modified
                if event.file_change_kind == FileChangeKind.MODIFIED
                else self._state.files_deleted
            )
            if event.file_path not in target:
                target.append(event.file_path)
        elif event.kind == EventKind.TEXT and event.text:
            self._state.text_chars_emitted += len(event.text)


class _FakeBridge(CodingBridge):
    def __init__(self):
        self.last_handle: Optional[_FakeHandle] = None

    def submit(self, request: TaskRequest) -> TaskHandle:
        h = _FakeHandle(request)
        self.last_handle = h
        return h

    def name(self) -> str:
        return "fake"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_runner_reports_no_active_task_initially(tmp_path: Path):
    runner = CodingTaskRunner(bridge=_FakeBridge(), log_path=tmp_path / "log.jsonl")
    assert not runner.has_active_task()
    assert runner.active_state() is None
    assert "No coding task" in runner.progress_narration()


def test_runner_blocks_concurrent_tasks(tmp_path: Path):
    bridge = _FakeBridge()
    runner = CodingTaskRunner(bridge=bridge, log_path=tmp_path / "log.jsonl")
    runner.start_task(TaskRequest(task_prompt="t1", cwd=tmp_path, label="task1"))
    with pytest.raises(RuntimeError):
        runner.start_task(TaskRequest(task_prompt="t2", cwd=tmp_path, label="task2"))


def test_progress_narration_shows_current_step_and_deltas(tmp_path: Path):
    bridge = _FakeBridge()
    runner = CodingTaskRunner(bridge=bridge, log_path=tmp_path / "log.jsonl")
    runner.start_task(TaskRequest(task_prompt="build it", cwd=tmp_path, label="x"))
    h = bridge.last_handle
    assert h is not None

    h.fire(TaskEvent(kind=EventKind.STATUS, stage="running"))
    h.fire(TaskEvent(kind=EventKind.TOOL_USE, tool_name="Write",
                     tool_input={"file_path": "main.py"}))
    h.fire(TaskEvent(kind=EventKind.FILE_CHANGE,
                     file_path=Path("main.py"),
                     file_change_kind=FileChangeKind.CREATED))

    n1 = runner.progress_narration()
    assert "Currently" in n1
    assert "1 new file" in n1 or "1 new file" in n1.lower()

    # Second poll should report no new since last poll.
    n2 = runner.progress_narration()
    assert "no new completed steps" in n2.lower() or "since you last asked" in n2.lower()


def test_completion_narration_summarizes_success(tmp_path: Path):
    bridge = _FakeBridge()
    runner = CodingTaskRunner(bridge=bridge, log_path=tmp_path / "log.jsonl")
    runner.start_task(TaskRequest(task_prompt="t", cwd=tmp_path, label="task"))
    h = bridge.last_handle
    assert h is not None
    h.fire(TaskEvent(kind=EventKind.FILE_CHANGE, file_path=Path("main.py"),
                     file_change_kind=FileChangeKind.CREATED))
    h.fire(TaskEvent(kind=EventKind.FILE_CHANGE, file_path=Path("test_main.py"),
                     file_change_kind=FileChangeKind.CREATED))
    h.finish(success=True, summary="all tests pass")

    narration = runner.completion_narration()
    assert "Done." in narration
    assert "2 files" in narration or "2 file" in narration
    # 2026-05-11 follow-up fix: narration speaks the project folder
    # leaf name only, never the absolute path. The absolute path
    # caused XTTS to hang trying to pronounce backslashes/drive
    # letters and pinned the GPU at 100 % in a real session.
    assert tmp_path.name in narration
    assert str(tmp_path) not in narration


def test_completion_narration_handles_failure(tmp_path: Path):
    bridge = _FakeBridge()
    runner = CodingTaskRunner(bridge=bridge, log_path=tmp_path / "log.jsonl")
    runner.start_task(TaskRequest(task_prompt="t", cwd=tmp_path, label="task"))
    h = bridge.last_handle
    assert h is not None
    h.finish(success=False, summary="something broke")

    narration = runner.completion_narration()
    assert "failed" in narration.lower()


def test_completion_narration_honest_when_success_but_zero_files(tmp_path: Path):
    """2026-05-11 narration honesty: a clean Claude exit with zero
    file changes happens when the model burned budget on exploration
    without writing anything (the PDF->docx bug). The legacy "Done."
    opener was misleading -- the user heard "Done" and opened an
    empty folder. The narration now surfaces the no-files case
    explicitly so the user knows to say continue or rephrase."""
    bridge = _FakeBridge()
    runner = CodingTaskRunner(bridge=bridge, log_path=tmp_path / "log.jsonl")
    runner.start_task(TaskRequest(task_prompt="t", cwd=tmp_path, label="task"))
    h = bridge.last_handle
    assert h is not None
    # Clean success exit, zero file changes, with a generic Claude
    # tail line that the bug log showed should NOT make it into
    # the narration (it added noise after the honest "no files"
    # opener).
    h.finish(success=True, summary="What would you like me to work on?")

    narration = runner.completion_narration()
    # Honest opener, not "Done."
    assert "Done." not in narration
    assert "without writing or modifying" in narration.lower() or \
           "no files" in narration.lower() or \
           "didn't" in narration.lower()
    # Generic tail line is suppressed (noise on top of the honest
    # opener -- the bug log showed this confused the user).
    assert "What would you like me to work on" not in narration
    # 2026-05-11 follow-up fix: project FOLDER NAME + elapsed are
    # still surfaced for visibility (the absolute path was dropped
    # because XTTS hung trying to pronounce Windows paths).
    assert tmp_path.name in narration
    assert str(tmp_path) not in narration
    assert "Elapsed:" in narration


def test_completion_narration_does_not_leak_full_path(tmp_path: Path):
    """2026-05-11 follow-up fix regression test. The legacy completion
    narration interpolated ``state.cwd`` (absolute Path) directly,
    producing voice text like ``"Project root: C:\\STC\\...\\sandbox\\X."``
    The XTTS-v2 neural TTS choked on the backslash-colon-drive-letter
    sequence in a live session: GPU pinned at 100 %, synth eventually
    timed out, computer lagged. The fix speaks only ``path.name`` (the
    project folder leaf). This test pins the no-backslashes /
    no-drive-letter invariant against future regressions.

    The Path-leaf approach also keeps the narration speakable on any
    platform: Posix tmpdirs (``/tmp/pytest-of-x/...``) and Windows
    tmpdirs (``C:\\Users\\...\\AppData\\Local\\Temp\\...``) both
    collapse to a single leaf component when ``.name`` is applied.
    """
    bridge = _FakeBridge()
    runner = CodingTaskRunner(bridge=bridge, log_path=tmp_path / "log.jsonl")
    runner.start_task(TaskRequest(task_prompt="t", cwd=tmp_path, label="task"))
    h = bridge.last_handle
    assert h is not None
    h.fire(TaskEvent(kind=EventKind.FILE_CHANGE, file_path=Path("main.py"),
                     file_change_kind=FileChangeKind.CREATED))
    h.finish(success=True, summary="implemented main.py")

    narration = runner.completion_narration()

    # No backslashes anywhere in the narration text -- they were the
    # exact character class that hung XTTS in the live session log.
    assert "\\" not in narration, (
        f"narration must not contain backslashes; got: {narration!r}"
    )
    # No Windows drive-letter prefix (e.g. ``C:\\`` or ``D:``).
    # Match anywhere in narration to catch path-like strings that
    # might be interpolated from elsewhere in the future.
    import re
    assert not re.search(r"\b[A-Za-z]:[\\/]", narration), (
        f"narration must not contain a Windows drive-letter path; "
        f"got: {narration!r}"
    )
    # And no full absolute path interpolation.
    assert str(tmp_path) not in narration, (
        f"narration must not contain the absolute path; "
        f"got: {narration!r}"
    )
    # The project folder leaf IS expected for human context.
    assert tmp_path.name in narration


def test_completion_narration_keeps_done_when_files_were_written(tmp_path: Path):
    """The honest-no-files path must NOT trigger when Claude actually
    wrote files. Existing test covers the simple case; this one
    explicitly pins the success-with-files branch so the new code
    can't accidentally widen its trigger."""
    bridge = _FakeBridge()
    runner = CodingTaskRunner(bridge=bridge, log_path=tmp_path / "log.jsonl")
    runner.start_task(TaskRequest(task_prompt="t", cwd=tmp_path, label="task"))
    h = bridge.last_handle
    assert h is not None
    h.fire(TaskEvent(kind=EventKind.FILE_CHANGE, file_path=Path("main.py"),
                     file_change_kind=FileChangeKind.CREATED))
    h.finish(success=True, summary="implemented main.py")

    narration = runner.completion_narration()
    assert "Done." in narration
    assert "1 file" in narration or "1 new file" in narration.lower()


def test_completion_narration_handles_cancellation(tmp_path: Path):
    bridge = _FakeBridge()
    runner = CodingTaskRunner(bridge=bridge, log_path=tmp_path / "log.jsonl")
    runner.start_task(TaskRequest(task_prompt="t", cwd=tmp_path, label="task"))
    h = bridge.last_handle
    assert h is not None
    # Mimic the direct bridge's cancellation flag flow.
    h._state.is_cancelled = True
    h.finish(success=False, summary="interrupted")
    narration = runner.completion_narration()
    assert "Cancelled." in narration


def test_runner_writes_audit_log(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    bridge = _FakeBridge()
    runner = CodingTaskRunner(bridge=bridge, log_path=log)
    runner.start_task(TaskRequest(task_prompt="t", cwd=tmp_path, label="task"))
    h = bridge.last_handle
    assert h is not None
    h.fire(TaskEvent(kind=EventKind.TOOL_USE, tool_name="Write",
                     tool_input={"file_path": "x.py"}))
    h.finish(success=True, summary="done")
    # At least the start record + a few events should be on disk.
    lines = log.read_text(encoding="utf-8").splitlines()
    assert any("\"kind\": \"start\"" in L for L in lines)
    assert any("\"tool_name\": \"Write\"" in L for L in lines)
    assert any("\"kind\": \"complete\"" in L for L in lines)
