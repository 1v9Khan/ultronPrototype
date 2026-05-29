"""Tests for :class:`CodingVoiceController` against a fake bridge."""

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
from ultron.coding.projects import (
    Project,
    ProjectRegistry,
    ProjectResolver,
    new_sandbox_project,
)
from ultron.coding.runner import CodingTaskRunner
from ultron.coding.voice import CodingVoiceController


# ---------------------------------------------------------------------------
# Reuse the fake bridge from the runner tests (kept inline here so this
# file stands alone if test_coding_runner.py is reorganized later).
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
        self._task_id = "fake"

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
        self._state.is_cancelled = True

    def wait(self, timeout=None) -> TaskResult:
        self._done.wait(timeout=timeout)
        return self._result  # type: ignore[return-value]

    def is_running(self) -> bool:
        return not self._done.is_set()

    def finish(self, success: bool = True, summary: str = "ok") -> None:
        self._result = TaskResult(
            success=success,
            exit_status=0 if success else 1,
            summary=summary,
            duration_s=time.time() - self._state.started_at,
            files_created=list(self._state.files_created),
            files_modified=list(self._state.files_modified),
        )
        self._state.is_complete = True
        self._state.success = success
        self._state.duration_s = self._result.duration_s
        self._state.final_summary = summary
        self._done.set()


class _FakeBridge(CodingBridge):
    def __init__(self):
        self.last: Optional[_FakeHandle] = None
        self.last_request: Optional[TaskRequest] = None

    def submit(self, request: TaskRequest) -> TaskHandle:
        h = _FakeHandle(request)
        self.last = h
        self.last_request = request
        return h

    def name(self) -> str:
        return "fake"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utter_and_dispatch(controller, text):
    """Run ``handle_utterance`` and complete any A4 deferred dispatch.

    The orchestrator (in production) calls ``response.deferred_dispatch()``
    after the pre-task confirmation TTS clears its barge-in watch. The
    voice-controller unit tests below assume the bridge is invoked as a
    side effect of the utterance, so they wrap with this helper to mimic
    the orchestrator without exercising the full TTS / wake-word stack.

    Returns the original :class:`VoiceResponse`.
    """
    out = controller.handle_utterance(text)
    if out is not None and getattr(out, "deferred_dispatch", None) is not None:
        out.deferred_dispatch()
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def setup(tmp_path: Path):
    """Build a fully-configured CodingVoiceController over fake plumbing."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    registry = ProjectRegistry(path=tmp_path / "projects.json")
    resolver = ProjectResolver(registry, embedder=None)
    bridge = _FakeBridge()
    runner = CodingTaskRunner(bridge=bridge, log_path=tmp_path / "log.jsonl")
    controller = CodingVoiceController(
        runner=runner,
        registry=registry,
        resolver=resolver,
        sandbox_root=sandbox,
    )
    return {
        "sandbox": sandbox,
        "registry": registry,
        "bridge": bridge,
        "runner": runner,
        "controller": controller,
    }


# ---------------------------------------------------------------------------
# Non-coding utterances pass through.
# ---------------------------------------------------------------------------


def test_non_coding_utterance_returns_none(setup):
    out = setup["controller"].handle_utterance("What's the weather today?")
    assert out is None


def test_empty_utterance_returns_none(setup):
    assert setup["controller"].handle_utterance("") is None
    assert setup["controller"].handle_utterance("   ") is None


# ---------------------------------------------------------------------------
# New-project flow.
# ---------------------------------------------------------------------------


def test_new_project_creates_sandbox_and_submits_task(setup):
    out = _utter_and_dispatch(
        setup["controller"],
        "Create a Python script called weather_fetcher that pulls forecasts.",
    )
    assert out is not None and out.handled
    assert "weather_fetcher" in out.text.lower()
    # Project registered.
    assert setup["registry"].get("weather_fetcher") is not None
    # Sandbox folder created.
    project = setup["registry"].get("weather_fetcher")
    assert Path(project.path).is_dir()
    # Bridge got a submit with the project as cwd.
    request = setup["bridge"].last_request
    assert request is not None
    assert request.cwd == Path(project.path).resolve() or request.cwd == Path(project.path)


def test_voice_dispatch_defaults_to_no_test_mandate(setup):
    """2026-05-11 token-efficiency: voice-dispatched coding tasks
    used to hardcode ``require_testing=True``, which prepended a
    "MUST write tests + run + fix + re-run" preamble to the Claude
    prompt and 3-5x'd the token spend on small utility asks. The
    voice path now reads ``coding.voice_task_require_testing``
    (default False) -- the bridge gets a clean ``require_testing=False``
    and the discipline preamble is skipped."""
    _utter_and_dispatch(
        setup["controller"],
        "Create a python script that prints hello.",
    )
    request = setup["bridge"].last_request
    assert request is not None
    assert request.require_testing is False, (
        "voice dispatch should default to require_testing=False; "
        "the testing-mandated preamble was inflating token cost on "
        "tiny one-file voice asks"
    )


def test_voice_dispatch_honors_config_flag_for_testing(setup, monkeypatch):
    """Operators who want the testing mandate on every voice dispatch
    can opt back in via ``coding.voice_task_require_testing: true``."""
    from ultron.config import get_config
    cfg = get_config()
    monkeypatch.setattr(cfg.coding, "voice_task_require_testing", True)
    _utter_and_dispatch(
        setup["controller"],
        "Create a python script that prints hello.",
    )
    request = setup["bridge"].last_request
    assert request is not None
    assert request.require_testing is True


def test_render_prompt_omits_discipline_preamble_without_testing():
    """The render_prompt seam itself: ``require_testing=False`` means
    the user's task text reaches Claude verbatim, not wrapped in the
    ~270-token "you MUST write tests" preamble. This is the load-
    bearing change for the token-efficiency win."""
    from pathlib import Path
    from ultron.coding.bridge import TaskRequest, render_prompt

    request_lean = TaskRequest(
        task_prompt="write a hello world",
        cwd=Path("."),
        require_testing=False,
    )
    request_heavy = TaskRequest(
        task_prompt="write a hello world",
        cwd=Path("."),
        require_testing=True,
    )
    lean = render_prompt(request_lean)
    heavy = render_prompt(request_heavy)
    # require_testing=False omits the ~270-token testing mandate (the token-
    # efficiency win); only the light always-on quality preamble + the
    # verbatim body remain.
    assert lean.endswith("write a hello world")
    assert "Write tests for each component" not in lean
    # Heavy preamble starts with "You are working on" and contains
    # the testing mandate items.
    assert lean != heavy
    assert "MUST" in heavy
    assert "tests" in heavy.lower()
    assert "MUST" not in lean
    # And the heavy preamble has substantial weight -- this is the
    # ~270 tokens the user pays for on every voice dispatch with
    # the testing mandate on. Quantify so the test fails noisily if
    # someone re-bloats the preamble.
    assert len(heavy) > len(lean) + 500


def test_coding_config_voice_task_require_testing_defaults_false():
    from ultron.config import CodingConfig
    cfg = CodingConfig()
    assert cfg.voice_task_require_testing is False


def test_new_project_concurrent_request_is_refused(setup):
    _utter_and_dispatch(
        setup["controller"],
        "Create a python script to convert tab files to csv.",
    )
    out = setup["controller"].handle_utterance(
        "Make another quick python tool to do something else."
    )
    assert out is not None
    assert "already running" in out.text.lower()


# ---------------------------------------------------------------------------
# Existing-project flow.
# ---------------------------------------------------------------------------


def test_existing_project_routes_via_resolver(setup, tmp_path: Path):
    # Pre-register a project on disk.
    proj_dir = setup["sandbox"] / "calculator"
    proj_dir.mkdir()
    setup["registry"].add(Project(
        name="Calculator",
        path=str(proj_dir),
        aliases=["calc", "calculator project"],
        language="python",
        description="basic math helpers",
    ))

    out = _utter_and_dispatch(
        setup["controller"],
        "Add a subtract function to my calculator project.",
    )
    assert out is not None and out.handled
    assert "calculator" in out.text.lower()
    request = setup["bridge"].last_request
    assert request is not None
    # Resolved to the existing folder, not a new sandbox subdir.
    assert Path(request.cwd) == proj_dir


def test_missing_directory_for_registered_project_aborts(setup):
    """If the registry points at a path that doesn't exist on disk, the
    voice layer must refuse to launch a task there."""
    setup["registry"].add(Project(
        name="Ghost",
        path=str(setup["sandbox"] / "ghost-not-on-disk"),
        aliases=["the ghost project"],
    ))
    out = setup["controller"].handle_utterance(
        "Add a feature to my ghost project."
    )
    assert out is not None
    assert "missing" in out.text.lower() or "ghost" in out.text.lower()
    # No submit happened.
    assert setup["bridge"].last_request is None


# ---------------------------------------------------------------------------
# Progress query + cancel + completion push.
# ---------------------------------------------------------------------------


def test_progress_query_during_active_task(setup):
    _utter_and_dispatch(
        setup["controller"],
        "Create a python script that prints hello.",
    )
    out = setup["controller"].handle_utterance("How's it going?")
    assert out is not None and out.handled
    assert "currently" in out.text.lower() or "task" in out.text.lower()


def test_cancel_during_active_task(setup):
    _utter_and_dispatch(
        setup["controller"],
        "Create a python script that prints hello.",
    )
    out = setup["controller"].handle_utterance("Stop the task.")
    assert out is not None and out.cancelled
    assert "cancelled" in out.text.lower()


def test_pending_completion_returns_none_until_transition(setup):
    assert setup["controller"].pending_completion() is None
    _utter_and_dispatch(
        setup["controller"],
        "Create a python script that prints hello.",
    )
    # While running, no completion.
    assert setup["controller"].pending_completion() is None
    # Finish the task on the fake bridge.
    setup["bridge"].last.finish(success=True, summary="all good")
    # Wait a tick so the runner's internal accounting has settled.
    time.sleep(0.1)
    narration = setup["controller"].pending_completion()
    assert narration is not None
    # The fake bridge's finish() doesn't emit FILE_CHANGE events, so
    # the runner sees zero file activity. Post-2026-05-11 honesty
    # fix the narration is "I finished without writing or modifying
    # any files..." when success=True AND no files touched; legacy
    # "Done." still fires when files were actually written. Accept
    # either form.
    assert (
        "Done." in narration
        or "complete" in narration.lower()
        or "without writing or modifying" in narration.lower()
    )
    # Subsequent calls return None (consumed).
    assert setup["controller"].pending_completion() is None


# ---------------------------------------------------------------------------
# A4: pre-task confirmation
# ---------------------------------------------------------------------------


def test_pre_task_confirmation_disabled_dispatches_immediately(setup, monkeypatch):
    """When pre_task_confirmation_enabled is OFF, behaviour is the legacy
    immediate-dispatch path.

    2026-05-26: the production-wiring pass flipped this flag to ON by
    default (config.yaml). The legacy path is now an explicit opt-out;
    the test temporarily disables the flag to exercise it.
    """
    # Force the flag OFF for this scenario via the legacy settings shim
    # (voice.py reads settings.CODING_PRE_TASK_CONFIRMATION_ENABLED).
    from config import settings as _settings  # noqa: E402
    monkeypatch.setattr(
        _settings, "CODING_PRE_TASK_CONFIRMATION_ENABLED", False, raising=False,
    )
    out = setup["controller"].handle_utterance(
        "Create a Python script called sample_one that prints hello."
    )
    assert out is not None and out.handled
    # Legacy path: no deferred dispatch, no confirmation.
    assert out.pre_task_confirmation is None
    assert out.deferred_dispatch is None
    # Bridge got the task synchronously.
    assert setup["bridge"].last_request is not None


def test_pre_task_confirmation_when_enabled_returns_deferred(setup, monkeypatch):
    """Operator opt-in path: with the flag on, _submit must NOT have
    run yet -- the response carries the deferred dispatch closure for
    the orchestrator to run after the barge-in window."""
    monkeypatch.setattr(
        "config.settings.CODING_PRE_TASK_CONFIRMATION_ENABLED", True,
    )
    out = setup["controller"].handle_utterance(
        "Create a Python script called sample_two that prints hello."
    )
    assert out is not None and out.handled
    # Confirmation phrase populated; bridge has NOT been called yet.
    assert out.pre_task_confirmation is not None
    # 2026-05-22 brand sanitization: voice surface refers to the
    # underlying coding subprocess as "AI coding agent" generically
    # (the prior brand-name literal was sanitized in session G/H per
    # the memory index).
    assert "ai coding agent" in out.pre_task_confirmation.lower()
    assert "going ahead" in out.pre_task_confirmation.lower()
    assert "sample_two" in out.pre_task_confirmation.lower()
    assert out.deferred_dispatch is not None
    assert out.pre_task_label == "sample_two"
    # Bridge submission deferred until orchestrator runs deferred_dispatch().
    assert setup["bridge"].last_request is None
    # Run the closure -> bridge fires.
    out.deferred_dispatch()
    assert setup["bridge"].last_request is not None


def test_pre_task_confirmation_existing_project_uses_on_phrase(setup, monkeypatch):
    monkeypatch.setattr(
        "config.settings.CODING_PRE_TASK_CONFIRMATION_ENABLED", True,
    )
    proj_dir = setup["sandbox"] / "calculator"
    proj_dir.mkdir()
    setup["registry"].add(Project(
        name="Calculator",
        path=str(proj_dir),
        aliases=["calc", "calculator project"],
        language="python",
    ))
    out = setup["controller"].handle_utterance(
        "Add a subtract function to my calculator project."
    )
    assert out is not None
    confirmation = out.pre_task_confirmation or ""
    assert "on the calculator project" in confirmation.lower()
    # Should include a verb extraction.
    assert "subtract" in confirmation.lower()


def test_pre_task_confirmation_caps_word_count(setup, monkeypatch):
    monkeypatch.setattr(
        "config.settings.CODING_PRE_TASK_CONFIRMATION_ENABLED", True,
    )
    monkeypatch.setattr(
        "config.settings.CODING_PRE_TASK_MAX_WORDS", 10,
    )
    long_request = (
        "Create a python script called long_one that does many things "
        "with database connections, web scraping, and external APIs."
    )
    out = setup["controller"].handle_utterance(long_request)
    assert out is not None and out.pre_task_confirmation is not None
    # Word cap is on the action phrase, not the wrapper, but the
    # confirmation must still be reasonably short.
    assert len(out.pre_task_confirmation.split()) < 30


def test_pre_task_confirmation_no_change_for_progress_query(setup, monkeypatch):
    """Read-only intents (progress, cancel) must NOT acquire pre-task
    confirmation -- they aren't destructive."""
    monkeypatch.setattr(
        "config.settings.CODING_PRE_TASK_CONFIRMATION_ENABLED", True,
    )
    seed = setup["controller"].handle_utterance(
        "Create a Python script called progress_one that prints hello."
    )
    # Run the deferred dispatch (the orchestrator does this in production).
    assert seed is not None and seed.deferred_dispatch is not None
    seed.deferred_dispatch()
    out = setup["controller"].handle_utterance("How's it going?")
    assert out is not None
    # Progress query carries plain text only; no pre-task confirmation.
    assert out.pre_task_confirmation is None
    assert out.deferred_dispatch is None


def test_pre_task_confirmation_summarise_strips_filler():
    """Direct unit test of the action-phrase summariser."""
    from ultron.coding.voice import CapabilityVoiceController
    summarise = CapabilityVoiceController._summarise_intent_for_voice

    assert summarise(
        intent_text="Can you add a subtract function?", max_words=10,
    ) == "add a subtract function"
    assert summarise(
        intent_text="please refactor the auth module", max_words=10,
    ) == "refactor the auth module"
    assert summarise(intent_text="", max_words=10) == "make the requested change"


def test_record_pre_task_aborted_writes_audit(setup, tmp_path):
    """The runner's audit-log helper writes a structured row."""
    setup["runner"].record_pre_task_aborted(
        label="weather_fetcher",
        reason="barge_in",
        intent_text="Make me a weather thing",
    )
    log_path = tmp_path / "log.jsonl"
    assert log_path.is_file()
    line = log_path.read_text(encoding="utf-8").splitlines()[-1]
    import json

    record = json.loads(line)
    assert record["event"] == "pre_task_aborted"
    assert record["label"] == "weather_fetcher"
    assert record["reason"] == "barge_in"
