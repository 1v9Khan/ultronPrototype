"""Phase 6: scripted mock bridge that simulates Claude Code in-process.

The real Claude bridge spawns a subprocess that emits events and calls
MCP tools over SSE. For mocked Phase 6 scenarios we replace the
subprocess with a worker thread that:

  * Emits :class:`TaskEvent` instances on the bridge handle (so the
    runner's TaskState is populated).
  * Calls MCP tool implementations directly on the
    :class:`UltronMCPServer` instance (so the coordinator + verifier +
    audit log paths fire exactly as they do in the real flow).

This bypasses the SSE wire protocol -- ``test_mcp_e2e.py`` already
covers the SSE round-trip with a real Claude subprocess. The point of
the in-process simulator is to exercise *orchestration* logic (voice ->
runner -> coordinator -> verifier -> narration) end-to-end without
burning Claude tokens.

Scripts are built fluently::

    script = (
        ClaudeScript()
          .progress("scaffolding", "set up dirs", ["main.py"])
          .write_file("main.py", "print('hi')")
          .test_results(passing=1, failing=0)
          .declare_complete(
              "Hello world script", entry_point="main.py",
              files_created=["main.py"],
          )
    )
    bridge = ScriptedClaudeBridge(server, script)

Each ``.declare_complete()``-terminating script ends the bridge handle
with success; ``.fail()`` ends it with failure. ``.clarify()`` issues a
synchronous request_clarification call and waits for the answer.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

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
from ultron.coding.mcp_server import UltronMCPServer
from ultron.coding.session import (
    ClarificationRequest,
    CompletionClaim,
    SessionStatus,
)
from ultron.utils.logging import get_logger

logger = get_logger("coding.mock_bridge")


# ---------------------------------------------------------------------------
# Script DSL
# ---------------------------------------------------------------------------


@dataclass
class _Step:
    kind: str  # "progress" | "write_file" | "modify_file" | "test_results" |
               # "clarify" | "adjustment_wait" | "declare_complete" | "sleep"
               # | "fail" | "emit_event" | "callback"
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaudeScript:
    """A fluent builder for scripted Claude behavior. Steps execute in order
    on the bridge's worker thread."""
    steps: List[_Step] = field(default_factory=list)

    # --- progress + state ---------------------------------------------------

    def progress(
        self, stage: str, summary: str = "", files_touched: Optional[List[str]] = None,
    ) -> "ClaudeScript":
        self.steps.append(_Step("progress", {
            "stage": stage, "summary": summary,
            "files_touched": list(files_touched or []),
        }))
        return self

    def write_file(self, relpath: str, content: str = "") -> "ClaudeScript":
        self.steps.append(_Step("write_file", {
            "relpath": relpath, "content": content, "kind": "create",
        }))
        return self

    def modify_file(self, relpath: str, content: str = "") -> "ClaudeScript":
        self.steps.append(_Step("write_file", {
            "relpath": relpath, "content": content, "kind": "modify",
        }))
        return self

    def test_results(
        self, passing: int = 0, failing: int = 0, skipped: int = 0, details: str = "",
    ) -> "ClaudeScript":
        self.steps.append(_Step("test_results", {
            "passing": passing, "failing": failing,
            "skipped": skipped, "details": details,
        }))
        return self

    def clarify(
        self,
        question: str,
        options: Optional[List[str]] = None,
        urgency: str = "blocking",
        on_answer: Optional[Callable[[str], None]] = None,
    ) -> "ClaudeScript":
        """Synchronous clarification call. Blocks the script thread until
        the supervisor responds. Optional ``on_answer`` callback receives
        the resolved answer text."""
        self.steps.append(_Step("clarify", {
            "question": question, "options": options or [],
            "urgency": urgency, "on_answer": on_answer,
        }))
        return self

    def declare_complete(
        self,
        summary: str = "task complete",
        entry_point: Optional[str] = None,
        run_command: Optional[str] = None,
        files_created: Optional[List[str]] = None,
        files_modified: Optional[List[str]] = None,
    ) -> "ClaudeScript":
        """Terminal step. Triggers the coordinator's verification +
        correction loop. The script waits for the handler's response;
        if a correction prompt comes back, the script does NOT
        automatically re-run -- that's a separate scenario the test
        author scripts explicitly."""
        self.steps.append(_Step("declare_complete", {
            "summary": summary, "entry_point": entry_point,
            "run_command": run_command,
            "files_created": list(files_created or []),
            "files_modified": list(files_modified or []),
        }))
        return self

    def fail(self, reason: str = "scripted failure") -> "ClaudeScript":
        """Terminal step. Bridge handle ends with success=False."""
        self.steps.append(_Step("fail", {"reason": reason}))
        return self

    def sleep(self, seconds: float) -> "ClaudeScript":
        self.steps.append(_Step("sleep", {"seconds": seconds}))
        return self

    def callback(self, fn: Callable[["_ScriptContext"], None]) -> "ClaudeScript":
        """Run an arbitrary callable with access to the script context.
        Useful for assertions mid-script or for scripted re-runs."""
        self.steps.append(_Step("callback", {"fn": fn}))
        return self


# ---------------------------------------------------------------------------
# Script execution context
# ---------------------------------------------------------------------------


@dataclass
class _ScriptContext:
    """Mutable state passed to script callbacks."""
    server: UltronMCPServer
    session_id: str
    project_root: Path
    handle: "_ScriptedHandle"
    declare_complete_response: Optional[str] = None
    clarifications_received: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bridge handle
# ---------------------------------------------------------------------------


class _ScriptedHandle(TaskHandle):
    """Bridge handle backed by the script worker thread."""

    def __init__(self, request: TaskRequest):
        self._task_id = uuid.uuid4().hex[:12]
        self._request = request
        self._listeners: List[EventListener] = []
        self._listeners_lock = threading.Lock()
        self._state = TaskState(
            label=request.label or "scripted",
            task_prompt=request.task_prompt,
            cwd=request.cwd,
            started_at=time.time(),
        )
        self._done = threading.Event()
        self._cancelled = threading.Event()
        self._result: Optional[TaskResult] = None
        self._claude_session_id: Optional[str] = (
            request.claude_session_id or uuid.uuid4().hex
        )

    @property
    def claude_session_id(self) -> Optional[str]:
        return self._claude_session_id

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
        with self._listeners_lock:
            self._listeners.append(listener)

    def cancel(self) -> None:
        self._cancelled.set()

    def is_running(self) -> bool:
        return not self._done.is_set()

    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def wait(self, timeout: Optional[float] = None) -> Optional[TaskResult]:
        if not self._done.wait(timeout=timeout):
            raise TimeoutError("scripted task did not complete in time")
        return self._result

    # --- internal: emit events / mark done ---------------------------------

    def _emit(self, event: TaskEvent) -> None:
        # Update the running TaskState.
        if event.kind == EventKind.STATUS and event.stage:
            self._state.current_step = event.stage
            self._state.completed_steps.append(event.stage)
        if event.kind == EventKind.FILE_CHANGE and event.file_path is not None:
            if event.file_change_kind == FileChangeKind.CREATED:
                self._state.files_created.append(event.file_path)
            elif event.file_change_kind == FileChangeKind.MODIFIED:
                self._state.files_modified.append(event.file_path)
            elif event.file_change_kind == FileChangeKind.DELETED:
                self._state.files_deleted.append(event.file_path)
        if event.kind == EventKind.TOOL_USE:
            self._state.tool_use_count += 1
        with self._listeners_lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception as e:
                logger.debug("listener error (ignored): %s", e)

    def _finish(
        self, *, success: bool, summary: str = "ok", error: Optional[str] = None,
    ) -> None:
        self._state.is_complete = True
        self._state.success = success
        self._state.duration_s = time.time() - self._state.started_at
        self._state.final_summary = summary
        if error:
            self._state.error = error
        self._result = TaskResult(
            success=success,
            exit_status=0 if success else 1,
            summary=summary,
            duration_s=self._state.duration_s,
            error=error,
            files_created=list(self._state.files_created),
            files_modified=list(self._state.files_modified),
            files_deleted=list(self._state.files_deleted),
        )
        self._done.set()


# ---------------------------------------------------------------------------
# ScriptedClaudeBridge
# ---------------------------------------------------------------------------


class ScriptedClaudeBridge(CodingBridge):
    """Mock CodingBridge that runs a :class:`ClaudeScript` in a worker
    thread, calling MCP tool implementations directly on the
    :class:`UltronMCPServer`.

    Args:
        server: the orchestrator's MCP server (must already be created;
            doesn't need ``start()`` since we don't go through SSE).
        script: the scripted Claude behavior.
        session_id: the session id this script targets. The script's
            MCP calls are recorded against this session.
    """

    def __init__(
        self,
        server: UltronMCPServer,
        script: ClaudeScript,
        *,
        session_id: str,
    ) -> None:
        self.server = server
        self.script = script
        self._session_id = session_id

    def name(self) -> str:
        return "scripted-mock"

    def submit(self, request: TaskRequest) -> TaskHandle:
        handle = _ScriptedHandle(request)
        ctx = _ScriptContext(
            server=self.server,
            session_id=self._session_id,
            project_root=Path(request.cwd),
            handle=handle,
        )
        # Worker thread runs the script.
        t = threading.Thread(
            target=self._run_script, args=(ctx,),
            daemon=True, name=f"mock-claude-{handle.task_id()}",
        )
        t.start()
        return handle

    # --- script driver -----------------------------------------------------

    def _run_script(self, ctx: _ScriptContext) -> None:
        """Walk the script's steps and dispatch each."""
        try:
            for step in self.script.steps:
                if ctx.handle.is_cancelled():
                    ctx.handle._finish(
                        success=False, summary="cancelled by user",
                        error="cancelled",
                    )
                    return
                handler = _DISPATCH.get(step.kind)
                if handler is None:
                    raise RuntimeError(f"unknown script step: {step.kind!r}")
                done = handler(ctx, step.args)
                if done:
                    return
        except Exception as e:
            logger.warning("script error: %s", e)
            ctx.handle._finish(
                success=False, summary=str(e), error=type(e).__name__,
            )
            return
        # If the script falls off the end without an explicit terminal
        # step, finish successfully.
        if not ctx.handle._done.is_set():
            ctx.handle._finish(success=True, summary="script complete")


# ---------------------------------------------------------------------------
# Step dispatch
# ---------------------------------------------------------------------------


def _step_progress(ctx: _ScriptContext, args: Dict[str, Any]) -> bool:
    stage = args["stage"]
    summary = args["summary"]
    files = args["files_touched"]
    # Update the session via the store directly (mirrors what the MCP
    # tool handler does internally).
    ctx.server.store.record_stage(
        ctx.session_id, stage=stage, summary=summary, files_touched=files,
    )
    ctx.handle._emit(TaskEvent(
        kind=EventKind.STATUS, stage=stage, text=summary,
    ))
    return False


def _step_write_file(ctx: _ScriptContext, args: Dict[str, Any]) -> bool:
    relpath = args["relpath"]
    content = args["content"]
    kind_str = args["kind"]
    target = ctx.project_root / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    if kind_str == "create" and target.exists():
        kind_str = "modify"
    target.write_text(content, encoding="utf-8")
    file_kind = (
        FileChangeKind.CREATED if kind_str == "create"
        else FileChangeKind.MODIFIED
    )
    ctx.handle._emit(TaskEvent(
        kind=EventKind.FILE_CHANGE,
        file_path=target,
        file_change_kind=file_kind,
    ))
    return False


def _step_test_results(ctx: _ScriptContext, args: Dict[str, Any]) -> bool:
    ctx.server.store.record_test_results(
        ctx.session_id,
        passing=args["passing"],
        failing=args["failing"],
        skipped=args["skipped"],
        details=args["details"],
    )
    return False


def _step_clarify(ctx: _ScriptContext, args: Dict[str, Any]) -> bool:
    """Mirror the MCP server's request_clarification tool exactly: register
    a pending clarification, drive the responder, block on the answer."""
    request_id = uuid.uuid4().hex
    request = ClarificationRequest(
        request_id=request_id,
        question=args["question"],
        options=list(args["options"]),
        urgency=args["urgency"],
    )
    ctx.server.store.set_pending_clarification(ctx.session_id, request)
    try:
        ctx.server.store.transition(
            ctx.session_id, SessionStatus.AWAITING_CLARIFICATION,
        )
    except Exception:
        pass

    # Drive the responder if one is wired (Phase 2 coordinator).
    answer_holder: Dict[str, Optional[str]] = {"value": None}
    answer_event = threading.Event()

    responder = ctx.server._clarification_responder  # type: ignore[attr-defined]
    if responder is not None:
        # Run the responder on a fresh asyncio loop in this thread.
        async def _drive() -> str:
            session = ctx.server.store.get(ctx.session_id)
            return await responder(ctx.session_id, request, session)

        try:
            answer = asyncio.run(_drive())
        except Exception as e:
            answer = f"responder error: {e}"
        answer_holder["value"] = answer or "use your default"
    else:
        # No responder -> resolve via the registry (test responds manually).
        async def _wait_for_resolution() -> str:
            loop = asyncio.get_running_loop()
            future = ctx.server._pending.register(request_id, loop)  # type: ignore[attr-defined]
            return await asyncio.wait_for(future, timeout=10.0)

        try:
            answer = asyncio.run(_wait_for_resolution())
        except Exception as e:
            answer = f"timeout: {e}"
        answer_holder["value"] = answer

    ctx.clarifications_received.append({
        "request_id": request_id,
        "question": args["question"],
        "answer": answer_holder["value"],
    })

    on_answer = args.get("on_answer")
    if on_answer is not None:
        try:
            on_answer(answer_holder["value"])
        except Exception as e:
            logger.debug("on_answer callback error: %s", e)
    return False


def _step_declare_complete(ctx: _ScriptContext, args: Dict[str, Any]) -> bool:
    """Mirror the MCP server's declare_complete tool."""
    claim = CompletionClaim(
        summary=args["summary"],
        entry_point=args["entry_point"],
        run_command=args["run_command"],
        files_created=list(args["files_created"]),
        files_modified=list(args["files_modified"]),
    )
    ctx.server.store.record_completion_claim(ctx.session_id, claim)
    try:
        ctx.server.store.transition(ctx.session_id, SessionStatus.VERIFYING)
    except Exception:
        pass

    handler = ctx.server._declare_complete_handler  # type: ignore[attr-defined]
    if handler is None:
        ctx.declare_complete_response = "claim recorded; verification pending"
    else:
        async def _drive() -> str:
            return await handler(ctx.session_id)
        try:
            ctx.declare_complete_response = asyncio.run(_drive())
        except Exception as e:
            ctx.declare_complete_response = (
                f"verification handler errored: {e}"
            )

    # Decide bridge result based on the session's terminal status.
    session = ctx.server.store.get(ctx.session_id)
    if session.status == SessionStatus.COMPLETE:
        ctx.handle._finish(
            success=True, summary=args["summary"] or "task complete",
        )
        return True
    if session.status == SessionStatus.FAILED:
        ctx.handle._finish(
            success=False, summary="verification gave up",
            error="verification_failed",
        )
        return True
    # If the session is back in EXECUTING (correction loop), don't end --
    # the script may have additional steps to fix things and re-declare.
    return False


def _step_fail(ctx: _ScriptContext, args: Dict[str, Any]) -> bool:
    ctx.handle._finish(
        success=False, summary=args["reason"], error=args["reason"],
    )
    return True


def _step_sleep(ctx: _ScriptContext, args: Dict[str, Any]) -> bool:
    time.sleep(args["seconds"])
    return False


def _step_callback(ctx: _ScriptContext, args: Dict[str, Any]) -> bool:
    args["fn"](ctx)
    return False


_DISPATCH: Dict[str, Callable[[_ScriptContext, Dict[str, Any]], bool]] = {
    "progress": _step_progress,
    "write_file": _step_write_file,
    "test_results": _step_test_results,
    "clarify": _step_clarify,
    "declare_complete": _step_declare_complete,
    "fail": _step_fail,
    "sleep": _step_sleep,
    "callback": _step_callback,
}
