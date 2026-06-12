"""Ultron MCP server.

One server hosts both the Qwen-facing supervisor surface (called as
direct Python methods, no transport) and the Claude-facing worker
surface (over Server-Sent Events, so any in-process AI coding agent
subprocess can connect via a generated ``.mcp.json``).

Single shared state. The Qwen supervisor and Claude both read/write
through :class:`SessionStore`, which is lock-protected so the asyncio
thread serving SSE doesn't race with the main thread.

Note on transport choice: the addendum spec mentioned stdio for the
Claude side; we deviate to SSE because it lets the orchestrator process
own the state cleanly. Stdio would require a separate shim subprocess
(spawned by Claude) that bridges back to the orchestrator -- equivalent
plumbing to SSE but with extra moving parts. Both give Claude a proper
MCP client experience.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from config import settings
from ultron.coding.audit import SessionAuditWriter
from ultron.coding.session import (
    ClarificationRequest,
    CompletionClaim,
    FollowupKind,
    ProjectSession,
    SessionMode,
    SessionStatus,
    SessionStore,
)
from ultron.errors import FilesystemError, MCPServerError
from ultron.resilience import get_error_log
from ultron.utils.logging import get_logger

logger = get_logger("coding.mcp_server")


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class _AuditLog:
    """Append-only JSONL log of every MCP call (both surfaces)."""

    def __init__(self, path: Optional[Path]) -> None:
        self.path = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, **fields: Any) -> None:
        if self.path is None:
            return
        record = {"ts": time.time(), **fields}
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            logger.debug("audit log write failed: %s", e)
            get_error_log().record(
                FilesystemError(
                    f"MCP audit-log write failed: {e}",
                    context={"path": str(self.path)},
                    recovery="audit write skipped; system continues",
                ),
                dependency="filesystem",
                include_traceback=False,
            )


# ---------------------------------------------------------------------------
# Pending-clarification registry
# ---------------------------------------------------------------------------


class _PendingRegistry:
    """Per-request asyncio Futures awaiting Qwen's clarification response.

    Created on Claude's ``request_clarification`` call (asyncio thread);
    resolved on the Qwen-side ``respond_to_clarification`` call
    (potentially main thread). Cross-thread future resolution uses
    ``loop.call_soon_threadsafe`` -- which is the *only* safe way to
    poke an asyncio.Future from outside its loop.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._waiters: Dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Future]] = {}

    def register(self, request_id: str, loop: asyncio.AbstractEventLoop) -> asyncio.Future:
        future = loop.create_future()
        with self._lock:
            self._waiters[request_id] = (loop, future)
        return future

    def resolve(self, request_id: str, value: str) -> bool:
        """Resolve from any thread. Returns True if a waiter existed."""
        with self._lock:
            entry = self._waiters.pop(request_id, None)
        if entry is None:
            return False
        loop, future = entry
        loop.call_soon_threadsafe(_set_future_safe, future, value)
        return True

    def cancel(self, request_id: str, error_text: str) -> bool:
        with self._lock:
            entry = self._waiters.pop(request_id, None)
        if entry is None:
            return False
        loop, future = entry
        loop.call_soon_threadsafe(_set_future_exception_safe, future, RuntimeError(error_text))
        return True

    def session_for_request(self, request_id: str) -> Optional[str]:
        # Not needed in Phase 1; reserved for multi-session disambiguation.
        return None


def _set_future_safe(future: asyncio.Future, value: Any) -> None:
    if not future.done():
        future.set_result(value)


def _set_future_exception_safe(future: asyncio.Future, exc: BaseException) -> None:
    if not future.done():
        future.set_exception(exc)


# ---------------------------------------------------------------------------
# UltronMCPServer
# ---------------------------------------------------------------------------


class UltronMCPServer:
    """Owns the FastMCP instance, the session store, and the SSE lifecycle.

    Args:
        host / port / sse_path: where Claude reaches us via SSE.
        log_path: path to the per-call audit log (JSONL).
        active_session_id_provider: a callable that the Claude-side tools
            use to look up which session their call is "for". Phase 1
            keeps things simple: at most one active session, so the
            provider returns it. Phase 2 will route by URL path or
            session-cookie.
    """

    def __init__(
        self,
        *,
        host: str = settings.CODING_MCP_HOST,
        port: int = settings.CODING_MCP_PORT,
        sse_path: str = settings.CODING_MCP_SSE_PATH,
        log_path: Optional[Path] = settings.CODING_MCP_LOG_PATH,
        clarification_timeout_s: float = float(settings.CODING_MCP_CLARIFICATION_TIMEOUT_S),
        session_audit_dir: Optional[Path] = None,
        memory: Optional[Any] = None,
    ) -> None:
        from mcp.server.fastmcp import FastMCP

        self.host = host
        self.port = port
        self.sse_path = sse_path
        self.audit = _AuditLog(log_path)
        self.clarification_timeout_s = clarification_timeout_s
        # Phase 1 (A3 wiring) -- when set, ``lookup_facts`` queries the
        # Qdrant ``facts`` collection via ``memory.search_facts``. None
        # preserves the legacy stub behaviour for tests that bypass the
        # orchestrator.
        self._memory = memory

        # Phase 7: per-session audit writer. Defaults to None so existing
        # tests that bypass the orchestrator don't get filesystem writes;
        # the orchestrator passes settings.CODING_SESSION_AUDIT_DIR.
        self.session_audit = (
            SessionAuditWriter(session_audit_dir)
            if session_audit_dir is not None else None
        )
        self.store = SessionStore(audit_writer=self.session_audit)
        self._pending = _PendingRegistry()

        # The Qwen-facing supervisor (Phase 2) plugs in via this hook.
        # When None, request_clarification falls back to the configured
        # default response policy (Phase 1 stub).
        self._clarification_responder: Optional[
            Callable[[str, ClarificationRequest, ProjectSession], Awaitable[str]]
        ] = None
        # Phase 4: declare_complete handler hook. When set, the
        # coordinator runs verification + drives the correction loop;
        # when None, declare_complete just records the claim and returns
        # a Phase-1 placeholder.
        self._declare_complete_handler: Optional[
            Callable[[str], Awaitable[str]]
        ] = None

        self._mcp = FastMCP(
            name=settings.CODING_MCP_SERVER_NAME,
            host=host,
            port=port,
            sse_path=sse_path,
            log_level="WARNING",
        )
        self._register_tools()

        # Lifecycle.
        self._server_thread: Optional[threading.Thread] = None
        self._uvicorn_server = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Handle to the started-watcher task so the _run finally block
        # can cancel it (and tests can introspect it). Pre-fix, a bind
        # failure left it pending when the loop closed, producing the
        # benign-but-noisy asyncio "Task was destroyed but it is
        # pending!" stderr line at startup.
        self._waiter_task: Optional["asyncio.Task"] = None
        self._started = threading.Event()
        self._stopped = threading.Event()

    # --- Qwen-side tools ----------------------------------------------------

    def create_session(
        self,
        *,
        project_root: Path,
        initial_prompt: str,
        mode: SessionMode = "new",
        model: str = "haiku",
        refined_goal: str = "",
    ) -> ProjectSession:
        """Spec: ``session.create``. Validates inputs and creates the
        :class:`ProjectSession`. Does NOT spawn Claude; the runner does
        that after writing the per-session ``.mcp.json``."""
        project_root = Path(project_root).resolve()
        self._validate_project_root(project_root, mode)
        session = self.store.create(
            project_root=project_root,
            user_intent=initial_prompt,
            mode=mode,
            model=model,
            refined_goal=refined_goal,
        )
        self.audit.write(
            kind="qwen_call", tool="session.create",
            session_id=session.session_id, project_root=str(project_root),
            mode=mode, model=model,
        )
        return session

    def get_session_state(self, session_id: str) -> ProjectSession:
        """Spec: ``session.get_state``.

        DEPRECATED in Phase C / Phase 1. Returns the full ProjectSession,
        which on a long session will overflow Qwen's context budget.
        Use the projection-based tools instead:

          * :meth:`get_status_delta`
          * :meth:`get_clarification_context`
          * :meth:`get_adjustment_context`
          * :meth:`get_correction_context`
          * :meth:`get_completion_context`

        For in-process Python callers (the runner / coordinator) that
        legitimately need the full state, use :meth:`get_full_state`
        instead -- it's the same data but explicitly marked as not for
        MCP/Qwen exposure.

        Will be removed in Phase D.
        """
        import warnings
        warnings.warn(
            "get_session_state is deprecated; use the projection-based "
            "tools (get_status_delta, get_clarification_context, etc.) "
            "or get_full_state for in-process Python callers.",
            DeprecationWarning, stacklevel=2,
        )
        return self.store.get(session_id)

    def get_full_state(self, session_id: str) -> ProjectSession:
        """In-process Python API for the runner / coordinator.

        Returns the full :class:`ProjectSession`. Explicitly NOT exposed
        as an MCP tool -- callers via Qwen must use the projection tools
        which respect token budgets. Internal supervisor code that runs
        in the same process and isn't subject to Qwen's context window
        can use this freely.
        """
        return self.store.get(session_id)

    # --- Phase C / Phase 1: projection-based state queries -----------------
    # These replace get_session_state for any caller subject to Qwen's
    # context budget. Each returns a bounded ProjectionResult; rendering
    # and truncation logic lives in projections.py.

    def get_status_delta(self, session_id: str):
        from ultron.coding.projections import project_status_delta
        return project_status_delta(self.store.get(session_id))

    def get_clarification_context(
        self, session_id: str, clarification_question: str,
        options=None, facts_lookup=None,
    ):
        from ultron.coding.projections import project_clarification_context
        return project_clarification_context(
            self.store.get(session_id),
            clarification_question=clarification_question,
            options=options,
            facts_lookup=facts_lookup,
        )

    def get_adjustment_context(
        self, session_id: str, adjustment_text: str,
        facts_lookup=None, conflict_detector=None,
    ):
        from ultron.coding.projections import project_adjustment_context
        return project_adjustment_context(
            self.store.get(session_id),
            adjustment_text=adjustment_text,
            facts_lookup=facts_lookup,
            conflict_detector=conflict_detector,
        )

    def get_correction_context(
        self, session_id: str, *, failures, failed_test_names=None,
        failed_test_messages: str = "",
    ):
        from ultron.coding.projections import project_correction_context
        return project_correction_context(
            self.store.get(session_id),
            failures=failures,
            failed_test_names=failed_test_names,
            failed_test_messages=failed_test_messages,
        )

    def get_completion_context(self, session_id: str):
        from ultron.coding.projections import project_completion_context
        return project_completion_context(self.store.get(session_id))

    def send_followup(
        self, session_id: str, prompt: str, kind: FollowupKind,
    ) -> None:
        """Spec: ``session.send_followup``. Phase 1 records the intent
        in the audit log and on the session; the actual subprocess
        re-prompt is wired in Phase 2 once the runner supports
        multi-turn ``--resume`` calls."""
        if not prompt or not prompt.strip():
            raise ValueError("send_followup: prompt must be non-empty")
        if kind not in ("clarification_response", "adjustment", "correction"):
            raise ValueError(f"send_followup: invalid kind {kind!r}")
        session = self.store.get(session_id)  # raises if unknown
        if kind == "adjustment":
            self.store.record_adjustment(session_id, prompt)
        self.audit.write(
            kind="qwen_call", tool="session.send_followup",
            session_id=session_id, followup_kind=kind, prompt_chars=len(prompt),
            session_status=session.status.value,
        )

    def terminate_session(self, session_id: str, reason: str) -> None:
        """Spec: ``session.terminate``."""
        try:
            session = self.store.get(session_id)
        except KeyError:
            return
        # Always allowed -- terminated is a universal sink.
        if session.status not in (
            SessionStatus.COMPLETE, SessionStatus.FAILED, SessionStatus.TERMINATED,
        ):
            self.store.transition(session_id, SessionStatus.TERMINATED)
        # Resolve any pending clarification with an error so Claude doesn't hang.
        if session.pending_clarification is not None:
            self._pending.cancel(
                session.pending_clarification.request_id,
                "session terminated",
            )
        self.audit.write(
            kind="qwen_call", tool="session.terminate",
            session_id=session_id, reason=reason,
        )

    def list_active(self) -> List[str]:
        """Spec: ``session.list_active``."""
        return [s.session_id for s in self.store.list_active()]

    def respond_to_clarification(
        self,
        request_id: str,
        answer: str,
        *,
        decision_path: str = "supervisor",
    ) -> bool:
        """Resolve a pending clarification. Called by Phase 2 coordinator
        (or, in Phase 1 tests, directly). Returns True if a waiter was
        actually resolved."""
        if not isinstance(answer, str):
            raise TypeError("respond_to_clarification: answer must be str")
        # Find the session this request belongs to and update its state.
        for session in self.store.list_all():
            if (
                session.pending_clarification
                and session.pending_clarification.request_id == request_id
            ):
                self.store.resolve_clarification(
                    session.session_id, answer, decision_path,
                )
                # Hand control back to executing -- Claude will resume.
                if session.status in (
                    SessionStatus.AWAITING_CLARIFICATION,
                    SessionStatus.AWAITING_USER,
                ):
                    self.store.transition(session.session_id, SessionStatus.EXECUTING)
                break
        resolved = self._pending.resolve(request_id, answer)
        self.audit.write(
            kind="qwen_call", tool="respond_to_clarification",
            request_id=request_id, decision_path=decision_path, resolved=resolved,
        )
        return resolved

    def lookup_facts(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        min_confidence: Optional[float] = None,
        max_age_days: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Spec: ``project.lookup_facts``.

        Returns up to ``k`` matching rows from the Qdrant ``facts``
        collection (populated by ``scripts/maintenance.py``). Each row is
        a plain dict with keys: ``fact``, ``confidence``, ``last_confirmed``,
        ``category``, ``score``, ``extracted_at``, ``extracted_from``,
        ``retrieval_weight`` (so the coordinator's clarification fast-path
        can consume them without importing :class:`FactRow`).

        When no memory is wired (test-isolation path), returns ``[]`` and
        logs an audit entry with ``result_count=0``. Failures inside
        ``memory.search_facts`` are already swallowed there; this method
        never raises.
        """
        if self._memory is None:
            self.audit.write(
                kind="qwen_call", tool="project.lookup_facts",
                query=query[:200], result_count=0, source="no_memory_wired",
            )
            return []
        cfg = settings.CODING_FACTS
        eff_k = k if k is not None else cfg["top_k"]
        eff_min_conf = (
            min_confidence if min_confidence is not None
            else cfg["min_confidence"]
        )
        eff_max_age = (
            max_age_days if max_age_days is not None
            else cfg["max_age_days"]
        )
        try:
            rows = self._memory.search_facts(
                query, k=eff_k, min_confidence=eff_min_conf,
                max_age_days=eff_max_age,
            )
        except Exception as e:
            logger.debug("lookup_facts: search_facts raised %s", e)
            rows = []
        result = [asdict(row) for row in rows]
        self.audit.write(
            kind="qwen_call", tool="project.lookup_facts",
            query=query[:200], result_count=len(result),
            min_confidence=eff_min_conf, max_age_days=eff_max_age,
        )
        return result

    def read_file_tree(self, project_root: Path) -> Dict[str, Any]:
        """Spec: ``project.read_file_tree``. Returns a flat list of files
        with sizes + mtimes; recursive but skips noisy directories."""
        root = Path(project_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(str(root))
        skip = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache"}
        entries: List[Dict[str, Any]] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(p.name in skip for p in path.parents):
                continue
            try:
                stat = path.stat()
                entries.append({
                    "path": str(path.relative_to(root)),
                    "size_bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                })
            except OSError:
                continue
        self.audit.write(
            kind="qwen_call", tool="project.read_file_tree",
            project_root=str(root), file_count=len(entries),
        )
        return {"root": str(root), "files": entries}

    def set_clarification_responder(
        self,
        responder: Optional[Callable[
            [str, ClarificationRequest, ProjectSession], Awaitable[str]
        ]],
    ) -> None:
        """Phase-2 hook: install the supervisor's responder. When set,
        :meth:`request_clarification` calls it instead of just blocking
        on the registry. The responder is async and must eventually
        either return a string answer or raise."""
        self._clarification_responder = responder

    def set_declare_complete_handler(
        self,
        handler: Optional[Callable[[str], Awaitable[str]]],
    ) -> None:
        """Phase-4 hook: install the coordinator's declare_complete
        handler. When set, the MCP ``declare_complete`` tool delegates
        to the coordinator (which runs verification + drives the
        correction loop). When None, the tool returns the Phase 1
        placeholder."""
        self._declare_complete_handler = handler

    # --- helpers ------------------------------------------------------------

    def _validate_project_root(self, project_root: Path, mode: SessionMode) -> None:
        sandbox = Path(settings.CODING_SANDBOX_PATH).resolve()
        # In production the project must live under the sandbox root. Tests
        # often point at tmp_path -- we relax the check there by allowing
        # any directory if ULTRON_CODING_MCP_ALLOW_ANY_ROOT=1.
        import os
        relax = os.environ.get("ULTRON_CODING_MCP_ALLOW_ANY_ROOT") == "1"
        if not relax:
            try:
                project_root.relative_to(sandbox)
            except ValueError:
                raise ValueError(
                    f"project_root must be inside the sandbox "
                    f"({sandbox}); got {project_root}"
                )
        if mode == "new":
            # New sessions point at a path that should be empty or fresh.
            # We don't strictly require non-existence (the runner may
            # have just mkdir'd it), but we DO require it not be a non-
            # directory if it exists.
            if project_root.exists() and not project_root.is_dir():
                raise ValueError(f"project_root exists but is not a directory: {project_root}")
        else:
            if not project_root.is_dir():
                raise FileNotFoundError(
                    f"project_root must exist for mode={mode!r}: {project_root}"
                )

    # --- Claude-side tools (registered with FastMCP) -----------------------

    def _register_tools(self) -> None:
        from mcp.server.fastmcp import Context  # noqa: F401  (typing in handler signatures)

        server = self  # capture in closures

        @self._mcp.tool(
            description=(
                "Report a meaningful unit of work completed. Call this after "
                "you finish a stage like 'scaffolding', 'implementing auth', "
                "'writing tests for X', 'fixing test failures', etc."
            )
        )
        async def report_progress(stage: str, summary: str, files_touched: List[str]) -> str:
            session = server._claude_active_session()
            stage = (stage or "").strip() or "unspecified"
            summary = (summary or "").strip() or "(no summary)"
            files_touched = [str(f) for f in (files_touched or [])]
            server.store.record_stage(
                session.session_id,
                stage=stage, summary=summary, files_touched=files_touched,
            )
            server.audit.write(
                kind="claude_call", tool="report_progress",
                session_id=session.session_id, stage=stage,
                file_count=len(files_touched),
            )
            return "ok"

        @self._mcp.tool(
            description=(
                "Ask the supervisor (Ultron) for information you need to "
                "proceed. The supervisor may answer from stored context or "
                "escalate to the user. Set urgency='preference' if you have "
                "a sensible default and just want input; 'blocking' if you "
                "cannot proceed without an answer. options is a list of "
                "explicit choices to pick from, if applicable."
            )
        )
        async def request_clarification(
            question: str,
            options: Optional[List[str]] = None,
            urgency: str = "blocking",
        ) -> str:
            session = server._claude_active_session()
            if urgency not in ("blocking", "preference"):
                urgency = "blocking"
            request_id = uuid.uuid4().hex
            request = ClarificationRequest(
                request_id=request_id,
                question=(question or "").strip(),
                options=[str(o) for o in (options or [])],
                urgency=urgency,  # type: ignore[arg-type]
            )
            server.store.set_pending_clarification(session.session_id, request)
            try:
                server.store.transition(
                    session.session_id, SessionStatus.AWAITING_CLARIFICATION,
                )
            except Exception as e:
                logger.debug("clarification transition skipped: %s", e)

            server.audit.write(
                kind="claude_call", tool="request_clarification",
                session_id=session.session_id, request_id=request_id,
                urgency=urgency, question=request.question[:200],
                options=request.options,
            )

            loop = asyncio.get_running_loop()
            future = server._pending.register(request_id, loop)

            # Phase-2 hook: if a supervisor responder is installed, run it
            # concurrently with the timeout. It can resolve the future via
            # respond_to_clarification or by returning a string directly.
            responder = server._clarification_responder
            if responder is not None:
                async def _drive_responder() -> None:
                    try:
                        answer = await responder(session.session_id, request, session)
                    except Exception as e:
                        server._pending.cancel(request_id, f"responder error: {e}")
                        return
                    if answer is not None and not future.done():
                        server.respond_to_clarification(
                            request_id, str(answer), decision_path="supervisor_async",
                        )
                asyncio.create_task(_drive_responder())

            try:
                answer = await asyncio.wait_for(
                    future, timeout=server.clarification_timeout_s,
                )
            except asyncio.TimeoutError:
                server._pending.resolve(request_id, "use your default")
                logger.warning(
                    "clarification timed out for request %s; returning default",
                    request_id,
                )
                answer = "use your default"
            return answer

        @self._mcp.tool(
            description=(
                "Declare the project complete. The supervisor will run "
                "verification before accepting completion. Provide the "
                "user-facing summary, the entry point file (if applicable), "
                "and the run command. List all files created and modified."
            )
        )
        async def declare_complete(
            summary: str,
            entry_point: Optional[str] = None,
            run_command: Optional[str] = None,
            files_created: Optional[List[str]] = None,
            files_modified: Optional[List[str]] = None,
        ) -> str:
            session = server._claude_active_session()
            claim = CompletionClaim(
                summary=(summary or "").strip(),
                entry_point=(entry_point or None),
                run_command=(run_command or None),
                files_created=[str(f) for f in (files_created or [])],
                files_modified=[str(f) for f in (files_modified or [])],
            )
            server.store.record_completion_claim(session.session_id, claim)
            try:
                server.store.transition(session.session_id, SessionStatus.VERIFYING)
            except Exception as e:
                logger.debug("declare_complete transition skipped: %s", e)
            server.audit.write(
                kind="claude_call", tool="declare_complete",
                session_id=session.session_id,
                files_created=len(claim.files_created),
                files_modified=len(claim.files_modified),
                summary_chars=len(claim.summary),
            )
            # Phase 4: hand off to the coordinator's verification +
            # correction loop. When no handler is wired, fall back to
            # the Phase 1 placeholder so the protocol still works.
            handler = server._declare_complete_handler
            if handler is None:
                return "claim recorded; verification pending"
            try:
                return await handler(session.session_id)
            except Exception as e:
                logger.warning(
                    "declare_complete handler raised for %s: %s",
                    session.session_id, e,
                )
                return (
                    f"claim recorded; verification handler errored ({e}). "
                    f"The supervisor will surface this to the user."
                )

        @self._mcp.tool(
            description=(
                "Report test-suite results. Call after running tests; the "
                "supervisor uses the counts and details to decide whether "
                "to accept declare_complete."
            )
        )
        async def report_test_results(
            passing: int, failing: int, skipped: int = 0, details: str = "",
        ) -> str:
            session = server._claude_active_session()
            server.store.record_test_results(
                session.session_id,
                passing=int(passing), failing=int(failing),
                skipped=int(skipped), details=str(details or ""),
            )
            server.audit.write(
                kind="claude_call", tool="report_test_results",
                session_id=session.session_id,
                passing=int(passing), failing=int(failing), skipped=int(skipped),
            )
            return "ok"

    def _claude_active_session(self) -> ProjectSession:
        """Phase 1 single-session lookup. Phase 2 will route by SSE
        connection / URL path so multi-session is supported."""
        active = self.store.list_active()
        if not active:
            err = MCPServerError(
                "Claude called an MCP tool but no session is active. "
                "Did the runner forget to call create_session?",
                context={"active_session_count": 0},
                recovery="MCP call rejected; runner should retry with a session",
            )
            get_error_log().record(err, dependency="mcp_server")
            raise err
        # Pick the most recent active session.
        return max(active, key=lambda s: s.started_at)

    # --- lifecycle: SSE server ---------------------------------------------

    def is_running(self) -> bool:
        """True once the SSE server thread has started and bound its port.

        Used by the voice controller to decide whether to write a
        per-project ``.mcp.json`` for a dispatched coding task.
        """
        return bool(
            self._server_thread is not None
            and self._server_thread.is_alive()
            and self._started.is_set()
        )

    def start(self, *, ready_timeout_s: float = 5.0) -> None:
        """Spin up the SSE server on a background thread. Returns once
        the server reports it's accepting connections, or raises if the
        socket fails to bind."""
        if self._server_thread is not None:
            return  # idempotent

        self._started.clear()
        self._stopped.clear()
        bind_error: List[BaseException] = []

        def _run() -> None:
            import uvicorn
            try:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                app = self._mcp.sse_app()
                config = uvicorn.Config(
                    app, host=self.host, port=self.port,
                    log_level="warning", lifespan="off",
                )
                self._uvicorn_server = uvicorn.Server(config)
                # Mark started once uvicorn has accepted the listening socket.
                async def _wait_for_started() -> None:
                    deadline = time.monotonic() + ready_timeout_s
                    while not self._uvicorn_server.started:
                        if time.monotonic() > deadline:
                            return
                        await asyncio.sleep(0.05)
                    self._started.set()
                self._waiter_task = self._loop.create_task(
                    _wait_for_started()
                )
                self._loop.run_until_complete(self._uvicorn_server.serve())
            except BaseException as e:  # noqa: BLE001 -- propagate to start()
                bind_error.append(e)
                self._started.set()
            finally:
                self._stopped.set()
                # Snapshot: stop() can null self._loop concurrently
                # when its join times out mid-cleanup.
                loop = self._loop
                if loop is not None and not loop.is_closed():
                    # 2026-06-12: cancel + await any pending tasks
                    # BEFORE closing the loop. When serve() raises
                    # before the listening socket binds (port already
                    # taken: SystemExit via uvicorn), _wait_for_started
                    # is still pending; closing the loop around it made
                    # asyncio's Task.__del__ emit "Task was destroyed
                    # but it is pending!" to stderr. Sweeps uvicorn-
                    # internal stragglers too. Happy path is unchanged
                    # (the waiter completes; pending is empty).
                    try:
                        pending = [
                            t for t in asyncio.all_tasks(loop)
                            if not t.done()
                        ]
                        for t in pending:
                            t.cancel()
                        if pending:
                            loop.run_until_complete(
                                asyncio.gather(
                                    *pending, return_exceptions=True,
                                )
                            )
                    except BaseException:  # noqa: BLE001
                        # Fail-open: cleanup must never mask the
                        # original bind error (which itself travels as
                        # a BaseException -- uvicorn raises SystemExit).
                        pass
                    try:
                        loop.close()
                    except Exception:  # noqa: BLE001
                        pass

        self._server_thread = threading.Thread(
            target=_run, daemon=True, name="ultron-mcp-server",
        )
        self._server_thread.start()
        if not self._started.wait(timeout=ready_timeout_s):
            err = MCPServerError(
                f"MCP server failed to start within {ready_timeout_s}s",
                context={
                    "host": self.host,
                    "port": self.port,
                    "sse_path": self.sse_path,
                    "ready_timeout_s": ready_timeout_s,
                },
                recovery="coding tasks unavailable until MCP server starts",
            )
            get_error_log().record(err, dependency="mcp_server")
            raise err
        if bind_error:
            original = bind_error[0]
            err = MCPServerError(
                f"MCP server bind failed: {original}",
                context={
                    "host": self.host,
                    "port": self.port,
                    "sse_path": self.sse_path,
                    "underlying": type(original).__name__,
                },
                recovery="coding tasks unavailable until bind succeeds",
            )
            get_error_log().record(err, dependency="mcp_server")
            raise err from original
        logger.info(
            "Ultron MCP server listening on http://%s:%d%s",
            self.host, self.port, self.sse_path,
        )

    def stop(self, *, timeout_s: float = 5.0) -> None:
        if self._uvicorn_server is None:
            return
        self._uvicorn_server.should_exit = True
        if self._server_thread is not None:
            self._server_thread.join(timeout=timeout_s)
        self._server_thread = None
        self._uvicorn_server = None
        self._loop = None

    def is_running(self) -> bool:
        return (
            self._server_thread is not None
            and self._server_thread.is_alive()
            and not self._stopped.is_set()
        )

    @property
    def sse_url(self) -> str:
        return f"http://{self.host}:{self.port}{self.sse_path}"


# ---------------------------------------------------------------------------
# .mcp.json writer (used by the runner once it spawns a session)
# ---------------------------------------------------------------------------


def write_mcp_config(project_root: Path, sse_url: str) -> Path:
    """Write a per-session ``.mcp.json`` so AI coding agent, when invoked with
    ``cwd=project_root``, automatically connects to our running server.
    Returns the path written."""
    project_root = Path(project_root)
    project_root.mkdir(parents=True, exist_ok=True)
    config = {
        "mcpServers": {
            settings.CODING_MCP_SERVER_NAME: {
                "type": "sse",
                "url": sse_url,
            },
        },
    }
    target = project_root / ".mcp.json"
    target.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return target


def remove_mcp_config(project_root: Path) -> None:
    """Best-effort cleanup of a previously-written ``.mcp.json``."""
    target = Path(project_root) / ".mcp.json"
    if target.is_file():
        try:
            target.unlink()
        except OSError:
            pass
