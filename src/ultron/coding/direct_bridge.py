"""Direct subprocess bridge to Claude Code.

Spawns ``claude --print --output-format stream-json ...`` as a
subprocess in the project's cwd, parses the JSONL event stream into our
standardized :class:`TaskEvent` vocabulary, and exposes a thread-safe
:class:`TaskHandle` to the runner.

OpenClaw is NOT a coding-bridge alternative under the new architecture
(Foundation Part 5) — it's a peer dispatcher reachable via
``ultron.openclaw_routing``.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings
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
    _StateMutex,
    diff_snapshots,
    directory_snapshot,
    render_prompt,
)
from ultron.errors import AnthropicAPIError, ClaudeCodeError
from ultron.resilience import get_error_log
from ultron.utils.logging import get_logger

logger = get_logger("coding.direct_bridge")


# Substrings in stream-json error payloads / Claude Code stderr that
# indicate the failure originated in the Anthropic API rather than the
# subprocess itself. Matched case-insensitively against error text.
_ANTHROPIC_API_ERROR_SIGNS = (
    "rate_limit",
    "rate limit",
    "overloaded",
    "invalid_api_key",
    "invalid api key",
    "authentication_error",
    "api_error",
    "anthropic",
    "529",
    "529 ",
    "529)",
)


def _looks_like_anthropic_api_error(text: str) -> bool:
    """True if ``text`` smells like an Anthropic API failure surfaced
    by Claude Code (rate-limited / overloaded / auth / etc.)."""
    if not text:
        return False
    low = text.lower()
    return any(sign in low for sign in _ANTHROPIC_API_ERROR_SIGNS)


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class DirectClaudeCodeBridge(CodingBridge):
    """Direct ``subprocess.Popen([claude, ...])`` bridge.

    Args:
        claude_cli: path to the claude executable. If missing we look on
            PATH; if still not found, :meth:`submit` will raise.
        log_path: if set, every JSON event line from the subprocess is
            tee'd to this file (in addition to being parsed) so failures
            can be reproduced.
    """

    def __init__(
        self,
        claude_cli: Optional[str] = None,
        log_path: Optional[Path] = None,
    ) -> None:
        self._claude_cli = self._resolve_cli(claude_cli)
        self._log_path = Path(log_path) if log_path else None
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def name(self) -> str:
        return "direct"

    @staticmethod
    def _resolve_cli(explicit: Optional[str]) -> str:
        candidates: List[str] = []
        if explicit:
            candidates.append(explicit)
        candidates.append(settings.CODING_CLAUDE_CLI)
        candidates.append("claude")
        candidates.append("claude.cmd")
        for c in candidates:
            if not c:
                continue
            if Path(c).is_file():
                return c
            found = shutil.which(c)
            if found:
                return found
        raise FileNotFoundError(
            f"Could not locate the Claude Code CLI. Tried: {candidates}. "
            f"Set ULTRON_CLAUDE_CLI to the absolute path of claude.cmd / claude."
        )

    def submit(self, request: TaskRequest) -> TaskHandle:
        cwd = request.cwd.resolve()
        if not cwd.is_dir():
            raise FileNotFoundError(
                f"Coding task cwd does not exist or is not a directory: {cwd}"
            )

        # Resolve / generate the Claude session id. Round-tripped onto the
        # handle so multi-turn callers can pass it back next time.
        claude_session_id = request.claude_session_id or uuid.uuid4().hex
        is_new_session = request.claude_session_id is None

        argv = self._build_argv(request, cwd, claude_session_id, is_new_session)
        logger.info(
            "Submitting coding task: cwd=%s model=%s session=%s mode=%s argv0=%s",
            cwd, request.model, claude_session_id[:8],
            "new" if is_new_session else "resume",
            argv[0],
        )
        return DirectTaskHandle(
            argv=argv,
            cwd=cwd,
            request=request,
            log_path=self._log_path,
            claude_session_id=claude_session_id,
            is_new_session=is_new_session,
        )

    def _build_argv(
        self,
        request: TaskRequest,
        cwd: Path,
        claude_session_id: str,
        is_new_session: bool,
    ) -> List[str]:
        # Claude Code requires UUID format for --session-id / --resume. We
        # carry an unhyphenated 32-char id internally so the audit log is
        # easy to read; insert hyphens at the CLI boundary.
        cli_session_id = _format_uuid(claude_session_id)
        argv: List[str] = [
            self._claude_cli,
            "--print",
            "--output-format", "stream-json",
            "--include-partial-messages",
            "--include-hook-events",
            "--verbose",  # required for --output-format stream-json with --print
            "--model", request.model,
            "--add-dir", str(cwd),
        ]
        if is_new_session:
            argv.extend(["--session-id", cli_session_id])
        else:
            argv.extend(["--resume", cli_session_id])
        if request.skip_permissions:
            argv.append("--dangerously-skip-permissions")
        if request.allowed_tools:
            argv.append("--allowedTools")
            argv.extend(request.allowed_tools)
        if request.disallowed_tools:
            argv.append("--disallowedTools")
            argv.extend(request.disallowed_tools)
        if request.mcp_config_path is not None:
            argv.extend(["--mcp-config", str(request.mcp_config_path)])
        argv.append(render_prompt(request))
        return argv


def _format_uuid(raw: str) -> str:
    """Accept a 32-char hex string or an already-hyphenated UUID; return
    canonical 8-4-4-4-12 form. Claude Code rejects other shapes."""
    s = raw.replace("-", "")
    if len(s) != 32:
        raise ValueError(f"invalid claude session id (expected 32 hex chars): {raw!r}")
    return f"{s[0:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:32]}"


# ---------------------------------------------------------------------------
# Task handle
# ---------------------------------------------------------------------------


# Tools whose use we treat as "the model just touched a file". Used to
# bump the file-tracking heuristic in real time; the post-run directory
# snapshot remains the source of truth.
_FILE_TOUCHING_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
# Tools we surface in the spoken progress narration. Bash is noisy
# (every command counts) so we only summarize the count.
_NARRATABLE_TOOLS = {
    "Edit", "Write", "MultiEdit", "Read", "Bash", "Grep", "Glob",
    "TodoWrite",
}


class DirectTaskHandle(TaskHandle):
    """One in-flight Claude Code subprocess + its parsed event stream."""

    def __init__(
        self,
        argv: List[str],
        cwd: Path,
        request: TaskRequest,
        log_path: Optional[Path],
        claude_session_id: Optional[str] = None,
        is_new_session: bool = True,
    ) -> None:
        self._task_id = uuid.uuid4().hex[:12]
        self._argv = argv
        self._cwd = cwd
        self._request = request
        self._log_path = log_path
        self.claude_session_id = claude_session_id
        self.is_new_session = is_new_session
        self._listeners: List[EventListener] = []
        self._listeners_lock = threading.Lock()
        self._done = threading.Event()
        self._result: Optional[TaskResult] = None
        self._proc: Optional[subprocess.Popen] = None
        self._started_at = time.time()
        self._before_snapshot = directory_snapshot(cwd)

        state = TaskState(
            label=request.label or f"task-{self._task_id}",
            task_prompt=request.task_prompt,
            cwd=cwd,
            started_at=self._started_at,
        )
        self._state = _StateMutex(state)

        # Reader threads -- one for stdout (event stream), one for stderr.
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._wait_thread: Optional[threading.Thread] = None

        self._launch()

    # --- abstract API -------------------------------------------------------

    def task_id(self) -> str:
        return self._task_id

    def state(self) -> TaskState:
        return self._state.snapshot()

    def add_listener(self, listener: EventListener) -> None:
        with self._listeners_lock:
            self._listeners.append(listener)

    def cancel(self) -> None:
        if self._done.is_set() or self._proc is None:
            return
        logger.info("Cancelling task %s", self._task_id)
        self._state.mutate(lambda s: setattr(s, "is_cancelled", True))
        try:
            if os.name == "nt":
                # Windows: SIGTERM is mapped to TerminateProcess by Python;
                # claude.cmd is a .cmd shim spawning node, so we kill the
                # whole process tree.
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(self._proc.pid)],
                    capture_output=True, check=False,
                )
            else:
                self._proc.send_signal(signal.SIGTERM)
        except Exception as e:
            logger.warning("Cancel failed: %s", e)

    def wait(self, timeout: Optional[float] = None) -> TaskResult:
        if not self._done.wait(timeout=timeout):
            raise TimeoutError(f"Task {self._task_id} timed out after {timeout}s")
        if self._result is None:  # defensive
            raise RuntimeError(f"Task {self._task_id} finished without producing a result")
        return self._result

    def is_running(self) -> bool:
        return not self._done.is_set()

    # --- internals ----------------------------------------------------------

    def _launch(self) -> None:
        try:
            # Inherit env but drop NO_COLOR / FORCE_COLOR -- we want raw JSON.
            env = os.environ.copy()
            env.pop("FORCE_COLOR", None)
            env["NO_COLOR"] = "1"
            self._proc = subprocess.Popen(
                self._argv,
                cwd=str(self._cwd),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,  # line-buffered
            )
        except Exception as e:
            logger.error("Failed to launch claude: %s", e)
            get_error_log().record(
                ClaudeCodeError(
                    f"failed to launch claude subprocess: {e}",
                    context={
                        "task_id": self._task_id,
                        "argv0": self._argv[0] if self._argv else "",
                        "cwd": str(self._cwd),
                        "label": self._request.label or "",
                    },
                    recovery="task aborted before subprocess started; user notified",
                ),
                dependency="claude_code",
            )
            self._finalize(success=False, exit_status=-1, error=str(e), summary="")
            return

        self._emit(TaskEvent(kind=EventKind.STATUS, stage="starting"))

        self._stdout_thread = threading.Thread(
            target=self._read_stdout, daemon=True, name=f"claude-stdout-{self._task_id}",
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, daemon=True, name=f"claude-stderr-{self._task_id}",
        )
        self._wait_thread = threading.Thread(
            target=self._wait_for_exit, daemon=True, name=f"claude-wait-{self._task_id}",
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        self._wait_thread.start()

    def _read_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if self._log_path is not None:
                try:
                    with self._log_path.open("a", encoding="utf-8") as f:
                        f.write(line + "\n")
                except OSError:
                    pass
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Some claude versions emit a non-JSON banner first; ignore.
                logger.debug("Non-JSON stdout: %s", line[:120])
                continue
            self._handle_stream_event(event)

    def _read_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            logger.info("[claude stderr] %s", line)

    def _wait_for_exit(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            exit_status = proc.wait(timeout=self._request.timeout_s)
        except subprocess.TimeoutExpired:
            logger.warning("Task %s exceeded timeout; cancelling", self._task_id)
            get_error_log().record(
                ClaudeCodeError(
                    f"subprocess exceeded {self._request.timeout_s:.0f}s timeout",
                    context={
                        "task_id": self._task_id,
                        "label": self._request.label or "",
                        "timeout_s": self._request.timeout_s,
                    },
                    recovery="cancelled subprocess; task marked failed",
                ),
                dependency="claude_code",
            )
            self.cancel()
            try:
                exit_status = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                exit_status = -2
        # Drain stdout/stderr threads so we don't lose late events.
        if self._stdout_thread is not None:
            self._stdout_thread.join(timeout=2.0)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1.0)

        # Compute file diff against the cwd snapshot for ground truth.
        after_snapshot = directory_snapshot(self._cwd)
        created, modified, deleted = diff_snapshots(
            self._before_snapshot, after_snapshot
        )
        # Emit one synthesized FILE_CHANGE per discovered file -- this is
        # the authoritative list, even if we missed an Edit/Write event.
        existing_created = {p for p in self._state.snapshot().files_created}
        existing_modified = {p for p in self._state.snapshot().files_modified}
        for rel in created:
            if rel not in existing_created:
                self._emit(TaskEvent(
                    kind=EventKind.FILE_CHANGE,
                    file_path=rel,
                    file_change_kind=FileChangeKind.CREATED,
                ))
        for rel in modified:
            if rel not in existing_modified:
                self._emit(TaskEvent(
                    kind=EventKind.FILE_CHANGE,
                    file_path=rel,
                    file_change_kind=FileChangeKind.MODIFIED,
                ))
        for rel in deleted:
            self._emit(TaskEvent(
                kind=EventKind.FILE_CHANGE,
                file_path=rel,
                file_change_kind=FileChangeKind.DELETED,
            ))

        snapshot = self._state.snapshot()
        success = (exit_status == 0) and not snapshot.is_cancelled
        summary = (snapshot.final_summary or snapshot.last_text_snippet or "").strip()
        # Log nonzero-exit subprocess failures to errors.jsonl.
        # Skip cancellation (user-initiated), timeout (-2; already logged
        # above), and stream-json errors (snapshot.error set; logged by
        # the per-event handler).
        should_log_exit = (
            not success
            and not snapshot.is_cancelled
            and exit_status not in (0, -2)
            and snapshot.error is None
        )
        if should_log_exit:
            err_text = snapshot.last_text_snippet or ""
            if _looks_like_anthropic_api_error(err_text):
                get_error_log().record(
                    AnthropicAPIError(
                        f"Anthropic API failure during Claude Code session "
                        f"(exit {exit_status})",
                        context={
                            "task_id": self._task_id,
                            "label": self._request.label or "",
                            "exit_status": exit_status,
                            "snippet": err_text[:200],
                        },
                        recovery="task marked failed; user notified",
                    ),
                    dependency="anthropic_api",
                )
            else:
                get_error_log().record(
                    ClaudeCodeError(
                        f"subprocess exited nonzero ({exit_status})",
                        context={
                            "task_id": self._task_id,
                            "label": self._request.label or "",
                            "exit_status": exit_status,
                        },
                        recovery="task marked failed; user notified",
                    ),
                    dependency="claude_code",
                )
        self._finalize(
            success=success,
            exit_status=exit_status,
            error=snapshot.error,
            summary=summary,
            created=[Path(p) for p in created],
            modified=[Path(p) for p in modified],
            deleted=[Path(p) for p in deleted],
        )

    def _finalize(
        self,
        *,
        success: bool,
        exit_status: int,
        error: Optional[str],
        summary: str,
        created: Optional[List[Path]] = None,
        modified: Optional[List[Path]] = None,
        deleted: Optional[List[Path]] = None,
    ) -> None:
        if self._done.is_set():
            return
        duration = time.time() - self._started_at
        result = TaskResult(
            success=success,
            exit_status=exit_status,
            summary=summary,
            duration_s=duration,
            files_created=created or [],
            files_modified=modified or [],
            files_deleted=deleted or [],
            error=error,
        )
        self._result = result

        def _apply(s: TaskState) -> None:
            s.is_complete = True
            s.success = success
            s.duration_s = duration
            s.current_step = "complete" if success else "failed"
            s.final_summary = summary
            if error and not s.error:
                s.error = error
        self._state.mutate(_apply)

        self._emit(TaskEvent(
            kind=EventKind.COMPLETE,
            summary=summary,
            exit_status=exit_status,
            files_created=result.files_created,
            files_modified=result.files_modified,
            duration_s=duration,
        ))
        self._done.set()

    # --- event translation --------------------------------------------------

    def _handle_stream_event(self, raw: Dict[str, Any]) -> None:
        """Translate one stream-json line into 0+ :class:`TaskEvent` instances."""
        rtype = raw.get("type")

        if rtype == "system":
            # Init / status events; mostly ignored, but we treat the very
            # first one as "running".
            subtype = raw.get("subtype", "")
            if subtype == "init":
                self._emit(TaskEvent(kind=EventKind.STATUS, stage="running"))
            return

        if rtype == "assistant":
            self._handle_assistant(raw)
            return

        if rtype == "user":
            # Tool results come back as user messages with tool_result content.
            self._handle_tool_result(raw)
            return

        if rtype == "result":
            self._handle_result(raw)
            return

        if rtype == "stream_event":
            # Partial-message chunks from --include-partial-messages. Some
            # versions surface text deltas here -- handle them so the live
            # progress narration stays current.
            self._handle_partial(raw)
            return

        # Hooks, errors, anything else: log + raw event for debug.
        if rtype == "error":
            err = str(raw.get("error") or raw.get("message") or "unknown")
            self._state.mutate(lambda s: setattr(s, "error", err))
            self._emit(TaskEvent(kind=EventKind.ERROR, error=err, raw=raw))
            # Pattern-match the error text: if it looks like an Anthropic
            # API failure, log as AnthropicAPIError; otherwise as a
            # generic ClaudeCodeError. Either way the typed entry lands
            # in logs/errors.jsonl for triage.
            if _looks_like_anthropic_api_error(err):
                get_error_log().record(
                    AnthropicAPIError(
                        "Anthropic API error reported by Claude Code stream",
                        context={
                            "task_id": self._task_id,
                            "label": self._request.label or "",
                            "snippet": err[:200],
                        },
                        recovery="task will fail; user notified",
                    ),
                    dependency="anthropic_api",
                )
            else:
                get_error_log().record(
                    ClaudeCodeError(
                        "Claude Code stream-json error event",
                        context={
                            "task_id": self._task_id,
                            "label": self._request.label or "",
                            "snippet": err[:200],
                        },
                        recovery="task will fail; user notified",
                    ),
                    dependency="claude_code",
                )

    def _handle_assistant(self, raw: Dict[str, Any]) -> None:
        message = raw.get("message") or {}
        content = message.get("content") or []
        for block in content:
            btype = block.get("type")
            if btype == "text":
                text = block.get("text") or ""
                if text:
                    self._record_text(text)
            elif btype == "tool_use":
                name = str(block.get("name") or "")
                inp = block.get("input") or {}
                self._record_tool_use(name, inp, raw=raw)
        # Phase 7: forward Claude's per-message usage block to the runner.
        # Claude API usage shape: {"input_tokens": int, "output_tokens": int,
        # "cache_creation_input_tokens": int, "cache_read_input_tokens": int}
        usage = message.get("usage") or {}
        if usage:
            self._emit(TaskEvent(
                kind=EventKind.USAGE,
                usage_input=int(usage.get("input_tokens") or 0),
                usage_output=int(usage.get("output_tokens") or 0),
                usage_cache_creation=int(usage.get("cache_creation_input_tokens") or 0),
                usage_cache_read=int(usage.get("cache_read_input_tokens") or 0),
            ))

    def _handle_tool_result(self, raw: Dict[str, Any]) -> None:
        message = raw.get("message") or {}
        content = message.get("content") or []
        for block in content:
            if block.get("type") != "tool_result":
                continue
            tool_name = str(
                block.get("tool_use_id_name")
                or block.get("name")
                or ""
            )
            is_error = bool(block.get("is_error"))
            brief = ""
            payload = block.get("content")
            if isinstance(payload, list):
                texts = [b.get("text", "") for b in payload if isinstance(b, dict)]
                brief = " ".join(t for t in texts if t)[:200]
            elif isinstance(payload, str):
                brief = payload[:200]
            self._emit(TaskEvent(
                kind=EventKind.TOOL_RESULT,
                tool_name=tool_name,
                tool_success=not is_error,
                tool_brief=brief,
                raw=block,
            ))

    def _handle_result(self, raw: Dict[str, Any]) -> None:
        # Final summary emitted by claude --print just before it exits.
        summary = (
            raw.get("result") or raw.get("text") or raw.get("output") or ""
        )
        if isinstance(summary, dict):
            summary = summary.get("text") or summary.get("content") or ""
        if summary:
            self._state.mutate(lambda s: setattr(s, "final_summary", str(summary)))

    def _handle_partial(self, raw: Dict[str, Any]) -> None:
        ev = raw.get("event") or {}
        etype = ev.get("type")
        if etype == "content_block_delta":
            delta = ev.get("delta") or {}
            if delta.get("type") == "text_delta":
                self._record_text(delta.get("text") or "")
        elif etype == "content_block_start":
            block = ev.get("content_block") or {}
            if block.get("type") == "tool_use":
                self._record_tool_use(
                    str(block.get("name") or ""),
                    block.get("input") or {},
                    raw=raw,
                )

    # --- state mutations ----------------------------------------------------

    def _record_text(self, text: str) -> None:
        if not text:
            return
        def apply(s: TaskState) -> None:
            s.text_chars_emitted += len(text)
            s.last_text_snippet = (s.last_text_snippet + text)[-200:]
        self._state.mutate(apply)
        self._emit(TaskEvent(kind=EventKind.TEXT, text=text))

    def _record_tool_use(self, name: str, inp: Dict[str, Any], raw: Dict[str, Any]) -> None:
        def apply(s: TaskState) -> None:
            s.tool_use_count += 1
            s.last_tool_use = name
            s.current_step = _step_label(name, inp)
            if s.completed_steps and s.completed_steps[-1].startswith(name):
                pass  # de-dup repeated tool invocations
            elif name in _NARRATABLE_TOOLS:
                s.completed_steps.append(_step_label(name, inp))
        self._state.mutate(apply)

        # Record file-touching tools as live FILE_CHANGE events even
        # before the post-run snapshot diff.
        if name in _FILE_TOUCHING_TOOLS:
            file_path = inp.get("file_path") or inp.get("path")
            if isinstance(file_path, str) and file_path:
                try:
                    rel = Path(file_path)
                    if rel.is_absolute():
                        try:
                            rel = rel.relative_to(self._cwd)
                        except ValueError:
                            rel = Path(file_path)
                    kind = (
                        FileChangeKind.CREATED
                        if name == "Write"
                        else FileChangeKind.MODIFIED
                    )

                    def apply_files(s: TaskState) -> None:
                        target = (
                            s.files_created
                            if kind == FileChangeKind.CREATED
                            else s.files_modified
                        )
                        if rel not in target:
                            target.append(rel)
                    self._state.mutate(apply_files)
                    self._emit(TaskEvent(
                        kind=EventKind.FILE_CHANGE,
                        file_path=rel,
                        file_change_kind=kind,
                    ))
                except Exception:
                    pass

        self._emit(TaskEvent(
            kind=EventKind.TOOL_USE,
            tool_name=name,
            tool_input=inp,
            raw=raw,
        ))

    # --- listener fan-out ---------------------------------------------------

    def _emit(self, event: TaskEvent) -> None:
        with self._listeners_lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception as e:
                logger.warning("Listener error on %s: %s", event.kind, e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step_label(tool_name: str, inp: Dict[str, Any]) -> str:
    """Human-readable label for a tool invocation, used in voice progress."""
    if tool_name in {"Edit", "Write", "MultiEdit"}:
        path = inp.get("file_path") or inp.get("path") or "(file)"
        return f"{tool_name.lower()} {Path(path).name}"
    if tool_name == "Bash":
        cmd = (inp.get("command") or "").strip()
        return f"running shell: {cmd[:80]}"
    if tool_name == "Read":
        path = inp.get("file_path") or "(file)"
        return f"reading {Path(path).name}"
    if tool_name == "Grep":
        pat = inp.get("pattern") or ""
        return f"searching for {pat[:40]}"
    if tool_name == "Glob":
        pat = inp.get("pattern") or ""
        return f"listing files {pat[:40]}"
    if tool_name == "TodoWrite":
        return "updating internal plan"
    return tool_name
