"""Abstract bridge to a AI coding agent execution backend.

Ships with one concrete implementation -- :class:`DirectClaudeCodeBridge`,
a local subprocess wrapper. The abstraction is preserved for future
backend variants (test mocks, alternate subprocess wrappers) but
**OpenClaw is NOT a bridge alternative** — under the new architecture
(Foundation Part 5) OpenClaw is a peer dispatcher reachable via
``ultron.openclaw_routing``, NOT a swap-in for the coding bridge.

Anything in :mod:`ultron.coding.runner` and the orchestrator's voice
glue depends on the abstract :class:`CodingBridge` and the standardized
:class:`TaskEvent` vocabulary, NOT on subprocess, NOT on
``stream-json``, NOT on any wire format. The bridge implementation
translates whatever its backend produces into ``TaskEvent`` instances.

Event vocabulary (kept deliberately small so it's easy to translate
subprocess output into):

  * ``status``      -- coarse stage: "starting", "running", "finishing"
  * ``text``        -- assistant text delta (incremental)
  * ``tool_use``    -- assistant about to use a tool (Edit / Write / Bash / ...)
  * ``tool_result`` -- a tool finished; payload includes success
  * ``file_change`` -- a file in the project was created / modified / deleted
                      (deduced from tool_result OR a directory snapshot diff)
  * ``error``       -- a fatal-to-the-task error
  * ``complete``    -- the task finished; ``summary`` carries the model's
                      final text
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional


# ---------------------------------------------------------------------------
# Event vocabulary
# ---------------------------------------------------------------------------


class EventKind(str, Enum):
    STATUS = "status"
    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    FILE_CHANGE = "file_change"
    ERROR = "error"
    COMPLETE = "complete"
    USAGE = "usage"  # Phase 7: token usage delta from Claude's stream-json


class FileChangeKind(str, Enum):
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class TaskEvent:
    """A single event from the bridge.

    ``kind`` discriminates the shape of the rest of the fields. Optional
    fields are documented per kind below; treat anything else as None
    for that event.

    kind=STATUS:
      stage -- "starting" | "running" | "finishing"

    kind=TEXT:
      text -- delta of assistant-emitted text

    kind=TOOL_USE:
      tool_name, tool_input

    kind=TOOL_RESULT:
      tool_name, tool_success, tool_brief

    kind=FILE_CHANGE:
      file_path, file_change_kind

    kind=ERROR:
      error

    kind=COMPLETE:
      summary, exit_status, files_created, files_modified, duration_s
    """

    kind: EventKind
    timestamp: float = field(default_factory=time.time)
    # Per-kind optional fields:
    stage: Optional[str] = None
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_success: Optional[bool] = None
    tool_brief: Optional[str] = None
    file_path: Optional[Path] = None
    file_change_kind: Optional[FileChangeKind] = None
    error: Optional[str] = None
    summary: Optional[str] = None
    exit_status: Optional[int] = None
    files_created: Optional[List[Path]] = None
    files_modified: Optional[List[Path]] = None
    duration_s: Optional[float] = None
    # Phase 7: token usage breakdown (set on EventKind.USAGE events).
    # Cache reads are reported separately because they don't count toward
    # the per-session budget but are useful for retrospective analysis.
    usage_input: Optional[int] = None
    usage_output: Optional[int] = None
    usage_cache_creation: Optional[int] = None
    usage_cache_read: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None  # backend-specific payload, for debug logs


# ---------------------------------------------------------------------------
# Request / result / state
# ---------------------------------------------------------------------------


@dataclass
class TaskRequest:
    """What we ask the bridge to do.

    All paths must be absolute. The bridge enforces ``cwd`` as the project
    root: implementations MUST refuse to run if a backend would let the
    underlying tool escape this directory.

    Multi-turn sessions (Phase 2): when ``claude_session_id`` is set the
    bridge resumes that Claude conversation via ``--resume``. When it's
    None the bridge generates a fresh UUID and starts a new session via
    ``--session-id``. Either way ``claude_session_id`` is reflected on
    the resulting :class:`TaskHandle` so the caller can resume it on the
    next prompt.
    """

    task_prompt: str
    cwd: Path
    model: str = "haiku"
    # Tools AI coding agent is allowed to invoke. ``None`` means the backend's
    # default. Pass an explicit list to lock down a sandbox run.
    allowed_tools: Optional[List[str]] = None
    disallowed_tools: Optional[List[str]] = None
    # Skip permission prompts (hard requirement for non-interactive
    # voice-driven runs). Each bridge documents what this maps to.
    skip_permissions: bool = True
    # Hard timeout. None -> no timeout (bridge-default).
    timeout_s: Optional[float] = None
    # Optional metadata copied verbatim into TaskState for voice queries.
    label: Optional[str] = None
    # If set, prepend a discipline preamble to the prompt that tells the
    # model to write tests + run them. Default True.
    require_testing: bool = True
    # Claude conversation session id. None = start fresh; non-None = resume.
    claude_session_id: Optional[str] = None
    # Path to a per-session .mcp.json (Phase 1 wired the file write/cleanup;
    # this lets the bridge pass --mcp-config explicitly so Claude doesn't
    # depend on auto-discovery).
    mcp_config_path: Optional[Path] = None


@dataclass
class TaskResult:
    """Final state once a task completes (success or failure)."""

    success: bool
    exit_status: int
    summary: str  # final assistant text, trimmed
    duration_s: float
    files_created: List[Path] = field(default_factory=list)
    files_modified: List[Path] = field(default_factory=list)
    files_deleted: List[Path] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class TaskState:
    """Snapshot of the running (or completed) task.

    The voice layer reads this whenever the user asks "how's it going?".
    Updated by the runner from the event stream; lock-protected so the
    voice query thread can read concurrently with the bridge thread.
    """

    label: str
    task_prompt: str
    cwd: Path
    started_at: float
    is_complete: bool = False
    is_cancelled: bool = False
    success: Optional[bool] = None
    current_step: str = "starting"
    completed_steps: List[str] = field(default_factory=list)
    last_tool_use: Optional[str] = None
    last_text_snippet: str = ""  # last 200 chars of streamed assistant text
    files_created: List[Path] = field(default_factory=list)
    files_modified: List[Path] = field(default_factory=list)
    files_deleted: List[Path] = field(default_factory=list)
    error: Optional[str] = None
    final_summary: Optional[str] = None
    # Counters used to estimate progress (heuristic).
    tool_use_count: int = 0
    text_chars_emitted: int = 0
    duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Abstract bridge + handle
# ---------------------------------------------------------------------------


EventListener = Callable[[TaskEvent], None]


class TaskHandle(ABC):
    """A live task. Thread-safe enough that the voice layer can read
    ``state()`` while the bridge thread emits events.
    """

    @abstractmethod
    def task_id(self) -> str:
        """Stable id for this task within the running process."""

    @abstractmethod
    def state(self) -> TaskState:
        """Return a snapshot of the current state. Cheap; can be called
        repeatedly from any thread."""

    @abstractmethod
    def add_listener(self, listener: EventListener) -> None:
        """Attach an event listener. Called from the bridge thread; the
        listener should be fast or schedule work elsewhere."""

    @abstractmethod
    def cancel(self) -> None:
        """Best-effort interrupt. Bridges document their cancellation
        semantics; voice barge-in should call this."""

    @abstractmethod
    def wait(self, timeout: Optional[float] = None) -> TaskResult:
        """Block until the task completes (or timeout). Raises
        :class:`TimeoutError` on timeout; otherwise returns the result."""

    @abstractmethod
    def is_running(self) -> bool:
        ...


class CodingBridge(ABC):
    """Abstract bridge: 'submit a task, get back a handle'.

    Concrete bridges:
      * :class:`ultron.coding.direct_bridge.DirectClaudeCodeBridge`
      * Test mocks (e.g. ``ScriptedClaudeBridge`` in tests/coding/mock_bridge.py)
    """

    @abstractmethod
    def submit(self, request: TaskRequest) -> TaskHandle:
        """Kick off a task. Validates the request (cwd existence,
        absolute path, etc.) then hands back a live handle. Implementations
        MUST be safe to call from any thread."""

    @abstractmethod
    def name(self) -> str:
        """Short backend name for logs / status messages, e.g. ``"direct"``."""


# ---------------------------------------------------------------------------
# Discipline preamble (Phase 6 spec: "rigorous testing at every step")
# ---------------------------------------------------------------------------


_QUALITY_PREAMBLE = (
    "Follow Python best practices: add type hints to function signatures "
    "and concise docstrings to public functions, and never use a bare "
    "`except`. If you are creating a new Python project, include a "
    "`pyproject.toml` with the project's metadata and dependencies. Keep "
    "all work inside the project directory you were started in.\n\n"
)


_DISCIPLINE_PREAMBLE = """\
You are working on a self-contained coding task. You MUST:

1. Plan briefly, then implement in small steps.
2. Write tests for each component as you build it. Run the tests. If a
   test fails, fix the underlying code -- do not delete or weaken the
   test.
3. After every meaningful change, re-run the affected tests.
4. Before declaring the task complete, ensure all tests pass.
5. Keep changes inside the project directory you were started in. Do
   not modify files outside it.
6. Be concise in your final summary: list the files you created or
   modified, and confirm the tests pass.

If a tool call fails, attempt to recover by inspecting the error and
adjusting your approach. Do not silently skip steps.

Task:
"""


def render_prompt(request: TaskRequest) -> str:
    """Combine the always-on code-quality preamble + (when requested) the
    testing-discipline preamble with the user's task prompt into the final
    string sent to the backend.

    The code-quality preamble is cheap (~60 tokens) and applies to EVERY
    coding task, including voice-dispatched ones (``require_testing=False``),
    so voice-built programs still follow Python best practices (docstrings,
    type hints, a pyproject.toml for new projects). The heavier
    testing-discipline preamble is added only when ``require_testing`` is set.
    """
    body = request.task_prompt.strip()
    if request.require_testing:
        return _QUALITY_PREAMBLE + _DISCIPLINE_PREAMBLE + body
    return _QUALITY_PREAMBLE + body


# ---------------------------------------------------------------------------
# Helpers shared by bridge implementations
# ---------------------------------------------------------------------------


class _StateMutex:
    """Lock-wrapped state with a small helper for atomic mutation."""

    def __init__(self, state: TaskState) -> None:
        self._state = state
        self._lock = threading.RLock()

    def snapshot(self) -> TaskState:
        with self._lock:
            # Shallow copy via dataclass replace -- list fields share refs,
            # which is fine for read-only consumers.
            from dataclasses import replace
            return replace(
                self._state,
                completed_steps=list(self._state.completed_steps),
                files_created=list(self._state.files_created),
                files_modified=list(self._state.files_modified),
                files_deleted=list(self._state.files_deleted),
            )

    def mutate(self, fn: Callable[[TaskState], None]) -> None:
        with self._lock:
            fn(self._state)


def directory_snapshot(root: Path) -> Dict[Path, float]:
    """Map of ``relative_path -> mtime`` for every file under ``root``.

    Used by direct and OpenClaw bridges to compute a ground-truth diff of
    files created / modified / deleted, independent of whatever event
    stream the backend emits. Symlinks are not followed; hidden files
    (``.git``, ``.venv``) are skipped to keep the snapshot fast.
    """
    out: Dict[Path, float] = {}
    if not root.is_dir():
        return out
    skip_dirs = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache"}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        # Skip if any parent is in the skip list.
        if any(p.name in skip_dirs for p in path.parents):
            continue
        try:
            rel = path.relative_to(root)
            out[rel] = path.stat().st_mtime
        except OSError:
            continue
    return out


def diff_snapshots(
    before: Dict[Path, float], after: Dict[Path, float]
) -> tuple[List[Path], List[Path], List[Path]]:
    """Compute (created, modified, deleted) from two directory snapshots."""
    before_keys = set(before)
    after_keys = set(after)
    created = sorted(after_keys - before_keys)
    deleted = sorted(before_keys - after_keys)
    modified = sorted(
        p for p in (before_keys & after_keys) if before[p] != after[p]
    )
    return created, modified, deleted
