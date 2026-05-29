"""Run / launch a finished sandbox program on voice command.

The voice-controlled coding engineer can build + edit programs; this module
closes the loop by letting the user say "run the calculator" or "launch the
server" and actually executing the program in its sandbox project directory.

Two modes:

* **run** -- execute the program, capture stdout/stderr with a hard timeout,
  and report the outcome back by voice. For short utilities / scripts.
* **launch** -- start the program detached (no capture, no wait) and return
  immediately. For long-running things (a GUI, a dev server).

Safety model (load-bearing):

* **Sandbox confinement** -- the resolved project directory MUST live under the
  configured sandbox root. Anything else is refused outright. This is the hard
  guard; it does not depend on the validator being available.
* **Safety validator** -- every run/launch additionally routes through the
  runtime tool-call validator (``tool_name=coding.sandbox.run`` /
  ``coding.sandbox.launch``) for audit + defence in depth. Validator
  unavailability is fail-open (the confinement check already bounds the blast
  radius); an explicit BLOCK verdict is fail-closed (refused).

The execution + spawn primitives are injectable so the whole module is
hermetically testable without ever launching a real process.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from ultron.utils.logging import get_logger

logger = get_logger("coding.sandbox_runner")

DEFAULT_RUN_TIMEOUT_S = 30.0

# Windows: keep spawned consoles hidden (matches the rest of the codebase).
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

# Entry-point discovery, highest-priority first.
_PYTHON_ENTRY_CANDIDATES = (
    "main.py", "app.py", "__main__.py", "run.py", "manage.py",
    "cli.py", "server.py", "index.py", "start.py",
)
_JS_ENTRY_CANDIDATES = ("index.js", "server.js", "main.js", "app.js")


# ---------------------------------------------------------------------------
# Voice-intent matcher
# ---------------------------------------------------------------------------

_RUN_VERBS = r"run|execute"
_LAUNCH_VERBS = r"launch|start up|fire up|boot up|spin up|open up|start"
# Strip leading determiners from the captured object.
_LEADING_DET = re.compile(r"^(the|my|that|a|an|this)\s+", re.IGNORECASE)
# Trailing program-nouns we drop when deriving the project hint.
_TRAILING_NOUN = re.compile(
    r"\s+(program|project|script|app|application|server|tool|utility|game|gui|code)$",
    re.IGNORECASE,
)
_RUN_RE = re.compile(
    rf"^\s*(?:please\s+|can you\s+|could you\s+)?(?P<verb>{_RUN_VERBS}|{_LAUNCH_VERBS})\s+(?P<obj>.+?)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_LAUNCH_SET = {"launch", "start up", "fire up", "boot up", "spin up", "open up", "start"}


@dataclass(frozen=True)
class RunProgramMatch:
    """A matched run/launch command. ``project_hint`` is the user's reference
    to the project (possibly empty for "run it"); the caller resolves it
    against the project registry / sandbox and falls through on no match."""

    mode: str            # "run" | "launch"
    project_hint: str     # "" means "the most-recent project" ("run it")
    raw_text: str


def match_run_program(text: str) -> Optional[RunProgramMatch]:
    """Strict matcher for "run/launch the <project>" voice commands.

    Deliberately loose on the object (the caller's project resolver is the
    real gate -- an unresolvable hint falls through to normal routing), but
    strict on the verb so ordinary conversation never trips it. Returns
    ``None`` when the utterance is not a run/launch command.
    """
    if not text or not text.strip():
        return None
    m = _RUN_RE.match(text.strip())
    if m is None:
        return None
    verb = m.group("verb").lower()
    mode = "launch" if verb in _LAUNCH_SET else "run"
    obj = (m.group("obj") or "").strip()
    # "run it" / "launch that" -> empty hint (use the most-recent project).
    if obj.lower() in {"it", "that", "that one", "this", "this one"}:
        hint = ""
    else:
        hint = _LEADING_DET.sub("", obj)
        hint = _TRAILING_NOUN.sub("", hint).strip()
    return RunProgramMatch(mode=mode, project_hint=hint, raw_text=text.strip())


# ---------------------------------------------------------------------------
# Entry-point resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntryPoint:
    argv: List[str]      # the command to execute, e.g. [python, "main.py"]
    display: str          # human/TTS-friendly form, e.g. "python main.py"
    entry_path: Optional[Path] = None  # the resolved entry file (for validator paths)


def resolve_entry_point(project_path: Path) -> Optional[EntryPoint]:
    """Best-effort discovery of how to run a sandbox project.

    Priority: a top-level Python entry file -> a Python package with
    ``__main__.py`` (``python -m pkg``) -> a JS entry file (``node file``).
    Returns ``None`` when nothing runnable is found.
    """
    project_path = Path(project_path)
    if not project_path.is_dir():
        return None
    try:
        # 1) Top-level Python entry files.
        for name in _PYTHON_ENTRY_CANDIDATES:
            candidate = project_path / name
            if candidate.is_file():
                return EntryPoint(
                    argv=[sys.executable, str(candidate)],
                    display=f"python {name}",
                    entry_path=candidate,
                )
        # 2) A single Python package dir with __main__.py -> python -m pkg.
        for child in sorted(project_path.iterdir()):
            if child.is_dir() and (child / "__main__.py").is_file():
                return EntryPoint(
                    argv=[sys.executable, "-m", child.name],
                    display=f"python -m {child.name}",
                    entry_path=child / "__main__.py",
                )
        # 3) JS entry files.
        for name in _JS_ENTRY_CANDIDATES:
            candidate = project_path / name
            if candidate.is_file():
                return EntryPoint(
                    argv=["node", str(candidate)],
                    display=f"node {name}",
                    entry_path=candidate,
                )
    except OSError as e:
        logger.debug("resolve_entry_point: walk failed (%s)", e)
        return None
    return None


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    ok: bool
    mode: str = "run"                 # "run" | "launch"
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    launched: bool = False            # True for a successful detached launch
    command: str = ""                  # the display command
    project_name: str = ""
    error: Optional[str] = None        # set on refusal / failure to start


# ---------------------------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------------------------


def _is_within(path: Path, root: Path) -> bool:
    """True iff ``path`` is ``root`` or a descendant of it (resolved)."""
    try:
        p = Path(path).resolve()
        r = Path(root).resolve()
    except OSError:
        return False
    return p == r or r in p.parents


def _validator_blocks(entry: EntryPoint, *, mode: str, user_text: str) -> Optional[str]:
    """Run the safety validator for audit + defence in depth. Returns a reason
    string when the validator BLOCKS, else ``None``. Fail-open: validator
    unavailability returns ``None`` (the sandbox-confinement check is the hard
    guard that has already run)."""
    try:
        from ultron.safety.validator import RuleContext, get_validator
        ctx = RuleContext(
            tool_name=f"coding.sandbox.{mode}",
            arguments={"command": entry.display, "path": str(entry.entry_path or "")},
            capability="coding_sandbox_run",
            paths=(entry.entry_path,) if entry.entry_path else (),
            user_text=user_text,
        )
        verdict = get_validator().check(ctx)
        if not verdict.is_allowed:
            return verdict.reason or "blocked by the safety validator"
        return None
    except Exception as e:                                           # noqa: BLE001
        logger.debug("sandbox run validator skipped (%s)", e)
        return None


# ---------------------------------------------------------------------------
# Run / launch
# ---------------------------------------------------------------------------

# Injection seams for tests: a subprocess.run-like callable + a Popen-like one.
RunFn = Callable[..., subprocess.CompletedProcess]
SpawnFn = Callable[..., object]


def run_program(
    project_path: Path,
    *,
    sandbox_root: Path,
    project_name: str = "",
    timeout_s: float = DEFAULT_RUN_TIMEOUT_S,
    user_text: str = "",
    run_fn: Optional[RunFn] = None,
) -> RunResult:
    """Execute a sandbox program, capturing output, with a hard timeout.

    Refuses any project outside ``sandbox_root`` (hard confinement) and routes
    through the safety validator. Never raises -- failures come back as a
    ``RunResult`` with ``ok=False`` + an ``error`` string.
    """
    project_path = Path(project_path)
    name = project_name or project_path.name
    if not _is_within(project_path, sandbox_root):
        return RunResult(ok=False, mode="run", project_name=name,
                         error="that project is outside the sandbox; I only run sandbox programs.")
    entry = resolve_entry_point(project_path)
    if entry is None:
        return RunResult(ok=False, mode="run", project_name=name,
                         error="I couldn't find an entry point to run in that project.")
    blocked = _validator_blocks(entry, mode="run", user_text=user_text)
    if blocked:
        return RunResult(ok=False, mode="run", project_name=name,
                         command=entry.display, error=f"the safety check blocked it: {blocked}")
    runner = run_fn or subprocess.run
    try:
        proc = runner(
            entry.argv, cwd=str(project_path), capture_output=True, text=True,
            timeout=timeout_s, creationflags=_CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        return RunResult(ok=False, mode="run", project_name=name, command=entry.display,
                         timed_out=True,
                         error=f"it was still running after {int(timeout_s)} seconds, so I stopped it.")
    except Exception as e:                                           # noqa: BLE001
        return RunResult(ok=False, mode="run", project_name=name, command=entry.display,
                         error=f"I couldn't start it: {e}")
    rc = getattr(proc, "returncode", None)
    return RunResult(
        ok=(rc == 0), mode="run", returncode=rc,
        stdout=(getattr(proc, "stdout", "") or ""),
        stderr=(getattr(proc, "stderr", "") or ""),
        command=entry.display, project_name=name,
    )


def launch_program(
    project_path: Path,
    *,
    sandbox_root: Path,
    project_name: str = "",
    user_text: str = "",
    spawn_fn: Optional[SpawnFn] = None,
) -> RunResult:
    """Start a sandbox program detached (no capture, no wait). For GUIs /
    servers the user wants left running. Same confinement + validator gate as
    :func:`run_program`. Never raises."""
    project_path = Path(project_path)
    name = project_name or project_path.name
    if not _is_within(project_path, sandbox_root):
        return RunResult(ok=False, mode="launch", project_name=name,
                         error="that project is outside the sandbox; I only launch sandbox programs.")
    entry = resolve_entry_point(project_path)
    if entry is None:
        return RunResult(ok=False, mode="launch", project_name=name,
                         error="I couldn't find an entry point to launch in that project.")
    blocked = _validator_blocks(entry, mode="launch", user_text=user_text)
    if blocked:
        return RunResult(ok=False, mode="launch", project_name=name,
                         command=entry.display, error=f"the safety check blocked it: {blocked}")
    spawn = spawn_fn or subprocess.Popen
    try:
        spawn(
            entry.argv, cwd=str(project_path),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, creationflags=_CREATE_NO_WINDOW,
        )
    except Exception as e:                                           # noqa: BLE001
        return RunResult(ok=False, mode="launch", project_name=name, command=entry.display,
                         error=f"I couldn't launch it: {e}")
    return RunResult(ok=True, mode="launch", launched=True,
                     command=entry.display, project_name=name)


# ---------------------------------------------------------------------------
# Voice summary
# ---------------------------------------------------------------------------


def _tts_safe_tail(text: str, *, max_chars: int = 200) -> str:
    """Last non-empty line of program output, trimmed for TTS."""
    if not text:
        return ""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    tail = lines[-1]
    return tail[:max_chars] + ("..." if len(tail) > max_chars else "")


def summarize_run_result(result: RunResult) -> str:
    """One short, TTS-safe sentence describing the run/launch outcome."""
    name = result.project_name or "it"
    if result.error:
        return f"I tried to {result.mode} {name}, but {result.error}"
    if result.launched:
        return f"Launched {name}. It's running."
    if result.ok:
        out = _tts_safe_tail(result.stdout)
        if out:
            return f"Ran {name}. It finished cleanly. Output: {out}"
        return f"Ran {name}. It finished cleanly with no output."
    # Non-zero exit.
    err = _tts_safe_tail(result.stderr) or _tts_safe_tail(result.stdout)
    if err:
        return f"Ran {name}. It exited with an error: {err}"
    return f"Ran {name}. It exited with code {result.returncode}."


__all__ = [
    "DEFAULT_RUN_TIMEOUT_S",
    "RunProgramMatch", "match_run_program",
    "EntryPoint", "resolve_entry_point",
    "RunResult", "run_program", "launch_program",
    "summarize_run_result",
]
