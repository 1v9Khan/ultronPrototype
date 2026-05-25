"""Shadow-repo tracker — git CLI wrapper with timeout + folder lock.

Each :class:`ShadowRepoTracker` manages a parallel git repository whose
``.git`` directory lives under ``data/checkpoints/<hash>/.git`` and
whose ``core.worktree`` points at the user's workspace. Commits include
the workspace files (subject to the exclusion list); the user's own
``.git`` directory is untouched.

The implementation uses git via :mod:`subprocess` (pure-python git
libraries are heavy and slow); the surface is intentionally minimal —
``init`` / ``commit`` / ``head`` / ``hard_reset`` / ``log`` —
matching the four operations the registry actually calls.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .exclusions import compose_gitignore

LOGGER = logging.getLogger(__name__)

#: Hard timeout (seconds) on the init pass. Cline uses 15 s; ultron
#: matches that since git can take a while on big monorepos.
DEFAULT_INIT_TIMEOUT_SECONDS: float = 15.0

#: Warning threshold (seconds) — log a WARN when init crosses this
#: but is still in progress. Cline uses 7 s.
DEFAULT_INIT_WARNING_SECONDS: float = 7.0

#: Hard timeout (seconds) on a single commit. Generous because a big
#: index can be slow on the first commit.
DEFAULT_COMMIT_TIMEOUT_SECONDS: float = 30.0

#: Hard timeout (seconds) on git status / log / reset operations.
DEFAULT_GIT_OPERATION_TIMEOUT_SECONDS: float = 10.0

# Suppress phantom console windows on Windows subprocess spawns.
_CREATE_NO_WINDOW = (
    getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
)


class CheckpointInitError(RuntimeError):
    """Raised when the shadow repo cannot be initialised."""


@dataclass(frozen=True)
class CheckpointCommit:
    """One commit recorded by :meth:`ShadowRepoTracker.commit`.

    Attributes:
        commit_hash: full git hash of the commit (40-char hex).
        message: commit message (``checkpoint-<cwdhash>-<event_index>``).
        timestamp: monotonic seconds when the commit landed.
        empty: True when the commit had no file changes (``--allow-empty``
            covers this).
        elapsed_seconds: wall-clock duration of the commit pass.
    """

    commit_hash: str
    message: str
    timestamp: float
    empty: bool = False
    elapsed_seconds: float = 0.0


def hash_working_dir(value: str | os.PathLike[str]) -> str:
    """Return the cline-style 12-char SHA hash of ``value``.

    Used as the per-session checkpoint directory name so two sessions
    on different cwds get independent shadow repos.
    """
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return digest[:12]


def _run_git(
    args: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: float,
) -> subprocess.CompletedProcess[bytes]:
    """Invoke git with ``args`` in ``cwd`` returning the completed process.

    Raises:
        FileNotFoundError: when ``git`` is not on PATH.
        subprocess.TimeoutExpired: when the call exceeds ``timeout_seconds``.
    """
    return subprocess.run(  # noqa: S603 - controlled args
        ["git", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
        creationflags=_CREATE_NO_WINDOW,
    )


class ShadowRepoTracker:
    """Per-session shadow-repo tracker.

    Args:
        workspace_path: absolute path to the user's workspace; commits
            include files under this directory (subject to the exclusion
            list).
        repo_root: absolute path to the parent dir where the shadow
            repo will live (typically ``data/checkpoints/<hash>/``).
            The ``.git`` directory is created INSIDE this path; the
            workspace is referenced via ``core.worktree``.
        session_id: caller's session identifier (used in commit
            message labelling).
        gitignore_body: optional pre-composed gitignore content. When
            absent, :func:`ultron.checkpoints.exclusions.compose_gitignore`
            is invoked with defaults.
        init_timeout_seconds / init_warning_seconds / commit_timeout_seconds:
            see module-level defaults.
        clock: optional monotonic clock callable (test hook).

    Notes:
        The tracker enforces a per-session :class:`threading.RLock` so
        concurrent commits serialise without corrupting the index.
        The lock does NOT protect cross-process access — only one
        ultron instance should touch a given checkpoint dir at a time.
    """

    def __init__(
        self,
        *,
        workspace_path: Path,
        repo_root: Path,
        session_id: str,
        gitignore_body: Optional[str] = None,
        init_timeout_seconds: float = DEFAULT_INIT_TIMEOUT_SECONDS,
        init_warning_seconds: float = DEFAULT_INIT_WARNING_SECONDS,
        commit_timeout_seconds: float = DEFAULT_COMMIT_TIMEOUT_SECONDS,
        op_timeout_seconds: float = DEFAULT_GIT_OPERATION_TIMEOUT_SECONDS,
        clock: Optional[object] = None,
    ) -> None:
        self._workspace = Path(workspace_path).resolve()
        self._repo_root = Path(repo_root).resolve()
        self._git_dir = self._repo_root / ".git"
        self._session_id = session_id
        self._workspace_hash = hash_working_dir(self._workspace)
        self._gitignore_body = gitignore_body
        self._init_timeout = max(1.0, float(init_timeout_seconds))
        self._init_warning = max(0.1, float(init_warning_seconds))
        self._commit_timeout = max(1.0, float(commit_timeout_seconds))
        self._op_timeout = max(1.0, float(op_timeout_seconds))
        self._clock = clock or time.monotonic
        self._lock = threading.RLock()
        self._initialised: bool = False
        self._disabled: bool = False
        self._disabled_reason: str = ""
        self._commit_count: int = 0
        self._env = self._build_env()

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    @property
    def initialised(self) -> bool:
        return self._initialised

    @property
    def disabled(self) -> bool:
        return self._disabled

    @property
    def disabled_reason(self) -> str:
        return self._disabled_reason

    @property
    def repo_root(self) -> Path:
        return self._repo_root

    @property
    def git_dir(self) -> Path:
        return self._git_dir

    @property
    def workspace(self) -> Path:
        return self._workspace

    @property
    def session_id(self) -> str:
        return self._session_id

    def initialise(self) -> None:
        """Create + configure the shadow repo (idempotent).

        Raises:
            CheckpointInitError: when init fails or times out.
        """
        with self._lock:
            if self._initialised or self._disabled:
                return
            try:
                self._do_init()
            except CheckpointInitError as exc:
                self._disabled = True
                self._disabled_reason = str(exc)
                raise

    def commit(
        self,
        message: str = "",
        *,
        allow_empty: bool = True,
        extra_message: str = "",
    ) -> Optional[CheckpointCommit]:
        """Stage workspace changes + create a commit on the shadow repo.

        Args:
            message: optional message suffix; the default is
                ``checkpoint-<workspace-hash>-<index>``.
            allow_empty: when True (default), commits even when the
                workspace has no changes — useful for marking event
                boundaries.
            extra_message: optional extra body appended to the commit
                message after a newline. Convention: structured tag
                like ``"event=memory_write turn=42 ts=..."``.

        Returns:
            :class:`CheckpointCommit` describing the new commit, or
            ``None`` when the tracker is disabled.
        """
        with self._lock:
            if self._disabled:
                return None
            if not self._initialised:
                try:
                    self._do_init()
                except CheckpointInitError as exc:
                    LOGGER.warning("shadow repo init failed: %s", exc)
                    self._disabled = True
                    self._disabled_reason = str(exc)
                    return None
            start = self._clock()
            self._commit_count += 1
            label = message or f"checkpoint-{self._workspace_hash}-{self._commit_count}"
            full_message = label
            if extra_message:
                full_message = f"{label}\n\n{extra_message.strip()}"
            # Stage everything (errors swallowed because git emits noise on
            # unreadable files; we still proceed to commit what staged).
            self._run_git(
                ["add", "--", "."],
                timeout=self._commit_timeout,
                allow_nonzero=True,
            )
            add_args = [
                "-c", "user.name=ultron",
                "-c", "user.email=ultron@local.invalid",
                "commit",
                "--no-verify",
                "--no-gpg-sign",
                "-m", full_message,
            ]
            if allow_empty:
                add_args.insert(-2, "--allow-empty")
            completed = self._run_git(
                add_args,
                timeout=self._commit_timeout,
                allow_nonzero=True,
            )
            if completed.returncode != 0:
                stderr = (completed.stderr or b"").decode("utf-8", errors="replace")
                LOGGER.warning(
                    "shadow repo commit failed (rc=%s): %s",
                    completed.returncode,
                    stderr.strip()[:400],
                )
                return None
            # Look up the new commit hash.
            head_completed = self._run_git(
                ["rev-parse", "HEAD"],
                timeout=self._op_timeout,
                allow_nonzero=True,
            )
            commit_hash = (
                (head_completed.stdout or b"").decode("utf-8", errors="replace").strip()
            )
            elapsed = self._clock() - start
            return CheckpointCommit(
                commit_hash=commit_hash,
                message=full_message,
                timestamp=self._clock(),
                empty=False,
                elapsed_seconds=elapsed,
            )

    def head(self) -> Optional[str]:
        """Return the current HEAD hash (or ``None`` when no commit exists)."""
        with self._lock:
            if not self._initialised or self._disabled:
                return None
            completed = self._run_git(
                ["rev-parse", "HEAD"],
                timeout=self._op_timeout,
                allow_nonzero=True,
            )
            if completed.returncode != 0:
                return None
            return (completed.stdout or b"").decode("utf-8", errors="replace").strip()

    def log(self, max_count: int = 50) -> list[CheckpointCommit]:
        """Return the most-recent commits (newest first)."""
        with self._lock:
            if not self._initialised or self._disabled:
                return []
            completed = self._run_git(
                [
                    "log",
                    "--max-count", str(max(1, int(max_count))),
                    "--format=%H%x09%s",
                ],
                timeout=self._op_timeout,
                allow_nonzero=True,
            )
            if completed.returncode != 0:
                return []
            text = (completed.stdout or b"").decode("utf-8", errors="replace")
            commits: list[CheckpointCommit] = []
            for line in text.splitlines():
                if "\t" not in line:
                    continue
                commit_hash, _, message = line.partition("\t")
                commits.append(
                    CheckpointCommit(
                        commit_hash=commit_hash.strip(),
                        message=message.strip(),
                        timestamp=0.0,
                        empty=False,
                        elapsed_seconds=0.0,
                    )
                )
            return commits

    def hard_reset(self, commit_hash: str) -> bool:
        """Reset the workspace to ``commit_hash`` (``git reset --hard``).

        Args:
            commit_hash: target commit hash. Pass an empty string to
                reset to the current HEAD (used for the "undo
                in-progress edit" voice intent).

        Returns:
            True on success, False otherwise.
        """
        with self._lock:
            if not self._initialised or self._disabled:
                return False
            target = commit_hash.strip() or "HEAD"
            completed = self._run_git(
                ["reset", "--hard", target],
                timeout=self._commit_timeout,
                allow_nonzero=True,
            )
            if completed.returncode != 0:
                stderr = (completed.stderr or b"").decode("utf-8", errors="replace")
                LOGGER.warning(
                    "shadow repo hard_reset failed (rc=%s): %s",
                    completed.returncode,
                    stderr.strip()[:400],
                )
                return False
            return True

    def close(self) -> None:
        """Mark the tracker disabled (no-op for git state on disk)."""
        with self._lock:
            self._disabled = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["GIT_DIR"] = str(self._git_dir)
        env["GIT_WORK_TREE"] = str(self._workspace)
        # Avoid the user's global git templates / hooks influencing
        # checkpoint commits.
        env["GIT_TEMPLATE_DIR"] = ""
        env.pop("GIT_TERMINAL_PROMPT", None)
        env["GIT_TERMINAL_PROMPT"] = "0"
        return env

    def _do_init(self) -> None:
        """Initialise the shadow repo on disk."""
        start = self._clock()
        self._repo_root.mkdir(parents=True, exist_ok=True)
        # When invoking ``git init`` we must NOT pre-set GIT_DIR or
        # GIT_WORK_TREE — git rejects work-tree-without-git-dir and the
        # init pass derives both from the working directory.
        init_env = {
            k: v for k, v in self._env.items()
            if k not in ("GIT_DIR", "GIT_WORK_TREE")
        }
        if not self._git_dir.exists():
            try:
                completed = _run_git(
                    ["init", "--initial-branch", "ultron"],
                    cwd=self._repo_root,
                    env=init_env,
                    timeout_seconds=self._init_timeout,
                )
            except FileNotFoundError as exc:
                raise CheckpointInitError("git binary not found on PATH") from exc
            except subprocess.TimeoutExpired as exc:
                raise CheckpointInitError(
                    f"git init timed out after {self._init_timeout:.1f}s",
                ) from exc
            if completed.returncode != 0:
                # Older git versions don't accept --initial-branch.
                fallback = _run_git(
                    ["init"],
                    cwd=self._repo_root,
                    env=init_env,
                    timeout_seconds=self._init_timeout,
                )
                if fallback.returncode != 0:
                    stderr = (fallback.stderr or b"").decode("utf-8", errors="replace")
                    raise CheckpointInitError(
                        f"git init failed: {stderr.strip()[:200]}",
                    )
        # Configure core.worktree -> workspace.
        self._run_git(
            ["config", "core.worktree", str(self._workspace)],
            timeout=self._op_timeout,
            allow_nonzero=False,
        )
        # Disable submodule recursion + force hashed-cache to keep things fast.
        self._run_git(
            ["config", "submodule.recurse", "false"],
            timeout=self._op_timeout,
            allow_nonzero=True,
        )
        # Compose + write the info/exclude file.
        if self._gitignore_body is None:
            self._gitignore_body = compose_gitignore()
        info_dir = self._git_dir / "info"
        info_dir.mkdir(parents=True, exist_ok=True)
        (info_dir / "exclude").write_text(self._gitignore_body, encoding="utf-8")
        elapsed = self._clock() - start
        if elapsed >= self._init_warning:
            LOGGER.warning(
                "shadow repo init took %.1fs (warning threshold %.1fs)",
                elapsed,
                self._init_warning,
            )
        self._initialised = True

    def _run_git(
        self,
        args: Sequence[str],
        *,
        timeout: float,
        allow_nonzero: bool = False,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run a git command inside the shadow repo's environment."""
        try:
            completed = _run_git(
                args,
                cwd=self._repo_root,
                env=self._env,
                timeout_seconds=timeout,
            )
        except FileNotFoundError as exc:
            raise CheckpointInitError("git binary not found on PATH") from exc
        except subprocess.TimeoutExpired:
            LOGGER.warning(
                "git %s timed out after %.1fs", " ".join(args[:3]), timeout,
            )
            # Synthesise a non-zero CompletedProcess so callers can
            # uniformly check returncode.
            return subprocess.CompletedProcess(
                args=list(args), returncode=124, stdout=b"", stderr=b"timeout",
            )
        if completed.returncode != 0 and not allow_nonzero:
            stderr = (completed.stderr or b"").decode("utf-8", errors="replace")
            raise CheckpointInitError(
                f"git {' '.join(args[:3])} failed: {stderr.strip()[:200]}",
            )
        return completed


__all__ = [
    "CheckpointCommit",
    "CheckpointInitError",
    "DEFAULT_COMMIT_TIMEOUT_SECONDS",
    "DEFAULT_GIT_OPERATION_TIMEOUT_SECONDS",
    "DEFAULT_INIT_TIMEOUT_SECONDS",
    "DEFAULT_INIT_WARNING_SECONDS",
    "ShadowRepoTracker",
    "hash_working_dir",
]
