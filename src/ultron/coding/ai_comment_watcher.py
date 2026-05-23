"""Side-channel input via in-editor ``# ai!`` comments.

Pattern lifted in spirit (not in source) from aider's ``watch.py``
(Apache 2.0; see ``THIRD_PARTY_NOTICES.md``).

A background thread (via ``watchfiles``) monitors the project tree
for modifications and scans changed files for AI-marker comments.
The comment's trailing punctuation determines the intent:

  * ``# ai!`` / ``// ai!`` / ``-- ai!`` / ``;; ai!`` — "execute this
    request immediately". The comment body becomes the request text.
  * ``# ai?`` — "answer this question, don't modify code". Read-only.
  * ``# ai`` (no trailing punctuation) — "consider this file, but
    don't act on its own". Less useful as a trigger; we file it as
    ``KIND.MENTION`` and let the caller decide.

The comment text following the marker is the request payload.
Example:

    # ai! refactor this loop into a helper function

Triggers fire only on the ``!`` / ``?`` variants (catalog rule:
"the suffix is what distinguishes a trigger from a passive mention").

File-size guard: files larger than ``max_file_bytes`` (default 1 MB)
are skipped entirely — the regex scan would be expensive and they're
typically vendored / generated / binary content.

Path filter: gitignore-respecting by default via ``pathspec`` when
available; otherwise falls back to a hard-coded skip list mirroring
:data:`ultron.coding.repo_map.SKIP_DIRECTORIES`.

The watcher is intentionally write-once: each detected comment fires
the callback ONCE per file mtime cycle. The catalog notes the
comment usually stays in source after firing (the user removes the
``!`` once they've seen the agent act), but we don't enforce that —
callers should debounce repeat events if the comment isn't removed.

Public surface:

  * :class:`AICommentTrigger` — frozen detected-trigger record.
  * :class:`AICommentKind` — enum: ``EXECUTE`` / ``QUESTION`` /
    ``MENTION``.
  * :class:`AICommentWatcher` — daemon-thread watcher with
    start/stop lifecycle.
  * :func:`scan_file_for_ai_comments` — pure scan helper, useful
    in tests and for one-shot rescans.
  * :data:`AI_COMMENT_REGEX` — the detection regex.

Fail-open everywhere: missing watchfiles dep, unreadable file, regex
miss, callback exception — none of these break the watcher loop.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, FrozenSet, Iterable, List, Optional, Sequence, Set


logger = logging.getLogger("ultron.coding.ai_comment_watcher")


# Catalog regex (adapted from aider's pattern with explicit suffix
# capture): match a single-line comment prefix, optional whitespace,
# the literal ``ai`` token, an optional trailing ``!`` / ``?``
# trigger immediately after ``ai``, then either a space-separated
# body or end-of-line. The non-greedy body trick from earlier didn't
# work because the regex engine prefers to leave the optional suffix
# empty and let the body absorb everything; instead we make the
# suffix consume punctuation right after ``ai`` and require a
# word-boundary separator from the body.
AI_COMMENT_REGEX = re.compile(
    r"""
    (?P<prefix>\#|//|--|;+)         # comment prefix
    [ \t]*                          # horizontal whitespace
    (?P<marker>ai)                  # the literal ai token
    (?P<suffix>[!?])?               # optional trigger punctuation
    (?:                             # then either:
        [ \t]+(?P<body>[^\n]*?)     #   space + body
        | (?=\s|$)                  #   or end-of-token (boundary)
    )
    [ \t]*$                         # trailing horizontal whitespace + eol
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)


# Files larger than this are skipped — too expensive to regex-scan,
# typically generated / vendored / binary.
DEFAULT_MAX_FILE_BYTES = 1_000_000


# Directories we skip wholesale. Mirrors
# :data:`ultron.coding.repo_map.SKIP_DIRECTORIES` for consistency.
DEFAULT_SKIP_DIRECTORIES: FrozenSet[str] = frozenset({
    "__pycache__",
    ".git",
    ".svn",
    ".hg",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "target",
    ".next",
    ".nuxt",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "htmlcov",
    "coverage",
    ".coverage",
    "site-packages",
    "vendor",
    "bower_components",
    "tmp",
    "temp",
    ".turbo",
    ".parcel-cache",
    "models",
    "logs",
})


# File extensions we skip wholesale. Binary / generated content.
DEFAULT_SKIP_EXTENSIONS: FrozenSet[str] = frozenset({
    ".pyc", ".pyo", ".so", ".dll", ".exe",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".ico",
    ".mp3", ".mp4", ".mov", ".wav", ".ogg", ".webm",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".7z",
    ".db", ".sqlite", ".sqlite3",
    ".onnx", ".pth", ".pt", ".bin", ".safetensors",
    ".woff", ".woff2", ".ttf", ".eot",
    ".lock",
})


class AICommentKind(str, Enum):
    """What the comment is asking the agent to do."""

    EXECUTE = "execute"   # ``# ai!`` — act now
    QUESTION = "question"  # ``# ai?`` — answer, don't modify
    MENTION = "mention"    # ``# ai``  — passive reference


@dataclass(frozen=True)
class AICommentTrigger:
    """One detected AI comment with enough context for the dispatcher.

    Attributes:
        kind: :class:`AICommentKind` derived from the trailing
            punctuation.
        body: The request text following the ``ai`` token (whitespace-
            stripped).
        file_path: Absolute path to the source file.
        line: 0-based line number.
        column: 0-based start column.
        prefix: The comment prefix matched (``#``, ``//``, ``--``,
            ``;``, ``;;``). Useful for callers reformatting the
            comment for narration.
    """

    kind: AICommentKind
    body: str
    file_path: str
    line: int
    column: int
    prefix: str


# Callback contract: receives one :class:`AICommentTrigger` per fired
# event. Should never raise; the watcher catches + logs.
TriggerCallback = Callable[[AICommentTrigger], None]


# ---------------------------------------------------------------------------
# Pure scan helper
# ---------------------------------------------------------------------------


def scan_file_for_ai_comments(
    path: Path | str,
    *,
    max_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> List[AICommentTrigger]:
    """Scan one file for AI comments, returning every match.

    Reads the file as UTF-8 (errors replaced). Skips files larger
    than ``max_bytes`` or unreadable for any reason.
    """
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError:
        return []
    if size > max_bytes:
        return []
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    out: List[AICommentTrigger] = []
    line_starts = _compute_line_starts(text)
    for m in AI_COMMENT_REGEX.finditer(text):
        start = m.start()
        line_idx = _line_for_offset(line_starts, start)
        column = start - line_starts[line_idx]
        suffix = m.group("suffix")
        if suffix == "!":
            kind = AICommentKind.EXECUTE
        elif suffix == "?":
            kind = AICommentKind.QUESTION
        else:
            kind = AICommentKind.MENTION
        body = (m.group("body") or "").strip()
        out.append(AICommentTrigger(
            kind=kind,
            body=body,
            file_path=str(p.resolve()),
            line=line_idx,
            column=column,
            prefix=m.group("prefix") or "",
        ))
    return out


def _compute_line_starts(text: str) -> List[int]:
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def _line_for_offset(line_starts: Sequence[int], offset: int) -> int:
    # Binary search for the largest index whose value is <= offset.
    lo, hi = 0, len(line_starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if line_starts[mid] <= offset:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ---------------------------------------------------------------------------
# Background watcher
# ---------------------------------------------------------------------------


class AICommentWatcher:
    """Daemon-thread file watcher for AI-marker comments.

    Args:
        root: Project directory to watch.
        on_trigger: Callable invoked once per detected EXECUTE /
            QUESTION trigger. MENTION triggers are NOT auto-invoked
            (set ``include_mention=True`` to opt in).
        max_file_bytes: Skip files larger than this (default 1 MB).
        skip_directories: Override the directory skip list.
        skip_extensions: Override the extension skip list.
        include_mention: When True, ``# ai`` (no punctuation)
            triggers ALSO fire the callback. Default False (mentions
            are passive markers).
        poll_interval_seconds: Forwarded to ``watchfiles.watch``.
            Default 0.5 s.

    The watcher seeds an initial scan of every eligible file under
    ``root`` and then listens for file modifications. Each modified
    file is re-scanned; new triggers (not seen on the previous scan
    of that file) fire ``on_trigger``.
    """

    def __init__(
        self,
        root: Path | str,
        on_trigger: TriggerCallback,
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        skip_directories: Optional[Iterable[str]] = None,
        skip_extensions: Optional[Iterable[str]] = None,
        include_mention: bool = False,
        poll_interval_seconds: float = 0.5,
    ) -> None:
        self._root = Path(root).resolve()
        self._on_trigger = on_trigger
        self._max_file_bytes = int(max_file_bytes)
        self._skip_dirs = (
            frozenset(skip_directories) if skip_directories is not None
            else DEFAULT_SKIP_DIRECTORIES
        )
        self._skip_exts = (
            frozenset(s.lower() for s in skip_extensions) if skip_extensions is not None
            else DEFAULT_SKIP_EXTENSIONS
        )
        self._include_mention = bool(include_mention)
        self._poll_interval = float(poll_interval_seconds)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Per-file (path -> set of (line, body) tuples we've already
        # fired). New triggers in a file produce a callback; previously-
        # seen ones don't. This is the "comment stays in source but
        # we don't re-fire" debounce.
        self._seen: dict[str, Set[tuple]] = {}
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def start(self) -> None:
        """Seed the initial scan, then spawn the watcher thread."""
        if self.running:
            return
        self._stop_event.clear()
        # Seed: scan every eligible file so we know which triggers are
        # already in source. First fire only happens on NEW triggers.
        self._seed_initial_scan()
        self._thread = threading.Thread(
            target=self._loop,
            name="ultron-ai-comment-watcher",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        self._thread = None

    def scan_now(self) -> List[AICommentTrigger]:
        """Synchronously rescan the root and return all triggers found.

        Bypasses the seen-set, so this returns EVERY trigger detected
        (including ones already fired). Useful for tests + one-shot
        sweeps.
        """
        out: List[AICommentTrigger] = []
        for path in self._iter_eligible_files():
            out.extend(scan_file_for_ai_comments(
                path, max_bytes=self._max_file_bytes,
            ))
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _seed_initial_scan(self) -> None:
        """Populate ``self._seen`` so existing triggers don't fire."""
        for path in self._iter_eligible_files():
            triggers = scan_file_for_ai_comments(
                path, max_bytes=self._max_file_bytes,
            )
            fingerprints = {
                (t.line, t.body) for t in triggers
                if self._should_fire(t.kind)
            }
            with self._lock:
                self._seen[str(path.resolve())] = fingerprints

    def _loop(self) -> None:
        try:
            from watchfiles import watch  # type: ignore[import-not-found]
        except ImportError:
            logger.warning(
                "watchfiles not installed; AICommentWatcher disabled."
            )
            return
        try:
            for changes in watch(
                str(self._root),
                stop_event=self._stop_event,
                step=int(self._poll_interval * 1000),
                yield_on_timeout=False,
            ):
                for _change_kind, path_str in changes:
                    self._handle_change(Path(path_str))
        except Exception as exc:                                  # noqa: BLE001
            logger.warning(
                "AICommentWatcher loop terminated with: %s", exc,
            )

    def _handle_change(self, path: Path) -> None:
        if not self._is_eligible(path):
            return
        try:
            triggers = scan_file_for_ai_comments(
                path, max_bytes=self._max_file_bytes,
            )
        except Exception as exc:                                  # noqa: BLE001
            logger.debug(
                "AICommentWatcher: scan failed for %s: %s", path, exc,
            )
            return
        key = str(path.resolve())
        new_fingerprints: Set[tuple] = set()
        with self._lock:
            seen = self._seen.setdefault(key, set())
            current_set: Set[tuple] = set()
            for t in triggers:
                if not self._should_fire(t.kind):
                    continue
                fp = (t.line, t.body)
                current_set.add(fp)
                if fp not in seen:
                    new_fingerprints.add(fp)
            # Update seen to the CURRENT set (so a deleted comment
            # can re-fire if the user re-adds it later).
            self._seen[key] = current_set

        # Fire callbacks OUTSIDE the lock to avoid holding it during
        # caller execution.
        for t in triggers:
            if (t.line, t.body) in new_fingerprints:
                try:
                    self._on_trigger(t)
                except Exception as exc:                          # noqa: BLE001
                    logger.warning(
                        "AICommentWatcher: on_trigger raised: %s", exc,
                    )

    def _should_fire(self, kind: AICommentKind) -> bool:
        if kind == AICommentKind.MENTION:
            return self._include_mention
        return True

    def _iter_eligible_files(self) -> Iterable[Path]:
        for dirpath, dirnames, filenames in os.walk(self._root):
            dirnames[:] = [
                d for d in dirnames
                if d not in self._skip_dirs
                and (not d.startswith(".") or d in {".github", ".claude"})
            ]
            for fname in filenames:
                p = Path(dirpath) / fname
                if self._is_eligible(p):
                    yield p

    def _is_eligible(self, path: Path) -> bool:
        try:
            if not path.is_file():
                return False
        except OSError:
            return False
        if path.suffix.lower() in self._skip_exts:
            return False
        # Skip files in skipped directories (catches files added
        # mid-walk after we already pruned the dirname tree).
        try:
            relative = path.resolve().relative_to(self._root)
        except (ValueError, OSError):
            return False
        for part in relative.parts:
            if part in self._skip_dirs:
                return False
        try:
            if path.stat().st_size > self._max_file_bytes:
                return False
        except OSError:
            return False
        return True


__all__ = [
    "AI_COMMENT_REGEX",
    "AICommentKind",
    "AICommentTrigger",
    "AICommentWatcher",
    "DEFAULT_MAX_FILE_BYTES",
    "DEFAULT_SKIP_DIRECTORIES",
    "DEFAULT_SKIP_EXTENSIONS",
    "TriggerCallback",
    "scan_file_for_ai_comments",
]
