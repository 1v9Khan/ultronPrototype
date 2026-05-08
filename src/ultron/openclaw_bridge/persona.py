"""Persona file loader.

Reads the six OpenClaw workspace persona files
(``IDENTITY.md``, ``SOUL.md``, ``USER.md``, ``AGENTS.md``,
``HEARTBEAT.md``, ``BOOTSTRAP.md``) and composes a system prompt
appropriate for the requested *mode*.

## Modes

Ultron's character is for user-facing channels only. Internal
background jobs (heartbeat, cron, compaction, tool selection,
summarization, RAG retrieval gating, etc.) use a plain
task-focused prompt. This protects the user-facing voice character
from being trained-out by internal traffic, saves context budget on
non-user-facing turns, and improves reliability for tasks where
character-driven hedging or terseness would obscure the answer.

Modes (passed to :meth:`PersonaLoader.get_system_prompt`):

- ``"user_facing"`` (default) — the full Ultron persona. Use for
  voice path, Telegram, any channel where the user reads or hears
  the response. Composition: IDENTITY + SOUL + USER + AGENTS.
- ``"background"`` — plain operating rules, no character. Use for
  internal worker jobs. Composition: AGENTS only, prefixed with a
  short "you are an internal worker" framing.
- ``"heartbeat"`` — minimal checklist. Use for the periodic
  heartbeat tick. Composition: HEARTBEAT only.
- ``"bootstrap"`` — one-time init. Composition: BOOTSTRAP only.

Both Ultron's voice pipeline and OpenClaw read the same workspace
files — a change to ``SOUL.md`` is reflected in both consumers'
next user-facing system-prompt build. Background prompts deliberately
DON'T pull in SOUL.md / IDENTITY.md, so persona changes don't leak
into worker reliability.

Reload-on-change is opt-in via :meth:`PersonaLoader.refresh_if_stale`,
which compares mtime + size to the last load and reloads only if
something changed. There's no thread / watcher running by default —
keeping the loader cheap and predictable.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

from ultron.utils.logging import get_logger

logger = get_logger("openclaw_bridge.persona")

# File names + composition order. Ordered list = canonical render order.
_FILES_IN_ORDER = (
    "IDENTITY.md",
    "SOUL.md",
    "USER.md",
    "AGENTS.md",
    "HEARTBEAT.md",
    "BOOTSTRAP.md",
)

# Mode names. ``user_facing`` is the default — full Ultron character.
# Background/heartbeat/bootstrap deliberately strip the character so
# internal worker reliability isn't compromised by persona drift.
PromptMode = Literal["user_facing", "background", "heartbeat", "bootstrap"]
_DEFAULT_MODE: PromptMode = "user_facing"

# Per-mode file inclusion lists (in render order).
#
# user_facing deliberately omits AGENTS.md: the operating rules
# there (tool selection, memory ops, escalation policy) are for
# internal workers, not the voice path. Including them on the voice
# hot path adds ~900 tokens of prefill per turn — measured +218 ms
# TTFT regression vs the original config prompt. Voice-relevant
# rules ("complete what is asked", "do not lecture", uncertainty
# handling) live in SOUL.md alongside the voice/tone content.
_MODE_FILES: Dict[PromptMode, Tuple[str, ...]] = {
    "user_facing": ("IDENTITY.md", "SOUL.md", "USER.md"),
    "background": ("AGENTS.md",),
    "heartbeat": ("HEARTBEAT.md",),
    "bootstrap": ("BOOTSTRAP.md",),
}

# Prefix prepended to each mode's composed system prompt.
_MODE_PREFIX: Dict[PromptMode, str] = {
    "user_facing": "",
    "background": (
        "You are an internal worker for the Ultron system. The user is "
        "not reading or hearing this turn directly. Respond to the "
        "instruction with the literal output it asks for — no "
        "personality, no preamble, no hedging, no NO_REPLY sentinel "
        "unless explicitly instructed. Keep responses minimal."
    ),
    "heartbeat": (
        "You are running a periodic background check. Follow the "
        "checklist below. If nothing requires attention, reply with "
        "exactly HEARTBEAT_OK and nothing else."
    ),
    "bootstrap": (
        "You are running a one-time initialization. Follow the "
        "instructions below. Keep output minimal."
    ),
}


def default_workspace_dir() -> Path:
    """Resolve the OpenClaw workspace dir.

    Order: ``ULTRON_OPENCLAW_WORKSPACE`` env var → user-home default
    (``~/.openclaw/workspace``). Does not call ``openclaw`` CLI to
    avoid a subprocess on every load. Tests can pass an explicit
    ``workspace_dir`` to :class:`PersonaLoader`.
    """
    override = os.environ.get("ULTRON_OPENCLAW_WORKSPACE")
    if override:
        return Path(override)
    return Path.home() / ".openclaw" / "workspace"


_HTML_COMMENT_RE = __import__("re").compile(
    r"<!--.*?-->", flags=__import__("re").DOTALL,
)


@dataclass(frozen=True)
class PersonaFile:
    """One persona file's content + on-disk fingerprint."""

    name: str
    content: str
    mtime_ns: int
    size_bytes: int

    @property
    def is_empty(self) -> bool:
        """True if the file has no content the LLM should see.

        Whitespace-only content is empty. Files that are nothing but
        HTML comments are also treated as empty — the comment is
        useful documentation for the human reader (e.g., "auto-
        populated by maintenance"), but injecting it into a system
        prompt wastes tokens.
        """
        without_comments = _HTML_COMMENT_RE.sub("", self.content)
        return not without_comments.strip()


@dataclass(frozen=True)
class PersonaBundle:
    """The full set of persona files at a point in time."""

    files: Dict[str, PersonaFile]
    workspace_dir: Path
    loaded_at_monotonic: float

    def get(self, name: str) -> Optional[PersonaFile]:
        return self.files.get(name)

    @property
    def fingerprint(self) -> Tuple[Tuple[str, int, int], ...]:
        """Stable signature for change detection: (name, mtime_ns, size).

        Sorted by name so two bundles loaded from identical files have
        identical fingerprints regardless of dict iteration order.
        """
        return tuple(
            (f.name, f.mtime_ns, f.size_bytes)
            for f in sorted(self.files.values(), key=lambda x: x.name)
        )


class PersonaLoader:
    """Load + compose the workspace persona files.

    Args:
        workspace_dir: directory containing the persona files. If
            ``None``, resolves via :func:`default_workspace_dir`.

    Example::

        loader = PersonaLoader()
        prompt = loader.get_system_prompt()  # IDENTITY + SOUL + USER + AGENTS

    The instance caches the last :class:`PersonaBundle`. Call
    :meth:`refresh_if_stale` before each use to pick up file changes,
    or :meth:`load` to force a re-read.

    Thread-safe: a lock is held during file I/O and bundle replacement.
    """

    def __init__(self, workspace_dir: Optional[Path] = None) -> None:
        self.workspace_dir = (
            Path(workspace_dir) if workspace_dir is not None
            else default_workspace_dir()
        )
        self._bundle: Optional[PersonaBundle] = None
        self._lock = threading.Lock()

    # --- loading -----------------------------------------------------------

    def load(self) -> PersonaBundle:
        """Force a fresh read from disk. Returns the new bundle."""
        with self._lock:
            files: Dict[str, PersonaFile] = {}
            for name in _FILES_IN_ORDER:
                path = self.workspace_dir / name
                files[name] = self._read_file(path, name)
            import time as _t  # local to avoid top-level import cost
            self._bundle = PersonaBundle(
                files=files,
                workspace_dir=self.workspace_dir,
                loaded_at_monotonic=_t.monotonic(),
            )
            return self._bundle

    def refresh_if_stale(self) -> PersonaBundle:
        """Reload only if a file's mtime or size has changed since last load.

        Returns the (possibly cached) current bundle. Thread-safe.
        """
        with self._lock:
            current = self._bundle
        if current is None:
            return self.load()
        try:
            stale = self._is_stale(current)
        except Exception as e:
            logger.warning("refresh_if_stale stat failed (%s); reloading", e)
            stale = True
        if stale:
            return self.load()
        return current

    def _is_stale(self, bundle: PersonaBundle) -> bool:
        for name in _FILES_IN_ORDER:
            path = self.workspace_dir / name
            try:
                st = path.stat()
            except FileNotFoundError:
                # If a file was deleted, the cached PersonaFile has
                # size=0/mtime=0 (set by the missing-file path in
                # _read_file). Stale only if the cache disagrees.
                cached = bundle.files.get(name)
                if cached is None or not cached.is_empty:
                    return True
                continue
            cached = bundle.files.get(name)
            if cached is None:
                return True
            if cached.mtime_ns != st.st_mtime_ns or cached.size_bytes != st.st_size:
                return True
        return False

    @property
    def current(self) -> Optional[PersonaBundle]:
        """The last-loaded bundle (or None if never loaded)."""
        return self._bundle

    # --- composition -------------------------------------------------------

    def get_system_prompt(self, mode: PromptMode = _DEFAULT_MODE) -> str:
        """Compose the system prompt for the requested *mode*.

        ``mode`` controls which persona files are included:

        - ``"user_facing"`` — IDENTITY + SOUL + USER + AGENTS. Full
          Ultron character. Default.
        - ``"background"`` — short "internal worker" prefix + AGENTS.
          No character. For heartbeat preflight, cron, summarization,
          tool selection, RAG gating, internal classifiers.
        - ``"heartbeat"`` — short heartbeat-instruction prefix + the
          HEARTBEAT.md checklist.
        - ``"bootstrap"`` — bootstrap prefix + BOOTSTRAP.md.

        Files that are missing or empty contribute nothing. The
        composed string is whitespace-trimmed and uses double-newline
        separators between sections.
        """
        if mode not in _MODE_FILES:
            raise ValueError(
                f"unknown PersonaLoader mode: {mode!r}; "
                f"expected one of {sorted(_MODE_FILES)}"
            )
        bundle = self.refresh_if_stale()
        sections: List[str] = []
        prefix = _MODE_PREFIX.get(mode, "")
        if prefix:
            sections.append(prefix)
        for name in _MODE_FILES[mode]:
            f = bundle.files.get(name)
            if f is None or f.is_empty:
                continue
            # Strip HTML comments from the rendered output. Files use
            # them for human-reader documentation (e.g.,
            # "<!-- auto-populated by maintenance -->") that the LLM
            # doesn't need to see.
            cleaned = _HTML_COMMENT_RE.sub("", f.content).strip()
            if cleaned:
                sections.append(cleaned)
        return "\n\n".join(sections)

    # --- internals ---------------------------------------------------------

    @staticmethod
    def _read_file(path: Path, name: str) -> PersonaFile:
        """Read one persona file. Missing/unreadable files become empty.

        Logs a one-line warning per missing file. Decoding errors are
        replaced (``errors='replace'``) so a corrupt byte doesn't crash
        the loader.
        """
        try:
            stat = path.stat()
        except FileNotFoundError:
            logger.warning("persona file missing: %s", path)
            return PersonaFile(name=name, content="", mtime_ns=0, size_bytes=0)
        except OSError as e:
            logger.warning("persona file unreadable %s (%s)", path, e)
            return PersonaFile(name=name, content="", mtime_ns=0, size_bytes=0)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.warning("persona file read failed %s (%s)", path, e)
            return PersonaFile(
                name=name, content="", mtime_ns=stat.st_mtime_ns,
                size_bytes=stat.st_size,
            )
        return PersonaFile(
            name=name,
            content=content,
            mtime_ns=stat.st_mtime_ns,
            size_bytes=stat.st_size,
        )


__all__ = [
    "PersonaFile",
    "PersonaBundle",
    "PersonaLoader",
    "PromptMode",
    "default_workspace_dir",
]
