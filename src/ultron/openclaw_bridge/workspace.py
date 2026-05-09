"""Shared workspace writer (Phase 3.3).

Both Ultron and OpenClaw read from the same workspace directory
(``~/.openclaw/workspace`` by default). Ultron's maintenance pass
writes to ``USER.md`` (auto-populated user facts), and both processes
may append to ``memory/YYYY-MM-DD.md`` daily files. Without
coordination, simultaneous writes can corrupt the file or race past
each other's content.

This module provides a :class:`WorkspaceWriter` with three guarantees:

1. **Atomic writes** — every replacement goes through a temp file
   then ``os.replace``. Readers never see a partial file.
2. **Advisory lockfile** — a sibling ``.lock`` file gates concurrent
   writers to the same target. ``filelock`` is cross-platform; it
   uses ``fcntl`` on POSIX and ``msvcrt`` on Windows.
3. **Async surface** — methods are ``async`` and dispatch the actual
   file IO via :func:`asyncio.to_thread` so the orchestrator's event
   loop is never blocked.

What this writer does NOT do: it doesn't touch persona files
(SOUL.md, IDENTITY.md, AGENTS.md, BOOTSTRAP.md, HEARTBEAT.md). Those
are version-controlled human-edited content; modifying them from
code would race with the user's editor. Read-only access is in
:mod:`ultron.openclaw_bridge.persona`.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from filelock import FileLock, Timeout

from ultron.errors import FilesystemError
from ultron.openclaw_bridge.persona import default_workspace_dir
from ultron.utils.logging import get_logger

logger = get_logger("openclaw_bridge.workspace")


# Section heading regex used by :meth:`WorkspaceWriter.update_memory_md`.
# Matches Markdown ATX headings (any level): ``# Foo``, ``## Foo``, etc.
_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class WriteResult:
    """Outcome of a workspace write. Returned by every public method."""

    path: Path
    bytes_written: int
    created: bool                                   # True iff the file did not exist
    section_replaced: Optional[str] = None          # for ``update_memory_md``
    error: Optional[str] = None


class WorkspaceWriter:
    """Coordinated writer for the shared OpenClaw workspace.

    Args:
        workspace_dir: workspace root. ``None`` triggers
            :func:`default_workspace_dir` (``~/.openclaw/workspace``).
        lock_timeout_s: max wait for the advisory lockfile before
            giving up. Short by design — write contention should be
            rare. Caller gets a :class:`FilesystemError`-wrapped
            timeout result rather than blocking forever.
    """

    def __init__(
        self,
        workspace_dir: Optional[Path] = None,
        *,
        lock_timeout_s: float = 5.0,
    ) -> None:
        self._workspace = (
            Path(workspace_dir) if workspace_dir is not None else default_workspace_dir()
        )
        self._lock_timeout_s = lock_timeout_s

    @property
    def workspace_dir(self) -> Path:
        return self._workspace

    # -------------------------------------------------------------------
    # Daily memory entries
    # -------------------------------------------------------------------

    async def write_memory_entry(
        self,
        entry: str,
        *,
        date: Optional[_dt.date] = None,
        prefix_timestamp: bool = True,
    ) -> WriteResult:
        """Append a single entry to the daily memory file.

        File path: ``<workspace>/memory/YYYY-MM-DD.md``. Created on
        first write of the day. Each entry is separated from the
        previous content by a blank line; when ``prefix_timestamp`` is
        True (default) the entry is prefixed with the local time
        ``HH:MM`` so subsequent reads can reconstruct chronology
        without sharing timezone state.
        """
        if not entry.strip():
            return WriteResult(
                path=self._daily_path(date or _dt.date.today()),
                bytes_written=0, created=False,
                error="empty entry",
            )
        target_date = date or _dt.date.today()
        target = self._daily_path(target_date)
        block = self._format_entry(entry, prefix_timestamp=prefix_timestamp)
        return await asyncio.to_thread(
            self._append_locked, target, block,
        )

    # -------------------------------------------------------------------
    # MEMORY.md (long-term curated memory)
    # -------------------------------------------------------------------

    async def update_memory_md(
        self,
        section: str,
        content: str,
        *,
        create_if_missing: bool = True,
    ) -> WriteResult:
        """Replace one Markdown section in ``MEMORY.md``.

        ``section`` is matched against ATX heading text exactly (case
        sensitive). The replacement preserves the heading line and
        substitutes the body up to (but not including) the next heading
        of equal-or-greater rank. When the section doesn't exist and
        ``create_if_missing`` is True, the section is appended as a
        level-2 heading.
        """
        if not section.strip():
            raise ValueError("section name must be non-empty")
        target = self._workspace / "MEMORY.md"
        return await asyncio.to_thread(
            self._update_section_locked,
            target, section, content, create_if_missing,
        )

    # -------------------------------------------------------------------
    # USER.md (auto-populated user info)
    # -------------------------------------------------------------------

    async def update_user_md(self, content: str) -> WriteResult:
        """Replace the entire ``USER.md`` content.

        USER.md is regenerated from the Qdrant facts collection during
        maintenance runs. The file is small and the write is rare, so
        full replacement (rather than section-level patching) is
        appropriate here.
        """
        target = self._workspace / "USER.md"
        return await asyncio.to_thread(
            self._replace_locked, target, content,
        )

    # -------------------------------------------------------------------
    # Internals (sync; called from to_thread)
    # -------------------------------------------------------------------

    def _daily_path(self, date: _dt.date) -> Path:
        return self._workspace / "memory" / f"{date.isoformat()}.md"

    @staticmethod
    def _format_entry(entry: str, *, prefix_timestamp: bool) -> str:
        if prefix_timestamp:
            stamp = _dt.datetime.now().strftime("%H:%M")
            return f"\n- {stamp} — {entry.strip()}\n"
        return f"\n- {entry.strip()}\n"

    def _lock_for(self, target: Path) -> FileLock:
        return FileLock(str(target) + ".lock")

    def _ensure_parent(self, target: Path) -> None:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise FilesystemError(
                f"failed to create workspace dir {target.parent}: {e}",
                context={"path": str(target.parent)},
            ) from e

    def _append_locked(self, target: Path, block: str) -> WriteResult:
        self._ensure_parent(target)
        existed = target.exists()
        try:
            with self._lock_for(target).acquire(timeout=self._lock_timeout_s):
                # Append directly — daily-memory files are append-only
                # and a partial append is recoverable (worst case: a
                # truncated trailing line). Atomic-replace is overkill
                # and would force readers to keep their handles fresh.
                with target.open("a", encoding="utf-8") as fh:
                    written = fh.write(block)
        except Timeout:
            return WriteResult(
                path=target, bytes_written=0, created=False,
                error=f"lock timeout after {self._lock_timeout_s:.1f}s",
            )
        except OSError as e:
            return WriteResult(
                path=target, bytes_written=0, created=False,
                error=f"write failed: {e}",
            )
        return WriteResult(
            path=target,
            bytes_written=len(block.encode("utf-8")),
            created=not existed,
        )

    def _replace_locked(self, target: Path, content: str) -> WriteResult:
        self._ensure_parent(target)
        existed = target.exists()
        try:
            with self._lock_for(target).acquire(timeout=self._lock_timeout_s):
                self._atomic_write(target, content)
        except Timeout:
            return WriteResult(
                path=target, bytes_written=0, created=False,
                error=f"lock timeout after {self._lock_timeout_s:.1f}s",
            )
        except OSError as e:
            return WriteResult(
                path=target, bytes_written=0, created=False,
                error=f"replace failed: {e}",
            )
        return WriteResult(
            path=target,
            bytes_written=len(content.encode("utf-8")),
            created=not existed,
        )

    def _update_section_locked(
        self,
        target: Path,
        section: str,
        new_body: str,
        create_if_missing: bool,
    ) -> WriteResult:
        self._ensure_parent(target)
        existed = target.exists()
        original = target.read_text(encoding="utf-8") if existed else ""
        try:
            with self._lock_for(target).acquire(timeout=self._lock_timeout_s):
                # Re-read inside the lock so we operate on the latest
                # content if a concurrent writer landed between
                # construction and lock acquisition.
                if existed:
                    original = target.read_text(encoding="utf-8")
                replaced, body_changed = self._splice_section(
                    original, section, new_body, create_if_missing,
                )
                if not body_changed:
                    return WriteResult(
                        path=target,
                        bytes_written=0,
                        created=not existed,
                        section_replaced=None,
                        error="section not found and create_if_missing=False",
                    )
                self._atomic_write(target, replaced)
        except Timeout:
            return WriteResult(
                path=target, bytes_written=0, created=False,
                error=f"lock timeout after {self._lock_timeout_s:.1f}s",
            )
        except OSError as e:
            return WriteResult(
                path=target, bytes_written=0, created=False,
                error=f"section update failed: {e}",
            )
        return WriteResult(
            path=target,
            bytes_written=len(replaced.encode("utf-8")),
            created=not existed,
            section_replaced=section,
        )

    @staticmethod
    def _splice_section(
        original: str,
        section: str,
        new_body: str,
        create_if_missing: bool,
    ) -> tuple[str, bool]:
        """Replace ``section`` in ``original`` with ``new_body``.

        Returns ``(updated_text, body_changed)``. When the section is
        missing and ``create_if_missing`` is True, the new section is
        appended as a level-2 heading. ``body_changed`` is False only
        when the section was missing and creation was disabled.
        """
        # Locate the heading line whose title matches ``section``.
        match_iter = list(_HEADING_RE.finditer(original))
        target_match = None
        for m in match_iter:
            if m.group("title").strip() == section.strip():
                target_match = m
                break
        if target_match is None:
            if not create_if_missing:
                return original, False
            sep = "" if original.endswith("\n") or not original else "\n"
            section_block = f"{sep}\n## {section.strip()}\n\n{new_body.rstrip()}\n"
            return original + section_block, True

        # Find the next heading at the same-or-higher rank as the boundary.
        heading_rank = len(target_match.group("hashes"))
        body_start = target_match.end() + 1  # past the newline after heading
        next_boundary = len(original)
        for m in match_iter:
            if m.start() <= target_match.start():
                continue
            other_rank = len(m.group("hashes"))
            if other_rank <= heading_rank:
                next_boundary = m.start()
                break

        head = original[: target_match.end()]
        tail = original[next_boundary:]
        # Normalize the new body: one leading blank line for breathing
        # room, trailing newline so subsequent sections start cleanly.
        body = "\n" + new_body.rstrip() + "\n\n"
        return head + body + tail, True

    @staticmethod
    def _atomic_write(target: Path, content: str) -> None:
        """Write ``content`` to ``target`` atomically.

        Strategy: write to a sibling temp file, fsync, then
        ``os.replace``. The replace is atomic on both POSIX and NTFS
        (Win32 ``MoveFileEx`` with ``MOVEFILE_REPLACE_EXISTING``).
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=target.name + ".", suffix=".tmp", dir=str(target.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    # fsync isn't supported on every filesystem (some
                    # tmpfs / network mounts). The replace-and-pray
                    # fallback is still safer than non-atomic writes.
                    pass
            os.replace(tmp_path, target)
        except Exception:
            # Best-effort cleanup; never raise from here.
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass
            raise


__all__ = [
    "WorkspaceWriter",
    "WriteResult",
]
