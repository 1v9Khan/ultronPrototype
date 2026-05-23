"""Multi-file undo stack for ultron-driven file edits.

Adapted from SWE-Agent's
``tools/edit_anthropic/bin/str_replace_editor:_file_history``
(MIT, Yang et al. 2024). The pattern: every file the supervisor
or coding pipeline writes gets its PRE-edit content snapshotted
into a per-session JSON store, keyed by file path. The user can
then say "undo what you just did to X" and the supervisor pops
the most recent snapshot off the stack.

Differences from SWE-Agent:

* **Per-session backing store.** SWE-Agent uses one global
  registry; ultron uses :class:`SessionRegistry` keyed by session
  id so concurrent coding sessions don't fight over the same undo
  stack.
* **Capped depth per file.** ``max_history_per_file`` (default 10)
  drops the oldest snapshot when the cap is reached. Without this,
  long-running sessions on large files would exhaust disk.
* **Optional narration metadata.** Each snapshot carries an
  optional ``narration`` string + timestamp so the voice path can
  match "undo the change about adding the close button" against
  recent narrations.
* **Atomic recovery.** :meth:`undo_last` writes the pre-edit
  content back to disk via a tempfile + ``os.replace`` so a
  partial-write crash leaves the file in a coherent state.
* **Fail-open.** Snapshot failures (disk full, permission denied)
  log WARN and let the original write proceed; undo failures
  raise so callers can surface a clear "couldn't undo" voice
  message.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ultron.coding.session_registry import SessionRegistry, get_session_registry

logger = logging.getLogger(__name__)

#: Default cap on per-file snapshot depth. SWE-Agent doesn't cap; we
#: do because long voice coding sessions on huge files would otherwise
#: balloon ``data/coding/sessions/<id>/registry.json``.
DEFAULT_MAX_HISTORY_PER_FILE: int = 10

#: Registry key under which the file-history dict lives. Mirrors
#: SWE-Agent's `file_history` key for cross-tool legibility.
REGISTRY_KEY: str = "file_history"


@dataclass(frozen=True)
class FileHistoryEntry:
    """One pre-edit snapshot for a single file.

    :param content: the file's text content BEFORE the edit. ``None``
        if the file did not exist before (the edit created it).
    :param recorded_at: epoch seconds when the snapshot was taken.
    :param narration: optional human-readable label (the supervisor
        narration line that accompanied the edit). Lets the voice
        path match "undo the change about X".
    :param origin: optional source label (``"runner"``, ``"supervisor"``,
        ``"manual"``, etc.) for audit visibility.
    """

    content: Optional[str]
    recorded_at: float
    narration: str = ""
    origin: str = ""


@dataclass
class UndoResult:
    """Output of :meth:`FileHistory.undo_last`.

    :param applied: True if the undo successfully wrote back; False
        if there was no history for the path.
    :param entry: the snapshot that was applied (None if nothing to
        undo).
    :param error: human-readable error string if the undo failed
        partway through writing (in which case the original file is
        left untouched).
    """

    applied: bool
    entry: Optional[FileHistoryEntry] = None
    error: str = ""


class FileHistory:
    """Per-session multi-file undo stack.

    Construct via :func:`get_file_history` (preferred, singleton per
    session_id) or directly with a custom :class:`SessionRegistry`
    for tests.

    Public API:

    * :meth:`record_pre_edit(path, narration="", origin="")`
    * :meth:`undo_last(path) -> UndoResult`
    * :meth:`peek_last(path) -> Optional[FileHistoryEntry]`
    * :meth:`history_for(path) -> list[FileHistoryEntry]`
    * :meth:`clear(path)` / :meth:`clear_all()`
    * :meth:`find_by_narration(query, n=5) -> list[tuple[str, FileHistoryEntry]]`
    """

    def __init__(
        self,
        *,
        registry: SessionRegistry,
        max_history_per_file: int = DEFAULT_MAX_HISTORY_PER_FILE,
    ) -> None:
        if max_history_per_file < 1:
            raise ValueError(
                f"max_history_per_file must be >= 1 (got {max_history_per_file})"
            )
        self.registry = registry
        self.max_history_per_file = int(max_history_per_file)

    # ----- snapshot recording -------------------------------------------

    def record_pre_edit(
        self,
        path: str,
        *,
        narration: str = "",
        origin: str = "",
    ) -> bool:
        """Snapshot the current content at ``path`` before an edit.

        Returns True if the snapshot was recorded, False otherwise
        (e.g., the path doesn't exist and isn't readable in any way
        we recognise as "creating new" vs "failed read"). On a
        missing-file case the snapshot stores ``content=None`` so
        :meth:`undo_last` can delete the file to roll back a creation.

        Fail-open: read errors log WARN and return False; the caller's
        original edit proceeds regardless.
        """
        key = self._canonical_path(path)
        if not key:
            return False
        existing_content: Optional[str]
        p = Path(key)
        if p.exists():
            try:
                existing_content = p.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning(
                    "file_history.record_pre_edit %s read failed: %s; "
                    "snapshot skipped",
                    key,
                    exc,
                )
                return False
        else:
            existing_content = None
        entry = FileHistoryEntry(
            content=existing_content,
            recorded_at=time.time(),
            narration=narration,
            origin=origin,
        )
        stored = self._load_store()
        stack = stored.get(key, [])
        stack.append(self._entry_to_dict(entry))
        # Trim the oldest snapshots.
        if len(stack) > self.max_history_per_file:
            stack = stack[-self.max_history_per_file :]
        stored[key] = stack
        self._save_store(stored)
        return True

    # ----- undo ---------------------------------------------------------

    def undo_last(self, path: str) -> UndoResult:
        """Pop the most recent snapshot off ``path`` and write it back.

        Returns an :class:`UndoResult`. If the snapshot's content was
        ``None`` (file was created by the edit), the file is DELETED
        to roll back the creation. Otherwise the bytes are restored
        atomically (tempfile + ``os.replace``).
        """
        key = self._canonical_path(path)
        if not key:
            return UndoResult(applied=False, error="invalid path")
        stored = self._load_store()
        stack = stored.get(key, [])
        if not stack:
            return UndoResult(applied=False)
        last_dict = stack[-1]
        entry = self._entry_from_dict(last_dict)
        try:
            if entry.content is None:
                p = Path(key)
                if p.exists():
                    p.unlink()
            else:
                self._atomic_write(Path(key), entry.content)
        except OSError as exc:
            return UndoResult(
                applied=False,
                entry=entry,
                error=f"write failed: {exc}",
            )
        # On success: pop the snapshot AFTER the write completes so a
        # crash mid-write leaves the snapshot intact for retry.
        stack.pop()
        if stack:
            stored[key] = stack
        else:
            stored.pop(key, None)
        self._save_store(stored)
        return UndoResult(applied=True, entry=entry)

    # ----- inspection ---------------------------------------------------

    def peek_last(self, path: str) -> Optional[FileHistoryEntry]:
        """Return the most recent snapshot without popping it.

        Useful for diagnostics and "are we going to lose anything if
        we undo?" prompts.
        """
        key = self._canonical_path(path)
        if not key:
            return None
        stored = self._load_store()
        stack = stored.get(key, [])
        if not stack:
            return None
        return self._entry_from_dict(stack[-1])

    def history_for(self, path: str) -> list[FileHistoryEntry]:
        """Return the full snapshot stack for ``path`` (oldest first)."""
        key = self._canonical_path(path)
        if not key:
            return []
        stored = self._load_store()
        stack = stored.get(key, [])
        return [self._entry_from_dict(d) for d in stack]

    def all_paths(self) -> list[str]:
        """Return every path with at least one snapshot recorded.

        Sorted lexicographically for stable test assertions.
        """
        stored = self._load_store()
        return sorted(stored.keys())

    def total_snapshots(self) -> int:
        """Return total snapshot count across all paths."""
        stored = self._load_store()
        return sum(len(v) for v in stored.values())

    def find_by_narration(
        self,
        query: str,
        *,
        n: int = 5,
    ) -> list[tuple[str, FileHistoryEntry]]:
        """Substring search across snapshot narrations.

        Returns up to ``n`` matches as ``(path, entry)`` tuples,
        most-recent first. Useful for voice intents like "undo the
        change about adding the Tkinter button".
        """
        if not query:
            return []
        query_lc = query.lower()
        results: list[tuple[float, str, FileHistoryEntry]] = []
        stored = self._load_store()
        for path, stack in stored.items():
            for snap in stack:
                narration = str(snap.get("narration", ""))
                if query_lc in narration.lower():
                    results.append(
                        (
                            float(snap.get("recorded_at", 0.0)),
                            path,
                            self._entry_from_dict(snap),
                        )
                    )
        results.sort(key=lambda triple: triple[0], reverse=True)
        return [(p, e) for _, p, e in results[:n]]

    # ----- clear --------------------------------------------------------

    def clear(self, path: str) -> int:
        """Drop every snapshot for ``path``. Returns the number dropped."""
        key = self._canonical_path(path)
        if not key:
            return 0
        stored = self._load_store()
        dropped = len(stored.pop(key, []))
        if dropped:
            self._save_store(stored)
        return dropped

    def clear_all(self) -> int:
        """Drop every snapshot across every path. Returns total dropped."""
        stored = self._load_store()
        total = sum(len(v) for v in stored.values())
        if total:
            self.registry.pop(REGISTRY_KEY, default=None)
        return total

    # ----- internals ----------------------------------------------------

    def _load_store(self) -> dict[str, list[dict]]:
        raw = self.registry.get(REGISTRY_KEY)
        if not isinstance(raw, dict):
            return {}
        # Defensive: coerce each entry to a list of dicts.
        out: dict[str, list[dict]] = {}
        for k, v in raw.items():
            if isinstance(v, list):
                out[str(k)] = [d for d in v if isinstance(d, dict)]
        return out

    def _save_store(self, store: dict[str, list[dict]]) -> None:
        self.registry.set(REGISTRY_KEY, store)

    @staticmethod
    def _entry_to_dict(entry: FileHistoryEntry) -> dict:
        return {
            "content": entry.content,
            "recorded_at": entry.recorded_at,
            "narration": entry.narration,
            "origin": entry.origin,
        }

    @staticmethod
    def _entry_from_dict(d: dict) -> FileHistoryEntry:
        return FileHistoryEntry(
            content=d.get("content"),
            recorded_at=float(d.get("recorded_at", 0.0)),
            narration=str(d.get("narration", "")),
            origin=str(d.get("origin", "")),
        )

    @staticmethod
    def _canonical_path(path: str) -> str:
        if not path:
            return ""
        try:
            return str(Path(path).expanduser().resolve())
        except (OSError, RuntimeError):
            return str(path)

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=path.name + ".",
            suffix=".undo.tmp",
            dir=str(path.parent),
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(str(tmp_path), str(path))


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------


def get_file_history(
    session_id: str,
    *,
    registry: Optional[SessionRegistry] = None,
    max_history_per_file: int = DEFAULT_MAX_HISTORY_PER_FILE,
) -> FileHistory:
    """Return a :class:`FileHistory` for ``session_id``.

    Uses :func:`get_session_registry` for the backing store unless
    ``registry`` is passed (for tests).
    """
    if registry is None:
        registry = get_session_registry(session_id)
    return FileHistory(
        registry=registry,
        max_history_per_file=max_history_per_file,
    )


__all__ = [
    "DEFAULT_MAX_HISTORY_PER_FILE",
    "FileHistory",
    "FileHistoryEntry",
    "REGISTRY_KEY",
    "UndoResult",
    "get_file_history",
]
