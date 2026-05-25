"""Bidirectional focus-chain markdown checklist with file watcher.

Adapted from cline's ``src/core/task/focus-chain/index.ts`` (Apache
2.0; see ``THIRD_PARTY_NOTICES.md``). The pattern: maintain a small
markdown todo list at a known path that BOTH the agent AND the user
can edit. The agent updates checked / unchecked state by rewriting the
file via :meth:`FocusChain.set_items`; the user can open the file in
their editor and flip ``- [ ]`` <-> ``- [x]`` by hand, with
:class:`FocusChainWatcher` (a ``watchdog``-based debounced observer)
propagating the user's diff into the agent's next prompt as a CRITICAL
INFORMATION block.

The contract on the file is minimal — only ``- [x]`` / ``- [X]`` /
``- [ ]`` lines are treated as items. Anything else is preserved as
"surrounding markdown" so the user can attach notes / context per
item without confusing the parser.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

LOGGER = logging.getLogger(__name__)

#: Default debounce window before the watcher fires (matches cline's 300 ms).
DEFAULT_DEBOUNCE_MS: int = 300

#: Default file name used when the caller does not supply one.
DEFAULT_FOCUS_CHAIN_FILENAME: str = "focus_chain.md"

#: Item-line regex: ``- [x]`` / ``- [X]`` / ``- [ ]`` followed by text.
_ITEM_LINE_PATTERN: re.Pattern[str] = re.compile(
    r"^(?P<indent>\s*)-\s\[(?P<state>[\sxX])\]\s(?P<text>.*?)\s*$",
)


@dataclass(frozen=True)
class FocusItem:
    """One line in the focus-chain checklist.

    Attributes:
        text: the checklist item text (no markup).
        done: True when the item is checked, False otherwise.
        order: zero-indexed source position in the file.
    """

    text: str
    done: bool
    order: int = 0


@dataclass(frozen=True)
class FocusChainDiff:
    """Difference between two :class:`FocusChain` states.

    Attributes:
        added: items present in the NEW state but absent from the OLD.
        removed: items present in the OLD state but absent from the NEW.
        completed: items toggled from ``- [ ]`` to ``- [x]``.
        uncompleted: items toggled from ``- [x]`` to ``- [ ]``.
        reordered: True when the set of items is unchanged but the
            sequence differs.
    """

    added: tuple[str, ...] = field(default_factory=tuple)
    removed: tuple[str, ...] = field(default_factory=tuple)
    completed: tuple[str, ...] = field(default_factory=tuple)
    uncompleted: tuple[str, ...] = field(default_factory=tuple)
    reordered: bool = False

    @property
    def is_empty(self) -> bool:
        return not (
            self.added
            or self.removed
            or self.completed
            or self.uncompleted
            or self.reordered
        )


def parse_focus_chain(text: str) -> list[FocusItem]:
    """Parse markdown text into focus items.

    Args:
        text: file contents (may contain non-item lines as surrounding
            commentary).

    Returns:
        List of :class:`FocusItem` in source order.
    """
    items: list[FocusItem] = []
    if not text:
        return items
    for idx, line in enumerate(text.splitlines()):
        match = _ITEM_LINE_PATTERN.match(line)
        if not match:
            continue
        state = match.group("state").lower()
        items.append(
            FocusItem(
                text=match.group("text").strip(),
                done=(state == "x"),
                order=len(items),
            )
        )
    return items


def render_focus_chain(items: Sequence[FocusItem], *, header: Optional[str] = None) -> str:
    """Render ``items`` as a markdown checklist suitable for the file.

    Args:
        items: focus items to render.
        header: optional leading markdown (e.g. ``"# Plan\\n\\n"``).
            When provided, the header is prepended verbatim.

    Returns:
        Markdown string suitable for atomic write.
    """
    out: list[str] = []
    if header:
        out.append(header.rstrip())
        out.append("")
    for item in items:
        check = "x" if item.done else " "
        out.append(f"- [{check}] {item.text}")
    return "\n".join(out) + "\n"


def diff_focus_chains(
    old: Sequence[FocusItem], new: Sequence[FocusItem],
) -> FocusChainDiff:
    """Compute the diff between two focus-chain states."""
    old_set = {item.text for item in old}
    new_set = {item.text for item in new}
    added = tuple(item.text for item in new if item.text not in old_set)
    removed = tuple(item.text for item in old if item.text not in new_set)
    old_done = {item.text for item in old if item.done}
    new_done = {item.text for item in new if item.done}
    completed = tuple(
        text for text in new_done if text in old_set and text not in old_done
    )
    uncompleted = tuple(
        text for text in old_done if text in new_set and text not in new_done
    )
    reordered = False
    if old_set == new_set:
        old_order = [item.text for item in old]
        new_order = [item.text for item in new]
        reordered = old_order != new_order
    return FocusChainDiff(
        added=added,
        removed=removed,
        completed=completed,
        uncompleted=uncompleted,
        reordered=reordered,
    )


def render_critical_info_block(diff: FocusChainDiff) -> str:
    """Render a user-edit diff as a CRITICAL INFORMATION block.

    Args:
        diff: outcome of :func:`diff_focus_chains`.

    Returns:
        Markdown string to prepend to the next prompt. Empty string
        when the diff is empty.
    """
    if diff.is_empty:
        return ""
    lines = [
        "**CRITICAL INFORMATION:** The user modified the focus-chain "
        "checklist — review every change before acting.",
    ]
    if diff.added:
        lines.append("Added:")
        for item in diff.added:
            lines.append(f"  + {item}")
    if diff.removed:
        lines.append("Removed:")
        for item in diff.removed:
            lines.append(f"  - {item}")
    if diff.completed:
        lines.append("Marked done:")
        for item in diff.completed:
            lines.append(f"  [x] {item}")
    if diff.uncompleted:
        lines.append("Marked pending:")
        for item in diff.uncompleted:
            lines.append(f"  [ ] {item}")
    if diff.reordered:
        lines.append("Item order was rearranged.")
    return "\n".join(lines)


def progress_hint(items: Sequence[FocusItem]) -> str:
    """Return a per-percent-complete hint for prompt inclusion.

    Mirrors cline's tailored instructions per progress band.
    """
    total = len(items)
    if total == 0:
        return ""
    done = sum(1 for item in items if item.done)
    ratio = done / total
    if ratio == 0.0:
        return "Plan is fresh; mark items as you complete them."
    if ratio < 0.25:
        return f"{done}/{total} done — keep updating the plan as you progress."
    if ratio < 0.5:
        return f"{done}/{total} done — about a quarter complete."
    if ratio < 0.75:
        return f"{done}/{total} done — past halfway; focus on the remaining items."
    if ratio < 1.0:
        return f"{done}/{total} done — almost there; finish strong."
    return f"All {total} items marked done. Confirm completion and wrap up."


@dataclass
class FocusChain:
    """In-memory + on-disk focus chain with atomic write semantics.

    Args:
        path: file location for the markdown checklist.
        header: optional markdown header rendered at the top of the file.
    """

    path: Path
    header: Optional[str] = None
    items: list[FocusItem] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load(self) -> list[FocusItem]:
        """Read the file (if it exists) and replace the in-memory items."""
        with self._lock:
            if not self.path.exists():
                self.items = []
                return list(self.items)
            try:
                text = self.path.read_text(encoding="utf-8")
            except OSError:
                self.items = []
                return list(self.items)
            self.items = parse_focus_chain(text)
            return list(self.items)

    def save(self) -> None:
        """Atomically write the current items to ``self.path``."""
        with self._lock:
            text = render_focus_chain(self.items, header=self.header)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp_path.write_text(text, encoding="utf-8")
            os.replace(tmp_path, self.path)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def set_items(self, items: Iterable[FocusItem | tuple[str, bool] | str]) -> None:
        """Replace the in-memory items wholesale.

        Args:
            items: iterable of :class:`FocusItem`, ``(text, done)``
                tuples, or bare strings (treated as pending).
        """
        normalised: list[FocusItem] = []
        for entry in items:
            if isinstance(entry, FocusItem):
                normalised.append(
                    FocusItem(
                        text=entry.text,
                        done=entry.done,
                        order=len(normalised),
                    )
                )
                continue
            if isinstance(entry, tuple) and len(entry) == 2:
                text, done = entry
                normalised.append(
                    FocusItem(
                        text=str(text),
                        done=bool(done),
                        order=len(normalised),
                    )
                )
                continue
            if isinstance(entry, str):
                normalised.append(
                    FocusItem(text=entry, done=False, order=len(normalised))
                )
        with self._lock:
            self.items = normalised

    def mark_done(self, text: str) -> bool:
        """Mark the first matching item as done. Returns True on hit."""
        with self._lock:
            for idx, item in enumerate(self.items):
                if item.text == text and not item.done:
                    self.items[idx] = FocusItem(text=item.text, done=True, order=item.order)
                    return True
            return False

    def mark_pending(self, text: str) -> bool:
        """Mark the first matching item as pending. Returns True on hit."""
        with self._lock:
            for idx, item in enumerate(self.items):
                if item.text == text and item.done:
                    self.items[idx] = FocusItem(text=item.text, done=False, order=item.order)
                    return True
            return False

    def progress_ratio(self) -> float:
        """Return done / total as a 0.0-1.0 float (0.0 when empty)."""
        with self._lock:
            if not self.items:
                return 0.0
            done = sum(1 for item in self.items if item.done)
            return done / len(self.items)

    def progress_hint(self) -> str:
        with self._lock:
            return progress_hint(list(self.items))


class FocusChainWatcher:
    """Watchdog-based observer that fires on external user edits.

    Args:
        chain: the :class:`FocusChain` to watch.
        on_user_edit: callback invoked with the :class:`FocusChainDiff`
            describing what the user changed.
        debounce_ms: debounce window between watcher fires.
        clock: optional monotonic clock (test hook).

    Notes:
        The watcher uses :mod:`watchdog` when installed; on systems
        where it is missing, the watcher exposes a manual
        :meth:`poll_now` helper that synchronously diffs the file
        against the in-memory state and fires the callback.
    """

    def __init__(
        self,
        chain: FocusChain,
        on_user_edit: Callable[[FocusChainDiff], None],
        *,
        debounce_ms: int = DEFAULT_DEBOUNCE_MS,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._chain = chain
        self._on_user_edit = on_user_edit
        self._debounce_ms = max(0, int(debounce_ms))
        self._clock = clock or time.monotonic
        self._lock = threading.RLock()
        self._last_fire_at: float = 0.0
        self._observer = None
        self._last_mtime_ns: int = self._current_mtime_ns()

    @property
    def watching(self) -> bool:
        return self._observer is not None

    def start(self) -> bool:
        """Begin watching the file. Returns True when watchdog is wired."""
        if self.watching:
            return True
        try:
            from watchdog.events import FileSystemEventHandler  # type: ignore
            from watchdog.observers import Observer  # type: ignore
        except ImportError:
            LOGGER.warning(
                "watchdog not available; FocusChainWatcher will run in "
                "manual-poll mode only.",
            )
            return False
        chain = self._chain
        watcher = self

        class _Handler(FileSystemEventHandler):
            def on_modified(self, event):  # type: ignore[no-untyped-def]
                if event.is_directory:
                    return
                if Path(event.src_path) != chain.path:
                    return
                watcher.poll_now()

            on_created = on_modified

        observer = Observer()
        try:
            observer.schedule(_Handler(), str(self._chain.path.parent), recursive=False)
            observer.daemon = True
            observer.start()
        except Exception:  # noqa: BLE001
            LOGGER.warning("failed to start watchdog observer", exc_info=True)
            return False
        self._observer = observer
        return True

    def stop(self) -> None:
        """Stop the watcher (idempotent)."""
        if self._observer is None:
            return
        try:
            self._observer.stop()
            self._observer.join(timeout=2.0)
        except Exception:  # noqa: BLE001
            pass
        self._observer = None

    def poll_now(self) -> Optional[FocusChainDiff]:
        """Synchronously diff the disk file vs in-memory and fire callback.

        Returns the diff (and dispatches it) when the file changed since
        the last load; None when nothing changed.
        """
        with self._lock:
            now_ms = self._clock() * 1000
            if (now_ms - self._last_fire_at * 1000) < self._debounce_ms:
                return None
            mtime = self._current_mtime_ns()
            if mtime == 0 or mtime == self._last_mtime_ns:
                return None
            self._last_mtime_ns = mtime
            old_items = list(self._chain.items)
            try:
                text = self._chain.path.read_text(encoding="utf-8")
            except OSError:
                return None
            new_items = parse_focus_chain(text)
            diff = diff_focus_chains(old_items, new_items)
            if diff.is_empty:
                self._chain.items = new_items
                return None
            self._chain.items = new_items
            self._last_fire_at = self._clock()
        try:
            self._on_user_edit(diff)
        except Exception:  # noqa: BLE001
            LOGGER.warning("focus-chain on_user_edit raised", exc_info=True)
        return diff

    def _current_mtime_ns(self) -> int:
        try:
            return os.stat(self._chain.path).st_mtime_ns
        except OSError:
            return 0


__all__ = [
    "DEFAULT_DEBOUNCE_MS",
    "DEFAULT_FOCUS_CHAIN_FILENAME",
    "FocusChain",
    "FocusChainDiff",
    "FocusChainWatcher",
    "FocusItem",
    "diff_focus_chains",
    "parse_focus_chain",
    "progress_hint",
    "render_critical_info_block",
    "render_focus_chain",
]
