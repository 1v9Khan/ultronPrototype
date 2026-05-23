"""Persistent windowed-file state machine.

Direct port of SWE-Agent's
``tools/windowed/lib/windowed_file.py:WindowedFile`` (MIT,
Yang et al. 2024) adapted to ultron's session-scoped registry.

The pattern: the "current file" + "first line in window" + "window
height" + "scroll overlap" all live in a per-session JSON store
(:class:`SessionRegistry`). Subsequent ``goto`` / ``scroll_up`` /
``scroll_down`` / ``view`` calls act on the open file without
re-passing the path.

For ultron the state machine has two consumers:

* The architect narrator + completion narrator can read
  ``current_file`` so the narration speaks about the right file
  without re-quoting the path the user already heard.
* A future supervisor / Claude-Code-bridge integration can use the
  state to ask "where is Claude looking right now?" and inject
  the answer into a dispatch prompt.

Differences from SWE-Agent:

* **Per-session isolation.** SWE-Agent uses one global state file
  at ``/root/.swe-agent-env``; ultron uses the per-session
  registry.
* **1-indexed external API.** SWE-Agent's internal `_first_line`
  is 0-indexed; the public-facing `WindowState.open(path, line=N)`
  takes 1-indexed N (matches user/voice expectations).
* **No mutation surface.** SWE-Agent's `WindowedFile` also
  implements `set_window_text`, `replace_in_window`, `insert`,
  `undo_edit`. Those mutating operations belong to ultron's
  existing safety + file-history layer (T20); this module is
  read-only state tracking.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ultron.coding.session_registry import SessionRegistry, get_session_registry
from ultron.coding.window_expand import (
    DEFAULT_MAX_ADDED_LINES,
    WindowExpander,
)

#: Default window height (lines). Mirrors SWE-Agent's `WINDOW=100`.
DEFAULT_WINDOW_LINES: int = 100

#: Default scroll overlap (lines retained when scrolling). SWE-Agent
#: uses 2 so the previous window's last 2 lines remain visible at the
#: top of the next view -- smooths the perceptual jump.
DEFAULT_OVERLAP_LINES: int = 2

#: Fraction of the window above the goto target line. SWE-Agent uses
#: 1/6 so the target lands ~1/6 down the window (natural reading
#: position).
GOTO_OFFSET_MULTIPLIER: float = 1.0 / 6.0

#: Registry keys -- match SWE-Agent's names so cross-tool inspection
#: (e.g. status dumps) works on either ecosystem.
KEY_CURRENT_FILE: str = "CURRENT_FILE"
KEY_FIRST_LINE: str = "FIRST_LINE"  # 0-indexed (matches SWE-Agent)
KEY_WINDOW: str = "WINDOW"
KEY_OVERLAP: str = "OVERLAP"


# ---------------------------------------------------------------------------
# View output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowView:
    """A rendered window snapshot.

    :param path: absolute path of the open file (or empty if none).
    :param text: the rendered window text (with optional line numbers
        and header / pre / post annotations).
    :param first_line: 1-indexed line number at the TOP of the window.
    :param last_line: 1-indexed line number at the BOTTOM of the
        window (inclusive).
    :param total_lines: total line count in the file.
    :param lines_above: how many lines exist above the window.
    :param lines_below: how many lines exist below the window.
    """

    path: str
    text: str
    first_line: int
    last_line: int
    total_lines: int
    lines_above: int
    lines_below: int


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class WindowState:
    """Session-scoped current-file + window position.

    Public API:

    * :meth:`open(path, line=None)` -- set the current file (and
      optionally jump to a 1-indexed line).
    * :meth:`close()` -- clear the current file.
    * :meth:`current_file()` -- the open file's absolute path or
      ``None``.
    * :meth:`goto(line)` -- 1-indexed jump within the current file.
    * :meth:`scroll_down()` / :meth:`scroll_up()` -- one window
      forward / backward with :data:`DEFAULT_OVERLAP_LINES` of
      preserved context.
    * :meth:`view(line_numbers=True, status_line=True,
      pre_post_line=True)` -- render the current window as a
      :class:`WindowView`.
    * :meth:`view_with_semantic_expansion(suffix=None)` -- like
      :meth:`view` but extends the displayed range to the nearest
      def/class boundary using :class:`WindowExpander` (T5).
    """

    def __init__(
        self,
        *,
        registry: SessionRegistry,
        default_window: int = DEFAULT_WINDOW_LINES,
        default_overlap: int = DEFAULT_OVERLAP_LINES,
    ) -> None:
        if default_window <= 0:
            raise ValueError(
                f"default_window must be > 0 (got {default_window})"
            )
        if default_overlap < 0:
            raise ValueError(
                f"default_overlap must be >= 0 (got {default_overlap})"
            )
        if default_overlap >= default_window:
            raise ValueError(
                f"default_overlap ({default_overlap}) must be < "
                f"default_window ({default_window})"
            )
        self.registry = registry
        self._default_window = int(default_window)
        self._default_overlap = int(default_overlap)
        self._initialise_window_settings()

    def _initialise_window_settings(self) -> None:
        """Ensure WINDOW + OVERLAP are present in the registry."""
        if KEY_WINDOW not in self.registry:
            self.registry[KEY_WINDOW] = self._default_window
        if KEY_OVERLAP not in self.registry:
            self.registry[KEY_OVERLAP] = self._default_overlap

    # ----- open / close -------------------------------------------------

    def open(self, path: str, line: Optional[int] = None) -> WindowView:
        """Set ``path`` as the current file (and optionally jump to
        a 1-indexed line). Returns the rendered window.

        :param path: absolute or relative path; the resolved absolute
            path is stored.
        :param line: 1-indexed line to centre the window near. ``None``
            means line 1 (top of file).
        """
        if not path:
            raise ValueError("path must be non-empty")
        resolved = self._canonical_path(path)
        if not Path(resolved).exists():
            raise FileNotFoundError(resolved)
        self.registry[KEY_CURRENT_FILE] = resolved
        target_line = max(1, int(line)) if line is not None else 1
        self._set_first_line_for_target(resolved, target_line)
        return self.view()

    def close(self) -> None:
        """Clear the current file (and its window position)."""
        self.registry.pop(KEY_CURRENT_FILE, default=None)
        self.registry.pop(KEY_FIRST_LINE, default=None)

    def current_file(self) -> Optional[str]:
        """Return the absolute path of the open file (or ``None``)."""
        v = self.registry.get(KEY_CURRENT_FILE)
        return str(v) if isinstance(v, str) and v else None

    # ----- navigation ---------------------------------------------------

    def goto(self, line: int) -> WindowView:
        """Jump to a 1-indexed line in the current file."""
        path = self._require_current_file()
        line = max(1, int(line))
        self._set_first_line_for_target(path, line)
        return self.view()

    def scroll_down(self, n_lines: Optional[int] = None) -> WindowView:
        """Advance the window by ``n_lines`` (defaults to one window
        minus the overlap)."""
        path = self._require_current_file()
        window = self._window_lines()
        overlap = self._overlap_lines()
        step = int(n_lines) if n_lines is not None else (window - overlap)
        first = self._first_line_0indexed()
        total = self._line_count(path)
        new_first = self._clamp_first_line(first + step, total, window)
        self.registry[KEY_FIRST_LINE] = new_first
        return self.view()

    def scroll_up(self, n_lines: Optional[int] = None) -> WindowView:
        """Scroll backward by ``n_lines`` (defaults to one window
        minus the overlap)."""
        path = self._require_current_file()
        window = self._window_lines()
        overlap = self._overlap_lines()
        step = int(n_lines) if n_lines is not None else (window - overlap)
        first = self._first_line_0indexed()
        total = self._line_count(path)
        new_first = self._clamp_first_line(first - step, total, window)
        self.registry[KEY_FIRST_LINE] = new_first
        return self.view()

    # ----- rendering ----------------------------------------------------

    def view(
        self,
        *,
        line_numbers: bool = True,
        status_line: bool = True,
        pre_post_line: bool = True,
    ) -> WindowView:
        """Render the current window as a :class:`WindowView`."""
        path = self.current_file()
        if path is None:
            return WindowView(
                path="",
                text="",
                first_line=0,
                last_line=0,
                total_lines=0,
                lines_above=0,
                lines_below=0,
            )
        text_lines = self._read_lines(path)
        total = len(text_lines)
        first = self._first_line_0indexed()
        window = self._window_lines()
        last_excl = min(first + window, total)
        slice_lines = text_lines[first:last_excl]
        return self._render(
            path=path,
            slice_lines=slice_lines,
            first=first,
            last_excl=last_excl,
            total=total,
            line_numbers=line_numbers,
            status_line=status_line,
            pre_post_line=pre_post_line,
        )

    def view_with_semantic_expansion(
        self,
        *,
        max_added_lines: int = DEFAULT_MAX_ADDED_LINES,
        suffix: Optional[str] = None,
        line_numbers: bool = True,
        status_line: bool = True,
        pre_post_line: bool = True,
    ) -> WindowView:
        """Render the window expanded to the nearest semantic boundary
        via :class:`WindowExpander` (T5).

        ``suffix`` overrides the file-extension lookup; default uses
        the open file's suffix.
        """
        path = self.current_file()
        if path is None:
            return self.view(
                line_numbers=line_numbers,
                status_line=status_line,
                pre_post_line=pre_post_line,
            )
        text_lines = self._read_lines(path)
        total = len(text_lines)
        first_0 = self._first_line_0indexed()
        window = self._window_lines()
        last_excl = min(first_0 + window, total)
        if total == 0:
            return self._render(
                path=path,
                slice_lines=[],
                first=0,
                last_excl=0,
                total=0,
                line_numbers=line_numbers,
                status_line=status_line,
                pre_post_line=pre_post_line,
            )
        expander = WindowExpander(
            suffix=(suffix or Path(path).suffix or None),
        )
        result = expander.expand_window(
            text_lines,
            start=first_0 + 1,
            stop=last_excl,
            max_added_lines=max_added_lines,
        )
        # Convert back to 0-indexed slice bounds.
        new_first_0 = max(0, result.start - 1)
        new_last_excl = min(total, result.stop)
        slice_lines = text_lines[new_first_0:new_last_excl]
        return self._render(
            path=path,
            slice_lines=slice_lines,
            first=new_first_0,
            last_excl=new_last_excl,
            total=total,
            line_numbers=line_numbers,
            status_line=status_line,
            pre_post_line=pre_post_line,
        )

    # ----- internals ----------------------------------------------------

    def _require_current_file(self) -> str:
        path = self.current_file()
        if path is None:
            raise RuntimeError("No file is currently open (call open() first)")
        return path

    def _window_lines(self) -> int:
        return max(1, int(self.registry.get(KEY_WINDOW, self._default_window)))

    def _overlap_lines(self) -> int:
        return max(0, int(self.registry.get(KEY_OVERLAP, self._default_overlap)))

    def _first_line_0indexed(self) -> int:
        return max(0, int(self.registry.get(KEY_FIRST_LINE, 0)))

    def _set_first_line_for_target(self, path: str, target_1: int) -> None:
        window = self._window_lines()
        total = self._line_count(path)
        # Place the target ~1/6 down the new window (SWE-Agent's
        # `offset_multiplier`).
        offset = int(round(window * GOTO_OFFSET_MULTIPLIER))
        proposed = (target_1 - 1) - offset
        first = self._clamp_first_line(proposed, total, window)
        self.registry[KEY_FIRST_LINE] = first

    @staticmethod
    def _clamp_first_line(value: int, total_lines: int, window: int) -> int:
        if total_lines <= window:
            return 0
        upper = total_lines - window
        return max(0, min(int(value), upper))

    @staticmethod
    def _canonical_path(path: str) -> str:
        try:
            return str(Path(path).expanduser().resolve())
        except (OSError, RuntimeError):
            return str(path)

    @staticmethod
    def _read_lines(path: str) -> list[str]:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        # splitlines() drops the trailing newline character so the
        # rendered window doesn't double-space.
        return text.splitlines()

    @staticmethod
    def _line_count(path: str) -> int:
        try:
            return sum(1 for _ in Path(path).open("r", encoding="utf-8", errors="replace"))
        except OSError:
            return 0

    @staticmethod
    def _render(
        *,
        path: str,
        slice_lines: list[str],
        first: int,
        last_excl: int,
        total: int,
        line_numbers: bool,
        status_line: bool,
        pre_post_line: bool,
    ) -> WindowView:
        out_lines: list[str] = []
        if status_line:
            out_lines.append(f"[File: {path} ({total} lines total)]")
        if pre_post_line:
            above = first
            if above > 0:
                out_lines.append(f"({above} more lines above)")
        for i, content in enumerate(slice_lines):
            line_number_1 = first + 1 + i
            if line_numbers:
                out_lines.append(f"{line_number_1}:{content}")
            else:
                out_lines.append(content)
        if pre_post_line:
            below = total - last_excl
            if below > 0:
                out_lines.append(f"({below} more lines below)")
        rendered = os.linesep.join(out_lines)
        return WindowView(
            path=path,
            text=rendered,
            first_line=first + 1 if slice_lines else 0,
            last_line=last_excl if slice_lines else 0,
            total_lines=total,
            lines_above=first,
            lines_below=max(0, total - last_excl),
        )


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------


def get_window_state(
    session_id: str,
    *,
    registry: Optional[SessionRegistry] = None,
    default_window: int = DEFAULT_WINDOW_LINES,
    default_overlap: int = DEFAULT_OVERLAP_LINES,
) -> WindowState:
    """Return a :class:`WindowState` for ``session_id``.

    Uses :func:`get_session_registry` unless ``registry`` is passed
    (for tests). The first call materialises WINDOW + OVERLAP keys
    in the registry; subsequent calls re-use the existing values.
    """
    if registry is None:
        registry = get_session_registry(session_id)
    return WindowState(
        registry=registry,
        default_window=default_window,
        default_overlap=default_overlap,
    )


__all__ = [
    "DEFAULT_OVERLAP_LINES",
    "DEFAULT_WINDOW_LINES",
    "GOTO_OFFSET_MULTIPLIER",
    "KEY_CURRENT_FILE",
    "KEY_FIRST_LINE",
    "KEY_OVERLAP",
    "KEY_WINDOW",
    "WindowState",
    "WindowView",
    "get_window_state",
]
