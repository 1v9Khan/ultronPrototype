"""Expand edit/view windows to natural semantic boundaries.

Direct port of SWE-Agent's
``tools/edit_anthropic/bin/str_replace_editor:WindowExpander``
(MIT, Yang et al. 2024). The algorithm walks outward from a line
range, scoring each candidate boundary line by how "natural" it is
to break there. The highest-scoring line within a budget wins.

For ultron the helper is used wherever a narration / display surface
quotes a code window:

* The architect narrator (T5 Phase 2) speaks a plan that often
  references "the function around line N" -- expanding to the
  enclosing function gives the audience a coherent unit instead of
  arbitrary ±4 lines.
* The completion narrator can describe a freshly-edited region by
  the smallest enclosing class/function the change touched.
* The supervisor's enriched-context dispatch (T7) can paste a
  function instead of a fragment when handing context to Claude.
* The eventual T1 lint-revert path quotes the would-be window AND
  the original window; both benefit from semantic expansion so the
  model sees a complete construct rather than a clip.

The expander is direction-aware:

* Extending DOWNWARD (toward later lines), the boundary is the
  line BEFORE the next def/class/decorator so the new declaration
  is its own block.
* Extending UPWARD (toward earlier lines), the boundary is the
  def/class/decorator line itself so the enclosing construct is
  fully included.

Per-suffix scoring rules cover Python, JavaScript/TypeScript, Go,
and a generic fallback. Tree-sitter is available elsewhere in the
codebase for more precise enclosing-function lookup; this regex
scorer is the cheap fast-path that needs no language pack.

Pure, side-effect-free, no I/O. Sub-millisecond on typical windows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

#: Default max additional lines walked from each boundary. Mirrors
#: SWE-Agent's default. Tunable per call.
DEFAULT_MAX_ADDED_LINES: int = 30

#: Score returned for a single blank line.
SCORE_BLANK_LINE: int = 1

#: Score returned for two consecutive blank lines.
SCORE_DOUBLE_BLANK: int = 2

#: Score returned for a def/class/decorator boundary.
SCORE_SEMANTIC_BOUNDARY: int = 3

#: Score returned at file edges (first / last line).
SCORE_FILE_EDGE: int = 3


# ---------------------------------------------------------------------------
# Per-language semantic patterns
# ---------------------------------------------------------------------------

_PYTHON_BOUNDARY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*def\s+"),
    re.compile(r"^\s*async\s+def\s+"),
    re.compile(r"^\s*class\s+"),
    re.compile(r"^\s*@"),
)

_JS_TS_BOUNDARY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*function\s+"),
    re.compile(r"^\s*async\s+function\s+"),
    re.compile(r"^\s*class\s+"),
    re.compile(r"^\s*export\s+"),
    re.compile(r"^\s*(?:const|let|var)\s+[A-Za-z_$][\w$]*\s*=\s*(?:async\s*)?\("),
)

_GO_BOUNDARY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*func\s+"),
    re.compile(r"^\s*type\s+"),
    re.compile(r"^\s*package\s+"),
)

_RUST_BOUNDARY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*fn\s+"),
    re.compile(r"^\s*impl\b"),
    re.compile(r"^\s*struct\s+"),
    re.compile(r"^\s*enum\s+"),
    re.compile(r"^\s*pub\s+(?:fn|struct|enum|trait)\s+"),
)

_JAVA_LIKE_BOUNDARY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?:public|private|protected|static|final|abstract|@\w)"),
    re.compile(r"^\s*class\s+"),
    re.compile(r"^\s*interface\s+"),
    re.compile(r"^\s*enum\s+"),
)

#: Map of file-extension suffix -> semantic-boundary regex set. The
#: helper looks up by extension; unknown suffixes fall through to the
#: blank-line / file-edge scoring only.
SEMANTIC_PATTERNS_BY_SUFFIX: dict[str, tuple[re.Pattern[str], ...]] = {
    ".py": _PYTHON_BOUNDARY_PATTERNS,
    ".pyi": _PYTHON_BOUNDARY_PATTERNS,
    ".js": _JS_TS_BOUNDARY_PATTERNS,
    ".mjs": _JS_TS_BOUNDARY_PATTERNS,
    ".cjs": _JS_TS_BOUNDARY_PATTERNS,
    ".ts": _JS_TS_BOUNDARY_PATTERNS,
    ".tsx": _JS_TS_BOUNDARY_PATTERNS,
    ".jsx": _JS_TS_BOUNDARY_PATTERNS,
    ".go": _GO_BOUNDARY_PATTERNS,
    ".rs": _RUST_BOUNDARY_PATTERNS,
    ".java": _JAVA_LIKE_BOUNDARY_PATTERNS,
    ".kt": _JAVA_LIKE_BOUNDARY_PATTERNS,
    ".cs": _JAVA_LIKE_BOUNDARY_PATTERNS,
}


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpansionResult:
    """Output of :meth:`WindowExpander.expand_window`.

    ``start`` / ``stop`` are 1-indexed line numbers (inclusive), matching
    SWE-Agent's convention. ``expanded_lines_above`` and
    ``expanded_lines_below`` count how many lines were added in each
    direction beyond the input range. ``reason_above`` /
    ``reason_below`` name which scoring rule fired.
    """

    start: int
    stop: int
    expanded_lines_above: int
    expanded_lines_below: int
    reason_above: str
    reason_below: str


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class WindowExpander:
    """Expand a line range to the nearest natural break above + below.

    Construct once per language family (or per-call with ``suffix``
    passed to :meth:`expand_window`). Pure -- no state beyond the
    chosen pattern set.

    Inputs:

    * ``lines`` -- the file split on ``"\\n"`` (or :func:`str.splitlines`),
      WITHOUT trailing newlines on each entry. Line numbers in
      ``start`` / ``stop`` are 1-indexed inclusive.
    * ``start`` / ``stop`` -- the input range (1-indexed inclusive,
      ``stop >= start``).
    * ``max_added_lines`` -- the budget in EACH direction. Default
      :data:`DEFAULT_MAX_ADDED_LINES`.
    * ``suffix`` -- file extension (e.g. ``.py``) selecting the
      semantic pattern set. ``None`` or unknown suffix uses blank-line
      / file-edge scoring only.

    Output: :class:`ExpansionResult` with the expanded ``start`` /
    ``stop``, the line deltas, and the reason each boundary chose.
    """

    def __init__(self, *, suffix: str | None = None) -> None:
        self.suffix = suffix.lower() if suffix else None

    # ------- public API -------------------------------------------------

    def expand_window(
        self,
        lines: Sequence[str],
        start: int,
        stop: int,
        *,
        max_added_lines: int = DEFAULT_MAX_ADDED_LINES,
        suffix: str | None = None,
    ) -> ExpansionResult:
        """Return an ExpansionResult covering at least [start, stop]."""
        if start < 1 or stop < 1:
            raise ValueError(
                f"start and stop must be >= 1 (got start={start}, stop={stop})"
            )
        if stop < start:
            raise ValueError(
                f"stop must be >= start (got start={start}, stop={stop})"
            )
        if max_added_lines < 0:
            raise ValueError(
                f"max_added_lines must be >= 0 (got {max_added_lines})"
            )
        suffix = (suffix or self.suffix or "").lower() or None
        patterns = SEMANTIC_PATTERNS_BY_SUFFIX.get(suffix, ()) if suffix else ()

        n = len(lines)
        # Clamp the input to the file bounds.
        if n == 0:
            return ExpansionResult(
                start=start,
                stop=stop,
                expanded_lines_above=0,
                expanded_lines_below=0,
                reason_above="empty_file",
                reason_below="empty_file",
            )
        clamped_start = max(1, min(start, n))
        clamped_stop = max(1, min(stop, n))
        if clamped_stop < clamped_start:
            clamped_stop = clamped_start

        new_start, reason_above = self._find_breakpoint(
            lines,
            current_line=clamped_start,
            direction=-1,
            max_added_lines=max_added_lines,
            patterns=patterns,
        )
        new_stop, reason_below = self._find_breakpoint(
            lines,
            current_line=clamped_stop,
            direction=1,
            max_added_lines=max_added_lines,
            patterns=patterns,
        )

        # Anti-shrinking guarantees: never move boundaries inward.
        if new_start > clamped_start:
            new_start = clamped_start
            reason_above = "no_outward_move"
        if new_stop < clamped_stop:
            new_stop = clamped_stop
            reason_below = "no_outward_move"

        return ExpansionResult(
            start=new_start,
            stop=new_stop,
            expanded_lines_above=max(0, clamped_start - new_start),
            expanded_lines_below=max(0, new_stop - clamped_stop),
            reason_above=reason_above,
            reason_below=reason_below,
        )

    # ------- scoring ----------------------------------------------------

    def _find_breakpoint(
        self,
        lines: Sequence[str],
        *,
        current_line: int,
        direction: int,
        max_added_lines: int,
        patterns: tuple[re.Pattern[str], ...],
    ) -> tuple[int, str]:
        """Walk outward in ``direction`` searching for the best boundary.

        Returns ``(line_1indexed, reason)``. ``direction`` is +1 (down)
        or -1 (up).
        """
        if direction not in (1, -1):
            raise ValueError(f"direction must be +1 or -1 (got {direction})")
        n = len(lines)
        if n == 0:
            return current_line, "empty_file"

        # Compute the walk range (1-indexed inclusive of both ends).
        if direction == 1:
            walk_start = current_line
            walk_stop = min(current_line + max_added_lines, n)
        else:
            walk_start = current_line
            walk_stop = max(current_line - max_added_lines, 1)
        step = direction

        best_score = -1
        best_line = current_line
        best_reason = "no_move"

        cursor = walk_start
        while True:
            score, reason = self._score_line(lines, cursor, n, patterns)
            if score > best_score:
                best_score = score
                best_line = cursor
                best_reason = reason
            if cursor == walk_stop:
                break
            cursor += step

        # When extending DOWN, prefer the line BEFORE the def/class so
        # the new declaration is its own block (per SWE-Agent).
        if (
            direction == 1
            and best_score == SCORE_SEMANTIC_BOUNDARY
            and best_line > current_line
        ):
            best_line = max(current_line, best_line - 1)
        return best_line, best_reason

    @staticmethod
    def _score_line(
        lines: Sequence[str],
        line_no: int,
        n_lines: int,
        patterns: tuple[re.Pattern[str], ...],
    ) -> tuple[int, str]:
        """Score a single 1-indexed line against the boundary rules.

        Highest score wins. Returns ``(score, reason)`` where reason
        names the rule for diagnostic output (``"def_class_decorator"``,
        ``"double_blank"``, ``"blank"``, ``"file_edge"``).
        """
        if line_no <= 1 or line_no >= n_lines:
            return SCORE_FILE_EDGE, "file_edge"
        text = lines[line_no - 1]  # 1-indexed -> 0-indexed
        next_text = lines[line_no] if line_no < n_lines else ""
        for pattern in patterns:
            if pattern.match(text):
                return SCORE_SEMANTIC_BOUNDARY, "def_class_decorator"
        stripped = text.strip()
        if not stripped:
            if not next_text.strip():
                return SCORE_DOUBLE_BLANK, "double_blank"
            return SCORE_BLANK_LINE, "blank"
        return 0, "no_match"


__all__ = [
    "DEFAULT_MAX_ADDED_LINES",
    "ExpansionResult",
    "SCORE_BLANK_LINE",
    "SCORE_DOUBLE_BLANK",
    "SCORE_FILE_EDGE",
    "SCORE_SEMANTIC_BOUNDARY",
    "SEMANTIC_PATTERNS_BY_SUFFIX",
    "WindowExpander",
]
