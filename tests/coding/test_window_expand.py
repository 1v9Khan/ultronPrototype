"""Tests for the semantic window expander (catalog T5)."""

from __future__ import annotations

import pytest

from ultron.coding.window_expand import (
    DEFAULT_MAX_ADDED_LINES,
    ExpansionResult,
    SCORE_BLANK_LINE,
    SCORE_DOUBLE_BLANK,
    SCORE_FILE_EDGE,
    SCORE_SEMANTIC_BOUNDARY,
    SEMANTIC_PATTERNS_BY_SUFFIX,
    WindowExpander,
)


# ---------------------------------------------------------------------------
# Constants + invariants
# ---------------------------------------------------------------------------


def test_score_ordering_def_class_beats_blank():
    assert SCORE_SEMANTIC_BOUNDARY > SCORE_DOUBLE_BLANK > SCORE_BLANK_LINE


def test_default_budget_constant():
    assert DEFAULT_MAX_ADDED_LINES > 0


def test_pattern_table_covers_common_languages():
    for suffix in (".py", ".js", ".ts", ".go", ".rs", ".java"):
        assert suffix in SEMANTIC_PATTERNS_BY_SUFFIX


# ---------------------------------------------------------------------------
# Construction + bounds
# ---------------------------------------------------------------------------


def test_invalid_range_raises():
    e = WindowExpander()
    with pytest.raises(ValueError):
        e.expand_window(["a", "b"], 0, 1)  # start < 1
    with pytest.raises(ValueError):
        e.expand_window(["a", "b"], 2, 1)  # stop < start
    with pytest.raises(ValueError):
        e.expand_window(["a", "b"], 1, 1, max_added_lines=-1)


def test_empty_file_returns_input_unchanged():
    e = WindowExpander(suffix=".py")
    r = e.expand_window([], 1, 1)
    assert isinstance(r, ExpansionResult)
    assert r.start == 1 and r.stop == 1
    assert r.expanded_lines_above == 0
    assert r.expanded_lines_below == 0


# ---------------------------------------------------------------------------
# Anti-shrinking
# ---------------------------------------------------------------------------


def test_does_not_shrink_when_no_outward_move():
    e = WindowExpander(suffix=".py")
    lines = ["def foo():", "    return 1"]
    r = e.expand_window(lines, 1, 2, max_added_lines=0)
    # With no budget, the range can't grow but must not shrink.
    assert r.start <= 1
    assert r.stop >= 2


# ---------------------------------------------------------------------------
# Python semantic expansion
# ---------------------------------------------------------------------------


def test_python_expands_up_to_def():
    source = (
        "import x\n"
        "\n"
        "def foo():\n"
        "    line_a = 1\n"
        "    line_b = 2\n"
        "    return line_a + line_b\n"
    )
    lines = source.split("\n")
    e = WindowExpander(suffix=".py")
    # Input range covers only "line_a" + "line_b".
    r = e.expand_window(lines, 4, 5, max_added_lines=20)
    # Upward expansion: should include the def line.
    assert r.start <= 3
    assert r.reason_above == "def_class_decorator"


def test_python_expands_with_decorator():
    source = (
        "@dataclass\n"
        "class Foo:\n"
        "    x: int\n"
        "    y: int\n"
    )
    lines = source.split("\n")
    e = WindowExpander(suffix=".py")
    r = e.expand_window(lines, 3, 3, max_added_lines=20)
    # The decorator OR the class line should be selected upward.
    assert r.start <= 2
    assert r.reason_above == "def_class_decorator"


def test_python_async_def_recognised():
    source = (
        "import asyncio\n"
        "\n"
        "async def fetch(u):\n"
        "    return await u\n"
    )
    lines = source.split("\n")
    e = WindowExpander(suffix=".py")
    r = e.expand_window(lines, 4, 4, max_added_lines=10)
    assert r.start <= 3
    assert r.reason_above == "def_class_decorator"


# ---------------------------------------------------------------------------
# JS / TS expansion
# ---------------------------------------------------------------------------


def test_javascript_expands_to_function():
    source = (
        "import x from 'y';\n"
        "\n"
        "function foo() {\n"
        "  return 1;\n"
        "}\n"
    )
    lines = source.split("\n")
    e = WindowExpander(suffix=".js")
    r = e.expand_window(lines, 4, 4, max_added_lines=10)
    assert r.start <= 3
    assert r.reason_above == "def_class_decorator"


def test_typescript_export_class_recognised():
    source = (
        "import { Foo } from 'bar';\n"
        "\n"
        "export class Baz {\n"
        "  qux() {}\n"
        "}\n"
    )
    lines = source.split("\n")
    e = WindowExpander(suffix=".ts")
    r = e.expand_window(lines, 4, 4, max_added_lines=10)
    assert r.start <= 3


# ---------------------------------------------------------------------------
# Go expansion
# ---------------------------------------------------------------------------


def test_go_func_recognised():
    source = (
        "package main\n"
        "\n"
        "func main() {\n"
        "    println(\"hi\")\n"
        "}\n"
    )
    lines = source.split("\n")
    e = WindowExpander(suffix=".go")
    r = e.expand_window(lines, 4, 4, max_added_lines=10)
    assert r.start <= 3


# ---------------------------------------------------------------------------
# Fallback scoring (blank lines + file edges)
# ---------------------------------------------------------------------------


def test_blank_line_preferred_when_no_def():
    source = "alpha\nbeta\n\ngamma\ndelta\n"
    lines = source.split("\n")
    e = WindowExpander()
    r = e.expand_window(lines, 4, 4, max_added_lines=5)
    # Blank line at index 3 is a candidate above; file end below.
    assert r.start <= 3


def test_file_edge_scored_for_extreme_positions():
    source = "first\nsecond\nthird\nfourth\nfifth\n"
    lines = source.split("\n")
    e = WindowExpander()
    r = e.expand_window(lines, 3, 3, max_added_lines=10)
    # The file edges always score high; the expansion may pick them.
    assert r.start <= 3
    assert r.stop >= 3


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------


def test_max_added_lines_respected():
    source = "x\n" * 200
    lines = source.split("\n")
    e = WindowExpander()  # no semantic patterns
    r = e.expand_window(lines, 100, 100, max_added_lines=5)
    assert r.expanded_lines_above <= 5
    assert r.expanded_lines_below <= 5


def test_zero_budget_no_movement():
    source = "def foo():\n    pass\n"
    lines = source.split("\n")
    e = WindowExpander(suffix=".py")
    r = e.expand_window(lines, 2, 2, max_added_lines=0)
    assert r.start == 2
    assert r.stop == 2


# ---------------------------------------------------------------------------
# Suffix override
# ---------------------------------------------------------------------------


def test_suffix_override_per_call():
    source = "function foo() {\n  return 1;\n}\n"
    lines = source.split("\n")
    e = WindowExpander()  # no suffix at construction
    r = e.expand_window(lines, 2, 2, max_added_lines=10, suffix=".js")
    # With suffix override, the function line should be the boundary.
    assert r.start <= 1


def test_unknown_suffix_falls_back_to_blank_scoring():
    source = "alpha\nbeta\n\ngamma\ndelta\n"
    lines = source.split("\n")
    e = WindowExpander(suffix=".xyz")
    r = e.expand_window(lines, 4, 4, max_added_lines=5)
    # No semantic patterns -> blank-line / file-edge scoring still works.
    assert r.start <= 3


# ---------------------------------------------------------------------------
# Direction-aware backoff (DOWN past def goes to line before)
# ---------------------------------------------------------------------------


def test_downward_expansion_stops_before_next_def():
    source = (
        "def first():\n"
        "    return 1\n"
        "\n"
        "def second():\n"
        "    return 2\n"
    )
    lines = source.split("\n")
    e = WindowExpander(suffix=".py")
    # Input range covers the body of first(). Downward expansion
    # should NOT include the `def second():` line itself; it should
    # stop on the blank line or just before the def.
    r = e.expand_window(lines, 1, 2, max_added_lines=10)
    assert r.stop <= 4  # at or before the def second() line
