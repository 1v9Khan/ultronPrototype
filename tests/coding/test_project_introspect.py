"""Tests for ultron.coding.project_introspect."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ultron.coding.project_introspect import (
    ENTRY_POINT_FILENAMES,
    LANGUAGE_BY_EXT,
    MARKER_FILES,
    SKIP_DIRECTORIES,
    ProjectSnapshot,
    invalidate_snapshot_cache,
    snapshot,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def python_project(tmp_path: Path) -> Path:
    """A small Python project: app.py + helper + README."""
    (tmp_path / "app.py").write_text(
        "def main():\n    print('hi')\n\nif __name__ == '__main__':\n    main()\n",
    )
    (tmp_path / "helpers.py").write_text(
        "def util():\n    return 42\n",
    )
    (tmp_path / "README.md").write_text("# Sample\n\nA test project.\n")
    (tmp_path / "requirements.txt").write_text("flask\n")
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "module.py").write_text("X = 1\n")
    return tmp_path


@pytest.fixture
def js_project(tmp_path: Path) -> Path:
    """A small JS project: index.js + package.json."""
    (tmp_path / "index.js").write_text("console.log('hi')\n")
    (tmp_path / "package.json").write_text('{"name": "test"}\n')
    (tmp_path / "README.md").write_text("test\n")
    return tmp_path


@pytest.fixture
def empty_project(tmp_path: Path) -> Path:
    """An empty project directory."""
    return tmp_path


# ---------------------------------------------------------------------------
# snapshot -- basic
# ---------------------------------------------------------------------------


def test_snapshot_returns_proper_dataclass(python_project: Path) -> None:
    snap = snapshot(python_project, use_cache=False)
    assert isinstance(snap, ProjectSnapshot)
    assert snap.project_path == python_project.resolve()
    assert snap.project_name == python_project.name
    assert snap.elapsed_ms >= 0.0


def test_snapshot_detects_python_language(python_project: Path) -> None:
    snap = snapshot(python_project, use_cache=False)
    assert snap.dominant_language == "python"
    assert "python" in snap.languages


def test_snapshot_detects_javascript_language(js_project: Path) -> None:
    snap = snapshot(js_project, use_cache=False)
    assert snap.dominant_language == "javascript"


def test_snapshot_walks_files(python_project: Path) -> None:
    snap = snapshot(python_project, use_cache=False)
    names = {f.relative_path for f in snap.files}
    assert "app.py" in names
    assert "helpers.py" in names
    assert "README.md" in names
    assert "src/module.py" in names


def test_snapshot_finds_entry_points(python_project: Path) -> None:
    snap = snapshot(python_project, use_cache=False)
    names = {p.name for p in snap.entry_points}
    assert "app.py" in names


def test_snapshot_detects_markers(python_project: Path) -> None:
    snap = snapshot(python_project, use_cache=False)
    assert "requirements.txt" in snap.markers


def test_snapshot_empty_project(empty_project: Path) -> None:
    snap = snapshot(empty_project, use_cache=False)
    assert snap.file_count == 0
    assert snap.dominant_language == ""
    assert snap.entry_points == []


def test_snapshot_nonexistent_path(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    snap = snapshot(missing, use_cache=False)
    assert snap.project_name == "does_not_exist"
    assert snap.file_count == 0


# ---------------------------------------------------------------------------
# snapshot -- limits + skip directories
# ---------------------------------------------------------------------------


def test_snapshot_respects_max_files(python_project: Path) -> None:
    # Add 50 extra files, then cap at 3.
    for i in range(50):
        (python_project / f"extra_{i}.txt").write_text(str(i))
    snap = snapshot(python_project, max_files=3, use_cache=False)
    assert snap.file_count == 3
    assert snap.truncated is True


def test_snapshot_skips_node_modules(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("ok\n")
    nm = tmp_path / "node_modules"
    nm.mkdir()
    (nm / "garbage.js").write_text("noise\n")
    snap = snapshot(tmp_path, use_cache=False)
    paths = {f.relative_path for f in snap.files}
    assert "app.py" in paths
    assert not any("node_modules" in p for p in paths)
    assert not any("node_modules" in d for d in snap.directories)


def test_snapshot_skips_pycache(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("ok\n")
    pc = tmp_path / "__pycache__"
    pc.mkdir()
    (pc / "garbage.pyc").write_text("noise\n")
    snap = snapshot(tmp_path, use_cache=False)
    paths = {f.relative_path for f in snap.files}
    assert not any("__pycache__" in p for p in paths)


def test_snapshot_respects_max_depth(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c" / "d" / "e" / "f" / "g"
    nested.mkdir(parents=True)
    (nested / "deep.py").write_text("ok\n")
    snap = snapshot(tmp_path, max_depth=2, use_cache=False)
    # The deep file should not be picked up.
    rels = {f.relative_path for f in snap.files}
    assert not any("g/deep.py" in r for r in rels)


# ---------------------------------------------------------------------------
# AST integration
# ---------------------------------------------------------------------------


def test_snapshot_parses_python_ast(python_project: Path) -> None:
    snap = snapshot(python_project, use_cache=False)
    # ast_metadata is keyed by relative path
    assert "app.py" in snap.ast_metadata
    md = snap.ast_metadata["app.py"]
    assert md.syntax_valid is True
    assert any("main" in f for f in md.functions_defined)


def test_snapshot_skips_ast_when_cap_is_zero(python_project: Path) -> None:
    snap = snapshot(python_project, ast_file_cap=0, use_cache=False)
    assert snap.ast_metadata == {}


# ---------------------------------------------------------------------------
# render_tree_summary
# ---------------------------------------------------------------------------


def test_render_tree_summary_returns_string(python_project: Path) -> None:
    snap = snapshot(python_project, use_cache=False)
    tree = snap.render_tree_summary()
    assert isinstance(tree, str)
    assert python_project.name in tree
    assert "app.py" in tree


def test_render_tree_summary_caps_lines(python_project: Path) -> None:
    for i in range(40):
        (python_project / f"file_{i}.py").write_text("x = 1\n")
    snap = snapshot(python_project, use_cache=False)
    tree = snap.render_tree_summary(max_lines=10)
    # Allow a couple of header lines + the "+N more" trailer.
    assert tree.count("\n") <= 15
    assert "more files" in tree


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_cache_returns_same_snapshot(python_project: Path) -> None:
    invalidate_snapshot_cache()
    s1 = snapshot(python_project, use_cache=True)
    s2 = snapshot(python_project, use_cache=True)
    # Same object identity = cache hit.
    assert s1 is s2


def test_cache_use_false_bypasses(python_project: Path) -> None:
    invalidate_snapshot_cache()
    s1 = snapshot(python_project, use_cache=True)
    s2 = snapshot(python_project, use_cache=False)
    assert s1 is not s2


def test_cache_invalidate_clears(python_project: Path) -> None:
    invalidate_snapshot_cache()
    s1 = snapshot(python_project, use_cache=True)
    invalidate_snapshot_cache(python_project)
    s2 = snapshot(python_project, use_cache=True)
    assert s1 is not s2


def test_cache_invalidate_global(python_project: Path) -> None:
    invalidate_snapshot_cache()
    snapshot(python_project, use_cache=True)
    invalidate_snapshot_cache()
    # Subsequent call should re-walk; we can't easily assert identity
    # but we can assert it succeeds.
    s2 = snapshot(python_project, use_cache=True)
    assert s2.file_count > 0


# ---------------------------------------------------------------------------
# Sanity check on constants
# ---------------------------------------------------------------------------


def test_language_by_ext_covers_common_extensions() -> None:
    assert LANGUAGE_BY_EXT[".py"] == "python"
    assert LANGUAGE_BY_EXT[".js"] == "javascript"
    assert LANGUAGE_BY_EXT[".rs"] == "rust"


def test_marker_files_includes_python_signals() -> None:
    assert "pyproject.toml" in MARKER_FILES
    assert "requirements.txt" in MARKER_FILES


def test_entry_point_filenames_include_python_main() -> None:
    assert "main.py" in ENTRY_POINT_FILENAMES
    assert "app.py" in ENTRY_POINT_FILENAMES
    assert "manage.py" in ENTRY_POINT_FILENAMES


def test_skip_directories_includes_common_caches() -> None:
    assert "__pycache__" in SKIP_DIRECTORIES
    assert "node_modules" in SKIP_DIRECTORIES
    assert ".venv" in SKIP_DIRECTORIES
