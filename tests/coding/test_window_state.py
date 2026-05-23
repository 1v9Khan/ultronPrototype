"""Tests for the windowed-file state machine (catalog T4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ultron.coding.session_registry import (
    SessionRegistry,
    reset_session_registries_for_testing,
)
from ultron.coding.window_state import (
    DEFAULT_OVERLAP_LINES,
    DEFAULT_WINDOW_LINES,
    GOTO_OFFSET_MULTIPLIER,
    KEY_CURRENT_FILE,
    KEY_FIRST_LINE,
    KEY_OVERLAP,
    KEY_WINDOW,
    WindowState,
    WindowView,
    get_window_state,
)


@pytest.fixture
def reg(tmp_path: Path) -> SessionRegistry:
    reset_session_registries_for_testing()
    return SessionRegistry(session_id="window-state-test", root=tmp_path)


@pytest.fixture
def state(reg: SessionRegistry) -> WindowState:
    return WindowState(registry=reg, default_window=10, default_overlap=2)


@pytest.fixture(autouse=True)
def _cleanup() -> None:
    yield
    reset_session_registries_for_testing()


def _make_file(tmp_path: Path, name: str, n_lines: int) -> Path:
    p = tmp_path / name
    p.write_text("\n".join(f"line {i+1}" for i in range(n_lines)), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Constants + construction
# ---------------------------------------------------------------------------


def test_constants_match_swe_agent_defaults():
    assert DEFAULT_WINDOW_LINES == 100
    assert DEFAULT_OVERLAP_LINES == 2
    assert 0 < GOTO_OFFSET_MULTIPLIER < 1


def test_constructor_validates_window(reg: SessionRegistry):
    with pytest.raises(ValueError):
        WindowState(registry=reg, default_window=0)
    with pytest.raises(ValueError):
        WindowState(registry=reg, default_window=10, default_overlap=10)
    with pytest.raises(ValueError):
        WindowState(registry=reg, default_window=10, default_overlap=-1)


def test_constructor_seeds_window_and_overlap_in_registry(reg: SessionRegistry):
    WindowState(registry=reg, default_window=42, default_overlap=5)
    assert reg[KEY_WINDOW] == 42
    assert reg[KEY_OVERLAP] == 5


def test_existing_window_value_preserved(reg: SessionRegistry):
    reg[KEY_WINDOW] = 99
    WindowState(registry=reg, default_window=10)
    assert reg[KEY_WINDOW] == 99


# ---------------------------------------------------------------------------
# open / close / current_file
# ---------------------------------------------------------------------------


def test_open_empty_path_rejected(state: WindowState):
    with pytest.raises(ValueError):
        state.open("")


def test_open_missing_file_raises(state: WindowState, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        state.open(str(tmp_path / "missing.py"))


def test_open_existing_file_sets_current(state: WindowState, tmp_path: Path):
    p = _make_file(tmp_path, "x.py", 20)
    view = state.open(str(p))
    assert state.current_file() == str(p.resolve())
    assert view.path == str(p.resolve())
    assert view.first_line == 1


def test_open_with_line_sets_window_position(state: WindowState, tmp_path: Path):
    p = _make_file(tmp_path, "x.py", 200)
    view = state.open(str(p), line=80)
    # 1/6 offset means the target appears ~1/6 down the 10-line window,
    # so first_line should be 80 - round(10/6) = 80 - 2 = 78. Tolerate
    # rounding ±1.
    assert 77 <= view.first_line <= 79
    assert 80 in range(view.first_line, view.last_line + 1)


def test_close_clears_state(state: WindowState, tmp_path: Path):
    p = _make_file(tmp_path, "x.py", 20)
    state.open(str(p))
    state.close()
    assert state.current_file() is None


# ---------------------------------------------------------------------------
# goto / scroll
# ---------------------------------------------------------------------------


def test_goto_without_open_raises(state: WindowState):
    with pytest.raises(RuntimeError):
        state.goto(5)


def test_goto_clamps_to_file_bounds(state: WindowState, tmp_path: Path):
    p = _make_file(tmp_path, "x.py", 20)
    state.open(str(p))
    view = state.goto(9999)  # beyond EOF
    assert view.last_line <= 20


def test_scroll_down_advances_window(state: WindowState, tmp_path: Path):
    p = _make_file(tmp_path, "x.py", 50)
    state.open(str(p))
    before = state.view()
    after = state.scroll_down()
    # window=10, overlap=2 -> step=8
    assert after.first_line == before.first_line + 8


def test_scroll_up_retreats_window(state: WindowState, tmp_path: Path):
    p = _make_file(tmp_path, "x.py", 50)
    state.open(str(p), line=30)
    before = state.view()
    after = state.scroll_up()
    assert after.first_line == max(1, before.first_line - 8)


def test_scroll_clamped_at_file_start(state: WindowState, tmp_path: Path):
    p = _make_file(tmp_path, "x.py", 30)
    state.open(str(p))
    view = state.scroll_up()
    assert view.first_line >= 1


def test_scroll_clamped_at_file_end(state: WindowState, tmp_path: Path):
    p = _make_file(tmp_path, "x.py", 30)
    state.open(str(p), line=25)
    # Scroll down past EOF: last_line should pin at total.
    state.scroll_down()
    state.scroll_down()
    view = state.view()
    assert view.last_line <= 30


# ---------------------------------------------------------------------------
# view rendering
# ---------------------------------------------------------------------------


def test_view_includes_status_line(state: WindowState, tmp_path: Path):
    p = _make_file(tmp_path, "x.py", 30)
    state.open(str(p))
    view = state.view()
    assert "[File: " in view.text
    assert "(30 lines total)" in view.text


def test_view_includes_line_numbers(state: WindowState, tmp_path: Path):
    p = _make_file(tmp_path, "x.py", 30)
    state.open(str(p))
    view = state.view()
    assert "1:line 1" in view.text


def test_view_pre_post_line_annotations(state: WindowState, tmp_path: Path):
    p = _make_file(tmp_path, "x.py", 50)
    state.open(str(p), line=25)
    view = state.view()
    assert "more lines above" in view.text or "more lines below" in view.text


def test_view_no_file_returns_empty(state: WindowState):
    view = state.view()
    assert view.text == ""
    assert view.path == ""


def test_view_without_line_numbers_strips_prefix(state: WindowState, tmp_path: Path):
    p = _make_file(tmp_path, "x.py", 30)
    state.open(str(p))
    view = state.view(line_numbers=False)
    # First content line should be raw, no "1:" prefix.
    assert "1:line 1" not in view.text
    assert "line 1" in view.text


# ---------------------------------------------------------------------------
# Semantic-expansion view
# ---------------------------------------------------------------------------


def test_view_with_semantic_expansion_extends_upward(
    state: WindowState, tmp_path: Path
):
    p = tmp_path / "x.py"
    p.write_text(
        "import sys\n"
        "\n"
        "def foo():\n"
        "    a = 1\n"
        "    b = 2\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    state.open(str(p), line=4)  # body of foo()
    view = state.view_with_semantic_expansion(max_added_lines=20)
    assert "def foo" in view.text


def test_view_with_semantic_expansion_no_file_falls_back(state: WindowState):
    view = state.view_with_semantic_expansion()
    assert view.path == ""


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------


def test_get_window_state_uses_registry(tmp_path: Path):
    reset_session_registries_for_testing()
    reg = SessionRegistry(session_id="factory", root=tmp_path)
    state = get_window_state("factory", registry=reg)
    assert isinstance(state, WindowState)
    assert state.registry is reg


def test_get_window_state_seeds_window_and_overlap(tmp_path: Path):
    reset_session_registries_for_testing()
    reg = SessionRegistry(session_id="factory", root=tmp_path)
    state = get_window_state("factory", registry=reg)
    assert KEY_WINDOW in reg
    assert KEY_OVERLAP in reg


# ---------------------------------------------------------------------------
# WindowView dataclass
# ---------------------------------------------------------------------------


def test_window_view_is_frozen():
    v = WindowView(
        path="/x", text="", first_line=0, last_line=0,
        total_lines=0, lines_above=0, lines_below=0,
    )
    with pytest.raises(Exception):
        v.path = "/y"  # type: ignore[misc]
