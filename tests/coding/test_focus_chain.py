"""Tests for ultron.coding.focus_chain."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ultron.coding import focus_chain as fc


# ---------------------------------------------------------------------------
# parse_focus_chain / render_focus_chain
# ---------------------------------------------------------------------------

class TestParseAndRender:
    def test_empty_text(self) -> None:
        assert fc.parse_focus_chain("") == []

    def test_basic_items(self) -> None:
        text = "- [x] done item\n- [ ] pending\n- [X] also done\n"
        items = fc.parse_focus_chain(text)
        assert len(items) == 3
        assert items[0].done is True
        assert items[1].done is False
        assert items[2].done is True

    def test_non_item_lines_ignored(self) -> None:
        text = "# Plan\nsome notes\n- [ ] task\n"
        items = fc.parse_focus_chain(text)
        assert len(items) == 1
        assert items[0].text == "task"

    def test_indented_items_accepted(self) -> None:
        text = "  - [ ] nested item\n"
        items = fc.parse_focus_chain(text)
        assert len(items) == 1
        assert items[0].text == "nested item"

    def test_render_round_trip(self) -> None:
        items = [
            fc.FocusItem(text="alpha", done=False),
            fc.FocusItem(text="beta", done=True),
        ]
        rendered = fc.render_focus_chain(items)
        re_parsed = fc.parse_focus_chain(rendered)
        assert [i.text for i in re_parsed] == ["alpha", "beta"]
        assert [i.done for i in re_parsed] == [False, True]

    def test_render_with_header(self) -> None:
        items = [fc.FocusItem(text="x", done=False)]
        rendered = fc.render_focus_chain(items, header="# Plan\n")
        assert rendered.startswith("# Plan")
        assert "- [ ] x" in rendered


# ---------------------------------------------------------------------------
# diff_focus_chains
# ---------------------------------------------------------------------------

class TestDiff:
    def test_no_change(self) -> None:
        items = [fc.FocusItem(text="a", done=False)]
        diff = fc.diff_focus_chains(items, items)
        assert diff.is_empty

    def test_added(self) -> None:
        old = [fc.FocusItem(text="a", done=False)]
        new = old + [fc.FocusItem(text="b", done=False)]
        diff = fc.diff_focus_chains(old, new)
        assert diff.added == ("b",)
        assert not diff.is_empty

    def test_removed(self) -> None:
        old = [fc.FocusItem(text="a", done=False), fc.FocusItem(text="b", done=False)]
        new = [fc.FocusItem(text="a", done=False)]
        diff = fc.diff_focus_chains(old, new)
        assert diff.removed == ("b",)

    def test_completed_and_uncompleted(self) -> None:
        old = [
            fc.FocusItem(text="a", done=False),
            fc.FocusItem(text="b", done=True),
        ]
        new = [
            fc.FocusItem(text="a", done=True),
            fc.FocusItem(text="b", done=False),
        ]
        diff = fc.diff_focus_chains(old, new)
        assert diff.completed == ("a",)
        assert diff.uncompleted == ("b",)

    def test_reorder_only(self) -> None:
        a = fc.FocusItem(text="a", done=False)
        b = fc.FocusItem(text="b", done=False)
        diff = fc.diff_focus_chains([a, b], [b, a])
        assert diff.reordered is True
        assert diff.added == ()
        assert diff.removed == ()


# ---------------------------------------------------------------------------
# render_critical_info_block + progress_hint
# ---------------------------------------------------------------------------

class TestRenderHelpers:
    def test_critical_info_empty_when_no_change(self) -> None:
        diff = fc.FocusChainDiff()
        assert fc.render_critical_info_block(diff) == ""

    def test_critical_info_lists_changes(self) -> None:
        diff = fc.FocusChainDiff(
            added=("alpha",),
            removed=("beta",),
            completed=("gamma",),
            uncompleted=("delta",),
            reordered=True,
        )
        out = fc.render_critical_info_block(diff)
        assert "CRITICAL INFORMATION" in out
        assert "+ alpha" in out
        assert "- beta" in out
        assert "[x] gamma" in out
        assert "[ ] delta" in out
        assert "rearranged" in out

    def test_progress_hint_empty(self) -> None:
        assert fc.progress_hint([]) == ""

    def test_progress_hint_bands(self) -> None:
        all_pending = [fc.FocusItem(text=f"i{i}", done=False) for i in range(4)]
        out = fc.progress_hint(all_pending)
        assert "fresh" in out.lower() or "Plan" in out

        half_done = [
            fc.FocusItem(text="a", done=True),
            fc.FocusItem(text="b", done=True),
            fc.FocusItem(text="c", done=False),
            fc.FocusItem(text="d", done=False),
        ]
        msg = fc.progress_hint(half_done)
        assert "2/4" in msg

        all_done = [fc.FocusItem(text="x", done=True)]
        msg = fc.progress_hint(all_done)
        assert "All" in msg or "1" in msg


# ---------------------------------------------------------------------------
# FocusChain I/O
# ---------------------------------------------------------------------------

class TestFocusChain:
    def test_load_missing_file_empty(self, tmp_path: Path) -> None:
        chain = fc.FocusChain(path=tmp_path / "nope.md")
        items = chain.load()
        assert items == []

    def test_save_then_load(self, tmp_path: Path) -> None:
        path = tmp_path / "plan.md"
        chain = fc.FocusChain(path=path, header="# Plan\n")
        chain.set_items(["alpha", ("beta", True), fc.FocusItem(text="gamma", done=False)])
        chain.save()
        text = path.read_text(encoding="utf-8")
        assert "# Plan" in text
        assert "- [ ] alpha" in text
        assert "- [x] beta" in text
        # New chain reads them back.
        fresh = fc.FocusChain(path=path)
        loaded = fresh.load()
        assert {i.text for i in loaded} == {"alpha", "beta", "gamma"}

    def test_mark_done_and_pending(self, tmp_path: Path) -> None:
        chain = fc.FocusChain(path=tmp_path / "plan.md")
        chain.set_items(["alpha", ("beta", True)])
        assert chain.mark_done("alpha") is True
        assert chain.items[0].done is True
        assert chain.mark_pending("beta") is True
        assert chain.items[1].done is False
        # Non-existent text returns False.
        assert chain.mark_done("missing") is False
        assert chain.mark_pending("missing") is False

    def test_progress_ratio(self, tmp_path: Path) -> None:
        chain = fc.FocusChain(path=tmp_path / "plan.md")
        chain.set_items([("a", True), ("b", False), ("c", True)])
        assert abs(chain.progress_ratio() - 2 / 3) < 1e-6

    def test_progress_ratio_empty(self, tmp_path: Path) -> None:
        chain = fc.FocusChain(path=tmp_path / "plan.md")
        assert chain.progress_ratio() == 0.0


# ---------------------------------------------------------------------------
# FocusChainWatcher (manual poll mode — no watchdog dependency)
# ---------------------------------------------------------------------------

class TestWatcherPoll:
    def test_poll_now_detects_user_edit(self, tmp_path: Path) -> None:
        chain = fc.FocusChain(path=tmp_path / "plan.md")
        chain.set_items([("a", False), ("b", False)])
        chain.save()
        chain.load()
        captured: list[fc.FocusChainDiff] = []
        watcher = fc.FocusChainWatcher(
            chain, on_user_edit=captured.append, debounce_ms=0,
        )
        # Simulate external edit (mark 'a' done).
        time.sleep(0.05)
        chain.path.write_text("- [x] a\n- [ ] b\n", encoding="utf-8")
        import os as _os
        _os.utime(chain.path, None)
        diff = watcher.poll_now()
        assert diff is not None
        assert "a" in diff.completed
        assert len(captured) == 1

    def test_poll_now_no_change_returns_none(self, tmp_path: Path) -> None:
        chain = fc.FocusChain(path=tmp_path / "plan.md")
        chain.set_items([("a", False)])
        chain.save()
        chain.load()
        watcher = fc.FocusChainWatcher(
            chain, on_user_edit=lambda _diff: None, debounce_ms=0,
        )
        # No file modification between load + poll.
        assert watcher.poll_now() is None

    def test_poll_callback_exception_swallowed(self, tmp_path: Path) -> None:
        chain = fc.FocusChain(path=tmp_path / "plan.md")
        chain.set_items([("a", False)])
        chain.save()
        chain.load()

        def boom(_diff: fc.FocusChainDiff) -> None:
            raise RuntimeError("user-callback failure")

        watcher = fc.FocusChainWatcher(chain, on_user_edit=boom, debounce_ms=0)
        time.sleep(0.05)
        chain.path.write_text("- [x] a\n", encoding="utf-8")
        import os as _os
        _os.utime(chain.path, None)
        # Should not raise.
        diff = watcher.poll_now()
        assert diff is not None
