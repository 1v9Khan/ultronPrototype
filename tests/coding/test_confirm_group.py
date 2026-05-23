"""Tests for the batched confirmation group (catalog T14)."""

from __future__ import annotations

import pytest

from ultron.coding.confirm_group import ConfirmGroup, ConfirmGroupResolution


# ---------------------------------------------------------------------------
# add / dedupe / overflow
# ---------------------------------------------------------------------------


def test_add_stores_item():
    cg = ConfirmGroup()
    cg.add("Modify foo.py")
    assert "Modify foo.py" in cg
    assert len(cg) == 1


def test_empty_item_is_ignored():
    cg = ConfirmGroup()
    cg.add("")
    cg.add("   ")
    assert len(cg) == 0


def test_duplicate_item_is_ignored():
    cg = ConfirmGroup()
    cg.add("Modify foo")
    cg.add("Modify foo")
    cg.add("Modify foo")
    assert len(cg) == 1


def test_max_items_overflow_creates_summary():
    cg = ConfirmGroup(max_items=2)
    cg.add("first")
    cg.add("second")
    cg.add("third")
    cg.add("fourth")
    assert len(cg) == 2  # only the first two stored
    q = cg.render_question()
    assert "and 2 more" in q  # overflow_count reported


def test_constructor_rejects_invalid_max_items():
    with pytest.raises(ValueError):
        ConfirmGroup(max_items=0)


# ---------------------------------------------------------------------------
# render_question
# ---------------------------------------------------------------------------


def test_render_empty_returns_empty_string():
    cg = ConfirmGroup()
    assert cg.render_question() == ""


def test_render_one_item_is_single_clause():
    cg = ConfirmGroup()
    cg.add("modify foo.py")
    q = cg.render_question()
    assert q == "I'll modify foo.py. Okay?"


def test_render_two_items_uses_and():
    cg = ConfirmGroup()
    cg.add("modify foo")
    cg.add("delete bar")
    q = cg.render_question()
    assert q == "I'll modify foo and delete bar. Okay?"


def test_render_three_or_more_uses_oxford_comma():
    cg = ConfirmGroup()
    cg.add("modify foo")
    cg.add("delete bar")
    cg.add("create baz")
    q = cg.render_question()
    assert q == "I'll modify foo, delete bar, and create baz. Okay?"


def test_render_custom_prefix():
    cg = ConfirmGroup(prefix="Going to")
    cg.add("touch a file")
    assert cg.render_question().startswith("Going to touch a file.")


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def test_is_pending_true_when_unresolved_items_present():
    cg = ConfirmGroup()
    assert cg.is_pending() is False  # empty
    cg.add("x")
    assert cg.is_pending() is True


def test_resolve_returns_resolution_record():
    cg = ConfirmGroup()
    cg.add("alpha")
    cg.add("beta")
    res = cg.resolve(approved=True)
    assert isinstance(res, ConfirmGroupResolution)
    assert res.approved is True
    assert res.items == ("alpha", "beta")
    assert "alpha" in res.question
    assert "beta" in res.question


def test_resolve_marks_group_not_pending():
    cg = ConfirmGroup()
    cg.add("alpha")
    cg.resolve(False)
    assert cg.is_pending() is False
    assert cg.resolution.approved is False


def test_double_resolve_raises():
    cg = ConfirmGroup()
    cg.add("alpha")
    cg.resolve(True)
    with pytest.raises(RuntimeError):
        cg.resolve(False)


def test_add_after_resolve_raises():
    cg = ConfirmGroup()
    cg.add("alpha")
    cg.resolve(True)
    with pytest.raises(RuntimeError):
        cg.add("beta")
