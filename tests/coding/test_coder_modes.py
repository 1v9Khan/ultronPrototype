"""Tests for :mod:`ultron.coding.coder_modes`."""

from __future__ import annotations

import pytest

from ultron.coding.coder_modes import (
    CODER_MODES,
    CoderMode,
    EditFormat,
    edit_modes,
    get_coder_mode,
    list_coder_modes,
    read_only_modes,
)


def test_registry_has_expected_modes():
    expected = {"edit", "ask", "architect", "context", "whole_file", "udiff", "patch_v4a", "help"}
    assert expected.issubset(set(CODER_MODES))


def test_get_coder_mode_known():
    m = get_coder_mode("edit")
    assert m is not None
    assert m.name == "edit"
    assert m.edit_format == EditFormat.SEARCH_REPLACE
    assert m.produces_edits is True


def test_get_coder_mode_case_insensitive():
    assert get_coder_mode("EDIT") is not None
    assert get_coder_mode("Architect") is not None


def test_get_coder_mode_unknown_returns_none():
    assert get_coder_mode("nonexistent") is None


def test_get_coder_mode_empty_returns_none():
    assert get_coder_mode("") is None


def test_ask_mode_does_not_produce_edits():
    m = get_coder_mode("ask")
    assert m.produces_edits is False
    assert m.edit_format == EditFormat.NONE


def test_architect_mode_is_supervised():
    m = get_coder_mode("architect")
    assert m.is_supervised is True
    assert m.produces_edits is False
    assert m.edit_format == EditFormat.ARCHITECT_DISPATCH


def test_list_coder_modes_returns_sorted():
    names = list_coder_modes()
    assert names == sorted(names)
    assert "edit" in names
    assert "ask" in names


def test_edit_modes_subset():
    modes = edit_modes()
    assert all(m.produces_edits for m in modes)
    assert any(m.name == "edit" for m in modes)
    assert any(m.name == "whole_file" for m in modes)
    assert all(m.name != "ask" for m in modes)
    assert all(m.name != "architect" for m in modes)


def test_read_only_modes_subset():
    modes = read_only_modes()
    assert all(not m.produces_edits for m in modes)
    assert any(m.name == "ask" for m in modes)
    assert any(m.name == "architect" for m in modes)
    assert any(m.name == "help" for m in modes)


def test_coder_mode_is_frozen():
    m = CoderMode(
        name="x",
        description="x",
        prompt_template="x",
        edit_format=EditFormat.NONE,
        produces_edits=False,
    )
    with pytest.raises(Exception):
        m.name = "y"  # type: ignore[misc]


def test_edit_format_enum_values():
    """Sanity-check the enum has the expected hint values."""
    assert EditFormat.SEARCH_REPLACE.value == "search_replace"
    assert EditFormat.PATCH_V4A.value == "patch_v4a"
    assert EditFormat.NONE.value == "none"


def test_patch_v4a_mode_present():
    m = get_coder_mode("patch_v4a")
    assert m is not None
    assert m.edit_format == EditFormat.PATCH_V4A


def test_every_mode_has_description():
    """Every entry has a non-empty description for the future /help-by-voice."""
    for mode in CODER_MODES.values():
        assert mode.description, f"missing description for {mode.name}"
        assert mode.prompt_template, f"missing template for {mode.name}"
