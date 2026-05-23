"""Tests for :mod:`ultron.coding.patch_v4a`."""

from __future__ import annotations

import pytest

from ultron.coding.patch_v4a import (
    BEGIN_PATCH,
    END_PATCH,
    FUZZ_EXACT,
    FUZZ_RSTRIP,
    FUZZ_STRIP,
    PatchAction,
    PatchError,
    PatchFileBlock,
    PatchHunk,
    apply_patch,
    parse_v4a_patch,
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _wrap(*body_lines: str) -> str:
    return "\n".join([BEGIN_PATCH, *body_lines, END_PATCH])


def test_parse_empty_raises():
    with pytest.raises(PatchError):
        parse_v4a_patch("")


def test_parse_no_begin_marker_raises():
    with pytest.raises(PatchError):
        parse_v4a_patch("*** Update File: a.py\n+x\n*** End Patch")


def test_parse_no_end_marker_raises():
    with pytest.raises(PatchError):
        parse_v4a_patch("*** Begin Patch\n*** Update File: a.py\n+x\n")


def test_parse_empty_body_raises():
    with pytest.raises(PatchError):
        parse_v4a_patch(_wrap())


def test_parse_unknown_action_prefix_raises():
    with pytest.raises(PatchError):
        parse_v4a_patch(_wrap("*** Rename File: a.py", "+x"))


def test_parse_delete_block():
    text = _wrap("*** Delete File: stale.py")
    parsed = parse_v4a_patch(text)
    assert len(parsed.blocks) == 1
    block = parsed.blocks[0]
    assert block.action == PatchAction.DELETE
    assert block.file_path == "stale.py"


def test_parse_add_block():
    text = _wrap(
        "*** Add File: new.py",
        "+def foo():",
        "+    return 42",
    )
    parsed = parse_v4a_patch(text)
    block = parsed.blocks[0]
    assert block.action == PatchAction.ADD
    assert block.file_path == "new.py"
    assert block.hunks[0].added_lines == ("def foo():", "    return 42")


def test_parse_update_block_with_one_hunk():
    text = _wrap(
        "*** Update File: a.py",
        " context_before",
        "-old_line",
        "+new_line",
        " context_after",
    )
    parsed = parse_v4a_patch(text)
    block = parsed.blocks[0]
    assert block.action == PatchAction.UPDATE
    assert block.file_path == "a.py"
    assert len(block.hunks) == 1
    hunk = block.hunks[0]
    assert hunk.before_context == ("context_before",)
    assert hunk.removed_lines == ("old_line",)
    assert hunk.added_lines == ("new_line",)
    assert hunk.after_context == ("context_after",)


def test_parse_update_block_with_scope_marker():
    text = _wrap(
        "*** Update File: a.py",
        "@@ class Foo",
        " context_before",
        "-old",
        "+new",
        " context_after",
    )
    parsed = parse_v4a_patch(text)
    hunk = parsed.blocks[0].hunks[0]
    assert hunk.scope == "class Foo"


def test_parse_update_block_with_eof_marker():
    text = _wrap(
        "*** Update File: a.py",
        " context",
        "-old",
        "+new",
        "*** End of File",
    )
    parsed = parse_v4a_patch(text)
    assert parsed.blocks[0].hunks[0].ends_at_eof is True


def test_parse_multiple_blocks():
    text = _wrap(
        "*** Update File: a.py",
        " context",
        "-old",
        "+new",
        " context",
        "*** Add File: new.py",
        "+def foo(): pass",
        "*** Delete File: stale.py",
    )
    parsed = parse_v4a_patch(text)
    assert len(parsed.blocks) == 3
    kinds = [b.action for b in parsed.blocks]
    assert kinds == [PatchAction.UPDATE, PatchAction.ADD, PatchAction.DELETE]


def test_parse_update_with_unknown_line_prefix_raises():
    text = _wrap(
        "*** Update File: a.py",
        " ctx",
        "!! bad prefix",
        " ctx",
    )
    with pytest.raises(PatchError):
        parse_v4a_patch(text)


def test_parse_update_with_no_changes_raises():
    text = _wrap(
        "*** Update File: a.py",
        " just context",
        " no changes",
    )
    with pytest.raises(PatchError):
        parse_v4a_patch(text)


# ---------------------------------------------------------------------------
# Applier
# ---------------------------------------------------------------------------


def test_apply_delete_removes_path():
    patch = parse_v4a_patch(_wrap("*** Delete File: stale.py"))
    out = apply_patch(patch, {"stale.py": "garbage\n"})
    assert out == {"stale.py": None}


def test_apply_delete_missing_file_raises():
    patch = parse_v4a_patch(_wrap("*** Delete File: nope.py"))
    with pytest.raises(PatchError):
        apply_patch(patch, {})


def test_apply_add_creates_file():
    patch = parse_v4a_patch(_wrap(
        "*** Add File: new.py",
        "+def foo():",
        "+    return 42",
    ))
    out = apply_patch(patch, {})
    assert out["new.py"] == "def foo():\n    return 42\n"


def test_apply_add_existing_raises():
    patch = parse_v4a_patch(_wrap(
        "*** Add File: a.py",
        "+content",
    ))
    with pytest.raises(PatchError):
        apply_patch(patch, {"a.py": "existing\n"})


def test_apply_update_exact_match():
    original = "alpha\nbeta\ngamma\ndelta\n"
    patch = parse_v4a_patch(_wrap(
        "*** Update File: x.py",
        " alpha",
        "-beta",
        "+BETA",
        " gamma",
    ))
    out = apply_patch(patch, {"x.py": original})
    assert out["x.py"] == "alpha\nBETA\ngamma\ndelta\n"


def test_apply_update_missing_target_raises():
    patch = parse_v4a_patch(_wrap(
        "*** Update File: nope.py",
        " ctx",
        "-old",
        "+new",
        " ctx",
    ))
    with pytest.raises(PatchError):
        apply_patch(patch, {})


def test_apply_update_unique_match_required():
    """Context that appears twice -> hunk fails to locate."""
    original = "ctx\nold\nctx\nctx\nold\nctx\n"
    patch = parse_v4a_patch(_wrap(
        "*** Update File: x.py",
        " ctx",
        "-old",
        "+new",
        " ctx",
    ))
    with pytest.raises(PatchError):
        apply_patch(patch, {"x.py": original})


def test_apply_update_fuzz_rstrip_match():
    """Original has trailing whitespace; rstrip-fuzz should match."""
    original = "alpha   \nbeta\ngamma   \n"
    patch = parse_v4a_patch(_wrap(
        "*** Update File: x.py",
        " alpha",
        "-beta",
        "+BETA",
        " gamma",
    ))
    out = apply_patch(patch, {"x.py": original})
    assert out["x.py"].startswith("alpha")
    assert "BETA" in out["x.py"]


def test_apply_multi_block_combines_changes():
    """Update + Add + Delete in one patch."""
    patch = parse_v4a_patch(_wrap(
        "*** Update File: a.py",
        " ctx_a",
        "-old_a",
        "+new_a",
        " ctx_b",
        "*** Add File: brand_new.py",
        "+def hello(): return 1",
        "*** Delete File: stale.py",
    ))
    files = {
        "a.py": "ctx_a\nold_a\nctx_b\n",
        "stale.py": "old garbage\n",
    }
    out = apply_patch(patch, files)
    assert out["a.py"] == "ctx_a\nnew_a\nctx_b\n"
    assert out["brand_new.py"].startswith("def hello():")
    assert out["stale.py"] is None


# ---------------------------------------------------------------------------
# Module-level smoke
# ---------------------------------------------------------------------------


def test_fuzz_constants_match_catalog():
    """Catalog T17: 0 / 1 / 100 / 10000 fuzz tiers."""
    from ultron.coding.patch_v4a import FUZZ_EOF_MISMATCH

    assert FUZZ_EXACT == 0
    assert FUZZ_RSTRIP == 1
    assert FUZZ_STRIP == 100
    assert FUZZ_EOF_MISMATCH == 10_000


def test_patchhunk_is_frozen():
    h = PatchHunk(
        scope="",
        before_context=(),
        removed_lines=(),
        added_lines=("x",),
        after_context=(),
    )
    with pytest.raises(Exception):
        h.added_lines = ("y",)  # type: ignore[misc]


def test_patchfileblock_is_frozen():
    b = PatchFileBlock(action=PatchAction.DELETE, file_path="x.py")
    with pytest.raises(Exception):
        b.file_path = "y.py"  # type: ignore[misc]
