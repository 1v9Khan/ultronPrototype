"""Tests for the runtime validator's path canonicalisation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ultron.safety.path_resolver import (
    PathResolveError,
    PathResolver,
    get_path_resolver,
    reset_path_resolver,
)


def setup_function(_):
    reset_path_resolver()


def test_singleton_is_stable():
    r1 = get_path_resolver()
    r2 = get_path_resolver()
    assert r1 is r2


def test_reset_creates_fresh_instance():
    r1 = get_path_resolver()
    reset_path_resolver()
    r2 = get_path_resolver()
    assert r1 is not r2


def test_reject_percent_encoded():
    r = PathResolver()
    with pytest.raises(PathResolveError):
        r.resolve("c:/some%20path/file.txt")


def test_reject_unicode_bidi():
    r = PathResolver()
    # Right-to-left override character U+202E
    with pytest.raises(PathResolveError):
        r.resolve("c:/safe/‮suspicious/file.txt")


def test_reject_zero_width_space():
    r = PathResolver()
    with pytest.raises(PathResolveError):
        r.resolve("c:/safe/zwsp​/file.txt")


def test_reject_bom():
    r = PathResolver()
    with pytest.raises(PathResolveError):
        r.resolve("﻿c:/safe/file.txt")


def test_reject_non_string_input():
    r = PathResolver()
    with pytest.raises(PathResolveError):
        r.resolve(12345)  # type: ignore[arg-type]


def test_resolve_relative_against_project_root(tmp_path, monkeypatch):
    """Relative paths resolve against PROJECT_ROOT, not CWD."""
    r = PathResolver()
    # Force the resolver's PROJECT_ROOT to a known value via the
    # internal cache. (In production this comes from
    # ultron.config.PROJECT_ROOT.)
    r._project_root = tmp_path
    out = r.resolve("subdir/file.txt")
    expected = (tmp_path / "subdir" / "file.txt").resolve(strict=False)
    assert out == expected


def test_normalise_separators_windows():
    r = PathResolver()
    if not sys.platform.startswith("win"):
        pytest.skip("windows-specific separator normalisation")
    out = r.normalise_separators("c:/Windows\\System32")
    assert "/" not in out
    assert out == "c:\\Windows\\System32"


def test_strip_long_path_prefix():
    r = PathResolver()
    # Build the inputs explicitly so the test is unambiguous about
    # exactly which bytes the resolver sees.
    long_prefix = "\\\\?\\"             # \\?\ (4 chars: \, \, ?, \)
    long_prefixed = long_prefix + "C:\\Windows"      # \\?\C:\Windows
    assert r.strip_long_path_prefix(long_prefixed) == "C:\\Windows"

    unc_prefix = "\\\\?\\UNC\\"        # \\?\UNC\ (8 chars)
    unc_prefixed = unc_prefix + "server\\share"
    assert r.strip_long_path_prefix(unc_prefixed) == "\\\\server\\share"

    # Plain path without the prefix is returned unchanged.
    assert r.strip_long_path_prefix("C:\\Windows") == "C:\\Windows"


def test_is_inside_returns_true_for_descendant(tmp_path):
    r = PathResolver()
    parent = tmp_path / "a"
    parent.mkdir()
    child = parent / "b" / "c.txt"
    assert r.is_inside(candidate=child, allowed_root=parent)


def test_is_inside_returns_false_for_sibling(tmp_path):
    r = PathResolver()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert not r.is_inside(
        candidate=tmp_path / "b" / "x.txt",
        allowed_root=tmp_path / "a",
    )


def test_is_inside_treats_exact_match_as_inside(tmp_path):
    r = PathResolver()
    p = tmp_path / "exact"
    p.mkdir()
    assert r.is_inside(candidate=p, allowed_root=p)


def test_is_under_any_handles_unresolvable_roots(tmp_path):
    """One unresolvable root must not prevent matching against good ones."""
    r = PathResolver()
    good = tmp_path / "good"
    good.mkdir()
    child = good / "x.txt"
    # The first root is unresolvable (percent escape); the second is
    # legitimate. The "any" check must still return True.
    assert r.is_under_any(
        candidate=child,
        allowed_roots=["c:/bad%20path", str(good)],
    )


def test_uppercase_drive_letter_on_windows():
    if not sys.platform.startswith("win"):
        pytest.skip("windows-specific drive-letter normalisation")
    r = PathResolver()
    out = r.resolve("c:\\windows")
    assert str(out).startswith("C:")
