"""Tests for the T21 realpath-aware path-resolver extensions.

T21 (OpenClaw catalog port). Tests use ``tmp_path`` exclusively (R9
binding rule); no writes outside the pytest tmpdir.
"""

from __future__ import annotations

import errno
import os
import sys

import pytest

from ultron.safety.path_resolver import PathResolver


@pytest.fixture
def resolver() -> PathResolver:
    return PathResolver()


# ----------------------------------------------------------------------
# safe_realpath


def test_safe_realpath_returns_canonical_path_for_existing(
    resolver: PathResolver, tmp_path
) -> None:
    target = tmp_path / "file.txt"
    target.write_text("data", encoding="utf-8")
    real = resolver.safe_realpath(target)
    assert real is not None
    assert real == target.resolve()


def test_safe_realpath_returns_path_for_nonexistent(resolver: PathResolver, tmp_path) -> None:
    # safe_realpath with strict=False returns the resolved path even when
    # the file doesn't exist; that mirrors os.path.realpath.
    target = tmp_path / "missing.txt"
    real = resolver.safe_realpath(target)
    assert real is not None


def test_safe_realpath_returns_none_for_dangerous_input(resolver: PathResolver) -> None:
    # Percent-encoded escape sequences are explicitly rejected.
    assert resolver.safe_realpath("%2e%2e/etc/passwd") is None


def test_safe_realpath_returns_none_for_bidi_override(resolver: PathResolver) -> None:
    # bidi override character should be rejected.
    bad = "abc‮def.txt"
    assert resolver.safe_realpath(bad) is None


def test_safe_realpath_accepts_str_or_path(resolver: PathResolver, tmp_path) -> None:
    target = tmp_path / "x.txt"
    target.write_text("x", encoding="utf-8")
    real_from_str = resolver.safe_realpath(str(target))
    real_from_path = resolver.safe_realpath(target)
    assert real_from_str == real_from_path


@pytest.mark.skipif(sys.platform.startswith("win"), reason="symlinks require admin on Win")
def test_safe_realpath_resolves_symlink_target(resolver: PathResolver, tmp_path) -> None:
    target = tmp_path / "real.txt"
    target.write_text("hi", encoding="utf-8")
    link = tmp_path / "link.txt"
    os.symlink(target, link)
    real = resolver.safe_realpath(link)
    assert real == target.resolve()


# ----------------------------------------------------------------------
# is_inside_with_realpath


def test_is_inside_with_realpath_accepts_path_under_root(
    resolver: PathResolver, tmp_path
) -> None:
    target = tmp_path / "sub" / "file.txt"
    target.parent.mkdir()
    target.write_text("data", encoding="utf-8")
    assert resolver.is_inside_with_realpath(target, tmp_path) is True


def test_is_inside_with_realpath_rejects_path_outside_root(
    resolver: PathResolver, tmp_path
) -> None:
    other = tmp_path.parent / "other.txt"
    assert resolver.is_inside_with_realpath(other, tmp_path) is False


def test_is_inside_with_realpath_unresolvable_returns_false(
    resolver: PathResolver,
) -> None:
    # Dangerous input -> safe_realpath returns None -> containment check fails.
    assert resolver.is_inside_with_realpath("%2e%2e/etc", "/tmp") is False


@pytest.mark.skipif(sys.platform.startswith("win"), reason="symlinks require admin on Win")
def test_is_inside_with_realpath_rejects_symlink_to_outside(
    resolver: PathResolver, tmp_path
) -> None:
    # Symlink lives inside root but target is outside — the fast
    # is_inside path would accept it; the realpath path must reject.
    outside = tmp_path.parent / "outside_target.txt"
    outside.write_text("outside", encoding="utf-8")
    link = tmp_path / "link.txt"
    os.symlink(outside, link)
    try:
        # Realpath path catches the escape.
        assert resolver.is_inside_with_realpath(link, tmp_path) is False
    finally:
        try:
            outside.unlink()
        except FileNotFoundError:
            pass


# ----------------------------------------------------------------------
# is_symlink_open_error


def test_is_symlink_open_error_eloop_returns_true() -> None:
    err = OSError(errno.ELOOP, "too many symlinks")
    assert PathResolver.is_symlink_open_error(err) is True


def test_is_symlink_open_error_emlink_returns_true() -> None:
    err = OSError(errno.EMLINK, "max link")
    assert PathResolver.is_symlink_open_error(err) is True


def test_is_symlink_open_error_other_oserror_returns_false() -> None:
    err = OSError(errno.ENOENT, "no such file")
    assert PathResolver.is_symlink_open_error(err) is False


def test_is_symlink_open_error_non_oserror_returns_false() -> None:
    assert PathResolver.is_symlink_open_error(ValueError("nope")) is False


# ----------------------------------------------------------------------
# normalise_for_comparison


def test_normalise_for_comparison_lowercases_on_windows(
    resolver: PathResolver, tmp_path
) -> None:
    target = tmp_path / "FILE.TXT"
    target.write_text("x", encoding="utf-8")
    s = resolver.normalise_for_comparison(target)
    if resolver._is_windows:
        assert s == s.lower()
    else:
        # POSIX: case preserved.
        assert "FILE.TXT" in s


def test_normalise_for_comparison_returns_absolute(
    resolver: PathResolver, tmp_path
) -> None:
    target = tmp_path / "abs.txt"
    target.write_text("x", encoding="utf-8")
    s = resolver.normalise_for_comparison(target)
    assert os.path.isabs(s)
