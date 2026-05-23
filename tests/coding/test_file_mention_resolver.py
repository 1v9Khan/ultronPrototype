"""Tests for :mod:`ultron.coding.file_mention_resolver`."""

from __future__ import annotations

from pathlib import Path

import pytest

from ultron.coding.file_mention_resolver import (
    DEFAULT_NEVER_AUTO_ADD_BASENAMES,
    FileMention,
    resolve_mentions,
)


def test_exact_path_match():
    mentions = resolve_mentions(
        "look at src/foo/bar.py for context",
        ["src/foo/bar.py", "src/foo/other.py"],
    )
    assert len(mentions) == 1
    assert mentions[0].path == "src/foo/bar.py"
    assert mentions[0].kind == "exact"


def test_basename_match_with_unique_special_chars():
    mentions = resolve_mentions(
        "fix the parakeet_engine.py streaming bug",
        ["src/transcription/parakeet_engine.py", "src/transcription/other.py"],
    )
    assert any(m.path.endswith("parakeet_engine.py") for m in mentions)


def test_basename_match_requires_special_chars():
    """Plain-word basenames without dots/underscores/hyphens are skipped."""
    mentions = resolve_mentions(
        "what does mainfile do?",
        ["src/mainfile"],  # no special chars; ambiguous as a word
    )
    assert mentions == []


def test_basename_blocked_by_never_auto_add_list():
    """run.py / make.py / test.py / etc. need explicit disambiguation."""
    mentions = resolve_mentions(
        "let's run.py the demo",
        ["scripts/run.py", "scripts/other.py"],
    )
    assert mentions == []  # 'run' is in the blocklist


def test_basename_ambiguous_when_multiple_files_share_name():
    mentions = resolve_mentions(
        "fix the issue in utils.py",
        ["src/a/utils.py", "src/b/utils.py", "src/c/utils.py"],
    )
    # Ambiguous: three candidates share the same basename.
    assert mentions == []


def test_basename_with_unique_path_passes_blocklist():
    """An unambiguous non-blocked basename matches."""
    mentions = resolve_mentions(
        "look at parakeet_engine.py",
        ["src/transcription/parakeet_engine.py"],
    )
    assert mentions
    assert mentions[0].kind == "basename"


def test_already_in_chat_excluded():
    mentions = resolve_mentions(
        "look at src/foo/bar.py",
        ["src/foo/bar.py"],
        already_in_chat=["src/foo/bar.py"],
    )
    assert mentions == []


def test_ignore_set_respected():
    mentions = resolve_mentions(
        "look at src/foo/bar.py",
        ["src/foo/bar.py"],
        ignore=["src/foo/bar.py"],
    )
    assert mentions == []


def test_custom_never_set_override():
    """Caller can replace the default blocklist entirely."""
    mentions = resolve_mentions(
        "look at run.py",
        ["scripts/run.py"],
        never_auto_add_basenames=set(),  # empty -> nothing is blocked
    )
    assert mentions
    assert mentions[0].path == "scripts/run.py"


def test_windows_paths_normalised():
    mentions = resolve_mentions(
        "look at src/foo/bar.py",
        [r"src\foo\bar.py"],
    )
    assert mentions
    # Returned path is POSIX form.
    assert mentions[0].path == "src/foo/bar.py"


def test_punctuation_at_token_edges_stripped():
    mentions = resolve_mentions(
        "look at parakeet_engine.py, please?",
        ["src/transcription/parakeet_engine.py"],
    )
    assert mentions


def test_empty_inputs_return_empty():
    assert resolve_mentions("", ["src/foo.py"]) == []
    assert resolve_mentions("hello", []) == []


def test_default_never_set_covers_common_words():
    expected = {"run", "make", "test", "main", "build"}
    assert expected.issubset(DEFAULT_NEVER_AUTO_ADD_BASENAMES)


def test_filemention_is_frozen():
    m = FileMention(path="x.py", kind="exact", confidence=1.0)
    with pytest.raises(Exception):
        m.path = "y.py"  # type: ignore[misc]


def test_returns_deterministic_order():
    """Same inputs -> same outputs, every call."""
    candidates = [
        "src/alpha_thing.py",
        "src/beta_other.py",
        "src/gamma_extra.py",
    ]
    text = "fix gamma_extra.py and alpha_thing.py"
    r1 = resolve_mentions(text, candidates)
    r2 = resolve_mentions(text, candidates)
    assert [m.path for m in r1] == [m.path for m in r2]


def test_provider_cache_wires_resolver(tmp_path: Path):
    """End-to-end: RepoMapProviderCache.__call__ extracts mentions
    from user_text and threads them through to RepoMap as
    mentioned_fnames."""
    from ultron.coding.repo_map import RepoMap, RepoMapProviderCache

    # Seed a project with two files; one referenced by name.
    (tmp_path / "obvious_target.py").write_text("def obvious(): pass\n")
    (tmp_path / "untouched.py").write_text("def untouched(): pass\n")

    cache = RepoMapProviderCache(
        max_map_tokens=200,
        max_map_tokens_no_chat=200,
    )
    rendered = cache(
        str(tmp_path),
        "what does obvious_target.py do?",
    )
    assert rendered
    assert "obvious_target.py" in rendered
