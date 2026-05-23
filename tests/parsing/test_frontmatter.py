"""Tests for the YAML frontmatter parser (T11 from the OpenHands catalog)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ultron.parsing import (
    FrontmatterResult,
    parse_frontmatter,
    parse_frontmatter_text,
    walk_directory_with_frontmatter,
)


def test_no_frontmatter_returns_full_body_no_error():
    text = "no leading delimiter here\nsecond line"
    result = parse_frontmatter_text(text)
    assert result.frontmatter is None
    assert result.body == text
    assert result.error is None
    assert result.ok is True
    assert result.has_frontmatter is False


def test_well_formed_frontmatter_parses():
    text = "---\nname: gaming\ntype: knowledge\ntriggers:\n  - valorant\n---\nBody content here."
    result = parse_frontmatter_text(text, source_path="example.md")
    assert isinstance(result, FrontmatterResult)
    assert result.frontmatter == {
        "name": "gaming",
        "type": "knowledge",
        "triggers": ["valorant"],
    }
    assert result.body == "Body content here."
    assert result.error is None
    assert result.ok is True
    assert result.has_frontmatter is True
    assert result.path == Path("example.md")


def test_invalid_yaml_returns_error_but_preserves_body():
    text = "---\nname: gaming\ntype: : invalid: : yaml\n---\nstill the body"
    result = parse_frontmatter_text(text)
    assert result.frontmatter is None
    assert result.error is not None
    assert "YAML error" in result.error
    # Body should still be reachable so callers can fall back gracefully.
    assert result.body == "still the body"


def test_missing_closing_delimiter_returns_error():
    text = "---\nname: missing\nclose: delim\nstill no closer"
    result = parse_frontmatter_text(text)
    assert result.frontmatter is None
    assert result.error is not None
    assert "closing" in result.error.lower()


def test_non_mapping_frontmatter_returns_error():
    # YAML list at top level shouldn't be accepted as a frontmatter mapping.
    text = "---\n- one\n- two\n---\nbody"
    result = parse_frontmatter_text(text)
    assert result.frontmatter is None
    assert result.error is not None
    assert "not a mapping" in result.error


def test_empty_frontmatter_block_returns_empty_dict():
    text = "---\n---\nbody after empty fm"
    result = parse_frontmatter_text(text)
    assert result.frontmatter == {}
    assert result.body == "body after empty fm"
    assert result.error is None


def test_get_helper_reads_from_frontmatter():
    text = "---\nname: x\nvalue: 42\n---\nbody"
    result = parse_frontmatter_text(text)
    assert result.get("name") == "x"
    assert result.get("value") == 42
    assert result.get("missing") is None
    assert result.get("missing", "default") == "default"


def test_get_on_none_frontmatter_returns_default():
    text = "no fm here"
    result = parse_frontmatter_text(text)
    assert result.get("name") is None
    assert result.get("name", "fallback") == "fallback"


def test_parse_frontmatter_reads_file(tmp_path: Path):
    target = tmp_path / "skill.md"
    target.write_text(
        "---\nname: example\ntriggers: [foo]\n---\nA short body.",
        encoding="utf-8",
    )
    result = parse_frontmatter(target)
    assert result.ok is True
    assert result.frontmatter == {"name": "example", "triggers": ["foo"]}
    assert result.body == "A short body."
    assert result.path == target


def test_parse_frontmatter_missing_file_returns_error(tmp_path: Path):
    missing = tmp_path / "absent.md"
    result = parse_frontmatter(missing)
    assert result.frontmatter is None
    assert result.body == ""
    assert result.error is not None
    assert "not found" in result.error


def test_parse_frontmatter_handles_decode_error(tmp_path: Path):
    target = tmp_path / "bad.md"
    target.write_bytes(b"---\nname: x\n---\n\xff\xfe\xfd")
    result = parse_frontmatter(target)
    # The decode error happens on read_text; expect a populated error.
    assert result.frontmatter is None
    assert result.error is not None


def test_walk_directory_yields_per_file(tmp_path: Path):
    (tmp_path / "a.md").write_text(
        "---\nname: a\n---\nA body",
        encoding="utf-8",
    )
    (tmp_path / "b.md").write_text("plain body", encoding="utf-8")
    (tmp_path / "c.txt").write_text("ignored ext", encoding="utf-8")

    results = list(walk_directory_with_frontmatter(tmp_path))
    by_name = {r.path.name: r for r in results}
    assert set(by_name.keys()) == {"a.md", "b.md"}
    assert by_name["a.md"].frontmatter == {"name": "a"}
    assert by_name["b.md"].frontmatter is None
    assert by_name["b.md"].ok is True


def test_walk_directory_skips_readme(tmp_path: Path):
    (tmp_path / "README.md").write_text(
        "---\nname: readme\n---\nshould be skipped",
        encoding="utf-8",
    )
    (tmp_path / "other.md").write_text(
        "---\nname: other\n---\nshould fire",
        encoding="utf-8",
    )
    names = sorted(
        r.path.name for r in walk_directory_with_frontmatter(tmp_path)
    )
    assert names == ["other.md"]


def test_walk_directory_skips_default_dirs(tmp_path: Path):
    (tmp_path / "good.md").write_text(
        "---\nname: good\n---\n",
        encoding="utf-8",
    )
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "buried.md").write_text("---\nname: bad\n---\n", encoding="utf-8")

    names = sorted(r.path.name for r in walk_directory_with_frontmatter(tmp_path))
    assert names == ["good.md"]


def test_walk_directory_non_recursive(tmp_path: Path):
    (tmp_path / "top.md").write_text("---\nname: top\n---\n", encoding="utf-8")
    nested = tmp_path / "deeper"
    nested.mkdir()
    (nested / "inside.md").write_text(
        "---\nname: nested\n---\n",
        encoding="utf-8",
    )

    rec = sorted(
        r.path.name for r in walk_directory_with_frontmatter(tmp_path, recursive=True)
    )
    flat = sorted(
        r.path.name for r in walk_directory_with_frontmatter(tmp_path, recursive=False)
    )
    assert rec == ["inside.md", "top.md"]
    assert flat == ["top.md"]


def test_walk_directory_returns_empty_for_missing_root(tmp_path: Path):
    missing = tmp_path / "does-not-exist"
    assert list(walk_directory_with_frontmatter(missing)) == []


def test_walk_directory_swallows_per_file_parse_error(tmp_path: Path):
    (tmp_path / "bad.md").write_text(
        "---\nname: bad\nyaml: : :\n---\nbody",
        encoding="utf-8",
    )
    (tmp_path / "ok.md").write_text(
        "---\nname: ok\n---\nbody",
        encoding="utf-8",
    )
    results = list(walk_directory_with_frontmatter(tmp_path))
    by_name = {r.path.name: r for r in results}
    assert set(by_name.keys()) == {"bad.md", "ok.md"}
    # bad.md surfaces the error but doesn't break discovery of ok.md
    assert by_name["bad.md"].error is not None
    assert by_name["ok.md"].ok is True


def test_carriage_return_line_endings_handled():
    text = "---\r\nname: crlf\r\n---\r\nbody"
    result = parse_frontmatter_text(text)
    assert result.frontmatter == {"name": "crlf"}
    assert "body" in result.body


def test_walk_directory_custom_extensions(tmp_path: Path):
    (tmp_path / "yaml_one.yml").write_text(
        "---\nname: yaml_one\n---\nbody",
        encoding="utf-8",
    )
    (tmp_path / "skipped.md").write_text(
        "---\nname: skipped\n---\nbody",
        encoding="utf-8",
    )
    names = sorted(
        r.path.name
        for r in walk_directory_with_frontmatter(tmp_path, extensions=(".yml",))
    )
    assert names == ["yaml_one.yml"]


def test_parse_frontmatter_str_path_accepted(tmp_path: Path):
    target = tmp_path / "skill.md"
    target.write_text("---\nname: s\n---\nbody", encoding="utf-8")
    result = parse_frontmatter(str(target))
    assert result.ok is True
    assert result.frontmatter == {"name": "s"}


def test_default_constants_are_stable():
    """Pin the public-API constants so refactors are deliberate."""
    from ultron.parsing import frontmatter as fm

    assert fm._FRONTMATTER_DELIMITER == "---"
    assert ".md" in fm._DEFAULT_FILE_EXTENSIONS
    assert ".git" in fm._DEFAULT_SKIP_DIRECTORIES
    assert "node_modules" in fm._DEFAULT_SKIP_DIRECTORIES


def test_frontmatter_result_is_frozen():
    text = "---\nname: x\n---\nbody"
    result = parse_frontmatter_text(text)
    with pytest.raises(Exception):
        result.frontmatter = {"other": "value"}  # type: ignore[misc]
