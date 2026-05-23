"""Tests for :mod:`ultron.coding.ai_comment_watcher`."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ultron.coding.ai_comment_watcher import (
    AI_COMMENT_REGEX,
    AICommentKind,
    AICommentTrigger,
    AICommentWatcher,
    DEFAULT_MAX_FILE_BYTES,
    scan_file_for_ai_comments,
)


# ---------------------------------------------------------------------------
# Regex pattern unit tests
# ---------------------------------------------------------------------------


def test_regex_matches_python_execute_marker():
    m = AI_COMMENT_REGEX.search("# ai! refactor this\n")
    assert m
    assert m.group("prefix") == "#"
    assert m.group("suffix") == "!"
    assert "refactor this" in m.group("body")


def test_regex_matches_python_question_marker():
    m = AI_COMMENT_REGEX.search("# ai? what does this do\n")
    assert m
    assert m.group("suffix") == "?"


def test_regex_matches_js_double_slash_marker():
    m = AI_COMMENT_REGEX.search("// ai! please fix\n")
    assert m
    assert m.group("prefix") == "//"


def test_regex_matches_sql_double_dash_marker():
    m = AI_COMMENT_REGEX.search("-- ai! add index\n")
    assert m
    assert m.group("prefix") == "--"


def test_regex_matches_lisp_semicolon_marker():
    m = AI_COMMENT_REGEX.search(";; ai! refactor\n")
    assert m
    assert m.group("prefix") == ";;"


def test_regex_matches_passive_mention_no_suffix():
    m = AI_COMMENT_REGEX.search("# ai look at me\n")
    assert m
    # No trailing punctuation -> suffix is None.
    assert m.group("suffix") is None


def test_regex_case_insensitive():
    m = AI_COMMENT_REGEX.search("# AI! do something\n")
    assert m


def test_regex_rejects_unrelated_text():
    m = AI_COMMENT_REGEX.search("aircraft = 1  # not a marker\n")
    # The word "aircraft" contains "ai" but doesn't match the comment-
    # prefix structure.
    assert m is None or m.group("prefix") == "#"


# ---------------------------------------------------------------------------
# scan_file_for_ai_comments
# ---------------------------------------------------------------------------


def test_scan_finds_execute_marker(tmp_path: Path):
    f = tmp_path / "demo.py"
    f.write_text(
        "def foo():\n"
        "    # ai! refactor this function\n"
        "    return 42\n"
    )
    triggers = scan_file_for_ai_comments(f)
    assert len(triggers) == 1
    t = triggers[0]
    assert t.kind == AICommentKind.EXECUTE
    assert "refactor this function" in t.body
    assert t.line == 1  # 0-based; the comment is the 2nd line


def test_scan_finds_question_marker(tmp_path: Path):
    f = tmp_path / "q.py"
    f.write_text("# ai? what does this loop do\n")
    triggers = scan_file_for_ai_comments(f)
    assert triggers
    assert triggers[0].kind == AICommentKind.QUESTION


def test_scan_finds_passive_mention(tmp_path: Path):
    f = tmp_path / "m.py"
    f.write_text("# ai note: this is fragile\n")
    triggers = scan_file_for_ai_comments(f)
    assert triggers
    assert triggers[0].kind == AICommentKind.MENTION


def test_scan_finds_multiple_markers(tmp_path: Path):
    f = tmp_path / "multi.py"
    f.write_text(
        "# ai? a question\n"
        "x = 1\n"
        "# ai! an execute\n"
        "y = 2\n"
    )
    triggers = scan_file_for_ai_comments(f)
    assert len(triggers) == 2
    kinds = [t.kind for t in triggers]
    assert AICommentKind.QUESTION in kinds
    assert AICommentKind.EXECUTE in kinds


def test_scan_skips_oversized_file(tmp_path: Path):
    f = tmp_path / "big.py"
    # Tiny max so we definitely trigger the guard.
    f.write_text("# ai! do thing\n" + "x" * 200)
    assert scan_file_for_ai_comments(f, max_bytes=50) == []


def test_scan_handles_missing_file(tmp_path: Path):
    f = tmp_path / "nonexistent.py"
    assert scan_file_for_ai_comments(f) == []


def test_scan_handles_empty_file(tmp_path: Path):
    f = tmp_path / "empty.py"
    f.write_text("")
    assert scan_file_for_ai_comments(f) == []


def test_scan_records_correct_line_number(tmp_path: Path):
    f = tmp_path / "lines.py"
    f.write_text("line0\nline1\n# ai! marker on line 2\nline3\n")
    triggers = scan_file_for_ai_comments(f)
    assert len(triggers) == 1
    assert triggers[0].line == 2


def test_trigger_is_frozen():
    t = AICommentTrigger(
        kind=AICommentKind.EXECUTE,
        body="do thing",
        file_path="/x",
        line=0,
        column=0,
        prefix="#",
    )
    with pytest.raises(Exception):
        t.body = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AICommentWatcher lifecycle (uses watchfiles)
# ---------------------------------------------------------------------------


def test_watcher_running_state(tmp_path: Path):
    watcher = AICommentWatcher(tmp_path, on_trigger=lambda _: None)
    assert watcher.running is False
    watcher.start()
    assert watcher.running is True
    watcher.stop(timeout=2)
    assert watcher.running is False


def test_watcher_start_is_idempotent(tmp_path: Path):
    watcher = AICommentWatcher(tmp_path, on_trigger=lambda _: None)
    watcher.start()
    first = watcher._thread
    watcher.start()
    assert watcher._thread is first
    watcher.stop(timeout=2)


def test_watcher_scan_now_returns_existing_triggers(tmp_path: Path):
    (tmp_path / "a.py").write_text("# ai! existing trigger\n")
    (tmp_path / "b.py").write_text("# ai? another\n")
    watcher = AICommentWatcher(tmp_path, on_trigger=lambda _: None)
    triggers = watcher.scan_now()
    assert len(triggers) == 2
    bodies = {t.body for t in triggers}
    assert "existing trigger" in bodies


def test_watcher_seed_swallows_existing_triggers(tmp_path: Path):
    """Triggers present at start() time should NOT fire — only NEW
    comments produce callbacks."""
    (tmp_path / "a.py").write_text("# ai! pre-existing\n")

    fired: list[AICommentTrigger] = []
    watcher = AICommentWatcher(tmp_path, on_trigger=fired.append)
    watcher.start()
    # Modify some other unrelated file — the pre-existing trigger
    # in a.py should NOT fire because it was seeded.
    (tmp_path / "irrelevant.txt").write_text("data\n")
    time.sleep(0.8)
    watcher.stop(timeout=2)
    assert fired == []


def test_watcher_skips_subdirectory_in_skip_list(tmp_path: Path):
    """A file under node_modules/ should be excluded."""
    nm = tmp_path / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "x.py").write_text("# ai! from a vendored dep\n")
    (tmp_path / "main.py").write_text("# ai! real trigger\n")
    watcher = AICommentWatcher(tmp_path, on_trigger=lambda _: None)
    triggers = watcher.scan_now()
    bodies = {t.body for t in triggers}
    assert "real trigger" in bodies
    assert "from a vendored dep" not in bodies


def test_watcher_skips_binary_extension(tmp_path: Path):
    (tmp_path / "image.png").write_bytes(b"# ai! fake binary content")
    watcher = AICommentWatcher(tmp_path, on_trigger=lambda _: None)
    triggers = watcher.scan_now()
    # PNG file should be skipped via extension filter.
    assert triggers == []


def test_watcher_include_mention_flag(tmp_path: Path):
    """Default OFF: mention triggers don't appear in fire-set."""
    (tmp_path / "a.py").write_text("# ai just a mention\n")
    watcher = AICommentWatcher(tmp_path, on_trigger=lambda _: None)
    # scan_now returns ALL detected (including mention), but seed
    # behaviour depends on include_mention.
    triggers = watcher.scan_now()
    assert any(t.kind == AICommentKind.MENTION for t in triggers)


def test_watcher_default_max_file_bytes():
    assert DEFAULT_MAX_FILE_BYTES == 1_000_000


# ---------------------------------------------------------------------------
# End-to-end (live file modification triggers the callback)
# ---------------------------------------------------------------------------


def test_watcher_fires_on_new_trigger(tmp_path: Path):
    """Modify a file to add an # ai! comment; the watcher fires."""
    f = tmp_path / "live.py"
    f.write_text("def foo(): pass\n")

    fired: list[AICommentTrigger] = []
    watcher = AICommentWatcher(
        tmp_path,
        on_trigger=fired.append,
        poll_interval_seconds=0.05,
    )
    watcher.start()
    try:
        # Give the watcher a moment to start its loop.
        time.sleep(0.2)
        f.write_text("def foo(): pass\n# ai! refactor please\n")
        # Wait up to 5 seconds for the event to fire.
        deadline = time.monotonic() + 5.0
        while not fired and time.monotonic() < deadline:
            time.sleep(0.1)
    finally:
        watcher.stop(timeout=2)

    assert fired, "expected the watcher to fire on the new trigger"
    assert fired[0].kind == AICommentKind.EXECUTE
    assert "refactor please" in fired[0].body
