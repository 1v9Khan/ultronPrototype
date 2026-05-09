"""Tests for ``ultron.openclaw_bridge.workspace.WorkspaceWriter``.

Real filesystem under tmp_path. Concurrency is exercised with
threading + a real :class:`filelock.FileLock` to verify the
advisory locking actually serialises writers.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import threading
import time
from pathlib import Path

import pytest

from ultron.openclaw_bridge.workspace import WorkspaceWriter, WriteResult


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def writer(workspace: Path) -> WorkspaceWriter:
    return WorkspaceWriter(workspace, lock_timeout_s=2.0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_writer_uses_default_workspace_when_none(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ULTRON_OPENCLAW_WORKSPACE", str(tmp_path / "auto"))
    w = WorkspaceWriter()
    assert w.workspace_dir == tmp_path / "auto"


# ---------------------------------------------------------------------------
# write_memory_entry
# ---------------------------------------------------------------------------


async def test_memory_entry_creates_file(writer: WorkspaceWriter, workspace: Path) -> None:
    result = await writer.write_memory_entry("first thought of the day")
    assert result.created is True
    assert result.error is None
    today = _dt.date.today().isoformat()
    target = workspace / "memory" / f"{today}.md"
    content = target.read_text(encoding="utf-8")
    assert "first thought" in content


async def test_memory_entry_appends_to_existing(
    writer: WorkspaceWriter, workspace: Path,
) -> None:
    today = _dt.date.today()
    await writer.write_memory_entry("one", date=today)
    result = await writer.write_memory_entry("two", date=today)
    assert result.created is False
    target = workspace / "memory" / f"{today.isoformat()}.md"
    content = target.read_text(encoding="utf-8")
    assert "one" in content and "two" in content
    # Two entries means at least two list bullets.
    assert content.count("- ") >= 2


async def test_memory_entry_rejects_empty(writer: WorkspaceWriter) -> None:
    result = await writer.write_memory_entry("   ")
    assert result.error == "empty entry"
    assert result.bytes_written == 0


async def test_memory_entry_optional_timestamp(
    writer: WorkspaceWriter, workspace: Path,
) -> None:
    await writer.write_memory_entry("plain entry", prefix_timestamp=False)
    today = _dt.date.today().isoformat()
    content = (workspace / "memory" / f"{today}.md").read_text(encoding="utf-8")
    # No "HH:MM —" pattern when timestamp omitted.
    assert "—" not in content
    assert "plain entry" in content


# ---------------------------------------------------------------------------
# update_memory_md
# ---------------------------------------------------------------------------


async def test_update_memory_md_replaces_section(
    writer: WorkspaceWriter, workspace: Path,
) -> None:
    target = workspace / "MEMORY.md"
    target.write_text(
        "# Top\n\n## Alpha\n\nold alpha\n\n## Beta\n\nbeta body\n",
        encoding="utf-8",
    )
    result = await writer.update_memory_md("Alpha", "new alpha body")
    assert result.section_replaced == "Alpha"
    content = target.read_text(encoding="utf-8")
    assert "new alpha body" in content
    assert "old alpha" not in content
    # Beta section preserved.
    assert "## Beta" in content
    assert "beta body" in content


async def test_update_memory_md_creates_when_missing_default(
    writer: WorkspaceWriter, workspace: Path,
) -> None:
    target = workspace / "MEMORY.md"
    target.write_text("# Top\n\n## Beta\n\nbeta body\n", encoding="utf-8")
    result = await writer.update_memory_md("Alpha", "fresh alpha")
    assert result.section_replaced == "Alpha"
    content = target.read_text(encoding="utf-8")
    assert "## Alpha" in content
    assert "fresh alpha" in content
    assert "## Beta" in content                                     # preserved


async def test_update_memory_md_skips_when_create_false(
    writer: WorkspaceWriter, workspace: Path,
) -> None:
    target = workspace / "MEMORY.md"
    target.write_text("# Top\n\n## Beta\n\nbeta body\n", encoding="utf-8")
    result = await writer.update_memory_md(
        "Alpha", "ignored", create_if_missing=False,
    )
    assert result.section_replaced is None
    assert "section not found" in (result.error or "")


async def test_update_memory_md_writes_atomic_replace(
    writer: WorkspaceWriter, workspace: Path,
) -> None:
    target = workspace / "MEMORY.md"
    await writer.update_memory_md("Alpha", "value 1")
    snapshot_1 = target.read_text(encoding="utf-8")
    await writer.update_memory_md("Alpha", "value 2")
    snapshot_2 = target.read_text(encoding="utf-8")
    # No temp-file leaked.
    leftovers = list(target.parent.glob(target.name + ".*.tmp"))
    assert leftovers == []
    assert "value 1" in snapshot_1 and "value 2" in snapshot_2


# ---------------------------------------------------------------------------
# update_user_md
# ---------------------------------------------------------------------------


async def test_update_user_md_replaces_full_content(
    writer: WorkspaceWriter, workspace: Path,
) -> None:
    target = workspace / "USER.md"
    target.write_text("old content", encoding="utf-8")
    result = await writer.update_user_md("# Updated\n\nnew content\n")
    assert result.error is None
    content = target.read_text(encoding="utf-8")
    assert "old content" not in content
    assert "new content" in content


# ---------------------------------------------------------------------------
# Concurrency — advisory lockfile serialises writers
# ---------------------------------------------------------------------------


def test_concurrent_appends_serialise(workspace: Path) -> None:
    """Two threads append in parallel; advisory lockfile prevents
    interleaving so all entries land intact."""
    writer = WorkspaceWriter(workspace, lock_timeout_s=5.0)
    today = _dt.date.today()

    def worker(idx: int) -> None:
        async def _go() -> None:
            for j in range(20):
                await writer.write_memory_entry(
                    f"thread-{idx}-entry-{j}",
                    date=today, prefix_timestamp=False,
                )
        asyncio.run(_go())

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
        assert not t.is_alive()

    target = workspace / "memory" / f"{today.isoformat()}.md"
    content = target.read_text(encoding="utf-8")
    # 4 threads * 20 entries = 80 entries.
    bullet_count = content.count("- thread-")
    assert bullet_count == 80


# ---------------------------------------------------------------------------
# Splice helper
# ---------------------------------------------------------------------------


def test_splice_section_no_change_when_missing_and_no_create() -> None:
    original = "# Top\n\n## Beta\n\nbody\n"
    updated, changed = WorkspaceWriter._splice_section(
        original, "Alpha", "new", create_if_missing=False,
    )
    assert updated == original
    assert changed is False


def test_splice_section_appends_when_create_and_missing() -> None:
    original = "# Top\n\n## Beta\n\nbody\n"
    updated, changed = WorkspaceWriter._splice_section(
        original, "Alpha", "new", create_if_missing=True,
    )
    assert changed is True
    assert "## Alpha" in updated
    assert "new" in updated
    # Original Beta still present.
    assert "## Beta" in updated


def test_splice_section_replaces_inline() -> None:
    original = "# Top\n\n## Alpha\n\nold\n\n## Beta\n\nbeta body\n"
    updated, changed = WorkspaceWriter._splice_section(
        original, "Alpha", "new", create_if_missing=False,
    )
    assert changed is True
    assert "old" not in updated
    assert "new" in updated
    # Beta untouched.
    assert "beta body" in updated
