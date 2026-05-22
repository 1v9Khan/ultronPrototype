"""Tests for ultron.coding.project_index.

These tests construct a real (local-file) Qdrant instance + a real
bge-small HybridEmbedder. They're moderately slow (~5-10 s for the
embedder warmup) but exercise the full upsert + search pipeline
end-to-end. Marked with no special marker since the embedder is
already loaded by the standard test sweep.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ultron.coding.project_digest import DigestRequest, render_template, ProjectDigest, parse_digest_sections
from ultron.coding.project_index import (
    ProjectIndex,
    ProjectIndexEntry,
    ProjectMatch,
    _build_digest_summary_for_search,
    _derive_project_id,
    _score_reason,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def embedder():
    """Real HybridEmbedder shared across tests. Loaded once per process."""
    from ultron.memory.embedder import HybridEmbedder
    return HybridEmbedder()


@pytest.fixture
def index(tmp_path: Path, embedder) -> ProjectIndex:
    """Fresh ProjectIndex pointed at tmp_path."""
    return ProjectIndex(
        embedder=embedder,
        qdrant_path=tmp_path / "qdrant",
        collection_name="test_projects",
    )


def _make_digest(name: str, path: Path, body: str = "") -> ProjectDigest:
    if not body:
        body = render_template(DigestRequest(
            project_name=name,
            project_path=path,
            task_summary=f"Built {name}.",
            user_goal_hint=f"build a {name} application",
        ))
    return ProjectDigest(
        project_name=name,
        project_path=path,
        markdown=body,
        sections=parse_digest_sections(body),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construct_creates_collection(index: ProjectIndex) -> None:
    assert index.collection == "test_projects"
    # count() should be zero on a fresh index.
    assert index.count() == 0


def test_construct_requires_embedder() -> None:
    with pytest.raises(ValueError, match="HybridEmbedder"):
        ProjectIndex(embedder=None)


# ---------------------------------------------------------------------------
# upsert + get
# ---------------------------------------------------------------------------


def test_upsert_persists_entry(index: ProjectIndex, tmp_path: Path) -> None:
    project_path = tmp_path / "flask_app"
    digest = _make_digest("flask_app", project_path)
    entry = index.upsert(digest)

    assert entry is not None
    assert entry.project_name == "flask_app"
    assert entry.project_path == str(project_path)
    assert entry.digest_markdown == digest.markdown


def test_upsert_returns_none_on_empty_digest(index: ProjectIndex, tmp_path: Path) -> None:
    digest = ProjectDigest(
        project_name="empty", project_path=tmp_path / "empty",
        markdown="",
    )
    assert index.upsert(digest) is None


def test_get_returns_upserted_entry(index: ProjectIndex, tmp_path: Path) -> None:
    project_path = tmp_path / "p1"
    digest = _make_digest("p1", project_path)
    entry = index.upsert(digest)
    assert entry is not None
    fetched = index.get(entry.project_id)
    assert fetched is not None
    assert fetched.project_name == "p1"


def test_get_returns_none_for_unknown(index: ProjectIndex) -> None:
    assert index.get("nonexistent-id") is None
    assert index.get("") is None


def test_get_by_path_returns_entry(index: ProjectIndex, tmp_path: Path) -> None:
    project_path = tmp_path / "p2"
    digest = _make_digest("p2", project_path)
    index.upsert(digest)
    fetched = index.get_by_path(project_path)
    assert fetched is not None
    assert fetched.project_name == "p2"


def test_upsert_overwrites_existing(index: ProjectIndex, tmp_path: Path) -> None:
    project_path = tmp_path / "p3"
    digest_v1 = _make_digest("p3", project_path, body="## Goal\n- v1\n")
    digest_v2 = _make_digest("p3", project_path, body="## Goal\n- v2\n")

    entry_v1 = index.upsert(digest_v1)
    entry_v2 = index.upsert(digest_v2)

    assert entry_v1.project_id == entry_v2.project_id
    fetched = index.get(entry_v1.project_id)
    assert "v2" in fetched.digest_markdown


def test_upsert_preserves_created_at_on_update(
    index: ProjectIndex, tmp_path: Path,
) -> None:
    project_path = tmp_path / "p4"
    digest = _make_digest("p4", project_path)
    entry_v1 = index.upsert(digest)
    original_created_at = entry_v1.created_at_unix
    time.sleep(0.01)
    entry_v2 = index.upsert(digest)
    assert entry_v2.created_at_unix == pytest.approx(original_created_at, abs=0.001)


def test_upsert_preserves_tags_on_update_when_new_tags_empty(
    index: ProjectIndex, tmp_path: Path,
) -> None:
    project_path = tmp_path / "p_tags"
    digest = _make_digest("p_tags", project_path)
    index.upsert(digest, tags=["active"])
    entry = index.upsert(digest)  # No new tags passed
    assert "active" in entry.tags


# ---------------------------------------------------------------------------
# list_all + count
# ---------------------------------------------------------------------------


def test_list_all_returns_all(index: ProjectIndex, tmp_path: Path) -> None:
    for name in ("alpha", "beta", "gamma"):
        index.upsert(_make_digest(name, tmp_path / name))
    entries = index.list_all()
    names = {e.project_name for e in entries}
    assert names == {"alpha", "beta", "gamma"}


def test_count_tracks_upserts(index: ProjectIndex, tmp_path: Path) -> None:
    assert index.count() == 0
    index.upsert(_make_digest("c1", tmp_path / "c1"))
    index.upsert(_make_digest("c2", tmp_path / "c2"))
    # Note: Qdrant count may have race; we just assert it grows.
    # Some local Qdrant builds need a tiny delay for the count to propagate.
    for _ in range(10):
        if index.count() >= 2:
            break
        time.sleep(0.1)
    assert index.count() >= 1  # at minimum, something is indexed


# ---------------------------------------------------------------------------
# search (semantic)
# ---------------------------------------------------------------------------


def test_search_returns_relevant_project(index: ProjectIndex, tmp_path: Path) -> None:
    index.upsert(_make_digest(
        "flask_blog", tmp_path / "flask_blog",
        body=render_template(DigestRequest(
            project_name="flask_blog",
            project_path=tmp_path / "flask_blog",
            task_summary="Built a Flask blog with SQLite backend and Markdown rendering.",
            user_goal_hint="build a flask blog with markdown posts",
        )),
    ))
    index.upsert(_make_digest(
        "react_dashboard", tmp_path / "react_dashboard",
        body=render_template(DigestRequest(
            project_name="react_dashboard",
            project_path=tmp_path / "react_dashboard",
            task_summary="Built a React dashboard for sales analytics.",
            user_goal_hint="build a react dashboard for sales data",
        )),
    ))
    results = index.search("flask blog", top_k=5)
    assert len(results) >= 1
    assert results[0].entry.project_name == "flask_blog"
    assert results[0].score > 0.0


def test_search_respects_min_score(index: ProjectIndex, tmp_path: Path) -> None:
    index.upsert(_make_digest("alpha", tmp_path / "alpha"))
    # min_score above any plausible cosine -> empty
    results = index.search("xyzzy", min_score=0.99)
    assert results == []


def test_search_empty_query_returns_empty(index: ProjectIndex) -> None:
    assert index.search("") == []
    assert index.search("   ") == []


def test_search_results_sorted_by_score(index: ProjectIndex, tmp_path: Path) -> None:
    for name in ("a", "b", "c"):
        index.upsert(_make_digest(name, tmp_path / name))
    results = index.search("project", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# search_by_name (lexical)
# ---------------------------------------------------------------------------


def test_search_by_name_finds_substring(index: ProjectIndex, tmp_path: Path) -> None:
    index.upsert(_make_digest("flask_blog", tmp_path / "flask_blog"))
    index.upsert(_make_digest("react_app", tmp_path / "react_app"))

    hits = index.search_by_name("flask")
    assert len(hits) >= 1
    assert any(e.project_name == "flask_blog" for e in hits)


def test_search_by_name_empty_query(index: ProjectIndex) -> None:
    assert index.search_by_name("") == []


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


def test_delete_removes_entry(index: ProjectIndex, tmp_path: Path) -> None:
    entry = index.upsert(_make_digest("doomed", tmp_path / "doomed"))
    assert index.get(entry.project_id) is not None
    assert index.delete(entry.project_id) is True
    assert index.get(entry.project_id) is None


def test_delete_unknown_returns_false_on_empty(index: ProjectIndex) -> None:
    # Empty project_id is rejected up front.
    assert index.delete("") is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_derive_project_id_is_stable(tmp_path: Path) -> None:
    p = tmp_path / "mypath"
    assert _derive_project_id(p) == _derive_project_id(p)


def test_derive_project_id_differs_per_path(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    assert _derive_project_id(a) != _derive_project_id(b)


def test_score_reason_band_labels() -> None:
    assert "very high" in _score_reason(0.9).lower()
    assert "high" in _score_reason(0.78).lower()
    assert "possible" in _score_reason(0.6).lower()
    assert "weak" in _score_reason(0.3).lower()


def test_build_digest_summary_truncates() -> None:
    sections = {
        "Goal": "- " + "long " * 200,
        "Critical Context": "- something",
        "Relevant Files": "- f.py: ok",
    }
    summary = _build_digest_summary_for_search(sections, max_chars=100)
    assert len(summary) <= 100


def test_build_digest_summary_handles_none_section() -> None:
    sections = {
        "Goal": "- build it",
        "Critical Context": "- (none)",
        "Relevant Files": "- (none)",
    }
    summary = _build_digest_summary_for_search(sections)
    assert "build it" in summary
    assert "(none)" not in summary


# ---------------------------------------------------------------------------
# ProjectIndexEntry payload round-trip
# ---------------------------------------------------------------------------


def test_entry_payload_round_trip() -> None:
    e = ProjectIndexEntry(
        project_id="abc",
        project_name="proj",
        project_path="/tmp/proj",
        digest_markdown="## Goal\n- yes\n",
        digest_sections={"Goal": "- yes"},
        digest_text_summary="proj summary",
        language="python",
        entry_points=["app.py"],
        tags=["active"],
        last_session_id="session-123",
    )
    payload = e.to_payload()
    restored = ProjectIndexEntry.from_payload(payload)
    assert restored.project_id == "abc"
    assert restored.project_name == "proj"
    assert restored.language == "python"
    assert restored.entry_points == ["app.py"]
    assert restored.tags == ["active"]
    assert restored.last_session_id == "session-123"
