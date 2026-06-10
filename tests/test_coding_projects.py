"""Tests for the project registry + voice-reference resolver."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from ultron.coding.projects import (
    Project,
    ProjectRegistry,
    ProjectResolver,
    ResolutionKind,
    new_sandbox_project,
    slugify_for_path,
)


# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------


def test_slugify_basic():
    assert slugify_for_path("Weather Fetcher") == "weather_fetcher"
    assert slugify_for_path("My Flask App!") == "my_flask_app"


def test_slugify_strips_leading_trailing_underscores():
    assert slugify_for_path("  --foo--  ") == "foo"


def test_slugify_falls_back_when_empty():
    out = slugify_for_path("???")
    assert out.startswith("project_")


# ---------------------------------------------------------------------------
# Registry CRUD
# ---------------------------------------------------------------------------


def _registry(tmp_path: Path) -> ProjectRegistry:
    return ProjectRegistry(path=tmp_path / "projects.json")


def _proj(name: str, **kw) -> Project:
    return Project(
        name=name,
        path=kw.get("path", "/abs/" + name.lower().replace(" ", "_")),
        aliases=kw.get("aliases", []),
        language=kw.get("language", "python"),
        description=kw.get("description", ""),
        tags=kw.get("tags", []),
    )


def test_registry_starts_empty(tmp_path: Path):
    r = _registry(tmp_path)
    assert r.list() == []
    assert r.get("anything") is None


def test_registry_add_and_get(tmp_path: Path):
    r = _registry(tmp_path)
    p = r.add(_proj("Flask App"))
    assert p.name == "Flask App"
    assert r.get("flask app").name == "Flask App"
    assert len(r.list()) == 1


def test_registry_persists_across_instances(tmp_path: Path):
    r = _registry(tmp_path)
    r.add(_proj("Flask App"))
    r.add(_proj("Dashboard", aliases=["the dashboard"]))

    r2 = ProjectRegistry(path=tmp_path / "projects.json")
    names = sorted(p.name for p in r2.list())
    assert names == ["Dashboard", "Flask App"]


def test_registry_rejects_duplicate_name(tmp_path: Path):
    r = _registry(tmp_path)
    r.add(_proj("Flask App"))
    with pytest.raises(ValueError):
        r.add(_proj("flask app"))  # case-insensitive collision


def test_registry_remove(tmp_path: Path):
    r = _registry(tmp_path)
    r.add(_proj("Flask App"))
    assert r.remove("flask app") is True
    assert r.remove("flask app") is False
    assert r.list() == []


def test_registry_touch_updates_last_accessed(tmp_path: Path):
    r = _registry(tmp_path)
    p = r.add(_proj("Flask App"))
    before = p.last_accessed
    import time
    time.sleep(0.01)
    r.touch("flask app")
    after = r.get("Flask App").last_accessed
    assert after > before


# ---------------------------------------------------------------------------
# Resolver: lexical paths (no embedder).
# ---------------------------------------------------------------------------


def test_resolver_exact_match(tmp_path: Path):
    r = _registry(tmp_path)
    r.add(_proj("Flask App"))
    res = ProjectResolver(r).resolve("Flask App")
    assert res.kind == ResolutionKind.EXACT
    assert res.project.name == "Flask App"
    assert res.confidence == 1.0


def test_resolver_alias_match(tmp_path: Path):
    r = _registry(tmp_path)
    r.add(_proj("Flask App", aliases=["the api", "the backend"]))
    res = ProjectResolver(r).resolve("the api")
    assert res.kind == ResolutionKind.ALIAS
    assert res.project.name == "Flask App"


def test_resolver_substring_unique(tmp_path: Path):
    r = _registry(tmp_path)
    r.add(_proj("Flask App"))
    r.add(_proj("Dashboard"))
    res = ProjectResolver(r).resolve("flask")
    assert res.kind == ResolutionKind.SUBSTRING
    assert res.project.name == "Flask App"


def test_resolver_substring_multiple_returns_ambiguous_without_embedder(tmp_path: Path):
    r = _registry(tmp_path)
    r.add(_proj("Flask App", description="user-facing API"))
    r.add(_proj("Other Flask Service", description="background worker"))
    res = ProjectResolver(r).resolve("flask")
    assert res.kind == ResolutionKind.AMBIGUOUS
    names = sorted(p.name for p in res.candidates)
    assert names == ["Flask App", "Other Flask Service"]


def test_resolver_returns_not_found_on_empty_registry(tmp_path: Path):
    r = _registry(tmp_path)
    res = ProjectResolver(r).resolve("anything")
    assert res.kind == ResolutionKind.NOT_FOUND


def test_resolver_empty_reference(tmp_path: Path):
    r = _registry(tmp_path)
    r.add(_proj("Flask App"))
    res = ProjectResolver(r).resolve("   ")
    assert res.kind == ResolutionKind.NOT_FOUND


# ---------------------------------------------------------------------------
# Resolver: semantic path with a stub embedder.
# ---------------------------------------------------------------------------


class _StubEmbedder:
    """Tiny embedder that hashes text into a deterministic vector. Good
    enough to exercise the resolver's semantic-match codepath without
    pulling in the real sentence-transformer."""

    def __init__(self, mapping: dict[str, np.ndarray]):
        self.mapping = mapping

    def encode_query_dense(self, text: str) -> np.ndarray:
        return self.mapping[text.lower()]

    def encode_dense(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self.mapping[t.lower()] for t in texts])


def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def test_resolver_semantic_disambiguates_substring_hits(tmp_path: Path):
    r = _registry(tmp_path)
    r.add(_proj("Flask Api", description="user-facing api server"))
    r.add(_proj("Other Flask Service", description="background worker"))

    # Embeddings: "the api" close to "flask api ...", far from "other ...".
    near_api = _norm([1.0, 0.0])
    near_worker = _norm([0.0, 1.0])
    queries = {
        "the api": near_api,
        "flask api. user-facing api server. python": near_api,
        "other flask service. background worker. python": near_worker,
    }
    embedder = _StubEmbedder(queries)
    res = ProjectResolver(r, embedder=embedder, semantic_threshold=0.5).resolve(
        "the api"
    )
    # "the api" matches the alias-less project via substring on description+name.
    # Both projects contain "flask"; both are substring hits. Semantic wins.
    assert res.project is not None
    assert res.project.name == "Flask Api"


def test_resolver_semantic_misses_when_threshold_not_met(tmp_path: Path):
    r = _registry(tmp_path)
    r.add(_proj("Calculator", description="basic math toy"))

    # All projects "near" but below threshold.
    queries = {
        "weather thing": _norm([1.0, 0.0]),
        "calculator. basic math toy. python": _norm([0.4, 1.0]),
    }
    embedder = _StubEmbedder(queries)
    res = ProjectResolver(r, embedder=embedder, semantic_threshold=0.95).resolve(
        "weather thing"
    )
    assert res.kind == ResolutionKind.NOT_FOUND


# ---------------------------------------------------------------------------
# Sandbox project creation.
# ---------------------------------------------------------------------------


def test_new_sandbox_project_creates_dir(tmp_path: Path):
    r = _registry(tmp_path)
    p = new_sandbox_project(
        r,
        name="Hello World",
        sandbox_root=tmp_path / "sandbox",
        aliases=["hello"],
    )
    assert Path(p.path).is_dir()
    assert Path(p.path).parent == (tmp_path / "sandbox")
    assert Path(p.path).name == "hello_world"
    assert r.get("Hello World").path == p.path


def test_new_sandbox_project_handles_collisions(tmp_path: Path):
    r = _registry(tmp_path)
    p1 = new_sandbox_project(
        r, name="Alpha",
        sandbox_root=tmp_path / "sandbox",
    )
    # Re-using the same name -- the registry rejects duplicate names, but
    # the directory-naming code should still pick a unique slug if the
    # caller dropped the duplicate-name check (which a legitimate caller
    # might do if they removed the prior project but left the folder).
    # rmtree (not rmdir): a fresh sandbox project is no longer empty --
    # it is git-initialised for context isolation.
    import shutil

    shutil.rmtree(p1.path)  # so we have a fresh slot
    r.remove("Alpha")
    p2 = new_sandbox_project(
        r, name="Alpha",
        sandbox_root=tmp_path / "sandbox",
    )
    assert Path(p2.path).is_dir()
