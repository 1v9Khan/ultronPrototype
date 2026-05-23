"""Tests for :mod:`ultron.coding.repo_map`."""

from __future__ import annotations

from pathlib import Path

import pytest

from ultron.coding.repo_map import (
    RepoMap,
    SKIP_DIRECTORIES,
    extract_idents_from_text,
    find_source_files,
)
from ultron.utils.mtime_cache import MtimeCache


# ---------------------------------------------------------------------------
# extract_idents_from_text
# ---------------------------------------------------------------------------


def test_extract_idents_finds_snake_case():
    idents = extract_idents_from_text(
        "Fix the snapshot_guard race in background_summarizer"
    )
    assert "snapshot_guard" in idents
    assert "background_summarizer" in idents


def test_extract_idents_finds_kebab_case():
    idents = extract_idents_from_text("Look at the search-replace fuzz cascade")
    assert "search-replace" in idents


def test_extract_idents_finds_camel_case():
    idents = extract_idents_from_text(
        "RelativeIndenter and ProjectSupervisor both matter"
    )
    assert "RelativeIndenter" in idents
    assert "ProjectSupervisor" in idents


def test_extract_idents_finds_dotted_paths():
    idents = extract_idents_from_text(
        "Edit ultron.coding.repo_map and verify ultron.utils.mtime_cache"
    )
    # Both dotted forms picked up.
    assert any("ultron.coding" in i for i in idents)
    assert any("ultron.utils" in i for i in idents)


def test_extract_idents_ignores_plain_words():
    idents = extract_idents_from_text(
        "the quick brown fox jumps over the lazy dog"
    )
    assert idents == set()


def test_extract_idents_empty_text():
    assert extract_idents_from_text("") == set()


# ---------------------------------------------------------------------------
# find_source_files
# ---------------------------------------------------------------------------


def test_find_source_files_lists_python(tmp_path: Path):
    (tmp_path / "a.py").write_text("def alpha(): pass\n")
    (tmp_path / "b.js").write_text("function beta() {}\n")
    (tmp_path / "readme.md").write_text("# readme\n")
    (tmp_path / "data.bin").write_bytes(b"\x00\x01")
    paths = find_source_files(tmp_path)
    names = {p.name for p in paths}
    # Source languages with grep_ast support are included.
    assert "a.py" in names
    assert "b.js" in names
    # data.bin (no known language) is excluded.
    assert "data.bin" not in names


def test_find_source_files_skips_node_modules(tmp_path: Path):
    (tmp_path / "src.py").write_text("def alpha(): pass\n")
    nm = tmp_path / "node_modules" / "package"
    nm.mkdir(parents=True)
    (nm / "dep.py").write_text("def beta(): pass\n")
    paths = find_source_files(tmp_path)
    names = {p.relative_to(tmp_path).as_posix() for p in paths}
    assert "src.py" in names
    # No file under node_modules should appear.
    assert all("node_modules" not in n for n in names)


def test_find_source_files_empty_dir(tmp_path: Path):
    assert find_source_files(tmp_path) == []


def test_find_source_files_single_file(tmp_path: Path):
    f = tmp_path / "lone.py"
    f.write_text("def x(): pass\n")
    assert find_source_files(f) == [f]


def test_skip_directories_contains_common_entries():
    expected = {"__pycache__", ".git", ".venv", "node_modules", "dist", "build"}
    assert expected.issubset(SKIP_DIRECTORIES)


# ---------------------------------------------------------------------------
# RepoMap.get_map
# ---------------------------------------------------------------------------


_SAMPLE_PY = """\
import os


class Greeter:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"hello {self.name}"


def make_greeter(name):
    return Greeter(name)


def shout(text):
    return text.upper()
"""

_CALLER_PY = """\
from greeter import Greeter, make_greeter, shout


def run():
    g = make_greeter("world")
    print(g.greet())
    print(shout("hi"))
"""


def _seed_project(tmp_path: Path) -> Path:
    (tmp_path / "greeter.py").write_text(_SAMPLE_PY)
    (tmp_path / "caller.py").write_text(_CALLER_PY)
    (tmp_path / "README.md").write_text("# demo\n")
    return tmp_path


def test_repo_map_renders_non_empty(tmp_path: Path):
    _seed_project(tmp_path)
    rm = RepoMap(tmp_path, max_map_tokens=1024, max_map_tokens_no_chat=4096)
    rendered = rm.get_map()
    assert rendered, "expected non-empty repo map"
    # Greeter (a heavily-referenced class) should be in the map.
    assert "Greeter" in rendered


def test_repo_map_empty_project_returns_empty(tmp_path: Path):
    rm = RepoMap(tmp_path, max_map_tokens=512, max_map_tokens_no_chat=2048)
    assert rm.get_map() == ""


def test_repo_map_with_mentioned_ident_biases(tmp_path: Path):
    _seed_project(tmp_path)
    # Add a 3rd file with a function that's NOT heavily referenced.
    (tmp_path / "extra.py").write_text(
        "def obscure_helper():\n    return 1\n"
    )
    rm = RepoMap(tmp_path, max_map_tokens=200, max_map_tokens_no_chat=200)
    # Without bias, obscure_helper may or may not surface.
    biased = rm.get_map(mentioned_idents={"obscure_helper"})
    assert "obscure_helper" in biased


def test_repo_map_excludes_chat_files(tmp_path: Path):
    _seed_project(tmp_path)
    rm = RepoMap(tmp_path, max_map_tokens=2048, max_map_tokens_no_chat=4096)
    chat = [tmp_path / "greeter.py"]
    rendered = rm.get_map(chat_files=chat)
    # The chat file should NOT appear in the rendered output; the LLM
    # already has it. Path appears as filename token only.
    # Sanity-check: caller.py still appears (it references Greeter).
    assert "caller.py" in rendered
    # greeter.py shouldn't be rendered as a tagged file (no `greeter.py:`
    # block); orphan-path-only rows are skipped via chat_rel exclusion.
    assert "greeter.py:" not in rendered


def test_repo_map_respects_token_budget(tmp_path: Path):
    _seed_project(tmp_path)
    rm = RepoMap(tmp_path, max_map_tokens=120, max_map_tokens_no_chat=120)
    rendered = rm.get_map()
    # char_count_tokens counter: len // 4. So 120 tokens ≈ 480 chars.
    # We accept a generous tolerance because binary search may overshoot
    # slightly under the 15 % default tolerance band.
    assert len(rendered) // 4 <= 200, f"oversized map: {len(rendered)} chars"


def test_repo_map_with_mtime_cache(tmp_path: Path):
    _seed_project(tmp_path)
    cache = MtimeCache(tmp_path / ".cache")
    rm = RepoMap(
        tmp_path,
        max_map_tokens=1024,
        max_map_tokens_no_chat=2048,
        mtime_cache=cache,
    )
    first = rm.get_map()
    second = rm.get_map()
    assert first == second, "deterministic on stable inputs"
    # Cache should have some tag entries.
    assert len(cache) > 0


def test_repo_map_promotes_important_files(tmp_path: Path):
    _seed_project(tmp_path)
    # README has no inbound refs but the allowlist should hoist it.
    rm = RepoMap(tmp_path, max_map_tokens=2048, max_map_tokens_no_chat=4096)
    rendered = rm.get_map()
    # Order: important files are prepended ahead of PageRank output.
    # So README.md should appear earlier than greeter.py in rendered.
    readme_pos = rendered.find("README.md")
    greeter_pos = rendered.find("greeter.py")
    assert readme_pos >= 0
    if greeter_pos >= 0:
        assert readme_pos < greeter_pos


def test_repo_map_force_refresh_runs_clean(tmp_path: Path):
    _seed_project(tmp_path)
    rm = RepoMap(tmp_path, max_map_tokens=1024, max_map_tokens_no_chat=2048)
    rendered = rm.get_map(force_refresh=True)
    assert rendered


# ---------------------------------------------------------------------------
# Supervisor integration
# ---------------------------------------------------------------------------


def test_supervisor_attaches_repo_map_via_provider(tmp_path: Path):
    """End-to-end: a SupervisorDecision gains repo_map_text from the
    configured provider."""
    from ultron.coding.project_supervisor import (
        ProjectSupervisor,
        SupervisorAction,
        SupervisorDecision,
    )
    from ultron.coding.projects import ProjectRegistry

    registry_path = tmp_path / "projects.json"
    registry = ProjectRegistry(registry_path)
    calls = {"count": 0, "last_path": None, "last_text": None}

    def provider(project_path: str, user_text: str):
        calls["count"] += 1
        calls["last_path"] = project_path
        calls["last_text"] = user_text
        return f"--- REPO MAP for {project_path} ---"

    supervisor = ProjectSupervisor(
        index=None,
        registry=registry,
        repo_map_provider=provider,
    )
    decision = SupervisorDecision(
        action=SupervisorAction.EDIT,
        target_project_path="/fake/project",
        user_text="add error handling",
    )
    supervisor._attach_repo_map(decision)
    assert calls["count"] == 1
    assert calls["last_path"] == "/fake/project"
    assert decision.repo_map_text == "--- REPO MAP for /fake/project ---"


def test_supervisor_skips_repo_map_for_clarify(tmp_path: Path):
    from ultron.coding.project_supervisor import (
        ProjectSupervisor,
        SupervisorAction,
        SupervisorDecision,
    )
    from ultron.coding.projects import ProjectRegistry

    registry = ProjectRegistry(tmp_path / "projects.json")
    calls = {"count": 0}

    def provider(project_path: str, user_text: str):
        calls["count"] += 1
        return "should not be invoked"

    supervisor = ProjectSupervisor(
        index=None,
        registry=registry,
        repo_map_provider=provider,
    )
    decision = SupervisorDecision(
        action=SupervisorAction.CLARIFY,
        target_project_path="/fake/project",
        user_text="which one?",
    )
    supervisor._attach_repo_map(decision)
    assert calls["count"] == 0
    assert decision.repo_map_text is None


def test_supervisor_skips_repo_map_when_no_path(tmp_path: Path):
    from ultron.coding.project_supervisor import (
        ProjectSupervisor,
        SupervisorAction,
        SupervisorDecision,
    )
    from ultron.coding.projects import ProjectRegistry

    registry = ProjectRegistry(tmp_path / "projects.json")
    calls = {"count": 0}

    def provider(project_path: str, user_text: str):
        calls["count"] += 1
        return "x"

    supervisor = ProjectSupervisor(
        index=None,
        registry=registry,
        repo_map_provider=provider,
    )
    decision = SupervisorDecision(
        action=SupervisorAction.NEW,
        target_project_path=None,
        user_text="make a thing",
    )
    supervisor._attach_repo_map(decision)
    assert calls["count"] == 0
    assert decision.repo_map_text is None


def test_supervisor_swallows_provider_errors(tmp_path: Path):
    """Provider exceptions must not propagate."""
    from ultron.coding.project_supervisor import (
        ProjectSupervisor,
        SupervisorAction,
        SupervisorDecision,
    )
    from ultron.coding.projects import ProjectRegistry

    registry = ProjectRegistry(tmp_path / "projects.json")

    def provider(project_path: str, user_text: str):
        raise RuntimeError("boom")

    supervisor = ProjectSupervisor(
        index=None,
        registry=registry,
        repo_map_provider=provider,
    )
    decision = SupervisorDecision(
        action=SupervisorAction.EDIT,
        target_project_path="/fake/project",
        user_text="do thing",
    )
    # Should not raise.
    supervisor._attach_repo_map(decision)
    assert decision.repo_map_text is None


def test_supervisor_decision_log_dict_excludes_repo_map_text(tmp_path: Path):
    from ultron.coding.project_supervisor import (
        SupervisorAction,
        SupervisorDecision,
    )

    decision = SupervisorDecision(
        action=SupervisorAction.EDIT,
        target_project_path="/fake",
        repo_map_text="long blob " * 1000,
    )
    log = decision.to_log_dict()
    assert "repo_map_text" not in log
    assert log["repo_map_attached"] is True


# ---------------------------------------------------------------------------
# RepoMapProviderCache
# ---------------------------------------------------------------------------


def test_provider_cache_reuses_repo_map_instance(tmp_path: Path):
    from ultron.coding.repo_map import RepoMapProviderCache

    _seed_project(tmp_path)
    cache = RepoMapProviderCache(
        max_map_tokens=1024,
        max_map_tokens_no_chat=2048,
    )
    rm1 = cache.get_or_create(str(tmp_path))
    rm2 = cache.get_or_create(str(tmp_path))
    assert rm1 is rm2, "cache should return the same RepoMap instance"


def test_provider_cache_returns_none_for_missing_path(tmp_path: Path):
    from ultron.coding.repo_map import RepoMapProviderCache

    cache = RepoMapProviderCache()
    assert cache.get_or_create(str(tmp_path / "does-not-exist")) is None


def test_provider_cache_call_returns_rendered_map(tmp_path: Path):
    from ultron.coding.repo_map import RepoMapProviderCache

    _seed_project(tmp_path)
    cache = RepoMapProviderCache(
        max_map_tokens=1024,
        max_map_tokens_no_chat=2048,
    )
    rendered = cache(str(tmp_path), "look at the Greeter class")
    assert rendered
    assert "Greeter" in rendered


def test_provider_cache_call_returns_none_on_invalid_path(tmp_path: Path):
    from ultron.coding.repo_map import RepoMapProviderCache

    cache = RepoMapProviderCache()
    assert cache(str(tmp_path / "missing"), "anything") is None


def test_provider_cache_mines_idents_from_user_text(tmp_path: Path):
    """The cache call must run extract_idents on the utterance and pass
    them through to RepoMap.get_map for personalization."""
    from ultron.coding.repo_map import RepoMapProviderCache

    _seed_project(tmp_path)
    (tmp_path / "obscure_extra.py").write_text(
        "def obscure_helper_function():\n    return 1\n"
    )
    cache = RepoMapProviderCache(
        max_map_tokens=200,
        max_map_tokens_no_chat=200,
    )
    # User mentions the obscure ident — it should get a 10x weight
    # bump and surface in the map even with a tight token budget.
    rendered = cache(
        str(tmp_path),
        "what does obscure_helper_function do?",
    )
    assert rendered
    assert "obscure_helper_function" in rendered
