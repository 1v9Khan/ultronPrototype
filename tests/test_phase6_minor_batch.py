"""V1-gap minor batch: B2 / B3 / B4 / C1 / C2 verification.

Pure-function tests that lock in the small fixes from the audit so a
future refactor can't silently break them.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# B2: Brave query deduplication
# ---------------------------------------------------------------------------


def test_normalise_search_query_canonicalises_token_set():
    from ultron.web_search.search import _normalise_search_query

    a = _normalise_search_query("Tampa weather today")
    b = _normalise_search_query("today's weather in Tampa")
    assert a == b


def test_normalise_search_query_handles_empty():
    from ultron.web_search.search import _normalise_search_query
    assert _normalise_search_query("") == ""
    assert _normalise_search_query("    ") == ""


def test_dedupe_queries_preserves_first_seen_order():
    from ultron.web_search.search import _dedupe_queries

    out = _dedupe_queries([
        "Tampa weather today",
        "today's weather in Tampa",   # duplicate of first
        "NVDA stock price",
        "Today's Weather In Tampa",   # case-insensitive duplicate
        "stock price NVDA",           # token-order duplicate of NVDA query
    ])
    # First-seen order preserved.
    assert out[0] == "Tampa weather today"
    # Both NVDA forms collapse together.
    assert any("NVDA" in q for q in out)
    # Expected length is 2.
    assert len(out) == 2


def test_dedupe_queries_drops_blanks():
    from ultron.web_search.search import _dedupe_queries
    assert _dedupe_queries(["", "   ", "x"]) == ["x"]


# ---------------------------------------------------------------------------
# B3: superscript citation rendering
# ---------------------------------------------------------------------------


def test_render_inline_marker_bracket_default():
    from ultron.web_search.search import _render_inline_marker
    assert _render_inline_marker(1, fmt="bracket") == "[1]"
    assert _render_inline_marker(7, fmt="bracket") == "[7]"


def test_render_inline_marker_superscript():
    from ultron.web_search.search import _render_inline_marker
    assert _render_inline_marker(1, fmt="superscript") == "¹"
    assert _render_inline_marker(2, fmt="superscript") == "²"
    assert _render_inline_marker(10, fmt="superscript") == "¹⁰"
    assert _render_inline_marker(123, fmt="superscript") == "¹²³"


def test_render_inline_marker_unknown_format_falls_back_to_bracket():
    from ultron.web_search.search import _render_inline_marker
    assert _render_inline_marker(5, fmt="not-a-real-format") == "[5]"


def test_format_sources_for_prompt_uses_superscript_by_default():
    """Default-ON: V1-spec Part 4.4 wording (Unicode superscripts)."""
    from ultron.web_search.search import SearchSource, format_sources_for_prompt

    sources = [SearchSource(
        url="https://example.com/a", title="Anthropic", snippet="snip",
        full_text=None, rank=0,
    )]
    block = format_sources_for_prompt(sources)
    assert "¹ Anthropic" in block


def test_format_sources_for_prompt_uses_bracket_when_configured(monkeypatch):
    """Operator opt-out for ASCII-only consumers."""
    from ultron.web_search import search as search_mod

    sources = [search_mod.SearchSource(
        url="https://example.com/a", title="A", snippet="snip",
        full_text=None, rank=0,
    )]
    monkeypatch.setattr(
        search_mod, "_resolve_citation_format",
        lambda: "bracket",
    )
    block = search_mod.format_sources_for_prompt(sources)
    assert "[1] A" in block


# ---------------------------------------------------------------------------
# B4: Brave default count is 5
# ---------------------------------------------------------------------------


def test_brave_default_count_is_5():
    from ultron.config import BraveConfig
    assert BraveConfig().count == 5


# ---------------------------------------------------------------------------
# C1: Project dataclass spec fields
# ---------------------------------------------------------------------------


def test_project_dataclass_has_v1_spec_fields():
    """V1 spec Part 6.2 listed: name, aliases, path, language,
    description, last_accessed."""
    from ultron.coding.projects import Project
    import time

    p = Project(
        name="test", path="/tmp", aliases=["t"],
        language="python", description="a tool",
        last_accessed=time.time(),
    )
    assert p.name == "test"
    assert p.aliases == ["t"]
    assert p.language == "python"
    assert p.description == "a tool"
    assert p.last_accessed > 0


def test_project_from_dict_loads_v1_fields_from_legacy_payload():
    from ultron.coding.projects import Project

    p = Project.from_dict({
        "name": "old-project",
        "path": "/tmp/old",
        "aliases": ["legacy"],
        # No description / last_accessed -> defaults.
    })
    assert p.description == ""
    assert p.last_accessed == 0.0  # defaults to 0 when missing


# ---------------------------------------------------------------------------
# C2: last_session.py alias
# ---------------------------------------------------------------------------


def test_last_session_script_exists_and_imports():
    last_session = _REPO / "scripts" / "last_session.py"
    assert last_session.is_file()
    spec = importlib.util.spec_from_file_location(
        "_test_last_session", last_session,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main")


def test_last_session_main_is_callable(tmp_path, monkeypatch):
    """Calling last_session.main with --help shouldn't crash."""
    last_session = _REPO / "scripts" / "last_session.py"
    spec = importlib.util.spec_from_file_location(
        "_test_last_session_callable", last_session,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # --list with a non-existent sessions dir should return exit 0
    # (printed empty list) -- verifies the alias forwards correctly.
    rc = mod.main(["--list", "--sessions-dir", str(tmp_path)])
    assert rc == 0
