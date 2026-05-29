"""Tests for the strict deep-code-exploration voice-intent matcher.

``match_code_exploration`` gates the orchestrator's
``_maybe_handle_code_exploration`` short-circuit (a bounded ripgrep loop over
the project source). It MUST be strict: coding TASKS ("build X", "fix the bug")
belong to the coding engineer; web/memory requests belong to their own matchers;
plain "find/where" questions without a code referent must fall through to the
normal LLM path.
"""

from __future__ import annotations

import pytest

from ultron.search.code_exploration import (
    CodeExplorationMatch,
    match_code_exploration,
)


@pytest.mark.parametrize("text", [
    "search the codebase for the rate limiter",
    "where is the safety validator defined",
    "find the function that parses dates",
    "locate the auth module in the source",
    "which files import the reranker",
    "explore the code for the retry logic",
    "grep the source for TODO markers",
    "show me where the wake word detection is implemented",
])
def test_matches_code_search_commands(text):
    m = match_code_exploration(text)
    assert isinstance(m, CodeExplorationMatch)
    assert m.raw_text == text
    assert m.topic  # non-empty best-effort topic


@pytest.mark.parametrize("text", [
    # Coding TASKS -> the coding engineer, not exploration.
    "build a calculator app",
    "fix the bug in the code",
    "create a function that sorts a list",
    "refactor the source code",
    "make me a script",
    "run the project",
    # Web research -> match_deep_research.
    "search the web for python tips",
    "look up the latest news online",
    "google the rate limiter pattern",
    # Memory recall -> match_deep_recall.
    "remember what we discussed about the code",
    "recall the function we talked about",
    # No code referent -> falls through to the normal LLM path.
    "find my keys",
    "where is the bathroom",
    "what time is it",
    "hello there",
])
def test_rejects_non_code_search(text):
    assert match_code_exploration(text) is None


def test_empty_and_whitespace_return_none():
    assert match_code_exploration("") is None
    assert match_code_exploration("   ") is None
    assert match_code_exploration(None) is None  # type: ignore[arg-type]


def test_topic_extraction_drops_pivot():
    m = match_code_exploration("search the codebase for the token budget logic")
    assert m is not None
    # The "for" pivot is consumed; the topic carries the subject.
    assert "token budget" in m.topic.lower()
    assert "codebase" not in m.topic.lower()


def test_task_verb_wins_over_code_referent():
    # Has a code referent ("the code") AND a search word would match, but the
    # coding-task verb "fix" must reject it so it routes to the engineer.
    assert match_code_exploration("fix where the code crashes") is None
