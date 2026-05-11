"""Rule-based coding-intent classifier tests."""

from __future__ import annotations

import pytest

from ultron.coding.intent import (
    CodingIntentKind,
    classify,
    derive_project_name,
)


# ---------------------------------------------------------------------------
# CODE_TASK detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("utterance", [
    "Create a Python script that prints hello world.",
    "Build me a Flask app for tracking todos.",
    "Write a TypeScript CLI that processes csv files.",
    "Make a small Python tool for renaming files in bulk.",
    "Generate a node service that exposes a /healthz route.",
    "Spin up a quick Python project to scrape my email.",
])
def test_creation_phrases_are_code_tasks(utterance):
    intent = classify(utterance)
    assert intent.kind == CodingIntentKind.CODE_TASK
    assert intent.is_new_project is True


@pytest.mark.parametrize("utterance", [
    "Add a function to my flask app to handle login.",
    "Fix the bug in my calculator project.",
    "Refactor the dashboard code so it's faster.",
    "Patch the script that handles uploads.",
    "Implement an endpoint to my api.",
])
def test_editing_phrases_are_code_tasks(utterance):
    intent = classify(utterance)
    assert intent.kind == CodingIntentKind.CODE_TASK


def test_existing_project_reference_is_captured():
    intent = classify("Fix the bug in my flask app.")
    assert intent.kind == CodingIntentKind.CODE_TASK
    assert intent.is_new_project is False
    assert intent.project_reference is not None
    assert "flask" in intent.project_reference.lower()


def test_explicit_name_is_captured():
    intent = classify("Create a python script called weather_fetcher that pulls forecasts.")
    assert intent.kind == CodingIntentKind.CODE_TASK
    assert intent.explicit_name == "weather_fetcher"


def test_existing_project_with_explicit_name_keeps_both():
    intent = classify("Edit my flask app called inventory_api to add a /search endpoint.")
    assert intent.kind == CodingIntentKind.CODE_TASK
    # 'flask' in candidates_for_resolver is OK; explicit_name also recorded.
    assert intent.explicit_name == "inventory_api"
    assert any("flask" in c.lower() for c in intent.candidates_for_resolver)


# ---------------------------------------------------------------------------
# Negative cases -- these MUST NOT route to the coding pipeline.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("utterance", [
    "What's the weather today?",
    "Tell me about black holes.",
    "Are you afraid of death?",
    "Explain how hash tables work.",
    "What time is it?",
    "Play some music.",
    "Cancel my reminder.",  # 'cancel' alone shouldn't fire CANCEL when no task
    "How tall is Mount Everest?",
    "Write a poem about loneliness.",  # creative, not coding
    "",
])
def test_non_coding_utterances_classify_as_none(utterance):
    intent = classify(utterance)
    assert intent.kind == CodingIntentKind.NONE, (
        f"unexpected coding intent on {utterance!r}: {intent.kind} ({intent.reason})"
    )


# ---------------------------------------------------------------------------
# PROGRESS_QUERY / CANCEL only fire with an active task.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("utterance", [
    "How's it going?",
    "Are you done?",
    "What's claude doing?",
    "Any progress?",
    "Status?",
    "Still running?",
    "Finished yet?",
    "What's the update?",
])
def test_progress_queries_with_active_task(utterance):
    intent = classify(utterance, has_active_task=True)
    assert intent.kind == CodingIntentKind.PROGRESS_QUERY


# 2026-05-11 follow-up fix: broadened progress patterns to accept
# ``<determiner> <coding-noun>`` after ``how is / what's / is``. The
# live-session bug was "How is that project going?" -- the original
# regex required ``going`` immediately after ``that`` and didn't
# tolerate the ``project`` in between. The fix is documented in
# ``intent.py`` next to ``_DETERMINER_NOUN``.
@pytest.mark.parametrize("utterance", [
    # The actual live-session phrasing that fell through.
    "How is that project going?",
    # Determiner variants paired with "project".
    "How's the project going?",
    "How is my project going?",
    "How's your project going?",
    "How is our project going?",
    "How is this project going?",
    # Other coding nouns.
    "How is the build going?",
    "How's the app going?",
    "How is my code going?",
    "How's the run going?",
    # Coming / coming along.
    "How is the project coming along?",
    "How's it coming?",
    # "what's the project doing" variants.
    "What's the project doing?",
    "What's my build doing?",
    "What is your app working on?",
    # "is the project done" variants.
    "Is the project done?",
    "Is my build done yet?",
    "Is the app done?",
])
def test_progress_queries_broadened_phrasings(utterance):
    """Regression coverage for the 2026-05-11 follow-up fix: the
    ``<determiner> <coding-noun>`` subject group must match the same
    way the legacy ``it / things / claude / the task / that`` group
    did."""
    intent = classify(utterance, has_active_task=True)
    assert intent.kind == CodingIntentKind.PROGRESS_QUERY, (
        f"expected PROGRESS_QUERY for {utterance!r} but got "
        f"{intent.kind.value} ({intent.reason})"
    )


@pytest.mark.parametrize("utterance", [
    "How's it going?",
    "Any progress?",
    "Status?",
    # 2026-05-11 follow-up fix: the broadened patterns must still
    # respect the has_active_task gate -- they cannot hijack the
    # utterance when no coding task is running. Otherwise asking
    # "How's the project going?" in passing conversation would be
    # misrouted.
    "How is that project going?",
    "How's the build going?",
    "Is the project done?",
])
def test_progress_queries_without_active_task_fall_through(utterance):
    """No coding task running -> progress patterns must NOT hijack the
    utterance. They get handled by the regular LLM path."""
    intent = classify(utterance, has_active_task=False)
    assert intent.kind == CodingIntentKind.NONE


@pytest.mark.parametrize("utterance", [
    "Stop the task.",
    "Cancel the build.",
    "Abort the run.",
    "Kill claude.",
])
def test_cancel_phrases_with_active_task(utterance):
    intent = classify(utterance, has_active_task=True)
    assert intent.kind == CodingIntentKind.CANCEL


def test_cancel_without_active_task_falls_through():
    intent = classify("Stop the task.", has_active_task=False)
    assert intent.kind == CodingIntentKind.NONE


# ---------------------------------------------------------------------------
# derive_project_name
# ---------------------------------------------------------------------------


def test_derive_uses_explicit_name_when_present():
    intent = classify("Make a python script called inventory_tool that reads csv.")
    assert derive_project_name(intent) == "inventory_tool"


def test_derive_falls_back_to_phrase_slug():
    intent = classify("Create a tool to manage podcast subscriptions.")
    name = derive_project_name(intent)
    # We don't pin the exact slug -- heuristic -- but it should contain
    # some content from the phrase.
    assert any(w in name for w in ("manage", "podcast", "subscriptions"))


def test_derive_falls_back_when_phrase_missing():
    # Forge an intent without a creation verb captured in task_text:
    intent = classify("Make a thing.")
    name = derive_project_name(intent)
    assert name  # at least produces something
