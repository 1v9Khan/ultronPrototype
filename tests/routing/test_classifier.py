"""Routing classifier tests.

Coverage per category — the spec asks for 20 utterances on
BROWSER_AUTOMATION and 10+ on each other; we run the classifier on
parameterized lists and assert the predicted kind.

Conversational + coding-passthrough cases are in
``test_classifier_passthrough.py`` to keep this file focused.
"""

from __future__ import annotations

import pytest

from ultron.openclaw_routing import classify_routing
from ultron.openclaw_routing.intents import RoutingIntentKind


# ---------------------------------------------------------------------------
# BROWSER_AUTOMATION — 20 utterances
# ---------------------------------------------------------------------------


_BROWSER = [
    "open hacker news",
    "open https://example.com/article",
    "open the page at example.com",
    "navigate to https://github.com/anthropics",
    "go to the page about quantum computing",
    "pull up the wikipedia article on Tesla",
    "pull up hacker news",
    "open up wikipedia",
    "open up youtube",
    "open up github",
    "open reddit",
    "click on the submit button",
    "click the login link",
    "fill in the form with my details",
    "fill out the contact form",
    "take a screenshot of the page",
    "log into my github account",
    "sign into gmail",
    "submit the form",
    "scroll down the page",
]


@pytest.mark.parametrize("utt", _BROWSER)
def test_browser_automation_classified(utt):
    intent = classify_routing(utt)
    assert intent.kind == RoutingIntentKind.BROWSER_AUTOMATION, (
        f"got {intent.kind.value} for {utt!r}; reason={intent.reason}"
    )


# ---------------------------------------------------------------------------
# MEDIA_GENERATION — 10 utterances
# ---------------------------------------------------------------------------


_MEDIA = [
    "make me an image of a cat in a top hat",
    "make me an image of a sunset over mountains",
    "generate a picture of a Victorian library",
    "generate an artwork inspired by Monet",
    "create a song about late-night coding",
    "compose a tune for the start of the show",
    "compose music that sounds like rain",
    "generate a short video of a flag waving",
    "draw me a logo for the project",
    "render me an image of a robot in a forest",
]


@pytest.mark.parametrize("utt", _MEDIA)
def test_media_generation_classified(utt):
    intent = classify_routing(utt)
    assert intent.kind == RoutingIntentKind.MEDIA_GENERATION, (
        f"got {intent.kind.value} for {utt!r}; reason={intent.reason}"
    )


# ---------------------------------------------------------------------------
# MESSAGING — 10 utterances
# ---------------------------------------------------------------------------


_MESSAGING = [
    "send a message to my phone when the build finishes",
    "send a notification to my phone",
    "text me when it's done",
    "notify me when the deploy completes",
    "tell me on telegram when the test suite passes",
    "send to telegram: build green",
    "ping me on telegram when the server starts",
    "shoot me a message when you're done",
    "alert me when the script crashes",
    "send me a push notification when memory is low",
]


@pytest.mark.parametrize("utt", _MESSAGING)
def test_messaging_classified(utt):
    intent = classify_routing(utt)
    assert intent.kind == RoutingIntentKind.MESSAGING, (
        f"got {intent.kind.value} for {utt!r}; reason={intent.reason}"
    )


# ---------------------------------------------------------------------------
# FILE_OPERATION — 10 utterances
# ---------------------------------------------------------------------------


_FILE = [
    "read the file at C:/users/me/notes.txt",
    "read the file at /etc/hosts",
    "show me the contents of the file at C:/test.log",
    "open the file at C:/data/report.csv",
    "write to the file at C:/scratch/output.txt",
    "save to a file at C:/scratch/result.json",
    "delete the file at C:/tmp/old.bak",
    "remove the file at /tmp/old.log",
    "list the files in C:/users/me/Downloads",
    "what's in the directory C:/Projects",
]


@pytest.mark.parametrize("utt", _FILE)
def test_file_operation_classified(utt):
    intent = classify_routing(utt)
    assert intent.kind == RoutingIntentKind.FILE_OPERATION, (
        f"got {intent.kind.value} for {utt!r}; reason={intent.reason}"
    )


# ---------------------------------------------------------------------------
# SHELL_OPERATION — 10 utterances
# ---------------------------------------------------------------------------


_SHELL = [
    "run dir on the desktop",
    "run ls -la in my home directory",
    "run pwd",
    "run git status",
    "run npm install",
    "run pip list",
    "run python --version",
    "execute the command echo hello",
    "what's the output of git status",
    "in the terminal run uptime",
]


@pytest.mark.parametrize("utt", _SHELL)
def test_shell_operation_classified(utt):
    intent = classify_routing(utt)
    assert intent.kind == RoutingIntentKind.SHELL_OPERATION, (
        f"got {intent.kind.value} for {utt!r}; reason={intent.reason}"
    )


# ---------------------------------------------------------------------------
# HYBRID_TASK — 10 utterances; classifier produces HYBRID_TASK kind
# (decomposition handled separately by HybridTaskDecomposer).
# ---------------------------------------------------------------------------


_HYBRID = [
    "set up a development environment for this project",
    "set up a venv for the project",
    "install dependencies for the project",
    "deploy this to the staging server",
    "ship this to production",
    "automate my excel workflow",
    "write a script that opens chrome and clicks the login button",
    "build a tool for my browser that scrapes hacker news",
    "make a script that controls obs studio",
    "automate the process of pulling logs and analyzing them",
]


@pytest.mark.parametrize("utt", _HYBRID)
def test_hybrid_task_classified(utt):
    intent = classify_routing(utt)
    assert intent.kind == RoutingIntentKind.HYBRID_TASK, (
        f"got {intent.kind.value} for {utt!r}; reason={intent.reason}"
    )


# ---------------------------------------------------------------------------
# CONVERSATIONAL — 10 utterances; the default fallback
# ---------------------------------------------------------------------------


_CONVERSATIONAL = [
    "good morning",
    "what is the boiling point of water",
    "tell me a joke",
    "how are you",
    "who was Nikola Tesla",
    "what's the meaning of life",
    "I'm tired",
    "explain how photosynthesis works",
    "actually never mind",
    "thanks",
]


@pytest.mark.parametrize("utt", _CONVERSATIONAL)
def test_conversational_default(utt):
    intent = classify_routing(utt)
    assert intent.kind == RoutingIntentKind.CONVERSATIONAL, (
        f"got {intent.kind.value} for {utt!r}; reason={intent.reason}"
    )


# ---------------------------------------------------------------------------
# CODE_TASK — preserved from existing coding classifier
# ---------------------------------------------------------------------------


_CODE_TASKS = [
    "build me a python flask app",
    "create a small typescript cli tool",
    "make me a script that prints hello world",
    "scaffold a fastapi project called weather",
    "write me a bash script for backups",
    "fix the bug in my flask app",
    "update my flask app to add authentication",
    "refactor the dashboard module",
]


@pytest.mark.parametrize("utt", _CODE_TASKS)
def test_code_task_classified(utt):
    intent = classify_routing(utt)
    assert intent.kind == RoutingIntentKind.CODE_TASK, (
        f"got {intent.kind.value} for {utt!r}; reason={intent.reason}"
    )


# ---------------------------------------------------------------------------
# SYSTEM_STATUS — Phase 13 voice queries about Ultron's own state
# ---------------------------------------------------------------------------


_SYSTEM_STATUS_ALERT_QUERIES = [
    "what alerts did you flag",
    "any pending alerts",
    "any heartbeat alerts",
    "show me the alerts",
    "list my alerts",
    "alerts pending",
    "what alerts did you raise",
    "any new alerts",
]


_SYSTEM_STATUS_PROJECT_QUERIES = [
    "what is Ultron working on",
    "what are you working on",
    "what's running",
    "what is currently running",
    "list active projects",
    "any active coding sessions",
    "what standing orders are active",
    "any active tasks",
]


_SYSTEM_STATUS_BOTH_QUERIES = [
    "status report",
    "system status",
    "give me a status update",
    "what's going on",
]


@pytest.mark.parametrize("utt", _SYSTEM_STATUS_ALERT_QUERIES)
def test_system_status_alert_focus(utt):
    intent = classify_routing(utt)
    assert intent.kind == RoutingIntentKind.SYSTEM_STATUS, (
        f"got {intent.kind.value} for {utt!r}"
    )
    assert intent.system_status_intent is not None
    assert intent.system_status_intent.focus == "alerts"


@pytest.mark.parametrize("utt", _SYSTEM_STATUS_PROJECT_QUERIES)
def test_system_status_project_focus(utt):
    intent = classify_routing(utt)
    assert intent.kind == RoutingIntentKind.SYSTEM_STATUS, (
        f"got {intent.kind.value} for {utt!r}"
    )
    assert intent.system_status_intent is not None
    assert intent.system_status_intent.focus == "projects"


@pytest.mark.parametrize("utt", _SYSTEM_STATUS_BOTH_QUERIES)
def test_system_status_combined_focus(utt):
    intent = classify_routing(utt)
    assert intent.kind == RoutingIntentKind.SYSTEM_STATUS, (
        f"got {intent.kind.value} for {utt!r}"
    )
    assert intent.system_status_intent is not None
    assert intent.system_status_intent.focus == "all"


def test_system_status_does_not_hijack_unrelated():
    """Conversational utterances mentioning "status" but not the
    canonical patterns should NOT classify as SYSTEM_STATUS."""
    samples = [
        "what's the status of mongodb deployment best practices",
        "I am working on machine learning",
        "alerts can be useful for monitoring",
    ]
    for utt in samples:
        intent = classify_routing(utt)
        assert intent.kind != RoutingIntentKind.SYSTEM_STATUS, (
            f"unexpected SYSTEM_STATUS for {utt!r}; reason={intent.reason}"
        )


def test_system_status_skipped_during_pending_clarification():
    """Mid-coding-clarification, status queries should still pass
    through as clarification responses (not hijacked by status)."""
    intent = classify_routing(
        "what alerts did you flag",
        has_active_coding_task=True,
        has_pending_clarification=True,
    )
    # Pending clarification has higher precedence than system status.
    assert intent.kind != RoutingIntentKind.SYSTEM_STATUS


# ---------------------------------------------------------------------------
# Empty / whitespace
# ---------------------------------------------------------------------------


def test_empty_utterance_is_conversational():
    intent = classify_routing("")
    assert intent.kind == RoutingIntentKind.CONVERSATIONAL


def test_whitespace_utterance_is_conversational():
    intent = classify_routing("   \n  ")
    assert intent.kind == RoutingIntentKind.CONVERSATIONAL
