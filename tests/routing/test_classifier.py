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


@pytest.fixture(autouse=True)
def _enable_openclaw_features():
    """V1-gap A1 / C3: gaming / desktop / window classifier patterns
    are gated on the live OpenClaw flags. Tests in this file assume
    those features are wired (so the patterns fire). We flip the
    flags ON for the duration of each test and restore the original
    state afterward.

    The fixture is autouse so it applies to every test in the file
    without requiring each parametrise block to opt in.
    """
    from ultron.config import get_config

    cfg = get_config()
    saved = (
        cfg.openclaw.enabled,
        cfg.gaming_mode.enabled,
        cfg.desktop.enabled,
        cfg.window_control.enabled,
    )
    cfg.openclaw.enabled = True
    cfg.gaming_mode.enabled = True
    cfg.desktop.enabled = True
    cfg.window_control.enabled = True
    try:
        yield
    finally:
        cfg.openclaw.enabled = saved[0]
        cfg.gaming_mode.enabled = saved[1]
        cfg.desktop.enabled = saved[2]
        cfg.window_control.enabled = saved[3]


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
    # Comprehensive harness regression — "scroll the <noun> <direction>" was
    # missed by the original `scroll\s+(?:down|up|to)\s+the` pattern.
    "scroll the page down",
    "scroll the window up",
    "scroll the tab to the bottom",
    # 2026-05-10 live-session regression: "Can you open a browser
    # window with Google's homepage for me?" fell through to the
    # CONVERSATIONAL LLM which apologised that it couldn't open
    # browsers. The determiner-less navigate pattern required either
    # nothing or "the" before the noun; "a browser window" missed.
    "open a browser window with Google's homepage",
    "Can you open a browser window with Google's homepage for me?",
    "open a browser to YouTube",
    "open a browser tab with Reddit",
    "open the browser to YouTube",
    "open a new browser window",
    "open a new tab to GitHub",
    "open my browser tab with hacker news",
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
    # Comprehensive harness regression — "render <det> <media-noun>" without
    # the "me" reflexive was missed by the original `render\s+me\s+...`
    # pattern.
    "render an image of a dragon in flight",
    "render the picture of a sunset",
    "render a video of waves",
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
    # Comprehensive harness regression — "notify me on <channel>" missed.
    "notify me on telegram if anything alerts",
    "notify me via signal when the build is done",
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
    # Comprehensive harness regression — "show me the contents of <file.ext>"
    # missed when the literal word "file" was absent.
    "show me the contents of config.yaml",
    "show me the contents of README.md",
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


# ---------------------------------------------------------------------------
# V1-gap A1 — gaming mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("utterance", [
    "gaming mode",
    "gaming mode on",
    "engage gaming mode",
    "I'm about to play Valorant",
    "I'm gonna play Valorant",
    "I'm about to play CS2",
    "I'm fixing to play Apex",
    "shutting down desktop control",
    "kill desktop control",
])
def test_classify_routing_gaming_mode_engage(utterance):
    intent = classify_routing(utterance)
    assert intent.kind == RoutingIntentKind.GAMING_MODE
    assert intent.gaming_mode_intent is not None
    assert intent.gaming_mode_intent.action == "engage"


@pytest.mark.parametrize("utterance", [
    "gaming mode off",
    "disengage gaming mode",
    "exit gaming mode",
    "done playing",
    "I'm done playing",
    "restore desktop control",
    "full control restored",
])
def test_classify_routing_gaming_mode_disengage(utterance):
    intent = classify_routing(utterance)
    assert intent.kind == RoutingIntentKind.GAMING_MODE
    assert intent.gaming_mode_intent is not None
    assert intent.gaming_mode_intent.action == "disengage"


@pytest.mark.parametrize("utterance", [
    "are we in gaming mode",
    "is gaming mode on",
    "is gaming mode active",
    "gaming mode status",
])
def test_classify_routing_gaming_mode_status(utterance):
    intent = classify_routing(utterance)
    assert intent.kind == RoutingIntentKind.GAMING_MODE
    assert intent.gaming_mode_intent.action == "status"


def test_classify_routing_gaming_mode_priority_over_hybrid():
    """An utterance with both 'gaming mode' AND a HYBRID phrase routes
    GAMING_MODE -- gaming mode is checked first."""
    intent = classify_routing(
        "I'm about to play Valorant, set up my dev environment afterward",
    )
    assert intent.kind == RoutingIntentKind.GAMING_MODE


def test_gaming_mode_suppressed_during_pending_clarification():
    intent = classify_routing(
        "gaming mode",
        has_active_coding_task=True,
        has_pending_clarification=True,
    )
    assert intent.kind != RoutingIntentKind.GAMING_MODE


# ---------------------------------------------------------------------------
# V1-gap C3 — desktop / windows control
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("utterance,expected_action", [
    ("take a screenshot of the desktop", "screenshot"),
    ("take a screenshot of the screen", "screenshot"),
    ("take a screenshot of the active window", "screenshot"),
    ("screenshot the screen", "screenshot"),
    ("screenshot my desktop", "screenshot"),
    ("snap a screenshot", "screenshot"),
    ("capture the desktop", "screenshot"),
    ("list my open windows", "list_windows"),
    ("what windows are open", "list_windows"),
    ("show me all open windows", "list_windows"),
    ("find the chrome window", "find_window"),
    ("locate the visual studio window", "find_window"),
    ("where is the slack window", "find_window"),
])
def test_classify_routing_desktop_automation(utterance, expected_action):
    intent = classify_routing(utterance)
    assert intent.kind == RoutingIntentKind.DESKTOP_AUTOMATION
    assert intent.desktop_intent is not None
    assert intent.desktop_intent.action == expected_action


def test_screenshot_with_url_routes_browser_not_desktop():
    """Browser-screenshot context (URL present) wins over desktop."""
    intent = classify_routing(
        "Take a screenshot of github.com after navigating to the repo",
    )
    assert intent.kind != RoutingIntentKind.DESKTOP_AUTOMATION


@pytest.mark.parametrize("utterance,expected_action", [
    ("focus the chrome window", "focus"),
    ("activate the cursor window", "focus"),
    ("bring the slack window to front", "focus"),
])
def test_classify_routing_window_focus(utterance, expected_action):
    intent = classify_routing(utterance)
    assert intent.kind == RoutingIntentKind.WINDOW_AUTOMATION
    assert intent.window_intent.action == expected_action


def test_classify_routing_window_type():
    intent = classify_routing(
        "type 'hello world' into the search box",
    )
    assert intent.kind == RoutingIntentKind.WINDOW_AUTOMATION
    assert intent.window_intent.action == "type"
    assert intent.window_intent.value == "hello world"


def test_classify_routing_window_click():
    intent = classify_routing(
        "click the submit button in the form window",
    )
    assert intent.kind == RoutingIntentKind.WINDOW_AUTOMATION
    assert intent.window_intent.action == "click"


def test_desktop_window_suppressed_during_pending_clarification():
    intent = classify_routing(
        "take a screenshot of the desktop",
        has_pending_clarification=True,
    )
    assert intent.kind != RoutingIntentKind.DESKTOP_AUTOMATION


# ---------------------------------------------------------------------------
# V1-gap A1 / C3 — feature flag gating
#
# The autouse fixture above turns the flags ON for every test in this
# file so the patterns fire. The tests below override that by setting
# specific flags OFF and verifying the new intent kinds fall through
# to CONVERSATIONAL -- confirming we don't surface "isn't reachable"
# stub messages on installs that haven't wired OpenClaw yet.
# ---------------------------------------------------------------------------


def _classify_with_flags(
    utterance: str,
    *,
    openclaw: bool,
    gaming_mode: bool = True,
    desktop: bool = True,
    window_control: bool = True,
):
    """Run classify_routing under explicit flag overrides."""
    from ultron.config import get_config

    cfg = get_config()
    saved = (
        cfg.openclaw.enabled, cfg.gaming_mode.enabled,
        cfg.desktop.enabled, cfg.window_control.enabled,
    )
    try:
        cfg.openclaw.enabled = openclaw
        cfg.gaming_mode.enabled = gaming_mode
        cfg.desktop.enabled = desktop
        cfg.window_control.enabled = window_control
        return classify_routing(utterance)
    finally:
        cfg.openclaw.enabled = saved[0]
        cfg.gaming_mode.enabled = saved[1]
        cfg.desktop.enabled = saved[2]
        cfg.window_control.enabled = saved[3]


def test_gaming_mode_pattern_skipped_when_openclaw_offline():
    intent = _classify_with_flags(
        "I'm about to play Valorant", openclaw=False,
    )
    assert intent.kind == RoutingIntentKind.CONVERSATIONAL


def test_gaming_mode_pattern_skipped_when_per_feature_off():
    """openclaw.enabled=true but gaming_mode.enabled=false -> falls through."""
    intent = _classify_with_flags(
        "gaming mode", openclaw=True, gaming_mode=False,
    )
    assert intent.kind == RoutingIntentKind.CONVERSATIONAL


def test_desktop_pattern_skipped_when_openclaw_offline():
    """With openclaw.enabled=false, the DESKTOP_AUTOMATION classifier
    branch is skipped. Utterances like "take a screenshot of the
    desktop" share the ``take a screenshot`` token with the legacy
    BROWSER_INTERACT pattern, so they fall back to BROWSER_AUTOMATION
    (the pre-Phase-3 routing). What we care about: the utterance is
    NOT routed to DESKTOP_AUTOMATION."""
    intent = _classify_with_flags(
        "take a screenshot of the desktop", openclaw=False,
    )
    assert intent.kind != RoutingIntentKind.DESKTOP_AUTOMATION


def test_desktop_pattern_skipped_when_per_feature_off():
    intent = _classify_with_flags(
        "take a screenshot of the desktop",
        openclaw=True, desktop=False,
    )
    assert intent.kind != RoutingIntentKind.DESKTOP_AUTOMATION


def test_list_windows_falls_through_when_desktop_off():
    """Pure desktop-only phrasing (no browser-overlap) falls all the
    way through to CONVERSATIONAL when the feature is gated off."""
    intent = _classify_with_flags(
        "list my open windows", openclaw=False,
    )
    assert intent.kind == RoutingIntentKind.CONVERSATIONAL


def test_window_pattern_skipped_when_openclaw_offline():
    intent = _classify_with_flags(
        "focus the chrome window", openclaw=False,
    )
    assert intent.kind == RoutingIntentKind.CONVERSATIONAL


def test_window_pattern_skipped_when_per_feature_off():
    intent = _classify_with_flags(
        "focus the chrome window",
        openclaw=True, window_control=False,
    )
    assert intent.kind == RoutingIntentKind.CONVERSATIONAL


def test_other_routing_kinds_unaffected_by_openclaw_flag():
    """Sanity: BROWSER / MEDIA / etc. are NOT gated by openclaw.enabled
    because their dispatchers also check it. Ungating them would
    break existing tests + the messaging dispatcher's actual live
    path. Only the V1-gap A1 / C3 kinds get the classifier gate."""
    # BROWSER pattern still fires when openclaw is off.
    intent = _classify_with_flags(
        "open hacker news", openclaw=False,
    )
    assert intent.kind == RoutingIntentKind.BROWSER_AUTOMATION
