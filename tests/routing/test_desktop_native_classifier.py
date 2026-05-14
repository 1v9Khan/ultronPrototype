"""Tests for the Phase 8 native desktop classifier patterns.

Covers:

- _extract_monitor_target: parse "on my second monitor" / "on monitor 2"
  / directional words.
- _classify_screen_context: SCREEN_CONTEXT_QUERY detection.
- _classify_app_launch: APP_LAUNCH detection (named apps, image search,
  bare URLs, monitor targeting, fullscreen/maximize flags).
- classify_routing end-to-end for the new intent kinds.
"""

from __future__ import annotations

import pytest

from ultron.openclaw_routing.classifier import (
    _classify_app_launch,
    _classify_screen_context,
    _extract_monitor_target,
    classify_routing,
)
from ultron.openclaw_routing.intents import (
    AppLaunchIntent,
    RoutingIntentKind,
    ScreenContextIntent,
)


# ---------------------------------------------------------------------------
# _extract_monitor_target
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected_idx,expected_query_partial", [
    ("open chrome on my second monitor", 1, "second"),
    ("on my 2nd monitor", 1, "2nd"),
    ("on my first monitor", 0, "first"),
    ("on my third monitor", 2, "third"),
    ("on my 3rd monitor", 2, "3rd"),
    ("on my fourth monitor", 3, "fourth"),
    ("open it on monitor 2", 1, "monitor 2"),
    ("open it on screen 3", 2, "screen 3"),
    ("on display 1", 0, "display 1"),
    ("on monitor two", 1, "monitor two"),
    ("on screen four", 3, "screen four"),
])
def test_extract_monitor_target_explicit_index(text, expected_idx, expected_query_partial):
    idx, query = _extract_monitor_target(text)
    assert idx == expected_idx, f"text={text!r} got idx={idx}"
    assert expected_query_partial.lower() in query.lower()


@pytest.mark.parametrize("text,expected_word", [
    ("on my left monitor", "left"),
    ("on my right screen", "right"),
    ("on the center display", "center"),
    ("on my top monitor", "top"),
    ("on my bottom monitor", "bottom"),
    # 2026-05-14: "main" and "primary" are now position-based (resolved by
    # find_monitor at dispatch) so they don't collapse to index 0 at parse
    # time. "main" -> physical center; "primary" -> Win32-primary.
    ("on my main monitor", "main"),
    ("open it on my primary monitor", "primary"),
])
def test_extract_monitor_target_directional(text, expected_word):
    idx, query = _extract_monitor_target(text)
    # Directional words yield None idx + the word as the query.
    assert idx is None
    assert query == expected_word


def test_extract_monitor_target_no_match():
    assert _extract_monitor_target("open chrome") == (None, "")
    assert _extract_monitor_target("") == (None, "")
    assert _extract_monitor_target("how is the weather") == (None, "")


# ---------------------------------------------------------------------------
# _classify_screen_context
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "explain what I'm looking at",
    "Explain what I am looking at",
    "what's on my screen?",
    "what is on my screen",
    "what am I looking at",
    "what do you see",
    "what can you see right now",
    "look at my screen and tell me",
    "describe my screen",
    "describe what's on my screen",
    "tell me about my screen",
    "what does this error mean",
    "what does this dialog mean",
    "help me with what I'm doing",
    "help me with this",
    "explain this code",
    "explain this error",
    "explain this page",
])
def test_classify_screen_context_matches(text):
    intent = _classify_screen_context(text)
    assert intent is not None, f"failed to match: {text!r}"
    assert isinstance(intent, ScreenContextIntent)
    assert intent.include_vlm is True
    assert intent.raw_text == text


@pytest.mark.parametrize("text", [
    "what's the weather like",
    "explain quantum physics",
    "what is the capital of France",
    "tell me a joke",
    "open chrome",
])
def test_classify_screen_context_no_match(text):
    assert _classify_screen_context(text) is None


# ---------------------------------------------------------------------------
# _classify_app_launch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected_app", [
    ("open chrome", "chrome"),
    ("Open Google Chrome", "chrome"),
    ("launch cursor", "cursor"),
    ("open discord", "discord"),
    ("start vscode", "vscode"),
    ("open vs code", "vscode"),
    ("open visual studio code", "vscode"),
    ("launch edge", "edge"),
    ("open firefox", "firefox"),
    ("open spotify", "spotify"),
    ("launch slack", "slack"),
    ("open notepad", "notepad"),
    ("open explorer", "explorer"),
    ("open file explorer", "explorer"),
    ("open the terminal", "terminal"),
    ("launch windows terminal", "terminal"),
    ("open obs", "obs"),
])
def test_classify_app_launch_named_apps(text, expected_app):
    intent = _classify_app_launch(text)
    assert intent is not None, f"failed to match: {text!r}"
    assert isinstance(intent, AppLaunchIntent)
    assert intent.app_name == expected_app


@pytest.mark.parametrize("text,expected_url_fragment", [
    ("open youtube on monitor 2", "youtube.com"),
    ("launch youtube on my second monitor", "youtube.com"),
    ("open gmail on monitor 1", "mail.google.com"),
    ("open github on monitor 3", "github.com"),
    ("open reddit on my left monitor", "reddit.com"),
    ("launch netflix on my second monitor", "netflix.com"),
])
def test_classify_app_launch_known_sites_with_monitor_become_chrome_with_url(
    text, expected_url_fragment,
):
    """Site words fire on APP_LAUNCH only when a monitor target is present.
    Without a monitor target, they defer to BROWSER_AUTOMATION (existing
    behavior preserved). This was the test originally written before the
    site-needs-monitor fix.
    """
    intent = _classify_app_launch(text)
    assert intent is not None
    assert intent.app_name == "chrome"
    assert intent.url is not None
    assert expected_url_fragment in intent.url


@pytest.mark.parametrize("text", [
    "open youtube",
    "launch youtube",
    "open gmail",
    "open github",
    "open reddit",
    "launch netflix",
])
def test_classify_app_launch_known_sites_without_monitor_defer(text):
    """Site names without a monitor target defer to existing
    BROWSER_AUTOMATION routing (returns None from app-launch classifier).
    """
    intent = _classify_app_launch(text)
    assert intent is None


def test_classify_app_launch_monitor_target():
    intent = _classify_app_launch("open chrome on my second monitor")
    assert intent is not None
    assert intent.app_name == "chrome"
    assert intent.monitor_index == 1


def test_classify_app_launch_directional_monitor():
    intent = _classify_app_launch("open cursor on my left monitor")
    assert intent is not None
    assert intent.app_name == "cursor"
    assert intent.monitor_index is None
    assert intent.monitor_query == "left"


def test_classify_app_launch_fullscreen_flag():
    intent = _classify_app_launch("open youtube fullscreen on monitor 2")
    assert intent is not None
    assert intent.fullscreen is True
    assert intent.monitor_index == 1


def test_classify_app_launch_maximize_flag():
    intent = _classify_app_launch("open chrome maximized on monitor 3")
    assert intent is not None
    assert intent.maximize is True
    assert intent.monitor_index == 2


def test_classify_app_launch_chrome_with_youtube_url():
    intent = _classify_app_launch("pull up youtube on my second monitor")
    assert intent is not None
    assert intent.app_name == "chrome"
    assert "youtube.com" in (intent.url or "")
    assert intent.monitor_index == 1


# ---------------------------------------------------------------------------
# 2026-05-14: YouTube channel / video / search deep-linking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,fragment", [
    # "with the channel <name>" -- the user's reported phrasing
    ("Open YouTube on my right monitor with the channel Ordinary Things",
     "Ordinary+Things"),
    # "to channel <name>"
    ("open youtube to channel Veritasium on my main monitor", "Veritasium"),
    # Trailing "the <name> channel" form (channel noun AFTER the name)
    ("open youtube to the Veritasium channel on monitor 2", "Veritasium"),
    # "for the channel <name>"
    ("launch youtube for the channel Vsauce on the right monitor", "Vsauce"),
])
def test_classify_app_launch_youtube_channel(text, fragment):
    """Channel cues build a YouTube results URL biased toward the channel."""
    intent = _classify_app_launch(text)
    assert intent is not None, f"failed for {text!r}"
    assert intent.app_name == "chrome"
    assert intent.url is not None
    assert "youtube.com/results?search_query=" in intent.url
    assert fragment in intent.url
    # The word "channel" is appended to bias the YouTube search result.
    assert "channel" in intent.url.lower()


@pytest.mark.parametrize("text,fragment", [
    ("open youtube to video boston dynamics atlas on monitor 2",
     "boston+dynamics+atlas"),
    ("launch youtube playing the video Inception trailer on my right monitor",
     "Inception+trailer"),
    ("open youtube searching for octopus camouflage on the main monitor",
     "octopus+camouflage"),
    ("open youtube search for python tutorial on monitor 1",
     "python+tutorial"),
])
def test_classify_app_launch_youtube_video_or_search(text, fragment):
    """Video / search cues build a generic YouTube search URL."""
    intent = _classify_app_launch(text)
    assert intent is not None, f"failed for {text!r}"
    assert intent.app_name == "chrome"
    assert intent.url is not None
    assert "youtube.com/results?search_query=" in intent.url
    assert fragment in intent.url


def test_classify_app_launch_youtube_no_channel_falls_back_to_home():
    """Plain "open youtube on monitor X" still opens the home page."""
    intent = _classify_app_launch("open youtube on monitor 2")
    assert intent is not None
    assert intent.url == "https://www.youtube.com"
    assert "results?search_query" not in (intent.url or "")


# ---------------------------------------------------------------------------
# 2026-05-14: SCREEN_CONTEXT_QUERY -- position-adjective qualified screens
# ("what's on my MAIN screen?" / "what's on my LEFT monitor?")
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    # User's actual session-log phrasing that previously fell through to
    # the conversational LLM and returned "A task interface."
    "What's on my main screen?",
    "what's on my main monitor",
    "what is on my main screen",
    "what's on my primary monitor",
    "what's on my left monitor",
    "what's on my right screen",
    "what's on my second monitor",
    "what's on my center display",
    "what's on my third screen",
    "what is on the active monitor",
    "what's on my current screen",
    "tell me about what's on my main screen",
    "describe my left monitor",
    "look at my right screen",
])
def test_classify_screen_context_adjective_qualified(text):
    """Position adjectives between 'my' and 'screen' must still route
    to SCREEN_CONTEXT_QUERY instead of falling through to the
    conversational LLM."""
    intent = _classify_screen_context(text)
    assert intent is not None, f"failed to match: {text!r}"
    assert isinstance(intent, ScreenContextIntent)
    assert intent.include_vlm is True


@pytest.mark.parametrize("text,query_fragment", [
    ("show me a picture of a golden retriever", "golden retriever"),
    ("show me a picture of cats", "cats"),
    ("Show me a Picture of mountains", "mountains"),
    ("show me what a labrador looks like", "labrador"),
    ("find me a picture of the eiffel tower", "eiffel tower"),
    ("find me an image of pyramids", "pyramids"),
    ("i want to see a picture of an octopus", "octopus"),
])
def test_classify_app_launch_image_search(text, query_fragment):
    intent = _classify_app_launch(text)
    assert intent is not None, f"failed for {text!r}"
    assert intent.app_name == "chrome"
    assert intent.url is not None
    assert "tbm=isch" in intent.url
    # The query fragment should appear URL-encoded in the URL.
    from urllib.parse import quote_plus
    enc = quote_plus(query_fragment)
    assert enc.lower() in intent.url.lower()


def test_classify_app_launch_image_search_with_monitor():
    intent = _classify_app_launch(
        "show me a picture of a golden retriever on my second monitor",
    )
    assert intent is not None
    assert "tbm=isch" in (intent.url or "")
    assert intent.monitor_index == 1


def test_classify_app_launch_bare_url():
    # Use a domain NOT in _SITE_TO_URL so the bare-URL pattern fires
    # rather than the named-site pattern. wikipedia.org isn't in the
    # known-sites table.
    intent = _classify_app_launch("open wikipedia.org on monitor 2")
    assert intent is not None
    assert intent.app_name == "chrome"
    assert intent.url == "https://wikipedia.org"
    assert intent.monitor_index == 1


def test_classify_app_launch_named_site_with_monitor_resolves_to_full_url():
    """Named-site words (in _SITE_TO_URL) resolve to the canonical URL
    when a monitor target is present.
    """
    intent = _classify_app_launch("open youtube.com on monitor 2")
    assert intent is not None
    assert intent.app_name == "chrome"
    assert intent.url is not None
    assert "youtube.com" in intent.url


def test_classify_app_launch_bare_url_subpath_with_monitor():
    intent = _classify_app_launch("open reddit.com/r/python on monitor 2")
    assert intent is not None
    assert intent.app_name == "chrome"
    assert intent.url is not None
    assert "reddit.com" in intent.url


def test_classify_app_launch_bare_url_without_monitor_defers():
    """Bare URL without monitor target defers to BROWSER_AUTOMATION."""
    intent = _classify_app_launch("open wikipedia.org")
    assert intent is None


def test_classify_app_launch_no_match():
    assert _classify_app_launch("how is the weather today") is None
    assert _classify_app_launch("") is None
    assert _classify_app_launch("what is 2+2") is None


# ---------------------------------------------------------------------------
# classify_routing end-to-end
# ---------------------------------------------------------------------------


def test_classify_routing_screen_context_query():
    result = classify_routing("explain what I'm looking at")
    assert result.kind == RoutingIntentKind.SCREEN_CONTEXT_QUERY
    assert result.screen_context_intent is not None
    assert result.screen_context_intent.include_vlm is True


def test_classify_routing_app_launch_named_app():
    result = classify_routing("open chrome on my second monitor")
    assert result.kind == RoutingIntentKind.APP_LAUNCH
    assert result.app_launch_intent is not None
    assert result.app_launch_intent.app_name == "chrome"
    assert result.app_launch_intent.monitor_index == 1


def test_classify_routing_app_launch_image_search():
    result = classify_routing("show me a picture of a golden retriever")
    assert result.kind == RoutingIntentKind.APP_LAUNCH
    assert result.app_launch_intent is not None
    assert result.app_launch_intent.app_name == "chrome"
    assert "tbm=isch" in (result.app_launch_intent.url or "")


def test_classify_routing_app_launch_doesnt_swallow_conversational():
    """Verify the APP_LAUNCH regex doesn't false-match on bland queries."""
    convo = [
        "how do I install python",
        "tell me about the weather",
        "what's a transformer model",
        "explain how photosynthesis works",
        "summarize the latest news",
    ]
    for q in convo:
        result = classify_routing(q)
        assert result.kind == RoutingIntentKind.CONVERSATIONAL, (
            f"{q!r} unexpectedly routed to {result.kind}"
        )


def test_classify_routing_screen_context_doesnt_swallow_factual():
    """SCREEN_CONTEXT_QUERY shouldn't grab generic 'what is X' / 'explain X'."""
    convo = [
        "explain the theory of relativity",
        "what is a derivative in math",
        "describe the water cycle",
        "tell me about Hamlet",
    ]
    for q in convo:
        result = classify_routing(q)
        assert result.kind == RoutingIntentKind.CONVERSATIONAL, (
            f"{q!r} unexpectedly routed to {result.kind}"
        )


def test_classify_routing_pending_clarification_suppresses_app_launch():
    """When a clarification is pending, voice utterances feed back to the
    coordinator, not to a fresh routing decision.
    """
    result = classify_routing(
        "open chrome",
        has_pending_clarification=True,
    )
    # Falls to CLARIFICATION_RESPONSE (coding pipeline) or CONVERSATIONAL.
    # Critically: NOT APP_LAUNCH.
    assert result.kind != RoutingIntentKind.APP_LAUNCH


def test_classify_routing_pending_clarification_suppresses_screen_context():
    result = classify_routing(
        "what's on my screen",
        has_pending_clarification=True,
    )
    assert result.kind != RoutingIntentKind.SCREEN_CONTEXT_QUERY
