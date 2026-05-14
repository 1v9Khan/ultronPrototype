"""Routing classifier.

Layered:

1. **Coding triggers fire first.** Existing
   :func:`ultron.coding.intent.classify` handles CODE_TASK,
   PROGRESS_QUERY, CANCEL, MID_SESSION_ADJUSTMENT, CLARIFICATION_RESPONSE.
   When that returns NONE, fall through to the new categories.

2. **Hybrid signals next.** "set up environment for", "deploy",
   "automate workflow that..." — these mix coding + automation and need
   :class:`HybridTaskDecomposer` to split into subtasks.

3. **Automation rules.** Strong-signal regex per OpenClaw category
   (browser, media, messaging, files, shell).

4. **CONVERSATIONAL default.** Anything that doesn't match the above
   gets the default voice path.

Rule-based with explicit signals; LLM disambiguation kicks in via
:class:`IntentDisambiguator` when two categories tie.
"""

from __future__ import annotations

import re
from typing import Optional

from ultron.coding.intent import (
    CodingIntentKind,
    classify as classify_coding,
)


def _safe_get_config():
    """Best-effort config read for the gating logic below.

    Returns ``None`` on any failure (config not loaded, schema invalid)
    so the classifier always degrades to its default behaviour rather
    than raising. The downstream gates treat ``None`` as
    "all OpenClaw-bound features off" which matches the conservative
    pre-Phase-3 behaviour.
    """
    try:
        from ultron.config import get_config
        return get_config()
    except Exception:
        return None
from ultron.openclaw_routing.intents import (
    AppLaunchIntent,
    BrowserIntent,
    DesktopIntent,
    FileOpIntent,
    GamingModeIntent,
    MediaGenIntent,
    MessagingIntent,
    ModelSwitchIntent,
    RoutingIntent,
    RoutingIntentKind,
    ScreenContextIntent,
    ShellOpIntent,
    WindowIntent,
)


# ---------------------------------------------------------------------------
# Mapping coding-intent kinds to routing kinds
# ---------------------------------------------------------------------------


_CODING_KIND_MAP = {
    CodingIntentKind.NONE: None,                  # signals "fall through"
    CodingIntentKind.CODE_TASK: RoutingIntentKind.CODE_TASK,
    CodingIntentKind.PROGRESS_QUERY: RoutingIntentKind.PROGRESS_QUERY,
    CodingIntentKind.CANCEL: RoutingIntentKind.CANCEL,
    CodingIntentKind.MID_SESSION_ADJUSTMENT: RoutingIntentKind.MID_SESSION_ADJUSTMENT,
    CodingIntentKind.CLARIFICATION_RESPONSE: RoutingIntentKind.CLARIFICATION_RESPONSE,
}


# ---------------------------------------------------------------------------
# MODEL_SWITCH — voice-driven LLM preset swap (4B plan addition).
#
# Conservative regex: requires an action verb + an unambiguous model
# identifier so passing remarks like "the 4B should be faster" don't
# trigger an unwanted swap. Whisper homophones ("for B" / "four B")
# and spacing variants ("4 B", "4B", "4-B") are accepted because STT
# transcription of spoken letters/digits is inconsistent.
# ---------------------------------------------------------------------------


_MODEL_SWITCH_VERBS = (
    r"switch(?:\s+over)?|swap(?:\s+over)?|change(?:\s+over)?|"
    r"go|move|use|load|run|activate|engage|select"
)
_MODEL_SWITCH_4B_TOKEN = r"(?:4\s*[Bb]|four\s*[Bb]|for\s*[Bb]|4\s*-\s*[Bb])"
# 2026-05-14: 8B added because Josiefied-Qwen3-8B is the swap-back path
# from the new 4B default. Whisper transcribes spoken "8B" reliably as
# "8B" / "8 b" / "eight B".
_MODEL_SWITCH_8B_TOKEN = r"(?:8\s*[Bb]|eight\s*[Bb]|ate\s*[Bb]|8\s*-\s*[Bb])"
_MODEL_SWITCH_9B_TOKEN = r"(?:9\s*[Bb]|nine\s*[Bb]|9\s*-\s*[Bb])"
_MODEL_SWITCH_TOKEN = (
    rf"(?P<model>{_MODEL_SWITCH_4B_TOKEN}|{_MODEL_SWITCH_8B_TOKEN}|"
    rf"{_MODEL_SWITCH_9B_TOKEN})"
)
_MODEL_SWITCH_PATTERNS = re.compile(
    rf"\b(?:{_MODEL_SWITCH_VERBS})\s+(?:over\s+)?(?:to\s+|on\s+to\s+|onto\s+)?"
    # 2026-05-14: allow optional "the model" / "model" / "the llm" / etc.
    # between the verb and the model token so "switch to model 4B" /
    # "switch to the model 4B" both match. Previously only the trailing
    # "(?:\s+(?:model|llm|qwen))?\b" was honored, so users saying the
    # noun BEFORE the token (which Whisper transcribes naturally) hit
    # the conversational LLM instead of MODEL_SWITCH.
    rf"(?:the\s+)?(?:(?:the\s+)?(?:model|llm|preset|qwen)\s+)?"
    rf"{_MODEL_SWITCH_TOKEN}"
    r"(?:\s+(?:model|llm|qwen))?\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# System status — read-only voice queries about Ultron's own state
# (heartbeat alerts, active coding sessions, standing-order activity).
# Matched at high priority so utterances like "what is Ultron working on"
# don't get pulled into hybrid / coding rules below.
# ---------------------------------------------------------------------------


_SYSTEM_STATUS_ALERT_PATTERNS = re.compile(
    r"\b(?:"
    r"what\s+alerts(?:\s+did\s+you)?(?:\s+flag|\s+raise|\s+see)?|"
    r"any\s+(?:pending\s+|recent\s+|new\s+|open\s+)?alerts?|"
    r"any\s+(?:heartbeat\s+)?alerts?(?:\s+pending)?|"
    r"show\s+me\s+(?:the\s+)?alerts?|"
    r"list\s+(?:my\s+|the\s+)?alerts?|"
    r"alerts?\s+(?:status|summary|pending)|"
    r"what(?:'s|\s+is)?\s+pending(?:\s+for\s+me)?|"
    r"what\s+have\s+you\s+been\s+(?:flagging|noticing|tracking)"
    r")\b",
    re.IGNORECASE,
)

_SYSTEM_STATUS_PROJECT_PATTERNS = re.compile(
    r"\b(?:"
    r"what(?:'s|\s+is)?\s+(?:ultron\s+|currently\s+)?(?:working\s+on|running)|"
    r"what\s+(?:are\s+you|is\s+ultron)\s+(?:working\s+on|doing)|"
    r"(?:what\s+(?:are\s+the\s+|the\s+)?|list\s+(?:the\s+|all\s+)?)?"
    r"(?:active|in[-\s]?flight|pending)\s+(?:projects?|coding\s+(?:tasks?|sessions?)|tasks?|sessions?)|"
    r"any\s+active\s+(?:projects?|coding|tasks?|sessions?)|"
    r"(?:what\s+)?(?:standing\s+orders?|programs?)\s+(?:are\s+)?(?:active|running)|"
    r"any\s+(?:running|active)\s+coding"
    r")\b",
    re.IGNORECASE,
)

_SYSTEM_STATUS_BOTH_PATTERNS = re.compile(
    r"\b(?:"
    r"status\s+report|"
    r"system\s+status|"
    r"give\s+me\s+(?:a\s+)?status\s+update|"
    r"what(?:'s|\s+is)?\s+going\s+on|"
    r"what(?:'s|\s+is)?\s+(?:ultron's|your)\s+state"
    r")\b",
    re.IGNORECASE,
)


def _classify_system_status(text: str):
    """Return a (focus, reason) tuple if ``text`` is a SYSTEM_STATUS
    query, otherwise None. Focus is one of "alerts" / "projects" / "all"
    based on which pattern fired; ties prefer "all" (more inclusive)."""
    has_alerts = bool(_SYSTEM_STATUS_ALERT_PATTERNS.search(text))
    has_projects = bool(_SYSTEM_STATUS_PROJECT_PATTERNS.search(text))
    has_both = bool(_SYSTEM_STATUS_BOTH_PATTERNS.search(text))
    if has_both or (has_alerts and has_projects):
        return "all", "system-status combined pattern matched"
    if has_alerts:
        return "alerts", "system-status alerts pattern matched"
    if has_projects:
        return "projects", "system-status projects pattern matched"
    return None


def _resolve_model_switch_target(matched_token: str) -> str:
    """Map the matched-text variant back to the canonical preset name.

    ``matched_token`` is the contents of the ``(?P<model>...)`` group —
    e.g. "4B", "four B", "9 b", "for B", "8 b", "eight B". Returns one
    of the canonical preset names. 2026-05-14: "4B" / "8B" route to
    the Josiefied (abliterated) variants because those are the
    intentionally-maintained presets the user is choosing between
    after the 4B default landed; "9B" keeps the plain qwen3.5-9b
    swap-back path. Users who specifically want plain qwen3.5-4b set
    that in YAML or via ``ULTRON_LLM_PRESET``.
    """
    t = matched_token.lower().replace("-", "").replace(" ", "")
    if t.startswith("4") or t.startswith("four") or t.startswith("for"):
        return "josiefied-qwen3-4b"
    if t.startswith("8") or t.startswith("eight") or t.startswith("ate"):
        return "josiefied-qwen3-8b"
    if t.startswith("9") or t.startswith("nine"):
        return "qwen3.5-9b"
    # Defensive — regex shouldn't allow other tokens through.
    raise ValueError(f"Unrecognised model token: {matched_token!r}")


# ---------------------------------------------------------------------------
# Gaming mode (V1-spec gap A1) — anticheat-safe shutdown of OpenClaw
# desktop / windows control plugins. Voice triggers fire at HIGH
# priority (above HYBRID and automation rules) so a phrase like
# "I'm about to play Valorant" doesn't get pulled into a generic
# automation routing path.
# ---------------------------------------------------------------------------


_GAMING_MODE_ENGAGE = re.compile(
    r"\b(?:"
    r"gaming\s+mode(?:\s+on|\s+please)?|"
    r"engage\s+gaming\s+mode|"
    r"enter\s+gaming\s+mode|"
    r"(?:about\s+to|going\s+to|gonna|fixing\s+to)\s+play\s+"
    r"(?:valorant|cs(?:2|:?go)|counter[-\s]strike|fortnite|league|"
    r"overwatch|destiny|apex|warzone|rust|tarkov|pubg|"
    r"easy\s+anti[-\s]?cheat|vanguard|battleye)|"
    r"i'?m\s+(?:about\s+to\s+|going\s+to\s+|gonna\s+)?play(?:\s+a\s+game)?(?:\s+now)?|"
    r"shut(?:ting)?\s+down\s+desktop\s+(?:control|skills?)|"
    r"kill\s+desktop\s+control"
    r")\b",
    re.IGNORECASE,
)
_GAMING_MODE_DISENGAGE = re.compile(
    r"\b(?:"
    r"gaming\s+mode\s+off|"
    r"(?:disengage|exit|leave|end)\s+gaming\s+mode|"
    r"(?:done|finished)\s+(?:gaming|playing)|"
    r"restore\s+desktop\s+(?:control|skills?)|"
    r"full\s+control(?:\s+restored)?|"
    r"i'?m\s+done\s+playing"
    r")\b",
    re.IGNORECASE,
)
_GAMING_MODE_STATUS = re.compile(
    r"\b(?:"
    r"(?:are\s+we|am\s+i)\s+in\s+gaming\s+mode|"
    r"is\s+gaming\s+mode\s+(?:on|active|engaged)|"
    r"gaming\s+mode\s+status|"
    r"what(?:'s|\s+is)\s+(?:my\s+)?gaming\s+mode"
    r")\b",
    re.IGNORECASE,
)


def _classify_gaming_mode(text: str):
    """Return a (action, trigger_phrase) tuple if ``text`` matches a
    gaming-mode pattern, otherwise ``None``."""
    m = _GAMING_MODE_DISENGAGE.search(text)
    if m:
        return ("disengage", m.group(0))
    m = _GAMING_MODE_STATUS.search(text)
    if m:
        return ("status", m.group(0))
    m = _GAMING_MODE_ENGAGE.search(text)
    if m:
        return ("engage", m.group(0))
    return None


# ---------------------------------------------------------------------------
# Desktop automation (V1-spec gap C3) — voice routing for the
# OpenClaw ``desktop-control`` plugin. Distinct from BROWSER_AUTOMATION's
# screenshot pattern (which expects a URL or "of github.com" context);
# DESKTOP_AUTOMATION fires for "the screen", "my desktop", "the active
# window".
# ---------------------------------------------------------------------------


_DESKTOP_SCREENSHOT = re.compile(
    r"\b(?:"
    r"(?:take\s+|capture\s+|grab\s+)?(?:a\s+)?screenshot\s+of\s+"
    r"(?:my\s+|the\s+)?(?:screen|desktop|monitor|active\s+window|current\s+window)|"
    r"screenshot\s+(?:my\s+|the\s+)?(?:screen|desktop|monitor|active\s+window)|"
    r"(?:take|capture|grab)\s+(?:a\s+)?screenshot(?:\s+of\s+everything)?$|"
    r"snap\s+(?:a\s+|me\s+a\s+)?screenshot|"
    r"capture\s+the\s+(?:screen|desktop|monitor)"
    r")",
    re.IGNORECASE,
)
_DESKTOP_LIST_WINDOWS = re.compile(
    r"\b(?:"
    r"list\s+(?:my\s+|the\s+|all\s+)?(?:open\s+)?windows|"
    r"what\s+windows\s+(?:are\s+)?(?:open|running)|"
    r"show\s+(?:me\s+)?(?:my\s+|the\s+|all\s+)?(?:open\s+)?windows|"
    r"enumerate\s+(?:my\s+|the\s+)?windows"
    r")\b",
    re.IGNORECASE,
)
_DESKTOP_FIND_WINDOW = re.compile(
    r"\b(?:"
    r"find\s+(?:the\s+|my\s+)?(?P<query1>[\w\s]+?)\s+window|"
    r"locate\s+(?:the\s+|my\s+)?(?P<query2>[\w\s]+?)\s+window|"
    r"where(?:'s|\s+is)\s+(?:the\s+|my\s+)?(?P<query3>[\w\s]+?)\s+window"
    r")\b",
    re.IGNORECASE,
)


def _classify_desktop(text: str):
    """Return a desktop intent (action, target) if matched, else None."""
    if _DESKTOP_SCREENSHOT.search(text):
        # Try to extract a target phrase.
        target = None
        m_active = re.search(
            r"(?:active|current)\s+window", text, re.IGNORECASE,
        )
        if m_active:
            target = "active_window"
        return ("screenshot", target)
    if _DESKTOP_LIST_WINDOWS.search(text):
        return ("list_windows", None)
    m = _DESKTOP_FIND_WINDOW.search(text)
    if m:
        # Pick whichever named capture group fired.
        for name in ("query1", "query2", "query3"):
            try:
                v = m.group(name)
            except Exception:
                v = None
            if v and v.strip():
                return ("find_window", v.strip())
        return ("find_window", None)
    return None


# ---------------------------------------------------------------------------
# Window automation (V1-spec gap C3) — voice routing for OpenClaw
# ``windows-control`` plugin. UI Automation primitives.
# ---------------------------------------------------------------------------


_WINDOW_FOCUS = re.compile(
    r"\b(?:"
    r"focus\s+(?:the\s+|my\s+)?(?P<q1>[\w\s]+?)\s+window|"
    r"switch\s+to\s+(?:the\s+|my\s+)?(?P<q2>[\w\s]+?)(?:\s+window)?$|"
    r"bring\s+(?:the\s+|my\s+)?(?P<q3>[\w\s]+?)\s+(?:window\s+)?to\s+(?:the\s+)?front|"
    r"activate\s+(?:the\s+|my\s+)?(?P<q4>[\w\s]+?)\s+window"
    r")",
    re.IGNORECASE,
)
_WINDOW_TYPE = re.compile(
    r"\b(?:"
    r"type\s+(?:'(?P<v1>[^']*)'|\"(?P<v2>[^\"]*)\")\s+into\s+(?:the\s+|my\s+)?(?P<wq1>[\w\s]+)|"
    r"enter\s+(?:'(?P<v3>[^']*)'|\"(?P<v4>[^\"]*)\")\s+into\s+(?:the\s+|my\s+)?(?P<wq2>[\w\s]+)"
    r")",
    re.IGNORECASE,
)
_WINDOW_CLICK = re.compile(
    r"\b(?:"
    r"click\s+(?:the\s+|my\s+)?(?P<element>[\w\s]+?)\s+in\s+(?:the\s+|my\s+)?(?P<window>[\w\s]+?)\s+window|"
    r"click\s+(?:the\s+|my\s+)?(?P<element2>[\w\s]+?)\s+button\s+(?:in|on)\s+(?:the\s+|my\s+)?(?P<window2>[\w\s]+)"
    r")",
    re.IGNORECASE,
)


def _classify_window(text: str):
    """Return a window intent tuple (action, query, ref, value) if matched."""
    m = _WINDOW_TYPE.search(text)
    if m:
        value = m.group("v1") or m.group("v2") or m.group("v3") or m.group("v4") or ""
        query = (m.group("wq1") or m.group("wq2") or "").strip()
        return ("type", query, None, value)
    m = _WINDOW_CLICK.search(text)
    if m:
        element = (m.group("element") or m.group("element2") or "").strip()
        window = (m.group("window") or m.group("window2") or "").strip()
        return ("click", window, element, None)
    m = _WINDOW_FOCUS.search(text)
    if m:
        for name in ("q1", "q2", "q3", "q4"):
            try:
                v = m.group(name)
            except Exception:
                v = None
            if v and v.strip():
                return ("focus", v.strip(), None, None)
    return None


# ---------------------------------------------------------------------------
# Hybrid signals — coding-related verb + system context that requires
# both code generation AND filesystem / shell / browser automation.
# ---------------------------------------------------------------------------


_HYBRID_PATTERNS = re.compile(
    r"(?:"
    # Environment-setup workflows
    r"\bset\s+up\s+(?:a\s+)?(?:dev|development|local|build|test|staging|production)\s+environment\b|"
    r"\bset\s+up\s+(?:a\s+)?(?:env|venv|virtualenv)\b|"
    r"\binstall\s+(?:and\s+configure\s+)?dependencies\s+for\b|"
    # Deployment
    r"\bdeploy\s+(?:this|that|my|the)\b|"
    r"\bship\s+(?:this|that|my|the)\s+(?:to|on|over)\b|"
    # "automate my X workflow" / "automate the X process" — allow filler words
    # between the determiner and the workflow noun.
    r"\bautomate\s+(?:my|the|that)\s+(?:[\w\-]+\s+){0,3}(?:workflow|process|pipeline|task|setup|routine)\b|"
    r"\bautomate\s+the\s+process\s+of\b|"
    # Script/tool that drives existing software
    r"\b(?:write|build|make|create)\s+(?:a\s+)?(?:script|tool)\s+(?:that\s+)?"
    r"(?:opens|runs|controls|drives|automates|scrapes)\b|"
    r"\b(?:build|make|create|write)\s+(?:a\s+)?(?:script|tool)\s+(?:for|to)\s+"
    r"(?:my\s+)?(?:excel|browser|chrome|firefox|outlook|word)\b|"
    r"\b(?:build|make|create|write)\s+(?:a\s+)?(?:tool|script)\s+for\s+(?:my\s+)?browser\b"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Browser automation
# ---------------------------------------------------------------------------


_BROWSER_NAVIGATE = re.compile(
    r"\b(?:"
    r"open\s+(?:up\s+)?(?:the\s+)?(?:tab|page|browser|website|url|link)\s+|"
    r"open\s+(?:up\s+)?(?:hacker\s+news|wikipedia|youtube|github|reddit|twitter|x\.com|"
    r"google|gmail|stack\s*overflow|claude\.ai|chatgpt)\b|"
    # "open [a/an/the/your/my] [new] browser [window|tab]" with optional
    # destination ("with|to|for|on" + target). Catches "Can you open a
    # browser window with Google's homepage for me?" which the
    # determiner-less pattern above missed (it required "the" or no
    # determiner at all). Destination is optional so a bare "open a
    # new browser window" (no target) still routes to the browser
    # tool with the default landing page.
    r"open\s+(?:up\s+)?(?:a\s+new\s+|new\s+|a\s+|an\s+|the\s+|your\s+|my\s+)?"
    r"browser(?:\s+(?:window|tab))?\b|"
    # "open a/an/the/your/my [new] (window|tab) (with|to|for|on) X" -- the
    # window/tab variant without explicit "browser" word. Destination
    # required here so it doesn't false-match "open a new tab" in a
    # general non-browser sense (terminal tab, IDE tab, etc.).
    r"open\s+(?:up\s+)?(?:a\s+new\s+|new\s+|a\s+|an\s+|the\s+|your\s+|my\s+)"
    r"(?:window|tab)\s+(?:with|to|for|on)\s+|"
    r"navigate\s+to\s+|"
    r"go\s+to\s+(?:the\s+)?(?:url|link|page|site|website)\b|"
    r"pull\s+up\s+(?:the\s+)?(?:url|link|page|site|website|wikipedia|hacker\s+news)\b|"
    r"^open\s+https?://"
    r")",
    re.IGNORECASE,
)
_BROWSER_INTERACT = re.compile(
    r"\b(?:"
    r"click\s+(?:on\s+)?(?:the\s+)?(?:button|link|\w+\s+button|\w+\s+link)|"
    # Fill any kind of form — "fill in the form", "fill out the contact form"
    r"fill\s+(?:in|out)\s+(?:the\s+)?(?:[\w\-]+\s+)?form|"
    r"take\s+(?:a\s+)?screenshot|"
    r"log\s+(?:in)to\s+(?:my\s+)?(?:account|github|gmail)|"
    r"sign\s+(?:in)to\s+|"
    r"submit\s+(?:the\s+)?form|"
    r"scroll\s+(?:"
    r"(?:down|up|to)\s+the|"
    r"the\s+(?:page|window|tab|view|content|results|list)\s+(?:down|up|left|right|to)"
    r")"
    r")\b",
    re.IGNORECASE,
)
_BROWSER_LIVE_QUERY = re.compile(
    # "what does X say right now" / "search Google for X" — interactive
    # vs. a text-snippet web search
    r"\b(?:"
    r"what\s+does\s+\w+(?:\s+\w+){0,3}\s+say\s+(?:right\s+now|currently|today)|"
    r"search\s+(?:for\s+)?(?:[\w\s'\"-]+?)\s+on\s+google\b|"
    r"google\s+(?:[\w\s'\"-]+?)\s+for\s+me\b"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Media generation
# ---------------------------------------------------------------------------


_MEDIA_PATTERNS = re.compile(
    r"\b(?:"
    # "make me an image of" / "make me a song about" — open up the noun
    # set so the audio family of media matches consistently.
    r"make\s+me\s+an?\s+(?:image|picture|illustration|painting|drawing|render|"
    r"song|track|tune|video|clip)\s+(?:of|about|that)|"
    # Generate (a/an) (short/long/...) (image/video/...) — optional adjective
    r"generate\s+an?\s+(?:[\w\-]+\s+){0,2}"
    r"(?:image|picture|illustration|painting|drawing|render|artwork|video|clip|song|audio|music|tune|track)\b|"
    r"create\s+(?:an?\s+)?artwork|"
    r"create\s+(?:an?\s+)?(?:song|track|tune)\s+(?:about|that)|"
    r"compose\s+(?:a\s+)?(?:song|track|tune|piece|melody|beat|music)|"
    r"draw\s+me\s+|"
    r"render\s+(?:me\s+)?(?:an?|the)\s+(?:image|scene|picture|video|illustration|drawing|artwork)\b|"
    r"paint\s+me\s+(?:an?\s+)?(?:image|picture)|"
    r"give\s+me\s+(?:an?\s+)?(?:image|picture|video|song)\s+of"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Messaging
# ---------------------------------------------------------------------------


_MESSAGING_PATTERNS = re.compile(
    r"\b(?:"
    r"send\s+(?:me\s+)?(?:a\s+)?(?:message|notification|push|alert|text)\s+(?:to|on)\s+(?:my\s+)?phone|"
    r"send\s+(?:me\s+)?(?:a\s+)?push\s+(?:notification|notif)\b|"
    r"text\s+me\b|"
    r"notify\s+me\s+when\b|"
    r"notify\s+me\s+(?:on|via)\s+(?:telegram|signal|slack|discord)\b|"
    r"tell\s+me\s+on\s+(?:telegram|signal|slack|discord)|"
    r"send\s+(?:to\s+)?telegram|"
    r"ping\s+me\s+(?:on|when)|"
    r"shoot\s+me\s+(?:a\s+)?(?:message|text)|"
    r"alert\s+me\s+when"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# File operations (outside project sandbox)
# ---------------------------------------------------------------------------


_FILE_PATTERNS = re.compile(
    r"\b(?:"
    r"read\s+(?:the\s+)?file\s+at\s+|"
    r"show\s+me\s+(?:the\s+)?contents\s+of\s+(?:the\s+)?file\s+|"
    r"show\s+me\s+(?:the\s+)?contents\s+of\s+[\w./\\-]+\.[a-z]{1,5}\b|"
    r"open\s+(?:the\s+)?file\s+at\s+|"
    r"write\s+(?:to\s+)?(?:the\s+)?file\s+at\s+|"
    r"save\s+(?:to\s+)?(?:a\s+)?file\s+at\s+|"
    r"delete\s+(?:the\s+)?file\s+at\s+|"
    r"remove\s+(?:the\s+)?file\s+at\s+|"
    r"list\s+(?:the\s+)?files\s+in\s+|"
    r"show\s+(?:me\s+)?(?:the\s+)?files\s+in\s+(?:the\s+)?(?:directory|folder)\s+|"
    r"what(?:'s|\s+is)\s+in\s+(?:the\s+)?(?:directory|folder)\s+"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Shell operations
# ---------------------------------------------------------------------------


_SHELL_PATTERNS = re.compile(
    r"\b(?:"
    r"run\s+(?:the\s+command\s+)?[\"'`]?(?:dir|ls|pwd|whoami|hostname|date|uptime|"
    r"git\s+\w+|npm\s+\w+|pip\s+\w+|python\s+|node\s+|cargo\s+\w+|"
    r"echo\s+|cat\s+|grep\s+|find\s+|curl\s+|wget\s+)|"
    r"execute\s+(?:the\s+)?(?:command|shell)|"
    r"what(?:'s|\s+is)\s+the\s+output\s+of\s+|"
    r"in\s+(?:the\s+)?(?:terminal|shell|powershell|cmd|bash)\s+(?:run|execute|do)"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public classify
# ---------------------------------------------------------------------------


def classify_routing(
    utterance: str,
    has_active_coding_task: bool = False,
    has_pending_clarification: bool = False,
) -> RoutingIntent:
    """Classify ``utterance`` into a top-level :class:`RoutingIntent`.

    Order:
      1. Coding intent (delegated to ``ultron.coding.intent.classify``)
      2. Hybrid task signals
      3. Automation rules (browser / media / messaging / file / shell)
      4. CONVERSATIONAL fallback

    Args mirror the existing coding classifier so callers don't have to
    track two separate "is something running" flags.

    V1-gap A1 / C3: gaming-mode, desktop, and window-control routing
    is gated on the master ``openclaw.enabled`` flag plus the
    individual feature flags. When OpenClaw is offline (default) the
    new patterns DO NOT fire -- utterances like "take a screenshot of
    the desktop" fall through to the conversational LLM instead of
    being routed to a stub message. Once the user wires OpenClaw they
    take effect automatically.
    """
    text = (utterance or "").strip()
    # Resolve the OpenClaw / per-feature flags once per call. Cheap --
    # the config singleton is a hot read.
    cfg = _safe_get_config()
    openclaw_on = bool(cfg and cfg.openclaw.enabled)
    desktop_on = bool(openclaw_on and cfg and cfg.desktop.enabled)
    window_on = bool(openclaw_on and cfg and cfg.window_control.enabled)
    gaming_on = bool(openclaw_on and cfg and cfg.gaming_mode.enabled)
    if not text:
        return RoutingIntent(
            kind=RoutingIntentKind.CONVERSATIONAL,
            raw_text="",
            source="default",
            reason="empty utterance",
            confidence=1.0,
        )

    # 1) IN-FLIGHT TASK COMMANDS first — cancel/progress/adjustment/clarification
    #    must take precedence even when the rest of the utterance contains
    #    automation-keyword overlap. The coding classifier handles these
    #    when has_active_task=True (or has_pending_clarification=True);
    #    we only fall through to hybrid/automation when the coding result
    #    is one of the "starts a new task" or "no match" verdicts.
    coding = classify_coding(
        text,
        has_active_task=has_active_coding_task,
        has_pending_clarification=has_pending_clarification,
    )
    if coding.kind in (
        # In-flight commands: never override these with a routing rule.
        # They fire only when has_active_task / has_pending_clarification
        # is set, so by definition there's a task to act on.
        # CodingIntentKind enum values:
        # PROGRESS_QUERY, CANCEL, MID_SESSION_ADJUSTMENT, CLARIFICATION_RESPONSE
    ):
        pass  # placeholder for clarity
    if coding.kind.value in (
        "progress_query", "cancel",
        "mid_session_adjustment", "clarification_response",
    ):
        return RoutingIntent(
            kind=_CODING_KIND_MAP[coding.kind],
            raw_text=text,
            confidence=coding.confidence,
            source="rule",
            reason=coding.reason,
            coding_intent=coding,
        )

    # 1.4) SYSTEM_STATUS — read-only voice queries about Ultron's own
    #      state. Resolved without OpenClaw (read from heartbeat alert
    #      log + active session list). High-priority match so
    #      utterances like "what is Ultron working on" don't get pulled
    #      into hybrid / coding rules.
    if not has_pending_clarification:
        status_match = _classify_system_status(text)
        if status_match is not None:
            from ultron.openclaw_routing.intents import SystemStatusIntent
            focus, reason = status_match
            return RoutingIntent(
                kind=RoutingIntentKind.SYSTEM_STATUS,
                raw_text=text,
                confidence=0.9,
                source="rule",
                reason=reason,
                system_status_intent=SystemStatusIntent(
                    focus=focus, raw_text=text,
                ),
            )

    # 1.5) MODEL_SWITCH — must come BEFORE hybrid / automation rules so
    #      "switch to 4B" doesn't get pulled into a different category.
    #      We ignore these mid-task: an active coding task with a
    #      pending clarification has higher precedence; honoring them
    #      here would interrupt work-in-progress mid-flight.
    if not has_pending_clarification:
        m = _MODEL_SWITCH_PATTERNS.search(text)
        if m:
            return RoutingIntent(
                kind=RoutingIntentKind.MODEL_SWITCH,
                raw_text=text,
                confidence=0.95,
                source="rule",
                reason="model-switch pattern matched",
                model_switch_intent=ModelSwitchIntent(
                    target_preset=_resolve_model_switch_target(m.group("model")),
                    raw_text=text,
                ),
            )

    # 1.6) GAMING_MODE (V1-gap A1) — anticheat-safe shutdown of OpenClaw
    #      desktop / windows control plugins. Highest priority among the
    #      automation-adjacent rules so "I'm about to play Valorant"
    #      doesn't get pulled into HYBRID's "automate my workflow" branch.
    #      Gated on openclaw.enabled AND gaming_mode.enabled so the
    #      patterns don't fire (and produce stub messages) when the
    #      Gateway / feature is offline -- utterances fall through to
    #      conversational LLM instead.
    if not has_pending_clarification and gaming_on:
        gm = _classify_gaming_mode(text)
        if gm is not None:
            action, trigger = gm
            return RoutingIntent(
                kind=RoutingIntentKind.GAMING_MODE,
                raw_text=text,
                confidence=0.95,
                source="rule",
                reason=f"gaming-mode pattern matched ({action})",
                gaming_mode_intent=GamingModeIntent(
                    action=action,
                    trigger_phrase=trigger,
                    raw_text=text,
                ),
            )

    # 1.7) DESKTOP_AUTOMATION (V1-gap C3) — fires BEFORE BROWSER_AUTOMATION
    #      so "take a screenshot of the desktop" routes to desktop, not
    #      browser. Browser-screenshot still wins when the utterance has
    #      a URL (handled by classify by URL marker on the browser side).
    #      Gated on openclaw.enabled AND desktop.enabled.
    if not has_pending_clarification and desktop_on and not _URL_RE.search(text):
        desktop = _classify_desktop(text)
        if desktop is not None:
            action, target = desktop
            return RoutingIntent(
                kind=RoutingIntentKind.DESKTOP_AUTOMATION,
                raw_text=text,
                confidence=0.85,
                source="rule",
                reason=f"desktop pattern matched ({action})",
                desktop_intent=DesktopIntent(
                    action=action, target=target, raw_text=text,
                ),
            )

    # 1.8) WINDOW_AUTOMATION (V1-gap C3) — UI Automation primitives.
    #      Fires before BROWSER_AUTOMATION's interact patterns because
    #      window utterances ("focus the chrome window") share verbs
    #      ("click", "focus") with the browser branch. Gated on
    #      openclaw.enabled AND window_control.enabled.
    if not has_pending_clarification and window_on:
        window = _classify_window(text)
        if window is not None:
            action, query, ref, value = window
            return RoutingIntent(
                kind=RoutingIntentKind.WINDOW_AUTOMATION,
                raw_text=text,
                confidence=0.85,
                source="rule",
                reason=f"window pattern matched ({action})",
                window_intent=WindowIntent(
                    action=action, query=query, ref=ref, value=value,
                    raw_text=text,
                ),
            )

    # 1.9) SCREEN_CONTEXT_QUERY (Phase 8) -- "explain what I'm looking at",
    #      "what's on my screen". Native (no OpenClaw dependency); always
    #      fires when phrasing matches. Higher priority than the bare
    #      coding/browser rules below because the phrasing is specific.
    if not has_pending_clarification:
        sc = _classify_screen_context(text)
        if sc is not None:
            return RoutingIntent(
                kind=RoutingIntentKind.SCREEN_CONTEXT_QUERY,
                raw_text=text,
                confidence=0.9,
                source="rule",
                reason="screen-context query pattern matched",
                screen_context_intent=sc,
            )

    # 2.0) APP_LAUNCH (Phase 8) -- "open YouTube on monitor 2",
    #      "launch Cursor on my left monitor", "show me a picture of X".
    #      Native via :mod:`ultron.desktop.launcher`; routes to user's
    #      real Chrome / Cursor / etc. (NOT the OpenClaw Playwright
    #      plugin). Must fire BEFORE the BROWSER_AUTOMATION rule below
    #      so "open google.com" doesn't go to the isolated Playwright
    #      profile.
    if not has_pending_clarification:
        al = _classify_app_launch(text)
        if al is not None:
            return RoutingIntent(
                kind=RoutingIntentKind.APP_LAUNCH,
                raw_text=text,
                confidence=0.9,
                source="rule",
                reason="app-launch pattern matched",
                app_launch_intent=al,
            )

    # 2) HYBRID signals next — these often contain coding-trigger keywords
    #    ("write a script", "build a tool") so we have to win the race
    #    against CODE_TASK rules below.
    if _HYBRID_PATTERNS.search(text):
        return RoutingIntent(
            kind=RoutingIntentKind.HYBRID_TASK,
            raw_text=text,
            confidence=0.85,
            source="rule",
            reason="hybrid coding+automation pattern matched",
            # Subtasks populated by HybridTaskDecomposer downstream.
        )

    # 3) CODE_TASK (the only remaining non-NONE coding kind).
    if coding.kind.value == "code_task":
        return RoutingIntent(
            kind=RoutingIntentKind.CODE_TASK,
            raw_text=text,
            confidence=coding.confidence,
            source="rule",
            reason=coding.reason,
            coding_intent=coding,
        )

    # 3) Single-category automation rules.
    if _BROWSER_NAVIGATE.search(text) or _BROWSER_INTERACT.search(text) or _BROWSER_LIVE_QUERY.search(text):
        return RoutingIntent(
            kind=RoutingIntentKind.BROWSER_AUTOMATION,
            raw_text=text,
            confidence=0.85,
            source="rule",
            reason="browser-automation pattern matched",
            automation_intent=_build_browser_intent(text),
        )

    if _MEDIA_PATTERNS.search(text):
        return RoutingIntent(
            kind=RoutingIntentKind.MEDIA_GENERATION,
            raw_text=text,
            confidence=0.85,
            source="rule",
            reason="media-generation pattern matched",
            automation_intent=_build_media_intent(text),
        )

    if _MESSAGING_PATTERNS.search(text):
        return RoutingIntent(
            kind=RoutingIntentKind.MESSAGING,
            raw_text=text,
            confidence=0.85,
            source="rule",
            reason="messaging pattern matched",
            automation_intent=_build_messaging_intent(text),
        )

    if _FILE_PATTERNS.search(text):
        return RoutingIntent(
            kind=RoutingIntentKind.FILE_OPERATION,
            raw_text=text,
            confidence=0.85,
            source="rule",
            reason="file-operation pattern matched",
            automation_intent=_build_file_intent(text),
        )

    if _SHELL_PATTERNS.search(text):
        return RoutingIntent(
            kind=RoutingIntentKind.SHELL_OPERATION,
            raw_text=text,
            confidence=0.85,
            source="rule",
            reason="shell-operation pattern matched",
            automation_intent=_build_shell_intent(text),
        )

    # 4) CONVERSATIONAL fallback.
    return RoutingIntent(
        kind=RoutingIntentKind.CONVERSATIONAL,
        raw_text=text,
        confidence=0.6,
        source="default",
        reason="no rule matched; default conversational",
    )


# ---------------------------------------------------------------------------
# Light-weight intent builders (extract structure from raw text)
# ---------------------------------------------------------------------------


_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Phase 8 (2026-05-12): Desktop automation native classifier patterns.
#
# These fire BEFORE the OpenClaw-gated BROWSER / DESKTOP / WINDOW patterns
# because they route to native modules (ultron.desktop.*) that don't
# depend on the OpenClaw Gateway being enabled. The user's specific use
# cases:
#   - "open YouTube on my 2nd monitor" -> APP_LAUNCH with Chrome+URL+monitor
#   - "show me a picture of golden retriever" -> APP_LAUNCH (image search)
#   - "open Cursor on my left monitor" -> APP_LAUNCH
#   - "explain what I'm looking at" -> SCREEN_CONTEXT_QUERY
# ---------------------------------------------------------------------------


# Monitor target tokens: ordinal words, digits with "monitor"/"screen",
# directional. Captures the matched text for downstream parsing.
_MONITOR_TARGET_RE = re.compile(
    r"\bon\s+(?:my\s+|the\s+)?"
    r"(?:"
    r"(?:1st|first|2nd|second|3rd|third|4th|fourth|primary|main|"
    r"left|right|center|centre|middle|top|bottom)"
    r"\s+(?:monitor|screen|display)"
    r"|"
    r"monitor\s+(?:1|2|3|4|one|two|three|four)"
    r"|"
    r"screen\s+(?:1|2|3|4|one|two|three|four)"
    r"|"
    r"display\s+(?:1|2|3|4|one|two|three|four)"
    r")\b",
    re.IGNORECASE,
)

# Ordinal-words map for monitor extraction. Pure ordinals (first / second /
# etc.) resolve to a zero-based index. "main" and "primary" are NOT in
# this map -- they're position-based / Win32-based and need to flow
# through ``find_monitor`` at dispatch time so they pick up the user's
# physical layout instead of a hardcoded index.
_MONITOR_ORDINAL_TO_INDEX = {
    "first": 0, "1st": 0, "one": 0,
    "second": 1, "2nd": 1, "two": 1,
    "third": 2, "3rd": 2, "three": 2,
    "fourth": 3, "4th": 3, "four": 3,
}
# Words preserved as strings and resolved by find_monitor at dispatch.
# "main" -> physical center (user-direction 2026-05-14).
# "primary" -> Win32-primary (kept for callers who explicitly want it).
_MONITOR_DIRECTIONAL_WORDS = (
    "left", "right", "center", "centre", "middle", "top", "bottom",
    "main", "primary",
)


def _extract_monitor_target(text: str) -> tuple[Optional[int], str]:
    """Extract a monitor target from the utterance.

    Returns ``(monitor_index, monitor_query)`` -- when an explicit
    digit ("monitor 2") or ordinal ("second monitor", "primary
    monitor") is found, ``monitor_index`` is the zero-based index and
    ``monitor_query`` is the matched phrase. Directional words
    ("left monitor", "right screen") yield ``monitor_index=None``
    and the directional word as ``monitor_query`` so the launcher's
    :func:`find_monitor` can resolve at dispatch time.

    No match returns ``(None, "")``.
    """
    m = _MONITOR_TARGET_RE.search(text or "")
    if not m:
        return None, ""
    raw = m.group(0)
    raw_lower = raw.lower()

    # Ordinal/cardinal words (preferred -- explicit index).
    for word, idx in _MONITOR_ORDINAL_TO_INDEX.items():
        if word in raw_lower:
            return idx, raw

    # Explicit digit (monitor 2 / screen 3 / display 4).
    digit_match = re.search(
        r"(?:monitor|screen|display)\s+(?:(\d)|"
        r"(one|two|three|four))",
        raw_lower,
    )
    if digit_match:
        digit, word = digit_match.group(1), digit_match.group(2)
        if digit:
            return int(digit) - 1, raw  # "monitor 2" -> index 1
        if word in _MONITOR_ORDINAL_TO_INDEX:
            return _MONITOR_ORDINAL_TO_INDEX[word], raw

    # Directional: defer resolution.
    for d in _MONITOR_DIRECTIONAL_WORDS:
        if d in raw_lower:
            return None, d

    return None, raw


# SCREEN_CONTEXT_QUERY: utterances asking about the current screen state.
# Higher priority than BROWSER -- "show me my screen" shouldn't route
# to browser. Tight regex so it doesn't swallow general "what is X" queries.
# 2026-05-14: allow position adjectives (main / primary / left / right /
# center / second / etc.) between "my" and "screen|display|monitor" so
# "what's on my MAIN screen" / "what's on my left monitor" still route
# to SCREEN_CONTEXT_QUERY instead of falling through to the conversational
# LLM (which the user's session log showed answering "A task interface."
# with no actual screen context).
_SCREEN_NOUN_ADJ = (
    r"(?:main\s+|primary\s+|center\s+|centre\s+|middle\s+|"
    r"left\s+|right\s+|top\s+|bottom\s+|"
    r"first\s+|1st\s+|second\s+|2nd\s+|third\s+|3rd\s+|fourth\s+|4th\s+|"
    r"other\s+|active\s+|current\s+|focused\s+)?"
)
_SCREEN_NOUN = r"(?:screen|display|monitor)"
_SCREEN_DET = r"(?:my\s+|the\s+)?"
_SCREEN_CONTEXT_PATTERNS = re.compile(
    r"\b(?:"
    # "explain what I'm/I am looking at"
    r"explain\s+(?:what\s+)?(?:i'm|i\s+am|im)\s+(?:looking\s+at|seeing|doing|working\s+on)|"
    # "what(s) on my main screen" / "what's on my screen" / "what's on the active monitor"
    rf"what(?:'s|\s+is)?\s+(?:on\s+)?{_SCREEN_DET}{_SCREEN_NOUN_ADJ}{_SCREEN_NOUN}|"
    # "what am I looking at", "what are you seeing"
    r"what\s+am\s+i\s+(?:looking\s+at|seeing)|"
    r"what\s+(?:do|can)\s+you\s+(?:see|see\s+(?:right\s+)?now)|"
    # "look at my screen and ...", "look at what I'm doing"
    rf"look\s+at\s+{_SCREEN_DET}(?:{_SCREEN_NOUN_ADJ}{_SCREEN_NOUN}|"
    r"what\s+i'm\s+doing|this)|"
    # "tell me about (what's) on my screen / monitor / display"
    rf"tell\s+me\s+(?:about\s+)?(?:what(?:'s|\s+is)\s+on\s+)?{_SCREEN_DET}{_SCREEN_NOUN_ADJ}{_SCREEN_NOUN}|"
    # "describe (what's on) my screen / left monitor / etc."
    rf"describe\s+(?:what(?:'s|\s+is)\s+on\s+)?{_SCREEN_DET}{_SCREEN_NOUN_ADJ}{_SCREEN_NOUN}|"
    # "what is this (on the screen)" -- requires "this" alone or with screen context
    r"what(?:'s|\s+is)\s+this\s+(?:on\s+(?:my\s+|the\s+)?screen|here)|"
    # "help me with this", "help me with what I'm working on"
    r"help\s+me\s+(?:with\s+)?(?:this|what\s+i'm\s+(?:working\s+on|doing)|"
    r"my\s+screen)|"
    # "explain this (code|error|page|screen)" — high-specificity
    r"explain\s+this\s+(?:code|error|page|screen|window|message|dialog|"
    r"window|app|application)|"
    # "what does this (error|message|dialog) mean"
    r"what\s+does\s+this\s+(?:error|message|dialog|notification|alert|"
    r"button|window|popup|prompt)\s+(?:mean|say|do)"
    r")\b",
    re.IGNORECASE,
)


# APP_LAUNCH: "open <X>", "launch <X>", "pull up <X>", "start <X>",
# "fire up <X>", "throw up <X>", "bring up <X>". Combined with monitor
# targeting from _MONITOR_TARGET_RE.
_APP_LAUNCH_VERB_PATTERN = (
    r"(?:open(?:\s+up)?|launch|start|run|fire\s+up|pull\s+up|"
    r"bring\s+up|throw\s+(?:up|on)|show\s+me)"
)

# Apps the launcher's default registry knows about. Match against the
# user's phrase; the launcher's substring-fallback handles slight
# variants ("google chrome" → chrome, "vs code" → vscode).
_KNOWN_APP_PATTERN = (
    r"(?:"
    r"chrome|google\s+chrome|"
    r"edge|microsoft\s+edge|msedge|"
    r"firefox|mozilla|"
    r"cursor|"
    r"vscode|vs\s+code|visual\s+studio\s+code|code|"
    r"discord|"
    r"slack|"
    r"spotify|"
    r"obs|obs\s+studio|"
    r"notepad|"
    r"terminal|windows\s+terminal|wt|"
    r"explorer|file\s+explorer|files|"
    # Sites that go via Chrome with a URL (we synthesise the URL).
    r"youtube|gmail|twitter|x\.com|reddit|github|netflix|"
    r"hacker\s+news|hn"
    # Open-ended: arbitrary single word after the verb is harvested
    # by the AppLauncher's substring search. We pattern-match the
    # named cases explicitly so the dispatch doesn't waste a substring
    # search on every utterance.
    r")"
)

_APP_LAUNCH_PATTERNS = re.compile(
    rf"\b{_APP_LAUNCH_VERB_PATTERN}\s+"
    rf"(?:the\s+|my\s+|a\s+|an\s+|some\s+)?"
    rf"(?P<app>{_KNOWN_APP_PATTERN})\b",
    re.IGNORECASE,
)

# Image search: "show me a picture of X", "show me what X looks like",
# "find an image of X". Distinct from MEDIA_GENERATION (which creates
# images via ComfyUI) -- this just opens Google Images in a new Chrome
# window.
_IMAGE_SEARCH_PATTERNS = re.compile(
    r"\b(?:"
    r"show\s+me\s+(?:an?\s+)?(?:picture|image|photo)\s+of\s+(?P<q1>.+?)(?:\s+on\s+|\s*[.?]|\s*$)|"
    r"show\s+me\s+what\s+(?P<q2>.+?)\s+looks?\s+like(?:\s+on\s+|\s*[.?]|\s*$)|"
    r"find\s+(?:me\s+)?(?:an?\s+)?(?:picture|image|photo)\s+of\s+(?P<q3>.+?)(?:\s+on\s+|\s*[.?]|\s*$)|"
    r"i\s+want\s+to\s+see\s+(?:an?\s+)?(?:picture|image|photo)\s+of\s+(?P<q4>.+?)(?:\s+on\s+|\s*[.?]|\s*$)"
    r")",
    re.IGNORECASE,
)

# URL-only quick-open (so "open youtube.com" routes to APP_LAUNCH for Chrome).
_BARE_URL_OPEN_PATTERNS = re.compile(
    r"\b(?:open|pull\s+up|bring\s+up|launch|go\s+to|visit)\s+"
    r"(?P<dom>(?:[\w-]+\.)+(?:com|net|org|io|app|dev|ai|co|edu|gov|me|tv|gg|xyz|info))"
    r"(?:/\S*)?\b",
    re.IGNORECASE,
)


# Map known site words to URLs.
_SITE_TO_URL = {
    "youtube": "https://www.youtube.com",
    "gmail": "https://mail.google.com",
    "twitter": "https://twitter.com",
    "x.com": "https://x.com",
    "reddit": "https://www.reddit.com",
    "github": "https://github.com",
    "netflix": "https://www.netflix.com",
    "hacker news": "https://news.ycombinator.com",
    "hn": "https://news.ycombinator.com",
}


# 2026-05-14: YouTube deep-link parsing. When the utterance specifies a
# channel / video / search target alongside "open YouTube", build the
# matching results URL so the user lands on what they asked for instead
# of the YouTube home page. Anchors:
#   - "...channel Ordinary Things"      -> search "Ordinary Things channel"
#   - "...the Ordinary Things channel"  -> search "Ordinary Things channel"
#   - "...video <name>"                 -> search "<name>"
#   - "...searching for <name>"         -> search "<name>"
#   - "...play <name> on youtube"       -> search "<name>"  (handled by
#       a separate matcher below because the verb comes before "youtube")
# Each pattern stops at a sentence terminator OR an "on <monitor>"
# phrase (so monitor targeting doesn't bleed into the query).
_YT_STOP_TAIL = (
    r"(?=\s+on\s+(?:my\s+|the\s+)?(?:left|right|main|primary|center|centre|"
    r"middle|first|second|third|fourth|1st|2nd|3rd|4th|monitor|screen|"
    r"display)\b|\s*[.?!]|\s*$)"
)
_YT_CHANNEL_RE = re.compile(
    rf"(?:to|with|for|on|via|under)\s+(?:the\s+)?channel\s+"
    rf"(?P<name>.+?){_YT_STOP_TAIL}",
    re.IGNORECASE,
)
_YT_CHANNEL_TRAILING_RE = re.compile(
    # "the <name> channel" form -- noun comes BEFORE "channel".
    rf"(?:to|with|for|on)\s+(?:the\s+)?(?P<name>.+?)\s+channel{_YT_STOP_TAIL}",
    re.IGNORECASE,
)
_YT_VIDEO_RE = re.compile(
    rf"(?:to|with|for|playing|play|search(?:ing)?\s+for|search(?:ing)?)\s+"
    rf"(?:the\s+)?video\s+(?P<name>.+?){_YT_STOP_TAIL}",
    re.IGNORECASE,
)
_YT_GENERIC_SEARCH_RE = re.compile(
    # "search(ing) (for) X" -- no "video" keyword required. Comes last
    # in priority so the channel/video matchers above win first.
    rf"\bsearch(?:ing)?\s+(?:for\s+)?(?P<name>.+?){_YT_STOP_TAIL}",
    re.IGNORECASE,
)
_YT_PLAY_RE = re.compile(
    # "play X" -- last-resort. We require it to be tail of the utterance
    # so "play Valorant" (gaming-mode trigger) doesn't accidentally
    # match here -- though gaming-mode runs at higher priority anyway.
    rf"\bplay\s+(?P<name>.+?){_YT_STOP_TAIL}",
    re.IGNORECASE,
)


def _build_youtube_url(text: str) -> Optional[str]:
    """Try to parse a channel / video / search target out of the utterance
    and return the matching ``youtube.com/results?search_query=...`` URL.

    Returns ``None`` when no deep-link cue is present. The caller falls
    back to the bare ``https://www.youtube.com`` URL in that case.

    Detection priority:

    1. ``channel <name>`` / ``<name> channel`` -> append " channel" to
       the search query so YouTube biases toward the channel result.
    2. ``video <name>`` -> raw search.
    3. ``search(ing) (for) <name>`` -> raw search.
    4. Bare ``play <name>`` -> raw search (only when "youtube" is in
       the utterance, which the caller already ensured).

    Stops at monitor / sentence boundaries so "on my right monitor"
    doesn't get baked into the search query.
    """
    if not text:
        return None
    m = _YT_CHANNEL_RE.search(text)
    if m:
        name = m.group("name").strip()
        if name:
            return (
                "https://www.youtube.com/results?search_query="
                + _url_quote(f"{name} channel")
            )
    m = _YT_CHANNEL_TRAILING_RE.search(text)
    if m:
        name = m.group("name").strip()
        # Guard against grabbing "youtube" as the channel name when the
        # match window catches the verb-noun before the site word.
        lowered = name.lower()
        if name and lowered not in {"youtube", "the youtube"}:
            return (
                "https://www.youtube.com/results?search_query="
                + _url_quote(f"{name} channel")
            )
    m = _YT_VIDEO_RE.search(text)
    if m:
        name = m.group("name").strip()
        if name:
            return (
                "https://www.youtube.com/results?search_query="
                + _url_quote(name)
            )
    m = _YT_GENERIC_SEARCH_RE.search(text)
    if m:
        name = m.group("name").strip()
        if name:
            return (
                "https://www.youtube.com/results?search_query="
                + _url_quote(name)
            )
    m = _YT_PLAY_RE.search(text)
    if m:
        name = m.group("name").strip()
        if name and name.lower() not in {"youtube"}:
            return (
                "https://www.youtube.com/results?search_query="
                + _url_quote(name)
            )
    return None


# Map matched app phrase to launcher-registry name.
_APP_PHRASE_TO_NAME = {
    "google chrome": "chrome",
    "microsoft edge": "edge", "msedge": "edge",
    "mozilla": "firefox",
    "vs code": "vscode", "visual studio code": "vscode", "code": "vscode",
    "windows terminal": "terminal", "wt": "terminal",
    "file explorer": "explorer", "files": "explorer",
    "obs studio": "obs",
    "x.com": "chrome",  # site, routed via Chrome
    "hacker news": "chrome", "hn": "chrome",
}


def _classify_screen_context(text: str) -> Optional[ScreenContextIntent]:
    """Match SCREEN_CONTEXT_QUERY patterns. Returns the intent or None."""
    if not _SCREEN_CONTEXT_PATTERNS.search(text or ""):
        return None
    mon_idx, _ = _extract_monitor_target(text)
    return ScreenContextIntent(
        question=text.strip(),
        include_vlm=True,
        monitor_index=mon_idx,
        raw_text=text,
    )


def _classify_app_launch(text: str) -> Optional[AppLaunchIntent]:
    """Match APP_LAUNCH patterns (including image-search shortcut).

    Returns the intent or None when no pattern matches.
    """
    if not text:
        return None

    # Image search: "show me a picture of X".
    img = _IMAGE_SEARCH_PATTERNS.search(text)
    if img:
        query = (img.group("q1") or img.group("q2")
                 or img.group("q3") or img.group("q4") or "").strip()
        if query:
            mon_idx, mon_q = _extract_monitor_target(text)
            url = (
                "https://www.google.com/search?tbm=isch&q="
                + _url_quote(query)
            )
            return AppLaunchIntent(
                app_name="chrome",
                url=url,
                monitor_index=mon_idx,
                monitor_query=mon_q,
                fullscreen=False,
                maximize=False,
                raw_text=text,
            )

    # Explicit app match.
    m = _APP_LAUNCH_PATTERNS.search(text)
    if m:
        phrase = m.group("app").lower().strip()
        mon_idx, mon_q = _extract_monitor_target(text)
        is_site = phrase in _SITE_TO_URL
        # Sites (youtube / gmail / github / reddit / etc.) only fire on
        # APP_LAUNCH when an explicit monitor target is present. Without
        # one, defer to the existing BROWSER_AUTOMATION path so the
        # routing baseline is preserved. "open youtube" -> BROWSER, but
        # "open youtube on monitor 2" -> APP_LAUNCH (native Chrome).
        if is_site and mon_idx is None and not mon_q:
            return None
        app_name = _APP_PHRASE_TO_NAME.get(phrase, phrase)
        url = _SITE_TO_URL.get(phrase) or _SITE_TO_URL.get(app_name)
        # 2026-05-14: YouTube channel / video / search parsing. When the
        # site is YouTube and the utterance carries a "channel X" /
        # "video X" / "search for X" / "play X" hint, build the search
        # URL so the user lands on what they asked for instead of the
        # YouTube home page.
        if phrase == "youtube":
            yt_url = _build_youtube_url(text)
            if yt_url:
                url = yt_url
        fullscreen = bool(re.search(r"\bfull[- ]?screen\b", text, re.IGNORECASE))
        maximize = bool(re.search(
            r"\bmaximize|maximised|maximized|full(?:\s+window)?\b",
            text, re.IGNORECASE,
        ))
        if is_site:
            app_name = "chrome"
        return AppLaunchIntent(
            app_name=app_name,
            url=url,
            monitor_index=mon_idx,
            monitor_query=mon_q,
            fullscreen=fullscreen,
            maximize=maximize,
            raw_text=text,
        )

    # Bare URL ("open youtube.com"). Like the named-site case above,
    # only fires when a monitor target is present -- without one,
    # defer to the existing BROWSER_AUTOMATION path.
    bare = _BARE_URL_OPEN_PATTERNS.search(text)
    if bare:
        mon_idx, mon_q = _extract_monitor_target(text)
        if mon_idx is None and not mon_q:
            return None
        domain = bare.group("dom")
        url = f"https://{domain}"
        return AppLaunchIntent(
            app_name="chrome",
            url=url,
            monitor_index=mon_idx,
            monitor_query=mon_q,
            fullscreen=False,
            maximize=False,
            raw_text=text,
        )

    return None


def _url_quote(s: str) -> str:
    """Lazy import urllib.parse.quote_plus for URL-encoding."""
    from urllib.parse import quote_plus
    return quote_plus(s)


def _build_browser_intent(text: str) -> BrowserIntent:
    url_match = _URL_RE.search(text)
    url = url_match.group(0) if url_match else None
    lower = text.lower()
    if "screenshot" in lower:
        action = "screenshot"
    elif "click" in lower:
        action = "click"
    elif "fill" in lower or "submit" in lower:
        action = "fill"
    elif "log in" in lower or "login" in lower or "sign in" in lower:
        action = "login"
    elif url or "navigate" in lower or "open" in lower or "go to" in lower or "pull up" in lower:
        action = "navigate"
    else:
        action = "extract"
    return BrowserIntent(action=action, url=url, raw_text=text)


def _build_media_intent(text: str) -> MediaGenIntent:
    lower = text.lower()
    if any(k in lower for k in ("song", "music", "tune", "track", "melody", "beat", "compose")):
        medium = "audio"
    elif "video" in lower:
        medium = "video"
    else:
        medium = "image"
    return MediaGenIntent(medium=medium, description=text, raw_text=text)


def _build_messaging_intent(text: str) -> MessagingIntent:
    lower = text.lower()
    if "telegram" in lower:
        channel = "telegram"
    elif "signal" in lower:
        channel = "signal"
    elif "slack" in lower:
        channel = "slack"
    elif "email" in lower:
        channel = "email"
    elif "phone" in lower or "text me" in lower:
        channel = "phone"
    else:
        channel = "push"
    return MessagingIntent(channel=channel, body=text, raw_text=text)


_FILE_PATH_RE = re.compile(
    r"(?:file\s+at|in\s+(?:the\s+)?(?:directory|folder))\s+"
    r"['\"]?(?P<path>[A-Za-z]:[\\/][^\s'\"]+|/[^\s'\"]+|[\w./\\:-]+)['\"]?",
    re.IGNORECASE,
)


def _build_file_intent(text: str) -> FileOpIntent:
    lower = text.lower()
    if "delete" in lower or "remove" in lower:
        operation = "delete"
    elif "write" in lower or "save" in lower:
        operation = "write"
    elif "list" in lower or "what's in" in lower or "what is in" in lower or "show me the files" in lower:
        operation = "list"
    else:
        operation = "read"
    m = _FILE_PATH_RE.search(text)
    path = m.group("path") if m else ""
    return FileOpIntent(operation=operation, path=path, raw_text=text)


def _build_shell_intent(text: str) -> ShellOpIntent:
    # Try to lift the actual command from the utterance; fall back to the
    # whole utterance for the dispatcher to figure out.
    m = re.search(r"run\s+(?:the\s+command\s+)?[\"'`]?(?P<cmd>[^\"'`]+?)[\"'`]?\s*$", text, re.IGNORECASE)
    if not m:
        m = re.search(r"execute\s+(?:the\s+)?(?:command|shell)\s+(?P<cmd>.+?)$", text, re.IGNORECASE)
    cmd = (m.group("cmd").strip() if m else text).strip()
    return ShellOpIntent(command=cmd, raw_text=text)


__all__ = ["classify_routing"]
