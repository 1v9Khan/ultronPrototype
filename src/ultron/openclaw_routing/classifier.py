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
    ActiveWindowQueryIntent,
    AppLaunchIntent,
    BrowserIntent,
    DesktopIntent,
    FileOpIntent,
    GamingModeIntent,
    MediaGenIntent,
    MessagingIntent,
    ModelSwitchIntent,
    NavigateToSiteIntent,
    OpenLastSourceIntent,
    RoutingIntent,
    RoutingIntentKind,
    ScreenContextIntent,
    SemanticClickIntent,
    ShellOpIntent,
    WindowCloseConfirmationIntent,
    WindowCloseIntent,
    WindowIntent,
    WindowMoveIntent,
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
# 2026-05-19 Track 4 voice integration: word-named family tokens.
# Whisper transcribes spoken family names cleanly so "switch to
# gemma" / "switch to llama" route through the same MODEL_SWITCH
# intent as the digit-B forms. Optional trailing version suffix
# (e.g., "llama 3.2", "gemma 3 4B") is accepted but not required.
_MODEL_SWITCH_GEMMA_TOKEN = r"(?:gemma(?:\s+3(?:\s+4\s*[Bb])?)?)"
_MODEL_SWITCH_LLAMA_TOKEN = r"(?:llama(?:\s+3(?:[.\s]2)?(?:\s+3\s*[Bb])?)?)"
_MODEL_SWITCH_TOKEN = (
    rf"(?P<model>{_MODEL_SWITCH_4B_TOKEN}|{_MODEL_SWITCH_8B_TOKEN}|"
    rf"{_MODEL_SWITCH_9B_TOKEN}|{_MODEL_SWITCH_GEMMA_TOKEN}|"
    rf"{_MODEL_SWITCH_LLAMA_TOKEN})"
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
    e.g. "4B", "four B", "9 b", "for B", "8 b", "eight B",
    "gemma", "llama 3.2". Returns one of the canonical preset names.

    2026-05-14: "4B" / "8B" route to the Josiefied (abliterated)
    variants because those are the intentionally-maintained presets
    the user is choosing between after the 4B default landed; "9B"
    keeps the plain qwen3.5-9b swap-back path.

    2026-05-19 Track 4 voice integration: "gemma" -> Gemma 3 4B
    abliterated; "llama" / "llama 3.2" -> Llama 3.2 3B abliterated.
    Both presets are paper-only until the GGUFs are downloaded;
    swap_llm_preset's GGUF-presence validation surfaces the
    actionable error when the user tries to swap.
    """
    t = matched_token.lower().replace("-", "").replace(" ", "")
    # Word-named families first -- their tokens contain alphabetic
    # characters that would not match the digit-leading branches.
    if t.startswith("gemma"):
        return "gemma-3-4b-abliterated"
    if t.startswith("llama"):
        return "llama-3.2-3b-abliterated"
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
    r"(?:engage|enter|activate|start|begin|launch)\s+gaming\s+mode|"
    r"(?:switch|jump|drop|flip|swap|go|change)\s+"
    r"(?:to|into|over\s+to|on\s+to)\s+gaming\s+mode|"
    r"turn\s+on\s+gaming\s+mode|"
    r"gaming\s+mode\s+(?:engage|engaged)|"
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
    r"(?:disengage|exit|leave|end|stop|deactivate|cancel)\s+gaming\s+mode|"
    r"(?:switch|jump|drop|flip|swap|go|change)\s+"
    r"(?:out\s+of|off\s+of|away\s+from)\s+gaming\s+mode|"
    r"turn\s+off\s+gaming\s+mode|"
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
    # "make me an image of" / "make an image of" / "make a song about".
    # 2026-05-22 autonomous run found that requiring "me" missed the
    # common phrasing -- the recipient is implicit when speaking to a
    # voice assistant. Both forms now match.
    r"make\s+(?:me\s+)?an?\s+(?:image|picture|illustration|painting|drawing|render|"
    r"song|track|tune|video|clip)\s+(?:of|about|that)|"
    # Generate (a/an) (short/long/...) (image/video/...) — optional adjective
    r"generate\s+an?\s+(?:[\w\-]+\s+){0,2}"
    r"(?:image|picture|illustration|painting|drawing|render|artwork|video|clip|song|audio|music|tune|track)\b|"
    # Create -- audio + visual media; 2026-05-22 added "create a
    # picture/image/video of X" to cover the common phrasing.
    r"create\s+(?:an?\s+)?artwork|"
    r"create\s+(?:an?\s+)?(?:image|picture|illustration|painting|drawing|render|video|clip)\s+(?:of|about|that)|"
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
    """Public entry: time the classification + emit one observation.

    Delegates the real work to :func:`_classify_routing_impl`; this
    wrapper adds the latency measurement + an observation row so the
    eval harness and learning loops can attribute outcomes back to the
    classifier verdict. Observation IO is fail-open; routing never
    blocks or raises on observation failure.
    """
    import time as _time

    start = _time.perf_counter()
    intent = _classify_routing_impl(
        utterance,
        has_active_coding_task=has_active_coding_task,
        has_pending_clarification=has_pending_clarification,
    )
    latency_ms = (_time.perf_counter() - start) * 1000.0
    try:
        from ultron.observations import observe_routing_verdict

        observe_routing_verdict(
            utterance=utterance or "",
            intent_kind=intent.kind.value,
            confidence=float(intent.confidence or 0.0),
            source=intent.source or "",
            reason=intent.reason or "",
            latency_ms=latency_ms,
        )
    except Exception:
        # Observation IO must never break routing.
        pass
    return intent


def _classify_routing_impl(
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
    # 2026-05-22: gaming mode no longer requires openclaw. The
    # engage/disengage callbacks (LLM swap, Kokoro device flip, STT
    # swap, VLM unload) work without an OpenClaw client; only the
    # optional plugin-disable step needs it. Decoupling the
    # classifier gate lets users with openclaw.enabled=false still
    # trigger gaming mode by voice.
    gaming_on = bool(cfg and cfg.gaming_mode.enabled)
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

    # 1.75) SEMANTIC_CLICK (catalog 09 wiring) -- "click the Submit
    #       button", "press the OK button", "activate the File menu".
    #       MUST fire BEFORE WINDOW_AUTOMATION + BROWSER_AUTOMATION
    #       because the verb "click" overlaps with both branches; our
    #       pattern requires a specific element name + optional
    #       control-type noun so generic browser/window verbs don't
    #       false-positive.
    if not has_pending_clarification:
        sc_click = _classify_semantic_click(text)
        if sc_click is not None:
            return RoutingIntent(
                kind=RoutingIntentKind.SEMANTIC_CLICK,
                raw_text=text,
                confidence=0.85,
                source="rule",
                reason="semantic-click pattern matched",
                semantic_click_intent=sc_click,
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

    # 1.91) WINDOW_CLOSE_CONFIRMATION (catalog 09 wiring) -- voice yes/no
    #       reply when the orchestrator has a pending two-phase approval
    #       (e.g. "Close VS Code? It looks like there are unsaved
    #       changes."). The pending-approval state lives on the
    #       orchestrator; we surface the bare yes/no intent and let the
    #       orchestrator decide whether to consume it as a decision OR
    #       fall through (when no approval is pending) to the legacy
    #       CLARIFICATION_RESPONSE path.
    if not has_pending_clarification:
        wcc = _classify_window_close_confirmation(text)
        if wcc is not None:
            return RoutingIntent(
                kind=RoutingIntentKind.WINDOW_CLOSE_CONFIRMATION,
                raw_text=text,
                confidence=0.85,
                source="rule",
                reason="window-close yes/no reply matched",
                window_close_confirmation_intent=wcc,
            )

    # 1.92) ACTIVE_WINDOW_QUERY (catalog 09 wiring) -- "what's my active
    #       window?", "what am I looking at right now?". Lighter than
    #       SCREEN_CONTEXT_QUERY (no UIA walk, no capture, no VLM).
    #       Must fire BEFORE SCREEN_CONTEXT_QUERY's broader patterns OR
    #       AFTER -- here it's just AFTER because SCREEN_CONTEXT_QUERY's
    #       more specific phrasings ("explain", "what's going on")
    #       should still win when they match.
    if not has_pending_clarification:
        awq = _classify_active_window_query(text)
        if awq is not None:
            return RoutingIntent(
                kind=RoutingIntentKind.ACTIVE_WINDOW_QUERY,
                raw_text=text,
                confidence=0.9,
                source="rule",
                reason="active-window query pattern matched",
                active_window_query_intent=awq,
            )

    # 1.95) WINDOW_MOVE (2026-05-14) -- "Put Discord on my right monitor".
    #       Must fire BEFORE APP_LAUNCH because the noun "put" doesn't
    #       overlap with launch verbs but the monitor target would
    #       otherwise be the only signal; routing to APP_LAUNCH here
    #       would spawn a SECOND Discord instead of moving the existing
    #       window.
    if not has_pending_clarification:
        wm = _classify_window_move(text)
        if wm is not None:
            return RoutingIntent(
                kind=RoutingIntentKind.WINDOW_MOVE,
                raw_text=text,
                confidence=0.9,
                source="rule",
                reason="window-move pattern matched",
                window_move_intent=wm,
            )

    # 1.96) WINDOW_CLOSE (2026-05-14) -- "Close my YouTube video on my
    #       right monitor", "close Discord". Must fire BEFORE coding
    #       cancel rules already returned above; the regex denies
    #       "close the task" etc. to avoid hijacking those.
    if not has_pending_clarification:
        wc = _classify_window_close(text)
        if wc is not None:
            return RoutingIntent(
                kind=RoutingIntentKind.WINDOW_CLOSE,
                raw_text=text,
                confidence=0.9,
                source="rule",
                reason="window-close pattern matched",
                window_close_intent=wc,
            )

    # 1.93) NAVIGATE_TO_SITE keyword path (2026-05-22) -- "open the
    #       Netflix website", "show me the BBC site". Must fire
    #       BEFORE OPEN_LAST_SOURCE because both patterns can match
    #       "open the X website" -- but the explicit website keyword
    #       is a stronger signal of navigation than reference.
    if not has_pending_clarification:
        nts = _classify_navigate_to_site(text)
        if nts is not None:
            return RoutingIntent(
                kind=RoutingIntentKind.NAVIGATE_TO_SITE,
                raw_text=text,
                confidence=0.85,
                source="rule",
                reason="navigate-to-site pattern matched",
                navigate_to_site_intent=nts,
            )

    # 1.95) OPEN_LAST_SOURCE (2026-05-22) -- "show me that article",
    #       "open that link", "pull up the source". Must fire BEFORE
    #       APP_LAUNCH because "show me that article" would otherwise
    #       hit the bare "show me X" image-search rule and treat
    #       "article" as the search subject.
    if not has_pending_clarification:
        ols = _classify_open_last_source(text)
        if ols is not None:
            return RoutingIntent(
                kind=RoutingIntentKind.OPEN_LAST_SOURCE,
                raw_text=text,
                confidence=0.9,
                source="rule",
                reason="open-last-source pattern matched",
                open_last_source_intent=ols,
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


# OPEN_LAST_SOURCE (2026-05-22) -- "show me that article", "open that
# link", "pull up the source", "open the article on monitor 2", plus
# disambiguated variants like "show me the second one", "open the NBC
# story", "show me the article about Boeing".
#
# References the source(s) from the most recent search-augmented
# response. Must match BEFORE app_launch's bare "show me <X>" rule,
# which would otherwise treat "article" as an image-search subject.

# Verbs accepted as "open this source for me" actions. NOTE: navigation
# verbs (take me to / go to / navigate to / find me) deliberately
# excluded -- those route to NAVIGATE_TO_SITE which queries SearxNG
# for a NEW site, not the cited source from the previous response.
# 2026-05-22 fix: "Take me to the HBO Max website" was matching here
# because the verb list included navigation phrases.
_OPEN_LAST_SOURCE_VERB = (
    r"(?:show\s+me|open(?:\s+up)?|pull\s+up|bring\s+up|load)"
)

# Source nouns -- what the user is asking to open. Includes "one" as a
# pronoun standin ("show me the first one", "the NBC one") -- safe
# because the bare/before patterns require a "the|that|this"
# determiner, so naked "show me one" doesn't qualify.
_OPEN_LAST_SOURCE_NOUN = (
    r"(?:article|link|page|source|story|result|citation|website|site|url|"
    r"item|entry|piece|report|headline|one)"
)

# Pattern A: bare "show me that article" (no referent, just a demonstrative).
_OPEN_LAST_SOURCE_BARE_RE = re.compile(
    rf"\b{_OPEN_LAST_SOURCE_VERB}\s+(?:that|the|this)\s+{_OPEN_LAST_SOURCE_NOUN}\b",
    re.IGNORECASE,
)

# Pattern B: "show me the {referent} {noun}" -- e.g.
# "show me the NBC story", "open the first article", "pull up the
# second one". Captures the referent phrase between "the" and the noun.
_OPEN_LAST_SOURCE_REF_BEFORE_RE = re.compile(
    rf"""
    \b{_OPEN_LAST_SOURCE_VERB}\s+
    (?:the|that|this)\s+
    (?P<referent>[A-Za-z0-9 .,'\-/]+?)\s+
    {_OPEN_LAST_SOURCE_NOUN}\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Pattern C: "show me the {noun} about {referent}" -- e.g.
# "open the article about Boeing", "show me the story on the
# election", "pull up the one regarding the FDA". Captures the
# referent phrase that follows the about/on/regarding/covering link.
_OPEN_LAST_SOURCE_REF_AFTER_RE = re.compile(
    rf"""
    \b{_OPEN_LAST_SOURCE_VERB}\s+
    (?:the|that|this)\s+
    {_OPEN_LAST_SOURCE_NOUN}\s+
    (?:about|on|regarding|covering|concerning|of)\s+
    (?P<referent>[A-Za-z0-9 .,'\-/]+?)
    (?:\s*(?:\.|,|\?|$)|\s+on\s+(?:my\s+)?(?:monitor|main|primary|left|right|second|third)\b)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Pattern D: "open number 2" / "show me number three" -- pure ordinal.
_OPEN_LAST_SOURCE_NUMBER_RE = re.compile(
    rf"""
    \b{_OPEN_LAST_SOURCE_VERB}\s+
    (?:the\s+)?
    (?:number|source|result|story|article|item|one)\s+
    (?P<ord>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Combined "this is an open-last-source request" matcher -- one of the
# four patterns above. Used as the cheap pre-filter in the classifier.
_OPEN_LAST_SOURCE_RE = re.compile(
    rf"""
    \b{_OPEN_LAST_SOURCE_VERB}\s+
    (?:
        (?:that|the|this)\s+
        (?:[A-Za-z0-9 .,'\-/]+?\s+)?
        {_OPEN_LAST_SOURCE_NOUN}\b
    |
        (?:the\s+)?
        (?:number|source|result|story|article|item|one)\s+
        (?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Words->ordinal-index map for "the first/second/third" style refs.
_ORDINAL_WORDS = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    "sixth": 6, "6th": 6,
    "seventh": 7, "7th": 7,
    "eighth": 8, "8th": 8,
    "ninth": 9, "9th": 9,
    "tenth": 10, "10th": 10,
    "last": -1,
}

# Number-word->int map for "number two" style refs.
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


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
# 2026-05-14 second-pass: accept singular AND plural ("picture(s)",
# "image(s)", "photo(s)") and "some" as a determiner so phrasings like
# "show me pictures of Resident Evil Requiem" (the user's actual
# 2026-05-14 session phrasing) match too.
_IMAGE_NOUN = r"(?:pictures?|images?|photos?)"
_IMAGE_DET = r"(?:an?\s+|some\s+|the\s+)?"
_IMAGE_SEARCH_PATTERNS = re.compile(
    r"\b(?:"
    rf"show\s+me\s+{_IMAGE_DET}{_IMAGE_NOUN}\s+of\s+(?P<q1>.+?)(?:\s+on\s+|\s*[.?]|\s*$)|"
    r"show\s+me\s+what\s+(?P<q2>.+?)\s+looks?\s+like(?:\s+on\s+|\s*[.?]|\s*$)|"
    rf"find\s+(?:me\s+)?{_IMAGE_DET}{_IMAGE_NOUN}\s+of\s+(?P<q3>.+?)(?:\s+on\s+|\s*[.?]|\s*$)|"
    rf"i\s+want\s+to\s+see\s+{_IMAGE_DET}{_IMAGE_NOUN}\s+of\s+(?P<q4>.+?)(?:\s+on\s+|\s*[.?]|\s*$)"
    r")",
    re.IGNORECASE,
)

# 2026-05-14: implicit image-search shortcut -- "show me X on my main
# monitor" / "show me a chicken on my right screen" -- when no "picture
# of" keyword is present but the utterance carries a monitor target
# AND the subject isn't a known app / screen-context noun. The user's
# 2026-05-14 session said "Show me a chicken on my main monitor" and
# the explicit-keyword regex above didn't match, so the utterance
# leaked to the conversational LLM. With a monitor target present,
# the only sensible interpretation is "open an image of X there".
_IMAGE_SEARCH_IMPLICIT_RE = re.compile(
    r"\bshow\s+me\s+(?:an?\s+|the\s+|some\s+)?"
    r"(?P<q>[^.?!]+?)"
    r"\s+on\s+(?:my\s+|the\s+)?"
    r"(?:1st|first|2nd|second|3rd|third|4th|fourth|primary|main|"
    r"left|right|center|centre|middle|top|bottom)"
    r"\s+(?:monitor|screen|display)"
    r"\b",
    re.IGNORECASE,
)

# 2026-05-14 second-pass: even WITHOUT a monitor target, "show me a
# chicken" is reasonably interpreted as an image-search request when
# the subject is concrete (not a system / app / question phrase).
# Defaults to the main monitor at dispatch time. Matches the user's
# 2026-05-14 second-session phrasing ("Show me a chicken." with no
# monitor cue -- previously fell through to conversational and got a
# hallucinated "Displaying visuals via text only..." response).
_IMAGE_SEARCH_BARE_RE = re.compile(
    r"^\s*show\s+me\s+(?:an?\s+|the\s+|some\s+)?"
    r"(?P<q>[^.?!]+?)\s*[.?!]?\s*$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# 2026-05-14 second-pass: WINDOW_MOVE -- "Put / Move / Send <app> to /
# on <position> monitor". Distinct from APP_LAUNCH (which spawns a
# NEW process); WINDOW_MOVE finds an EXISTING open window and
# repositions it. The user's session said "Put Discord on my right
# monitor" expecting their existing Discord window to move; the
# previous classifier didn't have this verb so it leaked to
# conversational and Ultron hallucinated "I cannot perform actions
# on your device."
_WINDOW_MOVE_RE = re.compile(
    r"\b(?:put|move|send|throw|drag|relocate|push|bring|shift)\s+"
    r"(?:my\s+|the\s+|that\s+|this\s+)?"
    r"(?P<window>[^.?!]+?)"
    r"\s+(?:to|on|onto|over\s+to)\s+(?:my\s+|the\s+)?"
    r"(?P<mon>(?:1st|first|2nd|second|3rd|third|4th|fourth|primary|main|"
    r"left|right|center|centre|middle|top|bottom)\s+(?:monitor|screen|display)|"
    r"monitor\s+(?:1|2|3|4|one|two|three|four)|"
    r"screen\s+(?:1|2|3|4|one|two|three|four)|"
    r"display\s+(?:1|2|3|4|one|two|three|four))\b",
    re.IGNORECASE,
)

# WINDOW_CLOSE -- "Close <app>" / "close <app> on <monitor>" /
# "close my <app> tab" / "close the <app> window". Matches existing
# windows only. Distinct from coding CANCEL ("cancel the task") which
# is caught earlier in classify_routing.
_WINDOW_CLOSE_RE = re.compile(
    r"\b(?:close|exit|quit|shut|kill|dismiss)"
    r"(?:\s+(?:out\s+))?\s+"
    r"(?:my\s+|the\s+|that\s+|this\s+)?"
    r"(?P<window>[^.?!]+?)"
    r"(?:\s+(?:tab|window|app|application))?"
    r"(?:\s+on\s+(?:my\s+|the\s+)?(?P<mon>(?:1st|first|2nd|second|3rd|third|"
    r"4th|fourth|primary|main|left|right|center|centre|middle|top|bottom)\s+"
    r"(?:monitor|screen|display)|"
    r"monitor\s+(?:1|2|3|4|one|two|three|four)|"
    r"screen\s+(?:1|2|3|4|one|two|three|four)|"
    r"display\s+(?:1|2|3|4|one|two|three|four)))?"
    r"\s*[.?!]?\s*$",
    re.IGNORECASE,
)
# Subjects we deny on WINDOW_CLOSE to avoid hijacking other intents:
# - coding cancel ("close the task" / "kill the task") -- handled by CODING
# - vague references that don't map to a window ("close the file")
_WINDOW_CLOSE_DENY = frozenset({
    "task", "the task", "my task", "this task", "that task",
    "session", "the session",
    "file", "the file",  # file_op territory
    "everything", "all of it", "all of them",  # too broad
    "ultron", "yourself", "your mouth",  # don't shut Ultron down here
})


def _classify_window_move(text: str) -> Optional[WindowMoveIntent]:
    """Match "put / move / send <window> to <monitor>" pattern."""
    if not text:
        return None
    m = _WINDOW_MOVE_RE.search(text)
    if not m:
        return None
    window = (m.group("window") or "").strip()
    if not window:
        return None
    # Same deny-list as window-close: don't grab "put the task on..." etc.
    if window.lower() in _WINDOW_CLOSE_DENY:
        return None
    # Reuse the monitor target extractor on the full text -- it already
    # knows how to map "main" -> directional vs "1st" -> index.
    mon_idx, mon_q = _extract_monitor_target(text)
    return WindowMoveIntent(
        window_query=window,
        monitor_index=mon_idx,
        monitor_query=mon_q,
        raw_text=text,
    )


def _classify_window_close(text: str) -> Optional[WindowCloseIntent]:
    """Match "close / quit / exit <window>" pattern."""
    if not text:
        return None
    m = _WINDOW_CLOSE_RE.search(text)
    if not m:
        return None
    window = (m.group("window") or "").strip()
    if not window:
        return None
    lowered = window.lower()
    # Deny exact match OR prefix match (catches "yourself down" via
    # the "yourself" deny entry, "everything else" via "everything",
    # etc.).
    if any(
        lowered == bad or lowered.startswith(bad + " ")
        for bad in _WINDOW_CLOSE_DENY
    ):
        return None
    # Strip trailing "tab" / "window" / "app" / "application" -- the
    # regex consumed them only as optional context.
    window_clean = re.sub(
        r"\s+(?:tab|window|app|application)\s*$", "", window,
        flags=re.IGNORECASE,
    ).strip()
    if not window_clean:
        return None
    mon_q = ""
    if m.group("mon"):
        # Reuse extractor for the monitor disambiguator. We store the
        # raw monitor phrase here -- the handler can pass it through
        # find_monitor at dispatch time.
        _, mon_q = _extract_monitor_target(text)
    return WindowCloseIntent(
        window_query=window_clean,
        monitor_query=mon_q,
        raw_text=text,
    )

# Exclusion list: subjects that should NOT trigger implicit image search
# even with a monitor target. These overlap with screen-context /
# system-status / app-launch intents which fire earlier in the
# classifier anyway, but the explicit deny-list keeps the implicit
# branch from competing on borderline phrasings.
_IMAGE_SEARCH_IMPLICIT_DENY = frozenset({
    "what", "what's", "what is",
    "my screen", "the screen",
    "my desktop", "the desktop",
    "my window", "the window",
    "the file", "the contents",
    "the status", "the alerts",
    "youtube", "github", "gmail", "reddit", "netflix", "twitter",
    "x.com", "hacker news", "hn",
    "chrome", "edge", "firefox", "cursor", "vscode", "discord",
    "spotify", "slack", "obs", "notepad", "explorer", "terminal",
    "google chrome", "microsoft edge", "vs code",
})

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


# ---------------------------------------------------------------------------
# Catalog 09 wiring: ACTIVE_WINDOW_QUERY / SEMANTIC_CLICK / WINDOW_CLOSE_CONFIRMATION
# ---------------------------------------------------------------------------


# ACTIVE_WINDOW_QUERY: lightweight "what window am I on?" queries.
# Distinct from SCREEN_CONTEXT_QUERY (which captures + UIA + VLM) --
# this is a pure foreground-title probe, 1-2 ms with no GPU cost.
# Patterns deliberately specific to AVOID hijacking the broader
# screen-context phrasings ("what's on my screen", "explain ...").
_ACTIVE_WINDOW_QUERY_PATTERNS = re.compile(
    r"\b("
    r"what(?:'s|\s+is)?\s+(?:my\s+|the\s+)?(?:active|current|foreground|focused)\s+window"
    r"|what\s+window\s+(?:am\s+i|is)\s+(?:on|using|in|in front of me|focused|active)"
    r"|which\s+window\s+(?:am\s+i|is)\s+(?:on|using|in|focused|active)"
    r"|name\s+(?:my\s+|the\s+)?(?:active|current|foreground)\s+window"
    r"|(?:tell\s+me\s+)?(?:the\s+)?title\s+of\s+(?:my\s+|the\s+)?(?:active|current|foreground)\s+window"
    r")\b",
    re.IGNORECASE,
)


def _classify_active_window_query(text: str) -> Optional[ActiveWindowQueryIntent]:
    """Match ACTIVE_WINDOW_QUERY patterns. Returns the intent or None."""
    if not _ACTIVE_WINDOW_QUERY_PATTERNS.search(text or ""):
        return None
    return ActiveWindowQueryIntent(raw_text=text)


# SEMANTIC_CLICK: "click the X button", "press the OK button",
# "activate the File menu", "tap on Submit".
# Pattern shape: VERB + optional ARTICLE + NAME (greedy, up to 5
# tokens) + optional CONTROL-TYPE NOUN. The trailing control-type
# noun is detected from the matched name by stripping it off
# post-match so the regex can stay greedy while the noun gets
# extracted reliably.
_SEMANTIC_CLICK_VERB = r"(?:click|press|tap(?:\s+on)?|activate|select|hit|push)"
_SEMANTIC_CLICK_CONTROL_NOUNS = (
    "button", "menu", "tab", "item", "link",
    "checkbox", "option", "control",
)
_SEMANTIC_CLICK_ARTICLE = r"(?:the\s+|a\s+|on\s+the\s+|on\s+)?"

# SEMANTIC_CLICK requires an EXPLICIT control-type noun
# ("button" / "menu" / "tab" / "item" / "link" / "checkbox" / "option" /
# "control") so we don't hijack existing BROWSER_AUTOMATION /
# WINDOW_AUTOMATION patterns that share the verb "click" but lack
# the noun ("click on Save" / "activate the cursor window"). When the
# user says "click the X button", they want UIA-level element click;
# when they say "click on Save" or "activate the cursor window", the
# existing browser / window routes own those phrases.
_SEMANTIC_CLICK_CONTROL_NOUN_RE = (
    r"(?:button|menu|tab|item|link|checkbox|option|control)"
)
_SEMANTIC_CLICK_PATTERN = re.compile(
    rf"\b{_SEMANTIC_CLICK_VERB}\s+{_SEMANTIC_CLICK_ARTICLE}"
    r"(?P<name>[\w-]+(?:\s+[\w-]+){0,4})\s+"
    rf"(?P<noun>{_SEMANTIC_CLICK_CONTROL_NOUN_RE})"
    r"(?=\b)",
    re.IGNORECASE,
)


# Words that should be ignored when parsed as an element name (they're
# verb articles / generic pronouns that mean "no specific element").
_SEMANTIC_CLICK_GENERIC_NAMES = frozenset({
    "it", "this", "that", "there", "here", "yes", "no",
    "anywhere", "something", "anything", "everywhere",
})


_SEMANTIC_CLICK_WINDOW_PATTERN = re.compile(
    r"\bin\s+(?:the\s+)?(?P<window>[\w\s\.-]+?)\s+window\b",
    re.IGNORECASE,
)


_SEMANTIC_CLICK_CONTROL_TYPE_MAP = {
    "button": "Button",
    "menu": "MenuItem",
    "tab": "TabItem",
    "item": "ListItem",
    "link": "Hyperlink",
    "checkbox": "CheckBox",
    "option": "RadioButton",
}


def _classify_semantic_click(text: str) -> Optional[SemanticClickIntent]:
    """Match SEMANTIC_CLICK patterns. Returns the intent or None.

    Recognises: ``click the Submit button``, ``press OK``, ``activate
    the File menu``, ``tap on the Sign In link``, ``hit the Cancel
    button``, ``Click Save``, ``press the OK button``.

    Rejects: ``click here``, ``click that``, ``click anywhere`` --
    the element name has to be specific enough to look up via UIA.
    """
    if not text:
        return None
    m = _SEMANTIC_CLICK_PATTERN.search(text)
    if m is None:
        return None
    raw_name = (m.group("name") or "").strip().strip("\"' ")
    if not raw_name:
        return None

    # If the greedy name capture swallowed any control-type tokens
    # (e.g. "Send Message" + "button" matched as the noun group, but
    # if there were inline "button"/"menu" inside the name, prune
    # them). This handles edge cases where the user said something
    # like "click the OK menu button".
    tokens = raw_name.split()
    while (
        len(tokens) > 1
        and tokens[-1].lower() in _SEMANTIC_CLICK_CONTROL_NOUNS
    ):
        tokens.pop()
    cleaned_name = " ".join(tokens).strip()

    if not cleaned_name:
        return None
    # Reject generic referents -- the UIA lookup needs a real name.
    if cleaned_name.lower() in _SEMANTIC_CLICK_GENERIC_NAMES:
        return None

    # Control-type noun is now required by the regex (matched group).
    noun = (m.group("noun") or "").lower()
    control_type = _SEMANTIC_CLICK_CONTROL_TYPE_MAP.get(noun, "")

    # Optional window scope from "in the X window" phrasing.
    window_title = ""
    window_match = _SEMANTIC_CLICK_WINDOW_PATTERN.search(text)
    if window_match is not None:
        window_title = window_match.group("window").strip()
    return SemanticClickIntent(
        element_name=cleaned_name,
        window_title=window_title,
        control_type=control_type,
        raw_text=text,
    )


# WINDOW_CLOSE_CONFIRMATION: bare yes/no replies during a pending
# two-phase approval. The classifier doesn't know whether an approval
# is actually pending -- the orchestrator looks at its own state and
# falls through to the legacy clarification-response path when not.
# Recognised yes tokens: "yes" / "yeah" / "yep" / "yup" / "confirm" /
# "do it" / "go ahead" / "proceed". Negatives: "no" / "nope" /
# "nah" / "cancel" / "stop" / "abort". The classifier normalises
# everything to "yes" or "no".
_WINDOW_CLOSE_YES_RE = re.compile(
    r"^\s*("
    r"yes|yeah|yep|yup|sure|okay|ok|"
    r"confirm(?:\s+(?:it|that))?|"
    r"do\s+it|go\s+ahead|proceed|continue|go\s+for\s+it|"
    r"close\s+(?:it|the\s+window)"
    r")\s*[.!]?\s*$",
    re.IGNORECASE,
)


_WINDOW_CLOSE_NO_RE = re.compile(
    r"^\s*("
    r"no|nope|nah|"
    r"cancel(?:\s+(?:it|that))?|"
    r"stop|abort|don'?t(?:\s+(?:do\s+(?:it|that)|close.*))?|"
    r"never\s*mind|nvm|"
    r"keep\s+(?:it\s+)?open|leave\s+(?:it\s+)?(?:alone|open)"
    r")\s*[.!]?\s*$",
    re.IGNORECASE,
)


def _classify_window_close_confirmation(
    text: str,
) -> Optional[WindowCloseConfirmationIntent]:
    """Match WINDOW_CLOSE_CONFIRMATION patterns (bare yes/no replies).

    Returns ``WindowCloseConfirmationIntent(decision="yes")`` for
    affirmative replies, ``("no")`` for negatives. Returns None when
    the utterance carries any other content (so we don't hijack
    sentences that happen to start with "yes").
    """
    if not text:
        return None
    if _WINDOW_CLOSE_YES_RE.match(text):
        return WindowCloseConfirmationIntent(decision="yes", raw_text=text)
    if _WINDOW_CLOSE_NO_RE.match(text):
        return WindowCloseConfirmationIntent(decision="no", raw_text=text)
    return None


def _extract_open_last_source_referent(text: str) -> tuple[Optional[int], str]:
    """Parse ordinal + referent phrase from an OPEN_LAST_SOURCE utterance.

    Returns a tuple ``(ordinal, referent)``:
      * ``ordinal`` -- 1-based source index when the user said
        "the first one" / "number 2" / "the last one" (-1 for last).
        None when no ordinal phrasing is present.
      * ``referent`` -- noun phrase capturing what they meant ("NBC",
        "the Boeing crash", etc.). Empty when only a bare "that
        article" demonstrative is present.

    The patterns are intentionally permissive -- the orchestrator's
    resolver does the final semantic match.
    """
    if not text:
        return None, ""

    # Pure-ordinal pattern first ("show me number 2", "open source 3").
    m_num = _OPEN_LAST_SOURCE_NUMBER_RE.search(text)
    if m_num:
        tok = m_num.group("ord").lower()
        if tok.isdigit():
            return int(tok), ""
        if tok in _NUMBER_WORDS:
            return _NUMBER_WORDS[tok], ""

    # Topic-after-noun pattern ("the article about Boeing").
    m_after = _OPEN_LAST_SOURCE_REF_AFTER_RE.search(text)
    if m_after:
        ref = (m_after.group("referent") or "").strip().strip(".,?!")
        return None, ref

    # Referent-before-noun pattern ("the NBC story", "the first one").
    m_before = _OPEN_LAST_SOURCE_REF_BEFORE_RE.search(text)
    if m_before:
        ref = (m_before.group("referent") or "").strip().strip(".,?!")
        # Filter out trivial / monitor-leakage referents.
        if ref.lower() in {"that", "the", "this", ""}:
            return None, ""
        # Ordinal word in the referent slot ("first", "second", ...).
        ref_lc = ref.lower().strip()
        if ref_lc in _ORDINAL_WORDS:
            return _ORDINAL_WORDS[ref_lc], ""
        # Sometimes the referent is "the second" / "first NBC" -- try
        # leading-ordinal extraction.
        first_word = ref_lc.split()[0] if ref_lc.split() else ""
        if first_word in _ORDINAL_WORDS:
            ord_val = _ORDINAL_WORDS[first_word]
            rest = " ".join(ref.split()[1:]).strip()
            return ord_val, rest
        return None, ref

    # Bare demonstrative ("show me that article") -- no disambiguator.
    if _OPEN_LAST_SOURCE_BARE_RE.search(text):
        return None, ""

    return None, ""


def _classify_open_last_source(text: str) -> Optional[OpenLastSourceIntent]:
    """Match OPEN_LAST_SOURCE phrasing.

    Returns the intent (carrying any monitor target + ordinal +
    referent phrase) or None when no pattern matches. URL resolution
    happens later in the orchestrator, which has access to the
    last-search payload and last response text.
    """
    if not text:
        return None
    if _OPEN_LAST_SOURCE_RE.search(text) is None:
        return None
    mon_idx, mon_q = _extract_monitor_target(text)
    ordinal, referent = _extract_open_last_source_referent(text)
    return OpenLastSourceIntent(
        monitor_index=mon_idx,
        monitor_query=mon_q,
        ordinal=ordinal,
        referent=referent,
        raw_text=text,
    )


# NAVIGATE_TO_SITE (2026-05-22) -- "take me to HBO Max", "go to YouTube",
# "navigate to Disney Plus", "open the HBO Max website", "find me the
# Netflix site". Distinct from APP_LAUNCH (which handles registered
# apps + URL-quick-open like "open google.com") and OPEN_LAST_SOURCE
# (which reopens a cited URL). NAVIGATE_TO_SITE queries SearxNG and
# picks the best matching domain.
#
# Two phrasings:
#   A) Navigation verb + (the?) + site name
#      "take me to HBO Max", "go to Disney Plus", "navigate to Reuters"
#   B) Any verb + (the?) + site name + (website|site|page|.com)
#      "open the HBO Max website", "show me the Netflix site"

_NAV_TO_SITE_VERB = (
    r"(?:take\s+me\s+to|go\s+to|navigate\s+to|head\s+to|"
    r"find\s+me|bring\s+me\s+to)"
)
_NAV_TO_SITE_KEYWORD = (
    r"(?:website|site|page|homepage|\.com|dot\s+com|\.org|\.net|\.io)"
)

# Pattern A: explicit navigation verb. The site name is everything
# between the determiner and an optional monitor target / sentence end.
_NAVIGATE_TO_SITE_VERB_RE = re.compile(
    rf"""
    \b{_NAV_TO_SITE_VERB}\s+
    (?:the\s+)?
    (?P<site>[A-Za-z0-9 .,'\-]+?)
    (?:\s+{_NAV_TO_SITE_KEYWORD})?
    \s*(?:\.|,|\?|$|\s+on\s+(?:my\s+)?(?:monitor|main|primary|left|right|second|third)\b)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Pattern B: any open-style verb + explicit website keyword.
# Catches "open the HBO Max website" without requiring "take me to".
_NAVIGATE_TO_SITE_KEYWORD_RE = re.compile(
    rf"""
    \b(?:open(?:\s+up)?|pull\s+up|show\s+me|bring\s+up|load)\s+
    (?:the\s+)?
    (?P<site>[A-Za-z0-9 .,'\-]+?)\s+
    {_NAV_TO_SITE_KEYWORD}\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Stopwords that shouldn't be treated as a site name -- catches false
# positives like "go to bed" / "take me to the gym" / "find me a chair".
# Entries are lowercase; the dispatcher strips a leading "the " before
# the lookup, so both "bed" and "the gym" stop "take me to bed" and
# "take me to the gym".
_NAVIGATE_TO_SITE_SITENAME_DENY = {
    # Mundane places
    "bed", "sleep", "gym", "store", "doctor", "lunch", "dinner",
    "breakfast", "work", "school", "church", "home", "office",
    "bathroom", "kitchen", "room", "park", "beach", "mall", "chair",
    "snack", "couch", "car", "garage", "yard", "garden", "shower",
    "bar", "club", "concert", "movies", "hospital", "airport",
    "the gym", "the store", "the doctor", "the office",
    "the bathroom", "the kitchen", "my room", "the park",
    "the beach", "the mall", "the bar", "the hospital",
    # Generic referent words that aren't site names
    "him", "her", "them", "us", "you", "me", "it",
}


def _classify_navigate_to_site(text: str) -> Optional[NavigateToSiteIntent]:
    """Match NAVIGATE_TO_SITE phrasing and extract the site query.

    Returns the intent or None when no pattern matches. The
    orchestrator dispatcher resolves the actual URL via SearxNG +
    domain scoring.
    """
    if not text:
        return None

    site: Optional[str] = None
    for pat in (_NAVIGATE_TO_SITE_VERB_RE, _NAVIGATE_TO_SITE_KEYWORD_RE):
        m = pat.search(text)
        if m:
            site = (m.group("site") or "").strip().strip(".,?!")
            if site:
                break

    if not site:
        return None

    # Guard against everyday "go to bed" / "take me to the gym" etc.
    if site.lower() in _NAVIGATE_TO_SITE_SITENAME_DENY:
        return None
    # Single short common verb / pronoun in the site slot -- noise.
    if len(site) < 2:
        return None

    mon_idx, mon_q = _extract_monitor_target(text)
    return NavigateToSiteIntent(
        site_query=site,
        monitor_index=mon_idx,
        monitor_query=mon_q,
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

    # 2026-05-14: implicit image search -- "show me X on my main monitor".
    # No "picture of" keyword required. The monitor target is the
    # disambiguating signal (utterance is asking for something to be
    # displayed on a specific screen, not a conversational reply).
    implicit_img = _IMAGE_SEARCH_IMPLICIT_RE.search(text)
    if implicit_img:
        query = (implicit_img.group("q") or "").strip()
        # Guard: subject must not be a known app/screen-context noun.
        if query and query.lower() not in _IMAGE_SEARCH_IMPLICIT_DENY:
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

    # 2026-05-14 second-pass: bare "show me X" with no monitor cue and
    # no "picture of" keyword. Same deny-list, same Google-Images URL;
    # falls through to ``_resolve_monitor`` which defaults to "main".
    bare_img = _IMAGE_SEARCH_BARE_RE.match(text.strip())
    if bare_img:
        query = (bare_img.group("q") or "").strip()
        lowered = query.lower()
        # Tighter guards than the with-monitor pattern: at this priority
        # we don't have the monitor signal to disambiguate, so be strict
        # about which subjects qualify. Skip if:
        #   - in the explicit deny list
        #   - starts with a question word (what / who / how / etc.)
        #   - starts with a known app name (those route to APP_LAUNCH below)
        question_starts = ("what", "who", "how", "why", "where",
                           "when", "which", "whose")
        if (
            query
            and lowered not in _IMAGE_SEARCH_IMPLICIT_DENY
            and not any(lowered.startswith(q + " ") or lowered == q
                        for q in question_starts)
            and not any(lowered.startswith(app + " ") or lowered == app
                        for app in _IMAGE_SEARCH_IMPLICIT_DENY)
        ):
            url = (
                "https://www.google.com/search?tbm=isch&q="
                + _url_quote(query)
            )
            return AppLaunchIntent(
                app_name="chrome",
                url=url,
                monitor_index=None,
                monitor_query="",  # _resolve_monitor will default to "main"
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
