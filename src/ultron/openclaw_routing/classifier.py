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

from ultron.coding.intent import (
    CodingIntentKind,
    classify as classify_coding,
)
from ultron.openclaw_routing.intents import (
    BrowserIntent,
    FileOpIntent,
    MediaGenIntent,
    MessagingIntent,
    ModelSwitchIntent,
    RoutingIntent,
    RoutingIntentKind,
    ShellOpIntent,
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
_MODEL_SWITCH_9B_TOKEN = r"(?:9\s*[Bb]|nine\s*[Bb]|9\s*-\s*[Bb])"
_MODEL_SWITCH_TOKEN = (
    rf"(?P<model>{_MODEL_SWITCH_4B_TOKEN}|{_MODEL_SWITCH_9B_TOKEN})"
)
_MODEL_SWITCH_PATTERNS = re.compile(
    rf"\b(?:{_MODEL_SWITCH_VERBS})\s+(?:over\s+)?(?:to\s+|on\s+to\s+|onto\s+)?"
    rf"(?:the\s+)?{_MODEL_SWITCH_TOKEN}"
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
    e.g. "4B", "four B", "9 b", "for B". Returns one of the canonical
    preset names: ``"qwen3.5-4b"`` / ``"qwen3.5-9b"``.
    """
    t = matched_token.lower().replace("-", "").replace(" ", "")
    if t.startswith("4") or t.startswith("four") or t.startswith("for"):
        return "qwen3.5-4b"
    if t.startswith("9") or t.startswith("nine"):
        return "qwen3.5-9b"
    # Defensive — regex shouldn't allow other tokens through.
    raise ValueError(f"Unrecognised model token: {matched_token!r}")


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
    r"scroll\s+(?:down|up|to)\s+the"
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
    r"render\s+me\s+(?:an?\s+)?(?:image|scene|picture)|"
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
    """
    text = (utterance or "").strip()
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
