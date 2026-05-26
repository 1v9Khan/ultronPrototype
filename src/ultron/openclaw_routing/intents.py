"""Routing intent dataclasses + enum.

Top-level :class:`RoutingIntent` is what the orchestrator sees. It wraps
either a coding intent (existing :class:`ultron.coding.intent.CodingIntent`)
or one of the OpenClaw-bound automation intents below.

Conversational utterances also carry a :class:`RoutingIntent` (kind=
CONVERSATIONAL) so the orchestrator has a uniform handoff for every
classified utterance — no None-checking the dispatch type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class RoutingIntentKind(str, Enum):
    """Top-level routing categories. Includes coding intents (delegated
    to the existing coding pipeline), the new openclaw-bound categories,
    and the explicit CONVERSATIONAL fallback."""

    # Default / fallback
    CONVERSATIONAL = "conversational"

    # Coding (delegated to ultron.coding.intent)
    CODE_TASK = "code_task"
    PROGRESS_QUERY = "progress_query"
    CANCEL = "cancel"
    MID_SESSION_ADJUSTMENT = "mid_session_adjustment"
    CLARIFICATION_RESPONSE = "clarification_response"

    # OpenClaw-bound (stubbed in this phase; filled by integration prompt)
    BROWSER_AUTOMATION = "browser_automation"
    MEDIA_GENERATION = "media_generation"
    MESSAGING = "messaging"
    FILE_OPERATION = "file_operation"
    SHELL_OPERATION = "shell_operation"

    # Hybrid (coding + automation; needs decomposition)
    HYBRID_TASK = "hybrid_task"

    # Self-management — voice-driven runtime model swap. 4B plan addition.
    MODEL_SWITCH = "model_switch"

    # System status — "what alerts did you flag?", "what is Ultron working on?",
    # "any pending alerts?". Resolved Ultron-side (no OpenClaw call) by reading
    # the heartbeat alert log + active coding session list. Phase 13 finish.
    SYSTEM_STATUS = "system_status"

    # Gaming mode (V1-spec gap A1) — anticheat-safe shutdown of the
    # desktop-control / windows-control plugins before launching a
    # Vanguard / EAC-protected game. Voice phrases: "gaming mode",
    # "I'm about to play Valorant", "gaming mode off".
    GAMING_MODE = "gaming_mode"

    # Desktop automation (V1-spec gap C3) — voice routing for the
    # OpenClaw ``desktop-control`` plugin: take a screenshot, list open
    # windows, find a window by title.
    DESKTOP_AUTOMATION = "desktop_automation"

    # Window automation (V1-spec gap C3) — voice routing for the
    # OpenClaw ``windows-control`` plugin: focus / click / type into a
    # specific window via UI Automation.
    WINDOW_AUTOMATION = "window_automation"

    # App launch (desktop automation, 2026-05-12 Phase 8) — "open
    # <X> on monitor <N>", "launch Cursor on my left monitor",
    # "pull up YouTube fullscreen on monitor 2". Routes natively
    # via :mod:`ultron.desktop.launcher` (NOT OpenClaw plugins).
    APP_LAUNCH = "app_launch"

    # Screen-context query (desktop automation, 2026-05-12 Phase 8) --
    # "explain what I'm looking at", "what's on my screen", "help me
    # with what I'm doing". Routes natively via
    # :mod:`ultron.desktop.screen_context` -- captures the foreground
    # monitor + UIA tree text + optional VLM description and injects
    # into the next LLM call as context.
    SCREEN_CONTEXT_QUERY = "screen_context_query"

    # Window move (2026-05-14 second-pass) -- "Put Discord on my right
    # monitor", "move YouTube to the main monitor", "send Cursor to
    # the left screen". Finds an existing window by name and
    # repositions it via :func:`ultron.desktop.placement.move_window_to_monitor`.
    # Distinct from APP_LAUNCH (which spawns a NEW process / window);
    # WINDOW_MOVE operates on already-open windows only.
    WINDOW_MOVE = "window_move"

    # Window close (2026-05-14 second-pass) -- "close Discord", "close
    # my YouTube tab", "close the YouTube window on my right monitor".
    # Finds an existing window by name and sends WM_CLOSE.
    WINDOW_CLOSE = "window_close"

    # Open last source (2026-05-22) -- "show me that article", "open
    # that link", "pull up the source". Opens the URL the LLM cited in
    # the most recent search-augmented response. The orchestrator
    # resolves the cited URL from its own state (last_search_payload +
    # last_response_text) since the classifier is stateless.
    OPEN_LAST_SOURCE = "open_last_source"

    # Navigate to a brand-named site (2026-05-22) -- "take me to HBO
    # Max", "go to YouTube", "open the Disney Plus website". Queries
    # SearxNG for the top results, scores by domain match + cleanliness,
    # and opens the best candidate. Distinct from OPEN_LAST_SOURCE
    # (which opens a cited URL) and APP_LAUNCH (which opens a registered
    # native app or known URL pattern).
    NAVIGATE_TO_SITE = "navigate_to_site"

    # Active window query (2026 catalog 08/09 wiring) -- "what's my
    # active window?", "what am I looking at?", "what window am I
    # using?". Returns the foreground window title via
    # :func:`ultron.desktop.windows.get_active_window_title`. Lighter
    # than SCREEN_CONTEXT_QUERY (no UIA walk, no capture, no VLM); a
    # ~1-2 ms pywin32 probe suitable for quick "where am I?" voice
    # queries that don't need the full screen context.
    ACTIVE_WINDOW_QUERY = "active_window_query"

    # Semantic click (2026 catalog 08/09 wiring) -- "click the Submit
    # button", "activate the File menu", "tap on the OK button", "press
    # the Cancel button". Routes via
    # :func:`ultron.desktop.element_click.click_element_by_name` which
    # walks the foreground UIA tree for the named element and clicks
    # via the gated :class:`InputController` (click-preview VLM +
    # foreground security + Cap-3 explicit-intent + rate limit all
    # apply uniformly).
    SEMANTIC_CLICK = "semantic_click"

    # Window-close confirmation (2026 catalog 08/09 wiring) -- voice
    # yes/no reply during a pending two-phase-approval prompt that the
    # orchestrator opened (e.g. "Close VS Code? It looks like there
    # are unsaved changes. Say yes or no."). The orchestrator routes
    # the spoken yes/no to :meth:`safety.two_phase_approval.ApprovalRegistry.record_decision`
    # via the pending-approval ID it stashed in its own state.
    WINDOW_CLOSE_CONFIRMATION = "window_close_confirmation"


# ---------------------------------------------------------------------------
# Per-category structured intents (for openclaw-bound ones)
# ---------------------------------------------------------------------------


@dataclass
class BrowserIntent:
    """Operations on web pages: navigate, click, fill, screenshot."""
    action: str  # "navigate" | "click" | "fill" | "screenshot" | "extract" | "login"
    url: Optional[str] = None
    target: Optional[str] = None     # selector / element label / form-field name
    value: Optional[str] = None      # for fill / login
    raw_text: str = ""               # original utterance for context


@dataclass
class MediaGenIntent:
    """Generate images, videos, music, etc."""
    medium: str  # "image" | "video" | "audio" | "music"
    description: str
    raw_text: str = ""


@dataclass
class MessagingIntent:
    """Send a message somewhere (Telegram, push, email, etc.)."""
    channel: str  # "telegram" | "push" | "email" | "phone"
    body: str
    recipient: Optional[str] = None
    raw_text: str = ""


@dataclass
class FileOpIntent:
    """Filesystem operations outside the project sandbox."""
    operation: str  # "read" | "write" | "list" | "delete" | "exists"
    path: str
    content: Optional[str] = None    # for write
    raw_text: str = ""


@dataclass
class ShellOpIntent:
    """Execute a shell command (via OpenClaw exec tool)."""
    command: str
    raw_text: str = ""


@dataclass
class ModelSwitchIntent:
    """Voice-driven LLM preset switch.

    The orchestrator's voice controller picks this up, calls
    :meth:`LLMEngine.reload_for_preset` with ``target_preset``, and
    speaks an acknowledgment + completion. The 9B GGUF stays on disk
    for the reverse swap. 4B plan addition — see
    docs/4b_optimization_plan.md.
    """
    target_preset: str  # e.g. "qwen3.5-4b" / "qwen3.5-9b"
    raw_text: str = ""


@dataclass
class GamingModeIntent:
    """V1-gap A1: voice-driven anticheat-safe shutdown of OpenClaw plugins.

    ``action`` mirrors the user's phrasing:
      * ``"engage"`` -- "gaming mode", "I'm about to play Valorant".
      * ``"disengage"`` -- "gaming mode off", "done playing".
      * ``"status"`` -- "is gaming mode on?", "are we in gaming mode?".

    The dispatcher routes to :class:`GamingModeManager` and shapes a
    :class:`DispatchResult` matching the spec's voice phrasing.
    """

    action: str  # "engage" | "disengage" | "status"
    trigger_phrase: str = ""
    raw_text: str = ""


@dataclass
class DesktopIntent:
    """V1-gap C3: voice routing for the OpenClaw ``desktop-control`` plugin.

    ``action`` is the high-level operation:
      * ``"screenshot"`` -- capture full screen or a named window.
      * ``"list_windows"`` -- enumerate currently open windows.
      * ``"find_window"`` -- locate a specific window by title query.
    """

    action: str  # "screenshot" | "list_windows" | "find_window"
    target: Optional[str] = None  # window title pattern (screenshot / find)
    raw_text: str = ""


@dataclass
class WindowIntent:
    """V1-gap C3: voice routing for the OpenClaw ``windows-control`` plugin.

    UI Automation primitives. ``action`` selects the operation:
      * ``"focus"`` -- bring a window to the foreground.
      * ``"click"`` -- click a specific UI element by ref.
      * ``"type"`` -- type text into a specific UI element.
      * ``"find"`` -- look up a window's UIA reference by query.
    """

    action: str  # "focus" | "click" | "type" | "find"
    query: Optional[str] = None  # window title pattern (focus / find)
    ref: Optional[str] = None    # UIA reference (click / type)
    value: Optional[str] = None  # text to type (type only)
    raw_text: str = ""


@dataclass
class AppLaunchIntent:
    """Native app launch (2026-05-12 Phase 8 desktop automation).

    Routes via :class:`ultron.desktop.launcher.AppLauncher`. Distinct
    from BROWSER_AUTOMATION (which goes through the OpenClaw browser
    plugin's isolated Playwright instance): this opens the user's
    REAL application binary with their REAL profile / sessions.

    Attributes:
        app_name: registry name or alias (``chrome``, ``cursor``,
            ``discord``, ``edge``, etc.) or a free-form name resolved
            via the launcher's substring fallback.
        url: optional URL to pass (Chrome opens to this URL via
            ``--new-window``; ignored for non-browser apps).
        monitor_index: target monitor index (None = wherever it lands).
        monitor_query: original monitor phrase from the user
            (``"my second monitor"``, ``"the left screen"``); kept
            for audit + diagnostic narration.
        fullscreen: place the launched window to fill the target monitor.
        maximize: ``SW_MAXIMIZE`` after placement (mutually exclusive
            with fullscreen).
    """

    app_name: str
    url: Optional[str] = None
    monitor_index: Optional[int] = None
    monitor_query: str = ""
    fullscreen: bool = False
    maximize: bool = False
    raw_text: str = ""


@dataclass
class ScreenContextIntent:
    """Screen-context query (2026-05-12 Phase 8 desktop automation).

    Routes via :func:`ultron.desktop.screen_context.build_screen_context`.
    The handler captures the relevant monitor + UIA text + optional
    VLM description and feeds the result back to the LLM as injected
    context so Ultron can answer about what's actually on the user's
    screen.

    Attributes:
        question: the user's actual question ("what does this error
            mean", "what am I looking at", etc.). Defaults to the
            raw_text when not parsed out specifically.
        include_vlm: when True, runs moondream2 on the capture. Slow
            (~5-8 s on CPU) but answers "describe the image" / "what
            does this picture show" cleanly.
        monitor_index: optional explicit monitor target.
    """

    question: str = ""
    include_vlm: bool = True
    monitor_index: Optional[int] = None
    raw_text: str = ""


@dataclass
class WindowMoveIntent:
    """Move an existing window to a target monitor (2026-05-14).

    Attributes:
        window_query: substring matched against window titles (and
            optionally process names) to find the target window.
        monitor_index: explicit target monitor index when the user
            said "monitor 2" / "second monitor".
        monitor_query: directional / named target ("left" / "right" /
            "main" / "primary") to resolve via
            :func:`ultron.desktop.monitors.find_monitor` at dispatch.
        fullscreen: fill the target monitor (as a regular window).
        maximize: SW_MAXIMIZE after placement.
    """

    window_query: str
    monitor_index: Optional[int] = None
    monitor_query: str = ""
    fullscreen: bool = False
    maximize: bool = False
    raw_text: str = ""


@dataclass
class WindowCloseIntent:
    """Close an existing window by name (2026-05-14).

    Attributes:
        window_query: substring matched against window titles.
        monitor_query: optional disambiguator -- when multiple windows
            match the name, restrict to ones on the specified monitor
            (``"right"`` / ``"main"`` / etc.).
    """

    window_query: str
    monitor_query: str = ""
    raw_text: str = ""


@dataclass
class OpenLastSourceIntent:
    """Open one of the URLs cited in the most recent search-augmented response.

    The classifier emits this whenever the user says "show me that
    article" / "open that link" / "pull up the source" / etc. The
    actual URL resolution happens in the orchestrator's dispatcher
    (which has access to ``_last_search_payload`` + the LLM's last
    response text) -- this intent only carries the optional monitor
    target plus disambiguating signals parsed from the utterance.

    Disambiguation: when the user says "the second one", "the NBC
    story", "the one about Boeing", the classifier captures an
    ordinal index, a publication/topic referent phrase, or both. The
    resolver tries ordinal first, then publication-name substring
    match against titles/domains, then semantic similarity against
    source titles via the dense embedder, then falls back to
    matching the cited publication in the LLM's last response.

    Attributes:
        monitor_index: explicit monitor index when the user said
            "open that article on monitor 2".
        monitor_query: directional / named monitor target
            ("left" / "right" / "main" / "primary").
        ordinal: 1-based source index parsed from "the first one" /
            "the second story" / "number 3". None when no ordinal.
        referent: noun phrase capturing what the user meant -- a
            publication name ("NBC"), a topic ("the Boeing crash"),
            or both. Empty string when the user said only a bare
            "that article" with no disambiguator.
        raw_text: the original utterance for logging.
    """

    monitor_index: Optional[int] = None
    monitor_query: str = ""
    ordinal: Optional[int] = None
    referent: str = ""
    raw_text: str = ""


@dataclass
class ActiveWindowQueryIntent:
    """Voice query for the current foreground window's title.

    Resolves via :func:`ultron.desktop.windows.get_active_window_title`
    -- a ~1-2 ms pywin32 probe with no UIA walk, capture, or VLM
    cost. Distinct from SCREEN_CONTEXT_QUERY (which builds a full
    snapshot for the LLM).

    Attributes:
        raw_text: original utterance for logging.
    """

    raw_text: str = ""


@dataclass
class SemanticClickIntent:
    """Voice command to click a UI element by its accessible name.

    Routes via :func:`ultron.desktop.element_click.click_element_by_name`
    which walks the foreground UIA tree for the named element and
    clicks through the gated :class:`InputController` (click-preview
    VLM + foreground security + Cap-3 explicit-intent + rate limit
    all apply uniformly).

    Attributes:
        element_name: the accessible name to look for ("Submit",
            "File", "OK", "Cancel"). Substring matched; exact match
            preferred (catalog 08 T3 stable-sort ranking).
        window_title: optional substring filter to scope the search
            to a specific window when multiple windows expose
            elements with the same name (e.g. multiple "OK" buttons
            across a foreground app + a background dialog).
        control_type: optional UIA control-type filter ("Button",
            "MenuItem"). Defaults to the whole 9-entry
            CLICKABLE_TYPES set when not specified.
        raw_text: original utterance for logging.
    """

    element_name: str
    window_title: str = ""
    control_type: str = ""
    raw_text: str = ""


@dataclass
class WindowCloseConfirmationIntent:
    """Voice yes/no reply during a pending two-phase approval prompt.

    The orchestrator stashes a pending approval ID when it opens a
    Cap-3 approval (e.g. closing a window with unsaved changes); the
    spoken yes/no reply routes to this intent which the orchestrator
    consumes by calling
    :meth:`safety.two_phase_approval.ApprovalRegistry.record_decision`.

    Attributes:
        decision: ``"yes"`` / ``"no"`` parsed from the utterance.
            Other affirmative / negative tokens normalise to one of
            these two at classification time.
        raw_text: original utterance for logging.
    """

    decision: str  # "yes" | "no"
    raw_text: str = ""


@dataclass
class NavigateToSiteIntent:
    """Navigate the user's browser to a brand-named site.

    Resolution flow (handled in the orchestrator dispatcher):

    1. Query SearxNG for ``{site_query} official website`` in the
       general category (top ~10 results).
    2. Score each result by domain match (hostname contains the
       brand name), domain cleanliness (no subdomain, plain .com /
       .net / .org TLD), and source rank.
    3. Open the best candidate via :func:`webbrowser.open` (default
       browser) OR through :func:`ultron.desktop.voice.handle_app_launch`
       with Chrome when a monitor target is set.

    Attributes:
        site_query: the brand / site name parsed from the utterance
            ("HBO Max", "YouTube", "the Disney Plus shop").
        monitor_index: explicit monitor index when the user said
            "take me to HBO Max on monitor 2".
        monitor_query: directional / named monitor target.
        raw_text: original utterance for logging.
    """

    site_query: str
    monitor_index: Optional[int] = None
    monitor_query: str = ""
    raw_text: str = ""


@dataclass
class SystemStatusIntent:
    """A voice query about Ultron's overall state.

    ``focus`` narrows the response: ``"alerts"`` reads from the
    heartbeat alert log only, ``"projects"`` lists active coding
    sessions only, ``"all"`` does both. The classifier picks the
    focus from the utterance ("what alerts" → alerts, "what's
    Ultron working on" → projects, "status report" → all).
    """

    focus: str = "all"  # "alerts" | "projects" | "all"
    raw_text: str = ""


@dataclass
class HybridSubtask:
    """One step in a HYBRID_TASK decomposition."""
    order: int
    type: str  # "coding" | "automation"
    subtype: Optional[str] = None    # e.g. "file_op", "browser", "shell"
    description: str = ""


# Union type the dispatcher accepts.
AutomationIntent = Union[
    BrowserIntent, MediaGenIntent, MessagingIntent, FileOpIntent, ShellOpIntent,
]


# ---------------------------------------------------------------------------
# Top-level RoutingIntent
# ---------------------------------------------------------------------------


@dataclass
class RoutingIntent:
    """Output of the routing classifier.

    Exactly one of ``coding_intent`` or ``automation_intent`` is set when
    ``kind`` is a coding or automation category respectively;
    CONVERSATIONAL leaves both None; HYBRID_TASK populates ``subtasks``.
    """
    kind: RoutingIntentKind
    raw_text: str
    confidence: float = 0.0
    source: str = ""                 # "rule" | "llm_disambiguator" | "default"
    reason: str = ""

    # Wrapped sub-intents (one of these is populated based on kind)
    coding_intent: Optional[Any] = None  # avoid hard import; Any = CodingIntent
    automation_intent: Optional[AutomationIntent] = None
    subtasks: List[HybridSubtask] = field(default_factory=list)
    model_switch_intent: Optional[ModelSwitchIntent] = None  # MODEL_SWITCH only
    system_status_intent: Optional[SystemStatusIntent] = None  # SYSTEM_STATUS only
    gaming_mode_intent: Optional[GamingModeIntent] = None     # GAMING_MODE only (V1-gap A1)
    desktop_intent: Optional[DesktopIntent] = None            # DESKTOP_AUTOMATION only (V1-gap C3)
    window_intent: Optional[WindowIntent] = None              # WINDOW_AUTOMATION only (V1-gap C3)
    app_launch_intent: Optional[AppLaunchIntent] = None       # APP_LAUNCH only (Phase 8)
    screen_context_intent: Optional[ScreenContextIntent] = None  # SCREEN_CONTEXT_QUERY only (Phase 8)
    window_move_intent: Optional[WindowMoveIntent] = None     # WINDOW_MOVE only (2026-05-14)
    window_close_intent: Optional[WindowCloseIntent] = None   # WINDOW_CLOSE only (2026-05-14)
    open_last_source_intent: Optional[OpenLastSourceIntent] = None  # OPEN_LAST_SOURCE only (2026-05-22)
    navigate_to_site_intent: Optional[NavigateToSiteIntent] = None  # NAVIGATE_TO_SITE only (2026-05-22)
    active_window_query_intent: Optional[ActiveWindowQueryIntent] = None  # ACTIVE_WINDOW_QUERY only (2026 catalog 08/09 wiring)
    semantic_click_intent: Optional[SemanticClickIntent] = None  # SEMANTIC_CLICK only (2026 catalog 08/09 wiring)
    window_close_confirmation_intent: Optional[WindowCloseConfirmationIntent] = None  # WINDOW_CLOSE_CONFIRMATION only (2026 catalog 08/09 wiring)

    # Disambiguation: when the rule-based + LLM disambiguator can't decide,
    # the orchestrator asks the user a clarifying question.
    needs_user_clarification: bool = False
    clarification_question: Optional[str] = None


# ---------------------------------------------------------------------------
# Dispatch result
# ---------------------------------------------------------------------------


@dataclass
class DispatchResult:
    """Result of an OpenClawDispatcher call.

    In Phase 5 every call returns ``success=False`` with a stub voice
    message; the OpenClaw integration prompt replaces the stubs with
    real responses where ``success`` reflects the actual operation.
    """
    success: bool
    voice_message: str = ""           # what to speak to the user
    error: Optional[str] = None       # short error label for logs
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Task tracking (for AutomationTaskRunner)
# ---------------------------------------------------------------------------


@dataclass
class TaskInfo:
    task_id: str
    kind: RoutingIntentKind
    description: str
    started_at: float
    completed_at: Optional[float] = None
    success: Optional[bool] = None
    voice_summary: Optional[str] = None
