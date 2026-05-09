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
