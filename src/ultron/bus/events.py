"""Canonical event catalog.

Every event ultron publishes lives here as a named module constant.
Subsystems import the constant they need and pass it to
:func:`~ultron.bus.publish` or :func:`~ultron.bus.subscribe`. New
events are added here, not scattered through call sites, so the
full event taxonomy is greppable in one file (mirrors opencode's
per-module ``BusEvent.define`` pattern but consolidated for
discoverability in a smaller codebase).

Schema entries use a Python type. ``str``, ``int``, ``float``,
``bool``, ``dict``, ``list``, ``tuple`` are the common ones.
Validation is best-effort + non-fatal -- producer bugs are
loggable but never crashy.
"""

from __future__ import annotations

from ultron.bus.event import BusEvent

# ---------------------------------------------------------------------------
# Voice loop lifecycle
# ---------------------------------------------------------------------------

TurnStartedEvent = BusEvent.define(
    type="turn.started",
    schema={
        "turn_id": int,
        "session_id": str,
        "trigger": str,  # "wake_word" | "follow_up" | "barge_in"
    },
    description="A new user-utterance turn is beginning capture.",
)

TurnCompletedEvent = BusEvent.define(
    type="turn.completed",
    schema={
        "turn_id": int,
        "session_id": str,
        "duration_ms": int,
        "via": str,  # "respond" | "capability" | "open_last_source" | ...
    },
    description="A user-utterance turn finished end-to-end (TTS done).",
)

STTTranscribedEvent = BusEvent.define(
    type="stt.transcribed",
    schema={
        "turn_id": int,
        "text": str,
        "chars": int,
        "elapsed_ms": int,
        "engine": str,  # "moonshine" | "parakeet" | "whisper"
    },
    description="STT produced a finalized transcript for the captured audio.",
)

# ---------------------------------------------------------------------------
# Routing + gating
# ---------------------------------------------------------------------------

RoutingClassifiedEvent = BusEvent.define(
    type="routing.classified",
    schema={
        "turn_id": int,
        "kind": str,  # RoutingIntentKind value
        "confidence": float,
        "source": str,  # "rule" | "default" | "llm_disambiguator"
        "reason": str,
    },
    description="Routing classifier emitted a verdict for the user text.",
)

GateVerdictEvent = BusEvent.define(
    type="gate.verdict",
    schema={
        "turn_id": int,
        "decision": str,  # "SEARCH" | "NO_SEARCH" | "UNCERTAIN"
        "confidence": str,  # "high" | "medium" | "low"
        "source": str,  # "rule" | "preflight" | "intent_recognizer" | ...
        "reason": str,
        "elapsed_ms": int,
    },
    description="Web-search gate emitted a verdict.",
)

# ---------------------------------------------------------------------------
# Memory layer
# ---------------------------------------------------------------------------

MemoryRetrievedEvent = BusEvent.define(
    type="memory.retrieved",
    schema={
        "turn_id": int,
        "query_chars": int,
        "k": int,
        "returned": int,
        "elapsed_ms": int,
    },
    description="ConversationMemory.retrieve() returned a result set.",
)

# ---------------------------------------------------------------------------
# LLM streaming
# ---------------------------------------------------------------------------

LLMStreamTokenEvent = BusEvent.define(
    type="llm.stream.token",
    schema={
        "turn_id": int,
        "token": str,
    },
    description="A single token streamed from the LLM (high-volume).",
)

LLMStreamCompleteEvent = BusEvent.define(
    type="llm.stream.complete",
    schema={
        "turn_id": int,
        "chars": int,
        "elapsed_ms": int,
        "ttft_ms": int,
    },
    description="The LLM stream finished for this turn.",
)

# ---------------------------------------------------------------------------
# TTS playback
# ---------------------------------------------------------------------------

TTSPlayedEvent = BusEvent.define(
    type="tts.played",
    schema={
        "turn_id": int,
        "chars": int,
        "duration_ms": int,
    },
    description="A TTS clip finished playback.",
)

# ---------------------------------------------------------------------------
# Coding + supervisor
# ---------------------------------------------------------------------------

CodingFileChangedEvent = BusEvent.define(
    type="coding.file_changed",
    schema={
        "task_id": str,
        "project_name": str,
        "file_path": str,
        "kind": str,  # "created" | "modified" | "deleted"
    },
    description="A file in an active project was created/modified/deleted by Claude.",
)

ProjectIndexedEvent = BusEvent.define(
    type="project.indexed",
    schema={
        "project_id": str,
        "project_name": str,
        "digest_chars": int,
    },
    description="A project digest was upserted into the project index.",
)

ProjectDigestGeneratedEvent = BusEvent.define(
    type="project.digest_generated",
    schema={
        "project_id": str,
        "project_name": str,
        "task_id": str,
        "elapsed_ms": int,
        "fallback": bool,
    },
    description="A project digest finished generating (success or fallback template).",
)

SupervisorDecidedEvent = BusEvent.define(
    type="supervisor.decided",
    schema={
        "turn_id": int,
        "action": str,  # "new" | "edit" | "resume" | "clarify"
        "target_project": str,
        "confidence": float,
        "reasoning": str,
        "candidates": list,  # serialized for offline tuning
    },
    description="Project supervisor emitted a routing decision.",
)

# ---------------------------------------------------------------------------
# Safety + capability
# ---------------------------------------------------------------------------

SafetyViolatedEvent = BusEvent.define(
    type="safety.violated",
    schema={
        "rule_id": str,
        "category": str,
        "verdict": str,  # "BLOCK_HARD" | "BLOCK_SOFT" | "NEEDS_EXPLICIT_INTENT"
        "tool_name": str,
        "reason": str,
    },
    description="Safety validator blocked or flagged a tool call.",
)

GamingEngagedEvent = BusEvent.define(
    type="gaming.engaged",
    schema={
        "trigger_phrase": str,
        "vram_freed_mb": int,
    },
    description="Gaming mode engaged; VRAM reclaim chain completed.",
)

GamingDisengagedEvent = BusEvent.define(
    type="gaming.disengaged",
    schema={
        "trigger_phrase": str,
    },
    description="Gaming mode disengaged; runtime restored.",
)

VRAMReclaimedEvent = BusEvent.define(
    type="vram.reclaimed",
    schema={
        "freed_mb": int,
        "source": str,  # "gaming_mode" | "vlm_unload" | "stt_swap" | ...
    },
    description="VRAM was freed by an explicit reclaim action.",
)

# ---------------------------------------------------------------------------
# Catalog (every event listed here for introspection / docs)
# ---------------------------------------------------------------------------

BUS_EVENT_CATALOG: list = [
    TurnStartedEvent,
    TurnCompletedEvent,
    STTTranscribedEvent,
    RoutingClassifiedEvent,
    GateVerdictEvent,
    MemoryRetrievedEvent,
    LLMStreamTokenEvent,
    LLMStreamCompleteEvent,
    TTSPlayedEvent,
    CodingFileChangedEvent,
    ProjectIndexedEvent,
    ProjectDigestGeneratedEvent,
    SupervisorDecidedEvent,
    SafetyViolatedEvent,
    GamingEngagedEvent,
    GamingDisengagedEvent,
    VRAMReclaimedEvent,
]
