"""Coding orchestration: Kenning drives AI coding agent to do real work.

Phase 6 architecture:
  user voice -> intent detection -> project resolution -> AI coding agent subprocess
  with dynamic cwd -> progress tracking -> voice progress queries -> final
  narration.

The AI coding agent subprocess always runs with the project as cwd (NOT the
sandbox root); this is the spec's hard "dynamic project root" requirement.
For new projects we create a fresh subdirectory under the sandbox first;
for edits we resolve to an existing registered project.
"""

from kenning.coding.bridge import (
    CodingBridge,
    EventKind,
    FileChangeKind,
    TaskEvent,
    TaskHandle,
    TaskRequest,
    TaskResult,
    TaskState,
)
from kenning.coding.direct_bridge import DirectClaudeCodeBridge
from kenning.coding.intent import (
    CodingIntent,
    CodingIntentKind,
    classify as classify_intent,
    derive_project_name,
)
from kenning.coding.narration import NarrationDelta, StatusNarrator
from kenning.coding.projections import (
    AdjustmentContextProjection,
    ClarificationContextProjection,
    CompletionContextProjection,
    CorrectionContextProjection,
    ProjectionResult,
    StatusDeltaProjection,
    project_adjustment_context,
    project_clarification_context,
    project_completion_context,
    project_correction_context,
    project_status_delta,
)
from kenning.coding.projects import (
    Project,
    ProjectRegistry,
    ProjectResolution,
    ProjectResolver,
    ResolutionKind,
    new_sandbox_project,
    slugify_for_path,
)
from kenning.coding.runner import CodingTaskRunner, build_default_bridge
from kenning.coding.session import (
    AdjustmentRecord,
    ClarificationRequest,
    CompletionClaim,
    FileRecord,
    ProjectSession,
    SessionStatus,
    SessionStore,
    StageRecord,
    StateTransitionError,
    TestStatus,
    is_valid_transition,
)
from kenning.coding.templates import (
    PromptTooLargeError,
    SchemaValidationError,
    TemplateError,
    TemplateRenderer,
)
# LAZY (PEP 562): the mcp_server + voice submodules are the heavy ones --
# mcp_server pulls the SSE/server stack and voice pulls the OpenClaw dispatcher
# (-> kenning.openclaw_bridge). Importing the kenning.coding PACKAGE (e.g. the
# GamingModeManager chain does, transitively) must NOT drag those into RAM in a
# lean gaming boot. They resolve on first ACCESS, so the gated coding load-paths
# (non-gaming) still get them via `from kenning.coding import X`.
_LAZY = {
    "KenningMCPServer": "mcp_server",
    "remove_mcp_config": "mcp_server",
    "write_mcp_config": "mcp_server",
    "CapabilityVoiceController": "voice",
    "CodingVoiceController": "voice",
    "VoiceResponse": "voice",
}


def __getattr__(name):
    sub = _LAZY.get(name)
    if sub is not None:
        import importlib
        return getattr(importlib.import_module(f"kenning.coding.{sub}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AdjustmentRecord",
    "ClarificationRequest",
    "CodingBridge",
    "CodingIntent",
    "CodingIntentKind",
    "CapabilityVoiceController",
    "CodingTaskRunner",
    "CodingVoiceController",
    "CompletionClaim",
    "DirectClaudeCodeBridge",
    "EventKind",
    "FileChangeKind",
    "FileRecord",
    "NarrationDelta",
    "Project",
    "ProjectRegistry",
    "ProjectResolution",
    "ProjectResolver",
    "ProjectSession",
    "PromptTooLargeError",
    "ResolutionKind",
    "SchemaValidationError",
    "SessionStatus",
    "SessionStore",
    "StageRecord",
    "StatusNarrator",
    "StateTransitionError",
    "TaskEvent",
    "TaskHandle",
    "TaskRequest",
    "TaskResult",
    "TaskState",
    "TemplateError",
    "TemplateRenderer",
    "TestStatus",
    "KenningMCPServer",
    "VoiceResponse",
    "build_default_bridge",
    "classify_intent",
    "derive_project_name",
    "is_valid_transition",
    "new_sandbox_project",
    "remove_mcp_config",
    "slugify_for_path",
    "write_mcp_config",
]
