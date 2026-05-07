"""Coding orchestration: Ultron drives Claude Code to do real work.

Phase 6 architecture:
  user voice -> intent detection -> project resolution -> Claude Code subprocess
  with dynamic cwd -> progress tracking -> voice progress queries -> final
  narration.

The Claude Code subprocess always runs with the project as cwd (NOT the
sandbox root); this is the spec's hard "dynamic project root" requirement.
For new projects we create a fresh subdirectory under the sandbox first;
for edits we resolve to an existing registered project.
"""

from ultron.coding.bridge import (
    CodingBridge,
    EventKind,
    FileChangeKind,
    TaskEvent,
    TaskHandle,
    TaskRequest,
    TaskResult,
    TaskState,
)
from ultron.coding.direct_bridge import DirectClaudeCodeBridge
from ultron.coding.intent import (
    CodingIntent,
    CodingIntentKind,
    classify as classify_intent,
    derive_project_name,
)
from ultron.coding.mcp_server import (
    UltronMCPServer,
    remove_mcp_config,
    write_mcp_config,
)
from ultron.coding.narration import NarrationDelta, StatusNarrator
from ultron.coding.projects import (
    Project,
    ProjectRegistry,
    ProjectResolution,
    ProjectResolver,
    ResolutionKind,
    new_sandbox_project,
    slugify_for_path,
)
from ultron.coding.runner import CodingTaskRunner, build_default_bridge
from ultron.coding.session import (
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
from ultron.coding.templates import (
    PromptTooLargeError,
    SchemaValidationError,
    TemplateError,
    TemplateRenderer,
)
from ultron.coding.voice import CodingVoiceController, VoiceResponse

__all__ = [
    "AdjustmentRecord",
    "ClarificationRequest",
    "CodingBridge",
    "CodingIntent",
    "CodingIntentKind",
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
    "UltronMCPServer",
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
