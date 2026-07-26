"""Phase 4 — typed exception hierarchy.

Every external dependency has a typed exception so callers can branch on
"this Brave call timed out" vs "this Brave call returned malformed JSON"
without parsing exception messages.

Hierarchy:

    KenningError                       (root — all Kenning-raised errors)
    ├── DependencyUnavailableError    (external service unreachable / failing)
    │   ├── BraveAPIError
    │   ├── JinaReaderError
    │   ├── QdrantUnavailableError
    │   ├── AnthropicAPIError
    │   ├── OllamaUnavailableError    (declared for completeness;
    │   │                              voice pipeline doesn't use Ollama —
    │   │                              see feedback_llm_runtime_decision.md)
    │   └── OpenClawGatewayError      (anticipated; surfaces in Part 5+)
    ├── ClaudeCodeError               (subprocess failures)
    ├── AudioPipelineError            (Whisper / Piper / RVC / wake-word)
    │   ├── WhisperTranscriptionError
    │   ├── PiperSynthesisError
    │   ├── RVCConversionError
    │   ├── WakeWordModelError
    │   └── AddressingClassifierError
    ├── MCPServerError
    ├── ConfigurationError            (invalid config.yaml at startup)
    └── FilesystemError

Errors carry an optional ``context`` dict so the structured error log
(``logs/errors.jsonl``) records useful diagnostic detail without having
to parse the exception message.

Errors also carry an optional ``recovery`` string describing the
degraded path the caller intends to take. The error log records both;
that pair makes operational triage substantially easier.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class KenningError(Exception):
    """Base for every Kenning-raised exception.

    Attributes:
        message: short human-readable summary
        context: optional dict of diagnostic key-value pairs
        recovery: optional one-line description of the fallback path the
            caller is taking; populated by the wrapper, read by the
            error log writer.
    """

    def __init__(
        self,
        message: str = "",
        *,
        context: Optional[Dict[str, Any]] = None,
        recovery: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context: Dict[str, Any] = dict(context or {})
        self.recovery: Optional[str] = recovery

    def with_recovery(self, recovery: str) -> "KenningError":
        """Annotate this error with a recovery description (the fallback
        path being taken). Returns self for chaining."""
        self.recovery = recovery
        return self

    def with_context(self, **kwargs: Any) -> "KenningError":
        self.context.update(kwargs)
        return self

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "error_type": type(self).__name__,
            "message": self.message,
            "context": self.context,
            "recovery": self.recovery,
        }


# ---------------------------------------------------------------------------
# External-service unavailability
# ---------------------------------------------------------------------------


class DependencyUnavailableError(KenningError):
    """Generic 'the thing we needed wasn't reachable / functional'.

    Subclassed per dependency so callers can branch on which one failed.
    """


class BraveAPIError(DependencyUnavailableError):
    """Brave Search API failure (timeout, 5xx, 4xx, malformed JSON,
    rate-limited)."""


class JinaReaderError(DependencyUnavailableError):
    """Jina Reader full-text fetch failure."""


class QdrantUnavailableError(DependencyUnavailableError):
    """Embedded Qdrant is unreachable / collection missing / corrupt."""


class AnthropicAPIError(DependencyUnavailableError):
    """Anthropic Claude API failure (used during AI coding agent sessions)."""


class OllamaUnavailableError(DependencyUnavailableError):
    """Reserved for completeness. Voice pipeline does NOT use Ollama;
    see feedback_llm_runtime_decision.md. Raised only if a future
    integration uses Ollama and the daemon is down."""


class OpenClawGatewayError(DependencyUnavailableError):
    """OpenClaw Gateway HTTP unreachable / unhealthy. Generic transport
    failure for the bridge — connection refused, timeout, 5xx, malformed
    response. The bridge logs and continues; voice pipeline is unaffected
    (``openclaw.fail_open: true``)."""


class OpenClawAuthError(OpenClawGatewayError):
    """Gateway rejected our credentials. 401/403 from the Gateway HTTP
    API, typically because the auth token in ``~/.openclaw/openclaw.json``
    has rotated. The bridge stops trying until the user reauths; voice
    pipeline keeps working."""


class OpenClawToolError(KenningError):
    """An OpenClaw tool invocation failed at the application layer (the
    Gateway responded, but the tool returned an error result). Distinct
    from transport failures (``OpenClawGatewayError``) — this means the
    Gateway is healthy but the requested action couldn't complete."""


# ---------------------------------------------------------------------------
# Subprocess
# ---------------------------------------------------------------------------


class ClaudeCodeError(KenningError):
    """AI coding agent subprocess failures: nonzero exit, malformed
    stream-json, hang/timeout, killed by external signal."""


# ---------------------------------------------------------------------------
# Local audio pipeline
# ---------------------------------------------------------------------------


class AudioPipelineError(KenningError):
    """Anything in the audio path failed in a way the caller should
    surface to the user."""


class WhisperTranscriptionError(AudioPipelineError):
    """Whisper failed to transcribe (model error, malformed audio
    array, runtime exception)."""


class PiperSynthesisError(AudioPipelineError):
    """Piper TTS failed to synthesize. Critical: voice path can't
    speak. Caller falls back to printing to terminal."""


class RVCConversionError(AudioPipelineError):
    """RVC conversion failed (CUDA OOM, model corruption, etc.).
    Caller falls back to neutral Piper."""


class WakeWordModelError(AudioPipelineError):
    """openWakeWord model failed to load or evaluate."""


class AddressingClassifierError(AudioPipelineError):
    """Addressing classifier (rules or zero-shot) failed."""


# ---------------------------------------------------------------------------
# Other
# ---------------------------------------------------------------------------


class MCPServerError(KenningError):
    """MCP server crashed / unreachable / protocol error."""


class ConfigurationError(KenningError):
    """Invalid config.yaml or schema mismatch. Raised loudly at startup;
    we never proceed with a partial config."""


class FilesystemError(KenningError):
    """Audit log write failed, sandbox creation failed, project
    registry I/O failed, etc."""


__all__ = [
    "KenningError",
    "DependencyUnavailableError",
    "BraveAPIError",
    "JinaReaderError",
    "QdrantUnavailableError",
    "AnthropicAPIError",
    "OllamaUnavailableError",
    "OpenClawGatewayError",
    "OpenClawAuthError",
    "OpenClawToolError",
    "ClaudeCodeError",
    "AudioPipelineError",
    "WhisperTranscriptionError",
    "PiperSynthesisError",
    "RVCConversionError",
    "WakeWordModelError",
    "AddressingClassifierError",
    "MCPServerError",
    "ConfigurationError",
    "FilesystemError",
]
