"""Phase 3 unified configuration loader.

Single source of truth for every tunable parameter in the system. The
canonical values live in ``config.yaml`` at the project root; this
module loads them, validates against a pydantic schema, and exposes a
typed singleton via :func:`get_config`.

Usage::

    from ultron.config import get_config

    cfg = get_config()
    if cfg.web_search.enabled:
        client = BraveSearchClient(timeout=cfg.web_search.brave.timeout_seconds)

Compatibility shim: ``config/settings.py`` re-exports every legacy
``settings.X`` constant by reading from this loader. Existing code that
does ``from config import settings; settings.LLM_TEMPERATURE`` keeps
working unchanged. Subsystems migrate to ``get_config()`` over time;
once a subsystem stops referencing ``settings.X``, those names are
removed from the shim.

Path conventions:
- ``PROJECT_ROOT`` is computed from this module's location (parent of
  ``src/``). Never tunable.
- All path values in ``config.yaml`` are interpreted relative to
  ``PROJECT_ROOT`` unless absolute. :func:`resolve_path` does the
  conversion.

Env-var substitution: ``${VAR_NAME}`` in any string value is replaced
with ``os.environ[VAR_NAME]`` (or ``""`` if unset). Used for secrets
like the Brave API key — declare ``brave_api_key_env: "ULTRON_BRAVE_API_KEY"``
to keep keys out of the file.

Hot-reload: :func:`reload_config` clears the singleton and reloads from
disk. Some sections require process restart (LLM model_path, Qdrant
data_dir, audio sample_rate); the loader doesn't track which — caller's
responsibility per the docs.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# This module lives at src/ultron/config.py; project root is two parents up.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"
MODELS_DIR: Path = PROJECT_ROOT / "models"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

LOGS_DIR.mkdir(parents=True, exist_ok=True)


def resolve_path(value: str | Path) -> Path:
    """Resolve a config-file path relative to PROJECT_ROOT.

    Absolute paths pass through unchanged. Relative paths are resolved
    against PROJECT_ROOT. Always returns an absolute Path.
    """
    p = Path(value)
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


# ---------------------------------------------------------------------------
# Schema (one BaseModel per subsystem)
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    """All sub-models share strict 'no unknown keys' for typo detection."""
    model_config = ConfigDict(extra="forbid")


class AudioConfig(_Strict):
    sample_rate: int = 16000
    channels: int = 1
    blocksize: int = 512
    dtype: str = "float32"
    input_device: Optional[str] = None        # env: ULTRON_AUDIO_DEVICE
    output_device: Optional[str] = None       # env: ULTRON_AUDIO_OUTPUT_DEVICE
    barge_in_enabled: bool = True
    barge_in_grace_seconds: float = 0.5
    ring_buffer_seconds: float = 0.5


class VADConfig(_Strict):
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_speech_duration_ms: int = Field(default=250, ge=0)
    min_silence_duration_ms: int = Field(default=500, ge=0)
    window_samples: int = 512


class WakeWordConfig(_Strict):
    name: str = "ultron"
    model_path: str = "models/openwakeword/ultron.onnx"
    fallback_model: str = "hey_jarvis"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    cooldown_seconds: float = 1.5


class STTConfig(_Strict):
    model: str = "small.en"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = Field(default=5, ge=1)
    temperature: float = 0.0
    condition_on_previous_text: bool = False
    vad_filter: bool = False


class LLMServerConfig(_Strict):
    """OpenAI-compat HTTP server endpoint for the shared local Qwen.

    Used when ``llm.runtime == "http_server"``. Mirrors the
    llama-cpp-python ``llama_cpp.server`` we run via
    ``scripts/start_llamacpp_server.py``. Both Ultron's voice
    pipeline (when migrated) and OpenClaw point at this endpoint —
    one model load, one VRAM allocation.

    Defaults match the launcher and the OpenClaw provider config.
    """

    base_url: str = "http://127.0.0.1:8765/v1"
    api_key: str = "local-ultron"
    model_alias: str = "qwen3.5-9b-local"
    request_timeout_s: float = 120.0
    connect_timeout_s: float = 5.0


class LLMConfig(_Strict):
    # Pinned to llama_cpp per feedback_llm_runtime_decision.md (2026-05-08).
    provider: Literal["llama_cpp"] = "llama_cpp"
    # Where the model actually runs:
    #   "in_process"  — load via llama-cpp-python in this Python process
    #                   (current default; what the voice pipeline uses today).
    #   "http_server" — talk to a separately-run llama-cpp-server over HTTP
    #                   (OpenAI-compat /v1/chat/completions). Use this once
    #                   the server is running and verified.
    # The HTTP path is opt-in: existing behaviour stays unchanged unless
    # this flag is flipped. Phase 0 of the OpenClaw integration verified
    # the HTTP server exists; Phase 2 / Item 6 wires the voice path
    # through it.
    runtime: Literal["in_process", "http_server"] = "in_process"
    model_path: str = "models/Qwen3.5-9B-Q4_K_M.gguf"
    n_ctx: int = Field(default=8192, ge=1)
    gpu_layers: int = -1
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_max_tokens: int = Field(default=512, ge=1)
    default_repeat_penalty: float = 1.1
    history_turns: int = Field(default=6, ge=0)
    flash_attn: bool = True
    kv_cache_type: int = 8                    # 8=q8_0, 1=F16
    system_prompt: str = ""
    server: LLMServerConfig = Field(default_factory=LLMServerConfig)


class EmbeddingsConfig(_Strict):
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    dense_dim: int = 384


class QdrantCollections(_Strict):
    conversations: str = "conversations"
    facts: str = "facts"
    web_results: str = "web_results"


class QdrantConfig(_Strict):
    data_dir: str = "data/qdrant"             # NOT ./qdrant_data/, per existing layout
    collections: QdrantCollections = Field(default_factory=QdrantCollections)


class MemoryConfig(_Strict):
    enabled: bool = True
    jsonl_legacy_path: str = "data/memory.jsonl"
    recent_turns: int = Field(default=20, ge=0)
    rag_top_k: int = Field(default=5, ge=0)
    rag_exclude_recent: int = Field(default=20, ge=0)
    facts_top_k: int = Field(default=3, ge=0)
    write_queue_maxsize: int = Field(default=256, ge=1)


class BraveConfig(_Strict):
    endpoint: str = "https://api.search.brave.com/res/v1/web/search"
    count: int = Field(default=5, ge=1)
    timeout_seconds: float = 8.0
    rate_limit_seconds: float = 2.0


class JinaConfig(_Strict):
    endpoint: str = "https://r.jina.ai/"
    timeout_seconds: float = 15.0
    max_fetch: int = Field(default=3, ge=0)
    max_bytes: int = Field(default=200_000, ge=0)


class WebCacheConfig(_Strict):
    ttl_volatile_seconds: int = Field(default=86400, ge=0)
    ttl_stable_seconds: int = Field(default=2_592_000, ge=0)


class WebSearchConfig(_Strict):
    enabled: bool = True
    brave_api_key_env: str = "ULTRON_BRAVE_API_KEY"
    brave: BraveConfig = Field(default_factory=BraveConfig)
    jina: JinaConfig = Field(default_factory=JinaConfig)
    cache: WebCacheConfig = Field(default_factory=WebCacheConfig)


class AddressingConfig(_Strict):
    follow_up_enabled: bool = True
    # CONFIRMED 30s, NOT the Foundation prompt's 10s, per feedback_ultron_extension.md
    warm_mode_duration_seconds: float = 30.0
    default_uncertain_to_not_addressed: bool = True
    rule_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    zero_shot_model: str = "google/flan-t5-small"
    load_eagerly: bool = True
    log_path: str = "logs/addressing.jsonl"


class CodingMCPConfig(_Strict):
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=19761, ge=1, le=65535)
    sse_path: str = "/sse"
    log_path: str = "logs/mcp_calls.jsonl"
    server_name: str = "ultron_coding"
    clarification_timeout_seconds: int = Field(default=600, ge=0)


class CodingVerificationConfig(_Strict):
    smoke_timeout_seconds: int = Field(default=5, ge=0)
    test_timeout_seconds: int = Field(default=120, ge=0)
    lint_timeout_seconds: int = Field(default=30, ge=0)


class CodingConfig(_Strict):
    enabled: bool = True
    bridge: str = "direct"
    mcp: CodingMCPConfig = Field(default_factory=CodingMCPConfig)
    template_dir: str = "prompts/coding"
    prompt_token_budget: int = Field(default=4000, ge=1)
    prompt_chars_per_token: int = Field(default=4, ge=1)
    default_model: str = "haiku"
    escalation_model: str = "sonnet"
    escalation_threshold_default: int = Field(default=3, ge=1)
    escalation_threshold_escalation: int = Field(default=2, ge=1)
    verification: CodingVerificationConfig = Field(default_factory=CodingVerificationConfig)
    session_audit_dir: str = "logs/sessions"
    token_budget_per_session: int = Field(default=100_000, ge=1)
    token_warning_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    progress_timeout_seconds: int = Field(default=300, ge=0)
    test_sandbox_path: str = "tests/coding/sandbox"
    # Defaulting to a Windows-typical claude.cmd path; users on other OSes
    # override via config.yaml or the ULTRON_CLAUDE_CLI env var.
    claude_cli: str = "${USERPROFILE}/AppData/Roaming/npm/claude.cmd"
    claude_model: str = "haiku"
    sandbox_root: str = "data/sandbox"
    project_registry_path: str = "data/projects.json"
    audit_log_path: str = "logs/coding_tasks.jsonl"
    task_timeout_seconds: int = Field(default=1800, ge=0)
    skip_permissions: bool = True


class ProjectionsBudgets(_Strict):
    clarification_context: int = Field(default=1500, ge=1)
    status_delta: int = Field(default=600, ge=1)
    adjustment_context: int = Field(default=1200, ge=1)
    correction_context: int = Field(default=1500, ge=1)
    completion_context: int = Field(default=800, ge=1)


class ProjectionsConfig(_Strict):
    tokenizer: str = "tiktoken_cl100k_base"
    budgets: ProjectionsBudgets = Field(default_factory=ProjectionsBudgets)
    truncation_warning_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    log_truncations: bool = True


class RVCConfig(_Strict):
    enabled: bool = True
    model_dir: str = "ultron_james_spader_mcu_6941"
    model_path: str = "ultron_james_spader_mcu_6941/Ultron.pth"
    index_path: str = "ultron_james_spader_mcu_6941/added_IVF301_Flat_nprobe_1_Ultron_v2.index"
    support_dir: str = "models/rvc"
    hubert_path: str = "models/rvc/hubert_base.pt"
    rmvpe_path: str = "models/rvc/rmvpe.pt"
    device: str = "cuda:0"
    pitch_shift: int = -2
    index_rate: float = Field(default=0.66, ge=0.0, le=1.0)
    protect: float = Field(default=0.45, ge=0.0, le=0.5)
    f0_method: str = "rmvpe"
    rms_mix_rate: float = Field(default=0.35, ge=0.0, le=1.0)
    filter_radius: int = Field(default=1, ge=0)


class TTSConfig(_Strict):
    piper_voice_path: str = "models/piper/en_US-ryan-medium.onnx"
    piper_voice_config_path: str = "models/piper/en_US-ryan-medium.onnx.json"
    output_sample_rate: int = 22050
    sentence_flush_chars: str = ".!?\n"
    inter_sentence_pause_ms: int = Field(default=250, ge=0)
    piper_length_scale: float = Field(default=1.15, ge=0.1)
    pause_ms: int = Field(default=180, ge=0)
    edge_fade_ms: int = Field(default=4, ge=0)
    rvc: RVCConfig = Field(default_factory=RVCConfig)


class LoggingConfig(_Strict):
    file: str = "logs/ultron.log"
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"


class RoutingClassifierConfig(_Strict):
    rule_based_first: bool = True
    llm_fallback_enabled: bool = True
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class RoutingConfig(_Strict):
    """Phase 5 capability routing knobs."""
    llm_disambiguation_enabled: bool = True
    hybrid_task_decomposition_enabled: bool = True
    disambiguation_question_template: str = (
        "Did you mean to {coding_interpretation}, or to {automation_interpretation}?"
    )
    routing_log_path: str = "logs/routing_decisions.jsonl"
    classifier: RoutingClassifierConfig = Field(default_factory=RoutingClassifierConfig)
    # Stub responses are emitted while the OpenClaw integration is incomplete;
    # the OpenClaw integration prompt sets this to false.
    stub_responses_enabled: bool = True


class OpenClawConfig(_Strict):
    """Phase 5 placeholder for the OpenClaw peer Gateway. The dispatcher
    reads this to decide whether to attempt real calls or return stubs."""
    enabled: bool = False
    gateway_url: Optional[str] = None
    auth_token_env: str = "OPENCLAW_AUTH_TOKEN"
    health_check_timeout_seconds: float = 30.0
    health_check_interval_seconds: float = 60.0
    fail_open: bool = True              # treat unreachable as a stub, not a hard error
    required_agent_id: str = "ultron"


class ErrorPhrasesConfig(_Strict):
    """User-facing voice messages for dependency failures, in Ultron's voice.

    Each list is a phrase pool — TTS picks shuffled, no two consecutive
    plays. Empty list disables narration for that failure mode (the
    error still logs to errors.jsonl regardless).
    """
    qdrant_unavailable: List[str] = Field(default_factory=lambda: [
        "Memory's not responding right now.",
        "I can't reach my long-term memory at the moment.",
    ])
    brave_unavailable: List[str] = Field(default_factory=lambda: [
        "Search isn't working right now.",
        "I can't reach the web search service.",
    ])
    jina_unavailable: List[str] = Field(default_factory=lambda: [
        "I got search results but couldn't read the full pages.",
    ])
    anthropic_unavailable: List[str] = Field(default_factory=lambda: [
        "Anthropic's API isn't responding.",
        "I've lost connection to Claude.",
    ])
    rvc_unavailable: List[str] = Field(default_factory=lambda: [
        "Voice conversion is offline. You'll hear me without the Ultron filter for now.",
    ])
    openclaw_unavailable: List[str] = Field(default_factory=lambda: [
        "I'd ask the gateway to handle that, but it's not responding right now.",
    ])
    piper_unavailable: List[str] = Field(default_factory=lambda: [
        # Spoken via fallback path, may not actually be heard if Piper is down;
        # printed to terminal as well.
        "Speech output failed. I'm replying in text.",
    ])
    whisper_repeated_failures: List[str] = Field(default_factory=lambda: [
        "Speech recognition is having trouble.",
    ])
    addressing_classifier_failure: List[str] = Field(default_factory=lambda: [
        "Addressing detection is degraded; I'll respond to everything for now.",
    ])
    wake_word_model_failure: List[str] = Field(default_factory=lambda: [
        "Wake-word detection is offline. I'll listen continuously for now.",
    ])
    mcp_server_lost: List[str] = Field(default_factory=lambda: [
        "Lost connection to the coding orchestrator. The current task can't continue.",
    ])
    claude_code_subprocess_failed: List[str] = Field(default_factory=lambda: [
        "The coding subprocess failed. Want me to retry, or abandon the task?",
    ])
    config_invalid: List[str] = Field(default_factory=lambda: [
        # Almost certainly never spoken (config errors fail at startup before
        # voice stack loads), but defined for completeness.
        "Configuration is invalid. I can't start.",
    ])


class UltronConfig(_Strict):
    """Top-level configuration. Matches the structure of ``config.yaml``."""
    version: str = "1.0"
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    wake_word: WakeWordConfig = Field(default_factory=WakeWordConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    addressing: AddressingConfig = Field(default_factory=AddressingConfig)
    coding: CodingConfig = Field(default_factory=CodingConfig)
    projections: ProjectionsConfig = Field(default_factory=ProjectionsConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    error_phrases: ErrorPhrasesConfig = Field(default_factory=ErrorPhrasesConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    openclaw: OpenClawConfig = Field(default_factory=OpenClawConfig)


# ---------------------------------------------------------------------------
# Env-var substitution
# ---------------------------------------------------------------------------


_ENV_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _substitute_env_vars(value: Any) -> Any:
    """Recursively replace ``${VAR}`` placeholders in string values.

    Missing env vars resolve to an empty string. We do NOT raise here so
    a missing optional secret (e.g. unconfigured Brave key) doesn't break
    config loading; the consuming subsystem handles emptiness with a
    clear error at the point of use.
    """
    if isinstance(value, str):
        return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env_vars(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Loader + singleton
# ---------------------------------------------------------------------------


_CONFIG_INSTANCE: Optional[UltronConfig] = None
_CONFIG_PATH: Optional[Path] = None


def load_config(path: Optional[Path] = None) -> UltronConfig:
    """Load + validate ``config.yaml``. Caches the result.

    Resolution order for the file path:
      1. Explicit ``path`` argument.
      2. ``ULTRON_CONFIG_PATH`` env var.
      3. ``DEFAULT_CONFIG_PATH`` (``<project root>/config.yaml``).

    A missing file raises ``FileNotFoundError`` — we don't fall through
    to defaults because silent fallback hides config-file typos. To run
    with all defaults, pass ``UltronConfig()`` directly via tests; the
    production path REQUIRES a real file.
    """
    global _CONFIG_INSTANCE, _CONFIG_PATH
    if path is None:
        env_override = os.environ.get("ULTRON_CONFIG_PATH")
        path = Path(env_override) if env_override else DEFAULT_CONFIG_PATH

    # Late import: ultron.errors imports nothing from this module, but we
    # avoid a top-level import to keep the loader's bootstrap dependencies
    # minimal in case future error types pull in heavier modules.
    from ultron.errors import ConfigurationError

    path = Path(path)
    if not path.is_file():
        raise ConfigurationError(
            f"config.yaml not found at {path}. Set ULTRON_CONFIG_PATH or "
            f"create the file at the project root.",
            context={"path": str(path)},
        )

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"config.yaml is not valid YAML: {e}",
            context={"path": str(path)},
        ) from e
    except OSError as e:
        raise ConfigurationError(
            f"config.yaml could not be read: {e}",
            context={"path": str(path)},
        ) from e

    raw = _substitute_env_vars(raw)
    try:
        _CONFIG_INSTANCE = UltronConfig.model_validate(raw)
    except Exception as e:
        # Surface the validation error with the file path so the user can
        # find the offending key fast.
        raise ConfigurationError(
            f"config.yaml validation failed: {e}",
            context={"path": str(path)},
        ) from e
    _CONFIG_PATH = path
    return _CONFIG_INSTANCE


def get_config() -> UltronConfig:
    """Singleton accessor. Loads on first call from the default path.

    Tests that want a fresh in-memory config should call
    :func:`set_config` to inject a test instance and remember to restore
    on teardown.
    """
    if _CONFIG_INSTANCE is None:
        load_config()
    return _CONFIG_INSTANCE  # type: ignore[return-value]


def reload_config(path: Optional[Path] = None) -> UltronConfig:
    """Force reload from disk. Useful for dev workflows; some sections
    need a process restart (LLM model path, Qdrant data dir, audio
    sample rate, etc.) — that's the caller's call to enforce."""
    global _CONFIG_INSTANCE
    _CONFIG_INSTANCE = None
    return load_config(path)


def set_config(cfg: UltronConfig) -> None:
    """Inject a pre-built config (for tests). Pair with a fixture that
    captures and restores the previous instance."""
    global _CONFIG_INSTANCE
    _CONFIG_INSTANCE = cfg


def current_config_path() -> Optional[Path]:
    """Return the file the singleton was loaded from, or None if no
    file load has happened yet."""
    return _CONFIG_PATH


__all__ = [
    "PROJECT_ROOT",
    "MODELS_DIR",
    "LOGS_DIR",
    "DEFAULT_CONFIG_PATH",
    "resolve_path",
    "AudioConfig",
    "VADConfig",
    "WakeWordConfig",
    "STTConfig",
    "LLMConfig",
    "EmbeddingsConfig",
    "QdrantCollections",
    "QdrantConfig",
    "MemoryConfig",
    "BraveConfig",
    "JinaConfig",
    "WebCacheConfig",
    "WebSearchConfig",
    "AddressingConfig",
    "CodingMCPConfig",
    "CodingVerificationConfig",
    "CodingConfig",
    "ProjectionsBudgets",
    "ProjectionsConfig",
    "RVCConfig",
    "TTSConfig",
    "LoggingConfig",
    "ErrorPhrasesConfig",
    "RoutingClassifierConfig",
    "RoutingConfig",
    "OpenClawConfig",
    "UltronConfig",
    "load_config",
    "get_config",
    "reload_config",
    "set_config",
    "current_config_path",
]
