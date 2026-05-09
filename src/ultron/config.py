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
from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class LLMRagConfig(_Strict):
    """4B optimization plan Stage G — RAG-snippet injection position.

    Where retrieved Qdrant memories land in the LLM context window:

    - ``"system"`` (legacy): folded into the leading system message
      (today's path until Stage G). Qwen3's chat template rejects a
      second system-role message, so RAG content was concatenated to
      the persona-system content.
    - ``"recency"`` (new default at Stage G): emitted as a
      ``[Relevant context]\\n…\\n\\n`` prefix on the user message,
      placing it in the recency / strongest-attention zone right
      before the user query. Qwen3's chat template accepts this
      because the user role is unaffected. Per the optimization plan,
      this gives +10–20% recall on injected memories on the 4B.

    The system position is kept for back-compat (rollback path if the
    recency injection regresses voice character or surfaces other
    unforeseen issues).
    """

    position: Literal["system", "recency"] = "recency"


class LLMCompressionConfig(_Strict):
    """4B optimization plan Item 4 — context compression for RAG / web /
    history blocks before LLM injection.

    Per LLMLingua (and the lighter EDU-style follow-ups), token-level
    compression of high-redundancy text (retrieved Qdrant memories,
    Jina-fetched articles, conversation history) can free 1.5–5×
    context budget without measurable answer-quality loss. The 4B
    benefits more than the 9B because it has less attention to spare
    on filler.

    Default OFF — over-aggressive compression CAN drop nuance, so the
    flip is gated on live measurement. When enabled, the heuristic
    compressor (no extra model) drops stopwords, redundant
    punctuation, repeated paragraph signatures, and contractions to
    approximate LLMLingua's coarse pass. A real perplexity-scorer
    hook is plumbed in via :class:`ultron.llm.compression.Compressor`
    so a follow-up swap to true LLMLingua (using the Stage C
    speculative-decoding 0.8B as the scorer) is a one-call change.

    ``target_ratio`` is the desired compression — 1.5 means drop ~33%
    of the input. The heuristic is best-effort; actual ratio depends
    on input redundancy. Per-block flags let you opt in to specific
    surfaces without globally turning compression on.
    """

    enabled: bool = False
    target_ratio: float = Field(default=1.5, ge=1.0, le=10.0)
    compress_rag: bool = True       # Qdrant retrieval block before injection
    compress_web: bool = True       # Jina-fetched article body
    compress_history: bool = False  # conversation history (riskiest — has user voice)


class LLMSelfConsistencyConfig(_Strict):
    """4B optimization plan Item 6 — self-consistency on high-stakes calls.

    Self-consistency samples N diverse reasoning paths at non-zero
    temperature and majority-votes the most consistent answer. Per the
    paper this gives big lifts on chain-of-thought tasks (GSM8K +17.9 %).

    Default OFF — applied only at projection-driven call sites flagged
    by the orchestrator (decomposer, web-gating preflight, etc.); never
    on the voice hot path. The 3× token cost is acceptable on those
    specific calls because they're already off the critical TTFT path.

    ``disabled_sites`` is a per-site opt-out so individual call sites
    can be excluded without flipping the global flag — useful for
    measurement A/B testing.
    """

    enabled: bool = False
    n: int = Field(default=3, ge=1)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    disabled_sites: List[str] = Field(default_factory=list)


class LLMPersonaConfig(_Strict):
    """Where the voice-path system prompt comes from.

    Phase 1 of the OpenClaw integration migrated Ultron's persona
    out of ``config.yaml:llm.system_prompt`` and into the shared
    workspace files (``SOUL.md``, ``IDENTITY.md``, ``USER.md``,
    ``AGENTS.md``). Both Ultron's voice pipeline and OpenClaw read
    from the same workspace so a SOUL.md edit is reflected in both.

    Defaults to ``workspace`` (the migrated path). Set to ``config``
    to revert to the hardcoded ``llm.system_prompt`` string — useful
    for tests and for environments without a workspace.

    Hot reload: when source is ``workspace``, the loader's
    ``refresh_if_stale`` is called on every LLM turn so SOUL.md edits
    land without restart. The cost is ~6 stat() calls per turn,
    sub-millisecond.
    """

    source: Literal["workspace", "config"] = "workspace"
    workspace_dir: Optional[str] = None  # None -> default_workspace_dir()
    fallback_to_config_on_empty: bool = True
    hot_reload: bool = True


# 4B optimization plan Stage A — preset table.
# Switching presets is a one-line config change. Each preset bundles the
# model_path / n_ctx / draft_model_path that go together for that model
# size. Users on `preset: "custom"` get no auto-resolution and must
# specify `model_path` themselves (back-compat for tests + advanced
# users). See docs/4b_optimization_plan.md.
LLM_PRESETS: dict[str, dict[str, Any]] = {
    "qwen3.5-9b": {
        "model_path": "models/Qwen3.5-9B-Q4_K_M.gguf",
        "n_ctx": 8192,
        "draft_model_path": None,  # 9B doesn't pair well with a draft (diminishing returns)
    },
    "qwen3.5-4b": {
        "model_path": "models/Qwen3.5-4B-Q4_K_M.gguf",
        # n_ctx pinned at 8192 to match the 9B-era voice-path TTFT
        # baseline. The 4B + 16384 ctx the original plan recipe
        # specified is for the HTTP-server launcher (`--n-ctx 16384`),
        # not the in-process voice path — a larger context measurably
        # raises TTFT (86 ms → 125 ms in live measurement) because the
        # KV cache is twice as big. Users who want the 16384 context
        # can override `n_ctx: 16384` explicitly in config.yaml.
        "n_ctx": 8192,
        "draft_model_path": "models/Qwen3.5-0.8B-Q4_K_M.gguf",
    },
}


class LLMConfig(_Strict):
    # Pinned to llama_cpp per feedback_llm_runtime_decision.md (2026-05-08).
    provider: Literal["llama_cpp"] = "llama_cpp"
    # 4B optimization plan Stage A — model preset.
    #   "qwen3.5-9b" — current default; resolves to the 9B GGUF + n_ctx=8192,
    #                  no draft model.
    #   "qwen3.5-4b" — 4B target + 0.8B draft for speculative decoding,
    #                  n_ctx=16384. Flipped on after Stage H regression
    #                  sweep passes.
    #   "custom"     — no auto-resolution; raw model_path / n_ctx /
    #                  draft_model_path fields are used as-is. For tests
    #                  and ad-hoc model swaps.
    # Preset defaults only fill in fields the user did NOT explicitly
    # set in YAML — see ``_apply_preset``.
    preset: Literal["qwen3.5-9b", "qwen3.5-4b", "custom"] = "qwen3.5-9b"
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
    # Optional draft model for speculative decoding. None = no spec
    # decoding. Wired into ``scripts/start_llamacpp_server.py`` in
    # Stage C of the 4B plan. The voice in_process path doesn't use it
    # yet (llama-cpp-python's speculative API is server-only at the
    # moment we're integrating).
    draft_model_path: Optional[str] = None
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
    persona: LLMPersonaConfig = Field(default_factory=LLMPersonaConfig)
    rag: LLMRagConfig = Field(default_factory=LLMRagConfig)
    # 4B plan Item 6 — self-consistency on high-stakes projection-driven calls.
    self_consistency: LLMSelfConsistencyConfig = Field(
        default_factory=LLMSelfConsistencyConfig,
    )
    # 4B plan Item 4 — context compression (RAG / web / history blocks).
    compression: LLMCompressionConfig = Field(
        default_factory=LLMCompressionConfig,
    )

    @model_validator(mode="after")
    def _apply_preset(self) -> "LLMConfig":
        """Fill in preset-derived fields the user didn't explicitly set.

        Only fields absent from ``model_fields_set`` (i.e., left at
        their factory defaults during instantiation) are touched.
        Explicit user values in YAML always win — this is what makes
        ``preset: "custom"`` + raw fields work for tests, and what lets
        an advanced user override one knob (e.g., `n_ctx: 4096`) while
        keeping the rest of the preset's defaults.

        Raises ``ValueError`` for ``preset: "custom"`` only when no
        ``model_path`` is supplied (impossible in practice because the
        field has a default, but enforced for explicitness).
        """
        if self.preset == "custom":
            if not self.model_path:
                raise ValueError(
                    "llm.preset='custom' requires llm.model_path to be set"
                )
            return self
        defaults = LLM_PRESETS.get(self.preset)
        if defaults is None:  # pragma: no cover — Literal narrows this away
            return self
        for field, value in defaults.items():
            if field not in self.model_fields_set:
                # Bypass pydantic's frozen-after-validation by going
                # through __dict__ directly. Safe here because we're
                # still inside model construction.
                object.__setattr__(self, field, value)
        return self


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


class MemoryRetrievalConfig(_Strict):
    """V1-gap A2: multi-pass per-category retrieval.

    Default ON. The fan-out path only fires when the gate verdict has
    ``context_categories`` populated, which only the LLM-preflight
    branch produces. The standard 10-query voice baseline routes
    through hard rules whose verdicts have empty categories — those
    queries automatically fall back to single-pass retrieval, so the
    voice TTFT contract (median ≤ 79 ms) is preserved.

    Memory-heavy or ambiguous queries that DO trigger the LLM
    preflight pay an additional ~150-200 ms for the fan-out + composite
    re-ranking. That is the spec-intended cost; the queries that
    benefit from "you didn't ask but you'd want to know" memory
    retrieval are the same ones whose preflight already added latency.
    """

    multi_pass_enabled: bool = True
    max_categories_per_query: int = Field(default=4, ge=0, le=10)
    candidates_per_category_multiplier: int = Field(default=4, ge=1, le=20)


class MemoryRankingConfig(_Strict):
    """V1-gap A2: weighted blend of RRF + recency + surprise - redundancy."""

    rrf_weight: float = Field(default=1.0, ge=0.0)
    recency_weight: float = Field(default=0.2, ge=0.0)
    recency_half_life_days: float = Field(default=7.0, gt=0.0)
    surprise_weight: float = Field(default=0.15, ge=0.0)
    redundancy_weight: float = Field(default=0.3, ge=0.0)


class MemoryConfig(_Strict):
    enabled: bool = True
    jsonl_legacy_path: str = "data/memory.jsonl"
    recent_turns: int = Field(default=20, ge=0)
    rag_top_k: int = Field(default=5, ge=0)
    rag_exclude_recent: int = Field(default=20, ge=0)
    facts_top_k: int = Field(default=3, ge=0)
    write_queue_maxsize: int = Field(default=256, ge=1)
    # V1-gap A2.
    retrieval: MemoryRetrievalConfig = Field(default_factory=MemoryRetrievalConfig)
    ranking: MemoryRankingConfig = Field(default_factory=MemoryRankingConfig)


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


class CitationConfig(_Strict):
    """V1-gap B3: citation rendering format.

    Default ``"superscript"`` matches the V1-spec Part 4.4 wording
    (Unicode ¹²³ inline citations). The references list at the end
    of the prompt + the visible transcript keep the bracketed
    ``[N]`` form for monospace clarity, so the user can match
    inline ¹ to the bracketed [1] reference unambiguously.

    Set to ``"bracket"`` for ASCII-only consumers.
    """

    inline_marker_format: str = "superscript"  # "bracket" | "superscript"


class WebSearchConfig(_Strict):
    enabled: bool = True
    brave_api_key_env: str = "ULTRON_BRAVE_API_KEY"
    brave: BraveConfig = Field(default_factory=BraveConfig)
    jina: JinaConfig = Field(default_factory=JinaConfig)
    cache: WebCacheConfig = Field(default_factory=WebCacheConfig)
    # V1-gap B3.
    citation: CitationConfig = Field(default_factory=CitationConfig)


class AddressingConfig(_Strict):
    follow_up_enabled: bool = True
    # CONFIRMED 30s, NOT the Foundation prompt's 10s, per feedback_ultron_extension.md
    warm_mode_duration_seconds: float = 30.0
    default_uncertain_to_not_addressed: bool = True
    rule_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    zero_shot_model: str = "google/flan-t5-small"
    load_eagerly: bool = True
    log_path: str = "logs/addressing.jsonl"


class CodingCanonicalMonitorConfig(_Strict):
    """4B optimization plan Item 7 — canonical-path monitor.

    Tracks per-session tool-call sequences. When too many off-canonical
    tool calls land in the early window, signals abort so the runner
    can reset the session with a cleaner prompt instead of letting it
    fail at verification.

    Per the underlying paper (Each off-canonical call raises the next
    one's probability of being off-canonical by 22.7 pp), restarting
    the bottom tercile of runs lifts success rates by +8.8 pp. Tuning
    the thresholds aggressively risks restarting healthy sessions —
    leave defaults conservative, default OFF, and turn on after live
    measurement.

    Canonical tools per task type are defined in
    :mod:`ultron.coding.canonical_monitor`; they cover the standard
    coding actions (Read / Write / Edit / Glob / Grep / Bash /
    TodoWrite). Anything outside that set in a CODE_TASK session is
    counted as off-canonical.
    """

    enabled: bool = False
    # Trigger abort when this many off-canonical calls land in the
    # early window.
    off_canonical_threshold: int = Field(default=3, ge=1)
    # Window size: count over the FIRST N tool calls of a session.
    early_window_calls: int = Field(default=10, ge=1)


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


class CodingFactsConfig(_Strict):
    """A3 wiring -- stored-facts fast-path on clarifications.

    The Coordinator's ``decide_clarification`` consults the Qdrant
    ``facts`` collection (populated by ``scripts/maintenance.py``) before
    calling the LLM. A high-confidence fact in a directive category
    short-circuits the decision and answers Claude directly.

    Defaults err on the cautious side: a fact must clear both a
    confidence and an RRF-score threshold before it answers, and the
    age cap is null (off) so newer installs without long history aren't
    dependent on calendar age.
    """

    top_k: int = Field(default=5, ge=1)
    min_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    min_score: float = Field(default=0.85, ge=0.0)
    max_age_days: Optional[float] = None


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
    # 4B plan Item 7 — canonical-path monitor (off by default).
    canonical_monitor: CodingCanonicalMonitorConfig = Field(
        default_factory=CodingCanonicalMonitorConfig,
    )
    # A3 wiring -- stored-facts fast-path on clarifications.
    facts: CodingFactsConfig = Field(default_factory=CodingFactsConfig)
    # A4 pre-task confirmation. Default OFF -- the spoken confirmation
    # adds ~0.5 s of TTS playback before every coding task dispatch,
    # which is a UX cost that has to fire universally to provide its
    # safety value. Flip true to opt in to the wake-word barge-in
    # window before destructive coding actions.
    pre_task_confirmation_enabled: bool = False
    pre_task_confirmation_max_words: int = Field(default=30, ge=4)
    pre_task_barge_in_window_seconds: float = Field(default=0.5, ge=0.0, le=10.0)
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


class RoutingIRMAConfig(_Strict):
    """4B optimization plan Item 5 — IRMA-style input reformulation.

    When enabled, the IntentDisambiguator wraps the raw utterance with
    relevant context (recent intent decisions, active session, routing
    hints) before sending it to the LLM. Per the IRMA paper, this
    significantly outperforms ReAct / Function-Calling / Self-Reflection
    on ambiguous tool calls — but the gain is on the disambiguator
    pass only; the voice-path hot loop is unaffected.

    Default OFF — flip when the live disambiguator stats show enough
    "wrong-side" decisions to justify the extra context tokens.
    """

    enabled: bool = False
    max_recent_decisions: int = Field(default=5, ge=0)


class RoutingConfig(_Strict):
    """Phase 5 capability routing knobs."""
    llm_disambiguation_enabled: bool = True
    hybrid_task_decomposition_enabled: bool = True
    disambiguation_question_template: str = (
        "Did you mean to {coding_interpretation}, or to {automation_interpretation}?"
    )
    routing_log_path: str = "logs/routing_decisions.jsonl"
    classifier: RoutingClassifierConfig = Field(default_factory=RoutingClassifierConfig)
    # 4B plan Item 5 — IRMA-style input reformulation for the disambiguator.
    irma: RoutingIRMAConfig = Field(default_factory=RoutingIRMAConfig)
    # Stub responses are emitted while the OpenClaw integration is incomplete;
    # the OpenClaw integration prompt sets this to false.
    stub_responses_enabled: bool = True


class OpenClawBlockAndReviseConfig(_Strict):
    """4B optimization plan Item 8 — block-and-revise on OpenClaw tool calls.

    Per the runtime-verifier-mediation paper, a pre-flight LLM check
    asking "does this tool call advance the user's stated goal?" can
    intercept misdirected actions before they fire. When the validator
    rejects, the dispatcher returns the validator's reason as the
    voice message rather than executing the tool — analogous to the
    coding pipeline's verification + corrective-prompt loop, but for
    the automation side.

    Default OFF. The validator adds one short LLM call per dispatch
    (~50 tokens output) so it's not free; flip after live measurement
    shows the validator catches enough mis-dispatches to justify the
    overhead.

    The validator falls open: if the LLM call fails or the response
    is unparseable, the dispatcher proceeds as if the validator wasn't
    there. Better to occasionally allow a borderline call than to
    block legitimate work on a transient LLM failure.
    """

    enabled: bool = False


class OpenClawBridgeConfig(_Strict):
    """Phase 3 bridge layer — CLI transport, MCP registration, workspace
    IO, inbound event routing.

    The bridge is consulted only when:

    - Ultron's orchestrator wants to call an OpenClaw tool (browser, image
      generation, messaging, etc.).
    - Ultron starts up (registers Ultron MCP with the Gateway).
    - OpenClaw forwards an inbound event Ultron should react to.

    The voice pipeline does NOT touch the bridge. Voice queries flow
    through the existing in-process pipeline without consulting OpenClaw.
    All bridge ops fail open per the parent ``fail_open`` flag — when
    the Gateway is down the bridge logs and degrades capabilities, but
    the voice path keeps working.

    Phase 3 deviates from the integration spec's HTTP transport: OpenClaw
    2026.5.7 doesn't expose ``/tools/invoke`` or ``/messages`` HTTP
    endpoints. The CLI is the documented public interface, so the
    bridge invokes it via subprocess. ``cli_path`` points at
    ``openclaw.cmd`` (Windows) or ``openclaw`` (POSIX).
    """

    # CLI transport
    cli_path: Optional[str] = None
    cli_timeout_seconds: float = 30.0

    # MCP registration (Phase 3.2 + Phase 13 auto-resolve)
    mcp_server_name: str = "ultron-mcp"
    mcp_server_command: Optional[str] = "auto"
    """Stdio entry-point command for OpenClaw to spawn when calling
    Ultron's MCP. Three semantics:

    - ``"auto"`` (default): resolve to the canonical entry script
      ``scripts/run_ultron_mcp_for_openclaw.py`` invoked via the
      project's ``.venv`` Python. The bridge holder does the
      resolution at construction; the registrar receives the
      concrete path.
    - explicit path: use as-is (absolute path strongly recommended
      so OpenClaw's spawn from any cwd resolves correctly).
    - ``None``: disable registration entirely (the bridge runs
      without MCP exposure)."""

    mcp_server_args: List[str] = Field(default_factory=list)
    """Extra args appended to ``command`` at spawn time. With the
    ``"auto"`` resolution, the bridge holder also adds the entry
    script path automatically — set this list to add extra flags
    after that. With an explicit command, populate fully here."""

    retry_registration_interval_seconds: float = 60.0

    # Workspace IO (Phase 3.3)
    workspace_dir: Optional[str] = None
    workspace_lock_timeout_seconds: float = 5.0

    # Inbound events (Phase 3.4 — gated off until Phase 4+)
    inbound_voice_handoff_enabled: bool = False
    inbound_voice_handoff_prefix: str = "[voice]"

    # Per-call timeouts
    tool_invocation_timeout_seconds: float = 30.0
    message_send_timeout_seconds: float = 10.0


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
    # 4B plan Item 8 — block-and-revise pre-flight validator on tool calls.
    block_and_revise: OpenClawBlockAndReviseConfig = Field(
        default_factory=OpenClawBlockAndReviseConfig,
    )
    # Phase 3 bridge layer — CLI transport, MCP registration, workspace IO.
    bridge: OpenClawBridgeConfig = Field(default_factory=OpenClawBridgeConfig)


# ---------------------------------------------------------------------------
# Notifications (Phase 4)
# ---------------------------------------------------------------------------


class TelegramNotifyOnConfig(_Strict):
    """Per-event opt-in flags for Telegram notifications."""

    coding_task_completion: bool = True
    coding_task_clarification_needed: bool = True
    heartbeat_alerts: bool = True
    standing_order_outputs: bool = True
    search_results_async: bool = False                  # opt-in; can be noisy


class TelegramNotificationsConfig(_Strict):
    """Telegram channel for proactive notifications (Phase 4)."""

    enabled: bool = False
    user_id_env: str = "TELEGRAM_USER_ID"
    """Env var that resolves to the Telegram user id messages are sent
    to. Stored in env (not config) so the user id stays out of git."""

    fallback_user_id: Optional[str] = None
    """Direct user id used when ``user_id_env`` is unset. Useful for
    tests; production setups should leave this ``None`` and rely on the
    env var."""

    notify_on: TelegramNotifyOnConfig = Field(default_factory=TelegramNotifyOnConfig)


class NotificationsConfig(_Strict):
    """Phase 4 — proactive notifications from Ultron to remote channels.

    These are off-hot-path: the orchestrator fires-and-forgets the
    bridge call after a coding task completes, after a heartbeat
    alert lands, etc. Failures are logged and never propagate to the
    voice path.
    """

    telegram: TelegramNotificationsConfig = Field(
        default_factory=TelegramNotificationsConfig,
    )


# ---------------------------------------------------------------------------
# Heartbeat (Phase 5)
# ---------------------------------------------------------------------------


class BrowserConfig(_Strict):
    """Phase 6 — browser tool integration via OpenClaw.

    Controls voice-side ack timing and per-call timeouts. The
    browser tool itself is provided by OpenClaw's bundled plugin;
    Ultron's wrapper (:class:`BrowserTool`) just routes intent to
    its primitives via :meth:`OpenClawClient.invoke_tool`.
    """

    enabled: bool = True
    """Master switch. When False, MESSAGING / BROWSER intents fall
    back to the dispatcher's stub voice messages even when a bridge
    is wired. Use this to suppress browser dispatch without
    disabling the entire bridge."""

    default_snapshot_mode: Literal["ai", "aria"] = "ai"

    default_navigation_timeout_seconds: float = 30.0
    default_action_timeout_seconds: float = 10.0
    default_screenshot_timeout_seconds: float = 30.0

    long_running_progress_threshold_seconds: float = 5.0
    """Above this threshold, the orchestrator should narrate
    intermediate progress rather than wait silently."""

    acknowledgment_phrases: List[str] = Field(default_factory=lambda: [
        "Pulling up that page now.",
        "Looking at it.",
        "Loading the site.",
        "Give me a moment to navigate.",
    ])
    """Voice ack played within ~200ms of a browser intent firing.
    Mirrors the existing web-search ack pool. Phrases stay in
    Ultron's voice (precise, weighted, no filler)."""


class MediaGenerationConfig(_Strict):
    """Phase 12 — image / video / music generation via OpenClaw.

    Like the browser tool (Phase 6), media generation rides through
    :meth:`OpenClawClient.invoke_tool` against a tool slug from
    OpenClaw's provider plugin set.

    **Provider policy (project-wide):** Ultron only accepts
    free-or-local providers. ComfyUI (local Stable Diffusion) is the
    canonical option. Pay-per-use APIs (Fal, Runway, Suno, etc.) are
    NOT supported — Claude Code is the only paid service in the
    stack. Concrete provider configuration lives in
    ``~/.openclaw/openclaw.json`` under ``models.providers.<slug>``;
    Ultron just routes the intent. See
    ``docs/openclaw_media_generation_setup.md`` for the full setup.
    """

    enabled: bool = True
    """Master switch. False suppresses media-gen dispatch even when
    the bridge is wired."""

    image_tool: str = "image_generate"
    video_tool: str = "video_generate"
    music_tool: str = "music_generate"
    """Tool slugs OpenClaw exposes when the provider plugins are
    enabled. Override if the user's provider uses different names."""

    default_image_provider: Optional[str] = None
    default_video_provider: Optional[str] = None
    default_music_provider: Optional[str] = None
    """Optional explicit provider names passed through as a tool
    parameter. ``None`` lets OpenClaw pick its default."""

    default_timeout_seconds: float = 120.0
    """Generation jobs run for tens of seconds typically."""

    delivery_voice: str = "telegram"
    """Where to deliver media when the user issued the intent via
    voice. Voice channels can't display images, so the result is
    forwarded to Telegram + a voice ack confirms delivery."""

    delivery_text: str = "inline"
    """Where to deliver media when the user issued the intent via
    Telegram. ``"inline"`` returns the result in the same chat;
    overrides per-channel can be set in OpenClaw's channel config."""

    acknowledgment_phrases: List[str] = Field(default_factory=lambda: [
        "Working on that. Should be a moment.",
        "Generating now.",
        "I'll send it when it's ready.",
    ])


class HeartbeatConfig(_Strict):
    """Phase 5 — heartbeat alert persistence + Ultron-side query.

    The OpenClaw-side heartbeat agent (configured separately in
    ``~/.openclaw/openclaw.json`` under ``agents[].heartbeat``) raises
    alerts that Ultron records here so a voice query like "what alerts
    did you flag?" can pull from the alert log. This config controls
    local persistence and retention.
    """

    alert_log_path: str = "logs/heartbeat_alerts.jsonl"
    """Where heartbeat alerts get appended (JSONL). Path resolves
    against the project root if relative."""

    alert_retention_days: int = Field(default=30, ge=1, le=365)
    """How long alerts stay in the log before pruning. Pruning runs
    on demand via ``HeartbeatAlertLog.prune()``; not automatic."""

    auto_notify_telegram: bool = True
    """When true, every recorded alert is also fired through
    :class:`NotificationDispatcher.notify_heartbeat_alert`. Per-event
    notify gating still applies — set
    ``notifications.telegram.notify_on.heartbeat_alerts: false`` to
    suppress Telegram delivery without touching this flag."""


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


class GamingModeConfig(_Strict):
    """V1-gap A1: anticheat-safe shutdown of OpenClaw plugins.

    When the user says "gaming mode", "I'm about to play Valorant",
    etc., the manager calls ``openclaw plugins disable <id>`` for each
    slug listed below, optionally stops Docker Desktop, and logs the
    transition. ``gaming mode off`` reverses the cycle.

    Default OFF -- this is a safety-critical toggle that disables
    OpenClaw plugins. Operator opts in explicitly once they have the
    plugins installed and want the voice trigger active.
    """

    enabled: bool = False
    plugins_to_disable: List[str] = Field(
        default_factory=lambda: ["desktop-control", "windows-control"],
    )
    toggle_docker: bool = False
    docker_executable_path: Optional[str] = None
    docker_process_name: str = "Docker Desktop"
    log_path: str = "logs/gaming_mode.jsonl"


class DesktopConfig(_Strict):
    """V1-gap C3: voice routing for the OpenClaw ``desktop-control`` plugin.

    Tool slugs are configurable so plugin renames don't require code
    changes. Default ``enabled=True`` -- the dispatcher gates each
    call on bridge availability + plugin reachability and falls back
    to a clear "isn't wired up yet" voice message if either is missing.
    """

    enabled: bool = True
    default_screenshot_timeout_seconds: float = 10.0
    default_action_timeout_seconds: float = 5.0
    plugin_slug: str = "desktop-control"
    tool_slug_screenshot: str = "desktop_screenshot"
    tool_slug_list_windows: str = "desktop_list_windows"
    tool_slug_find_window: str = "desktop_find_window"


class WindowControlConfig(_Strict):
    """V1-gap C3: voice routing for the OpenClaw ``windows-control`` plugin
    (UI Automation).

    Same shape + posture as :class:`DesktopConfig`. Default ON; runtime
    fail-open behaviour matches the desktop wrapper.
    """

    enabled: bool = True
    default_action_timeout_seconds: float = 5.0
    plugin_slug: str = "windows-control"
    tool_slug_focus: str = "windows_focus_window"
    tool_slug_click: str = "windows_click_element"
    tool_slug_type: str = "windows_type_text"


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
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    media_generation: MediaGenerationConfig = Field(default_factory=MediaGenerationConfig)
    # V1-gap A1 / C3.
    gaming_mode: GamingModeConfig = Field(default_factory=GamingModeConfig)
    desktop: DesktopConfig = Field(default_factory=DesktopConfig)
    window_control: WindowControlConfig = Field(default_factory=WindowControlConfig)


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

    # 4B optimization plan — on-the-fly preset switching via env var.
    # ``ULTRON_LLM_PRESET=qwen3.5-4b python -m ultron`` picks the preset
    # without editing config.yaml. The env var also clears any explicit
    # ``model_path`` / ``draft_model_path`` / ``n_ctx`` overrides in the
    # YAML so the preset's table values win — that's what makes the
    # switch a single env-var change rather than a four-line YAML edit.
    # If you DO want the env-var preset to inherit your YAML overrides,
    # set ``ULTRON_LLM_PRESET_KEEP_OVERRIDES=1``.
    env_preset = os.environ.get("ULTRON_LLM_PRESET")
    if env_preset:
        llm_block = raw.setdefault("llm", {})
        llm_block["preset"] = env_preset
        if not os.environ.get("ULTRON_LLM_PRESET_KEEP_OVERRIDES"):
            for key in ("model_path", "draft_model_path", "n_ctx"):
                llm_block.pop(key, None)

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
