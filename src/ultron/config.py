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
    # 2026-05-16 latency pass 2: 512 -> 256 (32 ms -> 16 ms at 16 kHz).
    # Halves the mic-to-consumer queue latency. Silero VAD's internal
    # window is 512 samples regardless -- it buffers two 256-sample
    # chunks for one decision -- so VAD timing is unchanged, but the
    # orchestrator's per-block silence-onset detection has finer
    # granularity (16 ms steps vs 32 ms) for the speculative-Whisper
    # kick-off in Phase 4.
    blocksize: int = 256
    dtype: str = "float32"
    input_device: Optional[str] = None        # env: ULTRON_AUDIO_DEVICE
    output_device: Optional[str] = None       # env: ULTRON_AUDIO_OUTPUT_DEVICE
    barge_in_enabled: bool = True
    barge_in_grace_seconds: float = 0.5
    # Total capacity of the audio ring buffer. The orchestrator slices
    # mode-specific pre-roll out of this -- see ``cold_pre_roll_seconds``
    # and ``warm_pre_roll_seconds``. Capacity should be >= max(cold, warm)
    # so the largest slice has the audio it needs.
    ring_buffer_seconds: float = 0.5
    # COLD path pre-roll (post-wake-word capture). Short by design so
    # the wake-word "Ultron" tail does not bleed into Whisper as a
    # "Tron" prefix. The 2026-05-09 latency hot-fix sized this for the
    # custom wake-word model's typical fire latency. If you switch
    # wake-word models, re-tune.
    cold_pre_roll_seconds: float = Field(default=0.15, ge=0.0)
    # WARM path pre-roll (post-TTS follow-up listening). Longer than
    # COLD because there is no wake-word firing event to align against
    # -- the user just starts talking from silence and Silero VAD has
    # ~100-200 ms detection latency. Without enough pre-roll the
    # leading word gets clipped (e.g. "What's the weather like" comes
    # through as "the weather like").
    warm_pre_roll_seconds: float = Field(default=0.5, ge=0.0)
    # 2026-05-09 audio-quality pass: linear gain applied in dB to the
    # captured audio chunk BEFORE it reaches VAD / wake-word / Whisper.
    # 0.0 dB = no-op (legacy behaviour). Use a positive value when the
    # mic is hot enough to capture the user up close but too quiet at
    # range. Negative values attenuate (rare; only useful when the mic
    # is clipping at close range). Hard-clipped to int16 range after
    # gain to prevent distortion.
    input_gain_db: float = Field(default=0.0, ge=-20.0, le=40.0)


class SmartTurnConfig(_Strict):
    """Smart Turn V3 -- semantic end-of-turn confirmation (CPU-only).

    Runs AFTER Silero VAD declares end-of-speech. When enabled with
    a present model file, the orchestrator uses
    ``fast_path_silence_duration_ms`` as the baseline VAD silence
    requirement (typically much shorter than the legacy value); on
    SPEECH_END the captured audio is fed to the model and the verdict
    determines whether to submit immediately (complete) or keep
    listening (incomplete). Long utterances beyond ``window_seconds``
    of speech bypass the model -- the existing adaptive long-utterance
    backstop already handles that case at the VAD layer.

    Fail-open at every level: model file missing -> orchestrator
    silently falls back to legacy VAD-only behaviour; inference
    failure mid-call -> verdict treated as undecided -> caller trusts
    VAD. Voice baseline is preserved when this is disabled or the
    model is missing.
    """

    # 2026-05-12 Smart Turn V3 default ON: production-grade fail-open
    # at every layer; missing model file degrades to legacy behaviour
    # without any UX change. Operators can flip false to opt out.
    enabled: bool = True
    # Path to ``smart-turn-v3.2-cpu.onnx`` (or compatible newer revision).
    # Relative paths resolve against PROJECT_ROOT.
    model_path: str = "models/smart_turn/smart-turn-v3.2-cpu.onnx"
    # Sigmoid output threshold above which the model declares the
    # turn complete. Pipecat's production-tested default is 0.5;
    # tightening (0.6-0.7) reduces false-positive cut-offs at the
    # cost of longer perceived latency on confidently-done turns.
    # Loosening (0.3-0.4) is rarely useful -- it just trusts VAD more.
    completion_threshold: float = Field(default=0.5, ge=0.05, le=0.95)
    # 2026-05-16 latency pass 2: gradient-fire confidence threshold for
    # the EARLY check at ``fast_path_silence_duration_ms``. When the
    # model returns prob >= early_completion_threshold at the early
    # checkpoint (300 ms by default), we submit immediately. When prob
    # is in [completion_threshold, early_completion_threshold) we wait
    # an additional ``medium_grace_ms`` and re-check with the lower
    # threshold (compensates for the model being less confident on the
    # shorter silence tail). When prob < completion_threshold we enter
    # the existing ``incomplete_extension_ms`` path (user trailed off).
    # Pipecat's smart-turn-v3.2 blog notes higher accuracy on short
    # utterances; 0.65 is a conservative early-fire bar that empirically
    # eliminates virtually all false-positive cut-offs while still
    # firing on the common "definite end-of-turn" case.
    early_completion_threshold: float = Field(default=0.65, ge=0.05, le=0.99)
    # Reduced VAD ``min_silence_duration_ms`` baseline when smart-turn
    # is active. Smart Turn confirms or rejects the early end-of-speech
    # so we can declare SPEECH_END much sooner and rely on the model
    # to catch trailed-off mid-thought cases. 2026-05-16 latency pass 2
    # dropped this from 500 -> 300 ms after web research showed
    # smart-turn-v3.2 maintains accuracy at the lower bound; combined
    # with ``early_completion_threshold`` raised to 0.65 we get the
    # full 200 ms win on confidently-complete turns while preserving
    # the legacy 500 ms behaviour for medium-confidence turns via the
    # gradient-fire path.
    fast_path_silence_duration_ms: int = Field(default=300, ge=100, le=2000)
    # 2026-05-16 latency pass 2: silence grace appended to
    # ``fast_path_silence_duration_ms`` when the early check returns
    # in the "uncertain" band [completion_threshold, early_completion_threshold).
    # The orchestrator waits this long then re-checks the verdict
    # against ``completion_threshold`` (legacy 0.5). 200 ms takes us
    # from 300 ms fast-path back to the 500 ms legacy baseline, so the
    # medium-confidence case never regresses vs the prior pass.
    medium_grace_ms: int = Field(default=200, ge=0, le=1500)
    # Additional silence required AFTER smart-turn says "incomplete"
    # before the orchestrator finally accepts end-of-turn. This is the
    # second-chance grace window for the user to resume speaking; if
    # they don't, we submit anyway so we don't hang indefinitely. The
    # default 700 ms takes the total from fast_path (500) to roughly
    # the legacy 1200 ms backstop.
    incomplete_extension_ms: int = Field(default=700, ge=0, le=3000)
    # Audio window cap (seconds). Smart Turn V3 was trained on the
    # LAST 8 seconds of speech; longer utterances are truncated head-
    # first by the wrapper. Beyond this duration of contiguous speech,
    # the orchestrator bypasses smart-turn entirely and uses the
    # existing adaptive long-utterance silence bump.
    window_seconds: float = Field(default=8.0, ge=1.0, le=30.0)
    # ONNX runtime intra-op thread count. 1 is the recommended default;
    # the model is small enough that threading overhead exceeds the
    # parallelism win on typical CPUs.
    num_threads: int = Field(default=1, ge=1, le=16)


class VADConfig(_Strict):
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_speech_duration_ms: int = Field(default=250, ge=0)
    min_silence_duration_ms: int = Field(default=500, ge=0)
    window_samples: int = 512
    # 2026-05-11 adaptive end-of-turn: when an utterance has been
    # going for ``long_utterance_threshold_seconds`` of speech, the
    # orchestrator bumps the VAD silence requirement to
    # ``long_utterance_silence_duration_ms`` so a thinking pause
    # mid-sentence doesn't prematurely close a long technical prompt.
    # Short utterances are unaffected -- the threshold only applies
    # once the speaker has been going for a while. Setting threshold
    # to 0 or a very large number disables the adaptive bump.
    long_utterance_threshold_seconds: float = Field(default=8.0, ge=0.0, le=60.0)
    long_utterance_silence_duration_ms: int = Field(default=2400, ge=0)
    # 2026-05-11 follow-up fix: hard ceiling on a single VAD-bounded
    # capture. The orchestrator stops recording when ``elapsed_samples``
    # exceeds this value even if speech is still active -- a guard
    # against unbounded recording (background noise that never resolves
    # to SPEECH_END, a stuck microphone, etc.). A real session hit the
    # legacy 15 s ceiling mid-sentence on a long technical coding ask
    # ("write me a program that converts PDF to Docx..." -- 244 chars,
    # cut off at "a button with a box show"). 30 s comfortably covers
    # detailed one-breath asks while still bounding pathological cases.
    max_utterance_seconds: float = Field(default=30.0, ge=5.0, le=120.0)
    # 2026-05-12 Smart Turn V3 semantic end-of-turn confirmation
    # (CPU-only, ~12 ms inference). See SmartTurnConfig docstring.
    smart_turn: SmartTurnConfig = Field(default_factory=SmartTurnConfig)


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
    # 2026-05-15 latency: default changed 5 -> 1 (greedy decoding).
    # Live bench on the 4070 Ti with small.en int8_float16 shows
    # beam=1 saves ~80 ms median on 5s audio (78 ms vs 157 ms) at
    # negligible WER impact for short English voice queries. Raise to
    # 3 or 5 if downstream WER regresses on noisier audio.
    beam_size: int = Field(default=1, ge=1)
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
    # 2026-05-12 -- Josiefied-Qwen3-8B-abliterated-v1 (Goekdeniz-Guelmez)
    # quantised by mradermacher. Abliterated (refusal vectors removed) +
    # Josiefied fine-tune (improved personality / instruction-following).
    # The voice path keeps Ultron's persona via SOUL.md system prompt;
    # the model's abliterated nature removes content-level refusals while
    # the runtime tool-call validator (src/ultron/safety/) gates the
    # actual capability surface. Q5_K_M strikes the balance between
    # quality and VRAM headroom (~5.5 GB on disk; peak ~10 GB stack vs
    # the 11.5 GB cap on the user's 4070 Ti). No matching 0.8B draft is
    # published, so speculative decoding is off for this preset.
    # NOTE 2026-05-14: retained for swap-back; default rolled forward to
    # the 4B abliterated variant below for VRAM relief on the 4070 Ti.
    "josiefied-qwen3-8b": {
        "model_path": "models/Josiefied-Qwen3-8B-abliterated-v1.Q5_K_M.gguf",
        "n_ctx": 8192,
        "draft_model_path": None,
    },
    # 2026-05-14 -- Josiefied-Qwen3-4B-abliterated-v2 (Goekdeniz-Guelmez)
    # quantised by mradermacher. Base model: Qwen/Qwen3-4B-Instruct-2507.
    # Same abliterated + Josiefied fine-tune lineage as the 8B variant
    # above; preserves the runtime tool-call validator pairing at
    # roughly half the VRAM footprint.
    #
    # Quant choice (2026-05-14 second-pass): **Q4_K_M** (~2.6 GB on disk)
    # instead of Q5_K_M. The Q5_K_M variant ran fine but the user's
    # actual workstation has ~4.7 GB of background GPU usage from
    # Chrome / Discord / EdgeWebView / NVIDIA Broadcast / Cursor.
    # With Q5_K_M's ~3.5 GB VRAM and the rest of the voice stack the
    # total still pushed past 11 GB. Q4_K_M saves another ~500 MB at
    # negligible quality impact (Q4_K_M vs Q5_K_M MMLU delta is
    # <0.5 percentage points on Qwen3-4B per the mradermacher quant
    # ladder annotations); the abliterated content layer is unchanged.
    # The Q5_K_M file is retained on disk for swap-back via the
    # "custom" preset or an explicit model_path override.
    #
    # n_ctx=6144 (down from the 8192 default the other presets use):
    # Q8_0 KV cache at n_ctx=8192 costs ~580 MB for this model; 6144
    # saves ~150 MB without affecting voice typical use (history capped
    # at 4 turns + RAG top-3 + system prompt fits in ~2k tokens; even
    # the heaviest screen-context query with 30+ windows + UIA tree +
    # VLM description lands well under 4k).
    "josiefied-qwen3-4b": {
        "model_path": "models/Josiefied-Qwen3-4B-abliterated-v2.Q4_K_M.gguf",
        "n_ctx": 6144,
        "draft_model_path": None,
    },
    # 2026-05-19 -- Gemma 3 4B abliterated (mradermacher quants of the
    # Goekdeniz-Guelmez Josiefied abliterated fine-tune over Google's
    # gemma-3-4b-it base model). IFEval scores 90.2 vs Qwen3's pattern
    # of length miscalibration (over-explains factual queries,
    # under-delivers on procedural depth). Designed as a candidate
    # daily-use swap whose stronger instruction-following directly
    # addresses the duck/cake verbosity issue documented in the
    # 2026-05-19 design pass. Pairs with the Gemma 3 1B IT draft for
    # speculative decoding (same tokenizer/vocab so acceptance rate
    # holds; ~60-75% on conversational text).
    #
    # NOT default. Default stays josiefied-qwen3-4b until the GGUFs
    # are downloaded + a week of A/B against the eval harness. Swap
    # via ``python scripts/swap_llm_preset.py gemma-3-4b-abliterated``
    # once weights are present on disk.
    "gemma-3-4b-abliterated": {
        # Filenames here MUST match what ``scripts/download_models.py``
        # writes -- otherwise ``swap_llm_preset.py``'s
        # _validate_preset_files refuses the swap with
        # "preset files missing".
        #
        # Main 4B from mradermacher uses dot separator
        # (``{name}.Q4_K_M.gguf``). The 1B draft comes from bartowski's
        # ``google_gemma-3-1b-it-GGUF`` repo (note the ``google_``
        # prefix in both the repo and the filename -- bartowski's
        # plain ``gemma-3-1b-it-GGUF`` slug is a 404). The 1B and 4B
        # share the same tokenizer so speculative decoding accepts
        # ~60-75% of drafted tokens on conversational text.
        "model_path": "models/gemma-3-4b-it-abliterated.Q4_K_M.gguf",
        "n_ctx": 4096,
        "draft_model_path": "models/google_gemma-3-1b-it-Q4_K_M.gguf",
    },
    # 2026-05-19 -- Llama 3.2 3B abliterated (mradermacher quants of
    # Meta's Llama-3.2-3B-Instruct base with refusal vectors removed).
    # Designed as the gaming-mode preset: smaller VRAM footprint
    # (~1.9 GB Q4 vs 2.5 GB for Qwen3-4B), naturally brief
    # conversational tone (field-tested in voice pipelines per the
    # 2026-05-19 design conversation), and Llama 3.2 1B IT as a
    # tokenizer-compatible draft for speculative decoding. Tool-call
    # discipline is weaker than Qwen but acceptable in gaming mode
    # where OpenClaw orchestration is disabled.
    #
    # NOT default. Intended as a swap target when ``MODEL_SWITCH``
    # voice intent or ``swap_llm_preset.py`` engages gaming mode.
    # n_ctx=2048 because gaming-channel utterances are short and the
    # smaller KV cache frees ~400 MB for Valorant + OBS headroom.
    "llama-3.2-3b-abliterated": {
        # See gemma note above on naming conventions -- main from
        # mradermacher (dot), draft from bartowski (hyphen).
        "model_path": "models/Llama-3.2-3B-Instruct-abliterated.Q4_K_M.gguf",
        "n_ctx": 2048,
        "draft_model_path": "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    },
}


class LLMConfig(_Strict):
    # Pinned to llama_cpp per feedback_llm_runtime_decision.md (2026-05-08).
    provider: Literal["llama_cpp"] = "llama_cpp"
    # 4B optimization plan Stage A — model preset.
    #   "qwen3.5-9b"        — pre-4B-plan default; 9B GGUF + n_ctx=8192,
    #                         no draft model. Retained for swap-back.
    #   "qwen3.5-4b"        — 4B target + 0.8B draft for speculative
    #                         decoding, n_ctx=8192. Default through 2026-05-11.
    #   "josiefied-qwen3-8b" — Goekdeniz-Guelmez Josiefied + abliterated
    #                         Qwen3-8B Q5_K_M (default 2026-05-12 ->
    #                         2026-05-13). Retained for swap-back.
    #   "josiefied-qwen3-4b" — Goekdeniz-Guelmez Josiefied + abliterated
    #                         Qwen3-4B-v2 Q5_K_M; new default 2026-05-14.
    #                         Same abliterated lineage as the 8B above
    #                         at ~half the VRAM. Pairs with the runtime
    #                         tool-call validator in src/ultron/safety/.
    #                         No paired abliterated draft published, so
    #                         speculative decoding stays off for now.
    #   "custom"            — no auto-resolution; raw model_path / n_ctx /
    #                         draft_model_path fields are used as-is. For
    #                         tests and ad-hoc model swaps.
    # Preset defaults only fill in fields the user did NOT explicitly
    # set in YAML — see ``_apply_preset``.
    preset: Literal[
        "qwen3.5-9b",
        "qwen3.5-4b",
        "josiefied-qwen3-8b",
        "josiefied-qwen3-4b",
        # 2026-05-19: added but NOT default. See LLM_PRESETS comments
        # for swap-readiness contract (GGUFs must be on disk first).
        "gemma-3-4b-abliterated",
        "llama-3.2-3b-abliterated",
        "custom",
    ] = "josiefied-qwen3-4b"
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
    # 2026-05-15 latency: explicit n_batch / n_ubatch tuning. Defaults
    # are llama.cpp's own (n_batch=512, n_ubatch=512 in 0.3.22). For
    # voice-length prompts (1-2 KB context) on this 4070 Ti, sweeping
    # showed n_ubatch=256 trims ~30-80 ms of prefill TTFT vs the
    # default; n_batch=1024 helps if context grows. Set to None to
    # inherit llama.cpp's defaults (safest fallback on unknown
    # hardware). Range bounds match llama.cpp's internal validation.
    n_batch: Optional[int] = Field(default=None, ge=1, le=32768)
    n_ubatch: Optional[int] = Field(default=None, ge=1, le=32768)
    # 2026-05-16 latency pass 2: prefix KV cache. When > 0, the in-process
    # Llama instance gets a ``LlamaRAMCache`` attached at init so completed
    # session KV state is stored in host RAM keyed by the longest-common-
    # prefix of the token sequence. Subsequent calls with a shared prefix
    # (the stable system prompt + prior turns) restore the cached state
    # instead of re-evaluating those tokens.
    #
    # **DEFAULT 0 (disabled) after live bench on 4070 Ti + josiefied-4B-
    # Q4_K_M showed a ~15 ms TTFT REGRESSION vs cache-off.** llama.cpp's
    # internal KV cache already handles intra-session prefix reuse (which
    # is what every voice turn within one engine instance does); the
    # explicit LlamaRAMCache adds a ``load_state`` memcpy that exceeds
    # the eval savings on our short 280-token system prompts. The knob
    # stays in place so operators with different workloads (longer
    # prompts, cross-session reloads, slower-prefill models) can opt in.
    # See ``scripts/bench_llm_prefix_cache.py`` and the
    # ``baselines.json:llm_prefix_cache_bench`` block for the measurement.
    # Set to e.g. 2147483648 (2 GiB) to re-enable. Host RAM only -- the
    # cache does NOT touch the 11.5 GB VRAM budget.
    prefix_cache_ram_bytes: int = Field(default=0, ge=0)
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
    # 2026-05-19 Track 2: when True, single-query dense + sparse
    # encoding runs on a 2-worker ThreadPoolExecutor so the two
    # ONNX inferences overlap (~5-15 ms saved per retrieve call on
    # CPU). Default False -- byte-for-byte legacy serial path until
    # operators opt in.
    parallel_query_embedding: bool = False


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
    # 2026-05-19 Issue 2 fix: suppress RAG retrieval entirely for
    # short greetings / acks. The bge-small embedder cosine-matches
    # these to off-topic stored memory above the 0.6 ``rag_min_relevance``
    # cutoff (live session 2026-05-19: 'Say hello.' returned a
    # response about Salesforce Agentforce pricing). Net-benefit
    # default ON; opt out by setting False if you specifically want
    # greetings to draw from memory for some downstream feature.
    skip_rag_for_short_queries: bool = True


class MemoryRankingConfig(_Strict):
    """V1-gap A2: weighted blend of RRF + recency + surprise - redundancy.

    2026-05-19 Track 1h: extended with topic_match_weight and
    discourse_match_weight to factor in the new payload metadata
    populated by Tracks 1a (topical chunking) and 1b (discourse
    tagging). Both default to 0.0 so the legacy retrieval path is
    byte-for-byte unchanged until operators tune them up against
    referential-query eval rows.
    """

    rrf_weight: float = Field(default=1.0, ge=0.0)
    recency_weight: float = Field(default=0.2, ge=0.0)
    recency_half_life_days: float = Field(default=7.0, gt=0.0)
    surprise_weight: float = Field(default=0.15, ge=0.0)
    redundancy_weight: float = Field(default=0.3, ge=0.0)
    # Track 1h additions -- default 0.0 (no behaviour change).
    topic_match_weight: float = Field(default=0.0, ge=0.0)
    discourse_match_weight: float = Field(default=0.0, ge=0.0)


class TopicalChunkingConfig(_Strict):
    """2026-05-19 Track 1a: topical chunking on the memory write path.

    When ``enabled`` is True, ``ConversationMemory.add`` consults a
    :class:`TopicTracker` to detect topic boundaries via cosine
    similarity between consecutive turn embeddings. Each turn's
    payload gets a ``topic_id`` for use by the ranking layer
    (Track 1h).

    Default OFF -- the metadata fields ship by default but the
    boundary detection only runs when the flag is set. With the flag
    off, turns get no topic_id; the ranking layer's topic_match_score
    returns 0.0 for those payloads which is the byte-for-byte legacy
    behaviour.
    """

    enabled: bool = False
    boundary_similarity_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    idle_timeout_seconds: float = Field(default=300.0, ge=0.0)


class DiscourseTaggingConfig(_Strict):
    """2026-05-19 Track 1b: discourse-type tagging on the memory write
    path. Adds a 6-way classification (QUESTION / STATEMENT /
    DECISION / CLARIFICATION_REQUEST / ACKNOWLEDGMENT / TOPIC_SHIFT)
    to each turn's payload via rule layer + optional embedding-
    centroid fallback.

    Default OFF. With the flag off, no discourse_type metadata is
    attached and the ranking layer's discourse_match_score returns
    0.0 for those payloads (byte-for-byte legacy behaviour).

    When ``use_embedding_fallback`` is True and the rule layer
    returns None, the classifier embeds the turn via the existing
    HybridEmbedder and dispatches to the nearest precomputed
    centroid. Adds ~5-20 ms CPU per write on the rule-miss path.
    """

    enabled: bool = False
    use_embedding_fallback: bool = True
    centroid_confidence_floor: float = Field(default=0.25, ge=0.0, le=1.0)


class BackgroundSummarizerConfig(_Strict):
    """2026-05-19 Tracks 1c + 1d + 1e: periodic LLM-driven summary
    + structured fact extraction.

    When enabled, the orchestrator's idle hook calls
    :class:`ultron.memory.background_summarizer.BackgroundSummarizer.maybe_summarize`
    when conversational activity has been quiet for at least
    ``idle_threshold_seconds`` AND at least ``cadence_turns`` new
    turns have accumulated since the last summary.

    The pass invokes the SAME in-process LLM as the foreground voice
    path (lock-serialized). No additional model load, no VRAM cost
    -- the only resource cost is the LLM call itself (~1-2 s on
    Qwen3-4B), gated to idle windows.

    Storage hook writes the summary + extracted facts as new Qdrant
    entries (``type=session_summary | fact | decision | preference``).

    Default OFF. The orchestrator never calls the summarizer until
    the flag is on, so legacy behaviour is byte-for-byte unchanged.
    """

    enabled: bool = False
    cadence_turns: int = Field(default=10, ge=1, le=100)
    min_turns: int = Field(default=3, ge=1, le=100)
    idle_threshold_seconds: float = Field(default=30.0, ge=0.0)
    # 2026-05-19 orchestrator integration: where the storage hook
    # appends one JSON line per :class:`SummaryResult`. The default
    # lives under ``data/`` (gitignored). A null / empty value disables
    # the JSONL write but leaves the summarizer itself running --
    # callers who want to wire a custom store_fn can do so by patching
    # the orchestrator's :meth:`_build_default_background_summary_store`.
    output_path: str = "data/background_summaries.jsonl"


class MemoryConfig(_Strict):
    enabled: bool = True
    jsonl_legacy_path: str = "data/memory.jsonl"
    recent_turns: int = Field(default=20, ge=0)
    rag_top_k: int = Field(default=5, ge=0)
    rag_exclude_recent: int = Field(default=20, ge=0)
    facts_top_k: int = Field(default=3, ge=0)
    write_queue_maxsize: int = Field(default=256, ge=1)
    # 2026-05-09 nuanced-retrieval pass: CAP on recent-turn history fed
    # to the LLM as conversation context. The in-process recent-turns
    # CACHE size remains ``recent_turns: 20`` (used by ``retrieve``'s
    # exclude_recent + the public ``recent()`` API). This new field
    # caps how many of those land in the LLM message list per call.
    # Smaller = less topic-bleed when the user pivots topics; larger
    # = more conversational continuity for follow-ups. 4 = 2 user +
    # assistant pairs, enough for natural follow-ups but not a wall
    # of stale context.
    history_turns_for_llm: int = Field(default=4, ge=0)
    # 2026-05-09 nuanced-retrieval pass: minimum cosine similarity
    # between the user query embedding and a candidate turn's
    # embedding for the candidate to be included in the LLM's RAG
    # context. Below this, the candidate is treated as irrelevant
    # noise and filtered out entirely. Cosine sim is in [0, 1] for
    # text embeddings.
    #
    # 0.6 was tuned empirically with bge-small INT8 (the production
    # embedder) against the seeded conversation corpus -- truly
    # off-topic content peaks around 0.55-0.57 (e.g. apex-predator
    # chatter cosine'd against a Mariana-Trench query topped at
    # 0.567), while genuinely relevant matches score 0.7-0.95.
    # Set 0.0 to disable filtering (pre-2026-05-09 legacy behaviour).
    rag_min_relevance: float = Field(default=0.6, ge=0.0, le=1.0)
    # V1-gap A2.
    retrieval: MemoryRetrievalConfig = Field(default_factory=MemoryRetrievalConfig)
    ranking: MemoryRankingConfig = Field(default_factory=MemoryRankingConfig)
    # 2026-05-19 Track 1a -- topical chunking write-side metadata.
    topical_chunking: TopicalChunkingConfig = Field(
        default_factory=TopicalChunkingConfig,
    )
    # 2026-05-19 Track 1b -- discourse-type tagging write-side metadata.
    discourse_tagging: DiscourseTaggingConfig = Field(
        default_factory=DiscourseTaggingConfig,
    )
    # 2026-05-19 Tracks 1c + 1d + 1e -- periodic background summary
    # + structured fact extraction (idle-gated, default OFF).
    background_summary: BackgroundSummarizerConfig = Field(
        default_factory=BackgroundSummarizerConfig,
    )


class BraveConfig(_Strict):
    endpoint: str = "https://api.search.brave.com/res/v1/web/search"
    count: int = Field(default=5, ge=1)
    timeout_seconds: float = 8.0
    rate_limit_seconds: float = 2.0


class JinaConfig(_Strict):
    endpoint: str = "https://r.jina.ai/"
    # Per-fetch HTTP timeout. Reduced from 15.0 -> 6.0 (2026-05-09 latency
    # fix) so a single pathological page can't dominate the search-path
    # latency. The collective deadline below caps the executor's wait
    # across ALL parallel fetches; per-fetch timeout is the secondary
    # ceiling the underlying request honours.
    timeout_seconds: float = 6.0
    # How many ranked snippets get full-text fetches. Reduced from 3 ->
    # 2 (2026-05-09 latency fix) -- two sources is enough context for
    # the LLM to answer most queries, and dropping the third halves the
    # tail-latency exposure on the slowest page.
    max_fetch: int = Field(default=2, ge=0)
    max_bytes: int = Field(default=200_000, ge=0)
    # Collective deadline for ALL parallel Jina fetches launched by a
    # single executor.run() call. After this many seconds, any fetch
    # still in flight is abandoned (its result is dropped; the source
    # falls back to snippet-only). Independent of per-fetch
    # ``timeout_seconds`` -- this is the executor-side cap that
    # protects the voice path even when one URL is slow but not
    # timing out at the HTTP layer. (2026-05-09 latency fix.)
    collective_deadline_seconds: float = Field(default=6.0, ge=0.0)


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
    # 2026-05-11 false-positive guard: flan-t5-small zero-shot
    # saturates around 0.75 confidence on borderline utterances --
    # third-person narration about Ultron was getting routed to
    # ADDRESSED at exactly that level. Require ``>=`` this confidence
    # before treating a zero-shot YES as direct address; below the
    # bar falls through to ``default_uncertain_to_not_addressed``.
    # Set to 0.0 to disable the gate (back-compat).
    zero_shot_addressed_min_confidence: float = Field(default=0.80, ge=0.0, le=1.0)
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


class CodingAstMetadataConfig(_Strict):
    """2026-05-19 Tracks 1f + 1g: AST-based structural metadata
    extraction from coding-task FILE_CHANGE events.

    When ``enabled`` is True AND ``syntax_check_on_file_change`` is
    True, the :class:`CodingTaskRunner` registers a FILE_CHANGE
    listener that parses each created / modified Python file via
    :mod:`ultron.coding.ast_metadata` and emits an audit-log row
    (``ast_syntax_ok`` or ``ast_syntax_failure``) so completion
    narration can fact-check the success claim.

    Default OFF: the AST parse adds ~5-50 ms per FILE_CHANGE on .py
    files (invisible inside the coding-task latency budget which is
    measured in seconds, but the metadata storage decision is
    operator-led).

    The audit signal is the primary deliverable -- if Claude Code
    writes broken Python and reports success, the listener catches
    it. The structural metadata (functions_defined, imports, etc.)
    is captured for future use in code-context retrieval but not
    consumed yet.
    """

    enabled: bool = False
    syntax_check_on_file_change: bool = True
    attach_metadata_to_audit: bool = True


class CodingGoalAnchorsConfig(_Strict):
    """E2 goal-anchor planning (Phase 0+1 build, 2026-05-18+).

    When enabled, the :class:`CodingTaskRunner` decomposes incoming
    task prompts into named milestones (anchors) with per-anchor
    token budgets. As USAGE events arrive from Claude Code, the
    runner attributes tokens to the active anchor and surfaces voice
    narration when an anchor completes, when it's near its budget,
    and when the next anchor begins.

    Resume support: when a task is paused mid-plan (budget exhausted,
    user cancelled, etc.), the runner's :meth:`send_followup` can
    prepend the next unfinished anchor's description to the
    follow-up prompt so Claude Code resumes at the right milestone
    instead of restarting from scratch.

    Default OFF: the narration adds extra voice turns mid-task; the
    operator opts in.
    """

    enabled: bool = False
    min_anchors: int = Field(default=1, ge=1, le=10)
    max_anchors: int = Field(default=6, ge=1, le=10)
    # Per-anchor warning threshold. ``0.8`` matches the existing
    # session-budget warning convention.
    warn_threshold: float = Field(default=0.8, ge=0.1, le=1.0)
    # When True, ``send_followup`` prepends the next unfinished
    # anchor's description to the follow-up prompt.
    resume_prepend_next_anchor: bool = True


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
    # E2 goal-anchor planning (Phase 0+1 build) -- off by default.
    goal_anchors: CodingGoalAnchorsConfig = Field(
        default_factory=CodingGoalAnchorsConfig,
    )
    # 2026-05-19 Tracks 1f + 1g -- AST-based syntax verification on
    # FILE_CHANGE events. Default OFF.
    ast_metadata: CodingAstMetadataConfig = Field(
        default_factory=CodingAstMetadataConfig,
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
    # 2026-05-11 token-efficiency: voice-dispatched coding tasks used
    # to hardcode ``require_testing=True``, which prepended a "MUST
    # write tests, run them, fix failures, re-run" discipline preamble
    # to the Claude prompt. For ad-hoc voice utility requests ("write
    # me a PDF to DOCX converter") this triples-to-quintuples the
    # token spend -- Claude writes the script, then writes tests, then
    # fixes import errors, then re-runs, etc. A direct Claude Code
    # invocation of the same ask completes in ~2 k tokens; the
    # testing-mandated voice path was burning 130 k+. Default off
    # gets voice asks back to the natural cadence -- if the user
    # actually wants tests they can say so in their request ("with
    # unit tests") or flip this. Existing non-voice callers (the
    # coordinator's correction loop in runner.py) pass require_testing
    # explicitly and aren't affected.
    voice_task_require_testing: bool = False


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


class XttsV3Config(_Strict):
    """Configuration for the XTTS v2 + v3 Ultron filter TTS engine.

    Selected when ``tts.engine == "xtts_v3"``. The engine spawns the
    XTTS HTTP server as a subprocess in its own isolated venv (so
    Coqui TTS's transformers / hydra / omegaconf pins don't conflict
    with what fairseq / RVC needs in the main venv).
    """
    # Path to the Python interpreter inside the isolated XTTS venv.
    # Defaults to the layout established during the 2026-05-10 voice
    # swap: a ``.venv-xtts`` next to the audio prep work.
    server_python: str = "ultronVoiceAudio/.venv-xtts/Scripts/python.exe"
    # The XTTS HTTP server entry script. Lives outside ``src/ultron/``
    # because it has to import Coqui's TTS package which only exists
    # in the isolated venv.
    server_script: str = "ultronVoiceAudio/scripts/xtts_server.py"
    # Reference WAV used as the speaker conditioning source. The
    # cleaned mono Ultron reference is the production default.
    reference_audio: str = "ultronVoiceAudio/Ultron_vocals_mono_v1.wav"
    # Bind details for the local-only HTTP server.
    host: str = "127.0.0.1"
    # ``null`` -> the engine picks a free port at startup.
    port: Optional[int] = None
    # v3_heavy is the user-locked production preset (2026-05-10).
    # Other valid presets: ``v1_subtle``, ``v2_medium``.
    filter_preset: str = "v3_heavy"
    # Trailing silence padded onto each synthesised clip BEFORE the
    # filter, so the reverb tail can decay into it without being
    # clipped at the buffer end. Runtime default 200 ms (audible
    # portion of the v3 reverb); offline standalone samples use
    # 500 ms for full decay.
    filter_tail_silence_ms: float = Field(default=200.0, ge=0.0, le=2000.0)
    # 2026-05-11 cadence tune: XTTS native default is 1.0; production
    # sits at 1.15 to make speech ~15% faster without slurring. Passed
    # to ``model.inference_stream(speed=...)`` on the server side, which
    # adjusts the GPT duration tokens BEFORE waveform decoding -- so
    # the v3 pedalboard filter (pitch shift / reverb / etc.) is
    # untouched and processes the shorter audio buffer identically.
    # Below ~0.7 sounds drawn out; above ~1.4 starts to slur consonants.
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    # 2026-05-12 phantom-token mitigation: XTTS-v2's GPT duration head
    # is stochastic and occasionally produces fragmentary syllables at
    # sentence boundaries. The server-library default temperature of
    # 0.75 is on the high side -- 0.65 keeps the speaker character bit-
    # identical (timbre is set by the locked speaker embedding + the
    # v3 filter chain) while sharpening the duration-token distribution
    # so phantom-token rate drops dramatically. Passed in the HTTP body
    # to the server, which forwards it to ``model.inference_stream``.
    # Range [0.4, 1.0] -- below 0.4 prosody collapses; above 1.0 the
    # model becomes unstable.
    temperature: float = Field(default=0.65, ge=0.4, le=1.0)
    # 2026-05-12 phantom-token mitigation, defence in depth: even with
    # temperature lowered, a small residual rate of phantom tokens can
    # slip through. This client-side post-process detects the specific
    # XTTS phantom-tail signature -- a short audio fragment isolated by
    # silence on both sides at the end of a synthesised clip -- and
    # trims everything after the last sustained-speech region. Runs
    # BEFORE the v3 filter so the reverb tail decays normally into its
    # tail_silence_ms padding. When no phantom pattern is detected the
    # audio passes through unchanged. Set false to disable (e.g. for
    # A/B comparison against the unfiltered output).
    phantom_tail_trim_enabled: bool = True
    # The RMS threshold below which a window counts as silence when
    # detecting the phantom pattern. 0.005 corresponds to roughly
    # -46 dBFS which is comfortably below typical XTTS speech RMS
    # (-15 to -25 dBFS post-normalisation) and above the inherent
    # noise floor of the generation.
    phantom_tail_silence_threshold: float = Field(default=0.005, ge=0.0001, le=0.05)
    # A trailing audio event shorter than this is a phantom candidate;
    # longer events are legitimate tail-end speech. 200 ms is the
    # empirical ceiling -- XTTS phantoms are typically 20-150 ms.
    phantom_tail_max_event_ms: float = Field(default=200.0, ge=50.0, le=500.0)
    # The minimum silent gap that must precede a phantom-candidate
    # event for it to qualify as a phantom (rather than a brief pause
    # between two legitimate utterances). 150 ms is well below normal
    # inter-word pauses and well above mid-word micro-pauses.
    phantom_tail_min_lead_silence_ms: float = Field(default=150.0, ge=50.0, le=500.0)
    # 2026-05-19 Issue 1 fix: cap per-synth-call text length so a
    # single sentence can't overflow the XTTS-v2 GPT 4096-audio-token
    # context window.
    #
    # 2026-05-19 round 4 retune: bumped from 240 -> 600. The original
    # 240 was too aggressive -- it broke ordinary multi-clause
    # sentences into 3-4 fragments, each picking up the v3 filter's
    # ``tail_silence_ms=200`` padding, producing audibly jagged
    # pacing with random mid-sentence pauses (live session feedback:
    # "horrible pacing, pauses randomly between words"). 600 chars
    # at typical English token-density (~1.5 audio tokens per char)
    # is ~900 audio tokens -- well under the 4096 cap with plenty of
    # margin for URL-laden content (URL strip in normalize_text_for_tts
    # already pulls the worst offenders). Sentences longer than 600
    # chars still get sub-split, just much less often.
    max_chars_per_synth_call: int = Field(default=600, ge=80, le=2000)


class KokoroConfig(_Strict):
    """2026-05-19 Track 5: Kokoro TTS engine configuration.

    Selected when ``tts.engine == "kokoro"``. The engine loads the
    StyleTTS2 + ISTFTNet weights from ``model_path`` lazily on first
    inference. With ``apply_runtime_filter`` enabled, the v3
    pedalboard chain runs on Kokoro output at runtime (useful pre-
    fine-tune so the voice character matches XTTS). Post-fine-tune,
    the filter character is baked into the model weights and this
    flag stays False.
    """

    # Directory containing Kokoro weights + voice tensors.
    model_path: str = "models/kokoro"
    # Voice tensor name. Production target is a fine-tuned Ultron
    # voice from the synth corpus; the stock ``af_alloy`` boots the
    # engine before the fine-tune lands.
    voice: str = "af_alloy"
    # CPU is the production target -- Kokoro is fast enough on CPU
    # and keeps the GPU free for LLM + Whisper. Set "cuda" for ~3x
    # faster synthesis when the user has the VRAM budget.
    device: str = Field(default="cpu", pattern="^(cpu|cuda)$")
    # Speech-rate multiplier; 1.0 = native cadence.
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    # Pre-fine-tune: run the v3 Ultron pedalboard filter on Kokoro
    # output to match XTTS voice character. Disable post-fine-tune
    # (the filter character will be baked into the weights).
    apply_runtime_filter: bool = False
    filter_preset: str = "v3_heavy"


class TTSConfig(_Strict):
    # 2026-05-10 voice swap: ``"piper_rvc"`` is the legacy stack
    # (Piper voice + RVC timbre transfer); ``"xtts_v3"`` is the new
    # XTTS v2 streaming + v3 filter stack. 2026-05-19 Track 5:
    # ``"kokoro"`` is the new lightweight StyleTTS2 + ISTFTNet
    # engine, intended as the post-fine-tune target. Switching
    # engines requires a process restart (the chosen engine is
    # loaded once at orchestrator construction).
    engine: str = Field(default="piper_rvc", pattern="^(piper_rvc|xtts_v3|kokoro)$")
    piper_voice_path: str = "models/piper/en_US-ryan-medium.onnx"
    piper_voice_config_path: str = "models/piper/en_US-ryan-medium.onnx.json"
    output_sample_rate: int = 22050
    sentence_flush_chars: str = ".!?\n"
    inter_sentence_pause_ms: int = Field(default=250, ge=0)
    piper_length_scale: float = Field(default=1.15, ge=0.1)
    pause_ms: int = Field(default=180, ge=0)
    edge_fade_ms: int = Field(default=4, ge=0)
    rvc: RVCConfig = Field(default_factory=RVCConfig)
    xtts_v3: XttsV3Config = Field(default_factory=XttsV3Config)
    # 2026-05-19 Track 5 -- Kokoro engine config (used when
    # tts.engine == "kokoro").
    kokoro: KokoroConfig = Field(default_factory=KokoroConfig)
    # 2026-05-09 latency hot-fix: split Piper and RVC into two separate
    # worker stages connected by a bounded queue. With the legacy
    # single-worker shape, sentence N+1's Piper synthesis only began
    # AFTER sentence N's RVC finished. With the split, Piper N+1 runs
    # in parallel with RVC N, saving ~Piper-time (~50 ms) per
    # subsequent sentence on multi-sentence responses. Voice output is
    # bit-identical because Piper produces the same buffer that RVC
    # then converts; only the timing of the stages overlaps.
    # Set false to revert to the legacy single-worker pipeline.
    pipeline_parallel_enabled: bool = True
    # 2026-05-09 latency hot-fix: open the audio output stream
    # speculatively at the expected RVC sample rate while Piper+RVC of
    # the first sentence is still synthesising, instead of waiting for
    # the first clip to arrive. The existing sample-rate-mismatch
    # close-and-reopen path covers the rare case where actual output
    # rate differs from the speculative one. Saves ~20-30 ms first-
    # sentence latency.
    speculative_stream_open_enabled: bool = True
    # Expected output sample rate for speculative open. The Ultron RVC
    # model outputs 48000 Hz (verified against live captures);
    # Piper-only stacks (``rvc.enabled=false``) should set this to
    # match ``output_sample_rate`` (22050 by default). Mismatch falls
    # back to the close-and-reopen path (~50-100 ms wasted per turn);
    # keep this aligned to the actual first-clip rate.
    speculative_stream_sample_rate: int = Field(default=48000, ge=8000)
    # 2026-05-09 latency hot-fix: pass ``latency='low'`` to
    # ``sd.OutputStream`` so PortAudio asks the host audio API for the
    # smallest acceptable buffer. Saves 30-100 ms of OS-level audio
    # buffering on most Windows systems. Falls back gracefully on
    # platforms / devices that don't honour the hint.
    output_low_latency_mode: bool = True


class LoggingConfig(_Strict):
    file: str = "logs/ultron.log"
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"


class RoutingClassifierConfig(_Strict):
    rule_based_first: bool = True
    llm_fallback_enabled: bool = True
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class AmbiguityBandClarificationConfig(_Strict):
    """Confidence-band clarification (Phase 0+1 build, 2026-05-18+).

    When a routing classifier returns an intent with confidence inside
    ``[band_low, band_high)``, the orchestrator can be configured to
    ask one clarifying question via :class:`IntentDisambiguator`
    instead of executing the ambiguous verdict directly.

    The pure predicate :func:`ultron.openclaw_routing.ambiguity.should_clarify`
    consumes these knobs; the actual orchestrator wiring is a follow-up
    behavioural change (default OFF preserves today's flow).
    """

    enabled: bool = False
    band_low: float = Field(default=0.4, ge=0.0, le=1.0)
    band_high: float = Field(default=0.65, ge=0.0, le=1.0)


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
    # Phase 0+1 build (2026-05-18+): confidence-band clarification knobs.
    ambiguity_band_clarification: AmbiguityBandClarificationConfig = Field(
        default_factory=AmbiguityBandClarificationConfig
    )
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


class SafetyConfig(_Strict):
    """Runtime tool-call validator configuration (Phase 2 -- 2026-05-12).

    Paired with the abliterated default LLM (Josiefied-Qwen3-8B,
    ``llm.preset = "josiefied-qwen3-8b"``). The validator gates the
    actual capability surface even when the model is willing to
    attempt anything at the content level.

    Master switch: ``enabled``. Off = validator is a permissive
    no-op (every tool call returns ALLOW). When false the model has
    no policy enforcement at the runtime layer; this should only be
    used for testing / one-off troubleshooting.

    Per-rule toggles live in ``rules`` -- e.g.
    ``rules: {"K1": false, "A4": true}``. Missing keys default to
    True (rules are on unless explicitly disabled). The user's
    2026-05-12 restriction-list rule IDs (K1-K10, A1-A12, B1-B9,
    C1-C12, D1-D17, E1-E8, F1-F9, G1-G5, H1-H12, I1-I7, J1-J9,
    K1-K10, L1-L4, M1-M12, N1-N6, O1-O8, P1-P5, Q1-Q5, R1-R7,
    S1-S5) are the addressable units. Phase 2 ships K1-K10 only;
    Phases 3-5 add the rest.

    Sandbox + protected-path overrides let operators extend the
    in-code defaults (e.g. mark an additional training-data
    directory as protected). Paths are PROJECT_ROOT-relative.
    """

    # Master switch. False = permissive no-op for every tool call.
    enabled: bool = True
    # Per-rule enable map. Missing keys default to True.
    rules: dict[str, bool] = Field(default_factory=dict)
    # Project-root-relative directory roots where destructive ops are
    # allowed. Defaults to data/sandbox if empty.
    sandbox_roots: list[str] = Field(default_factory=list)
    # Additional protected files (on top of the built-in K-list).
    extra_protected_files: list[str] = Field(default_factory=list)
    # Additional protected directory trees (on top of the built-in
    # K-list).
    extra_protected_dirs: list[str] = Field(default_factory=list)
    # Cap-1 screen-context capture cache directory (Phase 4 wiring).
    # PROJECT_ROOT-relative; the Cap-1 OUT-gate blocks writes of
    # captured frames outside this directory.
    screen_cache_dir: Optional[str] = None
    # Hostnames the model is allowed to reach via outbound network
    # calls (Brave, Jina, Anthropic API). Used by Categories I / J
    # in Phase 4.
    approved_outbound_apis: list[str] = Field(
        default_factory=lambda: ["api.search.brave.com", "r.jina.ai", "api.anthropic.com"]
    )
    # Audit log path. Relative to PROJECT_ROOT.
    audit_log_path: str = "logs/safety_audit.jsonl"


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
    # 2026-05-12 Phase 2 -- runtime tool-call validator (paired with the
    # abliterated Josiefied Qwen3-8B default LLM).
    safety: "SafetyConfig" = Field(default_factory=lambda: SafetyConfig())


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
