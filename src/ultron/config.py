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

import logging
import os
import re
from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional

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
    # 2026-05-21 frontier-enhancement Item 5 -- STT engine selector.
    # ``auto`` (the default): use Parakeet TDT if NVIDIA NeMo is
    # installed in the venv, fall back to Whisper otherwise. This
    # gives the operator a one-step opt-in to Parakeet (just
    # ``pip install nemo_toolkit[asr]``) without breaking the
    # out-of-box experience on fresh installs.
    #
    # ``whisper`` (explicit): force Whisper regardless of what's
    # installed. The clean swap-back path if Parakeet misbehaves
    # ("set ``stt.engine: whisper`` and the issue is gone").
    #
    # ``parakeet`` (explicit): force Parakeet; raise if NeMo isn't
    # available. Use this to verify the swap is actually taking
    # effect in your environment.
    #
    # *** SUSPECT THIS FLAG FIRST IF VOICE QUALITY REGRESSES AFTER
    # 2026-05-21. *** The Parakeet swap is a brand-new path; if
    # the transcription suddenly mishears proper nouns, accents, or
    # technical jargon that used to work, try ``stt.engine: whisper``
    # to confirm it's the engine before chasing other causes.
    engine: Literal["auto", "whisper", "parakeet", "moonshine"] = "auto"
    # 2026-05-22 -- dual-STT swap for gaming mode. When set to any
    # value other than "" or the primary ``engine``, the orchestrator
    # loads BOTH engines at startup and the gaming-mode engage callback
    # swaps ``self.stt`` to this engine while the game is running.
    # Disengage restores the primary engine.
    #
    # Default ``moonshine``: when paired with ``engine: parakeet``
    # (Parakeet on CUDA for standby, ~700 MB VRAM), gaming mode flips
    # to Moonshine on CPU (~700 MB RAM, 0 VRAM) -- freeing the VRAM
    # for the game. Set to "" to disable the dual-engine setup
    # entirely (gaming mode then only flips Kokoro + VLM + LLM, not
    # the STT engine).
    gaming_engine: Literal["", "whisper", "parakeet", "moonshine"] = "moonshine"
    # --- Whisper-side config (used when engine resolves to whisper) ---
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
    # --- Parakeet-side config (used when engine resolves to parakeet) ---
    # 2026-05-21 frontier-enhancement Item 5. Parakeet TDT is
    # NVIDIA's RNN-Transducer ASR model -- streaming-friendly,
    # ~RTFx 2000+ on consumer GPUs (Whisper base is ~100). The
    # 0.6B-v3 variant is the production sweet spot in 2026.
    #
    # **Isolation pattern (DEFAULT):** NeMo is installed in an
    # isolated venv at ``ultronVoiceAudio/.venv-parakeet/`` to
    # avoid breaking the main venv's pinned numpy<2.0,
    # transformers 4.41.2, librosa 0.9.1, hydra<1.1 (all used by
    # the rest of the voice stack). The ``ParakeetEngine`` client
    # in the main venv talks to a long-running FastAPI server
    # started from .venv-parakeet via HTTP at
    # ``http://127.0.0.1:8771``. Mirrors the XTTS pattern at
    # ``.venv-xtts`` + ``ultronVoiceAudio/scripts/xtts_server.py``.
    #
    # To set up: run ``python -m venv ultronVoiceAudio/.venv-parakeet``
    # then in that venv ``pip install nemo_toolkit[asr] fastapi
    # uvicorn soundfile`` (the parakeet_server.py only needs those
    # plus NeMo's own transitives).
    #
    # **Easy reversibility (the "variable switch"):** flip
    # ``stt.engine: whisper`` in config.yaml to force Whisper. The
    # Parakeet server stays installed but is never contacted; the
    # main venv stays clean.
    parakeet_model: str = "nvidia/parakeet-tdt-0.6b-v3"
    parakeet_device: Literal["cuda", "cpu"] = "cuda"
    # Isolated-venv server location. When ``parakeet_use_isolated_venv``
    # is True (default), the engine spawns parakeet_server.py from
    # this venv on first use and talks to it via HTTP.
    parakeet_use_isolated_venv: bool = True
    parakeet_server_python: str = "ultronVoiceAudio/.venv-parakeet/Scripts/python.exe"
    parakeet_server_script: str = "ultronVoiceAudio/scripts/parakeet_server.py"
    parakeet_server_url: str = "http://127.0.0.1:8771"
    # How long to wait for the server's /healthz to return on startup.
    parakeet_server_startup_timeout_seconds: float = 60.0
    # Per-transcribe HTTP timeout. Parakeet is fast (~5-20 ms for
    # 5 s audio); 30 s headroom covers cold-start + worst-case.
    parakeet_request_timeout_seconds: float = 30.0
    # --- Moonshine-side config (used when engine resolves to moonshine) ---
    # 2026-05-22. Moonshine ONNX is the smallest-footprint STT option
    # in the stack -- 27 MB tiny / 58 MB base, runs on CPU via
    # onnxruntime, streaming-native. Per the openasr-leaderboard
    # average, moonshine/base WER is ~10% which is HIGHER than
    # Whisper base.en's ~5% on LibriSpeech-clean -- but moonshine's
    # numbers are averaged over harder mixed-noise / accented sets,
    # not directly comparable. The latency win is the headline:
    # ~5-15 ms on a 5 s clip vs Whisper's ~80 ms on the same hardware.
    #
    # Model names:
    #   Streaming variants (recommended for live voice; emit partial
    #   transcripts during capture):
    #     "medium-streaming-en"  -- 200M params, ~6.65% WER avg, default
    #     "small-streaming-en"   -- mid-tier
    #     "base-streaming-en"    -- smaller streaming
    #     "tiny-streaming-en"    -- 26M params, smallest streaming
    #   Non-streaming variants (lower footprint, one-shot only):
    #     "moonshine/base"  -- 58 MB, ~10.07% WER avg
    #     "moonshine/tiny"  -- 27 MB, ~12.66% WER avg
    moonshine_model: str = "medium-streaming-en"
    # ``device`` is accepted for API parity with the other engines but
    # Moonshine runs on CPU via ONNX runtime. (A future
    # ``onnxruntime-gpu`` swap could honour cuda; not wired today.)
    moonshine_device: Literal["cpu", "cuda"] = "cpu"
    # Precision is selected via the model asset bundle in moonshine-voice
    # (the English default downloads as quantized). The ``moonshine_precision``
    # field is retained for compatibility with the prior useful-moonshine-onnx
    # engine; on moonshine-voice it's accepted but doesn't change behavior.
    moonshine_precision: Literal["float", "quantized"] = "float"
    # How often the C++ core flushes a partial transcript during
    # streaming. 0.2 s is the sweet spot for a voice agent -- frequent
    # enough that the speculative LLM hand-off has fresh text,
    # infrequent enough that the encoder isn't thrashing.
    moonshine_update_interval_s: float = Field(default=0.2, ge=0.05, le=2.0)
    # When True, the orchestrator's capture loop calls
    # ``engine.feed_audio(chunk)`` on every audio block while VAD is
    # active, so partial transcripts are continuously available via
    # ``engine.get_partial_text()``. Set False to bypass streaming and
    # use the one-shot ``transcribe(buffer)`` path even on a streaming
    # model (useful for benchmarking or debugging).
    moonshine_streaming_capture: bool = True


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


class LLMHistoryCompressionConfig(_Strict):
    """SWE-Agent porting batch 2 (catalog T2 + T9) — LLM history-shape
    compression knobs applied inside :meth:`LLMEngine._build_messages`.

    Two independent processors share this config:

    * **closed_window_enabled** (T2): walk the recent-turn history
      in reverse; when the same file's view appears more than once,
      collapse the older snapshots to a one-line
      ``Outdated window with N lines omitted...`` summary. Catches
      the common redundancy when the user / model re-opens the same
      file in consecutive turns. Pure text rewrite -- zero token cost
      to construct, frees model attention budget. Default ON.
    * **last_n_enabled** (T9): elide all but the last ``last_n``
      observations to ``Old environment output: (M lines omitted)``.
      The ``last_n_polling`` parameter slows the elision-window
      update so Anthropic prompt caching stays warm for ``polling``
      turns at a time. Default OFF on the in-process voice path
      (Qwen 3.5 4B doesn't use prompt caching and the existing
      ``memory.history_turns_for_llm`` cap already serves this role)
      but the knob is in place for the future ACP / HTTP-client
      paths that go to Anthropic.

    Both processors are pure -- they don't touch the system message
    or the current user message, only the recent-history block
    between them. Compressor exceptions are caught and the raw
    history flows through unchanged (fail-open per the binding
    rule).
    """

    enabled: bool = True
    closed_window_enabled: bool = True
    last_n_enabled: bool = False
    last_n: int = Field(5, ge=1, le=100)
    last_n_polling: int = Field(1, ge=1, le=50)
    # Catalog 09 batch G wiring: pick a condenser per-intent (NoOp for
    # short conversational turns, Recent for factual, LLMSummarizing
    # for long coding contexts) before applying the closed-window /
    # last-N processors. Default ON -- the conversational + lightweight
    # quick-probe intents map to NoOp which is a zero-cost passthrough
    # so the voice-path TTFT baseline is preserved on the common path;
    # the costlier branches only fire on coding / hybrid intents
    # where the LLM call dwarfs the condense overhead anyway. Fail-open
    # at every layer (factory exception / condenser exception /
    # CondenseResult.error all leave the raw history flowing through
    # unchanged). Set to ``False`` to revert to the legacy fixed
    # pipeline.
    intent_adaptive: bool = True


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
    # n_ctx -- 2026-05-22 bumped 2048 -> 6144 after a live "what time in
    # Frankfurt" turn during gaming mode silently produced 0 LLM chars
    # because the search-augmented prompt (system 4156 chars + 3 web
    # sources ~3000 chars + recent turns) totalled 2109 tokens, just
    # past the 2048 cap. KV cache at 6144 F16 adds ~150 MB vs ~50 MB
    # at 2048; on a 12 GB card the gain (search-augmented queries
    # actually answer in gaming mode) is worth the marginal VRAM cost.
    "llama-3.2-3b-abliterated": {
        # See gemma note above on naming conventions -- main from
        # mradermacher (dot), draft from bartowski (hyphen).
        "model_path": "models/Llama-3.2-3B-Instruct-abliterated.Q4_K_M.gguf",
        "n_ctx": 6144,
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
    # Optional draft model path. When non-None, BOTH the HTTP server
    # path AND the in-process path enable prompt-lookup-decoding (PLD)
    # for speculative decoding. As of 2026-05-21 (Phase 1 of the
    # frontier-enhancement pass), the in-process ``Llama`` instance
    # constructs ``LlamaPromptLookupDecoding`` and passes it via
    # ``draft_model=`` -- closing the round-8d-surfaced gap where
    # spec decoding was HTTP-server-only. NOTE: PLD is N-gram-based
    # against the prompt; it does NOT actually load the GGUF at this
    # path. The path's presence acts as the toggle (matching the
    # server's behaviour at ``llama_cpp/server/model.py:212``). A
    # future round can swap this for a custom ``LlamaDraftModel``
    # subclass that wraps the actual draft GGUF for genuine model-
    # based drafting.
    draft_model_path: Optional[str] = None
    # 2026-05-22 -- speculative-decoding flavour selector. The old
    # ``draft_model_path is not None`` toggle conflated two very
    # different paths; this knob makes them explicit.
    #
    #   "none"  -- no in-process speculative decoding at the llama-cpp
    #              layer. Orchestrator-level speculative LLM still fires.
    #              Default until live verification proves "model" is
    #              stable on this stack.
    #   "pld"   -- LlamaPromptLookupDecoding (n-gram matching against
    #              the prompt). Cheap but limited; hit llama_decode -1
    #              bugs on 0.3.22, currently disabled by default.
    #   "model" -- Load draft_model_path as a SECOND Llama instance
    #              and use it as a real model-based draft via
    #              :mod:`ultron.llm.draft_model`. Theoretical 30-50%
    #              gen speedup when the draft agrees; same llama_decode
    #              C path that fails on PLD, so flip with verification.
    draft_kind: Literal["none", "pld", "model"] = "none"
    # 2026-05-21 -- PLD tunables. Used only when ``draft_kind == "pld"``.
    # ``num_pred_tokens`` speculates further at higher cost on mis-
    # predict; ``max_ngram_size`` makes the matcher more selective.
    speculative_max_ngram_size: int = Field(default=2, ge=1, le=8)
    speculative_num_pred_tokens: int = Field(default=10, ge=1, le=64)
    # 2026-05-22 -- real-model draft tunables. Used only when
    # ``draft_kind == "model"``. Conservative: 4 tokens per verification
    # round (typical accept-rate ~3-5 on conversational prompts;
    # emitting more wastes draft compute on tokens that will be
    # rejected).
    model_draft_num_pred_tokens: int = Field(default=4, ge=1, le=16)
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
    # 2026-05-22 -- when the voice path runs with ``enable_thinking=False``
    # (Qwen3 /no_think marker), the LLM skips its chain-of-thought block
    # entirely. That's the right call for latency (5-10 s saved on
    # factual / math turns) and was empirically MORE accurate on the
    # 7x8 case that triggered the fix. But "more accurate on one math
    # question" doesn't prove it for every harder class of question.
    # When this rate is > 0, the orchestrator dice-rolls each
    # no-think generate_stream call and -- on a hit -- emits a
    # ``thinking_drift_sample`` observation pairing the user text with
    # the resulting response. Offline review of the JSONL spots
    # regression classes before they hit a user.
    #
    # 0.02 (1 in 50) gives a few samples per session without spamming.
    # Set 0.0 to disable; 1.0 to record every turn (debug only).
    enable_thinking_drift_sample_rate: float = Field(
        default=0.02, ge=0.0, le=1.0,
    )
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
    # SWE-Agent batch 2 (T2 + T9) — history-shape compression knobs
    # for the recent-turn block inside :meth:`LLMEngine._build_messages`.
    # Defaults: closed-window ON (no-op when no file-view headers in
    # history -- ultron's voice path rarely sees them), last-N OFF
    # (the existing memory.history_turns_for_llm cap covers this for
    # the in-process voice path; the knob is in place for the future
    # ACP / HTTP-client path that goes to Anthropic with prompt
    # caching).
    history_compression: LLMHistoryCompressionConfig = Field(
        default_factory=LLMHistoryCompressionConfig,
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
    # 2026-05-21 frontier-enhancement Item 3 -- embedder upgrade
    # PATH AVAILABLE but DEFAULT STAYS bge-small.
    #
    # Live bench (2026-05-21): jina-embeddings-v3 produces 568 ms
    # per encode call on CPU vs bge-small's 3 ms -- a **183x
    # slowdown** for the +3 MTEB quality lift. On the voice memory
    # write path that hammers the embedder every turn, this is
    # a catastrophic trade. The migration script and the dim-
    # mismatch detection logic stay in place so operators who
    # specifically want jina-v3 (e.g., for offline batch processing
    # of a large memory corpus) can still flip to it -- just two
    # config edits + a migration run:
    #
    #     dense_model: "jinaai/jina-embeddings-v3"
    #     dense_dim: 1024
    #     # then: python scripts/migrate_embeddings.py
    #
    # Note: we evaluated Qwen3-Embedding-0.6B too but it's NOT in
    # FastEmbed's catalog (would require a parallel sentence-
    # transformers backend). Jina-v3 was the most attractive
    # FastEmbed-supported frontier model on paper -- the per-call
    # latency just makes it the wrong default for this workload.
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
    # 2026-05-22 supervisor stack: separate collection for project
    # digests. Kept distinct from conversations so RAG over chat
    # history never accidentally surfaces project source via the
    # cross-encoder reranker.
    projects: str = "projects"


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


class MemoryHistoryCompressionConfig(_Strict):
    """2026-05-22 catalog batch 3: tail-preserve history compression.

    When enabled, the orchestrator can call
    :meth:`ultron.memory.background_summarizer.BackgroundSummarizer.compress_history_for_llm`
    to fold the older half of an over-budget message list into a
    single summary message via an LLM call, preserving the most recent
    half verbatim. Race-protected via :class:`SnapshotGuard` so
    foreground turn appends during the LLM call invalidate the
    compression silently rather than clobbering newer state.

    Default OFF: enabling changes how history is fed to the LLM and
    needs a measure_baseline.py pass to confirm voice TTFT impact.
    The :class:`BackgroundSummarizer` method itself is always
    constructable when ``compress_summarize_fn`` is supplied; this
    flag just gates the orchestrator integration that calls it.
    """

    enabled: bool = False
    # Token budget the compressed history should fit in. Defaults to
    # match aider's 1024 sweet-spot.
    max_tokens: int = Field(default=1024, ge=128, le=32768)
    # Maximum recursion depth before falling back to summarising
    # everything in one pass. Aider uses 3.
    max_depth: int = Field(default=3, ge=1, le=10)


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
    # Deep-memory recall: when True (default), an explicit exhaustive-recall
    # voice command ("recall everything we discussed about X" / "dig deep into
    # your memory about X") runs a bounded multi-pass DeepMemoryLoop instead of
    # the single-pass RAG answer. A strict matcher gates it, so normal recall
    # questions are unaffected (they stay on the fast path). Set False to
    # disable the deep-recall short-circuit entirely.
    deep_recall_enabled: bool = True
    # Conversation-history recall: when True (default), an explicit verbatim-
    # recall question about THIS conversation ("what did I say earlier about
    # X", "what did you tell me about Y", "remind me what I asked") is answered
    # from the in-memory dual-history store -- speaking back the exact turn --
    # instead of routing to the LLM. A strict matcher gates it (normal
    # questions are unaffected); it needs no LLM/Qdrant so it works even when
    # memory is disabled. Set False to disable the short-circuit.
    history_recall_enabled: bool = True
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
    # 2026-05-22 catalog batch 3: tail-preserve history compression
    # with race protection. Default OFF.
    history_compression: MemoryHistoryCompressionConfig = Field(
        default_factory=MemoryHistoryCompressionConfig,
    )
    # 2026-05-21 frontier-enhancement Item 2 -- cross-encoder reranker.
    reranking: "MemoryRerankingConfig" = Field(
        default_factory=lambda: MemoryRerankingConfig(),
    )
    # 2026-05-21 frontier-enhancement Item 4 -- contextual retrieval.
    contextual_retrieval: "MemoryContextualRetrievalConfig" = Field(
        default_factory=lambda: MemoryContextualRetrievalConfig(),
    )


class MemoryContextualRetrievalConfig(_Strict):
    """Contextual retrieval (Anthropic technique) -- frontier item 4.

    When ``enabled`` is True, every memory turn gets a 5-15 word
    LLM-generated topic phrase prepended to its content BEFORE
    embedding. Original content is preserved unchanged in the
    payload; the phrase is also stored separately at
    ``context_summary`` for visibility.

    Why: short utterances ("yes", "OK", "later") have almost no
    embeddable signal. The context phrase restores their topical
    meaning so retrieval can find them when the conversation
    circles back. Anthropic reports up to 67% reduction in
    retrieval failures on chunk-based document corpora; for
    conversational memory the gain is concentrated on short /
    acknowledgement turns.

    Default OFF because it requires loading a second small LLM
    (typically the spec-decoding draft GGUF, ~0.6 GB on CPU) and
    adds ~50-200 ms per memory write -- background-only, so no
    voice-path impact, but real RAM cost on small machines.

    Pairs naturally with the reranker (item 2): the reranker
    benefits most when the candidate set has informative content,
    and the contextualizer makes short utterances informative.
    """

    enabled: bool = False
    # When None, falls back to ``llm.draft_model_path`` (typically
    # the spec-decoding draft GGUF -- already on disk for users on
    # the qwen3.5-4b preset).
    generator_model_path: Optional[str] = None
    # ``cpu`` is correct for the typical write rate (~5-10 turns/min).
    # Switch to ``cuda`` only if you observe write-queue backlog AND
    # have ~0.6 GB free in the voice-path VRAM budget for the
    # 0.8B Q4_K_M draft.
    generator_device: Literal["cpu", "cuda"] = "cpu"
    # Max tokens to generate per context phrase. 40 = 5-15 words
    # with plenty of headroom for the LLM to be slightly verbose
    # before our post-process trims.
    max_context_tokens: int = Field(default=40, ge=10, le=200)
    # Low temperature so context for the same turn is stable
    # across regenerations / migrations.
    generator_temperature: float = Field(default=0.2, ge=0.0, le=1.0)


class MemoryRerankingConfig(_Strict):
    """Cross-encoder reranker config (frontier item 2, 2026-05-21).

    When ``enabled`` is True, retrieval pulls a wider candidate set
    (top ``candidate_count``, typically 20) from Qdrant's hybrid
    dense+sparse fused output, then a cross-encoder model scores each
    ``(query, candidate.content)`` pair directly and re-orders the
    final top-``rag_top_k``. Industry-standard 2026 RAG pattern --
    quality lift 15-30% on RAGAS-style metrics at the cost of
    ~20-50 ms per retrieval turn on CPU.

    2026-05-21 (frontier search pass): default flipped from False to
    True now that the cross-encoder is also wired for web-search
    snippet ranking. The model loads ONCE per process (shared via
    ``_CROSS_ENCODER_CACHE`` in ``web_search.search``), so memory
    reranking pays no additional load cost. Per-turn latency adds
    ~265 ms on memory.retrieve() calls -- accepted in exchange for
    measurably better RAG context per industry benchmarks.

    Set to False to revert: cosine + RRF + recency composite only.
    Fail-open: model load or predict failures still fall back to
    the pre-rerank order. The voice path never crashes.
    """

    # 2026-05-22 -- flipped from True to False after live testing
    # measured bge-reranker-v2-m3 at 17-18 seconds per memory retrieval
    # on this user's CPU even after the 500-char content cap. The
    # latency cost overwhelms the 15-30% RAGAS quality lift the
    # literature claims. Cosine + RRF + recency composite (the
    # fallback path) is good enough; tighter ``rag_min_relevance``
    # (0.78 in config.yaml) and lower ``rag_top_k`` (3) restore
    # signal quality without the cross-encoder tax. Flip back to True
    # if you ever move to a GPU reranker or accept the latency.
    enabled: bool = False
    # Default model: ``BAAI/bge-reranker-v2-m3`` -- the 2026
    # production-standard cross-encoder reranker. 568M params,
    # ~1.1 GB on disk, multilingual, strong on conversational
    # text. Alternatives: ``BAAI/bge-reranker-base`` (~560 MB,
    # smaller, English-only), ``jinaai/jina-reranker-v1-turbo-en``
    # (~140 MB, smaller still). Set to "custom-id/repo" to swap.
    model: str = "BAAI/bge-reranker-v2-m3"
    # ``cpu`` is correct for typical candidate counts (~20). Move
    # to ``cuda`` only if measurement shows CPU is the bottleneck
    # AND voice-path VRAM headroom permits (~600 MB for v2-m3 on
    # CUDA).
    device: Literal["cpu", "cuda"] = "cpu"
    # Cross-encoder context truncation. Conversational memory chunks
    # are typically <300 tokens, so 512 is generous.
    max_length: int = Field(default=512, ge=64, le=2048)
    # How many candidates to retrieve from the hybrid layer BEFORE
    # reranking. Reranker then narrows to ``rag_top_k``. Higher =
    # better recall at higher CPU cost. 20 is the production
    # sweet spot per the 2026 RAG literature.
    candidate_count: int = Field(default=20, ge=1, le=100)


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


class TrafilaturaConfig(_Strict):
    """Local-extraction reader config (frontier 2026-05-21).

    ``trafilatura`` is a pure-Python boilerplate-stripping library --
    we use it to convert raw HTML into clean markdown locally, instead
    of round-tripping to Jina Reader. ~50-150 ms per page locally vs
    Jina's ~1-3 s round-trip. Trade-off: trafilatura sees only the
    raw HTML response, so JS-heavy SPAs return empty (the reader
    chain falls through to Jina in that case).
    """
    timeout_seconds: float = Field(default=6.0, gt=0.0)
    # 2026-05-22: tightened from 200_000 to 32_000. The LLM's 8 k context
    # can never use 200 k of source text; clipping at ~32 k chars (~8 k
    # tokens) keeps one whole source within budget while leaving room
    # for the system prompt, RAG block, and other sources. Trailing-edge
    # truncation.
    max_bytes: int = Field(default=32_000, ge=0)
    # 2026-05-22: cap raw HTML before trafilatura.extract runs. Live
    # session hit a page that produced 200 k chars in 5.75 s of CPU --
    # the parse cost scales with input size. 1 MB of HTML is enough
    # for the article body of any real news / docs / blog page; SPAs
    # exceeding this likely have no useful raw HTML anyway and the
    # reader chain falls through to Jina.
    max_html_bytes: int = Field(default=1_048_576, ge=0)


class SearxNGConfig(_Strict):
    """SearxNG self-hosted meta-search config (frontier 2026-05-21).

    SearxNG is an OSS aggregator that runs as a local service and
    relays queries to Google / Bing / DDG / Brave / Wikipedia in
    parallel. Running it locally gives unlimited queries with no API
    keys and is typically faster than any single public API.

    Setup (operator-side):
      docker run -d --name searxng -p 8888:8080 searxng/searxng
    OR:
      pip install searxng && searxng-run

    Then add ``searxng`` to ``web_search.providers`` -- it's already
    in the default list, so installing the service is the only step.
    """
    base_url: str = "http://localhost:8888"
    timeout_seconds: float = Field(default=3.0, gt=0.0)
    count: int = Field(default=5, ge=1, le=20)
    # Comma-separated category filter ("general", "news"). Empty
    # uses SearxNG's default.
    categories: str = ""
    # Comma-separated engine constraint ("google,duckduckgo,wikipedia").
    # Empty uses SearxNG's full engine set.
    engines: str = ""


class DuckDuckGoConfig(_Strict):
    """DuckDuckGo public-search fallback config (frontier 2026-05-21).

    No API key required. Uses the community ``duckduckgo-search``
    library to scrape DDG's HTML / Lite endpoints. Typical latency
    ~500-1500 ms (slower than Brave API; faster than running a full
    browser). Intended as the LAST fallback after SearxNG + Brave.
    """
    timeout_seconds: float = Field(default=5.0, gt=0.0)
    count: int = Field(default=5, ge=1, le=20)
    # DDG region code. ``"wt-wt"`` = worldwide; ``"us-en"`` = US English.
    region: str = "us-en"
    # ``"moderate"``, ``"strict"``, or ``"off"``.
    safesearch: Literal["moderate", "strict", "off"] = "moderate"


class WebSearchQueryReformulationConfig(_Strict):
    """Catalog 12 (felo-search T1): pre-search query reformulation.

    Felo returns the reformulated sub-queries it used internally
    (``query_analysis``); decomposing a complex question into several
    targeted searches improves recall. Before the provider fan-out the
    primary query is expanded into up to ``max_variants`` additional
    queries that :class:`~ultron.web_search.search.WebSearchExecutor`
    merges into its existing query list (deduped + URL-merged through
    the same provider chain + cache; a hard
    :data:`~ultron.web_search.query_rewrite.MAX_TOTAL_QUERIES` ceiling
    bounds the fan-out).

    ``use_llm=False`` (default) uses ZERO-COST structural rules ("X vs Y"
    -> two balanced queries; "how to X" -> "X tutorial"/"X guide"; "best
    X" -> "X review"/"X comparison"; leading temporal qualifier -> bare
    subject). ``use_llm=True`` adds ONE short in-process Qwen call
    (~150-250 ms, only on the SEARCH path which already pays a network
    round-trip) that decomposes the question. Both paths FAIL OPEN to the
    original query, so the search path is never broken.

    Default ON (rule-based variant is zero-cost + only affects the SEARCH
    path); the LLM variant is opt-in because of its per-search latency
    cost (this is the catalog's documented recommendation, not a
    voice-baseline-preservation default-off).
    """

    enabled: bool = True
    use_llm: bool = False
    max_variants: int = Field(default=2, ge=0, le=4)


class WebSearchConfig(_Strict):
    enabled: bool = True
    brave_api_key_env: str = "ULTRON_BRAVE_API_KEY"
    # T6 auth-profile rotation: optional ADDITIONAL Brave API-key env-var
    # names. When two or more keys resolve to non-empty values (the primary
    # ``brave_api_key_env`` plus these), the search chain rotates across them
    # via the auth-profile store -- a rate-limited (429) key is cooled down
    # and the next key is tried before the chain falls through to DuckDuckGo;
    # a key returning auth errors (401/403) is disabled for the session. With
    # a single key (the default) the legacy single-client + circuit-breaker
    # path is used unchanged. Local/no-auth providers (SearxNG, DuckDuckGo,
    # Jina reader, trafilatura, playwright) need no rotation and ignore this.
    brave_additional_api_key_envs: List[str] = Field(default_factory=list)
    brave: BraveConfig = Field(default_factory=BraveConfig)
    jina: JinaConfig = Field(default_factory=JinaConfig)
    cache: WebCacheConfig = Field(default_factory=WebCacheConfig)
    # V1-gap B3.
    citation: CitationConfig = Field(default_factory=CitationConfig)
    # 2026-05-21 frontier: multi-provider search chain with local-
    # first fallback ladder. Try SearxNG (local, unlimited) -> Brave
    # (API, 2000/mo free) -> DuckDuckGo (HTML scrape, no key, slow).
    # First non-empty result wins. SearxNG missing service silently
    # falls through to Brave; Brave rate-limit / circuit-open falls
    # through to DDG. Set to a single-element list to disable
    # fallback (e.g., ``["brave"]`` to keep legacy single-provider
    # behaviour).
    providers: List[str] = Field(
        default_factory=lambda: ["searxng", "brave", "duckduckgo"]
    )
    searxng: SearxNGConfig = Field(default_factory=SearxNGConfig)
    duckduckgo: DuckDuckGoConfig = Field(default_factory=DuckDuckGoConfig)
    # 2026-05-21 frontier: multi-reader chain for full-page extraction.
    # Try trafilatura (local, fast, ~50-150 ms) -> Jina Reader (external,
    # ~1-3 s round-trip, handles JS-heavy sites). First non-empty
    # extraction wins. Set to ``["jina"]`` to disable the local
    # reader entirely (legacy behaviour).
    readers: List[str] = Field(
        default_factory=lambda: ["trafilatura", "jina"]
    )
    trafilatura: TrafilaturaConfig = Field(default_factory=TrafilaturaConfig)
    # 2026-05-21 frontier: snippet ranking dispatch.
    # - ``"cross_encoder"`` (DEFAULT): use bge-reranker-v2-m3 cross-
    #   encoder. ~20-50 ms for 10-20 candidate snippets on CPU. Same
    #   model as ``memory.reranking`` so it loads once + caches.
    #   Specialised for query-document relevance ranking.
    # - ``"llm"`` (legacy): use the local Qwen with a JSON-emit prompt.
    #   ~500-1500 ms. The original path; kept for swap-back.
    # - ``"none"``: skip ranking; take the provider's native order +
    #   slice to top_n. ~0 ms. Reasonable when SearxNG / Brave already
    #   rank well and you want the absolute fastest path.
    ranker: Literal["cross_encoder", "llm", "none"] = "cross_encoder"
    # Catalog 12 (felo-search T1): pre-search query reformulation. Default
    # ON (rule-based, zero-cost); flip ``use_llm`` for in-process-LLM
    # decomposition on the SEARCH path.
    query_reformulation: WebSearchQueryReformulationConfig = Field(
        default_factory=WebSearchQueryReformulationConfig,
    )
    # Catalog 12 (felo-search T4): surface the search strategy (the
    # reformulated queries actually fanned out) to the user. Default ON --
    # the orchestrator appends it to the VISIBLE TRANSCRIPT only (never
    # spoken), so there is zero voice-reply impact. Mirrors felo-search's
    # "Query Analysis" disclosure. Set False to suppress the transcript line.
    expose_search_strategy: bool = True


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

    The audit signal is the primary deliverable -- if AI coding agent
    writes broken Python and reports success, the listener catches
    it. The structural metadata (functions_defined, imports, etc.)
    is captured for future use in code-context retrieval but not
    consumed yet.
    """

    enabled: bool = False
    syntax_check_on_file_change: bool = True
    attach_metadata_to_audit: bool = True


class CodingPreWriteLintConfig(_Strict):
    """2026-05-22 catalog batch 4: pre-write lint cascade.

    When enabled, the runner's FILE_CHANGE listener runs a syntax-
    check pass on every written file BEFORE narrating completion.
    For Python files: tree-sitter ERROR/missing walk + ``compile()``
    + flake8 FATAL-rule subset (E9, F821, F823, F831, F406, F407,
    F701, F702, F704, F706 — NEVER style or line-length, only
    "guaranteed breaks at runtime" rules). For other languages
    (JS, Bash, Go, Rust, etc.): tree-sitter syntax check only.

    The listener emits ``pre_write_lint_ok`` or
    ``pre_write_lint_fail`` audit rows and appends failed leaves to
    the per-task lint-failure tracker so completion narration can
    fact-check a "Done." claim. The listener never cancels the task;
    it only surfaces a signal.

    Default OFF: enabling kicks in another ~50-500 ms per file (for
    .py files; tree-sitter-only is <50 ms). Voice baseline binding
    means we ship default-OFF and the user flips on after measuring.
    """

    enabled: bool = False
    # Apply to .py / .pyi files. When False, those files still get
    # the tree-sitter base check; only the compile + flake8 layers
    # are gated.
    python_full_cascade: bool = True
    # Apply to non-Python languages (JS, Bash, Go, etc.). When False,
    # only Python files are linted.
    multi_language: bool = True
    # flake8 subprocess timeout (seconds).
    flake8_timeout_seconds: float = Field(default=5.0, ge=0.5, le=60.0)
    # Attach the rendered lint summary to the audit row.
    attach_summary_to_audit: bool = True


class CodingGoalAnchorsConfig(_Strict):
    """E2 goal-anchor planning (Phase 0+1 build, 2026-05-18+).

    When enabled, the :class:`CodingTaskRunner` decomposes incoming
    task prompts into named milestones (anchors) with per-anchor
    token budgets. As USAGE events arrive from AI coding agent, the
    runner attributes tokens to the active anchor and surfaces voice
    narration when an anchor completes, when it's near its budget,
    and when the next anchor begins.

    Resume support: when a task is paused mid-plan (budget exhausted,
    user cancelled, etc.), the runner's :meth:`send_followup` can
    prepend the next unfinished anchor's description to the
    follow-up prompt so AI coding agent resumes at the right milestone
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


class CodingAiCommentWatcherConfig(_Strict):
    """2026-05-22 catalog batch 10: in-editor ``# ai!`` watcher.

    When enabled, the orchestrator spawns an
    :class:`ultron.coding.ai_comment_watcher.AICommentWatcher` rooted
    at ``root_path`` (or :data:`settings.CODING_SANDBOX_PATH` when
    unset). The watcher scans tracked files for ``# ai!`` /
    ``# ai?`` markers and dispatches each new occurrence as a
    synthetic coding intent via the supervisor — no voice utterance
    needed.

    Default OFF because the watcher reads every file mutation in the
    project tree, which adds disk I/O on a hot loop. Operators who
    want the side-channel can flip it on once they're comfortable
    with the latency cost.
    """

    enabled: bool = False
    # Override watch root (absolute path). When empty, the
    # orchestrator falls back to settings.CODING_SANDBOX_PATH so the
    # watcher monitors all the user's sandbox projects.
    root_path: str = ""
    # Skip files larger than this (catalog default 1 MB).
    max_file_bytes: int = Field(default=1_000_000, ge=1024, le=10_000_000)
    # When True, ``# ai`` (no punctuation) also fires the callback.
    # Default False (mentions are passive).
    include_mention: bool = False
    # watchfiles polling interval.
    poll_interval_seconds: float = Field(default=0.5, ge=0.05, le=10.0)


class CodingDialogAutoHandlerConfig(_Strict):
    """2026 catalog 08 + 09 wiring: dialog auto-handler for coding tasks.

    When enabled (default ON because the safety + UX value is high),
    the :class:`CodingTaskRunner` subscribes to the bus's
    :data:`ultron.bus.events.DialogAppearedEvent` for the lifetime of
    each task. On dialog appearance during a Claude session (save-as,
    overwrite-confirm, UAC-adjacent, installer prompt), the runner
    queues a voice-friendly narration into
    ``_pending_dialog_narrations`` which the orchestrator drains and
    speaks on its next poll.

    The narration says "A '<title>' dialog appeared in <process>.
    Say yes to confirm or no to dismiss." -- the actual click /
    dismiss happens when the user answers via the
    WINDOW_CLOSE_CONFIRMATION voice intent (batch E wiring) which
    routes to the orchestrator's two-phase approval registry.

    Operators who want fully-silent automation can flip enabled=False
    -- dialogs will still be detected by the poller, but no
    narration is queued.
    """

    enabled: bool = True


class CodingPreEditSnapshotConfig(_Strict):
    """2026 catalog 08 / SWE-Agent T1 + T14 wiring: pre-edit snapshot.

    When enabled (default ON because the safety value is large vs the
    one-file-read overhead), the AI coding agent's
    :class:`DirectClaudeCodeBridge` parses each agent ``tool_use``
    event for file-write tools (``Edit`` / ``Write`` / ``MultiEdit``)
    and reads the file's current content into
    :class:`ultron.coding.file_history.FileHistory` BEFORE the CLI's
    tool executor runs the actual write.

    This unlocks:

    * **SWE-Agent T1 auto-revert** -- when the pre-write lint
      cascade (``coding.pre_write_lint``) detects a new flake8 error
      introduced by the edit, the runner can call
      :meth:`FileHistory.undo_last(path)` to roll the file back to
      the captured snapshot.
    * **SWE-Agent T14 edit_recovery** --
      :func:`ultron.coding.edit_recovery.run_edit_with_recovery` can
      use the captured ``original`` content to compare against the
      post-edit ``current`` content and decide whether a tool
      exception was spurious (i.e. the edit DID land despite the
      reported error).

    The snapshot is taken at the narrowest possible window: the
    bridge parses the assistant's ``tool_use`` event, then the CLI's
    tool executor runs the edit asynchronously. The disk file is
    almost always still in its pre-edit state when the snapshot
    fires; a worst-case race (executor faster than the bridge's
    parse) results in capturing the post-edit content, which means
    undo would be a no-op (the saved content matches current).
    Either outcome is safe.
    """

    enabled: bool = True


class CodingArchitectConfig(_Strict):
    """2026-05-22 catalog batch 6 (Phase 1): pre-dispatch architect.

    When enabled, the orchestrator constructs an
    :class:`ultron.coding.architect_supervisor.ArchitectSupervisor` and
    wires it as the supervisor's ``architect_provider``. After each
    EDIT/RESUME decision the supervisor invokes the architect to
    produce a prose plan from the user's utterance plus optional
    repo-map context. The plan is attached to the
    :class:`SupervisorDecision` as ``architect_plan_text``; downstream
    callers (``supervisor_dispatch``) can prepend it to the editor
    LLM's prompt.

    Phase 1 only: produces a plan string. Phase 2 (narrate plan via
    TTS with barge-in window) and Phase 3 (forward plan as user
    message to the editor) live in follow-up changes because they
    touch the voice hot path and need a fresh measure_baseline.py
    pass per the voice-baseline binding rule.

    Default OFF: enabling adds ~3-5 seconds of LLM call latency per
    coding dispatch on the local Qwen path. Voice-baseline binding
    means we ship default-OFF and the user flips on after measuring.
    """

    enabled: bool = False
    # Max prompt size in characters for the primary architect LLM.
    # Architectures bigger than this fall through to the cascade's
    # next entry (a future remote-LLM tier) or fail-open to no plan.
    max_prompt_chars: int = Field(default=32000, ge=1024, le=200000)
    # 2026-05-22 catalog batch 14 (T5 Phase 2): narrate the plan via
    # TTS before dispatching. Opens a barge-in window between sentences
    # so the user can interrupt with a follow-up before the editor
    # LLM kicks off. Default OFF; flip on after measure_baseline.py
    # confirms the per-turn overhead is acceptable in your environment.
    narrate_enabled: bool = False
    # Cap on the number of characters of plan text that get spoken
    # before the narrator gives up and just dispatches. Long plans are
    # noisy; the editor LLM still gets the full plan text via
    # supervisor_dispatch. Tunable per operator preference.
    narrate_max_chars: int = Field(default=400, ge=0, le=8000)
    # Grace pause in milliseconds between sentences during narration
    # so a wake-word interrupt has time to register before the next
    # sentence starts playing. Voice-baseline binding rule: keep this
    # under 250 ms so the architect cost stays bounded.
    narrate_inter_sentence_pause_ms: int = Field(default=120, ge=0, le=2000)


class CodingRepoMapConfig(_Strict):
    """2026-05-22 catalog batch 2: PageRank-weighted repo map.

    When enabled and a :class:`ProjectSupervisor` is constructed, the
    orchestrator instantiates a
    :class:`ultron.coding.repo_map.RepoMapProviderCache` and wires it
    as the supervisor's ``repo_map_provider``. After each EDIT/RESUME
    decision the supervisor attaches the rendered map to the decision
    as ``repo_map_text``; downstream callers (supervisor_dispatch)
    prepend this to the Claude prompt body so the coding agent starts
    the session with structural awareness of the project.

    The map mines the user's utterance for identifiers
    (snake_case/camelCase/etc.) and uses them to bias the PageRank
    personalization vector — so a turn like "fix the parakeet
    streaming bug" automatically surfaces ``parakeet_engine.py`` near
    the top of the map.

    Default OFF per the net-benefit feature-flag policy: enabling
    adds ~50-300 ms of pre-dispatch compute per turn (mostly tree-
    sitter parse) for projects with cached tags, more on first scan.
    """

    enabled: bool = False
    # Token budget when at least one file is in the chat set (i.e.,
    # the LLM already has visibility into some of the project). A
    # tighter budget here keeps the map from displacing the user's
    # actual question in the prompt.
    max_map_tokens: int = Field(default=1024, ge=128, le=16384)
    # Token budget when no chat files are set (the supervisor's first-
    # turn dispatch, typically). Wider so the LLM gets a broader
    # initial view of the project.
    max_map_tokens_no_chat: int = Field(default=8192, ge=128, le=32768)
    # On-disk cache directory for parsed tags. Relative paths are
    # resolved against the project root (the main worktree, not the
    # specific source project being mapped — the cache is global).
    cache_dir: str = "data/.ultron_repomap_cache"


SUPERVISOR_TIERS: dict[str, dict[str, bool]] = {
    # "off"  -- legacy path; supervisor never constructed. Equivalent
    # to enabled=False with everything else False; included for
    # explicitness in operator-facing config.
    "off": {
        "enabled": False,
        "digests_enabled": False,
        "index_enabled": False,
        "decide_enabled": False,
        "narrate_enabled": False,
        "enriched_context_enabled": False,
    },
    # "indexing_only" -- data layer only. Digests are generated at the
    # end of every Claude session and upserted to the projects
    # collection; no decision-side effect. Use this for the "let
    # projects accumulate for a week" rollout step.
    "indexing_only": {
        "enabled": True,
        "digests_enabled": True,
        "index_enabled": True,
        "decide_enabled": False,
        "narrate_enabled": False,
        "enriched_context_enabled": False,
    },
    # "deciding" -- add the decision layer. The supervisor intercepts
    # coding utterances and emits RESUME/EDIT/CLARIFY/NEW verdicts,
    # but narration + enriched context stay off so the UX is
    # unchanged. Decisions land in the JSONL audit log for review.
    "deciding": {
        "enabled": True,
        "digests_enabled": True,
        "index_enabled": True,
        "decide_enabled": True,
        "narrate_enabled": False,
        "enriched_context_enabled": False,
    },
    # "full" -- everything on. Decision layer + narration with
    # barge-in + enriched-context Claude dispatch. The final
    # rollout target.
    "full": {
        "enabled": True,
        "digests_enabled": True,
        "index_enabled": True,
        "decide_enabled": True,
        "narrate_enabled": True,
        "enriched_context_enabled": True,
    },
}


class CodingSupervisorConfig(_Strict):
    """2026-05-22 supervisor stack -- opencode-style project digest +
    semantic-resolution layer that sits between the routing classifier
    and the AI coding agent dispatch.

    Five sub-features, each independently flagged so they ship + flip
    in order (digests -> index -> decide -> narrate -> enriched).
    All default OFF until live verification per the binding
    feature-flag rollout policy.

    Default OFF = the legacy ProjectResolver + CapabilityVoiceController
    path is byte-for-byte unchanged. Flip ``enabled`` once digests are
    populated to start letting the supervisor decide; flip the per-
    phase flags individually if you want to A/B specific sub-features.

    Tier rollup
    -----------
    Setting ``tier`` to one of ``"off" | "indexing_only" | "deciding"
    | "full"`` (see :data:`SUPERVISOR_TIERS`) auto-fills the individual
    phase flags. Explicit per-flag YAML values still win over the
    tier-derived defaults -- the validator only fills fields the
    operator left unset. This keeps the per-flag knobs available as
    debug overrides while making the typical "advance one rollout
    step" change a single line.
    """

    # Tier rollup. Resolves to a pre-blessed combination of the
    # per-phase flags below; per-flag overrides in the same config
    # block win. Default ``"off"`` keeps legacy behaviour.
    tier: Literal["off", "indexing_only", "deciding", "full"] = "off"

    # Master switch. When False, every other knob is ignored and the
    # supervisor is never constructed.
    enabled: bool = False

    # Phase A: generate a digest at session end. Cheap (1 LLM call,
    # background thread); no dispatch-side effect. Safe to enable
    # before the decision layer to start building up the index.
    digests_enabled: bool = False

    # Phase B: upsert digests to Qdrant + expose semantic search.
    # No effect alone; consumed by the decide layer when on.
    index_enabled: bool = False

    # Phase C: supervisor intercepts CODE_TASK / MID_SESSION_ADJUSTMENT /
    # CLARIFICATION_RESPONSE and makes a routing decision. Requires
    # index_enabled = True to use semantic candidates; falls back to
    # registry-only candidates otherwise.
    decide_enabled: bool = False

    # Phase D: speak the supervisor's decision before dispatch with
    # a barge-in window. Requires decide_enabled = True.
    narrate_enabled: bool = False
    # Barge-in window in seconds while the decision is being spoken.
    # Within this window, a fresh wake-word fires re-classification
    # instead of proceeding with dispatch.
    narration_barge_in_window_seconds: float = Field(
        default=1.5, ge=0.0, le=10.0,
    )

    # Phase E: enrich the Claude dispatch prompt with digest + file
    # tree context so Claude doesn't rediscover state.
    enriched_context_enabled: bool = False

    # Cosine thresholds for the decision algorithm.
    #   - resolve_threshold: above this -> EDIT without clarification.
    #   - clarify_threshold: in [clarify, resolve) -> ask user to pick.
    # Below clarify_threshold -> NEW scaffold.
    resolve_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    clarify_threshold: float = Field(default=0.55, ge=0.0, le=1.0)

    # Digest generation knobs.
    digest_max_summary_chars: int = Field(default=4000, ge=200)
    digest_max_files_in_prompt: int = Field(default=40, ge=1)

    # Where the decision audit log lives. Append-only JSONL. Empty
    # disables logging.
    decisions_log_path: str = "logs/supervisor_decisions.jsonl"

    # Max candidates retained in a Decision (for CLARIFY presentation
    # + audit log).
    max_candidates_in_decision: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def _apply_tier(self) -> "CodingSupervisorConfig":
        """Fill the phase flags from ``tier`` for fields the user left unset.

        Mirrors :meth:`LLMConfig._apply_preset`: only fields absent
        from ``model_fields_set`` are touched. Explicit YAML values
        always win, so an operator can sit at tier ``"deciding"``
        while temporarily flipping ``narrate_enabled: true`` for an
        A/B without writing out the full tier preset by hand.
        """
        defaults = SUPERVISOR_TIERS.get(self.tier)
        if defaults is None:  # pragma: no cover -- Literal narrows this
            return self
        for field, value in defaults.items():
            if field not in self.model_fields_set:
                object.__setattr__(self, field, value)
        return self


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
    # 2026-05-22 catalog batch 4: pre-write lint cascade. Default OFF.
    pre_write_lint: CodingPreWriteLintConfig = Field(
        default_factory=CodingPreWriteLintConfig,
    )
    # A3 wiring -- stored-facts fast-path on clarifications.
    facts: CodingFactsConfig = Field(default_factory=CodingFactsConfig)
    # 2026-05-22 supervisor stack (opencode-inspired). Default OFF
    # across the board until digests are populated -- see
    # CodingSupervisorConfig for per-phase flags.
    supervisor: CodingSupervisorConfig = Field(
        default_factory=CodingSupervisorConfig,
    )
    # 2026-05-22 catalog batch 2: PageRank repo map for supervisor
    # dispatch. Default OFF -- adds 50-300 ms pre-dispatch latency.
    # See CodingRepoMapConfig for tuning knobs.
    repo_map: CodingRepoMapConfig = Field(
        default_factory=CodingRepoMapConfig,
    )
    # 2026-05-22 catalog batch 6 Phase 1: pre-dispatch architect.
    # Default OFF -- voice-baseline binding means we ship off and
    # let the operator measure before flipping. See
    # CodingArchitectConfig for tuning knobs.
    architect: CodingArchitectConfig = Field(
        default_factory=CodingArchitectConfig,
    )
    # 2026-05-22 catalog batch 10: in-editor # ai! marker watcher.
    # Default OFF. See CodingAiCommentWatcherConfig.
    ai_comment_watcher: CodingAiCommentWatcherConfig = Field(
        default_factory=CodingAiCommentWatcherConfig,
    )
    # 2026 catalog 08 / SWE-Agent T1 + T14 wiring: pre-edit content
    # snapshot in direct_bridge.py. When the AI coding agent's
    # tool_use event surfaces a file write, the bridge reads the
    # current file content and stashes it in FileHistory BEFORE the
    # CLI's tool executor runs. This unlocks SWE-Agent T1 auto-revert
    # (lint-revert via FileHistory.undo_last) AND T14 edit_recovery
    # (run_edit_with_recovery on SEARCH/REPLACE failures). Default
    # ON because the safety value is large and the overhead is one
    # file-read per write tool-call (~1-5 ms).
    pre_edit_snapshot: "CodingPreEditSnapshotConfig" = Field(
        default_factory=lambda: CodingPreEditSnapshotConfig(),
    )
    # 2026 catalog 08 + 09 wiring: dialog auto-handler. When enabled
    # (default ON), CodingTaskRunner subscribes to the bus's
    # DialogAppearedEvent for the lifetime of each coding task and
    # queues a voice-friendly narration on dialog appearance. The
    # user's spoken yes/no reply routes via WINDOW_CLOSE_CONFIRMATION
    # to the orchestrator's two-phase approval registry. Operators
    # who want silent automation can flip enabled=False.
    dialog_auto_handler: "CodingDialogAutoHandlerConfig" = Field(
        default_factory=lambda: CodingDialogAutoHandlerConfig(),
    )
    # 2026 catalog wiring (T1): per-task loop detection on the coding
    # event stream. When enabled (default ON), CodingTaskRunner runs the
    # 5-detector LoopDetectionManager over each task's TOOL_RESULT stream
    # and speaks a single heads-up if a hard escalation fires (the same
    # tool failing identically ~20-30 times). It LOGS + NARRATES only --
    # it never cancels the task (canceling the coding subprocess mid-flight
    # could lose work; the user can say "stop", which routes to the existing
    # cancel path). A backstop layer above the coding subprocess's + the
    # OpenClaw agents' own turn limits. Flip False for silent operation.
    loop_detection_enabled: bool = True
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
    # fixes import errors, then re-runs, etc. A direct AI coding agent
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
    # 2026-05-20 round 9: extended reference-window conditioning. XTTS
    # v2's ``get_conditioning_latents()`` has Coqui library defaults
    # of ``gpt_cond_len=6`` and ``max_ref_length=30`` -- so even though
    # we hand it a 3-minute Ultron_vocals_mono_v1.wav, only ~6 s
    # reach the GPT for prosody conditioning and ~30 s reach the
    # HiFi-GAN speaker encoder. Bumping both gives the speaker
    # embedding more prosodic variety from the same clip without
    # touching the locked v3 filter chain or requiring a fine-tune.
    # ``gpt_cond_len`` is the total seconds fed to the GPT
    # conditioning encoder; ``gpt_cond_chunk_len`` is the per-chunk
    # size (XTTS averages over N=gpt_cond_len/gpt_cond_chunk_len
    # chunks); ``max_ref_length`` is the total seconds fed to the
    # HiFi-GAN speaker encoder. Per-startup cost is ~1-2 s extra of
    # conditioning latent computation but it only happens once.
    gpt_cond_len: int = Field(default=30, ge=3, le=120)
    gpt_cond_chunk_len: int = Field(default=6, ge=3, le=30)
    max_ref_length: int = Field(default=60, ge=10, le=180)


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
    # 2026-05-22: lightweight spectral magnitude smoothing pass on
    # the synth output. Masks the pitch wobble produced by the
    # partial Ultron fine-tune (Stage 1 complete + Stage 2 epoch 0
    # only; SLM joint adversarial training at epoch 3+ never ran).
    # Cost ~10 ms / sec audio; hidden by the round-8c producer-
    # consumer pipeline on clips 2+. Pre-applied at ack-cache build
    # time so cached phrases pay zero runtime cost. Default ON
    # while the fine-tune ships partially trained; flip OFF once
    # the model is fully trained (epochs 3-9 add WavLM smoothing
    # pressure at the weight level).
    apply_spectral_smooth: bool = False
    # STFT magnitude median-filter width in frames. 5 frames at
    # hop=512, sr=24 kHz = ~107 ms window -- the post-A/B sweet spot
    # on the partial-fine-tune corpus (2026-05-22 user pick after
    # comparing windows 3/5/7/9 on the 16-sentence Ultron test set).
    # 3 frames (~64 ms) leaves audible wobble; 7+ frames (~150 ms+)
    # starts softening fricatives. Pass 1 to no-op without removing
    # the call site.
    spectral_smooth_window: int = Field(default=5, ge=1, le=15)
    # 2026-05-22 boundary artifact trimmer. The partial fine-tune
    # (Stage 1 + Stage 2 epoch 0 only; SLM joint never ran) generates
    # brief noise bursts before and after speech. This trims those
    # boundary regions via RMS energy detection and applies short
    # fade-in/fade-out to prevent abrupt clicks. Default ON for the
    # partial fine-tune; disable once the model is fully trained and
    # produces clean boundaries natively.
    apply_trim_fade: bool = True
    trim_fade_threshold_db: float = Field(default=-40.0, ge=-80.0, le=-10.0)


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
    piper_length_scale: float = Field(default=1.15, ge=0.1)
    # ``inter_sentence_pause_ms`` was removed 2026-05-20 round 8e --
    # it had no consumer; ``pause_ms`` is the actual silence written
    # between sentence clips by every TTS engine's speak_stream.
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
    # SWE-Agent T14: when a HYBRID_TASK decomposition returns malformed JSON,
    # re-query the LLM up to N times with the broken output + a remediation
    # prompt before falling back to coding-only. Off the voice hot path
    # (HYBRID_TASK decomposition only); 0 disables. Default 2.
    decomposition_requery_max_retries: int = Field(default=2, ge=0, le=5)
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
    NOT supported — the AI coding agent is the only paid service in the
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
    # 2026-05-22 -- gaming mode swaps the active LLM to a smaller
    # preset on engage to free VRAM for the game. Empty string
    # disables the swap; any other value must be a key in
    # ``LLM_PRESETS`` (validated at runtime). The matching disengage
    # restores whatever preset was active before engage.
    #
    # Default ``llama-3.2-3b-abliterated`` (~2.0 GB on GPU vs ~2.7 GB
    # for Qwen 3.5 4B + ~700 MB for its 8192-ctx KV cache) frees
    # roughly 1.5 GB of VRAM. The 2048-context KV cache also fits a
    # short gaming session conversation comfortably. Designed as the
    # gaming preset (see LLM_PRESETS comment on this preset).
    #
    # Set to ``""`` to disable the swap (keep the standby LLM during
    # gaming -- only Kokoro + VLM swaps fire).
    llm_preset: str = "llama-3.2-3b-abliterated"


class ClickPreviewConfig(_Strict):
    """2026-05-24 SWE-Agent batch 7 (T16): visual crosshair preview before clicks.

    When enabled, every desktop click runs through the
    :mod:`ultron.desktop.click_preview` gate: a red crosshair is drawn
    on a screenshot at the proposed click coordinate; the annotated
    image is shown to the VLM with the user's intent description; the
    click only fires when the VLM confirms the target. Subsequent
    clicks within ``auto_pass_radius_px`` of a recently-confirmed
    point skip the VLM round-trip (the auto-pass tier).

    Default OFF -- this adds a 1-2 s VLM round-trip per first-click
    in a region. Enable when the user wants the extra safety on
    pixel-coordinate-driven desktop automation (per the catalog's
    "auto-pass tier amortises cost" guidance).
    """

    enabled: bool = False
    auto_pass_radius_px: int = Field(default=100, ge=0, le=1000)
    crosshair_size: int = Field(default=20, ge=1, le=200)
    crosshair_thickness: int = Field(default=3, ge=1, le=50)
    require_confirmation_keyword: str = "yes"
    history_depth: int = Field(default=20, ge=1, le=500)
    # When True, a DEGRADED preview (VLM unavailable / screenshot
    # failed) BLOCKS the click. Default False -- DEGRADED treats as
    # ALLOW with an audit-log note so click flow doesn't grind to a
    # halt on a transient VLM hiccup.
    block_on_degraded: bool = False


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
    # 2026-05-22 user preference: when a launch / navigation utterance
    # gives no explicit monitor cue, place on this 1-based monitor
    # index. None falls back to the legacy "main" behaviour.
    # Set to 2 for "right monitor" on a typical dual-monitor desktop.
    default_monitor_index: Optional[int] = Field(default=2, ge=1, le=8)
    # 2026-05-24 SWE-Agent batch 7 (T16): click-preview gate.
    click_preview: ClickPreviewConfig = Field(default_factory=ClickPreviewConfig)


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


class BrowserUseConfig(_Strict):
    """Catalog 10: CDP-backed browser automation via the external
    ``browser-use`` CLI.

    Adds a second-tier browser surface on top of the existing UIA
    extraction in :func:`ultron.desktop.uia.extract_browser_content`.
    Covers everything the UIA tier cannot: CSS-selector-scoped HTML
    extraction, JavaScript evaluation (batch 3, YELLOW),
    cookie management (batch 4, YELLOW), named-session isolation
    (batch 5, YELLOW), profile-attached automation (batch 6, YELLOW),
    raw CDP passthrough (batch 7, YELLOW), and screenshot+VLM
    sequence verification via :class:`BrowserSequenceRunner` (batch 8).

    Default ON: every method fails-open when the ``browser-use``
    binary is missing, so flipping enabled does not crash the
    orchestrator on systems without the CLI installed. The
    ``binary_path`` knob accepts an explicit path; ``None`` triggers
    PATH-based discovery against ``browser-use`` / ``bu`` / ``browseruse``.

    Per the catalog's binding skip list:

    * Cloud mode (``cloud connect`` / ``cloud signup``) is OUT OF SCOPE
      under ``feedback_no_paid_apis.md`` -- the wrappers refuse to
      emit ``cloud`` subcommands at the implementation level.
    * Cloudflare tunnel (``tunnel`` subcommand) is OUT OF SCOPE --
      inbound attack surface + outbound exfiltration vector.
    * ``--cdp-url`` external URL argument is BLOCKED at the wrapper.
    * ``BROWSER_USE_SESSION`` env var is SCRUBBED on every subprocess
      call -- session selection is always explicit.

    Attributes:
        enabled: master switch. Default True; flip to False to
            short-circuit every call without invoking the binary.
        binary_path: explicit absolute path to the ``browser-use``
            executable. ``None`` triggers PATH discovery.
        default_session: bind a session name on the singleton tool
            constructed at orchestrator startup. ``None`` means "no
            session flag" (the upstream defaults to ``default``).
        default_timeout_seconds: per-call subprocess wall-clock
            timeout. The upstream documents ~50 ms warm, ~200-500 ms
            cold-start.
        default_wait_timeout_ms: default page-level timeout for
            :meth:`BrowserUseTool.wait_selector` and
            :meth:`BrowserUseTool.wait_text`.
        max_sessions: cap on simultaneous named sessions managed by
            :class:`BrowserSessionManager` (batch 5).
        headed: when True, every ``open`` call appends ``--headed``.
            Useful for debugging.
        screen_context_fallback_enabled: when True, the new tier
            slots into :func:`ultron.desktop.screen_context.build_screen_context`
            between the UIA tier and the Moondream2 VLM tier (wired
            in batch 9). When False, the new tier ships as standalone
            infrastructure callable directly via the module singleton.
    """

    enabled: bool = True
    binary_path: Optional[str] = None
    default_session: Optional[str] = None
    default_timeout_seconds: float = Field(default=30.0, gt=0.0, le=600.0)
    default_wait_timeout_ms: int = Field(default=30_000, gt=0, le=600_000)
    max_sessions: int = Field(default=3, ge=1, le=16)
    headed: bool = False
    screen_context_fallback_enabled: bool = True


class IntentConfig(_Strict):
    """Engine-agnostic intent recognizer (2026-05-22).

    Wraps ``moonshine_voice.IntentRecognizer`` to short-circuit common
    voice commands to local handlers without an LLM roundtrip. Operates
    on transcript text from ANY STT engine -- runs identically whether
    the active STT is Parakeet, Moonshine, Whisper, or typed input.

    Default OFF -- the embedding model is ~300 MB CPU RAM (q4) and the
    recognizer needs configured phrases to be useful. Operator flips
    ``enabled: true`` after deciding which commands to short-circuit.

    Performance: ~5-15 ms per ``process_utterance`` on CPU. Matches
    fire before the LLM gating path, so a matched intent saves the
    full LLM TTFT (~140-400 ms) for that turn.
    """

    enabled: bool = False
    # Currently only "embeddinggemma-300m" is supported by
    # moonshine_voice; field is here for forward-compat.
    model_name: str = "embeddinggemma-300m"
    # Quantization variant of the embedding model.
    #   "q4"    -- ~300 MB, default; smallest with no quality loss
    #   "q8"    -- ~450 MB; slightly more accurate cosine sims
    #   "fp16"  -- ~600 MB
    #   "fp32"  -- ~1.2 GB
    #   "q4f16" -- ~350 MB; mixed precision
    model_variant: Literal["q4", "q8", "fp16", "fp32", "q4f16"] = "q4"
    # Minimum cosine similarity for a match. 0.8 mirrors the
    # moonshine_voice default; raise to reduce false positives, lower
    # to catch more variations of the same intent.
    threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    # Pre-load the embedding model at orchestrator startup. False =
    # first matching call pays the ~1-3 s init cost. True trades
    # ~300 MB RAM allocation upfront for snappier first hits.
    warmup_on_init: bool = True
    # Canonical phrases registered at startup. Each becomes a
    # short-circuit path: matching utterances skip the LLM and fire
    # the registered handler (handlers are wired in the orchestrator
    # by canonical_phrase value).
    phrases: List[str] = Field(default_factory=lambda: [
        # Gaming mode shortcuts -- intent matching is faster than
        # rule classification + LLM preflight gating.
        "engage gaming mode",
        "disengage gaming mode",
        "gaming mode status",
        # Time/date local shortcuts (already handled by
        # local_clock_reply, but having them here means the recognizer
        # captures any phrasing the regex misses).
        "what time is it",
        "what is today's date",
    ])


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
    # SWE-Agent batch 4 (T11) -- Category IT (Interactive Tools)
    # blocklist. Mirrors `ToolFilterConfig` from SWE-Agent.
    interactive_tools: "InteractiveToolsBlockConfig" = Field(
        default_factory=lambda: InteractiveToolsBlockConfig(),
    )


class InteractiveToolsBlockConfig(_Strict):
    """SWE-Agent batch 4 (catalog T11) -- Category IT block configuration.

    Three independent blocklists for hang-prone interactive commands:

    * `prefix_blocklist` -- block any command starting with one of
      these (e.g. ``vim ...``, ``tail -f ...``, ``python -m venv ...``).
    * `standalone_blocklist` -- block any command that EXACTLY equals
      one of these with no arguments (catches bare ``python``,
      ``bash``, etc. that drop into a REPL).
    * `unless_regex` -- map of command name -> allow-regex; commands
      whose first token matches a key are blocked UNLESS the full
      command matches the regex (e.g. ``radare2`` allowed only with
      ``-c "..."``).

    Defaults are inherited from `category_it.DEFAULT_*` and mirror
    SWE-Agent's `ToolFilterConfig` verbatim. Empty list / dict in
    config.yaml means "use defaults"; pass an explicit list / dict
    to override entirely.
    """

    enabled: bool = True
    prefix_blocklist: list[str] = Field(default_factory=list)
    standalone_blocklist: list[str] = Field(default_factory=list)
    unless_regex: dict[str, str] = Field(default_factory=dict)
    block_message: str = (
        "Operation '{action}' is not supported by this environment."
    )


# Resolve forward ref so SafetyConfig.interactive_tools points at the
# class defined just above.
SafetyConfig.model_rebuild()


class EventsConfig(_Strict):
    """Canonical event store + bus sink (OpenHands catalog T2 + T13).

    All knobs default to a safe / no-op posture so the voice baseline
    contract holds: with ``enabled=False`` no events are persisted
    and the bus dispatch path is byte-identical.

    Attributes:
        enabled: Master switch. Default False (opt-in).
        store_backend: One of ``memory`` / ``jsonl`` / ``qdrant``.
        base_dir: Directory for the JSONL backend (relative paths are
            resolved against PROJECT_ROOT).
        qdrant_collection: Collection name when ``store_backend`` is
            ``qdrant``.
        default_session_id: Fallback identifier when a bus event
            doesn't carry an explicit ``session_id`` field.
        install_bus_sink: When True, subscribe the bus to the store so
            every bus event becomes a persisted row. Default True when
            ``enabled`` flips on; can be disabled to use the store as
            a programmatic-only surface.
    """

    enabled: bool = False
    store_backend: str = "jsonl"
    base_dir: str = "data/events"
    qdrant_collection: str = "events"
    default_session_id: str = "default"
    install_bus_sink: bool = True


class SkillsConfig(_Strict):
    """Trigger-loaded skills (OpenHands catalog T1).

    All knobs default to a no-op posture so the voice baseline contract
    holds: with ``enabled=False`` the system prompt is byte-identical
    to the pre-skills path.

    Attributes:
        enabled: Master switch. Default False (opt-in).
        always_on_only: When True, ONLY always-on skills are injected.
            Useful for debug or when keyword-trigger false-fires are
            suspected.
        disabled_skills: Names of skills to suppress even when matched.
        default_min_user_text_chars: Floor for the per-skill
            ``min_user_text_chars`` guard on keyword triggers. Prevents
            one-word utterances ("ssh") from loading stale ops skills.
            Skills can override per-file in their frontmatter.
        max_matches_per_turn: Cap on the number of triggered (non-
            always-on) skills injected per turn. Always-on skills are
            unaffected.
        max_skill_block_chars: Hard cap on the assembled skills block.
            ``0`` disables truncation.
        public_dirname: Relative directory under the project root
            containing the public skill catalog.
        user_dirname: Relative directory under the user home containing
            user-level skills.
        project_dirname: Relative directory under the active project
            containing per-project skills.
        extra_dirs: Additional absolute directories to scan with
            PROJECT precedence (e.g. a shared team skills repo).
    """

    enabled: bool = False
    # Security: scan skills from UNTRUSTED sources (USER / PROJECT / OTHER --
    # ~/.ultron/skills, a project .ultron/skills, the autonomous
    # data/evolution/skills dir) for prompt-injection / instruction-override
    # content before loading; quarantine + log on a hit. PUBLIC
    # (ultron-shipped) skills are trusted and never scanned. Fail-open.
    # Default ON.
    scan_untrusted_skills: bool = True
    always_on_only: bool = False
    disabled_skills: List[str] = Field(default_factory=list)
    default_min_user_text_chars: int = Field(default=8, ge=0, le=64)
    max_matches_per_turn: int = Field(default=6, ge=0, le=32)
    max_skill_block_chars: int = Field(default=8000, ge=0, le=64000)
    public_dirname: str = "skills"
    user_dirname: str = ".ultron/skills"
    project_dirname: str = ".ultron/skills"
    extra_dirs: List[str] = Field(default_factory=list)


class DeepResearchConfig(_Strict):
    """Catalog 12 (felo-search T3): bounded agentic deep-research loop.

    An EXPLICIT-opt-in multi-step research mode over ultron's FREE
    local-first search ladder. When the user asks to "research X in depth"
    / "do a deep dive on X" (matched by
    :func:`ultron.web_search.deep_research.match_deep_research`), the
    orchestrator runs a
    :class:`~ultron.web_search.deep_research.DeepResearchLoop` (subclass of
    the catalog-11 :class:`~ultron.agent_loop.base.AgentLoop`): decompose ->
    search each sub-question via the normal
    :class:`~ultron.web_search.search.WebSearchExecutor` -> identify gaps ->
    search again, bounded by ``max_steps`` (the load-bearing AgentLoop cap),
    then hand the accumulated sources to the existing search-augmented
    synthesis path.

    Default ON: the feature only fires on the explicit voice trigger, so a
    normal sub-second search turn is never affected. A deep-research turn
    costs ~10-18 s (several full searches), which is why it is opt-in. The
    loop fails open at every layer (LLM decomposition / gap analysis / each
    sub-query search); the per-provider rate-limit tracker + the
    ``web_results`` cache apply throughout because the same executor is used.
    """

    enabled: bool = True
    max_steps: int = Field(default=3, ge=1, le=8)
    max_sub_queries_per_step: int = Field(default=3, ge=1, le=6)
    top_n_per_query: int = Field(default=3, ge=1, le=10)
    max_accumulated_sources: int = Field(default=8, ge=1, le=30)


class EvolutionConfig(_Strict):
    """Catalog 13 (clawhub-capability-evolver clean-room): bounded autonomous
    self-improvement.

    Ultron observes its own turns, mints success/failure *capsules*, and --
    once it has seen a pattern repeat -- distills a new trigger-loaded skill
    into ``data/evolution/skills/*.md`` (a gitignored, checkpointed, live
    skills source). Every proposal runs through the bounded
    :class:`~ultron.evolution.evolution_loop.EvolutionLoop` (a subclass of the
    catalog-11 :class:`~ultron.agent_loop.base.AgentLoop`): pre-flight gate
    (fail-closed) -> autonomy tier check -> reversible checkpoint -> write ->
    blast-radius + constraint check -> regression guardrails -> keep or
    auto-revert -> hash-chained audit.

    HARD SAFETY CONTRACT (enforced in the engine, surfaced here as knobs):
    proposals are DATA ONLY (skills markdown / in-range config), NEVER
    generated code, NEVER ``src/ultron/``, NEVER a Category-K surface; the
    safety validator / audit ledger / evolution engine itself sit behind a
    Tier-3 hard wall that is never autonomously rewritten; zero network; the
    voice baseline (SOUL.md / RVC / Piper / Kokoro) is untouchable.

    Default ON: every layer is fail-open, so a construction or runtime
    failure degrades to a disabled service, never a crashed voice path. The
    per-turn hooks are microseconds; the actual cycle runs single-flight on a
    daemon thread off the hot path.
    """

    enabled: bool = True
    # The load-bearing AgentLoop step cap for a single evolution cycle.
    max_steps: int = Field(default=3, ge=1, le=8)
    # How many recorded turns must elapse before the autonomous trigger
    # considers running a background cycle. Keeps cycles rare + off the hot
    # path; a cycle still only proposes when the distiller's thresholds
    # (>=10 successes, >=7 of last 10, 24h cooldown) are met.
    cycle_check_interval_turns: int = Field(default=25, ge=1, le=10000)
    # When a surface's auto-revert rate crosses the demotion threshold,
    # whether to PAUSE that surface (require manual re-enable) vs merely
    # dropping it to propose-only. Conservative default: don't pause.
    pause_on_demote: bool = False
    # Whether the learned temperament hint (concise / detailed / warmer)
    # is prepended to the user turn before LLM generation. Tier-0 trait
    # tuning only -- never touches SOUL.md or the voicepack.
    apply_temperament: bool = True
    # ---- Catalog 14 (clawhub-self-improving-agent) qualitative capture ----
    # All default-ON + fail-open; the detectors are microsecond regex passes
    # over the turn text, so the voice baseline is unaffected.
    # Detect "the user corrected me" / "the user supplied a fact I lacked" on
    # the turn following a response, feeding them to the repair distiller.
    correction_detection_enabled: bool = True
    # Capture "I wish you could X" feature requests to a separate, never-
    # distilled backlog surfaced in the evolution digest.
    feature_request_capture_enabled: bool = True
    # Detect command / tool failures routed in from the coding task stream.
    command_failure_capture_enabled: bool = True
    # Inject a bounded "[Evolution: N pending ...]" self-evaluation nudge
    # through the SAME system-prompt seam personality uses (never the user
    # text, so the web-gate / local-clock raw-text detectors are unaffected).
    pre_turn_nudge_enabled: bool = True
    # Hard character cap on that nudge (~<=50 tokens). 0 disables the cap.
    pre_turn_nudge_max_chars: int = Field(default=240, ge=0, le=2000)
    # A pattern_key must recur at least this many times to count as
    # distill-ready -- the explicit, auditable promote threshold.
    recurrence_threshold: int = Field(default=3, ge=2, le=20)


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
    # Catalog 10 -- CDP-backed browser automation via the external
    # ``browser-use`` CLI. Second-tier alongside the UIA extraction
    # in ``ultron.desktop.uia.extract_browser_content``. Default ON
    # with fail-open contract: every method returns a structured
    # failure when the binary is missing.
    browser_use: BrowserUseConfig = Field(default_factory=BrowserUseConfig)
    # Catalog 12 (felo-search T3) -- bounded agentic deep-research loop
    # over the FREE search ladder. Explicit voice opt-in ("research X in
    # depth"); the normal sub-second search path is untouched.
    deep_research: DeepResearchConfig = Field(default_factory=DeepResearchConfig)
    # Catalog 13 (clawhub-capability-evolver clean-room) -- bounded
    # autonomous self-improvement. Observes turns, distills repeated
    # success patterns into live trigger-loaded skills under
    # ``data/evolution/skills/``, every proposal gated by a fail-closed
    # pre-flight + reversible checkpoint + regression guardrails. Data-only,
    # zero-network, Tier-3-walled. Default ON, fully fail-open.
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    # 2026-05-12 Phase 2 -- runtime tool-call validator (paired with the
    # abliterated Josiefied Qwen3-8B default LLM).
    safety: "SafetyConfig" = Field(default_factory=lambda: SafetyConfig())
    # 2026-05-22 -- engine-agnostic intent recognizer (Gemma-300M
    # embeddings via moonshine_voice). Works with any STT engine.
    intent: "IntentConfig" = Field(default_factory=lambda: IntentConfig())
    # 2026-05-23 OpenHands batch 2 (T1) -- trigger-loaded skills.
    # When enabled, walks ``skills/`` (public), ``~/.ultron/skills/`` (user),
    # ``<project>/.ultron/skills/`` (project), and injects matching skill
    # bodies into the system prompt for any keyword / slash-command that
    # matches the current user utterance. Always-on skills (no triggers)
    # fire every turn.
    skills: "SkillsConfig" = Field(default_factory=lambda: SkillsConfig())
    # 2026-05-23 OpenHands batch 3 (T2 + T13) -- canonical event store
    # with optional hash chain. When ``events.enabled`` is True, the
    # orchestrator builds the configured backend (memory / jsonl /
    # qdrant) and -- when ``install_bus_sink`` is True -- subscribes
    # the bus so every published event becomes a persisted row.
    # Default OFF so the voice baseline + bus latency are unchanged.
    events: "EventsConfig" = Field(default_factory=lambda: EventsConfig())


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


# ---------------------------------------------------------------------------
# Effective-config startup log (2026-05-22)
# ---------------------------------------------------------------------------


# Known environment overrides that affect runtime behaviour. Listed
# here so the startup log explicitly calls them out when set. Not
# exhaustive -- the log also dumps every ULTRON_* env var present, so
# new overrides don't need to be added here to be visible. But the
# names in this set get a one-line plain-English note describing what
# they override; mystery names get a generic "override active" line.
_ENV_OVERRIDE_NOTES: dict[str, str] = {
    "ULTRON_LLM_PRESET": "llm.preset",
    "ULTRON_LLM_MODEL_PATH": (
        "llm.model_path (BEWARE: silently overrides the preset's auto-fill -- "
        "this was the root cause of the 9B/4B mix-up; verify the model_path "
        "matches the intended preset)"
    ),
    "ULTRON_AUDIO_DEVICE": "audio.input_device",
    "ULTRON_AUDIO_OUTPUT_DEVICE": "audio.output_device",
    "ULTRON_LOG_LEVEL": "logging.level",
    "ULTRON_CONFIG_PATH": "config file path",
    "ULTRON_BRAVE_API_KEY": "web_search.brave API key (value not logged)",
    "ULTRON_CLAUDE_CLI": "coding.claude_cli",
    "ULTRON_WHISPER_BEAM_SIZE": "stt.beam_size",
    "ULTRON_WAKE_WORD_THRESHOLD": "wake_word.threshold",
    "ULTRON_VAD_MIN_SILENCE_MS": "vad.min_silence_duration_ms",
    "ULTRON_OPENCLAW_CLI": "openclaw.bridge.cli_path",
    "ULTRON_OPENCLAW_WORKSPACE": "openclaw.bridge.workspace_dir",
    "ULTRON_CODING_MCP_ALLOW_ANY_ROOT": (
        "coding.mcp sandbox escape (test-only; should NEVER be set in production)"
    ),
}


def log_effective_config(
    cfg: Optional["UltronConfig"] = None,
    *,
    logger: Optional[logging.Logger] = None,
    env: Optional[Mapping[str, str]] = None,
) -> None:
    """Log the effective runtime configuration at INFO level.

    Surfaces the most-frequently-confused settings:
      * Every ``ULTRON_*`` environment variable that's set (values
        elided for the Brave API key; everything else logged verbatim
        because the env var names are operator-set and not secrets in
        themselves).
      * Active LLM preset + resolved model_path + draft_model_path +
        n_ctx + runtime mode + draft_kind.
      * Active TTS engine + voice (for the kokoro / xtts_v3 engines).
      * Active STT engine + STT model (for the whisper engine).
      * Voice-baseline-adjacent toggles whose values are easy to lose
        track of: memory.reranking.enabled, memory.rag_min_relevance,
        coding.supervisor.tier, and a small set of default-OFF flags
        that change retrieval behaviour when flipped on.

    Args:
        cfg: explicit config to log. When ``None``, calls
            :func:`get_config`.
        logger: explicit logger. When ``None``, uses
            ``ultron.config.effective``.
        env: explicit env mapping (for tests). When ``None``, uses
            :data:`os.environ`.

    Fail-open: any individual field-read or log call is caught and
    the next field is logged. The function never raises.
    """
    if logger is None:
        logger = logging.getLogger("ultron.config.effective")
    if env is None:
        env = os.environ
    try:
        cfg = cfg if cfg is not None else get_config()
    except Exception as e:                                         # noqa: BLE001
        logger.warning("effective-config: get_config() failed: %s", e)
        return

    # --- ULTRON_* env vars ---
    ultron_keys = sorted(k for k in env.keys() if k.startswith("ULTRON_"))
    if ultron_keys:
        logger.info("effective-config: %d ULTRON_* env var(s) set:", len(ultron_keys))
        for key in ultron_keys:
            note = _ENV_OVERRIDE_NOTES.get(key, "override active")
            value = env.get(key, "")
            # Don't log secret values verbatim. Known-secret env vars
            # get elided; everything else (paths, preset names, log
            # levels) is operator-set and useful in the log.
            if "API_KEY" in key or "TOKEN" in key or "SECRET" in key:
                value_repr = "<set>" if value else "<empty>"
            else:
                value_repr = repr(value)
            logger.info("  %s=%s  (%s)", key, value_repr, note)
    else:
        logger.info("effective-config: no ULTRON_* env vars set")

    # --- LLM ---
    try:
        llm = cfg.llm
        logger.info(
            "effective-config: LLM preset=%r model=%r draft=%r n_ctx=%d "
            "runtime=%r draft_kind=%r",
            llm.preset,
            llm.model_path,
            getattr(llm, "draft_model_path", None),
            getattr(llm, "n_ctx", 0),
            getattr(llm, "runtime", "in_process"),
            getattr(llm, "draft_kind", "none"),
        )
    except Exception as e:                                         # noqa: BLE001
        logger.warning("effective-config: LLM section failed: %s", e)

    # --- TTS ---
    try:
        tts = cfg.tts
        engine = tts.engine
        engine_info = f"engine={engine!r}"
        if engine == "kokoro":
            kk = getattr(tts, "kokoro", None)
            if kk is not None:
                engine_info += (
                    f" voice={kk.voice!r} device={kk.device!r} speed={kk.speed}"
                )
        elif engine == "xtts_v3":
            xv = getattr(tts, "xtts_v3", None)
            if xv is not None:
                engine_info += f" filter_preset={xv.filter_preset!r} speed={xv.speed}"
        logger.info("effective-config: TTS %s", engine_info)
    except Exception as e:                                         # noqa: BLE001
        logger.warning("effective-config: TTS section failed: %s", e)

    # --- STT ---
    try:
        stt = cfg.stt
        engine = getattr(stt, "engine", "whisper")
        info = f"engine={engine!r}"
        if engine == "moonshine":
            info += f" model={stt.moonshine_model!r}"
        elif engine == "parakeet":
            info += f" model={stt.parakeet_model!r}"
        else:
            info += f" model={stt.model!r} beam={stt.beam_size}"
        gaming = getattr(stt, "gaming_engine", "")
        if gaming and gaming != engine:
            info += f" gaming_engine={gaming!r}"
        logger.info("effective-config: STT %s", info)
    except Exception as e:                                         # noqa: BLE001
        logger.warning("effective-config: STT section failed: %s", e)

    # --- Audio ---
    try:
        audio = cfg.audio
        logger.info(
            "effective-config: AUDIO input_device=%r gain_db=%s",
            audio.input_device, audio.input_gain_db,
        )
    except Exception as e:                                         # noqa: BLE001
        logger.warning("effective-config: AUDIO section failed: %s", e)

    # --- Memory ---
    try:
        mem = cfg.memory
        reranking_enabled = getattr(getattr(mem, "reranking", None), "enabled", None)
        topical = getattr(getattr(mem, "topical_chunking", None), "enabled", None)
        discourse = getattr(getattr(mem, "discourse_tagging", None), "enabled", None)
        bg_sum = getattr(getattr(mem, "background_summary", None), "enabled", None)
        contextual = getattr(
            getattr(mem, "contextual_retrieval", None), "enabled", None,
        )
        logger.info(
            "effective-config: MEMORY rag_top_k=%d rag_min_relevance=%s "
            "reranking=%s topical_chunking=%s discourse_tagging=%s "
            "background_summary=%s contextual_retrieval=%s",
            mem.rag_top_k, mem.rag_min_relevance,
            reranking_enabled, topical, discourse, bg_sum, contextual,
        )
    except Exception as e:                                         # noqa: BLE001
        logger.warning("effective-config: MEMORY section failed: %s", e)

    # --- Embedder + parallel encoding ---
    try:
        emb = cfg.embeddings
        logger.info(
            "effective-config: EMBEDDINGS dense=%r dim=%d parallel_query=%s",
            emb.dense_model, emb.dense_dim,
            getattr(emb, "parallel_query_embedding", None),
        )
    except Exception as e:                                         # noqa: BLE001
        logger.warning("effective-config: EMBEDDINGS section failed: %s", e)

    # --- Coding supervisor (tier + per-flag) ---
    try:
        sup = cfg.coding.supervisor
        logger.info(
            "effective-config: SUPERVISOR tier=%r enabled=%s digests=%s index=%s "
            "decide=%s narrate=%s enriched=%s thresholds=(%s/%s)",
            sup.tier, sup.enabled, sup.digests_enabled, sup.index_enabled,
            sup.decide_enabled, sup.narrate_enabled, sup.enriched_context_enabled,
            sup.clarify_threshold, sup.resolve_threshold,
        )
    except Exception as e:                                         # noqa: BLE001
        logger.warning("effective-config: SUPERVISOR section failed: %s", e)

    # --- Coding AST metadata + goal anchors ---
    try:
        coding = cfg.coding
        ast_enabled = getattr(getattr(coding, "ast_metadata", None), "enabled", None)
        anchors = getattr(getattr(coding, "goal_anchors", None), "enabled", None)
        logger.info(
            "effective-config: CODING ast_metadata=%s goal_anchors=%s "
            "voice_task_require_testing=%s",
            ast_enabled, anchors, coding.voice_task_require_testing,
        )
    except Exception as e:                                         # noqa: BLE001
        logger.warning("effective-config: CODING section failed: %s", e)

    # --- Routing ---
    try:
        rt = cfg.routing
        amb = getattr(rt, "ambiguity_band_clarification", None)
        amb_enabled = getattr(amb, "enabled", None) if amb is not None else None
        logger.info(
            "effective-config: ROUTING ambiguity_band_clarification=%s",
            amb_enabled,
        )
    except Exception as e:                                         # noqa: BLE001
        logger.warning("effective-config: ROUTING section failed: %s", e)

    # --- Gaming mode ---
    try:
        gm = cfg.gaming_mode
        logger.info(
            "effective-config: GAMING_MODE enabled=%s llm_preset=%r",
            gm.enabled, getattr(gm, "llm_preset", ""),
        )
    except Exception as e:                                         # noqa: BLE001
        logger.warning("effective-config: GAMING_MODE section failed: %s", e)

    # --- OpenClaw ---
    try:
        oc = cfg.openclaw
        logger.info(
            "effective-config: OPENCLAW enabled=%s gateway_url=%r",
            oc.enabled, oc.gateway_url,
        )
    except Exception as e:                                         # noqa: BLE001
        logger.warning("effective-config: OPENCLAW section failed: %s", e)


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
    "SUPERVISOR_TIERS",
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
    "log_effective_config",
]
