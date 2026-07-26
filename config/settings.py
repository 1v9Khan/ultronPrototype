"""Compat shim: re-export every legacy ``settings.X`` constant from config.yaml.

Bridge between the new pydantic-loaded ``kenning.config`` and the existing
35 source files that import ``from config import settings; settings.X``.
Each subsystem migrates over time to read directly from
:func:`kenning.config.get_config`; once a subsystem stops referencing
``settings.X``, those names disappear from this shim.

Env-var overrides preserved here for transitional accommodation: existing
``.env`` files that set e.g. ``KENNING_LLM_FLASH_ATTN`` continue to work,
overriding the value loaded from ``config.yaml``. After all subsystems
migrate to direct ``get_config()`` use, env-var overrides die with the
shim — secrets (Brave key, etc.) stay in env vars by design.

Source of truth: ``config.yaml`` at the project root, validated via the
schema in ``src/kenning/config.py``. Behavior of this module is now a pure
function of that file plus environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# HuggingFace cache: redirect to project-local if the existing HF_HOME points
# at an unwritable path (e.g. a stale env var pointing at a removed drive).
# Respects a working user setup; only overrides when the existing path is
# actually broken. Must run before any ``huggingface_hub`` / ``faster_whisper``
# import so those libraries pick up the override.
#
# Stays here (rather than moving into config.yaml) because it's not a
# tunable — it's a startup workaround for a specific class of env-var
# corruption. Runs at import time before any HF library can cache.
# ---------------------------------------------------------------------------


def _ensure_writable_hf_cache() -> None:
    project_cache = (
        Path(__file__).resolve().parent.parent / "models" / ".hf-cache"
    ).resolve()
    for name in (
        "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE",
        "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "XET_CACHE_DIR",
    ):
        value = os.environ.get(name)
        if not value:
            continue
        try:
            Path(value).mkdir(parents=True, exist_ok=True)
        except OSError:
            os.environ.pop(name, None)
    home = os.environ.get("HF_HOME")
    if not home:
        project_cache.mkdir(parents=True, exist_ok=True)
        (project_cache / "xet" / "logs").mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(project_cache)
        home = str(project_cache)
    home_path = Path(home)
    os.environ.setdefault("HF_HUB_CACHE", str(home_path / "hub"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(home_path / "hub"))
    os.environ.setdefault("XET_CACHE_DIR", str(home_path / "xet"))


_ensure_writable_hf_cache()


# ---------------------------------------------------------------------------
# Load canonical config + re-export legacy names
# ---------------------------------------------------------------------------

from kenning.config import (  # noqa: E402  (must follow HF cache init)
    PROJECT_ROOT, MODELS_DIR, LOGS_DIR,
    get_config, resolve_path,
)

_cfg = get_config()


# Env-var override helpers -- preserve existing override surface during
# the migration. Will be removed when subsystems migrate to direct
# get_config() reads (env-substitution at YAML level handles secrets).


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------

SAMPLE_RATE = _cfg.audio.sample_rate
CHANNELS = _cfg.audio.channels
BLOCKSIZE = _cfg.audio.blocksize
DTYPE = _cfg.audio.dtype
AUDIO_DEVICE = os.getenv("KENNING_AUDIO_DEVICE") or _cfg.audio.input_device
AUDIO_OUTPUT_DEVICE = (
    os.getenv("KENNING_AUDIO_OUTPUT_DEVICE") or _cfg.audio.output_device
)
BARGE_IN_ENABLED = _env_bool("KENNING_BARGE_IN_ENABLED", _cfg.audio.barge_in_enabled)
# Always-on "Ultron, stop": the interrupt watcher runs when EITHER barge-in OR
# this is set, so "stop" works even while BARGE_IN_ENABLED is held off.
STOP_COMMAND_ENABLED = _env_bool(
    "KENNING_STOP_COMMAND_ENABLED", _cfg.audio.stop_command_enabled,
)
BARGE_IN_GRACE_SECONDS = _env_float(
    "KENNING_BARGE_IN_GRACE_SECONDS", _cfg.audio.barge_in_grace_seconds,
)
RING_BUFFER_SECONDS = _cfg.audio.ring_buffer_seconds
# Auto push-to-talk (Valorant team chat): hold the in-game PTT key via an
# external USB-HID microcontroller while a relay line plays. Default OFF;
# host writes serial bytes only (never synthetic input). See kenning.ptt.
PUSH_TO_TALK_ENABLED = _env_bool(
    "KENNING_PTT_ENABLED", _cfg.push_to_talk.enabled,
)
PUSH_TO_TALK_SERIAL_PORT = (
    os.getenv("KENNING_PTT_SERIAL_PORT") or _cfg.push_to_talk.serial_port
)


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------

VAD_THRESHOLD = _cfg.vad.threshold
MIN_SPEECH_DURATION_MS = _cfg.vad.min_speech_duration_ms
MIN_SILENCE_DURATION_MS = _cfg.vad.min_silence_duration_ms
VAD_WINDOW_SAMPLES = _cfg.vad.window_samples


# ---------------------------------------------------------------------------
# Wake word
# ---------------------------------------------------------------------------

WAKE_WORD_NAME = _cfg.wake_word.name
WAKE_WORD_MODEL_PATH = resolve_path(_cfg.wake_word.model_path)
WAKE_WORD_FALLBACK = _cfg.wake_word.fallback_model
WAKE_WORD_THRESHOLD = _env_float("KENNING_WAKE_WORD_THRESHOLD", _cfg.wake_word.threshold)
WAKE_WORD_COOLDOWN_SECONDS = _env_float(
    "KENNING_WAKE_WORD_COOLDOWN_SECONDS", _cfg.wake_word.cooldown_seconds,
)


# ---------------------------------------------------------------------------
# Whisper STT
# ---------------------------------------------------------------------------

WHISPER_MODEL = _cfg.stt.model
WHISPER_DEVICE = _cfg.stt.device
WHISPER_COMPUTE_TYPE = _cfg.stt.compute_type
WHISPER_BEAM_SIZE = _env_int("KENNING_WHISPER_BEAM_SIZE", _cfg.stt.beam_size)
WHISPER_TEMPERATURE = _env_float("KENNING_WHISPER_TEMPERATURE", _cfg.stt.temperature)
WHISPER_CONDITION_ON_PREVIOUS_TEXT = _env_bool(
    "KENNING_WHISPER_CONDITION_ON_PREVIOUS_TEXT",
    _cfg.stt.condition_on_previous_text,
)
WHISPER_VAD_FILTER = _cfg.stt.vad_filter


# LLM block migrated to direct kenning.config use in src/kenning/llm/inference.py;
# LLM_MAX_TOKENS preserved here because coordinator.py temporarily overrides
# it via attribute write at runtime. Migrating coordinator removes this shim line.
LLM_MAX_TOKENS = _cfg.llm.default_max_tokens

# Memory + RAG block migrated to direct kenning.config use in
# src/kenning/memory/{qdrant_store,embedder}.py + src/kenning/llm/inference.py.
# Remaining shim re-exports below are for callers (coordinator, orchestrator,
# scripts, tests) that haven't been migrated yet.

MEMORY_ENABLED = _env_bool("KENNING_MEMORY_ENABLED", _cfg.memory.enabled)
MEMORY_JSONL_PATH = resolve_path(_cfg.memory.jsonl_legacy_path)
MEMORY_PATH = MEMORY_JSONL_PATH  # back-compat alias
MEMORY_QDRANT_PATH = resolve_path(_cfg.qdrant.data_dir)
MEMORY_QDRANT_CONVERSATIONS = _cfg.qdrant.collections.conversations
MEMORY_QDRANT_FACTS = _cfg.qdrant.collections.facts
MEMORY_QDRANT_WEB_RESULTS = _cfg.qdrant.collections.web_results
MEMORY_FACTS_TOP_K = _env_int("KENNING_MEMORY_FACTS_TOP_K", _cfg.memory.facts_top_k)


# Web search block migrated to direct kenning.config use in
# src/kenning/web_search/{brave,jina,search,cache}.py and the orchestrator's
# build path — no shim re-exports here.


# ---------------------------------------------------------------------------
# Coding orchestration
# ---------------------------------------------------------------------------

CODING_ENABLED = _env_bool("KENNING_CODING_ENABLED", _cfg.coding.enabled)
CODING_BRIDGE = os.getenv("KENNING_CODING_BRIDGE", _cfg.coding.bridge)
CODING_MCP_ENABLED = _env_bool("KENNING_CODING_MCP_ENABLED", _cfg.coding.mcp.enabled)
CODING_MCP_HOST = os.getenv("KENNING_CODING_MCP_HOST", _cfg.coding.mcp.host)
CODING_MCP_PORT = _env_int("KENNING_CODING_MCP_PORT", _cfg.coding.mcp.port)
CODING_MCP_SSE_PATH = _cfg.coding.mcp.sse_path
CODING_MCP_LOG_PATH = resolve_path(_cfg.coding.mcp.log_path)
CODING_MCP_SERVER_NAME = _cfg.coding.mcp.server_name
CODING_MCP_CLARIFICATION_TIMEOUT_S = _env_int(
    "KENNING_CODING_MCP_CLARIFICATION_TIMEOUT_S",
    _cfg.coding.mcp.clarification_timeout_seconds,
)

CODING_TEMPLATE_DIR = resolve_path(_cfg.coding.template_dir)
CODING_PROMPT_TOKEN_BUDGET = _env_int(
    "KENNING_CODING_PROMPT_TOKEN_BUDGET", _cfg.coding.prompt_token_budget,
)
CODING_PROMPT_CHARS_PER_TOKEN = _env_int(
    "KENNING_CODING_PROMPT_CHARS_PER_TOKEN", _cfg.coding.prompt_chars_per_token,
)

CODING_DEFAULT_MODEL = os.getenv(
    "KENNING_CODING_DEFAULT_MODEL", _cfg.coding.default_model,
)
CODING_ESCALATION_MODEL = os.getenv(
    "KENNING_CODING_ESCALATION_MODEL", _cfg.coding.escalation_model,
)
CODING_ESCALATION_THRESHOLD_DEFAULT = _env_int(
    "KENNING_CODING_ESCALATION_THRESHOLD_DEFAULT",
    _cfg.coding.escalation_threshold_default,
)
CODING_ESCALATION_THRESHOLD_ESCALATION = _env_int(
    "KENNING_CODING_ESCALATION_THRESHOLD_ESCALATION",
    _cfg.coding.escalation_threshold_escalation,
)

CODING_VERIFICATION_SMOKE_TIMEOUT_S = _env_int(
    "KENNING_CODING_VERIFICATION_SMOKE_TIMEOUT_S",
    _cfg.coding.verification.smoke_timeout_seconds,
)
CODING_VERIFICATION_TEST_TIMEOUT_S = _env_int(
    "KENNING_CODING_VERIFICATION_TEST_TIMEOUT_S",
    _cfg.coding.verification.test_timeout_seconds,
)
CODING_VERIFICATION_LINT_TIMEOUT_S = _env_int(
    "KENNING_CODING_VERIFICATION_LINT_TIMEOUT_S",
    _cfg.coding.verification.lint_timeout_seconds,
)

CODING_SESSION_AUDIT_DIR = resolve_path(_cfg.coding.session_audit_dir)
CODING_TOKEN_BUDGET_PER_SESSION = _env_int(
    "KENNING_CODING_TOKEN_BUDGET_PER_SESSION", _cfg.coding.token_budget_per_session,
)
CODING_TOKEN_WARNING_THRESHOLD = _env_float(
    "KENNING_CODING_TOKEN_WARNING_THRESHOLD", _cfg.coding.token_warning_threshold,
)
CODING_PROGRESS_TIMEOUT_S = _env_int(
    "KENNING_CODING_PROGRESS_TIMEOUT_S", _cfg.coding.progress_timeout_seconds,
)
CODING_TEST_SANDBOX_PATH = resolve_path(_cfg.coding.test_sandbox_path)

# A3 wiring -- stored-facts fast-path on clarifications. Exposed as a
# dict so the Coordinator can read all four values via a single
# ``settings.CODING_FACTS`` reference (matching the access pattern in
# coordinator.py).
CODING_FACTS = {
    "top_k": _cfg.coding.facts.top_k,
    "min_confidence": _cfg.coding.facts.min_confidence,
    "min_score": _cfg.coding.facts.min_score,
    "max_age_days": _cfg.coding.facts.max_age_days,
}

# A4 pre-task confirmation. Default OFF for safe rollout.
CODING_PRE_TASK_CONFIRMATION_ENABLED = _env_bool(
    "KENNING_CODING_PRE_TASK_CONFIRMATION_ENABLED",
    _cfg.coding.pre_task_confirmation_enabled,
)
CODING_PRE_TASK_MAX_WORDS = _env_int(
    "KENNING_CODING_PRE_TASK_MAX_WORDS",
    _cfg.coding.pre_task_confirmation_max_words,
)
CODING_PRE_TASK_BARGE_IN_WINDOW_S = _env_float(
    "KENNING_CODING_PRE_TASK_BARGE_IN_WINDOW_S",
    _cfg.coding.pre_task_barge_in_window_seconds,
)

CODING_CLAUDE_CLI = os.getenv("KENNING_CLAUDE_CLI", _cfg.coding.claude_cli)
CODING_CLAUDE_MODEL = os.getenv("KENNING_CLAUDE_MODEL", _cfg.coding.claude_model)
CODING_SANDBOX_PATH = resolve_path(_cfg.coding.sandbox_root)
CODING_PROJECT_REGISTRY_PATH = resolve_path(_cfg.coding.project_registry_path)
CODING_TASK_LOG_PATH = resolve_path(_cfg.coding.audit_log_path)
CODING_TASK_TIMEOUT_S = _env_int(
    "KENNING_CODING_TASK_TIMEOUT_S", _cfg.coding.task_timeout_seconds,
)
CODING_SKIP_PERMISSIONS = _env_bool(
    "KENNING_CODING_SKIP_PERMISSIONS", _cfg.coding.skip_permissions,
)


# Addressing + follow-up block migrated to direct kenning.config use in
# pipeline/orchestrator.py and scripts/review_addressing.py — no shim
# re-exports here.


# ---------------------------------------------------------------------------
# TTS (Piper)
# ---------------------------------------------------------------------------

TTS_VOICE_PATH = resolve_path(_cfg.tts.piper_voice_path)
TTS_VOICE_CONFIG_PATH = resolve_path(_cfg.tts.piper_voice_config_path)
TTS_OUTPUT_SAMPLE_RATE = _cfg.tts.output_sample_rate
TTS_SENTENCE_FLUSH_CHARS = _cfg.tts.sentence_flush_chars
# TTS_INTER_SENTENCE_PAUSE_MS removed 2026-05-20 round 8e -- the
# field had no consumer in src/kenning/. The actual inter-sentence
# silence is TTS_PAUSE_MS below (tts.pause_ms).
TTS_LENGTH_SCALE = _env_float(
    "KENNING_TTS_LENGTH_SCALE", _cfg.tts.piper_length_scale,
)
TTS_PAUSE_MS = _env_int("KENNING_TTS_PAUSE_MS", _cfg.tts.pause_ms)
TTS_EDGE_FADE_MS = _env_int("KENNING_TTS_EDGE_FADE_MS", _cfg.tts.edge_fade_ms)


# ---------------------------------------------------------------------------
# RVC (voice conversion)
# ---------------------------------------------------------------------------

RVC_ENABLED = _cfg.tts.rvc.enabled
RVC_MODEL_DIR = resolve_path(_cfg.tts.rvc.model_dir)
RVC_MODEL_PATH = resolve_path(_cfg.tts.rvc.model_path)
RVC_INDEX_PATH = resolve_path(_cfg.tts.rvc.index_path)
RVC_SUPPORT_DIR = resolve_path(_cfg.tts.rvc.support_dir)
RVC_HUBERT_PATH = resolve_path(_cfg.tts.rvc.hubert_path)
RVC_RMVPE_PATH = resolve_path(_cfg.tts.rvc.rmvpe_path)
RVC_DEVICE = _cfg.tts.rvc.device
RVC_PITCH_SHIFT = _env_int("KENNING_RVC_PITCH_SHIFT", _cfg.tts.rvc.pitch_shift)
RVC_INDEX_RATE = _env_float("KENNING_RVC_INDEX_RATE", _cfg.tts.rvc.index_rate)
RVC_PROTECT = _env_float("KENNING_RVC_PROTECT", _cfg.tts.rvc.protect)
RVC_F0_METHOD = _cfg.tts.rvc.f0_method
RVC_RMS_MIX_RATE = _env_float("KENNING_RVC_RMS_MIX_RATE", _cfg.tts.rvc.rms_mix_rate)
RVC_FILTER_RADIUS = _env_int("KENNING_RVC_FILTER_RADIUS", _cfg.tts.rvc.filter_radius)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

KENNING_SYSTEM_PROMPT = _cfg.llm.system_prompt


# Logging block migrated to direct kenning.config use in
# src/kenning/utils/logging.py — no shim re-exports here.
