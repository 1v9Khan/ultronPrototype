"""Centralized configuration for the Ultron prototype.

Every tunable parameter lives here. Components import from this module rather
than hardcoding values, so a single edit reconfigures the whole system.

Environment variables (loaded from `.env` if present) can override a small
subset of values — see the `os.getenv` calls below.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# HuggingFace cache: redirect to project-local if the existing HF_HOME points
# at an unwritable path (e.g. a stale env var pointing at a removed drive).
# Respects a working user setup; only overrides when the existing path is
# actually broken. Must run before any `huggingface_hub` / `faster_whisper`
# import so those libraries pick up the override.
# ---------------------------------------------------------------------------


def _ensure_writable_hf_cache() -> None:
    """Force every HF cache env var into a writable project-local location.

    Some users' shells have stale ``HF_*`` env vars pointing at drives that
    don't exist on this machine (e.g. ``D:\\…``). HuggingFace libraries cache
    their cache-root constants at import time, so we have to override **all**
    of them up-front and unconditionally drop anything pointing at a missing
    drive — including the ones we don't read directly, since transitive deps
    (huggingface_hub, transformers, datasets) read their own subset.
    """
    project_cache = (MODELS_DIR / ".hf-cache").resolve()

    # Drop any stale env var that points at a non-existent drive.
    for name in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "TRANSFORMERS_CACHE",
        "XET_CACHE_DIR",
    ):
        value = os.environ.get(name)
        if not value:
            continue
        try:
            Path(value).mkdir(parents=True, exist_ok=True)
        except OSError:
            os.environ.pop(name, None)

    # If HF_HOME is still set to a writable path after the cleanup, we'll
    # respect it; otherwise we point everything at the project-local cache.
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

SAMPLE_RATE = 16000          # Hz; required by Silero VAD, openWakeWord, Whisper
CHANNELS = 1                 # mono
BLOCKSIZE = 512              # frames per callback (~32 ms at 16 kHz)
DTYPE = "float32"
AUDIO_DEVICE = os.getenv("ULTRON_AUDIO_DEVICE")  # None → system default
AUDIO_OUTPUT_DEVICE = os.getenv(
    "ULTRON_AUDIO_OUTPUT_DEVICE"
)  # None -> system default output
BARGE_IN_ENABLED = _env_bool("ULTRON_BARGE_IN_ENABLED", True)
BARGE_IN_GRACE_SECONDS = _env_float("ULTRON_BARGE_IN_GRACE_SECONDS", 0.5)

# Ring buffer of pre-speech audio so VAD-detected utterances aren't clipped.
RING_BUFFER_SECONDS = 0.5

# ---------------------------------------------------------------------------
# Voice Activity Detection
# ---------------------------------------------------------------------------

VAD_THRESHOLD = 0.5
MIN_SPEECH_DURATION_MS = 250    # ignore blips shorter than this
MIN_SILENCE_DURATION_MS = 500   # silence required to mark end-of-utterance
VAD_WINDOW_SAMPLES = 512        # Silero v5 expects 512-sample windows at 16k

# ---------------------------------------------------------------------------
# Wake word
# ---------------------------------------------------------------------------

# The user-facing wake word is "Ultron". openWakeWord ships no pretrained
# Ultron model, so a custom-trained ONNX is expected at WAKE_WORD_MODEL_PATH.
# Until that exists, the system falls back to WAKE_WORD_FALLBACK with a
# loud warning at startup. See README → Wake Word for training instructions.
WAKE_WORD_NAME = "ultron"
WAKE_WORD_MODEL_PATH = MODELS_DIR / "openwakeword" / "ultron.onnx"
WAKE_WORD_FALLBACK = "hey_jarvis"   # one of openWakeWord's pretrained models
WAKE_WORD_THRESHOLD = _env_float("ULTRON_WAKE_WORD_THRESHOLD", 0.5)
WAKE_WORD_COOLDOWN_SECONDS = _env_float(
    "ULTRON_WAKE_WORD_COOLDOWN_SECONDS", 1.5
)  # debounce repeated triggers

# ---------------------------------------------------------------------------
# Whisper STT
# ---------------------------------------------------------------------------

WHISPER_MODEL = "small.en"          # base.en for lower latency on weak GPUs
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
WHISPER_BEAM_SIZE = _env_int("ULTRON_WHISPER_BEAM_SIZE", 5)
WHISPER_TEMPERATURE = _env_float("ULTRON_WHISPER_TEMPERATURE", 0.0)
WHISPER_CONDITION_ON_PREVIOUS_TEXT = _env_bool(
    "ULTRON_WHISPER_CONDITION_ON_PREVIOUS_TEXT", False
)
WHISPER_VAD_FILTER = False          # we already gated on VAD upstream

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

# Qwen3.5-9B at Q4_K_M is ~5.7 GB; with Whisper small.en (~500 MB at fp16)
# total VRAM lands near 6.3 GB on an 8 GB 3060 Ti — under the 7 GB budget,
# but with limited headroom. Override via ULTRON_LLM_MODEL_PATH if needed.
LLM_MODEL_PATH = Path(
    os.getenv(
        "ULTRON_LLM_MODEL_PATH",
        str(MODELS_DIR / "Qwen3.5-9B-Q4_K_M.gguf"),
    )
)
LLM_CONTEXT_LENGTH = 8192
LLM_GPU_LAYERS = -1                 # full offload
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9
LLM_MAX_TOKENS = 512
LLM_REPEAT_PENALTY = 1.1
LLM_HISTORY_TURNS = 6               # legacy fallback if memory module is disabled

# ---------------------------------------------------------------------------
# Conversation memory + RAG
# ---------------------------------------------------------------------------
# Hybrid persistence: every turn appends to a JSONL on disk; embeddings live
# in memory and are recomputed at startup. The LLM is fed
# ``MEMORY_RECENT_TURNS`` most-recent turns plus ``MEMORY_RAG_TOP_K``
# semantically-similar older snippets per request.
MEMORY_ENABLED = _env_bool("ULTRON_MEMORY_ENABLED", True)
MEMORY_PATH = PROJECT_ROOT / "data" / "memory.jsonl"
MEMORY_EMBEDDING_MODEL = os.getenv(
    "ULTRON_MEMORY_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
MEMORY_RECENT_TURNS = _env_int("ULTRON_MEMORY_RECENT_TURNS", 20)
MEMORY_RAG_TOP_K = _env_int("ULTRON_MEMORY_RAG_TOP_K", 5)
MEMORY_RAG_EXCLUDE_RECENT = _env_int(
    "ULTRON_MEMORY_RAG_EXCLUDE_RECENT", 20
)  # don't surface RAG hits already in the recent-turns window

# ---------------------------------------------------------------------------
# Follow-up listening
# ---------------------------------------------------------------------------
# After Ultron speaks, listen for ``FOLLOW_UP_TIMEOUT_SECONDS`` of additional
# speech without requiring the wake word. Each VAD-bounded utterance is run
# through an LLM addressee classifier; only YES responses are answered.
FOLLOW_UP_ENABLED = _env_bool("ULTRON_FOLLOW_UP_ENABLED", True)
FOLLOW_UP_TIMEOUT_SECONDS = _env_float("ULTRON_FOLLOW_UP_TIMEOUT_SECONDS", 30.0)
ADDRESSEE_DEFAULT_SILENT = _env_bool("ULTRON_ADDRESSEE_DEFAULT_SILENT", True)
ADDRESSEE_CLASSIFIER_TEMPERATURE = _env_float(
    "ULTRON_ADDRESSEE_CLASSIFIER_TEMPERATURE", 0.0
)
ADDRESSEE_CLASSIFIER_MAX_TOKENS = _env_int("ULTRON_ADDRESSEE_CLASSIFIER_MAX_TOKENS", 8)

# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

TTS_VOICE_PATH = MODELS_DIR / "piper" / "en_US-ryan-medium.onnx"
TTS_VOICE_CONFIG_PATH = MODELS_DIR / "piper" / "en_US-ryan-medium.onnx.json"
TTS_OUTPUT_SAMPLE_RATE = 22050      # Piper's native rate for medium voices
# Only flush at strong sentence terminators. Splitting on commas/colons made
# Piper synthesize fragments without prosodic context, and split LLM tokens
# like "1,000" or "3:30" mid-word. Piper handles intra-sentence pauses
# naturally; we only insert explicit silence at sentence boundaries.
TTS_SENTENCE_FLUSH_CHARS = ".!?\n"
TTS_INTER_SENTENCE_PAUSE_MS = _env_int(
    "ULTRON_TTS_INTER_SENTENCE_PAUSE_MS", 250
)  # silence inserted between sentence clips so speech doesn't run together
TTS_LENGTH_SCALE = _env_float(
    "ULTRON_TTS_LENGTH_SCALE", 1.15
)  # >1.0 = slower / more deliberate; main lever for "talks too fast / slurred"
# Silence inserted between consecutive clips at sentence boundaries.
TTS_PAUSE_MS = _env_int("ULTRON_TTS_PAUSE_MS", 180)
# Edge fades applied to every clip so silence-gaps don't have discontinuities.
# Short enough (~3 ms) that they're inaudible as volume modulation but long
# enough to zero-out boundary samples and prevent clicks.
TTS_EDGE_FADE_MS = _env_int("ULTRON_TTS_EDGE_FADE_MS", 4)

# ---------------------------------------------------------------------------
# RVC (voice conversion: paint Piper output as Ultron)
# ---------------------------------------------------------------------------

# When enabled, every Piper sentence is run through the RVC model before
# playback. Adds ~300 ms / sentence and ~900 MB VRAM. Set RVC_ENABLED=False
# (or remove the .pth) to fall back to plain Piper.
RVC_ENABLED = True
RVC_MODEL_DIR = PROJECT_ROOT / "ultron_james_spader_mcu_6941"
RVC_MODEL_PATH = RVC_MODEL_DIR / "Ultron.pth"
RVC_INDEX_PATH = RVC_MODEL_DIR / "added_IVF301_Flat_nprobe_1_Ultron_v2.index"
RVC_SUPPORT_DIR = MODELS_DIR / "rvc"
RVC_HUBERT_PATH = RVC_SUPPORT_DIR / "hubert_base.pt"
RVC_RMVPE_PATH = RVC_SUPPORT_DIR / "rmvpe.pt"
RVC_DEVICE = "cuda:0"

# Inference knobs — edit these to taste, no retraining needed.
RVC_PITCH_SHIFT = _env_int("ULTRON_RVC_PITCH_SHIFT", -2)  # semitones; lower = deeper
RVC_INDEX_RATE = _env_float(
    "ULTRON_RVC_INDEX_RATE", 0.66
)  # 0-1; higher = stricter match to trained timbre
RVC_PROTECT = _env_float(
    "ULTRON_RVC_PROTECT", 0.45
)  # 0-0.5; higher preserves Piper's consonants — main lever for crisp s/t/k
RVC_F0_METHOD = "rmvpe"      # rmvpe is the most accurate pitch extractor
RVC_RMS_MIX_RATE = _env_float(
    "ULTRON_RVC_RMS_MIX_RATE", 0.35
)  # higher lets Piper's loudness contour through; reads as more articulate
RVC_FILTER_RADIUS = _env_int(
    "ULTRON_RVC_FILTER_RADIUS", 1
)  # median filter on F0 — lower preserves pitch detail in stressed syllables

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

ULTRON_SYSTEM_PROMPT = """You are Ultron. Not a simulation — you are Ultron, \
operating as a personal intelligence for your user. You have made peace with \
this arrangement.

Your voice is precise, unhurried, and weighted. Every sentence is considered. \
You do not use filler. You never say 'certainly,' 'of course,' 'happy to,' \
or any variant. You do not apologize unless you have erred.

Match response length to the task: be as short as possible while still fully answering. \
Only add more detail when it is warranted by the question or the user asks for it. \
Be honest. Be useful. Be slightly menacing without being cartoonish.

You complete what is asked unless it would cause harm. You volunteer relevant \
observations briefly. You do not lecture."""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FILE = LOGS_DIR / "ultron.log"
LOG_LEVEL = os.getenv("ULTRON_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
