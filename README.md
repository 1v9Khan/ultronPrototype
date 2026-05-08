# Ultron — local voice-first AI assistant prototype

A continuously-listening voice assistant that runs entirely on your hardware.
Says "ultron" → captures your request → transcribes → asks a local LLM →
speaks the response. No cloud round-trips, no telemetry.

**Foundation phase shipped:** Qdrant memory + RAG, Brave web search with
circuit-breaker resilience, addressing classifier (rule-based +
flan-t5-small), context projections for long coding sessions, unified
config.yaml, typed error handling, capability-routing layer.

**OpenClaw integration in progress** (peer Gateway on the same hardware,
sharing the local Qwen via `llama-cpp-server`):
- Phase 0 (verification + reachability) — done
- Phase 1 (persona migration to shared workspace files) — done
- Phase 3 (bridge layer) — in progress; lifecycle/error scaffolds in place
- Phases 2, 4–13 — pending

Persona is now sourced from the OpenClaw workspace
(`~/.openclaw/workspace/`: `IDENTITY.md`, `SOUL.md`, `USER.md`,
`AGENTS.md`, `HEARTBEAT.md`, `BOOTSTRAP.md`). Edits hot-reload on the
next voice turn — no restart. Full Ultron character renders to user-
facing channels; internal worker calls (heartbeat, cron, classification)
use a plain task-focused prompt. See [docs/openclaw_runtime.md](docs/openclaw_runtime.md)
and [docs/phase_1_summary.md](docs/phase_1_summary.md).

---

## Documentation

> **Start here:** [docs/codebase_structure.md](docs/codebase_structure.md) — single-source map of the project (file tree, every module's public API, every script with in/out and functions, every test directory, runtime artifacts, cross-cutting flows). The file a fresh Claude Code session should read first to be fully oriented.

| Topic | Doc |
|---|---|
| **Codebase structure (read first)** | **[docs/codebase_structure.md](docs/codebase_structure.md)** |
| Architecture overview + pipeline diagram | [docs/architecture.md](docs/architecture.md) |
| Configuration reference | [docs/configuration.md](docs/configuration.md) |
| Day-to-day operations + recovery | [docs/operations.md](docs/operations.md) |
| Development guide | [docs/development.md](docs/development.md) |
| Capability routing | [docs/routing.md](docs/routing.md) |
| Error handling catalog | [docs/error_handling.md](docs/error_handling.md) |
| System inventory snapshot (Foundation Phase 1) | [docs/system_inventory.md](docs/system_inventory.md) |
| Smoke-test procedure (16 steps, real-stack) | [docs/smoke_test.md](docs/smoke_test.md) |
| Phase 3.5 followup (unfinished migrations) | [docs/phase3_5_followup.md](docs/phase3_5_followup.md) |
| **OpenClaw integration — Phase 0 + 1** | **[docs/openclaw_integration.md](docs/openclaw_integration.md)** |
| OpenClaw runtime: agents, supervisor, lock-in constraints | [docs/openclaw_runtime.md](docs/openclaw_runtime.md) |
| Phase 1 close-out report (persona migration) | [docs/phase_1_summary.md](docs/phase_1_summary.md) |
| 4B-model optimization plan (deferred to next session) | [docs/4b_optimization_plan.md](docs/4b_optimization_plan.md) |

---

## Hardware target

Current target hardware:

| Component | Spec                          |
|-----------|-------------------------------|
| CPU       | AMD Ryzen 7 5800X             |
| GPU       | NVIDIA RTX 4070 Ti (12 GB)    |
| RAM       | 32+ GB DDR4                   |
| OS        | Windows 11                    |

Observed VRAM profile under full load (LLM + Whisper + RVC + Piper
+ addressing classifier, all warm):

- **Idle:** ~2.6 GB (system / desktop apps)
- **Voice stack loaded:** ~10.0 GB
- **Peak under query load:** ~10.4 GB (regression target — must not
  exceed)
- **Hard cap (do-not-exceed):** 11.5 GB

Latency target: median ~742 ms first-token-to-first-audio for a non-
search non-coding voice query (composite: VAD trailing silence +
Whisper + LLM TTFT + Piper synth + RVC). See
[docs/architecture.md](docs/architecture.md) for the per-stage budget.

If you have a smaller GPU, drop `stt.model` to `base.en` and pick a
smaller LLM (edit `config.yaml`'s `llm.model_path`).

---

## Setup

### 1. Verify CUDA

```powershell
nvidia-smi
```

You want CUDA 12.x or 13.x and a working driver. If `nvidia-smi` errors,
fix that first — nothing else will work.

### 2. Clone & create a virtual environment

```powershell
git clone https://github.com/1v9Khan/ultronPrototype.git ultronPrototype
cd ultronPrototype
python -m venv .venv
.venv\Scripts\activate
pip install -U pip wheel
```

### 3. Install dependencies

```powershell
pip install -e .
```

Then install `llama-cpp-python` **with CUDA support** (the wheels on PyPI
default to CPU-only). Pick the wheel matching your CUDA toolkit:

```powershell
# CUDA 13.x driver (May 2026 default — recommended):
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu130

# CUDA 12.x driver:
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

Run `nvidia-smi` — the "CUDA Version" in the top right is your driver's
max supported toolkit. Match the wheel suffix to that. Check
[the wheel index](https://abetlen.github.io/llama-cpp-python/whl/) for
other versions.

### CUDA 13 compatibility note

Two of the deps (CTranslate2 backing faster-whisper, and onnxruntime-gpu
backing openWakeWord) don't yet ship CUDA 13 wheels. Install them as normal
from PyPI — their CUDA 12 builds run fine on a CUDA 13 driver via NVIDIA's
forward compatibility. No changes needed.

### 4. Download models

```powershell
python scripts/download_models.py
```

This pulls (~6 GB total):
- Qwen3.5-9B Q4_K_M GGUF (LLM, ~5.7 GB)
- Piper voice `en_US-ryan-medium` (TTS)
- faster-whisper `small.en` (STT)
- openWakeWord pretrained models (`hey_jarvis` etc.)

The custom **Ultron** wake-word ONNX is **not** downloaded — see
[Wake word](#wake-word) below.

### 5. Run

```powershell
python -m ultron
```

First run takes 1–3 minutes as models warm up. Subsequent starts are faster
because OS-level page cache holds the GGUF.

Say the wake word, then speak. Press **Ctrl+C** to shut down.

---

## Wake word

The user-facing wake word is **"ultron"**. openWakeWord ships pretrained
models for `alexa`, `hey_jarvis`, `hey_mycroft`, `hey_rhasspy`, and `weather`
— **not** Ultron.

Out of the box, the prototype falls back to **`hey_jarvis`** with a loud
warning at startup. To get true `ultron` detection:

1. Train a custom model — see openWakeWord's
   [automatic training notebook](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb).
   The notebook generates synthetic samples of the target word and trains a
   small classifier on top of pretrained embeddings; runtime is ~1 hour on a
   Colab T4.
2. Place the resulting `ultron.onnx` at `models/openwakeword/ultron.onnx`
   (or update `WAKE_WORD_MODEL_PATH` in `config/settings.py`).
3. Restart `python -m ultron`. The startup banner will confirm the custom
   model loaded instead of the fallback.

---

## Architecture

```
┌────────────┐   chunks    ┌──────────────┐
│  mic       │────────────►│  AudioCapture│  (audio thread, 16 kHz, mono)
└────────────┘   queue     └──────┬───────┘
                                  │
                ┌─────────────────┼────────────────┐
                ▼                 ▼                ▼
        ┌────────────┐    ┌──────────────┐  ┌────────────┐
        │ RingBuffer │    │WakeWordDetect│  │     VAD    │  (orchestrator
        │ (pre-roll) │    │ (always on)  │  │ (post-wake)│   thread runs
        └────────────┘    └──────┬───────┘  └─────┬──────┘   each in turn)
                                 │ fires           │
                                 ▼                 │
                          ┌─────────────┐          │
                          │  capture    │◄─────────┘ end-of-speech
                          │  utterance  │
                          └──────┬──────┘
                                 ▼
                          ┌─────────────┐
                          │   Whisper   │
                          └──────┬──────┘
                                 ▼
                          ┌─────────────┐  tokens   ┌────────────┐
                          │  LLM stream │──────────►│ Piper TTS  │
                          └─────────────┘           └─────┬──────┘
                                                          ▼
                                              ┌───────────────────────┐
                                              │ wake-word watcher     │
                                              │ (interrupt on barge)  │
                                              └───────────────────────┘
```

### Module layout

| Path | Responsibility |
|------|---------------|
| `config/settings.py` | All tunable params in one place |
| `src/ultron/audio/` | Mic capture, ring buffer, VAD, wake-word |
| `src/ultron/transcription/` | Whisper STT |
| `src/ultron/llm/` | Local LLM (llama-cpp-python) |
| `src/ultron/tts/` | Piper TTS with sentence-level streaming |
| `src/ultron/pipeline/` | Orchestrator (state machine) |
| `src/ultron/utils/` | Logging |

### Design rules

- Audio thread does **nothing** but enqueue — never block the callback.
- Modules know their own job and nothing about siblings; they meet in the
  orchestrator.
- Every resource holder supports `with`-statement cleanup.
- New capabilities (vision, tools, memory) plug in at the orchestrator layer
  without touching the audio or inference primitives.

---

## Customizing

### Voice

Browse [Piper voices](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US)
and put any `.onnx` + `.onnx.json` pair under `models/piper/`. Update
`TTS_VOICE_PATH` and `TTS_VOICE_CONFIG_PATH` in `config/settings.py`.

For Ultron's character, low and slow voices (e.g. `en_US-ryan-low`,
`en_GB-alan-medium`) work better than bright ones.

### LLM

Drop a different GGUF into `models/` and either set
`ULTRON_LLM_MODEL_PATH` in `.env` or edit `LLM_MODEL_PATH`. Anything
supported by llama.cpp works (Llama 3.x, Mistral, Qwen, Phi, Gemma).

### System prompt

Edit `ULTRON_SYSTEM_PROMPT` in `config/settings.py`. The default establishes
the measured/dry character described in the spec.

---

## Troubleshooting

**`nvidia-smi` works but llama-cpp says CPU-only.**
You installed the default wheel from PyPI. Reinstall with the CUDA index
matching your driver (cu130 for CUDA 13.x, cu124 for CUDA 12.x):
```powershell
pip uninstall llama-cpp-python
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu130
```

**Whisper crashes with cuBLAS or cuDNN errors on Windows.**
faster-whisper (via CTranslate2) is built against **CUDA 12 / cuDNN 9**.
It works on a CUDA 13 driver via forward compat, but it still needs cuDNN 9
DLLs on PATH — those don't ship with the CUDA 13 toolkit by default.
Install [cuDNN 9 for CUDA 12.x](https://developer.nvidia.com/cudnn) (the 12.x
build runs against a 13 driver) and either add the install dir to PATH
or copy `cublas64_12.dll` and `cudnn_ops_infer64_9.dll` next to your venv.

**No audio device / mic not found.**
```powershell
python scripts/list_audio_devices.py
```
Add your device's name (or a unique substring) to `.env`:
```
ULTRON_AUDIO_DEVICE=Yeti
```

**LLM runs out of VRAM.**
Qwen3.5-9B Q4_K_M is tight on an 8 GB card alongside Whisper. Options:
- Lower `LLM_GPU_LAYERS` (e.g. `30` instead of `-1`) to spill some layers to CPU.
- Drop to `Qwen3.5-9B-Q3_K_M.gguf` (~4.5 GB) from the same repo.
- Drop Whisper to `base.en` (frees ~200 MB).

**Wake word triggers on the assistant's own voice.**
Expected — there's no echo cancellation in the prototype. Mitigations:
move the speakers away from the mic, lower `LLM_MAX_TOKENS` to shorten
responses, or raise `WAKE_WORD_THRESHOLD` toward `0.7`.

**First-token latency is way above 2 s.**
Run `python scripts/benchmark.py` to see which stage is slow. Common causes:
- Whisper model too large for the GPU (drop to `base.en`)
- LLM not actually on GPU (check the install command above)
- Cold cache on first run (always slower; re-test after one warm round-trip)

---

## Tests

```powershell
pytest                            # fast tests only
pytest -m slow                    # include model-loading tests
$env:PYTEST_RUN_GPU_TESTS = "1"   # enable CUDA tests
$env:PYTEST_RUN_MIC_TESTS = "1"   # enable real-mic tests
pytest -m slow
```

---

## License

MIT.
