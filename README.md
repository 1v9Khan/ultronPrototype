# Ultron — local voice-first AI assistant prototype

A continuously-listening voice assistant that runs entirely on your hardware.
Says "ultron" → captures your request → transcribes → asks a local LLM →
speaks the response. No cloud round-trips, no telemetry.

This is a **prototype** — the foundation for adding vision, memory, tool use,
and home-control capabilities later.

---

## Hardware target

Built for and tested against:

| Component | Spec                          |
|-----------|-------------------------------|
| CPU       | AMD Ryzen 7 5800X             |
| GPU       | NVIDIA RTX 3060 Ti (8 GB)     |
| RAM       | 32 GB DDR4                    |
| OS        | Windows 11                    |

Resource budget the prototype is designed to fit inside:

- **VRAM:** ≤ 7 GB (LLM + Whisper + VAD + wake-word)
- **System RAM:** ≤ 24 GB
- **First-spoken-word latency:** < 2 s from end-of-speech to first TTS audio

If you have a smaller GPU, drop `WHISPER_MODEL` to `base.en` and pick a
smaller LLM in `config/settings.py`.

---

## Setup

### 1. Verify CUDA

```powershell
nvidia-smi
```

You want CUDA 12.x or 13.x and a working driver. If `nvidia-smi` errors,
fix that first — nothing else will work.

### 2. Clone & create a virtual environment

**Important:** the RVC stack requires **Python 3.11** (not 3.12). `infer-rvc-python`
hard-pins `faiss-cpu==1.7.3`, which has no Python 3.12 wheel.

```powershell
git clone https://github.com/1v9Khan/ultronPrototype.git ultronPrototype
cd ultronPrototype
py -3.11 -m venv .venv
.venv\Scripts\activate
# pip 24.1+ rejects fairseq's old omegaconf metadata — pin pip to 24.0:
pip install "pip<24.1" wheel
```

### 3. Install dependencies

```powershell
pip install -e .
```

`pip install -e .` pulls **CPU** wheels for both PyTorch and llama-cpp-python.
You need to force-replace both with their CUDA equivalents:

```powershell
# CUDA-enabled PyTorch (also brings cudart64_12.dll, cublas64_12.dll, cudnn)
pip install --force-reinstall --no-cache-dir `
  --index-url https://download.pytorch.org/whl/cu124 `
  torch torchaudio

# CUDA-enabled llama-cpp-python (~454 MB wheel)
pip install --force-reinstall --no-deps --no-cache-dir `
  --index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 `
  llama-cpp-python
```

Note `--index-url` (not `--extra-index-url`) — without it, pip prefers the
cached CPU wheel from PyPI.

`cu124` is the latest available wheel set as of May 2026 (cu125 has a 404,
cu126+ don't exist yet). The CUDA 12.4 builds run fine on CUDA 13.x drivers
via NVIDIA forward compatibility.

The `ultron/__init__.py` bootstrap registers `torch/lib/` on the Windows DLL
search path, so llama-cpp's `ggml-cuda.dll` finds the CUDA runtime DLLs that
ship with PyTorch — no separate CUDA Toolkit install required.

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

The default pipeline is **Piper → RVC**: Piper synthesizes neutral speech,
then an RVC v2 model re-paints it as the trained target voice (Ultron / James
Spader by default). Drop your trained `.pth` + `.index` pair into
`ultron_james_spader_mcu_6941/` (or update `RVC_MODEL_DIR` in
`config/settings.py`).

To swap the **base** Piper voice, browse
[Piper voices](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US)
and put any `.onnx` + `.onnx.json` pair under `models/piper/`. Update
`TTS_VOICE_PATH` and `TTS_VOICE_CONFIG_PATH` in `config/settings.py`. Since
RVC re-skins the timbre, the upstream voice mostly affects **pacing** —
slower base voices like `en_US-lessac-medium` give a more deliberate Ultron.

To **disable RVC** and hear plain Piper, set `RVC_ENABLED = False`.

#### Tuning the voice

Edit these in `config/settings.py` and restart — no retraining needed:

| Setting | Effect |
|---|---|
| `TTS_LENGTH_SCALE` | Piper pacing. >1.0 = slower (default 1.25). The main lever for "talks too fast." |
| `RVC_PITCH_SHIFT` | Semitones. Negative = deeper (more menacing). |
| `RVC_INDEX_RATE` | 0–1. Higher = stricter adherence to the trained voice's exact tone. |
| `RVC_PROTECT` | 0–0.5. Prevents artifacts on consonants; raise if you hear chirps. |
| `RVC_F0_METHOD` | `"rmvpe"` (default, best), `"crepe"`, `"harvest"`, `"pm"`. |

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
The default `pip install -e .` pulled the CPU wheel. You must `--force-reinstall`
with `--index-url` (not `--extra-index-url`) to make pip use the CUDA build:
```powershell
pip install --force-reinstall --no-deps --no-cache-dir `
  --index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 `
  llama-cpp-python
```
Verify the install picked the right wheel: it should be ~454 MB
(`llama_cpp_python-0.3.22-py3-none-win_amd64.whl` from the cu124 release tag).

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
