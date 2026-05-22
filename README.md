# Ultron

**A local, voice-first AI assistant.** No cloud round-trips. No telemetry. Sub-second response latency on a single consumer GPU.

> Say "ultron" → the assistant captures your request, transcribes it locally, routes it through a tiered intent classifier, fetches web context when needed, generates a reply with a local LLM, and speaks it back in a custom voice. The whole loop runs in-process on your machine.

---

## At a glance

|  |  |
|---|---|
| **Tests** | 4104 passing / 16 skipped / 0 failed (~85 s sweep) |
| **TTFT** | ~210–300 ms cache-hit conversational turn (mic-stop → first audible token) |
| **VRAM** | ~4.4 GB standby on RTX 4070 Ti; gaming-mode reclaim drops to ~2.1 GB |
| **Active stack** | Moonshine STT (CPU) · Qwen 3.5 4B Q4_K_M (CUDA) · Kokoro StyleTTS2 (CUDA, fine-tuned voice) |
| **License** | MIT |

---

## What it does

- **Always-listening wake-word capture** with a custom-trained `ultron` OpenWakeWord model + Silero VAD + Smart Turn V3 (an 8 MB CPU ONNX that confirms end-of-turn in ~12 ms).
- **Hot-swappable STT** via a dual-engine registry — Moonshine for streaming (CPU), Parakeet TDT for accuracy (CUDA via isolated HTTP server), Whisper for fallback. Swap by flipping one config field.
- **Local LLM in-process** through `llama-cpp-python` (Qwen 3.5 4B Q4_K_M default, n_ctx=8192, speculative decoding wired). Per-utterance preset hot-swap via voice: "switch to the 8B".
- **Custom TTS voice** — Kokoro StyleTTS2 + ISTFTNet on CUDA with a fine-tuned voicepack. Producer-consumer pipeline so synth of clip N+1 overlaps playback of clip N. Boundary-artifact mute via cosine fades + tail aggressive zero.
- **23-kind intent router** — coding tasks (delegated to Claude Code subprocess), app launching, browser navigation, gaming-mode VRAM reclaim, model switching, "show me that article" / "take me to HBO Max" / "what's the latest news?" — each a typed routing intent with a dedicated handler.
- **Web search with intelligent ranking** — local SearxNG (Docker, news-category aware) → Brave API → DuckDuckGo cascade; Trafilatura → Jina reader cascade. Cross-encoder reranking optional.
- **Three-layer memory** — recent conversation cache, Qdrant-backed semantic RAG (bge-small dense + BM25 sparse hybrid RRF), separate project digest collection (opencode-inspired session-end summaries).
- **141-rule safety validator** with tamper-evident SHA-256 hash-chain audit log. Gates every desktop / file / shell tool call before execution.
- **Gaming mode** — voice-triggered VRAM reclaim chain (LLM hot-swap to 3B, STT → Moonshine CPU, Kokoro CUDA → CPU, VLM unload, Parakeet server stop). Frees ~2.3 GB on demand.
- **Typed event bus** — opencode-inspired pub/sub backbone (`turn.started` / `gate.verdict` / `supervisor.decided` / 14 more) so future tracing, observability, and analytics hooks are declarative subscriptions instead of scattered callbacks.

---

## Pipeline at a glance

```
mic → wake "ultron" OR addressing classifier (WARM)
  → Silero VAD + Smart Turn V3 (CPU, ~12 ms)
  → STT: DualSTTRegistry (moonshine | parakeet | whisper)
  → Intent recognizer (Gemma-300M CPU): short-circuits gaming / fresh-data intents
  → Local clock reply for bare time/date queries (~5 ms, no LLM)
  → classify_routing() → 23 RoutingIntentKind dispatches
      ├─ coding kinds → Claude Code subprocess (optional supervisor stack)
      ├─ OPEN_LAST_SOURCE → opens cited URL from prior search
      ├─ NAVIGATE_TO_SITE → SearxNG top-10 → domain-score → opens best
      ├─ APP_LAUNCH        → native Chrome/Cursor/Discord launcher
      ├─ GAMING_MODE       → VRAM reclaim chain (~2.3 GB freed)
      ├─ conversational    → LLM (Qwen 3.5 4B) with optional:
      │                       · web-search gate (rules + preflight LLM)
      │                       · multi-pass RAG retrieval
      │                       · news-category SearxNG routing
      └─ stream tokens → Kokoro TTS (CUDA, fine-tuned voice)
  → typed-bus events publish at every stage
  → async-write conversation turn to Qdrant
  → enter WARM follow-up window (30 s)
```

For the full per-module breakdown, see [`docs/codebase_structure.md`](docs/codebase_structure.md) (the binding single-source reference).

---

## Quick start

This is a personal-research prototype tuned to one operator's hardware + use case. Treat the setup as a recipe, not a turn-key install. Roughly:

1. **Hardware.** RTX 4070 Ti or comparable (12 GB VRAM). A USB mic + decent speakers / headphones. Tested on Windows 11; Linux/macOS untested.
2. **Python 3.11**, then `pip install -e .` (creates a `.venv/`, install `~7 GB` of deps including PyTorch CUDA, llama-cpp-python with CUDA flash-attn, faster-whisper, FastEmbed, Qdrant client, Kokoro, etc.).
3. **Models** — run `python scripts/download_models.py` to fetch the OpenWakeWord, Smart Turn V3, Moonshine, Kokoro, and Qwen GGUFs (~5 GB total).
4. **Config** — copy `.env.example` to `.env` and set any API keys (Brave search is optional; the SearxNG path doesn't need any). Tune `config.yaml` for your mic / monitors / preferences.
5. **Launch** — `python -m ultron` and say `ultron`.

> ⚠️ **This is not a packaged product.** The repo represents one developer's voice-assistant prototype that's been iterated on intensively over many sessions. Some integrations (OpenClaw Gateway, Telegram channel, ComfyUI media generation, mobile node) require additional credential-dependent setup — see the per-component docs below.

---

## System requirements

| | Recommended | Minimum |
|---|---|---|
| GPU | RTX 4070 Ti (12 GB) | RTX 3060 (12 GB) — untested, expect higher latency |
| CPU | AMD Ryzen 7 5800X+ (8c/16t) | 4 cores / 8 threads |
| RAM | 32 GB | 16 GB (constrained) |
| Disk | 30 GB free | 20 GB free |
| OS | Windows 11 | Windows 10 / Linux (untested) |
| Python | 3.11 | 3.10+ |
| CUDA | 12.4+ | 11.8 |

---

## Configuration

Everything tunable lives in `config.yaml` at the project root — schema-validated by pydantic in `src/ultron/config.py`. Key sections: `audio` (mic + I/O), `vad`, `stt` (engine selector), `llm` (preset + n_ctx), `tts` (engine selector), `memory` (Qdrant + RAG knobs), `web_search` (provider chain + readers), `safety` (rule toggles), `coding.supervisor` (opencode-inspired project digest stack, default OFF), `gaming_mode`, plus more.

Override via environment variables prefixed `ULTRON_*` (see `.env.example`). Restart `python -m ultron` after any config change.

---

## Documentation

> **Start here:** [`docs/codebase_structure.md`](docs/codebase_structure.md) — the binding single-source map of every module, script, test, runtime artifact, and cross-cutting flow. Kept current via a maintenance contract enforced on every commit.

<details>
<summary>Architecture + operations references (foundation-era snapshots)</summary>

| Doc | Topic |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Pipeline overview + hardware target |
| [`docs/configuration.md`](docs/configuration.md) | Per-key config reference |
| [`docs/operations.md`](docs/operations.md) | Day-to-day running + recovery |
| [`docs/development.md`](docs/development.md) | Test layout + debugging recipes |
| [`docs/routing.md`](docs/routing.md) | Capability routing |
| [`docs/error_handling.md`](docs/error_handling.md) | Phase 4 error catalog |
| [`docs/4b_optimization_plan.md`](docs/4b_optimization_plan.md) | 4B LLM migration (complete) |

Foundation snapshots are kept for historical reference; `codebase_structure.md` is the live ground truth.

</details>

<details>
<summary>OpenClaw integration (peer Gateway for proactive comms + tools)</summary>

| Doc | Topic |
|---|---|
| [`docs/openclaw_integration_final_summary.md`](docs/openclaw_integration_final_summary.md) | Cross-phase summary + setup-readiness checklist |
| [`docs/openclaw_telegram_setup.md`](docs/openclaw_telegram_setup.md) | Telegram channel (bot token) |
| [`docs/openclaw_heartbeat_setup.md`](docs/openclaw_heartbeat_setup.md) | Heartbeat agents block |
| [`docs/openclaw_browser_setup.md`](docs/openclaw_browser_setup.md) | Browser tool (Playwright + Chromium) |
| [`docs/openclaw_cron_setup.md`](docs/openclaw_cron_setup.md) | Cron jobs (Task Scheduler fallback) |
| [`docs/openclaw_hooks_setup.md`](docs/openclaw_hooks_setup.md) | Bundled hooks |
| [`docs/openclaw_memory_wiki_setup.md`](docs/openclaw_memory_wiki_setup.md) | Memory Wiki plugin |
| [`docs/openclaw_media_generation_setup.md`](docs/openclaw_media_generation_setup.md) | Local ComfyUI media generation |
| [`docs/mobile_node_setup.md`](docs/mobile_node_setup.md) | iOS / Android pairing |

</details>

<details>
<summary>Test pass reports</summary>

| Doc | Topic |
|---|---|
| [`docs/comprehensive_test_plan.md`](docs/comprehensive_test_plan.md) / [`comprehensive_test_report.md`](docs/comprehensive_test_report.md) | Functional / correctness pass (16 phases, 38 dimensions) |
| [`docs/comprehensive_quality_plan.md`](docs/comprehensive_quality_plan.md) / [`comprehensive_quality_report.md`](docs/comprehensive_quality_report.md) | Quality pass (Q0–Q13, 38 dimensions) — includes prompt-injection defense audit |
| [`docs/smoke_test.md`](docs/smoke_test.md) | 16-step interactive smoke procedure |

</details>

---

## Project status

This is a **research prototype**, not a production product. It evolves intensively across many developer-AI pair sessions. Behavior-changing features land behind feature flags (default OFF) until live-validated. The voice-quality baseline is treated as a strict latency / VRAM contract — any hot-path change re-runs `scripts/measure_baseline.py` and documents the delta. See the project-root standards doc (project root) for the binding constraints.

If you're reading the source, the highest-leverage entry point is [`src/ultron/pipeline/orchestrator.py`](src/ultron/pipeline/orchestrator.py) — that's the main event loop everything else hangs off.

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Acknowledgments

Built on top of (in alphabetical order): [bge-small](https://huggingface.co/BAAI/bge-small-en-v1.5), [Claude Code](https://docs.claude.com/claude-code), [DuckDuckGo](https://duckduckgo.com/), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [flan-t5-small](https://huggingface.co/google/flan-t5-small), [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M), [llama.cpp](https://github.com/ggerganov/llama.cpp), [moondream2](https://huggingface.co/vikhyatk/moondream2), [Moonshine](https://github.com/usefulsensors/moonshine), [opencode](https://github.com/sst/opencode) (event bus + project digest pattern inspiration), [openWakeWord](https://github.com/dscripka/openWakeWord), [Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), [Piper](https://github.com/rhasspy/piper) (legacy TTS), [pywinauto](https://github.com/pywinauto/pywinauto), [Qdrant](https://qdrant.tech/), [Qwen 3.5](https://huggingface.co/Qwen), [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) (legacy voice conversion), [SearxNG](https://github.com/searxng/searxng), [Silero VAD](https://github.com/snakers4/silero-vad), [Smart Turn V3](https://huggingface.co/pipecat-ai/smart-turn-v3) (end-of-turn detection), [Trafilatura](https://github.com/adbar/trafilatura), [XTTS v2](https://huggingface.co/coqui/XTTS-v2) (alternative TTS).
