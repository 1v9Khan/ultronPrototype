<div align="center">

# Kenning

### A local, voice-first AI assistant — no cloud, no telemetry, sub-second latency.

*Say "kenning." Talk. Get answers in a custom voice. Everything runs on your GPU.*

[![tests](https://img.shields.io/badge/tests-10133%20passing-brightgreen?style=flat-square)](https://github.com/1v9Khan/ultronPrototype)
[![latency](https://img.shields.io/badge/TTFA-~266ms-blueviolet?style=flat-square)](#-at-a-glance)
[![VRAM](https://img.shields.io/badge/VRAM-6.3GB%20standby-orange?style=flat-square)](#-at-a-glance)
[![python](https://img.shields.io/badge/python-3.11+-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![cuda](https://img.shields.io/badge/CUDA-12.4+-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-downloads)
[![platform](https://img.shields.io/badge/platform-Windows-0078D6?style=flat-square&logo=windows&logoColor=white)](https://github.com/1v9Khan/ultronPrototype)
[![license](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)

</div>

---

## ⚡ Why Kenning?

> **What would a voice assistant feel like if it lived entirely on your GPU instead of in someone else's data center?**

- 🔒 **Fully local.** Your voice, your queries, your context — none of it leaves the machine.
- ⚡ **Fast.** ~210–300 ms from "stop talking" to "Kenning starts speaking" on a cache-hit turn.
- 🧠 **Smart.** 23-kind intent router · three-layer memory · hot-swappable models · gaming-mode VRAM reclaim.
- 🎙️ **Yours.** Custom wake word · fine-tuned voicepack · your apps in the launcher · your safety rules.

---

## 🎬 What you say → What it does

| You say | Kenning does |
|---|---|
| 🌦️ &nbsp;`"kenning, what's the weather in Paris?"` | Detects fresh-data intent → SearxNG → reads result → speaks the forecast |
| 💻 &nbsp;`"kenning, write me a script that converts PDFs to Docx"` | Spawns isolated AI coding agent → scaffolds project → runs tests → narrates progress |
| 🎮 &nbsp;`"kenning, engage gaming mode"` | Swaps LLM → kills GPU services → frees **~2.3 GB VRAM** for your game |
| 🌐 &nbsp;`"kenning, take me to HBO Max"` | Recognizes navigate intent → opens Chrome to the best-matching domain |
| 🕐 &nbsp;`"kenning, what time is it in Tokyo?"` | Hits local zoneinfo cache → speaks the answer in ~5 ms (no LLM, no search) |
| 🧭 &nbsp;`"kenning, switch to the 8B"` | Hot-swaps the local LLM preset mid-conversation |
| 🗣️ &nbsp;`"kenning, tell my team they are pushing B"` | Valorant teammate-relay: tactical callouts resolve **deterministically** (subject-exact, every count/agent/location/ability preserved, never the LLM) — a fact-preserving fallback relays the literal rather than let the model drop or invert a callout. Nearly every line then carries a short, in-character **Ultron** flavor tail (a faithful *Avengers: Age of Ultron* clone) **selected for the callout** — agent-specific for a named enemy (*"Their Neon has ult. Overdrive. A finite surge."*), plural for a group, owner-aware (contempt at enemies, cold command for your orders, stoic for your own status — never mocking you) — from a ~2,100-line character-tailored library covering all 29 agents, adversarially audited to stay in-character. Off-snap lines (banter, economy, opinions, identity, Marvel, answers, greets) get the full persona → plays on a VoiceMeeter strip so your voice chat hears it. When you don't trust the model to improvise, **73 explicit fallback commands** (refuse a dumb question, criticize/praise a teammate by name, call out a throw, status reports, strategy with map callouts) each resolve to one of up to 40 curated full-Ultron lines. Covers all agents & maps and holds real conversation, no wake word mid-conversation; validated against a 20,000-case adversarial corpus (≈95% tactical-fact retention, ~67% personality coverage, deterministic path ~0.15 ms) |
| 🔊 &nbsp;`"kenning, repeat to my team watermelon"` | The soundboard check — when teammates ask you to say a specific word to prove a human's on comms, Ultron speaks the **exact** phrase verbatim (no LLM, any literal word) in his trained voice |
| 🎛️ &nbsp;`"kenning, pull up your settings"` | Spawns a detached dark-theme control panel → edit knobs at a glance → every toggle hot-applies live (no restart) → CLOSE leaves zero residue |
| 🎵 &nbsp;`"kenning, play some Daft Punk"` | Full hands-free Spotify control by voice — play / queue / "play X next" / pause / resume / skip / previous / restart / "what's playing" / volume up·down·"set it to 40" / mute·unmute / shuffle / repeat / like·unlike — understanding dozens of natural phrasings, with confirmations in Ultron's cold machine register. Web API over HTTPS only (no GPU, no LLM) so it stays live **even in gaming/anticheat mode** |
| 🛡️ &nbsp;`"kenning, engage gaming mode"` | Frees VRAM/Docker **and** hard-disables every desktop-interaction surface (input, capture, windows) — kernel-anticheat-safe; voice + team relay stay live |

---

## 📊 At a glance

|  |  |
|---|---|
| 🧪 &nbsp;**Tests** | 10018 passing · 39 skipped · 0 failed (~155 s sweep) |
| ⚡ &nbsp;**Latency (TTFA)** | ~266 ms composite cache-hit turn (LLM TTFT 172 ms, TTS synth 78 ms, STT 16 ms) |
| 🧠 &nbsp;**VRAM** | ~6.3 GB standby on RTX 4070 Ti (peak ~6.7 GB) → ~2.1 GB in gaming mode |
| 🛠️ &nbsp;**Active stack** | Parakeet TDT STT (CUDA) · Qwen 3.5 4B Q4_K_M (CUDA) · Kokoro StyleTTS2 (CUDA, fine-tuned voice) · OpenClaw bridge live |
| 📜 &nbsp;**License** | MIT |

---

## ✨ Features

<table>
<tr>
<td width="50%" valign="top">

### 🎤 Voice pipeline
- Custom-trained `kenning` / `ultron` wake words (OpenWakeWord), hot-swappable from the settings panel, with **per-word thresholds** + a **consecutive-frame gate** to reject confusables without retraining
- Silero VAD + **Smart Turn V3** — semantic end-of-turn in ~12 ms
- Dual-engine STT registry (Moonshine · Parakeet TDT · Whisper)
- **Custom fine-tuned voicepack** — Kokoro StyleTTS2 on CUDA
- **In-model prosody shaping** — scales the model's own pitch / energy / per-phoneme duration curves *before* the decoder for expressive, naturally-paced delivery at **zero added latency** (timbre + reverb preserved)
- Producer-consumer audio pipeline; clip N+1 synth overlaps clip N playback
- Boundary-artifact mute via cosine fades + tail aggressive zero
- **Game team-relay** — deterministic, fact-preserving snap callouts carrying short in-character *Age-of-Ultron* flavor (agent-specific, owner-aware) + full-persona off-snap lines, routed to a separate game-chat output strip; understands bare comms shorthand ("cypher is flank" → enemy callout), **73 explicit fallback commands** (~2,800 curated lines) for when you don't want the model improvising, and a **verbatim "repeat to my team X"** soundboard-check command
- **Streamer output routing** — plays to your default speakers *and*, in parallel, tees team-only callouts to one virtual device and **all** speech to another (VoiceMeeter B1/B3), with the listen mic untouched — zero added speaker-path latency
- **Voice waveform overlay** — a separate borderless, always-on-top window with a circular visualizer + a neon **ULTRON** nameplate (white-hot, readable letters in a soft Gaussian red bloom) that pulses as he speaks; add it in OBS as one Window Capture (background mode lets it hide behind your desktop yet stay captured)

</td>
<td width="50%" valign="top">

### 🧠 Reasoning
- Local LLM in-process via `llama-cpp-python`
- Speculative decoding wired (prompt-lookup + draft model)
- Hot-swap presets by voice: `"switch to the 8B"`
- Three-layer memory: recent cache · RAG (bge-small + BM25 RRF) · project digest
- Adaptive context window scoring; ambiguity gating
- Tiered web-search freshness gate (regex → semantic intent → preflight LLM)

</td>
</tr>
<tr>
<td valign="top">

### 🌐 Web + tools
- Local-first **SearxNG** (Docker) → Brave → DuckDuckGo cascade
- Trafilatura → Jina reader cascade
- 23-kind routing intent classifier
- Native desktop automation (12-entry launcher: Chrome, Discord, Spotify, +9)
- News-category routing for current-events queries
- Optional **OpenClaw** peer gateway for proactive comms

</td>
<td valign="top">

### 🛡️ Safety + ops
- **141-rule** tool-call validator across 19 categories
- Tamper-evident SHA-256 hash-chain audit log
- **Gaming-mode** VRAM reclaim chain (~2.3 GB freed on demand) + a **bare-bones profile** (optionally auto-engaged at boot): LLM swapped to a **CPU-only** 3B, Kokoro TTS → CPU, Parakeet stopped, VLM unloaded, and per-turn RAG retrieval / reranker / web-search skipped — near-zero GPU so it never costs game frames, while the voice + team relay stay live
- **Anticheat-safe mode** — a 3-layer hard block (module guards · validator BLOCK_HARD · surface-stop hooks) on *every* desktop-interaction surface (input injection, screen capture, OCR, UIA, clipboard, window control, browser CDP), pinnable always-on for running beside kernel anticheats; audio + the voice/team relay + the overlay stay live
- Typed event bus — `turn.started` · `gate.verdict` · `supervisor.decided` · 14 more
- opencode-inspired project digest + supervisor stack
- Pre-push hygiene hook on the repo itself

</td>
</tr>
</table>

---

## 🏗️ Pipeline

```text
mic → wake "kenning" OR addressing classifier (WARM)
  → Silero VAD + Smart Turn V3 (CPU, ~12 ms)
  → STT: DualSTTRegistry (moonshine | parakeet | whisper)
  → Intent recognizer (Gemma-300M CPU): short-circuits gaming / fresh-data intents
  → Local clock reply for bare time/date queries (~5 ms, no LLM)
  → classify_routing() → 23 RoutingIntentKind dispatches
      ├─ coding kinds → AI coding agent subprocess (optional supervisor stack)
      ├─ OPEN_LAST_SOURCE → opens cited URL from prior search
      ├─ NAVIGATE_TO_SITE → SearxNG top-10 → domain-score → opens best
      ├─ APP_LAUNCH        → native Chrome/Discord/Spotify launcher
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

> 📖 **Full per-module reference:** [`docs/codebase_structure.md`](docs/codebase_structure.md) — the binding single-source map of the system.

---

## 🚀 Quick start

```bash
# 1. Clone
git clone https://github.com/1v9Khan/ultronPrototype.git
cd ultronPrototype

# 2. Python 3.11 + deps (~7 GB; PyTorch CUDA, llama-cpp, faster-whisper, Kokoro, ...)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS (untested)
pip install -e .

# 3. Models (~5 GB; wake word, Smart Turn, Moonshine, Kokoro, Qwen GGUFs)
python scripts/download_models.py

# 4. Configure
copy .env.example .env          # optional: add Brave API key for web search
# tune config.yaml for your mic / monitors / preferences

# 5. Launch
python -m kenning
```

Then say: **"kenning"** — and start talking.

> ⚠️ **This is a research prototype**, not a turn-key product. It targets one developer's specific hardware (RTX 4070 Ti, AMD CPU, Windows 11) and use case. Treat the setup as a recipe to adapt, not a one-click install. Some optional integrations (OpenClaw, Telegram, ComfyUI media gen, mobile node) require additional credential-dependent setup — see the docs below.

---

## 💻 System requirements

|  | Recommended | Minimum |
|---|---|---|
| **GPU** | RTX 4070 Ti (12 GB) | RTX 3060 (12 GB) — untested, expect higher latency |
| **CPU** | AMD Ryzen 7 5800X+ (8c/16t) | 4 cores / 8 threads |
| **RAM** | 32 GB | 16 GB (constrained) |
| **Disk** | 30 GB free | 20 GB free |
| **OS** | Windows 11 | Windows 10 / Linux (untested) |
| **Python** | 3.11 | 3.10+ |
| **CUDA** | 12.4+ | 11.8 |

---

## ⚙️ Configuration

All tunables live in `config.yaml` at the project root — schema-validated by Pydantic in `src/kenning/config.py`. The top of that file lists the ~12 actively-tuned knobs.

Key sections:

| Section | What it controls |
|---|---|
| `audio` | Mic input device + output device + ring buffer |
| `vad` · `stt` | VAD silence thresholds + STT engine selector + gaming fallback |
| `llm` | Preset + n_ctx + speculative decoding + KV cache |
| `tts` | Engine + voicepack + boundary smoothing + cadence |
| `memory` | Qdrant store + RAG top-k + min-relevance + contextual retrieval |
| `web_search` | Provider chain + reader chain + ranker dispatch |
| `safety` | 141 rule toggles + sandbox roots + audit log path |
| `coding.supervisor` | opencode-inspired project digest stack (default OFF) |
| `gaming_mode` | VRAM reclaim chain triggers + targets |

Override via `KENNING_*` env vars; see `.env.example`. Restart after any change.

---

## 📚 Documentation

> 👉 **Start here:** [`docs/codebase_structure.md`](docs/codebase_structure.md) — the binding single-source map of every module, script, test, and runtime artifact. Maintenance contract enforced per commit.

<details>
<summary><b>🏛️ Architecture + operations</b></summary>

| Doc | Topic |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Pipeline overview + hardware target |
| [`docs/configuration.md`](docs/configuration.md) | Per-key config reference |
| [`docs/operations.md`](docs/operations.md) | Day-to-day running + recovery |
| [`docs/development.md`](docs/development.md) | Test layout + debugging recipes |
| [`docs/routing.md`](docs/routing.md) | Capability routing |
| [`docs/error_handling.md`](docs/error_handling.md) | Error catalog |
| [`docs/4b_optimization_plan.md`](docs/4b_optimization_plan.md) | 4B LLM migration (complete) |

</details>

<details>
<summary><b>🔌 OpenClaw integration</b> — peer gateway for proactive comms + tools</summary>

| Doc | Topic |
|---|---|
| [`docs/openclaw_integration_final_summary.md`](docs/openclaw_integration_final_summary.md) | Cross-phase summary + setup checklist |
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
<summary><b>🧪 Test pass reports</b></summary>

| Doc | Topic |
|---|---|
| [`docs/comprehensive_test_plan.md`](docs/comprehensive_test_plan.md) / [`comprehensive_test_report.md`](docs/comprehensive_test_report.md) | Functional pass (16 phases, 38 dimensions) |
| [`docs/comprehensive_quality_plan.md`](docs/comprehensive_quality_plan.md) / [`comprehensive_quality_report.md`](docs/comprehensive_quality_report.md) | Quality pass (Q0–Q13, 38 dimensions, prompt-injection defense audit) |
| [`docs/smoke_test.md`](docs/smoke_test.md) | 16-step interactive smoke procedure |

</details>

---

## 🧭 Project status

This is a **research prototype**, not a production product. It evolves through many tight iteration cycles. Behavior-changing features land behind feature flags (default OFF) until live-validated. The voice-quality baseline is treated as a strict latency / VRAM contract — any hot-path change re-runs `scripts/measure_baseline.py` and documents the delta.

If you're reading the source, the highest-leverage entry point is [`src/kenning/pipeline/orchestrator.py`](src/kenning/pipeline/orchestrator.py) — the main event loop everything else hangs off.

---

## ⭐ Star history

If you find Kenning interesting, a star helps it surface to other folks who want a local voice assistant.

[![Star History Chart](https://api.star-history.com/svg?repos=1v9Khan/ultronPrototype&type=Date)](https://star-history.com/#1v9Khan/ultronPrototype&Date)

---

## 📜 License

MIT — see [`LICENSE`](LICENSE).

---

## 🙏 Acknowledgments

Built on the shoulders of these open-source projects:

[bge-small](https://huggingface.co/BAAI/bge-small-en-v1.5) · [DuckDuckGo](https://duckduckgo.com/) · [faster-whisper](https://github.com/SYSTRAN/faster-whisper) · [flan-t5-small](https://huggingface.co/google/flan-t5-small) · [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) · [llama.cpp](https://github.com/ggerganov/llama.cpp) · [moondream2](https://huggingface.co/vikhyatk/moondream2) · [Moonshine](https://github.com/usefulsensors/moonshine) · [opencode](https://github.com/sst/opencode) · [openWakeWord](https://github.com/dscripka/openWakeWord) · [Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) · [Piper](https://github.com/rhasspy/piper) · [pywinauto](https://github.com/pywinauto/pywinauto) · [Qdrant](https://qdrant.tech/) · [Qwen 3.5](https://huggingface.co/Qwen) · [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) · [SearxNG](https://github.com/searxng/searxng) · [Silero VAD](https://github.com/snakers4/silero-vad) · [Smart Turn V3](https://huggingface.co/pipecat-ai/smart-turn-v3) · [Trafilatura](https://github.com/adbar/trafilatura) · [XTTS v2](https://huggingface.co/coqui/XTTS-v2)

<div align="center">

---

<sub>Built for one developer's RTX 4070 Ti, then shared.</sub>

</div>
