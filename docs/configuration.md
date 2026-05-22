# Ultron configuration reference


> **Currency note (2026-05-22):** this document is a historical snapshot.
> For the **current** state (DualSTTRegistry, Kokoro TTS, qwen3.5-4b,
> intent recognizer, supervisor stack, news-category SearxNG routing,
> gaming-mode VRAM reclaim, event bus, OPEN_LAST_SOURCE / NAVIGATE_TO_SITE
> intents, etc.), see [`codebase_structure.md`](codebase_structure.md)
> which is kept current via the binding maintenance contract. The
> high-level shape and intent here are still accurate; specific subsystem
> identities and per-knob defaults have evolved.

Single source of truth: [config.yaml](../config.yaml) at the project root.
Loader + schema: [src/ultron/config.py](../src/ultron/config.py).
Discovery doc (one-time): [config_discovery.md](config_discovery.md).

## Reading config

```python
from ultron.config import get_config

cfg = get_config()
if cfg.web_search.enabled:
    timeout = cfg.web_search.brave.timeout_seconds
```

The first call to `get_config()` loads + validates `config.yaml`. The
result is a process-singleton; subsequent calls are O(1).

## File path resolution

All path values in config.yaml are interpreted relative to the project
root unless absolute. Use `resolve_path()` to convert:

```python
from ultron.config import get_config, resolve_path

p = resolve_path(get_config().llm.model_path)  # absolute Path
```

`PROJECT_ROOT`, `MODELS_DIR`, `LOGS_DIR` are also exported as Path
constants.

## Env var substitution

Any string value in `config.yaml` may contain `${VAR_NAME}` to substitute
an environment variable. Missing vars resolve to empty string; the
consuming subsystem must handle empty values explicitly. Used for
secrets (Brave API key) and platform-specific paths:

```yaml
coding:
  claude_cli: "${USERPROFILE}/AppData/Roaming/npm/claude.cmd"
```

## Loader API

| Function | Purpose |
|---|---|
| `get_config() -> UltronConfig` | Singleton accessor. Auto-loads on first call. |
| `load_config(path=None) -> UltronConfig` | Force-load from explicit path or `ULTRON_CONFIG_PATH` env var. Caches. |
| `reload_config(path=None) -> UltronConfig` | Clear cache and reload. Some sections need a process restart. |
| `set_config(cfg)` | Test-only: inject a pre-built config. |
| `current_config_path() -> Path \| None` | Path the singleton was loaded from. |
| `resolve_path(p) -> Path` | Convert a config-file path to an absolute Path under PROJECT_ROOT. |

## Validation

`UltronConfig` (and every sub-model) uses `extra="forbid"` — unknown
keys fail validation at startup with a path-aware error. Many fields
have range constraints (`Field(ge=..., le=...)`); violations are
reported with the offending value.

To run with all schema defaults (no file), construct in Python:

```python
from ultron.config import UltronConfig, set_config
set_config(UltronConfig())
```

This is for tests only — production always reads a file.

## What lives where, what gets tuned how

The discovery doc ([config_discovery.md](config_discovery.md)) has the
full per-key catalog mapping every legacy `settings.X` constant to its
new home. Below is the section-level overview with tuning notes.

### `audio` — microphone capture

| Key | Default | Tuning notes |
|---|---|---|
| `sample_rate` | 16000 | Required by Silero VAD, openWakeWord, Whisper. Don't change without re-tuning all three. |
| `channels` | 1 | Mono. Stereo isn't supported by the wake-word path. |
| `blocksize` | 512 | 32 ms at 16 kHz. Smaller = lower latency, higher CPU. |
| `dtype` | "float32" | sounddevice constraint. |
| `input_device` | null | `null` = system default. Override per-host via env var `ULTRON_AUDIO_DEVICE` (substring match on device name). |
| `output_device` | null | Same; env var `ULTRON_AUDIO_OUTPUT_DEVICE`. |
| `barge_in_enabled` | true | Wake-word fires during TTS playback to interrupt Ultron mid-sentence. |
| `barge_in_grace_seconds` | 0.5 | Initial deaf period after TTS starts so Ultron's own onset doesn't self-trigger. |
| `ring_buffer_seconds` | 0.5 | Pre-speech audio kept so VAD-detected utterances don't clip the leading word. |

### `vad` — voice activity detection

| Key | Default | Tuning notes |
|---|---|---|
| `threshold` | 0.5 | silero-vad confidence cutoff. Lower = more permissive (false positives ↑). |
| `min_speech_duration_ms` | 250 | Below this, treat as a blip (don't emit speech-start). |
| `min_silence_duration_ms` | 500 | Silence required to emit speech-end. Higher = more pause-tolerant; lower = snappier turn-taking. |
| `window_samples` | 512 | Silero v5 hard requirement at 16 kHz. |

### `wake_word`

| Key | Default | Tuning notes |
|---|---|---|
| `name` | "ultron" | The trigger word (info only — model selection comes from `model_path`). |
| `model_path` | "models/openwakeword/ultron.onnx" | Custom-trained Ultron ONNX. |
| `fallback_model` | "hey_jarvis" | Used (with startup warning) if custom model missing. One of openWakeWord's pretrained set. |
| `threshold` | 0.5 | Detection confidence cutoff. Raise toward 0.7 if Ultron's own voice keeps re-triggering. |
| `cooldown_seconds` | 1.5 | Debounce window — same wake event won't fire twice. |

### `stt` — Whisper

| Key | Default | Tuning notes |
|---|---|---|
| `model` | "small.en" | `base.en` for lower latency on weak GPUs; English-only models are faster than `small`/`base`. |
| `device` | "cuda" | Set "cpu" if you must; Whisper-cpu is ~5–10× slower. |
| `compute_type` | "float16" | `int8_float16` saves memory at minor accuracy cost. |
| `beam_size` | 5 | Higher = better quality, slower. 1 is greedy. |
| `temperature` | 0.0 | Set >0 only when chasing alternative hypotheses for tricky audio. |
| `condition_on_previous_text` | false | False is more robust to long sessions; true gives slightly higher coherence on multi-turn. |
| `vad_filter` | false | We already gate on Silero upstream; double-VADing adds latency. |

### `llm`

| Key | Default | Tuning notes |
|---|---|---|
| `provider` | "llama_cpp" | Pinned. See [feedback_llm_runtime_decision.md](<ai-memory-dir>\feedback_llm_runtime_decision.md). |
| `preset` | "qwen3.5-9b" | One of `"qwen3.5-9b" \| "qwen3.5-4b" \| "custom"`. The preset auto-fills `model_path`, `n_ctx`, and `draft_model_path` *only when those keys are absent from the YAML* — explicit user values always win. **Override at runtime via `ULTRON_LLM_PRESET=...`** (clears YAML overrides too unless `ULTRON_LLM_PRESET_KEEP_OVERRIDES=1`). Use `"custom"` to disable auto-resolution. See [src/ultron/config.py:LLM_PRESETS](../src/ultron/config.py), [scripts/swap_llm_preset.py](../scripts/swap_llm_preset.py), and [docs/4b_optimization_plan.md](4b_optimization_plan.md). |
| `model_path` | "models/Qwen3.5-9B-Q4_K_M.gguf" | Env override: `ULTRON_LLM_MODEL_PATH`. With non-`"custom"` preset, omitting this key inherits from the preset. |
| `draft_model_path` | null | Optional speculative-decoding draft GGUF. Set by the `qwen3.5-4b` preset to `"models/Qwen3.5-0.8B-Q4_K_M.gguf"`. Wired into `scripts/start_llamacpp_server.py` in Stage C of the 4B plan. |
| `n_ctx` | 8192 | Context length. Larger costs proportional KV cache; this is the budgeted size. The `qwen3.5-4b` preset bumps this to 16384. |
| `gpu_layers` | -1 | -1 = all layers on GPU. Reduce (e.g. 30) to spill to CPU on small VRAM. |
| `default_temperature` | 0.7 | |
| `default_top_p` | 0.9 | |
| `default_max_tokens` | 512 | Per-call cap unless caller overrides. |
| `default_repeat_penalty` | 1.1 | |
| `history_turns` | 6 | Legacy fallback for when memory is disabled — recent-turn count to keep in context. |
| `flash_attn` | true | Required for non-F16 KV cache. |
| `kv_cache_type` | 8 | `8` = GGML_TYPE_Q8_0 (~½ VRAM vs F16); `1` = F16. |
| `system_prompt` | (Ultron persona) | Multi-line block scalar. Edit to retune voice. |
| `rag.position` | `"recency"` | One of `"system" \| "recency"`. Where retrieved Qdrant memories land in the LLM context. `"recency"` (Stage G default) prepends them to the user message — strongest-attention zone, +10–20% recall on the 4B. `"system"` folds them into the leading system message (legacy / rollback). |

### `embeddings`

| Key | Default | Notes |
|---|---|---|
| `dense_model` | "BAAI/bge-small-en-v1.5" | INT8 ONNX via FastEmbed; CPU only. |
| `sparse_model` | "Qdrant/bm25" | FastEmbed pretrained BM25. |
| `dense_dim` | 384 | Must match the dense model. |

### `qdrant`

| Key | Default | Notes |
|---|---|---|
| `data_dir` | "data/qdrant" | Embedded Qdrant store path. **Differs from the Foundation prompt's `./qdrant_data/` example** — matches existing layout. |
| `collections.conversations` | "conversations" | Per-turn embedded conversation history. |
| `collections.facts` | "facts" | Extracted facts (populated by maintenance). |
| `collections.web_results` | "web_results" | Cached Brave + Jina rows. |

### `memory`

| Key | Default | Notes |
|---|---|---|
| `enabled` | true | Disable to fall back to history-deque only (no Qdrant). |
| `jsonl_legacy_path` | "data/memory.jsonl" | Migration source / recovery fallback. |
| `recent_turns` | 20 | In-process recent cache size. Hot path serves from this. |
| `rag_top_k` | 5 | Top-K turns retrieved per query. |
| `rag_exclude_recent` | 20 | Skip recent-window turns when ranking RAG hits (avoid duplication). |
| `facts_top_k` | 3 | Top-K facts retrieved alongside conversation. |
| `write_queue_maxsize` | 256 | Background writer queue cap; drops with WARN if exceeded. |

### `web_search`

| Key | Default | Notes |
|---|---|---|
| `enabled` | true | Master switch. |
| `brave_api_key_env` | "ULTRON_BRAVE_API_KEY" | Env var name to read for the Brave key. Key value stays in env, never in config. |
| `brave.endpoint` | (Brave URL) | |
| `brave.count` | 5 | Results per query. |
| `brave.timeout_seconds` | 8.0 | |
| `brave.rate_limit_seconds` | 2.0 | Min seconds between Brave calls — free-tier protection. |
| `jina.endpoint` | "https://r.jina.ai/" | |
| `jina.timeout_seconds` | 15.0 | |
| `jina.max_fetch` | 3 | How many ranked snippets get full-text fetches. |
| `jina.max_bytes` | 200000 | Truncate giant pages so they don't blow up the LLM prompt. |
| `cache.ttl_volatile_seconds` | 86400 | 24 h — sports, weather, stocks. |
| `cache.ttl_stable_seconds` | 2592000 | 30 d — historical, definitional. |

### `addressing`

| Key | Default | Notes |
|---|---|---|
| `follow_up_enabled` | true | After Ultron speaks, listen for follow-up without requiring wake word. |
| `warm_mode_duration_seconds` | **30.0** | **Differs from the Foundation prompt's 10s example** — user override per [feedback_ultron_extension.md](<ai-memory-dir>\feedback_ultron_extension.md). Don't re-tighten without asking. |
| `default_uncertain_to_not_addressed` | true | Default-silent when classifier is uncertain. |
| `rule_confidence_threshold` | 0.8 | Rule verdicts above this short-circuit zero-shot. |
| `zero_shot_model` | "google/flan-t5-small" | CPU-only; ~300 MB. |
| `load_eagerly` | true | Load Flan-T5 at startup (~8 s) instead of on first ambiguous utterance. |
| `log_path` | "logs/addressing.jsonl" | |

### `coding`

Top-level: `enabled`, `bridge` ("direct" runs Claude Code as a subprocess; an `openclaw` slot is reserved but Part 5 of Foundation removes it).

| Subsection | Key | Default | Notes |
|---|---|---|---|
| `mcp` | `host` / `port` | 127.0.0.1 / 19761 | Localhost-only by design. |
| `mcp` | `clarification_timeout_seconds` | 600 | Long because user may need time to think. |
| `mcp` | `log_path` | "logs/mcp_calls.jsonl" | Audit log for tool calls. |
| (root) | `template_dir` | "prompts/coding" | Jinja templates for Claude prompts. |
| (root) | `prompt_token_budget` | 4000 | Hard cap on rendered prompt size. |
| (root) | `default_model` / `escalation_model` | "haiku" / "sonnet" | First try Haiku; escalate after threshold. |
| (root) | `escalation_threshold_default` / `_escalation` | 3 / 2 | Failures before escalating; failures on escalation model before failing the session. |
| `verification` | `smoke_timeout_seconds` | 5 | Cap on smoke-run check. |
| `verification` | `test_timeout_seconds` | 120 | |
| `verification` | `lint_timeout_seconds` | 30 | |
| (root) | `session_audit_dir` | "logs/sessions" | Per-session JSONL audit. |
| (root) | `token_budget_per_session` | 100000 | When 80 % crosses, warn user; at 100 % halt. |
| (root) | `token_warning_threshold` | 0.8 | |
| (root) | `progress_timeout_seconds` | 300 | Stall warning if no events from Claude for this long. |
| (root) | `claude_cli` | "${USERPROFILE}/AppData/Roaming/npm/claude.cmd" | Override via `ULTRON_CLAUDE_CLI` env. |
| (root) | `sandbox_root` | "data/sandbox" | New projects under this; existing projects can live anywhere via the registry. |
| (root) | `project_registry_path` | "data/projects.json" | |
| (root) | `audit_log_path` | "logs/coding_tasks.jsonl" | |
| (root) | `task_timeout_seconds` | 1800 | Outer cap on a single task (30 min). |
| (root) | `skip_permissions` | true | Pass `--dangerously-skip-permissions` to Claude Code. Sandbox is project-local. |

### `projections`

| Key | Default | Notes |
|---|---|---|
| `tokenizer` | "tiktoken_cl100k_base" | Token-counting backend. |
| `budgets.clarification_context` | 1500 | |
| `budgets.status_delta` | 600 | |
| `budgets.adjustment_context` | 1200 | |
| `budgets.correction_context` | 1500 | |
| `budgets.completion_context` | 800 | |
| `truncation_warning_threshold` | 0.95 | Above this fraction of budget, log WARN even when we fit. |
| `log_truncations` | true | Gate INFO log when truncations are applied. |

### `tts` + `tts.rvc`

| Key | Default | Notes |
|---|---|---|
| `piper_voice_path` | en_US-ryan-medium ONNX | |
| `output_sample_rate` | 22050 | Piper's native rate for medium voices. |
| `sentence_flush_chars` | ".!?\n" | Only strong sentence terminators trigger a TTS flush. |
| `piper_length_scale` | 1.15 | Legacy `piper_rvc` engine only; >1 = slower / more deliberate. |
| `pause_ms` | 180 | Silence at sentence boundaries (all engines). Currently set to 50 in config.yaml for snappy cadence. |
| `edge_fade_ms` | 4 | Short fade at clip edges to prevent clicks. |
| `rvc.enabled` | true | Voice conversion (Piper → Ultron). |
| `rvc.device` | "cuda:0" | RVC inference device. |
| `rvc.pitch_shift` | -2 | Semitones; lower = deeper. |
| `rvc.index_rate` | 0.66 | 0–1; higher = stricter match to trained timbre. |
| `rvc.protect` | 0.45 | 0–0.5; higher preserves Piper consonants (crisp s/t/k). |
| `rvc.f0_method` | "rmvpe" | Most accurate pitch extractor. |
| `rvc.rms_mix_rate` | 0.35 | Higher lets Piper loudness through; reads as more articulate. |
| `rvc.filter_radius` | 1 | Median filter on F0 — lower preserves stressed-syllable detail. |

### `logging`

| Key | Default | Notes |
|---|---|---|
| `file` | "logs/ultron.log" | Rotating handler at DEBUG. |
| `level` | "INFO" | Console handler level. Override via `ULTRON_LOG_LEVEL`. |
| `format` / `datefmt` | (standard) | |

## Hot reload

`reload_config()` clears the singleton and re-reads from disk. **Some
sections require a process restart even after hot-reload** because
their values are baked into long-lived objects:

- `llm.model_path`, `llm.n_ctx`, `llm.gpu_layers` — LLM is loaded once
- `qdrant.data_dir` — Qdrant client opens at startup
- `audio.sample_rate` — sound device is opened with this rate
- `tts.piper_voice_path`, `tts.rvc.*` — Piper / RVC loaded once

Sections safe to hot-reload include logging level, projection budgets
and thresholds, addressing rule confidence threshold, web cache TTLs,
and most coding orchestration knobs.

## Migration status (Foundation Phase 3)

This Phase 3 introduces `config.yaml` + the loader as the canonical
source of truth. A subset of subsystems already read directly via
`get_config()`; the rest go through a thin re-export shim at
[config/settings.py](../config/settings.py) that derives every
constant from `get_config()`. Behavior is identical either way — the
shim is a transitional artifact, not a parallel source of truth.

| Subsystem | Direct `get_config()` | Through shim | Notes |
|---|---|---|---|
| logging | ✓ | | [src/ultron/utils/logging.py](../src/ultron/utils/logging.py) |
| addressing | ✓ | | [src/ultron/pipeline/orchestrator.py](../src/ultron/pipeline/orchestrator.py) construction site + [scripts/review_addressing.py](../scripts/review_addressing.py) |
| web_search | ✓ | | [src/ultron/web_search/{brave,jina,search,cache}.py](../src/ultron/web_search) |
| llm | ✓ | | [src/ultron/llm/inference.py](../src/ultron/llm/inference.py) |
| embeddings + memory + qdrant | ✓ | | [src/ultron/memory/{embedder,qdrant_store}.py](../src/ultron/memory) |
| projections | ✓ | | [src/ultron/coding/projections.py](../src/ultron/coding/projections.py) (config wired into `_finalize_projection`) |
| audio + VAD | | ✓ | Through shim. |
| wake_word | | ✓ | Through shim. |
| stt (Whisper) | | ✓ | Through shim. |
| tts (Piper) | | ✓ | Through shim. |
| tts.rvc | | ✓ | Through shim. |
| coding (bridge, runner, mcp_server, coordinator, narration, projects, session, templates, verification, voice, audit) | | ✓ | Through shim — 14 files. |
| uncertainty | | ✓ | Through shim. |
| scripts (benchmark, download_models, maintenance, migrate_memory, measure_baseline_extended) | | ✓ | Through shim. |

The shim path is fully functional. Migrating the remaining subsystems
to direct `get_config()` reads is mechanical follow-up work — pattern
demonstrated in the six already-migrated sections; tests gate every
removal of a shim re-export line.
