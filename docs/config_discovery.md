# Config discovery — pre-Phase 3 catalog of every config source

Generated as Part 3.2 of the Foundation phase. Inventories every place a
tunable parameter currently lives, before consolidating into a single
`config.yaml`. After Phase 3 completes, the "after migration" column tells
you the new home of each value.

## Methodology

- `os.getenv` / `os.environ.get` reads inside `src/` and `config/`
- Module-level constants in `config/settings.py`
- Explicit env-var helpers `_env_bool / _env_int / _env_float` in `config/settings.py`
- Dataclass / function-signature defaults that look tunable
- Hardcoded literals in subsystem modules (separate from settings.py)

External secrets that stay in environment variables (NEVER in config.yaml):
- `ULTRON_BRAVE_API_KEY` — Brave Search API key
- `ULTRON_LLM_MODEL_PATH` — opt-in override of model path
- `ULTRON_AUDIO_DEVICE` / `ULTRON_AUDIO_OUTPUT_DEVICE` — operator-specific device strings

## Counts

- `config/settings.py` LOC: 494
- Module-level constants in settings.py: ~110
- `os.getenv` calls in settings.py: 19 (excluding 3 for HF cache pre-init)
- `_env_*` helper calls: 35
- Files outside `config/` that read env vars directly: **1** ([src/ultron/coding/mcp_server.py:476](src/ultron/coding/mcp_server.py:476) reads `ULTRON_CODING_MCP_ALLOW_ANY_ROOT` for test escape)
- Files referencing `settings.X`: **35** total (~280 reference sites) — see Section 4

Conclusion: with one exception (`ULTRON_CODING_MCP_ALLOW_ANY_ROOT`), every
tunable already funnels through `config/settings.py`. The migration's job
is to move the SOURCE of those values from settings.py constants into
`config.yaml`, then update subsystems to read from `get_config()` directly
and remove the now-empty settings.py constants.

## Decisions baked into config.yaml at Phase 3

These three deviations from the Foundation prompt's example config are
intentional — they reflect facts on the ground, recorded in memory:

| Item | Foundation example | Actual / config.yaml | Reason |
|---|---|---|---|
| `addressing.warm_mode_duration_seconds` | 10 | **30** | User override per [feedback_ultron_extension.md](<ai-memory-dir>\feedback_ultron_extension.md) |
| `llm.provider` | `"ollama"` | **`"llama_cpp"`** | Compat-test outcome [feedback_llm_runtime_decision.md](<ai-memory-dir>\feedback_llm_runtime_decision.md) |
| `qdrant.data_dir` | `"./qdrant_data/"` | **`"data/qdrant"`** | Existing data location |

## 1. Audio (settings.py:107-130)

| settings.py key | Value | Env var | New: config.yaml path |
|---|---|---|---|
| `SAMPLE_RATE` | 16000 | n/a | `audio.sample_rate` |
| `CHANNELS` | 1 | n/a | `audio.channels` |
| `BLOCKSIZE` | 512 | n/a | `audio.blocksize` |
| `DTYPE` | "float32" | n/a | `audio.dtype` |
| `AUDIO_DEVICE` | None | `ULTRON_AUDIO_DEVICE` | `audio.input_device` (env override stays) |
| `AUDIO_OUTPUT_DEVICE` | None | `ULTRON_AUDIO_OUTPUT_DEVICE` | `audio.output_device` (env override stays) |
| `BARGE_IN_ENABLED` | True | `ULTRON_BARGE_IN_ENABLED` | `audio.barge_in_enabled` |
| `BARGE_IN_GRACE_SECONDS` | 0.5 | `ULTRON_BARGE_IN_GRACE_SECONDS` | `audio.barge_in_grace_seconds` |
| `RING_BUFFER_SECONDS` | 0.5 | n/a | `audio.ring_buffer_seconds` |
| `VAD_THRESHOLD` | 0.5 | n/a | `vad.threshold` |
| `MIN_SPEECH_DURATION_MS` | 250 | n/a | `vad.min_speech_duration_ms` |
| `MIN_SILENCE_DURATION_MS` | 500 | n/a | `vad.min_silence_duration_ms` |
| `VAD_WINDOW_SAMPLES` | 512 | n/a | `vad.window_samples` |

## 2. Wake word (settings.py:139-146)

| settings.py key | Value | Env var | New: config.yaml path |
|---|---|---|---|
| `WAKE_WORD_NAME` | "ultron" | n/a | `wake_word.name` |
| `WAKE_WORD_MODEL_PATH` | `models/openwakeword/ultron.onnx` | n/a (path computed) | `wake_word.model_path` |
| `WAKE_WORD_FALLBACK` | "hey_jarvis" | n/a | `wake_word.fallback_model` |
| `WAKE_WORD_THRESHOLD` | 0.5 | `ULTRON_WAKE_WORD_THRESHOLD` | `wake_word.threshold` |
| `WAKE_WORD_COOLDOWN_SECONDS` | 1.5 | `ULTRON_WAKE_WORD_COOLDOWN_SECONDS` | `wake_word.cooldown_seconds` |

## 3. Whisper STT (settings.py:152-160)

| settings.py key | Value | Env var | New: config.yaml path |
|---|---|---|---|
| `WHISPER_MODEL` | "small.en" | n/a | `stt.model` |
| `WHISPER_DEVICE` | "cuda" | n/a | `stt.device` |
| `WHISPER_COMPUTE_TYPE` | "float16" | n/a | `stt.compute_type` |
| `WHISPER_BEAM_SIZE` | 5 | `ULTRON_WHISPER_BEAM_SIZE` | `stt.beam_size` |
| `WHISPER_TEMPERATURE` | 0.0 | `ULTRON_WHISPER_TEMPERATURE` | `stt.temperature` |
| `WHISPER_CONDITION_ON_PREVIOUS_TEXT` | False | `ULTRON_WHISPER_CONDITION_ON_PREVIOUS_TEXT` | `stt.condition_on_previous_text` |
| `WHISPER_VAD_FILTER` | False | n/a | `stt.vad_filter` |

## 4. LLM (settings.py:169-187)

| settings.py key | Value | Env var | New: config.yaml path |
|---|---|---|---|
| `LLM_MODEL_PATH` | `models/Qwen3.5-9B-Q4_K_M.gguf` | `ULTRON_LLM_MODEL_PATH` | `llm.model_path` (env stays for opt-in override) |
| `LLM_CONTEXT_LENGTH` | 8192 | n/a | `llm.n_ctx` |
| `LLM_GPU_LAYERS` | -1 | n/a | `llm.gpu_layers` |
| `LLM_TEMPERATURE` | 0.7 | n/a | `llm.default_temperature` |
| `LLM_TOP_P` | 0.9 | n/a | `llm.default_top_p` |
| `LLM_MAX_TOKENS` | 512 | n/a | `llm.default_max_tokens` |
| `LLM_REPEAT_PENALTY` | 1.1 | n/a | `llm.default_repeat_penalty` |
| `LLM_HISTORY_TURNS` | 6 | n/a | `llm.history_turns` |
| `LLM_FLASH_ATTN` | True | `ULTRON_LLM_FLASH_ATTN` | `llm.flash_attn` |
| `LLM_KV_CACHE_TYPE` | 8 (q8_0) | `ULTRON_LLM_KV_CACHE_TYPE` | `llm.kv_cache_type` |
| (NEW for Phase 3) | "llama_cpp" | n/a | `llm.provider` — non-Ollama; pinned per [feedback_llm_runtime_decision.md](<ai-memory-dir>\feedback_llm_runtime_decision.md) |

## 5. Memory / RAG / Qdrant (settings.py:197-229)

| settings.py key | Value | Env var | New: config.yaml path |
|---|---|---|---|
| `MEMORY_ENABLED` | True | `ULTRON_MEMORY_ENABLED` | `memory.enabled` |
| `MEMORY_JSONL_PATH` | `data/memory.jsonl` | n/a | `memory.jsonl_legacy_path` |
| `MEMORY_QDRANT_PATH` | `data/qdrant` | n/a | `qdrant.data_dir` |
| `MEMORY_QDRANT_CONVERSATIONS` | "conversations" | n/a | `qdrant.collections.conversations` |
| `MEMORY_QDRANT_FACTS` | "facts" | n/a | `qdrant.collections.facts` |
| `MEMORY_QDRANT_WEB_RESULTS` | "web_results" | n/a | `qdrant.collections.web_results` |
| `MEMORY_DENSE_MODEL` | "BAAI/bge-small-en-v1.5" | `ULTRON_MEMORY_DENSE_MODEL` | `embeddings.dense_model` |
| `MEMORY_SPARSE_MODEL` | "Qdrant/bm25" | `ULTRON_MEMORY_SPARSE_MODEL` | `embeddings.sparse_model` |
| `MEMORY_DENSE_DIM` | 384 | n/a | `embeddings.dense_dim` |
| `MEMORY_RECENT_TURNS` | 20 | `ULTRON_MEMORY_RECENT_TURNS` | `memory.recent_turns` |
| `MEMORY_RAG_TOP_K` | 5 | `ULTRON_MEMORY_RAG_TOP_K` | `memory.rag_top_k` |
| `MEMORY_RAG_EXCLUDE_RECENT` | 20 | `ULTRON_MEMORY_RAG_EXCLUDE_RECENT` | `memory.rag_exclude_recent` |
| `MEMORY_FACTS_TOP_K` | 3 | `ULTRON_MEMORY_FACTS_TOP_K` | `memory.facts_top_k` |
| `MEMORY_WRITE_QUEUE_MAXSIZE` | 256 | `ULTRON_MEMORY_WRITE_QUEUE_MAXSIZE` | `memory.write_queue_maxsize` |

## 6. Web search (settings.py:240-261)

| settings.py key | Value | Env var | New: config.yaml path |
|---|---|---|---|
| `WEB_SEARCH_ENABLED` | True | `ULTRON_WEB_SEARCH_ENABLED` | `web_search.enabled` |
| `WEB_SEARCH_BRAVE_API_KEY` | "" | `ULTRON_BRAVE_API_KEY` | **stays env-only** (`web_search.brave_api_key_env: "ULTRON_BRAVE_API_KEY"`) |
| `WEB_SEARCH_BRAVE_ENDPOINT` | https://api.search.brave.com/... | n/a | `web_search.brave.endpoint` |
| `WEB_SEARCH_BRAVE_COUNT` | 5 | `ULTRON_BRAVE_COUNT` | `web_search.brave.count` |
| `WEB_SEARCH_BRAVE_TIMEOUT_S` | 8.0 | `ULTRON_BRAVE_TIMEOUT_S` | `web_search.brave.timeout_seconds` |
| `WEB_SEARCH_BRAVE_RATE_LIMIT_S` | 2.0 | `ULTRON_BRAVE_RATE_LIMIT_S` | `web_search.brave.rate_limit_seconds` |
| `WEB_SEARCH_JINA_ENDPOINT` | https://r.jina.ai/ | n/a | `web_search.jina.endpoint` |
| `WEB_SEARCH_JINA_TIMEOUT_S` | 15.0 | `ULTRON_JINA_TIMEOUT_S` | `web_search.jina.timeout_seconds` |
| `WEB_SEARCH_JINA_MAX_FETCH` | 3 | `ULTRON_JINA_MAX_FETCH` | `web_search.jina.max_fetch` |
| `WEB_SEARCH_JINA_MAX_BYTES` | 200000 | `ULTRON_JINA_MAX_BYTES` | `web_search.jina.max_bytes` |
| `WEB_SEARCH_CACHE_TTL_VOLATILE_S` | 86400 | `ULTRON_WEB_CACHE_TTL_VOLATILE_S` | `web_search.cache.ttl_volatile_seconds` |
| `WEB_SEARCH_CACHE_TTL_STABLE_S` | 2592000 | `ULTRON_WEB_CACHE_TTL_STABLE_S` | `web_search.cache.ttl_stable_seconds` |

## 7. Coding orchestration (settings.py:272-374)

| settings.py key | Value | Env var | New: config.yaml path |
|---|---|---|---|
| `CODING_ENABLED` | True | `ULTRON_CODING_ENABLED` | `coding.enabled` |
| `CODING_BRIDGE` | "direct" | `ULTRON_CODING_BRIDGE` | `coding.bridge` |
| `CODING_MCP_ENABLED` | True | `ULTRON_CODING_MCP_ENABLED` | `coding.mcp.enabled` |
| `CODING_MCP_HOST` | "127.0.0.1" | `ULTRON_CODING_MCP_HOST` | `coding.mcp.host` |
| `CODING_MCP_PORT` | 19761 | `ULTRON_CODING_MCP_PORT` | `coding.mcp.port` |
| `CODING_MCP_SSE_PATH` | "/sse" | n/a | `coding.mcp.sse_path` |
| `CODING_MCP_LOG_PATH` | `logs/mcp_calls.jsonl` | n/a | `coding.mcp.log_path` |
| `CODING_MCP_SERVER_NAME` | "ultron_coding" | n/a | `coding.mcp.server_name` |
| `CODING_MCP_CLARIFICATION_TIMEOUT_S` | 600 | `ULTRON_CODING_MCP_CLARIFICATION_TIMEOUT_S` | `coding.mcp.clarification_timeout_seconds` |
| `CODING_TEMPLATE_DIR` | `prompts/coding` | n/a | `coding.template_dir` |
| `CODING_PROMPT_TOKEN_BUDGET` | 4000 | `ULTRON_CODING_PROMPT_TOKEN_BUDGET` | `coding.prompt_token_budget` |
| `CODING_PROMPT_CHARS_PER_TOKEN` | 4 | `ULTRON_CODING_PROMPT_CHARS_PER_TOKEN` | `coding.prompt_chars_per_token` |
| `CODING_DEFAULT_MODEL` | "haiku" | `ULTRON_CODING_DEFAULT_MODEL` | `coding.default_model` |
| `CODING_ESCALATION_MODEL` | "sonnet" | `ULTRON_CODING_ESCALATION_MODEL` | `coding.escalation_model` |
| `CODING_ESCALATION_THRESHOLD_DEFAULT` | 3 | `ULTRON_CODING_ESCALATION_THRESHOLD_DEFAULT` | `coding.escalation_threshold_default` |
| `CODING_ESCALATION_THRESHOLD_ESCALATION` | 2 | `ULTRON_CODING_ESCALATION_THRESHOLD_ESCALATION` | `coding.escalation_threshold_escalation` |
| `CODING_VERIFICATION_SMOKE_TIMEOUT_S` | 5 | `ULTRON_CODING_VERIFICATION_SMOKE_TIMEOUT_S` | `coding.verification.smoke_timeout_seconds` |
| `CODING_VERIFICATION_TEST_TIMEOUT_S` | 120 | `ULTRON_CODING_VERIFICATION_TEST_TIMEOUT_S` | `coding.verification.test_timeout_seconds` |
| `CODING_VERIFICATION_LINT_TIMEOUT_S` | 30 | `ULTRON_CODING_VERIFICATION_LINT_TIMEOUT_S` | `coding.verification.lint_timeout_seconds` |
| `CODING_SESSION_AUDIT_DIR` | `logs/sessions` | n/a | `coding.session_audit_dir` |
| `CODING_TOKEN_BUDGET_PER_SESSION` | 100000 | `ULTRON_CODING_TOKEN_BUDGET_PER_SESSION` | `coding.token_budget_per_session` |
| `CODING_TOKEN_WARNING_THRESHOLD` | 0.8 | `ULTRON_CODING_TOKEN_WARNING_THRESHOLD` | `coding.token_warning_threshold` |
| `CODING_PROGRESS_TIMEOUT_S` | 300 | `ULTRON_CODING_PROGRESS_TIMEOUT_S` | `coding.progress_timeout_seconds` |
| `CODING_TEST_SANDBOX_PATH` | `tests/coding/sandbox` | n/a | `coding.test_sandbox_path` |
| `CODING_CLAUDE_CLI` | path to claude.cmd | `ULTRON_CLAUDE_CLI` | `coding.claude_cli` |
| `CODING_CLAUDE_MODEL` | "haiku" | `ULTRON_CLAUDE_MODEL` | `coding.claude_model` |
| `CODING_SANDBOX_PATH` | `data/sandbox` | n/a | `coding.sandbox_root` |
| `CODING_PROJECT_REGISTRY_PATH` | `data/projects.json` | n/a | `coding.project_registry_path` |
| `CODING_TASK_LOG_PATH` | `logs/coding_tasks.jsonl` | n/a | `coding.audit_log_path` |
| `CODING_TASK_TIMEOUT_S` | 1800 | `ULTRON_CODING_TASK_TIMEOUT_S` | `coding.task_timeout_seconds` |
| `CODING_SKIP_PERMISSIONS` | True | `ULTRON_CODING_SKIP_PERMISSIONS` | `coding.skip_permissions` |
| (escape hatch, keep env-only) | — | `ULTRON_CODING_MCP_ALLOW_ANY_ROOT` | **stays env-only** (test-only escape; documented in mcp_server.py) |

## 8. Follow-up listening + addressing (settings.py:382-401)

| settings.py key | Value | Env var | New: config.yaml path |
|---|---|---|---|
| `FOLLOW_UP_ENABLED` | True | `ULTRON_FOLLOW_UP_ENABLED` | `addressing.follow_up_enabled` |
| `FOLLOW_UP_TIMEOUT_SECONDS` | **30.0** | `ULTRON_FOLLOW_UP_TIMEOUT_SECONDS` | `addressing.warm_mode_duration_seconds` (= 30, NOT the prompt's 10) |
| `ADDRESSEE_DEFAULT_SILENT` | True | `ULTRON_ADDRESSEE_DEFAULT_SILENT` | `addressing.default_uncertain_to_not_addressed` |
| `ADDRESSING_RULE_CONFIDENCE_THRESHOLD` | 0.8 | `ULTRON_ADDRESSING_RULE_CONFIDENCE_THRESHOLD` | `addressing.rule_confidence_threshold` |
| `ADDRESSING_ZERO_SHOT_MODEL` | "google/flan-t5-small" | `ULTRON_ADDRESSING_ZERO_SHOT_MODEL` | `addressing.zero_shot_model` |
| `ADDRESSING_LOAD_EAGERLY` | True | `ULTRON_ADDRESSING_LOAD_EAGERLY` | `addressing.load_eagerly` |
| `ADDRESSING_LOG_PATH` | `logs/addressing.jsonl` | n/a | `addressing.log_path` |

## 9. TTS (Piper) (settings.py:407-426)

| settings.py key | Value | Env var | New: config.yaml path |
|---|---|---|---|
| `TTS_VOICE_PATH` | `models/piper/en_US-ryan-medium.onnx` | n/a | `tts.piper_voice_path` |
| `TTS_VOICE_CONFIG_PATH` | `...onnx.json` | n/a | `tts.piper_voice_config_path` |
| `TTS_OUTPUT_SAMPLE_RATE` | 22050 | n/a | `tts.output_sample_rate` |
| `TTS_SENTENCE_FLUSH_CHARS` | ".!?\n" | n/a | `tts.sentence_flush_chars` |
| `TTS_LENGTH_SCALE` | 1.15 | `ULTRON_TTS_LENGTH_SCALE` | `tts.piper_length_scale` |
| `TTS_PAUSE_MS` | 180 | `ULTRON_TTS_PAUSE_MS` | `tts.pause_ms` |
| `TTS_EDGE_FADE_MS` | 4 | `ULTRON_TTS_EDGE_FADE_MS` | `tts.edge_fade_ms` |

## 10. RVC (settings.py:435-458)

| settings.py key | Value | Env var | New: config.yaml path |
|---|---|---|---|
| `RVC_ENABLED` | True | n/a | `tts.rvc.enabled` |
| `RVC_MODEL_DIR` | `ultron_james_spader_mcu_6941` | n/a | `tts.rvc.model_dir` |
| `RVC_MODEL_PATH` | `.../Ultron.pth` | n/a | `tts.rvc.model_path` |
| `RVC_INDEX_PATH` | `.../added_IVF301_..._Ultron_v2.index` | n/a | `tts.rvc.index_path` |
| `RVC_SUPPORT_DIR` | `models/rvc` | n/a | `tts.rvc.support_dir` |
| `RVC_HUBERT_PATH` | `models/rvc/hubert_base.pt` | n/a | `tts.rvc.hubert_path` |
| `RVC_RMVPE_PATH` | `models/rvc/rmvpe.pt` | n/a | `tts.rvc.rmvpe_path` |
| `RVC_DEVICE` | "cuda:0" | n/a | `tts.rvc.device` |
| `RVC_PITCH_SHIFT` | -2 | `ULTRON_RVC_PITCH_SHIFT` | `tts.rvc.pitch_shift` |
| `RVC_INDEX_RATE` | 0.66 | `ULTRON_RVC_INDEX_RATE` | `tts.rvc.index_rate` |
| `RVC_PROTECT` | 0.45 | `ULTRON_RVC_PROTECT` | `tts.rvc.protect` |
| `RVC_F0_METHOD` | "rmvpe" | n/a | `tts.rvc.f0_method` |
| `RVC_RMS_MIX_RATE` | 0.35 | `ULTRON_RVC_RMS_MIX_RATE` | `tts.rvc.rms_mix_rate` |
| `RVC_FILTER_RADIUS` | 1 | `ULTRON_RVC_FILTER_RADIUS` | `tts.rvc.filter_radius` |

## 11. System prompt (settings.py:464-484)

| settings.py key | Value | Env var | New: config.yaml path |
|---|---|---|---|
| `ULTRON_SYSTEM_PROMPT` | (8-paragraph multi-line string) | n/a | `llm.system_prompt` (multi-line YAML block) |

## 12. Logging (settings.py:490-493)

| settings.py key | Value | Env var | New: config.yaml path |
|---|---|---|---|
| `LOG_FILE` | `logs/ultron.log` | n/a | `logging.file` |
| `LOG_LEVEL` | "INFO" | `ULTRON_LOG_LEVEL` | `logging.level` |
| `LOG_FORMAT` | "%(asctime)s \| %(levelname)..." | n/a | `logging.format` |
| `LOG_DATEFMT` | "%Y-%m-%d %H:%M:%S" | n/a | `logging.datefmt` |

## 13. Paths (settings.py:23-27)

| settings.py key | Value | Env var | New: config.yaml path |
|---|---|---|---|
| `PROJECT_ROOT` | computed | n/a | derived in loader (not stored in yaml) |
| `MODELS_DIR` | `<root>/models` | n/a | derived (`paths.models_dir`) |
| `LOGS_DIR` | `<root>/logs` | n/a | derived (`paths.logs_dir`) |

## 14. Projections (currently no config; spec adds in Phase 3)

| Item | Value | Env var | New: config.yaml path |
|---|---|---|---|
| Tokenizer | "tiktoken_cl100k_base" | n/a | `projections.tokenizer` |
| `clarification_context` budget | 1500 | n/a | `projections.budgets.clarification_context` |
| `status_delta` budget | 600 | n/a | `projections.budgets.status_delta` |
| `adjustment_context` budget | 1200 | n/a | `projections.budgets.adjustment_context` |
| `correction_context` budget | 1500 | n/a | `projections.budgets.correction_context` |
| `completion_context` budget | 800 | n/a | `projections.budgets.completion_context` |
| `truncation_warning_threshold` | 0.95 | n/a | `projections.truncation_warning_threshold` |
| `log_truncations` | True | n/a | `projections.log_truncations` |

## 15. HF cache pre-init (settings.py:38-82)

This block runs BEFORE config loading because it must override stale env
vars before any HF library imports cache the values. **It stays in
`config/__init__.py` or a dedicated bootstrap module** rather than moving
into `config.yaml` — it's not a tunable, it's a startup workaround for a
specific class of env-var corruption.

Files affected:
- `HF_HOME`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_DATASETS_CACHE`,
  `TRANSFORMERS_CACHE`, `XET_CACHE_DIR`

Decision: **keep as-is, runs at module import**. Don't move to YAML. Will
relocate to `src/ultron/config_bootstrap.py` if Phase 3 ends up deleting
`config/settings.py` entirely.

## 16. Reference site distribution

Files in `src/`, `tests/`, `scripts/` that read from `settings.X` (35 total):

```
src/ultron/audio/capture.py             scripts/benchmark.py
src/ultron/audio/devices.py             scripts/download_models.py
src/ultron/audio/vad.py                 scripts/maintenance.py
src/ultron/audio/wake_word.py           scripts/measure_baseline_extended.py
src/ultron/coding/coordinator.py        scripts/migrate_memory_to_qdrant.py
src/ultron/coding/direct_bridge.py      scripts/review_addressing.py
src/ultron/coding/mcp_server.py
src/ultron/coding/narration.py
src/ultron/coding/projections.py
src/ultron/coding/projects.py
src/ultron/coding/runner.py
src/ultron/coding/templates.py
src/ultron/coding/verification.py
src/ultron/coding/voice.py
src/ultron/llm/inference.py
src/ultron/memory/embedder.py
src/ultron/memory/qdrant_store.py
src/ultron/pipeline/orchestrator.py
src/ultron/transcription/whisper_engine.py
src/ultron/tts/rvc.py
src/ultron/tts/speech.py
src/ultron/utils/logging.py
src/ultron/web_search/brave.py
src/ultron/web_search/cache.py
src/ultron/web_search/gating.py
src/ultron/web_search/jina.py
src/ultron/web_search/search.py

tests/coding/test_orchestration.py
tests/test_memory_qdrant.py
```

Total reference sites: ~280 across these 35 files.

## Migration order (Phase 3.5)

Going least-coupled to most-coupled so each migration step is testable in
isolation. Tests gating each step are listed for traceability.

1. **logging** — `utils/logging.py` only. Test: import sanity.
2. **paths** — `PROJECT_ROOT`, `MODELS_DIR`, `LOGS_DIR`. Test: `tests/test_pipeline.py` import.
3. **audio + VAD** — capture, devices, vad, ring_buffer. Test: `test_audio.py`.
4. **wake_word** — wake_word.py. Test: import + slow tests gated.
5. **stt (whisper)** — whisper_engine.py. Test: `test_transcription.py`.
6. **embeddings + memory** — embedder.py, qdrant_store.py. Test: `test_memory_qdrant.py`.
7. **llm** — inference.py. Test: `test_llm.py`.
8. **web_search** — brave, jina, gating, search, cache, acknowledgments. Test: `test_web_gating.py`.
9. **uncertainty** — uncertainty.py. Test: `test_uncertainty.py`.
10. **tts (piper)** — speech.py. Test: `test_tts.py`.
11. **tts.rvc** — rvc.py. Test: rvc-import in `test_tts.py`.
12. **addressing** — addressing/. Test: `test_addressing.py`.
13. **coding** (the big one — 14 files) — bridge, direct_bridge, runner, mcp_server, coordinator, narration, projections, projects, session, templates, verification, voice, audit. Tests: ~20 test files.
14. **projections config wiring** — wire `projections.log_truncations` and `truncation_warning_threshold` into `_finalize_projection`. Test: `test_projections.py`.

After every step: full `pytest tests/ -q` must pass.

## Net post-migration footprint

- **`config.yaml`**: single source of truth, ~250-line YAML, all values + env-var overrides documented
- **`src/ultron/config.py`**: pydantic schema + loader + singleton + typed accessors
- **`config/settings.py`**: removed entirely; the HF-cache pre-init relocates to `src/ultron/config_bootstrap.py`
- **35 source files**: each migrated to import from `ultron.config` instead of `config.settings`
- Reference shape changes from `settings.LLM_TEMPERATURE` to `get_config().llm.default_temperature`
