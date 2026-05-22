# Ultron prototype — system inventory

Snapshot of every component currently in the system, with file paths, key
classes/functions, and brief responsibility. Generated as Part 1 of the
Foundation phase to verify that what's checked-in matches the Foundation
prompt's CONTEXT description before any refactoring work begins.

- HEAD at snapshot: `4ecc7ec` ("Phase C / Phase 1: context projection refactor")
- Worktree: `C:\STC\ultronPrototype\.claude\worktrees\friendly-jang-ed9a06`
- Main checkout: `C:\STC\ultronPrototype` (where models + venv live)
- Total source LOC under `src/`: ~13,185 lines (excluding `__pycache__`)

## How to use this document

The inventory is grouped by the same sections the Foundation prompt's
CONTEXT block uses. Each row lists where the component lives, its
top-level public symbol(s), and a one-line note. Anything **PRESENT**
matches what the prompt describes. Anything **NOTE** is something
encountered in the codebase that the prompt didn't explicitly call out
but is relevant. Anything **DIFFERENT** flags a divergence that needs
discussion before any modification.

---

## 1. Voice and inference stack (pre-existing)

| Component | Location | Symbols | Status |
|---|---|---|---|
| LLM (Qwen3.5-9B Q4_K_M GGUF, llama-cpp-python) | [src/ultron/llm/inference.py](src/ultron/llm/inference.py) | `LLMEngine`, `_strip_thinking_blocks` | **PRESENT** |
| LLM model file | `C:\STC\ultronPrototype\models\Qwen3.5-9B-Q4_K_M.gguf` | n/a | **PRESENT** (5.29 GB; metadata: `qwen35` hybrid attn+SSM, EOS=`<\|im_end\|>`, Unsloth-quantized) |
| Whisper STT | [src/ultron/transcription/whisper_engine.py](src/ultron/transcription/whisper_engine.py) | `WhisperEngine` | **PRESENT** |
| Piper TTS | [src/ultron/tts/speech.py](src/ultron/tts/speech.py) | `TextToSpeech` | **PRESENT** (sentence flush on `.!?\n`, `TTS_LENGTH_SCALE` configurable) |
| RVC voice conversion | [src/ultron/tts/rvc.py](src/ultron/tts/rvc.py) | `RvcConverter` | **PRESENT** (cuda:0, RMVPE pitch, ~900MB VRAM) |
| Embeddings (bge-small-en-v1.5 INT8 ONNX, CPU) | [src/ultron/memory/embedder.py](src/ultron/memory/embedder.py) | `TextEmbedder` (in qdrant_store), embedder helpers | **PRESENT** |
| openWakeWord (CPU) | [src/ultron/audio/wake_word.py](src/ultron/audio/wake_word.py) | `WakeWordDetector` | **PRESENT** (custom `ultron.onnx` at `models/openwakeword/ultron.onnx`) |
| VAD (silero-vad) | [src/ultron/audio/vad.py](src/ultron/audio/vad.py) | `VoiceActivityDetector`, `SpeechEvent` | **PRESENT** |
| Audio capture / ring buffer | [src/ultron/audio/capture.py](src/ultron/audio/capture.py), [src/ultron/audio/ring_buffer.py](src/ultron/audio/ring_buffer.py) | `AudioCapture`, `RingBuffer` | **PRESENT** |
| Orchestrator (event loop, state machine) | [src/ultron/pipeline/orchestrator.py](src/ultron/pipeline/orchestrator.py) | `Orchestrator` | **PRESENT** (~700 lines; integrates everything) |

**NOTE on Ollama:** The user has Ollama installed and `qwen3:8b` registered, but
the voice pipeline does NOT use it. Loader is llama-cpp-python directly. See
[memory/feedback_llm_runtime_decision.md](<ai-memory-dir>\feedback_llm_runtime_decision.md)
for the runtime decision.

---

## 2. Phase A — Coding orchestration foundation

| Component | Location | Symbols | Status |
|---|---|---|---|
| Abstract bridge interface + standardized `TaskEvent` vocabulary | [src/ultron/coding/bridge.py](src/ultron/coding/bridge.py:254) | `CodingBridge`, `EventKind` (STATUS, TEXT, TOOL_USE, TOOL_RESULT, FILE_CHANGE, ERROR, COMPLETE, USAGE), `TaskHandle`, `TaskEvent`, `TaskRequest`, `TaskResult`, `TaskState` | **PRESENT** |
| Direct Claude Code subprocess bridge | [src/ultron/coding/direct_bridge.py:52](src/ultron/coding/direct_bridge.py:52) | `DirectClaudeCodeBridge` | **PRESENT** (spawns `claude --print --output-format stream-json --include-partial-messages --include-hook-events --model haiku --add-dir <cwd> --dangerously-skip-permissions`) |
| Project registry (atomic JSON CRUD) | [src/ultron/coding/projects.py:86](src/ultron/coding/projects.py:86) | `ProjectRegistry`, `Project` | **PRESENT** (registry path: [data/projects.json](data/projects.json)) |
| Project resolver (exact / alias / substring / semantic) | [src/ultron/coding/projects.py:197](src/ultron/coding/projects.py:197) | `ProjectResolver`, `ProjectResolution`, `ResolutionKind` | **PRESENT** |
| Sandbox project creation | [src/ultron/coding/projects.py:357](src/ultron/coding/projects.py:357) | `new_sandbox_project`, `slugify_for_path` | **PRESENT** (creates under `data/sandbox/`, slug collision-safe) |
| Coding task runner | [src/ultron/coding/runner.py:94](src/ultron/coding/runner.py:94) | `CodingTaskRunner`, `build_default_bridge` | **PRESENT** (one in-flight task; `progress_narration`, `completion_narration`, audit log) |
| Coding intent classifier | [src/ultron/coding/intent.py:202](src/ultron/coding/intent.py:202) | `classify`, `CodingIntent`, `CodingIntentKind` (CODE_TASK, PROGRESS_QUERY, CANCEL, NONE, MID_SESSION_ADJUSTMENT, CLARIFICATION_RESPONSE), `derive_project_name` | **PRESENT** |
| Voice-side facade | [src/ultron/coding/voice.py:53](src/ultron/coding/voice.py:53) | `CodingVoiceController`, `VoiceResponse`; methods `handle_utterance`, `pending_completion`, `pending_clarifications` | **PRESENT** |
| Orchestrator polling integration | [src/ultron/pipeline/orchestrator.py:639](src/ultron/pipeline/orchestrator.py:639) | `_announce_coding_completion_if_pending`, `_announce_pending_clarifications`, `_announce_pending_budget_warning` | **PRESENT** |
| Discipline preamble for Claude prompts | [src/ultron/coding/templates.py](src/ultron/coding/templates.py), [prompts/coding/](prompts/coding/) | `TemplateRenderer`, jinja2 prompts | **PRESENT** |

**OpenClawBridge slot-in:** referenced as a factory branch in
[src/ultron/coding/runner.py:58-68](src/ultron/coding/runner.py:58) that
raises `NotImplementedError` if `ULTRON_CODING_BRIDGE=openclaw` is set.
**No actual class file exists** (`src/ultron/coding/openclaw_bridge.py`
does not exist). Comments referencing the future bridge are scattered
through [bridge.py](src/ultron/coding/bridge.py) and
[direct_bridge.py](src/ultron/coding/direct_bridge.py). **Part 5 of the
Foundation phase removes this reservation cleanly.**

---

## 3. Phases 0-5 (LLM tuning + addressing + RAG + web + uncertainty)

### Phase 0 — Baseline measurement

| Item | Location | Status |
|---|---|---|
| Voice-path baseline data | [baselines.json](baselines.json) (top-level keys + nested `phase_foundation_start`) | **PRESENT** (~32 KB, both flat-style and nested-style data co-exist for compatibility) |
| Per-phase baseline snapshots | `baselines_phase{0..7}.json`, `baselines_phase_c{0,1}.json` at worktree root | **PRESENT** (historical record) |
| Voice-path measurement script | [scripts/measure_baseline.py](scripts/measure_baseline.py) | **PRESENT** |
| Foundation extended measurement script | [scripts/measure_baseline_extended.py](scripts/measure_baseline_extended.py) | **PRESENT** (added this phase, modes `--lite` / `--full` / `--all`) |

### Phase 1 — LLM tuning

| Item | Location | Status |
|---|---|---|
| Flash attention enabled | [config/settings.py:186](config/settings.py:186) `LLM_FLASH_ATTN=True` | **PRESENT** |
| q8_0 KV cache quantization | [config/settings.py:187](config/settings.py:187) `LLM_KV_CACHE_TYPE=8` | **PRESENT** |

### Phase 2 — Addressing

| Item | Location | Symbols | Status |
|---|---|---|---|
| Wake word "Ultron" via openWakeWord | [src/ultron/audio/wake_word.py](src/ultron/audio/wake_word.py) | `WakeWordDetector` | **PRESENT** |
| COLD/WARM mode state machine | [src/ultron/pipeline/orchestrator.py](src/ultron/pipeline/orchestrator.py) follow-up listening logic | n/a (in orchestrator) | **PRESENT** |
| Follow-up window | [config/settings.py:383](config/settings.py:383) `FOLLOW_UP_TIMEOUT_SECONDS=30.0` | **PRESENT — DEVIATES from spec's 10s, intentional per [memory/feedback_ultron_extension.md](<ai-memory-dir>\feedback_ultron_extension.md)** |
| Hybrid rule-based + zero-shot addressing classifier | [src/ultron/addressing/](src/ultron/addressing/) — [classifier.py:48](src/ultron/addressing/classifier.py:48), [rules.py:156](src/ultron/addressing/rules.py:156), [zero_shot.py](src/ultron/addressing/zero_shot.py) | `AddressingClassifier`, `AddressingDecision`, `classify` | **PRESENT** (CPU-only; flan-t5-small for ambiguous cases) |
| Addressing audit log | `logs/addressing.jsonl` (created at runtime by orchestrator) | n/a | **PRESENT** |
| Review script | [scripts/review_addressing.py](scripts/review_addressing.py) | n/a | **PRESENT** |

### Phase 3 — RAG / Qdrant memory

| Item | Location | Symbols | Status |
|---|---|---|---|
| Qdrant embedded mode | data dir at `./qdrant_data/` (configured in [config/settings.py:204](config/settings.py:204) as `data/qdrant`) | n/a | **PRESENT — DIFFERENT path from prompt's `./qdrant_data/`; current is `data/qdrant`. Same effect, different filename. Flag for Phase 3 unified-config alignment.** |
| Three collections (conversations, facts, web_results) | [config/settings.py:205-207](config/settings.py:205) | `MEMORY_QDRANT_CONVERSATIONS`, `MEMORY_QDRANT_FACTS`, `MEMORY_QDRANT_WEB_RESULTS` | **PRESENT** |
| Hybrid search (BM25 + dense bge-small) with RRF | [src/ultron/memory/qdrant_store.py:64](src/ultron/memory/qdrant_store.py:64) | `ConversationMemory` | **PRESENT** |
| Async write path | inside `ConversationMemory` (background writer thread) | n/a | **PRESENT** |
| Maintenance script | [scripts/maintenance.py](scripts/maintenance.py) | n/a | **PRESENT** |
| Maintenance state | `data/maintenance.sqlite` | n/a | **PRESENT** |
| JSONL migration source | `data/memory.jsonl` | n/a | **PRESENT** (Phase 3 ingest source per [memory/feedback_ultron_extension.md](<ai-memory-dir>\feedback_ultron_extension.md)) |
| Migration script | [scripts/migrate_memory_to_qdrant.py](scripts/migrate_memory_to_qdrant.py) | n/a | **PRESENT** |

### Phase 4 — Web search

| Item | Location | Symbols | Status |
|---|---|---|---|
| Brave API client (rate-limited) | [src/ultron/web_search/brave.py](src/ultron/web_search/brave.py) | `BraveSearchClient`, `BraveResult` | **PRESENT** (key from `ULTRON_BRAVE_API_KEY` env) |
| Two-stage gating (rules + pre-flight) | [src/ultron/web_search/gating.py:403](src/ultron/web_search/gating.py:403) | `WebSearchGate`, `GateVerdict`, `GateDecision`, `classify_by_rules`, `classify_by_preflight` | **PRESENT** |
| Jina Reader full-text | [src/ultron/web_search/jina.py](src/ultron/web_search/jina.py) | `JinaReaderClient` | **PRESENT** |
| Acknowledgment phrase pool | [src/ultron/web_search/acknowledgments.py:31](src/ultron/web_search/acknowledgments.py:31) | `AcknowledgmentSource` | **PRESENT** (8-phrase shuffled pool) |
| Citation formatting / source rendering | [src/ultron/web_search/search.py](src/ultron/web_search/search.py) | `WebSearchExecutor`, `format_sources_for_prompt`, `format_sources_for_transcript` | **PRESENT** |
| `web_results` cache | [src/ultron/web_search/cache.py:58](src/ultron/web_search/cache.py:58) | `WebResultsCache` | **PRESENT** (writes through Qdrant collection) |

### Phase 5 — Uncertainty signals

| Item | Location | Symbols | Status |
|---|---|---|---|
| Uncertainty detection in pre-flight pass | [src/ultron/web_search/gating.py:278](src/ultron/web_search/gating.py:278) | `classify_by_preflight` returns `knowledge_confidence` / `knowledge_source` / `has_temporal_dependency` on `GateVerdict` | **PRESENT** |
| Response style adaptation by confidence | [src/ultron/uncertainty.py:54](src/ultron/uncertainty.py:54) | `apply(verdict, user_text)` | **PRESENT** (single function, not a class — adapts user prompt with hedging/citation hints) |

---

## 4. Coding Orchestration Addendum

| Component | Location | Symbols | Status |
|---|---|---|---|
| MCP server (bidirectional tool surface) | [src/ultron/coding/mcp_server.py:137](src/ultron/coding/mcp_server.py:137) | `UltronMCPServer`, `write_mcp_config`, `remove_mcp_config` | **PRESENT** (~800 lines; in-process Python tools + SSE worker tools) |
| Conversation coordinator (clarification decision logic) | [src/ultron/coding/coordinator.py:280](src/ultron/coding/coordinator.py:280) | `ConversationCoordinator`; `decide_clarification`, `handle_declare_complete` | **PRESENT** (~1000 lines) |
| ProjectSession state model | [src/ultron/coding/session.py:164](src/ultron/coding/session.py:164) | `ProjectSession`, `SessionStatus`, `SessionStore`, `StateTransitionError`, `is_valid_transition`, `StageRecord`, `FileRecord`, `ClarificationRequest`, `AdjustmentRecord`, `CompletionClaim`, `TestStatus` | **PRESENT** |
| Five prompt templates | [prompts/coding/](prompts/coding/) — `claude_code_initial_new.j2`, `_initial_edit.j2`, `_correction.j2`, `_adjustment.j2`, `_clarification_response.j2` | rendered via `TemplateRenderer` | **PRESENT** (5 jinja files) |
| Verification layer (six checks) | [src/ultron/coding/verification.py:128](src/ultron/coding/verification.py:128) | `Verifier` | **PRESENT** (~790 lines; checks for syntax, smoke run, file claims, etc.) |
| Corrective loop with Haiku→Sonnet escalation | inside `ConversationCoordinator.handle_declare_complete` + `Verifier` results path; thresholds at [config/settings.py:311-318](config/settings.py:311) (`CODING_ESCALATION_THRESHOLD_DEFAULT=3`, `_ESCALATION=2`) | n/a | **PRESENT** |
| Delta-aware status narration | [src/ultron/coding/narration.py:107](src/ultron/coding/narration.py:107) | `StatusNarrator`, `NarrationDelta` | **PRESENT** |
| Integration test harness (10 scenarios + runner) | [tests/coding/test_orchestration.py](tests/coding/test_orchestration.py), [scripts/run_orchestration_tests.py](scripts/run_orchestration_tests.py) | `_StubLLM`, `OrchStack`, scenarios 1-10 + 7b | **PRESENT** (11 scenarios in 10.7 s wall — see `phase_foundation_start.measurements_extended.coding_orchestration_scenarios`) |
| Live e2e variant | [tests/coding/test_orchestration_real.py](tests/coding/test_orchestration_real.py) | n/a | **PRESENT** (gated on `PYTEST_RUN_GPU_TESTS=1` — metered against Claude API) |
| Mock bridge | [tests/coding/mock_bridge.py](tests/coding/mock_bridge.py) | `ScriptedClaudeBridge`, `ClaudeScript` | **PRESENT** |
| Audit logs | [logs/coding_tasks.jsonl](logs/coding_tasks.jsonl), [logs/verifications.jsonl](logs/verifications.jsonl), [logs/clarifications.jsonl](logs/clarifications.jsonl), [logs/mcp_calls.jsonl](logs/mcp_calls.jsonl), per-session under `logs/sessions/<id>.jsonl` | n/a | **PRESENT** |

---

## 5. Phase C / Phase 1 — Context projections (HEAD)

| Component | Location | Symbols | Status |
|---|---|---|---|
| Five projection functions | [src/ultron/coding/projections.py](src/ultron/coding/projections.py) | `project_clarification_context:158`, `project_status_delta:324`, `project_adjustment_context:483`, `project_correction_context:629`, `project_completion_context:765` | **PRESENT** (904 lines; `tiktoken` cl100k_base for token counting) |
| Projection schemas | same file | `ClarificationContextProjection`, `StatusDeltaProjection`, `AdjustmentContextProjection`, `CorrectionContextProjection`, `CompletionContextProjection`, `ProjectionResult` | **PRESENT** |
| Five new MCP tools | [src/ultron/coding/mcp_server.py](src/ultron/coding/mcp_server.py) | `get_status_delta`, `get_clarification_context`, `get_adjustment_context`, `get_correction_context`, `get_completion_context` (plus deprecated `get_session_state` and internal `get_full_state`) | **PRESENT** |
| Projection tests | [tests/test_projections.py](tests/test_projections.py) | 24 tests | **PRESENT, all passing** |

**This is the "Part 2" target of the Foundation prompt — verification only,
no new implementation needed.**

---

## 6. Configuration

| Item | Location | Status |
|---|---|---|
| Centralized settings (current scattered approach) | [config/settings.py](config/settings.py) | **PRESENT** (~494 lines; Phase 3 of Foundation will consolidate to `config.yaml`) |
| Project-level pyproject.toml | [pyproject.toml](pyproject.toml) | **PRESENT** |
| .env loading | via `python-dotenv` `load_dotenv()` in `config/settings.py:17` | **PRESENT** |
| Environment template | `.env.example` | **PRESENT** at worktree root |

---

## 7. Tests

| Suite | Location | Count | Status |
|---|---|---|---|
| Unit/integration (default) | [tests/](tests/) | 401 collected, 385 passed, 16 skipped (env-gated), 0 failed | **PRESENT, GREEN** |
| Slow tests (env-gated `PYTEST_RUN_GPU_TESTS=1`) | various, marked `@pytest.mark.skipif` or `@pytest.mark.slow` | 15 tests | **PRESENT** (not run in baseline pass; metered Claude API or GPU model loads) |
| Orchestration scenarios (mocked) | [tests/coding/test_orchestration.py](tests/coding/test_orchestration.py) | 11 (10 spec + 7b delta-tracking) | **PRESENT, ALL PASSING** |
| Orchestration scenarios (real Claude) | [tests/coding/test_orchestration_real.py](tests/coding/test_orchestration_real.py) | matching set | **PRESENT, env-gated** |

---

## 8. Scripts

| Script | Purpose |
|---|---|
| [scripts/download_models.py](scripts/download_models.py) | First-run model fetcher (Qwen GGUF, Piper voices, Whisper, openWakeWord) |
| [scripts/list_audio_devices.py](scripts/list_audio_devices.py) | Mic/output device introspection |
| [scripts/benchmark.py](scripts/benchmark.py) | First-token-latency benchmark |
| [scripts/measure_baseline.py](scripts/measure_baseline.py) | Voice-path VRAM + TTFT baseline (writes top-level of `baselines.json`) |
| [scripts/measure_baseline_extended.py](scripts/measure_baseline_extended.py) | **NEW (Foundation Phase 0)** — search VRAM, coding-session VRAM, scenario timing, TTA, composite TTFA. Modes `--lite` / `--full` / `--all`. |
| [scripts/migrate_memory_to_qdrant.py](scripts/migrate_memory_to_qdrant.py) | One-shot migration of `data/memory.jsonl` → Qdrant collections |
| [scripts/maintenance.py](scripts/maintenance.py) | Periodic Qdrant maintenance (summarization, fact extraction, cluster labeling) |
| [scripts/review_addressing.py](scripts/review_addressing.py) | Read `logs/addressing.jsonl` and report classifier verdicts |
| [scripts/run_orchestration_tests.py](scripts/run_orchestration_tests.py) | Run the 10 orchestration scenarios with reporting |

---

## 9. On-disk state

| Path | Purpose |
|---|---|
| `models/` (main checkout only) | LLM, Whisper, Piper, openWakeWord, RVC support files |
| `ultron_james_spader_mcu_6941/` | RVC voice model (`Ultron.pth`, `added_IVF301_…_Ultron_v2.index`) |
| `data/qdrant/` | Embedded Qdrant data |
| `data/memory.jsonl` | Legacy turn log; migration source |
| `data/projects.json` | Project registry |
| `data/sandbox/` | Created on demand for new sandbox projects |
| `data/maintenance.sqlite` | Maintenance script state |
| `data/summaries.jsonl` | Conversation summaries |
| `data/ollama_compat_test/` | **NEW (Foundation Phase 0)** — Ollama Modelfile from compat test |
| `logs/coding_tasks.jsonl` | Coding task progress |
| `logs/verifications.jsonl` | Verifier run records |
| `logs/clarifications.jsonl` | Clarification decisions |
| `logs/mcp_calls.jsonl` | MCP tool calls |
| `logs/sessions/<id>.jsonl` | Per-session audit (Phase 7) |
| `logs/addressing.jsonl` | Addressing classifier audit |
| `logs/errors.jsonl` | **PROMPT EXPECTS but NOT YET PRESENT** — Phase 4 of Foundation creates this |

---

## 10. Discrepancies and notes

These are not blockers but are worth surfacing before Part 2 begins.

### A. Qdrant data path

Foundation prompt's example `config.yaml` uses `data_dir: "./qdrant_data/"`.
Current code uses `data/qdrant/`. Functionally identical; flag for Phase 3
unified-config alignment so the example matches reality (or migrate the
data dir, but the data is large and stable — the docs change is cheaper).

### B. WARM mode duration

Foundation prompt's example config has
`addressing.warm_mode_duration_seconds: 10`. Current
`FOLLOW_UP_TIMEOUT_SECONDS=30.0`. **Intentional 30 s deviation per
[memory/feedback_ultron_extension.md](<ai-memory-dir>\feedback_ultron_extension.md)**;
do NOT re-tighten without asking. Phase 3 unified-config should record 30 s
as the canonical value, not 10.

### C. LLM provider

Foundation prompt's example config uses `provider: "ollama"`. **Current
code (and the runtime decision recorded in
[memory/feedback_llm_runtime_decision.md](<ai-memory-dir>\feedback_llm_runtime_decision.md))
keeps llama-cpp-python in-process.** Phase 3 should use
`llm.provider: "llama_cpp"`, not `ollama`.

### D. `OpenClawBridge` slot-in

Not a class file. Just the factory branch in
[src/ultron/coding/runner.py:58-68](src/ultron/coding/runner.py:58)
plus comment references. **Part 5 of Foundation removes this branch
cleanly** — the import the branch attempts (`ultron.coding.openclaw_bridge`)
already raises `NotImplementedError` because the module doesn't exist, so
removal is purely cosmetic.

### E. `tests/integration/` and `tests/e2e/` directories

Foundation prompt's Phase 0.2 references
`PYTEST_RUN_GPU_TESTS=1 pytest tests/integration -q` and
`PYTEST_RUN_E2E_TESTS=1 pytest tests/e2e -q`. **Neither directory exists.**
The current repo uses `@pytest.mark.skipif(...)` env-gating per file inside
the flat `tests/` tree (15 slow/GPU-gated tests). Either:
- Phase 6 of Foundation can create `tests/integration/` and migrate slow
  tests there (consistent with the prompt's expectations), OR
- Phase 6 can keep the flat-with-markers convention and document the
  command equivalents in `docs/development.md`.

Decision deferred to Phase 6 design discussion.

### F. Existing top-level `baselines_phase{0..7}.json` and `baselines_phase_c{0,1}.json`

Historical per-phase snapshots. Foundation phase uses a nested
`phase_foundation_start` key inside `baselines.json` instead of
creating new per-phase files. Not a conflict; flag this as a convention
shift for future phases to choose explicitly.

### G. `logs/errors.jsonl`

Not present. Phase 4 of Foundation creates this. Mentioned here so the
inventory is honest.

### H. `_strip_thinking_blocks` in [src/ultron/llm/inference.py:34](src/ultron/llm/inference.py:34)

Filters `<think>...</think>` reasoning blocks from streamed tokens before
they hit TTS. Not a Foundation-prompt-specified component but **load-bearing**
for the qwen35 hybrid model's streaming behavior — relevant context for
anyone touching the LLM layer.

---

## 11. Ollama compat test artifacts (Foundation Phase 0)

For completeness — these were generated during the LLM-runtime decision:

| Path | Purpose |
|---|---|
| [data/ollama_compat_test/Modelfile](data/ollama_compat_test/Modelfile) | Modelfile that mirrored every llama-cpp param from settings.py |
| `~/.ollama/models/blobs/...` (~5.7 GB on disk) | `ultron-cpp-mirror:latest` registered model, can be removed via `ollama rm ultron-cpp-mirror` if not wanted |
| Test results | `phase_foundation_start.scope.ollama_compat_test` block in [baselines.json](baselines.json) |
| Decision recorded | [memory/feedback_llm_runtime_decision.md](<ai-memory-dir>\feedback_llm_runtime_decision.md) |

---

## 12. Verification summary

- All Phase A coding-foundation components present.
- All Phases 0-5 (LLM tuning, addressing, RAG, web, uncertainty) present.
- All Coding Orchestration Addendum components present.
- All Phase C / Phase 1 (projection refactor) components present.
- Test suite green (385 passed, 16 env-gated skipped, 0 failed).
- VRAM and latency baselines captured (`phase_foundation_start.*`).
- Discrepancies (A-H above) are non-blocking; flagged for the relevant
  later Parts of the Foundation phase.

**Conclusion:** the system as checked in matches the Foundation prompt's
CONTEXT description. No silent additions, no missing components, no
broken state. Ready to begin Part 2 (context projection refactor —
which is largely a verification step since the work has already landed
at HEAD `4ecc7ec`).
