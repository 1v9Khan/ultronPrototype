# Ultron prototype — codebase structure (single-source reference)

> **Purpose:** complete map of the system's source files, scripts,
> tests, and runtime artifacts, with public APIs and information flow
> per module. A fresh Claude Code session should read this document
> together with the memory files (`MEMORY.md`,
> `project_ultron_foundation.md`, `feedback_*.md`) to get fully
> oriented without re-exploring the codebase.
>
> **Maintenance contract:** this file is the operating manual. Keep it
> current — see "Maintenance contract" at the bottom.

Last validated against HEAD: Foundation Phase 7 close + Phase 4 deferred
wrappers + OpenClaw integration Phase 0 + 1 (llama-cpp-server launcher
+ supervisor, persona migration, PersonaLoader with mode-based
composition + hot reload, LLMEngine HTTP-client opt-in, OpenClaw bridge
foundations) + 4B optimization plan Stage A (LLMConfig preset +
draft_model_path with model_validator) + Stage B (4B + 0.8B GGUFs
downloaded, structurally validated, SHA256 recorded) + Stage C
(start_llamacpp_server.py: --model-draft, --draft-num-pred-tokens,
--from-config flags + testable CLI helpers) — 762 passing tests, 16
skipped, 0 failed.

---

## Table of contents

1. [Quick orientation](#quick-orientation)
2. [File tree](#file-tree)
3. [Cross-cutting flows](#cross-cutting-flows)
4. [Source modules](#source-modules)
5. [Configuration](#configuration)
6. [Operational scripts](#operational-scripts)
7. [Tests](#tests)
8. [Runtime artifacts (logs / data)](#runtime-artifacts)
9. [Documentation index](#documentation-index)
10. [Maintenance contract](#maintenance-contract)

---

## Quick orientation

Ultron is a local voice-first AI assistant. The pipeline is:

```
mic → wake word ("ultron") OR addressing classifier (WARM mode)
    → VAD-bounded utterance capture
    → Whisper STT
    → classify_routing() ── coding ── CodingTaskRunner (Claude Code subprocess)
                         ├ conversational ── LLM (Qwen3.5-9B Q4 via llama-cpp-python)
                         │                   ├─ optional pre-flight web-search gate
                         │                   │  ├─ Brave + Jina (real)
                         │                   │  └─ acknowledgment phrase to TTS in <200 ms
                         │                   └─ stream tokens to Piper TTS → RVC → audio
                         ├ openclaw stub ── voice "gateway not connected yet"
                         └ hybrid stub ── voice "would split it up..."
    → async write turn to Qdrant (memory)
    → enter WARM mode (30 s follow-up window)
```

For the architectural picture see [docs/architecture.md](architecture.md).
For the current decisions and Foundation phase status see
[memory/project_ultron_foundation.md](C:\Users\alecf\.claude\projects\C--STC-ultronPrototype\memory\project_ultron_foundation.md).

---

## File tree

```
<project-root>/                    ← C:\STC\ultronPrototype (main checkout)
                                       worktrees: .claude/worktrees/<branch>/
├── README.md                       ← project entry point, doc index
├── config.yaml                     ← canonical configuration (Phase 3 source of truth)
├── pyproject.toml                  ← packaging + pytest config
├── .env (gitignored)               ← secrets + opt-in env-var overrides
├── .env.example
├── baselines.json                  ← VRAM + latency baselines (nested under phase_foundation_start)
├── baselines_phase{0..7}.json      ← per-phase historical snapshots
├── baselines_phase_c{0,1}.json     ← Phase C snapshots (pre-Foundation)
│
├── src/
│   └── ultron/
│       ├── __init__.py             ← CUDA DLL discovery (Windows-specific path injection)
│       ├── __main__.py             ← `python -m ultron` entry point → constructs Orchestrator
│       ├── config.py               ← Phase 3 pydantic loader, get_config() singleton
│       ├── errors.py               ← Phase 4 typed exception hierarchy
│       ├── uncertainty.py          ← Phase 5 (original prompts) uncertainty-signal application
│       │
│       ├── audio/                  ← Audio capture, VAD, wake-word
│       │   ├── capture.py          ← AudioCapture (sounddevice callback thread)
│       │   ├── devices.py          ← Device-resolution helpers (resolve_device, describe_device)
│       │   ├── ring_buffer.py      ← Pre-speech audio buffer
│       │   ├── vad.py              ← Silero-VAD wrapper
│       │   └── wake_word.py        ← openWakeWord (custom ultron.onnx + hey_jarvis fallback)
│       │
│       ├── addressing/             ← Phase 2 addressing classifier (CPU)
│       │   ├── classifier.py       ← AddressingClassifier (rule + zero-shot dispatcher)
│       │   ├── rules.py            ← Pure-rule classify(); regex patterns
│       │   └── zero_shot.py        ← Flan-T5-small wrapper for ambiguous cases
│       │
│       ├── transcription/          ← STT
│       │   └── whisper_engine.py   ← WhisperEngine (faster-whisper, CUDA fp16)
│       │
│       ├── llm/
│       │   └── inference.py        ← LLMEngine (llama-cpp-python; Qwen3.5-9B Q4_K_M)
│       │
│       ├── memory/                 ← Phase 3 (original) Qdrant memory
│       │   ├── embedder.py         ← HybridEmbedder (FastEmbed dense + BM25 sparse)
│       │   └── qdrant_store.py     ← ConversationMemory (3 collections, async writer thread)
│       │
│       ├── web_search/             ← Phase 4 (original) Brave + Jina
│       │   ├── acknowledgments.py  ← AcknowledgmentSource (shuffled phrase pool)
│       │   ├── brave.py            ← BraveSearchClient + circuit breaker (Phase 4 Foundation)
│       │   ├── cache.py            ← WebResultsCache (Qdrant-backed)
│       │   ├── gating.py           ← Two-stage gate (rules + LLM pre-flight)
│       │   ├── jina.py             ← JinaReaderClient + circuit breaker
│       │   └── search.py           ← WebSearchExecutor (orchestrates Brave + Jina + ranking)
│       │
│       ├── tts/                    ← Piper + RVC
│       │   ├── rvc.py              ← RvcConverter (Piper PCM → Ultron timbre)
│       │   └── speech.py           ← TextToSpeech (sentence-streaming Piper + optional RVC)
│       │
│       ├── coding/                 ← Phase A coding orchestration + Coding Addendum
│       │   ├── audit.py            ← SessionAuditWriter (per-session JSONL)
│       │   ├── bridge.py           ← Abstract CodingBridge + TaskEvent vocabulary
│       │   ├── coordinator.py      ← ConversationCoordinator (clarification + correction loops)
│       │   ├── direct_bridge.py    ← DirectClaudeCodeBridge (claude --print --stream-json)
│       │   ├── intent.py           ← Coding-pipeline intent classifier (CODE_TASK etc.)
│       │   ├── mcp_server.py       ← UltronMCPServer (in-process tools + SSE worker tools)
│       │   ├── narration.py        ← StatusNarrator (delta-aware progress narration)
│       │   ├── projections.py      ← Phase C / Foundation Part 2: 5 bounded projections
│       │   ├── projects.py         ← ProjectRegistry, ProjectResolver, new_sandbox_project
│       │   ├── runner.py           ← CodingTaskRunner (one in-flight task; bridge owner)
│       │   ├── session.py          ← ProjectSession state model + SessionStore
│       │   ├── templates.py        ← TemplateRenderer (Jinja2 prompts + budget enforcement)
│       │   ├── verification.py     ← Verifier (six checks + corrective loop)
│       │   └── voice.py            ← CapabilityVoiceController (Phase 5 rename; alias preserved)
│       │
│       ├── pipeline/
│       │   └── orchestrator.py     ← Main event loop / state machine
│       │
│       ├── openclaw_routing/       ← Phase 5 capability-routing layer
│       │   ├── classifier.py       ← classify_routing() - top-level intent classifier
│       │   ├── decision_log.py     ← RoutingDecisionLog (logs/routing_decisions.jsonl)
│       │   ├── decomposer.py       ← HybridTaskDecomposer (Qwen-driven JSON output)
│       │   ├── disambiguator.py    ← IntentDisambiguator (CODING/AUTOMATION/HYBRID/UNCLEAR)
│       │   ├── dispatcher.py       ← OpenClawDispatcher (5 stub methods)
│       │   ├── intents.py          ← RoutingIntentKind enum, RoutingIntent + per-category dataclasses
│       │   └── runner.py           ← AutomationTaskRunner (mirror of CodingTaskRunner)
│       │
│       ├── openclaw_bridge/        ← OpenClaw integration Phase 1 + 3 foundations
│       │   ├── persona.py          ← PersonaLoader (mode-based: user_facing/background/heartbeat/bootstrap) + hot reload
│       │   └── lifecycle.py        ← OpenClawLifecycle (health probes; never raises)
│       │
│       ├── resilience/             ← Phase 4 resilience primitives
│       │   ├── circuit_breaker.py  ← CircuitBreaker (3-state: CLOSED/OPEN/HALF_OPEN)
│       │   ├── error_log.py        ← ErrorLog (logs/errors.jsonl writer + singleton)
│       │   └── phrases.py          ← phrase_for() (shuffled phrase pool per failure mode)
│       │
│       └── utils/
│           ├── fairseq_compat.py   ← Workarounds for fairseq dataclass + torch.load issues
│           └── logging.py          ← configure_logging(), get_logger() (rotating file + console)
│
├── config/
│   ├── __init__.py                 ← (empty)
│   └── settings.py                 ← Phase 3 SHIM: re-exports legacy settings.X from config.yaml
│
├── prompts/
│   └── coding/                     ← Jinja2 templates rendered by TemplateRenderer
│       ├── claude_code_initial_new.j2
│       ├── claude_code_initial_edit.j2
│       ├── claude_code_correction.j2
│       ├── claude_code_adjustment.j2
│       └── claude_code_clarification_response.j2
│
├── docs/
│   ├── architecture.md             ← Pipeline + state machine + subsystem table
│   ├── configuration.md            ← Per-key config reference
│   ├── config_discovery.md         ← One-time Phase 3 discovery catalog
│   ├── operations.md               ← Day-to-day running, monitoring, recovery
│   ├── development.md              ← Test layout, debugging, how-to recipes
│   ├── error_handling.md           ← Phase 4 error catalog + circuit breaker reference
│   ├── routing.md                  ← Phase 5 capability routing
│   ├── system_inventory.md         ← Phase 1 verification snapshot
│   ├── phase3_5_followup.md        ← Punch list: remaining unified-config migrations
│   ├── smoke_test.md               ← 16-step real-stack walkthrough procedure
│   ├── openclaw_integration.md     ← OpenClaw integration architecture + Phase 0/1
│   ├── openclaw_runtime.md         ← OpenClaw runtime ops (agents, supervisor, locks)
│   ├── phase_1_summary.md          ← Phase 1 close-out report (persona migration)
│   ├── 4b_optimization_plan.md     ← Deferred 4B-model migration plan
│   └── codebase_structure.md       ← THIS FILE
│
├── scripts/                        ← Operational scripts (CLI tools)
│   ├── benchmark.py                ← Latency benchmark (existing from earlier phases)
│   ├── check_vram.py               ← Quick VRAM snapshot vs cap
│   ├── download_models.py          ← First-run model fetcher
│   ├── dump_session.py             ← Render coding-session audit log readable
│   ├── list_audio_devices.py       ← Mic/output device introspection
│   ├── maintenance.py              ← Periodic Qdrant maintenance (summarization, fact extraction)
│   ├── measure_baseline.py         ← Voice-path VRAM + TTFT baseline
│   ├── measure_baseline_extended.py ← Extended baseline (search/coding VRAM, scenario timing)
│   ├── migrate_memory_to_qdrant.py ← One-shot JSONL → Qdrant migration
│   ├── review_addressing.py        ← Read addressing.jsonl, print verdicts
│   ├── run_integration_tests.py    ← pytest wrapper for tests/integration|routing|error_recovery
│   ├── run_orchestration_tests.py  ← Run 10 orchestration scenarios with reporting
│   ├── validate_config.py          ← Schema-validate config.yaml without starting Ultron
│   ├── start_llamacpp_server.py    ← OpenClaw Phase 0: launch llama-cpp-server with voice-pipeline params
│   ├── supervised_llamacpp_server.py ← OpenClaw Phase 0: supervisor wrapper with auto-restart
│   ├── smoke_test_llamacpp.ps1     ← OpenClaw Phase 0: PowerShell health probe for llama-cpp-server
│   ├── _bench_llm_http.py          ← OpenClaw Phase 0: HTTP-mode TTFT benchmark
│   └── _log_proxy.py               ← OpenClaw Phase 0: tee proxy for debugging Gateway → server traffic
│
├── tests/
│   ├── conftest.py                 ← Path setup so `from ultron.*` works
│   ├── test_*.py                   ← ~25 unit/integration test files (default suite)
│   ├── coding/
│   │   ├── conftest.py
│   │   ├── mock_bridge.py          ← ScriptedClaudeBridge (in-process mock, ClaudeScript DSL)
│   │   ├── test_orchestration.py   ← 11 mock-bridge orchestration scenarios
│   │   ├── test_orchestration_real.py ← Same scenarios with real Claude (PYTEST_RUN_GPU_TESTS=1)
│   │   ├── test_mock_bridge_smoke.py
│   │   └── sandbox/                ← test fixture sandbox
│   ├── error_recovery/             ← Phase 4: per-dependency failure modes (78 tests)
│   │   ├── conftest.py
│   │   ├── test_brave_failures.py
│   │   ├── test_jina_failures.py
│   │   ├── test_qdrant_failures.py
│   │   ├── test_audio_failures.py
│   │   ├── test_addressing_failures.py
│   │   ├── test_config_failures.py
│   │   ├── test_circuit_breaker.py
│   │   ├── test_error_log.py
│   │   ├── test_claude_code_failures.py    ← Phase 4 deferred wrappers
│   │   ├── test_mcp_server_failures.py     ← Phase 4 deferred wrappers
│   │   └── test_filesystem_failures.py     ← Phase 4 deferred wrappers
│   ├── routing/                    ← Phase 5: classifier + dispatcher + decomposer (148 tests)
│   │   ├── conftest.py
│   │   ├── test_classifier.py
│   │   ├── test_dispatcher.py
│   │   ├── test_decomposer.py
│   │   ├── test_disambiguator.py
│   │   ├── test_decision_log.py
│   │   └── test_backward_compat.py
│   └── integration/                ← Phase 6: end-to-end pipeline (83 tests)
│       ├── conftest.py
│       ├── mocks.md                ← What's mocked vs real, per layer
│       ├── performance.json        ← Phase 6 perf snapshot
│       ├── test_routing_dispatch.py
│       ├── test_conversational_pipeline.py
│       ├── test_search_pipeline.py
│       ├── test_coding_pipeline.py
│       ├── test_addressing_pipeline.py
│       └── test_error_recovery_pipeline.py
│
├── data/                           ← runtime data (gitignored except for stub structure)
│   ├── qdrant/                     ← embedded Qdrant store
│   ├── memory.jsonl                ← legacy turn log / migration source
│   ├── projects.json               ← coding project registry
│   ├── sandbox/                    ← auto-created coding projects
│   ├── summaries.jsonl             ← maintenance summaries
│   ├── maintenance.sqlite          ← maintenance state
│   └── ollama_compat_test/         ← Modelfile from Foundation-phase Ollama compat test
│
├── logs/                           ← runtime logs (gitignored)
│   ├── ultron.log                  ← rotating main log
│   ├── addressing.jsonl            ← classifier audit
│   ├── coding_tasks.jsonl          ← coding task progress
│   ├── verifications.jsonl         ← verifier runs
│   ├── clarifications.jsonl        ← clarification decisions
│   ├── mcp_calls.jsonl             ← MCP tool calls
│   ├── sessions/<id>.jsonl         ← per-session coding audit
│   ├── errors.jsonl                ← Phase 4 typed errors
│   ├── routing_decisions.jsonl     ← Phase 5 routing audit
│   └── automation_tasks.jsonl      ← Phase 5 OpenClaw task records
│
├── models/                         ← (main checkout only — NOT in worktrees)
│   ├── Qwen3.5-9B-Q4_K_M.gguf      ← LLM (5.29 GB)
│   ├── openwakeword/ultron.onnx    ← custom wake word
│   ├── piper/en_US-ryan-medium.onnx ← TTS voice
│   └── rvc/{hubert_base.pt, rmvpe.pt} ← RVC support files
│
├── ultron_james_spader_mcu_6941/   ← (main checkout only) RVC voice model
│   ├── Ultron.pth
│   └── added_IVF301_Flat_nprobe_1_Ultron_v2.index
│
└── training/                       ← (gitignored except scripts) Wake-word training data
    ├── download_training_data.py
    ├── probe_datasets.py
    ├── run_training.py
    ├── smoketest_memory.py
    └── smoketest_orchestrator.py
```

---

## Cross-cutting flows

### Voice query (conversational) — happy path

```
1. AudioCapture callback → enqueues 32 ms blocks
2. Orchestrator.run() loop:
   a. WakeWordDetector or AddressingClassifier consumes blocks
      ├── COLD: "ultron" wake word required
      └── WARM: classifier verdict required
   b. On addressed: VoiceActivityDetector marks utterance start/end
   c. AudioCapture._capture_utterance() yields ndarray
3. WhisperEngine.transcribe(audio) → user_text
4. classify_routing(user_text, has_active_coding_task, has_pending_clarification)
   → RoutingIntent
5. CapabilityVoiceController.handle_capability_intent(routing_intent)
   ├── CONVERSATIONAL: returns None
   ├── coding kinds: routes through CodingTaskRunner
   └── automation kinds: OpenClawDispatcher (stub voice msg)
6. If None (conversational fall-through):
   a. Orchestrator._respond(user_text)
      ├── Optional: WebSearchGate.classify(text) → SEARCH/NO_SEARCH/UNCERTAIN
      ├── If SEARCH: AcknowledgmentSource.next_phrase() → TTS immediately
      │              → WebSearchExecutor.run(text) → SearchPayload
      │              → format_sources_for_prompt(payload.sources)
      │              → injected into LLM context
      ├── ConversationMemory.retrieve(text) → MemoryTurn[] (RAG)
      ├── LLMEngine.generate_stream(text) → tokens
      └── TextToSpeech.speak_stream(tokens) → Piper → RVC → audio device
   b. ConversationMemory.add(user/assistant) on background thread
7. Orchestrator enters FOLLOW_UP_LISTENING for 30 s (warm window)
```

### Coding task path

```
1-4. Same as voice query through classify_routing
5. RoutingIntent.kind == CODE_TASK
   a. CapabilityVoiceController.handle_capability_intent →
      handle_utterance(text)
   b. CodingIntent classification (intent.classify)
   c. ProjectResolver resolves "my flask app" → Project
      OR new_sandbox_project(name) creates a fresh dir
   d. UltronMCPServer.create_session(project_root, intent)
   e. CodingTaskRunner.start_task(TaskRequest)
      → DirectClaudeCodeBridge.submit() spawns:
         claude --print --output-format stream-json --include-partial-messages
                --include-hook-events --model haiku --add-dir <cwd>
                --dangerously-skip-permissions
   f. TaskHandle event stream:
      ├── TaskEvent(STATUS|TEXT|TOOL_USE|TOOL_RESULT|FILE_CHANGE|USAGE|ERROR|COMPLETE)
      ├── Listener feeds: SessionStore.record_stage(), record_test_results(),
      │                   set_pending_clarification(), record_completion_claim()
      └── Audit log line per event → logs/coding_tasks.jsonl
6. Orchestrator main loop returns; voice path resumes
7. On future "how's it going?" utterance:
   a. classify_routing → PROGRESS_QUERY (because runner.has_active_task())
   b. handle_capability_intent → handle_utterance →
      StatusNarrator.narrate(session_state) using project_status_delta
      → spoken narration
8. On Claude declare_complete:
   a. ConversationCoordinator.handle_declare_complete():
      → Verifier.verify(session) runs 6 checks
      → if pass: SessionStatus.COMPLETE
      → if fail and below escalation threshold:
            project_correction_context(session) → corrective prompt
            → Claude re-prompted with --resume
      → if escalation threshold crossed: switch to sonnet model
   b. CodingTaskRunner.completion_narration() generates final voice msg
9. Orchestrator polls voice.pending_completion() → speaks it
```

### Search-triggered path (web)

```
1-3. Same as voice query through Whisper
4-5. classify_routing → CONVERSATIONAL (web search isn't a routing kind)
6. Orchestrator._respond(user_text) flow:
   a. WebSearchGate.classify(user_text):
      ├── classify_by_rules → SEARCH if time-sensitive markers,
      │                       NO_SEARCH if personal-context queries
      └── classify_by_preflight (LLM call) for UNCERTAIN cases
         → returns GateVerdict with knowledge_confidence,
                  has_temporal_dependency, search_queries
   b. If SEARCH:
      ├── AcknowledgmentSource.next_phrase() → TTS within 200 ms
      ├── WebSearchExecutor.run(user_text, search_queries):
      │   ├── WebResultsCache.lookup(q) → cached payload OR None
      │   │   (3 collections: ttl_volatile_seconds=86400, ttl_stable_seconds=2592000)
      │   ├── BraveSearchClient.search(q) → BraveResult[]
      │   │   (wrapped in CircuitBreaker; raises BraveAPIError;
      │   │    failures log to errors.jsonl, return [])
      │   ├── _rank_snippets(llm, query, results, top_n) → ranked BraveResult[]
      │   ├── For top max_fetch: JinaReaderClient.fetch(url) → markdown
      │   │   (wrapped in CircuitBreaker; JinaReaderError → snippet-only)
      │   └── WebResultsCache.store(query, rows) — best effort
      └── format_sources_for_prompt(payload.sources) → injected into LLM context
   c. LLM generates response with citations
   d. TTS streams + format_sources_for_transcript(sources) printed (not spoken)
```

### OpenClaw stub dispatch path (Phase 5 — currently stubbed)

```
1-4. Same as voice query through classify_routing
5. RoutingIntent.kind in {BROWSER_AUTOMATION, MEDIA_GENERATION, MESSAGING,
                          FILE_OPERATION, SHELL_OPERATION, HYBRID_TASK}
   → CapabilityVoiceController.handle_capability_intent:
   ├── Single-category (browser/media/etc):
   │   AutomationTaskRunner.submit_task(intent) →
   │   OpenClawDispatcher.handle_X(intent.automation_intent) →
   │   DispatchResult(success=False, voice_message="gateway not connected yet")
   │   → audit row in logs/automation_tasks.jsonl
   │   → routing-decision row with outcome="stub"
   └── HYBRID_TASK: voice msg "I'd split it up and run both, but..."
6. VoiceResponse returned to orchestrator → speak
7. Orchestrator continues main loop
```

### Error / circuit-break path

```
External call (Brave, Jina) → CircuitBreaker.call(_do_X, ...)
├── If CLOSED, executes; on failure raises typed error
│   - 3rd failure within 5 min trips OPEN
├── If OPEN, raises CircuitOpenError immediately (no call)
│   - cooldown elapses → HALF_OPEN
├── If HALF_OPEN, executes once as a probe
│   ├── Success → CLOSED, failure counter reset
│   └── Failure → reopens, fresh cooldown
└── On any typed-error path:
    ErrorLog.record(error, dependency=...) → logs/errors.jsonl
    Optional: phrase_for("brave_unavailable") → spoken via TTS
```

---

## Source modules

### `src/ultron/__init__.py`

**Purpose:** package init. On Windows, adds CUDA runtime DLL directories to
the loader path so llama-cpp / ctranslate2 find `cudart64_12.dll`,
`cublas64_12.dll`, `cudnn_ops_infer64_9.dll`.

**No public API** beyond import side effects.

### `src/ultron/__main__.py`

**Purpose:** `python -m ultron` entry point.

**Public:**
- `main() -> int` — sets up logging, builds an `Orchestrator`, calls
  `.run()` until KeyboardInterrupt. Returns process exit code.

**In:** environment + config.yaml (via Orchestrator construction).
**Out:** stdout console transcript, log files.

### `src/ultron/config.py` (Phase 3)

**Purpose:** single source of truth for tunable parameters. Loads
`config.yaml`, validates against pydantic schema, exposes singleton.

**Public:**
- `PROJECT_ROOT`, `MODELS_DIR`, `LOGS_DIR` — Path constants
- `DEFAULT_CONFIG_PATH` — `<root>/config.yaml`
- `resolve_path(value: str | Path) -> Path` — resolve relative paths against PROJECT_ROOT
- Sub-models (all pydantic `_Strict`):
  `AudioConfig`, `VADConfig`, `WakeWordConfig`, `STTConfig`, `LLMConfig`,
  `EmbeddingsConfig`, `QdrantCollections`, `QdrantConfig`, `MemoryConfig`,
  `BraveConfig`, `JinaConfig`, `WebCacheConfig`, `WebSearchConfig`,
  `AddressingConfig`, `CodingMCPConfig`, `CodingVerificationConfig`,
  `CodingConfig`, `ProjectionsBudgets`, `ProjectionsConfig`, `RVCConfig`,
  `TTSConfig`, `LoggingConfig`, `ErrorPhrasesConfig`,
  `RoutingClassifierConfig`, `RoutingConfig`, `OpenClawConfig`
- `UltronConfig` — top-level model
- `load_config(path=None) -> UltronConfig` — explicit load (raises `ConfigurationError`)
- `get_config() -> UltronConfig` — singleton, lazy-load on first call
- `reload_config(path=None) -> UltronConfig` — clear cache, reload
- `set_config(cfg) -> None` — test injection
- `current_config_path() -> Path | None`
- `LLM_PRESETS: dict[str, dict]` (4B plan Stage A) — preset table for
  `LLMConfig.preset`. Two presets defined: `qwen3.5-9b` (default; 9B
  GGUF, n_ctx=8192, no draft) and `qwen3.5-4b` (4B GGUF + 0.8B draft +
  n_ctx=16384). `LLMConfig._apply_preset` (model_validator) fills
  in `model_path` / `n_ctx` / `draft_model_path` from this table only
  when those fields are absent from `model_fields_set`, so explicit
  YAML values always win.

**In:** `config.yaml`, `${ENV_VAR}` substitution from `os.environ`.
**Out:** typed `UltronConfig` instance.

### `src/ultron/errors.py` (Phase 4)

**Purpose:** typed exception hierarchy for every external dependency.

**Public hierarchy:**
- `UltronError` (base) — has `message`, `context: dict`, `recovery: str`,
  `with_recovery()`, `with_context()`, `to_log_dict()`
- `DependencyUnavailableError` (subclass)
  - `BraveAPIError`, `JinaReaderError`, `QdrantUnavailableError`,
    `AnthropicAPIError`, `OllamaUnavailableError`, `OpenClawGatewayError`
- `ClaudeCodeError`
- `AudioPipelineError`
  - `WhisperTranscriptionError`, `PiperSynthesisError`,
    `RVCConversionError`, `WakeWordModelError`, `AddressingClassifierError`
- `MCPServerError`, `ConfigurationError`, `FilesystemError`

**In:** raised from external-dep wrappers.
**Out:** caught by orchestrator + structured-logged via `ErrorLog`.

**Wired call sites (Phase 4 deferred wrappers, complete):**
- `ClaudeCodeError` + `AnthropicAPIError`: [coding/direct_bridge.py](../src/ultron/coding/direct_bridge.py)
  — launch failure, subprocess timeout, nonzero exit, stream-json error
  events. The pattern detector `_looks_like_anthropic_api_error` decides
  between the two based on error text (rate_limit / overloaded /
  invalid_api_key / etc.).
- `MCPServerError`: [coding/mcp_server.py](../src/ultron/coding/mcp_server.py)
  — bind failure (`raise … from OSError`), startup timeout, no-active-session
  on Claude tool call. `FilesystemError` covers the audit-log write path.
- `FilesystemError`: [coding/audit.py](../src/ultron/coding/audit.py),
  [coding/projects.py](../src/ultron/coding/projects.py),
  [coding/runner.py](../src/ultron/coding/runner.py) — session audit
  mkdir/write, project registry load/save, coding-tasks audit-log
  (first-failure dedup via `_AUDIT_WRITE_FAILURE_LOGGED` flag).

### `src/ultron/uncertainty.py`

**Purpose:** annotate user prompt with hedging hints based on the
pre-flight gate's uncertainty signals.

**Public:**
- `apply(verdict: GateVerdict, user_text: str) -> Tuple[GateVerdict, str]`
  — given a `GateVerdict` with `knowledge_confidence` /
  `has_temporal_dependency`, returns a possibly-prepended user prompt
  with style hints.

**In:** `GateVerdict` from `web_search.gating`, raw user text.
**Out:** `(verdict, augmented_prompt)`.

### `src/ultron/audio/`

#### `audio/capture.py`
- `class AudioCaptureError(RuntimeError)` — raised on device init failure
- `class AudioCapture` — sounddevice callback thread enqueueing 32 ms blocks
  - `start()` / `stop()`
  - `read_blocks() -> Iterator[np.ndarray]`
  - `_capture_utterance(...)` (used by Orchestrator)

#### `audio/devices.py`
- `class AudioDeviceError(ValueError)`
- `resolve_device(configured, kind) -> Optional[int]` — substring match on device name
- `describe_device(device, kind) -> str`

#### `audio/ring_buffer.py`
- `class RingBuffer` — fixed-duration audio backlog (pre-speech window)

#### `audio/vad.py`
- `class SpeechEvent(Enum)` — START / END / NONE
- `class VadResult` — dataclass: event, is_speech, prob
- `class VoiceActivityDetector` — silero-vad wrapper; consumes 512-sample windows

#### `audio/wake_word.py`
- `class WakeWordDetector` — openWakeWord wrapper
  - Loads `models/openwakeword/ultron.onnx` (custom)
  - Falls back to `hey_jarvis` with startup warning if missing
  - `predict(audio_block) -> Optional[str]` — fires a wake event

### `src/ultron/addressing/`

#### `addressing/rules.py`
- `class AddressingDecision(str, Enum)` — ADDRESSED / NOT_ADDRESSED / UNCERTAIN
- `class RuleHit` — dataclass: decision, confidence, reason
- `classify(utterance, seconds_since_response) -> Optional[RuleHit]`
- `explain_rules() -> List[Tuple[str, str]]` — for the review script

#### `addressing/zero_shot.py`
- `class ZeroShotAddresseeModel` — flan-t5-small wrapper (~300 MB CPU)
  - `_ensure_loaded()` — eager-load option
  - `classify(utterance, context, seconds_since_response) -> (verdict_str, confidence, latency_ms)`

#### `addressing/classifier.py`
- `class AddressingVerdict` — final decision + metadata
- `class AddressingClassifier` — combines rules + zero-shot
  - `classify(utterance, seconds_since_response) -> AddressingVerdict`
  - `_log(utterance, verdict)` → writes to `logs/addressing.jsonl`

### `src/ultron/transcription/whisper_engine.py`

- `class WhisperEngine` — faster-whisper wrapper, CUDA fp16
  - `transcribe(audio: np.ndarray, language="en") -> str`
  - On failure: returns `""`, logs `WhisperTranscriptionError` to errors.jsonl

### `src/ultron/llm/inference.py`

- `_strip_thinking_blocks(stream)` — filter `<think>...</think>` from token stream
- `class LLMEngine` — LLM client with two backends, selected by `llm.runtime`:
  - `in_process` (default): loads the GGUF via llama-cpp-python in this process. Voice-path mode.
  - `http_server` (opt-in): talks to llama-cpp-server over OpenAI-compat HTTP. For the OpenClaw + voice migration. Latency is +71 ms median TTFT vs in-process — kept opt-in so the voice path isn't regressed.
  - `__init__(model_path?, n_ctx?, n_gpu_layers?, system_prompt?, history_turns?, memory=None, runtime?)`
  - `generate(user_message) -> str` — blocking
  - `generate_stream(user_message) -> Iterator[str]` — token streaming
  - `cancel()` — signal to stop
  - `_build_messages(user_message)` — resolves system prompt fresh each turn (Phase 1 hot-reload), assembles RAG snippets + recent + user
  - `_resolve_system_prompt()` (Phase 1) — sources from `PersonaLoader.get_system_prompt("user_facing")` when `llm.persona.source == "workspace"` (default), else `cfg.system_prompt`. Falls back to config when workspace is empty.
  - `_http_chat_completion(...)` / `_http_stream(...)` — OpenAI-compat HTTP client (uses `requests`, SSE for streaming, cancel-aware).

**In:** user text + (optional) `ConversationMemory` for RAG. **Out:** generated text.

### `src/ultron/memory/`

#### `memory/embedder.py`
- `class _SparseVec` — thin wrapper over BM25 sparse output
- `class HybridEmbedder` — FastEmbed dense (bge-small-en-v1.5 INT8) + sparse (Qdrant/bm25)
  - `encode_dense(texts) -> np.ndarray`
  - `encode_query_dense(text)` / `encode_query_sparse(text)`
  - `dim` property → 384

#### `memory/qdrant_store.py`
- `class MemoryTurn` — dataclass: id, ts, role, content, summary, entities, ...
- `class ConversationMemory`
  - `__init__(path?, embedder, recent_cache_size=100, session_id?)`
  - `add(role, content)` — sync; queues to background writer
  - `recent(n) -> List[MemoryTurn]` — from in-process cache
  - `retrieve(query, k=cfg, exclude_recent=cfg) -> List[MemoryTurn]` — hybrid RRF
  - `__len__()` / `close()`

### `src/ultron/web_search/`

#### `web_search/acknowledgments.py`
- `class AcknowledgmentSource` — shuffled-pool phrase generator (8 phrases)
  - `next_phrase() -> str`

#### `web_search/brave.py`
- `_BRAVE_BREAKER` — module-level CircuitBreaker (3/5min, 5min cooldown)
- `class BraveResult` — dataclass: url, title, snippet, rank
- `class BraveSearchClient`
  - `search(query, count?) -> List[BraveResult]` — uses breaker + raises BraveAPIError
  - `_do_search(query, count)` — inner; raises typed errors

#### `web_search/cache.py`
- `_VOLATILE_KEYWORDS`, `freshness_category_for(query)`, `ttl_for(category)`
- `class WebResultsCache` — Qdrant-backed; collection = `web_results`
  - `lookup(query) -> Optional[List[(BraveResult, full_text)]]`
  - `store(query, rows)` — best-effort

#### `web_search/gating.py`
- `class GateDecision(str, Enum)` — SEARCH / NO_SEARCH / UNCERTAIN
- `class GateVerdict` — decision, confidence, source, search_queries, knowledge signals
- `classify_by_rules(utterance) -> Optional[GateVerdict]` — hard rules (time markers, URL, etc.)
- `classify_by_preflight(utterance, llm, memory_snippets) -> GateVerdict` — LLM call
- `class WebSearchGate` — orchestrates rules → LLM
  - `classify(utterance, recent_memory) -> GateVerdict`

#### `web_search/jina.py`
- `_JINA_BREAKER` — CircuitBreaker (5/5min, 3min cooldown)
- `class JinaReaderClient`
  - `fetch(url) -> Optional[str]` — uses breaker + raises JinaReaderError

#### `web_search/search.py`
- `class SearchSource` — dataclass: url, title, snippet, full_text, rank
- `class SearchPayload` — dataclass: query, sources, cache_hit, elapsed_ms, notes
- `_rank_snippets(llm, query, results, top_n)` — LLM-driven re-ranking
- `class WebSearchExecutor` — orchestrates Brave → rank → Jina → cache
  - `run(user_query, search_queries?, top_n=3) -> SearchPayload`
- `format_sources_for_prompt(sources)` / `format_sources_for_transcript(sources)`

### `src/ultron/tts/`

#### `tts/rvc.py`
- `class RvcConverter` — infer-rvc-python wrapper, cuda:0
  - `convert(pcm: np.ndarray, sample_rate: int) -> (pcm, sr)` — raises RVCConversionError on failure
  - `close()` — releases GPU memory

#### `tts/speech.py`
- `class TextToSpeech` — Piper + optional RVC
  - `__init__(rvc=None)` — loads Piper voice, optionally wraps with RVC
  - `speak(text)` — synchronous synthesize + play
  - `speak_stream(fragments)` — stream tokens, flush on sentence terminator
  - `warmup()` — primes Piper
  - `_synthesize(text)` — Piper → optional RVC; raises PiperSynthesisError / RVCConversionError
  - `stop()` — interrupt current playback

### `src/ultron/coding/` (Phase A foundation + Coding Addendum + Phase 2 projections)

#### `coding/audit.py`
- `class SessionAuditWriter` — per-session `logs/sessions/<id>.jsonl` writer
  - `write(kind, **fields)` — append one record

#### `coding/bridge.py`
- `class EventKind(str, Enum)` — STATUS / TEXT / TOOL_USE / TOOL_RESULT / FILE_CHANGE / ERROR / COMPLETE / USAGE
- `class FileChangeKind(str, Enum)` — CREATED / MODIFIED / DELETED
- `class TaskEvent` — dataclass with all event payload fields
- `class TaskRequest` — dataclass: task_prompt, cwd, model, timeout_s, label, etc.
- `class TaskResult` — dataclass: success, exit_status, summary, files_*, etc.
- `class TaskState` — running state
- `class TaskHandle(ABC)` — `task_id()`, `state()`, `add_listener()`, `cancel()`, `wait()`
- `class CodingBridge(ABC)` — `submit(request) -> TaskHandle`, `name()`
- `render_prompt(request)` — render TaskRequest into a string prompt
- `directory_snapshot(root)` / `diff_snapshots(...)` — ground-truth file diff

#### `coding/direct_bridge.py`
- `class DirectClaudeCodeBridge(CodingBridge)` — spawns `claude --print --stream-json ...`
- `class DirectTaskHandle(TaskHandle)` — parses event stream

#### `coding/intent.py`
- `class CodingIntentKind(str, Enum)` — NONE / CODE_TASK / PROGRESS_QUERY / CANCEL / MID_SESSION_ADJUSTMENT / CLARIFICATION_RESPONSE
- `class CodingIntent` — dataclass with kind, project_reference, etc.
- `classify(utterance, has_active_task=False, has_pending_clarification=False) -> CodingIntent`
- `derive_project_name(intent) -> str` — slug from task text

#### `coding/projects.py`
- `class Project` — dataclass: name, path, aliases, language
- `class ProjectRegistry` — atomic JSON CRUD on `data/projects.json`
- `class ResolutionKind(str, Enum)` — EXACT / ALIAS / SUBSTRING / SEMANTIC / NEW / UNRESOLVED
- `class ProjectResolution` — dataclass with kind + matched project
- `class ProjectResolver` — exact / alias / substring / semantic match
- `slugify_for_path(name) -> str` — collision-safe slug
- `new_sandbox_project(name, sandbox_root, registry) -> Project` — creates fresh dir + registers

#### `coding/session.py`
- `class SessionStatus(str, Enum)` — INITIALIZING / EXECUTING / VERIFYING / CORRECTING / AWAITING_CLARIFICATION / COMPLETE / FAILED / TERMINATED
- `is_valid_transition(from_status, to_status) -> bool`
- Records: `StageRecord`, `FileRecord`, `TestStatus`, `ClarificationRequest`, `AdjustmentRecord`, `CompletionClaim`
- `class ProjectSession` — full session state (large; passed only via projections)
- `class StateTransitionError(RuntimeError)`
- `class SessionStore` — owns sessions; `create()`, `get()`, `transition()`, `record_*()`

#### `coding/projections.py` (Phase C / Foundation Part 2)
- `count_tokens(text) -> int` — tiktoken cl100k_base
- `class ProjectionResult` — projection + text + token_count + budget + truncations_applied + truncation_warning
- `_finalize_projection(...)` — common end-of-projection: INFO log on truncations, ERROR on over-budget
- 5 projections, each with a dataclass + `project_X_context()` function:
  - `project_clarification_context(session, clarification_question, options?, facts_lookup?) -> ProjectionResult` (1500 tok)
  - `project_status_delta(session) -> ProjectionResult` (600 tok)
  - `project_adjustment_context(session, adjustment_text, facts_lookup?, conflict_detector?) -> ProjectionResult` (1200 tok)
  - `project_correction_context(session, failures, failed_test_names?, failed_test_messages?) -> ProjectionResult` (1500 tok)
  - `project_completion_context(session) -> ProjectionResult` (800 tok)

#### `coding/templates.py`
- `class TemplateError(RuntimeError)`, `PromptTooLargeError`, `SchemaValidationError`
- `class RenderResult` — dataclass: rendered text + token count
- `class TemplateRenderer` — Jinja2 wrapper for prompts/coding/*.j2
  - `render_initial_new(...)`, `render_initial_edit(...)`, `render_correction(...)`,
    `render_adjustment(...)`, `render_clarification_response(...)`

#### `coding/verification.py`
- `class CheckId(str, Enum)` — STRUCTURE / TESTS / SMOKE / LINT / FILES / PYTHON_SYNTAX
- `class CheckResult`, `VerificationReport` — dataclasses
- `class Verifier`
  - `verify(session) -> VerificationReport` — runs 6 checks + writes `logs/verifications.jsonl`
  - `verify_tests(session)` — single-check helper

#### `coding/narration.py`
- `class NarrationDelta` — dataclass tracking what's new since last query
- `class StatusNarrator` — voice-friendly progress narration
  - `narrate(session) -> str` — final completion narration
  - `progress_narration(session) -> str` — uses `project_status_delta` projection

#### `coding/runner.py`
- `build_default_bridge() -> CodingBridge` — picks DirectClaudeCodeBridge from config
- `class ProgressSinceLastQuery` — dataclass
- `class CodingTaskRunner`
  - `start_task(request)` — submits via bridge
  - `has_active_task() -> bool`
  - `cancel_active() -> bool`
  - `progress_narration() -> str`
  - `completion_narration() -> Optional[str]`
  - `pop_budget_warning() -> Optional[str]`

#### `coding/coordinator.py`
- `class DecisionPath(str, Enum)` — RULE_ANSWER / LLM_AGREED / LLM_PICKED / USER_ANSWER / TIMEOUT
- `class ClarificationDecision`, `AdjustmentDecision`, `PendingUserClarification` — dataclasses
- `class ConversationCoordinator`
  - `decide_clarification(session_id, request, session) -> str` — answer or escalate
  - `decide_adjustment(session_id, adjustment_text) -> AdjustmentDecision`
  - `handle_declare_complete(session_id) -> str` — runs Verifier, drives correction loop
  - `pending_user_clarifications() -> List[PendingUserClarification]`

#### `coding/mcp_server.py`
- `class UltronMCPServer`
  - In-process Python tools (called by Qwen via `get_config().coding.mcp.host:port`):
    - `create_session()`, `get_full_state()` (Python only), `get_status_delta()`,
      `get_clarification_context()`, `get_adjustment_context()`,
      `get_correction_context()`, `get_completion_context()`,
      `send_followup()`, `terminate_session()`, `list_active_sessions()`
  - SSE worker tools (called by Claude Code via SSE):
    - `report_progress()`, `request_clarification()`, `report_test_results()`,
      `declare_complete()`, `abandon_task()`, `record_file_change()`
  - `set_clarification_responder(fn)` / `set_declare_complete_handler(fn)` — coordinator hooks
  - `start()` / `stop()` — manage SSE server
- `write_mcp_config(project_root, sse_url)` / `remove_mcp_config(project_root)`

#### `coding/voice.py`
- `class VoiceResponse` — dataclass: text, handled, cancelled
- `class CapabilityVoiceController` (Phase 5 rename; alias = CodingVoiceController)
  - `pending_completion()` / `pending_clarifications()` / `pending_budget_warning()`
  - `has_pending_clarification() -> bool`
  - `handle_utterance(text) -> Optional[VoiceResponse]` — coding-only (delegated by capability dispatch)
  - `handle_capability_intent(routing_intent) -> Optional[VoiceResponse]` — top-level dispatch (Phase 5)

### `src/ultron/openclaw_routing/` (Phase 5)

#### `openclaw_routing/intents.py`
- `class RoutingIntentKind(str, Enum)` — 12 values: CONVERSATIONAL, CODE_TASK, PROGRESS_QUERY, CANCEL, MID_SESSION_ADJUSTMENT, CLARIFICATION_RESPONSE, BROWSER_AUTOMATION, MEDIA_GENERATION, MESSAGING, FILE_OPERATION, SHELL_OPERATION, HYBRID_TASK
- Per-category dataclasses: `BrowserIntent`, `MediaGenIntent`, `MessagingIntent`, `FileOpIntent`, `ShellOpIntent`
- `HybridSubtask` — dataclass: order, type, subtype, description
- `RoutingIntent` — top-level dataclass: kind, raw_text, confidence, source, reason, coding_intent, automation_intent, subtasks, needs_user_clarification, clarification_question
- `DispatchResult` — dataclass: success, voice_message, error, metadata
- `TaskInfo` — task tracking dataclass
- `AutomationIntent` = Union of the 5 automation intent classes

#### `openclaw_routing/classifier.py`
- `classify_routing(utterance, has_active_coding_task=False, has_pending_clarification=False) -> RoutingIntent`
  Layered: in-flight commands → hybrid → coding → automation rules → CONVERSATIONAL fallback
- `_build_browser_intent(text)`, `_build_media_intent(text)`, `_build_messaging_intent(text)`, `_build_file_intent(text)`, `_build_shell_intent(text)` — extract structured intent from raw text

#### `openclaw_routing/dispatcher.py`
- `class OpenClawDispatcher` (currently STUBBED)
  - `__init__(config?)` — reads openclaw.enabled + routing.stub_responses_enabled
  - `async handle_browser(intent)` / `handle_media_generation(intent)` / `handle_messaging(intent)` / `handle_file_operation(intent)` / `handle_shell_operation(intent)` — all return DispatchResult with stub voice message + `metadata={"stub": True}`

#### `openclaw_routing/runner.py`
- `class AutomationTaskRunner` — mirror of `CodingTaskRunner` for automation tasks
  - `async submit_task(routing_intent) -> task_id` — dispatches via OpenClawDispatcher
  - `async progress_narration(task_id) -> Optional[str]`
  - `async completion_narration(task_id) -> Optional[str]`
  - `async cancel(task_id) -> bool`
  - `list_active() -> List[TaskInfo]` / `get_task(task_id)`
  - Audit log: `logs/automation_tasks.jsonl`

#### `openclaw_routing/decomposer.py`
- `class DecompositionResult` — subtasks + fallback_used + raw_response
- `class HybridTaskDecomposer`
  - `async decompose(utterance) -> DecompositionResult` — calls Qwen with JSON-output prompt, parses, falls back to one-element coding plan on any failure

#### `openclaw_routing/disambiguator.py`
- `class DisambiguationResult` — kind (CODE_TASK / HYBRID_TASK / CONVERSATIONAL / None) + clarification_question
- `class IntentDisambiguator`
  - `async disambiguate(utterance) -> DisambiguationResult` — asks Qwen "CODING/AUTOMATION/HYBRID/UNCLEAR"

#### `openclaw_routing/decision_log.py`
- `class RoutingDecisionLog` — JSONL writer (`logs/routing_decisions.jsonl`)
  - `record(intent, *, handler, outcome, extra?)` — best-effort append
- `get_routing_log() -> RoutingDecisionLog` — singleton
- `set_routing_log(log)` — test injection

### `src/ultron/openclaw_bridge/` (OpenClaw Phase 1 + 3 foundations)

The bridge layer between Ultron and the OpenClaw Gateway peer. Voice
pipeline is unaffected when OpenClaw is unreachable (`fail_open: true`).

#### `openclaw_bridge/persona.py` (Phase 1)

- `class PersonaLoader` — reads the six workspace files
  (IDENTITY/SOUL/USER/AGENTS/HEARTBEAT/BOOTSTRAP) and composes a
  system prompt for the requested mode. Hot reload via `refresh_if_stale`
  (mtime+size check on each call).
  - `load() -> PersonaBundle` — force a fresh read.
  - `refresh_if_stale() -> PersonaBundle` — reload only if anything
    changed; cheap.
  - `get_system_prompt(mode="user_facing") -> str` — composes per mode.
- `PromptMode = Literal["user_facing", "background", "heartbeat", "bootstrap"]`
  - `user_facing` — IDENTITY + SOUL + USER. Voice path; full Ultron
    character.
  - `background` — AGENTS only, prefixed with internal-worker framing.
    For heartbeat preflight, cron, summarization, tool selection.
  - `heartbeat` — HEARTBEAT only.
  - `bootstrap` — BOOTSTRAP only.
- `default_workspace_dir() -> Path` — resolves
  `~/.openclaw/workspace/` or `ULTRON_OPENCLAW_WORKSPACE` env override.
- `class PersonaBundle` / `PersonaFile` — dataclasses with
  fingerprint (`(name, mtime_ns, size)`) for change detection.
- HTML-comment-only files (e.g., a placeholder USER.md with
  `<!-- auto-populated by maintenance -->`) are treated as empty so
  they don't bloat the prompt.

#### `openclaw_bridge/lifecycle.py` (Phase 3 foundation)

- `class OpenClawLifecycle` — health probes for the OpenClaw Gateway.
  Never raises; voice path keeps working when Gateway is unreachable.
  - `is_reachable() -> bool` — sub-second probe against
    `/__openclaw__/canvas/`.
  - `wait_for_ready(timeout_s, poll_interval_s) -> bool` — startup
    block.
  - `get_status() -> OpenClawStatus` — snapshot (version, default
    agent, configured channels).
  - `auth_token` property — reads `gateway.auth.token` from
    `~/.openclaw/openclaw.json` lazily; never logs the token.
- `class OpenClawStatus` — frozen dataclass.

### `src/ultron/pipeline/orchestrator.py`

- `class State(Enum)` — IDLE / CAPTURING / PROCESSING / FOLLOW_UP_LISTENING
- `class Orchestrator` — main event loop
  - `__init__()` — composes audio, wake, vad, addressing, stt, llm, memory, web_search, tts, coding_voice
  - `run()` — main loop (blocks; KeyboardInterrupt clean shutdown)
  - `_capture_utterance()` — VAD-bounded audio capture
  - `_follow_up_listen(deadline)` — WARM-mode VAD loop
  - `_respond(user_text)` — LLM stream → TTS pipeline (with optional web search)
  - `_speak(text)` — single-shot synthesize + play
  - `_announce_coding_completion_if_pending()`, `_announce_pending_clarifications()`, `_announce_pending_budget_warning()` — voice-loop poll hooks
  - `_load_memory_if_enabled()` — Qdrant init with graceful fallback

**In:** mic input (sounddevice), config.yaml, models on disk.
**Out:** speaker output (sounddevice), all audit logs.

### `src/ultron/resilience/` (Phase 4)

#### `resilience/circuit_breaker.py`
- `class CircuitState(str, Enum)` — CLOSED / OPEN / HALF_OPEN
- `class CircuitOpenError(Exception)` — short-circuit signal
- `class CircuitBreaker`
  - `__init__(name, failure_threshold=3, window_seconds=300, cooldown_seconds=300, expected_exceptions=(Exception,))`
  - `call(func, *args, **kwargs) -> result` — raises CircuitOpenError when OPEN
  - `state`, `failure_count` properties
  - `reset()` — test/operator only

#### `resilience/error_log.py`
- `class ErrorLog` — append-only JSONL writer to `logs/errors.jsonl`
  - `record(error, *, dependency, session_id?, extra?, include_traceback=True)` — best-effort
- `get_error_log() -> ErrorLog` — singleton
- `set_error_log(log)` — test injection

#### `resilience/phrases.py`
- `phrase_for(failure_mode: str) -> Optional[str]` — shuffled cycle from `config.error_phrases.<mode>`
- `reset_phrase_cache()` — test-only

### `src/ultron/utils/`

#### `utils/logging.py`
- `configure_logging(level=None, log_file=None) -> None` — idempotent
- `get_logger(name) -> logging.Logger` — namespaced under `ultron.`

#### `utils/fairseq_compat.py`
- `patch_fairseq_dataclasses()` — workaround for fairseq's invalid omegaconf metadata
- `patch_torch_load_for_fairseq()` — torch.load weights_only compat shim

---

## Configuration

### `config.yaml` (project root) — single source of truth

Sections:
- `version: "1.0"`
- `audio` (sample_rate, channels, blocksize, dtype, devices, barge-in, ring buffer)
- `vad` (threshold, min_speech/silence durations, window_samples)
- `wake_word` (name, model_path, fallback_model, threshold, cooldown)
- `stt` (model, device, compute_type, beam_size, temperature, etc.)
- `llm` (provider="llama_cpp", **preset** ["qwen3.5-9b"|"qwen3.5-4b"|"custom"; auto-fills model_path/n_ctx/draft_model_path when those keys are omitted — Stage A of the 4B plan], runtime ["in_process"|"http_server"], model_path, draft_model_path, n_ctx, gpu_layers, temperature, top_p, max_tokens, repeat_penalty, history_turns, flash_attn, kv_cache_type, system_prompt, server.{base_url,...}, persona.{source,...})
- `embeddings` (dense_model, sparse_model, dense_dim)
- `qdrant` (data_dir="data/qdrant", collections.{conversations,facts,web_results})
- `memory` (enabled, jsonl_legacy_path, recent_turns, rag_top_k, rag_exclude_recent, facts_top_k, write_queue_maxsize)
- `web_search` (enabled, brave_api_key_env, brave/jina/cache subsections)
- `addressing` (follow_up_enabled, **warm_mode_duration_seconds: 30.0** ← user override, NOT 10s; rule_confidence_threshold, zero_shot_model, log_path)
- `coding` (enabled, bridge="direct", mcp.{host,port,...}, template_dir, prompt_token_budget, default/escalation models + thresholds, verification.{smoke,test,lint}_timeout, session_audit_dir, token_budget_per_session, claude_cli, sandbox_root, project_registry_path, audit_log_path, task_timeout, skip_permissions)
- `projections` (tokenizer, budgets.{clarification,status_delta,adjustment,correction,completion}_context, truncation_warning_threshold, log_truncations)
- `tts` (piper paths, sample_rate, sentence_flush_chars, length_scale, pause_ms, edge_fade_ms, rvc subsection)
- `logging` (file, level, format, datefmt)
- `error_phrases` (13 pools — qdrant_unavailable, brave_unavailable, jina_unavailable, anthropic_unavailable, rvc_unavailable, openclaw_unavailable, piper_unavailable, whisper_repeated_failures, addressing_classifier_failure, wake_word_model_failure, mcp_server_lost, claude_code_subprocess_failed, config_invalid)
- `routing` (llm_disambiguation_enabled, hybrid_task_decomposition_enabled, disambiguation_question_template, routing_log_path, classifier subsection, stub_responses_enabled)
- `openclaw` (enabled=false [stub], gateway_url, auth_token_env, health_check_*_seconds, fail_open, required_agent_id)

### `config/settings.py` (Phase 3 SHIM)

Compatibility shim that re-exports legacy `settings.X` constants from `get_config()`. Thin layer; HF cache pre-init runs at import time. Used by subsystems still on the legacy reference path (audio, wake_word, stt, tts, rvc, coding cluster, scripts) — see [docs/phase3_5_followup.md](phase3_5_followup.md) for the migration punch list.

### `.env.example` (and the actual `.env` in main checkout)

Env vars:
- `ULTRON_BRAVE_API_KEY` — Brave Search API key (required for web search)
- `ULTRON_LLM_MODEL_PATH` — opt-in override of GGUF path
- `ULTRON_AUDIO_DEVICE` / `ULTRON_AUDIO_OUTPUT_DEVICE` — operator-specific device strings
- `ULTRON_LOG_LEVEL` — console log level
- `ULTRON_CODING_MCP_ALLOW_ANY_ROOT=1` — test-only sandbox escape
- `ULTRON_CONFIG_PATH` — alternative config.yaml path

---

## Operational scripts

All scripts assume venv active in main checkout (`C:\STC\ultronPrototype`). Worktrees inherit the venv via shared `.venv\Scripts\python.exe`.

### `scripts/benchmark.py`

**Purpose:** measure end-to-end first-token latency for a single voice query.
**Run:** `python scripts/benchmark.py`
**In:** loads full voice stack + config.
**Out:** stdout — TTFT for one synthetic query.

### `scripts/check_vram.py`

**Purpose:** quick VRAM snapshot.
**Run:** `python scripts/check_vram.py [--watch [N]] [--gpu N]`
**In:** nvidia-smi.
**Out:** stdout — `<used> MB used | of <total> MB | target 9216 MB | cap 11500 MB | [OK/above target/WARN/CRITICAL]`
**Functions:** `vram_used_mb(gpu_id) -> Optional[int]`, `vram_total_mb(gpu_id)`, `gpu_name(gpu_id)`, `_format_line(used, total)`, `main(argv)`.

### `scripts/download_models.py`

**Purpose:** first-run model fetcher (Qwen GGUF, Piper, faster-whisper, openWakeWord).
**Run:** `python scripts/download_models.py`
**In:** Hugging Face Hub.
**Out:** files under `models/`.

### `scripts/dump_session.py`

**Purpose:** render coding-session audit log into a readable transcript.
**Run:** `python scripts/dump_session.py [--list | --latest | <session_id> | <path/to/file.jsonl>] [--sessions-dir DIR]`
**In:** `logs/sessions/<id>.jsonl`.
**Out:** stdout — formatted event list (one line per event with timestamp + kind + summary).
**Functions:** `_resolve_session_path(token, dir)`, `_read_records(path)`, `_format_record(rec)`, `main(argv)`.

### `scripts/list_audio_devices.py`

**Purpose:** mic / output device introspection.
**Run:** `python scripts/list_audio_devices.py`
**Out:** stdout — devices indexed by ID + name.

### `scripts/maintenance.py`

**Purpose:** periodic Qdrant maintenance (summarize old conversations into `facts`, label clusters, prune stale `web_results`, extract entities).
**Run:** `python scripts/maintenance.py`
**In:** Qdrant `conversations` collection, LLM, `data/maintenance.sqlite` (state).
**Out:** writes to `facts` collection, `data/summaries.jsonl`, updates sqlite.

### `scripts/measure_baseline.py`

**Purpose:** voice-path VRAM + TTFT baseline (10 representative queries; full stack loaded).
**Run:** `python scripts/measure_baseline.py`
**In:** loads full voice stack; runs 10 hard-coded representative queries.
**Out:** writes `baselines.json` (top-level metadata/vram_mb/latency_ms keys).

### `scripts/measure_baseline_extended.py` (Foundation Phase 0)

**Purpose:** extended baselines — search VRAM, coding-session VRAM, TTA microbench, scenario timing, composite TTFA.
**Run:** `python scripts/measure_baseline_extended.py [--lite | --full | --all]`
**Modes:**
- `--lite`: CPU-only — TTA microbench, scenario timing, composite TTFA. ~30 s.
- `--full`: also loads voice stack + measures search/coding VRAM. ~3 min.
- `--all`: both (default).
**In:** config + models + tests/coding/test_orchestration.py runtime.
**Out:** writes `baselines.json` `phase_foundation_start.measurements_extended` block.

### `scripts/migrate_memory_to_qdrant.py`

**Purpose:** one-shot ingest of `data/memory.jsonl` into Qdrant `conversations` collection.
**Run:** `python scripts/migrate_memory_to_qdrant.py`
**In:** `data/memory.jsonl`.
**Out:** `data/qdrant/` collections populated.

### `scripts/review_addressing.py`

**Purpose:** read `logs/addressing.jsonl`, print recent classifier verdicts.
**Run:** `python scripts/review_addressing.py [--tail N] [--misses] [--log PATH]`
**Modes:** `--misses` shows only NOT_ADDRESSED for false-negative tuning.
**Out:** stdout — `HH:MM:SS  DECISION  source  conf  latency  "utt"  -- reason`

### `scripts/run_integration_tests.py` (Foundation Part 7)

**Purpose:** wraps `pytest tests/integration tests/routing tests/error_recovery` with `--gpu` for `PYTEST_RUN_GPU_TESTS=1`.
**Run:** `python scripts/run_integration_tests.py [--gpu] [-q]`
**In:** test files.
**Out:** pytest output to stdout + final summary line with wall-clock + exit code.

### `scripts/run_orchestration_tests.py`

**Purpose:** run the 10 orchestration scenarios in `tests/coding/test_orchestration.py` with reporting.
**Run:** `python scripts/run_orchestration_tests.py`
**Out:** stdout — per-scenario pass/fail + total timing.

### `scripts/validate_config.py` (Foundation Part 7)

**Purpose:** validate `config.yaml` against pydantic schema without starting Ultron.
**Run:** `python scripts/validate_config.py [path] [--print]`
**Out:** stdout — "Configuration is valid." or detailed `ConfigurationError` with path + message + context. Exit 0 = valid, 1 = invalid.

### `scripts/start_llamacpp_server.py` (OpenClaw integration Phase 0 + 4B plan Stage C)

**Purpose:** launch llama-cpp-server on `127.0.0.1:8765` with the same params as the in-process voice loader (n_ctx=8192, flash_attn, Q8_0 KV cache). Imports `ultron` first so bundled torch CUDA DLLs are found before `llama_cpp` initialises (Windows-specific quirk).
**Run:** `python scripts/start_llamacpp_server.py [--n-ctx N] [--port P] [--api-key K] [--chat-format F] [--model-draft <path>] [--draft-num-pred-tokens N] [--from-config]`. The Stage C flags add speculative decoding (`--model-draft` + `--draft-num-pred-tokens`, mapped to llama-cpp-python's `draft_model` / `draft_model_num_pred_tokens`) and a `--from-config` overlay that reads model/draft/n_ctx from `config.yaml:llm` (preset-aware). CLI flags override the overlay. Pure-Python helpers `_build_arg_parser`, `_resolve_kwargs`, `_config_overlay` factor out the testable pieces.
**Out:** uvicorn HTTP server on `--port` (default 8765); stays in foreground.

### `scripts/supervised_llamacpp_server.py` (OpenClaw integration Phase 0)

**Purpose:** Python supervisor wrapper for `start_llamacpp_server.py`. Spawns the launcher as a subprocess, restarts on death with exponential backoff (2 s → 60 s cap, healthy_after_s=30 resets). Lighter alternative to NSSM.
**Run:** `python scripts/supervised_llamacpp_server.py [--cwd ...] [--max-restarts N] [--child-arg ...]`
**Out:** tee'd stdout/stderr from the child + supervisor restart events to stderr.

### `scripts/_bench_llm_http.py` (OpenClaw integration Phase 0)

**Purpose:** TTFT benchmark for the HTTP-runtime LLMEngine. Same 10 representative queries as `measure_baseline.py`, hits llama-cpp-server over HTTP.
**Run:** `python scripts/_bench_llm_http.py` (server must be running on the configured base URL).
**Out:** writes `baselines.json` `llm_http_runtime` block (median, p95, per-query). Used to compare HTTP-mode vs in-process mode latency.

### `scripts/_log_proxy.py` (OpenClaw integration Phase 0; debug only)

**Purpose:** tee proxy on `127.0.0.1:8766` that forwards to `127.0.0.1:8765` and logs every request body + SSE stream to stdout. Used to debug what OpenClaw actually sends to llama-cpp-server.
**Run:** `python scripts/_log_proxy.py` (point OpenClaw's `models.providers.litellm.baseUrl` at the proxy port instead of the server port).

### `scripts/smoke_test_llamacpp.ps1` (OpenClaw integration Phase 0)

**Purpose:** PowerShell smoke test for llama-cpp-server. Hits `/v1/models` and `/v1/chat/completions` with a tiny prompt; prints timing + completion text. Used to verify the server is healthy before involving OpenClaw.
**Run:** `pwsh scripts/smoke_test_llamacpp.ps1`

---

## Tests

### `tests/conftest.py` — Path setup so `from ultron.*` works.

### Default suite (no env gate) — 762 tests, ~30 s wall

**Top-level (~25 files):**
- `test_addressing.py` — rule-based addressing classifier
- `test_audio.py` — capture, ring buffer, devices
- `test_coding_bridge.py` — CodingBridge abstract contract
- `test_coding_e2e.py` — coding e2e (PYTEST_RUN_GPU_TESTS gated)
- `test_coding_intent.py` / `test_coding_intent_phase2.py` — intent classifier
- `test_coding_projects.py` — registry + resolver + sandbox creation
- `test_coding_runner.py` — runner state machine
- `test_coding_templates.py` — template renderer
- `test_coding_voice.py` — voice controller (now CapabilityVoiceController)
- `test_coordinator.py` — clarification + correction loops
- `test_correction_loop.py` — corrective re-prompting
- `test_fairseq_compat.py` — torch.load + dataclass workarounds
- `test_llm.py` — LLM (PYTEST_RUN_GPU_TESTS gated)
- `test_maintenance.py` — periodic maintenance
- `test_mcp_e2e.py` / `test_mcp_server.py` / `test_mcp_session.py` — MCP layer
- `test_memory_qdrant.py` — Qdrant memory + embedder
- `test_narration.py` — StatusNarrator
- `test_phase7_audit_and_tokens.py` — per-session audit + token tracking
- `test_pipeline.py` — orchestrator construction (PYTEST_RUN_GPU_TESTS gated)
- `test_projections.py` — 29 projection tests (Phase 2 + Foundation Part 2)
- `test_transcription.py` — Whisper (PYTEST_RUN_GPU_TESTS gated)
- `test_tts.py` — Piper + RVC
- `test_uncertainty.py` — uncertainty signal application
- `test_verification.py` — six verification checks
- `test_web_gating.py` — two-stage gating
- `test_persona_loader.py` (20, OpenClaw Phase 1) — `PersonaLoader` modes / hot-reload / HTML-comment-only files
- `test_llm_persona_source.py` (8, OpenClaw Phase 1) — `LLMEngine` persona-source wiring + hot-reload + fallback
- `test_llm_http_runtime.py` (9, OpenClaw Phase 0) — HTTP-runtime construction, request shape, SSE streaming, cancel mid-stream
- `test_llm_preset.py` (13, 4B plan Stage A) — `LLMConfig.preset` resolution: 9b/4b/custom defaults, explicit-override wins, YAML round-trip, invalid preset rejected
- `test_start_llamacpp_server.py` (13, 4B plan Stage C) — launcher CLI: --help renders, default args back-compat, --model-draft attaches speculative decoding, --draft-num-pred-tokens override, --from-config overlay (4b/9b), CLI flags override overlay

**`tests/coding/`:**
- `mock_bridge.py` — `ScriptedClaudeBridge` + `ClaudeScript` DSL
- `test_orchestration.py` — 11 mock-bridge scenarios (10 spec + 7b delta-tracking)
- `test_orchestration_real.py` — same scenarios with real Claude (gated)
- `test_mock_bridge_smoke.py` — mock-bridge sanity
- `sandbox/` — fixture sandbox

**`tests/error_recovery/`** (Phase 4) — 78 tests:
- `test_brave_failures.py`, `test_jina_failures.py`, `test_qdrant_failures.py`
- `test_audio_failures.py`, `test_addressing_failures.py`, `test_config_failures.py`
- `test_circuit_breaker.py`, `test_error_log.py`
- `test_claude_code_failures.py` (18) — launch fail / timeout / nonzero exit / stream-json error events with API-pattern detection
- `test_mcp_server_failures.py` (3) — bind failure / no active session / audit-log write failure
- `test_filesystem_failures.py` (5) — session audit / project registry / coding tasks audit-log dedup

**`tests/routing/`** (Phase 5) — 148 tests:
- `test_classifier.py` (90: 20 BROWSER, 10 each MEDIA/MESSAGING/FILE/SHELL/HYBRID/CONVERSATIONAL, 8 CODE_TASK, 2 edge)
- `test_dispatcher.py` (12)
- `test_decomposer.py` (9)
- `test_disambiguator.py` (25)
- `test_decision_log.py` (8)
- `test_backward_compat.py` (4)

**`tests/integration/`** (Phase 6) — 83 tests:
- `test_routing_dispatch.py` (20)
- `test_conversational_pipeline.py` (21)
- `test_search_pipeline.py` (12)
- `test_coding_pipeline.py` (9)
- `test_addressing_pipeline.py` (13)
- `test_error_recovery_pipeline.py` (4)
- `mocks.md` + `performance.json` (reference files)

### Slow / GPU-gated tests (16 skipped by default)

Set `$env:PYTEST_RUN_GPU_TESTS = "1"` before pytest. Includes real Claude API calls (`test_coding_e2e.py`, `test_mcp_e2e.py`, `test_orchestration_real.py`) — burns tokens.

---

## Runtime artifacts

### `logs/`

| File | Writer | Format | Purpose |
|---|---|---|---|
| `ultron.log` | `utils.logging.configure_logging()` | text, rotating 5 MB×3 | Main log — all subsystem messages |
| `addressing.jsonl` | `AddressingClassifier._log()` | JSONL | Every classifier verdict |
| `coding_tasks.jsonl` | `CodingTaskRunner._make_log_listener()` | JSONL | Coding task progress events |
| `verifications.jsonl` | `Verifier.verify()` | JSONL | Per-verification report |
| `clarifications.jsonl` | `_ClarificationLog` (in coordinator) | JSONL | Clarification decisions |
| `mcp_calls.jsonl` | `_AuditLog` (in mcp_server) | JSONL | MCP tool calls |
| `sessions/<id>.jsonl` | `SessionAuditWriter` | JSONL | Per-session full event audit |
| `errors.jsonl` | `resilience.error_log.ErrorLog.record()` | JSONL | Phase 4 typed errors |
| `routing_decisions.jsonl` | `openclaw_routing.decision_log.RoutingDecisionLog.record()` | JSONL | Phase 5 routing audit |
| `automation_tasks.jsonl` | `AutomationTaskRunner._audit()` | JSONL | Phase 5 OpenClaw task records |

### `data/`

| Path | Owner | Purpose |
|---|---|---|
| `qdrant/` | `ConversationMemory`, `WebResultsCache` | Embedded Qdrant store; 3 collections |
| `memory.jsonl` | (legacy) | Pre-Qdrant turn log; migration source / recovery |
| `projects.json` | `ProjectRegistry` | Coding project registry |
| `sandbox/` | `new_sandbox_project()` | Auto-created coding projects |
| `summaries.jsonl` | `scripts/maintenance.py` | Conversation summaries |
| `maintenance.sqlite` | `scripts/maintenance.py` | Maintenance state (cursors, etc.) |
| `ollama_compat_test/` | (Foundation Phase 0) | Modelfile from Ollama compat test (not in active use) |

### `models/` (main checkout only)

| File | Used by | Size |
|---|---|---|
| `Qwen3.5-9B-Q4_K_M.gguf` | `LLMEngine` (when `llm.preset == "qwen3.5-9b"`, current default) | 5.29 GB |
| `Qwen3.5-4B-Q4_K_M.gguf` | `LLMEngine` (when `llm.preset == "qwen3.5-4b"`, primary after Stage H) | 2.55 GB |
| `Qwen3.5-0.8B-Q4_K_M.gguf` | speculative-decoding draft for 4B (paired by `qwen3.5-4b` preset; Stage C wires into `start_llamacpp_server.py`) | 0.50 GB |
| `openwakeword/ultron.onnx` | `WakeWordDetector` | small |
| `piper/en_US-ryan-medium.onnx[.json]` | `TextToSpeech` | ~60 MB |
| `rvc/hubert_base.pt` | `RvcConverter` | ~362 MB |
| `rvc/rmvpe.pt` | `RvcConverter` | ~178 MB |
| `.hf-cache/` | `HybridEmbedder`, addressing zero-shot | varies |

### `ultron_james_spader_mcu_6941/` (main checkout only)

RVC voice model for Ultron timbre.
- `Ultron.pth` — main RVC checkpoint
- `added_IVF301_Flat_nprobe_1_Ultron_v2.index` — speaker index

---

## Documentation index

Reading order for a fresh Claude:

1. **`MEMORY.md`** (auto-loaded) — index of memory files
2. **`project_ultron.md`**, **`project_ultron_foundation.md`** — current state
3. **`feedback_*.md`** — confirmed user decisions
4. **`docs/codebase_structure.md`** ← THIS FILE — single-source reference
5. **`docs/architecture.md`** — pipeline + diagrams
6. **`docs/phase3_5_followup.md`** — open punch list

For specific tasks:
- Day-to-day operation: [docs/operations.md](operations.md)
- Adding code / debugging: [docs/development.md](development.md)
- Config reference: [docs/configuration.md](configuration.md)
- Error handling: [docs/error_handling.md](error_handling.md)
- Capability routing: [docs/routing.md](routing.md)
- Test layout: [tests/integration/mocks.md](../tests/integration/mocks.md)
- 16-step end-to-end smoke test: [docs/smoke_test.md](smoke_test.md)
- Foundation Phase 1 inventory snapshot: [docs/system_inventory.md](system_inventory.md)
- Phase 3 discovery catalog: [docs/config_discovery.md](config_discovery.md)
- **OpenClaw integration architecture + Phase 0/1 status:** [docs/openclaw_integration.md](openclaw_integration.md)
- **OpenClaw runtime ops (agents, supervisor, locked-in constraints):** [docs/openclaw_runtime.md](openclaw_runtime.md)
- **Phase 1 close-out report (persona migration):** [docs/phase_1_summary.md](phase_1_summary.md)
- **4B-model optimization plan (Stages A + B done; C in flight):** [docs/4b_optimization_plan.md](4b_optimization_plan.md)
- **GGUF SHA256 reference:** [docs/model_checksums.md](model_checksums.md)

---

## Maintenance contract

**This document is the operating manual. Keep it current.**

Update this file when you:

1. **Add a new module file** under `src/ultron/` →
   - Add to the file tree.
   - Add a section under "Source modules" with the public API (classes, functions, brief in/out).
   - If it's a new subsystem (e.g. `src/ultron/notifications/`), add to the architecture diagram in `docs/architecture.md` too.

2. **Add a new public class or function** to an existing module →
   - Add it to the module's section under "Source modules".
   - Note the inputs and outputs in one line.

3. **Add a new script** under `scripts/` →
   - Add to the file tree.
   - Add a section under "Operational scripts" with purpose, run command, in/out, and functions.

4. **Add a new test directory or test category** →
   - Add to the file tree.
   - Add to the relevant "Tests" subsection.

5. **Add a new log file or data path** →
   - Add to the "Runtime artifacts" tables.

6. **Add a new doc** under `docs/` →
   - Add to the "Documentation index".
   - Cross-reference where relevant in other sections.

7. **Add a new config section / key** →
   - Add to the `config.yaml` summary in "Configuration".
   - Update [docs/configuration.md](configuration.md) too (per-key reference).

8. **Change a cross-cutting flow** (voice path, coding path, search path, dispatch path) →
   - Update the relevant diagram in "Cross-cutting flows".

9. **Migrate a subsystem out of the `config/settings.py` shim** →
   - Update [docs/phase3_5_followup.md](phase3_5_followup.md) (cross off).
   - If it changes the public API of the migrated module, update its "Source modules" section here.

The goal: a fresh Claude Code session reads this document + the memory
files and is fully oriented without needing to re-explore the
codebase. If that's not the case after your changes, fix this document.

To verify the document still matches reality:
```powershell
# Run after any non-trivial change
pytest tests/ -q
python scripts/validate_config.py
# Then re-read this doc and confirm tree + module sections are current
```
