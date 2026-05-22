# Comprehensive end-to-end test plan

User-requested exhaustive test architecture covering every subsystem,
component, and feature of project Ultron. Captures all metrics
(latency, quality, accuracy, resource usage, intelligence,
creativity, adaptability, coding ability, web search ability) and
fixes any bugs discovered without compromising the voice baseline
contract.

## Binding constraints (carried forward from the project-root standards doc)

1. **Voice baseline preserved:** TTFT median ≤ 79 ms, VRAM peak ≤
   7913 MB on the 4B preset. Any change that touches the hot path
   must re-measure and document the delta. **Same numbers required
   at end of test pass as start.**
2. **No paid-API sprawl.** Anthropic API (Claude Code) and Brave
   used **extremely sparingly** for proof-of-life only. ≤3 Brave
   calls; ≤1 Claude Code task. ComfyUI / mobile node setup
   intentionally skipped (interactive / hardware).
3. **No voice-quality regressions.** Don't modify Piper, RVC, the
   LLM model file, or any voice-quality parameter.
4. **Fail-open everywhere.** External-dependency failures must
   degrade gracefully, never crash the orchestrator.
5. **codebase_structure.md updated alongside any code change** per
   its binding maintenance contract.
6. **No documentation files created beyond this plan + the final
   report**, both of which are explicitly user-requested.

## Coverage matrix

| Subsystem | Components | Test layers |
|---|---|---|
| Audio capture | AudioCapture, RingBuffer, AudioDeviceError | unit (existing); resource snapshot |
| VAD | Silero, SpeechEvent, VadResult | unit (existing) |
| Wake word | openWakeWord, ultron.onnx, hey_jarvis fallback, fired_recently | unit (existing); model presence check |
| Addressing | rules.py, zero_shot.py, classifier.py | accuracy on labeled set |
| STT | WhisperEngine | resource snapshot; TTFT contribution |
| LLM | LLMEngine in_process + http_server, presets, enable_thinking, RAG position, reload_for_preset | TTFT distribution; VRAM peak; preset swap; thinking on/off; RAG position A/B |
| Embeddings | HybridEmbedder dense + BM25 | encoding latency; vector-dim assertions |
| Memory | ConversationMemory, FactRow, retrieve, retrieve_multi, retrieve_for_query, search_facts | concurrent writes; retrieval latency; multi-pass A/B |
| Memory ranking | RankingWeights, composite scoring | unit (existing); rank-stability checks |
| Web search Brave | BraveSearchClient, circuit breaker | rules accuracy; circuit-breaker state machine; live 1-2 query smoke |
| Web search Jina | JinaReaderClient, circuit breaker | unit (existing); live 1 fetch smoke |
| Web search gate | WebSearchGate, GateVerdict, _resolve_knowledge_source | accuracy on labeled queries; preflight latency |
| Web search exec | WebSearchExecutor, _dedupe_queries (V1-gap B2), _render_inline_marker (V1-gap B3) | dedup correctness; superscript rendering |
| TTS Piper | TextToSpeech, sentence streaming | resource snapshot; first-sentence latency |
| TTS RVC | RvcConverter | resource snapshot; conversion latency |
| Coding bridge | CodingBridge, DirectClaudeCodeBridge, TaskEvent vocabulary | live 1 task smoke (sparing); event stream parse |
| Coding intent | classify, CodingIntentKind | classifier accuracy on labeled utterances |
| Projects | ProjectRegistry, ProjectResolver, new_sandbox_project | unit (existing); CRUD round-trip |
| Coordinator | ConversationCoordinator, DecisionPath including FACT_ANSWER (A3), facts_lookup wiring | clarification fast-path coverage |
| MCP server | UltronMCPServer, lookup_facts (A3) | unit (existing); SSE handshake |
| Narration | StatusNarrator, NarrationDelta | unit (existing) |
| Projections | 5 projections + ProjectionResult, _finalize_projection | budget-respect under stress |
| Templates | TemplateRenderer | render-budget enforcement |
| Verification | Verifier, six checks | check-by-check correctness |
| Coding voice | CapabilityVoiceController, VoiceResponse, pre_task_confirmation (A4), deferred_dispatch | A4 barge-in flow; MODEL_SWITCH dispatch |
| Capability routing | classify_routing, OpenClawDispatcher, AutomationTaskRunner | accuracy on labeled set; gating regression |
| Decomposer / Disambiguator | HybridTaskDecomposer, IntentDisambiguator (with optional IRMA) | output-shape robustness; fallback paths |
| Decision log | RoutingDecisionLog, get_routing_log | append correctness |
| OpenClaw persona | PersonaLoader, four modes, refresh_if_stale | hot-reload behavior |
| OpenClaw lifecycle | OpenClawLifecycle, OpenClawStatus | health-probe edges |
| OpenClaw client | OpenClawClient, all CLI methods, plugin methods (A1) | mocked CLI path; structured-failure shapes |
| Workspace writer | WorkspaceWriter, atomic writes + filelock | concurrent-write integrity (covered) |
| OpenClaw events | OpenClawEventReceiver | prefix matching; dispatch swallowing |
| MCP registrar | UltronMcpRegistrar, idempotency, retry | mocked path |
| Bridge holder | OpenClawBridge, from_config / start / shutdown / fire_and_forget / record_heartbeat_alert / auto-resolve | construction with enabled=False vs True+offline |
| Notifications | NotificationDispatcher | per-event gating; recipient resolution; transport-error fail-open |
| Heartbeat alerts | HeartbeatAlertLog | record/get/ack/prune |
| Browser tool | BrowserTool | six primitives via mocked invoke_tool |
| Desktop / Window | DesktopTool, WindowControlTool (V1-gap C3) | tool-slug routing; gaming-mode short-circuit |
| Gaming mode | GamingModeManager (V1-gap A1) | engage/disengage state machine |
| MCP tools (Phase 13) | get_heartbeat_alerts / acknowledge_alert / run_maintenance / list_active_coding_sessions / get_recent_voice_alerts | impl-function correctness |
| System status | SystemStatusReporter (Phase 13), SYSTEM_STATUS intent | reporter rendering; classifier patterns |
| Resilience | CircuitBreaker, ErrorLog, phrases.phrase_for | three-state machine; phrase shuffling |
| Item 4 compression | Compressor, _format_rag_block wiring, format_sources_for_prompt | token-reduction measurement (live) |
| Item 5 IRMA | InputReformulator | reformulation shape, default-OFF passthrough |
| Item 6 self-consistency | majority_vote_text/json/label, run_self_consistency | accuracy lift @ p_correct=0.7 (Monte Carlo) |
| Item 7 canonical monitor | CanonicalPathMonitor | abort latching, voice narration queue |
| Item 8 block-and-revise | ToolCallValidator | ALLOW/BLOCK paths, fail-open shape |

## Phase plan (sequential; each phase has a verification gate)

### Phase 0 — Pre-flight baseline

- `python scripts/validate_config.py` — schema valid
- `git status -uno` + `git log --oneline -5` — clean state
- Snapshot `logs/` file sizes (we're recording on top of this)
- Snapshot `data/qdrant/` size
- **Output:** `docs/comprehensive_test_report.md` Phase 0 section.
- **Gate:** config valid; HEAD matches `2fb0988` validated reference.

### Phase 1 — Full pytest sweep (timed, categorized)

Single command: `pytest tests/ -q --no-header --ignore=tests/coding/test_orchestration_real.py --durations=50`

- Capture: pass/fail counts, slowest 50 tests, total wall, any unexpected skip categories.
- **Output:** `pytest_baseline.json` (parsed pytest result), report section.
- **Gate:** ≥1474 passed; 15 skipped; 0 failed.

### Phase 2 — Voice-baseline regression gate

Run from main checkout (worktree has no `models/`):
```
cd C:\STC\ultronPrototype
.venv\Scripts\python.exe scripts\measure_baseline.py
```

- Compare baselines.json TTFT median + VRAM peak against 79 ms / 7913 MB.
- **Output:** report section with delta.
- **Gate:** TTFT median ≤ 79 ms + 5 ms tolerance; VRAM peak ≤ 7913 MB.

### Phase 3 — Per-subsystem deep tests

Each subsystem is exercised under one of:
- (A) Existing unit tests — already covered by Phase 1; pull metrics
- (B) Stress / property tests added inline — write small harness, run, capture
- (C) Live-stack measurement — needs models loaded; runs in main checkout

Subsystems:
1. Audio + VAD + wake_word (A; B for wake-word model file presence)
2. Addressing (A; B = labeled-utterance accuracy)
3. Whisper STT (A; C = synthesis-driven transcribe loop, 5 known phrases)
4. LLM (A; C = TTFT distribution at 8/16/32 history turns; preset reload round-trip; thinking on/off A/B; RAG recency vs system A/B)
5. Embeddings (A; B = encoding latency for 100-text batch)
6. Memory (A; B = concurrent-write stress 8 threads × 50 turns)
7. Web search (A; B = circuit-breaker state machine simulation; C = 1 live query)
8. TTS Piper + RVC (A; C = synthesize 5 phrases, capture latency)
9. Coding subsystems (A; subsystem-by-subsystem timing pull from Phase 1 durations)
10. Capability routing (A; B = labeled-utterance classifier accuracy)
11. OpenClaw bridge (A; B = enabled-but-offline behaviour; bridge=None invariants)
12. Items 4–8 (A; C = `python scripts/verify_items_4_to_8.py`)

### Phase 4 — Capability routing accuracy

- Seed labeled-utterance set spanning all 17 RoutingIntentKind values.
- Run `classify_routing` on each.
- Compute confusion matrix + per-kind precision/recall.
- **Gate:** every per-kind precision ≥ 0.85 on the labeled set.

### Phase 5 — Web search gate accuracy + breakers

- 30 labeled queries (the existing `scripts/benchmark_preflight.py` set).
- Compute classifier accuracy + median preflight latency.
- Trip Brave breaker via 3 forced failures; verify state transitions.
- Trip Jina breaker similarly.

### Phase 6 — Memory + Qdrant stress

- 1000-turn ingest into a tmp Qdrant; measure write throughput.
- 100 retrieve calls; measure median + p95 latency.
- Multi-pass retrieval vs single-pass A/B on 20 queries with synthetic context_categories.

### Phase 7 — LLM characterization

- Cold load → full set of `_chat_completion_kwargs` permutations.
- TTFT distribution at history_turns=0/4/8/12.
- enable_thinking=False vs True latency delta (5 queries each).
- RAG position recency vs system on 5 queries with seeded snippets.
- preset swap 4B → 9B → 4B; verify failure-safe (with intentionally-invalid preset).

### Phase 8 — OpenClaw bridge + V1-gap classifier gating

- bridge=disabled (default): every OpenClaw-bound intent in dispatcher returns stub.
- bridge=enabled, Gateway down: WARN logs, retry thread launches.
- gating regression: utterances "I'm about to play Valorant", "take a screenshot of the desktop", "focus the chrome window" → CONVERSATIONAL when openclaw.enabled=False.
- Same utterances → GAMING_MODE / DESKTOP_AUTOMATION / WINDOW_AUTOMATION when enabled with feature flag set.

### Phase 9 — Items 4–8 measurable verification

```
python scripts/verify_items_4_to_8.py
```

Capture concrete deltas from script output (token reduction, accuracy lift, abort timing).

### Phase 10 — Real-API sparing smoke

**Strict budget:**
- Brave: ≤2 queries (1 to verify happy path, 1 cache-hit re-query).
- Jina: ≤1 fetch.
- Claude Code: ≤1 minimal task ("create hello.py with current ISO date").

Don't trigger any heavy automation. ComfyUI / mobile node skipped.

### Phase 11 — Fault injection

- Brave 401 (bad key) → breaker trips → in-character voice msg.
- Qdrant disabled mid-session → memory absent, voice keeps working.
- LLM cancel mid-stream → clean termination.
- subprocess timeout in DirectClaudeCodeBridge → typed error → audit.
- File system error in WorkspaceWriter → lockfile timeout returns WriteResult.error.

### Phase 12 — Bug fix iteration

For each issue found:
1. Reproduce on a small repro.
2. Root-cause via logs + code path analysis.
3. Implement minimal fix preserving constraints (voice baseline, no paid-API sprawl).
4. Add regression test.
5. Run full sweep again; confirm green.
6. Update `docs/codebase_structure.md` per binding maintenance contract.
7. Capture in report.

### Phase 13 — Final voice-baseline regression check

- Repeat Phase 2 measurement.
- Compare against Phase 2 results AND original 79 ms / 7913 MB targets.
- **Gate:** no regression on either target.

### Phase 14 — Comprehensive metrics report

Generate `docs/comprehensive_test_report.md` with the massive metrics
table the user requested. Categories:

- **Pre-flight + state**
- **Test pass/fail counts** (per directory, per file for top-N)
- **Voice latency** (TTFT / TTFA distributions, percentiles, range)
- **VRAM** (idle / per-subsystem / peak)
- **Quality / accuracy** (classifier accuracy, gate accuracy, addressing accuracy)
- **Throughput** (Whisper / LLM / Piper / Qdrant write)
- **Resilience** (breaker trip + recovery times)
- **Items 4–8 deltas** (token reduction, accuracy lift, abort timing)
- **Coding subsystem** (verification check distribution; projection budget headroom)
- **Web search** (preflight accuracy; cache hit rate; per-query latency)
- **OpenClaw bridge** (fail-open coverage; gating-regression test pass/fail)
- **Resource usage** (CPU peak; disk I/O)
- **Bug findings + fixes**

## Out of scope (interactive / external)

Documented as user-led; not run autonomously:
- 16-step real-stack smoke test ([smoke_test.md](smoke_test.md))
- Telegram bot setup
- ComfyUI install + media generation live test
- Mobile node pairing
- Stage E voice character A/B (already approved 2026-05-08)
- OpenClaw Gateway start (no gateway.cmd run; bridge tests use mock)

## Reporting cadence

Each phase emits a section in `docs/comprehensive_test_report.md`
with: gate status, command run, raw output (or summary), metrics
captured. Bug findings get their own section with reproduction +
fix + test added.
