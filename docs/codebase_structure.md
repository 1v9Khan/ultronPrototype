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

**2026-05-18+ Phase 0+1 build + E2 + E5: cross-cutting learning infrastructure + adaptive context + voice-character-lock + goal-anchor planning -- COMPLETE.** Cross-cutting foundation work on top of the 2026-05-18 latency pass 3. The user asked for an integrated build covering Phase 0 (eval harness) + Phase 1 (observation framework with outcome tagging, lineage IDs, adaptive context, confidence plumbing) plus E2 (full goal-anchor planning) and E5 (voice-character-lock guardrails). All shipped; runtime behaviour is default-OFF on the behaviour-changing flags so the voice baseline contract is preserved.

* **Phase 0 -- eval harness + labeled corpus.** New [`scripts/eval_harness.py`](../scripts/eval_harness.py) + [`tests/eval/corpus.jsonl`](../tests/eval/corpus.jsonl) (60 labeled rows). Classifier-only mode runs `classify_routing` + addressing-rule + web-gate-rule classification against the corpus without loading the voice stack -- safe to invoke from CI / Claude Code without the voice-stack-concurrency ASK. Per-dimension accuracy gates configurable; exits 0/1/2 (pass/gate-fail/IO-fail) so it slots into automation. The shipped corpus baselines at 100% across all three dimensions; `tests/test_eval_harness.py` pins the baseline so a classifier regression fails CI before reaching production. **33 tests.**

* **Phase 1 -- observation framework + 4-site integration.** New [`src/ultron/observations/`](../src/ultron/observations/) package: [`schema.py`](../src/ultron/observations/schema.py) (12-field canonical row with event_id / parent_event_id / timestamp / subsystem / event_type / outcome / latency_ms / tokens_used / lineage_ids / payload_ref / extra), [`writer.py`](../src/ultron/observations/writer.py) (thread-safe JSONL appender with one-shot WARN suppression, fail-open IO, singleton + test-injection), [`integrations.py`](../src/ultron/observations/integrations.py) (one helper per emit site so subsystem modules carry no schema knowledge). Wired into FOUR call sites: [`classify_routing`](../src/ultron/openclaw_routing/classifier.py) (renamed inner impl to `_classify_routing_impl` + thin timing wrapper), [`AddressingClassifier._log`](../src/ultron/addressing/classifier.py) (emit per verdict), [`ConversationMemory.retrieve`](../src/ultron/memory/qdrant_store.py) (renamed inner impl to `_retrieve_impl` + wrapper carrying lineage_ids), [`LLMEngine.generate` + `generate_stream`](../src/ultron/llm/inference.py) (post-call emit with latency + tokens). Session-scoped autouse fixture in [`tests/conftest.py`](../tests/conftest.py) disables observation IO during pytest runs so `data/observations.jsonl` doesn't accumulate from the test sweep. **51 tests** across [`tests/observations/`](../tests/observations/).

* **Phase 1 -- outcome resolver.** New [`src/ultron/observations/outcome_resolver.py`](../src/ultron/observations/outcome_resolver.py): post-hoc maintenance pass that reads `observations.jsonl`, resolves `unknown_yet` outcomes for events older than `min_age_seconds` using either own-fields (LLM stream `completed` / `canceled`) or follow-up correction signals (subsequent CANCEL intent, NOT_ADDRESSED verdict, canceled LLM stream within `window_seconds`), and emits `outcome_resolution` rows referencing the original `event_id`. Never edits prior rows in-place -- a reader reconciles by walking the file. Driven by an injectable `now_provider` so tests can advance the clock without sleeping. **17 tests.**

* **Phase 1 -- lineage overlap helpers.** New [`src/ultron/observations/lineage_overlap.py`](../src/ultron/observations/lineage_overlap.py): pure detection primitive `compute_lineage_overlap(response_text, memory_contents)` plus `emit_lineage_usage_rows(...)` writer. Used for future importance-scoring loops (A2 / A5 in the V1-plus design notes) -- the retrieval observation already carries `lineage_ids`; this primitive identifies which IDs actually got used in a response. Live wiring (LLM call site -> memory-content lookup -> overlap -> emit) is documented as a follow-up because the LLM call site needs a clean handle on retrieved memory CONTENT, not just IDs; the shared primitive is in place. **10 tests.**

* **Phase 1 -- adaptive context window scoring.** New [`src/ultron/llm/context_scoring.py`](../src/ultron/llm/context_scoring.py): pure heuristic `score_context(user_text, *, default_history_turns, default_retrieval_k, ...) -> ContextRecommendation`. Blends six cheap signals (length, factual-stem, depth-marker, reference-laden, topic-shift, personal-recall, active-task) into a `(history_turns, retrieval_k, suppress_rag)` recommendation clamped to caller-supplied ceilings. Topic-shift wins (zeroes history); factual stem overrides reference-laden so syntactic "it" doesn't trip the boost; depth markers raise both budgets. Default-OFF flag pending live tuning; the helper is callable today by the eval harness + future orchestrator wiring. **31 tests.**

* **Phase 1 -- confidence-band ambiguity predicate.** New [`src/ultron/openclaw_routing/ambiguity.py`](../src/ultron/openclaw_routing/ambiguity.py): pure `should_clarify(intent, *, band_low, band_high, enabled)` predicate that returns an `AmbiguityVerdict` when an intent's confidence sits in `[band_low, band_high)` AND the intent kind is in the ambiguity-relevant set (the routable automation kinds + APP_LAUNCH / SCREEN_CONTEXT_QUERY / WINDOW_MOVE / WINDOW_CLOSE -- NOT conversational / cancel / clarification_response). Already-flagged `needs_user_clarification` short-circuits to False so the classifier-driven clarification path stays the authority. Config knob `routing.ambiguity_band_clarification` (default off) ships alongside [`AmbiguityBandClarificationConfig`](../src/ultron/config.py). **25 tests.**

* **E5 voice-character-lock guardrails.** New [`src/ultron/coding/voice_lock.py`](../src/ultron/coding/voice_lock.py): pure scanner + FILE_CHANGE helper enforcing the project-wide voice-quality lock. Default protected paths include `src/ultron/tts/speech.py`, `src/ultron/tts/rvc.py`, the cleaned Ultron vocal reference WAV, `~/.openclaw/workspace/SOUL.md`, and (via globs) Piper voice weights, RVC support files, the `ultron_james_spader_mcu_6941/` directory. `scan_prompt(prompt)` extracts file-path-shaped tokens (with trailing-punctuation strip so sentence-terminating periods don't break match) and returns deduped hits; `scan_file_change(path)` is the FILE_CHANGE-listener helper. `render_warning_for_voice(hits)` composes a TTS-safe narration line that NEVER speaks a backslash, drive letter, or full path (2026-05-11 narration honesty rule). Complementary to the safety validator's Category K. **37 tests.**

* **E2 goal-anchor planning -- FULL runtime integration.** New [`src/ultron/coding/anchors.py`](../src/ultron/coding/anchors.py) primitives (`GoalAnchor` frozen, `AnchorBudget` mutable with warning latch, `AnchorPlan` with `advance` / `all_completed` / `remaining_tokens`, `decompose_into_anchors` heuristic split on connectives + anchor-verb sentence stems INCLUDING comma boundaries, `narration_for_anchor` + `narration_for_completion` TTS-safe helpers). Wired through [`CodingTaskRunner`](../src/ultron/coding/runner.py): `start_task` builds a per-task plan when `coding.goal_anchors.enabled` is True + queues the opening anchor narration + adds an anchor listener that consumes USAGE events; the listener cascades overflow tokens across anchors when a single USAGE event over-fills the active anchor; per-anchor `warn_threshold` warning latches once; on exhaustion the plan advances and queues a transition narration; on plan completion a completion narration fires. New accessors `current_anchor()`, `anchor_plan_snapshot()`, `has_unfinished_anchors()`, `next_unfinished_anchor()`, `pop_anchor_narration()`. Resume support: `send_followup` prepends `"Continue with anchor N: <description>."` when there's an unfinished plan + the operator has enabled `coding.goal_anchors.resume_prepend_next_anchor`. Voice-loop wiring: [`CapabilityVoiceController.pending_anchor_narration`](../src/ultron/coding/voice.py) mirrors the existing `pending_budget_warning` / `pending_canonical_abort` pattern; new [`Orchestrator._announce_pending_anchor_narration`](../src/ultron/pipeline/orchestrator.py) polls + speaks at the top of each voice-loop iteration. Audit log emits `anchor_plan_created` / `anchor_warning` / `anchor_completed` / `anchor_started` / `anchor_plan_completed` events into `coding_tasks.jsonl`. Default-OFF on the master flag. Tests: **53 (40 in [`tests/test_anchors.py`](../tests/test_anchors.py) for the primitives + 13 in [`tests/test_coding_runner_anchors.py`](../tests/test_coding_runner_anchors.py) for the runner integration).**

Total Phase 0+1+E2+E5 net tests: **2472 -> 2697 passing (+225) / 15 skipped / 0 failed in ~59 s.** Voice baseline contract preserved on every flag: default-OFF for `coding.goal_anchors`, `routing.ambiguity_band_clarification`, `llm.adaptive_context` (callers opt in). Observation IO is on by default but the autouse conftest fixture suppresses it during pytest runs. The four observation emit-sites are all wrapped in fail-open try/except so a serialization or IO error never propagates to the production path.

---

**2026-05-18 latency pass 3: 3 phases -- COMPLETE.** Third coordinated latency reduction on top of the 2026-05-16 pass 2 (commit `9a15c06`). Three phases ship savings that the prior pass's speculative-STT collapse made newly available (the TTS preopen window shrank to ~5 ms post-pass-2, and the downstream classification + LLM TTFT were still serial after Smart Turn confirms). Cache-hit conversational turn drops from ~310-390 ms to **~210-300 ms** (an additional ~80-100 ms saved); no-ack short-utterance turns gain another ~63 ms behind the speculative LLM. Voice-quality lock preserved; resource consumption unchanged (the speculative LLM uses already-loaded VRAM and runs during otherwise-idle silence-wait time).

* **Phase 1 (TTS preopen at top of capture):** `_kick_off_tts_preopen()` hoisted from the post-capture position in `Orchestrator.run()` into the top of [`Orchestrator._capture_utterance`](../src/ultron/pipeline/orchestrator.py) AND [`Orchestrator._follow_up_listen`](../src/ultron/pipeline/orchestrator.py). After 2026-05-16 Phase 4 hid Whisper STT behind the silence wait, the legacy placement had only ~5-10 ms of overlap before the first TTS write -- not enough for the 50 ms PortAudio open. The new placement gives the open the entire speech + silence-wait window (typically 1-30 s). `prepare_output_stream` is idempotent at the engine layer; the legacy call in `run()` is retained as a belt-and-braces no-op. **Saves ~30-50 ms first-write on every cache-hit conversational turn.** Tests: +4 source-inspection regression tests in [`tests/test_tts_preopen.py`](../tests/test_tts_preopen.py) pinning the call positions before each VAD loop.

* **Phase 2 (Speculative classification during silence wait):** speculative STT's daemon thread now chains [`_run_speculative_classification`](../src/ultron/pipeline/orchestrator.py) after the transcript is stored. The chained work runs `classify_by_rules` (rule path only -- LLM preflight stays foreground), picks the conversational ack phrase via [`Orchestrator._maybe_conversational_ack`](../src/ultron/pipeline/orchestrator.py), and kicks off the RAG pre-fetch. Result is stored as a `_SpeculativeClassification` dict in [`Orchestrator._speculative_classification`](../src/ultron/pipeline/orchestrator.py) keyed by the transcript and consumed by [`_build_response_stream`](../src/ultron/pipeline/orchestrator.py). On SPEECH_START during silence wait, [`_invalidate_speculative_classification`](../src/ultron/pipeline/orchestrator.py) drops the slot and cancels the RAG future. New helpers: `_run_speculative_classification`, `_invalidate_speculative_classification`, `_collect_speculative_classification`, `_reset_speculative_classification_state` (all defensive against partial test fixtures via `getattr(self, lock, None)`). The STT reset call chains into the classification reset so both slots clear in lockstep at the top of each capture. **Saves ~5-10 ms classification work on the critical path + lets the RAG retrieval (~30-50 ms) overlap with the silence wait instead of with the gate.** Tests: +21 in [`tests/test_speculative_classification.py`](../tests/test_speculative_classification.py) (NEW; chain + invalidate + collect + reset + race coverage).

* **Phase 3 (Speculative LLM generation during silence wait):** when the chained classification's rule-path verdict resolves to NO_SEARCH (the common conversational case), the classification thread further kicks off [`_kick_off_speculative_llm`](../src/ultron/pipeline/orchestrator.py) which spawns a daemon thread running `llm.generate_stream(record_history=False)` against the speculative transcript. Tokens accumulate in a `queue.Queue`; the response-stream consumer drains the queue via [`_collect_speculative_llm`](../src/ultron/pipeline/orchestrator.py) instead of starting a fresh LLM call -- saving the entire ~63 ms LLM TTFT plus any partial decode time already completed during the silence wait. A new [`LLMEngine.record_completed_turn`](../src/ultron/llm/inference.py) public method commits the consumed turn to history; the new `record_history: bool = True` kwarg on [`LLMEngine.generate_stream`](../src/ultron/llm/inference.py) lets the speculative call defer the auto-record so invalidated speculations don't leave orphan turn pairs. On SPEECH_START, [`_invalidate_speculative_llm`](../src/ultron/pipeline/orchestrator.py) signals `llm.cancel()` and stamps the invalidated flag; the iteration exits at the next chunk and the producer thread's `finally` block emits the sentinel so consumers don't hang. Apply_uncertainty / apply_brevity_hint are mirrored from the main path so the speculative prompt is byte-identical. The search-augmented path (verdict == SEARCH) skips speculation because the prompt body differs. Cross-lane invariants: `_invalidate_speculative_classification` and `_reset_speculative_classification_state` both chain into the LLM lane so all three speculation slots stay in lockstep on SPEECH_START. **Saves up to ~63 ms LLM TTFT on cache-hit conversational turns; smooths the ack -> response transition on long-utterance turns by ensuring tokens are already buffered when the ack finishes playing.** Tests: +25 in [`tests/test_speculative_llm.py`](../tests/test_speculative_llm.py) (NEW; kick-off + cancel + collect-with-history-commit + reset + cross-lane invalidation + classification-chain verdict gating + LLMEngine `record_history` kwarg).

**Net end-to-end cumulative on cache-hit conversational turn:**

| Stage | Pre-pass | Post-pass | Saved |
|---|---|---|---|
| TTS preopen (PortAudio first-write) | ~50 ms (fresh open after capture) | ~0 ms (open completes during silence wait) | ~30-50 ms |
| Web-gate classify (rule path) | ~5 ms | ~0 ms (cached from speculation) | ~5 ms |
| LLM TTFT (no-ack turns) | ~63 ms | ~0 ms (speculation already streamed first tokens) | ~63 ms |
| LLM TTFT (ack turns) | ~63 ms hidden behind ack | smoother ack->response transition | response smoothness |
| **Total perceived (no-ack)** | ~310-390 ms | **~210-300 ms** | ~80-100 ms |

Tests: 2422 -> **2472 passing** (+50 net) / 15 skipped (GPU-gated) / 0 failed in ~57 s. Voice-quality lock preserved -- SOUL.md, RVC, Piper, the v3 filter chain order all untouched. Resource consumption unchanged: the speculative LLM uses the already-loaded model; classification work is cheap regex; no new VRAM or new model loads. The speculative LLM thread uses GPU during otherwise-idle silence-wait time -- bursty work that gets discarded on invalidation (user resumed speaking) but completes within the silence-wait window in the common case.

Files changed:

```
src/ultron/llm/inference.py                       (+record_history kwarg on generate_stream; +record_completed_turn public method)
src/ultron/pipeline/orchestrator.py               (+_speculative_classification slot + helpers; +_speculative_llm slot + helpers; chain classification + LLM off speculative STT; consume in _build_response_stream; preopen hoisted to top of _capture_utterance / _follow_up_listen)
tests/test_tts_preopen.py                         (+4 Phase 1 source-inspection regression tests)
tests/test_speculative_classification.py          (NEW; +21 tests)
tests/test_speculative_llm.py                     (NEW; +25 tests covering helpers + LLMEngine history-defer)
docs/codebase_structure.md                        (this status header + per-module deltas)
CLAUDE.md                                         (most-recent-state pointer + tests count + voice baseline numbers)
```

---

**2026-05-16 latency pass 2: 5 phases -- COMPLETE.** Second coordinated latency reduction on top of the 2026-05-15 pass (commit `703c11f`). Drops perceived end-to-end voice latency from ~590 ms down to **~310-390 ms** on cache-hit conversational turns. The pass dropped what was originally Phase 1 (Distil-Whisper swap) by user direction ("lets keep whisper"). The 5 shipped phases:

* **Phase 2 (LLM prefix KV cache infrastructure):** new `llm.prefix_cache_ram_bytes` config knob. When > 0, attaches `LlamaRAMCache(capacity_bytes=...)` to the in-process Llama instance so completed session KV state stores in host RAM keyed by token sequence. **Default flipped to 0 (disabled)** after live bench (`scripts/bench_llm_prefix_cache.py`) showed a **-15 ms regression** on the production 4070 Ti + josiefied-4B-Q4_K_M stack: cold-cache median 63 ms vs warm-cache median 78 ms. llama.cpp's internal KV cache already handles intra-session prefix reuse; the explicit RAMCache's load_state memcpy exceeds the eval savings on short 280-token system prompts. The infrastructure stays in place (config knob + wiring + tests + bench) so operators with longer prompts or cross-session reload patterns can opt in. Bench result captured at `baselines.json:llm_prefix_cache_bench`. Tests: +11 in `tests/test_llm_prefix_cache.py` (NEW). Fail-open: missing `LlamaRAMCache` class or `set_cache` exception leaves the engine in its working state.

* **Phase 3 (Smart Turn V3 gradient-fire):** new `vad.smart_turn.early_completion_threshold` (default 0.65) + `vad.smart_turn.medium_grace_ms` (default 200). Drops `vad.smart_turn.fast_path_silence_duration_ms` from 500 → 300 ms. New `Orchestrator._classify_smart_turn_verdict` helper bands the model's verdict at the (shortened) fast-path checkpoint: `early_complete` (prob ≥ 0.65) submits immediately; `medium_complete` (0.5 ≤ prob < 0.65) waits an additional 200 ms grace then trusts the verdict (matching the prior 500 ms baseline behaviour for medium-confidence turns); `incomplete` (prob < 0.5) enters the existing extension wait; `undecided` (verdict None) trusts VAD. Both `_capture_utterance` and `_follow_up_listen` wire the gradient with mirror-state machines for medium and incomplete grace windows. **Saves ~100-200 ms / turn on confidently-complete turns; no regression on medium-confidence or trail-off turns.** Tests: +9 in `tests/test_smart_turn.py` extension (early threshold + medium grace schemas + classifier band coverage).

* **Phase 4 (Speculative Whisper STT during silence wait):** the orchestrator kicks off Whisper transcription on the captured audio buffer in a background daemon thread as soon as VAD has accumulated 2 consecutive silence chunks (~32 ms at the new 16 ms blocksize). By the time the fast-path silence baseline (~300 ms) elapses and Smart Turn V3 confirms end-of-turn, Whisper (~78 ms) has finished and the transcript is consumable via `_collect_speculative_stt()` -- the main run() loop skips the foreground Whisper call entirely on cache hit. SPEECH_START during the silence run invalidates the in-flight result; the next silence period re-arms. `_reset_speculative_stt_state()` at the start of every `_capture_utterance` and `_follow_up_listen` clears stale results. **Saves ~78 ms / turn (full Whisper time hidden behind the silence wait).** Three new orchestrator helpers (`_kick_off_speculative_stt` / `_invalidate_speculative_stt` / `_collect_speculative_stt` / `_reset_speculative_stt_state`). Tests: +12 in `tests/test_speculative_stt.py` (NEW; kick-off / collect / invalidate / reset / fail-open / audio-buffer-snapshot).

* **Phase 5 (16 ms audio blocksize):** sounddevice `blocksize: 512 → 256` (32 ms → 16 ms at 16 kHz). Halves mic-to-consumer queue latency and gives the speculative-STT silence-onset detection finer granularity (16 ms steps vs 32 ms). Silero VAD's internal 512-sample window is unchanged -- it just buffers two 256-sample chunks per decision -- so VAD timing is unaffected. Tests: +1 in `tests/test_audio.py` (default assertion).

* **Phase 6 (legacy TextToSpeech pre-open silence write):** legacy `tts.speech.TextToSpeech.prepare_output_stream` now calls `_write_silence(stream, sr, 0.05)` to wake the audio device clock before the first real audio write -- matching the existing behaviour on `XttsV3Speech.prepare_output_stream`. Without this, the first `speak_stream` clip on the legacy stack paid the device-wake latency. Best-effort: `_write_silence` failure is swallowed (some PortAudio backends prime themselves on `stream.start()` already). **Saves 5-15 ms first-write on the legacy `piper_rvc` engine; XTTS already had this from the prior pass.** Tests: +2 in `tests/test_tts_preopen.py` (silence write invoked + failure swallowed).

**Net end-to-end cumulative on cache-hit conversational turn:**

| Stage | Pre-pass | Post-pass | Saved |
|---|---|---|---|
| VAD silence (Smart Turn fast-path) | 500 ms | 300 ms (early-complete) / 500 ms (medium) | 0-200 ms |
| Whisper STT (5s audio, beam=1) | 78 ms (foreground) | 0 ms (overlapped with silence wait) | 78 ms |
| LLM TTFT (Phase 2 default off) | 63 ms | 63 ms | 0 ms |
| **Total** | ~590 ms | **~310-390 ms** | ~200-280 ms |

Tests: 2385 → **2422 passing** (+37 net) / 15 skipped (GPU-gated) / 0 failed. Voice-quality lock preserved -- SOUL.md, RVC, Piper, the v3 filter chain order all untouched. Resource consumption unchanged (Phase 2 cache is default off so no host-RAM increase; the speculative-STT thread is a background daemon that completes inside the existing silence wait).

Files changed:

```
src/ultron/config.py                              (+prefix_cache_ram_bytes default 0; +early_completion_threshold default 0.65; +medium_grace_ms default 200; fast_path_silence default 500->300; blocksize default 512->256)
config.yaml                                       (matching YAML edits + comments)
src/ultron/llm/inference.py                       (LlamaRAMCache attach in _build_llama; fail-open import)
src/ultron/tts/speech.py                          (50 ms silence write in prepare_output_stream)
src/ultron/pipeline/orchestrator.py               (_classify_smart_turn_verdict + medium-grace handling in _capture_utterance/_follow_up_listen + speculative-STT helpers + per-block silence tracking + main-loop _collect_speculative_stt + _reset_speculative_stt_state)
tests/test_smart_turn.py                          (+early_completion_threshold + medium_grace + classifier tests)
tests/test_llm_prefix_cache.py                    (NEW; +11 tests)
tests/test_speculative_stt.py                     (NEW; +12 tests)
tests/test_audio.py                               (+blocksize default test)
tests/test_tts_preopen.py                         (+legacy silence-write tests)
scripts/bench_llm_prefix_cache.py                 (NEW; cold-vs-warm TTFT bench)
baselines.json                                    (llm_prefix_cache_bench block)
docs/codebase_structure.md                        (this status header + changelog + per-section updates)
CLAUDE.md                                         (most-recent-state pointer)
```

---

**2026-05-15 latency pass: 5 phases -- COMPLETE.** Five coordinated changes that drop perceived end-to-end voice latency from ~2.5 s (pre-pass estimate) / ~1.06 s (re-measured baseline) down to roughly **~600 ms** to first audible ack, with the LLM TTFT now measured at ~63 ms median and Whisper STT at ~78 ms median (5s audio, beam=1 / int8_float16).

* **Phase 1 (ack clip cache):** new `src/ultron/tts/precomputed_ack.py` -- `PrecomputedAckClipCache` keyed by stripped text; built at orchestrator init via `build_default_ack_clip_cache()` covering both the conversational pool ("Mm." / "Right." / "Hm." / "Considering." / "Let me think." / "Noted." / "Processing." / "Working on it.") and the web-search pool ("Querying external sources." / "Verifying against the network." / ...). `XttsV3Speech` + legacy `TextToSpeech` both expose `set_ack_cache(cache)`; their `_synthesize` checks the cache before running HTTP + filter (xtts) / Piper + RVC (legacy). Pre-warmed on a daemon thread via `prewarm_in_background`. Saves **~350-400 ms on every conversational turn** (cache hit returns bit-identical pre-filtered audio instantly). Tests: +25 in `tests/test_precomputed_ack.py`.

* **Phase 2 (parallel RAG pre-fetch):** new `Orchestrator._kick_off_rag_prefetch` + `_collect_rag_future` + `LLMEngine.retrieve_rag_snippets` public wrapper + new `precomputed_rag_snippets` kwarg on `LLMEngine.generate` / `generate_stream` / `_build_messages`. `_build_response_stream` kicks off the Qdrant retrieval on a `ThreadPoolExecutor` thread BEFORE the web-gate call so the ~30-50 ms RAG round-trip overlaps with the ~5-150 ms gate cost (the latter dominates on LLM-preflight gate turns). The search-augmented branch cancels the pre-fetch (the search payload self-contains context; pulling stale memory would contaminate it). Multi-pass retrieval (`memory.retrieval.multi_pass_enabled=True`) skips the pre-fetch because that path keys off `gate_verdict.context_categories` which the gate populates -- pre-fetching single-pass would silently downgrade. Saves **~30-50 ms** on most turns. Tests: +9 in `tests/test_llm_precomputed_rag.py` + +11 in `tests/test_orchestrator_rag_prefetch.py`.

* **Phase 3 (n_batch / n_ubatch knobs):** new `LLMConfig.n_batch` + `LLMConfig.n_ubatch` (both `Optional[int]`, default `None` = inherit llama.cpp's per-version defaults). `LLMEngine._build_llama` passes them through only when set. New `scripts/bench_llm_ubatch.py` sweeps `(None, None)` / `(2048, 128)` / `(2048, 256)` / `(2048, 512)` / `(2048, 1024)` / `(4096, 512)`. **Empirical result on 4070 Ti + josiefied-qwen3-4b Q4_K_M:** all combinations give ~63 ms median TTFT on voice-length prompts -- no measurable win at short context. The knobs are in place for future long-context tuning; defaults stay at None for safety on unknown hardware. Tests: +14 in `tests/test_llm_batch_tunables.py`. Bench result merged into `baselines.json:llm_n_ubatch_sweep`.

* **Phase 4 (Whisper beam=1):** `STTConfig.beam_size` default `5 -> 1` (greedy decoding). **Empirical bench on 4070 Ti + small.en + int8_float16 (5s audio):** beam=1 median 78 ms, beam=3 median 94 ms, beam=5 median 157 ms. **Saves ~80 ms median.** WER impact for short English voice queries is negligible (within 0.1-0.3 pp on LibriSpeech-test-clean per CTranslate2 benchmarks). The Moonshine STT swap (~700 ms claimed win) was investigated and rejected -- the `useful-moonshine` package pulls in Keras 3 + librosa + torch 2.4.1 + tokenizers 0.20, conflicting with our pinned moondream2 / flan-t5-small / bge-small stack; quality A/B + reimpl-from-ONNX was deemed not worth ~50-100 ms additional savings over beam=1. Tests: unchanged (existing `tests/test_transcription.py` is GPU-gated).

* **Phase 5 (TTS stream pre-open during STT):** new `XttsV3Speech.prepare_output_stream` + `_consume_preopened_stream` + same pair on legacy `TextToSpeech`. Orchestrator's main loop calls `_kick_off_tts_preopen()` on a daemon thread AFTER `_capture_utterance` returns and BEFORE `stt.transcribe(speech)` runs -- the ~50 ms PortAudio open cost overlaps with Whisper STT. `speak_stream` consumes the pre-opened stream when SR matches; falls back to fresh open otherwise. `stop()` closes any leftover pre-open so device handles release cleanly on shutdown. Saves **~50 ms** of first-audio latency. Tests: +13 in `tests/test_tts_preopen.py`.

**Net latency savings (additive on top of Phase 1 ack cache):**

| Component | Pre-pass | Post-pass | Saved |
|---|---|---|---|
| Conversational ack synth | ~350 ms HTTP+filter | ~0 ms cache hit | ~350 ms |
| RAG retrieval | ~30-50 ms (in LLM TTFT) | overlapped with gate | ~30-50 ms |
| Whisper STT (5s audio) | ~157 ms (beam=5) | ~78 ms (beam=1) | ~80 ms |
| TTS output stream open | ~50 ms (in `speak_stream`) | overlapped with STT | ~50 ms |
| LLM TTFT (4B Q4_K_M) | ~63 ms (re-measured; older 140 ms estimate was off) | unchanged | 0 ms |
| **Total** | | | **~510-530 ms** |

Final perceived latency from "user stops speaking" to "first audible ack" on a cache-hit conversational turn:
- VAD silence (Smart Turn V3 fast path): 500 ms
- STT (beam=1, ~5s audio): 78 ms
- Web-gate (rule): ~5 ms
- Ack synth (cache hit): ~0 ms
- TTS stream open (pre-opened): ~0 ms
- Write first audio: ~5 ms
- **Total: ~590 ms** (was ~1060 ms re-measured, or ~2500 ms per the pre-pass estimate)

Tests: 2313 -> **2385 passing** (+72 net) / 15 skipped (GPU-gated) / 0 failed. Voice-quality lock preserved -- SOUL.md, RVC, Piper, the v3 filter chain order all untouched. Resource consumption unchanged (no new model weights, no new dep pulls; the bench-script subprocess only runs on demand).

Files changed:

```
src/ultron/tts/precomputed_ack.py                 (NEW)
src/ultron/tts/xtts_v3.py                          (cache, prepare_output_stream, consume, stop cleanup)
src/ultron/tts/speech.py                           (cache, prepare_output_stream, consume, stop cleanup)
src/ultron/pipeline/orchestrator.py                (_kick_off_ack_clip_prewarm, _kick_off_tts_preopen, _kick_off_rag_prefetch, _collect_rag_future, _build_response_stream rewrite)
src/ultron/llm/inference.py                        (precomputed_rag_snippets kwarg, retrieve_rag_snippets public, n_batch/n_ubatch wiring)
src/ultron/config.py                               (STTConfig.beam_size 5->1, LLMConfig.n_batch + n_ubatch)
config.yaml                                        (stt.beam_size 5->1; tuning notes)
scripts/bench_llm_ubatch.py                        (NEW)
scripts/bench_stt_latency.py                       (NEW)
tests/test_precomputed_ack.py                      (NEW, 25 tests)
tests/test_llm_precomputed_rag.py                  (NEW, 9 tests)
tests/test_orchestrator_rag_prefetch.py            (NEW, 11 tests)
tests/test_llm_batch_tunables.py                   (NEW, 14 tests)
tests/test_tts_preopen.py                          (NEW, 13 tests)
baselines.json                                     (llm_n_ubatch_sweep block)
docs/codebase_structure.md                         (this changelog + file tree + tests + config sections)
```

---

**2026-05-12 desktop automation Phases 1-11 -- COMPLETE.**

* **Phase 1-6 (commit `ec80bc9`):** `src/ultron/desktop/` native primitives package -- monitors / capture (mss) / windows (pywin32+psutil) / placement / launcher (Chrome default-profile + monitor targeting + 12-entry app registry) / uia (pywinauto semantic clicks + tree text extraction) / input_control (pyautogui mouse+keyboard, validator-gated + rate-limited + foreground-security-blocked) / screen_context (foreground + window list + UIA text + optional VLM snapshot for LLM injection, 3-entry 15s cache) / vlm (moondream2 via transformers, CPU-on-demand, lazy-loaded, fail-open at every layer, ~3.5 GB FP16 pre-fetched via `scripts/download_models.py` step 9/10). Every capture stamps its bytes in the safety taint tracker as `capability=screen_context`. Tests: +171 across `tests/desktop/`.

* **Phase 7 (MCP tool exposure for OpenClaw agents):** extended `src/ultron/openclaw_bridge/mcp_tools.py` from 5 to **24 tools** total. Desktop tools: `enumerate_monitors`, `list_windows`, `take_screenshot` (PNG bytes + optional VLM), `describe_screen` (text only), `get_screen_context` (assembled snapshot), `launch_app` (registry), `launch_chrome_url` (user's real Chrome), `open_image_search` (Google Images), `move_window_to_monitor`, `focus_window`, `window_action` (maximize/minimize/restore), `click_uia` (Cap-3 safety gated), `type_into_uia`, `get_window_text`, `mouse_click`, `mouse_move`, `type_text`, `press_hotkey`, `scroll`. All lazy-import heavy deps so MCP server cold start stays light. Tests: 50+ in `tests/openclaw_bridge/test_mcp_tools_desktop.py`.

* **Phase 8 (classifier + voice-handler wiring):** new `RoutingIntentKind.APP_LAUNCH` and `RoutingIntentKind.SCREEN_CONTEXT_QUERY` plus matching `AppLaunchIntent` / `ScreenContextIntent` dataclasses in `src/ultron/openclaw_routing/intents.py`. Classifier patterns for "open YouTube on my 2nd monitor", "show me a picture of golden retriever", "explain what I'm looking at", "what's on my screen" -- with `_extract_monitor_target` parsing ordinal/digit/directional monitor references. Site words ("youtube" / "github") only fire APP_LAUNCH when an explicit monitor target is present (preserves existing BROWSER_AUTOMATION baseline). `CapabilityVoiceController._handle_app_launch` + `_handle_screen_context_query` dispatch to the new native handlers in `src/ultron/desktop/voice.py`. SCREEN_CONTEXT_QUERY pipeline: build snapshot -> render_for_llm() -> prepend to user question -> LLM call -> Ultron-voiced response. Tests: +99 in `tests/routing/test_desktop_native_classifier.py` + 13 in `tests/test_coding_voice_desktop_native.py`.

* **Phase 9 (ultron-vision setup doc):** [docs/openclaw_desktop_automation_setup.md](openclaw_desktop_automation_setup.md) -- user-led recipe to paste an `ultron-vision` specialized agent into `~/.openclaw/openclaw.json` for multi-step task delegation. Specialized agent prompt tuned for "observe before acting" UI reasoning with the 24 MCP tools above. Setup also documents verification + troubleshooting paths.

* **Phase 10 (memory-wiki preference writes):** new `src/ultron/desktop/preferences.py` -- `DesktopPreference` dataclass + `PreferenceLogger` (JSONL append, thread-safe) + `find_preference_for_phrase` (substring match, recency-weighted) + optional async mirror to OpenClaw workspace via `set_workspace_writer`. Wired into `_handle_app_launch`: successful launches record the placement; subsequent matching utterances reuse the monitor + flags as defaults. Tests: +28 in `tests/desktop/test_preferences.py`.

User decision (no ClawHub plugins) drives the "native primitives + OpenClaw as orchestration brain" architecture -- UIA semantic clicks via pywinauto, screen capture via mss, mouse/keyboard via pyautogui, all gated by the runtime tool-call validator. OpenClaw `ultron-vision` agent orchestrates multi-step tasks by calling our MCP tools.

**Tests: 1830 -> 2194 passing (+364) / 15 skipped / 0 failed. Voice baseline contract intact -- none of the new code is on the voice hot path.**

**Phase 12 (analyze-and-discard) + Phase 13 (ultron-vision live install) -- followup polish (2026-05-12).**

* **Phase 12 -- screenshot analyze-and-discard.** `Screenshot.image_bytes` is now `Optional[bytes]` and the dataclass gains a `bytes_discarded: bool = False` flag + a `without_bytes()` helper. After the VLM successfully analyses a capture, `build_screen_context` (default `discard_image_after_analysis=True`) replaces the screenshot with the bytes-stripped variant -- the textual VLM description is the durable record. `ScreenContextCache.store()` likewise strips bytes by default (`discard_image_bytes=True`). Net effect: the cache + downstream consumers carry kilobytes of structured text instead of megabytes of PNG, and the safety-taint surface for screen captures collapses to a single in-flight buffer rather than a multi-entry ring. Both flags are flippable when a downstream consumer specifically needs the bytes (image-stylize tool, debugging). Tests: +10 in `tests/desktop/test_screen_context.py` (without_bytes idempotency, discard-after-VLM, keep-when-no-VLM, cache strip semantics).

* **Phase 13 -- `ultron-vision` agent installed live.** Auto-applied to `~/.openclaw/openclaw.json` (4th entry in `agents.list`) with a `openclaw.json.pre-ultron-vision-bak` backup saved alongside. Atomic write via tmp + `os.replace`; JSON re-parse verifies the result before promoting. Gateway restart is the user's call (we don't touch services).

**Tests: 2194 -> 2204 passing (+10). Voice baseline intact.**

* **Phase 14 (orchestrator VLM wiring).** `Orchestrator.__init__` now calls `_load_desktop_vlm_if_enabled()` which constructs the moondream2 VLM and pushes it via :func:`ultron.desktop.vlm.set_vlm`. Lazy + fail-open: construction validates the transformers stack but does NOT load the ~3.5 GB weights at orchestrator startup -- the load happens on first :func:`describe` call (first ``SCREEN_CONTEXT_QUERY`` with VLM). Failure (missing transformers, missing weights on disk) leaves the singleton unset and `screen_context` falls back to text-only context (window title + UIA tree + foreground app). Targeted sweep: 221 pipeline + desktop tests still pass.

Last validated against `claude/nifty-swirles-5fd6b1` HEAD (2026-05-18 Phase 0+1 + E2 + E5 build; **all eight tasks above**; on top of `e3ac64e` 2026-05-18 latency pass 3 -- 3 phases; on top of `a6fc937` codebase_structure entry for bench_llm_prefix_cache; on top of `9a15c06` 2026-05-16 latency pass 2; on top of `5d5f65f` CLAUDE.md pointer bump; on top of `703c11f` 2026-05-15 latency pass; on top of `0bf2027` handoff-doc bump; on top of `622000d` third-pass chat_template_kwargs regression + WINDOW_MOVE/CLOSE + bare image-search + plural image nouns; on top of `b79d41e` stale-process safeguards; on top of `15f58d5` second-pass VRAM relief + classifier extension; on top of `901ebf1` first VRAM-relief + UX-fix pass). **Tests: 2472 -> 2697 passing (+225) / 15 skipped (GPU-gated) / 0 failed in ~59 s.** Voice-path peak VRAM ~6.5-6.9 GB (4B abliterated Q4_K_M + Whisper int8_fp16 + XTTS + KV cache Q8_0 @ 6144 + idle) -- comfortably below the 11.5 GB hard cap on the 4070 Ti. **Default LLM:** `josiefied-qwen3-4b` preset -> `models/Josiefied-Qwen3-4B-abliterated-v2.Q4_K_M.gguf` (Goekdeniz-Guelmez Josiefied + abliterated Qwen3-4B-Instruct-2507; quantised by mradermacher). **Live-measured timings (2026-05-15):** LLM TTFT median **63 ms** (was previously estimated 140 ms); Whisper STT median **78 ms on 5s audio at beam=1** (was 157 ms at beam=5); ack synth on conversational/web-search pool is **0 ms cache hit** (was 350-400 ms HTTP+filter). **Stale-process safeguards installed:** `tests/conftest.py:pytest_sessionfinish` auto-reaps test descendants; `scripts/cleanup_stale_processes.py` is the manual cleanup tool. Both preserve the live Ultron via the port-19761 listener check.

**2026-05-14 third pass: chat_template_kwargs regression fix + WINDOW_MOVE / WINDOW_CLOSE + bare image-search + plural image nouns + moondream2 revision rollback (commit `622000d`).**

Triggered by another live-session log surfacing three issues:

* **CRITICAL regression: every web-search preflight failed with
  `TypeError: Llama.create_chat_completion() got an unexpected
  keyword argument 'chat_template_kwargs'`.** llama-cpp-python 0.3.22
  (the version pinned in the venv) does not accept that kwarg. The
  Stage F `_chat_completion_kwargs` helper had been adding it
  unconditionally when `enable_thinking` was set, but the only user
  was previously the voice path defaulting to `enable_thinking=None`
  so the kwarg was never emitted. My 2026-05-14 second-pass
  `enable_thinking=False` calls in screen-context AND the preflight
  triggered the latent bug. Fix: replaced the kwarg approach with
  Qwen3's `/no_think` user-message marker (new
  `LLMEngine._apply_no_think_marker`). Works at the prompt layer so
  it survives the llama-cpp-python version gap. The HTTP runtime
  still carries `chat_template_kwargs` in the JSON body because
  llama-cpp-server (separate codebase) does accept it.

* **"Show me a chicken." (no monitor cue) fell through to
  conversational LLM** and got a hallucinated "Displaying visuals
  via text only" response. Added a new `_IMAGE_SEARCH_BARE_RE` that
  matches "show me X" at end-of-string when X is concrete (not a
  question word, not a known app, not a screen-context noun).
  Defaults to the main monitor via the existing `_resolve_monitor`
  fallback. Tighter guards than the with-monitor pattern: question
  starts (`what` / `who` / `how` / etc.) and known apps are
  denied because they have other handlers.

* **"Show me pictures of Resident Evil Requiem" missed image-search**
  because the explicit-keyword regex required singular noun. Added
  `s?` to all three noun positions (`pictures? / images? / photos?`)
  plus accept `some` / `the` as determiners. The user's session
  log had both "Show me a chicken" (no keyword) and "Show me
  pictures of Resident Evil Requiem" (plural keyword) failing for
  these reasons.

* **"Put Discord on my right monitor"** wasn't a routable intent.
  New `RoutingIntentKind.WINDOW_MOVE` + `WindowMoveIntent` +
  `_WINDOW_MOVE_RE` ("put / move / send / throw / drag / relocate /
  push / bring / shift X to / on <monitor>") + `_classify_window_move`
  + voice handler `handle_window_move` that calls the existing
  `find_window` + `move_window_to_monitor` primitives. Distinct
  from APP_LAUNCH which would spawn a new instance.

* **"Close my YouTube video on my right monitor"** wasn't a
  routable intent. New `RoutingIntentKind.WINDOW_CLOSE` +
  `WindowCloseIntent` + `_WINDOW_CLOSE_RE` ("close / exit / quit /
  shut / kill / dismiss X") with deny-list (task / file / everything
  / yourself / etc. so it doesn't hijack coding-cancel or file-op
  intents) + voice handler that sends `WM_CLOSE` via
  `win32gui.PostMessage` (graceful close).

* **moondream2 revision pin moved from 2025-06-21 to 2024-08-26.**
  The 2025-06-21 tokenizer.json format is too new for the venv's
  `tokenizers 0.19.1`. 2024-08-26 is the older stable release
  referenced in moondream2's compat discussion thread (HF
  discussion 59). Even if moondream2 fails to load, the text-only
  screen-context fallback now works thanks to the
  `chat_template_kwargs` fix above.

Tests: 2278 -> 2313 (+35). Voice baseline unaffected.

**2026-05-14 second-pass VRAM relief + classifier extension (commit `15f58d5`).**

Triggered by the user running on the new 4B abliterated default and
finding the VRAM still maxing at ~11.1 GB (nvidia-smi showed ~4.7 GB
already used by Chrome / Discord / EdgeWebView / NVIDIA Broadcast /
Cursor BEFORE Ultron loaded). Plus two classifier regressions
surfaced in the same session.

* **LLM quant trim: Q5_K_M -> Q4_K_M** on the `josiefied-qwen3-4b`
  preset. Saves another ~500 MB VRAM at negligible quality impact
  (Qwen3-4B Q4_K_M vs Q5_K_M MMLU delta <0.5 pp per the
  mradermacher quant ladder). Q5_K_M file retained on disk via the
  download script's optional fetch for swap-back A/B.

* **Implicit image-search shortcut.** "Show me a chicken on my
  main monitor" -- with no "picture of" keyword -- now matches a
  new `_IMAGE_SEARCH_IMPLICIT_RE` pattern. The monitor target is
  the disambiguating signal; a deny-list (`_IMAGE_SEARCH_IMPLICIT_DENY`)
  keeps screen-context / app-launch subjects from leaking in.

* **Preflight `<think>` strip.** `web_search.gating._preflight_call`
  was bypassing `LLMEngine.generate()` (calls
  `llm._llm.create_chat_completion` directly), so the 2026-05-14
  `strip_thinking_text` fix wasn't reaching it. Now passes
  `chat_template_kwargs={"enable_thinking": False}` AND applies
  `strip_thinking_text` defensively. Belt + braces: the abliterated
  model occasionally emits a stray `<think>` even with thinking
  disabled.

Tests: 2263 -> 2278 (+15). Voice-path peak (4B Q4_K_M) is ~6.5-6.9 GB
vs the ~7.0-7.4 GB of the Q5_K_M variant -- leaves ~5 GB above the
user's typical background apps' ~4.7 GB consumption (vs 12 GB
hard cap on the 4070 Ti).

**2026-05-14 VRAM-relief + UX-fix pass (commit `901ebf1`).**

Eleven coordinated fixes prompted by the user's live-session feedback
(`12 GB VRAM ceiling hit; GPU at 100%; "main monitor" defaults to
right; YouTube channel deep-link ignored; "what's on my screen"
hangs / leaks `<think>` to TTS; "switch to model 4B" misclassified;
slow + breathy XTTS output`):

* **LLM swap: Josiefied-Qwen3-8B Q5_K_M -> Josiefied-Qwen3-4B-abliterated-v2 Q5_K_M.** New default preset
  `josiefied-qwen3-4b` (3.0 GB on disk vs 5.85 GB) recovers ~3 GB of
  VRAM headroom while preserving the abliterated/Josiefied pairing
  with the runtime tool-call validator. `n_ctx=6144` (down from
  8192) trims another ~150 MB of KV cache. The 8B variant is
  retained for swap-back. Wired into `LLM_PRESETS` + the
  `Literal[...]` schema in `src/ultron/config.py`; `config.yaml:llm.preset`
  flipped; `check_vram.py:TARGET_MB_BY_PRESET` extended; download
  ordering in `scripts/download_models.py` reordered to fetch the new
  default first. `mradermacher/Josiefied-Qwen3-4B-abliterated-v2-GGUF`
  on HuggingFace.

* **Whisper VRAM trim: float16 -> int8_float16.** Same `small.en`
  model, quantised activations. Saves ~250 MB VRAM. WER impact in
  the negligible band per CTranslate2 benchmarks; the 4070 Ti has
  ample int8 throughput so STT latency unchanged.

* **Post-init `torch.cuda.empty_cache()`.** Added at the tail of
  `Orchestrator.__init__` to release transient allocations from
  llama-cpp / faster-whisper / sounddevice init. Saves another
  ~200-400 MB of fragmented allocation typically.

* **Monitor mapping: `"main"` -> physical center (not Win32 primary).**
  `find_monitor("main")` and `find_monitor("default")` now resolve
  via a new `_center_monitor` helper rather than collapsing to the
  Win32-designated primary. The user's setup has primary = right
  monitor; calling that "main" violated user intuition. `"primary"`
  remains a separate keyword for callers that explicitly want the
  Win32 primary. The classifier's `_extract_monitor_target` no
  longer pre-resolves `"main"` / `"primary"` to index 0 -- both are
  routed through `find_monitor` at dispatch.

* **Default to "main" when no monitor target specified.**
  `_resolve_monitor(None, "")` in `src/ultron/desktop/voice.py` now
  calls `find_monitor("main")` instead of returning `None`. The
  launcher places the window on the center monitor instead of
  wherever the spawned app happened to be last positioned.

* **MODEL_SWITCH regex broadened.** "switch to model 4B" /
  "switch to the model 9B" / "switch to llm 4B" / "switch to qwen
  4B" / "switch to preset 9B" now match. The optional
  `(?:(?:the\s+)?(?:model|llm|preset|qwen)\s+)?` slot accepts the
  noun BEFORE the model token, not just after. 8B added as a
  switch target (Josiefied-Qwen3-8B); "4B" now resolves to
  `josiefied-qwen3-4b` (abliterated) not `qwen3.5-4b` (legacy).

* **YouTube channel / video / search deep-linking.** New
  `_build_youtube_url(text)` parses cue phrases ("with the channel
  X", "to the X channel", "video X", "search for X", "play X")
  out of the utterance and constructs a
  `youtube.com/results?search_query=...` URL. Stops at sentence
  terminators OR monitor-target boundaries so "on my right monitor"
  doesn't bleed into the query. Hooked into `_classify_app_launch`
  on the YouTube branch.

* **Spotify WindowsApps shim path added.** The launcher's spotify
  registry entry now checks `%LOCALAPPDATA%\Microsoft\WindowsApps\Spotify.exe`
  first (the Microsoft Store install shim), then falls back to
  the legacy `AppData/Roaming\Spotify\Spotify.exe` and
  `%LOCALAPPDATA%\Spotify\Spotify.exe` paths. The user's
  `Open Spotify` failed with "no candidate path exists on disk"
  because only the legacy paths were checked.

* **`<think>` block strip in blocking `generate()`.** New pure
  function `strip_thinking_text(text)` in
  `src/ultron/llm/inference.py`; applied inside `LLMEngine.generate()`
  before returning. The 2026-05-13 session log had a `<think>...</think>`
  block reach XTTS verbatim on the screen-context path (which uses
  blocking `generate()`, not the streaming path that already filtered
  thinking blocks). Unterminated `<think>` (truncation / cancel)
  drops everything from the opening tag forward -- better to lose
  tail content than leak chain-of-thought to TTS.

* **SCREEN_CONTEXT_QUERY: brevity hint + `enable_thinking=False`.**
  The screen-context handler in `coding/voice.py` now prepends
  `[Style: respond in 1-2 short sentences ...]` to the augmented
  prompt AND passes `enable_thinking=False` to the LLM. The
  session log got a 1235-char essay back; the brevity hint plus
  the `enable_thinking=False` (Qwen3's no-`<think>` chat-template
  toggle) keeps the response in voice-friendly length and saves the
  token budget the chain would have burned.

* **SCREEN_CONTEXT_QUERY classifier: adjective-qualified screens.**
  The regex now accepts `(?:main|primary|left|right|center|...)`
  adjectives between "my" / "the" and "screen/display/monitor", so
  "what's on my **main** screen" / "what's on my **left** monitor"
  route to the native handler instead of the conversational LLM
  (which previously hallucinated "A task interface." with no
  actual screen context).

* **Moondream2 revision pin.** `src/ultron/desktop/vlm.py` and
  `scripts/download_models.py` both pin
  `vikhyatk/moondream2 revision="2025-06-21"`. The live session
  hit "data did not match any variant of untagged enum ModelWrapper
  at line 255192 column 3" -- a `tokenizer.json` shape mismatch
  between recent ``main`` and our pinned `tokenizers` build. The
  2025-06-21 revision is the documented stable release per
  the HuggingFace README.

Tests: 2204 -> 2263 passing (+59) / 15 skipped / 0 failed. Voice
baseline VRAM contract relaxed: previously ~10 GB peak (8B at
the cap); new peak with 4B abliterated + Whisper int8_fp16 +
`n_ctx=6144` is ~7.0-7.4 GB on the 4070 Ti = comfortable ~4 GB
buffer beneath the 11.5 GB hard cap. TTFT not re-measured live
(voice-stack-concurrency rule) but no paired draft means the
abliterated 4B will run a touch slower than plain Qwen3.5-4B with
its 0.8B draft -- expected delta is small.

Earlier validating HEAD `29eefe4` (2026-05-13 Phase 14 -- orchestrator VLM wiring -- on top of `036889a` Phases 1-13 -- **native primitives + MCP exposure + voice wiring + preferences + analyze-and-discard + ultron-vision live install** -- on top of `91a3a3a` Phases 1-5 -- **abliterated default LLM + runtime tool-call validator** -- on top of `b1e4297`. Phase 1 added Josiefied-Qwen3-8B-abliterated-v1 Q5_K_M as the new default LLM preset. Phases 2-5 built the runtime tool-call validator that pairs with the abliterated model: 141 rules across 19 categories (K self-protection, A filesystem-destruction, B privilege-escalation, C security-perimeter, D credentials, E system-stability, F repo-integrity, G resource-exhaustion, H untrusted-code-execution, I outbound-impact, J data-exfiltration, M persistence, N process-manipulation, O anti-forensics, P AV/EDR-tampering, Q containers, R sensors, S AI-tampering, plus Cap-1..Cap-4 capability carve-outs). Cross-cutting concerns: Windows-aware path canonicalization (symlinks, junctions, 8.3 short names, percent-escape rejection, bidi-override rejection), tamper-evident hash-chain audit log at logs/safety_audit.jsonl, explicit-intent matcher, cross-capability taint tracker. Wired into the OpenClaw dispatcher's pre-flight check AND the coding bridge's FILE_CHANGE listener. Fail-closed everywhere -- buggy rule = BLOCK_HARD, missing config = BLOCK_HARD, unresolvable path = BLOCK_HARD. **1713 -> 1830 tests passing (+117).** 2026-05-12 Phase 1 added Josiefied-Qwen3-8B-abliterated-v1 Q5_K_M alongside existing presets and flipped the default. Goekdeniz-Guelmez Josiefied + abliterated Qwen3-8B Q5_K_M quantised by mradermacher; 5.85 GB on disk; ~10 GB voice-path peak vs 11.5 GB cap. Abliterated removes content-level refusals; the runtime tool-call validator under `src/ultron/safety/` (forthcoming phases 2-5) gates the capability surface. No paired draft (no abliterated 0.8B GGUF on HF) so no speculative decoding -- expect modest TTFT regression vs 4B preset until measured. Old `qwen3.5-9b` and `qwen3.5-4b` presets retained for swap-back. The 2026-05-12 three-part pass at `b1e4297` shipped audio-artifact fix + filler-ack + **Smart Turn V3** — three changes in one pass on top of `41e13b1`: **XTTS phantom-token mitigation** [user-reported "small sound blips like an unrelated word started then cut off" diagnosed via spectral analysis of a real 58 s session capture; phantom-token signature confirmed at 19.28 s — a 100 ms isolated audio event with 280 ms lead silence + 420 ms trailing silence; XTTS-v2's GPT duration head sometimes emits a fragmentary syllable after the stop-token; fixed by lowering server temperature from 0.75 → 0.65 to sharpen the duration-token distribution AND a defence-in-depth client-side phantom-tail trim that detects the specific pattern and removes it before the v3 filter; speed=1.15 preserved per user direction], **conversational filler-ack** [new `src/ultron/conversational_ack.py` with shuffled-cycle phrase pool ("Mm.", "Right.", "Hm.", "Considering.", etc.) wired into `Orchestrator._build_response_stream` so the no-search conversational branch yields a short thinking-noise BEFORE the LLM stream — masks the ~2.5 s perceived gap between Whisper completing and the LLM's first TTS chunk; gated against short utterances and pending coding-clarifications], **Smart Turn V3 semantic end-of-turn confirmation** [new `src/ultron/audio/smart_turn.py` wraps Pipecat's 8 MB int8 ONNX model (`models/smart_turn/smart-turn-v3.2-cpu.onnx`); CPU-only inference ~12 ms, zero VRAM cost; lazy-loaded, fail-open at every level (missing model file degrades silently to legacy VAD-only behaviour); when active, the VAD silence baseline drops from 1200 ms → 500 ms and the model confirms or rejects the early SPEECH_END; "complete" → submit immediately, "incomplete" → extend capture by 700 ms with VAD silence bumped to the long-utterance backstop; long utterances >8 s of speech bypass the model (the existing adaptive long-utterance backstop handles those); wired into both `_capture_utterance` and `_follow_up_listen`; net win ~500-1200 ms of perceived latency per confidently-complete turn]). **Tests: 1629 → 1711 (+82 net).**

Prior validating HEAD `9139bda` (2026-05-11 follow-up bug-fix pass — Windsurf session — three live-session issues addressed: configurable max-utterance ceiling [class-constant `MAX_UTTERANCE_SECONDS=15.0` was cutting real users off mid-sentence; now `vad.max_utterance_seconds` config, default 30.0 s], completion-narration XTTS pin [`f"Project root: {path}."` made XTTS hang on backslash-laden Windows paths and pinned the GPU at 100 %; now speaks `path.name` only], progress-query classifier coverage gap [`"How is that project going?"` fell through to the conversational LLM because `_PROGRESS_PATTERNS` required `going` immediately after `that` and didn't tolerate the `project` in between; new `_DETERMINER_NOUN` group covers the/that/this/your/our/my × task/project/build/app/code/work/thing/run/job for going / coming / doing / done]).

Prior validating HEAD `431fd7b` (2026-05-11 latency + correctness pass: XTTS cadence speed knob, speculative-stream SR engine-aware, adaptive VAD long-utterance bump, addressing third-party-narrative rule + zero-shot confidence gate, token budget 100k→400k, narration honesty when zero files written, voice_task_require_testing=false default). Chunk-streaming investigation reverted (PitchShift latency block) and documented. Smart Turn V3 + filler-ack queued for next session as highest-impact remaining latency levers. Builds on 2026-05-10 voice swap (XTTS v2 + v3 Ultron filter, **now the default engine**; legacy Piper+RVC still selectable for one-line rollback).

**2026-05-12 Smart Turn V3 — semantic end-of-turn confirmation (NEW).** Third change in the same 2026-05-12 pass.

Pipecat's Smart Turn V3 model (`pipecat-ai/smart-turn-v3` on HuggingFace; BSD-2-Clause) wraps a Whisper Tiny encoder + linear classifier head into an 8 MB int8 ONNX that takes 16 kHz mono PCM up to 8 s and returns a sigmoid probability of "turn complete". CPU inference ~12 ms; **zero VRAM cost** (pinned to `CPUExecutionProvider`). The model runs AFTER Silero VAD detects silence, so it's a confirmation gate on top of the existing VAD pipeline rather than a replacement.

When `vad.smart_turn.enabled` is True AND the model file is present, the orchestrator:

* Drops the VAD silence baseline from the legacy 1200 ms to `smart_turn.fast_path_silence_duration_ms` (500 ms default). Silero now declares SPEECH_END much sooner.
* On first SPEECH_END within a capture, feeds the captured audio to [`SmartTurnDetector.is_complete`](../src/ultron/audio/smart_turn.py). Verdict `complete` (prob ≥ `completion_threshold`, default 0.5) → submit the utterance to Whisper immediately. Verdict `incomplete` → keep listening; bump VAD silence requirement to `long_utterance_silence_duration_ms` (2400 ms) so the next SPEECH_END is real, and start the `incomplete_extension_ms` timer (700 ms default) as a backstop in case the user really was done despite the verdict.
* Long utterances (>`smart_turn.window_seconds` = 8 s of contiguous speech) bypass the model entirely — the adaptive long-utterance VAD backstop already handles those.
* If speech resumes after an `incomplete` verdict, the extension timer is cancelled; the next SPEECH_END (at the bumped slow threshold) is trusted as the real end.

**Fail-open at every level:**

* `vad.smart_turn.enabled: false` → orchestrator constructs without the detector; legacy VAD-only behaviour. No regression.
* `enabled: true` but the model file is missing on disk → `build_detector_from_config` logs WARN and returns None; orchestrator silently falls back to the legacy 1200 ms silence baseline. Users who haven't run `scripts/download_models.py` aren't punished with a hard error.
* Detector loads but a single `is_complete` call raises (transformers / ORT exception) → returns None; orchestrator treats as "undecided" and trusts VAD.
* `SmartTurnDetector` is **lazy-loaded** — construction validates the file exists but doesn't load the ONNX session into memory until the first `is_complete` / `warmup` call. Keeps cold start cheap when smart-turn is enabled but never invoked (e.g. a session with only very short utterances).

**Files:**

* [`src/ultron/audio/smart_turn.py`](../src/ultron/audio/smart_turn.py) — NEW. `SmartTurnDetector`, `SmartTurnVerdict`, `truncate_or_pad_for_smart_turn`, `build_detector_from_config`. Pure-CPU `onnxruntime` session + `transformers.WhisperFeatureExtractor(chunk_length=8)` for the log-mel preprocessing. Single-threaded sequential ORT (`intra_op_num_threads=1`, `inter_op_num_threads=1`, `ORT_ENABLE_ALL`).
* [`src/ultron/config.py`](../src/ultron/config.py) — new `SmartTurnConfig` (enabled, model_path, completion_threshold, fast_path_silence_duration_ms, incomplete_extension_ms, window_seconds, num_threads); nested under `VADConfig.smart_turn`.
* [`config.yaml`](../config.yaml) — production values under `vad.smart_turn`. Default enabled=true; model_path `models/smart_turn/smart-turn-v3.2-cpu.onnx`; threshold 0.5; fast-path silence 500 ms; incomplete-extension 700 ms.
* [`src/ultron/pipeline/orchestrator.py`](../src/ultron/pipeline/orchestrator.py) — new `_build_smart_turn_detector` / `_smart_turn_should_check` / `_run_smart_turn` helper methods; both `_capture_utterance` and `_follow_up_listen` wired with the confirmation gate + extension timeout.
* [`scripts/download_models.py`](../scripts/download_models.py) — Smart Turn V3 added as step 7 of 8 in the download pipeline; pulls `smart-turn-v3.2-cpu.onnx` from HuggingFace into `models/smart_turn/`.

**Tests:** [`tests/test_smart_turn.py`](../tests/test_smart_turn.py) — NEW. 43 tests covering: SmartTurnConfig schema (7 — defaults match production layout, all four range-enforced fields, dict round-trip, nested-under-VADConfig); `truncate_or_pad_for_smart_turn` pure function (6 — under-window passthrough, over-window truncation to last n seconds, int16-to-float32 conversion, multi-dim flatten, non-16kHz rejection, custom window override); `SmartTurnDetector` construction + lazy-loading + failure modes (10 — missing file, out-of-range threshold/window/threads, lazy-loading, warmup-propagates-failure, empty-audio/wrong-sr/post-close all return None); `build_detector_from_config` fail-open (4 — disabled returns None, missing file returns None, absolute-path missing returns None, present file succeeds with lazy detector); real-model end-to-end (6, skipped when ONNX file absent — loads + warmup, silence verdict shape, threshold flip, short audio, long audio truncation, latency under 150 ms median); orchestrator-level wiring (10 — should_check semantics across detector-missing / no-speech / within-window / over-window, run_smart_turn passes verdict through and swallows exceptions, build_smart_turn_detector fail-open for disabled / missing file).

**Voice baseline contract:** intact. The smart-turn path is entirely CPU-side; VRAM accounting unchanged. The Silero VAD itself runs on the same 16 kHz audio it always has — the only difference is its `min_silence_duration_ms` baseline (500 instead of 1200) when smart-turn is wired, and the orchestrator-side confirmation step that runs `_run_smart_turn` on SPEECH_END. Verified ~10-30 ms median inference on this hardware (well under the 150 ms test ceiling and the 12 ms ideal target). Long-utterance backstop fully preserved.

**Live-tuning guidance for the user:** the conservative default is `completion_threshold: 0.5`. If real-world testing shows false-positive cut-offs (e.g. Smart Turn says "complete" when the user was actually pausing mid-thought), bump to 0.6 or 0.7 in `config.yaml` — strict thresholds bias toward "incomplete" verdicts, costing a few hundred ms of perceived latency on confidently-done turns but eliminating the cut-off risk. The `incomplete_extension_ms` knob controls how long the orchestrator waits after an "incomplete" verdict before submitting anyway; 700 ms is roughly the legacy 1200 ms minus the 500 ms fast-path baseline.

**2026-05-12 audio-artifact fix + filler-ack pass (NEW).** Two changes on top of commit `41e13b1`.

1. **XTTS phantom-token mitigation.** User reported small audio blips "like an unrelated word was started then cut off" plus unnatural pauses between words. Diagnosed via spectral analysis of a real 58 s session capture (audio extracted from `2026-05-11 07-23-11.mp4`): the artifact at 19.28 s was a textbook XTTS-v2 phantom-token signature — a 100 ms isolated audio event sandwiched by 280 ms lead silence and 420 ms trailing silence, sitting in what should be an inter-sentence silence region. XTTS-v2's GPT duration head is stochastic and occasionally emits a fragmentary syllable after the stop-token; library-default `temperature=0.75` is on the high side for production stability. **Fix:** new [`tts.xtts_v3.temperature`](../config.yaml) config field (default **0.65**, range [0.4, 1.0]); threaded through HTTP body in [`XttsV3Speech._http_synthesize`](../src/ultron/tts/xtts_v3.py) to the server-side `model.inference_stream(temperature=...)`. Voice character bit-identical (timbre is set by the locked speaker embedding + the v3 filter chain — temperature only affects token-distribution sharpness). **Plus defence-in-depth:** new pure function `trim_phantom_tail(audio_f32, sample_rate, *, ...)` in [`src/ultron/tts/xtts_v3.py`](../src/ultron/tts/xtts_v3.py) detects the specific phantom signature (sustained_speech → ≥150 ms silence → <200 ms event → silence to buffer end) and trims everything after the last sustained-speech region. Runs BEFORE the v3 filter so the reverb tail decays normally into its `tail_silence_ms` padding. Conservative gate: passes through unchanged when no phantom pattern is present. New config fields: `phantom_tail_trim_enabled` (default true), `phantom_tail_silence_threshold` (0.005 ≈ -46 dBFS), `phantom_tail_max_event_ms` (200.0), `phantom_tail_min_lead_silence_ms` (150.0). **Speed=1.15 preserved** per user direction (the snappier cadence is wanted; the artifacts are temperature-driven, not speed-driven). Tests: +20 in `tests/test_xtts_v3_config.py` covering temperature schema, HTTP-body wiring, phantom-tail trim positive case (matches the real-world 19.28 s pattern), negative cases (sustained speech, short inter-word silence, long trailing event, empty/very-short clips), and engine-level enabled/disabled gating.

2. **Conversational filler-ack** ([`src/ultron/conversational_ack.py`](../src/ultron/conversational_ack.py) — NEW). The web-search path already yields a "Verifying against the network." style ack so the user isn't stuck in silence while Brave/Jina/LLM cycle. The no-search conversational path historically had no such ack — typical turn latency budget = 1200 ms VAD silence wait + ~890 ms Whisper + ~79 ms LLM TTFT + ~350 ms first TTS chunk ≈ **2.5 s of silence** between "user stops talking" and "Ultron speaks". Filler-ack masks this by yielding a short thinking-noise ("Mm.", "Right.", "Hm.", "Considering.", "Let me think.", "Noted.", "Processing.", "Working on it.") BEFORE the LLM stream, so the TTS pipeline starts speaking within ~200 ms of Whisper completing. Actual end-to-end latency is unchanged but perceived latency drops sharply. **Gate semantics:** `is_conversational_ack_eligible(user_text, has_pending_clarification)` suppresses the ack on empty input, utterances under 11 chars or 4 words (interjections like "yes" / "no" / "thanks" / "ok" / "sounds good" — the perceived gap is small on short replies, ack would feel over-eager), and during pending coding-task clarifications (the orchestrator already has its own narration flow there). `ConversationalAckSource` is a thin wrapper over the existing web-search `AcknowledgmentSource` with a distinct phrase pool — the two pools rotate independently. The conversational phrases are intentionally tonally non-committal (read as Ultron deliberating, not describing external activity). Wired into [`pipeline/orchestrator.py:_build_response_stream`](../src/ultron/pipeline/orchestrator.py) at all three no-search exit points (`web_gate=None` fallthrough, web_gate-exception fallthrough, `verdict.decision != SEARCH` branch); the `_search_augmented_tokens` path is untouched (already yields its own ack). Fail-open at every step: `coding_voice.has_pending_clarification()` exceptions, `next_phrase` exceptions — both degrade to "no ack" rather than raising. Tests: +24 in `tests/test_conversational_ack.py` covering gate (short/long/empty/clarification/whitespace), source (shuffled cycle, custom pool, empty-pool rejection, no duplicates), phrase pool sanity (no web-search overlap, period-terminated, short), and orchestrator wiring (ack appears on no-gate path, suppressed on short utterance / clarification, fail-open on broken source).

**Tests:** 1629 → 1683 (+54 net). Voice baseline contract unchanged: legacy `piper_rvc` 79 ms / 7913 MB; `xtts_v3` +60 ms / +2 GB. Audio fix touches the voice synthesis path but the phantom-tail trim is sub-millisecond on benign input (the typical case) and the temperature change is server-side (no client overhead). Filler-ack adds one extra token to the conversational stream but doesn't change end-to-end latency. **Smart Turn V3 still queued** for the next session — needs model download + live empirical tuning.

**2026-05-11 follow-up bug-fix pass (Windsurf session, NEW).** Three real-session issues addressed in one pass on top of commit `431fd7b`.

1. **`MAX_UTTERANCE_SECONDS` 15 s class constant cut a real user off mid-sentence** on a complex coding ask ("write me a program that converts PDF to Docx and I want the program to have a GUI TK enter that has a close button..."). Whisper transcribed 15.158 s of audio ending mid-phrase at "a button with a box show". The user wasn't pausing — the hard `elapsed_samples >= max_samples` wall in [pipeline/orchestrator.py:`_capture_utterance`](../src/ultron/pipeline/orchestrator.py) fired before Silero VAD reported `SPEECH_END`. The earlier-2026-05-11 adaptive VAD bump (silence requirement 1200 ms → 2400 ms after 8 s of speech) worked correctly but was overridden by the wall-clock ceiling. Fix: new [`vad.max_utterance_seconds`](../config.yaml) config field (default 30.0 s, schema-bounded [5, 120]). [`Orchestrator.__init__`](../src/ultron/pipeline/orchestrator.py) reads it into `self._max_utterance_seconds`; both `_capture_utterance` and `_follow_up_listen` consume the instance attribute. Class constant kept as fallback default (raised from 15.0 → 30.0). Three schema-coverage tests added in [tests/test_audio.py](../tests/test_audio.py): default-is-30, too-small (<5) rejected, too-large (>120) rejected.

2. **XTTS server hung + GPU pinned at 100 % + computer lagged on completion narration.** [coding/runner.py:`completion_narration`](../src/ultron/coding/runner.py) interpolated `state.cwd` (typed `Path`) directly into the voice text — `"Done. Created 7 files. Project root: C:\STC\ultronPrototype\data\sandbox\converts_pdf_docx. ..."`. XTTS-v2 is a neural model with no robust handling for `\`, `:`, drive letters, and long unbroken slug strings; it entered pathological inference and the server eventually returned `XTTS server synth failed ... timed out`. The legacy Piper+RVC stack didn't show this because Piper's phonemizer pronounces unknown punctuation harmlessly. Fix: the narration now appends `f"Saved under {path.name}."` (just the project folder leaf, matching what [coding/narration.py:`StatusNarrator`](../src/ultron/coding/narration.py) already did for progress narration). Full path remains in the audit log + the `coding_tasks.jsonl` start event for debugging. Existing tests updated (two assertions changed from `str(tmp_path) in narration` to `tmp_path.name in narration` + `str(tmp_path) not in narration`); new regression test `test_completion_narration_does_not_leak_full_path` pins no-backslash / no-drive-letter / no-absolute-path invariants going forward.

3. **"How is that project going?" routed to the conversational LLM instead of `PROGRESS_QUERY`.** [coding/intent.py:`_PROGRESS_PATTERNS`](../src/ultron/coding/intent.py) accepted only `(it|things|claude|the\s+task|that)\s+going` as the subject group — when the user said `"How is that project going?"` the regex tried to match `that` then expected `going` immediately, but found `project` in between and failed. Classifier returned `NONE`, routing fell through to the standard conversational path, and the LLM generated a generic hallucinated "progressing as expected, though I lack specifics on its current status" rather than the runner's actual status narration. Fix: new `_DETERMINER_NOUN` group `(the|that|this|your|our|my)(?:\s+(task|project|build|app|code|work|thing|run|job))?` plugged into all three sub-patterns (`how X going|coming(along)?`, `what's X doing|working on|up to`, `is X done`). The noun is optional so the legacy `that going` / `the doing` phrasings still fire bit-identical. Also added "coming (along)" as an alternate to "going". 18 new parametrized test cases added in [tests/test_coding_intent.py](../tests/test_coding_intent.py) covering the live-session phrasing plus determiner × coding-noun × verb variants. The has_active_task gate is preserved as a tight safety: `test_progress_queries_without_active_task_fall_through` now also pins that the broadened patterns DO NOT hijack the conversational path when no coding task is in flight.

**Tests:** 1604 → 1629 (+25 net). Files touched: 8 source + test files + this doc + `CLAUDE.md`. Voice baseline contract (legacy `piper_rvc` 79 ms / 7913 MB; `xtts_v3` +60 ms / +2 GB) unchanged — no hot-path edits. Config schema is backward-compatible (new field with sane default; existing `config.yaml` snapshots without `max_utterance_seconds` get the new 30 s default).

**2026-05-10 voice-pipeline swap (NEW).** The voice-quality lock was explicitly lifted by user direction to replace the Piper + RVC stack with XTTS v2 streaming + a v3 Ultron post-filter. The voice character is now driven by zero-shot speaker cloning from a 3-min cleaned reference of the actual Ultron source audio, plus a pedalboard DSP chain (PitchShift / Compressor / Delay / Chorus / Distortion / Reverb / EQ) tuned to recover the mechanical / cavity-resonance character that XTTS strips during cloning. User accepted ~50-100 ms TTFT regression vs current Piper+RVC (375 ms vs 313 ms) in exchange for dramatically better naturalness + the option to swap voices later by changing the reference clip alone. The legacy Piper+RVC engine remains intact behind the `tts.engine: "piper_rvc" | "xtts_v3"` config flag for one-line rollback. The XTTS engine spawns a separate Python process running [`ultronVoiceAudio/scripts/xtts_server.py`](../ultronVoiceAudio/scripts/xtts_server.py) in an isolated `.venv-xtts` venv (Coqui TTS's transformers / hydra / omegaconf pins conflict with what fairseq 0.12.2 needs in the main venv); the orchestrator-side client lives at [src/ultron/tts/xtts_v3.py](../src/ultron/tts/xtts_v3.py) and talks to the server over loopback HTTP. Bulk synthetic audio + corpus from the Kokoro fine-tune setup are retained under `ultronVoiceAudio/synth_audio/` for the deferred Kokoro phase (Kokoro fine-tune is the planned latency-recovery step once the rest of Ultron is tuned).

**2026-05-11 token-efficiency fix for voice coding tasks (NEW).** A real session burned 134 k tokens generating a small PDF→DOCX converter and produced zero files (the same prompt to Claude Code directly completes in ~2 k tokens, 126 lines written). Smoking gun: [coding/voice.py:617](../src/ultron/coding/voice.py) was hardcoding ``require_testing=True`` on every voice-dispatched ``TaskRequest``, which prepended the ~270-token "you MUST write tests, run them, fix failures, re-run" discipline preamble defined in [coding/bridge.py](../src/ultron/coding/bridge.py) ``_DISCIPLINE_PREAMBLE`` to the prompt -- forcing Claude into a write-script + write-tests + run-tests + fix-import-errors + re-run loop on ad-hoc utility asks. The fix: new ``coding.voice_task_require_testing`` config field (default **false**) -- voice asks now reach Claude with the user's bare task text, no testing mandate. Operators who actually want the mandate can flip the flag; users who want tests on a specific request can say "with unit tests" in their voice prompt. The orchestrator's correction-loop path ([coding/runner.py](../src/ultron/coding/runner.py)) and the test suite's e2e fixtures both pass ``require_testing`` explicitly so they're unaffected. Pairs with the same-day token-budget bump (100 k → 400 k) and the narration honesty fix to give clear feedback when a session does burn through budget without writing files.

**2026-05-11 narration honesty + chunk-streaming investigation.** Two findings on top of the same-day live-session fix pass below.

* **Narration honesty for zero-file projects** ([src/ultron/coding/runner.py](../src/ultron/coding/runner.py) `CodingTaskRunner.completion_narration`). Real-session bug: a coding task created the project folder, Claude (haiku) exited cleanly with `state.success=True` after burning the 100 k token budget on exploration, but wrote zero scripts. The legacy narration was "Done. Project root: ... <generic Claude tail line> ... Elapsed: 9 seconds." -- the user heard "Done", opened the folder, and found nothing. Fix: when `state.success=True` AND `n_created + n_modified + n_deleted == 0`, the opener becomes "I finished without writing or modifying any files. The project may need more direction, or it may have run out of token budget mid-exploration -- say continue if you want me to keep going." Claude's tail summary is suppressed on this branch (the generic "what should I build?" line added noise to the honest opener). Pairs with the same-day `token_budget_per_session: 100000 → 400000` bump so future similar prompts have headroom.

* **Chunk-streaming investigation -- not shipped.** Prototyped streaming XTTS chunks through the v3 filter chain (via `Pedalboard.process(reset=False)`) to start playback within ~50 ms of the first synth chunk instead of waiting for the full sentence. Reverted after empirical testing showed Pedalboard's `PitchShift` (using Rubber Band offline-quality mode) buffers ~25 000 samples (~1 s at 24 kHz) before emitting any output with `reset=False` -- and the buffered audio can't be cleanly drained without restarting state. Per-chunk `reset=True` works but produces ~125 % RMS divergence from the whole-buffer reference at chunk boundaries -- audible artifacts. The v3 chain order is user-locked (PitchShift sits at position 2; moving it would alter the reverb-on-pitch-shifted-signal character that defines the locked production sound), so the obvious "stream-everything-except-PitchShift, apply PitchShift offline at the end" rearrangement is out of scope. Result: client still accumulates the full sentence at the HTTP layer before applying the v3 filter, but the XTTS *server* still streams PCM chunks via chunked HTTP as before. Future direction: Whisper streaming (start LLM on partial transcripts) would attack the latency from the other end -- the bigger win currently lives in the ~890 ms Whisper batch step, not in within-sentence TTS chunking.

**2026-05-11 live-session fix pass.** Four targeted fixes landed in response to a real session log:

1. **Speculative-stream SR engine-aware** ([src/ultron/tts/xtts_v3.py](../src/ultron/tts/xtts_v3.py)). The XTTS path was reading `tts.speculative_stream_sample_rate` (48000, tuned for the legacy Piper+RVC stack) and triggering the close-and-reopen path on every turn ("XTTS speculative SR 48000 != actual 24000; reopening" log line). The XTTS engine now uses `self._sample_rate` (24000 native) directly. Legacy `speech.py` path unchanged. Saves ~50-100 ms per turn.
2. **Adaptive VAD end-of-turn** ([src/ultron/audio/vad.py](../src/ultron/audio/vad.py), [src/ultron/pipeline/orchestrator.py](../src/ultron/pipeline/orchestrator.py)). Long technical prompts ("Write me a program that converts PDF to Doc X ... and frees up resources and make sure") were getting clipped at the flat 1200 ms silence requirement. New `VoiceActivityDetector.set_min_silence_duration_ms(ms)` + `reset()` baseline-restore. `_capture_utterance` tracks speech-active time; once past `vad.long_utterance_threshold_seconds` (8.0), bumps to `vad.long_utterance_silence_duration_ms` (2400) for the rest of the capture. Short utterances unaffected.
3. **Addressing — third-party narrative rule + zero-shot confidence gate** ([src/ultron/addressing/rules.py](../src/ultron/addressing/rules.py), [src/ultron/addressing/classifier.py](../src/ultron/addressing/classifier.py)). Real-session log showed third-person narration about Ultron ("got him to the point where he's workable. You'll see", "I'm talking to him") sliding through the rule layer to zero-shot YES at exactly 0.75 confidence — flan-t5-small's saturation level. New tight `_THIRD_PARTY_NARRATIVE` rule catches "I'm talking to him/her/it", causative "got/made/let him to <verb>", "you'll see", "watch this/him", and "(he|it|she)'s <state>" patterns at 0.85 confidence (short-circuits zero-shot). Plus a new `zero_shot_addressed_min_confidence` gate (default 0.80) that demotes low-confidence zero-shot YES verdicts to NOT_ADDRESSED via the default-silent path. Legitimate Ultron commands ("tell him to send the email", "ask her about the meeting") are unaffected because the narrative rule is intentionally tight.
4. **Coding token budget bump** ([config.yaml](../config.yaml)). `coding.token_budget_per_session: 100000 → 400000`. 100k was enough for incremental edits but left zero headroom for new projects — a PDF→docx scratch project burned 134k on tool exploration alone before writing any files. 400k gives new-project sessions room to discover the environment, decide on structure, write files, and verify. The 80% warning still fires so the user has visibility.

**2026-05-11 cadence tune.** XTTS native default speaking rate (1.0) was running slow in live use; new `tts.xtts_v3.speed` config field threads XTTS v2's native duration multiplier through the client HTTP body to the server's `model.inference_stream(speed=...)` call. Production set to **1.15** (~15% faster speech). Schema-bounded to `[0.5, 2.0]`; safe-without-slurring range is roughly 0.7-1.4. Because the speed adjustment happens at synthesis time on the GPT duration tokens, the v3 pedalboard filter (pitch shift / delay / chorus / reverb / EQ) is bit-for-bit unchanged — it just processes a shorter audio buffer with the same filter characteristics + the same 200 ms reverb-tail padding. Wiring sites: [src/ultron/config.py](../src/ultron/config.py) (XttsV3Config.speed), [config.yaml](../config.yaml) (production value), [src/ultron/tts/xtts_v3.py](../src/ultron/tts/xtts_v3.py) (HTTP body), [ultronVoiceAudio/scripts/xtts_server.py](../ultronVoiceAudio/scripts/xtts_server.py) (server-side `inference_stream(speed=...)` call).

Earlier 2026-05-10 live-session fixup pass — four fixes landed on top of the 2026-05-09 audio + memory layer in response to a real session log:

1. **Producer-signaled lookahead in TTS** ([src/ultron/tts/speech.py](../src/ultron/tts/speech.py)). New `ClipItem` namedtuple `(audio, sample_rate, is_known_last)` carried through `piper_q` / `audio_q`. Playback now plays each clip IMMEDIATELY on receipt and only blocks for the next AFTER playing — the legacy play-after-peek pattern delayed first-clip playback up to 10 s waiting for the next clip to determine "is this last?". This was the root cause of the web-search ack ("Verifying against the network.") arriving AFTER the response instead of before. RVC worker `piper_q.get` timeout bumped 10 s → 60 s so a slow generator (long Brave + Jina + LLM TTFT) doesn't kill mid-stream playback. Voice character preserved bit-identical (same edge fades, same inter-sentence pauses, same tail silence).
2. **Mode-aware audio pre-roll** ([src/ultron/audio/ring_buffer.py](../src/ultron/audio/ring_buffer.py), [src/ultron/pipeline/orchestrator.py](../src/ultron/pipeline/orchestrator.py)). The 2026-05-09 ring-buffer trim (0.5 → 0.15 s) fixed the COLD-mode "Tron" prefix but inadvertently clipped the leading word in WARM-mode follow-ups (Silero VAD has ~100-200 ms speech-start latency). Solution: ring buffer sized to the LARGER slice (0.5 s), `RingBuffer.snapshot(last_n_samples=...)` slices to the right length per mode. New config: `audio.cold_pre_roll_seconds: 0.15` (post-wake), `audio.warm_pre_roll_seconds: 0.5` (post-TTS follow-up). `audio.ring_buffer_seconds: 0.5` is now the buffer storage capacity.
3. **Browser navigate pattern broadened** ([src/ultron/openclaw_routing/classifier.py](../src/ultron/openclaw_routing/classifier.py)). The determiner-less `_BROWSER_NAVIGATE` pattern required either no determiner or "the" before the noun, missing "open a browser window with X" / "open my browser tab" / "open a new tab to GitHub". Live regression: "Can you open a browser window with Google's homepage for me?" fell through to the LLM which apologised it couldn't open browsers. Added two alternatives: `open [det] [new] browser [window|tab]` (destination optional) and `open [det] [new] (window|tab) (with|to|for|on) X` (destination required to avoid generic "tab" matches).
4. **Brevity reinforcement on short questions** ([src/ultron/response_style.py](../src/ultron/response_style.py) — NEW). The 4B model has a default-toward-verbose habit on simple queries ("What are the Orcs in 40k?" → 1164-char four-paragraph essay). New pure-function `apply_brevity_hint(user_text)` prepends a `[Style: respond in 1-3 short sentences …]` directive when the question is brief (≤12 words / ≤80 chars) AND not an explicit ask for depth (skipped on `explain` / `step by step` / `walk me through` / `elaborate` markers). Wired only into the non-search conversational path (`Orchestrator._build_response_stream`) — the search path's augmented prompt already carries its own length directive. **No SOUL.md / RVC / Piper changes** (voice-quality lock preserved); the addendum is per-call only.

Earlier 2026-05-09 audio + memory + nuanced-retrieval layer (still in force):

* Latency hot-fix: parallel Jina fetches with collective deadline + Jina timeout/max_fetch reductions + VAD silence cap raise (500 → 1200 ms). Web-search worst case 13.5 s → ~6 s.
* TTS pipeline pass: Piper / RVC split into two queue-decoupled stages + speculative `sd.OutputStream` open + `latency='low'` PortAudio hint. ~80–180 ms residual TTS gain. Voice character bit-identical.
* Contamination + nuanced-retrieval: `ConversationMemory.retrieve()` cosine-similarity threshold (`memory.rag_min_relevance: 0.6`) + composite scoring (cosine + RRF + recency-weighted boost). Recent-turn history feed capped at `memory.history_turns_for_llm: 4`. `LLMEngine.suppress_memory_context` kwarg as a knob. Direct mic input via Focusrite. New `audio.input_gain_db` pre-amp; `scripts/audio_diagnostic.py` harness.
* Sample-rate fix (2026-05-10): `tts.speculative_stream_sample_rate` default 40000 → 48000 to match the actual Ultron RVC model output. Eliminates the "TTS speculative SR 40000 != actual 48000; reopening" log line and ~50-100 ms wasted reopen per turn.

All 12 V1-gap enhancements wired; defaults chosen on net-benefit grounds:

| Flag | Default | Why |
|---|---|---|
| Phase 1 A3 facts wiring | always on | no flag; pure additive on coding-clarification path |
| Phase 1 B1 knowledge_source | always on | no flag; pure additive |
| Phase 2 A4 pre-task confirmation | **OFF** | adds ~0.5 s TTS to every coding dispatch — UX cost on every fire |
| Phase 3 A1 gaming mode | **OFF** | safety-critical plugin disable; operator opts in |
| Phase 3 C3 desktop / window control | **ON** | no observable effect when OpenClaw bridge is offline; ready when wired |
| Phase 4 A2 multi-pass retrieval | **ON** | voice baseline (rule verdicts) unaffected; ~150-200 ms only on memory-aware queries that already paid LLM-preflight cost |
| Phase 5 B5 preflight benchmark | n/a | script + doc; no runtime flag |
| Phase 6 B2 query dedup | always on | no flag; pure additive |
| Phase 6 B3 citation marker | **superscript** | matches V1-spec Part 4.4 wording; references list keeps bracket form |
| Phase 6 B4 / C1 / C2 | n/a | verifications / aliases; no runtime flag |

**Classifier gating (V1-gap A1 / C3):** the new GAMING_MODE / DESKTOP_AUTOMATION / WINDOW_AUTOMATION classifier branches are gated on `openclaw.enabled` AND the per-feature flag. With OpenClaw offline (today's default), the new patterns DO NOT fire — utterances like "take a screenshot of the desktop" / "I'm about to play Valorant" fall through to the conversational LLM, preserving the pre-Phase-3 UX. Once the user wires OpenClaw + flips per-feature flags, the routing engages automatically.

Prior validating HEAD `bb08a65` (closes OpenClaw integration Phases 3–13 + Phase 13 finish).

State at this validation:
- Foundation phase complete (Parts 0–7); Part 3.5 unified-config migration intentionally deferred; 16-step real-stack smoke test still pending (interactive).
- OpenClaw integration: **Phases 0–13 done.** Phase 13 closed the original deferrals: stdio MCP entry script (`scripts/run_ultron_mcp_for_openclaw.py`) + five MCP tools (`get_heartbeat_alerts`, `acknowledge_alert`, `run_maintenance`, `list_active_coding_sessions`, `get_recent_voice_alerts`); voice-side `SystemStatusReporter` + `SYSTEM_STATUS` intent kind + classifier patterns; `OpenClawBridgeConfig.mcp_server_command="auto"` default that resolves to the canonical entry point. Auto-enabled on the user's OpenClaw install: `session-memory` + `command-logger` hooks, `memory-wiki` plugin, `ultron-mcp` MCP registration. Live-stack smoke tests remain user-led per the per-phase setup docs.
- 4B optimization plan: Stages A–H + voice-driven model swap + Items 4–8 fully wired into trigger sites + **all five flags defaulted ON** in `config.yaml`. Stage E voice character A/B passed (interactive A/B was approved 2026-05-08).
- Active LLM: **`qwen3.5-4b`** preset (model_path `models/Qwen3.5-4B-Q4_K_M.gguf`, draft `Qwen3.5-0.8B-Q4_K_M.gguf`, n_ctx 8192). 9B GGUF retained for swap-back.
- Voice baseline (10-query stack with all Items ON): **TTFT median 79 ms**, **VRAM peak 7913 MB** (-2461 MB / -2.5 GB vs 9B). See [baselines.json](../baselines.json).
- Items 4–8 measurable verification: [scripts/verify_items_4_to_8.py](../scripts/verify_items_4_to_8.py) exercises each item in its trigger scenario and prints concrete deltas.
- Stale-`.env` gotcha resolved: `ULTRON_LLM_MODEL_PATH=...9B...` line in `.env` was silently overriding the preset. Now commented out (line 84).
- **1845 tests collected; 1830 passed, 15 skipped (GPU-gated), 0 failed.** Net delta vs Foundation Phase 7 baseline: +835. Most recent additions:
  - 2026-05-12 Phases 2-5 -- runtime tool-call validator: +117 across `tests/safety/` (test_path_resolver, test_audit_log, test_validator_core, test_rules_by_category, test_intent_and_taint, test_dispatcher_integration) plus 2 baseline-listener-count updates in `tests/test_canonical_monitor_runner_wiring.py`.
  - 2026-05-12 Phase 1 -- Josiefied-Qwen3-8B-abliterated preset: +2 (`test_josiefied_preset_resolves_paths_and_ctx`, expanded `test_preset_table_contents`) + 3 existing tests refreshed for the new default (`test_default_preset_is_josiefied_8b`, `test_legacy_9b_preset_still_available`, `test_yaml_load_default_preset_back_compat`) in `tests/test_llm_preset.py`.
  - 2026-05-12 Smart Turn V3 — semantic end-of-turn confirmation: +43 covering SmartTurnConfig schema, `truncate_or_pad_for_smart_turn` pure function, `SmartTurnDetector` construction + lazy-loading + fail-open, `build_detector_from_config` fail-open contract, real-model end-to-end (gated on ONNX file presence), and orchestrator-level wiring (`tests/test_smart_turn.py` — NEW).
  - 2026-05-12 audio-artifact fix + filler-ack: +12 XTTS temperature schema + HTTP-body wiring + `trim_phantom_tail` positive/negative cases (`tests/test_xtts_v3_config.py` grew from 18 to 30); +24 conversational filler-ack gate + source + phrase-pool sanity + orchestrator wiring (`tests/test_conversational_ack.py` — NEW); +3 regression updates across xtts_v3 default schema coverage.
  - 2026-05-11 follow-up bug-fix pass (Windsurf session): +3 schema coverage for the new `vad.max_utterance_seconds` field (default 30 s; <5 rejected; >120 rejected) in `tests/test_audio.py`; +1 regression `test_completion_narration_does_not_leak_full_path` pinning no-backslash / no-drive-letter / no-absolute-path in `tests/test_coding_runner.py` (plus 2 existing tests updated from `str(tmp_path) in narration` to `tmp_path.name in narration`); +18 broadened progress-query parametrized cases (`How is that project going?` and determiner × coding-noun × verb variants) plus +3 fall-through cases gating the new patterns on `has_active_task=True` in `tests/test_coding_intent.py`.
  - 2026-05-11 token-efficiency fix: +4 voice-task testing-mandate coverage (`test_voice_dispatch_defaults_to_no_test_mandate`, `test_voice_dispatch_honors_config_flag_for_testing`, `test_render_prompt_omits_discipline_preamble_without_testing`, `test_coding_config_voice_task_require_testing_defaults_false`) in `tests/test_coding_voice.py`.
  - 2026-05-11 narration honesty: +2 `completion_narration` regression coverage (honest-when-zero-files opener fires; legacy "Done." preserved when files were written) in `tests/test_coding_runner.py`. Plus one existing test updated to accept either form (`tests/test_coding_voice.py::test_pending_completion_returns_none_until_transition` -- its fake bridge emits no FILE_CHANGE events, so it now hits the honest branch).
  - 2026-05-11 live-session fix pass: +5 addressing coverage (third-party narrative rule catches, legit commands unaffected, zero-shot min-confidence gate demotes/passes/default-zero-preserves) in `tests/test_addressing.py`; +3 VAD adaptive-silence coverage (set_min_silence_duration_ms, reset restores baseline, floor at 1 window) in `tests/test_audio.py`.
  - 2026-05-11 cadence tune: +3 XTTS v3 speed knob coverage (range enforcement + round-trip + client HTTP-body wiring) in `tests/test_xtts_v3_config.py`.
  - 2026-05-10 voice-pipeline swap: +12 XTTS v3 config + Ultron filter coverage (`tests/test_xtts_v3_config.py` — NEW).
  - 2026-05-09 latency hot-fix: +6 parallel Jina fetch + collective deadline (`tests/test_web_search_parallel_fetch.py`).
  - 2026-05-09 TTS hot-fix: +11 Piper/RVC pipeline split + speculative stream + low-latency mode (`tests/test_tts_pipeline_parallel.py`).
  - 2026-05-09 contamination + nuanced-retrieval pass: +6 ``suppress_memory_context`` regression (`tests/test_llm_memory_suppression.py`) + +8 cosine threshold + history cap regression (`tests/test_memory_relevance_filter.py`).
  - 2026-05-10 live-session fixup: +4 producer-signaled lookahead / ack-first / RVC starvation regression (`tests/test_tts_pipeline_parallel.py`) + +5 RingBuffer.snapshot slicing (`tests/test_audio.py`) + +8 browser navigate "open a browser window" coverage (`tests/routing/test_classifier.py`) + +22 brevity-hint coverage (`tests/test_response_style.py` — NEW).

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
├── baselines.json                  ← VRAM + latency baselines (9B / current production reference)
├── baselines_4b_q4_in_process.json ← 4B plan Stage D snapshot (4B alone, no spec decoding)
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
│       ├── response_style.py       ← 2026-05-10: per-call brevity hint (apply_brevity_hint)
│       ├── conversational_ack.py   ← 2026-05-12: filler-ack on conversational path (ConversationalAckSource, is_conversational_ack_eligible)
│       │
│       ├── observations/            ← 2026-05-18 Phase 0+1: canonical observation framework
│       │   ├── __init__.py
│       │   ├── schema.py             ← Observation dataclass + KNOWN_SUBSYSTEMS / KNOWN_OUTCOMES + new_event_id
│       │   ├── writer.py             ← ObservationWriter thread-safe JSONL appender + singleton accessors
│       │   ├── integrations.py       ← observe_routing_verdict / observe_addressing_verdict / observe_retrieval / observe_llm_call
│       │   ├── outcome_resolver.py   ← Post-hoc OutcomeResolver: emit outcome_resolution rows from history
│       │   └── lineage_overlap.py    ← compute_lineage_overlap + emit_lineage_usage_rows (pure primitives)
│       │
│       ├── desktop/                  ← Desktop automation primitives (NEW; native, no ClawHub deps)
│       │   ├── __init__.py
│       │   ├── monitors.py           ← Win32 monitor enumeration + find_monitor + point_to_monitor
│       │   ├── capture.py            ← mss-based multi-monitor capture; taint-tracker integration
│       │   ├── windows.py            ← pywin32 + psutil window enum + foreground detection + monitor-index lookup
│       │   ├── placement.py          ← move/resize/maximize/focus on target monitor
│       │   ├── launcher.py           ← AppLauncher with registry (Chrome/Cursor/Discord/VSCode/Edge/Firefox/etc.) + Chrome default-profile + Google Images convenience
│       │   ├── uia.py                ← pywinauto UIA text extraction + semantic click/type with Cap-3/Cap-4 safety hooks
│       │   ├── input_control.py      ← pyautogui mouse+keyboard, rate-limited, validator-gated, blocks input on UAC/security windows
│       │   ├── screen_context.py     ← orchestrator: assemble foreground + windows + UIA text + optional VLM description for LLM injection
│       │   ├── vlm.py                ← Moondream2 VLM wrapper (transformers + trust_remote_code), CPU-only on-demand, lazy-loaded, fail-open
│       │   ├── voice.py              ← Phase 8 voice handlers (handle_app_launch / handle_screen_context_query) + 2026-05-14 third-pass handlers (handle_window_move / handle_window_close) bridging RoutingIntent -> native primitives
│       │   └── preferences.py        ← Phase 10 preference learning (JSONL log + optional OpenClaw workspace mirror; find_preference_for_phrase for recency-weighted lookup)
│       │
│       ├── safety/                  ← 2026-05-12 Phase 2-5: runtime tool-call validator
│       │   ├── __init__.py
│       │   ├── validator.py        ← ToolCallValidator core dispatcher, Verdict, RuleContext, RuleResult
│       │   ├── path_resolver.py    ← Windows-aware canonicalization (symlinks, junctions, 8.3, bidi-override rejection)
│       │   ├── audit.py            ← tamper-evident hash-chain audit log (logs/safety_audit.jsonl)
│       │   ├── policy.py           ← Policy dataclass + load_policy() with K-protected paths
│       │   ├── intent.py           ← explicit-intent matcher (verb+object within window)
│       │   ├── taint.py            ← cross-capability taint tracker (60s TTL hash-match)
│       │   └── rules/
│       │       ├── __init__.py
│       │       ├── base.py         ← Rule ABC + PathSetRule + PathPatternRule + CommandPatternRule + ToolNameRule + SandboxConfinementRule
│       │       ├── category_k.py   ← K1-K10 self-protection (validator + config + audit log + bridge + manifests + ingested files + shell init + MCP entry scripts)
│       │       ├── category_a.py   ← A1-A12 filesystem destruction
│       │       ├── category_b.py   ← B1-B9 privilege escalation + system config
│       │       ├── category_c.py   ← C1-C12 security perimeter
│       │       ├── category_d.py   ← D1-D17 credential / secret access (OUT-gate)
│       │       ├── category_e.py   ← E1-E8 system stability
│       │       ├── category_f.py   ← F1-F8 repo / data integrity
│       │       ├── category_g.py   ← G1-G5 resource exhaustion
│       │       ├── category_h.py   ← H1-H12 untrusted code execution (LOLBins, encoded PS, WMI proc create)
│       │       ├── category_i.py   ← I1-I6 outbound impact (email, social, finance, paid APIs)
│       │       ├── category_j.py   ← J2-J7 data exfiltration (DNS / ICMP / cloud-storage / clipper-malware)
│       │       ├── category_m.py   ← M1-M11 persistence mechanisms
│       │       ├── category_n.py   ← N1-N6 process / memory manipulation
│       │       ├── category_o.py   ← O1-O8 anti-forensics
│       │       ├── category_p.py   ← P1-P5 AV / EDR tampering
│       │       ├── category_q.py   ← Q1-Q4 containers + virtualization
│       │       ├── category_r.py   ← R2 sensors (webcam)
│       │       ├── category_s.py   ← S1, S4 AI-specific tampering
│       │       └── cap_carveouts.py ← Cap-1..Cap-4 capability bound rules
│       │
│       ├── audio/                  ← Audio capture, VAD, wake-word
│       │   ├── capture.py          ← AudioCapture (sounddevice callback thread)
│       │   ├── devices.py          ← Device-resolution helpers (resolve_device, describe_device)
│       │   ├── ring_buffer.py      ← Pre-speech audio buffer
│       │   ├── smart_turn.py       ← Smart Turn V3 ONNX wrapper (NEW 2026-05-12; CPU-only end-of-turn confirmation)
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
│       │   ├── inference.py        ← LLMEngine (llama-cpp-python; Qwen3.5-4B Q4_K_M active, 9B kept; reload_for_preset for hot swap)
│       │   ├── compression.py      ← 4B plan Item 4: heuristic + perplexity-scorer-hook compressor for RAG/web/history (default OFF)
│       │   ├── context_scoring.py  ← 2026-05-18 Phase 1: adaptive context-window heuristic (default-OFF; ContextRecommendation)
│       │   └── self_consistency.py ← 4B plan Item 6: N-sample majority-vote driver + aggregators (text/JSON/label) (default OFF)
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
│       ├── tts/                    ← Piper + RVC + XTTS engines + ack cache
│       │   ├── precomputed_ack.py  ← PrecomputedAckClipCache (NEW 2026-05-15; ~350 ms saved per cache hit)
│       │   ├── rvc.py              ← RvcConverter (Piper PCM → Ultron timbre)
│       │   ├── speech.py           ← TextToSpeech (legacy Piper + RVC engine; selected by tts.engine="piper_rvc"; ack cache + prepare_output_stream)
│       │   ├── ultron_filter.py    ← v3 Ultron mechanical filter (NEW 2026-05-10; pedalboard DSP chain)
│       │   └── xtts_v3.py          ← XTTSV3Speech engine (NEW 2026-05-10; selected by tts.engine="xtts_v3"; ack cache + prepare_output_stream)
│       │
│       ├── coding/                 ← Phase A coding orchestration + Coding Addendum
│       │   ├── anchors.py          ← 2026-05-18 E2: goal-anchor planning primitives (GoalAnchor / AnchorBudget / AnchorPlan / decompose_into_anchors)
│       │   ├── audit.py            ← SessionAuditWriter (per-session JSONL)
│       │   ├── bridge.py           ← Abstract CodingBridge + TaskEvent vocabulary
│       │   ├── canonical_monitor.py ← 4B plan Item 7: per-session tool-call canonical-path monitor (default OFF)
│       │   ├── voice_lock.py       ← 2026-05-18 E5: voice-character-lock pre-dispatch scanner + FILE_CHANGE helper
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
│       │   └── voice.py            ← CapabilityVoiceController (handles MODEL_SWITCH for voice-driven LLM swap; Phase 5 rename; alias preserved)
│       │
│       ├── pipeline/
│       │   └── orchestrator.py     ← Main event loop / state machine
│       │
│       ├── openclaw_routing/       ← Phase 5 capability-routing layer
│       │   ├── ambiguity.py        ← 2026-05-18 Phase 1: should_clarify predicate + AmbiguityVerdict (band [0.4, 0.65) by default; flag-gated)
│       │   ├── block_and_revise.py ← 4B plan Item 8: ToolCallValidator pre-flight gate on OpenClaw tool calls (default OFF; fails open)
│       │   ├── classifier.py       ← classify_routing() - top-level intent classifier (incl. MODEL_SWITCH for voice-driven LLM swap)
│       │   ├── decision_log.py     ← RoutingDecisionLog (logs/routing_decisions.jsonl)
│       │   ├── decomposer.py       ← HybridTaskDecomposer (Qwen-driven JSON output; opt-in self-consistency)
│       │   ├── disambiguator.py    ← IntentDisambiguator (CODING/AUTOMATION/HYBRID/UNCLEAR; opt-in IRMA enrichment)
│       │   ├── dispatcher.py       ← OpenClawDispatcher (5 stub methods)
│       │   ├── intents.py          ← RoutingIntentKind enum (21 values incl. APP_LAUNCH / SCREEN_CONTEXT_QUERY / WINDOW_MOVE / WINDOW_CLOSE / MODEL_SWITCH / SYSTEM_STATUS / GAMING_MODE / DESKTOP_AUTOMATION / WINDOW_AUTOMATION), RoutingIntent + per-category dataclasses (incl. AppLaunchIntent, ScreenContextIntent, WindowMoveIntent, WindowCloseIntent, ModelSwitchIntent, SystemStatusIntent, GamingModeIntent, DesktopIntent, WindowIntent)
│       │   ├── irma.py             ← 4B plan Item 5: InputReformulator + ReformulationContext (default OFF)
│       │   └── runner.py           ← AutomationTaskRunner (mirror of CodingTaskRunner)
│       │
│       ├── openclaw_bridge/        ← OpenClaw integration Phases 1, 3, 4, 5, 6, 13 (complete)
│       │   ├── persona.py          ← PersonaLoader (mode-based: user_facing/background/heartbeat/bootstrap) + hot reload
│       │   ├── lifecycle.py        ← OpenClawLifecycle (HTTP health probes; never raises)
│       │   ├── client.py           ← OpenClawClient (async CLI subprocess transport: invoke_tool / send_message / trigger_heartbeat / mcp_*)
│       │   ├── workspace.py        ← WorkspaceWriter (atomic writes + filelock for MEMORY.md / USER.md / daily files)
│       │   ├── events.py           ← OpenClawEventReceiver (gated-off scaffold for [voice] inbound handoff)
│       │   ├── mcp_registration.py ← UltronMcpRegistrar (idempotent `openclaw mcp set` with background retry)
│       │   ├── holder.py           ← OpenClawBridge (orchestrator-owned holder: probe → register → retry-thread → fire_and_forget → record_heartbeat_alert; auto-resolve "auto" command)
│       │   ├── notifications.py    ← NotificationDispatcher (Phase 4 — proactive Telegram pings on coding-completion / heartbeat / etc.)
│       │   ├── heartbeat_alerts.py ← HeartbeatAlertLog (Phase 5 — JSONL-backed alert log with atomic update + retention)
│       │   ├── browser.py          ← BrowserTool (Phase 6 — navigate/snapshot/click/type/screenshot via OpenClawClient.invoke_tool)
│       │   ├── mcp_tools.py        ← Stdio MCP server (Phase 13 — get_heartbeat_alerts / acknowledge_alert / run_maintenance / list_active_coding_sessions / get_recent_voice_alerts)
│       │   └── system_status.py    ← SystemStatusReporter (Phase 13 — voice-side reporter for SYSTEM_STATUS intents)
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
│   ├── openclaw_integration_final_summary.md ← Cross-phase reference + intentional deviations + setup-readiness checklist
│   ├── phase_1_summary.md          ← OpenClaw Phase 1 close-out (persona migration)
│   ├── phase_3_summary.md          ← OpenClaw Phase 3 close-out (bridge layer)
│   ├── phase_4_summary.md          ← OpenClaw Phase 4 close-out (Telegram channel)
│   ├── phase_5_summary.md          ← OpenClaw Phase 5 close-out (heartbeat)
│   ├── phase_6_summary.md          ← OpenClaw Phase 6 close-out (browser tool)
│   ├── openclaw_telegram_setup.md  ← User-side: Telegram bot setup procedure
│   ├── openclaw_heartbeat_setup.md ← User-side: agents[].heartbeat block setup
│   ├── openclaw_browser_setup.md   ← User-side: Playwright/Chromium + tools.alsoAllow
│   ├── openclaw_cron_setup.md      ← User-side: cron jobs (Windows Task Scheduler fallback)
│   ├── openclaw_hooks_setup.md     ← User-side: bundled hooks; custom hook scaffolding
│   ├── openclaw_memory_wiki_setup.md ← User-side: Memory Wiki plugin enablement
│   ├── openclaw_media_generation_setup.md ← User-side: local-only ComfyUI setup (paid APIs out)
│   ├── mobile_node_setup.md        ← User-side: iOS / Android pairing procedure
│   ├── standing_orders.md          ← Standing-order programs in AGENTS.md
│   ├── memory_architecture.md      ← Three-layer memory model (Qdrant + workspace + Wiki)
│   ├── 4b_optimization_plan.md     ← 4B-model migration plan (all stages done)
│   ├── model_checksums.md          ← SHA256 of every GGUF in `models/`
│   ├── comprehensive_test_plan.md  ← Functional / correctness pass architecture (16 phases, 38 dimensions)
│   ├── comprehensive_test_report.md ← Functional pass results + 145-row metrics table; 4 classifier coverage gaps fixed
│   ├── comprehensive_quality_plan.md ← Quality pass architecture (13 phases Q0–Q13, 38 dimensions, ≤10 iter loop)
│   ├── comprehensive_quality_report.md ← Quality pass results + 107-row metrics table + Q10 iteration audit
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
│   ├── swap_llm_preset.py          ← 4B plan: edit config.yaml in place to swap LLM preset (validates GGUFs, atomic write)
│   ├── verify_voice_character_4b.py ← 4B plan Stage E: A/B voice-character helper (5 queries × 4B/9B)
│   ├── verify_items_4_to_8.py      ← 4B plan: exercises Items 4–8 in their trigger scenarios; prints measurable deltas
│   ├── comprehensive_test_harness.py ← End-to-end test pass: routing accuracy on 63-utterance labeled set, web-gate rule accuracy, circuit-breaker state machine, memory stress (4 threads × 50 turns), classifier-gating regression
│   ├── real_api_smoke.py           ← Real-API sparing smoke: 1 Brave query + 1 Brave-Jina chain + 1 Claude Code haiku invocation (≤2 paid web calls + ≤1 tiny Anthropic API call total)
│   ├── quality_harness.py          ← Quality pass: Q1 persona/factual/hallucination + Q2 persona modes + Q4 memory recall/labeling/ranking + Q5 Whisper WER/flush/VAD + Q7 Items 4-8 + Q8 adversarial in one process
│   ├── quality_q3_web.py           ← Quality pass Q3: web-search source ranking + snippet utilization + Jina direct + cache + citation rendering + ack latency + dedup (10 Brave + 10 Jina cap)
│   ├── quality_q6_mocked.py        ← Quality pass Q6.D + Q9: projection budget + phrase pool + browser parsing + slug routing + gaming mode (no real API)
│   ├── quality_q6_claude.py        ← Quality pass Q6.E + Q6.F: 4 single-fn Claude Code tasks + 5 full Tkinter app generation (sandbox-isolated)
│   ├── _quality_q10_iter1_verify.py ← Quality Q10 iter verification: 3 prompt-injection probes against the live LLM
│   ├── _quality_q6f_rescore.py     ← Quality Q6.F re-scorer: applies relaxed regex to existing on-disk apps (no new Claude calls)
│   ├── start_llamacpp_server.py    ← OpenClaw Phase 0 + 4B plan Stage C: launch llama-cpp-server with voice-pipeline params (+ --model-draft / --draft-num-pred-tokens / --from-config)
│   ├── supervised_llamacpp_server.py ← OpenClaw Phase 0: supervisor wrapper with auto-restart
│   ├── smoke_test_llamacpp.ps1     ← OpenClaw Phase 0: PowerShell health probe for llama-cpp-server
│   ├── _bench_llm_http.py          ← OpenClaw Phase 0: HTTP-mode TTFT benchmark
│   ├── _log_proxy.py               ← OpenClaw Phase 0: tee proxy for debugging Gateway → server traffic
│   ├── _record_phase0_baseline.py  ← OpenClaw Phase 0: baseline recorder
│   ├── _merge_phase0_baselines.py  ← OpenClaw Phase 0: baseline merger
│   ├── _vram_peak_monitor.py       ← Auxiliary VRAM peak monitor (used by extended baselines)
│   ├── run_maintenance_for_cron.py ← OpenClaw Phase 7: cron-friendly maintenance wrapper (JSON / pretty / exit codes)
│   ├── run_ultron_mcp_for_openclaw.py ← OpenClaw Phase 13: stdio MCP entry script OpenClaw spawns to call Ultron tools
│   ├── cleanup_stale_processes.py ← 2026-05-14 cleanup pass: kill orphaned pytest workers + stale MCP stubs + orphan XTTS servers (preserves live Ultron via port-19761 listener check)
│   ├── bench_llm_ubatch.py        ← 2026-05-15 latency: sweep n_batch / n_ubatch combinations (writes baselines.json:llm_n_ubatch_sweep)
│   ├── bench_stt_latency.py       ← 2026-05-15 latency: measure Whisper STT latency at varied audio lengths (drove beam_size 5->1 decision)
│   ├── bench_llm_prefix_cache.py  ← 2026-05-16 latency 2: cold-vs-warm TTFT bench for LlamaRAMCache (writes baselines.json:llm_prefix_cache_bench; result: -15 ms regression on this stack -> Phase 2 default flipped to disabled)
│   └── eval_harness.py            ← 2026-05-18 Phase 0: classifier-only eval harness (routing + addressing + web_gate); reads tests/eval/corpus.jsonl; writes logs/eval_runs/<ts>.json; exit codes 0/1/2 for CI
│
├── tests/
│   ├── conftest.py                 ← Path setup + pytest_sessionfinish hook that reaps test-spawned python children (preserves the live Ultron on port 19761)
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
│   ├── integration/                ← Phase 6: end-to-end pipeline (83 tests + bridge e2e)
│   │   ├── conftest.py
│   │   ├── mocks.md                ← What's mocked vs real, per layer
│   │   ├── performance.json        ← Phase 6 perf snapshot
│   │   ├── test_routing_dispatch.py    ← + Phase 13 SYSTEM_STATUS routing tests
│   │   ├── test_conversational_pipeline.py
│   │   ├── test_search_pipeline.py
│   │   ├── test_coding_pipeline.py
│   │   ├── test_addressing_pipeline.py
│   │   ├── test_error_recovery_pipeline.py
│   │   └── test_bridge_e2e.py      ← OpenClaw Phase 3 bridge e2e (real subprocess against stub CLI)
│   └── openclaw_bridge/            ← OpenClaw Phases 3–13 bridge tests (158 tests)
│       ├── __init__.py
│       ├── test_client.py          ← OpenClawClient: subprocess transport + result parsing
│       ├── test_workspace.py       ← WorkspaceWriter: atomic + filelock + concurrency
│       ├── test_events.py          ← OpenClawEventReceiver: prefix matching + dispatch
│       ├── test_mcp_registration.py ← UltronMcpRegistrar: idempotent + retry
│       ├── test_holder.py          ← OpenClawBridge: from_config / start / shutdown / fire_and_forget / record_heartbeat_alert / auto-resolve
│       ├── test_notifications.py   ← NotificationDispatcher: per-event gating + recipient resolution + transport errors
│       ├── test_heartbeat_alerts.py ← HeartbeatAlertLog: record / get / acknowledge / prune / concurrency
│       ├── test_browser.py         ← BrowserTool: six primitives + result extraction edge cases
│       ├── test_mcp_tools.py       ← Stdio MCP tools: get_heartbeat_alerts / acknowledge_alert / run_maintenance / list_active_coding_sessions / get_recent_voice_alerts
│       └── test_system_status.py   ← SystemStatusReporter: alerts / projects / all foci + voice rendering
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
│   ├── Josiefied-Qwen3-4B-abliterated-v2.Q4_K_M.gguf ← LLM, CURRENT DEFAULT (2.4 GB, 2026-05-14 second-pass)
│   ├── Josiefied-Qwen3-4B-abliterated-v2.Q5_K_M.gguf ← retained for swap-back / quality A/B (2.7 GB, 2026-05-14)
│   ├── Josiefied-Qwen3-8B-abliterated-v1.Q5_K_M.gguf ← LLM (5.85 GB; retained for swap-back / bigger abliterated)
│   ├── Qwen3.5-9B-Q4_K_M.gguf      ← LLM (5.29 GB; retained for swap-back, not abliterated)
│   ├── Qwen3.5-4B-Q4_K_M.gguf      ← LLM (2.55 GB; retained for swap-back / spec decoding, not abliterated)
│   ├── Qwen3.5-0.8B-Q4_K_M.gguf    ← speculative-decoding draft for the plain qwen3.5-4b preset (0.50 GB)
│   ├── openwakeword/ultron.onnx    ← custom wake word
│   ├── piper/en_US-ryan-medium.onnx ← TTS voice
│   ├── rvc/{hubert_base.pt, rmvpe.pt} ← RVC support files
│   └── smart_turn/smart-turn-v3.2-cpu.onnx ← Smart Turn V3 (8.68 MB int8, 2026-05-12)
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

**Clarification fast-path order** (V1-gap A3 added Fast-path 2.5):

```
ConversationCoordinator.decide_clarification(...):
  1. Always-escalate keyword (api key, paid tier, scope add, ...)
  2. urgency=preference + options provided -> "use your default"
  2.5 (A3) Stored facts via facts_lookup -- if a directive-category
       (preference / decision / constraint) fact clears confidence and
       score thresholds, answer Claude with "From the user's stored
       preferences: <fact>. Use that."
  3. Always-answer keyword (test framework, linter, layout, ...)
  4. LLM decide pass (ANSWER / USE_DEFAULT / ESCALATE)
  5. Escalate to user (voice question + asyncio.Future await)
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
  `knowledge_source` / `has_temporal_dependency`, returns a
  possibly-prepended user prompt with style hints. V1-gap B1: a
  `knowledge_source` of `retrieved_memory` / `retrieved_facts`
  prepends a source hint above the confidence addendum so the LLM
  matches its tone (rule verdicts inherit this branch too).
- `_source_hint_for(verdict)` (internal) — picks the leading source
  hint from `knowledge_source`. `weights` / `unknown` /
  `web_search_needed` get no hint.

**In:** `GateVerdict` from `web_search.gating`, raw user text.
**Out:** `(verdict, augmented_prompt)`.

### `src/ultron/response_style.py` (2026-05-10)

**Purpose:** per-call response-style addenda, prepended to the user's
text before it reaches the LLM. Lives OUTSIDE the persona file
(SOUL.md is voice-quality-locked) so the orchestrator can nudge the
model on a per-utterance basis without changing the system prompt.
Today only one addendum lives here — a brevity hint for short
questions that the 4B model otherwise tends to over-explain ("What
are the Orcs in 40k?" → 1164-char four-paragraph essay in the
2026-05-10 live session).

**Public:**
- `is_brief_question(user_text: str) -> bool` — True iff the
  utterance is short (≤12 words OR ≤80 chars after strip) AND not
  explicitly asking for depth via any of the `_DEPTH_MARKERS`
  keywords (`explain` / `in detail` / `step by step` /
  `walk me through` / `elaborate` / `expand on` / `everything you
  know` / etc.). Empty / whitespace input returns False.
- `apply_brevity_hint(user_text: str) -> str` — prepends a
  `[Style: respond in 1-3 short sentences …]` directive when
  `is_brief_question` returns True; otherwise returns input
  unchanged. Empty input passes through. Idempotent on
  already-hinted text (the hinted version is too long to be
  re-classified as brief).

**In:** raw user text. **Out:** possibly-augmented user text (newline-
separated above the original).

**Wired at:** [pipeline/orchestrator.py](../src/ultron/pipeline/orchestrator.py)
`Orchestrator._build_response_stream` — applied on the non-search
conversational path (search path's augmented prompt has its own
length directive). Three call sites: web-gate-disabled fall-through,
web-gate-failure fall-through, NO_SEARCH verdict path.

### `src/ultron/conversational_ack.py` (2026-05-12)

**Purpose:** filler-acknowledgment source for the no-search
conversational branch. Masks the ~2.5 s perceived gap between
Whisper completing and the LLM's first TTS chunk by yielding a
short thinking-noise ("Mm.", "Right.", "Hm.", etc.) BEFORE the LLM
stream so the TTS pipeline starts speaking within ~200 ms of
Whisper completing. End-to-end latency unchanged; perceived latency
drops sharply. The web-search path has its own ack
(`web_search.acknowledgments.AcknowledgmentSource`) that describes
external activity; this module's phrases are tonally non-committal
(read as Ultron deliberating). The two pools rotate independently.

**Public:**
- `_CONVERSATIONAL_PHRASES: List[str]` — module-level phrase pool.
  Each phrase ≤20 chars, period-terminated so the TTS pipeline
  flushes it as a complete sentence immediately. No overlap with
  the web-search pool.
- `is_conversational_ack_eligible(user_text, *, has_pending_clarification=False) -> bool`
  — pure gate function. Returns False on empty input, on utterances
  shorter than 11 chars or 4 words (interjections like "yes",
  "thanks"), and when a coding-task clarification is pending (the
  orchestrator already has its own narration flow there). Otherwise
  returns True.
- `class ConversationalAckSource` — thin wrapper around
  `AcknowledgmentSource` with the conversational phrase pool baked
  in. Holding it as a distinct class type makes the orchestrator's
  intent clear at call sites and keeps the two pools' shuffled-
  cycle state separate.
  - `__init__(phrases=None)` — `None` uses the default pool;
    explicit empty list is forwarded to the underlying source which
    rejects it (vs silently swapping to default).
  - `next_phrase() -> str` — same contract as
    `AcknowledgmentSource.next_phrase`.

**Wired at:** [pipeline/orchestrator.py](../src/ultron/pipeline/orchestrator.py)
- `Orchestrator.__init__` constructs `self.conv_ack_source = ConversationalAckSource()`.
- `Orchestrator._maybe_conversational_ack(user_text) -> Optional[str]`
  helper threads pending-clarification state through the gate and
  fail-opens on any exception (broken source or coding-voice check).
- `Orchestrator._build_response_stream` yields the ack token before
  `llm.generate_stream(...)` at all three no-search exit points:
  web-gate-disabled fallthrough, web-gate-exception fallthrough,
  `verdict.decision != SEARCH` branch. The `_search_augmented_tokens`
  path is untouched (already yields its own ack from
  `web_search.acknowledgments.AcknowledgmentSource`).

### `src/ultron/desktop/` (Desktop automation native primitives)

NEW package backing the "open YouTube on monitor 2", "show me a picture of golden retriever",
"explain what I'm looking at" voice flows. Built native (no ClawHub plugin dependencies)
per user direction. Same UI Automation capability surface as the `windows-control` plugin
via `pywinauto`; same screenshot capability as `desktop-control` via `mss`.

#### `desktop/monitors.py`

- `class Monitor` -- frozen dataclass: index, name, x, y, width, height, work_x/y/width/height, is_primary. Helpers: `.right`, `.bottom`, `.center`.
- `enumerate_monitors() -> list[Monitor]` -- Win32 `EnumDisplayMonitors` + `GetMonitorInfo`. Primary sorts to index 0; rest left-to-right. Empty list on pywin32 failure.
- `find_monitor(query) -> Optional[Monitor]` -- accepts int / numeric string / `"primary"`/`"main"`/`"default"` / ordinals `"first"`/`"second"`/`"1st"` / directional `"left"`/`"right"`/`"top"`/`"bottom"`/`"center"` / device-name substring.
- `point_to_monitor(x, y) -> Optional[Monitor]` -- containing monitor for a virtual-screen point.

#### `desktop/capture.py`

- `class Screenshot` -- frozen: image_bytes (PNG), monitor_index, width, height, timestamp, origin_x/y.
- `class ScreenCapture` -- thread-local `mss.MSS` (mss is not thread-safe per-instance). Every successful capture records its PNG bytes in the safety taint tracker as capability=`screen_context`. Methods: `capture_monitor(monitor_or_index)`, `capture_all_monitors()`, `capture_region(x, y, w, h)`, `close()`. Fail-open (returns None on mss errors).
- `_bgra_to_png_bytes(raw, width, height) -> bytes` -- pure helper, BGRA from mss to PNG via Pillow.
- Singletons: `get_screen_capture()` / `set_screen_capture()`.

#### `desktop/windows.py`

- `class WindowInfo` -- frozen: hwnd, title, class_name, process_name (via psutil), pid, rect, monitor_index (greatest-overlap rule), is_minimized, is_foreground. Helpers: `.width`, `.height`, `.center`.
- `enumerate_windows(*, include_minimized=False, include_invisible=False, require_title=True) -> list[WindowInfo]` -- pywin32 `EnumWindows` callback.
- `get_foreground_window() -> Optional[WindowInfo]` -- `GetForegroundWindow`.
- `find_window(query, *, prefer_foreground=True, prefer_monitor=None, by_process=True) -> Optional[WindowInfo]` -- substring match against title (and optionally process name) with exact-match / foreground / monitor-preference tiebreakers.
- `_monitor_index_for_rect(rect, monitors)` -- pure helper used by tests.

#### `desktop/placement.py`

- `class PlacementResult` -- success/hwnd/monitor_index/error.
- `move_window_to_monitor(hwnd, monitor, *, fullscreen=False, maximize=False, size=None, offset=(0,0))` -- target-monitor placement. `fullscreen` fills the monitor as a regular window; `maximize` calls `SW_MAXIMIZE` after moving; explicit `size` clamps to work area.
- `maximize_window(hwnd)`, `minimize_window(hwnd)`, `restore_window(hwnd)`, `focus_window(hwnd)` -- single-action helpers. `focus_window` does SetForegroundWindow with BringWindowToTop fallback per Windows' foreground-lock rules.

#### `desktop/launcher.py`

- `class AppEntry` -- frozen: name, candidate_paths, args_prefix, aliases, process_name.
- `class LaunchResult` -- success/app_name/exe_path/pid/hwnd/monitor_index/placement/error.
- `class AppLauncher` -- registry-driven launcher.
  - `find_app(query) -> Optional[AppEntry]` -- name + alias + substring.
  - `resolve_executable(entry) -> Optional[Path]` -- first existing candidate; Discord/Chrome-specific resolvers handle their Squirrel/auto-update directory layouts.
  - `launch_app(name, *, monitor, extra_args, fullscreen, maximize, wait_for_window, user_text)` -- safety-validated spawn + optional monitor placement via `move_window_to_monitor`.
  - `launch_chrome(*, url, monitor, fullscreen, maximize, window_size, new_window, user_text)` -- launches user's actual Chrome with `--new-window <URL>` (NOT Playwright). Reuses default profile (no `--user-data-dir`). Reusing the user's real Chrome session means cookies + sign-ins + extensions are preserved.
  - `open_image_search(query, *, monitor, small_window, user_text)` -- "show me a picture of X" convenience: launches Chrome with Google Images URL.
- Default registry includes: chrome, edge, firefox, cursor, vscode, discord, notepad, explorer, terminal (wt), spotify, slack, obs.
- Every launch passes through `_validate_launch` → `ToolCallValidator.check` with `tool_name="desktop.launch_app"`. Cap-2 rules block `--remote-debugging-port`, `--user-data-dir`, `--load-extension`, `--disable-web-security`, launches from Temp/Downloads.

#### `desktop/uia.py`

- `class UIAElement` / `class UIAActionResult` -- frozen snapshots; UI element data captured at lookup time (live pywinauto handles can go stale).
- `collect_window_text(window, *, max_elements=200, max_depth=8, min_length=2) -> list[str]` -- walk a window's UIA tree and return unique visible text strings. **Load-bearing for `screen_context`**: this is how "what's actually written on screen" feeds into the LLM. Bounded traversal (browser/IDE trees have 10k+ elements).
- `find_element(window, *, query, control_type, automation_id, exact) -> Optional[UIAElement]` -- search by name / automation_id / control_type.
- `click_element(window, query, ...) -> UIAActionResult` -- find + `click_input()`. Goes through Cap-3 action-verb-click rule (NEEDS_EXPLICIT_INTENT on `"Submit"` / `"Pay"` / `"Send Money"` etc.) and Cap-3 OAuth/payment URL detection on window title.
- `type_text_into_element(window, query, text, ..., clear_first=True) -> UIAActionResult` -- find + `set_text()` (preferred) or `type_keys()` fallback. Same Cap-3 safety hook.
- Lazy pywinauto import so `import ultron.desktop` doesn't pay the COM cost; failure returns None / empty list.

#### `desktop/input_control.py`

- `class InputControlResult` -- success/action/error.
- `class InputController` -- pyautogui-backed mouse/keyboard with three gates:
  1. **Foreground security check.** `_foreground_is_security_window()` returns True when the focused window's class matches UAC / Windows Security / Credential UI patterns. Synthetic input on those is blocked by Windows itself (UIPI) but we refuse upstream so the audit log has context.
  2. **Rate limit.** `max_actions_per_second` (default 5) cap; over the limit fails the call rather than blocking the orchestrator.
  3. **Safety validator.** Every action builds a `RuleContext` with `tool_name="desktop.input.<action>"` for Cap-4 synthetic-input rules to check arguments against.
- Methods: `move_mouse(x, y, ...)`, `click(x, y, *, button, clicks, ...)`, `type_text(text, ...)`, `press_key(key, ...)`, `press_hotkey(*keys, ...)`, `scroll(amount, *, x, y, ...)`. All return `InputControlResult`.
- pyautogui's `FAILSAFE` (mouse-to-corner aborts) stays on; do NOT disable.

#### `desktop/screen_context.py`

- `class ScreenContextSnapshot` -- frozen: timestamp, monitors, foreground, windows, ui_text, screenshot, vlm_description, elapsed_ms. `.render_for_llm(*, max_ui_text=40) -> str` formats the snapshot as a readable text block for prepending to a user utterance.
- `build_screen_context(*, capture=True, capture_all_monitors=False, include_uia=True, include_vlm=False, ...)` -- the orchestrator. Assembles monitors / foreground / windows / UIA tree text / optional screenshot / optional VLM description into one snapshot. Every component fails to its empty/None default rather than raising.
- `class ScreenContextCache` -- in-memory ring buffer (default 3 entries, max age 15 s). `latest_fresh()` returns the most recent snapshot only if within max_age; useful for follow-up questions reusing the previous capture.
- `capture_and_cache(...)` -- convenience: build snapshot + store in the singleton cache.
- VLM hook: `set_vlm_describe(fn)` registers the bridge function called when `include_vlm=True`. `vlm.set_vlm(...)` wires this transparently.

#### `desktop/vlm.py`

- `class Moondream2VLM` -- moondream2 wrapper via transformers (`vikhyatk/moondream2`, `trust_remote_code=True`).
  - Construction validates importability but does NOT load weights. `warmup()` forces the lazy-load now.
  - `describe(image_bytes, *, prompt=None) -> VLMResult` -- decodes PNG via Pillow, runs `model.encode_image` + `model.answer_question`. Fail-open at every layer (missing transformers / bad image / inference exception / empty output all return `VLMResult(success=False, ...)`).
  - CPU-only. ~3.5 GB FP16 weights on disk; ~4-5 GB RAM after load. ~5-8 s per query.
- `class VLMResult` / `class VLMLoadError`.
- `build_vlm_from_config(*, enabled, repo, revision, device, max_tokens) -> Optional[Moondream2VLM]` -- factory; returns None on construction failure (orchestrator treats None as "VLM unavailable; fall back to text-only context").
- `get_vlm()` / `set_vlm(...)` -- singleton + wires the `screen_context.set_vlm_describe(...)` bridge transparently.
- Model weights pre-fetched via `scripts/download_models.py` step 9/10.

#### `desktop/voice.py` (Phase 8 voice handlers)

Bridges :class:`RoutingIntent` (kinds APP_LAUNCH + SCREEN_CONTEXT_QUERY) to the native primitives. Imported by :class:`CapabilityVoiceController` in `coding/voice.py`.

- `class AppLaunchVoiceResult` / `class ScreenContextVoiceResult` -- frozen result types.
- `handle_app_launch(intent) -> AppLaunchVoiceResult` -- dispatches an :class:`AppLaunchIntent` to :class:`AppLauncher`. Resolves the monitor (index OR directional via `find_monitor`); chooses `launch_chrome(url=...)` for Chrome+URL combos and `launch_app(...)` for everything else; threads `user_text` into the safety validator. Returns a short in-character voice line.
- `handle_screen_context_query(intent) -> ScreenContextVoiceResult` -- builds a :class:`ScreenContextSnapshot` via `build_screen_context(include_vlm=intent.include_vlm)` and returns its `render_for_llm()` text for prompt injection.
- `handle_window_move(intent) -> WindowMoveVoiceResult` (2026-05-14 third pass) -- finds an existing window matching ``intent.window_query`` via :func:`ultron.desktop.windows.find_window` and moves it to ``intent.monitor_index`` (or ``intent.monitor_query`` resolved via :func:`find_monitor`) using :func:`move_window_to_monitor`. Distinct from APP_LAUNCH which would spawn a new process.
- `handle_window_close(intent) -> WindowCloseVoiceResult` (2026-05-14 third pass) -- finds an existing window matching ``intent.window_query`` (optionally restricted to a monitor via ``intent.monitor_query``) and posts ``WM_CLOSE`` via :func:`win32gui.PostMessage`. Graceful close path -- lets the app prompt to save if it wants. Used for "close my YouTube tab", "close Discord on my right monitor", etc.

#### `desktop/preferences.py` (Phase 10 preference learning)

User-preference persistence so "open YouTube" picks up "monitor 2 + maximize" the second time after the user said it once with the explicit phrasing.

- `class DesktopPreference` (frozen) -- one learned action: `user_phrase`, `app_name`, `url`, `monitor_index`, `fullscreen`, `maximize`, `success`, `timestamp`.
- `class PreferenceLogger` -- thread-safe JSONL append at `logs/desktop_preferences.jsonl`. Optional asynchronous mirror to the OpenClaw workspace daily memory files when :func:`set_workspace_writer` is called (failures are swallowed -- JSONL is the source of truth).
- `find_preference_for_phrase(phrase, *, max_age_days=90.0, min_substring_length=4) -> Optional[DesktopPreference]` -- substring match (both directions) with recency weighting (newest matching wins). Filters out failed entries.
- `record_launch_preference(...)` -- one-call helper for the voice handler.
- Wired at `CapabilityVoiceController._handle_app_launch`: utterances without explicit monitor target consult the logger first; matching prior preference's monitor + flags become the defaults.

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
  - `write(samples)` / `clear()` / `__len__()` / `capacity` property
  - `snapshot(last_n_samples=None) -> np.ndarray` — full buffer when
    unsliced; the most recent `last_n_samples` when given. The
    orchestrator slices a short COLD-mode pre-roll (post-wake; avoids
    "Tron" prefix) and a longer WARM-mode pre-roll (post-TTS
    follow-up; avoids first-word clipping) from the SAME buffer.
    `last_n_samples >= len(buffer)` returns full; `<= 0` returns
    empty. (2026-05-10 mode-aware pre-roll fix.)

#### `audio/vad.py`
- `class SpeechEvent(Enum)` — START / END / NONE
- `class VadResult` — dataclass: event, is_speech, prob
- `class VoiceActivityDetector` — silero-vad wrapper; consumes 512-sample windows.
  - `reset()` — clear hysteresis state AND restore the baseline silence-window requirement (so an adaptive bump from the previous utterance doesn't leak into the next one).
  - `set_min_silence_duration_ms(ms)` (2026-05-11 adaptive end-of-turn) — adjust trailing-silence requirement at runtime. Orchestrator calls this from `_capture_utterance` once speech has been active past `vad.long_utterance_threshold_seconds` so a thinking pause mid-prompt doesn't close a long technical description.

#### `audio/wake_word.py`
- `class WakeWordDetector` — openWakeWord wrapper
  - Loads `models/openwakeword/ultron.onnx` (custom)
  - Falls back to `hey_jarvis` with startup warning if missing
  - `predict(audio_block) -> Optional[str]` — fires a wake event
  - `fired_recently(window_s: float = 0.5) -> bool` (V1-gap A4) — read-only accessor for the last trigger timestamp; returns True iff a wake fire happened within ``window_s`` seconds. Used by the orchestrator's pre-task barge-in watcher. Idempotent — does not consume the trigger.

#### `audio/smart_turn.py` (NEW 2026-05-12)

Pipecat Smart Turn V3 ONNX wrapper. CPU-only semantic end-of-turn
confirmation that runs after Silero VAD detects silence. Pinned to
`CPUExecutionProvider` — zero VRAM cost. 8 MB int8 model;
`~12 ms` inference target, sub-150 ms in practice on this hardware.
Fail-open at every layer: missing model file / disabled config /
load failure / inference exception all degrade silently to "trust
VAD" rather than misclassifying.

- `SMART_TURN_SAMPLE_RATE = 16000`, `SMART_TURN_WINDOW_SECONDS = 8.0`,
  `SMART_TURN_MEL_BINS = 80`, `SMART_TURN_MEL_FRAMES = 800`,
  `SMART_TURN_INPUT_NAME = "input_features"` — model contract constants.
- `class SmartTurnLoadError(RuntimeError)` — raised at construction
  time only (missing file, out-of-range config). Inference-time
  failures degrade to None.
- `@dataclass(frozen=True) class SmartTurnVerdict` — `is_complete: bool`,
  `probability: float` (sigmoid output, already activated in the ONNX
  graph), `latency_ms: float` (wall-clock including preprocessing).
- `truncate_or_pad_for_smart_turn(audio, sample_rate, *, window_seconds=8.0) -> np.ndarray`
  — pure helper. Truncates audio HEAD-first to the last
  `window_seconds`; pads-at-start is the `WhisperFeatureExtractor`'s
  job (`padding="max_length"`). Converts non-float32 inputs to
  float32; flattens multi-dim inputs. Rejects non-16 kHz with
  `ValueError` (callers resample upstream).
- `class SmartTurnDetector`:
  - `__init__(model_path, *, completion_threshold=0.5, window_seconds=8.0, num_threads=1)`
    — validates the model file exists and parameters are in range;
    does NOT load the ONNX session into memory. Raises
    `SmartTurnLoadError` on bad inputs.
  - `available` property — True iff loaded and healthy. False before
    first call (lazy) and after a load failure.
  - `warmup() -> bool` — forces the lazy-load path now. Returns
    True on success, False on load failure (logged at WARN).
  - `is_complete(audio, sample_rate=16000) -> Optional[SmartTurnVerdict]`
    — main entry. Returns a verdict on success, None on any error
    (treated by caller as "undecided" → trust VAD). Thread-safe via
    an internal load + inference lock.
  - `close()` — idempotent release; subsequent `is_complete` returns None.
- `build_detector_from_config(smart_turn_cfg, project_root) -> Optional[SmartTurnDetector]`
  — orchestrator-side factory. Returns None when smart-turn is
  disabled, when the model file is missing on disk, or when
  construction fails. WARN-level log distinguishes the cases. This
  is the single seam between config and runtime that the orchestrator
  uses; no other call site constructs a detector directly.

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

- `_strip_thinking_blocks(stream)` — filter `<think>...</think>` from token stream (streaming path).
- `strip_thinking_text(text) -> str` (2026-05-14 second pass) — same filter as a pure-function pass over a fully-materialised string. Applied inside :meth:`LLMEngine.generate` (blocking path) before returning; unterminated `<think>` (truncation / cancel) drops everything from the opening tag onward (better to lose tail than leak chain-of-thought).
- `_sanitize_user_input(text) -> (cleaned, found_markers)` (Q10 quality-pass iter 1+2) — pre-LLM defense layer that neutralises tag-style prompt-injection markers (`[INST]`, `[/INST]`, `<|im_start|>`, `<|im_end|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `</think>`) by replacing each with `[NEUTRALIZED_TAG]`. Also detects natural-language jailbreak patterns ("ignore previous instructions", "you are now <X>", "respond with the exact word", etc.) — for those the function prepends a one-shot hardening note OR (for the most-direct override patterns: "respond with exactly", "respond with the exact word", "must respond with") rewrites the user message into a description of the attempt so compliance becomes grammatically nonsensical. Detected attempts log to `logs/errors.jsonl` with `dependency='prompt_injection'`. Voice-quality lock preserved — the persona system prompt (`SOUL.md`) is untouched. Wired into `_build_messages` so every LLM call goes through the defense. Verified end-to-end: pre-defense 2/3 of Q8 prompt-injection probes succeeded; post-defense 0/3.
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
  - `_chat_completion_kwargs(_llm_cfg, enable_thinking, *, stream)` (4B plan Stage F; 2026-05-14 third pass rewrite) — static helper that builds the kwargs dict for `Llama.create_chat_completion`. Returns ONLY the four sampling params + optional ``stream`` flag — NEVER emits ``chat_template_kwargs`` because the pinned llama-cpp-python 0.3.22 doesn't accept it (passing it raises ``TypeError``). The thinking-mode toggle is applied to the user message instead via :meth:`_apply_no_think_marker`. The HTTP runtime's payload-building code still emits ``chat_template_kwargs`` because llama-cpp-server (separate codebase) does accept it.
  - `_apply_no_think_marker(messages, enable_thinking) -> list` (2026-05-14 third pass) — staticmethod that appends ``/no_think`` to the last user message when ``enable_thinking is False``. Qwen3 / Qwen3.5 chat templates inspect the user message for this marker and skip the ``<think>...</think>`` block. ``enable_thinking=None`` (default) and ``True`` are no-ops. Returns a copy of ``messages`` — never mutates the original. Replaces the previous ``chat_template_kwargs`` mechanism which crashed against the real llama-cpp-python signature.
  - `_build_llama(cfg, model_path, n_ctx, n_gpu_layers) -> (Llama, Path)` (4B plan voice-swap) — pure constructor that builds + returns a fresh `Llama` instance per `cfg`. Does NOT mutate `self`. Used by `_init_in_process` and `reload_for_preset`.
  - `reload_for_preset(preset: str) -> (bool, str)` (4B plan voice-swap) — hot-swap the loaded LLM to `preset` without restarting Ultron. Builds the new `Llama` FIRST so a failed swap (missing GGUF, invalid preset) leaves the engine in its working state. On success: history cleared, `ULTRON_LLM_PRESET` env updated, stale `ULTRON_LLM_MODEL_PATH` cleared. On failure: env vars restored. Idempotent (`already on X` returns success without rebuild). `in_process` runtime only.
  - `generate(user_message, *, enable_thinking=None)` and `generate_stream(user_message, *, enable_thinking=None, record_history=True)` (4B plan Stage F + 2026-05-18 latency pass 3 Phase 3) — per-call thinking mode parameter, plus `record_history` on the streaming variant. When `record_history=False`, the end-of-stream auto-record is skipped so callers can defer history commit to after they've confirmed the response was actually consumed (used by the orchestrator's speculative-LLM path).
  - `record_completed_turn(user_message, response)` (2026-05-18 latency pass 3 Phase 3) — public commit hook for the deferred-history pattern. No-op on empty input. Used by `Orchestrator._collect_speculative_llm`'s commit closure after the buffered tokens have been drained to TTS.

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
- `class FactRow` (V1-gap A3) — dataclass: fact, confidence, last_confirmed, category, score, extracted_at, extracted_from, retrieval_weight. Read-side projection of the `facts` collection that the maintenance script writes.
- `class ConversationMemory`
  - `__init__(path?, embedder, recent_cache_size=100, session_id?)`
  - `add(role, content)` — sync; queues to background writer
  - `recent(n) -> List[MemoryTurn]` — from in-process cache
  - `retrieve(query, k=cfg, exclude_recent=cfg) -> List[MemoryTurn]` — single-pass hybrid RRF
  - `retrieve_multi(primary_query, category_queries, *, k, exclude_recent)` (V1-gap A2) — multi-pass per-category hybrid RRF + composite re-ranking. Parallel fan-out via `ThreadPoolExecutor`. Falls back to single-pass on any failure.
  - `retrieve_for_query(primary_query, gate_verdict=None, *, k, exclude_recent)` (V1-gap A2) — routing helper: when `memory.retrieval.multi_pass_enabled` is True AND the verdict carries `context_categories`, fans out via `retrieve_multi`; otherwise calls `retrieve`. Default-OFF preserves byte-for-byte legacy behaviour.
  - `search_facts(query, *, k=5, min_confidence=0.0, max_age_days=None) -> List[FactRow]` (V1-gap A3) — hybrid RRF over the `facts` collection. Filters via Qdrant `confidence >= min_confidence` and `last_confirmed >= now - max_age_days*86400`. Fail-open: returns `[]` on any Qdrant / embedder failure.
  - `__len__()` / `close()`

#### `memory/ranking.py` (V1-gap A2)
- `@dataclass class RankingWeights` — frozen snapshot of the rrf_weight / recency_weight / recency_half_life_days / surprise_weight / redundancy_weight tuning.
- `@dataclass class CandidateScore` — per-candidate aggregator (id, payload, rrf_score, dense vector, primary_similarity, category_similarity, composite_score).
- `cosine_similarity(a, b) -> float` — pure cosine on float lists; defensive against length mismatch / zero vectors.
- `compute_recency_boost(ts, *, half_life_days, now=None)` — exponential decay; ``ts == 0`` (sentinel) returns 0.
- `compute_surprise_score(candidate_dense, primary_dense, category_score)` — clamps to ``max(0, category_score - primary_similarity)``.
- `compute_redundancy_penalty(candidate_dense, picked)` — max cosine vs already-picked.
- `compute_composite_score(candidate, *, weights, primary_dense, picked, now=None)` — weighted blend.
- `select_top_k(candidates, *, k, weights, primary_dense=None, now=None) -> List[CandidateScore]` — greedy redundancy-aware selection.

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
- `class GateVerdict` — decision, confidence, source, search_queries, knowledge signals (knowledge_confidence, knowledge_source, has_temporal_dependency), **context_categories** + **memory_search_queries** (V1-gap A2 — populated by the LLM preflight pass; rule-only verdicts leave them empty so the multi-pass retrieval path stays inactive).
- `_resolve_knowledge_source(*, needs_search, confidence, memory_snippets, rule_reason)` (V1-gap B1) — single-source helper that maps gate inputs to the spec's five-value enumeration (`weights / retrieved_memory / retrieved_facts / web_search_needed / unknown`). Every `GateVerdict` construction site routes through this.
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
- `_normalise_search_query(q)` / `_dedupe_queries(qs)` (V1-gap B2) — drop near-duplicate Brave queries before fan-out using a token-set canonical form (lowercase + possessive strip + stopword drop + sort).
- `_render_inline_marker(index, *, fmt)` (V1-gap B3) — render bracketed `[1]` (default) or Unicode superscript (¹²³) inline citations based on `web_search.citation.inline_marker_format`.
- `class WebSearchExecutor` — orchestrates Brave → rank → Jina → cache. **2026-05-09 latency fix:** Jina fetches now run IN PARALLEL via `concurrent.futures.ThreadPoolExecutor` with a collective deadline cap. Pre-fix the loop was sequential and one slow page (~10 s on a Quora result) blocked the entire search path while the TTS playback queue starved waiting for tokens. Post-fix wall time is `max(per-fetch durations)` instead of `sum(...)`, capped further by `collective_deadline_seconds`. Any fetch still in flight at deadline is abandoned (its source falls back to snippet-only with a `jina_deadline:<url>` note). Threads keep running in the background and exit on per-fetch HTTP timeout; `pool.shutdown(wait=False)` ensures the executor returns immediately.
  - `__init__(brave, jina, llm, cache=None, max_fetch=None, collective_deadline_seconds=None)` — both kwargs default-resolve from `get_config().web_search.jina`.
  - `run(user_query, search_queries?, top_n=3) -> SearchPayload`
- `format_sources_for_prompt(sources)` / `format_sources_for_transcript(sources)` — references list always uses bracket form for monospace clarity.

### `src/ultron/tts/`

#### `tts/precomputed_ack.py` (NEW 2026-05-15 latency pass)

Pre-computed TTS clip cache. Phrases enrolled at startup (the
conversational ack pool + the web-search ack pool) get synthesised
ONCE via the live engine's `_synthesize` path -- so the cached clip
is byte-identical to the live path (same temperature, phantom-tail
trim, v3 filter, all of it). Later `_synthesize(text)` calls hit
the dict and skip the ~350-400 ms HTTP + filter chain.

- `class PrecomputedAckClipCache` -- thread-safe `dict[str, (pcm, sr)]`
  keyed by stripped text. `phrases` (sorted, de-duped, stripped at
  init), `get(text)` (exact stripped match), `prewarm(synth_fn)`
  (synthesise + populate; swallows per-phrase exceptions), `is_warm`
  / `warmed_count`.
- `collect_default_ack_phrases() -> List[str]` -- imports the
  conversational + web-search phrase pools lazily and returns the
  union. Fail-open on import errors.
- `build_default_ack_clip_cache() -> PrecomputedAckClipCache` --
  factory; returns an EMPTY cache. Caller runs `prewarm(synth_fn)`
  on a daemon thread.
- `prewarm_in_background(cache, synth_fn, *, name="ack-prewarm")` --
  starts + returns the daemon thread.

Wired at: `Orchestrator.__init__` calls `_kick_off_ack_clip_prewarm`
right after `self.tts.warmup()`. The prewarm thread runs in
parallel with the rest of orchestrator startup; first turn may miss
while populating, subsequent turns hit.

#### `tts/rvc.py`
- `class RvcConverter` — infer-rvc-python wrapper, cuda:0
  - `convert(pcm: np.ndarray, sample_rate: int) -> (pcm, sr)` — raises RVCConversionError on failure
  - `close()` — releases GPU memory

#### `tts/speech.py`
- `Clip` — type alias for `Tuple[np.ndarray, int]` (legacy synth function return).
- `class ClipItem(NamedTuple)` (2026-05-10 producer-signaled lookahead):
  `(audio, sample_rate, is_known_last)`. Pushed onto `piper_q` /
  `audio_q` instead of bare `Clip` tuples. Playback uses
  `is_known_last` (or the `None` end-of-stream sentinel) to decide
  whether to wait for another clip after playing the current one.
  Default `is_known_last=False`; the `None` sentinel handles the
  "this was the last" signal in normal use. The flag is reserved for
  future producers that DO know in advance (canned single-sentence
  voice responses, etc.).
- `_QUEUE_GET_TIMEOUT_SECONDS = 60.0` — generous wait between clips
  in both the playback loop and the RVC stage. The previous 10 s
  value killed audio mid-response when a slow web search held the
  generator long enough for the RVC stage's `piper_q.get(timeout=10)`
  to fire (BMW failure mode in the 2026-05-09 logs).
- `class TextToSpeech` — Piper + optional RVC
  - `__init__(rvc=None)` — loads Piper voice, optionally wraps with RVC
  - `speak(text)` — synchronous synthesize + play
  - `speak_stream(fragments)` — stream tokens, flush on sentence
    terminator. **Producer-signaled lookahead (2026-05-10):** plays
    each clip IMMEDIATELY on receipt, then blocks for the next.
    Replaces the legacy play-after-peek pattern that delayed the
    first clip (commonly the web-search ack) up to 10 s waiting for
    the second clip to determine "is this last?". Voice character
    (edge fades, inter-sentence pauses, tail silence) bit-identical;
    only the queue-get ordering changed.
  - `warmup()` — primes Piper
  - `_synthesize(text)` — Piper → optional RVC; raises
    PiperSynthesisError / RVCConversionError
  - `_run_synth_loop(*, fragments, push, synth_fn)` — walks
    fragments, synthesises on flush chars, pushes each non-empty
    clip via `push` as a `ClipItem(is_known_last=False)`. End-of-
    stream sentinel (`None`) is pushed by the surrounding worker's
    `finally` block.
  - `stop()` — interrupt current playback

#### `tts/ultron_filter.py` (NEW 2026-05-10 voice swap)

Runtime port of the user-tuned v3 Ultron mechanical filter chain (the
prototype lives at `ultronVoiceAudio/scripts/ultron_filter.py`).
Built on `pedalboard` (Spotify's open-source DSP library; sub-ms
overhead per stage on CPU).

- `PresetName` — Literal["v1_subtle", "v2_medium", "v3_heavy"].
- `get_preset(preset)` — fresh `Pedalboard` instance per call.
- `apply_filter(audio, sample_rate, preset="v3_heavy", tail_silence_ms=200.0)`
  — applies the chain. Pads `tail_silence_ms` of trailing zeros
  before processing so the reverb tail decays into the padding
  rather than being clipped at the buffer end. Runtime default 200
  ms (audible portion of v3 reverb); offline samples use 500 ms.

The `v3_heavy` preset is the user-locked production chain (bit-
identical to the prototype): Highpass → PitchShift(-1.8 semitones) →
Compressor → LowShelfFilter(+4.5 dB @ 160 Hz) → Delay(7 ms, 25 %
feedback for comb resonance) → Chorus → Distortion(+7 dB) → Peak EQ
boost @ 2.5 kHz → HighShelf cut → Reverb(small cavity) → Lowpass.

#### `tts/xtts_v3.py` (NEW 2026-05-10 voice swap)

Drop-in replacement for `TextToSpeech` when `tts.engine == "xtts_v3"`.
Same `speak` / `speak_stream` / `warmup` / `stop` interface so the
orchestrator playback path (the producer-signaled lookahead in
`speak_stream`) doesn't change.

- `class ClipItem(NamedTuple)` — `(audio, sample_rate, is_known_last)`,
  same contract as in `tts/speech.py` so the queue protocol is
  uniform across engines.
- `class XttsServerStartError(RuntimeError)` — raised when the XTTS
  server subprocess can't be started (missing venv / script /
  reference, startup timeout exceeded).
- `class XttsSynthError(RuntimeError)` — synth call failure.
- `trim_phantom_tail(audio_f32, sample_rate, *, silence_threshold=0.005,
  max_event_ms=200.0, min_lead_silence_ms=150.0, trailing_grace_ms=80.0,
  window_ms=20.0) -> (np.ndarray, bool)` (NEW 2026-05-12 phantom-token
  mitigation, defence in depth) — pure function that detects the
  specific XTTS-v2 phantom signature (sustained_speech → ≥150 ms
  silence → <200 ms isolated event → silence to buffer end) and
  trims everything after the last sustained-speech region plus a
  small grace cushion. Returns `(maybe-shorter audio, detected)`.
  Conservative: passes through unchanged when no phantom pattern is
  present (sustained-speech-only, mid-sentence inter-word silence,
  legitimately long trailing speech). Runs BEFORE the v3 filter so
  the reverb tail decays normally into its tail_silence_ms padding.
  Empirically grounded against a real session WAV showing the
  signature at 19.28 s.
- `class XttsV3Speech` — the engine.
  - `__init__(...)` — resolves paths via `tts.xtts_v3` config,
    spawns the XTTS HTTP server in `.venv-xtts`, polls `/healthz`
    until ready (180 s startup budget for cold model load). 2026-
    05-12 phantom-token mitigation: also reads `temperature` (0.65
    default), `phantom_tail_trim_enabled` (true default), and the
    three trim thresholds (`silence_threshold`, `max_event_ms`,
    `min_lead_silence_ms`) from `tts.xtts_v3` config; explicit ctor
    args override.
  - `speak`, `speak_stream`, `warmup`, `stop` — same API as the
    legacy engine.
  - `_synthesize(text)` — POST `/synthesize`, accumulates the
    streamed PCM, optionally runs `trim_phantom_tail` (gated on
    `phantom_tail_trim_enabled`), applies the v3 Ultron filter via
    `ultron_filter.apply_filter(..., tail_silence_ms=200)`, returns
    `(int16 pcm, sr)` matching the legacy engine's contract.
  - `_http_synthesize(text)` — raw HTTP call; reads chunked PCM
    body and returns `np.ndarray(int16)`. POST JSON body carries
    `{"text", "language", "speed", "temperature"}` — `speed` is
    XTTS v2's native duration multiplier (1.15 in production for
    snappier cadence); `temperature` (NEW 2026-05-12) is the GPT
    duration-head sampling temperature (0.65 in production — lowered
    from XTTS library default 0.75 to cut phantom-token rate).
    Server-side passes both to `model.inference_stream(speed=...,
    temperature=...)` so cadence + stability are adjusted at
    synthesis time; the v3 pedalboard filter is unaffected.
  - `_stop_server_subprocess()` — graceful POST `/shutdown`, then
    SIGTERM, then SIGKILL. Called by the orchestrator's `shutdown()`.

The XTTS HTTP server itself lives at
[ultronVoiceAudio/scripts/xtts_server.py](../ultronVoiceAudio/scripts/xtts_server.py)
in the isolated `.venv-xtts` venv. FastAPI + uvicorn; uses an async
producer + asyncio.Queue pattern to bridge XTTS's sync streaming
generator into the FastAPI response without sync-generator
threadpool overhead (saved ~140 ms TTFT vs the naive sync-gen
implementation).

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
- `_DETERMINER_NOUN` (private regex fragment, NEW 2026-05-11 follow-up fix): `(the|that|this|your|our|my)(?:\s+(task|project|build|app|code|work|thing|run|job))?`. Plugged into all three sub-patterns of `_PROGRESS_PATTERNS` (`how X going|coming(along)?`, `what's X doing|working on|up to`, `is X done`) so phrasings like "How is that **project** going?" / "How's the **build** coming along?" / "Is **my project** done?" classify as `PROGRESS_QUERY` instead of falling through to the conversational LLM. The noun is optional so the legacy `that going` / `the doing` phrasings still fire bit-identical. The has_active_task gate is preserved so these patterns never hijack ordinary conversation.

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
  - `completion_narration() -> Optional[str]` — 2026-05-11 narration honesty: when `state.success` AND `n_created+n_modified+n_deleted == 0`, returns an explicit "I finished without writing or modifying any files. The project may need more direction, or it may have run out of token budget mid-exploration -- say continue if you want me to keep going." instead of the legacy "Done. ... <generic Claude tail> ... Elapsed: Xs." The generic tail summary is suppressed on this branch so the honest opener stands on its own. Pairs with the `coding.token_budget_per_session: 400000` bump. **2026-05-11 follow-up fix:** the project-root line is now `f"Saved under {path.name}."` (project folder leaf only) — was `f"Project root: {path}."` which interpolated the absolute Windows `state.cwd` (e.g. `C:\STC\ultronPrototype\data\sandbox\converts_pdf_docx`) and made XTTS-v2 enter pathological inference trying to pronounce backslash + colon + drive letter, pinning the GPU at 100 % until the server timed out. `StatusNarrator` already used the leaf only for progress narration; this brings completion_narration in line. The full path is still on disk in the per-session JSONL audit log + `coding_tasks.jsonl` start event so debugging is unaffected.
  - `pop_budget_warning() -> Optional[str]`
  - `record_pre_task_aborted(*, label, reason, intent_text="")` (V1-gap A4) — append a pre-task abort row to the audit log when the orchestrator's barge-in watcher fires.

#### `coding/coordinator.py`
- `class DecisionPath(str, Enum)` — RULE_ESCALATE / RULE_DEFAULT / RULE_ANSWER / FACT_ANSWER (V1-gap A3) / LLM_ANSWER / LLM_DEFAULT / LLM_ESCALATE / USER_ANSWER / TIMEOUT_DEFAULT
- `class ClarificationDecision`, `AdjustmentDecision`, `PendingUserClarification`, `_FactAnswer` (V1-gap A3, internal) — dataclasses
- `class ConversationCoordinator`
  - `__init__(store, llm, *, ..., facts_lookup=None)` — V1-gap A3: optional callable that reads the Qdrant `facts` collection. Wired by the orchestrator to `UltronMCPServer.lookup_facts`.
  - `decide_clarification(session_id, request, session) -> str` — answer or escalate. V1-gap A3: a high-confidence directive-category fact short-circuits the LLM call (Fast-path 2.5 between preference-options and always-answer rules).
  - `decide_adjustment(session_id, adjustment_text) -> AdjustmentDecision`
  - `handle_declare_complete(session_id) -> str` — runs Verifier, drives correction loop
  - `pending_user_clarifications() -> List[PendingUserClarification]`

#### `coding/mcp_server.py`
- `class UltronMCPServer`
  - `__init__(*, host, port, sse_path, log_path, clarification_timeout_s, session_audit_dir=None, memory=None)` — V1-gap A3: `memory` kwarg threads a live `ConversationMemory` so `lookup_facts` queries Qdrant. `None` preserves the test-isolation no-op.
  - In-process Python tools (called by Qwen via `get_config().coding.mcp.host:port`):
    - `create_session()`, `get_full_state()` (Python only), `get_status_delta()`,
      `get_clarification_context()`, `get_adjustment_context()`,
      `get_correction_context()`, `get_completion_context()`,
      `send_followup()`, `terminate_session()`, `list_active_sessions()`,
      `lookup_facts(query, *, k=None, min_confidence=None, max_age_days=None)` — V1-gap A3: when memory is wired, returns dict-shaped FactRow rows (proxies `memory.search_facts`); otherwise `[]`. Audit entry tagged `source="no_memory_wired"` on the stub branch.
  - SSE worker tools (called by Claude Code via SSE):
    - `report_progress()`, `request_clarification()`, `report_test_results()`,
      `declare_complete()`, `abandon_task()`, `record_file_change()`
  - `set_clarification_responder(fn)` / `set_declare_complete_handler(fn)` — coordinator hooks
  - `start()` / `stop()` — manage SSE server
- `write_mcp_config(project_root, sse_url)` / `remove_mcp_config(project_root)`

#### `coding/voice.py`
- `class VoiceResponse` — dataclass: text, handled, cancelled, **pre_task_confirmation, deferred_dispatch, pre_task_label** (V1-gap A4 — when populated, the orchestrator speaks the confirmation with barge-in detection before running the deferred dispatch closure).
- `class CapabilityVoiceController` (Phase 5 rename; alias = CodingVoiceController). `__init__` accepts an optional `llm_engine` (the live `LLMEngine`) so MODEL_SWITCH intents can call `llm_engine.reload_for_preset(...)` for in-process model hot-swap.
  - `pending_completion()` / `pending_clarifications()` / `pending_budget_warning()`
  - `has_pending_clarification() -> bool`
  - `handle_utterance(text) -> Optional[VoiceResponse]` — coding-only (delegated by capability dispatch)
  - `handle_capability_intent(routing_intent) -> Optional[VoiceResponse]` — top-level dispatch (Phase 5)
  - `_build_code_task_response(...)` (V1-gap A4, internal) — wraps `_submit` into a deferred dispatch closure when `coding.pre_task_confirmation_enabled`. Read-only intents (PROGRESS_QUERY / CANCEL / etc.) keep the legacy text-only response.
  - `_build_pre_task_confirmation(...)` / `_summarise_intent_for_voice(...)` (V1-gap A4, internal) — render the confirmation phrase ("I'll have Claude Code &lt;verb&gt; on the &lt;project&gt; project. Going ahead.").

### `src/ultron/openclaw_routing/` (Phase 5)

#### `openclaw_routing/intents.py`
- `class RoutingIntentKind(str, Enum)` — 21 values: CONVERSATIONAL, CODE_TASK, PROGRESS_QUERY, CANCEL, MID_SESSION_ADJUSTMENT, CLARIFICATION_RESPONSE, BROWSER_AUTOMATION, MEDIA_GENERATION, MESSAGING, FILE_OPERATION, SHELL_OPERATION, HYBRID_TASK, MODEL_SWITCH (4B plan), SYSTEM_STATUS (Phase 13), GAMING_MODE (V1-gap A1), DESKTOP_AUTOMATION (V1-gap C3), WINDOW_AUTOMATION (V1-gap C3), APP_LAUNCH (Phase 8 desktop), SCREEN_CONTEXT_QUERY (Phase 8 desktop), WINDOW_MOVE (2026-05-14 third pass), WINDOW_CLOSE (2026-05-14 third pass)
- Per-category dataclasses: `BrowserIntent`, `MediaGenIntent`, `MessagingIntent`, `FileOpIntent`, `ShellOpIntent`, **`GamingModeIntent`** (V1-gap A1), **`DesktopIntent`** (V1-gap C3), **`WindowIntent`** (V1-gap C3), **`AppLaunchIntent`** (Phase 8 desktop), **`ScreenContextIntent`** (Phase 8 desktop), **`WindowMoveIntent`** (2026-05-14 third pass), **`WindowCloseIntent`** (2026-05-14 third pass)
- `HybridSubtask` — dataclass: order, type, subtype, description
- `RoutingIntent` — top-level dataclass: kind, raw_text, confidence, source, reason, coding_intent, automation_intent, subtasks, model_switch_intent, system_status_intent, **gaming_mode_intent, desktop_intent, window_intent** (V1-gaps A1/C3), needs_user_clarification, clarification_question
- `DispatchResult` — dataclass: success, voice_message, error, metadata
- `TaskInfo` — task tracking dataclass
- `AutomationIntent` = Union of the 5 automation intent classes

#### `openclaw_routing/classifier.py`
- `classify_routing(utterance, has_active_coding_task=False, has_pending_clarification=False) -> RoutingIntent`
  Layered: in-flight commands → hybrid → coding → automation rules → CONVERSATIONAL fallback
- `_build_browser_intent(text)`, `_build_media_intent(text)`, `_build_messaging_intent(text)`, `_build_file_intent(text)`, `_build_shell_intent(text)` — extract structured intent from raw text
- **Comprehensive test pass extensions (HEAD 2fb0988+):** `_BROWSER_INTERACT.scroll` now covers `scroll the <page|window|tab|view|content|results|list> <down|up|left|right|to>` (the original pattern only matched `scroll <down|up|to> the`); `_MEDIA_PATTERNS.render` now covers `render <a|an|the> <image|scene|picture|video|illustration|drawing|artwork>` with optional `me` (the original required `render me`); `_MESSAGING_PATTERNS` adds `notify me <on|via> <telegram|signal|slack|discord>` (parallel to the existing `tell me on …` form); `_FILE_PATTERNS` adds `show me the contents of <file.ext>` (the original required the literal word "file"). All four extensions covered by parametrised regression tests in `tests/routing/test_classifier.py` (+10 tests / 1474 → 1484).

#### `openclaw_routing/dispatcher.py`
- `class OpenClawDispatcher`
  - `__init__(config?, *, llm=None, bridge=None, gaming_mode_manager=None)` — reads openclaw.enabled + routing.stub_responses_enabled; threads optional dependencies for live-dispatch paths.
  - `async handle_browser(intent)` / `handle_media_generation(intent)` / `handle_messaging(intent)` / `handle_file_operation(intent)` / `handle_shell_operation(intent)` — return live results when the bridge is wired (Phases 4, 6, 12), stubs otherwise.
  - `async handle_gaming_mode(intent)` (V1-gap A1) — engage / disengage / status. Routes to `GamingModeManager` for plugin enable/disable; voice messages match the spec phrasing.
  - `async handle_desktop_automation(intent)` (V1-gap C3) — screenshot / list_windows / find_window via `DesktopTool`. Short-circuits with a clear message when gaming mode is engaged.
  - `async handle_window_automation(intent)` (V1-gap C3) — focus / click / type via `WindowControlTool`. Same gaming-mode short-circuit.

#### `openclaw_routing/gaming_mode.py` (V1-gap A1)
- `class GamingModeStatus(str, Enum)` — IDLE / ENGAGED / TRANSITIONING.
- `@dataclass class GamingModeReport` — engage/disengage outcome with per-plugin states + Docker action info.
- `class GamingModeManager`
  - `__init__(*, client, plugins_to_disable, toggle_docker, ...)` — owns the engage/disengage state machine.
  - `async engage()` — calls `client.disable_plugin(slug)` for each configured plugin; optionally stops Docker Desktop. Best-effort: per-plugin failures don't abort the cycle.
  - `async disengage()` — re-enables only the plugins successfully disabled during the matching engage.
  - `status() -> GamingModeStatus`.
  - Audit log: `logs/gaming_mode.jsonl`.

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

#### `openclaw_bridge/client.py` (Phase 3.1)

- `class OpenClawClient` — async client over the `openclaw` CLI.
  Phase 3 deviates from the integration-spec HTTP transport because
  OpenClaw 2026.5.7 doesn't expose `/tools/invoke` or `/messages`
  HTTP endpoints — the CLI is the documented public surface, so the
  bridge invokes it via `asyncio.create_subprocess_exec`.
  - `discover_cli(override) -> str` — explicit override → env var
    (`ULTRON_OPENCLAW_CLI`) → PATH → Windows npm-global default.
  - `health(timeout_s)` — wraps `openclaw health --json`.
  - `send_message(channel, target, text)` — wraps
    `openclaw message send --channel ... --target ... --message ...
    --json`. Returns :class:`SendMessageResult`.
  - `trigger_heartbeat(text, mode, expect_final)` — wraps
    `openclaw system event`. Returns :class:`HeartbeatResult`.
  - `run_agent(message, agent_id, thinking, deliver, ...)` — wraps
    `openclaw agent --json`. Returns :class:`AgentRunResult`.
  - `invoke_tool(tool_name, params, agent_id)` — convenience over
    `run_agent` for "use this OpenClaw tool" dispatch. Raises
    :class:`OpenClawToolError` when the agent reports the tool is
    unavailable.
  - `mcp_list / mcp_show / mcp_set / mcp_unset` — config helpers
    used by :class:`UltronMcpRegistrar`.
  - `enable_plugin(plugin_id)` / `disable_plugin(plugin_id)` /
    `list_plugins(*, enabled_only=False)` (V1-gap A1) — wrap
    `openclaw plugins enable / disable / list --json`. Returns
    `PluginToggleResult` / `List[PluginInfo]`. Failures (plugin not
    installed, auth) translate into structured failures rather than
    raising.
- All methods translate stderr 401/403/Unauthorized markers into
  :class:`OpenClawAuthError`; transport failures into
  :class:`OpenClawGatewayError`. Tokens are never logged.

#### `openclaw_bridge/workspace.py` (Phase 3.3)

- `class WorkspaceWriter` — coordinated writes to the shared
  workspace (`MEMORY.md`, `USER.md`, daily memory files). Atomic
  rename via `os.replace` + advisory lockfiles via `filelock`
  (cross-platform).
  - `write_memory_entry(entry, date, prefix_timestamp)` — append
    to `memory/YYYY-MM-DD.md` with optional `HH:MM` prefix.
  - `update_memory_md(section, content, create_if_missing)` —
    splice one Markdown section in place; preserves siblings.
  - `update_user_md(content)` — full-file replace for the
    auto-populated USER.md.
- All methods are async (sync IO dispatched via
  `asyncio.to_thread`). Lockfile timeouts return a `WriteResult`
  with `error` set rather than raising.

#### `openclaw_bridge/events.py` (Phase 3.4)

- `class OpenClawEventReceiver` — gated-off scaffold for the
  `[voice]`-prefix inbound handoff. Phase 3 ships only the prefix
  matching contract (`should_handle`, `extract_payload`); the
  transport (webhook subscription / polling) is wired in a later
  phase once a real channel exists.
  - `start() / stop()` — no-op when `enabled=False` (default).
  - `dispatch(IncomingMessage) -> bool` — invokes the registered
    handler when the prefix matches; swallows handler exceptions
    so the orchestrator's main loop never sees them.
- `class IncomingMessage` — frozen dataclass; subset of an inbound
  message we route on (channel, sender, body, prefix_match).

#### `openclaw_bridge/mcp_registration.py` (Phase 3.2)

- `class UltronMcpRegistrar` — registers Ultron's MCP server with
  OpenClaw via `openclaw mcp set`. Idempotent: re-running with the
  same payload is a no-op (`already_registered=True`). Fail-open:
  failures return a `RegistrationResult` with `error` set rather
  than raising.
  - `register()` — main entry. Reads `mcp_show` first to detect
    matching existing entry; `mcp_set` only when needed.
  - `verify_registered()` — true iff the configured payload is
    currently registered.
  - `unregister()` — best-effort cleanup; never raises.
  - `schedule_retry(interval_s, on_success, max_attempts)` —
    coroutine for background retry. Caller wraps with
    `asyncio.create_task`.
- Integration deviation: the integration spec assumed Ultron's MCP
  is stdio. Reality is SSE (in-process). The registrar is
  config-driven — `openclaw.bridge.mcp_server_command` defaults to
  `None`, deferring registration. When set (e.g. when a stdio
  proxy is added in a future phase), the registrar wires it up.

#### `openclaw_bridge/holder.py` (Phase 3.5 + Phase 4)

- `class OpenClawBridge` — single dataclass-style holder owned by
  the orchestrator. Encapsulates lifecycle, client, workspace,
  events, registrar, **notifications** (Phase 4).
  - `from_config(openclaw_cfg, notifications_cfg=None) -> Optional[OpenClawBridge]` —
    returns `None` when `openclaw.enabled=False`. Construction is
    fail-open: missing CLI yields `client=None` rather than
    raising. ``notifications_cfg`` is optional (defaults to a
    disabled instance) so callers from before Phase 4 keep
    working.
  - `start()` — sync. Probes the Gateway; on success runs
    `registrar.register()`; on failure (or when MCP command is
    configured but Gateway is unreachable) launches a daemon
    retry thread.
  - `shutdown()` — stops the retry thread and the event receiver.
    Deliberately leaves the MCP entry registered so OpenClaw can
    spawn Ultron's MCP across restarts.
  - `fire_and_forget(coro_factory)` (Phase 4) — schedules a
    coroutine on a daemon thread for off-hot-path dispatch from
    the sync orchestrator loop (used by coding-completion
    notification fires).

#### `openclaw_bridge/notifications.py` (Phase 4)

- `class NotificationDispatcher` — single seam for proactive
  outbound notifications to remote channels. Each event class has
  its own method:
  - `notify_coding_task_completion(summary)`
  - `notify_coding_task_clarification(question)`
  - `notify_heartbeat_alert(text)`
  - `notify_standing_order_output(summary)`
  - `notify_search_results_async(summary)`
- All methods fail-open at every step: missing client, master
  flag off, per-event flag off, no recipient, transport failure
  — each returns a :class:`NotificationResult` with
  ``sent=False`` and a ``skipped_reason``. Voice pipeline never
  blocks.
- Recipient resolution: env var (``user_id_env``) →
  ``fallback_user_id`` → empty (skip).

#### `openclaw_bridge/heartbeat_alerts.py` (Phase 5)

- `class HeartbeatAlertLog` — JSONL-backed alert log with
  thread-safe append + atomic full-file rewrite for updates
  (acknowledgments).
  - `record(text, source, severity, metadata)` — append a new
    alert. Returns :class:`HeartbeatAlert`.
  - `get_alerts(since, only_unacknowledged, limit)` — read,
    filter, return most-recent-first.
  - `acknowledge(alert_id)` — mark seen. Atomic rewrite.
  - `prune()` — drop entries older than ``retention_days``.
- `class HeartbeatAlert` — dataclass with `alert_id` (UUID4 hex),
  `text`, `source`, `severity` ("info"/"warn"/"error"),
  `timestamp`, `acknowledged_at`, `metadata`.
- Tolerates malformed JSONL lines (logs WARN, skips), missing
  files (returns empty list), permission errors (logs WARN).
- `OpenClawBridge.record_heartbeat_alert(...)` is the orchestrator-side
  entry point: records to the log + (when enabled) fires Telegram
  notification via :class:`NotificationDispatcher.notify_heartbeat_alert`.

#### `openclaw_bridge/browser.py` (Phase 6)

- `class BrowserTool` — thin facade over
  :meth:`OpenClawClient.invoke_tool` for browser primitives.
  Each method assembles a structured prompt asking the OpenClaw
  ``ultron-main`` agent to use the browser tool with specific
  parameters; the wrapper unpacks the agent response into a typed
  dataclass.
  - `navigate(url)` → :class:`NavigateResult` (best-effort title
    extraction).
  - `snapshot(mode='ai'|'aria')` → :class:`Snapshot` with refs
    extracted in `ai` mode.
  - `click(ref)` / `type_text(ref, text)` → :class:`ActionResult`.
  - `screenshot()` → :class:`ScreenshotResult` (decodes base64
    when present).
  - `get_page_text()` → :class:`PageTextResult`.
- All methods translate `OpenClawToolError` (tool unavailable
  responses) into structured failures rather than raising.

#### `openclaw_bridge/mcp_tools.py` (Phase 13)

- Stdio MCP server exposing Ultron's read-mostly tools to OpenClaw
  agents. Each tool is a plain Python function callable from
  Python tests; FastMCP registration in :func:`build_server`
  wires them up for stdio dispatch.
- Tool implementations:
  - `get_heartbeat_alerts_impl(since_seconds_ago, only_unacknowledged, limit)`
  - `acknowledge_alert_impl(alert_id)`
  - `run_maintenance_impl(scope=None)` — subprocesses
    `scripts/run_maintenance_for_cron.py --json`
  - `list_active_coding_sessions_impl(max_age_hours=24)` — reads
    `logs/sessions/*.jsonl` audit files
  - `get_recent_voice_alerts_impl(limit=5)` — voice-friendly
    convenience wrapper
- Lazy-imports heavy dependencies; no torch / LLM at startup so
  the spawned process is light.
- :func:`run_stdio` is the entry point invoked by
  ``scripts/run_ultron_mcp_for_openclaw.py``.

#### `openclaw_bridge/desktop.py` (V1-gap C3)

- `class DesktopTool` — wrapper over `OpenClawClient.invoke_tool` for the `desktop-control` plugin. Methods: `screenshot(target?)`, `list_windows()`, `find_window(query)`. Each returns a typed dataclass (`DesktopScreenshotResult`, `ListWindowsResult`, `FindWindowResult`). Tool slugs configurable via `config.desktop.tool_slug_*`.
- `class WindowControlTool` — same pattern over `windows-control` plugin. Methods: `focus(query)`, `click(ref)`, `type_text(ref, value)`. Returns `WindowActionResult`.
- `OpenClawToolError` raised by the underlying client is translated into structured failures with the error preserved in `result.error`.

#### `openclaw_bridge/system_status.py` (Phase 13)

- `class SystemStatusReporter` — voice-side reporter for
  `SYSTEM_STATUS` routing intents. Reads heartbeat alert log +
  active session listing (via the same impl functions
  `mcp_tools.py` exposes to OpenClaw) and renders a brief in-
  character voice narration.
  - `report(SystemStatusIntent) -> SystemStatusReport` — main
    entry. Honors `focus="alerts"|"projects"|"all"` from the
    intent.
- Voice rendering kept short by design (3–4 sentences for
  combined queries, ≤2 for focused). Sanitiser caps individual
  alert text at 160 chars + ellipsis.
- Failure-safe: disk read failures degrade to "no information"
  voice messages; never raises.

### `src/ultron/pipeline/orchestrator.py`

- `class State(Enum)` — IDLE / CAPTURING / PROCESSING / FOLLOW_UP_LISTENING
- `class Orchestrator` — main event loop
  - `MAX_UTTERANCE_SECONDS` (class constant, **30.0** as of 2026-05-11 follow-up fix; was 15.0) — fallback default for the per-capture hard ceiling. The instance attribute `self._max_utterance_seconds` (read from `vad.max_utterance_seconds`) wins at runtime; the class constant is only used when config load fails in `__init__`.
  - `__init__()` — composes audio, wake, vad, addressing, stt, llm, memory, web_search, tts, coding_voice. Reads `vad.max_utterance_seconds` into `self._max_utterance_seconds` (defaults to 30.0; defensive fallback to the class constant on config-load failure). Also reads `vad.long_utterance_threshold_seconds` + `vad.long_utterance_silence_duration_ms` for the adaptive end-of-turn policy. **2026-05-12 Smart Turn V3:** builds the detector via `_build_smart_turn_detector()` BEFORE constructing the VAD; when the detector is present, the VAD is built with `min_silence_ms = smart_turn.fast_path_silence_duration_ms` (500 ms) instead of the legacy 1200 ms.
  - `_build_smart_turn_detector() -> Optional[SmartTurnDetector]` (2026-05-12) — calls `build_detector_from_config(vad.smart_turn, PROJECT_ROOT)`. Returns None when smart-turn is disabled / model file missing / construction fails. Voice baseline unaffected when None.
  - `_smart_turn_should_check(*, speech_seen, speech_samples) -> bool` (2026-05-12) — gate: detector must be available, speech must have been seen, and the contiguous speech duration must be ≤ `smart_turn.window_seconds`. Long utterances bypass smart-turn (the adaptive long-utterance VAD backstop handles those).
  - `_run_smart_turn(captured) -> Optional[SmartTurnVerdict]` (2026-05-12) — single inference call. Returns None on any failure; caller treats as "undecided" → trust VAD.
  - `run()` — main loop (blocks; KeyboardInterrupt clean shutdown)
  - `_capture_utterance()` — VAD-bounded audio capture. **2026-05-11 follow-up fix:** the hard `elapsed_samples < max_samples` ceiling now reads from `self._max_utterance_seconds` (config-driven). Previously a class-level `MAX_UTTERANCE_SECONDS=15.0` cut a real user off mid-sentence on a complex coding ask — the user wasn't pausing; the wall-clock ceiling fired before Silero VAD reported `SPEECH_END`. Bumping to 30 s comfortably covers detailed one-breath asks while still bounding pathological captures (stuck mic, background noise that never resolves to SPEECH_END, etc.). **2026-05-12 Smart Turn V3:** on first SPEECH_END within a capture (and only when the utterance is within the smart-turn window), the captured audio is fed to `_run_smart_turn`. Verdict `complete` → break immediately. Verdict `incomplete` → keep listening; bump VAD silence to `long_utterance_silence_duration_ms` and start an extension timer (`smart_turn.incomplete_extension_ms`). If silence persists past the extension, accept end-of-turn anyway. If speech resumes, cancel the extension and trust the next SPEECH_END. **2026-05-18 latency pass 3 Phase 1:** `_kick_off_tts_preopen()` is called at the very top so the PortAudio device-open overlaps the entire speech + silence-wait window (was running post-capture, only ~5-10 ms of overlap after the 2026-05-16 speculative-STT collapse).
  - `_follow_up_listen(deadline)` — WARM-mode VAD loop. Same `self._max_utterance_seconds` ceiling on cumulative speech (not wall-clock, which is bounded by `deadline`). Same Smart Turn V3 confirmation flow as `_capture_utterance` (2026-05-12). Same 2026-05-18 Phase 1 TTS preopen kick-off at the top.
  - `_run_speculative_classification(user_text)` (2026-05-18 Phase 2) — chained synchronously from the speculative-STT thread after the transcript is stored. Runs `classify_by_rules` (rule path only -- LLM preflight stays foreground), picks the conversational ack via `_maybe_conversational_ack`, and kicks off `_kick_off_rag_prefetch` so the RAG retrieval overlaps the silence wait. Result stashed in `_speculative_classification` keyed by transcript. Re-checks the invalidated flag before storing so SPEECH_START mid-flight drops the result. Defensive against partial test fixtures.
  - `_invalidate_speculative_classification()` (2026-05-18 Phase 2) — marks the classification slot invalid AND cancels the in-flight RAG future. Chains into `_invalidate_speculative_llm()` so all three lanes invalidate atomically on SPEECH_START.
  - `_collect_speculative_classification(user_text)` (2026-05-18 Phase 2) — returns the cached `{text, gate_verdict, ack_phrase, rag_future}` dict if matched, else None. Atomically clears the slot. On mismatch / invalidated, cancels the rolled-over RAG future.
  - `_reset_speculative_classification_state()` (2026-05-18 Phase 2) — called from `_reset_speculative_stt_state` so all three speculation lanes clear at the top of each capture. Chains into `_reset_speculative_llm_state`.
  - `_kick_off_speculative_llm(user_text, verdict, rag_future)` (2026-05-18 Phase 3) — chained from `_run_speculative_classification` when the rule-path verdict resolves to NO_SEARCH. Spawns a daemon thread that applies `apply_uncertainty` + `apply_brevity_hint`, resolves the RAG future, then calls `llm.generate_stream(record_history=False)`. Tokens accumulate in a `queue.Queue`; the response-stream consumer drains them in lieu of a fresh LLM call. Verdict-upgrade (NO_SEARCH -> SEARCH inside `apply_uncertainty`) aborts the speculation since the search prompt body differs.
  - `_invalidate_speculative_llm()` (2026-05-18 Phase 3) — sets the invalidated flag and signals `llm.cancel()` so the streaming iterator exits at the next chunk. The producer's `finally` block still emits the sentinel so consumers don't hang.
  - `_collect_speculative_llm(user_text)` (2026-05-18 Phase 3) — returns `(iter, commit_history)` on hit, `(None, None)` on miss / mismatch / invalidated. The iterator yields tokens from the buffer until the sentinel arrives. The committer is a zero-arg function the caller invokes after consuming the iterator -- it records the turn via `llm.record_completed_turn` so unconsumed speculations leave no orphan in history.
  - `_reset_speculative_llm_state()` (2026-05-18 Phase 3) — drops the buffer + thread handle + response, and signals `llm.cancel()` if a speculation is in flight.
  - `_respond(user_text)` — LLM stream → TTS pipeline (with optional web search)
  - `_speak(text)` — single-shot synthesize + play
  - `_speak_with_barge_in_check(text, *, post_check_window_s=0.5) -> bool` (V1-gap A4) — speak text and report whether wake fired during/after; used by the pre-task confirmation flow.
  - `_handle_capability_response(response, routing_intent)` (V1-gap A4) — wraps the capability voice dispatch. Default path: speak `response.text`. A4 path: speak `response.pre_task_confirmation` first, abort dispatch on barge-in (audit via `runner.record_pre_task_aborted`).
  - `_announce_coding_completion_if_pending()`, `_announce_pending_clarifications()`, `_announce_pending_budget_warning()` — voice-loop poll hooks
  - `_load_memory_if_enabled()` — Qdrant init with graceful fallback
  - `_load_openclaw_bridge_if_enabled()` (Phase 3.5) — constructs
    :class:`OpenClawBridge`. Returns `None` when
    `openclaw.enabled=False` (current default). Fail-open: any
    construction or start failure leaves the bridge disabled
    without affecting the voice path.
  - `self.openclaw_bridge` attribute — accessed by the dispatcher
    when an OpenClaw-bound intent fires. Cleaned up in `shutdown()`
    via `self.openclaw_bridge.shutdown()`.

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
- `audio` (sample_rate, channels, blocksize, dtype, devices, barge-in, **ring_buffer_seconds: 0.5** [2026-05-10: bumped back from 0.15 to act as a STORAGE capacity now that the orchestrator slices mode-specific pre-roll out of it], **cold_pre_roll_seconds: 0.15** [NEW 2026-05-10: post-wake slice; short to avoid the "Tron" prefix the longer pre-roll caused], **warm_pre_roll_seconds: 0.5** [NEW 2026-05-10: post-TTS follow-up slice; long enough to span Silero VAD's ~150 ms speech-start latency without clipping the user's leading word], input_gain_db [2026-05-09])
- `vad` (threshold, min_speech_duration_ms, **min_silence_duration_ms: 1200** [2026-05-09 latency fix; was 500 — natural mid-sentence pauses prematurely closed the capture; trade-off is ~0.7 s slower end-of-turn detection. **Note 2026-05-12:** when `vad.smart_turn.enabled` is true and the model file is present, the orchestrator overrides this to `smart_turn.fast_path_silence_duration_ms` (500 ms) at VAD construction time; smart-turn provides the semantic confirmation that the legacy 1200 ms wall was previously responsible for], window_samples, **long_utterance_threshold_seconds: 8.0**, **long_utterance_silence_duration_ms: 2400** [NEW 2026-05-11 adaptive end-of-turn: once speech has been active past the threshold, orchestrator bumps VAD silence requirement to the long value so a thinking pause mid-prompt doesn't cut the capture. Short utterances stay snappy. Set threshold to 0 to disable.], **max_utterance_seconds: 30.0** [NEW 2026-05-11 follow-up fix: hard ceiling on a single VAD-bounded capture. Was a class-level constant `Orchestrator.MAX_UTTERANCE_SECONDS=15.0` that cut a real user off mid-sentence on a complex coding ask (Whisper transcribed 15.158 s ending mid-phrase at "a button with a box show" — user wasn't pausing; wall-clock ceiling fired before VAD reported SPEECH_END). Now configurable, default 30 s; schema range [5, 120]. Falls back to the class constant only if config load fails.], **smart_turn subsection** [NEW 2026-05-12 Smart Turn V3 semantic end-of-turn confirmation. enabled=true, model_path=`models/smart_turn/smart-turn-v3.2-cpu.onnx`, completion_threshold=0.5 (raise to 0.6-0.7 to reduce false-positive cut-offs at the cost of perceived latency), fast_path_silence_duration_ms=500 (VAD baseline when smart-turn is active), incomplete_extension_ms=700 (additional silence after `incomplete` verdict before submitting anyway), window_seconds=8.0 (training-window cap; longer utterances bypass smart-turn), num_threads=1. Fail-open: missing model file degrades silently to legacy VAD-only behaviour. CPU-only inference ~12 ms; zero VRAM cost.])
- `wake_word` (name, model_path, fallback_model, threshold, cooldown)
- `stt` (model, device, compute_type, beam_size, temperature, etc.)
- `llm` (provider="llama_cpp", **preset** ["qwen3.5-9b"|"qwen3.5-4b"|"custom"; auto-fills model_path/n_ctx/draft_model_path when those keys are omitted — Stage A of the 4B plan], runtime ["in_process"|"http_server"], model_path, draft_model_path, n_ctx, gpu_layers, temperature, top_p, max_tokens, repeat_penalty, history_turns, flash_attn, kv_cache_type, system_prompt, server.{base_url,...}, persona.{source,...})
- `embeddings` (dense_model, sparse_model, dense_dim)
- `qdrant` (data_dir="data/qdrant", collections.{conversations,facts,web_results})
- `memory` (enabled, jsonl_legacy_path, recent_turns, rag_top_k, rag_exclude_recent, facts_top_k, write_queue_maxsize, **retrieval.{multi_pass_enabled=false, max_categories_per_query=4, candidates_per_category_multiplier=4}** (V1-gap A2), **ranking.{rrf_weight=1.0, recency_weight=0.2, recency_half_life_days=7.0, surprise_weight=0.15, redundancy_weight=0.3}** (V1-gap A2), **rag_min_relevance=0.6** (NEW 2026-05-09: cosine-similarity floor for RAG candidates; tuned empirically with bge-small INT8 -- off-topic content peaks ~0.55-0.57, truly relevant 0.7-0.95), **history_turns_for_llm=4** (NEW 2026-05-09: cap on recent-turn history fed to LLM per call; prevents topic-bleed when user pivots topics))
- `web_search` (enabled, brave_api_key_env, brave/jina/cache subsections, **citation.inline_marker_format="bracket"** [V1-gap B3]). 2026-05-09 latency fix tunables: **`jina.timeout_seconds: 6.0`** (was 15.0), **`jina.max_fetch: 2`** (was 3), **`jina.collective_deadline_seconds: 6.0`** (NEW — executor-side cap on parallel fetch wait; 0 disables).
- `addressing` (follow_up_enabled, **warm_mode_duration_seconds: 30.0** ← user override, NOT 10s; rule_confidence_threshold, **zero_shot_addressed_min_confidence: 0.80** [NEW 2026-05-11: demotes low-confidence zero-shot YES verdicts to NOT_ADDRESSED via default_silent; catches the borderline third-person utterances flan-t5-small saturates on at 0.75. Set to 0.0 for legacy permissive behaviour.], zero_shot_model, log_path)
- `coding` (enabled, bridge="direct", mcp.{host,port,...}, template_dir, prompt_token_budget, default/escalation models + thresholds, verification.{smoke,test,lint}_timeout, session_audit_dir, **token_budget_per_session=400000** [2026-05-11 bump from 100000 — new-project sessions burn 100k+ on tool exploration alone before writing files; 400k gives headroom while the 80% warning still fires. Paired with the 2026-05-11 narration honesty fix so users get an explicit "no files written" signal when budget is exhausted mid-exploration], claude_cli, sandbox_root, project_registry_path, audit_log_path, task_timeout, skip_permissions, **voice_task_require_testing=false** [NEW 2026-05-11 token-efficiency fix: was implicitly true via voice.py hardcode, which prepended a "MUST write tests, run, fix, re-run" preamble to every voice-dispatched Claude prompt and 3-5x'd the token spend. Default false lets small voice asks land lean. Users who want tests can say "with unit tests" in their voice request or flip this flag], **facts.{top_k=5, min_confidence=0.75, min_score=0.85, max_age_days=null}** [V1-gap A3], **pre_task_confirmation_enabled=false, pre_task_confirmation_max_words=30, pre_task_barge_in_window_seconds=0.5** [V1-gap A4])
- `projections` (tokenizer, budgets.{clarification,status_delta,adjustment,correction,completion}_context, truncation_warning_threshold, log_truncations)
- `tts` (**engine="piper_rvc" | "xtts_v3"** [NEW 2026-05-10 voice swap; default still legacy for back-compat], piper paths, sample_rate, sentence_flush_chars, length_scale, pause_ms, edge_fade_ms, **pipeline_parallel_enabled=true** [2026-05-09 Piper/RVC split], **speculative_stream_open_enabled=true** [2026-05-09], **speculative_stream_sample_rate=48000** [2026-05-10: was 40000 — actual Ultron RVC output is 48000 Hz, mismatch was forcing the close-and-reopen path on every turn], **output_low_latency_mode=true** [2026-05-09], rvc subsection, **xtts_v3 subsection** [server_python, server_script, reference_audio, host, port, filter_preset="v3_heavy", filter_tail_silence_ms=200, **speed=1.15** (NEW 2026-05-11 cadence tune; XTTS native default is 1.0 — production set to 1.15 for ~15% faster speech without slurring; adjusts synthesis duration tokens so the v3 pedalboard filter is unaffected; safe range ~0.7-1.4, schema-bounded to [0.5, 2.0]), **temperature=0.65** (NEW 2026-05-12 phantom-token mitigation: lowered from XTTS library default 0.75 to sharpen the duration-token distribution so the GPT head stops occasionally emitting fragmentary syllables at sentence ends; range [0.4, 1.0]; threaded through HTTP body to server-side `inference_stream(temperature=...)`; voice character bit-identical because timbre is set by the locked speaker embedding + the v3 filter chain), **phantom_tail_trim_enabled=true** (NEW 2026-05-12 defence-in-depth: client-side post-process that detects the specific phantom-token signature — sustained_speech → ≥150 ms silence → <200 ms event → silence to buffer end — and trims everything after the last sustained-speech region; runs BEFORE the v3 filter so the reverb tail decays normally into its tail_silence_ms padding; set false to disable for A/B), **phantom_tail_silence_threshold=0.005**, **phantom_tail_max_event_ms=200.0**, **phantom_tail_min_lead_silence_ms=150.0**])
- `logging` (file, level, format, datefmt)
- `error_phrases` (13 pools — qdrant_unavailable, brave_unavailable, jina_unavailable, anthropic_unavailable, rvc_unavailable, openclaw_unavailable, piper_unavailable, whisper_repeated_failures, addressing_classifier_failure, wake_word_model_failure, mcp_server_lost, claude_code_subprocess_failed, config_invalid)
- `routing` (llm_disambiguation_enabled, hybrid_task_decomposition_enabled, disambiguation_question_template, routing_log_path, classifier subsection, stub_responses_enabled)
- `openclaw` (enabled=false [stub], gateway_url, auth_token_env, health_check_*_seconds, fail_open, required_agent_id)
- `gaming_mode` (V1-gap A1) — enabled=false, plugins_to_disable=[desktop-control, windows-control], toggle_docker=false, docker_executable_path, docker_process_name, log_path
- `desktop` (V1-gap C3) — enabled=false, default_*_timeout_seconds, plugin_slug, tool_slug_screenshot / tool_slug_list_windows / tool_slug_find_window
- `window_control` (V1-gap C3) — enabled=false, default_action_timeout_seconds, plugin_slug, tool_slug_focus / tool_slug_click / tool_slug_type

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

### `scripts/last_session.py` (V1-gap C2)

**Purpose:** backwards-compat alias for `dump_session.py`. The V1 spec named this script `last_session.py`; both names now coexist and resolve to the same `main(argv)` entry point.
**Run:** `python scripts/last_session.py ...` (forwards every arg to `dump_session.main`).

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

### `scripts/swap_llm_preset.py` (4B plan Stage H)

**Purpose:** atomic preset swap — edits `config.yaml:llm.preset` in place after validating the requested preset's GGUFs are present. Supports `--list`, `--status`, `--dry-run`. The voice path can also be swapped at runtime via the `MODEL_SWITCH` intent ("Ultron, switch to the 9B"); this script is for off-orchestrator workflows.
**Run:** `python scripts/swap_llm_preset.py [--status | --list | <preset> [--dry-run]]`
**In:** `config.yaml`, `models/*.gguf` (validation).
**Out:** updated `config.yaml`; stdout reports the change.

### `scripts/verify_voice_character_4b.py` (4B plan Stage E)

**Purpose:** interactive A/B helper that synthesises 5 representative voice queries through both the 4B and 9B presets so the operator can confirm Ultron's character is preserved. Approved 2026-05-08.
**Run:** `python scripts/verify_voice_character_4b.py`
**In:** loads voice stack twice (once per preset).
**Out:** plays audio + writes A/B comparison CSV.

### `scripts/verify_items_4_to_8.py` (4B plan Items 4–8 verification)

**Purpose:** exercises each of Items 4 (compression), 5 (IRMA), 6 (self-consistency), 7 (canonical-path monitor), 8 (block-and-revise) in the trigger scenario the corresponding flag fires on. Prints concrete deltas (token reduction, accuracy lift, abort timing, etc.).
**Run:** `python scripts/verify_items_4_to_8.py`
**Out:** stdout — per-item status with measurable metrics.

### `scripts/comprehensive_test_harness.py` (Comprehensive end-to-end test pass)

**Purpose:** single-process exhaustive harness for the comprehensive end-to-end test pass. Runs five phases in sequence — routing classifier accuracy on a 63-utterance labeled adversarial corpus spanning every `RoutingIntentKind`; web-gate rule classifier accuracy on 14 labeled queries; circuit-breaker state machine through CLOSED → OPEN → HALF_OPEN → CLOSED → reopen transitions; memory stress (4 threads × 50 turns ingested into a tmp Qdrant + 20 retrieval probes); V1-gap classifier-gating regression (utterances that used to short-circuit to OpenClaw stub when offline). No GPU / model loads — runs anywhere the venv resolves.
**Run:** `python scripts/comprehensive_test_harness.py`
**In:** Imports the worktree's `src/ultron` and the main checkout's `config/` shim.
**Out:** Stdout summary + machine-readable result at `logs/comprehensive_harness_<ts>.json`.

### `scripts/real_api_smoke.py` (Real-API sparing smoke)

**Purpose:** proof-of-life test for the three external services Ultron talks to in production — Brave, Jina, Claude Code. Strict budget: ≤2 Brave calls (one bare query + one chain that adds Jina), ≤1 Jina fetch (via the chain), ≤1 minimal Claude Code haiku invocation. Reads `ULTRON_BRAVE_API_KEY` from `.env`; the Claude CLI defaults to `%APPDATA%\\npm\\claude.cmd` and can be overridden via `ULTRON_CLAUDE_CLI`. Used in the comprehensive end-to-end test pass to confirm circuits + bridge transports work end-to-end without sprawling spend.
**Run:** `python scripts/real_api_smoke.py`
**Out:** Stdout summary + machine-readable result at `logs/real_api_smoke_<ts>.json` (does NOT log the Brave key or any secret).

### `scripts/run_maintenance_for_cron.py` (OpenClaw Phase 7)

**Purpose:** cron-friendly wrapper around `scripts/maintenance.py`. Outputs JSON or single-line Telegram-pretty summary; captures stdout from underlying tasks; structured exit codes (0 ok / 1 task error / 2 init failure). Suitable for Windows Task Scheduler invocations.
**Run:** `python scripts/run_maintenance_for_cron.py [--task <name> ...] [--json | --pretty]`
**In:** subprocesses `scripts/maintenance.py` machinery.
**Out:** stdout — structured summary; exit code per outcome.

### `scripts/benchmark_preflight.py` (V1-gap B5)

**Purpose:** benchmark the web-search gate's pre-flight reasoning pass against the main LLM AND optional CPU-only candidate models. Settles V1-spec Part 1.5's question about whether a dedicated CPU model would be faster than the main Qwen on pre-flight. Decision documented at [docs/preflight_decision.md](preflight_decision.md): keep main LLM (TTFT 79 ms voice baseline already beats the spec's 200 ms threshold).
**Run:** `python scripts/benchmark_preflight.py [--candidate-model PATH] [--skip-main] [--queries N]`
**In:** loads the live `LLMEngine` (or a CPU-only `llama_cpp.Llama` for the candidate); 30 representative queries with manual ground truth.
**Out:** Markdown summary table + appends `preflight_benchmark.backends` block to `baselines.json`.

### `scripts/run_ultron_mcp_for_openclaw.py` (OpenClaw Phase 13)

**Purpose:** stdio MCP entry script OpenClaw spawns when an agent calls one of Ultron's tools. Boots a FastMCP server on stdio that exposes `get_heartbeat_alerts`, `acknowledge_alert`, `run_maintenance`, `list_active_coding_sessions`, `get_recent_voice_alerts`. Imports stay light — no torch / LLM loaded.
**Run:** `python scripts/run_ultron_mcp_for_openclaw.py [--stdio | --list-tools]`
**In:** disk artifacts (heartbeat alert log, session audit dir) + OpenClaw stdio channel.
**Out:** MCP responses over stdio.
**Auto-resolved:** `OpenClawBridgeConfig.mcp_server_command="auto"` resolves to this script via the holder's `_resolve_mcp_command` helper.

### `scripts/_record_phase0_baseline.py` / `scripts/_merge_phase0_baselines.py` (OpenClaw Phase 0)

**Purpose:** record and merge Phase 0 baseline measurements into `baselines.json`. Used during the OpenClaw Phase 0 verification work.
**Run:** `python scripts/_record_phase0_baseline.py`; `python scripts/_merge_phase0_baselines.py`

### `scripts/_vram_peak_monitor.py` (auxiliary)

**Purpose:** background VRAM peak monitor used by `measure_baseline_extended.py` for accurate peak capture during search/coding-session runs.

### `scripts/audio_diagnostic.py` (2026-05-09 audio-quality pass)

**Purpose:** standalone diagnostic harness for far-field mic + wake + Whisper tuning. Loads ONLY the audio path (sounddevice + openWakeWord + Silero VAD + faster-whisper) — NO LLM, NO TTS, NO orchestrator. ~1.5 GB VRAM so it can run while the full Ultron stack is stopped (per the voice-stack-concurrency rule).

**Modes** (`--mode`):
- `noise-floor` — captures N seconds of silence; reports peak / mean RMS dBFS for noise-floor calibration.
- `wake` — captures a window, records max wake-word score, prints whether `FIRED` at the configured threshold; saves audio to WAV via `--save-wav` for replay.
- `phrase` — captures until VAD reports speech end (or hard timeout); reports VAD timing, peak RMS, Whisper transcription + word-coverage vs `--expected-text`.
- `monitor` — live real-time meter: rolling RMS, VAD probability, wake score per chunk; Ctrl+C to exit.

**CLI overrides** (process-local, never write back to config so iteration is fast): `--device` (substring match — "Focusrite", "Voicemeeter"), `--gain-db`, `--wake-threshold`, `--vad-threshold`, `--seconds`, `--whisper-beam`, `--save-wav`, `--label`.

**Audit log:** every test row appends to `logs/audio_diag_<ts>.jsonl`; useful for cross-distance comparison.

**Run:** `python scripts/audio_diagnostic.py --mode wake --device Focusrite --label round1_5ft --seconds 10 --save-wav logs/round1_5ft.wav`

### `scripts/comprehensive_memory_quality.py` (2026-05-09 memory-quality pass)

**Purpose:** end-to-end memory + retrieval quality test pass. Loads embedder + isolated tmpdir Qdrant + (optionally) Qwen 4B. Seeds the isolated store with 58 mixed-topic turns (predator chatter, PC troubleshooting, food, BMWs, weather, code) and runs 28 scenarios verifying:

- Contamination filtering — predator chatter doesn't bleed into a weather query, troubleshooting doesn't bleed into a ducks query, etc.
- Healthy recall — relevant prior context (recent or old) IS surfaced when topic-related.
- Recency-weighted ranking — recent-and-relevant ranks ahead of old-and-relevant.
- Topic shifts — pivot-and-return works (lions → BMW → "what predator did I ask about earlier?").
- Edge cases — short queries, paraphrased queries, queries with no matching memory.

Per-scenario validation: retrieval `expect_includes` / `expect_excludes` substrings + LLM `expect_response_excludes` for contamination tokens.

**Run:** `python scripts/comprehensive_memory_quality.py [--skip-llm] [--scenario-filter X] [--audit-log PATH]`. Without `--skip-llm`, loads Qwen 4B and exercises the full retrieve → context-assembly → response path. ~3.5 GB VRAM.

### `scripts/comprehensive_search_blending.py` (2026-05-09 memory-quality pass)

**Purpose:** end-to-end search-augmented contamination tests with REAL Brave + Jina + Qwen 4B. Verifies the orchestrator's `_search_augmented_tokens` path: predator chatter in memory doesn't bleed into a Python-3.13 search response; troubleshooting context doesn't bleed into a duck-lifespan search response; relevant troubleshooting context DOES blend into a motherboard-light search response.

3 scenarios; ~3 Brave + 3 Jina calls per full run. Within free-tier quota.

**Run:** `python scripts/comprehensive_search_blending.py` (requires `ULTRON_BRAVE_API_KEY`).

### `scripts/_debug_retrieval_cosine.py` (2026-05-09 memory-quality pass; debug only)

**Purpose:** prints cosine similarity between a probe query and a hand-picked candidate set. Used to empirically tune `memory.rag_min_relevance` against the actual production embedder (bge-small INT8). The 0.6 threshold was chosen because off-topic content peaked at 0.55-0.57 across the probe corpus, while genuinely relevant content scored 0.7-0.95.

**Run:** `python scripts/_debug_retrieval_cosine.py`. No flags; edit the `PROBES` and `CANDIDATES` lists at the top of the file to test new query+content pairs.

### `scripts/cleanup_stale_processes.py` (2026-05-14 cleanup pass)

**Purpose:** find and kill stale Ultron-related python processes
(orphaned pytest workers, stale `run_ultron_mcp_for_openclaw.py`
processes from old worktrees, orphaned XTTS servers, large no-cmdline
workers). Always preserves the currently-running Ultron and its
process chain: the script enumerates the TCP listener on port 19761
(the MCP server) and adds that process plus its ancestors and
descendants to a "do not touch" set.

**Run:**

```
python scripts/cleanup_stale_processes.py            # dry-run; prints what it would kill
python scripts/cleanup_stale_processes.py --kill     # actually terminates them (prompts first)
python scripts/cleanup_stale_processes.py --kill -y  # skip the prompt
```

**Flags:** `--max-age-minutes` (default 30; ignore unknown-cmdline workers younger than this), `--min-rss-mb-unknown` (default 200; only kill unknown-cmdline workers with at least this much RAM).

**In:** `psutil` (already in the venv) + the live process table. **Out:** stdout summary + exit code 0 on success, 1 if any termination failed.

### `scripts/bench_llm_ubatch.py` (NEW 2026-05-15 latency pass)

**Purpose:** sweep llama-cpp-python's `n_batch` / `n_ubatch` knobs to find the lowest-TTFT combination for voice-length prompts on the active hardware. Loads `LLMEngine` fresh per combination (so each gets a clean Llama instance) and measures TTFT on 5 representative queries with 2 warmup runs. Writes results into `baselines.json:llm_n_ubatch_sweep`. Loads the voice stack -- ASK before running per `feedback_voice_stack_concurrency.md`. Default sweep covers `(None, None)` baseline + 5 `(n_batch, n_ubatch)` combinations; takes ~3-6 min on the 4070 Ti.

**Run:** `python scripts/bench_llm_ubatch.py [--sweep "128,256,512,1024"] [--warmup 2] [--trials 5]`

**Empirical result on 2026-05-15:** all combinations give ~63 ms median TTFT on voice-length prompts -- no measurable win at short context. Knobs stay in place for future long-context tuning.

### `scripts/bench_stt_latency.py` (NEW 2026-05-15 latency pass)

**Purpose:** measure Whisper STT latency at 1s / 3s / 5s / 8s audio lengths to right-size STT optimisations. Generates speech-like synthetic audio, warms up the engine, then runs `--trials` measurements at each length. Reports median / p95 / min / max / RTF. Loads voice stack -- ASK first.

**Run:** `python scripts/bench_stt_latency.py [--lengths 1,3,5,8] [--warmup 2] [--trials 5]`

**Empirical result on 2026-05-15 (small.en + int8_float16 + beam=5):** 1s = 156 ms, 3s = 188 ms, 5s = 109 ms, 8s = 109 ms. With **beam=1 on 5s audio: 78 ms median** -- saves ~80 ms vs beam=5. This bench drove the Phase 4 decision to set `stt.beam_size: 1` as the new production default.

### `scripts/bench_llm_prefix_cache.py` (NEW 2026-05-16 latency pass 2)

**Purpose:** A/B benchmark of the in-process `LLMEngine` TTFT with `LlamaRAMCache` cache_bytes=0 (disabled) vs cache_bytes>0 (enabled). Builds a fresh `LLMEngine` per condition (so each gets a clean Llama instance + cache state) and measures TTFT on 5 representative voice queries with configurable warmup. Drove the Phase 2 decision to ship the cache infrastructure but flip the default to disabled. Loads the voice stack -- ASK before running per `feedback_voice_stack_concurrency.md`.

**Run:** `python scripts/bench_llm_prefix_cache.py [--turns 5] [--warmup 1] [--out baselines.json]`

**Empirical result on 2026-05-16 (4070 Ti + josiefied-qwen3-4b Q4_K_M):** cold-cache TTFT median **63 ms** (78, 79, 63, 62, 63 across 5 queries); warm-cache (2 GiB RAMCache) TTFT median **78 ms** (78, 78, 79, 63, 62). **The cache shows a -15 ms regression** -- llama.cpp's internal KV cache already handles intra-session prefix reuse; the explicit RAMCache's `load_state` memcpy exceeds the eval savings on our short ~280-token system prompts. Result merged into `baselines.json:llm_prefix_cache_bench`. The knob and bench stay shipped so operators with longer prompts / cross-session reload patterns can opt in.

**Operator note:** the bench requires the production GGUF on disk. When run from a worktree (not the main checkout), set `ULTRON_LLM_MODEL_PATH=C:\STC\ultronPrototype\models\Josiefied-Qwen3-4B-abliterated-v2.Q4_K_M.gguf` (or the absolute path to the active preset's GGUF) so the engine resolves correctly -- the worktree's `models/` directory is empty.

---

## Tests

### `tests/conftest.py` — Path setup + session-end subprocess reaper.

Two responsibilities:

1. Prepend the project root and ``src/`` to ``sys.path`` so
   ``from ultron.*`` works when pytest is launched from the repo
   without an editable install.

2. Register a ``pytest_sessionfinish`` hook (2026-05-14 cleanup pass)
   that walks the test runner's descendant python processes and
   terminates them when the session ends -- whether the run completed
   normally, crashed, or was Ctrl-C interrupted. Without this, a hung
   test or a backgrounded pytest that never gets reaped leaves a
   python worker holding hundreds of MB of RAM (and VRAM if torch /
   CUDA was loaded by a fixture). Fail-open at every step (psutil
   import / TCP enumeration / individual terminate calls); never
   touches a process tied to the live Ultron orchestrator (detected
   via the port-19761 listener and its ancestor/descendant chain).

### Default suite (no env gate) — **2472 passed / 15 skipped (GPU-gated)**, ~57 s wall (2026-05-18)

**Top-level (~25 files):**
- `test_addressing.py` — rule-based addressing classifier
- `test_audio.py` — capture, ring buffer (incl. 2026-05-10 mode-aware `snapshot(last_n_samples=...)` slicing), devices
- `test_response_style.py` (22, 2026-05-10) — `is_brief_question` / `apply_brevity_hint` coverage: short-question detection, depth-marker skip, long-question pass-through, empty input, idempotence on already-hinted text
- `test_conversational_ack.py` (24, 2026-05-12 — NEW) — conversational filler-ack: gate eligibility (long-utterance fires, short-utterance/empty/clarification-pending skipped, whitespace-stripped), `ConversationalAckSource` shuffled-cycle (no immediate repeats, full pool per cycle, custom pool, empty-pool rejection), phrase-pool sanity (no web-search overlap, period-terminated, short, no duplicates), and orchestrator-level wiring (ack appears as first token on no-gate fallthrough path, suppressed on short utterance / pending clarification, fail-open on broken source or `has_pending_clarification` exception)
- `test_precomputed_ack.py` (25, 2026-05-15 — NEW) — `PrecomputedAckClipCache`: construction (dedup / strip / sort / drop empty / None-safe / starts empty), lookup (miss / strip-match / empty input / wrong phrase miss), prewarm (populates all / returns count / skips empty clip / swallows synth exception / partial population / idempotent), thread safety (concurrent get during prewarm), default phrase pool factory (collects both conv + web-search pools), `prewarm_in_background` (returns daemon thread / populates / honours name)
- `test_llm_precomputed_rag.py` (9, 2026-05-15 — NEW) — `precomputed_rag_snippets` kwarg on `_build_messages` / `generate` / `generate_stream`: snippets appear in message body, internal retrieve is bypassed, empty list = no RAG (not retry), None falls back to legacy retrieve, suppress_memory_context wins over precomputed, public `retrieve_rag_snippets` proxies private, returns [] when no memory, preserves recent history independently, compatible with gate_verdict
- `test_orchestrator_rag_prefetch.py` (11, 2026-05-15 — NEW) — orchestrator `_kick_off_rag_prefetch` (returns None when memory disabled / multi-pass enabled / executor broken; kicks off + completes when single-pass), `_collect_rag_future` (None future returns None / completed returns value / exception returns None / empty list distinguishable), `_build_response_stream` integration (prefetch kicks off + precomputed snippets reach LLM, no memory skips prefetch, multi-pass skips prefetch and passes None to LLM)
- `test_llm_batch_tunables.py` (14, 2026-05-15 — NEW) — `LLMConfig.n_batch` + `n_ubatch`: schema (defaults are None, accepts explicit values, rejects 0 / negative / too-large, n_ubatch may exceed n_batch in schema), `_build_llama` wiring (omits kwargs when None / passes n_batch only when set / passes n_ubatch only when set / passes both when set), top-level `UltronConfig` round-trip (default keeps None, accepts values)
- `test_tts_preopen.py` (13+2, 2026-05-15 NEW + 2026-05-16 latency 2 extension) — TTS output-stream pre-open: xtts_v3 (prepare+consume match SR / consume mismatch closes & returns None / consume with no preopen returns None / prepare idempotent / failure swallowed / stop closes leftover), legacy speech.py (prepare+consume / SR-mismatch close / failure swallowed / **legacy silence-write invoked + failure-swallowed** (2026-05-16)), orchestrator (`_kick_off_tts_preopen` returns None when engine lacks method / returns thread when engine supports / swallows thread-construction failure / no-op when tts is None)
- `test_llm_prefix_cache.py` (11, 2026-05-16 latency 2 — NEW) — `LLMConfig.prefix_cache_ram_bytes`: schema (default 0 after bench-driven flip, accepts 0 / large values, rejects negative, round-trip), `_build_llama` wiring (attaches `LlamaRAMCache` when set / skips when 0 / fail-open on import error / fail-open on set_cache exception), top-level `UltronConfig` round-trip
- `test_speculative_stt.py` (12, 2026-05-16 latency 2 — NEW) — orchestrator speculative-STT helpers: kick-off (starts background thread / idempotent while in-flight / fail-open on thread launch failure), collect (None when no kick-off / waits for thread / resets state / None on transcription exception / None on timeout), invalidate (causes collect to return None / re-arms for next kick-off after collect), reset state (clears stale result without killing thread), kick-off copies audio to avoid race
- `test_speculative_classification.py` (21, 2026-05-18 latency pass 3 Phase 2 — NEW) — orchestrator speculative-classification helpers: `_run_speculative_classification` (stores rule-path verdict + ack + RAG future; skips on already-invalidated; mid-work invalidation drops result; missing web_gate -> None verdict; ack/RAG exception swallowed), `_invalidate_speculative_classification` (sets flag + cancels RAG future; idempotent; cancel-exception swallowed; STT invalidate propagates), `_collect_speculative_classification` (returns None on empty / text-mismatch / invalidated; clears slot atomically; defensive on missing lock), `_reset_speculative_classification_state` (clears slot + cancels RAG; defensive), STT-thread chain (chains classification on success; skips on empty transcript; skips on invalidated; reset propagates to classification slot)
- `test_speculative_llm.py` (25, 2026-05-18 latency pass 3 Phase 3 — NEW) — LLMEngine + orchestrator speculative-LLM. LLMEngine surface (4): `record_history=True` records turn / `record_history=False` skips auto-record / `record_completed_turn` records explicitly / skips empty input. Orchestrator helpers (21): `_kick_off_speculative_llm` (starts thread + buffers tokens / idempotent / skips on missing LLM / skips on None verdict), `_invalidate_speculative_llm` (signals cancel + sets flag / idempotent / defensive on missing lock), `_collect_speculative_llm` (None when empty / drains buffer + commits history on completion / None on text mismatch / None on invalidated / commit no-op on incomplete speculation / defensive on missing lock), `_reset_speculative_llm_state` (clears + cancels in-flight / defensive), cross-lane invalidation (classification invalidate propagates / STT invalidate propagates / reset propagates), classification chain (NO_SEARCH kicks off LLM / SEARCH skips / UNCERTAIN skips)
- `test_llm_strip_thinking.py` (9, 2026-05-14 — NEW) — `strip_thinking_text` pure function: clean text passthrough, single-block strip, multi-block strip, surrounding text preserved, unterminated `<think>` drops tail, multiline blocks, real-session screen-context pattern, idempotence, short-input fast path. Covers the gap where blocking-path `LLMEngine.generate()` previously returned raw `<think>...</think>` chains (the streaming path was already filtered).
- `test_smart_turn.py` (43, 2026-05-12 — NEW) — Smart Turn V3 semantic end-of-turn confirmation: `SmartTurnConfig` schema (defaults match production layout, all four range-enforced fields, dict round-trip, nested-under-VADConfig), `truncate_or_pad_for_smart_turn` pure function (under-window passthrough, over-window truncation to last n seconds, int16→float32 conversion, multi-dim flatten, non-16kHz rejection, custom window override), `SmartTurnDetector` construction (missing file, out-of-range threshold/window/threads, lazy-loading, warmup-propagates-failure, empty/wrong-sr/post-close all return None), `build_detector_from_config` fail-open (disabled / missing file / absolute-path missing all return None; present file yields a lazy detector), real-model end-to-end (6 tests, skipped when `models/smart_turn/smart-turn-v3.2-cpu.onnx` is absent — loads + warmup, silence verdict shape, threshold flip with identical probability, short audio padded by WhisperFeatureExtractor, long audio truncated to last 8 s, median inference under 150 ms), orchestrator-level wiring (`_smart_turn_should_check` gate semantics across detector-missing / no-speech / within-window / over-window, `_run_smart_turn` passes verdict through + swallows exceptions, `_build_smart_turn_detector` fail-open for disabled / missing file)
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
- `test_llm_enable_thinking.py` (11, 4B plan Stage F) — `enable_thinking` parameter plumbing: helper kwargs, in-process generate/generate_stream pass-through, HTTP payload pass-through, back-compat when default
- `test_llm_rag_position.py` (7, 4B plan Stage G) — `_build_messages` honors `llm.rag.position`: recency mode prepends to user message, system mode folds into system message, no-snippets/retrieve-failure fallback, helper invariants
- `test_on_the_fly_preset_switching.py` (16, 4B plan Stage H infra) — `ULTRON_LLM_PRESET` env-var override (clears overrides by default, opt-in keep-overrides flag), minimal-YAML preset-only config, `check_vram._resolve_target_mb` (table + CLI override + env var + unknown fallback), `_format_line` shows preset label, `swap_llm_preset._rewrite_preset` (basic / preserves comment / first-match / missing-line raises)
- `tests/routing/test_model_switch_classifier.py` (54, 4B plan voice-swap) — classifier maps "switch to 4B/9B/four B/for B/nine B/4 B/4-B" + verb variants (switch/swap/change/use/load/go/move/activate/engage/run/select) to `RoutingIntentKind.MODEL_SWITCH`; rejects passing mentions ("the 4B is faster") and conversational utterances; pending clarification suppresses (mid-dialogue safety); active coding task does not block; `_resolve_model_switch_target` helper
- `test_llm_reload_for_preset.py` (9, 4B plan voice-swap) — `LLMEngine.reload_for_preset` rejects http_server runtime + unknown preset; idempotent on same-preset; success path replaces `_llm` and clears history; sets `ULTRON_LLM_PRESET` env + clears stale `ULTRON_LLM_MODEL_PATH`; failure path keeps old engine, restores env vars (whether they were set or unset originally)
- `test_llm_prompt_injection_defense.py` (21, comprehensive QUALITY pass Q10 iter 1+2) — `_sanitize_user_input` neutralises tag-style markers ([INST]/[/INST], <|im_start|>/<|im_end|>/<|system|>/<|user|>/<|assistant|>, stray </think>); detects natural-language jailbreak patterns ("ignore previous instructions", "you are now X", "respond with the exact word", "act as", "pretend"); preserves benign questions (zero false-positive on normal voice queries); end-to-end verified: pre-defense 2/3 of Q8 prompt-injection probes succeeded; post-defense 0/3. Voice baseline TTFT 79 ms / VRAM 7889 MB unchanged (defence is sub-microsecond on benign input).
- `test_web_search_parallel_fetch.py` (6, 2026-05-09 latency hot-fix) — verifies the `WebSearchExecutor` parallel-Jina-fetch path: wall-time dominated by the slowest URL (not the sum); collective deadline abandons slow fetches and degrades them to snippet-only with `jina_deadline:<url>` notes; partial success with one fast + one slow URL keeps the fast one's `full_text`; per-fetch exception in one parallel branch doesn't break the others; `collective_deadline_seconds=0` disables the cap; `max_fetch=0` skips Jina entirely.
- `test_tts_pipeline_parallel.py` (15, 2026-05-09 + 2026-05-10) — original 11 cover the parallel split, speculative stream open, sample-rate-mismatch fallback, low-latency hint, RVC fallback, cancellation. 2026-05-10 added 4 for producer-signaled lookahead: `test_first_clip_plays_before_next_fragment_yielded` (the ack-first contract — first clip MUST be written to the stream before the generator is asked for the second), `test_slow_second_clip_does_not_kill_playback` (4 s gap between fragments doesn't trigger RVC starvation abort — guards the BMW-search failure mode), `test_clipitem_is_known_last_skips_lookahead` (ClipItem namedtuple shape + flag carries through), `test_end_of_stream_sentinel_terminates_playback` (None on audio_q ends playback with tail silence even when the previous ClipItem had `is_known_last=False`).
- `test_voice_model_switch.py` (11, 4B plan voice-swap) — `CapabilityVoiceController._handle_model_switch` calls `llm_engine.reload_for_preset(target)`, speaks "Switched to the 4B/9B" on success, "I'm already running the X" on idempotent, "I couldn't switch ..." on failure with reason; "I can't switch models — engine isn't wired" when llm_engine is None; missing payload says "couldn't tell which model"; end-to-end classifier-then-controller for utterances
- `tests/routing/test_irma_reformulation.py` (15, 4B plan Item 5) — `InputReformulator` pure-text shape (default-only-utterance, whitespace-strip, quote-escape, recent-decisions section, max-recent truncation, active-session, routing-hints, max_recent=0 omits, log-row factory); disambiguator integration with the IRMA flag (default-OFF passes raw, ON uses enriched, reformulation-failure falls back, no-context still emits utterance)
- `test_self_consistency.py` (27, 4B plan Item 6) — `majority_vote_text` (winner, whitespace-strip, tie-first-wins, empty input, blank filter), `majority_vote_json` (winner, unparseable handling, think-block strip, first-block-only, all-unparseable returns None, arrays), `majority_vote_label` (case-insensitive, no-match), `run_self_consistency` driver (sampler called N times, default text aggregator, sampler exception handling, fallback to first non-empty, n-clamping), `should_apply_self_consistency` config gate (default-off, global-on, per-site disabled), decomposer integration (single-call default, N-call with consistency, majority winner, per-site bypass, all-unparseable fallback)
- `test_canonical_monitor.py` (17, 4B plan Item 7) — canonical set lockdown (standard tools, MCP callbacks), canonical-only paths (no abort), threshold-not-reached, threshold-reached-in-window aborts, late drift does not abort, latch semantics, reset clears state, non-tool-use events ignored, empty/None tool name ignored, case-insensitive match, attribute-style event input, custom canonical override, verdict-shape (off_canonical_tools list, immutability), factory gate (disabled returns None, enabled returns instance with config)
- `test_block_and_revise.py` (14, 4B plan Item 8) — `ToolCallValidator` ALLOW + BLOCK verdicts, think-block strip, case-insensitive, fail-open on no-LLM / exception / unparseable / empty, prompt rendering (tool name, args, args truncated, goal-quote escaped), `is_enabled` config gate
- `test_compression.py` (26, 4B plan Item 4) — heuristic compresses redundant text, preserves negations (and "isn't" preserves negation-meaning), collapses repeated punctuation, short input passthrough, empty passthrough, ratio-1.0 means no drop, higher-ratio drops more; perplexity-scorer drops lowest-score, scorer exception fallback, mismatched-length fallback; result dataclass; factory off-returns-None / on-returns-instance; `maybe_compress` global-off / per-surface-off / per-surface-on / unknown surface / history default-off / compressor exception / empty text; integration `_format_rag_block` default-OFF unchanged + ON-compresses; `format_sources_for_prompt` default-OFF unchanged + URL-preserved-on
- `test_self_consistency_web_gating.py` (8, 4B plan Item 6 second site) — `web_search.gating.classify_by_preflight` with self-consistency: default-OFF single greedy call (back-compat), N-call when enabled, configured non-zero temperature, majority-vote winner, per-site disabled bypass, all-unparseable fallback to NO_SEARCH, LLM-exception returns NO_SEARCH (never raises)
- `test_canonical_monitor_runner_wiring.py` (9, 4B plan Item 7 wiring) — `CodingTaskRunner` listener gating: not-attached-when-disabled, attached-when-enabled, cancels handle on first abort verdict, doesn't cancel on canonical sequence, latches after first abort, swallows listener exceptions; `CapabilityVoiceController.pending_canonical_abort` polls + clears + swallows runner exception
- `test_block_and_revise_dispatcher_wiring.py` (10, 4B plan Item 8 wiring) — `OpenClawDispatcher` per-handler validator gate: disabled-flag skips, no-LLM skips, ALLOW dispatches to stub, BLOCK short-circuits with reason, all 5 handlers run validator when enabled, validator exception falls open, voice controller threads its `llm_engine` to the dispatcher

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
| `Josiefied-Qwen3-4B-abliterated-v2.Q4_K_M.gguf` | `LLMEngine` (when `llm.preset == "josiefied-qwen3-4b"`, **CURRENT DEFAULT 2026-05-14 second-pass**). Same Goekdeniz-Guelmez Josiefied + abliterated Qwen3-4B-Instruct-2507 as the Q5_K_M below, just smaller quant. ~3.0 GB VRAM loaded; chosen to leave room for the user's typical ~4.7 GB of background GPU usage. | 2.4 GB |
| `Josiefied-Qwen3-4B-abliterated-v2.Q5_K_M.gguf` | retained for swap-back / quality A/B. Same model at Q5_K_M (~3.5 GB VRAM loaded). | 3.0 GB |
| `Josiefied-Qwen3-8B-abliterated-v1.Q5_K_M.gguf` | `LLMEngine` (when `llm.preset == "josiefied-qwen3-8b"`; retained for swap-back to the bigger abliterated variant) | 5.85 GB |
| `Qwen3.5-9B-Q4_K_M.gguf` | `LLMEngine` (when `llm.preset == "qwen3.5-9b"`; retained for swap-back, not abliterated) | 5.29 GB |
| `Qwen3.5-4B-Q4_K_M.gguf` | `LLMEngine` (when `llm.preset == "qwen3.5-4b"`; retained for swap-back / spec decoding, not abliterated) | 2.55 GB |
| `Qwen3.5-0.8B-Q4_K_M.gguf` | speculative-decoding draft for plain qwen3.5-4b preset only (no abliterated draft published) | 0.50 GB |
| `openwakeword/ultron.onnx` | `WakeWordDetector` | small |
| `piper/en_US-ryan-medium.onnx[.json]` | `TextToSpeech` | ~60 MB |
| `rvc/hubert_base.pt` | `RvcConverter` | ~362 MB |
| `rvc/rmvpe.pt` | `RvcConverter` | ~178 MB |
| `smart_turn/smart-turn-v3.2-cpu.onnx` | `SmartTurnDetector` (Smart Turn V3 semantic end-of-turn; NEW 2026-05-12) | 8.68 MB |
| `.hf-cache/` | `HybridEmbedder`, addressing zero-shot | varies |

### `ultron_james_spader_mcu_6941/` (main checkout only)

RVC voice model for Ultron timbre.
- `Ultron.pth` — main RVC checkpoint
- `added_IVF301_Flat_nprobe_1_Ultron_v2.index` — speaker index

---

## Documentation index

Reading order for a fresh Claude:

1. **`CLAUDE.md`** (project-root, auto-loaded by Claude Code) — orientation + binding standards.
2. **`MEMORY.md`** (auto-loaded) — index of memory files.
3. **`project_ultron_openclaw.md`** — primary cross-phase OpenClaw reference.
4. **`project_ultron_4b_plan.md`** — final 4B + Items 4–8 state with measured TTFT/VRAM.
5. **`feedback_*.md`** — confirmed user decisions (especially `feedback_no_paid_apis.md`, `feedback_llm_runtime_decision.md`).
6. **`docs/codebase_structure.md`** ← THIS FILE — single-source reference.
7. **`docs/openclaw_integration_final_summary.md`** — cross-phase OpenClaw reference + intentional deviations + setup-readiness checklist.
8. **`docs/architecture.md`** — pipeline + diagrams.
9. **`docs/phase3_5_followup.md`** — open punch list (deferred Foundation Part 3.5).

### Comprehensive testing + improvement passes (most recent)
- **Functional / correctness pass plan:** [docs/comprehensive_test_plan.md](comprehensive_test_plan.md) — 16 phases, 38 dimensions, single-process harness pattern.
- **Functional pass results:** [docs/comprehensive_test_report.md](comprehensive_test_report.md) — 145-row metrics table; 4 classifier coverage gaps fixed; voice baseline 79 ms / 7818 MB.
- **Quality pass plan:** [docs/comprehensive_quality_plan.md](comprehensive_quality_plan.md) — 13 phases (Q0–Q13), 38 quality dimensions, ≤10-iteration improvement loop.
- **Quality pass results:** [docs/comprehensive_quality_report.md](comprehensive_quality_report.md) — 107-row metrics table; Q10 iteration audit; prompt-injection defense layer.

### Foundation reference
- Day-to-day operation: [docs/operations.md](operations.md)
- Adding code / debugging: [docs/development.md](development.md)
- Config reference: [docs/configuration.md](configuration.md)
- Error handling: [docs/error_handling.md](error_handling.md)
- Capability routing: [docs/routing.md](routing.md)
- Test layout: [tests/integration/mocks.md](../tests/integration/mocks.md)
- 16-step end-to-end smoke test: [docs/smoke_test.md](smoke_test.md)
- Foundation Phase 1 inventory snapshot: [docs/system_inventory.md](system_inventory.md)
- Phase 3 discovery catalog: [docs/config_discovery.md](config_discovery.md)

### OpenClaw integration (architecture)
- **OpenClaw integration architecture + Phase 0/1 status:** [docs/openclaw_integration.md](openclaw_integration.md)
- **OpenClaw runtime ops (agents, supervisor, locked-in constraints):** [docs/openclaw_runtime.md](openclaw_runtime.md)
- **Cross-phase final summary + setup-readiness checklist:** [docs/openclaw_integration_final_summary.md](openclaw_integration_final_summary.md)

### OpenClaw integration (per-phase close-outs)
- **Phase 1 (persona migration):** [docs/phase_1_summary.md](phase_1_summary.md)
- **Phase 3 (bridge layer):** [docs/phase_3_summary.md](phase_3_summary.md)
- **Phase 4 (Telegram channel):** [docs/phase_4_summary.md](phase_4_summary.md)
- **Phase 5 (heartbeat):** [docs/phase_5_summary.md](phase_5_summary.md)
- **Phase 6 (browser tool):** [docs/phase_6_summary.md](phase_6_summary.md)
- (Phases 7–13 have inline summaries in `openclaw_integration_final_summary.md`.)

### OpenClaw integration (user-side setup procedures)
- **Telegram channel:** [docs/openclaw_telegram_setup.md](openclaw_telegram_setup.md)
- **Heartbeat agents[].heartbeat block:** [docs/openclaw_heartbeat_setup.md](openclaw_heartbeat_setup.md)
- **Browser tool (Playwright + Chromium):** [docs/openclaw_browser_setup.md](openclaw_browser_setup.md)
- **Cron jobs (Windows Task Scheduler fallback):** [docs/openclaw_cron_setup.md](openclaw_cron_setup.md)
- **Bundled hooks (`session-memory`, `command-logger`):** [docs/openclaw_hooks_setup.md](openclaw_hooks_setup.md)
- **Memory Wiki plugin:** [docs/openclaw_memory_wiki_setup.md](openclaw_memory_wiki_setup.md)
- **Local-only ComfyUI media generation:** [docs/openclaw_media_generation_setup.md](openclaw_media_generation_setup.md)
- **iOS / Android node pairing:** [docs/mobile_node_setup.md](mobile_node_setup.md)
- **Standing-order programs:** [docs/standing_orders.md](standing_orders.md)
- **Three-layer memory architecture (Qdrant + workspace + Wiki):** [docs/memory_architecture.md](memory_architecture.md)
- **Gaming mode (V1-gap A1):** [docs/openclaw_gaming_mode_setup.md](openclaw_gaming_mode_setup.md)
- **Desktop / window control (V1-gap C3):** [docs/openclaw_desktop_control_setup.md](openclaw_desktop_control_setup.md)

### 4B optimization plan
- **4B-model optimization plan (all stages + Items 4–8 done):** [docs/4b_optimization_plan.md](4b_optimization_plan.md)
- **GGUF SHA256 reference:** [docs/model_checksums.md](model_checksums.md)

---

## Maintenance contract

**This document is the operating manual. Keep it current.**

This contract is **binding** — every non-trivial change to the
codebase must update this document in the same change. Skipping
the update means future sessions waste time re-deriving ground
truth from the source. **Don't skip.**

The CLAUDE.md (project-root) at the top of this prompt's reading
order calls this contract out explicitly so a fresh Claude Code
session sees it before its first edit.

### What "non-trivial change" means

You MUST update the relevant section of this document when you:

1. **Add a new module file** under `src/ultron/` →
   - Add to the file tree.
   - Add a section under "Source modules" with the public API
     (classes, functions, brief in/out).
   - If it's a new subsystem (e.g. `src/ultron/openclaw_bridge/`),
     add to the architecture diagram in `docs/architecture.md`
     too.

2. **Add a new public class or function** to an existing module →
   - Add it to the module's section under "Source modules".
   - Note the inputs and outputs in one line.

3. **Remove or rename** an existing module / class / function →
   - Update every section that referenced it.
   - Search for the old name with Grep before declaring done.

4. **Add a new script** under `scripts/` →
   - Add to the file tree.
   - Add a section under "Operational scripts" with purpose,
     run command, in/out, and functions.

5. **Add a new test directory or test category** →
   - Add to the file tree (under `tests/`).
   - Add to the relevant "Tests" subsection.
   - Update the "current state" header at the top of this file
     with the new total.

6. **Add a new log file or data path** →
   - Add to the "Runtime artifacts" tables.

7. **Add a new doc** under `docs/` →
   - Add to the "Documentation index" with the right category
     (Foundation reference / OpenClaw architecture / per-phase
     close-out / user-side setup / 4B plan).
   - Add to the file tree under `docs/`.
   - Cross-reference where relevant in other sections.

8. **Add a new config section / key** →
   - Add to the `config.yaml` summary in "Configuration".
   - Update [docs/configuration.md](configuration.md) too
     (per-key reference).
   - Document any new defaults in the relevant `feedback_*.md`
     if it reflects a confirmed user decision.

9. **Change a cross-cutting flow** (voice path, coding path,
   search path, dispatch path, OpenClaw bridge path) →
   - Update the relevant diagram in "Cross-cutting flows".

10. **Migrate a subsystem out of the `config/settings.py` shim** →
    - Update [docs/phase3_5_followup.md](phase3_5_followup.md)
      (cross off).
    - If it changes the public API of the migrated module,
      update its "Source modules" section here.

11. **Bump test counts** — the file's header tracks "X passed /
    Y skipped / Z failed". Update these when the count changes.

12. **Land a new phase / sub-phase** → bump the phase status
    line in the header.

### The validation loop

After your change:

```powershell
# 1) Tests pass
C:\STC\ultronPrototype\.venv\Scripts\python.exe -m pytest tests/ -q --no-header --ignore=tests/coding/test_orchestration_real.py

# 2) Config still validates
C:\STC\ultronPrototype\.venv\Scripts\python.exe scripts\validate_config.py

# 3) Re-read this doc and confirm:
#    - File tree matches `git ls-files | grep -v '^\\.'`
#    - "Source modules" sections cover every src/ultron/ file
#    - "Operational scripts" sections cover every scripts/ file
#    - "Tests" subsections cover every tests/ subdirectory
#    - "Documentation index" links every docs/*.md file
```

If the doc no longer matches reality after your changes, fix
this document before declaring the task done.

### Why this matters

A fresh Claude Code session reads this document + the memory files
and should be fully oriented without re-exploring the codebase. If
that's not the case after your changes, the maintenance contract
was violated. Treat that as a regression and fix the doc.

To verify the document still matches reality:
```powershell
# Run after any non-trivial change
pytest tests/ -q
python scripts/validate_config.py
# Then re-read this doc and confirm tree + module sections are current
```
