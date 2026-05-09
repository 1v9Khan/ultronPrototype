# 4B-model optimization plan (deferred to next session)

Plan for downgrading the primary LLM from Qwen3.5-9B Q4_K_M (~5.7 GB)
to Qwen3.5-4B Q4_K_M (~2.7 GB), then recovering most of the 9B's
quality via eight targeted optimizations.

The 9B GGUF stays in `models/` so we can swap back. The active model
will be controlled by config.

## Why downgrade

- VRAM headroom: 9B leaves ~1 GB headroom under our 11.5 GB cap.
  4B leaves ~4 GB — room for a 0.8B speculative-decoding draft, RAG
  reranker, or vision encoder later.
- Speculative decoding works well for 4B + 0.8B but offers diminishing
  returns at 9B (draft model adds VRAM without speedup proportional
  to the slowdown).
- 4B's prefill is faster — gives latency margin to apply techniques
  that add tokens (selective thinking mode, self-consistency).

## Migration plan (sequenced; verification gate at every step)

### Stage A — Multi-model config schema ✅ DONE
Added `llm.preset: "qwen3.5-9b" | "qwen3.5-4b" | "custom"` (default
`"qwen3.5-9b"`; flips to `"qwen3.5-4b"` after Stage H gate passes).
Preset auto-resolves `model_path` + `draft_model_path` + `n_ctx` via
the `LLM_PRESETS` table in
[src/ultron/config.py](../src/ultron/config.py), but **only when those
keys are absent from the YAML** — explicit user values always win
(via `model_fields_set` check in the after-validator). `preset: "custom"`
disables auto-resolution entirely (back-compat for tests + advanced
users). Verification:
- 13 new tests in [tests/test_llm_preset.py](../tests/test_llm_preset.py)
  cover all three presets, mixed-mode override, YAML round-trip, and
  invalid-preset rejection.
- Full pytest sweep: 749 passed (+13 from baseline 736), 16 skipped, 0 failed.
- `python scripts/validate_config.py`: passes against the updated
  `config.yaml` (which now includes an explicit `preset: "qwen3.5-9b"`
  key for clarity).

### Stage B — Download GGUFs ✅ DONE
- Added 4B + 0.8B sections to
  [scripts/download_models.py](../scripts/download_models.py) (sections
  [2/7] and [3/7]; idempotent, skips files already on disk).
- Pulled both files into `C:\STC\ultronPrototype\models\`:
  - `Qwen3.5-4B-Q4_K_M.gguf` — 2,740,937,888 bytes
  - `Qwen3.5-0.8B-Q4_K_M.gguf` — 532,517,120 bytes
- Validated structurally via `vocab_only` load — both report
  `arch=qwen35`, `n_vocab=248320` (same as 9B, which is required for
  the 0.8B to serve as a speculative draft for the 4B).
- Recorded SHA256 of all three GGUFs in
  [docs/model_checksums.md](model_checksums.md) for re-pull
  verification (Unsloth doesn't publish a centralised checksum file;
  HF Hub's content-addressed transfer plus this local record is the
  integrity story).
- 9B kept intact in `models/` for swap-back.

### Stage C — Speculative decoding launcher ✅ DONE
[`scripts/start_llamacpp_server.py`](../scripts/start_llamacpp_server.py)
gained three new flags:
- `--model-draft <path>` — optional draft GGUF for speculative
  decoding. Default: None (no spec decoding; back-compat).
- `--draft-num-pred-tokens N` — how many tokens the draft predicts
  before the target verifies. Default: 8 (matches the recipe's
  `--draft-max 8`). Ignored when `--model-draft` is unset.
- `--from-config` — overlay launcher params from `config.yaml:llm`
  via `LLMConfig` (so the active preset's `model_path`, `n_ctx`, and
  `draft_model_path` are picked up automatically). CLI flags still
  override the overlay for ad-hoc swaps without editing YAML.

**API discrepancy from the recipe:** the upstream `llama.cpp` CLI
exposes both `--draft-max` and `--draft-min`, but
`llama-cpp-python==0.3.22` only surfaces a single combined parameter
(`draft_model_num_pred_tokens`). I mapped `--draft-num-pred-tokens`
to that and dropped `--draft-min` (no equivalent in the Python
server). This is documented in the launcher docstring + this plan.
If a future llama-cpp-python version exposes the min/max pair, we can
add `--draft-min` then.

Refactor: `_build_arg_parser`, `_resolve_kwargs`, and `_config_overlay`
were extracted from `main()` so the CLI logic is testable without
loading CUDA DLLs or starting uvicorn.

Verification:
- 13 new tests in
  [tests/test_start_llamacpp_server.py](../tests/test_start_llamacpp_server.py)
  cover help-render, default-args back-compat, draft flags,
  draft-num-pred-tokens override, draft-num-pred-tokens-ignored-without-draft,
  --from-config overlay (4b + 9b), and CLI-override-wins.
- `python scripts/start_llamacpp_server.py --help` renders without errors.
- Full pytest sweep: 762 passed (+13 from Stage B's 749), 16 skipped,
  0 failed.

Tests for both single-model and spec-decoding launches in place. Stage D
will measure the actual TTFT delta on the live stack.

### Stage D — 4B baseline ✅ DONE (4B alone; 4B+spec deferred to Stage H sweep)
First-pass measurement: 4B alone via the in-process llama-cpp-python
loader (no speculative decoding yet), driven by
`scripts/measure_baseline.py` with `ULTRON_LLM_MODEL_PATH` overriding
the default 9B path. Recorded snapshot:
[baselines_4b_q4_in_process.json](../baselines_4b_q4_in_process.json).

| Metric | 9B baseline | **4B (no spec)** | Δ |
|---|---|---|---|
| TTFT median | 109 ms | **86 ms** | **−21%** |
| TTFA median (Whisper + LLM TTFT + first-sentence synth) | 609 ms | **546 ms** | **−10%** |
| Whisper median | 109 ms | 109 ms | unchanged (same model) |
| TTS synth median | (similar) | 343 ms | unchanged |
| LLM-only VRAM delta | +5550 MB | **+3266 MB** | **−2284 MB** |
| Full-stack VRAM peak | 10370 MB | **7825 MB** | **−2545 MB (24%)** |
| VRAM headroom under 11500 MB cap | ~1130 MB | **~3675 MB** | +2545 MB |

**Decision gate: PASSED.** 4B alone already beats 9B on TTFT and frees
~2.5 GB VRAM. With that headroom, the speculative-decoding 0.8B draft
(~500 MB) and any future RAG reranker / vision encoder fit comfortably.

The full 4B + 0.8B speculative-decoding measurement (via
`scripts/start_llamacpp_server.py --from-config` + `_bench_llm_http.py`)
is part of Stage H's full regression sweep — it requires running the
HTTP server in foreground while a separate benchmark process pings it,
which is more orchestration than is needed to clear Stage D's gate.

(Earlier Stage D draft expected `4B+spec must beat 9B on TTFT`. The
4B-alone measurement above already clears that bar; Stage H will
confirm the spec-decoding throughput gain is additive on top.)

### Stage E — Voice character verification (interactive) ⏳ READY FOR USER
Helper script written:
[scripts/verify_voice_character_4b.py](../scripts/verify_voice_character_4b.py).
Runs five representative queries through the live stack, once with 4B
and once with 9B for direct A/B. Output is a side-by-side table of
first-sentence responses + TTFT.

To run:

```powershell
cd C:\STC\ultronPrototype
.venv\Scripts\python.exe scripts\verify_voice_character_4b.py
```

The plan's verification criterion is qualitative — "user confirms
Ultron sounds unchanged". The script collects the data; the gate is
your judgement. If the 4B sounds the same (cadence, terseness,
character), Stage E passes and Stage H can flip the default. If not,
the rollback is to keep the `qwen3.5-9b` preset (one-line YAML revert).

### Stage F — Selective thinking mode ✅ DONE (parameter wired; per-call routing in Stages G/H)
Added `enable_thinking: Optional[bool] = None` parameter to
`LLMEngine.generate(...)` and `LLMEngine.generate_stream(...)`. Plumbs
through to llama-cpp-python's
`chat_template_kwargs={"enable_thinking": <bool>}` (in-process) and to
the OpenAI-compat HTTP payload's `chat_template_kwargs` field
(http_server runtime). The 4B GGUF's chat template was verified to
support both `enable_thinking` and the `/think` / `/no_think`
soft-switch directives — we use the cleaner `chat_template_kwargs`
route.

`None` (default) preserves bit-for-bit back-compat — no extra kwarg is
set, so today's behaviour ("thinking on" via Qwen3.5's template
default) is unchanged. Per-call routing (which intent types pass
`False` vs `True`) is wired into the orchestrator and projection-driven
callers in Stage G + Stage H — this stage is just the parameter plumbing.

Map (applied incrementally in Stage G/H):

| Intent | Thinking |
|--------|----------|
| Simple conversation | OFF |
| Acknowledgment phrases | OFF |
| Pre-flight uncertainty pass | OFF (already structured) |
| Tool-routing decisions | ON for ambiguous, OFF for clear |
| Clarification decisions | ON |
| Correction-prompt generation | ON |
| HYBRID_TASK decomposition | ON |
| Adjustment context processing | ON |

Voice path defaults OFF (latency matters). Background workers can opt in.

Verification: 11 new tests in
[tests/test_llm_enable_thinking.py](../tests/test_llm_enable_thinking.py)
cover the helper, both runtimes, both methods, and back-compat
(default omits the kwarg). Full pytest sweep: 773 passed (+11 from
Stage D 762), 16 skipped, 0 failed.

### Stage G — Position-aware RAG injection ✅ DONE
Refactored `LLMEngine._build_messages` to inject retrieved Qdrant
memories at the position dictated by `cfg.llm.rag.position`:

- `"system"` — legacy fold-in to the leading system message (preserved
  for back-compat / rollback path).
- `"recency"` — **new default** — prepended to the final user message,
  putting RAG content in the strongest-attention zone right before
  the user query. Per the plan, +10–20% recall on injected memories
  on the 4B.

Qwen3's chat template rejects a second `system` message ("System
message must be at the beginning"), so the new path uses the user
message rather than emitting a separate context message — sidesteps
the template constraint while still placing content in the recency
zone.

Composition after the change (with `position: "recency"`):
1. System prompt + persona (start)
2. Conversation history (middle)
3. RAG-block + current user query (end / recency position)

Verification:
- 7 new tests in
  [tests/test_llm_rag_position.py](../tests/test_llm_rag_position.py)
  cover both positions, no-snippets fallback, retrieve-failure
  fallback, helper invariants, and history-not-duplicated guard.
- Existing `test_llm_persona_source.py` (which exercises
  `_build_messages` for persona resolution) still passes — it doesn't
  use memory so the RAG path isn't triggered.
- Refactored `_retrieve_rag_snippets()` and `_format_rag_block()`
  helpers extracted for testability + reuse if a third position is
  ever added.
- `python scripts/validate_config.py`: passes against the updated
  `config.yaml` (which now has explicit `llm.rag.position: "recency"`).
- Full pytest sweep: 780 passed (+7 from Stage F 773), 16 skipped,
  0 failed.

### Stage H — End-to-end regression sweep ⏳ READY (last step is the flip)

**Already done:**
- Full pytest sweep at HEAD `ae096e8`: **780 passed, 16 skipped, 0
  failed**. No regressions.
- Stage D's `measure_baseline.py` 4B in-process baseline:
  TTFT median 86 ms (vs 9B 109 ms). Gate cleared.
- VRAM peak under load 7825 MB (vs 9B 10370 MB; ~2.5 GB headroom).
- Schema, paths, kwargs, and config all back-compat: every existing
  test passed unchanged after Stages A–G.

**On-the-fly switching infrastructure (4B plan late addition):**
The flip is now a single switch via any of **four** paths:

```text
4. Voice command (no keyboard / file edit needed):
   "Ultron, switch to the 4B"
   "Ultron, use the 9B model"
   "Ultron, load 4B"
   "Ultron, swap to nine B"
```

The voice command:
1. Routes to `RoutingIntentKind.MODEL_SWITCH` (rule-based classifier;
   54 patterns covered including Whisper homophones like "for B"
   for "four B").
2. The `CapabilityVoiceController` calls
   `LLMEngine.reload_for_preset(target)`, which builds the new
   `Llama` instance BEFORE releasing the old one (failure-safe — a
   missing GGUF leaves the engine intact). On success the env var
   `ULTRON_LLM_PRESET` is updated so subsequent reload paths agree;
   in-memory history is cleared (different `n_ctx`); the cancel
   flag is reset.
3. Ultron speaks the result: `"Switched to the 4B."` or
   `"I couldn't switch to the 4B. Reason: …"`.

Mid-clarification utterances are suppressed — model-swap commands
during an active clarification dialogue would interrupt work-in-
progress, so the classifier requires `has_pending_clarification ==
False` to fire MODEL_SWITCH. Active coding tasks DO permit a swap
(the user explicitly asked).

The reload blocks ~1.7 s for 4B, ~3–5 s for 9B (in-process load
times measured in Stage D). VRAM peaks briefly at `old + new` size
during the swap (e.g. 4B → 9B: 2.5 + 5.3 = 7.8 GB) before the old
instance is released — comfortably under the 11.5 GB cap.

The other three paths (file edit, env var, swap helper) remain
available for non-voice contexts:

```yaml
# 1. Edit config.yaml — ONE line
llm:
  preset: "qwen3.5-4b"   # was "qwen3.5-9b"
```

```powershell
# 2. Env var, no file edit needed
$env:ULTRON_LLM_PRESET = "qwen3.5-4b"; python -m ultron

# 3. Swap helper script (validates GGUFs exist + atomic write)
python scripts/swap_llm_preset.py qwen3.5-4b
python scripts/swap_llm_preset.py --status   # show current preset
python scripts/swap_llm_preset.py --list     # show all presets + paths
```

The env-var path also clears any explicit `model_path` /
`draft_model_path` / `n_ctx` overrides in YAML so the preset's table
values fully take effect. Override that with
`ULTRON_LLM_PRESET_KEEP_OVERRIDES=1` if you want YAML-pinned values to
survive the env-var preset switch.

VRAM target follows the preset: `check_vram.py` now reads
`ULTRON_LLM_PRESET` / `config.yaml:llm.preset` and reports the matching
soft target (`qwen3.5-9b → 9216 MB`, `qwen3.5-4b → 6700 MB`,
`custom → 9216 MB` fallback). The hard cap (11500 MB) is unchanged —
it's the GPU physics.

**User-led steps remaining:**
1. Run [scripts/verify_voice_character_4b.py](../scripts/verify_voice_character_4b.py)
   (Stage E). If Ultron sounds unchanged on the five A/B queries → continue.
2. Run the 16-step real-stack smoke test in [docs/smoke_test.md](smoke_test.md).
3. Flip via any of the three paths above. Recommended:
   `python scripts/swap_llm_preset.py qwen3.5-4b` (validates GGUFs,
   atomic write, re-validates schema).
4. Re-run pytest sweep + `validate_config.py` to confirm.
5. Optional: start `scripts/start_llamacpp_server.py --from-config` and
   run the speculative-decoding bench (`scripts/_bench_llm_http.py`)
   for an additional throughput measurement.

**Rollback path:** swap back via the same three paths (e.g.
`python scripts/swap_llm_preset.py qwen3.5-9b`). The 9B GGUF stays in
`models/`. `llm.rag.position` and the `enable_thinking` parameter
remain available regardless of preset (orthogonal to the model swap).

## Items 4–8 — second-pass optimization ✅ MACHINERY SHIPPED (all flags default OFF)

All five items landed as additive, flag-gated, fully-tested machinery.
Default OFF on every flag — live behaviour byte-for-byte unchanged
until the user opts each in. Each is an independent commit.

### Item 4 — LLMLingua-style compression ✅
[`src/ultron/llm/compression.py`](../src/ultron/llm/compression.py).
Heuristic compressor (no extra model) drops stopwords (negation-
preserving), contractions, redundant punctuation, repeated sentence
signatures. `Compressor(perplexity_scorer=...)` kwarg lets a future
drop-in plug a real perplexity model — the Stage C speculative-
decoding 0.8B is the natural fit. Wired into
`LLMEngine._format_rag_block` and `format_sources_for_prompt`. Per-
surface flags: `compress_rag` / `compress_web` / `compress_history`
(history default OFF — has user voice). 26 tests. Config:
`llm.compression.{enabled, target_ratio, compress_rag, compress_web,
compress_history}`.

### Item 5 — IRMA-style input reformulation ✅
[`src/ultron/openclaw_routing/irma.py`](../src/ultron/openclaw_routing/irma.py).
`InputReformulator` is a pure-text shaper (no LLM call) — wraps the
disambiguator's input with optional recent-decision context, active
session summary, and routing hints. Wired into `IntentDisambiguator`
behind `routing.irma.enabled`. Reformulation failure falls back to the
legacy prompt — disambiguator path never crashes. 15 tests. Config:
`routing.irma.{enabled, max_recent_decisions}`.

### Item 6 — Self-consistency for high-stakes calls ✅
[`src/ultron/llm/self_consistency.py`](../src/ultron/llm/self_consistency.py).
Three aggregator families: `majority_vote_text`, `majority_vote_json`,
`majority_vote_label`. `run_self_consistency(sampler, n, temperature,
aggregator)` driver. Wired into `HybridTaskDecomposer.decompose` (JSON
voting). Other call sites (web-gating preflight, IntentDisambiguator)
ready-to-extend with the same pattern. 27 tests. Config:
`llm.self_consistency.{enabled, n, temperature, disabled_sites}`.

### Item 7 — Canonical-path monitor for coding sessions ✅
[`src/ultron/coding/canonical_monitor.py`](../src/ultron/coding/canonical_monitor.py).
`CanonicalPathMonitor.observe(event)` ingests `TaskEvent`-shaped
objects (duck-typed; works with both the dataclass and dicts in
tests). Tracks tool_use events; signals abort when off-canonical
count crosses the threshold inside the early window. Latches once
triggered. Conservative defaults (3 off-canonical in first 10 calls)
because false-positive aborts are real. 17 tests. Wiring into the
runner is intentionally NOT in this commit — machinery first, abort-
and-restart-with-cleaner-prompt flow as a follow-up. Config:
`coding.canonical_monitor.{enabled, off_canonical_threshold,
early_window_calls}`.

### Item 8 — Block-and-revise validator on OpenClaw tool calls ✅
[`src/ultron/openclaw_routing/block_and_revise.py`](../src/ultron/openclaw_routing/block_and_revise.py).
`ToolCallValidator(llm).validate(goal, tool_name, tool_args)` runs a
short pre-flight LLM check ("does this tool call advance the goal?")
returning `ValidationResult(allow, reason, verdict, raw_response)`.
Fails open on no-LLM / exception / unparseable response — never
hard-blocks on flaky LLM. Wiring into `OpenClawDispatcher` is NOT in
this commit (dispatcher is currently stubbed in Phase 5; the
validator's interface stays identical when the real Gateway lands).
14 tests. Config: `openclaw.block_and_revise.enabled`.

### Combined verification across Items 4–8

99 new tests across the five items. Live system behaviour unchanged
because every flag is OFF by default. Integration sweeps after each
item all green. To enable any item live, flip its flag in
`config.yaml` (or test the change in isolation via the env-var
overrides per item).

## What's already done

- KV cache q8_0 quantization (Foundation phase — `llm.kv_cache_type: 8`).
- Flash attention enabled (`llm.flash_attn: true`).
- Persona-split architecture (Phase 1) — user_facing prompt
  preserved at original size; background workers get plain task
  prompts. This gives item 1's "selective thinking" a clean intent
  routing point.
- Multi-mode `PersonaLoader` (Phase 1) — natural extension point for
  the `enable_thinking` parameter.
- HTTP-client mode for `LLMEngine` (opt-in; keeps in_process default
  to avoid the +71 ms TTFT regression measured in Phase 0).

## Risk + verification

The user's directive: "rigorous testing at every single stage to
properly integrate the model into our framework and make improvements
without breaking everything and preserving functionality."

Each stage has a verification gate:

- Stage A-C: pytest sweep + the launcher's `--help` rendering.
- Stage D: full `measure_baseline.py`, compare TTFT/TTFA/VRAM to 9B.
- Stage E: user-driven voice-quality check (5 queries).
- Stage F-G: targeted regression tests + `measure_baseline.py`.
- Stage H: full pytest + 16-step smoke test + flip default.

If any gate fails, back out the offending change before proceeding.

## Resumption notes

When resuming this work in a fresh session:

1. Read [docs/phase_1_summary.md](phase_1_summary.md) for the persona
   wiring (Stage F's intent-thinking map plugs into PersonaLoader's
   mode infrastructure).
2. Read this file.
3. Verify the current model is still 9B via
   `python scripts/check_vram.py` after `python -m ultron` cold-start
   (peak ~10 GB = 9B; ~7 GB = 4B already swapped).
4. Start with Stage A.
