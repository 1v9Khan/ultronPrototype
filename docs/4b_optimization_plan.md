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

**User-led steps remaining:**
1. Run [scripts/verify_voice_character_4b.py](../scripts/verify_voice_character_4b.py)
   (Stage E). If Ultron sounds unchanged on the five A/B queries → continue.
2. Run the 16-step real-stack smoke test in [docs/smoke_test.md](smoke_test.md).
3. **Flip** `config.yaml`:
   ```yaml
   llm:
     preset: "qwen3.5-4b"                                  # <<< change
     model_path: "models/Qwen3.5-4B-Q4_K_M.gguf"           # <<< change
     draft_model_path: "models/Qwen3.5-0.8B-Q4_K_M.gguf"   # <<< change (was null)
     n_ctx: 16384                                          # <<< change (was 8192)
   ```
   (preset alone is insufficient — explicit YAML values win over preset
   defaults via `model_fields_set`. Update all four for clarity.)
4. Re-run pytest sweep + `validate_config.py` to confirm.
5. Optional: start `scripts/start_llamacpp_server.py --from-config` and
   run the speculative-decoding bench (`scripts/_bench_llm_http.py`)
   for an additional throughput measurement.

**Rollback path:** revert the four lines above to the current
`qwen3.5-9b` values. The 9B GGUF stays in `models/` for swap-back at
any time. `llm.rag.position` and the `enable_thinking` parameter
remain available regardless of preset (they're orthogonal to the
model swap).

## Items 4-8 — second-pass optimization

Lower priority; defer until Stages A-H are stable. Each is additive
and behind its own config flag so they can be enabled / rolled back
independently.

### Item 4 — LLMLingua-style RAG compression
Token-level compression of retrieved Qdrant snippets and Jina-fetched
articles before injection. The 0.8B draft model from Stage C doubles
as the perplexity scorer (no extra VRAM). Start at 1.5x compression,
tune from there.

### Item 5 — IRMA-style tool-call input reformulation
Wrapper between intent classification and the disambiguation LLM
call. Enriches the raw utterance with: recently-used tools (avoid
suggesting failed ones), active session state summary, relevant
routing rules. Paper claims +12-19% on ambiguous tool calling.

### Item 6 — Self-consistency for high-stakes calls
Apply N-sample majority vote (N=3, temperature 0.7-1.0) to:
- Coding correction-prompt generation
- HYBRID_TASK decomposition
- Pre-flight uncertainty when initial confidence is borderline

3x token cost on those specific calls; not on the voice hot path.

### Item 7 — Canonical path monitor (coding sessions)
Per-session "canonical adherence" tracker. If a session has 3+
unexpected tool calls in the first 30% of execution, abort and
restart with a cleaner prompt. Paper claims +8.8 percentage points
on long coding sessions.

### Item 8 — Block-and-revise on OpenClaw tool calls
Pre-flight validator: "given the user's stated goal, does this tool
call advance it?" If no, block + ask the agent to revise. Extends
the existing coding verification pattern.

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
