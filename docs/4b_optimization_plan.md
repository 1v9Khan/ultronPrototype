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

### Stage A — Multi-model config schema
Add `llm.preset: "qwen3.5-9b" | "qwen3.5-4b" | "custom"` (defaulting
to `"qwen3.5-9b"` until 4B is verified). Preset resolves
`model_path` + `draft_model_path` + `n_ctx` automatically. Keep
`llm.model_path` working for `preset: "custom"` (back-compat for
tests + advanced users).

### Stage B — Download GGUFs
Add to `scripts/download_models.py`:
- `Qwen3.5-4B-Q4_K_M.gguf` from `unsloth/Qwen3.5-4B-GGUF`
- `Qwen3.5-0.8B-Q4_K_M.gguf` from `unsloth/Qwen3.5-0.8B-GGUF` (draft model)

Verify SHA256 against Unsloth's release. Keep 9B intact.

### Stage C — Speculative decoding launcher
`scripts/start_llamacpp_server.py` gets `--model-draft`, `--draft-max`,
`--draft-min` flags. Mirrors the user's pasted recipe:

```
llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf \
  --model-draft models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --draft-max 8 --draft-min 4 \
  -ngl 99 -fa --jinja -c 16384
```

Tests for both single-model and spec-decoding launches.

### Stage D — 4B+spec baseline
`scripts/measure_baseline.py` against the 4B + spec setup. Compare
TTFT/TTFA/throughput to the 9B baseline. Decision gate: 4B+spec
must beat 9B on TTFT (paper claims 1.5-2x throughput; should be 50-90 ms TTFT).

### Stage E — Voice character verification (interactive)
Five representative voice queries through the live stack with the 4B.
User confirms Ultron sounds unchanged.

### Stage F — Selective thinking mode
Add `enable_thinking: bool` parameter to `LLMEngine.generate*`. Map
intent types to thinking on/off:

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

### Stage G — Position-aware RAG injection
Move retrieved Qdrant memories from the system-prompt fold-in
(currently in `_build_messages`) to the position just before the user
query. Recency bias makes this the strongest attention zone. Expected
+10-20% recall improvement.

Composition order after the change:
1. System prompt + persona (start)
2. Conversation history (middle)
3. Retrieved memories ranked by relevance (just before user query) — NEW
4. Current user query (end / recency position)

### Stage H — End-to-end regression sweep
Full pytest, full `measure_baseline.py`, full smoke test (16 steps from
[docs/smoke_test.md](smoke_test.md)). Decision gate: voice character
unchanged, TTFT ≤ 9B baseline, no test regressions. Flip
`llm.preset` default from `"qwen3.5-9b"` to `"qwen3.5-4b"`.

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
