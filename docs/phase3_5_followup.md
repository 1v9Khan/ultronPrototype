# Phase 3.5 — Finish unified-config subsystem migration

**Status:** OPEN. Architecture from Phase 3 is in place; this is the
mechanical-cleanup phase that retires the `config/settings.py` shim by
migrating the remaining subsystems to read directly from
`ultron.config.get_config()`.

**Why this exists:** Phase 3 of the Foundation prompt landed `config.yaml`
+ the pydantic loader + a thin re-export shim at `config/settings.py`.
Six subsystems read directly from `get_config()`; seven still go through
the shim. The shim is fully functional and behavior-preserving — but the
spec's "old scattered sources removed (no dead code)" verification
criterion isn't met until the shim disappears.

**Total remaining work:** ~150 reference-site edits across ~20 files,
following a pattern proven by 6 already-migrated subsystems.

---

## Where Phase 3 left off

Read these in order to get oriented:
- [docs/configuration.md](configuration.md) — config reference + migration-status table at the bottom
- [docs/config_discovery.md](config_discovery.md) — full per-key inventory (already-mapped, already-decided)
- [src/ultron/config.py](../src/ultron/config.py) — pydantic schema + loader
- [config.yaml](../config.yaml) — canonical values
- [config/settings.py](../config/settings.py) — the shim (this is what shrinks)

---

## What's done (don't redo)

These subsystems read directly from `get_config()`. Their `settings.X`
re-exports are already gone from the shim. Use them as migration
templates:

| Subsystem | Reference for the pattern |
|---|---|
| logging | [src/ultron/utils/logging.py:27-46](../src/ultron/utils/logging.py:27) |
| addressing + follow-up | [src/ultron/pipeline/orchestrator.py:276-283](../src/ultron/pipeline/orchestrator.py:276) — `_addr_cfg = get_config().addressing` cached in `run()` and reused; [scripts/review_addressing.py](../scripts/review_addressing.py) |
| web_search | [src/ultron/web_search/brave.py:48-67](../src/ultron/web_search/brave.py:48) — `__init__` defaults are `None`, look up inside body |
| llm | [src/ultron/llm/inference.py:91-150](../src/ultron/llm/inference.py:91) — covers ctor defaults + per-method `_llm_cfg = get_config().llm` |
| embeddings + memory + qdrant | [src/ultron/memory/embedder.py:55-68](../src/ultron/memory/embedder.py:55), [src/ultron/memory/qdrant_store.py:77-110](../src/ultron/memory/qdrant_store.py:77) |
| projections (config-aware logging) | [src/ultron/coding/projections.py:120-158](../src/ultron/coding/projections.py:120) |

---

## What's left

In recommended migration order (least → most coupled):

### 1. uncertainty (1 file, ~5 refs)

[src/ultron/uncertainty.py](../src/ultron/uncertainty.py). Single function
`apply()`. Should be a 5-line edit. Tests: `tests/test_uncertainty.py`.

### 2. audio + VAD (4 files, small)

- [src/ultron/audio/capture.py](../src/ultron/audio/capture.py)
- [src/ultron/audio/devices.py](../src/ultron/audio/devices.py)
- [src/ultron/audio/vad.py](../src/ultron/audio/vad.py)
- [src/ultron/audio/ring_buffer.py](../src/ultron/audio/ring_buffer.py) (if it reads settings)

Tests: `tests/test_audio.py`. After: remove the audio + vad blocks from
the shim (`SAMPLE_RATE`, `CHANNELS`, `BLOCKSIZE`, `DTYPE`, `AUDIO_DEVICE`,
`AUDIO_OUTPUT_DEVICE`, `BARGE_IN_*`, `RING_BUFFER_SECONDS`,
`VAD_THRESHOLD`, `MIN_SPEECH_DURATION_MS`, `MIN_SILENCE_DURATION_MS`,
`VAD_WINDOW_SAMPLES`).

### 3. wake_word (1 file)

[src/ultron/audio/wake_word.py](../src/ultron/audio/wake_word.py). Reads
`WAKE_WORD_*`. Tests: integration via orchestrator construction.

### 4. stt — Whisper (1 file)

[src/ultron/transcription/whisper_engine.py](../src/ultron/transcription/whisper_engine.py).
Tests: `tests/test_transcription.py`.

### 5. tts — Piper (1 file)

[src/ultron/tts/speech.py](../src/ultron/tts/speech.py). Tests:
`tests/test_tts.py`.

### 6. tts.rvc (1 file)

[src/ultron/tts/rvc.py](../src/ultron/tts/rvc.py). Reads `RVC_*`. Tests:
rvc-import path in `tests/test_tts.py`.

### 7. coding cluster (the big one — 14 files)

This is the largest and most intricate. ~80 reference sites. Subsystems
within:

- [src/ultron/coding/audit.py](../src/ultron/coding/audit.py)
- [src/ultron/coding/bridge.py](../src/ultron/coding/bridge.py)
- [src/ultron/coding/coordinator.py](../src/ultron/coding/coordinator.py) — **see warning below about `LLM_MAX_TOKENS` mutation**
- [src/ultron/coding/direct_bridge.py](../src/ultron/coding/direct_bridge.py)
- [src/ultron/coding/intent.py](../src/ultron/coding/intent.py)
- [src/ultron/coding/mcp_server.py](../src/ultron/coding/mcp_server.py)
- [src/ultron/coding/narration.py](../src/ultron/coding/narration.py)
- [src/ultron/coding/projects.py](../src/ultron/coding/projects.py)
- [src/ultron/coding/runner.py](../src/ultron/coding/runner.py)
- [src/ultron/coding/session.py](../src/ultron/coding/session.py)
- [src/ultron/coding/templates.py](../src/ultron/coding/templates.py)
- [src/ultron/coding/verification.py](../src/ultron/coding/verification.py)
- [src/ultron/coding/voice.py](../src/ultron/coding/voice.py)

Plus the test file [tests/coding/test_orchestration.py](../tests/coding/test_orchestration.py)
which references settings (likely just for `CODING_TEST_SANDBOX_PATH`).

Tests: every `tests/test_coding_*.py` and `tests/coding/*.py`. Heavy test
suite — run after every coding-file migration.

**KNOWN HAZARD (caught during Phase 3):**
[coordinator.py:895-900](../src/ultron/coding/coordinator.py:895) does
`settings.LLM_MAX_TOKENS = max_tokens` to temporarily override the
default before an LLM call, then restores. This is a pre-existing hack
that mutates module state. Removing `LLM_MAX_TOKENS` from the shim
without first refactoring the call to pass `max_tokens` explicitly per
call WILL break `tests/test_coordinator.py::test_decide_adjustment_escalates_on_conflict`.

The clean fix: change the LLM call site to pass `max_tokens=max_tokens`
explicitly (one of the LLM API methods accepts it as a kwarg already),
then drop the surrounding mutate-restore block. Then `LLM_MAX_TOKENS`
can come out of the shim. Phase 3 left that one constant in the shim
with a comment so the test passes; this is the correct step to undo
it cleanly.

### 8. orchestrator residuals + scripts (5 files)

Remaining `settings.X` sites in:

- [src/ultron/pipeline/orchestrator.py](../src/ultron/pipeline/orchestrator.py) — leftover memory / coding refs
- [scripts/benchmark.py](../scripts/benchmark.py)
- [scripts/download_models.py](../scripts/download_models.py)
- [scripts/maintenance.py](../scripts/maintenance.py)
- [scripts/migrate_memory_to_qdrant.py](../scripts/migrate_memory_to_qdrant.py)
- [scripts/measure_baseline_extended.py](../scripts/measure_baseline_extended.py) — added in Phase 0; only uses settings indirectly via the loader path

### 9. tests/test_memory_qdrant.py

Test file references settings.X. Migrate or refactor the test to use
`get_config()` directly.

---

## Migration recipe (proven on 6 subsystems)

For each subsystem, follow this exact loop. Tests gate every step.

```
1. Open the file. Note every `settings.X` reference inside.
2. Replace `from config import settings` with
   `from ultron.config import get_config` (and `resolve_path` if any
   path values are used). Add `from typing import Optional` if you'll
   be widening function signatures.
3. For class __init__ defaults like `param: T = settings.X`:
       a. Change to `param: Optional[T] = None`
       b. Inside the body, if param is None: `param = cfg.subsystem.x`
       c. If the value is a path: `resolve_path(...)` to get an absolute Path
4. For inline reads `settings.X`: replace with `get_config().subsystem.x`,
   ideally cached at the top of the function/method as `cfg = get_config().subsystem`
5. Run the relevant test file: `pytest tests/test_<subsystem>.py -q`
6. Run the full suite: `pytest tests/ -q`
   Both must pass before moving on.
7. Remove the now-unused settings.X re-export(s) from
   `config/settings.py`. Add a one-line comment in their place pointing
   at the new home. Run full tests AGAIN.
8. Commit-worthy checkpoint reached. Move to next subsystem.
```

**Critical:** never remove from the shim until you've migrated EVERY
caller of that constant. Use grep across `src/ tests/ scripts/` to
verify no remaining references before deleting.

```bash
# Always run before removing a shim line:
grep -rn "settings\.<NAME>" src/ tests/ scripts/ | grep -v __pycache__
```

If grep returns anything, that file isn't migrated yet. Don't remove
the shim line.

---

## End state

When Phase 3.5 is complete:

- `config/settings.py` is gone (or contains only the HF cache pre-init,
  which legitimately must run before any HF import — see Phase 3
  discovery doc §15)
- Zero references to `from config import settings` anywhere in the tree
- Zero `settings.X` references anywhere in `src/`, `tests/`, `scripts/`
- All test pass: `pytest tests/ -q` → 390+ passed
- VRAM and latency unchanged from Phase 3 finish state (idle ~2640 MB)

The HF cache pre-init either stays in `config/__init__.py` as a small
bootstrap module, or moves to `src/ultron/config_bootstrap.py` and is
imported once by `src/ultron/__init__.py`. Either is fine; pick whichever
makes the entry-point story cleaner.

---

## Why we stopped here

The architectural value of Phase 3 (single source of truth via
`config.yaml`, typed loader, env-var substitution, validation, hot-reload
support) is fully realized. The shim is a legitimate transitional
artifact — it does NOT introduce a parallel source of truth, it only
re-exports values from `config.yaml` under their legacy names so
unmigrated callers keep working unchanged.

Phase 4 (comprehensive error handling) and Part 5+ (capability routing,
integration testing, polish) can land cleanly against either state — the
shim is purely a code-cleanliness concern, not a functional one.

Recommended ordering for a fresh session: do Phase 3.5 (this doc) BEFORE
Phase 4, since Phase 4's error-handling pass naturally touches many of
the same subsystems and you'll want them on the direct `get_config()`
path before adding more complexity. But it's not strictly required.

---

## Quick-start for the next session

```bash
# Validate state
cd C:\STC\ultronPrototype\.claude\worktrees\friendly-jang-ed9a06
C:/STC/ultronPrototype/.venv/Scripts/python.exe -m pytest tests/ -q
# Should be: 390+ passed, 16 skipped, 0 failed

# Count remaining sites
grep -rn "from config import settings\|settings\." src/ tests/ scripts/ \
  | grep -v __pycache__ | wc -l
# Phase 3 finish state: 201

# Confirm config loads cleanly
C:/STC/ultronPrototype/.venv/Scripts/python.exe -c "
from ultron.config import get_config
c = get_config()
print('addressing.warm:', c.addressing.warm_mode_duration_seconds)
print('llm.provider:', c.llm.provider)
print('qdrant.data_dir:', c.qdrant.data_dir)
"
# Expected: 30.0 / llama_cpp / data/qdrant

# Start with subsystem 1 (uncertainty.py — smallest), follow recipe.
```

Read [project_ultron_foundation.md](<ai-memory-dir>\project_ultron_foundation.md) and
[feedback_llm_runtime_decision.md](<ai-memory-dir>\feedback_llm_runtime_decision.md)
in memory before touching anything LLM-related.
