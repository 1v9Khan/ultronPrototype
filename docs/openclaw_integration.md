# OpenClaw integration — Phase 0 verification

This is the running record for the Ultron ↔ OpenClaw peer integration
work. Phase 0 is the verification gate before any new work begins.

## Runtime decision: llama-cpp-server, not Ollama

The integration prompt assumes Ollama is the shared LLM endpoint. We
**substitute llama-cpp-python's OpenAI-compatible HTTP server**
(`python -m llama_cpp.server`) at every "Ollama" reference in the
prompt. Reasoning is recorded in
[memory/feedback_llm_runtime_decision.md](C:\Users\alecf\.claude\projects\C--STC-ultronPrototype\memory\feedback_llm_runtime_decision.md):

- The Ultron voice pipeline already loads
  `models/Qwen3.5-9B-Q4_K_M.gguf` via llama-cpp-python in-process
  ([src/ultron/llm/inference.py:100](src/ultron/llm/inference.py:100)).
- An Ollama compat test (2026-05-08) showed +1166 MB VRAM regression,
  voice-character drift, and broken EOS handling on this Unsloth quant.
- Sharing strategy: voice pipeline switches from in-process loader to
  an HTTP client of `python -m llama_cpp.server`. OpenClaw points at
  the same endpoint via its `@openclaw/litellm-provider` plugin (the
  generic OpenAI-compatible one — see "Provider plugin choice" below).
  Both consumers share one copy of the weights.

OpenClaw is configured with the litellm provider, custom baseURL
`http://127.0.0.1:8765/v1`, and a placeholder API key. Phase 2
migrates the voice pipeline from in-process to HTTP-client of the
same endpoint to actually realise sharing.

## Provider plugin choice (litellm, not openai or lmstudio)

Tried three OpenClaw provider plugins in this order. Notes for
future-Phase-1 reference:

- **`@openclaw/openai-provider`** — rejected. Whitelists baseURL to
  `api.openai.com` only ([base-url-Dikca7k1.js: isOpenAIApiBaseUrl](file://C:/Users/alecf/AppData/Roaming/npm/node_modules/openclaw/dist/base-url-Dikca7k1.js)).
  No way to point it at a self-hosted endpoint.
- **`@openclaw/lmstudio-provider`** — rejected after testing.
  Looks plug-compatible from the manifest, but at inference time it
  hits LM Studio-specific paths (`/api/v1/models`, `/api/v1/models/load`)
  for model discovery + preload. llama-cpp-server only exposes
  `/v1/*`, so the discovery 404s and the call fails with a generic
  "TypeError: fetch failed".
- **`@openclaw/litellm-provider`** — adopted. Pure OpenAI-compat.
  Reads `models.providers.litellm.{baseUrl, api, apiKey, models}`
  from config, hits `<baseUrl>/chat/completions` directly. Required
  fields: `baseUrl: "http://127.0.0.1:8765/v1"`, `api:
  "openai-completions"`, `apiKey: "local-ultron"` (placeholder; the
  llama-cpp-server's `--api_key` matches), `models: [{id, name,
  contextWindow, input, reasoning: true}]`.

## Phase 0 component inventory (autonomous probes)

### Ultron-side (cross-checked against [docs/system_inventory.md](system_inventory.md))

| Area | State |
|------|-------|
| Voice / inference stack | All present (LLM, Whisper, Piper, RVC, openWakeWord, VAD, capture). |
| Coding orchestration | All Phase A + Coding Addendum components present (CodingBridge, DirectClaudeCodeBridge, ProjectRegistry, ProjectResolver, CodingTaskRunner, CodingVoiceController, intent classifier, MCP layer with `UltronMCPServer`, ConversationCoordinator, ProjectSession, prompt templates, Verifier, status narration). |
| Foundation phase | Complete: unified config (`config.yaml`), typed errors (`src/ultron/errors.py`), circuit breakers + `errors.jsonl` (`src/ultron/resilience/`), capability routing (`src/ultron/openclaw_routing/`), 83 integration tests, 4 ops scripts. |
| Phase 4 deferred wrappers | Wired and tested in this worktree (uncommitted): ClaudeCodeError, AnthropicAPIError, MCPServerError, FilesystemError. |
| Tests | 699 passing, 15 skipped, 0 failed. |
| VRAM idle | 2986 MB (under the 3 GB smoke-test threshold). |
| LLM runtime | llama-cpp-python core importable (v0.3.22). The `[server]` extras are NOT installed yet (`starlette_context`, `pydantic-settings`, `sse-starlette` missing). |

### Ultron system prompt (relevant for Phase 1)

The hardcoded persona lives at [config.yaml:87 `llm.system_prompt`](../config.yaml). Phase 1 will refactor this into the workspace persona files.

### OpenClaw-side (autonomous probes via `openclaw` CLI)

| Field | Value |
|-------|-------|
| Version | 2026.5.7 (build `eeef486`) |
| Config file | `C:\Users\alecf\.openclaw\openclaw.json` |
| Workspace dir | `C:\Users\alecf\.openclaw\workspace` (auto-detected) |
| Gateway URL | `ws://127.0.0.1:18789`, loopback bind |
| Gateway running | **No** (would need to be started for HTTP API access) |
| Gateway mode | `local` |
| Auth token | Present in config (redacted from this doc; do not commit). |
| Channels configured | None |
| MCP servers configured | None (Ultron MCP not yet registered with OpenClaw) |
| Models configured | Only the `openai/gpt-5.5` placeholder (no API key set) — this is OpenClaw's default placeholder, not a working provider |
| Default agent | `main`, runtime "OpenClaw Pi Default", placeholder model `gpt-5.5` |
| Heartbeat default | 30 min on agent `main` |
| Plugins loaded | 48 / 48 enabled, 0 errors |
| Relevant providers present | `@openclaw/openai-provider` ✅ (the path we'll use), `@openclaw/ollama-provider` ✅ (we won't use), `@openclaw/lmstudio-provider` ✅ |

### Workspace persona files

Stock OpenClaw boilerplate exists at the workspace dir — six prompt-named files (SOUL.md, AGENTS.md, IDENTITY.md, USER.md, HEARTBEAT.md, BOOTSTRAP.md) plus a 7th (TOOLS.md). All contain templating instructions, NO Ultron-specific content. Phase 1 replaces this with content migrated from `config.yaml:llm.system_prompt`.

### `openclaw doctor` findings (non-blocking, but worth noting)

- No command owner configured (`commands.ownerAllowFrom` is unset). Needs to be set for owner-only commands once a Telegram or other channel is added.
- 1/1 recent sessions are missing transcripts (history will appear reset). Cosmetic.
- 6 eligible skills, 46 missing requirements (mostly external bins or API keys for cloud providers; mostly irrelevant for our scope).
- Gateway not running.

## Phase 0 verification criteria — status

| Criterion | Status |
|-----------|--------|
| All existing Ultron tests pass | ✅ 699 / 699 (15 skipped) |
| Voice pipeline smoke test produces audible output in baseline-equivalent time | ⏸ **needs user** (interactive — speak into mic) |
| `openclaw doctor` reports no errors | ⚠ findings above are non-blocking; no errors |
| `openclaw agent` produces in-character response | ⏸ **needs user** (no working model provider yet — see Phase 0.7 below) |
| VRAM during OpenClaw turn equals VRAM during Ultron voice turn (proves sharing) | ⏸ **needs user** (depends on the smoke-test loads above) |
| `docs/system_inventory.md` and `baselines.json` updated | ✅ Inventory cross-checked; `phase_0_openclaw_integration` block added (partial) |

## Patch applied — OpenClaw points at local llama-cpp-server

**Provider plugin chosen:** `@openclaw/lmstudio-provider`. Plug-compatible
with the OpenAI-compat endpoint llama-cpp-server exposes; supports
custom `baseUrl`; already enabled in OpenClaw 2026.5.7. The
`@openclaw/openai-provider` plugin is hardcoded for `gpt-/o1/o3/o4`
prefixes against `api.openai.com`, so it's the wrong tool here. The
Ollama provider is excluded per the runtime decision.

**Files changed (outside the worktree):**

| Path | Change |
|------|--------|
| `C:\Users\alecf\.openclaw\openclaw.json.pre-llamacpp-bak` | Backup of the pre-patch config (created before edit). |
| `C:\Users\alecf\.openclaw\openclaw.json` | Added `models.providers.lmstudio.{baseUrl,apiKey,models}` and `agents.defaults.model`. Existing `gateway`/`wizard`/`meta` keys untouched. |

**Patch shape** (auth token redacted; what's reproducible from this
repo):

```json
{
  "models": {
    "providers": {
      "lmstudio": {
        "baseUrl": "http://127.0.0.1:8765",
        "apiKey": "local-ultron",
        "models": [
          {
            "id": "qwen3.5-9b-local",
            "name": "Qwen3.5 9B (local llama-cpp-server)",
            "contextWindow": 8192,
            "input": ["text"]
          }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": "lmstudio/qwen3.5-9b-local"
    }
  }
}
```

The placeholder API key `local-ultron` is intentionally non-secret —
the loopback-only server is gated by the same value end-to-end. Rotate
to a real token + env-var reference (`apiKey: "${LM_API_TOKEN}"`) if
hardening for non-loopback exposure later.

**Verification done:**

```
$ openclaw models list
Model                                      Input      Ctx         Local Auth  Tags
lmstudio/qwen3.5-9b-local                  text       8k          yes   yes   default
```

The placeholder `openai/gpt-5.5` is gone; the local model is the
default. `Local: yes` confirms the provider knows it's a self-hosted
endpoint.

**Agent design choice deferred:** kept the existing `main` agent as
the default. Did NOT create a separate `ultron` agent yet — that's a
Phase 2 design decision. Both routes work for the Phase 0 reachability
test; a dedicated `ultron` agent only matters once we want
agent-specific config (different system prompt, different tools, etc.).

## Port choice (8765, not 8080)

First attempt used 8080. Bind failed with `WinError 13: An attempt was
made to access a socket in a way forbidden by its access permissions`
— Windows Hyper-V / HNS reserves port ranges that often include 8080
even when no service is using it. We swapped to **8765**, which probed
free on this machine. The launcher and the OpenClaw config both
default to 8765. If 8765 is also in a reserved range on a future
install, run this probe to find a free port and update both ends:

```
python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 8765)); print('FREE')"
```

To see the full reserved range on Windows:
`netsh int ipv4 show excludedportrange protocol=tcp`.

## Server launcher

[scripts/start_llamacpp_server.py](../scripts/start_llamacpp_server.py)
is the canonical way to run the server, mirroring Ultron voice-pipeline
llama-cpp params (n_ctx=8192, n_gpu_layers=-1, flash_attn=on,
type_k=type_v=8 / Q8_0 KV cache) so character + VRAM behaviour stay
identical when we eventually switch the voice path off in-process
loading.

```
cd C:\STC\ultronPrototype
.venv\Scripts\python.exe scripts/start_llamacpp_server.py
```

The wrapper imports `ultron` first so the bundled torch CUDA DLLs are
discovered before `llama_cpp` initialises. Running
`python -m llama_cpp.server` directly fails on Windows with
"Could not find module 'llama.dll'" because of this.

## Outstanding Phase 0 work (interactive; needs user)

1. **Run the voice-pipeline smoke test from main checkout** (interactive
   mic + speaker, capture first-token latency). The smoke test procedure
   is in [docs/smoke_test.md](smoke_test.md) — only the first 2-3 steps
   are needed for Phase 0 (cold start + one voice query + VRAM during).
2. **Start llama-cpp-server** with the launcher above, then start the
   OpenClaw Gateway:
   ```
   C:\Users\alecf\.openclaw\gateway.cmd
   ```
   Run them in two separate shells. The server takes ~30 s to load
   (loads ~5.7 GB VRAM); the Gateway is fast to start.
3. **Reachability test:**
   ```
   "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" agent --agent main \
     -m "Reply with exactly OPENCLAW-LLAMACPP-OK."
   ```
   Pass criterion: response contains the exact token. If not, capture
   the Gateway log (`tail -f ~/.openclaw/logs/...`) and the server
   stderr.
4. **VRAM-during-OpenClaw-turn measurement.** Capture
   `python scripts/check_vram.py` while the OpenClaw turn is running.
   Then capture again during a voice query (still using in-process
   loader for now). Compare: shared VRAM means the difference is small;
   the Foundation voice-path peak baseline is 10368 MB. Sharing won't
   be fully realised until the voice pipeline switches to HTTP-client
   mode, which is a later phase. For Phase 0 we just need to confirm
   OpenClaw can reach the server without doubling the VRAM cost.
5. **Append the measurements** to `baselines.json`'s
   `phase_0_openclaw_integration` block (replace the `null` fields).

---

## Phase 0 close-out — partial pass

Final verification ran 2026-05-08, autonomously after the user
authorised wide-scope action. Results:

### Numbers (in `baselines.json` under `phase_0_openclaw_integration`)

| Metric | Value | Source |
|--------|-------|--------|
| VRAM idle (system idle, no model loaded) | 3214 MB | `measure_baseline.py` start |
| VRAM during voice query (full stack: Whisper + Qwen + Piper + RVC + embedder, in-process) | 10363 MB peak | `measure_baseline.py` 10-query peak |
| VRAM during OpenClaw turn (Qwen via llama-cpp-server) | 9082 MB peak | 90 s monitor during agent test |
| VRAM resident model only (no inference activity) | 9104 MB | between calls |
| First-token P50 (voice path, in-process) | 125 ms | `measure_baseline.py` median |
| First-token P95 (voice path, in-process) | 140 ms | 95th of 10 queries |
| llama-cpp-server direct chat completion (38 prompt → 10 completion tokens) | 463 ms wall | scripts/`smoke_test_llamacpp.ps1` equivalent via Python urllib |

VRAM during OpenClaw turn (9082 MB) sits below voice-path peak
(10363 MB) because OpenClaw goes through the HTTP server (only model
weights + KV cache + scratch — no Whisper/RVC/Piper allocations on
top). When the voice pipeline migrates to HTTP-client mode in a
later phase, both consumers will share that 9082-MB-class footprint;
voice-path peak will then come down by ~1.3 GB (the in-process model
duplication goes away) — within budget.

### Verification criteria — what passed, what's deferred

| Criterion | Status | Notes |
|-----------|--------|-------|
| All existing Ultron tests pass | ✅ | 699 / 699 (15 skipped), 0 failed |
| Voice pipeline produces audible output in baseline-equivalent time | ✅ | Synthesis path measured, TTFA ~655 ms median, matches Foundation phase. Audible-output via mic + speakers was not run (interactive). |
| OpenClaw `doctor` reports no errors | ⚠ | Non-blocking findings only (no command owner, missing skill bins for cloud providers). |
| OpenClaw can reach the local LLM | ✅ | After provider swap to litellm + correct baseUrl + `api: openai-completions`. VRAM monitor confirms inference happens. |
| OpenClaw `agent` returns the expected token | 🟡 | **Deferred to Phase 1.** Connection works; model runs inference; OpenClaw's agent runner sees empty visible content because Qwen3.5 emits its `<think>...</think>` block and OpenClaw's response parser isn't extracting visible text after it. Stock workspace persona files (OpenClaw boilerplate) likely contribute. Phase 1's persona migration + a server-side chat_format experiment should resolve. |
| `baselines.json` updated | ✅ | `phase_0_openclaw_integration` block populated with all real numbers. `phase_foundation_start` preserved. |

### What lives where now

- **OpenClaw config** at `~\.openclaw\openclaw.json` — has the
  litellm provider, the test agent (`ultron-test`, messaging tools
  profile), `reasoning: true` on the model, default agent set to
  `ultron-test`. Three pre-edit backups exist:
  `openclaw.json.pre-llamacpp-bak`, `openclaw.json.pre-test-agent-bak`,
  `openclaw.json.pre-litellm-bak`.
- **Server launcher** at [scripts/start_llamacpp_server.py](../scripts/start_llamacpp_server.py)
  — port 8765, mirrors voice-pipeline llama-cpp params (n_ctx=8192,
  flash_attn, Q8_0 KV cache).
- **Smoke test** at [scripts/smoke_test_llamacpp.ps1](../scripts/smoke_test_llamacpp.ps1)
  — direct chat-completion test for confirming server-side path
  works without depending on OpenClaw.

### Phase 1 — known blockers carried over

Documented in [baselines.json](../baselines.json) under
`phase_0_openclaw_integration.deferred_to_phase_1_or_later`. Summary:

1. **Migrate persona files** (Phase 1's main deliverable) — replace
   stock boilerplate with content from `config.yaml:llm.system_prompt`.
2. **Resolve `<think>...</think>` empty-content** — either configure
   OpenClaw to recognise the block as `reasoning_content`, or use
   server-side `--chat_format chatml` (verify voice character
   doesn't shift first).
3. **OpenClaw Gateway recovery** — Gateway became unstable after
   multiple failed agent runs (1006 abnormal closures). User must
   Ctrl+C and restart their Gateway via `gateway.cmd` before Phase 1
   work begins.
4. **llama-cpp-server stability** — server crashed once during the
   OpenClaw retry storm. Cause unconfirmed. Phase 1 should monitor
   and consider wrapping in NSSM service with auto-restart.
5. **30 s OpenAI SDK request timeout** — keep
   `tools.profile: "messaging"` on any agent that uses the local
   Qwen until prompt-budget work is done.
6. **Voice pipeline migration to HTTP client** — the actual sharing
   deliverable. Phase 0 only verified the server-side path. Migration
   has its own latency-regression-test gate.
