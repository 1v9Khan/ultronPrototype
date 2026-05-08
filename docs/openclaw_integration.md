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
  the same endpoint via its `@openclaw/openai-provider` plugin (custom
  `baseURL`). Both consumers share one copy of the weights.

This means Phase 2 will configure the OpenAI provider in OpenClaw with
a custom baseURL (`http://127.0.0.1:8080/v1`) and a placeholder API
key, NOT the Ollama provider.

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

## Outstanding Phase 0 work (interactive; needs user authorization)

1. **Install llama-cpp-server extras** in the venv:
   ```
   C:\STC\ultronPrototype\.venv\Scripts\pip.exe install "llama-cpp-python[server]"
   ```
2. **Start llama-cpp-server** pointing at the existing GGUF (loads ~5.7 GB VRAM). Suggested invocation (mirroring [src/ultron/llm/inference.py](../src/ultron/llm/inference.py) params for character preservation):
   ```
   python -m llama_cpp.server \
     --model C:\STC\ultronPrototype\models\Qwen3.5-9B-Q4_K_M.gguf \
     --n_gpu_layers -1 --n_ctx 8192 --port 8080 --api_key local-ultron
   ```
3. **Configure OpenClaw** to point at the local server. Approach: define a model entry under `models.providers.openai` with `baseURL: http://127.0.0.1:8080/v1` and `apiKey: ${LLAMACPP_API_KEY}`, then a model definition like `qwen3.5-9b-local` referencing that provider. Then set the agent's default model to `qwen3.5-9b-local`. (Exact schema to verify against current OpenClaw docs.)
4. **Start the Gateway** (`gateway.cmd` is at `C:\Users\alecf\.openclaw\gateway.cmd`).
5. **Run `openclaw agent --agent main -m "Reply with exactly OPENCLAW-LLAMACPP-OK."`** — verify the response.
6. **Run a representative voice query** through the Ultron orchestrator (interactive) and capture VRAM peak + first-token latency.
7. **Compare** OpenClaw-turn VRAM vs voice-query VRAM — should be within a few hundred MB (proves sharing).

## Open questions for the user

1. **OK to install `llama-cpp-python[server]` extras now?** This adds `starlette_context`, `pydantic-settings`, `sse-starlette` to the venv.
2. **OK to register the Ultron MCP server (`UltronMCPServer`) with OpenClaw via `openclaw mcp set ultron-mcp …`?** The MCP server currently runs over SSE on its own port (Foundation Phase A); this would write to `C:\Users\alecf\.openclaw\openclaw.json`.
3. **Should I draft the OpenClaw config patch as a JSON diff so you can apply it manually**, instead of editing `openclaw.json` directly? Given there's an auth token already in that file, I'd prefer not to handle it directly.
4. **OK to commit the Phase 4 deferred wrapper work + Phase 0 inventory doc to a feature branch on origin?** The Phase 4 wrappers are still uncommitted in this worktree.
