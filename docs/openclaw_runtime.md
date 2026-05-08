# OpenClaw runtime operations

How to run the shared local-LLM stack, what the agents are for, and
the constraints they're locked under. Updated at the close of Phase 1
follow-up work (2026-05-08).

## Process layout

```
[ llama-cpp-server (port 8765) ]──┬──→ in-process LLMEngine (voice path)
                                  ├──→ OpenClaw Gateway (port 18789)
                                  │       ├── ultron-test     (worker)
                                  │       ├── ultron-main     (user-facing)
                                  │       └── ultron-heartbeat
                                  └──→ direct curl / scripts
```

llama-cpp-server holds the only resident copy of
`models/Qwen3.5-9B-Q4_K_M.gguf`. Both Ultron's voice pipeline (when
opted in) and OpenClaw point at it via OpenAI-compat HTTP. One model,
one VRAM allocation.

## Starting the stack

**llama-cpp-server (preferred — supervised, auto-restarts on crash):**
```
cd C:\STC\ultronPrototype
.venv\Scripts\python.exe .claude\worktrees\fervent-meitner-98bfe7\scripts\supervised_llamacpp_server.py
```

The supervisor restarts the server with exponential backoff (2 s →
60 s cap) on any non-zero exit. A run lasting ≥30 s resets the
backoff. Ctrl+C in the supervisor window stops both supervisor and
child cleanly.

For a one-shot run (no supervisor) — use the underlying launcher
directly:
```
.venv\Scripts\python.exe .claude\worktrees\fervent-meitner-98bfe7\scripts\start_llamacpp_server.py --n-ctx 16384
```

The default `--n-ctx 16384` gives OpenClaw enough budget for its
tool-bundling overhead. For voice-only use, `--n-ctx 8192` is fine
and saves ~136 MiB KV cache. Both leave plenty of VRAM headroom.

**OpenClaw Gateway:**
```
C:\Users\alecf\.openclaw\gateway.cmd
```
Stays in foreground. Gateway is unhealthy after retry-storm errors
(1006 abnormal closures); if `openclaw status --json` shows
``gatewayRunning: false`` or sessions look stale, Ctrl+C and re-run.

## Three OpenClaw agents

Configured in `~/.openclaw/openclaw.json` under `agents.list[]`. All
three use the same `litellm/qwen3.5-9b-local` model, the same
`messaging` tool profile, and the same explicit deny list. The
difference is their `systemPromptOverride`.

| Agent | Audience | systemPromptOverride source | Default? |
|-------|----------|------------------------------|----------|
| `ultron-test` | Internal verification | Tight worker prompt: "respond plainly, no NO_REPLY" | yes |
| `ultron-main` | User (voice + Telegram) | `PersonaLoader.get_system_prompt(mode='user_facing')` — full Ultron character | no |
| `ultron-heartbeat` | Periodic background tick | `PersonaLoader.get_system_prompt(mode='heartbeat')` — minimal checklist | no |

Default agent is `ultron-test` so `openclaw agent ...` without
`--agent` runs through the verification harness. Use `--agent
ultron-main` for actual user-facing flows once channels are wired
in later phases.

## Persona separation

Ultron's character lives in user-facing channels only. Internal jobs
(heartbeat, cron, tool selection, summarization, RAG gating, etc.)
use a plain task-focused prompt. This:

- Protects the voice character from being trained-out by internal
  request patterns.
- Saves context budget — background mode is ~50% smaller than
  user-facing.
- Improves reliability for tasks where Ultron's terseness and
  hedging-aversion would obscure the answer.

Implemented via `PersonaLoader` modes — see
[persona.py](../src/ultron/openclaw_bridge/persona.py).

```python
loader.get_system_prompt("user_facing")  # full Ultron, voice + Telegram
loader.get_system_prompt("background")   # plain worker, internal jobs
loader.get_system_prompt("heartbeat")    # minimal checklist
loader.get_system_prompt("bootstrap")    # one-time init
```

## Locked-in constraints

These constraints are kept in place for the local-Qwen agents until
later-phase prompt-budget work. Do not remove them without re-running
Phase 0 verification:

1. **`tools.profile: "messaging"`** on every local-Qwen agent. The
   `coding` profile bundles ~50+ tool schemas which exceed Qwen3.5-9B
   prefill in the OpenAI SDK's hardcoded 30 s request timeout.
   Messaging profile bundles ~4 tools — tight fit but reliable.

2. **Explicit `tools.deny`**: `group:web`, `group:fs`,
   `group:runtime`, `browser`, `memory_search`, `send`. Belt-and-
   braces over the messaging profile; prevents tool surface drift if
   OpenClaw adds new tools to the profile defaults in a future
   release.

3. **Model `contextWindow: 16384`** in the OpenClaw model config.
   OpenClaw's prompt-budget calculations clamp at half this value
   (~8 k tokens for the prompt-budget pre-check). Lower values trip
   the precheck on agents with bundled tool schemas.

4. **llama-cpp-server `--n-ctx 16384`** (or higher). Must match or
   exceed the OpenClaw config's `contextWindow`. The KV cache cost
   at 16 k Q8_0 is ~272 MiB — well within the VRAM budget.

5. **`reasoning: true`** on the model entry. Tells OpenClaw the
   model emits chain-of-thought it should treat as
   `reasoning_content`.

6. **`api: "openai-completions"`** on the litellm provider. Required
   for the runtime to recognise the provider as a custom local
   provider.

## Server-stability watchdog

The supervisor (`scripts/supervised_llamacpp_server.py`) is the
recommended way to run the server day-to-day. It catches crashes,
backs off, and restarts. For unattended deployment (e.g., if Ultron
runs as a service and you want llama-cpp-server up before the voice
loop starts), the supervisor can be wrapped with NSSM:

```
nssm install ultron-llamacpp \
    "C:\STC\ultronPrototype\.venv\Scripts\python.exe" \
    "C:\STC\ultronPrototype\.claude\worktrees\fervent-meitner-98bfe7\scripts\supervised_llamacpp_server.py"
nssm set ultron-llamacpp AppDirectory "C:\STC\ultronPrototype"
nssm set ultron-llamacpp AppStdout "C:\STC\ultronPrototype\logs\llamacpp.out.log"
nssm set ultron-llamacpp AppStderr "C:\STC\ultronPrototype\logs\llamacpp.err.log"
nssm set ultron-llamacpp AppRotateFiles 1
nssm start ultron-llamacpp
```

NSSM not installed by default; download from nssm.cc. The
supervisor itself is sufficient for desk use without NSSM.

## Voice-pipeline runtime modes (HTTP migration: opt-in only)

The voice pipeline uses the in-process llama-cpp-python loader by
default (`llm.runtime: "in_process"`). HTTP-client mode is fully
wired and unit-tested but is NOT the default — measurements showed
it adds ~71 ms median first-token latency, which exceeds the 10%
no-regression gate.

### Latency comparison (2026-05-08, RTX 4070 Ti, Qwen3.5-9B Q4_K_M, n_ctx=16384)

| Metric | in_process (default) | http_server (opt-in) | delta |
|--------|----------------------|----------------------|-------|
| TTFT median | 125 ms | 196 ms | +71 ms (+57%) |
| TTFT p95 | 140 ms | 235 ms | +95 ms (+68%) |
| TTFT min | 94 ms | 140 ms | +46 ms |
| TTFT max | 140 ms | 344 ms | +204 ms |

Sources: `baselines.json` keys `latency_ms.aggregate` (in-process,
written by `scripts/measure_baseline.py`) and `llm_http_runtime`
(HTTP, written by `scripts/_bench_llm_http.py`).

### Why HTTP is slower
The gap comes from per-call overheads that are zero for in-process
calls: TCP loopback syscalls, HTTP request parsing on the server
side, SSE framing on the response side, JSON encode/decode round
trip. The model-side prefill and decode cost is the same — both
runtimes use the same llama-cpp-python build with the same params.

### When to opt in
- **Don't** opt in for the voice pipeline. ~70 ms added to TTFT is
  audible: it pushes TTFA from ~655 ms to ~725 ms, a noticeable
  delay between end-of-speech and first audio.
- **Do** opt in only for non-voice consumers (Telegram text replies
  via OpenClaw, batch jobs, Phase 7+ cron tasks where latency
  doesn't matter). Those already go through OpenClaw → litellm →
  llama-cpp-server, so they're effectively in HTTP mode regardless
  of `llm.runtime`.

### Re-running benchmarks

```
# In-process (full voice-pipeline measurement, ~3-5 min):
cd C:\STC\ultronPrototype
.venv\Scripts\python.exe .claude\worktrees\fervent-meitner-98bfe7\scripts\measure_baseline.py

# HTTP-mode TTFT (server must be running on :8765, ~30 sec):
.venv\Scripts\python.exe .claude\worktrees\fervent-meitner-98bfe7\scripts\_bench_llm_http.py
```

Both write to `baselines.json` for diff-able tracking. The 10%
no-regression gate applies if HTTP mode is reconsidered later (e.g.,
on faster IO, after llama-cpp-server tuning, or with a smaller model
where prefill is shorter).

### Opting in (if you choose)

Set `llm.runtime: "http_server"` in `config.yaml` and restart the
voice loop. The voice path will then talk to llama-cpp-server over
HTTP and share VRAM with OpenClaw. To revert, set back to
`"in_process"` and restart.

When HTTP mode is opted in:
- VRAM idle drops by ~5.7 GB in the voice-pipeline process (no
  in-process model load).
- VRAM peak in the system as a whole stays the same — the server
  holds the only model copy.
- TTFT regresses by ~70 ms median per the table above.
