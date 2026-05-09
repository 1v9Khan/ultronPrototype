# OpenClaw hooks setup

Phase 9 of the OpenClaw integration. Hooks are small lifecycle
handlers that fire on agent events (session start, command issued,
compaction begin/end, etc.). OpenClaw 2026.5.7 ships five bundled
hooks; Phase 9 enables the two relevant to Ultron's workflow and
documents the pattern for adding custom hooks later.

## Bundled hooks (as of OpenClaw 2026.5.7)

`openclaw hooks list` reports five ready-to-enable hooks:

| Hook | What it does | Use? |
|---|---|---|
| `session-memory` | Saves session context to `<workspace>/memory/<date>-<slug>.md` when the user issues `/new` or `/reset`. | **Yes** — pairs with our daily memory file pattern. |
| `command-logger` | Logs every command event to `~/.openclaw/logs/commands.log` (JSONL). | **Yes** — useful audit trail. |
| `boot-md` | Runs `BOOT.md` on gateway startup if it exists in the workspace. | Skip — `BOOTSTRAP.md` (Phase 1) is the existing workspace bootstrap; adding a parallel `BOOT.md` is duplicative. |
| `bootstrap-extra-files` | Injects extra workspace files via glob/path patterns at session start. | Skip — the persona files we have (SOUL/IDENTITY/USER/AGENTS/HEARTBEAT/BOOTSTRAP) are already injected via `agents[].systemPromptOverride`. |
| `compaction-notifier` | Sends a visible chat notice when session compaction starts/ends. | Skip — Ultron's voice path does its own compaction (the projection layer); the OpenClaw side rarely needs compaction since `isolatedSession: true` keeps each tick fresh. |

## Enabling the recommended pair

```powershell
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" hooks enable session-memory
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" hooks enable command-logger
```

Verify:

```powershell
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" hooks list `
    | Select-String -Pattern "enabled"
```

## Smoke test

After enabling, send `/new` to the bot via Telegram (or run an
agent turn that includes `/new`). Verify:

1. `<workspace>/memory/<date>-<slug>.md` exists with the session
   context.
2. `~/.openclaw/logs/commands.log` has a fresh entry for the
   `/new` command.

```powershell
Get-ChildItem "$env:USERPROFILE\.openclaw\workspace\memory\*.md" `
    | Sort-Object LastWriteTime -Descending | Select-Object -First 5

Get-Content "$env:USERPROFILE\.openclaw\logs\commands.log" `
    | Select-Object -Last 10
```

## Custom hook scaffolding (deferred)

OpenClaw supports custom hooks installed via `openclaw plugins
install`, defined as a directory under `~/.openclaw/hooks/<name>/`
with:

- `HOOK.md` — metadata (name, description, events the hook fires
  on, required permissions).
- `handler.ts` (or `handler.js`) — the handler implementation.

Useful Ultron-specific hooks for future implementation:

- **`coding-completion-bridge`** — listen for the OpenClaw agent's
  `tool:complete` event and forward the result to Ultron's
  `NotificationDispatcher.notify_coding_task_completion` if it
  came from a coding-related tool. Currently this routing happens
  Ultron-side via the orchestrator's `_announce_coding_completion_if_pending`
  hook, but a Gateway-side hook would catch completions that
  arrive via Telegram too.

- **`voice-pipeline-reset`** — clear Ultron's audio buffers when
  the user issues `/reset`. Useful for catching state drift.

- **`qdrant-snapshot`** — snapshot the Qdrant data on the
  `daily-end` event so backups happen automatically.

These are documented but not implemented — implement when a
specific need emerges. The pattern is straightforward: HOOK.md
declares the events, handler.ts subscribes to them, OpenClaw
manages the lifecycle.

## Hooks vs the Phase 4 / 5 dispatcher

Why not put coding-completion notification in a hook instead of
in `NotificationDispatcher`?

- **Hooks fire Gateway-side**, with access to the agent run's
  shape but not Ultron's in-process state (Qdrant sessions,
  active coding tasks, etc.).
- **NotificationDispatcher fires Ultron-side**, after the
  orchestrator already knows the task completed via the in-process
  `CodingTaskRunner`.

For events that originate Ultron-side (a coding session the user
started by voice), `NotificationDispatcher` is the right home.
For events that originate OpenClaw-side (the agent decides on its
own to send something), a hook is the right home. Phase 9
documents the seam; later phases add specific hooks as use cases
emerge.

## Configuration knobs

Hook enablement lives in `~/.openclaw/openclaw.json` under
`hooks.entries[name].enabled`. After running
`openclaw hooks enable <name>`, the JSON has:

```json5
{
  hooks: {
    entries: {
      "session-memory": { enabled: true },
      "command-logger": { enabled: true }
    }
  }
}
```

Both hooks read their own config from sub-objects under
`entries[name]`; the defaults are sensible. To override (e.g. send
command-logger output to a different path), see
[OpenClaw's hooks docs](https://docs.openclaw.ai/cli/hooks).

## Troubleshooting

- **Hook enabled but doesn't fire** — restart the Gateway
  (`Ctrl+C` in the Gateway window, then re-run `gateway.cmd`).
  Hooks are loaded at startup.

- **`session-memory` produces empty files** — check
  `agents.list[].id` of the agent that received `/new`. If the
  agent uses `isolatedSession: true` and `lightContext: true`
  (the heartbeat config), the saved session context is minimal
  by design.

- **`command-logger` log gets large** — rotate manually; OpenClaw
  doesn't ship a built-in rotation policy as of 2026.5.7. A
  weekly Windows Task Scheduler job that gzips the previous week's
  log works fine.

- **Custom hook crashes Gateway on load** — check
  `~/.openclaw/logs/gateway.log` for stack traces. OpenClaw
  isolates hook failures so a bad hook doesn't take down the
  Gateway, but the failed hook is disabled until the next
  successful load.
