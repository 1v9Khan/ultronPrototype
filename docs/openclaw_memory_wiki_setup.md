# Memory Wiki plugin setup

Phase 10 of the OpenClaw integration. The Memory Wiki plugin
(`@openclaw/memory-wiki`) ships bundled with OpenClaw 2026.5.7 in
disabled state. It complements Ultron's existing memory layers — it
is **not** a replacement for Qdrant or the workspace files. See
[docs/memory_architecture.md](memory_architecture.md) for the
three-layer model.

## What this enables

- Agent can author and update structured wiki pages with
  Obsidian-friendly Markdown.
- Pages carry claims with provenance (where did the claim come
  from), freshness tracking, and dashboards.
- The agent gains four tools: `wiki_search`, `wiki_get`,
  `wiki_apply`, `wiki_lint`.

## What this does NOT do

- Does **not** auto-import Qdrant conversations or facts. The
  wiki populates organically — see "Don't bulk-migrate" in the
  memory architecture doc.
- Does **not** replace the workspace's `MEMORY.md` (curated
  notes) or `memory/YYYY-MM-DD.md` (daily journal). Those stay
  as the human-readable persistence layer.
- Does **not** sit on the voice hot path. Wiki queries are
  on-demand only.

## Enabling the plugin

```powershell
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" plugins enable memory-wiki
```

Restart the Gateway so the plugin loads:

```powershell
# In the Gateway window: Ctrl+C, then re-run gateway.cmd
```

Verify:

```powershell
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" plugins list `
    | Select-String -Pattern "memory-wiki"
```

Expected: status `enabled`.

## Configuration

Add to `~/.openclaw/openclaw.json` under `plugins.entries`:

```json5
{
  plugins: {
    entries: {
      // ... existing entries (litellm, etc.) stay ...
      "memory-wiki": {
        enabled: true,
        config: {
          // Wiki vault path, relative to the workspace.
          vaultDir: "memory-wiki",

          // Generate dashboards (trends + recent additions). Lightweight;
          // dashboards re-render on demand.
          dashboards: { enabled: true },

          // Track claim freshness so stale claims surface for review.
          freshnessTracking: { enabled: true }
        }
      }
    }
  }
}
```

Refer to OpenClaw's docs for the current full config schema:
https://docs.openclaw.ai/cli/plugins (look for memory-wiki under
inspect output).

## Smoke test

After Gateway restart, ask the agent to author a small wiki page:

Via Telegram (or `openclaw agent --agent ultron-main`):

> "Make a wiki page summarizing what you know about my Ultron
> project structure based on our recent conversations."

Expected: agent calls `wiki_search` to find prior content,
`wiki_apply` to create or update a page, then confirms the result.

Verify the file exists:

```powershell
Get-ChildItem "$env:USERPROFILE\.openclaw\workspace\memory-wiki\*.md" `
    | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

## When to use wiki vs. other layers

| Question | Best layer | Why |
|---|---|---|
| "What did we talk about yesterday?" | Qdrant (conversations) | Latency-critical recency query. |
| "Who is the user?" | Workspace USER.md | Loaded automatically into prompt. |
| "What's our project structure?" | Memory Wiki | Audit-trail / provenance value. |
| "What's our position on framework X?" | Memory Wiki | Recurring decision worth one canonical page. |
| "Are there any heartbeat alerts?" | logs/heartbeat_alerts.jsonl | Real-time alert state. |

## Tools the agent gains

After the plugin enables and the Gateway restarts:

- **`wiki_search`** — semantic search over the wiki vault. Returns
  page titles + excerpts.
- **`wiki_get`** — fetch a specific page by title or path.
- **`wiki_apply`** — create or update a page with structured
  claims + provenance. Idempotent (re-running with same content
  is a no-op).
- **`wiki_lint`** — sanity-check a page for orphaned references,
  stale claims, etc.

`openclaw doctor` lists the tools after the plugin loads.

## Disabling the plugin

```powershell
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" plugins disable memory-wiki
```

Existing pages stay on disk; the agent just loses access to the
tools. Re-enabling is a single command + Gateway restart.

## Troubleshooting

- **Plugin enables but tools don't appear** — restart the Gateway.
  Plugins are loaded at startup.

- **`wiki_apply` says "permission denied"** — Phase 0 locked
  `tools.profile: messaging` + explicit `tools.deny` on
  `ultron-main`. Wiki tools may need `tools.alsoAllow: ["wiki_*"]`
  on the agent config to bypass the messaging-profile filter.

- **Wiki pages don't show up in `wiki_search`** — search uses an
  index that builds on first query and updates on `wiki_apply`.
  If a manually-edited page is missing, run
  `openclaw memory --reindex` (or whatever the current rebuild
  command is — check `openclaw plugins inspect memory-wiki`).

- **Wiki vault grows unexpectedly** — `wiki_apply` is idempotent
  but doesn't auto-prune. Stale pages accumulate. Periodically
  ask the agent to "review the wiki and propose pages to retire";
  approve manually.

## Security posture

Wiki pages are plaintext Markdown on local disk — same trust level
as the workspace files. The agent's authority to write the wiki
comes from `tools.alsoAllow` on its profile; keep that set
deliberately.

If the agent is asked to wiki something sensitive (credentials,
unreleased code, third-party content), it should refuse — the
"What you do not do" list in AGENTS.md covers that case. Verify
the policy still reads correctly after enabling the wiki tools.
