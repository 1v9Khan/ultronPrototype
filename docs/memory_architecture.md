# Memory architecture

Phase 10 of the OpenClaw integration. Ultron uses three memory layers,
each with a specific role. They complement each other — none replaces
another.

## The three layers

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Qdrant (vector RAG)                                    │
│ - Conversations, facts, web_results collections.                │
│ - Hybrid search (BGE dense + BM25 sparse, RRF fusion).          │
│ - Latency-critical: every voice query hits this layer.          │
│ - Authoritative for "what happened recently" queries.           │
│ - Maintenance via scripts/maintenance.py (extract_facts,        │
│   cluster_conversations, decay_stale_facts, cleanup_web_cache). │
└─────────────────────────────────────────────────────────────────┘
                                │
                  off-hot-path  │  read-only on hot path
                  writes        │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Workspace files (~/.openclaw/workspace/)               │
│ - SOUL.md, IDENTITY.md, USER.md, AGENTS.md, HEARTBEAT.md,       │
│   BOOTSTRAP.md  (persona + policy; auto-loaded into prompts)    │
│ - MEMORY.md  (curated long-term notes; rarely changes)          │
│ - memory/YYYY-MM-DD.md  (daily journal-style notes)             │
│ - Plain Markdown, human-readable, git-friendly.                 │
│ - Read by both Ultron (PersonaLoader) and OpenClaw agents.      │
│ - Written by maintenance + dreaming sweep + standing orders.    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │  promotion of durable knowledge
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: Memory Wiki (OpenClaw plugin)                          │
│ - Compiled, structured knowledge with provenance.               │
│ - Claims with freshness tracking, dashboards, lint.             │
│ - Used for "what do I know about X" queries that benefit from   │
│   structure (e.g. project architecture, recurring decisions).   │
│ - Populated organically, not via bulk import.                   │
│ - Plugin: @openclaw/memory-wiki (bundled, disabled by default). │
└─────────────────────────────────────────────────────────────────┘
```

## Read paths (who reads what when)

| When | Reads | Why |
|---|---|---|
| Every voice query | Qdrant (conversations + facts) | Latency-critical context retrieval. Hybrid search runs in parallel with LLM warmup. |
| Every voice / Telegram turn | Workspace files (via PersonaLoader) | System prompt composition. Hot-reloaded on file change. |
| Web-search-triggering queries | web_results cache (Qdrant) | Cached snippets + Jina full-text bypass redundant calls. |
| Heartbeat tick | Workspace HEARTBEAT.md + heartbeat alert log | Per-tick checklist + alert deduplication. |
| Standing-order programs | Whichever layer the program declares | Per-program; documented in AGENTS.md. |
| Wiki query ("what do I know about X") | Memory Wiki (when enabled) | Structured, provenance-tracked knowledge. |

## Write paths (who writes what when)

| When | Writes | Sync? |
|---|---|---|
| End of every conversation turn | Qdrant conversations collection | Async (off hot path). |
| Maintenance run | Qdrant facts (extract_facts), cluster labels, web cache cleanup | Sync (~minutes, run nightly). |
| Maintenance run + dreaming | Workspace MEMORY.md, USER.md | Sync, with `WorkspaceWriter` advisory locks. |
| Heartbeat alert | logs/heartbeat_alerts.jsonl + (optional) MEMORY.md | Sync (under lock). |
| User Markdown edit | Any workspace file | Detected by PersonaLoader on next read. |
| Memory Wiki promotion | Memory Wiki vault | Async, agent-driven (when `wiki_apply` runs). |

## Why three layers, not one

Each layer optimises for different access patterns:

- **Qdrant is fast but opaque.** Embeddings + BM25 are great for
  retrieval but bad for human review. You can't open Qdrant in a
  text editor and see what you remember about a topic.
- **Workspace files are readable but unstructured.** SOUL.md is
  a paragraph; MEMORY.md is a list. Great for persona + curated
  notes, terrible for "give me everything related to project X."
- **Memory Wiki is structured but slow.** Compiled wiki pages with
  claims + provenance let you ask "what's the latest on
  project X" with audit-trail-quality answers. Not appropriate
  for every-utterance retrieval.

The voice path uses Qdrant exclusively because it's the only layer
fast enough. The other two layers serve reflection and human
review.

## Critical: do NOT bulk-migrate between layers

When the Memory Wiki plugin lands, do **not** bulk-import Qdrant
content into the wiki. They serve different purposes. Specifically:

- Qdrant content is conversation-by-conversation; importing creates
  a wiki that's noisy, redundant, and hard to read.
- Memory Wiki claims have provenance (where did the claim come
  from?). Bulk import erases that provenance.
- Wiki retrieval is slower than Qdrant. Voice queries that hit
  the wiki regress hot-path latency.

Memory Wiki should be populated organically:

1. **Standing orders** identify durable knowledge worth wiki-ing
   (e.g., "Weekly Review notices a recurring topic; agent decides
   it deserves a wiki page").
2. **Dreaming sweep** (deferred — Phase 10+) promotes some
   short-term content from `memory/YYYY-MM-DD.md` to wiki claims.
3. **User on-demand**: "Make a wiki page about X based on what
   you know" — the agent runs `wiki_apply` with a specific scope.

## Maintenance contract

When you add a new memory-touching feature:

1. **Identify the right layer.** Hot-path read? Qdrant only.
   Persona + policy? Workspace file. Audit-quality knowledge?
   Wiki.
2. **Document the access pattern** in this file's tables above.
3. **Test the budget.** Voice path latency must not regress.
   For new Qdrant queries, measure TTFT before merging.
4. **Don't duplicate.** If a fact lives in Qdrant's facts
   collection, don't also write it to MEMORY.md unless there's
   a clear reason for the human-readable copy.

## Memory Wiki status (Phase 10)

The plugin is bundled with OpenClaw 2026.5.7 (`@openclaw/memory-wiki`)
but disabled by default. To enable:

```powershell
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" plugins enable memory-wiki
```

Verify and configure per
[`docs/openclaw_memory_wiki_setup.md`](openclaw_memory_wiki_setup.md).

Until enabled, the agent's wiki tools (`wiki_search`, `wiki_get`,
`wiki_apply`, `wiki_lint`) won't be available — agent calls to
those tools return "tool unavailable" which the dispatcher
translates to a clear voice message.

## Locked-in constraints

- **Voice-path memory stays Qdrant.** Don't migrate the
  conversation history to file-based or wiki-based storage.
  Hot-path latency requires the vector store.
- **Persona files stay Markdown.** Don't move SOUL.md /
  IDENTITY.md / etc. into a database. They're git-friendly and
  human-readable for a reason.
- **Memory Wiki stays opt-in.** The plugin enables a useful
  feature, not a foundational layer. Ultron should function
  identically whether the wiki is enabled or not.
