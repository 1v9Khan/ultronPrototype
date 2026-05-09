# Standing orders

Phase 8 of the OpenClaw integration. Standing orders are autonomous
programs the OpenClaw agent has permanent authority to execute
within explicit boundaries. They live in
`~/.openclaw/workspace/AGENTS.md` and are triggered by heartbeat
ticks, cron jobs, or on-demand voice / Telegram queries.

## Program registry

| Program | Trigger | Output | Source |
|---|---|---|---|
| Coding Project Watcher | Heartbeat tick (1h) + on-demand | Telegram alerts, optional voice | AGENTS.md `## Program: Coding Project Watcher` |
| Weekly Review | Cron (Fri 17:00 local) + on-demand | Telegram digest, ≤300 words | AGENTS.md `## Program: Weekly Review` |

Both programs are defined in [`~/.openclaw/workspace/AGENTS.md`](file://%USERPROFILE%/.openclaw/workspace/AGENTS.md)
under the **Standing orders** section. Edits to that file land on
the next session start or the next heartbeat tick that re-reads
the workspace.

## Required structure for a program

Every program section in AGENTS.md has these five required
headings:

1. **Authority** — what the program is permitted to do, in plain
   language. State the scope so the agent knows when it's
   over-reaching.
2. **Trigger** — when the program runs (cron expression, heartbeat
   tick, on-demand query pattern, etc.).
3. **Approval gate** — what (if any) confirmation is required
   before the program produces user-visible output. Most programs
   have no gate for compilation/reporting; gates apply when the
   program would take a destructive action.
4. **Escalation** — what to do when data sources are missing,
   thresholds are crossed, or the program would otherwise produce
   a degraded/partial result.
5. **Execution** — numbered steps the agent walks through.

Plus an explicit **What NOT to do** list — short, specific
prohibitions that prevent scope creep. ("Do not modify project
files autonomously", "Do not include credentials in summaries",
etc.)

## Maintenance contract

When you add or change a standing-order program:

1. **Update AGENTS.md** with the full program block (all five
   headings + What-NOT-to-do).
2. **Add a cron entry** if the program runs on a schedule, per
   [docs/openclaw_cron_setup.md](openclaw_cron_setup.md). Use
   names matching the program (e.g.  `weekly-review`).
3. **Wire output** through `NotificationDispatcher.notify_standing_order_output`
   so the program respects the user's
   `notifications.telegram.notify_on.standing_order_outputs`
   gating.
4. **Test the trigger** before relying on the schedule:
   - For cron: `openclaw cron run <name>`.
   - For heartbeat-triggered: `openclaw system event --text "..." --mode now`.
   - For on-demand voice: ask Ultron the relevant query through
     the live voice path or Telegram.
5. **Update this file's program registry** with the new entry.

## Disabling a program

Comment out the entire `## Program:` section in AGENTS.md (HTML
comments survive Markdown rendering and keep the source available
for audit). Don't delete — the comment-out makes the disable
visible in `git log`.

The heartbeat agent re-reads the workspace on each tick, so the
disable lands within one tick (default 1 h). For an immediate
disable, also stop the matching cron entry:

```powershell
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" cron remove weekly-review
```

## On-demand "what's running" query

The user can ask "what is Ultron working on?" or "what standing
orders are active?" via voice or Telegram. The classifier should
route this to a query handler that:

1. Reads the active program list from AGENTS.md (parses the
   `## Program:` headings).
2. Queries recent heartbeat alerts (the local alert log from
   Phase 5) for any open items.
3. Returns a brief in-character summary: "I'm watching three
   coding projects. The Flask app's stuck on a clarification
   from Tuesday. No alerts pending."

This handler is **not yet implemented** in Phase 8 — landing it
requires the OpenClaw stdio MCP entrypoint (so the agent can
read the alert log) or an in-process voice intent (which would
duplicate work Phase 8 deferred). Add it when the MCP entrypoint
lands or when there's a concrete need.

## Why programs live in AGENTS.md, not Python

- **Readable by both Ultron's voice path and OpenClaw's agent
  side.** The same file is loaded by `PersonaLoader` (in
  `background` mode) and by the OpenClaw agent runtime.
- **Editable by the user.** No re-deploy needed to change a
  program; edit AGENTS.md and the next tick picks it up.
- **Diff-friendly.** Programs evolve over time; Markdown +
  version control gives an audit trail.
- **Bounded scope.** A program is policy, not code. Putting it
  in code creates pressure to add features ("just one more loop,
  one more conditional"); keeping it in prose forces explicit,
  reviewable changes.

## Security posture

Standing orders run autonomously — they don't ask before producing
output. The What-NOT-to-do lists are the only explicit safety net.
Treat changes to any program's What-NOT-to-do list with
particularly careful review:

- Don't loosen "Do not modify project files autonomously" without
  a hard requirement.
- Don't widen scope into shell execution, network actions outside
  reads, or anything that could send messages to people other
  than the user.
- The block-and-revise validator (4B plan Item 8) intercepts
  OpenClaw tool calls — but it's a soft fence. The What-NOT-to-do
  list is the primary boundary.

If a program would benefit from being able to take an action it's
explicitly forbidden from taking, that's a sign to escalate and
add a one-shot user authorization rather than relaxing the
policy.
