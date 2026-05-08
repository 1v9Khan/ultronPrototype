# OpenClaw integration — Phase 1 close-out

Phase 1 of the OpenClaw integration: migrate Ultron's persona out of
the hardcoded `config.yaml:llm.system_prompt` string and into the
shared workspace files OpenClaw uses. After Phase 1, both Ultron's
voice pipeline and OpenClaw read from the same workspace —
a `SOUL.md` edit propagates to both consumers' next turn without
restart.

## Verification criteria — final status

Per Section 3.6 of the OpenClaw integration prompt:

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Six persona files created in workspace with content migrated from existing system prompt | ✅ | `~/.openclaw/workspace/{IDENTITY,SOUL,USER,AGENTS,HEARTBEAT,BOOTSTRAP}.md`. Stock OpenClaw boilerplate backed up to `.stock-bak/`. |
| `PersonaLoader` implemented with all documented methods | ✅ | [src/ultron/openclaw_bridge/persona.py](../src/ultron/openclaw_bridge/persona.py) — `load`, `refresh_if_stale`, `get_system_prompt(mode=...)`, `current` property. 19 unit tests in [tests/test_persona_loader.py](../tests/test_persona_loader.py). |
| System prompt builder updated to use loaded persona | ✅ | [src/ultron/llm/inference.py](../src/ultron/llm/inference.py): `LLMEngine._resolve_system_prompt()` reads from `PersonaLoader` on every `_build_messages` call when `llm.persona.source == "workspace"` (the default). 8 wiring tests in [tests/test_llm_persona_source.py](../tests/test_llm_persona_source.py). |
| Voice character unchanged on representative queries | ⏸ | Interactive — needs the user to speak through 5 representative voice queries and confirm Ultron sounds unchanged. The composed user-facing prompt is **1135 chars** vs the original config's **1136 chars** — bit-equivalent in content. |
| Hot reload works — modify SOUL.md, next response reflects the change without restart | ✅ | `test_workspace_source_hot_reloads_on_soul_edit` in [tests/test_llm_persona_source.py](../tests/test_llm_persona_source.py) edits `SOUL.md` mid-test and confirms the next `_build_messages` call sees the new content. |
| All existing tests pass | ✅ | 736 / 736, 15 skipped, 0 failed. |
| New persona tests pass | ✅ | 27 (19 PersonaLoader + 8 LLM wiring). |
| VRAM unchanged from Phase 0 baseline | ✅ | `full_loaded`: 10138 MB (vs 10124 MB Phase 0) = +14 MB, negligible. Voice-path peak: 10370 MB (vs 10363 MB) = within noise. |
| Latency unchanged from Phase 0 baseline | ✅ | TTFT median **109 ms** (vs 125 ms Phase 0) — slightly *better*. TTFA median 609 ms (vs 655 ms). Both well within the 10% no-regression gate. |

## Persona-split architecture

User-facing channels (voice, Telegram → user) get the full Ultron
character. Internal worker calls (heartbeat, cron, summarization,
RAG gating, tool selection) use a plain task-focused prompt. This:

- Protects voice character from being trained out by internal request
  patterns.
- Keeps the user-facing prompt at ~1135 chars (matching the original
  config), so prefill cost on the voice path is preserved.
- Improves reliability for tasks where Ultron's terseness and hedging-
  aversion would obscure the answer.

Implemented via four `PersonaLoader` modes:

```python
loader.get_system_prompt("user_facing")  # voice path; IDENTITY + SOUL + USER
loader.get_system_prompt("background")   # internal workers; AGENTS only
loader.get_system_prompt("heartbeat")    # periodic ticks; HEARTBEAT only
loader.get_system_prompt("bootstrap")    # one-time init; BOOTSTRAP only
```

`AGENTS.md` (operating rules: tool selection, memory ops, escalation
policy) is deliberately excluded from `user_facing`. Including it
inflated the voice-path prompt to ~4700 chars, which regressed TTFT
by **+218 ms** (a +175% blow-up). Voice-relevant rules (do-not-
lecture, uncertainty handling) live in `SOUL.md` so they stay on the
voice path; the operating-rules content stays for OpenClaw worker
agents.

## What landed in code

| File | Change |
|------|--------|
| [src/ultron/openclaw_bridge/persona.py](../src/ultron/openclaw_bridge/persona.py) | `PersonaLoader`, `PersonaBundle`, `PersonaFile`, `PromptMode`, `default_workspace_dir`. Mode-based composition + HTML-comment-only file detection + `refresh_if_stale` for hot reload. |
| [src/ultron/openclaw_bridge/__init__.py](../src/ultron/openclaw_bridge/__init__.py) | Public exports. |
| [src/ultron/llm/inference.py](../src/ultron/llm/inference.py) | `LLMEngine._resolve_system_prompt()` + `_maybe_build_persona_loader`. `_build_messages` resolves prompt fresh each turn (sub-millisecond stat() calls; same prefill cost). |
| [src/ultron/config.py](../src/ultron/config.py) | `LLMPersonaConfig` block: `source` ("workspace" \| "config"), `workspace_dir`, `fallback_to_config_on_empty`, `hot_reload`. Default `source` = "workspace". |
| [tests/test_persona_loader.py](../tests/test_persona_loader.py) | 20 tests for `PersonaLoader`. |
| [tests/test_llm_persona_source.py](../tests/test_llm_persona_source.py) | 8 tests for the `LLMEngine` wire-up + hot reload + fallback. |

Persona files (live in `~/.openclaw/workspace/`, not in this repo):

- `IDENTITY.md` (148 bytes) — "You are Ultron. Not a simulation..." paragraph.
- `SOUL.md` (991 bytes) — voice/tone + brevity + uncertainty handling.
- `USER.md` (82 bytes) — placeholder; auto-populated from Qdrant `facts`.
- `AGENTS.md` (2156 bytes) — internal-worker operating rules (background mode only).
- `HEARTBEAT.md` (315 bytes) — heartbeat checklist (Phase 5 will populate).
- `BOOTSTRAP.md` (265 bytes) — empty stub (kept so PersonaLoader doesn't warn on every load).
- Stock OpenClaw boilerplate at `.stock-bak/` for reference.

## What's deferred

- **Voice character verification** — interactive; user runs 5 representative voice queries and confirms Ultron sounds unchanged.
- **Persona file migration into a tracked workspace location** — the workspace lives at `~/.openclaw/workspace/` outside the repo. A future phase may add a `workspace/` mirror in the repo that the PersonaLoader can fall back to, so the persona ships with code.
- **Phase 3 bridge layer** — `OpenClawClient`, `UltronMcpRegistrar`, `WorkspaceWriter`, event receiver. `OpenClawLifecycle` and the OpenClaw typed errors landed in this commit as foundation; the rest is the next session's work.
