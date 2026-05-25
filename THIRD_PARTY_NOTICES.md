# Third-party notices

Ultron incorporates design patterns and a small set of vendored configuration
files derived from third-party open-source projects. Each is listed below with
its license and the scope of what was incorporated.

## aider (Apache License 2.0)

Repository: https://github.com/Aider-AI/aider
License: Apache License, Version 2.0 (a copy is included verbatim below).

The following ultron components are clean-room re-implementations whose
*approach* was informed by the corresponding aider modules. No source code
is copied verbatim except for the vendored tree-sitter query files listed
in the next paragraph.

| Ultron component | Inspired by | Notes |
| --- | --- | --- |
| `src/ultron/coding/important_files.py` | `aider/special.py` | Allowlist of well-known root files. List extended with ultron-specific entries. |
| `src/ultron/utils/mtime_cache.py` | `aider/repomap.py` cache layer | mtime-keyed SQLite cache with dict fallback. |
| `src/ultron/utils/token_budget.py` | `aider/repomap.py` binary search | Token-budget binary search with tolerance. |
| `src/ultron/utils/snapshot_guard.py` | `aider/coders/base_coder.py` summarize race protection | Snapshot-identity guard for background work. |
| `src/ultron/utils/relative_indent.py` | `aider/coders/search_replace.py` `RelativeIndenter` | Indent-relative text transform. |
| `src/ultron/coding/tree_sitter_tags.py` | `aider/repomap.py` `get_tags_raw` | Tree-sitter symbol extraction with pygments ref fallback. |
| `src/ultron/coding/repo_map.py` | `aider/repomap.py` | PageRank-weighted repo map (batch 2). |
| `src/ultron/memory/background_summarizer.py` (tail-preserve revisions) | `aider/history.py` | Tail-preserve binary split + race-protected summarize (batch 3). |
| `src/ultron/coding/python_lint.py` | `aider/linter.py` | Fatal-only Python lint cascade (batch 4). |

### Vendored tree-sitter query files

The `src/ultron/coding/queries/` directory contains tree-sitter `*-tags.scm`
files adapted from
`aider/queries/tree-sitter-language-pack/*-tags.scm`. Each file carries a
short attribution header. These query files are configuration data describing
how to extract symbol definitions and references from a parsed AST, not
executable source code.

## OpenHands (MIT License)

Repository: https://github.com/All-Hands-AI/OpenHands
License: MIT (the portions of OpenHands outside of `enterprise/` are MIT-licensed;
a copy of the MIT text is included in the SWE-Agent section below).

The following ultron components are clean-room re-implementations whose
*approach* is informed by the corresponding OpenHands V1 app-server modules.
Algorithm shapes and contract names are adapted; no source code is copied
verbatim. Ultron's versions are restructured to fit the voice-first,
single-host, native-Windows runtime (OpenHands is multi-user web-server +
Docker sandbox; the patterns being borrowed are the orchestration layer,
not the execution layer).

| Ultron component | Inspired by | Notes |
| --- | --- | --- |
| `src/ultron/parsing/frontmatter.py` | `openhands/app_server/user/skills_router.py:_parse_skill_frontmatter` + `_load_skills_from_dir` | Fail-open YAML frontmatter parser (T11). Ultron's version returns both the parsed mapping AND the post-frontmatter body in a frozen :class:`FrontmatterResult`, handles the empty-frontmatter edge case (`---\n---\nbody`), tolerates CRLF line endings, and ships a directory walker with skip-dir + skip-filename filters. |
| `src/ultron/utils/poll.py` | `openhands/app_server/event_callback/set_title_callback_processor.py:_poll_for_title` | Bounded-retry polling primitive (T14). Default `max_attempts=4`, `delay_seconds=3.0` mirror the OpenHands constants. Ultron's version generalises to arbitrary callables (sync + async), supports a custom `is_done` predicate, optional exponential backoff with ceiling, and an async-only `cancel_check` for "voice resumed -- abandon" semantics. |
| `src/ultron/install/idempotent.py` | `openhands/app_server/app_conversation/app_conversation_service_base.py:maybe_setup_git_hooks` | Marker-comment idempotent installer (T8). Ultron's marker is `# INSTALLED-BY-ULTRON-3f9a7d2` (UUID-suffixed per the catalog's marker-collision mitigation). Atomic write via tmp + `os.replace`; best-effort audit log at `logs/install_log.jsonl`; explicit refuse-vs-preserve-vs-replace policy for unmarked existing files; `dry_run` mode for safe pre-flight. |
| `src/ultron/skills/` | `openhands/app_server/user/skills_router.py` + `openhands/app_server/app_conversation/skill_loader.py` + `openhands/sdk/skills` (KeywordTrigger / TaskTrigger semantics) | Trigger-loaded skills (T1). The frontmatter-then-markdown convention + the keyword vs. slash-command trigger split + the public / user / project source merging are adapted from the OpenHands shape. Ultron's version is single-process (no HTTP indirection), adds an mtime-invalidated catalog cache, an explicit `min_user_text_chars` guard to suppress one-word false-fires, and a `max_matches_per_turn` cap to bound the per-turn token budget. Initial skill catalogue under `skills/` is original ultron-specific content (gaming / coding / security / system_status / memory_notes / image_gen). |
| `src/ultron/events/` | `openhands/app_server/event/event_service.py` (`EventService` ABC) + `filesystem_event_service.py` + `event_callback/set_title_callback_processor.py` (poll loop) | Canonical event store (T2). The five-method ABC (`save_event` / `get_event` / `search_events` / `count_events` / `batch_get_events`) and the per-session prefix-scoped storage shape are adapted from the OpenHands `EventService`. Ultron's version is synchronous (single-process voice-first), per-session-scoped, and ships three backends in this batch -- `MemoryEventStore`, `JsonlEventStore`, `QdrantEventStore` (with JSONL fallback for graceful Qdrant degradation). Adds zip-format session export with chain verification embedded in `meta.json`. |
| `src/ultron/events/chain.py` | Independent design extending the existing `src/ultron/safety/audit.py` SHA-256 chain | Hash-chained tamper evidence (T13). The OpenHands catalog flags T13 as a generalisation of the safety-audit pattern; this module ports the algorithm verbatim from ultron's own existing implementation and exposes it as a reusable `compute_event_chain_hash` + `verify_chain` pair so non-safety event sequences can carry the same integrity property. |
| `src/ultron/events/bus_sink.py` | OpenHands callback dispatcher pattern (event production decoupled from side effect) | Bus -> store sink (T2 + foundation for T3 callbacks). Adapts the "every event becomes a typed row" mental model into ultron's existing pub/sub bus: subscribe to all events, convert each envelope to a `StoredEvent`, persist with the chain stamp. Ultron's version adds per-session sequence counters for stable ordering when events share a timestamp. |
| `src/ultron/events/callbacks.py` | `openhands/app_server/event_callback/event_callback_service.py` + `event_callback_models.py` (the `EventCallback` + `EventCallbackProcessor` ABC + `execute_callbacks` dispatch loop) | Event callback registry (T3). The polymorphic `CallbackProcessor.__call__(event, callback)` shape + the per-callback session/kind filter + the self-deactivation pattern (`SetTitleCallbackProcessor` flips itself DISABLED after fulfilment) are adapted from the OpenHands shape. Ultron's version is synchronous + in-memory by default (with optional JSONL persistence) since the voice path is single-process. Adds a slow-callback watchdog mirroring the bus subscriber pattern. |
| `src/ultron/events/processors.py` | `openhands/app_server/event_callback/set_title_callback_processor.py` + the catalog's documented creative extensions | Six built-in processors (T3 + creative extensions): Logging / Counting / ThresholdSnapshot (the load-bearing one-shot pattern from `SetTitleCallbackProcessor`) / MemoryWrite (catalog's "memory writes as callbacks" extension) / ChannelGuard (catalog's "safety rule pack as callback" inverted into payload redaction) / SkillActivator (catalog's "conditional skill activation as callback" extension). |
| `src/ultron/llm/condensers/` | OpenHands SDK `Condenser` ABC (`NoOp` / `ObservationMasking` / `Recent` / `LLMSummarizing` / `Amortized` / `LLMAttention`) referenced by `openhands/app_server/app_conversation/app_conversation_service_base.py:_create_condenser` | Swappable history compression (T4). The ABC + condense-result shape + five concretes (NoOp / Recent / Amortized / ObservationMasking / LLMSummarizing) are adapted. Ultron's version uses a plain `(role, content)` tuple turn type matching the existing `LLMEngine.Turn` shape; the `summarize_fn` injection on the LLM-summarising variant keeps the package free of any concrete LLM dependency. LLMAttention is deferred per the catalog's "voice-baseline-sensitive" note. The intent-adaptive selector (greetings -> NoOp, factual -> Recent, coding -> LLMSummarizing) implements the catalog's "adaptive switching by intent" creative extension. |
| `src/ultron/lifecycle/start_task.py` | `openhands/app_server/app_conversation/app_conversation_models.py:AppConversationStartTask` + `live_status_app_conversation_service.py:start_app_conversation` (async generator pattern) | Typed start-task state machine streamed as an async iterator (T5). The OpenHands shape (status enum, intermediate transitions yielded for UI polling) is adapted; ultron's version targets voice cold-start + gaming-mode engage + coding bootstrap rather than the multi-tenant web server. The optional `StartTaskRecorder` persists transitions into the batch-3 event store. |
| `src/ultron/lifecycle/pending_message_queue.py` | `openhands/app_server/pending_messages/pending_message_service.py` + `update_conversation_id` rekey pattern | Queue + rebind pending messages (T16). The bucket-keyed-by-temp-task-id pattern + the rebind-on-ready transition are adapted from the OpenHands SQL-table shape. Ultron's version is in-memory by default with optional JSONL persistence, and the overflow drop-oldest + drain delivery + per-message state enum are extensions for the voice cold-start UX where the queue can fill during a multi-second LLM load. |
| `src/ultron/projects/discovery.py` | `openhands/app_server/app_conversation/app_conversation_service_base.py` (the `.openhands/setup.sh` / `.openhands/pre-commit.sh` / `.openhands/microagents/` discovery convention) | `.ultron/` per-project config discovery (T7). The directory-name convention + the per-file optional-component model are adapted. Ultron's version reads + parses but NEVER invokes -- the safety validator's `is_explicit_intent` matcher gates `setup.sh` and `pre_commit.sh` execution, the supervisor handles `test_command.json`, the skill registry receives `skills/` as a fourth source. Adds ultron-specific fields (`identity_override.md`, `safety_rules.yaml`, `voicepack_override.json`, `intent_triggers.yaml`). |
| `src/ultron/services/injector.py` + `engine_injectors.py` | `openhands/app_server/services/injector.py:Injector[T]` ABC + the `AppServerConfig`-holds-an-injector-per-service pattern | Dependency-injection primitives (T6 partial). The Injector[T] ABC + InjectorState shape are adapted. Ultron's version is sync (single-process voice-first) so the FastAPI / Starlette dependencies don't come along; `.context()` returns a regular contextmanager. The catalog's "gaming-mode hot-swap as injector state attribute" extension is implemented in `STTEngineInjector` + `TTSEngineInjector` -- both read `state.mode` and dispatch between a standby factory + a gaming factory. The migration is partial per the catalog's recommendation: STT + TTS first, the rest opportunistically. |

## SWE-Agent (MIT License)

Repository: https://github.com/SWE-agent/SWE-agent
Paper: arXiv 2405.15793 (Yang et al., "SWE-agent: Agent-Computer Interfaces
Enable Automated Software Engineering").
Copyright (c) 2024 John Yang, Carlos E. Jimenez, Alexander Wettig,
Shunyu Yao, Karthik Narasimhan, Ofir Press.
License: MIT (a copy is included verbatim below).

The following ultron components are clean-room re-implementations whose
*approach* is informed by the corresponding SWE-Agent modules. Sentinel
strings + error templates + line-shift arithmetic are quoted verbatim where
the exact bytes are load-bearing; algorithmic ports are restructured to fit
ultron's bus + supervisor + safety-validator stack.

| Ultron component | Inspired by | Notes |
| --- | --- | --- |
| `src/ultron/coding/sentinels.py` | `tools/submit/bin/submit`, `tools/forfeit/bin/exit_forfeit`, `tools/windowed_edit_replace/bin/edit` | Pair-marker + single-fire sentinel parser (T17). Strings namespaced to `ULTRON_*` to avoid clashing with any SWE-Agent harness that happens to share a process. |
| `src/ultron/coding/observation_format.py` | `config/bash_only.yaml:next_step_truncated_observation_template`, `config/default.yaml:next_step_no_output_template` | Truncated-observation head + tail + elided-chars template (T10); empty-output explicit message (T19). |
| `src/ultron/coding/session_registry.py` | `tools/registry/lib/registry.py:EnvRegistry` | Per-session JSON registry (T15). Ultron version adds per-session isolation, thread-safe RLock, atomic temp-file writes, transactions, and per-key TTL. |
| `src/ultron/llm/history_processors.py` | `sweagent/agent/history_processors.py` (ClosedWindowHistoryProcessor + LastNObservations + TagToolCallObservations) | History-shape compression (T2 + T9). File-view + line-block regex patterns are verbatim ports; polling-aware elision algorithm is verbatim; ultron version adds composer + build_default_processors factory + integration into LLMEngine._build_messages. |
| `src/ultron/coding/window_expand.py` | `tools/edit_anthropic/bin/str_replace_editor:WindowExpander` | Semantic window expansion (T5). Scoring rules (blank=1, double-blank=2, def/class/decorator=3, file-edge=3) ported verbatim; direction-aware stop-before-next-def behavior preserved; ultron version adds per-suffix pattern map for non-Python languages. |
| `src/ultron/coding/file_history.py` | `tools/edit_anthropic/bin/str_replace_editor:_file_history` | Multi-file undo stack (T20). Defaultdict-of-snapshots pattern preserved; ultron version stores in per-session SessionRegistry, adds atomic write-back, narration metadata, and `find_by_narration` substring search. |
| `src/ultron/coding/window_state.py` | `tools/windowed/lib/windowed_file.py:WindowedFile` | Persistent windowed-file state (T4). Registry key names match SWE-Agent for cross-tool legibility; goto offset (1/6 down the window) and scroll overlap (default 2 lines) match; ultron version is read-only (mutation lives in the safety + file_history layers). |
| `src/ultron/coding/edit_diagnostics.py` | `tools/windowed_edit_replace/bin/edit` error templates | Edit failure diagnostics (T12). Five failure modes + message templates ported verbatim; ultron adds the AMBIGUOUS_CROSS_FILE creative extension that names other session-touched files containing the search string. |
| `src/ultron/coding/lint_diff.py` | `tools/windowed/lib/flake8_utils.py` (`_update_previous_errors` + `_LINT_ERROR_TEMPLATE`) | Pre/post lint diff + revert (T1). Line-shift arithmetic ported verbatim; twin-window revert template structure preserved; ultron version returns a structured `LintDiffResult` rather than printing to stdout, ready for the runner-side auto-revert wiring. |
| `src/ultron/coding/search_primitives.py` | `tools/search/bin/search_dir|search_file|find_file` | Filenames-only search with hard cap (T3). The `> 100 files = hard error` semantic is preserved verbatim; ultron adds a tiered-narrowing hint listing the top extensions; ripgrep backend with pure-Python fallback. |
| `src/ultron/safety/rules/category_it.py` | `sweagent/tools/tools.py:ToolFilterConfig` | Interactive-tool blocklist (T11). All three default lists (prefix / standalone / unless-regex) ported verbatim from `ToolFilterConfig`; ultron integrates them as three rule classes in the existing 19-category validator framework. |
| `src/ultron/coding/diff_snapshot.py` | `tools/diff_state/bin/_state_diff_state` + `sweagent/agent/agents.py:attempt_autosubmission_after_error` | Cumulative diff snapshot + crash-recovery salvage (T6 + T13). The `git add -A && git diff --cached` capture pattern + the `submitted (<original>)` exit-status decoration are ported verbatim; ultron adds per-session isolation, a file-list fallback when git isn't usable, persistence skipping for empty diffs, and the AutosubmissionGuard context-manager wrapper. |
| `src/ultron/coding/submit_review.py` | `tools/review_on_submit_m/bin/submit` + `SUBMIT_REVIEW_MESSAGES` registry pattern | Multi-stage submit review (T7). Stage counter + per-stage prompt template substitution shape preserved; ultron's default stages are domain-specific (voice-lock contract, test sweep status, codebase_structure.md drift). |
| `src/ultron/coding/forfeit.py` | `tools/forfeit/bin/exit_forfeit` + `_ExitForfeit` handling in `agents.py` | Forfeit primitive (T8). The "give up gracefully + salvage" semantic is preserved; ultron adds the tiered SAFE / REVERT / FOLLOWUP variants from the catalog's creative extension + the minimum-effort threshold gate. |
| `src/ultron/llm/requery.py` | `sweagent/agent/agents.py:get_model_requery_history` | Format-error requery without history pollution (T14). The temp-history shape (real + broken-assistant + error-user) and the `max_requeries=3` cap are ported verbatim; ultron's RequeryLoop is generic over generate / validate callables so call sites in the in-process classifier / web-gate / decomposer paths can opt in. |
| `src/ultron/desktop/click_preview.py` | `tools/web_browser/lib/browser_manager.py:CROSSHAIR_JS` | Visual crosshair click preview (T16). Crosshair geometry (size 20, thickness 3, red, centred on target) ported from the JS injection; ultron's adaptation uses Pillow for native desktop screenshots instead of browser-DOM injection + adds the confidence-gated auto-pass tier per user direction. |
| `src/ultron/llm/image_markdown.py` | `tools/image_tools/bin/view_image` + `sweagent/agent/history_processors.py:ImageParsingHistoryProcessor` | Image-as-base64-markdown encoding + multimodal segmenter (T18). Markdown format `![<alt>](data:<mime>;base64,<b64>)` + regex pattern + MIME whitelist + `image/jpg` -> `image/jpeg` normalisation all ported verbatim; ultron adds the optional Pillow auto-thumbnail and the `history_to_multimodal` rewrite helper. |

### MIT License (verbatim)

```
MIT License

Copyright (c) 2024 John Yang, Carlos E. Jimenez, Alexander Wettig,
                   Shunyu Yao, Karthik Narasimhan, Ofir Press

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## cline (Apache License 2.0)

Repository: https://github.com/cline/cline
License: Apache License, Version 2.0 (a copy is included verbatim below;
the same text covers the aider attribution above).

The following ultron components are clean-room re-implementations whose
*approach* is informed by the corresponding cline modules. Algorithm
shapes, contract names, and template structures are adapted; no source
code is copied verbatim. Ultron's versions are restructured to fit the
voice-first, single-host, native-Windows runtime (cline is a VS Code
extension with React webview + multi-provider LLM abstraction; the
patterns being borrowed are the agent-loop discipline and the
user-control surfaces, not the IDE-integration layer).

| Ultron component | Inspired by | Notes |
| --- | --- | --- |
| `src/ultron/llm/response_format.py` | `src/core/prompts/responses.ts` | Structured templates for LLM-facing and user-facing notices (T22). Ultron's version is shaped to ultron's tool surfaces (desktop / voice / coding / memory / search) rather than cline's coding-specific set, and adds explicit voice-friendly variants suffixed `_voice` for templates that may be spoken via TTS. The progressive-escalation pattern (tier 1 suggestion / tier 2 directive / tier 3 forbid-and-pivot) is preserved for `write_to_file_missing_content_error` so the LLM learns the same self-correction discipline. |
| `src/ultron/utils/retry.py` | `src/core/api/retry.ts:withRetry` | Async + sync retry decorator with exponential backoff (T13b). The retry-after header parsing reuses cline's delta-seconds-vs-unix-timestamp heuristic ("integer comfortably greater than current unix time is an absolute timestamp; otherwise delta-seconds"). Ultron's variant adds async-generator decoration, sync-twin, `RetryBudget` for per-session cap, and `asyncio.CancelledError` pass-through so cancellation during the backoff sleep is not swallowed. The default 429 + `RetriableError` classifier matches the upstream contract. |
| `src/ultron/search/ripgrep.py` | `src/services/ripgrep/index.ts:regexSearchFiles` | Subprocess wrapper around `rg --json` with byte-capped grouped output (T25). The CLI flags (`--json -e <pattern> --glob <filter> --context N <directory>`), the grouped-by-file output shape with `│----` separators, and the dual caps (`MAX_RESULTS = 300` matches; `MAX_RIPGREP_MB = 0.25` matches) are preserved. Ultron's variant adds Windows `CREATE_NO_WINDOW` (consistent with the rest of ultron's subprocess spawns), wall-clock kill on hang, optional `ignore_predicate` for post-filtering against `.ultronignore` policy, and Windows install-location fallback for the binary lookup. |
| `src/ultron/coding/file_read_cache.py` | `TaskState.fileReadCache` map in `src/core/task/TaskState.ts` | Per-session mtime-validated file-read cache (T7a). The repeat-read-with-unchanged-mtime semantic and the increment-and-inject-notice pattern are preserved. Ultron's variant runs entirely outside the read primitive (callers wrap their existing reader with `maybe_serve_from_cache` → read → `record_read`), adds an `RLock`-backed registry keyed by session id, and ships an optional LRU-style cap that evicts the lowest-read-count entry when the cache exceeds the configured size. |
| `src/ultron/agent_loop/loop_detection.py` | `src/core/task/loop-detection.ts:toolCallSignature` + `checkRepeatedToolCall` | Generic loop detector with canonical-signature comparison (T7b). The soft/hard escalation tiers (defaults 3 and 5) and the JSON-stringify-sorted-keys-minus-noise signature shape are preserved. Ultron's variant adds an extended `DEFAULT_NOISE_KEYS` set (covering ultron's own metadata fields like `turn_id`, `trace_id`, `correlation_id`) and a `halted` flag that persists across subsequent observations once the hard tier has fired (so a callsite cannot accidentally bypass the kill). |
| `src/ultron/llm/dedup_file_reads.py` | `src/core/context/context-management/ContextManager.ts:attemptFileReadOptimizationCore` | In-place dedup of duplicate file-read tool results in API conversation history (T18). Walk the recent slice, group by `(tool_name, file_path)`, elide every duplicate except the latest (or first), report a bytes-saved estimate. The 30 % savings threshold (`DEFAULT_SAVINGS_SUPPRESS_THRESHOLD`) for skipping a separate compaction pass matches cline. Ultron's variant adds a generalised `dedup_payload_duplicates(...)` for non-file streams (nvidia-smi heartbeats, repeated web-search snippets) so the same elision pattern applies cross-system. |
| `src/ultron/observations/safe_capture.py` | `src/services/telemetry/TelemetryService.ts:safeCapture` | Fire-and-forget observability wrapper that guarantees an emit-site failure cannot crash the caller (T17). The sync + async + decorator triple matches cline's TypeScript decorator usage. Ultron's variant adds a module-level `SafeCaptureStats` counter (`total_calls`, `success_calls`, `failure_calls`, `last_failure_message`, `per_context_failures`) so chronic observation failure surfaces without trawling the log file. |
| `src/ultron/subprocess/zombie_killer.py` | `BACKGROUND_COMMAND_TIMEOUT_MS = 10 * 60 * 1000` enforcement in `src/integrations/terminal/CommandOrchestrator.ts` | Periodic subprocess reaper with the same 10-minute hard cap on non-persistent processes (T23). The persistent-tag carve-out (so the Parakeet HTTP server, MCP server, Kokoro stream daemon are never auto-killed) is the ultron extension. Adds a resource-budget warning tier (RSS > N MB AND age > N s) that logs a notice without killing, plus a clock + terminator + RSS-probe injection surface so tests run deterministically without spawning real subprocesses. |
| `src/ultron/safety/ignore.py` | `src/core/ignore/ClineIgnoreController.ts` | Three-layer `.ultronignore` policy file with gitignore syntax via the `pathspec` library (T6). The `!include other-file` directive matches cline's behaviour. The `validate_command(cmd)` helper tokenises a shell string and rejects path arguments to the file-reading commands cline lists (`cat` / `head` / `tail` / `less` / `more` / `grep` / `awk` / `sed`) plus the PowerShell aliases (`gc` / `type` / `Get-Content` / `Select-String` / `sls`). Ultron's variant stacks three layers (global at `~/.ultron/.ultronignore`, project at `<root>/.ultron/.ultronignore`, workspace at `<root>/.ultronignore`), caches per-layer compiled `PathSpec` objects keyed by mtime, and exposes a registry-keyed singleton via `get_ignore_controller`. |
| `src/ultron/rules/conditionals.py` | `src/core/context/instructions/user-instructions/rule-conditionals.ts:evaluateRuleConditionals` | Frontmatter `paths` / `intents` / `topics` / `system_state` conditional evaluator for rule activation (T10). The path-extraction heuristic mirrors cline's (strip fenced code + URLs, then match identifier-with-slash and identifier-with-extension patterns; cap token length at 256 chars). Ultron's variant adds three condition kinds beyond cline's `paths`: `intents` (against ultron's `RoutingIntentKind` namespace), `topics` (substring + slash-delimited regex), `system_state` (with comparator-prefixed strings like `">=2"` or `"=production"`), and `all_of` / `not_in_gaming_mode` combinators. `load_rule_set([(dir, layer), ...])` walks per-layer directories and merges with later-wins precedence. |
| `src/ultron/safety/auto_approval.py` | `src/core/task/tools/autoApprove.ts` (`shouldAutoApproveTool` + `shouldAutoApproveToolWithPath`) | Per-rule auto-approval matrix decoupled from the binary trust dial (T3). The four-mode shape (``always_ask`` / ``allow_local`` / ``allow_external`` / ``allow_all``) mirrors cline's local-vs-external split. The ``yolo_mode`` master override matches cline's ``yoloModeToggled``. Ultron's variant adds a per-session "warming" allowlist where N consecutive user approvals (default 5) promote a `(rule, target)` pair into a TTL'd auto-allow set (matching the catalog's "trust gradient over time" extension). Locality is determined by an injected ``LocalityProbe`` predicate that fail-opens to "unknown" → ASK_USER. |
| `src/ultron/llm/condensers/structured_8_section.py` | `src/core/prompts/contextManagement.ts:summarizeTask` | Structured 8-section history condenser (T15). The 8 canonical headers (Primary Request / Key Technical Concepts / Files and Code Sections / Problem Solving / Pending Tasks / Task Evolution / Current Work / Next Step) and the "summary replaces all earlier history" contract match cline. Ultron's variant adds a tolerant section parser that recovers per-section bodies from the model's output (with alias resolution for `Files` / `Pending` / `Next Steps`), reports missing canonical headers so the caller can retry with a stricter prompt, and ships a 3-section `compact_for_voice` renderer that produces a TTS-friendly continuity ack from the Primary Intent + Pending Tasks + Next Step sections. |
| `src/ultron/streaming/window.py` | `src/integrations/terminal/CommandOrchestrator.ts` (line-by-line buffer + `MAX_LINES_BEFORE_FILE` / `MAX_BYTES_BEFORE_FILE` spillover) | Bounded sliding-window writer (T8). The defaults match cline verbatim: 20-line / 2 KB / 100-ms debounce; 1000-line / 512 KB spill thresholds; head + tail = 100 lines preserved when overflow spills to disk. The `COMPILING_MARKERS` heuristic (substring match for "compiling", "building", "bundling", "transpiling", "generating") mirrors cline's hot-timeout detection. Ultron's variant generalises beyond terminal output (web-reader pages, supervisor narration, RAG snippet rendering, TTS sentence-boundary chunking) via a callback-driven `on_flush` surface. |
| `src/ultron/streaming/presentation_scheduler.py` | `src/core/task/TaskPresentationScheduler.ts` | Priority-banded chunk scheduler with environment-adaptive cadence (T12). The three-band shape (immediate / normal / low) and the per-environment cadence map (local PortAudio vs Bluetooth vs remote) mirror cline's debounce-by-priority approach. Ultron's variant maps onto TTS-appropriate priorities (sentence-boundary chunks immediate, mid-sentence normal, reasoning/thinking-text low or dropped), with a `set_drop_low_priority` flag that matches the `enable_thinking=False` voice-path default. |
| `src/ultron/streaming/reasoning_stream.py` | reasoning-vs-text demultiplexing in `src/core/task/index.ts:2872-2900` | Reasoning chunk accumulator with first-text-finalises semantics (T19). The discipline (reasoning chunks accumulate to a pending block; the first non-reasoning text chunk finalises that block + resets) matches cline. Ultron's variant routes reasoning to a dedicated audit channel (`ReasoningChunkEvent`) and never lets reasoning text leak through the text channel — preserving the voice-path contract that `enable_thinking=False` keeps reasoning out of TTS. |
| `src/ultron/streaming/coordinator.py` | `src/core/task/StreamChunkCoordinator.ts` + `Task.onRetryAttempt` | Stream coordinator + retry-status surface for invisible auto-retries (T20). The state machine (`IDLE` -> `STREAMING` -> `RETRYING` -> `STREAMING` / `COMPLETE` / `CANCELLED` / `FAILED`) matches cline's lifecycle. The "publish retry attempt as an in-place status update on the existing UI message" pattern maps to ultron as `RetryStatus` payloads emitted through `on_retry`; the orchestrator can render at configurable verbosity (silent / narrate / interrupt) without re-narrating the request. `on_usage` chunk callback fires live so token meters update during the stream. |
| `src/ultron/coding/mention_resolvers.py` | `src/core/mentions/index.ts:parseMentions` | Extended `@`-mention resolver covering URLs / problems / memory / clipboard / screenshot / last-file / diff (T14). The mention regex matches the cline taxonomy (`@http(s)://...`, `@workspace:relpath`, `@problems`, `@last`, `@diff`, `@clipboard`, `@screenshot`) plus path-like tokens. Ultron's variant is provider-driven (every external surface is an injected callable) so tests run hermetically; resolution dedup'd per call; per-mention body cap + per-call mention cap protect the prompt budget. The Windows drive-letter alternative (`C:/...`) is the ultron-specific extension. |
| `src/ultron/coding/focus_chain.py` | `src/core/task/focus-chain/index.ts` | Bidirectional markdown checklist with debounced file watcher (T11). The `- [x]` / `- [ ]` contract, the 300 ms debounce, the "user edit propagates as CRITICAL INFORMATION block on the next prompt" pattern, and the per-progress-band prompt tailoring all mirror cline. Ultron's variant adds atomic temp+rename writes for agent updates, fail-open on watchdog import (manual `poll_now` keeps the surface usable without the dependency), and a `progress_hint` helper that renders TTS-friendly continuity messages. |
| `src/ultron/checkpoints/` (exclusions / shadow_repo / restore / registry) | `src/integrations/checkpoints/CheckpointTracker.ts` + `controller/checkpoints/checkpointRestore.ts` + `CheckpointExclusions.ts` | Shadow-repo checkpoint system with three-axis restore (T1). The parallel git repo + `core.worktree` pointing back at the workspace + the per-cwd-hash naming convention + the 15 s init timeout + 7 s warning all mirror cline. The three restore axes (`voice_history` / `workspace` / `both`) extend cline's `task` / `workspace` / `taskAndWorkspace` to ultron's voice-memory + bus-event surface. Ultron's `VOICE_BASELINE_PROTECTED_PATTERNS` enforces the voice-quality contract by excluding `SOUL.md`, RVC weights, Piper voice, Kokoro voicepack, LLM model files, etc. from BOTH commits AND restores — a snapshot rewind cannot accidentally roll the voicepack to a stale state. Plan-then-execute restore flow makes the destructive operation an explicit confirmation gate. |
| `src/ultron/memory/dual_history.py` | `src/core/task/message-state/MessageStateHandler.ts` (`apiConversationHistory` vs `clineMessages` dual-array pattern) | Dual-array verbatim<->api history split (T4). The "verbatim record is what the user said + heard; api history is what the LLM saw (post-compaction, post-dedup, post-RAG-injection)" separation matches cline's MessageStateHandler. Ultron's variant promotes the shape from a per-task store to a per-session primitive any caller (voice, coding, supervisor) can use, with shared `turn_id` UUID indexing so verbatim<->api resolves O(1) -- the basis for "what did I say earlier?" voice queries. The `replace_api_range` condenser hook + the `compacted` / `elided_count` ApiTurn fields power the catalog's drift-reporting dashboard ("you were silenced 14 times by compression today") via `drift_report()`. The primitive is I/O-free; callers wire their own persistence (Qdrant payload, JSONL audit log, in-memory recency cache). |
| `src/ultron/hooks/` (lifecycle / discovery / runner / registry) | `src/core/hooks/` (hook-factory + precompact-executor + HookProcessRegistry + HookDiscoveryCache) | Out-of-process hook lifecycle system (T5 + T21). The 9 cline lifecycle points (TaskStart / TaskResume / TaskCancel / TaskComplete / UserPromptSubmit / PreToolUse / PostToolUse / PreCompact / Notification) all carry over; ultron adds 5 voice-specific extensions (PreLLMRequest / PreMemoryWrite / PreGamingEngage / PreDesktopAction / WakeWordTriggered). The JSON-over-stdin/stdout envelope + `{cancel, context_modification, error_message}` contract matches cline. Ultron's variant tightens the default timeout (10 s vs cline's 30 s — voice-path latency) and caps `context_modification` at 8 kB vs cline's 50 kB to keep the context budget honest. Discovery uses mtime-validated caching with a 30 s TTL; runner picks the interpreter per file suffix (`.py` → venv python, `.ps1` → `powershell -NoProfile -ExecutionPolicy Bypass`, `.sh` → bash, `.bat`/`.cmd` → cmd, no suffix → shebang). Registry's `fire(kind, payload)` parallel-fans-out via `concurrent.futures.ThreadPoolExecutor` with cap 4 by default; any `cancel: true` blocks, every `context_modification` is concatenated into `<hook_context source="..." script="..." layer="...">...</hook_context>` blocks for the next prompt. |

### Apache License 2.0 (verbatim)

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
   implied. See the License for the specific language governing permissions
   and limitations under the License.
```

The full Apache License 2.0 text is available at the URL above. Section 4(c)
requires retention of copyright notices in derivative works; this file
satisfies that obligation for the components listed above.
