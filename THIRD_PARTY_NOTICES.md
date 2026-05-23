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
