# Ultron 1.0 — Live Status

**ACTIVE (2026-06-23) — TWITCH SIDECAR PYTHONPATH FIX:** `orchestrator._start_twitch_sidecars` now
injects `PYTHONPATH=<repo>/src` into each sidecar's env before spawn, so the sidecar can import
kenning even when `sys.executable` is the system Python (launched via a launcher that patches its
own sys.path at start-up but does NOT propagate the patch to child processes via the inherited env).
Root cause: Twitch auth tokens also needed refresh (both expired ~4 hr TTL); rotated via
`auth.TokenStore.refresh()` without re-auth (refresh tokens still valid). Both fixes committed;
restart required to pick them up.

**PREVIOUS — STOP-WINDOW CHAT TOGGLE (2026-06-23):** Added CHAT ON/OFF button to the stop-button GUI
(`stop_button.py` + `config.py` `StopButtonConfig.chat_height`/`chat_label` + `orchestrator.py`
`_set_twitch_chat_reply_enabled` setter + loop reads `self._twitch_chat_reply_enabled`).
Purple/grey accent (Twitch brand). Only wired when `twitch.enabled: true`. 8 new tests + fixed
`test_orchestrator_hook.py::test_start_twitch_chat_mode_is_noop_when_disabled` (was reading live
config.yaml which now has `twitch.enabled: true` — now uses `set_config(disabled_cfg)` pattern).
Targeted suite: 859 passed, 0 failed. Commit `0253300` on local main; published to origin/main `bc2d09d`.

**PREVIOUS — GAP-C + MISTRAL DEFAULT + SPEC-DECODING AUTO-TOGGLE folded + pushed:** local `main` at
`ee3b2ba`; published to `origin/main` as a canon-excluded snapshot. **Wrapper regression-clean: 22 failed = exact
frozen baseline, 12176 passed, 39 skipped.** All twitch/turbo/gap-c tests green.

**GAP-C DELIVERED (2026-06-23, commits `aaedc26`–`c54a364`):** `src/kenning/twitch/economy/chat_games.py` —
`ChatGameRouter` (own-cursor chat drain mirroring the redeem router) dispatches the existing `commands.parse_command`
(which had no dispatcher) → ledger-backed `!gamble`/`!slots` (debit-first + RTP-derived multiplier payout, EV ==
`gamble_rtp`, leg-distinct idempotency keys) + `!points`/`!balance`/`!leaderboard`/`!help`; watch-time earn
(`earn_per_minute`, idempotent per minute); `per_stream_loss_cap` per-viewer ceiling; per-user cooldown. KEY: the
read sidecar buffers a FLAT chat dict (`{type:chat, message_id, chatter_login, ...}`), NOT the nested EventSub shape
`ChatEvent.from_eventsub` parses — use `chat_event_from_buffer`. Config `TwitchEconomyConfig.chat_commands_enabled`/
`command_cooldown_seconds`/`min_bet`/`max_bet` (default OFF). Orchestrator builds one `Ledger` singleton + a daemon
loop (gated on economy.enabled AND chat_commands_enabled), closed on shutdown. 22 unit tests. **TRIVIA** (commit
`a13ccf5`): mod-started, draws a provably-fair question, first correct chat answer in the window wins a house prize
(`trivia_prize`/`trivia_window_seconds`); round closes atomically BEFORE crediting (no double-award). +5 tests; full
twitch suite 779 green. Spec: `docs/twitch_integration/03_spec/gap_c_chat_economy_spec.md`. STILL DEFERRED: heist
join-window / duel challenge-accept / raffle / !give / RedeemRouter ledger-backing / delete message-id cross-process
plumb (design documented in spec).

**MISTRAL DEFAULT + SPEC-DECODING AUTO-TOGGLE (commit `7767b22`):** Reverted default from `josiefied-qwen3-8b-iq3xs`
back to `mistral-7b-v0.3-abliterated` (latency regression on IQ3_XS + in-process draft). `_apply_preset` now
auto-manages `draft_kind`: preset has NO `draft_model_path` → force `"none"` (even if stale "model" left in YAML);
preset HAS `draft_model_path` AND user didn't pin → auto-set `"model"`. Effect: switching to iq4xs/iq3xs
auto-enables spec decoding; switching away auto-disables. Gaming preset also reverted to Mistral. 37 preset + 16
on-the-fly-switching tests green.

**INTENT GATE TEST FIXES (commit `ee3b2ba`):** Updated `tests/pipeline/test_always_listening_wiring.py` to reflect
the 2026-06-22 gate redesign (commit `1c7bb6f` — PRIVATE_REPLY now requires explicit name/wake; un-named utterances
go direct to IGNORE, no LLM escalation). `test_config_yaml_default_off` made env-independent (tmp_path minimal YAML).

**PREVIOUS: RELEASE 2026-06-23 — TURBO MODE shipped + folded with the twitch fleet:** turbo committed `27e0817`; merged with
`claude/determined-sutherland-315683` (8 new twitch commits — games / moderation sidecars / redeem router / EventSub)
at merge `785682a`; golden reconciled `5043b3b`. Combined wrapper (turbo+twitch) regression-clean. Published to
`origin/main` `e42277e`.

**TURBO MODE (flag-gated default-OFF):** a runtime
master switch that AUTO-RELAYS inferred team callouts WITHOUT a "tell my team" prefix. ON => the loop listens
continuously (`_listening_now()` = `_always_listening OR relay_speech.turbo_mode_enabled()`) and the 4-class
intent gate treats a callout-shaped utterance as RELAY_TO_TEAM via the existing lexical recovery
(`command_normalizer.recover_relay_lead`) + `match_relay_command`, so a bare "rotate" / "sova hit 84" / "they have
breach ult, play off site" relays straight to the team through the LLM. OFF (default) => byte-identical keyword
behaviour (only explicit "tell my team X" / "ask <agent> Q" relay; safe to talk to the stream/chat). Voice:
"turbo mode on/off" + "turbo balanced/aggressive" (sensitivity); STOP-window amber TURBO button (flips the same
flag). Implementation: `relay_speech.py` (flag/sensitivity triplets + `match_turbo_toggle`/`match_turbo_sensitivity`),
`intent_gate.py` (the `turbo` branch in `_relay_signal` + `classify_scenario`; turbo matchers in `_is_command_local`
so "turbo mode off" survives the gate), `orchestrator.py` (`_listening_now`, `_classify_always_listening` threads
turbo + the configured addressee roster, boot-apply, `_maybe_handle_turbo_command` on RAW STT in both paths,
`_set_turbo_runtime_enabled`, GUI wiring, **and the `turbo_mode_enabled()`-gated relay BACKSTOP** before the router
that force-relays a RELAY_TO_TEAM verdict the strict matcher couldn't parse — closes the aggressive-band gate/dispatch
mismatch), `stop_button.py` (TURBO button), `config.py`/`config.yaml` (`turbo_mode`/`turbo_aggressive`/`turbo_height`/
`turbo_label`, all default OFF). Spec: `docs/ultron_1_0/04_implementation/10_turbo_mode_spec.md`. Adversarial 4-agent
review: anticheat/stub/persona SOUND; 1 P1 (aggressive mismatch) + 2 P2 (kill-turbo leak, names drift) found + FIXED.
Tests: `tests/audio/test_turbo_mode.py` (incl. full example-callout coverage under balanced; yes/no/thank-you held
back) — 129 turbo+wiring+gate green; affected files green; `validate_config` 0. **PENDING: full-wrapper sign-off in a
clean window (blocked by a concurrent user twitch-test sweep) + commit.** NOTE: a heavy-suite run earlier collided with
the user's LIVE Ultron on port 8772 and took it down (BR-P3) — NEVER run the E2E/integration suite while `-m kenning` is up.

**LIVE-FIX 2026-06-23 (`8f08254`):** Route-all compose commands now reach the LLM. `_maybe_handle_relay_speech`'s
thinking-mode gate forced `rephrase=False` (thinking mode default OFF) even with route-all ON → every conversational
relay ("explain to my team X", "Reyna asked you X") fell to `_fallback_line` = the canned "No soundboard, no strings."
every time. Gate is now `thinking_mode_enabled() OR u1_llm_route_enabled()`. **FOLLOW-UP (same day):** live re-test
showed IQ3_XS STILL spoke "No soundboard" — the gate fix made the LLM be CALLED, but the quantized model returned
**0 chars** on the qa answer path (its `"\n\n"` stop fires at position 0) → empty → pool. NEW `relay_speech._relay_llm_retry`
re-prompts the LLM (generic prompt, then relaxed+thinking) whenever route-all is ON and the primary result is empty —
the pool is now a fail-open last resort only if the model is unresponsive across both retries. **ROOT-CAUSE FIX
(proven by probe, no added latency):** the quantized Qwen3 leads its answer with a blank line, so the qa sampling's
`"\n\n"` stop fired at position 0 → empty. Removed `"\n\n"` from `_ultron_answer._ANSWER_SAMPLING["stop"]` → the FIRST
qa call now succeeds (probe: empty→`len=127`), so the retry never fires for qa. +6 regression tests total; changed-area
335 pass. **Prior (`0165418`):** TTS do-inversion ("Sage, do you have a heal?") at both
question-relay entry points + `josiefied-qwen3-8b-iq3xs` preset (IQ3_XS + 0.6B draft + n_batch 2048 + q8_0 KV; ~9.3 GB
peak). VRAM line: IQ4_XS `3f78191` + q8_0 KV `a8c37c0`. STILL-PENDING: FLAG-button stale-`_last_response_text` on relay
turns; live IQ3_XS-vs-Mistral quality A/B (user-driven).

**Updated:** 2026-06-20 (M0+M1+M2 + text-injection harness landed; all regression-clean)
**Current phase:** Phase 5/6 — next concrete step = M1-wire (flag-gated) + audio MP3 E2E harness
**DONE+committed:** M0 (8B default, verified, 0 new regress) · M1 (ultron_prompt.py module, 12 tests, live-validated) · M2 (verbosity differentiation, live-validated) · Phase5 text-injection harness (`scripts/relay_test/u1_text_harness.py`, REAL-fails=0, tracks 4 u1.0-gate-targets). Full-suite regression with all of this: 22 fail (same pre-existing) / 10978 pass / 39 skip.
**M1-WIRE DONE + REGRESSION-CONFIRMED (2026-06-20):** full suite 22 fail / 10982 pass / 39 skip; failure set BYTE-IDENTICAL to the frozen baseline (diff: 0 new, 0 lost) → provably zero regressions. (+4 new u1_llm_route tests pass.) `relay_speech.build_relay_line` generic-rephrase path now flag-gated `KENNING_U1_LLM_ROUTE` (default OFF): ON → lean `ultron_prompt.build_relay_prompt` (verbosity `relay_verbosity()` + flavor `flavor_tails_enabled()`), OFF → legacy `_build_rephrase_prompt`. Added helpers `u1_llm_route_enabled`/`set_u1_llm_route_enabled`/`relay_verbosity`/`set_relay_verbosity`. `<think>` strip guard added. Fact-guards (6427-6438) UNCHANGED (already wired) + the tactical-literal pre-route (6319) keeps slot callouts deterministic = the C_route_llm HYBRID. Fixed `normalize_verbosity` multi-word ("no/low/high flavor"). Tests: `test_ultron_prompt.py` (17) + `test_u1_llm_route.py` (4) pass; isolated relay/expansion files green (flag-OFF identical); LIVE 8B flag-ON verified (in-character, fact-preserved, no think leak; tactical stays fact-perfect). Research C_route_llm reframing recorded in synthesis (route-all → flag-gated hybrid).
**M2 VOICE COMMAND DONE + COMMITTED (`4d21015`, regression-clean: failure set = baseline 22, 0 new; cleanup `828d075`):** `match_verbosity_command` ("no/low/high flavor" + synonyms, returns none/low/high; "off/on" excluded → disjoint from the tail toggle) + orchestrator `_maybe_handle_verbosity_command` wired in BOTH dispatch paths (full=user_text, lean=_raw_stt), checked BEFORE the flavor toggle (so the legacy "no flavor"=tail-off overlap resolves to verbosity none = the new u1.0 meaning; "flavor off/on" still hit the toggle). 23 tests pass incl. a source-order dispatch assertion. (Lesson: Grep/Glob need an explicit `path=` and git needs `git -C "$wt"` — the tool cwd drifts.)
**M3 AGENT-KIT INJECTION DONE (2026-06-20, pending regression bg `bgs7ggh0v`):** NEW `src/kenning/audio/agent_kits.py` — hot-swappable, version-stamped (`v2026-06-20 Patch 12.10`) 29-agent compact kit dict from B_valorant_kits + C_domain corrections applied inline (Iso suppress, Clove 8pts, Veto 7pts; Waylay/Veto/Miks/Iso flagged must-inject for the 8B cutoff) + loader `agent_kit_fact`/`kit_facts_for` (tolerant, de-dup, cap 4). Wired into the M1-wire LLM branch: agents in the callout (addressee first, via `_roster_agents`) → `agent_context=` into `build_relay_prompt`. Entirely inside the flag-ON branch (default OFF) → flag-OFF byte-identical. 53 tests pass (test_agent_kits + the 2 new wiring tests). Earlier live probe already showed agent_context makes the 8B use the real kit ("Hunter's Fury").
**SESSION-3 COMPLETE (2026-06-20): M3 `6e1d546` + M4 `fc6e5af` + M6a `eb67ff6` + M5-classifier `caed7a0`, all regression-clean (failure set = frozen 22 baseline each).** The route-all-through-LLM pipeline is COMPLETE behind `KENNING_U1_LLM_ROUTE` (default OFF): lean prompt + no/low/high verbosity (+voice cmd) + flavor toggle + agent-kit injection + compound→one-response + private prompt + the 3-way intent-gate CLASSIFIER. **REMAINING (large/risky — see `04_implementation/00_state_and_continuation.md` "REMAINING" specs): M5b always-listening loop wiring (riskiest; reuse follow-up mechanism, flag default OFF), M6b PRIVATE_REPLY routing, audio MP3 E2E harness (explicit ask), M7 retire/unify + _DOMAIN_PROMPT bug fix + golden re-bless, M8 latency (user-deferred), M9 finalize+tag.** Each is precisely specified for a clean fresh-context continuation.

**M3 COMMITTED `6e1d546`** (regression-clean, 22=22 node-id diff).
**M4 COMPOUND→ONE-RESPONSE DONE (2026-06-20, pending regression bg `b4rs6k5va`):** 3 minimal edits in `build_relay_line`, all gated on `_u1_compound` (= flag ON + not verbatim + ≥2 split-parts) so flag-OFF is byte-identical: (1) skip the deterministic `_as_compound_callout` when `_u1_compound`, (2) skip the single-tactical literal pre-route when `_u1_compound`, (3) pass `compound=_u1_compound` to `build_relay_prompt`. REFINED HYBRID (verified live): pure-slot compounds ("Sova hit 84, Breach hit 97") are caught by the slot parser → ONE deterministic fact-perfect line; mixed compounds ("Sova hit 84 and they have no smokes") → ONE LLM call w/ compound directive when flag ON, deterministic when OFF. Both = ONE response, never N LLM calls. 406 relay tests pass + 2 new M4 tests.
**NEXT:** confirm M4 regression → commit. Then M6a (fix `build_private_prompt` empty-output: needs PRIVATE-appropriate exemplars, not relay ones), audio MP3 E2E harness, M5 always-listening gate (recover 9438fc5 fusion; 4 gate-targets=acceptance), M6b wire PRIVATE_REPLY (after M5), M7 retire/unify, M8 latency, M9 finalize+tag.
**Branch:** `claude/infallible-kepler-0a865d` (worktree off `main`)
**Last green test run:** _none yet (build not started)_
**Last commit (u1.0):** see git log (Phase 0–1 committed)

## Phase board
- [x] Phase 0 — Scaffolding (commit `12959ac`)
- [x] Phase 1 — Two recon boards (22 agents, all 22 raw docs in `01_recon/raw/`) + master synthesis `01_recon/00_codebase_map.md`. (First 22-wide launch rate-limited 18/22; redo in waves of 4 recovered all.)
- [x] Phase 2 — My frontier landscape brief `02_research/00_landscape_brief_opus.md` + live 8B serving probe `02_research/01_qwen3_8b_serving_probe.md`
- [x] Phase 3 — Research board (41 agents A/B/C, all succeeded) → synthesized `02_research/02_research_synthesis.md` (6 decisions RESOLVED). Docs in `02_research/board/`.
- [x] Phase 4 — Plan finalized `03_plan/00_ultron_1_0_architecture_and_roadmap.md` (§8 post-board resolutions).
- [~] Phase 5 — E2E harness (text-injection PRIMARY + audio MP3 E2E) — STARTING.
- [ ] Phase 6 — Implementation M0→M9.

## IMPLEMENTATION ENTRY — read `02_research/02_research_synthesis.md` + plan §8 for FINAL decisions. JIT-read board docs C_route_llm/C_persona (M1), C_domain (M3), C_anticheat (M7).

## KEY: read `01_recon/00_codebase_map.md` FIRST when regrounding (it has the pivot attach-point map + line refs).

## BACKGROUND TASKS
- `wmvj56sxu` — research board (40 agents A→B→C, waves) — STILL RUNNING. On completion: read `02_research/board/` docs, synthesize `02_research/02_research_synthesis.md`, resolve the 6 [PENDING BOARD] decisions, finalize plan → then build harness (Phase 5) → M0+.
- `budnj2d81` — pytest BASELINE — DONE. **10966 passed · 22 failed · 39 skipped** (145s). All 22 are PRE-EXISTING (pristine docs-only commit) → frozen in `05_testing/00_baseline.md`. 8 are relay/normalizer (my work area, deterministic; pivot should FIX several); 14 env/infra-sensitive. REGRESSION RULE: a fail is a regression only if NOT in those 22.

## HARNESS PREREQ resolved (A4)
- Wake-splice samples: `C:\STC\ultronPrototype\training\crosscheck_ultron\*.wav` (MAIN checkout; gitignored audio, NOT in worktree). The Phase-5 harness must reference this absolute path (or copy/junction) since `gen_commands.py` looks in `<root>/training/crosscheck_ultron`.

## TEST ENV (CORRECTED 2026-06-20) — use for ALL worktree tests/model runs
- **`$env:PYTHONPATH = "<worktree>\src;<worktree>"`** — BOTH the worktree root AND src. `src` resolves `kenning`; the ROOT resolves the top-level `config` package (`kenning.audio` imports `from config import settings`). src-only fails on any module that imports `config`. Python = `C:\STC\ultronPrototype\.venv\Scripts\python.exe`.
- `$env:KENNING_ROUTER_WAIT_SECONDS="0"` (skip 30s sidecar poll); relay/flavor tests set `KENNING_FLAVOR_TAILS=1`.
- llama_cpp DLL fix is **already in `kenning/__init__.py:_register_cuda_dll_paths()`** (adds torch/lib) — `import kenning` first and llama_cpp loads. The bare-probe failure was self-inflicted.
- **`models/` junction created: `<worktree>\models` -> `E:\UltronModels`** (gitignored; lets the worktree resolve `models/...` paths). Main checkout `models/` also has the GGUFs.
- **REGRESSION CHECK CAVEAT:** the suite has cross-file LRU global-state order-sensitivity (e.g. `test_drop_weapon_possessive...[True]` fails after `test_relay_speech.py` runs first, passes in isolation). Use the FULL suite (canonical order) OR per-file isolation for regression checks — NOT arbitrary multi-file slices.

## M0 PROGRESS (2026-06-20)
- ✅ 8B serves IN-CHARACTER via the real `LLMEngine` (probe `02_research/probes/qwen3_8b_engine_verify.py`): loads 2.3s @ n_ctx=4096, **VRAM 7.1 GB resident** (safe under 10 GB; +Kokoro ~1.5 GB OK), `enable_thinking=False` works (no `<think>` leak), 0.2-0.5s/gen. Tony-Stark line perfectly in-character. KEY: bare conversational persona DISMISSES callouts ("Irrelevant. Watch the map.") → relays NEED the route's relay prompt template + directive + exemplars (the M1 work). Foundation proven.
- ✅ config.yaml default → `josiefied-qwen3-8b` + `n_ctx: 4096` (VRAM cap). Verified ZERO new regressions vs baseline (per-file isolation: exactly the 8 pre-existing relay/normalizer fails).
- ✅ M1 PROMPT ASSEMBLER built+tested+live-validated: `src/kenning/audio/ultron_prompt.py` (12 tests pass; live 8B run correct, in-character, agent-context injection works, compound→one line, no `<think>` leak). Probes: `02_research/probes/m1_module_live.py`.
- ⚠️ M1 live findings (next-step requirements): (1) FACT DRIFT frequent → fact-guards MANDATORY on wiring; (2) no/low/high not differentiating → M2 stronger directives; (3) private path returns empty → M6.

- ✅ FULL REGRESSION with the 8B default + M1 module: **22 fail / 10978 pass / 39 skip** = SAME 22 pre-existing fails (ZERO new) + exactly +12 passes (the new ultron_prompt tests). Committed work is regression-clean. Log: `05_testing/regress_8b_default.txt`.
- ✅ M2 verbosity differentiation fixed: `none` now telegraphic ("Sova, 84, A main."), `low`/`high` clipped-vs-full sentence (live-validated; low/high mutual contrast still subtle — calibration note). 12 ultron_prompt tests pass.

## ➡️ RESUME POINT: `docs/ultron_1_0/04_implementation/00_state_and_continuation.md` — the precise sequenced M1-wire→M9 roadmap with the live findings + exact attach points. STATUS + that doc + `02_research/02_research_synthesis.md` are the regrounding anchors.

## SCOPE (honest): the full M1-wire→M9 production rearchitecture is multi-session. Per "no half-implementations / don't damage the pipeline", the pivot lands as tested, flag-gated, reversible increments (NOT half-wired & broken). M0 + the M1 module are DONE+validated; the live pipeline runs its proven deterministic path (now on the 8B) until each u1.0 increment is wired behind its flag (`KENNING_U1_LLM_ROUTE`) and green.

## NEXT (when both bg tasks done)
1. Record baseline counts (from budnj2d81) here.
2. Synthesize research board → finalize plan (resolve 6 PENDING).
3. Phase 5: build enhanced E2E battery harness scaffold. Phase 6: M0→M9 implementation (tested increments, commit each).
- [ ] Phase 3 — Massive deep-research board (waves/layers) + embedded 2nd codebase scan
- [ ] Phase 4 — Comprehensive plan & framework
- [ ] Phase 5 — E2E test harness + enhanced MP3 battery
- [ ] Phase 6 — Full autonomous implementation (tested, versioned)

## Environment (verified 2026-06-20)
- GPU: RTX 4070 Ti, 12282 MiB total (~11.1 GiB free at idle). **VRAM design cap: 10 GiB.**
- Core package: `src/kenning/` (orchestrator `src/kenning/pipeline/orchestrator.py`,
  relay `src/kenning/audio/relay_speech.py`, voice lines `src/kenning/audio/voice_lines.py`).
- 8B model (chosen, pending research confirmation):
  `E:\UltronModels\Josiefied-Qwen3-8B-abliterated-v1.Q5_K_M.gguf` — Qwen3 (thinking-mode capable),
  abliterated (won't refuse trash-talk callouts). Alternatives present: `Qwen3.5-9B-Q4_K_M`,
  `Qwen2.5-7B-Instruct-abliterated-v2.Q5_K_M`.
- Downloads target: `E:\ultron_resources\` (per user instruction).
- Existing audio-battery infra: `scripts/relay_test/audio_corpus/` + `scripts/autonomous_e2e_harness.py`.

## NEXT ACTION (when re-invoked after recon bg-task `wn9pwg5ty` completes)
1. Glob `docs/ultron_1_0/01_recon/raw/` — confirm all 22 docs present (boardA_*.md ×12, boardB_*.md ×10). Re-run any missing agent directly.
2. Synthesize a master recon doc `01_recon/00_codebase_map.md` (pipeline data-flow, routing decision tree, all extension points, retire-not-remove list) from the 22 raw docs.
3. Commit Phase 1. Then craft + launch the Phase 3 big research board (waves: embedded 2nd codebase-scan + frontier search + adversarial verify + synthesis), informed by recon + the landscape brief (`02_research/00_landscape_brief_opus.md`).

## Confirmed env / serving facts (2026-06-20)
- Main venv (deps incl. CUDA llama-cpp): `C:\STC\ultronPrototype\.venv\Scripts\python.exe` (Py 3.11). Editable install targets the MAIN checkout `src/`, NOT this worktree → to run worktree code set `PYTHONPATH` to the worktree `src` (or make a worktree venv). Embedder venv: `C:\STC\ultronVoiceAudio\.venv-embedder`.
- Recon QA (boardA_semantic_router.md): embedder sidecar = EmbeddingGemma-300M on CPU, loopback HTTP :8772, urllib+numpy only (anticheat-clean). **`LexicalBackend` already uses RapidFuzz (token_set_ratio/WRatio) + Metaphone** → fuzzy/phonetic layer already a dep. HybridBackend fuses emb 0.6 / lexical 0.4. Relay-intent gate = pos/neg exemplar clouds, margin 0.06, fail-open. Router thresholds UNCALIBRATED (the enhanced MP3 battery is the labeled set to calibrate them). Recon agent independently flagged the 3-way {relay/me-only/ignore} gap → confirms pivot design.

## Recon findings so far — LOAD-BEARING (A4,A5,A7,A9 done; 18 in redo bg `wfqvbkcjs`)
**The pivot is ~70% recomposition of existing machinery.** Key facts (full detail in `01_recon/raw/`):
- **LLM serving (A7, `llm/inference.py`):** `generate_stream(user_message, system_prompt=<override>, sampling=<dict>, enable_thinking=bool, suppress_memory_context=bool, record_history=bool)` IS the route-all-through-LLM surface. When `system_prompt=` is passed, `_build_messages` returns just `[system,user]` (fast path, no RAG/history/injection-defense). The `sampling` whitelist ALREADY includes **`grammar` + `logit_bias`** (unused) = my constrained-decoding hook for combined callouts. Thinking handled: `_strip_thinking_blocks`(stream)/`strip_thinking_text`(block) + `_apply_no_think_marker` auto-appends `/no_think` for qwen-family when `enable_thinking=False`. **`josiefied-qwen3-8b` is ALREADY an LLM preset** (n_ctx 8192, no draft) → default swap is one line. Relay already LLM-rephrases (`_REPHRASE_PROMPT`@relay_speech:2081, `_RELAY_SAMPLING` max_tokens=56, `_RELAY_REPHRASE_SYSTEM`). Adaptive answer pipeline `build_answer_call`→{marvel,think_respond} curated system+sampling. `llm_prompts.py`=prompt SSOT. `response_style.py` brevity hints (procedural/factual/brief)=no/low/high substrate. `cache_aware_chunks.py`=prefix-cache substrate. `match_thinking_toggle`+`match_flavor_toggle` voice cmds exist. GOTCHA: `/no_think` only for "qwen" in model path (Llama parrots it); flash_attn=True needs non-F16 KV; logits_all must be True when draft active.
- **Config/flags (A9, `config.py`):** Pydantic v2 `extra="forbid"` → MUST add new u1.0 fields to schema before YAML. `LLM_PRESETS` extensible w/o schema change. `barebones_*` (15 lean flags) = the retire-not-remove precedent. `addressing.follow_up_enabled=false` (fusion classifier + `KENNING_ADDRESSING_TAU`=0.20 live behind it). `runtime_overrides.json` = ephemeral GUI overlay (wiped each boot). `__main__.py` sets `KENNING_FLAVOR_TAILS=0` default under `python -m kenning` (tests must set it explicitly). config.yaml `llm.gpu_layers=0` + preset `qwen3.5-4b` currently (gaming CPU 3B). `_addr_cfg` captured once at run() → addressing change needs RESTART.
- **Normalization (A4):** `routing_rules.py`=data SSOT (gazetteers/mishears/NORM2 relay-lead regexes/thresholds). `_stt_correct.py`=L1 4-stage (phrase→context→slot-confirm→phonetic+fuzzy via RapidFuzz/jellyfish, difflib fallback). `command_normalizer.normalize_command` (called orchestrator:6131)=L2; a **zero-mistakes gate returns questions/Spotify/reactions/think-respond verbatim BEFORE L1** (don't corrupt conversational). relay-intent gate fires inside `recover_relay_lead`.
- **Semantic router (A5):** EmbeddingGemma-300M sidecar (CPU, loopback :8772, urllib+numpy); HybridBackend = 0.6 emb + 0.4 lexical(RapidFuzz+Metaphone); additive fallback under exact matchers; relay-intent pos/neg clouds margin 0.06 fail-open; thresholds UNCALIBRATED (battery = the calibration set).
- **DECISIONS from recon:** u1.0 default LLM = `josiefied-qwen3-8b` GPU (gpu_layers=-1, within 10GB); reuse generate_stream override surface for all routes; use `grammar` for combined multi-callout; capture `<think>` to trace (route the already-stripped text to a log instead of discarding); add a u1.0 config section (verbosity no/low/high, flavor tail on/off, always-listen gate) to the Pydantic schema; add `barebones_skip_*` style flags to retire legacy deterministic-output paths.

## Open risks / watch-items
- `docs/codebase_structure.md` is 821 KB — query, don't read whole.
- Anticheat binding rules (`feedback_no_default_load_anticheat.md`) remain in force.
- Concurrent sessions reset `origin/main`; confirm `git rev-parse origin/main` before trusting tips.

## Rewind points
- Ultron 0.1 / 0.1.1 standalone builds: `E:\Ultron-0.1\`, `E:\Ultron-0.1.1\` (untouched).
- Dev baseline this work branches from: `6064e5f`.
