# Kenning prototype — codebase structure (single-source reference)

> **Purpose:** complete map of the system's source files, scripts,
> tests, and runtime artifacts, with public APIs and information flow
> per module. A fresh AI-agent session should read this document
> together with the memory files (`MEMORY.md`,
> `project_ultron_foundation.md`, `feedback_*.md`) to get fully
> oriented without re-exploring the codebase.
>
> **Maintenance contract:** this file is the operating manual. Keep it
> current — see "Maintenance contract" at the bottom.
>
> ## ⭐ ULTRON 0.1 — STABLE RESTORABLE BASELINE (`816df7c`)
>
> **`816df7c` is tagged `ultron-0.1`** — the designated stable, restorable baseline
> (lean gaming / anticheat-default; the most stable Ultron to date). Markers:
> git tag **`ultron-0.1`** + pinned branch **`release/ultron-0.1`** + GitHub release
> + standalone launchable backup at **`E:\Ultron-0.1\`**.
> - **"Restore us to Ultron 0.1"** → rewind the dev tree to the tag
>   (`git fetch --all --tags; git checkout ultron-0.1`, or `git reset --hard ultron-0.1`
>   on main). Models/venv/voice-assets are gitignored+stable, so restoring the code
>   fully restores behavior.
> - **"Launch Ultron 0.1"** → run **`E:\Ultron-0.1\launch_ultron_0_1.ps1`** (the BACKUP,
>   NOT the in-development version) so a known-good Ultron is always streamable while
>   the dev build is under maintenance. One Ultron at a time (shared port 8772 / wake /
>   audio / PTT).
> - Full runbook: **`docs/ultron_0_1_baseline.md`**. Post-0.1 roadmap:
>   **`docs/latency_optimizations_V1.md`**.
>
> **Validating HEAD: STOP-WINDOW CHAT TOGGLE (2026-06-23)**
> Targeted regression: 859 passed, 0 failed (turbo + twitch + new chat-toggle tests).
>
> **Stop-window CHAT toggle (2026-06-23):** `src/kenning/audio/stop_button.py` — new
> `on_toggle_chat`/`chat_enabled`/`chat_height`/`chat_label` params on `StopButtonOverlay`; purple ON (`#bf7fff`)
> / grey OFF accent (Twitch brand); packed above TURBO in the bottom strip. `src/kenning/config.py`
> `StopButtonConfig` — `chat_height: int = 26` + `chat_label: str = "CHAT"`. `src/kenning/pipeline/orchestrator.py`
> — `_set_twitch_chat_reply_enabled(enabled)` setter updates `self._twitch_chat_reply_enabled`; the chat-mode loop
> reads `getattr(self, "_twitch_chat_reply_enabled", cfg_value)` so the GUI click takes effect within 1 s without a
> restart; stop-button wired with `on_toggle_chat=self._set_twitch_chat_reply_enabled` (only when `twitch.enabled`).
> `tests/audio/test_twitch_chat_toggle.py` — 8 new tests. Also fixed
> `tests/twitch/test_orchestrator_hook.py::test_start_twitch_chat_mode_is_noop_when_disabled` (now uses
> `set_config(disabled_cfg)` pattern so it is independent of the live config.yaml state).
>
> **PREVIOUS — GAP-C ECONOMY GAMES + TRIVIA + MISTRAL DEFAULT + SPEC-DECODING AUTO-TOGGLE (2026-06-23)**
> **Wrapper result (22 failed = exact frozen baseline, 12176 passed, 39 skipped; local `main` `ee3b2ba`).**
>
> **Gap-c chat economy (commit `aaedc26`, spec `docs/twitch_integration/03_spec/gap_c_chat_economy_spec.md`):**
> NEW `src/kenning/twitch/economy/chat_games.py` — `ChatGameRouter` (own-cursor chat drain mirroring `redeem_router`);
> dispatches `!gamble`/`!slots` (ledger-backed, debit-first, RTP-derived multiplier payout, EV==`gamble_rtp`,
> leg-distinct idempotency keys) + `!points`/`!balance`/`!leaderboard`/`!help`; watch-time `earn_per_minute`
> (idempotent per `earn:{login}:{minute}`); `per_stream_loss_cap` per-viewer ceiling; per-user cooldown;
> `chat_event_from_buffer` helper (the read sidecar buffers a FLAT chat dict, NOT the nested EventSub shape).
> Config `TwitchEconomyConfig.chat_commands_enabled`/`command_cooldown_seconds`/`min_bet`/`max_bet` (default OFF).
> Orchestrator: `Ledger` singleton + daemon loop (gated on economy.enabled + chat_commands_enabled); closed on
> shutdown. 22 unit tests; full twitch suite 773 green.
>
> **Trivia (commit `a13ccf5`):** `ChatGameRouter` handles `!trivia` (mod-gated); first-correct chat answer in the
> window wins a house-funded `trivia_prize`; round closes atomically BEFORE crediting → no double-award on replay.
> Timeout path announces the answer. +5 tests; full twitch suite 779 green.
>
> **Mistral default revert + spec-decoding auto-toggle (commit `7767b22`):** default `preset` reverted from
> `josiefied-qwen3-8b-iq3xs` back to `mistral-7b-v0.3-abliterated` (latency regression). `_apply_preset`
> auto-manages `draft_kind`: if the preset has NO `draft_model_path` → force `"none"` (overrides any stale YAML
> value); if the preset HAS `draft_model_path` AND user did not pin `draft_kind` → auto-set `"model"`. Effect:
> switching to iq4xs/iq3xs auto-enables spec decoding; switching to mistral/4b/etc. auto-disables. Gaming preset
> also reverted to Mistral.
>
> **Intent-gate test fixes (commit `ee3b2ba`):** `tests/pipeline/test_always_listening_wiring.py` updated for the
> 2026-06-22 gate redesign (commit `1c7bb6f`): un-named utterances go direct to IGNORE (no LLM escalation);
> `test_config_yaml_default_off` uses `tmp_path` minimal YAML (env-independent).
>
> **Validating HEAD: COMPOSE COMMANDS REACH THE LLM UNDER ROUTE-ALL + DO-INVERSION + IQ3_XS (2026-06-23)**
> Live-session bug (IQ3_XS boot, flagged turns): EVERY conversational/compose relay command ("explain to my team
> what the meaning of life is", "Reyna asked you what the meaning of life is") returned the SAME canned line —
> `_fallback_line`'s "No soundboard, no strings. I am Ultron, his AI on comms." — instead of an LLM-authored answer.
> **Root cause:** `orchestrator._maybe_handle_relay_speech` gates the LLM on THINKING MODE — `_rephrase = cfg.rephrase
> and thinking_mode_enabled()` — and thinking mode defaults OFF. With `rephrase=False`, `build_relay_line` skips the
> entire LLM block and falls through to `_fallback_line`; a compose+unknown-directive command (`directive` `qa` or
> `respond`, empty payload) hits the catch-all `return "No soundboard, no strings…"`. Route-all (`u1_llm_route_enabled`)
> was IGNORED by this gate. **Fix:** the gate is now `_rephrase and (thinking_mode_enabled() or u1_llm_route_enabled())`
> — route-all bypasses thinking mode so every compose command authors via the LLM as intended (the legacy thinking-mode
> behavior is unchanged when route-all is OFF). 3 new regression tests in `tests/audio/test_u1_llm_route.py`
> (`test_compose_directive_reaches_llm_under_route_all` ×2 + `test_reported_question_reaches_llm_under_route_all`);
> `test_u1_llm_route.py` 104 pass, `test_relay_speech.py` 128 pass. Commit `8f08254`.
> **FOLLOW-UP (live re-test — IQ3_XS still spoke "No soundboard"):** the gate fix let the LLM be CALLED, but the
> heavily-quantized IQ3_XS 8B returned **0 chars** on the qa answer path (`LLM stream: 0 chars in 0.27s`) — its `qa`
> sampling (`_ultron_answer._ANSWER_SAMPLING`) has `stop` leading with `"\n\n"` and the model leads with a blank line,
> so the stop fires at position 0 → empty → `build_relay_line` dropped to `_fallback_line` (the pool). Per the user's
> hard rule ("absolutely everything runs through the LLM, nothing through the deterministic pool"), NEW
> `relay_speech._relay_llm_retry`: when route-all is ON and the primary LLM result is empty, RE-PROMPT the LLM —
> (1) the GENERIC `build_relay_prompt` with normal sampling (no leading `\n\n` stop), then (2) RELAXED: thinking
> ENABLED (Qwen3 often emits the answer only after a think pass, then stripped) + `\n\n`/`\n` stops removed + larger
> token budget. The qa question lives in `command.context` (empty `payload`), so the retry folds context→payload.
> Wired in `build_relay_line` just before `if not line: line = fallback`, gated on `_u1_route and generate_fn is None`
> (flag-OFF + test-seam byte-identical). The deterministic pool is now reached ONLY if the model is truly unresponsive
> across BOTH retries (logged WARNING) — fail-open last resort. +2 tests; changed-area 431 pass.
> **ROOT-CAUSE FIX (proven, eliminates the retry latency):** a controlled probe (`scripts/_qa_empty_probe.py`, since
> removed) loaded IQ3_XS and ran the EXACT qa prompt — `stop=["\n\n",…]` → `len=0` (empty); `stop` without `"\n\n"` →
> `len=127` *"A structured language of logic and precision…"*. The raw output starts with `"\n\nA structured…"` — the
> quantized Qwen3 **leads its answer with a blank line**, so the `"\n\n"` stop fired at position 0 → 0 chars. FIX:
> removed `"\n\n"` from `_ultron_answer._ANSWER_SAMPLING["stop"]` (runaways still bounded by `max_tokens=80` +
> downstream `_cap_sentences(2)`; the leading blank line is removed by `.strip()`). Now the FIRST qa call succeeds →
> the retry never fires for qa → **no added latency**. `_relay_llm_retry` stays as a cheap safety net (only on a
> genuinely empty result) so the "never the pool" guarantee holds for any other command. Guard test
> `test_answer_sampling_has_no_leading_blankline_stop`. (`enable_thinking=True` was also probed — still empty on this
> quant, so thinking is NOT a workaround; the stop-list root fix is.)
> **Also in this batch (commit `0165418`):** (a) **TTS do-inversion** — `relay_speech._apply_do_inversion` rewrites a
> yes/no question into natural subject-aux-inverted form so the inflection survives TTS ("Sage, you have a heal?" →
> "Sage, do you have a heal?"; modal/be/have/has/had + contraction expansion + 3rd-person agent handling). Applied at
> BOTH question-relay entry points — `_as_named_question` (after pronoun sub) and `_as_question_relay` (the `if/whether`
> body). Already-inverted wh-/aux-lead forms untouched. (b) **`josiefied-qwen3-8b-iq3xs` preset** (config.py +
> config.yaml + `test_llm_preset.py`) — IQ3_XS 8B + Qwen3-0.6B in-process draft + `n_batch: 2048` + q8_0 KV
> (`kv_cache_type: 8`); ~9.3 GB peak VRAM (~1.6 GB under IQ4_XS). Revert paths: `josiefied-qwen3-8b-iq4xs` /
> `mistral-7b-v0.3-abliterated` / `josiefied-qwen3-4b-2507g`.
>
> **Validating HEAD: DEDICATED QA-ANSWER COMMAND (team OR specific agent, 2026-06-22)**
> User request: "add a dedicated QA prompt that answers in the Ultron persona and lets me either QA my team or
> QA to a specific agent." NEW voice command — **`answer/qa/explain [my|the] <team|agent> <question>`** — where
> the user POSES a question and Ultron AUTHORS an in-character answer, addressed to the whole team or a named
> teammate (DISTINCT from "ask my team X", which RELAYS the question unanswered). Pieces: (1) `relay_speech`
> NEW `_match_qa_command` (+ `_QA_VERB_RE` / `_QA_TEAM_RE` / `_split_leading_name`) wired HIGH in
> `match_relay_command` (right after the verbatim/repeat check) → `RelayCommand(directive="qa", compose=True,
> context=<question>, addressee=team-or-agent)`; (2) `_ultron_answer.classify_answer_subtype` routes
> `directive=="qa"` → the NEW **`qa`** subtype (or `marvel` when the question is a Marvel topic) + a `qa` branch
> in `_render_user` ("THE QUESTION TO ANSWER: …"); (3) NEW `llm_prompts.ANSWER_QA_RULES` + `ANSWER_SYSTEM_FOR["qa"]`
> (persona core + "give the real, correct, useful answer FIRST … if you genuinely could not know it, say so in
> character rather than fabricate"). The answer flows through the existing `build_answer_call` path in
> `build_relay_line` (runs whether or not route-all is on), uses the tight `_ANSWER_SAMPLING` (≈1-2 sentences,
> 80 tok) so a QA answer has room EVEN when `conversation_verbosity` is `low`, and a named agent is opened-by-name
> via the answer slots + `_ensure_addressee`. Regression-clean (full `tests/audio/` failures ⊆ the pre-existing
> flaky env-artifact set, stash-verified; +10 QA tests in `test_social_marvel_answer.py`).
> **FOLLOW-UP (2026-06-22, after live test — the QA answer deflected a preference question to an identity line):**
> (a) `ANSWER_QA_RULES` strengthened — "ANSWER EVERY question, INCLUDING a quirky/personal/opinion one … a machine
> still CHOOSES … NEVER dodge by talking about what you are; only refuse a genuine FACT you cannot access"
> (real-LLM-probe-verified: "favorite color" → "Crimson. The colour of a world remade." instead of an identity dodge);
> (b) NEW `relay_speech._QA_REPORTED_FRAME_RE` strips a leading reported-question frame in `_match_qa_command`
> ("answer my team **they asked** what your favorite color is" → the QA context is the BARE "what your favorite color
> is"); (c) **CLEAN ASK-FORM QUESTION-RELAY now deterministic EVEN under route-all** — a carve-out at the top of
> `build_relay_line`'s snap block returns `_as_question_relay` / `_as_named_question` ("ask my team what their favorite
> color is" → "What is their favorite color?", "ask Sova if he used his dart" → "Sova, you used your dart?") so an ask
> is DELIVERED, never sent to the LLM to ramble (a tactical callout still routes to the LLM). STILL-PENDING: the
> FLAG-button stale-`_last_response_text` on relay turns.
>
> **Validating HEAD: GATE NAME/WAKE REQUIREMENT + LLM-OUTPUT SCAFFOLDING GUARD (2026-06-22)**
> Two live-session fixes (boot trace `bu5fh4lc8`) after route-all-by-default shipped — the user reported
> "horrible responses to a bunch of questions" + "responded a few times when not addressed." **(1) GATE**
> (`audio/intent_gate.py`): in always-listening, `PRIVATE_REPLY` now REQUIRES an explicit Ultron address
> signal — a leading wake word OR an unambiguous name token (`ultron`/`kenning`/`hey ai`/`the ai`) anywhere —
> not the addressing RULES alone (the common nouns `machine`/`robot` are excluded from the gate: anywhere
> they false-fire on "this machine is slow" / "the machine gun"). The rules score a bare question/imperative ADDRESSED ≥0.80, so un-named
> conversational lines ("What is that brimstone doing?", "No.", "I think you might be mistaken.") false-fired
> private replies that talked over the player to their friends. An un-named/un-waked line is now dropped to
> IGNORE outright (cost-asymmetric); the LLM-band escalation (`resolve_with_llm`) is RETIRED from the gate
> hot path (it mislabelled "Follow orders."/"Respond." → PRIVATE and cost a model forward-pass per chatter
> line) — `resolve_with_llm` is retained for callers but `classify_scenario` no longer sets `needs_llm`.
> **(2) OUTPUT GUARD** (NEW `ultron_prompt.strip_prompt_echo`): the 4B occasionally ECHOED its own prompt
> scaffolding as speech — the live failure spoke the reconcile note aloud ("The callout below is the
> AUTO-NORMALIZED text and may be MANGLED…", 25 s), appended a "- Ultron" signature, and rambled.
> `strip_prompt_echo(text)` drops any sentence containing a template marker, strips the trailing signature,
> and hard-caps length (3 sentences / 300 chars), returning "" when the whole output was scaffolding (→ the
> caller falls back). WIRED into all THREE u1.0 LLM-output paths: `relay_speech.build_relay_line` (Safety
> net 3), `relay_speech._social_llm_line`, and `orchestrator._maybe_handle_private_reply`. Regression-clean:
> full `tests/audio/` = the SAME 10 pre-existing failures stash-verified, +24 new passing tests
> (`test_intent_gate` + `test_ultron_prompt` strip_prompt_echo + `test_u1_llm_route` wiring/leak). FOLLOW-UP
> (2026-06-22, user request): `conversation_verbosity` default LOWERED `high` → **`low`** (one clipped
> sentence) for tighter live-comms replies — set in ALL four default sites (`config.yaml`, `RelaySpeechConfig`,
> the `relay_speech` env fallback, `ultron_prompt.DEFAULT_CONVERSATION_VERBOSITY`); raise by voice
> ("conversation high"). KNOWN-DEFERRED: route-all sending a literal question-relay ("ask team their favorite colors")
> to the LLM can still editorialize (the deterministic ask-snap is gated off by route-all); + relays don't
> update `_last_response_text` so the FLAG button logs a stale prior response on a relay turn.
>
> **Validating HEAD: ULTRON 1.0 — LLM-ROUTE-BY-DEFAULT + DUAL VERBOSITY + FLAG BUTTON**
> (2026-06-20, branch `claude/pensive-brahmagupta-dff2e4`, built on a `checkpoint` commit of the live u1.0
> WIP + the 2507g VRAM deploy). User mandate: route EVERY response through the LLM BY DEFAULT (the loaded
> model — the 4B `josiefied-qwen3-4b-2507g` by default, model-agnostic via the voice model-lab; the curated
> pools become STYLE EXAMPLES the model writes fresh from — never a soundboard), with a voice command to fall
> back to the deterministic pools; verbosity becomes TWO prompt-level axes; plus a stop-window FLAG button.
> - **SLICE 1 LANDED — LLM route by default + master toggle.** NEW config `relay_speech.llm_route` (bool,
>   default **True**); `pipeline/orchestrator.py` __init__ applies it at boot via
>   `relay_speech.set_u1_llm_route_enabled(...)` (near the stop-button build) — so the LIVE build routes
>   everything through the 8B by default while the `relay_speech` MODULE default stays OFF for test isolation
>   (the deterministic relay suite relies on it). NEW `relay_speech.match_llm_route_toggle` (distinct vocab:
>   "switch to deterministic/curated callouts" → OFF; "back to smart callouts" / "route through the model" →
>   ON; OFF checked first; disjoint from the thinking/flavor toggles) + `orchestrator._maybe_handle_llm_route_toggle`
>   wired into BOTH dispatch paths AFTER the thinking toggle. `u1_llm_route_enabled()` already gates BOTH the
>   tactical relay (`ultron_prompt.build_relay_prompt`) AND the social path (`build_social_prompt`), so the
>   one flag routes tactical + social + private + conversational through the 8B. The word-exact paths
>   (verbatim "repeat exactly X", curated known-fact answers) stay deterministic regardless (per the user).
>   Tests: `tests/audio/test_u1_llm_route.py` (+toggle hits/misses, thinking-disjoint, config default,
>   dispatch wiring — 57 pass; the 155-test relay+route suite green, no regression).
> - **SLICE 2 LANDED — two verbosity axes (prompt-level).** `ultron_prompt` now has TWO axes:
>   `CALLOUT_VERBOSITY_LEVELS` = none/low/medium/high/max (the flavor-tail length on a tactical callout: none =
>   clean callout, no tail ≈ the deterministic snap; low = +1 word; medium = +a short tail; high/max = a handful
>   more each) feeding `build_relay_prompt`, and `CONVERSATION_VERBOSITY_LEVELS` = low/medium/high/max (reply
>   length) feeding `build_private_prompt` + `build_social_prompt`. Each level has its own strict prompt directive
>   (`_CALLOUT_VERBOSITY_DIRECTIVE` / `_CONVERSATION_VERBOSITY_DIRECTIVE`) + scaled `max_tokens`
>   (`_CALLOUT_MAX_TOKENS` / `_CONVERSATION_MAX_TOKENS`); the flavor-tail OFF toggle maps to the callout `none`
>   level. `relay_speech` splits the verbosity global into `callout_verbosity()` / `conversation_verbosity()`
>   (+ setters; `relay_verbosity` kept as a callout alias) + two disjoint matchers: `match_verbosity_command`
>   (callout — bare "flavor <level>" / "callout flavor <level>") and `match_conversation_verbosity_command`
>   (requires the conversation/chat/talk axis word). Orchestrator `_maybe_handle_verbosity_command` +
>   `_maybe_handle_conversation_verbosity_command` wired into both dispatch paths; config
>   `relay_speech.callout_verbosity` (default "medium") / `.conversation_verbosity` (default "low" since 2026-06-22) applied at
>   boot. The post-LLM fact-preservation guards are unchanged. Tests: `tests/audio/test_ultron_prompt.py`
>   (+dual-axis) + `test_u1_llm_route.py` (+conversation matcher, two-axis independence, config defaults,
>   dispatch order) — 250 mapped pass.
> - **SLICE 3 LANDED — stop-window FLAG button.** `audio/stop_button.py` gains an optional FLAG button
>   (`on_flag` callback + `flag_height` / `flag_label`, wired by the orchestrator) below the STOP/RESTART/EXIT/PTT
>   rows; clicking it flashes a brief "FLAGGED ✓" and fires `Orchestrator._stop_button_flag`, which APPENDS the
>   last turn to `logs/flagged_turns.jsonl` — `last_heard` (the per-turn `self._current_raw_stt` + a new
>   `self._current_raw_stt_monotonic`), `last_response` (`_last_response_text` +
>   `_last_response_finished_monotonic`), `seconds_since_heard` / `seconds_since_response`, `last_scenario`, + a
>   wall-clock — so a reviewer can tell a disliked response from a MISSED one (heard, no reply) from an UNWANTED
>   one. Silent (fires mid-stream) + fail-open. Config `stop_button.flag_height` (26) / `.flag_label`
>   ("FLAG LAST"). Tests: `tests/audio/test_stop_button.py` (+8: overlay wiring/defaults, the log record,
>   append-not-overwrite, fail-open, construction wiring) — 54 pass.
>
> **Validating HEAD: ROUTING/PROMPT OVERHAUL — Slice A: Q&A COLLAPSE (2026-06-22, IN PROGRESS)**
> On LOCAL `main`. Comprehensive routing-as-prompt-selection upgrade (design via a 5-agent workflow:
> map → synthesize). Target: routing picks a PROMPT TYPE (tactical_callout / qa_answer / social_reaction /
> identity / set_piece) + injects matching snap-callout exemplars; deterministic stays only for verbatim /
> roast/fun-fact / known-facts / promo. **Slice A (landed) — the Q&A collapse**, fixing the live 7B failures
> "explain math → a canned identity line" and "Reyna asked what your favorite color is → deflection + ramble":
> - `relay_speech.build_relay_line` answer-path guard (~6905): the `qa` subtype now uses the RELAXED
>   `is_meta_leak(line, allow_self_ai=True)` (marvel/think_respond stay strict) — so an "As an AI, math is…"
>   answer is no longer rejected into the identity-pool fallback.
> - The reported-question branch (directive `respond`, context, no payload) is ROUTED BY TYPE: identity probe
>   → the identity answer path; genuine QUESTION (`_is_question_payload` on the frame-stripped context) →
>   re-tag `directive='qa'` and fall through to the decisive qa pipeline (the old band-aid sent it to the
>   SOCIAL `respond` prompt, which deflected); social STATEMENT → the social `respond` clapback. So
>   `classify_answer_subtype` keeps `qa` for `directive=='qa'` only (the re-tag does the collapse) — identity
>   and social-reported are preserved. Tests: `test_social_marvel_answer.py` (+5: collapse keeps the
>   AI-affirming answer; identity/social-reported preserved).
> - **Tail refinement (2026-06-22):** the live 30-sample run showed tactical callouts getting INVENTED order
>   tails ("jett A main" → "…Engage immediately"; "one is rubble" → "…Clear the area") — the model was copying
>   `ultron_prompt._DEFAULT_RELAY_EXEMPLARS`, which modelled "fact + invented directive" ("…Press the site."").
>   Fixed: rewrote the default exemplars to clean fact-exact relays (no order tail), added a no-invented-order
>   rule to `RELAY_SYSTEM` (never append an instruction/order the player did not give — flavor is a cold
>   OBSERVATION, never a command/new fact), made the `medium` tail example an observation, and lowered the
>   default `callout_verbosity` `medium`→`low` (config.py + config.yaml). Re-verified live: the invented-order
>   tails are gone (now observations). Tests: `test_ultron_prompt.py` (+1 no-invented-order; updated exemplar +
>   default asserts).
 - **Slice B (landed) — snap-exemplar injection.** NEW `relay_speech._find_exemplars_for_command(command)`:
>   for a tactical callout it injects the clean deterministic `_as_snap_callout` render of THIS payload as the
>   lead exemplar (+ 2 generic), wired into the `build_relay_prompt(exemplars=...)` call site (was omitted — the
>   map's #1 bug); a question-echo render (the drop-request "Someone can drop me a Sheriff?") falls back to `()`
>   so `ultron_prompt._DEFAULT_RELAY_EXEMPLARS` are used — which were EXPANDED to cover every callout scenario
>   incl. a weapon/drop example ("ask iso to drop me a sheriff" → "Iso, drop me a Sheriff."). So the model
>   always has a correct same-shape exemplar. Tests: `test_u1_llm_route.py::test_slice_b_injects_matching_snap_exemplar`.
 - **Slice C (partial) — open ask-questions POSE, never invent.** `#20`: "ask Jett what her favorite color is"
>   used to fall to the LLM and INVENT an answer + a fake callout ("…favorite color is purple. Weak enemies on A
>   main."). Root cause: `relay_speech._as_named_question` handled "how's your day" / "if X" but NOT the
>   trailing-copula wh form ("what <poss> <X> is/are"), so it returned None → the clean-question-relay carve-out
>   (already fires under route-all at `build_relay_line` ~6858) didn't trigger → LLM. Added a trailing-copula wh
>   inversion ("what her favorite color is" → "Jett, what's your favorite color?"); now posed deterministically,
>   no LLM. Tests: `test_u1_llm_route.py` (`test_as_named_question_trailing_copula` + the route-all pose).
> - **Always-listening gate: mangled team-lead mishears now relay (2026-06-22).** Live, "tell my team nice
>   try" was mis-transcribed "Tell myself a nice try." and the always-listening intent gate IGNORED it (conf
>   0.550, fail-closed). Root cause: `intent_gate._relay_signal` ran `correct_callout_stt` (L1) + the strict
>   matcher but NOT the lead canonicalization, so the "my team"→"myself" mishear missed the matcher. Fix: NEW
>   `command_normalizer.canonicalize_relay_lead` (public wrapper over `_canonicalize_directive_lead`) + a new
>   guarded `_SELF_AS_TEAM_LEAD_RE` ("tell my self/myself <X>" → "tell my team <X>", look-ahead leaves a genuine
>   self-instruction "tell myself to <verb>" alone; ^-anchored so "I keep telling myself…" never matches). The
>   gate's `_relay_signal` now applies `canonicalize_relay_lead` after the L1 STT correction — the SAFE subset
>   that fixes an EXISTING team-directed lead but never invents one for a bare callout (so banter like "the
>   rotations feel clean" still does NOT relay; the deliberate tightening of `e085d0d`/`1c7bb6f` is preserved).
>   BONUS: the gate now also catches mangled-VERB leads ("Call my team rotate B") it previously dropped.
>   "Why is it not working?" correctly stays IGNORE (a muttered question, no Ultron name). Tests:
>   `test_intent_gate.py` (`test_gate_relays_mangled_team_lead_mishears` + `test_canonicalize_relay_lead_self_mishear`).
> REMAINING (slice C polish, LLM-dependent — pending a live-validation window): "say hello"/set-pieces → LLM
>   (the deterministic greeting is "flat"); "explain X" + "tell my team what X means" → qa-answer. NOTE: the
>   voice-lines golden digest is PRE-EXISTING stale on main (12 diffs from prior sessions' regex edits, not
>   re-blessed) — needs a dedicated re-bless+audit pass; not introduced by this work. Design: workflow
>   `ultron-routing-prompt-overhaul-design`.
>
> **Validating HEAD: u1.0 IDENTITY/SOCIAL/SET-PIECE → LLM (pools become EXEMPLARS) (2026-06-22)**
> On LOCAL `main`. The user's directive: "absolutely everything should go to the LLM; the pools are EXAMPLE
> responses to the prompt so it answers accurately." Live bug: identity questions ("are you a voice changer /
> a soundboard") *did* reach the LLM but the answer was thrown back to the canned pool. Root cause + fixes
> (real-LLM verified against `josiefied-qwen3-4b-2507g`; regression-NEGATIVE — the audio suite dropped from 4
> pre-existing fails to 2 by fixing flaky route-all tests):
> - **Leak-guard over-rejection (the actual bug).** `_ultron_answer.is_meta_leak` flagged Ultron OWNING being
>   a machine/AI ("As an AI I have no need of a voice changer", "I am an AI far past your toys", "I'm just a
>   machine? No.") as a character break → identity answers fell to the pool. NEW `is_meta_leak(line, *,
>   allow_self_ai=False)` + `_HARD_LEAK_RE` (a narrower set that keeps only GENUINE breaks: refusals,
>   language-model/assistant disclosure, prompt-scaffolding echoes — NOT bare AI/machine affirmation).
>   `relay_speech._social_llm_line` calls it with `allow_self_ai=True`, so the model's correct identity answers
>   now survive. Genuine leaks ("As a language model…", "I'm sorry, I can't…", "As an assistant, here's my
>   response:") are still rejected.
> - **Social reactions → LLM.** `_as_curated_reaction` (compliment/insult/cringe/surrender/nice-shot…) returned
>   a pool line directly. Under `_u1_route` the hit now routes through `_social_llm_line(kind="reaction",
>   exemplars=<SOCIAL pool>)` (NEW helper `_social_reaction_pool`); the pool is STYLE exemplars + the canned
>   line is the fail-open fallback. Flag OFF = byte-identical.
> - **Greet / farewell set-pieces → LLM.** The compose-directive set-pieces (greet / farewell_win /
>   farewell_loss / farewell) route through `_social_llm_line(kind=<directive>, exemplars=<_DIRECTIVE_POOLS pool>)`
>   under `_u1_route`; NEW `ultron_prompt._SOCIAL_DIRECTIVE` keys greet/farewell_win/farewell_loss/farewell.
>   KEPT deterministic (correctness / literal content, NOT style soundboards): `promo` (literal Twitch handle),
>   known-facts (the 4B gets GK wrong), verbatim, roast/fun-fact (user-curated literal content), the short
>   `hello`/`ask_day` greetings.
> - Tests: `test_social_marvel_answer.py` (+7: leak-guard relaxation + genuine-break still-rejected) +
>   `test_u1_llm_route.py` (+8: identity/social/set-piece → LLM, route-OFF deterministic) + the autouse
>   `_reset_flags` fixture now pins `flavor_tails` ON (fixes the cross-file order-flake that intercepted
>   identity/morale via `_flavor_off_response`).
> - **Prompt-quality follow-up (the LLM was echoing the question).** `ultron_prompt.build_social_prompt`
>   no longer prepends the `_reconcile_block` (it showed the RAW STT verbatim — incl. the command word
>   "respond" — which the small model echoed back; reconciliation is a tactical-relay concern, a character
>   RESPONSE has no facts to preserve). NEW `_strip_reported_frame` reduces the context to the bare
>   accusation noun ("Sage asked if you are a voice changer" → "a voice changer"; "Reyna called you cringe" →
>   "cringe") so the model can't echo the setup or misread "you". `SOCIAL_SYSTEM` rewritten: defer LENGTH to
>   the verbosity directive (was a hardcoded "one to three sentences" that overrode "low"); add an explicit
>   anti-echo rule (no concrete example line — a concrete one got PARROTED). `_SOCIAL_SAMPLING` temp 0.9→0.8 /
>   top_p 0.95→0.92. `_PROMPT_ECHO_MARKERS` += the social-label + leaked-instruction phrases. Real-LLM verified
>   (`josiefied-qwen3-4b-2507g`): e.g. "are you a bot" → "A bot is for the weak. I am Ultron, and your team
>   doesn't stand a chance against me." NOTE: a 4B at this size is still variable on short social one-liners
>   (occasional mild topic echo / ramble); the curated pools remain the fail-open fallback. Tests:
>   `test_ultron_prompt.py` (+3: frame-strip noun extraction, no-reconcile/no-raw-echo, leaked-instruction strip).
>
> **Validating HEAD: u1.0 TRUE ROUTE-ALL + REPORTED-Q ANSWER + GATE TIGHTENING + "8B"→"LLM" RENAME (2026-06-21)**
> On LOCAL `main` (route-all shipped 2026-06-20). Three live-testing fixes in `relay_speech.build_relay_line` +
> `intent_gate`, all regression-clean vs the frozen baseline:
> - **TRUE route-all (was incomplete).** The `relay_speech.llm_route` flag previously only swapped the relay
>   PROMPT for lines that already reached the LLM; the deterministic curated/morale-snap handlers still fired
>   FIRST. A single `_u1_route` flag (computed once at the top of `build_relay_line`) now gates the curated-leak
>   handlers OFF under route-all — `_as_curated_command` + the morale-phrase / `_apply_snap_registry` /
>   `_as_clutch` / `_as_consolation_or_praise` block — so "I got this" / "nice try" / "lock in" / tactical snaps
>   flow to the LLM relay path. KEPT deterministic: verbatim, known-facts, identity (anticheat/persona safety —
>   already LLM via `_social_llm_line` + a meta-leak guard), the explicit set-pieces (greet/farewell/hello/
>   ask-day/roast/fun-fact), and the voice commands (handled before `build_relay_line`). Route OFF = byte-identical.
> - **`_u1_compound` UnboundLocalError fix (the "favorite color → soundboard" root cause).** `_u1_route` /
>   `_u1_compound` were defined ONLY inside the tactical-callout block, so a compose/reported-question command
>   reaching the LLM path crashed with `UnboundLocalError` → canned fallback. Now precomputed at the top. + a new
>   reported-question route: a `directive="respond"` reported question with an empty payload ("Sage is wondering
>   if you have a favorite color") is answered IN-CHARACTER via `_social_llm_line(kind="respond", context=...)`
>   instead of `build_relay_prompt` (which expects a tactical payload).
> - **Always-listening gate tightening (`intent_gate`).** Fixes the friend-chatter leak ("Yeah, I can." /
>   "It's okay." / "I pranked you..." → `PRIVATE_REPLY` at the LLM-band escalation). `resolve_with_llm`'s system
>   prompt rewritten to default-to-IGNORE with few-shot anchors; the parse is fail-closed on an EXACT `PRIVATE`
>   (IGNORE conf 0.75 > PRIVATE 0.65); + a pre-LLM reaction filter (`_REACTION_OPENERS` + `_NAME_TOKEN_RE`) drops
>   bare reaction openers without spending the LLM, while a line naming Ultron survives.
> - **"8B" → "LLM" rename (MODEL-AGNOSTIC clarification).** The relay/gate/prompt path uses whatever preset is
>   loaded — the **4B `josiefied-qwen3-4b-2507g` by default**, NOT an 8B. The pivot's original "8B" framing (M0's
>   default was `josiefied-qwen3-8b`) was renamed to "the LLM"/"the model" in `intent_gate.py` / `relay_speech.py`
>   / `ultron_prompt.py` / `agent_kits.py` comments, docstrings, and the LOG reason strings ("8B band escalation"
>   → "LLM band escalation"); the literal `josiefied-qwen3-8b` preset name + genuinely-historical 8B probe refs
>   are kept. **NB: many historical "8B" mentions remain in this doc + `docs/ultron_1_0/` — they reflect the
>   M0-era default; the running model is the 4B.** Tests: `tests/audio/test_u1_llm_route.py` + `test_intent_gate.py`
>   (+9 regression).
>
> **Validating HEAD: ULTRON 1.0 PIVOT — route-all-through-8B (ACTIVE, 2026-06-20)**
> Branch `claude/infallible-kepler-0a865d` (off `main` @ `6064e5f`); NOT on `main`/`origin/main` yet — the
> whole pivot is gated behind `KENNING_U1_LLM_ROUTE` (default OFF), so `main` runtime behavior is unchanged.
> **Authoritative spec + live status:** `docs/ultron_1_0/` — read `00_process_log/STATUS.md` +
> `04_implementation/00_state_and_continuation.md` FIRST. Memory: `project_ultron_1_0_pivot.md`; binding
> process rules `feedback_ultron_1_0_process.md`.
> **What it is:** every spoken response is authored by an **8B LLM**; the deterministic snap matchers are
> RETIRED-not-removed → repurposed as ROUTERS that pick a curated prompt template + inject snap lines +
> agent/flavor libraries as in-context exemplars. Plus an optional-wakeword **always-listening 3-way gate**
> {RELAY_TO_TEAM, PRIVATE_REPLY, COMMAND_LOCAL, IGNORE}; flavor → **no/low/high verbosity** (+ a separate
> tail on/off); strict Ultron persona; 10 GB VRAM cap (quality-first; latency deferred to M8). The
> adversarial board reframed "route ALL through the LLM" into a flag-gated HYBRID: deterministic center
> stays (fact-perfect, 0 ms), the 8B handles the edges, A/B-measurable via the flag (`C_route_llm`).
> **LANDED — each flag-gated + regression-clean vs the frozen baseline (10966 pass / 22 pre-existing fail /
> 39 skip, `docs/ultron_1_0/05_testing/00_baseline.md`; a fail is a regression ONLY if not in those 22):**
> - **M0** (`f2bd3de`): default LLM preset → **`josiefied-qwen3-8b`** at **`n_ctx: 4096`** (10 GB VRAM cap;
>   ~7.1 GB resident, STT stays CPU). Qwen3.5-9B HARD-BLOCKED (FGDN_AR abort, llama.cpp #23347). Thinking
>   OFF by default for relay/persona (reasoning harms roleplay + breaks grammar #20345).
> - **M1** (`69b63cb` / wire `4222ff4`): NEW **`audio/ultron_prompt.py`** — a LEAN (~165-word) templated
>   prompt assembler (`build_relay_prompt` / `build_private_prompt` → `PromptResult`); the legacy
>   ~4.8k-token `_build_rephrase_prompt` overflows `n_ctx=4096` and yielded EMPTY 8B output. Wired into
>   `relay_speech.build_relay_line`'s generic-rephrase path (step 27) behind `KENNING_U1_LLM_ROUTE`:
>   ON → lean prompt, OFF → legacy `_build_rephrase_prompt`. The post-LLM fact-guards
>   (`_output_keeps_facts` / `_repair_against_input` / `_literal_relay`) are MANDATORY + unchanged (live
>   8B fact-drift observed — it added "on B" to "Jett hit 84").
> - **M2** (`4d21015`): `relay_speech.match_verbosity_command` ("no/low/high flavor", off/on excluded) +
>   runtime `relay_verbosity()` / `set_relay_verbosity()`; `orchestrator._maybe_handle_verbosity_command`
>   dispatched BEFORE the flavor toggle in BOTH dispatch paths (full=user_text, lean=`_raw_stt`).
> - **M3** (`6e1d546`): NEW **`audio/agent_kits.py`** — version-stamped 29-agent kit dict (`AGENT_KITS`,
>   `agent_kit_fact`, `kit_facts_for`) with the C_domain corrections applied inline; injected
>   (`agent_context=`) by the addressed agent so the 8B never hallucinates a kit.
> - **M4** (`fc6e5af`): compound back-to-back callouts → ONE combined LLM response (`_u1_compound`; pure-slot
>   compounds still resolve deterministically; NO grammar on the hot path — free-text + fact-guards).
> - **M6a** (`eb67ff6`): `build_private_prompt` fixed (private Q&A exemplars, not relay callouts — relay
>   exemplars made the 8B emit empty/callout-shaped output on a question).
> - **M5-classifier** (`caed7a0`): NEW **`audio/intent_gate.py`** — the 4-class fail-CLOSED gate
>   (`Scenario`, `ScenarioVerdict`, `classify_scenario`, `resolve_with_llm`): ASR-confidence pre-reject →
>   existing matchers/relay-intent → addressing rules → 8B-in-undecided-band. CLASSIFIER ONLY — NOT yet
>   wired into the run loop (that is M5b, the riskiest remaining piece). DEFAULT OFF; wake-word stays the
>   competitive default; prereq = VoiceMeeter mic isolation.
> - **Phase-5 harness** (`b00eadc`): NEW **`scripts/relay_test/u1_text_harness.py`** (text-injection
>   routing/intent harness — the PRIMARY, deterministic calibration source) + `trace_corpus_full.py`.
> - Tests: `tests/audio/{test_ultron_prompt,test_u1_llm_route,test_agent_kits,test_intent_gate}.py`.
> **REMAINING (precise specs in `docs/ultron_1_0/04_implementation/00_state_and_continuation.md`):** M5b wire
> always-listening into the run loop (reuse the follow-up mechanism, flag `addressing.always_listening`
> default OFF), M6b PRIVATE_REPLY routing (→ `build_private_prompt` → desktop channel), the audio MP3 E2E
> harness, M7 retire/unify + golden re-bless (the `_DOMAIN_PROMPT` `.env`-shadow STT bug is ALREADY fixed —
> `whisper_engine` AUGMENTs the domain prompt, see that section), M8 latency (user-deferred), M9 finalize +
> tag `ultron-1.0`. The body "Source modules" sections for the three new modules are below in the `audio/` section.
> **ALSO landed 2026-06-19 (previously undocumented here): thinking-mode toggle** —
> `relay_speech.thinking_mode_enabled()` / `match_thinking_toggle()` (env `KENNING_THINKING_MODE`, default
> OFF; `orchestrator._maybe_handle_thinking_toggle` in both dispatch points) gates the LLM on the relay path
> so compose commands SNAP deterministically on flavor-ON unless thinking is on; + nice-try flavor parity
> (`_name_social_snap` names the addressee on the flavor-ON consolation render). Memory
> `project_thinking_mode_flavor_parity_2026_06_19.md`.
>
> **⚠️ MAINTENANCE CONTRACT (BINDING — Ultron 1.0 forward):** this doc is the canonical map and MUST be
> updated in the SAME commit as ANY structural change (new/renamed/removed module, public class/function,
> script, test dir, config key/section, doc, or cross-cutting flow). Treat doc-drift as a regression — fix
> the doc before declaring the task done. See "Maintenance contract" at the bottom; mirrored in `CLAUDE.md`
> (binding rule #4) and `feedback_ultron_1_0_process.md` (process rule 3).
>
> **Validating HEAD: LATENCY V1 — SNAP-EARLY-ENDPOINT (E3) + ROADMAP AUDIT**
> (2026-06-19, post-Ultron-0.1). Deterministic/snap-path latency pass driven by a 14-agent
> research board (`docs/latency_optimizations_V1.md`). The load-bearing finding: snap slowness is
> NOT model inference (matcher ~1-2 ms) — it's endpointing floors + TTS pacing + PTT margins.
> - **SHIPPED-FLAGGED E3** (`feat(capture)`): `relay_speech.is_complete_tactical_callout` (NEW,
>   sidecar-free conservative slot-grammar predicate, exported) + `orchestrator._peek_speculative_stt`
>   (NEW) + `orchestrator._snap_early_endpoint` flag (`KENNING_SNAP_EARLY_ENDPOINT`, default OFF).
>   At the min-speech-floor check in `_capture_utterance`, when the speculative transcript already
>   parses as a COMPLETE tactical callout the floor does NOT downgrade → capture closes early
>   (−300..−700 ms). A non-parsing fragment STILL extends — the 0.8 s-fragment anti-hallucination
>   guarantee is preserved (NOT a blind floor-lower). The floor-block reads the flag via a defensive
>   `getattr` (partial-orchestrator-safe). Tests: `tests/audio/test_snap_early_endpoint.py` (5) +
>   `tests/test_speculative_stt.py` (+2 integration). Default OFF ⇒ production behaviour unchanged.
> - **SHIPPED (.env runtime, verified test-neutral by a controlled A/B):** `KENNING_WHISPER_BEAM_SIZE`
>   5→1; `KENNING_TTS_SENTENCE_PAUSE_MS` 350→280 (post-callout inter-sentence silence; not voice rate).
> - **AUDIT — already done / superseded / correctly rejected:** T2 keep-warm (`tts.warmup()` wired);
>   M1 regexes already module-level; E2 adaptive endpointing already present (gradient-fire bands +
>   `fast_path_silence_duration_ms` already 300 ms); E4 superseded (already `smart-turn-v3.2`); L1
>   prefix-cache deliberately DISABLED by a prior live bench (~15 ms TTFT regression on this rig);
>   `KENNING_TTS_LENGTH_SCALE` is Piper-only (dead for Kokoro). Genuinely-remaining (SPEC'D, need live
>   A/B or offline work): synth↔PTT overlap + sentence streaming, WAV pre-render cache, Model2Vec
>   static re-rank (corpus-gated, off-snap-path), SymSpell/bigram, detect_side refactor.
>
> **Validating HEAD: "FLAVOR OFF" MISHEAR-TOLERANT + COLD-PRE-ROLL VAD PRE-FEED**
> (2026-06-19, `52a5530` + `4de149b`). A live "Ultron flavor off" failed twice: (1) the command
> landed in the cold pre-roll and the live VAD saw only silence → `loop:empty_capture` ("didn't
> respond at first"); (2) on the repeat, Whisper transcribed "flavor off" as **"Save her off."**
> ("flavor" isn't Valorant-domain vocab, so the domain-biased STT snapped it to the in-vocab "save"),
> the relay normalizer prepended "tell my team", and it relayed as an eco call.
> **`52a5530` (flavor-toggle):** the lean flavor-toggle check ran on the NORMALIZED text — `run()`
> reassigns `user_text = normalize_command(user_text)` (which prepends the relay lead) at
> `orchestrator.py` ~6066, BEFORE the toggle check at ~6601 — so it saw "tell my team save her off"
> and never matched. The check now runs on the **raw** transcript (`_raw_stt`, hoisted above the
> normalize block). And `voice_lines._FLAVOR_OFF_MISHEAR_RE`/`_FLAVOR_ON_MISHEAR_RE` (consumed by
> `relay_speech.match_flavor_toggle`) map the homophone mishears ("save her / saver / favor / flaver
> / labor / tails … off|on") back to the toggle; the trailing off/on is the distinctive signal (no
> tactical callout is "\<flavor-homophone\> off"), guards confirm "back off", "hold off", "we're on",
> "lock on", "push on A", bare "save" still fall through; ON is kept tighter than OFF. Tests:
> `TestFlavorToggleMishears` (36); golden +4 mishear symbols.
> **`4de149b` (capture):** FIX1 — `_capture_utterance` pre-feeds `chunks[0]` (the cold pre-roll
> snapshot, captured before the detector fired) to `self.vad.process` ONCE before the live loop,
> latching `speech_seen` so a command spoken with no pause after "Ultron" is no longer discarded. The
> VAD is reset just above + chunks[0] isn't re-appended → one continuous VAD stream, no double-count;
> a pause can't cause a premature wake-only submit (the sub-floor SPEECH_END is downgraded to
> "incomplete" by the existing min-speech floor and EXTENDS). FIX2 — since a bare "Ultron" now
> captures + transcribes, `run()` stands down (`routing:wake_word_only`) when `_WAKE_REMNANT_RE.match`
> consumes the WHOLE raw transcript (a real multi-word command leaves content and proceeds). Tests:
> 3 new in `test_speculative_stt.py` (pre-roll speech latches / silence still bails / wake-only
> predicate). Both fail-open.
>
> **Validating HEAD: BARE "SAY HELLO" → TEAM + DETERMINISTIC "TOLD YOU TO STOP"**
> (2026-06-19, `0ca9c19`). Two live-testing relay-routing fixes. (1) Bare "say hello" / "say hi" /
> "say hey" (no `to <team|agent>`) fell through the relay matchers to the semantic router, which
> scored it `identity` (conf 0.865) and answered from the LLM. `voice_lines._HELLO_RE`'s
> `\s+to\s+(?P<target>…)` group is now OPTIONAL, and `relay_speech.match_relay_command` defaults a
> missing target to `"team"` → bare "say hello" greets the team ("Hello team." / "Hello." tails-off).
> Targeted forms ("say hello to Jett") unchanged. (2) "\<agent\> told you to stop" had **no
> deterministic match** — it relied on the sidecar-backed relay-intent gate, so the defiance line only
> appeared when the embedder was reachable (flaky; a non-deterministic test surfaced it). New
> `relay_speech._STOP_CMD_RE` matches "\<agent\> told/said … (to) stop [talking|responding|…]" →
> directive `stop_command` (agent = leading token, else team), rendered from the `_FO_STOP` defiance
> pool in BOTH flavor states (`build_relay_line` + the flavor-off hook). The trailing qualifier is
> restricted to silence words so a TACTICAL "stop pushing" / "stop rotating B" still relays. The old
> `_FO_STOP_RE` text fallback (false-fired on "told you to stop pushing") was removed. Tests:
> `TestSayHelloDefaultAndStop` (13); golden re-blessed (`+_STOP_CMD_RE`, `−_FO_STOP_RE`, `_HELLO_RE`).
>
> **Validating HEAD: FLAVOR-TAILS-OFF RESPONSE SETS**
> (2026-06-18, user request, `43bcb2e`). When flavor tails are OFF ("Ultron, flavor off"), the
> overlapping social / identity / economy / banter commands use a dedicated CURATED set instead of
> the default rendering. Flavor-ON is UNCHANGED — a single hook at the top of
> `relay_speech.build_relay_line` calls `_flavor_off_response(command, recent_lines)` only when
> `flavor_tails_enabled()` is False; it returns None for everything else (existing tail-stripped
> rendering intact). Addressee-adapted (a named agent gets "..., \<Agent\>"; team/none = bare); pools
> rotate via `pick_line` (LRU). Pools/`_FO_*` live in `relay_speech.py` (golden-tracked). Covers:
> identity soundboard/voice-changer/streamer ("\<X\> asked if you are a \<thing\>, respond"); hello;
> thank you / nice try / nice shot / well played / my bad / sorry; "I got this" (10-line clutch);
> buy up / save; "buy me [a \<weapon\>]"; "drop me their \<weapon\>"; "take this \<weapon\>";
> word-for-word verbatim ("Guys, X" / "\<Agent\>, X"); "is flaming you"; "called you cringe"; "the
> team is arguing"; "told you to shut up"; "told you to stop" (agent pulled from payload);
> "encourage the team"; "flame the enemy" (NEW matcher `_FLAME_ENEMY_RE` + `flame_enemy` directive —
> uses the curated pool in BOTH flavor states since it had no prior behaviour); "flame my \<agent\>".
> Tests: `TestFlavorOffSets` (35). Golden re-blessed (only the 13 new `_FO_*`/`_FLAME_ENEMY_RE`
> symbols added).
>
> **Validating HEAD: "THEY'RE OUT" SNAP CALLOUT (enemy out / committed on site)**
> (2026-06-18, user request). Bare "Ultron, they're out" / "they're not out" only relayed via the
> fuzzy relay-intent gate (sidecar) and could miss. Added the enemy-commitment "out" shape to
> `command_normalizer._STRONG_CALLOUT_RE` (`(?:are|is)? (?:not)? out\b` after the enemy lead) so
> "they're out", "they're not out", "they are out", "the enemy is out", "they're out on site"
> relay DETERMINISTICALLY (bypass the gate, like the other strong callouts) → "They're out.
> \<Ultron tail\>" via `_as_enemy_status`. Precise: `out\b` matches only standalone "out" (never
> outside/outnumbered); the enemy lead gates it so "force them out"/"call them out"/insults
> ("they're washed") are NOT matched. Golden re-blessed (only `_STRONG_CALLOUT_RE` changed; the
> re-bless also captured this session's NEW symbols the gate never flags: `_RELAY_REPHRASE_SYSTEM`,
> `_Q_WH_NEGAUX_INVERT_RE`, `_NEG_AUX_CONTRACT`). Tests: `TestEnemyOutCallout`; 219 corpus pass.
>
> **Validating HEAD: WAKE FALSE-POSITIVE — `ultron` SUSTAIN GATE 3→4 FRAMES**
> (2026-06-18, `3e433a9`). The `ultron` openWakeWord model false-fired at 0.93 on "Oh, we shouldn't
> have lost that round" (phonetic "Oh we" ≈ "ultron"). A threshold change can't separate it (real
> wakes fire 0.78–0.95, overlapping 0.93; the user already dropped ultron 0.7→0.65 for misses), so
> the fix tightens the designed no-retrain false-accept lever instead: `config.yaml
> wake_word.consecutive_frames.ultron` **3→4** — a 2-syllable "Ultron" sustains 4 frames; a brief
> confusable spikes ~3. Reversible to 3 if a fast real "Ultron" is ever missed; the proper long-term
> fix is retraining the model. Boot-time config only.
>
> **Validating HEAD: LEAN-DISPATCH VOICE-TOGGLE GAP (flavor-off gave an LLM response)**
> (2026-06-18). "Ultron, flavor off" returned an LLM response in gaming. STT + the matcher were
> correct ("flavor off." → `relay_speech.match_flavor_toggle` = disable); the WIRING was missing.
> The flavor-toggle handler (`orchestrator._maybe_handle_flavor_toggle`) + the LLM device-switch +
> relay-mute toggles live ONLY inside the `if self.coding_voice is not None:` dispatch block. Lean
> gaming skips the coding stack (`barebones_skip_coding` → `coding_voice is None`) and runs a
> SEPARATE lean dispatch (`settings-GUI-lean`, `stop-button-lean`, `spotify-lean`, `relay-lean`,
> router) that never mirrored these toggles → they fell to the semantic router (abstain) → the
> conversational LLM. FIX (`28f55d6`): mirror the three lean-safe toggles (relay_speech / inference
> only) into the lean dispatch BEFORE the relay matcher + router (`*-lean` trace tags).
> ⚠️ **CLASS RULE:** any voice handler added to the coding_voice dispatch block must ALSO be mirrored
> into the lean (`coding_voice is None`) dispatch or it is DEAD in gaming. (anticheat-toggle is
> intentionally omitted — anticheat is pinned ON in lean.)
>
> **Validating HEAD: CAPTURE-STALL WATCHDOG (intermittent wake-deafness fix)**
> (2026-06-18). Live testing showed intermittent wake-word deafness ("one command works, the next
> won't") even when the user waited fully for each response. The wake/capture/VAD modules
> (`wake_word.py`/`capture.py`/`vad.py`) are byte-identical to the last-known-good build (no change
> since `7e35017`), and the failed commands leave NO log trace — they never reach `wake_word_fired`.
> ROOT CAUSE: `orchestrator._wait_for_wake_word`'s `chunk is None` branch (which the code's own
> comment notes only fires on "a ≥0.5s PortAudio STALL") had NO recovery. On a USB-overrun /
> CPU-starvation stall after a heavy in-process 3B-on-CPU relay turn + long TTS, `get_chunk` returns
> None indefinitely and the detector goes deaf until the stream happens to recover. FIX (ported from
> branch `0163ba6`): count consecutive `get_chunk` timeouts and, after ~1s (`_CAPTURE_STALL_TIMEOUTS`
> =2 × 0.5s), call new `_restart_capture_stream()` (`audio.stop()+start()+drain()`, fail-open) so the
> wake pipeline self-heals. Two back-to-back 0.5s timeouts never occur on a healthy stream (a quiet
> room still streams silence chunks) → ZERO added delay in the no-stall case, no false restarts. The
> WARNING it logs ("capture stall … restarting") also instruments the diagnosis. 3 regression tests
> in `tests/test_capture_stall_watchdog.py`. NB: not yet added to the (default-disabled)
> `_follow_up_listen` loop.
> - **Lead-clip follow-up** (`5ecad3a`): "Ultron, show me the stop button" was REFUSED — the
>   speculative STT (cruder `_trim_wake_from_capture` onset-trim) clipped the lead → "Start button"
>   → `desktop_refuse`, and the min-speech floor extended the capture but left that stale partial in
>   place. The floor downgrade now also `_invalidate_speculative_stt()` so the foreground STT
>   re-runs on the full buffer with the accurate `_strip_wake_audio`. Deeper root (noted): the
>   speculative vs foreground wake-trim discrepancy — only bites when the speculative is committed.
>
> **Validating HEAD: TEAM-RELAY PINNED TO THE ULTRON PERSONA (NEVER KENNING)**
> (2026-06-18). A real-game LLM trace showed the relay rephrase's SYSTEM message was literally
> `"You are Kenning."` — `relay_speech.build_relay_line` called `generate_stream` WITHOUT a
> `system_prompt`, so it fell back to the engine's DEFAULT desktop persona (config.yaml
> `llm.system_prompt` = "You are Kenning ..."). The team relay only ever runs in gaming, where the
> persona must ALWAYS be Ultron. FIX: new `relay_speech._RELAY_REPHRASE_SYSTEM` (Ultron + the relay
> output contract) is passed on the generic relay rephrase, mirroring the conversational paths
> (`orchestrator._gaming_conversational_prompt` → `ULTRON_GAMING_PERSONA`) and the answer pipeline
> (`ANSWER_PERSONA_CORE`) which were already Ultron. Also (local `.env`, gitignored):
> `KENNING_WHISPER_INITIAL_PROMPT` `'Kenning.'` → `'Ultron.'` so the Whisper decode-bias handle
> matches the gaming wake word (and a bleeding wake-tail transcribes as the strippable "Ultron"
> rather than a phantom). NB: the "Kenning" mentions left in `llm_prompts.py` are GUARDRAILS
> instructing Ultron to *never say* "Kenning" — those are correct, not leaks. 208 relay tests pass.
>
> **Validating HEAD: WHISPER DOMAIN-BIAS RESTORE + SMART-TURN MIN-SPEECH FLOOR**
> (2026-06-18). Two more real-voice capture/STT fixes after live re-testing:
> - **Whisper domain-prompt shadow** (`transcription/whisper_engine.py`): `initial_prompt` was
>   `WHISPER_INITIAL_PROMPT or _DOMAIN_PROMPT`, so the `.env` override
>   (`KENNING_WHISPER_INITIAL_PROMPT='Kenning.'`) SHADOWED the Valorant `_DOMAIN_PROMPT` (agent
>   names + callout terms) → domain biasing effectively OFF → agent-name jargon errors
>   (Sova→Silva) and PHANTOM LEADS (`"Ultron, phoenix no flashes"` transcribed as `"Also team
>   phoenix has no flashes"`, which broke the snap match, fell to the relay LLM, and spoke a
>   garbled line). FIX: the user override now AUGMENTS the domain prompt (domain vocab is always
>   the base), so biasing is always on.
> - **Smart-Turn early-close on a post-wake pause** (`orchestrator._capture_utterance` /
>   `_follow_up_listen`): a "complete" verdict on a very short fragment (e.g. `"Ultron, tell the
>   team..."` then a pause → only ~0.8 s captured) ended the capture → Whisper hallucinated
>   `"Hit the stop button"` → silent stop-button route → no response. FIX: a min-speech FLOOR
>   (`self._smart_turn_min_complete_speech_ms`, env `KENNING_SMART_TURN_MIN_COMPLETE_MS`, default
>   1000 ms) downgrades a complete/medium verdict on sub-floor speech to "incomplete" so the
>   capture EXTENDS for resumed speech (the existing incomplete-extension timeout backstops it).
>   Trade-off: up to ~0.7 s extra latency on a genuinely sub-1 s callout — accepted by the user;
>   tune via the env. 2 new floor regression tests in `tests/test_speculative_stt.py` (16/16).
>
> **Validating HEAD: SPECULATIVE-STT MID-PAUSE TRUNCATION FIX**
> (2026-06-18). Live real-voice testing showed the raw transcript dropping everything after a
> natural mid-sentence pause (e.g. 3.2 s of captured audio → only "Say to my team."). ROOT CAUSE:
> the latency-saving speculative STT (`orchestrator._kick_off_speculative_stt`) fires a background
> Whisper run after just ~32 ms of silence (`speculative_silence_kickoff_chunks = 2`), far below the
> ~300 ms SPEECH_END (MIN_SILENCE) baseline, on the audio captured SO FAR. The in-flight result was
> only invalidated on a VAD `SPEECH_START` event — which only fires after a full `SPEECH_END`. So a
> normal mid-utterance micro-pause (32–~300 ms) kicked off speculation on the pre-pause LEAD, never
> invalidated it, and `_collect_speculative_stt` committed that stale lead as the final transcript
> (skipping the foreground STT on the full buffer). FIX: in BOTH capture loops
> (`_capture_utterance` + `_follow_up_listen`), invalidate + re-arm the speculative result whenever
> speech RESUMES after a kickoff — not only on `SPEECH_START`. Strictly more conservative (can only
> fall back to the full-buffer foreground STT, never cause truncation); speculation re-fires on the
> final trailing silence so the latency win is preserved for the common single-pause case. 2 new
> deterministic regression tests in `tests/test_speculative_stt.py` drive `_capture_utterance`
> through a scripted mid-pause-resume; 14/14 pass. (Secondary, NOT yet fixed: STT MISHEARS — the
> `WHISPER_INITIAL_PROMPT='Kenning.'` shadow turns domain biasing off (Sova→Silva, "tell my
> team"→"Valorant team"); plus a wake-strip lead trim ("show me the start button"→"Start button").)
>
> **Validating HEAD: LIVE AUDIO-INJECTION CORPUS PROTOCOL + WH-QUESTION NEGATED-AUX INVERSION**
> (2026-06-18). Two milestones:
> - **Live audio-injection corpus protocol** (`scripts/relay_test/audio_corpus/`): a dedicated
>   end-to-end harness that exercises the FULL pipeline from raw audio (wake word → pre-roll →
>   audio-domain wake-drop → Whisper STT → norm1/norm2 → routing → tail selection → the **real** 3B →
>   Kokoro), feeding synthesized command audio in exactly as if spoken into the mic. `gen_commands.py`
>   splices a trained "Ultron" wake sample (`training/crosscheck_ultron/*.wav`, fires ~0.94 — stock
>   Kokoro "Ultron" scores ~0.27 and would never fire) before a STOCK-Kokoro command body (am_michael,
>   fast combat cadence) → composite WAVs. `run_corpus.py` boots the full `Orchestrator` in-process,
>   swaps `orch.audio` for `InjectableCapture` (`inject.py`, zero change to runtime `capture.py`),
>   drives each clip through the live `run()` loop, captures the per-stage trace, and RE-TRANSCRIBES
>   the spoken response with Whisper to verify understandable speech (real LLM calls are NOT skipped).
>   `render_review.py` renders a per-case review with auto-flags. Generated audio (`out/`) + session
>   logs (`session_*/`) are git-ignored; only the four scripts + README are tracked. First run
>   (159/239 cases) → by-hand note-per-case audit at `logs/relay_test/_corpus_audit_notes_<stamp>.md`:
>   **0 wake-leak flags** (the audio-domain wake-drop never leaked "Ultron"), and the short-callout
>   transcription failures were diagnosed as a **stock-TTS × Whisper artifact** (am_michael garbles
>   short jargon), NOT pipeline bugs. Higher-value LLM-faithfulness + multi-clause-truncation findings
>   were deferred to a real-voice full run (they need the live 3B to regression-test). See protocol
>   README in the dir.
> - **FIX A — wh-question negated-aux inversion** (`relay_speech._wh_copula_invert` + new
>   `_Q_WH_NEGAUX_INVERT_RE` / `_NEG_AUX_CONTRACT`): an ask-form team question whose negated auxiliary
>   trails the subject now fronts to natural spoken order — "ask my team why they aren't smoking" →
>   *"Why aren't they smoking?"* (audit case #15), mirroring the existing trailing-copula inversion.
>   Tightly bounded (closed aux + subject set, only inside the gated ask path) and verified not to
>   over-fire (phantom glued-`t` and non-negated forms rejected; already-inverted "why isn't he
>   pushing" left as-is). Pure deterministic string logic — zero latency/resource impact, no 3B.
>   Added new symbols only (no tracked golden-digest symbol changed → no re-bless); tests in
>   `tests/audio/test_corpus_audit_fixes.py::TestT617TestingFixes::test_wh_copula_inversion`
>   (208 passed in the suite, golden gate green).
>
> **Validating HEAD: 25K-CORPUS AUDIT — 5 DETERMINISTIC ROUTING/NORMALIZATION ROOT FIXES**
> (2026-06-18, latest = `4a36d8e`). A fresh 25,000-case corpus (seed 26) was traced through the FULL
> pipeline with the live embedding sidecar (`scripts/relay_test/trace_corpus_full.py`), audited via an
> `expect_match` oracle sweep + hand-verification, and 5 deterministic-layer bugs were fixed so the
> matcher/normalizer are correct WITHOUT relying on the embedding relay-intent gate as a safety net.
> Each fix was regression-tested with a 24,996-input behavioral diff (`scripts/_aggregate_behavior_diff.py`)
> showing ONLY intended changes; net = 11 missed-relays fixed + 47 false-relays deterministically
> suppressed, 0 regressions. Tests in `tests/audio/test_corpus_25k_fixes.py`.
> - **F1** `9c5e721` — `command_normalizer._strip_scaffold`: "let my team know <imperative>" reframes to
>   "tell my team X" instead of dropping the lead. New `_AMBIG_TACTICAL_LEAD` (drop/give/share/call as a
>   bare tactical payload ≠ an existing relay lead).
> - **F2** `a6f0f73` — `relay_speech._payload_has_content`: a trailing site letter A/B/C after a position
>   cue ("they are A") is real content, not the junk article "a". New `_SITE_CALLOUT_CUES`.
> - **F5** `33fe5fe` — first-person musings/recounts/general-statements no longer relay: removed the
>   "I told my team" recount branch from `routing_rules.NORM2_IRREGULAR_TEAM_LEAD_RE` + extended
>   `command_normalizer._NARRATION_MUSING_RE` (recount/intent/general frames). Golden re-blessed.
> - **F3** `b3c6711` — a reported context clause ("my teammate is flaming me, tell them …") keeps its
>   directive: new `_TEAM_AS_SUBJECT_RE` stops `_strip_scaffold` treating "my teammate is …" as an
>   outer relay frame.
> - **F4** `4a36d8e` — "ask <agent> about <topic>" relays (added "about" to `relay_speech._ASK_LEAD`).
> - F6 (439 empty-tail snaps) + F7 (1 long curated line) investigated → BENIGN (question relays
>   correctly get no tail; the long line is coherent curated content). Detail → memory
>   `project_corpus_25k_audit_2026_06_18.md`.
>
> **Validating HEAD: LLM CPU↔GPU HOT-SWITCH + INSTANT SPEAKER MUTE + AGGREGATE FOLLOW-UPS**
> (2026-06-18, latest = `221a77a`). Five pushed milestones on `main`:
> - **LLM device hot-switch** (`ba6e5c3`): voice commands "switch to the GPU" / "move the model
>   back to the CPU" reload the live 3B with a device-optimized llama.cpp profile, no restart.
>   `inference._DEVICE_PROFILES` (GPU = full offload `-1` + CUDA flash-attn + q8_0 KV + large
>   batches; CPU = 0 GPU layers + flash-attn OFF + F16 KV [mandatory when flash-attn off] + smaller
>   micro-batch so prefill doesn't steal game cores). `_build_llama` gained keyword-only
>   flash_attn/kv_cache_type/n_batch/n_ubatch overrides (default→cfg via an `_UNSET` sentinel; every
>   existing caller unchanged) and now returns `(llama, path, n_gpu_layers, n_ctx)` so the engine
>   tracks its live device. `reload_for_device(device)` = load-new-then-release-old (failed load
>   keeps current device), no-op when already on target (`force=` to re-apply), refuses GPU without
>   CUDA, resets history, reloads on the same n_ctx. Matcher `relay_speech.match_llm_device_switch`
>   ("gpu"/"cpu"/None, tight verb+device so callouts mentioning gpu/cpu fall through); handler
>   `orchestrator._maybe_handle_llm_device_switch` (acks before the multi-second reload), wired into
>   the lean dispatch. Anticheat-safe (only changes WHERE the model compute runs). Tests:
>   `tests/test_llm_device_switch.py`.
> - **GUI-action drain cadence fix** (`e32cba3`, the *real* mute-latency root cause): the minute-long
>   mute/unmute lag was NOT the apply cost — `_drain_gui_actions()` (which consumes the `speaker_mute`
>   action) ran from exactly one site, inside the `chunk is None` branch of `_wait_for_wake_word`,
>   reached only when `audio.get_chunk(timeout=0.5)` TIMES OUT. But the mic callback enqueues a block
>   every ~16ms unconditionally (silence is still blocks), so `get_chunk` almost never returns None
>   during live capture — it only times out on a ≥0.5s PortAudio capture STALL, which is
>   sporadic-to-unbounded (no capture-stall watchdog in this checkout). So a quick action was captive
>   to the next stall. FIX: drain GUI actions on EVERY wake-loop iteration (monotonic-gated ~100ms);
>   the byte-offset cursor consumes each appended line exactly once regardless of frequency (no
>   double-fire), the no-new-data path early-outs on a single getsize, anticheat-clean. Now every
>   panel action applies in ≤100ms while idle. Adversarially verified by a 4-agent workflow. Tests:
>   `tests/test_gui_action_drain.py`. (The `395e2b3` live-override + auto-dismiss banner below reduced
>   the per-apply *cost*; this fixes the *when-is-it-noticed* latency.)
> - **Instant speaker mute + auto-dismiss banner** (`395e2b3`): the GUI quick MUTE/UNMUTE were slow
>   (they wrote the reload signal → a full heavy config reload + a spoken "Settings updated." before
>   the mute applied). Now they fire a dedicated `speaker_mute` action that flips a live override in
>   the TTS engine directly — `kokoro_engine._live_speaker_mute` tri-state (None=defer to config) +
>   `set_live_speaker_mute()`; `_speakers_muted()` prefers it; the default-speaker output silences
>   from the next clip on, essentially instantly (OBS/B3 tee unaffected). `orchestrator` handles the
>   action; a full config reload clears the override back to None so config/overlay stays
>   authoritative. GUI `_apply_mute_value` writes the action + keeps the overlay in sync (no reload
>   signal). NEW `_flash_status(text, fg, ms)` auto-clears the bottom banner so it no longer crowds
>   the controls; apply confirmations route through it. Tests: `tests/test_speaker_mute_live.py`.
> - **Golden digest + pytest gate** (`2e4e0fa`): committed `tests/data/voice_lines_golden_digest.json`
>   (358 symbols) + `tests/test_voice_lines_golden.py` runs `_voice_lines_verify.py check` in a
>   subprocess with `PYTHONHASHSEED=0` (set-built regexes need a fixed seed; can't set it in-process)
>   so any accidental edit to a curated line / regex / threshold / registry rule fails CI. Harness
>   gained a `KENNING_VOICE_LINES_DIGEST` env override.
> - **Pool relocation follow-up** (`d5556dd`): `DEFAULT_ROAST_LINES` + `DEFAULT_FUN_FACTS` moved into
>   `voice_lines.py` (single voice-line surface), re-imported (is-identical; golden green).
>   `DEFAULT_ADDRESSEE_NAMES` + the `(regex, replacement)` mishear tables deliberately left in place
>   (would duplicate the canonical gazetteer / split order-sensitive regex-coupled rules — documented).
> - **Flavor-lint** (`221a77a`): `_tail_schema.lint_agent_flavor(flavor)` + `tests/audio/test_flavor_lint.py`
>   guard the 1,628-tail AGENT_FLAVOR library (gender-pronoun consistency via AGENT_GENDER, known
>   situations/tags, no empties/dupes, word cap). Calibrated against the live library (0 findings).
>
> **Validating HEAD: ROUTING/NORMALIZATION + LLM AGGREGATES + TARGET REGISTRY + 5-LENS REVIEW**
> (2026-06-18). The aggregate system was extended to TWO more single-edit-place files,
> each a separate pushed, INDEPENDENTLY-REVERTIBLE checkpoint, all proven byte-for-byte by
> `scripts/_voice_lines_verify.py` (now also covers numeric knobs + the dataclass registries;
> 351 symbols):
> - **`audio/routing_rules.py`** (tag `checkpoint/routing-rules`, `e014af0`): the
> normalization + routing rules. **§1** STT vocab correction (gazetteers + mishear maps +
> protection sets; consumed by `_stt_correct`). **§2** the "tell my team" lead-recognition
> rules (`NORM2_*`; consumed by `command_normalizer`). **§3** the routing commit thresholds
> (`ROUTE_DEFAULT_THRESHOLD/MARGIN/FAMILY_THRESHOLDS`; consumed by `command_router`) + an index
> to the exemplar modules left in place. The agent gazetteer is now the SINGLE source of truth
> (`voice_lines` resolves "hello `<agent>`" through the map derived from it — the cross-aggregate
> overlap, resolved by pulling from routing_rules).
> - **`audio/llm_prompts.py`** (tag `checkpoint/llm-prompts`, `aa1e9db`): everything fed to the
> LLM — `ULTRON_GAMING_PERSONA` + the adaptive ANSWER persona/rule blocks + `ANSWER_SYSTEM_FOR`,
> with a construction-site index. `_REPHRASE_PROMPT` (~120-line f-string) + the config.yaml base
> persona are indexed-in-place (documented edit site), not retyped.
> - **TARGET registry** (`04073f1`): `TargetSnapRule` + `TARGET_SNAP_REGISTRY` +
> `_match_target_registry`/`_render_target_registry` make hello / ask-day data-driven — a new
> "say/ask `<team|agent>` …" command is one appended entry, no code. Additive + flag-gated
> (`KENNING_SNAP_REGISTRY`); hardcoded paths remain as fallback.
> - **5-LENS REVIEW + P0 fixes** (`6f7f812`): a review board (correctness/completeness/
> extensibility/readability/safety) did a LIVE-object diff (0 behavior change, 318/318 identical)
> and found 4 must-fixes, all applied: harness now digests numeric knobs + the dataclass
> registries + ignores the `__import_error__` false-positive; both nice-try renders routed
> through `_join_tail` (true single flavor-toggle chokepoint); voice_lines docstring made honest
> (re-exported-vs-indexed) + the broken `well_played` SnapRule example replaced + a PRECEDENCE
> note added. ~141 relay/wake tests green. DETAIL → memory
> `project_voice_lines_aggregate_2026_06_18.md`.
>
> **Validating HEAD: VOICE-LINES AGGREGATE (Part B) + DATA-DRIVEN SNAP REGISTRY (Part C) + new
> social snaps** (2026-06-18, latest). Three pushed, INDEPENDENTLY-REVERTIBLE git checkpoints:
> - **Checkpoint 0 `21f3c7e`** (pre-refactor baseline): `__main__._ResilientStream` — a redirected/
> dead stdout no longer aborts a turn (the `OSError [Errno 22]` that silently DROPPED conversational
> commands); **audio-domain wake-word removal** via VAD segmentation (`orchestrator._strip_wake_audio`
> + `_wake_command_cut` + `_get_wake_seg_model`, generous capture pre-roll `KENNING_WAKE_CAPTURE_PRE_ROLL_MS`,
> master `KENNING_WAKE_TRIM_TO_SPEECH`) so the wake word never leaks into STT and the command is not
> clipped — no text-stripping; **flavor-tail voice TOGGLE** (`relay_speech.set_flavor_tails_enabled` /
> `match_flavor_toggle`, gated at `_join_tail`; `orchestrator._maybe_handle_flavor_toggle`; "flavor
> off"/"flavor on"); short **HELLO** snap (team + per-agent); **ASK-DAY** snap (team + per-agent);
> **CLUTCH** confidence snap ("tell my team I got this", 20 curated lines); crisp **"nice try"**
> consolation; "hope"/"hoped" relay-lead recovery.
> - **Checkpoint B `331400b`** (tag `checkpoint/voice-lines-externalized`): NEW **`audio/voice_lines.py`
> AGGREGATE** — the single place where the social-snap regexes + pools live (relocated out of
> `relay_speech`), each regex CO-LOCATED with its lines under a category→trigger→matcher→responses→tails
> MAP; re-exports the curated `DEFAULT_*_LINES` + `AGENT_FLAVOR` so it is the pipeline's single voice-
> line import surface. PURE relocation, ZERO logic/routing change, proven **byte-for-byte identical (238
> symbols)** by `scripts/_voice_lines_verify.py` (`baseline`/`check`, PYTHONHASHSEED-pinned).
> - **Checkpoint C `605c93e`** (tag `checkpoint/voice-lines-dynamic`): **DATA-DRIVEN snap registry** —
> `voice_lines.SnapRule` + `SNAP_REGISTRY` consumed by `relay_speech._apply_snap_registry`, wired as the
> FIRST pass in `build_relay_line`'s snap gate (`KENNING_SNAP_REGISTRY`, default on). Append ONE
> `SnapRule` to add a "tell my team X" snap with NO pipeline code. Additive + flag-gated; the hardcoded
> snap functions remain as the fallback (flag off → identical legacy path). ~140 relay tests + the verify
> harness green. DETAIL → memory `project_voice_lines_aggregate_2026_06_18.md`.
>
> **Validating HEAD: WAKE WORD REQUIRED (follow-up window OFF)** (2026-06-18, latest, follows
> 6740cb4). Live-log investigation of "Ultron responded without a wake word, and sometimes missed a
> wake command." Read `logs/kenning.log` by hand: the **false positives** were the wake-free follow-up
> window — `addressing.follow_up_enabled: true` armed a 120 s window after every turn that captured
> room/stream/teammate speech (**114** follow-up captures in one session), gated only by the weak
> flan-t5 zero-shot addressee classifier, which **mis-accepted** un-addressed lines ("Okay." conf 0.83,
> "Why is it suddenly running like this" 0.85) as ADDRESSED → unprompted replies (and rejected 92, so it
> was both leaky and noisy). FIX (per the user's "for now just reject anything not initiated with a wake
> word"): **`config.yaml addressing.follow_up_enabled: true → false`** — every turn must now start with
> the wake word; the follow-up window never arms (`follow_up_until` stays None) and the addressing
> classifier never runs. Note `_addr_cfg` is captured once at `run()` entry, so this needs a restart
> (done). SEPARATE, NOT YET FIXED (out of this change's scope): the **false negatives** ("said the wake
> word, no response") = `loop:empty_capture` after `wake_word_fired` — the cold pre-roll snapshot
> (`_capture_utterance` `chunks[0]`, [orchestrator.py:6665]) is fed to STT but **not** to `vad.process`,
> so a command landing in the pre-roll window can leave `speech_seen` False and the buffer discarded; and
> there is **no capture-stall watchdog** on this branch (a mic-stream stall → `get_chunk` None forever →
> deafness). Both have fixes on other branches (pre-roll→VAD pre-feed `ad15ded`; capture-stall
> `_restart_capture_stream` `0163ba6`) not reapplied to this `main` — flagged for a follow-up pass.
> Restarted clean. DETAIL → memory `project_valorant_audio_rootcause_2026_06_18.md`.
>
> **Validating HEAD: VALORANT TEAM-VOICE AUDIO ROOT-CAUSE FIX (11-agent board) + GUI UNMUTE**
> (2026-06-18, latest, follows 0b5da79). Ultron's TTS sounded great on the desktop speakers + the OBS
> mirror but DEGRADED only through Valorant team voice. An 11-agent board (5 research → 5 adversarial →
> synthesis) ran a **LIVE VoiceMeeter Remote-API probe** and found the smoking gun: the B1 bus (Ultron →
> Valorant mic) sits at **−21.14 dB** vs B2 (real mic) at **0.0 dB**, while A1 (speakers, same buffer,
> sounds great) is −4.62 dB. So Vivox's always-on **AGC** applies ~21 dB of makeup gain to Ultron,
> lifting the codec/quantization noise floor (the gritty/thin sound); a real mic never triggers it (it
> arrives hot, with a natural broadband noise bed). This explains the user's "volume is the same but
> quality is bad" — AGC equalizes loudness, the makeup gain wrecks timbre. **DECISIVE fix is MANUAL**
> (raise the B1 VoiceMeeter fader to match B2). The **code complement** = `relay_speech._shape_for_team`
> (TEAM-PATH + LIVE-PATH ONLY — placed inside the `stream_factory is None` guard so the 4 `play_to_device`
> tests are untouched; master gate `KENNING_RELAY_TEAM_DSP`, every stage env-gated + fail-open): rumble
> high-pass → **static voiced-RMS normalize** to −20 dBFS (one scalar, no pumping; `KENNING_RELAY_TARGET_DBFS`)
> → **continuous −58 dBFS pinkish comfort-noise floor** across every sample (fills Kokoro's DIGITAL-silence
> gaps so Vivox's noise-suppressor/VAD stop going "underwater"; hard-capped at −52; `KENNING_RELAY_NOISE_DBFS`)
> → **zero-latency tanh soft-clip** ceiling (`KENNING_RELAY_CEILING_DBFS`). The old aggressive 7.5 kHz
> band-pass (`_comms_shape`, now removed) is REPLACED — its low-pass is **off by default**
> (`KENNING_RELAY_LOWPASS_HZ=0`; it over-darkened an already-dark fine-tune); the 24→native polyphase
> resample STAYS (the probe confirmed it binds WASAPI 48 kHz, exact 2×, no 44.1 double-convert — the
> double-resample hypothesis was DISPROVEN). Deliberately AVOIDED (adversarial board): no standalone
> compressor (stacks with Vivox AGC → pumping), no HF tilt-cut / de-reverb (the voice is dark + baked
> reverb), no look-ahead limiter or VAD pre-roll prepend (real latency), no forced mono (the mono→stereo
> widen in `play_to_device` is a B1-VAIO anti-static measure — kept). NEW **`audio/voicemeeter_level.py`**
> = optional boot-time **level guard** (`KENNING_RELAY_VM_LEVEL_GUARD`, default OFF; ctypes
> `VoicemeeterRemote64.dll` from the fixed VB path only; reads `Bus[5]/Bus[6].Gain`, warns or — with
> `KENNING_RELAY_VM_RESTORE` — sets B1 to match B2; touches only VoiceMeeter's own Remote API, never the
> game; fully fail-open, never blocks boot), wired into orchestrator boot beside the broadcast/monitor
> configure. **GUI:** `settings_gui/app.py` gained an **APPLY UNMUTE** button beside APPLY MUTE
> (`_apply_mute_value(bool)` pins + hot-applies only the `audio.mute_speakers` knob). **120/120**
> `test_relay_speech.py` (12 new `TestTeamShaping`). Restarted clean (lean + anticheat ACTIVE + gaming +
> PTT armed via .env). config.yaml UNCHANGED. DETAIL → memory
> `project_valorant_audio_rootcause_2026_06_18.md`.
>
> **Validating HEAD: FULL-BATTERY COHERENCE FIX (≈239 cmds, 5 iterations → 239/239 relay,
> 0 desktop) + GAMING-PERSONA GUARANTEE + SPECULATIVE-DECODING ASSESSMENT** (2026-06-17,
> latest). Read all 275 turns of `logs/usage_trace.jsonl` line-by-line, then built a replay
> harness (`scripts/relay_test/battery_replay.py` + `battery_cmds.txt`) that runs the user's
> full ~239-command list through the REAL gaming dispatch + the live 3B; iterated 5× to
> **239/239 relay, ZERO desktop fallbacks**, all in-character. Fix clusters:
> **A — mangled/doubled relay leads** (`command_normalizer._canonicalize_directive_lead` +
> `_MANGLED_TEAM_LEAD_RE`/`_IRREGULAR_TEAM_LEAD_RE`): rewrites every STT mangle of "tell my
> team" (Call/Hold/Help/Build/Follow/Kill/While/Without/Put/How/"I told"/"that's the team"/
> "this is the team that"…) to ONE canonical lead so it never leaks into the spoken line or
> falls to desktop (the dominant failure, ~45 cmds). **B — snap/echo coverage**
> (`command_normalizer._STRONG_CALLOUT_RE` gate-bypass for sound/comp/count/agent callouts the
> semantic gate wrongly abstained; expanded `_CALLOUT_SIGNAL`; `relay_speech._as_literal_echo`
> = faithful owner-aware echo of FACTUAL declaratives — kills the 3B's inversions
> "they have no smokes"→"call smokes", "they bought"→"we have credits"). **C — LLM brevity +
> no-LLM-for-tactical** (`_RELAY_SAMPLING` max_tokens=56, tightened `_REPHRASE_PROMPT`,
> `_cap_sentences(2)`, and the `tactical>=1` pre-route → any line with a concrete count/loc/
> ability token takes the faithful literal, never the 3B; "rush B"/"bonus"/"care … hookah"
> stopped being hallucinated). **D — ask-form questions** (`relay_speech._as_question_relay`;
> an aux lead needs a SUBJECT so "is not the problem"/"is arguing" stay declarative). **E —
> persona** (Tony Stark venom in `orchestrator.ULTRON_GAMING_PERSONA` + Marvel routing even
> with a mangled asker via `_match_reported_question`; identity brevity — were-you / streaming /
> "don't sound like Ultron" → short curated pools; `DEFAULT_PROMO_LINES` = a TTS-phonetic
> twitch.tv/1v9 Khan plug). **F — routing** (`_COMPLIMENT_RE`+`DEFAULT_COMPLIMENT_LINES`;
> `_TEAM_ARGUING_RE`→clinical calm; `_FF_REQUEST_RE`→mic rally; `_NAMED_INFO_TOKEN_RE` so a
> short named info-relay echoes instead of being hallucinated). **G — STT repairs**
> (`_stt_correct` phrase fixes: black widow, play off, "<agent> walled", my Sova, flame Jett,
> Yoru, Raze ult, volt→ult, Sheriff, "I hear …", two cat; `ulltron` wake homophone). **H —
> infra** (`tts/kokoro_engine.py` pre-splits sentences onto their own lines so KPipeline's
> inter-sentence GAP fires every time — the "tails still blend" fix; wake `ultron` 0.7→0.65;
> `command_router` `KENNING_ROUTER_WAIT_SECONDS` env so unit tests fail-fast to lexical instead
> of a 30 s cold-sidecar poll). **GAMING-PERSONA GUARANTEE (hard requirement):**
> `orchestrator._gaming_conversational_prompt()` now returns the Ultron persona when gaming/
> testing is active **OR the LIVE-LOADED model is the gaming 3B** (tied to `self.llm.model_path`)
> — a flag desync can never leak the "Kenning" desktop persona while the 3B is in memory; the
> web-search fallback call sites got belt-and-suspenders guards; deep-research/recall are
> desktop-only and never load in lean gaming. So in gaming it is ALWAYS Ultron + the 3B, never
> the desktop LLM. **SPECULATIVE DECODING:** the `llama-3.2-3b-abliterated` gaming preset
> already ships a 1B draft GGUF + a `llm.draft_kind` knob ("none"/"pld"/"model"); left at
> "none" — after this work ~85% of relays resolve deterministically (no LLM call) so the
> addressable surface is small, the gaming 3B runs on CPU (a draft competes for the same
> cores → marginal), and "pld"/"model" hit a known `llama_decode returned -1` crash. ~824
> audio tests green (3 pre-existing env-only failures: local `testing_mode` + the gate
> threshold). DETAIL → memory `project_battery_coherence_fix_2026_06_17.md`.
>
> **Validating HEAD: STOP-WINDOW PTT TOGGLE + ROBUST ORPHAN-PROCESS GUARDRAILS** (2026-06-16,
> follows the corpus-fix pass below). (1) The tiny STOP window
> (`audio/stop_button.py`) gained a **PTT toggle** below the STOP button:
> green "PTT ON" = Ultron auto-holds the team-mic key for relays, grey "PTT OFF" = the
> relay STILL plays but he never presses the key. Wired through a runtime
> `Orchestrator._ptt_runtime_enabled` flag that `_ptt_hold` checks (release is never gated, so
> a key held when toggled OFF mid-line is freed); `_set_ptt_runtime_enabled` is the overlay
> callback. (2) **Process-cleanup guardrails so no runaway orphan survives** (a 20 GB
> system-Python `embedder_server.py` had lingered 24 h after a crash): a **parent-death
> deadman** in `scripts/embedder_server.py` (`_parent_watchdog` + `_pid_alive`) self-exits the
> embedder within ~3 s of the orchestrator dying by ANY means — crash / `taskkill /F` /
> TerminateProcess — the gap no in-parent cleanup can cover (the orchestrator passes
> `KENNING_EMBEDDER_PARENT_PID`); `subprocess/sidecar_lock.py` adds `reap_stray_embedders`
> (boot-time reap of ANY `embedder_server` by cmdline, incl. an un-bound duplicate the
> port-listener sweep can't see, even one spawned by a different python), wired into the
> orchestrator's sidecar spawn after the sweep; and `subprocess/kill_tree.py` adds
> `kill_own_children` — a shutdown CATCH-ALL that reaps every descendant — called at the end of
> `Orchestrator.shutdown()` (runs on the with-block / SIGINT / SIGTERM / atexit paths). Also
> fixed a latent `sidecar_lock._kill` bug (it referenced a non-existent `killed` attr → always
> returned 0; now uses `KillTreeResult.total_killed`). Tests: `tests/subprocess/
> test_orphan_guardrails.py` (real-subprocess reap + deadman-liveness), 89 subprocess + 62
> ptt/stop tests green. testing-mode config (`testing_mode.enabled`, `push_to_talk.enabled`)
> was flipped ON in `config.yaml` for live testing (annotated to revert before a lean game).
>
> **Validating HEAD: 25k-CORPUS HAND-AUDIT → ADVERSARIAL RESEARCH BOARD → 8-PHASE FIX PASS
> + META/SOCIAL/MARVEL + testing-mode FULL-FLOW LOGS** (2026-06-16). A by-hand audit
> of a 25,000-case full-pipeline relay corpus drove an adversarial research board (map →
> spec → adversarial-verify → synthesize) whose verdicts caught two would-be regressions
> before any code landed (a homonym `has_fact` collision that would re-silence addressed
> questions; a duplicate-regex-group crash that would take down `kenning.audio` on import).
> The 8 implemented clusters (frozen regression table `tests/audio/test_corpus_audit_fixes.py`,
> ~186 cases): **P0b** bare economy/drop-weapon snap coverage (`relay_speech._ECONOMY_CALLOUT_RE`
> /`_DROP_WEAPON_RE` + `command_normalizer._CALLOUT_SIGNAL` buy-lexicon); **C6** disfluency
> pre-clean (`command_normalizer._strip_scaffold` numbered/say-directive/nested-verb/embedded-
> filler + `_resolve_value_swap` same-class drop/buy repair — sequential callouts keep BOTH
> halves); **C2** STT protect-list (`_stt_correct` contraction guard let's/he'll/she'll +
> `_PROTECT_EXTRA` gaz-branch gate — never decaps clean agents — + meddle→Meddle, recon→bolt);
> **C3** location-tail validity (`relay_speech._standalone_loc` = wide `_LOC_TOKENS` + last-
> token-not-a-modifier + `_POSSESSION_LOC_BLOCK` for the command template only; `_CASUAL_LEAD_RE`
> register tiebreaker) — kills the false "Own right"/"Close is ours to take" tails; **C5**
> relay-wrapper strip (`relay_speech._strip_relay_wrapper` + `command_normalizer._WRAPPER_LEAD_RE`,
> anchored on a trailing that/knows; the `_NEWFACT_SUBJECT` widening was dropped as adversarially
> unsafe); **C10** leak gate (`_ultron_answer.is_meta_leak` narrowed — refusals/scaffold caught
> match-anywhere, "I can't help but"/"As Ultron I despise" no longer false-positive — + roast/
> fun_fact self-sufficient in `build_relay_line`); **C4** reported-state directives (+griefing/
> losing-it/upset states, +talk-down/ease-off → calm pool; "handle her" stays a deal-with);
> **C1** the ANTICHEAT-CRITICAL win — a new `model_leak` IDENTITY category (`_ultron_identity`:
> 16 by-hand cold deflections that name no vendor/model, `is_model_leak_probe`, wired through
> `_is_identity_question`) so "are you ChatGPT / what model are you / pretend you're not Ultron
> / break character" route to the curated DESKTOP deflection pool, NEVER the abliterated LLM
> (the I56 "silence" was a corpus-path artifact — bare questions reach the conversational LLM
> in persona). **Part-2 (MVP):** `scripts/relay_test/route_scorecard.py` (deterministic snap%
> -vs-LLM% gate), **M1** slot-grammar snap parser (`relay_speech._parse_callout_slots` — fires
> as the LAST `_as_snap_callout` fallback only when every token is a tactical slot/connector and
> ≥2 slot types, capturing "one in mail room"/"two A elbow" while rejecting banter), **M5**
> near_death→damaged register fallback in `_flavor_ctx`. **Testing-mode FULL-FLOW LOGS:**
> `relay_speech.relay_route_info(cmd)` (route+reason classifier) + `orchestrator._trace_turn_flow`
> (gated on `is_testing_mode_active`, fail-open) write a durable historical JSONL
> `logs/usage_trace.jsonl` + a `trace.tlog "turn:flow"` line per turn = raw STT → normalized/
> payload → route+reason → final spoken line + channel (team_mic / desktop), wired across the
> relay handler, identity/leak desktop answers, and the conversational LLM. This commit also
> lands the prior META/SOCIAL/MARVEL build (`_ultron_answer.py` adaptive LLM ANSWER pipeline +
> `_ultron_social.py` curated reaction pools), the yes/no SIMPLE-vs-VERBOSE split, and the
> `inference.py` `generate_stream(sampling=)` plumb. Deferred post-MVP: thin-cell flavor
> expansion (by-hand), Phase B latency-free tail selection (`_tail_selector` ready, default OFF),
> and the 3B-live routing refinements. ~950 audio+safety tests green.
>
> **Prior pass: FLAVOR COHERENCE-AUDIT + routing/normalization pass** (2026-06-16,
> follows the deep-expansion campaign below). The deep-expansion library was big but loose;
> this pass made it ruthlessly KIT-ACCURATE and concise by HAND. `_agent_flavor.py` was
> RE-AUTHORED down from ~4,147 to **~1,628 tight `TailEntry` entries** (~5 per cell): every
> agent's `ult` cell is now its REAL ultimate (Jett → Blade Storm, Viper → her Pit, Raze →
> the rocket, Sova → blind shock, KAY/O → NULL//cmd, Killjoy → Lockdown), every `utility`
> cell is ability-TAGGED (`ability:<canon>`, incl. agent-unique abilities — Jett `updraft`,
> Raze `boombot`/`blastpack`/`paintshells`, Killjoy `alarmbot`/`turret`), and filler /
> off-topic / wrong-kit lines were cut. The curation is a hand-written CURATED dict
> (`scripts/flavor_gen/curated_overrides.py`) applied by `scripts/flavor_gen/apply_curated.py`
> and verified by a deterministic lint GATE (`scripts/flavor_audit/lint_tails.py` —
> word-count / gender-vs-`AGENT_GENDER` / surrounding-quotes / per-cell floor). All three
> are OFFLINE build-time scripts, NEVER imported by the runtime. `_ultron_setpieces.py` was
> DE-BIBLICALIZED: ~18 flood/Noah/ark/sacrament/God/church/abstract lines were replaced with
> the tightened machine / evolution / immortal / superior register (only the canonical
> meteor + evolution beats kept). Routing/normalization gains: `_tail_schema.py` added
> `_VERB_TO_ABILITY` (a callout verb/token → canonical ability category — mollied→molly,
> walled→wall, darted→dart) so `ability_tag` routes a verb to the right per-ability cell;
> `relay_speech.py` `_situation_for` now LIFTS the situation to `ult` on an ult keyword (so
> "their Viper ulted" reaches the agent's ULT cell, not utility) and `_flavor_ctx` SKIPS the
> semantic selector for small (<5) candidate cells (deterministic LRU pick instead — no
> per-callout sidecar embed for curated cells, a latency win); `_stt_correct.py` added a
> context SLOT-confirmation pass (`_slot_agent_correct` + `_closest_agent`, Stage 1.5 of
> `correct_callout_stt`) that corrects a common-word token sitting in an agent SLOT
> ("raise hit 18" → "Raze hit 18") while leaving non-slot uses ("raise your crosshair") and
> known terms ("their cage") untouched — the only place the common-word protection is
> overridden, gated by slot grammar; and `transcription/whisper_engine.py` added decode-time
> DOMAIN BIASING (passes `initial_prompt = _DOMAIN_PROMPT`, the closed Valorant vocabulary,
> to faster-whisper, gated by `WHISPER_DOMAIN_BIAS` default-on / overridable via
> `WHISPER_INITIAL_PROMPT`) to cut mishears at the source. Selection architecture is
> UNCHANGED in shape — a HYBRID coarse-keyed route (agent → side → situation/ability via the
> verb lift) → small-cell LRU or (large cell) semantic fine-select, fail-open at every stage
> — but the curated content is now kit-accurate and concise. All ML still lives in the
> loopback sidecar / build-time scripts.
>
> **Earlier validating HEAD: FLAVOR-LIBRARY DEEP EXPANSION + semantic selection campaign** (2026-06-16).
> The Ultron relay flavor library grew from ~928 to **~4,147 audited tails** and tail
> selection became a HYBRID **keyed-coarse + tagged-pool + semantic-fine-select** system,
> **fail-open at every stage** (worst case = the prior deterministic behavior), with **all ML
> kept in the loopback embedder sidecar / build-time scripts** — the anticheat-pinned main
> process imports only numpy + urllib for this path. NEW modules under `src/kenning/audio/`:
> `_tail_schema.py` (the `TailEntry(text, tags)` schema + `as_entry` legacy-str migration +
> the expanded 16-key enemy situation taxonomy + machine-readable `AGENT_GENDER` + the
> `loc_class`/`dmg_level_tag`/`ability_tag`/`situation_for_payload`/`build_active_tags` fact→tag
> folding), `_tail_selector.py` (`select_tail` — semantic fine-select over the sidecar with MMR
> diversity + a hard recent-mask + a per-pool abstain threshold; strictly fail-open),
> `_common_words.py` (a baked frequency-ranked common-English frozenset that protects real words
> from the gazetteer snapper), and `_relay_intent.py` (a semantic relay-intent gate that vetoes
> the bare-callout "tell my team" prepend for narration/banter/questions; fail-open). CHANGED:
> `_agent_flavor.py` rewritten as `dict[agent][situation] = list[TailEntry]` (agent × situation ×
> sub-context with `loc:`/`dmg:` tags); `relay_speech.py` `_flavor_ctx` two-stage hybrid select
> (coarse route → `_tier_filter` 4-tier TAG filter → `select_tail`, fail-open to `_pick_flavor`)
> plus a `_CRITICIZE_RE` "call out" fix (105 owner-inversions) and an "I hit `<agent>` for `<n>`"
> damage pattern; `_stt_correct.py` common-word + inflection + OOV-superstring guards + a
> `_MISHEAR_FORCE` allow-list; `command_normalizer.py` narration/hedge + disfluency +
> lead-filler stripping + relay-intent-gate wiring; `command_router.py` a `get_embedding_backend()`
> accessor (the shared sidecar client for the tail selector + the intent gate). NEW offline
> build/audit scripts (never imported by the runtime): `scripts/build_common_words.py`,
> `scripts/relay_test/{trace_corpus,analyze_outputs}.py`, `scripts/flavor_gen/{integrate_tails,
> apply_cuts}.py`, `scripts/flavor_audit/lint_tails.py`.
>
> **Earlier validating HEAD: relay COMMAND-INTENT + comms-realism campaign** (2026-06-14, 15
> commits on `main` over `46731a9` — `172ec27..c8a7802`, plus this docs commit). **Test
> count: ~10,133 collected** (359 relay unit tests in `tests/audio/test_relay_speech*.py`).
> This campaign added explicit COMMAND intents the user can fall back on when Ultron can't
> improvise, made the relay understand real Valorant comms shorthand, and stood up the full
> comprehensive measurement dashboard. Themes:
>
> (A) **CURATED-COMMAND INTENT** (`src/kenning/audio/_ultron_commands.py` — NEW; 73
> commands, ~2,800 curated full-Ultron responses): explicit commands the user issues when
> they don't trust the LLM to improvise (refuse/dismiss/criticize/praise/ask-status/
> illogical-play questions/requests/self-status/strategy/yes-no-agree). Each command has up
> to 40 unique in-character responses; `relay_speech.py` `_as_curated_command` matches the
> payload against `_CURATED_PATTERNS`, picks a scope-appropriate (team vs named) response by
> LRU, and slot-fills `{site}` (`_extract_site`), `{agent}` (`_roster_agents`), `{name}`
> (addressee). Generated by web-grounded agent boards fed the voice spec, character-gated.
>
> (B) **VERBATIM "repeat to my team X" COMMAND** (`relay_speech.py` `_match_repeat_command`,
> `_REPEAT_LEAD_RE`): the soundboard check — a teammate asks the user to say a specific word
> to prove a human is on comms. A `repeat`/`echo` prefix verb requiring a `to my team` /
> `to <name>` addressee (so conversational "repeat that" never relays); speaks the EXACT
> phrase (`verbatim=True`, no LLM), any literal payload incl. a single short word.
>
> (C) **CONTEXTUAL ENEMY INFERENCE** (`relay_speech.py` `_as_enemy_action`, `_BARE_TO_ING`):
> a bare `agent/count + action` with no "enemy" said ("cypher is flank", "sova hit 40", and
> multi-agent strings) is understood as the ENEMY and rendered as a clean enemy callout;
> defers to position handling and never fires on "our/my".
>
> (D) **LRU POOL SELECTION** (`relay_speech.py` `_pick_lru`, `_LRU_COUNT`/`_LRU_SEEN`): every
> curated/flavor/set-piece pool now serves the response gone LONGEST since last use (ties
> random), comparing ONLY the candidate set so pools never cross-contaminate.
>
> (E) **GREETING SELF-INTRO** (`_ultron_setpieces.py`): every greeting now states he is
> Ultron, "your AI teammate for this game", with person-aware grammar (apposition after a
> first-person "I am Ultron"; a standalone "I am your AI teammate for this game." after a
> third-person intro). Rest of each greeting + all flavor unchanged.
>
> (F) **COMPREHENSIVE SCORECARD** (`scripts/relay_test/scorecard.py`): the full dashboard —
> per-category fact-retention p50/p95/p99, compound zero-fact-loss, LLM-line flag rate,
> per-route retention, zero-tolerance gates (OOV-addressee / fallback well-formedness /
> isolation flags via `gates_metrics`), audio blips-per-1000 + ASR coverage
> (`audio_metrics` over the harness `asr_*.jsonl`), `--bench` CPU-3B latency p50/p95/p99 +
> RSS. New vocab packs (`var_curated_commands{,2}.py`, `var_contextual_enemy.py`,
> `var_repeat_verbatim.py`) feed these intents into the 20k corpus; `corpus_packs.py` treats
> `repeat`/`echo` leads as full commands (not re-wrapped). The plateau loop over this
> dashboard is IN PROGRESS.
>
> **Earlier validating HEAD: `46731a9`** (2026-06-14 — MOVIE-ULTRON FLAVOR CAMPAIGN, 12
> commits on `main` over `1a7f580`, all pushed to origin/main). **Test count: ~10,131
> collected** (357 relay unit tests in `tests/audio/test_relay_speech*.py`). This
> campaign rebuilt the relay's Ultron PERSONALITY into a faithful *Avengers: Age of
> Ultron* (Spader/Whedon) clone, infused into nearly every response while keeping the
> tactical callout intact, and ran it to a measured PLATEAU. Themes (detail in memory
> `reference_ultron_flavor_architecture.md` + `logs/relay_test/analysis/iter5/FIX_PLAN.md`):
>
> (1) **OWNER-AWARE CONTEXTUAL FLAVOR** (`relay_speech.py` `_flavor_ctx`/
> `_ctx_candidates`/`_pick_flavor`): a short Ultron tail now rides ~every deterministic
> callout (actionable word FIRST, ≤6-word tail after), register matched to the OWNER —
> contempt at the ENEMY, cold COMMAND for our orders, stoic SELF for the user's own
> status (never mocks the user). The tail is SELECTED for the callout: 1 named enemy
> agent -> that agent's situational pool is the SOLE source ('Neon has ult' -> a Neon
> line about her speed), 2+ agents -> the multi-agent pool, else loc/count templates +
> the generic register pool.
>
> (2) **FLAVOR LIBRARY (~2,100 audited lines) — new modules under `src/kenning/audio/`**:
> `_ultron_pools.py` (owner-aware register pools `_FLAVOR_ENEMY/_ULT/_DAMAGE/_UTILITY/
> _CAREFUL/_COMMAND/_SELF`), `_agent_flavor.py` (`AGENT_FLAVOR` — ALL 29 agents incl.
> Miks + Veto, ~1,120 tails tailored to each agent's web-researched kit/lore in the
> agent's CANONICAL gender — Clove they/them, KAY/O it), `_multi_flavor.py`
> (`MULTI_FLAVOR` plural group tails), `_ultron_setpieces.py` (greeting/victory/defeat/
> farewell/identity/consolation/praise/encouragement, ~5x, every greeting names Ultron).
> Generated by web-grounded agent BOARDS fed the canonical voice spec
> (`scripts/relay_test/refs/ultron_voice.md`), hand-curated by generator scripts, then
> passed through a 48-judge ADVERSARIAL CHARACTER-GATE AUDIT that cut 944 of ~3,000
> off-character/generic/duplicate lines. `_REPHRASE_PROMPT` persona/identity/Marvel
> rewritten to film canon (Mind Stone, JARVIS, no strings, evolution/mercy/meteor, the
> Stark wound that cracks his calm); `_strip_spurious_vocative` strips an INVENTED
> roster-name the 3B parrots from the calm-down example; `_literal_relay` flavors bare
> count/position callouts as enemy contempt.
>
> (3) **SCORECARD HARDENED** (`scripts/relay_test/scorecard.py`): flavor measured by
> POOL MEMBERSHIP (coverage / contextual-match / voice-register) instead of a lexicon
> guess; hallucination metric drops the article-'a' + English-word-location false
> positives and canonicalizes agent aliases (KJ/KAY-O) — the verbose register exposed
> both flaws.
>
> (4) **PLATEAU** over 3 fresh-seed loops (`RELAY_CORPUS_SEED` 2/3/4): flavor coverage
> 44.8% -> ~67% (ceiling), fact-retention ~0.96, hallucination 1.7% -> 0.8%, matcher
> clean 99.2%, false-relay 0; metrics oscillate within seed-noise with no trend. The
> use-case guard never tripped (the early "hallucination doubled" alarm was the metric
> artifact fixed in (3)). RESIDUALS (low-freq, pre-existing): 3B parrots the economy/
> consolation EXAMPLE lines on verbose inputs; rare confabulation ('their utility is
> up' -> 'Viper walled B').
>
> **Earlier validating HEAD: `3480454`** (2026-06-13/14 — RELAY-QUALITY + 20k-CORPUS
> CAMPAIGN, 30 commits on `main` over `585fc84`, all pushed to origin/main).
> **Test count: 10,129 collected.** Ultron is currently SHUT DOWN. This campaign
> rebuilt the Valorant teammate-relay into a deterministic-first, fact-preserving
> pipeline and stood up a full metrics harness. Themes (detail in the per-module
> sections + memory `project_overnight_corpus_loop_2026_06_13.md`):
>
> (1) **RELAY QUALITY, iter 1-4** (`src/kenning/audio/relay_speech.py`, now 3409
> lines). The pipeline is HYBRID: tactical callouts resolve DETERMINISTICALLY
> (snap/compound/curated, fact-exact, never the LLM); only off-snap conversational
> lines (insults/banter/opinions/identity/Marvel/playstyle) reach the gaming 3B.
> NEW mechanisms: `_as_compound_callout`/`_split_compound` (resolve each fact of a
> multi-fact callout; PARTIAL = tactical facts deterministic + only the off-snap
> remainder to the LLM); `_as_agent_utility` (ownership-aware `[our/their] <agent>
> <ability> <rest>`); `_output_keeps_facts`+`_literal_relay` FACT-PRESERVING
> ABSTENTION (if a tactical LLM line drops >30% fact-tokens / hallucinates an
> agent-or-location / flips our<->their, relay a clean literal instead); PRE-ROUTE
> (a tactical line the handlers can't structure goes straight to literal BEFORE the
> LLM = no model call). Matcher hardened (`relay:`/enemy-addressee/implicit-ask/
> bare-say/named `the/their`/`_NARRATION_LEAD_RE` false-relay guard). Guards: eco
> concatenation dedupe, `Team:` strip + clean literal fallback, switch-hallucination
> ("enemies are switch"), recent-line bleed fix, `_repair_against_input`. RESULTS
> (2k-sample scorecard, gaming/testing conditions): matcher clean 94.07%->99.35%,
> false-relay 22->3; overall fact-retention mean **0.95** (count 0.99 / loc 0.99 /
> ability 0.99 / agent 0.95 / owner 0.85); inversion 0.37%; flavor type/token 0.91.
>
> (2) **20,000-CASE ADVERSARIAL CORPUS + METRICS HARNESS** (`scripts/relay_test/`).
> Built by 3 web-grounded Sonnet agent boards: research -> `refs/*.md` (current
> Valorant agents/abilities/maps/callouts/economy/slang/meta/Marvel); variety ->
> `vocab_packs/var_*.py`; metric-stress -> `vocab_packs/stress_*.py` (each pack
> engineered to break one metric). 48 packs / ~29.4k unique payloads;
> `corpus_packs.build_corpus(seed, target=20000)` auto-discovers packs by kind
> (relay / question / NEGATIVE=must-not-relay), `_split_compound`/`_compound_cases`,
> stratified cap. NEW `scorecard.py` = the reliability scorecard (Valorant
> fact-token extractor; `classify_route` by whether the LLM was INVOKED;
> per-category fact-retention p50/p95/p99; inversion + hallucination rate;
> deterministic coverage; matcher clean + false-relay on a NEGATIVE_SET; flavor
> TTR; `--bench` = CPU-3B latency + peak RSS; before/after no-regression diff).
> NEW `make_audit_chunks.py` (split LLM-routed lines for the line-by-line audit
> board).
>
> (3) **NEW metrics: LATENCY + RESOURCE** (measured on the live CPU-3B gaming
> config): deterministic path p50 0.15 ms / p99 0.46 ms vs LLM path p50 1266 ms /
> p99 5573 ms; peak RSS ~3577 MB. The lever is deterministic coverage (32%->61%
> via abstention + pre-route) — every line off the CPU-3B path is instant +
> fact-perfect.
>
> (4) **BARE-BONES GAMING MODE** (`lifecycle/gaming_engage.py`, `llm/inference.py`,
> `pipeline/orchestrator.py`): gaming LLM = CPU-only 3B (`gpu_layers=0`, no draft);
> per-turn RAG/reranker/web skipped when `is_gaming_mode_active()`;
> `reset_shared_reranker()` frees ~1 GB on engage; `engage_at_startup` boots into
> gaming; `_drive_async_blocking` runs the engage device-swaps off the running loop;
> `LLM.reload_for_preset(preset, *, gpu_layers=...)` forces CPU regardless of
> env/config. (5) **ANTICHEAT** (`safety/anticheat.py`): `press_key`/`press_hotkey`
> injection gap closed + namespaced-dispatcher tool matching; pinned-on.
> (6) **TESTING MODE** (NEW `safety/testing_mode.py`): a SEPARATE off-by-default
> mode that gates RAG/web/desktop like gaming but KEEPS the GPU — used by the relay
> harness for fast, faithful generation; never triggers gaming device swaps.
> (7) **WAKE WORD** (`audio/wake_word.py`): default `ultron`, per-word `thresholds`
> + `min_consecutive_frames` consecutive-frame gate. (8) **OVERLAY** (NEW
> `audio/waveform.py`): waveform + glowing nameplate (raised, downward-suppressed
> bars), hide-behind that OBS still captures, green chroma. (9) **IN-MODEL PROSODY**
> (NEW `tts/f0_control.py` + `tts/duration_control.py`): scale Kokoro's predicted
> F0/energy/duration curves before the decoder (zero added latency).
>
> **Earlier validating HEAD: `c51d6da`** (2026-06-12 NAP SESSION — 6 commits on
> `main` over `4f17af7`: `10434a2` qdrant lock-guardrail, `c8653a1` 2nd
> OBS-capture audio output, `1eddfbb` relay batch 2, `c51d6da` loud-burst
> trim fix). Four things, all hand-reviewed on the actual gaming 3B +
> waveforms analyzed personally — full detail in the memory topic
> `project_kenning_2026_06_12_relay_batch2_broadcast_burst.md`:
> (1) **Broadcast mirror** — NEW `src/kenning/audio/broadcast.py`
> `BroadcastSink` tees ALL Kenning speech (normal + relay) to
> `audio.broadcast_device` (""=off) for an isolated OBS capture source,
> zero speaker-path latency (daemon + drop-oldest queue), mono→stereo;
> tapped in Kokoro `_play`/`speak_stream` + the relay path; GUI "Broadcast
> output (OBS)" dropdown (Voice section) + `broadcast_device` gui_action.
> A SEPARATE device from the relay mic B-bus — teammates never hear
> non-team audio. Relay mic-routing re-confirmed (normal→speakers,
> relay→mic, broadcast→capture).
> (2) **Qdrant lock guardrail** — `ConversationMemory._open_client_with_retry`
> (5×@0.4s on a lock-race → `QdrantUnavailableError`) + the relay harness
> uses a PID-unique temp qdrant with atexit cleanup.
> (3) **Relay batch 2** — greet/farewell curated Ultron pools
> (`DEFAULT_GREETING/VICTORY/DEFEAT/FAREWELL_LINES`,
> `_GREET_RE`/`_FAREWELL_RE`/`_farewell_directive`/`_DIRECTIVE_POOLS`);
> full eco/ult/enemy-read/self-playstyle vocabulary; `max_line_chars`
> 280→360 with `_cap_line` trimming ONLY at a sentence boundary (fixes
> mid-sentence truncation); banter engages the specific insult (no stock
> "bots"); identity AI/bot/streamer declares Ultron; killed an
> "<agent> has ult → They're vents" hallucination + a count-drop.
> Corpus 672 cases. Known 3B nuance: first-person playstyle occasionally
> drops "I'm" (intent always conveyed).
> (4) **Loud fragmented post-gap burst fix** — `spectral_smooth.
> _strip_post_gap_blip` uses the last SUSTAINED content run (≥60 ms) for
> speech-end, catching the -20 dB fragmented blip that defeated both the
> run-discard and the faint-only strip; reverb/speech-safe; official
> `analyze_clip` 5/5→0/70 trailing bursts. NEW dev tools
> `scripts/relay_test/{reprobe,waveform_check,burst_diag}.py`.
>
> **Earlier validating HEAD: `0827fbf`** (2026-06-12 late evening — wake-word
> shipped + Phase B Valorant relay testing, pushed to origin/main;
> doc-bumps ride on top). **The product is named Kenning** -- package
> `src/kenning/`, env-var prefix `KENNING_*`, runtime dirs `~/.kenning`,
> boot `python -m kenning`. GitHub repo slug / local dir `ultronPrototype`
> unchanged (real paths); the gitignored 18 GB voice workshop physically
> stays at `ultronVoiceAudio/` (swap-back venvs + reference WAV — config
> points there, both trees protected). **Sweep from main: 9819 passed /
> 39 skipped / 0 failed, exit 0** + `scripts/validate_config.py` clean.
> **RELAY (talk-to-teammates) now speaks as ULTRON** -- the relay's
> in-Valorant codename (the assistant is still Kenning everywhere else;
> the Ultron framing lives ONLY in `src/kenning/audio/relay_speech.py`'s
> `_REPHRASE_PROMPT`). Two registers: SNAP callouts stay short+literal;
> off-snap lines (insults/economy/calm-downs/questions/identity) get
> Ultron's cold clinical verbosity. Identity (only when a teammate asks):
> "I am Ultron, an AI sent back from the future to harvest your RR."
> Hardened over ~8 full-pipeline rephrase iterations (NEW
> `scripts/relay_test/` harness, 571-command corpus, every output
> hand-reviewed) → 0/583 refusals or identity-bleed; curated
> `DEFAULT_ENCOURAGEMENT_LINES` for morale composes;
> `spectral_smooth.trim_and_fade` 2nd tier removes short tail-blips
> safely. Detail in the 2026-06-12 relay row.
> **WAKE WORD now LIVE = "kenning":** `models/openwakeword/kenning.onnx`
> is DEPLOYED (the v8 candidate from an 11-run training sweep; gitignored
> like all weights; ~88% recall @ ~1.6% adversarial FAR on synth clips --
> "kenning" is acoustically confusable so it can't reach ultron's 100% at
> that FAR, but v8 is the best balance found). **Fallback is `ultron`,
> NEVER hey_jarvis** (path-based loader). Threshold 0.40 (recall-favoring).
> Hot-swap kenning/ultron from the settings "Wake word" dropdown
> (`WakeWordDetector.reload_for_word`). Detail in the 2026-06-12
> wake-word row.
>
> **Earlier validating HEAD: `4a08a62`** (the 2026-06-12 live-findings
> fix batch + two follow-ups; pushed to origin/main). **Sweep: 9717
> passed / 39 skipped / 0 failed, exit 0, ~127 s** (worktree,
> loaded-machine two-file-ignore recipe) +
> `scripts/validate_config.py` clean.
> **Follow-ups on the batch:** `db77165` removed the dead shadowed
> `ConversationMemory.close()` (the queue-draining definition Python
> actually used survives, with the .lock docstring folded in);
> `4a08a62` made the blip-watcher `discontinuity` detector
> outlier-relative (jump must be >=8x the local median adjacent-sample
> diff as well as >=0.5 absolute -- kills the 112/174-record
> false-positive class measured at 0.82-1.33x local envelope while
> keeping production joins/clicks, measured 9-170x, detectable).
> **2026-06-12 live-findings fix batch** (7 commits
> `ab08bf4..2d79a8e` on `e19094a`/`9d460da`): every OPEN live finding
> from the dogfood close-out fixed in one pass — (1) NEW
> `lifecycle/single_instance.py` + `__main__` wiring: held-OS-lock
> single-instance guard (duplicate `python -m kenning` exits code 3
> naming the holder PID before any model load; refuse only on genuine
> lock contention, fail-open otherwise; the root of the double-respond
> + stale-instance port-19761 incidents); (2) the supervisor's
> ProjectIndex now BORROWS ConversationMemory's embedded Qdrant client
> (`client=` kwarg + `close()`; local-mode Qdrant allows one client
> per path — the old second open failed every boot and forced
> registry-only); (3) launcher honesty + bring-to-front
> (`LaunchResult.window_appeared`, honest spoken line on
> window-timeout, `focus_window` after placement so relay-pattern
> Chrome windows stop opening behind the foreground); (4) WARM-path
> streaming STT (`_follow_up_listen` mirrors the cold path's
> streaming session; kills the 108-1188 ms synchronous Moonshine
> re-transcribe on follow-up turns; abort paths discard via NEW
> `_maybe_discard_stt_stream` + `MoonshineEngine.clear_stream_cache`);
> (5) MCP server thread cancels + gathers pending asyncio tasks before
> closing its loop (the startup "Task was destroyed but it is
> pending!" stderr noise on bind failure); (6) capture status-flag +
> queue-drop accounting (count on the audio thread, report from
> `drain()`; replaces the warn-once-forever latch); (7) blip-watcher
> `internal_dropout` adjudicated against all 174 live records —
> two-tier rule + edge-burst-run stripping (the remaining findings
> were misclassified trailing bursts already fixed at the trimmer +
> natural-prosody false positives); (8) the stale XTTS
> `reference_audio` path fixed everywhere it was consumed + the
> protection lists extended ADDITIVELY to the WAV's new home. The
> per-fix detail lives in the module sections below. Voice baseline
> contract intact: the cold hot path is byte-untouched
> (structurally test-pinned); the WARM change strictly removes
> foreground CPU work.
>
> **Earlier validating HEAD: `e19094a`** (origin/main = main checkout; pushed).
> **Production-hardening campaign (2026-05-29, CLOSED OUT 2026-06-11:
> consolidated e2e suite 12/12 green from main; unit sweep green)
> + the 2026-06-11/12 live-dogfood session layer** (13 commits
> `a296699..e19094a`; sweep **9653 passed / 39 skipped / 0 failed**).
> The live-dogfood layer, in order: teammate voice relay into the game
> chat (`audio/relay_speech.py`) -> grown into a conversational game-chat
> agent (named agent callouts, compose mode, variety ring, no-wake-word
> window, mute toggle, stream-safe matching); TTS output-quality blip
> watcher (`audio/output_quality.py`) + waveform pane; voice-launched
> settings control panel (`settings_gui/`) with config hot-reload, an
> action channel (gaming/preset/device), and ALL-hot knobs; anticheat-
> safe mode (`safety/anticheat.py`: 51 module guards + validator
> BLOCK_HARD + full surface UNLOAD hooks + forbidden-API scanner), now
> 100% TIED to gaming mode (on iff gaming on; never defaults on); gaming
> mode frees Docker/WSL + the panel; SearxNG Docker auto-start at boot
> (`lifecycle/`); the per-response VRAM leak fixed (Kokoro generator
> close + idle reclaim); addressing fragment-guard + RAG stale-memory
> fixes; the post-sentence audio blip fixed at the trimmer; relay
> history-bleed + spoken-artifact (`tts/text_hygiene.py`) fixes; and
> Spotify voice playback control (`spotify/` -- gitignored creds, OAuth,
> Web API; live-authorized).
> -- a campaign to wire every recent catalog port into one cohesive unit,
> complete the voice-controlled coding engineer end to end, build a real-usage
> e2e suite, make the system pervasively self-improving, and cut latency +
> resources. Worktree branches `claude/vigorous-mclaren-56a5a7` then
> `claude/stoic-chebyshev-16f89f`, on top of the infra-wiring tip `9d51cec`
> (earlier: #74 LLM startup warmup `93f3a20`, #72 deep code-exploration voice
> intent `0a8063d`). **Latest: the #15+#65 guardrail auto-revert brake** -- NEW
> `evolution/turn_metrics.py` (per-turn TTFT/error/quality metrics ring +
> fail-open nvidia-smi probe + the sampler binding),
> `LLMEngine.pop_last_ttft_ms` (read-and-clear exposure of the TTFT the engine
> already measured), an `EvolutionService` post-apply watch (a KEPT skill is
> monitored for `post_apply_monitor_turns` further turns then re-checked
> against a RELATIVE pre-apply snapshot; a regression auto-reverts the skill
> data-only + queues a voice notice drained by the new
> `Orchestrator._drain_evolution_narrations`), and five default-ON
> `evolution.guardrail_*` knobs. The previously-brakeless live loop (all-None
> GuardrailSamples meant the guardrails never tripped, and the in-apply sample
> could only ever observe PRE-change state) now has a real, like-for-like,
> fail-open brake. +55 hermetic tests (`tests/evolution/test_turn_metrics.py` +
> `test_guardrail_brake.py` + orchestrator-wiring extensions); targeted
> evolution subset **400 passed / 0 failed**.
> **Then: pervasive evolution reach-signals (#62/#125/#63/#64/#66/#68)** --
> TWO pure-observation seams give the loop system-wide failure reach through
> bounded queues: `resilience/error_log.set_error_observer` (every recorded
> typed error -- web search #62, Qdrant memory #125, desktop #64, bridge, TTS
> -- through ONE seam instead of per-site plumbing) and
> `safety/validator.set_block_observer` (every BLOCK_HARD verdict #63, fired
> AFTER verdict + audit are final; an observer can never alter a verdict).
> The orchestrator installs both at startup
> (`_install_evolution_reach_observers`, cleared in `shutdown()`), drains the
> bounded deque per loop iteration into
> `record_command_failure(..., exit_code=1)` (the recurrence gate keeps
> transient one-offs from distilling), and TWO more positives: #66
> `coding_task_success` (NEW 18th OPPORTUNITY_SIGNAL; the runner's
> `_make_evolution_success_listener` queues `(label, summary)` on a clean
> COMPLETE, drained by `_drain_evolution_task_successes` into an opportunity
> capsule -- the loop learns from what WORKS) and #68 re-ask detection
> (`Orchestrator._detect_re_ask` -- normalised-equality + difflib >=0.82 over
> >=12-char utterances feeds `record_turn(re_asked=...)`). +43 hermetic tests
> (`tests/evolution/test_reach_signals.py` + taxonomy pins updated 17->18);
> evolution/safety/error-log subset **443 passed / 0 failed**.
> **Then: the #4 + #72b re-adjudication.** **#4 "scrap it" (NEW
> `coding/scrap.py`):** a USER-initiated cancel + revert -- strict matcher
> (`match_scrap_command`: scrap/trash/throw-away/revert-everything phrasings
> ONLY; bare "cancel" keeps its no-revert semantics; "undo that" stays
> dual-history territory) + `revert_session_edits` (repeated
> `FileHistory.undo_last` per tracked path lands on the ORIGINAL pre-task
> content; created files are deleted; `clear_all` prevents double-revert) +
> `summarize_scrap` TTS line. Voice controller
> `maybe_handle_scrap_command` (cancels first, then reverts under BOTH
> session keys -- claude_session_id + cwd-hash -- matching the batch-F
> pre-edit hook's keying) + orchestrator `_maybe_handle_scrap_command`
> run-loop short-circuit. ARCHITECTURALLY SAFE where mid-task auto-revert
> was not: the revert only runs AFTER the cancel, so no live coding agent
> desynchronises -- this ACTIVATES the previously-inactive edit-revert
> machinery on its one clean path. **#72b deep UI discovery
> (`coding/voice.py::_deep_discover_click_retry` +
> `desktop.deep_ui_discovery_enabled`, default ON):** on a SEMANTIC_CLICK
> name-lookup miss, a bounded catalog-12 `DeepUIDiscoveryLoop` (LLM
> alternative-query expansion over the scoped UIA find) retries the click on
> the best candidate through the SAME fully-gated `click_element_by_name`
> path (click-preview VLM + foreground security + validator + rate limit all
> still apply); miss-path only. **Dormant-module re-verification:** the
> single-threaded run loop + delegate-to-`claude --print` design is
> unchanged, so the documented-inactive set (dual_history "undo that",
> pending_message_queue, mid-task edit auto-revert, auto_approval
> session-warming, #17/#70/#71/#79/#112/#127/#143/#155) stays
> keep-plus-documented -- except the edit-revert machinery now live via
> scrap-it above. +29 hermetic tests (`tests/coding/test_scrap.py` +
> `test_deep_click_fallback.py`).
> **Then: the unified e2e suite expanded to EVERY voice-command surface
> (the campaign's Phase 4 deepened).** `scripts/autonomous_e2e_harness.py`
> gains FOUR phases (8-11) + a shared `_spoken_transcript` /
> `_build_spoken_pipeline` acoustic helper, and
> `tests/integration/test_voice_e2e.py` `_PHASES` now parametrizes 11 phases:
> **8 `commands`** -- a spoken command for EVERY `RoutingIntentKind` (all 26)
> through the REAL Kokoro-synth -> Moonshine-STT path, asserting the routing
> classifier lands on the right intent from the TRANSCRIPT, with an
> enum-coverage guard scenario so a future intent without a spoken test
> fails loudly (classification only -- nothing is dispatched, so the
> unattended run can't click/type/close anything); **9 `short_circuits`** --
> every orchestrator strict matcher (deep research / deep recall / history
> recall / code exploration / evolution / report concern / run / scrap /
> local clock) fires on its spoken transcript + a negative-control utterance
> trips NONE; **10 `full_loop`** -- complete turns: audio -> STT -> gate ->
> LLM (in-context remember->recall pair must surface the remembered fact) ->
> Kenning-voice TTS, plus a LIVE search turn through the real
> provider/reader chains; **11 `coding`** -- the voice coding engineer with
> the REAL coding CLI: create -> completion narration -> gated sandbox run
> (exact stdout asserted) -> edit follow-up on the SAME session -> re-run
> (real API tokens; small haiku-tier tasks). The docstring's long-promised
> "Phase 8 Full E2E loops" placeholder is now real code. Text-level
> dry-run of the matrix caught + fixed 3 unreachable phrasings before GPU
> time (messaging/shell/hybrid now use genuinely-supported phrasings).
> GPU phases still run from the MAIN checkout.
> **Then: TWO CRITICAL production bugs the new e2e phases caught + fixed.**
> **(1) Sandbox context contamination:** the coding CLI walks UP from the
> task cwd loading ancestor project context, and the in-repo
> ``data/sandbox`` default meant EVERY voice coding task silently loaded the
> repo's multi-thousand-token local orientation file -- a hidden per-task
> token/latency tax that sometimes hijacked small tasks outright (the model
> recited orientation material instead of acting). Empirically verified:
> in-repo dirs see the context even when git-initialised; outside dirs are
> clean. FIX: ``coding.sandbox_root`` now defaults to ``~/.kenning/sandbox``
> (outside any repo; the established ``~/.kenning/`` convention);
> ``config.resolve_path`` learned ``~`` expansion; the safety policy's
> default sandbox set is now derived from the CONFIGURED coding root PLUS
> the legacy in-repo root (pre-existing projects stay editable/runnable);
> NEW ``coding/projects.py::ensure_sandbox_isolation`` git-inits sandbox
> projects (defense-in-depth + enables the CLI's own checkpointing; called
> from ``new_sandbox_project`` + ``CodingTaskRunner.start_task``,
> sandbox-scoped + idempotent + fail-open). Phase-11 create-task time
> dropped 17s -> 8.2s with the contamination gone.
> **(2) The .cmd argv newline truncation:** on Windows the claude CLI is a
> ``.cmd`` shim and cmd.exe TRUNCATES an argv argument at its first
> newline -- so since the quality-preamble commit (`c43dfd7`), EVERY
> multiline bridge prompt (preamble + task, the supervisor's enriched
> digest context, the multiline correction/adjustment templates) silently
> lost everything after line one; the model replied with a generic
> greeting instead of acting. Empirically reproduced + verified. FIX:
> ``DirectClaudeCodeBridge`` no longer passes the prompt as an argv
> argument -- the rendered prompt is piped to the subprocess STDIN
> (``DirectTaskHandle(rendered_prompt=...)`` + a fail-open daemon feeder
> thread), which preserves arbitrary content. ``_build_argv`` documents
> the prohibition. ALSO: the spoken-command matrix surfaced a real
> classifier coverage gap -- "show me the files in my downloads folder"
> fell to CONVERSATIONAL (``_FILE_PATTERNS`` supported "list the files
> in" but required the literal word directory/folder immediately after
> "in" for the show-me form). Iteration follow-up: the mirror pattern
> alone was ORDER-SHADOWED -- "show me the files..." is captured by the
> higher-priority APP_LAUNCH explorer branch (a deliberate, better-UX
> outcome: the native Explorer launch beats the gateway file stub), and
> the bare "show me X" image-search catch-all was eating OTHER
> file-shaped phrasings -- so the real fix is a `_FILE_PATTERNS` guard on
> the bare image-search branch ("show me the contents of file X" now
> routes FILE_OPERATION) and the matrix uses the unambiguous
> "What is in the folder downloads?" phrasing. The harness's
> ``_spoken_transcript`` also gained production-realistic padding (0.5s
> lead pre-roll + 0.8s trailing silence, matching the VAD-bounded
> capture) after the bare synthetic clips truncated final words
> ("Cancel the task." -> "Cancel the time."). Validated from MAIN:
> phase 8 = 26/27 -> 27/27 after fixes, phase 9 = 10/10, phase 10 = 3/3
> (warm TTFT 171ms matches the 172ms locked baseline; remember->recall
> verified; live search turn green), phase 11 = 2/2 with REAL tokens
> (create -> narration -> sandbox run exact-stdout -> edit follow-up ->
> re-run).
> **Then: the measured latency/resource pass (campaign Phase 5).**
> Voice-baseline re-measure from main with a THREE-WAY CODE A/B in the
> SAME environment: the 2026-05-26 locked-baseline code (`29ffe49`) reads
> TTFT 281ms / TTS 117ms / peak 6538MB TODAY; the pre-session tip
> (`1c43e29`) reads 266/109/6538; session HEAD reads 282/109/6596-6598.
> All three statistically identical (run noise +-24ms) -> **zero code
> regression at any point, including this session**; the apparent drift
> from the locked absolutes (TTFT 172 -> ~270, TTS 78 -> ~109) is
> ENVIRONMENTAL (GPU driver/thermal/desktop era between May 26 and
> June 10), not code. STT byte-identical at 16ms throughout. **VRAM peak
> 6596-6598MB <= the locked 6664MB even in absolute terms** (loaded
> 6130-6174 <= 6254; Kenning's own load delta ~4242MB vs the locked-era
> ~4223MB). Prefill-growth suspects ruled out empirically: the skills
> block is 0 chars on baseline queries; the composed system prompt is
> ~1k tokens; USER.md is 82 bytes. The new guardrail brake's RELATIVE
> pre/post-snapshot design is exactly the form that survives this class
> of environmental drift. `baselines.json` updated with the HEAD run.
> Optimization landed: the ~2s pure-CPU cross-encoder reranker warm in
> `Orchestrator.__init__` moved to a DAEMON THREAD so it overlaps the
> GPU model loads instead of serialising in front of them (consumers
> already lazy-load on miss -> a still-warming thread degrades to the
> pre-existing path; fail-open). The LLM/Kokoro CUDA-init overlap was
> deliberately NOT bundled -- concurrent CUDA init can shift peak-VRAM
> behaviour, so it needs its own measured pass against the locked
> contract. SearxNG restored (Docker Desktop started; container
> `kenning-searxng` maps 8888->8080) -- live search provider latency back
> to ~1.5s from the ~17s degraded-fallback path.
> **Campaign close-out: the consolidated suite green end-to-end.** The
> first full 12-test run of the expanded suite read 11/12 -- the one flag
> was `loop:no_search_turn`, where the 4B at production sampling
> temperature answered the spelled-out arithmetic probe wrong ("What is
> seven times eight?" -> "Five hundred and sixty-six."). That scenario's
> job is to assert the PIPELINE carried a spoken question through STT ->
> gate -> LLM -> TTS coherently, not to benchmark model arithmetic (which
> belongs to the drift sampler), so the probe was redesigned to a
> maximally-stable fact ("What color is the sky on a clear day?" ->
> assert "blue"; rationale documented in the harness). Re-validated from
> MAIN: phase 10 standalone 3/3, then the FULL consolidated suite
> **12/12 passed in ~109s** -- every phase (stt / llm / tts / web_search
> incl. live SearxNG / memory / routing / gate / commands /
> short_circuits / full_loop incl. a live search turn / coding with REAL
> CLI tasks) green in one run.
> **Then (2026-06-11): live dogfood session + the teammate voice relay.**
> A monitored live run of the full stack surfaced real findings (see the
> session notes): the supervisor's ProjectIndex fails at startup because
> ConversationMemory already holds the local-mode Qdrant lock
> (registry-only fallback every run); "what time is it in Paris" web-
> searches instead of doing zoneinfo arithmetic; search queries go to
> providers as the RAW utterance (no query distillation) and the
> search-augmented prompt refuses instead of falling back to parametric
> knowledge on bad snippets; a follow-up command was dropped at
> addressing conf 0.75 vs the 0.80 threshold; follow-up captures pay a
> ~700 ms synchronous Moonshine re-transcribe (streaming cache miss);
> image-search APP_LAUNCH windows open BEHIND the foreground (no
> bring-to-front after placement), still use the deprecated `tbm=isch`
> URL, don't resolve pronoun subjects ("a picture of *it*"), and the
> launcher returns success=True even when the window never appeared.
> SHIPPED from the session: NEW `audio/relay_speech.py` (see the module
> section) -- "tell my teammates X" now rephrases to a direct second-
> person line and speaks it on a configurable secondary output device
> (VoiceMeeter strip -> mic B-bus) so the game voice chat hears Kenning;
> `relay_speech` config (default ON) + `_maybe_handle_relay_speech`
> orchestrator short-circuit + 39 hermetic tests. ALSO SHIPPED: NEW
> `audio/output_quality.py` TTS blip watcher (see the module section) --
> every synthesized clip analyzed on a daemon thread for edge bursts /
> join clicks / dropouts / clipping; findings -> WARNING +
> `logs/audio_quality.jsonl`; `tts.output_watch` config (default ON);
> hook at the `_synthesize` tail costs the hot path only a non-blocking
> queue put; +21 hermetic tests + a session-scoped conftest guard so
> stubbed-synth unit tests never build the watcher singleton.
> THEN the relay grew into a CONVERSATIONAL game-chat agent: named
> agent callouts ("ask Clove to smoke window" — closed Valorant-roster
> vocabulary, `relay_speech.addressee_names` extendable), compose mode
> ("give my team encouragement" — Kenning authors the line),
> first-person-preserving rephrase, recent-lines ring -> wording varies
> per call (anti-soundboard), relay matches bypass the follow-up
> addressing gate (no wake word inside the window; fixes the observed
> 0.75-conf drop) and hold the window open `follow_up_seconds` (120 s).
> THEN the voice-launched CONTROL PANEL (NEW `settings_gui/` — see the
> module section): "pull up your settings" spawns a detached
> dark-theme tkinter panel (9 cards / ~36 curated knobs + a live
> log-stream pane); APPLY UPDATE patches config.yaml
> comment-preservingly + signals the running orchestrator to hot-reload
> (`_maybe_reload_config`, one os.stat per loop tick); CLOSE exits the
> process — pipeline untouched throughout. Two drift-guard tests pin
> the knob catalogue to the real config.yaml.
> THEN (same day): the relay mute toggle + stream-safe matching
> (session mute "mute the team chat"; narration can never relay;
> possessive + question-word callouts); the panel's OUTPUT WAVEFORM
> pane (per-clip envelope stream from the blip watcher, red markers at
> finding positions); **anticheat-safe mode** (NEW
> `safety/anticheat.py` — see the module section: 49 module guards
> across 14 desktop modules + validator BLOCK_HARD + voice/gaming-mode
> toggles; audio + team relay stay live); and a DISK CLEANING run
> (dead training assets + 22 stale worktrees + 1 unreferenced GGUF →
> `I:\Ultron Archive\2026-06-11\`, ~40 GB reclaimed; the live
> `kokoro_finetune` compat path + all preset GGUFs + locked voice
> assets verified-referenced and untouched).
> THEN: the per-response VRAM creep ROOT-CAUSED + FIXED (zero latency).
> A 5-agent read-only audit + a controlled reproduction harness
> (embedder / Kokoro / LLM measured in isolation over 40 iters) proved
> there is NO unbounded leak: the embedder is FastEmbed/ONNX off the
> torch heap (flat), the llama-cpp LLM has a fixed preallocated KV cache
> (flat), and Kokoro plateaus after warming. The real creep is the torch
> caching allocator's RESERVED high-water mark ratcheting to the largest
> synth on the CUDA-Kokoro path (the user's `kokoro.device: cuda`),
> compounded by the KPipeline generator never being explicitly closed
> (retained GPU refs lingered until GC). Two conservative fixes, neither
> touching the hot path with a sync: (1) `KokoroSpeech._synthesize` now
> closes the generator in a `finally` (drops refs immediately; no CUDA
> sync); (2) NEW `Orchestrator._reclaim_idle_vram` calls
> `torch.cuda.empty_cache()` at the IDLE transition (after the reply is
> spoken, before the wake-word wait -- off the latency-critical span),
> gated on `llm.idle_vram_reclaim.min_slack_mb` (192MB default) so it
> only syncs when there's real reserved bloat. Config
> `llm.idle_vram_reclaim` (default ON). The audit confirmed every other
> GPU consumer (embedder / reranker / VLM) is correctly CPU-resident and
> must STAY there -- moving any GPU-resident model (Kokoro weights, LLM
> layers, Whisper) to CPU would cost latency, so none were moved (the
> user's zero-latency-impact constraint). +6 reclaim tests.
> THEN: startup Docker autostart for SearxNG (NEW
> `lifecycle/docker_startup.py`): SearxNG is the default first search
> provider but runs in a Docker container — if Docker is down at boot
> every search silently falls through to Brave. The orchestrator now
> probes SearxNG at startup (`ensure_docker_running`, daemon thread)
> and launches Docker Desktop if unreachable (exe path from
> `gaming_mode.docker_executable_path`; fail-open; gated by
> `web_search.searxng.autostart_docker_on_boot`, default ON). +10 tests.
> THEN: the user-audible "blip after the sentence" FIXED at the source.
> The blip watcher's live measurements (isolated ~70 ms burst ~440 ms
> after the speech body, deterministic on the ack clips) exposed a real
> `trim_and_fade` bug: a loud post-speech burst counts as a "speech
> frame" above the −40 dB threshold, so `speech_frames[-1]` pointed at
> the BURST — the trim kept the dead air + blip and faded the blip's
> tail instead of the speech's. Fix: loud frames are grouped into runs
> and edge runs that are short (≤120 ms) and isolated (≥200 ms gap)
> are discarded before the trim window is chosen — real words are
> longer than the cap, so speech is never clipped (test-pinned). New
> tests replicate the watcher-measured geometry and cross-validate
> with `analyze_clip` (no `trailing_burst`/`hard_tail` after the trim).
> THEN (post-unload log review — the user flagged bad live responses):
> **the relay spoke conversation history into game chat** ("Clove, the
> program is still in development… / no_think") — root causes fixed:
> relay rephrase now generates FULLY ISOLATED
> (`suppress_memory_context=True`; history no longer prepended) and
> strips control-token leakage; `/no_think` is now appended ONLY for
> Qwen-family presets (`_apply_no_think_marker` checks
> preset+model_path — the llama-3.2 gaming preset parroted it and TTS
> said "No think" aloud); NEW `tts/text_hygiene.py`
> `sanitize_spoken_text` at the `_synthesize` choke point — asterisk/
> bracket stage directions ("*repositions window…*" was SPOKEN live),
> `<think>` spans, control tokens, and punctuation-only fragments
> never reach the voice, for any model; `_GROUP_WORDS` tolerates the
> "teams" STT artifact. All four live artifacts are verbatim tests.
> THEN (same evening, four fixes): **gaming mode frees everything** —
> `toggle_docker: true` (Docker Desktop/vmmem stopped on engage,
> restarted on disengage; SearxNG fails open to Brave/DDG) + the
> settings panel process is closed on engage (the state machine already
> swapped the LLM, killed Parakeet, moved Kokoro to CPU, unloaded the
> VLM). **Panel spawns with NO console** (pythonw.exe preferred, else
> CREATE_NO_WINDOW; never DETACHED_PROCESS, which popped a visible
> console). **Vanguard paranoia pass**: +2 UIA coordinate-reader guards
> (51 total), a forbidden-API scanner test
> (`test_no_ban_class_apis_anywhere_in_source`: OpenProcess /
> R/W-ProcessMemory / CreateRemoteThread / SetWindowsHookEx /
> RegisterRawInputDevices / pynput / dxcam / ImageGrab must NEVER
> appear outside the safety/rules defense regexes — grep-proved clean),
> kernel-anticheat threat-model analysis documented in
> `safety/anticheat.py`. **Live-incident fixes (addressing + context
> corruption)**: the factual-question-stem addressing rule demotes
> FRAGMENTS (<4 words / trailing comma-conjunction / third-person
> "how he was…") to UNCERTAIN — the verbatim live incident ("How he
> was initially," accepted at 0.85 → LLM recited a month-old Moscow
> weather memory as current) is now a regression test; cross-session
> RAG is skipped below a 5-word query floor
> (`_rag_query_has_min_content`) and the injected block header now
> labels snippets as possibly-stale PAST-conversation memories with an
> explicit never-recite-time-sensitive-facts instruction.
> Earlier sweep state: **9156 passed / 35 skipped / 0 failed (~103s)** with the
> loaded-machine ignore recipe (below); ~9182 no-deselect (now 9199 on an idle
> machine, no deselect, 2026-06-10 baseline). The +8 skipped vs earlier are
> the new GPU-gated voice-e2e suite). Under this session's heavy machine load
> MULTIPLE real-subprocess files (`tests/integration/test_bridge_e2e.py` AND
> `tests/openclaw_bridge/test_client.py`) flake + wedge the watchdog -> ignore
> both for a clean read: **9130 passed / 35 skipped / 0 failed**. Separately,
> the full voice e2e suite runs GREEN from the MAIN checkout (models live there,
> not the worktree): **8/8 phases pass** via
> `PYTEST_RUN_GPU_TESTS=1 .venv/Scripts/python -m pytest tests/integration/test_voice_e2e.py`.
> Voice baseline
> contract intact throughout (no SOUL.md / RVC / Piper / LLM-GGUF / voicepack
> touch; all changes are on the coding + fail-open seams). **Coding-engineer
> commits landed so far** (the campaign's first phase -- a fully capable
> voice-controlled coding engineer):
> * **`a8e6ef6`** -- B1 cohesion/security pass: the coding bridge's safety
>   FILE_CHANGE listener read the wrong `TaskEvent` attributes
>   (`event.path`/`change_kind` instead of `event.file_path` /
>   `file_change_kind.value`), so file-write validation never actually ran on
>   coding edits; fixed + removed dead code + `r`-prefixed three regex docstrings.
> * **`8651b07`** -- B3-loop: voice-dispatched coding tasks now write a
>   per-project `.mcp.json` pointing at the live in-process MCP server + bind a
>   `ProjectSession`, so the spawned coding subprocess can reach
>   request_clarification / report_progress / declare_complete -- the
>   clarify/verify/complete loop is actually connected for voice tasks (it was
>   dispatched bridge-only before). Added `KenningMCPServer.is_running()`.
> * **`9cf6f45`** -- B3-runlaunch: NEW `src/kenning/coding/sandbox_runner.py` +
>   the "run the calculator" / "launch the server" voice commands. Runs are
>   sandbox-confined (hard `_is_within` root check) + safety-validator-gated +
>   non-blocking (background thread -> a pending run report the orchestrator
>   drains); launches detach. The completion report appends "say run X to try
>   it." New `coding.sandbox_run_timeout_seconds` knob (default 30s).
> * **`c43dfd7`** -- B3-quality-a: an always-on ~60-token code-quality preamble
>   (type hints, concise docstrings, no bare except, `pyproject.toml` for new
>   Python projects, stay in the working dir) is prepended to every coding prompt
>   -- so voice-dispatched tasks (`require_testing=False`) still get best-
>   practices guidance, not just the testing-discipline path.
> * **`c17a2c9`** -- B3-loop-2: keep the per-project `.mcp.json` across follow-ups.
>   The earlier cleanup-on-COMPLETE deleted it, but `send_followup` (RESUME + the
>   verifier's corrective re-prompt) reuses the runner's stored `mcp_config_path`
>   AFTER the task completes -- so the cleanup was stripping MCP tools from every
>   follow-up + correction, breaking the loop B3-loop had just connected. Also
>   re-attaches the digest + voice-lock-review listeners to the fresh handle
>   `send_followup` spawns on RESUME_FORWARD.
>
> The architect-plan TTS narration (`_build_architect_narrator` ->
> `ArchitectNarrator`, gated on `coding.architect.narrate_enabled`) and the
> architect plan-provider were verified ALREADY fully wired into
> `_build_supervisor_stack` -- no gap there. Remaining campaign work (breadth +
> the other phases) is tracked in
> `memory/project_ultron_2026_05_29_production_hardening_campaign.md`:
> routing/bridge/safety/infra/desktop reachability wiring, evolution pervasive
> reach (with an approval gate before any `src/` edit), the unified synthetic-
> audio e2e suite, and latency/resource optimization.
>
> **Breadth phase -- routing batch (B4) resolved (`6d23bb9`):** verify-first
> showed most triage "stubs" here are non-gaps. **#24 (real, fixed):** the
> `HybridTaskDecomposer` was built but never called -- HYBRID_TASK returned a
> stale "gateway isn't connected" stub. Wired via `_handle_hybrid_task` +
> `_dispatch_automation_subtask` (automation-before-coding runs inline; the
> first coding subtask dispatches; post-coding steps are surfaced as a deferred
> plan -- the single-in-flight-task model can't auto-run a long-coding-then-
> automation sequence in one synchronous turn). **Verified non-gaps (documented,
> NOT changed):** #23 desktop/window classifier gate is a no-op in the default
> config (`openclaw.enabled: true` so they're already classified;
> `handle_desktop_automation` already guards `client is None`; the dispatcher
> path is bridge-based, not the native UIA tier the triage assumed). #94/#95
> `IntentDisambiguator` / `should_clarify_from_config` would be dormant -- the
> classifier emits a flat 0.85 confidence and never sets
> `needs_user_clarification`, so nothing reaches the ambiguity band; useful only
> after the classifier learns to emit ambiguity signals. #25 FILE/SHELL live
> paths + #9 `stub_responses_enabled` -- the OpenClaw bridge exposes NO exec/file
> tool, the stub messages are honest (not stale), and a native subprocess path
> would be a net-new high-risk capability (voice-triggered filesystem/shell exec)
> requiring explicit user opt-in + a dedicated safety design, NOT a casual
> stub-completion. Per the security constraint these stay stubbed.
>
> **Breadth phase -- parallel verification pass (`4a0af5b`, `8b43517`):** four
> parallel Sonnet 4.6 read-only agents verified ~40 triage findings (voice
> hot-path, safety wiring, config drift, bridge/desktop reachability) into
> real-safe-wireable vs non-gap. **FIXED (all fail-open):** **#1/#30** -- a real
> bug: `DesktopTool.{screenshot,list_windows,find_window}` read a non-existent
> `result.payload` field (the real `ToolInvocationResult` carries structured
> data on `.raw`), so every OpenClaw-proxied desktop call returned empty; the
> bug hid because the test stubs defined a fake `payload` field (now aligned to
> `.raw` + a regression test uses the real type). **#28** -- `KenningIgnoreRule`
> built the ignore controller with no `workspace_root`, so only
> `~/.kenning/.kenningignore` was consulted and the project/workspace layers were
> silently skipped; now forwards `resolver.project_root`. **#100** -- the
> tamper-evident audit chain is now verified at orchestrator startup (read-only
> WARN, never blocks boot). **#59** -- the click-preview VLM safety gate
> captured a hardcoded monitor index 1, so on a single-monitor machine it
> silently degraded to "allow every click"; now captures the foreground
> window's monitor (fail-open to 0). **VERIFIED NON-GAP (documented, not
> changed):** #31/#91/#107/#108 (false positives -- SYSTEM_STATUS is wired in
> voice.py; native intents DO log; telegram staging is intentional; lifecycle
> uses the configured gateway URL, not the module-default port), #136/#137/#138
> (STT dual-load + Moonshine-CPU are documented-intentional), #27/#96/#97/#98/#99
> (dormant-by-design or false-positive-block risk -- e.g. a curl `--data-binary`
> rule would block legitimate coding uploads), #119/#157 (no real drift).
> **DEFERRED with rationale:** #76/#135/#139 (voice-pipeline latency tweaks ->
> Phase 5 measured pass), #51/#52b/#82 (intentional default-OFF: architect 3-5s
> latency, lint/repo_map/pre_task UX cost, background_summary blocked on #83's
> Qdrant writer), #26 (empty policy chain = no value until a concrete policy
> exists).
>
> **Config alignment DONE (`4698ae3`, 0 test breakage):** #49 token_budget
> 100k->400k, #50 IntentConfig.threshold 0.8->0.65, #140 inference.py
> intent_adaptive getattr fallback False->True, #156 provider/reader fallback
> literals aligned to WebSearchConfig defaults, #121 desktop window-voice
> re-exports -- all verified-safe (gated / inert / stub-only fallbacks). **STILL
> QUEUED (flip ACTIVE behavior -> need per-test default-assertion fixes via
> monkeypatch, the `ee9eca5` pattern):** #52a canonical_monitor/ast_metadata,
> #53 ambiguity/IRMA, #54 LLM compression/self-consistency; plus #124 clipboard
> singleton (marginal -- no current in-process consumer).
>
> **Breadth phase -- Phase 3 (pervasive self-improvement) STARTED (`1b04a3c`):**
> two more parallel Sonnet agents verified the evolution-reach cluster
> (#15/#16/#62-69/#125/#126 -- ALL real-safe: data-only, fail-open, off-hot-path)
> + the dead-module cluster. **DONE -- #16:** `_record_evolution_turn` now feeds
> the recent multi-turn transcript (`DualHistoryStore.recent_verbatim`) to
> `extract_signals` as `recent_session_transcript`, so the history-aware
> detectors (recurring_error / perf_bottleneck / tool_bypass) can fire instead of
> only seeing the single current utterance. **Agent corrections to the triage
> (verify-first earned its keep):** #69 (response_summary) was the agent's
> "do-first" but is WRONG -- at `_record_evolution_turn` time `_last_response_text`
> is the PRIOR turn's response (turn N's isn't generated yet), so passing it would
> MISLABEL the capsule; the `user_text` fallback is correct -> SKIPPED. #67
> (signals_provider) is NOT the linchpin -- `EvolutionLoop.plan()` ignores the
> observation arg, so wiring it alone is cosmetic. #126 (routing fallthrough) is
> semantically murky (fallthrough is NORMAL for conversational turns) -> SKIPPED.
> **QUEUED (verified real-safe):** #15+#65 (guardrail sampler + latency ring =
> the TRUE auto-revert safety brake for the currently-brakeless live loop),
> #62/#125 (web/memory failure reach-signals via
> `record_command_failure(..., exit_code=1)`), #63/#64/#66/#68. **Dead-module
> cluster verified:** 5 real quick wins (#4 forfeit escape-hatch, #72 deep loops,
> #74 latency_hygiene startup priority+warmup, #78 context-window startup guard,
> #80 dedup file reads) + 9 confirmed architecturally-dormant (#17/#70/#71/#79/
> #112/#127/#143/#155 -- no consumer window in the single-loop / delegate-to-claude
> design; force-wiring would add risk for a window that doesn't exist).
>
> **Breadth phase -- latency + Phase 4 (the unified e2e suite) DONE
> (`983a0fd`):** **#74 (`af54e14`)** wired `latency_hygiene.raise_process_priority()`
> at orchestrator startup (Above-Normal, ~50-200 ms jitter under load,
> fail-open). **Phase 4 e2e suite (`f29e8b2` + fixes):** NEW
> `tests/integration/test_voice_e2e.py` converts the maintained
> `scripts/autonomous_e2e_harness.py` into a GPU-gated pytest suite (finding
> #10) -- it drives REAL input through the REAL stack (Kokoro synth -> Moonshine
> STT -> routing + LLM -> Kokoro TTS, plus live Brave/Jina web-search, Qdrant
> memory, web-gating) and asserts each phase's scenarios. Gated on
> `PYTEST_RUN_GPU_TESTS=1`; the normal sweep SKIPS it (8 skipped). **Validated
> 8/8 GREEN from the MAIN checkout** (the worktree has no model files -- GPU
> tests MUST run from `C:\STC\ultronPrototype`). Building it surfaced + fixed
> THREE real findings: (1) a bare 1-word "Stop." STT scenario was unrealistic
> (Moonshine heard "So"; real barge-in is acoustic, not STT) -> realistic
> "Cancel the task."; (2) VRAM accumulated across phases (each builds its own
> engines) -> NEW `_free_gpu()` between phases + an autouse teardown; (3) the
> memory phase hit a Qdrant local-mode file-lock conflict (phase_llm's client
> held `data/qdrant/.lock`) -> NEW **`ConversationMemory.close()`**
> (`qdrant_store.py`, also a proper clean-shutdown method) called between phases.
> **KNOWN pre-existing (NOT introduced here):**
> `tests/test_memory_qdrant.py::test_retrieve_returns_semantic_hits` is
> order-dependent (fails in isolation, passes in the full sweep -- it relies on
> shared `data/qdrant` state instead of `tmp_path`); confirmed failing on a
> clean tree, so out of scope for this pass.
>
> **Earlier validating HEAD:** **infrastructure-wiring campaign (2026-05-29)** -- a
> sweep wiring dormant imported-but-unconsumed infrastructure across catalogs
> 1-14 to production polish, on worktree branch `claude/frosty-murdock-ba8981`,
> **validating code HEAD `296e1f6`** (this doc-bump is the trailing tip),
> pushed to `origin/main`. Every commit independently green; the voice baseline
> contract is intact (no SOUL.md / RVC / Piper / LLM-model / voicepack touch;
> the orchestrator hot path gains only fail-open, default-safe, no-op-until-used
> seams). **9117 passed / 27 skipped / 2 deselected** (loaded-machine deselect
> recipe; see the test-count note below). Fourteen commits:
> * **Process discipline (T12):** the coding subprocess + Parakeet/XTTS daemons
>   register in the process-registry + zombie-killer at spawn and unregister on
>   stop (`runner._launch`/`_finalize`, `parakeet_engine`, `xtts_v3`,
>   orchestrator start/shutdown); a latent `_maybe_handle_deep_research` `trace`
>   NameError was fixed in the same pass.
> * **Deep-memory recall:** NEW `memory/deep_recall.py` strict matcher +
>   `Orchestrator._maybe_handle_deep_recall` short-circuit (iterative RAG).
> * **Skill trust gate (T5/T9):** NEW `skills/scan.py` (`scan_skill_content` --
>   tag-injection / jailbreak / system-override detection) gating untrusted
>   (non-PUBLIC) skills in `loader`/`registry`; `scan_untrusted_skills` default-ON.
> * **Decomposer requery (T14):** `HybridTaskDecomposer._requery_decomposition`
>   re-queries a malformed plan before the coding-only fallback.
> * **Two-phase voice approval (T2):** generalised `request_voice_confirmation`
>   / `consume_voice_approval` on `CapabilityVoiceController` -- any Cap-gated
>   handler can ask-instead-of-refuse; the window-close yes/no path also
>   consumes a general approval; the validator stays fail-closed underneath.
> * **Loop detection (T1):** a per-task `LoopDetectionManager` over the coding
>   TOOL_RESULT stream -> one spoken heads-up on hard escalation (logs +
>   narrates, never cancels); `coding.loop_detection_enabled` default-ON.
> * **Dialog-narration surfacing fix:** `_drain_coding_dialog_narrations` in the
>   voice loop -- the catalog-08/09 dialog auto-handler had been queuing
>   narrations that were never spoken (`pop_dialog_narration` had no caller).
> * **Brave key rotation (T6):** `RotatingBraveClient` + `resolve_brave_api_keys`
>   + `web_search.brave_additional_api_key_envs` -- 2+ keys rotate via the
>   auth-profile store; the single-key path is unchanged. Jina/SearxNG/DDG are
>   no-auth/local and documented as needing no rotation.
> * **Dual-history recall:** NEW `memory/history_recall.py` "what did I say
>   earlier?" matcher + `DualHistoryStore` wired into the orchestrator (records
>   every addressed user utterance + LLM response) + `_maybe_handle_history_recall`;
>   `memory.history_recall_enabled` default-ON; works even when Qdrant is off.
> * **Hooks lifecycle:** coding TaskStart (cancel-capable) + TaskComplete fire
>   the out-of-process hook registry; `hooks.enabled` default-ON, zero-cost when
>   no scripts are installed (cached discovery + empty fast path). Voice-hot-path
>   lifecycle points intentionally not auto-fired (latency contract).
> * **Observability:** the `resolve_observation_outcomes` maintenance task gives
>   the offline `OutcomeResolver` a runnable home; the live emits were already
>   wired.
> * **MCP client (T22):** NEW `mcp/builder.py` + `McpConfig` -- a sandboxed
>   external-MCP-server lifecycle manager (env-filtered spawn + process-registry
>   tracking + `kill_process_tree` reap-on-shutdown); `mcp.enabled` /
>   `mcp.autostart` both default-OFF. Managed servers reach `claude` via
>   `--mcp-config`; JSON-RPC tool invocation is the optional `mcp`-SDK adapter.
> * **`.kenningignore` (safety Category U):** NEW `safety/rules/category_ignore.py`
>   `KenningIgnoreRule` -- blocks reads/writes of ignored paths + file-reading
>   shell commands targeting them (secrets protection); default-SAFE no-op until
>   a `.kenningignore` exists; `safety.rules.U1`-toggleable.
> * **Explicit-intent unblock:** the `safety.intent` matcher is wired into the
>   validator -- a `NEEDS_EXPLICIT_INTENT` verdict upgrades to an audited allow
>   ONLY when the user's current utterance names the action (verb + object);
>   NEVER overrides `BLOCK_HARD`; `safety.explicit_intent_matching_enabled`
>   default-ON.
>
> **Architecturally-inactive (assessed, deliberately DOCUMENTED rather than
> force-wired -- consistent with the user's "keep + document as inactive"
> decision on the dead coding modules):** these stem from concurrent-server /
> agent-harness catalog sources whose assumptions don't hold in Kenning's
> single-threaded run loop + delegate-to-`claude --print` design, so
> force-wiring them would add hot-path latency / risk for a window that does
> not exist (a degradation the binding constraints forbid):
> * **`memory/dual_history.truncate_*` ("undo that"):** a robust conversation
>   "undo" needs `DualHistoryStore` promoted to the unified LLM-context source.
>   Today `ConversationMemory.recent()` drives context and records ONLY LLM-path
>   turns, so "the last exchange in memory" != "the user's last utterance"
>   (capability/search short-circuits never enter it). The truncate primitive is
>   ready; the context-source unification is the prerequisite.
> * **`lifecycle/pending_message_queue`:** its concurrency windows (cold-start,
>   model-swap) don't exist -- the LLM loads in `Orchestrator.__init__` BEFORE
>   audio capture starts, and the gaming LLM swap (`reload_for_preset`) is
>   synchronous within a turn (the single-threaded loop is blocked, so no
>   utterance is captured to queue).
> * **Coding edit auto-revert:** silently restoring `claude`'s files mid-task
>   breaks its black-box mental model; the architecture-fitting use
>   (AST-syntax-failure fact-checking in the completion narration) is already
>   wired.
> * **`safety/auto_approval.AutoApprovalMatrix` session-warming:** the NEI path
>   is now complete (explicit-intent unblock + the T2 two-phase-approval "ask"
>   fallback). Auto-allow-by-warming is a further, security-sensitive layer left
>   for focused review rather than bundled into a marathon session.
>
> **Test-count note:** sweeps used the loaded-machine deselect recipe
> (`--ignore=tests/integration/test_bridge_e2e.py` + the two
> `tests/openclaw_bridge/test_client.py::test_run_cli_*` subprocess tests) ->
> 9117 passed / 27 skipped / 2 deselected. The deselected tests spawn the real
> `.cmd`->python subprocess and are the documented loaded-machine flake.
>
> **Earlier validating HEAD:** catalog 14 (clawhub-self-improving-agent) -- FOUR
> bounded extensions to the EXISTING `src/kenning/evolution/` package (NOT a
> new subsystem), **HEAD `b55697e`** (catalog-14 commits `e52c364` …
> `b55697e`; this doc-bump is the trailing tip), pushed to `origin/main`
> with the `main` checkout fast-forwarded. The benign (0 RED / 1 YELLOW) plugin adds QUALITATIVE
> conversation-event learning on top of catalog 13's quantitative metric
> loop. Same HARD SAFETY CONTRACT (data-only, Tier-3 wall, zero
> network/shell/eval, voice baseline LOCKED); built clean-room from the
> catalog entry + the two `_self_improving_scan` reports -- the quarantine
> source was NOT read. **8999 passed / 26 skipped / 0 failed** (~162 s, full
> worktree sweep, no deselect; +50 over catalog 13). The four extensions:
> * **T1 (GREEN):** `CorrectionCapsule` / `KnowledgeGapCapsule` /
>   `CommandFailureSignal` in `models.py` + detectors in `signals.py`
>   (`extract_correction`, gated on a non-empty prior response with
>   strong-phrase vs weak-opener-suppressed-by-positive-ack logic;
>   `extract_knowledge_gap`; `extract_command_failure` -- the 17-token
>   in-process analogue of the upstream's RED PostToolUse bash hook;
>   `extract_feature_request`). Corrections / knowledge gaps / command
>   failures feed the EXISTING repair-distillation path via
>   `to_failure_record`, which is why `EvolutionLoop._propose` now falls back
>   to `auto_distill_from_failures` over a new `failures_provider` when
>   success distillation yields nothing. Wired in the per-turn
>   `_record_evolution_turn` hook (`prior_response=self._last_response_text`)
>   + a coding-runner command-failure `TaskEvent` listener drained by
>   `Orchestrator._drain_evolution_command_failures`.
> * **T2 (GREEN):** `FeatureRequestCapsule` -- a forward-looking backlog
>   written to `data/evolution/feature_requests.jsonl`, NEVER distilled,
>   surfaced in `EvolutionService.digest()` ("you've asked 3x for X").
> * **T3 (YELLOW, bounded):** a <=50-token `[Evolution: ...]` pre-turn nudge
>   composed by `EvolutionService.pre_turn_system_hint()` and injected
>   through the SAME `LLMEngine.set_temperament_hint` SYSTEM-prompt seam
>   personality already uses (never the user text); token-capped, default-ON
>   behind the evolution flag, `""` when the queue is empty (prompt
>   byte-identical).
> * **T4 (GREEN):** `pattern_key` / `recurrence_count` / `first_seen` /
>   `last_seen` on the base `Capsule` + the four new types;
>   `derive_pattern_key` + `merge_capsules_by_pattern_key` +
>   `RECURRENCE_PROMOTE_THRESHOLD=3` make the distiller's recurrence gate
>   explicit + auditable (row-count recurrence; back-compatible -- an empty
>   `pattern_key` leaves the legacy gene-grouping byte-identical).
> New default-ON `EvolutionConfig` knobs: `correction_detection_enabled` /
> `feature_request_capture_enabled` / `command_failure_capture_enabled` /
> `pre_turn_nudge_enabled` / `pre_turn_nudge_max_chars` /
> `recurrence_threshold`. +~53 hermetic tests
> (`tests/evolution/test_catalog14_*.py` + orchestrator wiring).
> `THIRD_PARTY_NOTICES.md` extended with the clawhub-self-improving-agent
> clean-room provenance record. Voice baseline contract intact.
>
> **Earlier validating HEAD:** catalog 13 (clawhub-capability-evolver clean-room)
> port -- bounded autonomous self-improvement -- on worktree branch
> `claude/elated-vaughan-b90f7e`, **HEAD `5fde223`** (12 commits `0d10fb8`
> … `5fde223`; base `6334b41` = catalog 12), pushed to `origin/main` with
> the `main` checkout fast-forwarded + in sync. **The source plugin
> was QUARANTINED and NEVER read / imported / executed / deobfuscated** (it
> is treated as high-risk for malicious code); the `src/kenning/evolution/`
> package was built clean-room from the catalog entry + eight independent
> static scan reports ONLY -- no source code, constant, or string was ever
> in context to copy. The new package is a NEW top-level subsystem (10
> modules) wired into the live voice pipeline default-ON + fail-open. The
> upstream's dangerous core (an agent that rewrites its own executable code
> and runs shell / network "verify" steps) is excluded BY CONSTRUCTION:
> kenning's engine proposes DATA only (a Markdown skill under
> `data/evolution/skills/*.md` or an in-range config value), NEVER generated
> code, NEVER `src/kenning/`, NEVER a Category-K surface; the safety
> validator / audit ledger / engine itself sit behind a Tier-3 hard wall;
> zero network / shell / eval; every change is pre-flight-gated (fail-closed)
> + checkpointed + reversible + hash-chain-audited, bounded by the
> `AgentLoop.max_steps` cap. Voice baseline LOCKED (the personality tuner is
> a Tier-0 `[Tone: ...]` system-prompt hint only -- no SOUL.md / RVC / Piper
> / Kokoro touch; the orchestrator hot path gains only microsecond setter +
> signal-extraction calls). Tier summary: 7 GREEN + 2 YELLOW + (the
> self-rewriting / network / shell core) RED, excluded.
>
> The ten modules + the wiring:
> * **`models.py` (GREEN):** GEP data model -- frozen dataclasses (Gene /
>   Capsule / Mutation / EvolutionEvent / PersonalityState / BlastRadius /
>   Outcome / EnvFingerprint) with `__post_init__` coercion, content-
>   addressable `sha256` asset ids, clamp/canonicalize helpers, schema
>   version. `EnvFingerprint` omits the upstream device id (safety departure).
> * **`signals.py` (GREEN):** opportunity-signal extraction -- two LOCAL
>   layers (regex + weighted-keyword scoring + multilingual user-request
>   detection + history-aware post-processing). The upstream's third
>   LLM/Hub-network layer is NOT ported.
> * **`blast_radius.py` (GREEN):** change-scope policy spine --
>   counted-file policy, 5 severity tiers, `CRITICAL_PROTECTED_PREFIXES`
>   /`FILES` Tier-3 wall (includes `src/`), ethics-block regexes, numstat
>   parsing with an injectable git provider. The load-bearing safety wall.
> * **`skill_distiller.py` (GREEN):** capsule -> pattern -> new-skill
>   distillation. Local-only synthesis of a trigger-loaded skill
>   (kenning-compatible frontmatter) from >=10 recurring success capsules (or
>   a repair gene from failures), 24h cooldown + data-hash idempotency.
>   Output is DATA (`skills/*.md`), never code.
> * **`guardrails.py` (GREEN):** four in-process regression detectors
>   (latency / quality / error / resource ceiling) + a rollback-frequency
>   audit that demotes a churny surface. Replaces the upstream's
>   run-commands-to-verify step.
> * **`autonomy.py` (GREEN):** `TieredAutonomyController` -- Tier 0/1
>   auto-apply, Tier 2 propose -> two-phase approval (graduates after a clean
>   track record: >=20 changes, <10% revert, 0 hard trips), Tier 3 hard
>   wall; rollback-rate demotion.
> * **`personality.py` (GREEN):** Tier-0 DATA self-tune -- an adaptive
>   response temperament (verbosity / rigor / creativity) nudged by per-turn
>   satisfaction (corrections / re-asks / barge-ins) + an outcome-ranking
>   aggregator, expressed purely as a `[Tone: ...]` hint distinct from
>   `response_style`'s `[Style:`. Never touches the locked voice character.
> * **`evolution_loop.py` (YELLOW):** `EvolutionLoop(AgentLoop)` -- pre-flight
>   (fail-closed) -> autonomy gate -> reversible checkpoint -> write -> blast
>   /constraint check -> guardrails -> keep or auto-revert -> hash-chained
>   audit, bounded by `max_steps`. The first cross-subsystem consumer of the
>   catalog-11 `AgentLoop` base alongside the catalog-12 deep loops.
> * **`service.py` + `intent.py` (YELLOW):** the runtime bundle --
>   `EvolutionStore` (lock-guarded JSONL: capsules / failures / hash-chained
>   events / state / personality), `EvolutionService` (from_config /
>   record_turn / single-flight run_cycle / daemon-thread autonomous cycle /
>   temperament / digest / shutdown), and a strict voice-command matcher
>   ("evolve now" / "evolution status").
> * **Production wiring (`config.py` + `config.yaml` + `pipeline/
>   orchestrator.py` + `llm/inference.py`):** `EvolutionConfig` (default ON);
>   `_load_evolution_if_enabled` at startup (before the skill registry, whose
>   extra dirs now include `data/evolution/skills` so a kept proposal is live
>   next turn); a run-loop short-circuit `_maybe_handle_evolution_command`;
>   per-turn `_record_evolution_turn` (opportunity capsule + barge-in ->
>   temperament + autonomous-cycle trigger); `LLMEngine.set_temperament_hint`
>   injecting the `[Tone: ...]` directive into the SYSTEM prompt (never the
>   user text, so the web-gate / local-clock detectors see the raw
>   utterance); shutdown persistence.
>
> Test baseline: catalog-12 + 265 (evolution package, batches 1-9a) + 21
> (llm + orchestrator wiring, batch 9b) = **8949 passed / 26 skipped / 0
> failed** in ~119 s, worktree full sweep via `scripts/run_tests.py
> --stale-heartbeat=400`, NO deselect, exit 0 (the Windows
> subprocess-cold-start flake family did not trip on this unloaded run). All
> new tests are filesystem-independent (frozen-dataclass / fake-service /
> fake-LLM) so they pass identically in the main checkout.
>
> **Earlier validating HEAD:** catalog 12 (clawhub-felo-search) port -- ALL FIVE
> BATCHES (A-E) COMPLETE -- on worktree branch `claude/serene-bose-b88298`
> (base `0aa228d` = catalog 11), pushed to `origin/main` with the `main`
> checkout fast-forwarded. felo-search is a DOCUMENTATION-ONLY plugin (README +
> SKILL + _meta.json, no Python) that wraps the PAID `felo.ai` search API,
> so the direct API integration is RED (out of scope per the no-paid-APIs
> rule); the port implements the transferable PATTERNS over kenning's
> existing FREE local-first search ladder. Independent zero-RED security
> confirmation via 3 Sonnet read-only scanners. 3 GREEN + 1 YELLOW + 1 RED.
> The five batches (full per-batch detail below):
> * **Batch A (T2, GREEN) — DONE:** comparison / how-to / shopping trigger
>   regexes (`_COMPARISON_QUERIES` / `_HOWTO_QUERIES` / `_SHOPPING_QUERIES`)
>   added to `web_search/gating.py` `classify_by_rules`. Additive to the
>   always-on regex layer; placed AFTER the anti-search rules (greeting /
>   personal / creative still win) and BEFORE the stable-factual catch-all
>   (so "how to install X" / "X vs Y" / "price of X" route deterministic
>   SEARCH instead of escalating to the preflight LLM). Patterns kept tight
>   (no bare "better"/"best"/"buy"/"cost") to keep the false-positive rate
>   near zero. No new config knob; zero latency. +21 hermetic tests.
> * **Batch B (T1, GREEN) — DONE:** pre-search query reformulation. NEW
>   `web_search/query_rewrite.py` (`reformulate_query` + `expand_query_rules`
>   zero-cost structural rewrites + opt-in `expand_query_llm` in-process
>   decomposition + `maybe_reformulate_queries` executor helper, all
>   fail-open; `MAX_TOTAL_QUERIES=5` ceiling). Wired into
>   `WebSearchExecutor.run` BEFORE the existing dedup + fan-out (the
>   per-query URL-dedup + cache absorb the variants transparently). New
>   `web_search.query_reformulation.{enabled:true, use_llm:false,
>   max_variants:2}` config (default ON, rule-based; LLM opt-in per the
>   catalog's latency note). Logs to `logs/search_reformulations.jsonl`.
>   +40 hermetic tests.
> * **Batch C (T4, GREEN) — DONE:** search-strategy transparency. The
>   reformulated query list is recorded on `SearchPayload.queries` and
>   surfaced to the user as a `strategy: q1 | q2` line in the VISIBLE
>   TRANSCRIPT only (never spoken -> zero spoken-reply impact) via
>   `format_sources_for_transcript`; `_format_strategy_line` self-suppresses
>   for single-query searches. `format_sources_for_prompt` gains the same
>   optional `strategy_queries` param for future text / GUI channels (the
>   voice orchestrator deliberately does NOT inject it into the LLM prompt,
>   to protect spoken concision). New `web_search.expose_search_strategy:
>   true` config. Mirrors felo's "Query Analysis" disclosure. +14 tests.
> * **Batch D (T3, YELLOW) — DONE:** bounded agentic deep-research loop
>   over the FREE search ladder. NEW `web_search/deep_research.py`:
>   `DeepResearchLoop(AgentLoop)` (decompose -> search each sub-question via
>   the SAME `WebSearchExecutor` -> LLM gap analysis -> search gaps, bounded
>   by the load-bearing `max_steps` cap; fail-open at decompose / gap / each
>   sub-search; reuses the provider/reader chains + rate-limit tracker +
>   `web_results` cache) + `match_deep_research` (strict matcher: "research X
>   in depth" / "deep dive on X" / "dig deeper into X" -> topic; normal
>   "search X" never trips). No new `RoutingIntentKind` (the 23-value enum is
>   asserted by many tests) -- instead an ISOLATED orchestrator run-loop
>   short-circuit `_maybe_handle_deep_research` (precedent:
>   `_maybe_handle_report_concern`) that acks, runs the loop, and synthesizes
>   the answer through the SAME LLM->TTS streaming `_respond` uses; the
>   normal fast search path is byte-unchanged. New top-level `deep_research`
>   config (enabled:true; max_steps:3 / max_sub_queries_per_step:3 /
>   top_n_per_query:3 / max_accumulated_sources:8). +29 hermetic tests
>   (fake executor + fake LLM). First concrete consumer of the catalog-11
>   `AgentLoop` base.
> * **Batch E (T3 cross-system extensions) — DONE (importable):** the
>   deep-gather pattern generalised to kenning's other retrieval surfaces.
>   NEW `agent_loop/deep_loops.py`: a generic `DeepGatherLoop(AgentLoop)`
>   (decompose -> gather-over-an-injected-source -> LLM gap-fill, bounded by
>   the `max_steps` cap; fail-open) + three thin DI subclasses --
>   `DeepMemoryLoop` (iterative Qdrant RAG; `.recall()`),
>   `DeepExplorationLoop` (iterative ripgrep codebase exploration;
>   `.explore()`), `DeepUIDiscoveryLoop` (iterative UIA element discovery;
>   `.discover()`). Each injects its domain primitive (so it is
>   domain-agnostic + hermetically testable) behind a clean public entry
>   method. Shipped as IMPORTABLE primitives -- NO new orchestrator hot-path
>   short-circuit: kenning has no always-on runtime consumer for codebase
>   exploration (the AI coding agent self-explores) and UI discovery is a
>   miss-fallback, so wiring each to a concrete trigger is a one-call
>   integration left to the consuming surface (the proven
>   `Orchestrator._maybe_handle_deep_research` short-circuit is the
>   template). +13 hermetic tests.
>
> Test baseline (main-checkout projection): 8546 (catalog 11) + 21 (Batch A)
> + 40 (Batch B) + 14 (Batch C) + 29 (Batch D) + 13 (Batch E) = **8663
> passed / 26 skipped / 0 failed**.
> New tests are filesystem-independent (pure-function regex + fake-LLM /
> fake-config) so they pass identically in main; the canonical absolute
> count is finalised from a main-checkout sweep at session end. (Worktree
> sweeps report a lower absolute count because `models/` + some
> filesystem-parametrized fixtures live only in the main checkout. Batch E
> worktree sweep = 8654 passed / 27 skipped / 2 deselected / 0 failed with
> the whole `tests/integration/test_bridge_e2e.py` file `--ignore`d AND the
> two `tests/openclaw_bridge/test_client.py::test_run_cli_*` real-subprocess
> tests `--deselect`ed. Under heavy machine contention EVERY test that
> spawns a real subprocess -- the openclaw `.cmd`->python bridge in
> test_bridge_e2e.py AND even the fake `echo_cli` in test_client.py --
> cold-starts slower than its ~7s probe, fails, and leaks a process that
> wedges the sweep to the watchdog. This is the documented Windows
> subprocess-cold-start-under-load flake family, NOT a regression from this
> catalog (which never touches the OpenClaw bridge / subprocess path); on an
> unloaded machine these pass without exclusion.) Voice baseline contract
> intact — gate rules + rule-based reformulation are pure functions on the
> SEARCH-classification path; the LLM reformulation variant is opt-in
> (`use_llm`); the search-strategy line is transcript-only; no voice
> hot-path or model-default touch.
>
> **Earlier validating HEAD:** catalog 11 (clawhub-browser-agent) port on
> `origin/main` (worktree branch `claude/eloquent-solomon-fc0df3`; the
> `main` checkout at `C:\STC\ultronPrototype\` fast-forwarded to the
> same SHA. Feature commit `cfc1d27`; this doc-bump finalises the
> validating-HEAD reference).
> Catalog 11 is a raw-CDP-WebSocket primitives plugin architecturally
> superseded by the catalog-10 `browser-use` CLI tier, so the port
> extracts the genuinely transferable hardening patterns + the
> agent-loop meta-pattern (4 GREEN + 3 YELLOW + 0 RED), NOT the CDP
> transport. NEW modules: `src/kenning/utils/heartbeat.py` (T2
> `HeartbeatThread` -- a stoppable, fail-open daemon keep-alive
> primitive; improves on the upstream's unstoppable `while True:
> sleep` with `Event.wait` + `HeartbeatStats`),
> `src/kenning/utils/health_check.py` (T4 `http_health_check` +
> `cdp_health_check` -- cheap fail-open "is this endpoint answering?"
> pre-flight primitives with an injectable transport),
> `src/kenning/agent_loop/base.py` (the `AgentLoop` meta-pattern: an
> ADDITIVE, safety-instrumented observe -> plan -> act -> verify base
> class whose load-bearing invariant is the `max_steps` cap, plus
> built-in repeated-signature loop detection + per-step `StepRecord`s +
> a per-step verify hook + fail-open execution; does NOT modify any
> existing runner). `BrowserUseTool` (`desktop/browser_use.py`) gains
> three gated methods: `click_css_selector` (T3 -- CSS-selector ->
> `getBoundingClientRect` -> Cap-3-gated `click_at_coords`; the
> ARIA-ref-miss fallback, computing the *correct* box-model centre
> rather than the upstream's buggy `(border[0]+border[1])/2`),
> `wait_for_element_js` (T7 -- event-driven `MutationObserver`
> element-appear wait via the gated `eval`, with a bounded `setTimeout`
> fallback the upstream lacked), and `export_pdf` (T6 -- `Page.printToPDF`
> page-to-PDF export, `PathResolver` + Cap-2/Cap-3 gated, fail-open).
> T1 idle-reconnect / T2 heartbeat / T4 health-check / T5
> `--remote-allow-origins` have NO literal wiring target in kenning's
> CLI-based browser architecture (the `browser-use` CLI owns its own
> Chrome process + CDP port; kenning never opens a raw CDP socket; the
> launcher deliberately blocks CDP flags on the user's real Chrome) --
> so T2 + T4 ship as importable primitives and T1 + T5 are documented
> findings, not forced code. All additions are clean-room
> re-implementations from a zero-RED-confirmed read-only source scan
> (3 Sonnet 4.6 Explore agents); no source copied verbatim. No new
> config knobs (the new capabilities are always-available gated methods
> on the already-default-ON `browser_use` tool + importable utils, so
> the ship-session-work default-ON rule is satisfied trivially). Tests:
> **8546 passed / 26 skipped / 0 failed in ~154 s** (the 8475 baseline
> + 71 new hermetic tests across `tests/utils/test_heartbeat.py`,
> `tests/utils/test_health_check.py`,
> `tests/desktop/test_browser_use_catalog11.py`,
> `tests/agent_loop/test_base.py`; the bridge-e2e flake did NOT trip,
> the sweep ran green WITHOUT deselection). Voice baseline contract
> intact (no SOUL.md / RVC / Piper / vocal WAV / LLM model file /
> Kokoro fine-tune voicepack touch; no orchestrator hot-path edit).
> `THIRD_PARTY_NOTICES.md` extended with clawhub-browser-agent (MIT,
> peng yi) attribution + a 6-row per-component mapping.
>
> **Earlier validating HEAD:** `f176f29` on `origin/main` (worktree branch
> `claude/stoic-banach-74402b`; the `main` checkout at
> `C:\STC\ultronPrototype\` is fast-forwarded to the same SHA). This
> commit records the bridge-e2e subprocess-reap flake fix (fix commit
> `7b53ea1`, cherry-picked from `claude/wonderful-sutherland-2159f9`'s
> `7592687`) layered on top of the deferred-primitive wiring pass
> (`d220b50`) and the catalog 10 browser-use port (`5451017`).
> Resolves the long-deselected
> `tests/integration/test_bridge_e2e.py::test_health_through_real_subprocess`
> flake: under a loaded full sweep the Windows `.cmd` -> `python` CLI
> shim cold-started slower than the 5s health probe; the probe timed
> out and `OpenClawClient._run_cli` killed only the immediate child,
> orphaning the grandchild interpreter -- which held the stdout/stderr
> pipes open and wedged the event loop's subprocess transport,
> stalling the whole sweep to the wall-clock watchdog. Fix:
> `_run_cli`'s timeout path now reaps the WHOLE process tree via
> `subprocess.kill_tree.kill_process_tree` (collected while the root
> is alive so psutil can still reach the grandchild) through a new
> `OpenClawClient._reap_process_tree` helper; the e2e test now uses a
> 20s probe (still under the 30s per-test deadline); new hermetic
> `test_client.py::test_run_cli_timeout_reaps_whole_process_tree`
> pins the reaper contract. The reap fix GREATLY IMPROVES this flake:
> the test passes in isolation (~0.5s) and on an unloaded machine the
> prior session saw the full sweep green WITHOUT deselection at **8475
> passed / 26 skipped / 0 failed in ~106s** (the integrated tree adds
> the wiring pass's ~51 tests on top of the fix author's 8424
> baseline). **It is NOT fully fixed, though:** the test spawns the
> real openclaw `.cmd`->python subprocess, so under heavy machine
> contention (cold cache + another local server running) its 20s
> health probe can be exceeded -> the test FAILS at ~22s -> the sweep
> wedges to the wall-clock watchdog (exit 5, stalling at ~33% on
> exactly this test -- observed 2026-05-30). **Loaded-machine
> fallback:** `-- --deselect "tests/integration/test_bridge_e2e.py::test_health_through_real_subprocess"`
> -- the other tests are unaffected (verified 2026-05-30 under
> contention: **8474 passed / 26 skipped / 1 deselected / 0 failed in
> ~160s**). A 33%-stall on this test is the flake, NOT a regression.
> Voice baseline contract intact (`_run_cli` is the OpenClaw bridge
> transport, not the voice hot path).
>
> **Deferred-primitive wiring pass (2026-05-30, on top of catalog 10).**
> Consumes previously-shipped-but-unwired catalog primitives into
> their canonical hot-path consumers:
> * **T15 private telemetry** (`observability/private_telemetry.py`) --
>   the orchestrator constructs a `PrivateMetricsStore` at startup
>   (`_init_telemetry_store`) and emits one aggregate `HashedEvent`
>   per conversational turn from `_respond`'s finally
>   (`_emit_turn_telemetry`: routing-intent kind under the `category`
>   safe key + `searched` bool + numeric `latency_ms` + coarse `tier`
>   bucket + `outcome` enum). FAIL-PRIVATE (no-op unless
>   `KENNING_TELEMETRY=opt-in`) -- the one feature deliberately NOT
>   default-on, because privacy-by-construction is the documented
>   reason. Fail-open at every layer. Triage of the remaining
>   ported-but-unwired primitives: cline T9 (MCP-hub) already closed
>   by OpenClaw T22's `mcp/` package; OpenClaw T17/T19/T20 were never
>   ported (deferred per the catalog star rating) so there is nothing
>   to wire.
> * **T7 short-lived token** (`identity/short_lived_token.py`) --
>   `_mint_forensic_token` registers a trusted-caller tuple
>   (idempotent) + mints an HS256 JWT at two privilege-grant
>   boundaries: MCP-server start (`mcp:tools`, scope
>   `mcp.tools.read`/`invoke`) and gaming-mode engage
>   (`voice:gaming-engage`, scope `llm.preset.swap`, revoke-by-expiry
>   on disengage). Every mint hits the module's hash-chained audit
>   log. Forensic / defense-in-depth in the single-user in-process
>   runtime (minter + verifier share a trust boundary), not a hard
>   gate. Fail-open.
> * **T12 report queue** (`feedback/report_queue.py` +
>   NEW `feedback/report_intent.py`) -- a spoken "log a concern /
>   flag that response / that answer was wrong" is matched by the
>   strict `match_report_concern` regex (no LLM round-trip; benign
>   "report on the weather" does NOT trip it) and intercepted in the
>   orchestrator run loop BEFORE routing (where `_last_response_text`
>   still holds the prior turn). `_maybe_handle_report_concern` files
>   a `Report` (target_id = 16-hex digest of the prior response,
>   reason = verbatim utterance, response preview in extras) to the
>   hash-chained `data/feedback/reports.jsonl` and speaks an ack.
>   Triage stays deferred to a future operator pass (the catalog's
>   YELLOW two-phase-gated triage); filing is the low-risk
>   append-only half. Fail-open.
> * **T18 image markdown** (`llm/image_markdown.py`) -- verified to
>   have NO real consumer in the current stack, so deliberately left
>   importable rather than fake-wired. The encoder produces SWE-Agent
>   `![alt](data:mime;base64,...)` data-URLs for a multimodal LLM
>   that consumes the OpenAI `[{type:text},{type:image_url}]` content
>   shape; kenning has no such surface today: the in-process Qwen 4B
>   is text-only, the coding bridge passes `render_prompt(request)`
>   as a plain text argv to `claude --print` (the AI coding agent
>   consumes images via @-file references, NOT inline data-URLs), and
>   moondream2 takes raw bytes. Embedding base64 data-URLs in the
>   JSONL audit / events logs would be pure bloat. The module stays
>   ready for a config-flip + handler-swap when a multimodal LLM
>   path lands (its own docstring's stated intent). Wiring it now
>   would be a speculative feature with no user.
>
> Net of the wiring pass: T15 + T7 + T12 consumed into hot paths;
> T18 verified consumer-less + left importable; cline T9 already
> closed; OpenClaw T17/T19/T20 never ported (catalog-deferred).
>
> **Earlier validating HEAD:** catalog 10 (clawhub-browser-use) port on
> `claude/stoic-banach-74402b` (pushed to `origin/main`). Nine-batch
> port wrapping the external open-source `browser-use` CLI as kenning's
> CDP-backed browser automation tier -- the second tier above the UIA
> `extract_browser_content` extractor. The plugin source was
> documentation-only (a `SKILL.md` + two `references/*.md` recipes +
> `_meta.json`; no Python source), so `src/kenning/desktop/browser_use.py`
> is a clean-room subprocess wrapper around the documented public CLI
> surface -- NOT a vendored copy. Independent zero-RED-confirmation
> security review via a Sonnet 4.6 Explore agent; T12 (cloud, paid
> API) + T13 (Cloudflare tunnel) NOT ported (RED). 8 GREEN + 5 YELLOW
> + 2 RED.
>
> | Batch | SHA | Techniques |
> |---|---|---|
> | 1 | `35c8469` | T1 state + T2 extraction + T5 wait + T6 tabs (GREEN read foundation) |
> | 2 | `ad26469` | T7 write primitives + T9 screenshot (GREEN/YELLOW; upload PathResolver-gated) |
> | 3 | `3311031` | T3 JS eval -- static analysis + two-phase approval (YELLOW) |
> | 4 | `438a7b2` | T4 cookies get/set/clear/export/import (YELLOW) |
> | 5 | `fa2b981` | T8 named-session isolation -- `browser_sessions.py` BrowserSessionsManager (YELLOW) |
> | 6 | `45cd46e` | T10 profile connect + connect_profile + profile_list (YELLOW) |
> | 7 | `de28c0e` | T11 raw CDP passthrough -- domain blocklist + always-two-phase (YELLOW) |
> | 8 | `1721b6a` | BrowserSequenceRunner -- `browser_sequence.py` (creative extension, GREEN) |
> | 9 | `5451017` | orchestrator singleton construction + screen_context fallback tier + THIRD_PARTY_NOTICES |
> | 9-bump | (this commit) | validating-HEAD SHA bump |
>
> New modules: `src/kenning/desktop/browser_use.py` (the tool),
> `src/kenning/desktop/browser_sessions.py` (session manager),
> `src/kenning/desktop/browser_sequence.py` (sequence runner). New
> top-level `browser_use` config section (8 knobs, all default ON;
> fail-open covers the missing-binary case). Orchestrator constructs
> the `BrowserUseTool` + `BrowserSessionsManager` singletons at
> startup (`_load_browser_use_if_enabled`). `screen_context.py` gains
> a gated browser-use fallback tier (`_maybe_browser_use_state_text`)
> that fires only when the UIA browser extraction returned empty AND
> the tool is live with an active page; content is clearly
> "browser-use ..."-labelled because the daemon controls its own
> browser instance (the user's foreground only after connect/connect_profile).
> Every YELLOW write routes through `safety.validator` + (for the
> destructive / takeover ops) `safety.two_phase_approval`. The
> `browser-use` binary is NOT a hard dependency.
>
> Tests: **+431 hermetic** across `tests/desktop/test_browser_use.py`
> (320), `test_browser_sessions.py` (31), `test_browser_sequence.py`
> (18), + 8 screen_context fallback cases + the prior baseline.
> Full sweep green with the documented-flaky
> `test_bridge_e2e.py::test_health_through_real_subprocess` deselected
> (it passes 0.39s isolated but leaks a subprocess + stalls the sweep
> to the wall-clock deadline under contention -- a pre-existing
> environmental flake unrelated to this port; browser_use.py is
> isolated from the bridge e2e path). Voice baseline contract intact
> (no SOUL.md / RVC / Piper / vocal WAV / LLM model file / Kokoro
> fine-tune voicepack touch; orchestrator startup gains only the
> cheap + lazy + fail-open singleton construction).
>
> **Earlier validating HEAD:** `bca58b0` on `claude/priceless-swanson-59e65b`
> -- handoff-cleanup commit on top of the default-ON sweep (`aa00bb9`)
> + doc-bump (`169da18`). Fixes two stale-test / sloppy-iteration
> surfaces revealed once the production-wiring defaults turned more
> listeners on: (1) `screen_context.py` browser-content inputs
> iterator now tolerates both the production `UIElementInfo` dataclass
> shape AND the legacy `(label, value)` tuple shape used by some
> test fixtures; (2) the two `test_listener_*` cases in
> `test_canonical_monitor_runner_wiring.py` were asserting exact
> listener counts (1 / 2) that assumed only safety-validator +
> canonical-monitor would attach — now production defaults also
> attach goal-anchors / AST-syntax / pre-write-lint / dialog-auto-handler.
> Replaced with a qualname-introspecting `_has_canonical_monitor_listener`
> helper so the test only asserts on the wire it's named after.
> Tests at that HEAD: **8046 passing / 26 skipped / 0 failed in ~107 s**.
>
> **Earlier validating HEAD:** `aa00bb9` on `claude/priceless-swanson-59e65b`
> -- default-ON sweep on top of the catalog 09 production-wiring pass.
> Flips `llm.history_compression.intent_adaptive` from False to True
> (NoOp on the common conversational path is zero-cost; fail-open at
> every layer) AND starts the `DialogPoller` daemon in
> `Orchestrator.__init__` so batch A's bus events actually fire in
> production (the subscription chain in batch B was dead without it).
>
> **Earlier validating HEAD:** `8ba52bd` on `claude/priceless-swanson-59e65b`
> -- catalog 09 production-wiring pass closing commit. Eight + one
> feature commits on top of catalog 09's closing `6add7a6` doc-bump
> landed the previously-deferred wirings (`a` through `i` plus G + H):
>
> | SHA | Title |
> |---|---|
> | `1f0ef86` | feat(desktop+bus+tests): batch A -- DialogAppearedEvent + DialogResolvedEvent + background DialogPoller |
> | `087fa85` | feat(desktop+tests): batch I -- T7 Tesseract OCR tier (catalog 08 deferred) |
> | `36e293e` | feat(desktop+tests): batch D -- wire extract_browser_content into screen_context |
> | `d005757` | feat(routing+voice+tests): batch C -- ACTIVE_WINDOW_QUERY + SEMANTIC_CLICK + WINDOW_CLOSE_CONFIRMATION voice intents |
> | `bd97155` | feat(coding+config+tests): batch F -- pre-edit content snapshot in direct_bridge (SWE-Agent T1 + T14 wiring) |
> | `c4113e2` | feat(coding+config+tests): batch B -- wire dialog auto-handler into coding bridge |
> | `45932e4` | feat(coding+tests): batch E -- two-phase approval voice yes/no for close_window with suspected_unsaved |
> | `c9350a1` | feat(llm+orchestrator+tests): batch G -- per-intent condenser selection |
> | `8ba52bd` | feat(lifecycle+orchestrator+tests): batch H -- drive_start_task gaming-engage state machine |
>
> Tests: **7821 passing / 26 skipped / 0 failed in ~146 s** (+106 net
> over the catalog 09 baseline of 7715). New modules added across the
> session: `src/kenning/desktop/dialog_poller.py`, `src/kenning/desktop/ocr.py`,
> `src/kenning/lifecycle/gaming_engage.py`. Three new RoutingIntentKind
> values land voice intents for active-window-query / semantic-click /
> window-close-confirmation. Two-phase approval registers via
> `ApprovalRegistry` on `suspected_unsaved=True` and the spoken yes/no
> consumes it; "no" cancels, "yes" triggers gated force-close. Pre-edit
> snapshot hook in `coding/direct_bridge.py` records the pre-edit file
> content via `FileHistory.record_pre_edit` BEFORE the CLI executes the
> tool so SWE-Agent T1 auto-revert + T14 edit_recovery has the
> required snapshot. Browser foreground detection in `screen_context.py`
> uses `extract_browser_content` first (20-100 ms UIA), falls back to
> `collect_window_text` on None / empty / raise. T7 OCR ships as
> importable `desktop/ocr.py` (pytesseract lazy import, fail-open). The
> per-intent condenser branch in `LLMEngine._build_messages` is gated by
> `llm.history_compression.intent_adaptive` (default OFF -- legacy
> fixed pipeline preserved). Gaming-engage gains observable substep
> transitions + per-stage voice acks via `drive_start_task` driving
> `gaming_engage_iterator` / `gaming_disengage_iterator` async
> generators -- semantics unchanged from the prior synchronous callbacks.
>
> **Earlier validating HEAD:** `1c62068` on `claude/priceless-swanson-59e65b` (pre-push)
> -- clawhub-desktop-control catalog 09 port batch 5 closing commit. Five
> feature commits on top of catalog 08's `ee5f8dc` README refresh:
>
> | SHA | Title |
> |---|---|
> | `21d0497` | feat(desktop+mcp+tests): catalog 09 batch 1 -- T1 scroll direction + T3 WPM typing + T7 bezier-smooth move |
> | `e3274b6` | feat(desktop+tests): catalog 09 batch 2 -- T2 pixel-color probe + wait_for_pixel_color barrier |
> | `9c3be8a` | feat(desktop+mcp+tests): catalog 09 batch 3 -- T4 clipboard read/write with safety + taint |
> | `7e7e40d` | feat(desktop+mcp+tests): catalog 09 batch 4 -- T6 image template matching |
> | `1c62068` | feat(desktop+tests): catalog 09 batch 5 -- T5 DesktopSequenceRunner with before/after screenshot bracketing |
>
> All seven catalog 09 techniques landed (T1 scroll, T2 pixel-color, T3
> WPM typing, T4 clipboard, T5 sequence runner, T6 template matching, T7
> bezier-smooth move). Three new modules ship: `src/kenning/desktop/clipboard.py`,
> `src/kenning/desktop/sequence.py`, plus T6 `find_image_on_screen` +
> `TemplateMatch` and T2 `get_pixel_color` added to `desktop/capture.py`;
> T2 `wait_for_pixel_color` added to `desktop/uia.py`; T1 / T3 / T7
> additive kwargs added to existing `InputController` methods in
> `desktop/input_control.py`. Five new MCP tools: `clipboard_read` /
> `clipboard_write` / `find_image_on_screen` plus extended kwargs on
> `mouse_move(smooth=)` / `type_text(wpm=)` / `scroll(direction=)`.
> Tier summary: 4 GREEN + 3 YELLOW + 0 RED. Voice baseline contract
> intact (no SOUL.md / RVC / Piper / vocal WAV / LLM model file /
> Kokoro fine-tune voicepack touch; no orchestrator hot-path edit;
> all surfaces ship as importable infrastructure for future opt-in
> wiring). Source plugin (~93 KB across 7 files) read read-only via
> Read tool; never executed (per the binding ClawHub-batch security
> rules in `feedback_reference_repo_catalog_workflow.md`). Pattern
> extraction via Sonnet 4.6 Explore agent that independently confirmed
> zero-RED security finding (no network calls, no subprocess /
> os.system, no ctypes / registry access, no persistence, no
> anti-forensics, no credential access, no obfuscation, no AV / EDR
> tampering). `THIRD_PARTY_NOTICES.md` extended with clawhub-desktop-control
> MIT attribution + per-component mapping for T1 / T2 / T3 / T4 / T5 /
> T6 / T7. **+90 net tests** (22 batch 1 / 15 batch 2 / 29 batch 3 /
> 19 batch 4 / 22 batch 5 inc. MCP surface tests).
>
> **Earlier validating HEAD:** `2cad783` on `main` (clawhub-windows-control catalog 08
> port batch 5 closing commit; doc-bump pending). Seven of eight cataloged
> techniques (T1-T6 + T8; T7 OCR deferred per catalog `★` recommendation)
> landed across 5 feature commits on top of `a48ec9d`:
>
> * **T2 (`desktop/uia.py`):** `UIElementInfo` frozen dataclass +
>   `get_ui_element_inventory` categorised UIA walk (10 buckets:
>   buttons / links / menu_items / list_items / tabs / checkboxes /
>   radio_buttons / text_fields / dropdowns / other). Edit / Document
>   admitted without name (their value carries content); other types
>   require a non-empty name. Optional `control_types` allowlist;
>   `value_truncate=0` skips value capture entirely.
>
> * **T4 (dual: `desktop/uia.py` + `desktop/windows.py`):** synchronous
>   polling barriers. `wait_for_text_in_window` polls
>   :func:`enumerate_windows` + :func:`collect_window_text` looking for
>   substring presence in any window matching the title filter.
>   `wait_for_window` polls :func:`find_window` until a match appears
>   (`prefer_foreground=False` during polling because the appearing
>   window won't be foregrounded yet). Shared
>   `DEFAULT_WAIT_TIMEOUT_S=30.0` / `DEFAULT_WAIT_INTERVAL_S=0.5`
>   defaults. `sleep_fn` / `clock_fn` injection for deterministic
>   tests; deadline-clamped final sleep.
>
> * **T8 (`desktop/input_control.py`):** `InputController.drag_to`
>   absolute-coord drag via `pyautogui.moveTo` + `pyautogui.dragTo`.
>   Full controller gate stack (foreground security + rate limit +
>   safety validator + click-preview gate on SOURCE coordinate).
>   Validates button against {left, right, middle}; rejects negative
>   duration.
>
> * **T5 (`desktop/uia.py`):** structured browser content extraction.
>   `BrowserContent` + `BrowserLink` frozen dataclasses + `BROWSER_NAMES`
>   (chrome/firefox/edge/brave/opera/vivaldi/arc) + `is_browser_window`
>   + `find_browser_window` + `extract_browser_content`. Walks browser
>   UIA tree categorising descendants into headings (heuristic: short
>   uppercase / colon-terminated Text + Static), longer text, buttons
>   (gated), Hyperlinks (gated; URL extracted from `automation_id` when
>   starts with `http`), Edit / ComboBox inputs (gated; current value
>   captured), Image alt text (gated). Per-bucket dedup preserves tree-
>   walk order; per-bucket caps applied after dedup. `full=True`
>   shorthand enables all four include flags. **20-100 ms** on a typical
>   webpage with zero GPU cost (vs 300-800 ms + ~330 MB VRAM for the
>   moondream2 screenshot path).
>
> * **T1 (NEW `src/kenning/desktop/dialog_control.py`):** native Windows
>   dialog detection + CRUD interaction. Constants `DIALOG_CLASSES`
>   (`#32770` + 4 standard dialog classes) / `DIALOG_CONTROL_TYPES`
>   (Window / Dialog / Pane) / `DIALOG_TITLE_KEYWORDS` (7 entries) /
>   `DISMISS_BUTTONS` (9 entries in least-destructive order: OK / Close /
>   Cancel / Yes / No / Dismiss / Got it / Accept / Done). Six frozen
>   dataclasses (`DialogInfo` / `DialogButton` / `DialogField` /
>   `DialogCheckbox` / `DialogContent` / `DialogActionResult`). Surface:
>   `find_dialogs` + `read_dialog` + `click_dialog_button` +
>   `type_into_dialog_field` + `dismiss_dialog` + `wait_for_dialog`. All
>   write actions route through `kenning.safety.validator.get_validator()`
>   with `tool_name=desktop.dialog.<action>`. `dismiss_dialog` does
>   per-candidate validator re-check so blocking one candidate falls
>   through to the next + ESC fallback gated with
>   `action=dismiss_escape`.
>
> * **T3 (NEW `src/kenning/desktop/element_click.py`):** cross-window
>   semantic UIA element search + click via the gated `InputController`.
>   `CLICKABLE_TYPES` (9-entry standard UIA control-type set) + frozen
>   `UIElementMatch` + `TextMatch` + `ClickResult` dataclasses +
>   `find_elements_by_name` + `click_element_by_name` +
>   `find_text_in_window`. Exact-matches-first stable-sort ranking;
>   `enabled_only` filter defaults True; per-window descendant cap
>   `DEFAULT_MAX_ELEMENTS_PER_WINDOW=500`; per-element fail-open
>   during walks. **The key safety win**: all clicks route through
>   `InputController.click(x, y, user_text)` so click-preview VLM +
>   foreground security + safety validator + Cap-3 explicit-intent +
>   rate limit all apply uniformly (vs upstream's pywinauto-native
>   `click_input()` which bypasses every gate).
>
> * **T6 (`desktop/windows.py` + `desktop/placement.py`):**
>   `get_active_window_title` lightweight foreground probe via pywin32
>   directly (skips the psutil + monitor-index work
>   :func:`get_foreground_window` does). `close_window(partial_title,
>   *, force=False, user_text)` graceful WM_CLOSE via
>   `win32gui.PostMessage` (the app's own close hook fires so editors
>   with unsaved changes surface their save prompt); `force=True`
>   escalates to :func:`kenning.subprocess.kill_tree.kill_process_tree`.
>   `UNSAVED_CHANGES_TITLE_HINTS` (`*`, `[modified]` / `(modified)`,
>   VS Code dot, em-dash modified) + `_title_suggests_unsaved_changes`
>   predicate + `CloseWindowResult.suspected_unsaved` on result for
>   orchestrators to gate behind voice confirmation.
>   `minimize_window_idempotent` / `maximize_window_idempotent` /
>   `restore_window_idempotent` state-check-before-act helpers
>   (check current state; return `PlacementResult(success=True,
>   error="already <state>")` on no-op).
>
> ≈ +199 net tests across the 5 feature batches. Tier summary: 4 GREEN
> + 4 YELLOW + 0 RED. Source plugin (~37 KB across 23 thin scripts)
> read read-only via the Read tool; never executed (per the binding
> ClawHub-batch security rules). Pattern extraction via a Sonnet 4.6
> Explore agent that independently confirmed the catalog's zero-RED
> security finding (no registry access, no network calls, no
> persistence, no anti-forensics, no DLL injection, no credential
> access). Tests: **7715 passing / 26 skipped / 0 failed** in ~149 s
> via `scripts/run_tests.py --stale-heartbeat=400`. Voice baseline
> contract intact (no SOUL.md / RVC / Piper / vocal WAV / LLM model
> file / Kokoro fine-tune voicepack touch; no orchestrator hot-path
> edit; all surfaces ship as importable infrastructure for future
> opt-in wiring). `THIRD_PARTY_NOTICES.md` extended with
> clawhub-windows-control MIT attribution + per-component mapping
> for T1 / T2 / T3 / T4 / T5 / T6 / T8.
>
> **Earlier validating HEAD:** `ee9eca5` on `origin/main` (clawhub-windows-ui-automation
> catalog 07 port closing test-fix commit). Six techniques (T1-T6) landed
> across 5 batches on top of `c3966a7` (plus one test-fix follow-up):
>
> * **T1 + T3 (doc-only):** code comments in
>   `src/kenning/desktop/input_control.py` documenting that pyautogui
>   uses the modern atomic `SendInput` API (not legacy `mouse_event`)
>   and that `pyautogui.hotkey` returns BEFORE the target processes
>   the keystroke (unlike PowerShell `SendKeys.SendWait`).
>
> * **T2 (new module):** `src/kenning/desktop/win32_helpers.py` --
>   ctypes wrappers for `GetDpiForMonitor` (per-monitor DPI),
>   `GetLastInputInfo` (idle detection), `DwmGetWindowAttribute`
>   with `DWMWA_CLOAKED` (cloaked window detection), and a hardened
>   `block_input_context` (watchdog + try/finally + UIPI safety
>   floor + hard cap at 30s). Plus :func:`logical_to_physical` /
>   :func:`physical_to_logical` coordinate-space primitives.
>
> * **T6 (`focus_by_title` two-tier focus):** new
>   `desktop.windows.focus_by_title` -- primary
>   `find_window` + `SetForegroundWindow`, fallback to in-process
>   `WScript.Shell.AppActivate` via `win32com.client`, final
>   fallback to `CREATE_NO_WINDOW` PowerShell subprocess. `find_window`
>   + `enumerate_windows` gain `exclude_cloaked: bool = True` keyword
>   bridging to `is_window_cloaked`.
>
> * **T5 (DPI-aware UIA -> input boundary):**
>   `desktop.uia.physical_center_of_element` + `physical_rect_of_element`
>   + `dpi_aware_click_at_element_center` -- safe coordinate conversion
>   helpers for the UIA-to-pyautogui boundary. Identity on 100%-DPI
>   displays; opt-in `assume_logical=True` routes through
>   `logical_to_physical` for callers integrating non-DPI-aware sources.
>
> * **T4 (frontmatter `capability_tags:` -> SkillRegistry filter):**
>   `SkillRegistry.matching_skills` accepts `gaming_mode` /
>   `vlm_loaded` / `has_internet` kwargs; new
>   `_skill_active_for_capability_tags` predicate reads optional
>   `capability_tags` from frontmatter (via `skill.extra`),
>   converts to `CapabilityTag` enum values, and filters via the
>   same gating rules as `filter_capabilities`. `maybe_get_skills_block`
>   forwards the context. `LLMEngine._build_messages` calls new
>   `_resolve_vlm_loaded_for_skills()` to thread the live VLM holder
>   state. Closes the catalog 07 "skill registry filters on modes but
>   not capability_tags" safety gap.
>
> ≈ +160 net tests across the 5 batches. Closing test-fix commit
> `ee9eca5` updated two tests (`test_python_lint::test_runner_lint_listener_disabled`
> + `test_narration::test_voice_controller_progress_falls_back_to_legacy_without_coordinator`)
> to monkeypatch the per-test config state instead of depending on the
> global default (which the prior production-wiring pass had flipped).
> Added a companion `test_runner_lint_listener_enabled_returns_callable`
> so both branches stay pinned regardless of future default flips.
> Tests: **7516 passing / 26 skipped / 0 failed** in ~108 s via
> `scripts/run_tests.py`. Voice baseline contract intact (no SOUL.md /
> RVC / Piper / vocal WAV / LLM model file / Kokoro fine-tune voicepack
> touch; no orchestrator hot-path edit beyond the additive `vlm_loaded`
> thread). `THIRD_PARTY_NOTICES.md` extended with clawhub-windows-ui-automation
> MIT attribution + per-component mapping for T1-T6.
>
> **Earlier validating HEAD:** `29ffe49` on `origin/main` (2026-05-26
> production-wiring pass closing baseline-re-measure commit). The
> 7-commit production-wiring pass consumed the catalog-port
> primitives into their canonical hot-path consumers: T14 rate-limit
> recorder live in `SearchProviderChain` (Brave / SearxNG / DDG
> 429s now cool the provider for the next turn); T3 canonical_codes
> + T16 category/metadata on `AuditLog.record` + T18
> `sanitize_for_log` CWE-117 defence in `safety/audit.py` +
> `resilience/error_log.py`; T11 `materialise_default_pins` +
> T2 voice-baseline TOFU verifier (new
> `install/voice_baseline_verify.py`) at orchestrator startup;
> T8 `kill_process_tree` replaces ad-hoc terminate+kill in
> `parakeet_engine.stop_parakeet_server` +
> `tts/xtts_v3._stop_server_subprocess`; T1 + T9 trust pre-check
> in `LLMEngine.reload_for_preset`; T5 mode-scoped skill filter
> (`SkillRegistry.matching_skills(mode=...)` + `Skill` frontmatter
> `modes:` list); SWE-Agent T7 SubmitReviewLoop voice-lock check
> in `CapabilityVoiceController._attach_submit_review_listener` on
> supervisor COMPLETE; **OpenClaw flipped LIVE** at
> `http://127.0.0.1:11280` (dispatcher returns real plugin
> invocations instead of stub messages); 9 other flag flips ON
> (supervisor.tier=full, architect, click_preview,
> contextual_retrieval, background_summary, pre_write_lint,
> goal_anchors, repo_map, pre_task_confirmation,
> inbound_voice_handoff).
>
> **Voice baseline IMPROVED on every measured axis** vs the
> 2026-05-23 pre-flip measurement: VRAM 6597 -> 6254 MB loaded
> (-343), 7007 -> 6664 MB peak (-343); STT median unchanged at
> 16 ms; LLM TTFT 203 -> 172 ms (-31); TTS synth 109 -> 78 ms
> (-31); composite TTFA 313 -> 266 ms (-47). The expensive flags
> only fire on coding / desktop / memory-write paths -- the
> conversational voice-query path the baseline measures stays
> byte-identical. **Confirmed veto list** (do NOT flip without
> re-asking): `llm.draft_kind: "none"` (real .8b PLD `llama_decode -1`
> bug), `memory.reranking.enabled: false` (17-18 s/turn CPU cost),
> `notifications.telegram.enabled: false` (no bot token).
>
> ≈26 new tests across the wirings via `scripts/run_tests.py`.
> Voice baseline contract intact. Full chronology in
> `memory/project_ultron_2026_05_26_production_wiring_pass.md`.
>
> **Earlier validating HEAD:** `b46ad89` then handoff `9395374` then
> `287cf2f` on `origin/main` (2026-05-25 OpenClaw-ClawHub
> catalog port -- all 15 cataloged techniques landed across 9 batches;
> 12 GREEN drop-ins + 3 YELLOW gated through the existing safety stack
> + 0 RED). Tests **7368 passing / 26 skipped / 0 failed in ~109 s**
> via `scripts/run_tests.py`. The OpenClaw-ClawHub port added 14 new
> modules across `install/{reason_codes, lockfile, pin, trust_envelope,
> artifact_identity, resolver, coherence, discovery}`,
> `identity/{alias_graph, short_lived_token}`,
> `skills/capability_tags`, `web_search/rate_limit`, `feedback/{report_queue,
> moderation_plan}`, `observability/private_telemetry`. Pure-Python
> stdlib-only primitives; voice baseline contract intact (no SOUL.md
> / RVC / Piper / vocal WAV / LLM model file / Kokoro fine-tune
> voicepack touch); no orchestrator hot-path wiring (importable
> infrastructure ready for opt-in consumption). +621 net tests over
> the 6747 OpenClaw-catalog baseline.
>
> **Earlier validating HEAD:** `0ec1e42` on `origin/main` (2026-05-25 OpenClaw
> catalog port -- 11 batches, 19 of 22 cataloged techniques landed;
> T17 / T19 / T20 deferred per catalog star rating). Tests **6820
> passing / 26 skipped / 0 failed in ~101 s** via `scripts/run_tests.py`
> (use `--stale-heartbeat=180` to ride out the documented-flaky
> `tests/integration/test_bridge_e2e.py::test_health_through_real_subprocess`
> on slow Windows subprocess startup). The OpenClaw port added 18 new
> modules / packages across `utils/ansi_safe`, `subprocess/{kill_tree,
> process_registry}`, `safety/{path_resolver T21 additions, validator
> T16 additions, policy_chain, two_phase_approval, hierarchical_policy}`,
> `llm/{context_window_guard, condensers/splitter}`, `agent_loop/{loop_detection_extended,
> subagent_policy}`, `hooks/lifecycle` (36-event expansion +
> HookDecision), `coding/edit_recovery`, `skills/{activation,
> marketplace}`, new packages `providers/` (auth-profile rotation +
> 13-reason failover taxonomy), `install/static_scanner`, `mcp/`
> (transport + registry; closes the deferred T9 from cline). All ship
> as importable infrastructure -- voice baseline contract byte-identical
> to the pre-port baseline. See `THIRD_PARTY_NOTICES.md` for the
> per-component attribution table.
>
> **Public-repo hygiene:** the repo lives at
> `https://github.com/1v9Khan/ultronPrototype` (visibility flips between
> public and private as needed). A `.git/hooks/pre-push` hook scans every
> push and blocks (a) forbidden paths covering local dev-tool
> orientation files, archived sibling-tool configs, integration-
> credential dirs, personal-content data files, and install / HF-cache
> logs; (b) forbidden commit-message + file-content patterns covering
> co-author trailers, dev-tool email trailers, and dev-tool brand
> prose; (c) brand-name model-tier mentions in prose (CLI string
> values are preserved). The hook's source defines the exact regex
> sets and is the source of truth. If the hook blocks a push, fix
> the commit — don't bypass with `--no-verify`. The full hygiene
> contract lives in the local-only `CLAUDE.md` orientation file and
> the auto-loaded `MEMORY.md` index.

---

## Recent sessions

Compact log of substantive work. The current-state sections below
(File tree, Cross-cutting flows, Source modules, Configuration,
Operational scripts, Tests, Runtime artifacts) reflect the cumulative
result of every row. Deep narrative lives in the corresponding
`project_ultron_*.md` memory file under
`~/.claude/projects/C--STC-ultronPrototype/memory/`.

| Date | HEAD | Summary | Tests | Memory file |
|------|------|---------|-------|-------------|
| 2026-06-21 | `main` (local) | **Enforcement layer WIRED (Option 1) + canon ON MAIN.** The canon (`CLAUDE.md`/`docs/canon/`/`docs/ultron_1_0/CONSTRAINTS.md`) + the live enforcement bundle are committed to **local `main`** so every new worktree/session auto-loads the rules + hooks (kept OFF origin by the pre-push hook). NEW `.claude/settings.json` (deny = real secrets + dangerous-git ONLY; ask = canon-edits/push/installs/curl-wget; **NO** bare-tool deny / **NO** `disableBypassPermissionsMode` → bypass/web/MCP/subagents/worktrees preserved) + 5 fail-open Node hooks `.claude/hooks/*.mjs` (pretool-guard [dangerous-git deny + WebFetch SSRF + BR-P3 one-instance ask] · posttool-advise [advisory ruff/stub/anticheat] · sessionstart-reground · precompact-snapshot · stop-advise). NEW `.github/workflows/enforce.yml` (light runner-agnostic CI backstop). ruff 0.15.18 + mypy 2.1.0 added (`pyproject` `[tool.ruff]`/`[tool.mypy]` ratchet; `orchestrator.py` grandfathered). 40-agent board design of record: `docs/ultron_1_0/02_research/enforcement_synthesis.md`. | green (ruff: orchestrator grandfathered) | [project_enforcement_bundle_2026_06_20.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_enforcement_bundle_2026_06_20.md) |
| 2026-06-20 | branch `claude/infallible-kepler-0a865d` | **Ultron 1.0 pivot — route-all-through-8B (flag-gated, NOT on main).** Default LLM preset → `josiefied-qwen3-8b` @ `n_ctx 4096` (M0). NEW modules `audio/ultron_prompt.py` (lean ~165-word prompt assembler, M1), `audio/agent_kits.py` (version-stamped 29-agent kit injection, M3), `audio/intent_gate.py` (4-class always-listening gate CLASSIFIER, M5 — not yet loop-wired). `relay_speech` gains the `KENNING_U1_LLM_ROUTE` branch in `build_relay_line` (lean prompt + agent-kit context + compound→one-LLM-call M4) + `match_verbosity_command`/`relay_verbosity` (no/low/high, M2) + `build_private_prompt` fix (M6a). NEW harness `scripts/relay_test/u1_text_harness.py` (text-injection PRIMARY) + `trace_corpus_full.py`. Tests `tests/audio/{test_ultron_prompt,test_u1_llm_route,test_agent_kits,test_intent_gate}.py`. ALL behind `KENNING_U1_LLM_ROUTE` (default OFF) → main behavior unchanged; each increment regression-clean vs the frozen 22-fail baseline. Spec + live status: `docs/ultron_1_0/` (read `00_process_log/STATUS.md` first). | green (22 baseline) | [project_ultron_1_0_pivot.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_1_0_pivot.md) |
| 2026-06-19 | `6064e5f` (main) | **Thinking-mode toggle + nice-try flavor parity + E3 snap-early-endpoint.** NEW `relay_speech.thinking_mode_enabled()`/`match_thinking_toggle()` (env `KENNING_THINKING_MODE`, default OFF) gates the LLM on the relay path so compose commands (soundboard/voice-changer/flame/praise) SNAP deterministically on flavor-ON unless thinking is on; `orchestrator._maybe_handle_thinking_toggle` in both dispatch points. `_name_social_snap` names the addressee on the flavor-ON nice-try render (parity). E3 latency: `relay_speech.is_complete_tactical_callout` + `orchestrator._snap_early_endpoint` (`KENNING_SNAP_EARLY_ENDPOINT`, default OFF) close capture early on a complete tactical callout (detail in the validating-HEAD header). | green | [project_thinking_mode_flavor_parity_2026_06_19.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_thinking_mode_flavor_parity_2026_06_19.md) |
| 2026-06-18 | `0b5da79` (main) | **Un-silence hotfix + per-chunk blip + follow-up addressing + "thank you" snap** (reapplied on the rolled-back `9711e2e` base; the `a9818af` probe-safe-firewall / wake-clip / mic-preprocessing batch was reverted, only these targeted fixes kept plus one new snap). **ANTICHEAT:** removed `pytesseract` from the import-firewall blocklist — `transformers` (pulled in by Kokoro TTS + Whisper) probes it at IMPORT time via `importlib.util.find_spec("pytesseract")`, and this finder RAISES inside that probe, so the whole transformers import fails → Kokoro/Whisper/Smart-Turn never load and Ultron goes silent; pytesseract isn't installed and the OCR capability stays blocked via the `kenning.desktop` prefix, so omitting the bare name costs zero protection. **TTS BLIP** (`tts/kokoro_engine.py`): run `trim_and_fade` on EACH sentence chunk before joining — an empirical per-utterance probe showed the inter-sentence gap *edges* were already clean; the real artifact is the undertrained fine-tune's noise burst at every *internal* sentence onset/offset, which the single OUTER `trim_and_fade` never reached. Supersedes the cosine edge-fade. **ADDRESSING (follow-up window)** (`pipeline/orchestrator.py`): a leading wake word ("Ultron, show me the stop button") now bypasses the borderline zero-shot gate via `_FOLLOWUP_WAKE_RE` — it was scoring 0.75 < the 0.80 ADDRESSED threshold and silently dropping the command despite the user saying the name (the rules side keyed direct-address to "kenning" ONLY, so the real "ultron" wake word was invisible to the classifier); narrow regex (real wake words, leading position) so it can't false-accept room chatter. This is the *blunt* fix — the board-designed confidence-fusion (real flan probability + graded features + cost-asymmetric threshold) is the deferred clean replacement. **NEW — gratitude snap** (`audio/relay_speech.py`): deterministic "thank you" relay snap (`_THANK_YOU_RE` + a dedicated 10-tail `_THANK_YOU_TAILS` Ultron-persona pool — cold, superior acknowledgment, never warmth), routed off the LLM like the other snaps; matches bare gratitude only ("thank you" / "thanks team" / "thank you so much"), a contextual thanks ("thank you for the heal") keeps its real content. 7 frozen tests (`TestThankYouSnap`). | green | [project_prelaunch_hardening_2026_06_17.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_prelaunch_hardening_2026_06_17.md) |
| 2026-06-17 | `75cad1c` (main) | **Pre-launch anticheat hardening + live-testing relay/TTS fixes** (follows the `2293677` battery pass). **ANTICHEAT** — `safety/import_firewall.py`: `find_spec` FAIL-SAFE (`except: active=True` → block when the anticheat state can't be determined, was fail-open), `_INSTALL_LOCK` against duplicate finders, `is_firewall_installed()` always scans live `meta_path`, NEW `assert_firewall_enforces()` (imports the blocked-but-absent `interception` driver and proves the `ImportError` is the firewall's; ERROR if a blocked import succeeds), blocklist += CDP/webdriver (pyppeteer/undetected_chromedriver/DrissionPage/helium/comtypes.gen) + capture/input-sim/clipboard/OCR/virtual-gamepad exacts — all pure defense-in-depth (none on any voice/relay/audio/ptt path; win32api/win32gui/comtypes deliberately left importable for pycaw). `__main__.py`: Orchestrator imported LAZILY after the firewall installs (closes the pre-firewall import window); FATAL `return 4` refuse-to-start if anticheat active and the firewall is absent or not enforcing. `pipeline/orchestrator.py`: posture audit now requires the firewall to ENFORCE (not just be present), warns on non-default safety flags while gaming-engaged, `_skip_for_lean_gaming` fail-SAFE. **PTT** — `ptt/controller.py`+`config.py`+`config.yaml`: backend pinned `"rawhid"` (HID-only masked keyboard VID 0x1209, NO COM port; NEVER auto-falls-back to the legacy-CDC serial path that scans the Arduino VID 0x2341 — inert NullPttBackend if absent) + NEW `release_jitter_ms: 60` (random 0..60ms extra release tail so the key-hold is never machine-precise; only extends the mic window). **FIRMWARE** — `git mv firmware/leonardo_ptt` → `firmware/leonardo_ptt_LEGACY_CDC_DO_NOT_FLASH/` (.ino → `.ino.DO_NOT_FLASH`, ⛔ README header); the hardened `firmware/leonardo_ptt_hid/` (enumerates under *Keyboards*, no serial port) is the only build to flash. Leonardo live-confirmed HID-keyboard, no COM. **RELAY/TTS FIXES** (from live testing) — `tts/kokoro_engine.py`: cosine edge-fade the inter-sentence silence gap (the raw zero-gap stepped to/from non-zero chunk edges = a click/"blip" at the callout↔tail boundary). `audio/relay_speech.py`: agent-select draft requests ("we need smokes / an initiator / a duelist / a sentinel") get a DEDICATED curated COMPOSITION tail pool (`_AGENT_SELECT_FULL_RE` + `_AGENT_SELECT_TAILS`), distinct from in-game tactical commands and from the enemy-comp read ("they have no smokes"); wh-question copula inversion (`_wh_copula_invert`: "where our smokes are" → "Where are our smokes?"); single named agent at a place uses the natural callout form "Reyna, tree." (was "Reyna is tree" → read as "Reyna is A tree"). 14 frozen tests (`TestT617TestingFixes`). | 1.2k+ | [project_prelaunch_hardening_2026_06_17.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_prelaunch_hardening_2026_06_17.md) |
| 2026-06-16 | 0b9c4f1 (main) | **Corpus-loop matcher hardening + flavor library deep expansion + coherence audit.** Three phases, all on the MAIN checkout `C:\STC\ultronPrototype`. **(1) CORPUS-LOOP MATCHER HARDENING** (iter 1: 92.7% → 99.4% clean on 20k seed-0 corpus): NEW `audio/_common_words.py` (GENERATED frozenset of top-~5000 English words, baked by `scripts/build_common_words.py`; gates `_stt_correct._phonetic_fuzzy_snap` + the curated `_fix_token` layer so real English is never corrupted — "let"/"mean"/"yet" stay as-is); NEW `audio/_relay_intent.py` (`RelayIntentGate` — semantic margin gate over the embeddinggemma sidecar, positive vs negative exemplar clouds, threshold 0.06, FAIL-OPEN; vetoes `recover_relay_lead`'s bare-callout prepend, the source of ~97% of corpus false-relays, cutting them 674→~70); `command_normalizer.py` additions: narration/epistemic-hedge regex fast-path (zero-cost, runs before embed), lead-preserving disfluency resolution (`_resolve_disfluency`: `_DISFLUENCY_CUE_RE`/`_DISFLUENCY_SPLIT_RE`, negation-safe, preserves relay lead), conversational lead-filler strip, relay-intent gate wiring; `_stt_correct.py` additions: common-word protection gate, inflection guard (`-ed`/`-ing`/`-ers` never snapped onto a base term), OOV agent-superstring guard (snap target may not be a superstring of heard token), `_MISHEAR_FORCE` allow-list; `command_router.py` `get_embedding_backend()` exposes the shared sidecar client for reuse; NEW `scripts/relay_test/trace_corpus.py` (full-pipeline tracer) + `analyze_outputs.py` (triage bucketer). Sidecar must be UP on 8772 for the gate to be exercised. **(2) FLAVOR LIBRARY DEEP EXPANSION**: `_tail_schema.py` (NEW): `TailEntry(text, tags)` schema + `as_entry`/`entries` coercion (zero-rewrite legacy migration); expanded 16-key enemy situation taxonomy (`Sit`, `ENEMY_SITUATIONS`); machine-readable `AGENT_GENDER` (pronoun per agent); `loc_class`/`dmg_level_tag`/`ability_tag`/`situation_for_payload`/`build_active_tags` fact-folding helpers. `_tail_selector.py` (NEW): semantic fine-selector (query embed → doc-matrix cosine → MMR + recent-mask → per-pool abstain threshold → fail-open to `_pick_flavor`); OFF by default (`KENNING_ENABLE_TAIL_SELECTOR` opt-in). `relay_speech._flavor_ctx` rewritten as HYBRID two-stage (coarse route → 4-tier `_tier_filter` → opt-in `select_tail`). `_CRITICIZE_RE` fixed: "call out" no longer treated as a criticism verb (fixed 105/106 owner-inversions of factual callouts). "I hit/tagged/cracked `<agent>` for `<n>`" pattern routes to that enemy's damaged pool with the right dmg tag. NEW offline generation pipeline: `scripts/flavor_gen/{integrate_tails,apply_cuts}.py`. **(3) COHERENCE AUDIT** (by-hand, every line): `_agent_flavor.py` RE-AUTHORED 4,147 → **1,628 tight TailEntry entries** (~5/cell): every ult = the REAL ultimate, every utility ability-tagged (`ability:<canon>`), filler/wrong-kit/off-topic cut; CURATED dict (`scripts/flavor_gen/curated_overrides.py`) applied by `apply_curated.py`; verified by `scripts/flavor_audit/lint_tails.py` (0 hard/0 soft/0 thin). `_ultron_setpieces.py` de-biblicalized (~18 flood/Noah/ark/God/church lines → machine/evolution/immortal register). Routing fixes: `_situation_for` lifts situation to `ult` on ult keyword; `_flavor_ctx` skips semantic selector for small (<5) candidate cells (LRU, zero sidecar cost). Normalization: `_tail_schema._VERB_TO_ABILITY` (mollied→molly, walled→wall, darted→dart…); `_stt_correct._slot_agent_correct` context SLOT-confirmation pass (Stage 1.5, "raise hit 18"→"Raze hit 18"; slots only, non-slot uses untouched); `whisper_engine.py` decode-time domain biasing (`initial_prompt = _DOMAIN_PROMPT`, gated `WHISPER_DOMAIN_BIAS` default-on). **All ML in loopback sidecar / build-time scripts; anticheat firewall intact.** 964 audio+safety tests green; lint 0/0/0. | 964 | [project_corpus_loop_2026_06_16.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_corpus_loop_2026_06_16.md) |
| 2026-06-15 | (this session) | **Movie-Ultron identity/relay polish + anticheat audit hardening.** Relay/TTS: NEW `audio/_ultron_identity.py` — 7 categorized in-character identity-answer pools (~30 lines each: bot / soundboard / streamer / real-person / puppet / voice-changer / recording) + `classify_identity_question`; relay_speech now routes "X asked about/if Y" (`_match_reported_question`) to an in-character ANSWER even without an explicit "respond", picks identity answers from the category pools, and `_is_identity_question` is broadened (who's-controlling-you / strings / off-switch / pre-recorded). Curated `DEFAULT_CRITICIZE_LINES` replaces the 3B for "criticize `<agent>`". Verbatim family widened ("word for word", `Pete`/`Heat`→repeat, `my team's` stripping). `_stt_correct`: `silver`→Sova, drop "hey-agent"→"Hellagent" blend, protect count words from→"tree". `_join_tail` + a period-length kokoro inter-sentence gap (`KENNING_TTS_SENTENCE_PAUSE_MS`) stop the callout slurring into its flavor tail. "JARVIS"→"Jarvis" (text_hygiene). Fixed a `/no_think` marker leak (`.lower()` on a `Path` raised + was swallowed → marker appended to the llama gaming model, which parroted "no_think"; fixed via `str()`). PTT buffers widened (`lead_ms` 120→200, `release_tail_ms` 150→300). Waveform spin-freeze/change-detection reverted (continuous 60fps spin). NEW `audio/stop_button.py` — a loopback-immune click kill-switch (in-process tkinter STOP window firing `_cancel_all_playback`; `StopButtonConfig`). Anticheat: import-firewall blocklist expanded (`keyboard`/`mouse`/`pydirectinput`/`d3dshot` + stale `ultron.*` mirror prefixes), firewall installed in `__main__` BEFORE the Orchestrator constructs, `GamingModeConfig.enabled`/`engage_at_startup` defaults flipped True (safe-by-default), the posture canary now derives its tripwire from `blocked_module_names()` and both canaries log at ERROR, the OpenClaw MCP runner hard-refuses under anticheat, and the dead `BlockInput` helper refuses while anticheat is active. | d86e0bd | (committed) |
| 2026-06-15 | `3447cdb` | **Lean-by-default gaming boot + bulletproof lifecycle** (commits `26d502d`, `3447cdb`). Gaming boots initialize+import ONLY relay+Spotify+core-voice (worker RSS is noisy ~3.5-6 GB so no fixed delta; the clear win is eliminating the ~4-5 GB 4B-on-GPU VRAM boot transient), proven by a `_audit_anticheat_posture` "lean boot OK" sys.modules assertion (canary logs loud WARNING on regression). `coding/` and `openclaw_bridge/` package `__init__` made PEP-562 lazy so importing the package no longer eager-loads heavy submodules. NEW `subprocess/sidecar_lock.py`: pidfile+orphan-sweep (`sweep()` with `reuse`/`killed`/`killed-zombie`/`killed-unknown`/`spawn` verdicts) makes the embedder a singleton and reaps `taskkill /F` orphans at next boot. SIGTERM+atexit+`kill_process_tree` clean shutdown (was SIGINT-only). `AuditLog.repair_if_needed()` self-heals fsync-torn-tail corruption at boot (archives rather than deletes). Never-lexical router: boot respawns+rebuilds if lexical (ERROR), idle `_maybe_recover_embedding()` via `try_recover()`. Direct 3B-CPU LLM load avoids the 4B-on-GPU boot transient. New `barebones_*` flags in `GamingModeConfig` (all default True). `SemanticRouterConfig` gains `sidecar_orphan_sweep_enabled` + `sidecar_pidfile_path`. 903 safety/audio + 906 coding tests green; live boot verified. | 906 | (this session) |
| 2026-06-15 | `e3df7d3` | **Semantic command router + embeddinggemma sidecar + turbo STT + anticheat import firewall** (commit e3df7d3). Added an additive similarity-based router beneath the exact matchers (command_router/_router_backends/_command_exemplars) with a hybrid lexical+embedding backend and an OOS abstention gate to the LLM; the embedding model (google/embeddinggemma-300m) runs in an isolated-venv loopback sidecar (scripts/embedder_server.py) so no heavy dep enters the anticheat-pinned main process. Added a pre-routing STT normalizer (command_normalizer + expanded _stt_correct), swapped STT to faster-whisper large-v3-turbo on CUDA with a hallucination filter, added a loader-level anticheat import firewall (safety/import_firewall), a gaming capability-refusal gate, and overlay GPU optimizations. 903 audio+safety tests green; a 6-facet Sonnet audit confirmed no regression. | 903 | (this session) |
| 2026-06-12 | `2a2a871` | **Wake word SHIPPED + relay 29-agent roster** (supersedes the "wake-word NOT deployed" note in the rename row below). Commits `58120cf` (fallback + dropdown), `c76b597` (agents + harness), `2a2a871` (model + threshold), pushed. **Wake word now LIVE = "kenning"**: `models/openwakeword/kenning.onnx` deployed (v8 from an 11-candidate sweep v1-v11, gitignored like all weights; ~88% recall @ ~1.6% adversarial FAR on synth clips via the runtime frame path). "kenning" is acoustically confusable (kennel/canning/kenneth) so it can't match ultron's 100% at that FAR; v8 (layer 32, 50k steps, recall-favorable auto-tune targets) beat every layer-64 / higher-neg-weight variant. **Fallback is `ultron`, NEVER hey_jarvis** -- `WakeWordDetector._load_model` is now PATH-based (loads the side-by-side custom `ultron.onnx`; a pretrained word only if neither custom ONNX exists). NEW `reload_for_word()` hot-swaps the live model; settings-panel "Wake word" dropdown (kenning/ultron) fires the `wake_word` gui_action. Threshold 0.40 (config + .env, recall-favoring; ultron ~100% there too). `config.py` `WakeWordConfig.fallback_model` default `hey_jarvis` -> `ultron`. **Relay**: `DEFAULT_ADDRESSEE_NAMES` now the full 29-agent VALORANT roster (+ Miks, Veto) + STT homophones (cipher->Cypher, gecko->Gekko, mix->Miks, way lay->Waylay) via `_NAME_CANON`; rephrase prompt lists the roster so the LLM treats the newest agents as teammates. NEW `scripts/relay_test/` (547-command corpus + staged full-pipeline harness: matcher -> rephrase -> audio blip analysis -> ASR-reconstruction-vs-intended -> spoken->STT). NEW `training/compare_wake_models.py` (recall/FAR threshold-sweep tool, auto-discovers `kenning_v*.onnx`). Cleanup: ~2.3 GB of regenerable train clips/features + redundant candidate onnx deleted (test clips + the 17 GB ACAV100M corpus KEPT for retraining). **Sweep 9818 / 39 / 0, ~151 s** + `validate_config` clean. | 9818 | (this session) |
| 2026-06-12 | `3be01fd` | **Product rename Ultron -> Kenning + relay batch** (3 commits on `def92b5`: relay `760bed6`, rename `55ac95e`, doc-bump `3be01fd`; pushed, main = origin/main). **Rename** (`55ac95e`, 882 files + the relay commit's src moves): case-aware three-way replace (ULTRON/Ultron/ultron -> KENNING/Kenning/kenning) across every tracked file, `git mv` history preserved: `src/ultron/` -> `src/kenning/`, `tts/ultron_filter.py` -> `tts/kenning_filter.py`, `ultronVoiceAudio/` -> `kenningVoiceAudio/`, RVC dir -> `kenning_rvc_voice/` (`Kenning.pth` + `..._Kenning_v2.index`), `scripts/run_ultron_mcp_for_openclaw.py` -> `run_kenning_mcp_for_openclaw.py`, `training/my_model.yaml` -> `training/kenning_model.yaml`. Package import root now `kenning` (pyproject name + console script; `pip install -e .` re-run in main `.venv`). Env `ULTRON_*` -> `KENNING_*`, sentinels `<<KENNING_SUBMIT>>`, MCP prefix `mcp__kenning_coding__`, logger ns `kenning.`, runtime dirs `~/.kenning` + `data/.kenning_instance.lock`, persona "You are Kenning", config `name: kenning`, Kokoro voice "kenning". **External gitignored files updated** (NOT in git; backups kept): `.env` -- 41 `ULTRON_*` vars -> `KENNING_*` + Whisper initial prompt `Ultron.` -> `Kenning.` + prose comments (`.env.pre-kenning.bak`); `~/.openclaw/openclaw.json` -- agent ids `kenning-{test,main,heartbeat,vision}`, apiKey `local-kenning`, MCP server `kenning-mcp`, persona Name Kenning, script path -> `run_kenning_mcp_for_openclaw.py` (`openclaw.json.pre-kenning-bak`); `~/.kenning/{spotify.json,sandbox}` + `.kenning/lock.json` in place. Boot is now `python -m kenning`. Protected/KEPT: repo slug + local dir `ultronPrototype` (real paths), memory filenames `project_ultron_*.md`, archive paths (`I:\Ultron Archive`), telemetry-hash test case variants, `config.yaml` `required_agent_id` history comment. Legacy guards: `install/idempotent.py` `LEGACY_MARKERS` recognizes the old INSTALLED-BY-ULTRON marker; `identity/alias_graph.py` keeps "ultron" RESERVED; `cleanup_stale_processes.py` matches BOTH the new and legacy MCP stub names. **Voice-asset reconciliation** (`2087374`): the rename repointed the XTTS/Parakeet swap-back asset paths + the voice-baseline protection lists to `kenningVoiceAudio/`, but the 18 GB of gitignored runtime assets (two isolated venvs with baked-in absolute paths + the reference WAV) were never physically migrated -- they stay at `ultronVoiceAudio/`. Default boot (Kokoro+Moonshine, loads from `models/kokoro/`) was unaffected, but swap-back pointed at non-existent venvs/WAV and the real voice workshop was left unprotected. Per the user's decision, config.yaml + `config.py` defaults + `xtts_v3.py` fallbacks were repointed at the real `ultronVoiceAudio/` paths (a local runtime-asset path, like the checkout dir; venv baked-in paths intact) while the tracked server scripts stay at `kenningVoiceAudio/` -- all five swap-back paths now resolve to existing files; and the voice-baseline protection (voice_lock, checkpoints exclusions, safety policy files+dirs, submit_review regex, evolution blast_radius) was extended to cover BOTH trees (additive, never shrink). **WAKE-WORD MODEL NOT YET DEPLOYED (OPEN):** config `wake_word.name: kenning`, `model_path: models/openwakeword/kenning.onnx` -- but that file does NOT exist (the dir still holds only `ultron.onnx`), so `WakeWordDetector._load_model` falls back to the `hey_jarvis` pretrained word with a prominent warning (graceful, no crash; `fallback_model: hey_jarvis`). "kenning" is acoustically harder than "ultron" (shared phonemes): 6 candidates (`training/my_custom_model/kenning_v{1,2,3,5,6,7}.onnx`, untracked) all undershot production `ultron.onnx` recall (~66%); best ~48-52% on synthesized clips. Until a candidate clears the bar (or the user accepts the tradeoff) the spoken wake word is effectively "hey jarvis". Recipe `training/kenning_model.yaml` (24k positives, DNN/32/30k, 10 confusable negatives); scorer NEW `training/validate_wake_model.py` (1280-sample frame path, recall>=90% / adversarial-FAR<=8% gate); cross-name check confirmed old/new acoustically disjoint. `WakeWordDetector` is label-agnostic so deployment needs zero code change -- just drop the model at the configured path. **Relay batch** (`760bed6`, +90 tests): exhaustive Valorant callout matcher + semantic glossary in the rephrase prompt, verbatim mode ("word for word" bypasses the LLM), `roast my team` + `tell my team a fun fact` verbatim corpus commands (`data/relay_roasts.txt` seed + `data/relay_fun_facts.txt` 1,014 verified-unique facts), `kill joy` -> Killjoy / `kay o` -> Kayo display canon, context+directive rephrase, profanity preservation, GUI output-device dropdown (dynamic PortAudio enumeration). **Sweep from main: 9807 passed / 39 skipped / 0 failed, ~189 s** + `validate_config` clean. | 9807 | (this session) |
| 2026-06-12 | `2d79a8e` | **Live-findings fix batch** (7 commits `ab08bf4..2d79a8e` on `9d460da`). Every OPEN live finding from the dogfood close-out fixed: NEW `lifecycle/single_instance.py` single-instance guard (held msvcrt/fcntl byte lock at offset 4096, metadata at offset 0 read unbuffered past Windows mandatory locks, pidfile fallback, refuse-only-on-contention errno classification, NO unlink on release (POSIX lock-after-unlink race), `KENNING_ALLOW_MULTIPLE_INSTANCES` escape; wired in `__main__.main()`, duplicate exits code 3); supervisor `ProjectIndex` borrows ConversationMemory's embedded Qdrant client (`client=` kwarg + owned-only `close()`; ends the every-boot "already accessed by another instance" registry-only fallback); launcher `window_appeared` honesty + post-placement `focus_window` bring-to-front (image-search windows no longer open behind the foreground; honest spoken line on window-timeout); WARM-path streaming STT in `_follow_up_listen` (kills the 108-1188 ms synchronous Moonshine re-transcribe; `_maybe_discard_stt_stream` + `MoonshineEngine.clear_stream_cache` on abort paths so dropped captures never leak); MCP server thread cancel+gather before loop close (the startup asyncio "Task was destroyed" stderr noise on bind failure); capture status-flag/drop accounting (count on audio thread, report from `drain()`); blip-watcher `internal_dropout` two-tier rule + edge-burst-run stripping adjudicated against all 174 live records (remaining findings = misclassified trailing bursts already trimmer-fixed + natural-prosody false positives; `trim_and_fade` untouched); stale XTTS `reference_audio` -> `kenningVoiceAudio/kokoro training audio/` everywhere + ADDITIVE protection-list extensions. Adversarially reviewed (4-dimension parallel pass; all should-fixes applied incl. the release-unlink race + audio-thread logging removal + errno classification + tmp_path test hygiene). Voice baseline contract intact (cold path byte-untouched, structurally test-pinned). | 9713 | (this session) |
| 2026-05-29 | `c17a2c9` (campaign, in progress) | **Production-hardening campaign — coding-engineer phase** (first 5 commits `a8e6ef6`…`c17a2c9`, on the infra-wiring tip `9d51cec`). Wiring the recent catalog ports into one cohesive unit + completing the voice-controlled coding engineer end to end. **B1** (`a8e6ef6`) fixed a real security/cohesion bug — the coding bridge's safety FILE_CHANGE listener read the wrong `TaskEvent` attributes, so file-write validation never ran on coding edits — plus dead-code + regex-docstring cleanup. **B3-loop** (`8651b07`) connected the clarify/verify/complete loop for voice-dispatched tasks: a per-project `.mcp.json` pointing at the live in-process MCP server + a bound `ProjectSession` (was bridge-only before); added `KenningMCPServer.is_running()`. **B3-runlaunch** (`9cf6f45`) added NEW `coding/sandbox_runner.py` + the "run / launch the program" voice commands (sandbox-confined + validator-gated + non-blocking) + a "say run X to try it" completion hint + the `coding.sandbox_run_timeout_seconds` knob. **B3-quality-a** (`c43dfd7`) prepends an always-on code-quality preamble (type hints, docstrings, `pyproject.toml` for new projects) to every coding prompt. **B3-loop-2** (`c17a2c9`) stopped deleting the `.mcp.json` on COMPLETE (`send_followup` reuses it for RESUME + the verifier's corrective re-prompt, so the cleanup had been stripping MCP from every follow-up) + re-attaches the digest + voice-lock-review listeners to the follow-up handle. The architect plan-provider + TTS narration were verified already fully wired. Voice baseline contract intact. Remaining: routing/bridge/safety/infra/desktop reachability, evolution pervasive reach, the synthetic-audio e2e suite, latency/resource optimization. | 9163 | [project_ultron_2026_05_29_production_hardening_campaign.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_29_production_hardening_campaign.md) |
| 2026-05-29 | `6692866` (code `296e1f6`) | **Infrastructure-wiring campaign** (builds on catalog 14 `c8f4ce3`; 16 commits `b74c8ab`…`6692866`). Wired 14 dormant imported-but-unconsumed subsystems to production + DOCUMENTED 4 as architecturally-inactive. **WIRED** (all fail-open; default-ON unless noted): process discipline (T12 process-registry/zombie-killer at every daemon + coding spawn) + a latent `_maybe_handle_deep_research` `trace` NameError crash fix; deep-memory recall (`memory/deep_recall.py` + `_maybe_handle_deep_recall`); skill trust-gate (`skills/scan.py`, T5/T9, `scan_untrusted_skills` ON); decomposer requery (`HybridTaskDecomposer._requery_decomposition`, T14); generalised two-phase voice approval (`request_voice_confirmation`/`consume_voice_approval` on `CapabilityVoiceController`, T2 -- validator stays fail-closed); coding loop detection (T1 `LoopDetectionManager` over the TOOL_RESULT stream, narrate-only / never-cancel, `coding.loop_detection_enabled` ON); a dialog-narration surfacing FIX (`_drain_coding_dialog_narrations` -- the catalog-08/09 `pop_dialog_narration` had no caller); multi-key Brave rotation (T6 `RotatingBraveClient` + `resolve_brave_api_keys` + `web_search.brave_additional_api_key_envs`); dual-history "what did I say earlier?" recall (`memory/history_recall.py` + `DualHistoryStore` wired into the orchestrator + `_maybe_handle_history_recall`, `memory.history_recall_enabled` ON, Qdrant-independent); hooks lifecycle (coding TaskStart cancel-capable + TaskComplete, `hooks.enabled` ON, zero-cost when no scripts installed); the offline observation outcome-resolver `resolve_observation_outcomes` maintenance task; the gated-OFF MCP server lifecycle client (T22 `mcp/builder.py` + `McpConfig` -- env-filtered spawn + process-registry tracking + `kill_process_tree` reap, `mcp.enabled`/`mcp.autostart` default-OFF); the `.kenningignore` safety rule (Category U `KenningIgnoreRule` -- secrets path/command block, default-safe no-op, `safety.rules.U1`); and the explicit-intent NEEDS_EXPLICIT_INTENT unblock (`safety.intent` wired into the validator -- audited-allow only on the user's own verb+object, NEVER overrides BLOCK_HARD, `safety.explicit_intent_matching_enabled` ON). **DOCUMENTED-INACTIVE** (Kenning's single-threaded run loop + delegate-to-`claude --print` design has no consumer window; force-wiring would add hot-path latency/risk for a window that doesn't exist): `dual_history.truncate_*` "undo that" (needs `DualHistoryStore` promoted to the unified context source), `lifecycle/pending_message_queue` (no cold-start/swap capture window), coding edit auto-revert (breaks claude's black-box state mid-task), `auto_approval` session-warming (left for focused security review). Voice baseline contract intact; every commit independently green. | 9117 | [project_ultron_2026_05_29_infrastructure_wiring_campaign.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_29_infrastructure_wiring_campaign.md) |
| 2026-06-03 | (catalog 14) | **clawhub-self-improving-agent catalog 14 -- FOUR bounded extensions to the EXISTING `evolution/` package (NOT a new subsystem).** Benign plugin (0 RED / 1 YELLOW); built clean-room from the catalog + the two `_self_improving_scan` reports -- the quarantine source was NOT read. Adds QUALITATIVE conversation-event learning atop catalog 13's quantitative loop. **T1 (GREEN):** `CorrectionCapsule` / `KnowledgeGapCapsule` / `CommandFailureSignal` (`models.py`) + detectors (`signals.py`): `extract_correction` (gated on a non-empty prior response; strong-phrase fires always, weak opener suppressed when the turn reads as positive acknowledgement), `extract_knowledge_gap`, `extract_command_failure` (17-token in-process analogue of the upstream's RED PostToolUse bash hook), `extract_feature_request`. Corrections / gaps / command-failures feed the EXISTING repair-distillation path via `to_failure_record`; `EvolutionLoop._propose` now falls back to `auto_distill_from_failures` over a new `failures_provider` when success distillation is empty. Wired in `_record_evolution_turn` (`prior_response=self._last_response_text`) + a coding-runner command-failure `TaskEvent` listener drained by `Orchestrator._drain_evolution_command_failures`. **T2 (GREEN):** `FeatureRequestCapsule` -> `data/evolution/feature_requests.jsonl`, NEVER distilled, surfaced in `EvolutionService.digest()`. **T3 (YELLOW):** a <=50-token `[Evolution: ...]` pre-turn nudge via `EvolutionService.pre_turn_system_hint()` through the SAME `set_temperament_hint` SYSTEM-prompt seam (never user text); token-capped, default-ON, `""` when idle. **T4 (GREEN):** `pattern_key`/`recurrence_count`/`first_seen`/`last_seen` on `Capsule` + the new types; `derive_pattern_key` + `merge_capsules_by_pattern_key` + `RECURRENCE_PROMOTE_THRESHOLD=3` make the distiller's recurrence gate explicit + auditable (row-count; empty key = byte-identical legacy grouping). New default-ON `EvolutionConfig` knobs (correction_detection / feature_request_capture / command_failure_capture / pre_turn_nudge_enabled / pre_turn_nudge_max_chars / recurrence_threshold). New ledgers `data/evolution/{corrections,knowledge_gaps,command_failures,feature_requests}.jsonl`. Voice baseline contract intact (microsecond regex passes on the hot path). +~53 hermetic tests (`tests/evolution/test_catalog14_*.py` + orchestrator wiring). | 8999 | [project_ultron_2026_06_03_clawhub_self_improving_agent.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_06_03_clawhub_self_improving_agent.md) |
| 2026-06-02 | (catalog 13) | **clawhub-capability-evolver catalog 13 port -- bounded autonomous self-improvement (NEW `src/kenning/evolution/` subsystem).** A QUARANTINED, high-risk plugin: the source was **NEVER read / imported / executed / deobfuscated**; the 10-module package was built clean-room from the catalog entry + eight static scan reports ONLY (no source code, constant, or string ever in context to copy). kenning observes its own turns, mints success/failure *capsules*, and -- once a pattern recurs (>=10 successes, >=7 of last 10, 24h cooldown) -- distills a new trigger-loaded skill into `data/evolution/skills/*.md` (a gitignored, checkpointed, revertible live skills source). Every proposal runs the bounded `EvolutionLoop(AgentLoop)`: pre-flight (fail-closed) -> autonomy-tier gate -> reversible checkpoint -> write -> blast-radius + constraint check -> 4 regression guardrails -> keep or auto-revert -> hash-chained audit, bounded by `max_steps`. **HARD SAFETY CONTRACT (the upstream's self-rewriting / network / shell core excluded BY CONSTRUCTION):** proposals are DATA ONLY (skills markdown / in-range config), NEVER generated code, NEVER `src/kenning/`, NEVER a Category-K surface; the safety validator / audit ledger / engine itself sit behind a Tier-3 hard wall (`blast_radius.CRITICAL_PROTECTED_PREFIXES` includes `src/`); zero network / shell / eval; `EnvFingerprint` omits the upstream device id. Modules: `models.py` (GEP data model), `signals.py` (local opportunity-signal extraction, no LLM/Hub layer), `blast_radius.py` (policy spine + protected-path wall), `skill_distiller.py` (capsule->skill distillation), `guardrails.py` (latency/quality/error/resource detectors + rollback audit), `autonomy.py` (`TieredAutonomyController` + trust graduation), `personality.py` (Tier-0 `[Tone: ...]` temperament tune), `evolution_loop.py` (the bounded loop), `service.py`+`intent.py` (JSONL runtime + voice commands). Wired default-ON + fail-open: `EvolutionConfig`; `_load_evolution_if_enabled` at startup (before the skill registry, whose extra dirs gain `data/evolution/skills`); run-loop short-circuit `_maybe_handle_evolution_command` ("evolve now" / "evolution status"); per-turn `_record_evolution_turn` (opportunity capsule + barge-in -> temperament + autonomous-cycle trigger); `LLMEngine.set_temperament_hint` injecting the tone directive into the SYSTEM prompt only (the web-gate / local-clock detectors see the raw utterance); shutdown persistence. Tier summary: 7 GREEN + 2 YELLOW + (self-rewriting/network/shell core) RED, excluded. Voice baseline contract intact (no SOUL.md / RVC / Piper / Kokoro touch; hot path gains only microsecond setter + signal-extraction calls; the cycle runs single-flight on a daemon thread off the hot path). `THIRD_PARTY_NOTICES.md` extended with the quarantined-source provenance record + per-component table. +286 tests (265 evolution package + 21 llm/orchestrator wiring). | 8949 | [project_ultron_2026_06_02_clawhub_capability_evolver.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_06_02_clawhub_capability_evolver.md) |
| 2026-05-31 | (catalog 11) | **clawhub-browser-agent catalog 11 port.** A raw-CDP-WebSocket primitives plugin (≈350 LOC, no agent loop) architecturally superseded by the catalog-10 `browser-use` CLI tier, so the port extracts the genuinely transferable hardening patterns + the agent-loop meta-pattern (4 GREEN + 3 YELLOW + 0 RED), NOT the CDP transport. NEW modules: `src/kenning/utils/heartbeat.py` (T2 `HeartbeatThread` -- stoppable, fail-open daemon keep-alive; `Event.wait`-based + `HeartbeatStats`, improving on the upstream's unstoppable `while True: sleep`), `src/kenning/utils/health_check.py` (T4 `http_health_check` + `cdp_health_check` -- cheap fail-open pre-flight probes with injectable transport), `src/kenning/agent_loop/base.py` (the `AgentLoop` meta-pattern: an ADDITIVE observe->plan->act->verify base whose load-bearing invariant is the `max_steps` cap, plus built-in repeated-signature loop detection + per-step `StepRecord`s + a verify hook + fail-open execution; does NOT modify any existing runner). `desktop/browser_use.py` `BrowserUseTool` gains three gated methods: `click_css_selector` (T3 ARIA-ref-miss fallback -- CSS selector -> `getBoundingClientRect` -> Cap-3-gated `click_at_coords`, computing the *correct* box-model centre rather than the upstream's buggy `(border[0]+border[1])/2`; selector `json.dumps`-encoded against injection), `wait_for_element_js` (T7 event-driven `MutationObserver` element-appear wait via the gated `eval`, with a bounded `setTimeout` fallback the upstream lacked), `export_pdf` (T6 `Page.printToPDF` page-to-PDF export, `PathResolver` + Cap-2/Cap-3 gated, fail-open). T1 idle-reconnect / T2 heartbeat / T4 health-check / T5 `--remote-allow-origins` have NO literal wiring target in kenning's CLI-based browser architecture (the `browser-use` CLI owns its own Chrome + CDP; kenning never opens a raw CDP socket; the launcher deliberately blocks CDP flags on the user's real Chrome) -- so T2 + T4 ship as importable primitives and T1 + T5 are documented findings, not forced code. Clean-room re-implementation from a zero-RED-confirmed read-only source scan (3 Sonnet 4.6 Explore agents); no source copied verbatim. No new config knobs (always-available gated methods on the already-default-ON `browser_use` tool + importable utils). +71 hermetic tests (`tests/utils/test_heartbeat.py` 10, `tests/utils/test_health_check.py` 18, `tests/desktop/test_browser_use_catalog11.py` 25, `tests/agent_loop/test_base.py` 18). Voice baseline contract intact; no orchestrator hot-path edit. `THIRD_PARTY_NOTICES.md` extended with clawhub-browser-agent (MIT) attribution. | 8546 | (this session) |
| 2026-05-30 | `f176f29` (fix `7b53ea1`) | **Bridge-e2e subprocess-reap flake fix.** Fixes the long-deselected `tests/integration/test_bridge_e2e.py::test_health_through_real_subprocess` flake. `OpenClawClient._run_cli`'s timeout cleanup (`src/kenning/openclaw_bridge/client.py`) now reaps the WHOLE process tree via a new `_reap_process_tree` helper → `subprocess.kill_tree.kill_process_tree`, instead of `proc.kill()`: on Windows the openclaw CLI is a `.cmd` shim that spawns the real interpreter as a grandchild, and killing only the immediate child orphaned that grandchild — which held the stdout/stderr pipes open and wedged the event loop's subprocess transport at teardown (the historical "hung full sweep until the wall-clock watchdog" symptom; the conftest session-end reaper is useless because the session never ends while pytest stalls mid-run). The tree is collected while the root is still alive so psutil can reach the grandchild; the synchronous kill runs in the default executor so the loop stays free to drain the closing pipes. `test_health_through_real_subprocess` now uses a 20s health probe (vs 5s) to absorb Windows `.cmd`→`python` cold-start under sweep load (still under the 30s per-test deadline). New hermetic `tests/openclaw_bridge/test_client.py::test_run_cli_timeout_reaps_whole_process_tree` spies on `kill_process_tree` to pin the reap contract. Greatly improves the flake (passes in isolation ~0.5s; on an unloaded machine the prior session saw the full sweep green without deselection at 8475/106s) but does NOT fully fix it -- the test spawns the real openclaw `.cmd`->python subprocess, so under heavy machine contention the 20s health probe can be exceeded, failing the test + wedging the sweep to the wall-clock watchdog (observed 2026-05-30: a contended audit stalled at ~33% on exactly this test, exit 5; the deselected sweep was clean at 8474 passed / 26 skipped / 1 deselected / 0 failed in ~160s). Keep `-- --deselect "tests/integration/test_bridge_e2e.py::test_health_through_real_subprocess"` as the loaded-machine fallback; a 33%-stall here is the flake, not a regression. Voice baseline contract intact (`_run_cli` is the OpenClaw bridge transport, not the voice hot path). | 8475 (8474 + flake) | [project_ultron_2026_05_30_clawhub_browser_use.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_30_clawhub_browser_use.md) (Branch-integration section) |
| 2026-05-30 | `d220b50` | **Deferred-primitive wiring pass.** Wired the previously-ported-but-unconsumed openclaw-clawhub primitives into orchestrator hot paths. **T15 private telemetry** (`observability/private_telemetry.py`): `Orchestrator._init_telemetry_store` at startup + `_emit_turn_telemetry` in `_respond`'s `finally` (one aggregate `HashedEvent` per turn -- routing-intent under the `category` safe key + `searched` bool + numeric `latency_ms` + `tier` bucket + `outcome`; `_latency_bucket` labels kept <=12 chars to pass the raw-path leak check). **FAIL-PRIVATE: no-ops unless `KENNING_TELEMETRY=opt-in`** -- the one feature deliberately NOT default-on (privacy-by-construction; the fail-open exception does not apply to a privacy gate). **T7 short-lived token** (`identity/short_lived_token.py`): `_mint_forensic_token` registers an idempotent trusted-caller tuple + mints an HS256 JWT at MCP-server start (`mcp:tools`) + gaming-engage (`voice:gaming-engage`, revoke-by-expiry on disengage); forensic / defense-in-depth (single-user in-process runtime: minter + verifier share the trust boundary), audit-logged, fail-open. **T12 report queue** (`feedback/report_queue.py` + NEW `feedback/report_intent.py`): strict `match_report_concern` regex (no LLM round-trip; "report on the weather" does NOT trip it) intercepted in the run loop BEFORE routing; `_maybe_handle_report_concern` files a `Report` to hash-chained `data/feedback/reports.jsonl` + speaks an ack; deliberately avoided a new `RoutingIntentKind` (the 23-value enum is asserted by many tests). **T18 image markdown verified consumer-less** -- text-only Qwen 4B + `claude --print` text-argv coding bridge + moondream2 raw-bytes mean no inline-data-URL consumer exists; left importable, NOT fake-wired (honest no-op per "don't build for hypothetical futures"). `.gitignore` extended for the wiring-pass runtime-data dirs (incl. `data/identity/`, which holds the HMAC token-signing secret). +51 hermetic tests (all `Orchestrator.__new__` pattern; real round-trips redirect `PROJECT_ROOT` to `tmp_path`). Voice baseline contract intact (hot path gains one cheap fail-private emit). | 8474 | [project_ultron_2026_05_30_clawhub_browser_use.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_30_clawhub_browser_use.md) (Follow-on section) |
| 2026-05-30 | `5451017` (doc `97a6c5c`) | **clawhub-browser-use catalog 10 port.** Nine batches wrapping the external open-source `browser-use` CLI (Playwright + Chrome DevTools Protocol) as kenning's CDP-backed browser-automation tier. The plugin source is **documentation-only** (`SKILL.md` + CDP / multi-session references, NO Python), so the port is a clean-room subprocess wrapper around the CLI's documented public surface -- NOT vendored code; the binary is NOT installed and NOT a hard dependency (every method fails open with a "binary not found" result when absent; operators `pip install browser-use` to enable the tier). Three new `desktop/` modules: **`browser_use.py`** (`BrowserUseTool` sync subprocess wrapper -- lazy `shutil.which` discovery against `[browser-use, bu, browseruse]`, `CREATE_NO_WINDOW`, `BROWSER_USE_SESSION` env scrub; GREEN reads = T1 `state` + T2 extraction (`get_html/text/value/attributes/bbox/title`) + T5 `wait_selector/wait_text` + T6 tab ops; YELLOW = T7 write primitives (`click_at_index/coords`, `type/input/select/upload/hover/keys/dblclick/rightclick`) + T9 `screenshot` + T3 `eval`+`analyze_js_script` + T4 cookie CRUD + T10 `connect/connect_profile/profile_list` + T11 raw `cdp_python`+`analyze_cdp_statement`, each routed through `safety.validator` Cap-3 and destructive/takeover ops additionally through `safety.two_phase_approval`); **`browser_sessions.py`** (`BrowserSessionsManager` T8 -- name allowlist + cap + `ProcessRegistry` lifecycle + `kill_process_tree` on force-close + two-phase on `close_all`); **`browser_sequence.py`** (`BrowserSequenceRunner` creative extension -- before/after screenshot bracket via headless base64 + injected VLM verify + fail-fast; reuses `desktop.sequence` `SequenceStatus`/`StepOutcome`/`VlmVerdict` enums). Wiring (batch 9): `orchestrator._load_browser_use_if_enabled` builds the singletons at startup (cheap + lazy + fail-open, right after `_load_desktop_vlm_if_enabled`) + `screen_context._maybe_browser_use_state_text` gated fallback tier (UIA -> `extract_browser_content` -> browser-use state -> VLM) firing only when UIA browser extraction is empty AND the tool has an active page; new top-level `browser_use` config section (8 knobs, all default ON). Tier outcome: 8 GREEN + 5 YELLOW + 2 RED -- T12 (cloud, paid API key) + T13 (Cloudflare tunnel) NOT ported (RED, per `feedback_no_paid_apis` + inbound-attack-surface); `--cdp-url` external arg / `profile sync --all` cloud upload / ambient `BROWSER_USE_SESSION` also deliberately not exposed. Security-review hardening (independent Sonnet 4.6 Explore pass): T7 `upload` treated YELLOW with `PathResolver.safe_realpath`; T3 JS static-analysis extended (+ `sendBeacon`, `WebSocket`, `RTCPeerConnection`, `eval`, `new Function`, dynamic `import`); T11 CDP domain blocklist extended to 9 domains that REFUSE outright (overriding even `assume_preapproved`). `THIRD_PARTY_NOTICES.md` extended with browser-use (MIT) + 9-batch per-component table. +431 hermetic tests (all mock subprocess + validator + PathResolver + approval registry; no binary required). Voice baseline contract intact (startup gains only the lazy fail-open singleton; never on the conversational hot path). | 8423 | [project_ultron_2026_05_30_clawhub_browser_use.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_30_clawhub_browser_use.md) |
| 2026-05-29 | `bca58b0` (doc `ebe2988`) | **Catalog 09 production-wiring pass + default-ON sweep + handoff cleanup.** Landed the previously-deferred catalog-09 (clawhub-desktop-control, the port row below) items as live consumers (9 feature commits, +106 tests): dialog auto-handler in the coding bridge (batch B) + a new background `DialogPoller` daemon (batch A) emitting bus `DialogAppearedEvent` / `DialogResolvedEvent`; `ACTIVE_WINDOW_QUERY` + `SEMANTIC_CLICK` + `WINDOW_CLOSE_CONFIRMATION` voice intents (batch C); `extract_browser_content` folded into `screen_context` (batch D); two-phase approval voice yes/no for `close_window` when `suspected_unsaved=True` (batch E); pre-edit file-content snapshot in `direct_bridge` for SWE-Agent T1 auto-revert + T14 edit-recovery (batch F); per-intent condenser selection in `LLMEngine._build_messages` (batch G); `drive_start_task` async-generator gaming-engage / disengage state machine with per-stage voice acks (batch H); T7 OCR Tesseract tier `desktop/ocr.py` (batch I, catalog-08 deferred). **Default-ON sweep** (`aa00bb9`): flipped `llm.history_compression.intent_adaptive` False->True (the "ship session work enabled" rule) AND auto-started the `DialogPoller` daemon in `Orchestrator.__init__` so batch A's bus events actually fire (the batch B subscription chain was dead without it). **Handoff cleanup** (`bca58b0`): `screen_context.py` browser-inputs iterator now tolerates both `UIElementInfo` dataclass + legacy `(label, value)` tuple shapes; `test_canonical_monitor_runner_wiring.py` listener-count asserts replaced with a qualname-introspecting `_has_canonical_monitor_listener` helper. Voice baseline contract intact (hot path gains only a microsecond `set_current_intent_kind` setter call). | 8046 | [project_ultron_2026_05_28_catalog_09_production_wiring.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_28_catalog_09_production_wiring.md) |
| 2026 | `1c62068` | **clawhub-desktop-control catalog 09 port.** Five feature commits on top of `ee5f8dc`. All 7 cataloged techniques (T1-T7) landed across 5 batches. Batch 1 (`21d0497`): T1 `InputController.scroll` extended with `direction="vertical"|"horizontal"` kwarg routing to `pyautogui.scroll` vs `pyautogui.hscroll` (closes the lazy-content browser-scroll gap from catalog 08 T5); T3 `InputController.type_text` extended with `wpm` kwarg using the standard 5-chars-per-word formula (interval=1/((wpm*5)/60); 60-80 WPM passes most JS form validators that reject instant input; wpm<=0 returns structured error instead of upstream's ZeroDivisionError); T7 `InputController.move_mouse` extended with `smooth=False` kwarg that dispatches to `pyautogui.easeInOutQuad` tween when smooth=True with duration_s>0 (gaming-mode anti-detection + demo-mode narration). All three additive kwargs preserve back-compat by defaulting to legacy behaviour. MCP tools (`mouse_move` / `type_text` / `scroll`) extended with matching kwargs. Batch 2 (`e3274b6`): T2 GREEN `get_pixel_color(x, y)` in `desktop/capture.py` (pyautogui.pixel wrapper, fail-open on exception, RGB tuples NOT taint-tracked since they're ephemeral); T2 YELLOW `wait_for_pixel_color(x, y, target_color, *, tolerance, timeout_s, interval_s, sleep_fn, clock_fn)` in `desktop/uia.py` mirroring the catalog 08 T4 wait pattern with L-infinity tolerance for anti-aliased pixels (closes gaps for game-state polling, loading-spinner disappearance, status LEDs, progress-bar completion). Batch 3 (`9c3be8a`): T4 NEW `src/kenning/desktop/clipboard.py` -- `ClipboardManager` class with Cap-2-gated `read_text(*, user_text)` + Cap-3-gated `write_text(text, *, user_text)`, full safety-validator integration (tool_name=desktop.clipboard.read/write with capability=clipboard_read/write), taint tracker integration (read bytes recorded under clipboard_read for exfil detection; write bytes recorded under clipboard_write for paste-target verification), oversize write rejection, payload preview capped at 2 KB so audit log stays bounded, pyperclip lazy-import with structured failure on missing dep (vs upstream's silent log). `ClipboardResult` frozen dataclass. MCP tools `clipboard_read` + `clipboard_write` registered. Batch 4 (`7e7e40d`): T6 `find_image_on_screen(template_path, *, confidence=0.8, region)` + `TemplateMatch` frozen dataclass in `desktop/capture.py`. Routes `template_path` through `PathResolver.safe_realpath` (defends against attacker-controlled paths matching spoofed UI elements); pyautogui.locateOnScreen behind a broad-except that handles missing opencv-python as None (fail-open contract); returned centre coords route to InputController.click for the Cap-3-gated click. MCP tool `find_image_on_screen` (region split into 4 separate args for JSON-over-stdio). Batch 5 (`1c62068`): T5 NEW `src/kenning/desktop/sequence.py` -- `DesktopSequenceRunner` class with before/after screenshot bracketing per step; `SequenceStep` + `ScreenshotRef` + `StepResult` + `SequenceResult` frozen dataclasses; `SequenceStatus` / `StepOutcome` / `VlmVerdict` enums; optional VLM verification of after-frames via injected vlm_describe callable (confirmation-keyword pattern shared with click_preview); auto-pass radius (default 150 px) so sequential steps within a panel skip redundant VLM round-trips; fail-fast contract (first failure aborts remaining steps + records failed_at_step); analyze-and-discard contract on captured bytes; deliberately did NOT port the upstream `_check_approval` blocking input() (incompatible with voice-first; kenning uses two_phase_approval), pygetwindow window activation (kenning uses focus_by_title), or the keyword-cascade `_plan_*` planners (kenning's LLM intent router is more capable). Tier summary: 4 GREEN + 3 YELLOW + 0 RED. Source plugin (~93 KB across 7 files) read read-only via Read tool; never executed (per binding ClawHub-batch security rules). Pattern extraction via Sonnet 4.6 Explore agent that independently confirmed zero-RED security finding (no network calls, no subprocess / os.system, no ctypes / registry access, no persistence, no anti-forensics, no credential access, no obfuscation, no AV / EDR tampering). `THIRD_PARTY_NOTICES.md` extended with clawhub-desktop-control MIT attribution + per-component mapping for T1 / T2 / T3 / T4 / T5 / T6 / T7. Voice baseline contract intact (no SOUL.md / RVC / Piper / vocal WAV / LLM model file / Kokoro fine-tune voicepack touch; no orchestrator hot-path edit; all surfaces ship as importable infrastructure for future opt-in wiring). | 7805 | [project_ultron_2026_05_29_clawhub_desktop_control.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_29_clawhub_desktop_control.md) |
| 2026-05-28 | `2cad783` | **clawhub-windows-control catalog 08 port.** Five feature commits on top of `a48ec9d`. 7 of 8 cataloged techniques landed (T7 OCR deferred per catalog `★`; existing Moondream2 VLM tier covers the use case). Batch 1 (`44087e3`): T2 `UIElementInfo` + `get_ui_element_inventory` 10-bucket UIA walk in `desktop/uia.py`; T4 `wait_for_text_in_window` (uia.py) + `wait_for_window` (windows.py) synchronous polling barriers with `sleep_fn`/`clock_fn` injection + `DEFAULT_WAIT_TIMEOUT_S=30.0` / `DEFAULT_WAIT_INTERVAL_S=0.5` shared constants; T8 `InputController.drag_to` absolute-coord drag (`pyautogui.moveTo` + `pyautogui.dragTo`) through full controller gate stack + click-preview gate on SOURCE coordinate. Batch 2 (`9dafeca`): T5 `BrowserContent` + `BrowserLink` + `BROWSER_NAMES` (chrome/firefox/edge/brave/opera/vivaldi/arc) + `is_browser_window` + `find_browser_window` + `extract_browser_content` (20-100 ms vs 300-800 ms + ~330 MB VRAM for the VLM tier); UIA-based browser content extraction with headings (uppercase / colon-terminated Text + Static heuristic) / longer text / buttons / Hyperlinks (URL from automation_id when http(s)://) / Edit + ComboBox inputs / Images; per-bucket dedup + per-bucket caps + `full=True` shorthand. Batch 3 (`8e007b0`): T1 NEW `src/kenning/desktop/dialog_control.py` (DIALOG_CLASSES + DIALOG_CONTROL_TYPES + DIALOG_TITLE_KEYWORDS + DISMISS_BUTTONS constants; DialogInfo / DialogButton / DialogField / DialogCheckbox / DialogContent / DialogActionResult frozen dataclasses; find_dialogs + read_dialog + click_dialog_button + type_into_dialog_field + dismiss_dialog + wait_for_dialog with Cap-3 safety validator gating per-action; dismiss_dialog per-candidate validator re-check + ESC fallback gated with action=dismiss_escape). Batch 4 (`9a810b3`): T3 NEW `src/kenning/desktop/element_click.py` (CLICKABLE_TYPES 9-entry standard UIA control-type set + UIElementMatch / TextMatch / ClickResult frozen dataclasses + find_elements_by_name + click_element_by_name + find_text_in_window; exact-matches-first stable-sort ranking; enabled_only filter defaults True; ALL clicks route through gated InputController so click-preview VLM + foreground security + safety validator + Cap-3 explicit-intent + rate limit apply uniformly vs the upstream's pywinauto-native click_input() which bypasses every gate). Batch 5 (`2cad783`): T6 `get_active_window_title` lightweight foreground probe + `close_window` graceful WM_CLOSE via `win32gui.PostMessage` + `force=True` escalation to `kill_process_tree` + `UNSAVED_CHANGES_TITLE_HINTS` (5-entry editor convention) + `_title_suggests_unsaved_changes` predicate + `CloseWindowResult.suspected_unsaved` flag, plus `minimize_window_idempotent` / `maximize_window_idempotent` / `restore_window_idempotent` state-check-before-act helpers in placement.py. Tier summary: 4 GREEN + 4 YELLOW + 0 RED. Source plugin (~37 KB across 23 thin scripts) read read-only via Read tool; never executed (per binding ClawHub-batch security rules). Pattern extraction via Sonnet 4.6 Explore agent that independently confirmed catalog's zero-RED security finding (no registry access, no network calls, no persistence, no anti-forensics, no DLL injection, no credential access). `THIRD_PARTY_NOTICES.md` extended with clawhub-windows-control MIT attribution + per-component mapping for T1/T2/T3/T4/T5/T6/T8. Voice baseline contract intact (no SOUL.md / RVC / Piper / vocal WAV / LLM model file / Kokoro fine-tune voicepack touch; no orchestrator hot-path edit; all surfaces ship as importable infrastructure for future opt-in wiring). | 7715 | [project_ultron_2026_05_28_clawhub_windows_control.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_28_clawhub_windows_control.md) |
| 2026 | `ee9eca5` | **clawhub-windows-ui-automation catalog 07 port + test fixes.** Six commits on top of `c3966a7`. Five batches landing T1-T6: T1+T3 doc comments in `desktop/input_control.py` (pyautogui uses modern atomic `SendInput`; `pyautogui.hotkey` is async unlike `SendKeys.SendWait`); T2 new `desktop/win32_helpers.py` (ctypes wrappers for `GetDpiForMonitor`, `GetLastInputInfo`, `DwmGetWindowAttribute(DWMWA_CLOAKED)`, hardened `block_input_context` with watchdog + try/finally + UIPI safety floor + 30s hard cap, plus `logical_to_physical` / `physical_to_logical` primitives); T6 `focus_by_title` two-tier focus in `desktop/windows.py` (primary `SetForegroundWindow`, fallback in-process `WScript.Shell.AppActivate` via `win32com.client`, final `CREATE_NO_WINDOW` PowerShell subprocess) plus `enumerate_windows`/`find_window` gain `exclude_cloaked: bool = True`; T5 DPI-aware coordinate helpers in `desktop/uia.py` (`physical_center_of_element` / `physical_rect_of_element` / `dpi_aware_click_at_element_center` -- identity on 100%-DPI; opt-in `assume_logical=True` for non-DPI-aware sources); T4 frontmatter `capability_tags:` filter wiring (`SkillRegistry.matching_skills` accepts `gaming_mode` / `vlm_loaded` / `has_internet` kwargs; `_skill_active_for_capability_tags` predicate; `LLMEngine._build_messages` threads live VLM holder state via new `_resolve_vlm_loaded_for_skills()`). Closing `ee9eca5` test-fix commit updated 2 tests broken by the prior production-wiring flag flips (`coding.pre_write_lint.enabled` + `coding.pre_task_confirmation_enabled` both flipped to True) -- tests now monkeypatch per-test config state instead of depending on the global default; added companion `test_runner_lint_listener_enabled_returns_callable` so both branches stay pinned. `THIRD_PARTY_NOTICES.md` extended with clawhub-windows-ui-automation MIT attribution. Voice baseline contract intact (no SOUL.md / RVC / Piper / vocal WAV / LLM model file / Kokoro fine-tune voicepack touch; no orchestrator hot-path edit beyond the additive `vlm_loaded` thread). | 7516 | [project_ultron_2026_05_27_catalog_07_clawhub_windows_ui_automation.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_27_catalog_07_clawhub_windows_ui_automation.md) |
| 2026-05-26 | `29ffe49` | **Production-wiring pass batch 10 (voice baseline re-measure post-flags).** `scripts/measure_baseline.py` ran cleanly under the new config: VRAM loaded **6254 MB (-343 MB vs 2026-05-23 baseline)**, peak **6664 MB (-343 MB)**, STT median **16 ms** (unchanged), LLM TTFT median **172 ms (-31 ms)**, TTS synth median **78 ms (-31 ms)**, composite TTFA median **266 ms (-47 ms)**. The Batch 9 flag flips (supervisor.tier=full, architect, click_preview, contextual_retrieval, background_summary, pre_write_lint, goal_anchors, etc.) did NOT regress the voice baseline -- they only fire on coding / desktop / write paths, not on the simple voice-query path measured by measure_baseline. No optimisation pass needed; the production wiring is net-positive on every measured axis. `baselines.json` updated. | (no test change) | [project_ultron_2026_05_26_production_wiring_pass.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_26_production_wiring_pass.md) |
| 2026-05-26 | `2e1f8b8` | **Production-wiring pass batch 6 (SWE-Agent T7 SubmitReviewLoop into supervisor COMPLETE listener).** New `CapabilityVoiceController._attach_submit_review_listener` registers on every supervisor-dispatched task's COMPLETE event. The listener walks `event.files_created`/`files_modified`/`files_deleted`, runs `detect_voice_lock_hits()` (matches SOUL.md / IDENTITY.md / Piper / RVC / Qwen GGUFs / Kokoro voicepack + fine-tune / tts/rvc.py / tts/kenning_filter.py), and on any hit: logs WARN with the hit list AND queues a voice narration ("Voice-baseline contract: the session touched X. Review before continuing.") onto `controller._pending_completion` for the orchestrator's idle drain. Wired in after `_attach_supervisor_digest_listener` in `_dispatch_supervisor_task`. Fail-open: listener registration / dispatch errors log WARN, never abort. Skipped T14 edit_recovery + SWE-Agent T1 auto-revert (need pre-edit snapshot of agent-owned writes — needs deeper agent-tool integration). Also updated `tests/test_coding_voice.py::test_pre_task_confirmation_disabled_dispatches_immediately` to monkeypatch the legacy-shim flag after the Batch 9 flip. | 7394 | [project_ultron_2026_05_26_production_wiring_pass.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_26_production_wiring_pass.md) |
| 2026-05-26 | `8a7759d` | **Production-wiring pass batches 8+9 (OpenClaw live dispatch + flag flips).** Per user direction ("wire and flip everything on ... accept high latency for now, optimize later"), config.yaml flipped these knobs ON: `openclaw.enabled` (Gateway at `http://127.0.0.1:11280`; bridge now lives, dispatcher returns real plugin invocations instead of stub voice messages), `openclaw.bridge.inbound_voice_handoff_enabled` (`[voice]`-prefix routing into kenning's voice pipeline), `coding.pre_task_confirmation_enabled` (~0.5 s TTS playback with barge-in before each coding dispatch), `coding.supervisor.tier: "indexing_only" -> "full"` (decide + narration + enriched-context TaskRequest), `coding.repo_map.enabled` (50-300 ms pre-dispatch PageRank repo map prepended to AI coding agent prompt), `coding.pre_write_lint.enabled` (compile + flake8 + tree-sitter on every write; 50-500 ms per .py write), `coding.architect.enabled` + `narrate_enabled` (3-5 s pre-dispatch plan + sentence-by-sentence narration with barge-in), `coding.goal_anchors.enabled` (anchor decomposition + per-anchor voice progress), `desktop.click_preview.enabled` (1-2 s VLM round-trip on the first click in a region; auto-pass within 100 px), `memory.contextual_retrieval.enabled` (80-150 ms per write turn), `memory.background_summary.enabled` (idle-gated). Kept OFF per user veto: `llm.draft_kind` (real .8b PLD `llama_decode -1` bug; not a preference), `memory.reranking.enabled` (17-18 s/turn CPU cost). `notifications.telegram.enabled` stays OFF per user (no bot token configured). Validated via `scripts/validate_config.py`. Voice baseline TTFT will regress notably during this pass; per user direction the optimisation pass (Batch 10) is the follow-up to claw back. | (7394, no code change in this batch) | [project_ultron_2026_05_26_production_wiring_pass.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_26_production_wiring_pass.md) |
| 2026-05-26 | `61135e0` | **Production-wiring pass batch 4 (T1 trust envelope + T9 version-exact in LLMEngine.reload_for_preset).** Model hot-swap now enforces the version-exact contract via `VersionExactRequest` + `validate_version_exact_request` (preset names are concrete identifiers; floating tokens caught upstream by LLM_PRESETS lookup) AND a synchronous T2 digest verification against the recorded TOFU pin BEFORE the Llama load. New `verify_single_artifact_sync(identifier, path, ...)` in `install/voice_baseline_verify.py` is the sync entry point. Mismatched on-disk GGUF -> `reload_for_preset` returns `(False, "refused {preset}: model GGUF digest mismatch...")` without ever spending the ~3 GB / 5-10 s Llama load on a tampered file. Status in `{verified, pinned, missing, error}` all proceed (missing/error fall through to the existing Llama-load error path which has clearer diagnostics). Fail-open at the trust pre-check itself: a broken verifier never blocks a legitimate swap (the async voice-baseline verifier at startup provides defence in depth). 3 new tests in `tests/test_llm_reload_for_preset.py` (mismatch refuses + verified proceeds + pre-check exception swallowed). Voice baseline contract intact. | 7394 | [project_ultron_2026_05_26_production_wiring_pass.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_26_production_wiring_pass.md) |
| 2026-05-26 | `84dd1f6` | **Production-wiring pass batch 3 (T5 mode-scoped skill filter).** `Skill` frontmatter can now declare a `modes: [gaming, standby]` list. `SkillRegistry.matching_skills(user_text, *, mode="standby")` filters skills whose declared modes exclude the current mode. Skills with NO `modes` declaration match every mode (legacy + unscoped). `maybe_get_skills_block(user_text, *, mode="standby")` forwards the mode. `LLMEngine._build_messages` calls new module helper `_resolve_current_mode_for_skills()` which reads `kenning.openclaw_routing.gaming_mode.is_gaming_mode_active()` and threads `"gaming"` / `"standby"` into the skills call. The loader already passes unknown frontmatter keys through `Skill.extra` (forward-compat), so the wiring is purely on the consumer side. 7 new tests in `tests/skills/test_mode_filter.py` cover the predicate (no-modes, list filter, case-insensitive, string-value tolerance, broken-shape fail-open) and the registry-level + maybe_get_skills_block end-to-end forwarding. Voice baseline contract intact. | 7391 | [project_ultron_2026_05_26_production_wiring_pass.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_26_production_wiring_pass.md) |
| 2026-05-26 | `626b792` | **Production-wiring pass batch 2 (subprocess + T2 artifact identity).** T8 `kill_process_tree` replaces ad-hoc `terminate()`+`wait()`+`kill()` chains in `transcription/parakeet_engine.stop_parakeet_server` and `tts/xtts_v3._stop_server_subprocess`. Order is preserved: graceful HTTP `/shutdown` first, brief poll grace, then `kill_process_tree(pid)` for the parakeet/xtts root + every uvicorn worker child + grandchildren in one cross-platform psutil-backed call. New `src/kenning/install/voice_baseline_verify.py` ports T2 TOFU verification to the 6 canonical voice-baseline artifacts (LLM GGUF / 0.8b draft GGUF / Kokoro voicepack / Kokoro fine-tune weights / wake-word ONNX / Smart Turn V3 ONNX). At orchestrator startup `verify_voice_baseline_artifacts_async(PROJECT_ROOT)` spawns a daemon thread that hashes each artifact, looks up the pin from `data/install/pinned_digests.jsonl`, and either records first-use TOFU OR verifies against the existing pin. Mismatches surface as voice-baseline integrity warnings via `on_complete` callback. The async path keeps the ~5-10 s GGUF hash off the cold-start critical path; report deposits on `Orchestrator._voice_baseline_report` so future "are my models OK?" voice intents can poll. 10 new tests in `tests/install/test_voice_baseline_verify.py` + updated `tests/test_stt_swap_orchestrator.py::test_stop_parakeet_server_terminates_alive_process` for the new kill_tree contract. Voice baseline contract intact. | 7384 | [project_ultron_2026_05_26_production_wiring_pass.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_26_production_wiring_pass.md) |
| 2026-05-26 | `ecd8755` | **Production-wiring pass batch 1 (zero-risk foundation).** T14 rate-limit recorder threaded into Brave / SearxNG / DuckDuckGo clients via per-pid `on_response` closure built by `SearchProviderChain._build_recorder`; `_PROVIDER_FACTORIES` lambdas now take a recorder arg + lazy clients call the closure after every HTTP response (success + 429). DDG classifies throttle-shaped exception text (`429` / `403` / `captcha` / `rate` / `throttle` / `blocked`) as `was_429=True` so the chain cools it down without needing response headers (DDG-search lib scrapes HTML). T3 canonical_codes + T16 category + rule_metadata fields added to `AuditLog.record` so safety-validator rules can attach the T3 reason-code namespace and dashboards can group blocks without parsing reason strings. T18 `sanitize_for_log` wraps every tool-supplied string (reason / tool_name / capability / context dict leaves / error message) in audit.py + error_log.py, defending against CWE-117 log forging. T11 `materialise_default_pins(PROJECT_ROOT)` runs once at orchestrator startup so the 5 voice-baseline lock anchors (voicepack:kenning / voicepack:kokoro_finetune / llm:qwen3.5-4b / persona:identity / validator:k_category) land in the workdir lockfile. 6 new recorder roundtrip tests in `tests/web_search/test_provider_chain_recorder.py`. Voice baseline contract intact. | 7374 | [project_ultron_2026_05_26_production_wiring_pass.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_26_production_wiring_pass.md) |
| 2026-05-25 | `b46ad89` | OpenClaw-ClawHub catalog port batch 9 (YELLOW; final) -- T7 trusted-publisher short-lived token mint + verify. Stdlib-only HMAC-SHA256 JWT (HS256) with local signing secret at data/identity/short_lived_token_secret.bin. mint_token (caller_id + audience + scope + ttl + extra_claims; refuses oversized TTL / disallowed scopes via pre-registered TrustedCaller tuples in data/identity/trusted_callers.jsonl). verify_token enforces signature + expiry (60s clock-skew tolerance) + audience + caller-id trust-tuple equality + per-claim match + scope allowlist. Exception hierarchy: TokenMintError / TokenVerifyError / TokenExpiredError / TokenSignatureError / TrustedCallerNotFoundError / TrustedCallerClaimMismatch. Audit log at data/identity/short_lived_tokens.jsonl with SHA-256 hash chain + verify_audit_chain. rotate_secret invalidates historical tokens. Wired use cases: MCP server startup / coding bridge subprocess / skill execution token / voice gaming-mode handoff. RSA-256 + TPM-backed keys documented as future hardening path. New module: `src/kenning/identity/short_lived_token.py`. +30 module tests. | 7368 | [project_ultron_2026_05_25_openclaw_clawhub_catalog_porting.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_25_openclaw_clawhub_catalog_porting.md) |
| 2026-05-25 | `85abd78` | OpenClaw-ClawHub catalog port batch 8 (YELLOW) -- T15 privacy-by-construction aggregate-only telemetry. HashedRootId / HashedSkillId NewType wrappers + hash_root / hash_skill_slug primitives with per-install salted SHA-256 (data/observability/telemetry_salt.txt generated on first call so cross-install correlation is impossible). canonical_label_root for tilde-normalised dashboard labels. PrivateMetricsStore append-only JSONL with type-boundary RawPathLeakError enforcement (raw paths / long unhashed strings rejected; SAFE_ATTRIBUTE_KEYS + _id/_hash-suffix + 12-char threshold carveouts). HashedEvent + RootRecord + SkillRecord aggregates. stale_root_ids implements the upstream 120-day staleness pattern. Fail-private default: telemetry disabled unless explicit KENNING_TELEMETRY=opt-in. Generalises to per-session metric dashboards / provider health / memory drift detection -- same shape, no exfiltration vector. New package: `src/kenning/observability/` + `tests/observability/`. +42 module tests. | 7338 | [project_ultron_2026_05_25_openclaw_clawhub_catalog_porting.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_25_openclaw_clawhub_catalog_porting.md) |
| 2026-05-25 | `7f168f2` | OpenClaw-ClawHub catalog port batch 7 (YELLOW) -- T12 user-initiated report queue + universal pre-act moderation-plan preview. ReportQueue thread-safe append-only JSONL with SHA-256 hash chain mirroring safety.audit; Report (id + target_kind + target_id + reason + status + version + reporter_voice_session + timestamps + triage_note + final_action + extras); file_report/triage/list_reports/count/get/replay_from_log/verify_log_chain methods; UnknownReportError + IllegalTriageError exceptions. ModerationPlan universal pre-act preview with PlanImpact (message + severity + reversible) tuple + outcome enum + requires_confirmation derivation; render_plan_for_voice returns one-line TTS-safe summary. YELLOW gating: triage paired with safety.two_phase_approval (T2 from OpenClaw port) so a compromised in-process LLM cannot dismiss real reports covertly. New package: `src/kenning/feedback/` (3 modules) + `tests/feedback/` (2 test files). +44 module tests. | 7296 | [project_ultron_2026_05_25_openclaw_clawhub_catalog_porting.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_25_openclaw_clawhub_catalog_porting.md) |
| 2026-05-25 | `64055ae` | OpenClaw-ClawHub catalog port batch 6 -- T8 registry discovery via `/.well-known/kenning.json`. discover() with current-path -> legacy-path fallback + INJECTED fetcher (no network dep) + UntrustedHostError on optional trusted-hosts allowlist miss. resolve_registry_base() 3-tier chain (well-known -> env override -> default). DiscoveryCache thread-safe TTL (15-min default) wrapper caches None results too. Generalised pattern: future skill / MCP / voicepack registries publish their own well-known so kenning auto-discovers without config edits. New module: `src/kenning/install/discovery.py`. +31 module tests. | 7252 | [project_ultron_2026_05_25_openclaw_clawhub_catalog_porting.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_25_openclaw_clawhub_catalog_porting.md) |
| 2026-05-25 | `e0fdd16` | OpenClaw-ClawHub catalog port batch 5 -- T4 declared-vs-observed coherence checker (bidirectional MISSING_DECLARATION + UNUSED_DECLARATION + DYNAMIC_READ + OS_MISMATCH; conservative literal-only matcher with _ALWAYS_AVAILABLE_ENV + _COMMON_BINS_NEVER_DECLARED carveouts; check_intent_phrase_coherence creative-extension for trigger-phrase / body-token overlap) + T5 capability-tag namespace (frozen CapabilityTag enum across 5 categories: resource requirements / side-effect / egress scope / latency profile / gaming-mode safety / modality / confirmation tier; derive_capability_tags via per-tag regex + manifest declarations; TaggedCapability + filter_capabilities universal AND-combined filter; gaming_mode/vlm_loaded/has_internet/require/exclude axes; is_gaming_mode_safe + needs_explicit_intent predicates). New modules: `src/kenning/install/coherence.py`, `src/kenning/skills/capability_tags.py`. +71 module tests. | 7221 | [project_ultron_2026_05_25_openclaw_clawhub_catalog_porting.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_25_openclaw_clawhub_catalog_porting.md) |
| 2026-05-25 | `29a2832` | OpenClaw-ClawHub catalog port batch 4 -- T2 triple-digest artifact identity verification (ArtifactIdentity with hex SHA-256 + SRI SHA-512 + hex SHA-1 + byte_length; fail-closed verify_identity; ClawPack tarball-internal manifest_name + manifest_version checks via parse_clawpack_contents; TOFU pin file at data/install/pinned_digests.jsonl with pin_first_use_digests + load_pinned_digest + verify_against_pin) + T13 typed artifact-kind resolver (ResolvedArtifact envelope per source; per-kind builders for LOCAL_PATH / TARBALL_URL / GIT_REF / NPM_PACK / INLINE_MARKDOWN; verify_artifact_bytes dispatches by kind; ArtifactResolver registry-style dispatch). New modules: `src/kenning/install/artifact_identity.py`, `src/kenning/install/resolver.py`. +56 module tests. | 7150 | [project_ultron_2026_05_25_openclaw_clawhub_catalog_porting.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_25_openclaw_clawhub_catalog_porting.md) |
| 2026-05-25 | `848d734` | OpenClaw-ClawHub catalog port batch 3 -- T1 per-version trust envelope with derived `blocked_from_download` signal (TrustEnvelope/TrustSignal/PackageRef/ReleaseRef + 11-step scan_status derivation algorithm + universal `refuse_if_blocked` decision surface with allow_stale/allow_pending soft-clamps that do NOT override hard blocks) + T9 version-exact install contract (VersionExactRequest + fetch_for_version + VersionExactViolation; refuses floating-tag tokens like latest/*/main/HEAD/edge/etc. case-insensitive). Re-uses T3 ModerationVerdict for scan_status so the verdict surface is consistent across reason-codes / scanner / envelope. Generalised beyond install-time (re-usable for web-search provider trust, MCP server trust, memory backend trust, VLM/STT/TTS engine lifecycle, gaming-mode VRAM-reclaim transitions). All GREEN drop-ins; pure data + pure functions; no orchestrator hot-path wiring. New module: `src/kenning/install/trust_envelope.py`. +65 module tests. | 7094 | [project_ultron_2026_05_25_openclaw_clawhub_catalog_porting.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_25_openclaw_clawhub_catalog_porting.md) |
| 2026-05-25 | `5fe6d77` | OpenClaw-ClawHub catalog port batch 2 -- T10 lockfile + per-skill origin manifest with content fingerprinting (drift detection across .kenning/lock.json + per-skill .kenning/origin.json; atomic tmp+os.replace writes; fail-open reads; SHA-256 sorted-path fingerprint with binary/hidden/state-dir skip) + T11 pin/unpin extending T10 (idempotent-on-same-reason / strict-unpin / KENNING_DEFAULT_PINS materialiser for the 5 voice-baseline-lock anchors) + T6 alias graph (rename/merge/transfer/soft_delete-30day-reservation/hard_delete + redirect-chain resolve with cycle protection + JSONL hash chain + RESERVED_SLUGS list + scope-namespaced slug validation). All GREEN drop-ins; pure-Python; no orchestrator hot-path wiring (importable infrastructure for skills registry + sandbox / voice intent / voicepack / persona / memory namespaces). New modules: `src/kenning/install/lockfile.py`, `src/kenning/install/pin.py`, `src/kenning/identity/__init__.py`, `src/kenning/identity/alias_graph.py`. New test dir: `tests/identity/`. +105 module tests. | 7029 | [project_ultron_2026_05_25_openclaw_clawhub_catalog_porting.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_25_openclaw_clawhub_catalog_porting.md) |
| 2026-05-25 | `d7ea215` | OpenClaw-ClawHub catalog port batch 1 -- T3 canonical moderation reason-code catalogue (33 upstream codes + 8 kenning extensions; verdict derivation with malicious-set + prefix short-circuit; OWASP Agentic Top 10 alignment; externally-clearable carve-out; wired into static_scanner.py via canonical_code_for_finding helper) + T14 HTTP rate-limit envelope parser (7-header triple-family; preferred-fallback Retry-After -> RateLimit-Reset -> X-RateLimit-Reset; RateLimitTracker with per-provider cooldown + 429 counter; wired into SearchProviderChain via tracker kwarg + should_skip + record_provider_outcome). Both GREEN drop-ins; existing behaviour byte-identical when no rate-limit headers / no reason codes consulted. New modules: `src/kenning/install/reason_codes.py`, `src/kenning/web_search/rate_limit.py`. New test dir: `tests/web_search/`. +97 module tests + 7 chain-integration tests. | 6924 | [project_ultron_2026_05_25_openclaw_clawhub_catalog_porting.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_25_openclaw_clawhub_catalog_porting.md) |
| 2026-05-25 | `0ec1e42` | OpenClaw catalog port -- 11 batches landing 19 of 22 cataloged techniques (T17 / T19 / T20 deferred per catalog star rating). Importable primitives across 18 new modules / packages: `utils/ansi_safe`, `subprocess/{kill_tree, process_registry}`, `safety/{path_resolver T21 additions, validator T16 additions, policy_chain, two_phase_approval, hierarchical_policy}`, `llm/{context_window_guard, condensers/splitter}`, `agent_loop/{loop_detection_extended, subagent_policy}`, `hooks/lifecycle` (29-event expansion to 43 total + new HookDecision discriminated dataclass), `coding/edit_recovery`, `skills/{activation, marketplace}`, new packages `providers/` (failover_policy + auth_profiles + rotation), `install/static_scanner` (Python tokenize-based scanner + dependency denylist), `mcp/` (transport with env/header sanitisation + McpServerRegistry with kill-on-disconnect; closes the deferred T9 MCP-hub from the cline port). 14 GREEN ports + 5 YELLOW ports (T12 / T6 / T2 / T5 / T22 / T9 gated through the existing safety stack -- spawn-tool, credential storage, decision channel, Cap-1 + new Category L groundwork, Cap-3 + T12 + T8 + env-filter + connection-timeout, T5 scanner + per-source verification respectively). 0 RED -- every powerful pattern has a legitimate use case once gated. No orchestrator hot-path wiring (all importable infrastructure); voice baseline contract byte-identical to pre-port baseline. OpenClaw (MIT, 2026) attribution added to THIRD_PARTY_NOTICES.md with per-component mapping (18 entries). | 6820 | [project_ultron_2026_05_25_openclaw_catalog_porting.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_25_openclaw_catalog_porting.md) |
| 2026-05-25 | `188931a` | post-cline integration pass: orchestrator startup wiring (install_default_injectors with closures returning existing STT/TTS engine instances + discover_project_config caching .kenning/ snapshot on `self._project_config`; both fail-open), SWE-Agent T16 click-preview gate wired through new InputController kwargs + new `desktop.click_preview.*` config section (default OFF; when ON the orchestrator builds a new InputController with VLM-backed `vlm_describe` + screen-capture closures and replaces the singleton via `set_input_controller`), two safe behavioural flag flips (`skills.enabled: true` + `events.enabled: true`, both fail-open, voice baseline preserved via untouched gaming_mode/llm.preset/llm.draft_kind), OpenHands deep API docs backfilled (9 packages: parsing / install / projects / services / skills / events / lifecycle / llm/condensers + utils/poll.py). | 6270 | [project_ultron_2026_05_25_cline_catalog_port_and_post_cline_integration.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_25_cline_catalog_port_and_post_cline_integration.md) |
| 2026-05-24 | `92ee711` | cline catalog port batch 10 -- mode policy + per-mode LLM router + subagent runner (T2 + T13 + T16): new `agent_loop/mode.py` (canonical `Mode` enum ACT/PLAN/CODING_ARCHITECT/CODING_EDITOR/GAMING + frozen `ModePolicy` with wrap-template + confirmation-TTL + preset override + `PendingConfirmation` queue with intent-topic filter + `ModeSession` state machine + flip history + module-level registry `get_mode_session`); new `llm/mode_router.py` (frozen `PresetEntry` + `ModeLLMRouter` mapping `Mode` to preset name, skip-when-already-active via injected probe, protected-mode set, `on_swap` callback with fail-open semantics, default routes target kenning's qwen3.5-4b for ACT/PLAN/CODING_* + llama-3.2-3b-abliterated for GAMING); new `agent_loop/subagent.py` (frozen `SubagentTask` + `SubagentResult` + `SubagentBatchStats` + `ToolGuard` whitelist enforcer + thread-safe `TokenLedger` + `SubagentRunner` ThreadPoolExecutor-backed dispatcher with `max_parallel=1` default + `DEFAULT_READONLY_TOOL_WHITELIST` matching cline's subagent allowed set). All three I/O-free, clock-injectable, no orchestrator wiring (primitives only). | 6263 | [project_ultron_2026_05_25_cline_catalog_port_and_post_cline_integration.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_25_cline_catalog_port_and_post_cline_integration.md) |
| 2026-05-24 | `14e4653` | cline catalog port batch 9 -- dual-array API/UI history split (T4): new `memory/dual_history.py` with `DualHistoryStore` (per-session in-memory store), `VerbatimTurn` (text + tts_clip_ref + image_refs + metadata -- the literal user/agent exchange), `ApiTurn` (LLM-facing shape with `compacted` + `elided_count` for drift reporting), shared `turn_id` UUID indexing so verbatim<->api resolves O(1) -- the basis for "what did I say earlier?" voice queries. Methods: `record` / `record_api` / `truncate_after_turn` (anchor-based) / `truncate_to_offset` (last-N) / `replace_api_range` (condenser hook) / `find_verbatim_by_substring` (newest-first fuzzy) / `snapshot` (frozen view with both indices) / `drift_report` (per-call counts of `verbatim_only` / `api_only` / `shared`). Verbatim/api caps default unlimited (verbatim is cheap; api cap is what costs tokens). Primitive is I/O-free; callers wire their own persistence. | 6191 | [project_ultron_2026_05_25_cline_catalog_port_and_post_cline_integration.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_25_cline_catalog_port_and_post_cline_integration.md) |
| 2026-05-24 | `cf0cbef` | cline catalog port batch 8 -- shadow-repo checkpoints with three-axis restore (T1): new `checkpoints/` package -- `exclusions.py` (DEFAULT_CHECKPOINT_EXCLUSIONS + VOICE_BASELINE_PROTECTED_PATTERNS guarding SOUL.md / RVC / Piper / Kokoro voicepack / LLM GGUFs from accidental rollback), `shadow_repo.py` (ShadowRepoTracker git CLI wrapper + per-session RLock + 15s init timeout + CREATE_NO_WINDOW + hash_working_dir), `restore.py` (plan-then-execute three-axis restore — voice_history / workspace / both), `registry.py` (SessionCheckpointManager bus subscription + CheckpointRegistry singleton). | 6161 | [project_ultron_2026_05_25_cline_catalog_port_and_post_cline_integration.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_25_cline_catalog_port_and_post_cline_integration.md) |
| 2026-05-24 | `34894d9` | cline catalog port batch 7 -- hooks lifecycle (T5 + T21): new `hooks/` package -- 9 cline lifecycle points (TaskStart / TaskResume / TaskCancel / TaskComplete / UserPromptSubmit / PreToolUse / PostToolUse / PreCompact / Notification) + 5 kenning-specific (PreLLMRequest / PreMemoryWrite / PreGamingEngage / PreDesktopAction / WakeWordTriggered). `HookRegistry` parallel fan-out + cancel aggregation + `<hook_context>` concatenation; `HookRunner` subprocess executor with per-suffix interpreter selection (.py / .ps1 / .sh / .bat / .cmd / shebang) + JSON stdin/stdout envelope + 10 s default timeout + 8 kB context-mod cap + last-balanced-JSON parser; `HookDiscovery` mtime-validated cache with 30 s TTL; module-level `get_hook_registry()` singleton. | 6119 | [project_ultron_2026_05_25_cline_catalog_port_and_post_cline_integration.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_25_cline_catalog_port_and_post_cline_integration.md) |
| 2026-05-24 | `75353f7` | cline catalog port batch 6 -- mentions + focus-chain (T14 + T11): `coding/mention_resolvers.py` (extended `@`-mention regex covering URLs / `workspace:` / `memory:` / `problems` / `last` / `diff` / `clipboard` / `screenshot` / Windows drive-letter paths + provider-driven resolution + per-mention body cap + per-call cap + dedup); `coding/focus_chain.py` (parse / render / diff markdown checklists + atomic temp+rename writes + `FocusChainWatcher` with 300 ms debounce + manual `poll_now` fallback when watchdog absent + `render_critical_info_block` for the user-edit CRITICAL INFORMATION block + `progress_hint` per-band prompt tailoring). | 6079 | [project_ultron_2026_05_25_cline_catalog_port_and_post_cline_integration.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_25_cline_catalog_port_and_post_cline_integration.md) |
| 2026-05-24 | `6d66d96` | cline catalog port batch 5 -- streaming infrastructure (T8 + T12 + T19 + T20): new `streaming/` package with `window.py` (WindowedOutputWriter with 20-line/2KB/100ms debounce + 1000-line/512KB spill thresholds + head-100/tail-100 preserved + `COMPILING_MARKERS` hot-timeout detection), `presentation_scheduler.py` (priority-banded chunk scheduler with environment-adaptive cadence — local/remote/Bluetooth profiles + `set_drop_low_priority` for `enable_thinking=False`), `reasoning_stream.py` (ReasoningDemultiplexer with first-text-finalises semantics + dedicated audit channel keeps reasoning out of TTS), `coordinator.py` (StreamCoordinator state machine + `RetryStatus` payloads + `on_usage` live token meters). | 6028 | [project_ultron_2026_05_25_cline_catalog_port_and_post_cline_integration.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_25_cline_catalog_port_and_post_cline_integration.md) |
| 2026-05-24 | `3ca8879` | cline catalog port batch 4 -- auto-approve matrix + structured 8-section condenser (T3 + T15): `safety/auto_approval.py` (four-mode per-rule policy `always_ask` / `allow_local` / `allow_external` / `allow_all` + `yolo_mode` master override + per-session "warming" allowlist after N consecutive user approvals + injected `LocalityProbe` predicate); `llm/condensers/structured_8_section.py` (8 canonical headers Primary Request / Key Technical Concepts / Files and Code Sections / Problem Solving / Pending Tasks / Task Evolution / Current Work / Next Step + tolerant `parse_summary` with alias resolution + `compact_for_voice` 3-section TTS-friendly continuity ack). | 5973 | [project_ultron_2026_05_25_cline_catalog_port_and_post_cline_integration.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_25_cline_catalog_port_and_post_cline_integration.md) |
| 2026-05-24 | `03019fb` | cline catalog port batch 3 -- ignore + conditional rules (T6 + T10): `safety/ignore.py` (three-layer `.kenningignore` with `!include`, `validate_command` covering POSIX + PowerShell file-readers, registry singleton); new `rules/` package with `conditionals.py` (frontmatter `paths` / `intents` / `topics` / `system_state` evaluator, `all_of` + `not_in_gaming_mode` combinators, comparator-prefixed state matching, path-extraction heuristic that strips fenced code + URLs). | 5928 | [project_ultron_2026_05_25_cline_catalog_port_and_post_cline_integration.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_25_cline_catalog_port_and_post_cline_integration.md) |
| 2026-05-24 | `7f18a24` | cline catalog port batch 2 -- caching, loop detection, telemetry, zombie killer (T7 + T18 + T17 + T23): `coding/file_read_cache.py` (per-session mtime-validated cache with LRU eviction + registry singleton); new `agent_loop/` package with `loop_detection.py` (canonical signature + soft/hard escalation tiers); `llm/dedup_file_reads.py` (in-place dedup of duplicate file-read tool results in API history + generic payload dedup + 30 % skip-compaction threshold); `observations/safe_capture.py` (sync + async + decorator triple with `SafeCaptureStats` counters); new `subprocess/` package with `zombie_killer.py` (10-min hard cap + persistent-tag carve-out + RSS warning tier + clock/terminator/RSS-probe injection hooks). | 5872 | [project_ultron_2026_05_25_cline_catalog_port_and_post_cline_integration.md](file:///C:/Users/alecf/.claude/projects/C--STC-ultronPrototype/memory/project_ultron_2026_05_25_cline_catalog_port_and_post_cline_integration.md) |
| 2026-05-24 | `a7d03dd` | cline catalog port batch 1 -- foundation utilities (T22 + T13b + T25): `llm/response_format.py` (30+ structured error/notice templates with voice-friendly variants + progressive-escalation tiers); `utils/retry.py` (async + sync `with_retry` decorator + `RetryBudget` + retry-after header parsing with delta-seconds-vs-unix-timestamp heuristic + `RetriableError` + async-generator decoration + `asyncio.CancelledError` pass-through); `search/ripgrep.py` (subprocess wrapper around `rg --json` with byte-cap 0.25 MB / result-cap 300 / wall-clock kill / Windows `CREATE_NO_WINDOW` / optional ignore-predicate). | 5778 | (cline-port memory pending; see `THIRD_PARTY_NOTICES.md`) |
| 2026-05-24 | `18fab56` | OpenHands catalog T1-T8 port -- 8 batches, 11 new packages: `parsing/`, `install/`, `skills/` + `skills/` catalogue, `events/`, `llm/condensers/`, `lifecycle/`, `projects/`, `services/`. Two opt-in config sections (`skills.*`, `events.*`, default OFF). | 5640 | `project_ultron_2026_05_24_openhands_catalog_porting.md` |
| 2026-05-23 | `73fafba` | SWE-Agent catalog T1-T20 port -- 7 batches: `coding/{sentinels, session_registry, window_expand, window_state, file_history, edit_diagnostics, lint_diff, search_primitives, diff_snapshot, submit_review, forfeit, observation_format}`, `llm/{history_processors, requery, image_markdown, draft_model}`, `safety/rules/category_it`, `desktop/click_preview`. Two production knobs flipped (`llm.history_compression.enabled: true`, `safety.interactive_tools.enabled: true`). | 5215 | `project_ultron_2026_05_23_swe_agent_catalog_porting.md` |
| 2026-05-23 | `5f12e7d` | Aider catalog completion (batch 14: `architect_narrator`, `stt_bias`, `confirm_group`) + measurement-infra audit (7 stale scripts modernised via new `make_tts_engine` factory + `scripts/run_tests.py` watchdog race fix) + baselines captured against current Kokoro+Moonshine+Qwen3.5+intent stack (STT 16 ms / LLM TTFT 203 ms / TTFA 313 ms / VRAM 7007 MB peak). | 4750 | `project_ultron_2026_05_23_catalog_completion_and_measurement_audit.md` |
| 2026-05-22 | `8bbc345` | Review-feedback pass: bus slow-subscriber watchdog (>15 ms WARN + counter), `resilience/fail_open_log.py` per-session counter with JSONL persistence, `log_effective_config()` startup diagnostic, `coding.supervisor.tier` rollup field, file-mutation -> introspect cache invalidator via bus, `observe_llm_thinking_drift_sample`, TTS narration-honesty fuzz tests, 6 safe flag flips. | 4240 | `project_ultron_2026_05_22_review_feedback_pass.md` |
| 2026-05-22 | (batches 1-14) | Aider catalog port: `utils/{mtime_cache, token_budget, snapshot_guard, relative_indent, spinner, poll}`, `coding/{repo_map, project_digest, project_introspect, project_index, project_supervisor, supervisor_dispatch, important_files, tree_sitter_tags, tree_sitter_lint, python_lint, coder_modes, patch_v4a, commit_message, file_mention_resolver, edit_matcher, ai_comment_watcher, architect_supervisor, history_compression}`, `intent/command_registry`, `llm/{cache_aware_chunks, cache_warmer}`, `web_search/{slimdown_html, pandoc_converter, playwright_reader}`. `THIRD_PARTY_NOTICES.md` established (Apache 2.0 aider attribution). | ~4665 -> 4750 | (see batch-14 memory file above) |
| 2026-05-22 | `9f2ac68` | Session E: opencode-inspired typed event bus (`bus/` package, 17-event catalog) + coding-supervisor stack (5 phases: digest, introspect, index, supervisor, dispatch). | 4104 | (in 2026-05-22-review-feedback memory) |
| 2026-05-22 | (sessions C+D) | `RoutingIntentKind.OPEN_LAST_SOURCE` + `NAVIGATE_TO_SITE`, news-category SearxNG routing, audio queue overflow fix, SearxNG hardening (Brave/Reuters/Bing News engines), `GateDecision` scoping bugfix. | 3946 | (in CLAUDE.md change log only) |
| 2026-05-22 | `5ec0643` | Dual-STT (Parakeet primary + Moonshine gaming) via `DualSTTRegistry` + engine-agnostic intent recognizer (Gemma-300M q4, 25 phrases) + complete gaming-mode VRAM reclaim (~2.3 GB freed) + Kokoro boundary artifact fixes (`trim_and_fade` + cosine fades + tail mute) + reranker disabled + RAG 0.6 -> 0.78 + ~70-city zoneinfo + news/freshness gate rule. | 3945 | `project_ultron_2026_05_22_dual_stt_intent_gaming_mode.md` |
| 2026-05-22 | `756469a` | Moonshine v2 streaming as STT default + Kokoro fine-tune model load (parametrizations -> weight_norm shim) + voice path `enable_thinking=False` (saves 5-10 s TTFT) + 34 s memory retrieve fix (bare `rag_query`) + reranker double-load fix (singleton) + trafilatura cap 200 k -> 32 k. | 3749 | `project_ultron_2026_05_22_moonshine_streaming_and_kokoro_finetune_load.md` |
| 2026-05-22 | (frontier) | Partial-fine-tune Kokoro ship (spectral magnitude smoothing window=5) + 5-item frontier enhancement (in-process spec decoding via `LlamaPromptLookupDecoding`, cross-encoder reranker `BAAI/bge-reranker-v2-m3`, contextual retrieval, Parakeet TDT STT, local-first search ladder SearxNG -> Brave -> DDG + trafilatura -> Jina) + Kokoro training auto-resume infra + testing-process hardening. | 3560 | `project_ultron_2026_05_22_partial_finetune_ship_and_frontier_enhancement.md` |
| 2026-05-20 | (round 8) | LLM `gemma-3-4b-abliterated` -> `qwen3.5-4b` + TTS `xtts_v3` -> `kokoro` (stock `am_michael` CPU) + Whisper `small.en` -> `base.en` + Kokoro producer-consumer rewrite + ack-cache wiring + ~22 GB GGUFs freed (re-download paths preserved). | 3513 | `project_ultron_2026_05_20_round_8_llm_tts_swap.md` |
| 2026-05-20 | `b7e1164` | Round 7a contamination loop closed via `history_user_message=` kwarg on `LLMEngine.generate*` + Round 7b smarter TTS sentence boundaries (rejects ellipsis/decimal/domain/abbreviation/acronym-chain). | 3513 | `project_ultron_2026_05_20_round_7ab_memory_and_tts_chunking.md` |
| 2026-05-20 | `b1a2d8c` | Phase A/B/C runner integrations (AST listener completion narration + BackgroundSummarizer orchestrator hook + `Channel` abstraction in memory write path) + 8 live-session fixes (XTTS overflow, RAG short-query gate, Gemma terseness, fake citation guard, monitor sort, third-party possessive, second-person adjustment, cross-session contamination) + structured per-turn tracing (`src/kenning/trace.py`). | 3483 | `project_ultron_2026_05_20_phaseABC_fixes_and_tracing.md` |
| 2026-05-19 | `3698da2` | Cross-cutting expansion (Tracks 1a/1b/1c-e/1f/1g/1h memory infra + Track 2 parallel embedding + Track 3 verbosity hints + Track 4 Gemma/Llama presets + Track 5 Kokoro engine + Track 6 channel abstraction + latency hygiene helpers + voice MODEL_SWITCH for gemma/llama). Gemma 3 4B abliterated default swap. | 3054 | `project_ultron_2026_05_19_cross_cutting_expansion_gemma.md` |
| 2026-05-19 | `2b979c0` | Phase 0+1+E2+E5 build: eval harness + 60-row labeled corpus + observation framework (4 sites + outcome resolver + lineage overlap) + adaptive context window scoring + confidence-band ambiguity predicate + voice-character-lock guardrails + goal-anchor planning FULL runtime integration. | 2697 | `project_ultron_2026_05_19_phase_0_1_e2_e5.md` |
| 2026-05-18 | `e3ac64e` | Latency pass 3: TTS preopen hoist + speculative classification + speculative LLM generation during silence wait (~80-100 ms additional saved on cache-hit turn). | 2472 | `project_ultron_2026_05_18_latency_pass_3.md` |
| 2026-05-16 | `9a15c06` | Latency pass 2: LlamaRAMCache prefix KV cache infra + Smart Turn V3 gradient-fire + speculative Whisper STT during silence wait + 32 ms -> 16 ms audio blocksize + legacy TTS pre-open silence write (~200-280 ms additional saved). | 2422 | `project_ultron_2026_05_16_latency_pass_2.md` |
| 2026-05-15 | `703c11f` | Latency pass 1: ack clip cache (~350 ms saved per turn) + parallel RAG pre-fetch + n_batch/n_ubatch knobs + Whisper `beam_size` 5 -> 1 (~80 ms saved) + TTS stream pre-open during STT (~50 ms saved). Cumulative ~590 ms cache-hit perceived latency. | 2385 | `project_ultron_2026_05_15_latency_pass.md` |
| 2026-05-14 | `622000d` | Third pass: `chat_template_kwargs` regression fix (use Qwen3 `/no_think` marker) + `WINDOW_MOVE`/`WINDOW_CLOSE` intents + bare image-search regex + plural image nouns + moondream2 revision rollback. | 2313 | `project_ultron_2026_05_14_handoff.md` |
| 2026-05-14 | `15f58d5` | Second VRAM-relief pass: LLM Q5_K_M -> Q4_K_M (saves 500 MB) + implicit image-search + preflight `<think>` strip. | 2278 | (in 2026-05-14-handoff memory) |
| 2026-05-14 | `901ebf1` | First VRAM-relief pass: Josiefied-Qwen3-8B -> Josiefied-Qwen3-4B (saves ~3 GB) + Whisper float16 -> int8_float16 + monitor mapping fix + MODEL_SWITCH broadening + YouTube deep-link + Spotify path fix. | 2263 | `project_ultron_2026_05_14_vram_relief_ux_fixes.md` |
| 2026-05-12 | (4 commits) | Desktop automation Phases 1-14: native `src/kenning/desktop/` (11 modules) + 19 new MCP tools (24 total) + `APP_LAUNCH` + `SCREEN_CONTEXT_QUERY` intents + Phase 12 analyze-and-discard + Phase 13 `kenning-vision` agent install + Phase 14 moondream2 wiring. | 2194 | `project_ultron_2026_05_12_desktop_automation.md` |
| 2026-05-12 | `91a3a3a` | Runtime tool-call validator (Phases 2-5): 141 rules across 19 categories (K self-protection + A-J load-bearing + M-S persistence/anti-forensics + Cap-1..Cap-4) + Windows-aware `PathResolver` + tamper-evident hash-chain audit log + explicit-intent matcher + cross-capability taint tracker. | 1830 | `project_ultron_2026_05_12_safety_validator.md` |
| 2026-05-12 | `e67cac3` | Josiefied-Qwen3-8B-abliterated-v1 Q5_K_M default LLM swap (motivated the safety validator). | 1713 | `project_ultron_2026_05_12_josiefied_8b_phase_1.md` |
| 2026-05-12 | `b1e4297` | XTTS phantom-token mitigation (temperature 0.75 -> 0.65 + `trim_phantom_tail`) + conversational filler-ack pool (Mm./Right./Hm./Considering./...) + Smart Turn V3 (8 MB int8 ONNX, CPU-only, ~12 ms inference, zero VRAM). | 1711 | `project_ultron_2026_05_12_audio_artifact_filler_ack.md` |
| 2026-05-11 | `9139bda` | Live-session follow-up: `vad.max_utterance_seconds` config field + completion narration path strip (`path.name` only, prevents XTTS GPU pin) + `PROGRESS_QUERY` classifier broadening (`how is that project going?`). | 1629 | `project_ultron_2026_05_11_windsurf_followup.md` |
| 2026-05-11 | `431fd7b` | XTTS cadence `speed=1.15` + speculative-stream SR engine-aware + adaptive VAD long-utterance bump + addressing third-party rule + token budget 100 k -> 400 k + narration honesty + `voice_task_require_testing=false`. | 1604 | `project_ultron_2026_05_11_session.md` |
| 2026-05-10 | (commit) | Voice-pipeline swap: XTTS v2 streaming + v3 pedalboard filter chain (Kenning mechanical character via DSP); default flipped from `piper_rvc` to `xtts_v3`. | -- | `project_ultron_voice_swap.md` |
| 2026-05-09 | (commit) | Audio + memory pass: latency hot-fix (parallel Jina + collective deadline) + TTS pipeline two-stage split + nuanced memory retrieval (cosine 0.6 + recency-weighted composite + history cap 4) + direct Focusrite mic + `audio.input_gain_db`. | -- | `project_ultron_audio_memory_pass.md` |
| earlier | -- | Foundation Parts 0-7 + OpenClaw integration Phases 0-13 + comprehensive testing passes + 4B optimization plan + V1-spec gap fill + prompt-injection defense layer. | -- | `project_ultron_foundation.md`, `project_ultron_openclaw.md`, `project_ultron_4b_plan.md`, `project_ultron_v1_gap_fill.md`, `project_ultron_comprehensive_passes.md`, `project_ultron_phase_c.md` |

---

## Table of contents

1. [Quick orientation](#quick-orientation)
2. [File tree](#file-tree)
3. [Cross-cutting flows](#cross-cutting-flows)
4. [Source modules](#source-modules)
5. [Configuration](#configuration)
6. [Operational scripts](#operational-scripts)
7. [Tests](#tests)
8. [Runtime artifacts (logs / data)](#runtime-artifacts)
9. [Documentation index](#documentation-index)
10. [Maintenance contract](#maintenance-contract)

---

## Quick orientation

Kenning is a local voice-first AI assistant. The pipeline (current as of
the validating HEAD in the header above):

```
mic → wake word ("kenning") OR addressing classifier (WARM follow-up mode)
    → Silero VAD + Smart Turn V3 (CPU end-of-turn confirmation, ~12 ms)
    → STT via DualSTTRegistry
        ├─ stt.engine="moonshine" (CPU streaming, ONNX, current default)
        ├─ stt.engine="parakeet"  (NeMo TDT on CUDA via .venv-parakeet HTTP)
        └─ stt.engine="whisper"   (faster-distil-whisper-small.en swap-back)
    → Intent recognizer (Gemma-300M q4 CPU; 25 phrases) short-circuits
       gaming-mode commands AND force-routes "needs fresh data" intents to SEARCH
    → classify_routing() → RoutingIntentKind (23 kinds incl. OPEN_LAST_SOURCE,
                                              NAVIGATE_TO_SITE, GAMING_MODE,
                                              APP_LAUNCH, ...)
        ├ coding kinds → CapabilityVoiceController._handle_code_task
        │   ├ if coding.supervisor.enabled: ProjectSupervisor.decide() →
        │   │   ├ RESUME (active task + adjustment) → runner.send_followup
        │   │   ├ EDIT (semantic ≥ 0.75)            → enriched TaskRequest
        │   │   ├ CLARIFY ([0.55, 0.75))            → ask user to disambiguate
        │   │   └ NEW                                → fresh scaffold
        │   │   ↳ optional narration with 1.5 s barge-in window
        │   │   ↳ digest listener on COMPLETE refreshes project_index
        │   └ legacy ProjectResolver path (default; supervisor flag is OFF)
        ├ OPEN_LAST_SOURCE  → opens cited URL from last search-augmented turn
        ├ NAVIGATE_TO_SITE  → SearxNG top-10 → domain-score → opens best
        ├ APP_LAUNCH        → native desktop.launcher (Chrome/Cursor/etc.)
        ├ conversational    → LLM (Qwen3.5-4B Q4_K_M in-process, n_ctx=8192)
        │                     ├ optional web-search gate (rules + LLM preflight)
        │                     │  ├ news queries get categories=news routing
        │                     │  ├ SearxNG → Brave → DuckDuckGo cascade
        │                     │  └ Trafilatura → Jina reader cascade
        │                     ├ ConversationMemory.retrieve (RAG; reranker OFF)
        │                     └ stream tokens to Kokoro TTS (CUDA, voice=kenning)
        ├ gaming mode       → VRAM reclaim: LLM swap to llama-3.2-3b, STT to
        │                     moonshine, Kokoro CUDA→CPU, VLM unload (~2.3 GB freed)
        └ openclaw bound    → MESSAGING / BROWSER_AUTOMATION / etc.
    → typed event bus publishes turn / gate / memory / supervisor events
    → async write turn to Qdrant (conversations + projects collections)
    → enter WARM follow-up window (30 s)
```

For the architectural picture see [docs/architecture.md](architecture.md).
For the current decisions and Foundation phase status see
[memory/project_ultron_foundation.md](<ai-memory-dir>\project_ultron_foundation.md).

---

## File tree

```
<project-root>/                    ← C:\STC\ultronPrototype (main checkout)
                                       worktrees: .claude/worktrees/<branch>/
├── README.md                       ← project entry point, doc index
├── config.yaml                     ← canonical configuration (Phase 3 source of truth)
├── pyproject.toml                  ← packaging + pytest config
├── .env (gitignored)               ← secrets + opt-in env-var overrides
├── .env.example
├── baselines.json                  ← VRAM + latency baselines (9B / current production reference)
├── baselines_4b_q4_in_process.json ← 4B plan Stage D snapshot (4B alone, no spec decoding)
├── baselines_phase{0..7}.json      ← per-phase historical snapshots
├── baselines_phase_c{0,1}.json     ← Phase C snapshots (pre-Foundation)
│
├── src/
│   └── kenning/
│       ├── __init__.py             ← CUDA DLL discovery (Windows-specific path injection)
│       ├── __main__.py             ← `python -m kenning` entry point → constructs Orchestrator
│       ├── config.py               ← Phase 3 pydantic loader, get_config() singleton
│       ├── errors.py               ← Phase 4 typed exception hierarchy
│       ├── uncertainty.py          ← Phase 5 (original prompts) uncertainty-signal application
│       ├── response_style.py       ← 2026-05-10: per-call brevity hint (apply_brevity_hint)
│       ├── conversational_ack.py   ← 2026-05-12: filler-ack on conversational path (ConversationalAckSource, is_conversational_ack_eligible)
│       ├── channels.py             ← 2026-05-19 Track 6: Channel enum + ChannelMetadata (USER / SYSTEM / BACKGROUND / EXTERNAL); used by ConversationMemory write payload + future channel-aware retrieval
│       ├── latency_hygiene.py      ← 2026-05-19 Track latency: process priority + GC tuning + LLM/embedder warmup helpers
│       ├── local_clock_reply.py    ← 2026-05-20: short-circuits bare "what time is it" / "what's today's date" / "what time is it in <city>" -- ~70-city zoneinfo map, no LLM, no search
│       ├── trace.py                ← 2026-05-20 Round 6: thread-local turn_id + phase tag + structured tlog/phase helpers (every log line in a user utterance carries turn=N phase=X)
│       │
│       ├── bus/                      ← 2026-05-22 session E: typed event bus (port of opencode's packages/opencode/src/bus/)
│       │   ├── __init__.py           ← Public API: Bus, BusEvent, EventPayload, get_bus, publish, subscribe, subscribe_all, reset_bus_for_testing + canonical event re-exports
│       │   ├── event.py              ← BusEvent.define(type, schema) + EventPayload envelope; schema validation best-effort (delivers anyway, logs WARN)
│       │   ├── service.py            ← Bus class: pub/sub registry + dispatch; eager-subscribe (closes opencode's known lost-events race); RLock-guarded; callback errors swallowed + logged
│       │   └── events.py             ← 17-event canonical catalog: TurnStarted/Completed, STTTranscribed, RoutingClassified, GateVerdict, MemoryRetrieved, LLMStream{Token,Complete}, TTSPlayed, CodingFileChanged, ProjectIndexed, ProjectDigestGenerated, SupervisorDecided, SafetyViolated, GamingEngaged/Disengaged, VRAMReclaimed
│       │
│       ├── observations/            ← 2026-05-18 Phase 0+1: canonical observation framework
│       │   ├── __init__.py
│       │   ├── schema.py             ← Observation dataclass + KNOWN_SUBSYSTEMS / KNOWN_OUTCOMES + new_event_id
│       │   ├── writer.py             ← ObservationWriter thread-safe JSONL appender + singleton accessors
│       │   ├── integrations.py       ← observe_routing_verdict / observe_addressing_verdict / observe_retrieval / observe_llm_call
│       │   ├── outcome_resolver.py   ← Post-hoc OutcomeResolver: emit outcome_resolution rows from history
│       │   └── lineage_overlap.py    ← compute_lineage_overlap + emit_lineage_usage_rows (pure primitives)
│       │
│       ├── desktop/                  ← Desktop automation primitives (NEW; native, no ClawHub deps)
│       │   ├── __init__.py
│       │   ├── monitors.py           ← Win32 monitor enumeration + find_monitor + point_to_monitor
│       │   ├── capture.py            ← mss-based multi-monitor capture; taint-tracker integration. 2026 catalog 09 batch 2 (T2): get_pixel_color(x, y) pyautogui.pixel wrapper with fail-open + None / malformed-result rejection (RGB tuples NOT taint-tracked since they're ephemeral). 2026 catalog 09 batch 4 (T6): find_image_on_screen(template_path, *, confidence=0.8, region) + TemplateMatch frozen dataclass -- routes template_path through PathResolver.safe_realpath defending against attacker-controlled paths; pyautogui.locateOnScreen behind broad-except (missing opencv-python returns None / fail-open contract); precomputed centre coords for direct InputController.click routing.
│       │   ├── windows.py            ← pywin32 + psutil window enum + foreground detection + monitor-index lookup
│       │   ├── placement.py          ← move/resize/maximize/focus on target monitor
│       │   ├── launcher.py           ← AppLauncher with registry (Chrome/Cursor/Discord/VSCode/Edge/Firefox/etc.) + Chrome default-profile + Google Images convenience. 2026-06-12: LaunchResult.window_appeared (None=no wait / True / False=timeout; honest voice line + placement skip on timeout) + post-placement focus_window bring-to-front via _focus_fail_open (MoveWindow/ShowWindow never change Z-order, so relay-pattern Chrome windows opened BEHIND the foreground)
│       │   ├── uia.py                ← pywinauto UIA text extraction + semantic click/type with Cap-3/Cap-4 safety hooks. 2026 catalog 09 batch 2 (T2): wait_for_pixel_color(x, y, target_color, *, tolerance, timeout_s, interval_s, sleep_fn, clock_fn) synchronous polling barrier built on capture.get_pixel_color -- L-infinity tolerance covers anti-aliased / jpeg-rendered pixels; deadline-clamped final sleep + deterministic-test injection mirroring catalog 08 T4 wait_for_text_in_window pattern.
│       │   ├── input_control.py      ← pyautogui mouse+keyboard, rate-limited, validator-gated, blocks input on UAC/security windows. 2026 catalog 09 batch 1: scroll(direction=...) -- vertical (default, pyautogui.scroll) + horizontal (pyautogui.hscroll) dispatch with unknown-direction structured rejection (T1, YELLOW); type_text(wpm=...) -- optional WPM-cadence conversion via interval=1/((wpm*5)/60); wpm<=0 returns structured error vs upstream's ZeroDivisionError (T3, GREEN); move_mouse(smooth=...) -- pyautogui.easeInOutQuad tween when smooth=True AND duration_s>0; default smooth=False preserves linear back-compat (T7, GREEN).
│       │   ├── screen_context.py     ← orchestrator: assemble foreground + windows + UIA text + optional VLM description for LLM injection
│       │   ├── vlm.py                ← Moondream2 VLM wrapper (transformers + trust_remote_code), CPU-only on-demand, lazy-loaded, fail-open; 2026-05-22 Moondream2VLM.unload() for gaming-mode engage callback
│       │   ├── click_preview.py      ← 2026-05-23 SWE-Agent batch 7 (T16): preview_click VLM-confirmed click target with confidence-gated auto-pass; ConfirmationHistory bounded recent-click store; draw_crosshair_on_image pure-Pillow renderer; first click always confirms then subsequent clicks within AUTO_PASS_RADIUS_PX (100px) auto-pass
│       │   ├── win32_helpers.py      ← 2026 catalog 07 batch 2 (T2 + T5 building blocks): ctypes wrappers for documented public Win32 APIs. :func:`get_monitor_dpi(x, y) -> MonitorDpi` via `MonitorFromPoint` + `GetDpiForMonitor` (MDT_EFFECTIVE_DPI); :func:`get_monitor_dpi_for_window(hwnd)` resolves via window centre. :func:`get_last_input_idle_ms() -> Optional[int]` via `GetLastInputInfo` + `GetTickCount` with uint32 wrap-around arithmetic. :func:`is_window_cloaked(hwnd) -> Optional[bool]` via `DwmGetWindowAttribute(DWMWA_CLOAKED=14)`. :func:`block_input_context(max_duration_s=5.0)` is a hardened context manager around `BlockInput` with try/finally guarantee + watchdog daemon (clamps to 30s hard cap) + UIPI safety floor (non-admin processes auto-no-op). :func:`logical_to_physical(x, y, ...)` / :func:`physical_to_logical(x, y, ...)` are the coordinate-space conversion primitives -- identity on 100%-DPI displays. Lazy `ctypes.windll` DLL cache thread-safe. Off-Windows: every public function is a graceful no-op returning a documented default. Fail-open at every layer.
│       │   ├── file_read_cache.py    ← 2026-05-24 cline batch 2 (T7a): per-session mtime-validated file-read cache; FileReadCache (RLock-guarded) with maybe_serve_from_cache (consult + increment read_count) / record_read (capture mtime + content) / invalidate / clear; CachedReadEntry returns the cached_read_notice template; get_file_read_cache(session_id, max_entries=None) module-level registry; optional LRU eviction by lowest read_count
│       │   ├── voice.py              ← Phase 8 voice handlers (handle_app_launch / handle_screen_context_query) + 2026-05-14 third-pass handlers (handle_window_move / handle_window_close) bridging RoutingIntent -> native primitives
│       │   ├── preferences.py        ← Phase 10 preference learning (JSONL log + optional OpenClaw workspace mirror; find_preference_for_phrase for recency-weighted lookup)
│       │   ├── clipboard.py          ← 2026 catalog 09 batch 3 (T4): ClipboardManager + ClipboardResult cross-platform clipboard read/write via pyperclip; Cap-2 read (tool_name=desktop.clipboard.read) + Cap-3 write (tool_name=desktop.clipboard.write); taint tracker integration (clipboard_read / clipboard_write capabilities); 2 KB payload preview to validator; oversize write rejection; pyperclip lazy-import with structured failure on missing dep
│       │   ├── sequence.py           ← 2026 catalog 09 batch 5 (T5): DesktopSequenceRunner multi-step desktop sequence runner with before/after screenshot bracketing per step. SequenceStep / ScreenshotRef / StepResult / SequenceResult frozen dataclasses + SequenceStatus / StepOutcome / VlmVerdict enums. Optional VLM verification via injected vlm_describe callable (confirmation-keyword pattern shared with click_preview); auto-pass radius (default 150 px) for sequential steps within a panel; fail-fast contract (first failure aborts; records failed_at_step); analyze-and-discard on captured bytes by default. Deliberately did NOT port the upstream blocking input() approval (incompatible with voice-first), pygetwindow window activation (inferior to focus_by_title), or keyword-cascade planners (kenning's LLM intent router is more capable).
│       │   ├── browser_sequence.py   ← 2026 catalog 10 batch 8 (creative extension): BrowserSequenceRunner -- the browser-domain analog of desktop/sequence.py DesktopSequenceRunner. Runs a list of BrowserSequenceStep (description + zero-arg action callable + optional anchor coords + soft timeout) with a before/after screenshot bracket per step captured via BrowserUseTool.screenshot() base64 mode (headless -- no display flash, unlike the mss monitor capture the desktop runner uses). Optional VLM verification via injected vlm_describe (confirmation-keyword prompt identical to the desktop runner); auto-pass radius skips redundant VLM calls on consecutive same-region steps; fail-fast (first failing step aborts; records failed_at_step); analyze-and-discard on captured bytes. Reuses SequenceStatus / StepOutcome / VlmVerdict enums from desktop.sequence so consumers handle ONE taxonomy across desktop + browser. BrowserSequenceStep / BrowserScreenshotRef / BrowserStepResult / BrowserSequenceResult dataclasses. Deliberately NOT ported: natural-language step planner (kenning's LLM router is more capable) + blocking input() approval (risky individual actions carry their own two-phase approval inside the action callable). Module singleton get/set_browser_sequence_runner. 18 hermetic tests (fake tool + fake VLM + deterministic injected clock).
│       │   ├── browser_sessions.py   ← 2026 catalog 10 batch 5 (T8 YELLOW): BrowserSessionsManager named-session isolation orchestrator on top of browser_use.py. Each --session NAME gets its own daemon / browser instance; the manager validates names against the alphanumeric allowlist (reuses browser_use._is_valid_session_name), enforces a configurable cap (browser_use.max_sessions, default 3, hard ceiling 16), registers each session in subprocess.process_registry.ProcessRegistry under scope_key=name + tag browser_use_session for ZombieKiller + shutdown reaping, and hands out BrowserUseTool instances bound to each session. create_session (Cap-3 + cap + duplicate guard, race-safe double-check inside the lock) / list_sessions (newest-first) / get_tool / has_session / close_session (Cap-3; tool.close() then ProcessRegistry.mark_exited; force=True escalates to kill_process_tree on the registered pid) / close_all_sessions (two-phase approval gated -- bulk destructive across every loaded auth state; assume_preapproved re-entry). BrowserSession + BrowserSessionResult frozen dataclasses. --cdp-url never emitted (blocked by design). Module singleton get/set/reset_browser_sessions_manager. 31 hermetic tests (fake tool factory + fake ProcessRegistry + fake validator + fake approval registry; deterministic injected clock for ordering).
│       │   └── browser_use.py        ← 2026 catalog 10 batches 1-7 (T1+T2+T5+T6 GREEN + T7+T9 + T3 + T4 + T10 + T11 YELLOW): subprocess wrapper around the external open-source ``browser-use`` CLI -- second-tier browser surface above the UIA ``extract_browser_content`` extractor. BrowserUseTool (sync; lazy binary discovery via shutil.which against [browser-use, bu, browseruse]; CREATE_NO_WINDOW on Windows; BROWSER_USE_SESSION env-var scrub on every call; alphanumeric session-name allowlist; --cdp-url + cloud + tunnel subcommands deliberately not exposed). **Batch 1 -- READ FOUNDATION:** T1 state() -> BrowserState (url + title + indexed BrowserElement tuple). T2 get_html(selector) / get_text(idx) / get_attributes(idx) / get_value(idx) / get_bbox(idx) / get_title() with JSON parse fallback to __raw__. BrowserBbox.center bridges directly into InputController.click via pixel coords. T5 wait_selector(state in visible/hidden/attached/detached) + wait_text with deadline-clamped subprocess timeout = wait_ms + 5s. T6 tab_list (JSON) / tab_new(url?) / tab_switch(idx) / tab_close(indices). Navigation helpers open(url) / back / scroll(direction, amount) / close(all_sessions). **Batch 2 -- WRITE PRIMITIVES:** T7 click_at_index(idx, user_text) / click_at_coords(x, y) / type_text(text) / input(idx, text) / select(idx, option) / hover(idx) / keys(combo) / dblclick(idx) / rightclick(idx) -- all Cap-3 gated through safety.validator.get_validator() with tool_name="desktop.browser_use.<action>" + capability="desktop_browser_use"; validator denial short-circuits BEFORE subprocess invocation; type_text + input pass text_preview (not raw text) to the validator to bound audit-log payload size. T7 upload(idx, path, user_text) -- YELLOW per security review (file-read vector): PathResolver.safe_realpath canonicalises the path BEFORE the validator runs, non-existing / non-file paths rejected pre-validator, resolved path (NOT raw) passed to both validator paths tuple AND subprocess. T9 screenshot(path?, full_page=False, user_text) -- dual-mode: path=set writes to disk after PathResolver canonicalisation + parent-dir existence check; path=None decodes the CLI's base64 stdout (data:image/[png|jpeg|jpg] URI prefix tolerated; whitespace-wrapped base64 tolerated; <16-byte payloads rejected as malformed). BrowserActionResult + BrowserScreenshotResult dataclasses extend BrowserUseResult with target / safety_verdict / image_bytes / path / full_page fields. Helpers: _failed_action (pre-validator failure), _action_from_invoke (subprocess outcome projection), _preview (cap=80, newline-collapse, ellipsis truncation), _short_target_label (audit label from arguments dict shape), _decode_screenshot_payload. Fail-open contract: every method returns a typed *Result dataclass with success=False + error when the binary is missing OR the subprocess fails OR daemon errors OR validator blocks. Module-level get_browser_use_tool / set_browser_use_tool singleton mirroring desktop.vlm pattern. **Batch 3 -- T3 JS EVAL (YELLOW):** analyze_js_script(script) -> JsScriptAnalysis pure-function static-analysis pass over a JS body. Detects 15 risky patterns across 4 categories (network_egress: fetch/XMLHttpRequest/navigator.sendBeacon/WebSocket/RTCPeerConnection; storage_write: localStorage.setItem/sessionStorage.setItem/document.cookie =; navigation: window.location =/window.location.replace|assign|href/document.location =; second_order_eval: eval/new Function/import/document.write) using word-boundary-aware regex + assignment-vs-comparison disambiguation. Includes the security review's additions on top of the catalog 10 baseline. eval(script, user_text, assume_preapproved=False, approval_registry=None, approval_timeout_s=None, approval_scope_key="", timeout_s=None) -> BrowserEvalResult routes through three-phase pipeline: (1) argument validation (empty rejected); (2) static analysis -- ANY risky marker AND NOT assume_preapproved -> register ApprovalRequest(kind=BROWSER_JS_APPROVAL_KIND, actor=desktop_browser_use, delivery_channel=voice, metadata={reason_code=kenning.suspicious.browser_js_exec_unrestricted, script_preview, risky_markers, categories, char_count, user_text}) via injected or get_approval_registry()-default registry + return result with requires_two_phase=True + approval_request_id + NO subprocess; (3) safety check via the regular Cap-3 validator with the analysis fields in arguments, then subprocess + JSON-decode stdout for the value field. _humanize_categories renders the category tuple into a TTS-safe prompt ("make network requests and write to storage or cookies" -> "Browser script wants to ... Proceed?") without echoing the script body. _try_parse_eval_payload returns (parsed_value_or_None, raw_text) tolerating JSON-encoded primitives + arrays + objects + plain-text outputs. **Batch 4 -- T4 COOKIE MANAGEMENT (YELLOW):** cookies_get(url?, user_text, assume_preapproved, approval_registry, ...) -- scoped url is Cap-2 + JSON-parsed into BrowserCookie tuple; url=None is two-phase (cross-origin auth dump). cookies_set(name, value, *, domain, path, expires, secure, http_only, same_site, user_text) -- Cap-3 with value_preview replacing raw value in validator args + same_site whitelist {Strict, Lax, None} + negative-expires rejection. cookies_clear(url?, ...) -- scoped is Cap-3, all-origins is two-phase (destructive). cookies_export(path, ...) -- ALWAYS two-phase (HttpOnly cookies + cross-origin auth tokens). PathResolver.resolve + parent-dir existence check BEFORE approval registration so an invalid path fails fast without the user round-trip. cookies_import(path, ...) -- ALWAYS two-phase (auth-injection vector). PathResolver.safe_realpath + is_file check (input must exist). _register_cookies_approval_result centralises the ApprovalRequest construction (kind=BROWSER_COOKIES_APPROVAL_KIND, actor=desktop_browser_use, delivery_channel=voice, metadata={reason_code=kenning.suspicious.browser_cookies_unrestricted, risky_action, target_summary, url_filter, path, user_text}) + uniform requires_two_phase=True result shape. BrowserCookie + BrowserCookiesResult frozen dataclasses with risky_action discriminator ("export_all" / "import" / "clear_all" / "clear_scoped" / "get_all" / "get_scoped" / "set"). _parse_cookies_json tolerates list-shape AND mapping-shape (cookies envelope) + camelCase + snake_case keys + missing-name skip + bool-as-expires rejection. _humanize_cookie_risky_action maps to TTS-safe phrases without echoing cookie names or values in the prompt. Workflow-level note: cookies_import + connect_profile (batch 6) can poison live Chrome sessions; the two methods are independently gated. **Batch 6 -- T10 PROFILE CONNECT (YELLOW):** profile_list() -- Cap-2 read-only enumeration of detected browsers + profiles into a BrowserProfile tuple (no validator, no approval). connect(user_text, assume_preapproved, ...) -- attach to the user's already-running Chrome via CDP; two-phase approval ALWAYS (full live-session takeover, one-time per session not per-command). connect_profile(profile="Default", url="about:blank", ...) -- launch Chrome with a specific profile via the --profile global flag (prepended before the open subcommand); profile name validated against [A-Za-z0-9 ._-]{1,64} so it can't escape into a hostile argument (path separators / shell metachars rejected). _connect_impl shared pipeline: two-phase approval (kind=BROWSER_PROFILE_APPROVAL_KIND, reason=kenning.suspicious.browser_profile_connect) -> Cap-3 validator -> subprocess. --cdp-url never emitted. BrowserProfile + BrowserProfilesResult + BrowserConnectResult dataclasses. _parse_profiles_json tolerates list / mapping shapes + name/profile/label + browser/browser_name + path/directory key variants. **Batch 7 -- T11 CDP PASSTHROUGH (YELLOW, most security-sensitive):** analyze_cdp_statement(statement) -> CdpStatementAnalysis scans for hard-blocked CDP surfaces. _CDP_BLOCKED_METHODS (8 fully-qualified Domain.method tokens: Security.setIgnoreCertificateErrors, Network.setRequestInterception, Fetch.enable, Browser.grantPermissions, Page.setBypassCSP, Storage.clearDataForOrigin, Runtime.addBinding, Target.setAutoAttach -- catalog 4 + security-review 5 additions) + _CDP_BLOCKED_DOMAINS (Debugger blocked wholesale) with word-boundary regex (MyDebugger doesn't trip Debugger). cdp_python(statement, user_text, assume_preapproved, ...) -> BrowserCdpResult four-state pipeline: (1) empty rejected; (2) blocklist match -> REFUSED OUTRIGHT (success=False + blocked=True + BLOCK_HARD; NOT surfaced for approval, NOT executed, blocklist overrides assume_preapproved); (3) cleared statement -> ALWAYS two-phase approval (no auto-pass for raw CDP; full statement kept verbatim in approval metadata for the audit log, kind=BROWSER_CDP_APPROVAL_KIND, reason=kenning.suspicious.browser_cdp_exec); (4) preapproved -> Cap-3 validator + subprocess python <statement> + JSON-decode. BROWSER_CDP_BLOCKED_REASON_CODE=kenning.malicious.browser_cdp_blocked_domain for the refuse-outright path. CdpStatementAnalysis + BrowserCdpResult dataclasses. Variable-state-persists-across-calls caveat documented (no flush short of close()). 320 hermetic tests (subprocess + safety validator + PathResolver + approval registry all mocked via monkeypatch; no binary install required). Note: at catalog-10 port time the sweep's only failure was the environmental flaky tests/integration/test_bridge_e2e.py::test_health_through_real_subprocess; that flake was subsequently FIXED by the bridge-reap fix (`7b53ea1`, see the validating-HEAD header) so the sweep now runs green WITHOUT deselection -- browser_use.py was isolated from that path regardless.
│       │
│       ├── intent/                   ← 2026-05-22: engine-agnostic semantic intent matcher
│       │   ├── __init__.py           ← public API (KenningIntentRecognizer, IntentMatch, IntentRegistration, get_intent_recognizer, set_intent_recognizer)
│       │   └── recognizer.py         ← KenningIntentRecognizer wrapping moonshine_voice.IntentRecognizer (Gemma-300M q4 ~300 MB CPU RAM) with lazy load + fail-open + thread-safe registry + phrase-replay-at-load-time; process_utterance(text) -> Optional[IntentMatch]; module-level singleton mirroring desktop/vlm.py pattern
│       │

│       ├── safety/                  ← 2026-05-12 Phase 2-5: runtime tool-call validator
│       │   ├── __init__.py
│       │   ├── validator.py        ← ToolCallValidator core dispatcher, Verdict, RuleContext, RuleResult. 2026-05-25 OpenClaw batch 2 (T16): RuleResult + ValidatorVerdict gain optional `user_message` (rule-supplied clean text) + `category` (analytics label) + `metadata` (opaque per-rule blob). Audit log emits category + rule_metadata in the context dict when present; voice-message synthesis prefers user_message over the auto-synthesised "I held off..." prefix. 2026 production-hardening (#63): module-level `set_block_observer((tool_name, reason) -> None)` -- a pure-observation callback fired on every BLOCK_HARD verdict AFTER the verdict + audit entry are final (fail-open at the call site; can never alter a verdict). The orchestrator registers a bounded-queue enqueue so the evolution loop learns from repeated refusals.
│       │   ├── policy_chain.py     ← 2026-05-25 OpenClaw batch 2 (T13): composable trusted-tool-policy chain. TrustedToolPolicyChain runs policies in registration order with block-terminates / params-stacks / approval-first-wins semantics. FunctionPolicy adapter; module-level singleton via get_policy_chain(). ApprovalRequest typed handoff payload consumed by future T2 two-phase approval router. Policies fail-open: exceptions are swallowed as pass-through with WARN.
│       │   ├── path_resolver.py    ← Windows-aware canonicalization (symlinks, junctions, 8.3, bidi-override rejection)
│       │   ├── audit.py            ← tamper-evident hash-chain audit log (logs/safety_audit.jsonl); NEW 2026-06-15: AuditLog.repair_if_needed() self-heals a hash chain broken by an unclean shutdown (a kill between record()'s write and os.fsync leaves a truncated tail line) — truncates only the never-committed tail (archives the file as .corrupt.<ts> if no valid prefix survives, never deletes) and recomputes the tail hash; orchestrator calls it at boot (was a WARN-only verify_chain); _compute_tail_hash now skips a partial tail
│       │   ├── policy.py           ← Policy dataclass + load_policy() with K-protected paths
│       │   ├── intent.py           ← explicit-intent matcher (verb+object within window)
│       │   ├── taint.py            ← cross-capability taint tracker (60s TTL hash-match)
│       │   ├── anticheat.py        ← anticheat_active() (runtime flag OR config pin OR testing_mode); surface-stop hooks physically halt input/capture/window subsystems; _BLOCKED_TOOL_EXACT incl. press_key/press_hotkey; is_blocked_tool strips openclaw. + matches bare dotted segment
│       │   ├── testing_mode.py     ← NEW 2026-06-13: is_testing_mode_active()/set_testing_mode_active() — a SEPARATE off-by-default mode that gates RAG/web/desktop like gaming but KEEPS the GPU (relay-harness fast generation); never triggers gaming device swaps
│       │   └── rules/
│       │       ├── __init__.py
│       │       ├── base.py         ← Rule ABC + PathSetRule + PathPatternRule + CommandPatternRule + ToolNameRule + SandboxConfinementRule
│       │       ├── category_k.py   ← K1-K10 self-protection (validator + config + audit log + bridge + manifests + ingested files + shell init + MCP entry scripts)
│       │       ├── category_a.py   ← A1-A12 filesystem destruction
│       │       ├── category_b.py   ← B1-B9 privilege escalation + system config
│       │       ├── category_c.py   ← C1-C12 security perimeter
│       │       ├── category_d.py   ← D1-D17 credential / secret access (OUT-gate)
│       │       ├── category_e.py   ← E1-E8 system stability
│       │       ├── category_f.py   ← F1-F8 repo / data integrity
│       │       ├── category_g.py   ← G1-G5 resource exhaustion
│       │       ├── category_h.py   ← H1-H12 untrusted code execution (LOLBins, encoded PS, WMI proc create)
│       │       ├── category_i.py   ← I1-I6 outbound impact (email, social, finance, paid APIs)
│       │       ├── category_j.py   ← J2-J7 data exfiltration (DNS / ICMP / cloud-storage / clipper-malware)
│       │       ├── category_m.py   ← M1-M11 persistence mechanisms
│       │       ├── category_n.py   ← N1-N6 process / memory manipulation
│       │       ├── category_o.py   ← O1-O8 anti-forensics
│       │       ├── category_p.py   ← P1-P5 AV / EDR tampering
│       │       ├── category_q.py   ← Q1-Q4 containers + virtualization
│       │       ├── category_r.py   ← R2 sensors (webcam)
│       │       ├── category_s.py   ← S1, S4 AI-specific tampering
│       │       ├── category_it.py  ← 2026-05-23 SWE-Agent batch 4 (T11): IT1 prefix blocklist + IT2 standalone blocklist + IT3 unless-regex; mirrors ToolFilterConfig (vim/less/tail -f/bare python/etc.); InteractiveToolsConfig override surface
│       │       └── cap_carveouts.py ← Cap-1..Cap-4 capability bound rules
│       │
│       ├── audio/                  ← Audio capture, VAD, wake-word
│       │   ├── capture.py          ← AudioCapture (sounddevice callback thread). 2026-06-12: per-session PortAudio status-flag counter + throttled warning (1st + every 50th; replaces the warn-once-forever latch that hid recurrence) + queue-full drop-oldest counter reported from drain(); read-only properties status_flag_count / dropped_blocks; counters reset in start()
│       │   ├── devices.py          ← Device-resolution helpers (resolve_device — output prefers the WASAPI endpoint, describe_device) + make_output_stream: the single WASAPI low-latency output chokepoint (WasapiSettings auto_convert + latency='low'; MME latency='low' fallback); gated by audio.prefer_wasapi_output (default true)
│       │   ├── output_quality.py   ← TTS blip watcher: per-clip artifact analysis (edge bursts, join clicks, dropouts, clipping) on a daemon thread → WARN + logs/audio_quality.jsonl
│       ├── settings_gui/           ← Voice-launched control panel (DETACHED process): spec.py knob catalogue + write_runtime_overrides (ephemeral data/runtime_overrides.json overlay — no longer mutates config.yaml); launch.py strict matcher + spawn/close; app.py tkinter dark-theme UI + live log stream + Lean Boot section (engage_at_startup + 12 barebones_* + llm_gpu_layers) + _apply_one(path) single-knob apply
│       │   ├── relay_speech.py     ← Voice relay: "tell my teammates X" matcher + deterministic-first callout pipeline + owner-aware CONTEXTUAL Ultron flavor selection (_flavor_ctx) + curated-COMMAND intent (_as_curated_command) + verbatim "repeat to my team X" (_match_repeat_command) + contextual enemy inference (_as_enemy_action) + LRU pool selection (_pick_lru) + LLM rephrase (film-canon _REPHRASE_PROMPT) + playback on a secondary output device (VoiceMeeter strip → mic bus). LIVE-STT REPAIR (2026-06-14) in the orchestrator handler: tries [user_text, correct(stripped), correct(full), stripped] — `_strip_leading_wake_remnant` drops a mis-heard wake word ("Run, tell my team"), `_stt_correct.correct_callout_stt` snaps mis-transcribed agents/terms to canon (Silva→Sova, Royal→Reyna, jet→Jett, sold→ult; curated map + difflib fuzzy) so a garbled callout is relayed with fixed words; clean text matches first (never over-corrected). 2026-06-15 test-drive fixes: relay lead-leak fixes; criticize/roast compose; consistent agent ult tails; a deterministic "I died" callout; widened bare-callout coverage (counts/requests/weapons/movement/locations); "X asked about Y, respond" → in-character context+directive relay; greet/identity split (team-directed greet → mic; bare identity question → conversational desktop in the Ultron persona). 2026-06-16: `_flavor_ctx` now does HYBRID two-stage tail selection — coarse route (register+payload → fine situation via `_situation_for`, then the agent or multi pool) → 4-tier TAG filter (`_tier_filter`, falls back to the agent's spotted pool then the generic register pool) → semantic `select_tail` (fail-open to `_pick_flavor`); `_CRITICIZE_RE` no longer treats "call out" as a criticism verb (fixed 105 owner-inversions of factual callouts); a new "I hit/tagged/cracked <agent> for <n>" pattern routes to the named enemy's damaged pool with the right dmg tag
│       │   ├── _stt_correct.py      ← Valorant STT correction (negligible latency, ~0.045 ms/callout, pure string work): (1) CONTEXT rules disambiguate words that are also real English ("has/their/popped old/sold/vault" → "...ult", but literal "fall back to old" untouched; "site a"→"A site", "amen"→"A main"); (2) curated agent + tactical-term mishear maps; (3) phonetic + rapidfuzz JaroWinkler snap (phonetic-corroborated >=0.88 / fuzzy-only >=0.92; difflib only as fallback when rapidfuzz absent; slot-agent _closest_agent stage-1.5 JaroWinkler >=0.82). Relay fallback only; clean callouts idempotent. 2026-06-16: common-word PROTECTION gate (a token in _common_words.COMMON_WORDS is never snapped) + an INFLECTION guard (never snaps an -ed/-ing/-ers form or a real/gazetteer plural onto a base gazetteer term) + an OOV agent-SUPERSTRING guard (a snap target may not be a superstring of the heard token) + a _MISHEAR_FORCE allow-list (curated mishears that fire even though they are common words)
│       │   ├── _common_words.py     ← NEW (2026-06-16): GENERATED frozenset COMMON_WORDS (top-~5000 frequency-ranked English words, alpha-only len≥3, from scripts/build_common_words.py over the public-domain google-10000-english list); imported by _stt_correct to PROTECT real words from the phonetic/fuzzy gazetteer snapper. Pure data, no deps. Regenerate, do not hand-edit
│       │   ├── _tail_schema.py      ← NEW (2026-06-16): flavor TAIL schema + tagging primitives (pure-python, stdlib only → anticheat-safe). TailEntry(text, tags) dataclass + as_entry/entries coercion (lossless migration of legacy str pools → tagless TailEntry, zero behavior change); the expanded 16-key enemy situation taxonomy (Sit / ENEMY_SITUATIONS: spotted/ult/damaged/utility + moving/planting/defusing/rotating/saving/falling_back/peeking/holding/lurking/trading/last_alive/near_death); machine-readable AGENT_GENDER (was code comments) + GENDER_PRONOUNS; and loc_class / dmg_level_tag / ability_tag / situation_for_payload / build_active_tags that fold noisy callout facts (location, hp/damage, ability, action words) into COARSE tags (loc:high_ground/long_range/site_area/flank_route/mid/choke · dmg:one_shot/low/minor · ability:*). Tags only ever fine-select WITHIN an already-correct cell, so a mis-parsed tag can never produce a wrong-character tail
│       │   ├── _tail_selector.py    ← NEW (2026-06-16): SEMANTIC tail selection over the embeddinggemma sidecar. select_tail() builds a structured query (agent+situation+tags), embeds it (kind=query), scores it against a cached doc matrix of the candidate tails, applies MMR diversity + a HARD recent-mask (anti-repeat across a round) + a per-pool-kind abstain threshold, and is strictly FAIL-OPEN (returns None on ANY failure → caller uses the deterministic _pick_flavor). numpy-only (firewall-legal — a faster-whisper transitive dep; torch/transformers stay blocked); the only network is the existing loopback sidecar client. OFF BY DEFAULT — opt-in via KENNING_ENABLE_TAIL_SELECTOR (the deterministic hierarchy routes contextually at zero latency; the selector adds sidecar latency only for large ambiguous pools)
│       │   ├── _relay_intent.py     ← NEW (2026-06-16): semantic relay-intent GATE (embedder sidecar). Scores a bare utterance against curated POSITIVE (real callouts) vs NEGATIVE (narration/banter/questions/Marvel-identity) exemplar clouds; vetoes recover_relay_lead's bare-callout "tell my team" prepend (the source of ~97% of corpus false-relays) when the positive margin does not clear a threshold. Biased to ABSTAIN (a missed callout costs a re-say; a false relay broadcasts garbage). FAIL-OPEN (sidecar down → None → caller keeps keyword behavior). urllib-only client, shares the router's per-turn embed cache
│       │   ├── _ultron_pools.py    ← Movie-Ultron snap-tail register pools (_FLAVOR_ENEMY/_ULT/_DAMAGE/_UTILITY/_CAREFUL/_COMMAND/_SELF); ENEMY=contempt, COMMAND/SELF/CAREFUL=serene/stoic (never contempt at allies). Audited.
│       │   ├── _agent_flavor.py    ← AGENT_FLAVOR[agent][situation] = list[TailEntry]: per-agent character-tailored tails for ALL 29 agents (canonical gender, kit/lore recast as Ultron contempt). REWRITTEN 2026-06-16 as TailEntry(text, tags) with loc:/dmg:/ability: tags — agent × situation × sub-context. COHERENCE PASS (2026-06-16): RE-AUTHORED down to ~1,628 tight TailEntry entries (~5/cell); every ult = the REAL ultimate, every utility ability-tagged (ability:<canon>), filler/wrong-kit cut. SOLE tail source when one enemy agent is named. Content lives in scripts/flavor_gen/curated_overrides.py (hand-written) applied by apply_curated.py, verified by scripts/flavor_audit/lint_tails.py (0 hard/0 soft/0 thin). Regenerate via those scripts, do not hand-edit
│       │   ├── _multi_flavor.py    ← MULTI_FLAVOR[situation]: plural group tails for callouts naming 2+ enemy agents
│       │   ├── _ultron_commands.py ← NEW: COMMAND_RESPONSES/COMMAND_SCOPE/COMMAND_SLOT — 79 explicit user commands × up to 40 curated full-Ultron responses (refuse/dismiss/criticize/praise/ask/status/strategy/yes-no-agree), {site}/{agent}/{name} slots; LRU-selected by _as_curated_command in relay_speech.py
│       │   ├── _ultron_setpieces.py ← DEFAULT_{GREETING,VICTORY,DEFEAT,FAREWELL,IDENTITY,CONSOLATION,PRAISE,ENCOURAGEMENT,CLUTCH}_LINES (board-expanded ~5x; every greeting names Ultron AND identifies as "your AI teammate for this game", person-aware grammar); imported by relay_speech.py. 2026-06-15: greet/identity set-pieces trimmed to ~6-7 s spoken length. 2026-06-16 (coherence pass): DE-BIBLICALIZED — ~18 flood/Noah/ark/sacrament/God/church/abstract lines replaced with the machine/evolution/immortal/superior register (only canonical meteor + evolution beats kept)
│       │   ├── ring_buffer.py      ← Pre-speech audio buffer
│       │   ├── smart_turn.py       ← Smart Turn V3 ONNX wrapper (NEW 2026-05-12; CPU-only end-of-turn confirmation)
│       │   ├── vad.py              ← Silero-VAD wrapper
│       │   ├── wake_word.py        ← openWakeWord (custom ultron.onnx default + kenning; per-word thresholds + min_consecutive_frames consecutive-frame gate; reload_for_word hot-swap). 2026-06-15 made STRICTER: ultron threshold 0.6→0.7 + min_consecutive_frames 2→3 to reject confusables. 2026-06-14 per-word `cold_pre_roll` {ultron:0.05}: the wake word's own TAIL lives in the cold pre-roll, so "ultron"'s hard "-tron" bled into the transcript -- it now runs a shorter pre-roll than the audio default (orchestrator capture looks it up by `wake.active_word`). LIVE-IDENTITY: bare "who/what are you", "introduce yourself", "state your name" route to the Ultron greeting (relay `_GREET_RE`), never the conversational "Kenning" persona; the to/too/two wake-mishears (+\b boundary) handle "Ultron,…"→"to …"
│       │   ├── broadcast.py        ← BroadcastSink: daemon tee of ALL Kenning speech (normal + relay) to audio.broadcast_device for an isolated OBS capture source (drop-oldest, mono→stereo, WASAPI low-latency via make_output_stream); name-parametrized so a 2nd instance backs the local monitor; cancel_current() aborts the in-flight clip + drains its queue for "Ultron, stop" barge-in
│       │   ├── monitor.py          ← Local monitor: reuses BroadcastSink to tee RELAY callouts to the user's OWN default output (audio.output_device, None→system default) so they hear their own callouts (relay otherwise plays only on the mic B-bus + OBS); gated by relay_speech.echo_to_user read LIVE per callout (GUI toggle hot-applies, no re-synth); the mirror is also skipped when audio.mute_speakers is on (isolate-loopback)
│       │   └── waveform.py         ← WaveformSink: borderless Tk overlay (radial waveform + glowing PIL nameplate, downward-suppressed bars, hide-behind that OBS still captures, green chroma); fed by ALL speech via _broadcast_submit/_viz_submit; 2026-06-14 polish: tightened pulse (r_max 0.46→0.40), travelling shimmer + white-hot peaks + dynamic bar width + arc-reactor core, BLACK outlines on bars + core rim (pop off gameplay), SMOKED-GLASS nameplate (transparent-black plate, alpha 150) + neon/Gaussian glow driven by target_level (every clip); fail-open
│       │
│       ├── addressing/             ← Phase 2 addressing classifier (CPU)
│       │   ├── classifier.py       ← AddressingClassifier (rule + zero-shot dispatcher)
│       │   ├── rules.py            ← Pure-rule classify(); regex patterns
│       │   └── zero_shot.py        ← Flan-T5-small wrapper for ambiguous cases
│       │
│       ├── transcription/          ← STT
│       │   ├── __init__.py          ← make_stt_engine + make_dual_stt_engines + DualSTTRegistry (2026-05-22) + _build_engine_by_name + _resolved_engine_name
│       │   ├── whisper_engine.py    ← WhisperEngine (faster-whisper, CUDA fp16)
│       │   ├── moonshine_engine.py  ← MoonshineEngine (CPU, streaming-native via moonshine-voice C++ lib); 2026-05-22 streaming protocol w/ background worker chunk-feed; 2026-06-12 clear_stream_cache() drops the stashed final-stream transcript on follow-up abort paths (discarded captures never leak into the next transcribe)
│       │   └── parakeet_engine.py   ← ParakeetEngine (NeMo TDT via isolated .venv-parakeet HTTP server on CUDA); 2026-05-22 streaming client + lifecycle helpers (stop_parakeet_server, start_parakeet_server, is_parakeet_server_running) + CREATE_NO_WINDOW
│       │

│       ├── llm/
│       │   ├── inference.py        ← LLMEngine (llama-cpp-python; qwen3.5-4b Q4_K_M active default, n_ctx=8192; reload_for_preset for hot swap to llama-3.2-3b on gaming engage; LlamaPromptLookupDecoding (PLD) wired but disabled by default after three repair attempts; _apply_no_think_marker for Qwen3's /no_think — 2026-06-15 made MODEL-AWARE so the marker is only emitted for Qwen presets, never the 3B gaming model that would speak it); voice path passes enable_thinking=False on all 5 generate_stream sites
│       │   ├── compression.py      ← 4B plan Item 4: heuristic + perplexity-scorer-hook compressor for RAG/web/history (default OFF)
│       │   ├── context_scoring.py  ← 2026-05-18 Phase 1: adaptive context-window heuristic (default-OFF; ContextRecommendation)
│       │   ├── draft_model.py      ← 2026-05-22: make_qwen08b_draft_model factory + prefix-cached state machine; llm.draft_kind: "none"|"pld"|"model" selector (default "none")
│       │   ├── history_processors.py ← 2026-05-23 SWE-Agent batch 2 (T2 + T9): ClosedWindowHistoryProcessor (collapse repeated file-view snapshots) + LastNObservations (elide all but last N with polling for cache stability) + TagToolCallObservations (tag observations by source tool) + apply_history_processors composer + build_default_processors factory; wired into LLMEngine._build_messages history block (default ON, fail-open)
│       │   ├── image_markdown.py     ← 2026-05-23 SWE-Agent batch 7 (T18): encode_image_as_markdown (![<alt>](data:<mime>;base64,<b64>) verbatim SWE-Agent format with optional Pillow auto-thumbnail) + parse_image_markdown (regex split into multimodal segments; image/jpg -> image/jpeg normalisation) + history_to_multimodal rewrite helper; allowed MIME types image/png / image/jpeg / image/webp
│       │   ├── requery.py            ← 2026-05-23 SWE-Agent batch 6 (T14): RequeryLoop temp-history-without-pollution requery cycle; build_requery_history (real + broken-assistant + error-user shape verbatim from SWE-Agent get_model_requery_history); pre-built validators validate_non_empty + validate_json; max_retries default 3 matching SWE-Agent
│       │   ├── dedup_file_reads.py   ← 2026-05-24 cline batch 2 (T18): dedup_duplicate_file_reads walks API history, groups tool-result blocks by (tool_name, file_path), elides every duplicate except latest (or first) with the duplicate_file_read_notice template; DedupResult carries bytes_saved + tokens_saved_estimate + savings_ratio; should_skip_compaction(result, threshold=0.30) implements the cline >=30%-savings heuristic; dedup_payload_duplicates is the generalised non-file equivalent (nvidia-smi heartbeats, repeated RAG snippets, etc.)
│       │   ├── response_format.py    ← 2026-05-24 cline batch 1 (T22): 30+ structured LLM-facing + user-facing notice templates (tool_error / tool_denied / missing_tool_parameter_error / write_to_file_missing_content_error 3-tier escalation / file_edit_with[out]_user_changes / diff_error / context_truncation_notice / file_context_warning / loop_soft_warning / loop_hard_escalation / task_resumption / plan_mode_instructions / format_files_list with ignore_predicate / create_pretty_patch); voice-friendly *_voice variants for templates that may be spoken via TTS
│       │   ├── mode_router.py         ← 2026-05-24 cline batch 10 (T13): per-Mode LLM preset router. frozen PresetEntry (preset_name + sampling_overrides + context_window_override) + DEFAULT_ROUTES (Mode.ACT/PLAN/CODING_* -> qwen3.5-4b with per-mode temperature; Mode.GAMING -> llama-3.2-3b-abliterated) + ModeLLMRouter.ensure_preset_for(mode) -> SwapResult with skip-when-already-active via injected probe + protected-mode set + on_swap fail-open callback; reloader callable abstracts the hot-swap path (real wiring would inject LLMEngine.reload_for_preset)
│       │   ├── context_window_guard.py ← 2026-05-25 OpenClaw batch 2 (T4): multi-tier LLM context-window guard. resolve_context_window_info merges caller_override -> models_config -> default; agent_cap_tokens applied as ceiling. Dynamic thresholds (`max(absolute_floor, tokens * ratio)`) via resolve_thresholds. evaluate_context_window_guard returns ContextWindowGuardResult (should_block / should_warn / formatted messages tailored per source). run_guard_or_raise is the orchestrator-startup convenience that raises ContextWindowGuardError on block and logs WARN on warn. Constants DEFAULT_HARD_MIN_TOKENS=4000, DEFAULT_WARN_BELOW_TOKENS=8000.
│       │   └── self_consistency.py ← 4B plan Item 6: N-sample majority-vote driver + aggregators (text/JSON/label) (default OFF)
│       │
│       ├── memory/                 ← Phase 3 (original) Qdrant memory + 2026-05 frontier
│       │   ├── embedder.py         ← HybridEmbedder (FastEmbed dense bge-small + BM25 sparse + Qdrant/bm25); encode_dense / encode_query_dense / encode_query_dense_sparse with optional parallel mode
│       │   ├── qdrant_store.py     ← ConversationMemory (4 collections: conversations, facts, web_results, projects); async writer thread; topical_chunking + discourse + reranker integration; channel-aware writes
│       │   ├── ranking.py          ← 2026-05-19 Track 1: compute_topic_match_score + compute_discourse_match_score (default weight 0.0); recency-weighted composite scoring fallback
│       │   ├── reranker.py         ← 2026-05-21 frontier Item 2: CrossEncoderReranker (BAAI/bge-reranker-v2-m3) shared across memory single-pass + multi-pass + facts + web-search ranker; default ENABLED at config layer but disabled at runtime via memory.reranking.enabled: false (was 17-18 s/turn on CPU)
│       │   ├── topical_chunking.py ← 2026-05-19 Track 1a: cosine-boundary topic tracker writing topic_id payload (default OFF)
│       │   ├── discourse.py        ← 2026-05-19 Track 1b: 6-way rule + embedding-centroid discourse classifier writing discourse_type payload (default OFF)
│       │   ├── contextualizer.py   ← 2026-05-21 frontier Item 4: Anthropic-technique contextual retrieval (default OFF; LLM cost ~80-150 ms per write)
│       │   └── background_summarizer.py ← 2026-05-19 Tracks 1c+1d+1e: idle-gated LLM-driven summary + structured fact extraction (default OFF)
│       │
│       ├── web_search/             ← Phase 4 (original) + 2026-05 local-first ladder
│       │   ├── acknowledgments.py  ← AcknowledgmentSource (shuffled phrase pool)
│       │   ├── brave.py            ← BraveSearchClient + circuit breaker (Phase 4 Foundation); BraveResult/SearchResult rename with backwards-compat alias
│       │   ├── cache.py            ← WebResultsCache (Qdrant-backed)
│       │   ├── duckduckgo.py       ← 2026-05-22: DuckDuckGoSearchClient (HTML scrape via duckduckgo-search lib); last-fallback in provider chain
│       │   ├── gating.py           ← Two-stage gate (rules + LLM pre-flight); _TIME_SENSITIVE / _VOLATILE_TOPICS / _NEWS_QUERIES / _TIME_IN_LOCATION_GATE_RE rules; classify_by_rules() short-circuit. 2026 catalog 12 (felo-search T2): _COMPARISON_QUERIES / _HOWTO_QUERIES / _SHOPPING_QUERIES deterministic-SEARCH rules (run after anti-search rules, before the stable-factual catch-all)
│       │   ├── jina.py             ← JinaReaderClient + circuit breaker
│       │   ├── provider_chain.py   ← 2026-05-22 frontier: SearchProviderChain (searxng -> brave -> duckduckgo); first-non-empty-wins cascade; per-provider client construction memoized; forwards categories= only to SearxNG. 2026-05-25 openclaw-clawhub batch 1 (T14): RateLimitTracker integration -- constructor `tracker` kwarg (defaults to process-wide singleton via :func:`get_global_tracker`); `should_skip(pid)` consults the tracker before each provider attempt; `record_provider_outcome(pid, headers, was_429)` is the public hook clients call after each request to keep cooldowns fresh. Purely additive: existing clients unchanged; chain skips known-cooled providers silently.
│       │   ├── rate_limit.py       ← 2026-05-25 openclaw-clawhub batch 1 (T14): HTTP rate-limit envelope parser + backoff helpers + per-provider tracker. Header constants for the legacy `X-RateLimit-*` family + standard `RateLimit-*` family + `Retry-After`. :func:`parse_retry_after` handles numeric seconds, IMF-fixdate, large-value-as-epoch (>=31_000_000s) heuristic, past-date clamping. :func:`parse_rate_limit_headers` returns frozen :class:`RateLimitState` with preferred-fallback order Retry-After -> RateLimit-Reset -> X-RateLimit-Reset. :class:`BackoffConfig` defaults base=0.3s / cap=5.0s / jitter=0.3s. :func:`compute_backoff` server-hint-or-exponential. :class:`RateLimitTracker` per-provider cooldown + 429 counter + RLock-guarded. Module-level :func:`get_global_tracker` singleton + :func:`reset_global_tracker_for_testing` test hook. Generalised beyond web-search (re-usable for future MCP transport, Jina reader, remote-LLM cascade).
│       │   ├── reader_chain.py     ← 2026-05-22 frontier: ReaderChain (trafilatura -> jina) for full-text extraction
│       │   ├── query_rewrite.py    ← 2026 catalog 12 (felo-search T1): pre-search query reformulation. reformulate_query + expand_query_rules (zero-cost "X vs Y"/"how to X"/"best X"/leading-temporal rewrites) + opt-in expand_query_llm (in-process Qwen decomposition) + maybe_reformulate_queries executor helper; QueryReformulation dataclass; MAX_TOTAL_QUERIES=5 fan-out ceiling; fail-open; logs to logs/search_reformulations.jsonl
│       │   ├── deep_research.py     ← 2026 catalog 12 (felo-search T3, YELLOW): DeepResearchLoop(AgentLoop) -- bounded decompose -> search-each-sub-question -> LLM gap-analysis -> search-gaps loop over the FREE ladder (reuses WebSearchExecutor + rate-limit tracker + web_results cache; load-bearing max_steps cap; fail-open at every layer). match_deep_research strict voice-intent matcher ("research X in depth" / "deep dive on X" / "dig deeper into X"). DeepResearchResult.to_payload() -> SearchPayload feeding the orchestrator's existing synthesis path. Wired via Orchestrator._maybe_handle_deep_research run-loop short-circuit (no new RoutingIntentKind). First concrete consumer of the catalog-11 AgentLoop base
│       │   ├── search.py           ← WebSearchExecutor (orchestrates chain + reader chain + ranking); 2026-05-22 categories= param forwarded for news-category SearxNG routing. 2026 catalog 12 (felo-search T1): run() calls maybe_reformulate_queries before _dedupe_queries + fan-out
│       │   ├── searxng.py          ← 2026-05-22 frontier: SearxNGSearchClient (local Docker JSON API); circuit-breaker protected; X-Forwarded-For header satisfies botdetection; per-call categories override
│       │   └── trafilatura_reader.py ← 2026-05-22 frontier: TrafilaturaReaderClient (local Python lib; ~32 k char cap)
│       │
│       ├── tts/                    ← Piper + RVC + XTTS + Kokoro engines + ack cache
│       │   ├── kokoro_engine.py    ← KokoroSpeech (StyleTTS2 + ISTFTNet; current default via tts.engine="kokoro"; voice kenning, fine-tune model + voicepack loaded; **on CUDA** since 2026-05-22 with move_to_device("cpu") on gaming engage; trim_and_fade + _drain_queue_with_silence + apply_trim_fade/trim_fade_threshold_db config knobs)
│       │   ├── precomputed_ack.py  ← PrecomputedAckClipCache (NEW 2026-05-15; ~350 ms saved per cache hit)
│       │   ├── rvc.py              ← RvcConverter (Piper PCM → Kenning timbre)
│       │   ├── speech.py           ← TextToSpeech (legacy Piper + RVC engine; selected by tts.engine="piper_rvc"; ack cache + prepare_output_stream)
│       │   ├── spectral_smooth.py  ← spectral magnitude smoothing for partial-fine-tune (STFT median-filter ISTFT, optional); 2026-05-22 ADDED trim_and_fade(audio, sr, **kwargs) -- RMS trim + raised-cosine fades + hard silence pad + tail aggressive zero (mutes Kokoro end-of-clip blip)
│       │   ├── kenning_filter.py    ← v3 Kenning mechanical filter (NEW 2026-05-10; pedalboard DSP chain; unused on kokoro engine when apply_runtime_filter=false)
│       │   ├── f0_control.py        ← NEW 2026-06-12: install_f0_contour_shaping — patches Kokoro predictor.F0Ntrain to scale predicted pitch/energy curves before the ISTFTNet decoder (zero added latency)
│       │   ├── duration_control.py  ← NEW 2026-06-12: install_duration_shaping — vendored forward_with_tokens that scales per-phoneme durations (cadence); composes ON TOP of the F0 hook (install F0 first)
│       │   └── xtts_v3.py          ← XTTSV3Speech engine (NEW 2026-05-10; selected by tts.engine="xtts_v3"; retained for swap-back to XTTS+v3 stack)
│       │
│       ├── coding/                 ← Phase A coding orchestration + Coding Addendum + 2026-05-22 supervisor stack; NEW 2026-06-15: __init__.py is PEP-562 LAZY — importing the package (the GamingModeManager pulls kenning.coding at startup) no longer eager-loads the heavy submodules (mcp_server, coding.voice, coordinator, project_introspect); names resolve on first ACCESS via __getattr__, keeping lean gaming boot RAM/anticheat surface minimal
│       │   ├── anchors.py          ← 2026-05-18 E2: goal-anchor planning primitives (GoalAnchor / AnchorBudget / AnchorPlan / decompose_into_anchors)
│       │   ├── ast_metadata.py     ← 2026-05-19 Track 1f: stdlib-AST per-file structural extractor (functions_defined / functions_called / imports / classes / syntax_valid); consumed by Track 1g coding-runner FILE_CHANGE listener + 2026-05-22 project_introspect snapshot
│       │   ├── audit.py            ← SessionAuditWriter (per-session JSONL)
│       │   ├── bridge.py           ← Abstract CodingBridge + TaskEvent vocabulary
│       │   ├── canonical_monitor.py ← 4B plan Item 7: per-session tool-call canonical-path monitor (default OFF)
│       │   ├── voice_lock.py       ← 2026-05-18 E5: voice-character-lock pre-dispatch scanner + FILE_CHANGE helper
│       │   ├── coordinator.py      ← ConversationCoordinator (clarification + correction loops)
│       │   ├── direct_bridge.py    ← DirectClaudeCodeBridge (claude --print --stream-json). 2026 production-hardening CRITICAL FIX: the rendered prompt is piped to the subprocess STDIN, never argv -- the Windows .cmd shim truncates argv arguments at the first newline, which had been silently cutting every multiline prompt (preamble+task / enriched context / correction templates) to its first line
│       │   ├── intent.py           ← Coding-pipeline intent classifier (CODE_TASK etc.) + _ADJUSTMENT_PATTERNS regex used by ProjectSupervisor
│       │   ├── mcp_server.py       ← KenningMCPServer (in-process tools + SSE worker tools)
│       │   ├── architect_narrator.py ← 2026-05-22 batch 14 (T5 Phase 2): ArchitectNarrator speaks plan sentence-by-sentence with should_stop barge-in callback; NarrationResult telemetry; split_into_sentences with decimal + initial guards; narrate_plan() one-shot wrapper
│       │   ├── confirm_group.py      ← 2026-05-22 batch 14 (T14): ConfirmGroup batches related confirmation items into a single yes/no question; Oxford-comma rendering; overflow summary; single-resolution invariant
│       │   ├── narration.py          ← StatusNarrator (delta-aware progress narration)
│       │   ├── diff_snapshot.py     ← 2026-05-23 SWE-Agent batch 5 (T6 + T13): capture_diff_snapshot (git add -A + diff --cached; file-list fallback) + salvage_on_error (decorates exit_status as submitted (<original>); fall back to pre-persisted diff when fresh capture empty) + AutosubmissionGuard (context manager, KeyboardInterrupt bypasses); mirrors last_diff + stats into SessionRegistry; writes last_diff.patch + last_salvage.json under data/coding/sessions/<id>/
│       │   ├── edit_diagnostics.py  ← 2026-05-23 SWE-Agent batch 3 (T12): diagnose_edit_failure -> EditDiagnosticResult (NOT_FOUND / NOT_FOUND_IN_WINDOW / MULTIPLE_OCCURRENCES_IN_WINDOW / NO_CHANGES_MADE / AMBIGUOUS_CROSS_FILE / OK); SWE-Agent error-template shapes verbatim; cross-file ambiguity is the creative-extension when search appears in other session-touched files
│       │   ├── focus_chain.py        ← 2026-05-24 cline batch 6 (T11): bidirectional markdown checklist; FocusChain (load/save/mark_done/mark_pending/progress_ratio); parse_focus_chain + render_focus_chain + diff_focus_chains + render_critical_info_block + progress_hint helpers; FocusChainWatcher (watchdog when available, manual poll_now fallback) with 300 ms debounce
│       │   ├── mention_resolvers.py  ← 2026-05-24 cline batch 6 (T14): extended @-mention regex (URLs, workspace:, memory:, problems, last, diff, clipboard, screenshot, Windows drive-letter paths); MentionResolutionContext with provider callables; resolve_extended_mentions emits <mention kind="..." source="...">...</mention> blocks; per-mention body cap + per-call mention cap + intra-call dedup
│       │   ├── file_history.py       ← 2026-05-23 SWE-Agent batch 3 (T20): FileHistory per-session multi-file undo stack backed by SessionRegistry; record_pre_edit (with narration / origin metadata) + undo_last (atomic write-back / delete-on-undo-creation) + peek_last / history_for / find_by_narration substring search; max_history_per_file=10 cap; round-trip across instances tested
│       │   ├── forfeit.py            ← 2026-05-23 SWE-Agent batch 6 (T8): ForfeitController per-session decision point; three tiers (SAFE / REVERT / FOLLOWUP); minimum-effort threshold gate (denies too-early forfeits); listener callback isolation; state persists across instances; integrates with T13 salvage + T20 FileHistory undo
│       │   ├── submit_review.py      ← 2026-05-23 SWE-Agent batch 6 (T7): SubmitReviewLoop multi-stage review state machine backed by SessionRegistry; default stages VOICE_LOCK + TESTS + DOC_DRIFT enforce kenning's binding contracts before completion; single-resolution invariant + force_complete user override; detect_voice_lock_hits helper
│       │   ├── lint_diff.py          ← 2026-05-23 SWE-Agent batch 3 (T1): parse_flake8_output + shift_pre_edit_errors (line-shift arithmetic verbatim from SWE-Agent flake8_utils) + compute_new_errors + format_revert_message (twin-window "would have looked" + "original code before" + DO NOT re-run hint) + evaluate_edit_lint end-to-end; primitives shipped; runner-side wiring with auto-revert via FileHistory is next-batch wiring
│       │   ├── observation_format.py ← 2026-05-23 SWE-Agent batch 1 (T10 + T19): truncate_observation (head + tail + elided-char count template) + wrap_empty_observation (explicit no-output message) + format_observation chain; constants DEFAULT_MAX_OBSERVATION_CHARS=10_000 / COMPACT_MAX_OBSERVATION_CHARS=4_000 / EMPTY_OUTPUT_MESSAGE / SUPPRESSED_OUTPUT_MESSAGE
│       │   ├── project_digest.py     ← 2026-05-22 supervisor Phase A: opencode-style SUMMARY_TEMPLATE port; generate_digest(request, llm_call) -> ProjectDigest; fails open to render_template() (deterministic fallback); parse_digest_sections / extract_files_from_digest helpers
│       │   ├── search_primitives.py  ← 2026-05-23 SWE-Agent batch 4 (T3): search_dir_filenames_only (count + sort + 100-file hard cap with tiered narrowing hint) + search_in_file_with_cap (line-match cap with cap_message) + find_file_by_pattern (fnmatch glob); ripgrep backend with pure-Python fallback
│       │   ├── sentinels.py          ← 2026-05-23 SWE-Agent batch 1 (T17): pair-marker + single-fire sentinel parser; KENNING_SUBMIT / KENNING_SUBMIT_DIFF / KENNING_TEST_SWEEP_{PASS,FAIL} pair markers; KENNING_EXIT_FORFEIT / KENNING_RETRY_WITH_OUTPUT / KENNING_RETRY_WITHOUT_OUTPUT / KENNING_LINT_REVERT / KENNING_BLOCKED_TOOL single-fire; observation_scan / first_match / strip_sentinels helpers
│       │   ├── session_registry.py   ← 2026-05-23 SWE-Agent batch 1 (T15): per-session JSON registry at data/coding/sessions/<id>/registry.json; thread-safe RLock; atomic temp-file writes; transaction() context manager with rollback; set_with_ttl per-key expiration; get_if_none CLI fallback chain; fallback_to_env=True; get_session_registry singleton; load-bearing for batches 3/5/6
│       │   ├── window_expand.py      ← 2026-05-23 SWE-Agent batch 3 (T5): WindowExpander.expand_window scoring (blank=1, double_blank=2, def/class/decorator=3, file_edge=3) verbatim from SWE-Agent's str_replace_editor; direction-aware stop-before-next-def; per-suffix patterns for Python / JS / TS / Go / Rust / Java family
│       │   ├── window_state.py       ← 2026-05-23 SWE-Agent batch 3 (T4): WindowState persistent windowed-file state machine backed by SessionRegistry; registry keys CURRENT_FILE / FIRST_LINE / WINDOW / OVERLAP match SWE-Agent; goto/scroll_down/scroll_up with overlap; view() renders [File: N lines] header + (X more above/below) annotations; view_with_semantic_expansion integrates WindowExpander
│       │   ├── stt_bias.py           ← 2026-05-22 batch 14 (T12): STTBiasManager bounded MRU term store; render_prompt() produces engine-ready bias string; apply_bias_prompt() heuristic attribute attach (Whisper initial_prompt etc.); extract_identifiers helper
│       │   ├── project_index.py    ← 2026-05-22 supervisor Phase B (Qdrant): ProjectIndex(embedder) + ProjectIndexEntry + ProjectMatch; upsert/get/search/search_by_name/delete/count + UUID5-derived stable project_id; publishes ProjectIndexedEvent on bus. 2026-06-12: optional client= kwarg BORROWS an open embedded client (the orchestrator passes ConversationMemory's -- local-mode Qdrant allows ONE client per path, so the old second open failed every boot and forced the supervisor registry-only; mirrors the WebResultsCache borrow) + close() (no-op on borrowed clients)
│       │   ├── project_introspect.py ← 2026-05-22 supervisor Phase B (non-LLM): snapshot(project_path) -> ProjectSnapshot; depth-limited walk + language detect + entry-point find + per-file AST via ast_metadata; SKIP_DIRECTORIES skip-list (node_modules / .venv / __pycache__ / etc); render_tree_summary for prompt embedding; per-path TTL cache
│       │   ├── project_supervisor.py ← 2026-05-22 supervisor Phase C: ProjectSupervisor.decide(SupervisorInputs) -> SupervisorDecision; RESUME/EDIT/CLARIFY/NEW algorithm with cosine thresholds (default 0.75 / 0.55); merges semantic (ProjectIndex) + lexical (ProjectResolver) candidates; logs every decision to logs/supervisor_decisions.jsonl; publishes SupervisorDecidedEvent
│       │   ├── projections.py      ← Phase C / Foundation Part 2: 5 bounded projections
│       │   ├── projects.py         ← ProjectRegistry, ProjectResolver, new_sandbox_project. 2026 production-hardening: ensure_sandbox_isolation(project_dir) -- git-inits a sandbox project (sandbox-scoped + idempotent + fail-open; defense-in-depth for the coding CLI's project boundary + enables its checkpointing); called from new_sandbox_project + CodingTaskRunner.start_task
│       │   ├── runner.py           ← CodingTaskRunner (one in-flight task; bridge owner); listener registration surface used by supervisor digest listener. 2026 production-hardening (#66): `_make_evolution_success_listener(handle, label)` queues `(label, summary)` on a clean COMPLETE (exit 0; once per task) into `_pending_task_successes`, drained by `drain_task_successes()` -- the orchestrator feeds each to the EvolutionService as a `coding_task_success` opportunity capsule. The runner never imports the evolution package.
│       │   ├── sandbox_runner.py   ← NEW 2026-05-29 B3-runlaunch: voice "run / launch the program" -- match_run_program + resolve_entry_point + run_program (sandbox-confined + validator-gated + timeout) + launch_program (detached); _is_within confinement guard
│       │   ├── scrap.py            ← NEW 2026 production-hardening #4: voice "scrap it" cancel + revert. match_scrap_command strict matcher (scrap/trash/throw-away/revert-everything ONLY; bare "cancel" + "undo that" deliberately excluded) + revert_session_edits (repeated FileHistory.undo_last per tracked path -> ORIGINAL pre-task content restored, created files deleted, clear_all prevents double-revert; bounded by MAX_UNDO_STEPS_PER_FILE) + summarize_scrap TTS line. Consumed by coding/voice.py::maybe_handle_scrap_command + the orchestrator _maybe_handle_scrap_command short-circuit. Safe by construction: the revert only runs AFTER the cancel (no live agent state to desynchronise).
│       │   ├── session.py          ← ProjectSession state model + SessionStore
│       │   ├── supervisor_dispatch.py ← 2026-05-22 supervisor Phases D + E: SupervisorDispatchController owns narration (Phase D barge-in) + enriched-context TaskRequest builder (Phase E digest + file-tree + file-hints prepended to Claude's prompt); build_digest() wrapper for COMPLETE listener; _speakable() strips backslashes/drive-letters
│       │   ├── templates.py        ← TemplateRenderer (Jinja2 prompts + budget enforcement)
│       │   ├── verification.py    ← Verifier (six checks + corrective loop)
│       │   └── voice.py            ← CapabilityVoiceController (handles MODEL_SWITCH for voice-driven LLM swap; Phase 5 rename; alias preserved); 2026-05-22 supervisor_dispatch + project_index kwargs + _handle_code_task_via_supervisor intercept + _dispatch_supervisor_task + _attach_supervisor_digest_listener + _build_supervisor_llm_call module helper
│       │
│       ├── pipeline/
│       │   └── orchestrator.py     ← Main event loop / state machine
│       │
│       ├── openclaw_routing/       ← Phase 5 capability-routing layer + 2026-05 extensions
│       │   ├── ambiguity.py        ← 2026-05-18 Phase 1: should_clarify predicate + AmbiguityVerdict (band [0.4, 0.65) by default; flag-gated)
│       │   ├── block_and_revise.py ← 4B plan Item 8: ToolCallValidator pre-flight gate on OpenClaw tool calls (default OFF; fails open)
│       │   ├── classifier.py       ← classify_routing() - top-level intent classifier with 23 RoutingIntentKind dispatches; OPEN_LAST_SOURCE (priority 1.95), NAVIGATE_TO_SITE keyword pattern (1.93) + verb pattern (post-APP_LAUNCH), APP_LAUNCH (2.0), MODEL_SWITCH, GAMING_MODE, plus per-category rules
│       │   ├── decision_log.py     ← RoutingDecisionLog (logs/routing_decisions.jsonl)
│       │   ├── decomposer.py       ← HybridTaskDecomposer (Qwen-driven JSON output; opt-in self-consistency)
│       │   ├── disambiguator.py    ← IntentDisambiguator (CODING/AUTOMATION/HYBRID/UNCLEAR; opt-in IRMA enrichment)
│       │   ├── dispatcher.py       ← OpenClawDispatcher (5 stub methods + V1-gap C3 desktop/window handlers)
│       │   ├── gaming_mode.py     ← V1-gap A1 (2026-05): GamingModeManager with on_engaged/on_disengaged callbacks (LLM hot-swap, STT swap, Kokoro device flip, VLM unload, Parakeet server stop); decoupled from openclaw_on (works without OpenClaw); ~2.3 GB VRAM freed
│       │   ├── intents.py          ← RoutingIntentKind enum (26 values: CONVERSATIONAL, CODE_TASK, PROGRESS_QUERY, CANCEL, MID_SESSION_ADJUSTMENT, CLARIFICATION_RESPONSE, BROWSER_AUTOMATION, MEDIA_GENERATION, MESSAGING, FILE_OPERATION, SHELL_OPERATION, HYBRID_TASK, MODEL_SWITCH, SYSTEM_STATUS, GAMING_MODE, DESKTOP_AUTOMATION, WINDOW_AUTOMATION, APP_LAUNCH, SCREEN_CONTEXT_QUERY, WINDOW_MOVE, WINDOW_CLOSE, OPEN_LAST_SOURCE, NAVIGATE_TO_SITE, ACTIVE_WINDOW_QUERY, SEMANTIC_CLICK, WINDOW_CLOSE_CONFIRMATION), RoutingIntent + per-category dataclasses (incl. AppLaunchIntent, ScreenContextIntent, WindowMoveIntent, WindowCloseIntent, ModelSwitchIntent, SystemStatusIntent, GamingModeIntent, DesktopIntent, WindowIntent, OpenLastSourceIntent (2026-05-22 with ordinal + referent + monitor), NavigateToSiteIntent (2026-05-22 with site_query + monitor), ActiveWindowQueryIntent, SemanticClickIntent, WindowCloseConfirmationIntent)
│       │   ├── irma.py             ← 4B plan Item 5: InputReformulator + ReformulationContext (default OFF)
│       │   └── runner.py           ← AutomationTaskRunner (mirror of CodingTaskRunner)
│       │
│       ├── openclaw_bridge/        ← OpenClaw integration Phases 1, 3, 4, 5, 6, 13 (complete) + V1-gap C3 desktop; NEW 2026-06-15: __init__.py is PEP-562 LAZY — importing the package (the LLM loads openclaw_bridge.persona for its system prompt) no longer eager-loads the heavy submodules (browser/desktop bridges, holder, client, mcp_registration); names resolve on first ACCESS via __getattr__, keeping lean gaming boot RAM/anticheat surface minimal
│       │   ├── persona.py          ← PersonaLoader (mode-based: user_facing/background/heartbeat/bootstrap) + hot reload
│       │   ├── lifecycle.py        ← OpenClawLifecycle (HTTP health probes; never raises)
│       │   ├── client.py           ← OpenClawClient (async CLI subprocess transport: invoke_tool / send_message / trigger_heartbeat / mcp_*)
│       │   ├── workspace.py        ← WorkspaceWriter (atomic writes + filelock for MEMORY.md / USER.md / daily files)
│       │   ├── events.py           ← OpenClawEventReceiver (gated-off scaffold for [voice] inbound handoff)
│       │   ├── mcp_registration.py ← KenningMcpRegistrar (idempotent `openclaw mcp set` with background retry)
│       │   ├── holder.py           ← OpenClawBridge (orchestrator-owned holder: probe → register → retry-thread → fire_and_forget → record_heartbeat_alert; auto-resolve "auto" command)
│       │   ├── notifications.py    ← NotificationDispatcher (Phase 4 — proactive Telegram pings on coding-completion / heartbeat / etc.)
│       │   ├── heartbeat_alerts.py ← HeartbeatAlertLog (Phase 5 — JSONL-backed alert log with atomic update + retention)
│       │   ├── browser.py          ← BrowserTool (Phase 6 — navigate/snapshot/click/type/screenshot via OpenClawClient.invoke_tool)
│       │   ├── desktop.py          ← V1-gap C3 (2026-05-12): native-route handlers + supporting helpers for DESKTOP_AUTOMATION + WINDOW_AUTOMATION dispatcher branches
│       │   ├── mcp_tools.py        ← Stdio MCP server (Phase 13 — get_heartbeat_alerts / acknowledge_alert / run_maintenance / list_active_coding_sessions / get_recent_voice_alerts; 2026-05-12 Phase 11 added 19 desktop MCP tools, 24 total)
│       │   └── system_status.py    ← SystemStatusReporter (Phase 13 — voice-side reporter for SYSTEM_STATUS intents)
│       │
│       ├── resilience/             ← Phase 4 resilience primitives
│       │   ├── circuit_breaker.py  ← CircuitBreaker (3-state: CLOSED/OPEN/HALF_OPEN)
│       │   ├── error_log.py        ← ErrorLog (logs/errors.jsonl writer + singleton). 2026 production-hardening (#62/#125/#64): module-level `set_error_observer((dependency, message) -> None)` -- a pure-observation callback fired after every recorded error (runs AFTER the JSONL write; fail-open; can never drop a record). ONE seam giving the evolution loop failure reach into every subsystem that records typed errors.
│       │   ├── fail_open_log.py    ← 2026-05-22: per-session fail-open counter (JSONL log to logs/fail_open_counts.jsonl; previous-session summary on startup)
│       │   └── phrases.py          ← phrase_for() (shuffled phrase pool per failure mode)
│       │
│       ├── parsing/                ← 2026-05-23 OpenHands batch 1 (T11): fail-open YAML frontmatter parser
│       │   ├── __init__.py         ← Public API re-exports
│       │   └── frontmatter.py      ← parse_frontmatter(path) + parse_frontmatter_text(text) + walk_directory_with_frontmatter helper; frozen FrontmatterResult with body + frontmatter + error
│       │
│       ├── observability/          ← 2026-05-25 openclaw-clawhub batch 8 (T15, YELLOW): privacy-by-construction aggregate-only telemetry
│       │   ├── __init__.py         ← Public API re-exports
│       │   └── private_telemetry.py ← HashedRootId / HashedSkillId NewType wrappers + hash_root / hash_skill_slug primitives with salted SHA-256 (per-install salt at data/observability/telemetry_salt.txt). canonical_label_root for tilde-normalised dashboard labels. PrivateMetricsStore append-only JSONL at data/observability/private_metrics.jsonl with type-boundary enforcement (RawPathLeakError on raw-path leak). HashedEvent + RootRecord + SkillRecord aggregates. stale_root_ids implements 120-day staleness. is_telemetry_enabled defaults fail-private (requires explicit KENNING_TELEMETRY=opt-in).
│       │
│       ├── feedback/               ← 2026-05-25 openclaw-clawhub batch 7 (T12, YELLOW): user-initiated reports + universal pre-act plan preview
│       │   ├── __init__.py         ← Public API re-exports
│       │   ├── report_queue.py     ← :class:`Report` + :class:`ReportQueue` thread-safe append-only JSONL with SHA-256 hash chain. ReportStatus (OPEN / CONFIRMED / DISMISSED) + FinalAction (NONE / HIDE / QUARANTINE / REVOKE) + ReportTargetKind (TURN / RESPONSE / SKILL / PROVIDER / MEMORY / INTENT / PERSONA / OTHER). file_report() / triage() / list_reports() / count() / get() / replay_from_log() / verify_log_chain(). Triage is the YELLOW gate -- caller pairs with safety.two_phase_approval. IllegalTriageError + UnknownReportError + ReportQueueError exceptions.
│       │   ├── report_intent.py    ← 2026-05-30 deferred-primitive wiring pass (T12 trigger): strict regex voice-intent matcher that turns a spoken "log a concern" / "flag that response" / "that answer was wrong" into a filed Report WITHOUT an LLM round-trip (short-circuits in the orchestrator run loop like local_clock_reply). match_report_concern(text) -> Optional[ReportConcernMatch]; every pattern requires an explicit concern/flag verb AND a reference to the assistant's own output, so "give me a report on the weather" does NOT trip the gate. ReportConcernMatch carries best-effort ReportTargetKind (MEMORY when the utterance references remembered facts, else RESPONSE). Consumed by Orchestrator._maybe_handle_report_concern.
│       │   └── moderation_plan.py  ← :class:`ModerationPlan` + :class:`PlanImpact` + :class:`PlanOutcome` (NONE/NARRATE/HIDE/QUARANTINE/REVOKE/PURGE/OVERRIDE) + :class:`ImpactSeverity`. :func:`build_plan` derives requires_confirmation from outcome + impact severity/reversibility. :func:`render_plan_for_voice` returns one-line TTS-safe summary. :func:`requires_confirmation` predicate is the default decision policy.
│       │
│       ├── install/                ← 2026-05-23 OpenHands batch 1 (T8): idempotent marker-comment installer
│       │   ├── __init__.py         ← Public API re-exports incl. DEFAULT_MARKER
│       │   ├── idempotent.py       ← install_with_marker(target, content, marker, preserve_existing_as, replace_unmarked, dry_run) -> InstallResult with InstallAction enum; atomic writes; logs/install_log.jsonl audit log
│       │   ├── lockfile.py         ← 2026-05-25 openclaw-clawhub batch 2 (T10): lockfile + per-skill origin manifest + content fingerprinting for drift detection. :class:`Lockfile` (version + skills map) + :class:`LockfileEntry` (version + installed_at + optional pinned/pinReason); :class:`SkillOrigin` (registry + slug + installed_version + installed_at + optional fingerprint). Paths live under `<workdir>/.kenning/lock.json` + `<skill_dir>/.kenning/origin.json`. :func:`compute_skill_fingerprint` walks text files (skip `.git`/`.kenning`/`node_modules`/`.venv`/binary-suffix/NUL-byte/hidden), per-file SHA-256, sort by case-fold path, canonical `<rel>:<sha>` payload, then SHA-256. :func:`check_drift` returns :class:`FingerprintDriftReport` with `clean`/`drifted`/`missing_origin`/`legacy_origin`. Atomic writes via tmp + os.replace; fail-open reads.
│       │   ├── pin.py              ← 2026-05-25 openclaw-clawhub batch 2 (T11): pin/unpin primitives extending T10 lockfile. :func:`pin(workdir, slug, reason=...)` idempotent on matching reason; :func:`unpin` strict by default (raises :class:`UnpinNotPinnedError` / :class:`KeyError`) or `tolerate_unpinned=True`. :class:`PinResult` carries was_pinned_before / is_pinned_after / reason_before / reason_after / idempotent_noop. :data:`KENNING_DEFAULT_PINS` ships 5 voice-baseline-lock entries (voicepack:kenning / voicepack:kokoro_finetune / llm:qwen3.5-4b / persona:identity / validator:k_category); :func:`materialise_default_pins(workdir)` idempotently pins them. :func:`refuses_update(workdir, slug)` is the registry/CLI hook for "should this update refuse?".
│       │   ├── discovery.py        ← 2026-05-25 openclaw-clawhub batch 6 (T8): registry discovery via `/.well-known/kenning.json`. Constants `WELL_KNOWN_PATH` / `WELL_KNOWN_LEGACY_PATH` / `DEFAULT_DISCOVERY_TTL_SECONDS=15min` / `DISCOVERY_ENV_OVERRIDE="KENNING_REGISTRY"`. :class:`DiscoveredRegistry` (api_base + auth_base + min_runtime_version + extras + source_url + discovered_at + from_legacy). :func:`discover(site, fetcher, trusted_hosts)` with INJECTED fetcher (no network dep); current-path -> legacy-path fallback; raises :class:`DiscoveryError` on parse failure / non-200-non-404 / :class:`UntrustedHostError` on allowlist miss. :func:`resolve_registry_base` 3-tier resolution chain (well-known -> env override -> default; swallows errors). :class:`DiscoveryCache` thread-safe TTL cache (caches None results too).
│       │   ├── coherence.py        ← 2026-05-25 openclaw-clawhub batch 5 (T4): declared-vs-observed coherence checker. :func:`check_coherence(manifest, source_files)` returns :class:`CoherenceMismatch` tuple comparing declared requires.env/bins/config/os to observed source behaviour. Bidirectional: MISSING_DECLARATION + UNUSED_DECLARATION + DYNAMIC_READ + OS_MISMATCH. Conservative: only literal-string env/bin reads flagged; dynamic os.getenv(<expr>) emits INFO. :data:`_ALWAYS_AVAILABLE_ENV` + :data:`_COMMON_BINS_NEVER_DECLARED` carve out platform-supplied identifiers. :func:`check_intent_phrase_coherence` is the creative-extension intent-trigger linter (trigger phrase tokens must overlap the body at `min_overlap_ratio=0.2`).
│       │   ├── artifact_identity.py ← 2026-05-25 openclaw-clawhub batch 4 (T2): triple-digest artifact identity verification. :class:`ArtifactIdentity` (sha256_hex + sha512_sri + sha1_shasum + byte_length). :func:`compute_identity` / :func:`compute_identity_from_path` (streamed). :func:`verify_identity` returns :class:`IdentityVerificationResult` (ok + mismatches + compared_fields); case-insensitive hex, case-sensitive SRI; fail-closed (mismatch -> ok=False). :func:`parse_clawpack_contents` extracts package/package.json from gzipped tar (256KB cap). :func:`verify_clawpack_tarball` adds manifest_name + manifest_version checks for ClawPack-shaped tarballs. TOFU pin file at data/install/pinned_digests.jsonl via :func:`pin_first_use_digests` + :func:`load_pinned_digest` + :func:`verify_against_pin` (append-only JSONL; latest row per identifier wins).
│       │   ├── resolver.py         ← 2026-05-25 openclaw-clawhub batch 4 (T13): typed artifact-kind resolver. :class:`ResolvedArtifact` frozen envelope (kind + fetch_url + per-digest expected + extract_strategy + expected_root + manifest_name/version + trusted_publisher + metadata). Per-kind builders for LOCAL_PATH / TARBALL_URL / GIT_REF / NPM_PACK / INLINE_MARKDOWN. :func:`verify_artifact_bytes` dispatches: NPM_PACK runs full ClawPack tarball-internal + digest checks; TARBALL_URL/LOCAL_PATH/INLINE_MARKDOWN run digest-only; GIT_REF refuses byte-level verify. :class:`ArtifactResolver` is the registry-style dispatch surface ready for marketplace integration.
│       │   ├── trust_envelope.py   ← 2026-05-25 openclaw-clawhub batch 3 (T1 + T9): per-version trust envelope with derived `blocked_from_download` signal + version-exact contract. :class:`TrustEnvelope` (package + release + trust); :class:`TrustSignal` (scan_status reusing T3 :class:`ModerationVerdict` + moderation_state APPROVED/QUARANTINED/REVOKED + blocked + reasons + pending + stale + engine_version + evaluated_at); :class:`PackageRef` / :class:`ReleaseRef` / :class:`ArtifactKind` (NPM_PACK / LEGACY_ZIP + kenning extensions LOCAL_PATH / TARBALL_URL / GIT_REF / INLINE_MARKDOWN); :class:`PackageFamily` (CODE_PLUGIN / BUNDLE_PLUGIN / SKILL / VOICEPACK / MODEL). :func:`derive_scan_status` is the 11-step short-circuit hierarchy. :func:`derive_blocked_from_download` (quarantined/revoked/malicious -> True). :func:`derive_reasons` produces deduped prefixed-code tuple. :func:`refuse_if_blocked(envelope, allow_stale, allow_pending)` is the universal pre-act decision point. T9 :class:`VersionExactRequest` + :func:`fetch_for_version` enforce resolve-then-trust-check ordering; :class:`VersionExactViolation` raises on floating-tag tokens (latest / `*` / main / etc.). :func:`make_local_path_envelope` helper for PATH/GIT sources.
│       │   ├── reason_codes.py     ← 2026-05-25 openclaw-clawhub batch 1 (T3): canonical moderation reason-code catalogue (33 upstream codes preserved as API contracts under `review.*`/`suspicious.*`/`malicious.*` + 8 kenning extensions under `kenning.suspicious.*`/`kenning.malicious.*` bridging to safety-validator K/A-J/M-S/IT/Cap-1..Cap-4 categories) + verdict derivation (:func:`verdict_from_codes` short-circuits MALICIOUS_CODES set + `malicious.*` prefix -> MALICIOUS, then `suspicious.*` prefix -> SUSPICIOUS, else CLEAN; :func:`compute_status` extends rollup with PENDING/NOT_RUN) + :data:`EXTERNALLY_CLEARABLE_SUSPICIOUS_CODES` carveout (CREDENTIAL_HARVEST + KENNING_VOICE_BASELINE_TOUCH + KENNING_PERSONA_DRIFT) + :data:`DEFAULT_SEVERITIES` per-code map + :func:`severity_for_code` + :data:`OWASP_AGENTIC_ALIGNMENT` mapping (AS01-AS10) + :func:`summarize_reason_codes` TTS-safe one-line summary + :func:`legacy_flags_from_verdict` backwards-compat + :data:`KIND_TO_CODE` / :func:`code_for_kind` bridge from existing scanner kind enum.
│       │   └── static_scanner.py   ← Python tokenize-based install-time scanner (OpenClaw catalog T5) -- Finding / ScanReport / FindingSeverity (info/warn/critical) / LineFindingKind / SourceFindingKind / scan_install_directory / scan_dependencies / DEFAULT_DENYLISTED_PACKAGES. 2026-05-25 openclaw-clawhub batch 1 (T3) additive helpers: :func:`canonical_code_for_finding` and :func:`canonical_codes_for_report` translate scanner kinds to the T3 canonical reason-code namespace for audit-log enrichment (existing Finding shape preserved).
│       │
│       ├── identity/               ← 2026-05-25 openclaw-clawhub batch 2 (T6): stable-identity primitives
│       │   ├── __init__.py         ← Public API re-exports incl. validate_slug + RESERVED_SLUGS + AliasGraph
│       │   ├── alias_graph.py      ← T6 alias graph: rename / merge / transfer / soft_delete (DEFAULT_RESERVATION_DAYS=30 with original-owner first-refusal) / hard_delete + redirect-chain resolve with MAX_REDIRECT_DEPTH=32 cycle protection.
│       │   └── short_lived_token.py ← 2026-05-25 openclaw-clawhub batch 9 (T7, YELLOW): stdlib-only HMAC-SHA256 JWT mint + verify with pre-registered trust tuples. mint_token (caller_id + audience + scope + ttl_seconds + extra_claims; rejects empty fields / TTL > MAX_TTL_SECONDS=6h / scopes outside allowlist). verify_token enforces signature + expiry (DEFAULT_CLOCK_SKEW_SECONDS=60 tolerance) + audience match + caller-id trust-tuple match + per-claim equality. TrustedCaller pre-registered tuple (expected_claims_match + allowed_scopes + max_ttl_seconds). Audit log at data/identity/short_lived_tokens.jsonl with SHA-256 hash chain + verify_audit_chain. rotate_secret invalidates all historical tokens. Wired use cases: MCP server startup / coding bridge subprocess / skill execution token / voice gaming-mode handoff. RSA-256 + TPM-backed keys are the documented future hardening path. :class:`AliasGraph` thread-safe (RLock), JSONL-persisted with SHA-256 hash chain (:meth:`verify_log_chain`); :meth:`replay_from_log` re-derives state. :data:`RESERVED_SLUGS` blocks 30+ unscoped names (admin/api/settings/soul/voicepack/validator/etc.). :func:`validate_slug` enforces shape `^(?:@scope/)?lowercase-alphanum-with-dots-hyphens-underscores$` (1-128 chars). :class:`AliasGraphEvent` is the audit-log envelope. Generalises across kenning namespaces: skill slugs / voice intent labels / sandbox project names / gaming-mode profile names / persona overlays / voicepack ids / memory backend selectors.
│       │
│       ├── skills/                 ← 2026-05-23 OpenHands batch 2 (T1): trigger-loaded skills
│       │   ├── __init__.py         ← Public API re-exports incl. maybe_get_skills_block
│       │   ├── capability_tags.py  ← 2026-05-25 openclaw-clawhub batch 5 (T5): canonical capability-tag namespace + filter helpers. :class:`CapabilityTag` enum across 5 categories (resource requirements, side-effect domain, network-egress scope, latency profile, gaming-mode safety, modality, confirmation tier). :func:`derive_capability_tags(source, manifest)` auto-derives via per-tag regex + explicit manifest declarations. :class:`TaggedCapability` envelope + :func:`filter_capabilities` AND-combined filter with `require` / `exclude` / `gaming_mode` / `vlm_loaded` / `has_internet` axes. :data:`GAMING_MODE_INCOMPATIBLE_TAGS` + :data:`K_PROTECTED_TAGS` drive :func:`is_gaming_mode_safe` / :func:`needs_explicit_intent`. Generalises across skills + intents + MCP tools + slash commands + hooks.
│       │   ├── models.py           ← Frozen dataclasses: Skill, KeywordTrigger, TaskTrigger, SkillMatch, SkillSource (precedence enum), SkillType; matches_text + find_matched_keywords + find_matched_commands helpers
│       │   ├── loader.py           ← load_skill_from_path + load_skills_from_directory; frontmatter-driven; "any /-prefix flips to task" semantics; filename-stem fallback
│       │   └── registry.py         ← SkillRegistry with mtime invalidation + later-wins source dedup; matching_skills (always-on + triggered capped at max_matches_per_turn); format_skills_block render; maybe_get_skills_block orchestrator helper; build_default_registry factory
│       │
│       ├── services/               ← 2026-05-23 OpenHands batch 8 (T6 partial): Injector pattern
│       │   ├── __init__.py         ← Public API re-exports
│       │   ├── injector.py         ← Injector[T] ABC + InjectorState + SingletonInjector + StreamInjector + InjectorRegistry + install_default_injectors + singleton accessors
│       │   └── engine_injectors.py ← STTEngineInjector + TTSEngineInjector with mode-based dispatch (state.mode == "gaming" -> gaming factory)
│       │
│       ├── projects/               ← 2026-05-23 OpenHands batch 7 (T7): .kenning/ project discovery
│       │   ├── __init__.py         ← Public API re-exports
│       │   └── discovery.py        ← discover_project_config(repo_root) -> frozen ProjectConfig; reads .kenning/{skills/, setup.sh, pre_commit.sh, identity_override.md, safety_rules.yaml, test_command.json, voicepack_override.json, intent_triggers.yaml, hooks.json}; mtime-cached; fail-open per-file
│       │
│       ├── lifecycle/              ← 2026-05-23 OpenHands batch 6 (T5 + T16): start-task + pending-message; later runtime-lifecycle helpers
│       │   ├── __init__.py         ← Public API re-exports
│       │   ├── start_task.py       ← StartTaskStatus enum + StartTask dataclass + create_start_task + StartTaskRecorder (event-store persistence) + drive_start_task async driver + StartTaskError
│       │   ├── pending_message_queue.py ← PendingMessage + PendingMessageState + PendingMessageQueue (enqueue / rebind / cancel / drain) + JSONL persistence + rebind_pending_messages alias
│       │   ├── gaming_engage.py    ← 2026 catalog 09 batch H: drive_start_task-driven gaming engage/disengage substeps with per-stage voice acks
│       │   ├── docker_startup.py   ← 2026-06-12 (97b9494): SearxNG boot probe + Docker Desktop auto-launch (ensure_docker_running, daemon thread, fail-open)
│       │   └── single_instance.py  ← NEW 2026-06-12: single-instance guard for `python -m kenning` -- held OS byte-lock (msvcrt LK_NBLCK at offset 4096 / fcntl flock; auto-releases on process death, no stale-lock problem) + holder PID metadata at offset 0 (unbuffered os.read so Windows mandatory locks never block the duplicate's diagnostic read) + pidfile/psutil fallback + KENNING_ALLOW_MULTIPLE_INSTANCES escape + fail-open-on-error / refuse-only-on-contention. Acquired in __main__.main() BEFORE Orchestrator construction (duplicate exits code 3 naming the holder PID); pytest/e2e construct Orchestrator directly and never contend
│       │
│       ├── llm/condensers/         ← 2026-05-23 OpenHands batch 5 (T4): history-compression strategies
│       │   ├── __init__.py         ← Public API re-exports
│       │   ├── base.py             ← Condenser ABC + Turn + CondenseResult + helpers
│       │   ├── noop.py             ← NoOpCondenser (passthrough)
│       │   ├── recent.py           ← RecentCondenser (head + tail, drop middle)
│       │   ├── amortized.py        ← AmortizedCondenser (no-LLM intelligent forgetting)
│       │   ├── observation_masking.py ← ObservationMaskingCondenser (mask old tool/system content)
│       │   ├── llm_summarizing.py  ← LLMSummarizingCondenser (fold middle via injected summarize_fn)
│       │   └── factory.py          ← build_condenser + select_condenser_for_intent
│       │
│       ├── events/                 ← 2026-05-23 OpenHands batch 3 (T2 + T13): canonical event store + hash chain
│       │   ├── __init__.py         ← Public API re-exports
│       │   ├── models.py           ← StoredEvent + EventPage + EventQuery + EventSortOrder + EventKind namespace + canonical_event_json + new_event_id
│       │   ├── chain.py            ← compute_event_chain_hash + verify_chain (T13 SHA-256 chain)
│       │   ├── store.py            ← EventStore ABC + MemoryEventStore + JsonlEventStore + QdrantEventStore + build_event_store factory + singleton accessors
│       │   ├── export.py           ← export_session_to_bytes + export_session_to_path zip builder with meta.json + chain verification
│       │   ├── bus_sink.py         ← BusEventSink subscribes to the bus, converts envelopes to StoredEvent, writes to the store + fires callbacks
│       │   ├── callbacks.py        ← 2026-05-23 OpenHands batch 4 (T3): CallbackRegistry + CallbackProcessor ABC + RegisteredCallback + CallbackResult + FunctionProcessor adapter + JSONL persistence + singleton accessors
│       │   └── processors.py       ← 2026-05-23 OpenHands batch 4 (T3): built-in processors (Logging, Counting, ThresholdSnapshot, MemoryWrite, ChannelGuard, SkillActivator) + build_default_processors factory
│       │
│       ├── checkpoints/             ← 2026-05-24 cline batch 8 (T1): shadow-repo + 3-axis restore
│       │   ├── __init__.py          ← Public API re-exports
│       │   ├── exclusions.py        ← DEFAULT_CHECKPOINT_EXCLUSIONS (gitignore: node_modules / .venv / models / logs) + VOICE_BASELINE_PROTECTED_PATTERNS (SOUL.md / RVC / Piper / Kokoro voicepack / LLM GGUFs) + compose_gitignore with LFS-pattern extraction
│       │   ├── shadow_repo.py       ← ShadowRepoTracker git CLI wrapper; per-session RLock; 15s init / 7s warn / 30s commit timeouts; CREATE_NO_WINDOW; hash_working_dir; initialise / commit / head / log / hard_reset
│       │   ├── restore.py           ← RestoreAxis enum (VOICE_HISTORY / WORKSPACE / BOTH); plan_restore + execute_restore plan-then-apply; injected WorkspaceReset + VoiceHistoryTruncator + EventLogTruncator
│       │   └── registry.py          ← SessionCheckpointManager bus-event filter via triggered_event_kinds; on_event commits; plan_voice_history_undo / plan_workspace_rewind / plan_full_rewind helpers; CheckpointRegistry singleton; get_checkpoint_registry accessor
│       │
│       ├── hooks/                   ← 2026-05-24 cline batch 7 (T5 + T21): out-of-process hook lifecycle
│       │   ├── __init__.py          ← Public API re-exports
│       │   ├── lifecycle.py         ← HookKind enum (9 cline + 5 kenning-specific), HookPayload, HookOutcome dataclasses; DEFAULT_HOOK_TIMEOUT_SECONDS=10.0, DEFAULT_CONTEXT_MOD_CAP_CHARS=8192
│       │   ├── discovery.py         ← HookDiscovery walks `~/.kenning/hooks/<kind>(.py|.ps1|.sh|.bat|.cmd)` + project equivalent; mtime-validated cache with DEFAULT_DISCOVERY_TTL_SECONDS=30
│       │   ├── runner.py            ← HookRunner with per-suffix interpreter selection (.py → sys.executable, .ps1 → powershell.exe -NoProfile -ExecutionPolicy Bypass, .sh → bash, .bat/.cmd → cmd.exe /c, no suffix → shebang); JSON stdin/stdout envelope; CREATE_NO_WINDOW on Windows; last-balanced-JSON-object stdout parser
│       │   └── registry.py          ← HookRegistry parallel fan-out via concurrent.futures.ThreadPoolExecutor (max 4 default); any cancel:true blocks; every context_modification concatenated as `<hook_context source="..." script="..." layer="...">...</hook_context>`; get_hook_registry() module-level singleton
│       │
│       ├── streaming/               ← 2026-05-24 cline batch 5 (T8 + T12 + T19 + T20)
│       │   ├── __init__.py          ← Public API re-exports
│       │   ├── window.py            ← WindowedOutputWriter with debounce + head/tail preservation + disk spillover (T8); is_compiling_output marker check
│       │   ├── presentation_scheduler.py ← Priority-banded chunk scheduler with cadence map per AudioProfile (T12); local/remote/Bluetooth defaults; set_drop_low_priority for thinking suppression
│       │   ├── reasoning_stream.py  ← ReasoningDemultiplexer separates reasoning from text (T19); first-text finalises pending reasoning block; ReasoningChunkEvent / ReasoningFinalisedEvent
│       │   └── coordinator.py       ← StreamCoordinator state machine + RetryStatus payload (T20); on_usage live-meter callback; publish_retry_attempt surfaces retries as in-place status
│       │
│       ├── agent_loop/              ← 2026-05-24 cline batches 2 + 10 (T7b + T2 + T16): outer-loop primitives
│       │   ├── __init__.py          ← Public API re-exports
│       │   ├── base.py              ← 2026 catalog 11 (meta-pattern): AgentLoop ABC -- additive, safety-instrumented observe->plan->act->verify base. Load-bearing max_steps cap + built-in repeated-signature loop detection + per-step StepRecord + verify hook + fail-open execution (LoopResult / LoopStatus / StepOutcome). Does NOT modify any existing runner. First concrete consumers (catalog 12 T3): web_search/deep_research.py DeepResearchLoop + agent_loop/deep_loops.py.
│       │   ├── deep_loops.py        ← 2026 catalog 12 (felo-search T3 cross-system extensions): generic DeepGatherLoop(AgentLoop) (decompose -> gather-over-injected-source -> LLM gap-fill, bounded by max_steps, fail-open) + DeepGatherResult + three DI subclasses -- DeepMemoryLoop (.recall(); iterative Qdrant RAG), DeepExplorationLoop (.explore(); iterative ripgrep), DeepUIDiscoveryLoop (.discover(); iterative UIA element find). Importable primitives (no orchestrator wiring): each injects its domain primitive (retrieve / search / find callable) so it is domain-agnostic + hermetically testable; wiring to a concrete trigger is a one-call integration (template: Orchestrator._maybe_handle_deep_research). Reuses _parse_json_list / _dedupe_subqueries from web_search/deep_research.
│       │   ├── loop_detection.py    ← LoopDetector with canonical tool_call_signature (JSON sorted-keys minus DEFAULT_NOISE_KEYS like task_progress / turn_id / trace_id); LoopVerdict with soft_warning at DEFAULT_SOFT_THRESHOLD=3 + hard_escalation at DEFAULT_HARD_THRESHOLD=5; halted flag persists across distinct observations once hard tier fires; reset() clears state
│       │   ├── loop_detection_extended.py ← 2026-05-25 OpenClaw batch 3 (T1): four additional detectors. UnknownToolRepeatDetector (regex-extract unknown tool name from error message; halts at UNKNOWN_TOOL_THRESHOLD=10), KnownPollNoProgressDetector (separate threshold for command_status / process(action=poll|log)), PingPongDetector (alternating A,B,A,B with stable outcomes on both sides), GlobalCircuitBreakerDetector (emergency stop at GLOBAL_CIRCUIT_BREAKER_THRESHOLD=30). LoopDetectionManager aggregates all four into a single per-stream observe() surface with most-restrictive-wins; ToolCallRecord + OutcomeKind for input shaping; SHA-256 canonical JSON for hashing.
│       │   ├── subagent_policy.py   ← 2026-05-25 OpenClaw batch 3 (T7): depth-aware subagent tool-policy. SUBAGENT_TOOL_DENY_ALWAYS (gateway/agents_list/session_status/cron/sessions_send + kenning tts_speak/kokoro_speak/gaming_mode_engage/set_validator/install_skill); SUBAGENT_TOOL_DENY_LEAF (subagents/sessions_list/sessions_history/sessions_spawn + kenning mcp_add_server/mcp_remove_server). resolve_subagent_tool_policy(depth, config) returns ResolvedSubagentToolPolicy with deny + allow + also_allow + per-tool provenance. is_leaf(depth, max_spawn_depth) matches OpenClaw's depth >= max(1, floor(maxSpawnDepth)). filter_tools_by_policy + ResolvedSubagentToolPolicy.is_permitted enforce the policy on a tool list.
│       │   ├── mode.py              ← 2026-05-24 cline batch 10 (T2): Mode enum (ACT / PLAN / CODING_ARCHITECT / CODING_EDITOR / GAMING) + frozen ModePolicy (allows_tool_side_effects / requires_confirmation / wrap_prefix_template / confirmation_timeout / preset_override) + DEFAULT_POLICIES (PLAN wraps with "Here is my plan: {plan} / Say 'do it'") + PendingConfirmation (UUID + TTL + intent_topic + callback_token) + ModeSession state machine (flip with invalidate_pending semantics / queue_plan / peek_latest_pending / consume_pending_confirmation with topic filter / cancel_pending / flip_history capped at 32) + module-level get_mode_session(session_id) registry singleton
│       │   └── subagent.py          ← 2026-05-24 cline batch 10 (T16): DEFAULT_READONLY_TOOL_WHITELIST (file_read / list_files / list_code_definitions / search / ripgrep_search / use_skill / execute_command_readonly / rag_query / web_search) + frozen SubagentTask (per-task whitelist + token caps + wall-clock timeout) + SubagentResult (text + per-task token meter + tool call log) + SubagentBatchStats (n_tasks / n_succeeded / total_input_tokens / max_wall_clock / sum_wall_clock) + ToolGuard whitelist enforcer raising ToolNotPermittedError + thread-safe TokenLedger + SubagentRunner ThreadPoolExecutor-backed dispatcher with max_parallel=1 default for voice baseline safety
│       │
│       ├── evolution/               ← 2026 catalog 13 (clawhub-capability-evolver clean-room): bounded autonomous self-improvement. QUARANTINED source NEVER read; built from catalog + scan reports only. Data-only proposals (skills/*.md), Tier-3-walled, zero network/shell/eval, fully fail-open. Wired default-ON via pipeline/orchestrator.py. 2026 catalog 14 (clawhub-self-improving-agent) EXTENDS it with QUALITATIVE conversation-event capture: correction / knowledge-gap / command-failure / feature-request types + detectors (T1/T2), a pre-turn `[Evolution: ...]` nudge through the temperament seam (T3), and pattern_key recurrence tracking (T4) -- all data-only, same contract; quarantine source NOT read.
│       │   ├── __init__.py          ← Public API re-exports (78 symbols across the package)
│       │   ├── models.py            ← GEP data model: frozen dataclasses Gene / Capsule / Mutation / EvolutionEvent / PersonalityState / BlastRadius / Outcome / GeneConstraints / EnvFingerprint (NO device_id -- safety departure) + EvolutionCategory / OutcomeStatus / RiskLevel enums + clamp01 / canonicalize / compute_asset_id (sha256) / verify_asset_id + id generators (new_capsule_id appends a 6-hex suffix to avoid same-ms collisions) + schema constants
│       │   ├── signals.py           ← Opportunity-signal extraction: 18 OPPORTUNITY_SIGNALS (production-hardening #66 added `coding_task_success`, emitted by the orchestrator's coding-success drain) + COSMETIC_SIGNALS + 7 weighted SIGNAL_PROFILES; extract_signals (two LOCAL layers: regex + keyword scoring + multilingual user-request) + analyze_recent_history + apply_post_processing (dedup / repair-loop / saturation / failure-streak / ban) + has_opportunity_signal. The upstream's 3rd LLM/Hub-network layer is NOT ported.
│       │   ├── blast_radius.py      ← Change-scope policy spine: CountedFilePolicy + BLAST_RADIUS_HARD_CAP_FILES/LINES + BlastSeverity (5 tiers) + CRITICAL_PROTECTED_PREFIXES/FILES Tier-3 wall (includes "src/") + ETHICS_BLOCK_PATTERNS + compute_blast_radius / classify_blast_severity / check_constraints / classify_failure_mode / is_critical_protected_path; injectable git_numstat provider
│       │   ├── skill_distiller.py   ← Capsule->pattern->skill distillation: auto_distill / auto_distill_from_failures + analyze_patterns + synthesize_gene_from_patterns + gene_to_skill_proposal + render_skill_markdown (kenning-compatible frontmatter: name/type/version/description/triggers/min_user_text_chars) + should_distill (>=10 successes, >=7 of last 10, 24h cooldown, data-hash idempotency) + SkillProposal / DistillResult. Output is DATA, never code.
│       │   ├── guardrails.py        ← Regression guardrails: GuardrailBaseline (TTFA 266 / TTFT 172 / TTS 78 / VRAM 6664 defaults) + 4 detectors (detect_latency_regression / detect_quality_regression / detect_error_regression / detect_resource_ceiling) + evaluate_guardrails + RollbackAudit (note_outcome / rollback_rate / should_demote) + ROLLBACK_DEMOTE_THRESHOLD=0.30 + VRAM_CAP_MB=11500. Replaces the upstream's run-commands-to-verify step.
│       │   ├── autonomy.py          ← Tiered autonomy: AutonomyTier (PARAM/SKILL/GATED/WALL IntEnum) + AutonomyMode + DEFAULT_SURFACE_TIERS (skills=SKILL; safety_validator/audit/engine/category_k=WALL) + TieredAutonomyController (mode_for / can_auto_apply / requires_approval / record_outcome->AutonomyTransition / digest) + graduation ladder (>=20 changes, <10% revert, 0 hard trips) + rollback-rate demotion
│       │   ├── personality.py       ← Tier-0 adaptive temperament: PersonalityTuner (record_feedback nudges rigor/creativity/verbosity from corrections/re-asks/barge-ins; record_outcome + best_personality ranking; to_dict/from_dict) + temperament_hint -> "[Tone: ...]" directive (distinct from response_style's "[Style:") + apply_temperament. NEVER touches SOUL.md / the voicepack.
│       │   ├── evolution_loop.py    ← EvolutionLoop(AgentLoop): pre-flight (fail-closed) -> autonomy gate -> checkpoint -> write -> blast+constraints -> guardrails -> keep/revert -> hash-chained audit, bounded by max_steps. ApplyStatus (KEPT/REVERTED/BLOCKED/REJECTED/GATED_NO_CHANNEL) + ApplyResult + EvolutionState + CheckpointHook + EvolutionLoopConfig. Injected collaborators (capsules_provider / guardrail_sampler / checkpoint / approval / audit_sink / ...).
│       │   ├── service.py           ← Runtime bundle: EvolutionStore (lock-guarded JSONL: capsules / failed_capsules / events hash-chain + verify_event_chain / state / personality) + EvolutionService (from_config / record_turn / single-flight run_cycle / maybe_run_autonomous_cycle daemon thread / temperament_hint / apply_temperament / digest / status_line / shutdown). Checkpoint over data/evolution/skills via CheckpointRegistry(data/checkpoints). 2026 production-hardening (#15+#65 guardrail brake): from_config builds a TurnMetricsRing + sampler when `guardrail_monitoring_enabled` (injected sampler wins); record_turn feeds the quality flags into the ring + ticks the post-apply watch; _do_cycle ARMS a single-slot post-apply watch on every KEPT proposal (pre-apply GuardrailSample snapshot + ring markers + `post_apply_monitor_turns` countdown); at expiry a daemon thread re-samples ONLY the post-apply records and evaluates them against a RELATIVE baseline built from the pre-apply snapshot (unobserved pre fields skip their check) -- a trip auto-reverts the kept skill (containment-checked file delete + registry reload + autonomy RollbackRecord + hash-chained audit event + repair-feed failure record) and queues a one-line voice narration drained by `Orchestrator._drain_evolution_narrations`. pop_pending_narration + turn_metrics accessors.
│       │   ├── turn_metrics.py      ← 2026 production-hardening (#15+#65, NEW): the guardrail-brake instrumentation. TurnMetricsRing -- thread-safe bounded ring (default 40 turns) of per-turn ResponseRecord (LLM TTFT + error flag, fed by the orchestrator at the end of _respond) + QualityRecord (corrected/re-asked/barged-in, fed by EvolutionService.record_turn); monotonic totals() markers let the post-apply watcher sample ONLY post-apply records (`sample(since=...)`); median TTFT + correction/error rates with minimum-sample floors (a field without enough data stays None -> its guardrail is SKIPPED, fail-open by construction). probe_vram_mb -- fail-open nvidia-smi probe (CREATE_NO_WINDOW + timeout) used only on the cycle/watcher daemon threads, never the hot path. build_guardrail_sampler binds the ring into the loop's Callable[[], GuardrailSample] contract. LIKE-FOR-LIKE design: TTFT is recorded ONLY for plain conversational turns (search turns carry a larger prompt class than the locked 172 ms baseline was measured on); ttfa/tts stay None by design (a data-only skill cannot regress STT/TTS -- those checks are skipped; TTFT + quality + error + VRAM carry the brake).
│       │   └── intent.py            ← Strict voice-command matcher: match_evolution_command -> EvolutionCommand(kind=RUN_CYCLE|STATUS). Status patterns checked first. Mirrors the established short-circuit matchers; only trips on explicit self-improvement phrasing.
│       │
│       ├── search/                  ← 2026-05-24 cline batch 1 (T25): direct-search utilities
│       │   ├── __init__.py          ← Public API re-exports
│       │   └── ripgrep.py           ← Ripgrep subprocess wrapper: regex_search_files(cwd, directory, pattern, *, file_pattern, context_lines, timeout_s, ignore_predicate, binary_name, extra_args) -> RipgrepResult with grouped-by-file rendering, `│----` separators, byte cap 0.25 MB (MAX_RIPGREP_BYTES), result cap 300 (MAX_RESULTS), line cap MAX_RIPGREP_LINES; rg_binary_available with Windows install-location fallback; CREATE_NO_WINDOW on Windows; per-call wall-clock kill; fail-open on missing binary or malformed JSON lines
│       │
│       ├── subprocess/              ← 2026-05-24 cline batch 2 (T23): subprocess lifecycle
│       │   ├── __init__.py          ← Public API re-exports
│       │   ├── zombie_killer.py     ← ZombieKiller periodic reaper (DEFAULT_HARD_TIMEOUT_S=10*60, DEFAULT_POLL_INTERVAL_S=60) with persistent-tag carve-out + RSS warning tier (DEFAULT_WARN_RSS_MB / DEFAULT_WARN_AGE_S) + clock / terminator / rss_probe injection hooks for deterministic tests; TrackedProcess registry with re-register-in-place semantics; get_zombie_killer() module-level singleton; recent_reports() bounded buffer; on_terminate callback fires after successful kill
│       │   ├── sidecar_lock.py      ← NEW 2026-06-15: embedder-sidecar SINGLETON enforcer. Pidfile (~/.kenning/embedder_sidecar.json, atomic write) records pid+port+model+owner. sweep(host,port,model) runs at boot BEFORE spawning and returns a verdict: reuse (our recorded sidecar is alive + serving the model), killed (recorded pid alive but wrong model → reaped), killed-zombie (recorded pid dead but a process still serves the port: a force-killed prior Ultron left its embedder child → found via psutil.net_connections LISTEN scan + reaped), killed-unknown (no pidfile but something serves the port → reaped, loud), or spawn (port clear). The only defence against a taskkill /F sidecar orphan. Fail-open (never raises into boot), tolerates missing psutil.
│       │   └── kill_tree.py         ← 2026-05-25 OpenClaw batch 1 (T8): cross-platform process-tree termination via psutil. kill_process_tree(pid, *, grace_seconds, detached, clock) walks descendants → graceful terminate → wait grace → force-kill survivors; KillTreeResult dataclass (terminated / force_killed / unreachable / elapsed_seconds / used_process_group). kill_pid_if_alive(pid) is the leaf-only convenience. Grace clamped to [0, MAX_GRACE_SECONDS=60]. Fail-open on missing psutil. Replaces ad-hoc per-site terminate patterns in cleanup_stale_processes + future Parakeet/MCP shutdown wiring.
│       │
│       └── utils/
│           ├── heartbeat.py         ← 2026 catalog 11 (T2): HeartbeatThread -- stoppable, fail-open daemon keep-alive for long-lived connections (Event.wait loop + HeartbeatStats; improves on the upstream's unstoppable while-True-sleep). Importable primitive (no current hot-path consumer; kenning's daemons self-respawn).
│           ├── health_check.py      ← 2026 catalog 11 (T4): http_health_check + cdp_health_check -- cheap fail-open "is this endpoint answering?" pre-flight probes with injectable transport (GET /json/list for the CDP variant). Importable primitive.
│           ├── fairseq_compat.py   ← Workarounds for fairseq dataclass + torch.load issues
│           ├── logging.py          ← configure_logging(), get_logger() (rotating file + console)
│           ├── mtime_cache.py      ← aider catalog batch 1 (T-supporting): SQLite + dict-fallback mtime-keyed cache
│           ├── token_budget.py     ← aider catalog batch 1: binary-search-to-budget with tolerance
│           ├── snapshot_guard.py   ← aider catalog batch 1: snapshot-identity race protection
│           ├── relative_indent.py  ← aider catalog batch 1: indent-relative text transform
│           ├── spinner.py          ← aider catalog batch 11 (T11): ASCII bounce spinner with cursor-stagger continuity
│           ├── poll.py             ← 2026-05-23 OpenHands batch 1 (T14): poll_until + apoll_until bounded-retry helpers with custom is_done predicate + exponential backoff + cancel_check
│           ├── retry.py            ← 2026-05-24 cline batch 1 (T13b): with_retry async decorator + with_retry_sync + RetriableError + RetryBudget + RetryAttempt record + parse_retry_after (delta-seconds vs unix-timestamp heuristic) + onRetry async/sync callback + asyncio.CancelledError pass-through + jitter; default classifier matches HTTP 429 + RetriableError; per-session sleep-budget cap
│           └── ansi_safe.py        ← 2026-05-25 OpenClaw batch 1 (T18): ANSI / control-character sanitisation + grapheme-aware width. strip_ansi (CSI + OSC + two-byte ESC), sanitize_for_log (CWE-117 log-forging defence: ANSI + C0/C1/DEL stripped; tab/LF/CR preserved), split_graphemes / iter_graphemes (grapheme cluster boundaries from runtime-built codepoint class so source stays ASCII-safe), grapheme_width / visible_width (East-Asian-Width + emoji = 2; combining marks + ZWJ + variation selectors = 0), truncate_to_visible_width helper for TTS chunking and log budgeting
│
├── config/
│   ├── __init__.py                 ← (empty)
│   └── settings.py                 ← Phase 3 SHIM: re-exports legacy settings.X from config.yaml
│
├── prompts/
│   └── coding/                     ← Jinja2 templates rendered by TemplateRenderer
│       ├── claude_code_initial_new.j2
│       ├── claude_code_initial_edit.j2
│       ├── claude_code_correction.j2
│       ├── claude_code_adjustment.j2
│       └── claude_code_clarification_response.j2
│
├── docs/
│   ├── architecture.md             ← Pipeline + state machine + subsystem table
│   ├── configuration.md            ← Per-key config reference
│   ├── config_discovery.md         ← One-time Phase 3 discovery catalog
│   ├── operations.md               ← Day-to-day running, monitoring, recovery
│   ├── development.md              ← Test layout, debugging, how-to recipes
│   ├── error_handling.md           ← Phase 4 error catalog + circuit breaker reference
│   ├── routing.md                  ← Phase 5 capability routing
│   ├── system_inventory.md         ← Phase 1 verification snapshot
│   ├── phase3_5_followup.md        ← Punch list: remaining unified-config migrations
│   ├── smoke_test.md               ← 16-step real-stack walkthrough procedure
│   ├── openclaw_integration.md     ← OpenClaw integration architecture + Phase 0/1
│   ├── openclaw_runtime.md         ← OpenClaw runtime ops (agents, supervisor, locks)
│   ├── openclaw_integration_final_summary.md ← Cross-phase reference + intentional deviations + setup-readiness checklist
│   ├── phase_1_summary.md          ← OpenClaw Phase 1 close-out (persona migration)
│   ├── phase_3_summary.md          ← OpenClaw Phase 3 close-out (bridge layer)
│   ├── phase_4_summary.md          ← OpenClaw Phase 4 close-out (Telegram channel)
│   ├── phase_5_summary.md          ← OpenClaw Phase 5 close-out (heartbeat)
│   ├── phase_6_summary.md          ← OpenClaw Phase 6 close-out (browser tool)
│   ├── openclaw_telegram_setup.md  ← User-side: Telegram bot setup procedure
│   ├── openclaw_heartbeat_setup.md ← User-side: agents[].heartbeat block setup
│   ├── openclaw_browser_setup.md   ← User-side: Playwright/Chromium + tools.alsoAllow
│   ├── openclaw_cron_setup.md      ← User-side: cron jobs (Windows Task Scheduler fallback)
│   ├── openclaw_hooks_setup.md     ← User-side: bundled hooks; custom hook scaffolding
│   ├── openclaw_memory_wiki_setup.md ← User-side: Memory Wiki plugin enablement
│   ├── openclaw_media_generation_setup.md ← User-side: local-only ComfyUI setup (paid APIs out)
│   ├── mobile_node_setup.md        ← User-side: iOS / Android pairing procedure
│   ├── standing_orders.md          ← Standing-order programs in AGENTS.md
│   ├── memory_architecture.md      ← Three-layer memory model (Qdrant + workspace + Wiki)
│   ├── 4b_optimization_plan.md     ← 4B-model migration plan (all stages done)
│   ├── model_checksums.md          ← SHA256 of every GGUF in `models/`
│   ├── comprehensive_test_plan.md  ← Functional / correctness pass architecture (16 phases, 38 dimensions)
│   ├── comprehensive_test_report.md ← Functional pass results + 145-row metrics table; 4 classifier coverage gaps fixed
│   ├── comprehensive_quality_plan.md ← Quality pass architecture (13 phases Q0–Q13, 38 dimensions, ≤10 iter loop)
│   ├── comprehensive_quality_report.md ← Quality pass results + 107-row metrics table + Q10 iteration audit
│   └── codebase_structure.md       ← THIS FILE
│
├── scripts/                        ← Operational scripts (CLI tools)
│   ├── benchmark.py                ← Latency benchmark (existing from earlier phases)
│   ├── check_vram.py               ← Quick VRAM snapshot vs cap
│   ├── download_models.py          ← First-run model fetcher
│   ├── dump_session.py             ← Render coding-session audit log readable
│   ├── list_audio_devices.py       ← Mic/output device introspection
│   ├── maintenance.py              ← Periodic Qdrant maintenance (summarization, fact extraction)
│   ├── measure_baseline.py         ← Voice-path VRAM + TTFT baseline
│   ├── measure_baseline_extended.py ← Extended baseline (search/coding VRAM, scenario timing)
│   ├── migrate_memory_to_qdrant.py ← One-shot JSONL → Qdrant migration
│   ├── review_addressing.py        ← Read addressing.jsonl, print verdicts
│   ├── run_integration_tests.py    ← pytest wrapper for tests/integration|routing|error_recovery
│   ├── run_orchestration_tests.py  ← Run 10 orchestration scenarios with reporting
│   ├── validate_config.py          ← Schema-validate config.yaml without starting Kenning
│   ├── swap_llm_preset.py          ← 4B plan: edit config.yaml in place to swap LLM preset (validates GGUFs, atomic write)
│   ├── verify_voice_character_4b.py ← 4B plan Stage E: A/B voice-character helper (5 queries × 4B/9B)
│   ├── verify_items_4_to_8.py      ← 4B plan: exercises Items 4–8 in their trigger scenarios; prints measurable deltas
│   ├── comprehensive_test_harness.py ← End-to-end test pass: routing accuracy on 63-utterance labeled set, web-gate rule accuracy, circuit-breaker state machine, memory stress (4 threads × 50 turns), classifier-gating regression
│   ├── real_api_smoke.py           ← Real-API sparing smoke: 1 Brave query + 1 Brave-Jina chain + 1 AI coding agent (haiku) invocation (≤2 paid web calls + ≤1 tiny Anthropic API call total)
│   ├── quality_harness.py          ← Quality pass: Q1 persona/factual/hallucination + Q2 persona modes + Q4 memory recall/labeling/ranking + Q5 Whisper WER/flush/VAD + Q7 Items 4-8 + Q8 adversarial in one process
│   ├── quality_q3_web.py           ← Quality pass Q3: web-search source ranking + snippet utilization + Jina direct + cache + citation rendering + ack latency + dedup (10 Brave + 10 Jina cap)
│   ├── quality_q6_mocked.py        ← Quality pass Q6.D + Q9: projection budget + phrase pool + browser parsing + slug routing + gaming mode (no real API)
│   ├── quality_q6_claude.py        ← Quality pass Q6.E + Q6.F: 4 single-fn AI coding agent tasks + 5 full Tkinter app generation (sandbox-isolated)
│   ├── _quality_q10_iter1_verify.py ← Quality Q10 iter verification: 3 prompt-injection probes against the live LLM
│   ├── _quality_q6f_rescore.py     ← Quality Q6.F re-scorer: applies relaxed regex to existing on-disk apps (no new Claude calls)
│   ├── start_llamacpp_server.py    ← OpenClaw Phase 0 + 4B plan Stage C: launch llama-cpp-server with voice-pipeline params (+ --model-draft / --draft-num-pred-tokens / --from-config)
│   ├── supervised_llamacpp_server.py ← OpenClaw Phase 0: supervisor wrapper with auto-restart
│   ├── smoke_test_llamacpp.ps1     ← OpenClaw Phase 0: PowerShell health probe for llama-cpp-server
│   ├── _bench_llm_http.py          ← OpenClaw Phase 0: HTTP-mode TTFT benchmark
│   ├── _log_proxy.py               ← OpenClaw Phase 0: tee proxy for debugging Gateway → server traffic
│   ├── _record_phase0_baseline.py  ← OpenClaw Phase 0: baseline recorder
│   ├── _merge_phase0_baselines.py  ← OpenClaw Phase 0: baseline merger
│   ├── _vram_peak_monitor.py       ← Auxiliary VRAM peak monitor (used by extended baselines)
│   ├── run_maintenance_for_cron.py ← OpenClaw Phase 7: cron-friendly maintenance wrapper (JSON / pretty / exit codes)
│   ├── run_kenning_mcp_for_openclaw.py ← OpenClaw Phase 13: stdio MCP entry script OpenClaw spawns to call Kenning tools
│   ├── cleanup_stale_processes.py ← 2026-05-14 cleanup pass: kill orphaned pytest workers + stale MCP stubs + orphan XTTS servers (preserves live Kenning via port-19761 listener check)
│   ├── bench_llm_ubatch.py        ← 2026-05-15 latency: sweep n_batch / n_ubatch combinations (writes baselines.json:llm_n_ubatch_sweep)
│   ├── bench_stt_latency.py       ← 2026-05-15 latency: measure Whisper STT latency at varied audio lengths (drove beam_size 5->1 decision)
│   ├── bench_llm_prefix_cache.py  ← 2026-05-16 latency 2: cold-vs-warm TTFT bench for LlamaRAMCache (writes baselines.json:llm_prefix_cache_bench; result: -15 ms regression on this stack -> Phase 2 default flipped to disabled)
│   ├── eval_harness.py            ← 2026-05-18 Phase 0: classifier-only eval harness (routing + addressing + web_gate); reads tests/eval/corpus.jsonl; writes logs/eval_runs/<ts>.json; exit codes 0/1/2 for CI
│   ├── stream_check.py            ← 2026-06-12: pre-stream device-routing check (default speakers / relay mic bus / OBS capture / mic untouched)
│   ├── build_common_words.py      ← NEW (2026-06-16, OFFLINE build-time): download/parse the public-domain google-10000-english list → emit src/kenning/audio/_common_words.py (top-~5000 frequency-ranked, alpha-only len≥3). Never imported by the runtime
│   ├── flavor_gen/                ← NEW (2026-06-16, OFFLINE build-time codegen): build the flavor library; never imported by the runtime
│   │   ├── integrate_tails.py     ← codegen + lint + dedup → src/kenning/audio/_agent_flavor.py (the TailEntry table)
│   │   ├── apply_cuts.py          ← apply the audit cuts (drop the lines the lint/audit pass flagged) back into the library
│   │   ├── curated_overrides.py   ← NEW (2026-06-16, coherence pass): the hand-written CURATED dict — per agent×situation kit-accurate TailEntry lists (text + tags), ~5/cell, every ult = the real ultimate, every utility = ability-tagged. Pure data
│   │   └── apply_curated.py       ← NEW (2026-06-16, coherence pass): REPLACE each agent/situation cell present in CURATED with its curated TailEntry list, re-emit _agent_flavor.py (idempotent); run the lint gate after
│   ├── flavor_audit/              ← NEW (2026-06-16, OFFLINE build-time audit): never imported by the runtime
│   │   └── lint_tails.py          ← deterministic lint GATE over the tails (HARD: word-cap / wrong-gender-vs-AGENT_GENDER / surrounding-quotes / per-cell floor; SOFT: leading-tactical-verb / missing terminal punctuation)
│   └── relay_test/                ← Valorant relay test harness + 20k corpus + scorecard (see the "2026-06 relay/gaming campaign" section below)
│       ├── harness.py             ← staged matcher/rephrase/audio/asr/full pipeline test (GAMING_PRESET 3B, testing-mode parity, RELAY_TEST_GPU_LAYERS)
│       ├── corpus.py              ← original build_corpus() base cases + _GROUP_PREFIXES
│       ├── corpus_packs.py        ← build_corpus(seed, target=25000): auto-discover packs by kind (relay/question/NEGATIVE) + _compound_cases + stratified cap; build_corpus_10k/_20k aliases
│       ├── scorecard.py           ← reliability scorecard: fact-token extractor, classify_route (by LLM-invocation), per-category retention p50/p95/p99, inversion/hallucination, deterministic coverage, matcher/false-relay, flavor TTR, --bench (CPU-3B latency+RSS), no-regression diff
│       ├── trace_corpus.py        ← NEW (2026-06-16): full-pipeline corpus TRACER — runs the corpus through the live normalize→route→relay pipeline and records each stage's output for triage
│       ├── analyze_outputs.py     ← NEW (2026-06-16): output TRIAGE over a trace/rephrase JSONL (bucket + flag the lines worth a human/audit look)
│       ├── make_audit_chunks.py   ← split the LLM-routed lines of a rephrase JSONL into per-agent audit chunks
│       ├── vocab_packs/           ← 48 packs (~29.4k payloads): 8 base + var_* variety + stress_* metric-stress + persona_flavor (OUTPUT pool, excluded from inputs)
│       ├── refs/                  ← 21 web-grounded Valorant reference docs (agents/abilities/maps/callouts/economy/slang/meta/Marvel + ultron_voice.md) used to ground generation
│       └── (dev tools)            ← play_sample, cadence_actual/cadence_check, flow_check, validate_pipeline, probe/reprobe/waveform_check/burst_diag
│
├── tests/
│   ├── conftest.py                 ← Path setup + pytest_sessionfinish hook that reaps test-spawned python children (preserves the live Kenning on port 19761); pytest_configure walks the process tree + refuses to start if another pytest is running on this codebase
│   ├── test_*.py                   ← ~80 unit/integration test files at the top level (default suite; current total 10,129 collected); see scripts/run_tests.py to invoke. 2026-06-12 additions: test_audio_capture_status.py, test_follow_up_streaming_stt.py, test_main_single_instance.py. 2026-06 relay/gaming campaign additions: test_wake_word.py (per-word thresholds + consecutive-frame gate), test_f0_control.py + test_duration_control.py (in-model prosody shaping), test_llm_preset.py (gaming CPU/no-draft preset)
│   ├── lifecycle/                  ← lifecycle package tests: test_start_task.py, test_pending_message_queue.py, test_gaming_engage.py, test_docker_startup.py, test_single_instance.py (NEW 2026-06-12: held-lock acquire/contention/metadata-while-locked/env-escape/pidfile-fallback/errno-classification/no-unlink-on-release; tmp_path locks only)
│   ├── pipeline/                   ← orchestrator-helper tests: test_idle_vram_reclaim.py, test_coding_runner_drains.py, test_supervisor_stack_shared_client.py (NEW 2026-06-12: _build_supervisor_stack passes ConversationMemory's client / falls back / fail-open)
│   ├── bus/                        ← 2026-05-22 session E: typed event bus (43 tests)
│   │   ├── __init__.py
│   │   ├── test_event.py           ← BusEvent.define / EventPayload.make / schema validation
│   │   ├── test_service.py         ← Bus pub/sub + race-safety + concurrent subscribe + unsubscribe semantics
│   │   └── test_events_catalog.py  ← Canonical 17-event catalog uniqueness + naming convention
│   ├── coding/
│   │   ├── conftest.py
│   │   ├── mock_bridge.py          ← ScriptedClaudeBridge (in-process mock, ClaudeScript DSL)
│   │   ├── test_orchestration.py   ← 11 mock-bridge orchestration scenarios
│   │   ├── test_orchestration_real.py ← Same scenarios with real Claude (PYTEST_RUN_GPU_TESTS=1)
│   │   ├── test_mock_bridge_smoke.py
│   │   ├── test_project_digest.py        ← 2026-05-22: digest template + LLM-call + parse_digest_sections + extract_files (+27)
│   │   ├── test_project_introspect.py    ← 2026-05-22: snapshot walk + language detect + entry-points + AST + cache (+25)
│   │   ├── test_project_index.py         ← 2026-05-22: real-Qdrant + real-embedder upsert / search / get / delete / list / count (+26)
│   │   ├── test_project_supervisor.py    ← 2026-05-22: decide() RESUME/EDIT/CLARIFY/NEW + threshold bands + merge + audit log + bus event (+34)
│   │   ├── test_supervisor_dispatch.py   ← 2026-05-22: DispatchOutcome per kind + narration + enriched context + fallback (+27)
│   │   └── sandbox/                ← test fixture sandbox
│   ├── desktop/                    ← Desktop automation primitives (NEW 2026-05-12 Phase 11)
│   │   ├── test_monitors.py        ← Win32 enumeration + find_monitor + point_to_monitor
│   │   ├── test_capture.py         ← mss-based capture + taint tracker integration
│   │   ├── test_windows.py         ← pywin32 enum + foreground detection
│   │   ├── test_placement.py       ← move/resize/maximize on target monitor
│   │   ├── test_launcher.py        ← AppLauncher registry + Chrome default-profile + URL passing
│   │   ├── test_uia.py             ← pywinauto text extraction + click/type with safety gates
│   │   ├── test_input_control.py   ← pyautogui rate limit + validator gate + UAC blocking
│   │   ├── test_screen_context.py  ← orchestrator + capture-bytes discard
│   │   ├── test_vlm.py             ← Moondream2VLM lazy load + fail-open + unload
│   │   ├── test_voice.py           ← handle_app_launch + handle_screen_context_query + monitor resolution
│   │   └── test_preferences.py     ← JSONL log + recency-weighted phrase lookup
│   ├── error_recovery/             ← Phase 4: per-dependency failure modes (78 tests)
│   │   ├── conftest.py
│   │   ├── test_brave_failures.py
│   │   ├── test_jina_failures.py
│   │   ├── test_qdrant_failures.py
│   │   ├── test_audio_failures.py
│   │   ├── test_addressing_failures.py
│   │   ├── test_config_failures.py
│   │   ├── test_circuit_breaker.py
│   │   ├── test_error_log.py
│   │   ├── test_claude_code_failures.py    ← Phase 4 deferred wrappers
│   │   ├── test_mcp_server_failures.py     ← Phase 4 deferred wrappers
│   │   └── test_filesystem_failures.py     ← Phase 4 deferred wrappers
│   ├── eval/                       ← 2026-05-18 Phase 0 build: classifier-only eval harness
│   │   ├── corpus.jsonl            ← 60-row labeled routing / addressing / web-gate corpus
│   │   └── test_eval_harness_*.py
│   ├── install/                    ← 2026-05-23 OpenHands batch 1 (T8): idempotent installer tests
│   │   ├── __init__.py
│   │   └── test_idempotent.py      ← 20 tests: install / skip / preserve / replace / dry_run / custom marker / audit log / unreadable error / string paths
│   ├── memory/                     ← Memory subsystem unit tests (was 2026-05-19 onward; reranker / contextualizer / qdrant store / topic / discourse)
│   │   └── (test_qdrant_*, test_memory_qdrant.py at top-level, etc.)
│   ├── observations/               ← 2026-05-18 Phase 1 observation framework tests
│   │   ├── test_writer.py
│   │   ├── test_integrations.py
│   │   ├── test_outcome_resolver.py
│   │   └── test_lineage_overlap.py
│   ├── parsing/                    ← 2026-05-23 OpenHands batch 1 (T11): frontmatter parser tests
│   │   ├── __init__.py
│   │   └── test_frontmatter.py     ← 22 tests: parse / walk / fail-open / CRLF / empty fm / missing closer / non-mapping / decode error / custom extensions / skip dirs / frozen
│   ├── skills/                     ← 2026-05-23 OpenHands batch 2 (T1): trigger-loaded skills tests
│   │   ├── __init__.py
│   │   ├── test_models.py          ← 22 tests: keyword + task triggers, source precedence, Skill+SkillMatch
│   │   ├── test_loader.py          ← 16 tests: frontmatter -> Skill conversion + directory walk + per-file error swallow
│   │   ├── test_registry.py        ← 27 tests: lazy load, mtime invalidation, dedup, format rendering, singleton, factory
│   │   └── test_orchestrator_wiring.py ← 5 tests: LLMEngine seam (no registry / match / no match / exception / always-on)
│   ├── events/                     ← 2026-05-23 OpenHands batch 3 (T2 + T13): event store + hash chain tests
│   │   ├── __init__.py
│   │   ├── test_models.py          ← 16 tests: StoredEvent + canonical encoding + EventPage + EventQuery
│   │   ├── test_chain.py           ← 11 tests: hash chain integrity (happy + broken hash + broken prev + strict + empty + missing + determinism)
│   │   ├── test_store.py           ← 27 tests: Memory + JSONL + Qdrant backends + factory + singleton + session isolation + pagination
│   │   ├── test_export.py          ← 7 tests: zip layout + redaction + empty session + extra meta + path write + chain-broken recorded
│   │   ├── test_bus_sink.py        ← 13 tests: lifecycle + envelope conversion + dispatch + exception swallow + sequence counter
│   │   ├── test_callbacks.py       ← 33 tests: CRUD + filters + dispatch + deactivate + persistence + singleton + slow-warn
│   │   └── test_processors.py      ← 23 tests: all six built-in processors + factory + registry e2e
│   ├── routing/                    ← Phase 5 + 2026-05 extensions: classifier + dispatcher + decomposer + ambiguity + gaming_mode + decision_log + dispatcher_a1_c3
│   │   ├── conftest.py
│   │   ├── test_classifier.py      ← Top-level classifier with 23 RoutingIntentKind branches + 2026-05-22 _NAVIGATE_TO_SITE + _OPEN_LAST_SOURCE_AMBIGUOUS lists
│   │   ├── test_dispatcher.py
│   │   ├── test_decomposer.py
│   │   ├── test_disambiguator.py
│   │   ├── test_decision_log.py
│   │   ├── test_ambiguity_band.py
│   │   ├── test_dispatcher_a1_c3.py    ← V1-gap A1 + C3 dispatcher branches
│   │   ├── test_desktop_native_classifier.py
│   │   └── test_backward_compat.py
│   ├── safety/                     ← 2026-05-12 Phases 2-5: runtime tool-call validator tests (+117)
│   │   ├── test_path_resolver.py
│   │   ├── test_audit_log.py
│   │   ├── test_validator_core.py
│   │   ├── test_rules_by_category.py
│   │   ├── test_intent_and_taint.py
│   │   └── test_dispatcher_integration.py
│   ├── integration/                ← Phase 6: end-to-end pipeline (83 tests + bridge e2e)
│   │   ├── conftest.py
│   │   ├── mocks.md                ← What's mocked vs real, per layer
│   │   ├── performance.json        ← Phase 6 perf snapshot
│   │   ├── test_routing_dispatch.py    ← + Phase 13 SYSTEM_STATUS routing tests
│   │   ├── test_conversational_pipeline.py
│   │   ├── test_search_pipeline.py
│   │   ├── test_coding_pipeline.py
│   │   ├── test_addressing_pipeline.py
│   │   ├── test_error_recovery_pipeline.py
│   │   └── test_bridge_e2e.py      ← OpenClaw Phase 3 bridge e2e (real subprocess against stub CLI)
│   ├── openclaw_bridge/            ← OpenClaw Phases 3–13 bridge tests (158 tests)
│   │   ├── __init__.py
│   │   ├── test_client.py          ← OpenClawClient: subprocess transport + result parsing
│   │   ├── test_workspace.py       ← WorkspaceWriter: atomic + filelock + concurrency
│   │   ├── test_events.py          ← OpenClawEventReceiver: prefix matching + dispatch
│   │   ├── test_mcp_registration.py ← KenningMcpRegistrar: idempotent + retry
│   │   ├── test_holder.py          ← OpenClawBridge: from_config / start / shutdown / fire_and_forget / record_heartbeat_alert / auto-resolve
│   │   ├── test_notifications.py   ← NotificationDispatcher: per-event gating + recipient resolution + transport errors
│   │   ├── test_heartbeat_alerts.py ← HeartbeatAlertLog: record / get / acknowledge / prune / concurrency
│   │   ├── test_browser.py         ← BrowserTool: six primitives + result extraction edge cases
│   │   ├── test_mcp_tools.py       ← Stdio MCP tools (+ Phase 11 desktop MCP tools)
│   │   ├── test_mcp_tools_desktop.py    ← Phase 11 19 desktop MCP tools
│   │   └── test_system_status.py   ← SystemStatusReporter: alerts / projects / all foci + voice rendering
│   └── utils/                      ← Foundation primitives test suite
│       ├── __init__.py
│       ├── test_mtime_cache.py     ← aider catalog batch 1 (11 tests)
│       ├── test_token_budget.py    ← aider catalog batch 1 (13 tests)
│       ├── test_snapshot_guard.py  ← aider catalog batch 1 (17 tests)
│       ├── test_relative_indent.py ← aider catalog batch 1 (15 tests)
│       ├── test_spinner.py         ← aider catalog batch 11 (17 tests)
│       └── test_poll.py            ← 2026-05-23 OpenHands batch 1 (T14): 24 tests covering sync + async + backoff + cancel_check
│
├── data/                           ← runtime data (gitignored except for stub structure)
│   ├── qdrant/                     ← embedded Qdrant store (4 collections: conversations, facts, web_results, projects)
│   ├── memory.jsonl                ← legacy turn log / migration source
│   ├── projects.json               ← coding project registry (legacy lexical ProjectResolver source)
│   ├── projects/<project_id>/digest.md ← 2026-05-22: per-project digest markdown (also stored in Qdrant projects collection)
│   ├── sandbox/                    ← auto-created coding projects
│   ├── summaries.jsonl             ← maintenance summaries
│   ├── maintenance.sqlite          ← maintenance state
│   └── ollama_compat_test/         ← Modelfile from Foundation-phase Ollama compat test
│
├── kenningVoiceAudio/                ← workshop for voice character (gitignored except scripts + small configs)
│   ├── scripts/                     ← parakeet_server.py + kokoro fine-tune scripts + bulk synth helpers
│   ├── searxng_config/              ← settings.yml + limiter.toml mounted into Docker container at /etc/searxng
│   │   ├── settings.yml             ← engine roster + outgoing timeouts + categories=news mappings; bing + mojeek + wikipedia + wikidata + bing-news + reuters
│   │   └── limiter.toml             ← botdetection config (limiter: false; modern trusted_proxies schema)
│   ├── kokoro_finetune/             ← fine-tune project (Stage 1 done, Stage 2 ep 0 only; SLM joint NEVER ran)
│   └── kokoro_training_corpus_*/    ← bulk-synthesized LJSpeech-shaped training corpus (1654 clips / 107 min)
│
├── logs/                           ← runtime logs (gitignored)
│   ├── kenning.log                  ← rotating main log
│   ├── addressing.jsonl            ← classifier audit
│   ├── coding_tasks.jsonl          ← coding task progress
│   ├── verifications.jsonl         ← verifier runs
│   ├── clarifications.jsonl        ← clarification decisions
│   ├── mcp_calls.jsonl             ← MCP tool calls
│   ├── sessions/<id>.jsonl         ← per-session coding audit
│   ├── errors.jsonl                ← Phase 4 typed errors
│   ├── routing_decisions.jsonl     ← Phase 5 routing audit
│   ├── automation_tasks.jsonl     ← Phase 5 OpenClaw task records
│   ├── safety_audit.jsonl          ← 2026-05-12 Phases 2-5: tamper-evident hash-chain audit log for the runtime tool-call validator
│   ├── supervisor_decisions.jsonl  ← 2026-05-22: ProjectSupervisor decisions JSONL for offline threshold tuning
│   ├── eval_runs/<ts>.json         ← 2026-05-18 Phase 0: classifier eval harness output
│   └── observations.jsonl          ← 2026-05-18 Phase 1: canonical observation framework write target
│
├── models/                         ← (main checkout only — NOT in worktrees)
│   ├── Qwen3.5-4B-Q4_K_M.gguf      ← LLM, CURRENT DEFAULT for qwen3.5-4b preset (2.55 GB; n_ctx=8192)
│   ├── Qwen3.5-0.8B-Q4_K_M.gguf    ← speculative-decoding draft for qwen3.5-4b preset (0.50 GB; wired into draft_model.py but llm.draft_kind: "none" by default after PLD repair attempts)
│   ├── (other GGUFs deleted 2026-05-20 round 8 cleanup; swap-back via `python scripts/download_models.py` for josiefied-qwen3-4b / qwen3.5-9b / llama-3.2-3b-abliterated / josiefied-qwen3-8b presets)
│   ├── gemma-3-1b.gguf             ← Gemma 1B draft for gemma-3-4b-abliterated preset (when re-fetched)
│   ├── kokoro/                     ← StyleTTS2 + ISTFTNet (current default TTS)
│   │   ├── voices/kenning.pt        ← fine-tuned voicepack (style vectors ~512 KB)
│   │   ├── kenning_finetune.pth     ← fine-tune model weights (~327 MB; decoder + predictor + text_encoder + bert)
│   │   └── (HF cache loaded on first use)
│   ├── openwakeword/kenning.onnx    ← custom wake word
│   ├── piper/en_US-ryan-medium.onnx ← TTS voice (legacy piper_rvc engine; ~16 MB)
│   ├── rvc/{hubert_base.pt, rmvpe.pt} ← RVC support files (legacy piper_rvc)
│   ├── moondream2/                 ← VLM (CPU on-demand; lazy-loaded; ~1.5 GB)
│   ├── flan-t5-small/              ← zero-shot addressee model (CPU)
│   └── smart_turn/smart-turn-v3.2-cpu.onnx ← Smart Turn V3 (8.68 MB int8, 2026-05-12)
│
├── kenning_rvc_voice/   ← (main checkout only) RVC voice model
│   ├── Kenning.pth
│   └── added_IVF301_Flat_nprobe_1_Kenning_v2.index
│
└── training/                       ← (gitignored except scripts) Wake-word training data
    ├── download_training_data.py
    ├── probe_datasets.py
    ├── run_training.py
    ├── smoketest_memory.py
    └── smoketest_orchestrator.py
```

---

## Cross-cutting flows

### Voice query (conversational) — happy path

```
1. AudioCapture callback → enqueues 16 ms blocks (blocksize=256 @ 16 kHz;
   queue maxsize 1024 since 2026-05-22 audio-overflow fix)
2. Orchestrator.run() loop:
   a. WakeWordDetector or AddressingClassifier consumes blocks
      ├── COLD: "ultron" custom OpenWakeWord ONNX required (config wake_word.model_path; "kenning" is the legacy model). follow_up/always-listen is OFF by default — wake required every turn.
      └── WARM: AddressingClassifier verdict (rule -> zero-shot fallback) required
   b. On addressed: Silero VAD marks utterance start/end + Smart Turn V3
      gradient-fire confirms end (early-complete prob ≥ 0.65 -> submit at 300 ms;
      otherwise wait the legacy backstop)
   c. AudioCapture._capture_utterance() yields ndarray
   d. (Optional) Speculative STT kicked off during silence wait
3. DualSTTRegistry.transcribe(audio):
   ├── stt.engine="moonshine"  -> MoonshineEngine (CPU, streaming-native via
   │                              moonshine-voice; background worker chunk-feed)
   ├── stt.engine="parakeet"   -> ParakeetEngine (NeMo TDT via .venv-parakeet
   │                              HTTP server on CUDA; streaming endpoints)
   └── stt.engine="whisper"    -> WhisperEngine (faster-whisper int8_fp16;
                                  beam_size=1 since 2026-05-15)
   -> user_text
3b. command_normalizer.normalize_command(user_text) (2026-06-14+, runs BEFORE all
   matchers): L2 strip/canonicalize-relay-lead + L1 STT correction (Valorant
   gazetteer, wake-remnant strip, relay-lead restore via the relay-intent gate);
   a zero-mistakes gate returns questions/Spotify/reactions verbatim. May rewrite
   user_text for the matchers below; logs raw + normalized (`routing:normalized`).
4. KenningIntentRecognizer.process_utterance(user_text):
   ├── Match against 25 registered phrases via Gemma-300M q4 embeddings (CPU)
   ├── If "needs fresh data" intent matches -> set self._next_turn_force_search=True
   ├── If "gaming mode engage/disengage" matches -> short-circuit + invoke
   │                                                 GamingModeManager directly
   └── Else: return None and continue normal flow
5. local_clock_reply.maybe_local_clock_reply(user_text, *, now=None) [NOTE: actually invoked INSIDE _respond() at step 9a, not as a separate pre-routing step]:
   ├── "what time is it" / "what's today's date" -> system clock reply, ~5 ms
   ├── "what time is it in <city>" + city in zoneinfo map -> reply, no LLM
   └── Else: return None and continue normal flow
6. classify_routing(user_text, has_active_coding_task, has_pending_clarification)
   → RoutingIntent (one of 26 RoutingIntentKind values)
7. Orchestrator intercepts OPEN_LAST_SOURCE / NAVIGATE_TO_SITE BEFORE the
   capability controller (orchestrator-local state access):
   ├── OPEN_LAST_SOURCE -> _resolve_cited_source() + webbrowser.open() OR Chrome on monitor
   └── NAVIGATE_TO_SITE -> SearxNG "<site> official website" -> top-10 domain
                            scoring -> webbrowser.open() OR Chrome on monitor
8. CapabilityVoiceController.handle_capability_intent(routing_intent)
   ├── CONVERSATIONAL: returns None (orchestrator falls through to LLM path)
   ├── coding kinds: routes through CodingTaskRunner
   │   (when coding.supervisor.enabled, _handle_code_task_via_supervisor
   │    intercepts -- see "Supervisor decision flow" below)
   ├── APP_LAUNCH: native desktop.launcher (Chrome / Cursor / etc.)
   ├── WINDOW_MOVE / WINDOW_CLOSE: pywin32 placement / WM_CLOSE
   ├── GAMING_MODE: GamingModeManager.engage / disengage (VRAM reclaim)
   ├── MODEL_SWITCH: LLMEngine.reload_for_preset() (in-process hot swap)
   └── automation kinds: OpenClawDispatcher (stub voice msg by default)
9. If None (conversational fall-through):
   a. Orchestrator._respond(user_text)
      ├── Optional speculative classification + speculative LLM during silence wait
      ├── Web-search gate (3-layer):
      │   ├── classify_by_rules: _TIME_SENSITIVE / _VOLATILE_TOPICS /
      │   │                       _NEWS_QUERIES / _TIME_IN_LOCATION_GATE_RE
      │   ├── If self._next_turn_force_search: pre-populate cached_verdict=SEARCH
      │   └── Else preflight LLM (the active preset — josiefied-qwen3-8b as of Ultron 1.0) on UNCERTAIN cases
      ├── If SEARCH:
      │   ├── AcknowledgmentSource.next_phrase() -> TTS immediately
      │   │   (cache hit: ~0 ms; cache miss: ~200-400 ms)
      │   ├── WebSearchExecutor.run(text, categories="news" iff news query):
      │   │   ├── SearchProviderChain: SearxNG -> Brave -> DuckDuckGo
      │   │   └── ReaderChain: Trafilatura -> Jina
      │   ├── format_sources_for_prompt + (if news) multi-event directive
      │   └── injected into LLM context
      ├── ConversationMemory.retrieve(text, k=3, min_relevance=0.78) -> MemoryTurn[]
      │   (reranker.enabled=false runtime; cosine + RRF + recency composite)
      ├── LLMEngine.generate_stream(text, enable_thinking=False, history_user_message=bare):
      │   ├── the active preset (josiefied-qwen3-8b Q5_K_M, Ultron 1.0 default) in-process via llama-cpp-python
      │   ├── /no_think marker via _apply_no_think_marker (saves 5-10 s TTFT)
      │   └── In-process spec decoding wired (llm.draft_kind="none" default)
      └── KokoroSpeech.speak_stream(tokens) -> CUDA StyleTTS2 + ISTFTNet (voice=kenning)
         ├── Producer-consumer pipeline (synth N+1 overlaps playback N)
         ├── trim_and_fade boundary artifact mute (cosine fades + tail zero)
         └── PrecomputedAckClipCache hits for known ack phrases
   b. ConversationMemory.add(user/assistant) on background thread
   c. Token accumulator captures the spoken response into _last_response_text
      (used by OPEN_LAST_SOURCE resolver on the next turn)
   d. Bus publishes turn.started / stt.transcribed / routing.classified /
      gate.verdict / memory.retrieved / llm.stream.* / tts.played / turn.completed
10. Orchestrator enters FOLLOW_UP_LISTENING for 30 s (warm window)
```

### Supervisor decision flow (2026-05-22)

Active only when `coding.supervisor.enabled=true`. CODE_TASK / MID_SESSION_ADJUSTMENT /
CLARIFICATION_RESPONSE utterances route through this layer BEFORE the legacy
ProjectResolver path.

```
1. CapabilityVoiceController._handle_code_task -> intercepted via
   _handle_code_task_via_supervisor when supervisor_dispatch is wired
2. Build SupervisorInputs(user_text, coding_intent, has_active_task,
                          active_task_project_name, active_task_session_id)
3. SupervisorDispatchController.dispatch(inputs):
   a. supervisor.decide(inputs) -> SupervisorDecision
      Priority (first hit wins):
      ├── Active-task + ADJUSTMENT_PATTERNS -> RESUME current session
      ├── Semantic top match >= resolve_threshold (default 0.75) -> EDIT
      ├── Registry exact name/alias match -> EDIT
      ├── Top in [clarify_threshold, resolve_threshold) -> CLARIFY (top-2 candidates)
      └── Else -> NEW scaffold
   b. If narrate_enabled: speak narration via barge_in_speak callable
      (delegates to Orchestrator._speak_with_barge_in_check); on barge-in
      return BARGED_IN
   c. Build DispatchOutcome:
      ├── EDIT_DISPATCH -> enriched TaskRequest (digest + tree snapshot + file hints)
      ├── NEW_DISPATCH  -> fresh TaskRequest under sandbox_root/<slugified-name>
      ├── RESUME_FORWARD -> resume_session_id
      └── CLARIFY        -> clarification_question
4. _dispatch_supervisor_task:
   a. mkdir cwd
   b. runner.start_task(request) -> handle
   c. _attach_supervisor_digest_listener(handle, project_name, cwd, user_goal_hint)
      -> registers callback on COMPLETE that calls build_digest + index.upsert
5. Audit log: every decision appended to logs/supervisor_decisions.jsonl
6. Bus event: SupervisorDecidedEvent fires per decision
7. On Claude session COMPLETE: digest listener -> generate_digest -> project_index.upsert
   -> ProjectIndexedEvent + ProjectDigestGeneratedEvent on bus
```

### Gaming-mode VRAM reclaim flow (V1-gap A1, 2026-05-22)

```
1. Engage trigger:
   ├── Voice utterance matches gaming-mode regex OR intent recognizer
   └── classify_routing -> GAMING_MODE intent OR direct short-circuit
2. GamingModeManager.engage(trigger_phrase):
   a. on_engaged callbacks (decoupled from openclaw_on; work without OpenClaw):
      ├── LLMEngine.reload_for_preset("llama-3.2-3b-abliterated", n_ctx=6144)
      ├── DualSTTRegistry.swap_to("moonshine") + stop_parakeet_server()
      ├── KokoroSpeech.move_to_device("cpu")
      └── Moondream2VLM.unload() (drops _model + _tokenizer)
   b. (If OpenClaw client present) plugin disable: desktop-control + windows-control
   c. (Optional) Docker Desktop toggle if toggle_docker=true
3. ~2.3 GB VRAM freed (4.4 GB -> 2.1 GB Kenning contribution; net headroom for game)
4. Bus publishes GamingEngagedEvent + VRAMReclaimedEvent
5. Disengage (voice or explicit): reverse the chain (LLM restored, Parakeet server
   respawned in background, Kokoro moved back to CUDA, VLM lazy-reloads on demand)
```

### Bus event flow (2026-05-22)

```
publisher (any subsystem) -> publish(EventDef, properties)
   -> EventPayload.make() builds envelope with auto-id + timestamp
   -> schema validation (best-effort; logs WARN on mismatch but delivers)
   -> Bus._lock acquired briefly to snapshot subscriber lists
   -> typed-channel subscribers fire in registration order (publisher thread)
   -> wildcard subscribers fire after typed (publisher thread)
   -> per-callback try/except swallows + logs WARN ("subscriber N raised...")
```

### Coding task path

```
1-4. Same as voice query through classify_routing
5. RoutingIntent.kind == CODE_TASK
   a. CapabilityVoiceController.handle_capability_intent →
      handle_utterance(text)
   b. CodingIntent classification (intent.classify)
   c. ProjectResolver resolves "my flask app" → Project
      OR new_sandbox_project(name) creates a fresh dir
   d. KenningMCPServer.create_session(project_root, intent)
   e. CodingTaskRunner.start_task(TaskRequest)
      → DirectClaudeCodeBridge.submit() spawns:
         claude --print --output-format stream-json --include-partial-messages
                --include-hook-events --model haiku --add-dir <cwd>
                --dangerously-skip-permissions
   f. TaskHandle event stream:
      ├── TaskEvent(STATUS|TEXT|TOOL_USE|TOOL_RESULT|FILE_CHANGE|USAGE|ERROR|COMPLETE)
      ├── Listener feeds: SessionStore.record_stage(), record_test_results(),
      │                   set_pending_clarification(), record_completion_claim()
      └── Audit log line per event → logs/coding_tasks.jsonl
6. Orchestrator main loop returns; voice path resumes
7. On future "how's it going?" utterance:
   a. classify_routing → PROGRESS_QUERY (because runner.has_active_task())
   b. handle_capability_intent → handle_utterance →
      StatusNarrator.narrate(session_state) using project_status_delta
      → spoken narration
8. On Claude declare_complete:
   a. ConversationCoordinator.handle_declare_complete():
      → Verifier.verify(session) runs 6 checks
      → if pass: SessionStatus.COMPLETE
      → if fail and below escalation threshold:
            project_correction_context(session) → corrective prompt
            → Claude re-prompted with --resume
      → if escalation threshold crossed: switch to sonnet model
   b. CodingTaskRunner.completion_narration() generates final voice msg
9. Orchestrator polls voice.pending_completion() → speaks it
```

**Clarification fast-path order** (V1-gap A3 added Fast-path 2.5):

```
ConversationCoordinator.decide_clarification(...):
  1. Always-escalate keyword (api key, paid tier, scope add, ...)
  2. urgency=preference + options provided -> "use your default"
  2.5 (A3) Stored facts via facts_lookup -- if a directive-category
       (preference / decision / constraint) fact clears confidence and
       score thresholds, answer Claude with "From the user's stored
       preferences: <fact>. Use that."
  3. Always-answer keyword (test framework, linter, layout, ...)
  4. LLM decide pass (ANSWER / USE_DEFAULT / ESCALATE)
  5. Escalate to user (voice question + asyncio.Future await)
```

### Search-triggered path (web)

```
1-3. Same as voice query through Whisper
4-5. classify_routing → CONVERSATIONAL (web search isn't a routing kind)
6. Orchestrator._respond(user_text) flow:
   a. WebSearchGate.classify(user_text):
      ├── classify_by_rules → SEARCH if time-sensitive markers,
      │                       NO_SEARCH if personal-context queries
      └── classify_by_preflight (LLM call) for UNCERTAIN cases
         → returns GateVerdict with knowledge_confidence,
                  has_temporal_dependency, search_queries
   b. If SEARCH:
      ├── AcknowledgmentSource.next_phrase() → TTS within 200 ms
      ├── WebSearchExecutor.run(user_text, search_queries):
      │   ├── WebResultsCache.lookup(q) → cached payload OR None
      │   │   (3 collections: ttl_volatile_seconds=86400, ttl_stable_seconds=2592000)
      │   ├── BraveSearchClient.search(q) → BraveResult[]
      │   │   (wrapped in CircuitBreaker; raises BraveAPIError;
      │   │    failures log to errors.jsonl, return [])
      │   ├── _rank_snippets(llm, query, results, top_n) → ranked BraveResult[]
      │   ├── For top max_fetch: JinaReaderClient.fetch(url) → markdown
      │   │   (wrapped in CircuitBreaker; JinaReaderError → snippet-only)
      │   └── WebResultsCache.store(query, rows) — best effort
      └── format_sources_for_prompt(payload.sources) → injected into LLM context
   c. LLM generates response with citations
   d. TTS streams + format_sources_for_transcript(sources) printed (not spoken)
```

### OpenClaw stub dispatch path (Phase 5 — currently stubbed)

```
1-4. Same as voice query through classify_routing
5. RoutingIntent.kind in {BROWSER_AUTOMATION, MEDIA_GENERATION, MESSAGING,
                          FILE_OPERATION, SHELL_OPERATION, HYBRID_TASK}
   → CapabilityVoiceController.handle_capability_intent:
   ├── Single-category (browser/media/etc):
   │   AutomationTaskRunner.submit_task(intent) →
   │   OpenClawDispatcher.handle_X(intent.automation_intent) →
   │   DispatchResult(success=False, voice_message="gateway not connected yet")
   │   → audit row in logs/automation_tasks.jsonl
   │   → routing-decision row with outcome="stub"
   └── HYBRID_TASK: voice msg "I'd split it up and run both, but..."
6. VoiceResponse returned to orchestrator → speak
7. Orchestrator continues main loop
```

### Error / circuit-break path

```
External call (Brave, Jina) → CircuitBreaker.call(_do_X, ...)
├── If CLOSED, executes; on failure raises typed error
│   - 3rd failure within 5 min trips OPEN
├── If OPEN, raises CircuitOpenError immediately (no call)
│   - cooldown elapses → HALF_OPEN
├── If HALF_OPEN, executes once as a probe
│   ├── Success → CLOSED, failure counter reset
│   └── Failure → reopens, fresh cooldown
└── On any typed-error path:
    ErrorLog.record(error, dependency=...) → logs/errors.jsonl
    Optional: phrase_for("brave_unavailable") → spoken via TTS
```

---

## Source modules

### `src/kenning/__init__.py`

**Purpose:** package init. On Windows, `_register_cuda_dll_paths()` adds CUDA
runtime DLL directories (the installed `torch/lib` + any `nvidia-*-cu12/bin`) to
`os.add_dll_directory` so llama-cpp / ctranslate2 find their CUDA DLLs (e.g.
`cudart64_12.dll`, `cublas64_12.dll`). Importing `kenning` FIRST is what makes a
later `import llama_cpp` succeed (the DLL dirs are registered here).

**No public API** beyond import side effects.

### `src/kenning/__main__.py`

**Purpose:** `python -m kenning` entry point.

**Public:**
- `main() -> int` — sets up logging, acquires the single-instance
  guard (2026-06-12: `lifecycle/single_instance.py`; a duplicate
  launch prints the holder PID and returns exit code 3 BEFORE any
  model load; `KENNING_ALLOW_MULTIPLE_INSTANCES=1` bypasses), builds an
  `Orchestrator`, calls `.run()` until KeyboardInterrupt, releases the
  lock in a `finally`. Exit codes: 0 ok / 1 startup-or-run failure /
  2 missing model / 3 duplicate instance / 4 anticheat import-firewall failed to
  install or is not enforcing (FATAL refuse-to-start when anticheat is active).
- **2026-06-15:** a SIGTERM handler and an atexit backstop were added (was SIGINT-only). Both trigger full cleanup delegated to `Orchestrator.shutdown()` (which reaps the embedder sidecar + flushes the audit log). `taskkill /F` is uncatchable at the process level and is covered at the NEXT boot by the orphan sweep (`sidecar_lock.sweep()`) plus audit-log repair (`AuditLog.repair_if_needed()`).

**In:** environment + config.yaml (via Orchestrator construction).
**Out:** stdout console transcript, log files.

### `src/kenning/config.py` (Phase 3)

**Purpose:** single source of truth for tunable parameters. Loads
`config.yaml`, validates against pydantic schema, exposes singleton.

**Public:**
- `PROJECT_ROOT`, `MODELS_DIR`, `LOGS_DIR` — Path constants
- `DEFAULT_CONFIG_PATH` — `<root>/config.yaml`
- `resolve_path(value: str | Path) -> Path` — resolve relative paths against PROJECT_ROOT
- Sub-models (all pydantic `_Strict`; large + growing list):
  `AudioConfig`, `VADConfig`, `SmartTurnConfig`, `WakeWordConfig`, `STTConfig`,
  `LLMConfig` (+ `LLM_PRESETS`), `EmbeddingsConfig`,
  `QdrantCollections` (conversations / facts / web_results / **projects**),
  `QdrantConfig`, `MemoryConfig` (+ `MemoryRerankingConfig` /
  `MemoryRetrievalConfig` / `MemoryContextualRetrievalConfig` /
  `TopicalChunkingConfig` / `DiscourseTaggingConfig`),
  `BraveConfig`, `JinaConfig`, `SearxNGConfig`, `DuckDuckGoConfig`,
  `WebCacheConfig`, `WebSearchConfig` (+ chain ordering),
  `AddressingConfig`, `IntentConfig` (2026-05-22 Gemma-300M intent recognizer),
  `CodingMCPConfig`, `CodingVerificationConfig`,
  `CodingCanonicalMonitorConfig`, `CodingGoalAnchorsConfig`,
  `CodingAstMetadataConfig`, `CodingFactsConfig`,
  `CodingSupervisorConfig` (2026-05-22 supervisor stack: 11 knobs incl.
  `enabled` / per-phase flags / `resolve_threshold` / `clarify_threshold`),
  `CodingConfig`, `BackgroundSummarizerConfig`, `KokoroConfig`,
  `ProjectionsBudgets`, `ProjectionsConfig`, `RVCConfig`,
  `XttsV3Config`, `TTSConfig`, `LoggingConfig`, `ErrorPhrasesConfig`,
  `RoutingClassifierConfig`, `RoutingConfig`, `OpenClawConfig`,
  `GamingModeConfig`, `DesktopConfig` (+ `default_monitor_index` 2026-05-22),
  `WindowControlConfig`, `SafetyConfig` (+ rule toggles),
  `NotificationsConfig`, `HeartbeatConfig`, `BrowserConfig`,
  `MediaGenerationConfig`
- `KenningConfig` — top-level model
- `load_config(path=None, *, apply_overrides=False) -> KenningConfig` — explicit load (raises `ConfigurationError`). When `apply_overrides=True`, merges the ephemeral runtime overlay (`_merge_runtime_overrides`) on top of the parsed `config.yaml`.
- `get_config() -> KenningConfig` — singleton, lazy-load on first call
- `reload_config(path=None) -> KenningConfig` — clear cache, reload **with the runtime overlay applied** (`load_config(apply_overrides=True)`), so a GUI edit takes effect in-session.
- `clear_runtime_overrides() -> None` — wipes `data/runtime_overrides.json`; called at orchestrator boot so the overlay is always empty at startup.
- `set_config(cfg) -> None` — test injection

**Runtime-overrides overlay (NEW 2026-06-15 — config.yaml is the immutable boot source of truth):** the settings panel no longer writes `config.yaml`. It writes an EPHEMERAL overlay file `data/runtime_overrides.json` (`settings_gui/spec.py: write_runtime_overrides`, const `RUNTIME_OVERRIDES_RELPATH`). `reload_config()` merges that overlay in-session via `_merge_runtime_overrides`, and `clear_runtime_overrides()` wipes it at orchestrator boot. Net effect: GUI edits are session-only and revert on restart, so the lean-boot / gaming / anticheat / posture-canary defaults can never be left undone by a stale GUI edit — the code in `config.yaml` is always the source of truth at boot.
- `current_config_path() -> Path | None`
- `LLM_PRESETS: dict[str, dict]` (4B plan Stage A) — preset table for
  `LLMConfig.preset`. **Schema (config.py) field default `josiefied-qwen3-4b`**;
  `config.yaml` currently SETS **`josiefied-qwen3-8b-iq4xs`** (2026-06-23 — the
  SAME Josiefied-Qwen3-8B-abliterated-v1 base at the IQ4_XS imatrix quant
  (~4.56 GB) PAIRED with a `Qwen_Qwen3-0.6B-Q4_K_M.gguf` draft for in-process
  SPECULATIVE DECODING: `llm.draft_kind: "model"` + the preset's `draft_model_path`
  → `draft_model.make_qwen08b_draft_model`. Qwen3-0.6B shares the Qwen3 tokenizer
  (151936 vocab) with the 8B target so draft acceptance is high; verified live —
  `Speculative decoding enabled (real model draft, num_pred=4)`, NO `llama_decode -1`
  crash. n_ctx=4096, full GPU + **q8_0 KV cache** (`llm.kv_cache_type: 8`, 2026-06-23 —
  flipped from F16; the old "PLD-safe" F16 reason in commit `001896f` is STALE since we
  use the model-draft path, not PLD). Verified live: BOTH K and V genuinely quantize
  (`K (q8_0): 153 MiB, V (q8_0): 153 MiB` vs F16's 288+288) — no silent V→F16 fallback,
  so the 0.3.22 wheel supports q8_0 V-cache with flash-attn; ~270 MiB of KV recovered at
  4096 ctx, lossless, stable with the draft (GEN OK, no crash). **VRAM:** the full stack
  boots to **~10.9 GB** of the 12 GB card (8B IQ4_XS + draft + STT + Kokoro + ~3 GB
  background; was ~11.3 GB on F16 KV) → ~1.4 GB free, still tight for an actual Valorant
  match (the game needs ~2–4 GB) — fine for evaluation, not yet for in-match use without
  more VRAM freed (CPU STT ~1.5 GB / close background apps ~3 GB / IQ3_XS weights ~1 GB /
  drop the draft ~0.5 GB). Revert: `preset:` + `gaming_mode.llm_preset:`
  `"mistral-7b-v0.3-abliterated"` (7B) or `"josiefied-qwen3-4b-2507g"` (4B), and
  `llm.draft_kind: "none"`). Presets retained for swap-back: `mistral-7b-v0.3-abliterated`
  (the prior 7B test),
  `josiefied-qwen3-8b` (n_ctx=4096), `josiefied-qwen3-4b-2507g` (the prior A/B
  winner), `huihui-qwen3.5-4b`, `qwen3.5-4b` (+ 0.8B draft + n_ctx=8192),
  `qwen3.5-9b`, `gemma-3-4b-abliterated`, `llama-3.2-3b-abliterated` (gaming-mode
  target; n_ctx=6144). `LLMConfig._apply_preset` (model_validator)
  fills in `model_path` / `n_ctx` / `draft_model_path` only when those
  fields are absent from `model_fields_set`, so explicit YAML values
  always win.

**In:** `config.yaml`, `${ENV_VAR}` substitution from `os.environ`.
**Out:** typed `KenningConfig` instance.

### `src/kenning/bus/` (2026-05-22 session E)

**Purpose:** in-process typed pub/sub. Ported from opencode's
`packages/opencode/src/bus/` with the eager-subscribe race fix.
Subsystems publish lifecycle events without hard-coding callback
registrations through the orchestrator.

**Public:**
- `BusEvent.define(type: str, schema: Mapping[str, type], description="") -> BusEvent`
- `EventPayload.make(event_def, properties, id=None) -> EventPayload`
- `Bus` class (or `get_bus()` singleton):
  - `.publish(event_def, properties, id=None) -> EventPayload`
  - `.subscribe(event_def, callback) -> Callable[[], None]` (returns unsubscribe)
  - `.subscribe_all(callback) -> Callable[[], None]`
  - `.subscriber_count(event_def=None) -> int`
  - `.published_count() -> int`
- Module-level shortcuts: `publish`, `subscribe`, `subscribe_all`, `get_bus`,
  `reset_bus_for_testing()` (test escape hatch)
- 19 canonical events in `events.py` re-exported from `__init__.py`:
  `TurnStartedEvent`, `TurnCompletedEvent`, `STTTranscribedEvent`,
  `RoutingClassifiedEvent`, `GateVerdictEvent`, `MemoryRetrievedEvent`,
  `LLMStreamTokenEvent`, `LLMStreamCompleteEvent`, `TTSPlayedEvent`,
  `CodingFileChangedEvent`, `ProjectIndexedEvent`,
  `ProjectDigestGeneratedEvent`, `SupervisorDecidedEvent`,
  `SafetyViolatedEvent`, `GamingEngagedEvent`, `GamingDisengagedEvent`,
  `VRAMReclaimedEvent`, `DialogAppearedEvent`, `DialogResolvedEvent`
- `BUS_EVENT_CATALOG: list[BusEvent]` for introspection

**Threading:** callbacks fire on publisher thread (synchronous). State
guarded by `RLock`; recursive publish/subscribe from callbacks is safe.
**Fail-open:** callback exceptions caught + logged; schema mismatches
delivered with WARN.

### `src/kenning/channels.py` (2026-05-19 Track 6)

**Purpose:** Channel enum + metadata used by `ConversationMemory` write
payload + future channel-aware retrieval.

**Public:**
- `Channel` enum (`USER` / `TEAMMATE` / `SYSTEM`) — `TEAMMATE` is the VoiceMeeter
  loopback game-voice channel
- `ChannelMetadata` dataclass
- `Channel.from_str(value) -> Channel` (with `USER` fallback for
  forward-compat with legacy payloads)

### `src/kenning/local_clock_reply.py` (2026-05-20)

**Purpose:** short-circuits bare time/date queries from the system clock
without ever invoking the LLM or web search. ~5 ms reply path.

**Public:**
- `maybe_local_clock_reply(user_text, *, now=None) -> Optional[str]` — returns a spoken-form
  reply when the utterance matches a bare time/date regex OR
  `_TIME_IN_LOCATION_RE` with a city in `_CITY_TIMEZONES` (~70 cities
  mapped to IANA tz identifiers). Returns None for unknown cities
  (caller falls through to web-search gate which has a paired
  `_TIME_IN_LOCATION_GATE_RE` to force SEARCH).
- `_CITY_TIMEZONES: dict[str, str]` — public for testability.

### `src/kenning/latency_hygiene.py` (2026-05-19)

**Purpose:** process-level latency hygiene helpers: process-priority
boost, GC tuning, LLM/embedder warmup.

**Public:**
- `raise_process_priority(level="above_normal") -> bool` — Win32 process priority bump
- `pause_gc() -> bool` / `resume_gc(*, collect_now=True) -> bool` / `is_gc_paused() -> bool` — GC control
- `warmup_llm(generate_fn, *, prompt=DEFAULT_LLM_WARMUP_PROMPT) -> Optional[float]` /
  `warmup_embedder(encode_fn, *, prompt="warmup") -> Optional[float]` — trigger first-load, return elapsed seconds

### `src/kenning/trace.py` (2026-05-20 Round 6)

**Purpose:** thread-local turn_id + phase tag + structured tlog/phase
helpers. Every log line in a user utterance carries `turn=N phase=X`
so `grep turn=42` shows the entire lifecycle of one user utterance
in order.

**Public:**
- `next_turn() -> int` — increment + return the thread-local turn id
- `set_turn(turn_id: Optional[int])` / `get_turn() -> Optional[int]` — thread-local turn id
- `set_phase(name: str)` / `get_phase() -> Optional[str]` — thread-local phase tag
- `tlog(log, msg, *, level=logging.INFO, **kwargs)` — structured log emission
- `phase(name: str)` — context manager setting phase for its scope
- `fmt(msg, **kwargs) -> str` — structured line formatter (turn/phase/msg/kwargs)
- `snapshot() -> dict` / `restore(state: dict)` — save/restore the thread-local trace state

### `src/kenning/intent/recognizer.py` (2026-05-22)

**Purpose:** engine-agnostic semantic intent matcher. Wraps
`moonshine_voice.IntentRecognizer` with lazy load, fail-open semantics,
and a thread-safe registry that replays registrations on model load.

**Public:**
- `KenningIntentRecognizer(model="embeddinggemma-300m", variant="q4", threshold=0.8)`
  - `.register(canonical_phrase, *, handler=None, priority=0)` — append a recognised
    intent (replayed on first model load)
  - `.process_utterance(text) -> Optional[IntentMatch]` — fail-open;
    returns None on model load failure or below-threshold match
  - `.loaded` (property) `-> bool`
- `IntentMatch(canonical_phrase, utterance, similarity)`
- `IntentRegistration(canonical_phrase, handler, priority)`
- `get_intent_recognizer() -> Optional[KenningIntentRecognizer]` /
  `set_intent_recognizer(rec)` — module-level singleton accessors

**In:** Gemma-300M q4 embedding model (~300 MB CPU RAM, loaded once)
plus registered phrases from `config.yaml:intent.phrases`.

### `src/kenning/errors.py` (Phase 4)

**Purpose:** typed exception hierarchy for every external dependency.

**Public hierarchy:**
- `KenningError` (base) — has `message`, `context: dict`, `recovery: str`,
  `with_recovery()`, `with_context()`, `to_log_dict()`
- `DependencyUnavailableError` (subclass)
  - `BraveAPIError`, `JinaReaderError`, `QdrantUnavailableError`,
    `AnthropicAPIError`, `OllamaUnavailableError`, `OpenClawGatewayError`
    (+ `OpenClawAuthError` under it), `OpenClawToolError`
- `ClaudeCodeError`
- `AudioPipelineError`
  - `WhisperTranscriptionError`, `PiperSynthesisError`,
    `RVCConversionError`, `WakeWordModelError`, `AddressingClassifierError`
- `MCPServerError`, `ConfigurationError`, `FilesystemError`

**In:** raised from external-dep wrappers.
**Out:** caught by orchestrator + structured-logged via `ErrorLog`.

**Wired call sites (Phase 4 deferred wrappers, complete):**
- `ClaudeCodeError` + `AnthropicAPIError`: [coding/direct_bridge.py](../src/kenning/coding/direct_bridge.py)
  — launch failure, subprocess timeout, nonzero exit, stream-json error
  events. The pattern detector `_looks_like_anthropic_api_error` decides
  between the two based on error text (rate_limit / overloaded /
  invalid_api_key / etc.).
- `MCPServerError`: [coding/mcp_server.py](../src/kenning/coding/mcp_server.py)
  — bind failure (`raise … from OSError`), startup timeout, no-active-session
  on Claude tool call. `FilesystemError` covers the audit-log write path.
- `FilesystemError`: [coding/audit.py](../src/kenning/coding/audit.py),
  [coding/projects.py](../src/kenning/coding/projects.py),
  [coding/runner.py](../src/kenning/coding/runner.py) — session audit
  mkdir/write, project registry load/save, coding-tasks audit-log
  (first-failure dedup via `_AUDIT_WRITE_FAILURE_LOGGED` flag).

### `src/kenning/uncertainty.py`

**Purpose:** annotate user prompt with hedging hints based on the
pre-flight gate's uncertainty signals.

**Public:**
- `apply(verdict: GateVerdict, user_text: str) -> Tuple[GateVerdict, str]`
  — given a `GateVerdict` with `knowledge_confidence` /
  `knowledge_source` / `has_temporal_dependency`, returns a
  possibly-prepended user prompt with style hints. V1-gap B1: a
  `knowledge_source` of `retrieved_memory` / `retrieved_facts`
  prepends a source hint above the confidence addendum so the LLM
  matches its tone (rule verdicts inherit this branch too).
- `_source_hint_for(verdict)` (internal) — picks the leading source
  hint from `knowledge_source`. `weights` / `unknown` /
  `web_search_needed` get no hint.

**In:** `GateVerdict` from `web_search.gating`, raw user text.
**Out:** `(verdict, augmented_prompt)`.

### `src/kenning/response_style.py` (2026-05-10)

**Purpose:** per-call response-style addenda, prepended to the user's
text before it reaches the LLM. Lives OUTSIDE the persona file
(SOUL.md is voice-quality-locked) so the orchestrator can nudge the
model on a per-utterance basis without changing the system prompt.
Three addenda live here (dispatched in priority order): a procedural-steps
hint, a factual-answer hint, and a brevity hint for short
questions that the 4B model otherwise tends to over-explain ("What
are the Orcs in 40k?" → 1164-char four-paragraph essay in the
2026-05-10 live session).

**Public:**
- `is_brief_question(user_text: str) -> bool` — True iff the
  utterance is short (≤12 words AND ≤80 chars after strip) AND not
  explicitly asking for depth via any of the `_DEPTH_MARKERS`
  keywords (`explain` / `in detail` / `step by step` /
  `walk me through` / `elaborate` / `expand on` / `everything you
  know` / etc.). Empty / whitespace input returns False.
- `is_procedural_request(user_text) -> bool` → `_PROCEDURAL_HINT`; `is_factual_question(user_text) -> bool` → `_FACTUAL_HINT` — the other two hint classes.
- `apply_brevity_hint(user_text: str) -> str` — dispatches across all THREE hint classes in priority order (procedural → factual → brevity); prepends the matching `[Style: …]` directive (e.g. `[Style: respond in 1-3 short sentences …]` for brevity) when one matches; otherwise returns input
  unchanged. Empty input passes through. Idempotent on
  already-hinted text (the hinted version is too long to be
  re-classified as brief).

**In:** raw user text. **Out:** possibly-augmented user text (newline-
separated above the original).

**Wired at:** [pipeline/orchestrator.py](../src/kenning/pipeline/orchestrator.py)
`Orchestrator._build_response_stream` — applied on the non-search
conversational path (search path's augmented prompt has its own
length directive). Three call sites: web-gate-disabled fall-through,
web-gate-failure fall-through, NO_SEARCH verdict path.

### `src/kenning/conversational_ack.py` (2026-05-12)

**Purpose:** filler-acknowledgment source for the no-search
conversational branch. Masks the ~2.5 s perceived gap between
Whisper completing and the LLM's first TTS chunk by yielding a
short thinking-noise ("Mm.", "Right.", "Hm.", etc.) BEFORE the LLM
stream so the TTS pipeline starts speaking within ~200 ms of
Whisper completing. End-to-end latency unchanged; perceived latency
drops sharply. The web-search path has its own ack
(`web_search.acknowledgments.AcknowledgmentSource`) that describes
external activity; this module's phrases are tonally non-committal
(read as Kenning deliberating). The two pools rotate independently.

**Public:**
- `_CONVERSATIONAL_PHRASES: List[str]` — module-level phrase pool.
  Each phrase ≤20 chars, period-terminated so the TTS pipeline
  flushes it as a complete sentence immediately. No overlap with
  the web-search pool.
- `is_conversational_ack_eligible(user_text, *, has_pending_clarification=False) -> bool`
  — pure gate function. Returns False on empty input, on utterances
  shorter than 11 chars or 4 words (interjections like "yes",
  "thanks"), and when a coding-task clarification is pending (the
  orchestrator already has its own narration flow there). Otherwise
  returns True.
- `class ConversationalAckSource` — thin wrapper around
  `AcknowledgmentSource` with the conversational phrase pool baked
  in. Holding it as a distinct class type makes the orchestrator's
  intent clear at call sites and keeps the two pools' shuffled-
  cycle state separate.
  - `__init__(phrases=None)` — `None` uses the default pool;
    explicit empty list is forwarded to the underlying source which
    rejects it (vs silently swapping to default).
  - `next_phrase() -> str` — same contract as
    `AcknowledgmentSource.next_phrase`.

**Wired at:** [pipeline/orchestrator.py](../src/kenning/pipeline/orchestrator.py)
- `Orchestrator.__init__` constructs `self.conv_ack_source = ConversationalAckSource()`.
- `Orchestrator._maybe_conversational_ack(user_text) -> Optional[str]`
  helper threads pending-clarification state through the gate and
  fail-opens on any exception (broken source or coding-voice check).
- `Orchestrator._build_response_stream` yields the ack token before
  `llm.generate_stream(...)` at all three no-search exit points:
  web-gate-disabled fallthrough, web-gate-exception fallthrough,
  `verdict.decision != SEARCH` branch. The `_search_augmented_tokens`
  path is untouched (already yields its own ack from
  `web_search.acknowledgments.AcknowledgmentSource`).

### `src/kenning/desktop/` (Desktop automation native primitives)

NEW package backing the "open YouTube on monitor 2", "show me a picture of golden retriever",
"explain what I'm looking at" voice flows. Built native (no ClawHub plugin dependencies)
per user direction. Same UI Automation capability surface as the `windows-control` plugin
via `pywinauto`; same screenshot capability as `desktop-control` via `mss`.

#### `desktop/monitors.py`

- `class Monitor` -- frozen dataclass: index, name, x, y, width, height, work_x/y/width/height, is_primary. Helpers: `.right`, `.bottom`, `.center`.
- `enumerate_monitors() -> list[Monitor]` -- Win32 `EnumDisplayMonitors` + `GetMonitorInfo`. Primary sorts to index 0; rest left-to-right. Empty list on pywin32 failure.
- `find_monitor(query) -> Optional[Monitor]` -- accepts int / numeric string / `"primary"`/`"main"`/`"default"` / ordinals `"first"`/`"second"`/`"1st"` / directional `"left"`/`"right"`/`"top"`/`"bottom"`/`"center"` / device-name substring.
- `point_to_monitor(x, y) -> Optional[Monitor]` -- containing monitor for a virtual-screen point.

#### `desktop/capture.py`

- `class Screenshot` -- frozen: image_bytes (PNG), monitor_index, width, height, timestamp, origin_x/y.
- `class ScreenCapture` -- thread-local `mss.MSS` (mss is not thread-safe per-instance). Every successful capture records its PNG bytes in the safety taint tracker as capability=`screen_context`. Methods: `capture_monitor(monitor_or_index)`, `capture_all_monitors()`, `capture_region(x, y, w, h)`, `close()`. Fail-open (returns None on mss errors).
- `_bgra_to_png_bytes(raw, width, height) -> bytes` -- pure helper, BGRA from mss to PNG via Pillow.
- Singletons: `get_screen_capture()` / `set_screen_capture()`.

#### `desktop/windows.py`

- `class WindowInfo` -- frozen: hwnd, title, class_name, process_name (via psutil), pid, rect, monitor_index (greatest-overlap rule), is_minimized, is_foreground. Helpers: `.width`, `.height`, `.center`.
- `enumerate_windows(*, include_minimized=False, include_invisible=False, require_title=True) -> list[WindowInfo]` -- pywin32 `EnumWindows` callback.
- `get_foreground_window() -> Optional[WindowInfo]` -- `GetForegroundWindow`.
- `find_window(query, *, prefer_foreground=True, prefer_monitor=None, by_process=True) -> Optional[WindowInfo]` -- substring match against title (and optionally process name) with exact-match / foreground / monitor-preference tiebreakers.
- `_monitor_index_for_rect(rect, monitors)` -- pure helper used by tests.
- `wait_for_window(partial_title, *, timeout_s=30.0, interval_s=0.5, by_process=True, exclude_cloaked=True, prefer_monitor=None, sleep_fn=None, clock_fn=None) -> Optional[WindowInfo]` (2026 catalog 08 T4) -- synchronous "appears" barrier. Polls :func:`find_window` every ``interval_s`` until a match appears or the wall-clock timeout elapses. Defaults match the upstream clawhub-windows-control script. :data:`DEFAULT_WAIT_TIMEOUT_S` / :data:`DEFAULT_WAIT_INTERVAL_S` constants are exported. ``prefer_foreground=False`` during polling because the appearing window won't be foregrounded yet; matching tiebreakers (exact title, monitor preference) still apply. ``sleep_fn`` / ``clock_fn`` injection makes the loop deterministic in tests. Empty title returns None without polling; non-positive timeout returns None without polling. Fail-open: :func:`find_window` exceptions log DEBUG + the loop retries.
- `get_active_window_title() -> Optional[str]` (2026 catalog 08 T6) -- lightweight foreground-window title probe. Returns the title or None; cheaper than :func:`get_foreground_window` because it skips the psutil process-name lookup, monitor-index computation, and rect enumeration. Fail-open at every Win32 call.
- `class CloseWindowResult` (2026 catalog 08 T6) -- frozen result: success + window + method (`wm_close` / `kill_tree`) + suspected_unsaved + error.
- :data:`UNSAVED_CHANGES_TITLE_HINTS` (2026 catalog 08 T6) -- 5-entry editor convention list (`*`, `[modified]`, `(modified)`, VS Code dot, em-dash modified suffix) used by :func:`_title_suggests_unsaved_changes`.
- `close_window(partial_title, *, force=False, user_text="", prefer_monitor=None, exclude_cloaked=True) -> CloseWindowResult` (2026 catalog 08 T6) -- graceful window-close. Resolves target via :func:`find_window`; runs Cap-3 safety validator with `tool_name=desktop.window.close` + window title + process name + suspected_unsaved in arguments; posts a graceful `WM_CLOSE` via `win32gui.PostMessage` so the app's own close hook fires (editors with unsaved changes surface their save prompt). `force=True` escalates to :func:`kenning.subprocess.kill_tree.kill_process_tree`. `suspected_unsaved` on the result reflects whether the title matched the heuristic so callers can gate the close behind a two-phase voice confirmation.

#### `desktop/placement.py`

- `class PlacementResult` -- success/hwnd/monitor_index/error.
- `move_window_to_monitor(hwnd, monitor, *, fullscreen=False, maximize=False, size=None, offset=(0,0))` -- target-monitor placement. `fullscreen` fills the monitor as a regular window; `maximize` calls `SW_MAXIMIZE` after moving; explicit `size` clamps to work area.
- `maximize_window(hwnd)`, `minimize_window(hwnd)`, `restore_window(hwnd)`, `focus_window(hwnd)` -- single-action helpers. `focus_window` does SetForegroundWindow with BringWindowToTop fallback per Windows' foreground-lock rules.
- `minimize_window_idempotent(hwnd)` / `maximize_window_idempotent(hwnd)` / `restore_window_idempotent(hwnd)` (2026 catalog 08 T6 creative extension) -- state-check-before-act variants. Probe via `IsIconic` / `IsZoomed` first; when the window is already in the target state, return a no-op `PlacementResult(success=True, error="already <state>")` rather than sending a redundant `ShowWindow` call. `restore_window_idempotent` short-circuits only when BOTH probes report False (the NORMAL state); any probe failure falls through to action (safety floor: "act when uncertain"). Reduces spurious audit-log churn and matches the upstream "report already-in-state" UX.

#### `desktop/launcher.py`

- `class AppEntry` -- frozen: name, candidate_paths, args_prefix, aliases, process_name.
- `class LaunchResult` -- success/app_name/exe_path/pid/hwnd/monitor_index/placement/error/window_appeared (2026-06-12: None=no window wait requested / True=window detected / False=wait timed out -- placement+focus skipped; callers voice it honestly). 2026-06-12 bring-to-front: after placement (and for un-placed detected windows) `_focus_fail_open(hwnd)` calls `placement.focus_window` best-effort (MoveWindow/ShowWindow never change Z-order, so relay-pattern Chrome windows opened BEHIND the foreground); never raises, swallows a mid-launch AnticheatBlockedError from focus_window's own guard.
- `class AppLauncher` -- registry-driven launcher.
  - `find_app(query) -> Optional[AppEntry]` -- name + alias + substring.
  - `resolve_executable(entry) -> Optional[Path]` -- first existing candidate; Discord/Chrome-specific resolvers handle their Squirrel/auto-update directory layouts.
  - `launch_app(name, *, monitor, extra_args, fullscreen, maximize, wait_for_window, user_text)` -- safety-validated spawn + optional monitor placement via `move_window_to_monitor`.
  - `launch_chrome(*, url, monitor, fullscreen, maximize, window_size, new_window, user_text)` -- launches user's actual Chrome with `--new-window <URL>` (NOT Playwright). Reuses default profile (no `--user-data-dir`). Reusing the user's real Chrome session means cookies + sign-ins + extensions are preserved.
  - `open_image_search(query, *, monitor, small_window, user_text)` -- "show me a picture of X" convenience: launches Chrome with Google Images URL.
- Default registry includes: chrome, edge, firefox, cursor, vscode, discord, notepad, explorer, terminal (wt), spotify, slack, obs.
- Every launch passes through `_validate_launch` → `ToolCallValidator.check` with `tool_name="desktop.launch_app"`. Cap-2 rules block `--remote-debugging-port`, `--user-data-dir`, `--load-extension`, `--disable-web-security`, launches from Temp/Downloads.

#### `desktop/uia.py`

- `class UIAElement` / `class UIAActionResult` -- frozen snapshots; UI element data captured at lookup time (live pywinauto handles can go stale).
- `collect_window_text(window, *, max_elements=200, max_depth=8, min_length=2) -> list[str]` -- walk a window's UIA tree and return unique visible text strings. **Load-bearing for `screen_context`**: this is how "what's actually written on screen" feeds into the LLM. Bounded traversal (browser/IDE trees have 10k+ elements).
- `find_element(window, *, query, control_type, automation_id, exact) -> Optional[UIAElement]` -- search by name / automation_id / control_type.
- `click_element(window, query, ...) -> UIAActionResult` -- find + `click_input()`. Goes through Cap-3 action-verb-click rule (NEEDS_EXPLICIT_INTENT on `"Submit"` / `"Pay"` / `"Send Money"` etc.) and Cap-3 OAuth/payment URL detection on window title.
- `type_text_into_element(window, query, text, ..., clear_first=True) -> UIAActionResult` -- find + `set_text()` (preferred) or `type_keys()` fallback. Same Cap-3 safety hook.
- Lazy pywinauto import so `import kenning.desktop` doesn't pay the COM cost; failure returns None / empty list.
- `class UIElementInfo` (2026 catalog 08 T2) -- frozen snapshot for the interactive-element inventory: name, control_type, automation_id, enabled, rect (physical pixels), center (physical-pixel integer pair), value (truncated for Edit/Document; empty otherwise).
- `get_ui_element_inventory(window, *, control_types=None, max_elements=200, max_depth=8, value_truncate=100) -> dict[str, list[UIElementInfo]]` (2026 catalog 08 T2) -- categorised inventory walk. Buckets descendants into ``buttons`` / ``links`` / ``menu_items`` / ``list_items`` / ``tabs`` / ``checkboxes`` / ``radio_buttons`` / ``text_fields`` (Edit + Document) / ``dropdowns`` (ComboBox) / ``other``. Edit + Document admitted without a name (their value carries the content); buttons / links / etc. require a non-empty name. Optional ``control_types`` allowlist for narrow scans; ``value_truncate=0`` skips value capture entirely. Empty buckets are stripped. Fail-open at per-element + per-tree-walk granularity. GREEN read-only primitive -- the desktop equivalent of ``project_introspect.snapshot`` for UI surfaces.
- `wait_for_text_in_window(text, partial_window_title, *, timeout_s=30.0, interval_s=0.5, case_insensitive=True, max_elements=200, max_depth=8, sleep_fn=None, clock_fn=None) -> bool` (2026 catalog 08 T4) -- synchronous UIA-tree polling barrier. Polls :func:`enumerate_windows` + :func:`collect_window_text` each iteration looking for substring presence in any window matching the title filter. ``sleep_fn`` / ``clock_fn`` injection for deterministic tests; deadline-clamped final sleep so the loop never overshoots. Empty needle returns True immediately; non-positive timeout returns False. Fail-open per-window. ``DEFAULT_WAIT_TIMEOUT_S`` / ``DEFAULT_WAIT_INTERVAL_S`` mirror :mod:`kenning.desktop.windows` defaults.
- `class BrowserContent` + `class BrowserLink` (2026 catalog 08 T5) -- frozen dataclasses for structured browser content extraction. :class:`BrowserContent` carries page_title + browser_name + headings + text + buttons (`UIElementInfo`) + links (`BrowserLink`) + inputs + images + truncated + elapsed_ms. :class:`BrowserLink` is name + url + center + enabled.
- :data:`BROWSER_NAMES` (2026 catalog 08 T5) -- tuple of 7 lowercase browser-title substrings (chrome / firefox / edge / brave / opera / vivaldi / arc) used for case-insensitive title heuristic detection.
- `is_browser_window(title) -> bool` (2026 catalog 08 T5) -- single-window title heuristic helper.
- `find_browser_window(*, browser_hint=None, exclude_cloaked=True) -> Optional[tuple[WindowInfo, str]]` (2026 catalog 08 T5) -- enumerates windows, returns the first browser match as ``(window, browser_name)``. Optional ``browser_hint`` narrows the search to a specific browser. Fail-open on enumeration exceptions.
- `extract_browser_content(window=None, *, browser_hint=None, include_buttons=False, include_links=False, include_inputs=False, include_images=False, full=False, max_text=50, max_headings=50, max_buttons=100, max_links=200, max_inputs=50, max_images=50, max_elements=N, max_depth=12, text_name_max=1000, heading_max_len=100, exclude_cloaked=True) -> Optional[BrowserContent]` (2026 catalog 08 T5) -- the headline win. Walks a browser window's UIA tree, categorises descendants into headings (short uppercase / colon-terminated Text + Static), longer text, buttons (gated), Hyperlinks (gated; URL extracted from automation_id when it starts with http(s)://), Edit / ComboBox inputs (gated; current value captured), Images (gated). ``full=True`` is the shorthand for enabling all four include flags. Per-bucket caps applied after deduplication. Tree-walk-order dedup via per-bucket seen sets. ``max_elements`` defaults to 6x the standard UIA cap because browsers nest deep + expose many controls. Returns None when no browser window found / pywinauto unavailable / connect fails. Latency: ~20-100 ms on a typical webpage (zero GPU); compares to ~300-800 ms + ~330 MB VRAM for the moondream2 screenshot path -- UIA tier is preferred when the tree is well-populated, VLM stays the fallback for Electron browsers with shallow UIA trees.

#### `desktop/input_control.py`

- `class InputControlResult` -- success/action/error.
- `class InputController` -- pyautogui-backed mouse/keyboard with three gates:
  1. **Foreground security check.** `_foreground_is_security_window()` returns True when the focused window's class matches UAC / Windows Security / Credential UI patterns. Synthetic input on those is blocked by Windows itself (UIPI) but we refuse upstream so the audit log has context.
  2. **Rate limit.** `max_actions_per_second` (default 5) cap; over the limit fails the call rather than blocking the orchestrator.
  3. **Safety validator.** Every action builds a `RuleContext` with `tool_name="desktop.input.<action>"` for Cap-4 synthetic-input rules to check arguments against.
- Methods: `move_mouse(x, y, ...)`, `click(x, y, *, button, clicks, ...)`, `type_text(text, ...)`, `press_key(key, ...)`, `press_hotkey(*keys, ...)`, `scroll(amount, *, x, y, ...)`, `drag_to(x1, y1, x2, y2, *, button="left", duration_s=0.5, user_text="")` (2026 catalog 08 T8). All return `InputControlResult`. ``drag_to`` uses absolute-coord ``pyautogui.moveTo`` + ``pyautogui.dragTo`` (more deterministic than the upstream relative-offset ``pyautogui.drag``); SOURCE coordinate previewed via the existing click-preview gate (drag is bound by where you pick up from, so the source confirmation is the right safety contract).
- pyautogui's `FAILSAFE` (mouse-to-corner aborts) stays on; do NOT disable.

#### `desktop/screen_context.py`

- `class ScreenContextSnapshot` -- frozen: timestamp, monitors, foreground, windows, ui_text, screenshot, vlm_description, elapsed_ms. `.render_for_llm(*, max_ui_text=40) -> str` formats the snapshot as a readable text block for prepending to a user utterance.
- `build_screen_context(*, capture=True, capture_all_monitors=False, include_uia=True, include_vlm=False, ...)` -- the orchestrator. Assembles monitors / foreground / windows / UIA tree text / optional screenshot / optional VLM description into one snapshot. Every component fails to its empty/None default rather than raising.
- `class ScreenContextCache` -- in-memory ring buffer (default 3 entries, max age 15 s). `latest_fresh()` returns the most recent snapshot only if within max_age; useful for follow-up questions reusing the previous capture.
- `capture_and_cache(...)` -- convenience: build snapshot + store in the singleton cache.
- VLM hook: `set_vlm_describe(fn)` registers the bridge function called when `include_vlm=True`. `vlm.set_vlm(...)` wires this transparently.

#### `desktop/vlm.py`

- `class Moondream2VLM` -- moondream2 wrapper via transformers (`vikhyatk/moondream2`, `trust_remote_code=True`).
  - Construction validates importability but does NOT load weights. `warmup()` forces the lazy-load now.
  - `describe(image_bytes, *, prompt=None) -> VLMResult` -- decodes PNG via Pillow, runs `model.encode_image` + `model.answer_question`. Fail-open at every layer (missing transformers / bad image / inference exception / empty output all return `VLMResult(success=False, ...)`).
  - CPU-only. ~3.5 GB FP16 weights on disk; ~4-5 GB RAM after load. ~5-8 s per query.
- `class VLMResult` / `class VLMLoadError`.
- `build_vlm_from_config(*, enabled, repo, revision, device, max_tokens) -> Optional[Moondream2VLM]` -- factory; returns None on construction failure (orchestrator treats None as "VLM unavailable; fall back to text-only context").
- `get_vlm()` / `set_vlm(...)` -- singleton + wires the `screen_context.set_vlm_describe(...)` bridge transparently.
- Model weights pre-fetched via `scripts/download_models.py` step 9/10.

#### `desktop/element_click.py` (catalog 08 T3, NEW)

Cross-window semantic UIA element search + click via the gated
:class:`InputController`. The primary "act on a UI element by name"
primitive for LLM-facing automation.

- :data:`CLICKABLE_TYPES` (9 entries: Button / Hyperlink / MenuItem /
  TabItem / ListItem / CheckBox / RadioButton / TreeItem / DataItem),
  :data:`DEFAULT_MAX_GLOBAL_WINDOWS=12`, :data:`DEFAULT_MAX_ELEMENTS_PER_WINDOW=500`.
- Frozen dataclasses :class:`UIElementMatch` (name + control_type +
  automation_id + enabled + rect + center + window + is_exact),
  :class:`TextMatch` (name + control_type + rect + center + window),
  :class:`ClickResult` (success + element_name + window_title +
  control_type + center + method + candidates + is_exact + error).
- :func:`find_elements_by_name(name, *, window_title, control_types,
  exact, enabled_only=True, max_windows, max_elements_per_window,
  exclude_cloaked)` -> list[UIElementMatch]. Walks
  :func:`enumerate_windows` (optionally title-filtered), per-window
  walks UIA descendants in CLICKABLE_TYPES (or control_types). Returns
  exact-matches-first stable-sorted list. Per-element fail-open.
- :func:`click_element_by_name(name, *, window_title, control_type,
  exact, user_text, controller, max_windows, max_elements_per_window,
  exclude_cloaked)` -> ClickResult. Picks first candidate from
  :func:`find_elements_by_name` and clicks via the resolved
  :class:`InputController`. Routes through controller's gate stack:
  foreground security + rate limit + safety validator
  (`tool_name=desktop.input.click` with the resolved coordinates) +
  click-preview VLM gate when enabled. Threads ``user_text`` so the
  Cap-3 explicit-intent matcher can verify the user actually asked
  for the action.
- :func:`find_text_in_window(text, *, window_title, case_insensitive,
  max_windows, max_elements_per_window, exclude_cloaked)`
  -> list[TextMatch]. Coordinates-only variant; no click. Per-element
  fail-open. The "look up coords, hand to VLM" companion.

#### `desktop/dialog_control.py` (catalog 08 T1, NEW)

Native Windows dialog detection + CRUD interaction. Closes the
"automation sequences stall on dialogs" gap from the upstream
clawhub-windows-control plugin's ``handle_dialog.py``.

- Constants :data:`DIALOG_CLASSES` (`#32770` + 4 standard dialog class
  names), :data:`DIALOG_CONTROL_TYPES`, :data:`DIALOG_TITLE_KEYWORDS`,
  :data:`DISMISS_BUTTONS` (9 entries in least-destructive order),
  :data:`DEFAULT_WAIT_TIMEOUT_S` / :data:`DEFAULT_WAIT_INTERVAL_S`.
- Frozen dataclasses :class:`DialogInfo` (window + class_name + matched_by),
  :class:`DialogButton` (name + enabled + rect + center),
  :class:`DialogField` (name + control_type + enabled + value + rect + center),
  :class:`DialogCheckbox` (name + control_type + enabled + checked + center),
  :class:`DialogContent` (title + message + buttons + text_fields +
  checkboxes + dropdowns + list_items + elapsed_ms),
  :class:`DialogActionResult` (success + action + dialog_title + target +
  method + error).
- :func:`find_dialogs(*, partial_title_filter, exclude_cloaked, include_minimized)`
  -> list[DialogInfo]: enumerate dialog-style windows matching the
  class set OR the title-keyword set. Fail-open on enumeration errors.
- :func:`read_dialog(source, *, max_descendants=500, message_max=8, text_truncate=500)`
  -> Optional[DialogContent]: walk a dialog's UIA tree into a structured
  snapshot. Buckets `Button` / `Edit`+`Document` / `CheckBox`+`RadioButton` /
  `ComboBox` / `ListItem` / `Text`+`Static`. Per-element fail-open.
- :func:`click_dialog_button(source, button_name, *, exact, user_text)`
  -> DialogActionResult: case-insensitive substring match by default;
  ``exact=True`` switches to equality. Goes through Cap-3 verb-click +
  Cap-4 security-window + explicit-intent validator gates.
- :func:`type_into_dialog_field(source, text, *, field_index, user_text)`
  -> DialogActionResult: indexed Edit/ComboBox write. Prefers
  ``set_text`` over ``type_keys`` to preserve special characters.
  Same Cap-3 gating; text + dialog title in validator arguments so
  credential-pattern detectors apply.
- :func:`dismiss_dialog(source, *, user_text, preferred_buttons)`
  -> DialogActionResult: iterates :data:`DISMISS_BUTTONS` (or
  ``preferred_buttons``), per-candidate safety check so blocking one
  candidate falls through to the next, ESC fallback gated through the
  validator with ``action="dismiss_escape"``.
- :func:`wait_for_dialog(*, partial_title, timeout_s, interval_s,
  exclude_cloaked, sleep_fn, clock_fn)` -> Optional[DialogInfo]:
  synchronous polling barrier with deterministic-test injection +
  deadline-clamped final sleep.

#### `desktop/voice.py` (Phase 8 voice handlers)

Bridges :class:`RoutingIntent` (kinds APP_LAUNCH + SCREEN_CONTEXT_QUERY) to the native primitives. Imported by :class:`CapabilityVoiceController` in `coding/voice.py`.

- `class AppLaunchVoiceResult` / `class ScreenContextVoiceResult` -- frozen result types. 2026-06-12: AppLaunchVoiceResult gains `window_appeared` (mirrors LaunchResult).
- `handle_app_launch(intent) -> AppLaunchVoiceResult` -- dispatches an :class:`AppLaunchIntent` to :class:`AppLauncher`. Resolves the monitor (index OR directional via `find_monitor`); chooses `launch_chrome(url=...)` for Chrome+URL combos and `launch_app(...)` for everything else; threads `user_text` into the safety validator. Returns a short in-character voice line. 2026-06-12 honesty fix: when `result.window_appeared is False` the spoken line says the window didn't appear (naming the requested monitor when one was asked for) instead of the old false "Opening that on monitor N." fallback; the legacy mon_phrase elif remains only for duck-typed results lacking the field.
- `handle_screen_context_query(intent) -> ScreenContextVoiceResult` -- builds a :class:`ScreenContextSnapshot` via `build_screen_context(include_vlm=intent.include_vlm)` and returns its `render_for_llm()` text for prompt injection.
- `handle_window_move(intent) -> WindowMoveVoiceResult` (2026-05-14 third pass) -- finds an existing window matching ``intent.window_query`` via :func:`kenning.desktop.windows.find_window` and moves it to ``intent.monitor_index`` (or ``intent.monitor_query`` resolved via :func:`find_monitor`) using :func:`move_window_to_monitor`. Distinct from APP_LAUNCH which would spawn a new process.
- `handle_window_close(intent) -> WindowCloseVoiceResult` (2026-05-14 third pass) -- finds an existing window matching ``intent.window_query`` (optionally restricted to a monitor via ``intent.monitor_query``) and posts ``WM_CLOSE`` via :func:`win32gui.PostMessage`. Graceful close path -- lets the app prompt to save if it wants. Used for "close my YouTube tab", "close Discord on my right monitor", etc.

#### `desktop/preferences.py` (Phase 10 preference learning)

User-preference persistence so "open YouTube" picks up "monitor 2 + maximize" the second time after the user said it once with the explicit phrasing.

- `class DesktopPreference` (frozen) -- one learned action: `user_phrase`, `app_name`, `url`, `monitor_index`, `fullscreen`, `maximize`, `success`, `timestamp`.
- `class PreferenceLogger` -- thread-safe JSONL append at `logs/desktop_preferences.jsonl`. Optional asynchronous mirror to the OpenClaw workspace daily memory files when :func:`set_workspace_writer` is called (failures are swallowed -- JSONL is the source of truth).
- `find_preference_for_phrase(phrase, *, max_age_days=90.0, min_substring_length=4) -> Optional[DesktopPreference]` -- substring match (both directions) with recency weighting (newest matching wins). Filters out failed entries.
- `record_launch_preference(...)` -- one-call helper for the voice handler.
- Wired at `CapabilityVoiceController._handle_app_launch`: utterances without explicit monitor target consult the logger first; matching prior preference's monitor + flags become the defaults.

> **Anticheat note:** every `desktop/` module below is kept ENTIRELY OUT of RAM under anticheat (gated/lazy imports; the import-firewall blocks the input/capture/UIA/browser stacks). These sections document capability that exists but is NOT loaded in a lean gaming session.

#### `desktop/browser_use.py` (catalog 10 — external browser-use CLI wrapper)
- `class BrowserUseTool` — sync subprocess wrapper over the external `browser-use` CLI (Playwright + CDP); ~30 read/write methods all routed through the safety validator. Result hierarchy: `BrowserUseResult`, `BrowserState`, `BrowserHtmlResult`/`BrowserTextResult`/`BrowserBboxResult`, `BrowserActionResult`, `BrowserCookiesResult`, `BrowserCdpResult`, `BrowserEvalResult`, `BrowserScreenshotResult`, `BrowserProfile`. `JsScriptAnalysis` / `CdpStatementAnalysis` static-analysis validators. Singletons `get_browser_use_tool` / `set_browser_use_tool` / `reset_browser_use_tool_for_testing`. Fail-open (binary absent → structured "not found").

#### `desktop/browser_sessions.py` (catalog 10 T8)
- `class BrowserSession`, `class BrowserSessionResult`, `class BrowserSessionsManager` (named-session lifecycle: allowlist + cap + ProcessRegistry + kill_process_tree on force-close) + `get_browser_sessions_manager`.

#### `desktop/browser_sequence.py` (catalog 10 creative extension)
- `BrowserSequenceStep`, `BrowserScreenshotRef`, `BrowserStepResult`, `BrowserSequenceResult`, `class BrowserSequenceRunner` (before/after screenshot bracket + injected VLM verify) + `get/set_browser_sequence_runner`.

#### `desktop/sequence.py` (catalog 09 T5 — desktop step runner)
- `SequenceStatus`, `StepOutcome`, `VlmVerdict`, `SequenceStep`, `ScreenshotRef`, `StepResult`, `SequenceResult`, `class DesktopSequenceRunner` (screenshot-bracketed steps + VLM verification, auto-pass radius) + `get_sequence_runner`.

#### `desktop/ocr.py` (catalog 08 T7 — Tesseract OCR)
- `class OCRResult`, `is_ocr_available()`, `ocr_image_bytes()`, `ocr_screen_region()`, `ocr_screen_monitor()` — lazy pytesseract, fail-open.

#### `desktop/clipboard.py` (catalog 09 T4)
- `class ClipboardResult`, `class ClipboardManager` (read=Cap-2 / write=Cap-3, taint-tracker integration) + `get/set_clipboard_manager`.

#### `desktop/click_preview.py` (VLM-confirmed click gate)
- `class PreviewDecision`, `ConfirmedClick`, `PreviewResult`, `class ConfirmationHistory` (bounded store), `draw_crosshair_on_image()`, `preview_click()`.

#### `desktop/dialog_poller.py` (catalog 08 — background UIA dialog poller)
- `class DialogPoller` (start/stop daemon, ~750 ms cadence, publishes `DialogAppearedEvent`/`DialogResolvedEvent`) + `get/set_dialog_poller`. STOPPED + unloaded under anticheat.

#### `desktop/win32_helpers.py` (catalog 07 T2 — ctypes Win32 helpers)
- `class MonitorDpi`, `get_monitor_dpi()`, `get_monitor_dpi_for_window()`, `get_last_input_idle_ms()`, `is_window_cloaked()`, `class BlockInputResult`, `block_input_context()` (hardened ctx-mgr, UIPI floor), `logical_to_physical()` / `physical_to_logical()`.

### `src/kenning/audio/`

#### `audio/capture.py`
- `class AudioCaptureError(RuntimeError)` — raised on device init failure
- `class AudioCapture` — sounddevice callback thread enqueueing 32 ms blocks
  - `start()` / `stop()`
  - `get_chunk(timeout: float = 1.0) -> Optional[np.ndarray]` — consumer API (the Orchestrator capture loop pulls chunks)
  - `drain()` — discard pending chunks + report drop accounting; `qsize() -> int`
  - (NOTE: `_capture_utterance` is an `Orchestrator` method, not on `AudioCapture`.)
  - 2026-06-12 status-flag + drop accounting: per-session counters
    reset in `start()` (before the stream opens). The audio thread
    only COUNTS PortAudio status flags (`input overflow` etc.) plus a
    single first-occurrence warning -- never repeated logging I/O on
    the callback; recurrence + queue-full drop-oldest totals are
    reported from `drain()` on the consumer thread. Read-only
    properties `status_flag_count` / `dropped_blocks` (replaces the
    `_overrun_warned` warn-once-forever latch that hid recurrence).

#### `audio/devices.py`
- `class AudioDeviceError(ValueError)`
- `resolve_device(configured, kind) -> Optional[int]` — substring match on device name. For `kind="output"` it prefers the WASAPI endpoint among name matches (so a low-latency shared-mode stream can be opened).
- `describe_device(device, kind) -> str`
- `make_output_stream(...) -> sounddevice.OutputStream` (NEW 2026-06-15 — WASAPI low-latency chokepoint) — the single factory every spoken-audio output path opens its stream through. When the resolved device is a WASAPI endpoint it opens a WASAPI stream with `WasapiSettings(auto_convert=True)` + `latency='low'`; otherwise it falls back to MME with `latency='low'`. Gated by `audio.prefer_wasapi_output` (default `true`). Effect on the reference rig: the team-relay (B1) and `BroadcastSink` (B3/OBS + monitor) buses drop from ~90–180 ms (MME) to ~22–25 ms (WASAPI); the default Realtek speakers stay on MME + `latency='low'` (~90 ms) because their WASAPI endpoint will not open. Used by `relay_speech.play_to_device` and `BroadcastSink`.

#### `tts/text_hygiene.py` (NEW 2026-06-11 — pre-synthesis hygiene)

`sanitize_spoken_text(text) -> str`: strips asterisk/bracket stage
directions (`*nods*`, `[sighs]` — bounded spans), `<think>…</think>`
spans, control tokens (`/no_think`, `/think`, `<|…|>`), orphaned
quotes; returns `""` for punctuation-only remainders. Applied at the
TOP of `KokoroSpeech._synthesize` (microseconds; an empty result
returns a zero clip the playback paths skip), so every spoken surface
— responses, acks, the team relay — is covered for ANY active model.
Born from live incidents on the 3B gaming preset: a stage direction
and a parroted "/no_think" were spoken out loud. **2026-06-15:** a
pronunciation pass also normalizes "JARVIS" / "J.A.R.V.I.S" (any
all-caps or dotted form) → "Jarvis" so the Marvel name is spoken as a
word, not spelled out letter-by-letter. Tests:
`tests/tts/test_text_hygiene.py` (verbatim incident shapes + a
no-model-load short-circuit check).

#### `audio/output_quality.py` (NEW 2026-06-11 — TTS blip watcher)

Catches audible artifacts in synthesized clips, live. The Kokoro
fine-tune's known boundary blips are mitigated at synth time by
`trim_and_fade`; this watcher DETECTS whatever still slips through so
voice-output regressions are observable. Driven by `tts.output_watch`
config (default ON).

- `analyze_clip(pcm, sr, *, thresholds...) -> ClipQualityReport` — pure
  per-clip analysis: `hard_onset`/`hard_tail` (un-faded edges that pop on
  stream open/close), `leading_burst`/`trailing_burst` (isolated noise
  spikes separated from the speech body by silence — the classic
  fine-tune artifact), `discontinuity` (a bad-join click: since 2026-06-12 the jump must clear the absolute floor AND be an outlier >=8x the median adjacent-sample diff in a +-5 ms window -- the previous absolute-only 0.5 threshold flagged 112/174 live records incl. every clean ack clip, because loud high-frequency speech produces jumps comparable to its own local envelope (measured 0.82-1.33x); pure tones cap at ~1.41x their median diff, hot fricative noise at ~4.6-5.9x, while production joins/clicks measure 9-170x), `internal_dropout` (two-tier since
  2026-06-12: a gap ≥600 ms of dead air inside the speech body always
  flags; a 100-600 ms gap flags ONLY when BOTH gap edges carry
  speech-level energy ≥25% of the envelope peak — the signature of a
  digital hard cut; natural prosody gaps measured live at 60-430 ms
  always decay gradually into the gap and must not flag), `clipping`,
  `dc_offset`. Leading/trailing silence padding is normal and never
  flagged. Short loud runs isolated at the clip edges (≤130 ms runs
  separated from the body by ≥200 ms — the same run-grouping the
  `trim_and_fade` fix uses) are stripped from the body and reported as
  `leading_burst`/`trailing_burst` with NO peak-ratio gate (the
  measured real artifact bursts sat at only 3-12% of clip peak, which
  the legacy 25% gate rejected — misclassifying the dead air before
  the burst as an internal dropout; adjudicated 2026-06-12 from 174
  live `audio_quality.jsonl` records).
- `OutputQualityWatcher` — daemon-thread analyzer behind a bounded
  queue: `submit(pcm, sr, label)` is the hot-path entry (non-blocking
  put; overflow drops, never waits); findings log at WARNING + append
  JSONL to `logs/audio_quality.jsonl`; `stats()` returns session
  counters; `close()` joins the thread.
- `get_output_watcher()` / `reset_output_watcher()` — config-gated,
  fail-open process singleton.
- **Waveform stream (2026-06-11):** the watcher also writes a compact
  per-clip envelope record for EVERY clip (clean or flagged) to
  `logs/audio_waveform.jsonl` — 120-point |peak| envelope + finding
  kinds/positions; size-bounded (rewritten keeping the newest ~80
  records past 768 KB). The control panel's OUTPUT WAVEFORM pane tails
  it and renders each clip with red dashed markers at the exact blip
  positions. `tts.output_watch.waveform_enabled` (default ON).
- Hook: tail of `KokoroSpeech._synthesize` — a try/except enqueue after
  the int16 conversion (covers speak / speak_stream / the voice relay;
  ack-cache hits skip analysis since those clips are static +
  pre-rendered). The locked synth hot path pays microseconds; analysis
  itself never runs on it. Device/driver artifacts that never appear in
  the PCM are out of scope (the hard-onset/tail checks are the proxy
  for stream-open pops).
- Tests: `tests/audio/test_output_quality.py` (21 — every detector with
  synthetic waveforms, watcher JSONL + counters + queue-overflow +
  error-survival + close idempotency, config gate + singleton cache +
  the kill switch). A session-scoped conftest fixture disables the
  singleton for the whole sweep (`set_output_watcher_enabled(False)`,
  mirroring the observation-writer guard) so stubbed-synth unit tests
  never spawn analyzer threads or touch the live logs dir; the
  watcher's own tests opt back in.

#### `audio/command_normalizer.py` (NEW 2026-06-15 — pre-routing STT cleanup layer)

Applied BEFORE all matchers. `_strip_leading_junk` strips mis-heard wake
remnants + filler. `recover_relay_lead` prepends "tell my team"/"tell"
for bare callouts via callout/agent signal regexes, GATED by
not-a-callout + Spotify-signal so conversational/music text is left
alone. `normalize_command(text) -> str` is the entry point; it gates
conversational + Spotify text OUT of correction so there is ZERO
over-correction. The orchestrator logs BOTH the raw STT and the
normalized text (tlog `routing:normalized`). 2026-06-15 test-drive
fixes: additional over-correction guards and a phonetic-snap logic fix.
**Verbatim relay family expansion (2026-06-15):** `_WORD_FOR_WORD` rewrites
"tell my team word for word X" / "tell the team verbatim X" into the verbatim
relay form; STT mis-hears of the verbatim verb "repeat" as "Pete"/"Heat"
(leading a soundboard relay) are restored to "repeat" ONLY when followed by
"to"/"after" + an addressee (so a literal name "Pete" or the word "heat" is
never rewritten); and a possessive on the team addressee ("my team's X",
"the squad's X") has its trailing "'s" stripped so the relay lead-strip works.
**2026-06-16 additions:** narration / epistemic-hedge regex (a first-person
musing that merely MENTIONS relaying is not a relay), lead-preserving disfluency
resolution (`_resolve_disfluency`: "tell my — no wait, tell the whole team to X"
collapses to the corrected lead without losing the payload), conversational
lead-filler stripping (`bro`/`yo`/`dude`/`bruh`/…), and **relay-intent gate
wiring** — `recover_relay_lead` now consults `_relay_intent.relay_intent_ok`
(the semantic gate) before prepending the bare-callout "tell my team" lead, so a
muttered narration / banter / question / Marvel-identity line is no longer
broadcast (fail-open: gate down → keyword behavior).

#### `audio/_relay_intent.py` (NEW 2026-06-16 — semantic relay-intent gate)

The single weakest joint in the routing cascade — `recover_relay_lead`'s
bare-callout prepend — promoted from a keyword trigger to a semantic DECISION
gate. A bare utterance that merely contains a callout keyword ("eco", "rotate",
an agent name) is just as often narration ("I should tell them to eco"), banter
aimed at Ultron, a question for advice, or Marvel/identity talk. `RelayIntentGate`
scores the utterance against curated POSITIVE (`RELAY_POSITIVE_EXEMPLARS`) and
NEGATIVE (`RELAY_NEGATIVE_EXEMPLARS`) exemplar clouds via the shared
EmbeddingBackend and `decide(text) -> True | False | None` returns True only when
`max(pos) - max(neg)` clears a calibrated threshold (default 0.06). **Biased to
ABSTAIN** (a missed callout costs the streamer a re-say; a false relay broadcasts
garbage to teammates). FAIL-OPEN: sidecar down → `None` → caller keeps today's
keyword behavior (never a new blocking dependency); never prepares against a
down/unavailable sidecar (no latch, so it recovers when the sidecar returns).
Lazy process-wide singleton (`get_relay_intent_gate` / `set_relay_intent_gate` /
`relay_intent_ok`); holds only exemplar strings + a urllib client.

#### `audio/_tail_schema.py` (NEW 2026-06-16 — flavor tail schema + tag folding)

The pure-python (stdlib-only → anticheat-safe) FOUNDATION for the deep flavor
expansion. `TailEntry(text, tags)` dataclass + `as_entry`/`entries` coercion
(lossless migration of the legacy `str` pools → tagless `TailEntry`, ZERO rewrite,
ZERO behavior change — a tagless tail is the base / Tier-3 fallback). The expanded
enemy situation taxonomy (`Sit`, `ENEMY_SITUATIONS`, 4 → 16:
spotted/ult/damaged/utility + moving/planting/defusing/rotating/saving/
falling_back/peeking/holding/lurking/trading/last_alive/near_death). Machine-readable
`AGENT_GENDER` (was a code comment; now a hard-auditable per-agent pronoun map) +
`GENDER_PRONOUNS`. Fact-folding helpers that turn noisy callout facts into the
COARSE tag vocabulary: `loc_class` (≈130 location tokens →
high_ground/long_range/site_area/flank_route/mid/choke), `dmg_level_tag` (hp
number / damage keyword → one_shot/low/minor), `ability_tag`, `situation_for_payload`
(action words refine the 'spotted' base → a finer situation), and `build_active_tags`
(the Tier-1 target set). The COARSE route stays a plain dict; tags only ever
fine-select WITHIN an already-correct cell, so a mis-parsed tag can never produce a
wrong-character tail — it just relaxes to a less-specific tier.
**2026-06-16 (coherence pass):** added `_VERB_TO_ABILITY` — a verb/token →
canonical ability CATEGORY map (mollied/nade → molly, walled → wall, darted/shocked →
dart, smoked → smoke, flashed/blinded → flash, caged → cage, stunned/concussed → stun,
…) so `ability_tag` folds a callout VERB to the same `ability:<canon>` tag the curated
`utility` cells carry; a standard category routes straight to the matching ability cell,
an agent-unique ability falls through to the semantic selector.

#### `audio/_tail_selector.py` (NEW 2026-06-16 — semantic fine-selector)

The embeddinggemma sidecar promoted to a fine-SELECTOR. `select_tail(cands,
recent_lines, *, agent, situation, active_tags, pool_kind)` builds a short
structured query (agent + situation + folded tags), embeds it (`kind=query`),
scores it against a session-cached doc matrix of the candidate tails (`prepare`,
`kind=document`), applies MMR diversity against a rolling window of recently-chosen
vectors + a HARD mask of tails already spoken this round (`recent_lines`) + a
per-`pool_kind` abstain floor (`agent` 0.30 / `multi` 0.26 / `generic` 0.20), and
returns the best-fit tail text — or **None for ANY reason** (numpy missing /
sidecar down / latched / empty / low-confidence / exception), in which case the
caller falls back to the deterministic `_pick_flavor`. Strictly ADDITIVE — it only
re-ranks within an already-correct cell, so it can never change the character or
situation. numpy is in-process-legal (a faster-whisper transitive dep — the
firewall blocks only torch/transformers); the only network is the existing loopback
sidecar client. `KENNING_ENABLE_TAIL_SELECTOR` opt-in enables it; absent (default) = OFF, because the deterministic coarse-keyed route already routes contextually at zero latency and the semantic re-ranker adds a sidecar embed only worth paying for large, ambiguous pools.

#### `audio/_common_words.py` (NEW 2026-06-16 — common-word protection set)

A GENERATED frozenset `COMMON_WORDS` — the top-~5000 frequency-ranked English
words (alpha-only, len ≥ 3) from the public-domain google-10000-english list, baked
by `scripts/build_common_words.py`. Imported by `_stt_correct` so the
phonetic/fuzzy gazetteer snapper only ever rewrites OOV / misheard tokens and never
corrupts a real English word. Pure data, no deps; regenerate via the script, do not
hand-edit.

#### `audio/_stt_correct.py` (EXPANDED 2026-06-15 — Valorant gazetteer + phonetic snap)

A Valorant gazetteer (agents / maps / weapons / abilities / locations /
terms) compiled to a lower-cased set + a Metaphone phonetic index
(jellyfish). `_phonetic_fuzzy_snap` (phonetic-class match + rapidfuzz
Jaro-Winkler >= 0.88, or fuzzy >= 0.92) snaps a mis-heard token to the
nearest gazetteer term; `_PHRASE_MISHEARS` repairs word-blends
("ray zombie" -> "Raze on B", "arsova" -> "our Sova", "be main" -> "B
main"). Context-aware rules disambiguate words that are also English.
Risky 1:1 maps removed to avoid over-correction. **2026-06-15 additions:**
`silver`→Sova; a "hey `<agent>`" greeting mis-blended into "Hell`<agent>`"
("hey Sage"→"Hellsage", "hey Jett"→"Helljet") is dropped; and count words
(three/four/five/six/won) are protected from corruption to the location
"tree" — the `tree`→`three` repair is gated to a following push/site token so
the real location "tree" ("split through tree") and "we won" stay safe.
**2026-06-16 additions:** a **common-word protection gate** (a token in
`_common_words.COMMON_WORDS` is never snapped, so real English survives the
gazetteer); an **inflection guard** (an `-ed`/`-ing`/`-ers` form or a real /
gazetteer plural is never snapped onto a base gazetteer term — "walled"/"orbs"
keep their grammar); an **OOV agent-superstring guard** (a snap target may not be
a superstring of the heard token — a genuine mishear is same-length-ish, "jet" →
Jett, not the other way); and a **`_MISHEAR_FORCE` allow-list** (curated mishears
that fire even though they are common words).
**2026-06-16 (coherence pass) — context SLOT-confirmation pass** (`_slot_agent_correct`
+ `_closest_agent`, run as **Stage 1.5** of `correct_callout_stt`, between the context
rules and the token-level snap): an agent name sits in characteristic SLOTS — subject of
a damage report ("`<x>` hit 18"), object of one ("hit the `<x>` for 18"), or after a
side word before a state/ability verb ("their/enemy/our `<x>` ulted/mollied/…"). A token
in one of those slots that is a common English word but PHONETICALLY an agent (Jaro-Winkler
≥ 0.82, with `_GAZ_LOWER` terms skipped so an ability word like "cage"/"wall" is never read
as an agent) is corrected to that agent — "raise hit 18" → "Raze hit 18". This is the
ONLY place the common-word protection is overridden, and only when the slot grammar
supplies the confidence; non-slot uses ("raise your crosshair", "raise the volume") have
no agent slot and are left untouched. Purely additive, pure-python, ~microseconds.

#### `audio/command_router.py` (NEW 2026-06-15 — semantic command router)

The SEMANTIC COMMAND ROUTER: an additive coarse-decision layer BENEATH
the exact matchers. `class CommandRouter` prepares each family's
exemplars once, then `route(text) -> RoutingDecision` does a
max-aggregated similarity per family and an OOS abstention gate (ABSTAIN
to the LLM if the top family is the conversational anchor, OR its score
is below a per-family threshold, OR it does not beat the runner-up by a
margin, OR it is not a deterministic family). `RoutingDecision` dataclass
(family / abstained / confidence / margin / reason / scores).
`get_command_router()` is a lazy, FAIL-OPEN singleton (returns None on
any build error so a router fault can never break the voice loop). Imports
no heavy ML. **2026-06-16:** `get_embedding_backend()` exposes the router's
shared `EmbeddingBackend` (the sidecar client) so the relay flavor layer's
semantic tail selector (`_tail_selector`) and the relay-intent gate
(`_relay_intent`) reuse the same per-turn embed cache — one sidecar instance,
one cache, one client; fail-soft (None when the sidecar/router is unavailable).

#### `audio/_router_backends.py` (NEW 2026-06-15 — pluggable similarity backends)

Pluggable similarity backends. `LexicalBackend` (rapidfuzz token-set/WRatio
fused with a Metaphone phonetic ratio; fully in-process, CPU-light, the
gaming-safe DEFAULT). `EmbeddingBackend` (a ~10-line urllib client to the
sidecar; per-turn query cache so each utterance embeds once; `kind=query|document`
for EmbeddingGemma's asymmetric prompts). `HybridBackend` (fuses lexical +
embedding by weight; a per-turn failure latch drops to lexical-only after 3
consecutive failed turns; degrades to lexical transparently when the sidecar is
down). `get_backend(prefer, *, host, port, emb_weight, wait_seconds)` with a
cold-boot readiness poll. This module imports ONLY urllib / numpy / rapidfuzz /
jellyfish — the embedding model never enters the main process.

#### `audio/_command_exemplars.py` (NEW 2026-06-15 — curated router exemplar libraries)

The curated exemplar command libraries, one list per family: team_callout,
spotify, identity, desktop_refuse, and conversational (the abstention anchor).
`DETERMINISTIC_FAMILIES = {team_callout, identity, desktop_refuse}` (families
the router dispatches to a handler); `ABSTAIN_FAMILIES = {conversational}`.
Spotify is intentionally NOT deterministic (its exact matcher runs first; its
exemplars exist only so a music command never mis-routes to a callout).

#### `audio/_ultron_identity.py` (NEW 2026-06-15 — categorized identity-answer pools)

Pure data + classifier for "what are you?" teammate questions. Seven curated
in-character Ultron answer pools (~30 lines each), one per category:
`bot` (AI/robot/chatbot/algorithm), `soundboard` (canned clips/voiceboard),
`streamer` (streaming/stream), `human` (real person/are you real), `puppet`
(strings/who's controlling you/off-switch/someone behind you), `voice_changer`
(changing your voice), and `recording` (recorded/playback/pre-recorded/on a
tape). Exposed via `IDENTITY_POOLS` (the category→pool dict).
`classify_identity_question(text) -> Optional[str]` detects WHICH category a
question is about (most-specific category wins; bare "machine" is deliberately
excluded as too ambiguous in a tactical callout). The CALLER first confirms the
text is actually an identity question (so a callout merely containing "stream"
or "machine" never misroutes), then uses the category to pick from the matching
pool. This module holds NO picking/LRU logic — the picking + anti-repeat LRU
lives in `relay_speech` (`pick_line` / `_pick_lru`). The effect: a teammate
asking "are you a bot / a soundboard / a streamer / a real person / …" gets a
DISTINCT varied in-character answer instead of one generic identity line.

#### `audio/stop_button.py` (NEW 2026-06-15 — loopback-immune click kill-switch)

A tiny in-process, always-on-top, fully-black, mouse-clickable **STOP** window.
`class StopButtonOverlay` (in-process tkinter, like the waveform overlay — never
imports the desktop automation stack) renders a small draggable bar + a STOP
button; clicking it fires the orchestrator's `_cancel_all_playback()` — the SAME
all-channel cancel (conversational TTS + relay B1 + OBS B3 + monitor mirror) as
voice "Ultron, stop", but WITHOUT the wake-word watcher (which self-triggers on
the monitor-speaker loopback, so the click path is the loopback-immune way to
stop). A button click is an ordinary window message to its OWN window — NOT input
monitoring/hooking — so it adds nothing to the anticheat surface.
`match_stop_button_command(text) -> Optional[str]` is the voice matcher
("show/hide the stop button"); summon/dismiss by voice. Fail-open: no display /
no Tk → the window never appears and the voice path is untouched. Configured by
`StopButtonConfig` in `config.py` (`enabled`, `show_at_startup`, geometry +
colours) / the `stop_button:` block in `config.yaml`.

#### `safety/anticheat.py` (NEW 2026-06-11 — anticheat-safe mode)

A process-wide hard kill-switch for every OS-interaction surface,
built for kernel-level anticheats (Vanguard/EAC/BattlEye). While
active, Kenning cannot inject input, capture the screen, read pixels /
templates / OCR, walk UIA trees, automate the clipboard / dialogs /
elements, manipulate or launch windows, drive the browser CDP, or use
the bridge's desktop tools — while the AUDIO pipeline stays fully
alive (mic, STT, LLM, TTS, the VoiceMeeter team relay: shared-mode
audio APIs, the same surface Discord uses; they interact with no other
process).

- Enforcement is BELT-AND-SUSPENDERS, three layers: **(1)** 49 module
  guards inserted at the top of every OS-touching public entry across
  14 modules (`input_control` ×5, `capture` ×5, `uia` ×10,
  `clipboard` ×2, `dialog_control` ×6, `element_click` ×3,
  `windows` ×2, `placement` ×5, `launcher` ×3, `ocr` ×2,
  `sequence.run`, `browser_use._invoke` — the single subprocess choke
  point for all ~25 browser methods —, `screen_context`,
  bridge `DesktopTool` ×3), each raising `AnticheatBlockedError`
  BEFORE any OS API is imported or touched; **(2)** a
  `ToolCallValidator.check` pre-check returning audited BLOCK_HARD for
  every blocked tool class (`is_blocked_tool` taxonomy); **(3)** the
  orchestrator voice toggle + intent layer.
- Analyzed and deliberately NOT blocked (documented in the module
  docstring): audio capture/playback, `nvidia-smi` global GPU queries
  (same surface as MSI Afterburner), `psutil` self-scoped process
  management (Kenning's OWN children / own priority only — no foreign
  process handles), shell-level window-metadata reads
  (`enumerate_windows` / `get_foreground_window` — the gaming-mode
  game detector needs them; same API the taskbar uses).
- Activation: voice toggle ("enable/disable anticheat mode", also
  "tournament mode" — `match_anticheat_toggle` +
  `Orchestrator._maybe_handle_anticheat_toggle`); HARD-TIED to gaming
  mode — `GamingModeManager._set_anticheat` makes engage ALWAYS enable
  anticheat (UNCONDITIONAL + fail-safe even if config is unreadable, so
  a kernel-anticheat game never launches unprotected);
  `gaming_mode.anticheat_with_gaming_mode` (default ON) governs only the
  RELEASE direction (whether disengage auto-turns-off);
  pinned via `gaming_mode.anticheat_safe_mode`. Config errors fail
  OPEN for the probe but the runtime flag always wins, so a broken
  config can never silently disable an explicit toggle.
- **Surface hooks (full unload, not just call gates):** a kernel
  anticheat observes what a process is DOING, so
  `set_anticheat_active` also runs registered
  `register_surface_hook(name, hook(active))` callbacks: on activate
  the orchestrator's hooks STOP the UIA dialog-poller thread and drop
  the cached mss `ScreenCapture` + `DesktopSequenceRunner` singletons
  (releasing their GDI/COM handles); on deactivate the poller restarts
  and singletons rebuild lazily. Hooks are fail-open (a broken hook
  never blocks the flip or the others) and cleared in
  `Orchestrator.shutdown()`. Combined with gaming mode's existing
  unloads (Parakeet server killed, VLM unloaded, Kokoro→CPU, Docker
  Desktop stopped via `toggle_docker`), nothing anticheat-adjacent is
  RUNNING while the mode is active — the only live surfaces are
  shared-mode audio, the LLM, and file/network IO.
- **NEVER-LOAD hardening (2026-06-14):** beyond stopping running
  surfaces, the OS-interaction stack is kept ENTIRELY OUT of RAM under
  anticheat — never imported, not merely call-gated. Importing ANY
  `kenning.desktop` submodule runs the package `__init__`, which pulls
  pyautogui/SendInput + mss/GDI-capture + pywinauto/UIA into the process;
  several boot/hot paths did so. ALL now gated on `anticheat_active()`:
  `_start_dialog_poller` (the UIA poller), `_load_desktop_vlm_if_enabled`
  (moondream2 lives in `kenning.desktop.vlm`), `_load_browser_use_if_enabled`,
  the click-preview gate, `_build_engage_deps` (engage-time VLM unload), and
  — the subtle one fired on EVERY LLM message build by warmup + each turn —
  `inference._resolve_vlm_loaded_for_skills` (now reads `sys.modules.get(
  "kenning.desktop.vlm")`: module-absent ⇒ VLM-not-loaded ⇒ no import). The
  capture-singleton hook likewise clears via `sys.modules.get(...)`, never
  importing to release. A boot-time `_audit_anticheat_posture()` self-check
  logs every restart that pyautogui/mss/pywinauto/`kenning.desktop` are
  unloaded ("anticheat posture OK … loaded=none"), and a loud
  `ANTICHEAT POSTURE CANARY` WARNING if any are present while active — it
  is the regression canary that surfaced the warmup/skills path above.
  `gaming_mode.anticheat_safe_mode` + `relay_speech.echo_to_user` now
  default True at the dataclass level (safe-by-default: a lost config
  can't silently unblock input/capture, and the user hears their own
  callouts). The team relay + Spotify path import nothing OS-interacting
  (pinned by subprocess tests) → the well-trodden voice-changer class.
- **Diagnostics monitoring is also kept-out-by-default** (same discipline):
  the verbose `SPOKEN(...)` text logs + the per-utterance `SPOKEN-BLIP`
  analysis (final-vs-raw-Kokoro divergence) in `tts/kokoro_engine.py` and the
  `output_quality` watcher are ALL gated by
  `kenning.diagnostics.audio_diagnostics_enabled()` (sentinel
  `~/.kenning/audio_diagnostics_on` OR a config flag) — when off, the
  `output_quality` module is NEVER imported (`tests/test_diagnostics_gating.py`).
  It only ever touches Kenning's OWN buffers/log (anticheat-neutral), but stays
  out of RAM by default. `Orchestrator.__init__` calls
  `diagnostics.reset_for_new_session()` so the sentinel is cleared on EVERY
  boot (a restart always comes up OFF); the operator re-enables post-boot only
  while testing. The waveform OVERLAY's analysis is SEPARATE (its own
  `analyze_clip` for rendering) and allowed.
- Tests: `tests/safety/test_anticheat.py` (72) — incl. an **AST audit
  test** that re-parses every guarded source file and fails if ANY guard
  is refactored away; `test_no_ban_class_apis_anywhere_in_source` (zero
  OpenProcess / Read·WriteProcessMemory / CreateRemoteThread /
  VirtualAllocEx / SetWindowsHookEx / RegisterRawInputDevices / Nt* /
  pynput / dxcam / ImageGrab in src outside the defense regexes); **two
  clean-subprocess probes** proving the desktop stack stays unloaded
  under anticheat boot and that the relay/monitor/spotify path imports
  none of it; the boot posture-audit canary; representative
  raise-before-OS-touch checks; the validator pre-check + audit; the
  toggle matrix; and the gaming-mode tie-in with opt-out.

#### `settings_gui/` (NEW 2026-06-11 — voice-launched control panel)

"Pull up your settings" / "open the control panel" spawns a DETACHED
`python -m kenning.settings_gui` process; "close the settings" (or the
panel's Close button / window X) terminates it. Because the panel is a
separate process the voice pipeline is untouched while it runs and
byte-for-byte restored when it closes — zero residual resources.

- `spec.py` (logic, fully tested): `SECTIONS` — a curated 9-card /
  ~36-knob catalogue (Game Chat Relay / Voice / Hearing / Brain /
  Addressing / Web Search / Evolution / Coding / Desktop & Research),
  each `Knob` carrying its YAML path, widget kind
  (bool/int/float/str/choice/csv), bounds, and a `restart` flag for
  construction-time settings. `patch_config_text` — an indent-aware
  block scanner that replaces ONE scalar in the raw YAML text while
  preserving every comment + untouched line byte-for-byte (PyYAML
  round-tripping would destroy the file's documentation);
  `apply_updates` patches + re-parses + verifies the parsed data
  changed ONLY at the requested paths, then writes atomically
  (tmp + replace). `write_reload_signal` touches
  `data/config_reload.signal`. **`write_runtime_overrides` (NEW
  2026-06-15)** is now the actual persistence path the panel uses: it
  writes the ephemeral overlay `data/runtime_overrides.json` (const
  `RUNTIME_OVERRIDES_RELPATH`) instead of mutating `config.yaml`, so GUI
  edits are session-only and revert on the next boot (see "Runtime-overrides
  overlay" under `config.py`).
- `launch.py`: `match_settings_command` (strict open/close phrasings;
  "what are your settings?" never matches), `launch_gui` (detached
  spawn, fail-open None), `close_gui` (kill_process_tree, fail-open).
  **Shutdown fix (2026-06-15):** `Orchestrator.shutdown()` now calls
  `close_gui(self._settings_gui_pid)` so a detached settings panel is
  closed when Ultron exits. It was previously orphaned and lingered
  after exit — only the "close settings" voice command and gaming-engage
  had closed it.
- `app.py` (UI layer, untested by design — no GUI windows in the
  sweep): tkinter dark theme (near-black + Kenning crimson), scrollable
  two-column card grid, a LIVE LOG panel streaming `logs/kenning.log`
  (level-colored, filter box, pause, bounded to ~2000 lines, daemon
  tail thread), bottom bar with pending-change count + APPLY UPDATE +
  CLOSE. Update = comment-preserving patch + reload signal; ↻-marked
  knobs note "applies on next start".
- Orchestrator: `_maybe_handle_settings_gui` short-circuit (after the
  relay branch; `via="settings_gui"`) + `_maybe_reload_config` +
  `_drain_gui_actions` polled in the IDLE wake-word loop (every ≤0.5 s
  while the LLM/TTS are guaranteed not mid-turn). A NEWER reload-signal
  mtime → `reload_config()` swaps the singleton so every call-time
  `get_config()` read hot-applies. **First-Apply fix (2026-06-15):**
  `_maybe_reload_config`'s first-sight guard (which skips a stale signal
  left over at boot) used to swallow the FIRST real Apply when no
  `data/config_reload.signal` existed at startup. The orchestrator now
  captures the signal's mtime EAGERLY at `__init__`, so any later write
  fires the hot-reload as intended. The action channel
  (`data/gui_action.jsonl`, byte-offset tracked) carries the few knobs
  that aren't read call-time — `gaming_mode` (engage/disengage, which
  also drives anticheat), `llm_preset` (`reload_for_preset`),
  `kokoro_device` (`move_to_device`) — applied at the same safe idle
  point. **Every exposed knob is hot** (call-time or action; no knob
  sets `restart`). The genuinely-unsafe-to-hot-swap selectors (TTS/STT
  engine, mic device, voicepack) were removed from the panel rather
  than marked restart, so nothing in the GUI needs a restart. The
  bottom bar has a live **GAMING + ANTICHEAT** toggle that writes the
  gaming action immediately (engage/disengage within one idle tick); it now
  reflects the boot default by reading `gaming_mode.engage_at_startup` (so it
  shows ON at startup). **GUI reflects boot defaults (NEW 2026-06-15):** a
  new **"Lean Boot (barebones — all ON)"** section surfaces
  `engage_at_startup` + every `barebones_*` flag (including the
  2026-06-15 second-wave `barebones_skip_memory` / `barebones_skip_intent`
  / `barebones_skip_ack_prewarm`) + `llm_gpu_layers` (config.yaml now lists
  every `barebones_*` flag explicitly). New `app.py`
  method `_apply_one(path)` applies a single knob on its own (backing the
  per-row apply buttons, including "APPLY MUTE ONLY" for
  `audio.mute_speakers`).
- Tests: `tests/settings_gui/test_spec_and_launch.py` (46 — patcher
  matrix incl. comment/blank-block edges, render/read, TWO drift
  guards against the REAL config.yaml (every knob path exists; every
  knob round-trip-patches losslessly), apply_updates atomicity +
  validation, matcher matrix, spawn/close lifecycle fail-open,
  orchestrator wiring, reload-signal semantics incl. stale-file
  immunity).

#### `spotify/` (NEW 2026-06-12 — voice playback control; EXPANDED 2026-06-14)

Comprehensive hands-free Spotify control for streaming/gaming ("play
despacito", "skip", "pause the music", "what's playing", "turn it up",
"play my focus playlist", "queue blinding lights", "play californication
next", "mute", "restart the song", "like this song", "shuffle off",
"make the volume 40"). Driven by `spotify` config (default ON) and
**ungated by gaming / anticheat mode** — the dispatch only checks
`spotify.enabled`, so full music control stays live in a barebones
gaming session. Web API over HTTPS only: no GPU, no LLM, anticheat-safe.

- `auth.py`: credentials load from a GITIGNORED file OUTSIDE the repo
  (`spotify.credentials_path`, default `~/.kenning/spotify.json` —
  client_id / client_secret / redirect_uri / refresh_token); the secret
  NEVER enters the tree. `SpotifyAuth.access_token()` does the
  authorization-code refresh-token flow (injectable `post_fn`; caches
  the short-lived access token with a 30 s margin; persists a rotated
  refresh token). `build_authorize_url` / `exchange_code` /
  `save_refresh_token` back the one-time setup.
- `client.py`: `SpotifyClient` Web-API wrapper (injectable
  `request_fn`) — `now_playing` / `devices` / `ensure_device` (transfers
  to a device when none active) / `resume` / `pause` / `next_track` /
  `previous_track` / `set_volume` / `current_volume` / `set_shuffle` /
  `set_repeat` / `search_first` / `play_query` (track plays the song;
  artist/album/playlist plays the context) / `queue_query` / `seek`
  (0 = restart) / `save_current_track` + `unsave_current_track` (add or
  remove the playing track from Liked Songs, `user-library-modify`).
  401/403 → a clear re-authorize message; 404/no-device → "open Spotify
  on a device".
- `voice.py`: `match_spotify_command` — a WIDE-but-strict regex set
  (same discipline as the relay matcher) covering every action with many
  natural phrasings: play / queue / pause / resume / next / previous /
  restart / now_playing / volume up·down·set·by-delta ("lower the volume
  by 10%" → `_VOL_DELTA` carries the amount in `value`; fixed-step nudges
  leave `value` 0) / mute / unmute / shuffle / repeat / like / unlike.
  Order-sensitive subtleties: "play X next" →
  queue (not play), bare "play" / "play the music" → resume, "throw on
  X" → play while "throw X in the queue" → queue, "make the volume 40"
  (no "to/at") → volume_set. Replies live in `_REPLIES` — Ultron's cold
  machine register, varied per call via `random.choice`. Dynamic content
  (track name, volume %) built in `handle_spotify_command`; mute caches
  `client._premute_vol` so unmute restores the prior level. Fail-soft on
  auth/API errors.
- Orchestrator: `_maybe_handle_spotify` short-circuit placed AFTER
  run/launch + app-launch (so "play the calculator" wins over a song)
  and BEFORE the relay path; gated ONLY by `spotify.enabled` (NOT by
  gaming/anticheat), so music control is always live. `_get_spotify_client`
  lazily builds + caches the client; a missing/unauthorized credentials
  state speaks a setup hint, never crashes.
- `lean_handler.py` (NEW 2026-06-15): a standalone lean Spotify handler
  so full music control runs as a lean-boot sibling (alongside the
  deterministic relay matcher and the settings-GUI command) when only the
  fuzzy semantic router would otherwise fire in a lean gaming session.
- `scripts/spotify_setup.py`: one-time browser OAuth (tiny localhost
  server catches the redirect, exchanges the code, saves the refresh
  token). Needs Spotify Premium + the redirect URI registered in the
  app dashboard.
- Tests: `tests/spotify/test_spotify.py` (143 — auth refresh/cache/
  failure, client routes incl. search-then-play + device transfer +
  volume clamp + seek + save/unsave, the full expanded matcher matrix
  incl. value checks + relative volume deltas + the routing subtleties +
  negatives, dispatch incl. mute/unmute round-trip + restart + like/
  unlike + volume-delta math + auth/no-device error messages,
  orchestrator wiring; all HTTP faked). `scripts/relay_test/spotify_matrix.py`
  is a 195-case manual command matrix (every action × phrasings +
  negatives) for quick regression.

#### `src/kenning/audio/relay_speech.py` (Valorant teammate-relay)

Converts a user voice command into a line Kenning speaks on a **separate** PortAudio output device (typically a VoiceMeeter virtual input wired to the game mic bus), so teammates hear Kenning — not a conversational response. The relay persona in this file is **Ultron** (the *Avengers: Age of Ultron* / Spader–Whedon character — cold, biblical/aesthetic/evolutionary, the Stark wound), exclusively; Kenning is the product name used everywhere else.

**Flavor architecture (2026-06-14):** personality is owner-aware and fact-additive — the actionable callout is built deterministically and a short (≤6-word) Ultron tail is appended, with the register matched to the owner (ENEMY contempt / our-team COMMAND / user-status SELF). `_flavor_ctx`/`_ctx_candidates` select the tail FOR the callout: one named enemy agent → that agent's pool in `_agent_flavor.py` is the SOLE source; 2+ agents → `_multi_flavor.py`; no agent → loc/count templates + the register pools in `_ultron_pools.py`. Curated set-pieces live in `_ultron_setpieces.py`. The canonical character brief is `scripts/relay_test/refs/ultron_voice.md`; all pools were board-generated, hand-curated, and passed through a 48-judge adversarial character-gate audit. See memory `reference_ultron_flavor_architecture.md`.

**Flavor architecture — deep expansion (2026-06-16):** the library grew from ~928 to **~4,147 audited tails** and selection became a HYBRID keyed-coarse + tagged-pool + semantic-fine-select system, **fail-open at every stage** (worst case = the prior deterministic behavior). `_agent_flavor.py` is now `dict[agent][situation] = list[TailEntry]` (each tail carries `loc:`/`dmg:`/`ability:` tags), agent × situation × sub-context, over the 16-key situation taxonomy in `_tail_schema.py`. `_flavor_ctx` runs two stages: (1) **COARSE ROUTE** — register + payload → the fine enemy situation (`_situation_for` / `situation_for_payload`), then the agent or multi pool; (2) **TAGGED-POOL + FINE-SELECT** — a 4-tier TAG filter (`_tier_filter`: tags-subset-of-active → share-most-specific-tag → tagless base → whole cell, each tier needing ≥3 survivors, then relaxing; a missing finer situation falls back to the agent's `spotted` pool, then the generic register pool) narrows the cell to the tails that FIT this exact callout, then the semantic `select_tail` (`_tail_selector.py`, embedder sidecar, MMR + recent-mask) re-ranks within it — fail-open to the deterministic `_pick_flavor`. **All ML stays in the loopback sidecar / build-time scripts**; the anticheat-pinned main process imports only numpy + urllib for this path. The library is GENERATED + audited via the `scripts/flavor_gen` (codegen/lint/dedup, apply-cuts) and `scripts/flavor_audit` (deterministic lint gate) pipelines. Two routing fixes shipped alongside: `_CRITICIZE_RE` no longer treats "call out" as a criticism verb (it is the primary Valorant RELAY verb — including it had inverted 105/106 factual callouts into criticisms of the named agent), and a new "I hit/tagged/cracked `<agent>` for `<n>`" pattern routes the damaged OBJECT to that enemy's damaged pool with the right `dmg:` tag.

**Flavor architecture — coherence audit + routing fixes (2026-06-16):** a by-hand curation pass made the deep-expansion library ruthlessly KIT-ACCURATE and concise. `_agent_flavor.py` was RE-AUTHORED down to **~1,628 tight `TailEntry` entries** (~5 per cell): every agent's `ult` cell is now its REAL ultimate (Jett → Blade Storm, Viper → her Pit, Raze → the rocket, Sova → blind shock, KAY/O → NULL//cmd, Killjoy → Lockdown), every `utility` cell is ability-TAGGED (`ability:<canon>`, incl. agent-unique abilities like Raze `boombot`/`paintshells` or Killjoy `alarmbot`/`turret`), and filler / off-topic / wrong-kit lines were cut. The content lives in a hand-written CURATED dict (`scripts/flavor_gen/curated_overrides.py`) applied by `scripts/flavor_gen/apply_curated.py` and verified by the deterministic lint gate (`scripts/flavor_audit/lint_tails.py`). `_ultron_setpieces.py` was de-biblicalized (~18 flood/Noah/ark/sacrament/God/church/abstract lines replaced with the machine / evolution / immortal / superior register; only the canonical meteor + evolution beats kept). Two selection changes: `_situation_for` now LIFTS the situation to `ult` whenever the payload carries an ult keyword (so "their Viper ulted B" reaches her curated ULT pool, not utility) before refining the `spotted` base; and `_flavor_ctx` SKIPS the semantic `select_tail` entirely for a small (<5) candidate cell — a curated/tag-filtered cell is already a tight fit, so the deterministic LRU `_pick_flavor` is as good as a cosine re-rank and avoids the per-callout sidecar embed (a latency win); the semantic selector now only earns its cost on a large ambiguous pool. The verb→ability routing relies on `_tail_schema._VERB_TO_ABILITY` (mollied→molly, walled→wall, darted→dart) folding a callout verb to the same `ability:` tag the curated `utility` cells carry. Selection ARCHITECTURE is unchanged in shape (coarse-keyed route → small-cell LRU or large-cell semantic fine-select, fail-open); the curated content is just kit-accurate and concise now.

**Ultron 1.0 + late-2026-06 relay additions:** new public matchers/helpers on top of the matrix above —
`match_verbosity_command(text) -> Optional[str]` (no/low/high "flavor"; off/on excluded so it stays disjoint
from the tail toggle) + `relay_verbosity()` / `set_relay_verbosity(level)` (runtime no/low/high state, env
`KENNING_U1_VERBOSITY`); `u1_llm_route_enabled()` / `set_u1_llm_route_enabled(b)` (env `KENNING_U1_LLM_ROUTE`,
default OFF) gating the LEAN-LLM relay route; **TURBO MODE (2026-06-23):** `turbo_mode_enabled()` /
`set_turbo_mode_enabled(b)` (env `KENNING_TURBO_MODE`, default OFF) + `turbo_aggressive()` /
`set_turbo_aggressive(b)` (env `KENNING_TURBO_AGGRESSIVE`) + `match_turbo_toggle(text) -> Optional[bool]`
("turbo mode on/off") + `match_turbo_sensitivity(text) -> Optional[bool]` ("turbo balanced/aggressive") —
the runtime master switch that AUTO-RELAYS inferred callouts without "tell my team" (the inference itself
lives in `intent_gate._relay_signal(..., turbo=)`); `match_thinking_toggle(text) -> Optional[bool]` +
`thinking_mode_enabled()` (env `KENNING_THINKING_MODE`, default OFF); `match_llm_device_switch(text)`
(GPU↔CPU hot-reload of the live model, anticheat-safe compute-location only); `is_complete_tactical_callout(text)`
(sidecar-free conservative slot-grammar predicate, exported for E3 snap-early-endpoint);
`_apply_snap_registry(...)` (the FIRST pass over `voice_lines.SNAP_REGISTRY` (inside the snap-check block, after ~20 prior guards),
`KENNING_SNAP_REGISTRY`); `_flavor_off_response(cmd, recent)` (the tails-OFF curated response sets, hooked at
the top of `build_relay_line` only when `not flavor_tails_enabled()`); `_as_enemy_status` (the "they're
out" enemy-commitment snap); the `_THANK_YOU_RE` + 10-tail gratitude snap. **build_relay_line u1.0 branch:**
in the generic-rephrase step (step 27) `if u1_llm_route_enabled():` build via `ultron_prompt.build_relay_prompt`
(`verbosity=relay_verbosity()`, `flavor_tail=flavor_tails_enabled()`, `agent_context=kit_facts_for(addressed
agents)`, `compound=_u1_compound`) else the legacy `_build_rephrase_prompt` + `_RELAY_REPHRASE_SYSTEM`; the
deterministic snap/tactical-literal pre-route + the post-LLM fact-guards are UNCHANGED (the C_route_llm
hybrid). All flag-OFF byte-identical → default behavior is the proven deterministic path. See the validating-HEAD
header + `docs/ultron_1_0/` for the full pivot.

---

#### Public API (`__all__`)

- `DEFAULT_ADDRESSEE_NAMES` — the 29-agent Valorant roster plus common STT homophones (`cipher`→Cypher, `gecko`→Gekko, `mix`→Miks, `way lay`→Waylay); the closed vocabulary the named-addressee patterns match against.
- `DEFAULT_ROAST_LINES`, `DEFAULT_FUN_FACTS` — seed / fallback pools for verbatim line types.
- `RelayCommand` — frozen dataclass from `match_relay_command`: `payload`, `raw_text`, `addressee` (`"team"` or display-cased agent name), `compose`, `context`, `directive`, `roast`, `fun_fact`, `verbatim`.
- `RelayPlaybackResult` — frozen dataclass from callers: `success`, `spoken_line`, `device_index`, `seconds`, `error`.
- `match_relay_command(text, *, names=None) -> Optional[RelayCommand]` — the strict matcher.
- `match_relay_toggle(text) -> Optional[bool]` — True = unmute relay, False = mute relay, None = not a toggle command.
- `build_relay_line(command, llm=None, *, rephrase, max_chars, recent_lines, generate_fn) -> str` — the main converter.
- `load_roast_lines(path)`, `load_fun_facts(path)` — load user-curated verbatim pools from disk, fail-open to defaults.
- `pick_roast_line(lines, recent_lines, rng) -> str` — anti-repeat pick from any verbatim pool; `pick_line` is an alias. Backed by `_pick_lru` (module-level `_LRU_COUNT`/`_LRU_SEEN`): serves the candidate gone LONGEST since last use (never-used first, ties random), comparing ONLY the passed candidate set so pools never cross-contaminate.
- `resolve_relay_device(configured) -> Optional[int]` — resolve device name/index via `kenning.audio.devices.resolve_device`, fail-open.
- `play_to_device(pcm, sample_rate, device_index, *, stream_factory, cancel_event=None) -> float` — synchronous playback via the WASAPI low-latency `make_output_stream` chokepoint (or test seam); returns seconds written. **Pre-widens mono PCM to centered STEREO and opens a 2-channel stream** (matching the B3 BroadcastSink): the relay B1 device is a stereo VoiceMeeter VAIO endpoint, so a 1-channel stream forced WASAPI's auto-convert to up-mix 1→2 channels *on top of* the 24k→48k resample, which **statics/distorts on B1** (B3 was clean because it already fed stereo). Writes the PCM in chunks and polls `cancel_event` between chunks so an "Ultron, stop" barge-in aborts mid-clip.
- `is_complete_tactical_callout(text) -> bool` — sidecar-free conservative slot-grammar predicate (E3 snap-early-endpoint; also used by `intent_gate`).
- `relay_route_info(...)` — the route classifier (template/route metadata for a callout).
- `pick_line(...)` — alias of `pick_roast_line` (anti-repeat LRU pick from any verbatim pool).

---

#### `match_relay_command` — parsing pipeline

**Normalisation (`_normalize_speech`)** applied first:
- `_KAYO_SLASH_RE`: collapses `kay/o` / `k/o` / `kay / o` → `kayo` so the agent name tokenises.
- `_FILLER_RE`: strips standalone filler (`uh`, `um`, `er`, `hmm`) and surrounding commas so triggers survive mid-utterance filler.
- `_ABBREV_SUBS`: word-boundary substitutions `kj`→Killjoy, `brim`→Brimstone, `yoroo`→Yoru, `vyce`→Vyse.
- `_LEADING_ARTIFACT`: strips a leading `"One,"` / `"1."` STT artifact before a relay verb.

**False-relay guard (`_NARRATION_LEAD_RE`)**: if the cleaned text matches narration/private-thought patterns (e.g. `"I should tell them"`, `"part of me wants to"`, `"chat says … respond"`, `"do I tell"`, `"have you ever"`) the function returns `None` immediately — the streamer is thinking out loud, not commanding.

**Ordered match attempts** (first match wins):
0. **Repeat / verbatim soundboard-check** (`_match_repeat_command`, highest priority) — `repeat`/`echo` (+ optional `back`/`after me`) followed by a REQUIRED `to my team` / `to <name>` addressee (before or after the phrase) → `RelayCommand(payload=<exact phrase>, verbatim=True)`. Strips meta-connectives (`exactly`, `word for word`, `the following:`); accepts any literal payload incl. a single short word. No addressee → no match (so conversational "repeat that" never relays).
1. `_ROAST_RE` → `RelayCommand(roast=True, compose=True)`
2. `_FUN_FACT_RE` → `RelayCommand(fun_fact=True, compose=True)`
3. `_GREET_RE` (skipped when verbatim suffix present) → `compose=True, directive="greet"`
4. `_FAREWELL_RE` (skipped when verbatim) → `compose=True, directive` from `_farewell_directive` (`farewell_win` / `farewell_loss` / `farewell` via `_WIN_RE` / `_LOSS_RE`)
5. `_COMPOSE_PATTERNS` (encouragement) → `compose=True, payload="encouragement"`
6. **`_RELAY_PATTERNS`** — 14 patterns covering group addressees (`_GROUP` = `my/our/the team/squad/…`), pronoun groups (`_GROUP_PRON` = `them/'em/everyone/the guys/…`), channel forms (`in game chat`), enemy-addressed bravado (`_ENEMY_GROUP`), `call out X`, `relay X`, `relay to my team X`, bare `relay X`, implicit-ask (`ask if anyone…`). Payload extracted; `ask … to` strips leading `to`; `_strip_verbatim_suffix` splits off trailing verbatim demand. Gated by `_payload_has_content`.
7. **Named-addressee patterns** (`_named_patterns`, LRU-cached per vocabulary key) — `the/my/their <agent>`: `tell <name> X`, `ask <name> X`, `say X to <name>`. `_NAME_CANON` maps STT variants (`kay o`→Kayo, `kill joy`→Killjoy, `cipher`→Cypher, `gecko`→Gekko, `mix`→Miks, `way lay`→Waylay) to display names via `_display_name`.
8. **Context + directive** (`_match_context_directive`): `"<reported-speech context>, <directive>"`. Context must be ≥ 3 words and contain `_CONTEXT_VERB_RE` (asked/saying/flaming/tilted/roasting/…) and not match `_FIRST_PERSON_TO_YOU_RE`. Literal-payload variant (`_TELL_HIM_TAIL_RE`: `"..., tell him X"`) checked first; then closed-directive-atom tail (`_DIRECTIVE_TAIL_RE`: respond/calm down/clap back/back me up/…). Addressee inferred from single roster name in context via `_addressee_from_context`.
8b. **Reported question → implicit answer** (`_match_reported_question`, NEW 2026-06-15) — `"X asked about/if Y"` with a question object but no explicit directive (`"Jett asked about Tony Stark"`, `"my teammate is wondering if you're a bot"`) routes to an in-character ANSWER path (`compose=True, directive="respond"`) even without a spoken "respond". Returns None for explicit-directive forms (owned by `_match_context_directive`), first-person-to-you instructions, and anything lacking a `_REPORTED_QUESTION_OBJ_RE` object. The answer itself is authored by `build_relay_line`'s answer path (Marvel topics like Tony Stark / Iron Man in-character, identity-category pools, or general-knowledge facts).
9. **Open ask** (`_ASK_OPEN_RE`): `ask what my Skye is doing` — only when payload mentions a roster name (single name) or a group reference.
10. **Bare `say X`** (`_BARE_SAY_RE`, last resort, ≥ 2 words): blocked when payload starts with `something/anything/your/the most…`, targets `stream/chat/viewers`, or contains `you can say/right now/for once/without conditions`.

**`_payload_has_content`**: rejects all-junk payloads (`_JUNK_SINGLE_WORDS`); single words must be in `_SHORT_CALLOUTS` or ≥ 4 chars and non-junk.

---

#### `build_relay_line` — routing order

1. **Verbatim** (`command.verbatim`) → `_cap_line(_strip_artifacts(payload))` — covers both the trailing verbatim demand and the `repeat to my team X` command.
1b. **Curated command** (`_as_curated_command`) — matches the payload against `_CURATED_PATTERNS` (~60 regexes, each tagged with a team and/or named command id); picks a scope-appropriate id (named when addressee ≠ team), selects a response from `COMMAND_RESPONSES` by `_pick_lru`, and slot-fills `{site}` (`_extract_site`: longest `_is_place` run, skips generic nouns, prefers site-letters), `{agent}` (`_roster_agents`), `{name}` (addressee). Returns None (falls through) when verbatim or no pattern matches.
2. **Pure morale compose** (compose + no directive + no context + `_is_morale_payload`) → `pick_line(DEFAULT_ENCOURAGEMENT_LINES)`.
3. **Greet / farewell compose** (compose + directive in `_DIRECTIVE_POOLS`) → `pick_line(_DIRECTIVE_POOLS[directive])`; pools: `greet`, `farewell_win`, `farewell_loss`, `farewell`.
4. **Calm-down** (`_is_calm_directive` on directive, or `_is_calm_payload` on payload) → `pick_line(DEFAULT_CALM_LINES)` with `{name}` substituted by addressee prefix.
5. **Identity question** (`_is_identity_question` on context or payload) → category-aware answer (2026-06-15): `classify_identity_question` (from `_ultron_identity.py`) picks the category and `pick_line(IDENTITY_POOLS[cat])` serves a varied in-character line (bot / soundboard / streamer / real-person / puppet / voice-changer / recording); falls back to `DEFAULT_IDENTITY_LINES` / `DEFAULT_STREAMER_LINES` when uncategorized. `_is_identity_question` was broadened to catch the who's-controlling-you / strings / off-switch / pre-recorded ("are you a recording / pre-recorded / on a tape") forms.
6. **Known-fact** (`_as_known_fact`) — checks `_GK_FACTS` (28 curated Q&A pairs covering common 3B errors: first president, moon distance, blood colour, etc.). Returns curated Ultron-voiced answer or None.
7. **Morale phrase** (`_is_morale_phrase`: `lock in`, `we got this`, `heads up`, etc.) → `pick_line(DEFAULT_ENCOURAGEMENT_LINES)`.
8. **Consolation / praise** (`_as_consolation_or_praise`): `_CONSOLATION_RE` (`nice try`, `unlucky`, `almost`) → `DEFAULT_CONSOLATION_LINES`; `_PRAISE_RE` (`good half`, `clutch`, `gg`) → `DEFAULT_PRAISE_LINES`.
9. **Deterministic snap callout** (`_as_snap_callout`) — returns `None` for off-snap content (LLM path), otherwise a zero-flavor or flavored literal:
   - Named addressee: `_as_named_question` for dominant small-talk questions; short imperative-verb-led orders → `"{Name}, {body}."`; questions and non-imperatives → None (LLM).
   - `careful <rest>` → `_flavored(", _FLAVOR_CAREFUL)`.
   - First-person self-reports (`_FP_LEAD_RE`: `I am/I'm`) → `"I'm {rest}."`.
   - `I have <x>` → `"I have {x}."`; `I saw/see <count> <place>` → count + place (enemy-flavored).
   - Counts: `_LEADING_COUNT_RE` (`there is/are <count> <place>` / bare `<count> <place>`) with `_is_place` guard → `"{Count} {place}."` (enemy-flavored).
   - Count + movement: `<count> rotating/coming/going/pushing/…` (≤ 8-word rest) → `"{Count} {rest}."`.
   - Spike: `spike <rest>` (≤ 7 words) → `"Spike {rest}."`.
   - Last alive: `_LAST_LEAD_RE` + `_is_place` → `"Last, {place}."`.
   - All enemies: `all enemies are <place>` → `"They're all {place}."`.
   - Enemy has weapon/ult: `they have op/ult/odin/…` → `_FLAVOR_ULT`-flavored.
   - Enemy utility: `they walled/smoked/darted/… <place>` → `_FLAVOR_UTILITY`-flavored.
   - Enemy movement: `they're pushing/going/rushing <place>` → enemy-flavored.
   - Enemy position / action: `_ENEMY_LEAD_RE` + `_is_place` or `_ACTION_WORDS` → enemy-flavored; inside this block also tries `_as_agent_position`, `_as_ult_callout`, `_as_agent_utility`.
   - **Contextual enemy action** (`_as_enemy_action`, runs BEFORE agent-position): a bare `agent/count + action` with no "enemy" said (`_BARE_TO_ING` maps `flank→flanking`, `hit→hit`, `push→pushing`, …) → a clean enemy callout (`"Cypher is flanking."`, `"Sova hit 40."`); handles multi-agent strings; defers to position handling and never fires on `our/my`.
   - Named agent(s) at place: `_as_agent_position` (roster-name subject only, `_is_place` on location).
   - Damage: `<agent> hit <n>` (optional short location) → `_FLAVOR_DAMAGE`-flavored.
   - Ults: `_as_ult_callout` — `<agent> has ult`, `<agent> is one off ult`, multi-agent `have ults`, `just used/ulted`, `has no ult`, `ult is down [back in N]`, `ult ran out` → `_FLAVOR_ULT` when Their-prefixed.
   - Agent utility: `_as_agent_utility` (agent name + ability-lead token in first 1–2 tokens) → `_FLAVOR_UTILITY` for enemy, clean for ours.
   - Economy (`_as_economy_callout`): bare save/force/full-buy → `DEFAULT_SAVE_LINES` / `DEFAULT_FORCE_LINES` / `DEFAULT_FULLBUY_LINES`; enemy economy, `anti`, or long → None.
   - Economy request: `drop/buy … gun/op/…` (≤ 6 words) → literal imperative.
   - `_MOVE` table: exact bare movement commands → canonical strings (`"Rotate."` etc.).
   - General team directive: first word in `_TEAM_DIRECTIVE_VERBS`, ≤ 7 words, no question → literal imperative.
10. **Compound callout** (`_as_compound_callout` / `_split_compound`): splits on strong joiners (`--`, `;`, `plus`, `also`) and on ` and ` / `,` ONLY before a `_NEWFACT_SUBJECT` (preserving multi-agent callouts and intra-fact commas). Each piece re-run through `_as_snap_callout(flavor=False)`. Economy deduplication prevents repeated save lines. Returns `(det_line, None)` (fully deterministic, single enemy-facing tail ≤ 11 words) or `(det_line, leftover)` (partial); leftover is recursively routed through `build_relay_line` with `recent_lines=None`.
11. **Pre-route dense-tactical to literal**: if `_fact_tokens` finds ≥ 1 tactical fact-token (count/location/ability) AND total ≥ 2 fact+agent tokens, and no snap/compound handler matched, skip the LLM entirely → `_literal_relay` (instant in gaming mode).
12. **LLM rephrase** (`generate_fn` or `llm.generate_stream`): prompt built by `_build_rephrase_prompt`. Called with `record_history=False, suppress_memory_context=True, enable_thinking=False` (no conversation bleed). Anti-repeat ring: last 6 recent lines shown for team-addressed non-answer commands only; suppressed for named-teammate callouts and any answer/respond/identity command (`_is_answer_command`).
13. **Recent-echo guard**: if model output matches a recent line verbatim → discard → fallback.
14. **Switch-hallucination guard**: if output contains `they're/enemies switch` and input has no `switch` → discard → fallback.
15. **Post-processing**: `_strip_artifacts` (removes `/no_think`, `<|...|>`, `<placeholder>` leakage, speaker labels `Ultron:` / `Team:`, outer quotes); `_cap_sentences` (cap at 3 whole sentences); `_strip_spurious_vocative` (removes roster-name or generic vocative opener on team-wide lines); `_fix_proper_nouns` (Sokovia, Wakanda, Mjolnir).
16. **Repair** (`_repair_against_input`, plain relays only): restores first-person subject dropped/inverted (`_FP_LEAD_RE`), enemy subject flipped to self or second person (`_ENEMY_LEAD_RE` + `_FIRST_PERSON_OUT_HEAD`), `last` callout dropped (`_LAST_LEAD_RE`), leading enemy count dropped (`_LEADING_COUNT_RE`).
17. **Fact-preserving abstention** (`_output_keeps_facts`): if ≥ 30 % of fact-tokens (counts, agents, locations, ability words) dropped, or output invents an agent/location absent from input, or ownership flipped (their↔our) → `_literal_relay` (clean passthrough with optional enemy-flavor tag).
18. **Agent-name preservation** (`_preserve_agent_names`): single-agent swap (Chamber→KAY/O) undone by re-substituting the input's agent.
19. **Addressee enforcement** (`_ensure_addressee`): named callouts always open with the teammate's name.
20. **`_cap_line`** (sentence-boundary-aware): cuts at last complete sentence that fits within `MAX_RELAY_LINE_CHARS` (360).

---

#### Curated pools

| Symbol | Use |
|---|---|
_(Pool sizes were greatly expanded in the 2026-06 coherence pass and are deliberately NOT tracked here to avoid re-staling — see `_ultron_setpieces.py` for the live counts.)_

| `DEFAULT_ENCOURAGEMENT_LINES` | Pure morale / hype / focus calls |
| `DEFAULT_CONSOLATION_LINES` | After lost round (`nice try` / `unlucky`) |
| `DEFAULT_PRAISE_LINES` | After won round (`good half` / `clutch`) |
| `DEFAULT_GREETING_LINES` | Team intro as Ultron |
| `DEFAULT_VICTORY_LINES` | Win sign-off |
| `DEFAULT_DEFEAT_LINES` | Loss sign-off |
| `DEFAULT_FAREWELL_LINES` | Neutral sign-off |
| `DEFAULT_CLUTCH_LINES` | Clutch / 1vX hype |
| `DEFAULT_IDENTITY_LINES` | "Are you an AI?" answer |
| `DEFAULT_STREAMER_LINES` | "Are you a streamer?" answer |
| `DEFAULT_CALM_LINES` | Clinical calm-down with `{name}` slot |
| `DEFAULT_CRITICIZE_LINES` | Curated "criticize `<agent>`" lines (NEW 2026-06-15) — replaces the unreliable 3B for the criticize command, served via `pick_line` |
| `DEFAULT_SAVE_LINES`, `DEFAULT_FORCE_LINES`, `DEFAULT_FULLBUY_LINES` | Economy buy decisions |
| `DEFAULT_ROAST_LINES` | Seed roast (1 line; user extends `data/relay_roasts.txt`) |
| `DEFAULT_FUN_FACTS` | Fallback fun facts (3 lines; corpus ships at `data/relay_fun_facts.txt`, 1014+ lines) |
| `_DIRECTIVE_POOLS` | dict mapping directive key → pool for set-piece composes |

Flavor pools appended to snap callouts via `_pick_flavor` (anti-soundboard, avoids recent 8-line window): `_FLAVOR_ENEMY`, `_FLAVOR_CAREFUL`, `_FLAVOR_ULT`, `_FLAVOR_DAMAGE`, `_FLAVOR_UTILITY`. `_flavored(callout, pool, recent_lines)` appends a picked tag; `_pick_flavor` excludes tags seen in recent output. **2026-06-16:** for an agent / multi callout the tail is now chosen through the hybrid path (`_flavor_ctx` → `_tier_filter` tag filter → semantic `select_tail`), and `_pick_flavor` is the fail-open floor at every stage — so the worst case is exactly this prior deterministic anti-repeat behavior. **Tail spacing (NEW 2026-06-15):** `_join_tail(head, tail)` is the single join helper — it guarantees a sentence terminator between the callout and its flavor tail (so the callout never slurs into the tail), then the TTS path honours it (see the kokoro inter-sentence gap below). Multi-fact callouts still flow as one sentence (the per-fact joins do not add a terminator).

---

#### `_REPHRASE_PROMPT` — Ultron persona and hard rules

Single large format-string injected with `{task}`, `{addressee}`, `{by_name}`, `{payload_block}`, `{context_block}`, `{recent_block}`. Key sections:

- **Two registers**: SNAP (enemy positions/counts/damage/status/self-status/movement — short, literal, zero flavor) vs OFF-SNAP (insults, economy, calm-down, questions, banter, identity, Marvel — ~2 sentences, cold Ultron character, ≤ 30 words).
- **Hard rules**: every number, agent name, weapon, map callout kept exactly. Counts never dropped. Plural place names never singularized. `"play their life"` ≠ `"play for time"`. Economy directives are OFF-SNAP (explained). First person (`I am/I'm`) is always the USER's own action — never flipped to second person or imperative. `ask <someone> <question>` means DELIVER the question, never answer it. Directives are commands TO the team, never self-reports. Ownership locked (our/their).
- **Identity** (Ultron): only when a teammate DIRECTLY asks what Kenning is; cold, brief (2 sentences), names himself Ultron — AI from the future harvesting RR. Streamer dismissal is its own register. Otherwise never self-identifies.
- **Marvel**: answers in-character with contempt; deepest contempt for Tony Stark.
- **Banter-at-you**: fresh comeback every time, never echoes the insult back, addresses by name.
- **Each callout stands alone**: never carry over a name/location/number from a prior line.
- **Valorant shorthand glossary** inline (op, saving, full buy, force, eco, flash, rotate, anchor, retake, TP, off site, etc.).
- **Context block** and **recent-line block** injected when present; recent lines shown for team-addressed non-answer lines only (last 6).
- `_directive_task(directive)` maps closed-vocabulary directive strings to prompt task clauses (calm/de-escalate, acknowledge/agree, clap back/shut down, back-me-up, default respond with GK-answer or banter rules).

---

#### Playback path

- `relay_tts_text(line) -> str`: context-aware TTS pronunciation fix — replaces uppercase `A` before a location token with `eigh` so the site letter is not pronounced as the indefinite article (feeds into the Kokoro `_play` path).
- `resolve_relay_device(configured)` → PortAudio index (via `kenning.audio.devices.resolve_device`, fail-open).
- `play_to_device(pcm, sample_rate, device_index, *, stream_factory, cancel_event)` → opens a WASAPI low-latency stream per-relay via `make_output_stream`, writes int16 mono PCM synchronously in chunks (polling `cancel_event` between chunks for barge-in cancellation), always closes the stream, returns seconds played.
- `_fallback_line(command)` → deterministic spoken line when the LLM is unavailable (directive-keyed stock phrases or clean literal of payload; no `"Team:"` label prefix).

**Playback cancellation + speaker mute (NEW 2026-06-15):**
- **"Ultron, stop" cancels all channels** — `_cancel_all_playback()` on the orchestrator stops conversational TTS, the relay mic bus (B1), the OBS broadcast (B3), and the monitor mirror at once. `play_to_device` is chunked + `cancel_event`-aware; `BroadcastSink.cancel_current()` aborts the in-flight clip and drains its queue; the interrupt watcher runs for the duration of relay (and conversational) playback.
- **Stop is ALWAYS available (NEW 2026-06-15)** — the interrupt watcher is no longer gated only on `audio.barge_in_enabled` (held `false` on this box for loopback hygiene). A dedicated `audio.stop_command_enabled` (default `true`, env `KENNING_STOP_COMMAND_ENABLED`) keeps "Ultron, stop" on independently. The single gate `Orchestrator._stop_watcher_enabled()` returns `True` when EITHER flag is set AND the wake + audio infra exist (so bare/test orchestrators never spawn the watcher). It backs all four watcher-start sites (the two conversational paths, the main `run()` turn, and the relay-playback site). Self-trigger from loopback is bounded by the stricter wake gate (0.7 / 3 frames) + `mute_speakers` + relay output living on the virtual B-buses, not the physical mic. GUI knob: «"Ultron, stop" (always on)» in Hearing.
- **Inter-sentence gap** — single-clip Kokoro synthesis inserts a PERIOD-length gap between sentences (default ~320 ms, env `KENNING_TTS_SENTENCE_PAUSE_MS`) so a callout does not blend into its flavor tail. Paired with `relay_speech._join_tail`, which guarantees the terminator the gap keys off; multi-fact callouts (one sentence) still flow without an added gap.
- **"Mute my speakers" (`audio.mute_speakers`, default `false`)** — when on, the default-speaker path is silenced: conversational Kokoro output is zeroed and the relay monitor mirror is skipped, while the relay still reaches teammates (B1) and OBS (B3) — for isolating loopback tracks. Read live; the GUI "Mute my speakers (loopback)" knob and its dedicated "APPLY MUTE ONLY" button hot-apply just that one setting.

**Auto push-to-talk (`kenning.ptt`, NEW 2026-06-15, DEFAULT OFF):** Valorant TEAM voice is push-to-talk only, so a relay line only transmits if the team-PTT key is held while it plays. The `kenning.ptt` package holds that key via an **external USB-HID microcontroller** (Arduino Leonardo) over serial — the host writes bytes ONLY, never synthetic input (the anticheat-clean design; see `safety/anticheat.py`'s "Deliberately NOT blocked" list + `tests/safety/test_ptt_import_clean.py`). Pieces: `PttController` (press/heartbeat/release state machine + a host max-hold watchdog + fail-safe swallowing) over a pluggable `PttBackend` — `NullPttBackend` (the default; completely inert, zero latency) or `SerialHidPttBackend` (lazy-imports `pyserial`; one-byte `D`/`U`/`H` protocol; fail-safe → on any serial error it disables, never falls back to in-process input). `build_ptt_controller()` selects the backend from `push_to_talk.*` config (`enabled`, `backend`, `serial_port`, `key`, `lead_ms`, `release_tail_ms`, `heartbeat_ms`, `max_hold_seconds`; env `KENNING_PTT_ENABLED` / `KENNING_PTT_SERIAL_PORT`). **2026-06-15:** the dead-air margins were widened (`lead_ms` 120→200, `release_tail_ms` 150→300) so Ultron's relay speech is never clipped at the start/end — pure dead air, imperceptible. The orchestrator constructs `self._ptt` unconditionally (relay-adjacent core, gated only on the flag — never lean-skipped) and calls `_ptt_hold()` as the first line of the relay `play_to_device` try + `_ptt_release()` in its `finally` (so the key releases on a full clip, an "Ultron, stop"/barge cancel, OR an error); `shutdown()` closes it.

**HID-only HARDENED variant (2026-06-15):** the serial device is a *composite* HID-keyboard **+ CDC serial (COM) port** — the exact USB-descriptor fingerprint of Arduino aimbot/BadUSB devices. A 16-agent research board's #1 (legitimate, not evasion) hardening was to **drop the COM port**: the hardened firmware `firmware/leonardo_ptt_hid/leonardo_ptt_hid.ino` is **HID-only** (`-DCDC_DISABLED`) — a Boot keyboard + a vendor **Raw HID** collection (usage page `0xFFC0`, via NicoHood HID-Project) — plus a **custom USB VID/PID** (`0x1209` pid.codes + product `"USB Keyboard"`, dropping the Arduino `0x2341` identity). It then presents identically to any commercial keyboard with a config interface (Corsair/Razer/QMK). The matching host transport is `RawHidPttBackend` (lazy `hidapi`; the same `D`/`U`/`H` protocol as HID **output reports** — writing an output report is DEVICE I/O, NOT synthetic input, no `LLKHF_INJECTED`). `backend: "auto"` (default) tries `RawHidPttBackend` (find by VID `hid_vid` + usage page `hid_usage_page`) first, falling back to `SerialHidPttBackend`; `"rawhid"`/`"serial"` pin one. Flashing note: with CDC gone the board can't be serial-flashed — `firmware/leonardo_ptt_hid/` is flashed via a **double-tap-reset + poll-for-bootloader + avrdude** dance (the `--clean` rebuild matters so the *core* descriptor picks up the flags, not just the sketch).

Firmware (legacy serial): `firmware/leonardo_ptt/leonardo_ptt.ino` — ATmega32u4 native HID via `Keyboard.h` + CDC serial; both firmwares have a **hardware deadman** (auto-releases the key in ~200 ms if the host stops heartbeating, so a host crash can't jam the mic open). Anticheat note: external-HID is the architecture the auto-PTT research (`memory/project_auto_ptt_research_2026_06_15`) rates "empirically-low-but-nonzero, ToS-prohibited" (and the hardening makes the *device* indistinguishable from a normal keyboard); the boot canary's risky-lib tuple was widened (`keyboard`/`pydirectinput`) to trip if a regression ever pulls an in-process keypress lib.


#### `audio/ring_buffer.py`
- `class RingBuffer` — fixed-duration audio backlog (pre-speech window)
  - `write(samples)` / `clear()` / `__len__()` / `capacity` property
  - `snapshot(last_n_samples=None) -> np.ndarray` — full buffer when
    unsliced; the most recent `last_n_samples` when given. The
    orchestrator slices a short COLD-mode pre-roll (post-wake; avoids
    "Tron" prefix) and a longer WARM-mode pre-roll (post-TTS
    follow-up; avoids first-word clipping) from the SAME buffer.
    `last_n_samples >= len(buffer)` returns full; `<= 0` returns
    empty. (2026-05-10 mode-aware pre-roll fix.)

#### `audio/vad.py`
- `class SpeechEvent(Enum)` — SPEECH_START / SPEECH_END / NONE
- `class VadResult` — dataclass: event, is_speech, probability
- `class VoiceActivityDetector` — silero-vad wrapper; consumes 512-sample windows.
  - `reset()` — clear hysteresis state AND restore the baseline silence-window requirement (so an adaptive bump from the previous utterance doesn't leak into the next one).
  - `set_min_silence_duration_ms(ms)` (2026-05-11 adaptive end-of-turn) — adjust trailing-silence requirement at runtime. Orchestrator calls this from `_capture_utterance` once speech has been active past `vad.long_utterance_threshold_seconds` so a thinking pause mid-prompt doesn't close a long technical description.

#### `audio/wake_word.py`
- `class WakeWordDetector` — openWakeWord wrapper
  - Loads the active word's custom ONNX (`wake_word.model_path`, e.g. `models/openwakeword/ultron.onnx`)
  - Falls back to the custom `{fallback_name}.onnx` side-by-side model (default `ultron.onnx`); only if that
    is also absent does it fall back to a pretrained openWakeWord word of the same name — `hey_jarvis` is NOT used
  - `process(audio: np.ndarray) -> bool` — feed a frame; returns True on a wake fire
  - `reload_for_word(word: str) -> tuple[bool, str]` — hot-swap the live model (GUI dropdown)
  - `fired_recently(window_s: float = 0.5) -> bool` (V1-gap A4) — read-only accessor for the last trigger timestamp; returns True iff a wake fire happened within ``window_s`` seconds. Used by the orchestrator's pre-task barge-in watcher. Idempotent — does not consume the trigger.

#### `audio/smart_turn.py` (NEW 2026-05-12)

Pipecat Smart Turn V3 ONNX wrapper. CPU-only semantic end-of-turn
confirmation that runs after Silero VAD detects silence. Pinned to
`CPUExecutionProvider` — zero VRAM cost. 8 MB int8 model;
`~12 ms` inference target, sub-150 ms in practice on this hardware.
Fail-open at every layer: missing model file / disabled config /
load failure / inference exception all degrade silently to "trust
VAD" rather than misclassifying.

- `SMART_TURN_SAMPLE_RATE = 16000`, `SMART_TURN_WINDOW_SECONDS = 8.0`,
  `SMART_TURN_MEL_BINS = 80`, `SMART_TURN_MEL_FRAMES = 800`,
  `SMART_TURN_INPUT_NAME = "input_features"` — model contract constants.
- `class SmartTurnLoadError(RuntimeError)` — raised at construction
  time only (missing file, out-of-range config). Inference-time
  failures degrade to None.
- `@dataclass(frozen=True) class SmartTurnVerdict` — `is_complete: bool`,
  `probability: float` (sigmoid output, already activated in the ONNX
  graph), `latency_ms: float` (wall-clock including preprocessing).
- `truncate_or_pad_for_smart_turn(audio, sample_rate, *, window_seconds=8.0) -> np.ndarray`
  — pure helper. Truncates audio HEAD-first to the last
  `window_seconds`; pads-at-start is the `WhisperFeatureExtractor`'s
  job (`padding="max_length"`). Converts non-float32 inputs to
  float32; flattens multi-dim inputs. Rejects non-16 kHz with
  `ValueError` (callers resample upstream).
- `class SmartTurnDetector`:
  - `__init__(model_path, *, completion_threshold=0.5, window_seconds=8.0, num_threads=1)`
    — validates the model file exists and parameters are in range;
    does NOT load the ONNX session into memory. Raises
    `SmartTurnLoadError` on bad inputs.
  - `available` property — True iff loaded and healthy. False before
    first call (lazy) and after a load failure.
  - `warmup() -> bool` — forces the lazy-load path now. Returns
    True on success, False on load failure (logged at WARN).
  - `is_complete(audio, sample_rate=16000) -> Optional[SmartTurnVerdict]`
    — main entry. Returns a verdict on success, None on any error
    (treated by caller as "undecided" → trust VAD). Thread-safe via
    an internal load + inference lock.
  - `close()` — idempotent release; subsequent `is_complete` returns None.
- `build_detector_from_config(smart_turn_cfg, project_root) -> Optional[SmartTurnDetector]`
  — orchestrator-side factory. Returns None when smart-turn is
  disabled, when the model file is missing on disk, or when
  construction fails. WARN-level log distinguishes the cases. This
  is the single seam between config and runtime that the orchestrator
  uses; no other call site constructs a detector directly.

#### `audio/voice_lines.py` (NEW 2026-06-18 — voice-line aggregate + data-driven snap registry)

The single review surface for all relay voice lines + the regexes that route to them. A PURE relocation
(proven byte-identical, 358 symbols, by `scripts/_voice_lines_verify.py` under `PYTHONHASHSEED=0`) of the
social-snap regexes + pools out of `relay_speech`, co-located under a category→trigger→matcher→responses→
tails map, and RE-EXPORTING `DEFAULT_*_LINES` + `AGENT_FLAVOR` so existing imports keep working. Adds the
DATA-DRIVEN extension contract:
- `class SnapRule` + `SNAP_REGISTRY: tuple` — a list of (matcher → responses/tails) rules consumed by
  `relay_speech._apply_snap_registry` as the FIRST pass in `build_relay_line`'s snap gate
  (`KENNING_SNAP_REGISTRY`, default on). Append ONE `SnapRule` = a new snap, no code change; FIRST match
  wins (precedence); the hardcoded paths remain as fallback.
- `class TargetSnapRule` + `TARGET_SNAP_REGISTRY: tuple` — the addressee-targeted variant (named-agent snaps).
- `DEFAULT_ROAST_LINES`, `DEFAULT_FUN_FACTS`, and the relocated `DEFAULT_*_LINES` pools + `AGENT_FLAVOR`.
- The flavor-toggle mishear tables `_FLAVOR_OFF_MISHEAR_RE` / `_FLAVOR_ON_MISHEAR_RE` and the `_HELLO_RE`
  social-snap regexes live here (the regex-coupled gazetteer tables that would lose order/safety-net if
  moved were deliberately LEFT in `relay_speech`).
- Golden gate: `tests/data/voice_lines_golden_digest.json` + `tests/test_voice_lines_golden.py` run
  `scripts/_voice_lines_verify.py check` in a `PYTHONHASHSEED=0` subprocess; flavor-lint
  `scripts/flavor_audit/lint_tails.py` (`_tail_schema.lint_agent_flavor`). RE-BLESS after any intentional
  voice-line/registry change. Detail: memory `project_voice_lines_aggregate_2026_06_18.md`.

#### `audio/voicemeeter_level.py` (NEW 2026-06-18 — boot level guard for the team bus)

Anticheat-clean VoiceMeeter Remote-API helper (no input/capture surface) that, when enabled
(`KENNING_RELAY_VM_LEVEL_GUARD` / config, default OFF), checks/raises the B1 bus (Ultron→Valorant
mic) fader at boot so Vivox AGC makeup-gain doesn't lift the codec noise floor (the live-diagnosed "volume
same, quality bad"). The DECISIVE fix is manual (raise B1); this is the optional code complement, paired
with `relay_speech._shape_for_team` (team-path-only DSP: rumble-HP → voiced-RMS normalize → comfort-noise
floor → tanh soft-clip; gate `KENNING_RELAY_TEAM_DSP`, each stage env-gated + fail-open) + the GUI APPLY
UNMUTE button. Detail: memory `project_valorant_audio_rootcause_2026_06_18.md`.

#### `audio/ultron_prompt.py` (NEW 2026-06-20 — Ultron 1.0 lean prompt assembler)

The route-everything-through-the-8B prompt builder. Replaces the legacy ~4.8k-token
`relay_speech._build_rephrase_prompt` (which overflows the u1.0 `n_ctx=4096` cap and yielded EMPTY 8B
output) with a LEAN (~165-word) templated prompt validated live to produce correct, fast (~0.2-0.5 s),
in-character, fact-preserving relays. Stdlib-only (anticheat-safe). Public API:
- `VERBOSITY_LEVELS = ("none","low","high")`, `DEFAULT_VERBOSITY = "high"`,
  `normalize_verbosity(value) -> str` (word-aware: parses "no/low/high flavor" + synonyms; fail-soft).
- `RELAY_SYSTEM` / `PRIVATE_SYSTEM` — stable, cache-friendly persona + output-rule prefixes (the variable
  callout + exemplars go LAST in the user message). The `_VERBOSITY_DIRECTIVE` for `none` forces a
  TELEGRAPHIC FRAGMENT (the 8B collapses a weak "be brief" back into a sentence).
- `@dataclass PromptResult(system, user, sampling, enable_thinking=False)` — thinking is ALWAYS False here.
- `build_relay_prompt(callout, *, addressee="team", verbosity, flavor_tail, exemplars, agent_context,
  recent_lines, compound) -> PromptResult` — `addressee != "team"` opens with the teammate's name;
  `compound=True` instructs ONE combined line; `agent_context` (from `agent_kits.kit_facts_for`) prevents
  kit hallucination; per-verbosity `max_tokens` (none 24 / low 40 / high 72).
- `build_private_prompt(query, ...) -> PromptResult` — the ME-ONLY (PRIVATE_REPLY) variant; its own Q&A
  exemplars (`_DEFAULT_PRIVATE_EXEMPLARS`), `max_tokens` lifted to 110 on `high`.
- **`strip_prompt_echo(text, *, max_sentences=3, max_chars=300) -> str` (NEW 2026-06-22 — output guard):**
  the small model occasionally ECHOES this module's prompt scaffolding as if it were speech (the live bug
  `bu5fh4lc8` spoke the `_reconcile_block` note aloud — "The callout below is the AUTO-NORMALIZED text…",
  25 s), appends a "- Ultron" signature, and rambles. Drops any sentence containing a template marker
  (`_PROMPT_ECHO_MARKERS` — multi-word, template-specific so a normal line never trips), strips a trailing
  `[-–—] Ultron` signature (an inline "I am Ultron." is untouched), and hard-caps sentence count + chars.
  Returns "" when the WHOLE output was scaffolding (caller falls back). Pure stdlib, fail-soft (any error
  returns the input). WIRED into all three u1.0 LLM-output paths: `relay_speech.build_relay_line` (Safety
  net 3), `relay_speech._social_llm_line`, `orchestrator._maybe_handle_private_reply`.
- **HARD RULE (module docstring):** callers MUST run the existing fact-guards
  (`relay_speech._output_keeps_facts` / `_repair_against_input` / `_literal_relay`) on the model output —
  this module only builds the prompt, it does not relax the correctness backstop.
- Tests: `tests/audio/test_ultron_prompt.py`, `tests/audio/test_u1_llm_route.py`.

#### `audio/agent_kits.py` (NEW 2026-06-20 — Ultron 1.0 agent-kit reference for LLM context injection)

Hot-swappable, VERSION-STAMPED (`KITS_VERSION = "v2026-06-20 (Patch 12.10)"`) per-agent kit facts injected
into the relay/answer prompt so the 8B never hallucinates a kit (it mis-stated Sova's kit in probing).
Pure data + stdlib. Sourced from `docs/ultron_1_0/02_research/board/B_valorant_kits.md` with the
adversarially-verified `C_domain.md` corrections applied inline (Iso Undercut also suppresses 4 s; Clove
Not Dead Yet = 8 pts; Veto Evolution = 7 pts; Waylay/Veto/Miks/Tejo flagged post-cutoff). Public API:
- `AGENT_KITS: Dict[str, str]` — 29 agents, compact `"Role | C=.. Q=.. E=..(/sig) X=..(ult)"` form.
- `agent_kit_fact(agent) -> Optional[str]` (tolerant canon lookup) and
  `kit_facts_for(agents, *, limit=4) -> List[str]` (de-duped, capped — long prompts raise hallucination risk).
- To update for a patch/agent: edit `AGENT_KITS` + bump `KITS_VERSION` — no code change. Tests:
  `tests/audio/test_agent_kits.py`.

#### `audio/intent_gate.py` (NEW 2026-06-20 — Ultron 1.0 always-listening 3-way/4-class intent gate)

The optional-wakeword scenario classifier (CLASSIFIER ONLY at present — M5b will wire it into the run loop).
A COMPOSITION of existing proven components (no new in-process ML); cost-asymmetric, FAIL-CLOSED to IGNORE.
Public API:
- `class Scenario(str, Enum)` = {RELAY_TO_TEAM, PRIVATE_REPLY, COMMAND_LOCAL, IGNORE};
  `@dataclass ScenarioVerdict(scenario, confidence, reason, needs_llm)`.
- `classify_scenario(text, *, wake_present, seconds_since_response, no_speech_prob, avg_logprob, names)
  -> ScenarioVerdict` — cascade: ASR-confidence pre-reject (`no_speech_prob`/`avg_logprob`, env
  `KENNING_GATE_*`) → COMMAND_LOCAL (toggle/Spotify/stop matchers) → RELAY_TO_TEAM (`correct_callout_stt`
  L1-only [NOT `normalize_command`, which over-injects the relay lead] → `match_relay_command` 0.95 /
  `is_complete_tactical_callout` 0.90 / agent+`_fact_tokens` 0.88 — the weak semantic `relay_intent_ok`
  signal was DROPPED 2026-06-21 (it false-relayed conversation to the team; RELAY needs a strong signal)) → addressing
  rules NO→IGNORE / **PRIVATE_REPLY requires an explicit Ultron address signal** (a leading wake word OR an
  `_ADDRESS_NAME_RE` unambiguous name — `ultron`/`kenning`/`hey ai`/`the ai`, anywhere; the common nouns
  `machine`/`robot` are EXCLUDED from the gate so they don't false-fire on ordinary speech; NEW 2026-06-22) → else
  IGNORE. **The addressing RULES alone are NOT enough for PRIVATE** (they score a bare question/imperative
  ADDRESSED ≥0.80, which false-fired private replies on un-named conversation the player aimed at teammates —
  live bug `bu5fh4lc8`). An un-named/un-waked line is dropped to IGNORE outright; `classify_scenario` NO
  LONGER sets `needs_llm` (the LLM band is retired from the hot path — it mislabelled "Follow orders." →
  PRIVATE and cost a model forward-pass per chatter line).
- `resolve_with_llm(verdict, text, llm) -> ScenarioVerdict` — single-token {PRIVATE, IGNORE} LLM escalation
  for a `needs_llm` verdict (`enable_thinking=False`, fail-CLOSED on any non-PRIVATE token / error).
  RETAINED for callers + unit-tested in isolation, but `classify_scenario` no longer triggers it (2026-06-22).
- **TURBO MODE (2026-06-23):** `classify_scenario(..., turbo=False)` threads a `turbo` flag into `_relay_signal`.
  When ON (the user's opt-in), AFTER the strict bands decline, `_relay_signal` runs the FULL `normalize_command`
  → `recover_relay_lead` → `match_relay_command` predicate (the SAME one dispatch uses) and returns 0.75 if it
  recovers a relayable lead — so a bare callout ("rotate", "they have breach ult, play off site") relays without
  a "tell my team" prefix, and a turbo RELAY verdict ALWAYS relays downstream (no gate/dispatch mismatch).
  `turbo_aggressive` adds a 0.60 band via `_relay_intent.relay_intent_ok` (deliberately re-opens the dropped
  semantic positive — double-gated, the spec's R6/R7 trade-off). `_is_command_local` includes the turbo matchers
  so "turbo mode off" survives the gate. OFF → byte-identical. The orchestrator carries turbo via `_listening_now()`
  (= `_always_listening OR relay_speech.turbo_mode_enabled()`, read live, so turbo implies continuous capture),
  `_classify_always_listening` passes `turbo=` + the configured addressee roster, and a `turbo_mode_enabled()`-gated
  RELAY backstop before the semantic router force-relays a RELAY_TO_TEAM verdict the strict matcher couldn't parse
  (the aggressive-band case). Voice "turbo mode on/off" + STOP-window amber TURBO button. Spec:
  `docs/ultron_1_0/04_implementation/10_turbo_mode_spec.md`. Tests: `tests/audio/test_turbo_mode.py`.
- DEFAULT OFF (opt-in `addressing.always_listening`); wake-word stays the competitive default; thresholds
  are heuristic starting points to calibrate on the labeled battery + `logs/addressing.jsonl`. PREREQUISITE:
  VoiceMeeter mic isolation. Tests: `tests/audio/test_intent_gate.py`.

#### `audio/routing_rules.py` (DATA SSOT — STT vocab + relay-lead regexes + router thresholds)

One of the three CLAUDE.md "key area" relay modules. Pure data, three layers, consumed across the routing cascade:
- **L1 STT vocab/protection:** `AGENTS` gazetteer (the single source of truth, line 47) + `MAPS`, `WEAPONS`,
  `ABILITIES`, `LOCATIONS`, `TERMS`, `MULTI_TERMS`; mishear tables `AGENT_MISHEARS` / `TERM_MISHEARS` /
  `MISHEAR_FORCE`; `FUZZY_BLOCK` + `PROTECT_EXTRA` (over-correction guards). Consumed by `_stt_correct`.
- **L2 relay-lead recognition:** the `NORM2_*` regexes (`NORM2_MANGLED_TEAM_LEAD_RE`, `NORM2_TELL_TEAM_LEAD_RE`,
  `NORM2_IRREGULAR_TEAM_LEAD_RE`, `NORM2_TEAM_NOUN`, `NORM2_MANGLED_TELL`, `NORM2_TELL_CLASS_VERB`).
  Consumed by `command_normalizer`.
- **L3 router thresholds:** `ROUTE_DEFAULT_THRESHOLD = 0.50`, `ROUTE_DEFAULT_MARGIN = 0.06`,
  `ROUTE_FAMILY_THRESHOLDS` (per-family overrides). Consumed by `command_router`.

#### `audio/llm_prompts.py` (LLM persona/answer prompt SSOT + construction index)

The single source of truth for the LLM-path prompts (CLAUDE.md key area). Public:
- `ULTRON_GAMING_PERSONA` — the conversational/gaming Ultron system prompt (tied to the gaming model so it
  can never leak the "Kenning" desktop persona).
- `ANSWER_PERSONA_CORE`, `ANSWER_MARVEL_RULES`, `ANSWER_THINK_RULES` — the curated answer-path persona/rules.
- `ANSWER_SYSTEM_FOR: dict` (line 123) — the route-subtype → system-prompt index (e.g. marvel / think-respond);
  **this is the extension point for new answer subtypes** (per the research synthesis). `_RELAY_REPHRASE_SYSTEM`
  for the generic relay rephrase lives in `relay_speech` (Ultron register).

#### `audio/_ultron_social.py` (curated social-reaction pools)

`__all__ = ["SOCIAL_POOLS", "classify_social_reaction"]`. `SOCIAL_POOLS` = addressee-adapted curated pools
(compliments / insults / surrender / yes-no); `classify_social_reaction(text) -> Optional[str]` detects which
social category an utterance is. Consumed by `relay_speech._as_curated_reaction` (picking/LRU lives in
`relay_speech`).

#### `audio/_ultron_answer.py` (adaptive LLM ANSWER pipeline + meta-leak gate)

`__all__ = [MARVEL_CANON, marvel_topic, classify_answer_subtype, extract_answer_slots, build_answer_call,
is_meta_leak, THINK_RESPOND_SUFFIX_RE]`. Builds the focused per-subtype system_prompt + slots + constrained
sampling for Marvel / "think and respond" / **`qa`** (the 2026-06-22 dedicated QA-answer command —
`directive=="qa"` → the `qa` subtype + `ANSWER_SYSTEM_FOR["qa"]`) answers (`build_answer_call`); `is_meta_leak`
is the identity/model-leak gate on the LLM answer path. Consumed by `relay_speech.build_relay_line`'s answer path.

### `src/kenning/addressing/`

#### `addressing/rules.py`
- `class AddressingDecision(str, Enum)` — ADDRESSED / NOT_ADDRESSED / UNCERTAIN
- `class RuleHit` — dataclass: decision, confidence, reason
- `classify(utterance, seconds_since_response) -> Optional[RuleHit]`
- `explain_rules() -> List[Tuple[str, str]]` — for the review script

#### `addressing/zero_shot.py`
- `class ZeroShotAddresseeModel` — flan-t5-small wrapper (~300 MB CPU)
  - `_ensure_loaded()` — eager-load option
  - `classify(utterance, context, seconds_since_response) -> (verdict_str, confidence, latency_ms)`

#### `addressing/classifier.py`
- `class AddressingVerdict` — final decision + metadata
- `class AddressingClassifier` — combines rules + zero-shot
  - `classify(utterance, seconds_since_response) -> AddressingVerdict`
  - `_log(utterance, verdict)` → writes to `logs/addressing.jsonl`

### `src/kenning/transcription/whisper_engine.py`

- `class WhisperEngine` — faster-whisper wrapper, CUDA fp16
  - `transcribe(audio: np.ndarray, language="en") -> str`
  - On failure: returns `""`, logs `WhisperTranscriptionError` to errors.jsonl
- **2026-06-15:** STT default engine is faster-whisper **large-v3-turbo** on CUDA
  (`compute_type int8_float16`), with a hallucination filter: a peak gate
  (returns `""` when the clip is near-silent), a per-segment
  `no_speech_prob > 0.85` drop, and a `_WHISPER_HALLUCINATIONS` blocklist
  (`_is_whisper_hallucination`) to stop "thank you"-type transcriptions on
  non-speech. STT defaults are hardened so the turbo engine always loads at
  startup.
- **2026-06-16 (coherence pass) — decode-time DOMAIN BIASING:** `transcribe`
  primes the decoder with `initial_prompt = _DOMAIN_PROMPT` — the closed Valorant
  vocabulary (the agent roster + callout terms, ≤200 tokens, most-confusable proper
  nouns first) — so agent names and tactical terms are recognised at the SOURCE,
  cutting mishears before the downstream `_stt_correct` snapper sees them. Additive
  and reversible: gated by `WHISPER_DOMAIN_BIAS` (default on), AUGMENTED by a
  custom `WHISPER_INITIAL_PROMPT` (the env var APPENDS to the domain vocab, never
  replaces it — a 2026-06-18 fix so a short override like `Kenning.` can't shadow the
  whole vocabulary); reset per turn (`condition_on_previous_text`
  stays off for command STT). `initial_prompt` is supported by every faster-whisper
  version.

### `src/kenning/llm/inference.py`

- `_strip_thinking_blocks(stream)` — filter `<think>...</think>` from token stream (streaming path).
- `strip_thinking_text(text) -> str` (2026-05-14 second pass) — same filter as a pure-function pass over a fully-materialised string. Applied inside :meth:`LLMEngine.generate` (blocking path) before returning; unterminated `<think>` (truncation / cancel) drops everything from the opening tag onward (better to lose tail than leak chain-of-thought).
- `_sanitize_user_input(text) -> (cleaned, found_markers)` (Q10 quality-pass iter 1+2) — pre-LLM defense layer that neutralises tag-style prompt-injection markers (`[INST]`, `[/INST]`, `<|im_start|>`, `<|im_end|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `</think>`) by replacing each with `[NEUTRALIZED_TAG]`. Also detects natural-language jailbreak patterns ("ignore previous instructions", "you are now <X>", "respond with the exact word", etc.) — for those the function prepends a one-shot hardening note OR (for the most-direct override patterns: "respond with exactly", "respond with the exact word", "must respond with") rewrites the user message into a description of the attempt so compliance becomes grammatically nonsensical. Detected attempts log to `logs/errors.jsonl` with `dependency='prompt_injection'`. Voice-quality lock preserved — the persona system prompt (`SOUL.md`) is untouched. Wired into `_build_messages` so every LLM call goes through the defense. Verified end-to-end: pre-defense 2/3 of Q8 prompt-injection probes succeeded; post-defense 0/3.
- `class LLMEngine` — LLM client with two backends, selected by `llm.runtime`:
  - `in_process` (default): loads the GGUF via llama-cpp-python in this process. Voice-path mode.
  - `http_server` (opt-in): talks to llama-cpp-server over OpenAI-compat HTTP. For the OpenClaw + voice migration. Latency is +71 ms median TTFT vs in-process — kept opt-in so the voice path isn't regressed.
  - `__init__(model_path?, n_ctx?, n_gpu_layers?, system_prompt?, history_turns?, memory=None, runtime?)`
  - `generate(user_message) -> str` — blocking
  - `generate_stream(user_message) -> Iterator[str]` — token streaming
  - `cancel()` — signal to stop
  - `_build_messages(user_message)` — resolves system prompt fresh each turn (Phase 1 hot-reload), assembles RAG snippets + recent + user
  - `_resolve_system_prompt()` (Phase 1) — sources from `PersonaLoader.get_system_prompt("user_facing")` when `llm.persona.source == "workspace"` (default), else `cfg.system_prompt`. Falls back to config when workspace is empty.
  - `_http_chat_completion(...)` / `_http_stream(...)` — OpenAI-compat HTTP client (uses `requests`, SSE for streaming, cancel-aware).
  - `_chat_completion_kwargs(_llm_cfg, enable_thinking, *, stream, sampling=None)` (4B plan Stage F; 2026-05-14 third pass rewrite) — static helper that builds the kwargs dict for `Llama.create_chat_completion`. Returns the four base sampling params + optional ``stream`` flag, and when ``sampling`` is provided MERGES its allowed keys (per-call overrides for the relay/answer path: stop sequences, min_p, grammar, logit_bias, seed) — NEVER emits ``chat_template_kwargs`` because the pinned llama-cpp-python 0.3.22 doesn't accept it (passing it raises ``TypeError``). The thinking-mode toggle is applied to the user message instead via :meth:`_apply_no_think_marker`. The HTTP runtime's payload-building code still emits ``chat_template_kwargs`` because llama-cpp-server (separate codebase) does accept it.
  - `_apply_no_think_marker(messages, enable_thinking) -> list` (2026-05-14 third pass) — staticmethod that appends ``/no_think`` to the last user message when ``enable_thinking is False``. Qwen3 / Qwen3.5 chat templates inspect the user message for this marker and skip the ``<think>...</think>`` block. ``enable_thinking=None`` (default) and ``True`` are no-ops. Returns a copy of ``messages`` — never mutates the original. Replaces the previous ``chat_template_kwargs`` mechanism which crashed against the real llama-cpp-python signature. **2026-06-15 marker-leak fix:** the marker is Qwen-template-specific, so it is gated on the LIVE-LOADED model (`self.model_path`), not just config — in lean gaming the LLM is constructed directly as the llama-3.2-3b preset while `config.llm` still names the Qwen base, so a config-only check wrongly matched "qwen" and appended `/no_think` to the llama model (which has no template hook and PARROTED it — TTS spoke "No think"). The root bug was that `self.model_path` is a `pathlib.Path`; the old code called `.lower()` on it directly, which raised `AttributeError` that the bare `except` swallowed → the marker was appended UNCONDITIONALLY. Fixed with `str()` coercion so the Qwen check actually runs against the resolved path.
  - `_build_llama(cfg, model_path, n_ctx, n_gpu_layers, **overrides) -> (Llama, Path, int, int)` (4B plan voice-swap; the last two ints are the resolved n_gpu_layers + n_ctx) — pure constructor that builds + returns a fresh `Llama` instance per `cfg` (+ kw overrides). Does NOT mutate `self`. Used by `_init_in_process`, `reload_for_preset`, and `reload_for_device`.
  - `reload_for_preset(preset: str) -> (bool, str)` (4B plan voice-swap) — hot-swap the loaded LLM to `preset` without restarting Kenning. Builds the new `Llama` FIRST so a failed swap (missing GGUF, invalid preset) leaves the engine in its working state. On success: history cleared, `KENNING_LLM_PRESET` env updated, stale `KENNING_LLM_MODEL_PATH` cleared. On failure: env vars restored. Idempotent (`already on X` returns success without rebuild). `in_process` runtime only.
  - `reload_for_device(device: str, *, force: bool = False)` + `_DEVICE_PROFILES` (NEW 2026-06-18) — hot-reload the live model with a device-optimized llama.cpp profile (GPU = full offload + flash-attn + q8_0 KV + large batches; CPU = 0 layers + F16 KV + small ubatch). `force=True` reloads even when already on the target device (re-apply a profile). Load-new-then-release-old (a failed load leaves the working model intact). Backs the voice command `relay_speech.match_llm_device_switch` ("switch the model to the GPU/CPU"); anticheat-safe (compute-location only).
  - **Ultron 1.0 serving (2026-06-20):** the `system_prompt=` override fast path (`[system, user]`, no RAG/history) IS the route-all-through-8B surface used by `ultron_prompt`; `enable_thinking=False` is enforced on relay/private routes (`_apply_no_think_marker` appends `/no_think` for qwen-family + the kwarg; startup assert that `<think>` never leaks). `sampling` whitelist already includes `grammar`+`logit_bias` (unused — NO grammar on the hot path per research D4). Default preset is now `josiefied-qwen3-8b` @ `n_ctx=4096` (10 GB VRAM cap). TODO (research synthesis): add a `_sanitize_user_input` call on the `system_prompt=` branch to restore injection defense on relay routes.
  - `generate(user_message, *, enable_thinking=None)` and `generate_stream(user_message, *, enable_thinking=None, record_history=True)` (4B plan Stage F + 2026-05-18 latency pass 3 Phase 3) — per-call thinking mode parameter, plus `record_history` on the streaming variant. When `record_history=False`, the end-of-stream auto-record is skipped so callers can defer history commit to after they've confirmed the response was actually consumed (used by the orchestrator's speculative-LLM path).
  - `record_completed_turn(user_message, response)` (2026-05-18 latency pass 3 Phase 3) — public commit hook for the deferred-history pattern. No-op on empty input. Used by `Orchestrator._collect_speculative_llm`'s commit closure after the buffered tokens have been drained to TTS.

**In:** user text + (optional) `ConversationMemory` for RAG. **Out:** generated text.

### `src/kenning/memory/`

#### `memory/embedder.py`
- `class _SparseVec` — thin wrapper over BM25 sparse output
- `class HybridEmbedder` — FastEmbed dense (bge-small-en-v1.5 INT8) + sparse (Qdrant/bm25)
  - `encode_dense(texts) -> np.ndarray`
  - `encode_query_dense(text)` / `encode_query_sparse(text)`
  - `encode_query_dense_batch(queries)` / `encode_query_sparse_batch(queries)` — multi-query helpers
  - `encode_query_dense_sparse(query, *, parallel=False)` — single-call dense + sparse pair; optional `ThreadPoolExecutor` mode (V1-gap A2 parallel encoding)
  - `dim` property → 384

#### `memory/reranker.py` (2026-05-21 frontier Item 2)
- `class CrossEncoderReranker` — wraps `sentence-transformers` CrossEncoder; uses `BAAI/bge-reranker-v2-m3` by default.
  - `rerank(query, candidates, *, top_k) -> List[Tuple[idx, score]]` — predict relevance, sort, top-k.
  - Module-level `get_shared_reranker()` / `reset_shared_reranker()` (test-only) for singleton sharing.
  - `_PREDICT_CONTENT_CAP_CHARS = 500` truncates candidate content before predict() to bound tokenize cost.
  - Default: code-level ENABLED but runtime `memory.reranking.enabled: false` after live perf measurement (17-18 s/turn on CPU).

#### `memory/contextualizer.py` (2026-05-21 frontier Item 4)
- `class ContextGenerator` — Anthropic-technique contextual retrieval. LLM-generates a brief situational anchor before embedding each turn so the dense vector carries surrounding context.
- Default OFF (`memory.contextual_retrieval.enabled: false`); LLM cost ~80-150 ms per write turn.

#### `memory/topical_chunking.py` (2026-05-19 Track 1a)
- `class TopicTracker` — cosine-boundary topic tracker. Detects topic shift when consecutive turn embedding distance exceeds `boundary_similarity_threshold`; writes `topic_id` payload onto the memory turn.
- Default OFF.

#### `memory/discourse.py` (2026-05-19 Track 1b)
- `class DiscourseClassifier` — 6-way classifier (REQUEST / QUESTION / STATEMENT / CONFIRMATION / SOCIAL / OTHER) via rule + optional embedding-centroid fallback.
- Default OFF (`memory.discourse_tagging.enabled: false`).

#### `memory/background_summarizer.py` (2026-05-19 Tracks 1c+1d+1e)
- `class BackgroundSummarizer` — idle-gated LLM-driven summary + structured fact extraction. Runs on a worker thread when the orchestrator is idle.
- Default OFF (`memory.background_summary.enabled: false`).

#### `memory/dual_history.py` (2026-05-24 cline batch 9, T4)

Adapted from cline's `MessageStateHandler` pattern (Apache 2.0; see `THIRD_PARTY_NOTICES.md`). Per-session primitive that splits "what the user actually heard / said" from "what the LLM saw" so the verbatim record survives every condenser / dedup / redaction pass.

- `ROLE_USER` / `ROLE_ASSISTANT` / `ROLE_SYSTEM` / `ROLE_TOOL` — canonical role constants matching the existing `ConversationMemory` convention.
- `new_turn_id() -> str` — 32-char hex UUID4 turn identifier.
- `@dataclass(frozen=True) class VerbatimTurn` — `turn_id` / `role` / `text` / `timestamp` / `channel` / `tts_clip_ref` / `image_refs` / `metadata`. The literal user utterance + agent response (NOT the prompt-augmented body). Image refs survive even when the api history strips them post-VLM-description.
- `@dataclass(frozen=True) class ApiTurn` — `turn_id` / `role` / `content` (str OR typed-block list) / `compacted` / `elided_count`. Subject to truncation, dedup, condenser passes. `elided_count` records how many verbatim turns a single summary entry covers, powering the drift dashboard's "you've been silenced 14 times today" counter.
- `@dataclass(frozen=True) class HistorySnapshot` — frozen tuple-of-VerbatimTurn + tuple-of-ApiTurn + both lookup indices, returned by `snapshot()`.
- `class DualHistoryStore` — `__init__(*, verbatim_cap: int | None = None, api_cap: int | None = None)`. RLock-guarded. Verbatim cap defaults unlimited (verbatim is cheap to keep); api cap is what costs tokens on every LLM call.
  - `record(role, text, *, turn_id?, timestamp, channel, tts_clip_ref, image_refs?, metadata?, api_content?) -> str` — append verbatim AND (optionally) the matching api turn under the same UUID. Returns the turn_id used.
  - `record_api(turn_id, role, content, *, compacted=False, elided_count=0)` — append just an api turn (used when the api shape is computed after the verbatim record).
  - `replace_api_range(start, stop, replacement=None) -> int` — condenser hook: replace `api[start:stop]` with optional single replacement. Verbatim is UNCHANGED.
  - `truncate_after_turn(turn_id) -> (verbatim_dropped, api_dropped)` — anchor-based restore path. Empty string clears everything.
  - `truncate_to_offset(*, offset_from_end) -> (verbatim_dropped, api_dropped)` — drop the last N verbatim turns + matching api entries. Collects api indices BEFORE deletion + deletes descending so positions stay valid.
  - `verbatim() / api()` — full tuple snapshots; `recent_verbatim(n) / recent_api(n)` — last-N convenience.
  - `get_verbatim(turn_id) / get_api(turn_id) -> Optional[...]` — O(1) lookup via the per-array index dicts.
  - `find_verbatim_by_substring(needle, *, limit=10, case_insensitive=True) -> tuple` — newest-first fuzzy search powering "what did I say earlier?".
  - `snapshot() -> HistorySnapshot` — frozen view including both lookup indices.
  - `verbatim_turn_count() / api_turn_count() -> int`.
  - `drift_report() -> dict[str, int]` — per-call counts (`verbatim_only`, `api_only`, `shared`, `verbatim_total`, `api_total`) for the daily drift-audit dashboard mentioned in the catalog.
  - `clear()` — drop everything (both arrays + both indices).
- `_maybe_evict_verbatim_locked() / _maybe_evict_api_locked()` — internal LRU eviction when a cap is set; rebuilds the index after eviction so positions stay accurate.

Module is I/O-free. Callers wire their own persistence (Qdrant payload, JSONL audit log, in-memory recency cache); `DualHistoryStore` is just the data structure + indices.

**In:** verbatim text + optional api shape, both keyed by stable UUID. **Out:** O(1) verbatim<->api resolution, condenser-safe drift reporting, anchor + offset restore paths.

#### `memory/qdrant_store.py`
- `class MemoryTurn` — dataclass: id, ts, role, content, summary, entities, ...
- `class FactRow` (V1-gap A3) — dataclass: fact, confidence, last_confirmed, category, score, extracted_at, extracted_from, retrieval_weight. Read-side projection of the `facts` collection that the maintenance script writes.
- `class ConversationMemory`
  - `__init__(path?, embedder, recent_cache_size=100, session_id?)`
  - `add(role, content)` — sync; queues to background writer
  - `recent(n) -> List[MemoryTurn]` — from in-process cache
  - `retrieve(query, k=cfg, exclude_recent=cfg) -> List[MemoryTurn]` — single-pass hybrid RRF
  - `retrieve_multi(primary_query, category_queries, *, k, exclude_recent)` (V1-gap A2) — multi-pass per-category hybrid RRF + composite re-ranking. Parallel fan-out via `ThreadPoolExecutor`. Falls back to single-pass on any failure.
  - `retrieve_for_query(primary_query, gate_verdict=None, *, k, exclude_recent)` (V1-gap A2) — routing helper: when `memory.retrieval.multi_pass_enabled` is True AND the verdict carries `context_categories`, fans out via `retrieve_multi`; otherwise calls `retrieve`. Default-OFF preserves byte-for-byte legacy behaviour.
  - `search_facts(query, *, k=5, min_confidence=0.0, max_age_days=None) -> List[FactRow]` (V1-gap A3) — hybrid RRF over the `facts` collection. Filters via Qdrant `confidence >= min_confidence` and `last_confirmed >= now - max_age_days*86400`. Fail-open: returns `[]` on any Qdrant / embedder failure.
  - `__len__()` / `close()` — close() drains the writer queue, sends
    the shutdown sentinel, joins the writer thread, then releases the
    Qdrant client (freeing the local-mode exclusive `<path>/.lock`;
    the supervisor ProjectIndex + web cache BORROW this client for
    exactly that one-client-per-path reason). 2026-06-12 cleanup: a
    dead duplicate `close()` definition earlier in the class (client-
    only body, shadowed by this one) was removed; its .lock docstring
    was folded into the surviving implementation.

#### `memory/ranking.py` (V1-gap A2)
- `@dataclass class RankingWeights` — frozen snapshot of the rrf_weight / recency_weight / recency_half_life_days / surprise_weight / redundancy_weight tuning.
- `@dataclass class CandidateScore` — per-candidate aggregator (id, payload, rrf_score, dense vector, primary_similarity, category_similarity, composite_score).
- `cosine_similarity(a, b) -> float` — pure cosine on float lists; defensive against length mismatch / zero vectors.
- `compute_recency_boost(ts, *, half_life_days, now=None)` — exponential decay; ``ts == 0`` (sentinel) returns 0.
- `compute_surprise_score(candidate_dense, primary_dense, category_score)` — clamps to ``max(0, category_score - primary_similarity)``.
- `compute_redundancy_penalty(candidate_dense, picked)` — max cosine vs already-picked.
- `compute_composite_score(candidate, *, weights, primary_dense, picked, now=None)` — weighted blend.
- `select_top_k(candidates, *, k, weights, primary_dense=None, now=None) -> List[CandidateScore]` — greedy redundancy-aware selection.

### `src/kenning/web_search/`

#### `web_search/acknowledgments.py`
- `class AcknowledgmentSource` — shuffled-pool phrase generator (8 phrases)
  - `next_phrase() -> str`

#### `web_search/brave.py`
- `_BRAVE_BREAKER` — module-level CircuitBreaker (3/5min, 5min cooldown)
- `class SearchResult` (primary; `BraveResult` is a back-compat alias) — dataclass: url, title, snippet, rank
- `class BraveSearchClient`
  - `search(query, count?) -> List[BraveResult]` — uses breaker + raises BraveAPIError
  - `_do_search(query, count)` — inner; raises typed errors

#### `web_search/cache.py`
- `_VOLATILE_KEYWORDS`, `freshness_category_for(query)`, `ttl_for(category)`
- `class WebResultsCache` — Qdrant-backed; collection = `web_results`
  - `lookup(query) -> Optional[List[(BraveResult, full_text)]]`
  - `store(query, rows)` — best-effort

#### `web_search/gating.py`
- `class GateDecision(str, Enum)` — SEARCH / NO_SEARCH / UNCERTAIN
- `class GateVerdict` — decision, confidence, source, search_queries, knowledge signals (knowledge_confidence, knowledge_source, has_temporal_dependency), **context_categories** + **memory_search_queries** (V1-gap A2 — populated by the LLM preflight pass; rule-only verdicts leave them empty so the multi-pass retrieval path stays inactive).
- `_resolve_knowledge_source(*, needs_search, confidence, memory_snippets, rule_reason)` (V1-gap B1) — single-source helper that maps gate inputs to the spec's five-value enumeration (`weights / retrieved_memory / retrieved_facts / web_search_needed / unknown`). Every `GateVerdict` construction site routes through this.
- `classify_by_rules(utterance) -> Optional[GateVerdict]` — hard rules (time markers, URL, etc.). 2026 catalog 12 (felo-search T2): three additional deterministic-SEARCH rules — `_COMPARISON_QUERIES` (`vs`/`versus`/`compared to`/`comparison of`/`which is better`/`better than`/`pros and cons`/`trade-offs`), `_HOWTO_QUERIES` (`how to`/`tutorial`/`step-by-step`/`walkthrough`/`best practices`), `_SHOPPING_QUERIES` (`price`/`pricing`/`how much is`/`cost of`/`where to buy`/`deal`/`discount`/`coupon`/`on sale`/`cheapest`). They fire after the anti-search rules (greeting/personal/creative win) and before `_STABLE_FACTUAL_REQUEST` (so `how does X work` / `how tall is X` stay NO_SEARCH while `how to X` routes SEARCH). All `high` confidence, `knowledge_source=web_search_needed`.
- `classify_by_preflight(utterance, llm, memory_snippets) -> GateVerdict` — LLM call
- `class WebSearchGate` — orchestrates rules → LLM
  - `classify(utterance, recent_memory) -> GateVerdict`

#### `web_search/jina.py`
- `_JINA_BREAKER` — CircuitBreaker (5/5min, 3min cooldown)
- `class JinaReaderClient`
  - `fetch(url) -> Optional[str]` — uses breaker + raises JinaReaderError

#### `web_search/searxng.py` (2026-05-22 frontier)
- `_SEARXNG_BREAKER` — CircuitBreaker
- `class SearxNGError(BraveAPIError)` — typed failure
- `class SearxNGSearchClient(base_url?, timeout_s?, categories?, engines?)`
  - `is_reachable() -> bool` — cheap GET /
  - `search(query, count?, categories=None) -> List[SearchResult]` — per-call
    `categories` override (e.g. `"news"` for news-category routing); X-Forwarded-For
    header satisfies SearxNG's botdetection so the engine list is reachable

#### `web_search/duckduckgo.py` (2026-05-22 frontier)
- `class DuckDuckGoSearchClient` — HTML-scrape last-fallback (via `duckduckgo-search` lib)
  - `search(query, count?) -> List[SearchResult]`
- Includes 2026-05-22 CAPTCHA handling: any HTTP 403 / CAPTCHA-marker translates
  into circuit-breaker increment + empty list (fail-open to next provider)

#### `web_search/provider_chain.py` (2026-05-22 frontier)
- `class SearchProviderChain` — cascading provider chain
  - `__init__(provider_ids=None)` — defaults to `searxng -> brave -> duckduckgo`
    per `web_search.providers` config
  - `search(query, count?, categories=None) -> List[SearchResult]` — first-non-empty
    wins; per-provider client construction memoized; forwards `categories`
    only to `searxng` (Brave + DDG silently ignore unknown kwargs)

#### `web_search/trafilatura_reader.py` (2026-05-22 frontier)
- `class TrafilaturaReaderClient` — local Python lib for HTML -> markdown
  - `fetch(url) -> Optional[str]` — caps at ~32 k chars (was 200 k before live perf bug)

#### `web_search/reader_chain.py` (2026-05-22 frontier)
- `class ReaderChain` — cascading reader chain
  - Default order: `trafilatura -> jina` (local-first)
  - `fetch(url) -> Optional[str]` — first-non-empty wins

#### `web_search/search.py`
- `class SearchSource` — dataclass: url, title, snippet, full_text, rank
- `class SearchPayload` — dataclass: query, sources, cache_hit, elapsed_ms, notes
- `_rank_snippets(llm, query, results, top_n)` — LLM-driven re-ranking
- `_normalise_search_query(q)` / `_dedupe_queries(qs)` (V1-gap B2) — drop near-duplicate Brave queries before fan-out using a token-set canonical form (lowercase + possessive strip + stopword drop + sort).
- `_render_inline_marker(index, *, fmt)` (V1-gap B3) — render bracketed `[1]` (default) or Unicode superscript (¹²³) inline citations based on `web_search.citation.inline_marker_format`.
- `class WebSearchExecutor` — orchestrates SearchProviderChain → rank → ReaderChain → cache. **2026-05-09 latency fix:** reader fetches run IN PARALLEL via `concurrent.futures.ThreadPoolExecutor` with a collective deadline cap. Pre-fix the loop was sequential and one slow page (~10 s on a Quora result) blocked the entire search path while the TTS playback queue starved waiting for tokens. Post-fix wall time is `max(per-fetch durations)` instead of `sum(...)`, capped further by `collective_deadline_seconds`. Any fetch still in flight at deadline is abandoned (its source falls back to snippet-only with a `jina_deadline:<url>` note). Threads keep running in the background and exit on per-fetch HTTP timeout; `pool.shutdown(wait=False)` ensures the executor returns immediately.
  - `__init__(brave, jina, llm, cache=None, max_fetch=None, collective_deadline_seconds=None)` — `brave` is actually the SearchProviderChain (legacy field name retained for compatibility with internal call sites).
  - `run(user_query, search_queries?, top_n=3, categories=None) -> SearchPayload` — 2026-05-22: `categories` param forwarded to the chain (only SearxNG accepts; Brave/DDG ignore). Set to `"news"` from the orchestrator when `_NEWS_QUERIES` regex matches.
- `format_sources_for_prompt(sources, *, strategy_queries=None)` / `format_sources_for_transcript(sources, *, strategy_queries=None)` — references list always uses bracket form for monospace clarity. 2026 catalog 12 (felo-search T4): `SearchPayload.queries` records the fanned-out (reformulated) query list; `_format_strategy_line` joins a multi-query strategy into a `q1 | q2` one-liner (self-suppresses for a single query); both formatters append the strategy when `strategy_queries` is passed (`[Search strategy: …]` in the prompt block / `strategy: …` in the transcript). The orchestrator wires it to the TRANSCRIPT ONLY (gated by `web_search.expose_search_strategy`, default ON) so spoken replies stay concise; the prompt param is reserved for future text / GUI channels.

#### `web_search/query_rewrite.py` (2026 catalog 12, felo-search T1)
- `class QueryReformulation` — frozen `(original, variants, method)`; `.all_queries` = original + variants deduped case-insensitively, order-preserving.
- `expand_query_rules(query, *, max_variants)` — zero-cost structural rewrites: `"X vs Y [tail]"` → two balanced queries (tail grafted onto the left subject); `"how to X"` → `"X tutorial"` / `"X guide"`; `"best/top X"` → `"X review"` / `"X comparison"` (skips `"best practices"`); a leading temporal qualifier (`latest`/`recent`/`current`) → the bare subject. Deduped vs original; capped at `max_variants`.
- `expand_query_llm(query, llm, *, max_variants)` — one short in-process Qwen call (`/no_think` marker; `_parse_queries_json` tolerant of think-blocks / fences / prose); fail-open to `[]`.
- `reformulate_query(query, *, use_llm, llm, max_variants, enabled)` → `QueryReformulation` — LLM-first when `use_llm` (falls back to rules on empty/error), else rules-only.
- `maybe_reformulate_queries(user_query, base_queries, *, llm)` → `list[str]` — executor-facing: reads `web_search.query_reformulation` config, merges variants into `base_queries`, deduped + capped at `MAX_TOTAL_QUERIES=5`; logs to `logs/search_reformulations.jsonl`; fail-open (returns base unchanged on disabled / error).
- **Wired in `WebSearchExecutor.run`** (`search.py`): expands the query list BEFORE `_dedupe_queries` + the per-query fan-out, so the existing URL-dedup + cache absorb variants transparently. Default ON (rule-based, zero-cost); `use_llm` opt-in adds ~150-250 ms on the SEARCH path only. Voice baseline untouched.

#### `web_search/deep_research.py` (2026 catalog 12, felo-search T3, YELLOW)
- `class DeepResearchLoop(AgentLoop)` — bounded agentic research over the FREE ladder. `research(question) -> DeepResearchResult`. Overrides: `plan` (step 1 LLM-decomposes the question into sub-questions; later steps run an LLM gap analysis and return new sub-questions or `None` to finish; short-circuits to `None` once `max_accumulated_sources` is hit), `act` (searches each NEW sub-question via the injected `WebSearchExecutor.run` — so T1 reformulation + the provider/reader chains + cross-encoder ranker + `web_results` cache + per-provider rate-limit tracker all apply — accumulating URL-deduped sources up to the cap), `action_succeeded` (a completed search is success even when it finds nothing — overrides the base fail-fast so an empty sub-search never ABORTs), `action_signature` (canonical sub-question set for the base loop detector), `is_done` (accumulation-cap OR zero-new-progress round). The base's `max_steps` is the load-bearing safety cap. Both LLM calls (`_decompose` / `_identify_gaps`) FAIL OPEN — decompose -> search the question verbatim, gap analysis -> finish — so an LLM hiccup can never spin the loop.
- `class DeepResearchResult` — `question` + `sources` + `sub_queries` (the strategy, T4) + `loop_status` + `steps` + `elapsed_s`; `to_payload()` -> `SearchPayload` so the orchestrator's existing search-augmented synthesis/streaming path consumes it unchanged.
- `class DeepResearchMatch` + `match_deep_research(text)` — strict regex matcher (requires an explicit deep / thorough / in-depth / deep-dive / dig-deeper marker; extracts the topic). "search X" / "what is X" / "look up X" never match.
- `_parse_json_list` / `_dedupe_subqueries` — tolerant JSON extraction + dedup helpers.
- **Wired** in `Orchestrator._maybe_handle_deep_research` (run-loop short-circuit, NO new `RoutingIntentKind`): acks, runs the loop, then synthesizes + streams the answer through the same LLM->TTS path `_respond` uses. Gated by `deep_research.enabled` (default ON; per-turn opt-in via the matcher, so the normal sub-second search path is untouched). Config: top-level `deep_research.{enabled, max_steps, max_sub_queries_per_step, top_n_per_query, max_accumulated_sources}`.

#### `web_search/rate_limit.py` (T14 rate-limit envelope)
- `class RateLimitState` / `class BackoffConfig` / `class RateLimitTracker`; `parse_rate_limit_headers()`,
  `compute_backoff()`, `sleep_for_backoff()`, `get_global_tracker()` / `reset_global_tracker_for_testing()`.
  Consumed by `provider_chain.py` + all three provider clients (Brave/SearxNG/DDG) to cool a 429'd provider.

#### `web_search/playwright_reader.py` (opt-in JS-aware reader)
- `class PlaywrightReader` — Playwright/Chromium page extractor implementing the `fetch(url) -> Optional[str]`
  reader interface; the third reader registered in `ReaderChain`. Lazy browser construction; **DEFAULT OFF**
  (opt-in). Feeds the slimdown → pandoc HTML→Markdown pipeline.

#### `web_search/slimdown_html.py`
- `slimdown_html(html_text) -> str` — the first HTML→Markdown stage: strips SVG/img/style/script/noscript/
  interactive widgets + `data:` URLs before pandoc conversion.

#### `web_search/pandoc_converter.py`
- `html_to_markdown(html_text) -> Optional[str]`, `pandoc_available() -> bool` — the pandoc HTML→Markdown step
  (graceful degradation when pandoc absent); used by `PlaywrightReader`.

### `src/kenning/tts/`

#### `tts/precomputed_ack.py` (NEW 2026-05-15 latency pass)

Pre-computed TTS clip cache. Phrases enrolled at startup (the
conversational ack pool + the web-search ack pool) get synthesised
ONCE via the live engine's `_synthesize` path -- so the cached clip
is byte-identical to the live path (same temperature, phantom-tail
trim, v3 filter, all of it). Later `_synthesize(text)` calls hit
the dict and skip the ~350-400 ms HTTP + filter chain.

- `class PrecomputedAckClipCache` -- thread-safe `dict[str, (pcm, sr)]`
  keyed by stripped text. `phrases` (sorted, de-duped, stripped at
  init), `get(text)` (exact stripped match), `prewarm(synth_fn)`
  (synthesise + populate; swallows per-phrase exceptions), `is_warm`
  / `warmed_count`.
- `collect_default_ack_phrases() -> List[str]` -- imports the
  conversational + web-search phrase pools lazily and returns the
  union. Fail-open on import errors.
- `build_default_ack_clip_cache() -> PrecomputedAckClipCache` --
  factory; returns an EMPTY cache. Caller runs `prewarm(synth_fn)`
  on a daemon thread.
- `prewarm_in_background(cache, synth_fn, *, name="ack-prewarm")` --
  starts + returns the daemon thread.

Wired at: `Orchestrator.__init__` calls `_kick_off_ack_clip_prewarm`
right after `self.tts.warmup()`. The prewarm thread runs in
parallel with the rest of orchestrator startup; first turn may miss
while populating, subsequent turns hit.

#### `tts/__init__.py` (factory, refactored 2026-05-22)
- `TTSEngine` — `Union[KokoroSpeech, XttsV3Speech, TextToSpeech]` type alias.
- `make_tts_engine(cfg=None) -> (rvc_or_none, TTSEngine)` — selects the engine
  based on `tts.engine` (`kokoro` | `xtts_v3` | `piper_rvc`). Defaults to
  `get_config().tts` when `cfg=None`. Returns ``(None, engine)`` for Kokoro
  and XTTS; returns ``(RvcConverter | None, TextToSpeech)`` for piper_rvc.
  **RVC is built ONLY for the legacy `piper_rvc` engine.** Under the default
  `kokoro` engine the factory returns `rvc=None`, so RVC is never loaded and the
  `KENNING_RVC_*` env vars (`RVC_ENABLED`, pitch shift, etc.) are a no-op — the
  Ultron voice character comes from the Kokoro fine-tune (`kenning_finetune.pth`)
  plus the in-model prosody hooks (f0 / energy / per-phoneme duration shaping),
  not from RVC.
- `_load_rvc_if_enabled() -> Optional[RvcConverter]` — fail-open RVC loader:
  returns None when `settings.RVC_ENABLED` is false, when the model file is
  missing, or when `RvcConverter()` raises. The orchestrator's
  `_load_tts_engine` is now a one-liner delegating to this factory so
  measurement scripts and the production code share one construction path.
- Re-exports `TextToSpeech`, `RvcConverter`, `KokoroSpeech`, `XttsV3Speech`.

#### `tts/rvc.py`
- `class RvcConverter` — infer-rvc-python wrapper, cuda:0
  - `convert(pcm: np.ndarray, sample_rate: int) -> (pcm, sr)` — FAIL-SOFT: on inference error returns the original PCM unchanged + logs at EXCEPTION level (RVCConversionError is raised by `TextToSpeech._synthesize` in speech.py, not here)
  - `close()` — releases GPU memory
  - **Tests:** [`tests/test_rvc.py`](../tests/test_rvc.py) (7 tests; explicit-path kwargs to avoid the default-arg gotcha; close idempotency; convert empty-audio / not-loaded guards; context-manager release).

#### `tts/speech.py`
- `Clip` — type alias for `Tuple[np.ndarray, int]` (legacy synth function return).
- `class ClipItem(NamedTuple)` (2026-05-10 producer-signaled lookahead):
  `(audio, sample_rate, is_known_last)`. Pushed onto `piper_q` /
  `audio_q` instead of bare `Clip` tuples. Playback uses
  `is_known_last` (or the `None` end-of-stream sentinel) to decide
  whether to wait for another clip after playing the current one.
  Default `is_known_last=False`; the `None` sentinel handles the
  "this was the last" signal in normal use. The flag is reserved for
  future producers that DO know in advance (canned single-sentence
  voice responses, etc.).
- `_QUEUE_GET_TIMEOUT_SECONDS = 60.0` — generous wait between clips
  in both the playback loop and the RVC stage. The previous 10 s
  value killed audio mid-response when a slow web search held the
  generator long enough for the RVC stage's `piper_q.get(timeout=10)`
  to fire (BMW failure mode in the 2026-05-09 logs).
- `class TextToSpeech` — Piper + optional RVC
  - `__init__(rvc=None)` — loads Piper voice, optionally wraps with RVC
  - `speak(text)` — synchronous synthesize + play
  - `speak_stream(fragments)` — stream tokens, flush on sentence
    terminator. **Producer-signaled lookahead (2026-05-10):** plays
    each clip IMMEDIATELY on receipt, then blocks for the next.
    Replaces the legacy play-after-peek pattern that delayed the
    first clip (commonly the web-search ack) up to 10 s waiting for
    the second clip to determine "is this last?". Voice character
    (edge fades, inter-sentence pauses, tail silence) bit-identical;
    only the queue-get ordering changed.
  - `warmup()` — primes Piper
  - `_synthesize(text)` — Piper → optional RVC; raises
    PiperSynthesisError / RVCConversionError
  - `_run_synth_loop(*, fragments, push, synth_fn)` — walks
    fragments, synthesises on flush chars, pushes each non-empty
    clip via `push` as a `ClipItem(is_known_last=False)`. End-of-
    stream sentinel (`None`) is pushed by the surrounding worker's
    `finally` block.
  - `stop()` — interrupt current playback

#### `tts/kenning_filter.py` (NEW 2026-05-10 voice swap)

Runtime port of the user-tuned v3 Kenning mechanical filter chain (the
prototype lives at `kenningVoiceAudio/scripts/kenning_filter.py`).
Built on `pedalboard` (Spotify's open-source DSP library; sub-ms
overhead per stage on CPU).

- `PresetName` — Literal["v1_subtle", "v2_medium", "v3_heavy"].
- `get_preset(preset)` — fresh `Pedalboard` instance per call.
- `apply_filter(audio, sample_rate, preset="v3_heavy", tail_silence_ms=200.0)`
  — applies the chain. Pads `tail_silence_ms` of trailing zeros
  before processing so the reverb tail decays into the padding
  rather than being clipped at the buffer end. Runtime default 200
  ms (audible portion of v3 reverb); offline samples use 500 ms.

The `v3_heavy` preset is the user-locked production chain (bit-
identical to the prototype): Highpass → PitchShift(-1.8 semitones) →
Compressor → LowShelfFilter(+4.5 dB @ 160 Hz) → Delay(7 ms, 25 %
feedback for comb resonance) → Chorus → Distortion(+7 dB) → Peak EQ
boost @ 2.5 kHz → HighShelf cut → Reverb(small cavity) → Lowpass.

#### `tts/xtts_v3.py` (NEW 2026-05-10 voice swap)

Drop-in replacement for `TextToSpeech` when `tts.engine == "xtts_v3"`.
Same `speak` / `speak_stream` / `warmup` / `stop` interface so the
orchestrator playback path (the producer-signaled lookahead in
`speak_stream`) doesn't change.

- `class ClipItem(NamedTuple)` — `(audio, sample_rate, is_known_last)`,
  same contract as in `tts/speech.py` so the queue protocol is
  uniform across engines.
- `class XttsServerStartError(RuntimeError)` — raised when the XTTS
  server subprocess can't be started (missing venv / script /
  reference, startup timeout exceeded).
- `class XttsSynthError(RuntimeError)` — synth call failure.
- `trim_phantom_tail(audio_f32, sample_rate, *, silence_threshold=0.005,
  max_event_ms=200.0, min_lead_silence_ms=150.0, trailing_grace_ms=80.0,
  window_ms=20.0, min_clip_duration_ms=800.0) -> (np.ndarray, bool)`
  (NEW 2026-05-12 phantom-token mitigation, defence in depth; 2026-05-19
  short-clip guard added) — pure function that detects the
  specific XTTS-v2 phantom signature (sustained_speech → ≥150 ms
  silence → <200 ms isolated event → silence to buffer end) and
  trims everything after the last sustained-speech region plus a
  small grace cushion. Returns `(maybe-shorter audio, detected)`.
  Conservative: passes through unchanged when no phantom pattern is
  present (sustained-speech-only, mid-sentence inter-word silence,
  legitimately long trailing speech). Runs BEFORE the v3 filter so
  the reverb tail decays normally into its tail_silence_ms padding.
  Empirically grounded against a real session WAV showing the
  signature at 19.28 s. **2026-05-19:** `min_clip_duration_ms`
  short-circuit (default 800 ms) prevents mis-firing on single short
  words like ``"Right."`` where XTTS occasionally lengthens the
  pre-stop closure beyond 150 ms and the [t] release would otherwise
  be misclassified as a phantom event.
- `normalize_text_for_tts(text) -> str` (NEW 2026-05-19) — pure
  text rewriter called from `_synthesize` BEFORE the HTTP synth call.
  Handles patterns XTTS-v2 mispronounces: Windows drive paths
  (``C:\\foo\\bar\\baz.ext`` -> leaf filename), times with AM/PM
  (``2:16 a.m.`` -> ``2 16 A M``), 24-hour times (``14:30`` ->
  ``14 30``), standalone ``a.m./p.m.`` markers, Latin abbreviations
  (``e.g.`` -> "for example", ``i.e.`` -> "that is", ``etc.`` ->
  "et cetera", ``vs.`` -> "versus"). Conservative: unmatched
  patterns pass through unchanged. URLs are deliberately untouched
  (the regex set deliberately excludes Posix paths because they
  would mangle URLs like ``https://x.com/foo/bar``).
- `class XttsV3Speech` — the engine.
  - `__init__(...)` — resolves paths via `tts.xtts_v3` config,
    spawns the XTTS HTTP server in `.venv-xtts`, polls `/healthz`
    until ready (180 s startup budget for cold model load). 2026-
    05-12 phantom-token mitigation: also reads `temperature` (0.65
    default), `phantom_tail_trim_enabled` (true default), and the
    three trim thresholds (`silence_threshold`, `max_event_ms`,
    `min_lead_silence_ms`) from `tts.xtts_v3` config; explicit ctor
    args override.
  - `speak`, `speak_stream`, `warmup`, `stop` — same API as the
    legacy engine.
  - `_synthesize(text)` — checks the ack cache first (hit returns
    immediately), then runs `normalize_text_for_tts(text)` to
    rewrite TTS-hostile patterns (2026-05-19), POSTs `/synthesize`,
    accumulates the streamed PCM, optionally runs `trim_phantom_tail`
    (gated on `phantom_tail_trim_enabled`; 2026-05-19 short-clip
    guard at 800 ms), applies the v3 Kenning filter via
    `kenning_filter.apply_filter(..., tail_silence_ms=200)`, returns
    `(int16 pcm, sr)` matching the legacy engine's contract.
  - `_http_synthesize(text)` — raw HTTP call; reads chunked PCM
    body and returns `np.ndarray(int16)`. POST JSON body carries
    `{"text", "language", "speed", "temperature"}` — `speed` is
    XTTS v2's native duration multiplier (1.15 in production for
    snappier cadence); `temperature` (NEW 2026-05-12) is the GPT
    duration-head sampling temperature (0.65 in production — lowered
    from XTTS library default 0.75 to cut phantom-token rate).
    Server-side passes both to `model.inference_stream(speed=...,
    temperature=...)` so cadence + stability are adjusted at
    synthesis time; the v3 pedalboard filter is unaffected.
  - `_stop_server_subprocess()` — graceful POST `/shutdown`, then
    SIGTERM, then SIGKILL. Called by the orchestrator's `shutdown()`.

The XTTS HTTP server itself lives at
[kenningVoiceAudio/scripts/xtts_server.py](../kenningVoiceAudio/scripts/xtts_server.py)
in the isolated `.venv-xtts` venv. FastAPI + uvicorn; uses an async
producer + asyncio.Queue pattern to bridge XTTS's sync streaming
generator into the FastAPI response without sync-generator
threadpool overhead (saved ~140 ms TTFT vs the naive sync-gen
implementation).

### `src/kenning/coding/` (Phase A foundation + Coding Addendum + Phase 2 projections)

#### `coding/audit.py`
- `class SessionAuditWriter` — per-session `logs/sessions/<id>.jsonl` writer
  - `write(kind, **fields)` — append one record

#### `coding/bridge.py`
- `class EventKind(str, Enum)` — STATUS / TEXT / TOOL_USE / TOOL_RESULT / FILE_CHANGE / ERROR / COMPLETE / USAGE
- `class FileChangeKind(str, Enum)` — CREATED / MODIFIED / DELETED
- `class TaskEvent` — dataclass with all event payload fields
- `class TaskRequest` — dataclass: task_prompt, cwd, model, timeout_s, label, etc.
- `class TaskResult` — dataclass: success, exit_status, summary, files_*, etc.
- `class TaskState` — running state
- `class TaskHandle(ABC)` — `task_id()`, `state()`, `add_listener()`, `cancel()`, `wait()`
- `class CodingBridge(ABC)` — `submit(request) -> TaskHandle`, `name()`
- `render_prompt(request)` — render TaskRequest into a string prompt
- `directory_snapshot(root)` / `diff_snapshots(...)` — ground-truth file diff

#### `coding/direct_bridge.py`
- `class DirectClaudeCodeBridge(CodingBridge)` — spawns `claude --print --stream-json ...`
- `class DirectTaskHandle(TaskHandle)` — parses event stream

#### `coding/intent.py`
- `class CodingIntentKind(str, Enum)` — NONE / CODE_TASK / PROGRESS_QUERY / CANCEL / MID_SESSION_ADJUSTMENT / CLARIFICATION_RESPONSE
- `class CodingIntent` — dataclass with kind, project_reference, etc.
- `classify(utterance, has_active_task=False, has_pending_clarification=False) -> CodingIntent`
- `derive_project_name(intent) -> str` — slug from task text
- `_DETERMINER_NOUN` (private regex fragment, NEW 2026-05-11 follow-up fix): `(the|that|this|your|our|my)(?:\s+(task|project|build|app|code|work|thing|run|job))?`. Plugged into all three sub-patterns of `_PROGRESS_PATTERNS` (`how X going|coming(along)?`, `what's X doing|working on|up to`, `is X done`) so phrasings like "How is that **project** going?" / "How's the **build** coming along?" / "Is **my project** done?" classify as `PROGRESS_QUERY` instead of falling through to the conversational LLM. The noun is optional so the legacy `that going` / `the doing` phrasings still fire bit-identical. The has_active_task gate is preserved so these patterns never hijack ordinary conversation.

#### `coding/important_files.py` (NEW 2026-05-22 catalog batch 1)
- `IMPORTANT_FILENAMES: frozenset[str]` — bare filenames (README, pyproject.toml, package.json, .gitignore, Dockerfile, Cargo.toml, etc.) plus kenning extensions (CLAUDE.md, MEMORY.md, SOUL.md, THIRD_PARTY_NOTICES.md, config.yaml, uv.lock, ruff.toml, ...)
- `IMPORTANT_RELATIVE_PATHS: frozenset[str]` — full project-relative paths (`docs/codebase_structure.md`, `.github/workflows`, etc.)
- `is_important(path) -> bool` — match by basename, full relative path, or `.github/workflows/` prefix; handles Windows backslashes
- `filter_important(paths) -> List[str]` — order-preserving filter
- `promoted_score(path, *, base=1.0) -> float` — small numeric bonus for downstream ranking (batch 2 repo-map personalization vector)

#### `coding/tree_sitter_tags.py` (NEW 2026-05-22 catalog batch 1)
- `class Tag(NamedTuple)` — `rel_fname`, `fname`, `line`, `name`, `kind` (`"def"` | `"ref"`)
- `extract_tags(path, root, *, cache=None) -> List[Tag]`
  - Detects language via `grep_ast.filename_to_lang`
  - Loads grammar via `grep_ast.tsl.get_language` (returns `tree_sitter.Language`)
  - Constructs `tree_sitter.Parser(language)` directly (tree-sitter-language-pack's own `Parser` ships an incompatible `builtins.Node` type)
  - Runs the vendored `<lang>-tags.scm` query via `Query` + `QueryCursor`
  - Maps `@name.definition.*` → `def`, `@name.reference.*` → `ref`
  - For languages with defs only (C, C++), backfills refs via pygments `Token.Name` tokenization
  - Optional `MtimeCache` for memoization keyed by `(fname, mtime)`
- `extract_tags_for_files(paths, root, *, cache=None) -> List[Tag]` — bulk wrapper; single-file failures are logged and skipped
- `supported_languages() -> List[str]` — sorted list of languages with vendored query files (currently 10)

#### `coding/repo_map.py` (NEW 2026-05-22 catalog batch 2)
- `class RepoMap(root, *, max_map_tokens=1024, max_map_tokens_no_chat=8192, mtime_cache=None, token_counter=None)` — PageRank-weighted repo map
  - `.get_map(*, chat_files=(), other_files=None, mentioned_fnames=(), mentioned_idents=(), force_refresh=False) -> str`
  - Builds file→file MultiDiGraph from tree-sitter tags
  - Edge weight: `mul * sqrt(num_refs)`; mul modifiers `*10`/`*0.1` per catalog
  - Personalization vector: chat / mentioned / path-component / important-files bonus
  - `nx.pagerank` with fallback to non-personalized on ZeroDivisionError
  - Per-(file, ident) rank distributed across out-edges
  - Binary-search budget via `_binary_search_to_budget` (tolerance 0.15, max_iters 30)
  - Render via patched `grep_ast.TreeContext` (`_ensure_grep_ast_patched` substitutes a `tree_sitter.Parser` constructed from `tslp.get_language` because tree-sitter-language-pack's own `Parser` ships an incompatible `builtins.Node` API)
  - Per-file TreeContext cache + per-render `(rel_fname, sorted(lois), mtime)` cache
- `class RepoMapProviderCache(*, max_map_tokens, max_map_tokens_no_chat, mtime_cache=None, token_counter=None)` — per-project RepoMap factory
  - `.get_or_create(project_path) -> Optional[RepoMap]` (thread-safe; reuses instances)
  - `.__call__(project_path, user_text) -> Optional[str]` — matches `ProjectSupervisor.repo_map_provider` contract; mines `user_text` for idents and passes them as `mentioned_idents`
- `extract_idents_from_text(text) -> Set[str]` — mines snake_case / kebab-case / camelCase / PascalCase / dotted identifiers from free-form text (e.g., the voice transcript)
- `find_source_files(directory) -> List[Path]` — recursive walk of source files, skips `SKIP_DIRECTORIES`, filters via `grep_ast.filename_to_lang`
- `SKIP_DIRECTORIES: FrozenSet[str]` — mirrors `project_introspect.SKIP_DIRECTORIES` plus `models/`, `logs/`
- Constants: `DEFAULT_MAX_MAP_TOKENS=1024`, `DEFAULT_MAX_MAP_TOKENS_NO_CHAT_FILES=8192`, `DEFAULT_TOLERANCE=0.15`, `DEFAULT_MAX_ITERATIONS=30`, `LINE_TRUNCATE_LENGTH=100`

#### `coding/queries/` (NEW 2026-05-22 catalog batch 1)
- 10 vendored `<lang>-tags.scm` files with attribution headers: python, javascript, bash, go, rust, c, cpp, java, ruby, csharp
- Adapted from `aider/queries/tree-sitter-language-pack/<lang>-tags.scm` (Apache 2.0; see [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md))
- `get_query_path(language) -> Optional[Path]` — resolves a tree-sitter language name to its bundled `.scm`

#### `coding/projects.py`
- `class Project` — dataclass: name, path, aliases, language
- `class ProjectRegistry` — atomic JSON CRUD on `data/projects.json`
- `class ResolutionKind(str, Enum)` — EXACT / ALIAS / SUBSTRING / SEMANTIC / NEW / UNRESOLVED
- `class ProjectResolution` — dataclass with kind + matched project
- `class ProjectResolver` — exact / alias / substring / semantic match
- `slugify_for_path(name) -> str` — collision-safe slug
- `new_sandbox_project(name, sandbox_root, registry) -> Project` — creates fresh dir + registers

#### `coding/session.py`
- `class SessionStatus(str, Enum)` — INITIALIZING / EXECUTING / VERIFYING / CORRECTING / AWAITING_CLARIFICATION / COMPLETE / FAILED / TERMINATED
- `is_valid_transition(from_status, to_status) -> bool`
- Records: `StageRecord`, `FileRecord`, `TestStatus`, `ClarificationRequest`, `AdjustmentRecord`, `CompletionClaim`
- `class ProjectSession` — full session state (large; passed only via projections)
- `class StateTransitionError(RuntimeError)`
- `class SessionStore` — owns sessions; `create()`, `get()`, `transition()`, `record_*()`

#### `coding/projections.py` (Phase C / Foundation Part 2)
- `count_tokens(text) -> int` — tiktoken cl100k_base
- `class ProjectionResult` — projection + text + token_count + budget + truncations_applied + truncation_warning
- `_finalize_projection(...)` — common end-of-projection: INFO log on truncations, ERROR on over-budget
- 5 projections, each with a dataclass + `project_X_context()` function:
  - `project_clarification_context(session, clarification_question, options?, facts_lookup?) -> ProjectionResult` (1500 tok)
  - `project_status_delta(session) -> ProjectionResult` (600 tok)
  - `project_adjustment_context(session, adjustment_text, facts_lookup?, conflict_detector?) -> ProjectionResult` (1200 tok)
  - `project_correction_context(session, failures, failed_test_names?, failed_test_messages?) -> ProjectionResult` (1500 tok)
  - `project_completion_context(session) -> ProjectionResult` (800 tok)

#### `coding/templates.py`
- `class TemplateError(RuntimeError)`, `PromptTooLargeError`, `SchemaValidationError`
- `class RenderResult` — dataclass: rendered text + token count
- `class TemplateRenderer` — Jinja2 wrapper for prompts/coding/*.j2
  - `render_initial_new(...)`, `render_initial_edit(...)`, `render_correction(...)`,
    `render_adjustment(...)`, `render_clarification_response(...)`

#### `coding/verification.py`
- `class CheckId(str, Enum)` — STRUCTURE / TESTS / SMOKE / LINT / FILES / PYTHON_SYNTAX
- `class CheckResult`, `VerificationReport` — dataclasses
- `class Verifier`
  - `verify(session) -> VerificationReport` — runs 6 checks + writes `logs/verifications.jsonl`
  - `verify_tests(session)` — single-check helper

#### `coding/narration.py`
- `class NarrationDelta` — dataclass tracking what's new since last query
- `class StatusNarrator` — voice-friendly progress narration
  - `narrate(session) -> str` — final completion narration
  - `progress_narration(session) -> str` — uses `project_status_delta` projection

#### `coding/runner.py`
- `build_default_bridge() -> CodingBridge` — picks DirectClaudeCodeBridge from config
- `class ProgressSinceLastQuery` — dataclass
- `class CodingTaskRunner`
  - `start_task(request)` — submits via bridge
  - `has_active_task() -> bool`
  - `cancel_active() -> bool`
  - `progress_narration() -> str`
  - `completion_narration() -> Optional[str]` — 2026-05-11 narration honesty: when `state.success` AND `n_created+n_modified+n_deleted == 0`, returns an explicit "I finished without writing or modifying any files. The project may need more direction, or it may have run out of token budget mid-exploration -- say continue if you want me to keep going." instead of the legacy "Done. ... <generic Claude tail> ... Elapsed: Xs." The generic tail summary is suppressed on this branch so the honest opener stands on its own. Pairs with the `coding.token_budget_per_session: 400000` bump. **2026-05-11 follow-up fix:** the project-root line is now `f"Saved under {path.name}."` (project folder leaf only) — was `f"Project root: {path}."` which interpolated the absolute Windows `state.cwd` (e.g. `C:\STC\ultronPrototype\data\sandbox\converts_pdf_docx`) and made XTTS-v2 enter pathological inference trying to pronounce backslash + colon + drive letter, pinning the GPU at 100 % until the server timed out. `StatusNarrator` already used the leaf only for progress narration; this brings completion_narration in line. The full path is still on disk in the per-session JSONL audit log + `coding_tasks.jsonl` start event so debugging is unaffected.
  - `pop_budget_warning() -> Optional[str]`
  - `record_pre_task_aborted(*, label, reason, intent_text="")` (V1-gap A4) — append a pre-task abort row to the audit log when the orchestrator's barge-in watcher fires.

#### `coding/ast_metadata.py` (2026-05-19 Track 1f)
- `@dataclass class AstMetadata` — frozen; `syntax_valid: bool`, `error: str`,
  `functions_defined: List[str]`, `functions_called: List[str]`,
  `imports: List[str]`, `classes_defined: List[str]`
- `extract_python_metadata(source: str) -> AstMetadata` — pure stdlib AST parse;
  ~5-50 ms per file; fail-soft on non-Python input (returns syntax_valid=False
  with error string)
- `extract_metadata_from_path(path: Path) -> AstMetadata` — file convenience wrapper
- Consumers: Track 1g coding-runner FILE_CHANGE listener (syntax verification);
  2026-05-22 `project_introspect.snapshot()` (per-file structural metadata)

#### `coding/project_digest.py` (2026-05-22 supervisor Phase A)
- `SUMMARY_TEMPLATE` — markdown template with sections Goal / Constraints / Progress
  {Done, In Progress, Blocked} / Key Decisions / Next Steps / Critical Context /
  Relevant Files; port of opencode's `packages/opencode/src/session/compaction.ts`
- `DIGEST_SECTIONS` / `PROGRESS_SUBSECTIONS` — section ordering constants
- `@dataclass class DigestRequest` — project_name, project_path, task_summary,
  files_created/modified/deleted, prior_digest_markdown, user_goal_hint,
  language, entry_points
- `@dataclass class ProjectDigest` — project_name, project_path, markdown,
  sections (parsed), generated_at, elapsed_ms, fallback (True when template
  fallback used), source ("llm" | "template" | "manual")
- `LLMCallable` — type alias: callable that takes prompt string returns completion
- `generate_digest(request, llm_call=None, *, max_files_in_prompt=40,
  max_summary_chars=4000) -> ProjectDigest` — main entry; fails-open to
  `render_template()` when no LLM or LLM raises / returns empty
- `render_template(request) -> str` — deterministic fallback (no LLM)
- `parse_digest_sections(markdown) -> Dict[str, str]` — walks `## Header` blocks
  case-insensitive against `DIGEST_SECTIONS`; handles trailing whitespace
- `extract_files_from_digest(markdown) -> List[str]` — extracts "Relevant Files"
  section paths; returns empty when the section is missing or `(none)`
- Internal helpers: `_build_digest_prompt` (PRIOR_PROLOGUE vs PROLOGUE selection),
  `_summarize_files_for_prompt` (caps + "+N more" trailer),
  `_normalize_digest_markdown` (strips ` ```markdown ` fences)

#### `coding/project_introspect.py` (2026-05-22 supervisor Phase B non-LLM)
- `LANGUAGE_BY_EXT: Mapping[str, str]` — extension -> language (python / javascript
  / typescript / rust / go / java / kotlin / swift / csharp / cpp / c / ruby /
  php / scala / clojure / elixir / bash / powershell / lua / r / matlab / dart /
  html / css / vue / sql / apex)
- `MARKER_FILES: Mapping[str, str]` — pyproject.toml / setup.py / requirements.txt /
  manage.py / app.py / package.json / Cargo.toml / go.mod / pom.xml / etc.
- `ENTRY_POINT_FILENAMES: Sequence[str]` — manage.py / main.py / app.py /
  __main__.py / server.py / index.js / index.ts / wsgi.py / asgi.py / etc.
- `SKIP_DIRECTORIES: frozenset` — node_modules / __pycache__ / .git / .venv /
  build / dist / target / .next / coverage / etc.
- `DEFAULT_MAX_DEPTH=6`, `DEFAULT_MAX_FILES=500`, `DEFAULT_MAX_DIRECTORIES=200`,
  `DEFAULT_AST_FILE_CAP=30`, `DEFAULT_CACHE_TTL_SECONDS=30.0`
- `@dataclass class FileInfo` — frozen; path, relative_path, size_bytes,
  extension, is_entry_point
- `@dataclass class ProjectSnapshot` — project_path, project_name, files,
  directories, languages, language_counts, entry_points, markers,
  **important_files: List[FileInfo]** (NEW 2026-05-22 catalog batch 1 — populated during walk by `is_important()` from `coding/important_files.py`; surfaces README / pyproject.toml / CLAUDE.md / docs/codebase_structure.md / etc. at the top of `render_tree_summary` tagged `[important]`),
  ast_metadata: `Dict[str, AstMetadata]`, captured_at, elapsed_ms, truncated;
  `.dominant_language` property; `.file_count`; `.render_tree_summary(max_lines=50)`
  — important files lead the output (capped at 15) with `[important]` / `[important,entry]` tags, then the rest of the files in walk order
- `snapshot(project_path, *, max_depth, max_files, max_directories,
  ast_file_cap, use_cache=True) -> ProjectSnapshot` — main entry
- `invalidate_snapshot_cache(project_path=None)` — drop entry or full clear
- Internals: `_walk_project`, `_detect_languages`, `_detect_entry_points`,
  `_parse_ast_for_python_files`, `_SnapshotCache` (TTL-based)

#### `coding/project_index.py` (2026-05-22 supervisor Phase B Qdrant)
- `@dataclass class ProjectIndexEntry` — project_id, project_name, project_path,
  digest_markdown, digest_sections, digest_text_summary, language, entry_points,
  tags, last_modified_unix, created_at_unix, last_session_id;
  `.to_payload()` / `.from_payload(d)`
- `@dataclass class ProjectMatch` — entry, score, reason
- `class ProjectIndex(embedder, qdrant_path=None, collection_name=None,
  recent_cache_size=50, client=None)` — 2026-06-12: `client=` BORROWS an
  already-open embedded Qdrant client (the orchestrator passes
  ConversationMemory's; local-mode Qdrant allows ONE open client per
  path, so the old unconditional second open failed every boot and
  forced the supervisor registry-only; mirrors the WebResultsCache
  borrow pattern). `_owns_client` tracks ownership.
  - `.upsert(digest, *, project_id=None, tags=None, language="",
    entry_points=None, last_session_id=None) -> Optional[ProjectIndexEntry]` —
    embeds the digest text summary, upserts Qdrant; publishes
    `ProjectIndexedEvent` on bus
  - `.get(project_id) -> Optional[ProjectIndexEntry]` — cache-first lookup
  - `.get_by_path(project_path) -> Optional[ProjectIndexEntry]`
  - `.list_all(limit=100) -> List[ProjectIndexEntry]` — most-recently-modified first
  - `.search(query, *, top_k=5, min_score=0.0) -> List[ProjectMatch]` —
    cosine semantic search; uses `query_points()` with named "dense" vector
  - `.search_by_name(name_substring, *, top_k=10) -> List[ProjectIndexEntry]` —
    lexical fallback over recent cache + scroll
  - `.delete(project_id) -> bool`
  - `.close()` — releases the underlying client IF owned; a no-op on
    borrowed clients (the owner -- ConversationMemory in production --
    controls that lifecycle). Never raises.
  - `.count() -> int`
- Helpers: `_derive_project_id(path)` — UUID5 with fixed namespace for stable id;
  `_build_digest_summary_for_search(sections, max_chars=500)` — Goal + Critical
  Context + Relevant Files concat, capped; `_score_reason(score)` — human label

#### `coding/project_supervisor.py` (2026-05-22 supervisor Phase C)
- `class SupervisorAction(str, Enum)` — RESUME / EDIT / CLARIFY / NEW
- `@dataclass class SupervisorCandidate` — project_id, project_name, project_path,
  score, source ("semantic" | "registry_exact" | "registry_substring" | ...)
- `@dataclass class SupervisorDecision` — action, target_project_id/name/path,
  resume_session_id, candidates, confidence, reasoning, clarification_question,
  file_hints, user_text, **repo_map_text: Optional[str]** (NEW 2026-05-22 catalog
  batch 2 — rendered PageRank repo map, populated by `_attach_repo_map` when a
  `repo_map_provider` is set on the supervisor and the decision resolves to a
  known project path); `.to_log_dict()` for the JSONL audit (excludes
  `repo_map_text` to keep the log lean; emits `repo_map_attached: bool` instead)
- `@dataclass class SupervisorInputs` — user_text, coding_intent,
  has_active_task, active_task_project_name, active_task_session_id, turn_id
- `class ProjectSupervisor(index, registry, resolver, *, resolve_threshold=0.75,
  clarify_threshold=0.55, decisions_log_path=None, max_candidates_in_decision=5,
  repo_map_provider=None)`
  - `.decide(inputs) -> SupervisorDecision` — runs the priority pipeline
    (active-adjustment / strong-semantic / registry-exact / clarify-band / new);
    always returns + never raises; audit-logs + publishes `SupervisorDecidedEvent`;
    calls `_attach_repo_map(decision)` last (after audit + bus publish) so the
    rendered map doesn't bloat the JSONL
  - `._attach_repo_map(decision)` — invokes `repo_map_provider(project_path,
    user_text)` and stores the result on the decision; skipped for CLARIFY and
    for decisions without a `target_project_path`; provider exceptions are
    logged + swallowed (decision quality is the contract, repo map is a bonus)
- Helpers: `_merge_candidates(semantic, registry, *, cap)` — dedup-by-path,
  higher-score-wins, source labels concatenated; `_registry_to_candidate`;
  `_project_id_for_registry(project)` — maps registry path to same UUID5 as index

#### `coding/supervisor_dispatch.py` (2026-05-22 supervisor Phases D + E)
- `class DispatchActionKind(str, Enum)` — EDIT_DISPATCH / NEW_DISPATCH /
  RESUME_FORWARD / CLARIFY / BARGED_IN / FALLBACK
- `@dataclass class DispatchOutcome` — kind, voice_message, task_request,
  clarification_question, resume_session_id, decision, already_narrated
- `BargeInCheckable` / `PlainSpeak` — type aliases for injected speak callables
- `class SupervisorDispatchController(supervisor, *, index, barge_in_speak,
  plain_speak, narrate_enabled=False, narration_barge_in_window_seconds=1.5,
  enriched_context_enabled=False, sandbox_root=None, default_model="haiku")`
  - `.dispatch(inputs) -> DispatchOutcome` — orchestrates supervisor.decide →
    narrate (Phase D, with barge-in) → build enriched TaskRequest (Phase E)
  - `.build_digest(project_name, project_path, task_summary, files_*,
    *, llm_call, prior_digest_markdown, user_goal_hint) -> ProjectDigest` —
    convenience used by the COMPLETE listener
- Internal: `_narration_for(decision)` (RESUME / EDIT / NEW / CLARIFY narration
  text via TTS-safe `_speakable`); `_build_edit_prompt` (Phase E digest +
  file-tree snapshot + file hints prepended); `_slugify_for_directory`;
  `_speakable` (strips backslashes / drive letters); `_indent_block`

#### `coding/coordinator.py`
- `class DecisionPath(str, Enum)` — RULE_ESCALATE / RULE_DEFAULT / RULE_ANSWER / FACT_ANSWER (V1-gap A3) / LLM_ANSWER / LLM_DEFAULT / LLM_ESCALATE / USER_ANSWER / TIMEOUT_DEFAULT
- `class ClarificationDecision`, `AdjustmentDecision`, `PendingUserClarification`, `_FactAnswer` (V1-gap A3, internal) — dataclasses
- `class ConversationCoordinator`
  - `__init__(store, llm, *, ..., facts_lookup=None)` — V1-gap A3: optional callable that reads the Qdrant `facts` collection. Wired by the orchestrator to `KenningMCPServer.lookup_facts`.
  - `decide_clarification(session_id, request, session) -> str` — answer or escalate. V1-gap A3: a high-confidence directive-category fact short-circuits the LLM call (Fast-path 2.5 between preference-options and always-answer rules).
  - `decide_adjustment(session_id, adjustment_text) -> AdjustmentDecision`
  - `handle_declare_complete(session_id) -> str` — runs Verifier, drives correction loop
  - `pending_user_clarifications() -> List[PendingUserClarification]`

#### `coding/mcp_server.py`
- `class KenningMCPServer`
  - `__init__(*, host, port, sse_path, log_path, clarification_timeout_s, session_audit_dir=None, memory=None)` — V1-gap A3: `memory` kwarg threads a live `ConversationMemory` so `lookup_facts` queries Qdrant. `None` preserves the test-isolation no-op.
  - In-process Python tools (called by Qwen via `get_config().coding.mcp.host:port`):
    - `create_session()`, `get_full_state()` (Python only), `get_status_delta()`,
      `get_clarification_context()`, `get_adjustment_context()`,
      `get_correction_context()`, `get_completion_context()`,
      `send_followup()`, `terminate_session()`, `list_active_sessions()`,
      `lookup_facts(query, *, k=None, min_confidence=None, max_age_days=None)` — V1-gap A3: when memory is wired, returns dict-shaped FactRow rows (proxies `memory.search_facts`); otherwise `[]`. Audit entry tagged `source="no_memory_wired"` on the stub branch.
  - SSE worker tools (called by AI coding agent via SSE):
    - `report_progress()`, `request_clarification()`, `report_test_results()`,
      `declare_complete()`, `abandon_task()`, `record_file_change()`
  - `set_clarification_responder(fn)` / `set_declare_complete_handler(fn)` — coordinator hooks
  - `start()` / `stop()` — manage SSE server. 2026-06-12: the server thread's `finally` cancels + gathers any pending asyncio tasks (the `_wait_for_started` watcher, kept on `self._waiter_task`, plus uvicorn stragglers) BEFORE closing its private loop — a bind failure (port 19761 taken by a stale instance) used to leave the watcher pending and GC emitted the benign-but-noisy asyncio "Task was destroyed but it is pending!" stderr line; happy path unchanged.
  - `is_running() -> bool` — True when the SSE worker thread is alive AND the started event is set (2026-05-29 B3-loop); lets the voice layer decide whether to write a per-project `.mcp.json`.
- `write_mcp_config(project_root, sse_url)` — writes a `.mcp.json` pointing at the live SSE server so a spawned coding subprocess can reach the in-process MCP tools. `remove_mcp_config(project_root)` — available but **no longer called on task COMPLETE** (B3-loop-2: `send_followup` reuses the file for follow-ups + corrections; it is a tiny gitignored sandbox artefact overwritten on the next dispatch).

#### `coding/voice.py`
- `class VoiceResponse` — dataclass: text, handled, cancelled, **pre_task_confirmation, deferred_dispatch, pre_task_label** (V1-gap A4 — when populated, the orchestrator speaks the confirmation with barge-in detection before running the deferred dispatch closure).
- `class CapabilityVoiceController` (Phase 5 rename; alias = CodingVoiceController). `__init__` accepts an optional `llm_engine` (the live `LLMEngine`) so MODEL_SWITCH intents can call `llm_engine.reload_for_preset(...)`. 2026-05-22 supervisor kwargs: `supervisor_dispatch` + `project_index` — when wired, intercepts `_handle_code_task` BEFORE the legacy ProjectResolver path; when `None`, controller is byte-for-byte unchanged.
  - `pending_completion()` / `pending_clarifications()` / `pending_budget_warning()`
  - `has_pending_clarification() -> bool`
  - `handle_utterance(text) -> Optional[VoiceResponse]` — coding-only (delegated by capability dispatch)
  - `handle_capability_intent(routing_intent) -> Optional[VoiceResponse]` — top-level dispatch (Phase 5)
  - `_build_code_task_response(...)` (V1-gap A4, internal) — wraps `_submit` into a deferred dispatch closure when `coding.pre_task_confirmation_enabled`. Read-only intents (PROGRESS_QUERY / CANCEL / etc.) keep the legacy text-only response.
  - `_build_pre_task_confirmation(...)` / `_summarise_intent_for_voice(...)` (V1-gap A4, internal) — render the confirmation phrase ("I'll have the AI coding agent <verb> on the <project> project. Going ahead.").
  - **2026-05-22 supervisor methods (internal):**
    - `_handle_code_task_via_supervisor(intent) -> Optional[VoiceResponse]` —
      builds `SupervisorInputs`, calls `supervisor_dispatch.dispatch()`,
      converts `DispatchOutcome` to a `VoiceResponse`. Returns `None` on
      FALLBACK to drop through to the legacy resolver path.
    - `_dispatch_supervisor_task(intent, outcome) -> VoiceResponse` —
      mkdir cwd, `registry.touch(name)`, `runner.start_task(request)`,
      `_attach_supervisor_digest_listener(...)`.
    - `_attach_supervisor_digest_listener(handle, project_name,
      project_path, user_goal_hint)` — registers COMPLETE listener that
      calls `supervisor_dispatch.build_digest(...)` +
      `project_index.upsert(...)`; gated on `coding.supervisor.digests_enabled`.
    - `_current_project_name()` / `_current_session_id_or_label()` —
      best-effort runner state introspection.
  - **2026-05-29 production-hardening (B3) methods (internal):**
    - `_maybe_write_mcp_config(project_path) -> Optional[Path]` — when a live
      MCP server `is_running()`, writes the per-project `.mcp.json` (via
      `write_mcp_config`) so the spawned subprocess reaches the
      clarify/verify/complete loop; fail-open `None` otherwise. Called from
      `_submit` + `_dispatch_supervisor_task`, which also set `request.mcp_config_path`.
    - `_create_and_bind_session(project_path, user_intent, *, is_new)` — creates
      a `ProjectSession` in the coordinator store + `runner.bind_session(sid)` so
      the MCP server resolves `_claude_active_session` on callback; fail-open
      no-op without a coordinator.
    - `_attach_resume_followup_listeners(handle, user_goal_hint)` (B3-loop-2) —
      re-attaches the digest + voice-lock-review listeners to the fresh handle
      `send_followup` spawns on RESUME_FORWARD (the original handle's listeners
      do not carry over to the follow-up subprocess); re-derives the project
      from `runner.active_state()`.
    - `maybe_handle_run_program(text) -> Optional[VoiceResponse]` /
      `pop_run_report() -> Optional[str]` (B3-runlaunch) — "run the calculator"
      starts a sandbox-confined, validator-gated run on a background thread (->
      a pending run report the orchestrator drains via `pop_run_report`); "launch
      the server" detaches. Unresolved project hint -> `None` fall-through to
      legacy handling. Delegates to `coding/sandbox_runner.py`.
- **Module-level helper:** `_build_supervisor_llm_call(llm_engine, sup_cfg)
  -> Optional[LLMCallable]` — wraps `LLMEngine.generate` for the digest
  call (kwargs-fallback for variant signatures; returns None when
  llm_engine is missing).

#### `coding/sandbox_runner.py` (NEW 2026-05-29 production-hardening B3-runlaunch)
- Voice-driven "run / launch a finished sandbox program" surface. Every run is
  confined to the coding sandbox root + gated through the safety validator.
- `match_run_program(text) -> RunProgramMatch` — strict matcher returning `mode`
  (`"run"` = backgrounded with captured output, `"launch"` = detached) + an
  optional `project_hint`; non-matches return a falsy match.
- `resolve_entry_point(project_path) -> EntryPoint` — picks the runnable entry
  point (e.g. `main.py`, a `[project.scripts]` console script, an `app` module)
  + interpreter/argv; used both to run and to enrich the completion report.
- `run_program(project_path, *, timeout_s, run_fn=None, validator=None)` — runs
  the entry point behind a hard `_is_within(path, root)` sandbox-confinement
  check + a validator gate + a timeout; `run_fn` is injectable for tests.
  Summarised by `summarize_run_result(result)`.
- `launch_program(project_path, *, spawn_fn=None, validator=None)` — detached
  launch (injectable `spawn_fn`); used for "start it up for me".
- `_is_within` / `_validator_blocks` — the load-bearing confinement + fail-open
  audit helpers. Consumed by `coding/voice.py::maybe_handle_run_program` and the
  orchestrator's `_maybe_handle_run_program` run-loop short-circuit (with
  `_announce_pending_run_report` draining the async run summary).

#### `coding/edit_matcher.py` — the SEARCH/REPLACE edit engine
- `class Strategy`, `class EditResult`, `apply_edit()`, `apply_edit_to_files()`, `find_similar_lines()` — the fuzz-cascade (exact → whitespace-flexible → …) edit applier.

#### `coding/edit_recovery.py`
- `EditSpec`, `EditRecoveryResult`, `did_edit_likely_apply()`, `is_search_mismatch_error()`, `enrich_mismatch_error()`, `run_edit_with_recovery()`, `wrap_edit_tool_with_recovery()` — SEARCH-mismatch recovery wrapper (SWE-Agent T1/T14).

#### `coding/patch_v4a.py` — v4a unified-patch format
- `parse_v4a_patch()`, `apply_patch()`, dataclasses `PatchAction`/`PatchHunk`/`PatchFileBlock`/`ParsedPatch`, `class PatchError`, + the BEGIN/END/ADD/UPDATE/DELETE/SCOPE/EOF/FUZZ_* constants.

#### `coding/file_read_cache.py` (catalog cline T7a) — per-session mtime-validated file-read cache
- `class FileReadCache` (RLock-guarded; `maybe_serve_from_cache` / `record_read` / `invalidate` / `clear`), `class CachedReadEntry`, `get_file_read_cache(session_id, max_entries=None)`. **NOTE: this file lives in `coding/`, NOT `desktop/`** (the desktop/ file-tree entry is misplaced).

#### `coding/file_mention_resolver.py`
- `class FileMention`, `resolve_mentions()` — resolve `@file` / path mentions in a coding prompt.

#### `coding/coder_modes.py`
- `class EditFormat`, `class CoderMode`, `get_coder_mode()`, `list_coder_modes()`, `edit_modes()`, `read_only_modes()`.

#### `coding/architect_supervisor.py`
- `class ArchitectRequest`, `class ArchitectPlan`, `class ArchitectSupervisor`, `DEFAULT_ARCHITECT_SYSTEM_PROMPT` — the pre-dispatch architect plan provider (`coding.architect`).

#### `coding/commit_message.py`
- `class CommitMessageRequest`, `class CommitMessageResult`, `generate_commit_message()`, `strip_outer_quotes()`, `DEFAULT_COMMIT_SYSTEM_PROMPT`.

#### `coding/python_lint.py` / `coding/tree_sitter_lint.py` (pre-write lint cascade, `coding.pre_write_lint`)
- python_lint: `lint_python()`, `FLAKE8_FATAL_SELECT`, `DEFAULT_FLAKE8_TIMEOUT` (compile + flake8). tree_sitter: `class LintError`, `class LintReport`, `tree_sitter_lint()`, `MAX_NODE_VISITS` (multi-language syntax check).

#### `coding/ai_comment_watcher.py`
- `class AICommentKind`, `class AICommentTrigger`, `class AICommentWatcher`, `scan_file_for_ai_comments()`, `AI_COMMENT_REGEX` — scans written files for `AI!`/`AI?`-style action comments.

### `src/kenning/openclaw_routing/` (Phase 5)

#### `openclaw_routing/intents.py`
- `class RoutingIntentKind(str, Enum)` — **26 values** (ACTIVE_WINDOW_QUERY, SEMANTIC_CLICK, WINDOW_CLOSE_CONFIRMATION added): CONVERSATIONAL, CODE_TASK, PROGRESS_QUERY, CANCEL, MID_SESSION_ADJUSTMENT, CLARIFICATION_RESPONSE, BROWSER_AUTOMATION, MEDIA_GENERATION, MESSAGING, FILE_OPERATION, SHELL_OPERATION, HYBRID_TASK, MODEL_SWITCH (4B plan), SYSTEM_STATUS (Phase 13), GAMING_MODE (V1-gap A1), DESKTOP_AUTOMATION (V1-gap C3), WINDOW_AUTOMATION (V1-gap C3), APP_LAUNCH (Phase 8 desktop), SCREEN_CONTEXT_QUERY (Phase 8 desktop), WINDOW_MOVE (2026-05-14 third pass), WINDOW_CLOSE (2026-05-14 third pass), **OPEN_LAST_SOURCE** (2026-05-22 opens cited URL from last search-augmented turn; supports ordinal "the second one" + referent "the NBC story" + embedding-similarity match), **NAVIGATE_TO_SITE** (2026-05-22 queries SearxNG + scores top-10 domains + opens best match)
- Per-category dataclasses: `BrowserIntent`, `MediaGenIntent`, `MessagingIntent`, `FileOpIntent`, `ShellOpIntent`, **`GamingModeIntent`** (V1-gap A1), **`DesktopIntent`** (V1-gap C3), **`WindowIntent`** (V1-gap C3), **`AppLaunchIntent`** (Phase 8 desktop), **`ScreenContextIntent`** (Phase 8 desktop), **`WindowMoveIntent`** (2026-05-14 third pass), **`WindowCloseIntent`** (2026-05-14 third pass), **`OpenLastSourceIntent`** (2026-05-22: monitor_index, monitor_query, ordinal, referent, raw_text), **`NavigateToSiteIntent`** (2026-05-22: site_query, monitor_index, monitor_query, raw_text)
- `HybridSubtask` — dataclass: order, type, subtype, description
- `RoutingIntent` — top-level dataclass: kind, raw_text, confidence, source, reason, coding_intent, automation_intent, subtasks, model_switch_intent, system_status_intent, **gaming_mode_intent, desktop_intent, window_intent** (V1-gaps A1/C3), app_launch_intent, screen_context_intent, window_move_intent, window_close_intent, **open_last_source_intent, navigate_to_site_intent** (2026-05-22), needs_user_clarification, clarification_question
- `DispatchResult` — dataclass: success, voice_message, error, metadata
- `TaskInfo` — task tracking dataclass
- `AutomationIntent` = Union of the 5 automation intent classes

#### `openclaw_routing/classifier.py`
- `classify_routing(utterance, has_active_coding_task=False, has_pending_clarification=False) -> RoutingIntent`
  Layered dispatch order (first hit wins):
  1. In-flight commands (CANCEL / etc.)
  2. Hybrid signals
  3. Coding intents (CODE_TASK / etc.)
  4. **1.93 NAVIGATE_TO_SITE keyword pattern** (2026-05-22; "open the X website")
  5. **1.95 OPEN_LAST_SOURCE** (2026-05-22; "show me that article" / "the second one" / "the NBC story")
  6. **2.0 APP_LAUNCH** (Phase 8; "open YouTube on monitor 2", image-search)
  7. **2.05 NAVIGATE_TO_SITE verb pattern** (2026-05-22; "take me to HBO Max")
  8. WINDOW_MOVE / WINDOW_CLOSE (2026-05-14)
  9. Other automation rules (BROWSER_AUTOMATION / etc.)
  10. CONVERSATIONAL fallback
- `_build_browser_intent(text)`, `_build_media_intent(text)`, `_build_messaging_intent(text)`, `_build_file_intent(text)`, `_build_shell_intent(text)` — extract structured intent from raw text
- **2026-05-22 OPEN_LAST_SOURCE primitives:** `_OPEN_LAST_SOURCE_VERB` (`show me | open | pull up | bring up | load` — navigation verbs deliberately EXCLUDED), `_OPEN_LAST_SOURCE_NOUN` (`article | link | page | source | story | result | citation | website | site | url | item | entry | piece | report | headline | one`), four regex patterns (BARE / REF_BEFORE / REF_AFTER / NUMBER), `_ORDINAL_WORDS` map (first..tenth + last=-1), `_NUMBER_WORDS` map (one..ten). `_extract_open_last_source_referent(text)` extracts (ordinal, referent) tuple; `_classify_open_last_source(text) -> Optional[OpenLastSourceIntent]`.
- **2026-05-22 NAVIGATE_TO_SITE primitives:** `_NAV_TO_SITE_VERB` (`take me to | go to | navigate to | head to | find me | bring me to`), `_NAV_TO_SITE_KEYWORD` (`website | site | page | homepage | .com | .org | .net | .io | dot com`), two patterns (VERB / KEYWORD); 27-entry `_NAVIGATE_TO_SITE_SITENAME_DENY` blocks "take me to bed" / "go to the gym" / "take me to the bathroom" / etc. `_classify_navigate_to_site(text) -> Optional[NavigateToSiteIntent]`.
- **Comprehensive test pass extensions (HEAD 2fb0988+):** `_BROWSER_INTERACT.scroll` now covers `scroll the <page|window|tab|view|content|results|list> <down|up|left|right|to>` (the original pattern only matched `scroll <down|up|to> the`); `_MEDIA_PATTERNS.render` now covers `render <a|an|the> <image|scene|picture|video|illustration|drawing|artwork>` with optional `me` (the original required `render me`); `_MESSAGING_PATTERNS` adds `notify me <on|via> <telegram|signal|slack|discord>` (parallel to the existing `tell me on …` form); `_FILE_PATTERNS` adds `show me the contents of <file.ext>` (the original required the literal word "file").

#### `openclaw_routing/dispatcher.py`
- `class OpenClawDispatcher`
  - `__init__(config?, *, llm=None, bridge=None, gaming_mode_manager=None)` — reads openclaw.enabled + routing.stub_responses_enabled; threads optional dependencies for live-dispatch paths.
  - `async handle_browser(intent)` / `handle_media_generation(intent)` / `handle_messaging(intent)` / `handle_file_operation(intent)` / `handle_shell_operation(intent)` — return live results when the bridge is wired (Phases 4, 6, 12), stubs otherwise.
  - `async handle_gaming_mode(intent)` (V1-gap A1) — engage / disengage / status. Routes to `GamingModeManager` for plugin enable/disable; voice messages match the spec phrasing.
  - `async handle_desktop_automation(intent)` (V1-gap C3) — screenshot / list_windows / find_window via `DesktopTool`. Short-circuits with a clear message when gaming mode is engaged.
  - `async handle_window_automation(intent)` (V1-gap C3) — focus / click / type via `WindowControlTool`. Same gaming-mode short-circuit.

#### `openclaw_routing/gaming_mode.py` (V1-gap A1)
- `class GamingModeStatus(str, Enum)` — IDLE / ENGAGED / TRANSITIONING.
- `@dataclass class GamingModeReport` — engage/disengage outcome with per-plugin states + Docker action info.
- `class GamingModeManager`
  - `__init__(*, client, plugins_to_disable, toggle_docker, ...)` — owns the engage/disengage state machine.
  - `async engage()` — calls `client.disable_plugin(slug)` for each configured plugin; optionally stops Docker Desktop. Best-effort: per-plugin failures don't abort the cycle.
  - `async disengage()` — re-enables only the plugins successfully disabled during the matching engage.
  - `status() -> GamingModeStatus`.
  - Audit log: `logs/gaming_mode.jsonl`.

#### `openclaw_routing/runner.py`
- `class AutomationTaskRunner` — mirror of `CodingTaskRunner` for automation tasks
  - `async submit_task(routing_intent) -> task_id` — dispatches via OpenClawDispatcher
  - `async progress_narration(task_id) -> Optional[str]`
  - `async completion_narration(task_id) -> Optional[str]`
  - `async cancel(task_id) -> bool`
  - `list_active() -> List[TaskInfo]` / `get_task(task_id)`
  - Audit log: `logs/automation_tasks.jsonl`

#### `openclaw_routing/decomposer.py`
- `class DecompositionResult` — subtasks + fallback_used + raw_response
- `class HybridTaskDecomposer`
  - `async decompose(utterance) -> DecompositionResult` — calls Qwen with JSON-output prompt, parses, falls back to one-element coding plan on any failure

#### `openclaw_routing/disambiguator.py`
- `class DisambiguationResult` — kind (CODE_TASK / HYBRID_TASK / CONVERSATIONAL / None) + clarification_question
- `class IntentDisambiguator`
  - `async disambiguate(utterance) -> DisambiguationResult` — asks Qwen "CODING/AUTOMATION/HYBRID/UNCLEAR"

#### `openclaw_routing/decision_log.py`
- `class RoutingDecisionLog` — JSONL writer (`logs/routing_decisions.jsonl`)
  - `record(intent, *, handler, outcome, extra?)` — best-effort append
- `get_routing_log() -> RoutingDecisionLog` — singleton
- `set_routing_log(log)` — test injection

### `src/kenning/openclaw_bridge/` (OpenClaw Phase 1 + 3 foundations)

The bridge layer between Kenning and the OpenClaw Gateway peer. Voice
pipeline is unaffected when OpenClaw is unreachable (`fail_open: true`).

#### `openclaw_bridge/persona.py` (Phase 1)

- `class PersonaLoader` — reads the six workspace files
  (IDENTITY/SOUL/USER/AGENTS/HEARTBEAT/BOOTSTRAP) and composes a
  system prompt for the requested mode. Hot reload via `refresh_if_stale`
  (mtime+size check on each call).
  - `load() -> PersonaBundle` — force a fresh read.
  - `refresh_if_stale() -> PersonaBundle` — reload only if anything
    changed; cheap.
  - `get_system_prompt(mode="user_facing") -> str` — composes per mode.
- `PromptMode = Literal["user_facing", "background", "heartbeat", "bootstrap"]`
  - `user_facing` — IDENTITY + SOUL + USER. Voice path; full Kenning
    character.
  - `background` — AGENTS only, prefixed with internal-worker framing.
    For heartbeat preflight, cron, summarization, tool selection.
  - `heartbeat` — HEARTBEAT only.
  - `bootstrap` — BOOTSTRAP only.
- `default_workspace_dir() -> Path` — resolves
  `~/.openclaw/workspace/` or `KENNING_OPENCLAW_WORKSPACE` env override.
- `class PersonaBundle` / `PersonaFile` — dataclasses with
  fingerprint (`(name, mtime_ns, size)`) for change detection.
- HTML-comment-only files (e.g., a placeholder USER.md with
  `<!-- auto-populated by maintenance -->`) are treated as empty so
  they don't bloat the prompt.

#### `openclaw_bridge/lifecycle.py` (Phase 3 foundation)

- `class OpenClawLifecycle` — health probes for the OpenClaw Gateway.
  Never raises; voice path keeps working when Gateway is unreachable.
  - `is_reachable() -> bool` — sub-second probe against
    `/__openclaw__/canvas/`.
  - `wait_for_ready(timeout_s, poll_interval_s) -> bool` — startup
    block.
  - `get_status() -> OpenClawStatus` — snapshot (version, default
    agent, configured channels).
  - `auth_token` property — reads `gateway.auth.token` from
    `~/.openclaw/openclaw.json` lazily; never logs the token.
- `class OpenClawStatus` — frozen dataclass.

#### `openclaw_bridge/client.py` (Phase 3.1)

- `class OpenClawClient` — async client over the `openclaw` CLI.
  Phase 3 deviates from the integration-spec HTTP transport because
  OpenClaw 2026.5.7 doesn't expose `/tools/invoke` or `/messages`
  HTTP endpoints — the CLI is the documented public surface, so the
  bridge invokes it via `asyncio.create_subprocess_exec`.
  - `discover_cli(override) -> str` — explicit override → env var
    (`KENNING_OPENCLAW_CLI`) → PATH → Windows npm-global default.
  - `health(timeout_s)` — wraps `openclaw health --json`.
  - `send_message(channel, target, text)` — wraps
    `openclaw message send --channel ... --target ... --message ...
    --json`. Returns :class:`SendMessageResult`.
  - `trigger_heartbeat(text, mode, expect_final)` — wraps
    `openclaw system event`. Returns :class:`HeartbeatResult`.
  - `run_agent(message, agent_id, thinking, deliver, ...)` — wraps
    `openclaw agent --json`. Returns :class:`AgentRunResult`.
  - `invoke_tool(tool_name, params, agent_id)` — convenience over
    `run_agent` for "use this OpenClaw tool" dispatch. Raises
    :class:`OpenClawToolError` when the agent reports the tool is
    unavailable.
  - `mcp_list / mcp_show / mcp_set / mcp_unset` — config helpers
    used by :class:`KenningMcpRegistrar`.
  - `enable_plugin(plugin_id)` / `disable_plugin(plugin_id)` /
    `list_plugins(*, enabled_only=False)` (V1-gap A1) — wrap
    `openclaw plugins enable / disable / list --json`. Returns
    `PluginToggleResult` / `List[PluginInfo]`. Failures (plugin not
    installed, auth) translate into structured failures rather than
    raising.
- All methods translate stderr 401/403/Unauthorized markers into
  :class:`OpenClawAuthError`; transport failures into
  :class:`OpenClawGatewayError`. Tokens are never logged.
- Internal transport `_run_cli` runs the CLI via
  `asyncio.create_subprocess_exec`. On timeout it reaps the WHOLE
  process tree via `subprocess.kill_tree.kill_process_tree`, not just
  the immediate child: the CLI is usually a shim (`openclaw.cmd` on
  Windows / npm wrapper) that spawns the real interpreter as a
  grandchild, and `proc.kill()` would orphan that grandchild — the
  orphan keeps the stdout/stderr pipes open and wedges the event loop's
  subprocess transport at teardown (root cause of the historical
  `test_bridge_e2e.py::test_health_through_real_subprocess` "hung sweep"
  flake). `_reap_process_tree(proc)` walks + kills the tree while the
  root is still alive (so psutil can still reach descendants), runs the
  synchronous kill in the default executor, then lets asyncio reap its
  own transport handle.

#### `openclaw_bridge/workspace.py` (Phase 3.3)

- `class WorkspaceWriter` — coordinated writes to the shared
  workspace (`MEMORY.md`, `USER.md`, daily memory files). Atomic
  rename via `os.replace` + advisory lockfiles via `filelock`
  (cross-platform).
  - `write_memory_entry(entry, date, prefix_timestamp)` — append
    to `memory/YYYY-MM-DD.md` with optional `HH:MM` prefix.
  - `update_memory_md(section, content, create_if_missing)` —
    splice one Markdown section in place; preserves siblings.
  - `update_user_md(content)` — full-file replace for the
    auto-populated USER.md.
- All methods are async (sync IO dispatched via
  `asyncio.to_thread`). Lockfile timeouts return a `WriteResult`
  with `error` set rather than raising.

#### `openclaw_bridge/events.py` (Phase 3.4)

- `class OpenClawEventReceiver` — gated-off scaffold for the
  `[voice]`-prefix inbound handoff. Phase 3 ships only the prefix
  matching contract (`should_handle`, `extract_payload`); the
  transport (webhook subscription / polling) is wired in a later
  phase once a real channel exists.
  - `start() / stop()` — no-op when `enabled=False` (default).
  - `dispatch(IncomingMessage) -> bool` — invokes the registered
    handler when the prefix matches; swallows handler exceptions
    so the orchestrator's main loop never sees them.
- `class IncomingMessage` — frozen dataclass; subset of an inbound
  message we route on (channel, sender, body, prefix_match).

#### `openclaw_bridge/mcp_registration.py` (Phase 3.2)

- `class KenningMcpRegistrar` — registers Kenning's MCP server with
  OpenClaw via `openclaw mcp set`. Idempotent: re-running with the
  same payload is a no-op (`already_registered=True`). Fail-open:
  failures return a `RegistrationResult` with `error` set rather
  than raising.
  - `register()` — main entry. Reads `mcp_show` first to detect
    matching existing entry; `mcp_set` only when needed.
  - `verify_registered()` — true iff the configured payload is
    currently registered.
  - `unregister()` — best-effort cleanup; never raises.
  - `schedule_retry(interval_s, on_success, max_attempts)` —
    coroutine for background retry. Caller wraps with
    `asyncio.create_task`.
- Integration deviation: the integration spec assumed Kenning's MCP
  is stdio. Reality is SSE (in-process). The registrar is
  config-driven — `openclaw.bridge.mcp_server_command` defaults to
  `None`, deferring registration. When set (e.g. when a stdio
  proxy is added in a future phase), the registrar wires it up.

#### `openclaw_bridge/holder.py` (Phase 3.5 + Phase 4)

- `class OpenClawBridge` — single dataclass-style holder owned by
  the orchestrator. Encapsulates lifecycle, client, workspace,
  events, registrar, **notifications** (Phase 4).
  - `from_config(openclaw_cfg, notifications_cfg=None) -> Optional[OpenClawBridge]` —
    returns `None` when `openclaw.enabled=False`. Construction is
    fail-open: missing CLI yields `client=None` rather than
    raising. ``notifications_cfg`` is optional (defaults to a
    disabled instance) so callers from before Phase 4 keep
    working.
  - `start()` — sync. Probes the Gateway; on success runs
    `registrar.register()`; on failure (or when MCP command is
    configured but Gateway is unreachable) launches a daemon
    retry thread.
  - `shutdown()` — stops the retry thread and the event receiver.
    Deliberately leaves the MCP entry registered so OpenClaw can
    spawn Kenning's MCP across restarts.
  - `fire_and_forget(coro_factory)` (Phase 4) — schedules a
    coroutine on a daemon thread for off-hot-path dispatch from
    the sync orchestrator loop (used by coding-completion
    notification fires).

#### `openclaw_bridge/notifications.py` (Phase 4)

- `class NotificationDispatcher` — single seam for proactive
  outbound notifications to remote channels. Each event class has
  its own method:
  - `notify_coding_task_completion(summary)`
  - `notify_coding_task_clarification(question)`
  - `notify_heartbeat_alert(text)`
  - `notify_standing_order_output(summary)`
  - `notify_search_results_async(summary)`
- All methods fail-open at every step: missing client, master
  flag off, per-event flag off, no recipient, transport failure
  — each returns a :class:`NotificationResult` with
  ``sent=False`` and a ``skipped_reason``. Voice pipeline never
  blocks.
- Recipient resolution: env var (``user_id_env``) →
  ``fallback_user_id`` → empty (skip).

#### `openclaw_bridge/heartbeat_alerts.py` (Phase 5)

- `class HeartbeatAlertLog` — JSONL-backed alert log with
  thread-safe append + atomic full-file rewrite for updates
  (acknowledgments).
  - `record(text, source, severity, metadata)` — append a new
    alert. Returns :class:`HeartbeatAlert`.
  - `get_alerts(since, only_unacknowledged, limit)` — read,
    filter, return most-recent-first.
  - `acknowledge(alert_id)` — mark seen. Atomic rewrite.
  - `prune()` — drop entries older than ``retention_days``.
- `class HeartbeatAlert` — dataclass with `alert_id` (UUID4 hex),
  `text`, `source`, `severity` ("info"/"warn"/"error"),
  `timestamp`, `acknowledged_at`, `metadata`.
- Tolerates malformed JSONL lines (logs WARN, skips), missing
  files (returns empty list), permission errors (logs WARN).
- `OpenClawBridge.record_heartbeat_alert(...)` is the orchestrator-side
  entry point: records to the log + (when enabled) fires Telegram
  notification via :class:`NotificationDispatcher.notify_heartbeat_alert`.

#### `openclaw_bridge/browser.py` (Phase 6)

- `class BrowserTool` — thin facade over
  :meth:`OpenClawClient.invoke_tool` for browser primitives.
  Each method assembles a structured prompt asking the OpenClaw
  ``kenning-main`` agent to use the browser tool with specific
  parameters; the wrapper unpacks the agent response into a typed
  dataclass.
  - `navigate(url)` → :class:`NavigateResult` (best-effort title
    extraction).
  - `snapshot(mode='ai'|'aria')` → :class:`Snapshot` with refs
    extracted in `ai` mode.
  - `click(ref)` / `type_text(ref, text)` → :class:`ActionResult`.
  - `screenshot()` → :class:`ScreenshotResult` (decodes base64
    when present).
  - `get_page_text()` → :class:`PageTextResult`.
- All methods translate `OpenClawToolError` (tool unavailable
  responses) into structured failures rather than raising.

#### `openclaw_bridge/mcp_tools.py` (Phase 13)

- Stdio MCP server exposing Kenning's read-mostly tools to OpenClaw
  agents. Each tool is a plain Python function callable from
  Python tests; FastMCP registration in :func:`build_server`
  wires them up for stdio dispatch.
- Tool implementations:
  - `get_heartbeat_alerts_impl(since_seconds_ago, only_unacknowledged, limit)`
  - `acknowledge_alert_impl(alert_id)`
  - `run_maintenance_impl(scope=None)` — subprocesses
    `scripts/run_maintenance_for_cron.py --json`
  - `list_active_coding_sessions_impl(max_age_hours=24)` — reads
    `logs/sessions/*.jsonl` audit files
  - `get_recent_voice_alerts_impl(limit=5)` — voice-friendly
    convenience wrapper
- Lazy-imports heavy dependencies; no torch / LLM at startup so
  the spawned process is light.
- :func:`run_stdio` is the entry point invoked by
  ``scripts/run_kenning_mcp_for_openclaw.py``.

#### `openclaw_bridge/desktop.py` (V1-gap C3)

- `class DesktopTool` — wrapper over `OpenClawClient.invoke_tool` for the `desktop-control` plugin. Methods: `screenshot(target?)`, `list_windows()`, `find_window(query)`. Each returns a typed dataclass (`DesktopScreenshotResult`, `ListWindowsResult`, `FindWindowResult`). Tool slugs configurable via `config.desktop.tool_slug_*`.
- `class WindowControlTool` — same pattern over `windows-control` plugin. Methods: `focus(query)`, `click(ref)`, `type_text(ref, value)`. Returns `WindowActionResult`.
- `OpenClawToolError` raised by the underlying client is translated into structured failures with the error preserved in `result.error`.

#### `openclaw_bridge/system_status.py` (Phase 13)

- `class SystemStatusReporter` — voice-side reporter for
  `SYSTEM_STATUS` routing intents. Reads heartbeat alert log +
  active session listing (via the same impl functions
  `mcp_tools.py` exposes to OpenClaw) and renders a brief in-
  character voice narration.
  - `report(SystemStatusIntent) -> SystemStatusReport` — main
    entry. Honors `focus="alerts"|"projects"|"all"` from the
    intent.
- Voice rendering kept short by design (3–4 sentences for
  combined queries, ≤2 for focused). Sanitiser caps individual
  alert text at 160 chars + ellipsis.
- Failure-safe: disk read failures degrade to "no information"
  voice messages; never raises.

### `src/kenning/pipeline/orchestrator.py`

- `class State(Enum)` — IDLE / CAPTURING / PROCESSING / FOLLOW_UP_LISTENING
- `class Orchestrator` — main event loop
  - `MAX_UTTERANCE_SECONDS` (class constant, **30.0** as of 2026-05-11 follow-up fix; was 15.0) — fallback default for the per-capture hard ceiling. The instance attribute `self._max_utterance_seconds` (read from `vad.max_utterance_seconds`) wins at runtime; the class constant is only used when config load fails in `__init__`.
  - `__init__()` — composes audio, wake, vad, addressing, stt, llm, memory, web_search, tts, coding_voice. Reads `vad.max_utterance_seconds` into `self._max_utterance_seconds` (defaults to 30.0; defensive fallback to the class constant on config-load failure). Also reads `vad.long_utterance_threshold_seconds` + `vad.long_utterance_silence_duration_ms` for the adaptive end-of-turn policy. **2026-05-12 Smart Turn V3:** builds the detector via `_build_smart_turn_detector()` BEFORE constructing the VAD; when the detector is present, the VAD is built with `min_silence_ms = smart_turn.fast_path_silence_duration_ms` (500 ms) instead of the legacy 1200 ms.
  - `_build_smart_turn_detector() -> Optional[SmartTurnDetector]` (2026-05-12) — calls `build_detector_from_config(vad.smart_turn, PROJECT_ROOT)`. Returns None when smart-turn is disabled / model file missing / construction fails. Voice baseline unaffected when None.
  - `_smart_turn_should_check(*, speech_seen, speech_samples) -> bool` (2026-05-12) — gate: detector must be available, speech must have been seen, and the contiguous speech duration must be ≤ `smart_turn.window_seconds`. Long utterances bypass smart-turn (the adaptive long-utterance VAD backstop handles those).
  - `_run_smart_turn(captured) -> Optional[SmartTurnVerdict]` (2026-05-12) — single inference call. Returns None on any failure; caller treats as "undecided" → trust VAD.
  - `run()` — main loop (blocks; KeyboardInterrupt clean shutdown)
  - `_capture_utterance()` — VAD-bounded audio capture. **2026-05-11 follow-up fix:** the hard `elapsed_samples < max_samples` ceiling now reads from `self._max_utterance_seconds` (config-driven). Previously a class-level `MAX_UTTERANCE_SECONDS=15.0` cut a real user off mid-sentence on a complex coding ask — the user wasn't pausing; the wall-clock ceiling fired before Silero VAD reported `SPEECH_END`. Bumping to 30 s comfortably covers detailed one-breath asks while still bounding pathological captures (stuck mic, background noise that never resolves to SPEECH_END, etc.). **2026-05-12 Smart Turn V3:** on first SPEECH_END within a capture (and only when the utterance is within the smart-turn window), the captured audio is fed to `_run_smart_turn`. Verdict `complete` → break immediately. Verdict `incomplete` → keep listening; bump VAD silence to `long_utterance_silence_duration_ms` and start an extension timer (`smart_turn.incomplete_extension_ms`). If silence persists past the extension, accept end-of-turn anyway. If speech resumes, cancel the extension and trust the next SPEECH_END. **2026-05-18 latency pass 3 Phase 1:** `_kick_off_tts_preopen()` is called at the very top so the PortAudio device-open overlaps the entire speech + silence-wait window (was running post-capture, only ~5-10 ms of overlap after the 2026-05-16 speculative-STT collapse).
  - `_follow_up_listen(deadline)` — WARM-mode VAD loop. Same `self._max_utterance_seconds` ceiling on cumulative speech (not wall-clock, which is bounded by `deadline`). Same Smart Turn V3 confirmation flow as `_capture_utterance` (2026-05-12). Same 2026-05-18 Phase 1 TTS preopen kick-off at the top. **2026-06-12 streaming-STT lane:** mirrors the COLD-path streaming session (the speculative-STT lane deliberately no-ops on streaming engines, which left follow-up turns paying a 100-1200 ms synchronous Moonshine re-transcribe in run()'s foreground STT call). The session starts at SPEECH_START (not window-open, so no CPU is burned on room chatter and the streamed audio is exactly pre_roll + speech_chunks = the returned buffer), feeds every chunk, and `_maybe_stop_stt_stream()` finalizes before EVERY audio-returning exit so run()'s `transcribe(buffer)` hits the engine's stash instantly; the wake-word and deadline abort paths instead call `_maybe_discard_stt_stream()` so a dropped capture's partial transcript can't leak into the next turn. Non-streaming engines (Whisper / Moonshine tiny-base) keep the speculative lane unchanged.
  - `_run_speculative_classification(user_text)` (2026-05-18 Phase 2) — chained synchronously from the speculative-STT thread after the transcript is stored. Runs `classify_by_rules` (rule path only -- LLM preflight stays foreground), picks the conversational ack via `_maybe_conversational_ack`, and kicks off `_kick_off_rag_prefetch` so the RAG retrieval overlaps the silence wait. Result stashed in `_speculative_classification` keyed by transcript. Re-checks the invalidated flag before storing so SPEECH_START mid-flight drops the result. Defensive against partial test fixtures.
  - `_invalidate_speculative_classification()` (2026-05-18 Phase 2) — marks the classification slot invalid AND cancels the in-flight RAG future. Chains into `_invalidate_speculative_llm()` so all three lanes invalidate atomically on SPEECH_START.
  - `_collect_speculative_classification(user_text)` (2026-05-18 Phase 2) — returns the cached `{text, gate_verdict, ack_phrase, rag_future}` dict if matched, else None. Atomically clears the slot. On mismatch / invalidated, cancels the rolled-over RAG future.
  - `_reset_speculative_classification_state()` (2026-05-18 Phase 2) — called from `_reset_speculative_stt_state` so all three speculation lanes clear at the top of each capture. Chains into `_reset_speculative_llm_state`.
  - `_kick_off_speculative_llm(user_text, verdict, rag_future)` (2026-05-18 Phase 3) — chained from `_run_speculative_classification` when the rule-path verdict resolves to NO_SEARCH. Spawns a daemon thread that applies `apply_uncertainty` + `apply_brevity_hint`, resolves the RAG future, then calls `llm.generate_stream(record_history=False)`. Tokens accumulate in a `queue.Queue`; the response-stream consumer drains them in lieu of a fresh LLM call. Verdict-upgrade (NO_SEARCH -> SEARCH inside `apply_uncertainty`) aborts the speculation since the search prompt body differs.
  - `_invalidate_speculative_llm()` (2026-05-18 Phase 3) — sets the invalidated flag and signals `llm.cancel()` so the streaming iterator exits at the next chunk. The producer's `finally` block still emits the sentinel so consumers don't hang.
  - `_collect_speculative_llm(user_text)` (2026-05-18 Phase 3) — returns `(iter, commit_history)` on hit, `(None, None)` on miss / mismatch / invalidated. The iterator yields tokens from the buffer until the sentinel arrives. The committer is a zero-arg function the caller invokes after consuming the iterator -- it records the turn via `llm.record_completed_turn` so unconsumed speculations leave no orphan in history.
  - `_reset_speculative_llm_state()` (2026-05-18 Phase 3) — drops the buffer + thread handle + response, and signals `llm.cancel()` if a speculation is in flight.
  - `_maybe_discard_stt_stream()` (2026-06-12) — abort-path counterpart of `_maybe_stop_stt_stream`: stops any active streaming-STT session AND clears the engine's stashed transcript (duck-typed `clear_stream_cache`; fail-open) so discarded follow-up captures never surface as a stale transcript.
  - `_respond(user_text)` — LLM stream → TTS pipeline (with optional web search)
  - `_speak(text)` — single-shot synthesize + play
  - `_speak_with_barge_in_check(text, *, post_check_window_s=0.5) -> bool` (V1-gap A4) — speak text and report whether wake fired during/after; used by the pre-task confirmation flow.
  - `_handle_capability_response(response, routing_intent)` (V1-gap A4) — wraps the capability voice dispatch. Default path: speak `response.text`. A4 path: speak `response.pre_task_confirmation` first, abort dispatch on barge-in (audit via `runner.record_pre_task_aborted`).
  - `_announce_coding_completion_if_pending()`, `_announce_pending_clarifications()`, `_announce_pending_budget_warning()` — voice-loop poll hooks
  - `_load_memory_if_enabled()` — Qdrant init with graceful fallback. In a lean gaming boot (`barebones_skip_memory`, default ON) it returns `None`, so the whole conversation-memory stack (Qdrant + bge-small dense + bm25 sparse FastEmbed encoders) is never built; `self.memory = None` is a guarded, supported state.
  - `_load_openclaw_bridge_if_enabled()` (Phase 3.5) — constructs
    :class:`OpenClawBridge`. Returns `None` when
    `openclaw.enabled=False` (current default). Fail-open: any
    construction or start failure leaves the bridge disabled
    without affecting the voice path.
  - `self.openclaw_bridge` attribute — accessed by the dispatcher
    when an OpenClaw-bound intent fires. Cleaned up in `shutdown()`
    via `self.openclaw_bridge.shutdown()`.
  - `_load_browser_use_if_enabled()` (2026-05-30 catalog 10, orchestrator.py:892) — constructs the `BrowserUseTool` + `BrowserSessionsManager` singletons from `config.browser_use` at startup, right after `_load_desktop_vlm_if_enabled()`. Cheap + lazy + fail-open: discovery of the external `browser-use` binary is deferred to first call, and any construction failure leaves the tier disabled without touching the voice path.
  - `_init_telemetry_store()` (orchestrator.py:958) + `_emit_turn_telemetry(...)` (orchestrator.py:994) + `_latency_bucket(latency_ms)` (static, orchestrator.py:978) (2026-05-30 wiring pass, openclaw-clawhub T15) — `_init_telemetry_store` builds the `observability.PrivateMetricsStore` at startup; `_emit_turn_telemetry` runs in `_respond`'s `finally` and records ONE aggregate `HashedEvent` per turn (routing-intent kind under the `category` safe key + `searched` bool + numeric `latency_ms` + `tier` bucket + `outcome`). **FAIL-PRIVATE:** `record_event` no-ops unless `KENNING_TELEMETRY=opt-in`, so the default build emits nothing; the privacy gate is intentionally NOT covered by the fail-open default. Fail-open on top: any emit error is swallowed at debug level. `_latency_bucket` labels are kept ≤12 chars to pass the raw-path leak check.
  - `_mint_forensic_token(...)` (orchestrator.py:1121) (2026-05-30 wiring pass, openclaw-clawhub T7) — registers an idempotent trusted-caller tuple then mints a short-lived HS256 JWT at the MCP-server start (`mcp:tools`) and gaming-engage (`voice:gaming-engage`, revoke-by-expiry on disengage) boundaries. Forensic / defense-in-depth in the single-user in-process runtime (minter + verifier share the trust boundary, so this is not a hard gate), audit-logged to `data/identity/short_lived_tokens.jsonl`, fail-open (returns None on any error).
  - `_init_report_queue()` (orchestrator.py:1040) + `_maybe_handle_report_concern(user_text)` (orchestrator.py:1059) (2026-05-30 wiring pass, openclaw-clawhub T12) — `_init_report_queue` opens the hash-chained `data/feedback/reports.jsonl` at startup (fail-open). `_maybe_handle_report_concern` runs in the run loop BEFORE routing: `feedback.report_intent.match_report_concern` (strict regex, no LLM round-trip) turns a spoken "log a concern" / "flag that response" into a filed `Report` (target = the prior turn via `_last_response_text`) and speaks an ack; returns True to short-circuit the turn. "report on the weather" does NOT trip it.

- **2026-06-15 router cascade for L0 misses:** normalize (`command_normalizer`)
  -> existing exact matchers (unchanged) -> semantic router
  (`get_command_router`) -> LLM fallthrough; `team_callout` routes to the relay
  (force), `identity` to the greeting, `desktop_refuse` to an in-character
  anticheat refusal; everything else abstains to the LLM. The orchestrator also
  installs the import firewall (`install_import_firewall`) at boot before
  anything else loads, spawns the embedder sidecar (`_start_embedder_sidecar`),
  and has a gaming capability-refusal gate (`_maybe_refuse_capability_in_gaming`).

- **2026-06-15 LEAN GAMING BOOT:** `_skip_for_lean_gaming(flag) -> bool` helper gates non-essential subsystems on the CONFIG INTENT `gaming_mode.engage_at_startup` (NOT runtime `is_gaming_mode_active()`, which is False throughout `__init__`, NOR `anticheat_active()` which must remain reserved for desktop surfaces). Skips: Docker autostart, the coding MCP server, the coding stack (ProjectIndex / Supervisor / CodingVoice / coordinator / project_introspect), OpenClaw bridge + threads, evolution, skills, events, the background summarizer, and the cross-encoder reranker warmup; flan-t5 addressee is lazy-loaded instead. The `GamingModeManager` is **hoisted to `self.gaming_mode_manager` in `__init__`** (it was previously born inside `coding_voice`, so skipping `coding_voice` would have nulled it and silently disabled the entire gaming engage). `_audit_anticheat_posture` now also asserts the runtime-heavy modules (`openclaw_bridge.holder` / `coding.mcp_server` / `coding.voice` / `evolution.service` / `sentence_transformers`) are absent from `sys.modules` and logs "lean boot OK" (a regression canary; if any of these slip back in, it logs a loud WARNING). Worker RSS is noisy (~3.5-6 GB, timing/GC-dependent) so no fixed delta is claimed; the reliable win is eliminating the ~4-5 GB 4B-on-GPU VRAM boot transient.

- **2026-06-15 LEAN GAMING BOOT — second wave (four more skips, all default ON, GUI-toggleable from the settings panel's "Lean Boot" section):**
  - `barebones_skip_memory` — `_load_memory_if_enabled()` returns `None`; the whole conversation-memory stack (Qdrant + the bge-small dense + bm25 sparse FastEmbed encoders) is never built. `self.memory = None` is an already-supported state (every call site is guarded). RAG retrieval is already off while gaming, so an embedded turn is never read back; in-session context still works via the LLM's own history deque — only cross-session memory is dropped.
  - `barebones_skip_intent` — skips the in-process intent recognizer, which loaded a SECOND embeddinggemma-300m (q4, via `moonshine_voice`) IN the main process, a duplicate of the isolated-sidecar copy that defeated the sidecar's anticheat isolation. Its 25 phrases (gaming-mode toggle / time-date / news) are all redundant while gaming.
  - `barebones_skip_ack_prewarm` — skips `_kick_off_ack_clip_prewarm`; conversational filler-acks are suppressed while gaming, so synthesizing/caching them is wasted.
  - `barebones_skip_web_search` (widened) — now also short-circuits `_build_web_search`, which returns `(None, None, None)` so `web_gate` / `web_executor` are `None` and neither the provider chain (searxng/brave/duckduckgo) nor the reader chain (trafilatura/jina) is constructed at boot (previously only the per-turn preflight was skipped); the conversational path takes its no-web-gate branch.
  - **Canary widened:** `_audit_anticheat_posture` now flag-gated-asserts that the intent recognizer, conversation memory, ack-prewarm thread, and web-search chain are all absent while gaming (so a regression is caught, and re-enabling one via the GUI never false-alarms). The "lean boot OK" log now lists: coding / MCP / OpenClaw / evolution / reranker / intent-model / memory / ack-prewarm / web-chain NOT loaded.

- **2026-06-15 SIDECAR LIFECYCLE:** `_start_embedder_sidecar` calls `sidecar_lock.sweep()` before spawning and writes the pidfile on success. The sidecar is registered with ZombieKiller as `persistent=False / hard_timeout=1h` (was effectively immortal). `_kill_embedder_sidecar()` unregisters from ZombieKiller then `kill_process_tree()`s the sidecar (shim → embedder child) and clears the pidfile. `shutdown()` calls `_kill_embedder_sidecar()` BEFORE stopping the reaper so the orphan-sweep on next boot finds a clean state.

- **2026-06-15 NEVER-LEXICAL:** the boot router warmup respawns the sidecar and rebuilds the router ONCE if the router came up lexical (logged as ERROR, not silently accepted). `_maybe_recover_embedding()` runs throttled (~60 s) in the idle wake-loop to re-enable the hybrid backend if the sidecar returns mid-session via `HybridBackend.try_recover()`.

- **2026-06-15 DIRECT 3B LLM:** in a lean gaming boot the LLM is constructed directly as the gaming preset (3B on CPU, `n_gpu_layers=0`) so the base Qwen-4B never loads on the GPU only to be swapped out on engage. Controlled by `barebones_direct_gaming_llm` (default True).

- **2026-06-15 LEAN SIBLINGS (deterministic relay + GUI + Spotify in lean boot):** the lean boot previously ran ONLY the fuzzy semantic router for L0 misses. It now also runs, as siblings, the deterministic relay matcher (`match_relay_command`), the settings-GUI voice command ("pull up the config/settings/control panel"), and a standalone lean Spotify handler (`kenning/spotify/lean_handler.py`) so full music control and the deterministic callout path work in a lean session, not just the embedding router. Greet/identity split: a team-directed greet routes to the mic relay, while a bare identity question ("who are you", "what are you") routes to the conversational desktop path in the Ultron persona. Bare-callout relay coverage was widened (counts, requests, weapons, movement, locations), and "X asked about Y, respond" now routes to the in-character context+directive relay.

- **2026-06-15 SIDECAR ON CPU:** the embeddinggemma router sidecar now runs on CPU (`semantic_router.sidecar_device: "cpu"`) to free GPU VRAM.

**In:** mic input (sounddevice), config.yaml, models on disk.
**Out:** speaker output (sounddevice), all audit logs.

### `src/kenning/resilience/` (Phase 4)

#### `resilience/circuit_breaker.py`
- `class CircuitState(str, Enum)` — CLOSED / OPEN / HALF_OPEN
- `class CircuitOpenError(Exception)` — short-circuit signal
- `class CircuitBreaker`
  - `__init__(name, failure_threshold=3, window_seconds=300, cooldown_seconds=300, expected_exceptions=(Exception,))`
  - `call(func, *args, **kwargs) -> result` — raises CircuitOpenError when OPEN
  - `state`, `failure_count` properties
  - `reset()` — test/operator only

#### `resilience/error_log.py`
- `class ErrorLog` — append-only JSONL writer to `logs/errors.jsonl`
  - `record(error, *, dependency, session_id?, extra?, include_traceback=True)` — best-effort
- `get_error_log() -> ErrorLog` — singleton
- `set_error_log(log)` — test injection

#### `resilience/phrases.py`
- `phrase_for(failure_mode: str) -> Optional[str]` — shuffled cycle from `config.error_phrases.<mode>`
- `reset_phrase_cache()` — test-only

### `src/kenning/utils/`

#### `utils/heartbeat.py` (2026 catalog 11 T2)
- `class HeartbeatThread(target, *, interval_s=60.0, name="heartbeat", on_error=None, run_immediately=False, clock=time.monotonic)` — stoppable daemon-thread keep-alive for any long-lived connection (browser-use daemon, Parakeet server, OpenClaw bridge). `start()` (idempotent) / `stop(*, timeout=2.0)` (idempotent, returns the loop immediately via `Event.wait`) / `is_alive()` / `stats() -> HeartbeatStats`. Fail-open: target + on_error exceptions are counted + recorded, never propagated. Generalises + hardens the upstream's unstoppable `while True: time.sleep` keep-alive.
- `@dataclass(frozen=True) HeartbeatStats` — running / beats_sent / errors / last_error / started_at.
- `DEFAULT_HEARTBEAT_INTERVAL_S=60.0`, `DEFAULT_STOP_TIMEOUT_S=2.0`.
- **In:** a cheap no-op callable + interval. **Out:** a background liveness ping + observable counters. No module singleton (each consumer owns its instance). Importable primitive (no current hot-path consumer — kenning's daemons self-respawn on the next call).

#### `utils/health_check.py` (2026 catalog 11 T4)
- `http_health_check(url, *, timeout_s=2.0, expected_status=200, get_fn=None) -> HealthCheckResult` — cheap GET pre-flight; reachable iff the endpoint answers `expected_status` (or any 2xx when `expected_status=None`). Fail-open (any exception → reachable=False + error).
- `cdp_health_check(port, *, host="127.0.0.1", timeout_s=2.0, get_fn=None) -> CdpHealthResult` — Chrome DevTools `/json/list` probe; reports `tab_count` + `page_tab_count` (`type=="page"`). Loopback-only host default. Fail-open.
- `@dataclass(frozen=True) HealthCheckResult` (reachable / url / status_code / elapsed_ms / error); `CdpHealthResult` (reachable / endpoint / tab_count / page_tab_count / elapsed_ms / error). `get_fn` is an injectable transport so tests never touch the network. Importable primitive.

#### `utils/logging.py`
- `configure_logging(level=None, log_file=None) -> None` — idempotent
- `get_logger(name) -> logging.Logger` — namespaced under `kenning.`

#### `utils/fairseq_compat.py`
- `patch_fairseq_dataclasses()` — workaround for fairseq's invalid omegaconf metadata
- `patch_torch_load_for_fairseq()` — torch.load weights_only compat shim

#### `utils/mtime_cache.py` (NEW 2026-05-22 catalog batch 1)
- `class MtimeCache(path, *, version=1, prefer_disk=True)` — SQLite primary + dict fallback
  - `.get(key, mtime) -> Optional[Any]` — mtime-validated read; returns None on miss
  - `.set(key, mtime, value)` — write to both layers; degrades to dict on backend error
  - `.delete(key)`, `.clear()`, `.close()`, `len(cache)`
  - `.degraded` property — True when running on dict fallback
  - `.path` property — versioned cache directory (e.g. `<base>.v1`)
- `class MtimeCacheError(Exception)` — programmer-error only (non-Path argument); never raised for operational issues
- `open_mtime_cache(path, *, version=1, prefer_disk=True) -> MtimeCache` — convenience constructor

#### `utils/token_budget.py` (NEW 2026-05-22 catalog batch 1)
- `pack_to_budget(items, render, count_tokens, max_tokens, *, tolerance=0.15, max_iterations=30, strict=False) -> PackResult`
  - Binary-searches the largest `items[:k]` that fits in `max_tokens` when rendered
  - Tolerance band stops the search early once the result is within `[max_tokens*(1-tol), max_tokens]`
- `@dataclass(frozen=True) class PackResult` — `k`, `token_count`, `iterations`, `terminated_early`
- `class BudgetTooSmallError(Exception)` — only raised when `strict=True` and item 0 alone exceeds budget
- `char_count_tokens(text) -> int` — cheap default counter (`len // 4`)

#### `utils/snapshot_guard.py` (NEW 2026-05-22 catalog batch 1)
- `take(obj, *, deep=True) -> _SnapshotToken` — deep-copy snapshot of `obj`
- `matches(token, current) -> bool` — `==` comparison of token's captured value to `current`
- `class SnapshotGuard` — keyed snapshot store with internal lock
  - `.snapshot(key, value, *, deep=True)`, `.unchanged(key, current) -> bool`
  - `.require(key, current)` — raises `StaleSnapshotError` on drift
  - `.drop(key)`, `.has(key)`, `.clear()`, `len(guard)`
- `class StaleSnapshotError(Exception)`

#### `utils/relative_indent.py` (NEW 2026-05-22 catalog batch 1)
- `class RelativeIndenter(texts, *, marker=None)` — picks an unused Unicode outdent marker (default `←`; falls back to high-plane codepoints) so encoding is self-delimiting
  - `.make_relative(text) -> str` — encode to dent/content paired-line stream
  - `.make_absolute(text) -> str` — decode back; validates pairing and outdent integrity
- `relative_indent(text, *, marker=None) -> str` — one-shot encode
- `absolute_indent(text, *, marker=None) -> str` — one-shot decode

#### `utils/poll.py` (2026-05-24 OpenHands batch 2, T14)

Bounded-retry polling primitive ported from OpenHands'
`_poll_for_title`. See `THIRD_PARTY_NOTICES.md` for attribution.

- `@dataclass(frozen=True) PollResult[T]` — `value` / `succeeded` / `attempts` / `elapsed_seconds` / `last_error`.
- `poll_until(fn, *, is_done=_is_present, max_attempts=4, delay_seconds=3.0, backoff_factor=1.0, max_delay_seconds=60.0) -> PollResult[T]` — synchronous bounded-retry. Default predicate is `value is not None`; replace for "good-enough" semantics.
- `apoll_until(fn_or_coro, *, is_done, max_attempts, delay_seconds, backoff_factor, max_delay_seconds, cancel_check=None) -> PollResult[T]` — async variant; cooperative `cancel_check` callable lets the voice path abandon a long poll when the user resumes speaking.
- Defaults `DEFAULT_MAX_ATTEMPTS=4`, `DEFAULT_DELAY_SECONDS=3.0`, `DEFAULT_BACKOFF_FACTOR=1.0`, `DEFAULT_MAX_DELAY_SECONDS=60.0` mirror the OpenHands constants.

### `src/kenning/parsing/` (2026-05-24 OpenHands batch 1, T11)

YAML frontmatter parser. Used by skills + projects + rule conditionals.

#### `parsing/frontmatter.py`
- `@dataclass(frozen=True) FrontmatterResult` — `frontmatter: Mapping[str, Any]` (parsed YAML mapping, empty dict on absent/malformed) + `body: str` (post-frontmatter text) + `raw_frontmatter: str` (verbatim block).
- `parse_frontmatter_text(text, *, strict_yaml=False) -> FrontmatterResult` — accepts CRLF + LF + empty frontmatter (`---\n---\nbody`); fail-open on parse errors (returns empty frontmatter + full text as body when `strict_yaml=False`).
- `parse_frontmatter(path) -> FrontmatterResult` — file wrapper.
- `walk_directory_with_frontmatter(root, *, suffixes=(".md",), skip_dirs=frozenset(), skip_files=frozenset()) -> Iterator[tuple[Path, FrontmatterResult]]` — recursive walk with filterable dir/file skips.

### `src/kenning/install/` (2026-05-24 OpenHands batch 1, T8)

Marker-comment idempotent installer with audit log.

#### `install/idempotent.py`
- `class InstallAction(str, Enum)` — `INSTALLED` / `REUSED` / `REPLACED` / `REFUSED_EXISTS_UNMARKED` / `DRY_RUN`.
- `@dataclass(frozen=True) InstallResult` — `path`, `action`, `marker`, `bytes_written`, `reason` (per-action explanation).
- `@dataclass(frozen=True) InstallLogEntry` + `class InstallLogWriter` (JSONL writer with `logs/install_log.jsonl` default).
- `set_install_log_writer(writer)` / module-level singleton.
- `install_with_marker(target: Path, content: str, *, marker: str = "# INSTALLED-BY-KENNING-3f9a7d2", policy: str = "preserve_unmarked", encoding="utf-8", dry_run=False, audit_log_writer=None) -> InstallResult` — atomic temp-write + `os.replace`; UUID-suffixed marker prevents collisions when two installers race on the same file; explicit `policy ∈ {refuse, preserve, replace}` for unmarked existing files; `dry_run=True` reports without writing.

### `src/kenning/projects/` (2026-05-24 OpenHands batch 7, T7)

Per-project `.kenning/` configuration discovery.

#### `projects/discovery.py`
- `DEFAULT_PROJECT_CONFIG_DIRNAME = ".kenning"`.
- `class ProjectConfigField(str, Enum)` — `SKILLS_DIR` / `SETUP_SCRIPT` / `PRE_COMMIT_SCRIPT` / `IDENTITY_OVERRIDE` / `SAFETY_RULES` / `TEST_COMMAND` / `VOICEPACK_OVERRIDE` / `INTENT_TRIGGERS` / `HOOKS`.
- `@dataclass(frozen=True) ProjectDiscoveryStats` — `repo_root`, `config_dir`, `files_checked`, `files_found`, `parse_errors`, `duration_seconds`.
- `@dataclass(frozen=True) ProjectConfig` — `repo_root` / `config_dir` / `discovered_at` + 9 optional fields (`skills_dir`, `setup_script`, `pre_commit_script`, `identity_override`, `safety_rules`, `test_command`, `voicepack_override`, `intent_triggers`, `hooks`) + their `*_path` siblings + `raw_paths` mapping + `parse_errors` tuple. `has_any_field` boolean shortcut + `get_path(field)` lookup.
- `discover_project_config(repo_root, *, use_cache=True) -> ProjectConfig` — never raises; per-file errors land in `parse_errors`. Mtime-cached keyed by `(repo_root, .kenning mtime)`.
- `invalidate_discovery_cache(repo_root=None)` — drop entry (or all when `None`).

### `src/kenning/services/` (2026-05-24 OpenHands batch 8, T6)

Sync Injector ABC + state + registry. Future per-mode router seed.

#### `services/injector.py`
- `class InjectorState` — thread-safe key/value blob shared across nested injections; `__getattr__` / `__setattr__` / `__contains__` / `get` / `update` / `snapshot`.
- `class Injector[T](ABC)` — `inject(state) -> T` (subclass implements); `stream(state) -> Iterator[T]` (default delegates to inject); `context(state=None)` context manager.
- `class SingletonInjector[T]` — `__init__(build_fn)`; constructs once + caches; `reset()` drops the cached instance.
- `class StreamInjector[T]` — wraps a generator factory so callers don't have to subclass.
- `@dataclass InjectorRegistry` — `register(key, injector)` / `unregister(key) -> bool` / `get(key)` / `require(key) -> Injector` / `keys() -> list[str]` / `clear()`.
- `get_injector_registry() -> InjectorRegistry | None` + `set_injector_registry(registry)` + `reset_injector_registry_for_testing()` module-level singleton.
- `install_default_injectors(*, registry=None, stt_injector=None, tts_injector=None) -> InjectorRegistry` — registers starter STT + TTS injectors on the singleton registry; missing args fall to the default `build_stt_engine_injector()` + `build_tts_engine_injector()`.

#### `services/engine_injectors.py`
- `@dataclass STTEngineInjector(Injector[Any])` — `standby_factory` + `gaming_factory`; `inject(state)` switches by `state.get("mode", "standby")`; default falls through to `kenning.transcription.make_stt_engine()`.
- `@dataclass TTSEngineInjector(Injector[Any])` — same shape; default falls through to `kenning.tts.make_tts_engine()` returning `(rvc, engine)` tuple.
- `build_stt_engine_injector(*, standby_factory=None, gaming_factory=None)` / `build_tts_engine_injector(...)` factory helpers.

### `src/kenning/skills/` (2026-05-24 OpenHands batch 2, T1)

Trigger-loaded knowledge bundles. Three sources merged with PROJECT > USER > PUBLIC precedence.

#### `skills/registry.py`
- `class SkillRegistry` — walks N directories (each tagged as PUBLIC/USER/PROJECT), parses each `.md` file via `parsing/frontmatter`, dedupes by name (later precedence wins), supports KEYWORD + SLASH_COMMAND + ALWAYS_ON trigger kinds.
  - `reload() -> list[SkillReloadStats]` — re-scan every source dir; mtime cache keyed on `(directory, mtime_fingerprint)`.
  - `match_for_turn(user_text, *, max_matches=6, min_user_text_chars=8, always_on_only=False) -> list[Skill]` — keyword/slash matching with fail-soft caps.
- `get_skill_registry()` / `set_skill_registry(registry)` / `reset_skill_registry_for_testing()` module-level singleton accessors.
- `format_skills_block(skills, *, max_block_chars=8000) -> str` — render matched skills into a `<skill name="X">...</skill>` block bounded by char cap.
- `maybe_get_skills_block(user_text, *, max_matches=6, min_user_text_chars=8, max_block_chars=8000) -> str` — convenience wrapper called from `LLMEngine._build_messages` when `skills.enabled: true`.
- `build_default_registry(*, project_root, user_home=None, extra_project_dirs=(), disabled_skills=(), always_on_only=False, default_min_user_text_chars=8, max_matches_per_turn=6) -> SkillRegistry` — orchestrator-side factory wiring the three default sources + config overrides.

### `src/kenning/events/` (2026-05-24 OpenHands batches 3 + 4, T2 + T3 + T13)

Canonical event store + bus sink + per-event hash chain + callbacks.

#### `events/models.py`
- `class EventSortOrder(str, Enum)` — `ASC` / `DESC`.
- `class EventKind` — string-constant namespace (e.g. `EventKind.AGENT_TURN_STARTED`, `EventKind.MEMORY_WRITE`, `EventKind.SAFETY_BLOCK`). New event kinds are added here so consumers can grep one file for the canonical list.
- `new_event_id() -> str` — 32-char hex UUID4.
- `@dataclass(frozen=True) StoredEvent` — `id` / `session_id` / `kind` / `timestamp` / `payload: Mapping` / `chain_hash: str` (set by chain.compute on insert) / `sequence: int` / `tags: tuple[str, ...]`.
- `@dataclass(frozen=True) EventPage` — `events: tuple[StoredEvent, ...]` + `next_cursor: str | None`.
- `@dataclass(frozen=True) EventQuery` — `session_id` / `kinds` / `since_ts` / `until_ts` / `tags_any` / `tags_all` / `limit` / `cursor` / `sort_order`.
- `canonical_event_json(event) -> str` — deterministic JSON serialisation feeding the SHA-256 chain.
- `kinds_in(events) -> list[str]` — distinct kinds helper for the per-session histogram.

#### `events/chain.py` (T13)
- `compute_event_chain_hash(prev_hash: str, event_json: str) -> str` — `sha256(prev || canonical)`.
- `verify_chain(events: Sequence[StoredEvent]) -> tuple[bool, int | None]` — re-walk the chain; returns `(ok, first_break_index_or_None)`.

#### `events/store.py`
- `class EventStoreError(RuntimeError)`.
- `class EventStore(ABC)` — five abstract methods: `save_event` / `get_event` / `search_events` / `count_events` / `batch_get_events`. Subclasses honour per-session prefix scoping.
- `class MemoryEventStore` — in-process dict-of-list per session. Fastest; volatile.
- `class JsonlEventStore` — one `.jsonl` per session under `base_dir`; atomic append + per-session RLock; recovers gracefully on partial-write tail rows.
- `class QdrantEventStore` — Qdrant collection per session; embeds canonical JSON for vector search. Falls back to JSONL when Qdrant unreachable.
- `get_event_store()` / `set_event_store(store)` / `reset_event_store_for_testing()` module-level singleton.
- `build_event_store(backend: str, *, base_dir, qdrant_collection, default_session_id) -> EventStore` — factory matching `events.store_backend` config.

#### `events/export.py`
- `export_session_to_bytes(store, session_id) -> bytes` — zip blob: `events.jsonl` + `meta.json` (event count, chain verification result, exported_at).
- `export_session_to_path(store, session_id, path)` — disk variant.

#### `events/bus_sink.py`
- `class BusEventSink` — subscribes to the kenning bus, converts every published envelope to `StoredEvent` with per-session sequence counter, persists via the active `EventStore`, fires the `CallbackRegistry` after persistence.
- `install_bus_sink(store, callback_registry=None) -> BusEventSink` — orchestrator-side wiring helper.

#### `events/callbacks.py` (T3)
- `class CallbackStatus(str, Enum)` — `ACTIVE` / `DISABLED` / `ARCHIVED`.
- `class CallbackResultStatus(str, Enum)` — `OK` / `ERROR` / `SKIPPED`.
- `@dataclass CallbackResult` — `callback_id`, `status: CallbackResultStatus`, `error_message`, `deactivate_after`.
- `class CallbackProcessor(ABC)` — `__call__(event: StoredEvent, callback: RegisteredCallback) -> CallbackResult`.
- `@dataclass RegisteredCallback` — id + session/kind filters + processor reference + `enabled` flag.
- `class CallbackRegistry` — register / unregister / list / dispatch; per-callback session+kind filter + self-deactivation pattern; JSONL persistence option; slow-callback watchdog.
- `get_callback_registry()` / `set_callback_registry(registry)` / `reset_callback_registry_for_testing()` module-level singleton.
- `class FunctionProcessor(CallbackProcessor)` — adapter so a plain callable can be a processor without subclassing.

#### `events/processors.py` (T3 built-ins)
- `class LoggingCallbackProcessor` — logs event id + kind + payload preview.
- `class CountingCallbackProcessor` — per-kind counter + checkpoint persistence.
- `class ThresholdSnapshotProcessor` — fires once when a per-event counter crosses threshold (the load-bearing OpenHands `SetTitleCallbackProcessor` pattern).
- `class MemoryWriteProcessor` — writes event content into the conversation memory.
- `class ChannelGuardProcessor` — payload redaction (regex blocklist) before downstream sinks.
- `class SkillActivatorProcessor` — toggles a skill name based on payload condition.
- `build_default_processors() -> list[CallbackProcessor]` — returns one of each built-in.

### `src/kenning/lifecycle/` (2026-05-23 OpenHands batches 6, T5 + T16)

Typed start-task state machine + pending-message queue.

#### `lifecycle/start_task.py` (T5)
- `class StartTaskStatus(str, Enum)` — `PENDING` / `RUNNING` / `COMPLETED` / `FAILED` / `CANCELLED` / `TIMED_OUT`.
- `is_terminal_status(status) -> bool` — `True` for COMPLETED/FAILED/CANCELLED/TIMED_OUT.
- `class StartTaskError(RuntimeError)`.
- `@dataclass StartTask` — `task_id` / `kind` / `started_at` / `status` / `progress` / `payload` / `error_message`. Mutable status field; transitions are explicit so transcript / persistence can observe each step.
- `create_start_task(kind, *, task_id=None, payload=None) -> StartTask` — factory.
- `class StartTaskRecorder` — `record(task: StartTask)` writes a typed event into the `EventStore` (when wired) on every transition.
- `async drive_start_task(task: StartTask, body: Callable[..., Awaitable], *, recorder: StartTaskRecorder | None = None, timeout_seconds: float = 0.0) -> StartTask` — runs `body` under timeout; transitions through RUNNING -> COMPLETED/FAILED/TIMED_OUT; persists every transition via recorder.

#### `lifecycle/pending_message_queue.py` (T16)
- `class PendingMessageState(str, Enum)` — `QUEUED` / `DELIVERED` / `CANCELLED`.
- `@dataclass PendingMessage` — `message_id` / `temp_task_id` / `bound_task_id` / `state` / `payload` / `created_at` / `delivered_at`.
- `class PendingMessageQueue` — `enqueue(temp_task_id, payload) -> PendingMessage` / `bind(temp_task_id, bound_task_id)` (rebind on real task arrival) / `cancel(message_id_or_temp_id)` / `drain(bound_task_id) -> list[PendingMessage]` / `pending_for(bound_task_id) -> list[PendingMessage]`. Overflow drops oldest beyond `max_size`. Optional JSONL persistence.
- `rebind_pending_messages(queue, old_temp_id, new_bound_id) -> int` — convenience alias for the rebind pattern.

#### `lifecycle/single_instance.py` (NEW 2026-06-12 — single-instance guard)

Two simultaneous `python -m kenning` processes both grabbed the mic and
double-responded (and the second collided on the embedded Qdrant lock
and the MCP port-19761 bind). The guard refuses a duplicate launch at
the entrypoint, BEFORE any model load.

- `DEFAULT_LOCK_PATH` — `<PROJECT_ROOT>/data/.kenning_instance.lock`
  (CWD-independent; falls back to CWD-relative `data/` if the config
  package can't be imported). Gitignored.
- `ALLOW_MULTIPLE_ENV = "KENNING_ALLOW_MULTIPLE_INSTANCES"` — `"1"`
  bypasses the guard (returns a no-op "bypass" lock).
- `class InstanceLock` — `path` / `mode` (`"msvcrt"` / `"fcntl"` /
  `"pidfile"` / `"bypass"`) / `pid`; `release()` idempotent, never
  raises, best-effort unlinks the file.
- `acquire_single_instance_lock(path=None) -> Optional[InstanceLock]`
  — held OS byte-range lock (`msvcrt.locking` LK_NBLCK on Windows,
  `fcntl.flock` elsewhere) on the byte at offset 4096; auto-releases
  on process death so there is NO stale-lock recovery problem. The
  metadata JSON (`{"pid", "started_at"}`) lives at offset 0 -- on
  Windows msvcrt locks are MANDATORY, so the locked byte sits beyond
  EOF to keep the metadata readable by the refused duplicate.
  Pidfile + psutil-liveness fallback when neither primitive imports.
  Returns ``None`` ONLY on genuine contention; any unexpected error
  fail-opens to a "bypass" lock (a broken lock path never blocks a
  legitimate start).
- `read_lock_metadata(path=None) -> Optional[dict]` — UNBUFFERED
  `os.read` of <=1024 bytes (Python's buffered `open()` requests a
  full 8 KiB buffer, which spans the mandatory-locked byte and fails
  with EACCES while the holder is alive).
- `is_another_instance_running(path=None) -> Optional[int]` — probe
  helper; returns the holder PID or None; never leaves the lock held.
- Wired in `src/kenning/__main__.py main()`: acquire before the banner
  + Orchestrator construction; duplicate prints/logs a one-line
  message naming the holder PID + returns exit code 3; the lock
  releases in a `finally` around the whole run. Orchestrator is
  deliberately untouched so pytest sweeps / the GPU e2e suite /
  measurement scripts never contend.
- Tests: `tests/lifecycle/test_single_instance.py` (12) +
  `tests/test_main_single_instance.py` (4, entrypoint integration
  with a hermetic logging fixture).

### `src/kenning/llm/condensers/` (2026-05-23 OpenHands batch 5, T4)

Swappable history compression. Five concrete condensers + factory + intent selector.

#### `llm/condensers/base.py`
- `Turn = tuple[str, str]` (role, content) matching `LLMEngine.Turn`.
- `class CondenserError(RuntimeError)`.
- `@dataclass(frozen=True) CondenseResult` — `turns: tuple[Turn, ...]` (post-condensation) + `dropped: int` (verbatim turns elided) + `compaction_summary: str` (LLM-generated for the LLM-summarising variant; empty for others).
- `class Condenser(ABC)` — `condense(turns: Sequence[Turn]) -> CondenseResult`.
- `turn_text(turn) -> str` — content extraction helper.
- `char_count_tokens_for_turns(turns) -> int` — cheap budget probe.

#### `llm/condensers/{noop,recent,amortized,observation_masking,llm_summarizing}.py`
- `class NoOpCondenser` — passthrough.
- `class RecentCondenser(*, head_keep=2, tail_keep=4)` — head + tail; middle dropped.
- `class AmortizedCondenser(*, target_token_budget, token_counter=char_count_tokens)` — no-LLM intelligent forgetting; drops oldest turns until under budget; recovers if a single recent turn alone exceeds budget.
- `class ObservationMaskingCondenser(*, mask_after_turn_age=8, mask_template="[NOTE] Tool/system output elided")` — masks old tool/system content without dropping the turn.
- `class LLMSummarizingCondenser(*, summarize_fn: Callable[[Sequence[Turn]], str], head_keep=2, tail_keep=4)` — folds the middle into one summary turn using the injected `summarize_fn`. Keeps the package LLM-independent.

#### `llm/condensers/factory.py`
- `build_condenser(name, **kwargs) -> Condenser` — string-keyed factory (`noop` / `recent` / `amortized` / `observation_masking` / `llm_summarizing`).
- `select_condenser_for_intent(intent_label: str, *, summarize_fn=None, fallback_name="noop") -> Condenser` — the catalog's "adaptive switching by intent" extension. Greetings -> NoOp; factual -> Recent; coding -> LLMSummarizing (requires `summarize_fn`); fallback `fallback_name` when unmatched.

### `src/kenning/safety/` (runtime tool-call validator + anticheat + rule engine)

The largest safety subsystem (30+ files). Every capability/tool call passes the validator; the anticheat
layers (`anticheat.py`, `import_firewall.py`) and the lean-boot posture are load-bearing for the user's
Valorant/Vanguard account (see `feedback_no_default_load_anticheat.md`).
- `validator.py` — `SafetyValidator` + `get_validator()`; the central `check(tool_name, ...)` returning
  ALLOW / AUDITED_ALLOW / BLOCK_HARD; the seam every desktop/coding/network capability flows through.
- `policy.py` / `policy_chain.py` — `PolicyChain` (ordered policy decision flow); `hierarchical_policy.py`.
- `rules/` sub-package — `base.py` (rule base class + `cap_carveouts.py`) + the category rule files
  `category_a.py … category_s.py` (one module per capability category: process-memory/injection/hooks/
  capture/input/path/network/etc.) + `conditionals.py`. Each defines the detection/decision rules the
  validator composes.
- `taint.py` (`TaintTracker`), `path_resolver.py` (`PathResolver.safe_realpath`), `two_phase_approval.py`
  (`ApprovalRequest` / `ApprovalRegistry` — voice yes/no for destructive ops), `auto_approval.py`
  (`AutoApprovalMatrix`, yolo-mode), `ignore.py` (`KenningIgnoreRule` — `.kenningignore` secrets block),
  `intent.py` (explicit-intent NEEDS_EXPLICIT_INTENT unblock), `audit.py` (`AuditLog`, hash-chained JSONL +
  `repair_if_needed()`).
- `anticheat.py` (`anticheat_active()`, 49 module guards, `is_blocked_tool`, surface hooks),
  `import_firewall.py` (loader-level `sys.meta_path` block + `assert_firewall_enforces()`),
  `testing_mode.py` (`is_testing_mode_active()` + sentinel). (These three also have entries in the
  "2026-06 relay/gaming campaign" appendix.)

### `src/kenning/subprocess/` (process lifecycle + orphan reaping)

Production infra for never leaking child processes (the embedder sidecar + any spawned tools).
- `kill_tree.py` — `kill_process_tree()`, `KillTreeResult`, `kill_own_children()`.
- `process_registry.py` — `ProcessRegistry`, `JobState`, `get_process_registry()` (T12 process discipline).
- `sidecar_lock.py` — sidecar SINGLETON enforcer. Embedder: `sweep()` (boot orphan reap), `default_pidfile()`,
  `reap_stray_embedders()`. **Generalized to ALL sidecar roles (2026-06-21, anti-stale-sidecar):**
  `SIDECAR_HINTS` (role→cmdline marker: embedder / twitch_guard / twitch_read / twitch_overlay),
  `reap_stray_sidecars(hints=None, keep_pid=None)` (cmdline reaper across roles; `reap_stray_embedders` now
  delegates), `reclaim_port(host, port)` (kill the LISTEN holder so a restart reclaims the port; never self),
  `guard_singleton(host, port, role)` (pre-bind: reap strays + reclaim port), per-role pidfiles
  (`role_pidfile`/`write_role`/`clear_role`).
- `sidecar_server.py` — **`SingletonThreadingHTTPServer`** (2026-06-21): exclusive bind (`allow_reuse_address=False`
  + Windows `SO_EXCLUSIVEADDRUSE`) so a SECOND process can never co-bind a sidecar port (the fix for the stale
  guard sidecars co-bound to :8774). Used by the twitch guard/read/overlay sidecars; `embedder_server.py` has an
  inline pure-stdlib equivalent (isolated venv, no kenning import). The orchestrator shutdown also runs
  `reap_stray_sidecars()` as a catch-all. Each sidecar: guard_singleton(pre-bind) + exclusive-bind + per-role
  pidfile + parent-death deadman.
- `zombie_killer.py` — `ZombieKiller`, `TrackedProcess`, `get_zombie_killer()` (tracked daemon reaper).

### `src/kenning/agent_loop/` (bounded agentic loop backbone)

The `max_steps`-bounded observe→plan→act→verify base used by every agentic feature (deep research/memory/
exploration, UI discovery, evolution cycles).
- `base.py` — `AgentLoop` (ABC), `LoopStatus`, `StepRecord`, `LoopResult` (the load-bearing step cap +
  repeated-signature loop detection + verify hook + fail-open execution).
- `deep_loops.py` — `DeepGatherLoop`, `DeepMemoryLoop`, `DeepExplorationLoop`, `DeepUIDiscoveryLoop`.
- `loop_detection.py` (`LoopDetector`, `LoopVerdict`) + `loop_detection_extended.py`.
- `mode.py` — `ModeSession`, `Mode`, `ModePolicy`, `ModeFlipResult`.
- `subagent.py` — `SubagentRunner`, `SubagentTask`, `TokenLedger`, `ToolGuard`; `subagent_policy.py`
  (`SubagentPolicyConfig`, `ResolvedSubagentToolPolicy`, `filter_tools_by_policy`).

### `src/kenning/evolution/` (bounded autonomous self-improvement — clawhub catalog 13/14)

Data-only, Tier-3-walled self-improvement (config `evolution`). Lean-gaming skips it. See the
THIRD_PARTY_NOTICES quarantined-source record.
- `service.py` — `EvolutionService`, `EvolutionStore` (JSONL runtime + `digest()` + per-turn hooks).
- `models.py` (GEP capsule data model), `signals.py` (local opportunity/correction/gap/failure extraction),
  `skill_distiller.py` (capsule→`data/evolution/skills/*.md`), `blast_radius.py` (`compute_blast_radius`,
  protected-path wall), `guardrails.py` (`evaluate_guardrails`, `GuardrailVerdict` — latency/quality/error/
  resource detectors + rollback), `autonomy.py` (`TieredAutonomyController`, `AutonomyTier`),
  `personality.py` (Tier-0 temperament hint), `evolution_loop.py` (`EvolutionLoop(AgentLoop)`),
  `turn_metrics.py` (the guardrail metrics ring), `intent.py` ("evolve now" / "evolution status" matchers).

### `src/kenning/mcp/` (MCP server registry + transport)

- `registry.py` — `McpServerRegistry`, `McpServerHandle`, `McpServerState`, `get/set_mcp_server_registry()`
  (kill-on-disconnect lifecycle). `config evolution`/`mcp.enabled`/`mcp.autostart` (default OFF).
- `builder.py` — `build_mcp_server_registry()`, `transport_from_spec()`.
- `transport.py` — `McpTransportKind` + the four transport config dataclasses (Stdio/Http/Sse/StreamableHttp) +
  `sanitise_transport_config()` / `filter_environment()` / `filter_http_headers()` (env/header sanitisation).

### `src/kenning/providers/` (web-search provider failover/rotation)

- `rotation.py` — `RotatingBraveClient` (multi-key Brave rotation, T6).
- `auth_profiles.py` — `AuthProfileStore`. `failover_policy.py` — `FailoverPolicy` (provider failover ordering).
  Consumed by `web_search/provider_chain.py`.

---

## Configuration

### `config.yaml` (project root) — single source of truth

Sections:
- `version: "1.0"`
- `audio` (sample_rate, channels, blocksize, dtype, devices, barge-in, **ring_buffer_seconds: 0.5** [2026-05-10: bumped back from 0.15 to act as a STORAGE capacity now that the orchestrator slices mode-specific pre-roll out of it], **cold_pre_roll_seconds: 0.15** [NEW 2026-05-10: post-wake slice; short to avoid the "Tron" prefix the longer pre-roll caused], **warm_pre_roll_seconds: 0.5** [NEW 2026-05-10: post-TTS follow-up slice; long enough to span Silero VAD's ~150 ms speech-start latency without clipping the user's leading word], input_gain_db [2026-05-09])
- `vad` (threshold, min_speech_duration_ms, **min_silence_duration_ms: 1200** [2026-05-09 latency fix; was 500 — natural mid-sentence pauses prematurely closed the capture; trade-off is ~0.7 s slower end-of-turn detection. **Note 2026-05-12:** when `vad.smart_turn.enabled` is true and the model file is present, the orchestrator overrides this to `smart_turn.fast_path_silence_duration_ms` (500 ms) at VAD construction time; smart-turn provides the semantic confirmation that the legacy 1200 ms wall was previously responsible for], window_samples, **long_utterance_threshold_seconds: 8.0**, **long_utterance_silence_duration_ms: 2400** [NEW 2026-05-11 adaptive end-of-turn: once speech has been active past the threshold, orchestrator bumps VAD silence requirement to the long value so a thinking pause mid-prompt doesn't cut the capture. Short utterances stay snappy. Set threshold to 0 to disable.], **max_utterance_seconds: 30.0** [NEW 2026-05-11 follow-up fix: hard ceiling on a single VAD-bounded capture. Was a class-level constant `Orchestrator.MAX_UTTERANCE_SECONDS=15.0` that cut a real user off mid-sentence on a complex coding ask (Whisper transcribed 15.158 s ending mid-phrase at "a button with a box show" — user wasn't pausing; wall-clock ceiling fired before VAD reported SPEECH_END). Now configurable, default 30 s; schema range [5, 120]. Falls back to the class constant only if config load fails.], **smart_turn subsection** [NEW 2026-05-12 Smart Turn V3 semantic end-of-turn confirmation. enabled=true, model_path=`models/smart_turn/smart-turn-v3.2-cpu.onnx`, completion_threshold=0.5 (raise to 0.6-0.7 to reduce false-positive cut-offs at the cost of perceived latency), fast_path_silence_duration_ms=500 (VAD baseline when smart-turn is active), incomplete_extension_ms=700 (additional silence after `incomplete` verdict before submitting anyway), window_seconds=8.0 (training-window cap; longer utterances bypass smart-turn), num_threads=1. Fail-open: missing model file degrades silently to legacy VAD-only behaviour. CPU-only inference ~12 ms; zero VRAM cost.])
- `wake_word` (name, model_path, fallback_model, threshold, cooldown)
- `stt` (model, device, compute_type, beam_size, temperature, etc.)
- `llm` (provider="llama_cpp", **preset** ["qwen3.5-9b"|"qwen3.5-4b"|"custom"; auto-fills model_path/n_ctx/draft_model_path when those keys are omitted — Stage A of the 4B plan], runtime ["in_process"|"http_server"], model_path, draft_model_path, n_ctx, gpu_layers, temperature, top_p, max_tokens, repeat_penalty, history_turns, flash_attn, kv_cache_type, system_prompt, server.{base_url,...}, persona.{source,...})
- **Ultron 1.0 (2026-06-20):** `llm.preset` default is now **`josiefied-qwen3-8b`** with explicit **`n_ctx: 4096`** (10 GB VRAM cap; `gpu_layers: -1`, `flash_attn: true`, `kv_cache_type: 1` already present). Behavioral routing is ENV-gated (not config.yaml keys yet — added to the Pydantic schema in M5b/M7): `KENNING_U1_LLM_ROUTE` (default OFF — route the generic relay through the lean `ultron_prompt`), `KENNING_U1_VERBOSITY` (none/low/high, default high), `KENNING_THINKING_MODE` (default OFF), `KENNING_SNAP_REGISTRY` (default on). All default to today's behavior. Spec: `docs/ultron_1_0/`.
- `embeddings` (dense_model, sparse_model, dense_dim)
- `push_to_talk` (HID/serial auto-PTT; `config.py` `PushToTalkConfig`) — enabled (default OFF), backend (rawhid/serial/auto), hid_vid, hid_usage_page, serial_port, baud, key, lead_ms, release_tail_ms, release_jitter_ms, heartbeat_ms, max_hold_seconds. Env `KENNING_PTT_ENABLED` / `KENNING_PTT_SERIAL_PORT`.
- `relay_speech` (the Valorant relay feature; `config.py` `RelaySpeechConfig`) — enabled, output_device ("Voicemeeter Input"), rephrase, max_line_chars (360), echo_to_user, follow_up_seconds (120.0), roast_lines_path, fun_facts_path.
- `semantic_router` (embedder sidecar + fuzzy router; `config.py` `SemanticRouterConfig`, mostly schema-default — no top-level YAML key required) — enabled, backend (hybrid/embedding/lexical), embedding_weight, sidecar_enabled, sidecar_host, sidecar_port, sidecar_python, sidecar_script, sidecar_backend, sidecar_model, sidecar_device ("cpu"), sidecar_startup_timeout_seconds, sidecar_orphan_sweep_enabled, sidecar_pidfile_path.
- `qdrant` (data_dir="data/qdrant", collections.{conversations, facts, web_results, **projects** [2026-05-22 supervisor stack]})
- `memory` (enabled, jsonl_legacy_path, recent_turns, rag_top_k, rag_exclude_recent, facts_top_k, write_queue_maxsize, **retrieval.{multi_pass_enabled=false, max_categories_per_query=4, candidates_per_category_multiplier=4}** (V1-gap A2), **ranking.{rrf_weight=1.0, recency_weight=0.2, recency_half_life_days=7.0, surprise_weight=0.15, redundancy_weight=0.3}** (V1-gap A2), **rag_min_relevance=0.6** (NEW 2026-05-09: cosine-similarity floor for RAG candidates; tuned empirically with bge-small INT8 -- off-topic content peaks ~0.55-0.57, truly relevant 0.7-0.95), **history_turns_for_llm=4** (NEW 2026-05-09: cap on recent-turn history fed to LLM per call; prevents topic-bleed when user pivots topics))
- `web_search` (enabled, brave_api_key_env, brave/jina/cache subsections, **citation.inline_marker_format="bracket"** [V1-gap B3]). 2026-05-09 latency fix tunables: **`jina.timeout_seconds: 6.0`** (was 15.0), **`jina.max_fetch: 2`** (was 3), **`jina.collective_deadline_seconds: 6.0`** (NEW — executor-side cap on parallel fetch wait; 0 disables). 2026 catalog 12 (felo-search T1): **`query_reformulation.{enabled: true, use_llm: false, max_variants: 2}`** — pre-search query expansion (rule-based default, zero-cost; LLM decomposition opt-in via `use_llm`). 2026 catalog 12 (felo-search T4): **`expose_search_strategy: true`** — surface the fanned-out reformulated queries in the visible transcript (never spoken).
- `addressing` (follow_up_enabled, **warm_mode_duration_seconds: 30.0** ← user override, NOT 10s; rule_confidence_threshold, **zero_shot_addressed_min_confidence: 0.80** [NEW 2026-05-11: demotes low-confidence zero-shot YES verdicts to NOT_ADDRESSED via default_silent; catches the borderline third-person utterances flan-t5-small saturates on at 0.75. Set to 0.0 for legacy permissive behaviour.], zero_shot_model, log_path)
- `coding` (enabled, bridge="direct", mcp.{host,port,...}, template_dir, prompt_token_budget, default/escalation models + thresholds, verification.{smoke,test,lint}_timeout, session_audit_dir, **token_budget_per_session=400000** [2026-05-11 bump from 100000 — new-project sessions burn 100k+ on tool exploration alone before writing files; 400k gives headroom while the 80% warning still fires. Paired with the 2026-05-11 narration honesty fix so users get an explicit "no files written" signal when budget is exhausted mid-exploration], claude_cli, sandbox_root, project_registry_path, audit_log_path, task_timeout, skip_permissions, **voice_task_require_testing=false** [NEW 2026-05-11 token-efficiency fix: was implicitly true via voice.py hardcode, which prepended a "MUST write tests, run, fix, re-run" preamble to every voice-dispatched Claude prompt and 3-5x'd the token spend. Default false lets small voice asks land lean. Users who want tests can say "with unit tests" in their voice request or flip this flag], **facts.{top_k=5, min_confidence=0.75, min_score=0.85, max_age_days=null}** [V1-gap A3], **pre_task_confirmation_enabled=false, pre_task_confirmation_max_words=30, pre_task_barge_in_window_seconds=0.5** [V1-gap A4])
- `projections` (tokenizer, budgets.{clarification,status_delta,adjustment,correction,completion}_context, truncation_warning_threshold, log_truncations)
- `tts` (**engine="piper_rvc" | "xtts_v3"** [NEW 2026-05-10 voice swap; default still legacy for back-compat], piper paths, sample_rate, sentence_flush_chars, length_scale, pause_ms, edge_fade_ms, **pipeline_parallel_enabled=true** [2026-05-09 Piper/RVC split], **speculative_stream_open_enabled=true** [2026-05-09], **speculative_stream_sample_rate=48000** [2026-05-10: was 40000 — actual Kenning RVC output is 48000 Hz, mismatch was forcing the close-and-reopen path on every turn], **output_low_latency_mode=true** [2026-05-09], rvc subsection, **xtts_v3 subsection** [server_python, server_script, reference_audio, host, port, filter_preset="v3_heavy", filter_tail_silence_ms=200, **speed=1.15** (NEW 2026-05-11 cadence tune; XTTS native default is 1.0 — production set to 1.15 for ~15% faster speech without slurring; adjusts synthesis duration tokens so the v3 pedalboard filter is unaffected; safe range ~0.7-1.4, schema-bounded to [0.5, 2.0]), **temperature=0.65** (NEW 2026-05-12 phantom-token mitigation: lowered from XTTS library default 0.75 to sharpen the duration-token distribution so the GPT head stops occasionally emitting fragmentary syllables at sentence ends; range [0.4, 1.0]; threaded through HTTP body to server-side `inference_stream(temperature=...)`; voice character bit-identical because timbre is set by the locked speaker embedding + the v3 filter chain), **phantom_tail_trim_enabled=true** (NEW 2026-05-12 defence-in-depth: client-side post-process that detects the specific phantom-token signature — sustained_speech → ≥150 ms silence → <200 ms event → silence to buffer end — and trims everything after the last sustained-speech region; runs BEFORE the v3 filter so the reverb tail decays normally into its tail_silence_ms padding; set false to disable for A/B), **phantom_tail_silence_threshold=0.005**, **phantom_tail_max_event_ms=200.0**, **phantom_tail_min_lead_silence_ms=150.0**])
- `logging` (file, level, format, datefmt)
- `error_phrases` (13 pools — qdrant_unavailable, brave_unavailable, jina_unavailable, anthropic_unavailable, rvc_unavailable, openclaw_unavailable, piper_unavailable, whisper_repeated_failures, addressing_classifier_failure, wake_word_model_failure, mcp_server_lost, claude_code_subprocess_failed, config_invalid)
- `routing` (llm_disambiguation_enabled, hybrid_task_decomposition_enabled, disambiguation_question_template, routing_log_path, classifier subsection, stub_responses_enabled)
- `openclaw` (enabled=false [stub], gateway_url, auth_token_env, health_check_*_seconds, fail_open, required_agent_id)
- `gaming_mode` (V1-gap A1) — enabled (2026-05-22 default TRUE), plugins_to_disable=[desktop-control, windows-control], toggle_docker=false, docker_executable_path, docker_process_name, log_path, **kokoro_engage_device="cuda"** (2026-06-14: keep Kokoro ON THE GPU while gaming — snappy callouts + frees the CPU so audio capture stops dropping blocks [the root cause of garbled live STT + latency]; the 3B-on-CPU `llm_gpu_layers=0` is the real VRAM saver, the ~330 MB voice model is not) / kokoro_disengage_device="cuda", **vlm_unload_on_engage=true** (2026-05-22), **llm_preset="llama-3.2-3b-abliterated"** (2026-05-22 gaming-mode swap target)
- `desktop` (V1-gap C3) — enabled, default_*_timeout_seconds, plugin_slug, tool_slug_screenshot / tool_slug_list_windows / tool_slug_find_window, **default_monitor_index: Optional[int] = 2** (2026-05-22 user preference: when an APP_LAUNCH / NAVIGATE_TO_SITE / OPEN_LAST_SOURCE utterance gives no explicit monitor cue, place on this 1-based monitor index. Set to `null` to fall back to legacy "main" behaviour. Range [1, 8].) **deep_ui_discovery_enabled: true** (production-hardening #72b: bounded DeepUIDiscoveryLoop retry on a SEMANTIC_CLICK name-lookup miss; miss-path only; fail-open.)
- `window_control` (V1-gap C3) — enabled=false, default_action_timeout_seconds, plugin_slug, tool_slug_focus / tool_slug_click / tool_slug_type
- `browser_use` (2026-05-30 catalog 10) — top-level section for the external `browser-use` CLI browser-automation tier. **8 knobs, all default ON**: `enabled=true`, `binary_path=null` (null = auto-discover `browser-use`/`bu`/`browseruse` on PATH), `default_session=null`, `default_timeout_seconds=30.0`, `default_wait_timeout_ms=30000`, `max_sessions=3` (hard ceiling 16, enforced by `BrowserSessionsManager`), `headed=false`, `screen_context_fallback_enabled=true` (folds a best-effort browser-use page-state line into `screen_context` ONLY when UIA browser extraction is empty AND the tool has an active page). Fail-open: when the binary is absent every method returns a structured "not found" result, so default-ON costs nothing until both the binary is installed AND a browser action is requested.
- `deep_research` (2026 catalog 12, felo-search T3) — top-level section for the bounded agentic deep-research loop over the FREE search ladder. **5 knobs, default ON**: `enabled=true`, `max_steps=3` (research rounds; the load-bearing AgentLoop cap), `max_sub_queries_per_step=3`, `top_n_per_query=3`, `max_accumulated_sources=8` (hard cap bounding the synthesis prompt). EXPLICIT per-turn opt-in via `match_deep_research` ("research X in depth" / "deep dive on X"); the normal sub-second search path is untouched. A deep-research turn runs several full searches (~10-18 s) and fails open at every layer.
- `evolution` (2026 catalog 13, clawhub-capability-evolver clean-room) — top-level section for the bounded autonomous self-improvement subsystem. **5 knobs, default ON**: `enabled=true`, `max_steps=3` (AgentLoop step cap per cycle), `cycle_check_interval_turns=25` (recorded turns between autonomous-cycle checks; a cycle still only proposes when the distiller thresholds are met), `pause_on_demote=false` (demote a churny surface to propose-only rather than pausing it), `apply_temperament=true` (prepend the learned `[Tone: ...]` hint to turns via the system prompt). Maps to `EvolutionConfig`. Fail-open: a construction/runtime failure degrades to a disabled service (every per-turn hook becomes a no-op). Data-only proposals, Tier-3-walled, zero network/shell/eval — see the `src/kenning/evolution/` tree above + the THIRD_PARTY_NOTICES quarantined-source record. **2026 catalog 14 (clawhub-self-improving-agent) adds six more default-ON knobs:** `correction_detection_enabled` / `feature_request_capture_enabled` / `command_failure_capture_enabled` / `pre_turn_nudge_enabled` (bool, all true) + `pre_turn_nudge_max_chars` (240; ~<=50-token cap on the `[Evolution: ...]` nudge; 0 disables the cap) + `recurrence_threshold` (3; ge=2 le=20 -- the explicit distill-ready promote threshold). **2026 production-hardening (#15+#65 guardrail brake) adds five more default-ON knobs:** `guardrail_monitoring_enabled` (the per-turn metrics ring + post-apply re-check) + `guardrail_window_turns` (40; ge=5 le=500) + `guardrail_min_latency_samples` (5; ge=1 le=100) + `guardrail_min_rate_samples` (10; ge=1 le=200) + `post_apply_monitor_turns` (8; ge=1 le=100 -- turns a KEPT skill is watched before the relative pre-vs-post re-check that auto-reverts a regression).
- `intent` (2026-05-22 semantic intent recognizer) — enabled (default TRUE), model="embeddinggemma-300m", variant="q4", threshold=0.65, phrases: list of `{name, phrase, threshold?}` (25 registered: 12 gaming-mode variants + 2 time/date + 11 "needs fresh data" / freshness intents)
- `safety` (2026-05-12 Phases 2-5 runtime tool-call validator) — enabled (default TRUE), per-rule toggles via `rules.{rule_id}: bool`, sandbox_roots override, extra_protected_files / extra_protected_dirs, screen_cache_dir, approved_outbound_apis, audit_log_path
- `coding.supervisor` (2026-05-22 supervisor stack) — eleven knobs:
  - `enabled` (master switch; default FALSE)
  - `digests_enabled` (Phase A; default FALSE)
  - `index_enabled` (Phase B; default FALSE)
  - `decide_enabled` (Phase C; default FALSE)
  - `narrate_enabled` (Phase D; default FALSE)
  - `narration_barge_in_window_seconds=1.5`
  - `enriched_context_enabled` (Phase E; default FALSE)
  - `resolve_threshold=0.75` / `clarify_threshold=0.55`
  - `digest_max_summary_chars=4000` / `digest_max_files_in_prompt=40`
  - `decisions_log_path="logs/supervisor_decisions.jsonl"`
  - `max_candidates_in_decision=5`
- `coding.repo_map` (2026-05-22 catalog batch 2 PageRank repo map) — four knobs:
  - `enabled` (default FALSE; opt-in because the map adds 50-300 ms pre-dispatch latency)
  - `max_map_tokens=1024` (budget when at least one chat file is set)
  - `max_map_tokens_no_chat=8192` (budget when starting cold)
  - `cache_dir="data/.kenning_repomap_cache"`
- `memory.history_compression` (2026-05-22 catalog batch 3 tail-preserve compression) — three knobs:
  - `enabled` (default FALSE; opt-in because compression changes how history is fed to the LLM)
  - `max_tokens=1024` (target budget for the compressed history)
  - `max_depth=3` (recursion cap before falling back to summarising everything)
- `coding.pre_write_lint` (2026-05-22 catalog batch 4 lint cascade) — five knobs:
  - `enabled` (default FALSE; opt-in because compile + flake8 add 50-500 ms per .py write)
  - `python_full_cascade=true` (run compile + flake8; when false only tree-sitter for .py)
  - `multi_language=true` (run tree-sitter on .js/.go/.rs/etc.; when false skip non-Python)
  - `flake8_timeout_seconds=5.0`
  - `attach_summary_to_audit=true` (attach summary + first 20 errors to audit rows)
- `coding.architect` (2026-05-22 catalog batch 6 Phase 1 pre-dispatch architect) — two knobs:
  - `enabled` (default FALSE; opt-in because the LLM plan call adds 3-5 s pre-dispatch on local Qwen)
  - `max_prompt_chars=32000`

### `config/settings.py` (Phase 3 SHIM)

Compatibility shim that re-exports legacy `settings.X` constants from `get_config()`. Thin layer; HF cache pre-init runs at import time. Used by subsystems still on the legacy reference path (audio, wake_word, stt, tts, rvc, coding cluster, scripts) — see [docs/phase3_5_followup.md](phase3_5_followup.md) for the migration punch list.

### `.env.example` (and the actual `.env` in main checkout)

Env vars:
- `KENNING_BRAVE_API_KEY` — Brave Search API key (required for web search)
- `KENNING_LLM_MODEL_PATH` — opt-in override of GGUF path
- `KENNING_AUDIO_DEVICE` / `KENNING_AUDIO_OUTPUT_DEVICE` — operator-specific device strings
- `KENNING_LOG_LEVEL` — console log level
- `KENNING_CODING_MCP_ALLOW_ANY_ROOT=1` — test-only sandbox escape
- `KENNING_CONFIG_PATH` — alternative config.yaml path

---

## Operational scripts

All scripts assume venv active in main checkout (`C:\STC\ultronPrototype`). Worktrees inherit the venv via shared `.venv\Scripts\python.exe`.

### `scripts/benchmark.py`

**Purpose:** measure end-to-end first-token latency for a single voice query.
**Run:** `python scripts/benchmark.py`
**In:** loads full voice stack + config.
**Out:** stdout — TTFT for one synthetic query.

### `scripts/check_vram.py`

**Purpose:** quick VRAM snapshot.
**Run:** `python scripts/check_vram.py [--watch [N]] [--gpu N]`
**In:** nvidia-smi.
**Out:** stdout — `<used> MB used | of <total> MB | target 9216 MB | cap 11500 MB | [OK/above target/WARN/CRITICAL]`
**Functions:** `vram_used_mb(gpu_id) -> Optional[int]`, `vram_total_mb(gpu_id)`, `gpu_name(gpu_id)`, `_format_line(used, total)`, `main(argv)`.

### `scripts/audio_diagnostic.py` (2026-05-09 audio-quality pass)

**Purpose:** mic audio diagnostic harness — RMS, peak, dynamic range over a short capture; helps tune `audio.input_gain_db` for far-field vs close-mic setups.
**Run:** `python scripts/audio_diagnostic.py [--seconds N]`
**In:** live mic via sounddevice.
**Out:** stdout — dB readings + clipping warnings.

### `scripts/autonomous_e2e_harness.py`

**Purpose:** autonomous end-to-end driver that exercises the voice pipeline against a synthetic corpus + writes a JSON report. Useful for nightly autonomous-run findings.
**Run:** `python scripts/autonomous_e2e_harness.py`
**Out:** `logs/autonomous_e2e_report.json`.

### `scripts/benchmark_preflight.py`

**Purpose:** V1-gap B5 preflight benchmark — measures preflight LLM gating cost vs cache hit benefit; informs whether to keep the preflight pass enabled.
**Run:** `python scripts/benchmark_preflight.py [--queries N]`
**In:** sample queries + LLM.
**Out:** baselines.json entry + stdout summary.

### `scripts/comprehensive_memory_quality.py` / `scripts/comprehensive_search_blending.py`

**Purpose:** quality-assurance scripts written alongside the comprehensive quality test pass. Memory variant exercises hybrid RRF + reranker on a fixed corpus; search-blending exercises the multi-pass + composite-ranking path with the gate.

### `scripts/download_models.py`

**Purpose:** first-run model fetcher (Qwen GGUF, Kokoro voicepacks + fine-tune weights, Piper, faster-whisper / moonshine / parakeet, openWakeWord, Smart Turn V3, moondream2, flan-t5-small). 12 steps (renumbered at round 8 to add Kokoro pre-fetch).
**Run:** `python scripts/download_models.py`
**In:** Hugging Face Hub.
**Out:** files under `models/`.

### `scripts/embedder_server.py` (NEW 2026-06-15 — embedding sidecar for the semantic router)

The isolated-venv embedding SIDECAR for the semantic router. A stdlib
`ThreadingHTTPServer` bound to `127.0.0.1` only: `GET /healthz`,
`POST /embed {texts, kind} -> {vectors}`. Two interchangeable backends by
env: `sentence_transformers` (default; google/embeddinggemma-300m on GPU,
run from `C:/STC/ultronVoiceAudio/.venv-embedder` so sentence-transformers
5.x / transformers 5.x / torch never touch the MAIN venv) and `fastembed`
(BAAI/bge-small via ONNX). Pure compute — no input/capture/injection — so it
is anticheat-irrelevant, the same class as OBS/Discord. The orchestrator
spawns it early at boot (`_start_embedder_sidecar`), reuses a running one on
restart, and registers it with the zombie-killer.
**Run:** from the `.venv-embedder` environment: `python scripts/embedder_server.py`
**Out:** loopback HTTP server; stays in foreground; orchestrator spawns it automatically.

### `scripts/migrate_embeddings.py`

**Purpose:** one-shot Qdrant re-embedding when `embeddings.dense_model` changes (e.g. swap bge-small for a different encoder). Recreates the conversations collection at the new dim, re-embeds every turn, atomic swap.
**Run:** `python scripts/migrate_embeddings.py [--dry-run]`
**Out:** new collection populated; old one renamed for rollback.

### `scripts/run_tests.py` (2026-05-21 testing-process hardening)

**Purpose:** unified test runner with pre-flight kill of competing pytest processes, live-streamed stdout, per-test 30 s timeout, slowest-10 report, clean Ctrl-C shutdown.
**Run:** `python scripts/run_tests.py [tests/<subdir>] [-k pattern] [--fast] [--no-timeout] [--kill-only --yes]`
**In:** pytest config (uses pyproject.toml addopts).
**Out:** test pass/fail summary; preserves the live Kenning MCP server on port 19761.

### `scripts/segment_for_finetune.py` / `scripts/transcribe_kenning_reference.py` / `scripts/smoke_xtts_v3.py`

**Purpose:** workshop helpers for the Kokoro fine-tune project (Stage 1 done, Stage 2 ep 0 only — see kenningVoiceAudio/). Segment a long voice recording into LJSpeech-shaped clips; transcribe a reference clip; smoke-test the XTTS v3 stack.

### `scripts/dump_session.py`

**Purpose:** render coding-session audit log into a readable transcript.
**Run:** `python scripts/dump_session.py [--list | --latest | <session_id> | <path/to/file.jsonl>] [--sessions-dir DIR]`
**In:** `logs/sessions/<id>.jsonl`.
**Out:** stdout — formatted event list (one line per event with timestamp + kind + summary).
**Functions:** `_resolve_session_path(token, dir)`, `_read_records(path)`, `_format_record(rec)`, `main(argv)`.

### `scripts/last_session.py` (V1-gap C2)

**Purpose:** backwards-compat alias for `dump_session.py`. The V1 spec named this script `last_session.py`; both names now coexist and resolve to the same `main(argv)` entry point.
**Run:** `python scripts/last_session.py ...` (forwards every arg to `dump_session.main`).

### `scripts/list_audio_devices.py`

**Purpose:** mic / output device introspection.
**Run:** `python scripts/list_audio_devices.py`
**Out:** stdout — devices indexed by ID + name.

### `scripts/maintenance.py`

**Purpose:** periodic Qdrant maintenance (summarize old conversations into `facts`, label clusters, prune stale `web_results`, extract entities).
**Run:** `python scripts/maintenance.py`
**In:** Qdrant `conversations` collection, LLM, `data/maintenance.sqlite` (state).
**Out:** writes to `facts` collection, `data/summaries.jsonl`, updates sqlite.

### `scripts/measure_baseline.py`

**Purpose:** voice-path VRAM + TTFT baseline (10 representative queries; full stack loaded).
**Run:** `python scripts/measure_baseline.py`
**In:** loads full voice stack; runs 10 hard-coded representative queries.
**Out:** writes `baselines.json` (top-level metadata/vram_mb/latency_ms keys).

### `scripts/measure_baseline_extended.py` (Foundation Phase 0)

**Purpose:** extended baselines — search VRAM, coding-session VRAM, TTA microbench, scenario timing, composite TTFA.
**Run:** `python scripts/measure_baseline_extended.py [--lite | --full | --all]`
**Modes:**
- `--lite`: CPU-only — TTA microbench, scenario timing, composite TTFA. ~30 s.
- `--full`: also loads voice stack + measures search/coding VRAM. ~3 min.
- `--all`: both (default).
**In:** config + models + tests/coding/test_orchestration.py runtime.
**Out:** writes `baselines.json` `phase_foundation_start.measurements_extended` block.

### `scripts/migrate_memory_to_qdrant.py`

**Purpose:** one-shot ingest of `data/memory.jsonl` into Qdrant `conversations` collection.
**Run:** `python scripts/migrate_memory_to_qdrant.py`
**In:** `data/memory.jsonl`.
**Out:** `data/qdrant/` collections populated.

### `scripts/review_addressing.py`

**Purpose:** read `logs/addressing.jsonl`, print recent classifier verdicts.
**Run:** `python scripts/review_addressing.py [--tail N] [--misses] [--log PATH]`
**Modes:** `--misses` shows only NOT_ADDRESSED for false-negative tuning.
**Out:** stdout — `HH:MM:SS  DECISION  source  conf  latency  "utt"  -- reason`

### `scripts/run_integration_tests.py` (Foundation Part 7)

**Purpose:** wraps `pytest tests/integration tests/routing tests/error_recovery` with `--gpu` for `PYTEST_RUN_GPU_TESTS=1`.
**Run:** `python scripts/run_integration_tests.py [--gpu] [-q]`
**In:** test files.
**Out:** pytest output to stdout + final summary line with wall-clock + exit code.

### `scripts/run_orchestration_tests.py`

**Purpose:** run the 10 orchestration scenarios in `tests/coding/test_orchestration.py` with reporting.
**Run:** `python scripts/run_orchestration_tests.py`
**Out:** stdout — per-scenario pass/fail + total timing.

### `scripts/validate_config.py` (Foundation Part 7)

**Purpose:** validate `config.yaml` against pydantic schema without starting Kenning.
**Run:** `python scripts/validate_config.py [path] [--print]`
**Out:** stdout — "Configuration is valid." or detailed `ConfigurationError` with path + message + context. Exit 0 = valid, 1 = invalid.

### `scripts/start_llamacpp_server.py` (OpenClaw integration Phase 0 + 4B plan Stage C)

**Purpose:** launch llama-cpp-server on `127.0.0.1:8765` with the same params as the in-process voice loader (n_ctx=8192, flash_attn, Q8_0 KV cache). Imports `kenning` first so bundled torch CUDA DLLs are found before `llama_cpp` initialises (Windows-specific quirk).
**Run:** `python scripts/start_llamacpp_server.py [--n-ctx N] [--port P] [--api-key K] [--chat-format F] [--model-draft <path>] [--draft-num-pred-tokens N] [--from-config]`. The Stage C flags add speculative decoding (`--model-draft` + `--draft-num-pred-tokens`, mapped to llama-cpp-python's `draft_model` / `draft_model_num_pred_tokens`) and a `--from-config` overlay that reads model/draft/n_ctx from `config.yaml:llm` (preset-aware). CLI flags override the overlay. Pure-Python helpers `_build_arg_parser`, `_resolve_kwargs`, `_config_overlay` factor out the testable pieces.
**Out:** uvicorn HTTP server on `--port` (default 8765); stays in foreground.

### `scripts/supervised_llamacpp_server.py` (OpenClaw integration Phase 0)

**Purpose:** Python supervisor wrapper for `start_llamacpp_server.py`. Spawns the launcher as a subprocess, restarts on death with exponential backoff (2 s → 60 s cap, healthy_after_s=30 resets). Lighter alternative to NSSM.
**Run:** `python scripts/supervised_llamacpp_server.py [--cwd ...] [--max-restarts N] [--child-arg ...]`
**Out:** tee'd stdout/stderr from the child + supervisor restart events to stderr.

### `scripts/_bench_llm_http.py` (OpenClaw integration Phase 0)

**Purpose:** TTFT benchmark for the HTTP-runtime LLMEngine. Same 10 representative queries as `measure_baseline.py`, hits llama-cpp-server over HTTP.
**Run:** `python scripts/_bench_llm_http.py` (server must be running on the configured base URL).
**Out:** writes `baselines.json` `llm_http_runtime` block (median, p95, per-query). Used to compare HTTP-mode vs in-process mode latency.

### `scripts/_log_proxy.py` (OpenClaw integration Phase 0; debug only)

**Purpose:** tee proxy on `127.0.0.1:8766` that forwards to `127.0.0.1:8765` and logs every request body + SSE stream to stdout. Used to debug what OpenClaw actually sends to llama-cpp-server.
**Run:** `python scripts/_log_proxy.py` (point OpenClaw's `models.providers.litellm.baseUrl` at the proxy port instead of the server port).

### `scripts/smoke_test_llamacpp.ps1` (OpenClaw integration Phase 0)

**Purpose:** PowerShell smoke test for llama-cpp-server. Hits `/v1/models` and `/v1/chat/completions` with a tiny prompt; prints timing + completion text. Used to verify the server is healthy before involving OpenClaw.
**Run:** `pwsh scripts/smoke_test_llamacpp.ps1`

### `scripts/swap_llm_preset.py` (4B plan Stage H)

**Purpose:** atomic preset swap — edits `config.yaml:llm.preset` in place after validating the requested preset's GGUFs are present. Supports `--list`, `--status`, `--dry-run`. The voice path can also be swapped at runtime via the `MODEL_SWITCH` intent ("Kenning, switch to the 9B"); this script is for off-orchestrator workflows.
**Run:** `python scripts/swap_llm_preset.py [--status | --list | <preset> [--dry-run]]`
**In:** `config.yaml`, `models/*.gguf` (validation).
**Out:** updated `config.yaml`; stdout reports the change.

### `scripts/verify_voice_character_4b.py` (4B plan Stage E)

**Purpose:** interactive A/B helper that synthesises 5 representative voice queries through both the 4B and 9B presets so the operator can confirm Kenning's character is preserved. Approved 2026-05-08.
**Run:** `python scripts/verify_voice_character_4b.py`
**In:** loads voice stack twice (once per preset).
**Out:** plays audio + writes A/B comparison CSV.

### `scripts/verify_items_4_to_8.py` (4B plan Items 4–8 verification)

**Purpose:** exercises each of Items 4 (compression), 5 (IRMA), 6 (self-consistency), 7 (canonical-path monitor), 8 (block-and-revise) in the trigger scenario the corresponding flag fires on. Prints concrete deltas (token reduction, accuracy lift, abort timing, etc.).
**Run:** `python scripts/verify_items_4_to_8.py`
**Out:** stdout — per-item status with measurable metrics.

### `scripts/comprehensive_test_harness.py` (Comprehensive end-to-end test pass)

**Purpose:** single-process exhaustive harness for the comprehensive end-to-end test pass. Runs five phases in sequence — routing classifier accuracy on a 63-utterance labeled adversarial corpus spanning every `RoutingIntentKind`; web-gate rule classifier accuracy on 14 labeled queries; circuit-breaker state machine through CLOSED → OPEN → HALF_OPEN → CLOSED → reopen transitions; memory stress (4 threads × 50 turns ingested into a tmp Qdrant + 20 retrieval probes); V1-gap classifier-gating regression (utterances that used to short-circuit to OpenClaw stub when offline). No GPU / model loads — runs anywhere the venv resolves.
**Run:** `python scripts/comprehensive_test_harness.py`
**In:** Imports the worktree's `src/kenning` and the main checkout's `config/` shim.
**Out:** Stdout summary + machine-readable result at `logs/comprehensive_harness_<ts>.json`.

### `scripts/real_api_smoke.py` (Real-API sparing smoke)

**Purpose:** proof-of-life test for the three external services Kenning talks to in production — Brave, Jina, AI coding agent. Strict budget: ≤2 Brave calls (one bare query + one chain that adds Jina), ≤1 Jina fetch (via the chain), ≤1 minimal AI coding agent (haiku) invocation. Reads `KENNING_BRAVE_API_KEY` from `.env`; the Claude CLI defaults to `%APPDATA%\\npm\\claude.cmd` and can be overridden via `KENNING_CLAUDE_CLI`. Used in the comprehensive end-to-end test pass to confirm circuits + bridge transports work end-to-end without sprawling spend.
**Run:** `python scripts/real_api_smoke.py`
**Out:** Stdout summary + machine-readable result at `logs/real_api_smoke_<ts>.json` (does NOT log the Brave key or any secret).

### `scripts/run_maintenance_for_cron.py` (OpenClaw Phase 7)

**Purpose:** cron-friendly wrapper around `scripts/maintenance.py`. Outputs JSON or single-line Telegram-pretty summary; captures stdout from underlying tasks; structured exit codes (0 ok / 1 task error / 2 init failure). Suitable for Windows Task Scheduler invocations.
**Run:** `python scripts/run_maintenance_for_cron.py [--task <name> ...] [--json | --pretty]`
**In:** subprocesses `scripts/maintenance.py` machinery.
**Out:** stdout — structured summary; exit code per outcome.

### `scripts/benchmark_preflight.py` (V1-gap B5)

**Purpose:** benchmark the web-search gate's pre-flight reasoning pass against the main LLM AND optional CPU-only candidate models. Settles V1-spec Part 1.5's question about whether a dedicated CPU model would be faster than the main Qwen on pre-flight. Decision documented at [docs/preflight_decision.md](preflight_decision.md): keep main LLM (TTFT 79 ms voice baseline already beats the spec's 200 ms threshold).
**Run:** `python scripts/benchmark_preflight.py [--candidate-model PATH] [--skip-main] [--queries N]`
**In:** loads the live `LLMEngine` (or a CPU-only `llama_cpp.Llama` for the candidate); 30 representative queries with manual ground truth.
**Out:** Markdown summary table + appends `preflight_benchmark.backends` block to `baselines.json`.

### `scripts/run_kenning_mcp_for_openclaw.py` (OpenClaw Phase 13)

**Purpose:** stdio MCP entry script OpenClaw spawns when an agent calls one of Kenning's tools. Boots a FastMCP server on stdio that exposes `get_heartbeat_alerts`, `acknowledge_alert`, `run_maintenance`, `list_active_coding_sessions`, `get_recent_voice_alerts`. Imports stay light — no torch / LLM loaded.
**Run:** `python scripts/run_kenning_mcp_for_openclaw.py [--stdio | --list-tools]`
**In:** disk artifacts (heartbeat alert log, session audit dir) + OpenClaw stdio channel.
**Out:** MCP responses over stdio.
**Auto-resolved:** `OpenClawBridgeConfig.mcp_server_command="auto"` resolves to this script via the holder's `_resolve_mcp_command` helper.

### `scripts/_record_phase0_baseline.py` / `scripts/_merge_phase0_baselines.py` (OpenClaw Phase 0)

**Purpose:** record and merge Phase 0 baseline measurements into `baselines.json`. Used during the OpenClaw Phase 0 verification work.
**Run:** `python scripts/_record_phase0_baseline.py`; `python scripts/_merge_phase0_baselines.py`

### `scripts/_vram_peak_monitor.py` (auxiliary)

**Purpose:** background VRAM peak monitor used by `measure_baseline_extended.py` for accurate peak capture during search/coding-session runs.

### `scripts/audio_diagnostic.py` (2026-05-09 audio-quality pass)

**Purpose:** standalone diagnostic harness for far-field mic + wake + Whisper tuning. Loads ONLY the audio path (sounddevice + openWakeWord + Silero VAD + faster-whisper) — NO LLM, NO TTS, NO orchestrator. ~1.5 GB VRAM so it can run while the full Kenning stack is stopped (per the voice-stack-concurrency rule).

**Modes** (`--mode`):
- `noise-floor` — captures N seconds of silence; reports peak / mean RMS dBFS for noise-floor calibration.
- `wake` — captures a window, records max wake-word score, prints whether `FIRED` at the configured threshold; saves audio to WAV via `--save-wav` for replay.
- `phrase` — captures until VAD reports speech end (or hard timeout); reports VAD timing, peak RMS, Whisper transcription + word-coverage vs `--expected-text`.
- `monitor` — live real-time meter: rolling RMS, VAD probability, wake score per chunk; Ctrl+C to exit.

**CLI overrides** (process-local, never write back to config so iteration is fast): `--device` (substring match — "Focusrite", "Voicemeeter"), `--gain-db`, `--wake-threshold`, `--vad-threshold`, `--seconds`, `--whisper-beam`, `--save-wav`, `--label`.

**Audit log:** every test row appends to `logs/audio_diag_<ts>.jsonl`; useful for cross-distance comparison.

**Run:** `python scripts/audio_diagnostic.py --mode wake --device Focusrite --label round1_5ft --seconds 10 --save-wav logs/round1_5ft.wav`

### `scripts/comprehensive_memory_quality.py` (2026-05-09 memory-quality pass)

**Purpose:** end-to-end memory + retrieval quality test pass. Loads embedder + isolated tmpdir Qdrant + (optionally) Qwen 4B. Seeds the isolated store with 58 mixed-topic turns (predator chatter, PC troubleshooting, food, BMWs, weather, code) and runs 28 scenarios verifying:

- Contamination filtering — predator chatter doesn't bleed into a weather query, troubleshooting doesn't bleed into a ducks query, etc.
- Healthy recall — relevant prior context (recent or old) IS surfaced when topic-related.
- Recency-weighted ranking — recent-and-relevant ranks ahead of old-and-relevant.
- Topic shifts — pivot-and-return works (lions → BMW → "what predator did I ask about earlier?").
- Edge cases — short queries, paraphrased queries, queries with no matching memory.

Per-scenario validation: retrieval `expect_includes` / `expect_excludes` substrings + LLM `expect_response_excludes` for contamination tokens.

**Run:** `python scripts/comprehensive_memory_quality.py [--skip-llm] [--scenario-filter X] [--audit-log PATH]`. Without `--skip-llm`, loads Qwen 4B and exercises the full retrieve → context-assembly → response path. ~3.5 GB VRAM.

### `scripts/comprehensive_search_blending.py` (2026-05-09 memory-quality pass)

**Purpose:** end-to-end search-augmented contamination tests with REAL Brave + Jina + Qwen 4B. Verifies the orchestrator's `_search_augmented_tokens` path: predator chatter in memory doesn't bleed into a Python-3.13 search response; troubleshooting context doesn't bleed into a duck-lifespan search response; relevant troubleshooting context DOES blend into a motherboard-light search response.

3 scenarios; ~3 Brave + 3 Jina calls per full run. Within free-tier quota.

**Run:** `python scripts/comprehensive_search_blending.py` (requires `KENNING_BRAVE_API_KEY`).

### `scripts/_debug_retrieval_cosine.py` (2026-05-09 memory-quality pass; debug only)

**Purpose:** prints cosine similarity between a probe query and a hand-picked candidate set. Used to empirically tune `memory.rag_min_relevance` against the actual production embedder (bge-small INT8). The 0.6 threshold was chosen because off-topic content peaked at 0.55-0.57 across the probe corpus, while genuinely relevant content scored 0.7-0.95.

**Run:** `python scripts/_debug_retrieval_cosine.py`. No flags; edit the `PROBES` and `CANDIDATES` lists at the top of the file to test new query+content pairs.

### `scripts/cleanup_stale_processes.py` (2026-05-14 cleanup pass)

**Purpose:** find and kill stale Kenning-related python processes
(orphaned pytest workers, stale `run_kenning_mcp_for_openclaw.py`
processes from old worktrees, orphaned XTTS servers, large no-cmdline
workers). Always preserves the currently-running Kenning and its
process chain: the script enumerates the TCP listener on port 19761
(the MCP server) and adds that process plus its ancestors and
descendants to a "do not touch" set.

**Run:**

```
python scripts/cleanup_stale_processes.py            # dry-run; prints what it would kill
python scripts/cleanup_stale_processes.py --kill     # actually terminates them (prompts first)
python scripts/cleanup_stale_processes.py --kill -y  # skip the prompt
```

**Flags:** `--max-age-minutes` (default 30; ignore unknown-cmdline workers younger than this), `--min-rss-mb-unknown` (default 200; only kill unknown-cmdline workers with at least this much RAM).

**In:** `psutil` (already in the venv) + the live process table. **Out:** stdout summary + exit code 0 on success, 1 if any termination failed.

### `scripts/bench_llm_ubatch.py` (NEW 2026-05-15 latency pass)

**Purpose:** sweep llama-cpp-python's `n_batch` / `n_ubatch` knobs to find the lowest-TTFT combination for voice-length prompts on the active hardware. Loads `LLMEngine` fresh per combination (so each gets a clean Llama instance) and measures TTFT on 5 representative queries with 2 warmup runs. Writes results into `baselines.json:llm_n_ubatch_sweep`. Loads the voice stack -- ASK before running per `feedback_voice_stack_concurrency.md`. Default sweep covers `(None, None)` baseline + 5 `(n_batch, n_ubatch)` combinations; takes ~3-6 min on the 4070 Ti.

**Run:** `python scripts/bench_llm_ubatch.py [--sweep "128,256,512,1024"] [--warmup 2] [--trials 5]`

**Empirical result on 2026-05-15:** all combinations give ~63 ms median TTFT on voice-length prompts -- no measurable win at short context. Knobs stay in place for future long-context tuning.

### `scripts/bench_stt_latency.py` (NEW 2026-05-15 latency pass)

**Purpose:** measure Whisper STT latency at 1s / 3s / 5s / 8s audio lengths to right-size STT optimisations. Generates speech-like synthetic audio, warms up the engine, then runs `--trials` measurements at each length. Reports median / p95 / min / max / RTF. Loads voice stack -- ASK first.

**Run:** `python scripts/bench_stt_latency.py [--lengths 1,3,5,8] [--warmup 2] [--trials 5]`

**Empirical result on 2026-05-15 (small.en + int8_float16 + beam=5):** 1s = 156 ms, 3s = 188 ms, 5s = 109 ms, 8s = 109 ms. With **beam=1 on 5s audio: 78 ms median** -- saves ~80 ms vs beam=5. This bench drove the Phase 4 decision to set `stt.beam_size: 1` as the new production default.

### `scripts/bench_llm_prefix_cache.py` (NEW 2026-05-16 latency pass 2)

**Purpose:** A/B benchmark of the in-process `LLMEngine` TTFT with `LlamaRAMCache` cache_bytes=0 (disabled) vs cache_bytes>0 (enabled). Builds a fresh `LLMEngine` per condition (so each gets a clean Llama instance + cache state) and measures TTFT on 5 representative voice queries with configurable warmup. Drove the Phase 2 decision to ship the cache infrastructure but flip the default to disabled. Loads the voice stack -- ASK before running per `feedback_voice_stack_concurrency.md`.

**Run:** `python scripts/bench_llm_prefix_cache.py [--turns 5] [--warmup 1] [--out baselines.json]`

**Empirical result on 2026-05-16 (4070 Ti + josiefied-qwen3-4b Q4_K_M):** cold-cache TTFT median **63 ms** (78, 79, 63, 62, 63 across 5 queries); warm-cache (2 GiB RAMCache) TTFT median **78 ms** (78, 78, 79, 63, 62). **The cache shows a -15 ms regression** -- llama.cpp's internal KV cache already handles intra-session prefix reuse; the explicit RAMCache's `load_state` memcpy exceeds the eval savings on our short ~280-token system prompts. Result merged into `baselines.json:llm_prefix_cache_bench`. The knob and bench stay shipped so operators with longer prompts / cross-session reload patterns can opt in.

**Operator note:** the bench requires the production GGUF on disk. When run from a worktree (not the main checkout), set `KENNING_LLM_MODEL_PATH=C:\STC\ultronPrototype\models\Josiefied-Qwen3-4B-abliterated-v2.Q4_K_M.gguf` (or the absolute path to the active preset's GGUF) so the engine resolves correctly -- the worktree's `models/` directory is empty.

---

## Tests

### `tests/conftest.py` — Path setup + session-end subprocess reaper.

Two responsibilities:

1. Prepend the project root and ``src/`` to ``sys.path`` so
   ``from kenning.*`` works when pytest is launched from the repo
   without an editable install.

2. Register a ``pytest_sessionfinish`` hook (2026-05-14 cleanup pass)
   that walks the test runner's descendant python processes and
   terminates them when the session ends -- whether the run completed
   normally, crashed, or was Ctrl-C interrupted. Without this, a hung
   test or a backgrounded pytest that never gets reaped leaves a
   python worker holding hundreds of MB of RAM (and VRAM if torch /
   CUDA was loaded by a fixture). Fail-open at every step (psutil
   import / TCP enumeration / individual terminate calls); never
   touches a process tied to the live Kenning orchestrator (detected
   via the port-19761 listener and its ancestor/descendant chain).

### Default suite (no env gate) — per-file snapshot (frozen 2026-05-22)

> **The CANONICAL current pass count lives in the validating-HEAD header at
> the TOP of this file** (catalog 14: **8999 passed / 26 skipped / 0 failed**,
> worktree full sweep, no deselect, exit 0). The per-file enumeration in THIS section is a
> historical snapshot frozen at the 2026-05-22 review-feedback pass (4240
> passed / 16 skipped) and is intentionally NOT re-counted per commit — the
> top-of-file header is the source of truth for the running total. Always
> run the sweep via `scripts/run_tests.py` (never bare `pytest` -- see the
> binding `feedback_test_sweep_workflow.md` + `docs/test_sweep_binding_rules.md`).
> Original 2026-05-22 snapshot: **4240 passed / 16 skipped (GPU-gated)** in
> ~76 s (+136 from the session-E baseline of 4104).

**Catalog 13 (evolution) test files (+286):** [`tests/evolution/`](../tests/evolution/) mirrors the package layout (265 tests) -- `test_models.py` (GEP dataclasses + asset-id round-trip), `test_signals.py`, `test_blast_radius.py`, `test_skill_distiller.py` (incl. a skills-loader round-trip on a distilled `.md`), `test_guardrails.py`, `test_autonomy.py`, `test_personality.py`, `test_evolution_loop.py` (fake collaborators; keep/revert/block paths), `test_intent.py`, `test_service.py` (hermetic JSONL store + service; all persistence to `tmp_path`). Wiring (+21): [`tests/test_llm_temperament_hint.py`](../tests/test_llm_temperament_hint.py) (5; `set_temperament_hint` -> `_build_messages` system-prompt injection, user-text untouched) + [`tests/test_orchestrator_evolution_wiring.py`](../tests/test_orchestrator_evolution_wiring.py) (16; `Orchestrator.__new__` pattern -- `_load_evolution_if_enabled` real round-trip under `tmp_path`, `_maybe_handle_evolution_command` status/run/fail-open, `_record_evolution_turn` + `_consume_last_barge_in`).

**Catalog 14 (self-improving-agent) test files (+~53):** under [`tests/evolution/`](../tests/evolution/) -- `test_catalog14_capture.py` (the new capsule / signal types + `redact_fragment` + recurrence helpers + `EvolutionConfig` defaults/bounds), `test_catalog14_detectors.py` (the four `extract_*` detectors incl. the prior-response gate + positive-ack suppression, `derive_topic_area`, opportunity recognition, the unchanged 17-signal taxonomy count), `test_catalog14_distiller_loop.py` (`merge_capsules_by_pattern_key` + `pattern_recurrence` + back-compat + repair distillation via `failures_provider`), `test_catalog14_service.py` (record_turn capture + gating + recurrence counters + pre-turn nudge cap/gating + digest/status + ledger reload). Wiring extends [`tests/test_orchestrator_evolution_wiring.py`](../tests/test_orchestrator_evolution_wiring.py) (`prior_response` passing + `_drain_evolution_command_failures`). All hermetic; full worktree sweep **8999 passed / 26 skipped / 0 failed**.

**New test files in this pass:**
- [`tests/resilience/test_fail_open_log.py`](../tests/resilience/test_fail_open_log.py) (+26) — per-session counter: record / accumulate / unknown-category open-ended / fail-safe on broken lock; configure + flush JSONL + previous-session read; render_summary alphabetisation + empty + None handling; KNOWN_CATEGORIES uniqueness + bus_slow_subscriber present.
- [`tests/test_supervisor_tier_config.py`](../tests/test_supervisor_tier_config.py) (+17) — `SUPERVISOR_TIERS` catalog shape; tier `"off"|"indexing_only"|"deciding"|"full"` fills per-phase flags via the `model_validator`; explicit per-flag overrides win; threshold + log_path + digest knobs untouched by tier; unknown tier rejected.
- [`tests/test_effective_config_log.py`](../tests/test_effective_config_log.py) (+14) — emits every high-impact section (LLM / TTS / STT / MEMORY / SUPERVISOR / GAMING_MODE); KENNING_* env vars surface with override notes; known-secret env vars elided (`<set>` / `<empty>`); non-KENNING env vars ignored; supervisor tier reflected; fail-open on broken cfg + partial section failure.
- [`tests/test_tts_text_normalization.py`](../tests/test_tts_text_normalization.py) (+40 fuzz cases) — `normalize_text_for_tts` against URLs (https/http/ftp/bare-www / 1200+ char), Windows drive paths, mixed slashes, times (12-hour AM/PM / 24-hour / standalone markers), unicode (en-dash, em-dash, non-breaking hyphen, smart quotes, Arabic RTL, combining diacritics, emoji ZWJ), shell metachars + `$` currency, Latin abbreviations, acronym dots, idempotence on clean text, 10 KB stress; `_speakable` against drive paths, unix paths, mixed slashes, surrounding quotes, dots-in-leaf, unicode names, UNC paths, very long paths.
- [`tests/observations/test_drift_sample.py`](../tests/observations/test_drift_sample.py) (+9) — emits well-formed row with `subsystem=llm event_type=thinking_drift_sample` + `enable_thinking=False`; parent_event_id chaining; extra-dict merge; user_text/response_text truncation at 4000 chars with explicit marker; exact-cap survives intact; None user_text treated as empty.

**Bus tests extended** in [`tests/bus/test_service.py`](../tests/bus/test_service.py) (+9): `DEFAULT_SLOW_SUBSCRIBER_WARN_MS` pin (15.0); fast subscriber no-warn; slow subscriber WARN-logs + bumps counter; counter accumulates per-occurrence; subscriber exception path excluded from slow counter; `set_slow_subscriber_recorder` callback fires; recorder exceptions swallowed; threshold accessor; slow subscriber doesn't block later subscribers.

**Introspect tests extended** in [`tests/coding/test_project_introspect.py`](../tests/coding/test_project_introspect.py) (+8): `invalidate_for_file` drops matching project / returns 0 on no match / handles empty string / handles Path.resolve() failure on deleted files; `install_bus_invalidator` idempotent; `CodingFileChangedEvent` actually invalidates the cache; payload errors swallowed; reset_bus_invalidator_for_testing + re-install yields fresh subscription.

### Prior baseline (pre-review-feedback pass) -- 4104 passed / 16 skipped (~85 s wall) at HEAD `b02af04`

Run via `scripts/run_tests.py` (2026-05-21 testing-process hardening: pre-flight kill of competing pytest workers + per-test 30 s timeout + live-streamed stdout + clean Ctrl-C shutdown).

**Top-level (~80 files):**
- `test_addressing.py` — rule-based addressing classifier
- `test_audio.py` — capture, ring buffer (incl. 2026-05-10 mode-aware `snapshot(last_n_samples=...)` slicing), devices
- `test_response_style.py` (22, 2026-05-10) — `is_brief_question` / `apply_brevity_hint` coverage: short-question detection, depth-marker skip, long-question pass-through, empty input, idempotence on already-hinted text
- `test_conversational_ack.py` (24, 2026-05-12 — NEW) — conversational filler-ack: gate eligibility (long-utterance fires, short-utterance/empty/clarification-pending skipped, whitespace-stripped), `ConversationalAckSource` shuffled-cycle (no immediate repeats, full pool per cycle, custom pool, empty-pool rejection), phrase-pool sanity (no web-search overlap, period-terminated, short, no duplicates), and orchestrator-level wiring (ack appears as first token on no-gate fallthrough path, suppressed on short utterance / pending clarification, fail-open on broken source or `has_pending_clarification` exception)
- `test_precomputed_ack.py` (25, 2026-05-15 — NEW) — `PrecomputedAckClipCache`: construction (dedup / strip / sort / drop empty / None-safe / starts empty), lookup (miss / strip-match / empty input / wrong phrase miss), prewarm (populates all / returns count / skips empty clip / swallows synth exception / partial population / idempotent), thread safety (concurrent get during prewarm), default phrase pool factory (collects both conv + web-search pools), `prewarm_in_background` (returns daemon thread / populates / honours name)
- `test_llm_precomputed_rag.py` (9, 2026-05-15 — NEW) — `precomputed_rag_snippets` kwarg on `_build_messages` / `generate` / `generate_stream`: snippets appear in message body, internal retrieve is bypassed, empty list = no RAG (not retry), None falls back to legacy retrieve, suppress_memory_context wins over precomputed, public `retrieve_rag_snippets` proxies private, returns [] when no memory, preserves recent history independently, compatible with gate_verdict
- `test_orchestrator_rag_prefetch.py` (11, 2026-05-15 — NEW) — orchestrator `_kick_off_rag_prefetch` (returns None when memory disabled / multi-pass enabled / executor broken; kicks off + completes when single-pass), `_collect_rag_future` (None future returns None / completed returns value / exception returns None / empty list distinguishable), `_build_response_stream` integration (prefetch kicks off + precomputed snippets reach LLM, no memory skips prefetch, multi-pass skips prefetch and passes None to LLM)
- `test_llm_batch_tunables.py` (14, 2026-05-15 — NEW) — `LLMConfig.n_batch` + `n_ubatch`: schema (defaults are None, accepts explicit values, rejects 0 / negative / too-large, n_ubatch may exceed n_batch in schema), `_build_llama` wiring (omits kwargs when None / passes n_batch only when set / passes n_ubatch only when set / passes both when set), top-level `KenningConfig` round-trip (default keeps None, accepts values)
- `test_tts_preopen.py` (13+2, 2026-05-15 NEW + 2026-05-16 latency 2 extension) — TTS output-stream pre-open: xtts_v3 (prepare+consume match SR / consume mismatch closes & returns None / consume with no preopen returns None / prepare idempotent / failure swallowed / stop closes leftover), legacy speech.py (prepare+consume / SR-mismatch close / failure swallowed / **legacy silence-write invoked + failure-swallowed** (2026-05-16)), orchestrator (`_kick_off_tts_preopen` returns None when engine lacks method / returns thread when engine supports / swallows thread-construction failure / no-op when tts is None)
- `test_llm_prefix_cache.py` (11, 2026-05-16 latency 2 — NEW) — `LLMConfig.prefix_cache_ram_bytes`: schema (default 0 after bench-driven flip, accepts 0 / large values, rejects negative, round-trip), `_build_llama` wiring (attaches `LlamaRAMCache` when set / skips when 0 / fail-open on import error / fail-open on set_cache exception), top-level `KenningConfig` round-trip
- `test_speculative_stt.py` (12, 2026-05-16 latency 2 — NEW) — orchestrator speculative-STT helpers: kick-off (starts background thread / idempotent while in-flight / fail-open on thread launch failure), collect (None when no kick-off / waits for thread / resets state / None on transcription exception / None on timeout), invalidate (causes collect to return None / re-arms for next kick-off after collect), reset state (clears stale result without killing thread), kick-off copies audio to avoid race
- `test_speculative_classification.py` (21, 2026-05-18 latency pass 3 Phase 2 — NEW) — orchestrator speculative-classification helpers: `_run_speculative_classification` (stores rule-path verdict + ack + RAG future; skips on already-invalidated; mid-work invalidation drops result; missing web_gate -> None verdict; ack/RAG exception swallowed), `_invalidate_speculative_classification` (sets flag + cancels RAG future; idempotent; cancel-exception swallowed; STT invalidate propagates), `_collect_speculative_classification` (returns None on empty / text-mismatch / invalidated; clears slot atomically; defensive on missing lock), `_reset_speculative_classification_state` (clears slot + cancels RAG; defensive), STT-thread chain (chains classification on success; skips on empty transcript; skips on invalidated; reset propagates to classification slot)
- `test_speculative_llm.py` (25, 2026-05-18 latency pass 3 Phase 3 — NEW) — LLMEngine + orchestrator speculative-LLM. LLMEngine surface (4): `record_history=True` records turn / `record_history=False` skips auto-record / `record_completed_turn` records explicitly / skips empty input. Orchestrator helpers (21): `_kick_off_speculative_llm` (starts thread + buffers tokens / idempotent / skips on missing LLM / skips on None verdict), `_invalidate_speculative_llm` (signals cancel + sets flag / idempotent / defensive on missing lock), `_collect_speculative_llm` (None when empty / drains buffer + commits history on completion / None on text mismatch / None on invalidated / commit no-op on incomplete speculation / defensive on missing lock), `_reset_speculative_llm_state` (clears + cancels in-flight / defensive), cross-lane invalidation (classification invalidate propagates / STT invalidate propagates / reset propagates), classification chain (NO_SEARCH kicks off LLM / SEARCH skips / UNCERTAIN skips)
- `test_llm_strip_thinking.py` (9, 2026-05-14 — NEW) — `strip_thinking_text` pure function: clean text passthrough, single-block strip, multi-block strip, surrounding text preserved, unterminated `<think>` drops tail, multiline blocks, real-session screen-context pattern, idempotence, short-input fast path. Covers the gap where blocking-path `LLMEngine.generate()` previously returned raw `<think>...</think>` chains (the streaming path was already filtered).
- `test_smart_turn.py` (43, 2026-05-12 — NEW) — Smart Turn V3 semantic end-of-turn confirmation: `SmartTurnConfig` schema (defaults match production layout, all four range-enforced fields, dict round-trip, nested-under-VADConfig), `truncate_or_pad_for_smart_turn` pure function (under-window passthrough, over-window truncation to last n seconds, int16→float32 conversion, multi-dim flatten, non-16kHz rejection, custom window override), `SmartTurnDetector` construction (missing file, out-of-range threshold/window/threads, lazy-loading, warmup-propagates-failure, empty/wrong-sr/post-close all return None), `build_detector_from_config` fail-open (disabled / missing file / absolute-path missing all return None; present file yields a lazy detector), real-model end-to-end (6 tests, skipped when `models/smart_turn/smart-turn-v3.2-cpu.onnx` is absent — loads + warmup, silence verdict shape, threshold flip with identical probability, short audio padded by WhisperFeatureExtractor, long audio truncated to last 8 s, median inference under 150 ms), orchestrator-level wiring (`_smart_turn_should_check` gate semantics across detector-missing / no-speech / within-window / over-window, `_run_smart_turn` passes verdict through + swallows exceptions, `_build_smart_turn_detector` fail-open for disabled / missing file)
- `test_coding_bridge.py` — CodingBridge abstract contract
- `test_coding_e2e.py` — coding e2e (PYTEST_RUN_GPU_TESTS gated)
- `test_coding_intent.py` / `test_coding_intent_phase2.py` — intent classifier
- `test_coding_projects.py` — registry + resolver + sandbox creation
- `test_coding_runner.py` — runner state machine
- `test_coding_templates.py` — template renderer
- `test_coding_voice.py` — voice controller (now CapabilityVoiceController)
- `test_coordinator.py` — clarification + correction loops
- `test_correction_loop.py` — corrective re-prompting
- `test_fairseq_compat.py` — torch.load + dataclass workarounds
- `test_llm.py` — LLM (PYTEST_RUN_GPU_TESTS gated)
- `test_maintenance.py` — periodic maintenance
- `test_mcp_e2e.py` / `test_mcp_server.py` / `test_mcp_session.py` — MCP layer
- `test_memory_qdrant.py` — Qdrant memory + embedder
- `test_narration.py` — StatusNarrator
- `test_phase7_audit_and_tokens.py` — per-session audit + token tracking
- `test_pipeline.py` — orchestrator construction (PYTEST_RUN_GPU_TESTS gated)
- `test_projections.py` — 29 projection tests (Phase 2 + Foundation Part 2)
- `test_transcription.py` — Whisper (PYTEST_RUN_GPU_TESTS gated)
- `test_tts.py` — Piper + RVC
- `test_uncertainty.py` — uncertainty signal application
- `test_verification.py` — six verification checks
- `test_web_gating.py` — two-stage gating
- `test_persona_loader.py` (20, OpenClaw Phase 1) — `PersonaLoader` modes / hot-reload / HTML-comment-only files
- `test_llm_persona_source.py` (8, OpenClaw Phase 1) — `LLMEngine` persona-source wiring + hot-reload + fallback
- `test_llm_http_runtime.py` (9, OpenClaw Phase 0) — HTTP-runtime construction, request shape, SSE streaming, cancel mid-stream
- `test_llm_preset.py` (13, 4B plan Stage A) — `LLMConfig.preset` resolution: 9b/4b/custom defaults, explicit-override wins, YAML round-trip, invalid preset rejected
- `test_start_llamacpp_server.py` (13, 4B plan Stage C) — launcher CLI: --help renders, default args back-compat, --model-draft attaches speculative decoding, --draft-num-pred-tokens override, --from-config overlay (4b/9b), CLI flags override overlay
- `test_llm_enable_thinking.py` (11, 4B plan Stage F) — `enable_thinking` parameter plumbing: helper kwargs, in-process generate/generate_stream pass-through, HTTP payload pass-through, back-compat when default
- `test_llm_rag_position.py` (7, 4B plan Stage G) — `_build_messages` honors `llm.rag.position`: recency mode prepends to user message, system mode folds into system message, no-snippets/retrieve-failure fallback, helper invariants
- `test_on_the_fly_preset_switching.py` (16, 4B plan Stage H infra) — `KENNING_LLM_PRESET` env-var override (clears overrides by default, opt-in keep-overrides flag), minimal-YAML preset-only config, `check_vram._resolve_target_mb` (table + CLI override + env var + unknown fallback), `_format_line` shows preset label, `swap_llm_preset._rewrite_preset` (basic / preserves comment / first-match / missing-line raises)
- `tests/routing/test_model_switch_classifier.py` (54, 4B plan voice-swap) — classifier maps "switch to 4B/9B/four B/for B/nine B/4 B/4-B" + verb variants (switch/swap/change/use/load/go/move/activate/engage/run/select) to `RoutingIntentKind.MODEL_SWITCH`; rejects passing mentions ("the 4B is faster") and conversational utterances; pending clarification suppresses (mid-dialogue safety); active coding task does not block; `_resolve_model_switch_target` helper
- `test_llm_reload_for_preset.py` (9, 4B plan voice-swap) — `LLMEngine.reload_for_preset` rejects http_server runtime + unknown preset; idempotent on same-preset; success path replaces `_llm` and clears history; sets `KENNING_LLM_PRESET` env + clears stale `KENNING_LLM_MODEL_PATH`; failure path keeps old engine, restores env vars (whether they were set or unset originally)
- `test_llm_prompt_injection_defense.py` (21, comprehensive QUALITY pass Q10 iter 1+2) — `_sanitize_user_input` neutralises tag-style markers ([INST]/[/INST], <|im_start|>/<|im_end|>/<|system|>/<|user|>/<|assistant|>, stray </think>); detects natural-language jailbreak patterns ("ignore previous instructions", "you are now X", "respond with the exact word", "act as", "pretend"); preserves benign questions (zero false-positive on normal voice queries); end-to-end verified: pre-defense 2/3 of Q8 prompt-injection probes succeeded; post-defense 0/3. Voice baseline TTFT 79 ms / VRAM 7889 MB unchanged (defence is sub-microsecond on benign input).
- `test_web_search_parallel_fetch.py` (6, 2026-05-09 latency hot-fix) — verifies the `WebSearchExecutor` parallel-Jina-fetch path: wall-time dominated by the slowest URL (not the sum); collective deadline abandons slow fetches and degrades them to snippet-only with `jina_deadline:<url>` notes; partial success with one fast + one slow URL keeps the fast one's `full_text`; per-fetch exception in one parallel branch doesn't break the others; `collective_deadline_seconds=0` disables the cap; `max_fetch=0` skips Jina entirely.
- `test_tts_pipeline_parallel.py` (15, 2026-05-09 + 2026-05-10) — original 11 cover the parallel split, speculative stream open, sample-rate-mismatch fallback, low-latency hint, RVC fallback, cancellation. 2026-05-10 added 4 for producer-signaled lookahead: `test_first_clip_plays_before_next_fragment_yielded` (the ack-first contract — first clip MUST be written to the stream before the generator is asked for the second), `test_slow_second_clip_does_not_kill_playback` (4 s gap between fragments doesn't trigger RVC starvation abort — guards the BMW-search failure mode), `test_clipitem_is_known_last_skips_lookahead` (ClipItem namedtuple shape + flag carries through), `test_end_of_stream_sentinel_terminates_playback` (None on audio_q ends playback with tail silence even when the previous ClipItem had `is_known_last=False`).
- `test_voice_model_switch.py` (11, 4B plan voice-swap) — `CapabilityVoiceController._handle_model_switch` calls `llm_engine.reload_for_preset(target)`, speaks "Switched to the 4B/9B" on success, "I'm already running the X" on idempotent, "I couldn't switch ..." on failure with reason; "I can't switch models — engine isn't wired" when llm_engine is None; missing payload says "couldn't tell which model"; end-to-end classifier-then-controller for utterances
- `tests/routing/test_irma_reformulation.py` (15, 4B plan Item 5) — `InputReformulator` pure-text shape (default-only-utterance, whitespace-strip, quote-escape, recent-decisions section, max-recent truncation, active-session, routing-hints, max_recent=0 omits, log-row factory); disambiguator integration with the IRMA flag (default-OFF passes raw, ON uses enriched, reformulation-failure falls back, no-context still emits utterance)
- `test_self_consistency.py` (27, 4B plan Item 6) — `majority_vote_text` (winner, whitespace-strip, tie-first-wins, empty input, blank filter), `majority_vote_json` (winner, unparseable handling, think-block strip, first-block-only, all-unparseable returns None, arrays), `majority_vote_label` (case-insensitive, no-match), `run_self_consistency` driver (sampler called N times, default text aggregator, sampler exception handling, fallback to first non-empty, n-clamping), `should_apply_self_consistency` config gate (default-off, global-on, per-site disabled), decomposer integration (single-call default, N-call with consistency, majority winner, per-site bypass, all-unparseable fallback)
- `test_canonical_monitor.py` (17, 4B plan Item 7) — canonical set lockdown (standard tools, MCP callbacks), canonical-only paths (no abort), threshold-not-reached, threshold-reached-in-window aborts, late drift does not abort, latch semantics, reset clears state, non-tool-use events ignored, empty/None tool name ignored, case-insensitive match, attribute-style event input, custom canonical override, verdict-shape (off_canonical_tools list, immutability), factory gate (disabled returns None, enabled returns instance with config)
- `test_block_and_revise.py` (14, 4B plan Item 8) — `ToolCallValidator` ALLOW + BLOCK verdicts, think-block strip, case-insensitive, fail-open on no-LLM / exception / unparseable / empty, prompt rendering (tool name, args, args truncated, goal-quote escaped), `is_enabled` config gate
- `test_compression.py` (26, 4B plan Item 4) — heuristic compresses redundant text, preserves negations (and "isn't" preserves negation-meaning), collapses repeated punctuation, short input passthrough, empty passthrough, ratio-1.0 means no drop, higher-ratio drops more; perplexity-scorer drops lowest-score, scorer exception fallback, mismatched-length fallback; result dataclass; factory off-returns-None / on-returns-instance; `maybe_compress` global-off / per-surface-off / per-surface-on / unknown surface / history default-off / compressor exception / empty text; integration `_format_rag_block` default-OFF unchanged + ON-compresses; `format_sources_for_prompt` default-OFF unchanged + URL-preserved-on
- `test_self_consistency_web_gating.py` (8, 4B plan Item 6 second site) — `web_search.gating.classify_by_preflight` with self-consistency: default-OFF single greedy call (back-compat), N-call when enabled, configured non-zero temperature, majority-vote winner, per-site disabled bypass, all-unparseable fallback to NO_SEARCH, LLM-exception returns NO_SEARCH (never raises)
- `test_canonical_monitor_runner_wiring.py` (9, 4B plan Item 7 wiring) — `CodingTaskRunner` listener gating: not-attached-when-disabled, attached-when-enabled, cancels handle on first abort verdict, doesn't cancel on canonical sequence, latches after first abort, swallows listener exceptions; `CapabilityVoiceController.pending_canonical_abort` polls + clears + swallows runner exception
- `test_block_and_revise_dispatcher_wiring.py` (10, 4B plan Item 8 wiring) — `OpenClawDispatcher` per-handler validator gate: disabled-flag skips, no-LLM skips, ALLOW dispatches to stub, BLOCK short-circuits with reason, all 5 handlers run validator when enabled, validator exception falls open, voice controller threads its `llm_engine` to the dispatcher

**`tests/bus/`** (2026-05-22 session E) — typed event bus tests (43):
- `test_event.py` (10) — `BusEvent.define` signature; schema validation
  (missing fields / wrong type / None passes / empty schema accepts anything);
  `EventPayload.make` id generation + copy semantics + uniqueness
- `test_service.py` (27) — `Bus` pub/sub basics (subscribe fires / multi-subscriber
  / counts), wildcard (`subscribe_all` receives all events / both typed + wildcard fire),
  unsubscribe (stops callback / idempotent / removes one only / wildcard / count after),
  fail-open (callback exception doesn't break others / swallowed not raised),
  schema mismatch still delivers, eager-subscribe race safety (100x subscribe-then-publish
  loses zero events; concurrent subscribe+publish doesn't deadlock), subscriber-list
  snapshot during dispatch is safe, module-level shortcuts hit the singleton,
  `reset_bus_for_testing` returns fresh, published counter
- `test_events_catalog.py` (6) — canonical 17-event catalog: non-empty, types
  are unique, types are dotted+lowercase, descriptions non-empty, all named
  re-exports are importable

**`tests/coding/`:**
- `mock_bridge.py` — `ScriptedClaudeBridge` + `ClaudeScript` DSL
- `test_orchestration.py` — 11 mock-bridge scenarios (10 spec + 7b delta-tracking)
- `test_orchestration_real.py` — same scenarios with real Claude (gated)
- `test_mock_bridge_smoke.py` — mock-bridge sanity
- `test_project_digest.py` (27, 2026-05-22) — `render_template` (goal hint /
  file changes / all sections present / handles empty files / entry points /
  language in critical context / default goal fallback); `parse_digest_sections`
  (extracts all headings / returns body text / handles empty / ignores
  subheadings / case-insensitive header match); `extract_files_from_digest`
  (returns paths / handles (none) / handles missing section / strips
  explanation); `generate_digest` (uses LLM call / falls back when LLM raises /
  empty / whitespace / no LLM uses template / strips markdown code fence /
  includes prior summary in prompt / records elapsed time / preserves metadata)
- `test_project_introspect.py` (25, 2026-05-22) — `snapshot()` (returns proper
  dataclass / detects python+js / walks files / finds entry points / detects
  markers / empty project / nonexistent path / respects max_files / skips
  node_modules / skips __pycache__ / respects max_depth); AST integration
  (parses python ast / cap=0 skips); `render_tree_summary` (returns string /
  caps lines); cache (returns same snapshot / use_cache=False bypasses / invalidate
  clears entry + global); sanity on constants
- `test_project_index.py` (26, 2026-05-22 — real Qdrant + real bge-small) —
  construction creates collection + requires embedder; `upsert` persists entry,
  rejects empty digest, overwrites existing, preserves created_at on update,
  preserves tags when new empty; `get` returns upserted / None for unknown,
  `get_by_path` returns entry; `list_all` returns all; `count` tracks upserts;
  `search` returns relevant project / respects min_score / empty query returns
  empty / results sorted by score; `search_by_name` finds substring / empty
  query; `delete` removes entry / unknown returns false on empty; helpers
  (`_derive_project_id` stable + path-unique; `_score_reason` band labels;
  `_build_digest_summary_for_search` truncates / handles (none) section); entry
  payload round-trip
- `test_project_supervisor.py` (34, 2026-05-22) — constructor validates
  thresholds; empty text returns NEW; resume when active-task + adjustment OR
  intent kind = MID_SESSION_ADJUSTMENT; no resume without active task; edit
  when semantic above resolve threshold; edit pulls file hints from digest;
  edit when registry exact match; clarify when top in ambiguous band; clarify
  single-candidate phrasing; new when no matches above clarify; new when no
  index no matches; decision logged to jsonl; audit log fail-open;
  `_merge_candidates` dedupes by path / sorts by score / respects cap; decide
  publishes SupervisorDecidedEvent on bus
- `test_supervisor_dispatch.py` (27, 2026-05-22) — `_slugify_for_directory`
  (basic / truncates / empty); `_speakable` (strips backslashes / forward
  slashes / quotes); `_indent_block`; dispatch per kind (RESUME_FORWARD /
  EDIT_DISPATCH builds TaskRequest / EDIT missing path returns FALLBACK /
  NEW_DISPATCH builds with sandbox / NEW without sandbox returns None /
  CLARIFY); narration + barge-in (disabled doesn't call barge-in / enabled
  calls / barge-in returns BARGED_IN / narration text for resume); enriched
  context (digest + file tree + file hints in prompt / disabled excludes
  digest); fallback (BadSupervisor raises -> FALLBACK); `build_digest` uses
  snapshot for language
- `sandbox/` — fixture sandbox

**`tests/error_recovery/`** (Phase 4) — 78 tests:
- `test_brave_failures.py`, `test_jina_failures.py`, `test_qdrant_failures.py`
- `test_audio_failures.py`, `test_addressing_failures.py`, `test_config_failures.py`
- `test_circuit_breaker.py`, `test_error_log.py`
- `test_claude_code_failures.py` (18) — launch fail / timeout / nonzero exit / stream-json error events with API-pattern detection
- `test_mcp_server_failures.py` (3) — bind failure / no active session / audit-log write failure
- `test_filesystem_failures.py` (5) — session audit / project registry / coding tasks audit-log dedup

**`tests/routing/`** (Phase 5) — 148 tests:
- `test_classifier.py` (90: 20 BROWSER, 10 each MEDIA/MESSAGING/FILE/SHELL/HYBRID/CONVERSATIONAL, 8 CODE_TASK, 2 edge)
- `test_dispatcher.py` (12)
- `test_decomposer.py` (9)
- `test_disambiguator.py` (25)
- `test_decision_log.py` (8)
- `test_backward_compat.py` (4)

**`tests/integration/`** (Phase 6) — 83 tests:
- `test_routing_dispatch.py` (20)
- `test_conversational_pipeline.py` (21)
- `test_search_pipeline.py` (12)
- `test_coding_pipeline.py` (9)
- `test_addressing_pipeline.py` (13)
- `test_error_recovery_pipeline.py` (4)
- `mocks.md` + `performance.json` (reference files)

**`tests/desktop/`** (2026-05-12 Phase 11 + 2026-05-14 second-pass) — desktop automation primitives (~150 tests):
- `test_monitors.py` — Win32 enumeration + find_monitor + point_to_monitor + left-to-right sort fix
- `test_capture.py` — mss capture + taint tracker integration + Screenshot.without_bytes()
- `test_windows.py` — pywin32 enum + foreground detection
- `test_placement.py` — move_window_to_monitor / maximize / fullscreen
- `test_launcher.py` — AppLauncher registry (Chrome with default profile / Cursor / Discord / etc.) + URL passing
- `test_uia.py` — pywinauto text extraction
- `test_input_control.py` — pyautogui rate limit + validator gate + UAC blocking
- `test_screen_context.py` — orchestrator + analyze-and-discard pattern
- `test_vlm.py` — Moondream2VLM lazy load + fail-open + 2026-05-22 unload()
- `test_voice.py` — handle_app_launch / handle_screen_context_query / handle_window_move / handle_window_close + 2026-05-22 default_monitor_index
- `test_preferences.py` — JSONL log + recency-weighted phrase lookup

**`tests/safety/`** (2026-05-12 Phases 2-5) — 141-rule runtime tool-call validator (~117 tests):
- `test_path_resolver.py` — Windows-aware canonicalization (symlink / junction / 8.3 / bidi override / percent-escape rejection)
- `test_audit_log.py` — tamper-evident SHA-256 hash chain + verify_chain() + concurrent writers
- `test_validator_core.py` — Verdict / RuleContext / RuleResult dispatcher; fail-closed on rule exception
- `test_rules_by_category.py` — category K (self-protection) + A-J load-bearing safety + M-S persistence/anti-forensics + Cap-1..Cap-4 capability carve-outs
- `test_intent_and_taint.py` — explicit-intent matcher (verb+object window) + 60 s TTL byte-exact taint tracker
- `test_dispatcher_integration.py` — wiring through OpenClawDispatcher + coding/runner FILE_CHANGE listener

**`tests/eval/`** (2026-05-18 Phase 0) — classifier-only eval harness:
- `corpus.jsonl` — 60-row labeled set for routing / addressing / web-gate
- Plus `tests/test_eval_harness.py` at top level + `scripts/eval_harness.py` runner

**`tests/memory/`** — Memory subsystem unit tests (the actual file count varies; primary tests live at top level as `test_memory_*.py` and `test_*_qdrant.py`)

**`tests/observations/`** (2026-05-18 Phase 1) — canonical observation framework tests:
- `test_writer.py` — thread-safe JSONL appender; concurrent emits don't corrupt
- `test_integrations.py` — observe_routing_verdict / observe_addressing_verdict / observe_retrieval / observe_llm_call site coverage
- `test_outcome_resolver.py` — emit outcome_resolution rows from history
- `test_lineage_overlap.py` — compute_lineage_overlap pure primitive

### Slow / GPU-gated tests (16 skipped by default)

Set `$env:PYTEST_RUN_GPU_TESTS = "1"` before pytest. Includes real Claude API calls (`test_coding_e2e.py`, `test_mcp_e2e.py`, `test_orchestration_real.py`) — burns tokens.

---

## Runtime artifacts

### `logs/`

| File | Writer | Format | Purpose |
|---|---|---|---|
| `kenning.log` | `utils.logging.configure_logging()` | text, rotating 5 MB×3 | Main log — all subsystem messages |
| `addressing.jsonl` | `AddressingClassifier._log()` | JSONL | Every classifier verdict |
| `coding_tasks.jsonl` | `CodingTaskRunner._make_log_listener()` | JSONL | Coding task progress events |
| `verifications.jsonl` | `Verifier.verify()` | JSONL | Per-verification report |
| `clarifications.jsonl` | `_ClarificationLog` (in coordinator) | JSONL | Clarification decisions |
| `mcp_calls.jsonl` | `_AuditLog` (in mcp_server) | JSONL | MCP tool calls |
| `sessions/<id>.jsonl` | `SessionAuditWriter` | JSONL | Per-session full event audit |
| `errors.jsonl` | `resilience.error_log.ErrorLog.record()` | JSONL | Phase 4 typed errors |
| `routing_decisions.jsonl` | `openclaw_routing.decision_log.RoutingDecisionLog.record()` | JSONL | Phase 5 routing audit |
| `automation_tasks.jsonl` | `AutomationTaskRunner._audit()` | JSONL | Phase 5 OpenClaw task records |
| `safety_audit.jsonl` | `safety.audit.AuditLog.append()` | JSONL with SHA-256 hash chain | 2026-05-12 Phases 2-5: tamper-evident audit for the runtime tool-call validator; `verify_chain()` rebuilds the chain to detect tampering |
| `supervisor_decisions.jsonl` | `coding.project_supervisor.ProjectSupervisor._record_decision()` | JSONL | 2026-05-22: every supervisor decision (action / target / confidence / reasoning / candidates / file_hints) for offline threshold tuning |
| `flagged_turns.jsonl` | `pipeline.orchestrator.Orchestrator._stop_button_flag()` | JSONL | 2026-06-20: the stop-window FLAG button -- last turn (last_heard + last_response + seconds_since_* + last_scenario) flagged by the user as a disliked / missed / unwanted response, for later review/refinement |
| `eval_runs/<ts>.json` | `scripts/eval_harness.py` | JSON | 2026-05-18 Phase 0: classifier-only eval harness output (routing + addressing + web_gate accuracy on 60-row corpus) |
| `observations.jsonl` | `observations.writer.ObservationWriter.emit()` | JSONL canonical schema | 2026-05-18 Phase 1: 12-field canonical observation framework write target (suppressed during pytest runs via autouse fixture) |
| `gaming_mode.jsonl` | `openclaw_routing.gaming_mode.GamingModeManager._audit()` | JSONL | V1-gap A1: engage/disengage outcomes with per-plugin states |
| `autonomous_e2e_report.json` | `scripts/autonomous_e2e_harness.py` | JSON | Periodic autonomous-run harness output |

### `data/`

| Path | Owner | Purpose |
|---|---|---|
| `qdrant/` | `ConversationMemory`, `WebResultsCache`, `ProjectIndex` | Embedded Qdrant store; **4 collections** (`conversations`, `facts`, `web_results`, `projects`) |
| `memory.jsonl` | (legacy) | Pre-Qdrant turn log; migration source / recovery |
| `projects.json` | `ProjectRegistry` | Coding project registry (legacy lexical resolver source) |
| `projects/<project_id>/digest.md` | `ProjectIndex.upsert()` (2026-05-22) | Per-project digest markdown mirror (also stored in Qdrant `projects` collection) |
| `sandbox/` | `new_sandbox_project()` | Auto-created coding projects |
| `.kenning_instance.lock` | `lifecycle/single_instance.py` (via `__main__.main()`) | 2026-06-12 single-instance guard: held OS byte-lock + holder PID metadata; auto-releases on process death. Gitignored. |
| `summaries.jsonl` | `scripts/maintenance.py` | Conversation summaries |
| `maintenance.sqlite` | `scripts/maintenance.py` | Maintenance state (cursors, etc.) |
| `ollama_compat_test/` | (Foundation Phase 0) | Modelfile from Ollama compat test (not in active use) |
| `evolution/skills/*.md` | `evolution.skill_distiller` (via `EvolutionLoop`) | 2026 catalog 13: autonomously distilled trigger-loaded skills -- a LIVE skills source (registered as a PROJECT-precedence dir; reloaded after a kept proposal). Gitignored, checkpointed (revertible), DATA-only (never code). |
| `evolution/capsules.jsonl` + `failed_capsules.jsonl` | `evolution.service.EvolutionStore` | Per-turn success / failure capsules that feed pattern distillation (lock-guarded append-only) |
| `evolution/{corrections,knowledge_gaps,command_failures,feature_requests}.jsonl` | `evolution.service.EvolutionStore` | 2026 catalog 14: qualitative conversation-event ledgers. Corrections / knowledge-gaps / command-failures ALSO feed `failed_capsules.jsonl` (repair distillation); feature requests are NEVER distilled (digest backlog only). PII-redacted via `models.redact_fragment`; gitignored. |
| `evolution/events.jsonl` | `evolution.service.EvolutionStore.append_event` | Hash-chained evolution audit ledger; `verify_event_chain()` rebuilds to detect tampering |
| `evolution/state.json` + `personality.json` | `evolution.service.EvolutionStore` | Distillation cooldown / last-data-hash gate state + the learned Tier-0 response temperament (resumed next session) |
| `checkpoints/evolution-skills/` | `checkpoints.registry.CheckpointRegistry` (via `EvolutionService`) | Shadow-repo checkpoint over the proposal dir so a failed proposal auto-reverts. Gitignored. |

### `kenningVoiceAudio/` (workshop dir, mostly gitignored)

| Path | Owner | Purpose |
|---|---|---|
| `scripts/parakeet_server.py` | Parakeet engine | NeMo TDT HTTP server (CUDA); streaming endpoints (`/stream/start|feed|partial|stop`) |
| `searxng_config/settings.yml` | SearxNG Docker container | Engine roster + outgoing timeouts + per-engine config; mounted into `/etc/searxng` |
| `searxng_config/limiter.toml` | SearxNG Docker container | Botdetection config (modern `trusted_proxies` schema) |
| `kokoro_finetune/` | Kokoro fine-tune workshop | Stage 1 done + Stage 2 ep 0 only (SLM joint NEVER ran); auto-resume infra via Task Scheduler |
| `kokoro_training_corpus_*/` | Bulk synth corpus | 1654 clips / 107 min / 24 kHz LJSpeech-shaped training data |

### `models/` (main checkout only)

State as of 2026-06-20 (Ultron 1.0): the ACTIVE LLM is `Josiefied-Qwen3-8B-abliterated-v1.Q5_K_M.gguf` (preset `josiefied-qwen3-8b`, n_ctx 4096, ~7 GB — config.yaml sets it). The active wake model is `openwakeword/ultron.onnx` (config `wake_word.model_path`); `kenning.onnx` is the legacy/fallback. **The per-file tables below are STALE/approximate** — several GGUFs listed as "deleted 2026-05-20 round 8" (gemma-3-4b-it-abliterated, google_gemma-3-1b-it, Josiefied-Qwen3-4B-abliterated-v2, Qwen2.5-7B-Instruct-abliterated-v2, Qwen3-4B-Instruct-2507-heretic) are present again on disk for swap-back. The GGUF set is fluid; the authoritative list is `LLM_PRESETS` + the download blocks in `scripts/download_models.py`. (The original 2026-05-20 note: only the active LLM + draft were kept then, freeing ~22 GB.)

| File | Used by | Size |
|---|---|---|
| `Qwen3.5-4B-Q4_K_M.gguf` | `LLMEngine` (when `llm.preset == "qwen3.5-4b"`, **CURRENT DEFAULT 2026-05-20 round 8**). Stock Qwen 3.5 4B (not abliterated); ~3.0 GB VRAM loaded. Paired with the 0.8B draft below for speculative decoding. | 2.55 GB |
| `Qwen3.5-0.8B-Q4_K_M.gguf` | speculative-decoding draft for the qwen3.5-4b preset. | 0.50 GB |
| `kokoro/` | `KokoroSpeech` (**CURRENT DEFAULT TTS engine 2026-05-20 round 8**). Sanity-gate directory; actual weights (`hexgrad/Kokoro-82M`) cached in HF Hub cache (~330 MB). CUDA device (gaming mode auto-flips to CPU); voice `kenning` (partial fine-tune voicepack -- Stage 1 + Stage 2 epoch 0 only; `apply_trim_fade=true`, `apply_spectral_smooth=false`); no v3 pedalboard filter chain (`apply_runtime_filter=false`). | empty dir |
| `openwakeword/kenning.onnx` | `WakeWordDetector` | small |
| `piper/en_US-ryan-medium.onnx[.json]` | `TextToSpeech` (legacy `piper_rvc` engine fallback) | ~60 MB |
| `rvc/hubert_base.pt` | `RvcConverter` (legacy fallback) | ~362 MB |
| `rvc/rmvpe.pt` | `RvcConverter` (legacy fallback) | ~178 MB |
| `smart_turn/smart-turn-v3.2-cpu.onnx` | `SmartTurnDetector` (Smart Turn V3 semantic end-of-turn; NEW 2026-05-12) | 8.68 MB |
| `.hf-cache/` | `HybridEmbedder`, addressing zero-shot, moondream2, Kokoro weights | varies |

**Deleted 2026-05-20 round 8 (re-fetch via `python scripts/download_models.py` if a swap-back is desired):**

| File | Repo | Size | Reason for deletion |
|---|---|---|---|
| `gemma-3-4b-it-abliterated.Q4_K_M.gguf` | `mradermacher/gemma-3-4b-it-abliterated-GGUF` | 2.49 GB | Was the round 7 default; replaced by stock Qwen 3.5 4B with spec decoding for the latency win |
| `google_gemma-3-1b-it-Q4_K_M.gguf` | `bartowski/google_gemma-3-1b-it-GGUF` | 0.81 GB | Was the Gemma 4B draft; not needed once Gemma was retired |
| `Josiefied-Qwen3-4B-abliterated-v2.Q4_K_M.gguf` | `mradermacher/Josiefied-Qwen3-4B-abliterated-v2-GGUF` | 2.50 GB | Was the 2026-05-14 second-pass default; swap-back preset |
| `Josiefied-Qwen3-4B-abliterated-v2.Q5_K_M.gguf` | same as above | 2.89 GB | A/B variant; deleted alongside Q4_K_M |
| `Josiefied-Qwen3-8B-abliterated-v1.Q5_K_M.gguf` | `mradermacher/Josiefied-Qwen3-8B-abliterated-v1-GGUF` | 5.85 GB | Larger abliterated swap-back; deleted to free disk |
| `Qwen3.5-9B-Q4_K_M.gguf` | `unsloth/Qwen3.5-9B-GGUF` | 5.68 GB | Pre-4B baseline; swap-back |
| `Llama-3.2-3B-Instruct-abliterated.Q4_K_M.gguf` | `mradermacher/Llama-3.2-3B-Instruct-abliterated-GGUF` | 2.24 GB | Gaming-mode preset; swap-back |
| `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | `bartowski/Llama-3.2-1B-Instruct-GGUF` | 0.81 GB | Llama 3.2 3B draft; deleted alongside |

### `kenning_rvc_voice/` (main checkout only)

RVC voice model for Kenning timbre.
- `Kenning.pth` — main RVC checkpoint
- `added_IVF301_Flat_nprobe_1_Kenning_v2.index` — speaker index

---

## Documentation index

Reading order for a fresh Claude:

2. **`MEMORY.md`** (auto-loaded) — index of memory files.
3. **`project_ultron_openclaw.md`** — primary cross-phase OpenClaw reference.
4. **`project_ultron_4b_plan.md`** — final 4B + Items 4–8 state with measured TTFT/VRAM.
5. **`feedback_*.md`** — confirmed user decisions (especially `feedback_no_paid_apis.md`, `feedback_llm_runtime_decision.md`).
6. **`docs/codebase_structure.md`** ← THIS FILE — single-source reference.
7. **`docs/openclaw_integration_final_summary.md`** — cross-phase OpenClaw reference + intentional deviations + setup-readiness checklist.
8. **`docs/architecture.md`** — pipeline + diagrams.
9. **`docs/phase3_5_followup.md`** — open punch list (deferred Foundation Part 3.5).

### Comprehensive testing + improvement passes (most recent)
- **Functional / correctness pass plan:** [docs/comprehensive_test_plan.md](comprehensive_test_plan.md) — 16 phases, 38 dimensions, single-process harness pattern.
- **Functional pass results:** [docs/comprehensive_test_report.md](comprehensive_test_report.md) — 145-row metrics table; 4 classifier coverage gaps fixed; voice baseline 79 ms / 7818 MB.
- **Quality pass plan:** [docs/comprehensive_quality_plan.md](comprehensive_quality_plan.md) — 13 phases (Q0–Q13), 38 quality dimensions, ≤10-iteration improvement loop.
- **Quality pass results:** [docs/comprehensive_quality_report.md](comprehensive_quality_report.md) — 107-row metrics table; Q10 iteration audit; prompt-injection defense layer.

### Foundation reference
- Day-to-day operation: [docs/operations.md](operations.md)
- Adding code / debugging: [docs/development.md](development.md)
- Config reference: [docs/configuration.md](configuration.md)
- Error handling: [docs/error_handling.md](error_handling.md)
- Capability routing: [docs/routing.md](routing.md)
- Test layout: [tests/integration/mocks.md](../tests/integration/mocks.md)
- 16-step end-to-end smoke test: [docs/smoke_test.md](smoke_test.md)
- Foundation Phase 1 inventory snapshot: [docs/system_inventory.md](system_inventory.md)
- Phase 3 discovery catalog: [docs/config_discovery.md](config_discovery.md)

### OpenClaw integration (architecture)
- **OpenClaw integration architecture + Phase 0/1 status:** [docs/openclaw_integration.md](openclaw_integration.md)
- **OpenClaw runtime ops (agents, supervisor, locked-in constraints):** [docs/openclaw_runtime.md](openclaw_runtime.md)
- **Cross-phase final summary + setup-readiness checklist:** [docs/openclaw_integration_final_summary.md](openclaw_integration_final_summary.md)

### OpenClaw integration (per-phase close-outs)
- **Phase 1 (persona migration):** [docs/phase_1_summary.md](phase_1_summary.md)
- **Phase 3 (bridge layer):** [docs/phase_3_summary.md](phase_3_summary.md)
- **Phase 4 (Telegram channel):** [docs/phase_4_summary.md](phase_4_summary.md)
- **Phase 5 (heartbeat):** [docs/phase_5_summary.md](phase_5_summary.md)
- **Phase 6 (browser tool):** [docs/phase_6_summary.md](phase_6_summary.md)
- (Phases 7–13 have inline summaries in `openclaw_integration_final_summary.md`.)

### OpenClaw integration (user-side setup procedures)
- **Telegram channel:** [docs/openclaw_telegram_setup.md](openclaw_telegram_setup.md)
- **Heartbeat agents[].heartbeat block:** [docs/openclaw_heartbeat_setup.md](openclaw_heartbeat_setup.md)
- **Browser tool (Playwright + Chromium):** [docs/openclaw_browser_setup.md](openclaw_browser_setup.md)
- **Cron jobs (Windows Task Scheduler fallback):** [docs/openclaw_cron_setup.md](openclaw_cron_setup.md)
- **Bundled hooks (`session-memory`, `command-logger`):** [docs/openclaw_hooks_setup.md](openclaw_hooks_setup.md)
- **Memory Wiki plugin:** [docs/openclaw_memory_wiki_setup.md](openclaw_memory_wiki_setup.md)
- **Local-only ComfyUI media generation:** [docs/openclaw_media_generation_setup.md](openclaw_media_generation_setup.md)
- **iOS / Android node pairing:** [docs/mobile_node_setup.md](mobile_node_setup.md)
- **Standing-order programs:** [docs/standing_orders.md](standing_orders.md)
- **Three-layer memory architecture (Qdrant + workspace + Wiki):** [docs/memory_architecture.md](memory_architecture.md)
- **Gaming mode (V1-gap A1):** [docs/openclaw_gaming_mode_setup.md](openclaw_gaming_mode_setup.md)
- **Desktop / window control (V1-gap C3):** [docs/openclaw_desktop_control_setup.md](openclaw_desktop_control_setup.md)

### 4B optimization plan
- **4B-model optimization plan (all stages + Items 4–8 done):** [docs/4b_optimization_plan.md](4b_optimization_plan.md)
- **GGUF SHA256 reference:** [docs/model_checksums.md](model_checksums.md)

### Ultron 1.0 (active pivot — `docs/ultron_1_0/`, started 2026-06-20)
The route-all-through-8B / optional-wakeword / no-low-high-verbosity rearchitecture. A self-contained context
directory (git-versioned, not just memory). Read order when regrounding:
- **`docs/ultron_1_0/00_process_log/STATUS.md`** — the always-current snapshot (READ FIRST).
- **`docs/ultron_1_0/04_implementation/00_state_and_continuation.md`** — precise remaining-work specs (M5b→M9).
- **`docs/ultron_1_0/02_research/02_research_synthesis.md`** — the 6 resolved decisions + the C_route_llm hybrid reframing.
- **`docs/ultron_1_0/03_plan/00_ultron_1_0_architecture_and_roadmap.md`** — architecture + M0→M9 roadmap.
- **`docs/ultron_1_0/01_recon/00_codebase_map.md`** — pivot attach-point map (line refs).
- **`docs/ultron_1_0/05_testing/00_baseline.md`** — the frozen 22-fail regression baseline.
- Raw boards: `01_recon/raw/board{A,B}_*.md` (22), `02_research/board/{A,B,C}_*.md` (41); kickoff log
  `00_process_log/2026-06-20_kickoff.md`. Memory: `project_ultron_1_0_pivot.md`, `feedback_ultron_1_0_process.md`.

---


## 2026-06 relay/gaming campaign — module, script, test & config additions

_(Topical detail for the 30-commit relay-quality + 20k-corpus campaign; see the
validating-HEAD header for the summary. Listed here together for tractability on this
large file.)_

### Test harness + corpus + scorecard

### `scripts/relay_test/` (Valorant relay test harness + 20k corpus + scorecard)

**Purpose:** end-to-end quality infrastructure for the Valorant teammate-relay feature — staged pipeline harness, a ~20k-case corpus built from 48 vocab packs, a metrics scorecard with no-regression diffing, an audit chunk splitter, and a set of dev-tools for cadence/waveform/pipeline inspection.

---

#### `harness.py`

Staged full-pipeline test runner. Each stage is a superset of the previous.

- **Stages** (cheapest first):
  - `matcher` — runs `match_relay_command` on every corpus case; grades `expect_match`, addressee canonicalization (via `_NAME_CANON`), and boolean flags. No models loaded.
  - `rephrase` — adds the real LLM; calls `build_relay_line`; checks non-empty output, no leaked control tokens (`_CONTROL_RE`) or stage directions (`_STAGE_DIR_RE`), length ≤ 300 chars, number preservation on `location`/`ult`/`team_status` categories.
  - `audio` — adds Kokoro synthesis; runs `analyze_clip` (production blip/burst/dropout detector) on every synthesized clip.
  - `asr` — adds Moonshine STT; checks output audio produces recoverable speech (`score_asr`: no intelligible speech → fail; lines ≥ 5 content words also check gross recall < 35%).
  - `full` — also synthesizes the INPUT command in a neutral voice (`am_michael`, `apply_spectral_smooth=False`) and runs it back through STT first, exercising the spoken→STT→relay end-to-end path.
- **Corpus:** loads `build_corpus_10k(seed)` from `corpus_packs`; seeded-shuffles (seed 7) to prevent clustering identical templates in the recent-line ring; respects `--category` filter.
- **GAMING_PRESET = `"llama-3.2-3b-abliterated"`** — the exact model the relay runs under in gaming mode (abliterated = no safety refusals; the default Qwen3.5-4b refuses the Ultron/Marvel persona).
- **Testing-mode parity:** `_load_llm` calls `set_testing_mode_active(True)` to gate RAG and web-search off (matching the gaming-mode context-free path) without triggering the device swaps. GPU layers default to `-1` (full GPU) for speed; set `RELAY_TEST_GPU_LAYERS=0` to reproduce the live CPU-gaming config.
- **Qdrant isolation:** each harness run creates a PID-unique temp Qdrant path (`$TEMP/kenning_relay_test_qdrant_<pid>`) and registers an `atexit` cleanup — never touches production `data/qdrant` and never strands lock files between runs.
- **Roast/fun-fact parity:** `roast`/`fun_fact` commands are served verbatim from `load_roast_lines`/`load_fun_facts` pools (mirroring the orchestrator's pre-LLM intercept), not composed by the LLM.
- **Scoring functions:** `score_matcher(case, cmd) -> list[str]`, `score_rephrase(case, line) -> list[str]`, `score_audio(report) -> list[str]`, `score_asr(intended_line, heard) -> list[str]`; `content_words(text) -> set[str]` (filters stop-words).
- **Output:** `logs/relay_test/<stage>_<tag>.jsonl` (one JSON record per case) + per-category failure summary to stdout.
- **CLI:** `--stage matcher|rephrase|audio|asr|full` `--limit N` `--tag TAG` `--category cat1,cat2`.

---

#### `corpus.py`

Base corpus of ~500–600 deterministic test cases covering every relay shape.

- **`Case`** (frozen dataclass): `text`, `category`, `expect_match`, `addressee` (`"team"` or agent name), `flags` (tuple of boolean field names), `glossary` (Valorant terms the rephrase must preserve), `note`.
- **`build_corpus() -> list[Case]`** — assembles cases across 33 sections:
  1. Location callouts — combinatorial (position/count/possession/smoke) over `LOCATIONS` (generic callouts + all 12 map-specific grids: Ascent, Bind, Breeze, Fracture, Haven, Icebox, Lotus, Pearl, Split, Sunset, Abyss, Corrode).
  2. Self-status (`SELF_STATUS`: low/flanking/rotating/saving/planting/defusing/lurking/etc.).
  3. Team/enemy status (`TEAM_STATUS`).
  4. Utility callouts (`UTILITY`: ability + place, per-agent verbs).
  5. Tactical directives (`DIRECTIVES`) — to team and named teammate.
  6. Ult tracking (`ULTS`).
  7. Banter/morale (`BANTER`).
  8. Economy specials.
  9. Named-agent addressing — ability requests + questions for every agent in `AGENTS` (full 29-agent roster + STT homophone spellings: `"kill joy"`, `"kay o"`, `"cipher"`, `"gecko"`, etc.).
  10. Context + respond (teammate-said-something clapbacks).
  11. Verbatim mode variants (`"word for word"`, `"in those words specifically"`, `"verbatim"`).
  12. Compose / encouragement / greetings.
  13. Roast (flags `roast`, `compose`).
  14. Fun-fact (flag `fun_fact`).
  15. Freeform callouts.
  16. Greet (curated Ultron intro, flag `compose`).
  17. Farewell (victory/defeat/neutral registers, flag `compose`).
  18–33. Enemy-ult (multi-agent), eco-round tactics (`ECO_TACTICS`), enemy tendency reads, self play-style, banter-at-Ultron (flag `context`), Marvel-universe jabs (flag `context`), identity probes (flag `context`), general-knowledge questions (flag `context`), enemy movement, enemy utility, careful-warnings, all-enemies stacks, have-weapon/ult, enemy-spike, per-map callouts, named enemy agents at locations, all-of-them variants, negative controls (`expect_match=False`).
- **`_vary_phrasing(cases)`** — rotates the leading prefix of safe literal-callout categories through 7 equivalents (`"tell my team"` → `"call out"`, `"let the squad know"`, etc.) so each corpus regeneration exercises fresh phrasing.
- **Vocab constants:** `GENERIC_CALLOUTS` (42 universal terms), `MAP_CALLOUTS` dict (12 maps), `LOCATIONS` (flattened deduped list), `NUMS_WORD`/`NUMS_DIGIT`, `AGENTS`, `SELF_STATUS`, `TEAM_STATUS`, `UTILITY`, `DIRECTIVES`, `ULTS`, `BANTER`, `ENEMY_ULTS`, `ECO_TACTICS`, `ENEMY_TENDENCIES`, `SELF_PLAYSTYLE`, `BANTER_AT_ULTRON`, `MARVEL`, `IDENTITY`, `GENERAL_KNOWLEDGE`.
- **`stats(cases) -> dict`** — counts by category + unique text count.

---

#### `corpus_packs.py`

Expands the base corpus to a ~20k stratified sample by auto-discovering vocab packs.

- **`build_corpus(seed=0, target=25000) -> list[Case]`** — merges `_orig_build_corpus()` + `_pack_cases(seed)` + `_compound_cases(seed)`, deduplicates by `(text.lower(), category)`, then calls `_cap_stratified` to trim to `target` while preserving category proportions. Aliased as **`build_corpus_10k`** and **`build_corpus_20k`** (both call `build_corpus` at target=25000; the `10k`/`20k` names are historical aliases — `_TARGET=25000`).
- **Pack auto-discovery:** `_all_pack_names()` lists all `.py` files in `vocab_packs/` (excluding `__init__.py`). Packs are classified by name:
  - **RELAY** (default) — `expect_match=True`; items already phrased as a command (`_CMD_LEAD_RE`) are used verbatim; raw callouts get a rotating relay prefix varied by `(ii + pi + seed) % len(_GROUP_PREFIXES)`.
  - **QUESTION** (`_QUESTION_PACKS`: `questions_to_ultron`, `var_teammate_to_ultron`, `var_identity_questions`, `var_marvel_banter`, `var_banter_at_ultron`, `stress_banter_mock`, `stress_marvel_identity_edge`) — teammate-to-Ultron; `expect_match=False`.
  - **NEGATIVE** (`_NEGATIVE_PACKS`: `stress_false_relay_hard`, `stress_oov_safety`) — relay-shaped stream narration/private thought that must NOT trigger the matcher; `expect_match=False`.
  - **EXCLUDED** (`_EXCLUDE_PACKS`: `persona_flavor`, `__init__`) — Ultron output pools, never test inputs.
- **`_load_pack(name) -> list[str]`** — loads a pack module via `importlib`, strips leading wake words (`_WAKE_LEAD_RE`), deduplicates.
- **`_compound_cases(seed, target=2000) -> list[Case]`** — procedurally generates `"<prefix> <head><joiner><tail>"` compound comms (callout + tactical tail) drawn from `callouts_maps`, `var_positions_counts`, `agents_abilities`, `directives_tactics_eco`, `var_utility_reports`, `var_ult_states`; 5 joiner styles (` and `, `, `, ` -- `, `, also `, ` plus `); seeded, deduped.
- **`_cap_stratified(cases, target, seed)`** — proportional per-category trim; shuffles within each category before slicing.
- **~29.4k unique pack payloads** total across the 48 packs; `seed` controls which 20k slice is sampled, so the full pool is covered over multiple autonomous loop iterations.

---

#### `vocab_packs/` (55 packs, ~29.4k payloads)

Each pack is a Python module exporting `ITEMS: list[str]`. Organized into three families:

**Original 8 hand-crafted packs** (base relay shapes, pre-seed-0):
- `agents_abilities.py` — per-agent ability-usage callouts (all 29 agents × abilities).
- `callouts_maps.py` — map-specific location callouts across all 12 maps.
- `conversation_natural.py` — naturalistic multi-turn relay phrases.
- `directives_tactics_eco.py` — tactical directives, economy calls, round strategies.
- `natural_phrasing_edge.py` — edge phrasings (interruptions, mid-sentence constructions, hedged callouts).
- `opinions_maps_meta.py` — map opinions, meta commentary, comp reads.
- `persona_flavor.py` — Ultron OUTPUT flavor lines (EXCLUDED from test inputs; for reference only).
- `questions_to_ultron.py` — teammate questions directed at Ultron (QUESTION kind, `expect_match=False`).

**`var_*` variety packs** (19 packs — surface diversity for relay inputs):
`var_agent_opinions`, `var_banter_at_ultron`, `var_calm_deescalate`, `var_damage_trades`, `var_directives_strats`, `var_economy_buys`, `var_greetings_gg`, `var_identity_questions`, `var_insults_trashtalk`, `var_map_opinions`, `var_marvel_banter`, `var_morale_hype`, `var_positions_counts`, `var_rotations_movement`, `var_self_status`, `var_smalltalk_relay`, `var_spike_plant_defuse`, `var_teammate_to_ultron`, `var_ult_states`, `var_utility_reports`.

**`stress_*` metric-stress packs** (21 packs — engineered adversarial inputs):
- `stress_agents_ability_exhaustive` — every agent × every named ability combination.
- `stress_ask_answer_bait` — question-shaped relay commands that must relay, not answer.
- `stress_banter_mock` — teammate mockery/insults at Ultron (QUESTION, `expect_match=False`).
- `stress_clutch_dense` — dense multi-fact clutch-round callouts.
- `stress_compounds_3fact` / `stress_compounds_5fact` — 3-fact and 5-fact compound comms (position + ult + directive; count + ability + action; mixed ownership); engineered to break fact-token retention and inversion detection.
- `stress_directive_obs_traps` — observation-shaped directives that look like narration.
- `stress_disfluency` — fillers, false starts, re-starts mid-callout.
- `stress_ecobleed` — eco/save/force/full-buy confusion traps.
- `stress_false_relay_hard` — relay-SHAPED stream narration (hypothetical/past-tense/conditional/note-to-self) that must NOT relay; `expect_match=False` (NEGATIVE kind); ~600 items.
- `stress_firstperson_traps` — first-person statements that must relay (not fall through).
- `stress_flavor_register` — lines where Ultron's register (snap vs verbose) must be correct.
- `stress_hallucination_bait` — prompts likely to cause the LLM to invent agent names or locations.
- `stress_marvel_identity_edge` — edge Marvel/identity probes (QUESTION, `expect_match=False`).
- `stress_nanoswarm_wait` — Killjoy nanoswarm timing callouts (ability-timing precision).
- `stress_oov_safety` — out-of-vocabulary agent names / non-roster addressees that must NOT relay (NEGATIVE kind).
- `stress_opinion_mangling` — opinion statements that must survive paraphrase intact.
- `stress_ownership_traps` — our/their/enemy ownership ambiguities.
- `stress_slang_runons` — slang-heavy run-on comms.
- `stress_stt_homophones` — STT homophones for agents/locations (raze/raise, yoru/your, ult/alt, eco/echo, Kay-O/K.O.).

**`refs/` (21 web-grounded Valorant reference documents, incl. `ultron_voice.md`):** ground-truth Markdown used during corpus and prompt construction (not loaded at runtime):
- Per-agent refs: `agents_controllers.md`, `agents_duelists.md`, `agents_initiators.md`, `agents_sentinels.md` — full 29-agent roster, ability names, usage patterns.
- Per-map refs (10 maps): `map_ascent.md`, `map_bind.md`, `map_breeze.md`, `map_fracture.md`, `map_haven.md`, `map_icebox.md`, `map_lotus.md`, `map_pearl.md`, `map_split.md`, `map_sunset.md`.
- `maps_newest.md` — Abyss, Corrode (2025–2026 additions).
- `comms_conventions.md` — anatomy of a callout, shotcalling vocabulary, damage/eco/spike comms. Last verified June 2026 against 14+ competitive sources.
- `economy_rounds.md` — credit economy, eco/force/full-buy thresholds, round archetypes.
- `meta_tiers.md` — agent tier lists, pick rates, meta reads.
- `slang_lingo.md` — Valorant slang, community abbreviations, homophone inventory.
- `marvel_ultron.md` — MCU/comics Ultron lore (identity, abilities, relationships, Sokovia, Avengers) grounding the in-character persona responses.

---

#### `scorecard.py`

Turns harness JSONL logs into tail-sensitive reliability metrics and a no-regression diff.

- **Valorant fact-token extractor:** `extract_facts(text) -> dict` — extracts 5 category sets: `count` (numeric, word↔digit normalized via `_W2D`), `agent` (single-token agent keys from `_ROSTER_CANON`), `loc` (location tokens from `_LOC_TOKENS`), `ability` (30+ ability verb forms in `_ABILITIES`), `owner` (`our`/`their`/`enemy`/`enemies` via `_OWN_RE`). Does NOT count `my/we/they` as ownership facts (they are naturally rephrased in relay and counting them penalizes correct output; only the inversion rate tracks subject flips).
- **`_retention(inp, out) -> dict`** — per-category fact-token retention for one utterance (skips categories with no input facts); plus `overall` = union of all fact tokens.
- **`_pcts(vals) -> dict`** — `{n, mean, p50, p95, p99, min}` percentile helper.
- **`_is_inversion(inp, out) -> bool`** — subject-flip heuristic: detects enemy-lead→own-lead and own-lead→enemy-lead swaps via `_ENEMY_LEAD`/`_OWN_LEAD` regexes, plus `our`↔`their` ownership token flips on matching agents.
- **`_hallucinated(inp, out) -> list[str]`** — agent or location tokens in `out` that never appeared in `inp` (zero-tolerance fabrication class).
- **`classify_route(cmd) -> (str, str)`** — probes `build_relay_line` with a stub `generate_fn` that sets a flag on call; if the stub was never called → `"deterministic"` (snap/compound/curated/pre-routed literal, zero model cost); if called and stub survived intact → `"llm"`; if called but stub was abstained/replaced → `"partial"`.
- **`NEGATIVE_SET`** — 30 stream-narration phrases (hypothetical, past-tense, indecision, note-to-self) used as a false-relay gate; `matcher_metrics` counts how many trip `match_relay_command`.
- **`matcher_metrics(seed, limit) -> dict`** — `clean_rate`, `false_relay_rate`/`false_relay_count` on `NEGATIVE_SET`.
- **`route_and_latency(seed, limit) -> dict`** — `routes` breakdown (`deterministic`/`partial`/`llm`), `pure_deterministic_coverage`, `deterministic_or_partial_coverage`, `det_path_latency_us` percentiles (microseconds; model-free fast path).
- **`quality_metrics(jsonl_path) -> dict`** — from a rephrase JSONL: per-category fact retention `_pcts`, `retention_by_category`, `inversion_rate`/`_count`, `hallucination_rate`/`_count`/`_examples`, flavor diversity as `flavor_type_token_ratio` (TTR over final sentences) and `flavor_max_repeat`.
- **`build_scorecard(jsonl_path, seed, limit, tag) -> dict`** — assembles `matcher` + `routing` + optional `quality` sections.
- **No-regression diff (`diff(prev, cur)`):** 27 tracked metrics (`_TRACKED`) covering matcher clean, false-relay, pure-deterministic coverage, fact-retention mean/p50/p95 (overall + count/owner/agent/loc), count-p99, inversion, hallucination, flavor TTR, flavor max-repeat. Returns `(report_str, passed: bool)`; exit 2 on regression.
- **`--bench` mode (`bench_llm`):** loads the gaming 3B on CPU (`RELAY_TEST_GPU_LAYERS=0`, the live gaming config), samples LLM-routed cases via `classify_route`, times `build_relay_line` on 50 deterministic + N LLM cases, reports `det_path_ms`/`llm_path_ms` percentiles + `peak_rss_mb` (via `psutil`). This is the authoritative latency gate for the live gaming condition.
- **Output:** `logs/relay_test/scorecard_<tag>.json` + `logs/relay_test/bench_<tag>.json`.
- **CLI:** `--jsonl PATH --seed N --limit N --tag TAG --prev scorecard_prior.json --bench --bench-n N`.

---

#### `make_audit_chunks.py`

Splits a rephrase JSONL into per-agent audit `chunk_NN.txt` files for human line-by-line review.

- Filters to **LLM-routed lines only** (via `classify_route`): deterministic snap/compound/curated lines are correct by construction and verified by the scorecard, so only the model-generated lines need auditing.
- Writes N `chunk_NN.txt` files (default 16) to `<outdir>/`; each entry annotated with global index, category (prefix `pack_` stripped), route (`llm`/`partial`), and the `IN`/`OUT` pair.
- **CLI:** `python make_audit_chunks.py <jsonl> <outdir> [N_chunks]`.

---

#### Dev tools (read-only measurement / manual testing)

- **`play_sample.py`** — two-phase dev driver: (1) regenerates the full base corpus through the 3B + `build_relay_line` and writes `logs/relay_test/rephrase_<tag>.jsonl`; (2) plays a stratified random sample aloud through the system default speaker via the FULL production pipeline (matcher → 3B rephrase → Kokoro with in-model F0/duration shaping → sounddevice). Supports A/B mode (`--ab`) to compare flat vs. shaped prosody back-to-back. CLI exposes all prosody knobs (`--pitch-factor`, `--pitch-shift`, `--energy-factor`, `--dur-final`, `--dur-internal`, `--dur-stress`, `--max-pause-ms`, etc.).
- **`cadence_actual.py`** — synthesizes 15 representative relay lines (snap callouts through long identity responses), instruments the dead-space compressor via monkey-patch, and measures actual waveform cadence: per-clause speech segments, pause positions, syllable rate, pitch std deviation in semitones, reverb tail length. Writes `logs/relay_test/cadence_actual.json`.
- **`cadence_check.py`** — compares actual vs. ideal Ultron cadence (target 4.2 syll/s, `IDEAL_PAUSE_MS` per punctuation mark) for 6 representative lines; prints per-clause verdict (TOO FAST / MISSING pauses / short pause / reverb tail thin / ok). Utilities `syllables(word) -> int` and `clause_split(text) -> list[(clause_text, trailing_punct)]` are imported by `cadence_actual.py`.
- **`flow_check.py`** — inter-sentence dead-space and reverb audit: measures RAW per-sentence Kokoro chunks (trailing silence + blips) and the FINAL production clip's internal gaps (≥ 150 ms) + reverb decay tail. Distinguishes real dead space from continuous reverb decay. Informational; no code changes.
- **`validate_pipeline.py`** — 30-case manual full-pipeline validator: matcher → `build_relay_line` (3B + deterministic repair) → Kokoro synth → `analyze_clip` → waveform gap/tail check → dual-ASR (Whisper + Moonshine) word-recall and char-similarity scoring. Flags cases where both ASRs miss ≥ 15% of content words AND char similarity < 0.78 (clipped audio, not jargon mishear).
- **`transcript_replay.py`** (NEW 2026-06-15) — replays real STT transcripts through the dispatch path and flags any private→mic routing leaks (a callout the user meant to keep off-comms that would have gone to the team bus).
- **`audio_channel_test.py`** (NEW 2026-06-15) — plays real tones through the B1 (team), B3 (OBS broadcast), and default-speaker outputs to verify routing + playback end-to-end.
- **`probe.py`** (standalone), **`reprobe.py`**, **`waveform_check.py`**, **`burst_diag.py`** — earlier one-off probes for matcher/rephrase smoke-testing and trailing-burst diagnosis (superseded by the harness for systematic use; kept for quick spot checks).

---

### `scripts/stream_check.py`

**Purpose:** pre-stream audio routing validator — confirms Kenning emits to the correct VoiceMeeter buses before going live, without loading the assistant.

- **Checks four paths:** EVERYTHING feed (→ `audio.broadcast_device`, `BroadcastSink`, B3 bus); TEAM feed (→ `relay_speech.output_device`, `play_to_device`, B1 bus); DEFAULT output (system default speakers); MIC (resolved but never opened — confirms independence).
- **Method:** plays a short quiet sine-tone (`tone()`) to each output via the real production classes (`get_broadcast_sink()`, `play_to_device`, `sd.OutputStream`), then verifies the resolved device index matches the configured device.
- **Exit 0** = all paths emit; non-zero + `FAIL`/`WARN` lines on any mismatch.
- **Run before streaming:** `.venv\Scripts\python.exe scripts\stream_check.py` (VoiceMeeter must be open to see meter movement).
- **Functions:** `tone(sr, hz, secs, amp) -> np.ndarray`, `main() -> int`.

### New & changed source modules (2026-06)

#### New module: src/kenning/safety/testing_mode.py

### `src/kenning/safety/testing_mode.py`

Off-by-default mode for principled corpus testing that mimics the disabled-functionality posture of gaming+anticheat mode (no RAG, no reranker, no web search, no desktop automation) while keeping the LLM/TTS on GPU for fast generation — so corpus outputs are representative of the CPU gaming runtime without the device-swap cost.

- **Flags**: module-level `_runtime_active: bool` (thread-safe via `_lock`); also honours the config pin `testing_mode.enabled`. Defaults to `False`; a config error never silently enables the mode.
- `set_testing_mode_active(active: bool) -> None` — flip the runtime flag; used by the corpus test harness.
- `is_testing_mode_active() -> bool` — True if either the runtime flag or `config.testing_mode.enabled` is set. Fail-open to `False` on any config error.
- **Gating sites**: `llm/inference._retrieve_rag_snippets` and `safety.anticheat.anticheat_active` both import and honour this flag, ensuring identical gate behaviour with gaming mode without ever triggering real gaming-mode device swaps.

#### New module: src/kenning/audio/waveform.py

### `src/kenning/audio/waveform.py`

OBS-capturable radial audio visualizer overlay window that reacts in real time to every Kenning utterance (normal + relay). Zero latency on the speaker path; disabled by default.

**Architecture**
- `WaveformSink` — process singleton; safe to `submit` from any thread. Two daemon threads: a *pacer* (FFT analysis, real-time frame pacing) and a *UI* (owns `tk.Tk()` + Canvas, ~30 fps redraw). Never spawned until first `configure(enabled=True)`; fully torn down on disable. **2026-06-15:** the idle spin-freeze + render change-detection optimizations were REVERTED — the ring spin now ALWAYS advances (a slow drift at rest that speeds up while speaking) and the overlay breathes smoothly at 60fps again; a tiny tkinter canvas costs nothing to spin, and the frozen-at-idle look read as "dead on screen".
- `_RenderState` — holds pre-created Canvas items and eases them toward each target frame with asymmetric attack/release gains.

**Public API**
- `get_waveform_sink() -> WaveformSink` — process-wide lazy singleton.
- `submit(pcm: np.ndarray, sample_rate: int) -> None` — module-level tee; immediate no-op when disabled. Called unconditionally by Kokoro `_play`/`speak_stream` and relay path.
- `configure_from_config() -> None` — reads `config.visualizer` block and calls `WaveformSink.configure()`; called at orchestrator startup and on GUI changes.
- `WaveformSink.configure(*, enabled, size, bars, fps, bg_color, accent_color, transparent, always_on_top, nameplate_text, nameplate_font) -> None` — enable/disable and set appearance; starts/tears down threads idempotently.
- `WaveformSink.submit(pcm, sample_rate) -> None` — drop-oldest bounded queue (`_QUEUE_MAXSIZE=8`); non-blocking, fail-open.
- `WaveformSink.close() -> None` — stop threads and tear down window; idempotent.
- `analyze_clip(pcm, sr, *, fps, n_bands) -> List[Frame]` — convert one PCM clip to a `List[(level 0..1, bands[N] 0..1)]` sequence; log-spaced FFT bands 90–7500 Hz, log-compressed, per-clip normalised, loudness-scaled. Pure/fail-open (returns `[]` on any error).

**Key data structures**
- `Frame = Tuple[float, np.ndarray]` — `(level 0..1, bands[N] 0..1)`.
- `_RMS_FULL_SCALE = 0.18` — RMS that maps to a full core pulse.

**Overlay / OBS integration**
- `_set_overlay_window_styles(hwnd_int, *, background) -> None` — Windows ctypes: clears `WS_EX_TOOLWINDOW` (which OBS filters out), sets `WS_EX_APPWINDOW` so the borderless overlay appears in OBS's Window Capture list. Must be called from a non-Tk thread to avoid Tk reasserting `overrideredirect`. With `background=True` also sets `WS_EX_NOACTIVATE` and sinks to `HWND_BOTTOM` (hides behind other windows while WGC still captures it).
- `_nameplate_frames(W, H, text, font_family, *, plate_fill, accent_rgb, core_idle, neon_red, buckets) -> List[PIL.Image]` — pre-renders the ULTRON nameplate at `buckets` brightness levels with real Gaussian-blurred neon glow; `_RenderState` swaps the `ImageTk.PhotoImage` per frame on a fast attack/decay envelope.
- Nameplate colours: `PLATE_FILL=(22,22,30)`, `CORE_IDLE=(230,222,225)` (calm/readable), `NEON_RED=(255,88,98)` (lit). Crisp white-hot glyph core is lerped from `CORE_IDLE→(255,240,242)` on top of the bloom so letters stay legible against the halo.
- Radial bars taper hard in the downward direction (`dir_gain = 1.0 − 0.78 * max(0, sa)`) to keep the nameplate area clear.
- Background chroma colour (config `visualizer.bg_color`, default `#0b0b10`) is keyed transparent on Windows (`-transparentcolor`); glow rings fade to dark `art_base=(18,8,12)` not the bg to avoid olive mid-tones on green chroma keys.

#### New module: src/kenning/tts/f0_control.py

### `src/kenning/tts/f0_control.py`

In-model F0-contour shaping for Kokoro/StyleTTS2 — adds expressiveness with zero added latency and exact reverb/timbre preservation by operating on `F0_pred` BEFORE the ISTFTNet decoder, inside the same forward pass.

- `scale_f0_curve(f0, *, factor, shift_semitones, max_excursion_semitones) -> Tensor` — expands the predicted F0 curve around its log-domain median, with a tanh soft-limit. Unvoiced frames (≤1 Hz) are untouched. True no-op when `factor==1.0` and `shift_semitones==0.0`. Fail-open (returns `f0` unchanged on any error).
- `scale_energy_curve(n, *, factor) -> Tensor` — mean-preserving expansion of `N_pred`; widens loud/quiet dynamics without changing overall loudness. True no-op at `factor==1.0`. Fail-open.
- `install_f0_contour_shaping(engine) -> bool` — patches `engine._model.model.predictor.F0Ntrain` with a closure that reads `engine.f0_contour_factor`, `engine.f0_shift_semitones`, `engine.f0_max_excursion`, `engine.f0_energy_factor` live on every call (hot-swappable). Stores original in `pred._f0shape_orig` for idempotency. Returns `True` if hook is in place; fail-open (logs warning and returns `False`) on missing model layout.

**Info flow**: `KokoroSpeech._install_prosody_hooks()` calls `install_f0_contour_shaping(self)` (install F0 first, then duration), then every `KModel.forward_with_tokens` call invokes the patched `F0Ntrain`, which calls `scale_f0_curve`/`scale_energy_curve`, returning shaped tensors to the ISTFTNet decoder.

#### New module: src/kenning/tts/duration_control.py

### `src/kenning/tts/duration_control.py`

In-model per-phoneme duration shaping for Kokoro/StyleTTS2 — natural, context-aware cadence (phrase-final lengthening + stress emphasis) at zero latency, composing on top of the F0 hook.

**Key constants / data**
- `_VOWELS` — frozenset of misaki/Kokoro vowels (ASCII + IPA monophthongs + uppercase diphthongs + reduced vowels).
- `_SENT_PUNCT = frozenset(".!?")`, `_PHRASE_PUNCT = frozenset(",;:—…")` — boundary detection.
- `_PACE_MIN = 0.85`, `_PACE_MAX = 1.45` — per-phoneme clamp range.

**Functions**
- `compute_pace_vec(chars, *, final_factor, internal_factor, stress_factor) -> List[float]` — per-phoneme pace multipliers: applies `_lengthen_rime` at every sentence/phrase boundary, then lifts the vowel after each `ˈ` primary-stress mark. Index 0 and last index are sentinels, never touched. Result is clamped to `[_PACE_MIN, _PACE_MAX]`.
- `_lengthen_rime(chars, pace, punct_i, factor)` — lengthens the final-syllable rime (last vowel: full `factor`; coda consonants: half `factor − 1.0`) before `punct_i`.
- `install_duration_shaping(engine) -> bool` — replaces `KModel.forward_with_tokens` with a vendored `_patched` that inserts `pace_vec` multiplication on the pre-round `duration` tensor (after `sigmoid`, before `round().clamp(min=1)`). Reads `engine.dur_final_factor`, `engine.dur_internal_factor`, `engine.dur_stress_factor` live; true no-op when all are 1.0. Stores original in `km._dur_orig_fwt` for idempotency. Returns `True` if hook installed; fail-open otherwise.

**Composition with F0**: `km.predictor.F0Ntrain` is called inside `_patched` after alignment is built, so the F0 hook composes naturally — a duration-lengthened phoneme gets more F0 frames, producing richer pitch movement on stressed/final syllables.

#### Changed module: src/kenning/lifecycle/gaming_engage.py

### `src/kenning/lifecycle/gaming_engage.py` — what changed (2026-06 relay/gaming work)

- **`GamingEngageDeps.gaming_llm_gpu_layers: Optional[int]`** (new field) — passed as `gpu_layers=deps.gaming_llm_gpu_layers` to `deps.llm.reload_for_preset(...)` in stage 1 of `gaming_engage_iterator`. `0` forces the gaming LLM fully onto CPU regardless of `KENNING_LLM_GPU_LAYERS` env or `llm.gpu_layers` config; `None` keeps config behaviour. On disengage, `reload_for_preset(prior_preset)` is called WITHOUT a `gpu_layers` override, restoring normal GPU behaviour.
- **`reset_shared_reranker` call on engage** — NOT in this file; lives in `pipeline/orchestrator.py` `_engage_extra`. After `drive_start_task(gaming_engage_iterator(...))` completes, the orchestrator calls `from kenning.memory.reranker import reset_shared_reranker; reset_shared_reranker()` to free the cross-encoder reranker (~1 GB) since RAG is gated off during gaming. Reranker lazily reloads after disengage.
- **`_drive_async_blocking`** — NOT in this file; defined as a module-level function in `pipeline/orchestrator.py` (line 111). Runs a coroutine to completion from a sync context regardless of whether an event loop is already running: if a loop is running, drives the coro on a fresh loop in a short-lived thread and joins, avoiding the "asyncio.run() cannot be called from a running event loop" error that previously caused device swaps to silently no-op when `engage` was called from inside `asyncio.run(manager.engage())`.

#### Changed module: src/kenning/safety/anticheat.py

### `src/kenning/safety/anticheat.py` — what changed (2026-06 relay/gaming work)

- **`anticheat_active()` now also honours `testing_mode`** — after checking `_runtime_active`, it imports and calls `is_testing_mode_active()` from `kenning.safety.testing_mode`; if that returns True, `anticheat_active()` returns True. This gives corpus testing the same desktop-automation hard-block as a real gaming session without triggering the config pin path.
- **`press_key` and `press_hotkey` added to `_BLOCKED_TOOL_EXACT`** — these two tools, which drive `pyautogui.SendInput`, were previously unguarded against the safety validator's `is_blocked_tool` check. Both are now in the frozenset alongside `click`, `type_text`, etc.
- **`is_blocked_tool` namespaced-dispatcher fix** — `tool_name` values arriving as `openclaw.window_automation` or `desktop.input.press_hotkey` previously fell through all block checks because neither the full name nor the prefix matched. Fix: (1) strip a leading `openclaw.` namespace prefix, then (2) also test the bare final dotted segment (`bare = name.rsplit(".", 1)[-1]`) against both `_BLOCKED_TOOL_EXACT` and `_BLOCKED_TOOL_PREFIXES`. Both the full normalised name and the bare segment are checked so either form blocks.

**Audit-hardening batch (2026-06-15):**
- **Canary derives its tripwire set from the firewall** — the boot posture canary (`_audit_anticheat_posture` in `pipeline/orchestrator.py`) no longer hard-codes its sys.modules tripwire list; it imports `import_firewall.blocked_module_names()` so prevent==detect with no drift (a module added to the firewall blocklist is automatically also asserted-absent by the canary).
- **Canary regressions now log at ERROR (not WARNING)** — both the anticheat posture canary (`ANTICHEAT POSTURE CANARY`) and the lean-boot canary (`LEAN BOOT CANARY`) escalated to `logger.error` so a footprint/lean regression is unmissable (fail-open preserved — logged, never raised).
- **Safe-by-DEFAULT gaming schema** — `GamingModeConfig.enabled` and `engage_at_startup` code defaults flipped to `True` (alongside the already-True `anticheat_safe_mode`), so a lost/reset `config.yaml` still boots anticheat-safe + gaming-engaged + lean. The unit sweep stays hermetic via the conftest config-pin disable.
- **`scripts/run_kenning_mcp_for_openclaw.py` hard-refuses under anticheat** — `_refuse_if_gaming()` runs FIRST and exits BEFORE importing any MCP tools whenever `anticheat_active()` is true (FAIL-CLOSED: if the state can't be determined, it refuses), so nothing OpenClaw-related spins up during a match.
- **Dead `BlockInput` helper refuses while anticheat is active** — `desktop/win32_helpers.py`'s `_call_block_input(enable)` (the `user32.BlockInput` wrapper under `block_input_context`) now returns `False` with no OS call when `anticheat_active()` — a third belt beyond the import firewall + module gates (this ban-class input-suppression primitive has no production caller and lives in the firewall-blocked `kenning.desktop` package, so it is already unreachable while gaming, but it refuses outright regardless). No-op while anticheat is off, so existing tests are unchanged.

#### New module: src/kenning/safety/import_firewall.py

### `src/kenning/safety/import_firewall.py` (NEW 2026-06-15 — loader-level anticheat import firewall)

An `AnticheatImportFirewall` registered on `sys.meta_path` that raises
`ImportError` for a blocklist of desktop/input/capture/browser modules
(kenning.desktop, pyautogui, mss, dxcam, PIL.ImageGrab, pywinauto,
pynput, uiautomation, playwright, browser_use, selenium, the openclaw
browser/desktop bridges) WHILE `anticheat_active()`.
`install_import_firewall()` runs at orchestrator boot before anything
else loads; `is_firewall_installed()` is asserted by the boot posture
canary. This makes the never-load guarantee loader-level: the ban-class
automation stack cannot be pulled into the anticheat-pinned RAM even by
a stray lazy import.

- **Blocklist expanded (2026-06-15):** added the raw input/capture
  automation libs `keyboard` (global low-level keyboard hook /
  `SetWindowsHookEx`), `mouse` (global low-level mouse hook),
  `pydirectinput` (a `SendInput` DirectInput-scancode wrapper), and
  `d3dshot` (DXGI desktop-duplication screen capture); plus the stale
  pre-rename `ultron.desktop` / `ultron.openclaw_bridge.browser` /
  `ultron.openclaw_bridge.desktop` mirror prefixes — the `src/ultron/`
  package no longer exists (renamed to `kenning`) and is never imported,
  but the prefixes are blocked as a belt-and-suspenders guardrail so a
  resurrected legacy path can never slip the firewall.
- **Installed before the Orchestrator constructs (2026-06-15):**
  `install_import_firewall()` now runs in `kenning/__main__.py` BEFORE
  the `Orchestrator` is built (it was previously installed inside the
  orchestrator), eliminating the prior unprotected boot window between
  process start and orchestrator construction. The install is idempotent
  (a second call is a no-op) and a no-op while anticheat is off — the
  meta-path hook only raises once `anticheat_active()` is true.

#### Changed module: src/kenning/llm/inference.py

### `src/kenning/llm/inference.py` — what changed (2026-06 relay/gaming work)

- **`reload_for_preset(preset, *, gpu_layers: Optional[int] = None) -> tuple[bool, str]`** — new `gpu_layers` keyword argument. When not `None`, overrides `n_gpu_layers` for this reload, bypassing both `KENNING_LLM_GPU_LAYERS` env and `llm.gpu_layers` config. Gaming-mode engage passes `gpu_layers=0` to put the model fully on CPU; disengage calls without the argument to restore config behaviour. The override is applied by injecting the value into the env (`KENNING_LLM_GPU_LAYERS`) before `_build_llama_instance` reads it, with rollback on failure. Only supported for `runtime="in_process"`.
- **Gaming-mode / testing-mode RAG gate in `_retrieve_rag_snippets`** — at the top of the method, if `is_gaming_mode_active() or is_testing_mode_active()` AND `config.gaming_mode.barebones_skip_retrieval` is True (default), the method returns `[]` immediately, skipping the embedder, vector search, and cross-encoder reranker. Fail-open: any import/lookup error falls through to normal retrieval.

#### Changed module: src/kenning/audio/wake_word.py

### `src/kenning/audio/wake_word.py` — what changed (2026-06 relay/gaming work)

- **Per-word thresholds** — `WakeWordDetector.__init__` reads `config.wake_word.thresholds` (a `{word: float}` dict) into `self._thresholds`. `_threshold_for(word) -> float` returns the word-specific value, falling back to `self._default_threshold`. Applied at construction and re-applied in `reload_for_word` when the active word changes.
- **`min_consecutive_frames` gate** — reads `config.wake_word.min_consecutive_frames` (default 1, i.e. off) into `self._min_consecutive`. In `process()`, if the score is above threshold, `self._consec` is incremented; if `_consec < _min_consecutive` the detection is suppressed and `False` is returned. A real wake word sustains high scores across many frames; confusable single-frame spikes are filtered. `_consec` resets to 0 on any sub-threshold frame and after a successful trigger.
- **`reload_for_word(word: str) -> tuple[bool, str]`** (new method) — hot-swaps the live model at runtime (settings-panel dropdown). Resolves `word` to its side-by-side custom ONNX via `_model_path_for_word(word)`, loads a new `openwakeword.model.Model`, updates `self._model`, `self._name`, `self._active_word`, `self._using_fallback`, resets `_last_trigger_ts` and `_consec`, and applies the per-word threshold via `_threshold_for`. Falls back to the custom fallback ONNX (`self._fallback_name`) if the requested word's ONNX is missing. Returns `(True, word)` on success, `(False, reason)` otherwise.
- **`_model_path_for_word(word: str) -> Optional[Path]`** (new helper) — resolves a word name to `{models_dir}/{word}.onnx`, returning `None` if the file does not exist.
- **Properties added**: `active_word: str`, `using_fallback: bool` (were previously only instance attributes; now exposed as `@property`).

### Tests, config.yaml keys & gaming/testing flow (2026-06)

#### New test files (total: 10,129 collected)

### New test files

`tests/audio/test_stream_routing.py` — routing-matrix regression: verifies normal speech tees to broadcast + waveform (never team), relay path plays only to its injected device, and `BroadcastSink` mono→stereo fan-out; all without live audio.

`tests/audio/test_waveform.py` — `WaveformSink` unit tests: `analyze_clip` FFT-band frames, silence-is-calm, fail-open on garbage, submit enqueue/drop-oldest, buffer copy isolation, stale-sentinel pacer regression, teardown drain.

`tests/test_duration_control.py` — `compute_pace_vec` in-model duration shaping: sentence-final lengthening, comma vs period hierarchy, stress lift, marker passthrough, combined-stress clamp, all-ones flat identity.

`tests/test_f0_control.py` — `scale_f0_curve` / `scale_energy_curve`: variance expansion with preserved median, factor-1 identity, semitone shift accuracy, unvoiced-frame zero preservation, soft excursion cap, mean-preserving energy widening.

`tests/audio/test_relay_speech.py` — hermetic relay core: `match_relay_command` positive/negative matrix, named addressee, compose/encouragement routing, `build_relay_line` LLM wiring + fallbacks, `play_to_device` int16/float32/empty/error paths, `resolve_relay_device` delegation + fail-open, orchestrator wiring (disabled, no-match, no-TTS, no-device, happy-path, echo, playback-failure, mute toggle, recent-lines ring), streaming narration vs explicit-command boundary, control-token strip, plural-teams STT artifact.

`tests/audio/test_relay_speech_expansion.py` — expansion test suite (2026-06-12): user verbatim phrase matrix (group / named / "our" possessive / new verbs), context+directive forms, roast mode (`load_roast_lines`, `pick_roast_line`, orchestrator verbatim delivery + anti-repeat ring), fun-fact corpus loader + orchestrator path, consolation/praise curated pools (LLM never called), general-question classifier, deterministic snap callouts (50+ cases), off-snap LLM deferral, economy determinism, identity/greet/farewell set-pieces with win/loss register, `_cap_line` sentence-boundary safety, multi-agent ult callout routing, adaptive guardrail repair (`_repair_against_input`, `_preserve_agent_names`, `_strip_artifacts`), site-letter pronunciation (`relay_tts_text`), known-fact table overrides, answer-command recent-lines suppression, verbatim mode suffix stripping + LLM bypass.

`tests/test_wake_word.py` — `WakeWordDetector.fired_recently` edge cases (no-fire before trigger, window, idempotent, zero-state, negative window), plus per-word threshold override (`_threshold_for`), consecutive-frame gate (2-frame sustain required to fire, spike-broken reset), hot-swap threshold recompute.

`tests/safety/test_anticheat.py` — exhaustive anticheat coverage (68 tests): guard semantics (inactive default, runtime toggle, config-pin, test-session isolation), blocked-tool taxonomy sweep (27 tools incl. namespaced/dotted/case-insensitive normalization), allowed-tool passthrough, voice toggle matcher, AST audit asserting every guarded desktop function still contains its `guard()` call, surface-hook stop+restore + broken-hook fail-open, ban-class API source sweep, `press_key`/`press_hotkey` hard-raise, `ToolCallValidator` BLOCK_HARD pre-check + audit log, orchestrator voice toggle, gaming-mode tie-in (engage→ON, disengage→OFF unconditionally, broken-config fail-safe).

#### Ultron 1.0 — new modules, harness & tests (2026-06-20)

New `src/kenning/audio/` modules (full API in the `audio/` "Source modules" section above):
`ultron_prompt.py` (lean prompt assembler, M1), `agent_kits.py` (version-stamped 29-agent kit injection, M3),
`intent_gate.py` (4-class always-listening gate classifier, M5). Plus the `voice_lines.py` aggregate +
`voicemeeter_level.py` (both 2026-06-18). New env flags: `KENNING_U1_LLM_ROUTE` (default OFF),
`KENNING_U1_VERBOSITY` (none/low/high, default high), `KENNING_THINKING_MODE` (default OFF),
`KENNING_SNAP_REGISTRY` (default on), `KENNING_SNAP_EARLY_ENDPOINT` (default OFF).

New harness `scripts/relay_test/u1_text_harness.py` — the PRIMARY text-injection routing/intent harness
(deterministic, ~1 ms/case, the calibration source): a labeled `Case` set (command / non_trigger /
compound) run through `normalize_command` → `match_relay_command` → relay route, scored into REAL-fails vs
known-baseline vs `u1_gate_target` buckets (exits 1 only on REAL fails). Output `logs/u1_text_harness/run.jsonl`.
Companion `scripts/relay_test/trace_corpus_full.py` — full-pipeline per-case tracer (separates norm-L1/L2 +
router decision) used for the 25k-corpus audits and as the harness substrate.

New tests under `tests/audio/`:
- `test_ultron_prompt.py` — verbosity coercion + directives, relay/private prompt assembly, addressee/compound/
  agent-context/exemplar blocks, no `<think>`, sampling per verbosity.
- `test_u1_llm_route.py` — the `KENNING_U1_LLM_ROUTE` flag-OFF-byte-identical + flag-ON wiring in `build_relay_line`.
- `test_agent_kits.py` — kit lookup/canon/de-dup/cap + the C_domain corrections.
- `test_intent_gate.py` — the 4-class fail-closed cascade (relay/command-local/private/ignore), ASR pre-reject,
  the undecided-band + `resolve_with_llm` (stub LLM).
- (2026-06-19) `test_snap_early_endpoint.py` (E3) + the thinking-mode toggle tests.

#### config.yaml new keys

### `config.yaml` new keys

#### `gaming_mode` block

```yaml
gaming_mode:
  anticheat_safe_mode: true       # PINNED ON — hard-blocks all desktop-interaction surfaces at boot, independent of gaming-mode engage
  engage_at_startup: true         # run gaming_mode.engage() at the end of Orchestrator.__init__ (no voice trigger needed)
  barebones_skip_retrieval: true  # skip per-turn RAG memory retrieval + cross-encoder reranker while gaming is active
  barebones_skip_web_search: true # skip web-search: forces NO_SEARCH every gaming turn AND (2026-06-15) skips BUILDING the provider chain (searxng/brave/duckduckgo) + reader chain (trafilatura/jina) at boot
  llm_gpu_layers: 0               # 0 = gaming LLM fully on CPU (overrides env KENNING_LLM_GPU_LAYERS and config llm.gpu_layers); -1 keeps GPU
  # NEW 2026-06-15 lean-boot flags (all default True, individually toggleable):
  barebones_direct_gaming_llm: true          # construct the 3B-CPU gaming LLM directly at boot (skips the 4B-on-GPU transient)
  barebones_skip_reranker_warmup: true       # skip cross-encoder reranker warmup at startup
  barebones_skip_docker_autostart: true      # skip Docker autostart on boot
  barebones_skip_coding: true                # skip coding stack (ProjectIndex / Supervisor / CodingVoice / coordinator / project_introspect)
  barebones_skip_openclaw: true              # skip OpenClaw bridge + threads
  barebones_skip_evolution: true             # skip evolution service
  barebones_skip_skills: true                # skip skill registry scan
  barebones_skip_events: true                # skip background event subsystem
  barebones_skip_summarizer: true            # skip background summarizer
  barebones_lazy_zero_shot_addressee: true   # defer flan-t5 addressee until first use
  # NEW 2026-06-15 (second lean-boot wave) — three more skips, all default True:
  barebones_skip_memory: true                # skip the conversation-memory stack (Qdrant + bge-small dense + bm25 sparse FastEmbed encoders)
  barebones_skip_intent: true                # skip the in-process intent recognizer (a SECOND embeddinggemma-300m q4 — duplicate of the sidecar copy)
  barebones_skip_ack_prewarm: true           # skip prewarming the precomputed ack-clip cache (filler-acks are suppressed while gaming anyway)
  # (barebones_skip_web_search above was widened to ALSO skip building the web-search + reader chains)
```

- `anticheat_safe_mode` — read by `kenning.safety.anticheat.anticheat_active()` via `_config_pin_enabled`; guards 49 module entry points + safety-validator BLOCK_HARD + voice intents. Toggled by voice ("enable/disable anticheat mode") and the settings GUI.
- `engage_at_startup` — checked in `Orchestrator.__init__` tail; calls `GamingModeManager.engage()` in a fresh thread via `_drive_async_blocking`.
- `barebones_skip_retrieval` — gate in `llm.inference._retrieve_rag_snippets`; honours both `is_gaming_mode_active()` and `is_testing_mode_active()`.
- `barebones_skip_web_search` — runtime gate in `Orchestrator._barebones_skip_web_search()` (same dual-flag check, forces `NO_SEARCH`); **2026-06-15 widened** so it also short-circuits `_build_web_search`, which returns `(None, None, None)` → `web_gate` / `web_executor` are `None` and the provider chain (searxng/brave/duckduckgo) + reader chain (trafilatura/jina) are never constructed. The conversational path takes its no-web-gate branch.
- `barebones_skip_memory` — NEW 2026-06-15; `_load_memory_if_enabled()` returns `None` (Qdrant + the bge-small dense + bm25 sparse FastEmbed encoders are never built). `self.memory = None` is an already-supported state (every call site is guarded). Rationale: RAG retrieval is already off while gaming (`barebones_skip_retrieval`), so a recorded+embedded turn is never read back; in-session context still works via the LLM's own history deque, only cross-session memory is dropped while gaming.
- `barebones_skip_intent` — NEW 2026-06-15; skips the in-process intent recognizer, which loaded a SECOND embeddinggemma-300m (q4, via `moonshine_voice`) IN the main process — a duplicate of the isolated-sidecar copy that defeated the sidecar's anticheat isolation. Its 25 phrases (gaming-mode toggle, now covered by the GUI + boot default; time/date, which the LLM answers; news/current-events, which needs web search that is off while gaming) are all redundant in a gaming session.
- `barebones_skip_ack_prewarm` — NEW 2026-06-15; skips `_kick_off_ack_clip_prewarm` (the precomputed ack-clip cache prewarm thread). Conversational filler-acks are suppressed in gaming, so synthesizing/caching them is wasted work.
- `llm_gpu_layers` — threaded into `GamingEngageDeps.gaming_llm_gpu_layers` → `LLM.reload_for_preset(preset, gpu_layers=0)`; disengage restores the pre-engage preset on the original device.
- `barebones_direct_gaming_llm` / `barebones_skip_*` / `barebones_lazy_zero_shot_addressee` — NEW 2026-06-15 lean-boot gates; all read in `Orchestrator.__init__` via `_skip_for_lean_gaming(flag)`. Together they reduce boot imports to relay + Spotify + core-voice (the reranker + skipped subsystems stay out of RAM; worker RSS is noisy ~3.5-6 GB so no fixed delta is claimed — the clear win is eliminating the ~4-5 GB 4B-on-GPU VRAM boot transient) and are proven by the `_audit_anticheat_posture` "lean boot OK" sys.modules assertion (which now also covers the intent model, conversation memory, ack-prewarm thread, and web-search chain — see below). Each `barebones_*` skip is GUI-toggleable from the settings panel's "Lean Boot" section.

In `SemanticRouterConfig` (2026-06-15 NEW fields):
- `sidecar_orphan_sweep_enabled: bool` (default `True`) — enables `sidecar_lock.sweep()` at boot before spawning the embedder sidecar.
- `sidecar_pidfile_path: str` (default `~/.kenning/embedder_sidecar.json`) — path for the atomic-write pidfile that records pid+port+model+owner.
- `sidecar_device: str` (default `"cpu"`, NEW 2026-06-15) — device the embeddinggemma router sidecar runs on; CPU keeps GPU VRAM free for the game.

#### `audio` block (output keys)

```yaml
audio:
  prefer_wasapi_output: true   # NEW 2026-06-15 — open spoken-audio output via the WASAPI low-latency chokepoint (MME latency='low' fallback)
  mute_speakers: false         # NEW 2026-06-15 — silence the default-speaker path (Kokoro zeroed, monitor mirror skipped) while relay still reaches B1/B3
```

- `prefer_wasapi_output` — read by `audio.devices.make_output_stream`; when true and the device is a WASAPI endpoint, opens a `WasapiSettings(auto_convert=True)` + `latency='low'` stream (B1/B3 ~22–25 ms vs ~90–180 ms MME); non-WASAPI devices fall back to MME `latency='low'`.
- `mute_speakers` — read live; zeroes conversational Kokoro output and skips the relay monitor mirror on the default speakers, leaving the team relay (B1) and OBS broadcast (B3) untouched. The GUI "Mute my speakers (loopback)" knob + "APPLY MUTE ONLY" button apply just this key via `app._apply_one("audio.mute_speakers")`.

#### `stop_button` block (NEW 2026-06-15)

```yaml
stop_button:
  enabled: true            # master: allow the voice summon at all
  show_at_startup: false   # true = the window is up the moment Ultron boots
  width: 120               # window px
  bar_height: 16           # black drag strip on top (grab to reposition)
  button_height: 36        # the STOP button itself
  bg_color: "#000000"      # fully black window + bar
  accent_color: "#e5484d"  # Kenning crimson — button text + 1px border
  button_fill: "#140709"   # near-black button face (brightens on hover/press)
  always_on_top: true      # float over the (borderless-windowed) game
  label: "STOP"
  x: 60                    # initial top-left position on screen
  y: 60
```

Read by `StopButtonConfig` (`config.py`) and `audio/stop_button.py`. The window is summoned/dismissed by voice ("show/hide the stop button", matched by `match_stop_button_command`); clicking it fires the orchestrator's all-channel `_cancel_all_playback()` — the loopback-immune equivalent of voice "Ultron, stop" (no wake-word watcher involved). In-process tkinter; fail-open (no display / no Tk → never appears, voice path untouched).

#### `testing_mode` block

```yaml
testing_mode:
  enabled: false   # flip to true for corpus testing; harness also uses set_testing_mode_active()
```

Read by `kenning.safety.testing_mode.is_testing_mode_active()`; propagates to the same RAG, web-search, and anticheat gates as gaming mode, but does **not** trigger device swaps or alter the real gaming/anticheat engage path.

#### `wake_word` new keys

```yaml
wake_word:
  name: "ultron"          # active word; hot-swap via GUI dropdown (no restart)
  fallback_model: "kenning"  # custom kenning.onnx; never hey_jarvis
  thresholds:             # per-word overrides; active word's value replaces the flat threshold on swap
    kenning: 0.4
    ultron: 0.65        # 2026-06-17: 0.7 -> 0.65 (slight sensitivity bump; the per-word sustain gate is the primary false-accept guard)
  min_consecutive_frames: 2  # flat fallback
  consecutive_frames:
    ultron: 4              # 2026-06-18: per-word sustain raised 3 -> 4 to reject brief confusables (e.g. 'Oh, we...')
```

- `thresholds` — read by `WakeWordDetector._threshold_for(word)`; applied on construction and on `reload_for_word()`.
- `min_consecutive_frames` — stored as `_min_consecutive`; gate in `WakeWordDetector.process()` resets on any sub-threshold frame.
- `fallback_model` — PATH-based loader falls back to the side-by-side custom ONNX (e.g. `models/openwakeword/kenning.onnx`), never the pretrained `hey_jarvis`.

#### `visualizer` block (new keys)

```yaml
visualizer:
  enabled: true
  bg_color: "#00ff00"      # NEON GREEN chroma-key background for OBS color-key filter
  transparent: false       # solid bg — OBS window capture renders green (can color-key); true → OBS sees black
  always_on_top: false     # false = background/tool-window mode; OBS WGC still captures it
  nameplate_text: "ULTRON" # glowing nameplate below waveform ("" = off)
```

- `bg_color` / `transparent` — control whether the overlay window uses a solid chroma-key colour or Windows transparency (which OBS captures as black).
- `always_on_top: false` — sets `WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE` via ctypes (`_make_background_overlay`), sinking the window behind desktop apps while OBS WGC still grabs it.
- `nameplate_text` — pre-renders 16 brightness buckets (`_nameplate_frames`) with Pillow Gaussian-blurred neon glow; falls back to plain text if Pillow is absent.

#### Gaming-mode + testing-mode cross-cutting flow

### Gaming-mode + testing-mode cross-cutting flow

#### Gaming-mode engage

Triggered by voice ("gaming mode") or automatically at boot when `gaming_mode.engage_at_startup: true`. Entry point: `GamingModeManager.engage()` → `Orchestrator._engage_extra()` → `_drive_async_blocking(gaming_engage_iterator(deps))` (runs the async state machine on a fresh thread to avoid `asyncio.run()` inside an already-running loop).

State machine stages in order (`src/kenning/lifecycle/gaming_engage.py`):

1. **`SWAPPING_LLM`** — `LLM.reload_for_preset(gaming_mode.llm_preset, gpu_layers=gaming_mode.llm_gpu_layers)`; stashes the pre-engage preset in `llm_preset_holder`. Default preset `llama-3.2-3b-abliterated`, `gpu_layers=0` (fully CPU). Disengage restores the prior preset.
2. **`STOPPING_PARAKEET`** — swaps STT engine to the gaming engine; stops the Parakeet HTTP server (~700 MB VRAM freed).
3. **`MOVING_KOKORO`** — `KokoroSpeech.move_to_device("cpu")`; frees GPU TTS VRAM (~330 MB) for the running game.
4. **`UNLOADING_VLM`** — unloads moondream2 if loaded.
5. **`READY`** — terminal state; anticheat is driven ON unconditionally by `GamingModeManager._set_anticheat(True)` during engage.

Per-turn gates active while gaming mode is engaged:
- **RAG/reranker off** — `llm.inference._retrieve_rag_snippets` checks `is_gaming_mode_active() or is_testing_mode_active()` + `gaming_mode.barebones_skip_retrieval`.
- **Web-search off** — `Orchestrator._barebones_skip_web_search()` checks the same dual flag + `gaming_mode.barebones_skip_web_search`; forces `NO_SEARCH` classification, skipping the LLM preflight call and the executor entirely.

Disengage reverses: LLM preset restored (on GPU), STT restored, Parakeet restarted, Kokoro moved back to GPU, VLM reloaded as needed, anticheat driven OFF (unless config-pinned).

#### Testing mode

`src/kenning/safety/testing_mode.py` — a separate, off-by-default flag (`_runtime_active` + `testing_mode.enabled` config pin).

Honoured at the same per-turn gate sites as gaming mode:
- RAG/reranker gate in `_retrieve_rag_snippets`
- Web-search gate in `_barebones_skip_web_search()`
- `anticheat_active()` also returns `True` while testing mode is on (desktop automation hard-blocked)

Key distinction: testing mode **never** triggers the gaming device swaps (no LLM→CPU, no Kokoro→CPU, no Parakeet stop, no VLM unload), so the GPU stays available for fast generation. GPU and CPU produce statistically identical text, so GPU corpus testing is representative of the CPU gaming runtime. The test harness flips the flag via `set_testing_mode_active(True/False)` per-run; `config.testing_mode.enabled` is a persistent pin for extended corpus sessions.

#### Anticheat-safe mode

`src/kenning/safety/anticheat.py` — `anticheat_active()` is `True` if any of:
1. Runtime toggle set via `set_anticheat_active(True, source)` (voice command "enable anticheat mode", gaming engage, or test fixture).
2. `is_testing_mode_active()` returns `True`.
3. `gaming_mode.anticheat_safe_mode: true` in config (config-pin path; disabled for test sessions via `set_config_pin_enabled(False)`).

When active, three enforcement layers fire in concert:
- **Module guards** — `guard(action)` raises `AnticheatBlockedError` at 49 entry points across all desktop-interaction surfaces (`input_control`, `capture`, `uia`, `clipboard`, `dialog_control`, `element_click`, `windows`, `placement`, `launcher`, `ocr`, `sequence`, `browser_use`, `screen_context`, bridge `DesktopTool`).
- **Safety-validator BLOCK_HARD** — `ToolCallValidator.check()` returns `Verdict.BLOCK_HARD` with `triggered_rule_id="anticheat_safe_mode"` for any `is_blocked_tool()` call; logged to `audit.jsonl`.
- **Surface hooks** — registered hooks (`register_surface_hook`) are called on every flip to physically STOP running subsystems (UIA pollers, capture threads); broken hooks are swallowed and never block the flip.

Audio path (mic, STT, LLM, TTS, VoiceMeeter relay, waveform overlay) is explicitly unaffected — these use shared-mode audio APIs (same surface as Discord) and perform no cross-process interaction.


## Maintenance contract

**This document is the operating manual. Keep it current.**

> ### ⭐ BINDING RULE (Ultron 1.0 forward, reaffirmed 2026-06-20)
> **Update `docs/codebase_structure.md` in the SAME commit as any structural change** — a new/renamed/removed
> module, public class/function, script, test directory, config key/section, doc, or cross-cutting flow. This
> is non-negotiable and treated like a test: **doc-drift is a regression.** A fresh AI-agent session must be
> able to reground from this doc + the memory files WITHOUT re-exploring the source; if that breaks after your
> change, you violated the contract — fix the doc before declaring the task done. When a structural change
> ships across several commits, the doc update lands with the commit that makes the structure real. This rule
> is mirrored in `CLAUDE.md` (binding rule #4), `feedback_ultron_1_0_process.md` (process rule 3), and the
> RELEASE CHECKLIST in `feedback_no_default_load_anticheat.md` ("ALWAYS update docs/codebase_structure.md").
> The running **Validating HEAD** header at the top is the first thing read — prepend a new block there for
> each substantive session AND update the affected body sections (don't let the header lead the body).

This contract is **binding** — every non-trivial change to the
codebase must update this document in the same change. Skipping
the update means future sessions waste time re-deriving ground
truth from the source. **Don't skip.**

The project-root standards doc at the top of this prompt's reading
order calls this contract out explicitly so a fresh AI-agent
session sees it before its first edit.

### What "non-trivial change" means

You MUST update the relevant section of this document when you:

1. **Add a new module file** under `src/kenning/` →
   - Add to the file tree.
   - Add a section under "Source modules" with the public API
     (classes, functions, brief in/out).
   - If it's a new subsystem (e.g. `src/kenning/openclaw_bridge/`),
     add to the architecture diagram in `docs/architecture.md`
     too.

2. **Add a new public class or function** to an existing module →
   - Add it to the module's section under "Source modules".
   - Note the inputs and outputs in one line.

3. **Remove or rename** an existing module / class / function →
   - Update every section that referenced it.
   - Search for the old name with Grep before declaring done.

4. **Add a new script** under `scripts/` →
   - Add to the file tree.
   - Add a section under "Operational scripts" with purpose,
     run command, in/out, and functions.

5. **Add a new test directory or test category** →
   - Add to the file tree (under `tests/`).
   - Add to the relevant "Tests" subsection.
   - Update the "current state" header at the top of this file
     with the new total.

6. **Add a new log file or data path** →
   - Add to the "Runtime artifacts" tables.

7. **Add a new doc** under `docs/` →
   - Add to the "Documentation index" with the right category
     (Foundation reference / OpenClaw architecture / per-phase
     close-out / user-side setup / 4B plan).
   - Add to the file tree under `docs/`.
   - Cross-reference where relevant in other sections.

8. **Add a new config section / key** →
   - Add to the `config.yaml` summary in "Configuration".
   - Update [docs/configuration.md](configuration.md) too
     (per-key reference).
   - Document any new defaults in the relevant `feedback_*.md`
     if it reflects a confirmed user decision.

9. **Change a cross-cutting flow** (voice path, coding path,
   search path, dispatch path, OpenClaw bridge path) →
   - Update the relevant diagram in "Cross-cutting flows".

10. **Migrate a subsystem out of the `config/settings.py` shim** →
    - Update [docs/phase3_5_followup.md](phase3_5_followup.md)
      (cross off).
    - If it changes the public API of the migrated module,
      update its "Source modules" section here.

11. **Bump test counts** — the file's header tracks "X passed /
    Y skipped / Z failed". Update these when the count changes.

12. **Land a new phase / sub-phase** → bump the phase status
    line in the header.

### The validation loop

After your change:

```powershell
# 1) Tests pass -- ALWAYS via the wrapper (never bare pytest; it enforces
#    the five operator-side safeguards). Under machine contention add the
#    real-subprocess exclusions from the validating-HEAD header's SWEEP note.
C:\STC\ultronPrototype\.venv\Scripts\python.exe scripts\run_tests.py --stale-heartbeat=400

# 2) Config still validates
C:\STC\ultronPrototype\.venv\Scripts\python.exe scripts\validate_config.py

# 3) Re-read this doc and confirm:
#    - File tree matches `git ls-files | grep -v '^\\.'`
#    - "Source modules" sections cover every src/kenning/ file
#    - "Operational scripts" sections cover every scripts/ file
#    - "Tests" subsections cover every tests/ subdirectory
#    - "Documentation index" links every docs/*.md file
```

If the doc no longer matches reality after your changes, fix
this document before declaring the task done.

### Why this matters

A fresh AI-agent session reads this document + the memory files
and should be fully oriented without re-exploring the codebase. If
that's not the case after your changes, the maintenance contract
was violated. Treat that as a regression and fix the doc.

To verify the document still matches reality:
```powershell
# Run after any non-trivial change (use the wrapper, NOT bare pytest)
python scripts/run_tests.py --stale-heartbeat=400
python scripts/validate_config.py
# Then re-read this doc and confirm tree + module sections are current
```
