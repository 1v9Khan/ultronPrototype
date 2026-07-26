# Snap carve-out — RESOLVED hello-only (2026-06-27)

**✅ RESOLVED 2026-06-27.** The user scoped this all the way down to ONE deterministic call:
"say hello" → **"Hello."** (it had been LLM-improvising "team online"). `_is_carveout_snap` was
NARROWED to `return getattr(command, "directive", None) == "hello"` — the broad tactical/payload/
compound recognizer (the unsolvable blocker described below) was DELETED, never built, ABANDONED —
and `KENNING_U1_SNAP_CARVEOUT` default flipped `"0"`→`"1"` (ON). Now "say hello"→"Hello." /
"say hello to <agent>"→"Hello, <Agent>." with NO LLM (0 generate_fn calls); EVERYTHING else
(tactical / ask-form / morale / strings / conversation) stays on the LLM. Live reboot 16; committed
`af6b7a0`, origin/main `e387a36`; golden re-blessed (435 syms); `test_snap_carveout.py` +
`test_u1_llm_route.py` re-spec'd to hello-only (343 tests green). **Everything below is the ORIGINAL
paused plan, kept for history — the general tactical discriminator was never solved and is not needed.**

---

**Status (original, superseded): PAUSED at user request. Infrastructure landed + green (flag default OFF, inert).
Blocked on the discriminator. Resume from "The blocker" + "Resume plan" below.**

## Goal (user spec, verbatim intent)
Re-enable deterministic pool generation **ONLY for very short SINGLE snap callouts**, under
route-all. Everything else stays on the LLM. The user was emphatic.

- **Deterministic (OK'd):** `tell my team to rush B` · `tell my team I am lurking` ·
  `tell my team I am flanking` · `sova hit 85` (NOT as part of a string) · `one backsite` ·
  `tell my team im rotating` · `hello` → rendered as just **"Hello."** (not "Hello team.").
- **MUST stay on the LLM:** strung-together callouts; `ask iso to drop me his sheriff`;
  `ask sage to heal me`; `ask sage if she has a heal`; `ask my team why they arent smoking`;
  `jett is flaming you`; `sage called you a soundboard`; `reyna asked if you are a voice
  changer`; any conversational line; morale (`I got this` / `nice try` / `lock in`).
- **Additive, reversible:** never alter the pipeline so we can't revert to the full snap pool.
- Flavor tails stay OFF by default.
- A **stop-button toggle** between hybrid (snaps back) and full-LLM (everything→LLM).

## What LANDED (in the working tree, UNCOMMITTED, green with flag OFF)
- `src/kenning/audio/relay_speech.py`
  - Flag `_u1_snap_carveout_enabled` + `set_snap_carveout_enabled()` / `snap_carveout_enabled()`
    (env `KENNING_U1_SNAP_CARVEOUT`, **default "0" / OFF — WIP**).
  - `_is_carveout_snap(command)` — the discriminator (**TOO BROAD — see blocker**).
  - In `build_relay_line`: when the carve-out qualifies, flips `_u1_route = False` **and**
    `rephrase = False` for that one command, so it reuses the EXISTING deterministic path
    (curated snap / literal relay, no LLM). Purely additive; the full snap pool is untouched.
  - Hardcoded hello render → **"Hello."** (was "Hello team."); registry-render condition
    reverted to original (registry now drives the hello render).
- `src/kenning/audio/voice_lines.py` — `TARGET_SNAP_REGISTRY` hello `team_lines`
  `("Hello team.",)` → `("Hello.",)`. Named target unchanged (`"Hello, {name}."`).
- `tests/data/voice_lines_golden_digest.json` — re-blessed (429 symbols; check OK). It also
  pins the flag's default value, so **re-bless again when the default flips to ON**.
- `tests/test_snap_carveout.py` — NEW, 24 tests: discriminator accept/reject on clean cases,
  end-to-end routing (deterministic vs LLM via generate_fn call-count), no-op for non-tactical,
  additive guarantee (route-all OFF stays fully deterministic regardless of the flag).
  ⚠️ These pass because they use CLEAN hand-built commands; they do NOT yet cover the
  real-parse leak cases in the table below. Add those when resuming.
- `tests/audio/test_relay_speech.py` (3 asserts) + `tests/audio/test_corpus_audit_fixes.py`
  (1 assert) — hello expectations updated to "Hello.".
- (earlier same turn, separate) `src/kenning/llm/draft_model.py` — draft KV q4_0 → q8_0 revert
  (kept the n_batch/n_ubatch compute-buffer cap). Per user: VRAM is workable, give back the
  ~1-2 ms accept-rate question mark.

## The blocker — the discriminator can't cleanly separate the two sets
`_is_carveout_snap` keys on compose/context/verbatim/question/compound/word-count, which
handles the clean cases but **leaks** under real parses. No existing classifier separates the
user's intent (it's semantic — tactical vs social/morale/request). Empirical table
(`match_relay_command` → `relay_route_info` + `_is_morale_payload`):

| utterance | route | payload | morale? | DESIRED |
|---|---|---|---|---|
| tell my team to rush B | relay_llm | rush B | False | **DET** |
| tell my team I am lurking | curated_command | I am lurking | False | **DET** |
| tell my team I am flanking | snap | I am flanking | False | **DET** |
| tell my team sova hit 85 | snap | sova hit 85 | False | **DET** |
| tell my team one back site | snap | one back site | False | **DET** |
| tell my team im rotating | snap | im rotating | False | **DET** |
| tell my team I got this | relay_llm | I got this | False | LLM |
| tell my team nice try | relay_llm | nice try | False | LLM |
| tell my team lock in | **snap** | lock in | False | LLM |
| tell my team good job | curated_command | good job | False | LLM |
| ask sage to heal me | **snap** | heal me | False | LLM |
| ask iso to drop me his sheriff | **snap** | drop me his sheriff | False | LLM |
| ask my team why they arent smoking | **snap** | why they arent smoking | False | LLM |

Key facts:
- `route ∈ {snap, relay_llm, curated_command}` each contain BOTH desired-DET and desired-LLM
  rows → the native route is NOT a usable discriminator.
- **ask-forms** are the worst leak: the parser strips them to a tactical-looking payload
  (`heal me`, `drop me his sheriff`) so the snap matcher (`_as_snap_callout`) claims them.
- `_is_morale_payload()` returns **False for every row** (including the morale ones) → not a
  usable morale filter as-is.
- Note `relay_route_info` has no `hello` route (handled by the registry, not modelled there) —
  so hello must be keyed on `directive == "hello"`, not the route.

## Resume plan
1. **Build a purpose-built tactical-callout recognizer** for the carve-out (don't reuse the
   route). Allowlist the tactical KINDS the user named: movement/exec (rush/push/rotate/flank/
   lurk/fall back/regroup/hold/default/split/take/hit-site/pinch/collapse), position/location
   (site A/B/C + long/short/mid/heaven/market/backsite/main/… with counts one/two/three),
   enemy info (agent + hit/<number>/location/spike/planted), self-status (I'm + lurking/
   flanking/rotating/pushing/holding/peeking).
2. **Hard exclusions (→ LLM), checked first:** ask-form (`raw_text` matches `^\s*ask\b`),
   morale/social lexicon (`I got this`, `nice try`, `nice shot`, `good job`, `well played`,
   `gg`, `lock in`, `let's go`, `clutch`, …), reported/`context` set, identity, questions,
   **strung-together callouts (count > 1 — see the box below)**, >~6 words,
   compose/verbatim/roast/fun_fact.

> ### String detection — COUNT commands, do NOT match conjunctions (user direction 2026-06-24)
> Detect a strung-together input by **how many distinct commands/callouts it contains** — if
> more than one, it's a string → LLM. Do **not** rely on conjunctions/commas ("and"/"then"/","),
> because the user won't always use them and may rattle off callouts fast and bare, e.g.
> **`sova hit 84, cypher heaven, sage backsite`** = three callouts (and the commas may not even
> be transcribed). The robust test is: run the tactical recognizer / callout segmenter over the
> input and count matches; `count > 1` → string → LLM; `count == 1` → eligible single snap.
> Constraint: this must add **no latency** (segment with the same cheap matcher pass, no extra
> model call). This SUPERSEDES the current `_is_carveout_snap` conjunction/comma heuristic.
3. **Validate against the table above** — every row must land on its DESIRED column. Promote
   the table into `tests/test_snap_carveout.py` as real-parse positive/negative cases (replace
   the clean hand-built reject cases that give false confidence).
4. Flip `KENNING_U1_SNAP_CARVEOUT` default → "1"; **re-bless the golden** (it pins the flag
   default). Update `tests/audio/test_u1_llm_route.py` to set `set_snap_carveout_enabled(False)`
   in its route-all fixture (those tests validate the FULL-LLM = carve-out-OFF mode); 6 of its
   tests went red purely because the carve-out (when ON) correctly intercepts snaps/hello.

## Remaining sub-tasks (all paused)
- [ ] Fix the discriminator (above) + flip default ON + re-bless golden + fix test_u1_llm_route.
- [ ] **Stop-button "SNAPS" toggle** (hybrid ↔ full-LLM) → wire to `set_snap_carveout_enabled`.
      3 sites: `stop_button.py` widget (copy the PTT/TURBO button), the `StopButtonOverlay(...)`
      call + a `_set_snaps` callback in `orchestrator.py`, and a `StopButtonConfig` field in
      `config.py`.
- [ ] THEN the queued Twitch backlog (moderation-confirm GUI Font fix + integration, stop-button
      chat/redeem/games/moderation toggles, the new "say something (not team)" redeem at
      1/10th cost) + **restart Ultron** when all done.

## Separate loose end (NOT this work — surfaced, not fixed)
`tests/audio/test_u1_llm_route.py::test_config_verbosity_defaults` is RED: it asserts
`RelaySpeechConfig().callout_verbosity == "low"`, but `config.py` intentionally defaults to
`"none"` (2026-06-23 latency + flavor-off-by-default pref). Either update the test to `"none"`
or revisit the default — a VRAM/latency decision, owned by the user.

## Baseline (frozen control) failures unrelated to any of this
`test_value_swap_keeps_last_buy` + `test_say_to_delivers_literal_payload` are in the frozen
control set (`docs/ultron_1_0/05_testing/00_baseline.md`). Expected red.
