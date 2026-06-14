# Comprehensive loop — iteration L1 (seed 11) HAND analysis (mine, not agents)

Scope: fresh 20,000-case corpus (seed 11) regenerated on the post-feature build (curated
commands ×73, repeat-verbatim, contextual-enemy, LRU, greeting self-intro). Matcher run over
the full 20k; deterministic outputs (13,425 of 16,420 matched = **81.8% deterministic
coverage**) hand-examined category by category; ASR full-pipeline run (seed 11, 1,200 cases)
for the audio/ASR/LLM-route metrics.

## Matcher (full 20k)
- clean 0.9912 → **0.9954** after fixes (≥ the 99.5% target); missed 172 → **89**;
  false-relay in the dedicated NEGATIVE gate packs = **0** (hard gate holds).
- The 3 residual "false-relays" are borderline corpus-label cases OUTSIDE the gate packs
  ("drop a fun fact", "close us out, what's the plan?", "...TikTok filter, respond") — all
  defensible behaviors, none a private-speech leak.

### Matcher recall fixes (verified no gate regression)
1. **Determiner-optional groups** — "relay to team:", "tell teammates X" now match (only
   ever after an explicit relay verb, so bare nouns still never trip it). Fixed the entire
   `directive_obs_traps` "relay to team:" cluster (37 cases).
2. **Comma/colon after the trigger** — "let my team know, X" / "warn my team, X" / "tell my
   team, X"; ask-form honours the pronoun group ("ask them if X").
3. **"on my behalf" stripped** — "ask the team on my behalf if X" → "ask the team if X".
4. **Narration-guard exemption** — a LEADING explicit group/named relay command is exempted
   so "tell my squad I was gonna say X" relays, while mid-sentence "...I should tell them"
   narration stays blocked (search-anywhere first-person guard restored). Re-verified: the 4
   `false_relay_hard` cases that briefly leaked are blocked again, 0 gate leaks.

### Defensible non-matches (LEFT unmatched on purpose — false-relay protection)
- Bare imperatives with no addressee ("let the nano burn out then defuse", "let's default",
  "let him cook") — relaying these risks broadcasting the user's private thinking. Non-match
  is the correct call under the zero-tolerance false-relay rule.
- `disfluency` self-corrections ("tell the entry -- no the lurker -- lurker push") — parsing
  them deterministically risks relaying the WRONG fact; safer to defer.

## Deterministic output quality (by category, read verbatim)
- **ownership_traps**: our/their PRESERVED, **no inversions**. Found+fixed: leading "our
  <agent>" dropped the possessive ("our Viper wall" → "Viper wall"); now "Our Viper wall"
  symmetric with "Their Viper wall".
- **firstperson_traps**: **no first-person→command inversions** (18/18). First person sacred.
- **contextual_enemy** (NEW): excellent — all render as enemy callouts + in-character flavor.
  Fixed the awkward "They chose {L} poorly" template → "They gain nothing at {L}".
- **curated_commands / curated_commands2** (NEW): excellent, correctly vocative-addressed,
  scope-correct, slots filled. Fixed: the `wait_for_me` pool literally said "the user" —
  rewritten to first-person ("hold until I move").
- **compounds / clutch_dense**: fact retention near-perfect across dense multi-fact callouts.
  Fixed: (a) a HIGH confabulation — "heal me in TIME, tell her to SLOW them" triggered a
  special-relativity lecture; the GK fact table now requires a question marker. (b) leftover
  pieces joined with a bare space ran two facts together; now joined as sentences.
- **repeat_verbatim** (NEW): PERFECT — exact phrase, no flavor (soundboard check).
- **map_opinions / ult_states / agent_opinions / damage_trades / self_status /
  positions_counts**: info + numbers preserved; agent flavor apt.
- **calm_deescalate**: found+fixed enemy-contempt tails on team-morale lines ("Staying level
  is a team effort... Obsolete") — article "a" was parsed as a location and a bare count
  modified a non-place noun; both now guarded.

## Residuals (low-frequency, deferred — fixing risks a worse failure)
- Compound **over-split** on "and <ability>" lists ("...walked through our stun and slow and
  flash" → "Slow. Flash and nothing landed."). All facts PRESERVED, only fragmented;
  tightening `_NEWFACT_SUBJECT` risks DROPPING facts in legit compounds (worse).
- Disfluency/filler leaks ("Bro ...", "shut up Ultron ...") on deliberately-messy inputs.
- Minor leftover-sentence casing in the LLM-fail fallback path (LLM normally rephrases).

## Fixes committed this iteration
`0d1d329` matcher recall · `c578f2d` GK-question gate + flavor template · ownership "Our"
· `ddb764a` compound leftover join · `0acbf1b` wait_for_me first-person · `67bc6b2` morale
enemy-contempt guard. All with 359 relay unit tests green and the false-relay gate at 0.

## ASR / audio / latency metrics (scorecard_L1.json, 1,200-case ASR sample)
NOTE: this ASR run executed on intermediate code (after the matcher batch `0d1d329`, BEFORE
the ownership/GK/morale fixes). owner-retention, inversion, and hallucination therefore
predate their fixes and should improve next iteration.

| metric | result | target | status |
|---|---|---|---|
| matcher clean | 0.9955 | ≥0.995 | PASS |
| false-relay | 0/31 (0.0%) | ≤0.1% | PASS |
| deterministic coverage | 61.2% pure / 87.2% +partial | ≥50% | PASS |
| det-path latency p99 | 0.70 ms | ≤500 ms | PASS (×700) |
| LLM-path latency p50/p99 (CPU-3B) | 1.45 s / 5.37 s | ≤2.5 s p99 | p50 ok; p99 over (CPU gaming) |
| fact-retention p50/p95/mean | 1.0 / 1.0 / 0.942 | .98/.90/.90 | PASS |
| count/agent/loc/ability | .975/.972/.983/.984 | ≥.94 | PASS |
| owner | 0.843 | ≥0.98 | BELOW — run predates the "Our" fix |
| compound-zero-loss | 0.932 | ≥0.90 | PASS |
| gates (OOV/fallback/isolation) | 0/0/0 | 0 | PASS |
| ASR-coverage | 0.9959 | ≥0.85 | PASS |
| flavor TTR / max-repeat | 0.916 / 7 | ≥0.45 / <4 | TTR pass; max-repeat slightly high |
| inversion | 0.51% (5) | ≤0.5% | borderline — predates owner fix |
| hallucination | 1.64% (16) | ≤1% | BELOW — predates GK/morale fixes |
| LLM-flag-rate | 26% | ≤5% | metric ARTIFACT (see below) |
| audio blips | 9.25/1000 (9) | ≤2 | BELOW — see breakdown |

### LLM-flag-rate (26%) — metric artifact, not a real defect
The flag is token fact-retention on the LLM-route lines, which are off-snap banter / insults /
opinions / identity — content that HAS no tactical facts to retain, so it scores ~0 and
"flags". This is the known measurement gap (the real number is fluent-wrong, which needs a
calibrated judge). Deterministic + partial routes (82% of traffic) retain 96–97%.

### Audio blips (9.25/1000) — breakdown by hand
- **3/9 are the relativity confabulation** ("Time slows for anything moving near light
  speed...") — ALREADY FIXED by the GK-question gate (`c578f2d`, landed after this run).
  These disappear next iteration → ~6 blips.
- **5/9 internal_dropout** = 600–670 ms dead air at a SENTENCE BOUNDARY inside a long (7–14 s)
  verbose line (identity declaration, map opinion). Kokoro's inter-sentence pause exceeds the
  watcher's ~550 ms threshold. SHORT snap callouts (the actual callout traffic) never hit this.
  Root cause is in the TTS segment-join silence, not relay_speech — deferred to a focused TTS
  follow-up (lowering inter-segment pause / trimming segment-edge silence risks join clicks, so
  it needs its own verification pass, not a rushed change mid-relay-loop).
- **1/9 trailing_burst** (120 ms) on a short line — the residual short-clip tail class.

### Plateau-loop status
Iteration L1 establishes the post-feature baseline and landed 9 fixes. Because owner / GK /
morale / leftover fixes post-date the ASR sample, iteration L2 (fresh seed, final code) is the
real measure of improvement and the first plateau comparison point.
