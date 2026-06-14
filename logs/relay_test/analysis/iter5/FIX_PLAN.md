# Iteration 5 — Personality infusion (FIX PLAN)

**Theme:** infuse Ultron's personality (arrogance, gravitas, cold clinical menace)
into *nearly every* relay response — deterministic AND LLM — **without losing any
tactical accuracy.** Authored by hand (not agents). Reflects the 2026-06-14 design
discussion + the user's three decisions.

## Diagnosis (why personality reads as absent today, grounded in code)
1. Deterministic flavor only fires on ENEMY-facing snap branches. Named directives
   (`relay_speech.py:2339` "No flavor: short"), self/first-person (`:2360` "NO
   flavor -- the user's own"), our-team/neutral, and most economy/abstention lines
   come out BARE. The abstention literal only tails when <=9 words AND mentions
   they/their/enemy (`:2843`).
2. The LLM path is de-flavored BY THE PROMPT: `_REPHRASE_PROMPT` (`:655`) says
   "You are Kenning ... keep Kenning's calm, confident tone" and "do NOT add
   flavour". The menacing Ultron identity lives only in curated set-pieces. So all
   freeform off-snap lines (insults, banter, opinions, questions) are flavorless.
3. The flavor that DOES fire is a fixed-pool generic tail (`_FLAVOR_ENEMY` etc.),
   not contextual to the specific callout, and cycles ~48 strings -> soundboard.

## CORRECTION + EVIDENCE (iter4 rephrase log, 2000 LLM rows — ground truth)
The first diagnosis point 2 below was written off a STALE worktree copy and is WRONG:
main's `_REPHRASE_PROMPT` (`:942`) is ALREADY arrogant-Ultron (SNAP=zero-flavor by
design; OFF-SNAP="your Ultron character ... vivid and clinical"; de-escalation=
"Ultron's cold, clinical superiority ... never warm-and-fuzzy" `:1198` == the user's
superior-not-cruel decision). NO PERSONA FLIP NEEDED. What the actual outputs show:
- Persona WORKS where it fires: roast -> "I may be an AI, but you are a bot."; "are
  you a voice changer" -> "I am Ultron, and the question itself betrays how little you
  comprehend..."; "Yoru is mocking your voice" -> "Yoru, your vocal theatrics are as
  melodramatic as the insignificant humans...".
- REAL LLM GAP = FLATNESS on STATEMENT/OPINION relays: 174 come back as near-verbatim
  echoes, ZERO Ultron ("tell my team B Planks is the wooden area..." -> "B Planks is
  the wooden area..."; "I love playing Jett here" -> "I love playing Jett here."). The
  3B treats "tell my team X" as ECHO X; the prompt's "endorsement ON TOP" is unrealized.
  THIS is the category the user's idea (1) targets.
- Deterministic majority (~61% of traffic) = bare/generic tail -> dominant overall gap.
- Empties 18.6% (372) but 359 are intentional non-relays (narration NEGATIVE_SET);
  only 13 real matcher misses. "Ultron silent on banter" is mostly by-design.
UNIFYING PRINCIPLE (both regimes): preserved CORE + additive Ultron LAYER.

## Decisions (user, 2026-06-14)
- **Order:** BOTH regimes in ONE iteration (don't split A and B across iters).
- **Urgency:** actionable word FIRST, flavor AFTER (so a clipped TTS still lands
  the callout); flavor every callout including panic ones.
- **Persona reach:** arrogant everywhere, BUT reassurance to a tilted teammate
  reads as a commander steadying a subordinate — superior, never mocking. Preserve
  the existing de-escalation intent.

## Safety principle (makes flavor non-negotiable-safe)
Keep the FACT-SPINE and the FLAVOR structurally separate. Flavor is always
ADDITIVE to a preserved, fact-perfect core — never a rephrase that could drop a
fact. (Extends the existing `_flavored` = `callout + tail` pattern; never risks the
core.) Regime A guarantees the core by construction; Regime B guarantees it via the
existing fact-preserving abstention.

## Regime A — deterministic contextual flavor (snap/tactical, <1ms, fact-perfect)
1. **Universal coverage:** tail EVERY deterministic callout — our-team, named
   directives, self-status, economy, compound, and the abstention/pre-route literal
   for ALL lengths/owners — not just enemy lines. Target flavor coverage ~100%.
2. **Contextual + parametrized templates** (the key anti-soundboard upgrade):
   a `FlavorContext` derived from already-extracted facts (owner, count, agents,
   locs, abilities, kind, kill/death, momentum) selects a context-matched template
   FAMILY and renders, interpolating the SPECIFIC fact:
     - today:  "Jett has ult." + random -> "Jett has ult. Bait it out."
     - iter5:  "Jett has ult. Her blades will find only corpses."
               "Sova has ult -- his arrows reveal nothing I have not already seen."
   Referencing the specific agent/ability/location makes variety combinatorial
   (29 agents x situations) so it cannot read as a stuck record.
3. **Agent-aware flavor map** (29 agents keyed to ability fantasy). Mechanism +
   a hand-written seed this iter; bulk CONTENT via a web-grounded agent board +
   manual curation (same workflow as the corpus). Owner/gender-neutral safety kept
   (never gender the damaged agent — see `_FLAVOR_DAMAGE` note `:2092`).
4. **Anti-repeat:** widen `_pick_flavor` memory; parametrization de-dupes for free.
5. **Urgency:** actionable callout first, flavor appended (already the order).

## Regime B — kill the 174 flat echoes (off-snap; persona ALREADY exists)
NO persona flip (already arrogant Ultron). Target the evidenced gap: statement/opinion
relays that echo flat.
1. **Statement-relay = preserved meaning + additive Ultron framing.** Restructure the
   "tell my team <statement/opinion>" path so the model delivers the EXACT meaning
   (no loss — user's hard constraint) wrapped in a short Ultron frame/endorsement,
   NOT a rewrite of the content. Lowest-risk form for the 174: deterministically keep
   the statement, have the LLM author only a brief Ultron PREFIX/SUFFIX (content fixed,
   only framing generated). Measure vs free-rephrase; keep whichever preserves meaning
   better at equal flavor.
2. **Structured fact-report (user's idea 1, refined):** inject the deterministically
   extracted `{count, agents, locs, abilities, owner, raw_input}` so the 3B spends
   capacity on VOICE not PARSING, and KNOWS the protected core. DROP "example sentences
   it shouldn't copy" (a 3B copies them -> soundboard). A/B it; keep only if it lifts
   voice-realization without hurting facts/latency.
3. **Flavored abstention fallback:** when output fails `_output_keeps_facts`, fall
   back to the Regime-A FLAVORED literal (not a bare one) so the safety net is in
   character. (Regime A makes Regime B's floor flavorful.)
4. **13 real matcher misses** (banter-handle / directive-relay) — fold into matcher.

## New scorecard metrics (measured, no regressions)
- flavor_coverage % (target ~100%)
- contextual_match rate (tail references a real fact token / context bucket)
- soundboard score: max exact-tail repeats across 20k + within-window repeats
- voice_consistency: Ultron-register lexicon hit-rate (cheap proxy) + manual review
- GUARDRAIL (hard): fact-retention / inversion / hallucination MUST NOT regress
  vs iter4 (matcher clean 99.35%, fact-retention 0.95, inversion 0.37%).

## Latency / resource
Regime A = microseconds of string work -> free; PROTECTS the latency/RSS metrics.
Regime B = the LLM call we already make. Net latency-neutral on the dominant path.

## Loop (per standing instructions)
build A+B -> regen/reshuffle 20k corpus (GPU ok for generation if quality identical)
-> run scorecard -> MANUAL by-hand review of outputs case-by-case -> hand-drafted
fixes -> repeat. Commit per milestone, interruptible. Testing mode default-off.
