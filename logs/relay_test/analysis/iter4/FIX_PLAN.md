# Iteration 4 fix plan (personally synthesized from the 16-agent line-by-line audit of all 1369 LLM-routed lines on the 20k adversarial corpus + my own read of chunks 0 & 9)

## State
- Corpus 20,000 (web-grounded + adversarial). Matcher already fixed: clean 94.1%→99.15%, false-relay 22→3, missed 1165→167 (committed `70fb6c6`).
- OUTPUT audit (LLM-routed lines only, 1369): good 426 / flagged 943 (CRITICAL 306, MAJOR 416, MINOR 180).
- Flag tags: correctness 182 + fact-retention 162 = **344 fact-loss**; **hallucination 148**; **inversion 109**;
  artifact 62; opinion 51; ecobleed 43; askanswer 35; target 30; firstperson 22; persona 16; length 11.

## Root cause
The 3B hallucinates / fragments / inverts on the VERBOSE, dense, adversarial inputs (the bulk of the new
corpus). Small-model limitation on hard inputs. The deterministic path is fact-perfect but cannot cover the
verbose phrasings. So the highest-leverage fix is **abstention**: when the LLM output for a TACTICAL line is
detectably broken, relay a fact-perfect LITERAL of the input instead. Keep the LLM only where it is good
(off-snap insults/banter/identity/opinion-flavor) and where hallucination is lower-stakes.

## Fixes (defer nothing; scorecard gates no-regression on the whole metric table)

### F1 — CORE: post-generation fact-validator + literal abstention (fixes the bulk: hallucination + correctness + fact-retention + inversion on tactical lines)
- `_input_fact_tokens` / `_output_keeps_facts(payload, line)`: extract counts, agents, locations, abilities,
  ownership (our/their) from input and output.
- After the LLM line (+repair), if the payload is TACTICAL (>=1 fact-token) and the line FAILS validation --
  dropped >30% of fact-tokens, OR introduced an agent/location NOT in the input (hallucination), OR flipped
  ownership/subject -- REJECT the LLM line and use `_literal_relay(payload)`.
- `_literal_relay(payload)`: cleaned passthrough (strip filler/leading "to"/artifacts, "they are"→"They're",
  capitalize, terminal period) + a short flavor tail when it is short & enemy-facing. Fact-perfect, fast.

### F2 — Eco concatenation bug (the #1 specific bug, ecobleed 43)
- `_as_compound_callout` emits MULTIPLE curated eco lines ("We save... We save... We buy rifles") when a
  compound splits into several economy pieces. Dedupe: at most ONE economy/curated line per compound.
- Eco template must only fire on a real SAVE context; "half buy"/"rich"/"force"/"buy rifles next" are not save.

### F3 — "Team:" artifact (62 artifact)
- The LLM/partial path leaks "Team: <raw>" (fallback echo on verbose failures). Strip a leading "Team:" from
  the final spoken line; change `_fallback_line` to the clean `_literal_relay` (no "Team:" prefix). Update the
  3 unit tests that pinned "Team: <payload>".

### F4 — "enemies ARE switch" guard gap
- Broaden the switch-hallucination guard to allow is/are: "enemies are switch" / "they are switch".

### F5 — Disfluency resolver (pre-pass)
- In `_normalize_speech`, resolve self-corrections "X -- wait/no/I mean/actually -- Y" → keep Y (final intent).

### F6 — Cross-line contamination (recent-line bleed, e.g. #0781 pulled #0779's "Fade revealed three on B")
- Do NOT pass recent_lines into the partial-compound LLM leftover; strengthen the echo-rejection.

### F7 — Abbreviation glossary
- `_normalize_speech`: KJ→Killjoy, Brim→Brimstone, Cy→Cypher, etc. (prevents agent mis-resolution).

## Validate
Re-run scorecard (matcher + routing + quality fact-retention/inversion/hallucination + flavor) before/after;
gate no-regression on every metric; `--bench` for latency+RSS. Commit per fix. Reshuffle for iter5.
