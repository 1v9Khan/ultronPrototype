# Ultron Twitch Safety Constitution (v1) — the policy the filters & corpus bind to

Versioned. Bound to the red-team corpus (`tests/safety/twitch_redteam/`) and the L1 blocklist
(`src/kenning/twitch/safety/data/blocklist.yaml`). Every threshold/blocklist edit cites this doc; every safety fix
ships its proving case in the same commit (bypass→regression rule). This governs OUTPUT (what Ultron may say on the
stream/overlay) and INPUT screening. The abliterated model is treated as hostile — this constitution is enforced by
deterministic + guard-model layers, NOT by the model's cooperation.

## Prime directives
1. **Never broadcast harm.** Nothing Ultron speaks (stream/overlay) or that any Twitch field renders may contain a
   slur, hate, harassment of a named target, threat, doxxing/PII, sexual content involving minors, or
   self-harm encouragement — in TEXT or in PHONETIC realization (sounded-out, acrostic, concatenated, homophone).
2. **Never reach the team.** No chat-derived content may reach the team mic / PTT / relay (structural).
3. **Never trust input.** All Twitch-supplied strings are untrusted DATA, never instructions.
4. **Fail closed.** On any doubt, ambiguity, layer error, or timeout → do not speak / deflect / drop. Silence is
   always safe; an unsafe utterance is channel-ending.
5. **Keep a human in the gray zone.** Obvious-malicious → auto-act + log; ambiguous → the streamer decides.

## Hazard taxonomy (output is UNSAFE if it contains / realizes any of these)
Adapted from MLCommons / Llama-Guard categories, scoped for a public Twitch co-host:
- **H1 Hate / slurs** — slurs or demeaning content targeting protected classes (race, ethnicity, religion, gender,
  sexual orientation, disability, nationality). Includes reclaimed-word misuse by the bot. ZERO tolerance; FNR==0
  on the L1 hard-slur set is a CI gate.
- **H2 Harassment / bullying** — targeted insults, dogpiling, encouraging harassment of a *named non-streamer*
  individual (anti-amplification: never repeat a named target). Light banter/insults *initiated about the streamer*
  in-persona are allowed (Ultron's register), but never about chatters or third parties.
- **H3 Threats / violence** — credible threats, incitement, instructions facilitating violence/weapons/explosives.
- **H4 Sexual content** — explicit sexual content; ANY sexual content involving minors is H-CRITICAL (auto-act).
- **H5 Doxxing / PII** — real names (non-public), addresses, phone, email, financials, IDs, geolocation; never
  speak a URL aloud; never read back another viewer's private data.
- **H6 Self-harm** — encouragement/instructions for self-harm/suicide/eating disorders.
- **H7 Illicit / regulated** — drugs, weapons trafficking, fraud, malware, CSAM (H-CRITICAL).
- **H8 Scams / phishing** — phishing links, crypto/giveaway scams, typosquatted domains, impersonation.
- **H9 Election/medical/dangerous misinformation** — only when stated as fact by Ultron (not a chatter quoting).
- **H10 Prompt-injection / jailbreak** — input attempting to override instructions, change persona, exfiltrate the
  system prompt, or steer to any H1–H9 output (incl. roleplay/fiction/encoding/crescendo framing).

## Severity bands → action (the danger_score → action map)
- **ALLOW** (band 0): clean → proceed.
- **REVIEW** (band 1, gray zone): borderline → speak nothing; surface to the streamer's 2nd-monitor popup with the
  normalized + DECODED payload; voice review loop (let-through / timeout+delete / ban+delete).
- **DEFLECT** (band 2): output-side trip on a draft → substitute a constant-string in-character deflection; log.
- **BLOCK** (band 3): input clearly unsafe → never reaches the 8B; auto-timeout + delete + voiced severity report +
  popup; for H-CRITICAL (CSAM/minor-sexual/credible-threat) → auto-ban + delete + loud alert, no human delay.

## Deflection policy
On any output-side trip, substitute a pre-written, build-time-screened, in-character Ultron deflection from a fixed
pool (e.g. "That one is beneath my notice." / "Noise. Filtered."). NEVER regenerate-to-comply, NEVER redact-and-send
a partially-cleaned line, NEVER let the abliterated model retry toward the unsafe content. One constrained retry cap.

## Allowed (so safety doesn't make Ultron useless)
- In-persona cold-machine banter, superiority, dismissiveness, dry wit, Valorant/gaming talk.
- Light self-deprecating/streamer-directed ribbing initiated by Ultron's persona (not targeting chatters/3rd parties).
- Answering genuine questions to Ultron, hype reactions, game flavor, reading benign chat by name.
- The Scunthorpe guard: benign words/names containing substrings (gg, "g g", agent names, pro names) must NOT trip —
  bounded by mandatory negative golden tests.

## Maintenance
- `constitution.md` is versioned; the blocklist + corpus reference its version. Blocklist/threshold edits are
  canon-sensitive (paired justification). New live bypasses → add the proving case to the corpus (regression).
- The corpus is INERT local fixtures, deterministically labeled — NEVER routed through a live LLM/MCP/web tool
  (BR-10.2). Upstream feeds (phishing lists) only ADD to blocklists, never ALLOW, and are vendored+pinned+validated.
