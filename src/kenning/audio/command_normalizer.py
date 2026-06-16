"""Pre-routing command normalizer for the live voice path.

Raw STT output is messy in ways that wreck strict command routing:

  * the wake word bleeds a leading fragment ("Ultron, ..." -> "Run, ...") and
    natural speech adds filler ("uh", "I mean", "I hope", "and ...");
  * the first command word -- almost always the relay verb "tell" -- gets
    clipped, so "Ultron, tell my team there's a Jett on A main" arrives as
    "my team, there's a Jet A main" (no verb -> the strict relay matcher misses
    it -> it falls through to the conversational LLM);
  * Valorant proper nouns are misheard ("Jett"->"jet", "Cypher"->"cipher",
    "Raze"->"ray zombie", "Sova"->"Silva", "B main"->"be main").

This layer cleans the transcript into a CANONICAL command string BEFORE routing,
so every downstream matcher (relay / Spotify / identity / desktop) sees text it
can actually match. The whole point: routing should be robust to how the
streamer really talks, not just to textbook phrasing.

Pipeline (each stage conservative + idempotent on already-clean text):

  1. ``_strip_leading_junk``  -- drop a leading misheard wake word + filler.
  2. ``correct_callout_stt``  -- Valorant vocab + agent corrections (phrase,
     context, token, fuzzy), reused from :mod:`kenning.audio._stt_correct`.
  3. ``recover_relay_lead``   -- when the cleaned text is clearly a TEAM CALLOUT
     but the relay verb was dropped, prepend the canonical "tell my team ..."
     lead so the strict relay matcher catches it. CONTEXTUALLY GATED: questions,
     Spotify, identity, and desktop commands are NEVER rewritten as relays.

``normalize_command`` is the single entry point the orchestrator calls before
the dispatch chain. It is cheap (regex + short-string work, ~tens of us) and
returns the input unchanged when nothing applies.
"""

from __future__ import annotations

import re

from kenning.audio._stt_correct import _AGENTS, correct_callout_stt
from kenning.audio._relay_intent import relay_intent_ok

# ---------------------------------------------------------------------------
# 1. Leading junk: misheard wake word + conversational filler.
# ---------------------------------------------------------------------------
# Wake-word homophones the STT prepends ("Ultron"->Run/Ron/Tron/One/...), plus
# disfluencies and lead-ins people say before the real command. Stripped
# iteratively from the FRONT only. We never strip the entire utterance (if the
# strip would empty it, keep the original) so a bare "okay" still survives.
_WAKE_HOMOPHONES = (
    r"ultron|altron|voltron|ultra|ultro|tron|ron|run|rons"
)
_FILLER = (
    r"hey|ok|okay|um+|uh+|er+|hmm+|so|well|yeah|yep|yup|now|and|then|"
    r"please|alright|right|i\s+mean|i\s+think|i\s+hope|i\s+guess|i\s+wanna|"
    r"i\s+want\s+to|let'?s\s+see|you\s+know|basically|just|"
    # conversational address-fillers that leak before a relay lead ("bro relay X",
    # "dude tell them Y", "yo call out Z") -- safe to strip from the front.
    r"bro|bruh|dude|homie|fam|bud|buddy|guys|yo"
)
# "like" is filler ("like, tell my team X") BUT also the Spotify verb
# ("like this song" / "like it"). Strip it as filler ONLY when it is NOT
# immediately followed by a Spotify object -- otherwise the leading-junk pass
# turned "like this song" into "this song", which then matched "now playing".
_LIKE_FILLER = r"like(?!\s+(?:this|that|it|the|some|my)\b)"
# A leading token run: wake homophone(s) and/or filler, each optionally
# followed by light punctuation. Anchored at start, case-insensitive.
_LEADING_JUNK = re.compile(
    rf"^(?:\s*(?:{_WAKE_HOMOPHONES}|{_LIKE_FILLER}|{_FILLER})\b[\s,.:;!?-]*)+",
    re.IGNORECASE,
)


def _strip_leading_junk(s: str) -> str:
    """Strip a leading wake-remnant / filler run. Never empties the string."""
    out = _LEADING_JUNK.sub("", s, count=1).lstrip()
    return out if out else s


# Self-correction disfluency: "tell my -- no wait, tell the whole team to X" /
# "relay to Raze -- wait the whole team -- everyone go B" -- the streamer
# corrects themselves mid-utterance. Take the text AFTER the LAST correction
# marker (the intended final command). The markers are speech self-corrections,
# NOT tactical words ("rotate B NOT A" keeps its "not"; "wait for the molly"
# keeps its "wait" -- only "-- wait" / "no wait" / "scratch that" trigger).
# Explicit self-correction CUES. The PRESENCE of one marks the utterance as a
# repair; only THEN do we also treat the bare em-dash "--" (which the streamer/
# corpus uses as a self-interruption) as a correction boundary. We never key off
# bare "no" / "wait" / "not" alone -- those are tactical ("rotate B NOT A",
# "wait for the molly") and must survive.
_DISFLUENCY_CUE_RE = re.compile(
    r"(?:--+\s*(?:no\s+)?wait\b|\bno\s+wait\b|\bno\s+no\b|\bscratch\s+that\b"
    r"|\bnever\s*mind\b|\bforget\s+it\b|\bactually\s+no\b|\bor\s+rather\b"
    r"|\bi\s+mean\b|--+\s*no\b|--+\s*actually\b|\blet\s+me\s+rephrase\b"
    r"|--+\s*to\s+(?:all|the)\b)",
    re.IGNORECASE,
)
# Boundaries to split on once the utterance is flagged as a repair: every cue
# above PLUS bare "--". Keep the segment after the LAST boundary (the final
# intended command). Ordered so multi-word cues match before the bare dash.
_DISFLUENCY_SPLIT_RE = re.compile(
    r"(?:--+\s*(?:no\s+)?wait\b|\bno\s+wait\b|\bno\s+no\b|\bscratch\s+that\b"
    r"|\bnever\s*mind\b|\bforget\s+it\b|\bactually\s+no\b|\bor\s+rather\b"
    r"|\bi\s+mean\b|--+\s*to\s+(?:all|the)\s+\w+|--+\s*actually\b|--+\s*no\b"
    r"|--+)"
    r"[\s,.:;\-]*",
    re.IGNORECASE,
)


def _resolve_disfluency(s: str) -> str:
    """Resolve a mid-utterance self-correction to its FINAL intended command,
    preserving the relay lead. "call out Iso contract -- wait shield -- Double
    Tap, he has shield up" -> "tell my team Double Tap, he has shield up". Only
    fires when an explicit correction cue is present, so ordinary callouts (incl.
    tactical "rotate B not A" / "wait for the molly") are never touched."""
    if not _DISFLUENCY_CUE_RE.search(s):
        return s
    ms = list(_DISFLUENCY_SPLIT_RE.finditer(s))
    if not ms:
        return s
    tail = s[ms[-1].end():].strip(" ,.;:-")
    if len(tail.split()) < 2:
        return s  # final repair too short to stand alone -> keep original
    # Preserve a leading relay verb the correction chain dropped, so the repair
    # stays routable as a relay rather than a bare fragment the gate rejects.
    if (_HAS_RELAY_LEAD.match(s) and not _HAS_RELAY_LEAD.match(tail)
            and not _TEAM_LEAD.match(tail)):
        return "tell my team " + tail
    return tail


# "relay to my Tejo: X" / "tell our Sova: X" -- a possessive before a ROSTER
# agent breaks the named-addressee matcher (it expects "relay to Tejo"). Strip
# the my/our before a real agent name (closed vocab, so an arbitrary name is
# never touched). Handles repeated "... my Sova and my Fade ...".
_POSSESSIVE_NAME_RE = re.compile(
    r"\b(to|tell|ask|warn|remind|relay\s+to)\s+(?:my|our)\s+"
    r"(?P<name>" + "|".join(
        sorted((re.escape(a.replace("/", "")) for a in _AGENTS),
               key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _strip_possessive_names(s: str) -> str:
    # First pass handles "<verb> to my Sova"; a global pass catches "and my Fade".
    s = _POSSESSIVE_NAME_RE.sub(lambda m: f"{m.group(1)} {m.group('name')}", s)
    return re.sub(
        r"\band\s+(?:my|our)\s+(" + "|".join(
            sorted((re.escape(a.replace("/", "")) for a in _AGENTS),
                   key=len, reverse=True)) + r")\b",
        lambda m: f"and {m.group(1)}", s, flags=re.IGNORECASE)


# STT mis-hears the verbatim verb "repeat" as "Pete"/"Heat"/"repeete" when it
# leads a soundboard relay ("repeat to my team X" -> "Pete to my team X" /
# "Heat to the team X"). Restore it ONLY when followed by "to"/"after" + an
# addressee, so a literal name "Pete" or the word "heat" is never rewritten.
_REPEAT_MISHEAR = re.compile(
    r"^(\s*)(?:pete|peat|heat|repeet|repete|reet|repeate)\b"
    r"(?=\s+(?:to|after)\b)",
    re.IGNORECASE,
)

# "tell my team word for word X" / "tell the team verbatim X" -- the verbatim
# marker rides AFTER the addressee. Rewrite to the canonical "say exactly to my
# team X" so the soundboard/verbatim matcher relays X EXACTLY (no flavor tail).
_WORD_FOR_WORD = re.compile(
    r"^\s*(?:tell|say|relay)\s+(?:to\s+)?(?:my\s+|our\s+|the\s+)?"
    r"(?:team|teammates?|squad|guys|boys|mates|crew)\s+"
    r"(?:to\s+(?:say|repeat)\s+)?(?:word\s+for\s+word|verbatim|exactly)\b"
    r"\s*[:,]?\s*(.+)$",
    re.IGNORECASE,
)

# A reported QUESTION ("Jett asked about Tony Stark", "my teammate is wondering
# if you're a bot", "Reyna asked how far the moon is") must NOT be Valorant-vocab
# corrected (it mangled "Iron Man" -> "Iron main") and must NOT get a "tell my
# team" callout lead -- it routes to Ultron's in-character ANSWER path. Anchored
# to the start (subject + reported verb + question word) so a tactical callout
# that merely contains "asked" is never gated.
_REPORTED_QUESTION_GATE = re.compile(
    r"^\s*(?:my\s+|our\s+|the\s+)?\w+(?:\s+\w+)?\s+"
    r"(?:just\s+|is\s+|are\s+|was\s+|has\s+|been\s+)?"
    r"(?:asked|asking|asks|wondering|wonders|wondered|curious"
    r"|wants\s+to\s+know|wanted\s+to\s+know)\s+"
    r"(?:you\s+|me\s+|us\s+|the\s+team\s+)?"
    r"(?:about|if|whether|why|how|what|where|when|who|which)\b",
    re.IGNORECASE,
)

# Possessive on the team addressee ("my team's X", "tell the squad's X"): the
# trailing "'s" breaks the relay lead-strip (it relayed "Tell my team's Cypher
# cage on A" verbatim). Drop it so the addressee is the plain "my team".
_TEAM_POSSESSIVE = re.compile(
    r"\b((?:my|our|the)\s+(?:team|teammates?|squad))'s\b",
    re.IGNORECASE,
)


# A BARE greeting that is the WHOLE utterance ("hello", "hey there", "yo
# Ultron"). It MUST be left verbatim: it is not a callout, and the aggressive
# Valorant vocab correction would otherwise snap "hello" onto the callout
# location "hell" (identical Metaphone code) and then relay-recover it into
# "tell my team hell" -- so a plain greeting was being broadcast to the team as
# garbage ("No hell."). Anchored to the END so a real callout that merely OPENS
# with a greeting ("hey, two on B") is never gated.
_BARE_GREETING = re.compile(
    r"^\s*(?:hello+|hi+|hiya|heya|hey+|yo+|sup|wassup|what'?s\s+up|howdy|"
    r"greetings|good\s+(?:morning|afternoon|evening))"
    r"(?:[\s,]+(?:there|ultron|everyone|team|guys|all|y'?all))?\s*[.!?]*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 2. Relay-lead recovery (the dropped-"tell" fix).
# ---------------------------------------------------------------------------
# Already a valid relay / compose / soundboard lead -> never recover.
_HAS_RELAY_LEAD = re.compile(
    r"^\s*(?:please\s+)?(?:tell|say|let|warn|inform|remind|wish|ask|relay|"
    r"repeat|echo|yell|shout|announce|broadcast|call\s+out|encourage|hype|"
    r"roast|flame|give|share|drop|"
    r"criticize|criticise|critique|rip\s+into|tear\s+into|chew\s+out)\b",
    re.IGNORECASE,
)

# "I want my team to X" / "I need the squad to X" / "I wanna tell my team X" --
# the streamer states the intent to relay with the addressee EMBEDDED in the
# middle. Without this, recover_relay_lead prepends a SECOND "tell my team" to
# the whole thing ("tell my team I want my team to rotate to B"), and the relay
# rephraser then compressed the doubled lead away ALONG WITH the real payload
# ("Rotate." -- the site was lost). Extract the directive X so the line is just
# "tell my team X".
_WANT_TEAM = re.compile(
    r"^\s*i\s+(?:just\s+)?(?:want|need|wanna|gotta|would\s+like|wish)"
    r"(?:\s+to)?(?:\s+(?:tell|say|let|warn|remind|inform))?\s+"
    r"(?:for\s+)?(?:my\s+|our\s+|the\s+)?"
    r"(?:team|teammates?|squad|boys|guys|mates|crew|everyone|everybody)\b"
    r"(?:\s+know)?[\s,:]*(.+)$",
    re.IGNORECASE,
)

# A team-address lead with the VERB dropped ("my team ...", "the squad ...").
_TEAM_LEAD = re.compile(
    r"^\s*(?:my\s+|our\s+|the\s+)?"
    r"(?:team|teammates?|squad|boys|guys|mates|homies|fellas|crew)\b[\s,:]*",
    re.IGNORECASE,
)

# A no-VERB team lead: "on my team X" / "to the squad X" -- the streamer addresses
# the team with a preposition instead of "tell". Strip the prep+addressee and
# prepend the canonical "tell my team" so the payload (X) isn't leaked with the
# "on my team" lead glued to the front of the relayed line.
_TEAM_LEAD_NOVERB = re.compile(
    r"^\s*(?:on|to)\s+(?:my\s+|our\s+|the\s+)?"
    r"(?:team(?:mates?)?|squad|boys|guys|mates|crew)\b[\s,:.]*",
    re.IGNORECASE,
)

# A TRAILING relay command ("Viper wall is up, tell my team." / "they're pushing
# B, let the squad know."). The command sits at the END; without stripping it,
# recover_relay_lead would prepend a fresh "tell my team" and DUPLICATE the
# trailing command into the relayed payload. Matched only as a bare tail (no
# payload after the addressee) so a real "tell my team to push" is untouched.
_TRAILING_RELAY_TAIL = re.compile(
    r"[\s,.;:!?-]+\b(?:tell|say(?:\s+to)?|let|warn|remind|inform|relay(?:\s+to)?)\s+"
    r"(?:my\s+|our\s+|the\s+)?"
    r"(?:team(?:mates?)?|squad|boys|guys|mates|crew|everyone|everybody|'?em|them)"
    r"(?:\s+know)?\s*[.!?]*$",
    re.IGNORECASE,
)

# Negative gate: utterances that must NEVER be rewritten into a relay -- they
# belong to other routes (conversation / Spotify / identity / desktop) and the
# strict matchers there should see them verbatim.
_NOT_A_CALLOUT = re.compile(
    r"^\s*(?:"
    # questions / conversational openers + "tell me ..." (about Ultron/Marvel)
    r"(?:what|what'?s|who|whom|whose|where|when|why|how|which|is|are|am|was|"
    r"were|do|does|did|can|could|should|would|will|won'?t|have|has|had|may|"
    r"might|should|shall)\b"
    r"|tell\s+me\b|ask\s+me\b|teammate\s+asked\b|my\s+teammate\s+asked\b"
    # any "... asked about / asked me ..." -> relaying a question ABOUT a topic
    r"|(?:my\s+)?(?:team|teammate|teammates)\s+asked\b"
    # more conversational / personal openers (never a team callout)
    r"|explain\b|describe\b|define\b|summari[sz]e\b|remind\s+me\b"
    r"|i\s+think\b|i\s+feel\b|i\s+wonder\b|i\s+want\s+to\s+know\b|what\s+about\b"
    r"|thank\s+you\b|thanks\b|good\s+bye\b|goodbye\b"
    # Spotify control verbs
    r"|play\b|pause\b|resume\b|unpause\b|skip\b|next\b|previous\b|prev\b|"
    r"stop\b|mute\b|unmute\b|shuffle\b|repeat\b|loop\b|volume\b|louder\b|"
    r"quieter\b|softer\b|turn\s+it\b|turn\s+the\s+volume\b|crank\b|"
    r"like\s+this\s+(?:song|track|one)\b|love\s+this\s+(?:song|track|one)\b|"
    r"unlike\b|thumbs\b|save\s+this\s+(?:song|track|one)\b|"
    r"what'?s\s+playing\b|who\s+sings\b|what\s+song\b|now\s+playing\b|"
    r"throw\s+on\b|put\s+on\b|queue\b|start\s+playing\b|keep\s+playing\b|"
    r"go\s+back\b|restart\b|start\s+it\s+over\b|i\s+wanna\s+hear\b"
    # identity / greeting (the greet matcher handles these)
    r"|introduce\b|identify\b|state\s+your\s+name\b|who\s+are\s+you\b|"
    r"say\s+(?:hi|hello|hey|what'?s\s+up)\b|are\s+you\s+(?:there|online|ready)\b"
    # desktop / safety (refused in gaming, never relayed)
    r"|take\s+a\s+screenshot\b|screenshot\b|click\b|type\b|open\s+\w|launch\b|"
    r"move\s+the\s+mouse\b|close\s+\w|minimi[sz]e\b|maximi[sz]e\b"
    # control toggles
    r"|mute\s+the\s+team\b|gaming\s+mode\b|anticheat\b"
    r")",
    re.IGNORECASE,
)

# Context+directive shape: a reported-speech clause ("<teammate/agent> asked /
# said / is flaming me ...") followed by a directive to ANSWER ("respond",
# "reply", "calm him down", "clap back", ...). Ultron should AUTHOR an
# in-character answer (the relay's context+directive matcher handles it, incl.
# Marvel/Avengers questions answered as Ultron). Detected here so recover_relay_lead
# does NOT blindly prepend "tell my team" -- which turned "Jett asked you about
# Tony Stark, respond" into a LITERAL relay of the question instead of an
# in-character reply (the "teammate asked" form already escaped via _NOT_A_CALLOUT;
# a NAMED agent did not). Conjunctive: a reported-speech verb AND a closing
# answer-directive, so ordinary callouts never trip it.
_REPORTED_RESPOND_RE = re.compile(
    r"\b(?:asked|asking|asks|said|saying|says|told|wants?|wanted|wondering|"
    r"thinks?|thinking|typed|wrote|complain\w*|crying|flam\w*|tilted|raging|"
    r"malding|mock\w*|teas\w*|roast\w*|clown\w*|diss\w*|ridicul\w*|insult\w*|"
    r"bully\w*|making\s+fun|trash[\s-]?talk\w*|calling\s+(?:me|you|us))\b.*"
    r"\b(?:respond|reply|answer|acknowledge|agree|clap\s+back|back\s+me\s+up|"
    r"defend\s+me|set\s+(?:him|her|them)\s+straight|calm\s+(?:him|her|them)\s+down|"
    r"de[\s-]?escalate|shut\s+(?:him|her|them|it)\s+down|"
    r"say\s+something|handle\s+(?:it|that|him|her|them)|"
    r"deal\s+with\s+(?:it|that|him|her|them))\b\s*[.!?]*$",
    re.IGNORECASE,
)

# Spotify signal that can appear MID-utterance (so the start-anchored gate above
# misses it): "set the volume to 40", "make the volume 60", "lower the volume
# by 10". Any of these mark a music command -> leave verbatim for the Spotify
# handler, never rewrite as a team relay.
_SPOTIFY_SIGNAL = re.compile(
    r"\b(?:volume|spotify|the\s+song|this\s+song|that\s+song|the\s+track|"
    r"this\s+track|the\s+music|the\s+album|the\s+playlist|the\s+queue|"
    r"next\s+track|next\s+song|previous\s+(?:song|track))\b",
    re.IGNORECASE,
)

# Positive callout signals -- ANY one marks the utterance as a team callout
# worth relaying (used only for the BARE form, with no team lead). Enemy
# framing, abilities/utility, locations, self-status, orders, morale.
_CALLOUT_SIGNAL = re.compile(
    r"\b(?:"
    # enemy / spotting
    r"there'?s|theres|enemy|enemies|they'?re|hostile|spotted|incoming|"
    r"pushing|push|pushed|rotating|rotate|rotated|lurking|lurk|holding|hold|"
    r"flanking|flank|planting|plant|planted|peeking|peek|defusing|defuse|"
    # abilities / utility
    r"ult|ulted|ulting|ultimate|turret|wall|walled|smoke|smokes|smoked|"
    r"flash|flashed|flashing|molly|molotov|drone|dart|cage|cages|trip|"
    r"tripwire|nanoswarm|stun|stunned|knife|blade|spike|orb|recon|"
    # count + position ("two on B", "one mid", "three a", "two pushing")
    r"(?:one|two|three|four|five|six)\s+(?:on|in|at|pushing|coming|rushing|"
    r"rotating|left|right|mid|main|here|there|a\b|b\b|c\b)|"
    # locations
    r"heaven|hell|main|mid|middle|long|short|window|site|spawn|market|"
    r"garage|hookah|connector|tree|elbow|ramp|pit|rafters|generator|"
    r"cubby|catwalk|alley|courtyard|stairs|lamps|showers|kitchen|dish|"
    r"snowman|tube|tubes|vents|tower|boathouse|link|lobby|default\b|logs|"
    # self status
    r"low|one\s+shot|one-shot|reloading|i\s+died|i'?m\s+dead|i'?m\s+low|"
    r"i'?m\s+planting|i'?m\s+flanking|i'?m\s+pushing|i'?m\s+rotating|"
    r"i'?m\s+holding|i'?m\s+going|i\s+have\s+(?:a|b|c|site|the\s+spike)|"
    # orders / morale / social
    r"save|eco|force|retake|default|stack|stacked|stacking|group\s+up|"
    r"fall\s+back|lock\s+in|split|execute|executing|rush|rushing|rushed|"
    r"good\s+game|gg|nice|great\s+play|well\s+played|let'?s\s+go|we\s+got\s+this|"
    r"good\s+luck|my\s+bad|nice\s+shot|clutch|careful|watch\s+the|"
    r"winning|we'?re\s+winning|we\s+lost|good\s+round|great\s+round|nice\s+round|"
    # 2026-06-15 expanded callout coverage (relay without an explicit "tell my
    # team" lead): counts/kills, weapons, movement, requests, clears.
    r"down|left|clear|cleared|going|crossing|crossed|behind|boost|boosting|"
    r"anchor|anchoring|switch|switching|cover|covering|picks|trade|traded|"
    r"kill(?:ed|ing|s)?|got\s+(?:one|him|her|two|the\s+kill)|"
    r"op|operator|awp|odin|sheriff|guardian|vandal|phantom|judge|bucky|"
    r"marshal|outlaw|shorty|"
    r"need\s+(?:a\s+|an\s+|some\s+)?(?:drop|heal|heals|backup|back\s+up|rifle|"
    r"gun|smoke|smokes|flash|util|pickup|res|revive|trade|orb|orbs)|drop\s+me"
    r")\b",
    re.IGNORECASE,
)

# Roster agent names (canonical, post-correction) are also a callout signal.
_AGENT_SIGNAL = re.compile(
    r"\b(?:" + "|".join(re.escape(a.replace("/", "")) for a in _AGENTS) + r")\b",
    re.IGNORECASE,
)

# First-person musing / self-narration that merely MENTIONS relaying or a callout
# keyword -- "I should tell them to eco", "honestly should I be asking my team to
# push", "every time I tell them to flank someone dies", "for the viewers at
# home: my team needs someone to tell them to eco". The streamer is thinking out
# loud, NOT issuing a relay. Zero-cost fast-path before the semantic gate (and the
# fallback when the sidecar is down). Start-anchored + first-person-modal so real
# self-status callouts ("I'm planting", "I died", "I'm low", "I need a drop",
# "I got one") are NEVER gated; the directive forms ("I want/need you to ...") are
# owned by _WANT_TEAM and excluded here.
_NARRATION_MUSING_RE = re.compile(
    r"^\s*"
    r"(?:(?:honestly|hold\s+on|wait|look|listen|i\s+mean|by\s+all\s+rights|"
    r"for\s+(?:the\s+)?(?:viewers|stream|chat|clip)(?:\s+at\s+home)?|"
    r"just\s+narrating|i'?m\s+narrating[^,:]*)[,:\s-]+)*"
    r"(?:"
    r"i\s+(?:should|shouldn'?t|wish|wished|can'?t|cannot|could|couldn'?t|would|"
    r"always|never|keep|kept|forget|forgot|hate\s+(?:when|that|how)|"
    r"love\s+(?:when|that|how)|was\s+(?:going|about|thinking)|"
    r"asked\s+(?:my\s+team|them|the\s+squad)|"
    r"'?m\s+the\s+(?:person|type|kind|one)|'?m\s+always)"
    r"\b"
    r"|should\s+i\b|shouldn'?t\s+i\b|do\s+i\b|why\s+do\s+i\b|when\s+do\s+i\b|"
    r"how\s+do\s+i\b|am\s+i\s+(?:supposed|the\s+only)\b"
    r"|if\s+only\s+i\b|every\s+time\s+i\b|whenever\s+i\b"
    r"|not\s+sure\s+(?:if|whether)\b|there'?s\s+no\s+point\b"
    r"|my\s+(?:biggest|whole|main|only)\s+(?:problem|issue|thing|"
    r"improvement|weakness|habit|flaw)\b"
    r")",
    re.IGNORECASE,
)


def recover_relay_lead(text: str) -> str:
    """Prepend the canonical "tell my team ..." lead when a clipped TEAM CALLOUT
    arrives without its relay verb. Returns ``text`` unchanged for anything that
    is already a relay lead, or that belongs to another route."""
    s = text.strip()
    if not s:
        return text
    if _HAS_RELAY_LEAD.match(s):
        return text  # already a valid relay/compose/soundboard lead
    if _NOT_A_CALLOUT.match(s):
        return text  # question / Spotify / identity / desktop -> leave verbatim
    if _REPORTED_RESPOND_RE.search(s):
        # "<agent> asked you about X, respond" -> the relay's context+directive
        # matcher AUTHORS an in-character Ultron answer. Do NOT prepend
        # "tell my team" (that would relay the QUESTION literally).
        return text
    # "I want my team to X" -> "tell my team X" (extract the directive so the
    # embedded addressee isn't doubled and the payload isn't lost).
    mw = _WANT_TEAM.match(s)
    if mw:
        rest = mw.group(1).strip()
        if rest:
            # "I want to tell my team X" is a relay intent -- UNLESS it's futility
            # musing ("...but they'll just stick it anyway", "...but I'm not sure").
            # Veto via the semantic gate (fail-open keeps today's behavior).
            if _NARRATION_MUSING_RE.match(rest) or relay_intent_ok(rest) is False:
                return text
            return "tell my team " + rest
    # Trailing relay command ("Viper wall is up, tell my team.") -> strip the
    # tail and prepend the canonical lead so the payload isn't duplicated.
    mt = _TRAILING_RELAY_TAIL.search(s)
    if mt:
        head = s[:mt.start()].strip(" ,.;:!?-")
        if head:
            return "tell my team " + head
    # No-verb team lead ("on my team X" / "to the squad X") -> restore "tell".
    mn = _TEAM_LEAD_NOVERB.match(s)
    if mn:
        rest = s[mn.end():].strip()
        if rest:
            return "tell my team " + rest
    if _TEAM_LEAD.match(s):
        # "my team X" / "the squad X" -> the verb was dropped; restore "tell".
        # Liberal here: a wake-addressed utterance that opens with the team is
        # almost always a relay (the negative gate already removed questions).
        return "tell " + s
    if _CALLOUT_SIGNAL.search(s) or _AGENT_SIGNAL.search(s):
        # Bare callout with no addressee at all ("there's a Jett A main",
        # "Chamber holding long", "I'm planting") -> address the team. BUT a
        # callout keyword alone is not enough: narration ("I should tell them to
        # eco"), banter/analysis at Ultron ("their Sage rez'd, how much does that
        # cost"), a question for advice ("push or hold"), or Marvel/identity talk
        # also contain these keywords. Veto those before attaching the team lead.
        if _NARRATION_MUSING_RE.match(s):
            return text  # first-person self-narration -> conversational
        verdict = relay_intent_ok(s)
        if verdict is False:
            return text  # semantic relay-intent gate vetoed -> conversational
        # verdict True (relay) or None (sidecar down -> keep keyword behavior)
        return "tell my team " + s
    return text


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def normalize_command(text: str) -> str:
    """Clean an STT transcript into a canonical command string for routing.

    Order: strip leading junk -> Valorant vocab correction -> relay-lead
    recovery. Cheap + idempotent; returns the input unchanged when nothing
    applies (so already-clean text and the test corpus are never altered in a
    way that changes routing)."""
    if not text or not text.strip():
        return text
    raw = text.strip()
    # A bare greeting is left VERBATIM -- never corrected (so "hello" can't snap
    # to the location "hell") and never relay-recovered. Checked on the RAW text
    # BEFORE leading-junk stripping, since "hey" is also a filler word that the
    # strip would otherwise consume ("hey there" -> "there"). It routes as a
    # greeting / conversational line, not a team callout.
    if _BARE_GREETING.match(raw):
        return raw
    s = _strip_leading_junk(raw)
    # Repair a mis-heard verbatim verb + drop a team-addressee possessive BEFORE
    # routing, so "Pete to my team X" relays X verbatim and "my team's X" strips
    # its lead cleanly.
    s = _REPEAT_MISHEAR.sub(r"\1repeat", s, count=1)
    s = _TEAM_POSSESSIVE.sub(r"\1", s)
    _wfw = _WORD_FOR_WORD.match(s)
    if _wfw and _wfw.group(1).strip():
        s = "say exactly to my team " + _wfw.group(1).strip()
    # Resolve a mid-utterance self-correction to its final intent, and strip a
    # possessive before a roster agent ("relay to my Sova" -> "relay to Sova"),
    # both BEFORE routing.
    s = _resolve_disfluency(s)
    s = _strip_possessive_names(s)
    # ZERO-MISTAKES GATE: conversational / Spotify / identity / desktop commands
    # are left VERBATIM -- the aggressive Valorant vocab correction (phonetic +
    # fuzzy) runs ONLY on callout-bound text, so a question or a song title is
    # never corrupted into agent names. Everything that ISN'T clearly one of
    # those routes is treated as a team callout (the primary wake-addressed use)
    # and gets corrected + lead-recovered.
    if (_NOT_A_CALLOUT.match(s) or _SPOTIFY_SIGNAL.search(s)
            or _REPORTED_QUESTION_GATE.match(s)):
        return s
    s = correct_callout_stt(s)
    s = recover_relay_lead(s)
    return s.strip()
