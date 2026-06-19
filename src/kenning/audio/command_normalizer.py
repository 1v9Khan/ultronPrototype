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
from kenning.audio._ultron_social import classify_social_reaction
from kenning.audio._ultron_answer import THINK_RESPOND_SUFFIX_RE

# ---------------------------------------------------------------------------
# 1. Leading junk: misheard wake word + conversational filler.
# ---------------------------------------------------------------------------
# Wake-word homophones the STT prepends ("Ultron"->Run/Ron/Tron/One/...), plus
# disfluencies and lead-ins people say before the real command. Stripped
# iteratively from the FRONT only. We never strip the entire utterance (if the
# strip would empty it, keep the original) so a bare "okay" still survives.
_WAKE_HOMOPHONES = (
    r"ultron|ulltron|ultronn|ultran|ultram|altron|voltron|ultra|ultro|"
    r"ultr|oltron|ultraun|tron|ron|run|rons"
)
_FILLER = (
    r"hey|ok|okay|um+|uh+|er+|hmm+|so|well|yeah|yep|yup|now|and|then|"
    r"please|alright|right|i\s+mean|i\s+think|i\s+hope|i\s+guess|i\s+wanna|"
    r"i\s+want\s+to|let'?s\s+see|you\s+know|basically|just|"
    # 2026-06-16 (C6): broadened edit-term / interregnum lexicon. These are
    # FIXED-PHRASE disfluencies, never tactical content.
    r"y'?know|ya\s+know|kind\s+of|kinda|sort\s+of|sorta|ugh+|mmk+|mhm+|hmph|"
    r"like\s+i\s+said|i\s+want\s+to\s+say|i\s+wanna\s+say|i\s+gotta\s+say|"
    # "man" is a discourse marker ONLY when comma/pause-followed ("man, we lost")
    # -- NEVER strip the callout noun in "man down" / "man advantage" (C6 R5).
    r"man(?=\s*,)|"
    # conversational address-fillers that leak before a relay lead ("bro relay X",
    # "dude tell them Y", "yo call out Z") -- safe to strip from the front.
    r"bro|bruh|dude|homie|fam|bud|buddy|guys|yo"
)
# "like" is filler ("like, tell my team X") BUT also the Spotify verb
# ("like this song" / "like it"). Strip it as filler ONLY when it is NOT
# immediately followed by a Spotify object -- otherwise the leading-junk pass
# turned "like this song" into "this song", which then matched "now playing".
_LIKE_FILLER = r"like(?!\s+(?:this|that|it|the|some|my)\b)"

# 2026-06-16 (C6 CHANGE 2): conversational SAY-DIRECTIVE lead-ins ("can you say
# X", "might be worth saying X", "I want you to say X", "hurry say X"). These are
# interregnum-class scaffolding that PRECEDE a real relay payload -- stripped and
# RE-FRAMED to "tell my team X" by _strip_scaffold (3c), never blindly dropped.
_SAY_DIRECTIVE = (
    r"can\s+you\s+(?:please\s+)?(?:say|tell|relay)|"
    r"could\s+you\s+(?:please\s+)?(?:say|tell|relay)|"
    r"(?:might\s+be|it'?d\s+be|would\s+be)\s+worth\s+(?:saying|telling)|"
    r"(?:you\s+)?(?:probably\s+)?should\s+(?:say|tell|relay|let)|"
    r"i\s+want\s+you\s+to\s+(?:say|tell|relay|let)|"
    r"i\s+need\s+you\s+to\s+(?:say|tell|relay|let)|"
    r"hurry\s+(?:up\s+(?:and\s+)?)?(?:say|tell|relay)|"
    r"make\s+sure\s+(?:to|you)\s+(?:say|tell|relay)|"
    r"go\s+ahead\s+and\s+(?:say|tell|relay)|"
    r"do\s+me\s+a\s+favor\s+and\s+(?:say|tell|relay)"
)
_SAY_DIRECTIVE_LEAD = re.compile(rf"^\s*(?:{_SAY_DIRECTIVE})\b[\s,:.]*", re.IGNORECASE)

# 2026-06-16 (C5/I48): performative relay-WRAPPER leads that do NOT begin with a
# recognised relay verb, so they otherwise fall to no_match ("make sure my team
# knows X", "let them know X", "give the team the heads up that X", "pass along
# that X", "shout out that X"). Reframed to "tell my team X" by _strip_scaffold
# (3c) so the payload routes. ANCHORED on the trailing "knows/know/that" object so
# a real callout ("make sure my team rotates") is never rewritten.
_WG = (r"(?:my\s+|our\s+|the\s+)?(?:whole\s+|entire\s+)?(?:team|teammates?|squad|"
       r"boys|guys|crew|fellas|lads|homies|gang|mates|fam|everyone|everybody|"
       r"them|'?em)")
_WRAPPER_LEAD_RE = re.compile(
    r"^\s*(?:bro|yo|ok(?:ay)?|hey|alright|please)?[\s,]*"
    r"(?:"
    rf"make\s+sure\s+{_WG}\s+knows?(?:\s+that)?"
    rf"|let\s+{_WG}\s+know(?:\s+that)?"
    rf"|give\s+{_WG}\s+(?:the\s+|a\s+)?heads[\s-]?up\s+that"
    r"|shout(?:\s+out)?\s+that"
    r"|pass(?:\s+(?:along|on|it))?\s+that"
    r")\s+",
    re.IGNORECASE,
)
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


# ---------------------------------------------------------------------------
# 1b. Mangled / doubled relay-verb lead canonicalization (2026-06-17 battery).
# ---------------------------------------------------------------------------
# STT routinely mis-hears the relay verb "tell" as a near-homophone or a
# plausible-but-wrong word ("Call/Hold/Help/Build/Follow/Kill/While/How/Put/
# Without my|the team ..."), or the streamer recounts the relay ("I told my
# team X", "that's the team ... X", "this is the team that X"). The downstream
# relay matcher only knows a fixed verb set, so a mangled lead EITHER leaks
# verbatim into the spoken line ("Call my team, we need smokes. The line holds.")
# OR, with no callout keyword, falls through to the desktop LLM and is never
# relayed at all (the single biggest failure mode in the 6-17 battery, ~45 cmds).
# Canonicalize ANY such lead -- including a SECOND one stacked in the payload
# ("tell my team Call my team X") -- to a single "tell my team " lead BEFORE the
# scaffold/gate pipeline. Anchored strictly on "<word> (my|the|a) team" (or the
# explicit irregular recounts) so ordinary speech is never touched.
# 2026-06-18 Part B: the lead-recognition RULES (team-noun word list, mangled-
# "tell" mishear list, tell-class verbs, and their lead regexes) are relocated to
# the routing aggregate kenning.audio.routing_rules (Section 2) -- edit them
# THERE. Imported here (aliased to the existing private names); the consuming
# functions below are UNCHANGED. Behaviour byte-for-byte identical (proven by
# scripts/_voice_lines_verify.py via the regex .pattern digests).
from kenning.audio.routing_rules import (  # noqa: E402
    NORM2_TEAM_NOUN as _TEAM_NOUN,
    NORM2_MANGLED_TELL as _MANGLED_TELL,
    NORM2_TELL_CLASS_VERB as _TELL_CLASS_VERB,
    NORM2_MANGLED_TEAM_LEAD_RE as _MANGLED_TEAM_LEAD_RE,
    NORM2_IRREGULAR_TEAM_LEAD_RE as _IRREGULAR_TEAM_LEAD_RE,
    NORM2_TELL_TEAM_LEAD_RE as _TELL_TEAM_LEAD_RE,
)
_ANY_TEAM_LEAD_OUTER_RE = re.compile(
    rf"^\s*(?:please\s+)?(?:{_TELL_CLASS_VERB}|ask|relay(?:\s+to)?)\s+(?:to\s+)?"
    rf"(?:my\s+|our\s+|the\s+)?{_TEAM_NOUN}\b(?:\s+know)?[\s,:.]*",
    re.IGNORECASE,
)


def _strip_stacked_team_leads(p: str) -> str:
    """Strip leading tell-class / mangled / irregular team leads from ``p`` (the
    payload after an outer lead), looping so a doubled lead clears. Only removes a
    lead that is itself "<verb> <det> team", never ordinary tactical content."""
    for _ in range(3):
        for rx in (_TELL_TEAM_LEAD_RE, _MANGLED_TEAM_LEAD_RE,
                   _IRREGULAR_TEAM_LEAD_RE):
            m = rx.match(p)
            if m and p[m.end():].strip():
                p = p[m.end():].lstrip()
                break
        else:
            break
    return p


# Bare "ask <question>" with NO addressee ("ask how they usually hit C") -> the
# streamer means ask the TEAM. Without an addressee the relay matcher's ASK form
# abstains to desktop; inject "my team" so it relays as a team question
# (2026-06-17 [154]).
_BARE_ASK_RE = re.compile(
    r"^\s*ask\s+(?=(?:if|whether|how|why|where|when|what|who|which|does|do|did|"
    r"are|is|can|could|should|would|will|has|have)\b)",
    re.IGNORECASE,
)
# "tell/ask someone to X" -> a team relay ("Someone take this Sheriff"). "someone"
# is not a roster name so the named matcher abstained to desktop (2026-06-17 [222]).
_SOMEONE_LEAD_RE = re.compile(
    r"^\s*(?:tell|ask|let|have|get)\s+someone\s+(?:to\s+)?",
    re.IGNORECASE,
)
# "give my team to <imperative>" is a "tell"->"give" STT mishear (live "give my
# team to rush mid" echoed literally). The trailing "to <verb>" disambiguates it
# from the COMPOSE form "give my team encouragement" (no "to"), which must stay.
_GIVE_TEAM_TO_RE = re.compile(
    r"^\s*give\s+(?:my|the|our)\s+team\s+to\s+", re.IGNORECASE,
)
# "ask Iso to drop me HIS sheriff" -> Ultron is asking the agent to drop ULTRON
# one of THEIR guns, so the possessive is "your", not "his/her/their" (live:
# "drop me his Sheriff" echoed "his" in flavor-ON). Normalized so BOTH flavor
# states render "your".
_DROP_POSSESSIVE_RE = re.compile(
    r"\bdrop\s+me\s+(?:his|her|their|its)\b", re.IGNORECASE,
)
# Run-together command lead from fast speech ("Tellmyteam"/"ask my team" with
# the spaces dropped). \s* (zero-or-more) catches the no-space form; \bteam\b
# keeps "tell my teammate" safe.
_RUNON_TEAM_LEAD_RE = re.compile(
    r"^\s*(tell|told|ask)\s*(?:my|the|our)\s*team\b[\s,:.]*", re.IGNORECASE,
)


def _canonicalize_directive_lead(s: str) -> str:
    """Rewrite a mangled / doubled relay-verb lead to a single canonical
    "tell my team " (or preserve a valid ask/relay verb), stripping any stacked
    junk lead. Returns ``s`` unchanged when no team-directed lead is present."""
    s = s.lstrip()
    if not s:
        return s
    # (a) outer lead is already a valid team verb (tell/ask/say/...) -> keep the
    #     verb, strip any junk lead stacked in the payload.
    m = _ANY_TEAM_LEAD_OUTER_RE.match(s)
    if m:
        payload = _strip_stacked_team_leads(s[m.end():])
        return s[:m.end()] + payload if payload.strip() else s
    # (b) outer lead is mangled / irregular -> rewrite to "tell my team " and
    #     strip any further stacked lead.
    m = _MANGLED_TEAM_LEAD_RE.match(s) or _IRREGULAR_TEAM_LEAD_RE.match(s)
    if m:
        payload = _strip_stacked_team_leads(s[m.end():])
        if payload.strip():
            return "tell my team " + payload
    return s


# ---------------------------------------------------------------------------
# 2026-06-16 (C6): scaffold strip -- numbered prefixes, say-directive lead-ins,
# a nested relay verb, and embedded fillers. The discourse scaffolding that
# leaks into the relayed line (often reordered to the END by the 3B). Runs
# BEFORE the disfluency resolver and the ZERO-MISTAKES gate so questions /
# Spotify / musings are still seen intact downstream.
# ---------------------------------------------------------------------------
_NUMBERED_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"(?:number\s+)?\d+\s*[.):,-]\s+"                  # "1. " "2) " "3, "
    r"|(?:first(?:ly)?|second(?:ly)?|third(?:ly)?)\s*[,.:-]\s+"
    r")",
    re.IGNORECASE,
)
# Spelled "one,"/"two," prefix -- a comma is REQUIRED so the count callout
# "two on B" (no comma) is never stripped. Only applied when the remainder is
# clearly a relay (guarded at the call site).
_NUMBERED_WORD_PREFIX_RE = re.compile(
    r"^\s*(?:one|two|three)\s*,\s+(?=\w)", re.IGNORECASE,
)
# A nested relay verb inside an already-relay-framed utterance ("tell my team
# that -- tell them -- rotate B"): strip the INNER verb, keep ONE outer frame.
_NESTED_RELAY_VERB_RE = re.compile(
    r"\b(?:tell|let|say(?:\s+to)?|relay(?:\s+to)?|inform|warn|remind|ask)\s+"
    r"(?:to\s+)?(?:my\s+|our\s+|the\s+)?"
    r"(?:team(?:mates?)?|squad|boys|guys|mates|crew|fellas|homies|'?em|them)"
    r"(?:\s+know)?(?:\s+(?:that|to))?\s+",
    re.IGNORECASE,
)
# Embedded standalone disfluencies that wedge mid-line. RESTRICTED to tokens
# that are NEVER tactical content (no bare so/well/like/man/right/now). Applied
# in a loop until stable so adjacent fillers ("-- so um -- kind of --") clear.
_EMBEDDED_FILLER_RE = re.compile(
    r"(?:^|[\s,;.-])\s*(?:--+\s*)?\b(?:uh+|um+|er+|erm|hmm+|ugh+|mmk+|y'?know|"
    r"ya\s+know|kind\s+of|kinda|sort\s+of|sorta|like\s+i\s+said)\b\s*(?:--+)?[\s,;]*",
    re.IGNORECASE,
)


def _is_protected_scaffold_remainder(remainder: str) -> bool:
    """R2: a say-directive reframe must NOT fire when the remainder is itself a
    question / reported-question / reaction / think-respond / Spotify / musing --
    re-framing those to a literal "tell my team ..." relay would mangle them
    (e.g. "can you say Reyna asked about Iron Man")."""
    return bool(
        _NOT_A_CALLOUT.match(remainder)
        or _REPORTED_QUESTION_GATE.match(remainder)
        or _REPORTED_RESPOND_RE.search(remainder)
        or _REPORTED_REACTION_RE.match(remainder)
        or THINK_RESPOND_SUFFIX_RE.search(remainder)
        or _SPOTIFY_SIGNAL.search(remainder)
        or _NARRATION_MUSING_RE.match(remainder)
    )


def _strip_scaffold(s: str) -> str:
    """Strip numbered prefixes, say-directive lead-ins, a nested relay verb, and
    embedded fillers. Idempotent on clean text (every sub-step no-ops when its
    pattern is absent)."""
    consumed_lead = False
    # (3a) numbered prefix ("1. ", "2) ", "first, ")
    new = _NUMBERED_PREFIX_RE.sub("", s, count=1)
    if new != s:
        s, consumed_lead = new.lstrip(), True
    else:
        m = _NUMBERED_WORD_PREFIX_RE.match(s)
        if m:
            rest = s[m.end():]
            if (_HAS_RELAY_LEAD.match(rest) or _TEAM_LEAD.match(rest)
                    or _SAY_DIRECTIVE_LEAD.match(rest)):
                s, consumed_lead = rest.lstrip(), True
    # (3c) say-directive / relay-wrapper lead-in -> reframe to a team relay
    # (R2-gated so a reported-question / Spotify / musing is never mangled).
    m = _SAY_DIRECTIVE_LEAD.match(s) or _WRAPPER_LEAD_RE.match(s)
    if m:
        remainder = s[m.end():].strip()
        if remainder and not _is_protected_scaffold_remainder(remainder):
            # Use the remainder AS-IS only when it is genuinely already a relay
            # command -- a team-noun lead, OR a relay verb that is NOT just a bare
            # ambiguous tactical imperative (drop/give/share/call X). Otherwise
            # prepend the canonical lead so the callout is relayed (F1 fix).
            already_relay = _TEAM_LEAD.match(remainder) or (
                _HAS_RELAY_LEAD.match(remainder)
                and not _AMBIG_TACTICAL_LEAD.match(remainder)
            )
            if already_relay:
                s = remainder
            else:
                s = "tell my team " + remainder
            consumed_lead = True
    # (3b) nested relay verb -- only when an OUTER relay frame already exists, so
    # a single legitimate "tell my team X" is never stripped to bare "X".
    # A team noun that is the SUBJECT of a reported clause ("my teammate is
    # flaming me, tell them to calm down") is CONTEXT, not an outer relay frame,
    # so it must NOT enable the nested-verb strip below (which would delete the
    # real "tell them ..." directive). F3.
    had_outer = consumed_lead or bool(
        _HAS_RELAY_LEAD.match(s)
        or (_TEAM_LEAD.match(s) and not _TEAM_AS_SUBJECT_RE.match(s)))
    if had_outer:
        lead_m = _HAS_RELAY_LEAD.match(s) or _TEAM_LEAD.match(s)
        start = lead_m.end() if lead_m else 0
        head, tail = s[:start], s[start:]
        new_tail = _NESTED_RELAY_VERB_RE.sub("", tail, count=1)
        if new_tail != tail:
            s = (head + new_tail).strip()
            if s and not _HAS_RELAY_LEAD.match(s) and not _TEAM_LEAD.match(s):
                s = "tell my team " + s
    # (3d) embedded fillers -- loop until stable (R4) so adjacent ones all clear.
    for _ in range(4):
        new = re.sub(r"\s{2,}", " ", _EMBEDDED_FILLER_RE.sub(" ", s)).strip(" ,;")
        if new == s:
            break
        s = new
    return s


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
    # 2026-06-16 (C6 CHANGE 4): MULTI-WORD self-correction edit-terms only --
    # never bare "no"/"wait"/"not"/"or" (those are tactical and must survive).
    r"|\bwell\s+no\b|\bno\s+actually\b|\bi\s+don'?t\s+know\b"
    r"|--+\s*to\s+(?:all|the)\b)",
    re.IGNORECASE,
)
# Boundaries to split on once the utterance is flagged as a repair: every cue
# above PLUS bare "--". Keep the segment after the LAST boundary (the final
# intended command). Ordered so multi-word cues match before the bare dash.
_DISFLUENCY_SPLIT_RE = re.compile(
    r"(?:--+\s*(?:no\s+)?wait\b|\bno\s+wait\b|\bno\s+no\b|\bscratch\s+that\b"
    r"|\bnever\s*mind\b|\bforget\s+it\b|\bactually\s+no\b|\bor\s+rather\b"
    r"|\bi\s+mean\b|\bwell\s+no\b|\bno\s+actually\b|\bi\s+don'?t\s+know\b"
    r"|--+\s*to\s+(?:all|the)\s+\w+|--+\s*actually\b|--+\s*no\b"
    r"|--+)"
    r"[\s,.:;\-]*",
    re.IGNORECASE,
)


# 2026-06-16 (C6 CHANGE 4 + R1): bare same-class value-swap repair, with NO
# explicit cue word. ONLY fires on a literal weapon-/item-request head-verb
# repeat ("drop Vandal -- drop Phantom") or an economy buy-class repeat ("full
# buy -- half buy", "eco -- save") across a "--" boundary. Deliberately does NOT
# trigger on LOCATION or COUNT repetition (R1: "rotate mid -- then push main",
# "two on A -- one on B", "one long -- watch short too" are legitimate
# sequential split-info callouts, not corrections) -- those keep BOTH halves.
_REPAIR_HEAD_VERBS = frozenset({"drop", "buy", "get", "grab", "pick"})
_BUY_CLASS_RE = re.compile(
    r"\b(?:full\s+buy|half\s+buy|light\s+buy|thrifty\s+buy|force(?:\s+buy)?"
    r"|forcing|eco|save)\b",
    re.IGNORECASE,
)


def _resolve_value_swap(s: str) -> str:
    if "--" not in s:
        return s
    parts = [p.strip(" ,.;:-") for p in re.split(r"\s*--+\s*", s)
             if p.strip(" ,.;:-")]
    if len(parts) < 2:
        return s
    head0 = parts[0].lower().split()
    headL = parts[-1].lower().split()
    head_repeat = bool(
        head0 and headL and head0[0] == headL[0]
        and head0[0] in _REPAIR_HEAD_VERBS)
    buy_repeat = bool(_BUY_CLASS_RE.search(parts[0])
                      and _BUY_CLASS_RE.search(parts[-1]))
    if not (head_repeat or buy_repeat):
        return s
    tail = parts[-1]
    if (_HAS_RELAY_LEAD.match(parts[0]) or _TEAM_LEAD.match(parts[0])) and not (
            _HAS_RELAY_LEAD.match(tail) or _TEAM_LEAD.match(tail)):
        return "tell my team " + tail
    return tail


def _resolve_disfluency(s: str) -> str:
    """Resolve a mid-utterance self-correction to its FINAL intended command,
    preserving the relay lead. "call out Iso contract -- wait shield -- Double
    Tap, he has shield up" -> "tell my team Double Tap, he has shield up". Only
    fires when an explicit correction cue is present, so ordinary callouts (incl.
    tactical "rotate B not A" / "wait for the molly") are never touched. With NO
    explicit cue, falls through to the bare same-class value-swap (C6 CHANGE 4)."""
    if not _DISFLUENCY_CUE_RE.search(s):
        return _resolve_value_swap(s)
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


# Multi-addressee ("relay to Sage and Clove: X" / "tell Jett and Sova X") -- the
# named matcher takes ONE teammate, so a two-name addressee was no_match. Since
# the message goes to both, collapse a leading "<verb> [to] <name> and <name>"
# to a team relay so the payload is delivered. Both sides must be ROSTER names so
# "tell Sage and push B" (name + action) is untouched.
_ROSTER_ALT = "|".join(sorted((re.escape(a.replace("/", "")) for a in _AGENTS),
                              key=len, reverse=True))
_MULTI_ADDR_RE = re.compile(
    r"^(\s*(?:please\s+)?(?:tell|warn|inform|remind|relay\s+to|ask)\s+)"
    r"(?:my\s+|our\s+|the\s+)?(?:" + _ROSTER_ALT + r")\s+and\s+"
    r"(?:my\s+|our\s+|the\s+)?(?:" + _ROSTER_ALT + r")\b\s*[:,]?\s*",
    re.IGNORECASE,
)


def _collapse_multi_addressee(s: str) -> str:
    return _MULTI_ADDR_RE.sub(lambda m: f"{m.group(1)}my team ", s, count=1)


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
    r"roast|flame|give|share|drop|compliment|praise|gas|prop|props|"
    r"criticize|criticise|critique|rip\s+into|tear\s+into|chew\s+out)\b",
    re.IGNORECASE,
)

# A remainder that merely STARTS with a relay-AMBIGUOUS tactical verb
# (drop/give/share/call) but does NOT address a group -- i.e. it's a tactical
# PAYLOAD ("drop spike on me", "give me a gun", "share credits"), not a nested
# relay command. Used to refine the wrapper reframe: without this, "let my team
# know drop spike on me" sees the remainder "drop spike on me" match
# _HAS_RELAY_LEAD (on "drop"), so the reframe used it AS-IS and dropped the
# "tell my team" prepend -> the relay was MISSED. "call out X", "drop the team
# X", "give everyone X" DO address a group and are excluded here (still treated
# as already-led). 2026-06-18 corpus audit F1.
_AMBIG_TACTICAL_LEAD = re.compile(
    r"^\s*(?:drop|give|share|call)\b"
    r"(?!\s+(?:out\b|to\b|them\b|'?em\b|everyone\b|everybody\b|"
    r"(?:my|our|the)\s+(?:team|teammates?|squad|boys|guys|mates|crew|gang|fam)\b))",
    re.IGNORECASE,
)

# A team noun used as the SUBJECT of a reported clause ("my teammate is flaming
# me", "the squad keeps dying", "my team just lost") -- context, NOT an address
# lead. Distinguished by a copula / auxiliary / reporting verb right after the
# noun. Used so _strip_scaffold's nested-verb strip does NOT treat such a context
# clause as an outer relay frame and delete the REAL relay directive that follows
# ("my teammate is flaming me, tell them to calm down"). 2026-06-18 audit F3.
_TEAM_AS_SUBJECT_RE = re.compile(
    r"^\s*(?:my\s+|our\s+|the\s+)?"
    r"(?:team|teammates?|squad|boys|guys|mates|crew|duo)\s+"
    r"(?:is|are|was|were|'?s|'?re|keeps?|kept|just|already|has|have|had|"
    r"wants?|wanted|said|says|asked|asking|started|stopped|won'?t|"
    r"can'?t|cannot|gets?|got|been|being|\w+ing)\b",
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
    r"thinks?|thinking|typed|wrote|mention\w*|brought\s+up|rais(?:ed|es|ing)|"
    r"talking\s+about|talked\s+about|complain\w*|crying|flam\w*|tilted|raging|"
    r"griefing|grief\w*|losing\s+it|losing\s+their\s+(?:mind|cool)|melting\s+down|"
    r"upset|mad|angry|heated|"
    r"malding|mock\w*|teas\w*|roast\w*|clown\w*|diss\w*|ridicul\w*|insult\w*|"
    r"bully\w*|making\s+fun|trash[\s-]?talk\w*|call(?:ed|ing|s)\s+(?:me|you|us))\b.*"
    r"\b(?:respond|reply|answer|acknowledge|agree|clap\s+back|back\s+me\s+up|"
    r"defend\s+me|set\s+(?:him|her|them)\s+straight|calm\s+(?:him|her|them)\s+down|"
    r"de[\s-]?escalate(?:\s+(?:him|her|them))?|talk\s+(?:him|her|them)\s+down|"
    r"ease\s+(?:him|her|them)\s+(?:off|up|down)|"
    r"shut\s+(?:him|her|them|it)\s+down|"
    r"say\s+something|handle\s+(?:it|that|him|her|them)|"
    r"deal\s+with\s+(?:it|that|him|her|them))\b\s*[.!?]*$",
    re.IGNORECASE,
)

# A reported SOCIAL statement with no explicit directive ("Jett said nice shot",
# "Yoru called you stupid", "the team is flaming you", "Miks is saying gg", "the
# team is giving up") -- left VERBATIM (no "tell my team" prefix, no Valorant
# vocab correction) so relay_speech._match_reported_reaction can author an
# in-character reaction. CONJUNCTIVE with classify_social_reaction at the call
# site, so a tactical callout that shares the frame ("Jett said two on B") is
# NOT gated and relays normally.
_REPORTED_REACTION_RE = re.compile(
    r"^\s*(?:my\s+|our\s+|the\s+)?\w+(?:\s+\w+)?\s+"
    r"(?:just\s+|is\s+|are\s+|was\s+|been\s+|keeps?\s+|wants?\s+to\s+)?"
    r"(?:said|says|saying|say|told|tells|telling|called|calling|calls|"
    r"thinks?|thinking|typed|wrote|insult(?:ed|ing|s)?|flam(?:e|ed|ing|es)?|"
    r"mock(?:ed|ing|s)?|clown(?:ed|ing|s)?|diss(?:ed|ing|es)?|roast(?:ed|ing|s)?|"
    r"trash[\s-]?talk\w*|mak(?:ing|es)\s+fun|made\s+fun|giv(?:ing|in'?)\s+up|"
    r"gave\s+up|being\s+(?:toxic|mean|rude)|complain\w*|compliment(?:ed|ing|s)?|"
    r"prais(?:e|ed|ing|es)|hyp(?:e|ed|ing|es)|throw(?:ing|n)?\s+in\s+the\s+towel|"
    r"forfeit\w*|surrender\w*)\b",
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
    r"rotating|left|right|mid|main|here|there|a\b|b\b|c\b|"
    # 2026-06-17: count + position/location word ("one back plat", "two cat",
    # "one close left", "one window", "three garage") -- these fell to desktop.
    r"back|close|far|deep|up|down|top|bottom|behind|window|garage|cat|plat|"
    r"platform|sewer|sewers|grass|hell|heaven|site|long|short|lurk|flank|"
    r"sneaky|wide|tight|connector|link|nest|snake|bridge|boba|pizza|tree|"
    r"more|of\s+them)|"
    # 2026-06-17: "there are <count> <loc>" / bare "<count> <loc>" reached here.
    r"there\s+(?:are|is)\s+(?:one|two|three|four|five|six)\b|"
    # 2026-06-17: sound callouts ("I hear one cat", "footsteps long", "I hear
    # lots grass", "I hear some sewers") -- enemy audio info, never desktop.
    r"i\s+hear|i\s+heard|i\s+can\s+hear|hear\s+(?:one|two|three|lots|some|"
    r"footsteps|a)|footsteps|"
    # 2026-06-17: enemy comp / economy state ("they have no smokes", "they
    # bought", "they saved", "the enemy needs to save", "they have 3 duelists").
    r"(?:they|enemy|the\s+enemy)\s+(?:have|has|had|bought|saved|sold|forced|"
    r"need|needs|are\s+saving|are\s+forcing|never|always|crossed|tripped|"
    r"could\s+be|might\s+be|may)\b|"
    r"initiator|initiators|sentinel|sentinels|duelist|duelists|controller|"
    r"controllers|flashes|"
    # locations
    r"heaven|hell|main|mid|middle|long|short|window|site|spawn|market|"
    r"garage|hookah|connector|tree|elbow|ramp|pit|rafters|generator|"
    r"cubby|catwalk|alley|courtyard|stairs|lamps|showers|kitchen|dish|"
    r"snowman|tube|tubes|vents|tower|boathouse|link|lobby|default\b|logs|"
    r"sewer|sewers|grass|plat|platform|nest|snake|bridge|boba|pizza|"
    # self status
    r"low|one\s+shot|one-shot|reloading|i\s+died|i'?m\s+dead|i'?m\s+low|"
    r"i'?m\s+planting|i'?m\s+flanking|i'?m\s+pushing|i'?m\s+rotating|"
    r"i'?m\s+holding|i'?m\s+going|i\s+have\s+(?:a|b|c|site|the\s+spike)|"
    # orders / morale / social
    r"save|saved|eco|force|forced|forcing|bought|retake|default|stack|stacked|"
    r"stacking|group\s+up|bonus|"
    r"buy|buying|bonus\s+round|fall\s+back|lock\s+in|split|execute|executing|"
    r"rush|rushing|rushed|"
    r"good\s+game|gg|nice|great\s+play|well\s+played|let'?s\s+go|we\s+got\s+this|"
    r"good\s+luck|my\s+bad|nice\s+shot|clutch|careful|watch\s+the|"
    r"winning|we'?re\s+winning|we\s+lost|good\s+round|great\s+round|nice\s+round|"
    # 2026-06-15 expanded callout coverage (relay without an explicit "tell my
    # team" lead): counts/kills, weapons, movement, requests, clears.
    r"down|left|clear|cleared|going|crossing|crossed|behind|boost|boosting|"
    r"anchor|anchoring|switch|switching|cover|covering|picks|trade|traded|"
    r"baiting|baited|bait\b|sticking|stuck|off\s+spike|"
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

# 2026-06-17: STRONG, unambiguous callout SHAPES that must relay even when the
# semantic relay-intent gate abstains (the gate was vetoing legitimate sound /
# comp / count callouts -- "I hear some sewers", "they have three duelists",
# "one back plat" -- to desktop). These shapes are NEVER stream-narration
# (those are caught by _NARRATION_MUSING_RE first), so they bypass the gate and
# relay directly. Start-anchored so they describe the WHOLE utterance's intent.
_STRONG_CALLOUT_RE = re.compile(
    r"^\s*(?:"
    # sound info ("I hear one cat", "hear one B", "footsteps long")
    r"(?:i\s+(?:can\s+)?)?hear\b|i\s+heard\b|footsteps\b|"
    # enemy comp / economy / movement state
    r"(?:they|they're|the\s+enemy|enemy|enemies)\s+(?:have|has|had|got|bought|"
    r"saved|sold|forced|forcing|are\s+saving|are\s+forcing|need|needs|never|"
    r"always|crossed|cross|tripped|will|won'?t|are\s+all|are\s+off|could\s+be|"
    r"might\s+be|may\s+be|wrapped|wrapping|re-?hit|committing|splitting|"
    # 2026-06-18 user request: "they're out" (enemy committed / out on site) and
    # "they're not out" -- enemy-commitment status. Optional copula + optional
    # negation; "out\b" only matches standalone "out" (never "outside"/"outnumbered").
    r"(?:are\s+|is\s+)?(?:not\s+)?out)\b|"
    # count / there-are + a following word ("one back plat", "two cat",
    # "there are two cat", "one close left", "there are 2 cat")
    r"(?:there\s+(?:are|is)\s+)?(?:one|two|three|four|five|six|\d+)\s+\w|"
    # agent-led callout: a roster name (optionally possessed) + a tactical keyword
    # is an enemy spotting/utility/status/gripe, NEVER a musing -- the semantic
    # gate was wrongly abstaining "Reyna and Sage B main" / "Raze ulted" / "my
    # Jett is baiting me" to desktop.
    r"(?:my\s+|their\s+|our\s+|the\s+)?"
    r"(?:" + "|".join(re.escape(a.replace("/", "")) for a in _AGENTS) + r")\b"
    r".*\b(?:main|site|long|short|mid|heaven|hell|window|garage|tree|cat|plat|"
    r"connector|link|ramp|market|sewer|spawn|cubby|elbow|pit|rafters|stairs|"
    r"ult|ulted|ulting|walled|smoked|flashed|darted|caged|stunned|droned|"
    r"naded|mollied|half|low|one\s+shot|dead|down|cracked|tree|nest|snake|"
    r"baiting|baited|baits|flanking|lurking|peeking|rotating|pushing)\b|"
    # spike status
    r"spike\b"
    r")",
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
    # 2026-06-18 F5: leading stream/chat ADDRESS + thinking-aloud framings.
    r"chat|stream|to\s+the\s+(?:viewers|stream|chat)|"
    r"(?:processing|thinking|musing|narrating)\s+(?:out\s+loud|aloud)(?:\s+here)?|"
    r"talking\s+to\s+(?:the\s+)?(?:stream|chat|viewers|myself)(?:\s+here)?|"
    r"just\s+narrating|i'?m\s+narrating[^,:]*)[,:\s-]+)*"
    r"(?:"
    r"i\s+(?:should|shouldn'?t|wish|wished|can'?t|cannot|could|couldn'?t|would|"
    r"always|never|keep|kept|forget|forgot|hate\s+(?:when|that|how)|"
    r"love\s+(?:when|that|how)|was\s+(?:going|about|thinking)|"
    # 2026-06-18 F5: first-person PAST recount ("I told my team/squad/guys X ...
    # and they ...") + intent-musing ("I'd tell ...", "I've been trying/wanting
    # to ...", "I'm going to tell ..."). A recount/announced-intent is narration.
    r"told\s+(?:my\s+(?:team|teammates?|squad|duo|boys|guys|lads|crew|mates)|"
    r"them|the\s+(?:team|squad|guys|lads|boys|crew))|"
    r"asked\s+(?:my\s+team|them|the\s+squad)|"
    r"'?ve\s+been\s+(?:trying|wanting|meaning)|'?m\s+(?:gonna|going\s+to)\s+tell|"
    r"'?d\s+(?:tell|ask|let|say|warn|remind)\b|"
    r"'?m\s+the\s+(?:person|type|kind|one)|'?m\s+always)"
    r"\b"
    r"|should\s+i\b|shouldn'?t\s+i\b|do\s+i\b|why\s+do\s+i\b|when\s+do\s+i\b|"
    r"how\s+do\s+i\b|am\s+i\s+(?:supposed|the\s+only)\b"
    r"|if\s+only\s+i\b|every\s+time\s+i\b|whenever\s+i\b"
    r"|not\s+sure\s+(?:if|whether)\b|there'?s\s+no\s+point\b"
    # 2026-06-18 F5: general-statement / detached-musing frames that merely
    # MENTION telling the team ("part of me wants to ...", "one side of me says
    # ...", "one of my biggest weaknesses is ...", "one of these days ...", "the
    # meta is to tell ...", "great controllers tell their team ...", "there's no
    # one to tell ...") -- none is a live first-person relay command.
    r"|part\s+of\s+me\b|one\s+side\s+of\s+me\b|one\s+of\s+(?:my|these|the)\b"
    r"|the\s+(?:meta|play|move|right\s+call|smart\s+thing)\s+(?:right\s+now\s+)?"
    r"(?:is|here\s+is)\b"
    r"|(?:great|good|smart|pro)\s+(?:controllers|players|igls?|teams?)\b"
    r"|there'?s\s+(?:no\s+one|nobody)\s+to\b"
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
    # STRONG unambiguous callout shapes (sound / enemy-comp / count+loc / spike)
    # relay even if the semantic gate would abstain -- it was vetoing legitimate
    # callouts to desktop. Narration was already excluded above.
    if _STRONG_CALLOUT_RE.match(s) and not _NARRATION_MUSING_RE.match(s):
        return "tell my team " + s
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
    # Run-together lead: fast speech makes STT drop the spaces in the command
    # prefix ("tell my team" -> "Tellmyteam", "ask my team" -> "Askmyteam"),
    # which the spaced lead canonicalizer below can't match, so a SECOND lead
    # gets prepended and the parser keeps only a fragment (live: "Tellmyteam,
    # Cypherhit84, Sova, Heaven..." relayed just "Sova heaven"). Re-space it.
    s = _RUNON_TEAM_LEAD_RE.sub(
        lambda m: ("ask" if m.group(1).lower() == "ask" else "tell")
        + " my team ", s, count=1)
    # Repair a mis-heard verbatim verb + drop a team-addressee possessive BEFORE
    # routing, so "Pete to my team X" relays X verbatim and "my team's X" strips
    # its lead cleanly.
    s = _REPEAT_MISHEAR.sub(r"\1repeat", s, count=1)
    s = _TEAM_POSSESSIVE.sub(r"\1", s)
    # Canonicalize a mangled / doubled relay-verb lead ("Call my team X",
    # "tell my team Call my team X", "I told the team X", "that's the team X")
    # to a single "tell my team " BEFORE the scaffold/gate pipeline, so the lead
    # never leaks into the spoken line and never falls through to desktop
    # (2026-06-17 battery: the dominant failure mode).
    s = _canonicalize_directive_lead(s)
    # "give my team to <imperative>" -> "tell my team <imperative>" (tell->give
    # mishear; the "to" guards the compose form "give my team encouragement").
    s = _GIVE_TEAM_TO_RE.sub("tell my team ", s, count=1)
    # "drop me his/her/their X" -> "drop me your X" (Ultron asks the agent to
    # drop ITS own gun -> second person).
    s = _DROP_POSSESSIVE_RE.sub("drop me your", s)
    # Bare "ask <question>" / "tell someone to X" -> route to the team (they were
    # abstaining to desktop with no addressee).
    s = _BARE_ASK_RE.sub("ask my team ", s, count=1)
    s = _SOMEONE_LEAD_RE.sub("tell my team someone ", s, count=1)
    # Strip discourse scaffolding (numbered prefixes, say-directive lead-ins, a
    # nested relay verb, embedded fillers) BEFORE the disfluency resolver and the
    # ZERO-MISTAKES gate, so a clean payload reaches routing but questions /
    # Spotify / musings are still seen intact downstream (C6).
    s = _strip_scaffold(s)
    _wfw = _WORD_FOR_WORD.match(s)
    if _wfw and _wfw.group(1).strip():
        s = "say exactly to my team " + _wfw.group(1).strip()
    # Resolve a mid-utterance self-correction to its final intent, and strip a
    # possessive before a roster agent ("relay to my Sova" -> "relay to Sova"),
    # both BEFORE routing.
    s = _resolve_disfluency(s)
    s = _strip_possessive_names(s)
    s = _collapse_multi_addressee(s)
    # ZERO-MISTAKES GATE: conversational / Spotify / identity / desktop commands
    # are left VERBATIM -- the aggressive Valorant vocab correction (phonetic +
    # fuzzy) runs ONLY on callout-bound text, so a question or a song title is
    # never corrupted into agent names. Everything that ISN'T clearly one of
    # those routes is treated as a team callout (the primary wake-addressed use)
    # and gets corrected + lead-recovered.
    if (_NOT_A_CALLOUT.match(s) or _SPOTIFY_SIGNAL.search(s)
            or _REPORTED_QUESTION_GATE.match(s)
            or (_REPORTED_REACTION_RE.match(s)
                and classify_social_reaction(s) is not None)
            or THINK_RESPOND_SUFFIX_RE.search(s)):
        return s
    s = correct_callout_stt(s)
    # Collapse a "KAY/O O" artifact -- STT renders the agent as "Kay O" (two
    # tokens) and the corrector snaps "Kay" -> "KAY/O", leaving a stray "O" that
    # breaks the named-addressee matcher ("ask KAY/O O to knife" -> no match).
    s = re.sub(r"\bKAY/O\s+[oO]\b", "KAY/O", s)
    s = recover_relay_lead(s)
    return s.strip()
