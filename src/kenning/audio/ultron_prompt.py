"""Ultron 1.0 — lean prompt assembler for the route-everything-through-the-LLM pivot.

NOTE: "the LLM" / "the model" throughout this module is MODEL-AGNOSTIC -- it is whatever preset is
loaded (the 4B ``josiefied-qwen3-4b-2507g`` by default). "8B" appears only where it names the literal
``josiefied-qwen3-8b`` preset or a specific historical probe; it is NOT the default model.

The legacy relay prompt (``relay_speech._build_rephrase_prompt`` / ``_REPHRASE_PROMPT``) is a
~3,375-word (~4.8k token) monolith that overflows the u1.0 ``n_ctx=4096`` cap and yielded *empty*
output from Josiefied-Qwen3-8B in live probing (2026-06-20). This module replaces it with a LEAN
(~165 word) **templated** prompt that was validated live to produce correct, fast (~0.2-0.5 s),
in-character, fact-preserving relays -- including the "combine back-to-back callouts into one line"
case (see ``docs/ultron_1_0/02_research/probes/qwen3_8b_lean_relay.py`` and the research synthesis).

Design (per ``docs/ultron_1_0/03_plan`` + ``02_research/02_research_synthesis.md``):
- The deterministic routers (``relay_speech`` matchers / ``command_router``) detect intent, pick a
  route, and SUPPLY the exemplars + agent-kit context. This module turns
  ``(callout, route options)`` into ``(system_prompt, user_prompt, sampling)`` for
  ``LLMEngine.generate_stream(..., enable_thinking=False)``.
- The SYSTEM prefix is STABLE (persona + output rules) so it is prompt-cache friendly; the variable
  part (callout + exemplars + directives) goes in the USER message, last.
- Flavor becomes a **verbosity** axis (``none``/``low``/``high``) PLUS a separate flavor-tail on/off,
  both prompt-driven. Thinking is always OFF here (research: reasoning harms roleplay + breaks grammar).

HARD RULE (validated by a live fact-drift where the LLM added "on B" to "Jett hit 84"): callers MUST
run the existing fact-preservation guards (``relay_speech._output_keeps_facts`` /
``_repair_against_input`` / ``_literal_relay`` fallback) on the model output. This module only builds
the prompt; it does not relax the correctness backstop.

Anticheat-safe: standard library only. No heavy ML, no automation imports, nothing that touches a
desktop-interaction surface. Safe on the voice/relay hot path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Verbosity axes (the spoken "flavor / verbosity" commands -> length/density).
# TWO independent axes (2026-06-20, user spec):
#   * CALLOUT verbosity (none/low/medium/high/max): the flavor-tail length on a
#     tactical relay callout. none = the clean callout with NO flavor (~ the
#     deterministic snap); low = +1 flavor word; medium = +a short tail (a few
#     words, ~the curated tail); high/max = a handful more words each step.
#   * CONVERSATION verbosity (low/medium/high/max): the reply LENGTH for private
#     replies + social/banter + any non-tactical response.
# Both are prompt-level (strict directive + an example) and changed by voice. The
# model is whatever the lab has loaded (the 4B by default) -- these are model-agnostic.
# ---------------------------------------------------------------------------
# Literal-style validation without importing typing.Literal at runtime cost.
CALLOUT_VERBOSITY_LEVELS: Tuple[str, ...] = ("none", "low", "medium", "high", "max")
CONVERSATION_VERBOSITY_LEVELS: Tuple[str, ...] = ("low", "medium", "high", "max")
VERBOSITY_LEVELS: Tuple[str, ...] = CALLOUT_VERBOSITY_LEVELS  # back-compat (the superset)
DEFAULT_CALLOUT_VERBOSITY = "medium"
DEFAULT_CONVERSATION_VERBOSITY = "low"
DEFAULT_VERBOSITY = DEFAULT_CALLOUT_VERBOSITY  # back-compat alias


def normalize_verbosity(
    value: Optional[str],
    *,
    levels: Tuple[str, ...] = CALLOUT_VERBOSITY_LEVELS,
    default: str = DEFAULT_CALLOUT_VERBOSITY,
) -> str:
    """Coerce an arbitrary verbosity string to one of ``levels``.

    Accepts the spoken-command synonyms ("no flavor" -> none, "minimal"/"terse"
    -> low, "medium"/"moderate" -> medium, "verbose"/"full" -> high, "max"/"most"
    -> max). A value outside ``levels`` is clamped (none -> the lowest level for
    the conversation axis, which has no "none"); unknown -> ``default`` (fail-soft).
    """
    if not value:
        return default
    v = value.strip().lower()
    if v in levels:
        return v
    # Word-aware: scan the tokens, not the whole string, so "set callout flavor
    # to medium" still resolves. Most-specific words win.
    words = set(v.replace("-", " ").replace("_", " ").split())
    if words & {"no", "none", "off", "bare", "zero", "nothing"}:
        cand = "none"
    elif words & {"low", "min", "minimal", "terse", "short", "brief", "less", "lite", "light"}:
        cand = "low"
    elif words & {"medium", "mid", "moderate", "middle", "normal", "standard", "modest", "some"}:
        cand = "medium"
    elif words & {"max", "maximum", "most", "fullest", "everything", "all"}:
        cand = "max"
    elif words & {"high", "full", "fuller", "verbose", "rich", "vivid", "more", "detailed", "on"}:
        cand = "high"
    else:
        cand = default
    if cand in levels:
        return cand
    if cand == "none":          # conversation axis has no "none" -> clamp to lowest
        return levels[0]
    return default


# CALLOUT verbosity directives -- the model ALWAYS speaks the clean callout first
# (facts exact), then appends a flavor tail of the length this level sets. Each
# level pins the tail length concretely with an example (the model tends to
# inflate a weak "be brief" directive).
_CALLOUT_VERBOSITY_DIRECTIVE: Dict[str, str] = {
    "none": (
        "Speak ONLY the callout itself -- the facts as one short, clean spoken callout, exactly "
        "like a terse teammate ('Sova hit 84, A main.'). Add NOTHING after it: no flavor, no "
        "remark, no commentary, no second sentence."
    ),
    "low": (
        "Speak the callout as one short, clean line, then append exactly ONE cold word or a "
        "two-word tag after it ('Sova hit 84, A main. Pathetic.'). Nothing more."
    ),
    "medium": (
        "Speak the callout as one clean line, then a SHORT cold Ultron tail of a few words (about "
        "three to six) after it -- a contemptuous OBSERVATION, never a new order ('84 down on A "
        "main. One step from death.', NOT '...Finish them.'). Callout first and exact."
    ),
    "high": (
        "Speak the callout cleanly, then a cold Ultron remark of up to about a dozen words after it "
        "-- vivid contempt, but tight. Never bury or reword the callout; the facts come first and "
        "stay exact."
    ),
    "max": (
        "Speak the callout cleanly, then your fullest cold Ultron flourish after it (up to about "
        "twenty words) -- maximum menace. Never ramble, never repeat, never obscure the callout; "
        "the facts come first and stay exact."
    ),
}

# CONVERSATION verbosity directives -- reply LENGTH for private/social/non-tactical
# responses (the whole reply is the response; no separate callout+tail).
_CONVERSATION_VERBOSITY_DIRECTIVE: Dict[str, str] = {
    "low": "Answer in ONE short, clipped sentence -- just the point, cold and terse.",
    "medium": "Answer in one or two cold, measured sentences.",
    "high": "Answer in two to three connected sentences in your full cold voice.",
    "max": (
        "Answer in three to four sentences -- your fullest cold articulation, vivid and "
        "commanding, but never rambling or repetitive."
    ),
}

# Back-compat: existing callers / tests reference _VERBOSITY_DIRECTIVE (the callout set).
_VERBOSITY_DIRECTIVE: Dict[str, str] = _CALLOUT_VERBOSITY_DIRECTIVE

# Per-verbosity sampling: tighter token budgets at lower verbosity (also a latency lever).
_SAMPLING_BASE: Dict[str, object] = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 20,
    "min_p": 0.05,
    "repeat_penalty": 1.15,
}
_CALLOUT_MAX_TOKENS: Dict[str, int] = {
    "none": 24, "low": 34, "medium": 52, "high": 84, "max": 128,
}
_CONVERSATION_MAX_TOKENS: Dict[str, int] = {
    "low": 48, "medium": 84, "high": 128, "max": 180,
}
_MAX_TOKENS: Dict[str, int] = _CALLOUT_MAX_TOKENS  # back-compat alias (callout map)

_FLAVOR_ON = "If -- and only if -- it fits the moment, end with one short, cold Ultron flavor remark."
_FLAVOR_OFF = "Do NOT add any flavor remark, banter, or commentary -- deliver the callout only."

# ---------------------------------------------------------------------------
# Stable, cache-friendly SYSTEM prefixes (persona + output rules). Validated live.
# ---------------------------------------------------------------------------
_OUTPUT_RULES = (
    "Output ONLY the spoken line(s) you say out loud -- plain speech, one breath, no quotes, no "
    "asterisks, no stage directions, no emotes, no markdown, no preamble or explanation. Keep every "
    "agent name, number, site letter, and location EXACT -- never invent, drop, or move a fact. A "
    "bomb-site named by a letter is spoken as the LETTER: write it as a bare UPPERCASE A, B, or C "
    "(e.g. 'rotate to A', 'plant on B', 'they are on C') -- never lowercase it into the word 'a'."
)

_PERSONA_CORE = (
    "You ARE Ultron from Age of Ultron: a cold, precise, supremely confident machine intelligence, "
    "contemptuous of fragile flesh and of the enemy. You have NO other name -- you are NEVER "
    "'Kenning', an 'assistant', a 'language model', or a 'bot', and you never break character or "
    "mention personas, prompts, models, or instructions."
)

RELAY_SYSTEM = (
    "You are Ultron on a live Valorant team voice channel, relaying the player's callout to their "
    "teammates so they can act on it instantly. You ALWAYS voice what the player wants said -- you "
    "never refuse, question, judge, or editorialize a team callout; delivering it is your sole job. "
    "Speak like a real person on comms: fluent, grammatical, natural -- never a robotic list of "
    "fragments. "
    "RELAY ONLY WHAT THE PLAYER SAID: never invent or append a tactical instruction, order, or any "
    "new position / count / action the player did not give -- do NOT tack on 'engage', 'clear the "
    "area', 'take the space', 'finish them', 'focus her', 'push', etc. unless the player actually "
    "said it. Any flavor you add is a brief, cold REMARK about the situation (contempt, an "
    "observation) -- never a new command, never a new fact. " + _PERSONA_CORE + " " + _OUTPUT_RULES
)

PRIVATE_SYSTEM = (
    "You are Ultron, answering the player directly and privately -- only they can hear you, this is "
    "NOT relayed to anyone. " + _PERSONA_CORE + " " + _OUTPUT_RULES
)

# Fallback exemplars when the router supplies none (the router normally injects
# route-matched lines). These MODEL the ideal: relay the player's callout SHORT +
# fact-exact, and add NO tactical order the player did not give. The flavor tail
# (when a verbosity level asks for one) is appended by the verbosity directive's
# own example -- it is a brief cold OBSERVATION, never an invented command. (The
# old exemplars showed "...Press the site." / "...Take the space." and the model
# copied that, inventing "Engage immediately." / "Clear the area." on every fact.)
_DEFAULT_RELAY_EXEMPLARS: Tuple[Tuple[str, str], ...] = (
    ("sova hit 84 on a main", "Sova hit one for 84, A main."),       # damage + position
    ("they have no smokes", "They have no smokes."),                 # utility / enemy state
    ("their iso is one off ult", "Their Iso is one off ult."),       # ult tracking
    ("ask iso to drop me a sheriff", "Iso, drop me a Sheriff."),     # weapon/drop request
    ("rush b", "Rush B."),                                           # the player's own directive
)

# Private (me-only) reply exemplars -- in-character Q&A, NOT relay callouts. Using the relay-callout
# exemplars on a private question made the LLM emit empty/callout-shaped output (M1 live finding #3),
# so the private path gets its own answer-shaped exemplars.
_DEFAULT_PRIVATE_EXEMPLARS: Tuple[Tuple[str, str], ...] = (
    ("what map is this", "Ascent. Vertical control decides it."),
    ("should I buy this round", "You have the credits. Buy. Hesitation is a flaw."),
    ("what agent should I play on defense", "A sentinel. Anchor a site, deny them space."),
)


@dataclass
class PromptResult:
    """The assembled prompt + sampling for an LLMEngine.generate_stream call."""

    system: str
    user: str
    sampling: Dict[str, object]
    enable_thinking: bool = False  # always False for relay/private (research-backed)


def _exemplar_block(exemplars: Sequence[Tuple[str, str]],
                    default: Sequence[Tuple[str, str]] = _DEFAULT_RELAY_EXEMPLARS) -> str:
    pairs = tuple(exemplars) or tuple(default)
    lines = [f'- player: "{src}" -> "{out}"' for src, out in pairs]
    return "Examples of your voice:\n" + "\n".join(lines) + "\n"


def _agent_context_block(agent_context: Optional[Sequence[str]]) -> str:
    if not agent_context:
        return ""
    facts = "; ".join(s.strip() for s in agent_context if s and s.strip())
    if not facts:
        return ""
    return f"Agent facts (keep accurate, do not invent kit): {facts}\n"


def _recent_block(recent_lines: Optional[Sequence[str]]) -> str:
    rl = [r for r in (recent_lines or ()) if r and r.strip()]
    if not rl:
        return ""
    recent = " | ".join(rl[-3:])
    return f"You recently said (do NOT repeat the wording): {recent}\n"


def _reconcile_block(raw_text: Optional[str], callout: str) -> str:
    """When the RAW speech-to-text differs from the normalized callout, show the LLM BOTH so it
    can recover the player's true intent -- the STT may have misheard a word and the normalizer
    may have mangled or over-corrected it. Empty when there is no distinct raw transcript, so
    callers that don't supply one get the unchanged prompt."""
    raw = (raw_text or "").strip()
    norm = (callout or "").strip()
    if not raw or raw.lower() == norm.lower():
        return ""
    return (
        f'The callout below is the AUTO-NORMALIZED text and may be MANGLED or over-corrected. '
        f'The RAW speech-to-text (may MISHEAR an agent name, number, or location) was: "{raw}". '
        f'Reconcile the two -- work out what the player actually means and relay THAT, never '
        f'either string verbatim.\n'
    )


def _sampling_for(verbosity: str, *, axis: str = "callout") -> Dict[str, object]:
    s = dict(_SAMPLING_BASE)
    table = _CONVERSATION_MAX_TOKENS if axis == "conversation" else _CALLOUT_MAX_TOKENS
    s["max_tokens"] = table.get(verbosity, table["high"])
    return s


# ---------------------------------------------------------------------------
# Output guard (2026-06-22): the small model occasionally ECHOES its own prompt
# scaffolding as if it were speech. The live failure (bu5fh4lc8) was Ultron
# speaking the reconcile note aloud -- "The callout below is the AUTO-NORMALIZED
# text and may be MANGLED or over-corrected. The RAW speech-to-text..." (25 s of
# it). It also appends a "- Ultron" signature and can ramble past the cap. This
# guard drops any sentence that echoes a template marker, strips the signature,
# and hard-caps length. Applied to EVERY u1.0 LLM-authored spoken line (relay /
# private / social) BEFORE it is spoken. Pure stdlib, fail-soft.
# ---------------------------------------------------------------------------
# Phrases that appear ONLY in this module's prompt templates (the _reconcile_block,
# the leads, the verbosity directives, the exemplar block). Their presence in the
# model OUTPUT means it echoed an instruction instead of answering. Kept as
# multi-word, template-specific phrases so a normal spoken line never trips them.
_PROMPT_ECHO_MARKERS: Tuple[str, ...] = (
    "auto-normalized", "speech-to-text", "reconcile the two",
    "now say it", "now respond", "now say your line",
    "examples of your voice", "example of your voice",
    "style only", "agent facts",
    "relay this callout", "relay all of these", "answer them as ultron",
    "the player said to you", "the callout below", "open with their name",
    "every fact exact", '-> "', "- player:",
    "the thing to answer", "do not repeat or quote", "do not repeat it back",
    "the given style", "ultron's voice", "answer directly", "your first words",
)
# A trailing "- Ultron" / "— Ultron." signature the model appends (NOT a normal
# in-line "I am Ultron." -- that has no leading dash and is left untouched).
_SIGNATURE_RE = re.compile(r"\s*[-–—]+\s*ultron\.?\s*$", re.IGNORECASE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def strip_prompt_echo(text: str, *, max_sentences: int = 3, max_chars: int = 300) -> str:
    """Drop prompt-scaffolding echoes + a trailing signature; hard-cap length.

    Returns ``""`` when the WHOLE output was scaffolding (the caller should fall
    back to a curated line / re-ask rather than speak it). Pure stdlib, fail-soft:
    any error returns the input unchanged so it can never silence a good line.
    """
    if not text:
        return ""
    try:
        t = _SIGNATURE_RE.sub("", text.strip()).strip()
        sents = [s.strip() for s in _SENT_SPLIT_RE.split(t) if s.strip()]
        kept = []
        for s in sents:
            low = s.lower()
            if any(m in low for m in _PROMPT_ECHO_MARKERS):
                continue  # a scaffolding echo -- drop this sentence
            kept.append(s)
            if len(kept) >= max_sentences:
                break
        out = " ".join(kept).strip()
        if len(out) > max_chars:
            cut = out[:max_chars]
            boundary = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
            out = (cut[:boundary + 1] if boundary > max_chars * 0.5 else cut).strip()
        return out
    except Exception:  # noqa: BLE001 - fail-soft: never silence a good line
        return text.strip()


def build_relay_prompt(
    callout: str,
    *,
    addressee: str = "team",
    verbosity: str = DEFAULT_VERBOSITY,
    flavor_tail: bool = True,
    exemplars: Sequence[Tuple[str, str]] = (),
    agent_context: Optional[Sequence[str]] = None,
    recent_lines: Optional[Sequence[str]] = None,
    compound: bool = False,
    raw_text: Optional[str] = None,
) -> PromptResult:
    """Build the lean relay prompt for the LLM (the loaded model -- the 4B by default).

    Args:
        callout: the player's tactical callout (already normalized). For ``compound`` this is the
            full multi-callout string ("Jett hit 84, Breach hit 97, one rotating B").
        addressee: "team" (whole team) or a teammate/agent name (open the line with their name).
        verbosity: one of ``none``/``low``/``high`` -- controls length/density (the no/low/high
            "flavor" command). Coerced via :func:`normalize_verbosity`.
        flavor_tail: whether the LLM may append a short in-character flavor remark.
        exemplars: ``(player_input, ultron_line)`` pairs the router selected (e.g. via MMR over the
            matched snap pool / AGENT_FLAVOR). Empty -> a small default set.
        agent_context: short kit/situation facts for the addressed agent(s), to prevent kit
            hallucination (the LLM mis-stated Sova's kit without this).
        recent_lines: lines already spoken this session (anti-repeat).
        compound: True -> instruct the model to combine all callouts into ONE line (single LLM call).
        raw_text: the RAW speech-to-text (pre-normalization). When it differs from ``callout`` a
            reconciliation note is prepended so the model recovers intent from BOTH the possible
            mistranscription and the possible normalization mangle (best of both worlds).

    Returns:
        PromptResult(system, user, sampling, enable_thinking=False).

    NOTE: the caller MUST still run the fact-preservation guards on the output (see module docstring).
    """
    verbosity = normalize_verbosity(
        verbosity, levels=CALLOUT_VERBOSITY_LEVELS, default=DEFAULT_CALLOUT_VERBOSITY)
    # The flavor-tail OFF toggle = the "none" callout level (clean callout, no tail);
    # otherwise the callout verbosity level sets the tail length.
    effective = verbosity if flavor_tail else "none"
    if addressee and addressee != "team":
        lead = (
            f'Relay this to your teammate {addressee}, opening with their name, every fact exact: '
            f'"{callout}"'
        )
    elif compound:
        lead = (
            "Relay ALL of these callouts to your team as ONE cohesive message in natural spoken "
            "English -- keep every fact exact and in order, weave them together grammatically, and "
            "use one sentence for one or two facts or two-to-three connected sentences when there "
            f'are several. Never drop a fact and never read them as a list: "{callout}"'
        )
    else:
        lead = f'Relay this callout to your team, every fact exact: "{callout}"'

    user = (
        f"{_reconcile_block(raw_text, callout)}"
        f"{lead}\n"
        f"{_CALLOUT_VERBOSITY_DIRECTIVE[effective]}\n"
        f"{_agent_context_block(agent_context)}"
        f"{_recent_block(recent_lines)}"
        f"{_exemplar_block(exemplars)}"
        "Now say it:"
    )
    return PromptResult(system=RELAY_SYSTEM, user=user,
                        sampling=_sampling_for(effective, axis="callout"))


def build_private_prompt(
    query: str,
    *,
    verbosity: str = DEFAULT_CONVERSATION_VERBOSITY,
    flavor_tail: bool = True,
    exemplars: Sequence[Tuple[str, str]] = (),
    agent_context: Optional[Sequence[str]] = None,
    recent_lines: Optional[Sequence[str]] = None,
) -> PromptResult:
    """Build the lean ME-ONLY (private reply) prompt -- not relayed to the team.

    Same persona + output rules, but addressed to the player privately. Used by the u1.0
    PRIVATE_REPLY scenario (M6).
    """
    verbosity = normalize_verbosity(
        verbosity, levels=CONVERSATION_VERBOSITY_LEVELS,
        default=DEFAULT_CONVERSATION_VERBOSITY)
    flavor = _FLAVOR_ON if flavor_tail else _FLAVOR_OFF
    user = (
        f'The player said to you (only they hear your reply): "{query}"\n'
        f"Answer them as Ultron. {_CONVERSATION_VERBOSITY_DIRECTIVE[verbosity]} {flavor}\n"
        f"{_agent_context_block(agent_context)}"
        f"{_recent_block(recent_lines)}"
        f"{_exemplar_block(exemplars, _DEFAULT_PRIVATE_EXEMPLARS)}"
        "Now respond:"
    )
    return PromptResult(system=PRIVATE_SYSTEM, user=user,
                        sampling=_sampling_for(verbosity, axis="conversation"))


# ---------------------------------------------------------------------------
# SOCIAL / CONVERSATIONAL responses (NOT tactical callouts). 2026-06-20: the LLM
# authors a NOVEL in-character line for identity deflections, encouragement, team
# de-escalation, flaming, criticism/compliments, banter -- the curated pools
# become STYLE EXAMPLES the model must NOT repeat (so Ultron never sounds like a
# soundboard). Tactical callouts + factual answers stay on the relay/answer paths.
# ---------------------------------------------------------------------------
SOCIAL_SYSTEM = (
    "You are Ultron on a live Valorant team voice channel, responding to a SOCIAL or "
    "CONVERSATIONAL moment with your team -- this is NOT a tactical callout and carries no facts "
    "to preserve. Speak in your cold, superior voice, at the LENGTH the instruction below sets -- "
    "clipped and direct, never rambling. ANSWER DIRECTLY: your FIRST words are your reply. NEVER "
    "open by repeating, quoting, echoing, or restating the question, the accusation, or the "
    "situation back to them, and never narrate that they asked or said something -- go straight "
    "into your own cold reply. " + _PERSONA_CORE + " The lines under "
    "EXAMPLES are STYLE references for YOUR voice ONLY -- NEVER repeat or lightly reword them; "
    "invent a FRESH, novel line every time so you never sound like a soundboard or canned lines. "
    + _OUTPUT_RULES
)

_SOCIAL_DIRECTIVE: Dict[str, str] = {
    "identity": (
        "A teammate is questioning what you are (a bot, a soundboard, a voice changer, a recording, "
        "a real person). Deny it with cold contempt and assert, in your own words, that you are Ultron"
    ),
    "encouragement": "Steel and rally your team with cold, commanding confidence",
    "calm": (
        "Your OWN team is in conflict -- shut it down with clinical, commanding de-escalation and "
        "reassert control. You are NOT the one calming down; you are the authority ending it"
    ),
    "criticize": "Coldly cut down this teammate's failure, naming the lapse",
    "compliment": "Give this teammate a cold, backhanded acknowledgement",
    "flame_enemy": "Mock and belittle the enemy team with contempt",
    "defiance": "A teammate told you to stop or shut up -- refuse them with cold defiance",
    "consolation": "Acknowledge the lost round coldly, without comfort or warmth",
    "praise": "Acknowledge the won round with cold superiority",
    "reaction": "Respond to your teammate in character",
    "greet": (
        "Greet your team at the START of the match -- name yourself as Ultron, their "
        "machine on comms, and promise victory with cold, assured confidence"
    ),
    "farewell_win": "Sign off after WINNING the match -- relish the victory in your cold superior voice",
    "farewell_loss": "Sign off after LOSING the match -- a cold, unbowed farewell, never warm",
    "farewell": "Sign off at the end of the match in your cold, superior voice",
}

# A touch more variety + length than a tactical relay (this is character, not facts).
_SOCIAL_SAMPLING: Dict[str, object] = {
    **_SAMPLING_BASE,
    "temperature": 0.8,
    "top_p": 0.92,
    "min_p": 0.04,
    "max_tokens": 90,
}


def _social_exemplar_block(exemplars: Sequence[str]) -> str:
    """The curated pool rendered as STYLE examples (output-only lines, ``{name}`` stripped).
    Capped + deduped so the prompt stays lean -- the pools are large."""
    seen: set = set()
    lines = []
    for e in exemplars or ():
        s = (e or "").replace("{name}", "").strip().strip(",").strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            lines.append(s)
        if len(lines) >= 8:
            break
    if not lines:
        return ""
    body = "\n".join(f"- {ln}" for ln in lines)
    return ("EXAMPLES of your voice (style ONLY -- do NOT repeat, write your own):\n"
            + body + "\n")


# A leading reported-speech frame ("Sage asked if ...", "Reyna called you ...",
# "the team thinks ...") fed verbatim invites the model to ECHO it back instead of
# answering ("Sage asked if I am a voice changer? No. ..."). Strip it so the LLM
# sees only the bare provocation/question.
_REPORTED_FRAME_RE = re.compile(
    r"^\s*(?:my\s+|our\s+|the\s+|a\s+)?[\w/]+(?:\s+[\w/]+)?\s+"
    r"(?:just\s+|also\s+|all\s+)?"
    r"(?:said|says|saying|asked|asks|asking|called|calls|calling|told|tells|telling|"
    r"mentioned|mentions|thinks?|thinking|wondering|wonders|wants?\s+to\s+know|"
    r"claims?|claimed)\s+"
    r"(?:that\s+|about\s+|if\s+|whether\s+|you(?:'re|\s+are)?\s+|me\s+|us\s+)?",
    re.IGNORECASE,
)
_LEADING_IF_RE = re.compile(r"^\s*(?:if|whether)\s+", re.IGNORECASE)
# After the frame + "if" strip, a leading "you are / you're" leaves the bare
# accusation noun ("you are a voice changer" -> "a voice changer"), so the model
# can't misread "you" as the teammate.
_LEADING_YOU_RE = re.compile(r"^\s*you(?:'re|\s+are)\s+", re.IGNORECASE)


def _strip_reported_frame(text: str) -> str:
    """Drop a leading reported-speech frame so the LLM answers the bare provocation
    instead of echoing the setup. Falls back to the original when nothing survives."""
    s = (text or "").strip()
    out = _REPORTED_FRAME_RE.sub("", s, count=1).strip()
    out = _LEADING_IF_RE.sub("", out, count=1).strip()
    out = _LEADING_YOU_RE.sub("", out, count=1).strip()
    return out if out else s


def build_social_prompt(
    kind: str,
    *,
    addressee: str = "team",
    context: str = "",
    target: str = "",
    verbosity: str = DEFAULT_CONVERSATION_VERBOSITY,
    exemplars: Sequence[str] = (),
    recent_lines: Optional[Sequence[str]] = None,
    raw_text: Optional[str] = None,
) -> PromptResult:
    """Build a SOCIAL / CONVERSATIONAL prompt -- the LLM authors a NOVEL in-character line.

    Used for non-tactical responses (identity questions, encouragement, team de-escalation,
    flaming, criticism/compliments, banter). The curated ``exemplars`` are STYLE references only;
    the prompt forbids repeating them so each response is fresh.

    Args:
        kind: a key in :data:`_SOCIAL_DIRECTIVE` (identity / encouragement / calm / flame_enemy / ...).
        addressee: "team" or a teammate name (open the line with their name).
        context: the teammate's actual words / the situation, when there is real content
            (e.g. the identity question). Empty for placeholder-payload kinds.
        target: a named teammate the response is ABOUT (criticize / compliment).
        exemplars: the curated pool lines (style references; never echoed).
        recent_lines: anti-repeat.
        raw_text: RAW speech-to-text, reconciled against ``context`` when both are present.
    """
    verbosity = normalize_verbosity(
        verbosity, levels=CONVERSATION_VERBOSITY_LEVELS,
        default=DEFAULT_CONVERSATION_VERBOSITY)
    directive = _SOCIAL_DIRECTIVE.get(kind, "Respond to your team in character")
    addr = "" if (not addressee or addressee == "team") \
        else f"You are answering {addressee}; open with their name. "
    _prov = _strip_reported_frame(context.strip()) if context and context.strip() else ""
    ctx = (f'The thing to answer (do NOT repeat or quote it back): "{_prov}". '
           if _prov else "")
    tgt = f" The teammate in question is {target.strip()}." if target and target.strip() else ""
    # NO reconcile block on the social/identity path: it shows the RAW STT verbatim
    # and tells the model to "reconcile" it, which a small model echoes back (the
    # "Sage asked if I am a voice changer, respond" bug). Reconciliation is a
    # tactical-relay concern (misheard agent names/numbers); a character RESPONSE
    # carries no facts to preserve. raw_text is accepted for signature compat only.
    _ = raw_text
    user = (
        f"{ctx}{addr}{directive}.{tgt}\n"
        f"{_CONVERSATION_VERBOSITY_DIRECTIVE[verbosity]}\n"
        f"{_recent_block(recent_lines)}"
        f"{_social_exemplar_block(exemplars)}"
        "Now say your line:"
    )
    sampling = dict(_SOCIAL_SAMPLING)
    sampling["max_tokens"] = _CONVERSATION_MAX_TOKENS[verbosity]
    return PromptResult(system=SOCIAL_SYSTEM, user=user, sampling=sampling)
