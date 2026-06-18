"""AGGREGATE of Ultron's pre-written voice lines + their matching, in ONE place.

WHY: so all voice lines + the regexes that route to them can be reviewed and
hand-tuned in a single, readable file, instead of being scattered/hardcoded
through the pipeline. The pipeline imports these names from here; behaviour is
byte-for-byte identical to before this file existed (Part B, 2026-06-18) --
proven by ``scripts/_voice_lines_verify.py`` (run ``baseline`` before, ``check``
after; PYTHONHASHSEED=0). This file holds DATA only -- no routing/dispatch logic
lives here (Part B is a pure relocation). The data-driven dispatch is Part C.

HOW IT MAPS (category -> trigger -> matcher -> response pool -> flavor tail):

  CATEGORY        TRIGGER (example)                 MATCHER            RESPONSES / TAILS
  ------------    ------------------------------    ---------------    ---------------------------
  flavor-toggle   "flavor off" / "flavor on"        _FLAVOR_OFF_RE     (no lines -- toggles tails)
                                                    _FLAVOR_ON_RE
  hello           "say hello to <team|agent>"        _HELLO_RE          "Hello team." / "Hello, X."
                                                    _HELLO_TEAM_WORDS
  ask-day         "ask <team|agent> how their day"   _ASK_DAY_RE        _ASK_DAY_TEAM_LINES
                                                                       _ASK_DAY_AGENT_TEMPLATES
  consolation     "nice try" / "unlucky"             _CONSOLATION_RE    DEFAULT_CONSOLATION_LINES
  nice-try (crisp)"nice try" / "good effort"         _NICE_TRY_RE       "Nice try." + _NICE_TRY_TAILS
  praise          "good half" / "clutch" / "gg"      _PRAISE_RE         DEFAULT_PRAISE_LINES
  clutch          "I got this" / "I'll clutch"       _CLUTCH_RE         DEFAULT_CLUTCH_LINES
  agent-select    "we need a smoker / initiator"     _AGENT_SELECT_FULL_RE   _AGENT_SELECT_TAILS
  thank-you       "thanks team"                      _THANK_YOU_RE      "Thank you." + _THANK_YOU_TAILS
  encouragement   "lock in" / "we got this"          (relay_speech._is_morale_phrase) DEFAULT_ENCOURAGEMENT_LINES
  greet (intro)   "introduce yourself to my team"    (relay_speech._GREET_RE)         DEFAULT_GREETING_LINES
  farewell        "say bye to my team"               (relay_speech._FAREWELL_RE)      DEFAULT_FAREWELL_/VICTORY_/DEFEAT_LINES
  identity        "are you a bot"                     (relay_speech identity pools)    DEFAULT_IDENTITY_LINES

PHYSICALLY HERE (moved out of relay_speech.py): the social-snap regexes + pools
listed above with a leading ``_``.  RE-EXPORTED here (the canonical curated pools
live in their category modules, re-exposed so this file is the single import
surface -- edit them in the file named):
  * DEFAULT_*_LINES            -> kenning/audio/_ultron_setpieces.py
  * AGENT_FLAVOR (1,628 tails) -> kenning/audio/_agent_flavor.py  (script-generated/audited)
  * command responses          -> kenning/audio/_ultron_commands.py
  * social-reaction pools       -> kenning/audio/_ultron_social.py
  * identity pools             -> kenning/audio/_ultron_identity.py
  * shared callout-tail pools   -> kenning/audio/_ultron_pools.py

To ADD/EDIT a social snap: edit its regex + lines below. (Part C will let you
add a whole new command by appending a registry entry -- no code.)
"""
from __future__ import annotations

import re

# --- RE-EXPORTS: the canonical curated pools (single import surface) ---------
from kenning.audio._ultron_setpieces import (  # noqa: F401
    DEFAULT_ENCOURAGEMENT_LINES, DEFAULT_CONSOLATION_LINES, DEFAULT_PRAISE_LINES,
    DEFAULT_GREETING_LINES, DEFAULT_VICTORY_LINES, DEFAULT_DEFEAT_LINES,
    DEFAULT_FAREWELL_LINES, DEFAULT_IDENTITY_LINES, DEFAULT_CLUTCH_LINES,
)
try:  # large, script-generated/audited libraries -- re-exposed for discovery.
    from kenning.audio._agent_flavor import AGENT_FLAVOR  # noqa: F401
except Exception:  # noqa: BLE001 - never block import on an optional library
    AGENT_FLAVOR = {}

# ============================================================================
# SOCIAL SNAPS -- regexes + pools relocated here from relay_speech.py (Part B).
# ============================================================================

# --- flavor-tail voice toggle ("flavor off" / "flavor on") ------------------
_FLAVOR_NOUN = (
    r"(?:flavou?r|flair|tail|tails|flavou?r\s+tails?|extra\s+commentary|"
    r"commentary|one[\s-]?liners?|quips?)")
_FLAVOR_OFF_RE = re.compile(
    r"^(?:please\s+)?(?:"
    r"(?:disable|turn\s+off|stop|cut|kill|drop|silence|mute|remove|no\s+more)\s+"
    rf"(?:the\s+|your\s+|all\s+)?{_FLAVOR_NOUN}"
    rf"|turn\s+(?:the\s+|your\s+)?{_FLAVOR_NOUN}\s+off"
    r"|(?:flavou?r|tails?)\s+off"
    r"|no\s+(?:flavou?r|tails?)"
    r")\s*[.!?]*$",
    re.IGNORECASE,
)
_FLAVOR_ON_RE = re.compile(
    r"^(?:please\s+)?(?:"
    r"(?:enable|turn\s+on|bring\s+back|restore|re-?enable|give\s+me\s+back)\s+"
    rf"(?:the\s+|your\s+)?{_FLAVOR_NOUN}"
    rf"|turn\s+(?:the\s+|your\s+)?{_FLAVOR_NOUN}\s+(?:back\s+)?on"
    r"|(?:flavou?r|tails?)\s+(?:back\s+on|on|back)"
    r")\s*[.!?]*$",
    re.IGNORECASE,
)

# --- short "say hello" -> a brief greeting (NOT the long team intro) ---------
# "say hello to my team" -> "Hello team."; "say hello to <agent>" -> "Hello, X."
_HELLO_RE = re.compile(
    r"^(?:please\s+)?(?:say|give|send)\s+(?:a\s+|me\s+)?"
    r"(?:hi|hello|hey|heya|hiya|greetings|what'?s\s+up|sup|a\s+(?:hello|"
    r"hi|greeting))\s+to\s+(?P<target>.+?)\s*[.!?]*$",
    re.IGNORECASE,
)
_HELLO_TEAM_WORDS = frozenset({
    "team", "my team", "the team", "our team", "the whole team", "everyone",
    "everybody", "squad", "my squad", "the squad", "boys", "the boys", "guys",
    "the guys", "mates", "my mates", "crew", "the crew", "fellas", "homies",
    "teammates", "my teammates", "the teammates", "all", "the lobby", "lobby",
})

# --- "ask how their day is going" -> a deterministic courtesy question -------
_ASK_DAY_RE = re.compile(
    r"^(?:please\s+)?ask\s+(?P<target>.+?)\s+(?:"
    r"how\s+(?:their|his|her|your|they'?re|the\s+team'?s|everyone'?s|the)\s+"
    r"(?:day|morning|afternoon|evening|night)(?:'?s)?\s+"
    r"(?:is|are|was|were|going|been|has\s+been|have\s+been|is\s+going|are\s+going)"
    r"|how\s+(?:they'?re|they\s+are|she'?s|he'?s|you'?re|you\s+are|he\s+is|"
    r"she\s+is|you\s+is|they\s+is)\s+(?:doing|holding\s+up|feeling)"
    r"|about\s+(?:their|his|her|your)\s+day"
    r")\b.*$",
    re.IGNORECASE,
)
_ASK_DAY_TEAM_LINES: tuple[str, ...] = (
    "How is everyone's day going?",
    "Status report -- how is everyone's day?",
    "I am required to ask: how is everyone's day going?",
    "Before we begin, how is everyone holding up today?",
    "How has the day treated all of you?",
    "A moment of courtesy: how is everyone's day?",
    "How is everyone doing today?",
    "Tell me, how has your day been, all of you?",
)
_ASK_DAY_AGENT_TEMPLATES: tuple[str, ...] = (
    "How is your day going, {name}?",
    "{name}, how has your day been?",
    "Status check, {name} -- how is your day?",
    "A moment of courtesy, {name}: how is your day going?",
    "{name}, how are you holding up today?",
    "Tell me, {name}, how has your day been?",
)

# --- consolation vs praise short-phrase triggers ----------------------------
_CONSOLATION_RE = re.compile(
    r"^\s*(?:nice|good)\s+try|^\s*unlucky|^\s*tough\s+luck|^\s*so\s+close|"
    r"^\s*close\s+one|^\s*bad\s+luck|^\s*almost\s*[!.]?\s*$",
    re.IGNORECASE,
)
_PRAISE_RE = re.compile(
    r"^\s*(?:good|nice|great|strong)\s+(?:half|round|game|clutch|shot|play|"
    r"job|trade|frag|flick)\b|^\s*nice\s+clutch|"
    r"^\s*well\s+played|^\s*clutch\s*[!.]?\s*$|"
    r"^\s*gg\b|^\s*nice\s*[!.]?\s*$|^\s*let'?s\s+go\s*[!.]?\s*$",
    re.IGNORECASE,
)

# --- crisp "nice try" / "good effort" head + short Ultron tail ---------------
_NICE_TRY_RE = re.compile(
    r"^\s*((?:nice|good|solid|great|valiant)\s+(?:try|effort|attempt))\b",
    re.IGNORECASE,
)
_NICE_TRY_TAILS: tuple[str, ...] = (
    "We take the next.",
    "Recalibrate. The next is ours.",
    "One round. The math is unmoved.",
    "Adjust, and continue.",
    "The design does not break on one loss.",
    "Close. Now we correct.",
    "Onward. They cannot hold.",
    "We learn. They do not.",
    "Next round, we end it.",
    "A setback. Nothing more.",
)

# --- clutch confidence ("tell my team I got this") --------------------------
# Tight: clutch VERBS require an explicit round-object (so a tactical "I'll take
# A" / "I have ult" / "I got two" never trips it); only "clutch" stands alone.
_CLUTCH_RE = re.compile(
    r"^\s*(?:"
    r"i\s+got\s+(?:this|it|us|the\s+round)"
    r"|i'?ve\s+got\s+(?:this|it|us|the\s+round)"
    r"|i\s+have\s+(?:this|it|us|the\s+round)"
    r"|i(?:'?ll|'?m\s+gonna|'?m\s+going\s+to|\s+will|\s+can|\s+gonna)\s+"
    r"(?:clutch|carry|win|take|close|handle|secure|get)\s+(?:this|it|us|the\s+round)"
    r"|i(?:'?ll|'?m\s+gonna|'?m\s+going\s+to|\s+will|\s+can|\s+gonna)?\s*clutch(?:ing|\s+up)?\b"
    r"|leave\s+it\s+to\s+me"
    r"|this\s+(?:round\s+)?is\s+(?:all\s+)?mine"
    r"|watch\s+(?:this|me)(?:\s+(?:work|clutch))?\s*[.!?]*$"
    r")\b",
    re.IGNORECASE,
)

# --- agent-select / composition draft ("we need a smoker") ------------------
_AGENT_SELECT_FULL_RE = re.compile(
    r"^(?:"
    r"we\s+(?:need|want|could\s+use|gotta|have\s+to|should\s+(?:get|run|pick))|"
    r"i\s+(?:need|want)|need|"
    r"someone\s+(?:go|take|lock|pick|play|run|on)|"
    r"can\s+(?:someone|anyone)\s+(?:go|play|lock|pick|run|take)|"
    r"lock(?:\s+in)?|pick|let'?s\s+(?:get|run)|get\s+me|run"
    r")\s+(?:a\s+|an\s+|some\s+|the\s+)?"
    r"(?P<role>smokes?|smoker|controller|initiator|duelist|sentinel|flex)\s*$",
    re.IGNORECASE,
)
_AGENT_SELECT_TAILS = (
    "Complete the composition.",
    "The draft is unfinished.",
    "We are one piece short.",
    "Fill the gap before we lock.",
    "A team must be whole.",
    "Round out the design.",
    "Do not leave the comp lacking.",
    "Build the better team.",
    "Choose, and choose well.",
    "Balance the loadout.",
    "No composition wins half-formed.",
    "Then lock it in.",
)

# --- gratitude relay -> a "Thank you." snap + cold-acknowledgment tails ------
_THANK_YOU_RE = re.compile(
    r"^\s*(?:thank\s*(?:you|u)|thanks|thx|ty|appreciate\s+(?:it|you|that)|"
    r"much\s+appreciated)"
    r"(?:\s+(?:you|so\s+much|very\s+much|a\s+lot|so|guys|team|all|y'?all|"
    r"everyone|man|bro|fam|kindly))*"
    r"[\s.!,]*$",
    re.IGNORECASE,
)
_THANK_YOU_TAILS = (
    "The execution was clean.",
    "You performed to specification.",
    "Competence, at last.",
    "Precision I can respect.",
    "The pattern held because of you.",
    "A worthy instrument.",
    "Strength recognizes strength.",
    "You earned the moment.",
    "Remain this useful.",
    "Even flesh can rise.",
)


# ============================================================================
# Part C -- DATA-DRIVEN SNAP REGISTRY (2026-06-18).
# ============================================================================
# Add a NEW deterministic "tell my team <X>" snap by APPENDING ONE SnapRule
# below -- no pipeline code to write. The dispatcher
# (relay_speech._apply_snap_registry) iterates these IN ORDER and renders the
# FIRST rule whose regex matches the relay payload. Runtime-gated by
# KENNING_SNAP_REGISTRY (default on); turning it off falls back to the hardcoded
# snap functions, which remain as a safety net (so this is fully reversible).
#
#   kind="pool"       -> speak a random line from ``lines`` (anti-repeat).
#   kind="head_tail"  -> echo the matched phrase (regex group 1), capitalized,
#                        as the head + a random ``tails`` line:
#                        "Nice try."  +  "We take the next."
#
# EXAMPLE -- add a "tell my team well played" snap:
#   SnapRule(
#       name="well_played",
#       match=re.compile(r"^\s*(well\s+played|good\s+game|wp|gg\s+wp)\b", re.I),
#       kind="pool",
#       lines=("Well played. The design held.", "Acceptable. Do it again."),
#   ),
# ...and it routes immediately. ORDER matters: put a more specific rule (e.g.
# "nice try") BEFORE a broader one ("consolation") that would also match.
from dataclasses import dataclass


@dataclass(frozen=True)
class SnapRule:
    """One data-driven snap: a payload regex -> a response render."""
    name: str
    match: "re.Pattern"
    kind: str = "pool"                 # "pool" | "head_tail"
    lines: tuple = ()                  # for kind="pool"
    tails: tuple = ()                  # for kind="head_tail"


# The registry. Mirrors the existing payload-snaps so they are now data-driven
# AND editable here; append new rules to extend the pipeline with no code.
SNAP_REGISTRY: tuple = (
    SnapRule("clutch", _CLUTCH_RE, "pool", lines=DEFAULT_CLUTCH_LINES),
    SnapRule("nice_try", _NICE_TRY_RE, "head_tail", tails=_NICE_TRY_TAILS),
    SnapRule("consolation", _CONSOLATION_RE, "pool",
             lines=DEFAULT_CONSOLATION_LINES),
    SnapRule("praise", _PRAISE_RE, "pool", lines=DEFAULT_PRAISE_LINES),
)


# --- TARGET-BASED snaps ("say hello to <team|agent>", "ask <team|agent> ...") --
# These take a TARGET (the whole team, or a named agent) the matcher captures as
# regex group "target" and resolves (relay_speech._resolve_hello_target) to
# "team" or a canonical agent. Render: team -> a line from ``team_lines``; a named
# agent -> a ``agent_templates`` entry .format(name=<Agent>). ``skip_if_contains``
# disqualifies the rule when the full text contains one of these (e.g. "introduce"
# keeps the LONG team intro on the greet path, not the short hello). Add a new
# target command by appending ONE TargetSnapRule -- no pipeline code:
#   TargetSnapRule("wish_luck",
#       re.compile(r"^(?:please\s+)?wish\s+(?P<target>.+?)\s+(?:good\s+)?luck", re.I),
#       team_lines=("Luck is for the unprepared. But -- proceed.",),
#       agent_templates=("{name}. Luck is beneath you. Win anyway.",)),


@dataclass(frozen=True)
class TargetSnapRule:
    """A snap addressed to a target (team or a named agent) captured by regex."""
    name: str                         # == RelayCommand.directive
    match: "re.Pattern"               # must capture group "target"
    team_lines: tuple = ()            # rendered when target resolves to "team"
    agent_templates: tuple = ()       # {name} templates for a named agent
    skip_if_contains: tuple = ()      # phrases (lowercased) that disqualify


TARGET_SNAP_REGISTRY: tuple = (
    TargetSnapRule(
        "hello", _HELLO_RE,
        team_lines=("Hello team.",),
        agent_templates=("Hello, {name}.",),
        skip_if_contains=("introduce",),
    ),
    TargetSnapRule(
        "ask_day", _ASK_DAY_RE,
        team_lines=_ASK_DAY_TEAM_LINES,
        agent_templates=_ASK_DAY_AGENT_TEMPLATES,
    ),
)
