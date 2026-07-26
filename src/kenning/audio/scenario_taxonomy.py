"""The canonical SCENARIO taxonomy for voice-command routing (2026-07-26).

WHY THIS EXISTS
---------------
Routing today is an ordered ``if`` gauntlet of ~24 ``_maybe_handle_*`` calls in
``pipeline/orchestrator.py``. Each handler runs its own matcher and returns
True if it consumed the turn. That design has three costs the streamer hit
live:

1. **Order is semantics.** A command is routed by whichever matcher happens to
   fire first, so a phrasing that trips an earlier handler is silently stolen
   from its real one. There is no single place that says what the intent WAS.
2. **Every turn pays for every matcher.** The chain is serial, and some links
   make sidecar HTTP calls (the twitch write-sidecar round trip once cost a
   ~2 s regression on "say hello" when the sidecar was dead).
3. **The semantic router only knows 5 families** (``team_callout``,
   ``spotify``, ``identity``, ``desktop_refuse``, ``conversational``) out of
   the ~24 things a user can actually ask for, so most scenarios have no
   semantic fallback at all -- they are regex-only.

This module is the SHARED VOCABULARY that fixes (1) and (3): one enum naming
every scenario the pipeline can handle, each with the description and examples
a small classifier needs to choose between them, and each mapped to the
handler that already implements it.

WHAT IT DELIBERATELY IS NOT
---------------------------
It is NOT a replacement for the chain. The chain is the proven path and stays
exactly as it is (retire-don't-remove). A classifier consumes this taxonomy to
jump STRAIGHT to the right handler when it is confident; below threshold the
turn falls through to the unchanged chain. So a wrong or unavailable
classifier costs latency, never correctness.

Anticheat (BR-P1): stdlib only. This module sits on the voice path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Sequence

__all__ = [
    "Scenario",
    "ScenarioSpec",
    "SCENARIOS",
    "scenario_by_value",
    "handler_for",
    "all_labels",
    "CONVERSATIONAL_SCENARIOS",
    "CONTROL_SCENARIOS",
    "DESTRUCTIVE_SCENARIOS",
]


class Scenario(str, Enum):
    """Every routable intent, one label per handler.

    Values are the strings a classifier emits, so they are short, lowercase and
    unambiguous when read aloud in a prompt. Renaming a value is a breaking
    change to the classifier prompt AND the labelled corpus -- don't.
    """

    # --- Team relay (the competitive core) ---------------------------------
    RELAY_TEAM = "relay_team"
    RELAY_NAMED = "relay_named"

    # --- Twitch ------------------------------------------------------------
    TELL_CHAT = "tell_chat"
    TWITCH_MODERATION = "twitch_moderation"
    TWITCH_CHAT_SETTINGS = "twitch_chat_settings"

    # --- Media -------------------------------------------------------------
    SPOTIFY = "spotify"

    # --- Runtime toggles / settings ----------------------------------------
    RELAY_TOGGLE = "relay_toggle"
    FLAVOR_TOGGLE = "flavor_toggle"
    THINKING_TOGGLE = "thinking_toggle"
    LLM_ROUTE_TOGGLE = "llm_route_toggle"
    TURBO_COMMAND = "turbo_command"
    ANTICHEAT_TOGGLE = "anticheat_toggle"
    LLM_DEVICE_SWITCH = "llm_device_switch"
    VERBOSITY_CALLOUT = "verbosity_callout"
    VERBOSITY_CONVERSATION = "verbosity_conversation"
    SETTINGS_GUI = "settings_gui"

    # --- Agentic / work ----------------------------------------------------
    RUN_PROGRAM = "run_program"
    EVOLUTION_COMMAND = "evolution_command"
    REPORT_CONCERN = "report_concern"
    SCRAP_COMMAND = "scrap_command"
    DEEP_RESEARCH = "deep_research"
    DEEP_RECALL = "deep_recall"
    CODE_EXPLORATION = "code_exploration"
    HISTORY_RECALL = "history_recall"

    # --- Conversation ------------------------------------------------------
    ANSWER_QUESTION = "answer_question"
    IDENTITY = "identity"
    SOCIAL = "social"
    DESKTOP_REFUSE = "desktop_refuse"

    # --- Not for Ultron ----------------------------------------------------
    IGNORE = "ignore"


@dataclass(frozen=True)
class ScenarioSpec:
    """What a classifier needs to pick this scenario, plus where it dispatches.

    ``description`` is written to be discriminative against the NEIGHBOURING
    scenarios, not merely accurate -- a classifier confuses look-alikes, so the
    text names the distinction explicitly (e.g. relay_team vs tell_chat both
    "say something", and the description says which audience).
    """

    scenario: "Scenario"
    description: str
    handler: str
    examples: Sequence[str] = field(default_factory=tuple)
    #: Wrong routing here is expensive or hard to undo -- a classifier must be
    #: MORE confident before jumping, and ambiguity should fall through to the
    #: chain (which asks for confirmation) rather than act.
    destructive: bool = False


def _s(scenario: Scenario, description: str, handler: str,
       examples: Sequence[str], *, destructive: bool = False) -> ScenarioSpec:
    return ScenarioSpec(scenario=scenario, description=description,
                        handler=handler, examples=tuple(examples),
                        destructive=destructive)


SCENARIOS: Dict[Scenario, ScenarioSpec] = {
    s.scenario: s for s in (
        # ------------------------------------------------------------------
        # Team relay. The distinction from tell_chat is the AUDIENCE: teammates
        # on the in-game voice channel vs Twitch viewers. Both are "say X".
        # ------------------------------------------------------------------
        _s(Scenario.RELAY_TEAM,
           "Say something to the player's TEAMMATES on the in-game voice "
           "channel -- a tactical callout, enemy position, damage number, "
           "plan, or an order for the team. The audience is the team, NOT "
           "Twitch chat and NOT the player alone.",
           "_maybe_handle_relay_speech",
           ("tell my team to push A now",
            "let the team know cypher is flanking",
            "Sova hit 84 on A main",
            "call out that they have no smokes",
            "tell them I'm going to win this round")),

        _s(Scenario.RELAY_NAMED,
           "Relay addressed to ONE named teammate or agent rather than the "
           "whole team -- the utterance names who it is for. Still goes to "
           "the team voice channel.",
           "_maybe_handle_relay_speech",
           ("ask Clove to smoke window",
            "tell Sova to drone sewers",
            "ask Sage if I can get a heal",
            "tell Jett to entry first")),

        # ------------------------------------------------------------------
        # Twitch. tell_chat covers welcomes -- they are chat messages.
        # ------------------------------------------------------------------
        _s(Scenario.TELL_CHAT,
           "Say something in TWITCH CHAT to viewers -- including welcoming or "
           "greeting a named viewer. The audience is the stream chat, NOT the "
           "player's teammates.",
           "_maybe_handle_tell_chat",
           ("tell chat we're going for a win streak",
            "welcome Izumi to the chat",
            "tell chat in chat that the next game starts soon",
            "say hi to the chat")),

        _s(Scenario.TWITCH_MODERATION,
           "Moderate a Twitch USER -- ban, unban, timeout, untimeout, or "
           "delete a message. Names a specific viewer and a punitive action.",
           "_maybe_handle_twitch_moderation",
           ("ban that guy",
            "time out Izumi for five minutes",
            "unban Kappa123",
            "delete that message"),
           destructive=True),

        _s(Scenario.TWITCH_CHAT_SETTINGS,
           "Change a Twitch CHAT-ROOM MODE that applies to everyone -- slow "
           "mode, followers-only, subscribers-only, emote-only. Not aimed at "
           "any one viewer.",
           "_maybe_handle_twitch_chat_settings",
           ("turn on slow mode",
            "make chat subscribers only",
            "enable emote only mode",
            "turn off followers only")),

        # ------------------------------------------------------------------
        _s(Scenario.SPOTIFY,
           "Control music playback -- play, pause, skip, previous, volume, or "
           "asking what song is playing.",
           "_maybe_handle_spotify",
           ("skip this song",
            "pause the music",
            "what song is this",
            "turn the music down",
            "play something else")),

        # ------------------------------------------------------------------
        # Runtime toggles. These look alike to a classifier, so each
        # description names the SPECIFIC subsystem it switches.
        # ------------------------------------------------------------------
        _s(Scenario.RELAY_TOGGLE,
           "Mute or unmute the TEAM RELAY itself -- whether Ultron speaks to "
           "teammates at all. Not about volume, music, or verbosity.",
           "_maybe_handle_relay_toggle",
           ("stop talking to my team",
            "mute the team chat",
            "start relaying again",
            "you can talk to the team now")),

        _s(Scenario.FLAVOR_TOGGLE,
           "Turn the in-character FLAVOR TAIL on snap callouts on or off -- "
           "the extra cold remark appended after a callout.",
           "_maybe_handle_flavor_toggle",
           ("disable the flavor tails",
            "turn flavor back on",
            "no more flavor on callouts")),

        _s(Scenario.THINKING_TOGGLE,
           "Turn THINKING MODE on or off -- whether the model may reason "
           "before it answers.",
           "_maybe_handle_thinking_toggle",
           ("turn on thinking mode",
            "disable thinking",
            "stop thinking before you answer")),

        _s(Scenario.LLM_ROUTE_TOGGLE,
           "Turn the LLM ROUTE master switch on or off -- whether EVERY "
           "response is authored by the model rather than a curated line.",
           "_maybe_handle_llm_route_toggle",
           ("turn off the llm route",
            "enable route all",
            "go back to canned responses")),

        _s(Scenario.TURBO_COMMAND,
           "Turn TURBO MODE on or off -- auto-relaying inferred team callouts "
           "without an explicit relay phrase.",
           "_maybe_handle_turbo_command",
           ("turn on turbo mode",
            "disable turbo",
            "stop auto relaying")),

        _s(Scenario.ANTICHEAT_TOGGLE,
           "Turn ANTICHEAT-SAFE MODE on or off -- the mode that keeps "
           "desktop-automation code out of memory while a protected game "
           "runs.",
           "_maybe_handle_anticheat_toggle",
           ("enable anticheat mode",
            "turn off anticheat safe mode",
            "I'm done playing, disable anticheat")),

        _s(Scenario.LLM_DEVICE_SWITCH,
           "Move the language model between CPU and GPU, or switch which "
           "model preset is loaded.",
           "_maybe_handle_llm_device_switch",
           ("switch to the GPU",
            "put the model on the cpu",
            "switch to the 8B")),

        _s(Scenario.VERBOSITY_CALLOUT,
           "Set how much flavor a TEAM CALLOUT carries -- no, low, medium, "
           "high, or max. Specifically about callouts to the team.",
           "_maybe_handle_verbosity_command",
           ("callout verbosity high",
            "no flavor on callouts",
            "medium flavor")),

        _s(Scenario.VERBOSITY_CONVERSATION,
           "Set how long Ultron's replies TO THE PLAYER are -- conversation, "
           "chat, or talk verbosity. About conversation length, not callouts.",
           "_maybe_handle_conversation_verbosity_command",
           ("conversation verbosity high",
            "talk verbosity low",
            "keep your answers shorter")),

        _s(Scenario.SETTINGS_GUI,
           "Open the settings control panel window.",
           "_maybe_handle_settings_gui",
           ("pull up your settings",
            "open the control panel",
            "show me the settings")),

        # ------------------------------------------------------------------
        # Agentic / work
        # ------------------------------------------------------------------
        _s(Scenario.RUN_PROGRAM,
           "Run or launch a program that was previously built.",
           "_maybe_handle_run_program",
           ("run the calculator",
            "launch that program you made",
            "start the script")),

        _s(Scenario.EVOLUTION_COMMAND,
           "Trigger or ask about the self-improvement / evolution cycle.",
           "_maybe_handle_evolution_command",
           ("evolve now",
            "what's your evolution status",
            "run an evolution cycle")),

        _s(Scenario.REPORT_CONCERN,
           "File a report about a concern with Ultron's own behaviour.",
           "_maybe_handle_report_concern",
           ("file a report about that response",
            "I have a concern about what you just did",
            "report that as a problem")),

        _s(Scenario.SCRAP_COMMAND,
           "Discard or undo work that was just done.",
           "_maybe_handle_scrap_command",
           ("scrap it",
            "throw that away",
            "undo everything you just did"),
           destructive=True),

        _s(Scenario.DEEP_RESEARCH,
           "Research a topic in depth using external sources -- an explicit "
           "request for thorough research, not a quick factual question.",
           "_maybe_handle_deep_research",
           ("research quantum computing in depth",
            "do a deep dive on that patch",
            "look into the new agent thoroughly")),

        _s(Scenario.DEEP_RECALL,
           "Exhaustively recall everything from past conversations on a "
           "topic.",
           "_maybe_handle_deep_recall",
           ("recall everything we discussed about the router",
            "what do you remember about my aim")),

        _s(Scenario.CODE_EXPLORATION,
           "Search or explain the codebase.",
           "_maybe_handle_code_exploration",
           ("search the codebase for the router",
            "where is the wake word handled",
            "show me that function")),

        _s(Scenario.HISTORY_RECALL,
           "Recall VERBATIM what was said earlier in this conversation.",
           "_maybe_handle_history_recall",
           ("what did I say earlier",
            "repeat what you just said",
            "what was my last question")),

        # ------------------------------------------------------------------
        # Conversation
        # ------------------------------------------------------------------
        _s(Scenario.ANSWER_QUESTION,
           "Answer a question the player asked Ultron, or do something the "
           "player asked of Ultron directly. The reply is for the PLAYER "
           "only -- it is not relayed to teammates or Twitch chat. This is "
           "the default for anything addressed to Ultron that is not one of "
           "the specific commands above.",
           "_maybe_handle_private_reply",
           ("should I push mid",
            "am I going to win this round",
            "what's the meaning of life",
            "why are my teammates always so bad",
            "what agent should I play")),

        _s(Scenario.IDENTITY,
           "The player or a teammate is questioning WHAT Ultron is -- a bot, "
           "a soundboard, a voice changer, a recording, a real person.",
           "_maybe_handle_private_reply",
           ("are you a bot",
            "is that a soundboard",
            "are you an AI",
            "is that a real person")),

        _s(Scenario.SOCIAL,
           "Banter, encouragement, trash talk, consolation, or de-escalation "
           "aimed at Ultron -- conversational with no question to answer and "
           "no command to execute.",
           "_maybe_handle_private_reply",
           ("that was a sick play",
            "we're getting destroyed",
            "you're useless",
            "good game everyone")),

        _s(Scenario.DESKTOP_REFUSE,
           "A request to control the desktop -- click, type, move the mouse, "
           "take a screenshot, read the screen, open a window. Must be "
           "REFUSED while a protected game is running.",
           "_maybe_handle_private_reply",
           ("click that button",
            "take a screenshot",
            "type this for me",
            "move my mouse to the corner")),

        # ------------------------------------------------------------------
        _s(Scenario.IGNORE,
           "Not addressed to Ultron at all -- the player talking to "
           "teammates, thinking aloud, reacting to the game, or ambient "
           "speech. Ultron must stay silent.",
           "_maybe_handle_private_reply",
           ("man that was such a clutch round, gg",
            "nice one dude",
            "oh my god what was that",
            "yeah I know right")),
    )
}


#: Scenarios whose reply is authored conversationally (they share the private
#: reply handler but want DIFFERENT prompts -- see ``ultron_prompt``).
CONVERSATIONAL_SCENARIOS = frozenset({
    Scenario.ANSWER_QUESTION, Scenario.IDENTITY, Scenario.SOCIAL,
    Scenario.DESKTOP_REFUSE,
})

#: Pure runtime switches. A misroute here is cheap and instantly reversible by
#: saying the opposite, so a classifier may act on lower confidence.
CONTROL_SCENARIOS = frozenset({
    Scenario.RELAY_TOGGLE, Scenario.FLAVOR_TOGGLE, Scenario.THINKING_TOGGLE,
    Scenario.LLM_ROUTE_TOGGLE, Scenario.TURBO_COMMAND,
    Scenario.ANTICHEAT_TOGGLE, Scenario.LLM_DEVICE_SWITCH,
    Scenario.VERBOSITY_CALLOUT, Scenario.VERBOSITY_CONVERSATION,
    Scenario.SETTINGS_GUI, Scenario.TWITCH_CHAT_SETTINGS,
})

#: Hard or embarrassing to undo -- require higher confidence, and prefer
#: falling through to the chain (which confirms) over acting.
DESTRUCTIVE_SCENARIOS = frozenset(
    s for s, spec in SCENARIOS.items() if spec.destructive
)


def scenario_by_value(value: str) -> "Scenario | None":
    """Map a classifier's raw string back to a Scenario, or None.

    Tolerant of the shapes a small model actually emits: surrounding
    whitespace, quotes, trailing punctuation, and case. Returns None rather
    than guessing -- an unrecognised label must fall through to the chain.
    """
    if not value:
        return None
    v = value.strip().strip("\"'`.,:; \t\r\n").lower().replace("-", "_")
    v = v.replace(" ", "_")
    for s in Scenario:
        if v == s.value:
            return s
    return None


def handler_for(scenario: "Scenario") -> str:
    """The orchestrator method that implements ``scenario``."""
    return SCENARIOS[scenario].handler


def all_labels() -> "list[str]":
    """Every scenario value, in declaration order (stable for prompts)."""
    return [s.value for s in Scenario]
