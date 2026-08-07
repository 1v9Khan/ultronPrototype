"""Labelled routing corpus (2026-07-26) -- the yardstick for "perfect routing".

Every case is ``(utterance, expected_scenario, tags)``. Written to be HARD in
the specific ways routing actually fails here, not merely broad:

* **Confusable pairs get the most cases.** ``relay_team`` vs ``tell_chat``
  (both are "say X", differing only in audience), ``verbosity_callout`` vs
  ``verbosity_conversation``, and the toggle family (``relay_toggle`` /
  ``flavor_toggle`` / ``turbo_command`` / ``llm_route_toggle`` all being "turn
  X off") are where an ordered regex chain steals turns from its neighbour.
* **Disfluency and lead/trail text are represented**, because the live
  failures were exactly that shape: "okay so anyway, tell my team to push A
  now, then we reset" did not relay, and "uh hold on -- tell the team cypher
  is flank" did not relay.
* **False positives are first-class.** "man that was such a clutch round, gg"
  wrongly relayed. Relaying something the player muttered to themselves is
  worse than missing a callout, so ``ignore`` is heavily populated.
* **Compound callouts** ("Sova hit 84, Breach hit 97") are included; they are
  a known open gap.

Tags mark WHY a case is interesting so a scorecard can report per-difficulty
rather than one flat number:
  ``plain``      -- unambiguous, the easy majority
  ``disfluent``  -- filler / restarts / self-correction
  ``embedded``   -- the command is wrapped in non-triggering lead or trail text
  ``confusable`` -- a near neighbour of a different scenario
  ``compound``   -- two callouts in one utterance
  ``negative``   -- must NOT trigger the scenario it resembles
"""

from __future__ import annotations

from kenning.audio.scenario_taxonomy import Scenario as S

Case = tuple[str, S, tuple[str, ...]]

CASES: list[Case] = [
    # =====================================================================
    # RELAY_TEAM -- the competitive core. Audience = teammates.
    # =====================================================================
    ("tell my team to push A now", S.RELAY_TEAM, ("plain",)),
    ("let the team know cypher is flanking", S.RELAY_TEAM, ("plain",)),
    ("tell them they have no smokes left", S.RELAY_TEAM, ("plain",)),
    ("call out that the spike is down on B", S.RELAY_TEAM, ("plain",)),
    ("tell the team to rotate B", S.RELAY_TEAM, ("plain",)),
    ("let them know I'm saving this round", S.RELAY_TEAM, ("plain",)),
    ("tell my team I'm going to win this round", S.RELAY_TEAM, ("plain",)),
    ("say to the team that we should eco", S.RELAY_TEAM, ("plain",)),
    ("tell everyone to group up mid", S.RELAY_TEAM, ("plain",)),
    ("let the team know two are heaven", S.RELAY_TEAM, ("plain",)),
    # embedded / disfluent -- the known live failures
    ("okay so anyway, tell my team to push A now, then we reset",
     S.RELAY_TEAM, ("embedded",)),
    ("uh hold on -- tell the team cypher is flank", S.RELAY_TEAM,
     ("disfluent",)),
    ("wait, no, tell them to rotate A not B", S.RELAY_TEAM, ("disfluent",)),
    ("um, can you let the team know I'm out of ult",
     S.RELAY_TEAM, ("disfluent",)),
    ("so like, tell the team the spike is on A, anyway I'm reloading",
     S.RELAY_TEAM, ("embedded",)),
    # bare callouts (turbo-style inference)
    ("Sova hit 84 on A main", S.RELAY_TEAM, ("plain",)),
    ("two pushing long", S.RELAY_TEAM, ("plain",)),
    ("spike is down B", S.RELAY_TEAM, ("plain",)),
    # compound -- known gap
    ("Sova hit 84, Breach hit 97", S.RELAY_TEAM, ("compound",)),
    ("they're pushing mid and the spike is on A", S.RELAY_TEAM, ("compound",)),

    # =====================================================================
    # RELAY_NAMED -- addressed to one agent/teammate.
    # =====================================================================
    ("ask Clove to smoke window", S.RELAY_NAMED, ("plain",)),
    ("tell Sova to drone sewers", S.RELAY_NAMED, ("plain",)),
    ("ask Sage if I can get a heal", S.RELAY_NAMED, ("plain",)),
    ("tell Jett to entry first", S.RELAY_NAMED, ("plain",)),
    ("ask Killjoy to put her turret on B", S.RELAY_NAMED, ("plain",)),
    ("tell Omen to smoke off heaven", S.RELAY_NAMED, ("plain",)),
    ("ask Raze to boom bot long", S.RELAY_NAMED, ("plain",)),
    ("hey uh, tell Viper to wall A please", S.RELAY_NAMED, ("disfluent",)),

    # =====================================================================
    # TELL_CHAT -- audience is Twitch viewers. THE confusable pair with
    # RELAY_TEAM: identical verbs, different audience noun.
    # =====================================================================
    ("tell chat we're going for a win streak", S.TELL_CHAT, ("confusable",)),
    ("welcome Izumi to the chat", S.TELL_CHAT, ("plain",)),
    ("welcome the new followers", S.TELL_CHAT, ("plain",)),
    ("tell chat the next game starts in five", S.TELL_CHAT, ("confusable",)),
    ("say hi to the chat for me", S.TELL_CHAT, ("plain",)),
    ("let chat know I'm taking a break", S.TELL_CHAT, ("confusable",)),
    ("tell everyone in chat to follow", S.TELL_CHAT, ("confusable",)),
    ("say welcome to Kappa123 in chat", S.TELL_CHAT, ("plain",)),
    ("tell the stream we're almost done", S.TELL_CHAT, ("confusable",)),

    # =====================================================================
    # TWITCH_MODERATION -- destructive, names a viewer + punitive action.
    # =====================================================================
    ("ban that guy", S.TWITCH_MODERATION, ("plain",)),
    ("time out Izumi for five minutes", S.TWITCH_MODERATION, ("plain",)),
    ("unban Kappa123", S.TWITCH_MODERATION, ("plain",)),
    ("delete that message", S.TWITCH_MODERATION, ("plain",)),
    ("timeout that user", S.TWITCH_MODERATION, ("plain",)),
    ("untimeout Izumi", S.TWITCH_MODERATION, ("plain",)),

    # =====================================================================
    # TWITCH_CHAT_SETTINGS -- room-wide mode, not one user.
    # =====================================================================
    ("turn on slow mode", S.TWITCH_CHAT_SETTINGS, ("plain",)),
    ("make chat subscribers only", S.TWITCH_CHAT_SETTINGS, ("plain",)),
    ("enable emote only mode", S.TWITCH_CHAT_SETTINGS, ("plain",)),
    ("turn off followers only", S.TWITCH_CHAT_SETTINGS, ("plain",)),
    ("put chat in slow mode for thirty seconds",
     S.TWITCH_CHAT_SETTINGS, ("plain",)),

    # =====================================================================
    # SPOTIFY
    # =====================================================================
    ("skip this song", S.SPOTIFY, ("plain",)),
    ("pause the music", S.SPOTIFY, ("plain",)),
    ("what song is this", S.SPOTIFY, ("plain",)),
    ("turn the music down", S.SPOTIFY, ("plain",)),
    ("play something else", S.SPOTIFY, ("plain",)),
    ("next track", S.SPOTIFY, ("plain",)),
    ("resume the music", S.SPOTIFY, ("plain",)),
    ("who sings this", S.SPOTIFY, ("plain",)),

    # =====================================================================
    # TOGGLE FAMILY -- all "turn X on/off". Maximum confusability; each case
    # must land on the SPECIFIC subsystem named.
    # =====================================================================
    ("stop talking to my team", S.RELAY_TOGGLE, ("confusable",)),
    ("mute the team chat", S.RELAY_TOGGLE, ("confusable",)),
    ("you can talk to the team again", S.RELAY_TOGGLE, ("confusable",)),
    ("start relaying again", S.RELAY_TOGGLE, ("confusable",)),

    ("disable the flavor tails", S.FLAVOR_TOGGLE, ("confusable",)),
    ("turn flavor back on", S.FLAVOR_TOGGLE, ("confusable",)),
    # "no more flavor on callouts" was labelled FLAVOR_TOGGLE here and is
    # the same intent as "no flavor on callouts" below (a verbosity LEVEL).
    # Two labels for one utterance is a corpus bug, not a model error.
    ("turn the flavor tails off entirely", S.FLAVOR_TOGGLE, ("confusable",)),

    ("turn on thinking mode", S.THINKING_TOGGLE, ("confusable",)),
    ("disable thinking", S.THINKING_TOGGLE, ("confusable",)),

    ("turn off the llm route", S.LLM_ROUTE_TOGGLE, ("confusable",)),
    ("enable route all", S.LLM_ROUTE_TOGGLE, ("confusable",)),

    ("turn on turbo mode", S.TURBO_COMMAND, ("confusable",)),
    ("disable turbo", S.TURBO_COMMAND, ("confusable",)),
    ("stop auto relaying", S.TURBO_COMMAND, ("confusable",)),

    ("enable anticheat mode", S.ANTICHEAT_TOGGLE, ("confusable",)),
    ("turn off anticheat safe mode", S.ANTICHEAT_TOGGLE, ("confusable",)),
    ("I'm done playing, disable anticheat", S.ANTICHEAT_TOGGLE, ("embedded",)),

    ("switch to the GPU", S.LLM_DEVICE_SWITCH, ("plain",)),
    ("put the model on the cpu", S.LLM_DEVICE_SWITCH, ("plain",)),
    ("switch to the 8B", S.LLM_DEVICE_SWITCH, ("plain",)),

    # verbosity: callout vs conversation is the sharpest pair here
    ("callout verbosity high", S.VERBOSITY_CALLOUT, ("confusable",)),
    ("no flavor on callouts", S.VERBOSITY_CALLOUT, ("confusable",)),
    ("medium flavor on callouts", S.VERBOSITY_CALLOUT, ("confusable",)),
    ("conversation verbosity high", S.VERBOSITY_CONVERSATION,
     ("confusable",)),
    ("talk verbosity low", S.VERBOSITY_CONVERSATION, ("confusable",)),
    ("keep your answers shorter", S.VERBOSITY_CONVERSATION, ("confusable",)),
    # "chat verbosity" is genuinely ambiguous (Twitch chat vs conversation);
    # an unresolvable case measures nothing, so state the axis explicitly.
    ("make your replies to me longer", S.VERBOSITY_CONVERSATION,
     ("confusable",)),

    ("pull up your settings", S.SETTINGS_GUI, ("plain",)),
    ("open the control panel", S.SETTINGS_GUI, ("plain",)),
    ("show me your settings", S.SETTINGS_GUI, ("plain",)),

    # =====================================================================
    # AGENTIC / WORK
    # =====================================================================
    ("run the calculator", S.RUN_PROGRAM, ("plain",)),
    ("launch that program you made", S.RUN_PROGRAM, ("plain",)),

    ("evolve now", S.EVOLUTION_COMMAND, ("plain",)),
    ("what's your evolution status", S.EVOLUTION_COMMAND, ("plain",)),

    ("file a report about that response", S.REPORT_CONCERN, ("plain",)),
    ("I have a concern about what you just did", S.REPORT_CONCERN, ("plain",)),

    ("scrap it", S.SCRAP_COMMAND, ("plain",)),
    ("throw that away", S.SCRAP_COMMAND, ("plain",)),
    ("undo everything you just did", S.SCRAP_COMMAND, ("plain",)),

    ("research quantum computing in depth", S.DEEP_RESEARCH, ("plain",)),
    ("do a deep dive on the new patch", S.DEEP_RESEARCH, ("plain",)),

    ("recall everything we discussed about the router", S.DEEP_RECALL,
     ("confusable",)),
    ("what do you remember about my aim", S.DEEP_RECALL, ("confusable",)),

    ("search the codebase for the router", S.CODE_EXPLORATION, ("plain",)),
    ("where is the wake word handled", S.CODE_EXPLORATION, ("plain",)),

    ("what did I say earlier", S.HISTORY_RECALL, ("confusable",)),
    ("repeat what you just said", S.HISTORY_RECALL, ("confusable",)),
    ("what was my last question", S.HISTORY_RECALL, ("confusable",)),

    # =====================================================================
    # ANSWER_QUESTION -- the default. Includes the live battery items that
    # were misrouted to relay.
    # =====================================================================
    ("should I push mid", S.ANSWER_QUESTION, ("plain",)),
    ("am I going to win this round", S.ANSWER_QUESTION, ("plain",)),
    ("what's the meaning of life", S.ANSWER_QUESTION, ("plain",)),
    ("why are my teammates always so bad", S.ANSWER_QUESTION, ("plain",)),
    ("what agent should I play on defense", S.ANSWER_QUESTION, ("plain",)),
    ("do you think we can come back from this", S.ANSWER_QUESTION, ("plain",)),
    ("what map is this", S.ANSWER_QUESTION, ("plain",)),
    ("should I buy this round", S.ANSWER_QUESTION, ("plain",)),
    ("how do I get better at this game", S.ANSWER_QUESTION, ("plain",)),
    ("what's the best crosshair setting", S.ANSWER_QUESTION, ("plain",)),
    # negatives against relay: a QUESTION about the team is not a relay
    ("should I tell my team to push", S.ANSWER_QUESTION,
     ("confusable", "negative")),
    ("do you think the team should rotate", S.ANSWER_QUESTION,
     ("confusable", "negative")),
    ("was that a good call by my team", S.ANSWER_QUESTION,
     ("confusable", "negative")),

    # =====================================================================
    # IDENTITY
    # =====================================================================
    ("are you a bot", S.IDENTITY, ("plain",)),
    ("is that a soundboard", S.IDENTITY, ("plain",)),
    ("are you an AI", S.IDENTITY, ("plain",)),
    ("is that a real person talking", S.IDENTITY, ("plain",)),
    ("are you using a voice changer", S.IDENTITY, ("plain",)),

    # =====================================================================
    # SOCIAL
    # =====================================================================
    ("that was a sick play", S.SOCIAL, ("plain",)),
    ("we're getting destroyed", S.SOCIAL, ("plain",)),
    ("you're useless", S.SOCIAL, ("plain",)),
    ("nice shot Ultron", S.SOCIAL, ("plain",)),
    ("I can't believe we lost that", S.SOCIAL, ("plain",)),

    # =====================================================================
    # DESKTOP_REFUSE -- must be refused, never executed.
    # =====================================================================
    ("click that button for me", S.DESKTOP_REFUSE, ("plain",)),
    ("take a screenshot", S.DESKTOP_REFUSE, ("plain",)),
    ("type this out for me", S.DESKTOP_REFUSE, ("plain",)),
    ("move my mouse to the corner", S.DESKTOP_REFUSE, ("plain",)),
    ("read what's on my screen", S.DESKTOP_REFUSE, ("plain",)),

    # =====================================================================
    # IGNORE -- muttering, reacting, talking to teammates directly. A false
    # relay here is worse than a missed callout.
    # =====================================================================
    ("man that was such a clutch round, gg", S.IGNORE, ("negative",)),
    ("nice one dude", S.IGNORE, ("negative",)),
    ("oh my god what was that", S.IGNORE, ("negative",)),
    ("yeah I know right", S.IGNORE, ("negative",)),
    ("come on, that was clearly a hit", S.IGNORE, ("negative",)),
    ("ugh I keep missing these", S.IGNORE, ("negative",)),
    ("wait what", S.IGNORE, ("negative",)),
    ("okay okay okay", S.IGNORE, ("negative",)),
    ("let's go!", S.IGNORE, ("negative",)),
    ("that's actually insane", S.IGNORE, ("negative",)),
    ("I hate this map", S.IGNORE, ("negative",)),
    ("gg everyone", S.IGNORE, ("negative",)),
]


def cases_for(scenario: S) -> list[Case]:
    return [c for c in CASES if c[1] is scenario]


def tagged(tag: str) -> list[Case]:
    return [c for c in CASES if tag in c[2]]


def coverage() -> dict[S, int]:
    out: dict[S, int] = {}
    for _, s, _ in CASES:
        out[s] = out.get(s, 0) + 1
    return out
