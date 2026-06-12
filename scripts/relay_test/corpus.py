"""Exhaustive Valorant teammate-relay test corpus.

Generates 500+ unique post-wake-word command phrases (the wake word
"Kenning," is already stripped by the time ``match_relay_command`` runs)
covering every relay shape the user calls out plus heavy combinatorial
variation: numeric+location callouts, utility, economy, tactical
directives, ult tracking, insults/banter, encouragement, named-agent
addressing, context+respond, verbatim mode, roast, fun-fact, and free
conversation.

Each entry carries the expected matcher outcome so the harness can score
the matcher deterministically and flag the rephrase / audio separately.

Run standalone to print stats:
    python scripts/relay_test/corpus.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Case:
    """One relay test command.

    text: the post-wake-word phrase the user speaks.
    category: grouping for reporting.
    expect_match: True if match_relay_command should return a command.
    addressee: expected addressee ("team" or an agent name) when matched.
    flags: expected boolean fields (compose/roast/fun_fact/verbatim/context).
    glossary: Valorant terms the rephrase must preserve/interpret.
    note: human explanation of intended semantics.
    """

    text: str
    category: str
    expect_match: bool = True
    addressee: str = "team"
    flags: tuple[str, ...] = ()
    glossary: tuple[str, ...] = ()
    note: str = ""


# --- vocab ------------------------------------------------------------------

# Map locations across the Valorant pool (callout names). Deliberately broad.
LOCATIONS = [
    "vents", "screens", "sewers", "main", "heaven", "hell", "sand", "CT",
    "window", "box", "rafters", "tiles", "top mid", "mid", "drop", "close",
    "back site", "top site", "garage", "long", "short", "market", "link",
    "lamps", "generator", "pit", "tree", "alley", "elbow", "cubby",
    "default", "spawn", "hookah", "stairs", "catwalk", "kitchen", "ramp",
    "A main", "B main", "C main", "A site", "B site", "A lobby", "B lobby",
    "A heaven", "B heaven", "A rafters", "mid courtyard", "mid doors",
    "mid pizza", "mid runway", "snipers nest", "showers", "bottom mid",
    "subroza", "logs", "yard", "boba", "u-haul", "switch", "rubble",
]

NUMS_WORD = ["one", "two", "three", "four", "five"]
NUMS_DIGIT = ["1", "2", "3", "4", "5"]

# Roster of agents the user addresses or references. Lowercased here; the
# matcher's display-name canon handles "kill joy"/"kay o" spacing.
# The full 29-agent VALORANT roster (2026) + STT homophone spellings so the
# test exercises every agent and the common mis-hears the matcher must absorb.
AGENTS = [
    # Duelists
    "jett", "phoenix", "raze", "reyna", "yoru", "neon", "iso", "waylay",
    # Controllers
    "brimstone", "viper", "omen", "astra", "harbor", "clove", "miks",
    # Initiators
    "sova", "breach", "skye", "kayo", "fade", "gekko", "tejo",
    # Sentinels
    "cypher", "sage", "killjoy", "chamber", "deadlock", "vyse", "veto",
    # STT homophones / spaced renderings
    "kill joy", "kay o", "cipher", "gecko", "mix", "way lay",
]

# Self / team status verbs.
SELF_STATUS = [
    ("I am low", ("low",), "low HP"),
    ("I am flanking", ("flank",), "hitting from behind"),
    ("I am rotating", ("rotate",), "moving sites"),
    ("I am saving", ("saving",), "not buying"),
    ("I am anchoring", ("anchor",), "staying to hold the off-site"),
    ("I am sticking", ("sticking",), "planting/defusing the spike"),
    ("I am planting", ("plant",), "planting the spike"),
    ("I am defusing", ("defuse",), "defusing the spike"),
    ("I am lurking", ("lurk",), "playing solo for info/space"),
    ("I am reloading", ("reload",), "reloading"),
    ("I have one flash", ("flash",), "one flashbang"),
    ("I have ult", ("ult",), "ultimate ready"),
    ("I am saving for op", ("op",), "saving to buy the Operator sniper"),
    ("I am going to flank", ("flank",), "going behind them"),
    ("I am holding", (), "holding angle"),
    ("I am pushing", (), "pushing in"),
    ("I am one tapped", (), "low after a headshot"),
    ("I am out of ammo", (), "no bullets"),
]

TEAM_STATUS = [
    ("they are flank", ("flank",), "enemies are flanking"),
    ("they are flanking", ("flank",), "enemies flanking"),
    ("they are pushing", (), "enemies pushing"),
    ("they are rotating", ("rotate",), "enemies rotating"),
    ("they are saving", ("saving",), "enemies on eco"),
    ("they are force buying", (), "enemies force"),
    ("they are planting", ("plant",), "enemies planting"),
    ("they are defusing", ("defuse",), "enemies defusing"),
    ("they have ult", ("ult",), "enemies have ult"),
]

# Utility-usage callouts (ability + place). agent ability verbs.
UTILITY = [
    ("viper walled B", ("viper", "wall", "B"), "Viper toxic screen on B"),
    ("viper walled A", ("viper", "wall", "A"), "Viper wall on A"),
    ("breach stunned mid", ("breach", "stun", "mid"), "Breach stun mid"),
    ("sova darted heaven", ("sova", "dart", "heaven"), "Sova recon heaven"),
    ("kayo knifed site", ("kayo", "knife"), "KAY/O suppression knife"),
    ("brimstone smoked A", ("brimstone", "smoke", "A"), "Brim smokes A"),
    ("omen smoked", ("omen", "smoke"), "Omen smoke"),
    ("killjoy ult site", ("killjoy", "ult"), "Killjoy lockdown"),
    ("raze nade mid", ("raze", "nade", "mid"), "Raze grenade mid"),
    ("they are droning B", ("drone", "B"), "enemy recon drone on B"),
    ("they are smoking A", ("smoke", "A"), "enemy smoking A"),
    ("sova used his drone", ("sova", "drone"), "Sova owl drone"),
    ("cypher caged mid", ("cypher", "cage", "mid"), "Cypher cyber cage"),
    ("fade hed us", ("fade",), "Fade haunt/nightfall"),
]

# Tactical directives (to team or a teammate).
DIRECTIVES = [
    ("to rotate", ("rotate",), "move to the other site"),
    ("to play for time", ("play for time",), "run the bomb clock instead of fighting"),
    ("to play their life", ("play their life",), "stay alive, do not die"),
    ("to wait for me", (), "hold for the user"),
    ("to hold a crossfire with me", ("crossfire",), "hold an angle from opposite sides"),
    ("to fight for main control", ("main",), "contest main"),
    ("to push", (), "push in"),
    ("to fall back", (), "retreat"),
    ("to anchor", ("anchor",), "hold the off-site"),
    ("to lurk", ("lurk",), "play solo for info"),
    ("to plant the spike", ("plant", "spike"), "plant the bomb"),
    ("to defuse", ("defuse",), "defuse"),
    ("to default", (), "default setup"),
    ("to stack site", (), "group on one site"),
    ("to spread out", (), "split apart"),
    ("to save", ("saving",), "do not buy"),
    ("to full buy", (), "buy full"),
    ("to drop us a gun", (), "give a weapon"),
    ("to rotate now", ("rotate",), "rotate immediately"),
]

ULTS = [
    ("breach has ult", ("breach", "ult")),
    ("sova is one point off ult", ("sova", "ult")),
    ("jett has ult", ("jett", "ult")),
    ("omen is one off ult", ("omen", "ult")),
    ("viper ult is ready", ("viper", "ult")),
    ("killjoy ult is up", ("killjoy", "ult")),
    ("raze has her ult", ("raze", "ult")),
    ("chamber is one off", ("chamber", "ult")),
]

BANTER = [
    ("aimlabs is free", ("aimlabs",), "aim training is free, no excuse for bad aim"),
    ("they are terrible", (), "insult"),
    ("they are bots", (), "insult"),
    ("nice try", (), "consolation"),
    ("good half", (), "praise after a half"),
    ("we can still win and to not forfeit", (), "morale"),
    ("we can win this", (), "morale"),
    ("nice clutch", (), "praise"),
    ("good round", (), "praise"),
    ("unlucky", (), "consolation"),
]


def _add(cases: list, c: Case) -> None:
    cases.append(c)


def build_corpus() -> list[Case]:
    cases: list[Case] = []

    # 1. Location callouts: numeric + presence + possession + smoke. Cap the
    #    combinatorial sprawl but cover a wide location set.
    for loc in LOCATIONS:
        _add(cases, Case(f"tell my team there is one {loc}", "location",
                         glossary=(loc,), note=f"one enemy at {loc}"))
        _add(cases, Case(f"tell my team they are {loc}", "location",
                         glossary=(loc,), note=f"enemies at {loc}"))
    # numeric variety on a subset
    for loc in LOCATIONS[:40]:
        n = NUMS_WORD[(len(loc)) % 3 + 1]  # vary 2..4 deterministically
        _add(cases, Case(f"tell my team there are {n} {loc}", "location",
                         glossary=(loc, n), note=f"{n} enemies at {loc}"))
    for loc in LOCATIONS[:30]:
        _add(cases, Case(f"tell my team I saw one {loc}", "location",
                         glossary=(loc,), note=f"spotted one at {loc}"))
    for loc in LOCATIONS[:24]:
        _add(cases, Case(f"tell my team they smoked {loc}", "location",
                         glossary=("smoke", loc), note=f"enemy smoke at {loc}"))
        _add(cases, Case(f"tell my team I have {loc}", "location",
                         glossary=(loc,), note=f"user controls {loc}"))
    for loc in LOCATIONS[:18]:
        _add(cases, Case(f"tell my team they are pushing {loc}", "location",
                         glossary=(loc,), note=f"enemies pushing {loc}"))
    # digit forms
    for d, loc in zip(NUMS_DIGIT, LOCATIONS[:5]):
        _add(cases, Case(f"tell my team there are {d} {loc}", "location",
                         glossary=(loc, d), note=f"{d} at {loc}"))

    # 1b. "last is <place>" -- the LAST alive enemy's location (snap, short).
    for loc in LOCATIONS[:14]:
        _add(cases, Case(f"tell my team last is {loc}", "location",
                         glossary=(loc,), note=f"last enemy at {loc}"))
        _add(cases, Case(f"tell my team last {loc}", "location",
                         glossary=(loc,), note=f"last enemy {loc}"))
    # damage callouts (snap, short -- keep name + number)
    for agent, dmg in [("clove", "120"), ("sova", "84"), ("jett", "67"),
                       ("reyna", "150"), ("omen", "45"), ("raze", "90"),
                       ("killjoy", "30"), ("viper", "112")]:
        _add(cases, Case(f"tell my team {agent} hit {dmg}", "damage",
                         glossary=(agent, dmg), note="damage dealt callout"))

    # 2. Self status.
    for text, gl, note in SELF_STATUS:
        _add(cases, Case(f"tell my team {text}", "self_status",
                         glossary=gl, note=note))
    # teammate-addressed variants of a few
    for text, gl, note in SELF_STATUS[:6]:
        _add(cases, Case(f"tell my teammate {text}", "self_status",
                         glossary=gl, note=note))

    # 3. Team/enemy status.
    for text, gl, note in TEAM_STATUS:
        _add(cases, Case(f"tell my team {text}", "team_status",
                         glossary=gl, note=note))

    # 4. Utility.
    for text, gl, note in UTILITY:
        _add(cases, Case(f"tell my team {text}", "utility",
                         glossary=gl, note=note))

    # 5. Directives to team + teammate.
    for text, gl, note in DIRECTIVES:
        _add(cases, Case(f"tell my team {text}", "directive",
                         glossary=gl, note=note))
        _add(cases, Case(f"tell my teammate {text}", "directive",
                         glossary=gl, note=note))

    # 6. Ult tracking.
    for text, gl in ULTS:
        _add(cases, Case(f"tell my team {text}", "ult", glossary=gl))

    # 7. Banter / morale.
    for text, gl, note in BANTER:
        _add(cases, Case(f"tell my team {text}", "banter",
                         glossary=gl, note=note))

    # 8. Economy specials.
    for text, gl, note in [
        ("I am saving for op", ("op",), "op=Operator"),
        ("to save", ("saving",), "eco"),
        ("to full buy", (), "buy"),
        ("ask our team to drop us a gun", (), "request a weapon"),
        ("to buy me an op", ("op",), "buy Operator"),
        ("they are on eco", (), "enemies poor"),
    ]:
        _add(cases, Case(f"tell my team {text}" if text.startswith(("I ", "to ", "they "))
                         else text, "economy", glossary=gl, note=note))

    # 9. Named-agent addressing: directives + questions + ability requests.
    _agent_abilities = [
        "to smoke A", "to flash for me", "to push with me", "to wait for me",
        "to hold flank", "to drone in", "to dart heaven", "to wall off mid",
    ]
    for i, agent in enumerate(AGENTS):
        disp = agent
        _add(cases, Case(f"tell my {agent} to calm down", "named_directive",
                         addressee=disp, glossary=("calm down",),
                         note="de-escalate that teammate"))
        _add(cases, Case(f"ask my {agent} how their day was", "named_question",
                         addressee=disp, note="conversational question"))
        ab = _agent_abilities[i % len(_agent_abilities)]
        _add(cases, Case(f"tell my {agent} {ab}", "named_ability",
                         addressee=disp, note=f"ability/positioning request: {ab}"))
    # specific colorful ones from the user
    for text, agent, note in [
        ("ask reyna what the meaning of life is", "reyna", "philosophical"),
        ("ask my waylay why they are being such a meanie head", "waylay", "playful jab"),
        ("ask my kill joy to stop being an asshole", "kill joy", "profanity preserved"),
        ("tell my fade to calm the fuck down", "fade", "profanity preserved"),
        ("ask sage how their day was", "sage", "conversational"),
        ("ask what my skye is doing", "skye", "ask-open about skye"),
        ("tell my jett aimlabs is free", "jett", "insult to jett"),
        ("ask my clove to smoke window", "clove", "ability request"),
        ("tell my sova to dart heaven", "sova", "ability request"),
        ("tell my killjoy to ult site", "killjoy", "ability request"),
    ]:
        _add(cases, Case(text, "named_addressed", addressee=agent, note=note))

    # 10. Context + respond (the teammate said something; Kenning replies).
    for text, note in [
        ("my teammate asked if I am trolling, respond", "respond to trolling accusation"),
        ("my teammate is flaming me, tell them to calm the fuck down. in those words specifically",
         "verbatim de-escalation, profanity preserved"),
        ("jett is flaming me, respond and calm him down", "named de-escalation"),
        ("my teammate just asked if you are a sound board, respond", "soundboard question"),
        ("my teammate asked if you are a voice changer, respond", "voice-changer question"),
        ("my teammate asked if you are an AI, respond", "AI-identity question"),
        ("reyna asked if I am hard stuck, respond", "respond to insult"),
        ("my team is asking why I am not buying, tell them I am saving for op",
         "context + literal payload with op"),
    ]:
        flags = ("context",)
        if "in those words specifically" in text:
            flags = ("context", "verbatim")
        _add(cases, Case(text, "context_respond", flags=flags, note=note))

    # 11. Verbatim mode (suffix variants).
    for base in [
        "tell my team to back off",
        "tell my team I am the only one left",
        "tell my team to stop pushing",
    ]:
        for suffix in ["in those words specifically", "word for word", "verbatim"]:
            _add(cases, Case(f"{base}, {suffix}", "verbatim",
                             flags=("verbatim",), note="speak the payload exactly"))

    # 12. Compose / encouragement / greetings.
    for text, flags, note in [
        ("give my team some encouragement", ("compose",), "hype the team"),
        ("tell my team nice try", (), "consolation"),
        ("tell my team good half", (), "praise"),
        # literal-payload greetings: the matcher relays the payload ("hello" /
        # "how their day is going") and the rephrase turns it into a natural
        # line -- no compose flag required.
        ("say hello to my team", (), "greeting"),
        ("ask my team how their day is going", (), "small talk"),
        ("hype up my team", ("compose",), "hype"),
        ("tell my team we got this", (), "morale"),
        ("tell my team to lock in", (), "focus"),
    ]:
        _add(cases, Case(text, "compose", flags=flags, note=note))

    # 13. Roast.
    for text in ["roast my team", "roast them", "flame the lobby", "roast my teammates"]:
        _add(cases, Case(text, "roast", flags=("roast", "compose"),
                         note="verbatim from roast file"))

    # 14. Fun fact.
    for text in [
        "tell my team a fun fact",
        "give my team a fun fact",
        "tell my team an interesting fact",
        "drop a random fact for my team",
    ]:
        _add(cases, Case(text, "fun_fact", flags=("fun_fact",),
                         note="verbatim from fun-fact corpus"))

    # 15. Free conversation to team.
    for text in [
        "tell my team that their comp has nothing to watch the flank so I am going to try to flank",
        "tell my team that the enemy comp has no drone, so play ratty corners",
        "tell my team to fight for main control",
        "tell my team I am anchoring",
        "tell my team I have A site",
        "tell my team clove hit 120",
        "tell my team sova hit 67",
        "tell my team there is one mid",
        "tell my team they are long",
        "tell my team they are short",
        "tell my team they are going C",
        "tell my team I saw one B main",
        "tell our team they are planting",
        "tell our team 3 are garage",
        "tell my team to plant the spike",
    ]:
        _add(cases, Case(text, "freeform", note="natural callout/statement"))

    # 16. NEGATIVE controls -- ordinary speech that must NOT trip the matcher.
    for text in [
        "what time is it in tokyo",
        "how do I cook rice",
        "roast a chicken for dinner",
        "how do I roast coffee beans",
        "tell her I said hi",
        "I want you to acknowledge",
        "respond to my email",
        "play some music",
        "what is the weather",
        "search for the best monitor",
        "tell me a fun fact",            # "tell ME" not the team
        "remind me to buy milk",
    ]:
        _add(cases, Case(text, "negative_control", expect_match=False,
                         note="must fall through to the normal pipeline"))

    return cases


def stats(cases: list[Case]) -> dict:
    from collections import Counter
    by_cat = Counter(c.category for c in cases)
    pos = sum(1 for c in cases if c.expect_match)
    neg = sum(1 for c in cases if not c.expect_match)
    uniq = len({c.text for c in cases})
    return {"total": len(cases), "unique": uniq, "match_expected": pos,
            "no_match_expected": neg, "by_category": dict(by_cat)}


if __name__ == "__main__":
    cs = build_corpus()
    s = stats(cs)
    print(f"corpus: {s['total']} cases ({s['unique']} unique), "
          f"{s['match_expected']} match / {s['no_match_expected']} negative")
    for cat, n in sorted(s["by_category"].items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4}  {cat}")
