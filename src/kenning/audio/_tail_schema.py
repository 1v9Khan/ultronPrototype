"""Tail schema + tagging primitives for the Ultron flavor library.

Pure-python (stdlib only) -> anticheat-safe, importable without the heavy audio
stack. This is the FOUNDATION for the deep flavor expansion (board 2026-06-16):

  * ``TailEntry(text, tags)`` -- a single <=6w Ultron tail plus a frozenset of
    fine-grained TAGS ('loc:high_ground', 'dmg:one_shot', 'ability:dart', ...).
  * ``as_entry`` -- coerces a plain ``str`` (the legacy pool form) into a tagless
    TailEntry, so the existing AGENT_FLAVOR/pool tuples migrate with ZERO rewrite
    and ZERO behavior change (they simply carry no tags and act as the base /
    Tier-3 fallback for their situation).
  * SituationKey constants -- the expanded enemy-facing situation taxonomy
    (4 -> 16) the deeper router sorts into.
  * AGENT_GENDER -- machine-readable canonical pronoun per agent (was only a code
    comment; now a hard-auditable map so a wrong-gender tail is impossible to ship).
  * loc_class / dmg_level_tag / ability_tag -- fold noisy callout facts (location,
    damage, ability) into the coarse TAG vocabulary used to fine-filter a pool.

The COARSE route (agent -> situation) stays a plain dict; tags only ever
fine-select WITHIN an already-correct cell, so a mis-parsed tag can never produce
a wrong-character tail -- it just relaxes to a less specific tier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Iterable, Optional, Union


# ---------------------------------------------------------------------------
# Tail entry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TailEntry:
    """A single Ultron tail (<=6 words) + fine-grained selection tags."""
    text: str
    tags: FrozenSet[str] = field(default_factory=frozenset)


def as_entry(x: Union[str, TailEntry]) -> TailEntry:
    """Coerce a legacy ``str`` tail into a tagless TailEntry (idempotent)."""
    if isinstance(x, TailEntry):
        return x
    return TailEntry(str(x), frozenset())


def entries(pool: Iterable[Union[str, TailEntry]]) -> list[TailEntry]:
    return [as_entry(x) for x in pool]


# ---------------------------------------------------------------------------
# Situation taxonomy (enemy-facing). The first four are the existing keys; the
# rest are the deep expansion. Ally/self registers (command/self/careful) are
# handled separately and NEVER carry these contempt situations.
# ---------------------------------------------------------------------------
class Sit:
    SPOTTED = "spotted"
    ULT = "ult"
    DAMAGED = "damaged"
    UTILITY = "utility"
    MOVING = "moving"
    PLANTING = "planting"
    DEFUSING = "defusing"
    ROTATING = "rotating"
    SAVING = "saving"
    FALLING_BACK = "falling_back"
    PEEKING = "peeking"
    HOLDING = "holding"
    LURKING = "lurking"
    TRADING = "trading"
    LAST_ALIVE = "last_alive"
    NEAR_DEATH = "near_death"


ENEMY_SITUATIONS: tuple[str, ...] = (
    Sit.SPOTTED, Sit.ULT, Sit.DAMAGED, Sit.UTILITY, Sit.MOVING, Sit.PLANTING,
    Sit.DEFUSING, Sit.ROTATING, Sit.SAVING, Sit.FALLING_BACK, Sit.PEEKING,
    Sit.HOLDING, Sit.LURKING, Sit.TRADING, Sit.LAST_ALIVE, Sit.NEAR_DEATH,
)


# ---------------------------------------------------------------------------
# Machine-readable canonical pronoun per agent (she | he | they | it).
# HARD-gated: a tail whose pronoun disagrees with this map must never ship.
# ---------------------------------------------------------------------------
AGENT_GENDER: dict[str, str] = {
    "Astra": "she", "Breach": "he", "Brimstone": "he", "Chamber": "he",
    "Clove": "they", "Cypher": "he", "Deadlock": "she", "Fade": "she",
    "Gekko": "he", "Harbor": "he", "Iso": "he", "Jett": "she", "KAY/O": "it",
    "Killjoy": "she", "Miks": "he", "Neon": "she", "Omen": "he",
    "Phoenix": "he", "Raze": "she", "Reyna": "she", "Sage": "she",
    "Skye": "she", "Sova": "he", "Tejo": "he", "Veto": "he", "Viper": "she",
    "Vyse": "she", "Waylay": "she", "Yoru": "he",
}

#: pronoun-set per gender (for the owner/gender clash lint + filter)
GENDER_PRONOUNS: dict[str, frozenset[str]] = {
    "she": frozenset({"she", "her", "hers", "herself"}),
    "he": frozenset({"he", "him", "his", "himself"}),
    "they": frozenset({"they", "them", "their", "theirs", "themself", "themselves"}),
    "it": frozenset({"it", "its", "itself"}),
}


def agent_gender(agent: Optional[str]) -> Optional[str]:
    return AGENT_GENDER.get(agent or "")


# ---------------------------------------------------------------------------
# Location classes -- fold the ~130 callout location tokens into 6 coarse
# classes so an agent need not have a unique tail per literal callout name
# ("heaven" and "rafters" both -> high_ground).
# ---------------------------------------------------------------------------
_LOC_CLASS_TOKENS: dict[str, frozenset[str]] = {
    "high_ground": frozenset({
        "heaven", "rafters", "tower", "attic", "perch", "nest", "balcony",
        "upper", "top", "ropes", "rope", "catwalk", "boathouse", "overheat",
        "tree", "tower top", "a tower", "b tower", "crane",
    }),
    "long_range": frozenset({
        "long", "a long", "b long", "c long", "window", "snipers", "sniper",
        "bridge", "fountain", "yard", "alley", "garden", "boba", "lane",
    }),
    "site_area": frozenset({
        "site", "a site", "b site", "c site", "main", "a main", "b main",
        "c main", "back site", "default", "plant", "spike", "a", "b", "c",
        "bombsite", "diamond", "hell",
    }),
    "flank_route": frozenset({
        "flank", "ct", "spawn", "behind", "rotation", "rotate", "link",
        "a link", "b link", "garage", "tunnel", "tunnels", "vents", "vent",
        "sewer", "sewers", "drop", "stairs", "back",
    }),
    "mid": frozenset({
        "mid", "middle", "connector", "market", "hookah", "courtyard",
        "pizza", "boba", "top mid", "mid courtyard", "b mid", "a mid",
    }),
    "choke": frozenset({
        "elbow", "corner", "choke", "doors", "door", "gap", "cubby", "nook",
        "dish", "pit", "showers", "shower", "kitchen", "generator", "logs",
        "lamps", "wine", "ramp",
    }),
}
# reverse index token -> class
_TOKEN_TO_LOCCLASS: dict[str, str] = {}
for _cls, _toks in _LOC_CLASS_TOKENS.items():
    for _t in _toks:
        _TOKEN_TO_LOCCLASS.setdefault(_t, _cls)


def loc_class(loc: Optional[str]) -> Optional[str]:
    """Map a (possibly noisy) location token/phrase to a coarse class tag, e.g.
    'heaven' -> 'loc:high_ground'. Returns None if unknown."""
    if not loc:
        return None
    L = loc.strip().lower()
    if not L:
        return None
    if L in _TOKEN_TO_LOCCLASS:
        return f"loc:{_TOKEN_TO_LOCCLASS[L]}"
    # phrase: try each whitespace token + the last word ('a tower' -> 'tower')
    toks = L.split()
    for t in (toks[-1], toks[0]) if toks else ():
        if t in _TOKEN_TO_LOCCLASS:
            return f"loc:{_TOKEN_TO_LOCCLASS[t]}"
    return None


# ---------------------------------------------------------------------------
# Damage level -- fold an hp number and/or damage keyword into 3 levels so a
# one-shot enemy gets the "breaking machine" angle and a scratch does not.
# ---------------------------------------------------------------------------
def dmg_level_tag(count: Optional[str] = None,
                  payload: Optional[str] = None) -> Optional[str]:
    p = (payload or "").lower()
    if any(k in p for k in ("one shot", "one-shot", "oneshot", "1 shot",
                            "1-shot", "lit", "cracked", "almost dead",
                            "more dead than alive", "critical", "no armor")):
        return "dmg:one_shot"
    # explicit "low" / "wounded"
    low_kw = ("low", "hurt", "wounded", "tagged", "chunked", "weak")
    # an hp number in the payload (or the count token if it is a damage value)
    import re as _re
    nums = [int(n) for n in _re.findall(r"\b(\d{1,3})\b", p)]
    if count and str(count).isdigit():
        nums.append(int(count))
    hp = max(nums) if nums else None
    if hp is not None:
        if hp >= 75:
            return "dmg:one_shot"
        if 40 <= hp < 75:
            return "dmg:low"
        if hp < 40:
            return "dmg:minor"
    if any(k in p for k in low_kw):
        return "dmg:low"
    return None


# Verb/token -> canonical ability CATEGORY, so a callout verb ("mollied", "walled",
# "darted") routes to the cell tagged by that category and the semantic query
# carries the ability. Standard categories match the cell tags directly; agent-
# unique abilities fall through to the semantic selector.
_VERB_TO_ABILITY: dict[str, str] = {
    "mollied": "molly", "molly": "molly", "mollies": "molly",
    "incendiary": "molly", "firebomb": "molly", "nade": "molly", "naded": "molly",
    "walled": "wall", "wall": "wall", "walls": "wall", "walling": "wall",
    "smoked": "smoke", "smoke": "smoke", "smokes": "smoke", "smoking": "smoke",
    "darted": "dart", "dart": "dart", "shock": "dart", "shocked": "dart",
    "flashed": "flash", "flash": "flash", "flashes": "flash", "flashing": "flash",
    "blinded": "flash", "caged": "cage", "cage": "cage", "cages": "cage",
    "stunned": "stun", "stun": "stun", "concussed": "stun", "concuss": "stun",
    "droned": "drone", "drone": "drone", "drones": "drone",
    "recon": "recon", "reconned": "recon", "recons": "recon",
    "healed": "heal", "heal": "heal", "healing": "heal", "heals": "heal", "rez": "heal",
    "slowed": "slow", "slow": "slow", "slows": "slow",
    "dashed": "dash", "dash": "dash", "dashes": "dash",
    "teleported": "teleport", "teleport": "teleport", "tp": "teleport", "tped": "teleport",
    "turret": "turret", "trip": "trap", "tripwire": "trap", "trap": "trap",
    "suppressed": "suppress", "suppress": "suppress",
}


def ability_tag(ability: Optional[str]) -> Optional[str]:
    if not ability:
        return None
    a = ability.strip().lower()
    if not a:
        return None
    return f"ability:{_VERB_TO_ABILITY.get(a, a)}"


# Action/state keywords that REFINE the generic enemy 'spotted' situation into a
# finer one. Ordered most-specific first (planting/defusing before generic moving).
_SITUATION_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (Sit.DEFUSING, ("defusing", "defuse", "on the defuse", "tapping it",
                    "tapping the spike", "on the bomb")),
    (Sit.PLANTING, ("planting", "plant the", "going for the plant",
                    "planting spike", "putting it down", "down the spike")),
    (Sit.LAST_ALIVE, ("last alive", "last one", "their last", "last is",
                      "1 left", "one left", "last man", "solo left")),
    (Sit.SAVING, ("saving", "they save", "they saved", "on eco", "eco round",
                  "playing for eco", "won't buy", "not buying", "force save")),
    (Sit.FALLING_BACK, ("falling back", "fell back", "retreating", "backing off",
                        "backing out", "disengaging", "pulled off", "pulling back",
                        "bailed", "they left site")),
    (Sit.ROTATING, ("rotating", "rotate", "rotated", "rotation")),
    (Sit.LURKING, ("lurking", "lurk", "flanking", "flank", "behind us",
                   "backstab", "back stab", "sneaking", "on our flank")),
    (Sit.PEEKING, ("peeking", "peek", "wide swing", "jiggle", "jiggling",
                   "shoulder peek", "dry peek")),
    (Sit.HOLDING, ("holding", "anchoring", "anchor", "camping", "posted up",
                   "sitting on", "waiting", "guarding", "playing retake")),
    (Sit.TRADING, ("traded", "refrag", "re-fragged", "refragged", "got the trade")),
    (Sit.MOVING, ("pushing", "push", "rushing", "rush", "coming", "hitting",
                  "flooding", "heading", "moving", "swinging", "crossing",
                  "splitting", "executing", "running it")),
)


def situation_for_payload(payload: Optional[str]) -> Optional[str]:
    """Refine the enemy 'spotted' situation to a finer state from the callout's
    action words. Returns None (-> caller keeps 'spotted') when nothing clear."""
    if not payload:
        return None
    p = " " + payload.lower().strip() + " "
    for sit, kws in _SITUATION_KEYWORDS:
        for kw in kws:
            if kw in p:
                return sit
    return None


def build_active_tags(*, loc: Optional[str] = None, count: Optional[str] = None,
                      payload: Optional[str] = None,
                      ability: Optional[str] = None) -> frozenset[str]:
    """The set of fine tags a callout's facts imply -- the Tier-1 target."""
    out: set[str] = set()
    for t in (loc_class(loc), dmg_level_tag(count, payload), ability_tag(ability)):
        if t:
            out.add(t)
    return frozenset(out)
