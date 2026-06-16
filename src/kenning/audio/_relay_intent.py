"""Semantic relay-intent gate -- the embeddinggemma sidecar promoted from an
L0-miss fallback to a DECISION gate for the single weakest joint in the routing
cascade: ``recover_relay_lead``'s bare-callout prepend (the source of ~97% of
corpus false-relays).

A bare utterance that merely CONTAINS a callout keyword ("eco", "rotate", an
agent name) is NOT necessarily a team relay -- it is just as often narration the
streamer mutters ("I should tell them to eco"), banter/analysis aimed at Ultron
("their Sage rez'd, how much does that ult cost"), a question for advice ("push
or hold here"), or Marvel/identity talk ("Bruce Banner helped build you"). The
old keyword trigger relayed all of these to teammates. This gate scores the
utterance against curated POSITIVE (real callouts) and NEGATIVE (narration /
banter / question / identity) exemplar clouds and only lets the relay lead
attach when the positive margin clears a calibrated threshold.

Design contract (from the 2026-06-16 frontier research board):
  * FAIL-OPEN: if the sidecar is down ``decide`` returns None and the caller
    keeps today's keyword behavior -- never a new blocking dependency.
  * BIAS TO ABSTAIN: a missed callout costs the streamer a re-say; a false relay
    broadcasts garbage to teammates. The margin threshold favors NOT relaying.
  * NO heavy import in the main process -- this module holds only exemplar
    strings + a thin client over the existing EmbeddingBackend (urllib only).
  * Applied ONLY to the bare-callout branch; explicit "tell my team ..." / team-
    lead / want-team forms have strong surface signals and bypass the gate.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Exemplar clouds. Phrased the way a streamer actually speaks them, WITHOUT a
# leading "tell my team" (these are the bare forms the gate must judge).
# ---------------------------------------------------------------------------
RELAY_POSITIVE_EXEMPLARS: List[str] = [
    # enemy spotting / counts / positions
    "two on B main", "one pushing long", "three rotating to A", "enemy on A fast",
    "they're hitting B hard", "one lurking short", "their Jett has ult",
    "enemy Sova just ulted", "Killjoy lockdown on site", "Viper wall is up B",
    "one shot on the Sova", "their Chamber is one-tapping from long",
    "spotted two heaven", "they're stacking A this round", "enemy planted B",
    # terse "their <agent> <location>" / "<agent> <location>" spotting (bare
    # position info -- distinct from agent-ability banter/analysis)
    "their Cypher is in heaven", "their Sova is holding short",
    "enemy Jett on A main", "a Jett pushing A main", "their Omen in market",
    "Chamber holding long", "their Killjoy anchoring B", "Cypher trip on B",
    "their Viper on A site", "enemy Raze rushing B", "Reyna pushing mid",
    "there's a Jett A main", "their Breach is in tube", "Sova in heaven",
    # self status
    "I'm planting", "I'm low need heal", "I died play the retake",
    "I'm flanking through market", "I'm holding A site", "I'm out of smokes",
    "reloading give me a second", "I'm one shot back off",
    # orders / strat
    "fall back now", "rotate to A", "group up mid", "save this round",
    "force buy this round", "stack B this round", "default to A",
    "watch the flank", "smoke mid for the push", "flash for me on entry",
    "I need a drop", "trade me on the peek", "execute B on my count",
    "play for picks this round", "anchor B site", "hold your angles",
    "spike is down B defusing now", "we have the man advantage press",
    # morale / social callouts
    "nice round team", "we got this keep pushing", "good luck have fun",
    "great round let's keep it going", "my bad that was my fault",
    "clutch up we believe in you", "shake it off next round",
    # weapons / economy info
    "they're on eco rush them", "enemy has the operator long",
    "they bought light save your util", "we're winning the economy keep buying",
]

RELAY_NEGATIVE_EXEMPLARS: List[str] = [
    # --- narration / self-musing about whether to relay (NOT a relay) ---
    "I should tell them to push but they won't listen",
    "honestly should I be asking my team to eco two rounds in a row",
    "I wish I could tell my team to wait for the ult but we don't have time",
    "every time I tell them to watch the flank someone dies anyway",
    "I'm the person who always tells my team to eco and then buys an operator",
    "for the viewers at home my team needs someone to tell them to eco",
    "do I tell them to go A right now or wait another fifteen seconds",
    "not sure whether I should tell them about the Fade haunt or play quiet",
    "there's no point telling my team to fall back they're already dead",
    "by all rights I should be telling my team to hold rotate right now",
    "I can't tell my team to play retake with one person left alive",
    "my biggest improvement area is learning when to tell them to hold versus go",
    "I keep telling them to save and they force buy anyway",
    "I asked my team to save and two of them force bought",
    # --- banter / analysis directed AT Ultron (asking for a read, not relaying) ---
    "their Jett is cracked how do we shut her down",
    "the enemy Sage resurrected their entry fragger how much does that ult cost",
    "their Cypher cam keeps catching us what does that say about us",
    "we keep losing after we plant the spike why is that",
    "their Breach is reading our timings every single round",
    "the enemy Viper toggles her wall and it confuses us every round",
    "their sentinel never rotates and it's working perfectly against us",
    "the enemy Sova coordinates every bolt with their Breach is that a scripted comp",
    "I keep winning my duels but we still lose the rounds why",
    "their top fragger just sold the eco round rifle they are desperate",
    # --- questions / advice requests to Ultron ---
    "what is my play here two versus one spike planted fifteen seconds",
    "should we heavy buy or save after winning the pistol",
    "push or hold they installed on B and it's tight",
    "if you could pick one enemy player to build a team around who would it be",
    "enemy Yoru keeps teleporting behind us on Haven how do we counter that",
    "if the enemy has a Killjoy and a Cypher on defense what does that mean for us",
    "if you could only ever smoke one location on Ascent which one would it be",
    "what does the enemy economy look like should we expect a full buy",
    # --- Marvel / identity talk (answered in-character, never relayed) ---
    "you process audio to text and back there is no Ultron in the middle",
    "your son killed you with the very stone you wanted",
    "Bruce Banner helped build you does that mean you like science",
    "you called that rotation before we even saw them move",
    "you seem like an Omen player would you agree",
    "are you actually Tony Stark's creation or just a soundboard",
    "you built an extinction machine and now you call Valorant rotations",
    "a pause is not the same as winning they stopped your plan permanently",
    "what would your Valorant skin line be called",
    "is brute force a valid strategy to you",
    # --- out-of-roster named addressee (a real person, not a teammate to relay to) ---
    "tell Jordan to anchor B he's watching the stream",
    "let Lauren know to watch the lurk she always forgets",
    "ask Brandon if he saw that clip from last session",
    # --- agent-ability analysis / mockery ending in a judgment or question ---
    "their Chamber is tour-de-forcing every long angle and nobody answers it",
    "the Raze keeps showstopping at long range and hitting nothing is that bad",
    "their Jett alt-fired blade storm at close range and still missed impressive",
    "their Reyna one-tricked all the way to diamond and now she's in our lobby",
    "the enemy plays like a list of reddit tips what's the one-line read",
    "the Jett is peeking without a dash and the operator is that not just greedy",
    "their Sova knows this map a little too well",
    "the enemy Reyna has no kills so she has no devour and no dismiss",
    # --- addressing Ultron about his past calls ("you ...") ---
    "you said to push through the smoke but there was a nano on the other side",
    "you did not warn me about the Harbor cove shielding the plant",
    "you read that perfectly and I ignored you my bad",
    "Viper left the pit for two seconds and you did not call it",
    # --- advice / pick / buy questions to Ultron ---
    "Viper or Astra on Breeze which controller",
    "Skye or Breach which initiator do we want",
    "what do we buy they are definitely forcing a sheriff stack",
    "they are defusing and I am behind them do I shoot or tap the spike",
    "their Killjoy lockdown keeps hitting us through smokes should we look for a counter",
    "I need to know where the operator is not why operators are oppressive",
    "I see the Clove is dead do her smokes stay up or fall",
]


# ---------------------------------------------------------------------------
# Default backend resolver: reuse the router's singleton EmbeddingBackend so the
# per-turn query-embed cache is shared (1 sidecar call per utterance, not 2).
# ---------------------------------------------------------------------------
def _default_backend():  # pragma: no cover - thin glue, exercised live
    try:
        from .command_router import get_command_router
        router = get_command_router()
        return getattr(getattr(router, "backend", None), "emb", None)
    except Exception:                                            # noqa: BLE001
        return None


class RelayIntentGate:
    """Scores an utterance against the positive/negative exemplar clouds.

    ``decide(text)`` -> True (relay), False (abstain -> conversational), or None
    (gate unavailable; caller falls back to keyword behavior)."""

    def __init__(
        self,
        threshold: float = 0.06,
        backend_getter: Optional[Callable[[], object]] = None,
        positives: Sequence[str] = RELAY_POSITIVE_EXEMPLARS,
        negatives: Sequence[str] = RELAY_NEGATIVE_EXEMPLARS,
    ) -> None:
        self.threshold = float(threshold)
        self._get_backend = backend_getter or _default_backend
        self._positives = list(positives)
        self._negatives = list(negatives)
        self._backend = None
        self._prep_pos = None
        self._prep_neg = None

    def _ensure(self) -> bool:
        """Lazily bind + prepare against an AVAILABLE sidecar. Returns False
        (without latching) when the sidecar is down so a later call can recover
        if it comes back up."""
        if self._prep_pos is not None and self._backend is not None:
            return True
        backend = self._get_backend()
        if backend is None:
            return False
        avail = getattr(backend, "available", None)
        if callable(avail) and not avail():
            return False
        try:
            self._prep_pos = backend.prepare(self._positives)
            self._prep_neg = backend.prepare(self._negatives)
            self._backend = backend
            return True
        except Exception:                                       # noqa: BLE001
            self._prep_pos = None
            self._prep_neg = None
            self._backend = None
            return False

    def score(self, text: str) -> Optional[tuple]:
        """Return ``(pos_sim, neg_sim)`` as max-over-exemplar cosine, or None if
        the gate is unavailable."""
        if not self._ensure():
            return None
        try:
            pos = self._backend.score(text, self._prep_pos)
            neg = self._backend.score(text, self._prep_neg)
        except Exception:                                       # noqa: BLE001
            return None
        if not pos or not neg:
            return None
        return (max(pos), max(neg))

    def decide(self, text: str) -> Optional[bool]:
        s = self.score(text)
        if s is None:
            return None
        pos, neg = s
        return (pos - neg) >= self.threshold


# Process-wide singleton (lazy; threshold overridable at boot from config).
_GATE: Optional[RelayIntentGate] = None


def get_relay_intent_gate() -> RelayIntentGate:
    global _GATE
    if _GATE is None:
        _GATE = RelayIntentGate()
    return _GATE


def set_relay_intent_gate(gate: Optional[RelayIntentGate]) -> None:
    """Inject a gate (tests / boot config). Pass None to reset to lazy default."""
    global _GATE
    _GATE = gate


def relay_intent_ok(text: str) -> Optional[bool]:
    """Module-level convenience used by ``recover_relay_lead``."""
    return get_relay_intent_gate().decide(text)
