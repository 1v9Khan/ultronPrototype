"""Adaptive response temperament (the cross-cutting Tier-0 self-tune).

Catalog 13 (clawhub-capability-evolver) clean-room synthesis. The upstream
stamped every evolution event with a five-trait ``PersonalityState`` and
ranked configs by cumulative success -- i.e. the agent learned which
"temperament" produced the best outcomes. ultron adopts the safe core of
that idea as a **Tier-0 auto-tune**: an adaptive response *temperament*
that drifts toward what actually satisfies the user (measured from
follow-up corrections, re-asks, and barge-ins), expressed purely as a
tunable DATA profile that shapes the response via a ``[Tone: ...]`` hint
fed to ``response_style``.

Crucially this NEVER touches the locked voice character -- no SOUL.md, no
RVC, no Piper, no TTS voicepack. It only nudges response *shaping*
(verbosity / precision / creativity) the same way
``response_style.apply_brevity_hint`` already does per utterance.

Two mechanisms:

* :class:`PersonalityTuner.record_feedback` -- the online gradient nudge.
  A correction raises rigor + lowers risk-tolerance; a barge-in lowers
  verbosity (be terser); a re-ask raises verbosity (give more); a smooth
  turn gently regresses every trait toward balanced (so extremes relax
  when things are going well). Every change is bounded + clamped to
  ``[0, 1]``.
* :meth:`PersonalityTuner.record_outcome` + :meth:`best_personality` --
  the outcome-ranking aggregator that reports which temperament config has
  the best mean outcome, so the drift can be sanity-checked.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Mapping, Optional

from ultron.evolution.models import PersonalityState, clamp01

# --- tuning constants -------------------------------------------------------

DEFAULT_LEARNING_RATE: float = 0.05
DEFAULT_MAX_DRIFT: float = 0.05  # max change to a trait per feedback step
SATISFIED_DECAY_FRACTION: float = 0.2  # of the learning rate, toward balanced
OUTCOME_BUFFER_SIZE: int = 500
RANKING_BUCKET = 0.1  # round traits to this when grouping for ranking
RANKING_MIN_SAMPLES: int = 3

# --- hint thresholds --------------------------------------------------------

VERBOSITY_LOW: float = 0.35
VERBOSITY_HIGH: float = 0.65
RIGOR_HIGH: float = 0.65
CREATIVITY_HIGH: float = 0.7

_TONE_PREFIX = "[Tone:"


@dataclass(frozen=True)
class PersonalityFeedback:
    """Per-turn satisfaction signals derived from the conversation.

    ``corrected`` = the user corrected the response; ``re_asked`` = the
    user had to ask again; ``barged_in`` = the user interrupted the
    response. None of these set = a smooth, satisfied turn.
    """

    corrected: bool = False
    re_asked: bool = False
    barged_in: bool = False

    @property
    def satisfied(self) -> bool:
        """True iff no negative signal fired."""
        return not (self.corrected or self.re_asked or self.barged_in)


@dataclass(frozen=True)
class PersonalityOutcomeRecord:
    """One (temperament, outcome-score) sample for the ranking aggregator."""

    state: PersonalityState
    score: float
    at: float = 0.0


def _toward(value: float, target: float, step: float) -> float:
    """Move ``value`` toward ``target`` by at most ``step`` (never past)."""
    if value < target:
        return min(target, value + step)
    if value > target:
        return max(target, value - step)
    return value


def temperament_hint(state: PersonalityState) -> str:
    """Map a temperament to a short, TTS-safe ``[Tone: ...]`` directive.

    Returns ``""`` for a balanced temperament (no directive needed). The
    marker prefix ``[Tone:`` is deliberately distinct from
    ``response_style``'s ``[Style:`` so the two hints compose without
    clobbering each other's idempotence check.
    """
    parts: list[str] = []
    if state.verbosity < VERBOSITY_LOW:
        parts.append("keep it concise")
    elif state.verbosity > VERBOSITY_HIGH:
        parts.append("be thorough")
    if state.rigor > RIGOR_HIGH:
        parts.append("be precise and flag any uncertainty")
    if state.creativity > CREATIVITY_HIGH:
        parts.append("a little creativity is welcome")
    if not parts:
        return ""
    return f"{_TONE_PREFIX} " + "; ".join(parts) + ".]"


def apply_temperament(user_text: str, state: PersonalityState) -> str:
    """Prepend the temperament hint to ``user_text`` (idempotent).

    A no-op when the temperament is balanced, the input is empty, or the
    text already carries a ``[Tone:`` hint."""
    hint = temperament_hint(state)
    if not hint or not user_text.strip():
        return user_text
    if user_text.lstrip().startswith(_TONE_PREFIX):
        return user_text
    return f"{hint}\n\n{user_text}"


class PersonalityTuner:
    """Owns the adaptive temperament + the outcome-ranking aggregator."""

    def __init__(
        self,
        *,
        state: Optional[PersonalityState] = None,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        max_drift: float = DEFAULT_MAX_DRIFT,
        outcome_buffer_size: int = OUTCOME_BUFFER_SIZE,
    ) -> None:
        self._state = state or PersonalityState.balanced()
        self._lr = float(learning_rate)
        self._max_drift = float(max_drift)
        self._outcomes: Deque[PersonalityOutcomeRecord] = deque(maxlen=outcome_buffer_size)

    @property
    def state(self) -> PersonalityState:
        """The current temperament."""
        return self._state

    def set_state(self, state: PersonalityState) -> None:
        """Replace the current temperament (e.g. loaded from disk)."""
        self._state = state

    def _nudge(self, value: float, delta: float) -> float:
        delta = max(-self._max_drift, min(self._max_drift, delta))
        return clamp01(value + delta)

    def record_feedback(self, feedback: PersonalityFeedback) -> PersonalityState:
        """Nudge the temperament from one turn's satisfaction signals and
        return the new state. Never raises."""
        s = self._state
        rigor, creativity, verbosity = s.rigor, s.creativity, s.verbosity
        risk, obedience = s.risk_tolerance, s.obedience
        lr = self._lr

        if feedback.corrected:
            rigor = self._nudge(rigor, +lr)
            risk = self._nudge(risk, -lr)
        if feedback.barged_in:
            verbosity = self._nudge(verbosity, -lr)
        if feedback.re_asked:
            verbosity = self._nudge(verbosity, +lr)
        if feedback.satisfied:
            decay = lr * SATISFIED_DECAY_FRACTION
            rigor = _toward(rigor, 0.5, decay)
            verbosity = _toward(verbosity, 0.5, decay)
            risk = _toward(risk, 0.5, decay)
            creativity = _toward(creativity, 0.5, decay)

        self._state = PersonalityState(
            rigor=rigor,
            creativity=creativity,
            verbosity=verbosity,
            risk_tolerance=risk,
            obedience=obedience,
        )
        return self._state

    def current_hint(self) -> str:
        """The current temperament's response-shaping hint."""
        return temperament_hint(self._state)

    # -- outcome ranking ----------------------------------------------------

    def record_outcome(
        self, score: float, *, state: Optional[PersonalityState] = None, at: float = 0.0
    ) -> None:
        """Append a (temperament, outcome-score) sample for ranking."""
        self._outcomes.append(
            PersonalityOutcomeRecord(state=state or self._state, score=clamp01(score), at=at)
        )

    def _bucket_key(self, state: PersonalityState) -> tuple[float, ...]:
        return tuple(
            round(round(v / RANKING_BUCKET) * RANKING_BUCKET, 2)
            for v in (
                state.rigor,
                state.creativity,
                state.verbosity,
                state.risk_tolerance,
                state.obedience,
            )
        )

    def best_personality(self, *, min_samples: int = RANKING_MIN_SAMPLES) -> Optional[PersonalityState]:
        """The temperament config with the best mean outcome (>= ``min_samples``
        observations), or ``None`` when there is not enough data."""
        if not self._outcomes:
            return None
        buckets: dict[tuple[float, ...], list[PersonalityOutcomeRecord]] = defaultdict(list)
        for rec in self._outcomes:
            buckets[self._bucket_key(rec.state)].append(rec)
        best_state: Optional[PersonalityState] = None
        best_mean = -1.0
        for recs in buckets.values():
            if len(recs) < min_samples:
                continue
            mean = sum(r.score for r in recs) / len(recs)
            if mean > best_mean:
                best_mean = mean
                # representative state = the first record in the bucket
                best_state = recs[0].state
        return best_state

    def report(self, *, top_n: int = 3, min_samples: int = RANKING_MIN_SAMPLES) -> str:
        """A human ranking of temperament configs by mean outcome."""
        if not self._outcomes:
            return "Personality ranking: no outcome data yet."
        buckets: dict[tuple[float, ...], list[PersonalityOutcomeRecord]] = defaultdict(list)
        for rec in self._outcomes:
            buckets[self._bucket_key(rec.state)].append(rec)
        ranked = sorted(
            (
                (key, sum(r.score for r in recs) / len(recs), len(recs))
                for key, recs in buckets.items()
                if len(recs) >= min_samples
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        if not ranked:
            return "Personality ranking: not enough samples per config yet."
        lines = ["Personality ranking (mean outcome by config):"]
        for key, mean, n in ranked[:top_n]:
            lines.append(
                f"  - rigor={key[0]:.1f} creativity={key[1]:.1f} verbosity={key[2]:.1f} "
                f"risk={key[3]:.1f} obedience={key[4]:.1f}: {mean:.2f} (n={n})"
            )
        return "\n".join(lines)

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict[str, float]:
        """Serialise the current temperament to a plain dict."""
        s = self._state
        return {
            "rigor": s.rigor,
            "creativity": s.creativity,
            "verbosity": s.verbosity,
            "risk_tolerance": s.risk_tolerance,
            "obedience": s.obedience,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], **kwargs: Any) -> "PersonalityTuner":
        """Construct a tuner from a serialised temperament dict (fail-open
        to balanced on a malformed dict)."""
        try:
            state = PersonalityState(
                rigor=data.get("rigor", 0.5),
                creativity=data.get("creativity", 0.5),
                verbosity=data.get("verbosity", 0.5),
                risk_tolerance=data.get("risk_tolerance", 0.5),
                obedience=data.get("obedience", 0.5),
            )
        except Exception:  # noqa: BLE001
            state = PersonalityState.balanced()
        return cls(state=state, **kwargs)


__all__ = [
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_MAX_DRIFT",
    "VERBOSITY_LOW",
    "VERBOSITY_HIGH",
    "RIGOR_HIGH",
    "CREATIVITY_HIGH",
    "PersonalityFeedback",
    "PersonalityOutcomeRecord",
    "PersonalityTuner",
    "temperament_hint",
    "apply_temperament",
]
