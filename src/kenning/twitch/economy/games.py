"""Provably-fair games — SpinTheWheel + Slots. Outcomes server-decided.

The OBS overlay is a DUMB RENDERER (MASTER.md SLICE 9): the winning segment and
the exact target angle are computed HERE, server-side, from the
:class:`~kenning.twitch.economy.rng.ProvablyFairRNG` draw — BEFORE any animation.
The overlay merely tweens the wheel to ``target_angle`` and lands on the segment
the server already chose; ``!verify`` re-derives the same result from the
revealed seed.

Wheel geometry: segment ``i`` occupies an arc proportional to its weight. Arc
``i`` spans ``[start_i, start_i + span_i)`` degrees clockwise from 0. The chosen
segment's ``target_angle`` is a deterministic point *strictly inside* that arc
(derived from a second nonce draw, with a small margin from the arc edges so the
pointer never lands on a boundary).

The ``lose ALL points`` consequence is AT-4-class: a segment whose
``consequence == LOSE_ALL`` is INERT unless the wheel was constructed with
``allow_lose_all=True``. With the flag off, such a segment can never be selected
(its effective weight is zeroed for selection) — a structural guarantee, not a
runtime check the caller can forget.

ANTICHEAT (BR-P1): stdlib only. No randomness except via the injected RNG.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from kenning.twitch.economy.rng import ProvablyFairRNG

logger = logging.getLogger("kenning.twitch.economy.games")

__all__ = [
    "WheelSegment",
    "SegmentResult",
    "SpinTheWheel",
    "Slots",
    "SlotsResult",
    "GameResult",
    "GameError",
    "LOSE_ALL",
]

# Sentinel consequence: clears the bettor's entire balance. AT-4 — OFF unless
# the wheel is explicitly constructed with allow_lose_all=True.
LOSE_ALL = "lose_all"

_FULL_TURN = 360.0
# Keep the pointer this fraction of the arc away from each edge so a render
# rounding error can't tip it into a neighbour segment.
_EDGE_MARGIN_FRAC = 0.10


class GameError(Exception):
    """Invalid game configuration or spin parameters."""


@dataclass(frozen=True)
class WheelSegment:
    """One wheel segment.

    :param label: display string (the overlay HTML-escapes it; we keep raw here).
    :param weight: relative probability weight (>= 0).
    :param payout: an opaque payout/consequence token the caller interprets
        (e.g. an int point delta, or :data:`LOSE_ALL` for the clear-all segment).
    """

    label: str
    weight: float
    payout: object = 0

    @property
    def consequence(self) -> str | None:
        return LOSE_ALL if self.payout == LOSE_ALL else None


@dataclass(frozen=True)
class GameResult:
    """Common provenance shared by every game result (for ``!verify``)."""

    game: str
    server_seed: str
    client_seed: str
    nonce: int
    commit: str  # sha256(server_seed) — what was published before the round


@dataclass(frozen=True)
class SegmentResult:
    """The outcome of one wheel spin."""

    index: int
    segment: WheelSegment
    target_angle: float          # degrees in [0,360); overlay tweens here
    arc_start: float             # degrees; chosen arc's clockwise start
    arc_span: float              # degrees; chosen arc's width (∝ weight)
    provenance: GameResult


def _arc_layout(weights: Sequence[float]) -> list[tuple[float, float]]:
    """Return ``[(start_deg, span_deg), ...]`` proportional to weights, summing
    to 360. Zero-weight segments get a zero-width arc (never landable)."""
    total = float(sum(weights))
    if total <= 0:
        raise GameError("sum of segment weights must be > 0")
    arcs: list[tuple[float, float]] = []
    cursor = 0.0
    for w in weights:
        span = (float(w) / total) * _FULL_TURN
        arcs.append((cursor, span))
        cursor += span
    return arcs


class SpinTheWheel:
    """A provably-fair weighted wheel.

    :param segments: list of :class:`WheelSegment`. Order is the visual order on
        the overlay (arc i starts where arc i-1 ended).
    :param rng: the shared :class:`ProvablyFairRNG`.
    :param allow_lose_all: gate for the AT-4 ``lose ALL points`` consequence. When
        ``False`` (default) any ``LOSE_ALL`` segment is excluded from selection
        (effective weight 0) — it can render on the wheel but can never be won.
    """

    def __init__(
        self,
        segments: Sequence[WheelSegment],
        *,
        rng: ProvablyFairRNG | None = None,
        allow_lose_all: bool = False,
    ) -> None:
        segs = self._coerce_segments(segments)
        self._segments: list[WheelSegment] = segs
        self._rng = rng or ProvablyFairRNG()
        self._allow_lose_all = bool(allow_lose_all)

        # Visual arcs use the DECLARED weights (so the wheel looks right);
        # SELECTION uses the effective weights (LOSE_ALL zeroed when gated off).
        self._visual_arcs = _arc_layout([s.weight for s in self._segments])
        self._selection_weights = self._effective_weights()
        if sum(self._selection_weights) <= 0:
            raise GameError(
                "no selectable segment (all zero-weight, or only LOSE_ALL with "
                "allow_lose_all=False)"
            )
        if not self._allow_lose_all and any(
            s.consequence == LOSE_ALL for s in self._segments
        ):
            logger.info(
                "SpinTheWheel: LOSE_ALL segment present but GATED OFF "
                "(allow_lose_all=False) — not selectable"
            )

    @staticmethod
    def _coerce_segments(segments: Sequence[WheelSegment]) -> list[WheelSegment]:
        if segments is None or len(segments) == 0:
            raise GameError("segments must be a non-empty sequence")
        out: list[WheelSegment] = []
        for i, s in enumerate(segments):
            if not isinstance(s, WheelSegment):
                raise GameError(f"segment[{i}] must be a WheelSegment")
            if isinstance(s.weight, bool) or not isinstance(s.weight, (int, float)):
                raise GameError(f"segment[{i}].weight must be a number")
            wf = float(s.weight)
            if wf != wf or wf in (float("inf"), float("-inf")) or wf < 0:
                raise GameError(f"segment[{i}].weight must be finite and >= 0")
            out.append(s)
        return out

    def _effective_weights(self) -> list[float]:
        eff: list[float] = []
        for s in self._segments:
            if s.consequence == LOSE_ALL and not self._allow_lose_all:
                eff.append(0.0)  # structurally unselectable while gated
            else:
                eff.append(float(s.weight))
        return eff

    @property
    def segments(self) -> tuple[WheelSegment, ...]:
        return tuple(self._segments)

    @property
    def allow_lose_all(self) -> bool:
        return self._allow_lose_all

    def spin(
        self,
        server_seed: str,
        client_seed: str | None = None,
        nonce: int = 0,
    ) -> SegmentResult:
        """Decide the winning segment + target angle, server-side.

        The winner is ``rng.weighted_choice`` over the EFFECTIVE weights; the
        target angle is a deterministic point strictly inside the winner's
        VISUAL arc (so the overlay lands on the segment as drawn). Deterministic
        for fixed (server_seed, client_seed, nonce); ``!verify``-reproducible.
        """
        cseed = client_seed if client_seed is not None else self._rng.default_client_seed
        index = self._rng.weighted_choice(
            server_seed, cseed, nonce, self._selection_weights
        )
        seg = self._segments[index]
        arc_start, arc_span = self._visual_arcs[index]

        # Second, independent draw (nonce+1 offset via a distinct client tag) for
        # the within-arc position so the angle isn't correlated with the index
        # draw. Keep a margin from both edges.
        pos = self._rng.uniform_unit(server_seed, f"{cseed}:angle", nonce)
        usable_span = arc_span * (1.0 - 2.0 * _EDGE_MARGIN_FRAC)
        if usable_span <= 0:
            # Degenerate tiny arc — land at its centre.
            target = (arc_start + arc_span / 2.0) % _FULL_TURN
        else:
            offset = arc_span * _EDGE_MARGIN_FRAC + pos * usable_span
            target = (arc_start + offset) % _FULL_TURN

        provenance = GameResult(
            game="spin_the_wheel",
            server_seed=server_seed,
            client_seed=cseed,
            nonce=nonce,
            commit=self._rng.commit_for(server_seed),
        )
        result = SegmentResult(
            index=index,
            segment=seg,
            target_angle=target,
            arc_start=arc_start,
            arc_span=arc_span,
            provenance=provenance,
        )
        logger.info(
            "wheel spin nonce=%d -> index=%d label=%r target=%.3f° "
            "arc=[%.3f,%.3f) lose_all=%s",
            nonce, index, seg.label, target, arc_start, arc_start + arc_span,
            seg.consequence == LOSE_ALL,
        )
        return result

    def angle_in_chosen_arc(self, result: SegmentResult) -> bool:
        """True iff ``target_angle`` lies within the chosen segment's arc
        (the overlay invariant; asserted in tests)."""
        start = result.arc_start
        end = result.arc_start + result.arc_span
        ang = result.target_angle
        # Normalize for the wrap-around case (arc straddling 360->0).
        if end <= _FULL_TURN:
            return start <= ang < end or (ang == start)
        # Wrapped arc.
        return ang >= start or ang < (end - _FULL_TURN)


@dataclass(frozen=True)
class SlotsResult:
    """The outcome of one slots pull."""

    reels: tuple[str, ...]       # the symbol landed on each reel
    indices: tuple[int, ...]     # the chosen index per reel
    is_win: bool                 # all reels equal
    win_symbol: str | None    # the matched symbol when is_win, else None
    provenance: GameResult


class Slots:
    """A simple N-reel slot machine over a shared symbol set.

    Each reel independently draws a symbol from ``symbols`` via a distinct nonce
    derived from the base nonce and the reel index, so all three reels come from
    ONE provably-fair seed/round. A win is all reels showing the same symbol.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        *,
        reels: int = 3,
        rng: ProvablyFairRNG | None = None,
    ) -> None:
        if symbols is None or len(symbols) < 2:
            raise GameError("symbols must have >= 2 entries")
        syms: list[str] = []
        for i, s in enumerate(symbols):
            if not isinstance(s, str) or not s:
                raise GameError(f"symbol[{i}] must be a non-empty str")
            syms.append(s)
        if isinstance(reels, bool) or not isinstance(reels, int) or reels < 1:
            raise GameError("reels must be a positive int")
        self._symbols = syms
        self._reels = int(reels)
        self._rng = rng or ProvablyFairRNG()

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self._symbols)

    @property
    def reels(self) -> int:
        return self._reels

    def pull(
        self,
        server_seed: str,
        client_seed: str | None = None,
        nonce: int = 0,
    ) -> SlotsResult:
        """Spin every reel from the one provably-fair round. Deterministic for
        fixed inputs; ``!verify``-reproducible. Each reel uses a distinct
        derived client tag so reels are independent yet reproducible."""
        cseed = client_seed if client_seed is not None else self._rng.default_client_seed
        n = len(self._symbols)
        indices: list[int] = []
        landed: list[str] = []
        for r in range(self._reels):
            # Distinct, reproducible per-reel draw: tag the client seed with the
            # reel index so reel r's outcome is independent of reel r-1.
            reel_client = f"{cseed}:reel{r}"
            idx = self._rng.outcome(server_seed, reel_client, nonce, n)
            indices.append(idx)
            landed.append(self._symbols[idx])

        is_win = len(set(landed)) == 1
        win_symbol = landed[0] if is_win else None
        provenance = GameResult(
            game="slots",
            server_seed=server_seed,
            client_seed=cseed,
            nonce=nonce,
            commit=self._rng.commit_for(server_seed),
        )
        logger.info(
            "slots pull nonce=%d -> reels=%s win=%s symbol=%s",
            nonce, tuple(landed), is_win, win_symbol,
        )
        return SlotsResult(
            reels=tuple(landed),
            indices=tuple(indices),
            is_win=is_win,
            win_symbol=win_symbol,
            provenance=provenance,
        )
