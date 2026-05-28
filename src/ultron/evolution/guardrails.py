"""Regression guardrails + auto-revert trigger + rollback-frequency audit.

Catalog 13 (clawhub-capability-evolver) clean-room synthesis. These are
what make autonomous auto-apply *safe*: after the evolution loop applies a
change it monitors a few turns, and if ANY guardrail regresses, the change
is automatically rolled back via the checkpoint. The four guardrails:

* **latency** -- the voice path's time-to-first-audio / TTFT / TTS synth
  must not regress past a tolerance over the locked baseline;
* **quality** -- the user-correction + re-ask + barge-in rate must not
  climb (a proxy for "the change made ultron worse to talk to");
* **errors** -- the error + fail-open + validator-block rate must not
  climb;
* **resource** -- VRAM peak must not approach the hard ceiling.

The :class:`RollbackAudit` is the negative-feedback brake: every applied
change records whether it was kept or auto-reverted, and a surface whose
recent rollback rate exceeds :data:`ROLLBACK_DEMOTE_THRESHOLD` is flagged
for demotion (the proposal generator or the guardrails themselves may be
miscalibrated). Every function is pure / in-memory; no IO, no model loads.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Sequence

# --- thresholds -------------------------------------------------------------

#: A latency observation may be at most this ratio of the baseline before it
#: counts as a regression (15% slower than the locked baseline trips).
LATENCY_TOLERANCE_RATIO: float = 1.15

#: The user-correction / error rate may climb by at most this many absolute
#: points (0.10 = 10 percentage points) before it trips.
QUALITY_RATE_ABS_INCREASE: float = 0.10
ERROR_RATE_ABS_INCREASE: float = 0.10

#: VRAM hard ceiling (physical cap) and the headroom fraction at which the
#: resource guardrail trips.
VRAM_CAP_MB: float = 11_500.0
VRAM_HEADROOM_RATIO: float = 0.95

#: A surface whose recent rollback rate exceeds this is auto-demoted.
ROLLBACK_DEMOTE_THRESHOLD: float = 0.30
ROLLBACK_MIN_SAMPLES: int = 5
ROLLBACK_WINDOW: int = 20

#: The four guardrail names.
GUARDRAILS: tuple[str, ...] = ("latency", "quality", "error", "resource")


# --- baseline + observed sample --------------------------------------------


@dataclass(frozen=True)
class GuardrailBaseline:
    """The locked reference values a change must not regress past.

    Defaults are ultron's measured voice baseline (TTFA 266 ms / TTFT
    172 ms / TTS 78 ms; VRAM peak ~6664 MB). ``correction_rate`` /
    ``error_rate`` default to 0 (the pre-change rate is supplied by the
    caller when known).
    """

    ttfa_ms: float = 266.0
    ttft_ms: float = 172.0
    tts_ms: float = 78.0
    correction_rate: float = 0.0
    error_rate: float = 0.0
    vram_peak_mb: float = 6664.0


@dataclass(frozen=True)
class GuardrailSample:
    """Observed metrics over the post-apply monitoring window.

    Any field left ``None`` means "not observed" -- its guardrail is
    skipped (a missing metric never trips a revert)."""

    ttfa_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    tts_ms: Optional[float] = None
    correction_rate: Optional[float] = None
    error_rate: Optional[float] = None
    vram_peak_mb: Optional[float] = None
    turns_observed: int = 0


@dataclass(frozen=True)
class GuardrailVerdict:
    """The aggregate verdict over all four guardrails."""

    tripped: bool
    tripped_guards: tuple[str, ...] = ()
    details: tuple[str, ...] = ()

    @property
    def should_revert(self) -> bool:
        """Whether the applied change must be auto-reverted."""
        return self.tripped


@dataclass(frozen=True)
class GuardrailConfig:
    """Per-guardrail enable flags + thresholds."""

    enabled: tuple[str, ...] = GUARDRAILS
    latency_tolerance_ratio: float = LATENCY_TOLERANCE_RATIO
    quality_abs_increase: float = QUALITY_RATE_ABS_INCREASE
    error_abs_increase: float = ERROR_RATE_ABS_INCREASE
    vram_cap_mb: float = VRAM_CAP_MB
    vram_headroom_ratio: float = VRAM_HEADROOM_RATIO


# --- the four detectors -----------------------------------------------------


def detect_latency_regression(
    baseline: GuardrailBaseline,
    sample: GuardrailSample,
    *,
    tolerance_ratio: float = LATENCY_TOLERANCE_RATIO,
) -> tuple[bool, tuple[str, ...]]:
    """Trip if any observed latency exceeds its baseline * ``tolerance_ratio``."""
    details: list[str] = []
    for name, observed, base in (
        ("ttfa_ms", sample.ttfa_ms, baseline.ttfa_ms),
        ("ttft_ms", sample.ttft_ms, baseline.ttft_ms),
        ("tts_ms", sample.tts_ms, baseline.tts_ms),
    ):
        if observed is None or base <= 0:
            continue
        if observed > base * tolerance_ratio:
            details.append(
                f"{name} {observed:.0f}ms exceeds baseline {base:.0f}ms "
                f"x{tolerance_ratio:.2f} ({base * tolerance_ratio:.0f}ms)"
            )
    return (bool(details), tuple(details))


def detect_quality_regression(
    baseline: GuardrailBaseline,
    sample: GuardrailSample,
    *,
    abs_increase: float = QUALITY_RATE_ABS_INCREASE,
) -> tuple[bool, tuple[str, ...]]:
    """Trip if the user-correction/re-ask/barge-in rate climbs by more than
    ``abs_increase`` over baseline."""
    if sample.correction_rate is None:
        return (False, ())
    delta = sample.correction_rate - baseline.correction_rate
    if delta > abs_increase:
        return (
            True,
            (
                f"correction rate climbed {delta:+.2f} "
                f"({baseline.correction_rate:.2f} -> {sample.correction_rate:.2f})",
            ),
        )
    return (False, ())


def detect_error_regression(
    baseline: GuardrailBaseline,
    sample: GuardrailSample,
    *,
    abs_increase: float = ERROR_RATE_ABS_INCREASE,
) -> tuple[bool, tuple[str, ...]]:
    """Trip if the error/fail-open/validator-block rate climbs by more than
    ``abs_increase`` over baseline."""
    if sample.error_rate is None:
        return (False, ())
    delta = sample.error_rate - baseline.error_rate
    if delta > abs_increase:
        return (
            True,
            (
                f"error rate climbed {delta:+.2f} "
                f"({baseline.error_rate:.2f} -> {sample.error_rate:.2f})",
            ),
        )
    return (False, ())


def detect_resource_ceiling(
    sample: GuardrailSample,
    *,
    cap_mb: float = VRAM_CAP_MB,
    headroom_ratio: float = VRAM_HEADROOM_RATIO,
) -> tuple[bool, tuple[str, ...]]:
    """Trip if observed VRAM peak approaches the hard cap."""
    if sample.vram_peak_mb is None:
        return (False, ())
    limit = cap_mb * headroom_ratio
    if sample.vram_peak_mb >= limit:
        return (
            True,
            (f"VRAM peak {sample.vram_peak_mb:.0f}MB >= {limit:.0f}MB ({cap_mb:.0f}MB cap)",),
        )
    return (False, ())


def evaluate_guardrails(
    baseline: GuardrailBaseline,
    sample: GuardrailSample,
    *,
    config: Optional[GuardrailConfig] = None,
) -> GuardrailVerdict:
    """Run every enabled guardrail and aggregate into one verdict.

    Any single trip means the applied change must be auto-reverted.
    """
    cfg = config or GuardrailConfig()
    tripped_guards: list[str] = []
    details: list[str] = []

    if "latency" in cfg.enabled:
        hit, d = detect_latency_regression(
            baseline, sample, tolerance_ratio=cfg.latency_tolerance_ratio
        )
        if hit:
            tripped_guards.append("latency")
            details.extend(d)
    if "quality" in cfg.enabled:
        hit, d = detect_quality_regression(baseline, sample, abs_increase=cfg.quality_abs_increase)
        if hit:
            tripped_guards.append("quality")
            details.extend(d)
    if "error" in cfg.enabled:
        hit, d = detect_error_regression(baseline, sample, abs_increase=cfg.error_abs_increase)
        if hit:
            tripped_guards.append("error")
            details.extend(d)
    if "resource" in cfg.enabled:
        hit, d = detect_resource_ceiling(
            sample, cap_mb=cfg.vram_cap_mb, headroom_ratio=cfg.vram_headroom_ratio
        )
        if hit:
            tripped_guards.append("resource")
            details.extend(d)

    return GuardrailVerdict(
        tripped=bool(tripped_guards),
        tripped_guards=tuple(tripped_guards),
        details=tuple(details),
    )


def summarize_guardrail_verdict(verdict: GuardrailVerdict) -> str:
    """A short, TTS-safe one-line summary of a verdict."""
    if not verdict.tripped:
        return "all guardrails passed"
    return "guardrail regression: " + ", ".join(verdict.tripped_guards)


# --- rollback-frequency audit (the brake) -----------------------------------


@dataclass(frozen=True)
class RollbackRecord:
    """An audit entry for one auto-revert."""

    surface: str
    change_id: str
    guardrail: str
    metric_delta: str = ""
    at: float = 0.0


@dataclass(frozen=True)
class SurfaceRollbackStats:
    """Per-surface rollback statistics over the recent window."""

    surface: str
    applied: int
    reverted: int
    rate: float
    window_size: int


def compute_rollback_rate(applied: int, reverted: int) -> float:
    """``reverted / applied`` (0.0 when nothing applied)."""
    if applied <= 0:
        return 0.0
    return reverted / applied


def should_demote_for_rollback_rate(
    rate: float,
    *,
    applied: int,
    threshold: float = ROLLBACK_DEMOTE_THRESHOLD,
    min_samples: int = ROLLBACK_MIN_SAMPLES,
) -> bool:
    """A surface should be demoted when it has enough samples AND its
    rollback rate exceeds the threshold."""
    return applied >= min_samples and rate > threshold


def format_rollback_audit_line(record: RollbackRecord) -> str:
    """Render a rollback record for the periodic digest / audit log."""
    base = f"reverted {record.change_id} on '{record.surface}' (guardrail: {record.guardrail})"
    if record.metric_delta:
        base += f" -- {record.metric_delta}"
    return base


class RollbackAudit:
    """Append-only audit of apply outcomes + per-surface rollback rates.

    Each applied change reports its outcome (kept or reverted) via
    :meth:`note_outcome`; the audit keeps a bounded per-surface window of
    outcomes so :meth:`rollback_rate` reflects RECENT behaviour, and
    :meth:`should_demote` is the negative-feedback brake on autonomy.
    """

    def __init__(self, *, window: int = ROLLBACK_WINDOW) -> None:
        self._window = max(1, int(window))
        self._outcomes: dict[str, Deque[bool]] = defaultdict(lambda: deque(maxlen=self._window))
        self._applied_total: dict[str, int] = defaultdict(int)
        self._reverted_total: dict[str, int] = defaultdict(int)
        self._records: list[RollbackRecord] = []

    def note_outcome(
        self, surface: str, *, reverted: bool, record: Optional[RollbackRecord] = None
    ) -> None:
        """Record that a change on ``surface`` was kept (``reverted=False``)
        or auto-reverted (``reverted=True``). When reverted, ``record`` is
        appended to the audit trail."""
        self._outcomes[surface].append(bool(reverted))
        self._applied_total[surface] += 1
        if reverted:
            self._reverted_total[surface] += 1
            if record is not None:
                self._records.append(record)

    def rollback_rate(self, surface: str) -> float:
        """The rollback rate over the recent window for ``surface``."""
        window = self._outcomes.get(surface)
        if not window:
            return 0.0
        return sum(1 for r in window if r) / len(window)

    def stats(self, surface: str) -> SurfaceRollbackStats:
        """Per-surface stats over the recent window."""
        window = self._outcomes.get(surface, deque())
        reverted = sum(1 for r in window if r)
        return SurfaceRollbackStats(
            surface=surface,
            applied=len(window),
            reverted=reverted,
            rate=self.rollback_rate(surface),
            window_size=self._window,
        )

    def should_demote(
        self,
        surface: str,
        *,
        threshold: float = ROLLBACK_DEMOTE_THRESHOLD,
        min_samples: int = ROLLBACK_MIN_SAMPLES,
    ) -> bool:
        """Whether ``surface`` should be demoted for an excessive recent
        rollback rate."""
        s = self.stats(surface)
        return should_demote_for_rollback_rate(
            s.rate, applied=s.applied, threshold=threshold, min_samples=min_samples
        )

    def records(self) -> tuple[RollbackRecord, ...]:
        """The append-only revert audit trail."""
        return tuple(self._records)

    def surfaces(self) -> tuple[str, ...]:
        """Every surface seen so far."""
        return tuple(self._outcomes.keys())

    def totals(self, surface: str) -> tuple[int, int]:
        """Lifetime ``(applied, reverted)`` for ``surface``."""
        return (self._applied_total.get(surface, 0), self._reverted_total.get(surface, 0))


__all__ = [
    "LATENCY_TOLERANCE_RATIO",
    "QUALITY_RATE_ABS_INCREASE",
    "ERROR_RATE_ABS_INCREASE",
    "VRAM_CAP_MB",
    "VRAM_HEADROOM_RATIO",
    "ROLLBACK_DEMOTE_THRESHOLD",
    "ROLLBACK_MIN_SAMPLES",
    "ROLLBACK_WINDOW",
    "GUARDRAILS",
    "GuardrailBaseline",
    "GuardrailSample",
    "GuardrailVerdict",
    "GuardrailConfig",
    "detect_latency_regression",
    "detect_quality_regression",
    "detect_error_regression",
    "detect_resource_ceiling",
    "evaluate_guardrails",
    "summarize_guardrail_verdict",
    "RollbackRecord",
    "SurfaceRollbackStats",
    "compute_rollback_rate",
    "should_demote_for_rollback_rate",
    "format_rollback_audit_line",
    "RollbackAudit",
]
