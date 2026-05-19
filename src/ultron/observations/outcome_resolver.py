"""Outcome tagging resolver.

Reads :class:`Observation` rows from a JSONL log, infers their final
outcome from on-event signals + cross-event correction patterns, and
emits ``outcome_resolution`` rows that reference the original
``event_id``. Resolution is **post-hoc** -- it runs as part of the
maintenance pipeline, not on the voice hot path -- so it never adds
latency to a live turn.

Resolution rules (V1):

1. **LLM stream events resolve from own fields.**
   * ``completed=True`` and ``canceled=False`` -> ``success``.
   * ``canceled=True`` -> ``corrected`` (caller invalidated /
     barged in).
   * ``completed=False`` and ``canceled=False`` -> ``failed``
     (truncated mid-stream).

2. **LLM blocking events default to success.** The emit point sits
   AFTER the call returns successfully; failed calls don't reach it
   today, so any emitted ``generate`` event implies success.

3. **Routing / addressing / retrieval events resolve from
   cross-event signals.** If any subsequent event within
   ``window_seconds`` (default 30) on the same correlation track
   matches a correction pattern (CANCEL intent / addressing
   default_silent / explicit user-corrects-self LLM call), the
   original event resolves to ``corrected``. Otherwise it resolves
   to ``success``.

4. **Events less than ``min_age_seconds`` old (default 30) are
   left as ``unknown_yet``.** Resolution needs the following window
   to have actually happened.

Each resolution becomes a new observation row with
``event_type="outcome_resolution"`` and ``parent_event_id`` set to
the original event. Readers reconcile by walking the file and
preferring the most-recent resolution per ``parent_event_id``.

This is a side-effect-only writer. Callers either supply a writer
explicitly (preferred for tests) or let the singleton handle dispatch.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

from .schema import Observation
from .writer import ObservationWriter, emit_observation, get_observation_writer

LOGGER = logging.getLogger("ultron.observations.resolver")

DEFAULT_OBSERVATIONS_PATH = Path("data") / "observations.jsonl"
DEFAULT_WINDOW_SECONDS = 30.0
DEFAULT_MIN_AGE_SECONDS = 30.0


@dataclass(frozen=True)
class _LoadedEvent:
    """Lightweight in-memory view of one observation row."""

    event_id: str
    timestamp_iso: str
    epoch_seconds: float
    subsystem: str
    event_type: str
    outcome: Optional[str]
    intent_kind: Optional[str]
    parent_event_id: Optional[str]
    extra: Mapping[str, object]


@dataclass(frozen=True)
class Resolution:
    """One outcome assignment for a previously-emitted event."""

    parent_event_id: str
    outcome: str
    reason: str

    def to_observation(self) -> Observation:
        """Return the :class:`Observation` row that records this resolution.

        Resolution rows carry the resolved outcome in their own
        ``outcome`` field too so reader code that only looks at the
        top-level field still sees the verdict.
        """
        return Observation.create(
            subsystem="observations",
            event_type="outcome_resolution",
            outcome=self.outcome,
            parent_event_id=self.parent_event_id,
            extra={"reason": self.reason},
        )


# ---------------------------------------------------------------------------
# Reader helpers
# ---------------------------------------------------------------------------


def _parse_iso_to_epoch(iso: str) -> float:
    """Best-effort ISO -> epoch-seconds; returns 0.0 on parse failure."""
    if not iso:
        return 0.0
    try:
        # ``Observation.timestamp`` is ISO 8601 with timezone offset.
        return datetime.fromisoformat(iso).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _load_observations(path: Path) -> List[_LoadedEvent]:
    """Read all observations from ``path``. Returns ``[]`` on missing file."""
    if not path.exists():
        return []
    rows: List[_LoadedEvent] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                LOGGER.debug("skip malformed observation line %d in %s", line_no, path)
                continue
            event_id = payload.get("event_id")
            if not isinstance(event_id, str):
                continue
            timestamp_iso = payload.get("timestamp") or ""
            rows.append(
                _LoadedEvent(
                    event_id=event_id,
                    timestamp_iso=timestamp_iso,
                    epoch_seconds=_parse_iso_to_epoch(timestamp_iso),
                    subsystem=payload.get("subsystem") or "",
                    event_type=payload.get("event_type") or "",
                    outcome=payload.get("outcome"),
                    intent_kind=payload.get("intent_kind"),
                    parent_event_id=payload.get("parent_event_id"),
                    extra=payload.get("extra") or {},
                )
            )
    rows.sort(key=lambda r: r.epoch_seconds)
    return rows


def _collect_existing_resolutions(events: Sequence[_LoadedEvent]) -> set[str]:
    """Set of parent_event_ids that already have a resolution row."""
    out: set[str] = set()
    for ev in events:
        if (
            ev.event_type == "outcome_resolution"
            and ev.parent_event_id
        ):
            out.add(ev.parent_event_id)
    return out


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def _llm_stream_outcome(extra: Mapping[str, object]) -> tuple[str, str]:
    """Outcome for an ``llm.generate_stream`` event from its own fields."""
    completed = bool(extra.get("completed", False))
    canceled = bool(extra.get("canceled", False))
    if canceled:
        return "corrected", "stream canceled by caller (likely barge-in)"
    if completed:
        return "success", "stream completed cleanly"
    return "failed", "stream did not complete and was not canceled"


def _is_correction_signal(ev: _LoadedEvent) -> Optional[str]:
    """Return a short reason if ``ev`` indicates a correction, else None.

    Correction signals observed without raw text:

    * ``routing`` event whose ``intent_kind`` is ``cancel``.
    * ``addressing`` event whose ``extra.decision`` is ``NOT_ADDRESSED``
      AND whose ``extra.classifier_source`` is one of the explicit
      negative-rule sources (``rule`` with a NOT_ADDRESSED decision is
      not always a correction -- it could be normal silence-stream --
      so we limit the signal to the addressing path that fired in
      response to a recent assistant turn, gated by the time window).
    * ``llm.generate_stream`` event whose ``canceled`` flag is True.
    """
    if ev.subsystem == "routing" and ev.intent_kind == "cancel":
        return "subsequent CANCEL intent"
    if ev.subsystem == "addressing":
        decision = ev.extra.get("decision")
        if decision == "NOT_ADDRESSED":
            return "subsequent NOT_ADDRESSED verdict"
    if ev.subsystem == "llm" and ev.event_type == "generate_stream":
        if bool(ev.extra.get("canceled", False)):
            return "subsequent LLM stream canceled"
    return None


def _resolve_event(
    ev: _LoadedEvent,
    later_events: Sequence[_LoadedEvent],
    *,
    window_seconds: float,
) -> Optional[Resolution]:
    """Resolve ``ev`` against the events that came after it.

    Returns None when the event can't be resolved yet (insufficient
    follow-up window).
    """
    # 1) LLM events resolve from own fields.
    if ev.subsystem == "llm":
        if ev.event_type == "generate_stream":
            outcome, reason = _llm_stream_outcome(ev.extra)
        elif ev.event_type == "generate":
            outcome, reason = "success", "blocking generate returned"
        else:
            outcome, reason = "success", f"llm event {ev.event_type}"
        return Resolution(ev.event_id, outcome, reason)

    # 2) Other events depend on follow-up window.
    for later in later_events:
        if later.event_id == ev.event_id:
            continue
        if later.epoch_seconds - ev.epoch_seconds > window_seconds:
            break
        signal = _is_correction_signal(later)
        if signal:
            return Resolution(ev.event_id, "corrected", signal)

    return Resolution(ev.event_id, "success", "no correction in follow-up window")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


@dataclass
class ResolverSummary:
    """Counts + sampled-detail emitted by :meth:`OutcomeResolver.resolve`."""

    scanned: int = 0
    already_resolved: int = 0
    resolved_now: int = 0
    deferred_age: int = 0
    emitted_failures: int = 0
    resolutions: List[Resolution] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "scanned": self.scanned,
            "already_resolved": self.already_resolved,
            "resolved_now": self.resolved_now,
            "deferred_age": self.deferred_age,
            "emitted_failures": self.emitted_failures,
            "by_outcome": _count_by(self.resolutions, lambda r: r.outcome),
        }


def _count_by(items: Iterable, key) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        k = key(item)
        counts[k] = counts.get(k, 0) + 1
    return counts


class OutcomeResolver:
    """Resolve ``unknown_yet`` observation outcomes from history.

    Parameters
    ----------
    observations_path:
        JSONL file to read. Defaults to ``data/observations.jsonl``.
    window_seconds:
        Cross-event correction-signal window. Events with a follow-up
        within this window get checked for correction markers.
    min_age_seconds:
        Don't resolve events younger than this -- the follow-up window
        hasn't had a chance to develop yet. Defaults to 30 s; tune up
        if the run cadence is sparse.
    writer:
        Where resolution rows go. Defaults to the module singleton.
    now_provider:
        Returns "current epoch seconds". Injectable so tests can drive
        the clock forward without sleeping.
    """

    def __init__(
        self,
        observations_path: Path = DEFAULT_OBSERVATIONS_PATH,
        *,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        min_age_seconds: float = DEFAULT_MIN_AGE_SECONDS,
        writer: Optional[ObservationWriter] = None,
        now_provider=None,
    ) -> None:
        self._path = Path(observations_path)
        self._window_seconds = float(window_seconds)
        self._min_age_seconds = float(min_age_seconds)
        self._writer = writer
        self._now = now_provider or time.time

    def resolve(self) -> ResolverSummary:
        """Walk the observation log, emit resolutions for ripe events.

        Returns a :class:`ResolverSummary` describing what happened.
        Never raises -- IO failures degrade to "scanned 0" and log
        WARN once.
        """
        summary = ResolverSummary()
        try:
            events = _load_observations(self._path)
        except OSError as exc:
            LOGGER.warning("outcome-resolver could not read %s: %s", self._path, exc)
            return summary

        if not events:
            return summary

        already = _collect_existing_resolutions(events)
        now = float(self._now())

        for idx, ev in enumerate(events):
            summary.scanned += 1
            if ev.event_type == "outcome_resolution":
                continue
            if ev.outcome and ev.outcome != "unknown_yet":
                continue
            if ev.event_id in already:
                summary.already_resolved += 1
                continue
            if (now - ev.epoch_seconds) < self._min_age_seconds:
                summary.deferred_age += 1
                continue

            later = events[idx + 1:]
            resolution = _resolve_event(
                ev, later, window_seconds=self._window_seconds
            )
            if resolution is None:
                summary.deferred_age += 1
                continue
            summary.resolutions.append(resolution)
            summary.resolved_now += 1
            try:
                obs = resolution.to_observation()
                writer = self._writer if self._writer is not None else get_observation_writer()
                if not writer.emit(obs):
                    summary.emitted_failures += 1
            except Exception as exc:  # noqa: BLE001 - never let emit kill the resolver
                summary.emitted_failures += 1
                LOGGER.warning("outcome-resolver emit failed: %s", exc)

        return summary


# Convenience for callers that don't want to build the class explicitly.
def resolve_outcomes(
    observations_path: Path = DEFAULT_OBSERVATIONS_PATH,
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    min_age_seconds: float = DEFAULT_MIN_AGE_SECONDS,
    writer: Optional[ObservationWriter] = None,
    now_provider=None,
) -> ResolverSummary:
    """One-shot resolve pass; returns the summary."""
    resolver = OutcomeResolver(
        observations_path=observations_path,
        window_seconds=window_seconds,
        min_age_seconds=min_age_seconds,
        writer=writer,
        now_provider=now_provider,
    )
    return resolver.resolve()
