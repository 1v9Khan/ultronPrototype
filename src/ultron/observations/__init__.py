"""Canonical observation framework.

Every meaningful action in Ultron emits one row to ``data/observations.jsonl``
via :class:`ObservationWriter`. The row format is intentionally narrow and
shared across subsystems so cross-cutting analysis (eval harness, latency
audits, learning loops) reads a single file instead of joining N specialised
logs.

Public surface:

* :class:`Observation` -- frozen dataclass; the canonical schema.
* :func:`new_event_id` -- 16-char hex id constructor.
* :class:`ObservationWriter` -- thread-safe JSONL appender.
* :func:`get_observation_writer` / :func:`set_observation_writer` --
  singleton accessor + test injection.
* :func:`emit_observation` -- module-level convenience for the singleton.

Fail-open contract: writer failures NEVER raise to callers. A WARN log
line is emitted once per failure-type-and-path; the observation is
dropped. The voice path must never block on observation IO.
"""

from .integrations import (
    observe_addressing_verdict,
    observe_llm_call,
    observe_llm_thinking_drift_sample,
    observe_retrieval,
    observe_routing_verdict,
)
from .lineage_overlap import (
    LineageOverlap,
    UsageEmitSummary,
    compute_lineage_overlap,
    emit_lineage_usage_rows,
)
from .outcome_resolver import (
    OutcomeResolver,
    Resolution,
    ResolverSummary,
    resolve_outcomes,
)
from .schema import (
    KNOWN_OUTCOMES,
    KNOWN_SUBSYSTEMS,
    Observation,
    new_event_id,
)
from .writer import (
    ObservationWriter,
    emit_observation,
    get_observation_writer,
    set_observation_writer,
)

__all__ = [
    "KNOWN_OUTCOMES",
    "KNOWN_SUBSYSTEMS",
    "LineageOverlap",
    "Observation",
    "ObservationWriter",
    "OutcomeResolver",
    "Resolution",
    "ResolverSummary",
    "UsageEmitSummary",
    "compute_lineage_overlap",
    "emit_lineage_usage_rows",
    "emit_observation",
    "get_observation_writer",
    "new_event_id",
    "observe_addressing_verdict",
    "observe_llm_call",
    "observe_llm_thinking_drift_sample",
    "observe_retrieval",
    "observe_routing_verdict",
    "resolve_outcomes",
    "set_observation_writer",
]
