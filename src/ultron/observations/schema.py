"""Canonical observation schema.

One observation row per "meaningful event" anywhere in Ultron. The shape
is deliberately narrow: only fields that benefit cross-subsystem
analysis live at the top level. Subsystem-specific detail goes in the
``extra`` mapping so the schema can stay stable while individual writers
evolve.
"""

from __future__ import annotations

import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional


KNOWN_SUBSYSTEMS: frozenset[str] = frozenset(
    {
        "routing",
        "addressing",
        "memory",
        "llm",
        "tts",
        "stt",
        "coding",
        "web_search",
        "orchestrator",
        "desktop",
        "safety",
        "openclaw",
    }
)

KNOWN_OUTCOMES: frozenset[str] = frozenset(
    {
        "success",
        "success_with_followup",
        "corrected",
        "failed",
        "unknown_yet",
    }
)


def new_event_id() -> str:
    """Return a fresh 16-char hex event id.

    Short enough to grep comfortably in JSONL; long enough that collisions
    inside any practical observation log are vanishingly unlikely.
    """
    return secrets.token_hex(8)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Observation:
    """One canonical event row.

    Required fields:

    * ``event_id`` -- stable identifier for this event.
    * ``timestamp`` -- ISO 8601 UTC of when the event occurred.
    * ``subsystem`` -- one of :data:`KNOWN_SUBSYSTEMS` (or a custom string;
      enforcement is convention, not validation, so new subsystems don't
      need a schema change to start emitting).
    * ``event_type`` -- free-form short string identifying the event
      kind within the subsystem (e.g. ``"classifier_verdict"``,
      ``"retrieval"``, ``"llm_call"``).

    Optional fields:

    * ``parent_event_id`` -- for causally-linked events (e.g. an
      ``llm_call`` whose prompt came from a ``retrieval``).
    * ``intent_kind`` -- routing intent kind value when known.
    * ``outcome`` -- one of :data:`KNOWN_OUTCOMES` or ``None``. Most
      writers stamp ``unknown_yet`` and let the outcome resolver fill in
      the verdict via a separate ``outcome_resolution`` row.
    * ``latency_ms`` -- wall-clock duration of the event if applicable.
    * ``tokens_used`` -- LLM call cost (input+output) when applicable.
    * ``lineage_ids`` -- list of memory / source ids whose content fed
      this event. Used by the post-response overlap pass to mark which
      lineage entries were actually consumed.
    * ``payload_ref`` -- path or key pointing to the specialized log
      with full detail (e.g. ``logs/routing_decisions.jsonl#L123``).
    * ``extra`` -- free-form structured payload. Keep small; the
      ``payload_ref`` is the right place for verbose detail.
    """

    event_id: str
    timestamp: str
    subsystem: str
    event_type: str
    parent_event_id: Optional[str] = None
    intent_kind: Optional[str] = None
    outcome: Optional[str] = None
    latency_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    lineage_ids: tuple[str, ...] = field(default_factory=tuple)
    payload_ref: Optional[str] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        subsystem: str,
        event_type: str,
        event_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        parent_event_id: Optional[str] = None,
        intent_kind: Optional[str] = None,
        outcome: Optional[str] = None,
        latency_ms: Optional[float] = None,
        tokens_used: Optional[int] = None,
        lineage_ids: Optional[tuple[str, ...]] = None,
        payload_ref: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> "Observation":
        """Construct an :class:`Observation`, filling in defaults.

        Callers that don't pass ``event_id`` or ``timestamp`` get freshly
        generated values; this is the ergonomic path most call sites
        should use.
        """
        return cls(
            event_id=event_id or new_event_id(),
            timestamp=timestamp or _utc_now_iso(),
            subsystem=subsystem,
            event_type=event_type,
            parent_event_id=parent_event_id,
            intent_kind=intent_kind,
            outcome=outcome,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            lineage_ids=tuple(lineage_ids or ()),
            payload_ref=payload_ref,
            extra=dict(extra or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (``lineage_ids`` becomes a list)."""
        payload = asdict(self)
        payload["lineage_ids"] = list(self.lineage_ids)
        payload["extra"] = dict(self.extra)
        return payload
