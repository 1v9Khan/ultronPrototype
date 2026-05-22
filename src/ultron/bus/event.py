"""Event definition primitives.

Mirrors opencode's ``BusEvent.define(type, schema)`` factory. An event
definition pairs a stable string ``type`` (the channel name) with a
schema describing the payload shape.

Opencode uses Effect's Schema for runtime codecs; we use a simpler
contract: schema is a ``dict[str, type]`` mapping field name to a
Python type. Validation is best-effort -- malformed payloads are
logged at WARNING but still delivered, so a producer bug never
silently wedges consumers (matches opencode's
``Effect.tryPromise(...).pipe(Effect.ignore)`` posture for callback
errors).

Every emitted event flows through :class:`EventPayload`, which carries
a unique id (for de-duplication / tracing), the channel ``type``, the
``properties`` dict, and the publish timestamp.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class BusEvent:
    """An event-type definition. Constructed via :meth:`define`."""

    type: str
    schema: Mapping[str, type]
    description: str = ""

    @classmethod
    def define(
        cls,
        type: str,
        schema: Mapping[str, type],
        description: str = "",
    ) -> "BusEvent":
        """Register a new event type.

        Args:
            type: dotted channel name, e.g. ``"turn.started"`` or
                ``"project.indexed"``. Must be globally unique within
                the process; collisions are detected at first
                :func:`~ultron.bus.publish` site, not here, so re-
                imports during test reloads don't crash.
            schema: mapping ``{field_name: python_type}`` describing
                the expected payload. Validation is best-effort.
            description: free-text human-readable description for
                docs / introspection.
        """
        return cls(type=type, schema=dict(schema), description=description)

    def validate(self, properties: Mapping[str, Any]) -> Optional[str]:
        """Best-effort check of ``properties`` against ``schema``.

        Returns ``None`` when valid, or an error string when a required
        field is missing or has the wrong type. Producers should NOT
        raise on a non-None result -- the bus delivers the event
        regardless and logs the discrepancy.
        """
        missing = [k for k in self.schema if k not in properties]
        if missing:
            return f"missing required fields: {sorted(missing)}"
        wrong_type = []
        for field_name, expected in self.schema.items():
            value = properties.get(field_name)
            if value is None:
                continue  # None passes; Optional fields signal absence
            if not isinstance(value, expected):
                wrong_type.append(
                    f"{field_name} expected {expected.__name__}, "
                    f"got {type(value).__name__}",
                )
        if wrong_type:
            return "; ".join(wrong_type)
        return None


@dataclass
class EventPayload:
    """A published-event envelope.

    Mirrors opencode's ``{ id, type, properties }`` shape with an
    added ``published_at`` monotonic timestamp for downstream
    latency profiling.
    """

    id: str
    type: str
    properties: Dict[str, Any]
    published_at: float = field(default_factory=time.monotonic)

    @classmethod
    def make(
        cls,
        event_def: BusEvent,
        properties: Mapping[str, Any],
        id: Optional[str] = None,
    ) -> "EventPayload":
        """Construct an envelope with auto-generated id when omitted."""
        return cls(
            id=id or f"evt_{uuid.uuid4().hex[:16]}",
            type=event_def.type,
            properties=dict(properties),
        )
