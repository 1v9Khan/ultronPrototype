"""Typed event bus -- ported from opencode's bus/ module.

A small in-process pub/sub layer used by every subsystem that wants
to publish or subscribe to lifecycle events (turn-started, gate-
verdict, project-indexed, etc) without hard-wiring callback
registrations through the orchestrator.

The opencode original (TypeScript / Effect) uses two PubSub queues
(typed + wildcard) with eager-subscribe semantics to close a race
window where a publish-before-stream-pull dropped events. Ours
mirrors that design with plain ``threading`` primitives:

  * ``BusEvent.define(name, schema)`` registers an event type.
  * ``publish(event_def, properties)`` fans out to typed-subscribers
    and wildcard-subscribers immediately on the calling thread.
  * ``subscribe(event_def, callback)`` and ``subscribe_all(callback)``
    register callbacks; both return an ``unsubscribe`` callable that
    is safe to call from any thread.

Threading model:
  * Subscribers run on the publishing thread. Keep callbacks fast.
    Subscribers that want async behavior should hand the payload off
    to their own queue / thread.
  * The registry is protected by an ``RLock``; subscribe + publish
    are race-safe (the eager-acquire pattern means no lost events
    even if a subscribe lands a microsecond before a publish).

Fail-open contract:
  * A callback exception is swallowed + logged at WARNING. One bad
    subscriber never breaks others.
  * Schema validation on payload is best-effort -- malformed payloads
    are passed through (logged) so a producer bug never wedges
    consumers.

Public API mirrors opencode's surface as closely as Python allows.
``Bus`` is a singleton accessed via :func:`get_bus`; module-level
:func:`publish`, :func:`subscribe`, :func:`subscribe_all` are
shortcuts.
"""

from ultron.bus.event import BusEvent, EventPayload
from ultron.bus.service import (
    DEFAULT_SLOW_SUBSCRIBER_WARN_MS,
    Bus,
    get_bus,
    publish,
    reset_bus_for_testing,
    set_slow_subscriber_recorder,
    subscribe,
    subscribe_all,
)
from ultron.bus.events import (
    BUS_EVENT_CATALOG,
    CodingFileChangedEvent,
    GamingEngagedEvent,
    GamingDisengagedEvent,
    GateVerdictEvent,
    LLMStreamCompleteEvent,
    LLMStreamTokenEvent,
    MemoryRetrievedEvent,
    ProjectDigestGeneratedEvent,
    ProjectIndexedEvent,
    RoutingClassifiedEvent,
    SafetyViolatedEvent,
    STTTranscribedEvent,
    SupervisorDecidedEvent,
    TTSPlayedEvent,
    TurnCompletedEvent,
    TurnStartedEvent,
    VRAMReclaimedEvent,
)

__all__ = [
    "BUS_EVENT_CATALOG",
    "Bus",
    "BusEvent",
    "CodingFileChangedEvent",
    "DEFAULT_SLOW_SUBSCRIBER_WARN_MS",
    "EventPayload",
    "GamingEngagedEvent",
    "GamingDisengagedEvent",
    "GateVerdictEvent",
    "LLMStreamCompleteEvent",
    "LLMStreamTokenEvent",
    "MemoryRetrievedEvent",
    "ProjectDigestGeneratedEvent",
    "ProjectIndexedEvent",
    "RoutingClassifiedEvent",
    "SafetyViolatedEvent",
    "STTTranscribedEvent",
    "SupervisorDecidedEvent",
    "TTSPlayedEvent",
    "TurnCompletedEvent",
    "TurnStartedEvent",
    "VRAMReclaimedEvent",
    "get_bus",
    "publish",
    "reset_bus_for_testing",
    "set_slow_subscriber_recorder",
    "subscribe",
    "subscribe_all",
]
