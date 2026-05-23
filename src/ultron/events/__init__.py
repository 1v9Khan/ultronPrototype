"""Canonical event store -- typed, queryable, replayable history of record.

Pattern lineage attributed in ``THIRD_PARTY_NOTICES.md``.

The OpenHands V1 server's ``EventService`` ABC + per-event JSON file +
opaque page-id pagination + per-conversation export-as-zip is the
shape this package ports. Ultron's version is sync (single-process,
voice-first) and per-session-scoped (the orchestrator owns the
session id) but the contract surface is the same: ``save_event`` /
``get_event`` / ``search_events`` / ``count_events`` / a
``batch_get_events`` convenience built on the others / and an
``export_session`` zip exporter for downloadable trajectories.

Hash chaining (T13 from the catalog) builds on top of the store:
each event carries ``chain_prev_hash`` + ``chain_hash`` so the
canonical sequence is tamper-evident at the session level.

Three backends ship:

* :class:`MemoryEventStore` for tests + in-process work.
* :class:`JsonlEventStore` for production -- append-only JSONL per
  session at ``data/events/<session_id>.jsonl``.
* :class:`QdrantEventStore` (opt-in) for callers that want events
  to be retrievable via the existing memory embedding pipeline.
"""

from ultron.events.chain import (
    ChainVerificationError,
    ChainVerificationResult,
    compute_event_chain_hash,
    verify_chain,
)
from ultron.events.export import (
    SessionExport,
    export_session_to_bytes,
    export_session_to_path,
)
from ultron.events.models import (
    DEFAULT_PAGE_LIMIT,
    DEFAULT_SEARCH_SORT,
    EventKind,
    EventPage,
    EventQuery,
    EventSortOrder,
    StoredEvent,
    canonical_event_json,
    new_event_id,
)
from ultron.events.store import (
    EventStore,
    EventStoreError,
    JsonlEventStore,
    MemoryEventStore,
    QdrantEventStore,
    build_event_store,
    get_event_store,
    reset_event_store_for_testing,
    set_event_store,
)
from ultron.events.bus_sink import (
    BusEventSink,
    install_bus_event_sink,
    uninstall_bus_event_sink,
)
from ultron.events.callbacks import (
    CallbackProcessor,
    CallbackRegistry,
    CallbackResult,
    CallbackResultStatus,
    CallbackStatus,
    FunctionProcessor,
    RegisteredCallback,
    get_callback_registry,
    reset_callback_registry_for_testing,
    set_callback_registry,
)
from ultron.events.processors import (
    ChannelGuardProcessor,
    CountingCallbackProcessor,
    LoggingCallbackProcessor,
    MemoryWriteProcessor,
    SkillActivatorProcessor,
    ThresholdSnapshotProcessor,
    build_default_processors,
)

__all__ = [
    "BusEventSink",
    "CallbackProcessor",
    "CallbackRegistry",
    "CallbackResult",
    "CallbackResultStatus",
    "CallbackStatus",
    "ChainVerificationError",
    "ChainVerificationResult",
    "ChannelGuardProcessor",
    "CountingCallbackProcessor",
    "FunctionProcessor",
    "LoggingCallbackProcessor",
    "MemoryWriteProcessor",
    "RegisteredCallback",
    "SkillActivatorProcessor",
    "ThresholdSnapshotProcessor",
    "build_default_processors",
    "get_callback_registry",
    "reset_callback_registry_for_testing",
    "set_callback_registry",
    "DEFAULT_PAGE_LIMIT",
    "DEFAULT_SEARCH_SORT",
    "EventKind",
    "EventPage",
    "EventQuery",
    "EventSortOrder",
    "EventStore",
    "EventStoreError",
    "JsonlEventStore",
    "MemoryEventStore",
    "QdrantEventStore",
    "SessionExport",
    "StoredEvent",
    "build_event_store",
    "canonical_event_json",
    "compute_event_chain_hash",
    "export_session_to_bytes",
    "export_session_to_path",
    "get_event_store",
    "install_bus_event_sink",
    "new_event_id",
    "reset_event_store_for_testing",
    "set_event_store",
    "uninstall_bus_event_sink",
    "verify_chain",
]
