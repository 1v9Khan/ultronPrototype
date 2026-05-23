"""Subscribe to the typed bus and write every payload into the event store.

Adopts the OpenHands "every conversation fact is an event row" mental
model on top of ultron's existing pub/sub bus. The sink is opt-in --
``install_bus_event_sink(store)`` subscribes once; ``uninstall_bus_event_sink()``
removes the subscription. The store implementation is whatever the
caller wired (memory / jsonl / qdrant).

The sink converts each bus event into a :class:`StoredEvent` by
extracting:

* ``kind`` from the event definition's ``type`` (or class name).
* ``session_id`` from the event envelope's payload (``session_id``
  field is conventional; falls back to ``"default"``).
* ``payload`` from the envelope's payload mapping (minus
  ``session_id`` so it isn't duplicated).

Errors are swallowed -- the bus dispatch loop already has a slow-
subscriber watchdog, but we add a try/except per dispatch so a bad
event can never wedge the entire bus.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from ultron.events.models import StoredEvent, new_event_id
from ultron.events.store import EventStore

logger = logging.getLogger(__name__)


class BusEventSink:
    """Glue subscriber: bus events -> EventStore writes (-> callback dispatch).

    Holds the unsubscribe callable so :func:`uninstall_bus_event_sink`
    can take it back down cleanly. Thread-safe; the underlying bus
    dispatches under its own lock.

    When the module-level callback registry is set
    (:func:`set_callback_registry`), each persisted event is also fed
    to :meth:`CallbackRegistry.execute_for_event` AFTER the store
    write returns -- a callback exception never loses the underlying
    event (the catalog's load-bearing invariant).
    """

    def __init__(
        self,
        store: EventStore,
        *,
        default_session_id: str = "default",
        dispatch_callbacks: bool = True,
    ) -> None:
        self._store = store
        self._default_session_id = default_session_id
        self._dispatch_callbacks = dispatch_callbacks
        self._sequence_counters: dict[str, int] = {}
        self._lock = threading.RLock()
        self._unsubscribe: Callable[[], None] | None = None
        self._dispatched = 0
        self._errors = 0
        self._callbacks_fired = 0

    @property
    def dispatched(self) -> int:
        with self._lock:
            return self._dispatched

    @property
    def errors(self) -> int:
        with self._lock:
            return self._errors

    def install(self) -> None:
        if self._unsubscribe is not None:
            return
        try:
            from ultron.bus import subscribe_all  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            logger.warning("bus event sink install skipped (no bus): %r", exc)
            return
        try:
            self._unsubscribe = subscribe_all(self._on_event)
        except Exception as exc:  # noqa: BLE001
            logger.warning("bus event sink install failed: %r", exc)
            self._unsubscribe = None

    def uninstall(self) -> None:
        cb = self._unsubscribe
        self._unsubscribe = None
        if cb is None:
            return
        try:
            cb()
        except Exception as exc:  # noqa: BLE001
            logger.warning("bus event sink uninstall failed: %r", exc)

    @property
    def callbacks_fired(self) -> int:
        with self._lock:
            return self._callbacks_fired

    def _on_event(self, envelope: Any) -> None:  # pragma: no cover - exercised via install_bus_event_sink
        try:
            event = self._envelope_to_stored_event(envelope)
            if event is None:
                return
            stored = self._store.save_event(event)
            with self._lock:
                self._dispatched += 1
            if self._dispatch_callbacks:
                self._fire_callbacks(stored)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._errors += 1
            logger.warning("bus event sink dispatch error: %r", exc)

    def _fire_callbacks(self, event: StoredEvent) -> None:
        # Import lazily so the bus sink stays usable without the
        # callbacks module loaded (e.g. minimal test setups).
        try:
            from ultron.events.callbacks import get_callback_registry
        except Exception:                                    # pragma: no cover
            return
        registry = get_callback_registry()
        if registry is None:
            return
        try:
            results = registry.execute_for_event(event)
        except Exception as exc:                            # noqa: BLE001
            logger.warning("callback dispatch failed for event %s: %r", event.id, exc)
            return
        with self._lock:
            self._callbacks_fired += len(results)

    def _envelope_to_stored_event(self, envelope: Any) -> StoredEvent | None:
        # Bus envelopes carry a ``properties`` dict-shaped payload and an
        # ``event_def`` reference. Defensive against shape drift.
        payload: dict[str, Any]
        kind: str
        event_def = getattr(envelope, "event_def", None)
        if event_def is not None and hasattr(event_def, "type"):
            kind = str(event_def.type)
        elif hasattr(envelope, "kind"):
            kind = str(envelope.kind)
        else:
            kind = envelope.__class__.__name__

        raw_props = getattr(envelope, "properties", None)
        if isinstance(raw_props, dict):
            payload = dict(raw_props)
        elif hasattr(envelope, "to_dict"):
            try:
                payload = dict(envelope.to_dict())  # type: ignore[arg-type]
            except Exception:
                payload = {}
        else:
            payload = {}

        session_id = (
            payload.pop("session_id", None)
            or getattr(envelope, "session_id", None)
            or self._default_session_id
        )

        timestamp: float
        ts = payload.pop("timestamp", None) or getattr(envelope, "timestamp", None)
        try:
            timestamp = float(ts) if ts is not None else time.time()
        except (TypeError, ValueError):
            timestamp = time.time()

        with self._lock:
            seq = self._sequence_counters.get(session_id, 0)
            self._sequence_counters[session_id] = seq + 1

        return StoredEvent.make(
            session_id=str(session_id),
            kind=kind,
            payload=payload,
            source="bus",
            timestamp=timestamp,
            event_id=new_event_id(),
            sequence=seq,
        )


# -- module-level installation helpers --


_SINK: BusEventSink | None = None
_SINK_LOCK = threading.RLock()


def install_bus_event_sink(
    store: EventStore,
    *,
    default_session_id: str = "default",
) -> BusEventSink:
    """Install a sink subscription pointed at ``store``.

    Idempotent: an existing sink is uninstalled first so callers can
    swap stores at runtime by re-calling.
    """

    global _SINK
    with _SINK_LOCK:
        if _SINK is not None:
            _SINK.uninstall()
            _SINK = None
        sink = BusEventSink(store, default_session_id=default_session_id)
        sink.install()
        _SINK = sink
        return sink


def uninstall_bus_event_sink() -> None:
    """Remove any installed sink. No-op when nothing is installed."""

    global _SINK
    with _SINK_LOCK:
        if _SINK is None:
            return
        _SINK.uninstall()
        _SINK = None


def get_bus_event_sink() -> BusEventSink | None:
    with _SINK_LOCK:
        return _SINK
