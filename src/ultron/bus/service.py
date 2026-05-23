"""Bus service: pub/sub registry + dispatch.

Ported from opencode's ``packages/opencode/src/bus/index.ts`` with the
Effect / PubSub primitives swapped for ``threading.RLock`` + dict-of-
lists. The key invariant we preserve from opencode is **eager
subscription acquisition** -- a subscribe call returns AFTER the
callback is registered in the dispatch table, so a publish that
races with subscribe is delivered (or not delivered if subscribe
hasn't run yet) but never lost-to-an-acquired-but-not-yet-listening
stream.

Threading model:
  * Callbacks fire on the publishing thread, synchronously.
  * Subscribers can subscribe / unsubscribe from any thread.
  * Bus state guarded by an ``RLock`` so a callback can subscribe
    or publish recursively without deadlock.

Fail-open posture:
  * Callback exceptions are caught and logged at WARNING. One bad
    subscriber never breaks others.
  * Schema validation failures are logged at WARNING but the event
    is still delivered (matches opencode's
    ``Effect.tryPromise(...).pipe(Effect.ignore)``).

Slow-subscriber watchdog (2026-05-22):
  * Every callback's wall-clock duration is measured. Subscribers that
    take longer than ``DEFAULT_SLOW_SUBSCRIBER_WARN_MS`` (default
    15 ms) emit a WARN log and bump :meth:`Bus.slow_subscriber_count`.
    Because dispatch is synchronous on the publishing thread, a slow
    callback blocks every later subscriber on that publish AND every
    later publish until it returns. The watchdog surfaces those before
    they wedge the voice loop. Subscribers needing async work must
    hand the payload off to their own queue.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Optional

from ultron.bus.event import BusEvent, EventPayload

logger = logging.getLogger("ultron.bus")

# Callback type alias. Callbacks receive the full :class:`EventPayload`
# (envelope), not just the properties dict, so subscribers can read the
# event id / type when needed.
EventCallback = Callable[[EventPayload], None]

# Wildcard channel constant. Subscribing here receives every published
# event. Matches opencode's ``subscribeAll``.
_WILDCARD = "*"

# Default threshold for the slow-subscriber watchdog. Callbacks taking
# longer than this on the synchronous dispatch path block every later
# subscriber AND every later publish, so we surface them with a WARN
# log. Tuned to comfortably exceed typical callback work (a few hundred
# microseconds for log writes, dict updates, lightweight bus
# republishes) but catch any subscriber that does blocking I/O,
# heavy compute, or hits a deadlock.
DEFAULT_SLOW_SUBSCRIBER_WARN_MS: float = 15.0

# Optional cross-module fail-open counter hook. Set by
# ``ultron.resilience.fail_open_log`` when wired in the orchestrator
# so the bus's own slow-subscriber events get tallied alongside other
# fail-open paths for the startup summary. The bus does NOT import
# the resilience module (avoiding a circular dependency); the
# resilience module installs itself.
_SLOW_SUBSCRIBER_RECORDER: Optional[Callable[[str, str], None]] = None


def set_slow_subscriber_recorder(
    recorder: Optional[Callable[[str, str], None]],
) -> None:
    """Install a callback called whenever a subscriber exceeds the threshold.

    The recorder receives ``(category, reason)`` where category is
    ``"bus_slow_subscriber"`` and reason names the event type. Used by
    :mod:`ultron.resilience.fail_open_log` to aggregate fail-open
    events across subsystems. Pass ``None`` to clear the hook.
    """
    global _SLOW_SUBSCRIBER_RECORDER
    _SLOW_SUBSCRIBER_RECORDER = recorder


class Bus:
    """In-process pub/sub registry with type-channel + wildcard support.

    Singleton in production (via :func:`get_bus`); a fresh instance
    can be constructed in tests + injected via
    :func:`reset_bus_for_testing`.
    """

    def __init__(
        self,
        *,
        slow_subscriber_warn_ms: float = DEFAULT_SLOW_SUBSCRIBER_WARN_MS,
    ) -> None:
        """Construct a fresh bus.

        Args:
            slow_subscriber_warn_ms: WARN threshold (ms) for the
                slow-subscriber watchdog. Callbacks taking longer
                than this trigger a log + counter bump. Set to a
                very large value to effectively disable.
        """
        # Subscriber registry: channel name -> list of (token, callback).
        # The token (int) is the unsubscribe key; using a token rather
        # than callback-identity lets the same function subscribe
        # multiple times if desired.
        self._subscribers: Dict[str, List[tuple[int, EventCallback]]] = (
            defaultdict(list)
        )
        self._next_token: int = 0
        self._lock = threading.RLock()
        # Publish counter for diagnostics + tests.
        self._published_count: int = 0
        # Slow-subscriber watchdog state.
        self._slow_subscriber_warn_ms: float = float(slow_subscriber_warn_ms)
        self._slow_subscriber_count: int = 0

    # --- public API ---------------------------------------------------------

    def publish(
        self,
        event_def: BusEvent,
        properties: Mapping[str, Any],
        id: Optional[str] = None,
    ) -> EventPayload:
        """Fan out an event to all matching subscribers.

        Args:
            event_def: the :class:`BusEvent` returned by
                :meth:`BusEvent.define`.
            properties: payload fields. Validated best-effort against
                ``event_def.schema``; mismatches logged but delivered.
            id: optional explicit id. Auto-generated when omitted.

        Returns:
            The :class:`EventPayload` envelope that was dispatched.
            Mostly useful for tests that need to inspect the
            generated id.
        """
        problem = event_def.validate(properties)
        if problem is not None:
            logger.warning(
                "bus: schema mismatch on %s: %s (delivering anyway)",
                event_def.type, problem,
            )
        payload = EventPayload.make(event_def, properties, id=id)

        with self._lock:
            self._published_count += 1
            # Snapshot the subscriber lists under lock so callbacks
            # that subscribe / unsubscribe during dispatch don't
            # mutate the iteration target.
            typed = list(self._subscribers.get(event_def.type, ()))
            wildcard = list(self._subscribers.get(_WILDCARD, ()))

        warn_threshold_ms = self._slow_subscriber_warn_ms
        for token, cb in typed + wildcard:
            t0 = time.perf_counter()
            raised = False
            try:
                cb(payload)
            except Exception as e:                                  # noqa: BLE001
                raised = True
                logger.warning(
                    "bus: subscriber %d on %r raised %s (swallowed)",
                    token, event_def.type, e,
                )
            if raised:
                # An exception path already short-circuits whatever
                # work the subscriber intended; don't count its
                # elapsed time against the slow-subscriber budget.
                continue
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if elapsed_ms > warn_threshold_ms:
                with self._lock:
                    self._slow_subscriber_count += 1
                logger.warning(
                    "bus: subscriber %d on %r took %.1f ms "
                    "(>%.1f ms threshold; sync dispatch blocks the "
                    "publishing thread -- hand off to a queue if slow)",
                    token, event_def.type, elapsed_ms, warn_threshold_ms,
                )
                recorder = _SLOW_SUBSCRIBER_RECORDER
                if recorder is not None:
                    try:
                        recorder("bus_slow_subscriber", event_def.type)
                    except Exception:                              # noqa: BLE001
                        pass

        return payload

    def subscribe(
        self,
        event_def: BusEvent,
        callback: EventCallback,
    ) -> Callable[[], None]:
        """Register ``callback`` to fire on every publish of ``event_def``.

        Returns an ``unsubscribe`` callable. Calling it more than once
        is a no-op. Calling it from inside a callback is safe (the
        next publish will see the updated subscriber list).
        """
        return self._add_subscriber(event_def.type, callback)

    def subscribe_all(self, callback: EventCallback) -> Callable[[], None]:
        """Register ``callback`` to fire on EVERY published event.

        Mirrors opencode's ``subscribeAll`` -- useful for tracing /
        logging that needs to see the full event stream without
        binding to specific types.
        """
        return self._add_subscriber(_WILDCARD, callback)

    # --- introspection ------------------------------------------------------

    def subscriber_count(self, event_def: Optional[BusEvent] = None) -> int:
        """Count of currently-registered subscribers.

        With ``event_def=None``, returns the total across all
        channels including wildcard. With a specific definition,
        returns just that channel.
        """
        with self._lock:
            if event_def is None:
                return sum(
                    len(subs) for subs in self._subscribers.values()
                )
            return len(self._subscribers.get(event_def.type, ()))

    def published_count(self) -> int:
        """Total publishes since construction. For tests / diagnostics."""
        with self._lock:
            return self._published_count

    def slow_subscriber_count(self) -> int:
        """Number of subscriber callbacks that exceeded the WARN threshold.

        Counts each occurrence, not each unique subscriber. Aggregated
        across all event types and all subscribers since bus
        construction. Use to spot subscribers that need to hand off
        to a queue.
        """
        with self._lock:
            return self._slow_subscriber_count

    def slow_subscriber_warn_ms(self) -> float:
        """Current WARN threshold (ms) for the slow-subscriber watchdog."""
        return self._slow_subscriber_warn_ms

    # --- internals ----------------------------------------------------------

    def _add_subscriber(
        self, channel: str, callback: EventCallback,
    ) -> Callable[[], None]:
        with self._lock:
            self._next_token += 1
            token = self._next_token
            self._subscribers[channel].append((token, callback))

        unsubscribed = [False]

        def unsubscribe() -> None:
            if unsubscribed[0]:
                return
            unsubscribed[0] = True
            with self._lock:
                subs = self._subscribers.get(channel)
                if not subs:
                    return
                self._subscribers[channel] = [
                    (t, c) for (t, c) in subs if t != token
                ]
                if not self._subscribers[channel]:
                    # Clean up empty channels so subscriber_count()
                    # stays accurate after subscribers go away.
                    del self._subscribers[channel]

        return unsubscribe


# ---------------------------------------------------------------------------
# Module-level singleton + shortcuts
# ---------------------------------------------------------------------------


_BUS_LOCK = threading.RLock()
_BUS_SINGLETON: Optional[Bus] = None


def get_bus() -> Bus:
    """Return the process-wide singleton bus, constructing on first call."""
    global _BUS_SINGLETON
    with _BUS_LOCK:
        if _BUS_SINGLETON is None:
            _BUS_SINGLETON = Bus()
        return _BUS_SINGLETON


def reset_bus_for_testing() -> Bus:
    """Replace the singleton with a fresh instance.

    Test-only escape hatch. Production code never calls this.
    """
    global _BUS_SINGLETON
    with _BUS_LOCK:
        _BUS_SINGLETON = Bus()
        return _BUS_SINGLETON


def publish(
    event_def: BusEvent,
    properties: Mapping[str, Any],
    id: Optional[str] = None,
) -> EventPayload:
    """Module-level publish -- shortcut for ``get_bus().publish(...)``."""
    return get_bus().publish(event_def, properties, id=id)


def subscribe(
    event_def: BusEvent,
    callback: EventCallback,
) -> Callable[[], None]:
    """Module-level subscribe -- shortcut for ``get_bus().subscribe(...)``."""
    return get_bus().subscribe(event_def, callback)


def subscribe_all(callback: EventCallback) -> Callable[[], None]:
    """Module-level subscribe_all -- shortcut for ``get_bus().subscribe_all(...)``."""
    return get_bus().subscribe_all(callback)
