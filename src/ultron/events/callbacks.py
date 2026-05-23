"""Event callback registry -- decouple reactive behaviour from the orchestrator.

Pattern lineage attributed in ``THIRD_PARTY_NOTICES.md``.

The OpenHands ``EventCallbackService`` is a SQL-backed CRUD over polymorphic
``EventCallbackProcessor`` rows; every event arrival fires
``execute_callbacks(conversation_id, event)`` which fans out to every
matching processor via ``asyncio.gather`` and persists each result. Ultron's
voice path is single-process + synchronous, so the port:

* Keeps the polymorphic processor ABC (``CallbackProcessor.__call__`` ->
  optional :class:`CallbackResult`).
* Uses an in-memory registry (with optional JSONL persistence for restart
  durability) keyed by callback id, session-id filter, event-kind filter,
  and active/disabled status.
* Fires callbacks AFTER :meth:`EventStore.save_event` returns -- a callback
  exception never loses the underlying event.

Self-deactivating callbacks (T3 creative-extension): when a processor sets
``result.deactivate=True`` the registry flips its status to DISABLED so a
future event of the same kind won't re-fire it. The catalog's "after 5 turns
of compliance" pattern is one motivating case; the title-generation example
(set once, then never again) is another.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

from ultron.events.models import StoredEvent

logger = logging.getLogger(__name__)


class CallbackStatus(str, Enum):
    """Lifecycle state of a registered callback."""

    ACTIVE = "active"
    DISABLED = "disabled"


class CallbackResultStatus(str, Enum):
    """Outcome of a single callback execution."""

    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"
    DEACTIVATED = "deactivated"


@dataclass(frozen=True)
class CallbackResult:
    """Outcome of a single callback invocation."""

    status: CallbackResultStatus
    callback_id: str
    event_id: str
    session_id: str
    detail: str | None = None
    deactivate: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class CallbackProcessor(ABC):
    """Polymorphic side-effect handler.

    Concrete subclasses implement ``__call__`` and return a
    :class:`CallbackResult` (or ``None`` to signal "I didn't apply").

    A processor that sets ``CallbackResult.deactivate=True`` requests
    the registry flip the owning :class:`RegisteredCallback` to
    :attr:`CallbackStatus.DISABLED` so it won't fire again. Use this
    for one-shot patterns ("set the title on the first message and
    then never re-fire").
    """

    @abstractmethod
    def __call__(
        self,
        event: StoredEvent,
        callback: "RegisteredCallback",
    ) -> CallbackResult | None:
        ...

    # Optional override for diagnostic display.
    @property
    def label(self) -> str:
        return self.__class__.__name__


@dataclass(frozen=True)
class RegisteredCallback:
    """One entry in the registry."""

    id: str
    processor: CallbackProcessor
    session_id: str | None
    event_kind: str | None
    status: CallbackStatus
    created_at: float
    label: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status == CallbackStatus.ACTIVE

    def matches(self, event: StoredEvent) -> bool:
        if self.session_id is not None and self.session_id != event.session_id:
            return False
        if self.event_kind is not None and self.event_kind != event.kind:
            return False
        return True


def _new_callback_id() -> str:
    return uuid.uuid4().hex


class CallbackRegistry:
    """Thread-safe in-memory registry with optional JSONL persistence.

    The registry owns three concurrent surfaces:

    * Catalog management: ``register`` / ``unregister`` / ``set_status`` /
      ``list_callbacks`` (CRUD).
    * Dispatch: ``execute_for_event`` runs every matching active callback
      and returns the per-callback :class:`CallbackResult`.
    * Auto-persistence: when ``persistence_path`` is set, the catalog is
      flushed to JSONL after every mutation so a process restart can
      restore the active callbacks via ``load_callbacks_from_disk``.

    Slow-callback watchdog: each invocation is timed; exceeding
    ``slow_callback_warn_ms`` emits a WARN log. Default 50 ms.
    """

    def __init__(
        self,
        *,
        persistence_path: Path | str | None = None,
        slow_callback_warn_ms: float = 50.0,
        max_dispatched: int = 10_000,
    ) -> None:
        self._lock = threading.RLock()
        self._callbacks: dict[str, RegisteredCallback] = {}
        self._persistence_path = Path(persistence_path) if persistence_path else None
        self._slow_warn_ms = slow_callback_warn_ms
        self._dispatched = 0
        self._errors = 0
        self._max_dispatched = max_dispatched

    # -- introspection --

    @property
    def dispatched(self) -> int:
        with self._lock:
            return self._dispatched

    @property
    def errors(self) -> int:
        with self._lock:
            return self._errors

    @property
    def persistence_path(self) -> Path | None:
        return self._persistence_path

    def list_callbacks(
        self,
        *,
        session_id: str | None = None,
        event_kind: str | None = None,
        include_disabled: bool = False,
    ) -> list[RegisteredCallback]:
        """Return registered callbacks matching the optional filters."""

        with self._lock:
            result: list[RegisteredCallback] = []
            for cb in self._callbacks.values():
                if not include_disabled and cb.status != CallbackStatus.ACTIVE:
                    continue
                if session_id is not None and cb.session_id != session_id:
                    continue
                if event_kind is not None and cb.event_kind != event_kind:
                    continue
                result.append(cb)
            return sorted(result, key=lambda c: c.created_at)

    def get(self, callback_id: str) -> RegisteredCallback | None:
        with self._lock:
            return self._callbacks.get(callback_id)

    # -- CRUD --

    def register(
        self,
        processor: CallbackProcessor,
        *,
        session_id: str | None = None,
        event_kind: str | None = None,
        callback_id: str | None = None,
        label: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> RegisteredCallback:
        """Register a new callback and return the stored row."""

        if not isinstance(processor, CallbackProcessor):
            raise TypeError(
                "processor must be a CallbackProcessor instance; got "
                f"{type(processor).__name__}"
            )
        with self._lock:
            cid = callback_id or _new_callback_id()
            if cid in self._callbacks:
                raise ValueError(f"callback id {cid!r} already exists")
            row = RegisteredCallback(
                id=cid,
                processor=processor,
                session_id=session_id,
                event_kind=event_kind,
                status=CallbackStatus.ACTIVE,
                created_at=time.time(),
                label=label or processor.label,
                extra=dict(extra or {}),
            )
            self._callbacks[cid] = row
            self._maybe_persist_locked()
            return row

    def unregister(self, callback_id: str) -> bool:
        with self._lock:
            existed = self._callbacks.pop(callback_id, None) is not None
            if existed:
                self._maybe_persist_locked()
            return existed

    def set_status(self, callback_id: str, status: CallbackStatus) -> bool:
        with self._lock:
            existing = self._callbacks.get(callback_id)
            if existing is None:
                return False
            updated = replace(existing, status=status)
            self._callbacks[callback_id] = updated
            self._maybe_persist_locked()
            return True

    def clear(self) -> None:
        """Drop every registration. Persistence (if any) is left untouched on disk."""

        with self._lock:
            self._callbacks.clear()
            self._dispatched = 0
            self._errors = 0

    # -- dispatch --

    def execute_for_event(self, event: StoredEvent) -> list[CallbackResult]:
        """Fire every active matching callback. Returns one result per fire."""

        with self._lock:
            candidates = [
                cb
                for cb in self._callbacks.values()
                if cb.is_active and cb.matches(event)
            ]
            candidates.sort(key=lambda c: c.created_at)

        results: list[CallbackResult] = []
        for callback in candidates:
            start = time.perf_counter()
            try:
                outcome = callback.processor(event, callback)
            except Exception as exc:                            # noqa: BLE001
                outcome = CallbackResult(
                    status=CallbackResultStatus.ERROR,
                    callback_id=callback.id,
                    event_id=event.id,
                    session_id=event.session_id,
                    detail=f"{type(exc).__name__}: {exc}",
                )
                with self._lock:
                    self._errors += 1
                logger.warning(
                    "callback %s raised %r on event %s",
                    callback.label,
                    exc,
                    event.kind,
                )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if outcome is None:
                # Processor opted out; record SKIPPED.
                outcome = CallbackResult(
                    status=CallbackResultStatus.SKIPPED,
                    callback_id=callback.id,
                    event_id=event.id,
                    session_id=event.session_id,
                )

            if elapsed_ms > self._slow_warn_ms:
                logger.warning(
                    "callback %s took %.1f ms on event %s (slow > %.0f ms)",
                    callback.label,
                    elapsed_ms,
                    event.kind,
                    self._slow_warn_ms,
                )

            results.append(outcome)
            with self._lock:
                self._dispatched += 1
                if self._dispatched > self._max_dispatched:
                    self._dispatched = self._max_dispatched
                # Honour self-deactivation requests.
                if outcome.deactivate and outcome.status != CallbackResultStatus.ERROR:
                    current = self._callbacks.get(callback.id)
                    if current is not None:
                        self._callbacks[callback.id] = replace(
                            current, status=CallbackStatus.DISABLED
                        )
                        self._maybe_persist_locked()
        return results

    # -- persistence --

    def _maybe_persist_locked(self) -> None:
        path = self._persistence_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "id": cb.id,
                    "session_id": cb.session_id,
                    "event_kind": cb.event_kind,
                    "status": cb.status.value,
                    "created_at": cb.created_at,
                    "label": cb.label,
                    "processor": cb.processor.__class__.__name__,
                    "extra": cb.extra,
                }
                for cb in self._callbacks.values()
            ]
            text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
            path.write_text(text + ("\n" if text else ""), encoding="utf-8")
        except OSError as exc:
            logger.warning("callback registry persist failed for %s: %s", path, exc)

    def load_metadata_from_disk(self) -> list[dict[str, Any]]:
        """Read the persisted metadata (callback ids + filters + label).

        Returns a list of dicts. Callers re-create the live processor
        instances themselves and call :meth:`register` with the desired
        ``callback_id`` for resume semantics. The registry intentionally
        does NOT auto-resurrect processors -- doing so would require
        each processor class to be globally identifiable + reconstructable,
        which the catalog flagged as a discriminated-union T23 follow-up.
        """

        path = self._persistence_path
        if path is None or not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("callback registry: skipping malformed row")
        except OSError as exc:
            logger.warning("callback registry load failed for %s: %s", path, exc)
        return rows


# -- module-level singleton --


_REGISTRY: CallbackRegistry | None = None
_REGISTRY_LOCK = threading.RLock()


def get_callback_registry() -> CallbackRegistry | None:
    with _REGISTRY_LOCK:
        return _REGISTRY


def set_callback_registry(registry: CallbackRegistry | None) -> None:
    global _REGISTRY
    with _REGISTRY_LOCK:
        _REGISTRY = registry


def reset_callback_registry_for_testing() -> None:
    set_callback_registry(None)


# -- adapter callable for use as a CallbackProcessor --


@dataclass(frozen=True)
class FunctionProcessor(CallbackProcessor):
    """Wrap a plain callable as a :class:`CallbackProcessor`.

    The callable receives ``(event, callback)`` and returns either:

    * ``None`` (treated as SKIPPED),
    * a :class:`CallbackResult` directly,
    * any truthy value (treated as SUCCESS with the value stringified
      into ``detail``).

    Convenient for one-off subscribers; production code should subclass
    :class:`CallbackProcessor` so the class name shows up in audit logs.
    """

    func: Callable[[StoredEvent, "RegisteredCallback"], Any]
    func_label: str | None = None

    @property
    def label(self) -> str:
        return self.func_label or getattr(self.func, "__name__", "function_callback")

    def __call__(
        self,
        event: StoredEvent,
        callback: "RegisteredCallback",
    ) -> CallbackResult | None:
        outcome = self.func(event, callback)
        if outcome is None:
            return None
        if isinstance(outcome, CallbackResult):
            return outcome
        return CallbackResult(
            status=CallbackResultStatus.SUCCESS,
            callback_id=callback.id,
            event_id=event.id,
            session_id=event.session_id,
            detail=str(outcome),
        )
