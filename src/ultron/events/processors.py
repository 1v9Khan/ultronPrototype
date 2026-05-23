"""Built-in :class:`CallbackProcessor` library.

Six processors ship in this batch; the registry can hold any number of
custom ones alongside these. Pattern lineage attributed in
``THIRD_PARTY_NOTICES.md``.

* :class:`LoggingCallbackProcessor` -- log every matched event at INFO.
* :class:`CountingCallbackProcessor` -- increment an in-memory counter
  per event kind (introspection helper).
* :class:`ThresholdSnapshotProcessor` -- generic "after N matching
  events, fire a snapshot callable then deactivate". The catalog's
  one-shot title-generator pattern is one instance of this; the
  cumulative-coding-diff-after-5-writes pattern is another.
* :class:`MemoryWriteProcessor` -- mirror event payloads into a caller-
  supplied memory writer (the orchestrator wires this to
  :class:`ConversationMemory` when desired).
* :class:`ChannelGuardProcessor` -- block emit of events whose
  payloads carry secret-like strings (defence layer on top of the
  safety validator).
* :class:`SkillActivatorProcessor` -- per the catalog's "conditional
  skill activation as a callback" extension, registers a transient
  keyword skill for N future turns when a topic-change event fires.

The :func:`build_default_processors` factory wires the safe-to-default
set onto a registry.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Callable

from ultron.events.callbacks import (
    CallbackProcessor,
    CallbackResult,
    CallbackResultStatus,
    RegisteredCallback,
)
from ultron.events.models import StoredEvent

logger = logging.getLogger(__name__)


# -- LoggingCallbackProcessor --


@dataclass
class LoggingCallbackProcessor(CallbackProcessor):
    """Log every matched event at INFO (or another configured level).

    Useful as a sanity probe ("is the bus sink actually firing?") and
    as a debug aid for new event-kind plumbing.
    """

    log_level: int = logging.INFO
    include_payload: bool = False

    def __call__(
        self,
        event: StoredEvent,
        callback: RegisteredCallback,
    ) -> CallbackResult:
        detail = f"kind={event.kind} session={event.session_id} id={event.id}"
        if self.include_payload:
            detail += f" payload={event.payload!r}"
        logger.log(self.log_level, "callback fire: %s", detail)
        return CallbackResult(
            status=CallbackResultStatus.SUCCESS,
            callback_id=callback.id,
            event_id=event.id,
            session_id=event.session_id,
            detail=detail,
        )


# -- CountingCallbackProcessor --


class CountingCallbackProcessor(CallbackProcessor):
    """Per-kind counter, useful in tests + diagnostics."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._counts: dict[str, int] = {}

    @property
    def counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def __call__(
        self,
        event: StoredEvent,
        callback: RegisteredCallback,
    ) -> CallbackResult:
        with self._lock:
            self._counts[event.kind] = self._counts.get(event.kind, 0) + 1
            count = self._counts[event.kind]
        return CallbackResult(
            status=CallbackResultStatus.SUCCESS,
            callback_id=callback.id,
            event_id=event.id,
            session_id=event.session_id,
            detail=f"count={count}",
            extra={"count": count},
        )


# -- ThresholdSnapshotProcessor --


@dataclass
class ThresholdSnapshotProcessor(CallbackProcessor):
    """Fire a snapshot callable once the matched-event count hits a threshold.

    The processor counts matching events per-session and -- when the
    count reaches :attr:`threshold` -- invokes :attr:`snapshot_fn`.
    With :attr:`deactivate_after_fire` True (default), the processor
    asks the registry to flip itself DISABLED so future events of the
    same kind don't keep firing the snapshot.

    Pattern matches the OpenHands ``SetTitleCallbackProcessor`` shape:
    "do an expensive thing once per conversation, then never again".
    """

    snapshot_fn: Callable[[StoredEvent, RegisteredCallback], str | None]
    threshold: int = 1
    deactivate_after_fire: bool = True
    label_override: str | None = None
    _counts_lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _counts: dict[str, int] = field(default_factory=dict, init=False)

    @property
    def label(self) -> str:
        return self.label_override or "ThresholdSnapshotProcessor"

    def __call__(
        self,
        event: StoredEvent,
        callback: RegisteredCallback,
    ) -> CallbackResult | None:
        with self._counts_lock:
            self._counts[event.session_id] = self._counts.get(event.session_id, 0) + 1
            count = self._counts[event.session_id]
        if count < self.threshold:
            return CallbackResult(
                status=CallbackResultStatus.SKIPPED,
                callback_id=callback.id,
                event_id=event.id,
                session_id=event.session_id,
                detail=f"below threshold ({count}/{self.threshold})",
                extra={"count": count, "threshold": self.threshold},
            )
        detail = self.snapshot_fn(event, callback)
        return CallbackResult(
            status=CallbackResultStatus.SUCCESS,
            callback_id=callback.id,
            event_id=event.id,
            session_id=event.session_id,
            detail=detail,
            deactivate=self.deactivate_after_fire,
        )


# -- MemoryWriteProcessor --


@dataclass
class MemoryWriteProcessor(CallbackProcessor):
    """Mirror event payloads into a caller-supplied memory writer.

    The writer is any callable accepting ``(role, content)`` -- this is
    the signature :class:`ConversationMemory.add` exposes, so the
    orchestrator can pass ``memory.add`` directly.

    The processor maps the event ``kind`` to a role using
    :attr:`role_for_kind` (default: "system" for everything not in the
    catch-all map). Override per-deployment if the producer kinds
    don't map cleanly.
    """

    writer: Callable[[str, str], None]
    role_for_kind: dict[str, str] = field(default_factory=dict)
    default_role: str = "system"
    content_field: str = "text"

    def __call__(
        self,
        event: StoredEvent,
        callback: RegisteredCallback,
    ) -> CallbackResult | None:
        content = event.payload.get(self.content_field)
        if not isinstance(content, str) or not content.strip():
            return CallbackResult(
                status=CallbackResultStatus.SKIPPED,
                callback_id=callback.id,
                event_id=event.id,
                session_id=event.session_id,
                detail="no string content to write",
            )
        role = self.role_for_kind.get(event.kind, self.default_role)
        try:
            self.writer(role, content)
        except Exception as exc:                                # noqa: BLE001
            return CallbackResult(
                status=CallbackResultStatus.ERROR,
                callback_id=callback.id,
                event_id=event.id,
                session_id=event.session_id,
                detail=f"writer raised: {type(exc).__name__}: {exc}",
            )
        return CallbackResult(
            status=CallbackResultStatus.SUCCESS,
            callback_id=callback.id,
            event_id=event.id,
            session_id=event.session_id,
            detail=f"wrote {len(content)} chars as role={role}",
        )


# -- ChannelGuardProcessor --


_SECRET_PATTERN = re.compile(
    r"(?P<token>(?:sk|pk|gho|github_pat|hf|api|secret|access|password)"
    r"[-_a-z0-9]*)\s*[:=][\s'\"\\]*[A-Za-z0-9_\-]{16,}",
    re.IGNORECASE,
)


@dataclass
class ChannelGuardProcessor(CallbackProcessor):
    """Scan event payloads for secret-looking substrings + redact.

    When a match is found, the processor returns ``ERROR`` with a
    redacted detail so the audit trail shows what happened without
    persisting the secret value. Use as a defence-in-depth on top of
    the safety validator -- the safety validator gates tool calls;
    this processor gates EVENT PAYLOADS.

    The processor doesn't mutate the underlying :class:`StoredEvent`
    (frozen). Operators wire a separate `before_persist` hook to
    actually redact the payload before storage.
    """

    redact_patterns: tuple[re.Pattern[str], ...] = field(
        default_factory=lambda: (_SECRET_PATTERN,)
    )

    def __call__(
        self,
        event: StoredEvent,
        callback: RegisteredCallback,
    ) -> CallbackResult | None:
        flat = _flatten_payload(event.payload)
        for pattern in self.redact_patterns:
            if pattern.search(flat):
                return CallbackResult(
                    status=CallbackResultStatus.ERROR,
                    callback_id=callback.id,
                    event_id=event.id,
                    session_id=event.session_id,
                    detail="secret-like substring detected in event payload",
                    extra={"matched_pattern": pattern.pattern},
                )
        return None


def _flatten_payload(payload: dict) -> str:
    """Best-effort flatten for substring scanning. Defensive against shape drift.

    Concatenates string-able values with whitespace so the regex can
    match across them without being thwarted by JSON-encoded backslash
    escaping of inner quotes.
    """

    parts: list[str] = []

    def _walk(value):
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (int, float, bool)) or value is None:
            parts.append(str(value))
        elif isinstance(value, dict):
            for k, v in value.items():
                parts.append(str(k))
                _walk(v)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                _walk(item)
        else:
            parts.append(repr(value))

    try:
        _walk(payload)
    except Exception:
        return repr(payload)
    return " ".join(parts)


# -- SkillActivatorProcessor --


@dataclass
class SkillActivatorProcessor(CallbackProcessor):
    """Trigger a transient skill in the global registry on matching events.

    When an event matching the processor's filter fires, call
    :attr:`activator_fn` so the consumer can register a skill (or set
    a "next N turns" flag). The processor stays ACTIVE -- callers can
    pair it with :class:`ThresholdSnapshotProcessor` for one-shot
    behaviour.
    """

    activator_fn: Callable[[StoredEvent, RegisteredCallback], str | None]

    def __call__(
        self,
        event: StoredEvent,
        callback: RegisteredCallback,
    ) -> CallbackResult | None:
        try:
            detail = self.activator_fn(event, callback)
        except Exception as exc:                                # noqa: BLE001
            return CallbackResult(
                status=CallbackResultStatus.ERROR,
                callback_id=callback.id,
                event_id=event.id,
                session_id=event.session_id,
                detail=f"activator raised: {type(exc).__name__}: {exc}",
            )
        return CallbackResult(
            status=CallbackResultStatus.SUCCESS,
            callback_id=callback.id,
            event_id=event.id,
            session_id=event.session_id,
            detail=detail,
        )


# -- factory --


def build_default_processors() -> list[CallbackProcessor]:
    """Return a sample list of safe-to-default processors.

    Operators wire these into a registry on opt-in. The default set is
    deliberately conservative -- only logging + counting are installed
    by default in the orchestrator wiring; the other processors require
    explicit configuration.
    """

    return [
        LoggingCallbackProcessor(log_level=logging.DEBUG),
        CountingCallbackProcessor(),
    ]
