"""S10 — chat-mode runtime: the toggle state machine + the batch tick loop.

Encapsulates the live chat-reply behavior so the orchestrator's only job is to (a)
construct this ONLY when ``twitch.enabled`` (so flags-OFF imports nothing -> main
runtime byte-identical), (b) route the Stream-Deck/voice toggle to
:meth:`enable`/:meth:`disable`, and (c) call :meth:`tick` from the idle loop.

Key invariants:
  * **Guard REQUIRED to enable.** :meth:`enable` consults ``chat_mode_can_enable``
    (guard wired + healthy + canary passes); if not, chat-reply stays OFF
    (fail-CLOSED on the feature, never on the relay).
  * **Buffer-then-batch.** While OFF, chat accumulates in the read sidecar (this
    object does nothing). While ON, each :meth:`tick` drains the buffer and runs
    ONE safety-gated :class:`ChatReplyPipeline` batch.
  * **Gray-zone review.** Flagged inbound messages are handed to ``on_flagged``
    (the 2nd-monitor popup / voice review loop) -- never spoken.
  * **No relay handle.** This object speaks only through the pipeline's stream-bus
    speak_fn (provenance TWITCH_CHAT); it cannot reach the team mic.

The live addressing / selection / 8B-reply / read-buffer drain are injected
callables (bound by the orchestrator to the EmbeddingGemma client / the leaf
modules / generate_stream / the read-sidecar client), so this is fully
offline-testable. FAIL-CLOSED: any tick error degrades to silence, never a crash.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable, Optional, Sequence

from kenning.twitch.guard import chat_mode_can_enable
from kenning.twitch.pipeline import BatchResult, ChatReplyPipeline, FlaggedMessage

logger = logging.getLogger("kenning.twitch.runtime")

__all__ = ["ChatModeState", "ChatModeRuntime"]


class ChatModeState(str, Enum):
    OFF = "off"            # buffering only; speaks nothing
    READY = "ready"        # chat-reply active
    LOCKDOWN = "lockdown"  # safety fault -> dropped the batch; auto-recovers next tick


class ChatModeRuntime:
    def __init__(
        self,
        *,
        pipeline: ChatReplyPipeline,
        drain_fn: Callable[[], Sequence[Any]],
        is_reply_target: Callable[[Any], bool],
        select_fn: Callable[[list], list],
        reply_fn: Callable[[list], str],
        guard_client: Optional[Any] = None,
        guard_required: bool = True,
        on_flagged: Optional[Callable[[FlaggedMessage], None]] = None,
        can_enable_fn: Callable[..., tuple] = chat_mode_can_enable,
    ) -> None:
        self._pipeline = pipeline
        self._drain = drain_fn
        self._is_reply_target = is_reply_target
        self._select = select_fn
        self._reply = reply_fn
        self._guard = guard_client
        self._guard_required = bool(guard_required)
        self._on_flagged = on_flagged
        self._can_enable = can_enable_fn
        self._state = ChatModeState.OFF

    @property
    def state(self) -> ChatModeState:
        return self._state

    @property
    def active(self) -> bool:
        return self._state in (ChatModeState.READY, ChatModeState.LOCKDOWN)

    def enable(self) -> tuple[bool, str]:
        """Turn chat-reply ON iff the guard precondition holds. Returns (ok, why)."""
        try:
            ok, why = self._can_enable(self._guard, guard_required=self._guard_required)
        except Exception as e:  # noqa: BLE001 — fail-CLOSED on the feature
            self._state = ChatModeState.OFF
            return False, f"enable check failed: {e}"
        if not ok:
            self._state = ChatModeState.OFF
            logger.warning("chat-reply NOT enabled: %s", why)
            return False, why
        self._state = ChatModeState.READY
        logger.info("chat-reply ENABLED (%s)", why)
        return True, why

    def disable(self) -> None:
        if self._state != ChatModeState.OFF:
            logger.info("chat-reply disabled")
        self._state = ChatModeState.OFF

    def tick(self) -> Optional[BatchResult]:
        """Drain one batch and process it. Returns the BatchResult, or None when
        OFF / nothing buffered. Never raises (fail-CLOSED to silence)."""
        if not self.active:
            return None
        try:
            events = list(self._drain() or [])
        except Exception as e:  # noqa: BLE001 — buffer drain error -> skip this tick
            logger.warning("chat buffer drain failed: %s", e)
            self._state = ChatModeState.LOCKDOWN
            return None
        if not events:
            self._state = ChatModeState.READY
            return None
        try:
            result = self._pipeline.process_batch(
                events,
                is_reply_target=self._is_reply_target,
                select_fn=self._select,
                reply_fn=self._reply,
            )
        except Exception as e:  # noqa: BLE001 — the pipeline is itself fail-closed,
            # but a binding error must not crash the idle loop either.
            logger.warning("chat batch processing failed: %s", e)
            self._state = ChatModeState.LOCKDOWN
            return None
        # surface flagged (gray-zone) inbound to the review loop; never spoken.
        if self._on_flagged is not None:
            for f in result.flagged:
                try:
                    self._on_flagged(f)
                except Exception as e:  # noqa: BLE001
                    logger.debug("on_flagged callback error: %s", e)
        self._state = ChatModeState.READY
        return result
