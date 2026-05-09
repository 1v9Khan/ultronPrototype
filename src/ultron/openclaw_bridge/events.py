"""Inbound event receiver (Phase 3.4 — gated-off scaffold).

For most inbound events (e.g. a Telegram message from the user)
OpenClaw handles the agent turn itself using the shared persona
files. Ultron's voice path is not involved.

There is one exception: when an inbound message starts with a
configured prefix (default ``[voice]``), OpenClaw should forward it
to Ultron's orchestrator so the response is spoken aloud through the
voice pipeline (assumes someone is at the desk to hear it). This is
the "voice handoff" pattern.

Phase 3 ships only the scaffolding:

- :class:`OpenClawEventReceiver` with ``start()`` / ``stop()`` and a
  prefix-matching helper that the orchestrator can plug into.
- The receiver is **disabled by default**. ``start()`` is a no-op
  unless ``inbound_voice_handoff_enabled`` is True in config.
- No webhook subscription or polling loop yet — the actual transport
  is wired in a later phase once a real channel (Telegram) lands.

The pure-logic helpers (:meth:`should_handle`, :meth:`extract_payload`)
are unit-tested today so the prefix-matching contract is locked down
before any transport machinery rides on top of it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from ultron.utils.logging import get_logger

logger = get_logger("openclaw_bridge.events")


# Type alias for the orchestrator-side voice handoff callback. Receives
# the message body with the prefix stripped, plus the channel/sender
# metadata so the orchestrator can decide whether to speak the reply.
VoiceHandoffHandler = Callable[["IncomingMessage"], Awaitable[None]]


@dataclass(frozen=True)
class IncomingMessage:
    """Subset of an inbound message we care about for routing.

    Phase 3 keeps this minimal — only the fields the voice-handoff
    decision needs. Later phases can extend with attachments, reply
    threading, etc., without breaking the receiver's public surface."""

    channel: str                                    # e.g. "telegram"
    sender: str                                     # provider-specific id (telegram chat id, etc.)
    body: str                                       # full message text
    prefix_match: bool = False                      # convenience: did this trigger a handoff?


class OpenClawEventReceiver:
    """Receives inbound events from OpenClaw and routes them.

    Args:
        prefix: prefix that triggers voice handoff (default ``[voice]``).
            Compared case-sensitively to the message body.
        on_voice_handoff: async callback invoked when an incoming
            message matches the prefix. Caller is the orchestrator;
            it's responsible for actually speaking the reply through
            the voice pipeline.
        enabled: master switch. When False, ``start()`` is a no-op
            and incoming events are silently dropped (the rest of
            the bridge keeps working — OpenClaw still handles inbound
            messages on its own side via its agent turns).
    """

    def __init__(
        self,
        *,
        prefix: str = "[voice]",
        on_voice_handoff: Optional[VoiceHandoffHandler] = None,
        enabled: bool = False,
    ) -> None:
        if not prefix:
            raise ValueError("prefix must be non-empty")
        self._prefix = prefix
        self._on_voice_handoff = on_voice_handoff
        self._enabled = bool(enabled)
        self._started = False
        self._stop_event: Optional[asyncio.Event] = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def started(self) -> bool:
        return self._started

    @property
    def prefix(self) -> str:
        return self._prefix

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    async def start(self) -> None:
        """Begin receiving events. No-op when ``enabled=False``.

        Phase 3 doesn't subscribe to any transport yet; this method
        only flips the started flag so :meth:`stop` is symmetric.
        Later phases will plug in webhook subscription or a polling
        loop here.
        """
        if not self._enabled:
            logger.debug(
                "voice handoff disabled (inbound_voice_handoff_enabled=false)",
            )
            return
        if self._started:
            return
        self._stop_event = asyncio.Event()
        self._started = True
        logger.info(
            "voice-handoff receiver started (prefix=%r, transport=stub)",
            self._prefix,
        )

    async def stop(self) -> None:
        """Stop receiving events. Safe to call when not started."""
        if self._stop_event is not None:
            self._stop_event.set()
        self._started = False
        # Drop the handler reference so a re-init doesn't leak the old
        # closure into a fresh receiver.
        self._stop_event = None

    # -------------------------------------------------------------------
    # Pure-logic helpers (unit-tested in Phase 3.6)
    # -------------------------------------------------------------------

    def should_handle(self, body: str) -> bool:
        """Return True iff ``body`` starts with the configured prefix.

        Whitespace at the very start is tolerated — ``  [voice] foo``
        matches the same as ``[voice] foo``. Case-sensitive otherwise.
        """
        if not isinstance(body, str):
            return False
        return body.lstrip().startswith(self._prefix)

    def extract_payload(self, body: str) -> str:
        """Return the message body with the prefix and any leading
        whitespace stripped. If the prefix doesn't match, return the
        body unchanged."""
        if not self.should_handle(body):
            return body
        stripped = body.lstrip()
        return stripped[len(self._prefix):].lstrip()

    # -------------------------------------------------------------------
    # Dispatch
    # -------------------------------------------------------------------

    async def dispatch(self, message: IncomingMessage) -> bool:
        """Dispatch one incoming message. Returns True iff the message
        triggered a voice handoff and the handler ran without raising.

        Currently called only by tests; later phases will hook the
        actual transport up to this entrypoint. Fail-open: handler
        exceptions are logged at WARN and never propagate to the
        orchestrator's main loop."""
        if not self._enabled or not self._started:
            return False
        if not self.should_handle(message.body):
            return False
        if self._on_voice_handoff is None:
            logger.debug(
                "voice-handoff prefix matched but no handler is wired",
            )
            return False
        payload = self.extract_payload(message.body)
        wrapped = IncomingMessage(
            channel=message.channel,
            sender=message.sender,
            body=payload,
            prefix_match=True,
        )
        try:
            await self._on_voice_handoff(wrapped)
            return True
        except Exception as exc:
            logger.warning(
                "voice-handoff handler raised (%s) — dropping event",
                exc,
            )
            return False


__all__ = [
    "IncomingMessage",
    "OpenClawEventReceiver",
    "VoiceHandoffHandler",
]
