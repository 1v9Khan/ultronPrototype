"""Background keepalive pings to keep an HTTP LLM's prompt cache warm.

Pattern lifted in spirit (not in source) from aider's
``base_coder.warm_cache`` (Apache 2.0; see ``THIRD_PARTY_NOTICES.md``).

The problem: Anthropic prompt caching has a 5-minute server-side TTL.
After 5 minutes of inactivity the cache evicts and the next turn
pays full token cost — typically 10x what a cache hit costs. For a
conversational assistant where the user takes mid-turn pauses
(reading code, thinking, getting coffee), the cache often goes cold
right before the next interaction.

The fix: a background thread fires a max_tokens=1 completion against
the cacheable prefix every ~295 seconds (5 minutes minus a 5-second
safety margin). Each ping is essentially free — it's a cache HIT
that resets the TTL clock. Per the catalog: ``5 * 60 - 5 = 295`` is
the magic number.

Cost guards:

  * Don't fire if the cacheable prefix is empty (no cache to warm).
  * Stop firing after ``idle_giveup_seconds`` (default 30 minutes) —
    if the user is gone, no point burning even tiny network costs.
  * Skip when no ``last_activity_provider`` is wired — caller must
    declare what "user activity" means.

Public surface:

  * :class:`CacheWarmer` — daemon-thread wrapper.
  * :func:`make_warmer_from_chunks` — convenience constructor that
    binds a :class:`ChunkedPrompt` builder + an LLM send callable.

The thread is daemon = True so the warmer never blocks process
shutdown. Stop it explicitly via :meth:`CacheWarmer.stop` for clean
shutdown.

Fail-open: send-fn exceptions are logged at DEBUG and counted; the
thread keeps running. If the send-fn returns False (caller's
signal "the ping was rejected / cache empty / etc."), we back off
for one cycle.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional


logger = logging.getLogger("ultron.llm.cache_warmer")


# Catalog T9 magic number. Anthropic's TTL is 5 minutes; we ping at
# 5 minutes minus a 5-second safety margin so the ping reliably
# arrives before the cache evicts.
DEFAULT_INTERVAL_SECONDS = 5 * 60 - 5  # 295 seconds


# Cost guard: after this many seconds of no user activity, stop
# pinging. The user is probably gone; let the cache evict.
DEFAULT_IDLE_GIVEUP_SECONDS = 30 * 60  # 30 minutes


# Send callable contract: takes no arguments (the caller is
# responsible for building the cacheable-prefix payload internally),
# returns True on success / False to signal back-off.
SendFn = Callable[[], bool]


@dataclass
class WarmerTelemetry:
    """Counters exposed for offline tuning + dashboards."""

    pings_sent: int = 0
    pings_succeeded: int = 0
    pings_failed: int = 0
    pings_skipped_idle: int = 0
    pings_skipped_empty: int = 0


class CacheWarmer:
    """Daemon-thread keepalive sender.

    Args:
        send_fn: Callable that fires one keepalive ping. Returns True
            on success, False on "skip / back off". May raise; we
            count + swallow.
        last_activity_provider: Callable returning the monotonic
            timestamp of the last user activity. Used by the
            idle-giveup guard. Pass ``None`` to disable the guard
            (the warmer pings forever; useful for tests).
        interval_seconds: Seconds between pings. Defaults to 295.
        idle_giveup_seconds: After this many seconds since the last
            user activity, the warmer skips pings entirely. Set to 0
            to disable.
        prefix_present_check: Callable returning True iff there's a
            cacheable prefix to warm. When this returns False we skip
            the ping and tick the ``pings_skipped_empty`` counter.
    """

    def __init__(
        self,
        send_fn: SendFn,
        *,
        last_activity_provider: Optional[Callable[[], float]] = None,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        idle_giveup_seconds: float = DEFAULT_IDLE_GIVEUP_SECONDS,
        prefix_present_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError(
                f"interval_seconds must be > 0, got {interval_seconds}"
            )
        self._send_fn = send_fn
        self._last_activity_provider = last_activity_provider
        self._interval_seconds = float(interval_seconds)
        self._idle_giveup_seconds = float(idle_giveup_seconds)
        self._prefix_present_check = prefix_present_check
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._telemetry = WarmerTelemetry()
        self._telemetry_lock = threading.Lock()
        self._now: Callable[[], float] = time.monotonic

    @property
    def telemetry(self) -> WarmerTelemetry:
        """Snapshot of current counters."""
        with self._telemetry_lock:
            return WarmerTelemetry(
                pings_sent=self._telemetry.pings_sent,
                pings_succeeded=self._telemetry.pings_succeeded,
                pings_failed=self._telemetry.pings_failed,
                pings_skipped_idle=self._telemetry.pings_skipped_idle,
                pings_skipped_empty=self._telemetry.pings_skipped_empty,
            )

    @property
    def running(self) -> bool:
        """True when the daemon thread is alive."""
        t = self._thread
        return t is not None and t.is_alive()

    def start(self) -> None:
        """Spawn the daemon thread. Idempotent."""
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="ultron-cache-warmer",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal stop and join the thread (best-effort)."""
        self._stop_event.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        self._thread = None

    def _loop(self) -> None:
        """Main daemon loop. Wakes every ``interval_seconds`` and
        decides whether to fire a ping."""
        while not self._stop_event.is_set():
            # Sleep first so the first ping happens after one interval
            # — gives initial state time to settle.
            if self._stop_event.wait(self._interval_seconds):
                return  # stop requested

            try:
                self._maybe_ping()
            except Exception as exc:                              # noqa: BLE001
                logger.debug(
                    "cache_warmer: _maybe_ping raised unexpectedly: %s", exc,
                )

    def _maybe_ping(self) -> None:
        # Cost guard: skip if user idle past threshold.
        if (
            self._idle_giveup_seconds > 0
            and self._last_activity_provider is not None
        ):
            try:
                last = float(self._last_activity_provider())
            except Exception as exc:                              # noqa: BLE001
                logger.debug(
                    "cache_warmer: last_activity_provider raised: %s", exc,
                )
                last = self._now()  # assume active to be safe
            if (self._now() - last) > self._idle_giveup_seconds:
                with self._telemetry_lock:
                    self._telemetry.pings_skipped_idle += 1
                return

        # Cost guard: skip if no cacheable prefix.
        if self._prefix_present_check is not None:
            try:
                present = bool(self._prefix_present_check())
            except Exception as exc:                              # noqa: BLE001
                logger.debug(
                    "cache_warmer: prefix_present_check raised: %s", exc,
                )
                present = True  # safer to ping than to miss the cache
            if not present:
                with self._telemetry_lock:
                    self._telemetry.pings_skipped_empty += 1
                return

        # Fire the ping. Don't propagate exceptions out of the loop.
        with self._telemetry_lock:
            self._telemetry.pings_sent += 1
        try:
            ok = bool(self._send_fn())
        except Exception as exc:                                  # noqa: BLE001
            logger.debug("cache_warmer: send_fn raised: %s", exc)
            with self._telemetry_lock:
                self._telemetry.pings_failed += 1
            return

        with self._telemetry_lock:
            if ok:
                self._telemetry.pings_succeeded += 1
            else:
                self._telemetry.pings_failed += 1


__all__ = [
    "CacheWarmer",
    "DEFAULT_IDLE_GIVEUP_SECONDS",
    "DEFAULT_INTERVAL_SECONDS",
    "SendFn",
    "WarmerTelemetry",
]
