"""Push-to-talk lifecycle controller + factory.

Drives a :class:`~kenning.ptt.backends.PttBackend` deterministically off the
relay playback lifecycle -- NOT off VAD. ``hold()`` is called right before a
relay clip plays to the team mic; ``release()`` right after it drains (the
reverb tail is already baked into the clip buffer). The key is held continuously
across back-to-back callouts and released after a short configurable tail so the
game's transmit codec doesn't clip the end.

Robustness:
  * A background driver thread emits keep-alive heartbeats while the key is held
    (refreshing the firmware's hardware deadman) and performs the actual UP after
    the release tail -- so ``release()`` returns immediately and never blocks the
    relay handler.
  * A host-side max-hold watchdog force-releases if a single hold ever runs too
    long (a second line of defense above the hardware deadman).
  * Every backend call is fail-safe: an error is logged and swallowed, never
    propagated into the relay path, and never escalated to in-process input.
  * When the backend is unavailable (disabled / no device), ``hold``/``release``
    are no-ops that add ZERO latency and never start a thread.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from kenning.ptt.backends import (
    NullPttBackend,
    PttBackend,
    SerialHidPttBackend,
    find_arduino_port,
)

logger = logging.getLogger(__name__)

_IDLE = 0
_HOLDING = 1
_RELEASING = 2


class PttController:
    def __init__(
        self,
        backend: PttBackend,
        *,
        heartbeat_ms: int = 50,
        release_tail_ms: int = 150,
        lead_ms: int = 120,
        max_hold_seconds: float = 8.0,
    ) -> None:
        self._backend = backend
        self._hb = max(0.005, heartbeat_ms / 1000.0)
        self._tail = max(0.0, release_tail_ms / 1000.0)
        self._lead = max(0.0, lead_ms / 1000.0)
        self._max_hold = max(0.1, float(max_hold_seconds))
        self._cv = threading.Condition()
        self._state = _IDLE
        self._hold_start = 0.0
        self._release_at = 0.0
        self._stop = False
        self._driver: Optional[threading.Thread] = None

    @property
    def available(self) -> bool:
        return bool(getattr(self._backend, "available", False))

    # -- public lifecycle --------------------------------------------------

    def hold(self) -> None:
        """Begin (or continue) holding the team-PTT key. Blocks ``lead_ms`` so
        the game's transmit channel is open before the first audio sample.
        No-op and ZERO delay when the backend is unavailable -- PTT off must
        never add latency to the relay path."""
        if not self.available:
            return
        try:
            with self._cv:
                if self._state == _IDLE:
                    self._safe_call(self._backend.press)
                self._hold_start = time.monotonic()
                self._state = _HOLDING
                self._ensure_driver_locked()
                self._cv.notify_all()
            if self._lead:
                time.sleep(self._lead)
        except Exception as e:  # noqa: BLE001 - never break the relay
            logger.debug("ptt hold failed: %s", e)

    def release(self) -> None:
        """Schedule release of the team-PTT key after ``release_tail_ms``. Returns
        immediately; the driver thread emits the UP once the tail elapses (so the
        clip's baked reverb tail + the game codec tail aren't clipped). A new
        ``hold()`` before the tail expires cancels the release and keeps the key
        down. No-op when unavailable."""
        if not self.available:
            return
        try:
            with self._cv:
                if self._state in (_HOLDING, _RELEASING):
                    self._state = _RELEASING
                    self._release_at = time.monotonic() + self._tail
                    self._cv.notify_all()
        except Exception as e:  # noqa: BLE001
            logger.debug("ptt release failed: %s", e)

    def close(self) -> None:
        """Stop the driver and release the key. Safe to call repeatedly."""
        try:
            with self._cv:
                self._stop = True
                self._state = _IDLE
                self._cv.notify_all()
            d = self._driver
            if d is not None:
                d.join(timeout=1.0)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._backend.close()
        except Exception:  # noqa: BLE001
            pass

    # -- internals ---------------------------------------------------------

    def _ensure_driver_locked(self) -> None:
        if self._driver is None or not self._driver.is_alive():
            self._stop = False
            self._driver = threading.Thread(
                target=self._run, daemon=True, name="ptt-driver",
            )
            self._driver.start()

    def _safe_call(self, fn) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - fail-safe: never raise into caller
            logger.debug("ptt backend call failed: %s", e)

    def _run(self) -> None:
        # All serial writes happen UNDER self._cv so hold()/release() can never
        # race a heartbeat/release write (the backend has its own lock too).
        while True:
            with self._cv:
                if self._stop:
                    return
                if self._state == _IDLE:
                    self._cv.wait(timeout=1.0)
                    continue
                now = time.monotonic()
                if self._state == _HOLDING:
                    if now - self._hold_start > self._max_hold:
                        self._safe_call(self._backend.release)
                        logger.warning(
                            "ptt: forced key release (max-hold watchdog "
                            "%.1fs exceeded)", self._max_hold,
                        )
                        self._state = _IDLE
                        continue
                    self._safe_call(self._backend.heartbeat)
                elif self._state == _RELEASING:
                    if now >= self._release_at:
                        self._safe_call(self._backend.release)
                        self._state = _IDLE
                        continue
                    self._safe_call(self._backend.heartbeat)
                self._cv.wait(timeout=self._hb)


def build_ptt_controller(config=None, *, enabled=None, serial_port=None) -> PttController:
    """Construct a :class:`PttController` from config. Always returns a controller
    (never None). Fail-safe selection of the backend:

      * PTT disabled, or no Arduino found, or the device can't be opened
        ->  :class:`NullPttBackend` (completely inert).
      * PTT enabled AND the serial device opens  ->  :class:`SerialHidPttBackend`.

    ``enabled`` / ``serial_port`` override the config values when given -- the
    orchestrator passes the env-resolved ``settings.PUSH_TO_TALK_*`` so the
    ``KENNING_PTT_*`` env vars take precedence over ``config.yaml`` (matching the
    settings.py override convention). There is NO path to an in-process input
    backend -- if the hardware isn't ready, PTT stays off.
    """
    if config is None:
        from kenning.config import get_config

        config = get_config()
    ptt = getattr(config, "push_to_talk", None)
    if enabled is None:
        enabled = bool(getattr(ptt, "enabled", False)) if ptt else False
    if serial_port is None:
        serial_port = (getattr(ptt, "serial_port", "") if ptt else "")

    def _mk(backend: PttBackend) -> PttController:
        return PttController(
            backend,
            heartbeat_ms=int(getattr(ptt, "heartbeat_ms", 50)) if ptt else 50,
            release_tail_ms=int(getattr(ptt, "release_tail_ms", 150)) if ptt else 150,
            lead_ms=int(getattr(ptt, "lead_ms", 120)) if ptt else 120,
            max_hold_seconds=float(getattr(ptt, "max_hold_seconds", 8.0)) if ptt else 8.0,
        )

    if not enabled:
        return _mk(NullPttBackend())

    port = (str(serial_port or "")).strip()
    if port.lower() == "auto" or not port:
        detected = find_arduino_port()
        if detected:
            logger.info("push-to-talk: auto-detected Arduino on %s", detected)
            port = detected
        else:
            logger.warning(
                "push-to-talk ENABLED but no Arduino auto-detected -- PTT INERT "
                "(no synthetic-input fallback). Plug it in or set "
                "push_to_talk.serial_port to a COM port.",
            )
            return _mk(NullPttBackend())

    baud = int(getattr(ptt, "baud", 9600))
    backend = SerialHidPttBackend(port, baud)
    if not backend.available:
        logger.warning(
            "push-to-talk ENABLED but serial device %r unavailable -- PTT INERT "
            "(no synthetic-input fallback).", port,
        )
        try:
            backend.close()
        except Exception:  # noqa: BLE001
            pass
        return _mk(NullPttBackend())

    logger.info(
        "push-to-talk ARMED via external USB-HID on %s @ %d baud "
        "(host writes serial bytes ONLY -- no synthetic input)", port, baud,
    )
    return _mk(backend)
