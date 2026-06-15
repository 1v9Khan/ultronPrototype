"""Push-to-talk output backends.

Read the anticheat boundary in :mod:`kenning.ptt` before touching this file.

The host's ONLY job is to write a byte to a serial port. The protocol is three
one-byte commands the external microcontroller understands:

    b"D"  key DOWN  -- start holding the team-PTT key
    b"U"  key UP    -- release it
    b"H"  heartbeat -- "still holding" (refreshes the firmware deadman)

The firmware presses a real USB-HID key on ``D``, releases on ``U``, and -- as a
hardware failsafe -- auto-releases if it stops receiving bytes for its deadman
window (so a host crash mid-hold cannot jam the mic open). Nothing here imports
or calls any synthetic-input library; pyserial is the only third-party import and
it is not on any anticheat block list.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# One-byte wire protocol (host -> microcontroller).
CMD_DOWN = b"D"
CMD_UP = b"U"
CMD_HEARTBEAT = b"H"

# USB identity of an Arduino Leonardo: VID = Arduino LLC, PID = the running
# sketch (the bootloader is 0x0036). Used for "auto" port detection so a
# drifting Windows COM assignment (the port can change across flashes/replugs)
# doesn't break PTT.
ARDUINO_VIDS = (0x2341,)
LEONARDO_SKETCH_PIDS = (0x8036,)


def find_arduino_port(vids=ARDUINO_VIDS, pids=LEONARDO_SKETCH_PIDS):
    """Return the COM port of a connected Arduino by USB VID/PID, or None.

    Prefers an exact VID+PID match (Leonardo sketch = 2341:8036), then falls
    back to any port with a matching VID. Imports pyserial lazily so the package
    stays import-clean when PTT is off; returns None if pyserial is unavailable
    or nothing matches (caller then stays inert -- never a fallback to input)."""
    try:
        from serial.tools import list_ports  # noqa: PLC0415
        ports = list(list_ports.comports())
    except Exception:  # noqa: BLE001 - fail-safe: no detection
        return None
    for p in ports:
        if p.vid in vids and (not pids or p.pid in pids):
            return p.device
    for p in ports:
        if p.vid in vids:
            return p.device
    return None


class PttBackend:
    """Interface for a push-to-talk key asserter. ``available`` is False unless a
    real device is wired up; the controller treats an unavailable backend as a
    complete no-op (fail-safe). Subclasses must never perform synthetic input."""

    available: bool = False

    def press(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def release(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def heartbeat(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass


class NullPttBackend(PttBackend):
    """The default backend: does nothing. Used whenever PTT is disabled or no
    serial device is present. Guarantees zero anticheat surface and zero latency
    -- the controller short-circuits on ``available`` being False."""

    available = False

    def press(self) -> None:
        logger.debug("ptt(null): press -- no device, no-op")

    def release(self) -> None:
        logger.debug("ptt(null): release -- no device, no-op")

    def heartbeat(self) -> None:
        pass

    def close(self) -> None:
        pass


class SerialHidPttBackend(PttBackend):
    """Writes the one-byte protocol to an external USB-HID microcontroller over a
    serial (COM) port. THE ONLY thing this does is ``serial.write(byte)`` -- the
    keypress happens in hardware. Fail-safe: any open/write error marks the
    backend unavailable (PTT stops firing); it NEVER raises into the caller and
    NEVER falls back to in-process input."""

    def __init__(
        self,
        port: str,
        baud: int = 9600,
        *,
        down: bytes = CMD_DOWN,
        up: bytes = CMD_UP,
        heartbeat: bytes = CMD_HEARTBEAT,
        open_serial: Optional[Callable[[str, int], object]] = None,
    ) -> None:
        self._port = port
        self._down, self._up, self._hb = down, up, heartbeat
        self._lock = threading.Lock()
        self._ser = None
        self._ok = False
        try:
            if open_serial is not None:
                # Test seam: inject a fake serial object without pyserial.
                self._ser = open_serial(port, baud)
            else:
                # Lazy import: pyserial (module name ``serial``) is a benign
                # leaf dep, NOT on the anticheat block list, and is only pulled
                # in when PTT is actually armed with a configured port.
                import serial  # noqa: PLC0415

                self._ser = serial.Serial(
                    port, baud, timeout=0, write_timeout=0.05,
                )
            self._ok = self._ser is not None
        except Exception as e:  # noqa: BLE001 - fail-safe: unavailable, never raise
            logger.warning(
                "push-to-talk serial open failed on %r (%s) -- PTT will not fire",
                port, e,
            )
            self._ser = None
            self._ok = False

    @property
    def available(self) -> bool:  # type: ignore[override]
        return self._ok

    def press(self) -> None:
        self._write(self._down)

    def release(self) -> None:
        self._write(self._up)

    def heartbeat(self) -> None:
        self._write(self._hb)

    def _write(self, b: bytes) -> None:
        with self._lock:
            if not self._ok or self._ser is None:
                return
            try:
                self._ser.write(b)
            except Exception as e:  # noqa: BLE001 - fail-safe: disable, don't raise
                logger.warning(
                    "push-to-talk serial write failed (%s) -- disabling PTT", e,
                )
                self._ok = False

    def close(self) -> None:
        with self._lock:
            ser = self._ser
            self._ser = None
            self._ok = False
        if ser is not None:
            try:
                ser.write(self._up)  # best-effort release before closing
            except Exception:  # noqa: BLE001
                pass
            try:
                ser.close()
            except Exception:  # noqa: BLE001
                pass
