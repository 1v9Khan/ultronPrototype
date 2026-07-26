"""Push-to-talk output backends.

Read the anticheat boundary in :mod:`kenning.ptt` before touching this file.

The host's ONLY job is to write a byte to a serial port. The protocol is three
one-byte commands the external microcontroller understands:

    b"D"  key DOWN  -- start holding the team-PTT key
    b"U"  key UP    -- release it
    b"H"  heartbeat -- "still holding" (refreshes the firmware deadman)

The firmware presses a real USB-HID key on ``D``, releases on ``U``, and -- as a
hardware failsafe -- auto-releases if it stops receiving bytes for its deadman
window (so a host crash mid-hold cannot jam the mic open).

Two transports, same protocol:
  * :class:`SerialHidPttBackend` -- the legacy device exposes a CDC serial (COM)
    port; commands are ``serial.write(byte)`` (pyserial).
  * :class:`RawHidPttBackend` -- the HARDENED device has NO COM port; it exposes
    a vendor Raw-HID collection, and commands are HID OUTPUT reports (hidapi).
    Writing an HID output report is DEVICE I/O -- sending bytes TO a peripheral --
    NOT synthetic input injection (no LLKHF_INJECTED, not the quarantined SendInput
    surface). Both pyserial and hidapi are benign leaf deps, on no block list.

Nothing here imports or calls any synthetic-input library.
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


# USB identity of the HARDENED HID-only device (no CDC/COM port): a custom
# open-hardware VID (pid.codes 0x1209) + the vendor Raw-HID collection used for
# the command channel (usage page in the 0xFF00-0xFFFF vendor range). The device
# presents as a plain keyboard + this vendor collection -- like any commercial
# gaming keyboard with a config interface.
HID_PTT_VID = 0x1209
HID_PTT_USAGE_PAGE = 0xFFC0


def find_hid_ptt_device(vid=HID_PTT_VID, usage_page=HID_PTT_USAGE_PAGE, pid=None):
    """Return the hidapi device path of the PTT vendor HID collection, or None.

    Matches on USB VID + the vendor usage page (the Raw-HID command channel), NOT
    the keyboard collection. Imports hidapi lazily so the package stays
    import-clean when PTT is off; returns None if hidapi is unavailable or nothing
    matches. Writing to this path is DEVICE I/O, not synthetic input."""
    try:
        import hid  # noqa: PLC0415
        devices = hid.enumerate(vid, pid or 0)
    except Exception:  # noqa: BLE001 - fail-safe: no detection
        return None
    for d in devices:
        if d.get("usage_page") == usage_page:
            return d.get("path")
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


class RawHidPttBackend(PttBackend):
    """Drives the HARDENED HID-only device via hidapi HID OUTPUT reports (no COM
    port). Writing an HID output report is DEVICE I/O -- bytes sent TO a
    peripheral -- NOT synthetic input: it sets no LLKHF_INJECTED flag and is not
    the quarantined SendInput surface. Same fail-safe contract as the serial
    backend: any open/write error disables the backend; it never raises and never
    falls back to in-process input."""

    def __init__(
        self,
        vid: int = HID_PTT_VID,
        usage_page: int = HID_PTT_USAGE_PAGE,
        *,
        pid: Optional[int] = None,
        down: bytes = CMD_DOWN,
        up: bytes = CMD_UP,
        heartbeat: bytes = CMD_HEARTBEAT,
        open_hid: Optional[Callable[..., object]] = None,
    ) -> None:
        self._down, self._up, self._hb = down, up, heartbeat
        self._lock = threading.Lock()
        self._dev = None
        self._ok = False
        try:
            if open_hid is not None:
                # Test seam: inject a fake hid device without hidapi.
                self._dev = open_hid(vid, pid, usage_page)
            else:
                # Lazy import: hidapi (module name ``hid``) -- device I/O, NOT an
                # input-injection lib, not on the anticheat block list, only
                # pulled in when PTT is actually armed.
                import hid  # noqa: PLC0415

                path = None
                for d in hid.enumerate(vid, pid or 0):
                    if d.get("usage_page") == usage_page:
                        path = d.get("path")
                        break
                if path is None:
                    raise OSError(
                        "no vendor HID collection (vid=%#06x usage_page=%#06x)"
                        % (vid, usage_page)
                    )
                self._dev = hid.device()
                self._dev.open_path(path)
            self._ok = self._dev is not None
        except Exception as e:  # noqa: BLE001 - fail-safe: unavailable, never raise
            logger.warning(
                "push-to-talk HID open failed (vid=%#06x): %s -- PTT will not fire",
                vid, e,
            )
            self._dev = None
            self._ok = False

    @property
    def available(self) -> bool:  # type: ignore[override]
        return self._ok

    def press(self) -> None:
        self._report(self._down)

    def release(self) -> None:
        self._report(self._up)

    def heartbeat(self) -> None:
        self._report(self._hb)

    def _report(self, cmd: bytes) -> None:
        # report id 0 + a 64-byte report; byte 0 carries the command.
        packet = bytes([0x00, cmd[0]]) + b"\x00" * 63
        with self._lock:
            if not self._ok or self._dev is None:
                return
            try:
                self._dev.write(packet)
            except Exception as e:  # noqa: BLE001 - fail-safe: disable, don't raise
                logger.warning(
                    "push-to-talk HID write failed (%s) -- disabling PTT", e,
                )
                self._ok = False

    def close(self) -> None:
        with self._lock:
            dev = self._dev
            self._dev = None
            self._ok = False
        if dev is not None:
            try:
                dev.write(bytes([0x00, self._up[0]]) + b"\x00" * 63)  # release
            except Exception:  # noqa: BLE001
                pass
            try:
                dev.close()
            except Exception:  # noqa: BLE001
                pass


class NetworkPttBackend(PttBackend):
    """Forwards the one-byte protocol to a PTT agent on ANOTHER PC over UDP.

    For the two-PC layout: Ultron on the AI/stream box, Valorant + the USB-HID
    device on the game box (relay audio already crosses via VBAN). The agent
    (``scripts/ptt_agent.py``) holds the real :class:`RawHidPttBackend`, so the
    anticheat boundary is unchanged -- this side only writes an authenticated
    datagram to a socket, and the peripheral still generates the keypress.

    Loss tolerance comes from the firmware, not from the transport: a dropped
    ``D`` is recovered by the next heartbeat and a dropped ``U`` by the 200 ms
    deadman, so a partition mid-hold releases the mic instead of jamming it
    open. See :mod:`kenning.ptt.netproto`.

    Same fail-safe contract as the local backends: any socket error disables
    the backend, and nothing here ever raises into the caller or falls back to
    synthetic input.
    """

    def __init__(
        self,
        host: str,
        port: int,
        token: str,
        *,
        probe_timeout: float = 0.5,
        probe: bool = True,
        open_socket: Optional[Callable[[], object]] = None,
    ) -> None:
        self._addr = (host, int(port))
        self._token = token
        self._lock = threading.Lock()
        self._counter = 0
        self._sock = None
        self._ok = False
        if not token:
            logger.warning(
                "push-to-talk network backend has no token -- refusing to arm "
                "(set push_to_talk.network_token or KENNING_PTT_NETWORK_TOKEN)"
            )
            return
        try:
            if open_socket is not None:
                # Test seam: inject a fake socket without touching the network.
                self._sock = open_socket()
            else:
                import socket  # noqa: PLC0415

                self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._sock.settimeout(probe_timeout)
            self._ok = self._sock is not None
        except Exception as e:  # noqa: BLE001 - fail-safe: unavailable, never raise
            logger.warning(
                "push-to-talk network socket open failed (%s:%d): %s -- PTT will "
                "not fire", self._addr[0], self._addr[1], e,
            )
            self._sock = None
            self._ok = False
            return
        # Confirm the agent is actually reachable. Without this a typo'd host
        # looks "enabled" at boot and silently never keys the mic -- the exact
        # failure mode a remote hop makes invisible.
        if probe and not self._probe():
            self._ok = False

    def _probe(self) -> bool:
        """Round-trip a PING. True only on a valid, authenticated PONG."""
        from kenning.ptt import netproto  # noqa: PLC0415

        try:
            sent = self._send_frame(netproto.CMD_PING)
            if not sent:
                return False
            reply = self._sock.recv(netproto.FRAME_LEN)  # type: ignore[union-attr]
            _, cmd = netproto.parse_frame(reply, self._token)
            if cmd != netproto.CMD_PONG:
                raise netproto.ProtocolError(f"unexpected reply {cmd!r}")
        except Exception as e:  # noqa: BLE001 - unreachable agent is not fatal
            logger.warning(
                "push-to-talk agent did not answer at %s:%d (%s) -- PTT INERT "
                "until it is running (start scripts/ptt_agent.py on the game PC)",
                self._addr[0], self._addr[1], e,
            )
            return False
        logger.info(
            "push-to-talk: remote agent reachable at %s:%d (relay will key the "
            "game PC's PTT device)", self._addr[0], self._addr[1],
        )
        return True

    @property
    def available(self) -> bool:  # type: ignore[override]
        return self._ok

    def press(self) -> None:
        from kenning.ptt import netproto  # noqa: PLC0415

        self._send_frame(netproto.CMD_DOWN)

    def release(self) -> None:
        from kenning.ptt import netproto  # noqa: PLC0415

        self._send_frame(netproto.CMD_UP)

    def heartbeat(self) -> None:
        from kenning.ptt import netproto  # noqa: PLC0415

        self._send_frame(netproto.CMD_HEARTBEAT)

    def _next_counter(self) -> int:
        """Strictly-increasing counter (replay guard on the agent side).

        Seeded from the wall clock so it keeps climbing across restarts of this
        process -- a counter that reset to 0 would be rejected as a replay by an
        agent that had already seen higher values.
        """
        import time  # noqa: PLC0415

        nxt = max(time.time_ns(), self._counter + 1)
        self._counter = nxt
        return nxt

    def _send_frame(self, cmd: bytes) -> bool:
        from kenning.ptt import netproto  # noqa: PLC0415

        with self._lock:
            if not self._ok or self._sock is None:
                return False
            try:
                frame = netproto.build_frame(self._next_counter(), cmd, self._token)
                self._sock.sendto(frame, self._addr)  # type: ignore[union-attr]
                return True
            except Exception as e:  # noqa: BLE001 - fail-safe: disable, don't raise
                logger.warning(
                    "push-to-talk network send failed (%s) -- disabling PTT", e,
                )
                self._ok = False
                return False

    def close(self) -> None:
        # Best-effort release BEFORE tearing the socket down, mirroring the HID
        # backend: never leave the far-side mic held open on shutdown. (The
        # firmware deadman would catch it in 200 ms regardless.)
        from kenning.ptt import netproto  # noqa: PLC0415

        self._send_frame(netproto.CMD_UP)
        with self._lock:
            sock = self._sock
            self._sock = None
            self._ok = False
        if sock is not None:
            try:
                sock.close()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
