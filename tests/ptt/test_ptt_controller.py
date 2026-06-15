"""Auto push-to-talk controller + backend behavior.

Covers the inert default, the press/heartbeat/release state machine, the
release tail + continuous-hold coalescing, the max-hold watchdog, fail-safe
swallowing, and the serial backend's one-byte protocol + fail-safe disabling.
"""
import threading
import time

import pytest

from kenning.ptt.backends import (
    CMD_DOWN,
    CMD_HEARTBEAT,
    CMD_UP,
    NullPttBackend,
    PttBackend,
    RawHidPttBackend,
    SerialHidPttBackend,
)
from kenning.ptt.controller import PttController, build_ptt_controller


class _FakeBackend(PttBackend):
    available = True

    def __init__(self) -> None:
        self.events: list[str] = []
        self._lock = threading.Lock()
        self.fail_press = False

    def press(self) -> None:
        if self.fail_press:
            raise RuntimeError("boom")
        with self._lock:
            self.events.append("press")

    def release(self) -> None:
        with self._lock:
            self.events.append("release")

    def heartbeat(self) -> None:
        with self._lock:
            self.events.append("heartbeat")

    def close(self) -> None:
        with self._lock:
            self.events.append("close")

    def count(self, name: str) -> int:
        with self._lock:
            return self.events.count(name)


def _wait_for(pred, timeout=2.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


# -- inert default -----------------------------------------------------------

def test_null_backend_is_inert():
    c = PttController(NullPttBackend(), heartbeat_ms=10, release_tail_ms=10, lead_ms=0)
    assert c.available is False
    c.hold()
    c.release()
    c.close()
    # No driver thread is ever started for the inert backend.
    assert c._driver is None  # noqa: SLF001


# -- press / heartbeat / release ---------------------------------------------

def test_hold_presses_then_heartbeats_then_releases():
    b = _FakeBackend()
    c = PttController(b, heartbeat_ms=10, release_tail_ms=30, lead_ms=0, max_hold_seconds=5)
    c.hold()
    assert b.count("press") == 1
    assert _wait_for(lambda: b.count("heartbeat") >= 2)
    assert b.count("release") == 0           # still holding
    c.release()
    assert _wait_for(lambda: b.count("release") == 1)
    c.close()


def test_continuous_hold_cancels_release_and_does_not_re_press():
    b = _FakeBackend()
    c = PttController(b, heartbeat_ms=10, release_tail_ms=200, lead_ms=0, max_hold_seconds=5)
    c.hold()
    c.release()              # schedules UP in ~200ms
    time.sleep(0.03)
    c.hold()                 # within the tail -> cancels release, key stays down
    time.sleep(0.06)
    assert b.count("release") == 0           # release was cancelled
    assert b.count("press") == 1             # NOT re-pressed (key never went up)
    c.close()


def test_max_hold_watchdog_force_releases():
    b = _FakeBackend()
    # max_hold floored to 0.1s by the controller; watchdog must fire on its own.
    c = PttController(b, heartbeat_ms=10, release_tail_ms=10, lead_ms=0, max_hold_seconds=0.1)
    c.hold()
    assert _wait_for(lambda: b.count("release") >= 1)
    c.close()


def test_hold_is_fail_safe_when_backend_raises():
    b = _FakeBackend()
    b.fail_press = True
    c = PttController(b, heartbeat_ms=10, release_tail_ms=10, lead_ms=0)
    c.hold()        # must NOT raise even though press() throws
    c.release()
    c.close()


def test_close_releases_and_closes_backend():
    b = _FakeBackend()
    c = PttController(b, heartbeat_ms=10, release_tail_ms=10, lead_ms=0)
    c.hold()
    c.close()
    assert b.count("close") == 1


# -- serial backend protocol -------------------------------------------------

def test_serial_backend_writes_one_byte_protocol():
    writes: list[bytes] = []

    class _FakeSerial:
        def write(self, b):
            writes.append(b)

        def close(self):
            pass

    b = SerialHidPttBackend("COM_TEST", 9600, open_serial=lambda p, baud: _FakeSerial())
    assert b.available is True
    b.press()
    b.heartbeat()
    b.release()
    assert writes == [CMD_DOWN, CMD_HEARTBEAT, CMD_UP]
    b.close()
    # close() emits a best-effort UP before closing.
    assert writes[-1] == CMD_UP


def test_serial_backend_open_failure_is_unavailable_and_silent():
    def boom(port, baud):
        raise RuntimeError("no such port")

    b = SerialHidPttBackend("COMX", 9600, open_serial=boom)
    assert b.available is False
    b.press()       # no-op, must not raise
    b.heartbeat()
    b.close()


def test_serial_backend_write_error_disables_fail_safe():
    class _Flaky:
        def __init__(self):
            self.n = 0

        def write(self, b):
            self.n += 1
            if self.n > 1:
                raise RuntimeError("unplugged mid-match")

        def close(self):
            pass

    b = SerialHidPttBackend("COM_TEST", 9600, open_serial=lambda p, baud: _Flaky())
    assert b.available is True
    b.press()           # ok
    b.heartbeat()       # raises internally -> disables, swallowed
    assert b.available is False
    b.release()         # now a no-op


# -- factory selection (fail-safe; never an in-process input backend) --------

class _PttCfg:
    def __init__(self, **kw):
        self.enabled = kw.get("enabled", False)
        # default to the serial path so the legacy-path tests stay deterministic
        # regardless of any real HID device plugged into the test machine.
        self.backend = kw.get("backend", "serial")
        self.hid_vid = kw.get("hid_vid", 0x1209)
        self.hid_usage_page = kw.get("hid_usage_page", 0xFFC0)
        self.serial_port = kw.get("serial_port", "")
        self.baud = kw.get("baud", 9600)
        self.heartbeat_ms = kw.get("heartbeat_ms", 50)
        self.release_tail_ms = kw.get("release_tail_ms", 150)
        self.lead_ms = kw.get("lead_ms", 120)
        self.max_hold_seconds = kw.get("max_hold_seconds", 8.0)


class _Cfg:
    def __init__(self, ptt):
        self.push_to_talk = ptt


def test_factory_disabled_returns_inert():
    c = build_ptt_controller(_Cfg(_PttCfg(enabled=False)))
    assert c.available is False


def test_factory_explicit_bad_port_returns_inert():
    # An explicit, non-existent COM port must fail SAFE (inert) -- never fall
    # back to in-process input.
    c = build_ptt_controller(_Cfg(_PttCfg(enabled=True, serial_port="COM_NOPE_ZZZ")))
    assert c.available is False


def test_factory_auto_with_no_device_returns_inert(monkeypatch):
    import kenning.ptt.controller as ctrl
    monkeypatch.setattr(ctrl, "find_arduino_port", lambda *a, **k: None)
    c = build_ptt_controller(_Cfg(_PttCfg(enabled=True, serial_port="auto")))
    assert c.available is False


# -- auto-detect by USB VID/PID ----------------------------------------------

class _FakePort:
    def __init__(self, device, vid, pid):
        self.device, self.vid, self.pid = device, vid, pid


def test_find_arduino_prefers_exact_vid_pid(monkeypatch):
    import serial.tools.list_ports as lp
    from kenning.ptt import backends
    monkeypatch.setattr(lp, "comports", lambda: [
        _FakePort("COM1", 0x1234, 0x0001),
        _FakePort("COM7", 0x2341, 0x8036),   # Leonardo sketch
    ])
    assert backends.find_arduino_port() == "COM7"


def test_find_arduino_falls_back_to_vid(monkeypatch):
    import serial.tools.list_ports as lp
    from kenning.ptt import backends
    # Arduino VID but bootloader PID (0x0036) -> no exact match, VID fallback.
    monkeypatch.setattr(lp, "comports", lambda: [_FakePort("COM9", 0x2341, 0x0036)])
    assert backends.find_arduino_port() == "COM9"


def test_find_arduino_none_when_absent(monkeypatch):
    import serial.tools.list_ports as lp
    from kenning.ptt import backends
    monkeypatch.setattr(lp, "comports", lambda: [_FakePort("COM1", 0x1234, 0x0001)])
    assert backends.find_arduino_port() is None


# -- raw-HID backend (the hardened HID-only device) --------------------------

def test_rawhid_backend_writes_one_byte_reports():
    writes: list[bytes] = []

    class _FakeHid:
        def write(self, data):
            writes.append(bytes(data))
            return len(data)

        def close(self):
            pass

    b = RawHidPttBackend(0x1209, 0xFFC0, open_hid=lambda v, p, u: _FakeHid())
    assert b.available is True
    b.press()
    b.heartbeat()
    b.release()
    # Each report is report-id 0 + a 64-byte report; byte[1] carries the command.
    assert [w[1] for w in writes] == [ord("D"), ord("H"), ord("U")]
    assert all(len(w) == 65 for w in writes)
    b.close()
    assert writes[-1][1] == ord("U")    # close emits a best-effort release


def test_rawhid_backend_open_failure_is_inert():
    b = RawHidPttBackend(
        0x1209, 0xFFC0,
        open_hid=lambda v, p, u: (_ for _ in ()).throw(OSError("no device")),
    )
    assert b.available is False
    b.press()
    b.heartbeat()
    b.close()       # must not raise


def test_rawhid_backend_write_error_disables_fail_safe():
    class _Flaky:
        def __init__(self):
            self.n = 0

        def write(self, d):
            self.n += 1
            if self.n > 1:
                raise OSError("unplugged mid-match")
            return len(d)

        def close(self):
            pass

    b = RawHidPttBackend(0x1209, 0xFFC0, open_hid=lambda v, p, u: _Flaky())
    assert b.available is True
    b.press()           # ok
    b.heartbeat()       # raises internally -> disables, swallowed
    assert b.available is False
    b.release()         # now a no-op


# -- backend selection in the factory ----------------------------------------

def test_factory_auto_prefers_rawhid(monkeypatch):
    import kenning.ptt.controller as ctrl

    class _Hid:
        available = True

        def close(self):
            pass

    monkeypatch.setattr(ctrl, "RawHidPttBackend", lambda *a, **k: _Hid())
    c = build_ptt_controller(_Cfg(_PttCfg(enabled=True, backend="auto")))
    assert c.available is True


def test_factory_auto_falls_back_to_serial_then_inert(monkeypatch):
    import kenning.ptt.controller as ctrl

    class _NoHid:
        available = False

        def close(self):
            pass

    monkeypatch.setattr(ctrl, "RawHidPttBackend", lambda *a, **k: _NoHid())
    monkeypatch.setattr(ctrl, "find_arduino_port", lambda *a, **k: None)
    c = build_ptt_controller(_Cfg(_PttCfg(enabled=True, backend="auto", serial_port="auto")))
    assert c.available is False     # no HID, no serial -> inert


def test_factory_serial_kind_never_tries_rawhid(monkeypatch):
    import kenning.ptt.controller as ctrl
    called = {"hid": False}

    class _Hid:
        available = True

        def close(self):
            pass

    def _mk(*a, **k):
        called["hid"] = True
        return _Hid()

    monkeypatch.setattr(ctrl, "RawHidPttBackend", _mk)
    monkeypatch.setattr(ctrl, "find_arduino_port", lambda *a, **k: None)
    c = build_ptt_controller(_Cfg(_PttCfg(enabled=True, backend="serial")))
    assert called["hid"] is False
    assert c.available is False


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
