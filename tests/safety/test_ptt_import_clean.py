"""Anticheat proof for auto-PTT: the package imports NO synthetic-input or
screen-capture library, and pyserial stays lazy.

Mirrors test_anticheat.py::test_relay_path_does_not_import_desktop_stack -- a
clean subprocess so we measure a pristine sys.modules, not one polluted by other
tests. This is the strongest guarantee that the host-side PTT path never pulls a
ban-class input lib into the process Vanguard sees.
"""
import subprocess
import sys
import textwrap


def test_ptt_imports_no_input_or_capture_libs():
    code = textwrap.dedent(
        """
        import sys
        import kenning.ptt
        from kenning.ptt import (
            build_ptt_controller, NullPttBackend, SerialHidPttBackend,
            RawHidPttBackend, PttController,
        )

        # Exercise the inert path + fail-safe serial AND raw-HID opens (bad
        # device) -- none may pull in an input/capture lib.
        c = PttController(NullPttBackend())
        c.hold(); c.release(); c.close()
        b = SerialHidPttBackend("NO_SUCH_PORT_XYZ", 9600,
                                open_serial=lambda p, baud: (_ for _ in ()).throw(OSError()))
        assert b.available is False
        h = RawHidPttBackend(0x1209, 0xFFC0,
                             open_hid=lambda v, p, u: (_ for _ in ()).throw(OSError()))
        assert h.available is False

        BANNED = [
            "pyautogui", "pynput", "keyboard", "pydirectinput", "inputs",
            "mss", "pyscreeze", "dxcam", "pywinauto", "uiautomation",
            "vgamepad", "vjoy", "pyvjoy", "interception",
            "kenning.desktop",
        ]
        loaded = [m for m in BANNED if m in sys.modules]
        assert not loaded, "PTT pulled in banned input/capture libs: %r" % loaded

        # pyserial AND hidapi must be LAZY (only imported when a real device is
        # opened), so importing the package + the inert paths never touch them.
        assert "serial" not in sys.modules, "pyserial should be lazy-imported"
        assert "hid" not in sys.modules, "hidapi should be lazy-imported"
        print("PTT_CLEAN_OK")
        """
    )
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
    )
    assert "PTT_CLEAN_OK" in res.stdout, (
        "clean-import probe failed\nSTDOUT:\n%s\nSTDERR:\n%s"
        % (res.stdout, res.stderr)
    )
