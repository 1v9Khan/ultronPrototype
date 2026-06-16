"""Pins the anticheat import firewall.

The firewall is the loader-level backstop guaranteeing that no lazy/conditional
import anywhere can pull a desktop/browser/input/capture/automation module into
the process while anticheat-safe mode is active. These tests assert: the block
list is correct (dangerous blocked, benign allowed), find_spec refuses blocked
imports ONLY while the mode is active, and install is idempotent.
"""

import pytest

from kenning.safety import import_firewall as fw
from kenning.safety import anticheat


DANGEROUS = [
    "kenning.desktop",
    "kenning.desktop.launcher",
    "kenning.desktop.screen_context",
    "kenning.desktop.vlm",
    "kenning.desktop.input_control",
    "kenning.desktop.capture",
    "kenning.openclaw_bridge.browser",
    "kenning.openclaw_bridge.desktop",
    "pyautogui",
    "mss",
    "dxcam",
    "d3dshot",
    "PIL.ImageGrab",
    "keyboard",
    "mouse",
    "pydirectinput",
    "playwright",
    "playwright.sync_api",
    "browser_use",
    "selenium",
    "pywinauto",
    "pynput",
    "pyscreeze",
    "uiautomation",
    # 2026-06-15 audit: the stale src/ultron mirror's desktop/browser submodules.
    "ultron.desktop",
    "ultron.desktop.input_control",
    "ultron.openclaw_bridge.browser",
]

BENIGN = [
    "win32gui",            # overlay window styling (OBS-capturable)
    "win32con",
    "PIL",                 # nameplate
    "PIL.Image",
    "numpy",
    "torch",
    "transformers",
    "faster_whisper",
    "kenning.config",
    "kenning.audio.waveform",
    "kenning.openclaw_bridge",        # the HTTP client package itself
    "kenning.openclaw_routing",       # routing logic, no in-process automation
    "kenning.safety.anticheat",
]


def test_is_blocked_module_dangerous():
    for m in DANGEROUS:
        assert fw.is_blocked_module(m), f"should be blocked: {m}"


def test_is_blocked_module_benign():
    for m in BENIGN:
        assert not fw.is_blocked_module(m), f"should NOT be blocked: {m}"


def test_find_spec_refuses_when_active(monkeypatch):
    monkeypatch.setattr(anticheat, "anticheat_active", lambda: True)
    finder = fw.AnticheatImportFirewall()
    for m in ("kenning.desktop.launcher", "pyautogui", "playwright.sync_api",
              "kenning.openclaw_bridge.browser", "mss"):
        with pytest.raises(ImportError):
            finder.find_spec(m)
    # Benign modules defer to the normal finders (return None) even when active.
    for m in ("win32gui", "PIL", "kenning.config", "numpy"):
        assert finder.find_spec(m) is None


def test_find_spec_noop_when_inactive(monkeypatch):
    monkeypatch.setattr(anticheat, "anticheat_active", lambda: False)
    finder = fw.AnticheatImportFirewall()
    # While the mode is OFF, even blocked modules defer to normal import
    # (non-gaming sessions keep full desktop/browser capability).
    for m in ("kenning.desktop.launcher", "pyautogui", "playwright"):
        assert finder.find_spec(m) is None


def test_install_is_idempotent():
    assert fw.install_import_firewall() is True
    assert fw.install_import_firewall() is True
    assert fw.is_firewall_installed() is True


def test_real_import_blocked_when_active(monkeypatch):
    """End-to-end: an actual `import` of a not-yet-loaded blocked module raises
    while anticheat is active (the firewall must be installed on sys.meta_path)."""
    import importlib
    fw.install_import_firewall()
    monkeypatch.setattr(anticheat, "anticheat_active", lambda: True)
    # Use a blocked module that is NOT already in sys.modules so find_spec runs.
    # `uiautomation` is a blocked prefix and not imported anywhere at import time.
    with pytest.raises(ImportError):
        importlib.import_module("uiautomation")
