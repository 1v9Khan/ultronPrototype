"""Anticheat import firewall.

A ``sys.meta_path`` finder that HARD-BLOCKS importing OS input-injection,
screen-capture, window-control, browser-automation, and desktop-automation
modules while anticheat-safe mode is active.

This is the backstop that makes the "nothing dangerous loads at runtime"
guarantee robust. The boot-time gates only stop EAGER (module-top) imports;
a lazy/conditional import buried inside a function body bypasses them until
that function is called. The firewall closes that hole at the loader level:
no matter WHERE an ``import`` statement lives, if anything attempts to import
a blocked module while :func:`kenning.safety.anticheat.anticheat_active` is
True, the import raises :class:`ImportError` BEFORE the module's code runs --
so its transitive ``pyautogui`` / ``mss`` / ``pywinauto`` / ``playwright``
imports never load into the process either.

It reads the anticheat flag LIVE on every blocked-module import, so it also
covers the case where the user toggles anticheat mode on mid-session.

Benign modules are NEVER blocked: ``win32gui`` / ``win32con`` (the overlay's
OBS-capturable window), ``PIL`` (the nameplate), ``torch`` / ``transformers``
/ ``faster_whisper`` / ``numpy``, and the rest of ``kenning.*`` outside the
desktop-automation package.
"""

from __future__ import annotations

import sys
from importlib.abc import MetaPathFinder

from kenning.utils.logging import get_logger

logger = get_logger("safety.import_firewall")

# Module-name PREFIXES that are blocked (the module itself AND any submodule).
# kenning.desktop is the in-process automation package whose __init__ eagerly
# pulls the whole pyautogui / mss / pywinauto stack, so blocking the prefix
# stops the entire stack from ever loading.
_BLOCK_PREFIXES = (
    "kenning.desktop",
    "kenning.openclaw_bridge.browser",
    "kenning.openclaw_bridge.desktop",
    # 2026-06-15 audit: src/ultron/ is a STALE pre-rename mirror of kenning,
    # never imported by the runtime, but its desktop/browser submodules exist on
    # disk. Block them too so a stray/accidental import can never load the stale
    # automation code while gaming. (Its dangerous deps -- pyautogui/mss/etc --
    # are already blocked by exact name regardless of importer; this is belt-2.)
    "ultron.desktop",
    "ultron.openclaw_bridge.browser",
    "ultron.openclaw_bridge.desktop",
    "playwright",
    "browser_use",
    "selenium",
    "pywinauto",
    "pynput",
    "pyscreeze",
    "uiautomation",
)

# Exact module names that are blocked (no submodule semantics needed).
_BLOCK_EXACT = frozenset({
    "pyautogui",
    "mss",
    "dxcam",
    "PIL.ImageGrab",
    # 2026-06-15 audit hardening: input-simulation / global-hook / capture libs
    # that the canary already watches for but the firewall previously did NOT
    # refuse at the loader. None are used by any allowed module, so blocking them
    # is pure defense-in-depth (keeps prevent and detect symmetric).
    "keyboard",       # global low-level keyboard hook (SetWindowsHookEx)
    "mouse",          # global low-level mouse hook
    "pydirectinput",  # SendInput wrapper (DirectInput scancodes)
    "d3dshot",        # DXGI desktop-duplication screen capture
})


def is_blocked_module(fullname: str) -> bool:
    """True if ``fullname`` is an anticheat-blocked module name."""
    if fullname in _BLOCK_EXACT:
        return True
    for p in _BLOCK_PREFIXES:
        if fullname == p or fullname.startswith(p + "."):
            return True
    return False


def blocked_module_names() -> tuple:
    """The full (prefixes + exact) block list, for the canary + tests."""
    return tuple(_BLOCK_PREFIXES) + tuple(sorted(_BLOCK_EXACT))


class AnticheatImportFirewall(MetaPathFinder):
    """A meta-path finder that refuses blocked imports while anticheat is on."""

    def find_spec(self, fullname, path=None, target=None):
        if not is_blocked_module(fullname):
            return None  # not our concern -> defer to the normal finders
        # Read the flag LIVE so a mid-session anticheat toggle is honoured and
        # so non-gaming sessions can still use the desktop/browser tools.
        try:
            from kenning.safety.anticheat import anticheat_active
            active = bool(anticheat_active())
        except Exception:                                            # noqa: BLE001
            active = False
        if not active:
            return None  # firewall only bites while anticheat-safe mode is on
        logger.error(
            "ANTICHEAT IMPORT FIREWALL: refused runtime import of %r -- "
            "input/capture/automation modules must never load into the "
            "process while a protected game is running.",
            fullname,
        )
        raise ImportError(
            f"anticheat import firewall: {fullname!r} is blocked while "
            f"anticheat-safe mode is active (no input/capture/automation code "
            f"may load into this process during a protected game)"
        )


_INSTALLED = False


def install_import_firewall() -> bool:
    """Insert the firewall at the FRONT of ``sys.meta_path`` (idempotent).

    Safe to call unconditionally at boot: while anticheat mode is inactive the
    firewall is a no-op (it returns ``None`` for every import), so non-gaming
    sessions keep full desktop/browser capability. Returns True if it installed
    (or was already installed)."""
    global _INSTALLED
    if _INSTALLED:
        return True
    sys.meta_path.insert(0, AnticheatImportFirewall())
    _INSTALLED = True
    logger.info(
        "anticheat import firewall installed (loader-level block on "
        "desktop/browser/input/capture modules whenever anticheat-safe mode "
        "is active): %s",
        ", ".join(blocked_module_names()),
    )
    return True


def is_firewall_installed() -> bool:
    """True if the firewall is present on ``sys.meta_path``."""
    if _INSTALLED:
        return True
    return any(isinstance(f, AnticheatImportFirewall) for f in sys.meta_path)
