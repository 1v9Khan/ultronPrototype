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
import threading
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
    # 2026-06-17 audit (defense-in-depth): more browser-automation / CDP drivers.
    # None are vendored or imported by any allowed module; blocking the prefix
    # keeps prevent==detect symmetric for the whole CDP/webdriver family.
    "pyppeteer",
    "undetected_chromedriver",
    "DrissionPage",
    "helium",
    "comtypes.gen",   # generated UIA COM proxies (the actual UIA element tree);
                      # comtypes itself is left importable (pycaw/audio may pull
                      # it) but the generated UIAutomation client is never needed.
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
    # 2026-06-17 audit (defense-in-depth): input-sim / capture / clipboard / OCR /
    # window-enum / gamepad libs that are NEVER imported by any allowed (voice /
    # relay / audio / ptt) path -- confirmed by a sys.modules probe of the
    # always-loaded stack. Blocking them by exact name keeps prevent==detect
    # symmetric so a future misplaced import (outside the kenning.desktop prefix)
    # is refused at the loader, not just transitively. (win32api/win32gui/comtypes
    # are deliberately NOT here -- they are general win32/COM libs that pycaw/audio
    # may pull transitively; they stay covered by the kenning.desktop prefix.)
    "pyscreenshot",   # cross-backend screen capture
    "bettercam",      # DXGI/Desktop-Duplication capture (dxcam successor)
    "windows_capture",  # Windows.Graphics.Capture wrapper
    "pygetwindow",    # window enumeration / geometry
    "pyperclip",      # clipboard read/write
    "win32clipboard", # pywin32 clipboard
    # NOTE: pytesseract (OCR) is DELIBERATELY NOT blocked here. transformers
    # (pulled in by Kokoro TTS + Whisper) probes it at IMPORT time via
    # importlib.util.find_spec("pytesseract"); this finder RAISES inside that
    # probe, so the whole transformers import fails and Kokoro/Whisper/Smart-Turn
    # go silent. pytesseract is not installed anyway, and the OCR capability is
    # already blocked via the kenning.desktop prefix (kenning.desktop.ocr), so
    # omitting the bare name costs zero protection. 2026-06-18 hotfix (reapplied).
    "ahk",            # AutoHotkey driver (synthetic input + hotkeys)
    "pyautoit",       # AutoIt driver (synthetic input)
    "autoit",         # AutoIt driver (synthetic input)
    "inputs",         # raw gamepad/keyboard/mouse event device access
    "interception",   # Interception driver (kernel input injection)
    "vgamepad",       # virtual gamepad (ViGEm)
    "pyvjoy",         # virtual joystick (vJoy)
    "pydivert",       # WinDivert packet interception
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
        # 2026-06-17 audit: FAIL-SAFE -- if we cannot DETERMINE the anticheat
        # state (e.g. a transient config-read error early in boot), BLOCK the
        # import. A clean, confident `anticheat_active() == False` (a deliberate
        # desktop session) still allows the import; only UNCERTAINTY blocks. This
        # closes the "config error -> firewall silently fails open" gap.
        try:
            from kenning.safety.anticheat import anticheat_active
            active = bool(anticheat_active())
        except Exception:                                            # noqa: BLE001
            active = True   # fail-SAFE: uncertain -> block (anticheat-correct)
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
_INSTALL_LOCK = threading.Lock()


def install_import_firewall() -> bool:
    """Insert the firewall at the FRONT of ``sys.meta_path`` (idempotent).

    Safe to call unconditionally at boot: while anticheat mode is inactive the
    firewall is a no-op (it returns ``None`` for every import), so non-gaming
    sessions keep full desktop/browser capability. Returns True if it installed
    (or was already installed)."""
    global _INSTALLED
    # 2026-06-17 audit: guard the check-and-set so two concurrent boot callers
    # (entry + Orchestrator.__init__ on different threads) can't both insert a
    # duplicate finder.
    with _INSTALL_LOCK:
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
    """True if the firewall is present on ``sys.meta_path`` (always scans the
    live list, so a manual removal can't be masked by the cached flag)."""
    return any(isinstance(f, AnticheatImportFirewall) for f in sys.meta_path)


def assert_firewall_enforces() -> bool:
    """Prove the firewall actually BITES (not just that it's installed).

    When anticheat is active, attempt to import a module that is on the blocklist
    but is NOT installed in the venv (so a clean machine never has it cached):
    the firewall MUST raise ImportError. Returns True iff enforcement is verified
    (or anticheat is inactive, where the firewall is intentionally a no-op).
    Logged at ERROR if a blocked import unexpectedly succeeds -- that is a
    loader-level regression that must be caught before going live.

    Uses ``interception`` (an input-injection driver, on the blocklist, not a
    dependency) so the probe never collides with a real, importable module."""
    try:
        from kenning.safety.anticheat import anticheat_active
        if not anticheat_active():
            return True  # firewall deliberately inert outside anticheat mode
    except Exception:                                                # noqa: BLE001
        pass  # uncertain -> still run the probe (the firewall fails safe anyway)
    probe = "interception"  # blocked + not installed -> ImportError is the firewall's
    try:
        __import__(probe)
    except ImportError as e:
        if "anticheat import firewall" in str(e):
            logger.info("anticheat import firewall ENFORCEMENT verified "
                        "(blocked %r raised the firewall's ImportError)", probe)
            return True
        # ImportError, but NOT ours -> the module is merely absent and the
        # firewall did not intercept it. Could mean the finder isn't biting.
        if is_blocked_module(probe) and is_firewall_installed():
            # Installed + the name IS blocked, yet a plain ModuleNotFound came
            # back -> ambiguous; treat as verified only if the firewall is on
            # meta_path[0]. Be conservative and log.
            logger.warning("anticheat import firewall: probe %r returned a "
                           "non-firewall ImportError (%s); finder present but "
                           "did not intercept -- verify meta_path ordering.",
                           probe, e)
        return is_firewall_installed()
    # The import SUCCEEDED -> a blocked module loaded. Loader-level regression.
    logger.error("ANTICHEAT IMPORT FIREWALL REGRESSION: blocked probe %r "
                 "imported WITHOUT being refused -- the loader-level block is "
                 "NOT enforcing. DO NOT go live until fixed.", probe)
    return False
