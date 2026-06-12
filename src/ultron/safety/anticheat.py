"""Anticheat-safe mode: a hard kill-switch for every OS-interaction surface.

Kernel-level anticheats (Vanguard, EAC, BattlEye) ban accounts for
input injection, screen capture of the game, window manipulation, and
anything resembling game-state reading. While this mode is ACTIVE,
Ultron disables EVERY capability in those classes -- synthetic input
(SendInput via pyautogui), screen capture (mss / pixel reads / template
matching / OCR), UIA tree reading, clipboard automation, dialog and
element automation, window close/move/launch placement, browser CDP
automation, desktop sequences, and the bridge's desktop tools -- while
keeping the audio-only pipeline fully alive: microphone capture, STT,
LLM, TTS to the speakers, and the voice relay into the VoiceMeeter
strip are pure shared-mode audio APIs (the same surface Discord uses)
and interact with no other process.

Deliberately NOT blocked (analyzed, categorically safe):
* Audio capture/playback (WASAPI/MME shared mode -- no game interaction).
* ``nvidia-smi`` GPU/VRAM queries (a signed NVIDIA binary doing global
  driver queries -- the same thing MSI Afterburner / GeForce overlay do;
  it never opens the game process).
* ``psutil`` self-management (killing Ultron's OWN child process tree,
  raising Ultron's OWN priority) -- never opens a foreign process handle
  beyond shell-level metadata.
* LLM / web search / memory / evolution (no OS interaction at all).

Enforcement is BELT-AND-SUSPENDERS, three layers:
1. Module guards -- every OS-touching public function calls
   :func:`guard` at entry and raises :class:`AnticheatBlockedError`
   BEFORE any OS API is imported or touched.
2. The safety validator -- :func:`is_blocked_tool` lets
   ``SafetyValidator.check`` return BLOCK_HARD for desktop tools, so
   every blocked attempt lands in the audit ledger.
3. The orchestrator -- desktop voice intents answer with
   :data:`BLOCKED_NOTICE` instead of dispatching.

The mode activates three ways: the voice toggle ("enable anticheat
mode"), automatically with gaming mode
(``gaming_mode.anticheat_with_gaming_mode``, default ON), or pinned
always-on via ``gaming_mode.anticheat_safe_mode``.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Optional

logger = logging.getLogger("ultron.safety.anticheat")

__all__ = [
    "AnticheatBlockedError",
    "BLOCKED_NOTICE",
    "anticheat_active",
    "set_anticheat_active",
    "guard",
    "is_blocked_tool",
    "match_anticheat_toggle",
]

BLOCKED_NOTICE = (
    "Anticheat mode is active. I won't touch the screen, keyboard, "
    "mouse, clipboard, or windows while you're in game."
)

_lock = threading.Lock()
_runtime_active = False
_activated_at: Optional[float] = None
_reason = ""


class AnticheatBlockedError(RuntimeError):
    """Raised by a guarded surface while anticheat-safe mode is active."""

    def __init__(self, action: str) -> None:
        super().__init__(
            f"anticheat-safe mode: {action!r} is disabled. {BLOCKED_NOTICE}"
        )
        self.action = action


def anticheat_active() -> bool:
    """True iff anticheat-safe mode is currently in force.

    Either the runtime toggle (voice / gaming-mode tie-in) or the
    pinned ``gaming_mode.anticheat_safe_mode`` config flag. Fail-open
    on config errors -- the RUNTIME flag alone still applies, so a
    config problem can never silently disable an explicit voice toggle.
    """
    if _runtime_active:
        return True
    try:
        from ultron.config import get_config

        return bool(getattr(
            getattr(get_config(), "gaming_mode", None),
            "anticheat_safe_mode", False,
        ))
    except Exception:  # noqa: BLE001
        return False


def set_anticheat_active(active: bool, reason: str = "") -> None:
    """Flip the runtime flag (voice toggle / gaming-mode engage)."""
    global _runtime_active, _activated_at, _reason
    with _lock:
        _runtime_active = bool(active)
        _activated_at = time.time() if active else None
        _reason = reason
    logger.info(
        "anticheat-safe mode %s%s",
        "ACTIVE" if active else "off",
        f" ({reason})" if reason else "",
    )


def guard(action: str) -> None:
    """Hard gate for OS-interaction entry points.

    Args:
        action: short label for logs/audit ("click", "screenshot", ...).

    Raises:
        AnticheatBlockedError: when anticheat-safe mode is active.
    """
    if anticheat_active():
        logger.warning("anticheat-safe mode blocked %r", action)
        raise AnticheatBlockedError(action)


# Tool names (and prefixes) the safety validator hard-blocks while the
# mode is active -- everything that reads the screen, injects input, or
# manipulates windows/clipboard.
_BLOCKED_TOOL_EXACT = frozenset({
    "click", "type_text", "scroll", "move_mouse", "drag_to",
    "screenshot", "get_pixel_color", "wait_for_pixel_color",
    "find_image_on_screen", "clipboard_read", "clipboard_write",
    "ocr", "semantic_click",
})
_BLOCKED_TOOL_PREFIXES = (
    "desktop_", "window_", "dialog_", "element_", "browser_use",
    "ui_", "screen_",
)


def is_blocked_tool(tool_name: str) -> bool:
    """True iff ``tool_name`` belongs to the anticheat-blocked classes."""
    name = (tool_name or "").strip().lower()
    if name in _BLOCKED_TOOL_EXACT:
        return True
    return any(name.startswith(p) for p in _BLOCKED_TOOL_PREFIXES)


# Voice toggle -- strict phrasings only.
_MODE_WORDS = r"(?:anticheat|anti-cheat|anti\s+cheat|tournament)\s+(?:safe\s+)?mode"
_TOGGLE_ON_RE = re.compile(
    rf"^(?:please\s+)?(?:enable|engage|activate|turn\s+on|start)\s+"
    rf"(?:the\s+)?{_MODE_WORDS}\s*[.!?]?$",
    re.IGNORECASE,
)
_TOGGLE_OFF_RE = re.compile(
    rf"^(?:please\s+)?(?:disable|disengage|deactivate|turn\s+off|stop|end)\s+"
    rf"(?:the\s+)?{_MODE_WORDS}\s*[.!?]?$",
    re.IGNORECASE,
)


def match_anticheat_toggle(text: str) -> Optional[bool]:
    """Match the strict anticheat-mode toggle phrasings.

    Returns:
        True for enable forms, False for disable forms, None otherwise.
    """
    if not text:
        return None
    cleaned = text.strip()
    if _TOGGLE_ON_RE.match(cleaned):
        return True
    if _TOGGLE_OFF_RE.match(cleaned):
        return False
    return None
