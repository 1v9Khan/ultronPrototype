"""Anticheat-safe mode: a hard kill-switch for every OS-interaction surface.

Kernel-level anticheats (Vanguard, EAC, BattlEye) ban accounts for
input injection, screen capture of the game, window manipulation, and
anything resembling game-state reading. While this mode is ACTIVE,
Kenning disables EVERY capability in those classes -- synthetic input
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
* ``psutil`` self-management (killing Kenning's OWN child process tree,
  raising Kenning's OWN priority) -- never opens a foreign process handle
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

Vanguard (kernel-level) analysis, 2026-06-11: the classes a boot-time
kernel anticheat actually bans for are foreign-process handle opens,
game-memory read/write, remote-thread/DLL injection, global input
hooks (``SetWindowsHookEx``), raw-input device registration, and
driver-level input emulation. Kenning's source contains ZERO uses of
any of these (pinned by ``test_no_ban_class_apis_anywhere_in_source``;
the only textual matches in the repo are the ``safety/rules/``
DEFENSE regexes that block those patterns in model-proposed commands).
What Kenning CAN do -- ``SendInput`` via pyautogui, GDI screen capture
via mss, UIA COM reads (which message target windows), clipboard,
window management -- is exactly the surface this mode hard-blocks,
out of maximum caution, even though none of it opens the game process.
The remaining active surface while blocked (shared-mode audio,
``nvidia-smi`` global driver queries, self-scoped ``psutil``,
shell-level window-metadata reads) is the same surface Discord, OBS,
MSI Afterburner, and the Windows taskbar exercise continuously on
every gamer's machine.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Optional

logger = logging.getLogger("kenning.safety.anticheat")

__all__ = [
    "AnticheatBlockedError",
    "BLOCKED_NOTICE",
    "anticheat_active",
    "set_anticheat_active",
    "guard",
    "is_blocked_tool",
    "match_anticheat_toggle",
    "register_surface_hook",
    "clear_surface_hooks",
]

BLOCKED_NOTICE = (
    "Anticheat mode is active. I won't touch the screen, keyboard, "
    "mouse, clipboard, or windows while you're in game."
)

_lock = threading.Lock()
_runtime_active = False
_activated_at: Optional[float] = None
_reason = ""
# Test-sweep guard: the unit sweep must stay hermetic even when the
# user's config.yaml pins ``gaming_mode.anticheat_safe_mode: true``
# (otherwise every desktop test would raise). conftest disables the
# CONFIG pin for the session; the runtime flag stays fully testable.
_config_pin_enabled = True


def set_config_pin_enabled(enabled: bool) -> None:
    """Enable/ignore the ``anticheat_safe_mode`` config pin (tests)."""
    global _config_pin_enabled
    _config_pin_enabled = bool(enabled)


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
    # Testing mode mimics the gaming/anticheat disabled-functionality posture
    # (desktop automation hard-blocked) while allowing the GPU. Separate flag;
    # never set by gaming/anticheat engage.
    try:
        from kenning.safety.testing_mode import is_testing_mode_active

        if is_testing_mode_active():
            return True
    except Exception:  # noqa: BLE001
        pass
    if not _config_pin_enabled:
        return False
    try:
        from kenning.config import get_config

        return bool(getattr(
            getattr(get_config(), "gaming_mode", None),
            "anticheat_safe_mode", False,
        ))
    except Exception:  # noqa: BLE001
        return False


# Surface hooks: callables invoked on every mode flip so RUNNING
# subsystems are physically STOPPED, not merely call-gated. A kernel
# anticheat observes what a process is DOING -- a UIA poller thread
# still polling, or a cached mss capture object holding GDI handles,
# is activity; the hooks shut those down on activate and restore them
# on deactivate. Each hook receives the new ``active`` state and is
# fail-open (one broken hook never blocks the mode flip or the others).
_surface_hooks: list[tuple[str, "object"]] = []


def register_surface_hook(name: str, hook) -> None:
    """Register a ``hook(active: bool)`` called on every mode flip.

    Args:
        name: short label for logs ("dialog_poller", ...).
        hook: callable taking the new active state. On ``True`` it
            should STOP/unload its subsystem; on ``False`` restore it.
    """
    _surface_hooks.append((name, hook))


def clear_surface_hooks() -> None:
    """Drop all registered hooks (tests + orchestrator shutdown)."""
    _surface_hooks.clear()


def set_anticheat_active(active: bool, reason: str = "") -> None:
    """Flip the runtime flag (voice toggle / gaming-mode engage) and
    run every registered surface hook so gated subsystems are fully
    stopped (or restored), not just call-blocked."""
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
    for name, hook in list(_surface_hooks):
        try:
            hook(bool(active))
            logger.info(
                "anticheat surface %r %s", name,
                "stopped" if active else "restored",
            )
        except Exception as e:  # noqa: BLE001 - hooks are fail-open
            logger.warning("anticheat surface hook %r failed: %s", name, e)


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
    "press_key", "press_hotkey",
    "screenshot", "get_pixel_color", "wait_for_pixel_color",
    "find_image_on_screen", "clipboard_read", "clipboard_write",
    "ocr", "semantic_click",
})
_BLOCKED_TOOL_PREFIXES = (
    "desktop_", "window_", "dialog_", "element_", "browser_use",
    "ui_", "screen_",
)


def is_blocked_tool(tool_name: str) -> bool:
    """True iff ``tool_name`` belongs to the anticheat-blocked classes.

    Dispatcher / bridge names arrive namespaced and dotted -- e.g.
    ``openclaw.window_automation`` (the OpenClaw dispatcher) or
    ``desktop.input.press_hotkey`` (the desktop input bridge). Normalize both
    forms so they still land in the BLOCK_HARD + audit-ledger layer: strip a
    leading ``openclaw.`` namespace, and also test the bare final dotted
    segment against the exact + prefix sets. Fail-open on junk input.
    """
    name = (tool_name or "").strip().lower()
    if not name:
        return False
    if name.startswith("openclaw."):
        name = name[len("openclaw."):]
    bare = name.rsplit(".", 1)[-1]
    for cand in (name, bare):
        if cand in _BLOCKED_TOOL_EXACT:
            return True
        if any(cand.startswith(p) for p in _BLOCKED_TOOL_PREFIXES):
            return True
    return False


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
