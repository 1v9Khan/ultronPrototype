"""Mouse + keyboard automation via ``pyautogui``.

Distinct from :mod:`ultron.desktop.uia` -- this module does
pixel-coordinate / synthetic-input control rather than semantic UIA
clicks. Use UIA when the target is a standard UI element; use this
module when:

- the target is canvas-rendered (games, image viewers, video players)
- you need keyboard hotkeys (Ctrl+S, Alt+Tab, etc.)
- you need scroll
- you have explicit coordinates from a VLM / OCR pass

Safety:

- Every action passes through the runtime tool-call validator. Cap-4
  rules block synthetic input near UAC / security-class windows;
  Cap-3 action-verb rules block clicks on payment / OAuth buttons by
  label match. Validator runs BEFORE pyautogui touches the OS.
- Foreground-window check: actions refuse when the current foreground
  window's class name matches a known security pattern (UAC consent
  dialog, Windows Defender, credential UI). This is belt-and-braces
  on top of the Cap-4 rule because the rule is a regex match on
  argument values; the foreground check is a runtime check on actual
  on-screen state.
- Rate limit: ``max_actions_per_second`` (default 5) caps how fast a
  caller (or runaway agent) can fire actions. Exceeding the rate
  limit fails the call rather than blocking -- the orchestrator
  doesn't want to deadlock waiting on a stuck input loop.
- pyautogui's own failsafe (move mouse to corner to abort) stays
  on; do NOT disable it.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Optional

import pyautogui  # type: ignore[import]

from ultron.utils.logging import get_logger

logger = get_logger("desktop.input_control")

# Window class names whose foreground presence blocks synthetic input.
# These match Windows' built-in security UI surfaces.
_SECURITY_WINDOW_CLASSES = frozenset({
    "Credential Dialog Xaml Host",
    "CredentialUIControl",
    "ConsentUI",
    "UACDialog",
    "Windows.UI.Core.CoreWindow",  # generic; further checked by title
    "Shell_Dialog",
})

# Window titles that further qualify _SECURITY_WINDOW_CLASSES matches
# (some legit UWP apps share Windows.UI.Core.CoreWindow).
_SECURITY_TITLE_KEYWORDS = (
    "user account control",
    "windows security",
    "credential",
    "sign in",  # Microsoft account sign-in dialogs
    "two-factor",
    "smartscreen",
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputControlResult:
    """Outcome of one input action."""

    success: bool
    action: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Foreground security check
# ---------------------------------------------------------------------------


def _foreground_is_security_window() -> bool:
    """True iff the current foreground window is a UAC / Windows Security UI.

    Synthetic input on these is blocked by Windows itself (UIPI: User
    Interface Privilege Isolation) but we refuse upstream so the
    refusal is logged with context for audit.
    """
    try:
        from ultron.desktop.windows import get_foreground_window
        fg = get_foreground_window()
    except Exception as e:  # noqa: BLE001
        logger.debug("foreground check failed: %s", e)
        return False
    if fg is None:
        return False

    if fg.class_name in _SECURITY_WINDOW_CLASSES:
        # Some classes are too broad; narrow by title.
        if fg.class_name == "Windows.UI.Core.CoreWindow":
            title_l = fg.title.lower()
            return any(kw in title_l for kw in _SECURITY_TITLE_KEYWORDS)
        return True
    return False


# ---------------------------------------------------------------------------
# Safety validator hook
# ---------------------------------------------------------------------------


def _validate_input_action(
    *,
    action: str,
    arguments: dict,
    user_text: str = "",
) -> object:
    """Run the safety validator against an input action.

    Cap-4 rules check for synthetic input near security windows by
    inspecting argument values; we ALSO check foreground state
    directly in the controller.
    """
    try:
        from ultron.safety.validator import RuleContext, get_validator

        ctx = RuleContext(
            tool_name=f"desktop.input.{action}",
            arguments=arguments,
            capability="desktop_input",
            user_text=user_text,
        )
        return get_validator().check(ctx)
    except Exception as e:  # noqa: BLE001
        logger.debug("input_control validator skipped: %s", e)
        from ultron.safety.validator import ValidatorVerdict, Verdict
        return ValidatorVerdict(
            verdict=Verdict.ALLOW, reason="validator unavailable",
        )


# ---------------------------------------------------------------------------
# InputController
# ---------------------------------------------------------------------------


class InputController:
    """Pyautogui-backed input controller with rate limiting + safety gate."""

    def __init__(
        self,
        *,
        max_actions_per_second: float = 5.0,
        enforce_security_window_block: bool = True,
        click_preview_enabled: bool = False,
        click_preview_capture_screen: Optional[object] = None,
        click_preview_vlm_describe: Optional[object] = None,
        click_preview_auto_pass_radius_px: int = 100,
        click_preview_crosshair_size: int = 20,
        click_preview_crosshair_thickness: int = 3,
        click_preview_require_confirmation_keyword: str = "yes",
        click_preview_history_depth: int = 20,
        click_preview_block_on_degraded: bool = False,
    ) -> None:
        self._rate = max(0.1, float(max_actions_per_second))
        self._enforce_security_block = bool(enforce_security_window_block)
        # Track timestamps of the last N actions to enforce the rate limit.
        self._action_times: deque[float] = deque(maxlen=64)
        self._lock = Lock()

        # 2026-05-24 SWE-Agent batch 7 (T16): click-preview gate.
        # Lazy-imported so the preview-disabled path doesn't pay the
        # Pillow import at module load.
        self._click_preview_enabled = bool(click_preview_enabled)
        self._click_preview_capture_screen = click_preview_capture_screen
        self._click_preview_vlm_describe = click_preview_vlm_describe
        self._click_preview_auto_pass_radius_px = int(click_preview_auto_pass_radius_px)
        self._click_preview_crosshair_size = int(click_preview_crosshair_size)
        self._click_preview_crosshair_thickness = int(click_preview_crosshair_thickness)
        self._click_preview_require_confirmation_keyword = str(
            click_preview_require_confirmation_keyword
        )
        self._click_preview_block_on_degraded = bool(click_preview_block_on_degraded)
        self._click_preview_history = None  # built lazily on first preview call
        self._click_preview_history_depth = int(click_preview_history_depth)

        # Keep pyautogui's failsafe enabled (move mouse to corner aborts).
        try:
            pyautogui.FAILSAFE = True  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    def _maybe_preview_click(
        self,
        *,
        x: Optional[int],
        y: Optional[int],
        user_text: str,
    ) -> Optional[InputControlResult]:
        """Run the SWE-Agent click-preview gate before firing.

        Returns:
            ``None`` when the preview path is disabled, the
            coordinates are absent, or the gate decides ALLOW /
            AUTO_PASS / DEGRADED-with-allow. Returns a failure
            :class:`InputControlResult` when the gate BLOCKs.
        """
        if not self._click_preview_enabled:
            return None
        if x is None or y is None:
            # No coordinate to preview -- pyautogui will click at the
            # current cursor location. Skip the preview rather than
            # try to capture an arbitrary point.
            return None
        capture = self._click_preview_capture_screen
        if capture is None:
            return None
        try:
            from ultron.desktop.click_preview import (
                ConfirmationHistory,
                PreviewDecision,
                preview_click,
            )
        except Exception as e:                                          # noqa: BLE001
            logger.debug("click_preview unavailable: %s", e)
            return None
        if self._click_preview_history is None:
            self._click_preview_history = ConfirmationHistory(
                max_entries=self._click_preview_history_depth,
            )
        try:
            result = preview_click(
                x=int(x),
                y=int(y),
                capture_screen=capture,
                vlm_describe=self._click_preview_vlm_describe,
                history=self._click_preview_history,
                intent_description=user_text or "click on the target",
                auto_pass_radius=self._click_preview_auto_pass_radius_px,
                crosshair_size=self._click_preview_crosshair_size,
                crosshair_thickness=self._click_preview_crosshair_thickness,
                require_confirmation_keyword=self._click_preview_require_confirmation_keyword,
            )
        except Exception as e:                                          # noqa: BLE001
            logger.warning("click_preview gate raised: %s", e)
            return None
        if result.decision is PreviewDecision.BLOCK:
            return InputControlResult(
                success=False,
                action="click",
                error=f"click_preview: BLOCK: {result.reason or result.vlm_response[:160]}",
            )
        if (
            result.decision is PreviewDecision.DEGRADED
            and self._click_preview_block_on_degraded
        ):
            return InputControlResult(
                success=False,
                action="click",
                error=f"click_preview: DEGRADED (blocked per config): {result.reason}",
            )
        return None

    # ---- gating helpers ----

    def _gate(
        self,
        *,
        action: str,
        arguments: dict,
        user_text: str = "",
    ) -> Optional[InputControlResult]:
        """Run security + rate-limit + validator. Returns a failure result
        when the action is refused, or None when the action may proceed.
        """
        if self._enforce_security_block and _foreground_is_security_window():
            return InputControlResult(
                success=False, action=action,
                error="refused: a Windows security window is in the foreground",
            )

        if not self._take_rate_slot():
            return InputControlResult(
                success=False, action=action,
                error=f"refused: rate limit of {self._rate:.1f} actions/s exceeded",
            )

        verdict = _validate_input_action(
            action=action, arguments=arguments, user_text=user_text,
        )
        if not verdict.is_allowed:
            return InputControlResult(
                success=False, action=action,
                error=f"safety: {verdict.reason}",
            )
        return None

    def _take_rate_slot(self) -> bool:
        """Track a new action timestamp; return False if over the rate cap."""
        now = time.monotonic()
        with self._lock:
            window = 1.0
            # Drop entries older than the rolling 1s window.
            while self._action_times and now - self._action_times[0] > window:
                self._action_times.popleft()
            if len(self._action_times) >= int(self._rate):
                return False
            self._action_times.append(now)
            return True

    # ---- public actions ----

    def move_mouse(
        self,
        x: int,
        y: int,
        *,
        duration_s: float = 0.1,
        smooth: bool = False,
        user_text: str = "",
    ) -> InputControlResult:
        """Move the cursor to (x, y). Duration smooths the motion.

        Catalog 09 T7 (GREEN): when ``smooth=True`` AND ``duration_s > 0``
        the move uses a quadratic ease-in / ease-out tween
        (:func:`pyautogui.easeInOutQuad`) so the cursor follows a
        bezier-shaped acceleration-then-deceleration curve rather than
        a straight constant-velocity line. Useful for gaming-mode
        anti-detection (some anti-cheat heuristics flag teleporting
        cursors) and demo-mode narration where the user is watching
        the cursor move. With ``smooth=False`` (default, back-compat)
        or ``duration_s=0``, pyautogui's default linear motion is used.
        """
        # Anticheat-safe mode: hard-blocked while the user is in game.
        from ultron.safety.anticheat import guard as _anticheat_guard
        _anticheat_guard('move_mouse')
        args = {
            "x": int(x),
            "y": int(y),
            "duration_s": float(duration_s),
            "smooth": bool(smooth),
        }
        gate = self._gate(
            action="move_mouse",
            arguments=args,
            user_text=user_text,
        )
        if gate is not None:
            return gate
        clamped_duration = max(0.0, float(duration_s))
        try:
            if smooth and clamped_duration > 0:
                pyautogui.moveTo(
                    int(x), int(y),
                    duration=clamped_duration,
                    tween=pyautogui.easeInOutQuad,
                )
            else:
                pyautogui.moveTo(
                    int(x), int(y), duration=clamped_duration,
                )
        except Exception as e:  # noqa: BLE001
            return InputControlResult(
                success=False, action="move_mouse", error=str(e)[:200],
            )
        return InputControlResult(success=True, action="move_mouse")

    def click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        *,
        button: str = "left",
        clicks: int = 1,
        interval_s: float = 0.05,
        user_text: str = "",
    ) -> InputControlResult:
        """Mouse click. When ``x``/``y`` are None, clicks at the current
        cursor location.

        ``button`` accepts ``"left"`` / ``"right"`` / ``"middle"``.
        ``clicks=2`` performs a double click.
        """
        # Anticheat-safe mode: hard-blocked while the user is in game.
        from ultron.safety.anticheat import guard as _anticheat_guard
        _anticheat_guard('click')
        if button not in ("left", "right", "middle"):
            return InputControlResult(
                success=False, action="click",
                error=f"unknown button {button!r}",
            )
        if clicks < 1 or clicks > 5:
            return InputControlResult(
                success=False, action="click",
                error=f"clicks out of range: {clicks}",
            )

        args = {
            "button": button,
            "clicks": int(clicks),
        }
        if x is not None:
            args["x"] = int(x)
        if y is not None:
            args["y"] = int(y)

        gate = self._gate(
            action="click", arguments=args, user_text=user_text,
        )
        if gate is not None:
            return gate

        # 2026-05-24 SWE-Agent batch 7 (T16): visual crosshair preview
        # gate. Runs only when ``click_preview_enabled=True``. Defaults
        # to OFF -- when off this is a single-attribute branch.
        preview_block = self._maybe_preview_click(
            x=x, y=y, user_text=user_text,
        )
        if preview_block is not None:
            return preview_block

        # T1 (catalog 07): pyautogui's Windows backend uses
        # ``SendInput`` (atomic, interleave-safe since Windows 2000),
        # not the legacy ``mouse_event`` API. Multi-event sequences
        # from this method coalesce into a single input block that
        # other input sources cannot interleave. Documenting the API
        # choice; no functional change.
        try:
            pyautogui.click(
                x=int(x) if x is not None else None,
                y=int(y) if y is not None else None,
                button=button, clicks=int(clicks),
                interval=max(0.0, float(interval_s)),
            )
        except Exception as e:  # noqa: BLE001
            return InputControlResult(
                success=False, action="click", error=str(e)[:200],
            )
        return InputControlResult(success=True, action="click")

    def type_text(
        self,
        text: str,
        *,
        interval_s: float = 0.0,
        wpm: Optional[int] = None,
        user_text: str = "",
    ) -> InputControlResult:
        """Type a string at the current focus.

        Use :meth:`ultron.desktop.uia.type_text_into_element` for
        targeting a specific UI element semantically.

        Catalog 09 T3 (GREEN): optional ``wpm`` kwarg overrides
        ``interval_s`` with a human-cadence per-character delay. The
        conversion uses the standard 5-characters-per-word assumption:

            chars_per_second = (wpm * 5) / 60
            interval_s       = 1.0 / chars_per_second

        Some web forms with JavaScript validators and remote desktop
        sessions reject instant input (``interval_s=0``). 60-80 WPM
        passes most rate detectors. The voice-side intent surface
        maps "type slowly" -> 30 WPM, "type normally" -> 60 WPM,
        "type fast" -> 120 WPM. Non-positive ``wpm`` values are
        rejected (the upstream plugin's bare division would raise
        ``ZeroDivisionError`` -- ultron returns a structured error
        instead).
        """
        # Anticheat-safe mode: hard-blocked while the user is in game.
        from ultron.safety.anticheat import guard as _anticheat_guard
        _anticheat_guard('type_text')
        if not isinstance(text, str):
            return InputControlResult(
                success=False, action="type_text", error="text must be str",
            )
        if not text:
            return InputControlResult(success=True, action="type_text")

        # T3: convert WPM -> per-character interval. wpm takes priority
        # over interval_s when both are supplied (matches the upstream
        # plugin's contract). Guard against wpm <= 0 -- the upstream's
        # bare ``1.0 / ((wpm * 5) / 60)`` would raise on 0 and produce
        # a negative interval on negative values.
        effective_interval = float(interval_s)
        if wpm is not None:
            if wpm <= 0:
                return InputControlResult(
                    success=False, action="type_text",
                    error=f"wpm must be positive, got {wpm}",
                )
            effective_interval = 1.0 / ((float(wpm) * 5.0) / 60.0)

        gate = self._gate(
            action="type_text",
            arguments={
                "text_preview": text[:120],
                "length": len(text),
                "interval_s": effective_interval,
                "wpm": int(wpm) if wpm is not None else None,
            },
            user_text=user_text,
        )
        if gate is not None:
            return gate

        try:
            pyautogui.write(text, interval=max(0.0, effective_interval))
        except Exception as e:  # noqa: BLE001
            return InputControlResult(
                success=False, action="type_text", error=str(e)[:200],
            )
        return InputControlResult(success=True, action="type_text")

    def press_key(self, key: str, *, user_text: str = "") -> InputControlResult:
        """Press and release a single key (``"enter"``, ``"esc"``,
        ``"f5"``, etc.).
        """
        if not isinstance(key, str) or not key.strip():
            return InputControlResult(
                success=False, action="press_key", error="empty key",
            )
        gate = self._gate(
            action="press_key", arguments={"key": key},
            user_text=user_text,
        )
        if gate is not None:
            return gate
        try:
            pyautogui.press(key)
        except Exception as e:  # noqa: BLE001
            return InputControlResult(
                success=False, action="press_key", error=str(e)[:200],
            )
        return InputControlResult(success=True, action="press_key")

    def press_hotkey(
        self, *keys: str, user_text: str = "",
    ) -> InputControlResult:
        """Press a hotkey combination (``ctrl, s``, ``alt, tab``, etc.).

        Keys are pressed in order then released in reverse.

        Ordering note (T3, catalog 07): ``pyautogui.hotkey`` returns
        BEFORE the target window has processed the keystroke. For
        sequences where the next action depends on the key landing
        (e.g., Alt+F4 then confirm-dialog Enter), callers must add
        an explicit short ``time.sleep`` or a UIA structure-change
        wait between calls. The synchronous-wait semantic that
        PowerShell's ``SendKeys.SendWait`` provides is not present
        here; use :meth:`type_text` for ordered text + key sequences
        where ordering matters, or sleep between calls.
        """
        if not keys:
            return InputControlResult(
                success=False, action="press_hotkey", error="no keys",
            )
        gate = self._gate(
            action="press_hotkey",
            arguments={"keys": list(keys)},
            user_text=user_text,
        )
        if gate is not None:
            return gate
        try:
            pyautogui.hotkey(*keys)
        except Exception as e:  # noqa: BLE001
            return InputControlResult(
                success=False, action="press_hotkey", error=str(e)[:200],
            )
        return InputControlResult(success=True, action="press_hotkey")

    def drag_to(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        *,
        button: str = "left",
        duration_s: float = 0.5,
        user_text: str = "",
    ) -> InputControlResult:
        """Drag from ``(x1, y1)`` to ``(x2, y2)`` with smooth animation.

        Catalog 08 T8 (YELLOW). The only pyautogui primitive missing
        from this controller pre-port. Designed for drag-to-reorder
        list items, drag-and-drop file moves between windows, drag-to-
        select rectangles in image editors -- any operation where the
        target ends up moved by a sustained pointer-down + motion.

        Goes through the same gate stack as :meth:`click` and
        :meth:`move_mouse`:

        1. **Foreground security check** -- refuses when a Windows
           security dialog (UAC, credential prompt) holds focus.
        2. **Rate limit** -- counts against the per-second budget.
        3. **Safety validator** -- runs the Cap-3 input rules with
           tool_name ``desktop.input.drag_to`` and coordinates +
           button + duration as arguments.
        4. **Click-preview gate** -- when ``click_preview_enabled``,
           previews the SOURCE coordinate ``(x1, y1)`` via the same
           VLM-confirmation path as :meth:`click`. The catalog flags
           drag as inherently irreversible (dragging a file to Trash,
           rearranging tabs, list reordering), so confirming the
           source pixel is the right safety contract; the destination
           is implied by the source confirmation.

        Implementation uses :func:`pyautogui.moveTo` to position the
        cursor at the source absolutely, then :func:`pyautogui.dragTo`
        to drag to the absolute destination. Absolute-coord variants
        avoid the relative-offset drift that bit the upstream plugin
        when the cursor moved between argparse and the drag call.

        Args:
            x1, y1: source coordinates (absolute physical pixels).
            x2, y2: destination coordinates (absolute physical pixels).
            button: ``"left"`` / ``"right"`` / ``"middle"``.
            duration_s: animation duration in seconds. Default 0.5 s
                matches the upstream pyautogui smoothness profile;
                shorten to 0.05 - 0.1 s for snappy applications.
            user_text: forwarded to the safety validator so the Cap-3
                explicit-intent matcher can verify the user actually
                asked for a drag.

        Returns:
            :class:`InputControlResult` with action ``"drag_to"``.
        """
        # Anticheat-safe mode: hard-blocked while the user is in game.
        from ultron.safety.anticheat import guard as _anticheat_guard
        _anticheat_guard('drag_to')

        if button not in ("left", "right", "middle"):
            return InputControlResult(
                success=False, action="drag_to",
                error=f"unknown button {button!r}",
            )
        if duration_s < 0:
            return InputControlResult(
                success=False, action="drag_to",
                error=f"duration_s must be non-negative, got {duration_s}",
            )

        args = {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "button": button,
            "duration_s": float(duration_s),
        }

        gate = self._gate(
            action="drag_to", arguments=args, user_text=user_text,
        )
        if gate is not None:
            return gate

        # Preview the SOURCE coordinate. Drag is bound by where you
        # pick up from (the file icon, the slider knob, the tab); the
        # destination is implied. Matching the click() pattern for
        # consistency.
        preview_block = self._maybe_preview_click(
            x=x1, y=y1, user_text=user_text,
        )
        if preview_block is not None:
            return InputControlResult(
                success=False,
                action="drag_to",
                error=preview_block.error,
            )

        try:
            pyautogui.moveTo(int(x1), int(y1))
            pyautogui.dragTo(
                int(x2), int(y2),
                duration=max(0.0, float(duration_s)),
                button=button,
            )
        except Exception as e:  # noqa: BLE001
            return InputControlResult(
                success=False, action="drag_to", error=str(e)[:200],
            )
        return InputControlResult(success=True, action="drag_to")

    def scroll(
        self,
        amount: int,
        *,
        direction: str = "vertical",
        x: Optional[int] = None,
        y: Optional[int] = None,
        user_text: str = "",
    ) -> InputControlResult:
        """Scroll the wheel at ``(x, y)`` or current cursor location.

        Catalog 09 T1 (YELLOW): supports both vertical and horizontal
        axis via the ``direction`` kwarg. ``direction="vertical"`` (the
        default, back-compat) dispatches to :func:`pyautogui.scroll`;
        ``direction="horizontal"`` dispatches to
        :func:`pyautogui.hscroll`. Catalog T5 browser-content
        extraction can read static UIA text but cannot scroll the
        browser to load lazy content -- T1 scroll closes that gap.

        ``amount`` is in OS-specific scroll units (typically ~120 per
        notch). Positive scrolls up (vertical) or left (horizontal);
        negative scrolls down or right, following pyautogui's
        convention. Unlike the upstream plugin which silently falls
        through unknown direction strings to ``hscroll``, ultron
        rejects unknown values with a structured error.
        """
        # Anticheat-safe mode: hard-blocked while the user is in game.
        from ultron.safety.anticheat import guard as _anticheat_guard
        _anticheat_guard('scroll')
        if direction not in ("vertical", "horizontal"):
            return InputControlResult(
                success=False, action="scroll",
                error=f"unknown direction {direction!r}; "
                      "must be 'vertical' or 'horizontal'",
            )
        args: dict = {"amount": int(amount), "direction": direction}
        if x is not None:
            args["x"] = int(x)
        if y is not None:
            args["y"] = int(y)
        gate = self._gate(
            action="scroll", arguments=args, user_text=user_text,
        )
        if gate is not None:
            return gate
        try:
            if direction == "horizontal":
                # pyautogui.hscroll positions at (x, y) when both are
                # provided; matches the vertical-scroll calling
                # convention. The upstream plugin moves the cursor
                # first with moveTo and then scrolls -- pyautogui
                # already does the equivalent via the x/y kwargs.
                pyautogui.hscroll(
                    int(amount),
                    x=int(x) if x is not None else None,
                    y=int(y) if y is not None else None,
                )
            else:
                pyautogui.scroll(
                    int(amount),
                    x=int(x) if x is not None else None,
                    y=int(y) if y is not None else None,
                )
        except Exception as e:  # noqa: BLE001
            return InputControlResult(
                success=False, action="scroll", error=str(e)[:200],
            )
        return InputControlResult(success=True, action="scroll")


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


_controller_singleton: Optional[InputController] = None


def get_input_controller() -> InputController:
    """Module-level singleton accessor."""
    global _controller_singleton
    if _controller_singleton is None:
        _controller_singleton = InputController()
    return _controller_singleton


def set_input_controller(controller: Optional[InputController]) -> None:
    """Test / orchestrator hook -- swap the singleton."""
    global _controller_singleton
    _controller_singleton = controller


__all__ = [
    "InputControlResult",
    "InputController",
    "get_input_controller",
    "set_input_controller",
]
