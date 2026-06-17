"""Tiny always-on-top "STOP" control window (in-process tkinter).

A manual, mouse-clickable kill switch for EVERY output channel -- it fires the
same :meth:`_cancel_all_playback` that voice "Ultron, stop" does, but driven by
a button click instead of the wake-word watcher.

Why a button at all: the wake watcher self-triggers on the monitor-speaker
loopback -- it hears Ultron's own audio as a wake word and barge-in-cancels
every line -- so it is held OFF. This window gives a reliable, loopback-immune
way to cut playback on demand.

Design notes:
  * In-process, exactly like the waveform overlay (``kenning/audio/waveform.py``):
    the Tk root lives in a dedicated daemon thread, so the button command can
    call the cancel callback DIRECTLY -- no IPC, no signal file, no polling, it
    is instant.
  * A button click is an ordinary window message to our OWN window. It is NOT
    input monitoring (no global keyboard hook and no system-wide key-state
    polling), so it adds NOTHING to the anticheat surface -- unlike a global
    hotkey would.
  * Borderless + always-on-top + fully black; summon/dismiss by voice.

Fail-open throughout: no display / no Tk -> the window simply never appears and
the voice path is untouched. Show/hide are idempotent and safe to call from the
voice loop; the button command runs on the Tk thread.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Callable, Optional

logger = logging.getLogger("kenning.audio.stop_button")

__all__ = ["StopButtonOverlay", "match_stop_button_command"]


# ---------------------------------------------------------------------------
# Voice matcher -- "show / hide the stop button" (and a few natural aliases).
# Strict: a sentence that merely mentions a stop button never matches.
# ---------------------------------------------------------------------------
_BUTTON_WORDS = (
    r"(?:stop\s+(?:button|panel|control|switch|window)"
    r"|panic\s+button|kill\s+switch|stop\s+sign)"
)
_OPEN_RE = re.compile(
    r"^(?:please\s+)?"
    r"(?:open|show(?:\s+me)?|pull\s+up|bring\s+up|give\s+me|launch|summon|put\s+up)"
    rf"\s+(?:the\s+|your\s+|my\s+|a\s+)?{_BUTTON_WORDS}\s*[.!?]?$",
    re.IGNORECASE,
)
_CLOSE_RE = re.compile(
    r"^(?:please\s+)?"
    r"(?:close|hide|dismiss|get\s+rid\s+of|take\s+down|put\s+away)"
    rf"\s+(?:the\s+|your\s+|my\s+)?{_BUTTON_WORDS}\s*[.!?]?$",
    re.IGNORECASE,
)


def match_stop_button_command(text: str) -> Optional[str]:
    """Match the strict show/hide stop-button phrasings.

    Args:
        text: the user's transcript for this turn.

    Returns:
        ``"open"`` / ``"close"`` / None. Ordinary sentences that merely mention
        a stop button never match.
    """
    if not text:
        return None
    cleaned = text.strip()
    if _OPEN_RE.match(cleaned):
        return "open"
    if _CLOSE_RE.match(cleaned):
        return "close"
    return None


class StopButtonOverlay:
    """Daemon-backed tiny STOP window. One per process.

    ``show`` / ``hide`` are idempotent and thread-safe (called from the voice
    loop); the button command runs on the Tk thread and calls ``on_stop``
    directly. Build-on-show / tear-down-on-hide mirrors the waveform overlay:
    ``overrideredirect`` windows don't reliably withdraw on Windows, so a fresh
    window is built each time it is summoned (cheap, and no stale-visibility
    ambiguity).
    """

    def __init__(
        self,
        on_stop: Callable[[], None],
        *,
        width: int = 120,
        bar_height: int = 16,
        button_height: int = 36,
        bg_color: str = "#000000",
        accent_color: str = "#e5484d",
        button_fill: str = "#140709",
        always_on_top: bool = True,
        label: str = "STOP",
        x: int = 60,
        y: int = 60,
        on_toggle_ptt: Optional[Callable[[bool], None]] = None,
        ptt_enabled: bool = True,
        ptt_height: int = 30,
    ) -> None:
        self._on_stop = on_stop
        self._width = max(72, int(width))
        self._bar_h = max(0, int(bar_height))
        self._btn_h = max(20, int(button_height))
        self._bg = bg_color or "#000000"
        self._accent = accent_color or "#e5484d"
        self._fill = button_fill or "#140709"
        self._always_on_top = bool(always_on_top)
        self._label = label or "STOP"
        self._x = int(x)
        self._y = int(y)
        # Optional PTT (push-to-talk) toggle row below the STOP button. When
        # wired, clicking it enables/disables Ultron's auto team-mic key-press
        # (the relay still plays either way -- this only stops Ultron from
        # holding the team-PTT key on every line). ``_ptt_enabled`` tracks the
        # displayed state across show/hide rebuilds.
        self._on_toggle_ptt = on_toggle_ptt
        self._ptt_enabled = bool(ptt_enabled)
        self._ptt_h = max(0, int(ptt_height))
        self._ui: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    @property
    def shown(self) -> bool:
        ui = self._ui
        return ui is not None and ui.is_alive()

    def show(self) -> None:
        """Build + raise the window. Idempotent (a no-op if already up)."""
        with self._lock:
            if self._ui is not None and self._ui.is_alive():
                return
            self._stop.clear()
            self._ui = threading.Thread(
                target=self._ui_loop, daemon=True, name="stop-button-ui")
            self._ui.start()

    def hide(self) -> None:
        """Tear the window down. Idempotent."""
        self._teardown()

    def close(self) -> None:
        """Alias for :meth:`hide` -- used on orchestrator shutdown."""
        self._teardown()

    def _teardown(self) -> None:
        """Signal the UI thread to quit its mainloop and join it. NEVER blocks
        when called from the UI thread itself (the right-click / X path)."""
        self._stop.set()
        ui = self._ui
        if ui is not None and ui is not threading.current_thread():
            try:
                ui.join(timeout=2.5)
            except Exception:  # noqa: BLE001
                pass
            self._ui = None

    # -- the click ---------------------------------------------------------

    def _fire(self) -> None:
        """Button command: cut every output channel. Fail-open so a callback
        error never kills the window."""
        try:
            self._on_stop()
            logger.info("stop button clicked -> all playback cancelled")
        except Exception as e:  # noqa: BLE001
            logger.warning("stop button callback failed: %s", e)

    # -- the window --------------------------------------------------------

    def _ui_loop(self) -> None:
        """Own the Tk root and run the redraw/poll loop. Fail-open."""
        try:
            import tkinter as tk
        except Exception as e:  # noqa: BLE001
            logger.warning("stop button unavailable (no tkinter: %s)", e)
            return
        root = None
        try:
            w = self._width
            bar_h = self._bar_h
            btn_h = self._btn_h
            _has_ptt = self._on_toggle_ptt is not None and self._ptt_h > 0
            ptt_h = self._ptt_h if _has_ptt else 0
            height = bar_h + btn_h + ptt_h
            root = tk.Tk()
            root.title("ULTRON // STOP")
            root.geometry(f"{w}x{height}+{self._x}+{self._y}")
            root.configure(bg=self._bg)
            root.overrideredirect(True)  # borderless
            if self._always_on_top:
                root.wm_attributes("-topmost", True)

            # Fully black drag bar -- grab it to reposition the window (e.g. out
            # of an OBS capture region). No visible marks: "fully black bar".
            if bar_h > 0:
                bar = tk.Frame(root, bg=self._bg, height=bar_h, width=w)
                bar.pack(fill="x", side="top")
                bar.pack_propagate(False)

                drag = {"x": 0, "y": 0}

                def _press(e):
                    drag["x"], drag["y"] = e.x, e.y

                def _move(e):
                    root.geometry(f"+{root.winfo_x() + e.x - drag['x']}"
                                  f"+{root.winfo_y() + e.y - drag['y']}")
                bar.bind("<Button-1>", _press)
                bar.bind("<B1-Motion>", _move)
                bar.bind("<Button-3>", lambda _e: self.hide())

            # PTT toggle -- packed at the BOTTOM (so STOP fills the middle).
            # Green "PTT ON" = Ultron auto-holds the team-mic key for relays;
            # grey "PTT OFF" = relay still plays but he never presses the key.
            if _has_ptt:
                _on_fg, _on_fill = "#3ddc84", "#0c1f13"      # green = ON
                _off_fg, _off_fill = "#8a8f98", "#141414"    # grey = OFF

                def _ptt_colors():
                    return ((_on_fg, _on_fill) if self._ptt_enabled
                            else (_off_fg, _off_fill))

                _fg0, _fill0 = _ptt_colors()
                ptt_btn = tk.Button(
                    root, text=f"PTT {'ON' if self._ptt_enabled else 'OFF'}",
                    bg=_fill0, fg=_fg0,
                    activebackground=_fill0, activeforeground="#ffffff",
                    relief="flat", bd=0, highlightthickness=1,
                    highlightbackground=_fg0, highlightcolor=_fg0,
                    font=("Segoe UI Semibold", 9), cursor="hand2",
                )

                def _toggle_ptt():
                    self._ptt_enabled = not self._ptt_enabled
                    fg, fill = _ptt_colors()
                    try:
                        ptt_btn.configure(
                            text=f"PTT {'ON' if self._ptt_enabled else 'OFF'}",
                            bg=fill, fg=fg, activebackground=fill,
                            highlightbackground=fg, highlightcolor=fg)
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        if self._on_toggle_ptt is not None:
                            self._on_toggle_ptt(self._ptt_enabled)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("ptt toggle callback failed: %s", e)
                    logger.info("ptt toggle -> %s",
                                "ON" if self._ptt_enabled else "OFF")

                ptt_btn.configure(command=_toggle_ptt)
                ptt_btn.pack(fill="x", side="bottom")
                ptt_btn.bind("<Button-3>", lambda _e: self.hide())

            # The STOP button -- the only visible element: red text + a 1px red
            # border on a near-black face, brightening while hovered/pressed.
            btn = tk.Button(
                root, text=f"■ {self._label}",
                command=self._fire,
                bg=self._fill, fg=self._accent,
                activebackground=self._accent, activeforeground="#ffffff",
                relief="flat", bd=0, highlightthickness=1,
                highlightbackground=self._accent, highlightcolor=self._accent,
                font=("Segoe UI Semibold", 12), cursor="hand2",
            )
            btn.pack(fill="both", expand=True)
            btn.bind("<Button-3>", lambda _e: self.hide())

            def _enter(_e):
                try:
                    btn.configure(bg="#3a1115")
                except Exception:  # noqa: BLE001
                    pass

            def _leave(_e):
                try:
                    btn.configure(bg=self._fill)
                except Exception:  # noqa: BLE001
                    pass
            btn.bind("<Enter>", _enter)
            btn.bind("<Leave>", _leave)

            def _poll():
                if self._stop.is_set():
                    try:
                        root.quit()  # return out of mainloop; destroy below
                    except Exception:  # noqa: BLE001
                        pass
                    return
                root.after(120, _poll)

            root.after(120, _poll)
            logger.info("stop button window up (%dx%d)", w, height)
            try:
                root.mainloop()
            finally:
                # Tear the Tcl interpreter down ON THIS thread (the one that
                # created it), matching the waveform overlay, so process exit
                # never triggers 'Tcl_AsyncDelete: ... wrong thread'.
                try:
                    root.destroy()
                except Exception:  # noqa: BLE001
                    pass
                root = None
                import gc
                gc.collect()
        except Exception as e:  # noqa: BLE001
            logger.warning("stop button window stopped (%s)", e)
            try:
                if root is not None:
                    root.destroy()
            except Exception:  # noqa: BLE001
                pass
