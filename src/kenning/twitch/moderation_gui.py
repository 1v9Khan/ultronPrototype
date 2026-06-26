"""Always-on-top confirmation window for a Twitch mod action (in-process tkinter).

When a moderator-issued action (timeout / ban / unban / untimeout / delete-last)
resolves a *fuzzy* username match, this little window puts the decision in front
of the human: it names the ACTION, shows the best-match USERNAME large and
prominent, lists the alternative candidates it considered, and offers three
explicit buttons -- YES (confirm), NO (reject this match, try again) and CANCEL
(abandon the action). The click drives a single ``on_result`` callback.

Design notes (mirrors ``kenning/audio/stop_button.py``):
  * In-process: the Tk root lives in a dedicated daemon thread that owns the
    mainloop, so the button command can call the result callback DIRECTLY -- no
    IPC, no polling.
  * A button click is an ordinary window message to our OWN window -- it is NOT
    input monitoring, so it adds nothing to the anticheat surface.
  * Always-on-top (``-topmost``) so it floats over the stream/game.
  * Unlike the borderless STOP control, this window is *resizable*: it uses a
    grid with row/column weights AND rescales every font on ``<Configure>`` so
    the header, the prominent username, the alternatives list and the three
    buttons reorganize and resize to fit the window at any size.

Fail-open throughout: no display / no Tk -> ``available`` is False and every
method is a graceful no-op. Construction and every method NEVER raise into a
boot or a pytest run. Cross-thread requests (``prompt`` / ``update_match`` /
``hide`` called from the moderation loop) are marshalled onto the Tk thread via
``after`` so all widget mutation happens on the thread that owns the root.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
from collections.abc import Callable

logger = logging.getLogger("kenning.twitch.moderation_gui")

__all__ = [
    "ModerationConfirmGUI",
    "ModerationControlPanel",
    "make_control_panel",
    "match_moderation_panel_command",
]


# ---------------------------------------------------------------------------
# Voice matcher -- "open / show the moderation panel" (and "close it").
# Strict: a short command that names the moderation/mod panel AND leads with an
# open/close verb. A long sentence or a question that merely mentions it never
# matches (it falls through to the LLM). Mirrors
# ``kenning.audio.log_viewer.match_logs_command``.
# ---------------------------------------------------------------------------
_MOD_PANEL_RE = re.compile(
    r"\b(?:moderation|mod)\s+(?:control\s+)?panel\b",
    re.IGNORECASE,
)
_MOD_OPEN_KW_RE = re.compile(
    r"\b(?:show|pull\s+up|open|bring\s+up|display|let\s+me\s+see|see|view|"
    r"give\s+me|pop\s+up)\b",
    re.IGNORECASE,
)
_MOD_CLOSE_KW_RE = re.compile(
    r"\b(?:close|hide|dismiss|get\s+rid|take\s+down|put\s+away|go\s+away|"
    r"remove|minimi[sz]e|stash)\b",
    re.IGNORECASE,
)
_MOD_NONCOMMAND_LEAD_RE = re.compile(
    r"^\s*(?:where|what'?s?|how|why|who|when|which|is|are|was|were|does|do|did|"
    r"has|have|i|we|he|she|they|that|this)\b",
    re.IGNORECASE,
)


def match_moderation_panel_command(text):
    """Match "open / show the moderation panel" (and "close the moderation panel").

    Returns ``"open"`` / ``"close"`` / None. Must reference the moderation/mod
    panel AND lead with an open/close verb, and be a short command (>8 words = not
    a command). Questions / narration fall through. Never raises."""
    if not text:
        return None
    cleaned = str(text).strip()
    if _MOD_PANEL_RE.search(cleaned) is None or len(cleaned.split()) > 8:
        return None
    if _MOD_NONCOMMAND_LEAD_RE.match(cleaned):
        return None
    if _MOD_CLOSE_KW_RE.search(cleaned):
        return "close"
    if _MOD_OPEN_KW_RE.search(cleaned):
        return "open"
    return None

# Force the fail-open / no-window path regardless of an available display. Used
# by lean/headless boots, CI, and tests (where building real Tk roots is both
# pointless and -- with rapid multi-root create/destroy on Windows -- a source
# of Tcl interpreter teardown faults). Any truthy value engages it.
_HEADLESS_ENV = "KENNING_MOD_GUI_HEADLESS"


def _headless_forced() -> bool:
    val = os.environ.get(_HEADLESS_ENV, "")
    return val.strip().lower() not in ("", "0", "false", "no", "off")


# Result tokens the YES / NO / CANCEL buttons emit.
_RESULT_YES = "yes"
_RESULT_NO = "no"
_RESULT_CANCEL = "cancel"
_VALID_RESULTS = (_RESULT_YES, _RESULT_NO, _RESULT_CANCEL)


class ModerationConfirmGUI:
    """Daemon-backed always-on-top, resizable mod-action confirm window.

    One per process. ``prompt`` shows (building the window lazily on first use)
    and (re)populates the window, wiring ``on_result`` to the three buttons.
    ``update_match`` refreshes the displayed username + alternatives after a
    re-search. ``hide`` lowers the window. ``available`` is False when Tk / a
    display is missing, in which case every method is a safe no-op.

    The Tk root is created on the UI thread on first ``prompt``; all subsequent
    widget mutation is marshalled onto that thread via a request queue drained
    by an ``after`` poll, so the public API is safe to call from the moderation
    loop thread.
    """

    def __init__(
        self,
        *,
        width: int = 380,
        height: int = 280,
        x: int = 80,
        y: int = 80,
        bg_color: str = "#0b0b0f",
        fg_color: str = "#e6e6ea",
        accent_color: str = "#bf7fff",
        yes_color: str = "#3ddc84",
        no_color: str = "#e0a82e",
        cancel_color: str = "#ff6b6b",
        always_on_top: bool = True,
        title: str = "ULTRON // CONFIRM MOD ACTION",
    ) -> None:
        self._width = max(220, int(width))
        self._height = max(160, int(height))
        self._x = int(x)
        self._y = int(y)
        self._bg = bg_color or "#0b0b0f"
        self._fg = fg_color or "#e6e6ea"
        self._accent = accent_color or "#bf7fff"
        self._yes_color = yes_color or "#3ddc84"
        self._no_color = no_color or "#e0a82e"
        self._cancel_color = cancel_color or "#ff6b6b"
        self._always_on_top = bool(always_on_top)
        self._title = title or "ULTRON // CONFIRM MOD ACTION"

        # ``available`` reflects whether a Tk display could be reached. It starts
        # optimistically True and is flipped False the first time Tk import or
        # window construction fails -- after which every method short-circuits.
        self.available: bool = self._probe_tk_available()

        # Current logical state, mutated only on the UI thread.
        self._action: str = ""
        self._username: str = ""
        self._alternatives: list[str] = []
        self._on_result: Callable[[str], None] | None = None
        # Guards against a double-fire (two clicks before the window hides).
        self._result_sent: bool = False

        # UI-thread machinery.
        self._ui: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._requests: queue.Queue[Callable[[], None]] = queue.Queue()
        # Populated on the UI thread; only touched there.
        self._tk = None
        self._root = None
        self._widgets: dict = {}

    # -- introspection -----------------------------------------------------

    @staticmethod
    def _probe_tk_available() -> bool:
        """True iff a Tk window may be built. False when the headless env flag
        is set or ``tkinter`` fails to import. We do NOT build a root here (that
        is deferred to the UI thread) -- a missing import is the cheap, common
        headless signal; a missing *display* is caught later and also flips
        ``available`` False."""
        if _headless_forced():
            logger.info("moderation confirm GUI forced headless via %s",
                        _HEADLESS_ENV)
            return False
        try:
            import tkinter  # noqa: F401
        except Exception as e:  # noqa: BLE001
            logger.info("moderation confirm GUI unavailable (no tkinter: %s)", e)
            return False
        return True

    @property
    def shown(self) -> bool:
        ui = self._ui
        return ui is not None and ui.is_alive()

    # -- public API --------------------------------------------------------

    def prompt(
        self,
        action: str,
        username: str,
        alternatives: list[str],
        on_result: Callable[[str], None],
    ) -> None:
        """Show / update the window for a pending mod action.

        Args:
            action: the action header, e.g. ``"TIMEOUT 10m"`` / ``"BAN"`` /
                ``"UNBAN"`` / ``"UNTIMEOUT"`` / ``"DELETE LAST MSG"``.
            username: the best-match username (shown large + prominent).
            alternatives: other candidate usernames the matcher considered.
            on_result: invoked exactly once with ``"yes"`` / ``"no"`` /
                ``"cancel"`` on the matching button click.

        Fail-open: a no-op when Tk / a display is unavailable.
        """
        if not self.available:
            return
        try:
            action_s = str(action or "")
            username_s = str(username or "")
            alts = [str(a) for a in (alternatives or []) if str(a)]
            cb = on_result if callable(on_result) else None

            def _apply() -> None:
                self._action = action_s
                self._username = username_s
                self._alternatives = alts
                self._on_result = cb
                self._result_sent = False
                self._render()
                self._raise_window()

            self._ensure_ui()
            self._requests.put(_apply)
            self._wake_ui()
        except Exception as e:  # noqa: BLE001
            logger.warning("moderation confirm prompt failed (fail-open): %s", e)
            self.available = False

    def update_match(self, username: str, alternatives: list[str]) -> None:
        """Refresh the displayed match after a re-search (NO click -> retry).

        Fail-open: a no-op when unavailable or no window is up.
        """
        if not self.available:
            return
        try:
            username_s = str(username or "")
            alts = [str(a) for a in (alternatives or []) if str(a)]

            def _apply() -> None:
                self._username = username_s
                self._alternatives = alts
                # A fresh candidate set means the prior result window is "live"
                # again -- allow a result to be sent for this new match.
                self._result_sent = False
                self._render()
                self._raise_window()

            self._requests.put(_apply)
            self._wake_ui()
        except Exception as e:  # noqa: BLE001
            logger.warning("moderation confirm update failed (fail-open): %s", e)

    def hide(self) -> None:
        """Withdraw the window (idempotent). Fail-open."""
        if not self.available:
            return
        try:
            def _apply() -> None:
                self._withdraw_window()

            self._requests.put(_apply)
            self._wake_ui()
        except Exception as e:  # noqa: BLE001
            logger.warning("moderation confirm hide failed (fail-open): %s", e)

    def close(self) -> None:
        """Tear the window + UI thread down. Used on orchestrator shutdown.

        Fail-open, and never blocks when called from the UI thread itself.
        """
        self._stop.set()
        self._wake_ui()
        ui = self._ui
        if ui is not None and ui is not threading.current_thread():
            try:
                ui.join(timeout=2.5)
            except Exception:  # noqa: BLE001
                pass
            self._ui = None

    # -- UI thread lifecycle ----------------------------------------------

    def _ensure_ui(self) -> None:
        """Start the UI thread on first use. Idempotent + thread-safe."""
        if not self.available:
            return
        with self._lock:
            if self._ui is not None and self._ui.is_alive():
                return
            self._stop.clear()
            self._ui = threading.Thread(
                target=self._ui_loop, daemon=True, name="mod-confirm-ui")
            self._ui.start()

    def _wake_ui(self) -> None:
        """Intentionally a no-op.

        Tk objects may be touched ONLY from the thread that created the root
        (mirroring ``stop_button.py``). Calling ``root.after``/``after_idle``
        from the moderation-loop thread corrupts the Tcl interpreter
        ('Tcl_AsyncDelete: ... wrong thread'). Instead the UI thread's own
        ``after``-driven ``_poll`` drains :attr:`_requests` every ~80 ms, so a
        cross-thread nudge is neither needed nor safe. Kept as a named seam so
        the public methods read clearly.
        """
        return

    def _ui_loop(self) -> None:
        """Own the Tk root + mainloop. Fail-open: any failure flips
        ``available`` False and returns cleanly."""
        try:
            import tkinter as tk
        except Exception as e:  # noqa: BLE001
            logger.warning("moderation confirm GUI: no tkinter (%s)", e)
            self.available = False
            return
        self._tk = tk
        root = None
        try:
            root = tk.Tk()
            self._root = root
            root.title(self._title)
            root.geometry(
                f"{self._width}x{self._height}+{self._x}+{self._y}")
            root.minsize(220, 160)
            root.configure(bg=self._bg)
            if self._always_on_top:
                try:
                    root.wm_attributes("-topmost", True)
                except Exception:  # noqa: BLE001
                    pass

            self._build_layout(root)

            # Start withdrawn; a prompt() raises it.
            try:
                root.withdraw()
            except Exception:  # noqa: BLE001
                pass

            # Rescale fonts whenever the window resizes so every element fits.
            root.bind("<Configure>", self._on_configure)

            def _poll() -> None:
                if self._stop.is_set():
                    try:
                        root.quit()
                    except Exception:  # noqa: BLE001
                        pass
                    return
                self._drain_requests()
                try:
                    root.after(80, _poll)
                except Exception:  # noqa: BLE001
                    pass

            try:
                root.after(80, _poll)
            except Exception:  # noqa: BLE001
                pass
            logger.info("moderation confirm GUI ready (%dx%d)",
                        self._width, self._height)
            try:
                root.mainloop()
            finally:
                try:
                    root.destroy()
                except Exception:  # noqa: BLE001
                    pass
                self._root = None
                self._widgets = {}
                # Release the tkfont.Font handles on THIS (UI) thread. A Font
                # garbage-collected on any OTHER thread raises
                # 'Tcl_AsyncDelete: ... wrong thread' and crashes the process
                # (exit 3) -- the same wrong-thread Tcl hazard _wake_ui documents.
                # The finally clears self._widgets but historically left
                # self._fonts holding the Font objects, so they were freed later
                # on whichever thread dropped the last reference. Dropping the refs
                # HERE + gc.collect() finalizes them on the thread that owns the
                # Tcl interpreter. (root is already destroyed, so each Font.__del__
                # hits a dead interpreter, which tkinter.font.Font.__del__ catches.)
                self._fonts = {}
                import gc
                gc.collect()
        except Exception as e:  # noqa: BLE001
            logger.warning("moderation confirm GUI stopped (%s)", e)
            self.available = False
            try:
                if root is not None:
                    root.destroy()
            except Exception:  # noqa: BLE001
                pass
            self._root = None

    def _drain_requests(self) -> None:
        """Run all queued UI mutations on the Tk thread. Fail-open per item."""
        while True:
            try:
                fn = self._requests.get_nowait()
            except queue.Empty:
                return
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                logger.warning("moderation confirm UI request failed: %s", e)

    # -- layout (built once, repopulated per prompt) -----------------------

    def _build_layout(self, root) -> None:
        """Construct the grid layout. Row/column weights make every region
        flex; fonts are (re)scaled in :meth:`_on_configure`."""
        tk = self._tk
        # Track the tk.font handles so <Configure> can rescale them.
        import tkinter.font as tkfont

        header_font = tkfont.Font(family="Segoe UI Semibold", size=12)
        user_font = tkfont.Font(family="Segoe UI", size=22, weight="bold")
        alt_header_font = tkfont.Font(family="Segoe UI", size=8)
        alt_font = tkfont.Font(family="Consolas", size=9)
        btn_font = tkfont.Font(family="Segoe UI Semibold", size=11)

        self._fonts = {
            "header": header_font,
            "user": user_font,
            "alt_header": alt_header_font,
            "alt": alt_font,
            "btn": btn_font,
        }
        # Base sizes (px proxy) used to scale fonts proportionally on resize.
        self._base_fonts = {
            "header": 12, "user": 22, "alt_header": 8, "alt": 9, "btn": 11,
        }

        # 4 logical rows: header / username / alternatives / buttons.
        # Weighted so the username + alternatives regions absorb extra height.
        root.grid_rowconfigure(0, weight=0)   # header
        root.grid_rowconfigure(1, weight=3)   # username (dominant)
        root.grid_rowconfigure(2, weight=2)   # alternatives
        root.grid_rowconfigure(3, weight=0)   # buttons
        root.grid_columnconfigure(0, weight=1)

        # Header -- the ACTION.
        header = tk.Label(
            root, text="", font=header_font, fg=self._accent, bg=self._bg,
            anchor="center", justify="center", wraplength=self._width,
        )
        header.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 2))

        # Username -- large + prominent.
        user = tk.Label(
            root, text="", font=user_font, fg=self._fg, bg=self._bg,
            anchor="center", justify="center", wraplength=self._width,
        )
        user.grid(row=1, column=0, sticky="nsew", padx=8, pady=2)

        # Alternatives -- a smaller secondary region.
        alt_frame = tk.Frame(root, bg=self._bg)
        alt_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=2)
        alt_frame.grid_columnconfigure(0, weight=1)
        alt_frame.grid_rowconfigure(0, weight=0)
        alt_frame.grid_rowconfigure(1, weight=1)

        alt_header = tk.Label(
            alt_frame, text="alternatives", font=alt_header_font,
            fg="#8a8f98", bg=self._bg, anchor="w",
        )
        alt_header.grid(row=0, column=0, sticky="ew")

        alt_list = tk.Label(
            alt_frame, text="", font=alt_font, fg="#b9bcc6", bg=self._bg,
            anchor="nw", justify="left", wraplength=self._width,
        )
        alt_list.grid(row=1, column=0, sticky="nsew")

        # Buttons -- YES / NO / CANCEL across a 3-column sub-grid.
        btn_frame = tk.Frame(root, bg=self._bg)
        btn_frame.grid(row=3, column=0, sticky="nsew", padx=6, pady=(2, 8))
        for c in range(3):
            btn_frame.grid_columnconfigure(c, weight=1, uniform="btns")
        btn_frame.grid_rowconfigure(0, weight=1)

        yes_btn = self._make_button(
            btn_frame, "YES", self._yes_color, "#0c1f13",
            lambda: self._fire(_RESULT_YES), btn_font)
        no_btn = self._make_button(
            btn_frame, "NO", self._no_color, "#1a160a",
            lambda: self._fire(_RESULT_NO), btn_font)
        cancel_btn = self._make_button(
            btn_frame, "CANCEL", self._cancel_color, "#1a0d0d",
            lambda: self._fire(_RESULT_CANCEL), btn_font)
        yes_btn.grid(row=0, column=0, sticky="nsew", padx=3)
        no_btn.grid(row=0, column=1, sticky="nsew", padx=3)
        cancel_btn.grid(row=0, column=2, sticky="nsew", padx=3)

        self._widgets = {
            "header": header,
            "user": user,
            "alt_header": alt_header,
            "alt_list": alt_list,
            "yes": yes_btn,
            "no": no_btn,
            "cancel": cancel_btn,
        }
        # Allow Esc to cancel.
        try:
            root.bind("<Escape>", lambda _e: self._fire(_RESULT_CANCEL))
        except Exception:  # noqa: BLE001
            pass

    def _make_button(self, parent, text, fg, fill, command, font):
        tk = self._tk
        b = tk.Button(
            parent, text=text, command=command,
            bg=fill, fg=fg, activebackground=fill, activeforeground="#ffffff",
            relief="flat", bd=0, highlightthickness=2,
            highlightbackground=fg, highlightcolor=fg,
            font=font, cursor="hand2",
        )
        return b

    # -- render / window-state (UI thread only) ---------------------------

    def _render(self) -> None:
        """Push current logical state into the widgets. UI thread only."""
        w = self._widgets
        if not w:
            return
        try:
            w["header"].configure(text=self._action or "MOD ACTION")
            w["user"].configure(text=self._username or "(no match)")
            if self._alternatives:
                w["alt_header"].configure(text="alternatives")
                w["alt_list"].configure(
                    text="\n".join(self._alternatives))
            else:
                w["alt_header"].configure(text="no other candidates")
                w["alt_list"].configure(text="")
        except Exception as e:  # noqa: BLE001
            logger.warning("moderation confirm render failed: %s", e)

    def _raise_window(self) -> None:
        root = self._root
        if root is None:
            return
        try:
            root.deiconify()
            if self._always_on_top:
                root.wm_attributes("-topmost", True)
            root.lift()
        except Exception:  # noqa: BLE001
            pass

    def _withdraw_window(self) -> None:
        root = self._root
        if root is None:
            return
        try:
            root.withdraw()
        except Exception:  # noqa: BLE001
            pass

    def _on_configure(self, event) -> None:
        """Rescale every font proportionally to the window size so the header,
        username, alternatives and buttons reorganize + resize to fit. UI
        thread only (fired by Tk)."""
        root = self._root
        if root is None or not getattr(self, "_fonts", None):
            return
        # Only react to the toplevel's own resize, not child <Configure>s.
        try:
            if event is not None and event.widget is not root:
                return
        except Exception:  # noqa: BLE001
            pass
        try:
            w = max(1, int(root.winfo_width()))
            h = max(1, int(root.winfo_height()))
        except Exception:  # noqa: BLE001
            return
        # Scale by the smaller of the width/height ratios against the base
        # geometry, clamped so text stays legible and never explodes.
        try:
            scale_w = w / float(self._width)
            scale_h = h / float(self._height)
            scale = min(scale_w, scale_h)
            scale = max(0.55, min(2.6, scale))
            for key, base in self._base_fonts.items():
                size = max(6, int(round(base * scale)))
                self._fonts[key].configure(size=size)
            # Keep wraplength in step so long names/alternatives wrap.
            wrap = max(80, w - 24)
            for wk in ("header", "user"):
                if wk in self._widgets:
                    self._widgets[wk].configure(wraplength=wrap)
            if "alt_list" in self._widgets:
                self._widgets["alt_list"].configure(wraplength=wrap)
        except Exception as e:  # noqa: BLE001
            logger.debug("moderation confirm font rescale skipped: %s", e)

    # -- the click ---------------------------------------------------------

    def _fire(self, result: str) -> None:
        """Button command: emit the result ONCE, then withdraw. Fail-open so a
        bad callback never kills the window. UI thread only."""
        if result not in _VALID_RESULTS:
            return
        if self._result_sent:
            return
        self._result_sent = True
        cb = self._on_result
        try:
            self._withdraw_window()
        except Exception:  # noqa: BLE001
            pass
        if cb is None:
            return
        try:
            cb(result)
            logger.info("moderation confirm -> %s (%s / %s)",
                        result, self._action, self._username)
        except Exception as e:  # noqa: BLE001
            logger.warning("moderation confirm result callback failed: %s", e)


# ===========================================================================
# ModerationControlPanel -- the always-on-top click-to-moderate sidebar.
# ===========================================================================

# The action tokens the panel emits on ``on_command``. They mirror the
# moderation service's verbs / chat-settings so the orchestrator can map them
# straight through. USER-targeted vs CHANNEL-wide is documented per token.
#
# USER-targeted (carry ``user=``; ``ban``/``unban``/``untimeout``/``delete``
# ignore ``seconds``; ``timeout`` carries ``seconds``):
_ACT_BAN = "ban"
_ACT_TIMEOUT = "timeout"
_ACT_UNBAN = "unban"
_ACT_UNTIMEOUT = "untimeout"
_ACT_DELETE = "delete_message"
# CHANNEL-wide (no ``user``; carry ``enabled=`` and, where relevant, ``seconds=``):
_ACT_CLEAR_CHAT = "clear_chat"
_ACT_SLOW = "slow_mode"            # enabled + seconds (wait time)
_ACT_FOLLOWERS = "followers_only"  # enabled + seconds (minutes-as-duration; 0 = any)
_ACT_SUBSCRIBERS = "subscribers_only"  # enabled
_ACT_EMOTE = "emote_only"          # enabled
_ACT_UNIQUE = "unique_chat"        # enabled

_USER_TARGETED_ACTIONS = frozenset(
    {_ACT_BAN, _ACT_TIMEOUT, _ACT_UNBAN, _ACT_UNTIMEOUT, _ACT_DELETE}
)
_CHANNEL_ACTIONS = frozenset(
    {
        _ACT_CLEAR_CHAT,
        _ACT_SLOW,
        _ACT_FOLLOWERS,
        _ACT_SUBSCRIBERS,
        _ACT_EMOTE,
        _ACT_UNIQUE,
    }
)

_DEFAULT_TIMEOUT_SECONDS = 600  # mirrors ModerationService._DEFAULT_TIMEOUT_S
_DEFAULT_SLOW_SECONDS = 30
_DEFAULT_FOLLOWERS_MINUTES = 0  # 0 => "any follower" (no minimum age)


class ModerationControlPanel:
    """Daemon-backed always-on-top, resizable click-to-moderate control panel.

    A vertical sidebar of buttons -- one per moderation command -- so the
    streamer can CLICK to moderate alongside voice. User-targeted commands pair
    each button with a username text box (timeout adds a small seconds field);
    channel-wide toggles pair an ON / OFF pair (slow / followers add a numeric
    field). Every control validates its inputs then invokes a SINGLE injected
    callback::

        on_command(action: str, *, user: str = "", seconds: int = 0,
                   enabled: bool = True) -> None

    The panel is fully DECOUPLED from the backend -- it never touches Helix, a
    socket, or the model (anticheat-clean; a click is an ordinary message to our
    OWN window). The orchestrator wires ``on_command`` to the
    :class:`~kenning.twitch.moderation.service.ModerationService` actions.

    Design mirrors :class:`ModerationConfirmGUI` exactly:
      * The Tk root lives in a dedicated daemon thread that owns the mainloop;
        button commands call the injected callback DIRECTLY.
      * Always-on-top (``-topmost``) so it floats over the stream / game.
      * Resizable: a grid with row/column weights AND a font rescale on
        ``<Configure>`` so the whole sidebar reorganizes to fit any size.
      * Fail-open throughout: no display / no Tk -> ``available`` is False and
        every method is a graceful no-op. Construction + every method NEVER
        raise into a boot or a pytest run.
      * Cross-thread ``show`` / ``hide`` requests are marshalled onto the Tk
        thread via the same ``after``-driven request queue.

    The ``ModerationConfirmGUI`` stays a separate, composable piece: the panel
    can OPTIONALLY own one (pass ``with_confirm=True`` or inject ``confirm=``)
    and expose it via :attr:`confirm`, but the two never entangle -- the
    fuzzy-match confirm flow is wired independently by the orchestrator.
    """

    def __init__(
        self,
        on_command: Callable[..., None],
        *,
        available: bool = True,
        width: int = 260,
        height: int = 560,
        x: int = 40,
        y: int = 40,
        bg_color: str = "#0b0b0f",
        fg_color: str = "#e6e6ea",
        accent_color: str = "#bf7fff",
        danger_color: str = "#ff6b6b",
        warn_color: str = "#e0a82e",
        ok_color: str = "#3ddc84",
        field_bg: str = "#16161c",
        always_on_top: bool = True,
        title: str = "ULTRON // MOD PANEL",
        default_timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        default_slow_seconds: int = _DEFAULT_SLOW_SECONDS,
        default_followers_minutes: int = _DEFAULT_FOLLOWERS_MINUTES,
        with_confirm: bool = False,
        confirm: ModerationConfirmGUI | None = None,
    ) -> None:
        self._on_command = on_command if callable(on_command) else None
        self._width = max(200, int(width))
        self._height = max(320, int(height))
        self._x = int(x)
        self._y = int(y)
        self._bg = bg_color or "#0b0b0f"
        self._fg = fg_color or "#e6e6ea"
        self._accent = accent_color or "#bf7fff"
        self._danger = danger_color or "#ff6b6b"
        self._warn = warn_color or "#e0a82e"
        self._ok = ok_color or "#3ddc84"
        self._field_bg = field_bg or "#16161c"
        self._always_on_top = bool(always_on_top)
        self._title = title or "ULTRON // MOD PANEL"
        self._default_timeout = self._coerce_int(default_timeout_seconds, _DEFAULT_TIMEOUT_SECONDS)
        self._default_slow = self._coerce_int(default_slow_seconds, _DEFAULT_SLOW_SECONDS)
        self._default_followers = self._coerce_int(
            default_followers_minutes, _DEFAULT_FOLLOWERS_MINUTES)

        # ``available`` reflects whether a Tk display could be reached. The
        # caller's ``available`` flag lets a lean/headless boot force the
        # no-window path; otherwise we probe Tk (which also honours the headless
        # env). It is flipped False the first time construction fails.
        self.available: bool = bool(available) and self._probe_tk_available()

        # Optional composed confirm popup -- a SEPARATE piece (never entangled).
        if confirm is not None:
            self.confirm: ModerationConfirmGUI | None = confirm
        elif with_confirm:
            try:
                self.confirm = ModerationConfirmGUI(always_on_top=always_on_top)
            except Exception as e:  # noqa: BLE001 - confirm is optional; fail-open
                logger.warning("control panel: confirm popup unavailable (%s)", e)
                self.confirm = None
        else:
            self.confirm = None

        # UI-thread machinery (identical pattern to ModerationConfirmGUI).
        self._ui: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._requests: queue.Queue[Callable[[], None]] = queue.Queue()
        self._tk = None
        self._root = None
        self._widgets: dict = {}
        self._fonts: dict = {}
        self._base_fonts: dict = {}
        # Per-control tk string vars + the latest inline status message. Touched
        # only on the UI thread.
        self._vars: dict = {}
        self._status_text: str = ""

    # -- introspection -----------------------------------------------------

    @staticmethod
    def _probe_tk_available() -> bool:
        """True iff a Tk window may be built. False under the headless env flag
        or when ``tkinter`` is unimportable. Mirrors
        :meth:`ModerationConfirmGUI._probe_tk_available` -- no root is built
        here (deferred to the UI thread)."""
        if _headless_forced():
            logger.info("moderation control panel forced headless via %s",
                        _HEADLESS_ENV)
            return False
        try:
            import tkinter  # noqa: F401
        except Exception as e:  # noqa: BLE001
            logger.info("moderation control panel unavailable (no tkinter: %s)", e)
            return False
        return True

    @staticmethod
    def _coerce_int(value, default: int) -> int:
        """Best-effort int coercion (used for constructor defaults). Never
        raises; a bad value falls back to ``default``."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    @property
    def shown(self) -> bool:
        ui = self._ui
        return ui is not None and ui.is_alive()

    # -- public API --------------------------------------------------------

    def show(self) -> None:
        """Summon (build-on-first-use) the panel + raise it. Fail-open: a no-op
        when Tk / a display is unavailable."""
        if not self.available:
            return
        try:
            def _apply() -> None:
                self._raise_window()

            self._ensure_ui()
            self._requests.put(_apply)
            self._wake_ui()
        except Exception as e:  # noqa: BLE001
            logger.warning("control panel show failed (fail-open): %s", e)
            self.available = False

    def hide(self) -> None:
        """Withdraw the panel (idempotent). Fail-open."""
        if not self.available:
            return
        try:
            def _apply() -> None:
                self._withdraw_window()

            self._requests.put(_apply)
            self._wake_ui()
        except Exception as e:  # noqa: BLE001
            logger.warning("control panel hide failed (fail-open): %s", e)

    def toggle(self) -> None:
        """Show if hidden / hide if shown. Fail-open."""
        if not self.available:
            return
        try:
            def _apply() -> None:
                root = self._root
                if root is None:
                    self._raise_window()
                    return
                try:
                    state = root.state()
                except Exception:  # noqa: BLE001
                    state = "withdrawn"
                if state == "withdrawn":
                    self._raise_window()
                else:
                    self._withdraw_window()

            self._ensure_ui()
            self._requests.put(_apply)
            self._wake_ui()
        except Exception as e:  # noqa: BLE001
            logger.warning("control panel toggle failed (fail-open): %s", e)

    def close(self) -> None:
        """Tear the panel (and any owned confirm popup) + UI thread down. Used on
        orchestrator shutdown. Fail-open; never blocks when called from the UI
        thread itself."""
        self._stop.set()
        self._wake_ui()
        ui = self._ui
        if ui is not None and ui is not threading.current_thread():
            try:
                ui.join(timeout=2.5)
            except Exception:  # noqa: BLE001
                pass
            self._ui = None
        # The confirm popup is independently owned only when we created it.
        cfm = self.confirm
        if cfm is not None:
            try:
                cfm.close()
            except Exception:  # noqa: BLE001
                pass

    # -- command emit (PURE -- directly unit-testable, no Tk required) ------

    def _emit(self, action: str, *, user: str = "", seconds: int = 0,
              enabled: bool = True) -> None:
        """Invoke the injected ``on_command`` exactly once. Fail-open: a missing
        / throwing callback never propagates out of a click. UI-thread-agnostic
        (the click handlers call this; tests drive it directly)."""
        cb = self._on_command
        if cb is None:
            logger.info("control panel: no on_command wired; %s dropped", action)
            return
        try:
            cb(action, user=user, seconds=seconds, enabled=enabled)
            logger.info("control panel -> %s (user=%r seconds=%s enabled=%s)",
                        action, user, seconds, enabled)
        except Exception as e:  # noqa: BLE001 - a bad backend never kills the panel
            logger.warning("control panel on_command(%s) failed: %s", action, e)

    def _emit_user_action(self, action: str, *, default_seconds: int = 0) -> bool:
        """Validate + fire a USER-targeted action from the current field state.

        Reads the action's username box (and, for ``timeout``, its seconds box),
        validates a non-empty username, then calls :meth:`_emit`. Returns True
        when the command fired, False on a validation no-op (empty username /
        invalid seconds) -- in which case an inline status message is shown and
        NO callback is invoked. Never raises.
        """
        try:
            user = self._read_var(f"{action}_user").strip()
            if not user:
                self._set_status(f"{self._label_for(action)}: enter a username")
                return False
            seconds = 0
            if action == _ACT_TIMEOUT:
                seconds = self._read_int(
                    f"{action}_seconds", default=default_seconds or self._default_timeout)
                if seconds <= 0:
                    self._set_status("Timeout: seconds must be > 0")
                    return False
            self._emit(action, user=user, seconds=seconds, enabled=True)
            self._set_status(f"{self._label_for(action)} -> {user}")
            return True
        except Exception as e:  # noqa: BLE001 - validation is fail-open
            logger.warning("control panel user-action %s failed: %s", action, e)
            return False

    def _emit_channel_action(self, action: str, *, enabled: bool) -> bool:
        """Validate + fire a CHANNEL-wide action (no user). ``slow_mode`` and
        ``followers_only`` pull their numeric field (seconds / minutes); when
        turning a mode OFF the field is ignored. ``clear_chat`` ignores
        ``enabled``. Returns True when fired (always True here -- channel actions
        have no per-target validation to fail). Never raises."""
        try:
            seconds = 0
            if enabled and action == _ACT_SLOW:
                seconds = self._read_int("slow_seconds", default=self._default_slow)
                if seconds <= 0:
                    seconds = self._default_slow
            elif enabled and action == _ACT_FOLLOWERS:
                # Followers-only duration is in MINUTES (0 = any follower). We
                # surface it as ``seconds`` on the callback BUT the value is a
                # minute count -- the orchestrator maps it to
                # ``follower_mode_duration`` (minutes). Documented on on_command.
                seconds = self._read_int("followers_minutes", default=self._default_followers)
                if seconds < 0:
                    seconds = 0
            self._emit(action, user="", seconds=seconds, enabled=enabled)
            state = "on" if enabled else "off"
            if action == _ACT_CLEAR_CHAT:
                self._set_status("Cleared chat")
            else:
                self._set_status(f"{self._label_for(action)} {state}")
            return True
        except Exception as e:  # noqa: BLE001 - fail-open
            logger.warning("control panel channel-action %s failed: %s", action, e)
            return False

    @staticmethod
    def _label_for(action: str) -> str:
        return {
            _ACT_BAN: "Ban",
            _ACT_TIMEOUT: "Timeout",
            _ACT_UNBAN: "Unban",
            _ACT_UNTIMEOUT: "Untimeout",
            _ACT_DELETE: "Delete message",
            _ACT_CLEAR_CHAT: "Clear chat",
            _ACT_SLOW: "Slow mode",
            _ACT_FOLLOWERS: "Followers-only",
            _ACT_SUBSCRIBERS: "Subscribers-only",
            _ACT_EMOTE: "Emote-only",
            _ACT_UNIQUE: "Unique-chat",
        }.get(action, action)

    # -- var access (UI thread; tests stub the dict) ----------------------

    def _read_var(self, key: str) -> str:
        """Read a tk StringVar's value (or a plain stored value in tests). Never
        raises -> ``""`` on any fault."""
        var = self._vars.get(key)
        if var is None:
            return ""
        try:
            getter = getattr(var, "get", None)
            return str(getter()) if callable(getter) else str(var)
        except Exception:  # noqa: BLE001
            return ""

    def _read_int(self, key: str, *, default: int) -> int:
        """Read a numeric field, tolerating blank / non-numeric input. Pulls the
        leading integer out of the text (so "600s" / "10 min" still parse); a
        blank field yields ``default``; junk yields ``default``."""
        raw = self._read_var(key).strip()
        if not raw:
            return int(default)
        m = re.search(r"-?\d+", raw)
        if not m:
            return int(default)
        try:
            return int(m.group(0))
        except (TypeError, ValueError):
            return int(default)

    def _set_status(self, text: str) -> None:
        """Set the inline status line (UI thread). Fail-open."""
        self._status_text = str(text or "")
        w = self._widgets.get("status")
        if w is None:
            return
        try:
            w.configure(text=self._status_text)
        except Exception:  # noqa: BLE001
            pass

    # -- UI thread lifecycle (mirrors ModerationConfirmGUI) ---------------

    def _ensure_ui(self) -> None:
        if not self.available:
            return
        with self._lock:
            if self._ui is not None and self._ui.is_alive():
                return
            self._stop.clear()
            self._ui = threading.Thread(
                target=self._ui_loop, daemon=True, name="mod-panel-ui")
            self._ui.start()

    def _wake_ui(self) -> None:
        """Intentionally a no-op -- see ModerationConfirmGUI._wake_ui. The UI
        thread's own ``after``-driven ``_poll`` drains :attr:`_requests`; a
        cross-thread ``after`` would corrupt the Tcl interpreter."""
        return

    def _ui_loop(self) -> None:
        """Own the Tk root + mainloop. Fail-open: any failure flips ``available``
        False and returns cleanly."""
        try:
            import tkinter as tk
        except Exception as e:  # noqa: BLE001
            logger.warning("moderation control panel: no tkinter (%s)", e)
            self.available = False
            return
        self._tk = tk
        root = None
        try:
            root = tk.Tk()
            self._root = root
            root.title(self._title)
            root.geometry(f"{self._width}x{self._height}+{self._x}+{self._y}")
            root.minsize(200, 320)
            root.configure(bg=self._bg)
            if self._always_on_top:
                try:
                    root.wm_attributes("-topmost", True)
                except Exception:  # noqa: BLE001
                    pass

            self._build_layout(root)

            try:
                root.withdraw()  # start hidden; show() raises it
            except Exception:  # noqa: BLE001
                pass

            root.bind("<Configure>", self._on_configure)

            def _poll() -> None:
                if self._stop.is_set():
                    try:
                        root.quit()
                    except Exception:  # noqa: BLE001
                        pass
                    return
                self._drain_requests()
                try:
                    root.after(80, _poll)
                except Exception:  # noqa: BLE001
                    pass

            try:
                root.after(80, _poll)
            except Exception:  # noqa: BLE001
                pass
            logger.info("moderation control panel ready (%dx%d)",
                        self._width, self._height)
            try:
                root.mainloop()
            finally:
                try:
                    root.destroy()
                except Exception:  # noqa: BLE001
                    pass
                self._root = None
                self._widgets = {}
                self._vars = {}
                # Release tkfont.Font handles on THIS (UI) thread -- a Font freed
                # on another thread raises 'Tcl_AsyncDelete: ... wrong thread'
                # (see ModerationConfirmGUI._ui_loop for the full rationale).
                self._fonts = {}
                import gc
                gc.collect()
        except Exception as e:  # noqa: BLE001
            logger.warning("moderation control panel stopped (%s)", e)
            self.available = False
            try:
                if root is not None:
                    root.destroy()
            except Exception:  # noqa: BLE001
                pass
            self._root = None

    def _drain_requests(self) -> None:
        while True:
            try:
                fn = self._requests.get_nowait()
            except queue.Empty:
                return
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                logger.warning("control panel UI request failed: %s", e)

    # -- layout ------------------------------------------------------------

    def _build_layout(self, root) -> None:
        """Construct the sidebar grid. Each USER row = [Button | username entry]
        (+ a seconds entry for timeout); each CHANNEL row = [ON | OFF] (+ a
        numeric entry for slow / followers). Fonts rescale in
        :meth:`_on_configure`."""
        tk = self._tk
        import tkinter.font as tkfont

        title_font = tkfont.Font(family="Segoe UI Semibold", size=12)
        section_font = tkfont.Font(family="Segoe UI", size=8)
        btn_font = tkfont.Font(family="Segoe UI Semibold", size=10)
        entry_font = tkfont.Font(family="Consolas", size=10)
        status_font = tkfont.Font(family="Segoe UI", size=8)
        self._fonts = {
            "title": title_font,
            "section": section_font,
            "btn": btn_font,
            "entry": entry_font,
            "status": status_font,
        }
        self._base_fonts = {
            "title": 12, "section": 8, "btn": 10, "entry": 10, "status": 8,
        }

        root.grid_columnconfigure(0, weight=1)
        r = 0

        title = tk.Label(
            root, text=self._title, font=title_font, fg=self._accent,
            bg=self._bg, anchor="center")
        title.grid(row=r, column=0, sticky="ew", padx=8, pady=(8, 4))
        root.grid_rowconfigure(r, weight=0)
        r += 1

        # --- USER-TARGETED section ---
        r = self._add_section(root, r, "VIEWER ACTIONS")
        r = self._add_user_row(root, r, _ACT_BAN, "Ban", self._danger)
        r = self._add_user_row(root, r, _ACT_TIMEOUT, "Timeout", self._warn,
                               with_seconds=True, seconds_default=self._default_timeout)
        r = self._add_user_row(root, r, _ACT_UNBAN, "Unban", self._ok)
        r = self._add_user_row(root, r, _ACT_UNTIMEOUT, "Untimeout", self._ok)
        r = self._add_user_row(root, r, _ACT_DELETE, "Delete last msg", self._warn)

        # --- CHANNEL-WIDE section ---
        r = self._add_section(root, r, "CHANNEL MODES")
        r = self._add_clear_row(root, r)
        r = self._add_toggle_row(root, r, _ACT_SLOW, "Slow mode",
                                 with_field=True, field_key="slow_seconds",
                                 field_default=self._default_slow, field_hint="sec")
        r = self._add_toggle_row(root, r, _ACT_FOLLOWERS, "Followers-only",
                                 with_field=True, field_key="followers_minutes",
                                 field_default=self._default_followers, field_hint="min")
        r = self._add_toggle_row(root, r, _ACT_SUBSCRIBERS, "Subscribers-only")
        r = self._add_toggle_row(root, r, _ACT_EMOTE, "Emote-only")
        r = self._add_toggle_row(root, r, _ACT_UNIQUE, "Unique-chat")

        # --- status line (absorbs the remaining height) ---
        status = tk.Label(
            root, text="", font=status_font, fg="#8a8f98", bg=self._bg,
            anchor="w", justify="left", wraplength=self._width)
        status.grid(row=r, column=0, sticky="sew", padx=8, pady=(6, 8))
        root.grid_rowconfigure(r, weight=1)
        self._widgets["status"] = status

        try:
            root.bind("<Escape>", lambda _e: self._withdraw_window())
        except Exception:  # noqa: BLE001
            pass

    def _add_section(self, root, r: int, label: str) -> int:
        tk = self._tk
        lbl = tk.Label(
            root, text=label, font=self._fonts["section"], fg="#8a8f98",
            bg=self._bg, anchor="w")
        lbl.grid(row=r, column=0, sticky="ew", padx=8, pady=(8, 1))
        root.grid_rowconfigure(r, weight=0)
        return r + 1

    def _add_user_row(self, root, r: int, action: str, label: str, color: str,
                      *, with_seconds: bool = False,
                      seconds_default: int = 0) -> int:
        """[ Button ] [ username entry ] (+ [seconds] for timeout)."""
        tk = self._tk
        frame = tk.Frame(root, bg=self._bg)
        frame.grid(row=r, column=0, sticky="ew", padx=6, pady=2)
        root.grid_rowconfigure(r, weight=0)
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)

        user_var = tk.StringVar(master=root, value="")
        self._vars[f"{action}_user"] = user_var

        default_secs = seconds_default
        btn = self._make_button(
            frame, label, color,
            lambda a=action, d=default_secs: self._emit_user_action(a, default_seconds=d),
            self._fonts["btn"])
        btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._widgets[f"{action}_btn"] = btn

        entry = tk.Entry(
            frame, textvariable=user_var, font=self._fonts["entry"],
            bg=self._field_bg, fg=self._fg, insertbackground=self._fg,
            relief="flat", highlightthickness=1, highlightbackground="#33333a")
        entry.grid(row=0, column=1, sticky="ew")
        self._widgets[f"{action}_user_entry"] = entry

        if with_seconds:
            secs_var = tk.StringVar(master=root, value=str(seconds_default))
            self._vars[f"{action}_seconds"] = secs_var
            frame.grid_columnconfigure(2, weight=0)
            secs_entry = tk.Entry(
                frame, textvariable=secs_var, font=self._fonts["entry"],
                width=5, bg=self._field_bg, fg=self._fg,
                insertbackground=self._fg, relief="flat",
                highlightthickness=1, highlightbackground="#33333a")
            secs_entry.grid(row=0, column=2, sticky="e", padx=(4, 0))
            self._widgets[f"{action}_seconds_entry"] = secs_entry
        return r + 1

    def _add_clear_row(self, root, r: int) -> int:
        """A single full-width Clear chat button (no toggle / field)."""
        tk = self._tk
        frame = tk.Frame(root, bg=self._bg)
        frame.grid(row=r, column=0, sticky="ew", padx=6, pady=2)
        root.grid_rowconfigure(r, weight=0)
        frame.grid_columnconfigure(0, weight=1)
        btn = self._make_button(
            frame, "Clear chat", self._danger,
            lambda: self._emit_channel_action(_ACT_CLEAR_CHAT, enabled=True),
            self._fonts["btn"])
        btn.grid(row=0, column=0, sticky="ew")
        self._widgets[f"{_ACT_CLEAR_CHAT}_btn"] = btn
        return r + 1

    def _add_toggle_row(self, root, r: int, action: str, label: str,
                        *, with_field: bool = False, field_key: str = "",
                        field_default: int = 0, field_hint: str = "") -> int:
        """[ label ] [ ON ] [ OFF ] (+ a numeric field for slow / followers)."""
        tk = self._tk
        frame = tk.Frame(root, bg=self._bg)
        frame.grid(row=r, column=0, sticky="ew", padx=6, pady=2)
        root.grid_rowconfigure(r, weight=0)
        frame.grid_columnconfigure(0, weight=1)  # label
        frame.grid_columnconfigure(1, weight=0)  # on
        frame.grid_columnconfigure(2, weight=0)  # off

        lbl = tk.Label(
            frame, text=label, font=self._fonts["btn"], fg=self._fg,
            bg=self._bg, anchor="w")
        lbl.grid(row=0, column=0, sticky="ew")

        on_btn = self._make_button(
            frame, "ON", self._ok,
            lambda a=action: self._emit_channel_action(a, enabled=True),
            self._fonts["btn"])
        on_btn.grid(row=0, column=1, sticky="e", padx=2)
        self._widgets[f"{action}_on_btn"] = on_btn

        off_btn = self._make_button(
            frame, "OFF", self._warn,
            lambda a=action: self._emit_channel_action(a, enabled=False),
            self._fonts["btn"])
        off_btn.grid(row=0, column=2, sticky="e", padx=2)
        self._widgets[f"{action}_off_btn"] = off_btn

        if with_field and field_key:
            field_var = tk.StringVar(master=root, value=str(field_default))
            self._vars[field_key] = field_var
            frame.grid_columnconfigure(3, weight=0)
            field_entry = tk.Entry(
                frame, textvariable=field_var, font=self._fonts["entry"],
                width=5, bg=self._field_bg, fg=self._fg,
                insertbackground=self._fg, relief="flat",
                highlightthickness=1, highlightbackground="#33333a")
            field_entry.grid(row=0, column=3, sticky="e", padx=(4, 0))
            self._widgets[f"{field_key}_entry"] = field_entry
        return r + 1

    def _make_button(self, parent, text, fg, command, font):
        tk = self._tk
        return tk.Button(
            parent, text=text, command=command,
            bg="#15151b", fg=fg, activebackground="#15151b",
            activeforeground="#ffffff", relief="flat", bd=0,
            highlightthickness=2, highlightbackground=fg, highlightcolor=fg,
            font=font, cursor="hand2")

    # -- window-state (UI thread only) ------------------------------------

    def _raise_window(self) -> None:
        root = self._root
        if root is None:
            return
        try:
            root.deiconify()
            if self._always_on_top:
                root.wm_attributes("-topmost", True)
            root.lift()
        except Exception:  # noqa: BLE001
            pass

    def _withdraw_window(self) -> None:
        root = self._root
        if root is None:
            return
        try:
            root.withdraw()
        except Exception:  # noqa: BLE001
            pass

    def _on_configure(self, event) -> None:
        """Rescale every font proportionally to the window size so the whole
        sidebar reorganizes + resizes to fit. UI thread only (fired by Tk).
        Mirrors ModerationConfirmGUI._on_configure."""
        root = self._root
        if root is None or not getattr(self, "_fonts", None):
            return
        try:
            if event is not None and event.widget is not root:
                return
        except Exception:  # noqa: BLE001
            pass
        try:
            w = max(1, int(root.winfo_width()))
            h = max(1, int(root.winfo_height()))
        except Exception:  # noqa: BLE001
            return
        try:
            scale_w = w / float(self._width)
            scale_h = h / float(self._height)
            scale = min(scale_w, scale_h)
            scale = max(0.6, min(2.2, scale))
            for key, base in self._base_fonts.items():
                size = max(6, int(round(base * scale)))
                self._fonts[key].configure(size=size)
            wrap = max(80, w - 24)
            if "status" in self._widgets:
                self._widgets["status"].configure(wraplength=wrap)
        except Exception as e:  # noqa: BLE001
            logger.debug("control panel font rescale skipped: %s", e)


def make_control_panel(
    on_command: Callable[..., None],
    *,
    available: bool = True,
    with_confirm: bool = False,
    **kwargs,
) -> ModerationControlPanel:
    """Factory the orchestrator can call to construct the panel.

    Always returns a panel object (never raises): when Tk / a display is
    unavailable the returned panel's ``available`` is False and every method is
    a graceful no-op. ``on_command`` is the single backend seam (see
    :class:`ModerationControlPanel` for its signature). Pass
    ``with_confirm=True`` to also own a :class:`ModerationConfirmGUI` exposed as
    ``panel.confirm`` for the fuzzy-match path.
    """
    try:
        return ModerationControlPanel(
            on_command, available=available, with_confirm=with_confirm, **kwargs)
    except Exception as e:  # noqa: BLE001 - construction is fail-open
        logger.warning("make_control_panel failed; returning inert panel (%s)", e)
        panel = ModerationControlPanel.__new__(ModerationControlPanel)
        panel.available = False
        panel._on_command = None
        panel.confirm = None
        panel._ui = None
        panel._requests = queue.Queue()
        panel._stop = threading.Event()
        panel._vars = {}
        panel._widgets = {}
        return panel
