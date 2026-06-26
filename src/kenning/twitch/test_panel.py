"""Dev TEST PANEL window for the Twitch integration (in-process tkinter).

A streamer-facing developer window that fires a SYNTHETIC event for every
testable Ultron Twitch feature -- a speak redeem, a raid, each chat-command game,
each channel-point redeem game, and a chat-reply -- exactly AS IF a real viewer
(or raider) had done it, but with NO real viewer, raid, redeem or network. The
streamer clicks a button (filling a text box where the feature takes input) and a
single injected callback ``on_test(action, **params)`` is invoked; the
orchestrator wires that callback to synthetic-event injection (it mints a
synthetic chat / redeem / raid event and feeds it straight into the existing
redeem router / chat-game router / raid handler / chat-reply path).

Design (mirrors ``kenning/twitch/moderation_gui.py`` +
``kenning/audio/stop_button.py`` exactly):
  * In-process: the Tk root lives in a dedicated daemon thread that owns the
    mainloop, so a button command calls ``on_test`` DIRECTLY -- no IPC, no
    polling. Cross-thread ``show`` / ``hide`` requests are marshalled onto the Tk
    thread via an ``after``-driven request queue (a cross-thread ``after`` would
    corrupt the Tcl interpreter).
  * Always-on-top (``-topmost``) so it floats over the stream / game; resizable,
    with a font rescale on ``<Configure>`` so the whole panel reorganizes to fit.
  * A button click is an ordinary window message to our OWN window -- it is NOT
    input monitoring, so it adds NOTHING to the anticheat surface.
  * Fail-open throughout: no display / no Tk -> ``available`` is False and every
    method is a graceful no-op. Construction and every method NEVER raise into a
    boot or a pytest run.

Fully DECOUPLED: the panel never touches a router, the model, a socket or Helix.
It only validates its own text boxes and calls ``on_test``. All backend behavior
(minting + injecting the synthetic event) is the orchestrator's job.

The ``on_test`` callback -- the single backend seam
---------------------------------------------------
::

    on_test(action: str, *, message: str = "", login: str = "",
            viewers: int = 0, bet: int = 0, target: str = "",
            amount: int = 0, command: str = "") -> None

Every control reads + validates its text boxes, then invokes ``on_test`` with the
action string and ONLY the params that action carries (all other params keep
their documented defaults). The complete action catalogue the orchestrator wires:

SPEAK REDEEMS (synthetic ``redeem`` event -> redeem_router speak path)
  * ``"speak_say"``   params: ``message``       -- the "ultron says" stream redeem
  * ``"speak_team"``  params: ``message``       -- the "speak to my team" redeem

RAID (synthetic ``raid`` event -> RaidHandler)
  * ``"raid"``        params: ``login`` (raider), ``viewers`` (int >= 0)

CHAT-COMMAND GAMES (synthetic ``chat`` event "!<cmd> ..." -> chat-game router)
  * ``"chat_slots"``       params: ``command="slots"``,  ``bet`` (int > 0)
  * ``"chat_wheel"``       params: ``command="wheel"``
  * ``"chat_heist"``       params: ``command="heist"``,  ``bet`` (int > 0)
  * ``"chat_duel"``        params: ``command="duel"``,   ``target`` (login), ``bet`` (int > 0)
  * ``"chat_give"``        params: ``command="give"``,   ``target`` (login), ``amount`` (int > 0)
  * ``"chat_leaderboard"`` params: ``command="leaderboard"``
  * ``"chat_trivia"``      params: ``command="trivia"``
  * ``"chat_raffle"``      params: ``command="raffle"``
  * ``"chat_ultron"``      params: ``command="ultron"``
  * ``"chat_help"``        params: ``command="help"``

CHANNEL-POINT REDEEM GAMES (synthetic ``redeem`` event -> redeem_router game path)
  * ``"redeem_wheel"``  (no params)  -- the "spin the wheel" reward
  * ``"redeem_slots"``  (no params)  -- the "slots" reward
  * ``"redeem_heist"``  (no params)  -- the "heist" reward
  * ``"redeem_duel"``   (no params)  -- the "duel" reward

CHAT-REPLY (synthetic ``chat`` event "Ultron, <message>" -> chat-reply path)
  * ``"chat_reply"``  params: ``message``

EXTRAS (optional convenience triggers)
  * ``"auto_trivia"``     (no params)  -- kick a timed auto-trivia round
  * ``"commands_panel"``  (no params)  -- post the condensed commands panel
  * ``"talk_hint"``       (no params)  -- post the "talk to Ultron" hint

``viewers`` / ``bet`` / ``amount`` are already-parsed ints (a blank / non-numeric
box is a validated no-op -- ``on_test`` is NOT called and an inline status line
explains why). ``login`` / ``target`` are stripped of a leading ``@`` and
lowercased. ``message`` is stripped; an empty message no-ops a speak / reply.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
from collections.abc import Callable

logger = logging.getLogger("kenning.twitch.test_panel")

__all__ = ["TestPanel", "make_test_panel", "match_test_panel_command"]


# ---------------------------------------------------------------------------
# Voice matcher -- "show / open the test panel" (and "close it"). Strict: a short
# command that names the test/dev panel AND leads with an open/close verb. Mirrors
# ``kenning.audio.log_viewer.match_logs_command``.
# ---------------------------------------------------------------------------
_TEST_PANEL_RE = re.compile(
    r"\b(?:test|dev|developer)\s+panel\b",
    re.IGNORECASE,
)
_TP_OPEN_KW_RE = re.compile(
    r"\b(?:show|pull\s+up|open|bring\s+up|display|let\s+me\s+see|see|view|"
    r"give\s+me|pop\s+up)\b",
    re.IGNORECASE,
)
_TP_CLOSE_KW_RE = re.compile(
    r"\b(?:close|hide|dismiss|get\s+rid|take\s+down|put\s+away|go\s+away|"
    r"remove|minimi[sz]e|stash)\b",
    re.IGNORECASE,
)
_TP_NONCOMMAND_LEAD_RE = re.compile(
    r"^\s*(?:where|what'?s?|how|why|who|when|which|is|are|was|were|does|do|did|"
    r"has|have|i|we|he|she|they|that|this)\b",
    re.IGNORECASE,
)


def match_test_panel_command(text):
    """Match "show / open the test panel" (and "close the test panel").

    Returns ``"open"`` / ``"close"`` / None. Must reference the test/dev panel AND
    lead with an open/close verb, short command (>8 words = not a command).
    Questions / narration fall through. Never raises."""
    if not text:
        return None
    cleaned = str(text).strip()
    if _TEST_PANEL_RE.search(cleaned) is None or len(cleaned.split()) > 8:
        return None
    if _TP_NONCOMMAND_LEAD_RE.match(cleaned):
        return None
    if _TP_CLOSE_KW_RE.search(cleaned):
        return "close"
    if _TP_OPEN_KW_RE.search(cleaned):
        return "open"
    return None

# Force the fail-open / no-window path regardless of an available display. Used
# by lean/headless boots, CI, and tests (where building real Tk roots is both
# pointless and -- with rapid multi-root create/destroy on Windows -- a source of
# Tcl interpreter teardown faults). Any truthy value engages it. A dedicated flag
# (analogous to ``KENNING_MOD_GUI_HEADLESS``) so the test panel can be forced off
# independently of the moderation GUI.
_HEADLESS_ENV = "KENNING_TEST_PANEL_HEADLESS"
# Also honour the moderation GUI's flag so one "headless GUIs" switch silences
# every in-process Tk window at once.
_MOD_HEADLESS_ENV = "KENNING_MOD_GUI_HEADLESS"


def _headless_forced() -> bool:
    for env in (_HEADLESS_ENV, _MOD_HEADLESS_ENV):
        val = os.environ.get(env, "")
        if val.strip().lower() not in ("", "0", "false", "no", "off"):
            return True
    return False


# --------------------------------------------------------------------------- #
# Action tokens -- the closed set ``on_test`` receives. Grouped by section.
# --------------------------------------------------------------------------- #
# SPEAK REDEEMS
ACT_SPEAK_SAY = "speak_say"
ACT_SPEAK_TEAM = "speak_team"
# RAID
ACT_RAID = "raid"
# CHAT-COMMAND GAMES (each carries command=<word>)
ACT_CHAT_SLOTS = "chat_slots"
ACT_CHAT_WHEEL = "chat_wheel"
ACT_CHAT_HEIST = "chat_heist"
ACT_CHAT_DUEL = "chat_duel"
ACT_CHAT_GIVE = "chat_give"
ACT_CHAT_LEADERBOARD = "chat_leaderboard"
ACT_CHAT_TRIVIA = "chat_trivia"
ACT_CHAT_RAFFLE = "chat_raffle"
ACT_CHAT_ULTRON = "chat_ultron"
ACT_CHAT_HELP = "chat_help"
# CHANNEL-POINT REDEEM GAMES
ACT_REDEEM_WHEEL = "redeem_wheel"
ACT_REDEEM_SLOTS = "redeem_slots"
ACT_REDEEM_HEIST = "redeem_heist"
ACT_REDEEM_DUEL = "redeem_duel"
# CHAT-REPLY
ACT_CHAT_REPLY = "chat_reply"
# EXTRAS
ACT_AUTO_TRIVIA = "auto_trivia"
ACT_COMMANDS_PANEL = "commands_panel"
ACT_TALK_HINT = "talk_hint"


class TestPanel:
    """Daemon-backed always-on-top, resizable dev TEST PANEL.

    Buttons (+ text boxes for inputs), grouped into labeled sections, each firing
    a SYNTHETIC event for one testable feature via the single injected
    ``on_test`` callback. See the module docstring for the full ``on_test``
    action catalogue.

    Mirrors :class:`kenning.twitch.moderation_gui.ModerationControlPanel`:
      * the Tk root lives in a dedicated daemon thread that owns the mainloop;
        button commands call ``on_test`` directly;
      * always-on-top + resizable (font rescale on ``<Configure>``);
      * fail-open: no display / no Tk -> ``available`` False, every method a
        graceful no-op; construction + every method NEVER raise into a boot or a
        pytest run;
      * cross-thread ``show`` / ``hide`` marshalled onto the Tk thread via an
        ``after``-driven request queue.

    The validation + emit logic (:meth:`_emit`, :meth:`_emit_*`) is PURE and
    Tk-independent -- it reads :attr:`_vars` (tk ``StringVar``s at runtime, or
    plain stored values in tests) -- so it is directly unit-testable without a
    real root.
    """

    # Tell pytest this is NOT a test class (the ``Test`` name prefix otherwise
    # triggers a PytestCollectionWarning when the package is collected).
    __test__ = False

    def __init__(
        self,
        on_test: Callable[..., None],
        *,
        available: bool = True,
        width: int = 360,
        height: int = 720,
        x: int = 320,
        y: int = 40,
        bg_color: str = "#0b0b0f",
        fg_color: str = "#e6e6ea",
        accent_color: str = "#bf7fff",
        section_color: str = "#8a8f98",
        speak_color: str = "#bf7fff",
        raid_color: str = "#33d6c7",
        game_color: str = "#3ddc84",
        redeem_color: str = "#e0a82e",
        reply_color: str = "#5b8cff",
        extra_color: str = "#8a8f98",
        field_bg: str = "#16161c",
        always_on_top: bool = True,
        title: str = "ULTRON // TWITCH TEST PANEL",
        default_bet: int = 100,
        default_viewers: int = 20,
    ) -> None:
        self._on_test = on_test if callable(on_test) else None
        self._width = max(280, int(width))
        self._height = max(420, int(height))
        self._x = int(x)
        self._y = int(y)
        self._bg = bg_color or "#0b0b0f"
        self._fg = fg_color or "#e6e6ea"
        self._accent = accent_color or "#bf7fff"
        self._section = section_color or "#8a8f98"
        self._speak_color = speak_color or "#bf7fff"
        self._raid_color = raid_color or "#33d6c7"
        self._game_color = game_color or "#3ddc84"
        self._redeem_color = redeem_color or "#e0a82e"
        self._reply_color = reply_color or "#5b8cff"
        self._extra_color = extra_color or "#8a8f98"
        self._field_bg = field_bg or "#16161c"
        self._always_on_top = bool(always_on_top)
        self._title = title or "ULTRON // TWITCH TEST PANEL"
        self._default_bet = self._coerce_int(default_bet, 100)
        self._default_viewers = self._coerce_int(default_viewers, 20)

        # ``available`` reflects whether a Tk display could be reached. The
        # caller's ``available`` flag lets a lean/headless boot force the
        # no-window path; otherwise we probe Tk (which also honours the headless
        # env). It is flipped False the first time construction fails.
        self.available: bool = bool(available) and self._probe_tk_available()

        # UI-thread machinery (identical pattern to ModerationControlPanel).
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
        # only on the UI thread (tests stub the dict directly).
        self._vars: dict = {}
        self._status_text: str = ""

    # -- introspection -----------------------------------------------------

    @staticmethod
    def _probe_tk_available() -> bool:
        """True iff a Tk window may be built. False under a headless env flag or
        when ``tkinter`` is unimportable. Mirrors
        :meth:`ModerationControlPanel._probe_tk_available` -- no root is built
        here (deferred to the UI thread)."""
        if _headless_forced():
            logger.info("twitch test panel forced headless via env")
            return False
        try:
            import tkinter  # noqa: F401
        except Exception as e:  # noqa: BLE001
            logger.info("twitch test panel unavailable (no tkinter: %s)", e)
            return False
        return True

    @staticmethod
    def _coerce_int(value, default: int) -> int:
        """Best-effort int coercion (for constructor defaults). Never raises; a
        bad value falls back to ``default``."""
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
            logger.warning("test panel show failed (fail-open): %s", e)
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
            logger.warning("test panel hide failed (fail-open): %s", e)

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
            logger.warning("test panel toggle failed (fail-open): %s", e)

    def close(self) -> None:
        """Tear the panel + UI thread down. Used on orchestrator shutdown.
        Fail-open; never blocks when called from the UI thread itself."""
        self._stop.set()
        self._wake_ui()
        ui = self._ui
        if ui is not None and ui is not threading.current_thread():
            try:
                ui.join(timeout=2.5)
            except Exception:  # noqa: BLE001
                pass
            self._ui = None

    # -- command emit (PURE -- directly unit-testable, no Tk required) ------

    def _emit(self, action: str, **params) -> None:
        """Invoke the injected ``on_test`` exactly once with ``action`` + only the
        params that action carries. Fail-open: a missing / throwing callback never
        propagates out of a click. UI-thread-agnostic (click handlers call this;
        tests drive it directly)."""
        cb = self._on_test
        if cb is None:
            logger.info("test panel: no on_test wired; %s dropped", action)
            return
        try:
            cb(action, **params)
            logger.info("test panel -> %s (%s)", action, params)
        except Exception as e:  # noqa: BLE001 - a bad backend never kills the panel
            logger.warning("test panel on_test(%s) failed: %s", action, e)

    # --- speak redeems ---

    def _emit_speak(self, action: str) -> bool:
        """Validate + fire a SPEAK redeem (``speak_say`` / ``speak_team``). Reads
        its message box; an empty message is a validated no-op. Returns True when
        fired. Never raises."""
        try:
            key = "say_message" if action == ACT_SPEAK_SAY else "team_message"
            message = self._read_var(key).strip()
            if not message:
                self._set_status(f"{self._label_for(action)}: enter a message")
                return False
            self._emit(action, message=message)
            self._set_status(f"{self._label_for(action)} fired")
            return True
        except Exception as e:  # noqa: BLE001 - validation is fail-open
            logger.warning("test panel speak %s failed: %s", action, e)
            return False

    # --- raid ---

    def _emit_raid(self) -> bool:
        """Validate + fire a synthetic raid. Reads the raider-login + viewer-count
        boxes; a blank login no-ops; a blank / junk viewer box falls back to the
        default. Returns True when fired. Never raises."""
        try:
            login = self._read_login("raid_login")
            if not login:
                self._set_status("Raid: enter a raider login")
                return False
            viewers = self._read_int("raid_viewers", default=self._default_viewers)
            if viewers < 0:
                viewers = 0
            self._emit(ACT_RAID, login=login, viewers=viewers)
            self._set_status(f"Raid: {login} ({viewers} viewers)")
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("test panel raid failed: %s", e)
            return False

    # --- chat-command games ---

    def _emit_chat_bet(self, action: str, command: str) -> bool:
        """Validate + fire a single-bet chat game (``!slots`` / ``!heist``). Reads
        its bet box (must parse to > 0). Returns True when fired. Never raises."""
        try:
            bet = self._read_int(f"{command}_bet", default=self._default_bet)
            if bet <= 0:
                self._set_status(f"!{command}: bet must be > 0")
                return False
            self._emit(action, command=command, bet=bet)
            self._set_status(f"!{command} {bet}")
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("test panel chat-bet %s failed: %s", action, e)
            return False

    def _emit_chat_duel(self) -> bool:
        """Validate + fire ``!duel @target <bet>``. Needs a non-empty target and a
        bet > 0. Returns True when fired. Never raises."""
        try:
            target = self._read_login("duel_target")
            if not target:
                self._set_status("!duel: enter a target")
                return False
            bet = self._read_int("duel_bet", default=self._default_bet)
            if bet <= 0:
                self._set_status("!duel: bet must be > 0")
                return False
            self._emit(ACT_CHAT_DUEL, command="duel", target=target, bet=bet)
            self._set_status(f"!duel @{target} {bet}")
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("test panel duel failed: %s", e)
            return False

    def _emit_chat_give(self) -> bool:
        """Validate + fire ``!give @target <amount>``. Needs a non-empty target and
        an amount > 0. Returns True when fired. Never raises."""
        try:
            target = self._read_login("give_target")
            if not target:
                self._set_status("!give: enter a target")
                return False
            amount = self._read_int("give_amount", default=self._default_bet)
            if amount <= 0:
                self._set_status("!give: amount must be > 0")
                return False
            self._emit(ACT_CHAT_GIVE, command="give", target=target, amount=amount)
            self._set_status(f"!give @{target} {amount}")
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("test panel give failed: %s", e)
            return False

    def _emit_chat_simple(self, action: str, command: str) -> bool:
        """Fire a no-arg chat command (``!wheel`` / ``!leaderboard`` / ``!trivia``
        / ``!raffle`` / ``!ultron`` / ``!help``). Always fires. Never raises."""
        try:
            self._emit(action, command=command)
            self._set_status(f"!{command}")
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("test panel chat-simple %s failed: %s", action, e)
            return False

    # --- channel-point redeem games ---

    def _emit_redeem(self, action: str) -> bool:
        """Fire a channel-point redeem game (wheel / slots / heist / duel). No
        params -- the orchestrator mints the synthetic redeem with the matching
        reward title. Always fires. Never raises."""
        try:
            self._emit(action)
            self._set_status(f"{self._label_for(action)} redeem")
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("test panel redeem %s failed: %s", action, e)
            return False

    # --- chat-reply ---

    def _emit_chat_reply(self) -> bool:
        """Validate + fire a viewer chat message addressed to Ultron. Reads the
        message box; an empty message no-ops. Returns True when fired. Never
        raises."""
        try:
            message = self._read_var("reply_message").strip()
            if not message:
                self._set_status("Chat reply: enter a message")
                return False
            self._emit(ACT_CHAT_REPLY, message=message)
            self._set_status("Chat reply fired")
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("test panel chat-reply failed: %s", e)
            return False

    # --- extras ---

    def _emit_extra(self, action: str) -> bool:
        """Fire a no-param convenience trigger (auto-trivia / commands panel /
        talk hint). Always fires. Never raises."""
        try:
            self._emit(action)
            self._set_status(self._label_for(action))
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("test panel extra %s failed: %s", action, e)
            return False

    @staticmethod
    def _label_for(action: str) -> str:
        return {
            ACT_SPEAK_SAY: "Make Ultron Speak",
            ACT_SPEAK_TEAM: "Make Ultron Speak To My Team",
            ACT_RAID: "Simulate Raid",
            ACT_CHAT_SLOTS: "!slots",
            ACT_CHAT_WHEEL: "!wheel",
            ACT_CHAT_HEIST: "!heist",
            ACT_CHAT_DUEL: "!duel",
            ACT_CHAT_GIVE: "!give",
            ACT_CHAT_LEADERBOARD: "!leaderboard",
            ACT_CHAT_TRIVIA: "!trivia",
            ACT_CHAT_RAFFLE: "!raffle",
            ACT_CHAT_ULTRON: "!ultron",
            ACT_CHAT_HELP: "!help",
            ACT_REDEEM_WHEEL: "Spin the Wheel",
            ACT_REDEEM_SLOTS: "Slots",
            ACT_REDEEM_HEIST: "Heist",
            ACT_REDEEM_DUEL: "Duel",
            ACT_CHAT_REPLY: "Chat reply",
            ACT_AUTO_TRIVIA: "Auto-trivia round",
            ACT_COMMANDS_PANEL: "Commands panel",
            ACT_TALK_HINT: "Talk-to-Ultron hint",
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

    def _read_login(self, key: str) -> str:
        """Read a login / target box -> a normalised Twitch login (leading ``@``
        stripped, lowercased, whitespace trimmed). ``""`` when blank."""
        raw = self._read_var(key).strip()
        if not raw:
            return ""
        return raw.lstrip("@").strip().lower()

    def _read_int(self, key: str, *, default: int) -> int:
        """Read a numeric field, tolerating blank / non-numeric input. Pulls the
        leading integer out of the text (so "100c" / "20 viewers" still parse); a
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

    # -- UI thread lifecycle (mirrors ModerationControlPanel) -------------

    def _ensure_ui(self) -> None:
        if not self.available:
            return
        with self._lock:
            if self._ui is not None and self._ui.is_alive():
                return
            self._stop.clear()
            self._ui = threading.Thread(
                target=self._ui_loop, daemon=True, name="twitch-test-panel-ui")
            self._ui.start()

    def _wake_ui(self) -> None:
        """Intentionally a no-op -- see
        ``ModerationConfirmGUI._wake_ui``. The UI thread's own ``after``-driven
        ``_poll`` drains :attr:`_requests`; a cross-thread ``after`` would corrupt
        the Tcl interpreter ('Tcl_AsyncDelete: ... wrong thread')."""
        return

    def _ui_loop(self) -> None:
        """Own the Tk root + mainloop. Fail-open: any failure flips ``available``
        False and returns cleanly."""
        try:
            import tkinter as tk
        except Exception as e:  # noqa: BLE001
            logger.warning("twitch test panel: no tkinter (%s)", e)
            self.available = False
            return
        self._tk = tk
        root = None
        try:
            root = tk.Tk()
            self._root = root
            root.title(self._title)
            root.geometry(f"{self._width}x{self._height}+{self._x}+{self._y}")
            root.minsize(280, 420)
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
            logger.info("twitch test panel ready (%dx%d)",
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
                # (see ModerationControlPanel._ui_loop for the full rationale).
                self._fonts = {}
                import gc
                gc.collect()
        except Exception as e:  # noqa: BLE001
            logger.warning("twitch test panel stopped (%s)", e)
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
                logger.warning("test panel UI request failed: %s", e)

    # -- layout ------------------------------------------------------------

    def _build_layout(self, root) -> None:
        """Construct the panel grid. Sections (SPEAK / RAID / CHAT GAMES / REDEEM
        GAMES / CHAT REPLY / EXTRAS), each a labeled header followed by its
        button+box rows. Fonts rescale in :meth:`_on_configure`."""
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

        # --- SPEAK REDEEMS ---
        r = self._add_section(root, r, "SPEAK REDEEMS")
        r = self._add_button_field_row(
            root, r, "Make Ultron Speak", self._speak_color, "say_message",
            placeholder="message", command=lambda: self._emit_speak(ACT_SPEAK_SAY))
        r = self._add_button_field_row(
            root, r, "Speak To My Team", self._speak_color, "team_message",
            placeholder="message", command=lambda: self._emit_speak(ACT_SPEAK_TEAM))

        # --- RAID ---
        r = self._add_section(root, r, "RAID")
        r = self._add_raid_row(root, r)

        # --- CHAT-COMMAND GAMES ---
        r = self._add_section(root, r, "CHAT-COMMAND GAMES")
        r = self._add_button_field_row(
            root, r, "!slots", self._game_color, "slots_bet",
            placeholder="bet", default=str(self._default_bet),
            command=lambda: self._emit_chat_bet(ACT_CHAT_SLOTS, "slots"))
        r = self._add_button_field_row(
            root, r, "!heist", self._game_color, "heist_bet",
            placeholder="bet", default=str(self._default_bet),
            command=lambda: self._emit_chat_bet(ACT_CHAT_HEIST, "heist"))
        r = self._add_duel_row(root, r)
        r = self._add_give_row(root, r)
        r = self._add_button_only_row(
            root, r, "!wheel", self._game_color,
            lambda: self._emit_chat_simple(ACT_CHAT_WHEEL, "wheel"),
            tag="chat_wheel")
        r = self._add_button_only_row(
            root, r, "!leaderboard", self._game_color,
            lambda: self._emit_chat_simple(ACT_CHAT_LEADERBOARD, "leaderboard"),
            tag="chat_leaderboard")
        r = self._add_button_only_row(
            root, r, "!trivia", self._game_color,
            lambda: self._emit_chat_simple(ACT_CHAT_TRIVIA, "trivia"),
            tag="chat_trivia")
        r = self._add_button_only_row(
            root, r, "!raffle", self._game_color,
            lambda: self._emit_chat_simple(ACT_CHAT_RAFFLE, "raffle"),
            tag="chat_raffle")
        r = self._add_button_only_row(
            root, r, "!ultron", self._game_color,
            lambda: self._emit_chat_simple(ACT_CHAT_ULTRON, "ultron"),
            tag="chat_ultron")
        r = self._add_button_only_row(
            root, r, "!help", self._game_color,
            lambda: self._emit_chat_simple(ACT_CHAT_HELP, "help"),
            tag="chat_help")

        # --- CHANNEL-POINT REDEEM GAMES ---
        r = self._add_section(root, r, "CHANNEL-POINT REDEEM GAMES")
        r = self._add_button_only_row(
            root, r, "Spin the Wheel", self._redeem_color,
            lambda: self._emit_redeem(ACT_REDEEM_WHEEL), tag="redeem_wheel")
        r = self._add_button_only_row(
            root, r, "Slots", self._redeem_color,
            lambda: self._emit_redeem(ACT_REDEEM_SLOTS), tag="redeem_slots")
        r = self._add_button_only_row(
            root, r, "Heist", self._redeem_color,
            lambda: self._emit_redeem(ACT_REDEEM_HEIST), tag="redeem_heist")
        r = self._add_button_only_row(
            root, r, "Duel", self._redeem_color,
            lambda: self._emit_redeem(ACT_REDEEM_DUEL), tag="redeem_duel")

        # --- CHAT-REPLY ---
        r = self._add_section(root, r, "CHAT-REPLY")
        r = self._add_button_field_row(
            root, r, "Viewer -> Ultron", self._reply_color, "reply_message",
            placeholder="message", command=self._emit_chat_reply)

        # --- EXTRAS ---
        r = self._add_section(root, r, "EXTRAS")
        r = self._add_button_only_row(
            root, r, "Auto-trivia round", self._extra_color,
            lambda: self._emit_extra(ACT_AUTO_TRIVIA), tag="auto_trivia")
        r = self._add_button_only_row(
            root, r, "Post commands panel", self._extra_color,
            lambda: self._emit_extra(ACT_COMMANDS_PANEL), tag="commands_panel")
        r = self._add_button_only_row(
            root, r, "Talk-to-Ultron hint", self._extra_color,
            lambda: self._emit_extra(ACT_TALK_HINT), tag="talk_hint")

        # --- status line (absorbs the remaining height) ---
        status = tk.Label(
            root, text="", font=status_font, fg=self._section, bg=self._bg,
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
            root, text=label, font=self._fonts["section"], fg=self._section,
            bg=self._bg, anchor="w")
        lbl.grid(row=r, column=0, sticky="ew", padx=8, pady=(8, 1))
        root.grid_rowconfigure(r, weight=0)
        return r + 1

    def _add_button_only_row(self, root, r: int, label: str, color: str,
                             command, *, tag: str) -> int:
        """A single full-width button (no text box)."""
        tk = self._tk
        frame = tk.Frame(root, bg=self._bg)
        frame.grid(row=r, column=0, sticky="ew", padx=6, pady=2)
        root.grid_rowconfigure(r, weight=0)
        frame.grid_columnconfigure(0, weight=1)
        btn = self._make_button(frame, label, color, command, self._fonts["btn"])
        btn.grid(row=0, column=0, sticky="ew")
        self._widgets[f"{tag}_btn"] = btn
        return r + 1

    def _add_button_field_row(self, root, r: int, label: str, color: str,
                              var_key: str, *, placeholder: str = "",
                              default: str = "", command) -> int:
        """[ Button ] [ text entry ]. The entry feeds ``var_key`` in
        :attr:`_vars`."""
        tk = self._tk
        frame = tk.Frame(root, bg=self._bg)
        frame.grid(row=r, column=0, sticky="ew", padx=6, pady=2)
        root.grid_rowconfigure(r, weight=0)
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)

        var = tk.StringVar(master=root, value=default)
        self._vars[var_key] = var

        btn = self._make_button(frame, label, color, command, self._fonts["btn"])
        btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._widgets[f"{var_key}_btn"] = btn

        entry = self._make_entry(frame, var)
        entry.grid(row=0, column=1, sticky="ew")
        self._widgets[f"{var_key}_entry"] = entry
        if placeholder:
            self._apply_placeholder(entry, var, placeholder)
        return r + 1

    def _add_raid_row(self, root, r: int) -> int:
        """[ Simulate Raid ] [ raider login ] [ viewers ]."""
        tk = self._tk
        frame = tk.Frame(root, bg=self._bg)
        frame.grid(row=r, column=0, sticky="ew", padx=6, pady=2)
        root.grid_rowconfigure(r, weight=0)
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=0)

        login_var = tk.StringVar(master=root, value="")
        viewers_var = tk.StringVar(master=root, value=str(self._default_viewers))
        self._vars["raid_login"] = login_var
        self._vars["raid_viewers"] = viewers_var

        btn = self._make_button(
            frame, "Simulate Raid", self._raid_color, self._emit_raid,
            self._fonts["btn"])
        btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._widgets["raid_btn"] = btn

        login_entry = self._make_entry(frame, login_var)
        login_entry.grid(row=0, column=1, sticky="ew")
        self._widgets["raid_login_entry"] = login_entry
        self._apply_placeholder(login_entry, login_var, "raider login")

        viewers_entry = self._make_entry(frame, viewers_var, width=5)
        viewers_entry.grid(row=0, column=2, sticky="e", padx=(4, 0))
        self._widgets["raid_viewers_entry"] = viewers_entry
        return r + 1

    def _add_duel_row(self, root, r: int) -> int:
        """[ !duel ] [ target ] [ bet ]."""
        tk = self._tk
        frame = tk.Frame(root, bg=self._bg)
        frame.grid(row=r, column=0, sticky="ew", padx=6, pady=2)
        root.grid_rowconfigure(r, weight=0)
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=0)

        target_var = tk.StringVar(master=root, value="")
        bet_var = tk.StringVar(master=root, value=str(self._default_bet))
        self._vars["duel_target"] = target_var
        self._vars["duel_bet"] = bet_var

        btn = self._make_button(
            frame, "!duel", self._game_color, self._emit_chat_duel,
            self._fonts["btn"])
        btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._widgets["chat_duel_btn"] = btn

        target_entry = self._make_entry(frame, target_var)
        target_entry.grid(row=0, column=1, sticky="ew")
        self._widgets["duel_target_entry"] = target_entry
        self._apply_placeholder(target_entry, target_var, "target")

        bet_entry = self._make_entry(frame, bet_var, width=5)
        bet_entry.grid(row=0, column=2, sticky="e", padx=(4, 0))
        self._widgets["duel_bet_entry"] = bet_entry
        return r + 1

    def _add_give_row(self, root, r: int) -> int:
        """[ !give ] [ target ] [ amount ]."""
        tk = self._tk
        frame = tk.Frame(root, bg=self._bg)
        frame.grid(row=r, column=0, sticky="ew", padx=6, pady=2)
        root.grid_rowconfigure(r, weight=0)
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=0)

        target_var = tk.StringVar(master=root, value="")
        amount_var = tk.StringVar(master=root, value=str(self._default_bet))
        self._vars["give_target"] = target_var
        self._vars["give_amount"] = amount_var

        btn = self._make_button(
            frame, "!give", self._game_color, self._emit_chat_give,
            self._fonts["btn"])
        btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._widgets["chat_give_btn"] = btn

        target_entry = self._make_entry(frame, target_var)
        target_entry.grid(row=0, column=1, sticky="ew")
        self._widgets["give_target_entry"] = target_entry
        self._apply_placeholder(target_entry, target_var, "target")

        amount_entry = self._make_entry(frame, amount_var, width=5)
        amount_entry.grid(row=0, column=2, sticky="e", padx=(4, 0))
        self._widgets["give_amount_entry"] = amount_entry
        return r + 1

    def _make_button(self, parent, text, fg, command, font):
        tk = self._tk
        return tk.Button(
            parent, text=text, command=command,
            bg="#15151b", fg=fg, activebackground="#15151b",
            activeforeground="#ffffff", relief="flat", bd=0,
            highlightthickness=2, highlightbackground=fg, highlightcolor=fg,
            font=font, cursor="hand2")

    def _make_entry(self, parent, var, *, width: int = 0):
        tk = self._tk
        kwargs = dict(
            textvariable=var, font=self._fonts["entry"], bg=self._field_bg,
            fg=self._fg, insertbackground=self._fg, relief="flat",
            highlightthickness=1, highlightbackground="#33333a")
        if width:
            kwargs["width"] = width
        return tk.Entry(parent, **kwargs)

    def _apply_placeholder(self, entry, var, placeholder: str) -> None:
        """Show a greyed placeholder while the box is empty + unfocused. Purely
        cosmetic; ``_read_var`` still returns ``""`` for an untouched box because
        the placeholder is cleared on focus and never committed to the var.
        Fail-open: any binding fault leaves a plain (placeholder-less) box."""
        ghost = "#5a5e66"
        try:
            entry.configure(fg=ghost)
            entry.insert(0, placeholder)
            entry._placeholder_active = True  # type: ignore[attr-defined]

            def _on_focus_in(_e):
                if getattr(entry, "_placeholder_active", False):
                    try:
                        entry.delete(0, "end")
                        entry.configure(fg=self._fg)
                        entry._placeholder_active = False  # type: ignore[attr-defined]
                    except Exception:  # noqa: BLE001
                        pass

            def _on_focus_out(_e):
                try:
                    if not entry.get().strip():
                        entry.delete(0, "end")
                        entry.insert(0, placeholder)
                        entry.configure(fg=ghost)
                        entry._placeholder_active = True  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass

            entry.bind("<FocusIn>", _on_focus_in)
            entry.bind("<FocusOut>", _on_focus_out)
        except Exception as e:  # noqa: BLE001
            logger.debug("placeholder skipped: %s", e)

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
        """Rescale every font proportionally to the window size so the whole panel
        reorganizes + resizes to fit. UI thread only (fired by Tk). Mirrors
        ModerationControlPanel._on_configure."""
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
            logger.debug("test panel font rescale skipped: %s", e)


def make_test_panel(
    on_test: Callable[..., None],
    *,
    available: bool = True,
    **kwargs,
) -> TestPanel:
    """Factory the orchestrator can call to construct the test panel.

    Always returns a panel object (never raises): when Tk / a display is
    unavailable the returned panel's ``available`` is False and every method is a
    graceful no-op. ``on_test`` is the single backend seam (see :class:`TestPanel`
    and the module docstring for its signature + action catalogue).
    """
    try:
        return TestPanel(on_test, available=available, **kwargs)
    except Exception as e:  # noqa: BLE001 - construction is fail-open
        logger.warning("make_test_panel failed; returning inert panel (%s)", e)
        panel = TestPanel.__new__(TestPanel)
        panel.available = False
        panel._on_test = None
        panel._ui = None
        panel._requests = queue.Queue()
        panel._stop = threading.Event()
        panel._lock = threading.Lock()
        panel._vars = {}
        panel._widgets = {}
        return panel
