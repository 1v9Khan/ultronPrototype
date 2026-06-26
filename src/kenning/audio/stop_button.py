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
# Keyword-based matching (robust to STT mangling): "stop button" / "kill switch"
# / "panic button" is an unambiguous phrase -- nothing in the game is called that
# -- so ANY short utterance referencing it is a stop-button command. The STT
# mangles the verb ("show me the stop button" -> "hit / call me the stop
# button"), so we key off the NOUN PHRASE, not the verb.
_BUTTON_RE = re.compile(_BUTTON_WORDS, re.IGNORECASE)
# Close-intent keywords. NB: bare "kill" is intentionally EXCLUDED -- it collides
# with the "kill switch" button phrase ("show me the kill switch" must OPEN).
_CLOSE_KW_RE = re.compile(
    r"\b(?:close|hide|dismiss|get\s+rid|take\s+down|tear\s+down|put\s+away|"
    r"go\s+away|remove|stash|minimi[sz]e|make\s+it\s+go)\b",
    re.IGNORECASE,
)
# A leading QUESTION word or SUBJECT pronoun means narration / a question about
# the button ("where is the stop button", "i hit the stop button earlier"), NOT
# a command -- leave those to the LLM. A mangled imperative ("Hit/Call me/Show me
# the stop button") leads with a VERB, so it is unaffected.
_NONCOMMAND_LEAD_RE = re.compile(
    r"^\s*(?:where|what'?s?|how|why|who|when|which|is|are|was|were|does|do|did|"
    r"has|have|i|we|he|she|they|you|it|that|this|there)\b",
    re.IGNORECASE,
)
# A clause that STATES something about the button rather than commanding it
# ("the stop button interface is not working", "the kill switch is broken") is a
# complaint / narration, NOT a summon -- guards the loose noun-phrase keying.
_BUTTON_STATEMENT_RE = re.compile(
    r"\b(?:not\s+work|isn'?t|wasn'?t|won'?t|doesn'?t|aren'?t|broke|broken|"
    r"interface|stopped|not\s+respond|is\s+not|are\s+not)\b",
    re.IGNORECASE,
)


def _stop_clauses(text: str) -> list[str]:
    """Split an utterance into clauses on sentence/comma boundaries so the command
    is found even when always-listening captured it amid surrounding speech
    ("oh, oh, Ultron started talking. show me the stop button.")."""
    return [c.strip() for c in re.split(r"[.,;!?]+", text) if c.strip()]


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
    if _BUTTON_RE.search(cleaned) is None:
        return None
    # Scan each CLAUSE so the command is found even buried in always-listening
    # filler ("oh, oh, Ultron started talking. show me the stop button."). The
    # SHORT-command cap applies to the matching CLAUSE, not the whole utterance,
    # so leading reaction speech no longer blocks the summon. First command clause
    # wins; a narration-lead question or a complaint about the button is skipped.
    for clause in _stop_clauses(cleaned):
        if _BUTTON_RE.search(clause) is None or len(clause.split()) > 8:
            continue
        if _NONCOMMAND_LEAD_RE.match(clause) or _BUTTON_STATEMENT_RE.search(clause):
            continue
        # A close-intent keyword anywhere in the clause -> close; else a summon.
        return "close" if _CLOSE_KW_RE.search(clause) else "open"
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
        on_toggle_turbo: Optional[Callable[[bool], None]] = None,
        turbo_enabled: bool = False,
        turbo_height: int = 26,
        turbo_label: str = "TURBO",
        on_toggle_chat: Optional[Callable[[bool], None]] = None,
        chat_enabled: bool = False,
        chat_height: int = 26,
        chat_label: str = "CHAT",
        on_toggle_chat_audio: Optional[Callable[[bool], None]] = None,
        chat_audio_enabled: bool = False,
        chat_audio_height: int = 26,
        chat_audio_label: str = "HEAR CHAT",
        on_toggle_say_name: Optional[Callable[[bool], None]] = None,
        say_name_enabled: bool = True,
        say_name_height: int = 26,
        say_name_label: str = "SAY NAME",
        on_restart: Optional[Callable[[], None]] = None,
        on_exit: Optional[Callable[[], None]] = None,
        restart_height: int = 28,
        exit_height: int = 28,
        on_flag: Optional[Callable[[], None]] = None,
        flag_height: int = 26,
        flag_label: str = "FLAG LAST",
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
        # Optional TURBO toggle row: when wired, clicking it flips auto-relay of
        # INFERRED team callouts (the same runtime flag the "turbo mode on/off"
        # voice command flips) -- ON = the loop reads your callouts and relays them
        # without "tell my team"; OFF = keyword relay only (stream/chat-safe).
        # ``_turbo_enabled`` tracks the displayed state across show/hide rebuilds.
        self._on_toggle_turbo = on_toggle_turbo
        self._turbo_enabled = bool(turbo_enabled)
        self._turbo_h = max(0, int(turbo_height))
        self._turbo_label = turbo_label or "TURBO"
        # Optional CHAT toggle row: flips twitch.chat.reply_enabled at runtime
        # (Ultron speaks to or goes silent in Twitch chat without restarting).
        self._on_toggle_chat = on_toggle_chat
        self._chat_enabled = bool(chat_enabled)
        self._chat_h = max(0, int(chat_height))
        self._chat_label = chat_label or "CHAT"
        # Optional HEAR-CHAT toggle row: routes CHAT-directed audio (chat-reply +
        # the "ultron says" redeem) to the LOCAL speakers (ON) or the OBS /
        # broadcast mirror ONLY (OFF, default). Distinct from the CHAT toggle,
        # which enables/disables chat-reply entirely. ``_chat_audio_enabled``
        # tracks the displayed state across show/hide rebuilds.
        self._on_toggle_chat_audio = on_toggle_chat_audio
        self._chat_audio_enabled = bool(chat_audio_enabled)
        self._chat_audio_h = max(0, int(chat_audio_height))
        self._chat_audio_label = chat_audio_label or "HEAR CHAT"
        # Optional SAY-NAME toggle row: flips whether the "ultron tells my team"
        # speak redeem announces the viewer name ("<viewer> says: ...") as a
        # prefix (ON, default) or speaks ONLY the message (OFF). Distinct from the
        # CHAT / HEAR-CHAT toggles (which govern chat-reply + its audio routing).
        # ``_say_name_enabled`` tracks the displayed state across show/hide rebuilds.
        self._on_toggle_say_name = on_toggle_say_name
        self._say_name_enabled = bool(say_name_enabled)
        self._say_name_h = max(0, int(say_name_height))
        self._say_name_label = say_name_label or "SAY NAME"
        # Optional RESTART + EXIT action buttons (orchestrator-wired). Restart =
        # full cleanup then relaunch the same build; Exit = full cleanup then quit.
        self._on_restart = on_restart
        self._on_exit = on_exit
        self._restart_h = max(0, int(restart_height))
        self._exit_h = max(0, int(exit_height))
        # Optional FLAG button: logs the last turn (disliked / missed / unwanted
        # response) to a review log via the orchestrator-wired callback.
        self._on_flag = on_flag
        self._flag_h = max(0, int(flag_height))
        self._flag_label = flag_label or "FLAG LAST"
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
            _has_turbo = self._on_toggle_turbo is not None and self._turbo_h > 0
            turbo_h = self._turbo_h if _has_turbo else 0
            _has_chat = self._on_toggle_chat is not None and self._chat_h > 0
            chat_h = self._chat_h if _has_chat else 0
            _has_chat_audio = (self._on_toggle_chat_audio is not None
                               and self._chat_audio_h > 0)
            chat_audio_h = self._chat_audio_h if _has_chat_audio else 0
            _has_say_name = (self._on_toggle_say_name is not None
                             and self._say_name_h > 0)
            say_name_h = self._say_name_h if _has_say_name else 0
            _has_restart = self._on_restart is not None and self._restart_h > 0
            _has_exit = self._on_exit is not None and self._exit_h > 0
            _has_flag = self._on_flag is not None and self._flag_h > 0
            restart_h = self._restart_h if _has_restart else 0
            exit_h = self._exit_h if _has_exit else 0
            flag_h = self._flag_h if _has_flag else 0
            height = (bar_h + btn_h + restart_h + exit_h + flag_h + ptt_h
                      + turbo_h + chat_h + chat_audio_h + say_name_h)
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

            # TURBO toggle -- amber "TURBO ON" = Ultron reads your callouts and
            # relays them to the team without "tell my team"; grey "TURBO OFF" =
            # keyword relay only (safe to talk to the stream/chat). Distinct amber
            # accent so it is not confused with the green PTT toggle. Flips the same
            # runtime flag the "turbo mode on/off" voice command flips.
            if _has_turbo:
                _t_on_fg, _t_on_fill = "#ff9d3d", "#23150a"      # amber = ON
                _t_off_fg, _t_off_fill = "#8a8f98", "#141414"    # grey = OFF

                def _turbo_colors():
                    return ((_t_on_fg, _t_on_fill) if self._turbo_enabled
                            else (_t_off_fg, _t_off_fill))

                _tfg0, _tfill0 = _turbo_colors()
                turbo_btn = tk.Button(
                    root,
                    text=f"{self._turbo_label} {'ON' if self._turbo_enabled else 'OFF'}",
                    bg=_tfill0, fg=_tfg0,
                    activebackground=_tfill0, activeforeground="#ffffff",
                    relief="flat", bd=0, highlightthickness=1,
                    highlightbackground=_tfg0, highlightcolor=_tfg0,
                    font=("Segoe UI Semibold", 9), cursor="hand2",
                )

                def _toggle_turbo():
                    self._turbo_enabled = not self._turbo_enabled
                    fg, fill = _turbo_colors()
                    try:
                        turbo_btn.configure(
                            text=f"{self._turbo_label} "
                                 f"{'ON' if self._turbo_enabled else 'OFF'}",
                            bg=fill, fg=fg, activebackground=fill,
                            highlightbackground=fg, highlightcolor=fg)
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        if self._on_toggle_turbo is not None:
                            self._on_toggle_turbo(self._turbo_enabled)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("turbo toggle callback failed: %s", e)
                    logger.info("turbo toggle -> %s",
                                "ON" if self._turbo_enabled else "OFF")

                turbo_btn.configure(command=_toggle_turbo)
                turbo_btn.pack(fill="x", side="bottom")
                turbo_btn.bind("<Button-3>", lambda _e: self.hide())

            # CHAT toggle -- purple "CHAT ON" = Ultron speaks to Twitch chat;
            # grey "CHAT OFF" = reads chat but stays silent. Distinct purple
            # accent (Twitch brand). Flips the same runtime flag that
            # _set_twitch_chat_reply_enabled writes.
            if _has_chat:
                _ch_on_fg, _ch_on_fill = "#bf7fff", "#150d20"    # purple = ON
                _ch_off_fg, _ch_off_fill = "#8a8f98", "#141414"  # grey = OFF

                def _chat_colors():
                    return ((_ch_on_fg, _ch_on_fill) if self._chat_enabled
                            else (_ch_off_fg, _ch_off_fill))

                _chfg0, _chfill0 = _chat_colors()
                chat_btn = tk.Button(
                    root,
                    text=f"{self._chat_label} {'ON' if self._chat_enabled else 'OFF'}",
                    bg=_chfill0, fg=_chfg0,
                    activebackground=_chfill0, activeforeground="#ffffff",
                    relief="flat", bd=0, highlightthickness=1,
                    highlightbackground=_chfg0, highlightcolor=_chfg0,
                    font=("Segoe UI Semibold", 9), cursor="hand2",
                )

                def _toggle_chat():
                    self._chat_enabled = not self._chat_enabled
                    fg, fill = _chat_colors()
                    try:
                        chat_btn.configure(
                            text=f"{self._chat_label} "
                                 f"{'ON' if self._chat_enabled else 'OFF'}",
                            bg=fill, fg=fg, activebackground=fill,
                            highlightbackground=fg, highlightcolor=fg)
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        if self._on_toggle_chat is not None:
                            self._on_toggle_chat(self._chat_enabled)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("chat toggle callback failed: %s", e)
                    logger.info("chat toggle -> %s",
                                "ON" if self._chat_enabled else "OFF")

                chat_btn.configure(command=_toggle_chat)
                chat_btn.pack(fill="x", side="bottom")
                chat_btn.bind("<Button-3>", lambda _e: self.hide())

            # HEAR-CHAT toggle -- teal "HEAR CHAT ON" = you hear chat-directed
            # audio (chat-reply + the "ultron says" redeem) through your speakers;
            # grey "HEAR CHAT OFF" (default) = that audio goes to OBS / the
            # broadcast mirror ONLY, so it isn't distracting mid-game. Distinct
            # teal accent (audio-routing, not the purple chat-enable toggle).
            if _has_chat_audio:
                _ca_on_fg, _ca_on_fill = "#33d6c7", "#0a1f1d"    # teal = ON
                _ca_off_fg, _ca_off_fill = "#8a8f98", "#141414"  # grey = OFF

                def _chat_audio_colors():
                    return ((_ca_on_fg, _ca_on_fill) if self._chat_audio_enabled
                            else (_ca_off_fg, _ca_off_fill))

                _cafg0, _cafill0 = _chat_audio_colors()
                chat_audio_btn = tk.Button(
                    root,
                    text=f"{self._chat_audio_label} "
                         f"{'ON' if self._chat_audio_enabled else 'OFF'}",
                    bg=_cafill0, fg=_cafg0,
                    activebackground=_cafill0, activeforeground="#ffffff",
                    relief="flat", bd=0, highlightthickness=1,
                    highlightbackground=_cafg0, highlightcolor=_cafg0,
                    font=("Segoe UI Semibold", 9), cursor="hand2",
                )

                def _toggle_chat_audio():
                    self._chat_audio_enabled = not self._chat_audio_enabled
                    fg, fill = _chat_audio_colors()
                    try:
                        chat_audio_btn.configure(
                            text=f"{self._chat_audio_label} "
                                 f"{'ON' if self._chat_audio_enabled else 'OFF'}",
                            bg=fill, fg=fg, activebackground=fill,
                            highlightbackground=fg, highlightcolor=fg)
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        if self._on_toggle_chat_audio is not None:
                            self._on_toggle_chat_audio(self._chat_audio_enabled)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("chat-audio toggle callback failed: %s", e)
                    logger.info("chat-audio toggle -> %s",
                                "ON" if self._chat_audio_enabled else "OFF")

                chat_audio_btn.configure(command=_toggle_chat_audio)
                chat_audio_btn.pack(fill="x", side="bottom")
                chat_audio_btn.bind("<Button-3>", lambda _e: self.hide())

            # SAY-NAME toggle -- purple "SAY NAME ON" (default) = the team speak
            # redeem prefixes the viewer name ("<viewer> says: ..."); grey "SAY
            # NAME OFF" = it speaks ONLY the message. Governs the TEAM redeem
            # framing (redeem_router.set_say_name_enabled). Distinct violet accent.
            if _has_say_name:
                _sn_on_fg, _sn_on_fill = "#bf7fff", "#1a1024"    # violet = ON
                _sn_off_fg, _sn_off_fill = "#8a8f98", "#141414"  # grey = OFF

                def _say_name_colors():
                    return ((_sn_on_fg, _sn_on_fill) if self._say_name_enabled
                            else (_sn_off_fg, _sn_off_fill))

                _snfg0, _snfill0 = _say_name_colors()
                say_name_btn = tk.Button(
                    root,
                    text=f"{self._say_name_label} "
                         f"{'ON' if self._say_name_enabled else 'OFF'}",
                    bg=_snfill0, fg=_snfg0,
                    activebackground=_snfill0, activeforeground="#ffffff",
                    relief="flat", bd=0, highlightthickness=1,
                    highlightbackground=_snfg0, highlightcolor=_snfg0,
                    font=("Segoe UI Semibold", 9), cursor="hand2",
                )

                def _toggle_say_name():
                    self._say_name_enabled = not self._say_name_enabled
                    fg, fill = _say_name_colors()
                    try:
                        say_name_btn.configure(
                            text=f"{self._say_name_label} "
                                 f"{'ON' if self._say_name_enabled else 'OFF'}",
                            bg=fill, fg=fg, activebackground=fill,
                            highlightbackground=fg, highlightcolor=fg)
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        if self._on_toggle_say_name is not None:
                            self._on_toggle_say_name(self._say_name_enabled)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("say-name toggle callback failed: %s", e)
                    logger.info("say-name toggle -> %s",
                                "ON" if self._say_name_enabled else "OFF")

                say_name_btn.configure(command=_toggle_say_name)
                say_name_btn.pack(fill="x", side="bottom")
                say_name_btn.bind("<Button-3>", lambda _e: self.hide())

            # RESTART + EXIT action buttons -- packed at the bottom (above PTT,
            # below STOP). Each runs Ultron's full cleanup; Restart then relaunches
            # the same build fresh, Exit quits leaving nothing running.
            def _make_action_btn(text, fg, fill, cb):
                b = tk.Button(
                    root, text=text, bg=fill, fg=fg,
                    activebackground=fill, activeforeground="#ffffff",
                    relief="flat", bd=0, highlightthickness=1,
                    highlightbackground=fg, highlightcolor=fg,
                    font=("Segoe UI Semibold", 9), cursor="hand2",
                )

                def _run(_cb=cb):
                    try:
                        _cb()
                    except Exception as e:  # noqa: BLE001
                        logger.warning("stop-window action failed: %s", e)
                b.configure(command=_run)
                b.pack(fill="x", side="bottom")
                b.bind("<Button-3>", lambda _e: self.hide())

            if _has_exit and self._on_exit is not None:
                _make_action_btn("EXIT", "#ff6b6b", "#1a0d0d", self._on_exit)
            if _has_restart and self._on_restart is not None:
                _make_action_btn("RESTART", "#e0a82e", "#1a160a", self._on_restart)

            # FLAG button -- log the last turn (disliked response / missed response
            # / response that should not have happened) to logs/flagged_turns.jsonl
            # for later review. Flashes a brief confirmation so the click is felt.
            if _has_flag and self._on_flag is not None:
                _flag_fg, _flag_fill = "#5b8cff", "#0d1626"      # cool blue
                flag_btn = tk.Button(
                    root, text=self._flag_label, bg=_flag_fill, fg=_flag_fg,
                    activebackground=_flag_fill, activeforeground="#ffffff",
                    relief="flat", bd=0, highlightthickness=1,
                    highlightbackground=_flag_fg, highlightcolor=_flag_fg,
                    font=("Segoe UI Semibold", 9), cursor="hand2",
                )

                def _do_flag(_btn=flag_btn):
                    try:
                        if self._on_flag is not None:
                            self._on_flag()
                    except Exception as e:  # noqa: BLE001
                        logger.warning("flag callback failed: %s", e)
                    # brief confirmation flash, then revert to the label
                    try:
                        _btn.configure(text="FLAGGED ✓", fg="#3ddc84",
                                       highlightbackground="#3ddc84")
                        _btn.after(900, lambda: _btn.configure(
                            text=self._flag_label, fg=_flag_fg,
                            highlightbackground=_flag_fg))
                    except Exception:  # noqa: BLE001
                        pass
                    logger.info("flag button clicked -> last turn logged")

                flag_btn.configure(command=_do_flag)
                flag_btn.pack(fill="x", side="bottom")
                flag_btn.bind("<Button-3>", lambda _e: self.hide())

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
