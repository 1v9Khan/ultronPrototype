"""The Ultron control panel (tkinter, dark theme).

Runs in its OWN process (``python -m ultron.settings_gui``). Layout:

    +--------------------------------------------------------------+
    |  ULTRON // CONTROL PANEL                     status text     |
    |  +--------------------+  +------------------------------+    |
    |  | settings cards     |  |  LIVE LOG  [filter] [pause]   |    |
    |  | (scrollable, 2-col)|  |  streaming logs/ultron.log    |    |
    |  +--------------------+  +------------------------------+    |
    |  [ pending-changes status ]      [ APPLY UPDATE ] [ CLOSE ]   |
    +--------------------------------------------------------------+

Update: patches config.yaml (comment-preserving), then touches the
reload signal so the RUNNING Ultron hot-applies every live-readable
knob (restart-marked knobs show a ``↻`` and apply on next start).
Close: stops the tail thread, destroys the window, exits the process --
the voice pipeline is untouched throughout.
"""

from __future__ import annotations

import queue
import re
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any, Optional

import yaml

from ultron.settings_gui.spec import (
    SECTIONS,
    Knob,
    apply_updates,
    read_value,
    render_value,
    write_reload_signal,
)

# ---------------------------------------------------------------------------
# Palette -- near-black with Ultron crimson accents.
# ---------------------------------------------------------------------------
BG = "#0b0e14"          # window background
CARD = "#11161f"        # card background
CARD_EDGE = "#1d2533"   # card border
FG = "#c9d1d9"          # primary text
DIM = "#8b949e"         # secondary text
ACCENT = "#e5484d"      # ultron red
ACCENT_DIM = "#7d2a2d"
OK = "#3fb950"
WARN = "#d29922"
ERR = "#f85149"
MONO = ("Consolas", 9)
UI = ("Segoe UI", 10)
UI_SMALL = ("Segoe UI", 9)
UI_BOLD = ("Segoe UI", 10, "bold")
TITLE = ("Segoe UI Semibold", 14)


class LogTailer(threading.Thread):
    """Daemon thread streaming new lines of a log file into a queue."""

    def __init__(self, path: Path, out: "queue.Queue[str]") -> None:
        super().__init__(name="gui-log-tail", daemon=True)
        self._path = path
        self._out = out
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:  # pragma: no cover - thread loop
        fh = None
        try:
            while not self._stop.is_set():
                if fh is None:
                    try:
                        fh = self._path.open("r", encoding="utf-8",
                                             errors="replace")
                        fh.seek(0, 2)  # start at the end: live lines only
                    except OSError:
                        time.sleep(0.5)
                        continue
                line = fh.readline()
                if line:
                    try:
                        self._out.put_nowait(line.rstrip("\n"))
                    except queue.Full:
                        pass
                else:
                    time.sleep(0.15)
        finally:
            if fh is not None:
                fh.close()


class ControlPanel:
    """The tkinter application object."""

    def __init__(self, config_path: Path, log_path: Path,
                 data_dir: Path,
                 waveform_path: Optional[Path] = None) -> None:
        self._config_path = config_path
        self._log_path = log_path
        self._data_dir = data_dir
        self._config_data: dict = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        )
        self._vars: dict[tuple[str, ...], tk.Variable] = {}
        self._initial: dict[tuple[str, ...], str] = {}
        self._knobs: dict[tuple[str, ...], Knob] = {}
        self._log_queue: "queue.Queue[str]" = queue.Queue(maxsize=2000)
        self._tailer = LogTailer(log_path, self._log_queue)
        self._wave_queue: "queue.Queue[str]" = queue.Queue(maxsize=64)
        self._wave_tailer: Optional[LogTailer] = None
        if waveform_path is not None:
            self._wave_tailer = LogTailer(waveform_path, self._wave_queue)
        self._paused = False

        self.root = tk.Tk()
        self.root.title("ULTRON // CONTROL PANEL")
        self.root.configure(bg=BG)
        self.root.geometry("1240x760")
        self.root.minsize(980, 600)
        self._style()
        self._build()
        self._tailer.start()
        if self._wave_tailer is not None:
            self._wave_tailer.start()
        self.root.after(120, self._drain_logs)
        self.root.after(160, self._drain_waveform)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------

    def _style(self) -> None:
        s = ttk.Style(self.root)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG, font=UI,
                    fieldbackground=CARD, bordercolor=CARD_EDGE,
                    lightcolor=CARD, darkcolor=CARD)
        s.configure("Card.TFrame", background=CARD)
        s.configure("Bg.TFrame", background=BG)
        s.configure("Card.TLabel", background=CARD, foreground=FG,
                    font=UI_SMALL)
        s.configure("CardTitle.TLabel", background=CARD, foreground=ACCENT,
                    font=UI_BOLD)
        s.configure("Dim.TLabel", background=CARD, foreground=DIM,
                    font=("Segoe UI", 8))
        s.configure("Title.TLabel", background=BG, foreground=FG, font=TITLE)
        s.configure("Status.TLabel", background=BG, foreground=DIM,
                    font=UI_SMALL)
        s.configure("TCheckbutton", background=CARD, foreground=FG,
                    font=UI_SMALL)
        s.map("TCheckbutton", background=[("active", CARD)])
        s.configure("TCombobox", fieldbackground=CARD, background=CARD,
                    foreground=FG, arrowcolor=FG, font=UI_SMALL)
        s.configure("TEntry", fieldbackground="#0e131b", foreground=FG,
                    insertcolor=FG, font=UI_SMALL)
        s.configure("Accent.TButton", background=ACCENT, foreground="#ffffff",
                    font=UI_BOLD, borderwidth=0, focusthickness=0,
                    padding=(18, 7))
        s.map("Accent.TButton",
              background=[("active", "#ff5a5f"), ("disabled", ACCENT_DIM)])
        s.configure("Ghost.TButton", background=CARD, foreground=FG,
                    font=UI, borderwidth=1, padding=(18, 7))
        s.map("Ghost.TButton", background=[("active", CARD_EDGE)])

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self) -> None:
        header = ttk.Frame(self.root, style="Bg.TFrame")
        header.pack(fill="x", padx=14, pady=(12, 6))
        ttk.Label(header, text="ULTRON", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="  //  CONTROL PANEL",
                  style="Status.TLabel").pack(side="left", pady=(4, 0))
        self._clock = ttk.Label(header, text="", style="Status.TLabel")
        self._clock.pack(side="right")

        body = ttk.Frame(self.root, style="Bg.TFrame")
        body.pack(fill="both", expand=True, padx=14, pady=4)
        body.columnconfigure(0, weight=11, uniform="cols")
        body.columnconfigure(1, weight=9, uniform="cols")
        body.rowconfigure(0, weight=1)

        # -- left: scrollable settings cards ---------------------------
        left_wrap = tk.Frame(body, bg=BG)
        left_wrap.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        canvas = tk.Canvas(left_wrap, bg=BG, highlightthickness=0)
        vbar = ttk.Scrollbar(left_wrap, orient="vertical",
                             command=canvas.yview)
        cards = tk.Frame(canvas, bg=BG)
        cards_id = canvas.create_window((0, 0), window=cards, anchor="nw")
        canvas.configure(yscrollcommand=vbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")
        cards.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(
            cards_id, width=e.width))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(
            int(-e.delta / 120), "units"))

        cards.columnconfigure(0, weight=1, uniform="cardcols")
        cards.columnconfigure(1, weight=1, uniform="cardcols")
        for i, section in enumerate(SECTIONS):
            self._build_card(cards, section, row=i // 2, col=i % 2)

        # -- right: waveform pane + live log stream ---------------------
        right_col = tk.Frame(body, bg=BG)
        right_col.grid(row=0, column=1, sticky="nsew")

        wave = tk.Frame(right_col, bg=CARD, highlightbackground=CARD_EDGE,
                        highlightthickness=1)
        wave.pack(fill="x", pady=(0, 8))
        wave_bar = tk.Frame(wave, bg=CARD)
        wave_bar.pack(fill="x", padx=10, pady=(8, 2))
        ttk.Label(wave_bar, text="OUTPUT WAVEFORM",
                  style="CardTitle.TLabel").pack(side="left")
        self._wave_info = ttk.Label(wave_bar, text="waiting for speech…",
                                    style="Dim.TLabel")
        self._wave_info.pack(side="right")
        self._wave_canvas = tk.Canvas(
            wave, height=150, bg="#0a0d12", highlightthickness=0,
        )
        self._wave_canvas.pack(fill="x", padx=10, pady=(0, 10))

        right = tk.Frame(right_col, bg=CARD, highlightbackground=CARD_EDGE,
                         highlightthickness=1)
        right.pack(fill="both", expand=True)
        bar = tk.Frame(right, bg=CARD)
        bar.pack(fill="x", padx=10, pady=(8, 4))
        ttk.Label(bar, text="LIVE LOG", style="CardTitle.TLabel").pack(
            side="left")
        self._pause_btn = ttk.Button(
            bar, text="pause", style="Ghost.TButton", width=7,
            command=self._toggle_pause)
        self._pause_btn.pack(side="right")
        self._filter_var = tk.StringVar()
        flt = ttk.Entry(bar, textvariable=self._filter_var, width=22)
        flt.pack(side="right", padx=6)
        ttk.Label(bar, text="filter", style="Dim.TLabel").pack(side="right")

        self._log = tk.Text(
            right, bg="#0a0d12", fg=FG, insertbackground=FG, font=MONO,
            wrap="none", state="disabled", relief="flat",
            highlightthickness=0,
        )
        self._log.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._log.tag_configure("err", foreground=ERR)
        self._log.tag_configure("warn", foreground=WARN)
        self._log.tag_configure("info", foreground=DIM)
        self._log.tag_configure("hot", foreground=FG)

        # -- bottom bar --------------------------------------------------
        bottom = ttk.Frame(self.root, style="Bg.TFrame")
        bottom.pack(fill="x", padx=14, pady=(4, 12))
        self._status = ttk.Label(bottom, text="No pending changes.",
                                 style="Status.TLabel")
        self._status.pack(side="left")
        ttk.Button(bottom, text="CLOSE", style="Ghost.TButton",
                   command=self.close).pack(side="right")
        self._apply_btn = ttk.Button(
            bottom, text="APPLY UPDATE", style="Accent.TButton",
            command=self._apply)
        self._apply_btn.pack(side="right", padx=8)
        self._tick_clock()

    def _build_card(self, parent: tk.Frame, section: Any, row: int,
                    col: int) -> None:
        card = tk.Frame(parent, bg=CARD, highlightbackground=CARD_EDGE,
                        highlightthickness=1)
        card.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
        card.columnconfigure(1, weight=1)
        ttk.Label(card, text=section.title.upper(),
                  style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(7, 3))
        for r, knob in enumerate(section.knobs, start=1):
            current = read_value(self._config_data, knob.path)
            label_text = knob.label + ("  ↻" if knob.restart else "")
            lbl = ttk.Label(card, text=label_text, style="Card.TLabel")
            lbl.grid(row=r, column=0, sticky="w", padx=(10, 6), pady=2)
            var, widget = self._make_widget(card, knob, current)
            widget.grid(row=r, column=1, sticky="ew", padx=(0, 10), pady=2)
            self._vars[knob.path] = var
            self._knobs[knob.path] = knob
            self._initial[knob.path] = str(var.get())
            var.trace_add("write", lambda *_: self._refresh_status())
        if any(k.restart for k in section.knobs):
            ttk.Label(card, text="↻ = applies on next start",
                      style="Dim.TLabel").grid(
                row=len(section.knobs) + 1, column=0, columnspan=2,
                sticky="w", padx=10, pady=(2, 7),
            )
        else:
            card.grid_rowconfigure(len(section.knobs) + 1, minsize=7)

    def _make_widget(self, parent: tk.Frame, knob: Knob,
                     current: Any) -> tuple[tk.Variable, tk.Widget]:
        if knob.kind == "bool":
            var: tk.Variable = tk.BooleanVar(value=bool(current))
            return var, ttk.Checkbutton(parent, variable=var)
        if knob.kind == "choice":
            var = tk.StringVar(value=str(current))
            return var, ttk.Combobox(
                parent, textvariable=var, values=list(knob.choices),
                state="readonly", width=14, font=UI_SMALL)
        if knob.kind == "csv":
            var = tk.StringVar(
                value=", ".join(current) if isinstance(current, list) else "")
            return var, ttk.Entry(parent, textvariable=var, width=18)
        var = tk.StringVar(value="" if current is None else str(current))
        return var, ttk.Entry(parent, textvariable=var, width=18)

    # ------------------------------------------------------------------
    # Behaviour
    # ------------------------------------------------------------------

    def _pending(self) -> dict[tuple[str, ...], str]:
        """Changed knobs -> rendered YAML values (validated)."""
        out: dict[tuple[str, ...], str] = {}
        for path, var in self._vars.items():
            if str(var.get()) == self._initial[path]:
                continue
            knob = self._knobs[path]
            value: Any = var.get()
            if knob.kind == "int":
                value = int(float(str(value)))
            elif knob.kind == "float":
                value = float(str(value))
            if knob.kind in ("int", "float"):
                if knob.minimum is not None and float(value) < knob.minimum:
                    raise ValueError(f"{knob.label}: below {knob.minimum}")
                if knob.maximum is not None and float(value) > knob.maximum:
                    raise ValueError(f"{knob.label}: above {knob.maximum}")
            out[path] = render_value(value, knob.kind)
        return out

    def _refresh_status(self) -> None:
        try:
            n = len(self._pending())
        except Exception:
            self._status.configure(text="Invalid value in a field.",
                                   foreground=ERR)
            return
        self._status.configure(
            text=("No pending changes." if n == 0
                  else f"{n} change{'s' if n != 1 else ''} pending."),
            foreground=(DIM if n == 0 else WARN),
        )

    def _apply(self) -> None:
        try:
            updates = self._pending()
            if not updates:
                self._status.configure(text="Nothing to apply.",
                                       foreground=DIM)
                return
            apply_updates(self._config_path, updates)
            write_reload_signal(self._data_dir)
            restart_needed = any(
                self._knobs[p].restart for p in updates)
            for path in updates:
                self._initial[path] = str(self._vars[path].get())
            msg = f"Applied {len(updates)} change" \
                  f"{'s' if len(updates) != 1 else ''} ✓ — live knobs " \
                  f"hot-reloaded."
            if restart_needed:
                msg += "  ↻ marked knobs apply on next start."
            self._status.configure(text=msg, foreground=OK)
        except Exception as e:  # noqa: BLE001 - surface, never crash
            self._status.configure(text=f"Apply failed: {e}", foreground=ERR)

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self._pause_btn.configure(text="resume" if self._paused else "pause")

    def _drain_logs(self) -> None:
        if not self._paused:
            flt = self._filter_var.get().strip().lower()
            batch: list[str] = []
            while True:
                try:
                    batch.append(self._log_queue.get_nowait())
                except queue.Empty:
                    break
            if batch:
                self._log.configure(state="normal")
                for line in batch:
                    if flt and flt not in line.lower():
                        continue
                    tag = "info"
                    if re.search(r"\| ERROR|\bCRITICAL\b|Traceback", line):
                        tag = "err"
                    elif "| WARNING" in line:
                        tag = "warn"
                    elif re.search(r"relay:|routing:classified|TTFT", line):
                        tag = "hot"
                    self._log.insert("end", line + "\n", tag)
                # Bound the widget: keep the last ~2000 lines.
                excess = int(float(self._log.index("end-1c").split(".")[0])) - 2000
                if excess > 0:
                    self._log.delete("1.0", f"{excess + 1}.0")
                self._log.see("end")
                self._log.configure(state="disabled")
        self.root.after(150, self._drain_logs)

    def _drain_waveform(self) -> None:
        """Render the newest synthesized clip's envelope with blip markers."""
        latest: Optional[str] = None
        while True:
            try:
                latest = self._wave_queue.get_nowait()
            except queue.Empty:
                break
        if latest:
            try:
                import json as _json

                rec = _json.loads(latest)
                self._draw_waveform(rec)
            except Exception:  # noqa: BLE001 - malformed line, skip
                pass
        self.root.after(180, self._drain_waveform)

    def _draw_waveform(self, rec: dict) -> None:
        c = self._wave_canvas
        c.delete("all")
        w = max(int(c.winfo_width()), 50)
        h = int(c["height"])
        mid = h // 2
        env = rec.get("env") or []
        if not env:
            return
        peak = max(max(env), 0.05)
        n = len(env)
        bar_w = max(w / n, 1.0)
        for i, v in enumerate(env):
            x = i * bar_w + bar_w / 2
            amp = (v / peak) * (mid - 8)
            c.create_line(x, mid - amp, x, mid + amp, fill=ACCENT, width=2)
        c.create_line(0, mid, w, mid, fill=CARD_EDGE)
        # Red markers at the analyzer's finding positions.
        duration_ms = float(rec.get("duration_s", 0.0)) * 1000.0
        findings = rec.get("findings") or []
        for f in findings:
            if duration_ms <= 0:
                break
            fx = min(max(f.get("position_ms", 0.0) / duration_ms, 0.0), 1.0) * w
            c.create_line(fx, 4, fx, h - 4, fill=ERR, width=2, dash=(3, 3))
            c.create_text(
                min(fx + 4, w - 40), 12, text=f.get("kind", "")[:14],
                fill=ERR, anchor="w", font=("Segoe UI", 7),
            )
        label = (rec.get("label") or "")[:46]
        kinds = ", ".join(f.get("kind", "") for f in findings)
        info = f"{label!r} · {rec.get('duration_s', 0):.2f}s"
        info += f" · ⚠ {kinds}" if kinds else " · clean"
        self._wave_info.configure(
            text=info, foreground=(ERR if kinds else DIM))

    def _tick_clock(self) -> None:
        self._clock.configure(text=time.strftime("%H:%M:%S"))
        self.root.after(1000, self._tick_clock)

    def close(self) -> None:
        """Stop the tail threads, destroy the window, exit the process."""
        for tailer in (self._tailer, self._wave_tailer):
            if tailer is None:
                continue
            try:
                tailer.stop()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.root.destroy()
        except Exception:  # noqa: BLE001
            pass

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    """Entry point for ``python -m ultron.settings_gui``."""
    from ultron.config import LOGS_DIR, PROJECT_ROOT

    panel = ControlPanel(
        config_path=Path(PROJECT_ROOT) / "config.yaml",
        log_path=Path(LOGS_DIR) / "ultron.log",
        data_dir=Path(PROJECT_ROOT) / "data",
        waveform_path=Path(LOGS_DIR) / "audio_waveform.jsonl",
    )
    panel.run()
    return 0
