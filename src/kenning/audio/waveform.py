"""Waveform overlay window -- a compact, dynamic visualizer of Kenning's voice
for OBS window-capture.

A separate always-on-top, borderless window (NOT the settings panel) that
renders a circular/radial audio visualizer reacting in real time to EVERY line
Kenning speaks -- normal conversation AND team relay -- so stream viewers can
see "him" talking. The user adds it in OBS as a single **Window Capture**
source; no second physical output device or virtual cable is needed for the
*visual*.

Architecture mirrors :class:`kenning.audio.broadcast.BroadcastSink`:

* **Zero latency on the speaker path.** ``submit`` only copies the clip and
  drops it on a bounded queue (drop-oldest). A daemon *pacer* thread analyses
  the clip (FFT band envelope + RMS) and walks it at real time, publishing the
  current frame to shared state.
* **A dedicated UI thread** owns its own ``tk.Tk()`` root + Canvas and an
  ~30 fps redraw loop that eases the rendered shape toward the published frame
  (and decays to an idle breath between utterances). All Tk calls live on that
  one thread; the audio side only ever touches a lock-guarded numpy frame.
* **Fail-open everywhere.** No display, no Tk, a backend hiccup -- the window
  just never appears; the voice path is untouched.
* **Near-free when off.** With the visualizer disabled (the default),
  ``submit`` is a single attribute check and an immediate return.

The window background is a single chroma colour; with ``transparent`` on
(Windows), that colour is keyed out so only the glowing visualizer shows over
your game -- drag the OBS source wherever you like.
"""
from __future__ import annotations

import math
import queue
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from kenning.utils.logging import get_logger

logger = get_logger("audio.waveform")

_QUEUE_MAXSIZE = 8
Frame = Tuple[float, np.ndarray]  # (level 0..1, bands[N] 0..1)

# Absolute RMS that maps to a "full" core pulse; clips quieter than this read
# proportionally smaller so silence stays calm rather than slamming to max.
_RMS_FULL_SCALE = 0.18


def analyze_clip(pcm: np.ndarray, sr: int, *, fps: int, n_bands: int) -> List[Frame]:
    """Turn one spoken clip into a per-UI-frame (level, band-envelope) sequence.

    Log-spaced magnitude bands over the speech range, log-compressed, per-clip
    normalised for lively motion, then scaled by absolute loudness so quiet
    frames render small. Pure/fail-open: returns ``[]`` on any anomaly.
    """
    try:
        x = np.asarray(pcm)
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = x.astype(np.float32) / 32768.0
        n = x.shape[0]
        if n < 8 or sr <= 0:
            return []
        hop = max(1, int(round(sr / max(1, fps))))
        win = 1024
        nyq = sr / 2.0
        fmin, fmax = 90.0, min(nyq * 0.9, 7500.0)
        if fmin >= fmax:                       # pathological / tiny sample rate
            return []
        edges = np.logspace(math.log10(fmin), math.log10(fmax), n_bands + 1)
        freqs = np.fft.rfftfreq(win, 1.0 / sr)
        band_bins = [
            np.where((freqs >= edges[b]) & (freqs < edges[b + 1]))[0]
            for b in range(n_bands)
        ]
        window = np.hanning(win).astype(np.float32)
        levels: List[float] = []
        raw_bands: List[np.ndarray] = []
        for start in range(0, n, hop):
            seg = x[start:start + win]
            if seg.shape[0] < win:
                seg = np.pad(seg, (0, win - seg.shape[0]))
            mag = np.abs(np.fft.rfft(seg * window))
            bands = np.array(
                [mag[ix].mean() if ix.size else 0.0 for ix in band_bins],
                dtype=np.float32,
            )
            raw_bands.append(np.log1p(bands * 6.0))
            levels.append(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))))
        if not raw_bands:
            return []
        allb = np.stack(raw_bands)
        bmax = max(1e-6, float(np.percentile(allb, 98.0)))
        frames: List[Frame] = []
        for lvl, bands in zip(levels, raw_bands):
            level = min(1.0, lvl / _RMS_FULL_SCALE)
            disp = np.clip(bands / bmax, 0.0, 1.0) * (0.28 + 0.72 * level)
            frames.append((level, disp.astype(np.float32)))
        return frames
    except Exception as e:  # noqa: BLE001 - never break the voice path
        logger.debug("waveform analyze failed (%s)", e)
        return []


def _lerp_color(c0: Tuple[int, int, int], c1: Tuple[int, int, int], t: float) -> str:
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    r = int(c0[0] + (c1[0] - c0[0]) * t)
    g = int(c0[1] + (c1[1] - c0[1]) * t)
    b = int(c0[2] + (c1[2] - c0[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(c: Tuple[int, int, int]) -> str:
    return f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}"


def _lerp_rgb(c0, c1, t):
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    return (int(c0[0] + (c1[0] - c0[0]) * t),
            int(c0[1] + (c1[1] - c0[1]) * t),
            int(c0[2] + (c1[2] - c0[2]) * t))


def _load_pil_font(family: str, size: int):
    """Best-effort TrueType load by family name -> Windows font file, with
    sensible fallbacks; PIL's default bitmap font as a last resort."""
    from PIL import ImageFont

    fl = (family or "").lower()
    cands = []
    if "bahnschrift" in fl:
        cands.append("bahnschrift.ttf")
    if "impact" in fl:
        cands.append("impact.ttf")
    if family:
        cands.append(family + ".ttf")
    cands += ["bahnschrift.ttf", "impact.ttf", "arialbd.ttf", "seguisb.ttf", "arial.ttf"]
    for cand in cands:
        try:
            return ImageFont.truetype(cand, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def _nameplate_frames(W, H, text, font_family, *, plate_fill, accent_rgb,
                      core_idle, neon_red, buckets):
    """Render the ULTRON plate as ``buckets`` RGBA PIL frames from idle (calm,
    readable) to full speech (bright neon + soft Gaussian bloom). The glow is a
    real blur -> rounded, soft, particle-like halo (like a neon tube), in the
    SAME red the glyphs light up to. Returns a list of PIL.Image (RGBA)."""
    from PIL import Image, ImageDraw, ImageFilter

    pad = W * 0.085
    px0, py0, px1, py1 = pad, H * 0.16, W - pad, H * 0.84
    radius = int(H * 0.26)
    font = _load_pil_font(font_family, max(10, int(H * 0.46)))
    n = max(1, len(text))
    inner_w = (px1 - px0) * 0.84
    left = (px0 + px1) / 2.0 - inner_w / 2.0
    ty = (py0 + py1) / 2.0
    xs = [left + (i + 0.5) * (inner_w / n) for i in range(len(text))]
    plate_rgba = (*plate_fill, 255)
    accent_rgba = (*accent_rgb, 255)
    nr = neon_red
    sw = max(1, int(H * 0.04))            # fatten the glow source so blur has mass
    frames = []
    for bi in range(buckets):
        level = bi / max(1, buckets - 1)
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([2, 2, W - 3, H - 3], radius=radius,
                            fill=plate_rgba, outline=accent_rgba, width=2)
        glow_inten = 0.16 + 0.84 * level
        # Wide soft halo + tight bright core, each composited a few times so the
        # bloom is actually visible while staying soft + rounded (real blur).
        for blur_fac, reps, a_mul in ((0.18, 2, 0.9), (0.07, 3, 1.0)):
            gl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            gd = ImageDraw.Draw(gl)
            for i, ch in enumerate(text):
                gd.text((xs[i], ty), ch, font=font, anchor="mm",
                        fill=(nr[0], nr[1], nr[2], 255),
                        stroke_width=sw, stroke_fill=(nr[0], nr[1], nr[2], 255))
            blur = max(0.8, H * blur_fac * (0.5 + 0.9 * level))
            gl = gl.filter(ImageFilter.GaussianBlur(blur))
            scale = min(1.0, glow_inten * a_mul)
            gl.putalpha(gl.split()[3].point(lambda p: int(p * scale)))
            for _ in range(reps):
                img = Image.alpha_composite(img, gl)
        # Crisp tube core LAST, on top of the bloom: a WHITE-HOT glyph (like the
        # lit glass of a neon tube) with a thin neon-red rim. The white-hot
        # centre stays legible against the equally-red, equally-bright halo --
        # the glow is unchanged, the letters just read. Brightens as he speaks.
        white_hot = (255, 240, 242)
        core = _lerp_rgb(core_idle, white_hot, level)
        rim = max(1, int(H * 0.02))
        cd = ImageDraw.Draw(img)
        for i, ch in enumerate(text):
            cd.text((xs[i], ty), ch, font=font, anchor="mm",
                    fill=(core[0], core[1], core[2], 255),
                    stroke_width=rim, stroke_fill=(nr[0], nr[1], nr[2], 255))
        frames.append(img)
    return frames


def _round_rect_points(x0, y0, x1, y1, r):
    """Point list for a rounded rectangle, used with create_polygon(smooth=True)."""
    return [
        x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r,
        x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
        x0, y1, x0, y1 - r, x0, y0 + r, x0, y0,
    ]


def _make_background_overlay(root) -> None:
    """Windows: turn the overlay into an unobtrusive BACKGROUND window -- a tool
    window (no taskbar / Alt-Tab entry) that never activates or steals focus, so
    it sinks behind your other windows and isn't distracting, while OBS Window
    Capture (the 'Windows 10 (1903+)' / WGC method) still grabs its pixels even
    when fully occluded. Used when ``always_on_top`` is off. Fail-open / no-op
    off Windows. No administrator rights required for this (non-elevated) window.
    """
    try:
        import sys
        if not sys.platform.startswith("win"):
            return
        import ctypes

        GWL_EXSTYLE = -20
        WS_EX_TOOLWINDOW = 0x00000080
        WS_EX_NOACTIVATE = 0x08000000
        u32 = ctypes.windll.user32
        u32.GetWindowLongW.restype = ctypes.c_long
        u32.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
        u32.SetWindowLongW.restype = ctypes.c_long
        u32.SetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
        root.update_idletasks()
        hwnd = ctypes.c_void_p(root.winfo_id())
        style = u32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        u32.SetWindowLongW(hwnd, GWL_EXSTYLE,
                           style | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
    except Exception as e:  # noqa: BLE001
        logger.debug("background overlay styles not applied (%s)", e)


class WaveformSink:
    """Daemon-backed voice visualizer. One per process (see
    :func:`get_waveform_sink`). Safe to ``submit`` from any thread."""

    def __init__(self) -> None:
        self._enabled = False
        self._lock = threading.Lock()
        self._queue: "queue.Queue[Optional[Tuple[np.ndarray, int]]]" = queue.Queue(
            maxsize=_QUEUE_MAXSIZE
        )
        self._pacer: Optional[threading.Thread] = None
        self._ui: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # Appearance (set by configure()).
        self._size = 300
        self._bars = 60
        self._fps = 30
        self._bg = "#0b0b10"
        self._accent = "#e5484d"
        self._transparent = True
        self._always_on_top = True
        self._title = "KENNING // VOICE"
        self._nameplate_text = "ULTRON"
        self._nameplate_font = "Bahnschrift"
        # Shared animation state (published by pacer, read by UI thread).
        self._target_level = 0.0
        self._target_bands = np.zeros(self._bars, dtype=np.float32)
        self._zero_bands = np.zeros(self._bars, dtype=np.float32)

    # -- producer side -----------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def configure(
        self,
        *,
        enabled: bool,
        size: Optional[int] = None,
        bars: Optional[int] = None,
        fps: Optional[int] = None,
        bg_color: Optional[str] = None,
        accent_color: Optional[str] = None,
        transparent: Optional[bool] = None,
        always_on_top: Optional[bool] = None,
        nameplate_text: Optional[str] = None,
        nameplate_font: Optional[str] = None,
    ) -> None:
        """Enable/disable the overlay and (re)apply appearance. Starts the
        pacer + UI threads on first enable; idempotent thereafter."""
        with self._lock:
            if size is not None:
                self._size = max(120, int(size))
            if bars is not None and int(bars) != self._bars:
                self._bars = max(8, int(bars))
                self._target_bands = np.zeros(self._bars, dtype=np.float32)
                self._zero_bands = np.zeros(self._bars, dtype=np.float32)
            if fps is not None:
                self._fps = max(10, min(60, int(fps)))
            if bg_color:
                self._bg = bg_color
            if accent_color:
                self._accent = accent_color
            if transparent is not None:
                self._transparent = bool(transparent)
            if always_on_top is not None:
                self._always_on_top = bool(always_on_top)
            if nameplate_text is not None:
                self._nameplate_text = str(nameplate_text)
            if nameplate_font:
                self._nameplate_font = str(nameplate_font)
            was = self._enabled
            self._enabled = bool(enabled)
            start = self._enabled and not was
            stop = not self._enabled and was
        # Start/stop the window OUTSIDE the lock: teardown joins the UI thread,
        # and that thread takes _lock every frame -- joining under _lock would
        # deadlock. Disable fully tears the window down (overrideredirect
        # windows don't reliably withdraw on Windows); re-enable builds a fresh
        # one (cheap, and avoids any stale-visibility ambiguity).
        if start:
            self._stop.clear()
            self._start_threads()
        elif stop:
            self._teardown()

    def submit(self, pcm: np.ndarray, sample_rate: int) -> None:
        """Tee one spoken clip to the visualizer. Non-blocking, fail-open."""
        if not self._enabled or pcm is None:
            return
        try:
            data = np.asarray(pcm)
            if data.size == 0:
                return
            if data.dtype != np.int16:
                data = np.clip(data.astype(np.float32), -32768.0, 32767.0).astype(np.int16)
            data = np.ascontiguousarray(data).copy()
        except Exception:  # noqa: BLE001
            return
        try:
            self._queue.put_nowait((data, int(sample_rate)))
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait((data, int(sample_rate)))
            except queue.Full:
                pass

    def close(self) -> None:
        """Stop threads and tear down the window. Best-effort/idempotent."""
        self._enabled = False
        self._teardown()

    # -- threads -----------------------------------------------------------

    def _start_threads(self) -> None:
        if self._pacer is None or not self._pacer.is_alive():
            self._pacer = threading.Thread(
                target=self._pace_loop, daemon=True, name="waveform-pacer")
            self._pacer.start()
        if self._ui is None or not self._ui.is_alive():
            self._ui = threading.Thread(
                target=self._ui_loop, daemon=True, name="waveform-ui")
            self._ui.start()

    def _teardown(self) -> None:
        """Stop the pacer + UI threads + window, then join them so the Tcl
        interpreter is torn down on its own thread before we return. NEVER call
        while holding ``_lock`` -- the UI thread takes ``_lock`` each frame, so
        joining under the lock would deadlock."""
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        for th in (self._pacer, self._ui):
            if th is not None and th is not threading.current_thread():
                try:
                    th.join(timeout=2.5)
                except Exception:  # noqa: BLE001
                    pass
        # Drain leftover clips + the sentinel so a later re-enable starts clean
        # (a fresh pacer must not immediately read a stale None and exit).
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        self._pacer = None
        self._ui = None

    def _pace_loop(self) -> None:
        """Analyse each queued clip and publish frames at real-time pace."""
        while not self._stop.is_set():
            try:
                item = self._queue.get()
            except Exception:  # noqa: BLE001
                break
            if item is None:
                if self._stop.is_set():
                    break
                continue          # stale wake-up (e.g. left over from a prior
            pcm, sr = item        # disable) -- keep waiting, don't exit
            with self._lock:
                fps = self._fps
                n_bands = self._bars          # capture so a mid-clip bars
            frames = analyze_clip(pcm, sr, fps=fps, n_bands=n_bands)  # change can't desync
            dt = 1.0 / max(1, fps)
            for level, bands in frames:
                if self._stop.is_set() or not self._enabled:
                    break
                with self._lock:
                    self._target_level = level
                    self._target_bands = bands
                time.sleep(dt)
            with self._lock:
                self._target_level = 0.0
                self._target_bands = self._zero_bands

    def _ui_loop(self) -> None:
        """Own the Tk root + Canvas and run the redraw loop. Fail-open."""
        try:
            import tkinter as tk
        except Exception as e:  # noqa: BLE001
            logger.warning("waveform overlay unavailable (no tkinter: %s)", e)
            return
        try:
            root = tk.Tk()
            root.title(self._title)
            size = self._size
            plate_text = (self._nameplate_text or "").strip()
            plate_h = int(round(size * 0.26)) if plate_text else 0
            height = size + plate_h
            root.geometry(f"{size}x{height}+80+80")
            root.configure(bg=self._bg)
            root.overrideredirect(True)  # borderless
            if self._always_on_top:
                root.wm_attributes("-topmost", True)
            if self._transparent:
                try:
                    root.wm_attributes("-transparentcolor", self._bg)
                except Exception:  # noqa: BLE001 - non-Windows / unsupported
                    pass
            if not self._always_on_top:
                # Background mode: let it hide behind your desktop windows while
                # OBS (Windows 10 1903+ capture) still grabs it.
                _make_background_overlay(root)
            canvas = tk.Canvas(
                root, width=size, height=height, bg=self._bg,
                highlightthickness=0, bd=0)
            canvas.pack(fill="both", expand=True)

            state = _RenderState(canvas, size, plate_h, self._bars, self._accent,
                                 self._bg, plate_text, self._nameplate_font)
            state.build()

            # Drag the window by grabbing the visualizer; right-click closes.
            def _press(e):
                state.drag_x, state.drag_y = e.x, e.y

            def _drag(e):
                root.geometry(f"+{root.winfo_x() + e.x - state.drag_x}"
                              f"+{root.winfo_y() + e.y - state.drag_y}")
            canvas.bind("<Button-1>", _press)
            canvas.bind("<B1-Motion>", _drag)
            canvas.bind("<Button-3>", lambda _e: self.close())

            frame_ms = max(16, int(1000 / max(1, self._fps)))

            def _tick():
                if self._stop.is_set():
                    try:
                        root.quit()  # return out of mainloop; teardown below
                    except Exception:  # noqa: BLE001
                        pass
                    return
                with self._lock:
                    tgt_level = self._target_level
                    tgt_bands = self._target_bands
                try:
                    state.render(tgt_level, tgt_bands)
                except Exception as e:  # noqa: BLE001
                    logger.debug("waveform render glitch (%s)", e)
                root.after(frame_ms, _tick)

            root.after(frame_ms, _tick)
            logger.info("waveform overlay window up (%dx%d)", size, size)
            try:
                root.mainloop()
            finally:
                # Tear the Tcl interpreter down ON THIS thread (the one that
                # created it) and force its finalization here, so the process
                # exit doesn't trigger 'Tcl_AsyncDelete: ... wrong thread'.
                try:
                    root.destroy()
                except Exception:  # noqa: BLE001
                    pass
                state = None  # drop the canvas-item refs
                root = canvas = None
                import gc
                gc.collect()
        except Exception as e:  # noqa: BLE001
            logger.warning("waveform overlay stopped (%s)", e)


class _RenderState:
    """Holds the pre-created Canvas items and eases them toward each frame."""

    def __init__(self, canvas, size: int, plate_h: int, bars: int, accent: str,
                 bg: str, nameplate_text: str = "", nameplate_font: str = "Bahnschrift") -> None:
        self.canvas = canvas
        self.size = size
        self.plate_h = plate_h
        self.bars = bars
        self.accent_rgb = _hex_to_rgb(accent)
        self.tip_rgb = (255, 240, 240)
        self.bg = bg
        self.cx = size / 2.0
        self.cy = size / 2.0
        self.r0 = size * 0.20          # inner ring radius
        self.r_max = size * 0.46       # max bar tip
        self.cur_level = 0.0
        self.cur_bands = np.zeros(bars, dtype=np.float32)
        self.angle = 0.0
        self.drag_x = 0
        self.drag_y = 0
        self.glow_items: list = []
        self.bar_items: list = []
        self.core = None
        # Nameplate (ULTRON): a dark plate (contrast over gameplay) with a REAL
        # Gaussian-blurred neon glow -- a bright tube core + soft, rounded halo
        # in the SAME red the glyphs light up to -- pre-rendered with PIL at N
        # brightness buckets and swapped per frame on a fast attack/decay
        # envelope (quick brighten, quick fade). Falls back to plain canvas text
        # if PIL is unavailable.
        self.text = (nameplate_text or "").strip()
        self.font_family = nameplate_font or "Bahnschrift"
        self.PLATE_FILL = (22, 22, 30)    # distinct from chroma -> opaque plate
        self.CORE_IDLE = (230, 222, 225)  # calm + readable when not speaking
        self.NEON_RED = (255, 88, 98)     # glyphs light up THIS; glow is the SAME red
        self.cur_glow = 0.0
        self._glow_buckets = 16
        self._last_bucket = -1
        self.plate_imgs: list = []        # ImageTk.PhotoImage per brightness bucket
        self.plate_img_item = None        # canvas image id
        self.fallback_text = None         # used only if PIL is unavailable

    def build(self) -> None:
        c = self.canvas
        # Outer glow rings (drawn first, behind everything).
        for _ in range(3):
            self.glow_items.append(
                c.create_oval(0, 0, 0, 0, outline=self.bg, width=2))
        # Radial bars.
        for _ in range(self.bars):
            self.bar_items.append(
                c.create_line(0, 0, 0, 0, fill=self.bg, width=3,
                              capstyle="round"))
        # Pulsing core.
        self.core = c.create_oval(0, 0, 0, 0, fill=self.bg, outline="")
        # ---- Nameplate ----
        if self.plate_h > 0 and self.text:
            try:
                self._build_nameplate()
            except Exception as e:  # noqa: BLE001 - degrade to plain text
                logger.debug("nameplate glow build failed; plain text (%s)", e)
                try:
                    self._build_nameplate_fallback()
                except Exception:  # noqa: BLE001
                    pass

    def _build_nameplate(self) -> None:
        """Pre-render the ULTRON plate at ``_glow_buckets`` brightness levels,
        each with a real Gaussian-blurred neon glow, and place the (level 0)
        image on the canvas. Requires PIL (Pillow + ImageTk)."""
        from PIL import ImageTk

        frames = _nameplate_frames(
            self.size, self.plate_h, self.text, self.font_family,
            plate_fill=self.PLATE_FILL, accent_rgb=self.accent_rgb,
            core_idle=self.CORE_IDLE, neon_red=self.NEON_RED,
            buckets=self._glow_buckets)
        self.plate_imgs = [ImageTk.PhotoImage(f) for f in frames]
        cx = self.size / 2.0
        cy = self.size + self.plate_h / 2.0
        self.plate_img_item = self.canvas.create_image(
            cx, cy, image=self.plate_imgs[0])

    def _build_nameplate_fallback(self) -> None:
        """Plain crisp name (no plate/glow) if PIL isn't available."""
        cx = self.size / 2.0
        cy = self.size + self.plate_h / 2.0
        fsize = max(10, int(self.plate_h * 0.42))
        self.fallback_text = self.canvas.create_text(
            cx, cy, text=self.text, anchor="center",
            font=(self.font_family, fsize, "bold"),
            fill=_rgb_to_hex(self.CORE_IDLE))

    def render(self, target_level: float, target_bands: np.ndarray) -> None:
        c = self.canvas
        # Ease current -> target (attack fast, release smooth).
        self.cur_level += (target_level - self.cur_level) * (
            0.55 if target_level > self.cur_level else 0.18)
        if target_bands.shape[0] != self.cur_bands.shape[0]:
            self.cur_bands = np.zeros(self.bars, dtype=np.float32)
        gain = np.where(target_bands > self.cur_bands, 0.6, 0.22)
        self.cur_bands = self.cur_bands + (target_bands - self.cur_bands) * gain
        # Idle breathing so it's never fully dead on screen.
        breath = 0.04 * (0.5 + 0.5 * math.sin(self.angle * 1.7))
        self.angle += 0.018
        level = max(self.cur_level, breath)

        accent, tip, bg = self.accent_rgb, self.tip_rgb, self.bg
        cx, cy, r0, r_max = self.cx, self.cy, self.r0, self.r_max
        n = self.bars
        half = n // 2
        for i in range(n):
            # Mirror left/right for symmetry.
            bi = i if i <= half else n - i
            bi = min(bi, self.cur_bands.shape[0] - 1)
            amp = float(self.cur_bands[bi]) + breath * 0.6
            ang = self.angle + (2.0 * math.pi * i / n)
            ca, sa = math.cos(ang), math.sin(ang)
            inner = r0 + 3.0
            outer = r0 + 6.0 + amp * (r_max - r0)
            x0, y0 = cx + ca * inner, cy + sa * inner
            x1, y1 = cx + ca * outer, cy + sa * outer
            col = _lerp_color(accent, tip, min(1.0, amp * 1.1))
            c.coords(self.bar_items[i], x0, y0, x1, y1)
            c.itemconfigure(self.bar_items[i], fill=col,
                            width=max(2, int(self.size * 0.012)))
        # Pulsing core.
        cr = r0 * (0.62 + 0.5 * level)
        core_col = _lerp_color((40, 12, 16), accent, 0.35 + 0.65 * level)
        c.coords(self.core, cx - cr, cy - cr, cx + cr, cy + cr)
        c.itemconfigure(self.core, fill=core_col)
        # Glow rings expand with level.
        for k, item in enumerate(self.glow_items):
            gr = r0 + (r_max - r0) * (0.5 + 0.5 * level) * (0.6 + 0.25 * k)
            shade = _lerp_color(bg if isinstance(bg, tuple) else _hex_to_rgb(bg),
                                accent, max(0.0, level - 0.15 * k) * 0.5)
            c.coords(item, cx - gr, cy - gr, cx + gr, cy + gr)
            c.itemconfigure(item, outline=shade)

        # ---- Nameplate: swap to the pre-rendered glow image for the current
        # speech level. Fast attack/decay -> a quick neon pulse (brighten fast,
        # fade fast) tracking his syllables. ----
        tgt = min(1.0, target_level * 1.3)
        self.cur_glow += (tgt - self.cur_glow) * (0.85 if tgt > self.cur_glow else 0.42)
        if self.plate_img_item is not None and self.plate_imgs:
            b = int(round(self.cur_glow * (len(self.plate_imgs) - 1)))
            b = max(0, min(len(self.plate_imgs) - 1, b))
            if b != self._last_bucket:
                self._last_bucket = b
                c.itemconfigure(self.plate_img_item, image=self.plate_imgs[b])
        elif self.fallback_text is not None:
            c.itemconfigure(
                self.fallback_text,
                fill=_lerp_color(self.CORE_IDLE, self.NEON_RED, self.cur_glow))


# ---------------------------------------------------------------------------
# Process-wide singleton + thin module-level helpers
# ---------------------------------------------------------------------------

_SINK: Optional[WaveformSink] = None
_SINK_LOCK = threading.Lock()


def get_waveform_sink() -> WaveformSink:
    """Return the shared :class:`WaveformSink`, creating it on first use."""
    global _SINK
    if _SINK is None:
        with _SINK_LOCK:
            if _SINK is None:
                _SINK = WaveformSink()
    return _SINK


def submit(pcm: np.ndarray, sample_rate: int) -> None:
    """Module-level tee used by the playback engines + relay path. Cheap no-op
    when the overlay is off, so engines can call it unconditionally."""
    sink = _SINK
    if sink is None or not sink._enabled:  # noqa: SLF001 - fast path
        return
    sink.submit(pcm, sample_rate)


def configure_from_config() -> None:
    """(Re)read the ``visualizer`` config block and apply it. Called at
    orchestrator startup and on live GUI changes. Fail-open."""
    try:
        from kenning.config import get_config

        v = get_config().visualizer
    except Exception as e:  # noqa: BLE001
        logger.debug("waveform configure_from_config: config read failed (%s)", e)
        return
    try:
        get_waveform_sink().configure(
            enabled=bool(getattr(v, "enabled", False)),
            size=getattr(v, "size", None),
            bars=getattr(v, "bars", None),
            fps=getattr(v, "fps", None),
            bg_color=getattr(v, "bg_color", None),
            accent_color=getattr(v, "accent_color", None),
            transparent=getattr(v, "transparent", None),
            always_on_top=getattr(v, "always_on_top", None),
            nameplate_text=getattr(v, "nameplate_text", None),
            nameplate_font=getattr(v, "nameplate_font", None),
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("waveform configure apply failed (%s)", e)
