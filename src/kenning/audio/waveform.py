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


# ---------------------------------------------------------------------------
# Valorant-red "Ultron machine" palette -- the cold angular #ff4655 tech look
# the rest of the on-stream overlay uses. The waveform + HAL eye are tuned to
# THESE so the visualizer reads as one cohesive red machine.
# ---------------------------------------------------------------------------
VALORANT_RED = (255, 70, 85)        # #ff4655 -- the signature accent
EYE_IRIS_IDLE = (46, 6, 10)         # dim, dark blood-red -- the eye at rest
EYE_IRIS_HOT = (255, 60, 70)        # lit iris when speaking
EYE_CORE_IDLE = (90, 14, 18)        # dull ember at the pupil when idle
EYE_CORE_HOT = (255, 248, 244)      # white-yellow-hot pinpoint when speaking
EYE_BEZEL = (118, 124, 132)         # brushed-chrome housing ring
EYE_BEZEL_HI = (196, 202, 210)      # chrome highlight glint
EYE_LENS = (8, 4, 6)                # near-black lens recess


def _eye_appearance(level: float) -> dict:
    """Pure, display-free model of the HAL-9000 eye at a given speech ``level``
    (0 = idle/quiet .. 1 = full speech). Returns the colours + glow scalars the
    renderer paints, so the idle-vs-speaking look can be unit-tested with no Tk.

    Idle  -> a low, dark-red ember (dim lens glow, dull pupil).
    Speak -> the iris lights up red and the pupil flashes white-hot, with the
             halo bloom intensity rising with ``level`` (amplitude-driven).
    """
    level = 0.0 if level < 0 else 1.0 if level > 1 else float(level)
    # Iris brightens from dim blood-red toward a lit red as he speaks; a touch
    # of ease so quiet speech already clearly lifts it off the idle floor.
    iris_t = level ** 0.8
    iris = _lerp_rgb(EYE_IRIS_IDLE, EYE_IRIS_HOT, iris_t)
    # Pupil: dull ember at rest, ramps to a white-hot pinpoint once he's loud.
    core_t = level ** 1.5
    core = _lerp_rgb(EYE_CORE_IDLE, EYE_CORE_HOT, core_t)
    # Halo bloom: barely-there at idle, swells with amplitude. Drives both the
    # number of visible glow rings' alpha-feel (via colour) and their radius.
    glow = 0.10 + 0.90 * level
    # Pupil radius (fraction of the eye radius): a small hot dot that grows as
    # the centre flares.
    pupil_frac = 0.12 + 0.16 * level
    return {
        "iris": iris,
        "core": core,
        "glow": glow,
        "pupil_frac": pupil_frac,
        "level": level,
    }


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
                      core_idle, neon_red, buckets, plate_alpha=255):
    """Render the ULTRON plate as ``buckets`` RGBA PIL frames from idle (calm,
    readable) to full speech (bright neon + soft Gaussian bloom). The glow is a
    real blur -> rounded, soft, particle-like halo (like a neon tube), in the
    SAME red the glyphs light up to. The plate is an OPAQUE dark backing
    (``plate_alpha`` fully opaque by default): a PARTIAL alpha gets faked by
    compositing against the green chroma window bg, tinting the panel green so
    OBS's chroma key removes it -- an opaque neutral fill survives the key and
    still reads as smoked glass. Returns a list of PIL.Image (RGBA)."""
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
    plate_rgba = (*plate_fill, int(plate_alpha))      # semi-transparent backing
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


def _radial_wave_points(cx, cy, base_r, bands, *, n_pts=96, amp=0.0,
                        ripple=0.0, phase=0.0):
    """Sample a CIRCULAR WAVEFORM: a closed ring of ``n_pts`` (x, y) points
    whose radius is ``base_r`` modulated by the live audio ``bands`` so the edge
    ripples and represents the spoken words.

    The per-band FFT magnitudes are wrapped symmetrically around the full circle
    (mirrored so the ring joins seamlessly at 0/2pi) and sampled with linear
    interpolation, then displaced radially. ``amp`` scales how far the bands push
    the edge out (the speech "ripple depth"); ``ripple`` adds a travelling sine
    so the wave visibly emanates outward; ``phase`` advances that travel each
    frame. At idle (amp~0, ripple~0) every point sits at ``base_r`` -> a calm,
    near-flat circle; speaking makes the band-shaped ripples bulge + breathe.

    Returns a flat ``[x0, y0, x1, y1, ...]`` list ready for
    ``create_polygon(smooth=True)``. Pure + display-free so it unit-tests
    headless. Fail-open: any anomaly falls back to a plain circle.
    """
    try:
        b = np.asarray(bands, dtype=np.float32).ravel()
        nb = b.shape[0]
        if nb == 0:
            b = np.zeros(1, dtype=np.float32)
            nb = 1
        # Mirror the band envelope so the profile is seamless across the wrap
        # point (the last sample meets the first): bands 0..nb-1..0.
        prof = np.concatenate([b, b[::-1]]) if nb > 1 else b
        m = prof.shape[0]
        pts: List[float] = []
        two_pi = 2.0 * math.pi
        for i in range(n_pts):
            frac = i / n_pts                       # 0..1 around the circle
            # Interpolate the (mirrored) band envelope at this angle.
            fpos = frac * m
            i0 = int(fpos) % m
            i1 = (i0 + 1) % m
            w = fpos - math.floor(fpos)
            band_val = float(prof[i0] * (1.0 - w) + prof[i1] * w)
            ang = two_pi * frac
            # Travelling ripple: a few cycles of sine riding outward (phase) so
            # the ring reads as a sound-wave emanating, scaled by loudness.
            trav = math.sin(ang * 5.0 - phase) * ripple
            r = base_r * (1.0 + amp * band_val + trav)
            pts.append(cx + math.cos(ang) * r)
            pts.append(cy + math.sin(ang) * r)
        return pts
    except Exception:  # noqa: BLE001 - never break the render loop
        # Plain circle fallback.
        pts = []
        for i in range(n_pts):
            ang = 2.0 * math.pi * i / n_pts
            pts.append(cx + math.cos(ang) * base_r)
            pts.append(cy + math.sin(ang) * base_r)
        return pts


def _round_rect_points(x0, y0, x1, y1, r):
    """Point list for a rounded rectangle, used with create_polygon(smooth=True)."""
    return [
        x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r,
        x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
        x0, y1, x0, y1 - r, x0, y0 + r, x0, y0,
    ]


def _set_overlay_window_styles(hwnd_int: int, *, background: bool) -> None:
    """Windows: make the borderless overlay show up in OBS's Window Capture list
    and, optionally, behave as an unobtrusive background window.

    Tk's ``overrideredirect`` auto-applies ``WS_EX_TOOLWINDOW`` -- and OBS
    filters tool windows OUT of its window list, so the overlay never appears as
    a capture target. We therefore CLEAR ``WS_EX_TOOLWINDOW`` and set
    ``WS_EX_APPWINDOW`` (taskbar-present + enumerable) so OBS lists it. When
    ``background`` is True we additionally set ``WS_EX_NOACTIVATE`` so it never
    steals focus and can sink behind your other windows (OBS's
    'Windows 10 (1903+)' capture still grabs it when occluded). The window is
    re-mapped (hide/show) so the APPWINDOW change registers.

    IMPORTANT: this takes a raw HWND (not the Tk root) and MUST be called from a
    thread OTHER than the Tk mainloop thread. Calling ShowWindow from inside the
    mainloop makes Tk's own window-proc re-assert ``overrideredirect`` (re-adding
    WS_EX_TOOLWINDOW) reentrantly, reverting the fix; from another thread the
    re-map is processed cleanly (this is why the change sticks). Fail-open /
    no-op off Windows; no admin rights needed.
    """
    try:
        import sys
        if not sys.platform.startswith("win"):
            return
        import ctypes

        GWL_EXSTYLE = -20
        WS_EX_TOOLWINDOW = 0x00000080
        WS_EX_APPWINDOW = 0x00040000
        WS_EX_NOACTIVATE = 0x08000000
        SW_HIDE, SW_SHOW, SW_SHOWNA = 0, 5, 8
        u32 = ctypes.windll.user32
        u32.GetWindowLongW.restype = ctypes.c_long
        u32.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
        u32.SetWindowLongW.restype = ctypes.c_long
        u32.SetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
        u32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
        u32.GetAncestor.restype = ctypes.c_void_p
        u32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        # Tk's winfo_id() can be the CONTENT child; the OS top-level window (the
        # one the WM/OBS see and that carries WS_EX_TOOLWINDOW) is its GA_ROOT.
        GA_ROOT = 2
        root_hwnd = u32.GetAncestor(ctypes.c_void_p(int(hwnd_int)), GA_ROOT)
        hwnd = ctypes.c_void_p(root_hwnd if root_hwnd else int(hwnd_int))
        ex = u32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ex = (ex & ~WS_EX_TOOLWINDOW) | WS_EX_APPWINDOW   # OBS-enumerable
        if background:
            ex |= WS_EX_NOACTIVATE
        u32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex)
        u32.ShowWindow(hwnd, SW_HIDE)
        u32.ShowWindow(hwnd, SW_SHOWNA if background else SW_SHOW)
        if background:
            # Sink the window to the BOTTOM of the z-order so it hides BEHIND
            # your desktop windows / game (not distracting), while OBS Window
            # Capture ("Windows 10 (1903+)" / WGC) still grabs its live pixels
            # even when fully occluded. No admin needed for this -- WGC reads
            # the window's own framebuffer regardless of z-order.
            HWND_BOTTOM = ctypes.c_void_p(1)
            SWP_NOMOVE, SWP_NOSIZE, SWP_NOACTIVATE = 0x0002, 0x0001, 0x0010
            u32.SetWindowPos.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_uint]
            u32.SetWindowPos(hwnd, HWND_BOTTOM, 0, 0, 0, 0,
                             SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)
    except Exception as e:  # noqa: BLE001
        logger.debug("overlay window styles not applied (%s)", e)


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
            # Plate rides at ~0.82*size (see _RenderState.plate_top) instead of
            # below a full square, so the window is shorter (less screen space).
            height = (int(round(size * 0.82)) + plate_h) if plate_text else size
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
            # Make the borderless window OBS-capturable: clear the auto-applied
            # WS_EX_TOOLWINDOW (OBS filters tool windows out of its list) + set
            # WS_EX_APPWINDOW. Done from a SEPARATE one-shot thread on the raw
            # HWND after the window settles -- doing it on the Tk thread makes
            # Tk re-assert overrideredirect (re-adding TOOLWINDOW). In background
            # mode it also sinks behind other windows and never steals focus.
            root.update_idletasks()
            _hwnd = root.winfo_id()
            _bg_mode = not self._always_on_top

            def _apply_styles_async():
                time.sleep(0.4)
                _set_overlay_window_styles(_hwnd, background=_bg_mode)
            threading.Thread(target=_apply_styles_async, daemon=True,
                             name="waveform-winstyle").start()
            canvas = tk.Canvas(
                root, width=size, height=height, bg=self._bg,
                highlightthickness=0, bd=0)
            canvas.pack(fill="both", expand=True)

            state = _RenderState(canvas, size, plate_h, self._bars, self._accent,
                                 self._bg, plate_text, self._nameplate_font,
                                 fps=self._fps)
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
                 bg: str, nameplate_text: str = "", nameplate_font: str = "Bahnschrift",
                 fps: int = 60) -> None:
        self.canvas = canvas
        self.size = size
        self.plate_h = plate_h
        self.bars = bars
        # The visualizer reads as the Valorant-red "Ultron machine": the bars +
        # rings ride from the dark accent toward a hot red-white. We honour a
        # custom accent from config but, if it's still the legacy Kenning
        # crimson, snap it to the #ff4655 signature so the default look matches
        # the rest of the on-stream overlay.
        acc = _hex_to_rgb(accent)
        if acc == (0xE5, 0x48, 0x4D):     # legacy default -> the new red
            acc = VALORANT_RED
        self.accent_rgb = acc
        # Bar tips flash a hot red-white (not pure white) so peaks stay inside
        # the red machine aesthetic instead of going icy/neon-blue-white.
        self.tip_rgb = (255, 196, 200)
        self.bg = bg
        # The radial art fades toward this DARK base (not the canvas bg) so a
        # chroma-key background (e.g. neon green) never bleeds into the glow as
        # an un-keyable olive mid-tone. Only the empty canvas bg is the key.
        self.art_base = (16, 6, 10)
        self.cx = size / 2.0
        self.cy = size / 2.0
        self.r0 = size * 0.20          # inner ring radius
        # Max bar tip: tightened 0.46 -> 0.40 so the pulse stays close to the
        # rings instead of fanning out across the whole canvas.
        self.r_max = size * 0.40       # max bar tip
        # The nameplate sits at this y (top of the plate band). Raised well into
        # the lower square -- the speaking bars fan out everywhere EXCEPT
        # straight down (see dir_gain in render), so the plate can ride high
        # with only a small gap and never get covered by the animation.
        self.plate_top = int(round(size * 0.82))
        self.cur_level = 0.0
        self.cur_bands = np.zeros(bars, dtype=np.float32)
        self.angle = 0.0    # breath/shimmer clock -- always advances
        self.spin = 0.0     # bar-ring rotation -- ALWAYS advances (continuous spin)
        # Motion is tuned as per-frame steps against a 30 fps reference; scale
        # the steps by the real fps so the spin + breath run at the SAME speed
        # whether the overlay redraws at 30 or 60. A higher fps then just makes
        # the identical motion smoother, never faster.
        self._mscale = 30.0 / float(max(1, fps))
        self.drag_x = 0
        self.drag_y = 0
        # Concentric circular speech-waveforms ("the expanding part"): each is a
        # smooth closed polygon whose edge ripples from the live per-band audio
        # and which expands outward as he speaks. n_pts samples around the circle
        # -> the ring's outline; n_rings rings emanate at increasing radius.
        self.glow_items: list = []
        self.n_rings = 3
        self.n_pts = 96
        self.wave_phase = 0.0               # outward-travel clock for the ripple
        self.bar_outline_items: list = []   # black underlay -> crisp bar edges
        self.bar_items: list = []
        # ---- HAL-9000 eye (the centre "core") ----
        # A glowing red lens set in a brushed-chrome ring: outer halos that
        # bloom with amplitude, a chrome bezel, a black lens recess, the red
        # iris, fine radial "mechanical" lines from the centre, and a white-hot
        # pupil pinpoint. DIM dark-red at idle; lights up + flares hot when he
        # speaks (glow scales with the live audio level). All on the SAME
        # tkinter Canvas -- no new deps.
        self.eye_halo_items: list = []      # soft red bloom rings (behind)
        self.eye_bezel = None               # brushed-chrome housing ring
        self.eye_bezel_hi = None            # chrome highlight arc
        self.eye_lens = None                # black lens recess
        self.eye_iris = None                # the red iris disc
        self.eye_ray_items: list = []       # fine radial lines from centre
        self.eye_pupil = None               # white-hot centre pinpoint
        self.eye_rays = 28                  # radial-line count
        self.core = None                    # legacy alias (kept = eye_iris)
        # Nameplate (ULTRON): a dark plate (contrast over gameplay) with a REAL
        # Gaussian-blurred neon glow -- a bright tube core + soft, rounded halo
        # in the SAME red the glyphs light up to -- pre-rendered with PIL at N
        # brightness buckets and swapped per frame on a fast attack/decay
        # envelope (quick brighten, quick fade). Falls back to plain canvas text
        # if PIL is unavailable.
        self.text = (nameplate_text or "").strip()
        self.font_family = nameplate_font or "Bahnschrift"
        # Smoked-glass nameplate -- OPAQUE dark panel. It MUST be opaque because
        # the overlay is captured via an OBS CHROMA KEY: a semi-transparent fill
        # is faked by Tk compositing the panel against the GREEN window bg, which
        # tints the panel green, so OBS's key then removes it along with the
        # background (it shows on the desktop popup but VANISHES in OBS). A solid
        # neutral dark fill (far from the green key) survives the key cleanly and
        # still reads as smoked glass with the neon glyphs floating on it.
        self.PLATE_FILL = (22, 18, 28)    # dark smoked charcoal (away from green)
        self.PLATE_ALPHA = 255            # OPAQUE -> no green bleed, survives key
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
        # Outer "speech rings" (drawn first, behind everything): concentric
        # CIRCULAR WAVEFORMS, not plain circles. Each is a closed smooth polygon
        # whose radius ripples around the circle from the live per-band audio --
        # a calm near-flat ring at idle, rippling + expanding outward while he
        # speaks (see _radial_wave_points + the render loop). create_polygon with
        # smooth=True turns the sampled points into a soft, rounded waveform.
        for _ in range(self.n_rings):
            self.glow_items.append(
                c.create_polygon(0, 0, 0, 0, 0, 0, outline=self.bg, width=2,
                                 fill="", smooth=True, splinesteps=12))
        # Radial bars: a black underlay line (slightly wider, drawn FIRST so it
        # sits behind) gives every neon bar a thin black outline -> the energy
        # reads crisply over busy gameplay for stream viewers.
        for _ in range(self.bars):
            self.bar_outline_items.append(
                c.create_line(0, 0, 0, 0, fill="#000000", width=5,
                              capstyle="round"))
        for _ in range(self.bars):
            self.bar_items.append(
                c.create_line(0, 0, 0, 0, fill=self.bg, width=3,
                              capstyle="round"))
        # ---- HAL-9000 eye, built outward-in so the bright centre lands on top.
        # Soft red bloom halos (behind the bezel) -- expand + brighten with
        # speech amplitude.
        for _ in range(4):
            self.eye_halo_items.append(
                c.create_oval(0, 0, 0, 0, outline="", fill=self.bg))
        # Brushed-chrome housing ring + a highlight arc for the metal glint.
        self.eye_bezel = c.create_oval(
            0, 0, 0, 0, outline=_rgb_to_hex(EYE_BEZEL), width=2, fill="#000000")
        self.eye_bezel_hi = c.create_arc(
            0, 0, 0, 0, start=58, extent=86, style="arc",
            outline=_rgb_to_hex(EYE_BEZEL_HI), width=2)
        # Black lens recess the red iris sits inside.
        self.eye_lens = c.create_oval(
            0, 0, 0, 0, outline="#000000", width=1, fill=_rgb_to_hex(EYE_LENS))
        # The red iris disc (dim idle -> lit when speaking).
        self.eye_iris = c.create_oval(
            0, 0, 0, 0, outline="", fill=_rgb_to_hex(EYE_IRIS_IDLE))
        self.core = self.eye_iris           # legacy alias
        # Fine radial "mechanical eye" lines fanning out from the centre.
        for _ in range(self.eye_rays):
            self.eye_ray_items.append(
                c.create_line(0, 0, 0, 0, fill=_rgb_to_hex(EYE_IRIS_IDLE),
                              width=1))
        # White-hot pupil pinpoint (the bright HAL centre).
        self.eye_pupil = c.create_oval(
            0, 0, 0, 0, outline="", fill=_rgb_to_hex(EYE_CORE_IDLE))
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
            buckets=self._glow_buckets, plate_alpha=self.PLATE_ALPHA)
        self.plate_imgs = [ImageTk.PhotoImage(f) for f in frames]
        cx = self.size / 2.0
        cy = self.plate_top + self.plate_h / 2.0
        self.plate_img_item = self.canvas.create_image(
            cx, cy, image=self.plate_imgs[0])

    def _build_nameplate_fallback(self) -> None:
        """Plain crisp name (no plate/glow) if PIL isn't available."""
        cx = self.size / 2.0
        cy = self.plate_top + self.plate_h / 2.0
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
        # Idle breathing so it's never fully dead on screen, plus a CONTINUOUS
        # ring spin. The spin ALWAYS advances -- a slow drift at rest that speeds
        # up smoothly while Ultron speaks -- so the overlay stays alive even when
        # idle. (An earlier optimization froze the spin at idle to save GPU; it
        # made the overlay look dead and the breathing choppy/low-fps, so it's
        # restored here. A tiny tkinter canvas costs nothing to spin.)
        breath = 0.05 * (0.5 + 0.5 * math.sin(self.angle * 1.7))
        self.angle += 0.018 * self._mscale
        self.spin += (0.018 + 0.020 * min(1.0, self.cur_level * 2.2)) * self._mscale
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
            ang = self.spin + (2.0 * math.pi * i / n)
            ca, sa = math.cos(ang), math.sin(ang)
            inner = r0 + 3.0
            # Fan out everywhere except straight DOWN (toward the nameplate):
            # sa>0 points down in screen coords, so taper those bars hard. This
            # frees the lower area so the plate can sit close above.
            dir_gain = 1.0 - 0.78 * max(0.0, sa)
            outer = r0 + 6.0 + amp * (r_max - r0) * dir_gain
            x0, y0 = cx + ca * inner, cy + sa * inner
            x1, y1 = cx + ca * outer, cy + sa * outer
            # Travelling shimmer highlight sweeps around the ring (a spectrum
            # glint), and peaks flash white-hot -- cool motion without clutter.
            # Shimmer phase rides the gated spin (not the breath clock) so it
            # travels at the identical rate while speaking but freezes at idle.
            shimmer = 0.5 + 0.5 * math.sin(ang * 2.0 - self.spin * 3.2)
            hot = min(1.0, amp * 1.2 + 0.22 * shimmer * level)
            col_rgb = _lerp_rgb(accent, tip, hot)
            if amp > 0.62:
                col_rgb = _lerp_rgb(col_rgb, (255, 255, 255), (amp - 0.62) * 0.9)
            col = _rgb_to_hex(col_rgb)
            # Loud bars get a touch thicker -> the energy reads as "fatter".
            bw = max(2, int(self.size * (0.011 + 0.006 * min(1.0, amp))))
            ow = bw + max(2, int(self.size * 0.006))   # black outline, a bit wider
            c.coords(self.bar_outline_items[i], x0, y0, x1, y1)
            c.itemconfigure(self.bar_outline_items[i], width=ow)
            c.coords(self.bar_items[i], x0, y0, x1, y1)
            c.itemconfigure(self.bar_items[i], fill=col, width=bw)
        # ---- HAL-9000 eye: dim red lens at idle, lit + white-hot when speaking.
        # The eye sits inside the bar ring (radius r0) and is the focal point.
        eye = _eye_appearance(level)
        glow = eye["glow"]
        # The whole eye breathes a little so it's alive at rest, and dilates
        # slightly as he speaks (a touch of life, not a big pulse).
        er = r0 * (0.92 + 0.10 * level)
        # Soft red bloom halos behind the bezel -- expand + brighten with level
        # so the eye visibly GLOWS outward only while he talks.
        nhalo = len(self.eye_halo_items)
        for k, item in enumerate(self.eye_halo_items):
            spread = 1.0 + (0.20 + 0.85 * glow) * (k + 1) / nhalo
            hr = er * spread
            # Outer halos are fainter; all fade toward the dark art_base at idle
            # (no olive bleed on a green key) and toward lit red as he speaks.
            halo_t = glow * (1.0 - 0.62 * k / max(1, nhalo - 1))
            hcol = _lerp_rgb(self.art_base, EYE_IRIS_HOT, max(0.0, halo_t))
            c.coords(item, cx - hr, cy - hr, cx + hr, cy + hr)
            c.itemconfigure(item, fill=_rgb_to_hex(hcol))
        # Brushed-chrome bezel ring + highlight glint (the housing).
        c.coords(self.eye_bezel, cx - er, cy - er, cx + er, cy + er)
        bez = _lerp_rgb(EYE_BEZEL, EYE_BEZEL_HI, 0.25 + 0.4 * level)
        c.itemconfigure(self.eye_bezel, outline=_rgb_to_hex(bez),
                        width=max(2, int(self.size * (0.010 + 0.004 * level))))
        hi = er * 0.995
        c.coords(self.eye_bezel_hi, cx - hi, cy - hi, cx + hi, cy + hi)
        # Black lens recess.
        lr = er * 0.86
        c.coords(self.eye_lens, cx - lr, cy - lr, cx + lr, cy + lr)
        # Red iris -- the glowing lens. Dim dark-red idle -> lit red speaking.
        ir = er * 0.80
        c.coords(self.eye_iris, cx - ir, cy - ir, cx + ir, cy + ir)
        c.itemconfigure(self.eye_iris, fill=_rgb_to_hex(eye["iris"]))
        # Fine radial "mechanical eye" lines from a white-hot centre outward.
        # They light up from the dim iris toward bright red as he speaks and
        # creep with the ring spin so the eye feels mechanical/alive.
        ray_rgb = _lerp_rgb(eye["iris"], EYE_IRIS_HOT, 0.4 + 0.6 * level)
        ray_col = _rgb_to_hex(ray_rgb)
        ray_in = ir * (0.30 + 0.10 * level)
        ray_out = ir * 0.97
        nr = len(self.eye_ray_items)
        for k, item in enumerate(self.eye_ray_items):
            ra = self.spin * 0.5 + (2.0 * math.pi * k / max(1, nr))
            rca, rsa = math.cos(ra), math.sin(ra)
            c.coords(item,
                     cx + rca * ray_in, cy + rsa * ray_in,
                     cx + rca * ray_out, cy + rsa * ray_out)
            c.itemconfigure(item, fill=ray_col)
        # White-hot pupil pinpoint: dull ember idle -> bright white-hot flare.
        pr = ir * eye["pupil_frac"]
        c.coords(self.eye_pupil, cx - pr, cy - pr, cx + pr, cy + pr)
        c.itemconfigure(self.eye_pupil, fill=_rgb_to_hex(eye["core"]))
        # Circular speech-waveforms: concentric rings that EXPAND outward and
        # whose edges RIPPLE from the live per-band audio so they represent the
        # spoken words (not a plain uniform circle). Each ring's radius is the
        # bar baseline pushed out by level, and its outline is displaced by the
        # band envelope (_radial_wave_points) plus a travelling ripple that
        # emanates outward (wave_phase). Idle -> a calm near-flat circle;
        # speaking -> band-shaped ripples that bulge + breathe + grow.
        #
        # Rings render in the SAME accent red as the bars (visible at idle) and
        # brighten toward white-hot as he speaks. Red is keyable on the green
        # chroma (only the pure-green empty canvas is keyed), so no olive bleed.
        self.wave_phase += (0.12 + 0.55 * self.cur_level) * self._mscale
        # Speech drives BOTH how far the rings sit out and how deep the edge
        # ripples; at idle these collapse toward ~0 -> a calm, near-flat round
        # ring (a whisper of motion so it isn't dead, then it blooms as he
        # speaks). cur_level (not the breath-lifted `level`) gates them so idle
        # truly settles flat.
        ripple_depth = 0.006 + 0.21 * self.cur_level
        amp_depth = 0.012 + 0.36 * self.cur_level
        for k, item in enumerate(self.glow_items):
            # Each ring sits at a growing baseline radius and expands with level.
            gr = r0 + (r_max - r0) * (0.5 + 0.5 * level) * (0.6 + 0.25 * k)
            # Outer rings travel further out and lag in phase so the waves look
            # like they radiate from the eye one after another.
            pts = _radial_wave_points(
                cx, cy, gr, self.cur_bands, n_pts=self.n_pts,
                amp=amp_depth, ripple=ripple_depth * (0.7 + 0.5 * k),
                phase=self.wave_phase - k * 1.1)
            shade = _lerp_color(self.accent_rgb, (255, 255, 255),
                                max(0.0, level - 0.15 * k) * 0.6)
            c.coords(item, *pts)
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
