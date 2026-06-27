"""Tests for the voice waveform overlay sink (no Tk window is created here)."""
from __future__ import annotations

import threading
import time

import numpy as np

from kenning.audio import waveform as wf


def _speechy(sr=24000, secs=1.0):
    t = np.arange(int(sr * secs)) / sr
    sig = sum(np.sin(2 * np.pi * f * t) for f in (200, 600, 1800))
    env = (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) ** 2
    return ((sig / 3.0) * env * 0.5 * 32767).astype(np.int16)


def test_analyze_clip_returns_frames():
    frames = wf.analyze_clip(_speechy(), 24000, fps=30, n_bands=60)
    assert len(frames) > 10
    level, bands = frames[len(frames) // 2]
    assert 0.0 <= level <= 1.0
    assert bands.shape == (60,)
    assert float(bands.max()) <= 1.0 + 1e-6
    assert float(bands.max()) > 0.0          # a voiced frame moves the bars


def test_analyze_clip_silence_is_calm():
    sil = np.zeros(24000, dtype=np.int16)
    frames = wf.analyze_clip(sil, 24000, fps=30, n_bands=48)
    # Silence still yields frames, but levels/bands stay ~0 (calm, not slammed).
    assert frames
    assert max(l for l, _ in frames) < 0.05
    assert max(float(b.max()) for _, b in frames) < 0.05


def test_analyze_clip_fail_open_on_garbage():
    assert wf.analyze_clip(np.array([], dtype=np.int16), 24000, fps=30, n_bands=60) == []
    assert wf.analyze_clip(np.zeros(4, dtype=np.int16), 0, fps=30, n_bands=60) == []


def test_submit_is_noop_when_disabled():
    sink = wf.WaveformSink()
    assert sink.enabled is False
    sink.submit(_speechy(), 24000)
    assert sink._queue.qsize() == 0          # nothing enqueued while off


def test_submit_enqueues_and_drops_oldest_when_enabled():
    sink = wf.WaveformSink()
    sink._enabled = True                     # flip flag WITHOUT starting the UI thread
    for _ in range(wf._QUEUE_MAXSIZE + 5):
        sink.submit(_speechy(secs=0.2), 24000)
    # Bounded queue: never exceeds maxsize (drop-oldest keeps newest).
    assert sink._queue.qsize() <= wf._QUEUE_MAXSIZE


def test_submit_copies_buffer():
    sink = wf.WaveformSink()
    sink._enabled = True
    buf = _speechy(secs=0.2)
    sink.submit(buf, 24000)
    buf[:] = 0                               # mutate caller buffer after submit
    queued, _sr = sink._queue.get_nowait()
    assert queued.any()                      # the sink kept its own copy


def test_module_submit_fast_path_no_sink():
    # With no global sink built, module submit must be a cheap no-op (no raise).
    wf._SINK = None
    wf.submit(_speechy(), 24000)             # should not raise / not build a sink
    assert wf._SINK is None


def test_pacer_survives_stale_sentinel():
    """A leftover None (e.g. from a prior disable) must NOT kill a fresh pacer
    -- only a None WHILE _stop is set ends it. (Regression: re-enable made the
    pacer read a stale sentinel and exit immediately.)"""
    sink = wf.WaveformSink()
    sink._enabled = True
    sink._stop.clear()
    th = threading.Thread(target=sink._pace_loop, daemon=True)
    th.start()
    try:
        sink._queue.put((_speechy(secs=0.1), 24000))
        time.sleep(0.1)
        sink._queue.put(None)                # stale sentinel, _stop NOT set
        time.sleep(0.15)
        assert th.is_alive(), "pacer exited on a stale sentinel"
    finally:
        sink._stop.set()
        sink._queue.put(None)                # real stop
        th.join(timeout=2.0)
    assert not th.is_alive()


def test_teardown_drains_queue_and_clears_threads():
    sink = wf.WaveformSink()
    sink._enabled = True
    for _ in range(3):
        sink.submit(_speechy(secs=0.1), 24000)
    assert sink._queue.qsize() > 0
    sink._teardown()                         # no threads running -> just drains
    assert sink._queue.qsize() == 0
    assert sink._pacer is None and sink._ui is None


# ---------------------------------------------------------------------------
# HAL-9000 eye (the centre "core"): dim red ember when idle, lit + white-hot
# glow when speaking, with the glow intensity driven by the live audio level.
# Pure / display-free so it runs headless (no Tk window, no display).
# ---------------------------------------------------------------------------


def test_eye_idle_is_dim_red():
    """At rest the eye is a dark, dim blood-red ember -- low glow, dull pupil,
    and a red-dominant (not white) iris."""
    e = wf._eye_appearance(0.0)
    assert e["glow"] < 0.2                    # halo barely there at idle
    assert e["iris"][0] > e["iris"][1] + 8    # red channel dominates -> RED eye
    assert e["iris"][0] > e["iris"][2] + 8
    # The idle iris is DARK (a dim ember, not a bright lens).
    assert e["iris"][0] < 90
    # Idle pupil is a dull ember, nowhere near white-hot.
    assert e["core"][0] < 160


def test_eye_speaking_is_bright_and_hotter_than_idle():
    """Speaking lights the eye up: higher glow, a brighter/redder iris, and a
    near-white-hot pupil. Every brightness axis must beat the idle value."""
    idle = wf._eye_appearance(0.0)
    talk = wf._eye_appearance(1.0)
    assert talk["glow"] > idle["glow"]                 # halo blooms
    assert sum(talk["iris"]) > sum(idle["iris"])       # iris lights up
    assert sum(talk["core"]) > sum(idle["core"])       # pupil flares
    assert talk["pupil_frac"] > idle["pupil_frac"]     # pupil dilates
    # The speaking pupil is a white-hot pinpoint.
    assert min(talk["core"]) > 200
    # And the iris is clearly RED (red channel still dominant when lit).
    assert talk["iris"][0] > talk["iris"][1] + 50


def test_eye_glow_scales_monotonically_with_amplitude():
    """The eye glow is amplitude-driven: louder speech -> brighter glow + a
    bigger hot pupil (a strict ramp from idle to full)."""
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    glows = [wf._eye_appearance(x)["glow"] for x in levels]
    pupils = [wf._eye_appearance(x)["pupil_frac"] for x in levels]
    assert glows == sorted(glows) and glows[0] < glows[-1]
    assert pupils == sorted(pupils) and pupils[0] < pupils[-1]


def test_eye_appearance_clamps_out_of_range_level():
    """Fail-safe: junk levels clamp into [0,1] rather than producing wild
    colours that could crash the Tk fill conversion."""
    lo = wf._eye_appearance(-5.0)
    hi = wf._eye_appearance(9.9)
    assert lo == wf._eye_appearance(0.0)
    assert hi == wf._eye_appearance(1.0)
    for col in (*lo["iris"], *lo["core"], *hi["iris"], *hi["core"]):
        assert 0 <= col <= 255


class _FakeCanvas:
    """A minimal stand-in for tk.Canvas: records created items + their config so
    the renderer can be exercised with NO display (headless). Each create_*
    returns a unique int id; coords/itemconfigure just store the last values."""

    def __init__(self):
        self._n = 0
        self.fills = {}

    def _new(self):
        self._n += 1
        return self._n

    def create_oval(self, *a, **k):
        i = self._new(); self.fills[i] = k.get("fill"); return i

    def create_line(self, *a, **k):
        i = self._new(); self.fills[i] = k.get("fill"); return i

    def create_arc(self, *a, **k):
        return self._new()

    def create_image(self, *a, **k):
        return self._new()

    def create_text(self, *a, **k):
        return self._new()

    def coords(self, *a, **k):
        return None

    def itemconfigure(self, item, **k):
        if "fill" in k:
            self.fills[item] = k["fill"]


def test_render_eye_lights_up_when_speaking_headless():
    """End-to-end (no display): build the renderer against a fake canvas, then
    render an idle frame and a loud frame; the iris + pupil fill must be
    BRIGHTER when speaking than when idle. Proves the audio level actually
    drives the eye through the real render() path."""
    canvas = _FakeCanvas()
    state = wf._RenderState(canvas, size=300, plate_h=0, bars=40,
                            accent="#e5484d", bg="#0b0b10",
                            nameplate_text="", fps=30)
    state.build()
    zero = np.zeros(40, dtype=np.float32)
    loud = np.ones(40, dtype=np.float32)

    def _brightness(item):
        h = canvas.fills[item].lstrip("#")
        return int(h[0:2], 16) + int(h[2:4], 16) + int(h[4:6], 16)

    # Settle to idle (level eases toward 0), then sample the eye.
    for _ in range(40):
        state.render(0.0, zero)
    iris_idle = _brightness(state.eye_iris)
    pupil_idle = _brightness(state.eye_pupil)

    # Drive a loud speaking frame repeatedly so the eased level rises.
    for _ in range(40):
        state.render(1.0, loud)
    iris_talk = _brightness(state.eye_iris)
    pupil_talk = _brightness(state.eye_pupil)

    assert iris_talk > iris_idle, "iris must light up when speaking"
    assert pupil_talk > pupil_idle, "pupil must flare white-hot when speaking"


def test_render_snaps_legacy_accent_to_valorant_red():
    """The default (legacy Kenning crimson) accent is snapped to the #ff4655
    Valorant red so the new red machine look is the default; a custom accent is
    honoured untouched."""
    c1 = _FakeCanvas()
    s1 = wf._RenderState(c1, size=200, plate_h=0, bars=16,
                         accent="#e5484d", bg="#0b0b10", nameplate_text="")
    assert s1.accent_rgb == wf.VALORANT_RED        # legacy -> red

    c2 = _FakeCanvas()
    s2 = wf._RenderState(c2, size=200, plate_h=0, bars=16,
                         accent="#00ffcc", bg="#0b0b10", nameplate_text="")
    assert s2.accent_rgb == (0x00, 0xff, 0xcc)     # custom honoured


def test_submit_then_pace_does_not_raise_headless():
    """A submitted clip flows through the pacer + analyze path without raising
    and publishes a non-trivial target level (no display involved)."""
    sink = wf.WaveformSink()
    sink._enabled = True
    sink._stop.clear()
    sink.submit(_speechy(secs=0.3), 24000)
    th = threading.Thread(target=sink._pace_loop, daemon=True)
    th.start()
    try:
        deadline = time.time() + 2.0
        seen = 0.0
        while time.time() < deadline:
            seen = max(seen, float(sink._target_level))
            if seen > 0.0:
                break
            time.sleep(0.02)
        assert seen > 0.0, "pacer never published a speaking level"
    finally:
        sink._stop.set()
        sink._queue.put(None)
        th.join(timeout=2.0)
