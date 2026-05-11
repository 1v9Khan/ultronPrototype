"""Audio module tests.

Capture-with-real-mic tests are skipped by default (CI has no audio device);
they only run if PYTEST_RUN_MIC_TESTS=1 is set in the environment.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from ultron.audio.ring_buffer import RingBuffer


# ---- RingBuffer (pure, no I/O) -------------------------------------------


def test_ring_buffer_retains_recent_samples():
    rb = RingBuffer(capacity_samples=10)
    rb.write(np.arange(15, dtype=np.float32))
    snap = rb.snapshot()
    assert snap.shape == (10,)
    assert np.array_equal(snap, np.arange(5, 15, dtype=np.float32))


def test_ring_buffer_clear():
    rb = RingBuffer(capacity_samples=8)
    rb.write(np.ones(4, dtype=np.float32))
    rb.clear()
    assert len(rb) == 0
    assert rb.snapshot().shape == (0,)


def test_ring_buffer_handles_2d_input():
    rb = RingBuffer(capacity_samples=10)
    # sounddevice gives shape (frames, channels); RingBuffer must flatten.
    chunk = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    rb.write(chunk)
    assert np.array_equal(rb.snapshot(), np.array([1, 2, 3], dtype=np.float32))


def test_ring_buffer_rejects_zero_capacity():
    with pytest.raises(ValueError):
        RingBuffer(capacity_samples=0)


# ---- RingBuffer.snapshot(last_n_samples=...) ------------------------------
# Mode-aware pre-roll relies on slicing the same buffer to different
# lengths -- COLD pulls a short slice (no Tron prefix), WARM pulls a
# longer one (no first-word clip). These tests lock the slicing
# semantics down so a future refactor can't quietly regress them.


def test_ring_buffer_snapshot_returns_full_when_unsliced():
    rb = RingBuffer(capacity_samples=10)
    rb.write(np.arange(10, dtype=np.float32))
    snap = rb.snapshot()
    assert snap.shape == (10,)
    assert np.array_equal(snap, np.arange(10, dtype=np.float32))


def test_ring_buffer_snapshot_returns_last_n_samples():
    rb = RingBuffer(capacity_samples=10)
    rb.write(np.arange(10, dtype=np.float32))
    snap = rb.snapshot(last_n_samples=3)
    assert snap.shape == (3,)
    assert np.array_equal(snap, np.array([7.0, 8.0, 9.0], dtype=np.float32))


def test_ring_buffer_snapshot_returns_full_when_n_exceeds_size():
    rb = RingBuffer(capacity_samples=10)
    rb.write(np.arange(5, dtype=np.float32))
    snap = rb.snapshot(last_n_samples=20)
    assert snap.shape == (5,)
    assert np.array_equal(snap, np.arange(5, dtype=np.float32))


def test_ring_buffer_snapshot_returns_empty_when_n_is_zero_or_negative():
    rb = RingBuffer(capacity_samples=10)
    rb.write(np.arange(10, dtype=np.float32))
    assert rb.snapshot(last_n_samples=0).shape == (0,)
    assert rb.snapshot(last_n_samples=-5).shape == (0,)


def test_ring_buffer_cold_warm_slices_share_same_buffer():
    """Concrete scenario: a 0.5 s @ 16 kHz buffer is sliced to 0.15 s
    (COLD) and 0.5 s (WARM). Verifies both slices come from the same
    underlying audio so the orchestrator can use one ring for both
    modes."""
    sr = 16000
    rb = RingBuffer(capacity_samples=int(0.5 * sr))
    audio = np.arange(0.5 * sr, dtype=np.float32)
    rb.write(audio)

    cold = rb.snapshot(last_n_samples=int(0.15 * sr))
    warm = rb.snapshot(last_n_samples=int(0.5 * sr))

    assert cold.shape == (int(0.15 * sr),)
    assert warm.shape == (int(0.5 * sr),)
    # COLD slice is the tail of the WARM slice.
    assert np.array_equal(cold, warm[-int(0.15 * sr):])


# ---- Audio device resolution (pure, no I/O) -------------------------------


def test_resolve_device_matches_name_substring(monkeypatch):
    from ultron.audio import devices

    fake_devices = [
        {"name": "Voicemeeter Out B2", "max_input_channels": 8, "max_output_channels": 0},
        {"name": "Microphone (NVIDIA Broadcast)", "max_input_channels": 2, "max_output_channels": 0},
        {"name": "Headphones (Realtek HD Audio 2nd output)", "max_input_channels": 0, "max_output_channels": 2},
    ]

    def fake_query_devices(device=None, kind=None):
        if device is None:
            return fake_devices
        return fake_devices[int(device)]

    monkeypatch.setattr(devices.sd, "query_devices", fake_query_devices)

    assert devices.resolve_device("nvidia broadcast", "input") == 1
    assert devices.resolve_device("Headphones", "output") == 2


def test_resolve_device_rejects_wrong_direction(monkeypatch):
    from ultron.audio import devices

    fake_devices = [
        {"name": "Microphone", "max_input_channels": 1, "max_output_channels": 0},
    ]

    def fake_query_devices(device=None, kind=None):
        if device is None:
            return fake_devices
        return fake_devices[int(device)]

    monkeypatch.setattr(devices.sd, "query_devices", fake_query_devices)

    with pytest.raises(devices.AudioDeviceError):
        devices.resolve_device("Microphone", "output")


def test_resolve_device_uses_default_index(monkeypatch):
    from ultron.audio import devices

    class FakeDefaultPair:
        def __getitem__(self, index):
            return [0, 1][index]

    class FakeDefault:
        device = FakeDefaultPair()

    fake_devices = [
        {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0},
        {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2},
    ]

    def fake_query_devices(device=None, kind=None):
        if device is None:
            return fake_devices
        return fake_devices[int(device)]

    monkeypatch.setattr(devices.sd, "query_devices", fake_query_devices)
    monkeypatch.setattr(devices.sd, "default", FakeDefault())

    assert devices.resolve_device(None, "input") == 0
    assert devices.resolve_device(None, "output") == 1


# ---- AudioCapture (real device) ------------------------------------------


@pytest.mark.skipif(
    os.environ.get("PYTEST_RUN_MIC_TESTS") != "1",
    reason="set PYTEST_RUN_MIC_TESTS=1 to enable mic tests",
)
def test_audio_capture_produces_chunks():
    from ultron.audio.capture import AudioCapture

    with AudioCapture(blocksize=512) as mic:
        chunk = mic.get_chunk(timeout=2.0)
        assert chunk is not None
        assert chunk.dtype == np.float32
        assert chunk.shape == (512,)


# ---- VAD adaptive silence (no model load required) -----------------------


def _make_vad_without_loading_silero(monkeypatch, min_silence_ms: int = 500):
    """Construct a VoiceActivityDetector with the Silero load patched
    out -- lets us test the silence-window math without needing the
    real model on disk."""
    from ultron.audio import vad as _vad
    monkeypatch.setattr(
        _vad.VoiceActivityDetector,
        "_load_model",
        staticmethod(lambda: object()),
    )
    return _vad.VoiceActivityDetector(min_silence_ms=min_silence_ms)


def test_vad_set_min_silence_duration_ms_changes_window_count(monkeypatch):
    """The adaptive end-of-turn knob lives on the VAD: orchestrator
    calls ``set_min_silence_duration_ms`` mid-utterance and the next
    process() call must use the new requirement."""
    vad = _make_vad_without_loading_silero(monkeypatch, min_silence_ms=500)
    baseline_windows = vad._silence_windows_required
    # 32 ms per window at 16 kHz; 500 ms -> ~15 windows; 2400 ms -> ~75.
    vad.set_min_silence_duration_ms(2400)
    assert vad._silence_windows_required > baseline_windows
    expected = max(1, int(2400 / vad.window_ms))
    assert vad._silence_windows_required == expected


def test_vad_reset_restores_baseline_silence_requirement(monkeypatch):
    """The bump must be one-shot per utterance: reset() restores the
    configured baseline so the next utterance doesn't inherit it."""
    vad = _make_vad_without_loading_silero(monkeypatch, min_silence_ms=1200)
    baseline_windows = vad._silence_windows_required
    vad.set_min_silence_duration_ms(2400)
    assert vad._silence_windows_required != baseline_windows
    vad.reset()
    assert vad._silence_windows_required == baseline_windows


def test_vad_set_min_silence_duration_floors_at_one_window(monkeypatch):
    """Pathological zero/negative input must not produce a zero-window
    requirement (would make SPEECH_END fire on every silence sample)."""
    vad = _make_vad_without_loading_silero(monkeypatch, min_silence_ms=500)
    vad.set_min_silence_duration_ms(0)
    assert vad._silence_windows_required >= 1


# ---- vad.max_utterance_seconds schema (2026-05-11 follow-up fix) ----------
# Hard ceiling on a single VAD-bounded capture is now configurable. The
# legacy class-level constant on ``Orchestrator`` (15.0 s) cut a real
# user off mid-sentence on a complex coding ask. The orchestrator now
# reads ``vad.max_utterance_seconds`` (default 30.0 s) at construction
# and falls back to the class constant only on config failure.


def test_vad_max_utterance_seconds_default_is_thirty():
    """Default must give complex one-breath asks comfortable headroom
    without unbounded runaway. 30 s was chosen because the live-session
    cut-off happened at 15 s, and we want ~2x to absorb similarly
    detailed asks while still bounding pathological captures."""
    from ultron.config import VADConfig
    assert VADConfig().max_utterance_seconds == 30.0


def test_vad_max_utterance_seconds_too_small_rejected():
    """Below 5 s isn't a useful ceiling (typical complete-sentence ask
    is 3-8 s; 5 s minimum keeps the schema honest)."""
    from pydantic import ValidationError
    from ultron.config import VADConfig
    with pytest.raises(ValidationError):
        VADConfig(max_utterance_seconds=3.0)


def test_vad_max_utterance_seconds_too_large_rejected():
    """Above 120 s is unbounded by any practical voice-prompt standard;
    catches typos (e.g. ``1200`` thinking it's milliseconds)."""
    from pydantic import ValidationError
    from ultron.config import VADConfig
    with pytest.raises(ValidationError):
        VADConfig(max_utterance_seconds=200.0)
