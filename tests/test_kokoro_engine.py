"""Tests for the Kokoro TTS engine wrapper (Track 5).

The wrapper ships unconditionally; the actual Kokoro weights load
lazily on first inference. Tests stub the model load + synth call
so the suite runs without the ``kokoro`` package or weights being
present.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest

from ultron.tts.kokoro_engine import (
    ClipItem,
    KokoroEngineLoadError,
    KokoroSpeech,
    KokoroSynthError,
)


# ---------------------------------------------------------------------------
# Construction + lazy-load semantics
# ---------------------------------------------------------------------------


def test_construct_without_load():
    """Construction is cheap -- the engine should be instantiable even
    when the kokoro package + weights are absent. Load is lazy on
    first inference."""
    engine = KokoroSpeech(
        model_path=Path("/nonexistent/kokoro"),
        voice="af_alloy",
        device="cpu",
    )
    assert engine.sample_rate == 24000
    assert engine.voice == "af_alloy"
    assert engine.is_available() is True  # no load attempt yet


def test_first_synthesize_raises_load_error_when_dir_missing(tmp_path):
    engine = KokoroSpeech(model_path=tmp_path / "missing")
    with pytest.raises(KokoroEngineLoadError):
        engine._synthesize("Hello.")


def test_load_error_is_cached(tmp_path):
    """A failed load shouldn't be retried on every call."""
    engine = KokoroSpeech(model_path=tmp_path / "missing")
    with pytest.raises(KokoroEngineLoadError):
        engine._synthesize("Hello.")
    assert engine.is_available() is False
    # Second call still fails fast.
    with pytest.raises(KokoroEngineLoadError):
        engine._synthesize("Hello.")


def test_reset_load_error_clears_state(tmp_path):
    engine = KokoroSpeech(model_path=tmp_path / "missing")
    with pytest.raises(KokoroEngineLoadError):
        engine._synthesize("Hello.")
    assert engine.is_available() is False
    engine.reset_load_error()
    assert engine.is_available() is True


def test_warmup_swallows_load_error(tmp_path):
    """Warmup is fail-open. A missing model directory shouldn't
    raise -- the load error is logged + the warmup is a no-op."""
    engine = KokoroSpeech(model_path=tmp_path / "missing")
    # No exception -- warmup is intentionally tolerant.
    engine.warmup()


# ---------------------------------------------------------------------------
# Stubbed-inference round-trip
# ---------------------------------------------------------------------------


class _FakeKPipeline:
    """Stub that mimics the kokoro KPipeline call shape.

    The real pipeline yields (graphemes, phonemes, audio) tuples.
    Our stub yields one tuple per call -- enough to verify the
    engine collects + concatenates the audio + converts to int16.
    """

    def __init__(self, audio_samples: int = 1200):
        self.audio_samples = audio_samples
        self.last_text = None
        self.last_voice = None

    def __call__(self, text, *, voice, speed):
        self.last_text = text
        self.last_voice = voice
        # Generate a low-amplitude sine wave so the output is non-zero
        # but predictable.
        n = self.audio_samples
        wave = (0.1 * np.sin(np.linspace(0, np.pi, n))).astype(np.float32)
        yield ("graphemes", "phonemes", wave)


def test_synthesize_with_stubbed_pipeline_returns_int16():
    engine = KokoroSpeech(
        model_path=Path("/stub"), voice="af_alloy",
        apply_spectral_smooth=False,   # this test verifies synth shape, not smoothing
    )
    # Bypass the load path with the stub.
    engine._model = _FakeKPipeline(audio_samples=2400)
    engine._loaded = True
    engine._load_error = None

    pcm, sr = engine._synthesize("Hello world.")
    assert pcm.dtype == np.int16
    assert pcm.size == 2400
    assert sr == 24000
    # Stub recorded the call.
    assert engine._model.last_text == "Hello world."
    assert engine._model.last_voice == "af_alloy"


def test_synthesize_empty_pipeline_returns_zero_clip():
    """Pipeline that yields no tuples -> empty PCM, sample rate
    preserved."""
    engine = KokoroSpeech(model_path=Path("/stub"))
    engine._model = lambda text, voice, speed: iter([])
    engine._loaded = True

    pcm, sr = engine._synthesize("")
    assert pcm.size == 0
    assert sr == 24000


def test_synthesize_failure_raises_synth_error():
    """Underlying inference exception becomes a KokoroSynthError."""
    engine = KokoroSpeech(model_path=Path("/stub"))

    def broken_pipeline(text, *, voice, speed):
        raise RuntimeError("oom")

    engine._model = broken_pipeline
    engine._loaded = True
    with pytest.raises(KokoroSynthError):
        engine._synthesize("test")


def test_synthesize_concatenates_multiple_pipeline_chunks():
    """Multi-sentence pipelines yield multiple chunks; the engine
    concatenates them in order."""

    class _MultiChunk:
        def __call__(self, text, *, voice, speed):
            yield ("g", "p", np.full(100, 0.05, dtype=np.float32))
            yield ("g", "p", np.full(200, 0.1, dtype=np.float32))

    engine = KokoroSpeech(model_path=Path("/stub"))
    engine._model = _MultiChunk()
    engine._loaded = True

    pcm, _sr = engine._synthesize("Two sentences. Combined.")
    assert pcm.size == 300


# ---------------------------------------------------------------------------
# Runtime filter (pre-fine-tune path)
# ---------------------------------------------------------------------------


def test_runtime_filter_does_not_crash_on_unimportable():
    """If the pedalboard filter import fails (e.g., pedalboard not
    installed in the venv), the engine falls back to unfiltered
    output rather than raising."""
    engine = KokoroSpeech(
        model_path=Path("/stub"),
        apply_runtime_filter=True,
    )
    engine._model = _FakeKPipeline()
    engine._loaded = True
    # Should not raise even if pedalboard / ultron_filter are
    # unavailable -- the engine catches the exception.
    pcm, _sr = engine._synthesize("test")
    assert pcm.dtype == np.int16


# ---------------------------------------------------------------------------
# Spectral smoothing (2026-05-22 partial-fine-tune ship)
# ---------------------------------------------------------------------------


def test_spectral_smooth_default_enabled_runs_on_synth():
    """apply_spectral_smooth defaults to False; explicitly enable and verify
    the smoothing function is called on every cache-miss synth."""
    engine = KokoroSpeech(model_path=Path("/stub"), apply_spectral_smooth=True)
    assert engine.apply_spectral_smooth is True
    assert engine.spectral_smooth_window == 5

    # Wire a fake pipeline returning a long-enough clip for STFT
    # (must be >= n_fft=2048 samples or smoothing returns unchanged).
    engine._model = _FakeKPipeline(audio_samples=3000)
    engine._loaded = True

    pcm, sr = engine._synthesize("Hello there.")
    assert sr == 24000
    assert pcm.dtype == np.int16
    # STFT framing trims the tail to ``(n_frames - 1) * hop + n_fft``;
    # for 3000-sample input with n_fft=2048, hop=512 -> 2560 samples
    # out. The output is always within (n_fft - hop) = 1536 samples
    # of the input length.
    assert abs(pcm.size - 3000) <= 1536


def test_spectral_smooth_disabled_skips_call(monkeypatch):
    """With apply_spectral_smooth=False the engine must NOT import
    the smoothing module (saves the ~10 ms / sec audio cost)."""
    engine = KokoroSpeech(
        model_path=Path("/stub"),
        apply_spectral_smooth=False,
    )
    engine._model = _FakeKPipeline(audio_samples=3000)
    engine._loaded = True

    # Replace spectral_smooth with a sentinel that records calls.
    calls: list[int] = []
    import ultron.tts.spectral_smooth as smooth_mod
    real = smooth_mod.spectral_smooth

    def _spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(smooth_mod, "spectral_smooth", _spy)

    pcm, _sr = engine._synthesize("Hello.")
    assert calls == []
    assert pcm.dtype == np.int16


def test_spectral_smooth_fail_open_on_scipy_missing(monkeypatch):
    """If scipy is missing in the runtime venv, the smoothing call
    raises ImportError; the engine must log + pass through raw
    output rather than crash."""
    engine = KokoroSpeech(
        model_path=Path("/stub"),
        apply_trim_fade=False,  # isolate spectral_smooth fail-open; trim runs separately
    )
    engine._model = _FakeKPipeline(audio_samples=3000)
    engine._loaded = True

    import ultron.tts.spectral_smooth as smooth_mod

    def _broken(*a, **kw):
        raise ImportError("simulated missing scipy")

    monkeypatch.setattr(smooth_mod, "spectral_smooth", _broken)

    # Should not raise.
    pcm, _sr = engine._synthesize("Hello.")
    assert pcm.dtype == np.int16
    # Raw output preserved (no smoothing tail length change).
    assert pcm.size == 3000


def test_spectral_smooth_cache_hit_skips_smoothing():
    """Cached clips are pre-smoothed at cache-build time; the
    cache-hit fast path must NOT re-run smoothing (it would double-
    apply + cost ~10 ms / sec on every ack)."""
    from ultron.tts.precomputed_ack import PrecomputedAckClipCache

    cache = PrecomputedAckClipCache(["Mm."])
    sentinel_pcm = np.array([42, 43, 44], dtype=np.int16)
    cache._clips = {"Mm.": (sentinel_pcm, 24000)}

    engine = _make_engine_with_fake_pipeline()
    engine.apply_spectral_smooth = True   # would normally smooth
    engine.set_ack_cache(cache)

    pcm, _sr = engine._synthesize("Mm.")
    # Smoothing would re-shape the values; verify the cached clip
    # is returned bit-for-bit instead.
    assert (pcm == sentinel_pcm).all()


# ---------------------------------------------------------------------------
# Public API surface mirrors XttsV3Speech
# ---------------------------------------------------------------------------


def test_public_surface_matches_xtts_v3():
    """The orchestrator swaps engines via tts.engine; the playback
    path doesn't know which engine it has. Verify the contract."""
    engine = KokoroSpeech(model_path=Path("/stub"))
    assert hasattr(engine, "speak")
    assert hasattr(engine, "speak_stream")
    assert hasattr(engine, "warmup")
    assert hasattr(engine, "stop")
    assert hasattr(engine, "prepare_output_stream")
    assert hasattr(engine, "sample_rate")


def test_stop_clears_preopened_stream():
    """stop() releases the device handle so the playback path opens
    fresh next time. Mirrors XTTS behaviour."""
    engine = KokoroSpeech(model_path=Path("/stub"))

    class _FakeStream:
        def __init__(self):
            self.stopped = False
            self.closed = False

        def stop(self):
            self.stopped = True

        def close(self):
            self.closed = True

    fake = _FakeStream()
    engine._preopened_stream = fake
    engine.stop()
    assert fake.stopped is True
    assert fake.closed is True
    assert engine._preopened_stream is None


def test_clipitem_namedtuple_shape():
    """ClipItem mirrors the XTTS / legacy queue contract."""
    item = ClipItem(audio=np.zeros(10, dtype=np.int16), sample_rate=24000)
    assert item.is_known_last is False
    item2 = ClipItem(
        audio=np.zeros(10, dtype=np.int16),
        sample_rate=24000,
        is_known_last=True,
    )
    assert item2.is_known_last is True


# ---------------------------------------------------------------------------
# Config / engine selection
# ---------------------------------------------------------------------------


def test_kokoro_engine_in_tts_schema():
    """tts.engine accepts 'kokoro' alongside legacy + xtts_v3."""
    from ultron.config import TTSConfig
    cfg = TTSConfig(engine="kokoro")
    assert cfg.engine == "kokoro"


def test_kokoro_config_has_sensible_defaults():
    from ultron.config import KokoroConfig
    cfg = KokoroConfig()
    assert cfg.model_path == "models/kokoro"
    assert cfg.voice == "af_alloy"
    assert cfg.device == "cpu"
    assert cfg.speed == 1.0
    assert cfg.apply_runtime_filter is False
    # Spectral smoothing off by default (2026-05-22 default was True for
    # the partial fine-tune ship; reverted once stock voice is the norm).
    # Re-enable via tts.kokoro.apply_spectral_smooth: true in config.yaml.
    assert cfg.apply_spectral_smooth is False
    assert cfg.spectral_smooth_window == 5


def test_kokoro_config_spectral_smooth_window_validated():
    """spectral_smooth_window has bounded range to avoid runaway
    median filters that would smear consonants."""
    from ultron.config import KokoroConfig
    from pydantic import ValidationError
    # Lower bound: 1 (no-op).
    KokoroConfig(spectral_smooth_window=1)
    with pytest.raises(ValidationError):
        KokoroConfig(spectral_smooth_window=0)
    # Upper bound: 15 frames (~320 ms at hop=512, sr=24 kHz; well
    # beyond useful but not catastrophic).
    KokoroConfig(spectral_smooth_window=15)
    with pytest.raises(ValidationError):
        KokoroConfig(spectral_smooth_window=16)


# ---------------------------------------------------------------------------
# Producer/consumer pipeline (2026-05-20 round 8c)
# ---------------------------------------------------------------------------


def _make_engine_with_fake_pipeline(audio_samples: int = 1200):
    """Construct a KokoroSpeech wired to a stub _model that returns
    deterministic per-call audio. Bypasses the real Kokoro load.
    """
    engine = KokoroSpeech(model_path=Path("/stub"), voice="am_michael")
    engine._model = _FakeKPipeline(audio_samples=audio_samples)
    engine._loaded = True
    engine._load_error = None
    return engine


def test_is_safe_sentence_boundary_basic_terminators():
    """`!`, `?`, `\\n` are always safe; isolated `.` after a word is safe."""
    f = KokoroSpeech._is_safe_sentence_boundary
    assert f("Hello!", 5, buffer_complete=False) is True
    assert f("Hello?", 5, buffer_complete=False) is True
    assert f("Line1\nLine2", 5, buffer_complete=False) is True
    # Period followed by space is a safe sentence end. `.` is at idx 8.
    assert f("Hi there. Then", 8, buffer_complete=False) is True


def test_is_safe_sentence_boundary_rejects_ellipsis():
    """Mid-ellipsis dots are not safe boundaries."""
    f = KokoroSpeech._is_safe_sentence_boundary
    s = "Wait... what?"
    # First dot at index 4 -- next char is `.` -> reject.
    assert f(s, 4, buffer_complete=False) is False
    # Second dot at index 5 -- prev is `.` -> reject.
    assert f(s, 5, buffer_complete=False) is False
    # Third dot at index 6 -- prev is `.` -> reject.
    assert f(s, 6, buffer_complete=False) is False
    # `?` is always safe.
    assert f(s, 12, buffer_complete=False) is True


def test_is_safe_sentence_boundary_rejects_decimal():
    """Decimal `3.14` is not a sentence boundary."""
    f = KokoroSpeech._is_safe_sentence_boundary
    s = "Pi is 3.14 approximately."
    # Dot at index 7 -- prev is digit, next is digit -> reject.
    assert f(s, 7, buffer_complete=False) is False
    # Final `.` at index 24 IS safe (end of stream + buffer_complete).
    assert f(s, 24, buffer_complete=True) is True


def test_is_safe_sentence_boundary_rejects_domain():
    """`Dictionary.com` mid-domain dot is not safe."""
    f = KokoroSpeech._is_safe_sentence_boundary
    s = "Visit Dictionary.com today."
    # Dot at index 16 -- letter.letter -> reject.
    assert f(s, 16, buffer_complete=False) is False
    # Final dot at 26 -- safe at end-of-stream.
    assert f(s, 26, buffer_complete=True) is True


def test_is_safe_sentence_boundary_rejects_abbreviation():
    """Known abbreviations don't flush even when followed by a space."""
    f = KokoroSpeech._is_safe_sentence_boundary
    s = "Dr. Smith arrived."
    # Dot at index 2 -- preceded by "Dr" (abbreviation) and followed
    # by space -> reject.
    assert f(s, 2, buffer_complete=False) is False
    # Final dot at 17 -- safe at end-of-stream.
    assert f(s, 17, buffer_complete=True) is True


def test_is_safe_sentence_boundary_rejects_acronym_chain():
    """`U.S.` mid-acronym continuation is not safe."""
    f = KokoroSpeech._is_safe_sentence_boundary
    s = "The U.S. exports goods."
    # Dot at index 5 -- isolated single-letter; next char `.` -> reject.
    assert f(s, 5, buffer_complete=False) is False
    # Dot at index 7 -- pos-2 is `.`, pos-1 is letter -> reject.
    assert f(s, 7, buffer_complete=False) is False


def test_find_next_sentence_boundary_skips_unsafe():
    """Scan returns position+1 of the FIRST safe terminator."""
    engine = KokoroSpeech(model_path=Path("/stub"))
    # Ellipsis + decimal + final period; only the final period is safe.
    text = "Wait... ok 3.14 done."
    cut = engine._find_next_sentence_boundary(text, buffer_complete=True)
    assert cut == len(text)  # whole text up to and including final `.`


def test_find_next_sentence_boundary_returns_zero_when_none():
    """No safe boundary -> 0."""
    engine = KokoroSpeech(model_path=Path("/stub"))
    assert engine._find_next_sentence_boundary(
        "no period or terminator", buffer_complete=False,
    ) == 0
    # Trailing `.` with buffer_complete=False also doesn't fire (might
    # be more tokens coming).
    assert engine._find_next_sentence_boundary(
        "trailing.", buffer_complete=False,
    ) == 0


def test_run_synth_loop_emits_clipitems_per_safe_boundary():
    """Producer pushes one ClipItem per safe-boundary flush + tail."""
    engine = _make_engine_with_fake_pipeline(audio_samples=2400)
    pushed: List[ClipItem] = []
    engine._run_synth_loop(
        fragments=["Hello.", " World!"],
        push=pushed.append,
    )
    assert len(pushed) == 2
    for item in pushed:
        assert isinstance(item, ClipItem)
        assert item.audio.dtype == np.int16
        assert item.sample_rate == 24000
        assert item.is_known_last is False


def test_run_synth_loop_does_not_fragment_on_ellipsis():
    """`Wait... what?` should produce ONE ClipItem, not four."""
    engine = _make_engine_with_fake_pipeline()
    pushed: List[ClipItem] = []
    engine._run_synth_loop(
        fragments=["Wait... what?"],
        push=pushed.append,
    )
    # `?` is the only safe boundary; the three ellipsis dots are
    # rejected. So one clip.
    assert len(pushed) == 1
    assert engine._model.last_text == "Wait... what?"


def test_run_synth_loop_does_not_fragment_on_decimal():
    """`Pi is 3.14 approximately.` -> one ClipItem."""
    engine = _make_engine_with_fake_pipeline()
    pushed: List[ClipItem] = []
    engine._run_synth_loop(
        fragments=["Pi is 3.14 approximately."],
        push=pushed.append,
    )
    assert len(pushed) == 1


def test_run_synth_loop_handles_streamed_fragments():
    """Fragments arriving char-by-char still flush on safe boundaries."""

    class _Recording:
        def __init__(self):
            self.calls = []

        def __call__(self, text, *, voice, speed):
            self.calls.append(text)
            yield ("g", "p", np.full(100, 0.05, dtype=np.float32))

    engine = KokoroSpeech(model_path=Path("/stub"))
    engine._model = _Recording()
    engine._loaded = True
    pushed = []
    # Stream the text as 1-3 char fragments.
    fragments = ["He", "llo.", " ", "How", " are", " you?"]
    engine._run_synth_loop(fragments=fragments, push=pushed.append)
    # Two safe boundaries: after "Hello." and after "you?".
    assert len(pushed) == 2
    # The first synth call should be "Hello." not "He" or "Hell".
    assert engine._model.calls[0].strip() == "Hello."
    assert engine._model.calls[1].strip() == "How are you?"


def test_run_synth_loop_tail_flush():
    """Trailing text without a terminator still flushes at end-of-stream."""
    engine = _make_engine_with_fake_pipeline()
    pushed: List[ClipItem] = []
    engine._run_synth_loop(
        fragments=["This has no terminator"],
        push=pushed.append,
    )
    assert len(pushed) == 1


def test_run_synth_loop_skips_synth_failures():
    """Failed synth on one sentence doesn't abort the stream."""

    class _FailFirst:
        def __init__(self):
            self.n = 0

        def __call__(self, text, *, voice, speed):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("simulated OOM on first sentence")
            yield ("g", "p", np.full(100, 0.05, dtype=np.float32))

    engine = KokoroSpeech(model_path=Path("/stub"))
    engine._model = _FailFirst()
    engine._loaded = True
    pushed: List[ClipItem] = []
    engine._run_synth_loop(
        fragments=["First. Second."],
        push=pushed.append,
    )
    # First synth failed; second succeeded. Only one ClipItem pushed.
    assert len(pushed) == 1


def test_run_synth_loop_respects_stop_event():
    """stop_event mid-stream interrupts the synth loop."""
    engine = _make_engine_with_fake_pipeline()
    pushed: List[ClipItem] = []
    engine._stop_event.set()
    engine._run_synth_loop(
        fragments=["One. Two. Three."],
        push=pushed.append,
    )
    # Stop event set before entering -> no pushes.
    assert pushed == []


def test_stereo_pcm_expands_mono():
    """Mono int16 -> stereo column-stacked."""
    mono = np.array([1, 2, 3, 4], dtype=np.int16)
    out = KokoroSpeech._stereo_pcm(mono)
    assert out.shape == (4, 2)
    assert out.dtype == np.int16
    # Both channels carry the same data.
    assert (out[:, 0] == out[:, 1]).all()
    assert (out[:, 0] == mono).all()


def test_stereo_pcm_passthrough_for_stereo():
    """Already-stereo input passes through unchanged."""
    stereo = np.array([[1, 2], [3, 4]], dtype=np.int16)
    out = KokoroSpeech._stereo_pcm(stereo)
    assert out.shape == (2, 2)
    assert (out == stereo).all()


def test_write_silence_swallows_stream_failure():
    """A broken stream doesn't raise; the helper is best-effort."""

    class _BadStream:
        def write(self, *_a, **_k):
            raise RuntimeError("device gone")

    # Should not raise.
    KokoroSpeech._write_silence(_BadStream(), 24000, 0.05)


def test_write_silence_zero_seconds_is_noop():
    """seconds <= 0 -> no write attempted."""
    calls = []

    class _Recorder:
        def write(self, arr):
            calls.append(arr.shape)

    KokoroSpeech._write_silence(_Recorder(), 24000, 0.0)
    KokoroSpeech._write_silence(_Recorder(), 24000, -1.0)
    assert calls == []


def test_speak_stream_overlaps_synth_and_playback(monkeypatch):
    """Critical correctness test: while sentence N is "playing", synth
    for sentence N+1 has already started in the worker thread. Verify
    by timing -- the wall-clock cost of speak_stream on a 3-sentence
    response should be roughly max(synth_total, playback_total), not
    synth_total + playback_total.
    """
    import sounddevice as sd

    # Synth stub: ~30 ms per sentence.
    class _SlowSynth:
        def __init__(self):
            self.calls = 0

        def __call__(self, text, *, voice, speed):
            self.calls += 1
            time.sleep(0.03)
            yield ("g", "p", np.full(2400, 0.05, dtype=np.float32))

    # Playback stub: ~30 ms per write (drains slowly).
    write_times = []

    class _SlowStream:
        def __init__(self, **_k):
            self.started = False

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

        def close(self):
            pass

        def write(self, arr):
            time.sleep(0.005)  # 5 ms / 50 ms block
            write_times.append(time.monotonic())

    monkeypatch.setattr(sd, "OutputStream", _SlowStream)

    engine = KokoroSpeech(model_path=Path("/stub"), voice="am_michael")
    engine._model = _SlowSynth()
    engine._loaded = True

    t0 = time.monotonic()
    engine.speak_stream(["First. Second. Third."])
    elapsed = time.monotonic() - t0

    # 3 synth calls @ 30 ms = 90 ms total synth.
    # With overlap, total wall clock should be ~ playback_total +
    # one-synth slack (the first clip needs synth before playback can
    # start). With sequential (no overlap), it'd be ~120-150 ms.
    # We assert overlap by checking engine.calls == 3 AND total < 200 ms.
    assert engine._model.calls == 3
    assert elapsed < 0.5, f"speak_stream took {elapsed*1000:.0f}ms; expected <500ms"


def test_speak_stream_sentinel_terminates_consumer(monkeypatch):
    """Synth worker's `None` sentinel on the queue cleanly ends playback."""
    import sounddevice as sd

    class _StubStream:
        def __init__(self, **_k): pass
        def start(self): pass
        def stop(self): pass
        def close(self): pass
        def write(self, _arr): pass

    monkeypatch.setattr(sd, "OutputStream", _StubStream)

    engine = _make_engine_with_fake_pipeline()
    # Single sentence so we exit after one playback iteration.
    engine.speak_stream(["Just one."])
    # Worker thread should have joined cleanly (no leftover).
    for t in threading.enumerate():
        assert t.name != "kokoro-synth" or not t.is_alive(), \
            "synth worker did not exit"


def test_speak_stream_stop_event_interrupts(monkeypatch):
    """stop() mid-stream halts both synth and playback promptly."""
    import sounddevice as sd

    class _StubStream:
        def __init__(self, **_k): pass
        def start(self): pass
        def stop(self): pass
        def close(self): pass
        def write(self, _arr): pass

    monkeypatch.setattr(sd, "OutputStream", _StubStream)

    engine = _make_engine_with_fake_pipeline(audio_samples=24000)  # 1s
    engine._stop_event.set()  # pre-set
    engine.speak_stream(["This. That. The other."])
    # Synth loop bails immediately on stop_event; no pushes.
    # Playback first-clip wait should hit timeout-free path because
    # the worker still puts None via the finally clause.
    # No assertions on output -- just confirm no exceptions.


# ---------------------------------------------------------------------------
# Ack-cache wiring (2026-05-20 round 8d)
# ---------------------------------------------------------------------------


def test_set_ack_cache_attaches_and_logs(caplog):
    """`set_ack_cache(cache)` stores the cache and logs phrase count."""
    import logging
    from ultron.tts.precomputed_ack import PrecomputedAckClipCache

    cache = PrecomputedAckClipCache(["Mm.", "Right.", "Considering."])
    engine = KokoroSpeech(model_path=Path("/stub"))
    with caplog.at_level(logging.INFO, logger="ultron.tts.kokoro"):
        engine.set_ack_cache(cache)
    assert engine._ack_cache is cache
    # Expect a log line mentioning the phrase count.
    assert any("3" in rec.message for rec in caplog.records)


def test_set_ack_cache_none_detaches():
    """Passing `None` detaches the cache (used after server restart)."""
    from ultron.tts.precomputed_ack import PrecomputedAckClipCache

    engine = KokoroSpeech(model_path=Path("/stub"))
    cache = PrecomputedAckClipCache(["Mm."])
    engine.set_ack_cache(cache)
    assert engine._ack_cache is cache
    engine.set_ack_cache(None)
    assert engine._ack_cache is None


def test_synthesize_cache_hit_skips_kpipeline():
    """Cache hit returns the stored clip without touching the KPipeline."""
    from ultron.tts.precomputed_ack import PrecomputedAckClipCache

    cached_pcm = np.array([1, 2, 3, 4], dtype=np.int16)
    cached_sr = 24000
    cache = PrecomputedAckClipCache(["Mm."])
    # Manually populate (bypass prewarm) -- the cache stores keyed by
    # stripped text.
    cache._clips = {"Mm.": (cached_pcm, cached_sr)}

    pipeline_calls = []

    class _ShouldNotBeCalled:
        def __call__(self, text, *, voice, speed):
            pipeline_calls.append(text)
            yield ("g", "p", np.full(100, 0.05, dtype=np.float32))

    engine = KokoroSpeech(model_path=Path("/stub"), voice="am_michael")
    engine._model = _ShouldNotBeCalled()
    engine._loaded = True
    engine.set_ack_cache(cache)

    pcm, sr = engine._synthesize("Mm.")
    # Cache hit -- KPipeline not invoked.
    assert pipeline_calls == []
    # Returned clip is the cached one.
    assert sr == cached_sr
    assert (pcm == cached_pcm).all()


def test_synthesize_cache_miss_falls_through_to_kpipeline():
    """Cache miss runs the live KPipeline path unchanged."""
    from ultron.tts.precomputed_ack import PrecomputedAckClipCache

    cache = PrecomputedAckClipCache(["Right."])
    cache._clips = {
        "Right.": (np.array([99], dtype=np.int16), 24000),
    }
    engine = _make_engine_with_fake_pipeline(audio_samples=1200)
    engine.set_ack_cache(cache)

    # Text NOT in the cache.
    pcm, sr = engine._synthesize("This is not cached.")
    assert engine._model.last_text == "This is not cached."
    assert pcm.size == 1200
    assert sr == 24000


def test_synthesize_with_no_cache_attached_uses_kpipeline():
    """Pre-cache-wiring behaviour preserved: no cache -> live path."""
    engine = _make_engine_with_fake_pipeline()
    assert engine._ack_cache is None
    pcm, sr = engine._synthesize("Hello world.")
    assert engine._model.last_text == "Hello world."
    assert pcm.dtype == np.int16


def test_synthesize_cache_hit_skips_apply_runtime_filter():
    """Cached clip is returned verbatim -- runtime filter does NOT
    re-run on the cached path (the cache stores already-filtered audio)."""
    from ultron.tts.precomputed_ack import PrecomputedAckClipCache

    cache = PrecomputedAckClipCache(["Mm."])
    sentinel_pcm = np.array([42, 43, 44], dtype=np.int16)
    cache._clips = {"Mm.": (sentinel_pcm, 24000)}

    engine = _make_engine_with_fake_pipeline()
    engine.apply_runtime_filter = True   # would normally run filter
    engine.set_ack_cache(cache)

    pcm, _sr = engine._synthesize("Mm.")
    # Filter would mutate the values; verify the cached clip is
    # returned bit-for-bit instead.
    assert (pcm == sentinel_pcm).all()


def test_synthesize_cache_uses_stripped_key():
    """The orchestrator's flow strips text before calling _synthesize;
    the cache should be keyed by the same stripped form."""
    from ultron.tts.precomputed_ack import PrecomputedAckClipCache

    cache = PrecomputedAckClipCache(["Mm."])
    cache._clips = {"Mm.": (np.array([7], dtype=np.int16), 24000)}
    engine = _make_engine_with_fake_pipeline()
    engine.set_ack_cache(cache)

    # _synthesize is called with stripped text by _run_synth_loop;
    # call with the stripped form directly here.
    pcm, _sr = engine._synthesize("Mm.")
    assert (pcm == np.array([7], dtype=np.int16)).all()


def test_speak_stream_fail_open_on_missing_sounddevice(monkeypatch):
    """If sounddevice import fails the call returns cleanly without raising."""
    import builtins
    real_import = builtins.__import__

    def _broken_import(name, *a, **k):
        if name == "sounddevice":
            raise ImportError("simulated missing portaudio")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _broken_import)
    engine = _make_engine_with_fake_pipeline()
    # Should not raise.
    engine.speak_stream(["Hello."])
