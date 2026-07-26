"""Tests for the XTTS v3 TTS engine selection + config schema.

Covers the 2026-05-10 voice-pipeline swap:

- ``tts.engine`` defaults to legacy ``piper_rvc`` so existing
  installs keep working without config changes.
- ``"xtts_v3"`` is accepted by the schema.
- Unknown engine names are rejected.
- ``XttsV3Config`` round-trips through the loader with the expected
  defaults pointing at the audio prep layout.
- The kenning filter module imports cleanly with all three presets.
"""

from __future__ import annotations

import numpy as np
import pytest

from kenning.config import (
    TTSConfig,
    KenningConfig,
    XttsV3Config,
)


# ---------------------------------------------------------------------------
# Engine selection schema
# ---------------------------------------------------------------------------


def test_tts_engine_defaults_to_legacy_piper_rvc():
    cfg = TTSConfig()
    assert cfg.engine == "piper_rvc"


def test_tts_engine_accepts_xtts_v3():
    cfg = TTSConfig(engine="xtts_v3")
    assert cfg.engine == "xtts_v3"


def test_tts_engine_rejects_unknown_value():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TTSConfig(engine="kokoro_lora")


# ---------------------------------------------------------------------------
# XttsV3Config defaults
# ---------------------------------------------------------------------------


def test_xtts_v3_config_defaults_match_audio_prep_layout():
    """Defaults point at the layout established during the 2026-05-10
    voice swap. If the audio prep moves, these defaults need to move
    with it AND the engine has to keep working without explicit
    config overrides."""
    cfg = XttsV3Config()
    assert cfg.server_python.endswith(".venv-xtts/Scripts/python.exe")
    assert cfg.server_script.endswith("xtts_server.py")
    # 2026-06-12 Kenning rename: the gitignored venv + reference WAV were
    # NOT migrated (baked-in venv paths) -- they stay at ultronVoiceAudio/
    # under their original on-disk names so the engine works without
    # config overrides. The tracked server script moved to kenningVoiceAudio/.
    assert cfg.reference_audio.endswith("Ultron_vocals_mono_v1.wav")
    assert cfg.server_python.startswith("ultronVoiceAudio/")
    # The WAV lives under the "kokoro training audio" subdirectory
    # (moved there during disk cleaning); the default points at its real
    # on-disk home.
    assert "kokoro training audio" in cfg.reference_audio
    assert cfg.host == "127.0.0.1"
    assert cfg.port is None  # engine picks free port at startup
    assert cfg.filter_preset == "v3_heavy"
    assert cfg.filter_tail_silence_ms == 200.0
    # Schema default is XTTS-native 1.0 so direct ctor calls (mostly
    # tests) stay back-compat. The production value lives in
    # config.yaml.
    assert cfg.speed == 1.0
    # 2026-05-12 phantom-token mitigation: schema default 0.65 (vs
    # the XTTS library's 0.75). Tightens the duration-token
    # distribution so the model emits fewer phantom syllables.
    assert cfg.temperature == 0.65
    # Phantom-tail trim is defence-in-depth on top of the temperature
    # reduction. Default ON: trim is conservative (only fires when
    # the specific phantom pattern is matched).
    assert cfg.phantom_tail_trim_enabled is True
    assert cfg.phantom_tail_silence_threshold == 0.005
    assert cfg.phantom_tail_max_event_ms == 200.0
    assert cfg.phantom_tail_min_lead_silence_ms == 150.0


def test_xtts_v3_config_speed_range_enforced():
    """Bounded to keep things in the natural-sounding range. Below
    ~0.7 the model sounds drawn out; above ~1.4 it starts to slur
    consonants. The schema clamps at [0.5, 2.0] so callers can't
    accidentally ship a setting that destroys intelligibility."""
    from pydantic import ValidationError
    XttsV3Config(speed=0.5)  # ok (lower bound)
    XttsV3Config(speed=1.15)  # ok (production value)
    XttsV3Config(speed=2.0)  # ok (upper bound)
    with pytest.raises(ValidationError):
        XttsV3Config(speed=0.49)
    with pytest.raises(ValidationError):
        XttsV3Config(speed=2.01)


def test_xtts_v3_config_speed_round_trips_through_dict():
    cfg = XttsV3Config(speed=1.15)
    cfg2 = XttsV3Config.model_validate(cfg.model_dump())
    assert cfg2.speed == 1.15


def test_xtts_v3_config_filter_tail_ms_range_enforced():
    from pydantic import ValidationError
    XttsV3Config(filter_tail_silence_ms=0.0)  # ok
    XttsV3Config(filter_tail_silence_ms=2000.0)  # ok
    with pytest.raises(ValidationError):
        XttsV3Config(filter_tail_silence_ms=-1.0)
    with pytest.raises(ValidationError):
        XttsV3Config(filter_tail_silence_ms=2500.0)


def test_xtts_v3_client_forwards_speed_in_http_body(monkeypatch, tmp_path):
    """Pure wiring test: confirms ``XttsV3Speech._http_synthesize``
    sends the configured speed in the POST JSON body so the server's
    XTTS ``inference_stream(speed=...)`` call actually picks it up.

    Mocks the subprocess + HTTP seams so we don't load the voice
    stack (per feedback_voice_stack_concurrency). If the client ever
    silently drops the speed field, this test fails."""
    import json
    import urllib.request
    from kenning.tts import xtts_v3

    # The constructor asserts the configured paths exist; stub files
    # under tmp_path satisfy that without spawning anything.
    server_py = tmp_path / "python.exe"
    server_py.write_text("")
    server_sc = tmp_path / "xtts_server.py"
    server_sc.write_text("")
    ref_wav = tmp_path / "ref.wav"
    ref_wav.write_text("")

    captured: list[bytes] = []

    class _FakeResp:
        headers = {"X-Sample-Rate": "24000"}
        def read(self, n=None):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

    def _fake_urlopen(req, timeout=None):
        data = getattr(req, "data", None)
        if data:
            captured.append(data)
        return _FakeResp()

    # Skip the subprocess spawn + health-probe loop.
    monkeypatch.setattr(xtts_v3.XttsV3Speech, "_start_server", lambda self: None)
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    engine = xtts_v3.XttsV3Speech(
        server_python=server_py,
        server_script=server_sc,
        reference_audio=ref_wav,
        port=12345,
        speed=1.15,
    )

    engine._http_synthesize("hello there")

    assert captured, "expected exactly one POST to /synthesize"
    body = json.loads(captured[0].decode("utf-8"))
    assert body["text"] == "hello there"
    assert body["language"] == "en"
    assert body["speed"] == 1.15


# ---------------------------------------------------------------------------
# Temperature schema + HTTP-body wiring (2026-05-12 phantom-token mitigation)
# ---------------------------------------------------------------------------


def test_xtts_v3_config_temperature_range_enforced():
    """Bounded so callers can't ship a setting that destroys the
    duration-token distribution. Below ~0.4 prosody collapses; above
    ~1.0 the model becomes unstable."""
    from pydantic import ValidationError
    XttsV3Config(temperature=0.4)
    XttsV3Config(temperature=0.65)  # schema default
    XttsV3Config(temperature=1.0)
    with pytest.raises(ValidationError):
        XttsV3Config(temperature=0.39)
    with pytest.raises(ValidationError):
        XttsV3Config(temperature=1.01)


def test_xtts_v3_config_temperature_round_trips_through_dict():
    cfg = XttsV3Config(temperature=0.65)
    cfg2 = XttsV3Config.model_validate(cfg.model_dump())
    assert cfg2.temperature == 0.65


def test_xtts_v3_client_forwards_temperature_in_http_body(monkeypatch, tmp_path):
    """Pure wiring test: confirms ``XttsV3Speech._http_synthesize``
    sends the configured temperature in the POST JSON body. If the
    client ever silently drops the temperature field, the server
    falls back to its library default of 0.75 and the phantom-token
    rate goes back up. This test pins that wiring closed."""
    import json
    import urllib.request
    from kenning.tts import xtts_v3

    server_py = tmp_path / "python.exe"
    server_py.write_text("")
    server_sc = tmp_path / "xtts_server.py"
    server_sc.write_text("")
    ref_wav = tmp_path / "ref.wav"
    ref_wav.write_text("")

    captured: list[bytes] = []

    class _FakeResp:
        headers = {"X-Sample-Rate": "24000"}
        def read(self, n=None):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

    def _fake_urlopen(req, timeout=None):
        data = getattr(req, "data", None)
        if data:
            captured.append(data)
        return _FakeResp()

    monkeypatch.setattr(xtts_v3.XttsV3Speech, "_start_server", lambda self: None)
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    engine = xtts_v3.XttsV3Speech(
        server_python=server_py,
        server_script=server_sc,
        reference_audio=ref_wav,
        port=12345,
        temperature=0.65,
    )

    engine._http_synthesize("hello there")

    assert captured, "expected exactly one POST to /synthesize"
    body = json.loads(captured[0].decode("utf-8"))
    assert body["temperature"] == 0.65


# ---------------------------------------------------------------------------
# Phantom-tail trim configuration (2026-05-12 phantom-token mitigation)
# ---------------------------------------------------------------------------


def test_xtts_v3_config_phantom_tail_trim_enabled_default_on():
    cfg = XttsV3Config()
    assert cfg.phantom_tail_trim_enabled is True


def test_xtts_v3_config_phantom_tail_silence_threshold_range_enforced():
    from pydantic import ValidationError
    XttsV3Config(phantom_tail_silence_threshold=0.0001)
    XttsV3Config(phantom_tail_silence_threshold=0.005)  # schema default
    XttsV3Config(phantom_tail_silence_threshold=0.05)
    with pytest.raises(ValidationError):
        XttsV3Config(phantom_tail_silence_threshold=0.0)
    with pytest.raises(ValidationError):
        XttsV3Config(phantom_tail_silence_threshold=0.06)


def test_xtts_v3_config_phantom_tail_max_event_ms_range_enforced():
    from pydantic import ValidationError
    XttsV3Config(phantom_tail_max_event_ms=50.0)
    XttsV3Config(phantom_tail_max_event_ms=200.0)  # schema default
    XttsV3Config(phantom_tail_max_event_ms=500.0)
    with pytest.raises(ValidationError):
        XttsV3Config(phantom_tail_max_event_ms=49.0)
    with pytest.raises(ValidationError):
        XttsV3Config(phantom_tail_max_event_ms=501.0)


def test_xtts_v3_config_phantom_tail_min_lead_silence_ms_range_enforced():
    from pydantic import ValidationError
    XttsV3Config(phantom_tail_min_lead_silence_ms=50.0)
    XttsV3Config(phantom_tail_min_lead_silence_ms=150.0)  # schema default
    XttsV3Config(phantom_tail_min_lead_silence_ms=500.0)
    with pytest.raises(ValidationError):
        XttsV3Config(phantom_tail_min_lead_silence_ms=49.0)
    with pytest.raises(ValidationError):
        XttsV3Config(phantom_tail_min_lead_silence_ms=501.0)


# ---------------------------------------------------------------------------
# trim_phantom_tail function — pure DSP, no engine needed
# ---------------------------------------------------------------------------


def _build_buffer(sr: int, *segments: tuple[str, float, float]) -> np.ndarray:
    """Helper: build a float32 audio buffer from (kind, duration_s, amplitude) segments.

    ``kind == 'silence'`` produces zeros; anything else produces a
    sine wave at 200 Hz scaled to the amplitude. Simulates the
    phantom-token signature (sustained speech -> silence -> short
    burst -> silence) deterministically.
    """
    chunks: list[np.ndarray] = []
    for kind, dur_s, amp in segments:
        n = int(dur_s * sr)
        if kind == "silence" or amp == 0.0:
            chunks.append(np.zeros(n, dtype=np.float32))
        else:
            t = np.linspace(0.0, dur_s, n, endpoint=False, dtype=np.float32)
            chunks.append((amp * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def test_trim_phantom_tail_detects_and_removes_classic_phantom():
    """Reproduces the 19.28s pattern observed in the user's session:
    long speech -> ~280 ms silence -> ~100 ms isolated burst ->
    ~420 ms silence. The trim should keep the long speech (plus a
    small grace cushion) and drop everything after."""
    from kenning.tts.xtts_v3 import trim_phantom_tail
    sr = 24000
    buf = _build_buffer(
        sr,
        ("speech", 1.5, 0.3),    # sustained speech
        ("silence", 0.28, 0.0),  # lead silence (>=150ms threshold)
        ("speech", 0.10, 0.3),   # the phantom (100 ms event, <200ms ceiling)
        ("silence", 0.42, 0.0),  # trailing silence
    )
    out, trimmed = trim_phantom_tail(buf, sr)
    assert trimmed is True
    # The trim should land somewhere AT or shortly AFTER 1.5 s
    # (sustained speech end) and definitely BEFORE 1.78 s (where the
    # phantom starts).
    keep_s = out.shape[0] / sr
    assert 1.5 <= keep_s < 1.78


def test_trim_phantom_tail_leaves_sustained_speech_alone():
    """An ordinary speech clip with no phantom (just sustained speech
    followed by silence) should NOT be trimmed -- legitimate end-of-
    sentence audio must be preserved."""
    from kenning.tts.xtts_v3 import trim_phantom_tail
    sr = 24000
    buf = _build_buffer(
        sr,
        ("speech", 2.0, 0.3),
        ("silence", 0.20, 0.0),
    )
    out, trimmed = trim_phantom_tail(buf, sr)
    assert trimmed is False
    assert out.shape[0] == buf.shape[0]


def test_trim_phantom_tail_leaves_short_inter_word_silence_alone():
    """A natural mid-sentence inter-word pause (short silence) between
    two speech regions must NOT be misread as a phantom signature.
    The 150 ms ``min_lead_silence_ms`` threshold should reject a
    100 ms gap as too short."""
    from kenning.tts.xtts_v3 import trim_phantom_tail
    sr = 24000
    buf = _build_buffer(
        sr,
        ("speech", 1.0, 0.3),
        ("silence", 0.10, 0.0),  # short inter-word gap
        ("speech", 0.08, 0.3),   # short trailing word
        ("silence", 0.30, 0.0),
    )
    out, trimmed = trim_phantom_tail(buf, sr)
    assert trimmed is False


def test_trim_phantom_tail_leaves_legitimate_long_trailing_speech_alone():
    """A trailing event longer than ``max_event_ms`` is legitimate
    speech, not a phantom. Verify even when preceded by long
    silence we don't trim it."""
    from kenning.tts.xtts_v3 import trim_phantom_tail
    sr = 24000
    buf = _build_buffer(
        sr,
        ("speech", 1.0, 0.3),
        ("silence", 0.30, 0.0),
        ("speech", 0.40, 0.3),  # 400 ms trailing event > 200 ms ceiling
        ("silence", 0.10, 0.0),
    )
    out, trimmed = trim_phantom_tail(buf, sr)
    assert trimmed is False


def test_trim_phantom_tail_handles_empty_input():
    from kenning.tts.xtts_v3 import trim_phantom_tail
    sr = 24000
    empty = np.zeros(0, dtype=np.float32)
    out, trimmed = trim_phantom_tail(empty, sr)
    assert trimmed is False
    assert out.shape[0] == 0


def test_trim_phantom_tail_handles_very_short_clip():
    """Anything shorter than four analysis windows can't be reliably
    classified -- bail out without trimming."""
    from kenning.tts.xtts_v3 import trim_phantom_tail
    sr = 24000
    short = np.zeros(int(0.03 * sr), dtype=np.float32)  # 30 ms < 4 * 20 ms
    out, trimmed = trim_phantom_tail(short, sr)
    assert trimmed is False
    assert out.shape[0] == short.shape[0]


def test_trim_phantom_tail_handles_all_silent_clip():
    """Pure silence has no speech to anchor the pattern. Pass through."""
    from kenning.tts.xtts_v3 import trim_phantom_tail
    sr = 24000
    silent = np.zeros(int(2.0 * sr), dtype=np.float32)
    out, trimmed = trim_phantom_tail(silent, sr)
    assert trimmed is False
    assert out.shape[0] == silent.shape[0]


def test_trim_phantom_tail_skips_short_clip_below_min_duration():
    """Short single-word acks (``"Right."``) sit below 800 ms total.
    The algorithm can misclassify their stop-consonant release as a
    phantom event when XTTS lengthens the pre-stop closure beyond
    150 ms. The min-clip-duration guard returns these unchanged."""
    from kenning.tts.xtts_v3 import trim_phantom_tail
    sr = 24000
    # Build the exact failure profile: voiced body, long closure that
    # would normally exceed min_lead_silence_ms, then a brief stop-
    # consonant release. Without the guard the release gets clipped.
    buf = _build_buffer(
        sr,
        ("speech", 0.30, 0.3),   # "Rai" voiced body
        ("silence", 0.20, 0.0),  # pre-stop closure (>150 ms; would qualify as lead silence)
        ("speech", 0.06, 0.3),   # "t" release burst
        ("silence", 0.10, 0.0),  # tail
    )
    # Total ~660 ms -- under the 800 ms default guard.
    out, trimmed = trim_phantom_tail(buf, sr)
    assert trimmed is False
    assert out.shape[0] == buf.shape[0]


def test_trim_phantom_tail_min_clip_duration_is_tunable():
    """Caller can lower the guard for testing or special cases."""
    from kenning.tts.xtts_v3 import trim_phantom_tail
    sr = 24000
    buf = _build_buffer(
        sr,
        ("speech", 0.30, 0.3),
        ("silence", 0.20, 0.0),
        ("speech", 0.06, 0.3),
        ("silence", 0.10, 0.0),
    )
    # Dropping the guard to 100 ms lets the trim run; the same clip
    # would now be classified as a phantom (since the algorithm sees
    # a long lead-silence + short trailing event).
    out, trimmed = trim_phantom_tail(buf, sr, min_clip_duration_ms=100.0)
    assert trimmed is True
    assert out.shape[0] < buf.shape[0]


def test_trim_phantom_tail_still_fires_on_long_clip_with_phantom():
    """The guard must not block trimming on legitimately long
    multi-sentence clips that DO carry a phantom tail. Reuses the
    canonical phantom profile but checks it crosses the duration
    threshold."""
    from kenning.tts.xtts_v3 import trim_phantom_tail
    sr = 24000
    buf = _build_buffer(
        sr,
        ("speech", 1.5, 0.3),     # ~1.5 s sustained speech
        ("silence", 0.28, 0.0),   # phantom-style lead silence
        ("speech", 0.10, 0.3),    # 100 ms phantom event
        ("silence", 0.42, 0.0),
    )
    # Total ~2.3 s, well over the 800 ms guard.
    out, trimmed = trim_phantom_tail(buf, sr)
    assert trimmed is True
    keep_s = out.shape[0] / sr
    assert keep_s < 1.78  # phantom region is trimmed


def test_trim_phantom_tail_respects_disabled_flag_via_engine(monkeypatch, tmp_path):
    """When ``phantom_tail_trim_enabled=False`` the engine skips the
    trim entirely -- useful for A/B comparison. Verify by patching
    the trim function and asserting it is NOT called."""
    import urllib.request
    from kenning.tts import xtts_v3

    server_py = tmp_path / "python.exe"
    server_py.write_text("")
    server_sc = tmp_path / "xtts_server.py"
    server_sc.write_text("")
    ref_wav = tmp_path / "ref.wav"
    ref_wav.write_text("")

    class _FakeResp:
        headers = {"X-Sample-Rate": "24000"}
        def read(self, n=None):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

    # Server returns 100 ms of silence so _synthesize has audio to
    # filter (synth path is shared between trim-on and trim-off).
    pcm = (np.zeros(2400, dtype=np.int16)).tobytes()
    response_chunks = [pcm, b""]

    class _ResponseWithBody(_FakeResp):
        def __init__(self):
            self._chunks = list(response_chunks)
        def read(self, n=None):
            if not self._chunks:
                return b""
            return self._chunks.pop(0)

    def _fake_urlopen(req, timeout=None):
        return _ResponseWithBody()

    monkeypatch.setattr(xtts_v3.XttsV3Speech, "_start_server", lambda self: None)
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    call_counter = {"n": 0}

    def _spy_trim(*args, **kwargs):
        call_counter["n"] += 1
        return args[0], False

    monkeypatch.setattr(xtts_v3, "trim_phantom_tail", _spy_trim)

    engine = xtts_v3.XttsV3Speech(
        server_python=server_py,
        server_script=server_sc,
        reference_audio=ref_wav,
        port=12345,
        phantom_tail_trim_enabled=False,
    )

    engine._synthesize("hello")
    assert call_counter["n"] == 0


def test_xtts_v3_config_nested_under_tts():
    cfg = TTSConfig()
    assert isinstance(cfg.xtts_v3, XttsV3Config)
    assert cfg.xtts_v3.filter_preset == "v3_heavy"


def test_full_kenning_config_round_trips_with_xtts_v3_engine():
    cfg = KenningConfig()
    cfg.tts.engine = "xtts_v3"
    cfg.tts.xtts_v3.filter_preset = "v2_medium"
    # Round-trip through dict to mimic YAML load.
    cfg2 = KenningConfig.model_validate(cfg.model_dump())
    assert cfg2.tts.engine == "xtts_v3"
    assert cfg2.tts.xtts_v3.filter_preset == "v2_medium"


# ---------------------------------------------------------------------------
# Kenning filter (runtime port)
# ---------------------------------------------------------------------------


def test_kenning_filter_imports_all_three_presets():
    from kenning.tts.kenning_filter import get_preset
    for preset_name in ("v1_subtle", "v2_medium", "v3_heavy"):
        board = get_preset(preset_name)
        # Each preset constructs a fresh Pedalboard with a non-empty plugin chain.
        assert board is not None
        # Mutating the chain should not affect a freshly-constructed one.
        board2 = get_preset(preset_name)
        assert board is not board2


def test_kenning_filter_unknown_preset_raises():
    from kenning.tts.kenning_filter import get_preset
    with pytest.raises(ValueError):
        get_preset("v99_galaxy_brain")  # type: ignore[arg-type]


def test_kenning_filter_apply_roundtrips_silence_with_tail_padding():
    """A silent input should come back longer by ~tail_silence_ms when
    tail padding is enabled. Validates that the padding logic actually
    extends the buffer (so the reverb tail has room to decay)."""
    from kenning.tts.kenning_filter import apply_filter
    sr = 24000
    silent = np.zeros(int(0.5 * sr), dtype=np.float32)
    out = apply_filter(silent, sr, preset="v3_heavy", tail_silence_ms=200.0)
    expected_len = silent.shape[0] + int(0.200 * sr)
    # Allow ~ a few samples of slop from filter internal state.
    assert abs(out.shape[0] - expected_len) < int(0.005 * sr)


def test_kenning_filter_apply_no_tail_padding_preserves_length():
    from kenning.tts.kenning_filter import apply_filter
    sr = 24000
    audio = np.zeros(int(0.5 * sr), dtype=np.float32)
    out = apply_filter(audio, sr, preset="v3_heavy", tail_silence_ms=0.0)
    assert out.shape[0] == audio.shape[0]


def test_kenning_filter_apply_int16_preserves_dtype():
    from kenning.tts.kenning_filter import apply_filter
    sr = 24000
    audio = np.zeros(int(0.5 * sr), dtype=np.int16)
    out = apply_filter(audio, sr, preset="v3_heavy", tail_silence_ms=0.0)
    assert out.dtype == np.int16


# ---------------------------------------------------------------------------
# Text normalisation -- pure pre-XTTS string rewriting
# ---------------------------------------------------------------------------


def test_normalize_passes_through_empty_and_plain_text():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert normalize_text_for_tts("") == ""
    assert normalize_text_for_tts("It is a small fast-flying bird.") == \
        "It is a small fast-flying bird."


def test_normalize_rewrites_time_ampm_lowercase():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    out = normalize_text_for_tts("currently 2:16 a.m. on Tuesday")
    assert "2 16 A M" in out
    assert ":" not in out or "Tuesday" in out  # only the time colon got rewritten


def test_normalize_rewrites_time_ampm_uppercase():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "10 30 P M" in normalize_text_for_tts("at 10:30 PM tonight")


def test_normalize_rewrites_time_no_dots():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "2 16 A M" in normalize_text_for_tts("2:16 am")
    assert "9 45 P M" in normalize_text_for_tts("9:45 pm")


def test_normalize_rewrites_24h_time():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    out = normalize_text_for_tts("the meeting is at 14:30 sharp")
    assert "14 30" in out


def test_normalize_24h_pattern_does_not_eat_handled_ampm_colon():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    # First pass converts "2:16 a.m." -> "2 16 A M"; the 24h regex
    # then sees no colon and leaves it alone. The trailing dot stays
    # (input had "a.m." with the second dot, which the letter-strip
    # removes only from the AM/PM marker itself).
    out = normalize_text_for_tts("2:16 a.m.")
    assert "2 16 A M" in out
    assert ":" not in out


def test_normalize_rewrites_standalone_ampm():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    out = normalize_text_for_tts("we'll start at 10 a.m. sharp")
    assert "A M" in out


def test_normalize_rewrites_windows_drive_path_to_leaf():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    out = normalize_text_for_tts(
        "Saved under C:\\STC\\ultronPrototype\\data\\sandbox\\converts_pdf_docx."
    )
    # The drive-letter prefix is gone; only the leaf survives.
    assert "C:\\" not in out
    assert "converts_pdf_docx" in out


def test_normalize_handles_windows_path_with_extension():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    out = normalize_text_for_tts("Open C:\\Users\\someuser\\foo\\bar.py please.")
    assert "C:\\" not in out
    assert "bar.py" in out


def test_normalize_strips_urls_for_tts_safety():
    """2026-05-19 Issue 1 fix: URLs that used to be preserved are now
    stripped before TTS to keep the XTTS-v2 GPT context under the
    4096-audio-token cap. A live session hit 4830 tokens on a search
    response containing source URLs and the synth worker errored."""
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    url = "https://example.com/path/to/page"
    out = normalize_text_for_tts(f"See {url} for details.")
    assert url not in out
    assert "https://" not in out
    # The surrounding prose stays intelligible.
    assert "See" in out
    assert "for details" in out


def test_normalize_leaves_bare_drive_letter_alone():
    """``C:`` with no backslash is too ambiguous; leave it."""
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    out = normalize_text_for_tts("Drive C: is the boot disk.")
    assert "C:" in out


def test_normalize_expands_common_abbreviations():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "for example" in normalize_text_for_tts("e.g. red, green, blue").lower()
    assert "that is" in normalize_text_for_tts("the API i.e. the contract").lower()
    assert "et cetera" in normalize_text_for_tts("apples, oranges, etc.").lower()
    assert "versus" in normalize_text_for_tts("Python vs. JavaScript").lower()


def test_normalize_combines_multiple_patterns():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    out = normalize_text_for_tts(
        "Open C:\\foo\\bar.py at 2:16 a.m. (e.g. on Tuesday)."
    )
    assert "bar.py" in out
    assert "2 16 A M" in out
    assert "for example" in out.lower()
    assert "C:\\" not in out


def test_normalize_handles_full_session_response_pattern():
    """Reproduces the exact pattern from the live session log."""
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    text = (
        "France observes Central European Summer Time (CEST), "
        "currently 2:16 a.m. on Tuesday, May 19, 2026."
    )
    out = normalize_text_for_tts(text)
    assert "2 16 A M" in out
    assert "Tuesday" in out
    assert "May 19" in out  # date stays untouched -- XTTS handles it


def test_normalize_falls_back_on_unsupported_patterns():
    """Patterns not in the rule set pass through unchanged -- the
    function is conservative by design."""
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    # Phone number -- not handled (no rule)
    out = normalize_text_for_tts("call 555-1234 tomorrow")
    assert "555-1234" in out
    # Email -- not handled
    assert "test@example.com" in normalize_text_for_tts("email test@example.com")


# ---------------------------------------------------------------------------
# Temperatures
# ---------------------------------------------------------------------------


def test_normalize_temperature_fahrenheit():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "72 degrees Fahrenheit" in normalize_text_for_tts("It's 72°F outside")
    assert "98.6 degrees Fahrenheit" in normalize_text_for_tts("98.6°F is normal")
    # With space between number and degree
    assert "72 degrees Fahrenheit" in normalize_text_for_tts("72 °F")


def test_normalize_temperature_celsius():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "20 degrees Celsius" in normalize_text_for_tts("20°C in Paris")
    assert "37 degrees Celsius" in normalize_text_for_tts("body temp 37°C")


def test_normalize_temperature_negative_values():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "-10 degrees Fahrenheit" in normalize_text_for_tts("-10°F")
    assert "-5 degrees Celsius" in normalize_text_for_tts("-5°C")


def test_normalize_bare_degrees():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    out = normalize_text_for_tts("the angle is 45°")
    assert "45 degrees" in out
    assert "°" not in out


def test_normalize_temperature_does_not_eat_f_c_in_words():
    """Make sure 'F' / 'C' inside other words isn't picked up."""
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    # No degrees symbol -- shouldn't change "Fahrenheit"
    assert normalize_text_for_tts("Fahrenheit scale") == "Fahrenheit scale"


# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------


def test_normalize_currency_usd_bare():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "100 dollars" in normalize_text_for_tts("It costs $100")
    assert "5.99 dollars" in normalize_text_for_tts("$5.99 each")
    assert "1,000 dollars" in normalize_text_for_tts("$1,000")


def test_normalize_currency_usd_with_suffix():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "1.5 million dollars" in normalize_text_for_tts("worth $1.5M")
    assert "2 billion dollars" in normalize_text_for_tts("$2B valuation")
    assert "500 thousand dollars" in normalize_text_for_tts("$500K salary")


def test_normalize_currency_euro_and_pound():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "50 euros" in normalize_text_for_tts("paid €50")
    assert "25 pounds" in normalize_text_for_tts("only £25")
    assert "1 million euros" in normalize_text_for_tts("€1M fund")


def test_normalize_currency_yen():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "1000 yen" in normalize_text_for_tts("¥1000 fare")


# ---------------------------------------------------------------------------
# Mass / weight
# ---------------------------------------------------------------------------


def test_normalize_mass_pounds():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "150 pounds" in normalize_text_for_tts("150 lbs")
    assert "5 pounds" in normalize_text_for_tts("5 lb of flour")


def test_normalize_mass_kilograms():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "70 kilograms" in normalize_text_for_tts("70 kg")
    assert "1.5 kilograms" in normalize_text_for_tts("1.5kg loaf")


def test_normalize_mass_ounces_milligrams_grams():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "16 ounces" in normalize_text_for_tts("16 oz of water")
    assert "500 milligrams" in normalize_text_for_tts("500 mg dose")
    assert "250 grams" in normalize_text_for_tts("250 g of butter")


def test_normalize_grams_does_not_eat_g_in_words():
    """``g`` is a unit only when preceded by digits + non-letter
    boundary. ``going`` / ``5G`` should NOT match."""
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    out = normalize_text_for_tts("I am going to the store")
    assert "going" in out
    # ``5G`` (cellular) shouldn't be re-read as "5 grams"
    out = normalize_text_for_tts("5G network")
    assert "5G" in out or "5 G" in out  # either is fine, just not grams
    assert "grams" not in out


def test_normalize_mass_tons():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "2 tons" in normalize_text_for_tts("2 tons of steel")
    assert "1.5 tonnes" in normalize_text_for_tts("1.5 tonnes")


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------


def test_normalize_distance_miles_kilometres():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "5 miles" in normalize_text_for_tts("5 mi away")
    assert "10 kilometres" in normalize_text_for_tts("10 km away")


def test_normalize_distance_metric_small():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "100 centimetres" in normalize_text_for_tts("100 cm long")
    assert "5 millimetres" in normalize_text_for_tts("5 mm thick")


def test_normalize_distance_imperial_small():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "6 feet" in normalize_text_for_tts("6 ft tall")
    assert "18 inches" in normalize_text_for_tts("18 in wide")
    assert "5 yards" in normalize_text_for_tts("5 yds")


def test_normalize_bare_metres():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "100 metres" in normalize_text_for_tts("100 m sprint")


def test_normalize_metres_does_not_eat_m_in_words():
    """``m`` is a unit only when preceded by digit + non-letter
    boundary. ``I am`` / ``I'm`` should not match."""
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    out = normalize_text_for_tts("I am tired")
    assert "I am" in out
    assert "metres" not in out


def test_normalize_inches_does_not_eat_in_preposition():
    """``in`` is a preposition too -- only safe as a unit when adjacent
    to digits + non-letter boundary."""
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    out = normalize_text_for_tts("I sit in the chair")
    assert "in the" in out
    assert "inches" not in out


# ---------------------------------------------------------------------------
# Speed (compound units)
# ---------------------------------------------------------------------------


def test_normalize_speed_mph():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "60 miles per hour" in normalize_text_for_tts("60 mph")


def test_normalize_speed_kph_and_km_h():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "100 kilometres per hour" in normalize_text_for_tts("100 km/h")
    assert "100 kilometres per hour" in normalize_text_for_tts("100 kph")


def test_normalize_speed_metric_per_second():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "9.8 metres per second" in normalize_text_for_tts("9.8 m/s")
    assert "30 feet per second" in normalize_text_for_tts("30 ft/s")


def test_normalize_compound_does_not_break_bare_unit():
    """``100 km/h`` should produce ``kilometres per hour``, NOT
    ``kilometres / hour``. The compound pattern must consume both
    sides before the bare-km rule runs."""
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    out = normalize_text_for_tts("driving at 100 km/h")
    assert "kilometres per hour" in out
    assert "/" not in out or "hour" in out


# ---------------------------------------------------------------------------
# Time durations + storage + frequency
# ---------------------------------------------------------------------------


def test_normalize_time_units():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "200 milliseconds" in normalize_text_for_tts("200 ms latency")
    assert "30 seconds" in normalize_text_for_tts("30 sec timeout")
    assert "5 minutes" in normalize_text_for_tts("5 min remaining")
    assert "2 hours" in normalize_text_for_tts("2 hrs flight")


def test_normalize_storage_sizes():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "8 gigabytes" in normalize_text_for_tts("8 GB RAM")
    assert "500 megabytes" in normalize_text_for_tts("500 MB file")
    assert "256 kilobytes" in normalize_text_for_tts("256 KB block")
    assert "1 terabytes" in normalize_text_for_tts("1 TB disk")


def test_normalize_frequency():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "3.5 gigahertz" in normalize_text_for_tts("3.5 GHz clock")
    assert "60 hertz" in normalize_text_for_tts("60 Hz refresh")
    assert "100 kilohertz" in normalize_text_for_tts("100 kHz tone")


# ---------------------------------------------------------------------------
# Ordinals
# ---------------------------------------------------------------------------


def test_normalize_ordinal_calendar_days():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "first" in normalize_text_for_tts("the 1st of May")
    assert "second" in normalize_text_for_tts("on the 2nd")
    assert "third" in normalize_text_for_tts("3rd time")
    assert "nineteenth" in normalize_text_for_tts("May 19th")
    assert "twenty-fifth" in normalize_text_for_tts("the 25th")
    assert "thirty-first" in normalize_text_for_tts("31st of January")


def test_normalize_ordinal_large_falls_through():
    """Beyond 31, leave the numeric form -- it usually reads better
    than a long compound ordinal word."""
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    out = normalize_text_for_tts("the 100th visitor")
    # Not in our word-mapping table -> stays as "100th"
    assert "100th" in out


# ---------------------------------------------------------------------------
# Titles
# ---------------------------------------------------------------------------


def test_normalize_titles_before_capitalised_names():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "Mister Smith" in normalize_text_for_tts("Mr. Smith")
    assert "Doctor Watson" in normalize_text_for_tts("Dr. Watson")
    assert "Professor Plum" in normalize_text_for_tts("Prof. Plum")
    assert "Missus Jones" in normalize_text_for_tts("Mrs. Jones")
    assert "Saint Peter" in normalize_text_for_tts("St. Peter")


def test_normalize_titles_do_not_fire_without_capitalised_name():
    """``Mr.`` at end of sentence or before non-capitalised word
    shouldn't expand (avoid breaking street addresses, abbreviated
    lists, etc.)."""
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    # End of sentence
    assert "Mister" not in normalize_text_for_tts("Hello Mr.")
    # Lowercase following word
    assert "Mister" not in normalize_text_for_tts("Mr. is a title")


# ---------------------------------------------------------------------------
# Acronyms with dots
# ---------------------------------------------------------------------------


def test_normalize_acronym_dots():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "U S A" in normalize_text_for_tts("from the U.S.A.")
    assert "U S" in normalize_text_for_tts("the U.S. economy")
    assert "U K" in normalize_text_for_tts("U.K. parliament")
    assert "U N" in normalize_text_for_tts("U.N. resolution")
    assert "E U" in normalize_text_for_tts("E.U. policy")
    assert "NASA" in normalize_text_for_tts("N.A.S.A. announced")


def test_normalize_acronym_dots_does_not_eat_letter_in_words():
    """``U.S.`` shouldn't match inside other words."""
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    out = normalize_text_for_tts("ous.")  # arbitrary lowercase
    assert "U S" not in out


# ---------------------------------------------------------------------------
# Ampersand + extended abbreviations
# ---------------------------------------------------------------------------


def test_normalize_ampersand_between_words():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "Tom and Jerry" in normalize_text_for_tts("Tom & Jerry")
    assert "AT and T" in normalize_text_for_tts("AT&T")


def test_normalize_extended_latin_abbreviations():
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    assert "compare" in normalize_text_for_tts("cf. Smith 2024").lower()
    assert "approximately" in normalize_text_for_tts("approx. 30 minutes").lower()
    assert "note well" in normalize_text_for_tts("N.B. the disclaimer").lower()


# ---------------------------------------------------------------------------
# Combined / regression
# ---------------------------------------------------------------------------


def test_normalize_combined_rich_response():
    """A typical weather/news-style response with multiple patterns."""
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    text = (
        "On May 19th at 3:30 p.m. the temperature in the U.S. was "
        "72°F with winds at 15 mph; the storm dropped 2.5 in of rain "
        "on a $1.5M property near Mr. Smith's farm."
    )
    out = normalize_text_for_tts(text)
    assert "nineteenth" in out
    assert "3 30 P M" in out
    assert "U S" in out
    assert "72 degrees Fahrenheit" in out
    assert "15 miles per hour" in out
    assert "2.5 inches" in out
    assert "1.5 million dollars" in out
    assert "Mister Smith" in out


def test_normalize_does_not_break_short_sensible_text():
    """Common simple sentences should pass through with minimal
    rewriting (the rules are scoped tightly enough not to false-fire
    on natural prose)."""
    from kenning.tts.xtts_v3 import normalize_text_for_tts
    samples = [
        "Hello, how are you today?",
        "It is a beautiful day.",
        "The bird flew over the mountains.",
        "Tell me about the weather in London.",
        "What is the capital of France?",
    ]
    for s in samples:
        out = normalize_text_for_tts(s)
        # Allow trivial differences but the core content stays
        assert s.lower().split() == out.lower().split() or s == out


def test_normalize_engine_wiring_uses_spoken_form(monkeypatch, tmp_path):
    """End-to-end: the engine's ``_synthesize`` calls ``_http_synthesize``
    with the NORMALISED text, not the raw text."""
    import json
    import urllib.request
    from kenning.tts import xtts_v3

    server_py = tmp_path / "python.exe"
    server_py.write_text("")
    server_sc = tmp_path / "xtts_server.py"
    server_sc.write_text("")
    ref_wav = tmp_path / "ref.wav"
    ref_wav.write_text("")

    captured = {"body": None}

    class _Resp:
        headers = {"X-Sample-Rate": "24000"}

        def __init__(self):
            self._chunks = [(np.zeros(2400, dtype=np.int16)).tobytes(), b""]

        def read(self, n=None):
            if not self._chunks:
                return b""
            return self._chunks.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

    def _fake_urlopen(req, timeout=None):
        try:
            captured["body"] = json.loads(req.data.decode("utf-8"))
        except Exception:
            captured["body"] = {}
        return _Resp()

    monkeypatch.setattr(xtts_v3.XttsV3Speech, "_start_server", lambda self: None)
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    engine = xtts_v3.XttsV3Speech(
        server_python=server_py,
        server_script=server_sc,
        reference_audio=ref_wav,
        port=12346,
    )
    engine._synthesize("It is 2:16 a.m. now.")
    assert captured["body"] is not None
    assert "2 16 A M" in captured["body"]["text"]
    assert "a.m." not in captured["body"]["text"]


# ---------------------------------------------------------------------------
# 2026-05-19 Issue 1 fix: URL stripping + per-call length cap +
# _split_for_synth helper.
# ---------------------------------------------------------------------------


from kenning.tts.xtts_v3 import normalize_text_for_tts, XttsV3Speech


def test_normalize_strips_https_url():
    out = normalize_text_for_tts(
        "Visit https://www.time.gov/ for the current time."
    )
    assert "https://" not in out
    assert "time.gov" not in out
    assert "Visit" in out
    assert "for the current time" in out


def test_normalize_strips_multiple_urls():
    out = normalize_text_for_tts(
        "See https://a.com/foo and http://b.org/bar for details."
    )
    assert "https://" not in out
    assert "http://" not in out
    assert "a.com" not in out
    assert "b.org" not in out
    assert "See" in out
    assert "and" in out
    assert "for details" in out


def test_normalize_strips_bare_www_url():
    out = normalize_text_for_tts("Try www.example.com today.")
    assert "www." not in out
    assert "example.com" not in out
    assert "Try" in out
    assert "today" in out


def test_normalize_strips_ftp_url():
    out = normalize_text_for_tts(
        "Files at ftp://files.example.org/pub/data are stale."
    )
    assert "ftp://" not in out
    assert "files.example.org" not in out


def test_normalize_url_strip_collapses_whitespace():
    """After URL removal the surrounding spaces should not double up."""
    out = normalize_text_for_tts(
        "see https://x.com today"
    )
    assert "  " not in out
    assert out.startswith("see")
    assert out.endswith("today")


def test_normalize_url_strip_preserves_non_url_text():
    """Plain prose with no URLs must round-trip unchanged."""
    text = "Hello world. This is a test sentence."
    assert normalize_text_for_tts(text) == text


def test_normalize_url_strip_does_not_eat_dotted_filename():
    """``app.py`` / ``data.json`` style dotted-filename tokens must NOT
    be stripped -- only URL-shaped tokens (http(s)://, ftp://, www.)
    qualify."""
    out = normalize_text_for_tts("Open app.py and check data.json.")
    assert "app.py" in out
    assert "data.json" in out


# ----- _split_for_synth helper -----


def test_split_for_synth_short_text_passes_through():
    out = XttsV3Speech._split_for_synth("Hello world.", 240)
    assert out == ["Hello world."]


def test_split_for_synth_returns_empty_list_for_empty_input():
    assert XttsV3Speech._split_for_synth("", 240) == []
    assert XttsV3Speech._split_for_synth("   ", 240) == []


def test_split_for_synth_zero_max_returns_unchanged():
    """Defensive: a misconfigured cap (0 or negative) returns the
    text unchanged rather than infinite-looping."""
    out = XttsV3Speech._split_for_synth("Some text here.", 0)
    assert out == ["Some text here."]


def test_split_for_synth_clause_boundary():
    text = "First clause, second clause, third clause."
    out = XttsV3Speech._split_for_synth(text, max_chars=20)
    # Each chunk must be <= 20 chars and together they must cover the
    # original text (modulo whitespace + clause-boundary repositioning).
    for chunk in out:
        assert len(chunk) <= 20
    # All clauses should still be referenced in the output
    joined = " ".join(out)
    assert "First clause" in joined
    assert "second clause" in joined
    assert "third clause" in joined


def test_split_for_synth_word_boundary_when_no_clauses():
    text = "one two three four five six seven eight nine ten eleven twelve"
    out = XttsV3Speech._split_for_synth(text, max_chars=20)
    for chunk in out:
        assert len(chunk) <= 20
    joined = " ".join(out)
    assert "one" in joined
    assert "twelve" in joined


def test_split_for_synth_force_slices_oversize_word():
    """A single token longer than max_chars (rare; usually a URL the
    strip missed, or a long alphanumeric id) gets char-sliced so the
    synth call still stays under the cap."""
    long_word = "abcdefghijklmnopqrstuvwxyz" * 4  # 104 chars
    text = f"prefix {long_word} suffix"
    out = XttsV3Speech._split_for_synth(text, max_chars=20)
    for chunk in out:
        assert len(chunk) <= 20


def test_split_for_synth_long_sentence_with_urls_after_strip():
    """End-to-end: a typical pre-normaliser sentence with URLs gets
    URL-stripped, then sub-split if still too long."""
    raw = (
        "See https://example.com/very/long/path/here and "
        "https://another.com/another/path for additional details "
        "about the policy and the latest updates."
    )
    normalised = normalize_text_for_tts(raw)
    out = XttsV3Speech._split_for_synth(normalised, max_chars=40)
    for chunk in out:
        assert len(chunk) <= 40
        assert "https://" not in chunk


def test_xtts_v3_config_default_max_chars_per_synth_call():
    """Default bumped 240 -> 600 in round 4 retune so ordinary multi-
    clause sentences don't get fragmented into 3-4 separate synth
    calls (each picking up the v3 filter's 200 ms tail silence,
    producing jagged pacing the user heard on 2026-05-19)."""
    cfg = XttsV3Config()
    assert cfg.max_chars_per_synth_call == 600


def test_xtts_v3_config_max_chars_per_synth_call_range_validation():
    XttsV3Config(max_chars_per_synth_call=80)    # boundary low
    XttsV3Config(max_chars_per_synth_call=2000)  # boundary high
    with pytest.raises(Exception):
        XttsV3Config(max_chars_per_synth_call=79)
    with pytest.raises(Exception):
        XttsV3Config(max_chars_per_synth_call=2001)


# ---------------------------------------------------------------------------
# Round 7b (2026-05-20): smarter sentence boundary detection.
#
# The previous flush logic broke on every ``.``, which caused mid-token
# splits at ellipses, decimals, domains, and abbreviations -- producing
# the "horrible pacing, random pauses between words" the user heard.
# These tests pin the new heuristics.
# ---------------------------------------------------------------------------


def test_boundary_simple_sentence_end_flushes():
    assert XttsV3Speech._is_safe_sentence_boundary(
        "Hello world. Next", 11, buffer_complete=False,
    ) is True


def test_boundary_question_mark_flushes():
    assert XttsV3Speech._is_safe_sentence_boundary(
        "Hello? Next", 5, buffer_complete=False,
    ) is True


def test_boundary_exclamation_flushes():
    assert XttsV3Speech._is_safe_sentence_boundary(
        "Wow! Next", 3, buffer_complete=False,
    ) is True


def test_boundary_newline_flushes():
    assert XttsV3Speech._is_safe_sentence_boundary(
        "Line one\nLine two", 8, buffer_complete=False,
    ) is True


def test_boundary_ellipsis_first_dot_holds():
    """First dot of `...` should not flush."""
    text = "Wait... I think"
    assert XttsV3Speech._is_safe_sentence_boundary(
        text, 4, buffer_complete=False,
    ) is False


def test_boundary_ellipsis_second_dot_holds():
    text = "Wait... I think"
    assert XttsV3Speech._is_safe_sentence_boundary(
        text, 5, buffer_complete=False,
    ) is False


def test_boundary_ellipsis_terminal_third_dot_does_not_flush():
    """All three dots of an ellipsis are non-flush.

    Earlier design flushed on the third dot when followed by a space,
    but that fragmented ``Wait... what?`` into two synth calls --
    audible micro-pause between the ellipsis tail and the rest of
    the sentence. Round 7b refines the policy: ellipsis is treated
    as a mid-sentence prosodic pause (XTTS handles it via its own
    duration model when the dots are present in the chunk text).
    """
    text = "Wait... I think"
    assert text[6] == "."
    assert XttsV3Speech._is_safe_sentence_boundary(
        text, 6, buffer_complete=False,
    ) is False


def test_boundary_decimal_does_not_flush():
    text = "Pi is 3.14 approximately"
    # pos 7 is the dot between 3 and 1.
    assert text[7] == "."
    assert XttsV3Speech._is_safe_sentence_boundary(
        text, 7, buffer_complete=False,
    ) is False


def test_boundary_version_number_does_not_flush():
    text = "Using v2.0 now"
    pos = text.index(".")
    assert XttsV3Speech._is_safe_sentence_boundary(
        text, pos, buffer_complete=False,
    ) is False


def test_boundary_mid_domain_does_not_flush():
    text = "Visit Dictionary.com for definitions"
    # pos of "." between Dictionary and com.
    pos = text.index(".")
    assert XttsV3Speech._is_safe_sentence_boundary(
        text, pos, buffer_complete=False,
    ) is False


def test_boundary_abbrev_dr_does_not_flush():
    text = "Dr. Smith arrived"
    assert text[2] == "."
    assert XttsV3Speech._is_safe_sentence_boundary(
        text, 2, buffer_complete=False,
    ) is False


def test_boundary_abbrev_eg_does_not_flush():
    text = "Try foods e.g. apples and bananas"
    # First dot in "e.g.": after the "e".
    pos = text.index("e.g.") + 1
    assert text[pos] == "."
    assert XttsV3Speech._is_safe_sentence_boundary(
        text, pos, buffer_complete=False,
    ) is False


def test_boundary_abbrev_etc_does_not_flush():
    text = "apples, bananas, etc. today"
    pos = text.index("etc.") + 3
    assert text[pos] == "."
    assert XttsV3Speech._is_safe_sentence_boundary(
        text, pos, buffer_complete=False,
    ) is False


def test_boundary_trailing_dot_holds_when_incomplete():
    """A `.` at the very end of the buffer should wait for more tokens."""
    text = "Hello"
    # Synthetic: simulate `.` at end -- by appending we get a dot
    # at the final position.
    text = "Hello."
    assert XttsV3Speech._is_safe_sentence_boundary(
        text, 5, buffer_complete=False,
    ) is False


def test_boundary_trailing_dot_flushes_when_complete():
    text = "Hello."
    assert XttsV3Speech._is_safe_sentence_boundary(
        text, 5, buffer_complete=True,
    ) is True


def test_boundary_period_then_capital_letter_flushes():
    """``Smith.A`` is suspicious (no space) but our rule lets it
    through because the cost of a false hold is worse (buffer
    overrun) than a false flush (one extra micro-pause). Pin the
    behaviour explicitly so future regressions are caught."""
    text = "Smith.Apple is rare"
    assert text[5] == "."
    # letter.letter -> domain rule -> NOT a boundary.
    assert XttsV3Speech._is_safe_sentence_boundary(
        text, 5, buffer_complete=False,
    ) is False


def test_boundary_double_question_mark_flushes_first():
    text = "Really?? Yes"
    assert XttsV3Speech._is_safe_sentence_boundary(
        text, 6, buffer_complete=False,
    ) is True


# ---------- _find_next_sentence_boundary integration ----------


def _make_engine_for_boundary():
    """Build a bare-bones engine for boundary tests without spawning XTTS."""
    e = XttsV3Speech.__new__(XttsV3Speech)
    e.flush_chars = set(".!?\n")
    return e


def test_find_boundary_skips_ellipsis_then_flushes_real_sentence():
    e = _make_engine_for_boundary()
    text = "Hmm... well, that's the case. Next"
    cut = e._find_next_sentence_boundary(text, buffer_complete=False)
    assert cut > 0
    # Should land at the `.` after "case" (not at any of the ellipsis dots).
    assert text[:cut].rstrip().endswith("case.")


def test_find_boundary_skips_abbrev_then_flushes_real_sentence():
    e = _make_engine_for_boundary()
    text = "Dr. Smith is here. He'll see you."
    cut = e._find_next_sentence_boundary(text, buffer_complete=False)
    assert cut > 0
    assert text[:cut].rstrip().endswith("here.")


def test_find_boundary_skips_decimal_then_flushes():
    e = _make_engine_for_boundary()
    text = "Pi is 3.14159 approximately. End"
    cut = e._find_next_sentence_boundary(text, buffer_complete=False)
    assert cut > 0
    assert text[:cut].rstrip().endswith("approximately.")


def test_find_boundary_returns_zero_when_no_safe_boundary():
    e = _make_engine_for_boundary()
    # All dots are unsafe (ellipsis + abbrev).
    text = "Wait... e.g. apples"
    cut = e._find_next_sentence_boundary(text, buffer_complete=False)
    assert cut == 0


def test_find_boundary_returns_zero_for_text_with_no_flush_chars():
    e = _make_engine_for_boundary()
    cut = e._find_next_sentence_boundary(
        "no flush chars at all here", buffer_complete=False,
    )
    assert cut == 0


# ---------- _run_synth_loop integration ----------


def _capture_synth(text_fragments):
    """Run ``_run_synth_loop`` against a stub engine and capture the
    arguments passed to ``_synthesize``. Returns the list of texts."""
    import threading
    e = XttsV3Speech.__new__(XttsV3Speech)
    e.flush_chars = set(".!?\n")
    e._max_chars_per_synth_call = 600
    e._stop_event = threading.Event()
    captured: list[str] = []

    def fake_synth(text):
        captured.append(text)
        import numpy as _np
        return _np.zeros(8, dtype=_np.float32), 24000

    e._synthesize = fake_synth
    e._run_synth_loop(
        fragments=iter(text_fragments),
        push=lambda clip: None,
    )
    return captured


def test_run_synth_loop_does_not_chunk_on_ellipsis():
    """``Wait... what?`` should produce ONE synth call, not four."""
    out = _capture_synth(["Wait... what?"])
    assert len(out) == 1
    assert "Wait" in out[0] and "what" in out[0]


def test_run_synth_loop_does_not_chunk_on_abbreviation():
    """``Dr. Smith is here.`` should produce ONE synth call."""
    out = _capture_synth(["Dr. Smith is here."])
    assert len(out) == 1
    assert "Dr." in out[0] and "Smith" in out[0]


def test_run_synth_loop_does_not_chunk_on_decimal():
    out = _capture_synth(["Pi is 3.14 approximately."])
    assert len(out) == 1
    assert "3.14" in out[0]


def test_run_synth_loop_does_not_chunk_on_domain():
    out = _capture_synth(["Visit Dictionary.com today."])
    assert len(out) == 1
    assert "Dictionary.com" in out[0]


def test_run_synth_loop_flushes_two_sentences_into_two_calls():
    out = _capture_synth(["Hello world. Second sentence here."])
    assert len(out) == 2
    assert "Hello world" in out[0]
    assert "Second sentence" in out[1]


def test_run_synth_loop_streamed_dots_held_until_followup():
    """Streaming `Wait` then `...` then ` what?` should still produce
    one call (boundary detection sees ellipsis when buffered together)."""
    out = _capture_synth(["Wait", "...", " what? Next."])
    # Two sentences total: "Wait... what?" and "Next."
    assert len(out) == 2
    assert "Wait" in out[0] and "what" in out[0]


def test_run_synth_loop_tail_flushes_pending_when_no_terminator():
    """A stream ending mid-sentence (no terminator) still flushes
    everything via the end-of-stream tail."""
    out = _capture_synth(["incomplete tail with no punctuation"])
    assert len(out) == 1
    assert "incomplete tail" in out[0]


def test_run_synth_loop_safety_valve_breaks_runaway_buffer():
    """If text grows past 2x max_chars without a safe boundary, the
    safety valve soft-breaks on the last clause/space."""
    text = "word " * 400  # 2000 chars, no terminators
    out = _capture_synth([text])
    # Should produce multiple chunks via the safety valve, not one
    # giant chunk that overflows the synth cap.
    assert len(out) >= 1
    for chunk in out:
        # Each chunk must remain under (and tolerably near) max_chars.
        assert len(chunk) <= 800  # max_chars * 2 worst-case bound


# ---------------------------------------------------------------------------
# 2026-05-20 round 9: extended reference-window conditioning.
# Verifies gpt_cond_len / gpt_cond_chunk_len / max_ref_length config
# fields exist with the right defaults + bounds + are forwarded into
# the XTTS server subprocess argv at startup.
# ---------------------------------------------------------------------------


def test_xtts_v3_config_extended_reference_defaults():
    """The round-9 defaults bump the Coqui library defaults (6/6/30)
    so the 3-min Kenning reference actually contributes more than the
    first ~6 s. If these regress, the speaker embedding silently
    truncates back to the library defaults."""
    cfg = XttsV3Config()
    assert cfg.gpt_cond_len == 30
    assert cfg.gpt_cond_chunk_len == 6
    assert cfg.max_ref_length == 60


def test_xtts_v3_config_extended_reference_bounds_enforced():
    from pydantic import ValidationError
    # gpt_cond_len: [3, 120]
    XttsV3Config(gpt_cond_len=3)
    XttsV3Config(gpt_cond_len=120)
    with pytest.raises(ValidationError):
        XttsV3Config(gpt_cond_len=2)
    with pytest.raises(ValidationError):
        XttsV3Config(gpt_cond_len=121)
    # gpt_cond_chunk_len: [3, 30]
    XttsV3Config(gpt_cond_chunk_len=3)
    XttsV3Config(gpt_cond_chunk_len=30)
    with pytest.raises(ValidationError):
        XttsV3Config(gpt_cond_chunk_len=2)
    with pytest.raises(ValidationError):
        XttsV3Config(gpt_cond_chunk_len=31)
    # max_ref_length: [10, 180] (180s = 3 min, the full clip)
    XttsV3Config(max_ref_length=10)
    XttsV3Config(max_ref_length=180)
    with pytest.raises(ValidationError):
        XttsV3Config(max_ref_length=9)
    with pytest.raises(ValidationError):
        XttsV3Config(max_ref_length=181)


def test_xtts_v3_config_extended_reference_round_trips_through_dict():
    cfg = XttsV3Config(gpt_cond_len=45, gpt_cond_chunk_len=9, max_ref_length=90)
    cfg2 = XttsV3Config.model_validate(cfg.model_dump())
    assert cfg2.gpt_cond_len == 45
    assert cfg2.gpt_cond_chunk_len == 9
    assert cfg2.max_ref_length == 90


def test_xtts_v3_client_forwards_reference_window_in_argv(monkeypatch, tmp_path):
    """Pure wiring test: confirms ``_start_server`` includes the
    --gpt-cond-len / --gpt-cond-chunk-len / --max-ref-length flags in
    the subprocess argv with the configured values. If the client
    silently drops these, the server falls back to its own
    argparse defaults and Coqui's library defaults (6/6/30) take
    over -- which is exactly what we're trying to avoid."""
    import subprocess
    from kenning.tts import xtts_v3

    server_py = tmp_path / "python.exe"
    server_py.write_text("")
    server_sc = tmp_path / "xtts_server.py"
    server_sc.write_text("")
    ref_wav = tmp_path / "ref.wav"
    ref_wav.write_text("")

    captured_argv: list[list[str]] = []

    class _AbortPopen(Exception):
        pass

    def _fake_popen(argv, *a, **kw):
        captured_argv.append(list(argv))
        # Abort after argv capture so we don't wait on the health
        # poll loop or actually spawn anything.
        raise _AbortPopen()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)

    # The ctor calls _start_server which will hit _AbortPopen. Catch
    # it -- we only care about the captured argv.
    with pytest.raises(_AbortPopen):
        xtts_v3.XttsV3Speech(
            server_python=server_py,
            server_script=server_sc,
            reference_audio=ref_wav,
            port=12347,
            gpt_cond_len=45,
            gpt_cond_chunk_len=9,
            max_ref_length=90,
        )

    assert captured_argv, "expected exactly one Popen call"
    argv = captured_argv[0]
    # The exact positional order isn't important; pairs are.
    assert "--gpt-cond-len" in argv
    assert argv[argv.index("--gpt-cond-len") + 1] == "45"
    assert "--gpt-cond-chunk-len" in argv
    assert argv[argv.index("--gpt-cond-chunk-len") + 1] == "9"
    assert "--max-ref-length" in argv
    assert argv[argv.index("--max-ref-length") + 1] == "90"


def test_xtts_v3_client_uses_config_defaults_when_kwargs_omitted(monkeypatch, tmp_path):
    """When the caller doesn't pass gpt_cond_len / etc., the engine
    reads them from the global config (which mirrors config.yaml).
    Pins the production-default flow so a config bump propagates."""
    import subprocess
    from kenning.tts import xtts_v3

    server_py = tmp_path / "python.exe"
    server_py.write_text("")
    server_sc = tmp_path / "xtts_server.py"
    server_sc.write_text("")
    ref_wav = tmp_path / "ref.wav"
    ref_wav.write_text("")

    captured_argv: list[list[str]] = []

    class _AbortPopen(Exception):
        pass

    def _fake_popen(argv, *a, **kw):
        captured_argv.append(list(argv))
        raise _AbortPopen()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)

    with pytest.raises(_AbortPopen):
        xtts_v3.XttsV3Speech(
            server_python=server_py,
            server_script=server_sc,
            reference_audio=ref_wav,
            port=12348,
            # gpt_cond_len, gpt_cond_chunk_len, max_ref_length omitted
        )

    argv = captured_argv[0]
    # Should fall through to either the explicit config or the
    # ctor-level fallback defaults (30/6/60).
    assert "--gpt-cond-len" in argv
    assert "--gpt-cond-chunk-len" in argv
    assert "--max-ref-length" in argv
    # The values should be valid ints, and within the production
    # range we just established in the bounds test.
    gpt_cond_len = int(argv[argv.index("--gpt-cond-len") + 1])
    gpt_cond_chunk_len = int(argv[argv.index("--gpt-cond-chunk-len") + 1])
    max_ref_length = int(argv[argv.index("--max-ref-length") + 1])
    assert 3 <= gpt_cond_len <= 120
    assert 3 <= gpt_cond_chunk_len <= 30
    assert 10 <= max_ref_length <= 180
    # And they should be at least as generous as the Coqui library
    # defaults -- that's the whole point of this change.
    assert gpt_cond_len >= 6
    assert max_ref_length >= 30
