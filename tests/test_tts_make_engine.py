"""Tests for the :func:`kenning.tts.make_tts_engine` factory.

The factory is the canonical TTS-construction surface used by both
``orchestrator._load_tts_engine`` and ``scripts/measure_baseline.py``.
Both paths route through this function so they always exercise the
same code; these tests pin the contract down.

No real model loads happen — the engine classes and the RVC helper
are monkeypatched.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kenning.tts import make_tts_engine


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeKokoroSpeech:
    last_kwargs: dict | None = None

    def __init__(self, **kwargs) -> None:
        type(self).last_kwargs = kwargs
        self.kwargs = kwargs


class _FakeXttsV3Speech:
    constructed = 0

    def __init__(self) -> None:
        type(self).constructed += 1


class _FakeTextToSpeech:
    last_rvc = "uninitialized"

    def __init__(self, *, rvc=None) -> None:
        type(self).last_rvc = rvc
        self.rvc = rvc


class _FakeRvcConverter:
    constructed = 0

    def __init__(self) -> None:
        type(self).constructed += 1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_engines(monkeypatch):
    """Swap the real engine classes for cheap fakes."""
    import kenning.tts as tts_module

    _FakeKokoroSpeech.last_kwargs = None
    _FakeXttsV3Speech.constructed = 0
    _FakeTextToSpeech.last_rvc = "uninitialized"
    _FakeRvcConverter.constructed = 0

    monkeypatch.setattr(tts_module, "KokoroSpeech", _FakeKokoroSpeech)
    monkeypatch.setattr(tts_module, "XttsV3Speech", _FakeXttsV3Speech)
    monkeypatch.setattr(tts_module, "TextToSpeech", _FakeTextToSpeech)
    monkeypatch.setattr(tts_module, "RvcConverter", _FakeRvcConverter)
    yield


def _kokoro_subcfg() -> SimpleNamespace:
    """Match the subset of attributes ``make_tts_engine`` reads from the
    real ``KokoroConfig``."""
    return SimpleNamespace(
        model_path="models/kokoro",
        voice="kenning",
        device="cuda",
        speed=1.3,
        apply_runtime_filter=False,
        filter_preset="v3_heavy",
        apply_spectral_smooth=False,
        spectral_smooth_window=5,
        apply_trim_fade=True,
        trim_fade_threshold_db=-40.0,
        f0_contour_factor=1.4,
        f0_shift_semitones=-0.5,
        f0_max_excursion=4.5,
        f0_energy_factor=1.2,
        dur_final_factor=1.3,
        dur_internal_factor=1.18,
        dur_stress_factor=1.08,
        max_pause_cap_ms=520.0,
    )


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


def test_kokoro_path_returns_none_rvc(patched_engines):
    cfg = SimpleNamespace(engine="kokoro", kokoro=_kokoro_subcfg())
    rvc, tts = make_tts_engine(cfg)
    assert rvc is None
    assert isinstance(tts, _FakeKokoroSpeech)
    # The factory should forward every Kokoro knob unchanged.
    kw = _FakeKokoroSpeech.last_kwargs
    assert kw["voice"] == "kenning"
    assert kw["device"] == "cuda"
    assert kw["speed"] == 1.3
    assert kw["apply_trim_fade"] is True
    assert kw["apply_spectral_smooth"] is False


def test_xtts_v3_path_returns_none_rvc(patched_engines):
    cfg = SimpleNamespace(engine="xtts_v3")
    rvc, tts = make_tts_engine(cfg)
    assert rvc is None
    assert isinstance(tts, _FakeXttsV3Speech)
    assert _FakeXttsV3Speech.constructed == 1


def test_piper_rvc_path_with_rvc_disabled(patched_engines, monkeypatch):
    """When ``settings.RVC_ENABLED is False`` the factory returns
    ``(None, TextToSpeech(rvc=None))`` -- plain Piper, no RVC."""
    # Patch settings.RVC_ENABLED to False
    from config import settings
    monkeypatch.setattr(settings, "RVC_ENABLED", False)
    cfg = SimpleNamespace(engine="piper_rvc")
    rvc, tts = make_tts_engine(cfg)
    assert rvc is None
    assert isinstance(tts, _FakeTextToSpeech)
    assert _FakeTextToSpeech.last_rvc is None
    assert _FakeRvcConverter.constructed == 0


def test_piper_rvc_path_with_rvc_enabled_but_model_missing(
    patched_engines, monkeypatch, tmp_path,
):
    """RVC enabled but model file missing -> WARN + return (None, TTS)."""
    from config import settings
    monkeypatch.setattr(settings, "RVC_ENABLED", True)
    monkeypatch.setattr(
        settings, "RVC_MODEL_PATH", tmp_path / "missing_rvc.pth",
    )
    cfg = SimpleNamespace(engine="piper_rvc")
    rvc, tts = make_tts_engine(cfg)
    assert rvc is None
    assert isinstance(tts, _FakeTextToSpeech)
    assert _FakeRvcConverter.constructed == 0


def test_piper_rvc_path_with_rvc_enabled_and_model_present(
    patched_engines, monkeypatch, tmp_path,
):
    """Happy path: RVC enabled, model on disk -> RVC constructed and
    handed to TextToSpeech."""
    fake_model = tmp_path / "rvc.pth"
    fake_model.write_bytes(b"\x00")
    from config import settings
    monkeypatch.setattr(settings, "RVC_ENABLED", True)
    monkeypatch.setattr(settings, "RVC_MODEL_PATH", fake_model)
    cfg = SimpleNamespace(engine="piper_rvc")
    rvc, tts = make_tts_engine(cfg)
    assert isinstance(rvc, _FakeRvcConverter)
    assert isinstance(tts, _FakeTextToSpeech)
    assert _FakeTextToSpeech.last_rvc is rvc


def test_piper_rvc_path_with_rvc_load_failure_degrades(
    patched_engines, monkeypatch, tmp_path,
):
    """If ``RvcConverter()`` raises, the factory must NOT propagate;
    it should fall back to plain Piper. This is the fail-open contract."""
    fake_model = tmp_path / "rvc.pth"
    fake_model.write_bytes(b"\x00")
    from config import settings
    monkeypatch.setattr(settings, "RVC_ENABLED", True)
    monkeypatch.setattr(settings, "RVC_MODEL_PATH", fake_model)

    def _exploding_rvc():
        raise RuntimeError("simulated RVC load failure")

    import kenning.tts as tts_module
    monkeypatch.setattr(tts_module, "RvcConverter", _exploding_rvc)

    cfg = SimpleNamespace(engine="piper_rvc")
    rvc, tts = make_tts_engine(cfg)
    assert rvc is None
    assert isinstance(tts, _FakeTextToSpeech)
    assert _FakeTextToSpeech.last_rvc is None


def test_unknown_engine_raises(patched_engines):
    cfg = SimpleNamespace(engine="not_a_real_engine")
    with pytest.raises(RuntimeError) as exc_info:
        make_tts_engine(cfg)
    assert "not_a_real_engine" in str(exc_info.value)


def test_factory_pulls_from_get_config_when_cfg_is_none(
    patched_engines, monkeypatch,
):
    """``cfg=None`` -> read ``get_config().tts``. Production callers
    (orchestrator + measure_baseline) rely on this default behaviour."""
    fake_tts_cfg = SimpleNamespace(engine="kokoro", kokoro=_kokoro_subcfg())
    fake_root = SimpleNamespace(tts=fake_tts_cfg)
    import kenning.config as cfg_module
    monkeypatch.setattr(cfg_module, "get_config", lambda: fake_root)

    rvc, tts = make_tts_engine()  # no cfg argument
    assert rvc is None
    assert isinstance(tts, _FakeKokoroSpeech)


def test_kokoro_path_handles_missing_subcfg(patched_engines):
    """Defensive: a TTS config without the ``kokoro`` sub-section
    should not crash -- the factory falls through to KokoroSpeech()
    with empty kwargs (i.e. KokoroSpeech defaults). This guards
    against a config-loader change accidentally dropping the
    sub-section."""
    cfg = SimpleNamespace(engine="kokoro")  # no .kokoro attribute
    rvc, tts = make_tts_engine(cfg)
    assert rvc is None
    assert isinstance(tts, _FakeKokoroSpeech)
    # Empty kwargs -> the fake records `{}`.
    assert _FakeKokoroSpeech.last_kwargs == {}
