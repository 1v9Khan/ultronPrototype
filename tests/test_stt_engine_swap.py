"""STT engine swap tests (frontier item 5, 2026-05-21).

Tests for the engine factory ``make_stt_engine`` that selects between
:class:`WhisperEngine` and :class:`ParakeetEngine` based on
``stt.engine`` config + NeMo availability.

No real model load happens here -- both engines are mocked. Tests
cover:
- Config schema accepts ``engine: auto|whisper|parakeet`` with safe defaults.
- Factory ``auto`` resolution: NeMo present -> Parakeet; absent -> Whisper.
- Factory ``parakeet`` (explicit): raises ImportError when NeMo missing
  (so operators see the issue immediately, not mid-turn).
- Factory ``whisper`` (explicit): always Whisper, even if NeMo present.
- ParakeetEngine constructor raises clearly when NeMo unavailable.
- Whisper construction failure does NOT happen in auto-when-Parakeet-fails
  fallback (the factory falls back to Whisper, doesn't re-raise the
  Parakeet error).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ultron.config import STTConfig, UltronConfig


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


def test_stt_default_engine_is_auto():
    """The frontier-enhancement Item 5 introduces ``stt.engine: auto``
    as the default -- automatically uses Parakeet when NeMo is
    available, falls back to Whisper otherwise. This way fresh
    installs without NeMo continue working with Whisper, while
    operators who ``pip install nemo_toolkit[asr]`` get Parakeet
    activated with no extra config flip."""
    cfg = STTConfig()
    assert cfg.engine == "auto"


def test_stt_explicit_engine_swap_back():
    """Setting ``engine: whisper`` is the clean swap-back path if
    Parakeet misbehaves."""
    cfg = STTConfig(engine="whisper")
    assert cfg.engine == "whisper"
    cfg2 = STTConfig(engine="parakeet")
    assert cfg2.engine == "parakeet"


def test_stt_parakeet_model_default():
    cfg = STTConfig()
    assert cfg.parakeet_model == "nvidia/parakeet-tdt-0.6b-v3"
    assert cfg.parakeet_device == "cuda"


def test_full_config_round_trip_with_explicit_whisper():
    cfg = UltronConfig.model_validate({"stt": {"engine": "whisper"}})
    assert cfg.stt.engine == "whisper"


def test_stt_engine_validates_literal():
    """Invalid engine name rejected by pydantic."""
    with pytest.raises(Exception):                                       # noqa: PT011
        STTConfig(engine="bogus")


# ---------------------------------------------------------------------------
# Factory behaviour
# ---------------------------------------------------------------------------


def _patch_engines(
    monkeypatch,
    nemo_available: bool = False,
    parakeet_raises: BaseException | None = None,
    whisper_raises: BaseException | None = None,
):
    """Patch both engines + NeMo availability check. Returns the
    two engine mock classes so the test can inspect calls."""
    import ultron.transcription as factory_module
    from ultron.transcription import parakeet_engine as parakeet_module

    monkeypatch.setattr(
        parakeet_module, "is_nemo_available", lambda: nemo_available,
    )
    monkeypatch.setattr(
        factory_module, "is_nemo_available", lambda: nemo_available,
    )
    # Stub the server-spawn helper so tests don't try to actually
    # subprocess.Popen the parakeet_server.py (which would fail in
    # CI / on machines without the .venv-parakeet venv).
    monkeypatch.setattr(
        parakeet_module, "_spawn_server_if_needed",
        lambda cfg: "http://127.0.0.1:8771",
    )

    parakeet_cls = MagicMock()
    if parakeet_raises:
        parakeet_cls.side_effect = parakeet_raises
    else:
        parakeet_cls.return_value = MagicMock(name="ParakeetEngine_instance")

    whisper_cls = MagicMock()
    if whisper_raises:
        whisper_cls.side_effect = whisper_raises
    else:
        whisper_cls.return_value = MagicMock(name="WhisperEngine_instance")

    monkeypatch.setattr(factory_module, "ParakeetEngine", parakeet_cls)
    monkeypatch.setattr(factory_module, "WhisperEngine", whisper_cls)
    return parakeet_cls, whisper_cls


def _stub_stt_cfg(**overrides):
    base = dict(
        engine="auto",
        parakeet_model="nvidia/parakeet-tdt-0.6b-v3",
        parakeet_device="cuda",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_factory_auto_with_nemo_available_picks_parakeet(monkeypatch):
    """``engine: auto`` + NeMo installed -> Parakeet is selected."""
    from ultron.transcription import make_stt_engine

    parakeet_cls, whisper_cls = _patch_engines(monkeypatch, nemo_available=True)
    cfg = _stub_stt_cfg(engine="auto")
    eng = make_stt_engine(cfg)
    parakeet_cls.assert_called_once()
    whisper_cls.assert_not_called()
    assert eng is parakeet_cls.return_value


def test_factory_auto_without_nemo_falls_back_to_whisper(monkeypatch):
    """``engine: auto`` + NeMo missing -> Whisper transparently."""
    from ultron.transcription import make_stt_engine

    parakeet_cls, whisper_cls = _patch_engines(monkeypatch, nemo_available=False)
    cfg = _stub_stt_cfg(engine="auto")
    eng = make_stt_engine(cfg)
    parakeet_cls.assert_not_called()
    whisper_cls.assert_called_once()
    assert eng is whisper_cls.return_value


def test_factory_auto_parakeet_load_failure_falls_back(monkeypatch):
    """``engine: auto`` + NeMo present but Parakeet construction fails
    -> Whisper fallback with a WARN log. The voice path keeps working."""
    from ultron.transcription import make_stt_engine

    parakeet_cls, whisper_cls = _patch_engines(
        monkeypatch,
        nemo_available=True,
        parakeet_raises=RuntimeError("simulated load failure"),
    )
    cfg = _stub_stt_cfg(engine="auto")
    eng = make_stt_engine(cfg)
    parakeet_cls.assert_called_once()
    whisper_cls.assert_called_once()
    assert eng is whisper_cls.return_value


def test_factory_explicit_parakeet_raises_when_nemo_missing(monkeypatch):
    """``engine: parakeet`` (explicit) with NeMo missing -> ImportError.
    This is intentional -- the user explicitly asked for Parakeet, so
    silently falling back would hide a misconfiguration."""
    from ultron.transcription import make_stt_engine

    _patch_engines(monkeypatch, nemo_available=False)
    cfg = _stub_stt_cfg(engine="parakeet")
    with pytest.raises(ImportError) as exc_info:
        make_stt_engine(cfg)
    # The error must surface the install hint so the operator knows
    # exactly how to fix it.
    assert "nemo_toolkit" in str(exc_info.value).lower()


def test_factory_explicit_parakeet_when_nemo_available(monkeypatch):
    """``engine: parakeet`` (explicit) with NeMo present -> Parakeet."""
    from ultron.transcription import make_stt_engine

    parakeet_cls, whisper_cls = _patch_engines(monkeypatch, nemo_available=True)
    cfg = _stub_stt_cfg(engine="parakeet")
    eng = make_stt_engine(cfg)
    parakeet_cls.assert_called_once()
    whisper_cls.assert_not_called()
    assert eng is parakeet_cls.return_value


def test_factory_explicit_whisper(monkeypatch):
    """``engine: whisper`` always returns Whisper, even when NeMo
    is installed (the swap-back path the user requested)."""
    from ultron.transcription import make_stt_engine

    parakeet_cls, whisper_cls = _patch_engines(monkeypatch, nemo_available=True)
    cfg = _stub_stt_cfg(engine="whisper")
    eng = make_stt_engine(cfg)
    parakeet_cls.assert_not_called()
    whisper_cls.assert_called_once()
    assert eng is whisper_cls.return_value


def test_factory_passes_parakeet_config(monkeypatch):
    """Custom ``parakeet_model`` / ``parakeet_device`` thread through
    to the ParakeetEngine constructor."""
    from ultron.transcription import make_stt_engine

    parakeet_cls, _ = _patch_engines(monkeypatch, nemo_available=True)
    cfg = _stub_stt_cfg(
        engine="parakeet",
        parakeet_model="nvidia/parakeet-tdt-0.6b-v2",
        parakeet_device="cpu",
    )
    make_stt_engine(cfg)
    args, kwargs = parakeet_cls.call_args
    assert kwargs.get("model_name") == "nvidia/parakeet-tdt-0.6b-v2"
    assert kwargs.get("device") == "cpu"


# ---------------------------------------------------------------------------
# ParakeetEngine constructor behaviour (no real model load)
# ---------------------------------------------------------------------------


def test_parakeet_engine_raises_without_nemo(monkeypatch):
    """Direct ParakeetEngine() construction without NeMo raises
    ImportError with an install hint -- so operators see the
    problem upfront rather than mid-turn."""
    from ultron.transcription import parakeet_engine as parakeet_module

    monkeypatch.setattr(parakeet_module, "is_nemo_available", lambda: False)
    # Make sure the spawn helper doesn't accidentally run if the
    # is_nemo_available check is bypassed in future refactors.
    monkeypatch.setattr(
        parakeet_module, "_spawn_server_if_needed",
        lambda cfg: "http://127.0.0.1:8771",
    )
    with pytest.raises(ImportError) as exc_info:
        parakeet_module.ParakeetEngine()
    assert "nemo_toolkit" in str(exc_info.value).lower()


def test_is_nemo_available_returns_bool():
    """``is_nemo_available`` must be a boolean -- safe to use in
    config-time / startup-time guards."""
    from ultron.transcription import is_nemo_available
    assert isinstance(is_nemo_available(), bool)
