"""Tests for the dual-engine STT registry + factory.

Covers the DualSTTRegistry pointer-swap behavior + the
make_dual_stt_engines factory's resolution and fallback semantics
when a gaming engine fails to load.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ultron.transcription import (
    DualSTTRegistry,
    MoonshineEngine,
    ParakeetEngine,
    WhisperEngine,
    make_dual_stt_engines,
)
import ultron.transcription as factory_module


# ---------------------------------------------------------------------------
# DualSTTRegistry contract
# ---------------------------------------------------------------------------


def _stub_engine(name: str = "stub"):
    """Minimal engine stand-in with a name attribute."""
    eng = MagicMock(name=f"engine_{name}")
    return eng


def test_registry_default_active_is_primary():
    primary = _stub_engine("primary")
    reg = DualSTTRegistry(primary=primary, primary_name="parakeet")
    assert reg.active is primary
    assert reg.active_name == "parakeet"


def test_registry_without_gaming_swap_is_noop():
    primary = _stub_engine("primary")
    reg = DualSTTRegistry(primary=primary, primary_name="parakeet")
    reg.swap_to("moonshine")
    # No gaming engine configured -- stays on primary.
    assert reg.active is primary
    assert reg.active_name == "parakeet"


def test_registry_swap_to_gaming_engine():
    primary = _stub_engine("primary")
    gaming = _stub_engine("gaming")
    reg = DualSTTRegistry(
        primary=primary, primary_name="parakeet",
        gaming=gaming, gaming_name="moonshine",
    )

    reg.swap_to("moonshine")
    assert reg.active is gaming
    assert reg.active_name == "moonshine"

    reg.swap_to("parakeet")
    assert reg.active is primary
    assert reg.active_name == "parakeet"


def test_registry_swap_to_unknown_logs_and_keeps_active():
    primary = _stub_engine("primary")
    gaming = _stub_engine("gaming")
    reg = DualSTTRegistry(
        primary=primary, primary_name="parakeet",
        gaming=gaming, gaming_name="moonshine",
    )
    reg.swap_to("moonshine")
    # Switch to a name neither engine knows.
    reg.swap_to("imaginary-engine")
    # Active stays on the previous valid engine.
    assert reg.active is gaming
    assert reg.active_name == "moonshine"


def test_registry_has_gaming_reflects_construction():
    primary = _stub_engine("primary")
    reg_no_gaming = DualSTTRegistry(primary=primary, primary_name="whisper")
    assert reg_no_gaming.has_gaming() is False

    reg_with = DualSTTRegistry(
        primary=primary, primary_name="whisper",
        gaming=_stub_engine("g"), gaming_name="moonshine",
    )
    assert reg_with.has_gaming() is True


def test_registry_swap_idempotent():
    """Swapping to the same name multiple times stays on that engine
    without churning the pointer."""
    primary = _stub_engine("primary")
    gaming = _stub_engine("gaming")
    reg = DualSTTRegistry(
        primary=primary, primary_name="parakeet",
        gaming=gaming, gaming_name="moonshine",
    )
    reg.swap_to("moonshine")
    reg.swap_to("moonshine")
    reg.swap_to("moonshine")
    assert reg.active is gaming
    assert reg.active_name == "moonshine"


# ---------------------------------------------------------------------------
# make_dual_stt_engines factory
# ---------------------------------------------------------------------------


class _FakeSTTConfig:
    """Minimal stand-in for STTConfig with only the fields the factory
    inspects."""

    def __init__(self, engine: str = "whisper", gaming_engine: str = ""):
        self.engine = engine
        self.gaming_engine = gaming_engine
        self.moonshine_model = None
        self.moonshine_device = None
        self.moonshine_precision = None
        self.parakeet_model = None
        self.parakeet_device = None
        self.model = "base.en"
        self.device = "cuda"
        self.compute_type = "float16"
        self.beam_size = 1
        self.temperature = 0.0
        self.condition_on_previous_text = False
        self.vad_filter = False


@pytest.fixture
def stub_engines(monkeypatch):
    """Mock the per-engine constructors so make_dual_stt_engines builds
    instances without loading real models."""
    whisper_instance = MagicMock(spec=WhisperEngine)
    parakeet_instance = MagicMock(spec=ParakeetEngine)
    moonshine_instance = MagicMock(spec=MoonshineEngine)

    monkeypatch.setattr(
        factory_module, "WhisperEngine",
        lambda *a, **kw: whisper_instance,
    )
    monkeypatch.setattr(
        factory_module, "ParakeetEngine",
        lambda *a, **kw: parakeet_instance,
    )
    monkeypatch.setattr(
        factory_module, "MoonshineEngine",
        lambda *a, **kw: moonshine_instance,
    )
    monkeypatch.setattr(
        factory_module, "is_nemo_available", lambda: True,
    )
    monkeypatch.setattr(
        factory_module, "is_moonshine_available", lambda: True,
    )

    # Stub _resolved_engine_name so MagicMock instances map to engine names.
    def _name(engine):
        if engine is whisper_instance:
            return "whisper"
        if engine is parakeet_instance:
            return "parakeet"
        if engine is moonshine_instance:
            return "moonshine"
        return "unknown"

    monkeypatch.setattr(factory_module, "_resolved_engine_name", _name)
    return whisper_instance, parakeet_instance, moonshine_instance


def test_make_dual_no_gaming_engine_returns_solo_registry(stub_engines):
    whisper, _p, _m = stub_engines
    cfg = _FakeSTTConfig(engine="whisper", gaming_engine="")
    reg = make_dual_stt_engines(cfg)
    assert reg.primary is whisper
    assert reg.has_gaming() is False


def test_make_dual_skips_when_gaming_same_as_primary(stub_engines):
    """Configuring gaming_engine: parakeet while engine: parakeet should
    not load two Parakeet instances (waste of VRAM)."""
    _w, parakeet, _m = stub_engines
    cfg = _FakeSTTConfig(engine="parakeet", gaming_engine="parakeet")
    reg = make_dual_stt_engines(cfg)
    assert reg.primary is parakeet
    assert reg.has_gaming() is False


def test_make_dual_loads_both_parakeet_and_moonshine(stub_engines):
    _w, parakeet, moonshine = stub_engines
    cfg = _FakeSTTConfig(engine="parakeet", gaming_engine="moonshine")
    reg = make_dual_stt_engines(cfg)
    assert reg.primary is parakeet
    assert reg.gaming is moonshine
    assert reg.primary_name == "parakeet"
    assert reg.gaming_name == "moonshine"


def test_make_dual_gaming_load_failure_returns_primary_only(stub_engines):
    """When the gaming-engine constructor raises, the factory must
    still return a working registry containing the primary -- the
    gaming swap just doesn't happen."""
    _w, parakeet, _m = stub_engines

    def _raise(*a, **kw):
        raise ImportError("simulated gaming engine missing")

    # MoonshineEngine raises during construction.
    factory_module.MoonshineEngine = _raise

    cfg = _FakeSTTConfig(engine="parakeet", gaming_engine="moonshine")
    reg = make_dual_stt_engines(cfg)
    assert reg.primary is parakeet
    assert reg.has_gaming() is False


def test_make_dual_uses_whisper_when_engine_says_whisper(stub_engines):
    whisper, _p, moonshine = stub_engines
    cfg = _FakeSTTConfig(engine="whisper", gaming_engine="moonshine")
    reg = make_dual_stt_engines(cfg)
    assert reg.primary is whisper
    assert reg.gaming is moonshine
    assert reg.primary_name == "whisper"
    assert reg.gaming_name == "moonshine"
