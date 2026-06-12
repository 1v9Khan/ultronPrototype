"""Tests for the kenning/ultron wake-word selection, custom-ONNX fallback,
and runtime hot-swap (:meth:`WakeWordDetector.reload_for_word`).

The detector loads real ONNX files via openWakeWord's ``Model`` at call
time. These tests stub ``openwakeword.model.Model`` with a fake that just
records which model(s) it was handed, and use empty temp files for the
path-existence checks (kenning.onnx / ultron.onnx live side by side).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from kenning.audio.wake_word import WakeWordDetector


class _FakeModel:
    """Records the ``wakeword_models`` it was constructed with."""

    last_loaded: list[str] = []

    def __init__(self, wakeword_models, inference_framework="onnx"):
        self.loaded = list(wakeword_models)
        _FakeModel.last_loaded = list(wakeword_models)

    def predict(self, pcm):
        return {"x": 0.0}

    def reset(self):
        pass


@pytest.fixture
def fake_oww(monkeypatch):
    """Install a fake ``openwakeword.model`` module so ``from
    openwakeword.model import Model`` inside the detector resolves to
    :class:`_FakeModel`."""
    mod = types.ModuleType("openwakeword.model")
    mod.Model = _FakeModel
    monkeypatch.setitem(sys.modules, "openwakeword.model", mod)
    _FakeModel.last_loaded = []
    return _FakeModel


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models" / "openwakeword"
    d.mkdir(parents=True)
    (d / "kenning.onnx").write_bytes(b"\x00")
    (d / "ultron.onnx").write_bytes(b"\x00")
    return d


def _make(models_dir: Path, *, name="kenning", model_file="kenning.onnx",
          fallback="ultron") -> WakeWordDetector:
    return WakeWordDetector(
        model_path=models_dir / model_file,
        fallback_name=fallback,
        name=name,
    )


# --- selection + fallback ---------------------------------------------------

def test_selected_kenning_loads_kenning(fake_oww, models_dir):
    det = _make(models_dir, name="kenning", model_file="kenning.onnx")
    assert det.active_word == "kenning"
    assert det.using_fallback is False
    assert fake_oww.last_loaded == [str(models_dir / "kenning.onnx")]


def test_missing_kenning_falls_back_to_custom_ultron(fake_oww, models_dir):
    (models_dir / "kenning.onnx").unlink()  # selected model gone
    det = _make(models_dir, name="kenning", model_file="kenning.onnx",
                fallback="ultron")
    assert det.using_fallback is True
    assert det.active_word == "ultron"
    # Loaded the CUSTOM ultron.onnx -- a path, never the pretrained word.
    assert fake_oww.last_loaded == [str(models_dir / "ultron.onnx")]


def test_fallback_is_never_hey_jarvis_when_ultron_exists(fake_oww, models_dir):
    (models_dir / "kenning.onnx").unlink()
    det = _make(models_dir, fallback="ultron")
    assert "hey_jarvis" not in (det.active_word,)
    assert all("hey_jarvis" not in m for m in fake_oww.last_loaded)


def test_pretrained_last_resort_only_when_no_custom_fallback(fake_oww, models_dir):
    (models_dir / "kenning.onnx").unlink()
    (models_dir / "ultron.onnx").unlink()  # no custom fallback either
    det = _make(models_dir, name="kenning", fallback="hey_jarvis")
    assert det.using_fallback is True
    # Falls through to the pretrained built-in word NAME (not a path).
    assert fake_oww.last_loaded == ["hey_jarvis"]


def test_model_path_for_word_resolves_only_existing(fake_oww, models_dir):
    det = _make(models_dir)
    assert det._model_path_for_word("kenning") == models_dir / "kenning.onnx"
    assert det._model_path_for_word("ultron") == models_dir / "ultron.onnx"
    assert det._model_path_for_word("nope") is None
    assert det._model_path_for_word("") is None


# --- hot-swap ---------------------------------------------------------------

def test_reload_swaps_kenning_to_ultron(fake_oww, models_dir):
    det = _make(models_dir, name="kenning")
    assert det.active_word == "kenning"
    ok, msg = det.reload_for_word("ultron")
    assert ok is True and msg == "ultron"
    assert det.active_word == "ultron"
    assert det.using_fallback is False
    assert fake_oww.last_loaded == [str(models_dir / "ultron.onnx")]


def test_reload_swaps_back_to_kenning(fake_oww, models_dir):
    det = _make(models_dir, name="ultron", model_file="ultron.onnx")
    ok, _ = det.reload_for_word("kenning")
    assert ok is True
    assert det.active_word == "kenning"
    assert fake_oww.last_loaded == [str(models_dir / "kenning.onnx")]


def test_reload_missing_word_falls_back_to_ultron(fake_oww, models_dir):
    det = _make(models_dir, name="kenning", fallback="ultron")
    ok, msg = det.reload_for_word("does_not_exist")
    assert ok is False
    assert det.active_word == "ultron"
    assert det.using_fallback is True
    assert "ultron" in msg


def test_reload_empty_word_is_noop(fake_oww, models_dir):
    det = _make(models_dir, name="kenning")
    ok, msg = det.reload_for_word("")
    assert ok is False
    assert det.active_word == "kenning"  # unchanged


def test_reload_resets_cooldown(fake_oww, models_dir):
    det = _make(models_dir, name="kenning")
    det._last_trigger_ts = 12345.0  # noqa: SLF001
    det.reload_for_word("ultron")
    assert det._last_trigger_ts == 0.0  # noqa: SLF001


def test_reload_case_insensitive(fake_oww, models_dir):
    det = _make(models_dir, name="kenning")
    ok, _ = det.reload_for_word("ULTRON")
    assert ok is True
    assert det.active_word == "ultron"
