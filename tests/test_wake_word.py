"""Unit tests for :class:`WakeWordDetector` (A4 ``fired_recently`` accessor).

The detector wraps openWakeWord's ``Model`` which loads an ONNX file at
construction time. These tests avoid the model load by patching the
underlying ``_load_model`` helper -- the methods under test only touch
``_last_trigger_ts`` and ``time.monotonic()``.
"""

from __future__ import annotations

import time

import pytest

from ultron.audio.wake_word import WakeWordDetector


@pytest.fixture
def detector(monkeypatch):
    monkeypatch.setattr(
        WakeWordDetector, "_load_model",
        lambda self, path, fallback: object(),
    )
    return WakeWordDetector(model_path=None, fallback_name="hey_jarvis")


def test_fired_recently_false_before_first_trigger(detector):
    assert detector.fired_recently(window_s=10.0) is False


def test_fired_recently_true_just_after_trigger(detector):
    detector._last_trigger_ts = time.monotonic()  # noqa: SLF001
    assert detector.fired_recently(window_s=0.5) is True


def test_fired_recently_false_after_window_elapsed(detector):
    detector._last_trigger_ts = time.monotonic() - 5.0  # noqa: SLF001
    assert detector.fired_recently(window_s=0.5) is False


def test_fired_recently_idempotent(detector):
    """Calling fired_recently must not consume / clear the trigger."""
    detector._last_trigger_ts = time.monotonic()  # noqa: SLF001
    assert detector.fired_recently(window_s=10.0) is True
    assert detector.fired_recently(window_s=10.0) is True
    assert detector.fired_recently(window_s=10.0) is True


def test_fired_recently_window_s_is_inclusive_lower_bound(detector):
    """Border case: setting window_s=0 always returns False (trigger
    happened in the past, not 'now')."""
    detector._last_trigger_ts = time.monotonic() - 0.001  # noqa: SLF001
    assert detector.fired_recently(window_s=0.0) is False


def test_fired_recently_zeroed_state_returns_false(detector):
    """A reset detector (initial _last_trigger_ts == 0) must NOT report
    a barge-in -- the timestamp would otherwise be 0 vs now and the
    delta would be huge but the contract is 'never fired = no barge-in'."""
    detector._last_trigger_ts = 0.0  # noqa: SLF001
    assert detector.fired_recently(window_s=1_000_000.0) is False


def test_fired_recently_negative_window_returns_false(detector):
    detector._last_trigger_ts = time.monotonic()  # noqa: SLF001
    # Caller-error case; the implementation clamps via float() comparison.
    assert detector.fired_recently(window_s=-1.0) is False
