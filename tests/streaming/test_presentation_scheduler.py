"""Tests for ultron.streaming.presentation_scheduler."""

from __future__ import annotations

import pytest

from ultron.streaming import presentation_scheduler as ps


class _Clock:
    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


# ---------------------------------------------------------------------------
# Cadence
# ---------------------------------------------------------------------------

class TestCadence:
    def test_for_priority_returns_correct_band(self) -> None:
        cadence = ps.Cadence(immediate_ms=0, normal_ms=60, low_ms=200)
        assert cadence.for_priority(ps.PresentationPriority.IMMEDIATE) == 0
        assert cadence.for_priority(ps.PresentationPriority.NORMAL) == 60
        assert cadence.for_priority(ps.PresentationPriority.LOW) == 200


# ---------------------------------------------------------------------------
# Profile detection
# ---------------------------------------------------------------------------

class TestProfileDetection:
    def test_bluetooth_detected(self) -> None:
        assert ps.detect_audio_profile("Bluetooth Headset") is ps.AudioProfile.BLUETOOTH
        assert ps.detect_audio_profile("AirPods Pro") is ps.AudioProfile.BLUETOOTH

    def test_remote_detected(self) -> None:
        assert ps.detect_audio_profile("RDP Remote") is ps.AudioProfile.REMOTE

    def test_fallback_local(self) -> None:
        assert ps.detect_audio_profile("Speakers (Realtek)") is ps.AudioProfile.LOCAL
        assert ps.detect_audio_profile("") is ps.AudioProfile.LOCAL


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class TestScheduler:
    def _capture(self) -> tuple[list[tuple[str, ps.PresentationPriority]], callable]:
        out: list[tuple[str, ps.PresentationPriority]] = []
        return out, lambda c, p: out.append((c, p))

    def test_enqueue_and_immediate_emit(self) -> None:
        out, cb = self._capture()
        scheduler = ps.PresentationScheduler(
            on_emit=cb,
            cadence=ps.Cadence(immediate_ms=0, normal_ms=0, low_ms=0),
            audio_profile=ps.AudioProfile.LOCAL,
            clock=_Clock(),
        )
        scheduler.enqueue("hello", ps.PresentationPriority.IMMEDIATE)
        assert scheduler.maybe_emit() == 1
        assert out == [("hello", ps.PresentationPriority.IMMEDIATE)]

    def test_normal_respects_cadence(self) -> None:
        clock = _Clock()
        out, cb = self._capture()
        scheduler = ps.PresentationScheduler(
            on_emit=cb,
            cadence=ps.Cadence(immediate_ms=0, normal_ms=100, low_ms=200),
            audio_profile=ps.AudioProfile.LOCAL,
            clock=clock,
        )
        scheduler.enqueue("a", ps.PresentationPriority.NORMAL)
        # First emit fires (last_emit_at=0).
        assert scheduler.maybe_emit() == 1
        scheduler.enqueue("b", ps.PresentationPriority.NORMAL)
        # Cadence window not yet elapsed.
        assert scheduler.maybe_emit() == 0
        clock.advance(0.2)
        assert scheduler.maybe_emit() == 1

    def test_low_priority_dropped_when_configured(self) -> None:
        out, cb = self._capture()
        scheduler = ps.PresentationScheduler(
            on_emit=cb,
            cadence=ps.Cadence(immediate_ms=0, normal_ms=0, low_ms=0),
            audio_profile=ps.AudioProfile.LOCAL,
            clock=_Clock(),
        )
        scheduler.set_drop_low_priority(True)
        scheduler.enqueue("noisy reasoning", ps.PresentationPriority.LOW)
        scheduler.flush()
        assert out == []

    def test_flush_emits_everything(self) -> None:
        out, cb = self._capture()
        scheduler = ps.PresentationScheduler(
            on_emit=cb,
            cadence=ps.Cadence(immediate_ms=10000, normal_ms=10000, low_ms=10000),
            audio_profile=ps.AudioProfile.LOCAL,
            clock=_Clock(),
        )
        scheduler.enqueue("imm", ps.PresentationPriority.IMMEDIATE)
        scheduler.enqueue("norm", ps.PresentationPriority.NORMAL)
        scheduler.enqueue("low", ps.PresentationPriority.LOW)
        # maybe_emit fires immediate (cadence=0 because last_emit_at=0)... wait,
        # the cadence is 10000ms so even immediate is blocked.
        # flush() bypasses cadence entirely.
        assert scheduler.flush() == 3
        priorities = [p for _, p in out]
        assert ps.PresentationPriority.IMMEDIATE in priorities
        assert ps.PresentationPriority.NORMAL in priorities
        assert ps.PresentationPriority.LOW in priorities

    def test_pending_count(self) -> None:
        scheduler = ps.PresentationScheduler(
            on_emit=lambda c, p: None,
            cadence=ps.Cadence(normal_ms=10000),
            audio_profile=ps.AudioProfile.LOCAL,
            clock=_Clock(),
        )
        scheduler.enqueue("a", ps.PresentationPriority.NORMAL)
        scheduler.enqueue("b", ps.PresentationPriority.NORMAL)
        assert scheduler.pending_count() == 2

    def test_on_emit_exception_does_not_break(self) -> None:
        def boom(_c: str, _p: ps.PresentationPriority) -> None:
            raise RuntimeError()
        scheduler = ps.PresentationScheduler(
            on_emit=boom,
            cadence=ps.Cadence(immediate_ms=0, normal_ms=0, low_ms=0),
            clock=_Clock(),
        )
        scheduler.enqueue("a", ps.PresentationPriority.IMMEDIATE)
        # Should not raise.
        scheduler.maybe_emit()

    def test_set_cadence_replaces(self) -> None:
        scheduler = ps.PresentationScheduler(
            cadence=ps.Cadence(normal_ms=60),
            clock=_Clock(),
        )
        scheduler.set_cadence(ps.Cadence(normal_ms=1000))
        assert scheduler.cadence().normal_ms == 1000

    def test_empty_enqueue_ignored(self) -> None:
        out, cb = self._capture()
        scheduler = ps.PresentationScheduler(on_emit=cb, clock=_Clock())
        scheduler.enqueue("", ps.PresentationPriority.IMMEDIATE)
        scheduler.flush()
        assert out == []


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_profile_map_keys_complete(self) -> None:
        for profile in ps.AudioProfile:
            assert profile in ps.DEFAULT_CADENCE_BY_PROFILE
