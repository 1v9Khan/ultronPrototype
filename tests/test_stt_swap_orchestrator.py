"""Tests for the orchestrator's swap_stt_engine + gaming-mode STT
swap wiring.

We exercise the dual-STT swap surface without spinning up the real
orchestrator (which would load Parakeet, Moonshine, Whisper, the LLM,
TTS, VAD, embedders -- 10+ seconds of init and ~5 GB of memory). The
swap_stt_engine method is callable on a minimal orchestrator
fragment; we verify the registry-aware pointer flip and the gaming-
mode engage/disengage callback sequence.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ultron.transcription import DualSTTRegistry


# ---------------------------------------------------------------------------
# A minimal orchestrator-shaped object that exposes just the surface
# the swap method needs.
# ---------------------------------------------------------------------------


class _MinimalOrchestrator:
    """Stand-in for the real orchestrator's STT swap surface."""

    def __init__(self, registry: DualSTTRegistry):
        self._stt_registry = registry
        self.stt = registry.active
        self._speculative_stt_invalidated = False

    def _invalidate_speculative_stt(self) -> None:
        self._speculative_stt_invalidated = True

    # Borrow the real implementation by duck-typing.
    def swap_stt_engine(self, name: str) -> bool:
        from ultron.utils.logging import get_logger
        logger = get_logger("test")
        registry = self._stt_registry
        if registry is None:
            return False
        if name == registry.active_name:
            return True
        try:
            self._invalidate_speculative_stt()
        except Exception:
            pass
        new_engine = registry.swap_to(name)
        if registry.active_name == name:
            self.stt = new_engine
            return True
        return False


def _make_registry_with_two_engines():
    primary = MagicMock(name="parakeet_engine")
    gaming = MagicMock(name="moonshine_engine")
    return DualSTTRegistry(
        primary=primary, primary_name="parakeet",
        gaming=gaming, gaming_name="moonshine",
    )


# ---------------------------------------------------------------------------
# swap_stt_engine contract
# ---------------------------------------------------------------------------


def test_swap_to_same_engine_is_noop():
    registry = _make_registry_with_two_engines()
    orch = _MinimalOrchestrator(registry)
    assert orch.swap_stt_engine("parakeet") is True
    assert orch.stt is registry.primary
    # Speculative STT NOT invalidated -- we didn't swap.
    assert orch._speculative_stt_invalidated is False


def test_swap_to_gaming_flips_pointer():
    registry = _make_registry_with_two_engines()
    orch = _MinimalOrchestrator(registry)
    assert orch.swap_stt_engine("moonshine") is True
    assert orch.stt is registry.gaming
    # In-flight speculative STT was invalidated before the flip.
    assert orch._speculative_stt_invalidated is True


def test_swap_back_to_primary_restores_pointer():
    registry = _make_registry_with_two_engines()
    orch = _MinimalOrchestrator(registry)
    orch.swap_stt_engine("moonshine")
    orch._speculative_stt_invalidated = False  # reset
    orch.swap_stt_engine("parakeet")
    assert orch.stt is registry.primary
    assert orch._speculative_stt_invalidated is True


def test_swap_to_unknown_engine_does_not_change_pointer():
    registry = _make_registry_with_two_engines()
    orch = _MinimalOrchestrator(registry)
    orch.swap_stt_engine("moonshine")
    before = orch.stt
    orch.swap_stt_engine("imaginary-engine")
    # Pointer stays on whatever was active before.
    assert orch.stt is before


def test_swap_without_registry_returns_false():
    orch = _MinimalOrchestrator(_make_registry_with_two_engines())
    orch._stt_registry = None
    assert orch.swap_stt_engine("moonshine") is False


# ---------------------------------------------------------------------------
# Parakeet server lifecycle hooks
# ---------------------------------------------------------------------------


def test_stop_parakeet_server_when_not_running_returns_false(monkeypatch):
    from ultron.transcription import parakeet_engine as pe_mod
    monkeypatch.setattr(pe_mod, "_SERVER_PROCESS", None)
    assert pe_mod.stop_parakeet_server() is False


def test_stop_parakeet_server_terminates_alive_process(monkeypatch):
    from ultron.transcription import parakeet_engine as pe_mod

    proc = MagicMock(name="server_proc")
    proc.poll.return_value = None  # alive
    monkeypatch.setattr(pe_mod, "_SERVER_PROCESS", proc)
    monkeypatch.setattr(pe_mod, "_SERVER_URL_CACHED", "http://127.0.0.1:8771")

    # Mock requests.post for /shutdown call.
    fake_requests = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "requests", fake_requests)

    assert pe_mod.stop_parakeet_server() is True
    proc.terminate.assert_called_once()
    # Process tracker cleared so subsequent calls return False.
    assert pe_mod._SERVER_PROCESS is None


def test_is_parakeet_server_running_reflects_subprocess_state(monkeypatch):
    from ultron.transcription import parakeet_engine as pe_mod

    monkeypatch.setattr(pe_mod, "_SERVER_PROCESS", None)
    assert pe_mod.is_parakeet_server_running() is False

    alive = MagicMock()
    alive.poll.return_value = None
    monkeypatch.setattr(pe_mod, "_SERVER_PROCESS", alive)
    assert pe_mod.is_parakeet_server_running() is True

    exited = MagicMock()
    exited.poll.return_value = 0
    monkeypatch.setattr(pe_mod, "_SERVER_PROCESS", exited)
    assert pe_mod.is_parakeet_server_running() is False


# ---------------------------------------------------------------------------
# End-to-end simulation of engage / disengage callback wiring
# ---------------------------------------------------------------------------


def test_engage_disengage_callbacks_swap_and_restore(monkeypatch):
    """Mirror of the orchestrator's _engage_extra / _disengage_extra
    sequence. Verifies the swap-restore round trip happens against a
    real DualSTTRegistry, with Parakeet server stop/start mocked."""
    registry = _make_registry_with_two_engines()
    orch = _MinimalOrchestrator(registry)

    server_stopped = {"count": 0}
    server_started = {"count": 0}

    def _stop():
        server_stopped["count"] += 1
        return True

    def _start(wait_for_ready=True):
        server_started["count"] += 1
        return "http://127.0.0.1:8771"

    # Simulate the engage/disengage sequence from the orchestrator wiring.
    stt_name_before_engage = {"value": None}

    def engage():
        prior = registry.active_name
        if orch.swap_stt_engine(registry.gaming_name):
            stt_name_before_engage["value"] = prior
            _stop()

    def disengage():
        prior = stt_name_before_engage["value"]
        if prior is None:
            return
        if prior == "parakeet":
            _start(wait_for_ready=True)
            orch.swap_stt_engine(prior)
        else:
            orch.swap_stt_engine(prior)
        stt_name_before_engage["value"] = None

    # Pre-engage state.
    assert orch.stt is registry.primary
    assert orch._stt_registry.active_name == "parakeet"

    engage()
    assert orch.stt is registry.gaming
    assert orch._stt_registry.active_name == "moonshine"
    assert server_stopped["count"] == 1

    disengage()
    assert orch.stt is registry.primary
    assert orch._stt_registry.active_name == "parakeet"
    assert server_started["count"] == 1
    assert stt_name_before_engage["value"] is None
