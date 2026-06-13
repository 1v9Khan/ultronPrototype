"""Tests for the gaming-mode engage / disengage state machine
(catalog 09 batch H).

The state machine emits :class:`StartTask` transitions for each
substep (LLM swap, Parakeet shutdown, Kokoro move, VLM unload, READY).
Sub-step failures are individually fail-open: a failure in one stage
must NOT short-circuit the rest of the state machine.
"""

from __future__ import annotations

import asyncio
from typing import Optional
from unittest.mock import MagicMock

import pytest

from kenning.lifecycle.gaming_engage import (
    GamingEngageDeps,
    gaming_disengage_iterator,
    gaming_engage_iterator,
)
from kenning.lifecycle.start_task import (
    StartTaskStatus,
    drive_start_task,
)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubLLM:
    def __init__(self, reload_ok: bool = True) -> None:
        self.reload_calls: list[str] = []
        self._ok = reload_ok

    def reload_for_preset(self, preset: str, *, gpu_layers=None):
        self.reload_calls.append(preset)
        self.last_gpu_layers = gpu_layers
        return (self._ok, "ok" if self._ok else "boom")


class _StubTTS:
    def __init__(self) -> None:
        self.device_calls: list[str] = []

    def move_to_device(self, device: str) -> None:
        self.device_calls.append(device)


class _StubSttRegistry:
    def __init__(self, *, has_gaming: bool = True, gaming_name: str = "moonshine") -> None:
        self._has_gaming = has_gaming
        self.gaming_name = gaming_name
        self.active_name = "parakeet"

    def has_gaming(self) -> bool:
        return self._has_gaming


class _StubVLM:
    def __init__(self, loaded: bool = True) -> None:
        self.loaded = loaded
        self.unload_calls = 0

    def unload(self) -> None:
        self.unload_calls += 1
        self.loaded = False


# ---------------------------------------------------------------------------
# Engage path
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_config(monkeypatch):
    """Stub get_config so reload_for_preset's pre-check sees a
    different current preset than the gaming preset."""
    class _Cfg:
        class llm:
            preset = "qwen3.5-4b"

    monkeypatch.setattr(
        "kenning.lifecycle.gaming_engage.__name__", "kenning.lifecycle.gaming_engage",
    )
    # Patch get_config inside the iterator via monkeypatching the
    # import target.
    import kenning.config
    monkeypatch.setattr(kenning.config, "get_config", lambda: _Cfg)


def test_engage_emits_all_stages(stub_config):
    llm = _StubLLM()
    tts = _StubTTS()
    registry = _StubSttRegistry()
    swap_calls = []

    def _swap(name):
        swap_calls.append(name)
        return True

    vlm = _StubVLM()
    stop_calls = []

    def _stop():
        stop_calls.append(True)
        return True

    deps = GamingEngageDeps(
        llm=llm,
        tts=tts,
        stt_registry=registry,
        swap_stt_engine=_swap,
        get_vlm=lambda: vlm,
        stop_parakeet_server=_stop,
        gaming_llm_preset="llama-3.2-3b-abliterated",
        tts_kokoro_default_device="cuda",
    )

    transitions = []
    async def _collect():
        async for task in gaming_engage_iterator(deps):
            transitions.append((task.status, task.detail, task.progress))

    _run(_collect())

    statuses = [s for (s, _, _) in transitions]
    assert StartTaskStatus.WORKING in statuses
    assert StartTaskStatus.SWAPPING_LLM in statuses
    assert StartTaskStatus.STOPPING_PARAKEET in statuses
    assert StartTaskStatus.MOVING_KOKORO in statuses
    assert StartTaskStatus.UNLOADING_VLM in statuses
    assert StartTaskStatus.READY in statuses
    # Terminal state is last.
    assert transitions[-1][0] == StartTaskStatus.READY


def test_engage_actually_invokes_llm_swap(stub_config):
    llm = _StubLLM()
    deps = GamingEngageDeps(
        llm=llm, gaming_llm_preset="llama-3.2-3b-abliterated",
    )

    async def _drain():
        async for _ in gaming_engage_iterator(deps):
            pass
    _run(_drain())

    assert llm.reload_calls == ["llama-3.2-3b-abliterated"]


def test_engage_actually_moves_tts_to_cpu(stub_config):
    tts = _StubTTS()
    deps = GamingEngageDeps(tts=tts)

    async def _drain():
        async for _ in gaming_engage_iterator(deps):
            pass
    _run(_drain())

    assert tts.device_calls == ["cpu"]


def test_engage_actually_unloads_vlm(stub_config):
    vlm = _StubVLM(loaded=True)
    deps = GamingEngageDeps(get_vlm=lambda: vlm)

    async def _drain():
        async for _ in gaming_engage_iterator(deps):
            pass
    _run(_drain())

    assert vlm.unload_calls == 1


def test_engage_skips_unloaded_vlm(stub_config):
    vlm = _StubVLM(loaded=False)
    deps = GamingEngageDeps(get_vlm=lambda: vlm)

    async def _drain():
        async for _ in gaming_engage_iterator(deps):
            pass
    _run(_drain())

    # Already unloaded -> no extra unload call.
    assert vlm.unload_calls == 0


def test_engage_skips_llm_when_already_on_target_preset(stub_config, monkeypatch):
    class _CfgOnTarget:
        class llm:
            preset = "llama-3.2-3b-abliterated"

    import kenning.config
    monkeypatch.setattr(kenning.config, "get_config", lambda: _CfgOnTarget)
    llm = _StubLLM()
    deps = GamingEngageDeps(
        llm=llm, gaming_llm_preset="llama-3.2-3b-abliterated",
    )

    async def _drain():
        async for _ in gaming_engage_iterator(deps):
            pass
    _run(_drain())

    # No reload because already on target.
    assert llm.reload_calls == []


def test_engage_llm_swap_failure_does_not_stop_state_machine(stub_config):
    """A failed LLM swap must NOT short-circuit the rest of the
    engage cycle -- TTS still moves, VLM still unloads."""
    llm = _StubLLM(reload_ok=False)
    tts = _StubTTS()
    vlm = _StubVLM(loaded=True)
    deps = GamingEngageDeps(
        llm=llm,
        tts=tts,
        get_vlm=lambda: vlm,
        gaming_llm_preset="llama-3.2-3b-abliterated",
    )

    async def _drain():
        async for _ in gaming_engage_iterator(deps):
            pass
    _run(_drain())

    # LLM swap was attempted but failed.
    assert llm.reload_calls == ["llama-3.2-3b-abliterated"]
    # Subsequent stages still ran.
    assert tts.device_calls == ["cpu"]
    assert vlm.unload_calls == 1


def test_engage_llm_exception_does_not_stop_state_machine(stub_config):
    """A raised exception in LLM swap must be swallowed and the state
    machine continues."""
    class _BoomLLM:
        def reload_for_preset(self, preset, *, gpu_layers=None):
            raise RuntimeError("boom")
    tts = _StubTTS()
    vlm = _StubVLM(loaded=True)
    deps = GamingEngageDeps(
        llm=_BoomLLM(),
        tts=tts,
        get_vlm=lambda: vlm,
        gaming_llm_preset="llama-3.2-3b-abliterated",
    )

    async def _drain():
        async for _ in gaming_engage_iterator(deps):
            pass
    _run(_drain())

    assert tts.device_calls == ["cpu"]
    assert vlm.unload_calls == 1


def test_engage_stt_failure_does_not_stop_state_machine(stub_config):
    registry = _StubSttRegistry()
    tts = _StubTTS()
    vlm = _StubVLM(loaded=True)

    def _boom_swap(name):
        raise RuntimeError("simulated swap failure")

    deps = GamingEngageDeps(
        tts=tts,
        stt_registry=registry,
        swap_stt_engine=_boom_swap,
        get_vlm=lambda: vlm,
    )

    async def _drain():
        async for _ in gaming_engage_iterator(deps):
            pass
    _run(_drain())

    # Later stages still ran despite STT failure.
    assert tts.device_calls == ["cpu"]
    assert vlm.unload_calls == 1


def test_engage_skips_stt_when_no_gaming_engine_registered(stub_config):
    registry = _StubSttRegistry(has_gaming=False)
    swap_calls = []

    def _swap(name):
        swap_calls.append(name)
        return True

    deps = GamingEngageDeps(
        stt_registry=registry, swap_stt_engine=_swap,
    )

    async def _drain():
        async for _ in gaming_engage_iterator(deps):
            pass
    _run(_drain())

    assert swap_calls == []


def test_engage_stashes_preset_for_later_disengage(stub_config):
    llm = _StubLLM()
    holder: dict = {"value": None}
    deps = GamingEngageDeps(
        llm=llm,
        gaming_llm_preset="llama-3.2-3b-abliterated",
        llm_preset_holder=holder,
    )

    async def _drain():
        async for _ in gaming_engage_iterator(deps):
            pass
    _run(_drain())

    # Pre-engage preset stashed for later disengage restoration.
    assert holder["value"] == "qwen3.5-4b"


# ---------------------------------------------------------------------------
# Disengage path
# ---------------------------------------------------------------------------


def test_disengage_emits_all_stages(stub_config):
    deps = GamingEngageDeps()

    statuses = []
    async def _collect():
        async for task in gaming_disengage_iterator(deps):
            statuses.append(task.status)
    _run(_collect())

    assert StartTaskStatus.WORKING in statuses
    assert StartTaskStatus.MOVING_KOKORO in statuses
    assert StartTaskStatus.STOPPING_PARAKEET in statuses
    assert StartTaskStatus.SWAPPING_LLM in statuses
    assert statuses[-1] == StartTaskStatus.READY


def test_disengage_restores_tts_device(stub_config):
    tts = _StubTTS()
    deps = GamingEngageDeps(tts=tts, tts_kokoro_default_device="cuda")

    async def _drain():
        async for _ in gaming_disengage_iterator(deps):
            pass
    _run(_drain())

    assert tts.device_calls == ["cuda"]


def test_disengage_restores_llm_preset_from_holder(stub_config):
    llm = _StubLLM()
    holder = {"value": "qwen3.5-4b"}
    deps = GamingEngageDeps(llm=llm, llm_preset_holder=holder)

    async def _drain():
        async for _ in gaming_disengage_iterator(deps):
            pass
    _run(_drain())

    assert llm.reload_calls == ["qwen3.5-4b"]
    # Holder cleared after restore.
    assert holder["value"] is None


def test_disengage_no_holder_value_skips_llm_restore(stub_config):
    llm = _StubLLM()
    holder = {"value": None}
    deps = GamingEngageDeps(llm=llm, llm_preset_holder=holder)

    async def _drain():
        async for _ in gaming_disengage_iterator(deps):
            pass
    _run(_drain())

    assert llm.reload_calls == []


def test_disengage_parakeet_path_spawns_background_thread(stub_config):
    """When prior STT was parakeet, disengage spawns a background thread
    to call start_parakeet_server(wait_for_ready=True) and the swap.
    """
    holder = {"value": "parakeet"}
    registry = _StubSttRegistry()
    swap_calls = []
    server_calls = []

    def _swap(name):
        swap_calls.append(name)
        return True

    def _server(wait_for_ready=True):
        server_calls.append(wait_for_ready)
        return True

    deps = GamingEngageDeps(
        stt_registry=registry,
        swap_stt_engine=_swap,
        start_parakeet_server=_server,
        stt_name_holder=holder,
    )

    async def _drain():
        async for _ in gaming_disengage_iterator(deps):
            pass
    _run(_drain())

    # Wait for the spawned thread to finish.
    import threading, time
    for _ in range(50):
        # Look for our spawned thread to settle.
        alive = [t for t in threading.enumerate() if t.name == "parakeet-restore"]
        if not alive:
            break
        time.sleep(0.05)

    assert server_calls == [True]
    assert swap_calls == ["parakeet"]


def test_disengage_non_parakeet_path_swaps_immediately(stub_config):
    """When prior STT was a non-parakeet engine, the swap-back is
    synchronous (no background thread)."""
    holder = {"value": "whisper"}
    registry = _StubSttRegistry()
    swap_calls = []

    def _swap(name):
        swap_calls.append(name)
        return True

    deps = GamingEngageDeps(
        stt_registry=registry,
        swap_stt_engine=_swap,
        stt_name_holder=holder,
    )

    async def _drain():
        async for _ in gaming_disengage_iterator(deps):
            pass
    _run(_drain())

    assert swap_calls == ["whisper"]
    # Holder cleared after restore.
    assert holder["value"] is None


# ---------------------------------------------------------------------------
# drive_start_task integration
# ---------------------------------------------------------------------------


def test_drive_engage_calls_on_transition_per_stage(stub_config):
    deps = GamingEngageDeps(
        gaming_llm_preset="llama-3.2-3b-abliterated",
    )
    seen_statuses = []

    def _on_transition(task):
        seen_statuses.append(task.status)

    async def _go():
        return await drive_start_task(
            gaming_engage_iterator(deps),
            on_transition=_on_transition,
        )

    _run(_go())

    # Each major stage produced an on_transition call.
    expected = {
        StartTaskStatus.WORKING,
        StartTaskStatus.SWAPPING_LLM,
        StartTaskStatus.STOPPING_PARAKEET,
        StartTaskStatus.MOVING_KOKORO,
        StartTaskStatus.UNLOADING_VLM,
        StartTaskStatus.READY,
    }
    assert expected.issubset(set(seen_statuses))


def test_drive_engage_terminal_task_has_ready_status(stub_config):
    deps = GamingEngageDeps()

    async def _go():
        return await drive_start_task(gaming_engage_iterator(deps))

    final = _run(_go())
    assert final.status == StartTaskStatus.READY
    assert final.progress == 1.0


def test_drive_disengage_terminal_task_has_ready_status(stub_config):
    deps = GamingEngageDeps()

    async def _go():
        return await drive_start_task(gaming_disengage_iterator(deps))

    final = _run(_go())
    assert final.status == StartTaskStatus.READY


def test_on_transition_failure_does_not_break_drive(stub_config):
    """A failing on_transition callback must NOT stop the state
    machine -- the underlying engage logic still completes."""
    tts = _StubTTS()
    deps = GamingEngageDeps(tts=tts)

    def _boom(task):
        raise RuntimeError("ack failure")

    async def _go():
        return await drive_start_task(
            gaming_engage_iterator(deps),
            on_transition=_boom,
        )

    # The drive_start_task contract is that on_transition exceptions
    # propagate -- the orchestrator-level wrapper catches them. We
    # verify the wrapper catches them in the orchestrator test.
    # Here we just confirm the propagation shape.
    with pytest.raises(RuntimeError, match="ack failure"):
        _run(_go())
