"""Tests for the idle-time VRAM reclaim (2026-06-11 memory hygiene)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from ultron.pipeline.orchestrator import Orchestrator


def _bare():
    o = Orchestrator.__new__(Orchestrator)
    return o


def _fake_torch(*, available=True, reserved=0, allocated=0, calls=None):
    mod = types.ModuleType("torch")
    cuda = SimpleNamespace(
        is_available=lambda: available,
        memory_reserved=lambda: reserved,
        memory_allocated=lambda: allocated,
        empty_cache=lambda: (calls.append("empty") if calls is not None else None),
    )
    mod.cuda = cuda
    return mod


def _patch_cfg(monkeypatch, *, enabled=True, min_slack_mb=192.0):
    import ultron.config as config_mod

    monkeypatch.setattr(
        config_mod, "get_config",
        lambda: SimpleNamespace(llm=SimpleNamespace(
            idle_vram_reclaim=SimpleNamespace(
                enabled=enabled, min_slack_mb=min_slack_mb,
            ),
        )),
    )


def test_reclaims_when_slack_exceeds_threshold(monkeypatch) -> None:
    calls: list[str] = []
    # 700MB reserved, 400MB allocated -> 300MB slack > 192MB threshold.
    monkeypatch.setitem(
        sys.modules, "torch",
        _fake_torch(reserved=700_000_000, allocated=400_000_000, calls=calls),
    )
    _patch_cfg(monkeypatch)
    _bare()._reclaim_idle_vram()
    assert calls == ["empty"]


def test_noop_when_slack_below_threshold(monkeypatch) -> None:
    calls: list[str] = []
    # 100MB slack < 192MB threshold -> no empty_cache.
    monkeypatch.setitem(
        sys.modules, "torch",
        _fake_torch(reserved=500_000_000, allocated=400_000_000, calls=calls),
    )
    _patch_cfg(monkeypatch)
    _bare()._reclaim_idle_vram()
    assert calls == []


def test_noop_when_disabled(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setitem(
        sys.modules, "torch",
        _fake_torch(reserved=900_000_000, allocated=100_000_000, calls=calls),
    )
    _patch_cfg(monkeypatch, enabled=False)
    _bare()._reclaim_idle_vram()
    assert calls == []


def test_noop_when_cuda_unavailable(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setitem(
        sys.modules, "torch",
        _fake_torch(available=False, reserved=900_000_000, allocated=0,
                    calls=calls),
    )
    _patch_cfg(monkeypatch)
    _bare()._reclaim_idle_vram()
    assert calls == []


def test_fail_open_on_torch_missing(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    _patch_cfg(monkeypatch)
    # Must not raise even though `import torch` yields None.
    _bare()._reclaim_idle_vram()


def test_config_defaults() -> None:
    from ultron.config import IdleVramReclaimConfig, LLMConfig

    cfg = IdleVramReclaimConfig()
    assert cfg.enabled is True
    assert cfg.min_slack_mb == 192.0
    assert LLMConfig().idle_vram_reclaim.enabled is True
