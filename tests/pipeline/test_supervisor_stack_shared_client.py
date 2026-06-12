"""Tests for the supervisor stack's shared-Qdrant-client wiring.

2026-06-12 fix: local-mode Qdrant allows ONE open client per path, and
ConversationMemory holds the ``data/qdrant`` lock for the process
lifetime -- so ``_build_supervisor_stack`` must BORROW the memory's
client instead of double-opening the path (which failed on every boot
and silently degraded the supervisor to registry-only).

Hermetic: ``Orchestrator.__new__`` pattern (no models, no audio, no
Qdrant); ProjectIndex is monkeypatched at its source module, which is
where the orchestrator's function-local import resolves it from.
"""

from __future__ import annotations

from types import SimpleNamespace

from ultron.pipeline.orchestrator import Orchestrator


def _cfg() -> SimpleNamespace:
    # decide_enabled=False short-circuits the supervisor + dispatch
    # branches so only the index block runs.
    return SimpleNamespace(index_enabled=True, decide_enabled=False)


def _capture_index(monkeypatch) -> dict:
    captured: dict = {}

    class _FakeIndex:
        def __init__(self, **kw) -> None:
            captured.update(kw)

    monkeypatch.setattr(
        "ultron.coding.project_index.ProjectIndex", _FakeIndex
    )
    return captured


def test_passes_memory_client_when_memory_present(monkeypatch) -> None:
    captured = _capture_index(monkeypatch)
    o = Orchestrator.__new__(Orchestrator)
    o.memory = SimpleNamespace(_client=object())
    idx, sup, dispatch = o._build_supervisor_stack(
        _cfg(), registry=None, resolver=None,
        embedder=object(), runner=None,
    )
    assert idx is not None
    assert sup is None and dispatch is None
    assert captured["client"] is o.memory._client


def test_falls_back_to_own_client_when_memory_disabled(monkeypatch) -> None:
    captured = _capture_index(monkeypatch)
    o = Orchestrator.__new__(Orchestrator)
    o.memory = None
    idx, _, _ = o._build_supervisor_stack(
        _cfg(), registry=None, resolver=None,
        embedder=object(), runner=None,
    )
    assert idx is not None
    assert captured["client"] is None


def test_memory_client_attribute_error_falls_back(monkeypatch) -> None:
    captured = _capture_index(monkeypatch)

    class _BrokenMemory:
        @property
        def _client(self):
            raise RuntimeError("client unavailable")

    o = Orchestrator.__new__(Orchestrator)
    o.memory = _BrokenMemory()
    idx, _, _ = o._build_supervisor_stack(
        _cfg(), registry=None, resolver=None,
        embedder=object(), runner=None,
    )
    assert idx is not None
    assert captured["client"] is None


def test_index_construction_failure_stays_fail_open(monkeypatch) -> None:
    class _Boom:
        def __init__(self, **kw) -> None:
            raise RuntimeError("qdrant exploded")

    monkeypatch.setattr("ultron.coding.project_index.ProjectIndex", _Boom)
    o = Orchestrator.__new__(Orchestrator)
    o.memory = None
    idx, sup, dispatch = o._build_supervisor_stack(
        _cfg(), registry=None, resolver=None,
        embedder=object(), runner=None,
    )
    assert idx is None and sup is None and dispatch is None


def test_index_disabled_skips_construction(monkeypatch) -> None:
    captured = _capture_index(monkeypatch)
    o = Orchestrator.__new__(Orchestrator)
    o.memory = None
    cfg = SimpleNamespace(index_enabled=False, decide_enabled=False)
    idx, _, _ = o._build_supervisor_stack(
        cfg, registry=None, resolver=None, embedder=object(), runner=None,
    )
    assert idx is None
    assert captured == {}
