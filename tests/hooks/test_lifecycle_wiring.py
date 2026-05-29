"""Tests for the coding-runner hooks lifecycle wiring.

CodingTaskRunner fires the TaskStart hook fan-out (cancel-capable) at
start_task and a TaskComplete fan-out (observability) on the COMPLETE event.
Both are gated by config.hooks.enabled, fail-open, and a zero-cost no-op when
no hook scripts are installed. Exercised via a __new__ stub runner.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ultron.coding.runner import CodingTaskRunner


def _runner():
    r = CodingTaskRunner.__new__(CodingTaskRunner)
    r._bound_session_id = None
    return r


def _fanout(cancelled, per_hook=()):
    return SimpleNamespace(cancelled=cancelled, per_hook_results=per_hook)


# --- TaskStart --------------------------------------------------------------


def test_task_start_hook_cancel_raises(monkeypatch):
    import ultron.hooks as H

    blocked = SimpleNamespace(
        outcome=SimpleNamespace(cancel=True, error_message="blocked by policy"),
    )
    fake = SimpleNamespace(fire=lambda kind, payload: _fanout(True, (blocked,)))
    monkeypatch.setattr(H, "get_hook_registry", lambda *a, **k: fake)

    r = _runner()
    with pytest.raises(RuntimeError, match="blocked by policy"):
        r._fire_task_start_hook(SimpleNamespace(prompt="do a thing"))


def test_task_start_hook_pass_proceeds(monkeypatch):
    import ultron.hooks as H

    monkeypatch.setattr(
        H, "get_hook_registry",
        lambda *a, **k: SimpleNamespace(fire=lambda k, p: _fanout(False)),
    )
    r = _runner()
    # Must not raise.
    r._fire_task_start_hook(SimpleNamespace(prompt="x"))


def test_task_start_no_scripts_real_registry():
    """Real registry, no hooks installed -> empty fast path -> no raise."""
    from ultron.hooks import reset_hook_registry_for_testing

    reset_hook_registry_for_testing()
    try:
        r = _runner()
        r._fire_task_start_hook(SimpleNamespace(prompt="x"))
    finally:
        reset_hook_registry_for_testing()


def test_task_start_fire_error_is_swallowed(monkeypatch):
    import ultron.hooks as H

    def _boom(*a, **k):
        raise RuntimeError("registry exploded")

    monkeypatch.setattr(H, "get_hook_registry", _boom)
    r = _runner()
    # Fail-open: a broken registry never blocks the task.
    r._fire_task_start_hook(SimpleNamespace(prompt="x"))


def test_task_start_disabled_skips_registry(monkeypatch):
    from ultron.config import get_config

    monkeypatch.setattr(get_config().hooks, "enabled", False)
    import ultron.hooks as H

    called: list[int] = []
    monkeypatch.setattr(
        H, "get_hook_registry",
        lambda *a, **k: called.append(1) or SimpleNamespace(fire=lambda k, p: _fanout(False)),
    )
    r = _runner()
    r._fire_task_start_hook(SimpleNamespace(prompt="x"))
    assert called == []  # disabled -> returns before touching the registry


# --- TaskComplete -----------------------------------------------------------


def test_complete_listener_fires_on_complete(monkeypatch):
    import ultron.hooks as H
    from ultron.coding.bridge import EventKind

    fired: list = []
    monkeypatch.setattr(
        H, "get_hook_registry",
        lambda *a, **k: SimpleNamespace(fire=lambda kind, payload: fired.append((kind, payload))),
    )
    r = _runner()
    listener = r._make_hook_lifecycle_listener(handle=None)
    assert listener is not None

    listener(SimpleNamespace(kind=EventKind.COMPLETE, exit_status=0, summary="done"))
    assert len(fired) == 1
    assert fired[0][0] == H.HookKind.TASK_COMPLETE


def test_complete_listener_ignores_non_complete(monkeypatch):
    import ultron.hooks as H
    from ultron.coding.bridge import EventKind

    fired: list = []
    monkeypatch.setattr(
        H, "get_hook_registry",
        lambda *a, **k: SimpleNamespace(fire=lambda kind, payload: fired.append(kind)),
    )
    r = _runner()
    listener = r._make_hook_lifecycle_listener(handle=None)
    listener(SimpleNamespace(kind=EventKind.TEXT, text="hi"))
    assert fired == []


def test_complete_listener_none_when_disabled(monkeypatch):
    from ultron.config import get_config

    monkeypatch.setattr(get_config().hooks, "enabled", False)
    r = _runner()
    assert r._make_hook_lifecycle_listener(handle=None) is None
