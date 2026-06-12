"""Tests for the startup Docker/SearxNG autostart helper."""

from __future__ import annotations

import subprocess

import pytest

from ultron.lifecycle.docker_startup import (
    DockerStartupResult,
    ensure_docker_running,
    searxng_reachable,
)


def _opts(**over):
    base = dict(
        base_url="http://localhost:8888",
        enabled=True,
        docker_executable_path="X:/Docker/Docker Desktop.exe",
        background=False,
    )
    base.update(over)
    return base


def test_skipped_when_disabled() -> None:
    res = ensure_docker_running(**_opts(enabled=False))
    assert res.action == "skipped"


def test_already_up_when_searxng_reachable() -> None:
    spawned: list = []
    res = ensure_docker_running(**_opts(
        probe_fn=lambda url: True,
        spawn_fn=lambda *a, **k: spawned.append(a),
        exists_fn=lambda p: True,
    ))
    assert res.action == "already_up"
    assert spawned == []  # Docker never launched when already up


def test_launches_docker_when_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.platform", "win32")
    spawned: list = []

    res = ensure_docker_running(**_opts(
        probe_fn=lambda url: False,
        exists_fn=lambda p: True,
        spawn_fn=lambda argv, **k: spawned.append(argv),
    ))
    assert res.action == "launched"
    assert spawned and spawned[0][0].endswith("Docker Desktop.exe")


def test_unavailable_when_docker_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.platform", "win32")
    res = ensure_docker_running(**_opts(
        probe_fn=lambda url: False,
        exists_fn=lambda p: False,
        spawn_fn=lambda *a, **k: pytest.fail("must not spawn"),
    ))
    assert res.action == "unavailable"
    assert "not found" in res.detail


def test_fail_open_on_spawn_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.platform", "win32")

    def boom(argv, **k):
        raise OSError("spawn denied")

    res = ensure_docker_running(**_opts(
        probe_fn=lambda url: False,
        exists_fn=lambda p: True,
        spawn_fn=boom,
    ))
    assert res.action == "unavailable"


def test_non_windows_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    res = ensure_docker_running(**_opts())
    assert res.action == "skipped"
    assert "windows" in res.detail.lower()


def test_background_returns_immediately() -> None:
    res = ensure_docker_running(**_opts(
        background=True, probe_fn=lambda url: False, exists_fn=lambda p: True,
        spawn_fn=lambda *a, **k: None,
    ))
    assert res.action == "launched"
    assert "daemon" in res.detail


def test_searxng_reachable_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda url, timeout=0: _Resp(),
    )
    assert searxng_reachable("http://localhost:8888") is True


def test_searxng_unreachable_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url, timeout=0):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert searxng_reachable("http://localhost:8888") is False


def test_searxng_config_autostart_default() -> None:
    from ultron.config import SearxNGConfig

    assert SearxNGConfig().autostart_docker_on_boot is True
