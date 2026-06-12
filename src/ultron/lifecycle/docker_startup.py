"""Startup helper: ensure SearxNG's Docker backend is running.

SearxNG is Ultron's default first search provider, but it lives in a
Docker container -- if Docker Desktop is down at boot, every search
silently falls through to Brave. This module probes SearxNG once at
startup and, if it's unreachable, launches Docker Desktop in the
background (Docker takes 30-60s to bring the engine + container up, so
the launch never blocks the voice loop; the container auto-restarts
with the daemon).

Fully fail-open and side-effect-light: a reachable SearxNG, a missing
Docker binary, a non-Windows host, or any probe error simply returns a
structured result and leaves the provider chain to fall through as
before. Gated by ``web_search.searxng.autostart_docker_on_boot``.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("ultron.lifecycle.docker_startup")

__all__ = ["DockerStartupResult", "searxng_reachable", "ensure_docker_running"]

_DEFAULT_DOCKER_PATH = r"C:\Program Files\Docker\Docker\Docker Desktop.exe"


@dataclass(frozen=True)
class DockerStartupResult:
    """Outcome of the startup Docker check.

    Attributes:
        action: ``"already_up"`` (SearxNG reachable, nothing to do),
            ``"launched"`` (Docker Desktop spawn issued),
            ``"skipped"`` (disabled / non-Windows),
            ``"unavailable"`` (Docker binary missing or spawn failed).
        detail: short human-readable note.
    """

    action: str
    detail: str = ""


def searxng_reachable(base_url: str, timeout_s: float = 2.0) -> bool:
    """True iff SearxNG answers an HTTP probe at ``base_url``.

    Args:
        base_url: the SearxNG root (e.g. ``http://localhost:8888``).
        timeout_s: per-probe timeout.

    Returns:
        True on any HTTP response (even an error code -- the service is
        up); False on connection failure / timeout. Never raises.
    """
    url = base_url.rstrip("/") + "/healthz"
    for candidate in (url, base_url.rstrip("/") + "/"):
        try:
            with urllib.request.urlopen(candidate, timeout=timeout_s) as resp:
                if 200 <= getattr(resp, "status", 200) < 500:
                    return True
        except Exception:  # noqa: BLE001 - any failure = not reachable here
            continue
    return False


def ensure_docker_running(
    *,
    base_url: str,
    enabled: bool,
    docker_executable_path: Optional[str] = None,
    probe_fn: Optional[Callable[[str], bool]] = None,
    spawn_fn: Optional[Callable[..., object]] = None,
    exists_fn: Optional[Callable[[str], bool]] = None,
    background: bool = True,
) -> DockerStartupResult:
    """Probe SearxNG and launch Docker Desktop if it's unreachable.

    Args:
        base_url: SearxNG root to probe.
        enabled: the ``autostart_docker_on_boot`` flag.
        docker_executable_path: explicit path to ``Docker Desktop.exe``;
            falls back to the standard install location.
        probe_fn / spawn_fn / exists_fn: test seams (default to
            :func:`searxng_reachable`, ``subprocess.Popen``, and
            ``os.path.exists``).
        background: when True the probe+launch runs on a daemon thread
            and the call returns immediately with ``action="launched"``
            being decided off-thread; set False for synchronous tests.

    Returns:
        A :class:`DockerStartupResult`. In background mode the result
        describes the dispatch, not the eventual Docker state.
    """
    if not enabled:
        return DockerStartupResult("skipped", "autostart disabled")
    if sys.platform != "win32":
        return DockerStartupResult("skipped", "docker autostart is windows-only")

    probe = probe_fn or searxng_reachable
    spawn = spawn_fn or subprocess.Popen
    exists = exists_fn or (lambda p: Path(p).exists())
    executable = docker_executable_path or _DEFAULT_DOCKER_PATH

    def _work() -> DockerStartupResult:
        try:
            if probe(base_url):
                logger.info("SearxNG already reachable; Docker check skipped")
                return DockerStartupResult("already_up", base_url)
            if not exists(executable):
                logger.warning(
                    "SearxNG unreachable and Docker not found at %s; "
                    "search will fall through to Brave/DDG", executable,
                )
                return DockerStartupResult(
                    "unavailable", f"docker not found at {executable}",
                )
            spawn(
                [executable],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
            )
            logger.info(
                "SearxNG unreachable; launched Docker Desktop (%s) -- "
                "container will come up in ~30-60s", executable,
            )
            return DockerStartupResult("launched", executable)
        except Exception as e:  # noqa: BLE001 - fail-open
            logger.warning("docker autostart failed: %s", e)
            return DockerStartupResult("unavailable", str(e)[:200])

    if background:
        threading.Thread(
            target=_work, name="docker-autostart", daemon=True,
        ).start()
        return DockerStartupResult("launched", "dispatched on daemon thread")
    return _work()
