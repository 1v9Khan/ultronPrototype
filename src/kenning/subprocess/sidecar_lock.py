"""Singleton lifecycle lock for the command-router embedder sidecar.

The sidecar (``scripts/embedder_server.py``, EmbeddingGemma) runs as a SEPARATE
process in an isolated venv. Two failure modes must NEVER happen:

* **ORPHAN** -- a previous Ultron was force-killed (``taskkill /F`` /
  TerminateProcess is uncatchable, so no in-process cleanup ran) and left the
  embedder bound to the port. A naive next boot would blindly reuse an
  unknown/stale process (``EmbeddingBackend.available()`` returns True for ANY
  HTTP 200 on the port).
* **DUPLICATE** -- a manually-run sidecar, or a half-dead one, racing on the
  port; only one wins the socket, the others linger as silent RAM/VRAM consumers.

This module makes the sidecar a VERIFIED SINGLETON: a pidfile records who owns
the port and which model it serves, and a boot-time :func:`sweep` positively
reaps anything on the port we don't recognise BEFORE spawning. Every operation
FAILS OPEN (never raises into boot) and tolerates a missing ``psutil``.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

from kenning.utils.logging import get_logger

logger = get_logger("subprocess.sidecar_lock")


def default_pidfile() -> Path:
    return Path.home() / ".kenning" / "embedder_sidecar.json"


def _path(path: "str | Path | None") -> Path:
    return Path(path) if path else default_pidfile()


def write(pid: int, port: int, model: str, backend: str,
          path: "str | Path | None" = None) -> None:
    """Atomically record the sidecar owner (write-temp then replace). Never raises."""
    p = _path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps({
            "pid": int(pid), "port": int(port),
            "model": model, "backend": backend, "owner_pid": os.getpid(),
        }), encoding="utf-8")
        tmp.replace(p)
    except Exception as e:                                        # noqa: BLE001
        logger.debug("sidecar pidfile write failed (%s)", e)


def read(path: "str | Path | None" = None) -> Optional[dict]:
    try:
        return json.loads(_path(path).read_text(encoding="utf-8"))
    except Exception:                                            # noqa: BLE001
        return None


def clear(path: "str | Path | None" = None) -> None:
    try:
        _path(path).unlink()
    except Exception:                                            # noqa: BLE001
        pass


def _healthz(host: str, port: int, timeout: float = 0.6) -> Optional[dict]:
    """GET /healthz -> parsed dict, or None when nothing answers."""
    try:
        req = urllib.request.Request(f"http://{host}:{port}/healthz", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status == 200:
                return json.loads(r.read().decode("utf-8"))
    except Exception:                                            # noqa: BLE001
        return None
    return None


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return bool(psutil.pid_exists(int(pid)))
    except Exception:                                            # noqa: BLE001
        return False


def _listener_pid(port: int) -> Optional[int]:
    """PID holding a LISTEN socket on ``port`` (psutil). Per-socket AccessDenied
    is swallowed; returns None when nothing matches / psutil is unreadable."""
    try:
        import psutil
        conns = psutil.net_connections(kind="inet")
    except Exception:                                            # noqa: BLE001
        return None
    for c in conns:
        try:
            if (c.laddr and c.laddr.port == int(port)
                    and c.status == psutil.CONN_LISTEN and c.pid):
                return int(c.pid)
        except Exception:                                        # noqa: BLE001
            continue
    return None


def _kill(pid: Optional[int]) -> int:
    if not pid or int(pid) <= 0:
        return 0
    try:
        from kenning.subprocess.kill_tree import kill_process_tree
        res = kill_process_tree(int(pid), grace_seconds=4.0)
        return int(getattr(res, "killed", 0) or getattr(res, "terminated", 0) or 0)
    except Exception as e:                                        # noqa: BLE001
        logger.debug("sidecar kill_process_tree(%s) failed (%s)", pid, e)
        return 0


def sweep(host: str, port: int, model: str,
          path: "str | Path | None" = None) -> Tuple[str, Optional[int]]:
    """Reconcile the embedder port to a clean singleton state BEFORE spawn.

    Returns ``(verdict, pid)``:
      * ``("reuse", pid)``          -- our recorded sidecar is alive AND serving
        ``model`` -> caller should reuse it (and OWN it for cleanup).
      * ``("killed", None)``        -- recorded pid alive but WRONG model /
        unhealthy -> reaped; caller spawns fresh.
      * ``("killed-zombie", None)`` -- recorded pid DEAD but a process still
        serves the port (a force-killed shim left its embedder child) -> reaped.
      * ``("killed-unknown", None)``-- no pidfile but something serves the port
        -> reaped (loud WARNING).
      * ``("spawn", None)``         -- port is clear; caller spawns fresh.

    Never raises (fail-open to ``"spawn"``).
    """
    try:
        meta = read(path)
        rec_pid = int(meta["pid"]) if (meta and isinstance(meta.get("pid"), int)) else None
        health = _healthz(host, port)

        # 1) Our recorded sidecar is alive.
        if rec_pid is not None and _pid_alive(rec_pid):
            if health and str(health.get("model", "")) == str(model):
                return ("reuse", rec_pid)
            _kill(rec_pid)
            clear(path)
            logger.info("sidecar sweep: recorded pid %s alive but model "
                        "mismatch/unhealthy -> reaped, will respawn", rec_pid)
            return ("killed", None)

        # 2) Recorded pid dead (or no record) but SOMETHING serves the port:
        #    an orphan from a force-killed prior Ultron, or an unknown process.
        if health is not None:
            lp = _listener_pid(port)
            killed = _kill(lp)
            clear(path)
            if meta is not None:
                logger.warning("sidecar sweep: orphan embedder on %s:%d "
                               "(recorded pid %s is dead) -> reaped listener "
                               "pid=%s killed=%d", host, port, rec_pid, lp, killed)
                return ("killed-zombie", None)
            logger.warning("sidecar sweep: UNKNOWN process serving %s:%d with no "
                           "pidfile -> reaped pid=%s killed=%d", host, port, lp, killed)
            return ("killed-unknown", None)

        # 3) Nothing answers; clear any stale record and spawn fresh.
        if meta is not None:
            clear(path)
        return ("spawn", None)
    except Exception as e:                                        # noqa: BLE001
        logger.debug("sidecar sweep failed (%s); proceeding to spawn", e)
        return ("spawn", None)
