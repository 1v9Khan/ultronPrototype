"""Single-instance loopback HTTP server for sidecars (the anti-stale-sidecar guard).

The failure this prevents (observed 2026-06-21: three guard sidecars co-bound to
:8774, requests round-robining to stale-code instances): on Windows,
``ThreadingHTTPServer`` defaults to ``allow_reuse_address=True`` (SO_REUSEADDR),
which lets MULTIPLE processes bind the SAME ``127.0.0.1:port`` at once — a
force-killed-but-still-bound predecessor keeps serving alongside a freshly started
instance, and a naive PID-tracking restart (``$!`` is the nohup wrapper, not the
python) never reaps it.

:class:`SingletonThreadingHTTPServer` binds EXCLUSIVELY: a second bind on a live
port FAILS LOUDLY (``OSError``) instead of silently co-serving. Use it together
with :func:`kenning.subprocess.sidecar_lock.guard_singleton` (reap same-role
strays + reclaim the port) BEFORE constructing the server, so a restart
deterministically reclaims the port from a crashed predecessor and there is never
more than one live instance.

Pure stdlib — importable from any sidecar process.
"""
from __future__ import annotations

import socket
import sys
from http.server import ThreadingHTTPServer

__all__ = ["SingletonThreadingHTTPServer"]


class SingletonThreadingHTTPServer(ThreadingHTTPServer):
    """A ThreadingHTTPServer that NEVER shares its port.

    ``allow_reuse_address = False`` so the base ``server_bind`` does not set
    SO_REUSEADDR; on Windows we additionally set SO_EXCLUSIVEADDRUSE (the inverse
    of SO_REUSEADDR) so NO other socket — in any process — can bind this port
    while we hold it. The net effect: a second instance trying to bind a live
    port raises ``OSError`` at startup rather than silently co-serving stale code.
    """

    daemon_threads = True
    allow_reuse_address = False

    def server_bind(self) -> None:  # noqa: D401 - stdlib override
        if sys.platform == "win32":
            excl = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
            if excl is not None:
                try:
                    self.socket.setsockopt(socket.SOL_SOCKET, excl, 1)
                except OSError:
                    # Already-bound / unsupported -> let the bind below fail loudly.
                    pass
        super().server_bind()
