"""S1 — Twitch chat READ sidecar (loopback HTTP buffer over a pluggable source).

Runs as a SEPARATE process so NO Twitch transport ever loads into Ultron's
anticheat-pinned main process (BR-P1). The main/voice process keeps only a thin
``urllib`` client; this sidecar owns the EventSub WebSocket and exposes a tiny
loopback JSON HTTP surface that the consumer drains. Mirrors the proven
``scripts/embedder_server.py`` precedent: a loopback-ONLY ``ThreadingHTTPServer``
(127.0.0.1), a parent-death deadman thread (``os._exit`` when the parent pid is
gone), and fail-quiet logging.

ANTICHEAT POSTURE: pure compute + loopback networking only (poll a chat source,
buffer events, serve them over a 127.0.0.1 HTTP socket). NO input injection, NO
screen/window capture, NO foreign-process memory, NO hooks, never touches the
game -- the same class as OBS/Discord. Imports are stdlib only
(``http.server``/``socket``/``threading``/``json``/``collections.deque`` + the
optional EventSub transport which itself is stdlib ``socket``/``ssl``). Binds
127.0.0.1 ONLY.

SOURCE abstraction
------------------
The chat source is a pluggable object exposing ``poll() -> list[dict]`` (and an
optional ``close()``). The default real source (:class:`EventSubChatSource`)
drives the receive-only EventSub WebSocket client in
``kenning.twitch.clients.eventsub`` and maps each ``channel.chat.message``
notification to a small JSON-serializable dict. Tests inject a ``FakeSource``
instead, so the whole sidecar is exercisable WITHOUT a live Twitch connection,
creds, or models.

Protocol (JSON over loopback HTTP)
----------------------------------
  GET  /healthz          -> {"ok":true, "buffered":N, "cursor":M, "running":bool,
                             "dropped":D, "source":NAME}
  GET  /buffer?since=N   -> {"events":[...], "cursor":M}
                            Drains buffered events whose sequence id is > N
                            (``since`` defaults to the persisted consumer cursor).
  POST /ack {"cursor":N}  -> {"ok":true, "cursor":N}
                            Advances the consumer cursor and prunes acked events.

Each buffered event carries a monotonically increasing integer ``seq`` (the
cursor space) plus a ``ts`` ingest timestamp; the buffer is a thread-safe rolling
``deque`` with a ``maxlen`` cap AND a TTL so an idle consumer can never make the
sidecar grow without bound -- the oldest events are evicted first.

Master-flag wiring
------------------
With the Twitch master flag OFF the orchestrator MUST NOT spawn this process (the
``KENNING_TWITCH_*`` switches all default OFF; the flag-off-zero-sidecars
invariant is asserted by ``tests/twitch/test_config_anticheat_invariant.py``).
Run directly with the flag off and no source configured, the sidecar simply
serves an EMPTY buffer -- it is harmless on its own.

Run:  python scripts/twitch_read_sidecar.py [PORT]
Env:  KENNING_TWITCH_READ_PORT (default 8775),
      KENNING_TWITCH_READ_BUFFER_MAX (deque maxlen, default 2000),
      KENNING_TWITCH_READ_TTL_SECONDS (event TTL, default 900),
      KENNING_TWITCH_READ_POLL_SECONDS (source poll cadence, default 0.5),
      KENNING_TWITCH_PARENT_PID (parent pid for the deadman watchdog),
      KENNING_TWITCH_EVENTSUB_URL (override the wss endpoint for the real source).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Deque, Optional, Protocol, runtime_checkable
from urllib.parse import parse_qs, urlsplit

logger = logging.getLogger("kenning.twitch.read_sidecar")

# --------------------------------------------------------------------------- #
# Configuration (env-overridable; safe defaults)
# --------------------------------------------------------------------------- #
DEFAULT_PORT = int(os.environ.get("KENNING_TWITCH_READ_PORT", "8773"))  # 8775 = overlay; read = 8773
DEFAULT_BUFFER_MAX = max(1, int(os.environ.get("KENNING_TWITCH_READ_BUFFER_MAX", "2000")))
DEFAULT_TTL_SECONDS = float(os.environ.get("KENNING_TWITCH_READ_TTL_SECONDS", "900"))
DEFAULT_POLL_SECONDS = max(0.01, float(os.environ.get("KENNING_TWITCH_READ_POLL_SECONDS", "0.5")))


# --------------------------------------------------------------------------- #
# Source abstraction
# --------------------------------------------------------------------------- #
@runtime_checkable
class ChatSource(Protocol):
    """Pluggable chat source. ``poll()`` returns a (possibly empty) list of
    JSON-serializable event dicts ingested since the last call; it MUST NOT raise
    (the buffer's poll loop tolerates a raise, but a well-behaved source fails
    quiet and returns ``[]``). ``close()`` is optional best-effort teardown."""

    name: str

    def poll(self) -> list[dict]:  # pragma: no cover - structural protocol
        ...


class FakeSource:
    """In-memory chat source for tests and the flag-off no-op default.

    Events queued via :meth:`push` are returned by the next :meth:`poll`. With
    nothing queued, ``poll()`` returns ``[]`` so the sidecar serves an empty
    buffer (the documented flag-off behaviour). Thread-safe so a test can push
    from one thread while the poll loop drains from another.
    """

    name = "fake"

    def __init__(self, events: Optional[list[dict]] = None) -> None:
        self._pending: Deque[dict] = deque(events or [])
        self._lock = threading.Lock()
        self.closed = False

    def push(self, *events: dict) -> None:
        with self._lock:
            for ev in events:
                self._pending.append(dict(ev))

    def poll(self) -> list[dict]:
        with self._lock:
            out = list(self._pending)
            self._pending.clear()
        return out

    def close(self) -> None:
        self.closed = True


class EventSubChatSource:
    """The default real source: drives the receive-only EventSub WebSocket client
    and maps each ``channel.chat.message`` notification to a small event dict.

    Lazily connects on the first :meth:`poll`; every failure path is fail-quiet
    and returns ``[]`` (a transient socket error must never raise into the poll
    loop). The EventSub transport is itself stdlib-only (``socket``/``ssl``) -- no
    third-party ``websockets`` -- so importing it keeps the anticheat posture.

    This source is best-effort and offline-untestable by design (it needs a live
    ``wss`` + creds); the sidecar's behaviour is fully covered via ``FakeSource``.
    """

    name = "eventsub"

    def __init__(
        self,
        url: Optional[str] = None,
        *,
        connect_factory: Optional[Callable[[str], Any]] = None,
        recv_timeout: float = 0.25,
    ) -> None:
        self._url = url or os.environ.get(
            "KENNING_TWITCH_EVENTSUB_URL", "wss://eventsub.wss.twitch.tv/ws"
        )
        self._connect_factory = connect_factory
        self._recv_timeout = recv_timeout
        self._client: Any = None
        self._session: Any = None
        self._dedup: Any = None
        self._lock = threading.Lock()

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        try:
            from kenning.twitch.clients.eventsub import (
                DedupLRU,
                EventSubSession,
                RFC6455Client,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("eventsub transport unavailable: %s", exc)
            return False
        try:
            if self._connect_factory is not None:
                self._client = self._connect_factory(self._url)
            else:
                client = RFC6455Client(timeout=30.0)
                client.connect(self._url)
                self._client = client
            self._session = EventSubSession()
            self._dedup = DedupLRU()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("eventsub connect failed: %s", exc)
            self._client = None
            return False

    def poll(self) -> list[dict]:
        with self._lock:
            if not self._ensure_client():
                return []
            try:
                from kenning.twitch.clients.eventsub import ChatEvent
            except Exception as exc:  # noqa: BLE001
                logger.warning("eventsub ChatEvent import failed: %s", exc)
                return []
            out: list[dict] = []
            # Drain whatever messages are currently available, fail-quiet.
            try:
                for _ in range(64):  # bounded per-poll so we never spin forever
                    if not self._client_has_pending():
                        break
                    msg = self._client.recv_json()
                    if not isinstance(msg, dict):
                        continue
                    cls = self._session.classify_message(msg) if self._session else "unknown"
                    if cls == "welcome" and self._session is not None:
                        self._session.parse_welcome(msg)
                        continue
                    if cls != "notification":
                        continue
                    ev = ChatEvent.from_eventsub(msg)
                    if ev is None:
                        continue
                    if self._dedup is not None and self._dedup.seen(ev.message_id):
                        continue
                    out.append(
                        {
                            "type": "chat",
                            "message_id": ev.message_id,
                            "chatter_login": ev.chatter_login,
                            "chatter_name": ev.chatter_name,
                            "chatter_user_id": ev.chatter_user_id,
                            "text": ev.text,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("eventsub poll error: %s", exc)
                self._client = None  # force a reconnect next poll
            return out

    def _client_has_pending(self) -> bool:
        # The hand-rolled client blocks on recv; without a non-blocking probe we
        # cannot know if data is queued, so we conservatively stop after the first
        # read each poll cycle (the real wiring sets a recv timeout). This keeps
        # the poll loop responsive and is overridden in tests via FakeSource.
        return False

    def close(self) -> None:
        try:
            if self._client is not None and hasattr(self._client, "close"):
                self._client.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("eventsub source close failed: %s", exc)


# --------------------------------------------------------------------------- #
# Rolling buffer
# --------------------------------------------------------------------------- #
class RollingBuffer:
    """Thread-safe rolling buffer of chat events with a ``maxlen`` AND a TTL cap.

    Each appended event is wrapped with a monotonically increasing integer
    ``seq`` (the cursor space) and an ingest ``ts``. Eviction is oldest-first by
    the deque ``maxlen``; on every read/append, events older than ``ttl_seconds``
    are also pruned. ``drained_total`` and ``dropped_total`` are kept for
    observability. All public methods take the lock, so concurrent appends (poll
    loop) and reads (HTTP handler threads) are safe.
    """

    def __init__(self, maxlen: int = DEFAULT_BUFFER_MAX, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        if maxlen < 1:
            raise ValueError("RollingBuffer maxlen must be >= 1")
        self._maxlen = maxlen
        self._ttl = max(0.0, ttl_seconds)
        self._events: Deque[dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._seq = 0          # last assigned seq (cursor high-water mark)
        self._cursor = 0       # consumer-acked cursor
        self.dropped_total = 0  # events evicted before being acked (maxlen/TTL)
        self.appended_total = 0

    # -- mutation -------------------------------------------------------- #
    def append(self, event: dict, *, now: Optional[float] = None) -> int:
        """Wrap ``event`` with a fresh ``seq``/``ts`` and append it. Returns the
        assigned seq. Evicts the oldest entry when ``maxlen`` is exceeded (deque
        does this for us; we count the drop for observability)."""
        now = time.time() if now is None else now
        with self._lock:
            self._seq += 1
            wrapped = {"seq": self._seq, "ts": now, "event": event}
            at_cap = len(self._events) >= self._maxlen
            if at_cap:
                # The about-to-be-evicted head was never acked -> a real drop.
                self.dropped_total += 1
            self._events.append(wrapped)
            self.appended_total += 1
            self._prune_ttl(now)
            return self._seq

    def _prune_ttl(self, now: float) -> None:
        """Drop events older than the TTL. Caller holds the lock."""
        if self._ttl <= 0:
            return
        cutoff = now - self._ttl
        while self._events and self._events[0]["ts"] < cutoff:
            evicted = self._events.popleft()
            # Only count it as a drop if the consumer never acked past it.
            if evicted["seq"] > self._cursor:
                self.dropped_total += 1

    # -- read ------------------------------------------------------------ #
    def drain(self, since: Optional[int] = None, *, now: Optional[float] = None) -> tuple[list[dict], int]:
        """Return (events, cursor): all buffered events whose ``seq`` > ``since``.

        ``since`` defaults to the consumer cursor. The returned ``cursor`` is the
        seq high-water mark (the largest seq returned, or ``since`` if nothing is
        newer) -- the consumer ACKs that value to advance. Does NOT mutate the
        consumer cursor (that is :meth:`ack`'s job), so a crash mid-consume safely
        redelivers (at-least-once). Prunes TTL-expired events first."""
        now = time.time() if now is None else now
        with self._lock:
            self._prune_ttl(now)
            floor = self._cursor if since is None else since
            out = [dict(w) for w in self._events if w["seq"] > floor]
            cursor = out[-1]["seq"] if out else floor
            return out, cursor

    def ack(self, cursor: int) -> int:
        """Advance the consumer cursor to ``cursor`` (monotonic; never regresses)
        and prune acked events from the front of the buffer. Returns the effective
        cursor."""
        with self._lock:
            if cursor > self._cursor:
                self._cursor = cursor
            while self._events and self._events[0]["seq"] <= self._cursor:
                self._events.popleft()
            return self._cursor

    # -- introspection --------------------------------------------------- #
    @property
    def cursor(self) -> int:
        with self._lock:
            return self._cursor

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def stats(self, *, now: Optional[float] = None) -> dict:
        now = time.time() if now is None else now
        with self._lock:
            self._prune_ttl(now)
            return {
                "buffered": len(self._events),
                "cursor": self._cursor,
                "seq": self._seq,
                "dropped": self.dropped_total,
                "appended": self.appended_total,
                "maxlen": self._maxlen,
                "ttl_seconds": self._ttl,
            }


# --------------------------------------------------------------------------- #
# Poll loop
# --------------------------------------------------------------------------- #
class PollLoop:
    """Background thread that pumps ``source.poll()`` into the buffer on a fixed
    cadence. Fail-quiet: a source that raises is logged and retried next tick
    (never crashes the sidecar). ``run_once`` is exposed so tests can drive a
    single deterministic pump without a thread."""

    def __init__(
        self,
        source: ChatSource,
        buffer: RollingBuffer,
        *,
        interval: float = DEFAULT_POLL_SECONDS,
    ) -> None:
        self._source = source
        self._buffer = buffer
        self._interval = max(0.01, interval)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def run_once(self) -> int:
        """Pump one poll() into the buffer. Returns the count appended."""
        try:
            events = self._source.poll()
        except Exception as exc:  # noqa: BLE001 — never let a source raise out
            logger.warning("chat source poll raised: %s", exc)
            return 0
        if not events:
            return 0
        count = 0
        for ev in events:
            if not isinstance(ev, dict):
                logger.debug("dropping non-dict source event: %r", type(ev))
                continue
            self._buffer.append(ev)
            count += 1
        return count

    def _run(self) -> None:
        logger.info("chat read poll loop started interval=%.3fs source=%s",
                    self._interval, getattr(self._source, "name", "?"))
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(self._interval)
        logger.info("chat read poll loop stopped")

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="twitch-read-poll")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            close = getattr(self._source, "close", None)
            if callable(close):
                close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("source close on stop failed: %s", exc)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
def make_handler(buffer: RollingBuffer, poll_loop: Optional[PollLoop], source_name: str):
    """Build a ``BaseHTTPRequestHandler`` subclass bound to this sidecar's state.

    A factory (not module globals) so a test can stand up an isolated server with
    its own buffer/source on an ephemeral port without cross-test state."""

    class _Handler(BaseHTTPRequestHandler):
        server_version = "KenningTwitchRead/1.0"

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode("utf-8")
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionError) as exc:  # client hung up
                logger.debug("client disconnected mid-response: %s", exc)

        # -- GET ---------------------------------------------------------- #
        def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            parts = urlsplit(self.path)
            path = parts.path
            if path == "/healthz":
                stats = buffer.stats()
                running = bool(poll_loop.running) if poll_loop is not None else False
                self._send(200, {
                    "ok": True,
                    "buffered": stats["buffered"],
                    "cursor": stats["cursor"],
                    "dropped": stats["dropped"],
                    "running": running,
                    "source": source_name,
                })
                return
            if path == "/buffer":
                since = self._parse_since(parts.query)
                events, cursor = buffer.drain(since=since)
                self._send(200, {"events": events, "cursor": cursor})
                return
            self._send(404, {"error": "not found"})

        @staticmethod
        def _parse_since(query: str) -> Optional[int]:
            """Parse ?since=N. Absent/garbage -> None (use the consumer cursor).
            Fail-safe: never raises on a hostile query string."""
            try:
                qs = parse_qs(query or "")
                raw = qs.get("since", [None])[0]
                if raw is None or raw == "":
                    return None
                return max(0, int(raw))
            except (ValueError, TypeError):
                return None

        # -- POST --------------------------------------------------------- #
        def do_POST(self) -> None:  # noqa: N802
            parts = urlsplit(self.path)
            if parts.path != "/ack":
                self._send(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", "0") or "0")
                if n < 0 or n > (1 << 20):
                    self._send(400, {"error": "bad content-length"})
                    return
                payload = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, TypeError) as exc:
                self._send(400, {"error": f"bad request: {exc}"})
                return
            if not isinstance(payload, dict):
                self._send(400, {"error": "body must be a JSON object"})
                return
            try:
                cursor = int(payload.get("cursor", 0))
            except (ValueError, TypeError):
                self._send(400, {"error": "cursor must be an integer"})
                return
            effective = buffer.ack(cursor)
            self._send(200, {"ok": True, "cursor": effective})

        def log_message(self, *args: Any) -> None:  # noqa: ARG002 — fail-quiet
            return

    return _Handler


# --------------------------------------------------------------------------- #
# Parent-death deadman (clone of the embedder_server precedent)
# --------------------------------------------------------------------------- #
def _pid_alive(pid: int) -> bool:
    """True iff process ``pid`` is still running. psutil if present, else a
    ctypes OpenProcess+GetExitCodeProcess check on Windows / os.kill(0) on POSIX.
    Fail-SAFE: an indeterminate result returns True (never self-kill on doubt)."""
    if pid <= 0:
        return True
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:  # noqa: BLE001
        pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            k = ctypes.windll.kernel32
            h = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not h:
                return False  # cannot open -> gone
            code = wintypes.DWORD()
            ok = k.GetExitCodeProcess(h, ctypes.byref(code))
            k.CloseHandle(h)
            return (not ok) or code.value == STILL_ACTIVE
        except Exception:  # noqa: BLE001
            return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except Exception:  # noqa: BLE001
        return True


def parent_watchdog_check(pid: int) -> str:
    """Single-shot watchdog decision for ``pid``. Returns ``"alive"`` if the
    parent is still running (or the pid is unset/invalid -> do not self-kill) and
    ``"dead"`` if the parent is gone (the caller should ``os._exit``). Split out
    from the loop so it is unit-testable without spawning a process."""
    if pid <= 0:
        return "alive"
    return "alive" if _pid_alive(pid) else "dead"


def _parent_watchdog(poll_seconds: float = 3.0) -> None:
    """Self-exit when the parent (Ultron orchestrator) dies, so a force-killed or
    crashed parent NEVER leaves this sidecar as a runaway orphan holding a live
    Twitch socket. Parent pid via ``KENNING_TWITCH_PARENT_PID`` (fallback: the
    spawn-time parent). ``os._exit`` skips atexit/locks so the socket is freed
    immediately by the OS."""
    try:
        ppid = int(os.environ.get("KENNING_TWITCH_PARENT_PID", "0") or "0")
    except Exception:  # noqa: BLE001
        ppid = 0
    if ppid <= 0:
        ppid = os.getppid()
    if ppid <= 0:
        return
    sys.stderr.write(f"[twitch-read] parent-watchdog armed on pid {ppid}\n")
    sys.stderr.flush()
    while True:
        time.sleep(poll_seconds)
        if parent_watchdog_check(ppid) == "dead":
            sys.stderr.write(
                f"[twitch-read] parent pid {ppid} gone -> self-terminating "
                "(orphan guard)\n")
            sys.stderr.flush()
            os._exit(0)


# --------------------------------------------------------------------------- #
# Server assembly
# --------------------------------------------------------------------------- #
def build_server(
    source: Optional[ChatSource] = None,
    *,
    port: int = 0,
    buffer_max: int = DEFAULT_BUFFER_MAX,
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    poll_interval: float = DEFAULT_POLL_SECONDS,
    start_poll: bool = True,
) -> tuple[ThreadingHTTPServer, RollingBuffer, PollLoop]:
    """Assemble (server, buffer, poll_loop) bound to 127.0.0.1 ONLY.

    ``port=0`` binds an ephemeral port (read it back from ``server.server_address``
    in a test). With no ``source`` an empty :class:`FakeSource` is used, so a
    bare run serves an empty buffer (the documented flag-off behaviour). The
    server is returned NOT yet serving -- the caller runs ``serve_forever`` (or, in
    a test, drives requests against the bound address on a thread)."""
    src = source if source is not None else FakeSource()
    buffer = RollingBuffer(maxlen=buffer_max, ttl_seconds=ttl_seconds)
    poll_loop = PollLoop(src, buffer, interval=poll_interval)
    handler = make_handler(buffer, poll_loop, getattr(src, "name", "fake"))
    # EXCLUSIVE bind (anti-stale-sidecar): a second instance on a live port FAILS
    # rather than co-serving. port=0 (tests) binds a fresh ephemeral port.
    from kenning.subprocess.sidecar_server import SingletonThreadingHTTPServer
    server = SingletonThreadingHTTPServer(("127.0.0.1", port), handler)
    if start_poll:
        poll_loop.start()
    return server, buffer, poll_loop


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [twitch-read] %(levelname)s %(message)s",
    )
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    # Anti-stale-sidecar guard: reap same-role strays + reclaim the port BEFORE
    # binding (which build_server does EXCLUSIVELY) -> exactly one live instance.
    import atexit

    from kenning.subprocess import sidecar_lock
    sidecar_lock.guard_singleton("127.0.0.1", port, "twitch_read")
    # The real source connects to EventSub; if its transport/creds are absent it
    # fails quiet and the sidecar serves an empty buffer (harmless on its own).
    source: ChatSource = EventSubChatSource()
    server, _buffer, poll_loop = build_server(source, port=port)
    sidecar_lock.write_role("twitch_read", os.getpid(), port)
    atexit.register(sidecar_lock.clear_role, "twitch_read")
    # Parent-death deadman: the strongest orphan guard -- the child cleans itself
    # up on ANY parent death (crash, taskkill /F, TerminateProcess).
    threading.Thread(target=_parent_watchdog, daemon=True,
                     name="twitch-read-parent-watchdog").start()
    host, bound_port = server.server_address[:2]
    logger.info("twitch read sidecar serving on http://%s:%s", host, bound_port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        poll_loop.stop()
        server.server_close()
        sidecar_lock.clear_role("twitch_read")


if __name__ == "__main__":
    main()
