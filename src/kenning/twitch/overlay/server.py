"""The local OBS Browser Source overlay server (SSE, strict CSP, per-session token).

:class:`OverlayServer` runs a ``ThreadingHTTPServer`` bound to ``127.0.0.1`` ONLY.
Every route requires a per-session ``secrets.token_urlsafe`` token as ``?token=``
(missing/wrong -> ``403``, constant-time compared). Routes:

  * ``GET /``                 -> the self-contained ``static/overlay.html`` with a
                                 STRICT ``Content-Security-Policy`` response header.
  * ``GET /events?token=...`` -> a ``text/event-stream`` (SSE); streams the JSON
                                 events pushed via :meth:`OverlayServer.emit`.

The page is a DUMB renderer (it never decides outcomes — the sidecar's crypto RNG
picks the wheel winner first, then sends the target angle). All inbound events are
schema-validated (:func:`validate_event`); unknown event types are rejected before
ever reaching a client. The fan-out is a per-client bounded ``queue`` (drop-OLDEST
on overflow so a stalled OBS source can never wedge the producer). Client
disconnects fail QUIET.

ANTICHEAT (BR-P1): pure stdlib only — ``http.server`` / ``socketserver`` /
``secrets`` / ``json`` / ``queue`` / ``threading`` / ``html`` / ``hmac``. No
desktop/screen-capture/input-injection libs, no third-party network library, no
screen capture anywhere.
"""
from __future__ import annotations

import hmac
import json
import logging
import secrets
import threading
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Optional
from urllib.parse import parse_qs, urlsplit

from kenning.subprocess.sidecar_server import SingletonThreadingHTTPServer

logger = logging.getLogger("kenning.twitch.overlay.server")

__all__ = [
    "OverlayServer",
    "OverlayError",
    "validate_event",
    "CSP_POLICY",
    "ALLOWED_EVENT_TYPES",
]

# The single source of truth for the CSP. The served overlay.html embeds a
# byte-identical <meta http-equiv> copy; a test asserts they match.
#
# ``script-src 'self' 'unsafe-inline'`` is REQUIRED: the renderer is a single
# self-contained inline <script>. Without an explicit ``script-src`` directive the
# CSP falls back to ``default-src 'none'`` and OBS's CEF (Chromium Embedded) BLOCKS
# the inline script outright — so ``init()`` never runs, ``connect()`` is never
# called, and no ``GET /events`` SSE stream is ever opened (the page just re-fetches
# ``/`` on each OBS refresh and renders nothing). This mirrors the already-granted
# ``style-src 'self' 'unsafe-inline'`` for the inline <style>. The page still uses
# NO inline event-handler attributes and NO markup-setting sinks (textContent only),
# so this does not widen the real injection surface.
CSP_POLICY = (
    "default-src 'none'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; "
    "connect-src 'self'; "
    "img-src 'self' data:"
)

# The closed set of event types the overlay renders. Anything else is rejected at
# emit() time and never streamed. ``chat_game`` is the UNIFIED game-card type fed
# by BOTH the typed chat-command games AND the channel-point redeem games (they
# render as the SAME compact bottom-left card, distinguished only by ``source``);
# ``speech`` is the speak-redeem card. The legacy ``wheel``/``alert``/``ticker``
# types remain accepted for back-compat but are no longer produced by the routers.
ALLOWED_EVENT_TYPES: frozenset[str] = frozenset(
    {"wheel", "alert", "ticker", "chat_game", "speech"}
)

# The closed set of game discriminators a ``chat_game`` event may carry. The
# overlay renders a tailored, game-accurate card per value; an unknown game is
# rejected at validate_event() time so a typo never reaches a client.
ALLOWED_CHAT_GAMES: frozenset[str] = frozenset(
    {"slots", "wheel", "heist", "duel", "trivia", "raffle"}
)

# Where a unified game/speech card originated. Drives a small CHAT/REDEEM tag on
# the card; an unknown source is coerced to "chat" (never rejected).
ALLOWED_CARD_SOURCES: frozenset[str] = frozenset({"chat", "redeem"})

_HTML_PATH = Path(__file__).resolve().parent / "static" / "overlay.html"

# Bounded SSE fan-out queue: a slow/stalled OBS client drops its OLDEST event
# rather than growing without bound or blocking the producer.
_CLIENT_QUEUE_MAXSIZE = 256
# How often an idle SSE handler wakes to flush a keepalive comment + re-check that
# the server (and its parent) is still alive, so a dropped OBS source is reaped.
_SSE_POLL_SECONDS = 1.0
# Replay ring buffer: the last N already-vetted SSE frames are retained and
# REPLAYED to every newly-connected client, so a freshly-(re)connected OBS source
# (e.g. after a refresh or a server reboot) isn't blank and a card fired moments
# before the (re)connect still shows. Bounded + cheap (a fixed-size deque of small
# strings); secret keys are already stripped by validate_event() BEFORE a frame is
# ever buffered, so nothing sensitive is replayed.
_REPLAY_BUFFER_MAXLEN = 16


class OverlayError(ValueError):
    """A rejected overlay event (unknown type / malformed payload / oversize)."""


def _require_str(obj: dict[str, Any], key: str, *, max_len: int) -> str:
    val = obj.get(key, "")
    if not isinstance(val, str):
        raise OverlayError(f"field {key!r} must be a string, got {type(val).__name__}")
    if len(val) > max_len:
        raise OverlayError(f"field {key!r} too long ({len(val)} > {max_len})")
    return val


def _require_number(obj: dict[str, Any], key: str, *, lo: float, hi: float, default: Optional[float] = None) -> float:
    if key not in obj and default is not None:
        return default
    val = obj.get(key)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise OverlayError(f"field {key!r} must be a number")
    fval = float(val)
    if fval != fval or fval in (float("inf"), float("-inf")):  # NaN / inf
        raise OverlayError(f"field {key!r} is not finite")
    if not (lo <= fval <= hi):
        raise OverlayError(f"field {key!r} out of range [{lo},{hi}]: {fval}")
    return fval


def validate_event(event: Any) -> dict[str, Any]:
    """Schema-validate one overlay event; return a NEW normalized dict or raise.

    The returned dict contains only the known, range-checked fields for the type
    — defense in depth so a sink only ever sees a vetted shape. All string fields
    are passed through verbatim (the browser renders them via ``textContent``, so
    an ``<img onerror>`` payload is inert); JSON-encoding on the wire additionally
    escapes them. Fail-CLOSED: any unknown type / bad field raises.
    """
    if not isinstance(event, dict):
        raise OverlayError(f"event must be an object, got {type(event).__name__}")
    etype = event.get("type")
    if not isinstance(etype, str):
        raise OverlayError("event 'type' must be a string")
    if etype not in ALLOWED_EVENT_TYPES:
        raise OverlayError(f"unknown event type: {etype!r}")

    out: dict[str, Any] = {"type": etype}
    if etype == "wheel":
        # The OUTCOME is decided server-side; the overlay only spins to the
        # server-supplied target angle. The label is the (already-safety-screened)
        # winner display string.
        out["angle"] = _require_number(event, "angle", lo=-1.0e6, hi=1.0e6)
        out["label"] = _require_str(event, "label", max_len=200)
        out["duration_ms"] = _require_number(event, "duration_ms", lo=0.0, hi=60000.0, default=4000.0)
    elif etype == "alert":
        out["title"] = _require_str(event, "title", max_len=200)
        out["body"] = _require_str(event, "body", max_len=500)
        out["duration_ms"] = _require_number(event, "duration_ms", lo=0.0, hi=60000.0, default=6000.0)
    elif etype == "ticker":
        out["label"] = _require_str(event, "label", max_len=120)
        out["points"] = _require_number(event, "points", lo=-1.0e12, hi=1.0e12)
    elif etype == "chat_game":
        # A typed chat-command game OUTCOME (!slots/!wheel/!heist/!duel/!trivia/
        # !raffle). The discriminator ``game`` selects the overlay card; ``won``
        # color-codes win vs loss; ``detail`` is a SMALL, length-checked sub-dict
        # of per-game specifics the card renders game-accurately (reels, landed
        # segment, crew size, the trivia answer, ...). Every field is range/length
        # checked so a hostile chatter display name can only ever land as inert
        # text inside a vetted shape.
        game = _require_str(event, "game", max_len=32)
        if game not in ALLOWED_CHAT_GAMES:
            raise OverlayError(f"unknown chat_game game: {game!r}")
        out["game"] = game
        # discriminator: typed chat command vs a channel-point redeem. Both render
        # as the SAME card; the only visible difference is a CHAT/REDEEM tag.
        src = event.get("source", "chat")
        out["source"] = src if src in ALLOWED_CARD_SOURCES else "chat"
        out["viewer"] = _require_str(event, "viewer", max_len=80)
        out["outcome"] = _require_str(event, "outcome", max_len=120)
        out["title"] = _require_str(event, "title", max_len=120)
        out["amount"] = int(_require_number(event, "amount", lo=-1.0e12, hi=1.0e12, default=0.0))
        out["won"] = bool(event.get("won", False))
        out["detail"] = _validate_chat_game_detail(event.get("detail"))
        out["duration_ms"] = _require_number(
            event, "duration_ms", lo=0.0, hi=60000.0, default=7000.0
        )
    elif etype == "speech":
        # A SPEAK redeem: a viewer paid points to have Ultron speak their (already
        # guard-screened + sanitized) message. Rendered as the same bottom-left
        # card with a speaker/quote icon. ``bus`` distinguishes the stream-broadcast
        # "say" from the "team" voice variant (a subtle accent change on the card).
        bus = event.get("bus", "say")
        out["bus"] = bus if bus in ("say", "team") else "say"
        out["viewer"] = _require_str(event, "viewer", max_len=80)
        out["text"] = _require_str(event, "text", max_len=300)
        out["duration_ms"] = _require_number(
            event, "duration_ms", lo=0.0, hi=60000.0, default=7000.0
        )
    return out


# Per-game ``detail`` keys the overlay actually renders, with their validators.
# Anything not listed here is DROPPED (defense in depth: the card only ever sees a
# vetted shape). String values are length-capped; the browser renders them via
# textContent so they are inert regardless.
_CHAT_GAME_DETAIL_STR_KEYS: dict[str, int] = {
    "win_symbol": 40,
    "segment": 60,
    "answer": 200,
    "winner": 80,
    "loser": 80,
    "phase": 16,    # "open" / "result" — lets the card distinguish a join window from a payout
}
_CHAT_GAME_DETAIL_NUM_KEYS: tuple[str, ...] = (
    "payout", "pot", "crew", "wager", "entrants", "net", "stake", "multiplier",
)


def _validate_chat_game_detail(detail: Any) -> dict[str, Any]:
    """Validate a chat_game ``detail`` sub-dict: only the known, range/length-checked
    keys survive (unknown keys dropped). ``reels`` is a short list of short strings
    (the slot symbols the card settles on). Fail-CLOSED on a bad type."""
    out: dict[str, Any] = {}
    if detail is None:
        return out
    if not isinstance(detail, dict):
        raise OverlayError("chat_game 'detail' must be an object")
    # A None value means "absent" (e.g. slots win_symbol on a loss) -> skip it
    # rather than reject, so producers can pass an optional key unconditionally.
    for key, max_len in _CHAT_GAME_DETAIL_STR_KEYS.items():
        if key in detail and detail[key] is not None:
            out[key] = _require_str(detail, key, max_len=max_len)
    for key in _CHAT_GAME_DETAIL_NUM_KEYS:
        if key in detail and detail[key] is not None:
            out[key] = _require_number(detail, key, lo=-1.0e12, hi=1.0e12)
    reels = detail.get("reels")
    if reels is not None:
        if not isinstance(reels, (list, tuple)) or len(reels) > 8:
            raise OverlayError("chat_game detail 'reels' must be a list of <=8 symbols")
        clean_reels: list[str] = []
        for sym in reels:
            if not isinstance(sym, str) or len(sym) > 40:
                raise OverlayError("chat_game detail 'reels' entries must be short strings")
            clean_reels.append(sym)
        out["reels"] = clean_reels
    return out


class _OverlayHTTPServer(SingletonThreadingHTTPServer):
    """Exclusive-bind server (anti-stale-sidecar) carrying the owning OverlayServer.

    Base class sets ``daemon_threads=True`` + ``allow_reuse_address=False`` +
    SO_EXCLUSIVEADDRUSE so a second overlay can never co-bind the port."""

    def __init__(self, addr: tuple[str, int], handler: type[BaseHTTPRequestHandler], owner: "OverlayServer") -> None:
        self.owner = owner
        super().__init__(addr, handler)


class _OverlayHandler(BaseHTTPRequestHandler):
    server_version = "KenningOverlay/1.0"
    protocol_version = "HTTP/1.1"

    # --- helpers --------------------------------------------------------------
    @property
    def _owner(self) -> "OverlayServer":
        return self.server.owner  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        # Route stdlib access logs to our logger at DEBUG (never to stderr).
        logger.debug("overlay %s - %s", self.address_string(), fmt % args)

    def _token_ok(self, query: dict[str, list[str]]) -> bool:
        supplied = (query.get("token") or [""])[0]
        if not supplied:
            return False
        return hmac.compare_digest(supplied, self._owner.token)

    def _send_simple(self, status: HTTPStatus, body: bytes, content_type: str, *, extra_headers: Optional[dict[str, str]] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # client vanished mid-write — fail quiet

    def _send_403(self) -> None:
        self._send_simple(
            HTTPStatus.FORBIDDEN,
            b"403 Forbidden: missing or invalid token\n",
            "text/plain; charset=utf-8",
        )

    # --- routing --------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        parts = urlsplit(self.path)
        path = parts.path
        query = parse_qs(parts.query, keep_blank_values=True)

        if not self._token_ok(query):
            self._send_403()
            return

        if path in ("/", "/index.html", "/overlay", "/overlay.html"):
            self._serve_overlay()
        elif path == "/events":
            self._serve_events()
        else:
            self._send_simple(
                HTTPStatus.NOT_FOUND,
                b"404 Not Found\n",
                "text/plain; charset=utf-8",
            )

    def _serve_overlay(self) -> None:
        html = self._owner.overlay_html_bytes()
        self._send_simple(
            HTTPStatus.OK,
            html,
            "text/html; charset=utf-8",
            extra_headers={
                "Content-Security-Policy": CSP_POLICY,
                "Referrer-Policy": "no-referrer",
            },
        )

    def _serve_events(self) -> None:
        # Streamed SSE. We chunk via HTTP/1.1 Connection: keep-alive without a
        # Content-Length (text/event-stream is open-ended).
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

        client_q: "Queue[str]" = Queue(maxsize=_CLIENT_QUEUE_MAXSIZE)
        # Atomically register + capture the recent-event backlog so a freshly
        # (re)connected OBS source replays the last few cards instead of showing
        # blank. Each buffered frame is delivered exactly once (see _register_with_replay).
        backlog = self._owner._register_with_replay(client_q)
        try:
            # Prime the stream so EventSource fires `open` promptly.
            self._write_raw(": connected\n\n")
            # Replay the recent ring buffer to THIS new client before going live.
            for frame in backlog:
                if not self._write_raw(frame):
                    return  # client already gone mid-replay — fail quiet
            while not self._owner.is_stopped:
                try:
                    payload = client_q.get(timeout=_SSE_POLL_SECONDS)
                except Empty:
                    # keepalive comment (ignored by EventSource); also detects a
                    # dropped OBS source via the write failing.
                    if not self._write_raw(": keepalive\n\n"):
                        break
                    continue
                if not self._write_raw(payload):
                    break
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # client disconnected — fail quiet
        finally:
            self._owner._unregister(client_q)

    def _write_raw(self, text: str) -> bool:
        try:
            self.wfile.write(text.encode("utf-8"))
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False


_OVERLAY_TOKEN_FILE = Path.home() / ".kenning" / "overlay_token"


def _resolve_overlay_token(configured: str = "") -> str:
    """Return a STABLE overlay token so the OBS Browser-Source URL survives reboots.

    Priority: an explicit ``configured`` token (``twitch.overlay.token``) -> a token
    persisted at ``~/.kenning/overlay_token`` -> a freshly generated one (written back
    so it is reused next boot). Fail-safe: any filesystem error degrades to a fresh
    ephemeral token rather than refusing to start. (Previously the token was a fresh
    ``secrets.token_urlsafe`` every boot, so the OBS source URL had to be re-pasted
    after each restart.)"""
    configured = (configured or "").strip()
    if configured:
        return configured
    try:
        if _OVERLAY_TOKEN_FILE.is_file():
            tok = _OVERLAY_TOKEN_FILE.read_text(encoding="utf-8").strip()
            if tok:
                return tok
        tok = secrets.token_urlsafe(32)
        _OVERLAY_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        _OVERLAY_TOKEN_FILE.write_text(tok, encoding="utf-8")
        logger.info("overlay token persisted to %s (stable across reboots)", _OVERLAY_TOKEN_FILE)
        return tok
    except Exception as e:  # noqa: BLE001 — never block boot on token persistence
        logger.warning("overlay token persistence failed (%s); using an ephemeral token", e)
        return secrets.token_urlsafe(32)


class OverlayServer:
    """Local 127.0.0.1 overlay HTTP/SSE server for the OBS Browser Source.

    Parameters
    ----------
    host:
        Always coerced to a loopback address. Any non-loopback value is rejected
        (defensive: the overlay must never be reachable off-box).
    port:
        ``0`` (default) picks an ephemeral port; read it back via :meth:`url`.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0, token: str = "") -> None:
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise OverlayError(f"overlay must bind loopback only, got {host!r}")
        self._host = "127.0.0.1" if host == "localhost" else host
        self._req_port = int(port)
        self.token = _resolve_overlay_token(token)

        self._httpd: Optional[_OverlayHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()
        self._stopped.set()  # not running yet

        self._clients_lock = threading.Lock()
        self._clients: set["Queue[str]"] = set()
        # Bounded replay buffer of the last vetted frames (guarded by the SAME
        # lock as the client set so register+snapshot is atomic vs. a concurrent
        # emit -> a new client gets each frame exactly once, never doubled/missed).
        self._replay: "deque[str]" = deque(maxlen=_REPLAY_BUFFER_MAXLEN)

        self._html_cache: Optional[bytes] = None

    # --- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        if self._httpd is not None:
            return
        self._stopped.clear()
        # For a FIXED port, reclaim it from any stale holder before the exclusive
        # bind (port 0 = ephemeral, nothing to reclaim). Fail-open.
        if self._req_port != 0:
            try:
                from kenning.subprocess import sidecar_lock
                sidecar_lock.reclaim_port(self._host, self._req_port)
            except Exception as e:  # noqa: BLE001
                logger.debug("overlay port reclaim skipped (%s)", e)
        self._httpd = _OverlayHTTPServer((self._host, self._req_port), _OverlayHandler, self)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="kenning-overlay",
            daemon=True,
        )
        self._thread.start()
        logger.info("overlay server listening on %s (token-gated)", self._bound_addr())

    def stop(self) -> None:
        self._stopped.set()
        # Drain registered clients so blocked SSE loops wake and exit.
        with self._clients_lock:
            clients = list(self._clients)
            self._clients.clear()
        for q in clients:
            try:
                q.put_nowait(": shutdown\n\n")
            except Full:
                pass
        httpd = self._httpd
        if httpd is not None:
            try:
                httpd.shutdown()
            except Exception as e:  # noqa: BLE001 - shutdown must not raise
                logger.debug("overlay shutdown raised: %s", e)
            try:
                httpd.server_close()
            except Exception as e:  # noqa: BLE001
                logger.debug("overlay server_close raised: %s", e)
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._httpd = None
        self._thread = None
        logger.info("overlay server stopped")

    def __enter__(self) -> "OverlayServer":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    @property
    def is_stopped(self) -> bool:
        return self._stopped.is_set()

    # --- addressing -----------------------------------------------------------
    def _bound_addr(self) -> tuple[str, int]:
        if self._httpd is None:
            return (self._host, self._req_port)
        return self._httpd.server_address[:2]  # type: ignore[return-value]

    @property
    def port(self) -> int:
        return int(self._bound_addr()[1])

    def url(self) -> str:
        """The full overlay URL incl. the per-session token (paste into OBS)."""
        host, port = self._bound_addr()
        return f"http://{host}:{port}/?token={self.token}"

    def events_url(self) -> str:
        host, port = self._bound_addr()
        return f"http://{host}:{port}/events?token={self.token}"

    # --- the overlay page -----------------------------------------------------
    def overlay_html_bytes(self) -> bytes:
        if self._html_cache is None:
            try:
                self._html_cache = _HTML_PATH.read_bytes()
            except OSError as e:
                logger.error("overlay.html unreadable at %s: %s", _HTML_PATH, e)
                raise OverlayError(f"overlay.html missing: {e}") from e
        return self._html_cache

    # --- SSE fan-out ----------------------------------------------------------
    def _register(self, q: "Queue[str]") -> None:
        with self._clients_lock:
            self._clients.add(q)

    def _register_with_replay(self, q: "Queue[str]") -> list[str]:
        """Register a new SSE client AND atomically snapshot the replay buffer.

        Done under the single clients lock so a frame emitted concurrently is
        delivered EXACTLY once: either it is already in the returned backlog (the
        client must NOT also receive it live) or it lands in the now-registered
        queue (and is NOT yet in the backlog). The caller writes the backlog first,
        then drains its live queue. Bounded: the backlog is at most
        ``_REPLAY_BUFFER_MAXLEN`` small frames.
        """
        with self._clients_lock:
            backlog = list(self._replay)
            self._clients.add(q)
        return backlog

    def _unregister(self, q: "Queue[str]") -> None:
        with self._clients_lock:
            self._clients.discard(q)

    @property
    def client_count(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    def emit(self, event: dict[str, Any]) -> bool:
        """Schema-validate ``event`` and enqueue it to all connected SSE clients.

        Returns ``True`` if accepted (and fanned out), ``False`` never — an
        invalid event raises :class:`OverlayError` (fail-CLOSED; an unknown type
        never reaches a client). A per-client bounded queue drops its OLDEST event
        on overflow so a stalled OBS source can't wedge the producer.
        """
        vetted = validate_event(event)  # raises OverlayError on bad/unknown
        # Serialize ONCE; json.dumps escapes any '<'/'>'/'&' is NOT default, but
        # the browser renders via textContent so a payload is inert either way.
        # We additionally ensure no newline can break SSE framing by JSON-encoding
        # (json escapes embedded newlines to \n inside the string literal).
        data_line = json.dumps(vetted, ensure_ascii=False, separators=(",", ":"))
        frame = f"event: overlay\ndata: {data_line}\n\n"

        # Buffer for replay AND snapshot the current clients atomically. Appending
        # to the bounded deque BEFORE releasing the lock keeps it consistent with
        # _register_with_replay: a client registering after this point sees this
        # frame in its backlog; clients captured here get it live below. (vetted
        # already had secret/unknown keys stripped by validate_event, so only safe
        # fields are ever retained.)
        with self._clients_lock:
            self._replay.append(frame)
            clients = list(self._clients)
        for q in clients:
            self._offer(q, frame)
        return True

    @staticmethod
    def _offer(q: "Queue[str]", frame: str) -> None:
        # drop-OLDEST: if full, evict one then enqueue. Bounded so a wedged client
        # never causes unbounded memory growth.
        try:
            q.put_nowait(frame)
        except Full:
            try:
                q.get_nowait()
            except Empty:
                pass
            try:
                q.put_nowait(frame)
            except Full:
                pass  # racing consumer refilled it; drop this frame, fail quiet
