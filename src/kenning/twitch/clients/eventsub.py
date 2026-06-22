"""S3 — Twitch EventSub read transport (pure stdlib, receive-only).

A minimal, hand-rolled RFC 6455 WebSocket CLIENT (no third-party ``websockets``)
plus the EventSub session/notification parsing the Twitch chat sidecar needs to
consume the ``wss`` event stream. Everything here is import-safe for the
anticheat-pinned voice process (BR-P1): the only imports are
``socket``/``ssl``/``base64``/``hashlib``/``struct``/``json``/``os``/``logging``/
``dataclasses`` + stdlib ``collections``/``re``.

Design notes
------------
* RECEIVE-ONLY. :class:`RFC6455Client` performs the upgrade handshake, decodes
  server frames, auto-replies PONG to a server PING, and answers a server CLOSE,
  but it NEVER sends an application data frame. Twitch EventSub clients receive
  notifications only; subscriptions are created out-of-band over Helix.
* PURE codec. ``_build_frame`` / ``_parse_frame`` / ``compute_accept`` are pure
  functions of their inputs so the unit tests drive them with synthetic bytes via
  a :class:`FakeSocket`. No real network, no creds, no models.
* Fail-safe. Every external/parse path logs structured context and fails closed —
  a malformed notification yields ``None`` / a benign default rather than raising
  into the sidecar's receive loop.

Transport choice (S_report): EventSub WebSocket (wss), never webhooks — a
localhost sidecar has no public listener. The full mod EventSub set
(``automod.message.hold``, ``channel.suspicious_user.message``, ``channel.moderate``
v2, ...) rides the same wss session and is classified by
:meth:`EventSubSession.classify_message`.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import socket
import ssl
import struct
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "WebSocketError",
    "WebSocketClosed",
    "HandshakeError",
    "RFC6455Client",
    "FakeSocket",
    "compute_accept",
    "ChatEvent",
    "DedupLRU",
    "EventSubSession",
]

logger = logging.getLogger("kenning.twitch.clients.eventsub")

# RFC 6455 §1.3 — the magic GUID appended to the client key before the SHA-1.
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# Opcodes (RFC 6455 §5.2).
OPCODE_CONT = 0x0
OPCODE_TEXT = 0x1
OPCODE_BINARY = 0x2
OPCODE_CLOSE = 0x8
OPCODE_PING = 0x9
OPCODE_PONG = 0xA

# Default Twitch EventSub wss endpoint (overridable for reconnect / mock).
DEFAULT_EVENTSUB_URL = "wss://eventsub.wss.twitch.tv/ws"

# Guard rails: a single control payload is capped at 125 bytes by the RFC; a data
# frame we are willing to reassemble is capped so a hostile server can't make us
# allocate unbounded memory (Twitch notifications are a few KB at most).
_MAX_CONTROL_PAYLOAD = 125
_MAX_MESSAGE_BYTES = 1 << 20  # 1 MiB hard ceiling for one reassembled message
_RECV_CHUNK = 4096


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class WebSocketError(Exception):
    """Base class for transport-level WebSocket faults."""


class HandshakeError(WebSocketError):
    """The HTTP Upgrade handshake failed or the accept token did not verify."""


class WebSocketClosed(WebSocketError):
    """The peer sent a CLOSE frame (or the socket reached EOF).

    ``code`` is the RFC 6455 close code (``1006`` = abnormal/EOF when the peer
    gave none); ``reason`` is the optional UTF-8 reason text.
    """

    def __init__(self, code: int = 1006, reason: str = "") -> None:
        super().__init__(f"websocket closed: code={code} reason={reason!r}")
        self.code = code
        self.reason = reason


# --------------------------------------------------------------------------- #
# Pure handshake helper
# --------------------------------------------------------------------------- #
def compute_accept(sec_websocket_key: str) -> str:
    """Return the ``Sec-WebSocket-Accept`` value for a client key (RFC 6455 §1.3).

    ``accept = base64(sha1(key + GUID))``. Pure + deterministic so the test can
    assert the canonical RFC example
    (``dGhlIHNhbXBsZSBub25jZQ==`` -> ``s3pPLMBiTxaQ9kYGzzhZRbK+xOo=``).
    """
    digest = hashlib.sha1((sec_websocket_key + _WS_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def _parse_url(url: str) -> tuple[bool, str, int, str]:
    """Split a ws(s) URL into (is_tls, host, port, path). Fail-safe, never raises."""
    m = re.match(r"^(wss|ws)://([^/:]+)(?::(\d+))?(/.*)?$", url.strip(), re.IGNORECASE)
    if not m:
        raise HandshakeError(f"unparseable websocket url: {url!r}")
    scheme, host, port_s, path = m.group(1).lower(), m.group(2), m.group(3), m.group(4)
    is_tls = scheme == "wss"
    port = int(port_s) if port_s else (443 if is_tls else 80)
    return is_tls, host, port, path or "/"


# --------------------------------------------------------------------------- #
# Test transport
# --------------------------------------------------------------------------- #
class FakeSocket:
    """An in-memory duplex socket for unit tests (and offline replay).

    ``inbound`` is the byte stream the "server" has queued for the client to read;
    ``sent`` accumulates every byte the client writes. ``sendall``/``recv`` mirror
    the ``socket.socket`` surface the client uses, so no real network is touched.
    """

    def __init__(self, inbound: bytes = b"") -> None:
        self._inbound = bytearray(inbound)
        self._pos = 0
        self.sent = bytearray()
        self.closed = False
        self.timeout: Optional[float] = None

    # -- server-side test helpers -------------------------------------------- #
    def feed(self, data: bytes) -> None:
        """Append more bytes for the client to read (e.g. a follow-up frame)."""
        self._inbound.extend(data)

    # -- socket.socket surface ----------------------------------------------- #
    def recv(self, bufsize: int) -> bytes:
        if self._pos >= len(self._inbound):
            return b""  # EOF — mirrors a closed peer
        chunk = bytes(self._inbound[self._pos : self._pos + bufsize])
        self._pos += len(chunk)
        return chunk

    def sendall(self, data: bytes) -> None:
        if self.closed:
            raise OSError("send on closed FakeSocket")
        self.sent.extend(data)

    def settimeout(self, value: Optional[float]) -> None:
        self.timeout = value

    def close(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------- #
# RFC 6455 client
# --------------------------------------------------------------------------- #
class RFC6455Client:
    """A minimal receive-only RFC 6455 WebSocket client over ``socket`` + ``ssl``.

    The frame codec and the handshake-accept check are pure; I/O is funnelled
    through a tiny ``_sock`` object that only needs ``recv``/``sendall``/``close``/
    ``settimeout`` — so :class:`FakeSocket` can stand in for a real TLS socket in
    tests. The client masks every frame it sends (RFC §5.3 requires it) and only
    ever sends control frames (PONG / CLOSE) — never application data.
    """

    def __init__(
        self,
        sock: Any = None,
        *,
        timeout: float = 30.0,
        max_message_bytes: int = _MAX_MESSAGE_BYTES,
    ) -> None:
        self._sock = sock
        self._timeout = timeout
        self._max_message_bytes = max_message_bytes
        self._recv_buf = bytearray()  # undecoded bytes carried between recv() calls
        self._closed = False
        self.connected = sock is not None

    # ------------------------------------------------------------------ #
    # Connection / handshake
    # ------------------------------------------------------------------ #
    def connect(self, url: str, *, sock_factory: Any = None) -> None:
        """Open a TLS socket to ``url`` and perform the HTTP Upgrade handshake.

        ``sock_factory(host, port, is_tls, timeout) -> sock`` is injectable so a
        test can supply a :class:`FakeSocket` preloaded with the server's 101
        response; in production it defaults to a real ``ssl``-wrapped TCP socket.
        Raises :class:`HandshakeError` on any non-101 / bad-accept response.
        """
        is_tls, host, port, path = _parse_url(url)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        expected_accept = compute_accept(key)

        if sock_factory is not None:
            sock = sock_factory(host, port, is_tls, self._timeout)
        else:
            sock = self._default_connect(host, port, is_tls)
        self._sock = sock
        self.connected = True

        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        ).encode("ascii")
        try:
            sock.sendall(request)
        except OSError as exc:
            logger.error("eventsub handshake send failed: %s host=%s", exc, host)
            raise HandshakeError(f"handshake send failed: {exc}") from exc

        header_bytes = self._read_until_headers(sock)
        self._verify_handshake(header_bytes, expected_accept, host)
        logger.info("eventsub websocket connected host=%s path=%s", host, path)

    def _default_connect(self, host: str, port: int, is_tls: bool) -> Any:
        try:
            raw = socket.create_connection((host, port), timeout=self._timeout)
        except OSError as exc:
            logger.error("eventsub tcp connect failed: %s host=%s port=%s", exc, host, port)
            raise HandshakeError(f"tcp connect failed: {exc}") from exc
        if not is_tls:
            return raw
        try:
            ctx = ssl.create_default_context()
            return ctx.wrap_socket(raw, server_hostname=host)
        except (ssl.SSLError, OSError) as exc:
            try:
                raw.close()
            except OSError:
                pass
            logger.error("eventsub tls wrap failed: %s host=%s", exc, host)
            raise HandshakeError(f"tls wrap failed: {exc}") from exc

    def _read_until_headers(self, sock: Any) -> bytes:
        """Read bytes until the end-of-headers CRLFCRLF; carry any frame overflow."""
        buf = bytearray()
        while b"\r\n\r\n" not in buf:
            try:
                chunk = sock.recv(_RECV_CHUNK)
            except OSError as exc:
                raise HandshakeError(f"handshake recv failed: {exc}") from exc
            if not chunk:
                raise HandshakeError("connection closed during handshake")
            buf.extend(chunk)
            if len(buf) > 1 << 16:  # 64 KiB of headers is pathological
                raise HandshakeError("handshake header too large")
        head, _, rest = bytes(buf).partition(b"\r\n\r\n")
        # Any bytes after the headers are the start of the frame stream.
        self._recv_buf.extend(rest)
        return head

    def _verify_handshake(self, header_bytes: bytes, expected_accept: str, host: str) -> None:
        try:
            text = header_bytes.decode("iso-8859-1")
        except Exception as exc:  # noqa: BLE001
            raise HandshakeError(f"undecodable handshake response: {exc}") from exc
        lines = text.split("\r\n")
        status = lines[0] if lines else ""
        if "101" not in status:
            logger.error("eventsub handshake non-101 status=%r host=%s", status, host)
            raise HandshakeError(f"expected 101 Switching Protocols, got: {status!r}")
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if ":" in line:
                name, _, value = line.partition(":")
                headers[name.strip().lower()] = value.strip()
        if headers.get("upgrade", "").lower() != "websocket":
            raise HandshakeError("missing/invalid Upgrade header in handshake")
        accept = headers.get("sec-websocket-accept", "")
        if accept != expected_accept:
            logger.error(
                "eventsub handshake accept mismatch got=%r want=%r host=%s",
                accept, expected_accept, host,
            )
            raise HandshakeError("Sec-WebSocket-Accept mismatch")

    # ------------------------------------------------------------------ #
    # Pure frame codec
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_frame(opcode: int, payload: bytes = b"", *, mask: bool = True) -> bytes:
        """Encode a single FIN frame. Client frames MUST be masked (RFC §5.3).

        Control frames (close/ping/pong) carry <=125 bytes; we enforce that so a
        caller can never emit an illegal oversized control frame.
        """
        if opcode in (OPCODE_CLOSE, OPCODE_PING, OPCODE_PONG) and len(payload) > _MAX_CONTROL_PAYLOAD:
            raise WebSocketError(f"control frame payload {len(payload)} exceeds 125 bytes")
        b0 = 0x80 | (opcode & 0x0F)  # FIN=1
        length = len(payload)
        header = bytearray([b0])
        mask_bit = 0x80 if mask else 0x00
        if length < 126:
            header.append(mask_bit | length)
        elif length <= 0xFFFF:
            header.append(mask_bit | 126)
            header.extend(struct.pack(">H", length))
        else:
            header.append(mask_bit | 127)
            header.extend(struct.pack(">Q", length))
        if mask:
            masking_key = os.urandom(4)
            header.extend(masking_key)
            masked = bytes(b ^ masking_key[i & 3] for i, b in enumerate(payload))
            return bytes(header) + masked
        return bytes(header) + payload

    @staticmethod
    def _parse_frame(data: bytes) -> Optional[tuple[int, bool, bytes, int]]:
        """Decode ONE frame from the front of ``data``.

        Returns ``(opcode, fin, payload, consumed)`` or ``None`` if ``data`` does
        not yet hold a complete frame (the caller should read more bytes). Handles
        the 7-bit / 16-bit (126) / 64-bit (127) length forms and unmasks a masked
        frame (servers normally do not mask, but we honour the bit if set).
        """
        if len(data) < 2:
            return None
        b0, b1 = data[0], data[1]
        fin = bool(b0 & 0x80)
        opcode = b0 & 0x0F
        masked = bool(b1 & 0x80)
        length = b1 & 0x7F
        idx = 2
        if length == 126:
            if len(data) < idx + 2:
                return None
            length = struct.unpack(">H", data[idx : idx + 2])[0]
            idx += 2
        elif length == 127:
            if len(data) < idx + 8:
                return None
            length = struct.unpack(">Q", data[idx : idx + 8])[0]
            idx += 8
        mask_key = b""
        if masked:
            if len(data) < idx + 4:
                return None
            mask_key = data[idx : idx + 4]
            idx += 4
        if len(data) < idx + length:
            return None
        payload = bytes(data[idx : idx + length])
        if masked:
            payload = bytes(b ^ mask_key[i & 3] for i, b in enumerate(payload))
        consumed = idx + length
        return opcode, fin, payload, consumed

    # ------------------------------------------------------------------ #
    # Receive loop
    # ------------------------------------------------------------------ #
    def _next_frame(self) -> tuple[int, bool, bytes]:
        """Pull the next complete frame off the wire (blocking on more bytes)."""
        while True:
            parsed = self._parse_frame(bytes(self._recv_buf))
            if parsed is not None:
                opcode, fin, payload, consumed = parsed
                del self._recv_buf[:consumed]
                return opcode, fin, payload
            try:
                chunk = self._sock.recv(_RECV_CHUNK)
            except OSError as exc:
                logger.warning("eventsub recv error: %s", exc)
                raise WebSocketClosed(1006, f"recv error: {exc}") from exc
            if not chunk:
                raise WebSocketClosed(1006, "peer closed (eof)")
            self._recv_buf.extend(chunk)
            if len(self._recv_buf) > self._max_message_bytes + 16:
                raise WebSocketError("incoming frame exceeds max message size")

    def recv_json(self) -> dict:
        """Return the next TEXT message decoded as a JSON object.

        Convenience over :meth:`recv` for the EventSub stream, every message of
        which is a JSON envelope. Fail-safe: a non-object or undecodable body
        yields ``{}`` (logged) so the caller's classify step sees ``"unknown"``
        rather than raising into the receive loop.
        """
        text = self.recv()
        try:
            obj = json.loads(text)
        except (ValueError, TypeError) as exc:
            logger.warning("eventsub non-json message dropped: %s", exc)
            return {}
        return obj if isinstance(obj, dict) else {}

    def recv(self) -> str:
        """Return the next reassembled TEXT message as ``str``.

        Transparently: reassembles a fragmented (CONT) text message; auto-replies
        PONG to a server PING; ignores a stray PONG; raises :class:`WebSocketClosed`
        on a server CLOSE (after echoing the close). Binary frames are decoded as
        UTF-8 with replacement (EventSub is text-only, but we never crash on one).
        """
        fragments: list[bytes] = []
        frag_opcode: Optional[int] = None
        total = 0
        while True:
            opcode, fin, payload = self._next_frame()

            if opcode == OPCODE_PING:
                self.send_pong(payload)
                continue
            if opcode == OPCODE_PONG:
                continue  # unsolicited / keepalive pong — nothing to do
            if opcode == OPCODE_CLOSE:
                code, reason = self._parse_close_payload(payload)
                self._handle_server_close(code, reason)
                raise WebSocketClosed(code, reason)

            if opcode in (OPCODE_TEXT, OPCODE_BINARY):
                frag_opcode = opcode
                fragments = [payload]
                total = len(payload)
            elif opcode == OPCODE_CONT:
                if frag_opcode is None:
                    logger.warning("eventsub continuation without start frame — dropping")
                    fragments = []
                    continue
                fragments.append(payload)
                total += len(payload)
            else:
                logger.warning("eventsub unknown opcode 0x%x — dropping", opcode)
                continue

            if total > self._max_message_bytes:
                raise WebSocketError("reassembled message exceeds max message size")
            if fin:
                data = b"".join(fragments)
                fragments = []
                frag_opcode = None
                return data.decode("utf-8", "replace")

    @staticmethod
    def _parse_close_payload(payload: bytes) -> tuple[int, str]:
        if len(payload) >= 2:
            code = struct.unpack(">H", payload[:2])[0]
            reason = payload[2:].decode("utf-8", "replace")
            return code, reason
        return 1005, ""  # no status code present

    def _handle_server_close(self, code: int, reason: str) -> None:
        logger.info("eventsub server close code=%s reason=%r", code, reason)
        if not self._closed:
            try:
                self._sock.sendall(self._build_frame(OPCODE_CLOSE, struct.pack(">H", code)))
            except (OSError, WebSocketError) as exc:
                logger.debug("eventsub close echo failed: %s", exc)
            self._closed = True

    # ------------------------------------------------------------------ #
    # Control sends (the ONLY frames we ever emit)
    # ------------------------------------------------------------------ #
    def send_pong(self, payload: bytes = b"") -> None:
        """Reply to a server PING. Payload is echoed (capped to 125 bytes)."""
        if self._closed:
            return
        try:
            frame = self._build_frame(OPCODE_PONG, payload[:_MAX_CONTROL_PAYLOAD])
            self._sock.sendall(frame)
        except (OSError, WebSocketError) as exc:
            logger.warning("eventsub pong send failed: %s", exc)
            raise WebSocketClosed(1006, f"pong send failed: {exc}") from exc

    def close(self, code: int = 1000, reason: str = "") -> None:
        """Send a CLOSE frame and shut the socket. Idempotent + fail-safe."""
        if self._closed:
            return
        self._closed = True
        self.connected = False
        try:
            payload = struct.pack(">H", code) + reason.encode("utf-8")[:_MAX_CONTROL_PAYLOAD - 2]
            self._sock.sendall(self._build_frame(OPCODE_CLOSE, payload))
        except (OSError, WebSocketError, AttributeError) as exc:
            logger.debug("eventsub close send failed: %s", exc)
        finally:
            try:
                if self._sock is not None:
                    self._sock.close()
            except OSError as exc:
                logger.debug("eventsub socket close failed: %s", exc)


# --------------------------------------------------------------------------- #
# EventSub notification -> ChatEvent
# --------------------------------------------------------------------------- #
@dataclass
class ChatEvent:
    """A parsed ``channel.chat.message`` notification (the to-Ultron unit).

    ``raw`` keeps the full event dict so a downstream safety layer can inspect
    anything not surfaced as a typed field (provenance taint). Lists default to
    empty so a partial/garbage payload can never produce ``None`` iterables.
    """

    broadcaster_user_id: str
    chatter_user_id: str
    chatter_login: str
    chatter_name: str
    text: str
    fragments: list[dict] = field(default_factory=list)
    badges: list[dict] = field(default_factory=list)
    reply_parent_user_id: Optional[str] = None
    cheer_bits: int = 0
    message_id: str = ""
    message_type: str = "text"
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_eventsub(cls, payload_dict: dict) -> Optional[ChatEvent]:
        """Parse a ``channel.chat.message`` EventSub notification into a ChatEvent.

        Accepts either the full notification envelope
        (``{"metadata":..., "payload":{"event": {...}}}``), a bare
        ``{"event": {...}}`` payload, or the bare ``event`` dict itself. Returns
        ``None`` (logged) on a non-chat-message or structurally invalid input —
        fail-safe so the receive loop never raises on hostile data.
        """
        try:
            event = cls._locate_event(payload_dict)
            if event is None:
                return None

            message = event.get("message")
            if not isinstance(message, dict):
                message = {}
            text = message.get("text")
            if not isinstance(text, str):
                text = ""
            fragments = message.get("fragments")
            if not isinstance(fragments, list):
                fragments = []

            badges = event.get("badges")
            if not isinstance(badges, list):
                badges = []

            reply = event.get("reply")
            reply_parent_user_id = None
            if isinstance(reply, dict):
                rp = reply.get("parent_user_id")
                if isinstance(rp, str) and rp:
                    reply_parent_user_id = rp

            cheer = event.get("cheer")
            cheer_bits = 0
            if isinstance(cheer, dict):
                cheer_bits = cls._coerce_int(cheer.get("bits"))

            return cls(
                broadcaster_user_id=cls._coerce_str(event.get("broadcaster_user_id")),
                chatter_user_id=cls._coerce_str(event.get("chatter_user_id")),
                chatter_login=cls._coerce_str(event.get("chatter_user_login")),
                chatter_name=cls._coerce_str(event.get("chatter_user_name")),
                text=text,
                fragments=fragments,
                badges=badges,
                reply_parent_user_id=reply_parent_user_id,
                cheer_bits=cheer_bits,
                message_id=cls._coerce_str(event.get("message_id")),
                message_type=cls._coerce_str(message.get("message_type")) or "text",
                raw=event,
            )
        except Exception as exc:  # noqa: BLE001 — never raise into the receive loop
            logger.warning("eventsub chat-message parse failed: %s", exc)
            return None

    # -- helpers -------------------------------------------------------- #
    @staticmethod
    def _locate_event(payload_dict: Any) -> Optional[dict]:
        if not isinstance(payload_dict, dict):
            return None
        # Full envelope: verify the subscription type before trusting it.
        meta = payload_dict.get("metadata")
        if isinstance(meta, dict):
            sub_type = meta.get("subscription_type")
            if isinstance(sub_type, str) and sub_type != "channel.chat.message":
                return None
        payload = payload_dict.get("payload")
        if isinstance(payload, dict):
            sub = payload.get("subscription")
            if isinstance(sub, dict):
                stype = sub.get("type")
                if isinstance(stype, str) and stype != "channel.chat.message":
                    return None
            event = payload.get("event")
            if isinstance(event, dict):
                return event
        # Bare {"event": {...}}.
        event = payload_dict.get("event")
        if isinstance(event, dict):
            return event
        # Bare event dict — accept only if it looks like a chat message.
        if "chatter_user_id" in payload_dict or "message" in payload_dict:
            return payload_dict
        return None

    @staticmethod
    def _coerce_str(value: Any) -> str:
        return value if isinstance(value, str) else ("" if value is None else str(value))

    @staticmethod
    def _coerce_int(value: Any) -> int:
        try:
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str) and value.strip():
                return int(value.strip())
        except (TypeError, ValueError):
            return 0
        return 0


# --------------------------------------------------------------------------- #
# At-least-once dedup
# --------------------------------------------------------------------------- #
class DedupLRU:
    """Bounded LRU set of seen ``message_id`` values for at-least-once delivery.

    EventSub may redeliver a notification; the sidecar must process each message
    exactly once. :meth:`seen` returns ``True`` if the id was already recorded
    (a duplicate to drop) and otherwise records it, evicting the oldest entry once
    ``maxsize`` is exceeded. An empty / ``None`` id is treated as never-seen so a
    malformed notification is not silently coalesced with another.
    """

    def __init__(self, maxsize: int = 4096) -> None:
        if maxsize < 1:
            raise ValueError("DedupLRU maxsize must be >= 1")
        self._maxsize = maxsize
        self._seen: OrderedDict[str, None] = OrderedDict()

    def seen(self, message_id: Optional[str]) -> bool:
        if not message_id:
            return False  # cannot dedup an absent id — let it through, fail-open
        if message_id in self._seen:
            self._seen.move_to_end(message_id)
            return True
        self._seen[message_id] = None
        if len(self._seen) > self._maxsize:
            self._seen.popitem(last=False)  # evict oldest
        return False

    def __len__(self) -> int:
        return len(self._seen)

    def __contains__(self, message_id: object) -> bool:
        return isinstance(message_id, str) and message_id in self._seen


# --------------------------------------------------------------------------- #
# EventSub session bookkeeping
# --------------------------------------------------------------------------- #
@dataclass
class EventSubSession:
    """Tracks the EventSub session lifecycle off the control messages.

    Twitch sends a ``session_welcome`` (carrying the ``session_id`` to attach Helix
    subscriptions to), periodic ``session_keepalive`` heartbeats (silence beyond
    ``keepalive_timeout_seconds`` means the connection is dead — reconnect), and a
    ``session_reconnect`` carrying a new ``reconnect_url`` to migrate to without
    dropping events. This class classifies each control/notification message and
    tracks staleness; it holds no socket.
    """

    session_id: Optional[str] = None
    last_keepalive: Optional[float] = None
    keepalive_timeout_seconds: float = 10.0

    # -- classification ------------------------------------------------- #
    @staticmethod
    def message_type_of(msg: Any) -> str:
        """Return the EventSub ``metadata.message_type`` (``""`` if absent)."""
        if isinstance(msg, dict):
            meta = msg.get("metadata")
            if isinstance(meta, dict):
                mt = meta.get("message_type")
                if isinstance(mt, str):
                    return mt
        return ""

    def classify_message(self, msg: Any) -> str:
        """Map a raw EventSub message to one of the five logical classes.

        Returns one of ``{welcome, keepalive, notification, reconnect,
        revocation}``. An unrecognised message_type is reported as
        ``notification`` only if it carries a payload event, else ``keepalive`` is
        NOT assumed — it falls through to ``"unknown"`` which the caller logs and
        ignores (fail-safe; we never act on a message we can't classify).
        """
        mt = self.message_type_of(msg)
        mapping = {
            "session_welcome": "welcome",
            "session_keepalive": "keepalive",
            "notification": "notification",
            "session_reconnect": "reconnect",
            "revocation": "revocation",
        }
        return mapping.get(mt, "unknown")

    # -- lifecycle ------------------------------------------------------ #
    def parse_welcome(self, msg: Any) -> Optional[str]:
        """Extract + store the ``session_id`` from a ``session_welcome`` message.

        Also adopts the server-advertised ``keepalive_timeout_seconds`` and primes
        the keepalive clock from ``metadata.message_timestamp`` when present (so an
        immediate :meth:`is_stale` check after welcome is not falsely stale).
        Returns the session id, or ``None`` if the message is not a valid welcome.
        """
        if self.classify_message(msg) != "welcome" or not isinstance(msg, dict):
            return None
        payload = msg.get("payload")
        session = payload.get("session") if isinstance(payload, dict) else None
        if not isinstance(session, dict):
            logger.warning("eventsub welcome missing session object")
            return None
        sid = session.get("id")
        if not isinstance(sid, str) or not sid:
            logger.warning("eventsub welcome missing session id")
            return None
        self.session_id = sid
        timeout = session.get("keepalive_timeout_seconds")
        if isinstance(timeout, (int, float)) and timeout > 0:
            self.keepalive_timeout_seconds = float(timeout)
        logger.info(
            "eventsub session welcome id=%s keepalive_timeout=%.0fs",
            sid, self.keepalive_timeout_seconds,
        )
        return sid

    def note_keepalive(self, now: Optional[float] = None) -> None:
        """Record that a keepalive (or any traffic) arrived at ``now``.

        Any received message resets the staleness clock — keepalives, welcome, and
        notifications all count as liveness (a busy channel may never send a bare
        keepalive). ``now`` defaults to the monotonic clock.
        """
        import time

        self.last_keepalive = time.monotonic() if now is None else now

    def is_stale(self, now: Optional[float] = None, timeout: Optional[float] = None) -> bool:
        """True if no traffic has arrived within the keepalive window.

        Twitch guarantees a keepalive every ``keepalive_timeout_seconds`` of
        silence; missing it past the window means the connection is dead and the
        sidecar must reconnect. If we have never seen traffic yet, we are NOT stale
        (the welcome handshake is still in flight).
        """
        if self.last_keepalive is None:
            return False
        import time

        current = time.monotonic() if now is None else now
        window = self.keepalive_timeout_seconds if timeout is None else timeout
        # A small grace multiplier absorbs scheduling jitter on the heartbeat.
        return (current - self.last_keepalive) > window

    def handle_reconnect(self, msg: Any) -> Optional[str]:
        """Return the new ``reconnect_url`` from a ``session_reconnect`` message.

        Per Twitch's protocol the old socket stays open until the new one receives
        its own welcome; this method only surfaces the URL the caller dials.
        Returns ``None`` if the message is not a valid reconnect.
        """
        if self.classify_message(msg) != "reconnect" or not isinstance(msg, dict):
            return None
        payload = msg.get("payload")
        session = payload.get("session") if isinstance(payload, dict) else None
        if not isinstance(session, dict):
            logger.warning("eventsub reconnect missing session object")
            return None
        url = session.get("reconnect_url")
        if not isinstance(url, str) or not url:
            logger.warning("eventsub reconnect missing reconnect_url")
            return None
        logger.info("eventsub session reconnect -> %s", url)
        return url
