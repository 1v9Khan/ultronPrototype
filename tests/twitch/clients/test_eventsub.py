"""S3 — tests for the Twitch EventSub read transport (offline, no real network).

Drives the pure frame codec + handshake-accept + EventSub session/notification
parsing entirely through synthetic bytes and a :class:`FakeSocket`. No sockets,
no creds, no models.
"""
from __future__ import annotations

import json
import struct

import pytest

from kenning.twitch.clients.eventsub import (
    OPCODE_BINARY,
    OPCODE_CLOSE,
    OPCODE_CONT,
    OPCODE_PING,
    OPCODE_PONG,
    OPCODE_TEXT,
    ChatEvent,
    DedupLRU,
    EventSubSession,
    FakeSocket,
    HandshakeError,
    RFC6455Client,
    WebSocketClosed,
    WebSocketError,
    compute_accept,
)


# --------------------------------------------------------------------------- #
# Server-side frame builders (UNMASKED — servers never mask, RFC §5.1)
# --------------------------------------------------------------------------- #
def server_frame(opcode: int, payload: bytes = b"", *, fin: bool = True) -> bytes:
    """Build a server->client frame (unmasked) for feeding into FakeSocket."""
    b0 = (0x80 if fin else 0x00) | (opcode & 0x0F)
    out = bytearray([b0])
    length = len(payload)
    if length < 126:
        out.append(length)
    elif length <= 0xFFFF:
        out.append(126)
        out.extend(struct.pack(">H", length))
    else:
        out.append(127)
        out.extend(struct.pack(">Q", length))
    out.extend(payload)
    return bytes(out)


def text_frame(s: str, *, fin: bool = True, opcode: int = OPCODE_TEXT) -> bytes:
    return server_frame(opcode, s.encode("utf-8"), fin=fin)


# --------------------------------------------------------------------------- #
# Handshake accept — the canonical RFC 6455 §1.3 example
# --------------------------------------------------------------------------- #
def test_compute_accept_matches_rfc6455_example():
    # The exact vector from RFC 6455 §1.3.
    assert compute_accept("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_handshake_success_via_fake_socket():
    # Pre-load the FakeSocket with a valid 101 response, then connect with a
    # sock_factory so the client uses our fake. The client computes its own random
    # key; we intercept it to craft a correct accept.
    captured: dict = {}

    def factory(host, port, is_tls, timeout):
        # We don't know the key yet (urandom inside connect), so we feed a 101 with
        # a placeholder and patch the accept after observing the request. Instead,
        # we monkeypatch by reading the key the client just sent — but recv happens
        # after sendall, so feed lazily via a subclass.
        return _HandshakeFake(captured)

    client = RFC6455Client(timeout=5.0)
    client.connect("wss://eventsub.wss.twitch.tv/ws", sock_factory=factory)
    assert client.connected is True
    # The client must have sent a well-formed upgrade request.
    req = bytes(client._sock.sent)
    assert req.startswith(b"GET /ws HTTP/1.1\r\n")
    assert b"Upgrade: websocket\r\n" in req
    assert b"Sec-WebSocket-Key: " in req
    assert b"Sec-WebSocket-Version: 13\r\n" in req


class _HandshakeFake(FakeSocket):
    """A FakeSocket that synthesises a correct 101 accept from the client's key."""

    def __init__(self, captured: dict):
        super().__init__()
        self._captured = captured
        self._responded = False

    def sendall(self, data: bytes) -> None:  # noqa: D401
        super().sendall(data)
        text = bytes(self.sent).decode("iso-8859-1")
        if "\r\n\r\n" in text and not self._responded:
            # Extract the client's Sec-WebSocket-Key and craft the matching accept.
            key = ""
            for line in text.split("\r\n"):
                if line.lower().startswith("sec-websocket-key:"):
                    key = line.split(":", 1)[1].strip()
            accept = compute_accept(key)
            self._captured["key"] = key
            resp = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n"
                "\r\n"
            ).encode("iso-8859-1")
            self.feed(resp)
            self._responded = True


def test_handshake_rejects_bad_accept():
    def factory(host, port, is_tls, timeout):
        sock = FakeSocket()
        sock.feed(
            b"HTTP/1.1 101 Switching Protocols\r\n"
            b"Upgrade: websocket\r\n"
            b"Sec-WebSocket-Accept: WRONGWRONGWRONG=\r\n"
            b"\r\n"
        )
        return sock

    client = RFC6455Client()
    with pytest.raises(HandshakeError):
        client.connect("wss://eventsub.wss.twitch.tv/ws", sock_factory=factory)


def test_handshake_rejects_non_101():
    def factory(host, port, is_tls, timeout):
        sock = FakeSocket()
        sock.feed(b"HTTP/1.1 400 Bad Request\r\n\r\n")
        return sock

    client = RFC6455Client()
    with pytest.raises(HandshakeError):
        client.connect("wss://eventsub.wss.twitch.tv/ws", sock_factory=factory)


# --------------------------------------------------------------------------- #
# Frame codec
# --------------------------------------------------------------------------- #
def test_build_parse_round_trip_text():
    payload = b"hello eventsub"
    frame = RFC6455Client._build_frame(OPCODE_TEXT, payload, mask=True)
    # Client frame MUST be masked.
    assert frame[1] & 0x80, "client frame must set the mask bit"
    parsed = RFC6455Client._parse_frame(frame)
    assert parsed is not None
    opcode, fin, out, consumed = parsed
    assert opcode == OPCODE_TEXT
    assert fin is True
    assert out == payload
    assert consumed == len(frame)


def test_masked_client_frame_unmasks_correctly():
    # Round-trip a payload containing every byte value to exercise the XOR mask.
    payload = bytes(range(256)) * 2  # 512 bytes -> 16-bit extended length path
    frame = RFC6455Client._build_frame(OPCODE_BINARY, payload, mask=True)
    assert frame[1] & 0x7F == 126  # used the 16-bit length form
    opcode, fin, out, _ = RFC6455Client._parse_frame(frame)
    assert opcode == OPCODE_BINARY
    assert out == payload


def test_parse_frame_64bit_length_form():
    # >0xFFFF bytes forces the 64-bit (127) length form.
    payload = b"x" * (0xFFFF + 5)
    frame = RFC6455Client._build_frame(OPCODE_BINARY, payload, mask=False)
    assert frame[1] & 0x7F == 127
    opcode, fin, out, _ = RFC6455Client._parse_frame(frame)
    assert out == payload


def test_parse_frame_incomplete_returns_none():
    frame = RFC6455Client._build_frame(OPCODE_TEXT, b"partial", mask=True)
    assert RFC6455Client._parse_frame(frame[:1]) is None  # not even the 2-byte head
    assert RFC6455Client._parse_frame(frame[:4]) is None  # head ok, payload short


def test_build_frame_rejects_oversized_control_payload():
    with pytest.raises(WebSocketError):
        RFC6455Client._build_frame(OPCODE_PING, b"x" * 126, mask=True)


# --------------------------------------------------------------------------- #
# recv(): reassembly, ping/pong, close
# --------------------------------------------------------------------------- #
def test_recv_single_text_message():
    sock = FakeSocket(text_frame("notification body"))
    client = RFC6455Client(sock=sock)
    assert client.recv() == "notification body"


def test_recv_fragmented_text_reassembly():
    # "Hello" split into TEXT(fin=0) + CONT(fin=0) + CONT(fin=1).
    frames = (
        text_frame("Hel", fin=False)
        + server_frame(OPCODE_CONT, b"lo ", fin=False)
        + server_frame(OPCODE_CONT, b"world", fin=True)
    )
    sock = FakeSocket(frames)
    client = RFC6455Client(sock=sock)
    assert client.recv() == "Hello world"


def test_ping_triggers_auto_pong():
    # A PING (with payload) arrives, then a TEXT message. recv() should auto-reply
    # PONG (echoing the payload) and return the text.
    ping_payload = b"are-you-there"
    sock = FakeSocket(
        server_frame(OPCODE_PING, ping_payload) + text_frame("pong-then-data")
    )
    client = RFC6455Client(sock=sock)
    msg = client.recv()
    assert msg == "pong-then-data"
    # Assert a PONG frame was actually written to the fake socket.
    sent = bytes(sock.sent)
    parsed = RFC6455Client._parse_frame(sent)
    assert parsed is not None
    opcode, fin, payload, _ = parsed
    assert opcode == OPCODE_PONG
    assert payload == ping_payload


def test_unsolicited_pong_is_ignored():
    sock = FakeSocket(server_frame(OPCODE_PONG, b"keepalive") + text_frame("data"))
    client = RFC6455Client(sock=sock)
    assert client.recv() == "data"


def test_recv_close_raises_websocket_closed():
    close_payload = struct.pack(">H", 1000) + b"bye"
    sock = FakeSocket(server_frame(OPCODE_CLOSE, close_payload))
    client = RFC6455Client(sock=sock)
    with pytest.raises(WebSocketClosed) as ei:
        client.recv()
    assert ei.value.code == 1000
    assert ei.value.reason == "bye"
    # The client must echo a CLOSE frame back.
    parsed = RFC6455Client._parse_frame(bytes(sock.sent))
    assert parsed is not None
    assert parsed[0] == OPCODE_CLOSE


def test_recv_eof_raises_closed_abnormal():
    sock = FakeSocket(b"")  # immediate EOF
    client = RFC6455Client(sock=sock)
    with pytest.raises(WebSocketClosed) as ei:
        client.recv()
    assert ei.value.code == 1006


def test_send_pong_only_control_frame_is_emitted():
    # RECEIVE-ONLY guarantee: the public API exposes only pong/close as sends.
    sock = FakeSocket()
    client = RFC6455Client(sock=sock)
    client.send_pong(b"ping-echo")
    parsed = RFC6455Client._parse_frame(bytes(sock.sent))
    assert parsed is not None
    opcode, _, payload, _ = parsed
    assert opcode == OPCODE_PONG
    assert payload == b"ping-echo"
    assert client._sock.sent  # exactly one control frame written


def test_close_sends_close_and_shuts_socket():
    sock = FakeSocket()
    client = RFC6455Client(sock=sock)
    client.close(code=1001, reason="going away")
    assert sock.closed is True
    parsed = RFC6455Client._parse_frame(bytes(sock.sent))
    assert parsed is not None
    opcode, _, payload, _ = parsed
    assert opcode == OPCODE_CLOSE
    assert struct.unpack(">H", payload[:2])[0] == 1001
    # Idempotent: a second close is a no-op (no new bytes, no raise).
    before = bytes(sock.sent)
    client.close()
    assert bytes(sock.sent) == before


def test_recv_json_decodes_object():
    obj = {"metadata": {"message_type": "session_keepalive"}, "payload": {}}
    sock = FakeSocket(text_frame(json.dumps(obj)))
    client = RFC6455Client(sock=sock)
    assert client.recv_json() == obj


def test_recv_json_non_object_yields_empty():
    sock = FakeSocket(text_frame("[1, 2, 3]"))  # valid JSON, not an object
    client = RFC6455Client(sock=sock)
    assert client.recv_json() == {}


# --------------------------------------------------------------------------- #
# DedupLRU
# --------------------------------------------------------------------------- #
def test_dedup_basic_and_eviction():
    d = DedupLRU(maxsize=3)
    assert d.seen("a") is False  # first sight
    assert d.seen("a") is True   # duplicate
    assert d.seen("b") is False
    assert d.seen("c") is False
    # Adding a 4th distinct id evicts the oldest ("a").
    assert d.seen("d") is False
    assert "a" not in d
    assert "d" in d
    assert len(d) == 3


def test_dedup_lru_recency_refresh():
    d = DedupLRU(maxsize=2)
    d.seen("a")
    d.seen("b")
    # Touch "a" so it becomes most-recently-used; inserting "c" evicts "b".
    assert d.seen("a") is True
    d.seen("c")
    assert "a" in d
    assert "b" not in d
    assert "c" in d


def test_dedup_empty_id_never_coalesces():
    d = DedupLRU(maxsize=4)
    assert d.seen("") is False
    assert d.seen("") is False  # absent id always "unseen" (fail-open)
    assert d.seen(None) is False
    assert len(d) == 0


def test_dedup_rejects_bad_maxsize():
    with pytest.raises(ValueError):
        DedupLRU(maxsize=0)


# --------------------------------------------------------------------------- #
# ChatEvent.from_eventsub
# --------------------------------------------------------------------------- #
def _realistic_chat_notification() -> dict:
    """A realistic channel.chat.message EventSub notification envelope."""
    return {
        "metadata": {
            "message_id": "befa7b53-d79d-478f-86b9-120f112b044e",
            "message_type": "notification",
            "message_timestamp": "2023-09-22T12:34:56.789Z",
            "subscription_type": "channel.chat.message",
            "subscription_version": "1",
        },
        "payload": {
            "subscription": {
                "id": "f1c2a387-161a-49f9-a165-0f21d7a4e1c4",
                "type": "channel.chat.message",
                "version": "1",
                "status": "enabled",
            },
            "event": {
                "broadcaster_user_id": "1971641",
                "broadcaster_user_login": "streamer",
                "broadcaster_user_name": "Streamer",
                "chatter_user_id": "4145994",
                "chatter_user_login": "viewer32",
                "chatter_user_name": "viewer32",
                "message_id": "cc106a89-1814-919d-454c-f4f2f970aae7",
                "message": {
                    "text": "Hey chat! cheer100",
                    "fragments": [
                        {"type": "text", "text": "Hey chat! ", "cheermote": None},
                        {
                            "type": "cheermote",
                            "text": "cheer100",
                            "cheermote": {"prefix": "cheer", "bits": 100, "tier": 1},
                        },
                    ],
                    "message_type": "text",
                },
                "color": "#00FF7F",
                "badges": [
                    {"set_id": "subscriber", "id": "12", "info": "16"},
                    {"set_id": "vip", "id": "1", "info": ""},
                ],
                "cheer": {"bits": 100},
                "reply": {
                    "parent_message_id": "abc",
                    "parent_user_id": "9999",
                    "parent_user_login": "op",
                    "parent_user_name": "OP",
                    "thread_message_id": "abc",
                    "thread_user_id": "9999",
                },
                "channel_points_custom_reward_id": None,
            },
        },
    }


def test_from_eventsub_parses_realistic_notification():
    ev = ChatEvent.from_eventsub(_realistic_chat_notification())
    assert ev is not None
    assert ev.broadcaster_user_id == "1971641"
    assert ev.chatter_user_id == "4145994"
    assert ev.chatter_login == "viewer32"
    assert ev.chatter_name == "viewer32"
    assert ev.text == "Hey chat! cheer100"
    assert len(ev.fragments) == 2
    assert ev.fragments[1]["type"] == "cheermote"
    assert len(ev.badges) == 2
    assert ev.reply_parent_user_id == "9999"
    assert ev.cheer_bits == 100
    assert ev.message_id == "cc106a89-1814-919d-454c-f4f2f970aae7"
    assert ev.message_type == "text"
    assert ev.raw["chatter_user_login"] == "viewer32"


def test_from_eventsub_accepts_bare_event():
    bare = _realistic_chat_notification()["payload"]["event"]
    ev = ChatEvent.from_eventsub(bare)
    assert ev is not None
    assert ev.chatter_login == "viewer32"
    assert ev.cheer_bits == 100


def test_from_eventsub_no_reply_no_cheer_defaults():
    ev = ChatEvent.from_eventsub(
        {
            "event": {
                "broadcaster_user_id": "1",
                "chatter_user_id": "2",
                "chatter_user_login": "plainuser",
                "chatter_user_name": "PlainUser",
                "message_id": "m1",
                "message": {"text": "hi", "fragments": [], "message_type": "text"},
            }
        }
    )
    assert ev is not None
    assert ev.reply_parent_user_id is None
    assert ev.cheer_bits == 0
    assert ev.badges == []
    assert ev.fragments == []


def test_from_eventsub_rejects_wrong_subscription_type():
    env = _realistic_chat_notification()
    env["metadata"]["subscription_type"] = "automod.message.hold"
    assert ChatEvent.from_eventsub(env) is None


def test_from_eventsub_fail_safe_on_garbage():
    assert ChatEvent.from_eventsub({}) is None
    assert ChatEvent.from_eventsub({"payload": {"event": "not-a-dict"}}) is None
    assert ChatEvent.from_eventsub(None) is None  # type: ignore[arg-type]
    # Hostile types in fields must not raise — coerced/defaulted.
    weird = ChatEvent.from_eventsub(
        {"event": {"chatter_user_id": 12345, "message": {"text": 999}}}
    )
    assert weird is not None
    assert weird.chatter_user_id == "12345"  # int coerced to str
    assert weird.text == ""  # non-str text defaulted


def test_from_eventsub_cheer_bits_string_coercion():
    ev = ChatEvent.from_eventsub(
        {"event": {"chatter_user_id": "1", "message": {"text": "x"}, "cheer": {"bits": "250"}}}
    )
    assert ev is not None
    assert ev.cheer_bits == 250


# --------------------------------------------------------------------------- #
# EventSubSession
# --------------------------------------------------------------------------- #
def _welcome_msg(session_id: str = "AgoQ-sid-123", timeout: int = 10) -> dict:
    return {
        "metadata": {"message_type": "session_welcome", "message_timestamp": "t"},
        "payload": {
            "session": {
                "id": session_id,
                "status": "connected",
                "keepalive_timeout_seconds": timeout,
                "reconnect_url": None,
                "connected_at": "2023-07-19T14:56:51.616329898Z",
            }
        },
    }


def test_classify_message_all_types():
    s = EventSubSession()
    assert s.classify_message(_welcome_msg()) == "welcome"
    assert s.classify_message({"metadata": {"message_type": "session_keepalive"}}) == "keepalive"
    assert s.classify_message({"metadata": {"message_type": "notification"}}) == "notification"
    assert s.classify_message({"metadata": {"message_type": "session_reconnect"}}) == "reconnect"
    assert s.classify_message({"metadata": {"message_type": "revocation"}}) == "revocation"
    assert s.classify_message({"metadata": {"message_type": "???"}}) == "unknown"
    assert s.classify_message({}) == "unknown"
    assert s.classify_message("not-a-dict") == "unknown"


def test_parse_welcome_stores_session_and_timeout():
    s = EventSubSession()
    sid = s.parse_welcome(_welcome_msg("sess-XYZ", timeout=42))
    assert sid == "sess-XYZ"
    assert s.session_id == "sess-XYZ"
    assert s.keepalive_timeout_seconds == 42.0


def test_parse_welcome_rejects_non_welcome():
    s = EventSubSession()
    assert s.parse_welcome({"metadata": {"message_type": "notification"}}) is None
    assert s.parse_welcome({"metadata": {"message_type": "session_welcome"}, "payload": {}}) is None
    assert s.session_id is None


def test_keepalive_staleness():
    s = EventSubSession(keepalive_timeout_seconds=10.0)
    # Never seen traffic -> not stale (handshake still in flight).
    assert s.is_stale(now=1000.0) is False
    s.note_keepalive(now=1000.0)
    # Within the window: fresh.
    assert s.is_stale(now=1005.0) is False
    # Exactly at the window edge: still fresh (strictly greater is stale).
    assert s.is_stale(now=1010.0) is False
    # Past the window: stale.
    assert s.is_stale(now=1011.0) is True
    # Explicit override timeout.
    assert s.is_stale(now=1003.0, timeout=2.0) is True


def test_handle_reconnect_returns_new_url():
    s = EventSubSession()
    msg = {
        "metadata": {"message_type": "session_reconnect"},
        "payload": {
            "session": {
                "id": "sid",
                "status": "reconnecting",
                "reconnect_url": "wss://eventsub.wss.twitch.tv/ws?challenge=new",
                "connected_at": "x",
            }
        },
    }
    assert s.handle_reconnect(msg) == "wss://eventsub.wss.twitch.tv/ws?challenge=new"


def test_handle_reconnect_rejects_bad_messages():
    s = EventSubSession()
    assert s.handle_reconnect({"metadata": {"message_type": "session_keepalive"}}) is None
    assert (
        s.handle_reconnect(
            {"metadata": {"message_type": "session_reconnect"}, "payload": {"session": {}}}
        )
        is None
    )


def test_note_keepalive_default_clock_advances_staleness():
    # Using the real monotonic clock path: a fresh note is never immediately stale.
    s = EventSubSession(keepalive_timeout_seconds=100.0)
    s.note_keepalive()
    assert s.last_keepalive is not None
    assert s.is_stale() is False
