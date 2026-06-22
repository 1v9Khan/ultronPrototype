"""Voice-process-safe Twitch transport clients (read-only).

ANTICHEAT POSTURE (BR-P1). Every module in this package imports ONLY the
allow-listed set — ``socket``/``ssl``/``base64``/``hashlib``/``struct``/``json``/
``os``/``logging``/``dataclasses`` + stdlib. NO desktop-automation, NO
screen-capture, NO input-injection, NO ``requests``/``aiohttp``/``websockets``/
``transformers``. The RFC6455 WebSocket client is a hand-rolled stdlib codec so the
sidecar (and tests) need no third-party network library.

The EventSub read transport here is RECEIVE-ONLY: it speaks the WebSocket control
protocol (PING/PONG/CLOSE) but never emits application data frames. Subscriptions
are created out-of-band over Helix; this client only consumes the wss notification
stream. Keep this docstring import-light — nothing heavy at module import time.
"""
from __future__ import annotations

__all__: list[str] = []
