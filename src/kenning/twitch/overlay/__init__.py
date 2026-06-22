"""S8 — the local OBS Browser Source overlay server (textContent-only renderer).

ONE local HTTP endpoint (``http://127.0.0.1:PORT/?token=...``) that the streamer
pastes into a single OBS Browser Source. The page is a DUMB renderer: it never
decides an outcome (the sidecar's crypto RNG picks the wheel winner FIRST, then
the overlay animates deterministically to a server-supplied target angle), it
renders ALL dynamic text via ``textContent`` (never ``innerHTML``), and it is
locked down by a STRICT Content-Security-Policy served as both a response header
and a ``<meta>`` tag.

ANTICHEAT POSTURE (BR-P1): this module is pure stdlib only —
``http.server`` / ``socketserver`` / ``secrets`` / ``json`` / ``queue`` /
``threading`` / ``html``. It imports NO desktop-automation, screen-capture,
input-injection, or third-party network library. There is ZERO game/screen
capture anywhere (the overlay is fed by the sidecar over SSE; it never reads the
screen). Bind is 127.0.0.1 ONLY + a per-session ``secrets`` token on every route.

Flag-gated default-OFF: with chat-mode OFF the server is never started, no port
binds. Design + threat model: docs/twitch_integration/ (MASTER.md §5 overlays).
"""
from __future__ import annotations

from kenning.twitch.overlay.server import (
    OverlayError,
    OverlayServer,
    validate_event,
)

__all__ = ["OverlayServer", "OverlayError", "validate_event"]
