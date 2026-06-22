"""Ultron Twitch capability — chat interaction, content-creation, moderation.

ALL behavior is flag-gated default-OFF via ``config.twitch`` + ``KENNING_TWITCH_*``.
With the master switch OFF nothing here is imported into the anticheat-pinned
voice/relay process (no sidecar spawns, no port binds, no DB opens, no hooks) ->
the competitive runtime is byte-identical (BR-P1; yardstick = the frozen 24-fail
control set).

ANTICHEAT POSTURE (BR-P1). This package is split by import-safety:
  * VOICE-PROCESS-SAFE clients — thin ``urllib`` loopback shims that import only
    stdlib + urllib + numpy + rapidfuzz; the main process MAY import these behind
    the flag (same class as the EmbeddingGemma router client).
  * SIDECAR code — EventSub WebSocket, guard/Prompt-Guard models, OBS, SQLite —
    runs ONLY in separate sidecar processes (``scripts/twitch_*.py``); the main
    process NEVER imports it. Heavy deps live in a separate ``.venv-twitch``.

The deterministic safety layers (L1 normalize/blocklist, L5 reassembly, L6
phonetic, deflection, the provenance taint) are pure stdlib+rapidfuzz and run in
either process. The model-backed layers fail-CLOSED when their sidecar/deps are
absent: chat-reply mode refuses to enable without a healthy guard model.

Design + threat model: docs/twitch_integration/ (REQUIREMENTS / DESIGN /
constitution / 02_board/MASTER.md).
"""
from __future__ import annotations

__all__: list[str] = []
