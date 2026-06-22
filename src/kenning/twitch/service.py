"""S10 — ChatModeService: connect the chat-mode runtime to the live sidecars.

The orchestrator constructs ONE of these at boot IFF ``twitch.enabled`` (so flags
OFF imports nothing -> byte-identical), passing its live 8B / EmbeddingGemma / TTS.
The service connects to the ALREADY-RUNNING guard + read sidecars over loopback
(the user / a launch script starts them with the anti-stale-sidecar guards), builds
the :class:`ChatModeRuntime`, and exposes ``set_chat_mode`` / ``tick`` / ``stop``.

Design choices that keep the orchestrator change tiny + safe:
  * **Connect, don't spawn.** The service does NOT spawn sidecars (subprocess from
    the golden-path process); it consumes the running guard/read sidecars via thin
    urllib clients. Lifecycle stays with the standalone sidecars + their deadman.
  * **Everything off the hot path.** ``tick`` is called from the idle loop and is
    fully fail-CLOSED; it never touches the relay/team path.
  * **Live deps injected.** ``llm_fn`` (8B), ``embed_fn`` (EmbeddingGemma),
    ``orchestrator_speak`` (Kokoro -> speakers+OBS), ``on_flagged`` (review popup)
    are passed in -> the service logic is offline-testable with mocks.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any, Callable, Optional

from kenning.twitch.guard import GuardModelClient
from kenning.twitch.integration import build_chat_mode_runtime, make_stream_speak_fn
from kenning.twitch.runtime import ChatModeRuntime

logger = logging.getLogger("kenning.twitch.service")

__all__ = ["ChatModeService", "make_read_drain_fn"]


def make_read_drain_fn(read_endpoint: str, *, timeout: float = 1.0) -> Callable[[], list]:
    """Build a drain callable that pulls + acks new chat events from the read
    sidecar's rolling buffer and parses them into ChatEvents. Fail-safe: any error
    returns an empty batch (the runtime just skips the tick)."""
    base = read_endpoint.rstrip("/")
    cursor = {"v": 0}

    def drain() -> list:
        from kenning.twitch.clients.eventsub import ChatEvent
        try:
            req = urllib.request.Request(f"{base}/buffer?since={cursor['v']}", method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read() or b"{}")
        except Exception as e:  # noqa: BLE001 — read sidecar down -> empty batch
            logger.debug("read-sidecar drain failed: %s", e)
            return []
        cursor["v"] = int(data.get("cursor", cursor["v"]) or cursor["v"])
        out = []
        for raw in data.get("events", []) or []:
            try:
                ev = ChatEvent.from_eventsub(raw) if isinstance(raw, dict) else None
                if ev is not None:
                    out.append(ev)
            except Exception:  # noqa: BLE001 — skip a malformed event, never crash
                continue
        return out

    return drain


class ChatModeService:
    def __init__(
        self,
        twitch_cfg: Any,
        *,
        llm_fn: Callable[[str, str], str],
        orchestrator_speak: Callable[[str], Any],
        embed_fn: Optional[Callable[[str], Any]] = None,
        on_flagged: Optional[Callable[..., None]] = None,
        drain_fn: Optional[Callable[[], list]] = None,
        guard_client: Optional[Any] = None,
        bot_user_id: str = "",
        streamer_user_id: str = "",
    ) -> None:
        self._cfg = twitch_cfg
        safety_cfg = getattr(twitch_cfg, "safety", None)
        guard_ep = str(getattr(safety_cfg, "guard_endpoint", "http://127.0.0.1:8774"))
        read_ep = str(getattr(twitch_cfg, "read_sidecar_endpoint", "http://127.0.0.1:8773"))
        self._guard = guard_client if guard_client is not None else GuardModelClient(guard_ep)
        drain = drain_fn if drain_fn is not None else make_read_drain_fn(read_ep)
        self._runtime: ChatModeRuntime = build_chat_mode_runtime(
            twitch_cfg,
            llm_fn=llm_fn,
            speak_fn=make_stream_speak_fn(orchestrator_speak),
            drain_fn=drain,
            guard_client=self._guard,
            embed_fn=embed_fn,
            bot_user_id=bot_user_id,
            streamer_user_id=streamer_user_id,
            on_flagged=on_flagged,
        )

    @property
    def active(self) -> bool:
        return self._runtime.active

    @property
    def state(self) -> str:
        return self._runtime.state.value

    def set_chat_mode(self, on: bool) -> tuple[bool, str]:
        """Stream-Deck / voice toggle -> enable (guard-gated) or disable chat-reply."""
        if on:
            return self._runtime.enable()
        self._runtime.disable()
        return True, "disabled"

    def tick(self) -> Optional[Any]:
        """Called from the idle loop while chat-mode may be on. Fail-CLOSED."""
        try:
            return self._runtime.tick()
        except Exception as e:  # noqa: BLE001 — never let chat ticking crash the loop
            logger.warning("chat-mode tick failed: %s", e)
            return None

    def sync_and_tick(self, want_on: bool) -> Optional[Any]:
        """Reconcile chat-mode to ``want_on`` (the live ``reply_enabled`` flag the
        Stream-Deck/GUI flips), then tick. The single call the orchestrator's
        background loop makes — all the state logic stays here (testable)."""
        try:
            if want_on and not self.active:
                ok, why = self.set_chat_mode(True)
                if not ok:
                    logger.warning("chat-reply enable refused: %s", why)
            elif not want_on and self.active:
                self.set_chat_mode(False)
        except Exception as e:  # noqa: BLE001
            logger.debug("chat-mode sync error: %s", e)
        return self.tick() if self.active else None

    def stop(self) -> None:
        try:
            self._runtime.disable()
        except Exception:  # noqa: BLE001
            pass
