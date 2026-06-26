"""S10 — assemble the live chat-mode runtime from the leaf modules + config + deps.

This factory is the single place the addressing / selection / reply / safety
modules get bound into a :class:`ChatModeRuntime`. The LIVE dependencies are
INJECTED (the 8B as ``llm_fn``, the EmbeddingGemma loopback as ``embed_fn``, the
stream-bus TTS as ``speak_fn``, the guard sidecar client, the read-sidecar buffer
drain), so this is fully offline-testable and the orchestrator supplies the real
ones at boot. The orchestrator imports this ONLY when ``twitch.enabled`` -> with
the flag OFF nothing here loads (main runtime byte-identical).

The stream-bus speak helper guarantees team isolation at the output: chat audio is
spoken on the streamer's normal speaker + OBS/broadcast path (which every
conversational line already uses) and is tagged ``Provenance.TWITCH_CHAT`` — it
NEVER touches the relay/team-mic path (the relay boundary refuses that provenance).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from kenning.audio.provenance import Provenance
from kenning.twitch.addressing import ChatAddress, classify_chat
from kenning.twitch.pipeline import ChatReplyPipeline
from kenning.twitch.reply import generate_reply
from kenning.twitch.runtime import ChatModeRuntime
from kenning.twitch.safety.validator import build_chat_validator
from kenning.twitch.selection import select_messages

logger = logging.getLogger("kenning.twitch.integration")

__all__ = ["build_chat_mode_runtime", "make_stream_speak_fn"]


def make_stream_speak_fn(orchestrator_speak: Callable[[str], Any]) -> Callable[..., None]:
    """Wrap the orchestrator's normal speak (synthesize -> speakers + OBS/broadcast
    tee, NOT the relay) into the pipeline's ``speak_fn(text, provenance=...)``
    contract. Chat audio thus reaches the stream + the streamer, never teammates."""
    def speak(text: str, *, provenance: Provenance = Provenance.TWITCH_CHAT) -> None:
        # provenance is carried for the defense-in-depth relay guard; this path
        # already cannot reach the team mic (it calls the conversational speak).
        if provenance is not Provenance.TWITCH_CHAT:
            logger.error("stream speak called with non-chat provenance %r; refusing", provenance)
            return
        orchestrator_speak(text)
    return speak


def build_chat_mode_runtime(
    twitch_cfg: Any,
    *,
    llm_fn: Callable[[str, str], str],
    speak_fn: Callable[..., None],
    drain_fn: Callable[[], Any],
    guard_client: Optional[Any] = None,
    embed_fn: Optional[Callable[[str], Any]] = None,
    bot_user_id: str = "",
    streamer_user_id: str = "",
    on_flagged: Optional[Callable[..., None]] = None,
    busy_estimator: Optional[Any] = None,
    chat_post_fn: Optional[Callable[[str], Any]] = None,
) -> ChatModeRuntime:
    """Wire the full chat-reply stack into a runtime. ``twitch_cfg`` is the
    ``config.twitch`` section. The returned runtime starts OFF; the toggle calls
    ``enable()`` (guard-gated)."""
    bot_login = str(getattr(getattr(twitch_cfg, "auth", None), "bot_login", "") or "")
    streamer_login = str(getattr(getattr(twitch_cfg, "auth", None), "broadcaster_login", "") or "")
    chat_cfg = getattr(twitch_cfg, "chat", None)
    safety_cfg = getattr(twitch_cfg, "safety", None)
    max_msgs = int(getattr(chat_cfg, "batch_max_messages", 40) or 40)
    max_chars = int(getattr(chat_cfg, "reply_max_chars", 240) or 240)
    cooldown_s = float(getattr(chat_cfg, "reply_cooldown_seconds", 120) or 0)
    guard_required = bool(getattr(safety_cfg, "guard_required", True))

    def is_reply_target(ev: Any) -> bool:
        v = classify_chat(
            ev, bot_login=bot_login, bot_user_id=bot_user_id,
            streamer_login=streamer_login, streamer_user_id=streamer_user_id,
            embed_fn=embed_fn,
        )
        return v.address == ChatAddress.TO_ULTRON

    def select_fn(events: list) -> list:
        return select_messages(events, max_messages=max_msgs).chosen

    def reply_fn(selected: list) -> str:
        return generate_reply(selected, llm_fn)

    validator = build_chat_validator(guard_client=guard_client)
    pipeline = ChatReplyPipeline(
        validator=validator, speak_fn=speak_fn, max_reply_chars=max_chars,
        cooldown_seconds=cooldown_s, chat_post_fn=chat_post_fn,
    )
    return ChatModeRuntime(
        pipeline=pipeline, drain_fn=drain_fn,
        is_reply_target=is_reply_target, select_fn=select_fn, reply_fn=reply_fn,
        guard_client=guard_client, guard_required=guard_required, on_flagged=on_flagged,
        busy_estimator=busy_estimator,
    )
