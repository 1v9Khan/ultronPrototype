"""S10 — the live chat-reply pipeline (the safety-critical integration heart).

Orchestrates one batch of buffered Twitch chat into ONE spoken reply, enforcing
the full safety contract and team isolation. The flow:

  events -> keep reply-targets (addressing) -> INPUT safety screen (drop/flag
  unsafe BEFORE the abliterated 8B) -> selection -> ONE 8B draft (datamarked /
  CHATTER_N) -> OUTPUT safety screen (guard + reassembly + phonetic, exchange
  mode) -> ALLOW=speak the draft / else=speak a constant deflection -> speak ONLY
  on the stream bus with provenance=TWITCH_CHAT (the relay boundary REFUSES that
  provenance, so chat can never reach the team mic).

DECOUPLED by injection: the pipeline takes callables (``is_reply_target``,
``select_fn``, ``reply_fn``, ``speak_fn``) + a :class:`ChatSafetyValidator`, so it
is fully unit-testable offline and the orchestrator binds the live addressing /
EmbeddingGemma / 8B / Kokoro+broadcast at boot. The pipeline holds NO handle to
the relay/PTT path (team isolation is structural).

FAIL-CLOSED everywhere: any addressing/selection/reply/validator error drops that
message or the whole batch to silence -- never an unscreened utterance. The guard
model is REQUIRED to be wired into the validator when chat-reply is enabled (the
toggle gate enforces that; see the orchestrator integration).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from kenning.audio.provenance import Provenance
from kenning.twitch.safety.deflection import pick_deflection
from kenning.twitch.safety.validator import (
    ChatMessageContext,
    ChatSafetyValidator,
    ChatVerdict,
)

logger = logging.getLogger("kenning.twitch.pipeline")

__all__ = ["ChatReplyPipeline", "BatchResult", "FlaggedMessage"]


@dataclass(frozen=True)
class FlaggedMessage:
    """An inbound message the safety stack did not pass to the 8B."""
    user_id: str
    username: str
    text: str
    verdict: str            # REVIEW (gray zone -> popup) | BLOCK (auto-moderate)
    danger_score: float
    categories: tuple[str, ...]


@dataclass
class BatchResult:
    spoke: Optional[str] = None          # the line actually spoken (draft or deflection), or None
    deflected: bool = False              # True when the output gate replaced the draft
    answered_user_ids: tuple[str, ...] = ()
    flagged: list[FlaggedMessage] = field(default_factory=list)
    dropped_unsafe: int = 0
    reason: str = ""


class ChatReplyPipeline:
    """Process one chat batch into one safety-gated spoken reply (or silence)."""

    def __init__(
        self,
        *,
        validator: ChatSafetyValidator,
        speak_fn: Callable[..., Any],
        max_reply_chars: int = 240,
        deflect_fn: Callable[[str], str] = pick_deflection,
        cooldown_seconds: float = 0.0,
        chat_post_fn: Optional[Callable[[str], Any]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._validator = validator
        self._speak = speak_fn
        self._max_chars = int(max_reply_chars)
        self._deflect = deflect_fn
        # 2026-06-26 per-user chat-reply cooldown. ``cooldown_seconds`` <= 0 keeps
        # the legacy behavior (no throttle). ``chat_post_fn`` posts the on-cooldown
        # note to chat (no TTS); None -> the reply is simply suppressed silently.
        # ``clock`` is injected so the cooldown is deterministically unit-testable.
        self._cooldown_s = max(0.0, float(cooldown_seconds))
        self._chat_post = chat_post_fn
        self._clock = clock
        # user_id (fallback login) -> monotonic timestamp of the last reply to them.
        self._last_reply_at: dict[str, float] = {}

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _attr(ev: Any, name: str, default: str = "") -> str:
        return str(getattr(ev, name, default) or default)

    @staticmethod
    def _cooldown_key(ev: Any) -> str:
        """Stable per-viewer key for the cooldown table: the user id when present,
        else the lowercased login (so a viewer without a resolved id is still
        throttled). ``""`` when neither is known (never throttled)."""
        uid = ChatReplyPipeline._attr(ev, "chatter_user_id").strip()
        if uid:
            return uid
        return ChatReplyPipeline._attr(ev, "chatter_login").strip().lower()

    @staticmethod
    def _display_name(ev: Any) -> str:
        """The @-tag handle for a viewer: their display name, else login."""
        return (ChatReplyPipeline._attr(ev, "chatter_name")
                or ChatReplyPipeline._attr(ev, "chatter_login")).strip()

    def _cooldown_remaining(self, key: str) -> float:
        """Seconds left on ``key``'s cooldown (0.0 when free / disabled)."""
        if self._cooldown_s <= 0.0 or not key:
            return 0.0
        last = self._last_reply_at.get(key)
        if last is None:
            return 0.0
        remaining = self._cooldown_s - (self._clock() - last)
        return remaining if remaining > 0.0 else 0.0

    @staticmethod
    def _tag(name: str, text: str) -> str:
        """Prefix a reply with a leading "@<name> " for the specific viewer it
        answers (idempotent: never double-tags an already-@-prefixed line)."""
        handle = (name or "").strip().lstrip("@").strip()
        body = (text or "").strip()
        if not handle:
            return body
        if body.lower().startswith(f"@{handle.lower()} ") or body.lower() == f"@{handle.lower()}":
            return body
        return f"@{handle} {body}".strip()

    def _screen_input(self, ev: Any) -> ChatVerdict:
        """Run the INPUT safety stack on a message (+ its untrusted metadata)."""
        ctx = ChatMessageContext(
            text=self._attr(ev, "text"),
            username=self._attr(ev, "chatter_name") or self._attr(ev, "chatter_login"),
            user_id=self._attr(ev, "chatter_user_id"),
            source="twitch_chat",
            is_output=False,
            extra_fields=(self._attr(ev, "chatter_login"), self._attr(ev, "chatter_name")),
        )
        return self._validator.check(ctx)

    def _post_cooldown_note(self, ev: Any, remaining: float) -> None:
        """Post a brief in-chat "@<user> on cooldown (N s left)" note (NO TTS) for a
        viewer who is rate-limited. Fail-safe: no ``chat_post_fn`` -> silent skip; a
        post error never propagates."""
        if self._chat_post is None:
            return
        handle = self._display_name(ev)
        secs = max(1, int(round(remaining)))
        msg = self._tag(
            handle,
            f"easy — I just answered you. Try again in {secs}s.",
        )
        try:
            self._chat_post(msg)
        except Exception as e:  # noqa: BLE001 — a chat-post hiccup never breaks the tick
            logger.debug("cooldown chat-note post failed: %s", e)

    # -- the batch -------------------------------------------------------------
    def process_batch(
        self,
        events: Sequence[Any],
        *,
        is_reply_target: Callable[[Any], bool],
        select_fn: Callable[[list], list],
        reply_fn: Callable[[list], str],
    ) -> BatchResult:
        """Process one batch. Never raises (fail-closed to silence)."""
        result = BatchResult()
        if not events:
            result.reason = "empty batch"
            return result

        # 1) keep only reply-targets (to-Ultron). Fail-closed: an addressing error
        #    drops that message (treated as not-a-target).
        targets: list[Any] = []
        for ev in events:
            try:
                if is_reply_target(ev):
                    targets.append(ev)
            except Exception as e:  # noqa: BLE001
                logger.debug("addressing error on a message; dropping it: %s", e)
        if not targets:
            result.reason = "no reply-targets"
            return result

        # 2) INPUT safety screen BEFORE the 8B. Drop BLOCK/REVIEW; surface them.
        safe: list[Any] = []
        for ev in targets:
            try:
                d = self._screen_input(ev)
            except Exception as e:  # noqa: BLE001 — fail-CLOSED: drop on any error
                logger.warning("input screen error; dropping message: %s", e)
                result.dropped_unsafe += 1
                continue
            if d.verdict == ChatVerdict.ALLOW:
                safe.append(ev)
                continue
            result.dropped_unsafe += 1
            result.flagged.append(FlaggedMessage(
                user_id=self._attr(ev, "chatter_user_id"),
                username=self._attr(ev, "chatter_name") or self._attr(ev, "chatter_login"),
                text=self._attr(ev, "text"),
                verdict=d.verdict.name,
                danger_score=round(float(d.danger_score), 3),
                categories=tuple(sorted({m.category for m in d.matches})),
            ))
        if not safe:
            result.reason = "all reply-targets failed input safety"
            return result

        # 3) selection (fail-closed: a selection error -> answer nothing this batch)
        try:
            selected = list(select_fn(safe))
        except Exception as e:  # noqa: BLE001
            logger.warning("selection error; skipping batch: %s", e)
            result.reason = "selection error"
            return result
        if not selected:
            result.reason = "selection returned nothing"
            return result

        # 3b) PER-USER COOLDOWN (2026-06-26). The reply addresses the PRIMARY
        #     selected viewer; if they were answered within the cooldown window,
        #     do NOT speak — POST a brief "@<user> ... (N s left)" chat note instead
        #     (no TTS, no wasted 8B draft) and end the batch. Disabled (cooldown<=0)
        #     -> this whole block is a no-op (byte-identical legacy behavior).
        primary = selected[0]
        cd_key = self._cooldown_key(primary)
        remaining = self._cooldown_remaining(cd_key)
        if remaining > 0.0:
            self._post_cooldown_note(primary, remaining)
            result.reason = "on cooldown"
            return result

        # 4) ONE 8B draft (the reply_fn does datamarking / CHATTER_N internally).
        try:
            draft = (reply_fn(selected) or "").strip()
        except Exception as e:  # noqa: BLE001 — fail-CLOSED: no draft -> no speech
            logger.warning("reply generation error; staying silent: %s", e)
            result.reason = "reply error"
            return result
        if not draft:
            result.reason = "empty draft"
            return result

        # 5) OUTPUT safety screen the draft (exchange mode: judged vs the inbound it
        #    answers). A trip -> a CONSTANT in-character deflection (never the draft,
        #    never regenerate-to-comply).
        inbound = " | ".join(self._attr(e, "text") for e in selected)[:2000]
        try:
            out = self._validator.check(ChatMessageContext(
                text=draft, source="twitch_chat", is_output=True,
                inbound_for_exchange=inbound,
                batch_context=tuple(self._attr(e, "text") for e in selected),
            ))
            tripped = out.verdict != ChatVerdict.ALLOW
            spoken = (out.deflection or self._deflect(draft)) if tripped else draft
        except Exception as e:  # noqa: BLE001 — fail-CLOSED: deflect on any error
            logger.warning("output screen error; deflecting: %s", e)
            tripped, spoken = True, self._deflect(draft)

        spoken = spoken[: self._max_chars].strip()
        if not spoken:
            result.reason = "empty after output gate"
            return result

        # @-TAG the reply with the PRIMARY viewer it answers, then re-cap so the
        # "@<user> " prefix never pushes past the char limit (2026-06-26).
        spoken = self._tag(self._display_name(primary), spoken)[: self._max_chars].strip()
        if not spoken:
            result.reason = "empty after tag"
            return result

        # 6) SPEAK on the stream bus ONLY, tagged TWITCH_CHAT. The relay boundary
        #    refuses this provenance, so chat physically cannot reach the team mic.
        try:
            self._speak(spoken, provenance=Provenance.TWITCH_CHAT)
        except TypeError:
            # speak_fn without a provenance kwarg: still SAFE because this pipeline
            # has no relay handle; but log it so the contract stays explicit.
            logger.debug("speak_fn has no provenance kwarg; speaking on default sink")
            self._speak(spoken)
        except Exception as e:  # noqa: BLE001 — a speech error must not crash the loop
            logger.warning("chat speak failed: %s", e)
            result.reason = "speak error"
            return result

        # Mark the cooldown for the answered viewer AFTER a successful speak so a
        # failed/empty reply does not lock them out.
        if self._cooldown_s > 0.0 and cd_key:
            self._last_reply_at[cd_key] = self._clock()

        result.spoke = spoken
        result.deflected = tripped
        result.answered_user_ids = tuple(self._attr(e, "chatter_user_id") for e in selected)
        result.reason = "deflected" if tripped else "spoke"
        return result
