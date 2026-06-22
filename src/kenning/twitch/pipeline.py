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
    ) -> None:
        self._validator = validator
        self._speak = speak_fn
        self._max_chars = int(max_reply_chars)
        self._deflect = deflect_fn

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _attr(ev: Any, name: str, default: str = "") -> str:
        return str(getattr(ev, name, default) or default)

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

        result.spoke = spoken
        result.deflected = tripped
        result.answered_user_ids = tuple(self._attr(e, "chatter_user_id") for e in selected)
        result.reason = "deflected" if tripped else "spoke"
        return result
