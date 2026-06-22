"""S10a — semantic chat addressing (deterministic-metadata-first, FAIL-CLOSED).

Classifies an inbound Twitch ``channel.chat.message`` into WHO the chatter is
addressing, so the selection/reply engine (SLICE 10) only ever drafts a reply to
lines actually directed at Ultron. Mirrors the ethos of
:mod:`kenning.audio.intent_gate` — cost-asymmetric and **FAIL-CLOSED to IGNORE**.
A false TO_ULTRON costs a wasted (rate-limited, 1 msg/sec/channel) public reply
to someone talking to chat/the streamer/another viewer, which reads as the bot
butting in; a missed one is invisible. So anything ambiguous or erroring defaults
to :attr:`ChatAddress.IGNORE`.

Cascade (deterministic-first; trust the IMMUTABLE ``user_id``, NEVER the spoofable
``display_name``):

1. ``!``-prefixed body                                   -> COMMAND
2. ``reply.parent_user_id == bot_user_id``               -> TO_ULTRON (reply to us)
3. an @mention whose RESOLVED user is the bot            -> TO_ULTRON
4. an @mention of the streamer                           -> TO_STREAMER
5. an @mention of another (non-bot, non-streamer) user   -> TO_OTHER
6. a leading 'ultron'/bot-name/bot-login token           -> TO_ULTRON
7. RESIDUAL (no @mention, no name token): cosine-vs-exemplar-clouds if an
   ``embed_fn`` is supplied (with a margin), else                 -> IGNORE

Mentions are resolved from BOTH the typed ``event.fragments`` (a ``mention``
fragment carries the immutable ``user_id`` — authoritative) AND raw ``@name``
tokens in the body (matched by login, the stable handle, not display name).
A spoofed display name therefore never wins: resolution is by ``user_id`` /
login, both immutable.

ANTICHEAT (BR-P1): stdlib + ``rapidfuzz`` only. No network, no models — the
optional ``embed_fn`` is injected by the caller (a loopback EmbeddingGemma shim in
the sidecar) so this module and its tests run fully OFFLINE with mocks.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger("kenning.twitch.addressing")

__all__ = [
    "ChatAddress",
    "AddressVerdict",
    "classify_chat",
    "TO_ULTRON_EXEMPLARS",
    "NOT_TO_ULTRON_EXEMPLARS",
]


class ChatAddress(str, Enum):
    """Who an inbound chat line is addressing."""

    TO_ULTRON = "TO_ULTRON"     # directed at the bot -> eligible for a reply
    TO_STREAMER = "TO_STREAMER"  # directed at the human broadcaster
    TO_OTHER = "TO_OTHER"        # @ another viewer -> not ours
    COMMAND = "COMMAND"          # a ``!command`` -> the economy/command router owns it
    IGNORE = "IGNORE"            # chatter-to-chat / ambient / ambiguous -> drop (fail-closed)


@dataclass(frozen=True)
class AddressVerdict:
    address: ChatAddress
    confidence: float
    reason: str


# --- residual exemplar clouds (used ONLY when an embed_fn is injected) ------------
# Small, hand-curated text clouds: lines that ARE to the bot (questions/imperatives
# directed at an AI teammate) vs lines that are NOT (chat banter, stream reactions,
# self-talk). The caller embeds these once and passes ``embed_fn``; we never embed
# in-process. Kept short on purpose — this is a residual tie-breaker, not the
# primary signal (which is deterministic metadata above).
TO_ULTRON_EXEMPLARS: tuple[str, ...] = (
    "are you real",
    "what do you think about that",
    "can you answer my question",
    "tell me a joke",
    "what's your opinion on this play",
    "do you actually understand what we say",
    "say something funny",
    "what would you do here",
    "are you watching the game",
    "respond to this",
)
NOT_TO_ULTRON_EXEMPLARS: tuple[str, ...] = (
    "that play was insane",
    "gg everyone",
    "lol same",
    "what a game",
    "nice clutch streamer",
    "i love this stream",
    "anyone else lagging",
    "first time here hello chat",
    "poggers",
    "what time is it where you live",
)

# Residual cosine margin: the to-Ultron mean similarity must beat the not-to-Ultron
# mean by at least this much (env-tunable) to commit TO_ULTRON. Otherwise IGNORE.
# Asymmetric on purpose — a false TO_ULTRON is the expensive error.
_RESIDUAL_MARGIN = 0.06
# And it must clear an absolute floor so a line dissimilar to BOTH clouds (random
# banter) cannot squeak through on a razor-thin relative edge.
_RESIDUAL_MIN_SIM = 0.35


# --------------------------------------------------------------------------- #
# small text helpers
# --------------------------------------------------------------------------- #
def _norm_login(value: Any) -> str:
    """Normalize a login/name to its comparable form: lowercase, stripped, no '@'."""
    if not isinstance(value, str):
        return ""
    return value.strip().lstrip("@").strip().lower()


# Leading wake/name token: 'ultron' (and the common ASR/typo variants chat uses)
# optionally behind a 'hey'/'ok'/'yo'/'@'. The leading anchor matters — a third
# person mention mid-sentence ("ultron is broken lol") is NOT a leading address;
# we leave that to the @mention path, which carries the immutable user_id.
_LEADING_NAME_RE = re.compile(
    r"^\s*(?:hey[\s,]+|hi[\s,]+|yo[\s,]+|ok[\s,]+|okay[\s,]+|@)?"
    r"(?:ultron|altron|ultraun|ultronn|ultro)\b",
    re.IGNORECASE,
)

# Any '@name' token in the raw body (Twitch logins: 4-25 chars, [A-Za-z0-9_]).
_AT_MENTION_RE = re.compile(r"@([A-Za-z0-9_]{1,25})")


def _leading_name_token(text: str, *, bot_login: str, bot_name: str) -> bool:
    """True if the line LEADS with the bot's name/login (or a known ultron variant)."""
    if not isinstance(text, str) or not text.strip():
        return False
    if _LEADING_NAME_RE.match(text):
        return True
    # Also honor a leading literal bot login / display name token, e.g. a custom
    # bot called "kenbot": "kenbot what's the score".
    lead = re.match(r"^\s*@?([A-Za-z0-9_]{2,25})\b", text)
    if lead is None:
        return False
    tok = lead.group(1).lower()
    return bool(tok) and (tok == _norm_login(bot_login) or tok == _norm_login(bot_name))


# --------------------------------------------------------------------------- #
# mention resolution (immutable user_id first, login second)
# --------------------------------------------------------------------------- #
def _fragment_mentions(fragments: Sequence[Any]) -> list[tuple[str, str]]:
    """Extract (user_id, login) pairs from typed ``mention`` fragments.

    Twitch ``channel.chat.message`` mention fragments look like
    ``{"type": "mention", "text": "@bot",
       "mention": {"user_id": "...", "user_login": "...", "user_name": "..."}}``.
    We also tolerate a flat ``{"type": "mention", "user_id": ..., "user_login": ...}``.
    The ``user_id`` is the authoritative, immutable key. Fail-safe: a malformed
    fragment is skipped, never raised.
    """
    out: list[tuple[str, str]] = []
    if not isinstance(fragments, (list, tuple)):
        return out
    for frag in fragments:
        if not isinstance(frag, dict):
            continue
        if frag.get("type") != "mention":
            continue
        m = frag.get("mention")
        src = m if isinstance(m, dict) else frag
        uid = src.get("user_id")
        login = src.get("user_login")
        uid_s = uid.strip() if isinstance(uid, str) else ""
        login_s = _norm_login(login)
        if not login_s:
            # Fall back to the visible '@text' if the structured login is absent.
            txt = frag.get("text")
            if isinstance(txt, str):
                login_s = _norm_login(txt)
        if uid_s or login_s:
            out.append((uid_s, login_s))
    return out


def _raw_at_logins(text: str) -> list[str]:
    """Every distinct '@login' token in the body, normalized (order-preserving)."""
    if not isinstance(text, str):
        return []
    seen: list[str] = []
    for m in _AT_MENTION_RE.finditer(text):
        login = _norm_login(m.group(1))
        if login and login not in seen:
            seen.append(login)
    return seen


# --------------------------------------------------------------------------- #
# residual semantic tie-break
# --------------------------------------------------------------------------- #
def _cosine(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Cosine similarity of two vectors; None on shape/zero/NaN problems."""
    try:
        if not a or not b or len(a) != len(b):
            return None
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            fx = float(x)
            fy = float(y)
            dot += fx * fy
            na += fx * fx
            nb += fy * fy
        if na <= 0.0 or nb <= 0.0:
            return None
        sim = dot / (math.sqrt(na) * math.sqrt(nb))
        if math.isnan(sim) or math.isinf(sim):
            return None
        # Clamp tiny FP overshoot.
        return max(-1.0, min(1.0, sim))
    except (TypeError, ValueError):
        return None


def _embed(embed_fn: Callable[[str], Optional[Sequence[float]]], text: str) -> Optional[Sequence[float]]:
    """Call the injected embedder, fail-safe to None on any error/empty result."""
    try:
        vec = embed_fn(text)
    except Exception as exc:  # noqa: BLE001 — never raise into the classifier
        logger.warning("twitch addressing embed_fn raised: %s", exc)
        return None
    if not vec:
        return None
    try:
        # Reject a non-finite / non-numeric vector early.
        if any((not isinstance(v, (int, float)) or v != v) for v in vec):
            return None
    except TypeError:
        return None
    return vec


def _mean_sim(query: Sequence[float],
              cloud: Sequence[Sequence[float]]) -> Optional[float]:
    sims = [s for s in (_cosine(query, c) for c in cloud) if s is not None]
    if not sims:
        return None
    return sum(sims) / len(sims)


def _residual_verdict(
    text: str,
    embed_fn: Callable[[str], Optional[Sequence[float]]],
) -> AddressVerdict:
    """Cosine the line vs the to-Ultron and not-to-Ultron exemplar clouds.

    Commits TO_ULTRON only when the to-Ultron mean similarity clears an absolute
    floor AND beats the not-to-Ultron mean by ``_RESIDUAL_MARGIN``. Any embed
    failure / missing cloud -> IGNORE (fail-closed).
    """
    q = _embed(embed_fn, text)
    if q is None:
        return AddressVerdict(ChatAddress.IGNORE, 0.55, "residual: query embed unavailable (fail-closed)")

    pos = [v for v in (_embed(embed_fn, e) for e in TO_ULTRON_EXEMPLARS) if v is not None]
    neg = [v for v in (_embed(embed_fn, e) for e in NOT_TO_ULTRON_EXEMPLARS) if v is not None]
    if not pos or not neg:
        return AddressVerdict(ChatAddress.IGNORE, 0.55, "residual: exemplar embeds unavailable (fail-closed)")

    sim_pos = _mean_sim(q, pos)
    sim_neg = _mean_sim(q, neg)
    if sim_pos is None or sim_neg is None:
        return AddressVerdict(ChatAddress.IGNORE, 0.55, "residual: similarity unavailable (fail-closed)")

    margin = sim_pos - sim_neg
    if sim_pos >= _RESIDUAL_MIN_SIM and margin >= _RESIDUAL_MARGIN:
        conf = max(0.55, min(0.90, 0.55 + margin))
        return AddressVerdict(
            ChatAddress.TO_ULTRON, conf,
            f"residual: to-ultron sim={sim_pos:.2f} vs not={sim_neg:.2f} (margin {margin:.2f})",
        )
    return AddressVerdict(
        ChatAddress.IGNORE,
        max(0.50, min(0.80, 0.50 + abs(margin))),
        f"residual: below margin (to-ultron sim={sim_pos:.2f} vs not={sim_neg:.2f}); fail-closed",
    )


# --------------------------------------------------------------------------- #
# the classifier
# --------------------------------------------------------------------------- #
def classify_chat(
    event: Any,
    *,
    bot_login: str,
    bot_user_id: str,
    streamer_login: str,
    streamer_user_id: str,
    embed_fn: Optional[Callable[[str], Optional[Sequence[float]]]] = None,
) -> AddressVerdict:
    """Classify who an inbound :class:`ChatEvent` is addressing.

    Deterministic-metadata-first cascade (see module docstring); FAIL-CLOSED to
    :attr:`ChatAddress.IGNORE` on any ambiguity, missing field, or error. The
    immutable ``*_user_id`` values are trusted; the spoofable display name is
    never used for resolution.

    ``embed_fn`` (``text -> vector | None``) is the optional residual tie-breaker
    for a bare line with no @mention and no leading name token; when absent, such
    a line IGNOREs (fail-closed). Injected so tests run offline.
    """
    try:
        bot_uid = (bot_user_id or "").strip()
        bot_login_n = _norm_login(bot_login)
        streamer_uid = (streamer_user_id or "").strip()
        streamer_login_n = _norm_login(streamer_login)

        text = getattr(event, "text", "") or ""
        if not isinstance(text, str):
            text = ""
        stripped = text.strip()
        fragments = getattr(event, "fragments", None) or []
        bot_name = getattr(event, "chatter_name", "")  # only used as a login-equivalent token below

        # 0) Empty / whitespace -> IGNORE.
        if not stripped:
            return AddressVerdict(ChatAddress.IGNORE, 0.99, "empty")

        # 1) '!'-prefixed body -> COMMAND (the economy/command router owns it).
        if stripped.startswith("!"):
            return AddressVerdict(ChatAddress.COMMAND, 0.99, "leading '!' command")

        # 2) Reply to the bot (immutable parent_user_id) -> TO_ULTRON, highest conf.
        reply_parent = getattr(event, "reply_parent_user_id", None)
        if isinstance(reply_parent, str) and bot_uid and reply_parent.strip() == bot_uid:
            return AddressVerdict(ChatAddress.TO_ULTRON, 0.99, "reply to bot (parent_user_id)")

        # --- resolve mentions: typed fragments (authoritative user_id) + raw @logins.
        frag_mentions = _fragment_mentions(fragments)
        raw_logins = _raw_at_logins(text)

        # Build the set of mentioned (uid, login) facts. Typed fragments first so
        # their immutable user_id is preferred; raw @logins fill any gaps.
        frag_uids = {uid for (uid, _login) in frag_mentions if uid}
        frag_logins = {login for (_uid, login) in frag_mentions if login}

        mention_present = bool(frag_mentions) or bool(raw_logins)

        # 3) @mention resolving to the BOT -> TO_ULTRON.
        #    - by immutable user_id (a typed fragment), OR
        #    - by login (typed-fragment login OR raw '@login' — login is immutable).
        bot_by_uid = bool(bot_uid) and bot_uid in frag_uids
        bot_by_login = bool(bot_login_n) and (
            bot_login_n in frag_logins or bot_login_n in raw_logins
        )
        if bot_by_uid or bot_by_login:
            why = "user_id" if bot_by_uid else "login"
            return AddressVerdict(ChatAddress.TO_ULTRON, 0.95, f"@mention resolves to bot ({why})")

        # 4) @mention of the STREAMER -> TO_STREAMER.
        streamer_by_uid = bool(streamer_uid) and streamer_uid in frag_uids
        streamer_by_login = bool(streamer_login_n) and (
            streamer_login_n in frag_logins or streamer_login_n in raw_logins
        )
        if streamer_by_uid or streamer_by_login:
            why = "user_id" if streamer_by_uid else "login"
            return AddressVerdict(ChatAddress.TO_STREAMER, 0.92, f"@mention of streamer ({why})")

        # 5) @mention of SOME OTHER (non-bot, non-streamer) user -> TO_OTHER.
        if mention_present:
            return AddressVerdict(ChatAddress.TO_OTHER, 0.90, "@mention of another user")

        # 6) Leading 'ultron'/bot-name/bot-login token (no @mention) -> TO_ULTRON.
        if _leading_name_token(text, bot_login=bot_login_n, bot_name=bot_name):
            return AddressVerdict(ChatAddress.TO_ULTRON, 0.88, "leading bot-name token")

        # 7) RESIDUAL (no @mention, no leading name): semantic tie-break if we have
        #    an embedder; else FAIL-CLOSED to IGNORE.
        if embed_fn is not None:
            return _residual_verdict(stripped, embed_fn)
        return AddressVerdict(ChatAddress.IGNORE, 0.55, "no addressing signal (fail-closed)")

    except Exception as exc:  # noqa: BLE001 — never raise into the sidecar receive loop
        logger.warning("twitch addressing classify failed; failing CLOSED to IGNORE: %s", exc)
        return AddressVerdict(ChatAddress.IGNORE, 0.50, f"classify error (fail-closed): {type(exc).__name__}")
