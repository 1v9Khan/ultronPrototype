"""S10b — batch selection engine for the Twitch chat sidecar.

When the chat-relay turn fires, more messages have arrived than Ultron can or
should answer in one reply. This module distils a raw batch of
:class:`~kenning.twitch.clients.eventsub.ChatEvent` into a small, fair, high-value
shortlist that the LLM turn will actually respond to.

Pipeline (:func:`select_messages`)
----------------------------------
1. **Drop empty** — a blank / whitespace-only / non-``str`` body carries nothing
   to answer; discard it before it can consume a slot.
2. **Near-duplicate dedupe** — chat spams the same line ("W", "LMAO",
   "pog pog pog") from many users. A normalized key (casefold, strip, collapse
   runs of whitespace AND of repeated characters, drop most punctuation) folds
   these together; the FIRST occurrence wins, the rest are dropped. A SimHash over
   token shingles catches the looser near-duplicates ("that was so clean" vs
   "that was sooo clean!!!") that the exact key misses.
3. **Fairness cap** — at most ``per_user_cap`` messages survive per distinct
   ``chatter_user_id`` so one chatter cannot monopolise the reply. Earlier
   (already-deduped) messages from that user are kept.
4. **recently_answered skip** — chatters whose ``chatter_user_id`` is in
   ``recently_answered`` are dropped entirely (we just replied to them; spread the
   attention).
5. **Priority sort** — staff first (moderator > vip > subscriber, read off the
   EventSub ``badges`` ``set_id``), then most-recent (later in the batch = more
   recent), then a small length-quality nudge (a substantive question beats a bare
   "W"). Ties are broken by original batch order so the sort is stable + total.
6. **Cap** — keep at most ``max_messages``.

Fail-safe: :func:`select_messages` NEVER raises. Any malformed event, hostile
field, or internal error degrades to a best-effort / empty :class:`Selection`
rather than propagating into the sidecar turn loop. This mirrors the fail-CLOSED
ethos of :mod:`kenning.audio.intent_gate` — when in doubt, drop the message rather
than risk answering garbage.

ANTICHEAT (BR-P1): pure stdlib (``re`` / ``hashlib`` / ``dataclasses`` /
``logging`` / ``collections``) + the committed :mod:`kenning.twitch.clients`
``ChatEvent`` type. No third-party imports, no desktop surface, no network.
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from kenning.twitch.clients.eventsub import ChatEvent

logger = logging.getLogger("kenning.twitch.selection")

__all__ = ["Selection", "select_messages", "normalized_key", "simhash"]


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class Selection:
    """The chosen shortlist plus accounting of what was discarded.

    ``chosen`` is in final priority order (highest-priority first). ``dropped`` is
    the count of input events that did NOT make the cut for ANY reason (empty,
    duplicate, over the per-user/global cap, recently answered, or malformed).
    ``reason`` is a short human-readable summary of the dominant drop cause, for
    logs / the operator overlay.
    """

    chosen: list[ChatEvent] = field(default_factory=list)
    dropped: int = 0
    reason: str = ""


# --------------------------------------------------------------------------- #
# Dedupe helpers (pure stdlib)
# --------------------------------------------------------------------------- #
# Collapse runs of the same character down to a single one ("soooo" -> "so",
# "!!!" -> "!"). We keep 1 of the run rather than 2 so "good"->"god" — that is
# acceptable for a coarse dedupe key (it only ever *merges* near-dups, never
# splits distinct messages apart in a harmful way).
_REPEAT_RUN_RE = re.compile(r"(.)\1+", re.DOTALL)
_WS_RE = re.compile(r"\s+")
# Strip everything that is not a word char or whitespace for the normalized key
# (emote colons, punctuation, zero-width tricks) so "pog!" == "POG" == "pog...".
_NON_KEY_RE = re.compile(r"[^\w\s]", re.UNICODE)
# Zero-width / bidi characters spammers use to defeat exact-match dedupe.
_ZERO_WIDTH_RE = re.compile(
    "[​‌‍‎‏‪‫‬‭‮﻿]"
)
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# SimHash Hamming distance at/under which two messages are "near-duplicate".
_SIMHASH_BITS = 64
_SIMHASH_NEAR_DISTANCE = 4


def normalized_key(text: object) -> str:
    """Return a coarse dedupe key for ``text`` (casefold + collapse + de-punct).

    Folds the common chat-spam variants of one line onto a single key:
    whitespace runs collapse, repeated-character runs collapse, punctuation and
    zero-width characters are dropped, and the result is casefolded. A non-``str``
    or empty input yields ``""`` (which the caller treats as "no usable key").
    """
    if not isinstance(text, str):
        return ""
    t = _ZERO_WIDTH_RE.sub("", text)
    t = _NON_KEY_RE.sub("", t)
    t = _REPEAT_RUN_RE.sub(r"\1", t)
    t = _WS_RE.sub(" ", t).strip()
    return t.casefold()


def _tokens(text: str) -> list[str]:
    """Lowercased word tokens, with intra-token character runs collapsed.

    "sooo" and "so" both tokenise to "so" so the SimHash treats them as the same
    feature — catching elongations the exact key would also catch, but here at the
    per-token level so a single elongated word inside a longer sentence still folds.
    """
    out: list[str] = []
    for m in _TOKEN_RE.findall(text.casefold()):
        out.append(_REPEAT_RUN_RE.sub(r"\1", m))
    return out


def _shingles(tokens: list[str]) -> list[str]:
    """Feature set for the SimHash: unigrams + adjacent bigrams.

    Bigrams give word-order sensitivity so "rush B now" and "now rush B" are not
    forced together while still collapsing trivial restatements.
    """
    feats: list[str] = list(tokens)
    feats.extend(f"{a}\x00{b}" for a, b in zip(tokens, tokens[1:], strict=False))
    return feats


def simhash(text: object, *, bits: int = _SIMHASH_BITS) -> int:
    """Charikar SimHash of ``text`` over unigram+bigram shingles (pure stdlib).

    Each feature is hashed with BLAKE2b (deterministic, no per-process seed unlike
    ``hash``) into ``bits`` bits; bit ``i`` of the result is 1 iff more features
    voted 1 than 0 at position ``i``. Two near-identical messages produce hashes a
    small Hamming distance apart. Empty / non-``str`` / token-less input -> ``0``.
    """
    if not isinstance(text, str) or not text:
        return 0
    feats = _shingles(_tokens(text))
    if not feats:
        return 0
    counts = [0] * bits
    byte_len = (bits + 7) // 8
    for feat in feats:
        digest = hashlib.blake2b(feat.encode("utf-8", "replace"), digest_size=byte_len).digest()
        h = int.from_bytes(digest, "big")
        for i in range(bits):
            if (h >> i) & 1:
                counts[i] += 1
            else:
                counts[i] -= 1
    value = 0
    for i in range(bits):
        if counts[i] > 0:
            value |= 1 << i
    return value


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# --------------------------------------------------------------------------- #
# Priority scoring
# --------------------------------------------------------------------------- #
# Higher rank = answered first. Mods outrank VIPs outrank subscribers outrank
# everyone else. ``set_id`` is the EventSub badge key.
_BADGE_RANK = {
    "broadcaster": 4,
    "moderator": 3,
    "vip": 2,
    "subscriber": 1,
    "founder": 1,  # founder is an early-subscriber badge — treat as subscriber
}


def _badge_rank(event: ChatEvent) -> int:
    """Highest staff-rank implied by an event's badges (0 = ordinary chatter).

    Reads each badge's ``set_id`` (the EventSub field). Fail-safe: a badges list
    that is not a list of dicts, or any per-badge error, contributes 0 rather than
    raising.
    """
    best = 0
    badges = getattr(event, "badges", None)
    if not isinstance(badges, list):
        return 0
    for badge in badges:
        try:
            if not isinstance(badge, dict):
                continue
            set_id = badge.get("set_id")
            if isinstance(set_id, str):
                best = max(best, _BADGE_RANK.get(set_id.strip().casefold(), 0))
        except Exception:  # noqa: BLE001 — one bad badge never sinks the rank
            continue
    return best


# A short, substantive line scores best; below the floor a message gets no
# length bonus; above the ceiling we stop rewarding length (a wall of text is not
# more answerable than a crisp question).
_LEN_FLOOR = 6
_LEN_CEILING = 160


def _length_quality(event: ChatEvent) -> float:
    """A small [0,1] nudge favouring substantive-but-not-rambling messages.

    A bare "W" (below the floor) gets ~0; a focused question gets a higher score;
    an essay (over the ceiling) is capped so length alone cannot dominate the
    badge/recency signal. A ``+1`` keeps the score in front of recency only as a
    final tie-breaker (the sort weights it least).
    """
    text = getattr(event, "text", "") or ""
    n = len(text.strip())
    if n <= _LEN_FLOOR:
        return 0.0
    if n >= _LEN_CEILING:
        return 0.5
    # Peak quality around a tweet-length question; linear ramp up then gentle taper.
    span = _LEN_CEILING - _LEN_FLOOR
    return min(1.0, (n - _LEN_FLOOR) / span * 1.4)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def select_messages(
    events: Iterable[ChatEvent],
    *,
    max_messages: int = 6,
    per_user_cap: int = 1,
    recently_answered: Iterable[str] | None = frozenset(),
) -> Selection:
    """Distil a raw batch of chat events into a fair, high-value shortlist.

    Args:
        events: the raw batch (any iterable of :class:`ChatEvent`; tolerates
            ``None`` / non-event items, which are dropped).
        max_messages: hard ceiling on the returned ``chosen`` list (clamped to
            ``>= 0``).
        per_user_cap: max surviving messages per distinct ``chatter_user_id``
            (clamped to ``>= 1``).
        recently_answered: ``chatter_user_id`` values to skip this turn (we just
            answered them). ``None`` is treated as the empty set.

    Returns:
        A :class:`Selection`. NEVER raises — on any internal error it returns the
        best partial result assembled so far (or an empty selection) with a
        ``reason`` describing the failure.
    """
    try:
        return _select_messages_impl(
            events,
            max_messages=max_messages,
            per_user_cap=per_user_cap,
            recently_answered=recently_answered,
        )
    except Exception as exc:  # noqa: BLE001 — selection must never sink the turn loop
        logger.warning("select_messages failed; returning empty selection: %s", exc)
        return Selection(chosen=[], dropped=0, reason=f"error: {type(exc).__name__}")


def _select_messages_impl(
    events: Iterable[ChatEvent],
    *,
    max_messages: int,
    per_user_cap: int,
    recently_answered: Iterable[str] | None,
) -> Selection:
    # --- normalize arguments (fail-safe clamps) ------------------------------ #
    max_messages = max(0, int(max_messages)) if isinstance(max_messages, (int, float)) else 6
    per_user_cap = max(1, int(per_user_cap)) if isinstance(per_user_cap, (int, float)) else 1
    try:
        recently = {str(u) for u in (recently_answered or ()) if u is not None}
    except TypeError:
        recently = set()

    # Materialize once; keep original index for stable ordering + recency.
    raw = list(events) if events is not None else []
    total = len(raw)
    if total == 0:
        return Selection(chosen=[], dropped=0, reason="empty batch")

    drop_counts = {
        "malformed": 0,
        "empty": 0,
        "duplicate": 0,
        "recently_answered": 0,
        "per_user_cap": 0,
        "max_cap": 0,
    }

    # --- pass 1: drop empty / malformed, build dedupe state ------------------ #
    # Keep (original_index, event) so recency = larger index and the sort is stable.
    survivors: list[tuple[int, ChatEvent]] = []
    seen_keys: set[str] = set()
    seen_hashes: list[int] = []

    for idx, event in enumerate(raw):
        text = getattr(event, "text", None)
        if not isinstance(text, str) or not text.strip():
            if isinstance(event, ChatEvent):
                drop_counts["empty"] += 1
            else:
                drop_counts["malformed"] += 1
            continue

        key = normalized_key(text)
        if key and key in seen_keys:
            drop_counts["duplicate"] += 1
            continue

        digest = simhash(text)
        if digest and _is_near_dup(digest, seen_hashes):
            drop_counts["duplicate"] += 1
            # Still record the (distinct) exact key so a later identical line dedups.
            if key:
                seen_keys.add(key)
            continue

        if key:
            seen_keys.add(key)
        if digest:
            seen_hashes.append(digest)
        survivors.append((idx, event))

    # --- pass 2: recently_answered skip + per-user fairness cap -------------- #
    per_user: dict[str, int] = {}
    fair: list[tuple[int, ChatEvent]] = []
    for idx, event in survivors:
        uid = _chatter_id(event)
        if uid and uid in recently:
            drop_counts["recently_answered"] += 1
            continue
        # An absent id cannot be fairness-capped or recency-skipped; let it through
        # (it still competes for the global cap). Bucket empty-id events together
        # under a sentinel so a flood of id-less spam can't bypass the cap either.
        bucket = uid or "\x00anon"
        count = per_user.get(bucket, 0)
        if count >= per_user_cap:
            drop_counts["per_user_cap"] += 1
            continue
        per_user[bucket] = count + 1
        fair.append((idx, event))

    # --- pass 3: priority sort ----------------------------------------------- #
    # Sort key (descending): badge rank, then recency (original index), then
    # length-quality, with the original index as the final stable tie-breaker.
    def _priority(item: tuple[int, ChatEvent]) -> tuple:
        idx, event = item
        return (
            _badge_rank(event),
            idx,                      # later in the batch == more recent
            _length_quality(event),
            idx,                      # stable, total order
        )

    fair.sort(key=_priority, reverse=True)

    # --- pass 4: global cap -------------------------------------------------- #
    chosen_pairs = fair[:max_messages]
    drop_counts["max_cap"] = len(fair) - len(chosen_pairs)
    chosen = [event for _idx, event in chosen_pairs]

    dropped = total - len(chosen)
    reason = _summarize_drops(drop_counts, kept=len(chosen), total=total)
    logger.debug(
        "select_messages: %d/%d kept (%s)", len(chosen), total, reason,
    )
    return Selection(chosen=chosen, dropped=dropped, reason=reason)


def _is_near_dup(digest: int, seen_hashes: list[int]) -> bool:
    for prior in seen_hashes:
        if _hamming(digest, prior) <= _SIMHASH_NEAR_DISTANCE:
            return True
    return False


def _chatter_id(event: ChatEvent) -> str:
    uid = getattr(event, "chatter_user_id", "")
    return uid.strip() if isinstance(uid, str) else ""


def _summarize_drops(drop_counts: dict[str, int], *, kept: int, total: int) -> str:
    parts = [f"kept={kept}/{total}"]
    for name in ("duplicate", "per_user_cap", "recently_answered", "empty", "max_cap", "malformed"):
        n = drop_counts.get(name, 0)
        if n:
            parts.append(f"{name}={n}")
    return " ".join(parts)
