"""S12 — channel-point redemptions + reward manager.

Parse ``channel.channel_points_custom_reward_redemption.add`` EventSub
notifications into a typed :class:`RedeemAction`, map a reward *title* to a
logical action kind, and own the reward CRUD + redemption fulfilment/refund
lifecycle over an INJECTED Helix callable so the unit tests run fully offline.

Design invariants (docs/twitch_integration/02_board/MASTER.md SLICE 12 + §"Exactly
-once"):

  * REDEEM TEXT + TITLE ARE UNTRUSTED. Twitch does not reliably AutoMod redeem
    prompts, so ``user_input`` (and the reward title) are a second untrusted
    channel. We carry them verbatim on :class:`RedeemAction` for a downstream
    sanitizer; nothing here interprets them as instructions (BR-10.2). The only
    trust decision made here is the title -> kind MAP supplied by the caller.
  * REFUNDABLE ⇔ QUEUED. Only an UNFULFILLED redemption of a QUEUED reward
    (``should_redemptions_skip_request_queue == false``) can be refunded — a
    skip-queue reward auto-fulfils on Twitch's side and is permanently
    non-refundable. ``parse_redemption`` sets ``refundable`` from exactly this:
    queued reward AND status ``unfulfilled``.
  * IDEMPOTENT INTAKE. EventSub is at-least-once and has no replay; a redemption
    may be re-delivered. :class:`RedemptionDedup` is a bounded LRU that reports a
    repeat ``redemption_id`` so the caller acts on each redemption exactly once.
  * REFUND == CANCELED (ONE L). ``update_redemption_status`` accepts only the two
    terminal states Twitch allows from UNFULFILLED — ``FULFILLED`` (grant) or
    ``CANCELED`` (refund). The status string is ``CANCELED`` (Twitch's spelling,
    one L); ``CANCELLED`` is rejected loudly so a typo can never silently fail to
    refund.
  * 50-CAP AWARENESS. A channel may own at most 50 *manageable* custom rewards;
    :meth:`RewardManager.create_reward` refuses past that ceiling rather than
    issue a doomed Helix POST, and forces ``is_user_input_required``/queueing
    consistent with a refundable action.

ANTICHEAT (BR-P1): stdlib only (logging / dataclasses / collections / typing).
The Helix transport is an injected ``Callable`` — no urllib/requests/network and
no desktop/input/screen libs are imported here.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

logger = logging.getLogger("kenning.twitch.channel_points")

__all__ = [
    "RedeemAction",
    "RedemptionDedup",
    "RewardManager",
    "ChannelPointsError",
    "parse_redemption",
    "STATUS_FULFILLED",
    "STATUS_CANCELED",
    "STATUS_UNFULFILLED",
    "MAX_MANAGEABLE_REWARDS",
    "CUSTOM_KIND",
    "HelixCall",
]

# Twitch redemption status strings. CANCELED is spelled with ONE L (Twitch's
# spelling) and is the REFUND path; FULFILLED grants the redemption.
STATUS_FULFILLED = "FULFILLED"
STATUS_CANCELED = "CANCELED"  # one L — the refund terminal state
STATUS_UNFULFILLED = "UNFULFILLED"

# A redemption may only be moved from UNFULFILLED to one of these terminal states.
_TERMINAL_STATUSES = (STATUS_FULFILLED, STATUS_CANCELED)

# Twitch caps a channel at 50 custom rewards the app can manage.
MAX_MANAGEABLE_REWARDS = 50

# Fallback action kind for a recognised redemption whose title is not in the map.
CUSTOM_KIND = "custom"

# The injected Helix transport: helix(method, path, body) -> parsed-json dict.
# ``method`` is an HTTP verb ("POST"/"PATCH"/"GET"/"DELETE"); ``path`` is the
# Helix path (e.g. "/channel_points/custom_rewards"); ``body`` is the request
# payload dict (query params + json merged by the caller's transport) or None.
HelixCall = Callable[[str, str, Optional[Mapping[str, Any]]], Mapping[str, Any]]


class ChannelPointsError(RuntimeError):
    """A non-recoverable channel-points fault (bad status, cap exceeded, transport).

    Carries an optional ``status`` (HTTP-ish) and raw ``detail`` for the caller's
    audit trail. Raised LOUD for programmer/permission errors (a bad terminal
    status, the 50-cap, a None Helix response) so they surface rather than being
    silently swallowed (BR-2.5).
    """

    def __init__(self, message: str, *, status: Optional[int] = None, detail: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


# --------------------------------------------------------------------------- #
# RedeemAction + payload parsing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RedeemAction:
    """A parsed channel-point redemption (the to-handler unit).

    ``user_input`` and ``reward_title`` are UNTRUSTED (Twitch does not reliably
    AutoMod redeem prompts) — carried verbatim for a downstream sanitizer, never
    interpreted here. ``kind`` is the logical action mapped from the reward title
    via the caller's ``title_map`` (or :data:`CUSTOM_KIND` for an unmapped but
    structurally-valid redemption). ``refundable`` is True only when the
    redemption is an UNFULFILLED redemption of a QUEUED reward.
    """

    kind: str
    redemption_id: str
    reward_id: str
    reward_title: str
    user_id: str
    user_login: str
    user_name: str
    user_input: str
    status: str
    refundable: bool
    broadcaster_user_id: str = ""
    cost: int = 0
    raw: dict = field(default_factory=dict)


def _coerce_str(value: Any) -> str:
    """Coerce any JSON scalar to ``str`` (None -> "")."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _coerce_int(value: Any) -> int:
    """Best-effort int coercion; 0 on anything non-numeric. Never raises."""
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.strip():
            return int(value.strip())
    except (TypeError, ValueError):
        return 0
    return 0


def _locate_redemption_event(payload: Any) -> Optional[dict]:
    """Find the redemption ``event`` dict inside any of the accepted envelopes.

    Accepts the full notification (``{"metadata":..., "payload":{"event":{...}}}``),
    a bare ``{"event":{...}}`` payload, or the bare event dict itself. When a
    subscription type is present it is verified to be the redemption-add type so a
    different EventSub message can never be mis-parsed. Returns ``None`` (the
    caller logs) on anything structurally invalid — fail-safe.
    """
    if not isinstance(payload, dict):
        return None
    expected_type = "channel.channel_points_custom_reward_redemption.add"

    meta = payload.get("metadata")
    if isinstance(meta, dict):
        sub_type = meta.get("subscription_type")
        if isinstance(sub_type, str) and sub_type and sub_type != expected_type:
            return None

    inner = payload.get("payload")
    if isinstance(inner, dict):
        sub = inner.get("subscription")
        if isinstance(sub, dict):
            stype = sub.get("type")
            if isinstance(stype, str) and stype and stype != expected_type:
                return None
        event = inner.get("event")
        if isinstance(event, dict):
            return event

    event = payload.get("event")
    if isinstance(event, dict):
        return event

    # Bare event dict — accept only if it looks like a redemption.
    if "reward" in payload and ("user_id" in payload or "user_login" in payload):
        return payload
    return None


def parse_redemption(
    event_payload: dict,
    *,
    title_map: Mapping[str, str],
) -> Optional[RedeemAction]:
    """Parse a redemption-add EventSub payload into a :class:`RedeemAction`.

    ``title_map`` maps a reward *title* to a logical action ``kind`` (e.g.
    ``{'Spin the Wheel': 'wheel', 'Slots': 'slots', 'Alert': 'alert',
    'Speak to Team': 'speak_to_team'}``). A title not in the map yields
    ``kind == CUSTOM_KIND`` (so the caller still sees a structurally-valid
    redemption it can ignore or handle generically) — a redemption is never
    dropped merely for an unknown title.

    ``refundable`` is True only for an UNFULFILLED redemption of a QUEUED reward
    (``should_redemptions_skip_request_queue`` falsy): a skip-queue reward
    auto-fulfils server-side and can never be refunded.

    Returns ``None`` (logged) on a structurally invalid / non-redemption payload —
    fail-safe so a hostile or malformed notification never raises into the
    receive loop. ``user_input``/``reward_title`` are kept verbatim (UNTRUSTED).
    """
    try:
        event = _locate_redemption_event(event_payload)
        if event is None:
            logger.warning("channel_points: not a redemption payload — dropping")
            return None

        reward = event.get("reward")
        if not isinstance(reward, dict):
            reward = {}
        reward_title = _coerce_str(reward.get("title"))
        reward_id = _coerce_str(reward.get("id"))

        title_map = title_map if isinstance(title_map, Mapping) else {}
        kind = title_map.get(reward_title, CUSTOM_KIND) if reward_title else CUSTOM_KIND
        if not isinstance(kind, str) or not kind:
            kind = CUSTOM_KIND

        # Status governs refundability. The redemption-add event carries
        # status="unfulfilled"; default to UNFULFILLED when absent (the add event
        # is, by definition, a fresh unfulfilled redemption).
        status_raw = _coerce_str(event.get("status")) or STATUS_UNFULFILLED
        status = status_raw.upper()

        # QUEUED iff the reward does NOT skip the request queue. The flag may be
        # absent on the redemption event itself (it lives on the reward object);
        # treat an explicit True as skip-queue, anything else as queued so the
        # SAFE default is "potentially refundable" (we never auto-refund here —
        # the caller decides — but we must not mislabel a queued reward as final).
        skip_queue = bool(reward.get("should_redemptions_skip_request_queue", False))
        is_queued = not skip_queue
        refundable = is_queued and status == STATUS_UNFULFILLED

        action = RedeemAction(
            kind=kind,
            redemption_id=_coerce_str(event.get("id")),
            reward_id=reward_id,
            reward_title=reward_title,
            user_id=_coerce_str(event.get("user_id")),
            user_login=_coerce_str(event.get("user_login")),
            user_name=_coerce_str(event.get("user_name")),
            user_input=_coerce_str(event.get("user_input")),
            status=status,
            refundable=refundable,
            broadcaster_user_id=_coerce_str(event.get("broadcaster_user_id")),
            cost=_coerce_int(reward.get("cost")),
            raw=event,
        )
        logger.info(
            "channel_points: redemption id=%s kind=%s queued=%s refundable=%s",
            action.redemption_id, action.kind, is_queued, action.refundable,
        )
        return action
    except Exception as exc:  # noqa: BLE001 — never raise into the receive loop
        logger.warning("channel_points: redemption parse failed: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# At-least-once dedup (EventSub may re-deliver a redemption)
# --------------------------------------------------------------------------- #
class RedemptionDedup:
    """Bounded LRU of seen ``redemption_id`` values for at-least-once delivery.

    EventSub re-delivers and has no replay; the manager must act on each
    redemption exactly once. :meth:`seen` returns ``True`` if the id was already
    recorded (a duplicate to drop) and otherwise records it, evicting the oldest
    entry once ``maxsize`` is exceeded. An empty / ``None`` id is treated as
    never-seen (fail-open) so a malformed event is not silently coalesced with an
    unrelated one.
    """

    def __init__(self, maxsize: int = 4096) -> None:
        if maxsize < 1:
            raise ValueError("RedemptionDedup maxsize must be >= 1")
        self._maxsize = int(maxsize)
        self._seen: "OrderedDict[str, None]" = OrderedDict()

    def seen(self, redemption_id: Optional[str]) -> bool:
        """True if ``redemption_id`` was already recorded (a duplicate to drop)."""
        if not redemption_id:
            return False  # cannot dedup an absent id — let it through (fail-open)
        if redemption_id in self._seen:
            self._seen.move_to_end(redemption_id)
            return True
        self._seen[redemption_id] = None
        if len(self._seen) > self._maxsize:
            self._seen.popitem(last=False)  # evict oldest
        return False

    def __len__(self) -> int:
        return len(self._seen)

    def __contains__(self, redemption_id: object) -> bool:
        return isinstance(redemption_id, str) and redemption_id in self._seen


# --------------------------------------------------------------------------- #
# Reward CRUD + redemption status manager (injected Helix callable)
# --------------------------------------------------------------------------- #
class RewardManager:
    """Owns custom-reward CRUD and redemption fulfilment over an injected Helix.

    The transport is a single callable ``helix(method, path, body) -> dict`` so
    tests drive it with a mock — no network, no creds. Every Helix response is
    expected to be the parsed Twitch envelope ``{"data": [...]}``; a ``None`` /
    non-dict response is treated as a LOUD failure (a write whose effect is
    unknown must never be reported as success — BR-2.5).

    All methods scope to ``broadcaster_id`` (the channel that owns the rewards);
    Helix requires the same id as both the authenticated user and the
    ``broadcaster_id`` query param.
    """

    def __init__(self, broadcaster_id: str, helix: HelixCall) -> None:
        if not broadcaster_id:
            raise ValueError("broadcaster_id is required")
        if not callable(helix):
            raise ValueError("helix must be callable")
        self._broadcaster_id = str(broadcaster_id)
        self._helix = helix

    # ---- helpers ---------------------------------------------------------- #
    def _call(self, method: str, path: str, body: Optional[Mapping[str, Any]]) -> dict:
        """Invoke the injected Helix transport and normalise/validate the result."""
        try:
            resp = self._helix(method, path, body)
        except ChannelPointsError:
            raise
        except Exception as exc:  # noqa: BLE001 — surface a transport fault LOUD
            logger.error("channel_points: helix %s %s transport error: %s", method, path, exc)
            raise ChannelPointsError(
                f"helix {method} {path} transport error: {exc}", detail=str(exc)
            ) from exc
        if resp is None or not isinstance(resp, Mapping):
            logger.error("channel_points: helix %s %s returned %r", method, path, resp)
            raise ChannelPointsError(
                f"helix {method} {path} returned no/!mapping response",
                detail=repr(resp),
            )
        return dict(resp)

    @staticmethod
    def _data_list(resp: Mapping[str, Any]) -> list:
        data = resp.get("data")
        return data if isinstance(data, list) else []

    # ---- reward CRUD ------------------------------------------------------ #
    def list_rewards(self, *, only_manageable: bool = True) -> list[dict]:
        """GET the channel's custom rewards (optionally only app-manageable ones).

        Used both directly and for the 50-cap pre-check. Returns the ``data`` list
        (possibly empty); a missing/!list ``data`` is normalised to ``[]``.
        """
        body = {
            "broadcaster_id": self._broadcaster_id,
            "only_manageable_rewards": "true" if only_manageable else "false",
        }
        resp = self._call("GET", "/channel_points/custom_rewards", body)
        rewards = [r for r in self._data_list(resp) if isinstance(r, dict)]
        logger.info("channel_points: list_rewards manageable=%s n=%d", only_manageable, len(rewards))
        return rewards

    def manageable_reward_count(self) -> int:
        """Number of app-manageable custom rewards (for the 50-cap pre-check)."""
        return len(self.list_rewards(only_manageable=True))

    def create_reward(
        self,
        title: str,
        cost: int,
        *,
        refundable: bool = False,
        prompt: str = "",
        is_user_input_required: Optional[bool] = None,
        enforce_cap: bool = True,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        """POST a new custom reward, enforcing the refundable⇔queued contract.

        Args:
            title: the reward title (1..45 chars per Twitch). Required.
            cost: the channel-point cost (>= 1). Required.
            refundable: when True, the reward is created QUEUED
                (``should_redemptions_skip_request_queue=false``) so an
                UNFULFILLED redemption can later be CANCELED to refund. When
                False the reward skips the queue and is permanently non-refundable.
            prompt: optional viewer-facing prompt (UNTRUSTED if surfaced; carried
                verbatim).
            is_user_input_required: override; defaults to True iff a ``prompt`` is
                given (Twitch requires this True to collect ``user_input``).
            enforce_cap: when True (default), pre-check the 50 manageable-reward
                ceiling and refuse rather than issue a doomed POST.
            extra: additional Helix fields (e.g. ``background_color``,
                ``max_per_stream``) merged into the body; cannot override the
                queue/cap-driven keys.

        Raises :class:`ChannelPointsError` past the 50-cap, or :class:`ValueError`
        on a missing title / non-positive cost.
        """
        if not title or not title.strip():
            raise ValueError("reward title is required")
        if not isinstance(cost, int) or cost < 1:
            raise ValueError("cost must be an int >= 1")

        if enforce_cap:
            count = self.manageable_reward_count()
            if count >= MAX_MANAGEABLE_REWARDS:
                logger.error(
                    "channel_points: create_reward refused — 50-cap reached (%d/%d)",
                    count, MAX_MANAGEABLE_REWARDS,
                )
                raise ChannelPointsError(
                    f"manageable-reward cap reached ({count}/{MAX_MANAGEABLE_REWARDS})",
                    status=409,
                )

        if is_user_input_required is None:
            is_user_input_required = bool(prompt)

        body: dict[str, Any] = {}
        if extra:
            # extra cannot override the contract-critical keys set below.
            body.update({k: v for k, v in extra.items()})

        body.update(
            {
                "broadcaster_id": self._broadcaster_id,
                "title": title,
                "cost": int(cost),
                "is_user_input_required": bool(is_user_input_required),
                # Refundable ⇔ QUEUED: a refundable reward must NOT skip the queue,
                # because only UNFULFILLED queued redemptions can be CANCELED/refunded.
                "should_redemptions_skip_request_queue": (not refundable),
            }
        )
        if prompt:
            body["prompt"] = prompt

        resp = self._call("POST", "/channel_points/custom_rewards", body)
        data = self._data_list(resp)
        created = data[0] if data and isinstance(data[0], dict) else {}
        logger.info(
            "channel_points: create_reward title=%r cost=%d refundable=%s queued=%s id=%s",
            title, cost, refundable, refundable, created.get("id", ""),
        )
        return created

    def delete_reward(self, reward_id: str) -> bool:
        """DELETE a custom reward the app owns. Returns True on success.

        A 404-style "does not exist" is treated as already-deleted (idempotent):
        the transport may surface it by raising :class:`ChannelPointsError` with a
        404 status, which we swallow to True. Any other fault propagates.
        """
        if not reward_id:
            raise ValueError("reward_id is required")
        try:
            self._call(
                "DELETE",
                "/channel_points/custom_rewards",
                {"broadcaster_id": self._broadcaster_id, "id": str(reward_id)},
            )
        except ChannelPointsError as exc:
            if exc.status == 404:
                logger.info("channel_points: delete_reward already-gone id=%s", reward_id)
                return True
            raise
        logger.info("channel_points: delete_reward id=%s", reward_id)
        return True

    # ---- redemption status ----------------------------------------------- #
    def update_redemption_status(
        self,
        reward_id: str,
        redemption_id: str,
        status: str,
    ) -> dict:
        """PATCH a redemption to a terminal status: ``FULFILLED`` or ``CANCELED``.

        ``CANCELED`` (one L) is the REFUND path — Twitch returns the cost to the
        viewer. ``FULFILLED`` grants it. Any other status (including the common
        ``CANCELLED`` typo or the non-terminal ``UNFULFILLED``) is rejected with a
        :class:`ChannelPointsError` so a typo can never silently fail to refund.

        Returns the updated redemption dict from Helix.
        """
        if not reward_id:
            raise ValueError("reward_id is required")
        if not redemption_id:
            raise ValueError("redemption_id is required")
        norm = _coerce_str(status).strip().upper()
        if norm not in _TERMINAL_STATUSES:
            logger.error(
                "channel_points: refused invalid redemption status %r (want %s)",
                status, _TERMINAL_STATUSES,
            )
            raise ChannelPointsError(
                f"invalid redemption status {status!r}; "
                f"must be one of {_TERMINAL_STATUSES} (CANCELED has one L)",
                status=400,
            )
        body = {
            "broadcaster_id": self._broadcaster_id,
            "reward_id": str(reward_id),
            "id": str(redemption_id),
            "status": norm,
        }
        resp = self._call("PATCH", "/channel_points/custom_rewards/redemptions", body)
        data = self._data_list(resp)
        updated = data[0] if data and isinstance(data[0], dict) else {}
        logger.info(
            "channel_points: redemption %s -> %s%s",
            redemption_id, norm, " (refund)" if norm == STATUS_CANCELED else "",
        )
        return updated

    def fulfill_redemption(self, reward_id: str, redemption_id: str) -> dict:
        """Grant a redemption (status -> ``FULFILLED``)."""
        return self.update_redemption_status(reward_id, redemption_id, STATUS_FULFILLED)

    def refund_redemption(self, reward_id: str, redemption_id: str) -> dict:
        """Refund a redemption (status -> ``CANCELED``, one L — returns the cost)."""
        return self.update_redemption_status(reward_id, redemption_id, STATUS_CANCELED)
