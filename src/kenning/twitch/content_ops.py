"""S12 — content-ops: Stream Markers / clips + a chat-rate highlight scorer.

This module creates **Stream Markers** and **Clips** through Twitch's OWN Helix
API and scores chat-rate spikes to suggest *when* to mark a highlight. It performs
**ZERO game video / screen capture** of any kind — the clip's video source is
Twitch's own VOD, populated server-side. (See MASTER.md SLICE 8/12 / line 69:
"multimodal highlight scorer + instant Stream Marker live + deferred post-stream
clip ... ZERO game video/screen capture anywhere ... clip video source is Twitch's
own VOD.")

The Helix transport is INJECTED as a single callable:

    helix(method: str, path: str, body: Optional[Mapping]) -> dict

so tests run fully OFFLINE with a mock — no real network / creds / models. The
callable is expected to return Twitch's parsed JSON envelope (typically
``{"data": [ ... ]}``). It MAY raise on transport/HTTP failure; this module
catches that and returns a STRUCTURED failure dict (it never lets a content-ops
convenience crash the caller — content-ops is "low-risk delight", MASTER line 133).

Public API:
  * ``create_stream_marker(broadcaster_id, *, description='', helix) -> dict``
      POST /streams/markers. ``description`` is truncated to Twitch's 140-char max.
      Idempotent-tolerant (a duplicate marker is harmless; Twitch simply records a
      new one). Returns the marker dict on success or a structured failure.
  * ``create_clip(broadcaster_id, *, helix) -> dict``
      POST /clips. The clip's ``vod_offset`` populates only after a server-side
      delay (~2–3 min, MASTER line 69), so this returns the clip ``id`` +
      ``edit_url`` immediately and the caller polls/queues the rest later. Returns
      a structured failure on error.
  * ``HighlightScorer(window_seconds)`` — a pure-stdlib, deterministic chat-rate
      spike heuristic. ``note_message(ts)`` records a chat timestamp; ``score(now)``
      returns the recent-rate / baseline-rate ratio over a rolling window;
      ``should_mark(now, threshold)`` is the boolean gate. Deterministic given the
      timestamps passed in (no wall clock read inside).

ANTICHEAT (BR-P1): stdlib only here (``logging`` / ``collections`` / ``typing``).
NO requests/aiohttp/websockets/transformers/torch; NO pyautogui/mss/pynput/
desktop-automation; NO Win32 ban-class APIs; **NO screen/video capture anywhere**.
SCAN-PROOF: this module imports no capture libs — the only imports are the three
stdlib names listed above; ``mss``/``pyautogui``/``pywinauto``/``pynput``/``cv2``/
``PIL``/``mensa`` and every other capture/automation surface are ABSENT by
construction (asserted by ``tests/twitch/test_content_ops.py::test_no_capture_imports``).
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Callable, Deque, Mapping, Optional

logger = logging.getLogger("kenning.twitch.content_ops")

__all__ = [
    "Helix",
    "HighlightScorer",
    "MARKER_DESCRIPTION_MAX",
    "create_clip",
    "create_stream_marker",
]

# An injected Helix transport: (method, path, body) -> parsed JSON dict.
# ``body`` is None for bodyless requests. Tests inject a deterministic mock.
Helix = Callable[[str, str, Optional[Mapping[str, Any]]], Mapping[str, Any]]

# Twitch caps a Stream Marker description at 140 characters; longer is rejected.
MARKER_DESCRIPTION_MAX = 140


def _failure(action: str, error: Exception | str, **extra: Any) -> dict[str, Any]:
    """Build a STRUCTURED failure dict (never raises out of a content-ops call).

    Shape: ``{"ok": False, "action": ..., "error": "<msg>", "error_type": "<cls>",
    **extra}``. Callers branch on ``result["ok"]``; the failure is logged here so a
    swallowed error still leaves an audit trail (BR-2.5: stops are loud in the log).
    """
    if isinstance(error, Exception):
        msg = str(error) or error.__class__.__name__
        etype = error.__class__.__name__
    else:
        msg = str(error)
        etype = "ValueError"
    out: dict[str, Any] = {"ok": False, "action": action, "error": msg, "error_type": etype}
    out.update(extra)
    return out


def _first_data_item(envelope: Any) -> Optional[Mapping[str, Any]]:
    """Pull the first element of a Helix ``{"data": [ ... ]}`` envelope.

    Tolerant of either the full envelope or a bare list; returns None when there
    is no usable item (so the caller reports a structured 'empty response' failure
    rather than indexing into nothing).
    """
    data: Any = None
    if isinstance(envelope, Mapping):
        data = envelope.get("data")
    elif isinstance(envelope, (list, tuple)):
        data = envelope
    if isinstance(data, (list, tuple)) and data:
        first = data[0]
        if isinstance(first, Mapping):
            return first
    # Some endpoints (or mocks) may return the object directly.
    if isinstance(envelope, Mapping) and "data" not in envelope and envelope:
        return envelope
    return None


def create_stream_marker(
    broadcaster_id: str,
    *,
    description: str = "",
    helix: Helix,
) -> dict[str, Any]:
    """Create a Stream Marker on the live broadcast via POST /streams/markers.

    Args:
        broadcaster_id: the channel to mark (the running broadcast).
        description: an optional note; TRUNCATED to ``MARKER_DESCRIPTION_MAX``
            (140) characters to satisfy Twitch's limit (over-length is otherwise a
            400). ``None`` is treated as empty.
        helix: the injected ``(method, path, body) -> dict`` transport.

    Returns:
        On success: ``{"ok": True, "action": "create_stream_marker",
        "idempotent": False, "marker": <marker dict>, "id": <id>, ...}``.
        On any failure (bad input, transport raise, empty/204 response): a
        STRUCTURED failure dict (``ok`` False) — this NEVER raises, because a
        marker is a best-effort convenience and must not crash the stream loop.

    Marker creation is idempotent-tolerant: Twitch happily records a second marker
    if the command double-fires, which is harmless (markers are advisory), so this
    does not attempt server-side dedupe; callers debounce upstream if desired.
    """
    if not broadcaster_id:
        logger.error("create_stream_marker: broadcaster_id is required")
        return _failure("create_stream_marker", "broadcaster_id is required")
    if not callable(helix):
        logger.error("create_stream_marker: helix transport is not callable")
        return _failure("create_stream_marker", "helix transport must be callable")

    desc = (description or "")
    truncated = len(desc) > MARKER_DESCRIPTION_MAX
    if truncated:
        logger.info(
            "create_stream_marker: description %d>%d chars; truncating",
            len(desc), MARKER_DESCRIPTION_MAX,
        )
        desc = desc[:MARKER_DESCRIPTION_MAX]

    body: dict[str, Any] = {"user_id": str(broadcaster_id)}
    if desc:
        body["description"] = desc

    try:
        envelope = helix("POST", "/streams/markers", body)
    except Exception as exc:  # noqa: BLE001 - transport faults become structured failures, never crashes
        logger.error("create_stream_marker: helix transport failed: %s", exc)
        return _failure("create_stream_marker", exc, broadcaster_id=str(broadcaster_id))

    marker = _first_data_item(envelope)
    if marker is None:
        logger.error(
            "create_stream_marker: empty/unrecognized Helix response (channel may be offline)"
        )
        return _failure(
            "create_stream_marker",
            "empty response (is the channel live?)",
            broadcaster_id=str(broadcaster_id),
        )

    marker_id = marker.get("id", "")
    logger.info(
        "create_stream_marker: marked broadcaster=%s id=%s position=%s%s",
        broadcaster_id, marker_id, marker.get("position_seconds"),
        " (description truncated)" if truncated else "",
    )
    return {
        "ok": True,
        "action": "create_stream_marker",
        "idempotent": False,
        "id": marker_id,
        "marker": dict(marker),
        "description_truncated": truncated,
        "broadcaster_id": str(broadcaster_id),
    }


def create_clip(
    broadcaster_id: str,
    *,
    helix: Helix,
) -> dict[str, Any]:
    """Create a Clip of the live broadcast via POST /clips.

    The clip's ``vod_offset`` (and final playable URL) populate only AFTER a
    server-side processing delay (~2–3 min, MASTER line 69). This call therefore
    returns the clip ``id`` and ``edit_url`` IMMEDIATELY; a deferred batch should
    re-query the clip later to harvest ``vod_offset`` / the public URL.

    Args:
        broadcaster_id: the channel to clip (the running broadcast).
        helix: the injected ``(method, path, body) -> dict`` transport.

    Returns:
        On success: ``{"ok": True, "action": "create_clip", "id": <clip id>,
        "edit_url": <edit url>, "vod_offset_pending": True, "clip": <clip dict>}``.
        On failure (bad input, transport raise, empty response — e.g. the channel
        is offline, which Helix answers 404): a STRUCTURED failure dict. Never
        raises.
    """
    if not broadcaster_id:
        logger.error("create_clip: broadcaster_id is required")
        return _failure("create_clip", "broadcaster_id is required")
    if not callable(helix):
        logger.error("create_clip: helix transport is not callable")
        return _failure("create_clip", "helix transport must be callable")

    # POST /clips takes broadcaster_id as a query param and no JSON body.
    try:
        envelope = helix("POST", f"/clips?broadcaster_id={broadcaster_id}", None)
    except Exception as exc:  # noqa: BLE001 - transport faults become structured failures, never crashes
        logger.error("create_clip: helix transport failed: %s", exc)
        return _failure("create_clip", exc, broadcaster_id=str(broadcaster_id))

    clip = _first_data_item(envelope)
    if clip is None:
        logger.error(
            "create_clip: empty/unrecognized Helix response (channel may be offline)"
        )
        return _failure(
            "create_clip",
            "empty response (is the channel live?)",
            broadcaster_id=str(broadcaster_id),
        )

    clip_id = clip.get("id", "")
    edit_url = clip.get("edit_url", "")
    if not clip_id:
        logger.error("create_clip: Helix response missing clip id")
        return _failure(
            "create_clip", "response missing clip id", broadcaster_id=str(broadcaster_id)
        )
    logger.info(
        "create_clip: created clip id=%s edit_url=%s (vod_offset populates after ~2-3 min)",
        clip_id, edit_url,
    )
    return {
        "ok": True,
        "action": "create_clip",
        "id": clip_id,
        "edit_url": edit_url,
        # vod_offset / public URL are NOT available yet — a deferred batch harvests them.
        "vod_offset_pending": True,
        "clip": dict(clip),
        "broadcaster_id": str(broadcaster_id),
    }


class HighlightScorer:
    """A deterministic chat-rate spike heuristic — "is chat blowing up right now?".

    The model: a moment worth marking is one where chat activity in the most recent
    sub-window jumps far above its rolling baseline. We keep a rolling window of the
    last ``window_seconds`` of message timestamps (passed in by the caller — this
    class reads NO wall clock, so it is fully deterministic for a given timestamp
    sequence and trivially testable).

    :meth:`score` returns the RATIO of the recent (most-recent fifth of the window)
    message rate to the baseline (the rest of the window) rate. A ratio of ~1.0 is
    steady chat; a spike (a clip-worthy moment) pushes it well above 1.0. With too
    little baseline history the score is damped toward 1.0 so a cold start does not
    fire. :meth:`should_mark` thresholds that ratio.

    Pure stdlib (``collections.deque``); no network, no model, no capture.
    """

    # The recent sub-window is the most-recent fraction of the full window; the
    # rest is the baseline it's compared against.
    _RECENT_FRACTION = 0.2
    # Minimum baseline messages before a spike can fire (cold-start guard).
    _MIN_BASELINE_MESSAGES = 3

    def __init__(self, window_seconds: float) -> None:
        if not isinstance(window_seconds, (int, float)) or window_seconds <= 0:
            raise ValueError("window_seconds must be a positive number")
        self._window = float(window_seconds)
        self._recent_window = self._window * self._RECENT_FRACTION
        self._events: Deque[float] = deque()
        self._last_ts: Optional[float] = None

    def note_message(self, ts: float) -> None:
        """Record one chat message at timestamp ``ts`` (caller-supplied seconds).

        Timestamps are expected roughly monotonic; an out-of-order (older) ts is
        clamped to the latest seen so the rolling window stays well-formed (chat
        clients can deliver slightly out of order — we degrade gracefully instead
        of corrupting the deque order).
        """
        try:
            t = float(ts)
        except (TypeError, ValueError):
            logger.warning("HighlightScorer.note_message: non-numeric ts=%r ignored", ts)
            return
        if self._last_ts is not None and t < self._last_ts:
            # Out-of-order delivery: clamp to keep the deque monotonic.
            t = self._last_ts
        self._last_ts = t
        self._events.append(t)
        self._evict(t)

    def _evict(self, now: float) -> None:
        """Drop events older than ``window_seconds`` before ``now``."""
        cutoff = now - self._window
        ev = self._events
        while ev and ev[0] < cutoff:
            ev.popleft()

    def score(self, now: float) -> float:
        """Recent-rate / baseline-rate ratio over the rolling window at ``now``.

        Returns 1.0 for steady or insufficient-history chat; > 1.0 for a spike.
        Never raises and never reads a wall clock — deterministic in ``now`` and
        the recorded timestamps.
        """
        try:
            now_f = float(now)
        except (TypeError, ValueError):
            logger.warning("HighlightScorer.score: non-numeric now=%r -> neutral", now)
            return 1.0
        self._evict(now_f)
        if not self._events:
            return 1.0

        recent_cutoff = now_f - self._recent_window
        recent = 0
        baseline = 0
        for t in self._events:
            if t >= recent_cutoff:
                recent += 1
            else:
                baseline += 1

        # Cold-start guard: without enough baseline history a couple of early
        # messages must not read as a "spike".
        if baseline < self._MIN_BASELINE_MESSAGES:
            return 1.0

        baseline_span = self._window - self._recent_window
        recent_rate = recent / self._recent_window if self._recent_window > 0 else 0.0
        baseline_rate = baseline / baseline_span if baseline_span > 0 else 0.0
        if baseline_rate <= 0.0:
            # Recent activity with a zero baseline -> a clear spike; report a large
            # but finite ratio proportional to the recent burst.
            return float(max(1.0, recent_rate * self._recent_window))
        return recent_rate / baseline_rate

    def should_mark(self, now: float, threshold: float = 2.0) -> bool:
        """True when :meth:`score` at ``now`` meets/exceeds ``threshold``.

        ``threshold`` defaults to 2.0 (recent chat twice its baseline rate). A
        non-positive threshold is rejected so a misconfiguration can't fire on
        every message.
        """
        if not isinstance(threshold, (int, float)) or threshold <= 0:
            raise ValueError("threshold must be a positive number")
        return self.score(now) >= float(threshold)
