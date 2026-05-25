"""Priority-banded presentation scheduler with environment-adaptive cadence.

Adapted from cline's ``TaskPresentationScheduler`` pattern (Apache 2.0;
see ``THIRD_PARTY_NOTICES.md``). Ultron's variant maps cline's
text / reasoning / tool_calls priorities onto TTS-appropriate bands:

* :attr:`PresentationPriority.IMMEDIATE` — sentence-boundary chunks
  + tool transitions; emitted with zero debounce.
* :attr:`PresentationPriority.NORMAL` — mid-sentence chunks; emitted
  on per-environment cadence (60 ms local PortAudio, 200 ms Bluetooth
  / remote).
* :attr:`PresentationPriority.LOW` — reasoning text / verbose logs;
  emitted on the longest debounce, OR dropped entirely when
  ``enable_thinking=False`` is in effect.

The audio profile (local vs Bluetooth) is detected via
:func:`detect_audio_profile` which inspects sounddevice metadata when
available; callers may inject the profile explicitly to bypass
detection (test hook).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Optional

LOGGER = logging.getLogger(__name__)


class PresentationPriority(str, Enum):
    """Per-chunk priority band."""

    IMMEDIATE = "immediate"
    NORMAL = "normal"
    LOW = "low"


class AudioProfile(str, Enum):
    """Audio environment that drives the default cadence map."""

    LOCAL = "local"
    REMOTE = "remote"
    BLUETOOTH = "bluetooth"


@dataclass(frozen=True)
class Cadence:
    """Per-environment cadence map (milliseconds)."""

    immediate_ms: int = 0
    normal_ms: int = 60
    low_ms: int = 200

    def for_priority(self, priority: PresentationPriority) -> int:
        if priority is PresentationPriority.IMMEDIATE:
            return self.immediate_ms
        if priority is PresentationPriority.LOW:
            return self.low_ms
        return self.normal_ms


#: Cadence defaults per audio profile. Bluetooth uses larger windows to
#: absorb the codec's added latency without producing audible jitter.
DEFAULT_CADENCE_BY_PROFILE: Mapping[AudioProfile, Cadence] = {
    AudioProfile.LOCAL: Cadence(immediate_ms=0, normal_ms=60, low_ms=200),
    AudioProfile.REMOTE: Cadence(immediate_ms=20, normal_ms=200, low_ms=400),
    AudioProfile.BLUETOOTH: Cadence(immediate_ms=10, normal_ms=120, low_ms=300),
}


def detect_audio_profile(device_name: Optional[str] = None) -> AudioProfile:
    """Best-effort audio-profile detection.

    Args:
        device_name: optional explicit device name (test hook). When
            absent, the function attempts to query ``sounddevice`` for
            the current output device.

    Returns:
        :class:`AudioProfile` value (defaults to ``LOCAL`` when probing
        fails or the device looks generic).
    """
    name = (device_name or _probe_sd_device_name() or "").lower()
    if not name:
        return AudioProfile.LOCAL
    if "bluetooth" in name or "bt audio" in name or "airpods" in name:
        return AudioProfile.BLUETOOTH
    if "remote" in name or "rdp" in name:
        return AudioProfile.REMOTE
    return AudioProfile.LOCAL


def _probe_sd_device_name() -> Optional[str]:
    """Quietly query sounddevice for the default output device name."""
    try:
        import sounddevice as sd  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    try:
        device = sd.default.device
        if isinstance(device, (list, tuple)) and len(device) > 1:
            device = device[1]
        info = sd.query_devices(device)
        if isinstance(info, dict):
            return info.get("name")
    except Exception:  # noqa: BLE001
        return None
    return None


class PresentationScheduler:
    """Schedule chunks into priority-banded debounced flushes.

    Args:
        on_emit: callback invoked with each flushed chunk (the chunk's
            content + its priority).
        cadence: optional explicit cadence (overrides the profile map).
        audio_profile: optional explicit profile (overrides detection).
        clock: optional monotonic clock callable (test hook).

    Notes:
        The scheduler is single-threaded by contract; callers feed
        chunks via :meth:`enqueue` and pump via :meth:`maybe_emit`
        (or :meth:`flush` to force). The :class:`StreamCoordinator`
        in :mod:`ultron.streaming.coordinator` is the typical driver.
    """

    def __init__(
        self,
        on_emit: Optional[Callable[[str, PresentationPriority], None]] = None,
        *,
        cadence: Optional[Cadence] = None,
        audio_profile: Optional[AudioProfile] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._on_emit = on_emit
        self._profile = audio_profile or detect_audio_profile()
        self._cadence = cadence or DEFAULT_CADENCE_BY_PROFILE[self._profile]
        self._clock = clock or time.monotonic
        self._lock = threading.RLock()
        self._queues: dict[PresentationPriority, list[str]] = {
            PresentationPriority.IMMEDIATE: [],
            PresentationPriority.NORMAL: [],
            PresentationPriority.LOW: [],
        }
        self._last_emit_at: dict[PresentationPriority, float] = {
            PresentationPriority.IMMEDIATE: 0.0,
            PresentationPriority.NORMAL: 0.0,
            PresentationPriority.LOW: 0.0,
        }
        self._has_emitted: dict[PresentationPriority, bool] = {
            PresentationPriority.IMMEDIATE: False,
            PresentationPriority.NORMAL: False,
            PresentationPriority.LOW: False,
        }
        self._drop_low: bool = False

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def enqueue(
        self,
        content: str,
        priority: PresentationPriority = PresentationPriority.NORMAL,
    ) -> None:
        """Buffer ``content`` for emission at the priority's cadence."""
        if not content:
            return
        with self._lock:
            if priority is PresentationPriority.LOW and self._drop_low:
                return
            self._queues[priority].append(content)

    def maybe_emit(self) -> int:
        """Pump the scheduler — emit anything past its cadence window.

        Returns:
            Number of chunks emitted on this call.

        Notes:
            The first emit per priority on a fresh scheduler always
            passes — the ``last_emit_at`` sentinel of 0 is treated as
            "never emitted". Subsequent calls enforce the cadence.
        """
        with self._lock:
            emitted = 0
            now = self._clock() * 1000
            for priority in (
                PresentationPriority.IMMEDIATE,
                PresentationPriority.NORMAL,
                PresentationPriority.LOW,
            ):
                if not self._queues[priority]:
                    continue
                interval = self._cadence.for_priority(priority)
                if self._has_emitted[priority]:
                    last_at = self._last_emit_at[priority]
                    if (now - last_at * 1000) < interval:
                        continue
                content = "".join(self._queues[priority])
                self._queues[priority].clear()
                self._last_emit_at[priority] = self._clock()
                self._has_emitted[priority] = True
                if self._on_emit is not None:
                    self._safe_emit(content, priority)
                emitted += 1
            return emitted

    def flush(self) -> int:
        """Force-emit every buffered chunk regardless of cadence."""
        with self._lock:
            emitted = 0
            for priority in (
                PresentationPriority.IMMEDIATE,
                PresentationPriority.NORMAL,
                PresentationPriority.LOW,
            ):
                if not self._queues[priority]:
                    continue
                content = "".join(self._queues[priority])
                self._queues[priority].clear()
                self._last_emit_at[priority] = self._clock()
                self._has_emitted[priority] = True
                if self._on_emit is not None:
                    self._safe_emit(content, priority)
                emitted += 1
            return emitted

    def set_cadence(self, cadence: Cadence) -> None:
        """Replace the active cadence map."""
        with self._lock:
            self._cadence = cadence

    def set_drop_low_priority(self, drop: bool) -> None:
        """Toggle the "drop everything LOW priority" flag.

        Useful when the orchestrator runs with ``enable_thinking=False``
        and reasoning chunks should be silenced entirely.
        """
        with self._lock:
            self._drop_low = bool(drop)

    def pending_count(self) -> int:
        with self._lock:
            return sum(len(q) for q in self._queues.values())

    def cadence(self) -> Cadence:
        with self._lock:
            return self._cadence

    def profile(self) -> AudioProfile:
        with self._lock:
            return self._profile

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_emit(
        self, content: str, priority: PresentationPriority,
    ) -> None:
        try:
            self._on_emit(content, priority)  # type: ignore[misc]
        except Exception:  # noqa: BLE001
            LOGGER.warning(
                "on_emit raised for priority=%s", priority.value, exc_info=True,
            )


__all__ = [
    "AudioProfile",
    "Cadence",
    "DEFAULT_CADENCE_BY_PROFILE",
    "PresentationPriority",
    "PresentationScheduler",
    "detect_audio_profile",
]
