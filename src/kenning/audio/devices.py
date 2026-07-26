"""Helpers for resolving PortAudio device settings."""

from __future__ import annotations

from typing import Literal, Optional

import sounddevice as sd

from kenning.utils.logging import get_logger

logger = get_logger("audio.devices")

DeviceKind = Literal["input", "output"]


class AudioDeviceError(ValueError):
    """Raised when a configured audio device cannot be resolved."""


def _wasapi_preferred() -> bool:
    """Whether OUTPUT resolution + stream opens should prefer WASAPI low-latency.

    Reads ``audio.prefer_wasapi_output`` (default True). Fail-open to True so a
    config read error doesn't silently drop back to high-latency MME."""
    try:
        from kenning.config import get_config

        return bool(getattr(get_config().audio, "prefer_wasapi_output", True))
    except Exception:                                            # noqa: BLE001
        return True


def _is_wasapi(index: int) -> bool:
    """True iff PortAudio device ``index`` is a Windows WASAPI endpoint."""
    try:
        info = sd.query_devices(index)
        return "wasapi" in str(
            sd.query_hostapis(info["hostapi"])["name"]).casefold()
    except Exception:                                            # noqa: BLE001
        return False


def _prefer_wasapi_index(indices: list[int]) -> int:
    """Given equally-valid name matches, return the WASAPI one if present.

    On Windows the same physical/virtual device is exposed under MME (~90-180 ms
    buffer), DirectSound, AND WASAPI (~2-25 ms). MME enumerates first, so the
    naive "first match" lands on the slow endpoint; prefer WASAPI for OUTPUT."""
    for i in indices:
        if _is_wasapi(i):
            return i
    return indices[0]


def resolve_device(configured: Optional[str | int], kind: DeviceKind) -> Optional[int]:
    """Resolve an env/config device value to a PortAudio index.

    ``sounddevice`` accepts strings directly, but resolving here lets us validate
    the direction, support numeric strings, and log the exact device in use.
    """
    if configured is None:
        return _default_device_index(kind)
    if isinstance(configured, int):
        return _validate_device_index(configured, kind)

    value = configured.strip()
    if not value:
        return _default_device_index(kind)
    try:
        index = int(value)
    except ValueError:
        index = None
    if index is not None:
        return _validate_device_index(index, kind)

    needle = value.casefold()
    devices = list(sd.query_devices())
    matches = [
        index
        for index, device in enumerate(devices)
        if _supports_kind(device, kind)
        and needle in str(device.get("name", "")).casefold()
    ]
    exact = [
        index
        for index in matches
        if str(devices[index].get("name", "")).casefold() == needle
    ]
    # For OUTPUT, prefer the WASAPI endpoint among equally-valid matches so a
    # name like "Voicemeeter Input" resolves to the low-latency WASAPI device
    # instead of the high-latency MME one that enumerates first.
    if exact:
        if kind == "output" and _wasapi_preferred():
            return _prefer_wasapi_index(exact)
        return exact[0]
    if matches:
        if kind == "output" and _wasapi_preferred():
            return _prefer_wasapi_index(matches)
        return matches[0]

    available = ", ".join(_available_devices(kind)[:20])
    raise AudioDeviceError(
        f"No {kind} audio device matches {configured!r}. "
        f"Available {kind} devices: {available}"
    )


def describe_device(device: Optional[int], kind: DeviceKind) -> str:
    """Return a compact human-readable label for logs."""
    if device is None:
        return "system default"
    try:
        info = sd.query_devices(device, kind)
    except Exception:
        return str(device)
    return f"[{device}] {info['name']}"


def _default_device_index(kind: DeviceKind) -> Optional[int]:
    try:
        default = sd.default.device
    except Exception:
        return None

    slot = 0 if kind == "input" else 1
    if isinstance(default, int):
        index = default
    else:
        try:
            index = default[slot]
        except (TypeError, IndexError, KeyError):
            return None

    if index is None or int(index) < 0:
        return None
    return _validate_device_index(int(index), kind)


def _validate_device_index(index: int, kind: DeviceKind) -> int:
    try:
        device = sd.query_devices(index)
    except Exception as e:
        raise AudioDeviceError(f"No audio device exists at index {index}") from e
    if not _supports_kind(device, kind):
        raise AudioDeviceError(f"Audio device [{index}] is not an {kind} device")
    return index


def _supports_kind(device: dict, kind: DeviceKind) -> bool:
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    return int(device.get(key, 0) or 0) > 0


def _available_devices(kind: DeviceKind) -> list[str]:
    devices = list(sd.query_devices())
    return [
        f"[{index}] {device.get('name', '')}"
        for index, device in enumerate(devices)
        if _supports_kind(device, kind)
    ]


def make_output_stream(
    device: Optional[int],
    samplerate: int,
    channels: int,
    dtype: str = "int16",
    *,
    stream_factory=None,
):
    """Open the LOWEST-LATENCY output stream available for ``device``.

    The single chokepoint every spoken-audio output path uses (relay mic,
    OBS/monitor mirrors). Strategy, most-preferred first, each step falling back
    to the next so a fussy host never silences a channel:

      1. WASAPI device + ``prefer_wasapi_output``: ``latency='low'`` +
         ``WasapiSettings(auto_convert=True)`` -- the auto-convert lets Kokoro's
         24 kHz play on a 48 kHz WASAPI device (WASAPI won't resample itself,
         unlike MME). ~2-25 ms buffer.
      2. Any device: ``latency='low'`` -- MME/DirectSound honour this (~90 ms
         vs the ~180 ms default).
      3. Plain default open -- last-resort legacy behaviour.

    ``stream_factory`` (test seam) bypasses all host logic and is called with the
    plain kwargs, exactly as the legacy callers expected.
    """
    base = dict(samplerate=samplerate, channels=channels, dtype=dtype,
                device=device)
    if stream_factory is not None:
        return stream_factory(**base)

    want_wasapi = (
        device is not None and _wasapi_preferred() and _is_wasapi(device)
    )
    # 1. WASAPI low-latency (+ auto-convert for the 24k->48k case).
    if want_wasapi:
        try:
            return sd.OutputStream(
                **base, latency="low",
                extra_settings=sd.WasapiSettings(auto_convert=True),
            )
        except Exception as e:                               # noqa: BLE001
            logger.debug(
                "WASAPI low-latency open failed for device %s (%s); "
                "falling back", device, e,
            )
    # 2. Generic low-latency hint (MME / DirectSound honour it).
    try:
        return sd.OutputStream(**base, latency="low")
    except Exception as e:                                   # noqa: BLE001
        logger.debug(
            "low-latency open failed for device %s (%s); using default",
            device, e,
        )
    # 3. Legacy default open.
    return sd.OutputStream(**base)
