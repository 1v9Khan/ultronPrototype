"""Boot-time VoiceMeeter relay-bus level guard (DEFAULT OFF).

The Valorant team-voice degradation root cause was a VoiceMeeter B1 bus fader
sitting ~21 dB BELOW the real-mic B2 bus, so Vivox's always-on AGC over-amplified
Ultron and lifted the codec/quantization noise floor (the gritty/thin sound). The
DECISIVE fix is raising that fader by hand to match the mic. This guard makes the
fix STICK across VoiceMeeter scene reloads: at boot it reads B1 vs B2 via
VoiceMeeter's OWN public Remote API and, if B1 has drifted far below B2, logs a
WARNING (and, only when restore is explicitly enabled, sets B1 to match B2).

Anticheat: this touches ONLY VoiceMeeter's documented Remote API on our own
machine -- it NEVER touches Valorant's process, memory, or input (same principle
as the external-HID push-to-talk). It is entirely FAIL-OPEN: a missing/relocated
DLL, an offline VoiceMeeter, or any API error logs-and-returns; it never raises
and never blocks boot.

Env:
  KENNING_RELAY_VM_LEVEL_GUARD = 1   enable the guard         (default OFF)
  KENNING_RELAY_VM_RESTORE     = 1   also SET B1 to match B2  (default OFF; warn-only)
  KENNING_RELAY_VM_B1_INDEX    = 5   bus index Valorant reads (default 5 / Potato B1)
  KENNING_RELAY_VM_B2_INDEX    = 6   real-mic bus index       (default 6 / Potato B2)
  KENNING_RELAY_VM_DELTA_DB    = 6   warn when B2-B1 exceeds this many dB
  KENNING_VOICEMEETER_DLL      = <path to VoicemeeterRemote64.dll>
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_DEFAULT_DLL = r"C:\Program Files (x86)\VB\Voicemeeter\VoicemeeterRemote64.dll"


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() not in (
        "0", "false", "no", "off", "",
    )


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:  # noqa: BLE001
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:  # noqa: BLE001
        return default


def _load_dll():
    """ctypes-load VoicemeeterRemote64.dll from the known VB-Audio path ONLY
    (never PATH/CWD). Returns the configured DLL handle or None."""
    import ctypes

    path = os.getenv("KENNING_VOICEMEETER_DLL", _DEFAULT_DLL)
    if not path or not os.path.isfile(path):
        logger.debug("voicemeeter level guard: DLL not found at %r", path)
        return None
    try:
        dll = ctypes.WinDLL(path)
        dll.VBVMR_Login.restype = ctypes.c_long
        dll.VBVMR_Logout.restype = ctypes.c_long
        dll.VBVMR_IsParametersDirty.restype = ctypes.c_long
        dll.VBVMR_GetParameterFloat.argtypes = [
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_float)]
        dll.VBVMR_GetParameterFloat.restype = ctypes.c_long
        dll.VBVMR_SetParameterFloat.argtypes = [ctypes.c_char_p, ctypes.c_float]
        dll.VBVMR_SetParameterFloat.restype = ctypes.c_long
        return dll
    except Exception as e:  # noqa: BLE001 - fail-open
        logger.debug("voicemeeter level guard: DLL bind failed (%s)", e)
        return None


def _get_float(dll, name: str):
    import ctypes

    val = ctypes.c_float(0.0)
    rc = dll.VBVMR_GetParameterFloat(name.encode("ascii"), ctypes.byref(val))
    if rc != 0:
        return None
    return float(val.value)


def check_relay_bus_level() -> None:
    """Read B1 vs B2 bus gain via the VoiceMeeter Remote API and warn (or restore)
    if the Valorant mic bus is far below the real-mic bus. No-op unless
    KENNING_RELAY_VM_LEVEL_GUARD is set. Never raises."""
    if not _flag("KENNING_RELAY_VM_LEVEL_GUARD", "0"):
        return
    try:
        _run_guard()
    except Exception as e:  # noqa: BLE001 - never block boot
        logger.debug("voicemeeter level guard skipped (%s)", e)


def _run_guard() -> None:
    import ctypes
    import time

    dll = _load_dll()
    if dll is None:
        return
    login = dll.VBVMR_Login()
    # 0 == OK; 1 == OK but the VoiceMeeter application is not running -> no params
    # to read, so bail. Negative == failure.
    if login != 0:
        logger.debug("voicemeeter level guard: Login rc=%s (app not running?)",
                     login)
        try:
            dll.VBVMR_Logout()
        except Exception:  # noqa: BLE001
            pass
        return
    try:
        # First read after Login needs a refresh tick; poll IsParametersDirty a
        # few times (tiny, bounded) rather than sleeping a fixed slug.
        for _ in range(10):
            if dll.VBVMR_IsParametersDirty() == 0:
                break
            time.sleep(0.01)

        b1_idx = _int_env("KENNING_RELAY_VM_B1_INDEX", 5)
        b2_idx = _int_env("KENNING_RELAY_VM_B2_INDEX", 6)
        b1 = _get_float(dll, f"Bus[{b1_idx}].Gain")
        b2 = _get_float(dll, f"Bus[{b2_idx}].Gain")
        if b1 is None or b2 is None:
            logger.debug("voicemeeter level guard: could not read bus gains "
                         "(B1=%s B2=%s)", b1, b2)
            return

        delta = _float_env("KENNING_RELAY_VM_DELTA_DB", 6.0)
        if (b2 - b1) <= delta:
            logger.info("voicemeeter level guard OK | B1(Valorant)=%.2f dB "
                        "B2(mic)=%.2f dB (within %.0f dB)", b1, b2, delta)
            return

        # B1 is materially quieter than the mic bus -> Vivox AGC will over-gain
        # Ultron. Warn loudly; restore only if explicitly enabled.
        if _flag("KENNING_RELAY_VM_RESTORE", "0"):
            rc = dll.VBVMR_SetParameterFloat(
                f"Bus[{b1_idx}].Gain".encode("ascii"), ctypes.c_float(b2))
            logger.warning("voicemeeter level guard RESTORED B1 %.2f -> %.2f dB "
                           "to match the mic bus (rc=%s)", b1, b2, rc)
        else:
            logger.warning(
                "voicemeeter level guard | B1(Valorant)=%.2f dB is %.1f dB below "
                "B2(mic)=%.2f dB -> Vivox AGC will lift Ultron's noise floor. "
                "Raise the B1 bus fader toward B2 in VoiceMeeter (or set "
                "KENNING_RELAY_VM_RESTORE=1 to auto-match at boot).",
                b1, (b2 - b1), b2)
    finally:
        try:
            dll.VBVMR_Logout()
        except Exception:  # noqa: BLE001
            pass
