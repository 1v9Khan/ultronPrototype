"""Operator diagnostics toggle for the verbose spoken-audio logging.

The spoken-audio features it gates (SPOKEN text logging + the per-utterance
SPOKEN-BLIP raw-vs-final analysis in the TTS engine) ONLY ever read Kenning's
OWN audio-output buffers and write to Kenning's OWN log. They never touch a
foreign process's memory, the input devices, the screen, kernel objects, or any
hook -- i.e. NONE of the classes a kernel anticheat (Vanguard/EAC/BattlEye)
watches for. They are exactly as anticheat-neutral as Discord or OBS logging
their own state. This toggle exists purely so the operator can:

  * keep the live log quiet during an actual stream/game, and
  * skip the (small) per-utterance analysis cost when not debugging.

Default OFF. Two ways to enable -- a LIVE sentinel file (no restart) or a config
flag:

  * ``touch ~/.kenning/audio_diagnostics_on``   -> on immediately (delete -> off)
  * ``diagnostics.spoken_audio_logging: true``  -> on from boot

The sentinel is checked once per spoken utterance (utterances are infrequent;
the stat cost is negligible) so the operator can flip diagnostics on/off WITHOUT
restarting a live session.
"""
from __future__ import annotations

from pathlib import Path

# Live, restart-free override. Stored under the user's gitignored Kenning dir.
_SENTINEL = Path.home() / ".kenning" / "audio_diagnostics_on"


def audio_diagnostics_enabled() -> bool:
    """True iff verbose spoken-audio logging/analysis should run.

    OR of the live sentinel file and the ``diagnostics.spoken_audio_logging``
    config flag. Fail-safe: any error returns False (logging stays off).
    """
    try:
        if _SENTINEL.exists():
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        from kenning.config import get_config

        diag = getattr(get_config(), "diagnostics", None)
        return bool(getattr(diag, "spoken_audio_logging", False))
    except Exception:  # noqa: BLE001
        return False


def reset_for_new_session() -> None:
    """Clear the live sentinel at startup so EVERY boot defaults to diagnostics
    OFF. Testing mode is then an explicit, post-boot opt-in (the operator
    re-touches the sentinel) -- a manual restart never silently keeps the
    monitoring on. Fail-open. Called once from the orchestrator at boot."""
    try:
        _SENTINEL.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def enable_for_testing() -> None:
    """Turn diagnostics ON for the current session (post-boot, live). Used by
    the operator when actively analyzing; checked per-utterance so it takes
    effect immediately with no restart."""
    try:
        _SENTINEL.parent.mkdir(parents=True, exist_ok=True)
        _SENTINEL.touch()
    except Exception:  # noqa: BLE001
        pass
