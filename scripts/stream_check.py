"""Pre-stream routing check -- confirm Kenning emits to the right devices.

Run this before a stream (with VoiceMeeter open so you can watch the strip
meters move):

    .venv\\Scripts\\python.exe scripts\\stream_check.py

It exercises the REAL production audio paths -- it does NOT load the assistant:

  1. EVERYTHING feed  -> the live BroadcastSink (audio.broadcast_device),
     i.e. "Voicemeeter AUX Input" -> route that strip to bus B3 in VoiceMeeter
     and capture "Voicemeeter Out B3" in OBS.  Carries normal + team speech.
  2. TEAM feed        -> the relay play_to_device path (relay_speech.output_device),
     i.e. "Voicemeeter Input" -> route that strip to bus B1 -> select
     "Voicemeeter Out B1" as Kenning's mic in your game.  Team callouts ONLY.
  3. DEFAULT output   -> what YOU hear (normal conversation goes here directly).
  4. LOCAL MONITOR    -> the relay/team callout teed to your DEFAULT output too
     (kenning.audio.monitor, gated by relay_speech.echo_to_user) so you hear
     your OWN callouts -- relay otherwise only reaches the mic + OBS.
  5. MIC (input)      -> the device you talk to Kenning on; must be untouched.

A short, quiet tone is played to each OUTPUT device so you can watch the
meters. The mic is only resolved, never opened. Exit 0 = all paths OK.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np

from kenning.config import load_config
from kenning.audio.devices import resolve_device


def tone(sr: int, hz: float, secs: float = 0.3, amp: float = 0.18) -> np.ndarray:
    t = np.arange(int(sr * secs)) / sr
    env = np.minimum(1.0, np.minimum(t * 40, (secs - t) * 40))  # short fades
    return (np.sin(2 * np.pi * hz * t) * env * amp * 32767).astype(np.int16)


def main() -> int:
    sr = 24000
    cfg = load_config()
    team = cfg.relay_speech.output_device
    everything = cfg.audio.broadcast_device
    mic = cfg.audio.input_device
    default_out = cfg.audio.output_device  # None = system default

    print("=" * 64)
    print("KENNING STREAM ROUTING CHECK")
    print("=" * 64)
    ok = True

    # -- resolve + report -------------------------------------------------
    import sounddevice as sd

    def show(label: str, spec, kind: str) -> int | None:
        try:
            idx = resolve_device(spec, kind)
            name = sd.query_devices()[idx]["name"] if idx is not None else "(system default)"
            print(f"  {label:24} {spec!r:26} -> {kind} idx {idx}: {name}")
            return idx
        except Exception as e:  # noqa: BLE001
            print(f"  {label:24} {spec!r:26} -> UNRESOLVED: {e}")
            return None

    print("\nResolving devices:")
    team_idx = show("TEAM (-> B1)", team, "output")
    every_idx = show("EVERYTHING (-> B3)", everything, "output")
    default_idx = show("DEFAULT output", default_out, "output")
    mic_idx = show("MIC (talk to Kenning)", mic, "input")

    if team_idx is None or every_idx is None:
        print("\nFAIL: a stream output device did not resolve.")
        return 1

    # -- 1. EVERYTHING feed via the real BroadcastSink --------------------
    print("\n[1/3] EVERYTHING feed via BroadcastSink (-> B3) ...")
    from kenning.audio.broadcast import get_broadcast_sink
    sink = get_broadcast_sink()
    sink.configure(everything)
    sink.submit(tone(sr, 330.0), sr)      # private-style
    sink.submit(tone(sr, 660.0), sr)      # team-style -- BOTH hit 'everything'
    time.sleep(1.2)
    resolved = sink._resolved_index       # noqa: SLF001 - verifying emission
    if resolved == every_idx:
        print(f"      OK  streamed to device idx {resolved} (both clips).")
    else:
        ok = False
        print(f"      WARN streamed idx={resolved}, expected {every_idx}.")
    sink.close()

    # -- 2. TEAM feed via the real relay play_to_device -------------------
    print("\n[2/3] TEAM feed via relay play_to_device (-> B1) ...")
    from kenning.audio.relay_speech import play_to_device
    secs = play_to_device(tone(sr, 880.0), sr, team_idx)
    if secs and secs > 0:
        print(f"      OK  played {secs:.2f}s to team device idx {team_idx}.")
    else:
        ok = False
        print("      FAIL relay play_to_device returned no playback.")

    # -- 3. DEFAULT output (what you hear) --------------------------------
    print("\n[3/4] DEFAULT output (you hear this) ...")
    try:
        stream = sd.OutputStream(samplerate=sr, channels=2, dtype="int16",
                                 device=default_idx)
        stream.start()
        st = tone(sr, 440.0)
        stream.write(np.column_stack((st, st)))
        stream.stop(); stream.close()
        print(f"      OK  played to default output idx {default_idx}.")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"      FAIL default output: {e}")

    # -- 4. LOCAL MONITOR: relay teed to your default output --------------
    print("\n[4/4] LOCAL MONITOR via kenning.audio.monitor (relay -> you) ...")
    if not getattr(cfg.relay_speech, "echo_to_user", False):
        print("      SKIP relay_speech.echo_to_user is OFF (monitor disabled).")
    else:
        from kenning.audio import monitor as _mon
        _mon.maybe_submit(tone(sr, 550.0), sr)   # reads echo_to_user live
        time.sleep(0.8)
        armed = getattr(_mon.get_monitor_sink(), "_resolved_index", None)
        if armed == default_idx:
            print(f"      OK  relay callouts also play to your default idx {armed}.")
        else:
            ok = False
            print(f"      WARN monitor armed idx={armed}, expected {default_idx}.")
        _mon.get_monitor_sink().close()

    # -- 5. MIC independence ----------------------------------------------
    print("\nMic check: input device resolved but NEVER opened/changed by any "
          "of the above (outputs only).")
    if mic_idx is None:
        print("      NOTE mic did not resolve -- check audio.input_device.")

    print("\n" + "=" * 64)
    print("RESULT:", "PASS -- all stream paths emit." if ok else "ISSUES -- see above.")
    print("Next: in VoiceMeeter route 'Voicemeeter Input' strip -> B1 and "
          "'Voicemeeter AUX Input' strip -> B3, then select Voicemeeter Out "
          "B1 (game mic) and Voicemeeter Out B3 (OBS) as capture sources.")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
