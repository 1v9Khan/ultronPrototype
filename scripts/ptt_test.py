"""Manual auto-PTT hardware test -- run WHILE Valorant is open.

WHY THIS EXISTS: a USB-HID keypress that works perfectly at the OS level (types
into Notepad) can still be dropped by Vanguard once Valorant boots -- so testing
outside the game is NOT representative. This drives the PTT key-hold (no Ultron
audio, no pipeline) so you can see, inside Valorant, whether holding the team-PTT
key via the device actually makes the game transmit.

It uses the same backend selection as Ultron (``build_ptt_controller``), so it
works with either device: the hardened HID-only board (raw HID) or the legacy
serial board.

USAGE (from the repo root):
    .venv\\Scripts\\python.exe scripts\\ptt_test.py            # 5 cycles, 3s hold
    .venv\\Scripts\\python.exe scripts\\ptt_test.py --hold 4 --cycles 8

WHAT TO WATCH (two independent signals so we can localize any failure):
  1. The device's onboard LED -- it lights while the key is HELD. (Hardware OK.)
  2. Inside Valorant (a custom game / Swiftplay match), watch the team-voice
     "transmitting" indicator. It should light in sync with the LED.

DIAGNOSIS:
  * LED lights AND Valorant transmits in sync  -> the PTT path works through Vanguard.
  * LED lights BUT Valorant shows NO transmit   -> Vanguard is dropping the input.
  * LED does NOT light                          -> device/host issue, not Vanguard.

Do a BASELINE run with Notepad focused first (the key types into it), THEN
alt-tab into Valorant and run it again.
"""
import argparse
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description="Manual auto-PTT hardware test")
    ap.add_argument("--hold", type=float, default=3.0, help="seconds to hold the key per cycle")
    ap.add_argument("--gap", type=float, default=2.0, help="seconds between cycles")
    ap.add_argument("--cycles", type=int, default=5, help="number of hold/release cycles")
    ap.add_argument("--countdown", type=float, default=4.0, help="seconds before starting (alt-tab to Valorant)")
    args = ap.parse_args()

    from kenning.ptt.controller import build_ptt_controller

    # Force-enable for the test; backend (raw-HID vs serial) is auto-selected
    # from config exactly like Ultron does at boot.
    ctrl = build_ptt_controller(enabled=True)
    if not ctrl.available:
        print("ERROR: no PTT device found (raw-HID or serial). Is it plugged in / flashed?")
        return 1

    print(f"PTT test: {args.cycles} cycles, hold {args.hold}s, gap {args.gap}s.")
    print("ALT-TAB to Valorant now (mic test / custom game) and watch the transmit")
    print("indicator + the device's onboard LED.\n")
    for i in range(int(args.countdown), 0, -1):
        print(f"  starting in {i}...", end="\r", flush=True)
        time.sleep(1.0)
    print(" " * 30, end="\r")

    try:
        for c in range(1, args.cycles + 1):
            print(f"[{c}/{args.cycles}] HOLD    key down  (LED ON)  ...", flush=True)
            ctrl.hold()                       # presses + heartbeats (driver thread)
            time.sleep(args.hold)
            ctrl.release()                    # releases after the configured tail
            print(f"          RELEASE key up    (LED OFF)", flush=True)
            if c < args.cycles:
                time.sleep(args.gap)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        ctrl.close()

    print("\nDone. If Valorant's transmit indicator lit in sync with the LED, the")
    print("full PTT path works through Vanguard.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
