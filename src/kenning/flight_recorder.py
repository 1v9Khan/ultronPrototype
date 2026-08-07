"""Crash flight recorder -- last-known thread stacks for a native crash.

The 2026-07-26 Gemma-4-era crashes kill the process with a silent
``__fastfail`` (0xc0000409): no traceback, no stderr text, no WER dump on
this box. Nine targeted reproductions survived while the real boot dies
within a minute -- so instead of guessing, record what EVERY thread was
doing right up to the death.

A daemon thread snapshots ``sys._current_frames()`` every ``interval_s``
into two alternating files (A/B) so a snapshot torn by the crash always
leaves the previous one intact. After a crash, read the newer intact file:
the thread whose stack sits inside a native call (ctranslate2 / torch /
llama_cpp / sounddevice) at death is the culprit.

Anticheat posture: stdlib only (sys/threading/time/traceback/pathlib/os);
no capture, no OS interaction, no third-party imports. Default ON
(OPT-IN: set ``KENNING_FLIGHT_RECORDER=1``); ~1 ms per snapshot at 5 Hz, so
leaving it on costs nothing measurable.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from pathlib import Path

_started = False
_resolved_dir = ""
_lock = threading.Lock()


def enabled() -> bool:
    """Default ON (2026-07-26). It was opt-in for one round and that cost a
    real crash's evidence: the operator's own launches did not set the env,
    so the only runs that recorded anything were mine. At 5 Hz this is ~1 ms
    of stack formatting per second -- far cheaper than another blind repro
    cycle. Off unless ``KENNING_FLIGHT_RECORDER=1``."""
    return os.getenv("KENNING_FLIGHT_RECORDER", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def start(log_dir: str | Path | None = None, interval_s: float = 0.2) -> bool:
    """Start the recorder thread (idempotent). Returns True when running.

    ``log_dir`` defaults to ``<PROJECT_ROOT>/logs`` -- an ABSOLUTE path. It
    was relative ("logs") for one round and the recorder then wrote into
    whatever directory the process happened to be launched from, so a real
    04:38 crash left no snapshot even though the thread was running. The
    project root is resolved from this file's location, so no config import
    (and no import-order risk) is involved.
    """
    global _started
    with _lock:
        if _started:
            return True
        try:
            if log_dir is None:
                # .../src/kenning/flight_recorder.py -> project root
                log_dir = Path(__file__).resolve().parents[2] / "logs"
            d = Path(log_dir)
            d.mkdir(parents=True, exist_ok=True)
            paths = (d / "_flight_A.txt", d / "_flight_B.txt")
        except Exception:                                        # noqa: BLE001
            return False

        def _loop() -> None:
            n = 0
            while True:
                n += 1
                target = paths[n % 2]
                try:
                    frames = sys._current_frames()
                    names = {t.ident: (t.name, t.daemon)
                             for t in threading.enumerate()}
                    lines = [f"snapshot #{n} pid={os.getpid()} "
                             f"t={time.time():.3f} "
                             f"({time.strftime('%H:%M:%S')})\n"]
                    for tid, frame in frames.items():
                        name, daem = names.get(tid, ("?", "?"))
                        lines.append(
                            f"\n--- thread {tid} name={name!r} "
                            f"daemon={daem}\n")
                        lines.extend(traceback.format_stack(frame))
                    tmp = target.with_suffix(".tmp")
                    tmp.write_text("".join(lines), encoding="utf-8",
                                   errors="replace")
                    tmp.replace(target)          # atomic on same volume
                except Exception:                                # noqa: BLE001
                    pass                          # never disturb the host
                time.sleep(interval_s)

        t = threading.Thread(target=_loop, daemon=True,
                             name="flight-recorder")
        t.start()
        _started = True
        globals()["_resolved_dir"] = str(d)
        return True


def maybe_start_from_env() -> bool:
    """Start when KENNING_FLIGHT_RECORDER is set; also enables faulthandler
    (free extra signal for non-fastfail aborts)."""
    if not enabled():
        return False
    try:
        import faulthandler
        faulthandler.enable()
    except Exception:                                            # noqa: BLE001
        pass
    return start()


def resolved_dir() -> str:
    """Absolute directory the snapshots are being written to ('' if off)."""
    return _resolved_dir
