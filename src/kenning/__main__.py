"""Console entrypoint: ``python -m kenning``."""

from __future__ import annotations

import signal
import sys

from kenning.pipeline import Orchestrator
from kenning.utils.logging import configure_logging, get_logger


def _ensure_utf8_stdio() -> None:
    """Reconfigure stdout / stderr to UTF-8 with ``errors='replace'``.

    2026-05-19 fix: on Windows the default console encoding is cp1252
    which cannot encode many characters that show up in source titles
    / URLs (smart quotes, em-dashes, Unicode glyphs). A printed source
    list crashed the entire response pipeline with::

        UnicodeEncodeError: 'charmap' codec can't encode characters in
        position 160-161: character maps to <undefined>

    Forcing UTF-8 with ``errors='replace'`` makes every ``print()``
    call resilient: unencodable code points become ``?`` in the
    console instead of throwing. The audio pipeline is unaffected
    (TTS uses its own pipeline); only console output changes.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # Best-effort: a non-reconfigurable stream (rare; tests
            # sometimes wrap stdio) keeps its existing settings.
            pass


def main() -> int:
    _ensure_utf8_stdio()
    configure_logging()
    logger = get_logger("main")

    # 2026-06-12 single-instance guard: two simultaneous `python -m
    # kenning` processes both grab the mic and double-respond (and the
    # second collides on the embedded Qdrant lock + MCP port 19761).
    # Acquired BEFORE any model load; releases automatically on
    # process death (held OS file lock), so a crash never blocks the
    # next launch. The guard lives HERE (not in Orchestrator) so
    # pytest / the GPU e2e suite / measurement scripts that construct
    # the Orchestrator directly never contend.
    from kenning.lifecycle.single_instance import (
        ALLOW_MULTIPLE_ENV,
        DEFAULT_LOCK_PATH,
        acquire_single_instance_lock,
        read_lock_metadata,
    )

    instance_lock = acquire_single_instance_lock()
    if instance_lock is None:
        meta = read_lock_metadata(DEFAULT_LOCK_PATH) or {}
        other_pid = meta.get("pid", "unknown")
        msg = (
            f"Another Kenning instance is already running (PID {other_pid}); "
            f"refusing to start a duplicate. Set {ALLOW_MULTIPLE_ENV}=1 "
            "to override."
        )
        logger.error(msg)
        print(f"\n[!] {msg}\n")
        return 3

    try:
        print("\n" + "=" * 60)
        print("  KENNING")
        print("  Local voice-first AI assistant — prototype")
        print("=" * 60)
        print("  Loading models — this can take 1–3 minutes on first run.")

        try:
            orchestrator = Orchestrator()
        except FileNotFoundError as e:
            logger.error("Missing model: %s", e)
            print(f"\n[!] {e}")
            print("    Run: python scripts/download_models.py\n")
            return 2
        except Exception as e:
            logger.exception("Startup failed: %s", e)
            print(f"\n[!] Startup failed: {e}")
            return 1

        def _sigint(_sig, _frm):
            print("\n  shutting down…")
            orchestrator.shutdown()

        signal.signal(signal.SIGINT, _sigint)
        # SIGTERM (Linux/WSL `kill`, service-manager stop, os.kill(pid, SIGTERM))
        # must run the SAME full cleanup. NOTE: on Windows `taskkill /F` /
        # Task-Manager "End task" is TerminateProcess and is UNCATCHABLE -- that
        # force-kill path is covered at the NEXT boot by the sidecar orphan sweep
        # + audit-log repair, NOT in-process here.
        try:
            signal.signal(signal.SIGTERM, _sigint)
        except (OSError, ValueError, AttributeError):
            pass
        # atexit backstop: catches exit paths that bypass both the `with`
        # context-manager and the signal handlers (e.g. sys.exit deep in a
        # thread). shutdown() is idempotent, so the redundant calls are harmless.
        import atexit
        atexit.register(orchestrator.shutdown)

        try:
            with orchestrator:
                orchestrator.run()
        except Exception as e:
            logger.exception("Run loop failed: %s", e)
            return 1

        print("  goodbye.\n")
        return 0
    finally:
        instance_lock.release()


if __name__ == "__main__":
    sys.exit(main())
