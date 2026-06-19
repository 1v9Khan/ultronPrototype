"""Console entrypoint: ``python -m kenning``."""

from __future__ import annotations

import signal
import sys

# NOTE: the Orchestrator is imported LAZILY inside main() -- AFTER the anticheat
# import firewall is installed -- so there is no window in which the Orchestrator
# module's (transitive) import chain could pull a blocked input/capture/automation
# module before the loader-level block is live (2026-06-17 audit: close the
# structural pre-firewall import window). Do NOT add a module-top
# `from kenning.pipeline import Orchestrator` back here.
from kenning.utils.logging import configure_logging, get_logger


class _ResilientStream:
    """Wrap a text stream so ``write()`` / ``flush()`` NEVER raise.

    2026-06-18 fix: the orchestrator's run loop and response pipeline echo
    progress to the console with ~30 plain ``print()`` calls ("kenning: ",
    "you: ", state lines). When stdout is NOT a healthy console -- a
    background/detached launch with stdout redirected to a file, a closed
    parent shell, a ``pythonw`` windowless launch -- a write can throw
    ``OSError: [Errno 22] Invalid argument``. Because the first ``print()`` in
    the conversational branch (``_respond``) sits at the TOP of the try block,
    that exception aborted the ENTIRE turn before TTS ever ran -- so every
    command that routed to the conversational pipeline was silently DROPPED
    (observed live as "auto-rejected" callouts). The console echo is a UX
    nicety; the real output is TTS audio + the rotating log file, so a broken
    stdout must degrade to silence, never crash a turn. (Companion to the
    encoding guard below, which fixed the same crash class for cp1252
    ``UnicodeEncodeError``.) Delegates everything else to the wrapped stream.
    """

    def __init__(self, wrapped: object) -> None:
        self._w = wrapped

    def write(self, s):  # noqa: ANN001, ANN201
        try:
            return self._w.write(s)
        except Exception:  # noqa: BLE001 - a dead stdout must never crash a turn
            return len(s) if isinstance(s, (str, bytes)) else 0

    def flush(self) -> None:
        try:
            self._w.flush()
        except Exception:  # noqa: BLE001
            pass

    def __getattr__(self, name):  # noqa: ANN001, ANN204 - delegate isatty/fileno/encoding/...
        return getattr(self._w, name)


def _ensure_utf8_stdio() -> None:
    """Make stdout / stderr resilient: UTF-8 encoding AND crash-proof writes.

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

    2026-06-18 fix: ALSO wrap each stream in :class:`_ResilientStream` so a
    write to a broken/redirected/closed stdout (``OSError: [Errno 22]``)
    degrades to a no-op instead of aborting the turn. See that class.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                # Best-effort: a non-reconfigurable stream (rare; tests
                # sometimes wrap stdio) keeps its existing settings.
                pass
        # Wrap AFTER reconfigure so encoding settings are applied to the real
        # stream; the wrapper just guards write/flush. Idempotent guard so a
        # double call (or an already-wrapped stream) doesn't stack wrappers.
        if not isinstance(stream, _ResilientStream):
            try:
                setattr(sys, stream_name, _ResilientStream(stream))
            except Exception:  # noqa: BLE001
                pass


def main() -> int:
    _ensure_utf8_stdio()
    configure_logging()
    logger = get_logger("main")

    # 2026-06-19: flavor tails OFF by DEFAULT for the running app -- crisp,
    # tail-free callouts for competitive play ("Group up." not "Group up, but
    # their formation is predictable...") that survive restarts. Set here, before
    # the lazy Orchestrator import pulls relay_speech (which reads this env once
    # at import). setdefault, so a real KENNING_FLAVOR_TAILS env var AND the
    # runtime "Ultron, flavor on" toggle both still win. The library default
    # stays ON, so tests/standalone imports are unaffected. NB the app does not
    # auto-load .env, so a .env value alone would NOT take effect here.
    import os as _os_flavor_default
    _os_flavor_default.environ.setdefault("KENNING_FLAVOR_TAILS", "0")

    # 2026-06-15/06-17 audit hardening: install the anticheat import firewall as
    # the VERY FIRST action after logging -- BEFORE the single-instance lock,
    # BEFORE the Orchestrator MODULE is even imported (the import is lazy, below),
    # and BEFORE it is constructed -- so there is NO window in which a blocked
    # input/capture/automation module could be imported before the loader-level
    # block is live. The firewall is a NO-OP while anticheat-safe mode is off, so
    # this is free for non-gaming sessions; it is idempotent (the Orchestrator's
    # own install() in __init__ is a safe no-op); and on UNCERTAINTY it fails
    # SAFE (blocks). After install we PROVE it actually enforces (a live blocked-
    # import probe) and, while anticheat is active, treat a non-enforcing firewall
    # as FATAL -- we refuse to start rather than run a protected game without the
    # loader-level backstop.
    try:
        from kenning.safety.import_firewall import (
            assert_firewall_enforces,
            install_import_firewall,
            is_firewall_installed,
        )

        install_import_firewall()
        enforces = assert_firewall_enforces()
        try:
            from kenning.safety.anticheat import anticheat_active
            ac = bool(anticheat_active())
        except Exception:                                            # noqa: BLE001
            ac = True  # uncertain -> treat as protected (fail-safe)
        if ac and (not is_firewall_installed() or not enforces):
            logger.critical(
                "anticheat import firewall is NOT enforcing while anticheat-safe "
                "mode is active -- REFUSING to start (the loader-level block on "
                "input/capture/automation modules is the safety backstop beside a "
                "kernel anticheat). Investigate import_firewall before going live."
            )
            print("\n[!] Anticheat import firewall not enforcing; refusing to "
                  "start. See logs.\n")
            return 4
    except Exception as e:  # noqa: BLE001
        # The firewall machinery itself failed to load. While anticheat is the
        # default posture, this is fatal -- do not boot blind beside Vanguard.
        logger.critical(
            "anticheat import firewall FAILED to install/verify at entry (%s) -- "
            "REFUSING to start; the loader-level block could not be established.",
            e,
        )
        print(f"\n[!] Anticheat firewall failed to initialize ({e}); refusing to "
              "start. See logs.\n")
        return 4

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

        # Lazy import: the firewall is now installed + verified, so importing the
        # Orchestrator module (and its transitive chain) happens UNDER the
        # loader-level block (2026-06-17 audit: no pre-firewall import window).
        from kenning.pipeline import Orchestrator

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
