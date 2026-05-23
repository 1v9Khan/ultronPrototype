"""Idempotent file-installation helpers.

The OpenHands ``maybe_setup_git_hooks`` pattern (look for a marker comment,
no-op if already installed, preserve any user-written content otherwise)
generalises to every "install this file into the user's environment"
code path ultron has: the pre-push hygiene hook, the OpenClaw agent
config, the Task Scheduler entry for Kokoro training resume, voicepack
placement.

Pattern lineage attributed in ``THIRD_PARTY_NOTICES.md``.
"""

from ultron.install.idempotent import (
    DEFAULT_INSTALL_LOG_PATH,
    DEFAULT_MARKER,
    DEFAULT_PRESERVE_SUFFIX,
    InstallAction,
    InstallLogEntry,
    InstallLogWriter,
    InstallResult,
    install_with_marker,
    set_install_log_writer,
)

__all__ = [
    "DEFAULT_INSTALL_LOG_PATH",
    "DEFAULT_MARKER",
    "DEFAULT_PRESERVE_SUFFIX",
    "InstallAction",
    "InstallLogEntry",
    "InstallLogWriter",
    "InstallResult",
    "install_with_marker",
    "set_install_log_writer",
]
