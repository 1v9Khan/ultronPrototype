"""Logging configuration.

Two handlers are installed by default:
- A rotating file handler at DEBUG, writing every event to ``logs/kenning.log``.
- A console handler at the configured level, with a more compact format.

Modules call :func:`get_logger` to fetch their named logger; they do not need
to invoke :func:`configure_logging` themselves — the entrypoint owns that.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from kenning.config import get_config, resolve_path


_CONFIGURED = False


def configure_logging(
    level: str | None = None,
    log_file: Path | None = None,
) -> None:
    """Install handlers on the root logger. Idempotent.

    Args:
        level: Override for the console log level. Defaults to
            ``config.logging.level`` (env var ``KENNING_LOG_LEVEL`` overrides).
        log_file: Override for the file destination. Defaults to
            ``config.logging.file``.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    cfg = get_config().logging
    level = (level or os.getenv("KENNING_LOG_LEVEL") or cfg.level).upper()
    log_file = log_file or resolve_path(cfg.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Wipe any handlers added by libraries during import.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(cfg.format, cfg.datefmt)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(getattr(logging, level, logging.INFO))
    console.setFormatter(
        logging.Formatter("%(levelname)-7s | %(name)-20s | %(message)s")
    )
    root.addHandler(console)

    # Tame chatty third-party libraries. torio/torchaudio probe for optional
    # FFmpeg .pyd extensions at import and DEBUG-log the (expected) FileNotFound
    # tracebacks when they are absent -- Silero VAD uses the PyTorch backend, so
    # those 4 tracebacks every boot are pure noise. Raise them to WARNING.
    for noisy in ("numba", "matplotlib", "urllib3", "huggingface_hub",
                  "torio", "torchaudio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``kenning``."""
    if not name.startswith("kenning"):
        name = f"kenning.{name}"
    return logging.getLogger(name)
