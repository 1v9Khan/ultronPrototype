"""Thread-safe append-only writer for :class:`Observation` rows.

The writer never raises to callers. IO failures are logged WARN once
per (failure-type, path) pair and the affected observation is dropped --
the voice path must never block on observation IO.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from .schema import Observation

LOGGER = logging.getLogger("ultron.observations")

_DEFAULT_PATH = Path("data") / "observations.jsonl"


class ObservationWriter:
    """Append :class:`Observation` rows to a JSONL file.

    Parameters
    ----------
    path:
        Target JSONL file. Parents are created on first successful write.
    enabled:
        When False every :meth:`emit` is a no-op. Useful for tests or
        operators who want to suppress observation IO entirely without
        editing call sites.
    """

    def __init__(self, path: Path = _DEFAULT_PATH, *, enabled: bool = True) -> None:
        self._path = Path(path)
        self._enabled = enabled
        self._lock = threading.Lock()
        # (error_type, path-as-str) pairs we've already warned about, so
        # we don't spam the log if the disk is unwritable.
        self._warned: set[tuple[str, str]] = set()
        self._dropped = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def dropped(self) -> int:
        """Count of observations dropped due to IO failure."""
        return self._dropped

    def set_enabled(self, enabled: bool) -> None:
        """Toggle writer enable state at runtime."""
        self._enabled = enabled

    def emit(self, observation: Observation) -> bool:
        """Serialize + append ``observation``. Returns True on success.

        Never raises. Returns False when disabled or on IO failure
        (after logging WARN on the first failure of its kind).
        """
        if not self._enabled:
            return False
        try:
            line = json.dumps(observation.to_dict(), separators=(",", ":")) + "\n"
        except (TypeError, ValueError) as exc:
            self._warn_once(
                f"serialize:{type(exc).__name__}",
                f"failed to serialize observation {observation.event_id}: {exc}",
            )
            self._dropped += 1
            return False

        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
            return True
        except OSError as exc:
            self._warn_once(
                f"io:{type(exc).__name__}",
                f"failed to append to {self._path}: {exc}",
            )
            self._dropped += 1
            return False

    def _warn_once(self, key_suffix: str, message: str) -> None:
        key = (key_suffix, str(self._path))
        if key in self._warned:
            return
        self._warned.add(key)
        LOGGER.warning("observation-writer %s", message)

    def reset_warning_state(self) -> None:
        """Re-arm one-shot warnings. Test-only."""
        self._warned.clear()


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------


_singleton: Optional[ObservationWriter] = None
_singleton_lock = threading.Lock()


def get_observation_writer() -> ObservationWriter:
    """Return the process-wide :class:`ObservationWriter` singleton.

    Constructs a default writer pointed at ``data/observations.jsonl``
    on first call.
    """
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = ObservationWriter(_DEFAULT_PATH)
    return _singleton


def set_observation_writer(writer: Optional[ObservationWriter]) -> None:
    """Replace (or clear) the singleton. Test injection hook."""
    global _singleton
    with _singleton_lock:
        _singleton = writer


def emit_observation(observation: Observation) -> bool:
    """Convenience: emit ``observation`` via the singleton."""
    return get_observation_writer().emit(observation)
