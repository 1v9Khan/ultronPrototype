"""Marker-comment-detecting idempotent file installer.

Clean-room adaptation of the OpenHands ``maybe_setup_git_hooks`` install
pattern (``app_conversation_service_base.py``). The OpenHands version
specifically targets ``.git/hooks/pre-commit`` with the literal marker
``"This hook was installed by OpenHands"``. This module generalises that
shape to ANY install target with a configurable marker substring and
optional preserve-existing-as semantics.

The marker design here adds a UUID-style suffix to defend against the
catalog's documented foot-gun: a user accidentally writing the literal
marker text into a hook of their own. Ultron's marker is
``# INSTALLED-BY-ULTRON-3f9a7d2`` (the suffix matches the catalog's
worked example).

A best-effort audit log keeps a JSONL trail of every install action so
``ultron diag installs`` can summarise what ultron has ever touched.
Pattern lineage attributed in ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

DEFAULT_MARKER: str = "# INSTALLED-BY-ULTRON-3f9a7d2"
"""Default marker substring tagging files written by ultron's installer."""

DEFAULT_PRESERVE_SUFFIX: str = ".local"
"""Default suffix appended when preserving an existing non-ultron file."""

DEFAULT_INSTALL_LOG_PATH = Path("logs") / "install_log.jsonl"
"""Default path (relative to project root) for the install audit log."""


class InstallAction(str, Enum):
    """What an install_with_marker call did to disk.

    Values:
        INSTALLED: No prior file existed; the new content was written.
        SKIPPED_ALREADY_MARKED: An existing file already contains the marker;
            no write happened.
        REPLACED_UNMARKED: An existing file without the marker was REPLACED
            in place (no preservation requested).
        MOVED_THEN_INSTALLED: An existing unmarked file was moved to the
            ``preserve_existing_as`` path before the new content was written.
        DRY_RUN: ``dry_run=True`` was passed; no disk change occurred.
    """

    INSTALLED = "installed"
    SKIPPED_ALREADY_MARKED = "skipped_already_marked"
    REPLACED_UNMARKED = "replaced_unmarked"
    MOVED_THEN_INSTALLED = "moved_then_installed"
    DRY_RUN = "dry_run"


@dataclass(frozen=True)
class InstallResult:
    """Outcome of a single :func:`install_with_marker` call.

    Attributes:
        target_path: The file install_with_marker wrote (or would have
            written) to.
        action: The :class:`InstallAction` value describing what happened.
        marker: The marker substring used.
        moved_to: When ``action == MOVED_THEN_INSTALLED``, the path the
            previous content was renamed to. ``None`` otherwise.
        bytes_written: Length of the content written, ``0`` for no-op
            actions.
        already_present: ``True`` when the marker was found in an existing
            file (i.e. ``action == SKIPPED_ALREADY_MARKED``).
        error: ``None`` on success; otherwise a short description of why
            the install was not completed (writes fail-open by default).
    """

    target_path: Path
    action: InstallAction
    marker: str
    moved_to: Path | None = None
    bytes_written: int = 0
    already_present: bool = False
    error: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """``True`` iff the install completed without an error."""

        return self.error is None

    @property
    def changed_disk(self) -> bool:
        """``True`` iff disk state actually changed (file written or moved)."""

        return self.action in (
            InstallAction.INSTALLED,
            InstallAction.REPLACED_UNMARKED,
            InstallAction.MOVED_THEN_INSTALLED,
        )


@dataclass(frozen=True)
class InstallLogEntry:
    """One row of the install audit log."""

    timestamp: float
    target_path: str
    action: str
    marker: str
    moved_to: str | None
    bytes_written: int
    error: str | None
    extra: dict[str, str] = field(default_factory=dict)


class InstallLogWriter:
    """Append-only JSONL writer for install audit entries.

    Default location is ``logs/install_log.jsonl`` under the project root.
    Failures are swallowed at the WARN level so install operations are
    never blocked by audit-log I/O issues.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._lock = threading.Lock()
        self._path: Path | None = Path(path) if path else None

    def configure(self, path: Path | str | None) -> None:
        """Update the on-disk target for the audit log."""

        with self._lock:
            self._path = Path(path) if path else None

    @property
    def path(self) -> Path | None:
        with self._lock:
            return self._path

    def write(self, entry: InstallLogEntry) -> None:
        """Append one entry. Best-effort; logs WARN on failure."""

        with self._lock:
            target = self._path
        if target is None:
            return

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            row = json.dumps(
                {
                    "timestamp": entry.timestamp,
                    "target_path": entry.target_path,
                    "action": entry.action,
                    "marker": entry.marker,
                    "moved_to": entry.moved_to,
                    "bytes_written": entry.bytes_written,
                    "error": entry.error,
                    "extra": entry.extra,
                },
                ensure_ascii=False,
            )
            with target.open("a", encoding="utf-8") as fp:
                fp.write(row + "\n")
        except OSError as exc:
            logger.warning("install audit write failed for %s: %s", target, exc)


_DEFAULT_WRITER = InstallLogWriter()


def set_install_log_writer(writer: InstallLogWriter | None) -> None:
    """Swap the module-level writer (testing + advanced override)."""

    global _DEFAULT_WRITER
    if writer is None:
        _DEFAULT_WRITER = InstallLogWriter()
    else:
        _DEFAULT_WRITER = writer


def _atomic_write(target: Path, content: str, *, encoding: str) -> int:
    """Write ``content`` to ``target`` atomically via tmp + rename.

    Returns the number of bytes written. Raises :class:`OSError` on
    failure; callers decide whether to surface or swallow.
    """

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".ultron.tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fp:
            fp.write(content)
        # Best-effort fsync; ignore on platforms that don't support it.
        try:
            with target.parent.open("r") if False else open(tmp_path, "r+b") as raw:
                os.fsync(raw.fileno())
        except OSError:
            pass
        # ``os.replace`` is atomic on POSIX + Windows when both paths share a volume.
        os.replace(tmp_path, target)
    except Exception:
        # Clean up the orphaned tmp file before re-raising.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return len(content.encode(encoding))


def install_with_marker(
    target_path: Path | str,
    content: str,
    *,
    marker: str = DEFAULT_MARKER,
    preserve_existing_as: Path | str | None = None,
    replace_unmarked: bool = False,
    encoding: str = "utf-8",
    audit_writer: InstallLogWriter | None = None,
    extra: dict[str, str] | None = None,
    dry_run: bool = False,
) -> InstallResult:
    """Install ``content`` to ``target_path`` idempotently.

    Algorithm:
        1. If ``target_path`` does not exist: write ``content`` (action
           :attr:`InstallAction.INSTALLED`).
        2. If it exists and ``marker`` appears in the current content: no
           write happens (action :attr:`InstallAction.SKIPPED_ALREADY_MARKED`).
        3. If it exists, the marker is absent, and ``preserve_existing_as``
           is set: rename the existing content to that path, then write
           ``content`` (action :attr:`InstallAction.MOVED_THEN_INSTALLED`).
        4. If it exists, the marker is absent, and ``replace_unmarked`` is
           ``True`` (and no preserve target was provided): overwrite
           (action :attr:`InstallAction.REPLACED_UNMARKED`).
        5. Otherwise: refuse to overwrite the unmarked existing file
           (result has ``error`` populated; nothing is written).

    Args:
        target_path: Where to write.
        content: Content to write. Should contain ``marker`` somewhere so
            future calls recognise this file as ultron-installed.
        marker: Substring used to detect "ultron already installed this".
            Default :data:`DEFAULT_MARKER` (UUID-suffixed).
        preserve_existing_as: When the target exists without the marker,
            rename it to this path before writing the new content.
            Default ``None`` (no preservation).
        replace_unmarked: When ``True`` AND ``preserve_existing_as`` is
            ``None`` AND the target exists without the marker, overwrite
            it. Default ``False`` (errors instead).
        encoding: Text encoding (default ``utf-8``).
        audit_writer: Optional explicit writer; defaults to the module-
            level singleton.
        extra: Additional key/value metadata recorded into the audit log.
        dry_run: When ``True``, no disk changes occur; the returned result
            describes what WOULD have happened.

    Returns:
        :class:`InstallResult`. The result is always populated; the
        ``error`` field is non-``None`` when the install was refused or
        failed.
    """

    resolved_target = Path(target_path)
    resolved_preserve = Path(preserve_existing_as) if preserve_existing_as else None
    extra_metadata = dict(extra) if extra else {}
    writer = audit_writer if audit_writer is not None else _DEFAULT_WRITER

    if marker and marker not in content:
        # The catalog calls this out explicitly: an install whose body
        # doesn't contain the marker would re-install on every call.
        logger.warning(
            "install_with_marker: content for %s does not contain marker %r; "
            "future calls cannot detect this as already-installed",
            resolved_target,
            marker,
        )
        extra_metadata.setdefault("marker_in_content", "false")
    else:
        extra_metadata.setdefault("marker_in_content", "true")

    existing_text: str | None = None
    existed = resolved_target.exists()
    if existed:
        try:
            existing_text = resolved_target.read_text(encoding=encoding)
        except (OSError, UnicodeDecodeError) as exc:
            message = f"could not read existing file: {exc}"
            logger.warning("install_with_marker[%s]: %s", resolved_target, message)
            result = InstallResult(
                target_path=resolved_target,
                action=InstallAction.SKIPPED_ALREADY_MARKED,
                marker=marker,
                error=message,
                extra=extra_metadata,
            )
            _emit_audit(writer, result)
            return result

    if existing_text is not None and marker and marker in existing_text:
        result = InstallResult(
            target_path=resolved_target,
            action=InstallAction.SKIPPED_ALREADY_MARKED,
            marker=marker,
            already_present=True,
            extra=extra_metadata,
        )
        _emit_audit(writer, result)
        return result

    if dry_run:
        result = InstallResult(
            target_path=resolved_target,
            action=InstallAction.DRY_RUN,
            marker=marker,
            extra=extra_metadata,
        )
        _emit_audit(writer, result)
        return result

    if existed and existing_text is not None:
        if resolved_preserve is not None:
            try:
                resolved_preserve.parent.mkdir(parents=True, exist_ok=True)
                os.replace(resolved_target, resolved_preserve)
            except OSError as exc:
                message = f"could not preserve existing file: {exc}"
                logger.warning("install_with_marker[%s]: %s", resolved_target, message)
                result = InstallResult(
                    target_path=resolved_target,
                    action=InstallAction.SKIPPED_ALREADY_MARKED,
                    marker=marker,
                    error=message,
                    extra=extra_metadata,
                )
                _emit_audit(writer, result)
                return result
            try:
                bytes_written = _atomic_write(resolved_target, content, encoding=encoding)
            except OSError as exc:
                message = f"write failed after preserve: {exc}"
                logger.warning("install_with_marker[%s]: %s", resolved_target, message)
                # Best-effort: restore the preserved file so we don't leave
                # the user without their original content.
                try:
                    os.replace(resolved_preserve, resolved_target)
                except OSError:
                    pass
                result = InstallResult(
                    target_path=resolved_target,
                    action=InstallAction.SKIPPED_ALREADY_MARKED,
                    marker=marker,
                    moved_to=resolved_preserve,
                    error=message,
                    extra=extra_metadata,
                )
                _emit_audit(writer, result)
                return result
            result = InstallResult(
                target_path=resolved_target,
                action=InstallAction.MOVED_THEN_INSTALLED,
                marker=marker,
                moved_to=resolved_preserve,
                bytes_written=bytes_written,
                extra=extra_metadata,
            )
            _emit_audit(writer, result)
            return result

        if replace_unmarked:
            try:
                bytes_written = _atomic_write(resolved_target, content, encoding=encoding)
            except OSError as exc:
                message = f"replace_unmarked write failed: {exc}"
                logger.warning("install_with_marker[%s]: %s", resolved_target, message)
                result = InstallResult(
                    target_path=resolved_target,
                    action=InstallAction.SKIPPED_ALREADY_MARKED,
                    marker=marker,
                    error=message,
                    extra=extra_metadata,
                )
                _emit_audit(writer, result)
                return result
            result = InstallResult(
                target_path=resolved_target,
                action=InstallAction.REPLACED_UNMARKED,
                marker=marker,
                bytes_written=bytes_written,
                extra=extra_metadata,
            )
            _emit_audit(writer, result)
            return result

        # Existed, unmarked, neither preserve nor replace requested -> refuse.
        message = (
            "target exists without marker; supply preserve_existing_as or "
            "replace_unmarked=True to overwrite"
        )
        result = InstallResult(
            target_path=resolved_target,
            action=InstallAction.SKIPPED_ALREADY_MARKED,
            marker=marker,
            error=message,
            extra=extra_metadata,
        )
        _emit_audit(writer, result)
        return result

    # New file path: target did not exist.
    try:
        bytes_written = _atomic_write(resolved_target, content, encoding=encoding)
    except OSError as exc:
        message = f"initial write failed: {exc}"
        logger.warning("install_with_marker[%s]: %s", resolved_target, message)
        result = InstallResult(
            target_path=resolved_target,
            action=InstallAction.SKIPPED_ALREADY_MARKED,
            marker=marker,
            error=message,
            extra=extra_metadata,
        )
        _emit_audit(writer, result)
        return result

    result = InstallResult(
        target_path=resolved_target,
        action=InstallAction.INSTALLED,
        marker=marker,
        bytes_written=bytes_written,
        extra=extra_metadata,
    )
    _emit_audit(writer, result)
    return result


def _emit_audit(writer: InstallLogWriter, result: InstallResult) -> None:
    """Translate an :class:`InstallResult` into an :class:`InstallLogEntry` row."""

    entry = InstallLogEntry(
        timestamp=time.time(),
        target_path=str(result.target_path),
        action=result.action.value,
        marker=result.marker,
        moved_to=str(result.moved_to) if result.moved_to else None,
        bytes_written=result.bytes_written,
        error=result.error,
        extra=dict(result.extra),
    )
    writer.write(entry)


# Ensure the module always has a usable default audit writer pointing at
# the project-relative path. Tests / callers can replace via
# :func:`set_install_log_writer`. The path may not exist yet; the writer
# creates the parent directory lazily on first write.
_DEFAULT_WRITER.configure(DEFAULT_INSTALL_LOG_PATH)
