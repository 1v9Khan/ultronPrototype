"""Audit log for the runtime tool-call validator.

Append-only JSONL with a tamper-evident hash chain.

* Each entry includes a ``prev_hash`` field: SHA-256 of the previous
  entry's serialised JSON (line excluding the trailing newline).
* The first entry has ``prev_hash = "0" * 64`` (genesis).
* On open, the writer reads the file's tail entry and remembers its
  hash. Subsequent writes link from there.
* :meth:`verify_chain` walks the log from start to tail and confirms
  every entry's ``prev_hash`` matches the previous entry's computed
  hash. Returns the verification result so the validator can refuse
  to start when the chain is broken (operator must investigate
  before continuing -- something tampered with the log).

This defends against an attacker who gains write access to the log
file and tries to retroactively delete entries. Any retroactive
deletion breaks the chain at the deletion point and downstream;
verify_chain detects it.

What this does NOT defend against:

* An attacker who can stop the validator AND replace the log file
  entirely with a freshly-generated valid chain. That's outside the
  threat model -- if the attacker has that level of write access,
  they can also disable the validator binary.
* Real-time tampering during writes. The hash chain is checked at
  startup, not per-entry; mid-session tampering is detected the next
  time the chain is verified.

Other Phase 2 invariants preserved:
- Append-only JSONL.
- ``os.fsync`` after every write so a crash doesn't lose the most
  recent decisions.
- Thread-safe via an internal lock.
- Module-level singleton accessor.

Schema of each entry::

    {
      "ts": "2026-05-12T18:30:45.123456",       # ISO-8601 timestamp
      "rule_id": "K1",                           # which rule fired
      "verdict": "BLOCK_HARD",                   # the verdict
      "tool_name": "write_file",                 # what was attempted
      "capability": "openclaw_dispatcher",       # where the call came from
      "reason": "config.yaml safety.rules edit", # short human reason
      "context": { ... },                        # rule-specific details
      "prev_hash": "abc123..."                   # SHA-256 of previous entry
    }
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from kenning.utils.ansi_safe import sanitize_for_log

logger = logging.getLogger("kenning.safety.audit")

DEFAULT_AUDIT_PATH = "logs/safety_audit.jsonl"

# Hash of the genesis entry (no previous). 64 zeros so the chain has
# a well-defined starting point.
GENESIS_PREV_HASH = "0" * 64


def _hash_line(line: str) -> str:
    """SHA-256 of a serialised log line (without trailing newline)."""
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def _sanitize_context(value: Any) -> Any:
    """Recursively strip ANSI + control chars from string values inside
    a context dict / list / scalar.

    Used at audit-write time to ensure tool-supplied strings (which can
    embed CR/CSI cursor-jump bytes intended to forge log lines) are
    neutered before they reach the JSONL. Non-string leaves pass
    through unchanged. Fail-open per leaf: a sanitiser exception leaves
    the original value in place rather than dropping the record.
    """
    try:
        if isinstance(value, str):
            return sanitize_for_log(value)
        if isinstance(value, dict):
            return {
                str(k): _sanitize_context(v) for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            cleaned = [_sanitize_context(v) for v in value]
            return cleaned if isinstance(value, list) else tuple(cleaned)
        return value
    except Exception:  # noqa: BLE001
        return value


class AuditLog:
    """Append-only JSONL writer with tamper-evident hash chain.

    Thread-safe. Each :meth:`record` call:

    1. Builds the entry dict with a ``prev_hash`` field linking to
       the previously-written entry.
    2. Serialises to JSON, computes the entry's own hash.
    3. Appends + fsyncs.

    On construction the writer reads the file's tail line (if any)
    and remembers its hash so the next entry links correctly. Call
    :meth:`verify_chain` at startup to confirm the entire chain is
    intact.

    Args:
        path: Where to write the log. Created (with parent dirs) if
            it doesn't exist. Defaults to ``logs/safety_audit.jsonl``
            relative to the current working directory.

    Notes:
        Failures to write the log do NOT block the validator's
        decision (the verdict is already made). They DO log a WARN
        to the standard logger. Rationale: if the disk is full or
        the log file is locked, we still want the validator to
        operate -- but the operator needs to know logging is broken.
    """

    def __init__(self, path: Optional[str | Path] = None) -> None:
        if path is None:
            path = DEFAULT_AUDIT_PATH
        self._path = Path(path)
        self._lock = threading.Lock()
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(
                "audit log parent dir creation failed (%s); writes will "
                "be best-effort to %s",
                e, self._path,
            )
        # Initialise the tail-hash so the next write links from the
        # right place. Existing-file tail wins; non-existent file
        # starts the chain at genesis.
        self._tail_hash = self._compute_tail_hash()

    @property
    def path(self) -> Path:
        return self._path

    def _compute_tail_hash(self) -> str:
        """Read the file's last line (if any) and return its SHA-256.

        Returns ``GENESIS_PREV_HASH`` when the file is absent or
        empty. Used during construction to bootstrap the chain.

        Strips Windows ``\\r\\n`` line endings as well as plain
        ``\\n`` -- text-mode writes translate ``\\n`` to ``\\r\\n``
        on Windows, but ``record`` hashes the JSON line BEFORE
        appending the newline, so the on-disk bytes include a
        trailing ``\\r`` that the hash didn't see.
        """
        try:
            if not self._path.is_file():
                return GENESIS_PREV_HASH
            with self._path.open("rb") as f:
                data = f.read()
            if not data:
                return GENESIS_PREV_HASH
            # Normalise CRLF -> LF first, then strip trailing
            # newlines and split.
            data = data.replace(b"\r\n", b"\n")
            lines = data.rstrip(b"\n").split(b"\n")
            # Hash the last line that is VALID JSON. A truncated final line
            # (process killed mid-write, BEFORE the os.fsync in record()) is a
            # never-committed write; linking the next entry to its hash would
            # break the chain, so walk backwards past any partial / blank tail.
            # repair_if_needed() does the authoritative on-disk truncation.
            for raw in reversed(lines):
                s = raw.decode("utf-8", errors="replace")
                if not s.strip():
                    continue
                try:
                    json.loads(s)
                except json.JSONDecodeError:
                    continue
                return _hash_line(s)
            return GENESIS_PREV_HASH
        except OSError as e:
            logger.warning(
                "audit log tail read failed (%s); starting from genesis", e,
            )
            return GENESIS_PREV_HASH

    def record(
        self,
        *,
        rule_id: str,
        verdict: str,
        tool_name: str,
        capability: str,
        reason: str,
        context: Optional[dict[str, Any]] = None,
        canonical_codes: Optional[Sequence[str]] = None,
        category: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Append one decision entry to the log + fsync.

        Args:
            rule_id: e.g. "K1", "A3", "D7". Matches the user's
                restriction-list numbering.
            verdict: ``ALLOW``, ``BLOCK_HARD``, ``NEEDS_EXPLICIT_INTENT``,
                ``LOG_ONLY``. The validator's enum value is stringified.
            tool_name: the tool the model tried to call.
            capability: where the call originated -- ``coding_bridge``,
                ``openclaw_dispatcher``, ``mcp_tool``, ``file_op``, etc.
            reason: short human-readable reason for the verdict.
            context: rule-specific details. Optional. Keep small; the
                log is hot.
            canonical_codes: optional sequence of T3 canonical reason
                codes (e.g. ``"kenning.malicious.k_category_violation"``).
                When supplied the entry carries them under the
                ``canonical_codes`` key so downstream consumers
                (dashboards, voice narration via
                :func:`kenning.install.reason_codes.summarize_reason_codes`)
                can cross-reference upstream advisories without parsing
                ``reason`` strings. Empty / None omits the field.
            category: optional T16 analytics label (rule-supplied).
                Surfaces in the audit row's ``category`` key so
                dashboards can group blocks without re-parsing
                ``reason``.
            metadata: optional T16 opaque per-rule blob. Surfaces in
                the audit row's ``rule_metadata`` key. Keep small;
                large blobs bloat the log.
        """
        # T18 (openclaw-main catalog port). CWE-117 defence:
        # tool-supplied strings (reason / tool_name / capability) may
        # contain ANSI escapes or C0 control chars (cursor-jump,
        # fake-newline) intended to forge log lines. Sanitise before
        # serialising. Fail-open: a sanitiser failure must not block
        # the record write.
        try:
            reason = sanitize_for_log(reason) if reason else reason
            tool_name = sanitize_for_log(tool_name) if tool_name else tool_name
            capability = sanitize_for_log(capability) if capability else capability
        except Exception as e:  # noqa: BLE001
            logger.debug("audit sanitize failed (proceeding): %s", e)

        with self._lock:
            prev_hash = self._tail_hash
            entry: dict[str, Any] = {
                "ts": datetime.utcnow().isoformat(),
                "rule_id": rule_id,
                "verdict": verdict,
                "tool_name": tool_name,
                "capability": capability,
                "reason": reason,
                "prev_hash": prev_hash,
            }
            if context is not None:
                entry["context"] = _sanitize_context(context)
            if canonical_codes:
                codes = tuple(
                    c for c in canonical_codes
                    if isinstance(c, str) and c.strip()
                )
                if codes:
                    entry["canonical_codes"] = list(codes)
            if category:
                entry["category"] = sanitize_for_log(str(category))
            if metadata:
                entry["rule_metadata"] = _sanitize_context(metadata)
            line = json.dumps(entry, ensure_ascii=False, default=str, sort_keys=True)
            try:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                # Advance the chain only on a successful write -- a
                # failed write must not move ``self._tail_hash`` or the
                # next entry will link to a hash that isn't in the file.
                self._tail_hash = _hash_line(line)
            except OSError as e:
                logger.warning(
                    "audit log write failed (%s); entry lost: %s",
                    e, line[:200],
                )

    def verify_chain(self) -> tuple[bool, str]:
        """Walk the log start-to-tail and confirm each ``prev_hash``
        links correctly.

        Returns:
            ``(True, "ok")`` when the chain is intact.
            ``(False, "<reason>")`` when broken -- the reason names
            the line number where the chain breaks. The validator
            should call this at startup; a broken chain means the log
            has been tampered with and operator action is required.

        An empty / missing file returns ``(True, "empty")`` -- a
        fresh install has nothing to verify.
        """
        if not self._path.is_file():
            return True, "empty"
        try:
            with self._path.open("r", encoding="utf-8") as f:
                lines = [ln.rstrip("\n") for ln in f.readlines()]
        except OSError as e:
            return False, f"read failed: {e}"

        # Strip any trailing blank lines (legitimate -- caused by
        # process killed mid-write).
        while lines and not lines[-1].strip():
            lines.pop()

        if not lines:
            return True, "empty"

        expected_prev = GENESIS_PREV_HASH
        for i, line in enumerate(lines):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                return False, f"line {i+1}: invalid JSON: {e}"
            actual_prev = entry.get("prev_hash")
            if actual_prev != expected_prev:
                return False, (
                    f"line {i+1}: prev_hash mismatch "
                    f"(expected {expected_prev[:12]}..., got "
                    f"{(actual_prev or 'missing')[:12]}...)"
                )
            expected_prev = _hash_line(line)
        return True, "ok"

    def repair_if_needed(self) -> str:
        """Heal a hash chain broken by an unclean shutdown.

        A kill between ``record()``'s ``write()`` and ``os.fsync()`` leaves a
        truncated final line whose hash the live log never linked from -> the
        next boot's ``verify_chain`` reports a ``prev_hash`` mismatch. This
        finds the longest VALID PREFIX from genesis (each line parses AND links
        correctly), truncates the file to it, and recomputes the tail hash.

        Only the never-fsync'd tail is removed -- no durably-committed record is
        ever dropped. Returns ``"ok"`` (intact/empty), ``"repaired"`` (truncated
        a partial/corrupt tail), or ``"restarted"`` (no valid prefix -> the file
        is ARCHIVED as ``.corrupt.<ts>``, never deleted, and the chain restarts
        from genesis). Never raises -- audit integrity must not block boot.
        """
        with self._lock:
            try:
                if not self._path.is_file():
                    self._tail_hash = GENESIS_PREV_HASH
                    return "ok"
                with self._path.open("r", encoding="utf-8") as f:
                    lines = [ln.rstrip("\n") for ln in f.readlines()]
                while lines and not lines[-1].strip():
                    lines.pop()
                if not lines:
                    self._tail_hash = GENESIS_PREV_HASH
                    return "ok"
                expected_prev = GENESIS_PREV_HASH
                valid = 0
                for line in lines:
                    if not line.strip():
                        break
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    if entry.get("prev_hash") != expected_prev:
                        break
                    expected_prev = _hash_line(line)
                    valid += 1
                if valid == len(lines):
                    self._tail_hash = expected_prev      # fully intact
                    return "ok"
                if valid == 0:
                    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
                    archive = self._path.with_name(self._path.name + f".corrupt.{ts}")
                    try:
                        self._path.replace(archive)
                    except OSError:
                        pass
                    self._tail_hash = GENESIS_PREV_HASH
                    return "restarted"
                # Truncate the file to the valid prefix (atomic full rewrite +
                # fsync). Text mode matches record()'s append convention.
                with self._path.open("w", encoding="utf-8") as f:
                    f.write("\n".join(lines[:valid]) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                self._tail_hash = expected_prev
                return "repaired"
            except Exception as e:                                # noqa: BLE001
                logger.warning(
                    "audit log repair failed (%s); leaving the log as-is", e)
                return "ok"


_audit_singleton: Optional[AuditLog] = None
_audit_lock = threading.Lock()


def get_audit_log() -> AuditLog:
    """Module-level singleton accessor.

    Constructed on first call from
    :data:`DEFAULT_AUDIT_PATH`. Use :func:`set_audit_log` in tests
    to swap in a tmpdir-backed instance.
    """
    global _audit_singleton
    if _audit_singleton is None:
        with _audit_lock:
            if _audit_singleton is None:
                _audit_singleton = AuditLog()
    return _audit_singleton


def set_audit_log(audit: Optional[AuditLog]) -> None:
    """Test hook: replace the singleton. ``None`` resets to lazy init."""
    global _audit_singleton
    with _audit_lock:
        _audit_singleton = audit
