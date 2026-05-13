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
from typing import Any, Optional

logger = logging.getLogger("ultron.safety.audit")

DEFAULT_AUDIT_PATH = "logs/safety_audit.jsonl"

# Hash of the genesis entry (no previous). 64 zeros so the chain has
# a well-defined starting point.
GENESIS_PREV_HASH = "0" * 64


def _hash_line(line: str) -> str:
    """SHA-256 of a serialised log line (without trailing newline)."""
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


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
            last = lines[-1]
            if not last:
                return GENESIS_PREV_HASH
            return _hash_line(last.decode("utf-8", errors="replace"))
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
        """
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
                entry["context"] = context
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
