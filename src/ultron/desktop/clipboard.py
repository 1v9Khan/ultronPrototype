"""System clipboard read / write with safety validator + taint integration.

Catalog 09 T4 (YELLOW): closes the "no clipboard abstraction" gap in
ultron's desktop stack. The clipboard is the natural data-transfer
channel between desktop applications:

* "Copy the error message from the terminal" -> read.
* "Paste my prepared input into the form" -> write + Ctrl+V.
* "What's in my clipboard?" -> read + summarise.
* "Save the output for the next step" -> write.

Safety gating
=============

Reads (Cap-2 -- ephemeral observation):

* The clipboard can hold sensitive content (passwords, private keys,
  partial credit-card numbers, confidential documents). Every read
  records the returned bytes in :mod:`ultron.safety.taint` so any
  subsequent outbound tool call carrying those exact bytes trips the
  validator's exfil-detection layer.
* Reads do NOT require explicit-intent unless the caller is on the
  voice path (Cap-3). The :func:`ClipboardManager.read_text` surface
  accepts a ``user_text`` kwarg that the validator can match against
  the explicit-intent corpus when relevant.

Writes (Cap-3 -- user-directed mutation):

* Writing to the clipboard sets up a downstream Ctrl+V that lands in
  whatever app is foreground when the user hits paste. The validator
  receives the full text (or a preview when >2KB) so payload-based
  rules can block. The taint tracker records the written bytes so
  the orchestrator can verify a paste lands in the expected target.

Pyperclip absence
=================

The module deliberately uses a lazy ``try/import pyperclip`` pattern
rather than a top-of-file import. If pyperclip is missing or the
underlying win32clipboard / xclip / pbpaste binary is unavailable,
the methods log WARN and return a structured failure result instead
of raising. This mirrors the upstream clawhub-desktop-control plugin
(which catches ``ImportError`` separately from general ``Exception``)
but adds a typed :class:`ClipboardResult` carrying the failure mode
so orchestrators can branch on it.

Cross-platform: pyperclip uses ``win32clipboard`` on Windows,
``xclip`` / ``xsel`` on Linux, and ``pbcopy`` / ``pbpaste`` on macOS.
Ultron's production target is Windows but the abstraction stays
portable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ultron.utils.logging import get_logger

logger = get_logger("desktop.clipboard")


# Soft cap on payload size that's passed verbatim through the
# validator. Beyond this we forward only a 2 KB preview so the
# validator audit log stays bounded. The full bytes still go through
# the taint tracker so exfil detection sees the whole payload.
_VALIDATOR_PAYLOAD_PREVIEW_CHARS = 2048


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipboardResult:
    """Outcome of one clipboard read / write.

    Attributes:
        success: True iff the operation completed without error.
        action: ``"read"`` or ``"write"``.
        text: read content (only set on successful reads; None on
            writes and on failures).
        error: structured error message when ``success=False``.
        tainted: True iff the returned text was recorded in the
            taint tracker (always True for successful reads when
            taint recording is enabled; False on writes and failures).
    """

    success: bool
    action: str
    text: Optional[str] = None
    error: Optional[str] = None
    tainted: bool = False


# ---------------------------------------------------------------------------
# Safety validator hook
# ---------------------------------------------------------------------------


def _validate_clipboard_action(
    *,
    action: str,
    arguments: dict,
    user_text: str = "",
) -> object:
    """Run the safety validator against a clipboard read / write.

    Reads register under ``desktop.clipboard.read`` (capability
    ``clipboard_read``); writes under ``desktop.clipboard.write``
    (capability ``clipboard_write``). The validator can deny writes
    based on payload content (e.g. detected credential patterns,
    K-protected configuration keys) and deny reads based on context
    (e.g. an LLM that just lost focus shouldn't be reading the
    clipboard a second later).

    Fail-open at the import boundary so a broken safety stack doesn't
    deny legitimate clipboard ops; the validator unavailable case
    returns ALLOW.
    """
    try:
        from ultron.safety.validator import RuleContext, get_validator

        ctx = RuleContext(
            tool_name=f"desktop.clipboard.{action}",
            arguments=arguments,
            capability=f"clipboard_{action}",
            user_text=user_text,
        )
        return get_validator().check(ctx)
    except Exception as e:  # noqa: BLE001
        logger.debug("clipboard validator skipped: %s", e)
        from ultron.safety.validator import ValidatorVerdict, Verdict
        return ValidatorVerdict(
            verdict=Verdict.ALLOW, reason="validator unavailable",
        )


# ---------------------------------------------------------------------------
# pyperclip lazy import
# ---------------------------------------------------------------------------


def _import_pyperclip():
    """Lazy import so module load doesn't pay the pyperclip cost.

    Returns the ``pyperclip`` module or None when unavailable. Mirrors
    the upstream plugin's pattern of catching ``ImportError`` separately
    from the broader operational exceptions.
    """
    try:
        import pyperclip  # type: ignore[import]
        return pyperclip
    except ImportError as e:
        logger.warning("pyperclip unavailable: %s", e)
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("pyperclip import raised unexpectedly: %s", e)
        return None


# ---------------------------------------------------------------------------
# ClipboardManager
# ---------------------------------------------------------------------------


class ClipboardManager:
    """Safety-gated clipboard read / write.

    The orchestrator holds a single instance; downstream callers route
    through it rather than touching ``pyperclip`` directly so the
    safety validator and taint tracker fire uniformly. Like
    :class:`ultron.desktop.input_control.InputController`, the manager
    accepts injected hooks for the validator and taint tracker so
    tests can pin behaviour without monkey-patching the singletons.
    """

    def __init__(
        self,
        *,
        record_taint: bool = True,
        max_read_chars: int = 256_000,
        max_write_chars: int = 256_000,
    ) -> None:
        self._record_taint = bool(record_taint)
        self._max_read_chars = int(max_read_chars)
        self._max_write_chars = int(max_write_chars)

    # ---- read ----

    def read_text(self, *, user_text: str = "") -> ClipboardResult:
        """Return the current clipboard text.

        Records the returned text in the taint tracker (when enabled)
        so any subsequent outbound tool call carrying these exact
        bytes trips the safety validator's exfil check.

        Args:
            user_text: forwarded to the validator so Cap-2 explicit-
                intent rules can verify the user actually asked to
                read the clipboard.

        Returns:
            :class:`ClipboardResult` with ``action="read"``. On
            success, ``text`` holds the clipboard content (capped at
            ``max_read_chars``) and ``tainted`` reflects whether the
            taint tracker recorded the bytes.
        """
        # Anticheat-safe mode: hard-blocked while the user is in game.
        from ultron.safety.anticheat import guard as _anticheat_guard
        _anticheat_guard('clipboard_read')
        # Validator first -- it can refuse the read entirely before we
        # even touch pyperclip.
        verdict = _validate_clipboard_action(
            action="read",
            arguments={"max_read_chars": self._max_read_chars},
            user_text=user_text,
        )
        if not verdict.is_allowed:
            return ClipboardResult(
                success=False,
                action="read",
                error=f"safety: {verdict.reason}",
            )

        pyperclip = _import_pyperclip()
        if pyperclip is None:
            return ClipboardResult(
                success=False, action="read",
                error="pyperclip unavailable",
            )

        try:
            raw = pyperclip.paste()
        except Exception as e:  # noqa: BLE001
            logger.warning("clipboard read failed: %s", e)
            return ClipboardResult(
                success=False, action="read", error=str(e)[:200],
            )

        if raw is None:
            # Some pyperclip backends return None on an empty clipboard
            # (rather than an empty string). Treat as empty success.
            raw = ""

        if not isinstance(raw, str):
            try:
                raw = str(raw)
            except Exception as e:  # noqa: BLE001
                return ClipboardResult(
                    success=False, action="read",
                    error=f"non-string clipboard content: {e}",
                )

        if len(raw) > self._max_read_chars:
            raw = raw[: self._max_read_chars]

        # Record the bytes so downstream exfil detection works.
        tainted = False
        if self._record_taint and raw:
            tainted = self._record_taint_safe(raw, capability="clipboard_read")

        return ClipboardResult(
            success=True, action="read", text=raw, tainted=tainted,
        )

    # ---- write ----

    def write_text(
        self,
        text: str,
        *,
        user_text: str = "",
    ) -> ClipboardResult:
        """Write ``text`` to the system clipboard.

        Routes through the safety validator with full payload (or 2 KB
        preview for very large content) so payload-based rules can
        block. The validator audit log carries the preview; the taint
        tracker records the full bytes for downstream verification.

        Args:
            text: content to write. Coerced to ``str``. Empty string
                is a valid (no-op-ish) write that clears the clipboard.
            user_text: forwarded to the validator so the Cap-3
                explicit-intent matcher can verify the user asked for
                a clipboard write.

        Returns:
            :class:`ClipboardResult` with ``action="write"``.
        """
        # Anticheat-safe mode: hard-blocked while the user is in game.
        from ultron.safety.anticheat import guard as _anticheat_guard
        _anticheat_guard('clipboard_write')
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:  # noqa: BLE001
                return ClipboardResult(
                    success=False, action="write",
                    error="text must be coercible to str",
                )

        if len(text) > self._max_write_chars:
            return ClipboardResult(
                success=False, action="write",
                error=(
                    f"text length {len(text)} exceeds max_write_chars "
                    f"{self._max_write_chars}"
                ),
            )

        preview = text[:_VALIDATOR_PAYLOAD_PREVIEW_CHARS]
        verdict = _validate_clipboard_action(
            action="write",
            arguments={
                "text_preview": preview,
                "length": len(text),
            },
            user_text=user_text,
        )
        if not verdict.is_allowed:
            return ClipboardResult(
                success=False,
                action="write",
                error=f"safety: {verdict.reason}",
            )

        pyperclip = _import_pyperclip()
        if pyperclip is None:
            return ClipboardResult(
                success=False, action="write",
                error="pyperclip unavailable",
            )

        try:
            pyperclip.copy(text)
        except Exception as e:  # noqa: BLE001
            logger.warning("clipboard write failed: %s", e)
            return ClipboardResult(
                success=False, action="write", error=str(e)[:200],
            )

        # Record the written bytes so the orchestrator can verify a
        # downstream paste lands in the expected target.
        tainted = False
        if self._record_taint and text:
            tainted = self._record_taint_safe(
                text, capability="clipboard_write",
            )

        return ClipboardResult(
            success=True, action="write", tainted=tainted,
        )

    # ---- internals ----

    @staticmethod
    def _record_taint_safe(text: str, *, capability: str) -> bool:
        """Record clipboard bytes in the taint tracker. Fail-open."""
        try:
            from ultron.safety.taint import get_taint_tracker

            data = text.encode("utf-8", errors="replace")
            get_taint_tracker().record(data=data, capability=capability)
            return True
        except Exception as e:  # noqa: BLE001
            logger.debug("clipboard taint record skipped: %s", e)
            return False


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


_manager_singleton: Optional[ClipboardManager] = None


def get_clipboard_manager() -> ClipboardManager:
    """Module-level singleton accessor."""
    global _manager_singleton
    if _manager_singleton is None:
        _manager_singleton = ClipboardManager()
    return _manager_singleton


def set_clipboard_manager(manager: Optional[ClipboardManager]) -> None:
    """Test / orchestrator hook -- swap the singleton."""
    global _manager_singleton
    _manager_singleton = manager


__all__ = [
    "ClipboardResult",
    "ClipboardManager",
    "get_clipboard_manager",
    "set_clipboard_manager",
]
