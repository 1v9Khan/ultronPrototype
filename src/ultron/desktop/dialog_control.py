"""Native Windows dialog detection + CRUD interaction.

Catalog 08 T1 (YELLOW). Closes the "automation sequences stall on
dialogs" gap that the upstream clawhub-windows-control plugin's
``handle_dialog.py`` was designed for: coding tasks that pop a
save-as / overwrite-confirm / installer dialog used to require either
a screenshot+VLM round-trip (slow, error-prone) or a coordinate-based
pyautogui click (fragile). Native UIA-based dialog interaction is
neither: it sees the dialog as a tree of named buttons + fields and
can click + type by name in ~10-50 ms with no GPU cost.

Surface:

* :func:`find_dialogs` -- enumerate currently-open Windows dialogs.
  Matches against :data:`DIALOG_CLASSES` (standard Win32 dialog
  classes including ``#32770``) AND title-substring keywords
  (:data:`DIALOG_TITLE_KEYWORDS`); returns frozen :class:`DialogInfo`
  records carrying the live :class:`WindowInfo` for follow-up reads
  and writes.

* :func:`read_dialog` -- walk a dialog's UIA tree and return a
  :class:`DialogContent` snapshot: title + message text + per-button
  records + text fields + checkbox states + dropdowns + list items.
  Read-only; no input injection.

* :func:`click_dialog_button` -- find the first enabled Button with a
  matching name (case-insensitive substring by default; ``exact=True``
  switches to equality) and invoke ``click_input()``. Runs through
  :func:`ultron.safety.validator.ToolCallValidator.check` with
  ``tool_name="desktop.dialog.click_button"`` so Cap-3 verb-click
  rules + Cap-4 security-window guards + the explicit-intent matcher
  apply.

* :func:`type_into_dialog_field` -- find the indexed enabled Edit /
  ComboBox descendant and call ``set_text`` (preferred) or
  ``type_keys`` (fallback). Same Cap-3 gating.

* :func:`dismiss_dialog` -- iterate :data:`DISMISS_BUTTONS` in order,
  click the first one that exists, fall back to sending ``{ESC}`` via
  ``type_keys`` on the dialog root. Both fall-throughs are gated.

* :func:`wait_for_dialog` -- synchronous polling barrier that returns
  the first :class:`DialogInfo` matching an optional title filter,
  or None on timeout. Mirrors the upstream poll-every-500-ms shape
  with :func:`time.sleep` / :func:`time.monotonic` injectable for
  deterministic tests.

Gating architecture (per catalog 08 T1 YELLOW):

* Cap-2 (screen content read) covers :func:`find_dialogs` /
  :func:`read_dialog`. No explicit-intent requirement.
* Cap-3 (synthetic input) covers :func:`click_dialog_button` /
  :func:`type_into_dialog_field` / :func:`dismiss_dialog`. The
  validator's verb-click rules see ``"Submit"`` / ``"Pay"`` /
  ``"Send Money"`` button names and emit ``NEEDS_EXPLICIT_INTENT``,
  which blocks unless the user's recent utterance carries a matching
  verb+object pair.
* All write paths log to the safety audit log (via the validator's
  own audit hook) with dialog title + action + outcome so review can
  trace dialog interactions across sessions.

Fail-open at every layer: pywinauto / win32 unavailable -> empty
result, never raise. The orchestrator can call any of these from a
single try/except wrapping the whole automation step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ultron.desktop.windows import WindowInfo, enumerate_windows
from ultron.utils.logging import get_logger

logger = get_logger("desktop.dialog_control")


# Windows class names that identify dialog-style windows. ``#32770`` is
# the standard Win32 dialog box class; the rest are common XAML / UWP
# / Electron dialog wrappers + custom alert / popup conventions.
DIALOG_CLASSES: tuple[str, ...] = (
    "#32770",
    "Dialog",
    "MessageBox",
    "Alert",
    "Popup",
)

# UIA control types that report as dialog-like. ``Pane`` is included
# because Chrome / Electron browsers wrap their save-dialog UI in a
# Pane element when the OS-level dialog isn't used.
DIALOG_CONTROL_TYPES: tuple[str, ...] = (
    "Window",
    "Dialog",
    "Pane",
)

# Title-substring keywords that promote a window into the dialog set
# even when its class doesn't match a known dialog class. Lower-case
# for case-insensitive comparison.
DIALOG_TITLE_KEYWORDS: tuple[str, ...] = (
    "dialog",
    "save",
    "open",
    "confirm",
    "warning",
    "error",
    "alert",
)

# Buttons we try in order when auto-dismissing a dialog. The order is
# safety-conservative: "OK" / "Close" / "Cancel" don't commit
# destructive action on most dialogs; "Yes" / "No" do but are common
# for confirmation prompts; "Dismiss" / "Got it" / "Accept" / "Done"
# are common UWP / installer conventions.
DISMISS_BUTTONS: tuple[str, ...] = (
    "OK",
    "Close",
    "Cancel",
    "Yes",
    "No",
    "Dismiss",
    "Got it",
    "Accept",
    "Done",
)

# Polling defaults match the rest of the wait-primitive family in
# :mod:`ultron.desktop.uia` and :mod:`ultron.desktop.windows`.
DEFAULT_WAIT_TIMEOUT_S: float = 30.0
DEFAULT_WAIT_INTERVAL_S: float = 0.5


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DialogInfo:
    """One detected dialog window."""

    window: WindowInfo
    class_name: str
    control_type: str = ""
    matched_by: str = ""  # "class" / "control_type" / "title_keyword"

    @property
    def hwnd(self) -> int:
        return self.window.hwnd

    @property
    def title(self) -> str:
        return self.window.title


@dataclass(frozen=True)
class DialogButton:
    """Button entry in a :class:`DialogContent` snapshot."""

    name: str
    enabled: bool = True
    rect: tuple[int, int, int, int] = (0, 0, 0, 0)
    center: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class DialogField:
    """Editable text field (Edit / ComboBox) entry."""

    name: str
    control_type: str = "Edit"
    enabled: bool = True
    value: str = ""
    rect: tuple[int, int, int, int] = (0, 0, 0, 0)
    center: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class DialogCheckbox:
    """CheckBox / RadioButton entry with toggle state."""

    name: str
    control_type: str = "CheckBox"
    enabled: bool = True
    checked: Optional[bool] = None
    center: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class DialogContent:
    """Complete read-only snapshot of a dialog's interactive surface."""

    title: str
    message: tuple[str, ...] = ()
    buttons: tuple[DialogButton, ...] = ()
    text_fields: tuple[DialogField, ...] = ()
    checkboxes: tuple[DialogCheckbox, ...] = ()
    dropdowns: tuple[DialogField, ...] = ()
    list_items: tuple[str, ...] = ()
    elapsed_ms: float = 0.0


@dataclass(frozen=True)
class DialogActionResult:
    """Outcome of a write action against a dialog."""

    success: bool
    action: str
    dialog_title: str = ""
    target: str = ""
    method: str = ""  # "click" / "set_text" / "type_keys" / "escape"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Pywinauto helpers (lazy)
# ---------------------------------------------------------------------------


def _import_pywinauto():
    try:
        import pywinauto  # type: ignore[import]
        return pywinauto
    except Exception as exc:  # noqa: BLE001
        logger.debug("pywinauto unavailable: %s", exc)
        return None


def _connect_to_window(hwnd: int):
    pwa = _import_pywinauto()
    if pwa is None:
        return None
    try:
        app = pwa.Application(backend="uia").connect(handle=hwnd, timeout=2)
        return app.window(handle=hwnd)
    except Exception as exc:  # noqa: BLE001
        logger.debug("dialog_control connect hwnd=%d failed: %s", hwnd, exc)
        return None


def _rect_of(node) -> tuple[int, int, int, int]:
    try:
        r = node.rectangle()
        return (int(r.left), int(r.top), int(r.right), int(r.bottom))
    except Exception:  # noqa: BLE001
        return (0, 0, 0, 0)


def _center_of_rect(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    return ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2)


def _safe_text(node) -> str:
    try:
        return str(node.window_text() or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _safe_control_type(node) -> str:
    try:
        return str(node.element_info.control_type or "")
    except Exception:  # noqa: BLE001
        return ""


def _safe_is_enabled(node) -> bool:
    try:
        return bool(node.is_enabled())
    except Exception:  # noqa: BLE001
        return True


def _safe_class_name(node) -> str:
    try:
        return str(node.element_info.class_name or "")
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _matches_dialog(window: WindowInfo) -> Optional[str]:
    """Return the match kind (``"class"`` / ``"title_keyword"``) or None."""
    cls = (window.class_name or "")
    if cls:
        for candidate in DIALOG_CLASSES:
            if candidate in cls:
                return "class"
    title_lower = (window.title or "").lower()
    if title_lower:
        for kw in DIALOG_TITLE_KEYWORDS:
            if kw in title_lower:
                return "title_keyword"
    return None


def find_dialogs(
    *,
    partial_title_filter: Optional[str] = None,
    exclude_cloaked: bool = True,
    include_minimized: bool = False,
) -> list[DialogInfo]:
    """Enumerate currently-open dialog-style windows.

    Args:
        partial_title_filter: when set, only dialogs whose title
            contains this case-insensitive substring are returned.
        exclude_cloaked: forwarded to :func:`enumerate_windows`.
        include_minimized: include minimized windows (default False
            because a minimized dialog isn't interactable).

    Returns:
        List of :class:`DialogInfo` records in enumeration order.
        Empty when no dialogs are open or :func:`enumerate_windows`
        raised.
    """
    # Anticheat-safe mode: hard-blocked while the user is in game.
    from ultron.safety.anticheat import guard as _anticheat_guard
    _anticheat_guard('dialog_read')
    try:
        windows = enumerate_windows(
            include_minimized=include_minimized,
            exclude_cloaked=exclude_cloaked,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("find_dialogs enumerate failed: %s", exc)
        return []

    title_filter = (partial_title_filter or "").strip().lower()
    results: list[DialogInfo] = []

    for win in windows:
        matched_by = _matches_dialog(win)
        if matched_by is None:
            continue
        if title_filter and title_filter not in (win.title or "").lower():
            continue
        results.append(
            DialogInfo(
                window=win,
                class_name=win.class_name,
                control_type="",
                matched_by=matched_by,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def _coerce_hwnd(source: object) -> int:
    if isinstance(source, DialogInfo):
        return int(source.hwnd)
    if isinstance(source, WindowInfo):
        return int(source.hwnd)
    return int(source)


def read_dialog(
    source: object,
    *,
    max_descendants: int = 500,
    message_max: int = 8,
    text_truncate: int = 500,
) -> Optional[DialogContent]:
    """Walk a dialog's UIA tree and return a structured snapshot.

    Args:
        source: :class:`DialogInfo`, :class:`WindowInfo`, or raw hwnd.
        max_descendants: cap on tree elements visited.
        message_max: cap on message strings collected (some dialogs
            expose many Text/Static labels; only the first N matter
            for "what does this dialog say").
        text_truncate: per-string truncation length to bound memory.

    Returns:
        :class:`DialogContent` snapshot. ``None`` when pywinauto
        unavailable or the connect fails.
    """
    # Anticheat-safe mode: hard-blocked while the user is in game.
    from ultron.safety.anticheat import guard as _anticheat_guard
    _anticheat_guard('dialog_read')
    hwnd = _coerce_hwnd(source)
    spec = _connect_to_window(hwnd)
    if spec is None:
        return None

    started = time.monotonic()
    try:
        descendants = spec.descendants()
    except Exception as exc:  # noqa: BLE001
        logger.debug("dialog descendants() hwnd=%d failed: %s", hwnd, exc)
        return None

    title = _safe_text(spec)

    message: list[str] = []
    buttons: list[DialogButton] = []
    text_fields: list[DialogField] = []
    checkboxes: list[DialogCheckbox] = []
    dropdowns: list[DialogField] = []
    list_items: list[str] = []
    seen_messages: set[str] = set()
    visited = 0

    for node in descendants:
        if visited >= max_descendants:
            break
        visited += 1
        try:
            ctype = _safe_control_type(node)
            name = _safe_text(node)
            if name and len(name) > text_truncate:
                name = name[:text_truncate]

            if ctype == "Button" and name:
                rect = _rect_of(node)
                buttons.append(
                    DialogButton(
                        name=name,
                        enabled=_safe_is_enabled(node),
                        rect=rect,
                        center=_center_of_rect(rect),
                    )
                )
            elif ctype in ("Edit", "Document"):
                rect = _rect_of(node)
                value = ""
                try:
                    raw_value = getattr(node, "get_value", None)
                    if callable(raw_value):
                        value = str(raw_value() or "")[:text_truncate]
                except Exception:  # noqa: BLE001
                    value = ""
                text_fields.append(
                    DialogField(
                        name=name,
                        control_type=ctype,
                        enabled=_safe_is_enabled(node),
                        value=value,
                        rect=rect,
                        center=_center_of_rect(rect),
                    )
                )
            elif ctype in ("CheckBox", "RadioButton"):
                rect = _rect_of(node)
                toggle_state: Optional[bool] = None
                try:
                    is_checked = getattr(node, "is_checked", None)
                    if callable(is_checked):
                        toggle_state = bool(is_checked())
                except Exception:  # noqa: BLE001
                    toggle_state = None
                checkboxes.append(
                    DialogCheckbox(
                        name=name,
                        control_type=ctype,
                        enabled=_safe_is_enabled(node),
                        checked=toggle_state,
                        center=_center_of_rect(rect),
                    )
                )
            elif ctype == "ComboBox":
                rect = _rect_of(node)
                value = ""
                try:
                    selected = getattr(node, "selected_text", None)
                    if callable(selected):
                        value = str(selected() or "")[:text_truncate]
                except Exception:  # noqa: BLE001
                    value = ""
                dropdowns.append(
                    DialogField(
                        name=name,
                        control_type="ComboBox",
                        enabled=_safe_is_enabled(node),
                        value=value,
                        rect=rect,
                        center=_center_of_rect(rect),
                    )
                )
            elif ctype == "ListItem" and name:
                list_items.append(name)
            elif ctype in ("Text", "Static") and name:
                if name not in seen_messages and len(message) < message_max:
                    seen_messages.add(name)
                    message.append(name)
        except Exception:  # noqa: BLE001
            # Per-element fail-open mirrors the upstream's bare except: continue.
            continue

    elapsed_ms = (time.monotonic() - started) * 1000.0
    return DialogContent(
        title=title,
        message=tuple(message),
        buttons=tuple(buttons),
        text_fields=tuple(text_fields),
        checkboxes=tuple(checkboxes),
        dropdowns=tuple(dropdowns),
        list_items=tuple(list_items),
        elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Safety validator hook
# ---------------------------------------------------------------------------


def _validate_dialog_action(
    *,
    action: str,
    dialog_title: str,
    target: str,
    text: str = "",
    user_text: str = "",
) -> object:
    """Run the runtime tool-call validator against a write action."""
    try:
        from ultron.safety.validator import RuleContext, get_validator

        ctx = RuleContext(
            tool_name=f"desktop.dialog.{action}",
            arguments={
                "window_title": dialog_title,
                "element": f"'{target}'",
                "text": text,
            },
            capability="desktop_dialog",
            user_text=user_text,
        )
        return get_validator().check(ctx)
    except Exception as exc:  # noqa: BLE001
        logger.debug("dialog validator skipped: %s", exc)
        from ultron.safety.validator import ValidatorVerdict, Verdict
        return ValidatorVerdict(
            verdict=Verdict.ALLOW, reason="validator unavailable",
        )


# ---------------------------------------------------------------------------
# Write actions
# ---------------------------------------------------------------------------


def _find_button_by_name(spec, button_name: str, *, exact: bool = False):
    """Return the first enabled Button descendant whose name matches."""
    needle = button_name.strip()
    needle_lower = needle.lower()
    try:
        descendants = spec.descendants()
    except Exception:  # noqa: BLE001
        return None
    for node in descendants:
        try:
            if _safe_control_type(node) != "Button":
                continue
            name = _safe_text(node)
            if not name:
                continue
            name_lower = name.lower()
            ok = (name_lower == needle_lower) if exact else (needle_lower in name_lower)
            if not ok:
                continue
            if not _safe_is_enabled(node):
                continue
            return node
        except Exception:  # noqa: BLE001
            continue
    return None


def click_dialog_button(
    source: object,
    button_name: str,
    *,
    exact: bool = False,
    user_text: str = "",
) -> DialogActionResult:
    """Click a dialog button by name (case-insensitive substring by default).

    Goes through the safety validator first: Cap-3 verb-click rules
    and the explicit-intent matcher gate clicks on action-bearing
    labels (``"Submit"`` / ``"Pay"`` / ``"Send Money"`` / etc.).

    Args:
        source: :class:`DialogInfo`, :class:`WindowInfo`, or raw hwnd.
        button_name: button label to match. Empty string is refused.
        exact: when True, require exact name equality (case-insensitive).
            When False (default), case-insensitive substring match.
        user_text: forwarded to the safety validator so the
            explicit-intent matcher can verify the user actually asked
            for the action.
    """
    # Anticheat-safe mode: hard-blocked while the user is in game.
    from ultron.safety.anticheat import guard as _anticheat_guard
    _anticheat_guard('dialog_click')
    name = (button_name or "").strip()
    if not name:
        return DialogActionResult(
            success=False, action="click_button", target=button_name,
            error="empty button name",
        )

    hwnd = _coerce_hwnd(source)
    spec = _connect_to_window(hwnd)
    if spec is None:
        return DialogActionResult(
            success=False, action="click_button", target=name,
            error="couldn't connect to dialog",
        )

    try:
        dialog_title = _safe_text(spec)
    except Exception:  # noqa: BLE001
        dialog_title = ""

    verdict = _validate_dialog_action(
        action="click_button",
        dialog_title=dialog_title,
        target=name,
        user_text=user_text,
    )
    if not verdict.is_allowed:
        return DialogActionResult(
            success=False, action="click_button",
            dialog_title=dialog_title, target=name,
            error=f"safety: {verdict.reason}",
        )

    node = _find_button_by_name(spec, name, exact=exact)
    if node is None:
        return DialogActionResult(
            success=False, action="click_button",
            dialog_title=dialog_title, target=name,
            error=f"no enabled button matching '{name}'",
        )

    try:
        node.click_input()
    except Exception as exc:  # noqa: BLE001
        return DialogActionResult(
            success=False, action="click_button",
            dialog_title=dialog_title, target=name,
            method="click", error=f"click failed: {exc}",
        )

    return DialogActionResult(
        success=True, action="click_button",
        dialog_title=dialog_title, target=name, method="click",
    )


def _enabled_text_fields(spec) -> list:
    """Return the enabled Edit / ComboBox descendants in tree order."""
    out = []
    try:
        descendants = spec.descendants()
    except Exception:  # noqa: BLE001
        return out
    for node in descendants:
        try:
            ctype = _safe_control_type(node)
            if ctype not in ("Edit", "ComboBox", "Document"):
                continue
            if not _safe_is_enabled(node):
                continue
            out.append(node)
        except Exception:  # noqa: BLE001
            continue
    return out


def type_into_dialog_field(
    source: object,
    text: str,
    *,
    field_index: int = 0,
    user_text: str = "",
) -> DialogActionResult:
    """Type text into the indexed Edit / ComboBox field of a dialog.

    Args:
        source: :class:`DialogInfo`, :class:`WindowInfo`, or raw hwnd.
        text: literal string to type. Cap-3 sees this in the validator
            arguments as ``text`` so credential-pattern detectors can
            refuse credential-shaped writes.
        field_index: 0-based index into the enabled-field list. Use 0
            for the typical "single Edit field" dialog (save-as,
            open-dialog filename, search prompts).
        user_text: forwarded to the safety validator.
    """
    # Anticheat-safe mode: hard-blocked while the user is in game.
    from ultron.safety.anticheat import guard as _anticheat_guard
    _anticheat_guard('dialog_type')
    if text is None:
        return DialogActionResult(
            success=False, action="type_field",
            error="text is None",
        )
    if field_index < 0:
        return DialogActionResult(
            success=False, action="type_field",
            error=f"negative field_index: {field_index}",
        )

    hwnd = _coerce_hwnd(source)
    spec = _connect_to_window(hwnd)
    if spec is None:
        return DialogActionResult(
            success=False, action="type_field",
            error="couldn't connect to dialog",
        )

    try:
        dialog_title = _safe_text(spec)
    except Exception:  # noqa: BLE001
        dialog_title = ""

    verdict = _validate_dialog_action(
        action="type_field",
        dialog_title=dialog_title,
        target=f"field[{field_index}]",
        text=text,
        user_text=user_text,
    )
    if not verdict.is_allowed:
        return DialogActionResult(
            success=False, action="type_field",
            dialog_title=dialog_title,
            target=f"field[{field_index}]",
            error=f"safety: {verdict.reason}",
        )

    fields = _enabled_text_fields(spec)
    if not fields:
        return DialogActionResult(
            success=False, action="type_field",
            dialog_title=dialog_title,
            target=f"field[{field_index}]",
            error="no enabled text fields found",
        )
    if field_index >= len(fields):
        return DialogActionResult(
            success=False, action="type_field",
            dialog_title=dialog_title,
            target=f"field[{field_index}]",
            error=(
                f"field_index {field_index} out of range "
                f"(only {len(fields)} enabled field(s))"
            ),
        )

    target = fields[field_index]
    try:
        target.set_focus()
    except Exception:  # noqa: BLE001
        # set_focus can fail on some Edit wrappers without aborting the type.
        pass

    method = ""
    try:
        if hasattr(target, "set_text"):
            target.set_text(text)
            method = "set_text"
        else:
            target.type_keys(text, with_spaces=True)
            method = "type_keys"
    except Exception as exc:  # noqa: BLE001
        return DialogActionResult(
            success=False, action="type_field",
            dialog_title=dialog_title,
            target=f"field[{field_index}]",
            error=f"type failed: {exc}",
        )

    return DialogActionResult(
        success=True, action="type_field",
        dialog_title=dialog_title,
        target=f"field[{field_index}]",
        method=method,
    )


def dismiss_dialog(
    source: object,
    *,
    user_text: str = "",
    preferred_buttons: Optional[tuple[str, ...]] = None,
) -> DialogActionResult:
    """Auto-dismiss a dialog by trying common close buttons.

    Iterates :data:`DISMISS_BUTTONS` (or ``preferred_buttons`` when
    supplied) in order, clicking the first one that exists. Falls back
    to sending ``{ESC}`` via ``type_keys`` on the dialog root when no
    candidate button is found.

    Each candidate click goes through :func:`click_dialog_button`'s
    full safety gate; the fallback ESC press goes through the
    validator with ``action="dismiss_escape"``.

    Args:
        source: :class:`DialogInfo`, :class:`WindowInfo`, or raw hwnd.
        user_text: forwarded to the safety validator. Important: the
            explicit-intent matcher will block clicks on action-verb
            buttons (Submit / Pay / etc.) when ``user_text`` doesn't
            match -- but most dismiss-button names (OK / Cancel /
            Close) are not on the verb-click block list.
        preferred_buttons: override the default :data:`DISMISS_BUTTONS`
            order. Useful when the caller knows the dialog has an
            "Apply" or "Skip" button that should be preferred.
    """
    # Anticheat-safe mode: hard-blocked while the user is in game.
    from ultron.safety.anticheat import guard as _anticheat_guard
    _anticheat_guard('dialog_click')
    hwnd = _coerce_hwnd(source)
    spec = _connect_to_window(hwnd)
    if spec is None:
        return DialogActionResult(
            success=False, action="dismiss",
            error="couldn't connect to dialog",
        )

    try:
        dialog_title = _safe_text(spec)
    except Exception:  # noqa: BLE001
        dialog_title = ""

    candidates = preferred_buttons if preferred_buttons else DISMISS_BUTTONS
    for candidate in candidates:
        node = _find_button_by_name(spec, candidate, exact=True)
        if node is None:
            continue
        # Re-run safety gate for the SPECIFIC candidate (matters when
        # the dialog title mentions credentials / pay etc.).
        verdict = _validate_dialog_action(
            action="dismiss",
            dialog_title=dialog_title,
            target=candidate,
            user_text=user_text,
        )
        if not verdict.is_allowed:
            continue
        try:
            node.click_input()
            return DialogActionResult(
                success=True, action="dismiss",
                dialog_title=dialog_title, target=candidate,
                method="click",
            )
        except Exception:  # noqa: BLE001
            continue

    # Fallback: ESC.
    verdict = _validate_dialog_action(
        action="dismiss_escape",
        dialog_title=dialog_title,
        target="{ESC}",
        user_text=user_text,
    )
    if not verdict.is_allowed:
        return DialogActionResult(
            success=False, action="dismiss",
            dialog_title=dialog_title,
            error=f"safety: {verdict.reason}",
        )
    try:
        spec.type_keys("{ESC}")
        return DialogActionResult(
            success=True, action="dismiss",
            dialog_title=dialog_title, target="{ESC}",
            method="escape",
        )
    except Exception as exc:  # noqa: BLE001
        return DialogActionResult(
            success=False, action="dismiss",
            dialog_title=dialog_title,
            error=f"escape failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Wait
# ---------------------------------------------------------------------------


def wait_for_dialog(
    *,
    partial_title: Optional[str] = None,
    timeout_s: float = DEFAULT_WAIT_TIMEOUT_S,
    interval_s: float = DEFAULT_WAIT_INTERVAL_S,
    exclude_cloaked: bool = True,
    sleep_fn: Optional[object] = None,
    clock_fn: Optional[object] = None,
) -> Optional[DialogInfo]:
    """Poll until a dialog matching ``partial_title`` appears.

    Args:
        partial_title: case-insensitive substring filter. When None,
            any detected dialog satisfies the poll.
        timeout_s: wall-clock timeout in seconds.
        interval_s: poll interval in seconds.
        exclude_cloaked: forwarded to :func:`find_dialogs`.
        sleep_fn: optional injection for deterministic tests.
        clock_fn: optional injection for deterministic tests.

    Returns:
        The first matching :class:`DialogInfo`, or None on timeout.
    """
    # Anticheat-safe mode: hard-blocked while the user is in game.
    from ultron.safety.anticheat import guard as _anticheat_guard
    _anticheat_guard('dialog_read')
    if timeout_s <= 0:
        return None

    sleeper = sleep_fn if callable(sleep_fn) else time.sleep
    clock = clock_fn if callable(clock_fn) else time.monotonic

    deadline = clock() + float(timeout_s)
    poll_interval = max(0.01, float(interval_s))

    while True:
        try:
            dialogs = find_dialogs(
                partial_title_filter=partial_title,
                exclude_cloaked=exclude_cloaked,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("wait_for_dialog find_dialogs raised: %s", exc)
            dialogs = []
        if dialogs:
            return dialogs[0]

        now = clock()
        if now >= deadline:
            return None
        remaining = deadline - now
        sleeper(min(poll_interval, remaining))


__all__ = [
    "DIALOG_CLASSES",
    "DIALOG_CONTROL_TYPES",
    "DIALOG_TITLE_KEYWORDS",
    "DISMISS_BUTTONS",
    "DEFAULT_WAIT_TIMEOUT_S",
    "DEFAULT_WAIT_INTERVAL_S",
    "DialogInfo",
    "DialogButton",
    "DialogField",
    "DialogCheckbox",
    "DialogContent",
    "DialogActionResult",
    "find_dialogs",
    "read_dialog",
    "click_dialog_button",
    "type_into_dialog_field",
    "dismiss_dialog",
    "wait_for_dialog",
]
