"""UI Automation primitives via ``pywinauto``.

What this delivers without ClawHub's ``windows-control`` plugin:

- :func:`collect_window_text` -- walk a window's UIA tree and return the
  visible text strings. Used by the screen-context layer (Phase 5) to
  inject "what's actually written on screen" into Ultron's LLM context.
- :func:`find_element` -- semantic search by name / automation_id within
  a window. Returns a frozen :class:`UIAElement` snapshot.
- :func:`click_element` -- find + invoke a UIA control. Goes through
  the safety validator (Cap-3 action-verb rule, Cap-4 security-window
  rule).
- :func:`type_text_into_element` -- find + type into a UIA edit control.
- :func:`physical_center_of_element` /
  :func:`physical_rect_of_element` /
  :func:`dpi_aware_click_at_element_center` (catalog 07 T5) --
  DPI-aware coordinate helpers for the UIA-to-pyautogui boundary.
  UIA element bounding rects come from pywinauto's layer (physical
  pixels in DPI-aware processes), while pyautogui expects physical
  pixels too. The helpers route through
  :func:`ultron.desktop.win32_helpers.logical_to_physical` so callers
  receiving logical-pixel coordinates from non-DPI-aware sources
  (older VLMs, browser DOM coordinates) land on the right pixel on
  high-DPI / mixed-DPI displays.

Design notes:

- COM init: pywinauto's UIA backend uses comtypes; the first call from
  a thread initialises COM lazily. We accept that overhead per-call
  rather than maintain our own COM lifecycle.
- Live wrappers from pywinauto are mutable handles tied to the running
  process; we snapshot to :class:`UIAElement` so callers don't keep
  references that may go stale.
- Tree traversal is depth-limited. Deeply-nested apps (browsers, IDEs)
  can have 10k+ elements; the default cap of 200 elements is enough
  for "what's visible at the top" without blowing time.
- Fail-open at every level: a pywinauto exception logs WARN and
  returns ``None`` / empty list. The orchestrator never crashes.

Coordinate-space convention (catalog 07 T5):

- :attr:`UIAElement.rect` carries whatever pywinauto returned. In
  practice this is physical pixels in a per-monitor-DPI-aware Python
  process. Callers crossing the UIA -> pyautogui boundary by raw
  coordinates should use :func:`physical_center_of_element` (or
  :func:`dpi_aware_click_at_element_center`) which behaves as an
  identity on 100%-DPI displays and applies DPI conversion only when
  the caller explicitly opts in via ``assume_logical=True``.
- :class:`ultron.desktop.capture.Screenshot` returns physical pixels
  (mss reads the GDI surface). Crosshairs and bounding boxes drawn
  on those captures must use physical pixels too.
- :mod:`ultron.desktop.click_preview` uses physical pixel
  coordinates throughout.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ultron.desktop.windows import WindowInfo
from ultron.utils.logging import get_logger

logger = get_logger("desktop.uia")

# Cap on how many elements we visit during a single text-collection walk.
# Browsers and IDEs can expose tens of thousands of elements; we want
# "what's on the surface", not an exhaustive tree dump.
_DEFAULT_MAX_ELEMENTS = 200

# Cap on tree depth. Most UI controls relevant to "what's on screen"
# sit within 8 levels of the window root.
_DEFAULT_MAX_DEPTH = 8


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UIAElement:
    """Snapshot of one UI Automation element.

    Frozen because the underlying pywinauto wrapper handles are mutable
    and may go stale; this dataclass captures the metadata at lookup time.

    Attributes:
        name: element's accessible name (label text).
        control_type: UIA control type (``"Button"``, ``"Edit"``,
            ``"TabItem"``, ``"Window"``, etc.).
        automation_id: AutomationId property (set by app developers; not
            always present).
        class_name: Win32 class name (``"Chrome_WidgetWin_1"``,
            ``"Edit"``, etc.).
        rect: (left, top, right, bottom) in virtual-screen coordinates.
        is_enabled: True iff the element is enabled.
        is_visible: True iff the element is on-screen.
    """

    name: str
    control_type: str = ""
    automation_id: str = ""
    class_name: str = ""
    rect: tuple[int, int, int, int] = (0, 0, 0, 0)
    is_enabled: bool = True
    is_visible: bool = True


@dataclass(frozen=True)
class UIAActionResult:
    """Outcome of a UIA click / type action."""

    success: bool
    element_name: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# pywinauto lazy import
# ---------------------------------------------------------------------------


def _import_pywinauto():
    """Lazy import so ``import ultron.desktop`` doesn't pay the COM cost.

    Returns the ``pywinauto`` module, or None when import fails.
    """
    try:
        import pywinauto  # type: ignore[import]
        return pywinauto
    except Exception as e:  # noqa: BLE001
        logger.warning("pywinauto unavailable: %s", e)
        return None


def _resolve_hwnd(window: object) -> int:
    """Accept a :class:`WindowInfo` or raw hwnd; return integer hwnd."""
    if isinstance(window, WindowInfo):
        return int(window.hwnd)
    return int(window)


def _connect_window(hwnd: int):
    """Open a pywinauto connection to a window. Returns the WindowSpecification or None on failure."""
    pwa = _import_pywinauto()
    if pwa is None:
        return None
    try:
        # backend='uia' uses the modern UI Automation API; 'win32' is the
        # legacy fallback. We always use 'uia' here -- it covers
        # WPF/UWP/WinForms/Electron/Chromium, where 'win32' often returns
        # blank trees.
        app = pwa.Application(backend="uia").connect(handle=hwnd, timeout=2)
        return app.window(handle=hwnd)
    except Exception as e:  # noqa: BLE001
        logger.debug("pywinauto connect hwnd=%d failed: %s", hwnd, e)
        return None


# ---------------------------------------------------------------------------
# Text collection (the load-bearing function for screen context)
# ---------------------------------------------------------------------------


def collect_window_text(
    window: object,
    *,
    max_elements: int = _DEFAULT_MAX_ELEMENTS,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    min_length: int = 2,
) -> list[str]:
    """Walk a window's UIA tree and return visible text strings.

    Args:
        window: :class:`WindowInfo` or raw hwnd.
        max_elements: cap on total elements visited (defense against
            10k-element trees in browsers / IDEs).
        max_depth: cap on tree depth.
        min_length: skip strings shorter than this (drops single
            characters and noise).

    Returns:
        Ordered list of unique strings encountered in tree-walk order.
        Empty list when pywinauto unavailable, window can't be
        connected to, or no text was found.
    """
    hwnd = _resolve_hwnd(window)
    spec = _connect_window(hwnd)
    if spec is None:
        return []

    try:
        # Get the top-level element info for tree walk.
        elem = spec.element_info
    except Exception as e:  # noqa: BLE001
        logger.debug("element_info failed hwnd=%d: %s", hwnd, e)
        return []

    seen: set[str] = set()
    out: list[str] = []
    visited = [0]  # mutable counter to share across recursive calls

    def _walk(node, depth: int) -> None:
        if visited[0] >= max_elements:
            return
        visited[0] += 1
        try:
            name = (node.name or "").strip()
        except Exception:  # noqa: BLE001
            name = ""
        if name and len(name) >= min_length and name not in seen:
            seen.add(name)
            out.append(name)
        if depth >= max_depth:
            return
        try:
            children = node.children()
        except Exception:  # noqa: BLE001
            return
        for child in children:
            if visited[0] >= max_elements:
                return
            _walk(child, depth + 1)

    try:
        _walk(elem, 0)
    except Exception as e:  # noqa: BLE001
        logger.warning("UIA walk hwnd=%d failed: %s", hwnd, e)

    return out


# ---------------------------------------------------------------------------
# Element lookup
# ---------------------------------------------------------------------------


def find_element(
    window: object,
    *,
    query: str = "",
    control_type: Optional[str] = None,
    automation_id: Optional[str] = None,
    exact: bool = False,
) -> Optional[UIAElement]:
    """Find a UIA element within a window.

    Matching:

    - ``automation_id`` -- exact match on AutomationId (most reliable
      when app developers expose it).
    - ``query`` -- case-insensitive substring match on element name.
    - ``control_type`` -- when set, restrict to elements of this type
      (``"Button"``, ``"Edit"``, ``"Hyperlink"``, etc.).
    - ``exact`` -- when True, require exact name match (case-insensitive
      still).

    Returns the first matching :class:`UIAElement` snapshot, or None.
    """
    hwnd = _resolve_hwnd(window)
    spec = _connect_window(hwnd)
    if spec is None:
        return None

    try:
        elem = spec.element_info
    except Exception as e:  # noqa: BLE001
        logger.debug("element_info failed hwnd=%d: %s", hwnd, e)
        return None

    q = (query or "").strip().lower()
    auto_id = (automation_id or "").strip()
    ctype = (control_type or "").strip()

    if not q and not auto_id:
        return None

    found: list[UIAElement] = []
    visited = [0]

    def _matches(node) -> bool:
        try:
            name = (node.name or "")
        except Exception:
            name = ""
        try:
            actype = (node.control_type or "")
        except Exception:
            actype = ""
        try:
            aid = (node.automation_id or "")
        except Exception:
            aid = ""

        if auto_id and aid == auto_id:
            return True
        if q:
            name_l = name.lower()
            ok_name = (name_l == q) if exact else (q in name_l)
            if not ok_name:
                return False
            if ctype and actype.lower() != ctype.lower():
                return False
            return True
        return False

    def _walk(node, depth: int) -> None:
        if visited[0] >= _DEFAULT_MAX_ELEMENTS:
            return
        visited[0] += 1
        try:
            if _matches(node):
                snap = _snapshot(node)
                found.append(snap)
                return
        except Exception:  # noqa: BLE001
            pass
        if depth >= _DEFAULT_MAX_DEPTH:
            return
        try:
            children = node.children()
        except Exception:
            return
        for child in children:
            if found:
                return
            _walk(child, depth + 1)

    try:
        _walk(elem, 0)
    except Exception as e:  # noqa: BLE001
        logger.warning("UIA find_element hwnd=%d failed: %s", hwnd, e)

    return found[0] if found else None


def _snapshot(node) -> UIAElement:
    """Capture a UIA element's relevant fields into a frozen UIAElement."""
    def _safe(attr: str, default: str = "") -> str:
        try:
            v = getattr(node, attr, None)
            return str(v) if v else default
        except Exception:
            return default

    rect = (0, 0, 0, 0)
    try:
        r = node.rectangle
        rect = (int(r.left), int(r.top), int(r.right), int(r.bottom))
    except Exception:
        pass

    is_enabled = True
    is_visible = True
    try:
        is_enabled = bool(getattr(node, "enabled", True))
    except Exception:
        pass
    try:
        is_visible = bool(getattr(node, "visible", True))
    except Exception:
        pass

    return UIAElement(
        name=_safe("name"),
        control_type=_safe("control_type"),
        automation_id=_safe("automation_id"),
        class_name=_safe("class_name"),
        rect=rect,
        is_enabled=is_enabled,
        is_visible=is_visible,
    )


# ---------------------------------------------------------------------------
# Action helpers (click / type) with safety gate
# ---------------------------------------------------------------------------


def _validate_uia_action(
    *,
    action: str,
    window_title: str,
    element_query: str,
    text: str = "",
    user_text: str = "",
) -> object:
    """Run the safety validator against a UIA action.

    The Cap-3 action-verb-click rule, Cap-3 OAuth/payment rules, and
    Cap-4 security-window rules check argument values. Pass the window
    title (often contains a URL for browsers) and the element name in
    the arguments so those patterns can match.
    """
    try:
        from ultron.safety.validator import RuleContext, get_validator

        ctx = RuleContext(
            tool_name=f"desktop.uia.{action}",
            arguments={
                "window_title": window_title,
                "element": f"'{element_query}'",
                "text": text,
            },
            capability="desktop_uia",
            user_text=user_text,
        )
        return get_validator().check(ctx)
    except Exception as e:  # noqa: BLE001
        logger.debug("UIA validator skipped: %s", e)
        from ultron.safety.validator import ValidatorVerdict, Verdict
        return ValidatorVerdict(
            verdict=Verdict.ALLOW, reason="validator unavailable",
        )


def click_element(
    window: object,
    query: str,
    *,
    automation_id: Optional[str] = None,
    control_type: Optional[str] = None,
    exact: bool = False,
    user_text: str = "",
) -> UIAActionResult:
    """Find an element and click (invoke) it.

    Goes through the safety validator first: Cap-3 action-verb-click
    matches words like ``"Submit"``, ``"Pay"``, ``"Send Money"`` and
    returns ``NEEDS_EXPLICIT_INTENT`` -- the explicit-intent matcher
    needs the user's recent utterance to contain a matching
    verb+object, otherwise the click is refused.

    Returns :class:`UIAActionResult` -- ``success=False`` and ``error``
    populated on any failure.
    """
    hwnd = _resolve_hwnd(window)
    spec = _connect_window(hwnd)
    if spec is None:
        return UIAActionResult(success=False, error="couldn't connect to window")

    try:
        win_title = spec.window_text() or ""
    except Exception:
        win_title = ""

    verdict = _validate_uia_action(
        action="click",
        window_title=win_title,
        element_query=query,
        user_text=user_text,
    )
    if not verdict.is_allowed:
        return UIAActionResult(
            success=False, element_name=query,
            error=f"safety: {verdict.reason}",
        )

    snap = find_element(
        window, query=query, control_type=control_type,
        automation_id=automation_id, exact=exact,
    )
    if snap is None:
        return UIAActionResult(
            success=False, element_name=query,
            error=f"no element matching '{query}'",
        )
    if not snap.is_enabled:
        return UIAActionResult(
            success=False, element_name=snap.name,
            error=f"element '{snap.name}' is disabled",
        )

    # Re-find the live wrapper to perform the click (the snapshot is
    # data-only).
    try:
        # Try by automation_id first (most precise), then by title.
        if snap.automation_id:
            target = spec.child_window(
                auto_id=snap.automation_id,
                control_type=snap.control_type or None,
            )
        else:
            target = spec.child_window(
                title=snap.name,
                control_type=snap.control_type or None,
            )
        target.click_input()
    except Exception as e:  # noqa: BLE001
        return UIAActionResult(
            success=False, element_name=snap.name,
            error=f"click failed: {e}",
        )

    return UIAActionResult(success=True, element_name=snap.name)


def type_text_into_element(
    window: object,
    query: str,
    text: str,
    *,
    automation_id: Optional[str] = None,
    control_type: Optional[str] = None,
    exact: bool = False,
    clear_first: bool = True,
    user_text: str = "",
) -> UIAActionResult:
    """Find a UIA edit control and type ``text`` into it.

    Args:
        clear_first: when True, the target's existing content is
            cleared (Ctrl+A, Delete) before typing.
    """
    hwnd = _resolve_hwnd(window)
    spec = _connect_window(hwnd)
    if spec is None:
        return UIAActionResult(success=False, error="couldn't connect to window")

    try:
        win_title = spec.window_text() or ""
    except Exception:
        win_title = ""

    verdict = _validate_uia_action(
        action="type",
        window_title=win_title,
        element_query=query,
        text=text,
        user_text=user_text,
    )
    if not verdict.is_allowed:
        return UIAActionResult(
            success=False, element_name=query,
            error=f"safety: {verdict.reason}",
        )

    snap = find_element(
        window, query=query, control_type=control_type or "Edit",
        automation_id=automation_id, exact=exact,
    )
    if snap is None:
        return UIAActionResult(
            success=False, element_name=query,
            error=f"no edit element matching '{query}'",
        )
    if not snap.is_enabled:
        return UIAActionResult(
            success=False, element_name=snap.name,
            error=f"element '{snap.name}' is disabled",
        )

    try:
        if snap.automation_id:
            target = spec.child_window(
                auto_id=snap.automation_id,
                control_type=snap.control_type or "Edit",
            )
        else:
            target = spec.child_window(
                title=snap.name,
                control_type=snap.control_type or "Edit",
            )
        target.set_focus()
        if clear_first:
            target.type_keys("^a{DEL}", with_spaces=True)
        # type_keys escapes special chars when set_text mode isn't usable.
        # For arbitrary user input we want literal characters, so use
        # set_text where possible (supported on EditWrapper).
        if hasattr(target, "set_text"):
            target.set_text(text)
        else:
            # Fallback: type_keys with with_spaces=True. Note: special
            # characters like {, }, ^, +, %, ~, (, ) get interpreted by
            # type_keys; consumers should use set_text for arbitrary text.
            target.type_keys(text, with_spaces=True)
    except Exception as e:  # noqa: BLE001
        return UIAActionResult(
            success=False, element_name=snap.name,
            error=f"type failed: {e}",
        )

    return UIAActionResult(success=True, element_name=snap.name)


# ---------------------------------------------------------------------------
# T5: DPI-aware coordinate helpers (catalog 07)
# ---------------------------------------------------------------------------


def physical_center_of_element(
    element: UIAElement,
    *,
    assume_logical: bool = False,
) -> tuple[int, int]:
    """Return the physical-pixel centre of a :class:`UIAElement`.

    Args:
        element: a :class:`UIAElement` snapshot.
        assume_logical: when True, ``element.rect`` is treated as
            logical (unscaled) pixels and converted to physical via
            :func:`ultron.desktop.win32_helpers.logical_to_physical`.
            When False (default), the rect is treated as already
            physical (pywinauto's normal output in a DPI-aware
            Python process); the function returns the integer
            centre with no DPI lookup.

    On 100%-DPI displays the two branches are identical. The flag
    exists so callers crossing from a known-logical source can
    request conversion without leaking the implementation detail.

    Returns ``(x_physical, y_physical)``. When the element's rect is
    degenerate, the geometric centre is still returned -- callers
    should validate :attr:`UIAElement.is_visible` and the rect
    dimensions before clicking.
    """

    left, top, right, bottom = element.rect
    cx = (left + right) // 2
    cy = (top + bottom) // 2

    if not assume_logical:
        return int(cx), int(cy)

    # Lazy-import so the win32_helpers ctypes setup only happens
    # when DPI conversion is actually requested.
    try:
        from ultron.desktop.win32_helpers import logical_to_physical
    except Exception as exc:  # noqa: BLE001
        logger.debug("logical_to_physical unavailable: %s", exc)
        return int(cx), int(cy)

    return logical_to_physical(int(cx), int(cy))


def physical_rect_of_element(
    element: UIAElement,
    *,
    assume_logical: bool = False,
) -> tuple[int, int, int, int]:
    """Return ``element.rect`` in physical pixels.

    Same DPI conversion semantics as
    :func:`physical_center_of_element`. Returns
    ``(left, top, right, bottom)``.

    Useful when the caller needs the full bounding box (region crop
    on a capture, screen-context VLM prompt). The conversion uses the
    rect's geometric centre as the DPI lookup reference so both
    corners map to the same monitor's scale factor on mixed-DPI
    multi-monitor setups.
    """

    left, top, right, bottom = element.rect
    if not assume_logical:
        return int(left), int(top), int(right), int(bottom)

    try:
        from ultron.desktop.win32_helpers import logical_to_physical
    except Exception as exc:  # noqa: BLE001
        logger.debug("logical_to_physical unavailable: %s", exc)
        return int(left), int(top), int(right), int(bottom)

    ref_x = (int(left) + int(right)) // 2
    ref_y = (int(top) + int(bottom)) // 2
    pl, pt = logical_to_physical(
        int(left), int(top), reference_x=ref_x, reference_y=ref_y,
    )
    pr, pb = logical_to_physical(
        int(right), int(bottom), reference_x=ref_x, reference_y=ref_y,
    )
    return pl, pt, pr, pb


def dpi_aware_click_at_element_center(
    element: UIAElement,
    *,
    controller: Optional[object] = None,
    button: str = "left",
    clicks: int = 1,
    user_text: str = "",
    assume_logical: bool = False,
) -> UIAActionResult:
    """Click an element's centre via :class:`InputController` with
    DPI awareness.

    Designed for callers that already hold a :class:`UIAElement`
    (from :func:`find_element`) and want the coordinate-based
    pyautogui path rather than pywinauto's native ``click_input``.
    The DPI conversion happens at this boundary so pyautogui lands
    on the right pixel on high-DPI displays.

    Args:
        element: target element.
        controller: :class:`InputController` instance. When ``None``,
            the module-level singleton from
            :func:`ultron.desktop.input_control.get_input_controller`
            is used.
        button: ``"left"`` / ``"right"`` / ``"middle"``.
        clicks: number of clicks (2 = double click).
        user_text: forwarded to the controller so the safety
            validator's ``RuleContext.user_text`` reflects the
            originating utterance.
        assume_logical: forwarded to
            :func:`physical_center_of_element`. Default False.

    Returns a :class:`UIAActionResult` describing the outcome. The
    function defends against disabled elements and degenerate
    ``(0, 0, 0, 0)`` rects before touching the controller.
    """

    if not element.is_enabled:
        return UIAActionResult(
            success=False,
            element_name=element.name,
            error=f"element '{element.name}' is disabled",
        )

    if element.rect == (0, 0, 0, 0):
        return UIAActionResult(
            success=False,
            element_name=element.name,
            error=f"element '{element.name}' has no measurable rect",
        )

    if controller is None:
        try:
            from ultron.desktop.input_control import get_input_controller
            controller = get_input_controller()
        except Exception as exc:  # noqa: BLE001
            return UIAActionResult(
                success=False,
                element_name=element.name,
                error=f"input controller unavailable: {exc}",
            )

    cx, cy = physical_center_of_element(element, assume_logical=assume_logical)

    try:
        result = controller.click(
            x=cx,
            y=cy,
            button=button,
            clicks=int(clicks),
            user_text=user_text,
        )
    except Exception as exc:  # noqa: BLE001
        return UIAActionResult(
            success=False,
            element_name=element.name,
            error=f"controller.click raised: {exc}",
        )

    if getattr(result, "success", False):
        return UIAActionResult(success=True, element_name=element.name)
    return UIAActionResult(
        success=False,
        element_name=element.name,
        error=getattr(result, "error", None) or "controller refused click",
    )


# ---------------------------------------------------------------------------
# Catalog 08 T2: structured UI element inventory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UIElementInfo:
    """Snapshot of one interactive UI element with its click coordinates.

    Catalog 08 T2 read-only inventory primitive. Mirrors the upstream
    ``read_ui_elements`` shape: control_type + accessible name + enabled
    state + bounding rect + centre coordinates, plus an optional edit
    value for ``Edit``/``Document`` controls (the upstream plugin admits
    those even when ``name`` is empty, since they often have meaningful
    text content but no label).

    Attributes:
        name: accessible name (label text). May be empty for
            ``Edit``/``Document`` elements whose value field carries
            the content.
        control_type: UIA control type (``"Button"``, ``"Hyperlink"``,
            etc.).
        automation_id: AutomationId property (often empty).
        enabled: True iff the element is enabled and clickable.
        rect: ``(left, top, right, bottom)`` in physical pixels.
        center: ``(x, y)`` integer centre coordinates for click
            targeting (already physical pixels per the catalog 07 T5
            convention).
        value: current edit-field value (only populated for
            ``Edit``/``Document``; empty string otherwise). Truncated
            to ``value_truncate`` chars in :func:`get_ui_element_inventory`.
    """

    name: str
    control_type: str = ""
    automation_id: str = ""
    enabled: bool = True
    rect: tuple[int, int, int, int] = (0, 0, 0, 0)
    center: tuple[int, int] = (0, 0)
    value: str = ""


# Map from UIA control_type string -> inventory bucket key. Mirrors the
# clawhub-windows-control category split with the ultron addition that
# ``Document`` is treated as a text field (Edge / Chrome PDF viewer
# exposes the document body as a Document control with editable focus).
_INVENTORY_BUCKETS: dict[str, str] = {
    "Button": "buttons",
    "Hyperlink": "links",
    "MenuItem": "menu_items",
    "ListItem": "list_items",
    "TabItem": "tabs",
    "CheckBox": "checkboxes",
    "RadioButton": "radio_buttons",
    "Edit": "text_fields",
    "Document": "text_fields",
    "ComboBox": "dropdowns",
}

# Control types that get added to the inventory even when their name is
# empty (text content lives in the value attribute, not the label).
_INVENTORY_NAMELESS_OK = frozenset({"Edit", "Document"})


def get_ui_element_inventory(
    window: object,
    *,
    control_types: Optional[Sequence[str]] = None,
    max_elements: int = _DEFAULT_MAX_ELEMENTS,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    value_truncate: int = 100,
) -> dict[str, list[UIElementInfo]]:
    """Walk a window's UIA tree and bucket interactive controls by type.

    Catalog 08 T2 (GREEN, read-only). Adapted from the upstream
    ``read_ui_elements`` pattern in clawhub-windows-control: per
    descendant capture ``control_type`` + ``window_text()`` + ``enabled``
    + rect + centre, dispatch into ten buckets (buttons / links /
    menu_items / list_items / tabs / checkboxes / radio_buttons /
    text_fields / dropdowns / other). Empty buckets are omitted from
    the returned dict so the caller's iteration stays narrow.

    Args:
        window: :class:`WindowInfo` or raw hwnd.
        control_types: optional case-insensitive allowlist of UIA
            control types. When provided, only elements whose control
            type matches one of these strings are inventoried.
        max_elements: cap on total elements visited (defense against
            10k-element trees in browsers / IDEs). Default matches
            :func:`collect_window_text`.
        max_depth: cap on tree depth.
        value_truncate: cap on edit-field value length in the captured
            snapshot. Set to 0 to omit values entirely.

    Returns:
        Dict keyed by bucket name (``"buttons"``, ``"links"``, ...).
        Values are lists of :class:`UIElementInfo` in tree-walk order.
        Empty buckets are stripped. Empty dict when pywinauto is
        unavailable or the window can't be connected to.

    Fail-open at every layer: per-element exceptions are silently
    skipped; a failed tree walk logs WARN and returns ``{}``.
    """
    hwnd = _resolve_hwnd(window)
    spec = _connect_window(hwnd)
    if spec is None:
        return {}

    try:
        root = spec.element_info
    except Exception as exc:  # noqa: BLE001
        logger.debug("element_info failed hwnd=%d: %s", hwnd, exc)
        return {}

    allow_types: Optional[set[str]] = None
    if control_types is not None:
        allow_types = {str(t).strip().lower() for t in control_types if str(t).strip()}
        if not allow_types:
            allow_types = None

    visited = [0]
    buckets: dict[str, list[UIElementInfo]] = {}

    def _admit(node) -> None:
        try:
            ctype = (node.control_type or "")
        except Exception:  # noqa: BLE001
            return
        ctype_s = str(ctype)
        if allow_types is not None and ctype_s.lower() not in allow_types:
            return

        try:
            raw_name = node.name or ""
        except Exception:  # noqa: BLE001
            raw_name = ""
        name = str(raw_name).strip()
        if not name and ctype_s not in _INVENTORY_NAMELESS_OK:
            return

        try:
            enabled = bool(getattr(node, "enabled", True))
        except Exception:  # noqa: BLE001
            enabled = True

        rect: tuple[int, int, int, int] = (0, 0, 0, 0)
        center: tuple[int, int] = (0, 0)
        try:
            r = node.rectangle
            left = int(r.left)
            top = int(r.top)
            right = int(r.right)
            bottom = int(r.bottom)
            rect = (left, top, right, bottom)
            center = ((left + right) // 2, (top + bottom) // 2)
        except Exception:  # noqa: BLE001
            pass

        value = ""
        if value_truncate > 0 and ctype_s in _INVENTORY_NAMELESS_OK:
            try:
                raw_value = getattr(node, "value", None)
                if raw_value is not None:
                    value = str(raw_value)[:value_truncate]
            except Exception:  # noqa: BLE001
                value = ""

        try:
            auto_id = node.automation_id or ""
        except Exception:  # noqa: BLE001
            auto_id = ""

        info = UIElementInfo(
            name=name,
            control_type=ctype_s,
            automation_id=str(auto_id),
            enabled=enabled,
            rect=rect,
            center=center,
            value=value,
        )

        bucket = _INVENTORY_BUCKETS.get(ctype_s, "other")
        buckets.setdefault(bucket, []).append(info)

    def _walk(node, depth: int) -> None:
        if visited[0] >= max_elements:
            return
        visited[0] += 1
        try:
            _admit(node)
        except Exception:  # noqa: BLE001
            pass
        if depth >= max_depth:
            return
        try:
            children = node.children()
        except Exception:  # noqa: BLE001
            return
        for child in children:
            if visited[0] >= max_elements:
                return
            _walk(child, depth + 1)

    try:
        _walk(root, 0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("UI inventory walk hwnd=%d failed: %s", hwnd, exc)

    return buckets


# ---------------------------------------------------------------------------
# Catalog 08 T4 (partial): wait-for-text in window
# ---------------------------------------------------------------------------


# Defaults mirror the upstream clawhub-windows-control wait scripts: 30 s
# total timeout, 500 ms poll interval. The constants are module-level
# so callers can introspect / override.
DEFAULT_WAIT_TIMEOUT_S: float = 30.0
DEFAULT_WAIT_INTERVAL_S: float = 0.5


def wait_for_text_in_window(
    text: str,
    partial_window_title: str,
    *,
    timeout_s: float = DEFAULT_WAIT_TIMEOUT_S,
    interval_s: float = DEFAULT_WAIT_INTERVAL_S,
    case_insensitive: bool = True,
    max_elements: int = _DEFAULT_MAX_ELEMENTS,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    sleep_fn: Optional[object] = None,
    clock_fn: Optional[object] = None,
) -> bool:
    """Poll until ``text`` appears in any window matching ``partial_window_title``.

    Catalog 08 T4 (GREEN, read-only). Synchronous UIA-tree polling
    barrier. Each iteration re-resolves the target window via the
    foreground enumerator + walks its UIA descendants via
    :func:`collect_window_text`, checking for substring presence.
    Exits as soon as the text is found or the timeout elapses.

    Args:
        text: substring to search for in the window's UIA tree.
        partial_window_title: case-insensitive substring match against
            window title. Restricts the search scope (matching the
            upstream pattern of mandatory window filter). Empty string
            scans every visible window.
        timeout_s: wall-clock timeout in seconds.
        interval_s: poll interval in seconds.
        case_insensitive: when True (default), substring match is
            case-insensitive.
        max_elements: forwarded to :func:`collect_window_text` per
            poll iteration.
        max_depth: forwarded to :func:`collect_window_text`.
        sleep_fn: optional ``(float) -> None`` injection for tests so
            the polling loop doesn't actually sleep. Defaults to
            :func:`time.sleep`.
        clock_fn: optional ``() -> float`` injection for tests so the
            deadline computation is deterministic. Defaults to
            :func:`time.monotonic`.

    Returns:
        True when text found, False on timeout.

    Fail-open: per-window enumeration exceptions silently skip the
    affected window (the next poll re-tries). Empty ``text`` returns
    True immediately. Non-positive ``timeout_s`` returns False without
    polling.
    """
    needle = (text or "")
    if not needle:
        return True
    if timeout_s <= 0:
        return False

    sleeper = sleep_fn if callable(sleep_fn) else time.sleep
    clock = clock_fn if callable(clock_fn) else time.monotonic

    title_filter = (partial_window_title or "").strip().lower()
    needle_cmp = needle.lower() if case_insensitive else needle

    deadline = clock() + float(timeout_s)
    poll_interval = max(0.01, float(interval_s))

    # Lazy import so a test that monkeypatches enumerate_windows in this
    # module picks up the test double.
    from ultron.desktop.windows import enumerate_windows

    while True:
        try:
            windows = enumerate_windows()
        except Exception as exc:  # noqa: BLE001
            logger.debug("wait_for_text enumerate failed: %s", exc)
            windows = []

        for win in windows:
            try:
                if title_filter and title_filter not in (win.title or "").lower():
                    continue
            except Exception:  # noqa: BLE001
                continue
            try:
                names = collect_window_text(
                    win, max_elements=max_elements, max_depth=max_depth,
                )
            except Exception:  # noqa: BLE001
                continue
            for name in names:
                haystack = name.lower() if case_insensitive else name
                if needle_cmp in haystack:
                    return True

        now = clock()
        if now >= deadline:
            return False
        remaining = deadline - now
        sleeper(min(poll_interval, remaining))


__all__ = [
    "UIAElement",
    "UIAActionResult",
    "UIElementInfo",
    "DEFAULT_WAIT_TIMEOUT_S",
    "DEFAULT_WAIT_INTERVAL_S",
    "collect_window_text",
    "find_element",
    "click_element",
    "type_text_into_element",
    "physical_center_of_element",
    "physical_rect_of_element",
    "dpi_aware_click_at_element_center",
    "get_ui_element_inventory",
    "wait_for_text_in_window",
]
