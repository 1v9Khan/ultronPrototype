"""Cross-window semantic UIA element search + click.

Catalog 08 T3 (YELLOW). The primary "act on a UI element by name"
primitive for LLM-facing automation. Coding tasks and voice commands
say things like "click the Submit button", "activate the File menu",
"toggle the Remember me checkbox" -- those should not require the LLM
to first read coordinates from a screenshot. Semantic click via UIA
is dramatically more reliable: the orchestrator walks the desktop's
accessibility tree by name + control-type filter, picks the best
match (exact over substring, scoped over global), and clicks via the
gated :class:`InputController` so the click-preview VLM gate +
foreground-security check + Cap-3 safety validator + rate limit all
apply uniformly.

Surface:

* :func:`find_elements_by_name` -- enumerate clickable elements
  matching ``name`` across windows. Returns frozen
  :class:`UIElementMatch` records carrying name + control_type +
  enabled state + rect + centre + owning window + is_exact flag.
* :func:`click_element_by_name` -- the headline. Find + click via
  the gated :class:`ultron.desktop.input_control.InputController`.
  Promotes exact matches over substring; promotes window-scoped
  matches over global ones (when ``window_title`` is set).
* :func:`find_text_in_window` -- coordinate-only variant. Returns
  :class:`TextMatch` records WITHOUT clicking, suitable for the
  "look up coords, hand them to the VLM preview, decide what to do"
  workflow.

Gating architecture (catalog 08 T3 YELLOW):

* Cap-2 reads UIA. Read-only enumeration (:func:`find_elements_by_name`,
  :func:`find_text_in_window`) is GREEN.
* Cap-3 input. :func:`click_element_by_name` goes through the
  :class:`InputController` which carries the foreground-security
  refusal (UAC / credential dialog), rate limit, safety-validator
  hook (``tool_name="desktop.element.click"`` with element + window
  title in arguments), and click-preview VLM gate when enabled.
* click_preview is the safety differentiator for this surface. When
  enabled, the VLM sees a crosshair-marked screenshot of the target
  coordinate and either confirms (ALLOW), denies (BLOCK), or
  auto-passes on radius-cached recent clicks. Routing through the
  controller's coordinate click rather than pywinauto's native
  ``click_input()`` is what makes click_preview apply at all.

Fail-open at every layer: pywinauto / enumeration failures fall back
to empty lists; the orchestrator can call any function from a single
``try/except`` wrapping the whole step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from ultron.desktop.windows import WindowInfo, enumerate_windows
from ultron.utils.logging import get_logger

logger = get_logger("desktop.element_click")


# UIA control types we treat as "clickable" for the semantic-name
# match. Matches the upstream clawhub-windows-control set; these are
# standard UIA control type identifiers documented in the public
# Microsoft UI Automation API. Lower-case comparison is performed at
# match time so callers can pass arbitrary casing.
CLICKABLE_TYPES: tuple[str, ...] = (
    "Button",
    "Hyperlink",
    "MenuItem",
    "TabItem",
    "ListItem",
    "CheckBox",
    "RadioButton",
    "TreeItem",
    "DataItem",
)


# Cap on windows visited during a single global walk. Upstream caps
# its ``list_clickable`` at 5 when no title filter is given; we use a
# higher cap because production workloads sometimes have 6-10 windows
# (browser + IDE + slack + spotify + etc.) and the user may genuinely
# want a match in any of them. Set to 0 to disable the cap.
DEFAULT_MAX_GLOBAL_WINDOWS: int = 12

# Cap on UIA descendants visited per window. Browsers can expose 10k+
# elements; the click-target search only needs to look at the visible
# clickable surface, so a tight cap keeps the walk under 100 ms.
DEFAULT_MAX_ELEMENTS_PER_WINDOW: int = 500


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UIElementMatch:
    """Snapshot of one clickable element discovered by semantic search."""

    name: str
    control_type: str
    automation_id: str
    enabled: bool
    rect: tuple[int, int, int, int]
    center: tuple[int, int]
    window: WindowInfo
    is_exact: bool = False


@dataclass(frozen=True)
class TextMatch:
    """Coordinate-only snapshot of a text-bearing element."""

    name: str
    control_type: str
    rect: tuple[int, int, int, int]
    center: tuple[int, int]
    window: WindowInfo


@dataclass(frozen=True)
class ClickResult:
    """Outcome of a :func:`click_element_by_name` call."""

    success: bool
    element_name: str = ""
    window_title: str = ""
    control_type: str = ""
    center: tuple[int, int] = (0, 0)
    method: str = ""  # "controller_click" / "pywinauto_click_input" / ""
    candidates: int = 0  # how many candidates were considered before picking
    is_exact: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal: pywinauto lazy import + per-window walk
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
        logger.debug("element_click connect hwnd=%d failed: %s", hwnd, exc)
        return None


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


def _safe_automation_id(node) -> str:
    try:
        return str(node.element_info.automation_id or "")
    except Exception:  # noqa: BLE001
        return ""


def _safe_enabled(node) -> bool:
    try:
        return bool(node.is_enabled())
    except Exception:  # noqa: BLE001
        return True


def _safe_rect(node) -> tuple[int, int, int, int]:
    try:
        r = node.rectangle()
        return (int(r.left), int(r.top), int(r.right), int(r.bottom))
    except Exception:  # noqa: BLE001
        return (0, 0, 0, 0)


def _center_of(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    return ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2)


def _resolve_allowed_types(control_types: Optional[Sequence[str]]) -> set[str]:
    """Normalise control-type filter to a lower-case set.

    When ``None`` or empty, returns the lower-cased :data:`CLICKABLE_TYPES`
    set so the default behaviour is "any clickable control".
    """

    if not control_types:
        return {t.lower() for t in CLICKABLE_TYPES}
    out: set[str] = set()
    for t in control_types:
        s = str(t).strip().lower()
        if s:
            out.add(s)
    return out or {t.lower() for t in CLICKABLE_TYPES}


def _walk_window_for_clickables(
    window: WindowInfo,
    *,
    name_needle: str,
    name_needle_lower: str,
    allow_types: set[str],
    exact: bool,
    max_elements: int,
    name_must_be_set: bool = True,
) -> list[UIElementMatch]:
    """Walk one window's descendants, return clickable matches."""
    spec = _connect_to_window(int(window.hwnd))
    if spec is None:
        return []

    try:
        descendants = spec.descendants()
    except Exception as exc:  # noqa: BLE001
        logger.debug("descendants() hwnd=%d failed: %s", window.hwnd, exc)
        return []

    matches: list[UIElementMatch] = []
    visited = 0
    for node in descendants:
        if visited >= max_elements:
            break
        visited += 1
        try:
            ctype = _safe_control_type(node)
            if ctype.lower() not in allow_types:
                continue
            name = _safe_text(node)
            if name_must_be_set and not name:
                continue
            name_lower = name.lower()

            is_exact = name_lower == name_needle_lower
            if exact:
                if not is_exact:
                    continue
            else:
                if not is_exact and name_needle_lower not in name_lower:
                    continue

            enabled = _safe_enabled(node)
            if not enabled:
                # Disabled elements are kept but never picked first --
                # the ranker drops them after exact-match promotion.
                # Match the upstream behaviour: collect, mark, sort later.
                pass

            rect = _safe_rect(node)
            matches.append(
                UIElementMatch(
                    name=name,
                    control_type=ctype,
                    automation_id=_safe_automation_id(node),
                    enabled=enabled,
                    rect=rect,
                    center=_center_of(rect),
                    window=window,
                    is_exact=is_exact,
                )
            )
        except Exception:  # noqa: BLE001
            # Per-element fail-open; one bad descendant doesn't abort
            # the whole walk.
            continue

    return matches


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_elements_by_name(
    name: str,
    *,
    window_title: Optional[str] = None,
    control_types: Optional[Sequence[str]] = None,
    exact: bool = False,
    enabled_only: bool = True,
    max_windows: int = DEFAULT_MAX_GLOBAL_WINDOWS,
    max_elements_per_window: int = DEFAULT_MAX_ELEMENTS_PER_WINDOW,
    exclude_cloaked: bool = True,
) -> list[UIElementMatch]:
    """Find clickable elements by name across windows.

    Iterates :func:`enumerate_windows`, optionally filtered by title
    substring; for each window walks UIA descendants whose
    ``control_type`` is in ``control_types`` (default
    :data:`CLICKABLE_TYPES`), collecting matches by name. Returned
    list is ordered with **exact matches first**, then substring matches
    in tree-walk order. Disabled elements are excluded by default.

    Args:
        name: target name to match (case-insensitive). Empty string
            returns an empty list (no all-match semantics).
        window_title: case-insensitive substring; when set, only the
            matching window's tree is walked AND scoped matches rank
            before any unscoped match would have. When None, the
            walk visits every window up to ``max_windows``.
        control_types: optional allowlist of UIA control types. When
            None / empty, the full :data:`CLICKABLE_TYPES` set applies.
        exact: when True, require exact name match (case-insensitive
            still). When False (default), substring match.
        enabled_only: when True (default), disabled candidates are
            filtered out of the result. Set False to receive them
            and let the caller decide (e.g. to surface "the button
            is disabled" diagnostics).
        max_windows: cap on windows visited when no ``window_title``
            filter is supplied. 0 disables the cap.
        max_elements_per_window: cap on UIA descendants visited per
            window.
        exclude_cloaked: forwarded to :func:`enumerate_windows`.

    Returns:
        List of :class:`UIElementMatch` in ranked order. Exact matches
        first (preserving tree-walk order within exact); substring
        matches second (also preserving tree-walk order).
    """
    # Anticheat-safe mode: hard-blocked while the user is in game.
    from ultron.safety.anticheat import guard as _anticheat_guard
    _anticheat_guard('element_read')
    needle = (name or "").strip()
    if not needle:
        return []

    try:
        all_windows = enumerate_windows(exclude_cloaked=exclude_cloaked)
    except Exception as exc:  # noqa: BLE001
        logger.debug("find_elements enumerate_windows failed: %s", exc)
        return []

    title_filter = (window_title or "").strip().lower()
    candidate_windows: list[WindowInfo] = []
    for win in all_windows:
        if title_filter and title_filter not in (win.title or "").lower():
            continue
        candidate_windows.append(win)

    if not title_filter and max_windows > 0:
        candidate_windows = candidate_windows[:max_windows]

    allow_types = _resolve_allowed_types(control_types)
    needle_lower = needle.lower()

    matches: list[UIElementMatch] = []
    for win in candidate_windows:
        matches.extend(
            _walk_window_for_clickables(
                win,
                name_needle=needle,
                name_needle_lower=needle_lower,
                allow_types=allow_types,
                exact=exact,
                max_elements=max_elements_per_window,
            )
        )

    if enabled_only:
        matches = [m for m in matches if m.enabled]

    # Exact-wins-over-substring ranking. Stable sort preserves the
    # tree-walk order within each tier.
    matches.sort(key=lambda m: 0 if m.is_exact else 1)
    return matches


def click_element_by_name(
    name: str,
    *,
    window_title: Optional[str] = None,
    control_type: Optional[str] = None,
    exact: bool = False,
    user_text: str = "",
    controller: Optional[object] = None,
    max_windows: int = DEFAULT_MAX_GLOBAL_WINDOWS,
    max_elements_per_window: int = DEFAULT_MAX_ELEMENTS_PER_WINDOW,
    exclude_cloaked: bool = True,
) -> ClickResult:
    """Find a UI element by name and click it via the safety-gated controller.

    This is the primary "act on a button / link / menu by name"
    primitive. Routes the click through
    :class:`ultron.desktop.input_control.InputController` so the
    click-preview VLM gate + foreground security check + rate limit +
    safety validator + Cap-3 explicit-intent matcher all apply.

    Args:
        name: target name to match.
        window_title: optional case-insensitive substring to restrict
            the search to one window.
        control_type: optional single control-type filter. When None,
            the default :data:`CLICKABLE_TYPES` set applies.
        exact: when True, require exact name match.
        user_text: forwarded to the controller's safety validator so
            the explicit-intent matcher can verify the user actually
            asked for the action.
        controller: :class:`InputController` instance. When None,
            :func:`ultron.desktop.input_control.get_input_controller`
            resolves the module singleton.
        max_windows: cap on windows visited.
        max_elements_per_window: cap on descendants visited per window.
        exclude_cloaked: forwarded to :func:`enumerate_windows`.

    Returns:
        :class:`ClickResult` with success / error + the resolved
        center coordinate + which method actually fired the click.
    """
    # Anticheat-safe mode: hard-blocked while the user is in game.
    from ultron.safety.anticheat import guard as _anticheat_guard
    _anticheat_guard('element_click')
    types_filter: Optional[tuple[str, ...]] = None
    if control_type is not None:
        ctrl_str = str(control_type).strip()
        if ctrl_str:
            types_filter = (ctrl_str,)

    candidates = find_elements_by_name(
        name,
        window_title=window_title,
        control_types=types_filter,
        exact=exact,
        enabled_only=True,
        max_windows=max_windows,
        max_elements_per_window=max_elements_per_window,
        exclude_cloaked=exclude_cloaked,
    )

    if not candidates:
        return ClickResult(
            success=False,
            element_name=name,
            error=f"no enabled element matching {name!r}",
            candidates=0,
        )

    target = candidates[0]
    if controller is None:
        try:
            from ultron.desktop.input_control import get_input_controller
            controller = get_input_controller()
        except Exception as exc:  # noqa: BLE001
            return ClickResult(
                success=False,
                element_name=target.name,
                window_title=target.window.title,
                control_type=target.control_type,
                center=target.center,
                candidates=len(candidates),
                is_exact=target.is_exact,
                error=f"input controller unavailable: {exc}",
            )

    try:
        # Click via coordinate so the controller's gate stack
        # (foreground security + rate limit + safety validator +
        # click-preview VLM) runs. tool_name in the validator context
        # is desktop.input.click; we thread user_text through so
        # is_explicit_intent can match.
        cx, cy = target.center
        result = controller.click(
            x=int(cx),
            y=int(cy),
            user_text=user_text,
        )
    except Exception as exc:  # noqa: BLE001
        return ClickResult(
            success=False,
            element_name=target.name,
            window_title=target.window.title,
            control_type=target.control_type,
            center=target.center,
            candidates=len(candidates),
            is_exact=target.is_exact,
            error=f"controller.click raised: {exc}",
        )

    if getattr(result, "success", False):
        return ClickResult(
            success=True,
            element_name=target.name,
            window_title=target.window.title,
            control_type=target.control_type,
            center=target.center,
            method="controller_click",
            candidates=len(candidates),
            is_exact=target.is_exact,
        )
    return ClickResult(
        success=False,
        element_name=target.name,
        window_title=target.window.title,
        control_type=target.control_type,
        center=target.center,
        method="controller_click",
        candidates=len(candidates),
        is_exact=target.is_exact,
        error=getattr(result, "error", None) or "controller refused click",
    )


def find_text_in_window(
    text: str,
    *,
    window_title: Optional[str] = None,
    case_insensitive: bool = True,
    max_windows: int = DEFAULT_MAX_GLOBAL_WINDOWS,
    max_elements_per_window: int = DEFAULT_MAX_ELEMENTS_PER_WINDOW,
    exclude_cloaked: bool = True,
) -> list[TextMatch]:
    """Find text-bearing elements by substring across windows.

    Coordinate-only variant of :func:`click_element_by_name`. Returns
    every element whose ``window_text()`` contains the search string;
    no click is performed and no control-type filter is applied. The
    primary consumer is the "look up coords, hand them to the VLM
    preview, decide what to do next" workflow.

    Args:
        text: substring to search for. Empty string returns [].
        window_title: optional case-insensitive title filter.
        case_insensitive: when True (default), substring match is
            case-insensitive.
        max_windows: cap on windows visited.
        max_elements_per_window: cap on descendants visited per window.
        exclude_cloaked: forwarded to :func:`enumerate_windows`.

    Returns:
        List of :class:`TextMatch` in tree-walk order. Empty list on
        no match, pywinauto unavailable, or window-walk failure.
    """
    # Anticheat-safe mode: hard-blocked while the user is in game.
    from ultron.safety.anticheat import guard as _anticheat_guard
    _anticheat_guard('element_read')
    needle = (text or "")
    if not needle:
        return []

    try:
        all_windows = enumerate_windows(exclude_cloaked=exclude_cloaked)
    except Exception as exc:  # noqa: BLE001
        logger.debug("find_text enumerate_windows failed: %s", exc)
        return []

    title_filter = (window_title or "").strip().lower()
    candidate_windows: list[WindowInfo] = []
    for win in all_windows:
        if title_filter and title_filter not in (win.title or "").lower():
            continue
        candidate_windows.append(win)

    if not title_filter and max_windows > 0:
        candidate_windows = candidate_windows[:max_windows]

    needle_cmp = needle.lower() if case_insensitive else needle
    matches: list[TextMatch] = []

    for win in candidate_windows:
        spec = _connect_to_window(int(win.hwnd))
        if spec is None:
            continue
        try:
            descendants = spec.descendants()
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "find_text descendants hwnd=%d failed: %s", win.hwnd, exc,
            )
            continue
        visited = 0
        for node in descendants:
            if visited >= max_elements_per_window:
                break
            visited += 1
            try:
                name = _safe_text(node)
                if not name:
                    continue
                haystack = name.lower() if case_insensitive else name
                if needle_cmp not in haystack:
                    continue
                rect = _safe_rect(node)
                matches.append(
                    TextMatch(
                        name=name,
                        control_type=_safe_control_type(node),
                        rect=rect,
                        center=_center_of(rect),
                        window=win,
                    )
                )
            except Exception:  # noqa: BLE001
                continue

    return matches


__all__ = [
    "CLICKABLE_TYPES",
    "DEFAULT_MAX_GLOBAL_WINDOWS",
    "DEFAULT_MAX_ELEMENTS_PER_WINDOW",
    "UIElementMatch",
    "TextMatch",
    "ClickResult",
    "find_elements_by_name",
    "click_element_by_name",
    "find_text_in_window",
]
