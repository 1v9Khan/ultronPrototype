"""Voice-callable handlers for desktop automation intents.

This module bridges :class:`ultron.openclaw_routing.intents.RoutingIntent`
(for ``APP_LAUNCH`` and ``SCREEN_CONTEXT_QUERY``) into the native
desktop primitives:

- :func:`handle_app_launch` -> :class:`ultron.desktop.launcher.AppLauncher`
- :func:`handle_screen_context_query` -> :func:`ultron.desktop.screen_context.build_screen_context`

Returns plain dataclasses (not the routing layer's ``VoiceResponse``)
so the orchestrator can adapt to either path. The orchestrator-side
wiring -- which intercepts the new intent kinds, calls these handlers,
and either speaks the voice_message or injects screen_context for the
next LLM turn -- is the next integration step (Phase 8b).

Both handlers are sync. Launch + capture each complete in tens to
hundreds of ms; the VLM call within screen-context handling is the
only multi-second path and is gated by ``intent.include_vlm``.

Fail-open at every layer: validator block, launcher error, capture
failure, VLM exception all degrade to structured results with
``success=False`` and a user-readable ``voice_message``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ultron.utils.logging import get_logger

logger = get_logger("desktop.voice")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppLaunchVoiceResult:
    """Result of handling an APP_LAUNCH intent.

    Attributes:
        success: True iff the app spawned + (when monitor target set) the
            window was placed.
        voice_message: short in-character line for the user. Always
            populated.
        app_name: registry name of the launched app.
        monitor_index: monitor the window landed on, when applicable.
        hwnd: window handle on successful placement.
    """

    success: bool
    voice_message: str
    app_name: str = ""
    monitor_index: Optional[int] = None
    hwnd: Optional[int] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class ScreenContextVoiceResult:
    """Result of handling a SCREEN_CONTEXT_QUERY intent.

    Attributes:
        success: True iff a snapshot was assembled (the LLM injection
            text is non-empty).
        injection_text: ready-to-prepend context block for the next
            LLM turn (output of
            :meth:`ScreenContextSnapshot.render_for_llm`).
        elapsed_ms: total wall-clock of the snapshot build.
        used_vlm: True iff the VLM was invoked.
        error: failure reason when ``success=False``.
    """

    success: bool
    injection_text: str = ""
    elapsed_ms: float = 0.0
    used_vlm: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Monitor resolution
# ---------------------------------------------------------------------------


def _resolve_monitor(monitor_index: Optional[int], monitor_query: str):
    """Resolve a monitor target from an intent.

    Returns the :class:`Monitor` instance, or ``None`` when the
    desktop module can't be imported (caller treats as "no monitor
    placement"). When the caller provides no target (``monitor_index
    is None`` AND ``monitor_query`` is empty), defaults to the user's
    main monitor -- physical center on multi-display setups -- per
    user direction 2026-05-14 ("if no monitor is selected do the main
    monitor"). Fails open: any pywin32 / enum failure returns ``None``.
    """
    try:
        from ultron.desktop.monitors import enumerate_monitors, find_monitor
    except Exception as e:  # noqa: BLE001
        logger.debug("monitor resolve import failed: %s", e)
        return None

    if monitor_index is not None:
        mons = enumerate_monitors()
        if not (0 <= monitor_index < len(mons)):
            return None
        return mons[monitor_index]

    if monitor_query:
        return find_monitor(monitor_query)

    # 2026-05-14 default-to-main: when the utterance gives no monitor
    # cue, place on the user's main (physical center) monitor instead
    # of letting the launched app pick wherever it was last positioned.
    return find_monitor("main")


# ---------------------------------------------------------------------------
# APP_LAUNCH handler
# ---------------------------------------------------------------------------


def handle_app_launch(intent) -> AppLaunchVoiceResult:
    """Dispatch an :class:`AppLaunchIntent` to the native launcher.

    Args:
        intent: an :class:`ultron.openclaw_routing.intents.AppLaunchIntent`
            from the routing classifier.
    """
    app_name = (getattr(intent, "app_name", "") or "").strip()
    if not app_name:
        return AppLaunchVoiceResult(
            success=False,
            voice_message="I didn't catch which app you wanted opened.",
            error="empty app_name",
        )

    url = getattr(intent, "url", None)
    monitor_index = getattr(intent, "monitor_index", None)
    monitor_query = getattr(intent, "monitor_query", "") or ""
    fullscreen = bool(getattr(intent, "fullscreen", False))
    maximize = bool(getattr(intent, "maximize", False))
    user_text = getattr(intent, "raw_text", "") or ""

    monitor = _resolve_monitor(monitor_index, monitor_query)

    try:
        from ultron.desktop.launcher import get_app_launcher
    except Exception as e:  # noqa: BLE001
        return AppLaunchVoiceResult(
            success=False,
            voice_message="The desktop launcher isn't available right now.",
            error=f"launcher import failed: {e}",
        )

    launcher = get_app_launcher()

    # Chrome + URL goes through launch_chrome so we get the default
    # profile + new-window semantics.
    if app_name.lower() == "chrome" and url:
        result = launcher.launch_chrome(
            url=url,
            monitor=monitor,
            fullscreen=fullscreen,
            maximize=maximize,
            user_text=user_text,
        )
    else:
        result = launcher.launch_app(
            app_name=app_name,
            monitor=monitor,
            extra_args=None,
            fullscreen=fullscreen,
            maximize=maximize,
            wait_for_window=monitor is not None,
            user_text=user_text,
        )

    if not result.success:
        msg = (
            f"I couldn't open {app_name}."
            + (f" {result.error}" if result.error else "")
        )
        # Record failure for diagnostic purposes (doesn't become a default).
        _record_preference_safe(
            user_phrase=user_text, app_name=result.app_name or app_name,
            monitor_index=result.monitor_index, fullscreen=fullscreen,
            maximize=maximize, url=url, success=False,
        )
        return AppLaunchVoiceResult(
            success=False,
            voice_message=msg,
            app_name=result.app_name or app_name,
            error=result.error,
        )

    # Voice message shape: short, in-character.
    mon_phrase = ""
    if result.monitor_index is not None:
        mon_phrase = f" on monitor {result.monitor_index + 1}"
    elif monitor is not None:
        mon_phrase = f" on monitor {monitor.index + 1}"

    if url:
        msg = f"Opening that{mon_phrase}."
    else:
        msg = f"Opening {result.app_name or app_name}{mon_phrase}."

    # Record the successful preference for next-time learning.
    _record_preference_safe(
        user_phrase=user_text,
        app_name=result.app_name or app_name,
        monitor_index=result.monitor_index,
        fullscreen=fullscreen,
        maximize=maximize,
        url=url,
        success=True,
    )

    return AppLaunchVoiceResult(
        success=True,
        voice_message=msg,
        app_name=result.app_name or app_name,
        monitor_index=result.monitor_index,
        hwnd=result.hwnd,
    )


def _record_preference_safe(
    *,
    user_phrase: str,
    app_name: str,
    monitor_index,
    fullscreen: bool,
    maximize: bool,
    url,
    success: bool,
) -> None:
    """Record a launch preference. Fail-open at every layer."""
    if not user_phrase:
        return
    try:
        from ultron.desktop.preferences import record_launch_preference

        record_launch_preference(
            user_phrase=user_phrase,
            app_name=app_name,
            monitor_index=monitor_index,
            fullscreen=fullscreen,
            maximize=maximize,
            url=url,
            success=success,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("preference write skipped: %s", e)


# ---------------------------------------------------------------------------
# SCREEN_CONTEXT_QUERY handler
# ---------------------------------------------------------------------------


def handle_screen_context_query(intent) -> ScreenContextVoiceResult:
    """Dispatch a :class:`ScreenContextIntent`.

    Builds the snapshot and returns the ``render_for_llm`` injection
    text. The orchestrator prepends this to the user's utterance on
    the next LLM call so Ultron can answer about what's on screen.
    """
    include_vlm = bool(getattr(intent, "include_vlm", True))

    try:
        from ultron.desktop.screen_context import build_screen_context
    except Exception as e:  # noqa: BLE001
        return ScreenContextVoiceResult(
            success=False,
            error=f"screen_context import failed: {e}",
        )

    try:
        snap = build_screen_context(
            capture=include_vlm,            # only capture if VLM will read it
            include_uia=True,
            include_vlm=include_vlm,
        )
    except Exception as e:  # noqa: BLE001
        return ScreenContextVoiceResult(
            success=False,
            error=f"snapshot build failed: {e}",
        )

    injection = snap.render_for_llm()
    used_vlm = snap.vlm_description is not None
    return ScreenContextVoiceResult(
        success=bool(injection),
        injection_text=injection,
        elapsed_ms=snap.elapsed_ms,
        used_vlm=used_vlm,
    )


__all__ = [
    "AppLaunchVoiceResult",
    "ScreenContextVoiceResult",
    "handle_app_launch",
    "handle_screen_context_query",
]
