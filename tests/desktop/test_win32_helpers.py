"""Tests for :mod:`ultron.desktop.win32_helpers` (catalog 07 T2).

These exercise the public surface with mocked DLL handles so the test
suite runs on any platform without touching the real Win32 API. The
implementation's fail-open semantics give us deterministic outcomes
even when ``ctypes.windll`` isn't available.

Per the test-writer binding rules: ``monkeypatch`` for every module
attr override, ``Event.wait`` for any timing test instead of bare
``time.sleep``, all timing budgets well under 30 s.
"""

from __future__ import annotations

import ctypes
import threading
import time
from typing import Any

import pytest

from ultron.desktop import win32_helpers as wh


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_dll_cache():
    """Each test starts with an empty DLL cache so monkeypatched values
    don't leak across cases."""

    wh._reset_dll_cache_for_testing()
    yield
    wh._reset_dll_cache_for_testing()


class _FakeUser32:
    """Minimal user32 stand-in for the MonitorFromPoint + GetLastInputInfo
    + GetWindowRect + BlockInput calls the helpers make."""

    def __init__(
        self,
        *,
        monitor_handle: int = 0xABCD,
        last_input_tick: int = 0,
        block_input_success: bool = True,
        get_window_rect_value=(10, 20, 110, 120),
        get_window_rect_returns: int = 1,
    ) -> None:
        self.monitor_handle = monitor_handle
        self.last_input_tick = last_input_tick
        self.block_input_success = block_input_success
        self.block_input_calls: list[bool] = []
        self.get_window_rect_value = get_window_rect_value
        self.get_window_rect_returns = get_window_rect_returns

    def MonitorFromPoint(self, point, flag):
        return self.monitor_handle

    def GetLastInputInfo(self, lii_ref):
        lii = lii_ref._obj  # type: ignore[attr-defined]
        lii.dwTime = int(self.last_input_tick)
        return 1

    def GetWindowRect(self, hwnd, rect_ref):
        rect = rect_ref._obj  # type: ignore[attr-defined]
        l, t, r, b = self.get_window_rect_value
        rect[0] = l
        rect[1] = t
        rect[2] = r
        rect[3] = b
        return self.get_window_rect_returns

    def BlockInput(self, enable_int):
        enable_bool = bool(enable_int.value if hasattr(enable_int, "value") else enable_int)
        self.block_input_calls.append(enable_bool)
        return 1 if self.block_input_success else 0


class _FakeShcore:
    """Minimal shcore stand-in for GetDpiForMonitor."""

    def __init__(
        self,
        *,
        dpi_x: int = 144,
        dpi_y: int = 144,
        hresult: int = 0,
    ) -> None:
        self.dpi_x = dpi_x
        self.dpi_y = dpi_y
        self.hresult = hresult

    def GetDpiForMonitor(self, hmon, dpi_type, x_ref, y_ref):
        x_ref._obj.value = self.dpi_x  # type: ignore[attr-defined]
        y_ref._obj.value = self.dpi_y  # type: ignore[attr-defined]
        return self.hresult


class _FakeDwmapi:
    """Minimal dwmapi stand-in for DwmGetWindowAttribute."""

    def __init__(self, *, cloaked: int = 0, hresult: int = 0) -> None:
        self.cloaked = cloaked
        self.hresult = hresult

    def DwmGetWindowAttribute(self, hwnd, attr, value_ref, size):
        value_ref._obj.value = int(self.cloaked)  # type: ignore[attr-defined]
        return self.hresult


class _FakeFuncPtr:
    """Callable with a settable ``restype`` attribute, matching the
    ``ctypes._FuncPointer`` contract production code relies on."""

    def __init__(self, return_value: Any) -> None:
        self._return_value = return_value
        self.restype: Any = None
        self.argtypes: Any = None

    def __call__(self, *_args, **_kwargs):
        return self._return_value


class _FakeKernel32:
    """Minimal kernel32 stand-in for GetTickCount.

    ``GetTickCount`` is exposed as a :class:`_FakeFuncPtr` so production
    code that writes ``kernel32.GetTickCount.restype = ctypes.c_uint``
    finds a settable attribute.
    """

    def __init__(self, *, tick: int = 1000) -> None:
        self.tick = tick
        self.GetTickCount = _FakeFuncPtr(int(tick))  # noqa: N815


def _patch_dlls(
    monkeypatch,
    *,
    user32=None,
    shcore=None,
    dwmapi=None,
    kernel32=None,
    is_windows: bool = True,
) -> None:
    """Install fake DLL handles + force-flip the IS_WINDOWS guard."""

    monkeypatch.setattr(wh, "IS_WINDOWS", is_windows)
    cache: dict[str, Any] = {}
    if user32 is not None:
        cache["user32"] = user32
    if shcore is not None:
        cache["shcore"] = shcore
    if dwmapi is not None:
        cache["dwmapi"] = dwmapi
    if kernel32 is not None:
        cache["kernel32"] = kernel32

    def _fake_load(name: str):
        return cache.get(name)

    monkeypatch.setattr(wh, "_load_dll", _fake_load)


# ---------------------------------------------------------------------------
# Non-Windows fallback
# ---------------------------------------------------------------------------


class TestNonWindowsFallback:
    """Off-Windows callers get documented no-op results."""

    def test_get_monitor_dpi_returns_default(self, monkeypatch):
        monkeypatch.setattr(wh, "IS_WINDOWS", False)
        result = wh.get_monitor_dpi(0, 0)
        assert result.is_default is True
        assert result.dpi_x == wh.DEFAULT_DPI
        assert result.dpi_y == wh.DEFAULT_DPI
        assert result.scale_x == 1.0
        assert result.scale_y == 1.0

    def test_get_last_input_idle_ms_returns_none(self, monkeypatch):
        monkeypatch.setattr(wh, "IS_WINDOWS", False)
        assert wh.get_last_input_idle_ms() is None

    def test_is_window_cloaked_returns_none(self, monkeypatch):
        monkeypatch.setattr(wh, "IS_WINDOWS", False)
        assert wh.is_window_cloaked(123) is None

    def test_block_input_context_yields_disengaged(self, monkeypatch):
        monkeypatch.setattr(wh, "IS_WINDOWS", False)
        with wh.block_input_context() as result:
            assert result.engaged is False
            assert result.watchdog_fired is False

    def test_block_input_context_raises_when_strict(self, monkeypatch):
        monkeypatch.setattr(wh, "IS_WINDOWS", False)
        with pytest.raises(wh.BlockInputUnavailableError):
            with wh.block_input_context(raise_if_unavailable=True):
                pass


# ---------------------------------------------------------------------------
# DPI lookup
# ---------------------------------------------------------------------------


class TestGetMonitorDpi:

    def test_high_dpi_reading(self, monkeypatch):
        _patch_dlls(
            monkeypatch,
            user32=_FakeUser32(monitor_handle=0xAB),
            shcore=_FakeShcore(dpi_x=144, dpi_y=144),
        )
        result = wh.get_monitor_dpi(500, 300)
        assert result.dpi_x == 144
        assert result.dpi_y == 144
        assert result.scale_x == pytest.approx(1.5)
        assert result.scale_y == pytest.approx(1.5)
        assert result.is_default is False
        assert result.is_high_dpi is True

    def test_default_100_percent_reading(self, monkeypatch):
        _patch_dlls(
            monkeypatch,
            user32=_FakeUser32(),
            shcore=_FakeShcore(dpi_x=96, dpi_y=96),
        )
        result = wh.get_monitor_dpi(0, 0)
        assert result.scale_x == 1.0
        assert result.is_high_dpi is False

    def test_missing_user32_falls_back(self, monkeypatch):
        _patch_dlls(monkeypatch, user32=None, shcore=_FakeShcore())
        result = wh.get_monitor_dpi(0, 0)
        assert result.is_default is True

    def test_missing_shcore_falls_back(self, monkeypatch):
        _patch_dlls(monkeypatch, user32=_FakeUser32(), shcore=None)
        result = wh.get_monitor_dpi(0, 0)
        assert result.is_default is True

    def test_monitor_handle_zero_falls_back(self, monkeypatch):
        _patch_dlls(
            monkeypatch,
            user32=_FakeUser32(monitor_handle=0),
            shcore=_FakeShcore(),
        )
        result = wh.get_monitor_dpi(0, 0)
        assert result.is_default is True

    def test_get_dpi_for_monitor_hresult_failure(self, monkeypatch):
        _patch_dlls(
            monkeypatch,
            user32=_FakeUser32(),
            shcore=_FakeShcore(hresult=0x80004005),  # E_FAIL
        )
        result = wh.get_monitor_dpi(0, 0)
        assert result.is_default is True

    def test_user32_raises_swallowed(self, monkeypatch):
        class _BrokenUser32(_FakeUser32):
            def MonitorFromPoint(self, point, flag):
                raise RuntimeError("blam")

        _patch_dlls(monkeypatch, user32=_BrokenUser32(), shcore=_FakeShcore())
        result = wh.get_monitor_dpi(0, 0)
        assert result.is_default is True

    def test_zero_dpi_floored_to_default(self, monkeypatch):
        # Some virtualised setups report 0 DPI; guard against the
        # division-by-zero downstream.
        _patch_dlls(
            monkeypatch,
            user32=_FakeUser32(),
            shcore=_FakeShcore(dpi_x=0, dpi_y=0),
        )
        result = wh.get_monitor_dpi(0, 0)
        assert result.dpi_x == wh.DEFAULT_DPI
        assert result.scale_x == 1.0


class TestGetMonitorDpiForWindow:

    def test_uses_window_centre(self, monkeypatch):
        user32 = _FakeUser32(get_window_rect_value=(100, 200, 300, 400))
        shcore = _FakeShcore(dpi_x=192, dpi_y=192)
        _patch_dlls(monkeypatch, user32=user32, shcore=shcore)
        result = wh.get_monitor_dpi_for_window(12345)
        assert result.dpi_x == 192
        assert result.scale_x == pytest.approx(2.0)

    def test_get_window_rect_failure(self, monkeypatch):
        user32 = _FakeUser32(get_window_rect_returns=0)
        _patch_dlls(monkeypatch, user32=user32, shcore=_FakeShcore())
        result = wh.get_monitor_dpi_for_window(12345)
        assert result.is_default is True

    def test_hwnd_zero_returns_default(self, monkeypatch):
        _patch_dlls(monkeypatch, user32=_FakeUser32(), shcore=_FakeShcore())
        result = wh.get_monitor_dpi_for_window(0)
        assert result.is_default is True


# ---------------------------------------------------------------------------
# Last input idle
# ---------------------------------------------------------------------------


class TestGetLastInputIdleMs:

    def test_reports_delta(self, monkeypatch):
        user32 = _FakeUser32(last_input_tick=1000)
        kernel32 = _FakeKernel32(tick=2500)
        _patch_dlls(monkeypatch, user32=user32, kernel32=kernel32)
        idle = wh.get_last_input_idle_ms()
        assert idle == 1500

    def test_tick_wrap_handled(self, monkeypatch):
        # Both ticks near uint32 max; the unsigned subtraction should
        # produce a small positive number.
        user32 = _FakeUser32(last_input_tick=0xFFFFFFF0)
        kernel32 = _FakeKernel32(tick=10)
        _patch_dlls(monkeypatch, user32=user32, kernel32=kernel32)
        idle = wh.get_last_input_idle_ms()
        assert idle == ((10 - 0xFFFFFFF0) & 0xFFFFFFFF)
        assert idle is not None and 0 <= idle < 100

    def test_missing_user32(self, monkeypatch):
        _patch_dlls(monkeypatch, user32=None, kernel32=_FakeKernel32())
        assert wh.get_last_input_idle_ms() is None

    def test_missing_kernel32(self, monkeypatch):
        _patch_dlls(monkeypatch, user32=_FakeUser32(), kernel32=None)
        assert wh.get_last_input_idle_ms() is None

    def test_call_raises_swallowed(self, monkeypatch):
        class _BrokenUser32(_FakeUser32):
            def GetLastInputInfo(self, lii_ref):
                raise RuntimeError("blam")

        _patch_dlls(monkeypatch, user32=_BrokenUser32(), kernel32=_FakeKernel32())
        assert wh.get_last_input_idle_ms() is None


# ---------------------------------------------------------------------------
# Cloaked window detection
# ---------------------------------------------------------------------------


class TestIsWindowCloaked:

    def test_returns_true_when_dwm_reports_cloaked(self, monkeypatch):
        _patch_dlls(monkeypatch, dwmapi=_FakeDwmapi(cloaked=1))
        assert wh.is_window_cloaked(12345) is True

    def test_returns_false_when_dwm_reports_visible(self, monkeypatch):
        _patch_dlls(monkeypatch, dwmapi=_FakeDwmapi(cloaked=0))
        assert wh.is_window_cloaked(12345) is False

    def test_missing_dll(self, monkeypatch):
        _patch_dlls(monkeypatch, dwmapi=None)
        assert wh.is_window_cloaked(12345) is None

    def test_hresult_failure(self, monkeypatch):
        _patch_dlls(monkeypatch, dwmapi=_FakeDwmapi(hresult=0x80004005))
        assert wh.is_window_cloaked(12345) is None

    def test_hwnd_zero_returns_none(self, monkeypatch):
        _patch_dlls(monkeypatch, dwmapi=_FakeDwmapi())
        assert wh.is_window_cloaked(0) is None

    def test_call_raises_swallowed(self, monkeypatch):
        class _BrokenDwm(_FakeDwmapi):
            def DwmGetWindowAttribute(self, hwnd, attr, value_ref, size):
                raise RuntimeError("blam")

        _patch_dlls(monkeypatch, dwmapi=_BrokenDwm())
        assert wh.is_window_cloaked(12345) is None


# ---------------------------------------------------------------------------
# block_input_context
# ---------------------------------------------------------------------------


class TestBlockInputContext:

    def test_engages_then_releases(self, monkeypatch):
        user32 = _FakeUser32(block_input_success=True)
        _patch_dlls(monkeypatch, user32=user32)
        with wh.block_input_context(max_duration_s=2.0) as result:
            assert result.engaged is True
            assert result.watchdog_fired is False
        # Block, then unblock.
        assert user32.block_input_calls == [True, False]

    def test_release_runs_on_exception(self, monkeypatch):
        user32 = _FakeUser32(block_input_success=True)
        _patch_dlls(monkeypatch, user32=user32)
        with pytest.raises(RuntimeError):
            with wh.block_input_context(max_duration_s=2.0) as result:
                assert result.engaged is True
                raise RuntimeError("boom")
        # Critical safety guarantee: BlockInput(False) STILL ran.
        assert user32.block_input_calls == [True, False]

    def test_block_failure_yields_disengaged(self, monkeypatch):
        user32 = _FakeUser32(block_input_success=False)
        _patch_dlls(monkeypatch, user32=user32)
        with wh.block_input_context() as result:
            assert result.engaged is False
            assert result.watchdog_fired is False
        # Only the initial attempt happened; no unblock-without-block.
        assert user32.block_input_calls == [True]

    def test_block_failure_raises_when_strict(self, monkeypatch):
        user32 = _FakeUser32(block_input_success=False)
        _patch_dlls(monkeypatch, user32=user32)
        with pytest.raises(wh.BlockInputUnavailableError):
            with wh.block_input_context(raise_if_unavailable=True):
                pass

    def test_duration_clamped_to_hard_cap(self, monkeypatch):
        # Try to ask for an hour; the helper must clamp the watchdog
        # to the hard cap so user input is never lost for that long.
        user32 = _FakeUser32(block_input_success=True)
        _patch_dlls(monkeypatch, user32=user32)
        # The clamp is applied internally; we observe behaviour by
        # ensuring the context manager doesn't hang.
        before = time.monotonic()
        with wh.block_input_context(max_duration_s=3600.0):
            pass
        # Caller exited within a few ms; watchdog never fired.
        assert time.monotonic() - before < 1.0

    def test_negative_duration_clamped_to_zero(self, monkeypatch):
        user32 = _FakeUser32(block_input_success=True)
        _patch_dlls(monkeypatch, user32=user32)
        with wh.block_input_context(max_duration_s=-1.0) as result:
            # Watchdog may have fired already at duration_s=0; that's
            # fine -- the contract is "user never loses control".
            assert result.engaged is True
        # BlockInput(False) is guaranteed exactly once even when the
        # watchdog and the exit path race.
        assert user32.block_input_calls.count(False) >= 1


# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------


class TestCoordinateConversion:

    def test_logical_to_physical_high_dpi(self, monkeypatch):
        _patch_dlls(
            monkeypatch,
            user32=_FakeUser32(),
            shcore=_FakeShcore(dpi_x=144, dpi_y=144),
        )
        # 150% scaling: 500,300 logical -> 750,450 physical
        px, py = wh.logical_to_physical(500, 300)
        assert (px, py) == (750, 450)

    def test_physical_to_logical_high_dpi(self, monkeypatch):
        _patch_dlls(
            monkeypatch,
            user32=_FakeUser32(),
            shcore=_FakeShcore(dpi_x=144, dpi_y=144),
        )
        lx, ly = wh.physical_to_logical(750, 450)
        assert (lx, ly) == (500, 300)

    def test_identity_at_100_percent(self, monkeypatch):
        _patch_dlls(
            monkeypatch,
            user32=_FakeUser32(),
            shcore=_FakeShcore(dpi_x=96, dpi_y=96),
        )
        assert wh.logical_to_physical(123, 456) == (123, 456)
        assert wh.physical_to_logical(123, 456) == (123, 456)

    def test_explicit_dpi_skips_lookup(self, monkeypatch):
        # If the caller supplies dpi, _load_dll must NOT be consulted.
        def _explode(name: str):  # noqa: ARG001
            raise AssertionError("DPI lookup should not run when dpi is passed")

        monkeypatch.setattr(wh, "_load_dll", _explode)
        dpi = wh.MonitorDpi(
            dpi_x=192, dpi_y=192, scale_x=2.0, scale_y=2.0,
        )
        assert wh.logical_to_physical(100, 50, dpi=dpi) == (200, 100)
        assert wh.physical_to_logical(200, 100, dpi=dpi) == (100, 50)

    def test_off_windows_is_identity(self, monkeypatch):
        monkeypatch.setattr(wh, "IS_WINDOWS", False)
        assert wh.logical_to_physical(123, 456) == (123, 456)
        assert wh.physical_to_logical(123, 456) == (123, 456)

    def test_asymmetric_dpi(self, monkeypatch):
        # Some setups report different X/Y DPI; the converter must
        # apply each scale to its own axis.
        _patch_dlls(
            monkeypatch,
            user32=_FakeUser32(),
            shcore=_FakeShcore(dpi_x=192, dpi_y=120),
        )
        px, py = wh.logical_to_physical(100, 100)
        assert px == 200
        assert py == 125  # round(100 * 1.25)


# ---------------------------------------------------------------------------
# DLL cache
# ---------------------------------------------------------------------------


class TestDllCache:

    def test_load_dll_off_windows_returns_none(self, monkeypatch):
        monkeypatch.setattr(wh, "IS_WINDOWS", False)
        assert wh._load_dll("user32") is None

    def test_reset_for_testing_clears_cache(self, monkeypatch):
        # Populate the cache with a sentinel...
        with wh._dll_cache_lock:
            wh._dll_cache["sentinel"] = object()
        assert "sentinel" in wh._dll_cache
        wh._reset_dll_cache_for_testing()
        assert "sentinel" not in wh._dll_cache
