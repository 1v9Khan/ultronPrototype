"""Tests for the SEMANTIC_CLICK deep-discovery fallback (#72b).

Covers the gating (LLM wired + config knob), the retry-on-candidate
flow through the fully-gated click path, and fail-open behaviour. The
DeepUIDiscoveryLoop + element_click primitives are monkeypatched at
their modules so no UIA / LLM is touched (binding rules R4/R11).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import ultron.agent_loop.deep_loops as deep_loops_mod
import ultron.desktop.element_click as element_click_mod
import ultron.openclaw_routing as routing_pkg
from ultron.coding.voice import CapabilityVoiceController


def _controller(llm: Any = object()) -> Any:
    c = CapabilityVoiceController.__new__(CapabilityVoiceController)
    c.llm_engine = llm
    return c


def _intent(name: str = "Save", window: str = "") -> Any:
    return SimpleNamespace(element_name=name, window_title=window, control_type="")


def _routing_intent(raw: str = "click the save button") -> Any:
    return SimpleNamespace(raw_text=raw)


class _FakeLoop:
    """Stands in for DeepUIDiscoveryLoop; returns canned candidates."""

    instances: list = []

    def __init__(self, *, find, llm, max_steps=3, **kwargs: Any) -> None:
        self.find = find
        self.llm = llm
        self.max_steps = max_steps
        _FakeLoop.instances.append(self)

    def discover(self, target: str) -> Any:
        return SimpleNamespace(
            items=[SimpleNamespace(name="Save As"), SimpleNamespace(name="OK")],
            sub_queries=[target, "Save As", "OK"],
        )


@pytest.fixture(autouse=True)
def _reset_fake_loop() -> None:
    _FakeLoop.instances = []


class TestDeepClickFallback:
    def test_none_when_llm_missing(self) -> None:
        c = _controller(llm=None)
        assert c._deep_discover_click_retry(_intent(), _routing_intent()) is None

    def test_none_when_knob_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import ultron.config as cfgmod

        cfg = SimpleNamespace(desktop=SimpleNamespace(deep_ui_discovery_enabled=False))
        monkeypatch.setattr(cfgmod, "get_config", lambda: cfg)
        c = _controller()
        assert c._deep_discover_click_retry(_intent(), _routing_intent()) is None

    def test_successful_deep_click(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import ultron.config as cfgmod

        cfg = SimpleNamespace(desktop=SimpleNamespace(deep_ui_discovery_enabled=True))
        monkeypatch.setattr(cfgmod, "get_config", lambda: cfg)
        monkeypatch.setattr(deep_loops_mod, "DeepUIDiscoveryLoop", _FakeLoop)
        clicks: list[str] = []

        def _click(name: str, **kwargs: Any) -> Any:
            clicks.append(name)
            return SimpleNamespace(success=True, window_title="Notepad")

        monkeypatch.setattr(element_click_mod, "click_element_by_name", _click)
        logged: list[dict] = []
        monkeypatch.setattr(
            routing_pkg,
            "get_routing_log",
            lambda: SimpleNamespace(
                record=lambda *a, **k: logged.append(k)
            ),
        )
        c = _controller()
        resp = c._deep_discover_click_retry(_intent("Save"), _routing_intent())
        assert resp is not None
        assert clicks == ["Save As"]  # first candidate clicked
        assert "looked deeper" in resp.text
        assert "Save As" in resp.text
        assert logged and logged[0]["outcome"] == "dispatched_deep"

    def test_all_candidates_fail_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import ultron.config as cfgmod

        cfg = SimpleNamespace(desktop=SimpleNamespace(deep_ui_discovery_enabled=True))
        monkeypatch.setattr(cfgmod, "get_config", lambda: cfg)
        monkeypatch.setattr(deep_loops_mod, "DeepUIDiscoveryLoop", _FakeLoop)
        monkeypatch.setattr(
            element_click_mod,
            "click_element_by_name",
            lambda name, **k: SimpleNamespace(success=False, window_title=""),
        )
        c = _controller()
        assert c._deep_discover_click_retry(_intent(), _routing_intent()) is None

    def test_loop_raise_is_fail_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import ultron.config as cfgmod

        cfg = SimpleNamespace(desktop=SimpleNamespace(deep_ui_discovery_enabled=True))
        monkeypatch.setattr(cfgmod, "get_config", lambda: cfg)

        class _BoomLoop:
            def __init__(self, **kwargs: Any) -> None:
                raise RuntimeError("loop boom")

        monkeypatch.setattr(deep_loops_mod, "DeepUIDiscoveryLoop", _BoomLoop)
        c = _controller()
        assert c._deep_discover_click_retry(_intent(), _routing_intent()) is None

    def test_window_title_threaded_into_find(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import ultron.config as cfgmod

        cfg = SimpleNamespace(desktop=SimpleNamespace(deep_ui_discovery_enabled=True))
        monkeypatch.setattr(cfgmod, "get_config", lambda: cfg)
        monkeypatch.setattr(deep_loops_mod, "DeepUIDiscoveryLoop", _FakeLoop)
        finds: list[dict] = []

        def _find(name: str, **kwargs: Any) -> list:
            finds.append({"name": name, **kwargs})
            return []

        monkeypatch.setattr(element_click_mod, "find_elements_by_name", _find)
        monkeypatch.setattr(
            element_click_mod,
            "click_element_by_name",
            lambda name, **k: SimpleNamespace(success=False, window_title=""),
        )
        c = _controller()
        c._deep_discover_click_retry(_intent("Save", window="Notepad"), _routing_intent())
        # Exercise the loop's injected find adapter.
        assert _FakeLoop.instances
        _FakeLoop.instances[0].find("Save As")
        assert finds and finds[0]["window_title"] == "Notepad"
