"""Unit tests for the HYBRID_TASK decomposer wiring (``_handle_hybrid_task``).

Production-hardening finding #24: the ``HybridTaskDecomposer`` was fully built
but never called from the voice path -- HYBRID_TASK returned a hardcoded
"the gateway isn't connected yet" stub. These tests pin the bounded dispatch
contract that the wiring introduced:

  * Automation subtasks BEFORE any coding subtask run inline (the common
    "read this / open that, THEN build X" shape).
  * The first coding subtask dispatches through the coding pipeline.
  * Anything AFTER the coding dispatch (a 2nd coding subtask, or automation
    that must follow the code) is surfaced as a deferred plan, NOT fired out
    of order -- ultron holds one in-flight task and the voice turn is sync.

All hermetic: a fake async decomposer + stubbed dispatch seams; no LLM, no
subprocess, no real routing-log file (scoped to tmp_path).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ultron.coding.bridge import CodingBridge
from ultron.coding.projects import ProjectRegistry, ProjectResolver
from ultron.coding.runner import CodingTaskRunner
from ultron.coding.voice import CapabilityVoiceController, VoiceResponse
from ultron.openclaw_routing import RoutingDecisionLog, set_routing_log
from ultron.openclaw_routing.decomposer import DecompositionResult
from ultron.openclaw_routing.intents import HybridSubtask


class _NoopBridge(CodingBridge):
    def submit(self, request):
        raise AssertionError("these tests stub handle_utterance; no submit")

    def name(self) -> str:
        return "noop"


class _NoopRoutingLog:
    """Duck-typed RoutingDecisionLog whose record() is a no-op -- these tests
    pin dispatch ordering, not the (separately-tested) routing-log shape, and
    use lightweight fake intents that lack the full RoutingIntent fields."""

    def record(self, intent, **kwargs) -> None:
        pass


@pytest.fixture(autouse=True)
def _scoped_routing_log():
    set_routing_log(_NoopRoutingLog())
    yield
    set_routing_log(RoutingDecisionLog())


def _controller(tmp_path) -> CapabilityVoiceController:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir(exist_ok=True)
    registry = ProjectRegistry(path=tmp_path / "projects.json")
    resolver = ProjectResolver(registry, embedder=None)
    runner = CodingTaskRunner(bridge=_NoopBridge(), log_path=tmp_path / "log.jsonl")
    return CapabilityVoiceController(
        runner=runner, registry=registry, resolver=resolver, sandbox_root=sandbox,
    )


def _fake_decomposer(result):
    """Return a HybridTaskDecomposer-shaped class whose decompose() yields
    ``result`` (a coroutine, matching the real async API)."""
    class _FD:
        def __init__(self, llm):
            self._llm = llm

        async def decompose(self, utterance):
            return result

    return _FD


def _intent(text="do a then b"):
    return SimpleNamespace(raw_text=text)


def test_automation_before_coding_runs_inline(tmp_path, monkeypatch):
    ctrl = _controller(tmp_path)
    result = DecompositionResult(
        subtasks=[
            HybridSubtask(order=1, type="automation", subtype="browser",
                          description="open hacker news"),
            HybridSubtask(order=2, type="coding", description="build a scraper"),
        ],
        fallback_used=False,
    )
    monkeypatch.setattr(
        "ultron.openclaw_routing.HybridTaskDecomposer", _fake_decomposer(result),
    )
    calls = []
    monkeypatch.setattr(
        ctrl, "_dispatch_automation_subtask",
        lambda d: (calls.append(("auto", d)), "Opened the page.")[1],
    )
    monkeypatch.setattr(
        ctrl, "handle_utterance",
        lambda d: (calls.append(("code", d)),
                   VoiceResponse(text="On it.", handled=True))[1],
    )
    resp = ctrl._handle_hybrid_task(_intent())
    assert resp.handled is True
    # Automation ran inline (it precedes coding), then coding dispatched.
    assert calls == [("auto", "open hacker news"), ("code", "build a scraper")]
    assert "Opened the page." in resp.text and "On it." in resp.text


def test_automation_after_coding_is_deferred(tmp_path, monkeypatch):
    ctrl = _controller(tmp_path)
    result = DecompositionResult(
        subtasks=[
            HybridSubtask(order=1, type="coding", description="fix the login bug"),
            HybridSubtask(order=2, type="automation", subtype="browser",
                          description="open the app in chrome"),
        ],
        fallback_used=False,
    )
    monkeypatch.setattr(
        "ultron.openclaw_routing.HybridTaskDecomposer", _fake_decomposer(result),
    )
    calls = []
    monkeypatch.setattr(
        ctrl, "_dispatch_automation_subtask",
        lambda d: (calls.append(("auto", d)), "ran")[1],
    )
    monkeypatch.setattr(
        ctrl, "handle_utterance",
        lambda d: (calls.append(("code", d)),
                   VoiceResponse(text="Fixing it.", handled=True))[1],
    )
    resp = ctrl._handle_hybrid_task(_intent())
    # Coding dispatched; the post-coding automation is NOT run, only surfaced.
    assert calls == [("code", "fix the login bug")]
    assert "open the app in chrome" in resp.text
    assert "continue" in resp.text.lower()


def test_only_first_coding_subtask_dispatched(tmp_path, monkeypatch):
    ctrl = _controller(tmp_path)
    result = DecompositionResult(
        subtasks=[
            HybridSubtask(order=1, type="coding", description="build module A"),
            HybridSubtask(order=2, type="coding", description="build module B"),
        ],
        fallback_used=False,
    )
    monkeypatch.setattr(
        "ultron.openclaw_routing.HybridTaskDecomposer", _fake_decomposer(result),
    )
    seen = []
    monkeypatch.setattr(
        ctrl, "handle_utterance",
        lambda d: (seen.append(d),
                   VoiceResponse(text="Building.", handled=True))[1],
    )
    resp = ctrl._handle_hybrid_task(_intent())
    # Only the first coding subtask dispatches (single in-flight task model).
    assert seen == ["build module A"]
    assert "build module B" in resp.text


def test_decompose_failure_falls_back_to_coding(tmp_path, monkeypatch):
    ctrl = _controller(tmp_path)

    class _Boom:
        def __init__(self, llm):
            pass

        async def decompose(self, utterance):
            raise RuntimeError("llm down")

    monkeypatch.setattr("ultron.openclaw_routing.HybridTaskDecomposer", _Boom)
    seen = []
    monkeypatch.setattr(
        ctrl, "handle_utterance",
        lambda d: (seen.append(d),
                   VoiceResponse(text="Coding.", handled=True))[1],
    )
    resp = ctrl._handle_hybrid_task(_intent("make a thing"))
    # Decompose raised -> empty plan -> raw utterance dispatched as coding.
    assert seen == ["make a thing"]
    assert resp.handled is True


def test_automation_subtask_classified_and_dispatched(tmp_path, monkeypatch):
    """_dispatch_automation_subtask classifies the description and routes a
    real automation kind through the runner; HYBRID/CONVERSATIONAL re-classes
    are surfaced as text (no recursion)."""
    from ultron.openclaw_routing.intents import RoutingIntentKind

    ctrl = _controller(tmp_path)
    dispatched = []
    monkeypatch.setattr(
        ctrl, "_dispatch_via_automation_runner",
        lambda ri: (dispatched.append(ri.kind),
                    VoiceResponse(text="done", handled=True))[1],
    )

    # A real automation kind -> dispatched.
    monkeypatch.setattr(
        "ultron.openclaw_routing.classifier.classify_routing",
        lambda d: SimpleNamespace(kind=RoutingIntentKind.BROWSER_AUTOMATION),
    )
    assert ctrl._dispatch_automation_subtask("open a page") == "done"
    assert dispatched == [RoutingIntentKind.BROWSER_AUTOMATION]

    # A HYBRID re-class -> surfaced as text, NOT dispatched (no recursion).
    monkeypatch.setattr(
        "ultron.openclaw_routing.classifier.classify_routing",
        lambda d: SimpleNamespace(kind=RoutingIntentKind.HYBRID_TASK),
    )
    out = ctrl._dispatch_automation_subtask("a mixed thing")
    assert "mixed thing" in out
    assert dispatched == [RoutingIntentKind.BROWSER_AUTOMATION]  # unchanged
