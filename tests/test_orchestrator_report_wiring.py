"""Tests for the openclaw-clawhub T12 report-queue wiring in the
orchestrator (Batch C of the deferred-primitive wiring pass).

Orchestrator.__new__ pattern; no voice stack. The real round-trip
redirects ultron.config.PROJECT_ROOT to tmp_path so nothing touches
the repo data/ dir (R9).
"""

from __future__ import annotations

from typing import Any

import pytest


def _bare_orchestrator() -> Any:
    from ultron.pipeline.orchestrator import Orchestrator

    o = Orchestrator.__new__(Orchestrator)
    o._report_queue = None
    o._last_response_text = ""
    o.memory = None
    o._spoken: list = []
    o._speak = lambda text: o._spoken.append(text)  # type: ignore[attr-defined]
    return o


class _FakeQueue:
    def __init__(self) -> None:
        self.filed: list = []
        self.raises = False

    def file_report(self, **kwargs: Any) -> Any:
        if self.raises:
            raise RuntimeError("queue boom")
        self.filed.append(kwargs)
        return object()


# ---------------------------------------------------------------------------
# _init_report_queue
# ---------------------------------------------------------------------------


class TestInitReportQueue:
    def test_real_construction_under_tmp(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import ultron.config as cfgmod
        from ultron.feedback.report_queue import ReportQueue

        monkeypatch.setattr(cfgmod, "PROJECT_ROOT", tmp_path)
        o = _bare_orchestrator()
        q = o._init_report_queue()
        assert isinstance(q, ReportQueue)
        # The feedback dir is created.
        assert (tmp_path / "data" / "feedback").is_dir()


# ---------------------------------------------------------------------------
# _maybe_handle_report_concern
# ---------------------------------------------------------------------------


class TestMaybeHandleReportConcern:
    def test_non_match_returns_false(self) -> None:
        o = _bare_orchestrator()
        o._report_queue = _FakeQueue()
        assert o._maybe_handle_report_concern("what's the weather") is False

    def test_no_queue_falls_through(self) -> None:
        o = _bare_orchestrator()
        o._report_queue = None
        # Even a valid trigger falls through (returns False) when no
        # queue is wired, so the user gets an LLM response, not silence.
        assert o._maybe_handle_report_concern("flag that response") is False

    def test_files_report_and_acks(self) -> None:
        o = _bare_orchestrator()
        q = _FakeQueue()
        o._report_queue = q
        o._last_response_text = "The capital of France is Berlin."
        handled = o._maybe_handle_report_concern(
            "log a concern that the last response was wrong"
        )
        assert handled is True
        assert len(q.filed) == 1
        filed = q.filed[0]
        from ultron.feedback.report_queue import ReportTargetKind

        assert filed["target_kind"] is ReportTargetKind.RESPONSE
        assert filed["reason"].startswith("log a concern")
        # target_id is a 16-hex digest of the prior response.
        assert len(filed["target_id"]) == 16
        assert filed["extras"]["response_preview"].startswith("The capital")
        # An ack was spoken.
        assert o._spoken
        assert "filed a concern" in o._spoken[0]

    def test_no_prior_turn_target_id(self) -> None:
        o = _bare_orchestrator()
        q = _FakeQueue()
        o._report_queue = q
        o._last_response_text = ""
        handled = o._maybe_handle_report_concern("flag that response")
        assert handled is True
        assert q.filed[0]["target_id"] == "no_prior_turn"
        assert "no prior response" in o._spoken[0]

    def test_memory_target_kind(self) -> None:
        o = _bare_orchestrator()
        q = _FakeQueue()
        o._report_queue = q
        o._last_response_text = "x"
        o._maybe_handle_report_concern(
            "log a concern that you misremembered my preference"
        )
        from ultron.feedback.report_queue import ReportTargetKind

        assert q.filed[0]["target_kind"] is ReportTargetKind.MEMORY

    def test_fail_open_when_filing_raises(self) -> None:
        o = _bare_orchestrator()
        q = _FakeQueue()
        q.raises = True
        o._report_queue = q
        o._last_response_text = "x"
        # Still handled (True) + a clear message spoken; no exception.
        handled = o._maybe_handle_report_concern("flag that response")
        assert handled is True
        assert "couldn't log" in o._spoken[0]

    def test_real_queue_round_trip(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import ultron.config as cfgmod
        from ultron.feedback.report_queue import ReportQueue, ReportStatus

        monkeypatch.setattr(cfgmod, "PROJECT_ROOT", tmp_path)
        o = _bare_orchestrator()
        log_path = tmp_path / "data" / "feedback" / "reports.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        o._report_queue = ReportQueue(audit_log_path=log_path)
        o._last_response_text = "A wrong answer."
        handled = o._maybe_handle_report_concern("that answer was wrong")
        assert handled is True
        # The report was persisted with OPEN status.
        assert log_path.exists()
        body = log_path.read_text().strip()
        assert body != ""
        assert ReportStatus.OPEN.value in body
