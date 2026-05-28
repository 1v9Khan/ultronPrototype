"""Tests for ultron.evolution.guardrails -- regression detectors +
auto-revert + rollback-frequency audit. All hermetic."""

from __future__ import annotations

from ultron.evolution import guardrails as G


def _baseline(**kw):
    return G.GuardrailBaseline(**kw)


# --- latency ----------------------------------------------------------------


def test_latency_regression_trips_over_tolerance():
    base = _baseline(ttfa_ms=266)
    hit, details = G.detect_latency_regression(base, G.GuardrailSample(ttfa_ms=320))
    assert hit is True
    assert details


def test_latency_within_tolerance_no_trip():
    base = _baseline(ttfa_ms=266)
    # 266 * 1.15 = 305.9; 300 is under
    hit, _ = G.detect_latency_regression(base, G.GuardrailSample(ttfa_ms=300))
    assert hit is False


def test_latency_none_skipped():
    hit, _ = G.detect_latency_regression(_baseline(), G.GuardrailSample())
    assert hit is False


# --- quality / error --------------------------------------------------------


def test_quality_regression_trips():
    base = _baseline(correction_rate=0.1)
    hit, _ = G.detect_quality_regression(base, G.GuardrailSample(correction_rate=0.25))
    assert hit is True


def test_quality_small_increase_no_trip():
    base = _baseline(correction_rate=0.1)
    hit, _ = G.detect_quality_regression(base, G.GuardrailSample(correction_rate=0.15))
    assert hit is False


def test_error_regression_trips():
    base = _baseline(error_rate=0.05)
    hit, _ = G.detect_error_regression(base, G.GuardrailSample(error_rate=0.30))
    assert hit is True


# --- resource ---------------------------------------------------------------


def test_resource_ceiling_trips_near_cap():
    hit, _ = G.detect_resource_ceiling(G.GuardrailSample(vram_peak_mb=11000), cap_mb=11500)
    assert hit is True  # 11000 >= 11500*0.95 = 10925


def test_resource_ceiling_ok_with_headroom():
    hit, _ = G.detect_resource_ceiling(G.GuardrailSample(vram_peak_mb=7000), cap_mb=11500)
    assert hit is False


# --- aggregate --------------------------------------------------------------


def test_evaluate_no_trip():
    verdict = G.evaluate_guardrails(
        _baseline(),
        G.GuardrailSample(ttfa_ms=260, ttft_ms=170, tts_ms=75, correction_rate=0.0, error_rate=0.0, vram_peak_mb=6700),
    )
    assert verdict.tripped is False
    assert verdict.should_revert is False


def test_evaluate_multiple_trips():
    verdict = G.evaluate_guardrails(
        _baseline(),
        G.GuardrailSample(ttfa_ms=400, vram_peak_mb=11400),
    )
    assert verdict.tripped is True
    assert "latency" in verdict.tripped_guards
    assert "resource" in verdict.tripped_guards
    assert verdict.should_revert is True


def test_evaluate_respects_disabled_guard():
    cfg = G.GuardrailConfig(enabled=("quality", "error"))
    verdict = G.evaluate_guardrails(_baseline(), G.GuardrailSample(ttfa_ms=9999), config=cfg)
    assert verdict.tripped is False  # latency disabled


def test_summarize_verdict():
    assert G.summarize_guardrail_verdict(G.GuardrailVerdict(tripped=False)) == "all guardrails passed"
    v = G.GuardrailVerdict(tripped=True, tripped_guards=("latency",))
    assert "latency" in G.summarize_guardrail_verdict(v)


# --- rollback rate ----------------------------------------------------------


def test_compute_rollback_rate():
    assert G.compute_rollback_rate(0, 0) == 0.0
    assert G.compute_rollback_rate(10, 3) == 0.3


def test_should_demote_for_rollback_rate():
    assert G.should_demote_for_rollback_rate(0.4, applied=10) is True
    assert G.should_demote_for_rollback_rate(0.4, applied=3) is False  # too few samples
    assert G.should_demote_for_rollback_rate(0.2, applied=10) is False  # under threshold


def test_format_rollback_audit_line():
    rec = G.RollbackRecord(surface="skills", change_id="c1", guardrail="latency", metric_delta="ttfa 400 vs 266")
    line = G.format_rollback_audit_line(rec)
    assert "skills" in line and "latency" in line and "ttfa" in line


# --- RollbackAudit ----------------------------------------------------------


def test_rollback_audit_rate_and_records():
    audit = G.RollbackAudit(window=10)
    for _ in range(7):
        audit.note_outcome("skills", reverted=False)
    for i in range(3):
        audit.note_outcome(
            "skills", reverted=True, record=G.RollbackRecord("skills", f"c{i}", "latency")
        )
    assert audit.rollback_rate("skills") == 0.3
    s = audit.stats("skills")
    assert s.applied == 10 and s.reverted == 3
    assert len(audit.records()) == 3
    assert audit.totals("skills") == (10, 3)
    assert "skills" in audit.surfaces()


def test_rollback_audit_should_demote():
    audit = G.RollbackAudit(window=10)
    for i in range(10):
        audit.note_outcome("memory", reverted=(i < 5))  # 5/10 reverted
    assert audit.should_demote("memory") is True


def test_rollback_audit_windowed_eviction():
    audit = G.RollbackAudit(window=2)
    audit.note_outcome("s", reverted=False)
    audit.note_outcome("s", reverted=False)
    audit.note_outcome("s", reverted=True)
    audit.note_outcome("s", reverted=True)
    # window holds only the last 2 (both reverted)
    assert audit.rollback_rate("s") == 1.0
    assert audit.totals("s") == (4, 2)  # lifetime totals unaffected by window


def test_rollback_audit_unknown_surface():
    audit = G.RollbackAudit()
    assert audit.rollback_rate("nope") == 0.0
    assert audit.should_demote("nope") is False
