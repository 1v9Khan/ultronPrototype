"""The resolve_observation_outcomes maintenance task gives the offline
OutcomeResolver (previously consumer-less) a runnable home.

These validate the offline pass the task wraps -- resolve_outcomes over the
canonical observation log -- which needs neither the LLM nor Qdrant.
"""

from __future__ import annotations


def test_resolve_outcomes_runs_on_empty_log(tmp_path):
    from ultron.observations import resolve_outcomes

    log = tmp_path / "observations.jsonl"
    log.write_text("", encoding="utf-8")
    summary = resolve_outcomes(observations_path=log)
    assert summary.scanned == 0
    assert summary.resolved_now == 0


def test_resolve_outcomes_missing_log_is_safe(tmp_path):
    from ultron.observations import resolve_outcomes

    # A missing log must not raise -- the maintenance task is idempotent.
    summary = resolve_outcomes(observations_path=tmp_path / "nope.jsonl")
    assert summary.resolved_now == 0


def test_resolve_outcomes_summary_as_dict(tmp_path):
    from ultron.observations import resolve_outcomes

    log = tmp_path / "observations.jsonl"
    log.write_text("", encoding="utf-8")
    d = resolve_outcomes(observations_path=log).as_dict()
    assert d["scanned"] == 0
    assert d["resolved_now"] == 0
    assert "by_outcome" in d
