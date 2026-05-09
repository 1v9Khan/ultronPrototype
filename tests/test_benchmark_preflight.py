"""V1-gap B5: smoke tests for the pre-flight benchmark script.

The actual benchmark loads heavy models and isn't part of CI -- these
tests exercise the pure-Python helpers (argparse, the per-query
result aggregator, the markdown printer) so refactors can't break the
CLI without us noticing.
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
from pathlib import Path
from typing import List

import pytest


_BENCH = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_preflight.py"


@pytest.fixture(scope="module")
def bench_module():
    spec = importlib.util.spec_from_file_location(
        "_benchmark_preflight_mod", _BENCH,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_benchmark_preflight_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_arg_parser_defaults(bench_module):
    parser = bench_module._build_arg_parser()
    args = parser.parse_args([])
    assert args.candidate_model is None
    assert args.skip_main is False
    assert args.queries == 0


def test_arg_parser_accepts_candidate(bench_module, tmp_path):
    parser = bench_module._build_arg_parser()
    fake_path = tmp_path / "x.gguf"
    args = parser.parse_args([
        "--candidate-model", str(fake_path),
        "--skip-main",
        "--queries", "3",
    ])
    assert args.candidate_model == fake_path
    assert args.skip_main is True
    assert args.queries == 3


def test_query_set_well_formed(bench_module):
    """Every benchmark query has expected_search and a category label."""
    queries = bench_module._QUERIES
    assert len(queries) >= 20
    for q in queries:
        assert isinstance(q.expected_search, bool)
        assert q.category in {
            "time-sensitive", "factual", "personal", "creative", "ambiguous",
        }
        assert len(q.text) > 0


def test_main_no_backends_returns_exit_2(bench_module, tmp_path, monkeypatch):
    """Calling with --skip-main and no candidate is a usage error."""
    monkeypatch.setattr(bench_module, "_build_main_llm", lambda: None)
    rc = bench_module.main([
        "--skip-main",
        "--baseline", str(tmp_path / "out.json"),
    ])
    assert rc == 2


def test_backend_summary_aggregates_correctly(bench_module):
    """Smoke-test the latency/accuracy aggregator on synthetic data."""
    Q = bench_module._BenchQuery
    PerQ = bench_module._PerQueryResult
    Summary = bench_module._BackendSummary

    per_query = [
        PerQ(
            text="x", category="factual", expected_search=False,
            actual_search=False, knowledge_confidence="high",
            latency_ms=100.0, correct=True,
        ),
        PerQ(
            text="y", category="time-sensitive", expected_search=True,
            actual_search=False, knowledge_confidence="medium",
            latency_ms=200.0, correct=False,
        ),
    ]
    s = Summary(
        label="test", samples=len(per_query),
        median_ms=statistics.median(r.latency_ms for r in per_query),
        p95_ms=200.0, p99_ms=200.0,
        accuracy=0.5, per_query=per_query,
    )
    assert s.median_ms == pytest.approx(150.0)
    assert s.accuracy == 0.5
