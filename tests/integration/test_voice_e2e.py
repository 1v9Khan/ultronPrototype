"""Full voice-pipeline end-to-end suite (GPU-gated).

Production-hardening finding #10: a unified, run-all-at-once e2e suite that
drives REAL input through the REAL stack -- Kokoro synthesizes a user
utterance, Moonshine/Parakeet transcribes it, the routing classifier + LLM
produce a response, Kokoro synthesizes the reply, and the web-search / memory /
gating layers run live -- then ASSERTS the resulting behavior.

The heavy lifting already lives in ``scripts/autonomous_e2e_harness.py`` (a
maintained script that builds the real engines per phase and records a
``Scenario`` object -- ``scenario.errors`` is empty iff that scenario behaved
as expected). Finding #10's gap was that the harness is a *script*, not a
pytest suite with assertions that CI / the test runner can gate on. This module
closes that gap: it loads the harness, runs each phase, and fails the
corresponding pytest if any scenario recorded an error.

GATED on ``PYTEST_RUN_GPU_TESTS=1`` (the phases load the full model stack --
minutes + GPU VRAM), so the normal ~90 s sweep simply SKIPS this file and is
never destabilized by it. The harness is imported lazily inside a fixture, so
collection (and the skip) never touches the heavy modules.

Run the whole suite::

    PYTEST_RUN_GPU_TESTS=1 python scripts/run_tests.py -- tests/integration/test_voice_e2e.py

or a single phase::

    PYTEST_RUN_GPU_TESTS=1 python -m pytest tests/integration/test_voice_e2e.py -k routing

or the underlying script directly (aggregate pass/fail via exit code)::

    python scripts/autonomous_e2e_harness.py --phase all
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("PYTEST_RUN_GPU_TESTS") != "1",
        reason="set PYTEST_RUN_GPU_TESTS=1 to load the full voice pipeline",
    ),
]

# scripts/autonomous_e2e_harness.py, resolved relative to this test file
# (tests/integration/ -> repo root -> scripts/).
_HARNESS_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "autonomous_e2e_harness.py"
)

# Every phase exposed by the harness, in run order. Each maps to a
# ``phase_<name>()`` function returning ``List[Scenario]``.
_PHASES = ("stt", "llm", "tts", "web_search", "memory", "routing", "gate")


def _load_harness() -> Any:
    """Import the harness module by path (scripts/ is not a package).

    Lazy: only called from the (GPU-gated) fixture, so collection + the
    skip path never load the heavy modules the harness pulls in.
    """
    spec = importlib.util.spec_from_file_location(
        "ultron_autonomous_e2e_harness", _HARNESS_PATH,
    )
    if spec is None or spec.loader is None:        # pragma: no cover - defensive
        raise ImportError(f"cannot load harness from {_HARNESS_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def harness() -> Any:
    assert _HARNESS_PATH.is_file(), f"e2e harness not found: {_HARNESS_PATH}"
    return _load_harness()


@pytest.fixture(autouse=True)
def _free_gpu_between_phases():
    """Release each phase's GPU memory after it runs. The phases construct their
    own engines (LLM / TTS / STT / embedder); without freeing between them the
    full suite accumulates VRAM past the budget and later phases fail to load
    (every phase passes in isolation). Fail-open: no torch / no CUDA -> no-op."""
    yield
    try:
        import gc

        gc.collect()
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _assert_no_scenario_errors(scenarios: Any) -> None:
    assert scenarios, "phase produced no scenarios"
    failed = [s for s in scenarios if getattr(s, "errors", None)]
    if failed:
        lines = [f"  - {s.name}: {'; '.join(s.errors)}" for s in failed]
        pytest.fail(
            f"{len(failed)}/{len(scenarios)} e2e scenarios failed:\n"
            + "\n".join(lines)
        )


@pytest.mark.parametrize("phase_name", _PHASES)
def test_e2e_phase(harness: Any, phase_name: str) -> None:
    """Run one full-pipeline phase end-to-end; assert every scenario passed.

    Each phase drives real input through the real stack and records a
    ``Scenario`` per case. We fail the pytest (with the offending scenario
    names + error messages) if any scenario recorded an error.
    """
    fn = getattr(harness, f"phase_{phase_name}", None)
    assert fn is not None, f"harness has no phase_{phase_name}()"
    scenarios = fn()
    _assert_no_scenario_errors(scenarios)


def test_harness_exposes_all_phases(harness: Any) -> None:
    """Guard against a harness refactor silently dropping a phase from the
    pytest surface (the parametrize list above must stay in sync)."""
    missing = [p for p in _PHASES if not hasattr(harness, f"phase_{p}")]
    assert not missing, f"harness missing phase functions: {missing}"
