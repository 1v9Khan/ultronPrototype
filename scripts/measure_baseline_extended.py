"""Foundation phase extended baseline measurement (Option B).

Captures the metrics that ``scripts/measure_baseline.py`` doesn't already cover:

  - VRAM during a search-triggering query (with mocked Brave + Jina; real local LLM)
  - VRAM during an active coding session (real voice stack + scripted mock bridge)
  - Time-to-acknowledgment for search queries (CPU-only; ``AcknowledgmentSource``)
  - Coding orchestration scenario timing (10 mocked scenarios via pytest)
  - First-token-from-end-of-user-speech (composite: Whisper p50 + LLM TTFT p50)

Writes results into ``baselines.json`` under
``phase_foundation_start.measurements_extended``. Does NOT overwrite
the voice-path baseline already captured under that key.

Modes (CLI flags):

  ``--lite``  Run only the CPU-only metrics (TTA, scenario timing, composite TTFA).
              Fast (~30 s); doesn't touch the GPU; safe to run while other work is
              happening.

  ``--full``  Also load the voice stack and capture VRAM during a mocked search hop
              and a mocked coding session. Slow (several minutes); locks the GPU.

  ``--all``   ``--lite`` + ``--full``. Default.

Run from anywhere; imports resolve from MAIN_REPO_PATH (the main checkout).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup: import code + load models from the main checkout, regardless of
# where this script lives or where it's run from.
#
# Main repo's .env holds ULTRON_LLM_MODEL_PATH=models/... (relative). That path
# only resolves correctly when cwd == main repo, since python-dotenv's
# load_dotenv() walks up from cwd to find .env. We therefore chdir to the main
# repo BEFORE importing config.settings (which calls load_dotenv()). Output
# still goes to the worktree root via the absolute path captured below.
# ---------------------------------------------------------------------------
MAIN_REPO_PATH = Path(r"C:\STC\ultronPrototype")
WORKTREE_ROOT = Path(__file__).resolve().parent.parent  # absolute, so chdir is safe
OUTPUT_PATH = WORKTREE_ROOT / "baselines.json"

import os as _os
_os.chdir(str(MAIN_REPO_PATH))

sys.path.insert(0, str(MAIN_REPO_PATH / "src"))
sys.path.insert(0, str(MAIN_REPO_PATH))

# Reconfigure stdout for safe Unicode on Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def vram_used_mb() -> int:
    """Total VRAM currently used by all processes on GPU 0, in MB."""
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        text=True,
    ).strip()
    return int(out)


def load_existing_baselines() -> Dict[str, Any]:
    with OUTPUT_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_baselines(data: Dict[str, Any]) -> None:
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Measurement 1: time-to-acknowledgment (CPU-only)
# ---------------------------------------------------------------------------


def measure_time_to_acknowledgment() -> Dict[str, Any]:
    """Microbenchmark ``AcknowledgmentSource.next_phrase()``.

    The 200 ms target in Part 0.1 is end-to-end "search-decision-made -> phrase
    on its way to TTS". This measurement isolates the phrase-selection cost,
    which is the only piece in the orchestrator's hot path that we own; the
    rest is TTS synthesis (already captured in baselines.tts_synth_ms) and
    audio device latency (system-level, not in our control).
    """
    print("\n[lite] Measuring time-to-acknowledgment (phrase selection)...")
    from ultron.web_search.acknowledgments import AcknowledgmentSource

    src = AcknowledgmentSource()
    # Burn 8 to clear the first cycle and reach steady-state shuffle behavior.
    for _ in range(8):
        src.next_phrase()

    samples_us: List[float] = []
    for _ in range(1000):
        t0 = time.perf_counter()
        src.next_phrase()
        samples_us.append((time.perf_counter() - t0) * 1_000_000)

    out = {
        "n_samples": len(samples_us),
        "min_us": min(samples_us),
        "median_us": statistics.median(samples_us),
        "p95_us": _percentile(samples_us, 0.95),
        "p99_us": _percentile(samples_us, 0.99),
        "max_us": max(samples_us),
        "mean_us": statistics.mean(samples_us),
        "notes": (
            "Microbenchmark of AcknowledgmentSource.next_phrase() in isolation. "
            "End-to-end TTA also includes TTS synth (see "
            "phase_foundation_start.latency_ms.aggregate.tts_synth_ms; ~219-937 ms) "
            "and audio device output latency. The 200 ms Part-0.1 target is for "
            "the phrase being EMITTED to the TTS pipeline, not audible to user; "
            "phrase-selection is several orders of magnitude under the budget."
        ),
    }
    print(
        f"  TTA phrase-pick: median={out['median_us']:.1f} us  "
        f"p95={out['p95_us']:.1f} us  p99={out['p99_us']:.1f} us"
    )
    return out


def _percentile(samples: List[float], p: float) -> float:
    """Simple percentile (no numpy needed)."""
    s = sorted(samples)
    if not s:
        return 0.0
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


# ---------------------------------------------------------------------------
# Measurement 2: first-token-from-end-of-speech (composite from existing data)
# ---------------------------------------------------------------------------


def compute_first_token_from_end_of_speech(existing: Dict[str, Any]) -> Dict[str, Any]:
    """Composite estimate from already-captured baseline pieces.

    Real measurement requires the live VAD/mic path, which isn't available
    headlessly. The composite is: STT p50 (transcribe-then-flush) + LLM TTFT
    (post-prompt-submission to first-emitted-token).

    Reads the modern ``stt_2_5s_sample`` key first; falls back to the
    legacy ``whisper_2_5s_sample`` for older baseline files.
    """
    voice = existing.get("phase_foundation_start", existing)
    lat = voice.get("latency_ms", {})
    stt_block = lat.get("stt_2_5s_sample") or lat.get("whisper_2_5s_sample") or {}
    agg = lat.get("aggregate", {}).get("ttft_ms", {})

    stt_med = stt_block.get("median")
    ttft_med = agg.get("median")
    stt_min = stt_block.get("min")
    ttft_min = agg.get("min")
    stt_max = stt_block.get("max")
    ttft_max = agg.get("max")
    stt_engine = stt_block.get("engine", "whisper")

    if stt_med is None or ttft_med is None:
        return {
            "available": False,
            "reason": "latency_ms missing required pieces (stt or aggregate ttft)",
        }

    return {
        "available": True,
        "computed_as": f"{stt_engine} 2.5s sample + LLM TTFT (per-query aggregate)",
        "median_ms": stt_med + ttft_med,
        "min_ms": (stt_min or 0) + (ttft_min or 0),
        "max_ms": (stt_max or 0) + (ttft_max or 0),
        "components": {
            "stt_engine": stt_engine,
            "stt_p50_ms": stt_med,
            "llm_ttft_p50_ms": ttft_med,
        },
        "notes": (
            "Composite estimate only. True end-of-speech-to-first-token needs "
            "VAD + STT + pre-flight + LLM streaming through the live "
            "orchestrator with mic input; this script runs headless. Composite "
            "ignores VAD silence-detection delay (vad.min_silence_duration_ms) "
            "and pre-flight latency (~50-200 ms when search-triggering). "
            "The number is therefore a LOWER BOUND for real-world TTFA."
        ),
    }


# ---------------------------------------------------------------------------
# Measurement 3: coding orchestration scenario timing (CPU-only, via pytest)
# ---------------------------------------------------------------------------


def measure_scenario_timing() -> Dict[str, Any]:
    """Run the 10 orchestration scenarios via pytest --durations=0 and parse.

    The test file is ``tests/coding/test_orchestration.py``. All scenarios
    use the scripted mock bridge so no Claude tokens burn.
    """
    print("\n[lite] Measuring coding orchestration scenario timing...")
    target = MAIN_REPO_PATH / "tests" / "coding" / "test_orchestration.py"
    cmd = [
        sys.executable, "-m", "pytest", str(target),
        "--durations=0", "-q", "--tb=no", "-p", "no:cacheprovider",
    ]
    print(f"  cmd: {' '.join(cmd)}")
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(MAIN_REPO_PATH),
    )
    elapsed_s = time.monotonic() - t0
    out = proc.stdout + proc.stderr

    # Parse pytest's durations block. Looks like:
    #   ===== slowest durations =====
    #   1.23s call     tests/coding/test_orchestration.py::test_scenario_X_...
    durations: List[Dict[str, Any]] = []
    for line in out.splitlines():
        m = re.match(
            r"\s*([\d.]+)s\s+(call|setup|teardown)\s+"
            r"(tests/coding/test_orchestration\.py::test_scenario_[\w]+)",
            line,
        )
        if m:
            durations.append({
                "duration_s": float(m.group(1)),
                "phase": m.group(2),
                "test": m.group(3),
            })

    # Group by test, sum across phases. Pytest reports setup/call/teardown
    # separately when --durations is set; total time is the sum.
    by_test: Dict[str, float] = {}
    for d in durations:
        by_test[d["test"]] = by_test.get(d["test"], 0.0) + d["duration_s"]

    # Pull pass/fail summary line.
    summary_line = ""
    for line in reversed(out.splitlines()):
        if " passed" in line or " failed" in line:
            summary_line = line.strip()
            break

    result = {
        "command": " ".join(cmd),
        "wall_clock_s": round(elapsed_s, 2),
        "pytest_summary": summary_line,
        "exit_code": proc.returncode,
        "scenarios": [
            {"test": k, "total_s": round(v, 3)}
            for k, v in sorted(by_test.items())
        ],
        "scenario_count": len(by_test),
        "raw_durations": durations,
        "notes": (
            "Times are pytest's setup+call+teardown summed per test. Each "
            "scenario uses the scripted mock bridge (no Claude API). "
            "Variance run-to-run is dominated by Qdrant-embedded init "
            "(loaded once per scenario via UltronMCPServer constructor)."
        ),
    }
    print(
        f"  ran {len(by_test)} scenarios in {elapsed_s:.1f}s wall  "
        f"-- {summary_line}"
    )
    if by_test:
        slowest = max(by_test.items(), key=lambda kv: kv[1])
        print(f"  slowest: {slowest[0].split('::')[-1]} -> {slowest[1]:.2f}s")
    return result


# ---------------------------------------------------------------------------
# Measurement 4: search-triggering query VRAM (loads voice stack)
# ---------------------------------------------------------------------------


def _make_mock_brave():
    """Subclass that bypasses BraveSearchClient.__init__ (which requires API key)
    and returns canned BraveResult rows. Pure CPU; no network."""
    from ultron.web_search.brave import BraveResult, BraveSearchClient

    class _MockBrave(BraveSearchClient):
        def __init__(self):  # type: ignore[override]
            # Skip parent __init__ — no API key needed.
            self.api_key = "mock"
            self.endpoint = "mock://"
            self.rate_limit_s = 0.0
            self.timeout_s = 0.0
            self._last_call = 0.0
            import threading as _t
            self._lock = _t.Lock()
            self._fixture: List[BraveResult] = [
                BraveResult(
                    url="https://example.com/python-3.13-features",
                    title="Python 3.13: What's New",
                    snippet="Python 3.13 introduces several new features including improved error messages and a new REPL.",
                    rank=0,
                ),
                BraveResult(
                    url="https://docs.python.org/3.13/whatsnew/",
                    title="What's New in Python 3.13 - Official Docs",
                    snippet="The official changelog for Python 3.13: free-threading mode, JIT, mobile platform support.",
                    rank=1,
                ),
                BraveResult(
                    url="https://realpython.com/python-3-13/",
                    title="Python 3.13 Highlights",
                    snippet="A practical tour of Python 3.13's headline changes.",
                    rank=2,
                ),
                BraveResult(
                    url="https://example.org/python-version-history",
                    title="History of Python Versions",
                    snippet="Brief overview of Python versions from 1.0 to today.",
                    rank=3,
                ),
                BraveResult(
                    url="https://example.net/why-upgrade-python",
                    title="Why Upgrade Python",
                    snippet="Reasons to keep your Python interpreter current.",
                    rank=4,
                ),
            ]

        def search(self, query, count=5):  # type: ignore[override]
            return list(self._fixture[:count])

    return _MockBrave()


def _make_mock_jina():
    """Subclass that returns canned markdown without HTTP."""
    from ultron.web_search.jina import JinaReaderClient

    class _MockJina(JinaReaderClient):
        def __init__(self):  # type: ignore[override]
            self.endpoint = "mock://"
            self.timeout_s = 0.0
            self.max_bytes = 200_000

        def fetch(self, url):  # type: ignore[override]
            return (
                f"# Page from {url}\n\n"
                "This is a mocked Jina response used to measure local-side "
                "VRAM during the search hop without hitting the network. "
                "It contains enough text that the LLM-augmented response "
                "step has substantive context to feed into prompt assembly."
            )

    return _MockJina()


def measure_search_vram(llm) -> Dict[str, Any]:
    """Run a search-triggering query through the executor (mocked Brave/Jina)
    and capture VRAM at each phase.

    Uses the already-loaded LLM. Doesn't write to the cache (cache=None) so
    every run hits the full path.
    """
    print("\n[full] Measuring VRAM during search-triggering query (mocked Brave + Jina)...")
    from ultron.web_search.search import WebSearchExecutor

    brave = _make_mock_brave()
    jina = _make_mock_jina()
    executor = WebSearchExecutor(brave=brave, jina=jina, llm=llm, cache=None, max_fetch=3)

    user_query = "What are the latest features in Python 3.13?"
    pre_vram = vram_used_mb()
    t0 = time.monotonic()
    payload = executor.run(user_query, search_queries=[user_query], top_n=3)
    elapsed_ms = (time.monotonic() - t0) * 1000
    post_vram = vram_used_mb()

    return {
        "user_query": user_query,
        "vram_mb_before_search": pre_vram,
        "vram_mb_after_search": post_vram,
        "vram_mb_delta": post_vram - pre_vram,
        "executor_elapsed_ms": elapsed_ms,
        "executor_internal_elapsed_ms": payload.elapsed_ms,
        "sources_returned": len(payload.sources),
        "cache_hit": payload.cache_hit,
        "notes": (
            "Brave and Jina are mocked (no network). VRAM cost of the search "
            "hop is the LLM ranking call (one short prompt, ~128 max_tokens). "
            "The orchestrator would also issue a final-response LLM call "
            "afterward (not measured here -- already captured in voice-path "
            "per-query VRAM)."
        ),
    }


# ---------------------------------------------------------------------------
# Measurement 5: coding session VRAM (voice stack + scripted mock bridge)
# ---------------------------------------------------------------------------


def measure_coding_session_vram() -> Dict[str, Any]:
    """Run a scripted coding scenario while the voice stack is loaded.

    The scripted mock bridge runs in a worker thread; the LLM stays loaded
    in VRAM. We capture peak VRAM during the scripted scenario. The bridge
    doesn't add GPU load directly -- this measurement confirms that an
    active coding session doesn't trigger any unexpected secondary GPU
    load (e.g. accidental reloads, KV cache growth, etc.).
    """
    print("\n[full] Measuring VRAM during active coding session (scripted mock bridge)...")
    import os as _os
    import tempfile
    # The MCP server enforces that project_root lives under settings.CODING_SANDBOX_PATH
    # in production. Tests use this escape hatch to point at tmp_path. We do the
    # same here -- the measurement is about VRAM behavior, not sandbox safety.
    _os.environ["ULTRON_CODING_MCP_ALLOW_ANY_ROOT"] = "1"
    from ultron.coding import (
        CodingTaskRunner, CodingVoiceController, ProjectRegistry,
        ProjectResolver, StatusNarrator, UltronMCPServer,
    )
    from ultron.coding.bridge import TaskRequest
    from ultron.coding.coordinator import ConversationCoordinator
    from ultron.coding.session import SessionStatus
    from ultron.coding.verification import Verifier
    sys.path.insert(0, str(MAIN_REPO_PATH))
    from tests.coding.mock_bridge import ClaudeScript, ScriptedClaudeBridge

    # Stub LLM for orchestration -- the real LLM stays loaded for the search
    # measurement; this is a stub so coordinator tests don't accidentally
    # block on a real model.
    class _StubLLM:
        def generate(self, prompt: str) -> str:
            return "Use your default approach."

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_path = Path(tmp_str)
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        server = UltronMCPServer(host="127.0.0.1", port=0)
        verifier = Verifier(store=server.store)
        coordinator = ConversationCoordinator(
            store=server.store, llm=_StubLLM(), verifier=verifier,
        )
        server.set_clarification_responder(coordinator.decide_clarification)
        server.set_declare_complete_handler(coordinator.handle_declare_complete)
        narrator = StatusNarrator(llm=None)

        # Placeholder bridge so the runner can construct.
        placeholder = ScriptedClaudeBridge(
            server, ClaudeScript(), session_id="__unset__",
        )
        runner = CodingTaskRunner(
            bridge=placeholder, log_path=tmp_path / "audit.jsonl",
            narrator=narrator, store=server.store,
        )
        registry = ProjectRegistry(path=tmp_path / "projects.json")
        resolver = ProjectResolver(registry, embedder=None)
        _voice = CodingVoiceController(
            runner=runner, registry=registry, resolver=resolver,
            sandbox_root=sandbox, coordinator=coordinator,
        )

        # Start a session and run a small successful script.
        project = sandbox / "demo"
        project.mkdir()
        session = server.create_session(
            project_root=project, initial_prompt="demo script", mode="new",
        )
        server.store.transition(session.session_id, SessionStatus.EXECUTING)

        script = (
            ClaudeScript()
            .progress("scaffolding", "set up", ["pyproject.toml"])
            .write_file("pyproject.toml", '[project]\nname = "demo"\nversion = "0.1.0"\n')
            .progress("implementing", "wrote main.py", ["main.py"])
            .write_file("main.py", "def hello():\n    return 'hi'\n\nif __name__ == '__main__':\n    print(hello())\n")
            .progress("tests", "wrote test", ["test_main.py"])
            .write_file(
                "test_main.py",
                "from main import hello\n\ndef test_hello():\n    assert hello() == 'hi'\n",
            )
            .test_results(passing=1, failing=0)
            .declare_complete(
                summary="demo done", entry_point="main.py",
                files_created=["pyproject.toml", "main.py", "test_main.py"],
            )
        )
        bridge = ScriptedClaudeBridge(
            server, script, session_id=session.session_id,
        )

        pre_vram = vram_used_mb()
        peak_vram = pre_vram
        handle = bridge.submit(TaskRequest(
            task_prompt="demo", cwd=project, model="haiku",
            timeout_s=30.0, label="vram-measurement",
        ))

        # Poll VRAM while the script runs.
        samples = []
        end_time = time.monotonic() + 30.0
        while handle.is_running() and time.monotonic() < end_time:
            v = vram_used_mb()
            samples.append(v)
            if v > peak_vram:
                peak_vram = v
            time.sleep(0.05)
        result = handle.wait(timeout=5.0)
        post_vram = vram_used_mb()

        return {
            "vram_mb_before_session": pre_vram,
            "vram_mb_peak_during_session": peak_vram,
            "vram_mb_after_session": post_vram,
            "vram_mb_delta_peak": peak_vram - pre_vram,
            "samples_taken": len(samples),
            "task_succeeded": bool(result and result.success),
            "notes": (
                "Voice stack remains loaded throughout. The scripted mock "
                "bridge runs in a worker thread (no Claude subprocess). "
                "Real AI coding agent runs as a SUBPROCESS and consumes negligible "
                "local GPU; the LLM hosting the answer is at Anthropic, not "
                "local. So this measurement validates that the orchestration "
                "machinery itself doesn't spike local VRAM."
            ),
        }


# ---------------------------------------------------------------------------
# Voice stack loader for --full mode
# ---------------------------------------------------------------------------


def _load_voice_stack():
    """Load the production STT + LLM + TTS stack via the canonical
    factories. Returns ``(stt, llm, tts, rvc, checkpoints)`` where ``rvc``
    is None for non-piper_rvc engines. 2026-05-22: swapped the hard-
    coded Whisper + RVC + Piper trio for the production factories so
    this measurement always reflects whichever engines config.yaml
    currently selects.
    """
    print("\n[full] Loading voice stack...")
    import os as _os
    _os.environ["ULTRON_LOG_LEVEL"] = "WARNING"
    from ultron.utils.logging import configure_logging
    configure_logging(level="WARNING")

    checkpoints: Dict[str, int] = {}
    checkpoints["before_load_mb"] = vram_used_mb()
    print(f"  before load: {checkpoints['before_load_mb']} MB")

    from ultron.transcription import make_stt_engine
    t = time.monotonic()
    stt = make_stt_engine()
    print(
        f"  STT loaded in {time.monotonic() - t:.1f}s "
        f"({type(stt).__name__})"
    )
    checkpoints["after_stt_mb"] = vram_used_mb()

    from ultron.llm import LLMEngine
    t = time.monotonic()
    llm = LLMEngine(memory=None)
    print(f"  LLM loaded in {time.monotonic() - t:.1f}s")
    checkpoints["after_llm_mb"] = vram_used_mb()

    from ultron.tts import make_tts_engine
    t = time.monotonic()
    rvc, tts = make_tts_engine()
    print(
        f"  TTS loaded in {time.monotonic() - t:.1f}s "
        f"({type(tts).__name__})"
    )
    if hasattr(tts, "warmup"):
        tts.warmup()
    checkpoints["full_stack_loaded_mb"] = vram_used_mb()
    print(f"  full stack loaded: {checkpoints['full_stack_loaded_mb']} MB")

    # Warm the LLM so the first generate doesn't pay cold-cache cost.
    # Use enable_thinking=False to match the voice path's default.
    print("  warming LLM (first stream cancelled at first token)...")
    warm_stream = llm.generate_stream(
        "Say 'ready' and nothing else.",
        enable_thinking=False,
    )
    for _tok in warm_stream:
        llm.cancel()
        break
    for _ in warm_stream:
        pass

    return stt, llm, tts, rvc, checkpoints


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Foundation phase extended baseline measurement",
    )
    parser.add_argument(
        "--lite", action="store_true",
        help="CPU-only metrics: TTA, scenario timing, composite TTFA.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Also load voice stack to measure search/coding session VRAM.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="--lite + --full (default).",
    )
    args = parser.parse_args(argv)

    if not (args.lite or args.full or args.all):
        args.all = True
    do_lite = args.lite or args.all
    do_full = args.full or args.all

    print("=" * 60)
    print("Foundation phase extended baseline (Option B)")
    print("=" * 60)
    print(f"  worktree: {WORKTREE_ROOT}")
    print(f"  main repo: {MAIN_REPO_PATH}")
    print(f"  modes: lite={do_lite} full={do_full}")
    print(f"  output: {OUTPUT_PATH}")

    existing = load_existing_baselines()
    if "phase_foundation_start" not in existing:
        print("ERROR: phase_foundation_start key missing from baselines.json. "
              "Run measure_baseline.py first or add the block manually.")
        return 1

    # Merge into any prior measurements_extended block so a --lite run after
    # a --full run (or vice versa) doesn't wipe out the other half.
    extended: Dict[str, Any] = dict(
        existing["phase_foundation_start"].get("measurements_extended", {})
    )
    extended["captured_at"] = datetime.now(timezone.utc).isoformat()
    extended["modes_run"] = {"lite": do_lite, "full": do_full}

    # ----- Lite -----
    if do_lite:
        extended["time_to_acknowledgment"] = measure_time_to_acknowledgment()
        extended["first_token_from_end_of_speech"] = (
            compute_first_token_from_end_of_speech(existing)
        )
        extended["coding_orchestration_scenarios"] = measure_scenario_timing()

    # ----- Full -----
    if do_full:
        stt, llm, tts, rvc, checkpoints = _load_voice_stack()
        extended["voice_stack_load_checkpoints_mb"] = checkpoints
        try:
            extended["search_query_vram"] = measure_search_vram(llm)
        except Exception as e:
            extended["search_query_vram"] = {"error": str(e), "type": type(e).__name__}
            print(f"  search VRAM measurement failed: {e}")
        try:
            extended["coding_session_vram"] = measure_coding_session_vram()
        except Exception as e:
            extended["coding_session_vram"] = {"error": str(e), "type": type(e).__name__}
            print(f"  coding session VRAM measurement failed: {e}")
        # Cleanup
        try:
            rvc.close()
        except Exception:
            pass
        try:
            tts.stop()
        except Exception:
            pass

    # ----- Persist -----
    existing["phase_foundation_start"]["measurements_extended"] = extended
    save_baselines(existing)

    print("\n" + "=" * 60)
    print(f"Saved extended measurements to {OUTPUT_PATH}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
