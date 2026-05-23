# Test sweep binding rules

> **Status: BINDING.** Every new test added under `tests/` must satisfy
> every rule below. Existing tests that violate a rule are grandfathered
> only because they currently pass; they MUST be fixed when touched.
>
> Live since 2026-05-22 after the catalog-pass session demonstrated
> that lax test hygiene was the recurring cause of "the sweep keeps
> getting killed" frustration. Each rule below is an action a real
> bug took during that session.

## Why rules at all

The full sweep runs **~4600+ tests in 75–90 seconds** as the binding
baseline. That budget is non-negotiable: a slower sweep gets skipped,
a skipped sweep means regressions land. The rules below preserve that
budget AND keep individual tests from contaminating each other.

The runner ([`scripts/run_tests.py`](../scripts/run_tests.py)) defends
against operator-side mistakes (concurrent runs, orphan processes,
hung subprocesses) via five independent safeguards. These rules
defend against TEST-WRITER mistakes — things the runner can't catch
because they look like normal pytest code.

---

## Rules tests MUST follow

### R1 — No raw class / module-level mutation

A test that needs to alter a global or class attribute MUST use
`monkeypatch.setattr(target, "attr", value)`. The monkeypatch fixture
restores the original at teardown.

**Forbidden:**

```python
def test_chain_skips_unconstructable_reader(monkeypatch):
    # monkeypatch accepted but never used — this is the bug
    rc_module.ReaderChain._READER_FACTORIES = {  # ← raw assignment
        ...
    }
```

**Required:**

```python
def test_chain_skips_unconstructable_reader(monkeypatch):
    monkeypatch.setattr(
        rc_module.ReaderChain, "_READER_FACTORIES", {...},
    )
```

This was the actual bug fixed during the sweep-durability pass. The
raw assignment permanently mutated class state and broke every
downstream test that introspected the original factory set.

### R2 — Threads MUST be cleaned up

Tests that spawn threads (`threading.Thread`, watcher classes,
`WaitingSpinner`, `CacheWarmer`, `AICommentWatcher`, ...) MUST call
`.stop()` / `.join()` in a `try`/`finally` or fixture teardown.

**Forbidden:**

```python
def test_warmer_fires():
    w = CacheWarmer(send, interval_seconds=0.05)
    w.start()
    time.sleep(0.2)
    assert ... # ← leaks the daemon thread
```

**Required:**

```python
def test_warmer_fires():
    w = CacheWarmer(send, interval_seconds=0.05)
    w.start()
    try:
        time.sleep(0.2)
        assert ...
    finally:
        w.stop(timeout=1)
```

Threads alive at session end are flagged by
`tests/conftest.py::pytest_sessionfinish` and printed as warnings.

### R3 — Subprocesses MUST be reaped

Tests that `subprocess.Popen` or `multiprocessing.Process` MUST wait
for completion or kill explicitly in teardown.

The conftest's `_kill_test_descendants` is a session-end safety net,
NOT a license to leak. Reap your own children inside the test.

### R4 — No real network calls

Tests MUST NOT make HTTP/TCP calls to real external services. Use
`monkeypatch` stubs or a mocking library. Real network failures
cascade into flaky sweeps and slow the budget.

A test that needs to verify "fail-open when the server is down"
should mock the client at the transport layer (`requests.Session.get`,
`httpx.Client.send`, etc.) — not actually contact a server.

### R5 — No real-port binding

Tests MUST NOT `bind` to fixed ports. Use `socket.socket(...).bind(("",
0))` + `getsockname()` for dynamic ports, or mock the network layer
entirely.

The orchestrator binds to MCP port `19761` in production. A test
that binds it would block the live orchestrator (and vice versa).
The conftest's port-aware cleanup hooks PRESERVE the live
orchestrator process when reaping — they don't protect against
binding conflicts during the test itself.

### R6 — Per-test deadline

All tests run under `--timeout=30 --timeout-method=thread` (see
`pyproject.toml::tool.pytest.ini_options.addopts`). Tests that
legitimately need longer MUST decorate explicitly:

```python
@pytest.mark.timeout(120)
def test_actually_needs_two_minutes():
    """Loads a 600 MB model from disk; budget is 120 s."""
```

The comment explaining WHY the longer timeout is required. Tests
without a comment are reverted on review.

If the test needs >300 s, mark it `@pytest.mark.slow` so `--fast`
mode can skip it. The sweep budget (90 s total) wins over any single
test's preference.

### R7 — Order-independent

Each test MUST pass when run in isolation AND in the full sweep.
Order-dependent tests are pollution bugs IN THE POLLUTER, not
fragility of the dependent.

If you find a test that only passes in one order, the fix is to find
the polluter and apply R1 (`monkeypatch.setattr`). Adding a
`pytest.fixture(autouse=True)` reset to the dependent test is a
work-around, not a fix.

### R8 — Per-module test file

Every new module under `src/ultron/` MUST have a corresponding test
file under `tests/` mirroring the path. Aim for both:

* **Unit tests** — mock the seam, verify behaviour at the boundary.
* **At least one integration test** — exercise the real code path
  with realistic inputs.

This is already in [`MEMORY.md`](C:\Users\alecf\.claude\projects\C--STC-ultronPrototype\memory\MEMORY.md)
"Documentation + testing standards" — restated here for completeness.

### R9 — `data/` writes go to `tmp_path`

Tests MUST NOT write to `data/` directly. Use the `tmp_path` fixture
for any disk state the test needs.

The conftest's heartbeat / progress files DO write under `data/` —
those are the wrapper's observability surface, not test outputs.
Test code writing under `data/` confuses the wrapper's "did the
sweep complete cleanly?" validation.

### R10 — Full-sweep budget: 90 s

The full sweep currently runs in **75–90 s** on the dev machine. New
tests add at most **100 ms each** to the budget. A batch adding
30 tests at 100 ms each = 3 s of new time; that's the upper bound
per batch.

If the budget creeps past 90 s, the next batch reviewer reverts the
slowest new test and asks the author to mock more heavily.

### R11 — No voice-stack loading

Tests MUST NOT load the live LLM / TTS / STT / RVC / wake-word
stack. Mock the voice components.

The voice-stack-concurrency rule (see
[`feedback_voice_stack_concurrency.md`](C:\Users\alecf\.claude\projects\C--STC-ultronPrototype\memory\feedback_voice_stack_concurrency.md))
applies to test code. A test that loads Kokoro or Parakeet pushes
the sweep past the 90 s budget AND fights the live orchestrator
for GPU memory.

### R12 — No bare `time.sleep` > 0.5 s

Tests that wait on async behaviour MUST use
`threading.Event.wait(timeout=N)` or `pytest.wait_for` with explicit
timeouts, not `time.sleep()` polling.

A `time.sleep(5)` in a test is 5 seconds of sweep budget burned even
when the condition was satisfied after 50 ms.

---

## What the runner enforces

[`scripts/run_tests.py`](../scripts/run_tests.py) applies five
independent defensive layers — these defend against OPERATOR-side
mistakes and runtime hangs:

1. **Pre-flight env check** — psutil importable, venv reachable,
   `data/` writable. Hard refuse if any fails (exit code 6).
2. **Cross-instance mutex** — only one `scripts/run_tests.py` may
   run at a time per checkout. `data/.run_tests.lock` is the file.
   Stale-lock recovery from crashes. `--wait` to wait politely.
3. **Orphan pytest kill** — any pytest-on-this-codebase older than
   5 minutes is killed unconditionally (no prompt). Catches the
   harness-leak pattern.
4. **Heartbeat watchdog** — separate thread polls
   `data/.run_tests_heartbeat`; stale > 90 s kills pytest (exit 3).
   Catches C-extension hangs that the per-test thread-timeout
   can't interrupt.
5. **Wall-clock deadline** — `--max-runtime` (default 600 s) kills
   pytest if the entire sweep exceeds it (exit 5).
6. **Post-run validation** — checks
   `data/.run_tests_progress.jsonl` for a `session_end` event. If
   missing despite a 0 exit code, reports exit 7 ("sweep was killed
   mid-stream, exit code is suspect").
7. **Aggressive post-run cleanup** — every Python descendant + every
   late-arriving pytest is terminated.

---

## Observability — how to tell what's happening

The wrapper publishes three files for operators / dashboards /
external observers:

| File | What it shows |
| --- | --- |
| `data/.run_tests_heartbeat` | mtime + timestamp updated before every test. `stat data/.run_tests_heartbeat` shows freshness. |
| `data/.run_tests_current` | Name of the currently-running test, or `(session_ended status=N)` on completion. |
| `data/.run_tests_progress.jsonl` | One JSON event per test start/outcome with duration + session_start / session_end markers. Truncated at each sweep start. |

Operators can:

```powershell
# What's the sweep on right now?
cat data/.run_tests_current

# How long has the heartbeat been stale?
stat data/.run_tests_heartbeat

# Real-time event stream
Get-Content data/.run_tests_progress.jsonl -Tail 20 -Wait
```

even when the sweep was launched in a background pipe that buffers
stdout.

---

## Maintenance contract

* Any test that violates a rule is **reverted** on the PR that
  introduced it.
* Any new module without a paired test file is **reverted**.
* The sweep MUST stay green AND under the 90 s budget on the
  validating SHA before any new commit lands.
* `scripts/run_tests.py` is the single entry point. `python -m
  pytest tests/` directly is **forbidden** in scripts, docs, and
  shipped commands — it bypasses the five safeguards.

The memory file
[`feedback_test_sweep_workflow.md`](C:\Users\alecf\.claude\projects\C--STC-ultronPrototype\memory\feedback_test_sweep_workflow.md)
captures the operator-side workflow rules. This document captures
the test-WRITER rules. Both are binding.
