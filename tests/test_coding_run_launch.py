"""Tests for the B3 run/launch-on-command feature.

Covers ``src/ultron/coding/sandbox_runner.py`` (matcher, entry-point
resolution, sandbox-confined run/launch with injected process primitives,
voice summary) and the ``CapabilityVoiceController.maybe_handle_run_program``
handler (non-blocking background run + instant launch + fall-through on an
unresolved hint). Fully hermetic: no real subprocess is ever spawned.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from ultron.coding import sandbox_runner as sr
from ultron.coding.bridge import CodingBridge, TaskRequest
from ultron.coding.projects import Project, ProjectRegistry, ProjectResolver
from ultron.coding.runner import CodingTaskRunner
from ultron.coding.voice import CodingVoiceController


# --------------------------------------------------------------------------
# Matcher
# --------------------------------------------------------------------------


def test_match_run_basic():
    m = sr.match_run_program("run the calculator")
    assert m is not None and m.mode == "run" and m.project_hint == "calculator"


def test_match_launch_basic():
    m = sr.match_run_program("launch the server")
    assert m is not None and m.mode == "launch" and m.project_hint == "server"


def test_match_run_it_empty_hint():
    m = sr.match_run_program("run it")
    assert m is not None and m.mode == "run" and m.project_hint == ""


def test_match_strips_trailing_noun_and_determiner():
    m = sr.match_run_program("run my todo app")
    assert m is not None and m.mode == "run" and m.project_hint == "todo"


def test_match_start_up_is_launch():
    m = sr.match_run_program("start up the dashboard")
    assert m is not None and m.mode == "launch" and m.project_hint == "dashboard"


def test_match_ignores_non_commands():
    for t in [
        "what's the weather today",
        "tell me a joke",
        "how do I run a marathon",
        "the program crashed last night",
        "search for python tutorials",
        "",
    ]:
        assert sr.match_run_program(t) is None


# --------------------------------------------------------------------------
# Entry-point resolution
# --------------------------------------------------------------------------


def test_resolve_entry_point_main_py(tmp_path):
    (tmp_path / "main.py").write_text("print('hi')")
    ep = sr.resolve_entry_point(tmp_path)
    assert ep is not None and ep.display == "python main.py"
    assert ep.entry_path == tmp_path / "main.py"


def test_resolve_entry_point_package(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__main__.py").write_text("print('pkg')")
    ep = sr.resolve_entry_point(tmp_path)
    assert ep is not None and ep.display == "python -m mypkg"


def test_resolve_entry_point_none_when_empty(tmp_path):
    assert sr.resolve_entry_point(tmp_path) is None


# --------------------------------------------------------------------------
# run_program (injected run_fn -> no real subprocess)
# --------------------------------------------------------------------------


def _ok_run_fn(stdout="", stderr="", returncode=0):
    def _fn(argv, **kw):
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)
    return _fn


def test_run_program_success(tmp_path):
    sb = tmp_path / "sandbox"
    proj = sb / "calc"
    proj.mkdir(parents=True)
    (proj / "main.py").write_text("print('ok')")
    res = sr.run_program(proj, sandbox_root=sb, project_name="calc",
                         run_fn=_ok_run_fn(stdout="ok\n"))
    assert res.ok and res.returncode == 0 and "ok" in res.stdout
    assert res.mode == "run" and res.project_name == "calc"


def test_run_program_nonzero_exit(tmp_path):
    sb = tmp_path / "sandbox"
    proj = sb / "calc"
    proj.mkdir(parents=True)
    (proj / "main.py").write_text("x")
    res = sr.run_program(proj, sandbox_root=sb,
                         run_fn=_ok_run_fn(stderr="Traceback: boom", returncode=1))
    assert not res.ok and res.returncode == 1 and "boom" in res.stderr


def test_run_program_timeout(tmp_path):
    sb = tmp_path / "sandbox"
    proj = sb / "calc"
    proj.mkdir(parents=True)
    (proj / "main.py").write_text("x")

    def _timeout_fn(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw.get("timeout", 1))

    res = sr.run_program(proj, sandbox_root=sb, timeout_s=5, run_fn=_timeout_fn)
    assert not res.ok and res.timed_out


def test_run_program_refuses_outside_sandbox(tmp_path):
    sb = tmp_path / "sandbox"
    sb.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "main.py").write_text("x")
    called = {"n": 0}

    def _spy(argv, **kw):
        called["n"] += 1
        return subprocess.CompletedProcess(argv, 0, "", "")

    res = sr.run_program(outside, sandbox_root=sb, run_fn=_spy)
    assert not res.ok and "outside the sandbox" in (res.error or "")
    assert called["n"] == 0  # never executed


def test_run_program_no_entry_point(tmp_path):
    sb = tmp_path / "sandbox"
    proj = sb / "empty"
    proj.mkdir(parents=True)
    res = sr.run_program(proj, sandbox_root=sb, run_fn=_ok_run_fn())
    assert not res.ok and "entry point" in (res.error or "")


# --------------------------------------------------------------------------
# launch_program (injected spawn_fn)
# --------------------------------------------------------------------------


def test_launch_program_detached(tmp_path):
    sb = tmp_path / "sandbox"
    proj = sb / "srv"
    proj.mkdir(parents=True)
    (proj / "server.py").write_text("x")
    spawned = {}

    def _spawn(argv, **kw):
        spawned["argv"] = argv
        spawned["cwd"] = kw.get("cwd")
        return object()

    res = sr.launch_program(proj, sandbox_root=sb, project_name="srv", spawn_fn=_spawn)
    assert res.ok and res.launched and res.mode == "launch"
    assert spawned["argv"][-1].endswith("server.py")


def test_launch_program_refuses_outside_sandbox(tmp_path):
    sb = tmp_path / "sandbox"
    sb.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "main.py").write_text("x")
    res = sr.launch_program(outside, sandbox_root=sb, spawn_fn=lambda *a, **k: object())
    assert not res.ok and "outside the sandbox" in (res.error or "")


# --------------------------------------------------------------------------
# summarize_run_result
# --------------------------------------------------------------------------


def test_summarize_launch():
    r = sr.RunResult(ok=True, mode="launch", launched=True, project_name="srv")
    assert "Launched srv" in sr.summarize_run_result(r)


def test_summarize_run_with_output():
    r = sr.RunResult(ok=True, mode="run", returncode=0, stdout="answer is 42", project_name="calc")
    s = sr.summarize_run_result(r)
    assert "Ran calc" in s and "42" in s


def test_summarize_error():
    r = sr.RunResult(ok=False, mode="run", project_name="calc", error="it broke.")
    assert "it broke." in sr.summarize_run_result(r)


# --------------------------------------------------------------------------
# Voice handler: maybe_handle_run_program
# --------------------------------------------------------------------------


class _FakeBridge(CodingBridge):
    def submit(self, request: TaskRequest):
        raise NotImplementedError("not used in run/launch handler tests")

    def name(self) -> str:
        return "fake"


def _controller_with_project(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir(exist_ok=True)
    proj_dir = sandbox / "calculator"
    proj_dir.mkdir()
    (proj_dir / "main.py").write_text("print('42')")
    registry = ProjectRegistry(path=tmp_path / "projects.json")
    registry.add(Project(name="calculator", path=str(proj_dir)))
    resolver = ProjectResolver(registry, embedder=None)
    runner = CodingTaskRunner(bridge=_FakeBridge(), log_path=tmp_path / "log.jsonl")
    controller = CodingVoiceController(
        runner=runner, registry=registry, resolver=resolver, sandbox_root=sandbox,
    )
    return controller, proj_dir


def test_handler_returns_none_when_not_a_run_command(tmp_path):
    controller, _ = _controller_with_project(tmp_path)
    assert controller.maybe_handle_run_program("what's the weather") is None


def test_handler_falls_through_on_unresolved_named_hint(tmp_path):
    controller, _ = _controller_with_project(tmp_path)
    # "run the nonexistent" -> no such project -> None (fall through to routing).
    assert controller.maybe_handle_run_program("run the nonexistent thing") is None


def test_handler_launch_dispatches(tmp_path, monkeypatch):
    controller, proj_dir = _controller_with_project(tmp_path)
    captured = {}

    def _fake_launch(project_path, **kw):
        captured["path"] = Path(project_path)
        return sr.RunResult(ok=True, mode="launch", launched=True,
                            project_name=kw.get("project_name", ""))

    monkeypatch.setattr(sr, "launch_program", _fake_launch)
    resp = controller.maybe_handle_run_program("launch the calculator")
    assert resp is not None and "Launched" in resp.text
    assert captured["path"] == proj_dir


def test_handler_run_is_nonblocking_and_reports(tmp_path, monkeypatch):
    controller, proj_dir = _controller_with_project(tmp_path)

    def _fake_run(project_path, **kw):
        return sr.RunResult(ok=True, mode="run", returncode=0,
                            stdout="answer is 42", project_name=kw.get("project_name", ""))

    monkeypatch.setattr(sr, "run_program", _fake_run)
    resp = controller.maybe_handle_run_program("run the calculator")
    # Immediate, non-blocking ack.
    assert resp is not None and "Running calculator" in resp.text
    # Background thread stashes the report; drain it.
    report = None
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        report = controller.pop_run_report()
        if report:
            break
        time.sleep(0.02)
    assert report is not None and "Ran calculator" in report and "42" in report
    assert controller.pop_run_report() is None  # cleared after pop


def test_handler_empty_hint_no_project_asks(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    registry = ProjectRegistry(path=tmp_path / "projects.json")  # empty
    resolver = ProjectResolver(registry, embedder=None)
    runner = CodingTaskRunner(bridge=_FakeBridge(), log_path=tmp_path / "log.jsonl")
    controller = CodingVoiceController(
        runner=runner, registry=registry, resolver=resolver, sandbox_root=sandbox,
    )
    resp = controller.maybe_handle_run_program("run it")
    assert resp is not None and "don't have a recent project" in resp.text
