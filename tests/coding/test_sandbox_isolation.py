"""Tests for sandbox project isolation (the phase-11 e2e finding).

A sandbox project must become its own git root so the spawned coding
CLI stops walking up into the ultron repo (where it would load the
repo's large local orientation context into every voice coding task).
All hermetic: the git subprocess is injected (binding rule R3/R4).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import ultron.coding.projects as projects_mod
from ultron.coding.projects import (
    ProjectRegistry,
    ensure_sandbox_isolation,
    new_sandbox_project,
)


def _ok_run(calls: list) -> Any:
    def _run(argv: list, **kwargs: Any) -> Any:
        calls.append({"argv": argv, **kwargs})
        return SimpleNamespace(returncode=0)

    return _run


class TestEnsureSandboxIsolation:
    def test_git_inits_project_under_sandbox(self, tmp_path: Path) -> None:
        root = tmp_path / "sandbox"
        project = root / "calc"
        project.mkdir(parents=True)
        calls: list = []
        assert ensure_sandbox_isolation(
            project, sandbox_root=root, run_fn=_ok_run(calls)
        )
        assert calls and calls[0]["argv"][:2] == ["git", "init"]
        assert calls[0]["cwd"] == str(project)

    def test_existing_git_short_circuits(self, tmp_path: Path) -> None:
        root = tmp_path / "sandbox"
        project = root / "calc"
        (project / ".git").mkdir(parents=True)
        calls: list = []
        assert ensure_sandbox_isolation(
            project, sandbox_root=root, run_fn=_ok_run(calls)
        )
        assert calls == []  # idempotent -- no subprocess

    def test_outside_sandbox_never_touched(self, tmp_path: Path) -> None:
        root = tmp_path / "sandbox"
        root.mkdir()
        outside = tmp_path / "users_real_project"
        outside.mkdir()
        calls: list = []
        assert not ensure_sandbox_isolation(
            outside, sandbox_root=root, run_fn=_ok_run(calls)
        )
        assert calls == []

    def test_missing_directory_is_false(self, tmp_path: Path) -> None:
        root = tmp_path / "sandbox"
        root.mkdir()
        assert not ensure_sandbox_isolation(
            root / "ghost", sandbox_root=root, run_fn=_ok_run([])
        )

    def test_git_failure_is_fail_open(self, tmp_path: Path) -> None:
        root = tmp_path / "sandbox"
        project = root / "calc"
        project.mkdir(parents=True)
        assert not ensure_sandbox_isolation(
            project,
            sandbox_root=root,
            run_fn=lambda argv, **k: SimpleNamespace(returncode=128),
        )

    def test_run_exception_is_fail_open(self, tmp_path: Path) -> None:
        root = tmp_path / "sandbox"
        project = root / "calc"
        project.mkdir(parents=True)

        def _boom(argv: list, **k: Any) -> Any:
            raise FileNotFoundError("git not on PATH")

        assert not ensure_sandbox_isolation(
            project, sandbox_root=root, run_fn=_boom
        )


class TestNewSandboxProjectWiring:
    def test_new_project_is_isolated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        isolated: list[Path] = []
        monkeypatch.setattr(
            projects_mod,
            "ensure_sandbox_isolation",
            lambda target, *, sandbox_root=None, run_fn=None: isolated.append(
                Path(target)
            ),
        )
        registry = ProjectRegistry(path=tmp_path / "projects.json")
        project = new_sandbox_project(
            registry, name="calc", sandbox_root=tmp_path / "sandbox",
        )
        assert isolated == [Path(project.path)]

    def test_create_dir_false_skips_isolation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        isolated: list[Path] = []
        monkeypatch.setattr(
            projects_mod,
            "ensure_sandbox_isolation",
            lambda target, **k: isolated.append(Path(target)),
        )
        registry = ProjectRegistry(path=tmp_path / "projects.json")
        new_sandbox_project(
            registry, name="calc", sandbox_root=tmp_path / "sandbox",
            create_dir=False,
        )
        assert isolated == []
