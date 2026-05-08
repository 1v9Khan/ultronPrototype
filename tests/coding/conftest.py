"""Phase 6 test infrastructure: sandbox cleanup + shared fixtures.

The sandbox at ``tests/coding/sandbox/`` is wiped between test runs (per
spec). We do this in a session-scoped autouse fixture so the cleanup
happens once at the start of the orchestration test suite, not between
every individual test (which would be wasteful and slow).

Tests that need true per-test isolation should use ``tmp_path`` instead.
The shared sandbox is for tests that exercise the spec's
``CODING_SANDBOX_PATH`` semantics (project resolution, sandbox-only
project_root validation).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterator

import pytest


# Tests in this directory pass project_root values that may live outside the
# production sandbox path (under tests/coding/sandbox/). The MCP server's
# sandbox check would otherwise refuse them; this env var relaxes the
# check just for these integration tests.
os.environ.setdefault("ULTRON_CODING_MCP_ALLOW_ANY_ROOT", "1")


SANDBOX_DIR: Path = Path(__file__).resolve().parent / "sandbox"


@pytest.fixture(scope="session", autouse=True)
def _clean_sandbox_once() -> Iterator[None]:
    """Wipe the orchestration sandbox at session start. Subsequent tests
    seed their own subdirectories under it."""
    if SANDBOX_DIR.exists():
        for child in SANDBOX_DIR.iterdir():
            if child.name == ".gitkeep":
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except OSError:
                    pass
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    yield


@pytest.fixture
def sandbox_root(tmp_path: Path) -> Path:
    """Per-test sandbox under tmp_path so individual tests don't share
    state. The shared SANDBOX_DIR exists for tests that explicitly need
    the configured production path."""
    root = tmp_path / "sandbox"
    root.mkdir()
    return root
