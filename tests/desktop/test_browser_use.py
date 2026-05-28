"""Tests for the browser-use CLI wrapper (catalog 10 batch 1).

All subprocess.run calls are mocked via monkeypatch; no real
``browser-use`` binary is required to run these tests. No network
access. Per the binding rules in
``docs/test_sweep_binding_rules.md``:

* R1 -- every monkeypatch is via the fixture
* R4 -- no real network calls
* R7 -- order-independent
* R10 -- ~100 tests at <1 ms each, under budget
* R11 -- no voice-stack loading
* R12 -- no bare time.sleep
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Sequence
from unittest.mock import MagicMock

import pytest

from ultron.desktop import browser_use as bu


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@dataclass
class _SubprocessCall:
    """Capture of one subprocess.run invocation."""

    cmd: list[str]
    timeout: float
    env: dict[str, str]
    creationflags: int


class _FakeSubprocess:
    """Stand-in for :func:`subprocess.run` that records every call
    and yields scripted responses. Tests configure the next response
    by setting ``.returncode`` / ``.stdout`` / ``.stderr`` / ``.raises``.
    """

    def __init__(self) -> None:
        self.calls: list[_SubprocessCall] = []
        self.returncode: int = 0
        self.stdout: str = ""
        self.stderr: str = ""
        self.raises: BaseException | None = None
        self._responses: list[
            tuple[int, str, str]
        ] = []  # scripted (returncode, stdout, stderr) queue

    def queue_response(
        self, *, returncode: int = 0, stdout: str = "", stderr: str = ""
    ) -> None:
        """Push a scripted response onto the queue; consumed FIFO."""
        self._responses.append((returncode, stdout, stderr))

    def __call__(
        self,
        cmd: Sequence[str],
        *,
        capture_output: bool,
        text: bool,
        encoding: str,
        errors: str,
        timeout: float,
        creationflags: int,
        env: dict[str, str],
        check: bool,
    ) -> Any:
        self.calls.append(
            _SubprocessCall(
                cmd=list(cmd),
                timeout=float(timeout),
                env=dict(env),
                creationflags=creationflags,
            )
        )
        if self.raises is not None:
            exc = self.raises
            self.raises = None  # one-shot
            raise exc
        if self._responses:
            rc, out, err = self._responses.pop(0)
        else:
            rc, out, err = self.returncode, self.stdout, self.stderr
        result = MagicMock()
        result.returncode = rc
        result.stdout = out
        result.stderr = err
        return result


@pytest.fixture
def fake_subprocess(monkeypatch: pytest.MonkeyPatch) -> _FakeSubprocess:
    """Replace subprocess.run inside the browser_use module + force
    the binary-discovery cache to a deterministic fake path."""
    fake = _FakeSubprocess()
    monkeypatch.setattr(bu.subprocess, "run", fake)
    # Force binary discovery to succeed against a fake path.
    monkeypatch.setattr(
        bu.shutil, "which", lambda name: f"/fake/bin/{name}"
    )
    return fake


@pytest.fixture
def tool(fake_subprocess: _FakeSubprocess) -> bu.BrowserUseTool:
    """Construct a fresh tool that will use the fake subprocess."""
    return bu.BrowserUseTool()


@pytest.fixture(autouse=True)
def _reset_singleton() -> Any:
    """Reset the module-level singleton before AND after every test
    so cross-test state cannot leak. Required for R7 order-independence."""
    bu.reset_browser_use_tool_for_testing()
    yield
    bu.reset_browser_use_tool_for_testing()


# ---------------------------------------------------------------------------
# Construction + binary discovery
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_construction_does_not_resolve_binary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: list[str] = []
        monkeypatch.setattr(
            bu.shutil,
            "which",
            lambda name: called.append(name) or f"/fake/{name}",
        )
        bu.BrowserUseTool()
        assert called == [], (
            "construction should not call shutil.which until first invoke"
        )

    def test_invalid_default_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="default_timeout_s"):
            bu.BrowserUseTool(default_timeout_s=0)

    def test_negative_default_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="default_timeout_s"):
            bu.BrowserUseTool(default_timeout_s=-1.0)

    def test_invalid_session_name_raises(self) -> None:
        with pytest.raises(ValueError, match="session name"):
            bu.BrowserUseTool(session="has spaces")

    def test_session_name_too_long_raises(self) -> None:
        with pytest.raises(ValueError):
            bu.BrowserUseTool(session="a" * 33)

    def test_session_name_valid_characters(self) -> None:
        # Should accept alphanumeric + underscore + hyphen, 1-32 chars.
        bu.BrowserUseTool(session="default")
        bu.BrowserUseTool(session="agent-1")
        bu.BrowserUseTool(session="agent_2")
        bu.BrowserUseTool(session="A" * 32)


class TestBinaryDiscovery:
    def test_resolve_binary_returns_first_match(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bu.shutil, "which", lambda name: f"/usr/bin/{name}")
        tool = bu.BrowserUseTool()
        assert tool.resolve_binary() == "/usr/bin/browser-use"

    def test_resolve_binary_tries_aliases(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def which(name: str) -> str | None:
            return f"/usr/bin/{name}" if name == "bu" else None

        monkeypatch.setattr(bu.shutil, "which", which)
        tool = bu.BrowserUseTool()
        assert tool.resolve_binary() == "/usr/bin/bu"

    def test_resolve_binary_returns_none_when_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bu.shutil, "which", lambda name: None)
        tool = bu.BrowserUseTool()
        assert tool.resolve_binary() is None
        assert tool.is_available() is False

    def test_resolve_binary_caches_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = 0

        def which(name: str) -> str | None:
            nonlocal calls
            calls += 1
            return "/fake/bin"

        monkeypatch.setattr(bu.shutil, "which", which)
        tool = bu.BrowserUseTool()
        first = tool.resolve_binary()
        second = tool.resolve_binary()
        assert first == second == "/fake/bin"
        assert calls == 1, "second resolve should hit the cache"

    def test_reset_binary_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = 0

        def which(name: str) -> str | None:
            nonlocal calls
            calls += 1
            return "/fake/bin"

        monkeypatch.setattr(bu.shutil, "which", which)
        tool = bu.BrowserUseTool()
        tool.resolve_binary()
        tool.reset_binary_cache()
        tool.resolve_binary()
        assert calls == 2

    def test_explicit_binary_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bu.shutil, "which", lambda name: f"/from-which/{name}")
        tool = bu.BrowserUseTool(binary_path="/custom/bu")
        # shutil.which sees the override first.
        assert tool.resolve_binary() == "/from-which//custom/bu"

    def test_missing_binary_returns_fail_open_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bu.shutil, "which", lambda name: None)
        tool = bu.BrowserUseTool()
        result = tool.state()
        assert result.success is False
        assert "not found" in (result.error or "")
        assert result.action == "state"


# ---------------------------------------------------------------------------
# Subprocess construction details
# ---------------------------------------------------------------------------


class TestSubprocessInvocation:
    def test_session_flag_inserted_before_subcommand(
        self, fake_subprocess: _FakeSubprocess
    ) -> None:
        tool = bu.BrowserUseTool(session="agent-1")
        tool.state()
        cmd = fake_subprocess.calls[0].cmd
        # Binary, --session, agent-1, then subcommand.
        assert cmd[1] == "--session"
        assert cmd[2] == "agent-1"
        assert cmd[3] == "state"

    def test_no_session_flag_when_unset(
        self, fake_subprocess: _FakeSubprocess
    ) -> None:
        tool = bu.BrowserUseTool()
        tool.state()
        cmd = fake_subprocess.calls[0].cmd
        assert "--session" not in cmd

    def test_create_no_window_set_on_windows(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        tool.state()
        creationflags = fake_subprocess.calls[0].creationflags
        if sys.platform == "win32":
            assert creationflags == 0x08000000
        else:
            assert creationflags == 0

    def test_env_scrub_removes_browser_use_session(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_subprocess: _FakeSubprocess,
        tool: bu.BrowserUseTool,
    ) -> None:
        monkeypatch.setenv("BROWSER_USE_SESSION", "leaked-session")
        monkeypatch.setenv("OTHER_VAR", "kept")
        tool.state()
        env = fake_subprocess.calls[0].env
        assert "BROWSER_USE_SESSION" not in env
        assert env.get("OTHER_VAR") == "kept"

    def test_env_overrides_cannot_reintroduce_scrubbed_var(
        self,
        fake_subprocess: _FakeSubprocess,
    ) -> None:
        tool = bu.BrowserUseTool(
            env_overrides={"BROWSER_USE_SESSION": "should-not-appear"}
        )
        tool.state()
        env = fake_subprocess.calls[0].env
        assert "BROWSER_USE_SESSION" not in env

    def test_env_overrides_are_applied(
        self, fake_subprocess: _FakeSubprocess
    ) -> None:
        tool = bu.BrowserUseTool(env_overrides={"HTTP_PROXY": "http://proxy"})
        tool.state()
        env = fake_subprocess.calls[0].env
        assert env.get("HTTP_PROXY") == "http://proxy"

    def test_default_timeout_applied(
        self, fake_subprocess: _FakeSubprocess
    ) -> None:
        tool = bu.BrowserUseTool(default_timeout_s=12.0)
        tool.state()
        assert fake_subprocess.calls[0].timeout == 12.0

    def test_per_call_timeout_override(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        tool.open("https://example.com", timeout_s=5.0)
        assert fake_subprocess.calls[0].timeout == 5.0

    def test_non_zero_exit_returns_failure(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.returncode = 2
        fake_subprocess.stderr = "browser not running"
        result = tool.state()
        assert result.success is False
        assert result.exit_code == 2
        assert "browser not running" in (result.error or "")

    def test_subprocess_timeout_returns_failure(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.raises = subprocess.TimeoutExpired(
            cmd=["browser-use"], timeout=1.0
        )
        result = tool.state()
        assert result.success is False
        assert "timeout" in (result.error or "").lower()

    def test_subprocess_spawn_error_returns_failure(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.raises = FileNotFoundError("missing")
        result = tool.state()
        assert result.success is False
        assert "spawn failed" in (result.error or "")

    def test_os_error_returns_failure(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.raises = OSError("broken pipe")
        result = tool.state()
        assert result.success is False
        assert "os error" in (result.error or "")

    def test_large_stdout_truncated(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        # 1 MB of output -> truncated to 256 KB with elision marker.
        fake_subprocess.stdout = "x" * (1024 * 1024)
        result = tool.get_html()
        assert "bytes elided" in result.stdout
        assert len(result.stdout) < 300_000


# ---------------------------------------------------------------------------
# T1 -- state enumeration
# ---------------------------------------------------------------------------


class TestState:
    def test_state_passes_json_flag(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        tool.state()
        assert "--json" in fake_subprocess.calls[0].cmd

    def test_state_parses_canonical_json(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.stdout = json.dumps(
            {
                "url": "https://example.com/page",
                "title": "Example Page",
                "elements": [
                    {
                        "index": 0,
                        "label": "Sign in",
                        "type": "button",
                        "enabled": True,
                    },
                    {
                        "index": 1,
                        "label": "Search",
                        "type": "input",
                        "enabled": True,
                    },
                ],
            }
        )
        result = tool.state()
        assert result.success is True
        assert result.url == "https://example.com/page"
        assert result.title == "Example Page"
        assert len(result.elements) == 2
        assert result.elements[0].index == 0
        assert result.elements[0].label == "Sign in"
        assert result.elements[0].type == "button"

    def test_state_tolerates_alternative_key_names(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.stdout = json.dumps(
            {
                "url": "https://x.com",
                "title": "t",
                "interactive_elements": [
                    {"index": 0, "text": "Click", "role": "button"}
                ],
            }
        )
        result = tool.state()
        assert len(result.elements) == 1
        assert result.elements[0].label == "Click"
        assert result.elements[0].type == "button"

    def test_state_handles_json_parse_failure_softly(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.stdout = "not json"
        result = tool.state()
        # Still ``success=True`` because the CLI itself succeeded.
        assert result.success is True
        assert result.elements == ()
        assert "parse" in (result.error or "").lower()

    def test_state_failure_propagates_when_cli_fails(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.returncode = 1
        result = tool.state()
        assert result.success is False
        assert result.elements == ()


# ---------------------------------------------------------------------------
# T2 -- DOM-native extraction
# ---------------------------------------------------------------------------


class TestGetHtml:
    def test_no_selector(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.stdout = "<html><body>hi</body></html>"
        result = tool.get_html()
        assert result.success is True
        assert result.html == "<html><body>hi</body></html>"
        assert "--selector" not in fake_subprocess.calls[0].cmd

    def test_with_selector(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.stdout = "<h1>Title</h1>"
        result = tool.get_html(selector="h1")
        cmd = fake_subprocess.calls[0].cmd
        assert "--selector" in cmd
        assert "h1" in cmd
        assert result.selector == "h1"

    def test_empty_selector_rejected(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        result = tool.get_html(selector="   ")
        assert result.success is False
        assert "empty selector" in (result.error or "")
        # And subprocess should not be called.
        assert fake_subprocess.calls == []


class TestGetText:
    def test_basic(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.stdout = "Sign in\n"
        result = tool.get_text(0)
        assert result.success is True
        assert result.text == "Sign in"
        assert result.index == 0
        cmd = fake_subprocess.calls[0].cmd
        assert cmd[-3:] == ["get", "text", "0"]

    def test_negative_index_rejected(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        result = tool.get_text(-1)
        assert result.success is False
        assert "non-negative" in (result.error or "")
        assert fake_subprocess.calls == []


class TestGetValue:
    def test_basic(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.stdout = "user@example.com"
        result = tool.get_value(3)
        assert result.success is True
        assert result.value == "user@example.com"
        assert result.index == 3

    def test_negative_index_rejected(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        result = tool.get_value(-5)
        assert result.success is False


class TestGetAttributes:
    def test_parses_json_mapping(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.stdout = json.dumps(
            {"id": "btn1", "class": "primary", "type": "submit"}
        )
        result = tool.get_attributes(2)
        assert result.success is True
        assert result.attributes == {
            "id": "btn1",
            "class": "primary",
            "type": "submit",
        }

    def test_non_mapping_payload_falls_back_to_raw(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.stdout = json.dumps(["not", "a", "mapping"])
        result = tool.get_attributes(2)
        assert "__raw__" in result.attributes
        assert "non-mapping" in (result.error or "")

    def test_parse_failure_falls_back_to_raw(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.stdout = "not json at all"
        result = tool.get_attributes(2)
        assert "__raw__" in result.attributes
        assert "parse failed" in (result.error or "").lower()


class TestGetBbox:
    def test_canonical_shape(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.stdout = json.dumps(
            {"x": 100, "y": 200, "width": 50, "height": 30}
        )
        result = tool.get_bbox(0)
        assert result.success is True
        assert result.bbox is not None
        assert result.bbox.x == 100
        assert result.bbox.y == 200
        assert result.bbox.width == 50
        assert result.bbox.height == 30
        assert result.bbox.center_x == 125
        assert result.bbox.center_y == 215
        assert result.bbox.center == (125, 215)

    def test_left_top_shape(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.stdout = json.dumps(
            {"left": 10, "top": 20, "width": 40, "height": 80}
        )
        result = tool.get_bbox(0)
        assert result.bbox is not None
        assert result.bbox.x == 10
        assert result.bbox.y == 20

    def test_short_w_h_shape(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.stdout = json.dumps(
            {"x": 5, "y": 5, "w": 100, "h": 50}
        )
        result = tool.get_bbox(0)
        assert result.bbox is not None
        assert result.bbox.width == 100

    def test_negative_dimensions_rejected(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.stdout = json.dumps(
            {"x": 0, "y": 0, "width": -1, "height": 50}
        )
        result = tool.get_bbox(0)
        assert result.success is False
        assert "negative" in (result.error or "")

    def test_malformed_payload(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.stdout = "garbage"
        result = tool.get_bbox(0)
        assert result.success is False
        assert "parse" in (result.error or "").lower()


class TestGetTitle:
    def test_basic(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.stdout = "Page Title\n"
        result = tool.get_title()
        assert result.success is True
        assert result.title == "Page Title"


# ---------------------------------------------------------------------------
# T5 -- wait barriers
# ---------------------------------------------------------------------------


class TestWaitSelector:
    def test_canonical(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        result = tool.wait_selector(".content", timeout_ms=10_000)
        cmd = fake_subprocess.calls[0].cmd
        assert "wait" in cmd
        assert "selector" in cmd
        assert ".content" in cmd
        assert "--state" in cmd
        assert "visible" in cmd
        assert "--timeout" in cmd
        assert "10000" in cmd
        assert result.matched is True
        assert result.target == ".content"

    def test_default_state_visible(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        result = tool.wait_selector(".x")
        assert result.state == "visible"

    def test_all_states_accepted(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        for state in bu.WAIT_SELECTOR_STATES:
            r = tool.wait_selector(".x", state=state)
            assert r.success is True

    def test_invalid_state_rejected(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        result = tool.wait_selector(".x", state="exploded")
        assert result.success is False
        assert "state must" in (result.error or "")
        assert fake_subprocess.calls == []

    def test_empty_selector_rejected(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        result = tool.wait_selector("   ")
        assert result.success is False
        assert "empty selector" in (result.error or "")

    def test_subprocess_timeout_exceeds_wait_timeout(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        tool.wait_selector(".x", timeout_ms=10_000)
        # 10s wait -> subprocess timeout >= 15s (wait + 5s margin).
        assert fake_subprocess.calls[0].timeout >= 15.0

    def test_negative_timeout_rejected(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        result = tool.wait_selector(".x", timeout_ms=-1)
        assert result.success is False
        assert "positive" in (result.error or "")

    def test_wait_failure_returns_unmatched(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.returncode = 1
        fake_subprocess.stderr = "Timeout 5000ms exceeded"
        result = tool.wait_selector(".missing")
        assert result.success is False
        assert result.matched is False


class TestWaitText:
    def test_canonical(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        result = tool.wait_text("Welcome")
        cmd = fake_subprocess.calls[0].cmd
        assert "wait" in cmd
        assert "text" in cmd
        assert "Welcome" in cmd
        assert result.matched is True
        assert result.target == "Welcome"
        assert result.state == "text"

    def test_empty_text_rejected(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        result = tool.wait_text("")
        assert result.success is False
        assert fake_subprocess.calls == []


# ---------------------------------------------------------------------------
# T6 -- tab lifecycle
# ---------------------------------------------------------------------------


class TestTabList:
    def test_parses_canonical_list(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.stdout = json.dumps(
            [
                {
                    "index": 0,
                    "url": "https://a.com",
                    "title": "A",
                    "active": True,
                },
                {
                    "index": 1,
                    "url": "https://b.com",
                    "title": "B",
                    "active": False,
                },
            ]
        )
        result = tool.tab_list()
        assert result.success is True
        assert len(result.tabs) == 2
        assert result.tabs[0].active is True
        assert result.tabs[1].title == "B"

    def test_parses_tabs_under_mapping(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.stdout = json.dumps(
            {"tabs": [{"index": 0, "url": "https://a.com"}]}
        )
        result = tool.tab_list()
        assert len(result.tabs) == 1
        assert result.tabs[0].url == "https://a.com"
        assert result.tabs[0].active is False  # missing key -> False

    def test_parse_failure(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        fake_subprocess.stdout = "no json"
        result = tool.tab_list()
        assert result.tabs == ()
        assert "parse" in (result.error or "").lower()


class TestTabNew:
    def test_blank(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        tool.tab_new()
        cmd = fake_subprocess.calls[0].cmd
        assert cmd[-2:] == ["tab", "new"]

    def test_with_url(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        tool.tab_new("https://example.com")
        cmd = fake_subprocess.calls[0].cmd
        assert cmd[-3:] == ["tab", "new", "https://example.com"]

    def test_empty_url_rejected(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        result = tool.tab_new("   ")
        assert result.success is False
        assert fake_subprocess.calls == []


class TestTabSwitch:
    def test_basic(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        tool.tab_switch(2)
        cmd = fake_subprocess.calls[0].cmd
        assert cmd[-3:] == ["tab", "switch", "2"]

    def test_negative_rejected(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        result = tool.tab_switch(-1)
        assert result.success is False


class TestTabClose:
    def test_single_index(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        tool.tab_close([1])
        cmd = fake_subprocess.calls[0].cmd
        assert cmd[-3:] == ["tab", "close", "1"]

    def test_multiple_indices(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        tool.tab_close([1, 3, 5])
        cmd = fake_subprocess.calls[0].cmd
        # Last 5 args: tab, close, 1, 3, 5
        assert cmd[-5:] == ["tab", "close", "1", "3", "5"]

    def test_empty_rejected(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        result = tool.tab_close([])
        assert result.success is False

    def test_negative_in_list_rejected(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        result = tool.tab_close([1, -2, 3])
        assert result.success is False


# ---------------------------------------------------------------------------
# Navigation + lifecycle helpers
# ---------------------------------------------------------------------------


class TestOpenAndClose:
    def test_open_url(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        tool.open("https://example.com")
        cmd = fake_subprocess.calls[0].cmd
        assert cmd[-2:] == ["open", "https://example.com"]

    def test_open_headed_appends_flag(
        self, fake_subprocess: _FakeSubprocess
    ) -> None:
        tool = bu.BrowserUseTool(headed=True)
        tool.open("https://example.com")
        cmd = fake_subprocess.calls[0].cmd
        # Flag comes BEFORE the subcommand.
        assert "--headed" in cmd
        headed_idx = cmd.index("--headed")
        open_idx = cmd.index("open")
        assert headed_idx < open_idx

    def test_open_empty_url_rejected(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        result = tool.open("")
        assert result.success is False
        assert fake_subprocess.calls == []

    def test_back(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        tool.back()
        cmd = fake_subprocess.calls[0].cmd
        assert cmd[-1] == "back"

    def test_close(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        tool.close()
        cmd = fake_subprocess.calls[0].cmd
        assert cmd[-1] == "close"
        assert "--all" not in cmd

    def test_close_all(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        tool.close(all_sessions=True)
        cmd = fake_subprocess.calls[0].cmd
        assert "--all" in cmd


class TestScroll:
    def test_down(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        tool.scroll("down")
        cmd = fake_subprocess.calls[0].cmd
        assert cmd[-2:] == ["scroll", "down"]

    def test_up(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        tool.scroll("up")
        cmd = fake_subprocess.calls[0].cmd
        assert cmd[-2:] == ["scroll", "up"]

    def test_with_amount(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        tool.scroll("down", amount=500)
        cmd = fake_subprocess.calls[0].cmd
        assert "--amount" in cmd
        assert "500" in cmd

    def test_invalid_direction_rejected(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        result = tool.scroll("diagonal")
        assert result.success is False
        assert fake_subprocess.calls == []

    def test_negative_amount_rejected(
        self, fake_subprocess: _FakeSubprocess, tool: bu.BrowserUseTool
    ) -> None:
        result = tool.scroll("down", amount=-1)
        assert result.success is False


# ---------------------------------------------------------------------------
# Singleton + with_session
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_set_get_round_trip(self) -> None:
        tool = bu.BrowserUseTool()
        bu.set_browser_use_tool(tool)
        assert bu.get_browser_use_tool() is tool

    def test_unset_returns_none(self) -> None:
        assert bu.get_browser_use_tool() is None
        bu.set_browser_use_tool(bu.BrowserUseTool())
        bu.reset_browser_use_tool_for_testing()
        assert bu.get_browser_use_tool() is None


class TestWithSession:
    def test_returns_new_instance(self) -> None:
        a = bu.BrowserUseTool()
        b = a.with_session("agent-1")
        assert a is not b
        assert a.session is None
        assert b.session == "agent-1"

    def test_invalid_name_raises(self) -> None:
        a = bu.BrowserUseTool()
        with pytest.raises(ValueError):
            a.with_session("has spaces")

    def test_unset_via_none(self) -> None:
        a = bu.BrowserUseTool(session="x")
        b = a.with_session(None)
        assert b.session is None


# ---------------------------------------------------------------------------
# Helper-function unit tests
# ---------------------------------------------------------------------------


class TestSessionNameValidator:
    def test_alphanumeric_ok(self) -> None:
        assert bu._is_valid_session_name("default")
        assert bu._is_valid_session_name("Agent42")

    def test_with_underscore_and_hyphen(self) -> None:
        assert bu._is_valid_session_name("agent_1-test")

    def test_too_long(self) -> None:
        assert not bu._is_valid_session_name("a" * 33)

    def test_empty(self) -> None:
        assert not bu._is_valid_session_name("")

    def test_invalid_chars(self) -> None:
        assert not bu._is_valid_session_name("a/b")
        assert not bu._is_valid_session_name("a b")
        assert not bu._is_valid_session_name("a.b")


class TestEnvScrub:
    def test_drops_scrub_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BROWSER_USE_SESSION", "leaked")
        env = bu._build_scrubbed_env({})
        assert "BROWSER_USE_SESSION" not in env

    def test_keeps_other_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KEEP_ME", "yes")
        env = bu._build_scrubbed_env({})
        assert env.get("KEEP_ME") == "yes"

    def test_overrides_layer_on_top(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("X", "old")
        env = bu._build_scrubbed_env({"X": "new"})
        assert env["X"] == "new"

    def test_overrides_cannot_reintroduce_scrub_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = bu._build_scrubbed_env({"BROWSER_USE_SESSION": "smuggled"})
        assert "BROWSER_USE_SESSION" not in env


class TestTruncate:
    def test_small_payload_unchanged(self) -> None:
        assert bu._truncate("hello") == "hello"

    def test_empty_string(self) -> None:
        assert bu._truncate("") == ""

    def test_large_payload_truncated(self) -> None:
        text = "x" * (1024 * 1024)
        result = bu._truncate(text)
        assert "bytes elided" in result
        assert len(result.encode("utf-8")) < 300_000


class TestExtractCliError:
    def test_prefers_stderr(self) -> None:
        assert bu._extract_cli_error("oops\n", "stdout") == "oops"

    def test_falls_back_to_stdout(self) -> None:
        assert bu._extract_cli_error("", "from stdout") == "from stdout"

    def test_strips_whitespace_and_caps(self) -> None:
        long_line = "e" * 1000
        out = bu._extract_cli_error(long_line, "")
        assert out is not None
        assert len(out) <= 512

    def test_returns_none_when_both_blank(self) -> None:
        assert bu._extract_cli_error("   ", "  \n") is None


class TestParseStateJson:
    def test_canonical(self) -> None:
        payload = json.dumps(
            {
                "url": "https://x.com",
                "title": "X",
                "elements": [
                    {"index": 0, "label": "a", "type": "button"}
                ],
            }
        )
        result = bu._try_parse_state_json(payload)
        assert result is not None
        assert result["url"] == "https://x.com"
        assert result["elements"][0].index == 0

    def test_returns_none_on_garbage(self) -> None:
        assert bu._try_parse_state_json("not json") is None

    def test_returns_none_on_array(self) -> None:
        # The top-level must be a mapping.
        assert bu._try_parse_state_json("[]") is None

    def test_returns_none_on_empty(self) -> None:
        assert bu._try_parse_state_json("") is None


class TestParseBbox:
    def test_canonical(self) -> None:
        bbox, err = bu._try_parse_bbox(
            json.dumps({"x": 10, "y": 20, "width": 30, "height": 40})
        )
        assert bbox is not None
        assert err is None
        assert bbox.x == 10

    def test_empty(self) -> None:
        bbox, err = bu._try_parse_bbox("")
        assert bbox is None
        assert err is not None

    def test_invalid_types(self) -> None:
        bbox, err = bu._try_parse_bbox(
            json.dumps({"x": "abc", "y": 0, "width": 10, "height": 10})
        )
        assert bbox is None
        assert err is not None


class TestParseTabs:
    def test_list_shape(self) -> None:
        tabs, err = bu._try_parse_tabs(
            json.dumps([{"index": 0, "url": "https://x.com"}])
        )
        assert err is None
        assert len(tabs) == 1
        assert tabs[0].url == "https://x.com"

    def test_mapping_shape(self) -> None:
        tabs, err = bu._try_parse_tabs(
            json.dumps({"tabs": [{"index": 0, "url": "https://x.com"}]})
        )
        assert err is None
        assert len(tabs) == 1

    def test_missing_tabs_key(self) -> None:
        tabs, err = bu._try_parse_tabs(json.dumps({"other": []}))
        assert tabs == ()
        assert err is not None

    def test_empty_string(self) -> None:
        tabs, err = bu._try_parse_tabs("")
        assert tabs == ()
        assert err is not None

    def test_garbage_json(self) -> None:
        tabs, err = bu._try_parse_tabs("nope")
        assert tabs == ()


# ---------------------------------------------------------------------------
# Public API exposure
# ---------------------------------------------------------------------------


class TestPublicApi:
    def test_all_exports_present(self) -> None:
        for name in bu.__all__:
            assert hasattr(bu, name), f"__all__ lists {name!r} but it's missing"

    def test_constants_have_expected_values(self) -> None:
        assert bu.DEFAULT_TIMEOUT_S > 0
        assert bu.DEFAULT_WAIT_TIMEOUT_MS > 0
        assert "visible" in bu.WAIT_SELECTOR_STATES
        assert "down" in bu.SCROLL_DIRECTIONS
        assert "browser-use" in bu.BROWSER_USE_BINARY_CANDIDATES


# ===========================================================================
# Batch 2 -- write primitives (T7) + screenshot (T9)
# ===========================================================================


# ---------------------------------------------------------------------------
# Safety-validator fake + fixture
# ---------------------------------------------------------------------------


class _FakeValidator:
    """Records every RuleContext + yields a configurable verdict.

    Default verdict is ALLOW (production-realistic), tests flip
    ``next_verdict`` / ``next_message`` to simulate denials.
    """

    def __init__(self) -> None:
        from ultron.safety.validator import ValidatorVerdict, Verdict

        self.contexts: list = []
        self._allow_verdict = ValidatorVerdict(
            verdict=Verdict.ALLOW, reason="test-allow"
        )
        self.next_verdict = self._allow_verdict
        self.raises: BaseException | None = None
        self.call_count = 0

    def check(self, ctx: Any) -> Any:
        self.contexts.append(ctx)
        self.call_count += 1
        if self.raises is not None:
            exc = self.raises
            self.raises = None
            raise exc
        return self.next_verdict

    def block(self, *, message: str = "blocked by test") -> None:
        from ultron.safety.validator import ValidatorVerdict, Verdict

        self.next_verdict = ValidatorVerdict(
            verdict=Verdict.BLOCK_HARD,
            reason=message,
            user_message=message,
            triggered_rule_id="test_rule",
        )

    def needs_explicit_intent(self, *, message: str = "needs intent") -> None:
        from ultron.safety.validator import ValidatorVerdict, Verdict

        self.next_verdict = ValidatorVerdict(
            verdict=Verdict.NEEDS_EXPLICIT_INTENT,
            reason=message,
            user_message=message,
            triggered_rule_id="test_rule",
        )

    def allow(self) -> None:
        self.next_verdict = self._allow_verdict


class _FakePathResolver:
    """Records every safe_realpath / resolve call. Returns the value
    from ``real_paths`` for safe_realpath, falling back to ``None`` when
    the input isn't in the map. ``resolve`` returns the value from
    ``resolves`` or ``Path(raw)`` as a default."""

    def __init__(self) -> None:
        self.safe_realpath_calls: list[str] = []
        self.resolve_calls: list[str] = []
        self.real_paths: dict[str, Any] = {}
        self.resolves: dict[str, Any] = {}

    def safe_realpath(self, raw: Any) -> Any:
        self.safe_realpath_calls.append(str(raw))
        return self.real_paths.get(str(raw), None)

    def resolve(self, raw: Any) -> Any:
        from pathlib import Path

        self.resolve_calls.append(str(raw))
        return self.resolves.get(str(raw), Path(str(raw)))


@pytest.fixture
def fake_validator(monkeypatch: pytest.MonkeyPatch) -> _FakeValidator:
    """Install a fake validator that the wrapper resolves via
    :func:`get_validator`. The default verdict is ALLOW."""
    fake = _FakeValidator()
    monkeypatch.setattr(bu, "get_validator", lambda: fake)
    return fake


@pytest.fixture
def fake_path_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> _FakePathResolver:
    """Install a fake PathResolver that the wrapper resolves via
    :func:`get_path_resolver`."""
    fake = _FakePathResolver()
    monkeypatch.setattr(bu, "get_path_resolver", lambda: fake)
    return fake


# ---------------------------------------------------------------------------
# Safety check helper
# ---------------------------------------------------------------------------


class TestSafetyCheck:
    def test_allowed_returns_none(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        result = tool.click_at_index(0, user_text="click sign in")
        assert result.success is True
        assert result.safety_verdict == "ALLOW"
        assert len(fake_subprocess.calls) == 1

    def test_block_hard_short_circuits(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        fake_validator.block(message="not allowed in this app")
        result = tool.click_at_index(0, user_text="click")
        assert result.success is False
        assert result.safety_verdict == "BLOCK_HARD"
        assert "not allowed" in (result.error or "")
        # Subprocess MUST NOT have been called.
        assert fake_subprocess.calls == []

    def test_needs_explicit_intent_blocks(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        fake_validator.needs_explicit_intent(message="say the verb please")
        result = tool.click_at_index(0, user_text="?")
        assert result.success is False
        assert result.safety_verdict == "NEEDS_EXPLICIT_INTENT"
        assert fake_subprocess.calls == []

    def test_log_only_allows_call(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from ultron.safety.validator import ValidatorVerdict, Verdict

        fake_validator.next_verdict = ValidatorVerdict(
            verdict=Verdict.LOG_ONLY, reason="log it"
        )
        result = tool.click_at_index(0, user_text="click")
        assert result.success is True
        assert len(fake_subprocess.calls) == 1

    def test_validator_raises_blocks_call(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        fake_validator.raises = RuntimeError("validator broke")
        result = tool.click_at_index(0, user_text="click")
        assert result.success is False
        assert result.safety_verdict == "BLOCK_HARD"
        assert "raised" in (result.error or "")
        assert fake_subprocess.calls == []

    def test_validator_unavailable_is_permissive(
        self,
        fake_subprocess: _FakeSubprocess,
        tool: bu.BrowserUseTool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # No fake validator installed; the real NoOp validator returns
        # ALLOW. Verify the call proceeds.
        result = tool.click_at_index(0, user_text="click")
        assert result.success is True
        assert len(fake_subprocess.calls) == 1

    def test_context_carries_right_tool_name_and_capability(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        tool.click_at_index(7, user_text="click the link")
        assert len(fake_validator.contexts) == 1
        ctx = fake_validator.contexts[0]
        assert ctx.tool_name == "desktop.browser_use.click_at_index"
        assert ctx.capability == "desktop_browser_use"
        assert ctx.arguments == {"index": 7}
        assert ctx.user_text == "click the link"


# ---------------------------------------------------------------------------
# T7 -- click variants
# ---------------------------------------------------------------------------


class TestClickAtIndex:
    def test_canonical(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        result = tool.click_at_index(5, user_text="click sign in")
        assert result.success is True
        assert result.target == "5"
        cmd = fake_subprocess.calls[0].cmd
        assert cmd[-2:] == ["click", "5"]

    def test_negative_rejected_before_validator(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        result = tool.click_at_index(-1, user_text="click")
        assert result.success is False
        # Validator must NOT have been consulted -- arg validation is first.
        assert fake_validator.call_count == 0
        assert fake_subprocess.calls == []

    def test_blocked_target_field_populated(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        fake_validator.block(message="nope")
        result = tool.click_at_index(3, user_text="click")
        assert result.target == "3"
        assert result.success is False


class TestClickAtCoords:
    def test_canonical(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        result = tool.click_at_coords(150, 250, user_text="click here")
        assert result.success is True
        assert result.target == "150,250"
        cmd = fake_subprocess.calls[0].cmd
        assert cmd[-3:] == ["click", "150", "250"]

    def test_negative_rejected(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        for x, y in [(-1, 0), (0, -1), (-5, -5)]:
            result = tool.click_at_coords(x, y, user_text="click")
            assert result.success is False
        assert fake_subprocess.calls == []

    def test_zero_zero_allowed(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        result = tool.click_at_coords(0, 0, user_text="click corner")
        assert result.success is True


# ---------------------------------------------------------------------------
# T7 -- type_text / input / select
# ---------------------------------------------------------------------------


class TestTypeText:
    def test_canonical(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        result = tool.type_text("hello world", user_text="type hello")
        assert result.success is True
        cmd = fake_subprocess.calls[0].cmd
        assert cmd[-2:] == ["type", "hello world"]

    def test_empty_text_rejected_before_validator(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        result = tool.type_text("", user_text="type")
        assert result.success is False
        assert fake_validator.call_count == 0
        assert fake_subprocess.calls == []

    def test_validator_sees_text_preview_not_full_text(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        long_text = "x" * 500
        tool.type_text(long_text, user_text="type")
        ctx = fake_validator.contexts[0]
        # The arguments dict carries text_preview, not the raw text,
        # so the validator + audit log don't echo arbitrarily large
        # payloads.
        assert "text_preview" in ctx.arguments
        assert len(ctx.arguments["text_preview"]) <= 90
        assert "text" not in ctx.arguments  # raw text never lands in args


class TestInput:
    def test_canonical(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        result = tool.input(3, "user@example.com", user_text="fill email")
        assert result.success is True
        cmd = fake_subprocess.calls[0].cmd
        assert cmd[-3:] == ["input", "3", "user@example.com"]

    def test_negative_index_rejected(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        result = tool.input(-1, "text", user_text="input")
        assert result.success is False
        assert fake_subprocess.calls == []

    def test_empty_text_rejected(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        result = tool.input(3, "", user_text="input")
        assert result.success is False
        assert fake_subprocess.calls == []


class TestSelect:
    def test_canonical(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        result = tool.select(2, "United States", user_text="pick country")
        assert result.success is True
        cmd = fake_subprocess.calls[0].cmd
        assert cmd[-3:] == ["select", "2", "United States"]

    def test_negative_index_rejected(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        assert tool.select(-1, "X", user_text="").success is False
        assert fake_subprocess.calls == []

    def test_empty_option_rejected(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        assert tool.select(2, "", user_text="").success is False
        assert fake_subprocess.calls == []


# ---------------------------------------------------------------------------
# T7 upload (YELLOW per security review)
# ---------------------------------------------------------------------------


class TestUpload:
    def test_canonical_path_resolves_and_passes(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        fake_path_resolver: _FakePathResolver,
        tool: bu.BrowserUseTool,
        tmp_path: Any,
    ) -> None:
        real_file = tmp_path / "doc.pdf"
        real_file.write_bytes(b"%PDF-1.4\n")
        fake_path_resolver.real_paths[str(real_file)] = real_file
        result = tool.upload(
            5, str(real_file), user_text="upload my pdf"
        )
        assert result.success is True
        cmd = fake_subprocess.calls[0].cmd
        assert "upload" in cmd
        assert "5" in cmd
        # The CLI receives the RESOLVED path, not the raw input.
        assert str(real_file) in cmd
        # Validator saw the resolved path tuple too.
        ctx = fake_validator.contexts[0]
        assert ctx.paths == (real_file,)
        assert ctx.arguments["path"] == str(real_file)

    def test_negative_index_rejected_before_resolver(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        fake_path_resolver: _FakePathResolver,
        tool: bu.BrowserUseTool,
    ) -> None:
        result = tool.upload(-1, "/some/path", user_text="upload")
        assert result.success is False
        assert fake_path_resolver.safe_realpath_calls == []
        assert fake_validator.call_count == 0
        assert fake_subprocess.calls == []

    def test_empty_path_rejected(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        fake_path_resolver: _FakePathResolver,
        tool: bu.BrowserUseTool,
    ) -> None:
        result = tool.upload(3, "   ", user_text="upload")
        assert result.success is False
        assert fake_path_resolver.safe_realpath_calls == []

    def test_unresolved_path_rejected(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        fake_path_resolver: _FakePathResolver,
        tool: bu.BrowserUseTool,
    ) -> None:
        # PathResolver returns None -> reject without invoking validator
        # or subprocess.
        result = tool.upload(3, "/nope.pdf", user_text="upload")
        assert result.success is False
        assert "does not resolve" in (result.error or "")
        assert fake_validator.call_count == 0
        assert fake_subprocess.calls == []

    def test_non_file_rejected(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        fake_path_resolver: _FakePathResolver,
        tool: bu.BrowserUseTool,
        tmp_path: Any,
    ) -> None:
        # Resolves to a directory -> rejected (not a regular file).
        a_dir = tmp_path / "subdir"
        a_dir.mkdir()
        fake_path_resolver.real_paths[str(a_dir)] = a_dir
        result = tool.upload(3, str(a_dir), user_text="upload")
        assert result.success is False
        assert "not a regular file" in (result.error or "")
        assert fake_validator.call_count == 0
        assert fake_subprocess.calls == []

    def test_safety_block_short_circuits(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        fake_path_resolver: _FakePathResolver,
        tool: bu.BrowserUseTool,
        tmp_path: Any,
    ) -> None:
        real_file = tmp_path / "x.txt"
        real_file.write_text("data")
        fake_path_resolver.real_paths[str(real_file)] = real_file
        fake_validator.block(message="path not in sandbox")
        result = tool.upload(3, str(real_file), user_text="upload")
        assert result.success is False
        assert result.safety_verdict == "BLOCK_HARD"
        assert fake_subprocess.calls == []

    def test_inject_path_resolver_kwarg(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Verify the injectable resolver kwarg wins over the global.
        the_file = tmp_path / "thing.txt"
        the_file.write_text("data")
        injected = _FakePathResolver()
        injected.real_paths[str(the_file)] = the_file
        # Make the global resolver intentionally return None so failure
        # = "global was consulted instead".
        global_resolver = _FakePathResolver()
        monkeypatch.setattr(bu, "get_path_resolver", lambda: global_resolver)
        result = tool.upload(
            3,
            str(the_file),
            user_text="upload",
            path_resolver=injected,
        )
        assert result.success is True
        # The injected resolver was called; the global was not.
        assert injected.safe_realpath_calls == [str(the_file)]
        assert global_resolver.safe_realpath_calls == []


# ---------------------------------------------------------------------------
# T7 -- hover / keys / dblclick / rightclick
# ---------------------------------------------------------------------------


class TestHover:
    def test_canonical(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        tool.hover(3, user_text="hover the menu")
        cmd = fake_subprocess.calls[0].cmd
        assert cmd[-2:] == ["hover", "3"]

    def test_negative_rejected(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        assert tool.hover(-1, user_text="hover").success is False


class TestKeys:
    def test_canonical(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        tool.keys("Enter", user_text="press enter")
        cmd = fake_subprocess.calls[0].cmd
        assert cmd[-2:] == ["keys", "Enter"]

    def test_modifier_combo_passes_through(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        tool.keys("Control+a", user_text="select all")
        cmd = fake_subprocess.calls[0].cmd
        assert "Control+a" in cmd

    def test_empty_rejected(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        assert tool.keys("   ", user_text="keys").success is False


class TestDblclick:
    def test_canonical(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        tool.dblclick(4, user_text="double-click")
        cmd = fake_subprocess.calls[0].cmd
        assert cmd[-2:] == ["dblclick", "4"]

    def test_negative_rejected(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        assert tool.dblclick(-1, user_text="").success is False


class TestRightclick:
    def test_canonical(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        tool.rightclick(4, user_text="context menu")
        cmd = fake_subprocess.calls[0].cmd
        assert cmd[-2:] == ["rightclick", "4"]


# ---------------------------------------------------------------------------
# T9 -- screenshot
# ---------------------------------------------------------------------------


class TestScreenshotPathMode:
    def test_path_mode_resolves_via_path_resolver(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        fake_path_resolver: _FakePathResolver,
        tool: bu.BrowserUseTool,
        tmp_path: Any,
    ) -> None:
        target = tmp_path / "shot.png"
        # safe_realpath returns None (file doesn't exist yet) -> the
        # method falls through to ``resolve`` and validates the parent.
        fake_path_resolver.resolves[str(target)] = target
        result = tool.screenshot(str(target), user_text="screenshot")
        assert result.success is True
        assert result.path == str(target)
        cmd = fake_subprocess.calls[0].cmd
        assert str(target) in cmd

    def test_full_page_flag(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        fake_path_resolver: _FakePathResolver,
        tool: bu.BrowserUseTool,
        tmp_path: Any,
    ) -> None:
        target = tmp_path / "full.png"
        fake_path_resolver.resolves[str(target)] = target
        tool.screenshot(str(target), full_page=True, user_text="full")
        cmd = fake_subprocess.calls[0].cmd
        assert "--full" in cmd

    def test_empty_path_rejected(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        fake_path_resolver: _FakePathResolver,
        tool: bu.BrowserUseTool,
    ) -> None:
        result = tool.screenshot("   ", user_text="ss")
        assert result.success is False
        assert "empty path" in (result.error or "")
        assert fake_subprocess.calls == []

    def test_missing_parent_rejected(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        fake_path_resolver: _FakePathResolver,
        tool: bu.BrowserUseTool,
        tmp_path: Any,
    ) -> None:
        # Parent doesn't exist -> rejected.
        missing_dir = tmp_path / "does-not-exist" / "ss.png"
        fake_path_resolver.resolves[str(missing_dir)] = missing_dir
        result = tool.screenshot(str(missing_dir), user_text="ss")
        assert result.success is False
        assert "parent directory" in (result.error or "")
        assert fake_subprocess.calls == []


class TestScreenshotBase64Mode:
    @staticmethod
    def _png_base64() -> str:
        # Smallest valid-looking payload: 16 bytes is the min our
        # decoder accepts.
        import base64

        return base64.b64encode(b"\x89PNG\r\n\x1a\n" * 4).decode("ascii")

    def test_base64_mode_decodes(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        fake_subprocess.stdout = self._png_base64()
        result = tool.screenshot(user_text="ss")
        assert result.success is True
        assert result.path is None
        assert result.image_bytes is not None
        assert len(result.image_bytes) >= 16

    def test_data_uri_prefix_stripped(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        fake_subprocess.stdout = "data:image/png;base64," + self._png_base64()
        result = tool.screenshot(user_text="ss")
        assert result.success is True
        assert result.image_bytes is not None

    def test_jpeg_data_uri_supported(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        fake_subprocess.stdout = "data:image/jpeg;base64," + self._png_base64()
        result = tool.screenshot(user_text="ss")
        assert result.success is True
        assert result.image_bytes is not None

    def test_invalid_payload_records_decode_error(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        fake_subprocess.stdout = "@@@not base64@@@"
        result = tool.screenshot(user_text="ss")
        # Subprocess succeeded but decoding failed; the method records
        # the decode error on the result rather than flipping success.
        assert result.success is True
        assert result.image_bytes is None
        assert (result.error or "").startswith("base64 decode failed") or (
            "decode" in (result.error or "")
        )

    def test_too_small_payload_recorded(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        import base64

        fake_subprocess.stdout = base64.b64encode(b"abc").decode("ascii")
        result = tool.screenshot(user_text="ss")
        assert result.image_bytes is None
        assert "too small" in (result.error or "")

    def test_subprocess_failure_propagates(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        fake_subprocess.returncode = 1
        fake_subprocess.stderr = "no browser open"
        result = tool.screenshot(user_text="ss")
        assert result.success is False
        assert result.image_bytes is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestFailedAction:
    def test_shape(self) -> None:
        r = bu._failed_action("click_at_index", "negative index")
        assert r.success is False
        assert r.action == "click_at_index"
        assert r.error == "negative index"
        assert r.safety_verdict == ""


class TestActionFromInvoke:
    def test_passes_fields_through(self) -> None:
        inv = bu.BrowserUseResult(
            success=True,
            action="hover",
            stdout="ok",
            stderr="",
            elapsed_ms=12.3,
            exit_code=0,
        )
        a = bu._action_from_invoke(inv, target="5", safety_verdict="ALLOW")
        assert a.success is True
        assert a.action == "hover"
        assert a.stdout == "ok"
        assert a.target == "5"
        assert a.safety_verdict == "ALLOW"
        assert a.elapsed_ms == 12.3
        assert a.exit_code == 0


class TestPreview:
    def test_short_unchanged(self) -> None:
        assert bu._preview("hello") == "hello"

    def test_empty(self) -> None:
        assert bu._preview("") == ""

    def test_collapses_whitespace(self) -> None:
        assert bu._preview("a   b\n\tc") == "a b c"

    def test_truncates_with_ellipsis(self) -> None:
        out = bu._preview("x" * 200, cap=80)
        assert out.endswith("…")
        assert len(out) <= 80


class TestShortTargetLabel:
    def test_index_only(self) -> None:
        assert bu._short_target_label({"index": 3}) == "3"

    def test_index_and_text(self) -> None:
        assert (
            bu._short_target_label({"index": 3, "text_preview": "hello"})
            == "3:hello"
        )

    def test_index_and_option(self) -> None:
        assert (
            bu._short_target_label({"index": 2, "option": "US"}) == "2:US"
        )

    def test_index_and_path(self) -> None:
        assert (
            bu._short_target_label({"index": 1, "path": "/x"}) == "1:/x"
        )

    def test_coords(self) -> None:
        assert bu._short_target_label({"x": 10, "y": 20}) == "10,20"

    def test_combo(self) -> None:
        assert bu._short_target_label({"combo": "Enter"}) == "Enter"

    def test_unknown_returns_empty(self) -> None:
        assert bu._short_target_label({"foo": "bar"}) == ""


class TestDecodeScreenshotPayload:
    @staticmethod
    def _png_b64() -> str:
        import base64

        # >= 16 bytes after decode -- the decoder rejects shorter
        # payloads as "too small to be a real PNG / JPEG header".
        return base64.b64encode(
            b"\x89PNG\r\n\x1a\nIHDR\x00\x00\x00\x00\x00\x00\x00"
        ).decode("ascii")

    def test_raw_base64(self) -> None:
        data, err = bu._decode_screenshot_payload(self._png_b64())
        assert data is not None
        assert err is None

    def test_data_uri_png(self) -> None:
        data, err = bu._decode_screenshot_payload(
            "data:image/png;base64," + self._png_b64()
        )
        assert data is not None

    def test_data_uri_jpeg(self) -> None:
        data, err = bu._decode_screenshot_payload(
            "data:image/jpeg;base64," + self._png_b64()
        )
        assert data is not None

    def test_data_uri_jpg(self) -> None:
        data, err = bu._decode_screenshot_payload(
            "data:image/jpg;base64," + self._png_b64()
        )
        assert data is not None

    def test_whitespace_in_body_tolerated(self) -> None:
        # base64 wrapped onto multiple lines (CLI sometimes does this).
        b = self._png_b64()
        wrapped = "\n".join([b[i : i + 8] for i in range(0, len(b), 8)])
        data, err = bu._decode_screenshot_payload(wrapped)
        assert data is not None

    def test_empty(self) -> None:
        data, err = bu._decode_screenshot_payload("")
        assert data is None
        assert err is not None

    def test_invalid_base64(self) -> None:
        data, err = bu._decode_screenshot_payload("@@@@")
        assert data is None
        assert err is not None

    def test_too_small(self) -> None:
        import base64

        too_small = base64.b64encode(b"abc").decode("ascii")
        data, err = bu._decode_screenshot_payload(too_small)
        assert data is None
        assert err is not None
        assert "too small" in err


# ===========================================================================
# Batch 3 -- T3 YELLOW JS eval (static analysis + two-phase approval)
# ===========================================================================


# ---------------------------------------------------------------------------
# Static analysis -- analyze_js_script
# ---------------------------------------------------------------------------


class TestAnalyzeJsScript:
    def test_empty_is_safe(self) -> None:
        a = bu.analyze_js_script("")
        assert a.requires_two_phase is False
        assert a.risky_markers == ()
        assert a.categories == ()
        assert a.char_count == 0

    def test_whitespace_only_is_safe(self) -> None:
        a = bu.analyze_js_script("   \n  ")
        assert a.requires_two_phase is False
        assert a.risky_markers == ()

    def test_read_only_is_safe(self) -> None:
        a = bu.analyze_js_script("document.title")
        assert a.requires_two_phase is False
        assert a.risky_markers == ()
        assert a.script_preview == "document.title"

    def test_property_read_is_safe(self) -> None:
        # A read of document.cookie is safe; only the assignment is risky.
        a = bu.analyze_js_script("console.log(document.cookie)")
        assert a.requires_two_phase is False

    def test_equality_check_is_safe(self) -> None:
        # ``==`` / ``===`` comparisons must NOT trigger the
        # ``document.cookie = ...`` assignment pattern.
        for body in [
            "document.cookie === 'foo'",
            "document.cookie==='bar'",
            "window.location === 'x'",
            "if(document.cookie==myVar){return 1}",
        ]:
            a = bu.analyze_js_script(body)
            assert a.requires_two_phase is False, body

    def test_fetch_call_detected(self) -> None:
        a = bu.analyze_js_script("fetch('/api/data').then(r => r.json())")
        assert a.requires_two_phase is True
        assert "fetch() call" in a.risky_markers
        assert "network_egress" in a.categories

    def test_xmlhttprequest_detected(self) -> None:
        a = bu.analyze_js_script("const x = new XMLHttpRequest()")
        assert a.requires_two_phase is True
        assert "network_egress" in a.categories

    def test_sendbeacon_detected(self) -> None:
        a = bu.analyze_js_script(
            "navigator.sendBeacon('https://x.com', payload)"
        )
        assert a.requires_two_phase is True
        assert "network_egress" in a.categories

    def test_websocket_detected(self) -> None:
        a = bu.analyze_js_script("new WebSocket('wss://attacker.com')")
        assert a.requires_two_phase is True
        assert "network_egress" in a.categories

    def test_rtc_peer_connection_detected(self) -> None:
        a = bu.analyze_js_script("new RTCPeerConnection({iceServers: []})")
        assert a.requires_two_phase is True
        assert "network_egress" in a.categories

    def test_localstorage_setitem_detected(self) -> None:
        a = bu.analyze_js_script("localStorage.setItem('k', 'v')")
        assert a.requires_two_phase is True
        assert "storage_write" in a.categories

    def test_sessionstorage_setitem_detected(self) -> None:
        a = bu.analyze_js_script("sessionStorage.setItem('k', 'v')")
        assert a.requires_two_phase is True
        assert "storage_write" in a.categories

    def test_cookie_assignment_detected(self) -> None:
        for body in [
            "document.cookie = 'session=abc'",
            "document.cookie='session=abc'",
            "document.cookie  =  'foo'",
        ]:
            a = bu.analyze_js_script(body)
            assert a.requires_two_phase is True, body
            assert "storage_write" in a.categories

    def test_location_assignment_detected(self) -> None:
        a = bu.analyze_js_script("window.location = 'https://elsewhere.com'")
        assert a.requires_two_phase is True
        assert "navigation" in a.categories

    def test_location_replace_detected(self) -> None:
        a = bu.analyze_js_script(
            "window.location.replace('https://elsewhere.com')"
        )
        assert a.requires_two_phase is True
        assert "navigation" in a.categories

    def test_location_assign_detected(self) -> None:
        a = bu.analyze_js_script("window.location.assign('https://x.com')")
        assert a.requires_two_phase is True

    def test_location_href_detected(self) -> None:
        a = bu.analyze_js_script("window.location.href('https://x.com')")
        assert a.requires_two_phase is True

    def test_document_location_detected(self) -> None:
        a = bu.analyze_js_script("document.location = '/foo'")
        assert a.requires_two_phase is True
        assert "navigation" in a.categories

    def test_eval_detected(self) -> None:
        a = bu.analyze_js_script("eval(decoded)")
        assert a.requires_two_phase is True
        assert "second_order_eval" in a.categories

    def test_new_function_detected(self) -> None:
        a = bu.analyze_js_script("const f = new Function('x', 'return x*2')")
        assert a.requires_two_phase is True
        assert "second_order_eval" in a.categories

    def test_dynamic_import_detected(self) -> None:
        a = bu.analyze_js_script("import('https://x.com/mod.js')")
        assert a.requires_two_phase is True
        assert "second_order_eval" in a.categories

    def test_document_write_detected(self) -> None:
        a = bu.analyze_js_script("document.write('<script>...')")
        assert a.requires_two_phase is True
        assert "second_order_eval" in a.categories

    def test_multiple_markers_dedup(self) -> None:
        script = "fetch('/a'); fetch('/b'); fetch('/c')"
        a = bu.analyze_js_script(script)
        # Description deduped: only one "fetch() call" entry.
        assert a.risky_markers.count("fetch() call") == 1

    def test_multiple_categories_in_order(self) -> None:
        script = (
            "fetch('/a'); localStorage.setItem('k','v'); "
            "window.location = '/x'"
        )
        a = bu.analyze_js_script(script)
        # Categories appear in catalog scan order: network_egress
        # then storage_write then navigation.
        assert "network_egress" in a.categories
        assert "storage_write" in a.categories
        assert "navigation" in a.categories

    def test_identifier_substring_not_matched(self) -> None:
        # ``\b`` boundaries -- ``mySafeFetch`` should not trip ``fetch``.
        a = bu.analyze_js_script("const r = mySafeFetch('/api')")
        assert a.requires_two_phase is False

    def test_script_preview_is_capped(self) -> None:
        a = bu.analyze_js_script("a" * 1000)
        assert len(a.script_preview) <= 201  # cap + ellipsis
        assert a.char_count == 1000

    def test_script_preview_collapses_newlines(self) -> None:
        a = bu.analyze_js_script("line1\nline2\n\tline3")
        assert "\n" not in a.script_preview
        assert "line1 line2 line3" in a.script_preview


# ---------------------------------------------------------------------------
# Fake approval registry
# ---------------------------------------------------------------------------


class _FakeApprovalRegistry:
    """Records every register() call and returns scripted handles.

    Default: returns an ApprovalHandle with a deterministic
    approval_id and no pre_resolved decision.
    """

    def __init__(self) -> None:
        from ultron.safety.two_phase_approval import ApprovalHandle

        self.registrations: list = []
        self._next_id = 0
        self._pre_resolved: Any = None

    def set_pre_resolved(self, decision: Any) -> None:
        self._pre_resolved = decision

    def register(self, request: Any) -> Any:
        from ultron.safety.two_phase_approval import ApprovalHandle

        self.registrations.append(request)
        self._next_id += 1
        approval_id = f"test-approval-{self._next_id}"
        return ApprovalHandle(
            approval_id=approval_id,
            expires_at_seconds=999_999.0,
            request=request,
            pre_resolved=self._pre_resolved,
        )


# ---------------------------------------------------------------------------
# BrowserUseTool.eval
# ---------------------------------------------------------------------------


class TestEvalArgumentValidation:
    def test_empty_script_rejected(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        result = tool.eval("", user_text="eval")
        assert result.success is False
        assert result.error == "empty script"
        assert fake_subprocess.calls == []
        assert fake_validator.call_count == 0

    def test_whitespace_only_rejected(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        result = tool.eval("   \n\t", user_text="eval")
        assert result.success is False
        assert fake_subprocess.calls == []


class TestEvalSafeScriptPath:
    def test_safe_script_executes(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        fake_subprocess.stdout = '"My Page Title"'
        result = tool.eval(
            "document.title",
            user_text="get the title",
        )
        assert result.success is True
        assert result.requires_two_phase is False
        assert result.value == "My Page Title"
        assert result.raw_result == '"My Page Title"'
        cmd = fake_subprocess.calls[0].cmd
        assert cmd[-2:] == ["eval", "document.title"]

    def test_safe_script_json_array_value(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        fake_subprocess.stdout = "[1, 2, 3]"
        result = tool.eval(
            "Array.from(document.images).map(i=>i.width)",
            user_text="image widths",
        )
        assert result.success is True
        assert result.value == [1, 2, 3]

    def test_safe_script_non_json_value(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        fake_subprocess.stdout = "not json output"
        result = tool.eval("foo", user_text="eval")
        assert result.success is True
        assert result.value is None
        assert result.raw_result == "not json output"

    def test_safe_script_safety_validator_block(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        fake_validator.block(message="K1 reserved")
        result = tool.eval("document.title", user_text="eval")
        assert result.success is False
        assert result.safety_verdict == "BLOCK_HARD"
        assert fake_subprocess.calls == []

    def test_safe_script_validator_arguments_carry_analysis(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        tool.eval("document.title", user_text="eval")
        ctx = fake_validator.contexts[0]
        assert ctx.tool_name == "desktop.browser_use.eval"
        assert "script_preview" in ctx.arguments
        assert ctx.arguments["risky_markers"] == []
        assert ctx.arguments["categories"] == []
        assert ctx.arguments["assume_preapproved"] is False


class TestEvalRiskyScriptApprovalFlow:
    def test_risky_script_blocks_at_approval_phase(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        # No registry monkeypatch -> uses default get_approval_registry.
        # The default registry returns a handle but our pre-resolver
        # is None so no decision is cached.
        result = tool.eval(
            "fetch('/api/data')",
            user_text="get data",
        )
        assert result.success is False
        assert result.requires_two_phase is True
        assert "fetch() call" in result.risky_markers
        assert "network_egress" in result.categories
        assert result.approval_request_id != ""
        # Subprocess + validator MUST NOT have been touched.
        assert fake_subprocess.calls == []
        assert fake_validator.call_count == 0

    def test_risky_script_uses_injected_registry(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        registry = _FakeApprovalRegistry()
        result = tool.eval(
            "document.cookie = 'x=1'",
            user_text="set cookie",
            approval_registry=registry,
        )
        assert result.requires_two_phase is True
        assert len(registry.registrations) == 1
        req = registry.registrations[0]
        assert req.kind == bu.BROWSER_JS_APPROVAL_KIND
        assert req.actor == "desktop_browser_use"
        assert req.delivery_channel == "voice"
        assert req.metadata["reason_code"] == bu.BROWSER_JS_REASON_CODE
        assert "cookie" in req.prompt.lower() or "storage" in req.prompt.lower()
        assert result.approval_request_id == "test-approval-1"

    def test_risky_script_prompt_is_speakable(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        registry = _FakeApprovalRegistry()
        tool.eval(
            "fetch('/a'); localStorage.setItem('k','v')",
            user_text="dual",
            approval_registry=registry,
        )
        prompt = registry.registrations[0].prompt
        # No script body in the prompt; only humanised categories.
        assert "fetch" not in prompt
        assert "localStorage" not in prompt
        assert "Proceed?" in prompt
        assert prompt.startswith("Browser script wants to ")

    def test_risky_script_metadata_carries_analysis(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        registry = _FakeApprovalRegistry()
        tool.eval(
            "fetch('/a'); document.cookie = 'x=1'",
            user_text="risky",
            approval_registry=registry,
        )
        meta = registry.registrations[0].metadata
        assert "fetch() call" in meta["risky_markers"]
        assert "document.cookie assignment" in meta["risky_markers"]
        assert "script_preview" in meta
        assert meta["char_count"] > 0
        assert meta["user_text"] == "risky"

    def test_risky_script_scope_key_defaults_to_session(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
    ) -> None:
        registry = _FakeApprovalRegistry()
        sess_tool = bu.BrowserUseTool(session="agent-3")
        sess_tool.eval(
            "fetch('/x')",
            user_text="risky",
            approval_registry=registry,
        )
        assert registry.registrations[0].scope_key == "agent-3"

    def test_risky_script_scope_key_explicit_wins(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
    ) -> None:
        registry = _FakeApprovalRegistry()
        sess_tool = bu.BrowserUseTool(session="agent-3")
        sess_tool.eval(
            "fetch('/x')",
            user_text="risky",
            approval_registry=registry,
            approval_scope_key="custom-scope",
        )
        assert registry.registrations[0].scope_key == "custom-scope"

    def test_risky_script_timeout_forwarded(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        registry = _FakeApprovalRegistry()
        tool.eval(
            "fetch('/x')",
            user_text="risky",
            approval_registry=registry,
            approval_timeout_s=12.0,
        )
        assert registry.registrations[0].timeout_seconds == 12.0


class TestEvalPreapprovedPath:
    def test_preapproved_skips_approval_registry(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        registry = _FakeApprovalRegistry()
        fake_subprocess.stdout = "true"
        result = tool.eval(
            "document.cookie = 'x=1'",
            user_text="approved set cookie",
            assume_preapproved=True,
            approval_registry=registry,
        )
        # Risky markers still recorded, but registry never consulted.
        assert result.requires_two_phase is True
        assert len(registry.registrations) == 0
        # Subprocess + validator ran.
        assert len(fake_subprocess.calls) == 1
        assert fake_validator.call_count == 1
        assert result.success is True

    def test_preapproved_with_validator_block(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        # Preapproval doesn't override the safety validator -- if
        # category K (self-protection) or similar fires, the call is
        # still blocked.
        fake_validator.block(message="K-protected")
        result = tool.eval(
            "fetch('/x')",
            user_text="approved",
            assume_preapproved=True,
        )
        assert result.success is False
        assert result.safety_verdict == "BLOCK_HARD"
        assert fake_subprocess.calls == []

    def test_preapproved_arguments_flagged(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        tool.eval(
            "fetch('/x')",
            user_text="approved",
            assume_preapproved=True,
        )
        ctx = fake_validator.contexts[0]
        assert ctx.arguments["assume_preapproved"] is True
        assert ctx.arguments["risky_markers"] == ["fetch() call"]


class TestEvalSubprocessFailure:
    def test_subprocess_failure_propagates_with_analysis(
        self,
        fake_subprocess: _FakeSubprocess,
        fake_validator: _FakeValidator,
        tool: bu.BrowserUseTool,
    ) -> None:
        fake_subprocess.returncode = 1
        fake_subprocess.stderr = "no browser open"
        result = tool.eval("document.title", user_text="eval")
        assert result.success is False
        assert "no browser open" in (result.error or "")
        # Analysis fields still populated on failure.
        assert result.script_preview == "document.title"
        assert result.requires_two_phase is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHumanizeCategories:
    def test_empty(self) -> None:
        assert bu._humanize_categories(()) == "execute custom code"

    def test_single(self) -> None:
        assert (
            bu._humanize_categories(("network_egress",))
            == "make network requests"
        )

    def test_two(self) -> None:
        assert (
            bu._humanize_categories(("network_egress", "storage_write"))
            == "make network requests and write to storage or cookies"
        )

    def test_three(self) -> None:
        out = bu._humanize_categories(
            ("network_egress", "storage_write", "navigation")
        )
        assert "make network requests" in out
        assert "write to storage or cookies" in out
        assert "navigate to another page" in out
        # Oxford comma style for 3+ items.
        assert ", and " in out

    def test_unknown_category_passes_through(self) -> None:
        out = bu._humanize_categories(("network_egress", "mystery"))
        assert "make network requests" in out
        assert "mystery" in out


class TestTryParseEvalPayload:
    def test_empty(self) -> None:
        value, raw = bu._try_parse_eval_payload("")
        assert value is None
        assert raw == ""

    def test_whitespace_only(self) -> None:
        value, raw = bu._try_parse_eval_payload("   \n  ")
        assert value is None
        assert raw == ""

    def test_json_string(self) -> None:
        value, raw = bu._try_parse_eval_payload('"hello"')
        assert value == "hello"
        assert raw == '"hello"'

    def test_json_number(self) -> None:
        value, raw = bu._try_parse_eval_payload("42")
        assert value == 42

    def test_json_boolean(self) -> None:
        value, _ = bu._try_parse_eval_payload("true")
        assert value is True

    def test_json_null(self) -> None:
        value, raw = bu._try_parse_eval_payload("null")
        assert value is None
        assert raw == "null"

    def test_json_array(self) -> None:
        value, _ = bu._try_parse_eval_payload("[1, 2, 3]")
        assert value == [1, 2, 3]

    def test_json_object(self) -> None:
        value, _ = bu._try_parse_eval_payload('{"key": "val"}')
        assert value == {"key": "val"}

    def test_non_json_returns_raw(self) -> None:
        value, raw = bu._try_parse_eval_payload("not valid json")
        assert value is None
        assert raw == "not valid json"
