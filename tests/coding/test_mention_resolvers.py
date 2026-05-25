"""Tests for ultron.coding.mention_resolvers."""

from __future__ import annotations

from pathlib import Path

import pytest

from ultron.coding import mention_resolvers as mr


# ---------------------------------------------------------------------------
# find_mentions
# ---------------------------------------------------------------------------

class TestFindMentions:
    def test_empty(self) -> None:
        assert mr.find_mentions("") == []

    def test_url_mention(self) -> None:
        out = mr.find_mentions("see @https://example.com/path?x=1 for more")
        assert "@https://example.com/path?x=1" in out

    def test_workspace_mention(self) -> None:
        out = mr.find_mentions("look at @workspace:frontend/src/index.tsx today")
        assert "@workspace:frontend/src/index.tsx" in out

    def test_memory_mention(self) -> None:
        out = mr.find_mentions("recall @memory:budget-2026 from last week")
        assert "@memory:budget-2026" in out

    def test_bare_path(self) -> None:
        out = mr.find_mentions("fix the bug in @src/foo.py please")
        assert "@src/foo.py" in out

    def test_extension_token(self) -> None:
        out = mr.find_mentions("config @config.yaml is broken")
        assert "@config.yaml" in out

    def test_special_tokens(self) -> None:
        out = mr.find_mentions("review @problems and @diff before @clipboard")
        assert "@problems" in out
        assert "@diff" in out
        assert "@clipboard" in out

    def test_email_address_not_a_mention(self) -> None:
        # foo@bar.com should NOT be picked up as @bar.com.
        out = mr.find_mentions("contact foo@bar.com for help")
        assert all("bar.com" not in m for m in out)


# ---------------------------------------------------------------------------
# resolve_extended_mentions — URL fetcher
# ---------------------------------------------------------------------------

class TestUrlFetcher:
    def test_url_resolved(self) -> None:
        ctx = mr.MentionResolutionContext(
            url_fetcher=lambda url: f"<<body for {url}>>",
        )
        out = mr.resolve_extended_mentions(
            "see @https://example.com/x for details", ctx,
        )
        assert "kind=\"url\"" in out.rewritten_text
        assert "body for https://example.com/x" in out.rewritten_text
        assert out.mentions[0].kind == "url"

    def test_url_no_provider_missing_block(self) -> None:
        ctx = mr.MentionResolutionContext()
        out = mr.resolve_extended_mentions("@https://example.com", ctx)
        assert out.mentions[0].kind == "missing"
        assert "no url_fetcher" in (out.mentions[0].error or "")

    def test_url_fetcher_returns_none(self) -> None:
        ctx = mr.MentionResolutionContext(url_fetcher=lambda _: None)
        out = mr.resolve_extended_mentions("@https://example.com", ctx)
        assert out.mentions[0].kind == "missing"

    def test_url_body_truncated(self) -> None:
        ctx = mr.MentionResolutionContext(
            url_fetcher=lambda _: "x" * 20_000,
            max_body_chars=500,
        )
        out = mr.resolve_extended_mentions("@https://example.com", ctx)
        assert "(truncated)" in out.rewritten_text


# ---------------------------------------------------------------------------
# Memory provider
# ---------------------------------------------------------------------------

class TestMemory:
    def test_memory_resolved(self) -> None:
        ctx = mr.MentionResolutionContext(
            memory_provider=lambda topic, k: [
                ("note-a", f"snippet about {topic} 1"),
                ("note-b", f"snippet about {topic} 2"),
            ],
            memory_top_k=2,
        )
        out = mr.resolve_extended_mentions("recall @memory:budget", ctx)
        assert "kind=\"memory\"" in out.rewritten_text
        assert "snippet about budget 1" in out.rewritten_text
        assert "snippet about budget 2" in out.rewritten_text

    def test_memory_empty_returns_missing(self) -> None:
        ctx = mr.MentionResolutionContext(memory_provider=lambda _t, _k: [])
        out = mr.resolve_extended_mentions("@memory:nope", ctx)
        assert out.mentions[0].kind == "missing"


# ---------------------------------------------------------------------------
# Problems / diff / clipboard / screenshot
# ---------------------------------------------------------------------------

class TestSpecialProviders:
    def test_problems(self) -> None:
        ctx = mr.MentionResolutionContext(
            lint_provider=lambda: "E501 line too long\nW292 no newline at end of file",
        )
        out = mr.resolve_extended_mentions("look at @problems", ctx)
        assert "E501" in out.rewritten_text
        assert "kind=\"problems\"" in out.rewritten_text

    def test_diff(self) -> None:
        ctx = mr.MentionResolutionContext(diff_provider=lambda: "+++ a/foo.py\n+new line")
        out = mr.resolve_extended_mentions("review @diff", ctx)
        assert "+new line" in out.rewritten_text

    def test_clipboard(self) -> None:
        ctx = mr.MentionResolutionContext(clipboard_provider=lambda: "pasted text")
        out = mr.resolve_extended_mentions("use @clipboard", ctx)
        assert "pasted text" in out.rewritten_text

    def test_screenshot(self) -> None:
        ctx = mr.MentionResolutionContext(
            screenshot_provider=lambda: "a chat window with Discord open",
        )
        out = mr.resolve_extended_mentions("check @screenshot", ctx)
        assert "Discord" in out.rewritten_text

    def test_screenshot_empty_missing(self) -> None:
        ctx = mr.MentionResolutionContext(screenshot_provider=lambda: "")
        out = mr.resolve_extended_mentions("@screenshot", ctx)
        assert out.mentions[0].kind == "missing"


# ---------------------------------------------------------------------------
# File / workspace / last
# ---------------------------------------------------------------------------

class TestFileAndWorkspace:
    def test_file_resolved(self, tmp_path: Path) -> None:
        f = tmp_path / "a.py"
        f.write_text("def a(): pass", encoding="utf-8")
        ctx = mr.MentionResolutionContext(
            file_reader=lambda p: Path(p).read_text(encoding="utf-8"),
        )
        out = mr.resolve_extended_mentions(f"edit @{f.as_posix()}", ctx)
        assert "def a(): pass" in out.rewritten_text
        assert out.mentions[0].kind == "file"

    def test_workspace_resolved(self, tmp_path: Path) -> None:
        f = tmp_path / "frontend" / "index.ts"
        f.parent.mkdir(parents=True)
        f.write_text("export const x = 1;", encoding="utf-8")

        def resolver(label: str, rel: str) -> Path | None:
            assert label == "frontend"
            return f if rel == "index.ts" else None

        ctx = mr.MentionResolutionContext(
            workspace_resolver=resolver,
            file_reader=lambda p: Path(p).read_text(encoding="utf-8"),
        )
        out = mr.resolve_extended_mentions("see @workspace:frontend:index.ts", ctx)
        assert "export const x = 1;" in out.rewritten_text

    def test_workspace_missing_resolver(self, tmp_path: Path) -> None:
        ctx = mr.MentionResolutionContext()
        out = mr.resolve_extended_mentions("see @workspace:frontend:index.ts", ctx)
        assert out.mentions[0].kind == "missing"

    def test_last_file(self, tmp_path: Path) -> None:
        f = tmp_path / "last.py"
        f.write_text("# the most recent", encoding="utf-8")
        ctx = mr.MentionResolutionContext(
            last_file_provider=lambda: str(f),
            file_reader=lambda p: Path(p).read_text(encoding="utf-8"),
        )
        out = mr.resolve_extended_mentions("pull up @last", ctx)
        assert "# the most recent" in out.rewritten_text

    def test_file_not_found_missing(self) -> None:
        def bad_reader(_p: Path) -> str:
            raise FileNotFoundError()
        ctx = mr.MentionResolutionContext(file_reader=bad_reader)
        out = mr.resolve_extended_mentions("@src/nope.py", ctx)
        assert out.mentions[0].kind == "missing"


# ---------------------------------------------------------------------------
# Per-call caps + dedup
# ---------------------------------------------------------------------------

class TestCaps:
    def test_per_call_cap_truncates(self) -> None:
        ctx = mr.MentionResolutionContext(
            url_fetcher=lambda url: f"body-{url}",
            max_mentions_per_call=2,
        )
        text = "@https://a.com and @https://b.com and @https://c.com"
        out = mr.resolve_extended_mentions(text, ctx)
        assert out.truncated_count == 1
        assert "deferred past the per-call cap" in out.rewritten_text

    def test_duplicate_mentions_deduped(self) -> None:
        calls: list[str] = []
        def fetcher(url: str) -> str:
            calls.append(url)
            return f"body-{url}"
        ctx = mr.MentionResolutionContext(url_fetcher=fetcher)
        # Two identical mentions in the same call.
        out = mr.resolve_extended_mentions(
            "@https://example.com and again @https://example.com", ctx,
        )
        # Provider only called once.
        assert calls == ["https://example.com"]
        # Both occurrences in the rewritten text receive the body.
        assert out.rewritten_text.count("body-https://example.com") == 2


# ---------------------------------------------------------------------------
# Pass-through behaviour
# ---------------------------------------------------------------------------

class TestPassthrough:
    def test_text_without_mentions(self) -> None:
        ctx = mr.MentionResolutionContext()
        out = mr.resolve_extended_mentions("plain text with no mention", ctx)
        assert out.rewritten_text == "plain text with no mention"
        assert out.mentions == ()

    def test_empty_input(self) -> None:
        ctx = mr.MentionResolutionContext()
        out = mr.resolve_extended_mentions("", ctx)
        assert out.rewritten_text == ""
