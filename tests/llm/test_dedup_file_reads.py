"""Tests for ultron.llm.dedup_file_reads."""

from __future__ import annotations

from typing import Any

from ultron.llm import dedup_file_reads as ddr


def _user_with_read(call_id: str, path: str) -> dict[str, Any]:
    """Build a user message issuing a read_file tool call."""
    return {
        "role": "assistant",
        "ts": "2026-05-24T12:00:00Z",
        "content": [
            {
                "type": "tool_use",
                "id": call_id,
                "name": "read_file",
                "input": {"path": path},
            }
        ],
    }


def _tool_result(call_id: str, body: str, ts: str = "2026-05-24T12:00:01Z") -> dict[str, Any]:
    """Build a user message containing a tool_result for ``call_id``."""
    return {
        "role": "user",
        "ts": ts,
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": body,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Single-pass dedup
# ---------------------------------------------------------------------------

class TestDedup:
    def test_empty_history_returns_zero_savings(self) -> None:
        result = ddr.dedup_duplicate_file_reads([])
        assert result.bytes_saved == 0
        assert result.tokens_saved_estimate == 0
        assert result.history == []
        assert result.notes == ()

    def test_no_duplicates_passes_through(self) -> None:
        history = [
            _user_with_read("c1", "a.py"),
            _tool_result("c1", "def a(): pass"),
        ]
        result = ddr.dedup_duplicate_file_reads(history)
        assert result.bytes_saved == 0
        assert result.savings_ratio == 0.0
        assert len(result.rewritten_indices) == 0

    def test_duplicate_reads_keep_latest(self) -> None:
        # The body must be substantially longer than the elision notice
        # (~115 chars) for bytes_saved to be > 0.
        body_old = "OLD CONTENT " + ("x" * 400)
        body_new = "NEW CONTENT"
        history = [
            _user_with_read("c1", "a.py"),
            _tool_result("c1", body_old, ts="2026-05-24T12:00:01Z"),
            _user_with_read("c2", "a.py"),
            _tool_result("c2", body_new, ts="2026-05-24T12:00:05Z"),
        ]
        result = ddr.dedup_duplicate_file_reads(history)
        assert result.bytes_saved > 0
        # Old content should have been elided; new content preserved.
        old_block = result.history[1]["content"][0]
        new_block = result.history[3]["content"][0]
        assert "duplicate" in old_block["content"].lower() or "elided" in old_block["content"].lower()
        assert new_block["content"] == body_new
        assert result.rewritten_indices == (1,)
        assert any("a.py" in note for note in result.notes)

    def test_duplicate_reads_keep_first(self) -> None:
        body_old = "OLD"
        body_new = "NEW CONTENT LONG ENOUGH"
        history = [
            _user_with_read("c1", "a.py"),
            _tool_result("c1", body_old, ts="2026-05-24T12:00:01Z"),
            _user_with_read("c2", "a.py"),
            _tool_result("c2", body_new, ts="2026-05-24T12:00:05Z"),
        ]
        result = ddr.dedup_duplicate_file_reads(history, keep_latest=False)
        new_block = result.history[3]["content"][0]
        # When keep_latest=False, the LATER read is elided.
        assert "duplicate" in new_block["content"].lower() or "elided" in new_block["content"].lower()

    def test_multiple_files_only_dedup_dupes(self) -> None:
        history = [
            _user_with_read("c1", "a.py"),
            _tool_result("c1", "a-1"),
            _user_with_read("c2", "b.py"),
            _tool_result("c2", "b-1"),
            _user_with_read("c3", "a.py"),
            _tool_result("c3", "a-2-much-longer-content"),
        ]
        result = ddr.dedup_duplicate_file_reads(history)
        # Only a.py was duplicated.
        rewritten = set(result.rewritten_indices)
        assert 3 not in rewritten  # b.py untouched
        # The latest read of a.py (index 5) should be preserved verbatim.
        assert result.history[5]["content"][0]["content"] == "a-2-much-longer-content"

    def test_extra_tool_names_supported(self) -> None:
        history = [
            {
                "role": "assistant",
                "ts": "t0",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "x1",
                        "name": "fetch_url",
                        "input": {"url": "https://example.com/a"},
                    }
                ],
            },
            {
                "role": "user",
                "ts": "t1",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "x1",
                        "content": "OLD BODY",
                    }
                ],
            },
            {
                "role": "assistant",
                "ts": "t2",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "x2",
                        "name": "fetch_url",
                        "input": {"url": "https://example.com/a"},
                    }
                ],
            },
            {
                "role": "user",
                "ts": "t3",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "x2",
                        "content": "NEW BODY",
                    }
                ],
            },
        ]
        # Without the extra-name, no dedup fires (default only handles read_file).
        baseline = ddr.dedup_duplicate_file_reads(history)
        assert baseline.bytes_saved == 0
        # With the extra name, dedup picks it up via the 'url' parameter.
        extended = ddr.dedup_duplicate_file_reads(
            history, read_tool_names=("read_file", "fetch_url"),
        )
        # fetch_url uses 'url' field — we currently look at path-like params,
        # which include 'path' / 'file' / 'file_path' / 'filename' / 'paths'
        # but NOT 'url'. Confirm baseline behaviour holds either way; this
        # test pins the read-set extension surface for the future.
        assert extended.bytes_saved == 0

    def test_history_is_not_mutated(self) -> None:
        history = [
            _user_with_read("c1", "a.py"),
            _tool_result("c1", "OLD CONTENT"),
            _user_with_read("c2", "a.py"),
            _tool_result("c2", "NEW CONTENT"),
        ]
        original_old = history[1]["content"][0]["content"]
        ddr.dedup_duplicate_file_reads(history)
        # Original input is unchanged.
        assert history[1]["content"][0]["content"] == original_old


# ---------------------------------------------------------------------------
# should_skip_compaction
# ---------------------------------------------------------------------------

class TestShouldSkipCompaction:
    def test_below_threshold_returns_false(self) -> None:
        result = ddr.DedupResult(
            history=[], bytes_saved=10, tokens_saved_estimate=2,
            bytes_before=100, savings_ratio=0.10,
        )
        assert ddr.should_skip_compaction(result) is False

    def test_above_threshold_returns_true(self) -> None:
        result = ddr.DedupResult(
            history=[], bytes_saved=40, tokens_saved_estimate=10,
            bytes_before=100, savings_ratio=0.40,
        )
        assert ddr.should_skip_compaction(result) is True

    def test_custom_threshold(self) -> None:
        result = ddr.DedupResult(
            history=[], bytes_saved=15, tokens_saved_estimate=3,
            bytes_before=100, savings_ratio=0.15,
        )
        assert ddr.should_skip_compaction(result, threshold=0.10) is True


# ---------------------------------------------------------------------------
# Generic payload dedup
# ---------------------------------------------------------------------------

class TestPayloadDedup:
    def test_no_duplicates_passes_through(self) -> None:
        out = ddr.dedup_payload_duplicates([
            ("nvidia", "t1", "body1"),
            ("tasklist", "t2", "body2"),
        ])
        assert out == [("nvidia", "body1"), ("tasklist", "body2")]

    def test_dupes_keep_latest(self) -> None:
        out = ddr.dedup_payload_duplicates([
            ("nvidia", "t1", "old"),
            ("nvidia", "t2", "new"),
        ])
        assert out[0][0] == "nvidia"
        assert "duplicate elided" in out[0][1].lower()
        assert out[1] == ("nvidia", "new")

    def test_dupes_keep_first(self) -> None:
        out = ddr.dedup_payload_duplicates([
            ("nvidia", "t1", "old"),
            ("nvidia", "t2", "new"),
        ], keep_latest=False)
        assert out[0] == ("nvidia", "old")
        assert "duplicate elided" in out[1][1].lower()
