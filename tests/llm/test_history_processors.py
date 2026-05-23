"""Tests for the LLM history processors (catalog T2 + T9)."""

from __future__ import annotations

import pytest

from ultron.llm.history_processors import (
    DEFAULT_CLOSED_WINDOW_TEMPLATE,
    DEFAULT_OBSERVATION_ELISION_TEMPLATE,
    ClosedWindowHistoryProcessor,
    LastNObservations,
    TagToolCallObservations,
    apply_history_processors,
    build_default_processors,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _file_view(path: str, n_lines: int, *, line_offset: int = 1) -> str:
    """Build a fake SWE-Agent-style file-view string."""
    lines = [f"[File: {path} ({n_lines} lines total)]"]
    for i in range(n_lines):
        lines.append(f"{line_offset + i}:   some content here line {i}")
    return "\n".join(lines)


def _user(content: str, *, is_demo: bool = False) -> dict:
    item: dict = {"role": "user", "content": content}
    if is_demo:
        item["is_demo"] = True
    return item


def _observation(content: str, *, tags: list[str] | None = None, image_count: int = 0) -> dict:
    if image_count > 0:
        segs = [{"type": "text", "text": content}]
        segs.extend(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,XYZ{i}"}}
            for i in range(image_count)
        )
        c = segs
    else:
        c = content
    item: dict = {"role": "tool", "content": c, "message_type": "observation"}
    if tags:
        item["tags"] = list(tags)
    return item


def _action(tool_name: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "message_type": "action",
        "tool_calls": [{"function": {"name": tool_name, "arguments": "{}"}}],
    }


# ===========================================================================
# T2: ClosedWindowHistoryProcessor
# ===========================================================================


def test_closed_window_empty_history_returns_empty():
    proc = ClosedWindowHistoryProcessor()
    assert proc([]) == []


def test_closed_window_no_file_views_passthrough():
    proc = ClosedWindowHistoryProcessor()
    history = [
        {"role": "system", "content": "you are helpful"},
        _user("Hi"),
        {"role": "assistant", "content": "Hello"},
        _user("How are you?"),
    ]
    out = proc(history)
    assert len(out) == len(history)
    for i, item in enumerate(out):
        assert item["content"] == history[i]["content"]


def test_closed_window_single_file_view_kept():
    proc = ClosedWindowHistoryProcessor()
    history = [_user(_file_view("/tmp/foo.py", 5))]
    out = proc(history)
    # Only one snapshot -> kept verbatim.
    assert out[0]["content"].count("[File: /tmp/foo.py") == 1
    assert "1:" in out[0]["content"]
    assert "5:" in out[0]["content"]


def test_closed_window_repeated_file_collapses_older():
    proc = ClosedWindowHistoryProcessor()
    history = [
        _user(_file_view("/tmp/foo.py", 5)),  # older snapshot -- elided
        _user("Some chatter in between"),
        _user(_file_view("/tmp/foo.py", 5)),  # newer snapshot -- kept
    ]
    out = proc(history)
    # The OLDER (first) view's line blocks should be replaced with the summary.
    assert "Outdated window with 5 lines omitted" in out[0]["content"]
    # The HEADER survives so the model still knows what was elided.
    assert "[File: /tmp/foo.py" in out[0]["content"]
    # Per-line content of the old block is gone.
    assert "1:" not in out[0]["content"]
    # Chatter in the middle is untouched.
    assert out[1]["content"] == "Some chatter in between"
    # Newer snapshot intact.
    assert "1:" in out[2]["content"]
    assert "5:" in out[2]["content"]


def test_closed_window_different_files_dont_interfere():
    proc = ClosedWindowHistoryProcessor()
    history = [
        _user(_file_view("/tmp/foo.py", 3)),
        _user(_file_view("/tmp/bar.py", 3)),
        _user(_file_view("/tmp/foo.py", 3)),  # newer foo -- collapses index 0
    ]
    out = proc(history)
    # First foo: collapsed.
    assert "Outdated window" in out[0]["content"]
    # bar.py: still verbatim (only one snapshot).
    assert "1:" in out[1]["content"]
    # Second foo: verbatim.
    assert "1:" in out[2]["content"]


def test_closed_window_assistant_turns_not_modified():
    proc = ClosedWindowHistoryProcessor()
    fake_assistant_view = _file_view("/tmp/foo.py", 3)
    history = [
        _user(_file_view("/tmp/foo.py", 3)),
        {"role": "assistant", "content": fake_assistant_view},
        _user(_file_view("/tmp/foo.py", 3)),
    ]
    out = proc(history)
    # Assistant content is left alone even though it looks like a file view.
    assert out[1]["content"] == fake_assistant_view


def test_closed_window_demo_items_passthrough():
    proc = ClosedWindowHistoryProcessor()
    history = [
        _user(_file_view("/tmp/foo.py", 3), is_demo=True),  # demo -- not collapsed
        _user(_file_view("/tmp/foo.py", 3)),
    ]
    out = proc(history)
    assert "1:" in out[0]["content"]
    assert "1:" in out[1]["content"]


def test_closed_window_template_override():
    proc = ClosedWindowHistoryProcessor(
        template="[ELIDED N={n_lines}]\n"
    )
    history = [
        _user(_file_view("/tmp/foo.py", 4)),
        _user(_file_view("/tmp/foo.py", 4)),
    ]
    out = proc(history)
    assert "[ELIDED N=4]" in out[0]["content"]


def test_closed_window_disabled_passes_through():
    proc = ClosedWindowHistoryProcessor(enabled=False)
    history = [
        _user(_file_view("/tmp/foo.py", 3)),
        _user(_file_view("/tmp/foo.py", 3)),
    ]
    out = proc(history)
    # Disabled: both snapshots survive verbatim.
    assert "1:" in out[0]["content"]
    assert "1:" in out[1]["content"]


def test_closed_window_preserves_input_objects():
    proc = ClosedWindowHistoryProcessor()
    history = [
        _user(_file_view("/tmp/foo.py", 3)),
        _user(_file_view("/tmp/foo.py", 3)),
    ]
    original = history[0]["content"]
    out = proc(history)
    # Input list NOT mutated -- the in-place item is unchanged.
    assert history[0]["content"] == original
    assert out[0] is not history[0]


def test_closed_window_default_template_constant():
    assert "{n_lines}" in DEFAULT_CLOSED_WINDOW_TEMPLATE
    assert "Outdated" in DEFAULT_CLOSED_WINDOW_TEMPLATE


# ===========================================================================
# T9: LastNObservations
# ===========================================================================


def test_last_n_empty_history_returns_empty():
    proc = LastNObservations(n=3)
    assert proc([]) == []


def test_last_n_validates_n_positive():
    with pytest.raises(ValueError):
        LastNObservations(n=0)
    with pytest.raises(ValueError):
        LastNObservations(n=-1)


def test_last_n_validates_polling_positive():
    with pytest.raises(ValueError):
        LastNObservations(n=3, polling=0)


def test_last_n_keeps_last_n_observations_verbatim():
    proc = LastNObservations(n=2)
    history = [
        _observation("obs 0 -- instance template"),  # always kept (index 0)
        _action("ls"),
        _observation("obs 1 -- old"),
        _action("ls"),
        _observation("obs 2 -- old"),
        _action("ls"),
        _observation("obs 3 -- recent"),
        _action("ls"),
        _observation("obs 4 -- recent"),
    ]
    out = proc(history)
    # Last 2 observations are kept verbatim.
    assert out[-1]["content"] == "obs 4 -- recent"
    assert out[-3]["content"] == "obs 3 -- recent"
    # Index 0 (instance template) ALWAYS preserved.
    assert "instance template" in out[0]["content"]
    # Middle observations elided.
    assert "Old environment output" in out[2]["content"]
    assert "Old environment output" in out[4]["content"]


def test_last_n_polling_keeps_window_stable():
    """With polling > 1, the elision window doesn't slide every turn."""
    base = [
        _observation("instance template"),
        _action("ls"),
        _observation("obs 1"),
        _action("ls"),
        _observation("obs 2"),
        _action("ls"),
        _observation("obs 3"),
    ]
    # With n=1, polling=1 (naive sliding): only the last observation
    # survives at every history length.
    proc_naive = LastNObservations(n=1, polling=1)
    out_naive = proc_naive(base)
    # Naive: obs 1 and obs 2 elided (one observation -- the most recent --
    # survives).
    assert "Old environment output" in out_naive[2]["content"]
    assert "Old environment output" in out_naive[4]["content"]
    assert out_naive[-1]["content"] == "obs 3"

    # With polling=4, the elision window rounds DOWN to multiples of 4,
    # so with 4 observations (one instance + 3 real) the
    # ``len(obs) // polling * polling = 4`` and ``last_removed_idx =
    # max(0, 4 - 1) = 3``; obs at indices [1:3] = obs 1 + obs 2 are
    # elided. With one fewer observation, the multiple-of-4 floor
    # would drop to 0 so NOTHING gets elided -- the cache stays
    # warm across that boundary.
    proc_polled = LastNObservations(n=1, polling=4)
    out_polled = proc_polled(base)
    # obs 3 survives, obs 1 + 2 elided per the polling math.
    assert out_polled[-1]["content"] == "obs 3"


def test_last_n_skips_first_observation():
    """The instance-template observation (index 0) is always kept."""
    proc = LastNObservations(n=1)
    history = [
        _observation("template"),
        _action("ls"),
        _observation("obs 1"),
        _action("ls"),
        _observation("obs 2"),
    ]
    out = proc(history)
    # First observation MUST survive regardless of n.
    assert out[0]["content"] == "template"


def test_last_n_keep_tag_overrides_elision():
    proc = LastNObservations(n=1)
    history = [
        _observation("template"),
        _action("ls"),
        _observation("obs 1 -- precious", tags=["keep_output"]),
        _action("ls"),
        _observation("obs 2 -- recent"),
    ]
    out = proc(history)
    # The keep_output tag preserves obs 1 even though it falls in the
    # elision window.
    assert out[2]["content"] == "obs 1 -- precious"
    # obs 2 survives by virtue of being the last.
    assert out[-1]["content"] == "obs 2 -- recent"


def test_last_n_remove_tag_always_elides():
    proc = LastNObservations(n=5)  # n large enough that nothing would normally elide
    history = [
        _observation("template"),
        _action("ls"),
        _observation("obs 1"),
        _action("ls"),
        _observation("obs 2 -- huge image", tags=["remove_output"], image_count=2),
    ]
    out = proc(history)
    # remove_output tag forces elision regardless of recency.
    assert "Old environment output" in out[-1]["content"]
    # And the image count appears in the elision message.
    assert "2 images omitted" in out[-1]["content"]


def test_last_n_elision_reports_line_count():
    proc = LastNObservations(n=1)
    multi_line_content = "line a\nline b\nline c"
    history = [
        _observation("template"),
        _action("ls"),
        _observation(multi_line_content),
        _action("ls"),
        _observation("kept"),
    ]
    out = proc(history)
    assert "3 lines omitted" in out[2]["content"]


def test_last_n_default_template_constant():
    assert "{n_lines}" in DEFAULT_OBSERVATION_ELISION_TEMPLATE


def test_last_n_disabled_passes_through():
    proc = LastNObservations(n=1, enabled=False)
    history = [
        _observation("template"),
        _action("ls"),
        _observation("obs 1 -- would be elided"),
        _action("ls"),
        _observation("obs 2"),
    ]
    out = proc(history)
    # Disabled: obs 1 survives.
    assert out[2]["content"] == "obs 1 -- would be elided"


def test_last_n_no_observations_at_all_passthrough():
    proc = LastNObservations(n=2)
    history = [
        {"role": "system", "content": "system"},
        _user("just chatter"),
        {"role": "assistant", "content": "more chatter"},
    ]
    out = proc(history)
    assert [it["content"] for it in out] == ["system", "just chatter", "more chatter"]


def test_last_n_normalises_tag_containers():
    # Accept list/tuple/set in the constructor; normalise to frozenset.
    proc = LastNObservations(
        n=1,
        always_keep_output_for_tags={"keep"},
        always_remove_output_for_tags=["drop"],
    )
    assert "keep" in proc.always_keep_output_for_tags
    assert "drop" in proc.always_remove_output_for_tags


# ===========================================================================
# T9 companion: TagToolCallObservations
# ===========================================================================


def test_tag_tool_call_observations_tags_following_observation():
    proc = TagToolCallObservations(
        tags=frozenset({"keep_output"}),
        function_names=frozenset({"submit"}),
    )
    history = [
        _action("ls"),
        _observation("ls output -- not tagged"),
        _action("submit"),
        _observation("submit output -- tagged"),
    ]
    out = proc(history)
    assert "keep_output" not in (out[1].get("tags") or [])
    assert "keep_output" in (out[3].get("tags") or [])


def test_tag_tool_call_observations_ignores_unmatched_tools():
    proc = TagToolCallObservations(
        tags=frozenset({"remove_output"}),
        function_names=frozenset({"view_image"}),
    )
    history = [
        _action("ls"),
        _observation("ls output"),
        _action("grep"),
        _observation("grep output"),
    ]
    out = proc(history)
    for item in out:
        assert not (item.get("tags") or [])


def test_tag_tool_call_observations_multiple_calls_in_one_turn():
    """When one assistant turn has multiple tool calls, the NEXT
    observation gets all matching tags."""
    proc = TagToolCallObservations(
        tags=frozenset({"keep_output"}),
        function_names=frozenset({"submit"}),
    )
    history = [
        {
            "role": "assistant",
            "content": "",
            "message_type": "action",
            "tool_calls": [
                {"function": {"name": "ls"}},
                {"function": {"name": "submit"}},
            ],
        },
        _observation("combined output"),
    ]
    out = proc(history)
    assert "keep_output" in (out[1].get("tags") or [])


def test_tag_tool_call_observations_legacy_name_shape():
    """Some history shapes carry name at the top-level of the call."""
    proc = TagToolCallObservations(
        tags=frozenset({"keep_output"}),
        function_names=frozenset({"submit"}),
    )
    history = [
        {
            "role": "assistant",
            "content": "",
            "message_type": "action",
            "tool_calls": [{"name": "submit"}],
        },
        _observation("submit output"),
    ]
    out = proc(history)
    assert "keep_output" in (out[1].get("tags") or [])


def test_tag_tool_call_observations_no_observation_after_action_noop():
    proc = TagToolCallObservations(
        tags=frozenset({"keep_output"}),
        function_names=frozenset({"submit"}),
    )
    history = [
        _action("submit"),
        # No observation follows -- the tags simply don't get applied.
        {"role": "assistant", "content": "more thinking"},
    ]
    out = proc(history)
    for item in out:
        assert not (item.get("tags") or [])


def test_tag_tool_call_observations_disabled_passes_through():
    proc = TagToolCallObservations(
        tags=frozenset({"keep_output"}),
        function_names=frozenset({"submit"}),
        enabled=False,
    )
    history = [_action("submit"), _observation("submit output")]
    out = proc(history)
    assert not (out[1].get("tags") or [])


def test_tag_tool_call_observations_existing_tags_preserved():
    proc = TagToolCallObservations(
        tags=frozenset({"keep_output"}),
        function_names=frozenset({"submit"}),
    )
    history = [
        _action("submit"),
        _observation("output", tags=["pre-existing"]),
    ]
    out = proc(history)
    tags = out[1].get("tags") or []
    assert "pre-existing" in tags
    assert "keep_output" in tags


# ===========================================================================
# Composer: apply_history_processors + build_default_processors
# ===========================================================================


def test_apply_history_processors_empty_chain_returns_copy():
    history = [_user("hi")]
    out = apply_history_processors(history, [])
    assert out == history
    assert out is not history


def test_apply_history_processors_runs_in_order():
    def upper(h):
        return [{**item, "content": item["content"].upper()} for item in h]

    def append_x(h):
        return [{**item, "content": item["content"] + "X"} for item in h]

    out = apply_history_processors([_user("hi")], [upper, append_x])
    assert out[0]["content"] == "HIX"


def test_apply_history_processors_swallows_processor_exception():
    def broken(_):
        raise RuntimeError("intentional")

    def working(h):
        return [{**item, "content": "fine"} for item in h]

    history = [_user("hi")]
    # Broken processor logs WARN but doesn't break the chain.
    out = apply_history_processors(history, [broken, working])
    assert out[0]["content"] == "fine"


def test_build_default_processors_default_includes_closed_window():
    procs = build_default_processors()
    assert any(isinstance(p, ClosedWindowHistoryProcessor) for p in procs)


def test_build_default_processors_last_n_optional():
    procs_no_last = build_default_processors()
    assert not any(isinstance(p, LastNObservations) for p in procs_no_last)
    procs_with_last = build_default_processors(last_n=5)
    assert any(isinstance(p, LastNObservations) for p in procs_with_last)


def test_build_default_processors_tags_chain():
    procs = build_default_processors(
        keep_for_tools=["submit"],
        remove_for_tools=["view_image"],
    )
    tag_procs = [p for p in procs if isinstance(p, TagToolCallObservations)]
    assert len(tag_procs) == 2


def test_build_default_processors_disable_closed_window():
    procs = build_default_processors(closed_window_enabled=False)
    assert not any(isinstance(p, ClosedWindowHistoryProcessor) for p in procs)


# ===========================================================================
# End-to-end integration through the chain
# ===========================================================================


def test_chain_collapses_then_elides():
    """ClosedWindow runs before LastN so the latest file snapshot
    survives both compressors."""
    procs = build_default_processors(closed_window_enabled=True, last_n=2)
    history = [
        _observation("template"),
        _action("open"),
        _observation(_file_view("/tmp/foo.py", 4)),  # older view -- closed-window collapses
        _action("ls"),
        _observation("ls out"),
        _action("open"),
        _observation(_file_view("/tmp/foo.py", 4)),  # newer view -- survives last-N
    ]
    out = apply_history_processors(history, procs)
    # The newer foo.py view is the last observation: it survives both.
    assert "1:" in out[-1]["content"]
    # The instance template (index 0) survives.
    assert "template" in out[0]["content"]
