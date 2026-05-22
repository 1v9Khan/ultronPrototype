"""Tests for ultron.coding.project_digest."""

from __future__ import annotations

from pathlib import Path

import pytest

from ultron.coding.project_digest import (
    DIGEST_SECTIONS,
    SUMMARY_TEMPLATE,
    DigestRequest,
    ProjectDigest,
    extract_files_from_digest,
    generate_digest,
    parse_digest_sections,
    render_template,
)


# ---------------------------------------------------------------------------
# render_template -- deterministic fallback
# ---------------------------------------------------------------------------


def _basic_request(**overrides) -> DigestRequest:
    defaults = dict(
        project_name="testproj",
        project_path=Path("/tmp/testproj"),
        task_summary="I scaffolded the app and added a route.",
        files_created=[Path("/tmp/testproj/app.py")],
        files_modified=[Path("/tmp/testproj/README.md")],
        user_goal_hint="build a flask app",
    )
    defaults.update(overrides)
    return DigestRequest(**defaults)


def test_render_template_uses_goal_hint() -> None:
    req = _basic_request()
    md = render_template(req)
    assert "build a flask app" in md
    assert "## Goal" in md


def test_render_template_lists_file_changes() -> None:
    req = _basic_request()
    md = render_template(req)
    assert "app.py" in md
    assert "README.md" in md


def test_render_template_includes_all_sections() -> None:
    req = _basic_request()
    md = render_template(req)
    for section in DIGEST_SECTIONS:
        assert f"## {section}" in md, f"missing section {section!r}"


def test_render_template_handles_empty_files() -> None:
    req = DigestRequest(
        project_name="empty",
        project_path=Path("/tmp/empty"),
        task_summary="nothing happened",
    )
    md = render_template(req)
    assert "(none)" in md or "nothing happened" in md


def test_render_template_includes_entry_points() -> None:
    req = _basic_request(entry_points=[Path("/tmp/testproj/manage.py")])
    md = render_template(req)
    assert "manage.py" in md
    assert "entry point" in md


def test_render_template_falls_back_to_default_goal() -> None:
    req = _basic_request(user_goal_hint="")
    md = render_template(req)
    assert "testproj" in md.lower()


def test_render_template_includes_language_in_critical_context() -> None:
    req = _basic_request(language="python")
    md = render_template(req)
    sections = parse_digest_sections(md)
    assert "python" in sections.get("Critical Context", "").lower()


# ---------------------------------------------------------------------------
# parse_digest_sections
# ---------------------------------------------------------------------------


def test_parse_digest_sections_extracts_all_headings() -> None:
    md = render_template(_basic_request())
    sections = parse_digest_sections(md)
    for s in DIGEST_SECTIONS:
        assert s in sections, f"missing parsed section {s!r}"


def test_parse_digest_sections_returns_body_text() -> None:
    md = render_template(_basic_request())
    sections = parse_digest_sections(md)
    assert "build a flask app" in sections["Goal"].lower()


def test_parse_digest_sections_handles_empty() -> None:
    assert parse_digest_sections("") == {}


def test_parse_digest_sections_ignores_subheadings() -> None:
    md = "## Progress\n### Done\n- thing 1\n### In Progress\n- thing 2\n## Goal\n- finish it"
    sections = parse_digest_sections(md)
    assert "Progress" in sections
    assert "Done" in sections["Progress"]
    assert "thing 1" in sections["Progress"]


def test_parse_digest_sections_case_insensitive_header_match() -> None:
    md = "## goal\n- lowercase\n## Constraints & Preferences\n- normal"
    sections = parse_digest_sections(md)
    # "goal" lowercase maps to canonical "Goal".
    assert "Goal" in sections
    assert "lowercase" in sections["Goal"]


# ---------------------------------------------------------------------------
# extract_files_from_digest
# ---------------------------------------------------------------------------


def test_extract_files_from_digest_returns_paths() -> None:
    md = render_template(_basic_request())
    paths = extract_files_from_digest(md)
    # Both file paths should be picked up (modified files appear in Relevant Files).
    assert any("app.py" in p for p in paths)


def test_extract_files_from_digest_handles_none() -> None:
    md = "## Relevant Files\n- (none)\n"
    assert extract_files_from_digest(md) == []


def test_extract_files_from_digest_handles_missing_section() -> None:
    md = "## Goal\n- something\n"
    assert extract_files_from_digest(md) == []


def test_extract_files_from_digest_strips_explanation() -> None:
    md = (
        "## Relevant Files\n"
        "- src/app.py: main entry\n"
        "- README.md: documentation\n"
    )
    paths = extract_files_from_digest(md)
    assert "src/app.py" in paths
    assert "README.md" in paths
    assert "main entry" not in paths  # explanation stripped


# ---------------------------------------------------------------------------
# generate_digest -- with stubbed LLM
# ---------------------------------------------------------------------------


def test_generate_digest_uses_llm_call_when_provided() -> None:
    captured_prompt = []

    def fake_llm(prompt: str) -> str:
        captured_prompt.append(prompt)
        return (
            "## Goal\n- build a thing\n"
            "## Constraints & Preferences\n- (none)\n"
            "## Progress\n### Done\n- did stuff\n### In Progress\n- (none)\n"
            "### Blocked\n- (none)\n"
            "## Key Decisions\n- (none)\n"
            "## Next Steps\n- (none)\n"
            "## Critical Context\n- (none)\n"
            "## Relevant Files\n- (none)\n"
        )

    req = _basic_request()
    digest = generate_digest(req, llm_call=fake_llm)

    assert len(captured_prompt) == 1
    assert digest.source == "llm"
    assert digest.fallback is False
    assert "Goal" in digest.sections
    assert "build a thing" in digest.sections["Goal"]


def test_generate_digest_falls_back_when_llm_raises() -> None:
    def bad_llm(prompt: str) -> str:
        raise RuntimeError("LLM unavailable")

    req = _basic_request()
    digest = generate_digest(req, llm_call=bad_llm)

    assert digest.fallback is True
    assert digest.source == "template"
    assert "build a flask app" in digest.markdown


def test_generate_digest_falls_back_when_llm_returns_empty() -> None:
    req = _basic_request()
    digest = generate_digest(req, llm_call=lambda _: "")

    assert digest.fallback is True
    assert digest.source == "template"


def test_generate_digest_falls_back_when_llm_returns_whitespace() -> None:
    req = _basic_request()
    digest = generate_digest(req, llm_call=lambda _: "   \n\n  ")

    assert digest.fallback is True


def test_generate_digest_no_llm_uses_template() -> None:
    req = _basic_request()
    digest = generate_digest(req, llm_call=None)

    assert digest.fallback is True
    assert digest.source == "template"
    assert digest.elapsed_ms >= 0.0


def test_generate_digest_strips_markdown_code_fence() -> None:
    fenced = (
        "```markdown\n"
        "## Goal\n- a thing\n"
        "## Constraints & Preferences\n- (none)\n"
        "## Progress\n### Done\n- ok\n### In Progress\n- (none)\n### Blocked\n- (none)\n"
        "## Key Decisions\n- (none)\n"
        "## Next Steps\n- (none)\n"
        "## Critical Context\n- (none)\n"
        "## Relevant Files\n- (none)\n"
        "```"
    )
    req = _basic_request()
    digest = generate_digest(req, llm_call=lambda _: fenced)

    assert digest.source == "llm"
    assert not digest.markdown.startswith("```")


def test_generate_digest_includes_prior_summary_in_prompt() -> None:
    captured = []

    def fake_llm(prompt: str) -> str:
        captured.append(prompt)
        return "## Goal\n- new\n## Constraints & Preferences\n- (none)\n## Progress\n### Done\n- (none)\n### In Progress\n- (none)\n### Blocked\n- (none)\n## Key Decisions\n- (none)\n## Next Steps\n- (none)\n## Critical Context\n- (none)\n## Relevant Files\n- (none)\n"

    req = _basic_request(prior_digest_markdown="## Goal\n- prior thing\n")
    generate_digest(req, llm_call=fake_llm)

    assert len(captured) == 1
    # The prior summary appears wrapped in the anchor tag.
    assert "prior thing" in captured[0]
    assert "<previous-summary>" in captured[0]


def test_generate_digest_records_elapsed_time() -> None:
    req = _basic_request()
    digest = generate_digest(req, llm_call=lambda _: "## Goal\n- done\n")
    assert digest.elapsed_ms >= 0.0


def test_generate_digest_preserves_project_metadata() -> None:
    req = _basic_request()
    digest = generate_digest(req, llm_call=None)
    assert digest.project_name == "testproj"
    assert digest.project_path == Path("/tmp/testproj")
