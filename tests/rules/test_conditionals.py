"""Tests for ultron.rules.conditionals."""

from __future__ import annotations

from pathlib import Path

import pytest

from ultron.rules import conditionals as cc


def _write_rule(path: Path, frontmatter: str, body: str = "rule body text") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{frontmatter}---\n{body}\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Path-extraction heuristic
# ---------------------------------------------------------------------------

class TestExtractPaths:
    def test_empty_returns_empty(self) -> None:
        assert cc.extract_path_like_strings("") == set()
        assert cc.extract_path_like_strings(None) == set()  # type: ignore[arg-type]

    def test_slash_token_picked_up(self) -> None:
        tokens = cc.extract_path_like_strings("look at src/foo/bar.py please")
        assert "src/foo/bar.py" in tokens

    def test_extension_token_picked_up(self) -> None:
        tokens = cc.extract_path_like_strings("config.yaml is broken")
        assert "config.yaml" in tokens

    def test_url_stripped_before_extraction(self) -> None:
        tokens = cc.extract_path_like_strings(
            "see https://example.com/foo/bar then local/src/a.py",
        )
        assert "local/src/a.py" in tokens
        assert all("example.com" not in t for t in tokens)

    def test_code_fence_stripped(self) -> None:
        # Tokens inside fenced code blocks should be ignored.
        tokens = cc.extract_path_like_strings(
            "look at outside.py```\nfoo/bar.py\n```please",
        )
        assert "outside.py" in tokens
        assert "foo/bar.py" not in tokens

    def test_very_long_token_capped(self) -> None:
        long_name = "x" * 600 + ".py"
        tokens = cc.extract_path_like_strings(f"see {long_name}")
        # Long token should be rejected by the length cap.
        assert not any(len(t) > cc.MAX_EXTRACTED_TOKEN_LENGTH for t in tokens)

    def test_short_token_rejected(self) -> None:
        # Single-char extensions or names below MIN length should not fire.
        tokens = cc.extract_path_like_strings("a b c")
        assert tokens == set()


# ---------------------------------------------------------------------------
# Single-condition evaluation
# ---------------------------------------------------------------------------

class TestEvaluate:
    def _make(self, conditions: dict) -> cc.ConditionalRule:
        return cc.ConditionalRule(
            name="r",
            source_path=Path("/tmp/r.md"),
            body="body",
            conditions=conditions,
        )

    def test_no_conditions_always_active(self) -> None:
        rule = self._make({})
        out = cc.evaluate_rule(rule, cc.RuleEvaluationContext())
        assert out is not None
        assert "always" in out.matched_conditions

    def test_paths_match(self) -> None:
        rule = self._make({"paths": ["src/voice/**"]})
        ctx = cc.RuleEvaluationContext(user_text="please open src/voice/main.py")
        out = cc.evaluate_rule(rule, ctx)
        assert out is not None
        assert "paths" in out.matched_conditions

    def test_paths_empty_list_deactivates(self) -> None:
        rule = self._make({"paths": []})
        ctx = cc.RuleEvaluationContext(user_text="src/voice/main.py")
        assert cc.evaluate_rule(rule, ctx) is None

    def test_paths_extra_path_match(self) -> None:
        rule = self._make({"paths": ["src/voice/**"]})
        ctx = cc.RuleEvaluationContext(
            user_text="hi",
            extra_paths=("src/voice/main.py",),
        )
        out = cc.evaluate_rule(rule, ctx)
        assert out is not None

    def test_intents_match(self) -> None:
        rule = self._make({"intents": ["CODE_TASK", "PROGRESS_QUERY"]})
        ctx = cc.RuleEvaluationContext(intent_kind="CODE_TASK")
        out = cc.evaluate_rule(rule, ctx)
        assert out is not None
        assert "intents" in out.matched_conditions

    def test_intents_no_match(self) -> None:
        rule = self._make({"intents": ["CANCEL"]})
        ctx = cc.RuleEvaluationContext(intent_kind="CODE_TASK")
        assert cc.evaluate_rule(rule, ctx) is None

    def test_topics_substring_match(self) -> None:
        rule = self._make({"topics": ["budget"]})
        ctx = cc.RuleEvaluationContext(user_text="tell me about the budget")
        out = cc.evaluate_rule(rule, ctx)
        assert out is not None
        assert "topics" in out.matched_conditions

    def test_topics_regex_match(self) -> None:
        rule = self._make({"topics": ["/^urgent/"]})
        ctx = cc.RuleEvaluationContext(user_text="URGENT - server down")
        out = cc.evaluate_rule(rule, ctx)
        assert out is not None

    def test_topics_from_topics_field(self) -> None:
        rule = self._make({"topics": ["finance"]})
        ctx = cc.RuleEvaluationContext(user_text="x", topics=("finance",))
        out = cc.evaluate_rule(rule, ctx)
        assert out is not None

    def test_system_state_bool(self) -> None:
        rule = self._make({"system_state": {"gaming_mode": True}})
        on = cc.RuleEvaluationContext(system_state={"gaming_mode": True})
        off = cc.RuleEvaluationContext(system_state={"gaming_mode": False})
        assert cc.evaluate_rule(rule, on) is not None
        assert cc.evaluate_rule(rule, off) is None

    def test_system_state_comparator(self) -> None:
        rule = self._make({"system_state": {"n_active_skills": ">=2"}})
        hit = cc.RuleEvaluationContext(system_state={"n_active_skills": 3})
        miss = cc.RuleEvaluationContext(system_state={"n_active_skills": 1})
        assert cc.evaluate_rule(rule, hit) is not None
        assert cc.evaluate_rule(rule, miss) is None

    def test_system_state_string_match(self) -> None:
        rule = self._make({"system_state": {"environment": "production"}})
        hit = cc.RuleEvaluationContext(system_state={"environment": "production"})
        assert cc.evaluate_rule(rule, hit) is not None

    def test_not_in_gaming_mode(self) -> None:
        rule = self._make({"not_in_gaming_mode": True})
        ctx_off = cc.RuleEvaluationContext(system_state={"gaming_mode": False})
        ctx_on = cc.RuleEvaluationContext(system_state={"gaming_mode": True})
        assert cc.evaluate_rule(rule, ctx_off) is not None
        # When gaming is on, the rule should not activate.
        assert cc.evaluate_rule(rule, ctx_on) is None

    def test_multiple_conditions_one_fires(self) -> None:
        rule = self._make({
            "paths": ["src/voice/**"],
            "intents": ["CODE_TASK"],
        })
        # Only intents fires.
        ctx = cc.RuleEvaluationContext(intent_kind="CODE_TASK")
        out = cc.evaluate_rule(rule, ctx)
        assert out is not None
        assert "intents" in out.matched_conditions

    def test_all_of_requires_every_block(self) -> None:
        rule = self._make({
            "all_of": [
                {"intents": ["CODE_TASK"]},
                {"system_state": {"gaming_mode": False}},
            ],
        })
        good = cc.RuleEvaluationContext(
            intent_kind="CODE_TASK",
            system_state={"gaming_mode": False},
        )
        bad = cc.RuleEvaluationContext(
            intent_kind="CODE_TASK",
            system_state={"gaming_mode": True},
        )
        assert cc.evaluate_rule(rule, good) is not None
        assert cc.evaluate_rule(rule, bad) is None


# ---------------------------------------------------------------------------
# Loading from disk
# ---------------------------------------------------------------------------

class TestLoadFromDisk:
    def test_load_simple_rule(self, tmp_path: Path) -> None:
        path = _write_rule(
            tmp_path / "voice.md",
            "name: voice_rule\nintents: [GREETING]\n",
            body="Be brief.",
        )
        rule = cc.load_rule_from_path(path)
        assert rule is not None
        assert rule.name == "voice_rule"
        assert rule.body == "Be brief."
        assert rule.conditions["intents"] == ["GREETING"]

    def test_load_uses_stem_when_name_missing(self, tmp_path: Path) -> None:
        path = _write_rule(tmp_path / "casual.md", "intents: [CHAT]\n")
        rule = cc.load_rule_from_path(path)
        assert rule is not None
        assert rule.name == "casual"

    def test_load_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert cc.load_rule_from_path(tmp_path / "no.md") is None

    def test_load_rule_set_merges_layers(self, tmp_path: Path) -> None:
        global_dir = tmp_path / "global"
        proj_dir = tmp_path / "project"
        _write_rule(global_dir / "a.md", "name: a\nintents: [X]\n")
        _write_rule(proj_dir / "a.md", "name: a\nintents: [Y]\n")
        _write_rule(proj_dir / "b.md", "name: b\nintents: [Z]\n")
        rs = cc.load_rule_set([
            (global_dir, "global"),
            (proj_dir, "project"),
        ])
        assert "a" in rs.rules and "b" in rs.rules
        # Project layer should win on collision.
        assert rs.rules["a"].source_layer == "project"
        assert rs.rules["a"].conditions["intents"] == ["Y"]

    def test_load_rule_set_skips_unknown_ext(self, tmp_path: Path) -> None:
        proj_dir = tmp_path / "p"
        _write_rule(proj_dir / "ok.md", "name: ok\n")
        # A .txt file should not be loaded.
        (proj_dir / "skip.txt").write_text("not a rule", encoding="utf-8")
        rs = cc.load_rule_set([(proj_dir, "project")])
        assert "ok" in rs.rules
        assert "skip" not in rs.rules


# ---------------------------------------------------------------------------
# ConditionalRuleSet.evaluate
# ---------------------------------------------------------------------------

class TestRuleSetEvaluate:
    def test_returns_activated_rules_only(self, tmp_path: Path) -> None:
        proj = tmp_path / "p"
        _write_rule(proj / "casual.md", "name: casual\nintents: [GREETING]\n")
        _write_rule(proj / "coding.md", "name: coding\nintents: [CODE_TASK]\n")
        rs = cc.load_rule_set([(proj, "project")])
        ctx = cc.RuleEvaluationContext(intent_kind="GREETING")
        out = rs.evaluate(ctx)
        names = [a.name for a in out]
        assert "casual" in names
        assert "coding" not in names

    def test_always_on_rules_included(self, tmp_path: Path) -> None:
        proj = tmp_path / "p"
        _write_rule(proj / "core.md", "name: core\n")  # no conditions
        rs = cc.load_rule_set([(proj, "project")])
        ctx = cc.RuleEvaluationContext()
        out = rs.evaluate(ctx)
        assert any(a.name == "core" for a in out)
