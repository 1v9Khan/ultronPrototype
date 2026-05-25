"""Tests for the T4 declared-vs-observed coherence checker."""

from __future__ import annotations

import textwrap

import pytest

from ultron.install.coherence import (
    CoherenceMismatch,
    CoherenceMismatchKind,
    CoherenceSeverity,
    check_coherence,
    check_intent_phrase_coherence,
    declared_bins,
    declared_config_paths,
    declared_env_vars,
    declared_os,
    detect_os_signals,
    extract_bin_refs,
    extract_config_refs,
    extract_env_refs,
)


# ---------------------------------------------------------------------------
# extract_env_refs


def test_extract_env_literal_getenv() -> None:
    src = "x = os.getenv('TODOIST_API_KEY')"
    literals, dyn = extract_env_refs(src)
    assert literals == {"TODOIST_API_KEY"}
    assert dyn == 0


def test_extract_env_environ_indexing() -> None:
    src = 'token = os.environ["GITHUB_TOKEN"]'
    literals, _ = extract_env_refs(src)
    assert literals == {"GITHUB_TOKEN"}


def test_extract_env_environ_get() -> None:
    src = 'x = os.environ.get("HOME")'
    literals, _ = extract_env_refs(src)
    assert "HOME" in literals


def test_extract_env_dynamic_returns_count() -> None:
    src = "x = os.getenv(name)\ny = os.getenv(name_var)"
    literals, dyn = extract_env_refs(src)
    assert literals == set()
    assert dyn == 2


def test_extract_env_multiple_literals() -> None:
    src = textwrap.dedent("""
        a = os.getenv("ALPHA")
        b = os.getenv('BETA')
        c = os.environ["GAMMA"]
    """)
    literals, _ = extract_env_refs(src)
    assert literals == {"ALPHA", "BETA", "GAMMA"}


# ---------------------------------------------------------------------------
# extract_bin_refs


def test_extract_bin_shutil_which() -> None:
    src = 'p = shutil.which("rg")'
    assert extract_bin_refs(src) == {"rg"}


def test_extract_bin_subprocess() -> None:
    src = 'subprocess.run(["git", "status"])'
    assert extract_bin_refs(src) == {"git"}


def test_extract_bin_paths_are_skipped() -> None:
    src = 'subprocess.run(["/usr/bin/ls"])'
    assert extract_bin_refs(src) == set()


def test_extract_bin_relative_paths_skipped() -> None:
    src = 'subprocess.run(["./script.sh"])'
    assert extract_bin_refs(src) == set()


# ---------------------------------------------------------------------------
# extract_config_refs


def test_extract_config_basic_attribute_access() -> None:
    src = "x = config.web_search.providers"
    assert "web_search.providers" in extract_config_refs(src)


def test_extract_config_deeply_nested() -> None:
    src = "y = config.memory.qdrant.host"
    refs = extract_config_refs(src)
    # Multiple sub-paths emitted as the visitor walks the chain.
    assert "memory.qdrant.host" in refs


def test_extract_config_syntax_error_returns_empty() -> None:
    assert extract_config_refs("def broken( :") == set()


def test_extract_config_custom_root() -> None:
    src = "x = settings.foo.bar"
    refs = extract_config_refs(src, config_attr_root="settings")
    assert "foo.bar" in refs


# ---------------------------------------------------------------------------
# Manifest accessors


def test_declared_env_collects_all_sources() -> None:
    manifest = {
        "requires": {"env": ["FOO", "BAR"]},
        "envVars": [{"name": "BAZ"}, "QUX"],
        "primaryEnv": "PRIMARY",
    }
    assert declared_env_vars(manifest) == {"FOO", "BAR", "BAZ", "QUX", "PRIMARY"}


def test_declared_bins_union() -> None:
    manifest = {
        "requires": {"bins": ["git"], "anyBins": ["rg", "ripgrep"]},
    }
    assert declared_bins(manifest) == {"git", "rg", "ripgrep"}


def test_declared_config_paths() -> None:
    manifest = {"requires": {"config": ["a.b", "c"]}}
    assert declared_config_paths(manifest) == {"a.b", "c"}


def test_declared_os_normalises_case() -> None:
    manifest = {"os": ["MacOS", "WINDOWS"]}
    assert declared_os(manifest) == {"macos", "windows"}


def test_declared_os_single_string() -> None:
    assert declared_os({"os": "Linux"}) == {"linux"}


def test_declared_os_missing_returns_empty() -> None:
    assert declared_os({}) == set()


# ---------------------------------------------------------------------------
# OS signal detection


def test_detect_os_signals_win32_import() -> None:
    assert "windows" in detect_os_signals("import win32api")


def test_detect_os_signals_macos_osascript() -> None:
    src = 'subprocess.run(["osascript", "-e", "tell..."])'
    assert "macos" in detect_os_signals(src)


def test_detect_os_signals_clean_source() -> None:
    assert detect_os_signals("print('hello')") == set()


# ---------------------------------------------------------------------------
# check_coherence (the bidirectional linter)


def test_coherence_clean_passes() -> None:
    manifest = {
        "requires": {"env": ["TODOIST_API_KEY"]},
    }
    source = 'token = os.getenv("TODOIST_API_KEY")'
    findings = check_coherence(manifest, [("skill.py", source)])
    assert findings == ()


def test_coherence_missing_env_declaration() -> None:
    manifest = {}
    source = 'token = os.getenv("TODOIST_API_KEY")'
    findings = check_coherence(manifest, [("skill.py", source)])
    assert any(
        f.kind is CoherenceMismatchKind.MISSING_DECLARATION
        and f.category == "env"
        and f.identifier == "TODOIST_API_KEY"
        for f in findings
    )


def test_coherence_unused_env_declaration() -> None:
    manifest = {"requires": {"env": ["UNUSED_TOKEN"]}}
    source = "print('no env reads here')"
    findings = check_coherence(manifest, [("skill.py", source)])
    assert any(
        f.kind is CoherenceMismatchKind.UNUSED_DECLARATION
        and f.identifier == "UNUSED_TOKEN"
        for f in findings
    )


def test_coherence_dynamic_read_emits_info() -> None:
    manifest = {}
    source = "for name in names:\n    v = os.getenv(name)\n"
    findings = check_coherence(manifest, [("skill.py", source)])
    dyn = [f for f in findings if f.kind is CoherenceMismatchKind.DYNAMIC_READ]
    assert len(dyn) >= 1
    assert dyn[0].severity is CoherenceSeverity.INFO


def test_coherence_skips_always_available_env() -> None:
    """PATH and HOME should never trigger missing-declaration."""
    manifest = {}
    source = 'p = os.getenv("PATH")\nh = os.getenv("HOME")'
    findings = check_coherence(manifest, [("skill.py", source)])
    assert not any(
        f.category == "env" and f.identifier in {"PATH", "HOME"}
        for f in findings
    )


def test_coherence_missing_bin_declaration() -> None:
    manifest = {}
    source = 'p = shutil.which("rg")'
    findings = check_coherence(manifest, [("skill.py", source)])
    assert any(
        f.kind is CoherenceMismatchKind.MISSING_DECLARATION
        and f.category == "bin"
        and f.identifier == "rg"
        for f in findings
    )


def test_coherence_common_bins_never_flagged() -> None:
    manifest = {}
    source = 'subprocess.run(["python", "script.py"])'
    findings = check_coherence(manifest, [("skill.py", source)])
    assert not any(
        f.category == "bin" and f.identifier == "python"
        for f in findings
    )


def test_coherence_os_mismatch() -> None:
    manifest = {"os": ["macos"]}
    source = "import win32api"
    findings = check_coherence(manifest, [("skill.py", source)])
    assert any(
        f.kind is CoherenceMismatchKind.OS_MISMATCH
        and f.identifier == "windows"
        for f in findings
    )


def test_coherence_os_compatible_no_finding() -> None:
    manifest = {"os": ["macos", "windows"]}
    source = "import win32api"
    findings = check_coherence(manifest, [("skill.py", source)])
    assert not any(f.kind is CoherenceMismatchKind.OS_MISMATCH for f in findings)


def test_coherence_no_decl_os_skips_check() -> None:
    """Without a manifest os declaration, OS signals aren't compared."""
    manifest = {}
    source = "import win32api"
    findings = check_coherence(manifest, [("skill.py", source)])
    assert not any(f.kind is CoherenceMismatchKind.OS_MISMATCH for f in findings)


def test_coherence_config_subpath_match() -> None:
    """When manifest declares `web_search`, observing `web_search.providers` is OK."""
    manifest = {"requires": {"config": ["web_search"]}}
    source = "x = config.web_search.providers"
    findings = check_coherence(manifest, [("skill.py", source)])
    assert not any(
        f.category == "config"
        and f.kind is CoherenceMismatchKind.MISSING_DECLARATION
        for f in findings
    )


def test_coherence_multi_file() -> None:
    manifest = {"requires": {"env": ["TODOIST_API_KEY"]}}
    sources = [
        ("a.py", 'a = os.getenv("TODOIST_API_KEY")'),
        ("b.py", 'b = os.getenv("OTHER")'),
    ]
    findings = check_coherence(manifest, sources)
    # OTHER should be flagged missing; TODOIST should not.
    flagged = {(f.category, f.identifier) for f in findings if f.kind is CoherenceMismatchKind.MISSING_DECLARATION}
    assert ("env", "OTHER") in flagged
    assert ("env", "TODOIST_API_KEY") not in flagged


# ---------------------------------------------------------------------------
# check_intent_phrase_coherence


def test_intent_coherence_matching_phrase() -> None:
    body = "This skill helps you schedule a meeting on the calendar."
    findings = check_intent_phrase_coherence(
        ["schedule a meeting"], body
    )
    assert findings == ()


def test_intent_coherence_mismatched_phrase() -> None:
    body = "This skill manages calendar events."
    findings = check_intent_phrase_coherence(
        ["trade my crypto wallet"], body
    )
    assert len(findings) == 1
    assert findings[0].kind is CoherenceMismatchKind.MISSING_DECLARATION
    assert findings[0].category == "intent"


def test_intent_coherence_empty_phrase_skipped() -> None:
    findings = check_intent_phrase_coherence(["", "  "], "anything")
    assert findings == ()


def test_intent_coherence_short_tokens_skipped() -> None:
    """Tokens <=2 chars are skipped (too generic to count as overlap)."""
    body = "a b c"
    findings = check_intent_phrase_coherence(["a b c"], body)
    # All tokens <=2 chars filtered -> no phrase tokens -> skip silently.
    assert findings == ()
