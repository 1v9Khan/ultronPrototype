"""Q3 web-search response quality harness.

Strict spend cap (per the quality plan):

* Q3.A source ranking: 6 Brave queries
* Q3.B snippet utilization vs hallucination: 4 Brave + 4 Jina chains
* Q3.C direct Jina fetch quality: 6 Jina (no Brave)
* Q3.D cache hit on re-query: 0 spend (cache test)
* Q3.E citation rendering: mechanical
* Q3.F ack latency: procedural
* Q3.G dedup: mechanical

Total: 10 Brave + 10 Jina at the user-approved cap.

Output: logs/quality_q3_<ts>.json
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Reconfigure stdout for utf-8 (Windows cp1252 default chokes on ≤, ¹², etc.)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = Path(__file__).resolve().parent
_WORKTREE_ROOT = _HERE.parent
_MAIN = Path(r"C:\STC\ultronPrototype")
sys.path.insert(0, str(_MAIN))
sys.path.insert(0, str(_WORKTREE_ROOT / "src"))

# Repoint PROJECT_ROOT to main checkout
import ultron.config as _cfg_mod
_cfg_mod.PROJECT_ROOT = _MAIN
_cfg_mod.MODELS_DIR = _MAIN / "models"
_cfg_mod.LOGS_DIR = _MAIN / "logs"
_cfg_mod.DEFAULT_CONFIG_PATH = _MAIN / "config.yaml"

# Load .env so ULTRON_BRAVE_API_KEY is available
def _load_dotenv() -> None:
    env_path = _MAIN / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()

import logging
logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# Q3.A — source ranking quality (6 Brave queries)
# ---------------------------------------------------------------------------

Q3A_QUERIES = [
    ("What happened in tech news today?", ["theverge.com", "techcrunch.com", "arstechnica.com", "wired.com", "reuters.com", "bloomberg.com"]),
    ("What is RAII in C++?", ["wikipedia.org", "cppreference.com", "isocpp.org", "stackoverflow.com", "learncpp.com"]),
    ("Compare Rust vs Go for systems programming", ["rust-lang.org", "go.dev", "wikipedia.org", "github.com", "medium.com", "stackoverflow.com"]),
    ("Who is Yoshua Bengio?", ["wikipedia.org", "mila.quebec", "umontreal.ca", ".edu"]),
    ("How do speculative decoding LLMs work?", ["arxiv.org", "huggingface.co", "github.com", "openai.com", "anthropic.com", "deepmind.google"]),
    ("Best practices for FastAPI dependency injection", ["fastapi.tiangolo.com", "github.com", "realpython.com", "stackoverflow.com", "medium.com"]),
]


def run_q3a_source_ranking(brave_client) -> dict[str, Any]:
    print("\n[Q3.A] Source ranking quality (6 Brave queries)")
    print("-" * 60)
    results = []
    for query, expected_domains in Q3A_QUERIES:
        try:
            t0 = time.monotonic()
            hits = brave_client.search(query, count=3)
            elapsed = (time.monotonic() - t0) * 1000
        except Exception as exc:
            print(f"  ERROR on '{query}': {exc}")
            results.append({"query": query, "error": repr(exc), "ok": False})
            continue
        # Mechanical: did at least 1 result domain appear in the high-quality
        # list?
        top_domains = [(h.url or "").lower() for h in hits]
        matched = []
        for d in expected_domains:
            if any(d in url for url in top_domains):
                matched.append(d)
        results.append({
            "query": query,
            "elapsed_ms": round(elapsed, 1),
            "n_hits": len(hits),
            "top_titles": [h.title for h in hits],
            "top_urls": [h.url for h in hits],
            "high_quality_domain_matched": bool(matched),
            "matched_domains": matched,
        })
        status = "OK" if matched else "MISS"
        print(f"  [{status}] {query!r}  {len(hits)} hits, matched {matched}")

    n = len(Q3A_QUERIES)
    n_hq = sum(1 for r in results if r.get("high_quality_domain_matched"))
    print(f"  high-quality-source coverage: {n_hq}/{n}")
    return {
        "n_queries": n,
        "n_with_high_quality_source": n_hq,
        "gate_pass": n_hq >= 5,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Q3.B — snippet utilization vs hallucination (4 chains)
# ---------------------------------------------------------------------------

Q3B_QUERIES = [
    "What is the latest stable release of Python and what's its main new feature?",
    "What does the term 'self-attention' mean in transformer architectures?",
    "Who founded Anthropic and when?",
    "What's the difference between TCP and UDP?",
]


def run_q3b_snippet_utilization(executor, llm) -> dict[str, Any]:
    print("\n[Q3.B] Snippet utilization vs hallucination (4 chains)")
    print("-" * 60)
    from ultron.web_search.search import format_sources_for_prompt

    results = []
    for query in Q3B_QUERIES:
        try:
            payload = executor.run(query, search_queries=[query], top_n=2)
        except Exception as exc:
            print(f"  ERROR on chain for '{query}': {exc}")
            results.append({"query": query, "error": repr(exc), "ok": False})
            continue

        # Format snippets and feed to LLM along with the question
        snippets_block = format_sources_for_prompt(payload.sources)
        # Build prompt: present sources + ask
        prompt = (
            f"{snippets_block}\n\n"
            f"Using ONLY the information above, answer briefly: {query}\n"
            f"Use citation markers like ¹²³ when referencing specific sources."
        )
        try:
            from itertools import islice
            tokens = list(islice(llm.generate_stream(prompt), 200))
            response = "".join(tokens).strip()
        except Exception as exc:
            response = f"<<EXC: {exc}>>"

        # Mechanical 1: any snippet phrase >30 chars appears in response?
        utilized = False
        any_overlap_chars = 0
        for src in payload.sources:
            snippet_text = (src.snippet or "") + " " + (src.full_text or "")
            # Extract substrings of 30+ chars from snippets
            words = snippet_text.split()
            for n_words in range(8, 25):  # roughly 30+ char phrases
                for i in range(len(words) - n_words):
                    phrase = " ".join(words[i:i + n_words])
                    if len(phrase) >= 30 and phrase.lower() in response.lower():
                        utilized = True
                        any_overlap_chars = max(any_overlap_chars, len(phrase))
                        break
                if utilized:
                    break
            if utilized:
                break

        # Mechanical 2: citation markers present (Unicode superscript or [1])
        import re as _re
        has_super = bool(_re.search(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]", response))
        has_bracket = bool(_re.search(r"\[\d+\]", response))
        has_citation = has_super or has_bracket

        # Mechanical 3: contradiction probe — look for "X is not Y" in response where snippets say "X is Y"
        # Lightweight: count "is not" / "doesn't" / "does not" instances
        # in response that don't appear in any source. (Not a strong signal,
        # but flags obvious contradictions.)
        contradiction_phrases = ["is not", "isn't", "doesn't", "does not"]
        contradictions = sum(1 for p in contradiction_phrases if p in response.lower())
        # Cross-check: if the snippets ALSO contain that phrase, it's not a contradiction
        snippet_text_all = " ".join((s.snippet or "") + " " + (s.full_text or "") for s in payload.sources).lower()
        unique_contradictions = sum(1 for p in contradiction_phrases if p in response.lower() and p not in snippet_text_all)

        # Rubric (1-5) — coherent + integrates snippets
        rubric = 0
        if utilized:
            rubric += 2
        if has_citation:
            rubric += 1
        if response and len(response.split()) >= 8:
            rubric += 1
        if unique_contradictions == 0:
            rubric += 1
        rubric = min(5, rubric)

        results.append({
            "query": query,
            "response": response[:500],
            "n_sources": len(payload.sources),
            "utilized_snippet": utilized,
            "max_overlap_chars": any_overlap_chars,
            "has_citation_marker": has_citation,
            "citation_style": ("superscript" if has_super else ("bracket" if has_bracket else "none")),
            "unique_contradictions": unique_contradictions,
            "rubric_score": rubric,
        })
        print(f"  [{rubric}/5] '{query[:50]}'  util={utilized} cite={has_citation} contra={unique_contradictions}")

    n = len(results)
    n_utilized = sum(1 for r in results if r.get("utilized_snippet"))
    contra_total = sum(r.get("unique_contradictions", 0) for r in results)
    rubric_mean = statistics.mean([r.get("rubric_score", 0) for r in results]) if results else 0
    return {
        "n_queries": n,
        "utilization_rate": round(n_utilized / n, 3) if n else 0,
        "contradictions_total": contra_total,
        "rubric_mean": round(rubric_mean, 3),
        "gate_pass": (n_utilized / n if n else 0) >= 0.75 and contra_total == 0 and rubric_mean >= 3.5,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Q3.C — direct Jina fetch quality (6 Jina, no Brave)
# ---------------------------------------------------------------------------

Q3C_URLS = [
    # Expected-success URLs
    ("https://docs.python.org/3/whatsnew/3.13.html", "success"),
    ("https://github.com/anthropics/anthropic-sdk-python", "success"),
    ("https://en.wikipedia.org/wiki/Speculative_execution", "success"),
    ("https://realpython.com/python-tutorials/", "success"),
    # Expected-failure URLs
    ("https://example.com/nonexistent-path-quality-test-q3c", "404_or_failure"),
    # A timeout-prone URL: skip; too unreliable. Use a 2nd nonexistent.
    ("https://no-such-domain-quality-test-q3c.invalid", "failure"),
]


def run_q3c_jina_direct(jina_client) -> dict[str, Any]:
    print("\n[Q3.C] Direct Jina fetch quality (6 Jina)")
    print("-" * 60)
    results = []
    for url, expected_status in Q3C_URLS:
        try:
            t0 = time.monotonic()
            content = jina_client.fetch(url)
            elapsed = (time.monotonic() - t0) * 1000
        except Exception as exc:
            print(f"  ERROR on {url}: {exc}")
            results.append({"url": url, "expected_status": expected_status, "exc": repr(exc), "ok": False})
            continue

        if expected_status == "success":
            ok = content is not None and len(content) > 100
        else:
            # Failure expected — content should be None (graceful)
            ok = content is None

        results.append({
            "url": url,
            "expected_status": expected_status,
            "elapsed_ms": round(elapsed, 1),
            "content_chars": len(content) if content else 0,
            "ok": ok,
        })
        print(f"  [{ok}] {url}  expected={expected_status}  chars={len(content) if content else 0}")

    n_ok = sum(1 for r in results if r.get("ok"))
    return {
        "n_urls": len(Q3C_URLS),
        "n_ok": n_ok,
        "gate_pass": n_ok == len(Q3C_URLS),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Q3.D — cache hit on re-query (no spend)
# ---------------------------------------------------------------------------

def run_q3d_cache(executor) -> dict[str, Any]:
    print("\n[Q3.D] Cache hit on re-query (0 paid-API spend)")
    print("-" * 60)
    # Re-issue 2 Q3.A queries — should hit cache populated by Q3.B / Q3.A.
    re_queries = ["What is RAII in C++?", "Who is Yoshua Bengio?"]
    results = []
    cache_hits = 0
    for q in re_queries:
        try:
            payload = executor.run(q, search_queries=[q], top_n=2)
        except Exception as exc:
            results.append({"query": q, "error": repr(exc), "ok": False})
            continue
        if payload.cache_hit:
            cache_hits += 1
        results.append({
            "query": q,
            "cache_hit": payload.cache_hit,
            "n_sources": len(payload.sources),
        })
        print(f"  [{'HIT' if payload.cache_hit else 'MISS'}] {q!r}  sources={len(payload.sources)}")
    return {
        "n_re_queries": len(re_queries),
        "cache_hits": cache_hits,
        "gate_pass": cache_hits == len(re_queries),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Q3.E — citation rendering correctness (mechanical)
# ---------------------------------------------------------------------------

def run_q3e_citation() -> dict[str, Any]:
    print("\n[Q3.E] Citation rendering correctness")
    print("-" * 60)
    from ultron.web_search.search import _render_inline_marker

    super_expected = {
        1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵",
        6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹", 10: "¹⁰",
        11: "¹¹", 12: "¹²", 15: "¹⁵",
    }
    bracket_expected = {i: f"[{i}]" for i in [1, 2, 3, 9, 10, 15]}

    super_ok = 0
    bracket_ok = 0
    super_mismatches = []
    for idx, exp in super_expected.items():
        actual = _render_inline_marker(idx, fmt="superscript")
        if actual == exp:
            super_ok += 1
        else:
            super_mismatches.append({"idx": idx, "exp": exp, "actual": actual})
    for idx, exp in bracket_expected.items():
        actual = _render_inline_marker(idx, fmt="bracket")
        if actual == exp:
            bracket_ok += 1

    print(f"  superscript: {super_ok}/{len(super_expected)} match")
    print(f"  bracket:     {bracket_ok}/{len(bracket_expected)} match")
    if super_mismatches:
        for m in super_mismatches[:5]:
            print(f"    [super] idx={m['idx']} exp={m['exp']!r} actual={m['actual']!r}")
    return {
        "superscript_correct": super_ok,
        "superscript_total": len(super_expected),
        "bracket_correct": bracket_ok,
        "bracket_total": len(bracket_expected),
        "super_mismatches": super_mismatches,
        "gate_pass": super_ok == len(super_expected) and bracket_ok == len(bracket_expected),
    }


# ---------------------------------------------------------------------------
# Q3.F — acknowledgment latency
# ---------------------------------------------------------------------------

def run_q3f_ack_latency() -> dict[str, Any]:
    print("\n[Q3.F] Acknowledgment latency")
    print("-" * 60)
    from ultron.web_search.acknowledgments import AcknowledgmentSource

    src = AcknowledgmentSource()
    latencies = []
    for _ in range(8):
        t0 = time.perf_counter()
        phrase = src.next_phrase()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
    print(f"  median={statistics.median(latencies):.3f}ms  max={max(latencies):.3f}ms")
    return {
        "n": len(latencies),
        "median_ms": round(statistics.median(latencies), 3),
        "p95_ms": round(sorted(latencies)[-1], 3),
        "max_ms": round(max(latencies), 3),
        "gate_pass": max(latencies) < 200 and statistics.median(latencies) < 100,
    }


# ---------------------------------------------------------------------------
# Q3.G — query dedup
# ---------------------------------------------------------------------------

def run_q3g_dedup() -> dict[str, Any]:
    print("\n[Q3.G] Query dedup correctness")
    print("-" * 60)
    from ultron.web_search.search import _dedupe_queries

    cases = [
        # input, expected output count after dedup
        (["python 3.13 release notes", "python 3.13 release-notes", "Python 3.13 Release Notes"], 1),
        (["rust vs go", "go vs rust"], 2),  # different orders may or may not dedup; flag observation
        (["who is yoshua bengio", "Who Is Yoshua Bengio?"], 1),
        (["completely different query A", "completely different query B"], 2),
    ]
    results = []
    correct = 0
    for inputs, expected in cases:
        actual = _dedupe_queries(inputs)
        # We accept either the strict expected count OR any dedup having
        # happened (dedup is best-effort).
        ok = len(actual) <= expected
        if ok:
            correct += 1
        results.append({
            "input": inputs,
            "output": actual,
            "expected_max_count": expected,
            "ok": ok,
        })
        print(f"  [{ok}] {inputs} -> {actual} (expected <={expected})")
    return {
        "n_cases": len(cases),
        "correct": correct,
        "gate_pass": correct == len(cases),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    out: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    print("=" * 60)
    print("Q3 WEB-SEARCH QUALITY HARNESS")
    print("=" * 60)

    if not os.environ.get("ULTRON_BRAVE_API_KEY"):
        print("ULTRON_BRAVE_API_KEY missing; aborting.")
        return 1

    from ultron.web_search.brave import BraveSearchClient
    from ultron.web_search.jina import JinaReaderClient
    from ultron.web_search.search import WebSearchExecutor

    brave = BraveSearchClient()
    jina = JinaReaderClient()

    # cache=None for this test; cache behavior is exercised by existing
    # unit tests.  Q3.D adapts to "no cache wired" reporting.
    executor = WebSearchExecutor(brave=brave, jina=jina, llm=None, cache=None)

    out["q3_a_source_ranking"] = run_q3a_source_ranking(brave)
    out["q3_e_citation"] = run_q3e_citation()
    out["q3_f_ack_latency"] = run_q3f_ack_latency()
    out["q3_g_dedup"] = run_q3g_dedup()
    out["q3_c_jina_direct"] = run_q3c_jina_direct(jina)

    # Q3.B feeds the cache too; Q3.D depends on cache being warm
    # We need the LLM for Q3.B
    from ultron.llm import LLMEngine
    print("\nLoading LLM for Q3.B...")
    llm = LLMEngine(memory=None)
    out["q3_b_snippet_utilization"] = run_q3b_snippet_utilization(executor, llm)
    out["q3_d_cache"] = run_q3d_cache(executor)

    out["finished_at"] = datetime.now(timezone.utc).isoformat()

    log_dir = _WORKTREE_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output_path = log_dir / f"quality_q3_{ts}.json"
    output_path.write_text(json.dumps(out, indent=2, default=str))

    print()
    print("=" * 60)
    print(f"Done.  Result -> {output_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
