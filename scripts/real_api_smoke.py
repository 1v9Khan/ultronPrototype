"""Real-API sparing smoke test.

Strict budget:
* 1 Brave search query (proof-of-life — can the client reach Brave?)
* 0 Jina fetches by default (Jina is exercised through web_search.search.run
  which runs Brave + Jina in chain; running 1 chain = 1 of each)
* 1 chain of Brave -> Jina via WebSearchExecutor
* 1 Claude Code subprocess invocation against a tiny prompt

Run from the main checkout (or worktree — both work; no GPU needed):

    .venv\\Scripts\\python.exe scripts\\real_api_smoke.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_WORKTREE_ROOT = _HERE.parent
_MAIN = Path(r"C:\STC\ultronPrototype")
sys.path.insert(0, str(_MAIN))
sys.path.insert(0, str(_WORKTREE_ROOT / "src"))


# Pull .env so ULTRON_BRAVE_API_KEY is available.
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


def smoke_brave_one() -> dict:
    """1 Brave query.  Returns count + first-title fingerprint."""
    from ultron.web_search.brave import BraveSearchClient

    print("\n[Brave] 1 live query")
    print("-" * 50)
    client = BraveSearchClient()
    t0 = time.monotonic()
    try:
        results = client.search("python 3.13 release notes", count=3)
    except Exception as exc:
        print(f"  ERROR: {exc!r}")
        return {"ok": False, "error": repr(exc)}
    elapsed_ms = (time.monotonic() - t0) * 1000
    out = {
        "ok": True,
        "count": len(results),
        "elapsed_ms": round(elapsed_ms, 1),
        "first_title": results[0].title if results else "",
        # Don't include URLs; ranks may shift between runs.
    }
    print(f"  count={out['count']}  elapsed={out['elapsed_ms']}ms")
    if results:
        print(f"  first title: {results[0].title!r}")
    return out


def smoke_search_chain() -> dict:
    """1 full Brave -> Jina chain via WebSearchExecutor."""
    from ultron.web_search.brave import BraveSearchClient
    from ultron.web_search.jina import JinaReaderClient
    from ultron.web_search.search import WebSearchExecutor

    print("\n[Search chain] 1 Brave + Jina round-trip")
    print("-" * 50)
    exec_ = WebSearchExecutor(brave=BraveSearchClient(), jina=JinaReaderClient(), llm=None)
    t0 = time.monotonic()
    try:
        payload = exec_.run(
            "what's new in python 3.13",
            search_queries=["python 3.13 changelog"],
            top_n=2,
        )
    except Exception as exc:
        print(f"  ERROR: {exc!r}")
        return {"ok": False, "error": repr(exc)}
    elapsed_ms = (time.monotonic() - t0) * 1000
    out = {
        "ok": True,
        "elapsed_ms": round(elapsed_ms, 1),
        "sources": len(payload.sources),
        "cache_hit": payload.cache_hit,
        # Sample: first 80 chars of first source's snippet (avoid title due
        # to ranking variability).
        "first_snippet_prefix": (
            (payload.sources[0].snippet or "")[:80] if payload.sources else ""
        ),
        "any_full_text": any((s.full_text or "") for s in payload.sources),
    }
    print(f"  sources={out['sources']}  elapsed={out['elapsed_ms']}ms  cache={out['cache_hit']}  full_text={out['any_full_text']}")
    return out


def smoke_claude_code_one() -> dict:
    """1 minimal Claude Code subprocess invocation.

    Prompt: "Print exactly the single line: SMOKE_OK".  Should complete in
    1-3 seconds and burn ~50-150 tokens total.
    """
    import subprocess

    print("\n[Claude Code] 1 minimal subprocess invocation")
    print("-" * 50)

    claude_cli = os.environ.get(
        "ULTRON_CLAUDE_CLI",
        str(Path.home() / "AppData" / "Roaming" / "npm" / "claude.cmd"),
    )
    if not Path(claude_cli).exists():
        print(f"  Claude CLI not found at {claude_cli}")
        return {"ok": False, "error": f"missing CLI: {claude_cli}"}

    # We don't use --output-format stream-json here -- this is a sanity
    # ping, not the bridge.  --print + --model haiku keeps it cheap and
    # short.
    cmd = [
        claude_cli, "--print",
        "--model", "haiku",
        "--dangerously-skip-permissions",
        "Reply with exactly the single line: SMOKE_OK",
    ]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout 60s"}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}
    elapsed_ms = (time.monotonic() - t0) * 1000
    out = {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "elapsed_ms": round(elapsed_ms, 1),
        "stdout_first_line": (proc.stdout or "").strip().splitlines()[0] if (proc.stdout or "").strip() else "",
        "stdout_chars": len(proc.stdout or ""),
        "stderr_chars": len(proc.stderr or ""),
        "contains_smoke_ok": "SMOKE_OK" in (proc.stdout or ""),
    }
    print(f"  rc={out['returncode']}  elapsed={out['elapsed_ms']}ms  contains_SMOKE_OK={out['contains_smoke_ok']}")
    print(f"  first stdout line: {out['stdout_first_line']!r}")
    return out


def main() -> int:
    _load_dotenv()
    if not os.environ.get("ULTRON_BRAVE_API_KEY"):
        print("ULTRON_BRAVE_API_KEY missing from environment; skipping Brave probes.")
        brave_result = {"ok": False, "skipped": True}
        chain_result = {"ok": False, "skipped": True}
    else:
        brave_result = smoke_brave_one()
        chain_result = smoke_search_chain()

    claude_result = smoke_claude_code_one()

    log_dir = _WORKTREE_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output_path = log_dir / f"real_api_smoke_{ts}.json"
    output_path.write_text(json.dumps({
        "started_at": datetime.now(timezone.utc).isoformat(),
        "brave_one_query": brave_result,
        "search_chain": chain_result,
        "claude_code_one": claude_result,
    }, indent=2, default=str))

    print()
    print("=" * 60)
    print(f"Done.  Result -> {output_path}")
    print("=" * 60)

    # Exit non-zero if the chain failed (used by CI gates).
    overall_ok = brave_result.get("ok") and chain_result.get("ok") and claude_result.get("ok")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
