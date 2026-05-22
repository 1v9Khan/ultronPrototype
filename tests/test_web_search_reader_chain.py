"""Reader-chain tests (frontier 2026-05-21).

Local-first page-extraction cascade: trafilatura -> Jina Reader.

No real HTTP -- the reader clients are mocked. Tests cover:
- Config schema for the new ``readers`` list + ``trafilatura`` block.
- Chain construction with valid + invalid reader IDs.
- First non-empty wins (trafilatura succeeds -> Jina NEVER called).
- Empty/None falls through (trafilatura empty -> Jina called).
- Exception falls through (defence-in-depth).
- Provider construction failure is silently skipped.
- Empty URL short-circuits.
- All readers None -> chain returns None.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ultron.config import (
    TrafilaturaConfig,
    UltronConfig,
    WebSearchConfig,
)


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


def test_trafilatura_config_defaults():
    cfg = TrafilaturaConfig()
    assert cfg.timeout_seconds == 6.0
    assert cfg.max_bytes == 200_000


def test_trafilatura_config_validates_timeout():
    with pytest.raises(Exception):                                       # noqa: PT011
        TrafilaturaConfig(timeout_seconds=-1.0)
    with pytest.raises(Exception):                                       # noqa: PT011
        TrafilaturaConfig(timeout_seconds=0.0)


def test_trafilatura_config_validates_max_bytes():
    with pytest.raises(Exception):                                       # noqa: PT011
        TrafilaturaConfig(max_bytes=-1)
    # 0 is valid (means "no truncation").
    TrafilaturaConfig(max_bytes=0)


def test_web_search_default_readers():
    cfg = WebSearchConfig()
    assert cfg.readers == ["trafilatura", "jina"]


def test_web_search_round_trip_with_legacy_readers():
    """An operator can disable the local reader by setting only
    ``jina`` in the readers list (legacy behaviour)."""
    cfg = UltronConfig.model_validate({
        "web_search": {"readers": ["jina"]}
    })
    assert cfg.web_search.readers == ["jina"]


# ---------------------------------------------------------------------------
# Chain construction
# ---------------------------------------------------------------------------


def test_chain_default_construction():
    from ultron.web_search.reader_chain import ReaderChain
    chain = ReaderChain()
    assert chain.reader_ids == ["trafilatura", "jina"]


def test_chain_custom_construction():
    from ultron.web_search.reader_chain import ReaderChain
    chain = ReaderChain(["jina"])
    assert chain.reader_ids == ["jina"]


def test_chain_rejects_empty_list():
    from ultron.web_search.reader_chain import ReaderChain
    with pytest.raises(ValueError):
        ReaderChain([])


def test_chain_rejects_unknown_reader():
    from ultron.web_search.reader_chain import ReaderChain
    with pytest.raises(ValueError) as exc_info:
        ReaderChain(["beautifulsoup"])
    assert "Unknown" in str(exc_info.value)


def test_chain_normalises_case():
    from ultron.web_search.reader_chain import ReaderChain
    chain = ReaderChain(["TRAFILATURA", "JINA"])
    assert chain.reader_ids == ["trafilatura", "jina"]


# ---------------------------------------------------------------------------
# Chain behaviour with mocked readers
# ---------------------------------------------------------------------------


def _stub_reader(result):
    r = MagicMock()
    r.fetch.return_value = result
    return r


def test_chain_first_non_empty_wins():
    """When trafilatura returns text, Jina is never called."""
    from ultron.web_search.reader_chain import ReaderChain
    traf = _stub_reader("# Local extraction\nContent here.")
    jina = _stub_reader("# Jina extraction\nDifferent content.")
    chain = ReaderChain(["trafilatura", "jina"])
    chain._clients = {"trafilatura": traf, "jina": jina}

    out = chain.fetch("https://example.test/article")
    assert out is not None
    assert "Local extraction" in out
    traf.fetch.assert_called_once_with("https://example.test/article")
    jina.fetch.assert_not_called()


def test_chain_falls_through_on_none():
    """trafilatura returns None -> Jina called."""
    from ultron.web_search.reader_chain import ReaderChain
    traf = _stub_reader(None)
    jina = _stub_reader("# Jina got it")
    chain = ReaderChain(["trafilatura", "jina"])
    chain._clients = {"trafilatura": traf, "jina": jina}

    out = chain.fetch("https://spa.test/page")
    assert out == "# Jina got it"
    traf.fetch.assert_called_once()
    jina.fetch.assert_called_once()


def test_chain_falls_through_on_empty_string():
    """trafilatura returns empty string -> treated as failure -> Jina called."""
    from ultron.web_search.reader_chain import ReaderChain
    traf = _stub_reader("   ")
    jina = _stub_reader("# Jina got it")
    chain = ReaderChain(["trafilatura", "jina"])
    chain._clients = {"trafilatura": traf, "jina": jina}

    out = chain.fetch("https://spa.test/page")
    assert out == "# Jina got it"
    jina.fetch.assert_called_once()


def test_chain_falls_through_on_exception():
    """A reader that raises (vs returning None) is also caught."""
    from ultron.web_search.reader_chain import ReaderChain
    traf = MagicMock()
    traf.fetch.side_effect = RuntimeError("simulated parser crash")
    jina = _stub_reader("# Jina recovered")
    chain = ReaderChain(["trafilatura", "jina"])
    chain._clients = {"trafilatura": traf, "jina": jina}

    out = chain.fetch("https://bad.test")
    assert out == "# Jina recovered"
    jina.fetch.assert_called_once()


def test_chain_skips_unconstructable_reader(monkeypatch):
    """Reader whose factory raises gets skipped without crashing."""
    from ultron.web_search import reader_chain as rc_module

    rc_module.ReaderChain._READER_FACTORIES = {
        "trafilatura": lambda: (_ for _ in ()).throw(
            ImportError("trafilatura missing"),
        ),
        "jina": lambda: _stub_reader("# Jina to the rescue"),
    }
    chain = rc_module.ReaderChain(["trafilatura", "jina"])
    out = chain.fetch("https://example.test")
    assert out == "# Jina to the rescue"


def test_chain_all_readers_none_returns_none():
    from ultron.web_search.reader_chain import ReaderChain
    chain = ReaderChain(["trafilatura", "jina"])
    chain._clients = {
        "trafilatura": _stub_reader(None),
        "jina": _stub_reader(None),
    }
    assert chain.fetch("https://example.test") is None


def test_chain_empty_url_short_circuits():
    from ultron.web_search.reader_chain import ReaderChain
    chain = ReaderChain(["trafilatura"])
    r = _stub_reader("anything")
    chain._clients = {"trafilatura": r}
    assert chain.fetch("") is None
    assert chain.fetch("   ") is None
    r.fetch.assert_not_called()


# ---------------------------------------------------------------------------
# TrafilaturaReaderClient direct smoke tests (no real network)
# ---------------------------------------------------------------------------


def test_trafilatura_client_imports():
    from ultron.web_search.trafilatura_reader import TrafilaturaReaderClient
    client = TrafilaturaReaderClient()
    assert client.timeout_s == 6.0
    assert client.max_bytes == 200_000


def test_trafilatura_empty_url_returns_none():
    from ultron.web_search.trafilatura_reader import TrafilaturaReaderClient
    client = TrafilaturaReaderClient()
    assert client.fetch("") is None
    assert client.fetch("   ") is None
