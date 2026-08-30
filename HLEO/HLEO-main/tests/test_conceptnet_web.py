"""
Offline tests for the ConceptNet WEB fallback (WebConceptNetProvider).

The API provider (api.conceptnet.io) stays the PRIMARY channel; the web
frontend (conceptnet.io) is used ONLY when the API is unreachable
(502/503/504, timeout, connection error). These tests run fully offline on
HTML fixtures captured from the live web frontend (tests/fixtures/conceptnet)
covering the cases already verified manually: minoxidil, hypertrichosis,
propecia (not a node), caduta dei capelli (IT), a Japanese node, and a
nonexistent node.

No code under test performs network here: HTTP is stubbed at the boundary.
"""
import os

import pytest

from core.vocab.cache import VocabCache
from core.vocab.conceptnet import ConceptNetProvider

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "conceptnet")


def _fixture(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return f.read()


class _Resp:
    def __init__(self, status_code=200, text="", url="", history=None):
        self.status_code = status_code
        self.text = text
        self.url = url
        self.history = history or []
        self.content = text.encode("utf-8")

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def json(self):
        import json
        return json.loads(self.text)


def _web_get(fixture_map):
    """Build a requests.get stub: fixture_map maps URL-substring → _Resp."""
    def _get(url, params=None, headers=None, auth=None, timeout=None, **kw):
        for key, resp in fixture_map.items():
            if key in url:
                return resp
        return _Resp(404, "not found", url=url)
    return _get


# ── Parsing of the web node page ─────────────────────────────────────────────

def test_web_parse_minoxidil_extracts_synonyms_and_related():
    from core.vocab.conceptnet_web import parse_node_page
    page = parse_node_page(_fixture("minoxidil.html"))
    assert page["canonical"] == "minoxidil"
    assert page["language"] == "en"
    syn_langs = {lang for lang, _t in page["relations"]["Synonym"]}
    # multilingual synonyms (translations) verified live
    assert {"ja", "fi", "zh", "ar", "de", "it"} & syn_langs
    related = {t for _l, t in page["relations"]["RelatedTo"]}
    assert "baldness" in related or "vasodilator" in related


def test_web_parse_hypertrichosis_relations():
    from core.vocab.conceptnet_web import parse_node_page
    page = parse_node_page(_fixture("hypertrichosis.html"))
    assert page["canonical"] == "hypertrichosis"
    assert "Synonym" in page["relations"]
    assert "RelatedTo" in page["relations"]
    assert "FormOf" in page["relations"]
    related = {t for _l, t in page["relations"]["RelatedTo"]}
    assert "body hair" in related


def test_web_parse_italian_node_cross_language():
    from core.vocab.conceptnet_web import parse_node_page
    page = parse_node_page(_fixture("caduta_dei_capelli.html"))
    assert page["language"] == "it"
    syns = page["relations"]["Synonym"]
    assert ("de", "haarausfall") in syns


def test_web_parse_japanese_node():
    from core.vocab.conceptnet_web import parse_node_page
    page = parse_node_page(_fixture("minokishijiru_ja.html"))
    assert page["language"] == "ja"
    syns = page["relations"]["Synonym"]
    assert ("en", "minoxidil") in syns


def test_web_parse_nonexistent_node_is_not_found():
    from core.vocab.conceptnet_web import parse_node_page
    page = parse_node_page(_fixture("nonexistent.html"))
    assert page["not_found"] is True


def test_web_parse_propecia_not_a_node():
    from core.vocab.conceptnet_web import parse_node_page
    page = parse_node_page(_fixture("propecia.html"))
    assert page["not_found"] is True


# ── Fallback activation contract ─────────────────────────────────────────────

def _api_then_web(monkeypatch, api_resp, web_html):
    """Stub requests.get: first call (API) → api_resp; web call → fixture."""
    calls = []

    def _get(url, params=None, headers=None, auth=None, timeout=None, **kw):
        calls.append(url)
        if "api.conceptnet.io" in url:
            return api_resp
        for name, html in web_html.items():
            if name in url:
                return _Resp(200, html, url=url)
        return _Resp(404, "not found", url=url)

    monkeypatch.setattr("core.vocab.base.requests.get", _get)
    monkeypatch.setattr("core.vocab.conceptnet_web.requests.get", _get)
    return calls


def test_api_502_triggers_web_fallback(monkeypatch):
    p = ConceptNetProvider(cache=VocabCache())
    calls = _api_then_web(monkeypatch, _Resp(502, "Bad Gateway"),
                          {"minoxidil": _fixture("minoxidil.html")})
    matches = p.search("minoxidil", language="en")
    assert any("api.conceptnet.io" in u for u in calls)
    assert any("conceptnet.io/c/en/minoxidil" in u for u in calls)
    assert matches, "web fallback must return matches when API is 502"
    assert all(m.provider == "conceptnet" for m in matches)
    assert any(m.metadata.get("via") == "web" for m in matches)


def test_api_503_and_504_trigger_web_fallback(monkeypatch):
    for status in (503, 504):
        p = ConceptNetProvider(cache=VocabCache())
        _api_then_web(monkeypatch, _Resp(status, "err"),
                      {"minoxidil": _fixture("minoxidil.html")})
        assert p.search("minoxidil", language="en"), f"no fallback on {status}"


def test_api_timeout_triggers_web_fallback(monkeypatch):
    import requests as _rq

    def _get(url, **kw):
        if "api.conceptnet.io" in url:
            raise _rq.Timeout("timed out")
        if "minoxidil" in url:
            return _Resp(200, _fixture("minoxidil.html"), url=url)
        return _Resp(404, "", url=url)

    monkeypatch.setattr("core.vocab.base.requests.get", _get)
    monkeypatch.setattr("core.vocab.conceptnet_web.requests.get", _get)
    p = ConceptNetProvider(cache=VocabCache())
    assert p.search("minoxidil", language="en")


def test_api_connection_error_triggers_web_fallback(monkeypatch):
    import requests as _rq

    def _get(url, **kw):
        if "api.conceptnet.io" in url:
            raise _rq.ConnectionError("conn refused")
        if "minoxidil" in url:
            return _Resp(200, _fixture("minoxidil.html"), url=url)
        return _Resp(404, "", url=url)

    monkeypatch.setattr("core.vocab.base.requests.get", _get)
    monkeypatch.setattr("core.vocab.conceptnet_web.requests.get", _get)
    p = ConceptNetProvider(cache=VocabCache())
    assert p.search("minoxidil", language="en")


def test_api_4xx_does_not_trigger_web_fallback(monkeypatch):
    """A 404/other client error from the API is NOT an outage: no web call."""
    p = ConceptNetProvider(cache=VocabCache())
    calls = _api_then_web(monkeypatch, _Resp(404, "not found"),
                          {"minoxidil": _fixture("minoxidil.html")})
    matches = p.search("minoxidil", language="en")
    assert not any("conceptnet.io/c/" in u for u in calls), \
        "web fallback must NOT fire on API 4xx"


def test_api_healthy_no_web_call(monkeypatch):
    """When the API answers 200 the web fallback must stay inactive."""
    api_json = {"edges": [{
        "rel": {"label": "Synonym"},
        "start": {"@id": "/c/en/minoxidil", "label": "minoxidil", "language": "en"},
        "end": {"@id": "/c/ja/ミノキシジル", "label": "ミノキシジル", "language": "ja"},
    }]}
    import json as _json
    p = ConceptNetProvider(cache=VocabCache())
    calls = _api_then_web(monkeypatch, _Resp(200, _json.dumps(api_json)),
                          {"minoxidil": _fixture("minoxidil.html")})
    matches = p.search("minoxidil", language="en")
    assert matches
    assert not any("conceptnet.io/c/" in u for u in calls), \
        "web fallback must NOT fire when the API is healthy"


# ── Output contract identical to the API provider ────────────────────────────

def test_web_matches_contract_translation_and_related(monkeypatch):
    p = ConceptNetProvider(cache=VocabCache())
    _api_then_web(monkeypatch, _Resp(502, "err"),
                  {"minoxidil": _fixture("minoxidil.html")})
    matches = p.search("minoxidil", language="en")
    kinds = {m.match_kind for m in matches}
    assert "translation" in kinds          # cross-language Synonym
    assert "related_concept" in kinds      # RelatedTo/IsA
    for m in matches:
        assert m.provider == "conceptnet"
        assert m.semantic_group == "general"
        assert 0.0 <= m.confidence <= 1.0
        assert m.concept_id.startswith("/c/")


def test_web_not_found_returns_empty_not_invented(monkeypatch):
    p = ConceptNetProvider(cache=VocabCache())
    _api_then_web(monkeypatch, _Resp(502, "err"),
                  {"xyzqwvbnmk": _fixture("nonexistent.html")})
    assert p.search("xyzqwvbnmk", language="en") == []


def test_web_unavailable_when_both_api_and_web_down(monkeypatch):
    """API 502 AND web unreachable → empty (provider_unavailable), never
    invented results."""
    import requests as _rq

    def _get(url, **kw):
        if "api.conceptnet.io" in url:
            return _Resp(502, "err")
        raise _rq.ConnectionError("web down too")

    monkeypatch.setattr("core.vocab.base.requests.get", _get)
    monkeypatch.setattr("core.vocab.conceptnet_web.requests.get", _get)
    p = ConceptNetProvider(cache=VocabCache())
    assert p.search("minoxidil", language="en") == []


def test_web_cjk_term_via_search_redirect(monkeypatch):
    """CJK term: web node lookup follows the same node URL contract."""
    p = ConceptNetProvider(cache=VocabCache())

    def _get(url, **kw):
        if "api.conceptnet.io" in url:
            return _Resp(502, "err")
        if "ミノキシジル" in url or "%E3%83%9F" in url:
            return _Resp(200, _fixture("minokishijiru_ja.html"), url=url)
        return _Resp(404, "", url=url)

    monkeypatch.setattr("core.vocab.base.requests.get", _get)
    monkeypatch.setattr("core.vocab.conceptnet_web.requests.get", _get)
    matches = p.search("ミノキシジル", language="ja")
    assert any(m.match_kind == "translation" and m.preferred_term == "minoxidil"
               for m in matches)
