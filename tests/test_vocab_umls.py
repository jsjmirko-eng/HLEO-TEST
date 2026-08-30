"""Offline tests for the UMLS provider (UTS REST API).

Every test is hermetic: requests are faked, the API key is a dummy, and the
suite asserts the key never appears in errors/logs. No network, no real key.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests

from core.vocab.umls import UMLSProvider, _tokens

_is_hypernym = UMLSProvider._is_hypernym

KEY = "dummy-test-key"  # sentinel — must never leak into errors/logs


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("UMLS_API_KEY", KEY)
    monkeypatch.setattr("core.vocab.umls.time.sleep", lambda *_a, **_k: None)


def _resp(status=200, payload=None):
    return SimpleNamespace(status_code=status,
                           json=lambda: payload if payload is not None else {},
                           headers={}, text="")


def _search_payload(cui, name):
    return {"result": {"results": [{"ui": cui, "name": name}]}}


def _content_payload(name, stys):
    return {"result": {"name": name,
                       "semanticTypes": [{"name": s} for s in stys]}}


def _atoms_payload(names):
    return {"result": [{"name": n} for n in names]}


def _fake_http(monkeypatch, routes):
    """routes: list of (substring-in-url-or-searchType, payload/exception)."""
    def fake_get(url, params=None, headers=None, auth=None, timeout=None):
        probe = url + " " + str((params or {}).get("searchType", ""))
        for key, value in routes:
            if key in probe:
                if isinstance(value, Exception):
                    raise value
                return _resp(payload=value)
        return _resp(payload={"result": {"results": []}})
    monkeypatch.setattr("core.vocab.umls.requests.get", fake_get)
    return fake_get


# ── 1. API key absent ────────────────────────────────────────────────────────

def test_no_key_inactive_and_silent(monkeypatch):
    monkeypatch.delenv("UMLS_API_KEY", raising=False)
    monkeypatch.delenv("HLEO_UMLS_API_KEY", raising=False)
    p = UMLSProvider()
    assert p.available() is False
    assert p.search("alopecia") == []
    assert p.get_synonyms("C0002170") == []
    assert p.get_concept("C0002170") is None


# ── 2. API key present + valid concept with synonyms ─────────────────────────

def test_valid_concept_with_synonyms(monkeypatch):
    routes = [
        ("search/current exact", _search_payload("C0002170", "Alopecia")),
        ("content/current/CUI/C0002170/atoms",
         _atoms_payload(["Alopecia", "Hair Loss", "Baldness"])),
        ("content/current/CUI/C0002170",
         _content_payload("Alopecia", ["Disease or Syndrome"])),
    ]
    _fake_http(monkeypatch, routes)
    p = UMLSProvider()
    assert p.available() is True
    matches = p.search("alopecia")
    assert len(matches) == 1
    m = matches[0]
    assert m.provider == "umls"
    assert m.concept_id == "C0002170"
    assert m.preferred_term == "Alopecia"
    assert m.semantic_group == "condition"
    assert m.match_kind == "exact"
    assert "Hair Loss" in m.synonyms and "Baldness" in m.synonyms
    assert "umls" in m.source_url


# ── 3. HTTP error degrades without leaking the key ──────────────────────────

def test_http_error_no_key_leak(monkeypatch, caplog):
    def fake_get(url, params=None, headers=None, auth=None, timeout=None):
        return _resp(status=401)
    monkeypatch.setattr("core.vocab.umls.requests.get", fake_get)
    p = UMLSProvider()
    assert p.search("alopecia") == []          # degraded, no raise
    for rec in caplog.records:
        assert KEY not in rec.getMessage()


def test_exception_sanitised(monkeypatch):
    p = UMLSProvider()
    monkeypatch.setattr("core.vocab.umls.requests.get",
                        lambda *a, **k: _resp(status=503))
    with pytest.raises(RuntimeError) as exc:
        p._get_json(f"https://uts-ws.nlm.nih.gov/rest/x?apiKey={KEY}")
    assert KEY not in str(exc.value)
    assert "503" in str(exc.value)


# ── 4. Timeout ───────────────────────────────────────────────────────────────

def test_timeout_degrades(monkeypatch):
    monkeypatch.setattr(
        "core.vocab.umls.requests.get",
        lambda *a, **k: (_ for _ in ()).throw(requests.Timeout("slow")))
    p = UMLSProvider()
    assert p.search("finasteride") == []


# ── 5. Empty UMLS response ───────────────────────────────────────────────────

def test_empty_response(monkeypatch):
    _fake_http(monkeypatch, [])
    p = UMLSProvider()
    assert p.search("zzzz-not-a-concept") == []


# ── 6. Rate limit (429) retried once ─────────────────────────────────────────

def test_rate_limit_retried_once(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, params=None, headers=None, auth=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _resp(status=429)
        probe = url + " " + str((params or {}).get("searchType", ""))
        if "search/current exact" in probe:
            return _resp(payload=_search_payload("C0002170", "Alopecia"))
        if "atoms" in probe:
            return _resp(payload=_atoms_payload(["Alopecia"]))
        return _resp(payload=_content_payload("Alopecia", ["Disease or Syndrome"]))

    monkeypatch.setattr("core.vocab.umls.requests.get", fake_get)
    p = UMLSProvider()
    matches = p.search("alopecia")
    assert len(matches) == 1
    assert calls["n"] >= 2  # first 429 + retry


# ── 7. Generic vs specific (hypernym guard) ──────────────────────────────────

def test_hypernym_rejected_for_specific_query(monkeypatch):
    # UMLS returns generic "Pain" for the multi-token query "joint pain" in
    # the normalized pass → must be dropped (specific concept keeps priority).
    routes = [
        ("search/current exact", {"result": {"results": []}}),
        ("search/current normalizedWords",
         _search_payload("C0030193", "Pain")),
        ("content/current/CUI/C0030193/atoms", _atoms_payload(["Pain"])),
        ("content/current/CUI/C0030193",
         _content_payload("Pain", ["Sign or Symptom"])),
    ]
    _fake_http(monkeypatch, routes)
    p = UMLSProvider()
    assert p.search("joint pain") == []


def test_hypernym_helper():
    assert _is_hypernym("joint pain", "Pain") is True
    assert _is_hypernym("joint pain", "Joint pain") is False
    assert _is_hypernym("alopecia", "Alopecia") is False
    assert _tokens("Joint Pain!") == ["joint", "pain"]


# ── 8. Trichology Italian query: normalizedWords synonym match ──────────────

def test_trichology_english_concept(monkeypatch):
    routes = [
        ("search/current exact",
         _search_payload("C0020617", "Hypotrichosis")),
        ("content/current/CUI/C0020617/atoms",
         _atoms_payload(["Hypotrichosis", "Hair loss", "Loss of hair"])),
        ("content/current/CUI/C0020617",
         _content_payload("Hypotrichosis", ["Disease or Syndrome"])),
    ]
    _fake_http(monkeypatch, routes)
    p = UMLSProvider()
    matches = p.search("hair loss")
    assert len(matches) == 1
    m = matches[0]
    # queried "hair loss" != preferred "Hypotrichosis" but IS a synonym
    assert m.match_kind == "synonym"
    assert m.confidence <= 0.9
    assert m.semantic_group == "condition"


# ── 9. get_concept ───────────────────────────────────────────────────────────

def test_get_concept(monkeypatch):
    routes = [
        ("/atoms", _atoms_payload(["Minoxidil", "Rogaine"])),
        ("content/current/CUI/C0026171",
         _content_payload("Minoxidil", ["Pharmacologic Substance"])),
    ]
    _fake_http(monkeypatch, routes)
    p = UMLSProvider()
    m = p.get_concept("C0026171")
    assert m is not None
    assert m.semantic_group == "drug"
    assert m.match_kind == "canonical"
    assert "Rogaine" in m.synonyms
    assert p.get_concept("") is None


# ── 10. resolver registers umls by default, inactive without key ────────────

def test_resolver_default_includes_umls_inactive(monkeypatch):
    monkeypatch.delenv("UMLS_API_KEY", raising=False)
    monkeypatch.delenv("HLEO_UMLS_API_KEY", raising=False)
    monkeypatch.delenv("HLEO_VOCAB_PROVIDERS", raising=False)
    from core.vocab.resolver import VocabularyResolver
    r = VocabularyResolver()
    names = [p.name for p in r.providers]
    assert "umls" in names
    assert "umls" not in r.active_providers()
    # pipeline not broken: resolve_terms still works with the other providers
    res = r.resolve_terms(["alopecia"])
    assert "alopecia" in res
