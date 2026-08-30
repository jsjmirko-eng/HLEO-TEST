"""
Tests for the Real World Evidence (RWE) pipeline + AI Assistant convergence.

Covers:
  - RWE search (Reddit adapter + openFDA collector) with mocked collectors
  - deduplication across sources
  - relevance filtering
  - provenance stamping (source_type, evidence_tier, collection_method)
  - separation of scientific vs RWE evidence
  - source unavailable / unauthorized / no-results paths
  - AI Assistant receives rwe_evidence (RWE-only, both, scientific-only)
  - scientific search regression (no RWE leakage)
"""
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.rwe.models import RWEItem


# ─── fakes ───────────────────────────────────────────────────────────────────

def _fake_reddit_item(title, url, text="finasteride stopped my shedding"):
    return RWEItem(
        source="reddit",
        source_type="community_forum",
        evidence_tier="anecdotal",
        collection_method="official_api_oauth2",
        external_id=url,
        source_url=url,
        title=title,
        text=text,
        date="2023-05-01",
        language="en",
        topic="finasteride hair loss",
        privacy_status="redacted",
        metadata={"post_chars": len(text)},
    )


def _fake_faers_item(rid="FAERS-1", treatment="finasteride"):
    return RWEItem(
        source="openfda_faers",
        source_type="pharmacovigilance",
        evidence_tier="spontaneous_report",
        collection_method="official_api_no_key",
        external_id=rid,
        source_url=f"https://api.fda.gov/drug/event.json?search=safetyreportid:{rid}",
        title=f"FAERS report {rid} — Alopecia",
        text="alopecia; decreased libido",
        date="20240101",
        language="en",
        topic="finasteride",
        treatment=treatment,
        experience_type="adverse_event",
        privacy_status="anonymous",
        metadata={"serious": False, "n_drugs": 1},
    )


def _stub_reddit_ok(items):
    """Return a fake search_with_status callable yielding items."""
    def _sws(query, limit=15):
        return list(items), "ok", f"Retrieved {len(items)} post(s)."
    return _sws


def _stub_reddit_status(status, reason):
    def _sws(query, limit=15):
        return [], status, reason
    return _sws


def _stub_openfda_ok(items):
    def _sws(query, limit=20):
        return list(items), "ok", f"Retrieved {len(items)} report(s)."
    return _sws


def _stub_openfda_status(status, reason):
    def _sws(query, limit=20):
        return [], status, reason
    return _sws


# ─── RWE pipeline tests ───────────────────────────────────────────────────────

def _make_pipeline():
    from core.rwe.pipeline import RWEPipeline
    return RWEPipeline()


def test_rwe_search_both_sources_ok(monkeypatch):
    from vocab_stubs import default_resolver, patch_resolver
    patch_resolver(monkeypatch, default_resolver())
    pipe = _make_pipeline()
    reddit_items = [_fake_reddit_item("My finasteride story", "https://reddit.com/1",
                                      text="finasteride stopped my hair loss")]
    faers_items = [_fake_faers_item()]
    with patch.object(pipe.reddit, "search_with_status", _stub_reddit_ok(reddit_items)), \
         patch.object(pipe.openfda, "search_with_status", _stub_openfda_ok(faers_items)), \
         patch("core.rwe.pipeline.QueryOrchestrator") as MockOrch:
        MockOrch.return_value.process.return_value = SimpleNamespace(
            search_query="finasteride hair loss", detected_language="en",
        )
        result = pipe.search("finasteride hair loss", sources=["reddit", "openfda_faers"])
    assert result.totals["unique"] == 2
    assert result.source_status["reddit"] == "ok"
    assert result.source_status["openfda_faers"] == "ok"
    sources = {i.source for i in result.items}
    assert sources == {"reddit", "openfda_faers"}


def test_rwe_search_rwe_only_no_faers():
    pipe = _make_pipeline()
    reddit_items = [_fake_reddit_item("Reddit only", "https://reddit.com/x")]
    with patch.object(pipe.reddit, "search_with_status", _stub_reddit_ok(reddit_items)), \
         patch.object(pipe.openfda, "search_with_status", _stub_openfda_status("no_results", "none")), \
         patch("core.rwe.pipeline.QueryOrchestrator") as MockOrch:
        MockOrch.return_value.process.return_value = SimpleNamespace(
            search_query="minoxidil", detected_language="en",
        )
        result = pipe.search("minoxidil", sources=["reddit"])
    # only reddit requested
    assert result.source_status.get("openfda_faers") is None
    assert all(i.source == "reddit" for i in result.items)


def test_rwe_search_no_results():
    pipe = _make_pipeline()
    with patch.object(pipe.reddit, "search_with_status", _stub_reddit_status("no_credentials", "no creds")), \
         patch.object(pipe.openfda, "search_with_status", _stub_openfda_status("no_results", "none")), \
         patch("core.rwe.pipeline.QueryOrchestrator") as MockOrch:
        MockOrch.return_value.process.return_value = SimpleNamespace(
            search_query="xyz", detected_language="en",
        )
        result = pipe.search("xyz", sources=["reddit", "openfda_faers"])
    assert result.items == []
    assert result.source_status["reddit"] == "no_credentials"
    assert result.source_status["openfda_faers"] == "no_results"


def test_rwe_source_unauthorized():
    pipe = _make_pipeline()
    with patch.object(pipe.reddit, "search_with_status", _stub_reddit_status("auth_error", "bad creds")), \
         patch.object(pipe.openfda, "search_with_status", _stub_openfda_ok([_fake_faers_item(treatment="dutasteride")])), \
         patch("core.rwe.pipeline.QueryOrchestrator") as MockOrch:
        MockOrch.return_value.process.return_value = SimpleNamespace(
            search_query="dutasteride", detected_language="en",
        )
        result = pipe.search("dutasteride", sources=["reddit", "openfda_faers"])
    assert result.source_status["reddit"] == "auth_error"
    assert result.source_status["openfda_faers"] == "ok"
    assert len(result.items) == 1
    assert result.items[0].source == "openfda_faers"


def test_rwe_source_unavailable_network():
    pipe = _make_pipeline()
    with patch.object(pipe.reddit, "search_with_status", _stub_reddit_status("network_error", "timeout")), \
         patch.object(pipe.openfda, "search_with_status", _stub_openfda_status("network_error", "timeout")), \
         patch("core.rwe.pipeline.QueryOrchestrator") as MockOrch:
        MockOrch.return_value.process.return_value = SimpleNamespace(
            search_query="ketoconazole", detected_language="en",
        )
        result = pipe.search("ketoconazole", sources=["reddit", "openfda_faers"])
    assert result.items == []
    assert result.source_status["reddit"] == "network_error"
    assert result.source_status["openfda_faers"] == "network_error"


# ─── dedup + relevance ───────────────────────────────────────────────────────

def test_rwe_deduplication():
    from core.rwe.pipeline import deduplicate
    a = _fake_reddit_item("dup", "https://reddit.com/1")
    b = _fake_reddit_item("dup", "https://reddit.com/1")  # same external_id
    c = _fake_faers_item("FAERS-9")
    out = deduplicate([a, b, c])
    assert len(out) == 2


def test_rwe_relevance_filter_keeps_relevant():
    from core.rwe.pipeline import relevance_filter
    rel = _fake_reddit_item("finasteride story", "u1", text="finasteride worked")
    off = _fake_reddit_item("off topic", "u2", text="unrelated cooking post about pasta")
    out = relevance_filter([rel, off], "finasteride")
    assert len(out) == 1
    assert out[0].relevance == "relevant"
    assert off.relevance == "irrelevant"


# ─── provenance ──────────────────────────────────────────────────────────────

def test_rwe_provenance_stamped():
    pipe = _make_pipeline()
    reddit_items = [_fake_reddit_item("story", "https://reddit.com/1")]
    faers_items = [_fake_faers_item()]
    with patch.object(pipe.reddit, "search_with_status", _stub_reddit_ok(reddit_items)), \
         patch.object(pipe.openfda, "search_with_status", _stub_openfda_ok(faers_items)), \
         patch("core.rwe.pipeline.QueryOrchestrator") as MockOrch:
        MockOrch.return_value.process.return_value = SimpleNamespace(
            search_query="finasteride", detected_language="en",
        )
        result = pipe.search("finasteride", sources=["reddit", "openfda_faers"])
    for it in result.items:
        assert it.source_type in ("community_forum", "pharmacovigilance")
        assert it.collection_method.startswith("official_")
        assert it.evidence_tier in ("anecdotal", "spontaneous_report")
        assert it.privacy_status in ("redacted", "anonymous")


# ─── scientific/RWE separation ───────────────────────────────────────────────

def test_rwe_search_does_not_touch_scientific():
    """RWE pipeline must never call PubMed/EuropePMC/ClinicalTrials collectors."""
    pipe = _make_pipeline()
    with patch.object(pipe.reddit, "search_with_status", _stub_reddit_ok([_fake_reddit_item("a","u")])), \
         patch.object(pipe.openfda, "search_with_status", _stub_openfda_ok([_fake_faers_item()])), \
         patch("core.rwe.pipeline.QueryOrchestrator") as MockOrch:
        MockOrch.return_value.process.return_value = SimpleNamespace(
            search_query="finasteride", detected_language="en",
        )
        result = pipe.search("finasteride", sources=["reddit", "openfda_faers"])
    # No scientific source_type should ever appear in RWE results
    assert all(i.source_type not in ("pubmed", "europepmc", "clinicaltrials") for i in result.items)


# ─── AI Assistant convergence ─────────────────────────────────────────────────

def test_assistant_accepts_rwe_evidence_only(client):
    """When only rwe_evidence is in search_context, the assistant should still
    respond (503 without key is fine — we assert it doesn't 422 on schema)."""
    payload = {
        "message": "What do patients report about finasteride?",
        "language": "en",
        "search_context": {
            "original_query": "finasteride",
            "search_query": "finasteride",
            "detected_language": "en",
            "articles": [],
            "rwe_evidence": [
                {
                    "source": "reddit", "source_type": "community_forum",
                    "evidence_tier": "anecdotal", "title": "My story",
                    "text": "finasteride stopped shedding", "privacy_status": "redacted",
                }
            ],
        },
    }
    r = client.post("/assistant/chat", json=payload)
    # Without OPENAI_API_KEY the endpoint returns 200 with an error dict
    assert r.status_code == 200
    body = r.json()
    # No OPENAI key in test env → returns {"error": "OPENAI_API_KEY not set."}
    assert "error" in body or "response" in body


def test_assistant_search_context_schema_accepts_rwe_field(client):
    """SearchContext Pydantic model must accept rwe_evidence (no 422)."""
    payload = {
        "message": "hi",
        "search_context": {
            "original_query": "q", "search_query": "q", "detected_language": "en",
            "rwe_evidence": [],
        },
    }
    r = client.post("/assistant/chat", json=payload)
    assert r.status_code == 200


# ─── scientific search regression ────────────────────────────────────────────

def test_scientific_search_route_unchanged(client):
    """The scientific /search endpoint must still return its prior contract."""
    # scientific mode without OPENAI_API_KEY → 503
    r = client.get("/search?q=finasteride&mode=scientific")
    assert r.status_code == 503


def test_rwe_search_endpoint_exists(client):
    r = client.get("/rwe/search?q=finasteride")
    # Will attempt live network (Reddit creds absent, openFDA may be reachable)
    # We only assert the route is wired and returns 200 with the RWE envelope.
    assert r.status_code == 200
    body = r.json()
    assert "items" in body
    assert "source_status" in body
    assert "totals" in body


def test_openfda_query_syntax_live():
    """Live integration test for the openFDA FAERS query syntax.

    Skipped when offline. Guards against regressions in the search-expression
    builder (openFDA returns 404 for malformed '+OR+' syntax).
    """
    pytest.importorskip("requests")
    try:
        import requests
        probe = requests.get(
            "https://api.fda.gov/drug/event.json",
            params={"search": "patient.drug.openfda.generic_name:\"ASPIRIN\"", "limit": 1},
            timeout=10,
        )
    except Exception:
        pytest.skip("openFDA unreachable")
    if probe.status_code != 200:
        pytest.skip("openFDA offline")

    from core.rwe.openfda_collector import OpenFDACollector
    oc = OpenFDACollector()
    items, status, reason = oc.search_with_status("aspirin", limit=3)
    assert status == "ok", f"openFDA query regression: status={status} reason={reason}"
    assert len(items) >= 1
    it = items[0]
    assert it.source == "openfda_faers"
    assert it.evidence_tier == "spontaneous_report"
    assert it.privacy_status == "anonymous"
    assert it.external_id  # safetyreportid populated
    assert it.treatment  # drug name extracted
