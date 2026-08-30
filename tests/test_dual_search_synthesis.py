"""Backend tests for dual-mode search + Level 3 synthesis.

These tests mock OpenAI / RelationalSearch to avoid live network calls and
verify the new mode-aware behavior and the /synthesis contract.
"""
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import api.main as appmod


# ─── helpers / fakes ────────────────────────────────────────────────────────

def _fake_article(pmid, label="relevant"):
    """A minimal SearchResult-like object with relevance metadata."""
    md = {"journal": "Derm", "pubdate": "2020-01-01"}
    if label:
        md["relevance_label"] = label
        md["relevance_score"] = 1142.5 if label == "relevant" else 10.0
        md["relevance_reason"] = f"reason for {label}"
    return SimpleNamespace(
        title=f"Article {pmid}", source="pubmed", pmid=pmid, doi=None,
        abstract=f"Abstract for {pmid}", year=2020, score=100.0, metadata=md,
    )


def _fake_relation():
    return SimpleNamespace(
        original_query="rossore dopo minoxidil",
        agent={"normalized": "minoxidil", "term": "minoxidil"},
        event={"normalized": "", "term": ""},
        manifestation={"normalized": "irritation", "term": "irritation"},
        temporal="", relation_type="adverse_effect",
        scientific_query="minoxidil skin irritation adverse effect",
        relation_phrases=["minoxidil causes irritation"], fallback_needed=False,
        to_dict=lambda: {"relation_type": "adverse_effect",
                         "agent": {"normalized": "minoxidil"},
                         "manifestation": {"normalized": "irritation"}},
    )


def _fake_rel_out():
    return {
        "pubmed": [_fake_article("1", "relevant"), _fake_article("2", "not_relevant"),
                   _fake_article("3", "relevant")],
        "europepmc": [],
        "clinicaltrials": [],
        "reddit": [],
        "relation": _fake_relation(),
        "stats": {"total": 3},
    }


# ─── GET /search ────────────────────────────────────────────────────────────

def test_search_scientific_503_without_openai_key(client):
    """Scientific mode must return 503 (not silent fallback) when no LLM key."""
    os.environ.pop("OPENAI_API_KEY", None)
    r = client.get("/search", params={"q": "minoxidil irritation", "mode": "scientific"})
    assert r.status_code == 503
    assert "OPENAI_API_KEY" in r.json()["detail"]


def test_search_global_works_without_llm(monkeypatch, client):
    """Global mode must NOT require an LLM key — broad keyword pipeline."""
    fake_orch = SimpleNamespace(
        search_query="minoxidil", to_dict=lambda: {"search_query": "minoxidil"},
    )
    monkeypatch.setattr(appmod._orchestrator, "process", lambda q: fake_orch)

    class FakePipeline:
        def __init__(self, *a, **k):
            self.extractor = SimpleNamespace(client=None)
        def collect(self, q):
            return {"pubmed": [_fake_article("7", None)],
                    "europepmc": [], "clinicaltrials": [], "reddit": []}
    monkeypatch.setattr("core.pipeline.HLEOPipeline", FakePipeline)

    r = client.get("/search", params={"q": "minoxidil", "mode": "global"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["llm_extraction"] is False
    assert data["totals"]["pubmed"] == 1


def test_search_scientific_returns_results(monkeypatch, client):
    """Scientific mode with a working LLM returns relation + relevant articles."""
    fake_rs = MagicMock()
    fake_rs._client = object()  # truthy → key present
    fake_rs.search = lambda q: _fake_rel_out()
    monkeypatch.setattr("core.relational_search.RelationalSearch", lambda: fake_rs)

    r = client.get("/search", params={"q": "rossore dopo minoxidil", "mode": "scientific"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["llm_extraction"] is True
    rel = data["orchestration"]["relation"]
    assert rel["relation_type"] == "adverse_effect"
    assert data["totals"]["pubmed"] == 3


# ─── helper: relevant-only filtering ───────────────────────────────────────

def test_articles_from_relational_filters_relevant_only():
    from api.main import _articles_from_relational
    rel_out = _fake_rel_out()
    arts = _articles_from_relational(rel_out, only_relevant=True)
    # 2 relevant out of 3
    assert len(arts) == 2
    assert all(a["relevance_label"] == "relevant" for a in arts)
    # relevance metadata carried through
    assert arts[0]["relevance_score"] == 1142.5


def test_articles_from_relational_includes_all_when_no_filter():
    from api.main import _articles_from_relational
    rel_out = _fake_rel_out()
    arts = _articles_from_relational(rel_out, only_relevant=False)
    assert len(arts) == 3


# ─── POST /synthesis ────────────────────────────────────────────────────────

def test_synthesis_503_without_openai_key(client):
    os.environ.pop("OPENAI_API_KEY", None)
    r = client.post("/synthesis", json={"query": "x", "articles": []})
    assert r.status_code == 503


def test_synthesis_no_articles_error(monkeypatch, client):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    r = client.post("/synthesis", json={"query": "x", "articles": []})
    assert r.json()["error"].startswith("No relevant articles")


def test_synthesis_returns_structured_output(monkeypatch, client):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    fake_json = ('{"conclusion": "Evidence suggests minoxidil causes irritation.", '
                 '"evidence_summary": "Two studies concur.", '
                 '"claims": [{"article_index": 0, "claim": "direct irritation reported", '
                 '"evidence_type": "direct", "confidence": "high", "citation": "PMID 1"}], '
                 '"agreements": ["both report irritation"], "contradictions": [], '
                 '"confidence": {"level": "high", "rationale": "two studies"}, '
                 '"key_studies": ["PMID 1"]}')
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=fake_json))]
    )
    monkeypatch.setattr("openai.OpenAI", lambda api_key: fake_client)

    body = {
        "query": "minoxidil irritation",
        "relation": {"relation_type": "adverse_effect",
                     "agent": {"normalized": "minoxidil"},
                     "manifestation": {"normalized": "irritation"}},
        "articles": [{"source": "pubmed", "title": "Art 1", "abstract": "abc",
                       "pmid": "1", "relevance_label": "relevant"}],
    }
    r = client.post("/synthesis", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["conclusion"].startswith("Evidence")
    assert data["confidence"]["level"] == "high"
    assert data["relation"]["relation_type"] == "adverse_effect"
    assert data["query"] == "minoxidil irritation"

