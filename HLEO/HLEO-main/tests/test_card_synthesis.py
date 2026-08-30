"""
Tests for FASE 8 — on-demand per-card synthesis endpoint POST /synthesis/card.

Covers:
  - endpoint reachable without a key → 503
  - happy path: returns structured synthesis with provenance
  - no content → error
  - LLM quota exhausted → quota_exhausted flag
  - LLM failure after retries → error message
  - RWE vs scientific evidence_type labelling
  - provenance (source_id, source, evidence_tier) attached
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ok_response(content=None):
    """Build an OpenAI-like response object."""
    if content is None:
        content = json.dumps({
            "summary": "Test summary",
            "key_points": ["point 1", "point 2"],
            "relevance_to_query": "direct",
            "evidence_type": "scientific",
            "confidence": {"level": "moderate", "rationale": "small sample"},
            "limitations": ["single study"],
        })
    m = type("M", (), {"content": content})()
    c = type("C", (), {"message": m})()
    return type("R", (), {"choices": [c]})()


def _make_body(**kw):
    base = {
        "query": "finasteride hair shedding",
        "title": "Finasteride and hair loss: a retrospective study",
        "abstract": "We studied 200 patients taking finasteride. 15% reported shedding.",
        "source": "pubmed",
        "pmid": "12345",
        "source_type": "scientific_article",
        "evidence_tier": "RCT",
        "language": "en",
    }
    base.update(kw)
    return base


# ── Tests ────────────────────────────────────────────────────────────────────

class TestCardSynthesisNoKey:
    def test_503_without_openai_key(self, client):
        # No OPENAI_API_KEY in test env (conftest pops it).
        resp = client.post("/synthesis/card", json=_make_body())
        assert resp.status_code == 503
        assert "OPENAI_API_KEY" in resp.json()["detail"]


class TestCardSynthesisNoContent:
    def test_error_when_no_content(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        body = _make_body(abstract="", text="")
        resp = client.post("/synthesis/card", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data
        assert "No content" in data["error"]


class TestCardSynthesisHappyPath:
    def test_returns_structured_synthesis_with_provenance(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("openai.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create.return_value = _ok_response()
            resp = client.post("/synthesis/card", json=_make_body())

        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"] == "Test summary"
        assert len(data["key_points"]) == 2
        assert data["relevance_to_query"] == "direct"
        assert data["evidence_type"] == "scientific"
        assert data["confidence"]["level"] == "moderate"
        # Provenance attached
        assert data["source"] == "pubmed"
        assert "PMID 12345" in data["source_id"]
        assert data["evidence_tier"] == "RCT"
        assert data["query"] == "finasteride hair shedding"

    def test_llm_called_once_per_request(self, client, monkeypatch):
        """On-demand synthesis = exactly 1 LLM call (no auto-batch)."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("openai.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create.return_value = _ok_response()
            client.post("/synthesis/card", json=_make_body())

        assert mock_client.chat.completions.create.call_count == 1


class TestCardSynthesisRWE:
    def test_rwe_evidence_type_label(self, client, monkeypatch):
        """RWE records should be synthesised with experiential evidence_type."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        rwe_body = _make_body(
            query="finasteride shedding",
            title="My shedding experience",
            abstract="I started finasteride and noticed shedding after 2 weeks.",
            text="I started finasteride and noticed shedding after 2 weeks.",
            source="reddit",
            pmid="",
            external_id="abc123",
            source_type="community_forum",
            evidence_tier="anecdotal",
        )
        with patch("openai.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create.return_value = _ok_response(json.dumps({
                "summary": "Patient reported shedding 2 weeks after finasteride.",
                "key_points": ["shedding after 2 weeks"],
                "relevance_to_query": "direct",
                "evidence_type": "experiential",
                "confidence": {"level": "low", "rationale": "anecdotal report"},
                "limitations": ["single anecdote"],
            }))
            resp = client.post("/synthesis/card", json=rwe_body)

        data = resp.json()
        assert data["evidence_type"] == "experiential"
        assert data["source"] == "reddit"
        assert "ID abc123" in data["source_id"]
        assert data["evidence_tier"] == "anecdotal"

    def test_faers_record_synthesised(self, client, monkeypatch):
        """FAERS records get spontaneous_report labelling."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        faers_body = _make_body(
            query="dutasteride hair shedding",
            title="FAERS report",
            abstract="alopecia; hair shedding",
            text="alopecia; hair shedding",
            source="openfda_faers",
            pmid="",
            external_id="FAERS-999",
            source_type="pharmacovigilance",
            evidence_tier="spontaneous_report",
            treatment="DUTASTERIDE",
        )
        with patch("openai.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create.return_value = _ok_response(json.dumps({
                "summary": "Spontaneous report of alopecia with dutasteride.",
                "key_points": ["alopecia reported"],
                "relevance_to_query": "direct",
                "evidence_type": "spontaneous_report",
                "confidence": {"level": "low", "rationale": "spontaneous report"},
                "limitations": ["no causality assessment"],
            }))
            resp = client.post("/synthesis/card", json=faers_body)

        data = resp.json()
        assert data["evidence_type"] == "spontaneous_report"
        assert data["source"] == "openfda_faers"
        assert "FAERS-999" in data["source_id"]


class TestCardSynthesisQuotaAndFailure:
    def test_quota_exhausted_returns_flag(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("openai.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create.side_effect = Exception(
                "insufficient_quota: you exceeded your current quota"
            )
            resp = client.post("/synthesis/card", json=_make_body())

        data = resp.json()
        assert data.get("quota_exhausted") is True
        assert "quota" in data["error"].lower()

    def test_llm_failure_returns_error(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setattr("core.llm_guard.time.sleep", lambda d: None)
        with patch("openai.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create.side_effect = Exception("server error")
            resp = client.post("/synthesis/card", json=_make_body())

        data = resp.json()
        assert "error" in data
        assert "failed" in data["error"].lower()
