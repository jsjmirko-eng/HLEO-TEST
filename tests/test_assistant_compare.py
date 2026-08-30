"""
Tests for FASE 15 — POST /assistant/compare (Scientific vs RWE comparison).

Covers:
  - no key → error
  - no evidence → error
  - happy path: returns structured comparison
  - RWE never presented as clinical proof (evidence_quality_note present)
  - quota exhausted → flag
  - LLM failure → error
  - provenance (scientific_count, rwe_count)
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest


def _ok_compare():
    return json.dumps({
        "scientific_consensus": "Literature shows finasteride is effective.",
        "rwe_consensus": "Patients report initial shedding then stabilisation.",
        "agreements": ["Both cite shedding as a temporary phase"],
        "divergences": ["RWE reports more persistent side effects"],
        "gaps_filled_by_rwe": ["Time-to-onset of shedding not in literature"],
        "evidence_quality_note": "RWE is anecdotal; not clinical proof.",
        "practical_takeaway": "Shedding is common initially.",
        "confidence": {"level": "moderate", "rationale": "mixed evidence"},
    })


def _make_body(**kw):
    base = {
        "query": "finasteride hair shedding",
        "language": "en",
        "scientific_articles": [{
            "source": "pubmed",
            "title": "Finasteride and hair loss",
            "abstract": "RCT of 200 patients. 15% reported shedding.",
            "pmid": "12345",
        }],
        "rwe_evidence": [{
            "source": "reddit",
            "source_type": "community_forum",
            "evidence_tier": "anecdotal",
            "external_id": "t3_abc",
            "title": "My shedding experience",
            "text": "I noticed shedding after 2 weeks on finasteride.",
            "treatment": "finasteride",
        }],
    }
    base.update(kw)
    return base


def _mock_resp(content):
    m = type("M", (), {"content": content})()
    c = type("C", (), {"message": m})()
    return type("R", (), {"choices": [c]})()


class TestCompareNoKey:
    def test_no_key_returns_error(self, client):
        resp = client.post("/assistant/compare", json=_make_body())
        assert resp.status_code == 200
        assert resp.json()["error"] == "OPENAI_API_KEY not set."


class TestCompareNoEvidence:
    def test_no_evidence_error(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        resp = client.post("/assistant/compare", json={
            "query": "test", "scientific_articles": [], "rwe_evidence": []
        })
        data = resp.json()
        assert "error" in data
        assert "No evidence" in data["error"]


class TestCompareHappyPath:
    def test_returns_structured_comparison(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = _mock_resp(_ok_compare())
            resp = client.post("/assistant/compare", json=_make_body())

        data = resp.json()
        assert "scientific_consensus" in data
        assert "rwe_consensus" in data
        assert len(data["agreements"]) >= 1
        assert len(data["divergences"]) >= 1
        assert "evidence_quality_note" in data
        assert data["confidence"]["level"] == "moderate"
        # Provenance
        assert data["scientific_count"] == 1
        assert data["rwe_count"] == 1
        assert data["query"] == "finasteride hair shedding"

    def test_rwe_never_presented_as_proof(self, client, monkeypatch):
        """The response must include an evidence_quality_note flagging RWE as non-proof."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = _mock_resp(_ok_compare())
            resp = client.post("/assistant/compare", json=_make_body())

        data = resp.json()
        note = data.get("evidence_quality_note", "").lower()
        assert "anecdotal" in note or "not clinical proof" in note or "not proof" in note

    def test_single_llm_call(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = _mock_resp(_ok_compare())
            client.post("/assistant/compare", json=_make_body())
        assert mock_openai.return_value.chat.completions.create.call_count == 1


class TestCompareQuotaAndFailure:
    def test_quota_exhausted(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = Exception(
                "insufficient_quota"
            )
            resp = client.post("/assistant/compare", json=_make_body())
        data = resp.json()
        assert data.get("quota_exhausted") is True

    def test_llm_failure(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setattr("core.llm_guard.time.sleep", lambda d: None)
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = Exception("server error")
            resp = client.post("/assistant/compare", json=_make_body())
        data = resp.json()
        assert "error" in data
        assert "failed" in data["error"].lower()


class TestCompareEpisodeIds:
    def test_compare_uses_episode_ids(self, client, db_session, monkeypatch):
        """When clinical_profile_episode_ids and rwe_profile_episode_ids are provided,
        the backend must fetch the DB rows and include them in the LLM compare context.
        """
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        from core.models import ClinicalProfile, RWEProfile

        # Create two clinical profiles
        cp1 = ClinicalProfile(
            episode_id="CP1",
            user_id="u",
            extracted_payload={"diagnosis": ["alopecia"], "treatments": ["finasteride"], "outcomes": ["improved"]},
            validation_payload={"title": "Study 1", "pub_year": "2020"},
        )
        cp2 = ClinicalProfile(
            episode_id="CP2",
            user_id="u",
            extracted_payload={"diagnosis": ["alopecia"], "treatments": ["minoxidil"], "outcomes": ["nochange"]},
            validation_payload={"title": "Study 2", "pub_year": "2021"},
        )
        db_session.add(cp1); db_session.add(cp2)

        # Create two RWE profiles
        rp1 = RWEProfile(
            episode_id="RP1",
            source="reddit",
            title="RP1",
            raw_text="I noticed shedding",
            extracted_profile={"condition": "alopecia", "treatment": "finasteride"},
        )
        rp2 = RWEProfile(
            episode_id="RP2",
            source="hairlossexperiences",
            title="RP2",
            raw_text="No improvement",
            extracted_profile={"condition": "alopecia", "treatment": "minoxidil"},
        )
        db_session.add(rp1); db_session.add(rp2)
        db_session.commit()

        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = _mock_resp(_ok_compare())
            resp = client.post("/assistant/compare", json={
                "query": "finasteride hair shedding",
                "language": "en",
                "clinical_profile_episode_ids": ["CP1", "CP2"],
                "rwe_profile_episode_ids": ["RP1", "RP2"],
            })

        data = resp.json()
        assert "scientific_consensus" in data
        assert data["scientific_count"] == 2
        assert data["rwe_count"] == 2
        assert mock_openai.return_value.chat.completions.create.call_count == 1

