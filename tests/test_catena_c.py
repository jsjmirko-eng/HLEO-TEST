"""Catena C tests: vocabulary BEFORE retrieval, controlled expansion,
union+dedup with provenance, global judge pool, score>=0.20, max 400,
pagination 30, VOCAB OFF backward-compat.

All tests are offline: collectors, resolver and LLM are stubbed.
"""
import os
from unittest.mock import patch

import pytest

from core.relational_search import ClinicalRelation, RelationalSearch
from core.search_result import SearchResult


def _article(title, abstract, source="PubMed", doi=None, year=2023):
    return SearchResult(
        title=title, source=source, abstract=abstract,
        authors=["A"], year=year, doi=doi, metadata={},
    )


def _relation(query="minoxidil erythema"):
    return ClinicalRelation(
        original_query=query,
        agent={"term": "minoxidil", "normalized": "minoxidil",
               "role": "drug", "search_terms": ["minoxidil"]},
        event={"term": "", "normalized": ""},
        manifestation={"term": "erythema", "normalized": "erythema",
                       "role": "adverse_event", "search_terms": ["erythema"]},
        relation_type="adverse_effect",
        scientific_query="minoxidil erythema",
    )


def _search_with_stubs(monkeypatch, articles_by_query, resolver=None,
                       judge_score=0.9):
    """Build a RelationalSearch whose collectors/LLM are stubbed.

    articles_by_query: callable(query_string, source) -> list[SearchResult]
    """
    rs = RelationalSearch.__new__(RelationalSearch)
    rs._client = object()  # truthy → pipeline does not bail out
    rs._rel_cache = {}

    calls = {"pubmed": [], "europepmc": [], "clinicaltrials": []}

    class _StubCollector:
        def __init__(self, source):
            self.source = source

        def search(self, query, limit=None):
            calls[self.source].append((query, limit))
            return articles_by_query(query, self.source)

    rs.pubmed = _StubCollector("pubmed")
    rs.europepmc = _StubCollector("europepmc")
    rs.clinicaltrials = _StubCollector("clinicaltrials")

    monkeypatch.setattr(RelationalSearch, "_extract_relation",
                        lambda self, q: _relation(q))
    monkeypatch.setattr(RelationalSearch, "_llm_judge",
                        lambda self, batch, rel: [
                            {"i": j, "label": "relevant", "score": judge_score,
                             "reason": "stub"} for j in range(len(batch))])
    monkeypatch.setattr("core.relational_search.time.sleep", lambda *_: None)

    if resolver is not None:
        monkeypatch.setattr("core.vocab.resolver.build_resolver_from_env",
                            lambda: resolver)
    else:
        monkeypatch.setattr("core.vocab.resolver.build_resolver_from_env",
                            lambda: None)
    return rs, calls


class _FakeMatch:
    def __init__(self, preferred_term, synonyms, match_kind, provider="rxnorm"):
        self.preferred_term = preferred_term
        self.synonyms = synonyms
        self.match_kind = match_kind
        self.provider = provider
        self.confidence = 0.9
        self.concept_id = "C1"
        self.semantic_group = "drug"
        self.language = "en"
        self.source_url = ""
        self.metadata = {}

    def model_dump(self):
        return {"preferred_term": self.preferred_term,
                "synonyms": self.synonyms, "match_kind": self.match_kind,
                "provider": self.provider, "confidence": self.confidence,
                "concept_id": self.concept_id,
                "semantic_group": self.semantic_group,
                "language": self.language, "source_url": self.source_url,
                "metadata": self.metadata}


class _FakeResolution:
    def __init__(self, matches):
        self.matches = matches


class _FakeResolver:
    def __init__(self, mapping):
        self._mapping = mapping

    def resolve_terms(self, terms, language="en"):
        return {t: self._mapping[t] for t in terms if t in self._mapping}


# ── 1. Vocabulary expansion reaches the collector (pre-retrieval) ───────────

def test_vocab_expansion_reaches_collector(monkeypatch):
    resolver = _FakeResolver({
        "minoxidil": _FakeResolution([
            _FakeMatch("minoxidil", ["Rogaine"], "synonym"),
        ]),
    })

    def by_query(query, source):
        if "rogaine" in query.lower():
            return [_article("Rogaine induced erythema",
                             "Rogaine caused erythema in patients", source)]
        return [_article("Minoxidil erythema study",
                         "minoxidil erythema trial", source)]

    rs, calls = _search_with_stubs(monkeypatch, by_query, resolver=resolver)
    out = rs.search("minoxidil erythema")
    assert out is not None
    all_queries = [q for q, _ in calls["pubmed"]]
    assert any("rogaine" in q.lower() for q in all_queries), \
        f"vocabulary synonym never reached collector: {all_queries}"
    # original/canonical query also present
    assert any("minoxidil" in q.lower() for q in all_queries)


def test_vocab_off_no_expansion(monkeypatch):
    def by_query(query, source):
        return [_article("Minoxidil erythema", "minoxidil erythema", source)]

    rs, calls = _search_with_stubs(monkeypatch, by_query, resolver=None)
    out = rs.search("minoxidil erythema")
    assert out is not None
    assert out["stats"]["vocab_enabled"] is False
    all_queries = [q for q, _ in calls["pubmed"]]
    assert all("rogaine" not in q.lower() for q in all_queries)


# ── 2. Union + dedup: same doc via original and expansion = one candidate ───

def test_dedup_merges_provenance(monkeypatch):
    shared = _article("Minoxidil erythema RCT",
                      "minoxidil erythema randomized trial",
                      doi="10.1/xyz")

    def by_query(query, source):
        return [shared]  # same object via every query

    resolver = _FakeResolver({
        "minoxidil": _FakeResolution([
            _FakeMatch("minoxidil", ["Rogaine"], "synonym"),
        ]),
    })
    rs, calls = _search_with_stubs(monkeypatch, by_query, resolver=resolver)
    out = rs.search("minoxidil erythema")
    assert out is not None
    total = sum(len(out[k]) for k in ("pubmed", "europepmc", "clinicaltrials"))
    assert total == 1, f"expected 1 deduped candidate, got {total}"
    item = (out["pubmed"] or out["europepmc"] or out["clinicaltrials"])[0]
    prov = (item.metadata or {}).get("match_provenance", [])
    assert len(prov) >= 1


# ── 3. related_concept never becomes a retrieval expansion ─────────────────

def test_related_concept_not_expanded(monkeypatch):
    resolver = _FakeResolver({
        "minoxidil": _FakeResolution([
            _FakeMatch("alopecia", ["androgenetic alopecia"], "related_concept"),
        ]),
    })

    def by_query(query, source):
        return [_article("Minoxidil erythema", "minoxidil erythema", source)]

    rs, calls = _search_with_stubs(monkeypatch, by_query, resolver=resolver)
    out = rs.search("minoxidil erythema")
    assert out is not None
    all_queries = [q for q, _ in calls["pubmed"]]
    assert not any("androgenetic alopecia" in q.lower() for q in all_queries), \
        f"related_concept leaked into retrieval: {all_queries}"


# ── 4. Global judge pool + score threshold + max 400 ───────────────────────

def test_global_ranking_threshold_and_cap(monkeypatch):
    def by_query(query, source):
        # distinct articles per source
        return [_article(f"Minoxidil erythema {source} {i}",
                         "minoxidil erythema study", source,
                         doi=f"10.1/{source}-{i}")
                for i in range(150)]

    rs, _ = _search_with_stubs(monkeypatch, by_query, judge_score=0.9)
    out = rs.search("minoxidil erythema")
    assert out is not None
    total = sum(len(out[k]) for k in ("pubmed", "europepmc", "clinicaltrials"))
    assert total <= 400
    # every final item meets the 0.20 threshold
    for k in ("pubmed", "europepmc", "clinicaltrials"):
        for item in out[k]:
            assert float((item.metadata or {}).get("final_score", 0)) >= 0.20


def test_below_threshold_filtered(monkeypatch):
    def by_query(query, source):
        return [_article("Minoxidil erythema", "minoxidil erythema", source)]

    # The 0.20 threshold applies to final_score = judge raw + relation_bonus
    # (approved relation-aware contract). A clearly irrelevant judge score
    # (0.05) stays below 0.20 even with this article's relation bonus
    # (~0.11 → final_score 0.16 < 0.20).
    rs, _ = _search_with_stubs(monkeypatch, by_query, judge_score=0.05)
    out = rs.search("minoxidil erythema")
    assert out is not None
    total = sum(len(out[k]) for k in ("pubmed", "europepmc", "clinicaltrials"))
    assert total == 0, "final_score below 0.20 must be filtered by the threshold"


# ── 5. No per-source top-N before global ranking ────────────────────────────

def test_no_per_source_truncation_before_ranking(monkeypatch):
    def by_query(query, source):
        n = {"pubmed": 60, "europepmc": 50, "clinicaltrials": 40}[source]
        return [_article(f"Minoxidil erythema {source} {i}",
                         "minoxidil erythema", source,
                         doi=f"10.2/{source}-{i}")
                for i in range(n)]

    rs, _ = _search_with_stubs(monkeypatch, by_query, judge_score=0.9)
    out = rs.search("minoxidil erythema")
    assert out is not None
    # ClinicalTrials previously capped at 10 — now more can survive
    assert len(out["clinicaltrials"]) > 10




def test_scientific_relation_bonus_prefers_relation_specific_paper(monkeypatch):
    relation = ClinicalRelation(
        original_query="minoxidil hypertrichosis",
        agent={"term": "minoxidil", "normalized": "minoxidil",
               "role": "drug", "search_terms": ["minoxidil"]},
        event={"term": "", "normalized": ""},
        manifestation={"term": "hypertrichosis", "normalized": "hypertrichosis",
                       "role": "adverse_effect", "search_terms": ["hypertrichosis"]},
        relation_type="adverse_effect",
        scientific_query="minoxidil hypertrichosis",
    )

    def by_query(query, source):
        return [
            _article("Minoxidil overview",
                     "General discussion of minoxidil use in alopecia.",
                     source, year=2024),
            _article("Minoxidil hypertrichosis report",
                     "After minoxidil use, hypertrichosis appeared on the arms.",
                     source, year=2024),
        ]

    rs, _ = _search_with_stubs(monkeypatch, by_query, judge_score=0.8)
    monkeypatch.setattr(RelationalSearch, "_extract_relation", lambda self, q: relation)
    out = rs.search("minoxidil hypertrichosis")
    assert out is not None
    flat = [item for src in ("pubmed", "europepmc", "clinicaltrials") for item in out[src]]
    assert len(flat) >= 2
    assert flat[0].title == "Minoxidil hypertrichosis report"


def test_relation_match_boosts_adverse_effect_query(monkeypatch):
    """For an adverse-effect query the relation bonus must be wired into the
    final ranking used by search(): the relation-specific paper gets a higher
    relation_bonus than a generic drug paper and ranks first even when the
    judge gives both the same raw score."""
    relation = ClinicalRelation(
        original_query="minoxidil hypertrichosis",
        agent={"term": "minoxidil", "normalized": "minoxidil",
               "role": "drug", "search_terms": ["minoxidil"]},
        event={"term": "", "normalized": ""},
        manifestation={"term": "hypertrichosis", "normalized": "hypertrichosis",
                       "role": "adverse_effect", "search_terms": ["hypertrichosis"]},
        relation_type="adverse_effect",
        scientific_query="minoxidil hypertrichosis",
        relation_phrases=["hypertrichosis appeared"],
    )

    def by_query(query, source):
        return [
            _article("Minoxidil review",
                     "Minoxidil is widely used; safety and tolerability discussed.",
                     source, year=2024),
            _article("Minoxidil hypertrichosis report",
                     "After minoxidil use, hypertrichosis appeared on the arms.",
                     source, year=2024),
        ]

    rs, _ = _search_with_stubs(monkeypatch, by_query, judge_score=0.8)
    monkeypatch.setattr(RelationalSearch, "_extract_relation", lambda self, q: relation)
    out = rs.search("minoxidil hypertrichosis")
    assert out is not None
    flat = [item for src in ("pubmed", "europepmc", "clinicaltrials") for item in out[src]]
    assert len(flat) >= 2
    # relation bonus present in metadata and connected to the final ranking
    assert "relation_bonus" in (flat[0].metadata or {})
    assert flat[0].metadata["relation_bonus"] > 0
    assert flat[0].title == "Minoxidil hypertrichosis report"
    assert flat[0].metadata["relation_bonus"] > flat[1].metadata["relation_bonus"]

# ── 6. RWE endpoint pagination (30 per page, cached) ────────────────────────

def test_rwe_endpoint_pagination(client):
    from core.rwe.models import RWEItem, RWESearchResult
    from core.rwe.pipeline import RWEPipeline

    items = [RWEItem(
        source="openfda_faers", source_type="pharmacovigilance",
        evidence_tier="spontaneous_report", collection_method="official_api_no_key",
        source_url=f"https://example.org/{i}", title=f"finasteride report {i}",
        text="finasteride shedding", language="en",
        relevance_score=0.9, match_reason="exact_keyword",
    ) for i in range(75)]
    result = RWESearchResult(
        query="finasteride", original_query="finasteride",
        search_query="finasteride", canonical_query="finasteride",
        detected_language="en", translated_query="finasteride",
        translation_applied=False, expanded_queries=[],
        totals={"retrieved": 75, "final": 75}, items=items,
        source_status={"openfda_faers": "ok"},
    )
    with patch.object(RWEPipeline, "search", return_value=result) as m:
        r1 = client.get("/rwe/search?q=finasteride&sources=openfda_faers")
        assert r1.status_code == 200
        b1 = r1.json()
        assert len(b1["items"]) == 30
        pag = b1["pagination"]
        assert pag["page"] == 1 and pag["page_size"] == 30
        assert pag["total"] == 75 and pag["pages"] == 3
        sid = pag["search_id"]

        r2 = client.get(f"/rwe/search?q=finasteride&search_id={sid}&page=3")
        b2 = r2.json()
        assert len(b2["items"]) == 15  # 75 - 60
        # pipeline ran only once: page 2 served from cache
        assert m.call_count == 1


# ── 7. RWE: canonical preserved, original preserved ─────────────────────────

def test_rwe_plan_preserves_original_and_canonical():
    from core.rwe.query_engine import RWEQueryEngine
    eng = RWEQueryEngine()
    plan = eng.plan("finasteride shedding")
    assert plan.original_query == "finasteride shedding"
    assert plan.canonical_query
    exp_queries = [e.query for e in plan.expanded_queries]
    assert "finasteride shedding" in exp_queries
