"""Scientific Chain tests: ClinicalRelation-only query generation,
union+dedup with provenance, global judge pool, semantic tiers and max 400.

All tests are offline: collectors and LLM are stubbed.
"""
import inspect
import os
from unittest.mock import patch

import pytest

from core.relational_search import ClinicalRelation, RelationalSearch
from aggregator import HLEOAggregator

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
               "role": "drug"},
        event={"term": "", "normalized": ""},
        manifestation={"term": "erythema", "normalized": "erythema",
                       "role": "adverse_event"},
        relation_type="adverse_effect",
        scientific_query="minoxidil erythema",
    )


def test_recovery_query_keeps_agent_event_and_recovery_terms():
    relation = ClinicalRelation(
        original_query="caduta indotta da dutasteride, poi si recupera?",
        agent={"normalized": "dutasteride", "role": "drug"},
        event={"normalized": "initial hair shedding"},
        manifestation={"normalized": "hair loss", "role": "condition"},
        temporal="subsequent recovery",
        relation_type="adverse_effect",
        scientific_query="dutasteride AND initial hair shedding AND subsequent hair regrowth",
    )
    query = RelationalSearch._retrieval_query(relation)
    assert query == relation.scientific_query
    assert "dutasteride" in query
    assert "hair shedding" in query
    assert "hair regrowth" in query



def _search_with_stubs(monkeypatch, articles_by_query, judge_score=0.9):
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
                            {"i": j, "tier": "A", "label": "relevant",
                             "score": judge_score, "reason": "stub"}
                            for j in range(len(batch))])
    monkeypatch.setattr("core.relational_search.time.sleep", lambda *_: None)

    return rs, calls



@pytest.mark.parametrize(
    ("decision", "expected_reason"),
    [
        ({"tier": None, "label": "partial", "score": 0.5, "reason": "missing"},
         "judge tier is missing or invalid"),
        ({"tier": "B", "label": "partial", "score": 0.5},
         "judge reason is missing"),
        ({"tier": "B", "label": "partial", "score": 0.5, "reason": ""},
         "judge reason is missing"),
        (None, "judge result is not an object"),
    ],
)
def test_invalid_judge_decisions_are_not_semantic_tiers(decision, expected_reason):
    normalized = RelationalSearch._validate_judgement(decision)
    assert normalized["valid"] is False
    assert normalized["tier"] is None
    assert normalized["reason"] == expected_reason


@pytest.mark.parametrize(
    ("tier", "label"),
    [("A", "relevant"), ("B", "partial"),
     ("C", "contextual"), ("D", "not_relevant")],
)
def test_valid_judge_decisions_preserve_all_semantic_tiers(tier, label):
    normalized = RelationalSearch._validate_judgement({
        "tier": tier, "label": label, "score": 0.5, "reason": "grounded",
    })
    assert normalized == {
        "valid": True,
        "tier": tier,
        "label": label,
        "score": 0.5,
        "reason": "grounded",
    }


def test_incomplete_judge_is_not_counted_as_b_or_kept(monkeypatch):
    rs, _calls = _search_with_stubs(
        monkeypatch,
        lambda query, source: [_article("incomplete judge result", query, source)],
    )
    monkeypatch.setattr(
        RelationalSearch, "_llm_judge",
        lambda self, batch, rel: [{"i": 0, "label": "partial", "score": 0.5}],
    )
    result = rs.search("incomplete judge")
    assert result["stats"]["judge_invalid"] == 1
    assert result["stats"]["judge_tiers"]["B"] == 0
    assert result["stats"]["final_count"] == 0
    assert all(not result[source] for source in ("pubmed", "europepmc", "clinicaltrials"))


def test_malformed_judge_response_is_invalid(monkeypatch):
    rs, _calls = _search_with_stubs(
        monkeypatch,
        lambda query, source: [_article("malformed judge result", query, source)],
    )
    monkeypatch.setattr(RelationalSearch, "_llm_judge", lambda self, batch, rel: "not-json")
    result = rs.search("malformed judge")
    assert result["stats"]["judge_invalid"] == 1
    assert result["stats"]["judge_tiers"] == {"A": 0, "B": 0, "C": 0, "D": 0}
    assert result["stats"]["final_count"] == 0




# ── 1. Scientific Chain has no vocabulary-provider path ─────────────────────

def test_scientific_never_calls_vocabulary_provider(monkeypatch):
    def fail_if_called():
        raise AssertionError("Scientific Chain must not build a vocabulary resolver")

    monkeypatch.setattr("core.vocab.resolver.build_resolver_from_env", fail_if_called)
    rs = RelationalSearch.__new__(RelationalSearch)
    rs._client = object()
    rs._rel_cache = {}
    relation = _relation("gonfiore oculare con utilizzo di minoxidil topico")
    monkeypatch.setattr(RelationalSearch, "_extract_relation", lambda self, query: relation)
    rs.pubmed = type("Collector", (), {"search": lambda self, query, limit=None: []})()
    rs.europepmc = type("Collector", (), {"search": lambda self, query, limit=None: []})()
    rs.clinicaltrials = type("Collector", (), {"search": lambda self, query, limit=None: []})()
    monkeypatch.setattr(RelationalSearch, "_llm_judge", lambda self, batch, rel: [])
    result = rs.search(relation.original_query)
    assert result is not None
    assert result["relation"].expanded_queries[0]["query_origin"] != "vocabulary"



def test_scientific_chain_has_no_terminology_expansion_symbols():
    source = inspect.getsource(RelationalSearch)
    assert "build_resolver_from_env" not in source
    assert "core.vocab" not in source
    assert "search_terms" not in source
    assert "_CAUSAL" not in source
    assert "relation_cues" not in source


def test_semantic_query_is_sent_unchanged_to_all_collectors(monkeypatch):
    rs, calls = _search_with_stubs(
        monkeypatch,
        lambda query, source: [_article("semantic result", query, source)],
    )
    relation = ClinicalRelation(
        original_query="gonfiore oculare con utilizzo di minoxidil topico",
        agent={"normalized": "minoxidil", "role": "drug"},
        event={"normalized": "ocular swelling"},
        manifestation={"normalized": "swelling", "role": "adverse_effect",
                       "site": {"normalized": "ocular"}},
        formulation={"normalized": "topical"},
        relation_type="adverse_effect",
        scientific_query="minoxidil topical AND ocular swelling",
    )
    monkeypatch.setattr(RelationalSearch, "_extract_relation", lambda self, query: relation)
    result = rs.search(relation.original_query)
    assert result is not None
    queries = [query for source_calls in calls.values() for query, _ in source_calls]
    assert queries == [relation.scientific_query] * 3
    assert all(provenance["query_origin"] != "vocabulary"
               for provenance in relation.expanded_queries)


def test_relation_extraction_discards_terminology_lists(monkeypatch):
    rs = RelationalSearch.__new__(RelationalSearch)
    rs._rel_cache = {}
    rs._rel_cache_maxsize = 8
    monkeypatch.setattr(rs, "_llm_json", lambda *args, **kwargs: {
        "query_original": "gonfiore oculare con utilizzo di minoxidil topico",
        "agent": {"term": "minoxidil", "normalized": "minoxidil", "role": "drug",
                   "search_terms": ["Rogaine"]},
        "event": {"term": "ocular swelling", "normalized": "ocular swelling"},
        "manifestation": {"term": "swelling", "normalized": "swelling",
                           "role": "adverse_effect", "search_terms": ["Megamitochondria"],
                           "site": {"term": "oculare", "normalized": "ocular",
                                    "search_terms": ["periocular"]}},
        "formulation": {"term": "topico", "normalized": "topical",
                         "search_terms": ["cream"]},
        "relation_type": "adverse_effect",
        "scientific_query": "minoxidil topical AND ocular swelling",
        "relation_phrases": [],
    })
    relation = rs._extract_relation("gonfiore oculare con utilizzo di minoxidil topico")
    assert "search_terms" not in relation.agent
    assert "search_terms" not in relation.manifestation
    assert "search_terms" not in relation.manifestation["site"]
    assert "search_terms" not in relation.formulation


def test_recovery_query_has_no_internal_expansion():
    relation = ClinicalRelation(
        original_query="caduta indotta da dutasteride, poi si recupera?",
        agent={"normalized": "dutasteride", "role": "drug"},
        event={"normalized": "hair shedding"},
        manifestation={"normalized": "hair loss", "role": "adverse_effect"},
        temporal="subsequent recovery",
        relation_type="adverse_effect",
        scientific_query="dutasteride AND hair loss induced by dutasteride, then recovery",
    )
    assert RelationalSearch._retrieval_query(relation) == relation.scientific_query


# ── 2. Union + dedup: same doc via one semantic query = one candidate ───────

def test_dedup_merges_provenance(monkeypatch):
    shared = _article("Minoxidil erythema RCT",
                      "minoxidil erythema randomized trial",
                      doi="10.1/xyz")

    def by_query(query, source):
        return [shared]  # same object via every query

    rs, calls = _search_with_stubs(monkeypatch, by_query)
    out = rs.search("minoxidil erythema")
    assert out is not None
    total = sum(len(out[k]) for k in ("pubmed", "europepmc", "clinicaltrials"))
    assert total == 1, f"expected 1 deduped candidate, got {total}"
    item = (out["pubmed"] or out["europepmc"] or out["clinicaltrials"])[0]
    prov = (item.metadata or {}).get("match_provenance", [])
    assert len(prov) >= 1


# ── 3. ClinicalRelation produces exactly one retrieval query ─────────────────

def test_clinical_relation_produces_one_query(monkeypatch):
    def by_query(query, source):
        return [_article("Minoxidil erythema", query, source)]

    rs, calls = _search_with_stubs(monkeypatch, by_query)
    out = rs.search("minoxidil erythema")
    assert out is not None
    assert len(calls["pubmed"]) == 1
    assert len(calls["europepmc"]) == 1
    assert len(calls["clinicaltrials"]) == 1


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

    # Numeric confidence does not override the semantic tier. A Judge result
    # labelled relevant is retained even when its explanatory score is low.
    rs, _ = _search_with_stubs(monkeypatch, by_query, judge_score=0.05)
    out = rs.search("minoxidil erythema")
    assert out is not None
    total = sum(len(out[k]) for k in ("pubmed", "europepmc", "clinicaltrials"))
    assert total == 1
    item = out["pubmed"][0]
    assert item.metadata["semantic_tier"] == "A"
    assert item.metadata["final_score"] == 0.05


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
               "role": "drug"},
        event={"term": "", "normalized": ""},
        manifestation={"term": "hypertrichosis", "normalized": "hypertrichosis",
                       "role": "adverse_effect"},
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
               "role": "drug"},
        event={"term": "", "normalized": ""},
        manifestation={"term": "hypertrichosis", "normalized": "hypertrichosis",
                       "role": "adverse_effect"},
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
    monkeypatch.setattr(RelationalSearch, "_llm_judge", lambda self, batch, rel: [
        {"i": i,
         "tier": "A" if "hypertrichosis" in article.title.lower() else "B",
         "label": "relevant" if "hypertrichosis" in article.title.lower() else "partial",
         "score": 0.8, "reason": "stub"}
        for i, article in enumerate(batch)
    ])
    out = rs.search("minoxidil hypertrichosis")
    assert out is not None
    flat = [item for src in ("pubmed", "europepmc", "clinicaltrials") for item in out[src]]
    assert len(flat) >= 2
    assert flat[0].title == "Minoxidil hypertrichosis report"
    assert flat[0].metadata["semantic_tier"] == "A"
    assert flat[1].metadata["semantic_tier"] == "B"



def test_judge_prompt_contains_complete_relation_context(monkeypatch):
    relation = ClinicalRelation(
        original_query="ricrescita sulle tempie con dutasteride",
        agent={"term": "dutasteride", "normalized": "dutasteride", "role": "drug"},
        event={"term": "ricrescita", "normalized": "hair regrowth"},
        anatomical_site={"term": "tempie", "normalized": "temporal region", "role": "site"},
        manifestation={"term": "alopecia androgenetica", "normalized": "androgenetic alopecia", "role": "condition"},
        temporal="during treatment",
        relation_type="efficacy",
        scientific_query="dutasteride AND hair regrowth",
        canonical_query="dutasteride AND hair regrowth",
        relation_phrases=["dutasteride-induced hair regrowth"],
    )
    rs = RelationalSearch.__new__(RelationalSearch)
    rs._client = object()
    captured = {}

    def fake_llm_json(prompt, max_tokens=900, response_format=None):
        captured["prompt"] = prompt
        captured["response_format"] = response_format
        return {"results": [{"i": 0, "tier": "A", "label": "relevant",
                              "score": 1.0, "reason": "direct"}]}

    monkeypatch.setattr(rs, "_llm_json", fake_llm_json)
    rs._llm_judge([_article("Dutasteride temporal hairline regrowth", "direct study")], relation)
    prompt = captured["prompt"]
    for value in (
        relation.original_query, "dutasteride", "hair regrowth", "androgenetic alopecia",
        "temporal region", "during treatment", "efficacy", "dutasteride AND hair regrowth",
        "dutasteride-induced hair regrowth", "Dutasteride temporal hairline regrowth",
    ):
        assert value in prompt



def test_judge_prompt_includes_complete_abstract(monkeypatch):
    relation = _relation("ricrescita sulle tempie con dutasteride")
    rs = RelationalSearch.__new__(RelationalSearch)
    captured = {}
    abstract = "context " * 120 + "demonstrated successful DHT inhibition in vitro and in vivo"

    def fake_llm_json(prompt, max_tokens=900, response_format=None):
        captured["prompt"] = prompt
        return {"results": []}

    monkeypatch.setattr(rs, "_llm_json", fake_llm_json)
    rs._llm_judge([_article("Microneedle hair regeneration", abstract)], relation)
    assert "demonstrated successful DHT inhibition in vitro and in vivo" in captured["prompt"]


def test_dedup_normalizes_doi_and_bridges_missing_doi():
    abstract = "The same complete abstract for the publication."
    with_doi = _article(
        "Bifunctional patch for hair regeneration.", abstract,
        source="europepmc", doi="https://doi.org/10.1000/ABC."
    )
    without_doi = _article(
        "Bifunctional patch for hair regeneration", abstract,
        source="europepmc", doi=None
    )
    cleaned, stats = HLEOAggregator().deduplicate_across_sources({
        "pubmed": [], "europepmc": [without_doi, with_doi],
        "clinicaltrials": [], "reddit": [],
    })
    assert stats["unique"] == 1
    assert len(cleaned["europepmc"]) == 1
    assert cleaned["europepmc"][0].doi == "https://doi.org/10.1000/ABC."
    assert HLEOAggregator.create_key(with_doi) == "doi:10.1000/abc"


def test_dedup_keeps_distinct_articles_with_similar_titles():
    first = _article(
        "Dutasteride treatment of androgenetic alopecia",
        "An oral treatment trial measured hair density after six months.",
        source="europepmc", year=2025,
    )
    second = _article(
        "Dutasteride treatment in androgenetic alopecia",
        "A topical formulation study measured follicular delivery only.",
        source="europepmc", year=2025,
    )
    cleaned, stats = HLEOAggregator().deduplicate_across_sources({
        "pubmed": [], "europepmc": [first, second],
        "clinicaltrials": [], "reddit": [],
    })
    assert stats["unique"] == 2
    assert len(cleaned["europepmc"]) == 2


def test_semantic_judge_tier_ranks_a_before_c(monkeypatch):
    relation = ClinicalRelation(
        original_query="ricrescita sulle tempie con dutasteride",
        agent={"term": "dutasteride", "normalized": "dutasteride", "role": "drug"},
        event={"term": "ricrescita", "normalized": "hair regrowth"},
        anatomical_site={"term": "tempie", "normalized": "temporal region", "role": "site"},
        manifestation={"term": "alopecia androgenetica", "normalized": "androgenetic alopecia",
                       "role": "condition"},
        relation_type="efficacy",
        scientific_query="dutasteride AND hair regrowth",
        canonical_query="dutasteride AND hair regrowth",
    )

    def by_query(query, source):
        return [
            _article(
                "Dutasteride and temporal hairline regrowth in androgenetic alopecia",
                "A clinical study evaluates dutasteride for temporal hairline regrowth in androgenetic alopecia.",
                source, doi="10.1/direct",
            ),
            _article(
                "Review of androgenetic alopecia treatments",
                "A broad review mentions dutasteride among several androgenetic alopecia treatments.",
                source, doi="10.1/generic",
            ),
        ]

    rs, _ = _search_with_stubs(monkeypatch, by_query)
    monkeypatch.setattr(RelationalSearch, "_extract_relation", lambda self, q: relation)

    def judge(self, batch, rel):
        return [
            {
                "i": i,
                "tier": "A" if "temporal" in article.title.lower() else "C",
                "label": "relevant" if "temporal" in article.title.lower() else "contextual",
                "score": 1.0 if "temporal" in article.title.lower() else 0.15,
                "reason": "direct relation" if "temporal" in article.title.lower() else "generic context",
            }
            for i, article in enumerate(batch)
        ]

    monkeypatch.setattr(RelationalSearch, "_llm_judge", judge)
    out = rs.search(relation.original_query)
    flat = [item for src in ("pubmed", "europepmc", "clinicaltrials") for item in out[src]]
    assert [item.title for item in flat] == [
        "Dutasteride and temporal hairline regrowth in androgenetic alopecia",
        "Review of androgenetic alopecia treatments",
    ]
    assert [item.metadata["semantic_tier"] for item in flat] == ["A", "C"]
    assert flat[0].metadata["relevance_reason"] == "direct relation"
    assert flat[1].metadata["relevance_reason"] == "generic context"

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
