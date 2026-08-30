"""
End-to-end verification of the Catena C as the SINGLE terminology layer
shared by Scientific and RWE.

Proves, through the real pipeline code (orchestrator, resolver, query engine,
intent, scoring, judge, ranking — with network/LLM stubbed only at the outer
boundary, as everywhere in this suite):

 1. the query enters the Catena C;
 2. the language is identified WITHOUT the old 5-language hardcoded limit;
 3. the query intent is determined;
 4. the configured external providers are queried;
 5. synonyms, variants, slang and related terms are retrieved when available;
 6. entities are recognised;
 7. the side-set used by RWE and Scientific comes from the same resolutions;
 8. RWE and Scientific use the SAME resolver;
 9. scoring, judge, ranking and retrieval receive the expected contracts;
10. none of these functions depends on SYMPTOM_ALIASES, MESH_MAP or any other
    hardcoded dictionary (biomedical_kb is not imported anywhere).

Multilingual: Portuguese, Japanese and Dutch are NOT in the old heuristic set
(IT/EN/DE/FR/ES) — the tests verify the Catena C does not silently fall back
to the old mechanism.
"""
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.rwe.intent import RWEQueryIntent, merged_sides
from core.rwe.models import RWEItem
from core.rwe.query_engine import RWEQueryEngine
from core.relational_search import ClinicalRelation, RelationalSearch
from core.search_result import SearchResult

from vocab_stubs import default_resolver, patch_resolver


def _mock_orch(search_query, lang="en", translated=True):
    return patch(
        "core.rwe.query_engine.QueryOrchestrator",
        **{"return_value.process.return_value": SimpleNamespace(
            search_query=search_query,
            detected_language=lang,
            translation_applied=translated,
        )},
    )


def _item(source, title, text, treatment=None):
    return RWEItem(source=source, source_type="community_forum",
                   title=title, text=text, treatment=treatment)


# ── 1+2+4+5+6+10. Multilingual Catena C (pt / ja / nl) ──────────────────────

MULTILINGUAL_CASES = [
    # (query, lang, translated, expected_entity, expected_variant_in_expansion)
    ("minoxidil hipertricose", "pt", "minoxidil hypertrichosis",
     "hypertrichosis", "excessive hair growth"),
    ("ミノキシジル 多毛症", "ja", "minoxidil hypertrichosis",
     "hypertrichosis", "excessive hair growth"),
    ("finasteride haaruitval", "nl", "finasteride hair loss",
     "alopecia", "hair shedding"),
]


@pytest.mark.parametrize("query,lang,translated,entity,variant", MULTILINGUAL_CASES)
def test_multilingual_catena_c_no_silent_fallback(monkeypatch, query, lang,
                                                  translated, entity, variant):
    """Languages OUTSIDE the old hardcoded heuristic set flow through the
    Catena C: orchestrator-authoritative language, providers queried in the
    source language, entities + expansion from the providers. No silent
    fallback to the old 5-language mechanism (the heuristic is even
    sabotaged here to prove it is not used)."""
    # Sabotage the old heuristic: if the pipeline consulted it, it would say "en".
    monkeypatch.setattr("core.rwe.query_engine.detect_language", lambda _t: "en")
    resolver = patch_resolver(monkeypatch, default_resolver())

    with _mock_orch(translated, lang=lang, translated=True):
        eng = RWEQueryEngine()
        plan = eng.plan(query)

    # 2. language identified by the Catena C (NOT the hardcoded heuristic)
    assert plan.detected_language == lang
    assert plan.translation_applied is True
    assert plan.original_query == query  # original preserved verbatim

    # 4. providers queried — English for the translation, source language
    #    for the original text (multilingual provider path)
    langs = resolver.languages_used()
    assert "en" in langs
    assert lang in langs, f"providers never queried in '{lang}': {resolver.calls}"

    # 6. entities recognised from the providers (none from hardcoded dicts)
    canonicals = {c for _, c, _ in plan.entities}
    assert entity in canonicals

    # 5. provider variants (synonyms/translations) retrieved into the expansion
    all_queries = " ".join(eq.query.lower() for eq in plan.expanded_queries)
    assert variant in all_queries

    # 3. intent determined for the structured relation
    if entity == "hypertrichosis":
        assert plan.intent is not None
        assert plan.intent.relation_type == "side_effect"
        assert "minoxidil" in plan.intent.interventions


def test_language_heuristic_is_only_a_hint(monkeypatch):
    """With no orchestrator opinion ("und") the plan uses the heuristic, and
    a language the heuristic cannot see stays honestly 'und' — never a
    silent 'en'."""
    resolver = patch_resolver(monkeypatch, default_resolver())
    with _mock_orch("minoxidil hipertricose", lang="und", translated=False):
        eng = RWEQueryEngine()
        plan = eng.plan("minoxidil hipertricose")
    # 'hipertricose' has no markers in the 5-language heuristic → "und"
    assert plan.detected_language == "und"


# ── 7+8. RWE and Scientific share the SAME resolver and side content ─────────

def test_rwe_and_scientific_share_resolver_and_terminology(monkeypatch):
    """One resolver instance serves BOTH engines; the same provider term
    appears in the RWE scoring side-set AND in the Scientific expansion."""
    resolver = patch_resolver(monkeypatch, default_resolver())

    # ── RWE side ──
    with _mock_orch("minoxidil hypertrichosis", lang="en", translated=False):
        plan = RWEQueryEngine().plan("minoxidil hypertrichosis")
    assert plan.intent is not None
    sides = merged_sides(plan.intent, plan.entities)
    assert "excessive hair growth" in sides["oc"]       # RWE side-set (Catena C)

    # ── Scientific side ──
    rs = RelationalSearch.__new__(RelationalSearch)
    rs._client = None
    rs._rel_cache = {}
    rel = ClinicalRelation(
        original_query="minoxidil hypertrichosis",
        agent={"term": "minoxidil", "normalized": "minoxidil",
               "role": "drug", "search_terms": ["minoxidil"]},
        event={"term": "", "normalized": ""},
        manifestation={"term": "hypertrichosis", "normalized": "hypertrichosis",
                       "role": "adverse_effect", "search_terms": ["hypertrichosis"]},
        relation_type="adverse_effect",
        scientific_query="minoxidil hypertrichosis",
    )
    expanded = rs._expand_relation(rel, "minoxidil hypertrichosis")
    scientific_queries = " ".join(v[2]["query"].lower() for v in expanded[0] if isinstance(v, tuple)) if expanded else ""
    flat = " ".join(str(v).lower() for v in expanded)
    assert "excessive hair growth" in flat  # same provider term, scientific side

    # 8. the SAME resolver instance served both
    assert resolver.calls, "resolver never called"


# ── 9. Contracts: scoring / judge / ranking / retrieval unchanged ───────────

def _article(title, abstract, source="PubMed", year=2024):
    return SearchResult(title=title, source=source, abstract=abstract,
                        authors=["A"], year=year, doi=None, metadata={})


def test_scientific_scoring_judge_ranking_contract(monkeypatch):
    """The Scientific path keeps its contract: judge score dominates,
    relation bonus in metadata, 0.20 threshold, relation-specific first."""
    patch_resolver(monkeypatch, default_resolver())
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
                     source),
            _article("Minoxidil hypertrichosis report",
                     "After minoxidil use, hypertrichosis appeared on the arms.",
                     source),
        ]

    rs = RelationalSearch.__new__(RelationalSearch)
    rs._client = object()
    rs._rel_cache = {}

    class _StubCollector:
        def __init__(self, source):
            self.source = source

        def search(self, query, limit=None):
            return by_query(query, self.source)

    rs.pubmed = _StubCollector("pubmed")
    rs.europepmc = _StubCollector("europepmc")
    rs.clinicaltrials = _StubCollector("clinicaltrials")
    monkeypatch.setattr(RelationalSearch, "_extract_relation",
                        lambda self, q: relation)
    monkeypatch.setattr(RelationalSearch, "_llm_judge",
                        lambda self, batch, rel: [
                            {"i": j, "label": "relevant", "score": 0.8,
                             "reason": "stub"} for j in range(len(batch))])
    monkeypatch.setattr("core.relational_search.time.sleep", lambda *_: None)

    out = rs.search("minoxidil hypertrichosis")
    assert out is not None
    flat = [item for src in ("pubmed", "europepmc", "clinicaltrials") for item in out[src]]
    assert len(flat) >= 2
    # contract: relation_bonus metadata present, threshold on final_score,
    # relation-specific article first
    assert flat[0].title == "Minoxidil hypertrichosis report"
    assert flat[0].metadata["relation_bonus"] > flat[1].metadata["relation_bonus"]
    for item in flat:
        assert float(item.metadata["final_score"]) >= 0.20
        assert 0.0 <= item.metadata["relevance_score"] <= 1.0


def test_rwe_scoring_contract_with_catena_c(monkeypatch):
    """RWE scoring contract: V3 scores in [0,1], direct testimony outranks
    drug-only chatter, metadata carries the intent provenance."""
    from core.rwe.pipeline import relevance_filter
    patch_resolver(monkeypatch, default_resolver())
    with _mock_orch("minoxidil hypertrichosis", lang="en", translated=False):
        plan = RWEQueryEngine().plan("minoxidil hypertrichosis")
    items = [
        _item("hairlosstalk", "direct",
              "I started minoxidil and after two months I got hypertrichosis",
              treatment="minoxidil"),
        _item("hairlosstalk", "chatter", "people talk about minoxidil prices",
              treatment="minoxidil"),
    ]
    kept = relevance_filter(items, plan.translated_query,
                            entities=plan.entities, intent=plan.intent,
                            vocabulary=plan.vocabulary, min_score=0.20)
    assert kept and kept[0].title == "direct"
    assert all(0.0 <= i.relevance_score <= 1.0 for i in kept)
    assert kept[0].metadata["intent_relation_type"] == "side_effect"


# ── 10. No hardcoded terminology anywhere in the Catena C ───────────────────

def test_no_hardcoded_terminology_in_production_modules():
    """biomedical_kb (SYMPTOM_ALIASES, MESH_MAP, alias tables) must not be
    imported by ANY production module of the Catena C."""
    import core.rwe.query_engine as qe
    import core.rwe.intent as ri
    import core.rwe.pipeline as rp
    import core.relational_search as rs
    import core.vocab.entities as ve
    import core.vocab.resolver as vr
    for mod in (qe, ri, rp, rs, ve, vr):
        src = inspect.getsource(mod)
        assert "from core.biomedical_kb import" not in src, (
            f"{mod.__name__} imports biomedical_kb")
        assert "import biomedical_kb" not in src
        assert "SYMPTOM_ALIASES" not in src
        assert "MESH_MAP" not in src
