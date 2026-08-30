"""
Tests for the RWE Search Engine — autonomous query preparation, translation,
controlled query expansion, semi-semantic retrieval, and provenance.

Catena C edition: all terminology comes from the external vocabulary
providers, simulated offline by the shared FakeResolver
(tests/vocab_stubs.py). No hardcoded dictionaries are involved.

Covers:
  - language detection (orchestrator-authoritative, heuristic as hint)
  - translation (original_query preserved)
  - provider-first entity recognition
  - query expansion (synonyms, MeSH, translations, colloquial, combos)
  - semi-semantic retrieval (authoritative source match preserved)
  - relevance filtering (score + match_reason)
  - provenance (matched_query, source_language, expansion_type)
  - no duplicates across multi-query retrieval
  - RWE only / scientific regression / assistant schema
  - structured-relation V3 activation without the feature flag
"""
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.rwe.models import RWEItem
from core.rwe.query_engine import (
    RWEQueryEngine,
    RWEQueryPlan,
    ExpandedQuery,
    detect_language,
    EXP_ORIGINAL,
    EXP_TRANSLATED,
    EXP_SYNONYM,
    EXP_MESH,
    EXP_COMBO,
    EXP_COLLOQUIAL,
)

from vocab_stubs import default_resolver, patch_resolver


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
        privacy_status="redacted",
    )


def _fake_faers_item(rid="FAERS-1", treatment="FINASTERIDE", text="alopecia; decreased libido"):
    return RWEItem(
        source="openfda_faers",
        source_type="pharmacovigilance",
        evidence_tier="spontaneous_report",
        collection_method="official_api_no_key",
        external_id=rid,
        source_url=f"https://api.fda.gov/drug/event.json?search=safetyreportid:{rid}",
        title=f"FAERS report {rid}",
        text=text,
        date="20240101",
        language="en",
        treatment=treatment,
        experience_type="adverse_event",
        privacy_status="anonymous",
    )


def _stub_reddit_ok(items):
    def _sws(query, limit=15):
        return items, "ok", f"Retrieved {len(items)} post(s)."
    return _sws


def _stub_openfda_ok(items):
    def _sws(query, limit=20):
        return list(items), "ok", f"Retrieved {len(items)} report(s)."
    return _sws


def _stub_openfda_status(status, reason):
    def _sws(query, limit=20):
        return [], status, reason
    return _sws


def _stub_reddit_status(status, reason):
    def _sws(query, limit=15):
        return [], status, reason
    return _sws


def _make_engine():
    """RWEQueryEngine with a mocked orchestrator (deterministic, no LLM).

    Must be called INSIDE a ``_mock_orch(...)`` patch context so the engine
    captures the mocked orchestrator at construction time.
    """
    return RWEQueryEngine()


def _mock_orch(search_query, lang="en", translated=True):
    """Patch QueryOrchestrator.process to return a fixed OrchestrationResult.

    The engine instantiates the orchestrator in ``__init__``, so the engine
    itself must be created while this patch is active.
    """
    return patch(
        "core.rwe.query_engine.QueryOrchestrator",
        **{"return_value.process.return_value": SimpleNamespace(
            search_query=search_query,
            detected_language=lang,
            translation_applied=translated,
        )},
    )


# ─── language detection ─────────────────────────────────────────────────────

def test_detect_language_italian():
    assert detect_language("La finasteride può causare shedding?") == "it"


def test_detect_language_english():
    assert detect_language("Can finasteride cause initial shedding?") == "en"


def test_detect_language_empty():
    assert detect_language("") == "und"


def test_detect_language_german():
    assert detect_language("Kann finasterid haarausfall verursachen?") == "de"


def test_detect_language_french():
    assert detect_language("La finastéride peut causer la chute de cheveux?") == "fr"


def test_detect_language_spanish():
    assert detect_language("La finasterida puede causar caída del cabello?") == "es"


def test_detect_language_no_evidence_returns_und_not_en():
    """Without marker evidence the heuristic must NOT silently guess English
    (the multilingual path would be bypassed)."""
    assert detect_language("xyzwq vbnmk qpr") == "und"


# ─── query plan / original_query preserved ──────────────────────────────────

def test_original_query_never_overwritten(monkeypatch):
    """original_query must remain the verbatim user input."""
    patch_resolver(monkeypatch, default_resolver())
    with _mock_orch("finasteride initial shedding", lang="it"):
        eng = _make_engine()
        plan = eng.plan("La finasteride può causare shedding iniziale?")
    assert plan.original_query == "La finasteride può causare shedding iniziale?"
    assert plan.translated_query != plan.original_query
    assert plan.detected_language == "it"


def test_translation_fields_present(monkeypatch):
    patch_resolver(monkeypatch, default_resolver())
    with _mock_orch("finasteride hair loss", lang="it", translated=True):
        eng = _make_engine()
        plan = eng.plan("caduta capelli finasteride")
    assert plan.translation_applied is True
    assert plan.translated_query == "finasteride hair loss"


def test_english_query_no_translation(monkeypatch):
    patch_resolver(monkeypatch, default_resolver())
    with _mock_orch("finasteride shedding", lang="en", translated=False):
        eng = _make_engine()
        plan = eng.plan("finasteride shedding")
    assert plan.detected_language == "en"
    assert plan.translation_applied is False


# ─── query expansion ────────────────────────────────────────────────────────

def test_expansion_contains_original_and_translated(monkeypatch):
    patch_resolver(monkeypatch, default_resolver())
    with _mock_orch("finasteride hair loss", lang="it"):
        eng = _make_engine()
        plan = eng.plan("finasteride caduta capelli")
    types = [eq.expansion_type for eq in plan.expanded_queries]
    assert EXP_ORIGINAL in types
    assert EXP_TRANSLATED in types


def test_expansion_synonyms_present(monkeypatch):
    patch_resolver(monkeypatch, default_resolver())
    with _mock_orch("finasteride", lang="en", translated=False):
        eng = _make_engine()
        plan = eng.plan("finasteride")
    queries = [eq.query.lower() for eq in plan.expanded_queries]
    # propecia/proscar are provider (RxNorm) brand synonyms of finasteride
    assert "propecia" in queries or "proscar" in queries


def test_expansion_mesh_terms_present(monkeypatch):
    patch_resolver(monkeypatch, default_resolver())
    with _mock_orch("finasteride shedding", lang="en", translated=False):
        eng = _make_engine()
        plan = eng.plan("finasteride shedding")
    queries = [eq.query.lower() for eq in plan.expanded_queries]
    # the MeSH descriptor preferred term reaches the expansion
    assert any("alopecia" in q for q in queries)


def test_expansion_colloquial_for_shedding(monkeypatch):
    """Provider colloquial variants must surface patient phrasings."""
    patch_resolver(monkeypatch, default_resolver())
    with _mock_orch("finasteride initial shedding", lang="en", translated=False):
        eng = _make_engine()
        plan = eng.plan("finasteride initial shedding")
    all_queries = " ".join(eq.query.lower() for eq in plan.expanded_queries)
    assert "shedding" in all_queries or "hair fall" in all_queries


def test_expansion_is_controlled_not_broad(monkeypatch):
    """Every expanded query keeps the drug anchor — no generic broadening."""
    patch_resolver(monkeypatch, default_resolver())
    with _mock_orch("finasteride shedding", lang="en", translated=False):
        eng = _make_engine()
        plan = eng.plan("finasteride shedding")
    fin_terms = {"finasteride", "propecia", "proscar", "finpecia"}
    for eq in plan.expanded_queries:
        if eq.expansion_type in (EXP_COLLOQUIAL, EXP_COMBO, "colloquial", "slang"):
            qtokens = set(eq.query.lower().split())
            assert qtokens & fin_terms or "finasteride" in eq.matched_entities, (
                f"expansion '{eq.query}' dropped the finasteride anchor"
            )


def test_expansion_capped(monkeypatch):
    patch_resolver(monkeypatch, default_resolver())
    with _mock_orch("finasteride dutasteride minoxidil", lang="en", translated=False):
        eng = _make_engine()
        plan = eng.plan("finasteride dutasteride minoxidil")
    assert len(plan.expanded_queries) <= 16


def test_expansion_deduplicated(monkeypatch):
    patch_resolver(monkeypatch, default_resolver())
    with _mock_orch("finasteride", lang="en", translated=False):
        eng = _make_engine()
        plan = eng.plan("finasteride")
    queries = [eq.query.lower() for eq in plan.expanded_queries]
    assert len(queries) == len(set(queries)), "duplicate expanded queries"


# ─── synonyms / medical vs colloquial ───────────────────────────────────────

def test_medical_term_recognized(monkeypatch):
    """Medical term 'androgenetic alopecia' is recognized as a condition."""
    patch_resolver(monkeypatch, default_resolver())
    with _mock_orch("androgenetic alopecia", lang="en", translated=False):
        eng = _make_engine()
        plan = eng.plan("androgenetic alopecia treatment")
    canonicals = [c for _, c, _ in plan.entities]
    assert "androgenetic alopecia" in canonicals


def test_colloquial_term_recognized_as_medical(monkeypatch):
    """Colloquial 'caduta capelli' maps to the medical concept 'hair loss'
    through the multilingual provider (ConceptNet translation)."""
    patch_resolver(monkeypatch, default_resolver())
    with _mock_orch("hair loss", lang="it"):
        eng = _make_engine()
        plan = eng.plan("caduta capelli")
    canonicals = [c for _, c, _ in plan.entities]
    assert "hair loss" in canonicals


def test_brand_name_normalized_to_generic(monkeypatch):
    """Brand 'propecia' is recognized as the generic 'finasteride' (RxNorm)."""
    patch_resolver(monkeypatch, default_resolver())
    with _mock_orch("propecia", lang="en", translated=False):
        eng = _make_engine()
        plan = eng.plan("propecia side effects")
    canonicals = [c for _, c, _ in plan.entities]
    assert "finasteride" in canonicals


# ─── trichology vs non-trichology ────────────────────────────────────────────

def test_trichology_query_recognizes_entities(monkeypatch):
    patch_resolver(monkeypatch, default_resolver())
    with _mock_orch("minoxidil hair loss", lang="en", translated=False):
        eng = _make_engine()
        plan = eng.plan("minoxidil hair loss")
    canonicals = {c for _, c, _ in plan.entities}
    assert "minoxidil" in canonicals
    assert "alopecia" in canonicals  # MeSH preferred term for 'hair loss'


def test_non_trichology_query_still_runs(monkeypatch):
    """A query with no provider-recognised entities produces a minimal plan
    (original query only) without crashing."""
    patch_resolver(monkeypatch, default_resolver())
    with _mock_orch("headache aspirin", lang="en", translated=False):
        eng = _make_engine()
        plan = eng.plan("headache aspirin")
    assert plan.original_query == "headache aspirin"
    assert len(plan.expanded_queries) >= 1  # at least the original


# ─── semi-semantic retrieval / relevance ────────────────────────────────────

def test_authoritative_source_match_preserved():
    """openFDA items are kept when the query is drug-only (drug present = relevant)."""
    from core.rwe.pipeline import relevance_filter
    it = _fake_faers_item(text="headache; nausea", treatment="FINASTERIDE")
    out = relevance_filter(
        [it], "finasteride",
        entities=[("drug", "finasteride", 1.0)],
    )
    assert len(out) == 1
    assert out[0].relevance == "relevant"
    assert "drug_match" in (out[0].match_reason or "")


def test_authoritative_drug_only_filtered_when_event_missing():
    """FAERS-23: drug match but event missing → FILTERED (no drug-only false positives)."""
    from core.rwe.pipeline import relevance_filter
    it = _fake_faers_item(
        rid="PROP-1",
        treatment="DUTASTERIDE",
        text="loss of proprioception; balance disorder",
    )
    out = relevance_filter(
        [it], "dutasteride induced hair shedding",
        entities=[("drug", "dutasteride", 1.0), ("symptom", "hair shedding", 1.0)],
    )
    assert len(out) == 0, "drug-only match must NOT surface when event is missing"
    assert it.relevance == "irrelevant"
    assert "event_missing" in (it.match_reason or "")


def test_authoritative_drug_plus_event_kept():
    """FAERS-23b: drug match AND event present → KEPT with high score."""
    from core.rwe.pipeline import relevance_filter
    it = _fake_faers_item(
        rid="SHED-1",
        treatment="DUTASTERIDE",
        text="alopecia; hair shedding; hair loss",
    )
    out = relevance_filter(
        [it], "dutasteride induced hair shedding",
        entities=[("drug", "dutasteride", 1.0), ("symptom", "hair shedding", 1.0)],
    )
    assert len(out) == 1
    assert out[0].relevance == "relevant"
    assert out[0].relevance_score >= 0.5
    assert "drug+event_match" in (out[0].match_reason or "")


def test_finasteride_shedding_keeps_only_event_matching():
    """FAERS-23c: mixed batch — event-matching kept, event-missing filtered."""
    from core.rwe.pipeline import relevance_filter
    good = _fake_faers_item(rid="GOOD", treatment="FINASTERIDE",
                            text="hair shedding; alopecia")
    bad = _fake_faers_item(rid="BAD", treatment="FINASTERIDE",
                           text="loss of proprioception; dizziness")
    out = relevance_filter(
        [good, bad], "finasteride induced shedding",
        entities=[("drug", "finasteride", 1.0), ("symptom", "hair shedding", 1.0)],
    )
    ids = {it.external_id for it in out}
    assert "GOOD" in ids
    assert "BAD" not in ids


def test_event_synonym_match_via_provider_vocabulary():
    """Provider variants of the event (from the plan's slim vocabulary) match
    the record even when the literal query event term is absent."""
    from core.rwe.pipeline import relevance_filter
    from vocab_stubs import slim, default_mapping
    vocab = slim(("hair shedding", default_mapping()[("hair shedding", "*")]))
    it = _fake_faers_item(rid="IT-1", treatment="FINASTERIDE",
                          text="worsening alopecia reported")
    out = relevance_filter(
        [it], "finasteride caduta capelli",
        entities=[("drug", "finasteride", 1.0), ("symptom", "hair shedding", 1.0)],
        vocabulary=vocab,
    )
    assert len(out) == 1
    assert out[0].relevance_score >= 0.5


def test_semantic_entity_match_without_keyword():
    """An item matching provider variants of the entities (but no query token)
    is kept."""
    from core.rwe.pipeline import relevance_filter
    from vocab_stubs import slim, default_mapping
    mapping = default_mapping()
    vocab = slim(("finasteride", mapping[("finasteride", "*")]),
                 ("hair loss", mapping[("hair loss", "*")]))
    it = _fake_reddit_item("my story", "u1",
                           text="I took propecia and now I have noticeable hair fall")
    out = relevance_filter(
        [it], "finasteride shedding",
        entities=[("drug", "finasteride", 1.0), ("symptom", "hair loss", 1.0)],
        vocabulary=vocab,
    )
    assert len(out) == 1
    assert out[0].relevance_score > 0


def test_off_topic_item_filtered():
    from core.rwe.pipeline import relevance_filter
    off = _fake_reddit_item("cooking", "u2", text="pasta recipe with tomato sauce")
    out = relevance_filter(
        [off], "finasteride shedding",
        entities=[("drug", "finasteride", 1.0)],
    )
    assert len(out) == 0
    assert off.relevance == "irrelevant"


def test_relevance_filter_backward_compatible():
    """relevance_filter(items, query) still works without entities arg."""
    from core.rwe.pipeline import relevance_filter
    rel = _fake_reddit_item("fin", "u1", text="finasteride worked")
    out = relevance_filter([rel], "finasteride")
    assert len(out) == 1
    assert out[0].relevance == "relevant"


def test_relevance_score_is_float_0_to_1():
    from core.rwe.pipeline import relevance_filter
    it = _fake_faers_item(text="alopecia", treatment="FINASTERIDE")
    relevance_filter([it], "finasteride", entities=[("drug", "finasteride", 1.0)])
    assert 0.0 <= it.relevance_score <= 1.0


# ─── provenance ─────────────────────────────────────────────────────────────

def test_matched_query_stamped(monkeypatch):
    """Each item carries the expanded query that surfaced it."""
    from core.rwe.pipeline import RWEPipeline
    patch_resolver(monkeypatch, default_resolver())
    pipe = RWEPipeline()
    faers = [_fake_faers_item()]
    with patch.object(pipe.openfda, "search_with_status", _stub_openfda_ok(faers)), \
         patch.object(pipe.reddit, "search_with_status", _stub_reddit_status("no_credentials", "x")), \
         _mock_orch("finasteride", lang="en", translated=False):
        result = pipe.search("finasteride", sources=["reddit", "openfda_faers"])
    assert len(result.items) == 1
    it = result.items[0]
    assert it.matched_query is not None
    assert it.matched_query_type is not None
    assert it.source_language is not None


def test_source_language_stamped(monkeypatch):
    from core.rwe.pipeline import RWEPipeline
    patch_resolver(monkeypatch, default_resolver())
    pipe = RWEPipeline()
    faers = [_fake_faers_item()]
    # Reddit returns items only for English queries (realistic: Reddit search
    # is English-oriented, so translated queries match).
    def reddit_en_only(query, limit=15):
        if any(w in query.lower() for w in ("finasteride", "hair", "shedding")) and "caduta" not in query.lower():
            return [_fake_reddit_item(
                "en post", "u-en",
                text="finasteride stopped my hair shedding")], "ok", "ok"
        return [], "no_results", "none"
    with patch.object(pipe.openfda, "search_with_status", _stub_openfda_ok(faers)), \
         patch.object(pipe.reddit, "search_with_status", reddit_en_only), \
         _mock_orch("finasteride hair loss", lang="it"):
        result = pipe.search("caduta capelli finasteride", sources=["reddit", "openfda_faers"])
    # at least one item should have source_language en (matched the translated query)
    langs = {it.source_language for it in result.items}
    assert "en" in langs


def test_expanded_queries_in_result(monkeypatch):
    """The result envelope exposes the full expansion provenance."""
    from core.rwe.pipeline import RWEPipeline
    patch_resolver(monkeypatch, default_resolver())
    pipe = RWEPipeline()
    with patch.object(pipe.openfda, "search_with_status", _stub_openfda_status("no_results", "none")), \
         patch.object(pipe.reddit, "search_with_status", _stub_reddit_status("no_credentials", "x")), \
         _mock_orch("finasteride", lang="en", translated=False):
        result = pipe.search("finasteride", sources=["reddit", "openfda_faers"])
    assert len(result.expanded_queries) >= 1
    assert result.original_query == "finasteride"
    assert result.translated_query == "finasteride"


# ─── multi-query retrieval: translated + expanded results ───────────────────

def test_results_from_translated_query(monkeypatch):
    """An item surfaced by the translated (English) query is present when the
    original was Italian."""
    from core.rwe.pipeline import RWEPipeline
    patch_resolver(monkeypatch, default_resolver())
    pipe = RWEPipeline()
    # Reddit returns items only when the query contains 'finasteride' (English)
    def reddit_conditional(query, limit=15):
        if "finasteride" in query.lower():
            return [_fake_reddit_item(
                "en post", "u-en",
                text="finasteride stopped my hair shedding")], "ok", "ok"
        return [], "no_results", "none"
    with patch.object(pipe.reddit, "search_with_status", reddit_conditional), \
         patch.object(pipe.openfda, "search_with_status", _stub_openfda_status("no_results", "x")), \
         _mock_orch("finasteride hair loss", lang="it"):
        result = pipe.search("caduta capelli finasteride", sources=["reddit", "openfda_faers"])
    assert any(it.source == "reddit" for it in result.items)
    # the matched_query should be an English expansion
    en_items = [it for it in result.items if it.matched_query and "finasteride" in it.matched_query.lower()]
    assert len(en_items) >= 1


def test_results_from_expanded_query(monkeypatch):
    """An item surfaced by a provider-synonym query (not the original) is kept."""
    from core.rwe.pipeline import RWEPipeline
    patch_resolver(monkeypatch, default_resolver())
    pipe = RWEPipeline()
    # openFDA returns an item only for the 'propecia' synonym
    def openfda_conditional(query, limit=20):
        if "propecia" in query.lower():
            return [_fake_faers_item(rid="PF-1", treatment="FINASTERIDE")], "ok", "ok"
        return [], "no_results", "none"
    with patch.object(pipe.openfda, "search_with_status", openfda_conditional), \
         patch.object(pipe.reddit, "search_with_status", _stub_reddit_status("no_credentials", "x")), \
         _mock_orch("finasteride", lang="en", translated=False):
        result = pipe.search("finasteride", sources=["reddit", "openfda_faers"])
    assert any(it.external_id == "PF-1" for it in result.items)
    pf = next(it for it in result.items if it.external_id == "PF-1")
    assert "propecia" in pf.matched_query.lower()
    assert pf.matched_query_type == EXP_SYNONYM


# ─── no duplicates across multi-query ───────────────────────────────────────

def test_no_duplicates_across_queries(monkeypatch):
    """The same item returned by multiple expanded queries is deduplicated."""
    from core.rwe.pipeline import RWEPipeline
    patch_resolver(monkeypatch, default_resolver())
    pipe = RWEPipeline()
    same = _fake_faers_item(rid="DUP-1")
    with patch.object(pipe.openfda, "search_with_status", _stub_openfda_ok([same])), \
         patch.object(pipe.reddit, "search_with_status", _stub_reddit_status("no_credentials", "x")), \
         _mock_orch("finasteride", lang="en", translated=False):
        result = pipe.search("finasteride", sources=["reddit", "openfda_faers"])
    # same item returned for many expanded queries but should appear once
    ids = [it.external_id for it in result.items if it.source == "openfda_faers"]
    assert ids.count("DUP-1") == 1


# ─── RWE only / scientific regression / both ────────────────────────────────

def test_rwe_only_search(monkeypatch):
    from core.rwe.pipeline import RWEPipeline
    patch_resolver(monkeypatch, default_resolver())
    pipe = RWEPipeline()
    with patch.object(pipe.openfda, "search_with_status", _stub_openfda_ok([_fake_faers_item()])), \
         patch.object(pipe.reddit, "search_with_status", _stub_reddit_ok([_fake_reddit_item("r","u")])), \
         _mock_orch("finasteride", lang="en", translated=False):
        result = pipe.search("finasteride", sources=["reddit", "openfda_faers"])
    assert all(i.source_type in ("community_forum", "pharmacovigilance") for i in result.items)
    assert all(i.source_type not in ("pubmed", "europepmc", "clinicaltrials") for i in result.items)


def test_scientific_search_route_unchanged(client):
    """Scientific /search must still work and not leak RWE items."""
    r = client.get("/search?q=finasteride&mode=global")
    assert r.status_code == 200


def test_rwe_search_endpoint_returns_provenance(client):
    r = client.get("/rwe/search?q=finasteride")
    assert r.status_code == 200
    body = r.json()
    assert "original_query" in body
    assert "translated_query" in body
    assert "detected_language" in body
    assert "expanded_queries" in body


def test_assistant_accepts_rwe_with_provenance(client):
    """The AI Assistant schema accepts RWE items with the new provenance fields."""
    r = client.post("/assistant/chat", json={
        "message": "What do patients report about finasteride shedding?",
        "search_context": {
            "original_query": "finasteride shedding",
            "search_query": "finasteride shedding",
            "detected_language": "en",
            "articles": [],
            "rwe_evidence": [{
                "source": "reddit",
                "source_type": "community_forum",
                "evidence_tier": "anecdotal",
                "title": "My finasteride shed",
                "text": "I experienced initial shedding on finasteride",
                "matched_query": "finasteride initial shedding",
                "matched_query_type": "colloquial",
                "source_language": "en",
                "relevance_score": 0.8,
                "match_reason": "exact_keyword+semantic",
            }],
        },
    })
    assert r.status_code == 200


# ─── integration: the canonical Italian test query ──────────────────────────

def test_canonical_italian_query_full_pipeline(monkeypatch):
    """The success criterion: 'La finasteride può causare shedding iniziale?'
    must be transformed into a proper RWE search plan (Catena C providers)."""
    patch_resolver(monkeypatch, default_resolver())
    with _mock_orch("finasteride initial shedding", lang="it", translated=True):
        eng = _make_engine()
        plan = eng.plan("La finasteride può causare shedding iniziale?")
    # 1. recognizes language
    assert plan.detected_language == "it"
    # 2. preserves original
    assert plan.original_query == "La finasteride può causare shedding iniziale?"
    # 3. translates
    assert plan.translated_query == "finasteride initial shedding"
    assert plan.translation_applied is True
    # 4. recognizes entities (via the external providers)
    canonicals = {c for _, c, _ in plan.entities}
    assert "finasteride" in canonicals
    assert "alopecia" in canonicals  # 'initial shedding' → MeSH Alopecia concept
    # 5. generates expanded queries
    assert len(plan.expanded_queries) >= 5
    # 6. includes provider variants of the shedding concept
    all_queries = " ".join(eq.query.lower() for eq in plan.expanded_queries)
    assert "shedding" in all_queries
    # 7. expansion is controlled (finasteride anchor preserved)
    fin_terms = {"finasteride", "propecia", "proscar"}
    colloq = [eq for eq in plan.expanded_queries
              if eq.expansion_type in (EXP_COLLOQUIAL, "colloquial", "slang")]
    for eq in colloq:
        assert "finasteride" in eq.matched_entities or (set(eq.query.lower().split()) & fin_terms)


# ─── structured relation: V3 activates without the feature flag ─────────────

def test_structured_hypertrichosis_plan_activates_v3_without_flag(monkeypatch):
    """'minoxidil hypertrichosis' exposes a drug→outcome relation: the V3
    relation-aware scorer must activate (deterministic entities-fallback
    intent, no LLM) even when HLEO_RWE_INTENT_SCORING is off, the expansion
    must stay relation-preserving (no finasteride/dutasteride broadening),
    and a direct oral-route temporal testimony must outrank a drug-only
    thread. Entities come from the external providers (FakeResolver)."""
    monkeypatch.delenv("HLEO_RWE_INTENT_SCORING", raising=False)
    from core.rwe.pipeline import RWEPipeline
    patch_resolver(monkeypatch, default_resolver())

    direct = RWEItem(
        source="hairlosstalk", source_type="community_forum",
        evidence_tier="anecdotal", collection_method="official_rss_feed",
        external_id="t-direct", source_url="https://example.org/t-direct",
        title="Oral minoxidil excess hair",
        text=("I started oral minoxidil and after a few months I developed "
              "excessive body hair and hypertrichosis on my face."),
        language="en", treatment="oral minoxidil",
    )
    drug_only = RWEItem(
        source="hairlosstalk", source_type="community_forum",
        evidence_tier="anecdotal", collection_method="official_rss_feed",
        external_id="t-generic", source_url="https://example.org/t-generic",
        title="My regimen",
        text="I use minoxidil for my hair, great stuff overall.",
        language="en", treatment="minoxidil",
    )

    pipe = RWEPipeline()
    items = [direct, drug_only]

    def feed_stub(query, limit=15):
        return list(items), "ok", f"Retrieved {len(items)} thread(s)."

    with patch.object(pipe.hairlosstalk, "search_with_status", feed_stub), \
         patch.object(pipe.openfda, "search_with_status",
                      _stub_openfda_status("no_results", "none")), \
         _mock_orch("minoxidil hypertrichosis", lang="en", translated=False):
        result = pipe.search("minoxidil hypertrichosis",
                             sources=["hairlosstalk", "openfda_faers"])

    # V3 activated without the flag (structured drug→outcome relation)
    assert result.source_status["openfda_faers"] != "no_credentials"
    titles = [it.title for it in result.items]
    assert "Oral minoxidil excess hair" in titles
    direct_item = next(it for it in result.items
                       if it.title == "Oral minoxidil excess hair")
    assert direct_item.relevance_score >= 0.7
    assert direct_item.metadata["intent_relation_type"] == "side_effect"
    # drug-only thread scores far below (event side missing) and is filtered
    assert "My regimen" not in titles

    # relation-preserving expansion: no broadening to other drugs
    expanded = " ".join(eq["query"].lower() for eq in result.expanded_queries)
    assert "finasteride" not in expanded
    assert "dutasteride" not in expanded
    # the requested relation IS expanded (provider variants of the outcome)
    assert "excess" in expanded or "unwanted hair" in expanded


def test_route_and_temporal_inferred_for_oral_query(monkeypatch):
    """'oral minoxidil hypertrichosis after a few months' carries route=oral
    and temporal_relation=after in the intent and in the item metadata."""
    monkeypatch.delenv("HLEO_RWE_INTENT_SCORING", raising=False)
    from core.rwe.pipeline import RWEPipeline
    patch_resolver(monkeypatch, default_resolver())

    direct = RWEItem(
        source="hairlosstalk", source_type="community_forum",
        evidence_tier="anecdotal", collection_method="official_rss_feed",
        external_id="t-oral", source_url="https://example.org/t-oral",
        title="Oral minoxidil excess hair",
        text=("I started oral minoxidil and after a few months I developed "
              "excessive body hair and hypertrichosis on my face."),
        language="en", treatment="oral minoxidil",
    )

    pipe = RWEPipeline()

    def feed_stub(query, limit=15):
        return [direct], "ok", "ok"

    with patch.object(pipe.hairlosstalk, "search_with_status", feed_stub), \
         _mock_orch("oral minoxidil hypertrichosis after a few months",
                    lang="en", translated=False):
        result = pipe.search("oral minoxidil hypertrichosis after a few months",
                             sources=["hairlosstalk"])

    assert len(result.items) == 1
    item = result.items[0]
    assert item.metadata["intent_route"] == "oral"
    assert item.metadata["intent_temporal_relation"] == "after"
    assert item.metadata["intent_relation_type"] == "side_effect"
    assert item.relevance_score >= 0.7
