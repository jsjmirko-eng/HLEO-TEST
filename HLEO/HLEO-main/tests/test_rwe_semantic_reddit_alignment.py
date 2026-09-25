from types import SimpleNamespace
from unittest.mock import patch

from core.relational_search import ClinicalRelation
from core.rwe.models import RWE_SOURCES
from core.rwe.query_engine import RWEQueryEngine
from core.rwe.reddit_rss_adapter import RedditRWEAdapter
from core.vocab.entities import EntityRecognition


def _relation(query: str) -> ClinicalRelation:
    return ClinicalRelation(
        original_query=query,
        agent={
            "term": "dutasteride", "normalized": "dutasteride", "role": "drug",
            "identified": True, "search_terms": ["dutasteride", "Avodart"],
        },
        event={
            "term": "initial hair shedding", "normalized": "initial hair shedding",
            "search_terms": ["initial hair shedding", "hair shedding", "hair falling"],
        },
        manifestation={
            "term": "hair loss",
            "normalized": "hair loss",
            "role": "condition",
            "search_terms": ["hair loss", "hair shedding", "hair fall"],
        },
        temporal="subsequent recovery",
        relation_type="adverse_effect",
        relation_phrases=["induced by dutasteride", "hair loss caused by dutasteride"],
        scientific_query="dutasteride AND initial hair shedding AND subsequent hair regrowth",
        canonical_query="dutasteride AND initial hair shedding",
    )


def _engine(query: str, relation: ClinicalRelation):
    orchestrator = SimpleNamespace(
        process=lambda _query: SimpleNamespace(
            search_query=query,
            detected_language="it",
            translation_applied=False,
        )
    )
    with patch.object(RWEQueryEngine, "_scientific_relation", return_value=relation), \
            patch("core.rwe.query_engine.build_resolver_from_env", return_value=None):
        return RWEQueryEngine(orchestrator=orchestrator).plan(query)


def test_rwe_reuses_contextual_scientific_relation_without_polluting_canonical():
    query = "caduta indotta da dutasteride"
    relation = _relation(query)
    plan = _engine(query, relation)

    assert plan.original_query == query
    assert plan.clinical_relation is relation
    assert plan.clinical_relation.agent["normalized"] == "dutasteride"
    assert plan.clinical_relation.event["normalized"] == "initial hair shedding"
    assert plan.clinical_relation.manifestation["normalized"] == "hair loss"
    assert plan.clinical_relation.temporal == "subsequent recovery"
    assert plan.clinical_relation.relation_type == "adverse_effect"
    assert plan.clinical_relation.relation_phrases
    expected_context = "dutasteride initial hair shedding hair loss subsequent recovery"
    assert plan.translated_query == expected_context
    assert plan.canonical_query == expected_context
    assert plan.to_dict()["clinical_relation"]["relation_type"] == "adverse_effect"
    assert plan.intent.source == "clinical_relation"
    assert plan.intent.interventions == ["dutasteride"]
    assert plan.intent.outcomes == ["initial hair shedding"]
    assert plan.intent.conditions == ["hair loss"]
    assert "hair falling" in plan.intent.synonyms["initial hair shedding"]
    assert all(
        not any(term in eq.query.lower() for term in (
            "falls", "[x]falls", "alcohol withdrawal", "hallucinosis",
            "fallopian tube diseases",
        ))
        for eq in plan.expanded_queries
    )


def test_rwe_preserves_colloquial_outcome_surfaces_from_relation():
    query = "mi sono caduti un sacco di capelli dopo aver iniziato dutasteride, poi ricrescono?"
    plan = _engine(query, _relation(query))

    assert plan.original_query == query
    assert plan.clinical_relation.agent["normalized"] == "dutasteride"
    assert plan.clinical_relation.manifestation["normalized"] == "hair loss"
    assert plan.intent.synonyms["hair loss"]
    assert plan.expanded_queries[0].query == query


def test_relation_context_rejects_unanchored_provider_entities():
    query = "caduta indotta da dutasteride, poi si recupera?"
    relation = _relation(query)
    noisy = EntityRecognition(
        entities=[
            ("drug", "dutasteride", 1.0),
            ("condition", "hair loss", 1.0),
            ("condition", "falls", 1.0),
            ("condition", "alcohol withdrawal", 1.0),
            ("condition", "hallucinosis", 1.0),
            ("condition", "fallopian tube diseases", 1.0),
        ],
        surfaces={
            "dutasteride": "dutasteride",
            "hair loss": "hair loss",
            "falls": "caduta",
            "alcohol withdrawal": "poi si recupera",
            "hallucinosis": "recupera",
            "fallopian tube diseases": "caduta",
        },
    )
    with patch.object(RWEQueryEngine, "_scientific_relation", return_value=relation), \
            patch("core.rwe.query_engine.build_resolver_from_env", return_value=object()), \
            patch("core.rwe.query_engine.recognize", return_value=noisy):
        plan = RWEQueryEngine().plan(query)

    expanded = " ".join(eq.query.lower() for eq in plan.expanded_queries)
    assert "hair loss" in expanded
    for unrelated in ("falls", "[x]falls", "alcohol withdrawal", "hallucinosis", "fallopian tube diseases"):
        assert unrelated not in expanded




def test_rwe_propagates_nested_manifestation_site_without_entity_side():
    query = "Ho recuperato l'attaccatura dei capelli con dutasteride?"
    relation = _relation(query)
    relation.manifestation = {
        "term": "attaccatura dei capelli",
        "normalized": "hair regrowth",
        "role": "outcome",
        "search_terms": ["hair regrowth", "hair restoration", "hair recovery"],
        "site": {
            "term": "attaccatura dei capelli",
            "normalized": "hairline",
            "search_terms": ["hairline", "frontal hairline", "attaccatura dei capelli"],
        },
    }
    plan = _engine(query, relation)

    assert plan.canonical_query == "dutasteride initial hair shedding hair regrowth hairline subsequent recovery"
    assert plan.intent.manifestation["normalized"] == "hair regrowth"
    assert plan.intent.manifestation["site"]["normalized"] == "hairline"
    assert plan.intent.manifestation["site"]["search_terms"] == [
        "hairline", "frontal hairline", "attaccatura dei capelli"
    ]
    expanded = [eq.query.lower() for eq in plan.expanded_queries]
    assert "dutasteride hair regrowth hairline" in expanded
    assert "dutasteride hair regrowth frontal hairline" in expanded
    assert "dutasteride hair regrowth attaccatura dei capelli" in expanded
    assert all(
        not any(term in query_text for term in (
            "infection", "earlobe", "falls", "alcohol withdrawal", "hallucinosis",
            "fallopian tube diseases", "hair breakage", "hair discoloration",
            "dutasteride-tamsulosin",
        ))
        for query_text in expanded
    )


def test_rwe_does_not_invent_manifestation_site():
    query = "Ho recuperato i capelli con dutasteride?"
    relation = _relation(query)
    relation.manifestation = {
        "term": "capelli",
        "normalized": "hair regrowth",
        "role": "outcome",
        "search_terms": ["hair regrowth", "hair restoration", "hair recovery"],
    }
    plan = _engine(query, relation)

    assert "site" not in plan.intent.manifestation
    assert "hairline" not in plan.canonical_query.lower()
    assert all("hairline" not in eq.query.lower() for eq in plan.expanded_queries)

def test_reddit_rss_adapter_normalizes_public_atom_feed_without_credentials():
    feed = b'''<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
      <entry><title>Initial shedding on dutasteride</title>
      <link href="https://www.reddit.com/r/tressless/comments/abc/post/"/>
      <id>t3_abc</id><updated>2025-01-02T03:04:05Z</updated>
      <content>I noticed hair shedding after starting dutasteride.</content></entry>
    </feed>'''
    response = SimpleNamespace(status_code=200, content=feed)
    with patch("core.rwe.reddit_rss_adapter.requests.get", return_value=response) as get:
        items, status, _reason = RedditRWEAdapter().search_with_status(
            "dutasteride", limit=None
        )

    get.assert_called_once()
    assert status == "ok"
    assert len(items) == 1
    assert items[0].source == "reddit"
    assert items[0].collection_method == "official_rss_feed"
    assert items[0].text.startswith("I noticed hair shedding")


def test_rwe_pipeline_wires_rss_adapter_not_legacy_praw():
    from core.rwe.pipeline import RWEPipeline

    pipe = RWEPipeline()
    assert isinstance(pipe.reddit, RedditRWEAdapter)
    assert pipe.reddit.__class__.__module__ == "core.rwe.reddit_rss_adapter"
    assert RWE_SOURCES["reddit"]["collection_method"] == "official_rss_feed"
