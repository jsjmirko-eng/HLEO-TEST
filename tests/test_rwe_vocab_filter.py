import json
import pytest

from core.rwe.query_engine import RWEQueryEngine
from core.orchestrator import OrchestrationResult
from core.vocab.entities import _candidates, normalize_relation_connectors
from tests.vocab_stubs import FakeResolver, FakeResolution, rxnorm, mesh, patch_resolver

class DummyOrch:
    def process(self, q: str):
        # Return English, no translation applied — avoid LLM calls
        return OrchestrationResult(original_query=q, search_query=q, detected_language='en', translation_applied=False)


def test_finasteride_filters_combo(monkeypatch):
    # Fake resolver returns a rxnorm match with a combo product and a valid brand
    fake_map = {
        ("finasteride", "*"): FakeResolution([
            rxnorm("finasteride", [
                "finasteride 5 MG / tadalafil 5 MG Oral Capsule [Entadfi]",
                "finasteride 5 MG Oral Tablet [Proscar]",
            ], "RX1")
        ])
    }
    resolver = FakeResolver(fake_map)
    patch_resolver(monkeypatch, resolver)

    engine = RWEQueryEngine(orchestrator=DummyOrch())
    plan = engine.plan("Does Finasteride cause temporary hair loss?")
    expanded = [eq["expanded_term"] for eq in plan.to_dict()["expanded_queries"] if eq.get("query_origin") == "vocabulary"]
    # Entadfi (combo) must be blocked, Proscar (brand) must be allowed
    assert not any("entadfi" in (t or "").lower() for t in expanded)
    assert any("proscar" in (t or "").lower() for t in expanded)


def test_hair_condition_event_filter(monkeypatch):
    # Fake resolver for 'hair' with MeSH synonyms including noisy terms
    fake_map = {
        ("hair", "*"): FakeResolution([
            mesh("Hair", ["Hair Cells, Auditory", "Hair Analysis", "Hair Loss"], "M1")
        ])
    }
    resolver = FakeResolver(fake_map)
    patch_resolver(monkeypatch, resolver)

    engine = RWEQueryEngine(orchestrator=DummyOrch())
    plan = engine.plan("Does Finasteride cause temporary hair loss?")
    vocab_expanded = [eq["expanded_term"] for eq in plan.to_dict()["expanded_queries"] if eq.get("query_origin") == "vocabulary"]

    # 'Hair Loss' must remain; 'Hair Cells' and 'Hair Analysis' must be filtered out
    assert any("loss" in (t or "").lower() for t in vocab_expanded)
    assert not any("cells" in (t or "").lower() or "analysis" in (t or "").lower() for t in vocab_expanded)


def test_relation_connectors_are_not_entity_candidates():
    query = "hair loss induced by dutasteride after taking finasteride"
    candidates = _candidates(query)
    assert "induced" not in candidates
    assert "induced by" not in candidates
    assert "after taking" not in candidates
    assert "hair loss" in candidates
    assert "dutasteride" in candidates
    assert normalize_relation_connectors("caduta indotta da dutasteride") \
        == "caduta induced by dutasteride"


def test_relation_filter_blocks_generic_hair_variants_but_keeps_relevant_events():
    base = "hair loss induced by dutasteride"
    noisy = [
        "Hair, Fetal", "Fetal Hair", "Hair Analysis", "hair's breadth",
        "hair follicle",
    ]
    for candidate in noisy:
        assert not RWEQueryEngine._vocab_variant_allowed(
            "hair", candidate, "symptom", base, "mesh"
        )
    assert RWEQueryEngine._vocab_variant_allowed(
        "hair", "hair loss", "symptom", base, "mesh"
    )
    assert RWEQueryEngine._vocab_variant_allowed(
        "dutasteride", "dutasteride acid", "drug", base, "rxnorm"
    )
    assert RWEQueryEngine._vocab_variant_allowed(
        "hair loss", "alopecia", "symptom", base, "mesh"
    )
    assert RWEQueryEngine._vocab_variant_allowed(
        "finasteride", "Proscar", "drug", base, "rxnorm"
    )
