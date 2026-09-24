"""Focused regressions for provider-driven RWE canonicalization."""
from types import SimpleNamespace
from unittest.mock import patch

from core.rwe.query_engine import RWEQueryEngine
from vocab_stubs import FakeMatch, FakeResolution, FakeResolver, rxnorm


def _noisy_provider_resolver():
    return FakeResolver({
        ("dutasteride", "*"): FakeResolution([
            rxnorm("dutasteride", [
                "avodart", "dutasteride acid", "dutasteride-tamsulosin",
            ], "RX157"),
        ]),
        ("fall", "*"): FakeResolution([
            FakeMatch("falls", ["[X]Falls", "Falls, Accidental"],
                      match_kind="normalized", provider="umls",
                      semantic_group="symptom", confidence=0.7),
        ]),
        ("induced", "*"): FakeResolution([
            FakeMatch("autoimmune-inflammatory syndrome induced by adjuvants",
                      match_kind="normalized", provider="umls",
                      semantic_group="condition", confidence=0.7),
            FakeMatch("alcohol withdrawal hallucinosis",
                      match_kind="normalized", provider="umls",
                      semantic_group="condition", confidence=0.7),
        ]),
        ("fall induced", "*"): FakeResolution([
            FakeMatch("autoimmune-inflammatory syndrome induced by adjuvants",
                      match_kind="normalized", provider="umls",
                      semantic_group="condition", confidence=0.7),
        ]),
    })


def test_canonicalization_rejects_unrelated_provider_n_gram_hits(monkeypatch):
    resolver = _noisy_provider_resolver()
    monkeypatch.setattr("core.rwe.query_engine.build_resolver_from_env", lambda: resolver)
    with patch("core.rwe.query_engine.QueryOrchestrator") as mock_orchestrator:
        mock_orchestrator.return_value.process.return_value = SimpleNamespace(
            search_query="fall induced by dutasteride",
            detected_language="it",
            translation_applied=True,
        )
        plan = RWEQueryEngine().plan("caduta indotta da dutasteride")

    assert plan.canonical_query == "fall induced by dutasteride"
    canonicals = {canonical for _, canonical, _ in plan.entities}
    assert "dutasteride" in canonicals
    assert "fall" in canonicals
    assert "falls" not in canonicals
    assert not any("autoimmune" in canonical for canonical in canonicals)
    assert not any("alcohol" in canonical for canonical in canonicals)

    expanded = [item.query.lower() for item in plan.expanded_queries]
    joined = " ".join(expanded)
    assert any(term in joined for term in ("hair fall", "hair shedding", "hair loss", "fall"))
    for forbidden in (
        "autoimmune-inflammatory syndrome",
        "adjuvants",
        "alcohol withdrawal",
        "hallucinosis",
        "[xfalls]",
        "dutasteride acid",
        "dutasteride-tamsulosin",
    ):
        assert forbidden not in joined


def test_provider_variants_require_anchor_overlap_for_multiword_candidates():
    resolution = FakeResolution([
        FakeMatch("dutasteride", ["dutasteride acid", "dutasteride-tamsulosin", "Avodart"],
                  match_kind="synonym", provider="umls", semantic_group="drug"),
    ])
    assert RWEQueryEngine._vocab_variant_allowed(
        "dutasteride", "Avodart", "drug", "fall induced by dutasteride", "umls"
    ) is False
    assert RWEQueryEngine._vocab_variant_allowed(
        "dutasteride", "dutasteride acid", "drug", "fall induced by dutasteride", "umls"
    ) is False
    assert RWEQueryEngine._vocab_variant_allowed(
        "dutasteride", "dutasteride 0.5 MG Oral Capsule", "drug",
        "fall induced by dutasteride", "umls"
    ) is True
    assert resolution.matches[0].preferred_term == "dutasteride"
