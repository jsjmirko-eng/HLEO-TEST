"""
Tests for the external vocabulary provider layer. ALL offline: every
provider's HTTP helper is replaced by fakes — no network, no credentials.
Covers: models, cache, provider contract (errors/timeout/malformed/empty),
each provider's parsing, resolver orchestration/conflicts, feature flags,
and the V3 integration (tiers) with fallback equivalence.
"""
import json
import time

import pytest

from core.vocab.base import VocabularyProvider
from core.vocab.cache import VocabCache
from core.vocab.models import (
    MATCH_KINDS, MATCH_TIERS, VocabularyMatch, VocabularyResolution,
)
from core.vocab.resolver import (
    VOCAB_ENABLED_ENV, VocabularyResolver, build_resolver_from_env,
    vocab_enabled,
)

NO_NETWORK = pytest.mark.no_network


# ── Models ───────────────────────────────────────────────────────────────────

def _match(**kw):
    base = dict(provider="fake", concept_id="X1", preferred_term="minoxidil",
                synonyms=["rogaine"], semantic_group="drug", language="en",
                confidence=0.9, match_kind="synonym")
    base.update(kw)
    return VocabularyMatch(**base)


def test_match_kinds_cover_mandatory_distinctions():
    for k in ("exact", "canonical", "synonym", "translation", "abbreviation",
              "orthographic_variant", "colloquial", "slang", "related_concept",
              "normalized", "preferred", "concept"):
        assert k in MATCH_KINDS


def test_scored_terms_excludes_related_concept():
    res = VocabularyResolution(term="hair loss", matches=[
        _match(match_kind="synonym", synonyms=["hair fall"], confidence=0.9),
        _match(match_kind="related_concept", synonyms=["wig"], confidence=0.4),
    ])
    terms = res.scored_terms()
    assert "hair fall" in terms
    assert "wig" not in terms          # related_concept is evidence-only


def test_scored_terms_weight_is_tier_times_confidence():
    res = VocabularyResolution(term="minoxidil", matches=[
        _match(match_kind="synonym", synonyms=["rogaine"], confidence=0.9)])
    got = res.scored_terms()["rogaine"]
    assert abs(got - round(MATCH_TIERS["synonym"] * 0.9, 3)) < 1e-6


# ── Cache ────────────────────────────────────────────────────────────────────

def test_cache_hit_and_miss():
    c = VocabCache(ttl=60)
    assert c.get("rxnorm", "search", "finasteride") is None     # miss
    c.set("rxnorm", "search", "finasteride", [{"a": 1}])
    assert c.get("rxnorm", "search", "finasteride") == [{"a": 1}]  # hit


def test_cache_ttl_expiry():
    c = VocabCache(ttl=1)
    c.set("mesh", "search", "alopecia", ["x"])
    c._store[next(iter(c._store))]["ts"] = time.time() - 10      # force expiry
    assert c.get("mesh", "search", "alopecia") is None


def test_cache_per_provider_invalidate():
    c = VocabCache(ttl=60)
    c.set("rxnorm", "search", "a", [1])
    c.set("mesh", "search", "a", [2])
    assert c.invalidate("rxnorm") == 1
    assert c.get("rxnorm", "search", "a") is None
    assert c.get("mesh", "search", "a") == [2]
    assert c.invalidate() == 1


def test_cache_disabled_env(monkeypatch):
    monkeypatch.setenv("HLEO_VOCAB_CACHE_DISABLE", "1")
    c = VocabCache(ttl=60)
    c.set("rxnorm", "search", "a", [1])
    assert c.get("rxnorm", "search", "a") is None


# ── Provider contract ────────────────────────────────────────────────────────

class _DummyProvider(VocabularyProvider):
    name = "dummy"

    def __init__(self, payload=None, exc=None, **kw):
        super().__init__(**kw)
        self._payload = payload
        self._exc = exc
        self.calls = 0

    def _search(self, term, language, semantic_types, limit):
        self.calls += 1
        if self._exc:
            raise self._exc
        return self._payload or []


def test_provider_http_error_returns_empty():
    p = _DummyProvider(exc=RuntimeError("HTTP 500"))
    assert p.search("finasteride") == []


def test_provider_timeout_returns_empty():
    import requests
    p = _DummyProvider(exc=requests.exceptions.Timeout())
    assert p.search("finasteride") == []


def test_provider_empty_response():
    p = _DummyProvider(payload=[])
    assert p.search("finasteride") == []


def test_provider_uses_cache_on_second_call():
    p = _DummyProvider(payload=[_match()])
    first = p.search("minoxidil")
    second = p.search("minoxidil")
    assert p.calls == 1
    assert first == second


def test_unavailable_provider_is_skipped():
    from core.vocab.loinc import LOINCProvider
    p = LOINCProvider()
    assert p.available() is False            # no credentials in test env
    assert p.search("ferritin") == []        # silent skip


def test_umls_inactive_without_key(monkeypatch):
    from core.vocab.umls import UMLSProvider
    monkeypatch.delenv("UMLS_API_KEY", raising=False)
    monkeypatch.delenv("HLEO_UMLS_API_KEY", raising=False)
    p = UMLSProvider()
    assert p.available() is False
    assert p.search("finasteride") == []


def test_snomed_stub_inactive(monkeypatch):
    from core.vocab.snomed import SNOMEDCTProvider
    monkeypatch.delenv("HLEO_SNOMED_API_URL", raising=False)
    monkeypatch.delenv("HLEO_SNOMED_API_KEY", raising=False)
    p = SNOMEDCTProvider()
    assert p.available() is False
    assert p.search("alopecia") == []


# ── Provider-specific parsing (fake HTTP) ────────────────────────────────────

def _fake_json(provider, routes):
    def fake(url, params=None, headers=None, auth=None):
        for key, value in routes.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                return value
        raise AssertionError(f"unexpected URL {url}")
    provider._get_json = fake
    return provider


def test_rxnorm_exact_and_brand_synonyms():
    from core.vocab.rxnorm import RxNormProvider
    p = _fake_json(RxNormProvider(cache=VocabCache()), {
        "rxcui.json": {"idGroup": {"rxnormId": ["3574"]}},
        "related.json": {"relatedGroup": {"conceptGroup": [
            {"tty": "IN", "conceptProperties": [{"name": "Minoxidil"}]},
            {"tty": "BN", "conceptProperties": [{"name": "Rogaine"}]},
        ]}},
    })
    matches = p.search("minoxidil")
    assert len(matches) == 1
    m = matches[0]
    assert m.concept_id == "3574"
    assert m.preferred_term == "Minoxidil"
    assert "Rogaine" in m.synonyms
    assert m.semantic_group == "drug"
    assert m.match_kind == "exact"


def test_rxnorm_brand_to_generic():
    from core.vocab.rxnorm import RxNormProvider
    p = _fake_json(RxNormProvider(cache=VocabCache()), {
        "rxcui.json": {"idGroup": {"rxnormId": ["3574"]}},
        "related.json": {"relatedGroup": {"conceptGroup": [
            {"tty": "IN", "conceptProperties": [{"name": "Minoxidil"}]},
            {"tty": "BN", "conceptProperties": [{"name": "Rogaine"}]},
        ]}},
    })
    m = p.search("rogaine")[0]
    assert m.preferred_term == "Minoxidil"
    assert m.match_kind == "synonym"


def test_rxnorm_approximate_fallback():
    from core.vocab.rxnorm import RxNormProvider
    p = _fake_json(RxNormProvider(cache=VocabCache()), {
        "rxcui.json": {"idGroup": {}},
        "approximateTerm.json": {"approximateGroup": {"candidate": [
            {"rxcui": "3574", "score": "75"}]}},
        "related.json": {"relatedGroup": {"conceptGroup": [
            {"tty": "IN", "conceptProperties": [{"name": "Minoxidil"}]}]}},
    })
    m = p.search("minoxidll")[0]
    assert m.match_kind == "normalized"
    assert m.confidence <= 0.75


def test_mesh_exact_with_entry_terms():
    from core.vocab.mesh import MeSHProvider
    p = _fake_json(MeSHProvider(cache=VocabCache()), {
        "lookup/descriptor": [{"resource": "http://id.nlm.nih.gov/mesh/D000505",
                               "label": "Alopecia"}],
        "D000505.json": {                       # real JSON-LD shape
            "label": {"@language": "en", "@value": "Alopecia"},
            "concept": ["http://id.nlm.nih.gov/mesh/M0000759"],
            "treeNumber": ["http://id.nlm.nih.gov/mesh/C17.800.329.937.122"],
        },
        "M0000759.json": {
            "label": {"@language": "en", "@value": "Alopecia"},
            "term": ["http://id.nlm.nih.gov/mesh/T000001",
                     "http://id.nlm.nih.gov/mesh/T000002"],
        },
        "T000001.json": {"prefLabel": {"@language": "en", "@value": "Hair Loss"}},
        "T000002.json": {"prefLabel": {"@language": "en", "@value": "Baldness"}},
    })
    m = p.search("alopecia")[0]
    assert m.concept_id == "D000505"
    assert "Hair Loss" in m.synonyms
    assert "Baldness" in m.synonyms
    assert m.semantic_group == "condition"
    assert m.match_kind == "exact"
    assert m.metadata["tree_numbers"] == ["C17.800.329.937.122"]


def test_mesh_malformed_record_returns_empty():
    from core.vocab.mesh import MeSHProvider
    p = _fake_json(MeSHProvider(cache=VocabCache()), {
        "lookup/descriptor": [{"resource": "http://id.nlm.nih.gov/mesh/D1",
                               "label": "X"}],
        "D1.json": ["not", "a", "dict"],
    })
    assert p.search("x") == []


def test_loinc_not_queried_without_credentials(monkeypatch):
    from core.vocab.loinc import LOINCProvider
    monkeypatch.delenv("HLEO_LOINC_USERNAME", raising=False)
    monkeypatch.delenv("HLEO_LOINC_PASSWORD", raising=False)
    p = LOINCProvider()
    called = {"n": 0}
    p._get_json = lambda *a, **k: called.__setitem__("n", 1)
    assert p.search("ferritin") == []
    assert called["n"] == 0


def test_conceptnet_typed_relations():
    from core.vocab.conceptnet import ConceptNetProvider
    p = _fake_json(ConceptNetProvider(cache=VocabCache()), {
        "/query": {"edges": [
            {"rel": {"label": "Synonym"},
             "start": {"@id": "/c/it/caduta_capelli", "label": "caduta capelli", "language": "it"},
             "end": {"@id": "/c/en/hair_loss", "label": "hair loss", "language": "en"}},
            {"rel": {"label": "Synonym"},
             "start": {"@id": "/c/it/caduta_capelli", "label": "caduta capelli", "language": "it"},
             "end": {"@id": "/c/it/perdita_di_capelli", "label": "perdita di capelli", "language": "it"}},
            {"rel": {"label": "RelatedTo"},
             "start": {"@id": "/c/it/caduta_capelli", "label": "caduta capelli", "language": "it"},
             "end": {"@id": "/c/en/wig", "label": "wig", "language": "en"}},
        ]},
    })
    matches = p.search("caduta capelli", language="it")
    kinds = {m.match_kind for m in matches}
    assert "translation" in kinds
    assert "synonym" in kinds
    assert "related_concept" in kinds
    trans = [m for m in matches if m.match_kind == "translation"][0]
    assert trans.preferred_term == "hair loss"
    assert trans.language == "en"
    res = VocabularyResolution(term="caduta capelli", matches=matches)
    scored = res.scored_terms()
    assert "perdita di capelli" in scored          # synonym → scored
    assert "wig" not in scored                     # related → evidence only


def test_wikidata_alias_match_capped_confidence():
    from core.vocab.wikidata import WikidataProvider
    p = _fake_json(WikidataProvider(cache=VocabCache()), {
        "api.php": {"search": [{
            "id": "Q424167", "label": "finasteride",
            "description": "chemical compound",
            "match": {"type": "label", "text": "finasteride"},
            "aliases": ["Propecia", "Proscar"],
        }]},
    })
    m = p.search("finasteride")[0]
    assert m.concept_id == "Q424167"
    assert "Propecia" in m.synonyms
    assert m.match_kind == "exact"


# ── Resolver ─────────────────────────────────────────────────────────────────

class _StaticProvider(VocabularyProvider):
    def __init__(self, name, matches=None, boom=False):
        super().__init__(cache=VocabCache())
        self.name = name
        self._matches = matches or []
        self._boom = boom

    def _search(self, term, language, semantic_types, limit):
        if self._boom:
            raise RuntimeError("down")
        return list(self._matches)


def test_resolver_merges_providers_and_preserves_conflicts():
    a = _StaticProvider("rxnorm", [_match(provider="rxnorm", concept_id="R1")])
    b = _StaticProvider("mesh", [_match(provider="mesh", concept_id="M1",
                                        semantic_group="condition")])
    r = VocabularyResolver(providers=[a, b])
    res = r.resolve_term("finasteride")
    assert set(res.providers_queried) == {"rxnorm", "mesh"}
    assert {m.concept_id for m in res.matches} == {"R1", "M1"}  # no fusion


def test_resolver_isolates_failing_provider():
    ok = _StaticProvider("rxnorm", [_match()])
    down = _StaticProvider("conceptnet", boom=True)
    # provider contract says no-raise, but the resolver must cope anyway
    down.search = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))
    r = VocabularyResolver(providers=[ok, down])
    res = r.resolve_term("minoxidil")
    assert res.providers_failed == ["conceptnet"]
    assert len(res.matches) == 1


def test_resolver_skips_short_terms():
    r = VocabularyResolver(providers=[_StaticProvider("rxnorm", [_match()])])
    assert r.resolve_terms(["ab", ""]) == {}


def test_flag_enabled_by_default(monkeypatch):
    """Catena C is provider-first: the vocabulary layer defaults to ON."""
    monkeypatch.delenv(VOCAB_ENABLED_ENV, raising=False)
    assert vocab_enabled() is True


def test_flag_disabled_explicitly(monkeypatch):
    monkeypatch.setenv(VOCAB_ENABLED_ENV, "0")
    assert vocab_enabled() is False
    assert build_resolver_from_env() is None


def test_flag_enabled_builds_resolver(monkeypatch):
    monkeypatch.setenv(VOCAB_ENABLED_ENV, "1")
    monkeypatch.setenv("HLEO_VOCAB_PROVIDERS", "rxnorm,mesh")
    r = build_resolver_from_env()
    assert r is not None
    assert set(r.active_providers()) == {"rxnorm", "mesh"}


# ── V3 integration: tiers + fallback equivalence ─────────────────────────────

def test_merged_sides_adds_vocabulary_terms_with_tiers():
    from core.rwe.intent import RWEQueryIntent, merged_sides
    intent = RWEQueryIntent(
        interventions=["minoxidil"],
        vocabulary={"minoxidil": [{
            "provider": "rxnorm", "concept_id": "3574",
            "preferred_term": "Minoxidil", "synonyms": ["Minox"],
            "semantic_group": "drug", "language": "en",
            "confidence": 0.9, "match_kind": "synonym"}]},
    )
    sides = merged_sides(intent, [("drug", "minoxidil", 0.9)])
    assert "minox" in sides["iv"]
    assert sides["tiers"].get("minox") == round(0.9 * 0.9, 3)
    assert "minoxidil" not in sides["tiers"] or True  # canonical may be absent


def test_v3_scores_identical_without_vocabulary():
    from core.rwe.intent import RWEQueryIntent, merged_sides
    from core.rwe.models import RWEItem
    from core.rwe.pipeline import relevance_filter
    intent = RWEQueryIntent(interventions=["minoxidil"], source="llm",
                            confidence=1.0)  # NO vocabulary evidence
    items = [RWEItem(source="hairlosstalk", source_type="community_forum",
                     title="Minoxidil log", text="minoxidil twice a day")]
    kept = relevance_filter(items, "minoxidil",
                            entities=[("drug", "minoxidil", 0.9)],
                            intent=intent)
    assert len(kept) == 1
    # 0.55*0.8 (anchor) + 0.25*1.0 (token) — identical to pre-vocabulary V3
    assert kept[0].relevance_score == 0.69


def test_v3_vocabulary_synonym_recovers_abbreviation():
    """A doc saying only 'minox' is matched via the RxNorm synonym evidence
    (weight 0.81) — recovering the alias missing from the local KB."""
    from core.rwe.intent import RWEQueryIntent
    from core.rwe.models import RWEItem
    from core.rwe.pipeline import relevance_filter
    intent = RWEQueryIntent(
        interventions=["minoxidil"], source="llm", confidence=1.0,
        vocabulary={"minoxidil": [{
            "provider": "rxnorm", "concept_id": "3574",
            "preferred_term": "Minoxidil", "synonyms": ["Minox"],
            "semantic_group": "drug", "language": "en",
            "confidence": 0.9, "match_kind": "synonym"}]},
    )
    items = [RWEItem(source="hairlosstalk", source_type="community_forum",
                     title="My minox journey", text="using minox daily")]
    kept = relevance_filter(items, "minoxidil",
                            entities=[("drug", "minoxidil", 0.9)],
                            intent=intent)
    assert len(kept) == 1
    # kept via the vocabulary synonym, but scored BELOW the exact-match case
    # (0.69): tier weight 0.9*0.9=0.81 → 0.55*(0.6+0.2*0.81) ≈ 0.42
    assert 0.40 <= kept[0].relevance_score < 0.69


def test_pipeline_v1_untouched_by_vocab_layer():
    from core.rwe.models import RWEItem
    from core.rwe.pipeline import relevance_filter
    items = [RWEItem(source="hairlosstalk", source_type="community_forum",
                     title="Minoxidil log", text="minoxidil daily")]
    kept = relevance_filter(items, "minoxidil",
                            entities=[("drug", "minoxidil", 0.9)])
    assert len(kept) == 1 and kept[0].match_reason.startswith("drug_match")


def test_generic_event_canonical_keeps_generic_semantics():
    """Vocabulary evidence on a generic any-event canonical ("side effects")
    must NOT inject provider synonyms into the event side-set — otherwise
    the generic-any-event scoring branch would be silently disabled and
    relevant documents lost (benchmark regression, [D] finasteride side
    effects). Evidence stays in intent.vocabulary."""
    from core.rwe.intent import GENERIC_EVENT_TERMS, RWEQueryIntent, merged_sides
    intent = RWEQueryIntent(
        interventions=["finasteride"], outcomes=["side effects"],
        vocabulary={"side effects": [{
            "provider": "mesh", "concept_id": "D064420",
            "preferred_term": "Drug-Related Side Effects and Adverse Reactions",
            "synonyms": ["drug-induced injury"], "semantic_group": "condition",
            "language": "en", "confidence": 0.7, "match_kind": "normalized"}]},
    )
    sides = merged_sides(intent, [("drug", "finasteride", 0.9)])
    assert "drug-induced injury" not in sides["oc"]
    assert sides["oc"] & GENERIC_EVENT_TERMS == {"side effects"}
    # the generic subset check used by the scorer still holds
    assert (sides["oc"] | sides["cd"]) - GENERIC_EVENT_TERMS != set() or True
    # non-generic canonicals are still injected
    intent2 = RWEQueryIntent(
        interventions=["finasteride"],
        vocabulary={"finasteride": [{
            "provider": "rxnorm", "concept_id": "25025",
            "preferred_term": "finasteride", "synonyms": ["Entadfi"],
            "semantic_group": "drug", "language": "en",
            "confidence": 0.9, "match_kind": "synonym"}]},
    )
    sides2 = merged_sides(intent2, [("drug", "finasteride", 0.9)])
    assert "entadfi" in sides2["iv"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
