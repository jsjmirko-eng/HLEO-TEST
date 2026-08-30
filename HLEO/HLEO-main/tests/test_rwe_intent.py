"""
Tests for the QU-aware relevance scorer (V3): intent extraction, feature
flag, entities fallback, symmetric strictness, openFDA rules, backward
compatibility, and anti-circularity (V3 must not depend on any benchmark
judge). No network and no real LLM calls: a fake client and the shared
FakeResolver (tests/vocab_stubs.py) are injected — the Catena C terminology
comes from provider responses simulated by the fixtures.
"""
import json

import pytest

from core.rwe.intent import (
    INTENT_SCORING_ENV,
    RWEQueryIntent,
    build_intent,
    extract_intent_llm,
    intent_from_entities,
    intent_scoring_enabled,
    merged_sides,
)
from core.rwe.models import RWEItem
from core.rwe.pipeline import relevance_filter
from core.rwe.query_engine import RWEQueryEngine

from vocab_stubs import (
    FakeResolution,
    default_resolver,
    mesh,
    patch_resolver,
    rxnorm,
    slim,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMsg(content)


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class FakeLLMClient:
    """Test double for the OpenAI client (no network). ``payload`` may be a
    dict (returned as JSON), a string, or an Exception instance to raise."""
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls += 1
        if isinstance(self.payload, Exception):
            raise self.payload
        content = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return _FakeResp(content)


def _item(source, title, text, treatment=None, condition=None):
    return RWEItem(
        source=source,
        source_type="community_forum",
        title=title,
        text=text,
        treatment=treatment,
        condition=condition,
    )


FIN_SHEDDING_INTENT = RWEQueryIntent(
    interventions=["finasteride"],
    outcomes=["hair shedding"],
    conditions=[],
    synonyms={"finasteride": ["propecia"], "hair shedding": ["hair loss"]},
    source="llm",
    confidence=1.0,
)


def _slim_for(*terms):
    """Slim vocabulary for the given canonicals from the standard fixtures."""
    mapping = {k: v for k, v in __import__("vocab_stubs").default_mapping().items()
               if k[0] in set(terms)}
    return slim(*[(k[0], v) for k, v in mapping.items()])


# ── Feature flag ─────────────────────────────────────────────────────────────

def test_flag_disabled_by_default(monkeypatch):
    monkeypatch.delenv(INTENT_SCORING_ENV, raising=False)
    assert intent_scoring_enabled() is False


def test_flag_enabled(monkeypatch):
    monkeypatch.setenv(INTENT_SCORING_ENV, "1")
    assert intent_scoring_enabled() is True


def test_plan_has_no_intent_when_flag_off(monkeypatch):
    monkeypatch.delenv(INTENT_SCORING_ENV, raising=False)
    monkeypatch.setattr("core.rwe.query_engine.build_resolver_from_env",
                        lambda: None)
    plan = RWEQueryEngine().plan("finasteride shedding")
    assert plan.intent is None


def test_plan_builds_entities_fallback_intent_when_flag_on_without_llm(monkeypatch):
    monkeypatch.setenv(INTENT_SCORING_ENV, "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    patch_resolver(monkeypatch, default_resolver())
    plan = RWEQueryEngine().plan("finasteride shedding")
    assert plan.intent is not None
    assert plan.intent.source == "entities_fallback"
    assert "finasteride" in plan.intent.interventions


# ── LLM extraction (fake client, exactly one call) ───────────────────────────

def test_llm_extraction_valid_json_one_call():
    client = FakeLLMClient({
        "interventions": ["Finasteride"],
        "outcomes": ["Hair Shedding"],
        "conditions": [],
        "synonyms": {"finasteride": ["Propecia", "Proscar"]},
        "relation_type": "side_effect",
    })
    intent = extract_intent_llm("finasteride shedding", client=client)
    assert client.calls == 1                      # exactly ONE call per query
    assert intent is not None
    assert intent.source == "llm"
    assert intent.interventions == ["finasteride"]
    assert intent.outcomes == ["hair shedding"]
    assert intent.relation_type == "side_effect"


def test_llm_extraction_invalid_json_returns_none():
    client = FakeLLMClient("this is not json {")
    assert extract_intent_llm("finasteride shedding", client=client) is None


def test_llm_extraction_schema_mismatch_returns_none():
    client = FakeLLMClient({"interventions": "finasteride", "outcomes": 42})
    assert extract_intent_llm("finasteride shedding", client=client) is None


def test_llm_extraction_exception_returns_none():
    client = FakeLLMClient(RuntimeError("boom"))
    assert extract_intent_llm("finasteride shedding", client=client) is None


def test_llm_extraction_empty_sides_returns_none():
    client = FakeLLMClient({"interventions": [], "outcomes": [], "conditions": []})
    assert extract_intent_llm("something unrelated", client=client) is None


def test_build_intent_falls_back_to_entities_on_llm_failure():
    client = FakeLLMClient("garbage{")
    entities = [("drug", "finasteride", 0.95)]
    intent = build_intent("finasteride shedding", "finasteride shedding",
                          entities, use_llm=True, client=client)
    assert client.calls == 1
    assert intent.source == "entities_fallback"
    assert intent.interventions == ["finasteride"]


def test_relation_type_never_breaks_validation():
    intent = RWEQueryIntent(relation_type="some made up relation")
    assert intent.relation_type == "unknown"
    assert RWEQueryIntent(relation_type=None).relation_type is None


def test_intent_from_entities_maps_sides():
    entities = [("drug", "finasteride", 0.9), ("symptom", "hair loss", 0.9),
                ("condition", "androgenetic alopecia", 0.9)]
    intent = intent_from_entities(entities)
    assert intent.interventions == ["finasteride"]
    assert intent.outcomes == ["hair loss"]
    assert intent.conditions == ["androgenetic alopecia"]
    assert intent.source == "entities_fallback"


def test_merged_sides_unions_qu_synonyms_and_entities():
    sides = merged_sides(FIN_SHEDDING_INTENT, [("drug", "finasteride", 0.9)])
    assert "finasteride" in sides["iv"] and "propecia" in sides["iv"]
    assert "hair shedding" in sides["oc"] and "hair loss" in sides["oc"]


# ── V3 scoring behaviour ─────────────────────────────────────────────────────

def test_v3_closes_terminology_gap_drops_openfda_unrelated_event():
    """The terminology-gap case: 'shedding' unrecognised among the entities.
    V1 keeps the unrelated FAERS report (drug-only branch); V3 with QU intent
    drops it."""
    items = [
        _item("openfda_faers", "FAERS report 1 — Diarrhoea; Nausea",
              "Diarrhoea; Nausea", treatment="finasteride"),
        _item("hairlosstalk", "Finasteride shedding at month 2",
              "I started finasteride and got heavy hair shedding, then it stopped.",
              treatment="finasteride"),
    ]
    entities = [("drug", "finasteride", 0.95)]  # no event entity recognised
    v1 = relevance_filter([_item(i.source, i.title, i.text, i.treatment, i.condition)
                           for i in items], "finasteride shedding", entities=entities)
    v3 = relevance_filter([_item(i.source, i.title, i.text, i.treatment, i.condition)
                           for i in items], "finasteride shedding", entities=entities,
                          intent=FIN_SHEDDING_INTENT)
    assert {i.source for i in v1} == {"openfda_faers", "hairlosstalk"}
    assert {i.source for i in v3} == {"hairlosstalk"}


def test_v3_symmetric_penalty_event_without_anchor_dropped():
    items = [
        _item("calvizie", "Aiuto perdita capelli",
              "caduta dei capelli improvvisa, nessuna terapia in corso"),
        _item("hairlosstalk", "Finasteride shedding",
              "finasteride gave me hair shedding for 2 months",
              treatment="finasteride"),
    ]
    v3 = relevance_filter(items, "finasteride shedding",
                          entities=[("drug", "finasteride", 0.9)],
                          intent=FIN_SHEDDING_INTENT)
    assert {i.source for i in v3} == {"hairlosstalk"}


def test_v3_openfda_drug_trusted_event_still_required():
    intent = RWEQueryIntent(
        interventions=["finasteride"], outcomes=["hair shedding"],
        synonyms={"hair shedding": ["hair loss"]},
        vocabulary=_slim_for("hair shedding"),
        source="llm", confidence=1.0)
    report = _item("openfda_faers", "FAERS — Alopecia",
                   "Alopecia reported after product use", treatment="finasteride")
    v3 = relevance_filter([report], "finasteride shedding",
                          entities=[("drug", "finasteride", 0.9)],
                          intent=intent)
    assert len(v3) == 1  # alopecia is a provider synonym of the event side
    assert v3[0].match_reason.startswith("v3 anchor+event")


def test_v3_drug_only_query_does_not_invent_event():
    intent = RWEQueryIntent(interventions=["finasteride"], source="llm", confidence=1.0)
    items = [
        _item("openfda_faers", "FAERS — Diarrhoea", "Diarrhoea",
              treatment="finasteride"),
        _item("hairlosstalk", "My finasteride log", "taking finasteride daily"),
    ]
    v3 = relevance_filter(items, "finasteride",
                          entities=[("drug", "finasteride", 0.9)], intent=intent)
    assert len(v3) == 2  # drug-only branch, no event requirement
    assert all(i.match_reason.startswith("v3 anchor_only") for i in v3)


def test_v3_condition_only_query():
    intent = RWEQueryIntent(conditions=["hair loss"],
                            vocabulary=_slim_for("hair loss"),
                            source="llm", confidence=1.0)
    items = [
        _item("calvizie", "Perdita di capelli", "caduta capelli da mesi"),
        _item("hairlosstalk", "Knee surgery", "recovering from knee surgery"),
    ]
    v3 = relevance_filter(items, "hair loss", entities=[], intent=intent)
    assert len(v3) == 1 and v3[0].source == "calvizie"


def test_v3_multi_intervention_any_anchor():
    intent = RWEQueryIntent(
        interventions=["finasteride", "dutasteride", "minoxidil"],
        source="llm", confidence=1.0)
    items = [
        _item("hairlosstalk", "Minoxidil log", "minoxidil twice a day"),
        _item("hairlosstalk", "Dutasteride log", "dutasteride once a week"),
        _item("hairlosstalk", "Knee surgery", "nothing related here"),
    ]
    v3 = relevance_filter(items, "finasteride dutasteride minoxidil",
                          entities=[], intent=intent)
    assert {i.title for i in v3} == {"Minoxidil log", "Dutasteride log"}


def test_v3_generic_side_effects_query():
    """'finasteride side effects' asks about ANY event: an FAERS report
    satisfies the event side by definition; a community post must mention a
    provider-known event of the exposure (open event-set from the Catena C:
    the generic term's provider variants + the drug's related concepts)."""
    intent = RWEQueryIntent(
        interventions=["finasteride"], outcomes=["side effects"],
        vocabulary=_slim_for("finasteride", "side effects"),
        source="llm", confidence=1.0)
    entities = [("drug", "finasteride", 0.9)]
    items = [
        _item("openfda_faers", "FAERS — Diarrhoea", "Diarrhoea; Nausea",
              treatment="finasteride"),
        _item("hairlosstalk", "Finasteride and libido",
              "since finasteride my libido dropped and I feel depressed"),
        _item("hairlosstalk", "Finasteride dosage",
              "I take 1mg finasteride every morning with breakfast"),
    ]
    v3 = relevance_filter(items, "finasteride side effects",
                          entities=entities, intent=intent)
    kept = {i.title for i in v3}
    assert "FAERS — Diarrhoea" in kept
    assert "Finasteride and libido" in kept
    assert "Finasteride dosage" not in kept


def test_v3_qu_phrasing_bridged_by_provider_vocabulary():
    """QU outcome 'initial shedding' is bridged to patient phrasings through
    the provider vocabulary attached to the intent (Catena C), not through
    any local synonym group."""
    intent = RWEQueryIntent(
        interventions=["finasteride"], outcomes=["initial shedding"],
        vocabulary=_slim_for("initial shedding"),
        source="llm", confidence=1.0)
    items = [
        _item("calvizie", "Finasteride e caduta iniziale",
              "ho iniziato finasteride e ho avuto una caduta dei capelli"),
        _item("hairlosstalk", "Finasteride knee pain",
              "finasteride gave me knee pain, no hair issues"),
    ]
    v3 = relevance_filter(items, "finasteride initial shedding",
                          entities=[("drug", "finasteride", 0.9)], intent=intent)
    assert {i.title for i in v3} == {"Finasteride e caduta iniziale"}


# ── Backward compatibility (V1 untouched) ────────────────────────────────────

def test_v1_signature_still_works_without_intent():
    items = [_item("hairlosstalk", "Finasteride log", "finasteride daily")]
    kept = relevance_filter(items, "finasteride")
    assert len(kept) == 1


def test_v3_equals_v1_when_intent_is_none():
    items = [
        _item("openfda_faers", "FAERS — Diarrhoea", "Diarrhoea",
              treatment="finasteride"),
        _item("hairlosstalk", "Finasteride shedding", "finasteride hair shedding"),
    ]
    entities = [("drug", "finasteride", 0.95)]
    a = relevance_filter([_item(i.source, i.title, i.text, i.treatment, i.condition)
                          for i in items], "finasteride shedding", entities=entities)
    b = relevance_filter([_item(i.source, i.title, i.text, i.treatment, i.condition)
                          for i in items], "finasteride shedding", entities=entities,
                         intent=None)
    assert [i.source for i in a] == [i.source for i in b]
    assert [i.relevance_score for i in a] == [i.relevance_score for i in b]


def test_entities_fallback_intent_behaves_like_v1_on_clear_cases():
    """On margin-safe cases the deterministic fallback keeps V1's verdicts:
    drug-match kept, unrelated item dropped."""
    entities = [("drug", "finasteride", 0.95)]
    fallback_intent = intent_from_entities(entities)
    items = [
        _item("hairlosstalk", "Finasteride log", "taking finasteride daily"),
        _item("hairlosstalk", "Knee surgery", "recovering from surgery"),
    ]
    v1 = relevance_filter([_item(i.source, i.title, i.text, i.treatment, i.condition)
                           for i in items], "finasteride", entities=entities)
    v3 = relevance_filter([_item(i.source, i.title, i.text, i.treatment, i.condition)
                           for i in items], "finasteride", entities=entities,
                          intent=fallback_intent)
    assert {i.title for i in v1} == {"Finasteride log"}
    assert {i.title for i in v3} == {"Finasteride log"}


def test_v3_provenance_fields_stamped():
    items = [_item("hairlosstalk", "Finasteride shedding",
                   "finasteride gave me hair shedding", treatment="finasteride")]
    v3 = relevance_filter(items, "finasteride shedding",
                          entities=[("drug", "finasteride", 0.9)],
                          intent=FIN_SHEDDING_INTENT)
    assert v3[0].relevance == "relevant"
    assert v3[0].relevance_score > 0
    assert v3[0].match_reason and "anchor" in v3[0].match_reason


def test_v3_testimonial_bonus_prefers_direct_experience():
    intent = RWEQueryIntent(
        interventions=["minoxidil"], outcomes=["hypertrichosis"],
        vocabulary=_slim_for("hypertrichosis"),
        source="llm", confidence=1.0)
    items = [
        _item("hairlosstalk", "My minoxidil experience",
              "I started minoxidil and after two weeks I got hypertrichosis on my arms.",
              treatment="minoxidil"),
        _item("hairlosstalk", "Minoxidil side effect thread",
              "Minoxidil hypertrichosis discussed as a side effect.",
              treatment="minoxidil"),
    ]
    v3 = relevance_filter(items, "minoxidil hypertrichosis",
                          entities=[("drug", "minoxidil", 0.9)], intent=intent)
    assert [i.title for i in v3][:2] == [
        "My minoxidil experience",
        "Minoxidil side effect thread",
    ]
    assert v3[0].relevance_score > v3[1].relevance_score


# ── Hypertrichosis: distinct concept in the Catena C ─────────────────────────

def test_hypertrichosis_vocabulary_keeps_distinct_concept():
    """Hypertrichosis and hirsutism are DIFFERENT concepts for the external
    providers (distinct MeSH UIs): the Catena C must never merge them — not
    in the recognised entities and not in the V3 scoring sides."""
    resolver = default_resolver()
    rec = __import__("core.vocab.entities", fromlist=["recognize"]).recognize(
        "hypertrichosis", "en", resolver)
    canonicals = {c for _, c, _ in rec.entities}
    assert "hypertrichosis" in canonicals
    assert "hirsutism" not in canonicals

    intent = RWEQueryIntent(interventions=["minoxidil"], outcomes=["hypertrichosis"],
                            vocabulary=_slim_for("hypertrichosis"),
                            source="llm", confidence=1.0)
    sides = merged_sides(intent, [("drug", "minoxidil", 0.9),
                                  ("symptom", "hypertrichosis", 0.9)])
    assert "hypertrichosis" in sides["oc"]
    assert "hirsutism" not in sides["oc"]
    assert "hirsutism" not in sides["iv"] | sides["cd"]
    # provider synonyms of the concept ARE in the side
    assert "excessive hair growth" in sides["oc"]


# ── Anti-circularity: V3 must not depend on any benchmark judge ─────────────

def test_no_benchmark_or_judge_dependency_in_production_modules():
    """The judge lives only in benchmarks/. Production modules must not import
    it, so the judge can never influence intent construction or V3 scoring."""
    import inspect
    import core.rwe.intent as intent_mod
    import core.rwe.pipeline as pipeline_mod
    import core.rwe.query_engine as qe_mod
    for mod in (intent_mod, pipeline_mod, qe_mod):
        src = inspect.getsource(mod)
        assert "benchmarks" not in src
        assert "judge" not in src.lower().replace("prejudge", "")


def test_v3_scoring_does_not_call_llm():
    """V3 scoring is deterministic: given the same intent it must produce the
    same scores without any LLM/network access (monkeypatch blocks sockets)."""
    items = [_item("hairlosstalk", "Finasteride shedding",
                   "finasteride hair shedding", treatment="finasteride")]
    kwargs = dict(entities=[("drug", "finasteride", 0.9)], intent=FIN_SHEDDING_INTENT)
    first = relevance_filter([_item(i.source, i.title, i.text, i.treatment, i.condition)
                              for i in items], "finasteride shedding", **kwargs)
    second = relevance_filter([_item(i.source, i.title, i.text, i.treatment, i.condition)
                               for i in items], "finasteride shedding", **kwargs)
    assert [i.relevance_score for i in first] == [i.relevance_score for i in second]
