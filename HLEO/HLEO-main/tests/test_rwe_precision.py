"""Tests for the RWE relation-precision gate (Levels A/B/C), entity
sanitisation, and structured RWE profile extraction.

All offline: synthetic RWEItem instances + hand-built plans; no network,
no LLM, no DB.
"""
from __future__ import annotations

from types import SimpleNamespace

from core.rwe.models import RWEItem
from core.rwe.query_engine import RWEQueryEngine
from core.rwe.relation_filter import (
    apply_relation_gate,
    build_relation_context,
    extract_rwe_profile,
)

# ── Plan fixture: "finasteride sexual dysfunction adverse effects" ───────────
# Mirrors the REAL provider output shape observed live (RxNorm + MeSH), after
# entity sanitisation: finasteride (drug) + sexual dysfunction (symptom).

def _plan():
    vocabulary = {
        "finasteride": [
            {"provider": "rxnorm", "concept_id": "RX156",
             "preferred_term": "finasteride",
             "synonyms": ["propecia", "proscar"],
             "semantic_group": "drug", "language": "en",
             "confidence": 0.95, "match_kind": "exact"},
            {"provider": "conceptnet", "concept_id": "/c/en/finasteride",
             "preferred_term": "bicalutamide",
             "synonyms": ["casodex"],
             "semantic_group": "drug", "language": "en",
             "confidence": 0.6, "match_kind": "related_concept"},
        ],
        "sexual dysfunction, physiological": [
            {"provider": "mesh", "concept_id": "D012784",
             "preferred_term": "Sexual Dysfunction, Physiological",
             "synonyms": ["erectile dysfunction", "libido loss", "impotence",
                          "sexual side effects"],
             "semantic_group": "symptom", "language": "en",
             "confidence": 0.95, "match_kind": "exact"},
        ],
    }
    return SimpleNamespace(
        entities=[
            ("drug", "finasteride", 0.95),
            ("symptom", "sexual dysfunction, physiological", 0.95),
        ],
        vocabulary=vocabulary,
        surfaces={
            "finasteride": "finasteride",
            "sexual dysfunction, physiological": "sexual dysfunction",
        },
    )


def _item(title, text, source="hairlosstalk", source_type="community_forum",
          treatment=None, condition=None, score=0.8):
    return RWEItem(
        source=source, source_type=source_type, title=title, text=text,
        treatment=treatment, condition=condition, relevance_score=score,
    )


# ── Entity sanitisation ──────────────────────────────────────────────────────

def test_sanitize_drops_single_token_surface_multiword_canonical():
    entities = [
        ("drug", "finasteride", 0.95),
        ("condition", "child abuse, sexual", 0.9),
        ("condition", "military sexual trauma", 0.9),
        ("condition", "meibomian gland dysfunction", 0.9),
        ("symptom", "sexual dysfunction, physiological", 0.95),
        ("symptom", "alopecia", 0.7),           # from 1-token "shedding"
    ]
    surfaces = {
        "finasteride": "finasteride",
        "child abuse, sexual": "sexual",
        "military sexual trauma": "sexual",
        "meibomian gland dysfunction": "dysfunction",
        "sexual dysfunction, physiological": "sexual dysfunction",
        "alopecia": "shedding",
    }
    resolutions = {c: object() for c in surfaces}
    kept, res, surf = RWEQueryEngine._sanitize_entities(
        entities, resolutions, surfaces)
    kept_canonicals = {c for _t, c, _conf in kept}
    assert kept_canonicals == {
        "finasteride", "sexual dysfunction, physiological", "alopecia"}
    assert set(res) == kept_canonicals
    assert set(surf) == kept_canonicals


def test_sanitize_keeps_multitoken_surface_lexical_gap():
    # "hair loss" → "alopecia": zero lexical overlap but a specific phrase the
    # user wrote — trusted.
    entities = [("symptom", "alopecia", 0.9)]
    surfaces = {"alopecia": "hair loss"}
    kept, _, _ = RWEQueryEngine._sanitize_entities(entities, {}, surfaces)
    assert [c for _t, c, _conf in kept] == ["alopecia"]


# ── Relation context ─────────────────────────────────────────────────────────

def test_context_agent_and_manifestation_terms():
    ctx = build_relation_context(_plan())
    assert ctx is not None
    assert "finasteride" in ctx.agent_terms
    assert "propecia" in ctx.agent_terms and "proscar" in ctx.agent_terms
    assert "sexual dysfunction" in ctx.manifestation_terms
    assert "erectile dysfunction" in ctx.manifestation_terms
    assert "impotence" in ctx.manifestation_terms
    # related_concept of the anchor is evidence for other-agent detection
    assert "bicalutamide" in ctx.other_agent_terms
    assert ctx.manifestation_tokens == {"sexual", "dysfunction"}


def test_context_none_without_structured_relation():
    plan = SimpleNamespace(entities=[("drug", "finasteride", 0.9)],
                           vocabulary={}, surfaces={})
    assert build_relation_context(plan) is None
    items = [_item("t", "x")]
    out, stats = apply_relation_gate(items, plan)
    assert out is items and stats is None


# ── Level A/B/C gate ─────────────────────────────────────────────────────────

def test_true_positive_explicit_relation_kept():
    it = _item("Finasteride sides",
               "After taking finasteride I developed erectile dysfunction.")
    kept, stats = apply_relation_gate([it], _plan())
    assert kept == [it]
    gate = it.metadata["relation_gate"]
    assert gate["level_a"] and gate["level_b"] and gate["level_c"]
    assert stats["final"] == 1


def test_child_abuse_text_dropped():
    it = _item("Finasteride and my story",
               "I take finasteride for hair loss. On another note, the "
               "documentary about child abuse, sexual trauma survivors and "
               "military sexual trauma really moved me.")
    kept, stats = apply_relation_gate([it], _plan())
    assert kept == []
    assert stats["dropped_manifestation"] == 1


def test_meibomian_dropped():
    it = _item("Finasteride and dry eyes",
               "I use finasteride daily and my ophthalmologist diagnosed "
               "meibomian gland dysfunction last week.")
    kept, stats = apply_relation_gate([it], _plan())
    assert kept == []
    assert stats["dropped_manifestation"] == 1


def test_other_agent_experience_dropped_at_relation():
    # Finasteride mentioned in passing; the sexual problem is linked to
    # bicalutamide, far from the finasteride mention.
    it = _item("Oral bicalutamide log",
               "Week 12 on bicalutamide: libido gone and erectile dysfunction "
               "is real. "
               + "Padding text about shampoos and lifestyle. " * 8
               + "I once tried finasteride years ago, nothing to report.")
    kept, stats = apply_relation_gate([it], _plan())
    assert kept == []
    assert stats["dropped_relation"] == 1


def test_isolated_sexual_word_not_enough():
    it = _item("Finasteride dosage question",
               "Can I take finasteride at night? My partner and I have a "
               "great sexual life, no complaints at all.")
    kept, stats = apply_relation_gate([it], _plan())
    assert kept == []


def test_sentence_level_copresence_passes():
    # Phrase tokens co-present in one sentence (wording variant, no exact
    # provider phrase) + agent in the same sentence.
    it = _item("Fin progress",
               "My finasteride journey: the sexual side — a dysfunction that "
               "worried me — started in month two.")
    kept, _stats = apply_relation_gate([it], _plan())
    assert kept == [it]


def test_distant_mentions_fail_level_c():
    text = ("Finasteride 1mg daily log. " + "Hair looks great. " * 30
            + "By the way my brother has erectile dysfunction.")
    it = _item("My finasteride log", text)
    kept, stats = apply_relation_gate([it], _plan())
    assert kept == []
    assert stats["dropped_relation"] == 1


def test_authoritative_structured_record_kept():
    it = _item("FAERS report", "Erectile dysfunction reported.",
               source="openfda_faers", source_type="pharmacovigilance",
               treatment="finasteride", condition="erectile dysfunction")
    kept, _stats = apply_relation_gate([it], _plan())
    assert kept == [it]
    assert it.metadata["relation_gate"]["relation_kind"] == "structured_record"


def test_other_agent_title_penalised_in_ranking():
    strong = _item("Finasteride erectile dysfunction",
                   "After taking finasteride I developed erectile dysfunction.")
    weak = _item("Bicalutamide log — week 8",
                 "Switched from finasteride to bicalutamide; the erectile "
                 "dysfunction from finasteride persisted anyway.")
    kept, _ = apply_relation_gate([weak, strong], _plan())
    assert len(kept) == 2
    assert kept[0].metadata["base_relevance_score"] is not None
    # the other-agent-title item is penalised relative to its base score
    assert weak.relevance_score < weak.metadata["base_relevance_score"] + 0.10
    assert weak.metadata["relation_gate"]["other_agent_in_title"]


def test_negated_relation_dropped():
    # Real-world case: finasteride explicitly did NOT cause the sexual side
    # effects ("never gave me any…"); the adverse event is from darolutamide.
    it = _item("Any supplement or something of the like to restore morning wood?",
               "Used darolutamide after using finasteride/cb before that "
               "which never gave me any sexual side effects; but not long "
               "into daro usage I stopped getting nocturnal erections.")
    kept, stats = apply_relation_gate([it], _plan())
    assert kept == []
    assert stats["dropped_relation"] == 1
    assert stats["dropped_relation_negated"] == 1
    assert it.metadata["relation_gate"]["drop_reason"] == "relation_negated"


def test_affirmed_relation_not_dropped_by_negation_guard():
    it = _item("Finasteride erectile dysfunction",
               "After taking finasteride I developed erectile dysfunction.")
    kept, _stats = apply_relation_gate([it], _plan())
    assert kept == [it]


# ── Structured profile extraction ────────────────────────────────────────────

def test_profile_extraction_full():
    it = _item(
        "Gyno after 9 months on oral finasteride",
        "I've been on oral finasteride 1mg for roughly 9 months. After 8 "
        "months I noticed lumps. It gave me gyno. I came off it and it "
        "improved. I restarted microdosing 3 times a week.")
    prof = extract_rwe_profile(it, "finasteride", "gyno", "sentence")
    assert prof["agent"] == "finasteride"
    assert prof["manifestation"] == "gyno"
    assert prof["relation"] == "sentence"
    assert "9 months" in prof["duration"]
    assert "1mg" in prof["dose"]
    assert prof["discontinued"] is True
    assert prof["outcome"] == "improved"
    assert prof["rechallenge"] is True
    assert prof["context"] == "personal_testimony"
    assert prof["onset"]


def test_profile_empty_fields_never_invented():
    it = _item("Finasteride erectile dysfunction",
               "After taking finasteride I developed erectile dysfunction.")
    prof = extract_rwe_profile(it, "finasteride", "erectile dysfunction",
                               "sentence")
    assert prof["dose"] == ""
    assert prof["duration"] == ""
    assert prof["discontinued"] is False
    assert prof["rechallenge"] is False


# ── Generic-hypernym manifestation guard (isotretinoin case) ────────────────

def _iso_plan():
    """Mirrors the live isotretinoin plan: spurious generic 'pain' entity +
    intent with specific outcomes (joint pain / stiffness)."""
    vocabulary = {
        "isotretinoin": [
            {"provider": "rxnorm", "concept_id": "RX1",
             "preferred_term": "isotretinoin", "synonyms": ["accutane"],
             "semantic_group": "drug", "language": "en",
             "confidence": 1.0, "match_kind": "exact"},
        ],
        "pain": [
            {"provider": "mesh", "concept_id": "D010146",
             "preferred_term": "pain",
             "synonyms": ["abdominal pain", "back pain", "acute pain"],
             "semantic_group": "symptom", "language": "en",
             "confidence": 1.0, "match_kind": "exact"},
        ],
    }
    intent = SimpleNamespace(
        interventions=["isotretinoin"],
        outcomes=["joint pain", "stiffness"],
        synonyms={"joint pain": ["arthralgia", "joint ache"],
                  "stiffness": ["joint stiffness", "rigidity"]},
    )
    return SimpleNamespace(
        entities=[("drug", "isotretinoin", 1.0), ("symptom", "pain", 1.0)],
        vocabulary=vocabulary,
        surfaces={"isotretinoin": "isotretinoin", "pain": "pain"},
        intent=intent,
    )


def test_hypernym_pain_excluded_from_manifestation():
    ctx = build_relation_context(_iso_plan())
    assert ctx is not None
    assert "pain" not in ctx.manifestation_terms
    assert "abdominal pain" not in ctx.manifestation_terms
    assert "back pain" not in ctx.manifestation_terms
    assert "joint pain" in ctx.manifestation_terms
    assert "arthralgia" in ctx.manifestation_terms
    assert "stiffness" in ctx.manifestation_terms


def test_abdominal_pain_report_dropped_back_pain_kept_out():
    item = _item("FAERS report", "Abdominal pain; Headache",
                 source="openfda_faers", source_type="pharmacovigilance",
                 treatment="isotretinoin", condition="abdominal pain")
    joint = _item("FAERS report", "Arthralgia",
                  source="openfda_faers", source_type="pharmacovigilance",
                  treatment="isotretinoin", condition="arthralgia")
    kept, stats = apply_relation_gate([item, joint], _iso_plan())
    assert kept == [joint]
    assert stats["dropped_manifestation"] == 1


def test_conceptual_mapping_entity_kept():
    """'shedding' → 'alopecia' is a real provider mapping (canonical !=
    surface): the hypernym guard must NOT drop it."""
    plan = SimpleNamespace(
        entities=[("drug", "finasteride", 0.95), ("symptom", "alopecia", 0.9)],
        vocabulary={},
        surfaces={"finasteride": "finasteride", "alopecia": "shedding"},
        intent=SimpleNamespace(interventions=["finasteride"],
                               outcomes=["hair shedding"],
                               synonyms={}),
    )
    ctx = build_relation_context(plan)
    assert ctx is not None
    assert "alopecia" in ctx.manifestation_terms
    assert "shedding" in ctx.manifestation_terms


# ── Gate stats ───────────────────────────────────────────────────────────────

def test_gate_stats_counts():
    items = [
        _item("Finasteride erectile dysfunction",
              "After taking finasteride I developed erectile dysfunction."),
        _item("Minoxidil only", "No drug here, just foam talk."),
        _item("Finasteride general", "I love finasteride, great for hair."),
    ]
    _kept, stats = apply_relation_gate(items, _plan())
    assert stats["input"] == 3
    assert stats["after_agent"] == 2
    assert stats["after_relation"] == 1
    assert stats["final"] == 1
    assert stats["dropped_agent"] == 1
    assert stats["dropped_manifestation"] == 1
