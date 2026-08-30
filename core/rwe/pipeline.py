"""
RWE search pipeline — autonomous search engine.

Flow (analogous to the scientific pipeline, kept fully separate):
    User Question
      → RWEQueryEngine (language detect + prepare + translate + expand)
      → RWE collectors (Reddit adapter, openFDA FAERS, ...) per expanded query
      → normalization (RawTestimonial/FAERS → RWEItem)
      → deduplication (URL / external_id, keep best matched_query)
      → semi-semantic relevance (authoritative source + token + entity + synonym)
      → provenance stamped (matched_query, source_language, match_reason, score)
      → RWE results

Independent of the scientific search — calling /rwe/search never touches
PubMed/Europe PMC/ClinicalTrials.gov and never mutates scientific state.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from core.orchestrator import QueryOrchestrator  # noqa: F401  (patched by tests via core.rwe.pipeline.QueryOrchestrator)
from core.rwe.intent import GENERIC_EVENT_TERMS, merged_sides
from core.rwe.models import RWEItem, RWESearchResult
from core.rwe.query_engine import RWEQueryEngine
from core.rwe.relation_filter import apply_relation_gate

logger = logging.getLogger(__name__)


# ─── Relevance scoring (semi-semantic) ───────────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "with",
    "my", "i", "is", "are", "was", "it", "this", "that", "from", "by",
    "di", "il", "la", "per", "che", "e", "un", "una", "del", "della",
    "can", "cause", "does", "may", "will", "you", "your",
}

# Sources whose own search is authoritative: when they return an item for a
# query, the match is trusted even if the query term is absent from the item's
# text fields (e.g. openFDA matched on a drug buried in the report's drug list).
_AUTHORITATIVE_SOURCES = {"openfda_faers"}

_TESTIMONIAL_CUES = {
    "i started", "since i started", "since starting", "after taking",
    "after applying", "after using", "when i started", "i got", "i had",
    "it gave me", "gave me", "made me", "caused me", "i noticed",
    "ho iniziato", "dopo aver", "da quando", "mi ha dato", "mi ha fatto",
    "mi è venuto", "mi sono trovato", "mi ha causato", "ho avuto",
    "mi è comparso", "j'ai", "après avoir", "me dio", "me causó",
}

_ADVERSE_RELATION_CUES = {
    "adverse", "side effect", "side effects", "hypertrichosis", "shedding",
    "hair loss", "alopecia", "rash", "edema", "oedema", "irritation",
    "pruritus", "headache", "dizziness", "tachycardia", "palpitations",
    "effluvio", "caduta", "perdita", "peggioramento", "dolore", "nausea",
}

_EFFICACY_RELATION_CUES = {
    "worked", "helped", "improved", "improvement", "effective",
    "effectiveness", "efficacy", "regrowth", "regrew", "better",
    "response", "treatment", "benefit", "beneficial", "stopped",
    "stabilized", "stabilised",
}

_COMPARISON_RELATION_CUES = {
    "versus", "comparison", "compare", "compared", "network meta-analysis",
    "head-to-head", "noninferiority", "randomized", "randomised",
}

_ROUTE_CUES = {
    "oral": {
        "oral", "orally", "pill", "pills", "tablet", "tablets", "capsule",
        "capsules", "per os", "per bocca", "compresse", "pastiglie",
    },
    "topical": {
        "topical", "topically", "foam", "lotion", "solution", "spray",
        "serum", "schiuma", "lozione", "soluzione", "fiala", "fiale",
    },
}

_TEMPORAL_AFTER_CUES = {
    "after", "since", "following", "after starting", "after taking",
    "after applying", "after using", "weeks later", "months later",
    "dopo", "da quando", "dopo aver", "après", "depuis", "después",
}


def _cue_hits(text: str, cues: set[str]) -> List[str]:
    lower = (text or "").lower()
    return sorted({cue for cue in cues if cue and cue in lower})[:6]


def _relation_bonus(
    body: str,
    source_type: str,
    relation_type: str,
    route: str = "unknown",
    temporal_relation: str = "unknown",
) -> Tuple[float, List[str]]:
    """Return a small additive bonus for modality-aware relation cues.

    Component maxima: testimonial 0.15 + relation cues 0.10 + route 0.05 +
    temporal 0.05 = 0.35, capped at 0.25 — so the maximum final bonus is
    exactly 0.25 by construction."""
    source_type = (source_type or "").lower().strip()
    relation_type = (relation_type or "").lower().strip()
    route = (route or "unknown").lower().strip()
    temporal_relation = (temporal_relation or "unknown").lower().strip()
    bonus = 0.0
    reasons: List[str] = []

    testimonial_hits = _cue_hits(body, _TESTIMONIAL_CUES)
    if source_type == "community_forum" and testimonial_hits:
        bonus += min(0.15, 0.05 * len(testimonial_hits))
        reasons.append(f"testimonial={testimonial_hits[:3]}")

    if relation_type in {"side_effect", "adverse_effect", "outcome"}:
        relation_hits = _cue_hits(body, _ADVERSE_RELATION_CUES)
    elif relation_type in {"treatment", "efficacy"}:
        relation_hits = _cue_hits(body, _EFFICACY_RELATION_CUES)
    elif relation_type == "comparison":
        relation_hits = _cue_hits(body, _COMPARISON_RELATION_CUES)
    else:
        relation_hits = []

    if relation_hits:
        bonus += min(0.10, 0.03 * len(relation_hits))
        reasons.append(f"relation={relation_hits[:3]}")

    # Route-of-administration cue: the query's route must be confirmed in the
    # record text (e.g. "after starting oral minoxidil").
    route_hits = _cue_hits(body, _ROUTE_CUES.get(route, set()))
    if route in _ROUTE_CUES and route_hits:
        bonus += 0.05
        reasons.append(f"route={route}:{route_hits[:2]}")

    # Temporal cue: exposure → outcome ordering expressed in the record.
    if temporal_relation == "after":
        temporal_hits = _cue_hits(body, _TEMPORAL_AFTER_CUES)
        if temporal_hits:
            bonus += 0.05
            reasons.append(f"temporal=after:{temporal_hits[:2]}")

    return round(min(0.25, bonus), 3), reasons


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-zà-öø-ÿ]+", (text or "").lower())
            if len(w) > 2 and w not in _STOPWORDS}


def _vocab_scored_terms(vocabulary: Optional[dict], canonical: str) -> List[str]:
    """Provider-typed variants of a canonical term from the slim vocabulary
    attached to the query plan (Catena C — no local dictionaries)."""
    if not vocabulary or not canonical:
        return []
    entries = vocabulary.get(canonical.lower())
    if not entries:
        return []
    from core.vocab.models import VocabularyResolution
    return list(VocabularyResolution.from_slim(canonical, entries).scored_terms())


def _entity_terms(entities, vocabulary: Optional[dict] = None) -> List[str]:
    """Flat list of lowercase canonical names + provider variants for matching."""
    terms: List[str] = []
    for _etype, canonical, _ in entities or []:
        terms.append(canonical.lower())
        terms.extend(_vocab_scored_terms(vocabulary, canonical))
    return terms


def _split_entities(entities, vocabulary: Optional[dict] = None) -> Tuple[List[str], List[str], List[str]]:
    """Split recognised entities into (drug_terms, event_terms, context_terms).

    event_terms = symptoms + conditions (the 'event' side of a drug→event query).
    drug_terms  = drugs + active ingredients (the 'exposure' side).
    context_terms = any other entity types (procedures, organs, …) — light context boost.
    Each list carries the canonical name plus its provider-typed variants.
    """
    drugs: List[str] = []
    events: List[str] = []
    context: List[str] = []
    for etype, canonical, _ in entities or []:
        canon = canonical.lower()
        if etype == "drug" or etype == "active_ingredient":
            drugs.append(canon)
            drugs.extend(_vocab_scored_terms(vocabulary, canon))
        elif etype in ("symptom", "adverse_effect", "disease", "condition"):
            events.append(canon)
            events.extend(_vocab_scored_terms(vocabulary, canon))
        else:
            context.append(canon)
    return drugs, events, context


def _event_match_terms(query_event_terms: List[str]) -> set:
    """Match terms for the query's event side.

    Semantic equivalents (synonyms, translations, colloquial phrasings) are
    already present in the side-set itself — injected upstream from the
    external vocabulary providers (Catena C). No local synonym groups."""
    return {t.lower() for t in query_event_terms if t}


def _event_match(item_text_lower: str, query_event_terms: List[str]) -> Tuple[float, List[str]]:
    """Score how well the item's event/reaction text matches the query's event.

    Returns (0..1, matched_terms). A 0 means the record's reactions/text do not
    express the event the query asks about (the "dutasteride + loss of
    proprioception" for a "dutasteride induced hair shedding" query case).
    """
    if not query_event_terms:
        # No event was recognised in the query — treat event_match as neutral.
        return 0.5, []
    terms = _event_match_terms(query_event_terms)
    hits = [t for t in terms if t and t in item_text_lower]
    if not hits:
        return 0.0, []
    # Saturate quickly: a couple of distinct event hits = strong event match.
    score = min(1.0, 0.5 + 0.25 * len(set(hits)))
    return score, sorted(set(hits))[:6]


def _drug_match(item_text_lower: str, treatment_lower: str, query_drug_terms: List[str]) -> Tuple[float, List[str]]:
    """Score how well the item references the query's drug/exposure.

    For authoritative sources (openFDA), the drug is trusted to be present
    server-side, so the treatment field is checked too.
    """
    if not query_drug_terms:
        return 0.0, []
    hay = f"{item_text_lower} {treatment_lower}"
    hits = [t for t in query_drug_terms if t and t in hay]
    if not hits:
        return 0.0, []
    score = min(1.0, 0.6 + 0.2 * len(set(hits)))
    return score, sorted(set(hits))[:4]


def _score_item(
    it: RWEItem,
    query: str,
    entities: list,
    is_authoritative_match: bool,
    vocabulary: Optional[dict] = None,
) -> Tuple[float, str]:
    """
    Semi-semantic relevance score (0.0–1.0) + human match reason.

    Separates the two sides of a drug→event query:
      - drug_match   : does the record reference the query's drug/exposure?
      - event_match  : does the record's reaction/text express the query's
                       event (semantically, incl. IT/FR/ES/DE equivalents)?

    overall_relevance = weighted blend; drug-only matches (no event) score
    LOW so a dutasteride+proprioception record does not surface for a
    "dutasteride induced hair shedding" query.

    For authoritative sources (openFDA), the drug is trusted as present
    (server-side match), but the event is STILL verified against the record's
    reaction text — authoritative does not override the event requirement.
    """
    q_tokens = _tokens(query)
    body = f"{it.title} {it.text}".lower()
    treatment_lower = (it.treatment or "").lower()
    body_tokens = _tokens(f"{it.title} {it.text} {it.treatment or ''} {it.condition or ''}")
    overlap = q_tokens & body_tokens
    token_score = len(overlap) / max(1, len(q_tokens)) if q_tokens else 0.0

    drug_terms, event_terms, context_terms = _split_entities(entities, vocabulary)

    drug_score, drug_hits = _drug_match(body, treatment_lower, drug_terms)
    event_score, event_hits = _event_match(body, event_terms)

    # Context boost (light) — population/condition co-occurrence.
    ctx_hits = [t for t in context_terms if t and t in body]
    ctx_score = min(0.15, 0.05 * len(ctx_hits)) if context_terms else 0.0

    # ── overall_relevance ──────────────────────────────────────────────────
    # When the query has BOTH a drug and an event, both must be present for a
    # high score. A drug-only match (event_score=0) is heavily penalised.
    has_drug_and_event = bool(drug_terms) and bool(event_terms)

    if is_authoritative_match:
        # openFDA: trust the drug match (server-side), but still require the event.
        drug_component = 1.0   # authoritative drug match is trusted present
    else:
        drug_component = drug_score

    if has_drug_and_event:
        if event_score == 0.0:
            # The record does NOT express the queried event → low relevance.
            base = 0.15 * drug_component + 0.0 + ctx_score
            reason = (f"drug_match_only (drug={drug_hits[:2]}; "
                      f"event_missing — queried event not in record)")
            return round(min(1.0, base), 3), reason
        # Both sides present — strong relevance.
        base = 0.45 * drug_component + 0.45 * event_score + 0.10 * token_score + ctx_score
        reason = (f"drug+event_match (drug={drug_hits[:2]}; "
                  f"event={event_hits[:3]}; tokens={len(overlap)})")
        return round(min(1.0, base), 3), reason

    # Query has only a drug, or only an event, or neither recognised.
    if drug_terms and not event_terms:
        # Drug-only query: drug match is the signal.
        base = 0.6 * drug_component + 0.3 * token_score + ctx_score
        reason = f"drug_match (drug={drug_hits[:2]}; tokens={len(overlap)})"
        return round(min(1.0, base), 3), reason
    if event_terms and not drug_terms:
        base = 0.6 * event_score + 0.3 * token_score + ctx_score
        reason = f"event_match (event={event_hits[:3]}; tokens={len(overlap)})"
        return round(min(1.0, base), 3), reason

    # No structured entities — fall back to token/entity overlap.
    ent_terms = _entity_terms(entities, vocabulary)
    ent_hits = [t for t in ent_terms if t and t in body]
    entity_score = min(1.0, len(ent_hits) / 3.0) if ent_terms else 0.0
    if token_score >= 0.5 and entity_score > 0:
        base = max(0.5 * token_score + 0.5 * entity_score, 0.75)
        reason = f"exact_keyword+semantic ({len(overlap)} tokens, {len(ent_hits)} entities)"
    elif token_score > 0:
        base = 0.5 * token_score + 0.5 * entity_score
        reason = f"exact_keyword ({len(overlap)}/{len(q_tokens)} tokens: {sorted(overlap)})"
    elif entity_score > 0:
        base = max(0.5 * token_score + 0.5 * entity_score, 0.4)
        reason = f"semantic_entity_match ({len(ent_hits)} entities: {ent_hits[:3]})"
    else:
        base = 0.0
        reason = "no keyword or entity overlap"
    return round(min(1.0, base), 3), reason


# Generic "any adverse event" outcome terms (e.g. "finasteride side effects"):
# the user asks about ANY event, so the event side is satisfied by an open
# event-set rather than by one literal phrase.
# Canonical definition lives in core.rwe.intent (shared with merged_sides,
# which must not inject vocabulary synonyms into generic-event side-sets).
_GENERIC_EVENT_TERMS = GENERIC_EVENT_TERMS


def _v3_event_terms(event_side: set) -> set:
    """Match terms for the event side: the side-set itself, already enriched
    upstream with provider-typed variants (Catena C)."""
    return {t for t in event_side if t}


def _v3_event_score(body: str, condition_field: str, event_side: set,
                    is_authoritative_match: bool,
                    tiers: Optional[Dict[str, float]] = None,
                    generic_vocab: Optional[set] = None,
                    ) -> Tuple[float, List[str]]:
    """Event-side score for V3, including the generic-any-event case.

    ``tiers`` (optional, from external vocabulary evidence) weights each
    matched term: terms absent from the map weight 1.0, so without
    vocabulary evidence the score is byte-identical to the pre-vocabulary
    behaviour. ``generic_vocab`` is the provider-derived open event-set
    (generic canonical's provider terms + the anchor drug's related
    concepts) used ONLY for generic any-event queries."""
    def _score(hits: List[str]) -> float:
        if tiers:
            w = sum(tiers.get(h, 1.0) for h in hits)
        else:
            w = float(len(hits))
        return min(1.0, 0.5 + 0.25 * w)

    if not event_side:
        return 0.5, []
    if event_side <= _GENERIC_EVENT_TERMS:
        # "Any adverse event" query: authoritative reports ARE adverse-event
        # reports by definition; other sources must mention a provider-known
        # event term of the queried exposure.
        if is_authoritative_match:
            return 1.0, ["(any adverse event — authoritative report)"]
        vocab = set(_GENERIC_EVENT_TERMS) | set(generic_vocab or set())
        hits = sorted({t for t in vocab if t and t in body})[:6]
        return (_score(hits), hits) if hits else (0.0, [])
    terms = _v3_event_terms(event_side)
    hits = sorted({t for t in terms if t and t in body})[:6]
    if not hits and condition_field:
        hits = sorted({t for t in terms if t and t in condition_field})[:6]
    if not hits:
        # Word-order/inflection-tolerant fallback: a multi-token event phrase
        # counts as a WEAK hit when ALL its non-generic tokens appear in the
        # body ("sexual dysfunction" expressed as "sexual ... dysfunction").
        # Single tokens and generic any-event phrases never qualify. The
        # agent↔event LINK is verified downstream by the relation gate.
        body_tokens = _tokens(body)
        generic_tokens = {
            tok for term in _GENERIC_EVENT_TERMS for tok in _tokens(term)}
        for t in sorted(terms):
            base_tokens = [tok for tok in _tokens(t.split(",")[0])
                           if tok not in generic_tokens]
            if len(base_tokens) >= 2 and all(tok in body_tokens
                                             for tok in base_tokens):
                hits = [f"{' '.join(base_tokens)} (tokens)"]
                break
    if not hits:
        return 0.0, []
    return _score(hits), hits


def _score_item_v3(
    it: RWEItem,
    query: str,
    sides: Dict[str, set],
    is_authoritative_match: bool,
    intent=None,
) -> Tuple[float, str]:
    """
    QU-aware relevance score (V3). Same contract as ``_score_item`` (V1):
    returns (0–1 score, human match reason) and never mutates the item.

    Modality-aware rules for RWE:
    - the query sides come from the structured intent (interventions / outcomes /
      conditions + QU synonyms), UNIONED with KB entities;
    - community/forum testimonies are rewarded when they express a direct
      relation between the exposure and the experience;
    - openFDA keeps the V1 rule: drug trusted server-side, event verified.
    """
    q_tokens = _tokens(query)
    body = f"{it.title} {it.text}".lower()
    treatment_lower = (it.treatment or "").lower()
    condition_lower = (it.condition or "").lower()
    source_type = (getattr(it, "source_type", "") or "").lower()
    relation_type = str(getattr(intent, "relation_type", "") or "").lower()
    route = str(getattr(intent, "route", "") or "unknown").lower()
    temporal_relation = str(
        getattr(intent, "temporal_relation", "") or "unknown").lower()
    body_tokens = _tokens(
        f"{it.title} {it.text} {treatment_lower} {condition_lower}"
    )
    overlap = q_tokens & body_tokens
    token_score = len(overlap) / max(1, len(q_tokens)) if q_tokens else 0.0

    iv, oc, cd = sides["iv"], sides["oc"], sides["cd"]
    tiers = sides.get("tiers") or {}

    # ── side resolution (avoids the degenerate cd-anchor case) ──────────────
    if iv:
        anchor_terms = iv
        event_side = oc | cd
    else:
        anchor_terms = cd
        event_side = oc if oc else cd
    anchor_side = anchor_terms

    # ── anchor side (intervention or condition) ─────────────────────────────
    if is_authoritative_match and iv:
        anchor_score, anchor_hits = 1.0, ["(authoritative)"]
    else:
        hay = f"{body} {treatment_lower}"
        anchor_hits = sorted({t for t in anchor_terms if t and t in hay})
        anchor_w = sum(tiers.get(h, 1.0) for h in anchor_hits)
        anchor_score = min(1.0, 0.6 + 0.2 * anchor_w) if anchor_hits else 0.0
        if cd and not iv:
            sem_score, sem_hits = _v3_event_score(body, condition_lower, cd,
                                                  False, tiers)
            if sem_score > anchor_score:
                anchor_score, anchor_hits = sem_score, sem_hits

    # ── event side (QU outcomes/conditions + provider-typed variants) ───────
    event_score, event_hits = _v3_event_score(
        body, condition_lower, event_side, is_authoritative_match, tiers,
        generic_vocab=sides.get("generic_vocab"))

    # ── modality-aware RWE bonus: direct testimonies and relation cues ──────
    relation_bonus = 0.0
    relation_reasons: List[str] = []
    if body and (event_score > 0.0 or anchor_score > 0.0 or token_score > 0.0):
        relation_bonus, relation_reasons = _relation_bonus(
            body, source_type, relation_type,
            route=route, temporal_relation=temporal_relation)

    # ── structured-field boost (small, bounded) ─────────────────────────────
    boost = 0.0
    if treatment_lower and any(t in treatment_lower for t in anchor_side):
        boost += 0.05
    if condition_lower and any(t in condition_lower for t in event_side):
        boost += 0.05

    # ── combine ──────────────────────────────────────────────────────────────
    if anchor_side and event_side:
        if event_score == 0.0:
            base = 0.15 * anchor_score + 0.05 * token_score
            reason = (f"v3 anchor_match_only (anchor={anchor_hits[:2]}; "
                      f"event_missing)")
        elif anchor_score == 0.0:
            base = 0.15 * event_score + 0.05 * token_score
            reason = (f"v3 event_match_only (event={event_hits[:3]}; "
                      f"anchor_missing)")
        else:
            base = (0.40 * anchor_score + 0.40 * event_score
                    + 0.10 * token_score + boost)
            reason = (f"v3 anchor+event (anchor={anchor_hits[:2]}; "
                      f"event={event_hits[:3]}; tokens={len(overlap)})")
    elif anchor_side:
        base = 0.55 * anchor_score + 0.25 * token_score + boost
        reason = f"v3 anchor_only (anchor={anchor_hits[:2]}; tokens={len(overlap)})"
    elif event_side:
        base = 0.55 * event_score + 0.30 * token_score + boost
        reason = f"v3 event_only (event={event_hits[:3]})"
    else:
        base = 0.8 * token_score
        reason = "v3 token_fallback"

    if relation_bonus > 0:
        base = min(1.0, base + relation_bonus)
        if relation_reasons:
            reason = f"{reason}; modality=rwe; cues={'; '.join(relation_reasons[:2])}"
        else:
            reason = f"{reason}; modality=rwe"

    return round(min(1.0, base), 3), reason


def relevance_filter(
    items: List[RWEItem],
    query: str,
    entities: Optional[list] = None,
    authoritative_sources: Optional[set] = None,
    min_score: float = 0.20,
    intent=None,
    vocabulary: Optional[dict] = None,
) -> List[RWEItem]:
    """
    Semi-semantic relevance filter — respects the FULL intent of the query.

    Splits the query into a DRUG/exposure side and an EVENT/symptom side, and
    requires BOTH for a high score. A record that merely references the queried
    drug but expresses an unrelated event (e.g. dutasteride + loss of
    proprioception for a "dutasteride induced hair shedding" query) scores LOW
    and is filtered out.

    Authoritative sources (openFDA): the drug match is trusted server-side,
    but the event is STILL verified against the record's reaction text —
    authoritative does not override the event requirement.

    Marks each item with ``relevance`` (relevant|irrelevant), ``relevance_reason``,
    ``relevance_score`` (0–1), and ``match_reason``. Keeps only relevant items
    (score ≥ min_score). Default min_score=0.20 filters drug-only false positives.

    Backward compatible: ``entities`` and ``authoritative_sources`` default to
    None/empty so the call ``relevance_filter(items, query)`` still works.

    QU-aware mode (V3): when ``intent`` is a valid ``RWEQueryIntent`` (built by
    the query engine only under the HLEO_RWE_INTENT_SCORING feature flag), the
    scoring uses the structured intent sides via ``_score_item_v3``. When
    ``intent`` is None the behaviour is EXACTLY the V1 behaviour above.
    """
    q_tokens = _tokens(query)
    auth_sources = authoritative_sources or set()
    entities = entities or []

    if not q_tokens and not entities:
        for it in items:
            it.relevance = "unknown"
            it.relevance_reason = "No query terms or entities available."
            it.relevance_score = 0.0
            it.match_reason = "no_signal"
        return items

    # V3 side sets are precomputed once per call (deterministic, no LLM here).
    v3_sides = merged_sides(intent, entities) if intent is not None else None

    kept: List[RWEItem] = []
    for it in items:
        # Authoritative = the source's own search is trusted (hardcoded set),
        # optionally restricted by the caller via authoritative_sources.
        is_auth = it.source in _AUTHORITATIVE_SOURCES and (
            not auth_sources or it.source in auth_sources
        )
        if v3_sides is not None:
            score, reason = _score_item_v3(it, query, v3_sides, is_auth, intent)
        else:
            score, reason = _score_item(it, query, entities, is_auth,
                                        vocabulary=vocabulary)
        it.relevance_score = score
        it.match_reason = reason
        if intent is not None:
            it.metadata = dict(it.metadata or {})
            it.metadata["intent_relation_type"] = getattr(
                intent, "relation_type", None)
            it.metadata["intent_route"] = getattr(intent, "route", None)
            it.metadata["intent_temporal_relation"] = getattr(
                intent, "temporal_relation", None)
        if score >= min_score:
            it.relevance = "relevant"
            it.relevance_reason = reason
            kept.append(it)
        else:
            it.relevance = "irrelevant"
            it.relevance_reason = reason
    return kept


# ─── Deduplication ──────────────────────────────────────────────────────────

def deduplicate(items: List[RWEItem]) -> List[RWEItem]:
    """Union duplicate records while retaining every query provenance."""
    best: dict = {}
    order: List = []
    for it in items:
        key = (it.source, it.external_id or it.source_url or it.title)
        prov = {
            "matched_query": it.matched_query,
            "matched_query_type": it.matched_query_type,
            "source_language": it.source_language,
            "original_term": (it.metadata or {}).get("original_term"),
            "expanded_term": (it.metadata or {}).get("expanded_term"),
            "match_kind": (it.metadata or {}).get("match_kind"),
            "tier": (it.metadata or {}).get("tier"),
            "provider": (it.metadata or {}).get("provider"),
            "source_entity": (it.metadata or {}).get("source_entity"),
            "query_origin": (it.metadata or {}).get("query_origin"),
        }
        if key not in best:
            best[key] = it
            order.append(key)
            it.metadata = dict(it.metadata or {})
            it.metadata["matched_queries"] = [it.matched_query] if it.matched_query else []
            it.metadata["match_provenance"] = [prov]
            continue
        winner = best[key]
        winner.metadata = dict(winner.metadata or {})
        queries = winner.metadata.setdefault("matched_queries", [])
        if it.matched_query and it.matched_query not in queries:
            queries.append(it.matched_query)
        provenance = winner.metadata.setdefault("match_provenance", [])
        if prov not in provenance:
            provenance.append(prov)
        if (it.relevance_score or 0.0) > (winner.relevance_score or 0.0):
            it.metadata = winner.metadata
            best[key] = it
    return [best[k] for k in order]


# ─── Pipeline ───────────────────────────────────────────────────────────────


class RWEPipeline:
    """
    Orchestrates RWE collectors behind an autonomous query engine.

    Independent from the scientific HLEOPipeline.

    Usage:
        pipe = RWEPipeline()
        result = pipe.search("La finasteride può causare shedding iniziale?")
    """

    def __init__(self) -> None:
        from core.rwe.reddit_adapter import RedditRWEAdapter
        from core.rwe.openfda_collector import OpenFDACollector
        from core.rwe.calvizie_collector import CalvizieCollector
        from core.rwe.hairlosstalk_collector import HairLossTalkCollector
        from core.rwe.hairlossexperiences_collector import HairLossExperiencesCollector
        from core.rwe.maladiesrares_collector import MaladiesRaresCollector
        self.reddit = RedditRWEAdapter()
        self.openfda = OpenFDACollector()
        self.calvizie = CalvizieCollector()
        self.hairlosstalk = HairLossTalkCollector()
        self.hairlossexperiences = HairLossExperiencesCollector()
        self.maladiesrares = MaladiesRaresCollector()
        self._engine = RWEQueryEngine()

    def search(
        self,
        query: str,
        limit: int = 15,
        sources: Optional[List[str]] = None,
    ) -> RWESearchResult:
        """
        Run the RWE search pipeline.

        Args:
            query: user query (any language).
            limit: per-source item cap per expanded query.
            sources: restrict to a subset; None = all usable sources.
        """
        # ── 1. Build the query plan (detect → prepare → translate → expand) ──
        plan = self._engine.plan(query)

        # Resolve effective sources: if caller provided an explicit list, prefer it;
        # otherwise read active RWE sources from SourceRegistry. Fall back to legacy
        # default list if no registry rows are present.
        from core.database import SessionLocal
        from core.models import SourceRegistry
        from sqlalchemy import select

        registry_map: Dict[str, SourceRegistry] = {}
        effective_sources: List[str] = []

        if sources:
            # normalize provided list and try to resolve registry rows for them
            requested = [s.strip() for s in sources if s and s.strip()]
            if requested:
                with SessionLocal() as db:
                    rows = db.execute(select(SourceRegistry).where(SourceRegistry.source_id.in_(requested))).scalars().all()
                rows_by_id = {r.source_id: r for r in rows}
                for s in requested:
                    if s in rows_by_id:
                        registry_map[s] = rows_by_id[s]
                        effective_sources.append(s)
                    else:
                        # allow direct collector keys (legacy) to be passed through
                        effective_sources.append(s)
        else:
            # discover active RWE sources from the registry
            with SessionLocal() as db:
                rows = db.execute(
                    select(SourceRegistry).where(
                        SourceRegistry.category == "rwe_experience",
                        SourceRegistry.status == "active",
                    )
                ).scalars().all()
            if rows:
                for r in rows:
                    registry_map[r.source_id] = r
                    effective_sources.append(r.source_id)
            else:
                effective_sources = [
                    "reddit", "openfda_faers", "calvizie",
                    "hairlosstalk", "hairlossexperiences", "maladiesrares",
                ]

        per_source_limits = {s: limit for s in effective_sources}
        # ensure default caps for legacy keys if present
        per_source_limits.update({
            "reddit": limit,
            "openfda_faers": limit,
            "calvizie": limit,
            "hairlosstalk": limit,
            "hairlossexperiences": limit,
            "maladiesrares": limit,
        })

        all_items: List[RWEItem] = []
        source_status: dict = {}

        # Collector instance map for known collectors
        collector_map = {
            "reddit": self.reddit,
            "openfda_faers": self.openfda,
            "calvizie": self.calvizie,
            "hairlosstalk": self.hairlosstalk,
            "hairlossexperiences": self.hairlossexperiences,
            "maladiesrares": self.maladiesrares,
        }

        for src in effective_sources:
            row = registry_map.get(src)
            collector_key = (row.runtime_collector if row and row.runtime_collector else src)
            collector = collector_map.get(collector_key)
            # If this is a generic REST-configured source, instantiate GenericRESTCollector
            if collector_key == "generic_rest" and row:
                try:
                    from collectors.generic_rest import GenericRESTCollector
                    collector = GenericRESTCollector(row.connection_spec or {}, source_id=row.source_id, category=row.category)
                except Exception as exc:
                    logger.exception("Failed to instantiate GenericRESTCollector for %s: %s", src, exc)
                    source_status[src] = "no_collector"
                    continue

            if not collector:
                source_status[src] = "no_collector"
                continue

            try:
                items, status = self._collect_source(src, collector, plan, per_source_limits.get(src, limit))
                source_status[src] = status
                all_items.extend(items)
            except Exception as exc:
                logger.exception("RWE collector failed for %s: %s", src, exc)
                source_status[src] = "network_error"
                # continue with other sources
                continue

        # ── 2. Deduplicate across (query × source), keep best matched_query ──
        before = len(all_items)
        all_items = deduplicate(all_items)
        after = len(all_items)

        # ── 3. Semi-semantic relevance filtering ────────────────────────────
        # intent is None unless the V3 feature flag built it → V1 behaviour.
        relevance_query = plan.translated_query or plan.original_query
        all_items = relevance_filter(
            all_items,
            relevance_query,
            entities=plan.entities,
            intent=getattr(plan, "intent", None),
            vocabulary=getattr(plan, "vocabulary", None),
            min_score=0.20,
        )
        relevant_after_scoring = len(all_items)
        # ── 3b. Relation-precision gate (Levels A/B/C, RWE-only) ────────────
        # Active only for structured agent→manifestation queries; otherwise it
        # is a no-op passthrough. Drops off-agent / off-manifestation /
        # relation-unverified items and re-ranks the survivors.
        all_items, gate_stats = apply_relation_gate(all_items, plan)
        all_items.sort(key=lambda it: it.relevance_score, reverse=True)
        relevant_before_cap = len(all_items)
        all_items = all_items[:400]

        totals = {
            "retrieved": before,
            "deduped_removed": before - after,
            "unique": after,
            "relevant": relevant_after_scoring,
            "final": len(all_items),
            "score_threshold": 0.20,
            "max_results": 400,
            "queries_used": len(plan.expanded_queries),
        }
        if gate_stats is not None:
            totals["precision_filter"] = gate_stats

        return RWESearchResult(
            query=query,
            original_query=plan.original_query,
            search_query=relevance_query,
            canonical_query=plan.canonical_query,
            detected_language=plan.detected_language,
            translated_query=plan.translated_query,
            translation_applied=plan.translation_applied,
            expanded_queries=plan.to_dict()["expanded_queries"],
            totals=totals,
            items=all_items,
            source_status=source_status,
        )

    # ── Source collection ─────────────────────────────────────────────────────

    def _collect_source(
        self,
        name: str,
        collector,
        plan,
        cap: int,
    ) -> Tuple[List[RWEItem], str]:
        """
        Run a collector across every expanded query, stamping each item with the
        matched_query provenance. Stops early once ``cap`` items are collected.

        Returns (items, aggregate_status). The aggregate status is the best
        status seen across the expanded queries.
        """
        collected: List[RWEItem] = []
        statuses: List[str] = []

        # Forum-feed collectors (XenForo RSS / phpBB Atom) are not full-text
        # searchable: every query re-fetches the same feed. Send them only the
        # primary queries (original/translated/canonical); full-text APIs
        # (openFDA, Reddit) receive the complete expansion set.
        from core.rwe.xenforo_base import XenForoRSSCollector
        from core.rwe.maladiesrares_collector import MaladiesRaresCollector
        feed_like = isinstance(collector, (XenForoRSSCollector, MaladiesRaresCollector))
        if feed_like:
            primary_types = {"original", "translated", "canonical"}
            queries = [eq for eq in plan.expanded_queries
                       if eq.expansion_type in primary_types] or plan.expanded_queries[:1]
        else:
            # Server-side search APIs (openFDA, Reddit) execute one request
            # per query; bound the number of distinct queries sent to keep
            # latency and rate limits healthy while still using the
            # vocabulary expansions. Original/translated/canonical rank first
            # in the plan ordering.
            queries = plan.expanded_queries[:6]

        for eq in queries:
            try:
                items, status, reason = collector.search_with_status(
                    eq.query, limit=None
                )
            except Exception as exc:
                logger.warning(f"RWE collector {name} failed for '{eq.query}': {exc}")
                statuses.append("network_error")
                continue
            statuses.append(status)
            if status != "ok":
                continue
            for it in items:
                it.matched_query = eq.query
                it.matched_query_type = eq.expansion_type
                it.source_language = eq.source_language
                it.topic = eq.query
                it.metadata = dict(it.metadata or {})
                it.metadata.update({
                    "original_term": eq.original_term,
                    "expanded_term": eq.expanded_term,
                    "match_kind": eq.match_kind,
                    "tier": eq.tier,
                    "provider": eq.provider,
                    "source_entity": eq.source_entity,
                    "query_origin": eq.query_origin,
                })
                collected.append(it)

        # Aggregate: prefer ok > no_results > no_credentials > error
        status_rank = {
            "ok": 0, "no_results": 1, "no_credentials": 2,
            "auth_error": 3, "unsupported_query": 4, "rate_limited": 5,
            "network_error": 6,
        }
        agg = "no_results"
        if statuses:
            agg = min(statuses, key=lambda s: status_rank.get(s, 9))
        return collected, agg
