"""
RWE Query Intent — structured QU (Query Understanding) extraction for the
QU-aware relevance scorer (V3).

Design constraints (agreed with the project owner):
- V3 is FEATURE-FLAGGED (``HLEO_RWE_INTENT_SCORING``). When the flag is off,
  no intent is built, no LLM call is made, and ``relevance_filter`` behaves
  exactly like V1.
- AT MOST ONE LLM call per query (never per document). The LLM only
  structures the query; it never scores items.
- The LLM output is validated as strict JSON via Pydantic. On any failure
  (no API key, network error, invalid JSON, schema mismatch) the fallback is
  a deterministic intent rebuilt from the KB entities — i.e. V1 behaviour.
- ``relation_type`` is extracted for explainability only; it is NEVER used
  as a ranking criterion.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

INTENT_SCORING_ENV = "HLEO_RWE_INTENT_SCORING"
INTENT_MODEL = "gpt-4o-mini"

_ALLOWED_RELATIONS = {"side_effect", "treatment", "outcome", "comparison", "unknown"}
_ALLOWED_ROUTES = {"oral", "topical", "unknown"}
_ALLOWED_TEMPORAL = {"after", "during", "before", "unknown"}

# Lexical cues for deterministic route / temporal inference (no LLM needed —
# these are surface-level patient phrasings).
_ROUTE_ORAL_CUES = {
    "oral", "orally", "pill", "pills", "tablet", "tablets", "capsule",
    "capsules", "per os", "per bocca", "compresse", "pastiglie", "pillola",
}
_ROUTE_TOPICAL_CUES = {
    "topical", "topically", "foam", "lotion", "solution", "spray", "serum",
    "schiuma", "lozione", "soluzione", "fiala", "fiale",
}
_TEMPORAL_AFTER_CUES = {
    "after", "since", "following", "post", "dopo", "da quando", "dopo aver",
    "après", "depuis", "después", "nach", "seit",
}
_TEMPORAL_BEFORE_CUES = {"before", "prior", "prima", "avant", "antes", "vorher"}
_TEMPORAL_DURING_CUES = {
    "during", "while", "durante", "mentre", "pendant", "mientras", "während",
}


def infer_route(*texts: str) -> str:
    """Deterministic route-of-administration inference from query text.
    Returns "oral" | "topical" | "unknown"."""
    lower = " ".join(t for t in texts if t).lower()
    if any(cue in lower for cue in _ROUTE_ORAL_CUES):
        return "oral"
    if any(cue in lower for cue in _ROUTE_TOPICAL_CUES):
        return "topical"
    return "unknown"


def infer_temporal_relation(*texts: str) -> str:
    """Deterministic temporal-relation inference from query text.
    Returns "after" | "before" | "during" | "unknown"."""
    lower = " ".join(t for t in texts if t).lower()
    if any(cue in lower for cue in _TEMPORAL_AFTER_CUES):
        return "after"
    if any(cue in lower for cue in _TEMPORAL_DURING_CUES):
        return "during"
    if any(cue in lower for cue in _TEMPORAL_BEFORE_CUES):
        return "before"
    return "unknown"

# Generic "any adverse event" terms (e.g. "finasteride side effects"): the
# user asks about ANY event. Such canonicals keep their generic semantics in
# scoring — external vocabulary synonyms must NOT be injected into their
# side-set (they stay as evidence in intent.vocabulary only), otherwise the
# generic-any-event branch of the scorer would be silently disabled.
GENERIC_EVENT_TERMS = {
    "side effect", "side effects", "adverse effect", "adverse effects",
    "adverse event", "adverse events", "effetti collaterali",
    "effetto collaterale", "reaction", "reactions",
}
_MAX_TERMS_PER_SIDE = 12
_MAX_SYNONYMS = 8
_MAX_TERM_LEN = 40


def intent_scoring_enabled() -> bool:
    """Feature flag — read at call time so tests can monkeypatch the env."""
    return os.getenv(INTENT_SCORING_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


class RWEQueryIntent(BaseModel):
    """Structured representation of the query intent for V3 scoring.

    ``interventions`` = drugs/exposures, ``outcomes`` = events/symptoms the
    user asks about, ``conditions`` = diseases/conditions. ``synonyms`` maps
    a canonical term (present in one of the three lists) to extra aliases.
    ``vocabulary`` (optional, feature-flagged) maps each canonical term to
    slim external-vocabulary evidence (provider, concept_id, match_kind,
    confidence, synonyms) — evidence, never auto-synonyms.
    """
    interventions: List[str] = Field(default_factory=list)
    outcomes: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    synonyms: Dict[str, List[str]] = Field(default_factory=dict)
    relation_type: Optional[str] = None   # explainability only, never ranked
    route: Optional[str] = None           # "oral" | "topical" | "unknown" | None
    temporal_relation: Optional[str] = None  # "after" | "during" | "before" | "unknown" | None
    source: str = "kb_fallback"           # "llm" | "kb_fallback"
    confidence: float = 0.0
    vocabulary: Dict[str, List[dict]] = Field(default_factory=dict)

    @field_validator("relation_type")
    @classmethod
    def _check_relation(cls, v):
        if v is None:
            return None
        v = str(v).lower().strip()
        return v if v in _ALLOWED_RELATIONS else "unknown"

    @field_validator("route")
    @classmethod
    def _check_route(cls, v):
        if v is None:
            return None
        v = str(v).lower().strip()
        return v if v in _ALLOWED_ROUTES else "unknown"

    @field_validator("temporal_relation")
    @classmethod
    def _check_temporal(cls, v):
        if v is None:
            return None
        v = str(v).lower().strip()
        return v if v in _ALLOWED_TEMPORAL else "unknown"


def _clean_term(t) -> Optional[str]:
    if not isinstance(t, str):
        return None
    t = t.lower().strip()
    if len(t) < 3 or len(t) > _MAX_TERM_LEN:
        return None
    return t


def _clean_terms(values, cap: int = _MAX_TERMS_PER_SIDE) -> List[str]:
    out: List[str] = []
    for v in values or []:
        t = _clean_term(v)
        if t and t not in out:
            out.append(t)
        if len(out) >= cap:
            break
    return out


_INTENT_PROMPT = """You are a biomedical query-understanding component for a
hair-loss real-world-evidence search engine. Analyse the user query and
return ONLY a JSON object with these keys:

- "interventions": list of drugs / treatments / active ingredients the query
  is about (canonical generic names, e.g. "finasteride"). Empty list if none.
- "outcomes": list of events / symptoms / side effects the user asks about
  (e.g. "hair shedding", "depression", "sexual dysfunction"). Empty list if
  the query does not ask about any event.
- "conditions": list of diseases / conditions mentioned (e.g. "hair loss",
  "androgenetic alopecia"). Empty list if none.
- "synonyms": object mapping each canonical term above to a list of up to 8
  common aliases / brand names / patient phrasings, INCLUDING colloquial
  forms patients actually use in forums (e.g. "finasteride":
  ["propecia", "proscar"]; "sexual dysfunction": ["erectile dysfunction",
  "impotence", "low libido", "loss of libido", "ejaculation problems"]).
- "relation_type": one of "side_effect", "treatment", "outcome",
  "comparison", "unknown".

Rules:
- Do NOT invent an outcome for a query that only names a drug or only names
  a condition (e.g. "propecia" has no outcome; "hair loss" is a condition,
  not an outcome).
- A query that EXPLICITLY names an outcome concept — even a generic one —
  DOES have an outcome: "finasteride side effects" → outcomes:
  ["side effects"]; "finasteride shedding" → outcomes: ["hair shedding"].
- Keep the user's clinical intent: a colloquial phrasing like "my hair is
  falling out since I started X" means outcome "hair shedding" / "hair loss".
- Answer in English canonical terms even if the query is in another language.
- JSON only, no markdown, no commentary.

Query (English rendering): {translated}
Original user query: {original}"""


def extract_intent_llm(
    translated_query: str,
    original_query: str = "",
    client=None,
    model: str = INTENT_MODEL,
) -> Optional[RWEQueryIntent]:
    """ONE LLM call to structure the query. Returns None on any failure.

    The optional ``client`` parameter exists for testability (a fake client
    can be injected); production code creates a real OpenAI client lazily.
    """
    if not (translated_query or original_query):
        return None
    # Ordered (raw_client, model, provider_label) attempts: the configured
    # provider plus its one-way fallback (perplexity → openai). Each endpoint
    # gets exactly ONE call here (the existing no-retry policy of this
    # function); any failure moves to the next candidate, then to the
    # deterministic entity fallback. Never a loop.
    attempts: list = []
    if client is not None:
        attempts.append((client, model, "sdk"))
    else:
        from core.llm_provider import build_provider, resolve_model
        provider = build_provider()
        if provider is None:
            return None
        attempts.append((provider.client, resolve_model(provider.name, model), provider.name))
        if provider.fallback is not None:
            fb = provider.fallback
            attempts.append((fb.client, resolve_model(fb.name, model), fb.name))
    for raw_client, resolved_model, label in attempts:
        try:
            kwargs: dict = {
                "model": resolved_model,
                "messages": [{
                    "role": "user",
                    "content": _INTENT_PROMPT.format(
                        translated=translated_query or original_query,
                        original=original_query or translated_query,
                    ),
                }],
                "temperature": 0,
            }
            # Perplexity Sonar rejects OpenAI-style json_object; the prompt
            # already demands a bare JSON object.
            if label != "perplexity":
                kwargs["response_format"] = {"type": "json_object"}
            resp = raw_client.chat.completions.create(**kwargs)
            raw = json.loads(resp.choices[0].message.content)
            if not isinstance(raw, dict):
                continue
            intent = RWEQueryIntent(
                interventions=_clean_terms(raw.get("interventions")),
                outcomes=_clean_terms(raw.get("outcomes")),
                conditions=_clean_terms(raw.get("conditions")),
                synonyms={
                    c: _clean_terms(a, _MAX_SYNONYMS)
                    for c, a in (raw.get("synonyms") or {}).items()
                    if _clean_term(c)
                },
                relation_type=raw.get("relation_type"),
            )
            if not (intent.interventions or intent.outcomes or intent.conditions):
                return None  # empty extraction is not a usable signal
            intent.source = "llm"
            intent.confidence = 1.0
            return intent
        except Exception as exc:  # noqa: BLE001 — try next candidate, then KB
            logger.info("QU intent extraction failed (%s, provider=%s)",
                        type(exc).__name__, label)
            continue
    logger.info("QU intent extraction failed on all providers; falling back to KB")
    return None


def intent_from_entities(entities: list, *query_texts: str) -> RWEQueryIntent:
    """Deterministic fallback: rebuild the three intent sides from the
    provider-recognised entities of the query plan (no LLM call).

    When the query exposes a drug→event structure (an intervention AND an
    outcome), ``relation_type`` is set to "side_effect" so the modality-aware
    relation cues can be scored. Route and temporal cues are inferred
    deterministically from the query text when provided."""
    iv: List[str] = []
    oc: List[str] = []
    cd: List[str] = []
    for etype, canonical, _conf in entities or []:
        c = canonical.lower()
        if etype in ("drug", "active_ingredient"):
            iv.append(c)
        elif etype in ("symptom", "adverse_effect"):
            oc.append(c)
        elif etype in ("condition", "disease"):
            cd.append(c)
    relation_type = "side_effect" if (iv and oc) else None
    route = infer_route(*query_texts) if query_texts else None
    temporal = infer_temporal_relation(*query_texts) if query_texts else None
    return RWEQueryIntent(
        interventions=iv, outcomes=oc, conditions=cd,
        synonyms={}, relation_type=relation_type,
        route=route, temporal_relation=temporal,
        source="entities_fallback", confidence=0.5,
    )


def _slim_vocabulary(resolutions) -> Dict[str, List[dict]]:
    """Slim serialisable view of the resolver output for the intent."""
    out: Dict[str, List[dict]] = {}
    for term, res in (resolutions or {}).items():
        entries = []
        for m in res.matches:
            entries.append({
                "provider": m.provider,
                "concept_id": m.concept_id,
                "preferred_term": m.preferred_term,
                "synonyms": m.synonyms[:8],
                "semantic_group": m.semantic_group,
                "language": m.language,
                "confidence": m.confidence,
                "match_kind": m.match_kind,
            })
        if entries:
            out[term] = entries
    return out


def build_intent(
    translated_query: str,
    original_query: str,
    entities: list,
    use_llm: bool = True,
    client=None,
    resolver=None,
) -> RWEQueryIntent:
    """Build the query intent: LLM extraction (1 call) with automatic
    deterministic KB fallback on any failure. When a VocabularyResolver is
    provided (feature flag), canonical terms are resolved against external
    vocabularies and attached as typed evidence."""
    intent = None
    if use_llm:
        intent = extract_intent_llm(translated_query, original_query, client=client)
    if intent is None:
        intent = intent_from_entities(entities, translated_query, original_query)
    # Route / temporal cues are lexical: infer them deterministically when the
    # LLM (or the KB fallback) did not provide them.
    if intent.route is None:
        intent.route = infer_route(translated_query, original_query)
    if intent.temporal_relation is None:
        intent.temporal_relation = infer_temporal_relation(
            translated_query, original_query)
    if resolver is not None:
        try:
            canonicals = (intent.interventions + intent.outcomes
                          + intent.conditions)
            resolutions = resolver.resolve_terms(canonicals)
            intent.vocabulary = _slim_vocabulary(resolutions)
        except Exception as exc:  # noqa: BLE001 — vocabulary is optional evidence
            logger.info("vocabulary resolution skipped (%s)", type(exc).__name__)
    return intent


def merged_sides(intent: RWEQueryIntent, entities: list) -> Dict[str, set]:
    """Term sets for V3 scoring: intent sides + their synonyms (QU and
    provider vocabulary), UNIONED with the provider-recognised entities of
    the plan. Returns {"iv", "oc", "cd"} sets of lowercase terms plus
    "tiers": a term → evidence-weight map populated ONLY from external
    vocabulary evidence, and "generic_vocab": the provider-derived open
    event-set for generic any-event queries ("finasteride side effects").
    Terms absent from "tiers" default to weight 1.0, so without vocabulary
    evidence the scoring is unchanged.
    """
    iv = set(_clean_terms(intent.interventions))
    oc = set(_clean_terms(intent.outcomes))
    cd = set(_clean_terms(intent.conditions))

    # External vocabulary evidence (typed, tiered; related_concept excluded
    # by scored_terms in core.vocab.models). A provider term joins the SAME
    # side as the canonical it resolves — it never jumps sides.
    tiers: Dict[str, float] = {}
    generic_vocab: set = set()
    vocabulary = getattr(intent, "vocabulary", None) or {}
    if vocabulary:
        from core.vocab.models import VocabularyResolution
        side_of = {t: iv for t in iv}
        side_of.update({t: oc for t in oc})
        side_of.update({t: cd for t in cd})
        for canonical, entries in vocabulary.items():
            bucket = side_of.get(canonical)
            if bucket is None:
                continue
            if canonical in GENERIC_EVENT_TERMS:
                # Generic any-event canonical: its provider terms feed the
                # OPEN event-set, never the side itself.
                for e in entries:
                    for t in [e.get("preferred_term"), *(e.get("synonyms") or [])]:
                        t = (t or "").lower().strip()
                        if len(t) >= 3:
                            generic_vocab.add(t)
                continue
            res = VocabularyResolution.from_slim(canonical, entries)
            for term, weight in res.scored_terms().items():
                if term != canonical:
                    bucket.add(term)
                if term not in tiers or weight > tiers[term]:
                    tiers[term] = weight
        # Anchor (intervention) related concepts: for a generic any-event
        # query the user asks about ANY event of the exposure — the drug's
        # provider-typed related concepts ARE that open event-set. They are
        # NOT used as synonyms anywhere (related_concept stays unscored).
        for canonical in iv:
            for e in vocabulary.get(canonical, []):
                if e.get("match_kind") != "related_concept":
                    continue
                for t in [e.get("preferred_term"), *(e.get("synonyms") or [])]:
                    t = (t or "").lower().strip()
                    if len(t) >= 3:
                        generic_vocab.add(t)

    # QU synonyms, bucketed by where their canonical lives
    for canonical, aliases in (intent.synonyms or {}).items():
        c = _clean_term(canonical)
        if not c:
            continue
        bucket = iv if c in iv else oc if c in oc else cd if c in cd else None
        if bucket is not None:
            bucket.update(_clean_terms(aliases, _MAX_SYNONYMS))

    # Provider-recognised entities join their side (canonical form only —
    # their variants already arrived via the vocabulary evidence above).
    for etype, canonical, _conf in entities or []:
        c = (canonical or "").lower()
        if etype in ("drug", "active_ingredient"):
            iv.add(c)
        elif etype in ("symptom", "adverse_effect"):
            oc.add(c)
        elif etype in ("condition", "disease"):
            cd.add(c)

    return {"iv": iv, "oc": oc, "cd": cd, "tiers": tiers,
            "generic_vocab": generic_vocab}
