"""
Provider-first entity recognition for the Catena C.

Replaces the legacy hardcoded dictionaries (biomedical_kb lookup_entity /
alias tables): candidate terms are extracted from the query text and resolved
against the EXTERNAL vocabulary providers through a VocabularyResolver
(RxNorm → drugs, MeSH → conditions/symptoms via tree numbers, ConceptNet /
Wikidata → multilingual general concepts).

No terminology is stored here: every recognised entity comes from a provider
match with semantic_group ∈ {drug, condition, symptom} and confidence above
the floor. Offline behaviour is the resolver's own contract (providers never
raise; no providers → no entities → the caller degrades gracefully).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.vocab.models import VocabularyResolution

ENTITY_CONFIDENCE_FLOOR = 0.5

_GROUP_TO_ETYPE = {
    "drug": "drug",
    "condition": "condition",
    "symptom": "symptom",
}

# Only identity-level matches can mint an entity: a "concept" (class-level
# descriptor) or "related_concept" is a DIFFERENT concept, not the term the
# user wrote (it stays available as expansion evidence, never as an entity).
_ENTITY_MATCH_KINDS = {
    "exact", "canonical", "preferred", "synonym", "translation",
    "normalized", "abbreviation", "orthographic_variant", "colloquial",
}

# Grammatical stopwords (NOT terminology): candidates made only of these carry
# no biomedical signal and are not worth a provider lookup.
_GRAMMAR_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "with",
    "my", "i", "is", "are", "was", "it", "this", "that", "from", "by",
    "can", "cause", "does", "may", "will", "you", "your", "after", "before",
    "di", "il", "la", "lo", "per", "che", "e", "un", "una", "del", "della",
    "dei", "delle", "mi", "si", "ho", "ha", "non", "da", "dopo", "prima",
    "le", "les", "de", "des", "et", "en", "un", "une", "der", "die", "das",
    "und", "el", "los", "las", "una", "por", "para", "con",
}


@dataclass
class EntityRecognition:
    """Result of provider-first entity recognition over one text."""
    entities: List[Tuple[str, str, float]] = field(default_factory=list)
    # (etype, canonical, confidence) — same contract as the legacy KB lookup
    resolutions: Dict[str, VocabularyResolution] = field(default_factory=dict)
    # keyed by canonical preferred term (aligned with entities)
    surfaces: Dict[str, str] = field(default_factory=dict)
    # canonical → surface form found in the analysed text


def _candidates(text: str, max_n: int = 3) -> List[str]:
    """N-gram candidate terms from raw text.

    Whitespace tokens keep scripts the latin regex cannot tokenise (CJK,
    Cyrillic, Arabic): a Japanese query still yields lookup candidates.
    """
    out: List[str] = []
    words = [w.strip(".,;:!?()\"'«»").lower()
             for w in re.split(r"\s+", (text or "").strip()) if w.strip()]
    latin = re.findall(r"[a-zà-öø-ÿ0-9]+", (text or "").lower())
    for w in words:
        if len(w) >= 3:
            out.append(w)
    for n in range(1, max_n + 1):
        for i in range(len(latin) - n + 1):
            out.append(" ".join(latin[i:i + n]))
    seen: set = set()
    uniq: List[str] = []
    for c in out:
        if not c or len(c) < 3 or c in seen:
            continue
        if all(tok in _GRAMMAR_STOPWORDS for tok in c.split()):
            continue
        seen.add(c)
        uniq.append(c)
    return uniq


def recognize(
    text: str,
    language: str,
    resolver,
    max_n: int = 3,
    confidence_floor: float = ENTITY_CONFIDENCE_FLOOR,
) -> EntityRecognition:
    """Recognise biomedical entities in ``text`` via the resolver's providers.

    ``language`` is the ISO code of ``text`` (multilingual providers such as
    ConceptNet use it to look up native-language nodes; cross-language
    synonym edges come back as ``translation`` matches).
    """
    result = EntityRecognition()
    if resolver is None or not (text or "").strip():
        return result
    candidates = _candidates(text, max_n=max_n)
    if not candidates:
        return result
    lang = (language or "en").lower()
    if lang in {"und", ""}:
        lang = "en"
    found = resolver.resolve_terms(candidates, language=lang)

    best: Dict[str, Tuple[str, str, float]] = {}
    for candidate, resolution in (found or {}).items():
        for match in resolution.matches:
            etype = _GROUP_TO_ETYPE.get(match.semantic_group)
            if etype is None or match.confidence < confidence_floor:
                continue
            if match.match_kind not in _ENTITY_MATCH_KINDS:
                continue
            canonical = (match.preferred_term or "").strip().lower()
            if not canonical:
                continue
            if canonical not in best or match.confidence > best[canonical][2]:
                best[canonical] = (etype, canonical, match.confidence)
                result.surfaces[canonical] = candidate
            if canonical not in result.resolutions:
                result.resolutions[canonical] = resolution
    result.entities = sorted(best.values(), key=lambda e: -e[2])
    return result


def merge_recognitions(*recs: EntityRecognition) -> EntityRecognition:
    """Merge recognitions of the same query in different languages
    (e.g. the English translation + the original-language text)."""
    merged = EntityRecognition()
    best: Dict[str, Tuple[str, str, float]] = {}
    for rec in recs:
        for etype, canonical, conf in rec.entities:
            if canonical not in best or conf > best[canonical][2]:
                best[canonical] = (etype, canonical, conf)
        for canonical, resolution in rec.resolutions.items():
            merged.resolutions.setdefault(canonical, resolution)
        for canonical, surface in rec.surfaces.items():
            merged.surfaces.setdefault(canonical, surface)
    merged.entities = sorted(best.values(), key=lambda e: -e[2])
    return merged
