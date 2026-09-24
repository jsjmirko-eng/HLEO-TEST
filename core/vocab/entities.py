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
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
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


# Relation phrases are syntax, not biomedical concepts. Keep these separate from
# provider lookup so a connector can never become a recognised entity.
_RELATION_CONNECTOR_TRANSLATIONS = {
    "induced by": "induced by",
    "induced": "induced",
    "after taking": "after taking",
    "after applying": "after applying",
    "after using": "after using",
    "after starting": "after starting",
    "since taking": "since taking",
    "since starting": "since starting",
    "caused by": "caused by",
    "due to": "due to",
    "indotta da": "induced by",
    "indotto da": "induced by",
    "indotte da": "induced by",
    "indotti da": "induced by",
    "dopo aver assunto": "after taking",
    "dopo aver preso": "after taking",
    "dopo l'assunzione di": "after taking",
    "dopo aver applicato": "after applying",
    "dopo l'applicazione di": "after applying",
    "dopo aver usato": "after using",
    "dopo l'uso di": "after using",
    "dopo aver iniziato": "after starting",
    "dopo l'inizio di": "after starting",
    "da quando prende": "since taking",
    "da quando assume": "since taking",
    "da quando usa": "since using",
    "da quando ha iniziato": "since starting",
    "causata da": "caused by",
    "causato da": "caused by",
    "causate da": "caused by",
    "causati da": "caused by",
    "dovuta a": "due to",
    "dovuto a": "due to",
    "dovute a": "due to",
    "dovuti a": "due to",
}
_RELATION_CONNECTOR_PATTERN = re.compile(
    r"(?<!\w)(?:" + "|".join(
        re.escape(term) for term in sorted(
            _RELATION_CONNECTOR_TRANSLATIONS, key=len, reverse=True
        )
    ) + r")(?!\w)",
    flags=re.IGNORECASE,
)


def strip_relation_connectors(text: str) -> str:
    """Remove relation syntax before sending terms to clinical providers."""
    return _RELATION_CONNECTOR_PATTERN.sub(" ", text or "")


def normalize_relation_connectors(text: str) -> str:
    """Translate known relation syntax while preserving the surrounding text."""
    out = text or ""
    for source, target in sorted(
        _RELATION_CONNECTOR_TRANSLATIONS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        out = re.sub(rf"(?<!\w){re.escape(source)}(?!\w)", target, out,
                     flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def contains_relation_connector(text: str) -> bool:
    return bool(_RELATION_CONNECTOR_PATTERN.search(text or ""))



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
    searchable_text = strip_relation_connectors(text)
    words = [w.strip(".,;:!?()\"'«»").lower()
             for w in re.split(r"\s+", searchable_text.strip()) if w.strip()]
    latin = re.findall(r"[a-zà-öø-ÿ0-9]+", searchable_text.lower())
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


def _normalise_surface(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text.lower().strip())


def _match_surface_compatible(surface: str, match) -> bool:
    """Require provider evidence that a non-exact match names this surface."""
    surface_norm = _normalise_surface(surface)
    evidence = [match.preferred_term, *(match.synonyms or [])]
    source_term = (match.metadata or {}).get("source_term")
    if source_term:
        evidence.append(source_term)
    evidence_norm = {_normalise_surface(term) for term in evidence if term}
    if surface_norm in evidence_norm:
        return True

    surface_tokens = surface_norm.split()
    generic_tokens = {"hair", "capelli", "skin", "cutaneous", "clinical"}
    for term_norm in evidence_norm:
        term_tokens = set(term_norm.split())
        if len(surface_tokens) == 1 and SequenceMatcher(
                None, surface_norm, term_norm).ratio() >= 0.85:
            return True
        if set(surface_tokens) & term_tokens - generic_tokens:
            return True
    return False


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
            if (
                match.match_kind not in {"exact", "translation", "colloquial"}
                and not (
                    match.match_kind == "normalized"
                    and match.provider == "rxnorm"
                )
                and not _match_surface_compatible(candidate, match)
            ):
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
