"""
Normalized vocabulary models shared by every VocabularyProvider.

A provider never returns raw API payloads: it returns VocabularyMatch objects
so the resolver, the QU intent and the relevance scorer consume ONE shape.

match_kind distinguishes the strength of the evidence — the resolver and the
scorer must NOT treat every kind as an equivalent synonym:

    exact               the term IS the provider's exact concept label
    canonical           the term maps to the canonical/preferred concept name
    preferred           preferred-term match
    synonym             provider-declared synonym / entry term
    translation         multilingual equivalent (same concept, other language)
    abbreviation        abbreviation / acronym expansion
    orthographic_variant spelling variant
    colloquial          patient/colloquial phrasing of the concept
    slang               slang phrasing (weakest lexical evidence)
    normalized          normalized/approximate match (e.g. RxNorm approximate)
    concept             same concept via a structured link (not a lexical form)
    related_concept     weaker semantic relationship — EVIDENCE ONLY, never
                        injected into scoring side-sets
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

MATCH_KINDS = {
    "exact", "canonical", "preferred", "synonym", "translation",
    "abbreviation", "orthographic_variant", "colloquial", "slang",
    "normalized", "concept", "related_concept",
}

# Evidence tier per match_kind (1.0 = strongest). Used by the scorer to weight
# a matched term; kinds absent here are evidence-only and never scored.
MATCH_TIERS: Dict[str, float] = {
    "exact": 1.0,
    "canonical": 1.0,
    "preferred": 0.95,
    "synonym": 0.90,
    "translation": 0.85,
    "abbreviation": 0.85,
    "orthographic_variant": 0.85,
    "normalized": 0.70,
    "colloquial": 0.70,
    "concept": 0.60,
    "slang": 0.60,
    # related_concept: intentionally absent — never scored
}

SEMANTIC_GROUPS = {"drug", "condition", "symptom", "lab", "general"}


class VocabularyMatch(BaseModel):
    """One normalized provider result."""
    provider: str                      # "rxnorm" | "mesh" | "loinc" | ...
    concept_id: str                    # RXCUI, MeSH UI, LOINC code, QID, /c/en/x
    preferred_term: str
    synonyms: List[str] = Field(default_factory=list)
    semantic_group: str = "general"    # drug | condition | symptom | lab | general
    language: str = "en"
    confidence: float = 0.0            # 0–1, provider-assigned
    match_kind: str = "concept"        # one of MATCH_KINDS
    source_url: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class VocabularyResolution(BaseModel):
    """Outcome of resolving one term across all active providers."""
    term: str
    language: str = "en"
    matches: List[VocabularyMatch] = Field(default_factory=list)
    providers_queried: List[str] = Field(default_factory=list)
    providers_failed: List[str] = Field(default_factory=list)

    def scored_terms(self) -> Dict[str, float]:
        """term → evidence tier, for the kinds allowed into scoring side-sets.
        related_concept and unknown kinds are excluded by design."""
        out: Dict[str, float] = {}
        for m in self.matches:
            tier = MATCH_TIERS.get(m.match_kind)
            if tier is None:
                continue
            candidates = [m.preferred_term, *m.synonyms]
            for c in candidates:
                c = (c or "").lower().strip()
                if len(c) < 3:
                    continue
                w = round(tier * max(0.0, min(1.0, m.confidence)), 3)
                if c not in out or w > out[c]:
                    out[c] = w
        return out

    @classmethod
    def from_slim(cls, term: str, entries) -> "VocabularyResolution":
        """Rebuild a resolution from the slim serialisable view (list of
        match dicts) attached to intents / query plans."""
        return cls(term=term,
                   matches=[VocabularyMatch(**e) for e in entries or []])
