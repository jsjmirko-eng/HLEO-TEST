"""
RxNorm provider — NLM RxNav REST API (official, public, no API key).

Docs: https://lhncbc.nlm.nih.gov/RxNav/APIs/

Motivation for HLEO: drug canonicalisation — recognise a drug concept, map
brand names to generics (Rogaine → minoxidil) and generics to brands, and
normalise drug spellings. This is the drug-side evidence for the QU intent.
"""
from __future__ import annotations

from typing import List, Optional

from core.vocab.base import VocabularyProvider
from core.vocab.models import VocabularyMatch

BASE = "https://rxnav.nlm.nih.gov/REST"

# Term types that make useful lexical variants for RWE text matching.
_BRAND_TTYS = {"BN", "SBD", "BPCK"}
_GENERIC_TTYS = {"IN", "SCD", "PIN", "MIN", "GPCK"}


class RxNormProvider(VocabularyProvider):
    name = "rxnorm"
    semantic_group_hint = "drug"

    def _search(self, term: str, language: str,
                semantic_types: Optional[list],
                limit: int) -> List[VocabularyMatch]:
        # 1. Exact-ish lookup of the RXCUI by name.
        data = self._get_json(f"{BASE}/rxcui.json",
                              params={"name": term, "search": 1})
        ids = (data or {}).get("idGroup", {}).get("rxnormId", []) or []
        matches: List[VocabularyMatch] = []
        for rxcui in ids[: max(1, limit)]:
            match = self._concept_from_rxcui(rxcui, term)
            if match:
                matches.append(match)
        if matches:
            return matches[:limit]

        # 2. Approximate match fallback (normalized spellings).
        data = self._get_json(f"{BASE}/approximateTerm.json",
                              params={"term": term, "maxEntries": limit})
        candidates = ((data or {}).get("approximateGroup", {})
                      .get("candidate", []) or [])
        for cand in candidates[:limit]:
            rxcui = cand.get("rxcui")
            score = float(cand.get("score", 0) or 0) / 100.0
            if not rxcui:
                continue
            match = self._concept_from_rxcui(rxcui, term)
            if match:
                match.match_kind = "normalized"
                match.confidence = min(match.confidence, round(score, 3))
                matches.append(match)
        return matches[:limit]

    def _concept_from_rxcui(self, rxcui: str, queried_term: str) -> Optional[VocabularyMatch]:
        # NOTE: RxNav rejects URL-encoded %2B — the tty list must keep the
        # literal '+' (same pitfall as the openFDA query-syntax bug).
        data = self._get_json(
            f"{BASE}/rxcui/{rxcui}/related.json?tty=IN+BN+SCD+SBD")
        groups = (data or {}).get("relatedGroup", {}).get("conceptGroup", []) or []
        ingredient, brands, others = None, [], []
        for g in groups:
            tty = g.get("tty")
            for prop in g.get("conceptProperties", []) or []:
                name = (prop.get("name") or "").strip()
                if not name:
                    continue
                if tty == "IN" and ingredient is None:
                    ingredient = name
                elif tty in _BRAND_TTYS:
                    brands.append(name)
                elif tty in _GENERIC_TTYS:
                    others.append(name)
        preferred = ingredient or (others[0] if others else None)
        if not preferred:
            return None
        synonyms = []
        for n in brands + others:
            if n.lower() != preferred.lower() and n not in synonyms:
                synonyms.append(n)
        kind = ("exact" if queried_term.lower() == preferred.lower()
                else "synonym" if queried_term.lower()
                in {s.lower() for s in synonyms} else "normalized")
        return VocabularyMatch(
            provider=self.name,
            concept_id=str(rxcui),
            preferred_term=preferred,
            synonyms=synonyms[:12],
            semantic_group="drug",
            language="en",
            confidence=1.0 if kind == "exact" else 0.9,
            match_kind=kind,
            source_url=f"https://mor.nlm.nih.gov/RxNav/search?searchBy=RXCUI&searchTerm={rxcui}",
            metadata={"rxcui": str(rxcui)},
        )

    def _get_synonyms(self, concept_id: str) -> List[str]:
        match = self._concept_from_rxcui(concept_id, queried_term="")
        return match.synonyms if match else []

    def _get_concept(self, concept_id: str) -> Optional[VocabularyMatch]:
        return self._concept_from_rxcui(concept_id, queried_term="")
