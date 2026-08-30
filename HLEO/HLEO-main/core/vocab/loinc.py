"""
LOINC provider — official LOINC FHIR terminology server (fhir.loinc.org).

Docs: https://loinc.org/fhir/  (CodeSystem/$lookup, ValueSet/$expand)

Motivation for HLEO: laboratory tests, clinical observations and measurements
(e.g. DHT, testosterone, ferritin discussed in RWE posts) — the lab side of
the QU intent.

LOINC requires a FREE account. Credentials are read ONLY from environment
variables (HLEO_LOINC_USERNAME / HLEO_LOINC_PASSWORD); when absent the
provider reports available() == False and is skipped silently. Credentials
are never logged, cached or written anywhere.
"""
from __future__ import annotations

import os
from typing import List, Optional

from core.vocab.base import VocabularyProvider
from core.vocab.models import VocabularyMatch

FHIR = "https://fhir.loinc.org"


class LOINCProvider(VocabularyProvider):
    name = "loinc"
    semantic_group_hint = "lab"

    def _credentials(self):
        user = os.getenv("HLEO_LOINC_USERNAME", "").strip()
        pw = os.getenv("HLEO_LOINC_PASSWORD", "").strip()
        return (user, pw) if user and pw else None

    def available(self) -> bool:
        return self._credentials() is not None

    def _search(self, term: str, language: str,
                semantic_types: Optional[list],
                limit: int) -> List[VocabularyMatch]:
        data = self._get_json(
            f"{FHIR}/ValueSet/$expand",
            params={"url": "http://loinc.org/vs", "filter": term,
                    "count": limit},
            headers={"Accept": "application/fhir+json"},
            auth=self._credentials(),
        )
        contains = ((data or {}).get("expansion", {}) or {}).get("contains", []) or []
        out: List[VocabularyMatch] = []
        for row in contains[:limit]:
            code = row.get("code") or ""
            display = (row.get("display") or "").strip()
            if not code or not display:
                continue
            out.append(VocabularyMatch(
                provider=self.name,
                concept_id=code,
                preferred_term=display,
                synonyms=[],
                semantic_group="lab",
                language="en",
                confidence=0.8,
                match_kind=("exact" if term.lower() == display.lower()
                            else "concept"),
                source_url=f"https://loinc.org/{code}/",
                metadata={},
            ))
        return out

    def _get_concept(self, concept_id: str) -> Optional[VocabularyMatch]:
        data = self._get_json(
            f"{FHIR}/CodeSystem/$lookup",
            params={"system": "http://loinc.org", "code": concept_id},
            headers={"Accept": "application/fhir+json"},
            auth=self._credentials(),
        )
        display = ((data or {}).get("parameter", []) or [])
        name = ""
        for p in display:
            if p.get("name") == "display":
                name = p.get("valueString", "")
                break
        if not name:
            return None
        return VocabularyMatch(
            provider=self.name, concept_id=concept_id, preferred_term=name,
            semantic_group="lab", language="en", confidence=1.0,
            match_kind="concept", source_url=f"https://loinc.org/{concept_id}/",
        )
