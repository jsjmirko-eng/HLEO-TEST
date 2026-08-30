"""
Wikidata provider — general-purpose multilingual entity aliases.

Docs: https://www.wikidata.org/w/api.php  (official MediaWiki API, public)
License: CC0 — free reuse, cacheable.

Motivation for HLEO: entity-level aliases and multilingual labels (drug
brand spellings, alternative names) complement ConceptNet. Wikidata is
entity-centric: it does NOT know colloquial phrases, and plain word search
can return off-domain entities (e.g. "minox" → an asteroid). Matches are
therefore typed and confidence-capped — evidence, never auto-synonyms.
"""
from __future__ import annotations

from typing import List, Optional

from core.vocab.base import VocabularyProvider
from core.vocab.models import VocabularyMatch

API = "https://www.wikidata.org/w/api.php"
_UA = {"User-Agent": "HLEO-vocab-layer/1.0 (research; contact: repository owner)"}


class WikidataProvider(VocabularyProvider):
    name = "wikidata"
    semantic_group_hint = "general"

    def _search(self, term: str, language: str,
                semantic_types: Optional[list],
                limit: int) -> List[VocabularyMatch]:
        data = self._get_json(API, params={
            "action": "wbsearchentities", "search": term,
            "language": language, "format": "json",
            "limit": min(limit, 5), "type": "item",
        }, headers=_UA)
        out: List[VocabularyMatch] = []
        for row in (data or {}).get("search", []) or []:
            qid = row.get("id") or ""
            label = (row.get("label") or "").strip()
            if not qid or not label:
                continue
            match_type = (row.get("match") or {}).get("type", "")
            if match_type == "label" and label.lower() == term.lower():
                kind, conf = "exact", 0.9
            elif match_type == "alias":
                kind, conf = "abbreviation", 0.6
            else:
                kind, conf = "concept", 0.5
            aliases = [a for a in (row.get("aliases") or [])
                       if a.lower() != label.lower()][:8]
            out.append(VocabularyMatch(
                provider=self.name,
                concept_id=qid,
                preferred_term=label,
                synonyms=aliases,
                semantic_group="general",
                language=language,
                confidence=conf,
                match_kind=kind,
                source_url=f"https://www.wikidata.org/wiki/{qid}",
                metadata={"description": row.get("description", "")},
            ))
        return out[:limit]
