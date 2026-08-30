"""
ConceptNet provider — general-purpose multilingual knowledge graph.

Docs: https://github.com/commonsense/conceptnet5/wiki/API
API:  https://api.conceptnet.io  (public, no key, JSON)
License: CC BY-SA 4.0 — API responses cached per-term (allowed).

Motivation for HLEO: patients write colloquially and in many languages.
ConceptNet covers 300+ languages with typed relations, so it supplies
evidence the medical vocabularies cannot: multilingual equivalents,
spelling variants and common/colloquial phrasings — as EVIDENCE with typed
match kinds, never as automatic synonyms.

Relation → match_kind mapping (typed, conservative):
  /r/Synonym       → synonym (cross-language ends → translation)
  /r/FormOf        → orthographic_variant
  /r/RelatedTo,
  /r/IsA,
  /r/PartOf,
  /r/Causes        → related_concept  (evidence only, never scored)
"""
from __future__ import annotations

import logging
from typing import List, Optional

import requests

from core.vocab.base import VocabularyProvider
from core.vocab.models import VocabularyMatch

logger = logging.getLogger(__name__)

API = "https://api.conceptnet.io"

# The web frontend (conceptnet.io) is the fallback ONLY for API outages:
# 502/503/504, timeout or connection error. 4xx responses and empty results
# are NOT outages and never trigger the fallback.
_FALLBACK_STATUSES = {502, 503, 504}

_REL_KIND = {
    "Synonym": "synonym",
    "FormOf": "orthographic_variant",
    "RelatedTo": "related_concept",
    "IsA": "related_concept",
    "PartOf": "related_concept",
    "Causes": "related_concept",
}
_QUERY_RELS = "/r/Synonym,/r/FormOf,/r/RelatedTo,/r/IsA"


class ConceptNetProvider(VocabularyProvider):
    name = "conceptnet"
    semantic_group_hint = "general"

    @staticmethod
    def _node(term: str, language: str) -> str:
        return f"/c/{language}/{term.strip().lower().replace(' ', '_')}"

    def _search(self, term: str, language: str,
                semantic_types: Optional[list],
                limit: int) -> List[VocabularyMatch]:
        node = self._node(term, language)
        try:
            data = self._get_json(f"{API}/query",
                                  params={"node": node, "rel": _QUERY_RELS,
                                          "limit": 50})
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status not in _FALLBACK_STATUSES:
                raise  # 4xx / other: not an outage, no web fallback
            logger.info("conceptnet API %s for %r — trying web fallback",
                        status, term)
            return self._web_fallback(term, language, limit)
        except (requests.Timeout, requests.ConnectionError) as exc:
            logger.info("conceptnet API unreachable for %r (%s) — web fallback",
                        term, type(exc).__name__)
            return self._web_fallback(term, language, limit)
        edges = (data or {}).get("edges", []) or []

        preferred = term.strip().lower()
        synonyms, translations, variants, related = [], [], [], []
        for e in edges:
            rel = (e.get("rel") or {}).get("label") or ""
            kind = _REL_KIND.get(rel)
            if not kind:
                continue
            start, end = e.get("start", {}), e.get("end", {})
            other = end if (start.get("@id") == node) else start
            label = (other.get("label") or "").strip()
            lang = other.get("language") or ""
            if not label or label.lower() == preferred:
                continue
            if rel == "Synonym":
                if lang and lang != language:
                    translations.append((label, lang))
                else:
                    synonyms.append(label)
            elif rel == "FormOf":
                variants.append(label)
            else:
                related.append(label)

        matches: List[VocabularyMatch] = []
        if not (synonyms or translations or variants or related):
            return matches

        base = dict(provider=self.name, concept_id=node,
                    preferred_term=preferred, semantic_group="general",
                    source_url=f"https://conceptnet.io{node}")
        if synonyms:
            matches.append(VocabularyMatch(
                **base, synonyms=synonyms[:10], language=language,
                confidence=0.8, match_kind="synonym",
                metadata={"relation": "Synonym"}))
        for label, lang in translations[:10]:
            matches.append(VocabularyMatch(
                **dict(base, preferred_term=label),
                synonyms=[], language=lang,
                confidence=0.75, match_kind="translation",
                metadata={"source_term": preferred,
                          "source_language": language}))
        if variants:
            matches.append(VocabularyMatch(
                **base, synonyms=variants[:5], language=language,
                confidence=0.7, match_kind="orthographic_variant",
                metadata={"relation": "FormOf"}))
        if related:
            matches.append(VocabularyMatch(
                **base, synonyms=related[:10], language=language,
                confidence=0.4, match_kind="related_concept",
                metadata={"relation": "RelatedTo/IsA"}))
        return matches[: max(limit, len(matches))]

    def _get_synonyms(self, concept_id: str) -> List[str]:
        # concept_id is a ConceptNet node like /c/en/hair_loss
        try:
            _prefix, lang, term = concept_id.split("/", 3)[1:]
        except (ValueError, IndexError):
            return []
        return [t for m in self.search(term.replace("_", " "), language=lang)
                for t in m.synonyms]

    @staticmethod
    def _web_fallback(term: str, language: str,
                      limit: int) -> List[VocabularyMatch]:
        """Delegate to the conceptnet.io frontend during an API outage.

        Returns [] when the node does not exist (never invents results) and
        [] when the web frontend is also unreachable (provider unavailable —
        the base class contract is "never raise")."""
        from core.vocab import conceptnet_web
        matches = conceptnet_web.search_web(term, language, limit=limit)
        return matches if matches is not None else []
