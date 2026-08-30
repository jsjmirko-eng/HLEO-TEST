"""
VocabularyProvider — common interface for external structured vocabularies.

Contract for every provider (RxNorm, MeSH, LOINC, ConceptNet, Wikidata, and
the future UMLS / SNOMED CT):

- short timeouts, never hangs the pipeline;
- NEVER raises towards the caller: HTTP errors, timeouts, empty or malformed
  responses all degrade to [] / None with a log line;
- available() lets the resolver silently skip providers without credentials
  or configuration;
- responses go through the shared VocabCache (per provider, TTL).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import requests

from core.vocab.cache import VocabCache
from core.vocab.models import VocabularyMatch

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10


class VocabularyProvider(ABC):
    name: str = "abstract"
    semantic_group_hint: str = "general"

    def __init__(self, cache: Optional[VocabCache] = None,
                 timeout: int = DEFAULT_TIMEOUT) -> None:
        # NOTE: `cache or VocabCache()` would be wrong — an EMPTY VocabCache
        # is falsy (it defines __len__), which would silently discard a
        # shared empty cache.
        self.cache = cache if cache is not None else VocabCache()
        self.timeout = timeout

    # ── availability ─────────────────────────────────────────────────────
    def available(self) -> bool:
        """Public providers are available by default; credential-based
        providers (LOINC, future UMLS/SNOMED) override this."""
        return True

    # ── public API (cached, never raising) ───────────────────────────────
    def search(self, term: str, language: Optional[str] = None,
               semantic_types: Optional[list] = None,
               limit: int = 10) -> List[VocabularyMatch]:
        if not term or not term.strip():
            return []
        if not self.available():
            return []
        lang = (language or "en").lower()
        cached = self.cache.get(self.name, "search", term, lang)
        if cached is not None:
            return [VocabularyMatch(**m) for m in cached]
        try:
            matches = self._search(term.strip(), lang, semantic_types, limit)
        except Exception as exc:  # noqa: BLE001
            logger.info("%s search failed for %r (%s)", self.name, term,
                        type(exc).__name__)
            matches = []
        self.cache.set(self.name, "search", term,
                       [m.model_dump() for m in matches], lang)
        return matches

    def get_synonyms(self, concept_id: str) -> List[str]:
        if not concept_id or not self.available():
            return []
        cached = self.cache.get(self.name, "synonyms", concept_id)
        if cached is not None:
            return list(cached)
        try:
            syns = self._get_synonyms(concept_id)
        except Exception as exc:  # noqa: BLE001
            logger.info("%s get_synonyms failed for %r (%s)", self.name,
                        concept_id, type(exc).__name__)
            syns = []
        self.cache.set(self.name, "synonyms", concept_id, syns)
        return syns

    def get_concept(self, concept_id: str) -> Optional[VocabularyMatch]:
        if not concept_id or not self.available():
            return None
        cached = self.cache.get(self.name, "concept", concept_id)
        if cached is not None:
            return VocabularyMatch(**cached) if cached else None
        try:
            match = self._get_concept(concept_id)
        except Exception as exc:  # noqa: BLE001
            logger.info("%s get_concept failed for %r (%s)", self.name,
                        concept_id, type(exc).__name__)
            match = None
        self.cache.set(self.name, "concept", concept_id,
                       match.model_dump() if match else None)
        return match

    # ── provider-specific implementations ────────────────────────────────
    @abstractmethod
    def _search(self, term: str, language: str,
                semantic_types: Optional[list],
                limit: int) -> List[VocabularyMatch]:
        ...

    def _get_synonyms(self, concept_id: str) -> List[str]:
        return []

    def _get_concept(self, concept_id: str) -> Optional[VocabularyMatch]:
        return None

    # ── HTTP helper (requests, short timeout, JSON) ──────────────────────
    def _get_json(self, url: str, params: Optional[dict] = None,
                  headers: Optional[dict] = None, auth=None):
        resp = requests.get(url, params=params, headers=headers, auth=auth,
                            timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
