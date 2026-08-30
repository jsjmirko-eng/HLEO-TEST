"""
UMLS provider — official NLM UMLS Terminology Services (UTS) REST API.

Docs: https://documentation.uts.nlm.nih.gov/rest/home.html
Auth: UTS API key passed as the ``apiKey`` query parameter.

Scope in HLEO (Catena C): conservative normalisation/expansion of clinical
query terms — CUI, preferred concept name, synonyms (English atoms), semantic
types (mapped to HLEO semantic groups). Trichology queries (alopecia, hair
loss, finasteride, minoxidil, ...) are covered by UMLS vocabularies natively;
nothing trichology-specific is hardcoded here.

Design rules:
- The API key is read ONLY from the environment (``UMLS_API_KEY``; legacy
  ``HLEO_UMLS_API_KEY`` accepted). It is never logged, never embedded in
  exceptions, never written to cache keys or persisted payloads.
- Without the key the provider is INACTIVE (available() False) and the whole
  pipeline degrades gracefully — every operation returns [] / None.
- Conservative matching: exact search first; a broader normalized search only
  as fallback. A hypernym whose name is a strict token-subset of a multi-token
  query ("joint pain" → "Pain") is NOT accepted as a normalised substitute —
  the specific concept keeps priority.
- One bounded retry on HTTP 429 (rate limit); every other failure degrades
  to [] / None via the base contract (never raises).
- Responses are cached through the shared VocabCache (base class).
"""
from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

import requests

from core.vocab.base import VocabularyProvider
from core.vocab.models import VocabularyMatch

logger = logging.getLogger(__name__)

BASE = "https://uts-ws.nlm.nih.gov/rest"
CONCEPT_PAGE = "https://uts.nlm.nih.gov/uts/umls/concept/{}"

_TIMEOUT = 10
_MAX_SYNONYMS = 8
_429_BACKOFF_S = 2.0

# UMLS semantic-type name → HLEO semantic group. This is a TYPING map (how the
# provider classifies concepts), not terminology: it contains no clinical terms.
_STY_TO_GROUP = {
    "Pharmacologic Substance": "drug",
    "Clinical Drug": "drug",
    "Antibiotic": "drug",
    "Sign or Symptom": "symptom",
    "Finding": "symptom",
    "Disease or Syndrome": "condition",
    "Neoplastic Process": "condition",
    "Pathologic Function": "condition",
    "Mental or Behavioral Dysfunction": "condition",
    "Anatomical Abnormality": "condition",
    "Congenital Abnormality": "condition",
}

_WORD_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")


def _tokens(text: str) -> List[str]:
    return [t for t in "".join(
        c if c in _WORD_CHARS else " " for c in (text or "").lower()
    ).split() if t]


class UMLSProvider(VocabularyProvider):
    name = "umls"
    semantic_group_hint = "general"

    def __init__(self, cache=None, timeout: int = _TIMEOUT) -> None:
        super().__init__(cache=cache, timeout=timeout)

    # ── auth / availability ──────────────────────────────────────────────
    @staticmethod
    def _api_key() -> str:
        return (os.getenv("UMLS_API_KEY") or
                os.getenv("HLEO_UMLS_API_KEY") or "").strip()

    def available(self) -> bool:
        return bool(self._api_key())

    # ── HTTP layer: key injection + sanitized errors ─────────────────────
    def _get_json(self, url: str, params: Optional[dict] = None,
                  headers: Optional[dict] = None, auth=None):
        """UTS GET with the API key. Errors are re-raised WITHOUT the URL
        (which carries the key) so logs/exceptions can never leak it."""
        params = dict(params or {})
        params["apiKey"] = self._api_key()
        last_status = None
        for attempt in range(2):  # 1 initial + 1 bounded retry on 429
            try:
                resp = requests.get(url, params=params, headers=headers,
                                    auth=auth, timeout=self.timeout)
            except requests.Timeout as exc:
                raise RuntimeError("umls timeout") from exc
            except requests.RequestException as exc:
                raise RuntimeError(
                    f"umls network error ({type(exc).__name__})") from exc
            if resp.status_code == 429 and attempt == 0:
                retry_after = resp.headers.get("Retry-After")
                try:
                    delay = min(float(retry_after), 5.0) if retry_after \
                        else _429_BACKOFF_S
                except ValueError:
                    delay = _429_BACKOFF_S
                time.sleep(delay)
                continue
            if resp.status_code != 200:
                last_status = resp.status_code
                raise RuntimeError(f"umls HTTP {resp.status_code}")
            try:
                return resp.json()
            except ValueError as exc:
                raise RuntimeError("umls invalid JSON") from exc
        raise RuntimeError(f"umls HTTP {last_status}")

    # ── search ───────────────────────────────────────────────────────────
    def _search(self, term: str, language: str,
                semantic_types: Optional[list],
                limit: int) -> List[VocabularyMatch]:
        out: List[VocabularyMatch] = []
        seen = set()
        for search_type, kind, conf in (("exact", "exact", 0.95),
                                        ("normalizedWords", "normalized", 0.7)):
            if len(out) >= limit:
                break
            data = self._get_json(f"{BASE}/search/current", params={
                "string": term, "searchType": search_type,
                "pageSize": max(1, limit)})
            rows = ((data or {}).get("result") or {}).get("results") or []
            for row in rows:
                cui = (row.get("ui") or "").strip()
                name = (row.get("name") or "").strip()
                if not cui or not name or not cui.startswith("C") \
                        or cui in seen:
                    continue
                if kind == "normalized" and self._is_hypernym(term, name):
                    continue  # never substitute a specific term with a generic
                seen.add(cui)
                match = self._build_match(cui, name, term, kind, conf)
                if match is None:
                    continue
                if semantic_types and match.semantic_group not in semantic_types:
                    continue
                out.append(match)
                if len(out) >= limit:
                    break
        return out[:limit]

    @staticmethod
    def _is_hypernym(query: str, concept_name: str) -> bool:
        """True when the matched concept is a strict token-subset of a
        multi-token query — i.e. a generic hypernym (query "joint pain" →
        concept "Pain"). Exact-string matches are never hypernyms."""
        q_toks = _tokens(query)
        c_toks = _tokens(concept_name)
        if not q_toks or not c_toks:
            return False
        if " ".join(q_toks) == " ".join(c_toks):
            return False
        return len(c_toks) < len(q_toks) and set(c_toks) < set(q_toks)

    def _build_match(self, cui: str, name: str, queried: str,
                     kind: str, conf: float) -> Optional[VocabularyMatch]:
        """Enrich a search hit with semantic types + English synonyms (one
        content call + one atoms call per concept, cached upstream)."""
        semantic_names: List[str] = []
        try:
            content = self._get_json(f"{BASE}/content/current/CUI/{cui}")
            cres = (content or {}).get("result") or {}
            semantic_names = [st.get("name", "") for st in
                              (cres.get("semanticTypes") or []) if st.get("name")]
            preferred = (cres.get("name") or name).strip() or name
        except Exception as exc:  # noqa: BLE001 — enrichment is optional
            logger.info("umls content enrichment failed for %s (%s)",
                        cui, type(exc).__name__)
            preferred = name
        synonyms = self._get_synonyms(cui)
        group = "general"
        for st in semantic_names:
            if st in _STY_TO_GROUP:
                group = _STY_TO_GROUP[st]
                break
        if kind == "exact" and queried.strip().lower() != preferred.lower():
            # UMLS "exact" is case/space-insensitive at the string level; if
            # the preferred name differs it is a synonym-level match.
            if queried.strip().lower() in {s.lower() for s in synonyms}:
                kind = "synonym"
                conf = min(conf, 0.9)
        return VocabularyMatch(
            provider=self.name,
            concept_id=cui,
            preferred_term=preferred,
            synonyms=[s for s in synonyms
                      if s.strip().lower() != preferred.strip().lower()],
            semantic_group=group,
            language="en",
            confidence=conf,
            match_kind=kind,
            source_url=CONCEPT_PAGE.format(cui),
            metadata={"semantic_types": semantic_names},
        )

    def _get_synonyms(self, concept_id: str) -> List[str]:
        data = self._get_json(
            f"{BASE}/content/current/CUI/{concept_id}/atoms",
            params={"language": "ENG", "pageSize": 50})
        rows = (data or {}).get("result") or []
        out: List[str] = []
        for row in rows:
            name = (row.get("name") or "").strip()
            if name and len(name) >= 3 and name.lower() not in {
                    s.lower() for s in out}:
                out.append(name)
            if len(out) >= _MAX_SYNONYMS:
                break
        return out

    def _get_concept(self, concept_id: str) -> Optional[VocabularyMatch]:
        cui = (concept_id or "").strip()
        if not cui:
            return None
        content = self._get_json(f"{BASE}/content/current/CUI/{cui}")
        cres = (content or {}).get("result") or {}
        name = (cres.get("name") or "").strip()
        if not name:
            return None
        semantic_names = [st.get("name", "") for st in
                          (cres.get("semanticTypes") or []) if st.get("name")]
        group = "general"
        for st in semantic_names:
            if st in _STY_TO_GROUP:
                group = _STY_TO_GROUP[st]
                break
        return VocabularyMatch(
            provider=self.name,
            concept_id=cui,
            preferred_term=name,
            synonyms=self._get_synonyms(cui),
            semantic_group=group,
            language="en",
            confidence=1.0,
            match_kind="canonical",
            source_url=CONCEPT_PAGE.format(cui),
            metadata={"semantic_types": semantic_names},
        )
