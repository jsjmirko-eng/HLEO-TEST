"""
VocabularyResolver — orchestrates the active providers behind env flags.

Design rules:
- DISABLED BY DEFAULT: without HLEO_VOCAB_ENABLED the resolver is None and
  the pipeline behaves exactly as before (pure V1/V3).
- Provider selection via HLEO_VOCAB_PROVIDERS (comma list). Default:
  "rxnorm,mesh,conceptnet,wikidata,umls". LOINC only when its credentials
  exist. UMLS activates only when UMLS_API_KEY is set (available() False
  otherwise — silently skipped). SNOMED CT is registered but inactive until
  licensed.
- A provider that errors, times out or is unavailable never breaks the
  resolution: it is recorded in providers_failed and the others proceed.
- The resolver merges matches per term but never fuses concepts across
  providers: conflicts are preserved (all matches kept, each with its own
  provider + confidence + match_kind).
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from core.vocab.cache import VocabCache
from core.vocab.models import VocabularyMatch, VocabularyResolution

logger = logging.getLogger(__name__)

VOCAB_ENABLED_ENV = "HLEO_VOCAB_ENABLED"
VOCAB_PROVIDERS_ENV = "HLEO_VOCAB_PROVIDERS"
DEFAULT_PROVIDERS = "rxnorm,mesh,conceptnet,wikidata,umls"


def vocab_enabled() -> bool:
    """Catena C is provider-first and ON by default (the terminology layer of
    both Scientific and RWE). Set HLEO_VOCAB_ENABLED=0 to opt out (e.g. fully
    offline test environments) — the pipelines then degrade gracefully with
    no external terminology."""
    return os.getenv(VOCAB_ENABLED_ENV, "1").strip().lower() in {
        "1", "true", "yes", "on"}


def _build_provider(name: str, cache: VocabCache):
    name = name.strip().lower()
    if name == "rxnorm":
        from core.vocab.rxnorm import RxNormProvider
        return RxNormProvider(cache=cache)
    if name == "mesh":
        from core.vocab.mesh import MeSHProvider
        return MeSHProvider(cache=cache)
    if name == "loinc":
        from core.vocab.loinc import LOINCProvider
        return LOINCProvider(cache=cache)
    if name == "conceptnet":
        from core.vocab.conceptnet import ConceptNetProvider
        return ConceptNetProvider(cache=cache)
    if name == "wikidata":
        from core.vocab.wikidata import WikidataProvider
        return WikidataProvider(cache=cache)
    if name == "umls":
        from core.vocab.umls import UMLSProvider
        return UMLSProvider(cache=cache)
    if name in {"snomed", "snomed_ct"}:
        from core.vocab.snomed import SNOMEDCTProvider
        return SNOMEDCTProvider(cache=cache)
    logger.warning("unknown vocabulary provider %r — skipped", name)
    return None


# Process-wide default cache: resolvers created per plan still share cached
# provider responses (same pattern as the orchestrator's class-level cache).
_DEFAULT_CACHE: Optional[VocabCache] = None


def _default_cache() -> VocabCache:
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = VocabCache()
    return _DEFAULT_CACHE


class VocabularyResolver:
    def __init__(self, providers: Optional[list] = None,
                 cache: Optional[VocabCache] = None) -> None:
        self.cache = cache if cache is not None else _default_cache()
        self.providers = providers if providers is not None else self._from_env()

    def _from_env(self) -> list:
        names = os.getenv(VOCAB_PROVIDERS_ENV, DEFAULT_PROVIDERS)
        out = []
        for n in names.split(","):
            if not n.strip():
                continue
            p = _build_provider(n, self.cache)
            if p is not None:
                out.append(p)
        return out

    def active_providers(self) -> List[str]:
        return [p.name for p in self.providers if p.available()]

    def resolve_term(self, term: str, language: str = "en",
                     limit: int = 5) -> VocabularyResolution:
        res = VocabularyResolution(term=term, language=language)
        for p in self.providers:
            if not p.available():
                continue
            res.providers_queried.append(p.name)
            try:
                matches = p.search(term, language=language, limit=limit)
                res.matches.extend(matches)
            except Exception:  # noqa: BLE001 — provider contract is no-raise,
                res.providers_failed.append(p.name)  # but stay paranoid
        return res

    def resolve_terms(self, terms: List[str], language: str = "en",
                      limit: int = 5) -> Dict[str, VocabularyResolution]:
        out: Dict[str, VocabularyResolution] = {}
        for t in terms:
            t = (t or "").strip()
            if len(t) < 3:
                continue
            out[t] = self.resolve_term(t, language=language, limit=limit)
        return out


def build_resolver_from_env() -> Optional[VocabularyResolver]:
    """Return a resolver only when the feature flag is on; None otherwise."""
    if not vocab_enabled():
        return None
    return VocabularyResolver()
