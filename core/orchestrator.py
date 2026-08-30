"""
AI Research Orchestrator — Feature 001 (v1)
============================================
Intercepts every user query before it reaches the collectors.

v1 responsibilities
-------------------
  1. Detect the language of the query.
  2. Translate to scientific-grade English when the source language is not English.
  3. Return a normalised OrchestrationResult consumed by all search endpoints.

Designed for incremental extension:
  v2 — MeSH/UMLS term injection & query expansion
  v3 — session-aware context blending
  v4 — multi-language result fusion & re-ranking
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class OrchestrationResult:
    """
    Carries the original query, the normalised query actually sent to
    the collectors, and full provenance metadata.
    """
    original_query:    str
    search_query:      str    # query sent to PubMed / EuropePMC / ClinicalTrials
    detected_language: str    # ISO-639-1 code, e.g. "it", "en", "de"
    translation_applied: bool # True when search_query differs from original_query
    confidence: float  = 1.0  # reserved for future per-step confidence scores
    metadata:   dict   = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "original_query":      self.original_query,
            "search_query":        self.search_query,
            "detected_language":   self.detected_language,
            "translation_applied": self.translation_applied,
            "confidence":          self.confidence,
        }


# ── Orchestrator ──────────────────────────────────────────────────────────────

class QueryOrchestrator:
    """
    Stateless query normaliser.  Safe to instantiate once at module level
    and reuse across requests.

    Extension points
    ----------------
    Override or extend `_run()` to add pipeline steps (expansion, context
    injection, etc.) without changing the public `process()` interface.
    """

    # Process-lifetime cache: avoids repeat GPT calls for identical queries.
    # Key: MD5(query).  Replace with Redis/DB cache in v3+.
    _cache: dict[str, OrchestrationResult] = {}

    def __init__(self) -> None:
        self._client = None
        try:
            from core.llm_provider import build_provider
            self._client = build_provider()
            if self._client is not None:
                logger.debug(
                    "QueryOrchestrator: LLM client initialised (%s).",
                    self._client.name,
                )
        except Exception as exc:
            logger.warning(f"QueryOrchestrator: cannot init LLM client — {exc}")

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, query: str) -> OrchestrationResult:
        """
        Main entry point.

        Returns an OrchestrationResult with:
          - search_query  : the English query to send to the collectors
          - detected_language : ISO-639-1 code of the original query
          - translation_applied : whether a translation was performed

        Always succeeds — falls back to the original query on any error.
        """
        q = query.strip()
        if not q:
            return self._passthrough(q, "und")

        cache_key = hashlib.md5(q.encode()).hexdigest()
        if cache_key in self._cache:
            logger.debug(f"QueryOrchestrator cache hit: '{q[:50]}'")
            return self._cache[cache_key]

        result = self._run(q)
        self._cache[cache_key] = result
        return result

    # ── Internal pipeline (v1: detect + translate) ────────────────────────────

    def _run(self, query: str) -> OrchestrationResult:
        """Execute the v1 pipeline: language detection + translation."""
        if self._client is None:
            logger.info(
                "QueryOrchestrator: no OpenAI client — "
                "passing query through unchanged."
            )
            return self._passthrough(query, "und")

        try:
            lang, english_query = self._detect_and_translate(query)
        except Exception as exc:
            logger.warning(
                f"QueryOrchestrator: detect/translate failed ({exc}) — "
                "falling back to original query."
            )
            return self._passthrough(query, "und")

        # Already English or translation returned the same string
        if lang == "en" or english_query.lower() == query.lower():
            return self._passthrough(query, lang if lang else "en")

        logger.info(
            f"QueryOrchestrator: [{lang.upper()}] '{query[:60]}' "
            f"→ [EN] '{english_query[:60]}'"
        )
        return OrchestrationResult(
            original_query=query,
            search_query=english_query,
            detected_language=lang,
            translation_applied=True,
        )

    def _detect_and_translate(self, query: str) -> tuple[str, str]:
        """
        Single gpt-4o-mini call: detect language and translate to scientific English.

        Returns
        -------
        (iso_code, english_query)
            iso_code      — ISO-639-1 language code, e.g. "it", "en"
            english_query — the query in scientific English
        """
        prompt = (
            "You are a scientific language normaliser for a clinical research "
            "search engine.\n"
            "Given a user query, return a JSON object with exactly two keys:\n"
            '  "lang": the ISO-639-1 language code of the input query '
            '(e.g. "en", "it", "de", "fr")\n'
            '  "query_en": the query translated into concise scientific English.\n'
            "\nRules:\n"
            "  - If the query is already in English, copy it verbatim into query_en.\n"
            "  - Preserve all drug names, molecule names, and medical terms exactly.\n"
            "  - Translate meaning — do NOT add synonyms, expand abbreviations, or "
            "change scope.\n"
            "  - Output only the JSON object; no markdown, no commentary.\n"
            f"\nQuery: {query}"
        )

        from core.llm_guard import call_llm_json
        data = call_llm_json(
            self._client,
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",
            max_tokens=150,
            operation="orchestrator_detect_translate",
        )
        lang     = str(data.get("lang", "und")).lower().strip()[:10]
        query_en = str(data.get("query_en", query)).strip()

        if not query_en:
            query_en = query

        return lang, query_en

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _passthrough(self, query: str, lang: str) -> OrchestrationResult:
        """Return a no-op result that preserves the original query."""
        return OrchestrationResult(
            original_query=query,
            search_query=query,
            detected_language=lang,
            translation_applied=False,
        )
