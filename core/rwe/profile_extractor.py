"""
FASE 13 — RWE profile extractor.

Extracts a structured RWE profile from an RWE item (community forum post or
FAERS record). Uses the central llm_guard for the LLM call.

The RWE profile is distinct from the scientific ClinicalProfile:
  - community_forum posts → experience_summary, key_quotes, extraction_confidence
  - pharmacovigilance (FAERS) → adverse_events, seriousness, outcome
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


_RWE_PROFILE_PROMPT = """You are a patient-experience analyst. Extract a structured RWE profile from the source below.

Return ONLY a JSON object with these keys:
{
  "experience_summary": "1-2 neutral sentences summarising what the person/record reports",
  "key_quotes": ["verbatim phrase 1", "phrase 2", "phrase 3"],
  "treatment": "drug/intervention mentioned, or null",
  "condition": "condition discussed, or null",
  "adverse_events": ["event1", "event2"],
  "outcome": "improved|worsened|stable|unknown",
  "experience_type": "discussion|adverse_event|outcome_report|question",
  "extraction_confidence": "high|medium|low",
  "demographics": {"age": null, "sex": null, "country": null}
}

Rules:
- Extract ONLY what is explicitly stated. Never infer.
- key_quotes: verbatim phrases from the text (max 3, each ≤120 chars).
- For FAERS records: experience_type = adverse_event, outcome from the record.
- For forum posts: experience_type based on what the user describes.
- extraction_confidence: 'high' if clear personal health journey, 'low' if off-topic.
- Do NOT store usernames or PII beyond age/sex/country if stated.
- Use null for missing scalars; [] for missing lists.
"""


class RWEProfileExtractor:
    """Extracts structured profiles from RWE items via LLM."""

    MODEL = "gpt-4o-mini"

    def __init__(self):
        self._client = None
        try:
            from core.llm_provider import build_provider
            self._client = build_provider()
        except Exception as exc:
            logger.warning("RWEProfileExtractor: LLM provider init failed — %s", exc)

    def extract(self, *, title: str, text: str, source: str,
                source_type: str = "", treatment: str = "",
                condition: str = "") -> dict:
        """Extract a structured RWE profile.

        Returns a dict with the profile fields. Raises RuntimeError if no key,
        or LLMCallError/QuotaExhaustedError from the guard.
        """
        if self._client is None:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        from core.llm_guard import call_llm_json

        content = f"Title: {title}\n\n{text or '(no text)'}".strip()
        if len(content) > 3000:
            content = content[:2970] + "\n…[truncated]"

        meta = []
        if source:
            meta.append(f"Source: {source}")
        if source_type:
            meta.append(f"Source type: {source_type}")
        if treatment:
            meta.append(f"Treatment: {treatment}")
        if condition:
            meta.append(f"Condition: {condition}")
        meta_str = "\n".join(meta)

        prompt = (
            f"{_RWE_PROFILE_PROMPT}\n\n"
            f"=== SOURCE ===\n{meta_str}\n\n{content}"
        )

        return call_llm_json(
            self._client,
            messages=[{"role": "user", "content": prompt}],
            model=self.MODEL,
            temperature=0.0,
            max_tokens=700,
            operation="rwe_profile_extract",
        )
