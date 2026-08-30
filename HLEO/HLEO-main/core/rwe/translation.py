"""
RWE robust translation — plain-text IT→EN query translation.

Why this module exists
----------------------
The shared QueryOrchestrator translates via ONE structured-JSON LLM call.
Some providers (observed live: Groq gpt-oss-120b) fail JSON-mode generation
repeatedly ("json_validate_failed" x5), leaving the RWE query untranslated
in Italian — which then breaks retrieval (openFDA rejects long non-English
phrases with apostrophes; EN-oriented forums yield nothing).

This module is the RWE-only robust chain, independent of complex JSON:

  1. plain-text LLM translation (no response_format, output = raw English);
  2. one retry with a minimal prompt;
  3. deterministic fallback built from provider-recognised entities
     (Catena C canonical EN terms — no new hardcoded dictionaries);
  4. if everything fails, return the original query with method "none"
     (honest, logged — never a silent Italian query passed off as English).

The scientific pipeline is untouched: the shared orchestrator is not modified;
this chain runs only inside the RWE query engine.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_MODEL = "gpt-4o-mini"

_PROMPT_FULL = (
    "Translate the following user search query into concise English for a "
    "medical search engine.\n"
    "Rules:\n"
    "  - Preserve medical terms exactly.\n"
    "  - Write drug names in their international nonproprietary name (INN): "
    "localized forms must be normalised (isotretinoina → isotretinoin, "
    "paracetamolo → paracetamol).\n"
    "  - Output ONLY the English translation: one line, plain text.\n"
    "  - No JSON, no quotes, no markdown, no commentary, no explanation.\n"
    "\nQuery: {query}"
)

_PROMPT_MINIMAL = (
    "Translate to English. Reply with the translation only.\n\n{query}"
)


@dataclass
class TranslationResult:
    english_query: str
    method: str          # "llm" | "llm_retry" | "deterministic" | "none"
    error: str = ""


def _validate(candidate: str, original: str) -> str:
    """Accept only a plausible plain-text English translation."""
    t = (candidate or "").strip().strip('"').strip("'").strip()
    t = re.sub(r"\s+", " ", t)
    if not t:
        return ""
    if t.lower() == (original or "").strip().lower():
        return ""                       # echo of the source = not translated
    if len(t) > 3 * max(1, len(original or "")):
        return ""                       # rambled / hallucinated extra content
    if not re.search(r"[a-zA-Z]", t):
        return ""
    return t


def _llm_translate(client, query: str, prompt: str) -> str:
    from core.llm_guard import call_llm
    out = call_llm(
        client,
        messages=[{"role": "user", "content": prompt.format(query=query)}],
        model=_MODEL,
        temperature=0,
        # Reasoning models (e.g. gpt-oss-120b) spend part of the budget on
        # internal reasoning before producing content — 120 tokens was
        # observed to yield EMPTY translations on Groq. 512 leaves headroom.
        max_tokens=512,
        json_mode=False,
        operation="rwe_translate",
    )
    # Tolerate a model that still answers with a JSON object: extract the
    # value if parseable, otherwise treat as plain text and let validation
    # decide.
    txt = (out or "").strip()
    if txt.startswith("{"):
        import json
        try:
            data = json.loads(txt)
            if isinstance(data, dict):
                for key in ("query_en", "translation", "english", "en"):
                    if isinstance(data.get(key), str):
                        return data[key]
        except ValueError:
            pass
    return txt


def _deterministic_fallback(query: str, lang: str) -> str:
    """English keyword query from provider-recognised entities (Catena C).

    Resolves the source-language text through the vocabulary providers and
    joins the canonical (English) entity names — e.g. "isotretinoina" →
    "isotretinoin" (RxNorm), "dolore articolare" → "joint pain" (ConceptNet/
    MeSH when mapped). Returns "" when no entity is recognised.
    """
    try:
        from core.vocab.entities import recognize
        from core.vocab.resolver import build_resolver_from_env
        resolver = build_resolver_from_env()
        if resolver is None:
            return ""
        rec = recognize(query, lang or "en", resolver)
        canonicals = []
        for _etype, canonical, _conf in rec.entities:
            if canonical and canonical.lower() not in {
                    c.lower() for c in canonicals}:
                canonicals.append(canonical)
        return " ".join(canonicals)
    except Exception as exc:  # noqa: BLE001 — fallback must never raise
        logger.info("RWE deterministic translation fallback failed (%s)",
                    type(exc).__name__)
        return ""


def translate_for_rwe(
    query: str,
    lang: str,
    client=None,
) -> TranslationResult:
    """Robust RWE translation chain. Never raises; never returns a silent
    untranslated non-English query without saying so (method tells the truth).
    """
    q = (query or "").strip()
    if not q or (lang or "und").lower() in {"en", "und", ""}:
        return TranslationResult(english_query=q, method="none")

    if client is None:
        try:
            from core.llm_provider import build_provider
            client = build_provider()
        except Exception as exc:  # noqa: BLE001
            logger.info("RWE translation: no LLM client (%s)", type(exc).__name__)
            client = None

    if client is not None:
        for prompt, method in ((_PROMPT_FULL, "llm"), (_PROMPT_MINIMAL, "llm_retry")):
            try:
                candidate = _validate(_llm_translate(client, q, prompt), q)
            except Exception as exc:  # noqa: BLE001 — try the next chain step
                logger.info("RWE translation %s failed (%s)", method,
                            type(exc).__name__)
                candidate = ""
            if candidate:
                return TranslationResult(english_query=candidate, method=method)

    fallback = _validate(_deterministic_fallback(q, lang), q)
    if fallback:
        return TranslationResult(english_query=fallback, method="deterministic")

    logger.warning("RWE translation: all methods failed for lang=%s — "
                   "keeping original query", lang)
    return TranslationResult(english_query=q, method="none",
                             error="translation_unavailable")
