"""
openFDA FAERS collector — official FDA Adverse Event Reporting System API.

Source: https://api.fda.gov/drug/event.json
ToS:     https://open.fda.gov/terms  ("use of data is generally unrestricted")
Auth:    API key optional (free). Without key: 240 req/min, 1000/day per IP.

FAERS reports are spontaneous pharmacovigilance reports — they represent
suspected adverse events reported by consumers/professionals, NOT clinical
evidence of causality. They are classified as evidence_tier="spontaneous_report".

FASE 6B optimisations
---------------------
- HTTP uses core.http_retry.http_get → retry on 429/5xx/Timeout/ConnectionError
  (bounded by collector_max_retries from Global Limits).
- Timeout read from collector_timeout_s (Global Limits) instead of hardcoded.
- HLEO_OPENFDA_MAX_RESULTS default lowered from 1000 to 200 (2 pages), since
  FAERS relevance degrades rapidly beyond the top results and the pipeline
  already applies relevance filtering + relation gate downstream.
  Set HLEO_OPENFDA_MAX_RESULTS=1000 to restore the previous behaviour.
"""
from __future__ import annotations

import logging
import os
import re
import unicodedata
from typing import List, Optional, Tuple

import requests

from core.rwe.models import RWEItem, RWE_SOURCES

logger = logging.getLogger(__name__)

BASE_URL = "https://api.fda.gov/drug/event.json"

STATUS_OK = "ok"
STATUS_NO_RESULTS = "no_results"
STATUS_RATE_LIMITED = "rate_limited"
STATUS_NETWORK_ERROR = "network_error"
STATUS_UNSUPPORTED_QUERY = "unsupported_query"

# openFDA's Lucene parser rejects long quoted phrases containing stray
# apostrophes/quotes and some non-ASCII punctuation. Sanitise before sending.
_MAX_PHRASE_TOKENS = 8

# Default max results per query — lowered to 200 (2 pages of 100) to reduce
# HTTP calls and FDA daily quota consumption without losing meaningful results.
_DEFAULT_MAX_RESULTS = int(os.environ.get("HLEO_OPENFDA_MAX_RESULTS", "200"))


def _lim():
    """Return current Global Limits (cached, 5 s TTL)."""
    try:
        from core.llm_limits import get_limits
        return get_limits()
    except Exception:
        from core.llm_limits import HLEOLimits
        return HLEOLimits()


def sanitize_fda_term(term: str) -> str:
    """Make a query term safe for openFDA's quoted-phrase syntax.

    - NFKD-normalises unicode and strips combining marks (à→a, ﬁ→fi);
    - turns typographic quotes/apostrophes (' ' " ") into plain ASCII, then
      removes every quote/apostrophe (a stray ' breaks the quoted phrase);
    - drops brackets and other punctuation the parser chokes on;
    - collapses whitespace and caps the phrase length (Lucene "Search not
      supported" on very long phrases).
    """
    t = unicodedata.normalize("NFKD", term or "")
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = (t.replace("\u2018", "'").replace("\u2019", "'")
          .replace("\u201c", '"').replace("\u201d", '"'))
    t = t.replace("'", " ").replace('"', " ")
    t = re.sub(r"[\[\](){},;:\\/|@#$%^&*+=<>?~`!]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    tokens = t.split()
    if len(tokens) > _MAX_PHRASE_TOKENS:
        t = " ".join(tokens[:_MAX_PHRASE_TOKENS])
    return t


class OpenFDACollector:
    """Read-only openFDA FAERS adverse-event search."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key or "").strip()

    def search_with_status(
        self,
        query: str,
        limit: Optional[int] = None,
    ) -> Tuple[List[RWEItem], str, str]:
        """
        Search FAERS adverse events for a drug/term.

        HTTP is routed through http_get for retry on 429/5xx/Timeout/
        ConnectionError. The special 400-retry (query shortening) is kept
        manual because 400 is a permanent error in http_get's retry policy.

        Returns: (items, status_code, human_reason)
        """
        from core.http_retry import http_get

        if not query.strip():
            return [], STATUS_NO_RESULTS, "Empty query."

        lim = _lim()
        timeout = lim.collector_timeout_s
        max_retries = lim.collector_max_retries
        backoff_base = lim.backoff_base_s
        backoff_max = lim.backoff_max_s

        # FAERS search across drug generic name + reaction (meddra term).
        # openFDA's search parser expects literal spaces around the OR operator;
        # passing '+OR+' makes requests URL-encode the '+' as '%2B', which openFDA
        # treats as a 404. openFDA is case-insensitive on generic_name/reaction.
        term = sanitize_fda_term(query)
        if not term:
            return [], STATUS_UNSUPPORTED_QUERY, (
                f"Query not sanitisable for openFDA: '{query[:60]}'.")
        search_expr = (
            f'patient.drug.openfda.generic_name:"{term}" '
            f'OR patient.reaction.reactionmeddrapt:"{term}"'
        )
        target = limit if limit is not None else _DEFAULT_MAX_RESULTS
        page_size = max(1, min(target, 100))
        results = []
        skip = 0
        retried_shorter = False
        while len(results) < target:
            params = {"search": search_expr, "limit": page_size, "skip": skip}
            if self.api_key:
                params["api_key"] = self.api_key
            try:
                resp = http_get(
                    BASE_URL,
                    params=params,
                    timeout=timeout,
                    max_retries=max_retries,
                    backoff_base_s=backoff_base,
                    backoff_max_s=backoff_max,
                )
            except requests.HTTPError as exc:
                code = exc.response.status_code if exc.response is not None else 0
                if code == 429:
                    # All retries exhausted on 429
                    return [], STATUS_RATE_LIMITED, "openFDA rate limit reached. Retry later."
                if code == 404:
                    if not results:
                        return [], STATUS_NO_RESULTS, f"No FAERS reports matched '{query}'."
                    break
                if code == 400 and not retried_shorter:
                    # Query not supported as-is: retry ONCE with a shorter compatible
                    # variant (first 4 tokens only). 400 is permanent in http_get so
                    # we handle the one manual retry here.
                    retried_shorter = True
                    short = " ".join(term.split()[:4])
                    logger.warning(
                        "openFDA HTTP 400 (unsupported query) for '%s' — "
                        "retrying with '%s'", term[:80], short[:80])
                    term = short
                    search_expr = (
                        f'patient.drug.openfda.generic_name:"{term}" '
                        f'OR patient.reaction.reactionmeddrapt:"{term}"'
                    )
                    continue
                if code == 400:
                    return [], STATUS_UNSUPPORTED_QUERY, (
                        f"openFDA HTTP 400 (unsupported query) for '{term[:80]}'.")
                return [], STATUS_NETWORK_ERROR, f"openFDA HTTP {code}."
            except requests.exceptions.RequestException as exc:
                logger.warning(f"openFDA network error: {exc}")
                return [], STATUS_NETWORK_ERROR, str(exc)

            try:
                data = resp.json()
            except ValueError as exc:
                return [], STATUS_NETWORK_ERROR, f"openFDA JSON parse error: {exc}"
            batch = data.get("results") or []
            results.extend(batch)
            if limit is not None or not batch or len(batch) < page_size:
                break
            skip += len(batch)
        if limit is not None:
            results = results[:limit]
        if not results:
            return [], STATUS_NO_RESULTS, f"No FAERS reports matched '{query}'."

        items: List[RWEItem] = []
        meta = RWE_SOURCES["openfda_faers"]
        query_upper = query.strip().upper()
        for rec in results:
            patient = rec.get("patient", {})
            # Reactions (meddra preferred terms)
            reactions = [r.get("reactionmeddrapt", "") for r in patient.get("reaction", [])]
            reaction_text = "; ".join([r for r in reactions if r])

            # Drugs — prefer the queried drug as treatment (it matched the search
            # but may not be the first drug listed in the report)
            drugs = patient.get("drug", [])
            treatment = None
            for d in drugs:
                openfda = d.get("openfda", {}) or {}
                generics = openfda.get("generic_name") or []
                if any(query_upper in (g or "").upper() for g in generics):
                    treatment = generics[0]
                    break
            if not treatment:
                # fall back to first available generic/medicinal product
                for d in drugs:
                    openfda = d.get("openfda", {}) or {}
                    generics = openfda.get("generic_name") or []
                    if generics:
                        treatment = generics[0]
                        break
                    if d.get("medicinalproduct"):
                        treatment = d["medicinalproduct"]
                        break

            # Date
            date = rec.get("receiptdate") or rec.get("receivedate")

            # Outcome flags
            outcomes = patient.get("patient", {}).get("outcome", [])
            serious = rec.get("serious", "") == "1"

            items.append(RWEItem(
                source="openfda_faers",
                source_type=meta["source_type"],
                evidence_tier=meta["evidence_tier"],
                collection_method=meta["collection_method"],
                external_id=str(rec.get("safetyreportid", "")),
                source_url=(
                    f"https://api.fda.gov/drug/event.json?search=safetyreportid:"
                    f"{rec.get('safetyreportid','')}"
                ),
                title=f"FAERS report {rec.get('safetyreportid','')} — {reaction_text[:80]}",
                text=reaction_text or "(no reaction term)",
                date=date,
                language="en",
                topic=query,
                treatment=treatment,
                condition=None,
                experience_type="adverse_event",
                privacy_status="anonymous",
                metadata={
                    "serious": serious,
                    "patient_age": patient.get("patient", {}).get("patientonsetage"),
                    "patient_sex": patient.get("patient", {}).get("patientsex"),
                    "outcomes": outcomes,
                    "n_drugs": len(drugs),
                },
            ))

        return items, STATUS_OK, f"Retrieved {len(items)} FAERS report(s)."

    def search(self, query: str, limit: Optional[int] = None) -> List[RWEItem]:
        """Silent-fail wrapper for pipeline use."""
        items, status, reason = self.search_with_status(query, limit=limit)
        if status != STATUS_OK:
            logger.info(f"openFDA silent-fail [{status}]: {reason}")
        return items
