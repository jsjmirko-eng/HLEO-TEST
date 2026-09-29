"""Temporary, opt-in full-text content discovery for Scientific Search."""

from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import requests


CONTENT_SEARCH_EXPERIMENT_ENV = "CONTENT_SEARCH_EXPERIMENT"
DEFAULT_CONTEXT_CHARS = 500
DEFAULT_TIMEOUT_S = 20
DEFAULT_MAX_WORKERS = 8

_GENERIC_WORDS = frozenset(
    "a an and associated by cause caused causes con da de del due for il in la lead leads may of on or per the to with"
    .split()
)


@dataclass(frozen=True)
class ContentTerm:
    """A searchable term derived from the relation, not a HLEO vocabulary."""

    text: str
    concept: str
    source: str


def experiment_enabled() -> bool:
    return os.getenv(CONTENT_SEARCH_EXPERIMENT_ENV, "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _values(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key in ("normalized", "term"):
            text = str(value.get(key) or "").strip()
            if text:
                yield text
        site = value.get("site")
        if isinstance(site, dict):
            yield from _values(site)
    elif isinstance(value, str) and value.strip():
        yield value.strip()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\wα-ωÀ-ÿ]+(?:[-'][\wα-ωÀ-ÿ]+)?", text.casefold())


def _add_term(out: dict[tuple[str, str], ContentTerm], text: str,
              concept: str, source: str) -> None:
    normalized = " ".join(text.casefold().split()).strip(" .,:;()[]")
    if len(normalized) < 3:
        return
    if not any(char.isalpha() for char in normalized):
        return
    out.setdefault((concept, normalized), ContentTerm(normalized, concept, source))


def derive_content_terms(relation: Any) -> list[ContentTerm]:
    """Derive searchable terms from ClinicalRelation fields and phrases.

    The n-grams are a mechanical representation of the relation/query text;
    no clinical synonym list is maintained here.
    """

    terms: dict[tuple[str, str], ContentTerm] = {}
    agent = getattr(relation, "agent", {}) or {}
    event = getattr(relation, "event", {}) or {}
    manifestation = getattr(relation, "manifestation", {}) or {}
    site = getattr(relation, "anatomical_site", {}) or {}
    for value in _values(agent):
        _add_term(terms, value, "agent", "clinical_relation.agent")
    for value in (*_values(event), *_values(manifestation), *_values(site)):
        _add_term(terms, value, "manifestation", "clinical_relation")

    phrases = [str(p).strip() for p in (getattr(relation, "relation_phrases", []) or []) if str(p).strip()]
    scientific_query = str(getattr(relation, "scientific_query", "") or "").strip()
    source_texts = [(phrase, "relation_phrase") for phrase in phrases]
    if scientific_query:
        source_texts.append((scientific_query, "scientific_query"))

    agent_tokens = {
        token for term in terms.values() if term.concept == "agent"
        for token in _tokens(term.text)
    }
    for text, source in source_texts:
        words = _tokens(text)
        for size in (2, 3, 4):
            for start in range(0, max(0, len(words) - size + 1)):
                ngram_words = words[start:start + size]
                ngram = " ".join(ngram_words)
                ngram_tokens = set(ngram_words)
                meaningful = ngram_tokens - _GENERIC_WORDS - agent_tokens
                if len(meaningful) < 2 or ngram_words[0] in _GENERIC_WORDS or ngram_words[-1] in _GENERIC_WORDS:
                    continue
                if ngram_tokens & agent_tokens:
                    continue
                _add_term(terms, ngram, "manifestation", source)

    return list(terms.values())


def _article_identifier(article: Any) -> Optional[str]:
    metadata = getattr(article, "metadata", {}) or {}
    for value in (
        metadata.get("pmcid"),
        metadata.get("pmc_id"),
        getattr(article, "pmid", None),
        metadata.get("id"),
        getattr(article, "doi", None),
    ):
        if value:
            return str(value).strip()
    return None


def _resolve_pmcid(identifier: str, timeout_s: float) -> Optional[str]:
    if identifier.upper().startswith("PMC"):
        return identifier.upper()
    response = requests.get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={"query": f"EXT_ID:{identifier}", "format": "json", "resultType": "core", "pageSize": 1},
        timeout=timeout_s,
    )
    response.raise_for_status()
    rows = (response.json().get("resultList", {}).get("result", []) or [])
    if isinstance(rows, dict):
        rows = [rows]
    return (rows[0].get("pmcid") or "").strip() or None if rows else None


def _parse_full_text(xml_text: str) -> tuple[str, list[dict]]:
    root = ET.fromstring(xml_text)
    paragraphs: list[dict] = []

    def walk(node: ET.Element, section: str = "") -> None:
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "sec":
            title = next((child for child in node if child.tag.rsplit("}", 1)[-1] == "title"), None)
            if title is not None:
                section = " ".join("".join(title.itertext()).split())
        if tag == "p":
            text = " ".join("".join(node.itertext()).split())
            if text:
                paragraphs.append({"text": text, "section": section})
            return
        for child in node:
            walk(child, section)

    walk(root)
    return "\n\n".join(item["text"] for item in paragraphs), paragraphs


def _paragraph_spans(paragraphs: list[dict]) -> list[tuple[int, int, dict]]:
    spans = []
    cursor = 0
    for index, paragraph in enumerate(paragraphs):
        start = cursor
        end = start + len(paragraph["text"])
        spans.append((start, end, {**paragraph, "index": index}))
        cursor = end + 2
    return spans


def _find_span(spans: list[tuple[int, int, dict]], position: int) -> dict:
    for start, end, paragraph in spans:
        if start <= position < end:
            return paragraph
    return {"index": None, "section": ""}


def _scan_text(text: str, terms: list[ContentTerm], context_chars: int,
               paragraphs: Optional[list[dict]] = None) -> dict:
    normalized = text.casefold()
    spans = []
    # The normalized search text is only an index; original text is retained.
    paragraphs = paragraphs or [{"text": part, "section": ""} for part in text.split("\n\n")]
    paragraph_spans = _paragraph_spans(paragraphs)
    for term in terms:
        for match in re.finditer(re.escape(term.text), normalized):
            paragraph = _find_span(paragraph_spans, match.start())
            spans.append({
                "term": term.text,
                "concept": term.concept,
                "source": term.source,
                "start": match.start(),
                "end": match.end(),
                "paragraph_index": paragraph.get("index"),
                "section": paragraph.get("section", ""),
                "context": text[max(0, match.start() - context_chars):min(len(text), match.end() + context_chars)],
            })
    spans.sort(key=lambda item: (item["start"], item["end"], item["term"]))
    agent_matches = [item for item in spans if item["concept"] == "agent"]
    manifestation_matches = [item for item in spans if item["concept"] == "manifestation"]
    distances = []
    for agent in agent_matches:
        for manifestation in manifestation_matches:
            distances.append({
                "distance_characters": min(
                    abs(agent["start"] - manifestation["end"]),
                    abs(manifestation["start"] - agent["end"]),
                ),
                "same_paragraph": agent["paragraph_index"] == manifestation["paragraph_index"],
                "same_section": bool(agent["section"] and agent["section"] == manifestation["section"]),
                "agent_term": agent["term"],
                "manifestation_term": manifestation["term"],
            })
    matched_terms = sorted({item["term"] for item in spans})
    concept_names = sorted({item["concept"] for item in spans})
    return {
        "matched_terms": matched_terms,
        "match_count": len(spans),
        "match_positions": spans,
        "contexts": [item["context"] for item in spans],
        "relation_proximity": {
            "all_concepts_found": {name: name in concept_names for name in ("agent", "manifestation")},
            "minimum_distance_characters": min((item["distance_characters"] for item in distances), default=None),
            "same_paragraph": any(item["same_paragraph"] for item in distances),
            "same_section": any(item["same_section"] for item in distances),
            "pairs": distances,
        },
    }


def _scan_article(article: Any, terms: list[ContentTerm], context_chars: int,
                  timeout_s: float) -> tuple[dict, dict]:
    """Download, parse, and scan one article; return result and phase timings."""

    article_started = time.perf_counter()
    identifier = _article_identifier(article)
    base = {
        "article_id": identifier,
        "title": getattr(article, "title", ""),
        "source": getattr(article, "source", ""),
        "full_text_available": False,
        "full_text_status": "NOT_AVAILABLE",
        "matched_terms": [],
        "match_count": 0,
        "match_positions": [],
        "contexts": [],
        "relation_proximity": {
            "all_concepts_found": {"agent": False, "manifestation": False},
            "minimum_distance_characters": None,
            "same_paragraph": False,
            "same_section": False,
            "pairs": [],
        },
        "error": "",
    }
    timing = {"download": 0.0, "parse": 0.0, "search": 0.0}
    if not identifier:
        base["error"] = "no_public_identifier"
        base["timing_ms"] = round((time.perf_counter() - article_started) * 1000, 3)
        return base, timing
    try:
        download_started = time.perf_counter()
        pmcid = _resolve_pmcid(identifier, timeout_s)
        if not pmcid:
            timing["download"] = time.perf_counter() - download_started
            base["error"] = "full_text_not_available"
            base["timing_ms"] = round((time.perf_counter() - article_started) * 1000, 3)
            return base, timing
        response = requests.get(
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML",
            timeout=timeout_s,
        )
        timing["download"] = time.perf_counter() - download_started
        response.raise_for_status()
        base["full_text_available"] = True
        base["full_text_status"] = "AVAILABLE"
        base["pmcid"] = pmcid
        parse_started = time.perf_counter()
        text, paragraphs = _parse_full_text(response.text)
        timing["parse"] = time.perf_counter() - parse_started
        search_started = time.perf_counter()
        base.update(_scan_text(text, terms, context_chars, paragraphs))
        timing["search"] = time.perf_counter() - search_started
        base["paragraph_count"] = len(paragraphs)
    except Exception as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
    base["timing_ms"] = round((time.perf_counter() - article_started) * 1000, 3)
    return base, timing


def run_content_search(articles: list, relation: Any,
                       context_chars: int = DEFAULT_CONTEXT_CHARS,
                       timeout_s: float = DEFAULT_TIMEOUT_S) -> dict:
    """Discover relation evidence in public full text without an LLM."""

    started = time.perf_counter()
    terms = derive_content_terms(relation)
    max_workers = max(1, int(os.getenv("CONTENT_SEARCH_MAX_WORKERS", str(DEFAULT_MAX_WORKERS))))
    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(articles)))) as pool:
        scanned = list(pool.map(
            lambda article: _scan_article(article, terms, context_chars, timeout_s),
            articles,
        ))
    results = [result for result, _timing in scanned]
    download_seconds = sum(timing["download"] for _result, timing in scanned)
    parse_seconds = sum(timing["parse"] for _result, timing in scanned)
    search_seconds = sum(timing["search"] for _result, timing in scanned)

    available = [item for item in results if item["full_text_available"]]
    matched = [item for item in available if item["match_count"]]
    all_concepts = [
        item for item in available
        if all(item["relation_proximity"]["all_concepts_found"].values())
    ]
    near = [
        item for item in all_concepts
        if item["relation_proximity"]["minimum_distance_characters"] is not None
        and item["relation_proximity"]["minimum_distance_characters"] <= context_chars
    ]
    total_search_seconds = time.perf_counter() - started
    return {
        "enabled": True,
        "context_chars": context_chars,
        "derived_terms": [term.__dict__ for term in terms],
        "total_results": len(articles),
        "full_text_available": len(available),
        "full_text_not_available": len(results) - len(available),
        "articles_with_match": len(matched),
        "articles_with_all_concepts": len(all_concepts),
        "articles_with_near_concepts": len(near),
        "total_occurrences": sum(item["match_count"] for item in results),
        "timing": {
            "download_seconds": round(download_seconds, 6),
            "parse_seconds": round(parse_seconds, 6),
            "text_search_seconds": round(search_seconds, 6),
            "total_seconds": round(total_search_seconds, 6),
            "average_seconds_per_article": round(total_search_seconds / len(articles), 6) if articles else 0.0,
        },
        "articles": results,
    }
