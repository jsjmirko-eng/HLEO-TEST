"""
ConceptNet WEB fallback — html frontend (conceptnet.io) used ONLY when the
primary API (api.conceptnet.io) is unreachable (502/503/504, timeout,
connection error).

It returns the SAME VocabularyMatch contract as the API provider
(core/vocab/conceptnet.py): synonyms, cross-language translations, related
concepts, language and canonical term — parsed from the public node pages.
No local database, no permanent terminology cache, no hardcoded vocabulary:
the shared per-provider VocabCache (TTL) of the base class is the only cache.
"""
from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple

import requests

from core.vocab.models import VocabularyMatch

logger = logging.getLogger(__name__)

WEB = "https://conceptnet.io"
WEB_TIMEOUT = 10

# Same conservative relation → match_kind mapping as the API provider.
_REL_KIND = {
    "Synonym": "synonym",
    "FormOf": "orthographic_variant",
    "RelatedTo": "related_concept",
    "IsA": "related_concept",
    "PartOf": "related_concept",
    "Causes": "related_concept",
}
# Relations rendered by the web frontend that we actually consume.
_WANTED_RELS = set(_REL_KIND)


class _NodePageParser(HTMLParser):
    """Parse a conceptnet.io node page into canonical/lang/relations.

    Structure (verified live 2026-08-24):
      <h1 class="term lang-XX"> ... canonical term ... </h1>
      <h2><a href="...?rel=/r/RelName&limit=1000">...</a></h2>
      <li class="term lang-YY"> ... <a href="/c/YY/term">term</a> ... </li>
    A missing node renders <h1 class="error">Not found</h1>.
    """

    def __init__(self) -> None:
        super().__init__()
        self.canonical: str = ""
        self.language: str = "en"
        self.not_found: bool = False
        self.relations: Dict[str, List[Tuple[str, str]]] = {}
        self._rel: Optional[str] = None
        self._in_h2 = False
        self._in_term = False
        self._term_lang: Optional[str] = None
        self._buf: List[str] = []
        self._in_h1 = False
        self._skip_span = False

    @staticmethod
    def _lang_of(cls: str) -> Optional[str]:
        m = re.search(r"lang-([a-z]{2,3}|none|mul)\b", cls or "")
        return m.group(1) if m else None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "") or ""
        if tag == "h1":
            if "error" in cls:
                self.not_found = True
            elif "term" in cls:
                self._in_h1 = True
                self._in_term = True
                self._buf = []
                self.language = self._lang_of(cls) or "en"
        elif tag == "h2":
            self._in_h2 = True
        elif tag == "a" and self._in_h2:
            href = a.get("href", "") or ""
            m = re.search(r"[?&]rel=/r/([A-Za-z]+)", href)
            if m and m.group(1) in _WANTED_RELS:
                self._rel = m.group(1)
                self.relations.setdefault(self._rel, [])
        elif tag == "li" and "term" in cls:
            self._in_term = True
            self._term_lang = self._lang_of(cls)
            self._buf = []
        elif tag == "span" and "language" in cls and self._in_term:
            # the <span class="language">en</span> badge is NOT part of the term
            self._skip_span = True

    def handle_endtag(self, tag):
        if tag == "h2":
            self._in_h2 = False
        elif tag == "span":
            self._skip_span = False
        elif tag in ("li", "h1") and self._in_term:
            text = re.sub(r"\s+", " ", " ".join(self._buf)).strip()
            # strip the trailing edge-link glyph and sense markers ("( n )",
            # "(n, animal)" — spacing varies across node pages)
            text = text.replace("➜", "").strip()
            text = re.sub(r"\s*\(\s*[a-z]+\s*(?:,[^)]*)?\)\s*$", "", text).strip()
            if tag == "h1":
                self.canonical = text
                self._in_h1 = False
            elif self._rel and text:
                self.relations[self._rel].append((self._term_lang or "", text))
            self._in_term = False

    def handle_data(self, data):
        if self._in_term and not self._skip_span:
            self._buf.append(data)


def parse_node_page(html: str) -> dict:
    """Parse a conceptnet.io node page.

    Returns {"canonical", "language", "relations": {rel: [(lang, term)]},
             "not_found": bool}."""
    parser = _NodePageParser()
    parser.feed(html or "")
    return {
        "canonical": parser.canonical,
        "language": parser.language,
        "relations": parser.relations,
        "not_found": parser.not_found,
    }


def _node_url(term: str, language: str) -> str:
    return f"{WEB}/c/{language}/{term.strip().lower().replace(' ', '_')}"


def fetch_node_page(term: str, language: str,
                    timeout: int = WEB_TIMEOUT) -> Optional[dict]:
    """Fetch + parse the web node page. Follows redirects (requests default).

    Returns the parsed page dict, or None when the web frontend itself is
    unreachable (timeout / connection error / 5xx) — i.e. provider
    unavailable, NOT "node does not exist"."""
    try:
        resp = requests.get(_node_url(term, language), timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — timeout/conn error → unavailable
        logger.info("conceptnet web unreachable for %r (%s)", term,
                    type(exc).__name__)
        return None
    if resp.status_code >= 500:
        logger.info("conceptnet web %s for %r", resp.status_code, term)
        return None
    if resp.status_code >= 400:
        return {"canonical": "", "language": language, "relations": {},
                "not_found": True}
    return parse_node_page(resp.text)


def search_web(term: str, language: str, limit: int = 10) -> Optional[List[VocabularyMatch]]:
    """Web fallback search. Returns:
      - list of VocabularyMatch (same contract as the API provider),
      - [] when the node does not exist (NOT found — nothing invented),
      - None when the web frontend is unreachable (provider unavailable)."""
    page = fetch_node_page(term, language)
    if page is None:
        return None
    if page.get("not_found"):
        return []

    preferred = (page.get("canonical") or term).strip().lower()
    node = _node_url(preferred, page.get("language") or language).replace(WEB, "")
    synonyms: List[str] = []
    translations: List[Tuple[str, str]] = []
    variants: List[str] = []
    related: List[str] = []
    src_lang = (language or "en").lower()

    for rel, terms in (page.get("relations") or {}).items():
        kind = _REL_KIND.get(rel)
        if not kind:
            continue
        for lang, label in terms:
            label = (label or "").strip()
            if not label or label.lower() == preferred:
                continue
            if rel == "Synonym":
                if lang and lang != src_lang:
                    translations.append((label, lang))
                else:
                    synonyms.append(label)
            elif rel == "FormOf":
                variants.append(label)
            else:
                related.append(label)

    if not (synonyms or translations or variants or related):
        return []

    matches: List[VocabularyMatch] = []
    base = dict(provider="conceptnet", concept_id=node,
                preferred_term=preferred, semantic_group="general",
                source_url=f"{WEB}{node}")
    if synonyms:
        matches.append(VocabularyMatch(
            **base, synonyms=synonyms[:10], language=src_lang,
            confidence=0.8, match_kind="synonym",
            metadata={"relation": "Synonym", "via": "web"}))
    for label, lang in translations[:10]:
        matches.append(VocabularyMatch(
            **dict(base, preferred_term=label),
            synonyms=[], language=lang,
            confidence=0.75, match_kind="translation",
            metadata={"source_term": preferred, "source_language": src_lang,
                      "via": "web"}))
    if variants:
        matches.append(VocabularyMatch(
            **base, synonyms=variants[:5], language=src_lang,
            confidence=0.7, match_kind="orthographic_variant",
            metadata={"relation": "FormOf", "via": "web"}))
    if related:
        matches.append(VocabularyMatch(
            **base, synonyms=related[:10], language=src_lang,
            confidence=0.4, match_kind="related_concept",
            metadata={"relation": "RelatedTo/IsA", "via": "web"}))
    return matches[: max(limit, len(matches))]
