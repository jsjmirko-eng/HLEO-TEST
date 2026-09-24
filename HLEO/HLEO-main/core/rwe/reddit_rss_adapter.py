"""Public Reddit RSS adapter for the RWE pipeline."""
from __future__ import annotations

import html
import logging
from datetime import datetime
from xml.etree import ElementTree

import requests

from core.rwe.models import RWEItem, RWE_SOURCES

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_NO_RESULTS = "no_results"
STATUS_RATE_LIMITED = "rate_limited"
STATUS_NETWORK_ERROR = "network_error"

_ATOM = {"atom": "http://www.w3.org/2005/Atom"}


class RedditRWEAdapter:
    """Read-only Reddit search through the public RSS endpoint."""

    SEARCH_URL = "https://www.reddit.com/search.rss"
    USER_AGENT = "HLEO/1.0 (public RWE RSS reader)"
    timeout = 20

    def search_with_status(self, query: str, limit: int | None = 15):
        params = {"q": query, "sort": "relevance", "t": "all"}
        try:
            response = requests.get(
                self.SEARCH_URL,
                params=params,
                headers={
                    "Accept": "application/rss+xml, application/atom+xml, application/xml",
                    "User-Agent": self.USER_AGENT,
                },
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as exc:
            return [], STATUS_NETWORK_ERROR, str(exc)

        if response.status_code == 429:
            return [], STATUS_RATE_LIMITED, "Reddit RSS rate limit reached."
        if response.status_code != 200:
            return [], STATUS_NETWORK_ERROR, f"Reddit RSS HTTP {response.status_code}."

        try:
            root = ElementTree.fromstring(response.content)
        except ElementTree.ParseError as exc:
            return [], STATUS_NETWORK_ERROR, f"Reddit RSS parse error: {exc}"

        entries = root.findall("atom:entry", _ATOM)
        if limit is not None:
            entries = entries[:max(0, limit)]

        items = []
        for entry in entries:
            title = self._text(entry, "title")
            link = self._link(entry)
            body = self._text(entry, "content") or self._text(entry, "summary")
            if not title and not body:
                continue
            items.append(RWEItem(
                source="reddit",
                source_type=RWE_SOURCES["reddit"]["source_type"],
                evidence_tier="anecdotal",
                collection_method="official_rss_feed",
                source_url=link,
                external_id=link,
                title=title,
                text=body[:4000],
                date=self._date(entry),
                language="en",
                topic=query,
                privacy_status="redacted",
            ))

        if not items:
            return [], STATUS_NO_RESULTS, "No Reddit RSS items matched the query."
        return items, STATUS_OK, f"Retrieved {len(items)} Reddit RSS item(s)."

    @staticmethod
    def _text(entry, tag: str) -> str:
        node = entry.find(f"atom:{tag}", _ATOM)
        if node is None:
            return ""
        return html.unescape(" ".join(node.itertext())).strip()

    @staticmethod
    def _link(entry) -> str:
        for node in entry.findall("atom:link", _ATOM):
            href = node.attrib.get("href", "").strip()
            if href:
                return href
        return ""

    @classmethod
    def _date(cls, entry):
        value = cls._text(entry, "updated") or cls._text(entry, "published")
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            return value[:10]
