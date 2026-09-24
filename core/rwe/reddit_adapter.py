"""Public RSS Reddit collector and RWE normalizer.

Reddit is accessed through its public read-only RSS search feed. No API key,
OAuth flow, client credentials, account, or persistent cache is required.
"""
from __future__ import annotations

import html
import logging
import re
from datetime import datetime
from typing import List, Optional, Tuple
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests

from core.rwe.models import RWEItem, RWE_SOURCES

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_NO_RESULTS = "no_results"
STATUS_RATE_LIMITED = "rate_limited"
STATUS_NETWORK_ERROR = "network_error"

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _lim():
    try:
        from core.llm_limits import get_limits
        return get_limits()
    except Exception:
        from core.llm_limits import HLEOLimits
        return HLEOLimits()


def _plain_text(value: str) -> str:
    value = html.unescape(value or "")
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", value)).strip()


def _parse_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return value[:10] if len(value) >= 10 else value


def _entry_value(entry, local_name: str) -> str:
    for child in entry:
        if child.tag.rsplit("}", 1)[-1] == local_name:
            return (child.text or "").strip()
    return ""


def _entry_link(entry) -> str:
    for child in entry:
        if child.tag.rsplit("}", 1)[-1] == "link":
            return (child.attrib.get("href") or child.text or "").strip()
    return ""


def _subreddit(link: str) -> Optional[str]:
    match = re.search(r"/r/([^/]+)", urlparse(link).path, re.IGNORECASE)
    return match.group(1) if match else None


def _to_rwe_item(title: str, link: str, external_id: str, body: str,
                 date: Optional[str], topic: str) -> RWEItem:
    meta = RWE_SOURCES["reddit"]
    subreddit = _subreddit(link)
    metadata = {"post_chars": len(body)}
    if subreddit:
        metadata["subreddit"] = subreddit
    return RWEItem(
        source="reddit",
        source_type=meta["source_type"],
        evidence_tier=meta["evidence_tier"],
        collection_method=meta["collection_method"],
        external_id=external_id or link,
        source_url=link or None,
        title=title,
        text=body[:4000],
        date=date,
        language="en",
        topic=topic,
        privacy_status="redacted",
        metadata=metadata,
    )


class RedditRWEAdapter:
    """Read-only Reddit RSS search adapter using shared HTTP limits."""

    RSS_URL = "https://www.reddit.com/search.rss"

    def search_with_status(
        self, query: str, limit: int = 15
    ) -> Tuple[List[RWEItem], str, str]:
        if not (query or "").strip():
            return [], STATUS_NO_RESULTS, "Reddit RSS query is empty."

        limits = _lim()
        from core.http_retry import http_get
        try:
            response = http_get(
                self.RSS_URL,
                params={"q": query, "sort": "relevance", "t": "all", "limit": limit},
                headers={
                    "Accept": "application/atom+xml, application/rss+xml, application/xml",
                    "User-Agent": "hleo-rwe/1.0 (public RSS; read-only)",
                },
                timeout=limits.collector_timeout_s,
                max_retries=limits.collector_max_retries,
                backoff_base_s=limits.backoff_base_s,
                backoff_max_s=limits.backoff_max_s,
            )
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", 0)
            if status == 429:
                return [], STATUS_RATE_LIMITED, "Reddit RSS rate limit reached."
            return [], STATUS_NETWORK_ERROR, f"Reddit RSS HTTP error: {status or exc}."
        except requests.exceptions.RequestException as exc:
            return [], STATUS_NETWORK_ERROR, f"Reddit RSS network error: {exc}"

        try:
            root = ElementTree.fromstring(response.content)
        except ElementTree.ParseError as exc:
            return [], STATUS_NETWORK_ERROR, f"Reddit RSS parse error: {exc}"

        entries = [node for node in root.iter()
                   if node.tag.rsplit("}", 1)[-1] in {"entry", "item"}]
        items: List[RWEItem] = []
        for entry in entries[:limit]:
            title = _entry_value(entry, "title")
            link = _entry_link(entry)
            external_id = _entry_value(entry, "id") or _entry_value(entry, "guid")
            body = _plain_text(_entry_value(entry, "content") or _entry_value(entry, "description"))
            if not title and not body:
                continue
            items.append(_to_rwe_item(
                title=title,
                link=link,
                external_id=external_id,
                body=body,
                date=_parse_date(_entry_value(entry, "updated") or _entry_value(entry, "pubDate")),
                topic=query,
            ))

        if not items:
            return [], STATUS_NO_RESULTS, f'No Reddit RSS posts matched "{query}".'
        return items, STATUS_OK, f"Retrieved {len(items)} Reddit RSS post(s)."

    def search(self, query: str, limit: int = 15) -> List[RWEItem]:
        items, status, reason = self.search_with_status(query, limit=limit)
        if status != STATUS_OK:
            logger.info("Reddit RSS silent-fail [%s]: %s", status, reason)
        return items
