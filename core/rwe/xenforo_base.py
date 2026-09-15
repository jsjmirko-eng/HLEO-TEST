"""
Shared base for XenForo-based RWE collectors.

Several hair-loss community forums run on XenForo and expose the same official
RSS content-syndication channel: ``<forum-slug>.<id>/index.rss`` per sub-forum.
The feed returns title, link, guid (thread id), pubDate, author, category and
``content:encoded`` (first-post HTML). This module centralises the fetch/parse
logic so each per-site collector only declares its base URL and the curated set
of hair-loss-relevant sub-forums (excluding transplant / cosmetic sections).

Access method: official XenForo RSS feed (no API key, no OAuth). The feed is
the platform's documented syndication channel, linked from each forum's
homepage, and intended for public consumption. No CAPTCHA / login / anti-bot
challenge is placed on the RSS endpoint.

Privacy: the author element is deliberately NOT carried into RWEItem
(privacy_status="redacted"); only the metadata necessary for provenance and
source identification is retained.

FASE 6B optimisations
---------------------
- Per-slug HTTP uses core.http_retry.http_get → retry on 429/5xx/Timeout/
  ConnectionError (bounded by collector_max_retries from Global Limits).
- Timeout read from collector_timeout_s (Global Limits) instead of hardcoded.
- Forum slugs fetched in parallel (ThreadPoolExecutor, max 4 workers) with a
  per-instance threading.Lock protecting _feed_cache writes.
"""
from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional, Tuple
from xml.etree import ElementTree

import requests

from core.rwe.models import RWEItem

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_NO_RESULTS = "no_results"
STATUS_RATE_LIMITED = "rate_limited"
STATUS_NETWORK_ERROR = "network_error"

# XenForo RSS namespaces (content:encoded carries the first-post HTML body).
_NS = {
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "slash": "http://purl.org/rss/1.0/modules/slash/",
}

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# Max parallel slug fetches per collector instance (Opt 5).
_SLUG_WORKERS = 4


def _lim():
    """Return current Global Limits (cached, 5 s TTL)."""
    try:
        from core.llm_limits import get_limits
        return get_limits()
    except Exception:
        from core.llm_limits import HLEOLimits
        return HLEOLimits()


def _strip_html(html: str) -> str:
    """Strip XenForo bbCode-wrapper HTML to plain text (best-effort)."""
    if not html:
        return ""
    # Drop <a class="link link--internal">…</a> "Leggi di più" / wrapper links.
    text = re.sub(r'<a[^>]*class="[^"]*link--internal[^"]*"[^>]*>.*?</a>', " ", html, flags=re.S)
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _parse_date(date_str: Optional[str]) -> Optional[str]:
    """Parse an RFC-822 pubDate into ISO-8601 (date only)."""
    if not date_str:
        return None
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z"):
        try:
            return datetime.strptime(date_str.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _safe_int(val: Optional[str]) -> Optional[int]:
    if not val:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _aggregate_status(statuses: List[str]) -> str:
    """Best status across per-forum fetches (ok > no_results > rate > error)."""
    rank = {
        STATUS_OK: 0, STATUS_NO_RESULTS: 1, STATUS_RATE_LIMITED: 2,
        STATUS_NETWORK_ERROR: 3,
    }
    if not statuses:
        return STATUS_NO_RESULTS
    return min(statuses, key=lambda s: rank.get(s, 9))


def _best_reason(statuses: List[str], reasons: List[str], agg: str) -> str:
    """Return the reason of the first per-forum fetch matching the aggregate status."""
    for s, r in zip(statuses, reasons):
        if s == agg:
            return r
    return "No threads collected."


class XenForoRSSCollector:
    """Generic read-only XenForo RSS collector.

    Subclasses set ``source`` (RWE_SOURCES key), ``base_url`` and
    ``forum_slugs`` (curated hair-loss sub-forums; transplant/cosmetic forums
    excluded by omission). The fetch/parse/aggregate logic is shared.
    """

    source: str = ""
    base_url: str = ""
    language: str = "en"

    def __init__(self, forum_slugs: Optional[List[str]] = None) -> None:
        if forum_slugs is not None:
            self.forum_slugs = forum_slugs
        self._feed_cache: dict = {}
        self._cache_lock = threading.Lock()
        # forum_slugs must be declared on the subclass as a class attribute.

    def _meta(self) -> dict:
        from core.rwe.models import RWE_SOURCES
        return RWE_SOURCES[self.source]

    def _fetch_forum(self, slug: str, limit: int) -> Tuple[List[RWEItem], str, str]:
        """Fetch a single forum's RSS feed and normalize items.

        Thread-safe: _feed_cache is protected by _cache_lock.
        HTTP uses http_get for retry on 429/5xx/Timeout/ConnectionError.
        Timeout comes from collector_timeout_s (Global Limits).
        """
        from core.http_retry import http_get

        lim = _lim()
        timeout = lim.collector_timeout_s
        max_retries = lim.collector_max_retries
        backoff_base = lim.backoff_base_s
        backoff_max = lim.backoff_max_s

        with self._cache_lock:
            if slug in self._feed_cache:
                items, status, reason = self._feed_cache[slug]
                return items[:limit], status, reason

        url = f"{self.base_url}/{slug}/index.rss"
        try:
            resp = http_get(
                url,
                headers={"Accept": "application/rss+xml, application/xml, text/xml"},
                timeout=timeout,
                max_retries=max_retries,
                backoff_base_s=backoff_base,
                backoff_max_s=backoff_max,
            )
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            if code == 429:
                logger.warning(f"{self.source} rate limit for {slug} (after retries)")
                return [], STATUS_RATE_LIMITED, f"{self.source} rate limit reached. Retry later."
            if code == 404:
                return [], STATUS_NO_RESULTS, f"{self.source} forum '{slug}' not found (404)."
            logger.warning(f"{self.source} HTTP {code} for {slug}")
            return [], STATUS_NETWORK_ERROR, f"{self.source} HTTP {code} for {slug}."
        except requests.exceptions.RequestException as exc:
            logger.warning(f"{self.source} network error for {slug}: {exc}")
            return [], STATUS_NETWORK_ERROR, str(exc)

        try:
            root = ElementTree.fromstring(resp.content)
        except ElementTree.ParseError as exc:
            return [], STATUS_NETWORK_ERROR, f"{self.source} RSS parse error: {exc}"

        channel = root.find("channel")
        if channel is None:
            return [], STATUS_NO_RESULTS, f"{self.source} RSS has no channel."

        items: List[RWEItem] = []
        meta = self._meta()
        entries = channel.findall("item")
        for entry in entries:
            title = (entry.findtext("title") or "").strip()
            link = (entry.findtext("link") or "").strip()
            guid = (entry.findtext("guid") or "").strip()
            pub = entry.findtext("pubDate")
            category = (entry.findtext("category") or "").strip()
            body_html = entry.find("content:encoded", _NS)
            body_html = body_html.text if body_html is not None else ""
            body = _strip_html(body_html)

            if not title and not body:
                continue

            items.append(RWEItem(
                source=self.source,
                source_type=meta["source_type"],
                evidence_tier=meta["evidence_tier"],
                collection_method=meta["collection_method"],
                external_id=guid or link,
                source_url=link,
                title=title,
                text=body[:4000],
                date=_parse_date(pub),
                language=self.language,
                topic=category or slug.split(".")[0],
                privacy_status="redacted",  # author deliberately not carried
                metadata={
                    "forum": slug,
                    "comments": _safe_int(entry.findtext("slash:comments", namespaces=_NS)),
                },
            ))

        result_tuple = (items, STATUS_OK, (
            f"Retrieved {len(items)} {self.source} thread(s) from {slug.split('.')[0]}."
        ))
        with self._cache_lock:
            self._feed_cache[slug] = result_tuple
        return items[:limit], STATUS_OK, (
            f"Retrieved {len(items[:limit])} {self.source} thread(s) from {slug.split('.')[0]}."
        )

    def search_with_status(
        self,
        query: str,
        limit: int = 20,
    ) -> Tuple[List[RWEItem], str, str]:
        """
        Fetch XenForo RSS threads from the curated hair-loss sub-forums.

        The XenForo RSS feed is per-forum (not full-text searchable by query);
        this collector pulls recent threads from the curated forum set. The RWE
        pipeline's relevance filter keeps only items matching the (translated)
        query, so the query still drives relevance even though the upstream
        fetch is forum-scoped rather than query-scoped.

        Slugs are fetched in parallel (up to _SLUG_WORKERS concurrent fetches)
        to reduce wall-clock time for collectors with many sub-forums (e.g.
        Calvizie with 19 slugs).

        Returns: (items, status_code, human_reason)
        """
        if not query.strip():
            return [], STATUS_NO_RESULTS, "Empty query."

        target_limit = limit if limit is not None else 10**9
        all_items: List[RWEItem] = []
        statuses: List[str] = []
        reasons: List[str] = []

        n_workers = min(len(self.forum_slugs), _SLUG_WORKERS)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(self._fetch_forum, slug, 10**9): slug
                for slug in self.forum_slugs
            }
            for fut in as_completed(futures):
                items, status, reason = fut.result()
                statuses.append(status)
                reasons.append(reason)
                if status == STATUS_OK:
                    all_items.extend(items)

        if not all_items:
            if any(s == STATUS_OK for s in statuses):
                agg = STATUS_NO_RESULTS
                agg_reason = (
                    f"{self.source} feeds fetched successfully but returned no threads."
                )
            else:
                agg = _aggregate_status(statuses)
                agg_reason = _best_reason(statuses, reasons, agg)
            return [], agg, agg_reason

        return all_items if limit is None else all_items[:limit], STATUS_OK, f"Retrieved {len(all_items)} {self.source} thread(s)."

    def search(self, query: str, limit: int = 20) -> List[RWEItem]:
        """Silent-fail wrapper for pipeline use."""
        items, status, reason = self.search_with_status(query, limit=limit)
        if status != STATUS_OK:
            logger.info(f"{self.source} silent-fail [{status}]: {reason}")
            return []
        return items
