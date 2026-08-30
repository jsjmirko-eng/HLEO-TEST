"""
MaladiesRaresInfo.org collector — official phpBB Atom feed (alopecia areata).

Source: https://forums.maladiesraresinfo.org/app.php/feed/forum/<id>
Forum:  phpBB (https://www.phpbb.com) — "Le Forum maladies rares"
ToS:    phpBB exposes an official Atom syndication feed per forum at
        ``/app.php/feed/forum/<id>``. The site's robots.txt does not disallow
        the feed path. The feed is linked from the forum and intended for
        syndication. No API key, no OAuth. No CAPTCHA / login / anti-bot.
Auth:   None (public Atom feed).

MaladiesRaresInfo is a French-language support community for people affected by
rare diseases. Its "pelade universelle" sub-forum (alopecia universalis, the
most extensive form of alopecia areata) carries real patient experiences:
treatment attempts, shedding, regrowth, steroid/immunosuppressant outcomes and
disease progression. Because alopecia areata is explicitly in-scope (the
project targets it directly), this forum — though small-volume — provides
dedicated FR-language RWE that no other included source covers. Threads are
classified evidence_tier="anecdotal", source_type="community_forum",
language="fr".

Only the alopecia-related sub-forum (pelade universelle, f173) is collected;
the rest of the rare-disease community is out of scope (non-hair-loss).
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import List, Optional, Tuple
from xml.etree import ElementTree

import requests

from core.rwe.models import RWEItem, RWE_SOURCES

logger = logging.getLogger(__name__)

BASE_URL = "https://forums.maladiesraresinfo.org/app.php/feed/forum"

STATUS_OK = "ok"
STATUS_NO_RESULTS = "no_results"
STATUS_RATE_LIMITED = "rate_limited"
STATUS_NETWORK_ERROR = "network_error"

_NS = {"atom": "http://www.w3.org/2005/Atom"}

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_html(html: str) -> str:
    if not html:
        return ""
    text = _TAG_RE.sub(" ", html)
    return _WS_RE.sub(" ", text).strip()


def _parse_iso(date_str: Optional[str]) -> Optional[str]:
    """Parse an Atom <updated>/<published> ISO-8601 timestamp into a date."""
    if not date_str:
        return None
    s = date_str.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%Z"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    # Truncate fractional seconds if present (e.g. 2026-03-28T12:57:56.5+02:00)
    try:
        return datetime.fromisoformat(s).date().isoformat()
    except (ValueError, TypeError):
        return None


# phpBB forum IDs for the alopecia-areata sub-forums (pelade universelle =
# alopecia universalis, the most extensive form of alopecia areata). Other
# rare-disease sub-forums are out of scope (non-hair-loss) and omitted.
_FORUM_IDS: List[int] = [
    173,  # pelade-universelle (alopecia universalis)
]


class MaladiesRaresCollector:
    """Read-only MaladiesRaresInfo phpBB Atom collector (alopecia areata, FR)."""

    source = "maladiesrares"
    timeout = 20

    def __init__(self, forum_ids: Optional[List[int]] = None) -> None:
        self.forum_ids = forum_ids if forum_ids is not None else _FORUM_IDS
        self._feed_cache: dict = {}

    def _meta(self) -> dict:
        return RWE_SOURCES[self.source]

    def _fetch_forum(self, forum_id: int, limit: int) -> Tuple[List[RWEItem], str, str]:
        """Fetch a single phpBB Atom feed and normalize entries."""
        if forum_id in self._feed_cache:
            items, status, reason = self._feed_cache[forum_id]
            return items[:limit], status, reason
        url = f"{BASE_URL}/{forum_id}"
        try:
            resp = requests.get(url, timeout=self.timeout, headers={
                "Accept": "application/atom+xml, application/xml, text/xml",
            })
        except requests.exceptions.RequestException as exc:
            logger.warning(f"{self.source} network error for {forum_id}: {exc}")
            return [], STATUS_NETWORK_ERROR, str(exc)

        if resp.status_code == 429:
            return [], STATUS_RATE_LIMITED, f"{self.source} rate limit reached. Retry later."
        if resp.status_code == 404:
            return [], STATUS_NO_RESULTS, f"{self.source} forum {forum_id} not found (404)."
        if resp.status_code != 200:
            return [], STATUS_NETWORK_ERROR, f"{self.source} HTTP {resp.status_code} for {forum_id}."

        try:
            root = ElementTree.fromstring(resp.content)
        except ElementTree.ParseError as exc:
            return [], STATUS_NETWORK_ERROR, f"{self.source} Atom parse error: {exc}"

        meta = self._meta()
        items: List[RWEItem] = []
        for entry in root.findall("atom:entry", _NS):
            title = (entry.findtext("atom:title", default="", namespaces=_NS) or "").strip()
            link_el = entry.find("atom:link", _NS)
            link = ""
            if link_el is not None:
                link = (link_el.get("href") or "").strip()
            eid = (entry.findtext("atom:id", default="", namespaces=_NS) or "").strip()
            published = entry.findtext("atom:published", namespaces=_NS)
            updated = entry.findtext("atom:updated", namespaces=_NS)
            # phpBB Atom carries the post body in <content type="html">.
            content_el = entry.find("atom:content", _NS)
            body_html = content_el.text if content_el is not None else ""
            body = _strip_html(body_html)

            if not title and not body:
                continue

            items.append(RWEItem(
                source=self.source,
                source_type=meta["source_type"],
                evidence_tier=meta["evidence_tier"],
                collection_method=meta["collection_method"],
                external_id=eid or link,
                source_url=link,
                title=title,
                text=body[:4000],
                date=_parse_iso(published or updated),
                language="fr",
                topic=f"pelade-universelle-{forum_id}",
                privacy_status="redacted",  # author deliberately not carried
                metadata={"forum_id": forum_id},
            ))
        self._feed_cache[forum_id] = (items, STATUS_OK, f"Retrieved {len(items)} {self.source} thread(s) from forum {forum_id}.")
        return items[:limit], STATUS_OK, f"Retrieved {len(items[:limit])} {self.source} thread(s) from forum {forum_id}."

    def search_with_status(
        self,
        query: str,
        limit: Optional[int] = None,
    ) -> Tuple[List[RWEItem], str, str]:
        """Fetch MaladiesRaresInfo Atom threads from the alopecia sub-forums.

        The phpBB Atom feed is per-forum (not full-text searchable by query);
        this collector pulls recent threads from the curated alopecia-areata
        forum set. The RWE pipeline's relevance filter keeps only items
        matching the (translated) query, so the query still drives relevance.

        Returns: (items, status_code, human_reason)
        """
        if not query.strip():
            return [], STATUS_NO_RESULTS, "Empty query."

        target_limit = limit if limit is not None else 10**9
        all_items: List[RWEItem] = []
        statuses: List[str] = []
        reasons: List[str] = []

        for fid in self.forum_ids:
            if len(all_items) >= target_limit:
                break
            items, status, reason = self._fetch_forum(fid, 10**9)
            statuses.append(status)
            reasons.append(reason)
            if status == STATUS_OK:
                all_items.extend(items)

        if not all_items:
            if any(s == STATUS_OK for s in statuses):
                return [], STATUS_NO_RESULTS, (
                    f"{self.source} feeds fetched successfully but returned no threads."
                )
            # All forums failed — surface the most specific failure reason.
            rank = {STATUS_RATE_LIMITED: 0, STATUS_NETWORK_ERROR: 1, STATUS_NO_RESULTS: 2}
            worst = min(statuses, key=lambda s: rank.get(s, 9))
            for s, r in zip(statuses, reasons):
                if s == worst:
                    return [], worst, r
            return [], worst, f"{self.source} collection failed."

        return all_items if limit is None else all_items[:limit], STATUS_OK, f"Retrieved {len(all_items)} {self.source} thread(s)."

    def search(self, query: str, limit: Optional[int] = None) -> List[RWEItem]:
        """Silent-fail wrapper for pipeline use."""
        items, status, reason = self.search_with_status(query, limit=limit)
        if status != STATUS_OK:
            logger.info(f"{self.source} silent-fail [{status}]: {reason}")
            return []
        return items
