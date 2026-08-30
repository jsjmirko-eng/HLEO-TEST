"""
Reddit adapter for the RWE pipeline.

Wraps the existing ``collectors.reddit.RedditCollector`` (PRAW OAuth2) and
normalizes its ``RawTestimonial`` output into ``RWEItem`` objects, so the RWE
pipeline can treat Reddit and openFDA uniformly.

Reddit data is classified evidence_tier="anecdotal" and never presented as
clinical evidence.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Tuple

from collectors.reddit import RedditCollector, RawTestimonial
from core.rwe.models import RWEItem, RWE_SOURCES

logger = logging.getLogger(__name__)


def to_rwe_item(post: RawTestimonial, topic: str = "") -> RWEItem:
    """Normalize a RawTestimonial into an RWEItem (redacted)."""
    meta = RWE_SOURCES["reddit"]
    date_iso = None
    if isinstance(post.created_at, datetime):
        date_iso = post.created_at.date().isoformat()
    elif post.created_at:
        date_iso = str(post.created_at)

    return RWEItem(
        source="reddit",
        source_type=meta["source_type"],
        evidence_tier=meta["evidence_tier"],
        collection_method=meta["collection_method"],
        external_id=post.url,
        source_url=post.url,
        title=post.title,
        text=(post.text or "")[:4000],
        date=date_iso,
        language="en",
        topic=topic or post.title,
        # Author is deliberately NOT carried — privacy_status=redacted
        privacy_status="redacted",
        metadata={"post_chars": len(post.text or "")},
    )


class RedditRWEAdapter:
    """Thin adapter exposing the RedditCollector via the RWE status contract."""

    STATUS_OK = "ok"
    STATUS_NO_CREDENTIALS = "no_credentials"
    STATUS_AUTH_ERROR = "auth_error"
    STATUS_RATE_LIMITED = "rate_limited"
    STATUS_NO_RESULTS = "no_results"
    STATUS_NETWORK_ERROR = "network_error"

    def __init__(self) -> None:
        self._collector = RedditCollector()

    def search_with_status(
        self, query: str, limit: int = 15
    ) -> Tuple[List[RWEItem], str, str]:
        posts, status, reason = self._collector.search_with_status(query, limit=limit)
        if status != self.STATUS_OK:
            return [], status, reason
        items = [to_rwe_item(p, topic=query) for p in posts]
        return items, self.STATUS_OK, f"Retrieved {len(items)} Reddit post(s)."

    def search(self, query: str, limit: int = 10) -> List[RWEItem]:
        items, status, reason = self.search_with_status(query, limit=limit)
        if status != self.STATUS_OK:
            logger.info(f"Reddit RWE silent-fail [{status}]: {reason}")
        return items
