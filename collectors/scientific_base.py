"""Shared utilities for provider-specific scientific metadata collectors."""
from __future__ import annotations

import threading
import time
from typing import Any, Optional

from core.search_page import SearchPage, collect_search_pages
from core.search_result import SearchResult


class ScientificCollectorBase:
    """Small adapter base with bounded HTTP access and normalized results."""

    source = "Scientific"
    default_page_size = 50
    max_results_without_limit = 400
    min_interval_s = 0.0

    def __init__(self, *, timeout: Optional[float] = None, max_retries: Optional[int] = None):
        self.timeout_override = timeout
        self.max_retries_override = max_retries
        self._request_lock = threading.Lock()
        self._next_request_at = 0.0

    def _limits(self):
        try:
            from core.llm_limits import get_limits
            return get_limits()
        except Exception:
            from core.llm_limits import HLEOLimits
            return HLEOLimits()

    def _request(self, url: str, *, params: Optional[dict] = None, headers: Optional[dict] = None):
        from core.http_retry import http_get

        limits = self._limits()
        timeout = self.timeout_override or limits.collector_timeout_s
        retries = limits.collector_max_retries if self.max_retries_override is None else self.max_retries_override
        with self._request_lock:
            wait = self._next_request_at - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                return http_get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                    max_retries=retries,
                    backoff_base_s=limits.backoff_base_s,
                    backoff_max_s=limits.backoff_max_s,
                )
            finally:
                self._next_request_at = time.monotonic() + self.min_interval_s

    def search(self, query: str, limit: Optional[int] = None) -> list[SearchResult]:
        return collect_search_pages(self, query, limit=limit)

    def _result(
        self,
        *,
        title: str = "",
        abstract: str = "",
        authors: Optional[list[str]] = None,
        year: Optional[int] = None,
        doi: Optional[str] = None,
        pmid: Optional[str] = None,
        pmcid: Optional[str] = None,
        url: Optional[str] = None,
        full_text_url: Optional[str] = None,
        language: Optional[str] = None,
        original_title: Optional[str] = None,
        source_id: Optional[str] = None,
        full_text_available: Optional[bool] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SearchResult:
        standard = {
            "pmcid": pmcid,
            "full_text_url": full_text_url,
            "language": language,
            "original_title": original_title or title,
            "source": self.source,
            "source_id": source_id,
            "full_text_available": bool(full_text_available),
        }
        standard.update(metadata or {})
        return SearchResult(
            title=title,
            source=self.source,
            url=url,
            abstract=abstract,
            authors=authors or [],
            year=year,
            doi=doi,
            pmid=pmid,
            pmcid=pmcid,
            full_text_url=full_text_url,
            language=language,
            original_title=original_title or title,
            source_id=source_id,
            full_text_available=full_text_available,
            sources=[self.source],
            metadata=standard,
        )

    @staticmethod
    def _state(cursor: Optional[dict], limit: Optional[int]) -> tuple[dict, int]:
        target = limit if limit is not None else ScientificCollectorBase.max_results_without_limit
        state = dict(cursor or {})
        state.setdefault("collected", 0)
        return state, max(0, target - int(state["collected"]))

    @staticmethod
    def _year(value: Any) -> Optional[int]:
        if isinstance(value, int):
            return value
        text = str(value or "")
        return int(text[:4]) if text[:4].isdigit() else None

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        import re
        from html import unescape
        return " ".join(unescape(re.sub(r"<[^>]+>", " ", str(value))).split())


def page(items, *, cursor=None, has_more=False, total_available=None) -> SearchPage:
    """Construct a SearchPage while keeping provider state opaque."""
    return SearchPage(items, cursor, has_more, total_available)
