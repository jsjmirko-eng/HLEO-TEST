"""CiNii Research OpenSearch collector."""
from __future__ import annotations

from typing import Any, Optional

from collectors.scientific_base import ScientificCollectorBase
from core.search_page import SearchPage


class CiNiiResearchCollector(ScientificCollectorBase):
    """Search the public CiNii Research OpenSearch articles endpoint."""

    source = "CiNii Research"
    BASE_URL = "https://ci.nii.ac.jp/opensearch/search"
    default_page_size = 200
    min_interval_s = 0.1

    @staticmethod
    def _first(value: Any) -> Optional[Any]:
        if isinstance(value, list):
            return value[0] if value else None
        return value

    @staticmethod
    def _identifier(item: dict, *names: str) -> Optional[str]:
        wanted = {name.lower() for name in names}
        for identifier in item.get("dc:identifier", []) or []:
            kind = str(identifier.get("@type", "")).lower()
            value = identifier.get("@value")
            if value and (kind in wanted or any(name in kind for name in wanted)):
                return str(value)
        return None

    @staticmethod
    def _source_urls(item: dict) -> list[str]:
        urls = []
        for source in item.get("dc:source", []) or []:
            if isinstance(source, dict) and source.get("@id"):
                urls.append(str(source["@id"]))
        return urls

    @staticmethod
    def _year(value: Any) -> Optional[int]:
        if not value:
            return None
        text = str(value)
        return int(text[:4]) if text[:4].isdigit() else None

    def search_page(
        self,
        query: str,
        cursor: Optional[dict] = None,
        limit: Optional[int] = None,
    ) -> SearchPage:
        """Fetch one OpenSearch page using CiNii's native start/count paging."""
        state, remaining = self._state(cursor, limit)
        if remaining <= 0:
            return SearchPage([], None, False, state.get("total"))

        page_size = min(self.default_page_size, remaining)
        start = int(state.get("start", 1))
        params = {
            "format": "json",
            "q": query,
            "count": page_size,
            "start": start,
            "sortorder": 4,
        }
        response = self._request(self.BASE_URL, params=params)
        payload = response.json()
        total = int(payload.get("opensearch:totalResults", 0) or 0)
        raw_items = payload.get("items", []) or []
        items = []
        for record in raw_items:
            title = str(record.get("title") or "")
            detail_url = (record.get("link") or {}).get("@id") or record.get("@id")
            source_urls = self._source_urls(record)
            full_text_url = source_urls[0] if source_urls else None
            publication_date = record.get("prism:publicationDate")
            creators = record.get("dc:creator", []) or []
            identifiers = record.get("dc:identifier", []) or []
            items.append(self._result(
                title=title,
                original_title=title,
                abstract=self._clean_text(record.get("description")),
                authors=[str(author) for author in creators if author],
                year=self._year(publication_date),
                doi=self._identifier(record, "doi"),
                pmid=self._identifier(record, "pmid"),
                pmcid=self._identifier(record, "pmcid", "pmc"),
                url=detail_url,
                full_text_url=full_text_url,
                language=record.get("dc:language"),
                source_id=record.get("@id") or detail_url,
                full_text_available=bool(full_text_url),
                metadata={
                    "journal": record.get("prism:publicationName") or "",
                    "cinii_type": record.get("@type"),
                    "publisher": record.get("dc:publisher"),
                    "identifiers": identifiers,
                    "provenance": {
                        "provider": self.source,
                        "record_url": detail_url,
                        "source_urls": source_urls,
                    },
                },
            ))

        collected = int(state.get("collected", 0)) + len(items)
        next_start = start + len(items)
        target = min(total or collected, limit or self.max_results_without_limit)
        has_more = bool(items and collected < target and next_start > start and next_start <= total)
        next_state = None
        if has_more:
            next_state = {
                "start": next_start,
                "collected": collected,
                "total": total,
            }
        return SearchPage(items, next_state, has_more, total)
