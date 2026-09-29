"""HAL open research archive collector."""
from __future__ import annotations

from typing import Any, Optional

from collectors.scientific_base import ScientificCollectorBase
from core.search_page import SearchPage


class HALCollector(ScientificCollectorBase):
    """Search the public HAL Solr API and normalize one result page."""

    source = "HAL"
    BASE_URL = "https://api.archives-ouvertes.fr/search/"
    default_page_size = 100
    min_interval_s = 0.1
    FIELDS = ",".join((
        "docid", "title_s", "abstract_s", "authFullName_s", "doiId_s",
        "pmidId_s", "pmcidId_s", "uri_s", "uriFulltext_s", "fileMain_s",
        "language_s", "producedDateY_i", "publicationDateY_i",
        "journalTitle_s", "openAccess_bool", "docType_s",
    ))

    @staticmethod
    def _first(value: Any) -> Optional[str]:
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        return str(value) if value not in (None, "") else None

    def _urls(self, record: dict) -> tuple[Optional[str], Optional[str]]:
        url = self._first(record.get("uri_s"))
        full_text = self._first(record.get("fileMain_s"))
        full_text = full_text or self._first(record.get("uriFulltext_s"))
        return url, full_text

    def search_page(
        self,
        query: str,
        cursor: Optional[dict] = None,
        limit: Optional[int] = None,
    ) -> SearchPage:
        """Fetch one HAL page; cursor state remains private to this adapter."""
        state, remaining = self._state(cursor, limit)
        if remaining <= 0:
            return SearchPage([], None, False, state.get("total"))

        page_size = min(self.default_page_size, remaining)
        current_cursor = state.get("hal_cursor", "*")
        params = {
            "q": query,
            "wt": "json",
            "rows": page_size,
            "sort": "docid asc",
            "cursorMark": current_cursor,
            "fl": self.FIELDS,
        }
        response = self._request(self.BASE_URL, params=params)
        payload = response.json()
        response_data = payload.get("response", {}) or {}
        total = int(response_data.get("numFound", 0) or 0)
        results = response_data.get("docs", []) or []
        items = []
        for record in results:
            title = self._first(record.get("title_s")) or ""
            url, full_text_url = self._urls(record)
            open_access = bool(record.get("openAccess_bool"))
            items.append(self._result(
                title=title,
                original_title=title,
                abstract=self._clean_text(self._first(record.get("abstract_s"))),
                authors=[
                    str(author)
                    for author in record.get("authFullName_s", []) or []
                    if author
                ],
                year=self._year(
                    record.get("producedDateY_i")
                    or record.get("publicationDateY_i")
                ),
                doi=self._first(record.get("doiId_s")),
                pmid=self._first(record.get("pmidId_s")),
                pmcid=self._first(record.get("pmcidId_s")),
                url=url,
                full_text_url=full_text_url,
                language=self._first(record.get("language_s")),
                source_id=self._first(record.get("docid")),
                full_text_available=bool(full_text_url or open_access),
                metadata={
                    "journal": self._first(record.get("journalTitle_s")) or "",
                    "hal_id": self._first(record.get("docid")),
                    "open_access": open_access,
                    "document_type": self._first(record.get("docType_s")),
                },
            ))

        collected = int(state.get("collected", 0)) + len(items)
        next_cursor = payload.get("nextCursorMark")
        target = min(total or collected, limit or self.max_results_without_limit)
        has_more = bool(
            items
            and collected < target
            and next_cursor
            and next_cursor != current_cursor
        )
        next_state = None
        if has_more:
            next_state = {
                "hal_cursor": next_cursor,
                "collected": collected,
                "total": total,
            }
        return SearchPage(items, next_state, has_more, total)
