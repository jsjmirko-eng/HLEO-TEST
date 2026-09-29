"""DOAJ open-access article metadata collector."""
from __future__ import annotations

from typing import Optional

from collectors.scientific_base import ScientificCollectorBase, page


class DOAJCollector(ScientificCollectorBase):
    source = "DOAJ"
    BASE_URL = "https://doaj.org/api/search/articles/{query}"
    default_page_size = 100
    min_interval_s = 0.25

    @staticmethod
    def _identifier(bibjson, kind):
        for identifier in bibjson.get("identifier", []) or []:
            if str(identifier.get("type", "")).lower() == kind:
                return identifier.get("id")
        return None

    @staticmethod
    def _links(bibjson):
        links = bibjson.get("link", []) or []
        for link in links:
            if str(link.get("type", "")).lower() == "fulltext":
                return link.get("url")
        return links[0].get("url") if links else None

    def search_page(self, query: str, cursor: Optional[dict] = None, limit: Optional[int] = None):
        state, remaining = self._state(cursor, limit)
        if remaining <= 0:
            return page([], total_available=state.get("total"))
        page_size = min(self.default_page_size, remaining)
        page_number = int(state.get("page", 1))
        response = self._request(
            self.BASE_URL.format(query=query),
            params={"page": page_number, "pageSize": page_size},
        )
        payload = response.json()
        output = []
        for record in payload.get("results", []) or []:
            bibjson = record.get("bibjson", {}) or {}
            journal = bibjson.get("journal", {}) or {}
            full_text_url = self._links(bibjson)
            language = (journal.get("language") or [None])[0]
            output.append(self._result(
                title=bibjson.get("title", ""),
                abstract=bibjson.get("abstract", ""),
                authors=[a.get("name", "") for a in bibjson.get("author", []) or [] if a.get("name")],
                year=self._year(bibjson.get("year")),
                doi=self._identifier(bibjson, "doi"),
                pmid=self._identifier(bibjson, "pmid"),
                pmcid=self._identifier(bibjson, "pmcid"),
                url=full_text_url,
                full_text_url=full_text_url,
                language=language,
                source_id=record.get("id"),
                full_text_available=bool(full_text_url),
                metadata={"journal": journal.get("title", ""), "country": journal.get("country"), "doaj_id": record.get("id")},
            ))
        total = int(payload.get("total", 0) or 0)
        collected = int(state.get("collected", 0)) + len(output)
        next_url = payload.get("next")
        has_more = bool(output and next_url and collected < min(total or collected, limit or self.max_results_without_limit))
        next_cursor = {"page": page_number + 1, "collected": collected, "total": total} if has_more else None
        return page(output, cursor=next_cursor, has_more=has_more, total_available=total)
