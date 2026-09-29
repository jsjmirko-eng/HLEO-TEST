"""Crossref REST metadata collector."""
from __future__ import annotations

from typing import Optional

from collectors.scientific_base import ScientificCollectorBase, page


class CrossrefCollector(ScientificCollectorBase):
    source = "Crossref"
    BASE_URL = "https://api.crossref.org/works"
    default_page_size = 100
    min_interval_s = 0.34

    @staticmethod
    def _date(item):
        for key in ("published-print", "published-online", "issued", "created"):
            parts = ((item.get(key) or {}).get("date-parts") or [[]])[0]
            if parts:
                return parts[0]
        return None

    @staticmethod
    def _authors(item):
        names = []
        for author in item.get("author", []) or []:
            name = " ".join(filter(None, [author.get("given"), author.get("family")]))
            if name:
                names.append(name)
        return names

    @staticmethod
    def _links(item):
        links = item.get("link", []) or []
        for link in links:
            if link.get("content-type") == "application/pdf":
                return link.get("URL")
        return links[0].get("URL") if links else None

    def search_page(self, query: str, cursor: Optional[dict] = None, limit: Optional[int] = None):
        state, remaining = self._state(cursor, limit)
        if remaining <= 0:
            return page([], total_available=state.get("total"))
        rows = min(self.default_page_size, remaining)
        crossref_cursor = state.get("crossref_cursor", "*")
        params = {"query": query, "rows": rows, "cursor": crossref_cursor}
        mailto = __import__("os").getenv("CROSSREF_MAILTO")
        if mailto:
            params["mailto"] = mailto
        response = self._request(self.BASE_URL, params=params)
        payload = response.json().get("message", {})
        items = payload.get("items", []) or []
        output = []
        for item in items:
            full_text_url = self._links(item)
            date = self._date(item)
            output.append(self._result(
                title=(item.get("title") or [""])[0],
                abstract=self._clean_text(item.get("abstract")),
                authors=self._authors(item),
                year=self._year(date),
                doi=item.get("DOI"),
                url=item.get("URL"),
                full_text_url=full_text_url,
                language=item.get("language"),
                source_id=item.get("DOI") or item.get("URL"),
                full_text_available=bool(full_text_url),
                metadata={"journal": (item.get("container-title") or [""])[0], "type": item.get("type"), "publisher": item.get("publisher")},
            ))
        total = int(payload.get("total-results", 0) or 0)
        collected = int(state.get("collected", 0)) + len(output)
        next_value = payload.get("next-cursor")
        has_more = bool(output and next_value and collected < min(total or collected, limit or self.max_results_without_limit))
        next_cursor = {"crossref_cursor": next_value, "collected": collected, "total": total} if has_more else None
        return page(output, cursor=next_cursor, has_more=has_more, total_available=total)
