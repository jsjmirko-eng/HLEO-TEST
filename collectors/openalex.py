"""OpenAlex scholarly works collector."""
from __future__ import annotations

import re
from typing import Optional

from collectors.scientific_base import ScientificCollectorBase, page


class OpenAlexCollector(ScientificCollectorBase):
    source = "OpenAlex"
    BASE_URL = "https://api.openalex.org/works"
    default_page_size = 100
    min_interval_s = 1.0

    @staticmethod
    def _abstract(inverted):
        if not isinstance(inverted, dict):
            return ""
        words = []
        for word, positions in inverted.items():
            for position in positions or []:
                words.append((position, word))
        return " ".join(word for _, word in sorted(words))

    @staticmethod
    def _identifier(ids, name):
        value = (ids or {}).get(name)
        if not value:
            return None
        text = str(value).rstrip("/")
        match = re.search(r"([0-9]+)$", text)
        if not match:
            return text
        return f"PMC{match.group(1)}" if name == "pmcid" else match.group(1)

    def search_page(self, query: str, cursor: Optional[dict] = None, limit: Optional[int] = None):
        state, remaining = self._state(cursor, limit)
        if remaining <= 0:
            return page([], total_available=state.get("total"))
        per_page = min(self.default_page_size, remaining)
        page_number = int(state.get("page", 1))
        response = self._request(
            self.BASE_URL,
            params={"search": query, "page": page_number, "per-page": per_page},
        )
        payload = response.json()
        meta = payload.get("meta", {}) or {}
        total = int(meta.get("count", 0) or 0)
        results = payload.get("results", []) or []
        output = []
        for item in results:
            location = item.get("primary_location") or {}
            source = location.get("source") or {}
            links = item.get("locations") or []
            pdf_url = None
            landing_url = None
            for candidate in links:
                pdf_url = pdf_url or candidate.get("pdf_url")
                landing_url = landing_url or candidate.get("landing_page_url")
            full_text_url = pdf_url or landing_url
            language_value = item.get("language") or ""
            if isinstance(language_value, dict):
                language_value = language_value.get("id") or language_value.get("display_name")
            output.append(self._result(
                title=item.get("title") or "",
                abstract=self._abstract(item.get("abstract_inverted_index")),
                authors=[(a.get("author") or {}).get("display_name", "") for a in item.get("authorships", []) if (a.get("author") or {}).get("display_name")],
                year=self._year(item.get("publication_year")),
                doi=item.get("doi"),
                pmid=self._identifier(item.get("ids"), "pmid"),
                pmcid=self._identifier(item.get("ids"), "pmcid"),
                url=landing_url or item.get("id"),
                full_text_url=full_text_url,
                language=language_value,
                source_id=item.get("id"),
                full_text_available=bool(full_text_url or (item.get("open_access") or {}).get("is_oa")),
                metadata={"journal": source.get("display_name", ""), "openalex_id": item.get("id"), "cited_by_count": item.get("cited_by_count")},
            ))
        collected = int(state.get("collected", 0)) + len(output)
        has_more = bool(output) and collected < min(total or collected, limit or self.max_results_without_limit)
        next_cursor = {"page": page_number + 1, "collected": collected, "total": total} if has_more else None
        return page(output, cursor=next_cursor, has_more=has_more, total_available=total)
