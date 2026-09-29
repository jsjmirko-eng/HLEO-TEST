"""OpenAIRE Graph v3 research-products collector."""
from __future__ import annotations

from typing import Optional

from collectors.scientific_base import ScientificCollectorBase
from core.search_page import SearchPage


class OpenAIRECollector(ScientificCollectorBase):
    """Search OpenAIRE research products and normalize publication metadata."""

    source = "OpenAIRE"
    BASE_URL = "https://api.openaire.eu/graph/v3/research-products"
    default_page_size = 100
    min_interval_s = 0.1

    @staticmethod
    def _pid(item: dict, *schemes: str) -> Optional[str]:
        wanted = {scheme.lower() for scheme in schemes}
        for pid in item.get("pids", []) or []:
            if str(pid.get("scheme", "")).lower() in wanted and pid.get("value"):
                return str(pid["value"])
        return None

    @staticmethod
    def _language(value) -> Optional[str]:
        if isinstance(value, dict):
            value = value.get("code") or value.get("label")
        if not value:
            return None
        return {
            "eng": "en", "deu": "de", "ger": "de", "fra": "fr",
            "fre": "fr", "ita": "it", "spa": "es", "jpn": "ja",
        }.get(str(value).lower(), str(value))

    @staticmethod
    def _urls(item: dict) -> tuple[Optional[str], Optional[str]]:
        """Return a display URL and a direct full-text URL when available."""
        display_url = None
        full_text_url = None
        for instance in item.get("instances", []) or []:
            for url in instance.get("urls", []) or []:
                if not display_url:
                    display_url = url
                lowered = str(url).lower()
                if not any(marker in lowered for marker in (
                    "doi.org", "dx.doi.org", "pubmed.ncbi.nlm.nih.gov",
                )):
                    full_text_url = full_text_url or url
        return display_url, full_text_url

    def search_page(
        self,
        query: str,
        cursor: Optional[dict] = None,
        limit: Optional[int] = None,
    ) -> SearchPage:
        """Fetch one OpenAIRE page; provider pagination stays in this method."""
        state, remaining = self._state(cursor, limit)
        if remaining <= 0:
            return SearchPage([], None, False, state.get("total"))

        page_size = min(self.default_page_size, remaining)
        params = {
            "search": query,
            "type": "publication",
            "pageSize": page_size,
        }
        openaire_cursor = state.get("openaire_cursor")
        if openaire_cursor:
            params["cursor"] = openaire_cursor
        else:
            params["page"] = int(state.get("page", 1))

        response = self._request(self.BASE_URL, params=params)
        payload = response.json()
        header = payload.get("header", {}) or {}
        total = int(header.get("numFound", 0) or 0)
        items = []
        for record in payload.get("results", []) or []:
            display_url, full_text_url = self._urls(record)
            best_access = str(
                (record.get("bestAccessRight") or {}).get("label", "")
            ).upper()
            title = record.get("mainTitle") or ""
            collected_from = [
                entry.get("value")
                for entry in record.get("collectedFrom", []) or []
                if entry.get("value")
            ]
            items.append(self._result(
                title=title,
                original_title=title,
                abstract=" ".join(record.get("descriptions", []) or []),
                authors=[
                    author.get("fullName", "")
                    for author in record.get("authors", []) or []
                    if author.get("fullName")
                ],
                year=self._year(record.get("publicationDate")),
                doi=self._pid(record, "doi"),
                pmid=self._pid(record, "pmid"),
                pmcid=self._pid(record, "pmc", "pmcid"),
                url=display_url,
                full_text_url=full_text_url,
                language=self._language(record.get("language")),
                source_id=record.get("id"),
                full_text_available=bool(full_text_url or best_access == "OPEN"),
                metadata={
                    "journal": (record.get("container") or {}).get("name", ""),
                    "openaire_id": record.get("id"),
                    "collected_from": collected_from,
                },
            ))

        collected = int(state.get("collected", 0)) + len(items)
        next_openaire_cursor = header.get("nextCursor")
        target = min(total or collected, limit or self.max_results_without_limit)
        has_more = bool(
            items
            and collected < target
            and (next_openaire_cursor or len(items) == page_size)
        )
        next_state = None
        if has_more:
            next_state = {
                "page": int(state.get("page", 1)) + 1,
                "collected": collected,
                "total": total,
            }
            if next_openaire_cursor:
                next_state["openaire_cursor"] = next_openaire_cursor
        return SearchPage(items, next_state, has_more, total)
