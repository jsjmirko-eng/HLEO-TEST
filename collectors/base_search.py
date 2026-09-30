"""BASE Academic Search metadata collector."""
from __future__ import annotations

import os
import re
from typing import Any, Optional

from collectors.scientific_base import ScientificCollectorBase, page
from core.search_page import SearchPage


class BASECollector(ScientificCollectorBase):
    """Retrieve and normalize records from the BASE search API."""

    source = "BASE"
    BASE_URL = "https://api.base-search.net/cgi-bin/BaseHttpSearchInterface.fcgi"
    default_page_size = 50
    min_interval_s = 0.2

    @staticmethod
    def _values(value: Any) -> list[str]:
        if value in (None, "", [], {}):
            return []
        values = value if isinstance(value, list) else [value]
        output = []
        for item in values:
            if isinstance(item, dict):
                item = item.get("name") or item.get("value") or item.get("text")
            if item not in (None, ""):
                output.append(str(item).strip())
        return [item for item in output if item]

    @classmethod
    def _first(cls, record: dict[str, Any], *fields: str) -> Optional[str]:
        for field in fields:
            values = cls._values(record.get(field))
            if values:
                return values[0]
        return None

    @classmethod
    def _authors(cls, record: dict[str, Any]) -> list[str]:
        return cls._values(record.get("dccreator") or record.get("creator"))

    @classmethod
    def _doi(cls, record: dict[str, Any]) -> Optional[str]:
        fields = (
            "dcdoi",
            "doi",
            "dcidentifier",
            "identifier",
            "dcrelation",
            "relation",
        )
        pattern = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
        for field in fields:
            for value in cls._values(record.get(field)):
                match = pattern.search(value)
                if match:
                    return match.group(0).rstrip(".,;)")
        return None

    @classmethod
    def _source_id(cls, record: dict[str, Any]) -> Optional[str]:
        return cls._first(record, "dcdocid", "global_id", "dcoai", "base_id", "identifier")

    @staticmethod
    def _total(value: Any) -> Optional[int]:
        try:
            total = int(value)
        except (TypeError, ValueError):
            return None
        return total if total >= 0 else None

    @staticmethod
    def _api_error(payload: dict[str, Any]) -> Optional[str]:
        error = payload.get("error") or payload.get("errors")
        if not error:
            return None
        if isinstance(error, (dict, list)):
            return str(error)
        return str(error)

    def search_page(
        self,
        query: str,
        cursor: Optional[dict] = None,
        limit: Optional[int] = None,
    ) -> SearchPage:
        """Fetch one BASE page using the provider's offset pagination."""
        state, remaining = self._state(cursor, limit)
        if remaining <= 0:
            return page([], total_available=state.get("total"))

        page_size = min(self.default_page_size, remaining)
        offset = int(state.get("offset", 0))
        api_key = os.getenv("BASE_API_KEY")
        if not api_key:
            raise RuntimeError("BASE_API_KEY is not configured")

        response = self._request(
            self.BASE_URL,
            params={
                "func": "PerformSearch",
                "query": query,
                "format": "json",
                "hits": page_size,
                "offset": offset,
                "apikey": api_key,
            },
        )
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise RuntimeError("BASE API returned malformed JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("BASE API returned malformed JSON")
        api_error = self._api_error(payload)
        if api_error:
            raise RuntimeError(f"BASE API error: {api_error}")

        result = payload.get("response")
        if not isinstance(result, dict):
            raise RuntimeError("BASE API response is missing the response object")
        docs = result.get("docs", [])
        if not isinstance(docs, list):
            raise RuntimeError("BASE API response has invalid docs")

        total = self._total(result.get("numFound"))
        items = []
        for record in docs:
            if not isinstance(record, dict):
                continue
            title = self._first(record, "dctitle", "title") or ""
            url = self._first(record, "dclink", "dcidentifier", "link", "url")
            items.append(self._result(
                title=title,
                abstract=self._first(record, "dcdescription", "dcabstract", "abstract", "description") or "",
                authors=self._authors(record),
                year=self._year(self._first(record, "dcyear", "dcdate", "date")),
                doi=self._doi(record),
                url=url,
                language=self._first(record, "dclang", "dclanguage", "language"),
                source_id=self._source_id(record),
                full_text_available=None,
                metadata={"provenance": {"provider": self.source}},
            ))

        collected = int(state.get("collected", 0)) + len(items)
        maximum = limit if limit is not None else self.max_results_without_limit
        target = min(total, maximum) if total is not None else maximum
        has_more = bool(items) and collected < target and (
            total is None or len(items) >= page_size or offset + len(items) < total
        )
        next_cursor = None
        if has_more:
            next_cursor = {
                "offset": offset + len(items),
                "collected": collected,
                "total": total,
            }
        return page(items, cursor=next_cursor, has_more=has_more, total_available=total)
