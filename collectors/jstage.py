"""J-STAGE WebAPI collector."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional

from collectors.scientific_base import ScientificCollectorBase
from core.search_page import SearchPage


class JStageCollector(ScientificCollectorBase):
    """Search public J-STAGE articles through the official WebAPI."""

    source = "J-STAGE"
    BASE_URL = "https://api.jstage.jst.go.jp/searchapi/do"
    default_page_size = 1000
    min_interval_s = 0.1

    ATOM = "http://www.w3.org/2005/Atom"
    PRISM = "http://prismstandard.org/namespaces/basic/2.0/"
    OPENSEARCH = "http://a9.com/-/spec/opensearch/1.1/"

    @classmethod
    def _find(cls, element: ET.Element, namespace: str, name: str) -> Optional[ET.Element]:
        return element.find(f"{{{namespace}}}{name}")

    @classmethod
    def _text(cls, element: Optional[ET.Element]) -> str:
        if element is None:
            return ""
        return "".join(element.itertext()).strip()

    @classmethod
    def _localized(cls, entry: ET.Element, name: str) -> str:
        container = cls._find(entry, cls.ATOM, name)
        if container is None:
            return ""
        for locale in ("en", "ja"):
            value = cls._find(container, cls.ATOM, locale)
            text = cls._text(value)
            if text:
                return text
        return cls._text(container)

    @classmethod
    def _authors(cls, entry: ET.Element) -> list[str]:
        container = cls._find(entry, cls.ATOM, "author")
        if container is None:
            return []
        for locale in ("en", "ja"):
            localized = cls._find(container, cls.ATOM, locale)
            names = [
                cls._text(name)
                for name in localized.findall(f"{{{cls.ATOM}}}name")
                if cls._text(name)
            ] if localized is not None else []
            if names:
                return names
        return [
            cls._text(name)
            for name in container.findall(f".//{{{cls.ATOM}}}name")
            if cls._text(name)
        ]

    @classmethod
    def _field(cls, entry: ET.Element, namespace: str, name: str) -> str:
        return cls._text(cls._find(entry, namespace, name))

    def search_page(
        self,
        query: str,
        cursor: Optional[dict] = None,
        limit: Optional[int] = None,
    ) -> SearchPage:
        """Fetch one J-STAGE WebAPI page using its documented offset paging."""
        state, remaining = self._state(cursor, limit)
        if remaining <= 0:
            return SearchPage([], None, False, state.get("total"))

        page_size = min(self.default_page_size, remaining)
        start = int(state.get("start", 1))
        params = {
            "service": 3,
            "text": query,
            "start": start,
            "count": page_size,
        }
        response = self._request(self.BASE_URL, params=params)
        root = ET.fromstring(response.content)
        result = self._find(root, self.ATOM, "result")
        status = self._text(self._find(result, self.ATOM, "status")) if result is not None else ""
        message = self._text(self._find(result, self.ATOM, "message")) if result is not None else ""
        if status and status not in {"0", "WARN_002"}:
            if status == "ERR_001":
                return SearchPage([], None, False, 0)
            raise RuntimeError(f"J-STAGE WebAPI {status}: {message or status}")

        total_element = root.find(f"{{{self.OPENSEARCH}}}totalResults")
        total = int(self._text(total_element) or 0)
        entries = root.findall(f"{{{self.ATOM}}}entry")
        items = []
        for entry in entries:
            title = self._clean_text(self._localized(entry, "article_title"))
            article_link = self._localized(entry, "article_link")
            article_id = self._text(self._find(entry, self.ATOM, "id")) or article_link
            year = self._year(self._field(entry, self.ATOM, "pubyear"))
            doi = self._field(entry, self.PRISM, "doi") or None
            items.append(self._result(
                title=title,
                original_title=title,
                abstract="",
                authors=self._authors(entry),
                year=year,
                doi=doi,
                url=article_link or article_id,
                full_text_url=None,
                language=None,
                source_id=article_id,
                full_text_available=False,
                metadata={
                    "journal": self._clean_text(self._localized(entry, "material_title")),
                    "joi": self._field(entry, self.ATOM, "joi") or None,
                    "cdjournal": self._field(entry, self.ATOM, "cdjournal") or None,
                    "system_name": self._field(entry, self.ATOM, "systemname") or None,
                    "provenance": {
                        "provider": self.source,
                        "article_url": article_link or None,
                        "article_id": article_id or None,
                    },
                },
            ))

        collected = int(state.get("collected", 0)) + len(items)
        next_start = start + len(items)
        target = min(total or collected, limit or self.max_results_without_limit)
        has_more = bool(items and collected < target and next_start <= total)
        next_state = None
        if has_more:
            next_state = {
                "start": next_start,
                "collected": collected,
                "total": total,
            }
        return SearchPage(items, next_state, has_more, total)
