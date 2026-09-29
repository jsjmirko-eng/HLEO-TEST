from typing import Optional

from core.search_result import SearchResult


def _lim():
    try:
        from core.llm_limits import get_limits
        return get_limits()
    except Exception:
        from core.llm_limits import HLEOLimits
        return HLEOLimits()


class EuropePMCCollector:
    BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def search_page(
        self,
        query: str,
        cursor: Optional[dict] = None,
        limit: Optional[int] = None,
    ):
        from core.http_retry import http_get
        from core.search_page import SearchPage

        lim = _lim()
        timeout = lim.collector_timeout_s
        max_retries = lim.collector_max_retries
        backoff_base = lim.backoff_base_s
        backoff_max = lim.backoff_max_s

        target = limit if limit is not None else 400
        page_size = max(1, min(target, 100))
        state = cursor or {"cursor_mark": None, "collected": 0, "seen_ids": []}
        seen_ids = set(state.get("seen_ids", []))
        params = {
            "query": query,
            "format": "json",
            "pageSize": page_size,
            "resultType": "core",
        }
        if state.get("cursor_mark"):
            params["cursorMark"] = state["cursor_mark"]

        r = http_get(
            self.BASE_URL,
            params=params,
            timeout=timeout,
            max_retries=max_retries,
            backoff_base_s=backoff_base,
            backoff_max_s=backoff_max,
        )
        data = r.json()
        result_list = data.get("resultList", {}).get("result", [])
        if isinstance(result_list, dict):
            result_list = [result_list]
        total = int(data.get("hitCount", 0) or 0)

        unique_items = []
        for item in result_list:
            item_id = item.get("id") or item.get("pmid") or item.get("doi")
            if item_id and item_id in seen_ids:
                continue
            if item_id:
                seen_ids.add(item_id)
            unique_items.append(item)

        if limit is not None:
            remaining = max(0, limit - int(state.get("collected", 0)))
            unique_items = unique_items[:remaining]

        output = []
        for item in unique_items:
            abstract = item.get("abstractText") or item.get("abstract", "")
            author_list = item.get("authorList", {}).get("author", [])
            authors = []
            for a in author_list:
                name = a.get("fullName") or (
                    " ".join(filter(None, [a.get("firstName", ""), a.get("lastName", "")]))
                )
                if name:
                    authors.append(name)
            output.append(
                SearchResult(
                    title=item.get("title", ""),
                    source="Europe PMC",
                    abstract=abstract,
                    authors=authors,
                    year=int(item["pubYear"]) if item.get("pubYear") else None,
                    doi=item.get("doi"),
                    metadata={
                        "journal": item.get("journalTitle", ""),
                        "id": item.get("id"),
                        "hitCount": total,
                    },
                )
            )

        next_mark = data.get("nextCursorMark")
        collected = int(state.get("collected", 0)) + len(output)
        has_more = bool(next_mark and result_list and next_mark != state.get("cursor_mark"))
        if limit is not None:
            has_more = has_more and collected < limit
        next_cursor = (
            {"cursor_mark": next_mark, "collected": collected,
             "seen_ids": sorted(seen_ids)}
            if has_more else None
        )
        return SearchPage(output, next_cursor, has_more, total)

    def search(self, query: str, limit: Optional[int] = None):
        from core.search_page import collect_search_pages

        return collect_search_pages(self, query, limit=limit)
