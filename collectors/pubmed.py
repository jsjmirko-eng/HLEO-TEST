import time
from typing import Optional

from core.search_result import SearchResult


_BATCH_SIZE = 100


def _lim():
    try:
        from core.llm_limits import get_limits
        return get_limits()
    except Exception:
        from core.llm_limits import HLEOLimits
        return HLEOLimits()


class PubMedCollector:
    SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

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
        inter_sleep = lim.pubmed_inter_call_sleep_s

        target = limit if limit is not None else 400
        page_size = max(1, min(target, 100))
        state = cursor or {"retstart": 0, "collected": 0, "total": None}
        retstart = int(state.get("retstart", 0))
        collected = int(state.get("collected", 0))

        r = http_get(
            self.SEARCH_URL,
            params={"db": "pubmed", "term": query, "retmax": page_size,
                    "retstart": retstart, "retmode": "json"},
            timeout=timeout,
            max_retries=max_retries,
            backoff_base_s=backoff_base,
            backoff_max_s=backoff_max,
        )
        result = r.json()["esearchresult"]
        ids = result.get("idlist", [])
        total = int(result.get("count", state.get("total") or 0) or 0)
        collected += len(ids)

        if limit is not None:
            ids = ids[:max(0, limit - (collected - len(ids)))]

        if not ids:
            return SearchPage([], total_available=total)

        details: dict[str, dict] = {}
        try:
            time.sleep(inter_sleep)
            response = http_get(
                self.SUMMARY_URL,
                params={"db": "pubmed", "id": ",".join(ids),
                        "retmode": "json"},
                timeout=timeout,
                max_retries=max_retries,
                backoff_base_s=backoff_base,
                backoff_max_s=backoff_max,
            )
            details.update(response.json().get("result", {}))
        except Exception:
            pass

        abstract_map: dict[str, str] = {}
        try:
            time.sleep(inter_sleep)
            response = http_get(
                self.FETCH_URL,
                params={
                    "db": "pubmed",
                    "id": ",".join(ids),
                    "rettype": "abstract",
                    "retmode": "text",
                },
                timeout=timeout,
                max_retries=max_retries,
                backoff_base_s=backoff_base,
                backoff_max_s=backoff_max,
            )
            if response.status_code == 200:
                blocks = response.text.split("\n\n\n")
                for i, pmid in enumerate(ids):
                    if i < len(blocks):
                        abstract_map[pmid] = blocks[i].strip()
        except Exception:
            pass

        results = []
        for pmid in ids:
            art = details.get(pmid, {})
            results.append(
                SearchResult(
                    title=art.get("title", ""),
                    source="PubMed",
                    authors=[a.get("name", "") for a in art.get("authors", [])],
                    pmid=pmid,
                    abstract=abstract_map.get(pmid, ""),
                    metadata={
                        "journal": art.get("fulljournalname", ""),
                        "pubdate": art.get("pubdate", ""),
                    },
                )
            )

        next_start = retstart + len(ids)
        has_more = bool(ids) and len(ids) == page_size and next_start < total
        if limit is not None:
            has_more = has_more and collected < limit
        next_cursor = (
            {"retstart": next_start, "collected": collected, "total": total}
            if has_more else None
        )
        return SearchPage(results, next_cursor, has_more, total)

    def search(self, query: str, limit: Optional[int] = None):
        from core.search_page import collect_search_pages

        return collect_search_pages(self, query, limit=limit)
