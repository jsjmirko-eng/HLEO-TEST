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

    def search(self, query: str, limit: Optional[int] = None):
        from core.http_retry import http_get

        lim = _lim()
        timeout = lim.collector_timeout_s
        max_retries = lim.collector_max_retries
        backoff_base = lim.backoff_base_s
        backoff_max = lim.backoff_max_s
        inter_sleep = lim.pubmed_inter_call_sleep_s

        # 1 — get IDs (paginated)
        target = limit if limit is not None else 400
        page_size = max(1, min(target, 100))
        ids: list[str] = []
        retstart = 0
        total = None
        while len(ids) < target and (total is None or retstart < total):
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
            batch = result.get("idlist", [])
            ids.extend(batch)
            total = int(result.get("count", len(ids)) or 0)
            retstart += len(batch)
            if not batch or len(batch) < page_size or len(ids) >= target:
                break

        if limit is not None:
            ids = ids[:limit]

        # Sleep only when there are IDs to retrieve — skip if query returned nothing
        if not ids:
            return []

        # 2 — summary (title, authors, journal), in bounded batches
        details: dict[str, dict] = {}
        for start in range(0, len(ids), _BATCH_SIZE):
            batch = ids[start:start + _BATCH_SIZE]
            try:
                time.sleep(inter_sleep)
                response = http_get(
                    self.SUMMARY_URL,
                    params={"db": "pubmed", "id": ",".join(batch),
                            "retmode": "json"},
                    timeout=timeout,
                    max_retries=max_retries,
                    backoff_base_s=backoff_base,
                    backoff_max_s=backoff_max,
                )
                details.update(response.json().get("result", {}))
            except Exception:
                continue

        # 3 — fetch abstracts as plain text, in the same bounded batches
        abstract_map: dict[str, str] = {}
        for start in range(0, len(ids), _BATCH_SIZE):
            batch = ids[start:start + _BATCH_SIZE]
            try:
                time.sleep(inter_sleep)
                response = http_get(
                    self.FETCH_URL,
                    params={
                        "db": "pubmed",
                        "id": ",".join(batch),
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
                    for i, pmid in enumerate(batch):
                        if i < len(blocks):
                            abstract_map[pmid] = blocks[i].strip()
            except Exception:
                continue

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
        return results
