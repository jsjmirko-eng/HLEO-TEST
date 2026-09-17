import time
from typing import Optional

from core.search_result import SearchResult


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
            if limit is not None or not batch or len(batch) < page_size:
                break

        if limit is not None:
            ids = ids[:limit]

        # Sleep only when there are IDs to retrieve — skip if query returned nothing
        if not ids:
            return []

        results = []
        batch_size = 100
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start:start + batch_size]

            time.sleep(inter_sleep)

            # 2 — summary (title, authors, journal)
            try:
                r2 = http_get(
                    self.SUMMARY_URL,
                    params={"db": "pubmed", "id": ",".join(batch_ids), "retmode": "json"},
                    timeout=timeout,
                    max_retries=max_retries,
                    backoff_base_s=backoff_base,
                    backoff_max_s=backoff_max,
                )
                details = r2.json()
            except Exception:
                continue

            time.sleep(inter_sleep)

            # 3 — fetch abstracts as plain text, one call per batch
            abstract_map: dict[str, str] = {}
            try:
                r3 = http_get(
                    self.FETCH_URL,
                    params={
                        "db": "pubmed",
                        "id": ",".join(batch_ids),
                        "rettype": "abstract",
                        "retmode": "text",
                    },
                    timeout=timeout,
                    max_retries=max_retries,
                    backoff_base_s=backoff_base,
                    backoff_max_s=backoff_max,
                )
                if r3.status_code == 200:
                    blocks = r3.text.split("\n\n\n")
                    for i, pmid in enumerate(batch_ids):
                        if i < len(blocks):
                            abstract_map[pmid] = blocks[i].strip()
            except Exception:
                pass

            for pmid in batch_ids:
                art = details.get("result", {}).get(pmid, {})
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
