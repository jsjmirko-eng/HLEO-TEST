import requests
import time
from typing import Optional

from core.search_result import SearchResult


class PubMedCollector:
    SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def search(self, query: str, limit: Optional[int] = None):
        # 1 — get IDs. Explicit limits remain supported for callers/tests;
        # None follows E-utilities pagination, bounded only by the API's
        # practical page ceiling to keep NCBI rate limits healthy.
        target = limit if limit is not None else 400
        page_size = max(1, min(target, 100))
        ids: list[str] = []
        retstart = 0
        total = None
        while len(ids) < target and (total is None or retstart < total):
            r = requests.get(
                self.SEARCH_URL,
                params={"db": "pubmed", "term": query, "retmax": page_size,
                        "retstart": retstart, "retmode": "json"},
                timeout=15,
            )
            if r.status_code == 429 and limit is None:
                time.sleep(0.8)
                r = requests.get(
                    self.SEARCH_URL,
                    params={"db": "pubmed", "term": query, "retmax": page_size,
                            "retstart": retstart, "retmode": "json"},
                    timeout=15,
                )
            r.raise_for_status()
            result = r.json()["esearchresult"]
            batch = result.get("idlist", [])
            ids.extend(batch)
            total = int(result.get("count", len(ids)) or 0)
            retstart += len(batch)
            if limit is not None or not batch or len(batch) < page_size:
                break
        if limit is not None:
            ids = ids[:limit]
        if not ids:
            return []

        time.sleep(0.4)

        # 2 — summary (title, authors, journal)
        r2 = requests.get(
            self.SUMMARY_URL,
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            timeout=15,
        )
        r2.raise_for_status()
        details = r2.json()

        time.sleep(0.4)

        # 3 — fetch abstracts as plain text, one call for all IDs
        abstract_map: dict[str, str] = {}
        try:
            r3 = requests.get(
                self.FETCH_URL,
                params={
                    "db": "pubmed",
                    "id": ",".join(ids),
                    "rettype": "abstract",
                    "retmode": "text",
                },
                timeout=20,
            )
            if r3.status_code == 200:
                # Split on numbered entries like "\n\n1. " or "\n\n2. "
                blocks = r3.text.split("\n\n\n")
                for i, pmid in enumerate(ids):
                    if i < len(blocks):
                        abstract_map[pmid] = blocks[i].strip()
        except Exception:
            pass

        results = []
        for pmid in ids:
            art = details["result"].get(pmid, {})
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
