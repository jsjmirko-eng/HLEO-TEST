import requests
from typing import Optional

from core.search_result import SearchResult


class EuropePMCCollector:
    BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def search(self, query: str, limit: Optional[int] = None):
        target = limit if limit is not None else 400
        page_size = max(1, min(target, 100))
        results = []
        page = 1
        total = None
        while len(results) < target and (total is None or len(results) < total):
            r = requests.get(
                self.BASE_URL,
                params={"query": query, "format": "json", "pageSize": page_size,
                        "page": page, "resultType": "core"},
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            result_list = data.get("resultList", {}).get("result", [])
            if isinstance(result_list, dict):
                result_list = [result_list]
            total = int(data.get("hitCount", len(results) + len(result_list)) or 0)
            for item in result_list:
                results.append(item)
            if limit is not None or not result_list or len(result_list) < page_size:
                break
            page += 1
        if limit is not None:
            results = results[:limit]

        output = []
        for item in results:
            abstract = item.get("abstractText") or item.get("abstract", "")
            # Extract authors list (authorList → author[].fullName or .lastName + .firstName)
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
                    },
                )
            )
        return output
