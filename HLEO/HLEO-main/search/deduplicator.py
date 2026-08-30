from typing import List


class URLDeduplicator:

    def deduplicate(self, urls: List[str]) -> List[str]:
        seen = set()
        unique = []

        for url in urls:
            if url not in seen:
                seen.add(url)
                unique.append(url)

        return unique