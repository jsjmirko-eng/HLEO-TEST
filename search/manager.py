from typing import List

from search.bing import BingSearch
from search.searxng import SearXNGSearch
from search.deduplicator import URLDeduplicator
from search.websearch import WebSearch


class SearchManager:

    def __init__(self, brave_api_key: str = "", bing_api_key: str = ""):
        self.bing = BingSearch(bing_api_key)
        self.searxng = SearXNGSearch()
        self.web = WebSearch()
        self.deduplicator = URLDeduplicator()

    def search(self, query: str) -> List[str]:

        urls = []

        try:
            urls.extend(self.bing.search(query))
        except Exception as e:
            print(f"Bing error: {e}")

        try:
            urls.extend(self.searxng.search(query))
        except Exception as e:
            print(f"SearXNG error: {e}")

        try:
            urls.extend(self.web.search(query))
        except Exception as e:
            print(f"Web error: {e}")

        urls = self.deduplicator.deduplicate(urls)

        return urls