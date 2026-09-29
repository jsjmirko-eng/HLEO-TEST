import logging
from typing import List

from core.logging_utils import redact_text
from search.bing import BingSearch
from search.searxng import SearXNGSearch
from search.deduplicator import URLDeduplicator
from search.websearch import WebSearch

logger = logging.getLogger(__name__)


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
            logger.warning("Bing search failed: %s", redact_text(e, max_length=160))

        try:
            urls.extend(self.searxng.search(query))
        except Exception as e:
            logger.warning("SearXNG search failed: %s", redact_text(e, max_length=160))

        try:
            urls.extend(self.web.search(query))
        except Exception as e:
            logger.warning("Web search failed: %s", redact_text(e, max_length=160))

        urls = self.deduplicator.deduplicate(urls)

        return urls