import logging
from typing import List

from ddgs import DDGS

from core.logging_utils import redact_text

logger = logging.getLogger(__name__)


class WebSearch:

    def __init__(self):
        pass

    def search(self, query: str) -> List[str]:
        urls = []

        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=10)

                for result in results:
                    url = result.get("href")
                    if url:
                        urls.append(url)

        except Exception as e:
            logger.warning("WebSearch failed: %s", redact_text(e, max_length=160))

        return urls