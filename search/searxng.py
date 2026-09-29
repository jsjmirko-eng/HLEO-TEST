import logging
from typing import List

import requests

from core.logging_utils import query_fingerprint, redact_text

logger = logging.getLogger(__name__)


class SearXNGSearch:

    def __init__(self, base_url="https://search.inetol.net"):
        self.base_url = base_url.rstrip("/")

    def search(self, query: str) -> List[str]:
        logger.info("SearXNG request for %s", query_fingerprint(query))

        url = f"{self.base_url}/search"

        params = {
            "q": query,
            "format": "json"
        }

        try:
            response = requests.get(url, params=params, timeout=20)
            
            if response.status_code != 200:
                return []

            logger.debug(
                "SearXNG response status=%s bytes=%d",
                response.status_code, len(response.content),
            )

            if "application/json" not in response.headers.get("Content-Type", ""):
                return []

            data = response.json()

            urls = []

            for result in data.get("results", []):
                if "url" in result:
                    urls.append(result["url"])

            logger.debug("SearXNG returned %d URLs", len(urls))
            return urls

        except Exception as e:
            logger.warning("SearXNG request failed for %s: %s", query_fingerprint(query), redact_text(e, max_length=160))
            raise
