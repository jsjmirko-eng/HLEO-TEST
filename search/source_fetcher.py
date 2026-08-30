import re
import requests
from bs4 import BeautifulSoup


class SourceFetcher:
    URL_PATTERN = re.compile(r"https?://\S+")

    @classmethod
    def extract_urls(cls, text: str) -> list[str]:
        return cls.URL_PATTERN.findall(text)

    @classmethod
    def fetch(cls, text: str) -> dict[str, str]:
        raw_texts = {}

        for url in cls.extract_urls(text):
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")

                content = soup.get_text(" ", strip=True)

                raw_texts[url] = content

            except Exception:
                raw_texts[url] = ""

        return raw_texts