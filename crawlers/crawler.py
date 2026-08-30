import requests
from bs4 import BeautifulSoup


class ThreadCrawler:

    def __init__(self):
        self.session = requests.Session()

        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        })

    def fetch(self, url: str):

        try:

            response = self.session.get(
                url,
                timeout=30,
                allow_redirects=True
            )

            print("STATUS:", response.status_code)

            if response.status_code != 200:
                return None

            return response.text

        except Exception as e:

            print("Crawler error:", e)

            return None

    def extract_text(self, html: str):

        if html is None:
            return ""

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        return "\n".join(lines)