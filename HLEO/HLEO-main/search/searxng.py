from typing import List
import requests


class SearXNGSearch:

    def __init__(self, base_url="https://search.inetol.net"):
        self.base_url = base_url.rstrip("/")

    def search(self, query: str) -> List[str]:
        print("SEARX CHIAMATO:", query)
        
        url = f"{self.base_url}/search"

        params = {
            "q": query,
            "format": "json"
        }

        try:
            response = requests.get(url, params=params, timeout=20)
            
            if response.status_code != 200:
                return []

            print(response.text)

            if "application/json" not in response.headers.get("Content-Type", ""):
                return []

            data = response.json()

            urls = []

            for result in data.get("results", []):
                if "url" in result:
                    urls.append(result["url"])

            print(urls)
            return urls

        except Exception as e:
            print("URL:", url)
            print("ERRORE:", e)
            raise