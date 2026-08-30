from typing import List
from ddgs import DDGS


class WebSearch:

    def __init__(self):
        pass

    def search(self, query: str) -> List[str]:
        urls = []

        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=10)
                print(results)

                for result in results:
                    url = result.get("href")
                    if url:
                        urls.append(url)

        except Exception as e:
            print("WebSearch error:", e)

        return urls