from typing import List
import requests


class BingSearch:

    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    def search(self, query: str) -> List[str]:
        return []