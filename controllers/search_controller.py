from core.pipeline import HLEOPipeline


class SearchController:

    def __init__(self):
        self.pipeline = HLEOPipeline()

    def search(self, query: str):
        query = query.strip()

        if not query:
            return []

        return self.pipeline.process(query)