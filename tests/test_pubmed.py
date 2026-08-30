from collectors.pubmed import PubMedCollector
collector = PubMedCollector()

ids = collector.search("finasteride", limit=5)

articles = collector.search("finasteride", limit=3)

for article in articles:
    print(article)