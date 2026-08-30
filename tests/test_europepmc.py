from collectors.europepmc import EuropePMCCollector

collector = EuropePMCCollector()

results = collector.search("finasteride", limit=3)

print(f"Trovati {len(results)} articoli\n")

for article in results:
    print(article)