from aggregator import HLEOAggregator

collector = HLEOAggregator()

results = collector.search("finasteride", limit=3)

print(f"\nTotale risultati: {len(results)}\n")

for i, result in enumerate(results, 1):
    print(f"{i}. {result}")