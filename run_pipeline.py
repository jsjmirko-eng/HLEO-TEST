from core.pipeline import HLEOPipeline

pipeline = HLEOPipeline()

query = "finasteride"

print("=" * 50)
print("Avvio HLEO...")
print("=" * 50)

results = pipeline.process(query)

print(f"\nRisultati trovati: {len(results)}")

for i, result in enumerate(results, 1):
    print(f"\n===== RISULTATO {i} =====")
    print(result)