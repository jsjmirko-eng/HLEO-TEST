from collectors.clinicaltrials import ClinicalTrialsCollector

collector = ClinicalTrialsCollector()

results = collector.search("finasteride", limit=3)

print(f"Trovati {len(results)} studi\n")

for study in results:
    print(study)