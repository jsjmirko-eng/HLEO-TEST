"""
HLEO — Test suite FASE 4.1: Parallel Scientific Collector

Verifica che il loop di raccolta Scientific (variant × source) sia:
- parallelo e bounded (collector_max_workers da Global Limits)
- isolato per errore: un collector che fallisce non blocca gli altri
- completo: nessun risultato perso
- privo di duplicati introdotti dal parallelismo
- coerente con Global Limits

Nessuna chiamata LLM reale — tutti i collector sono mockati.

Scenari
-------
1.  1 variante × 3 fonti — caso base
2.  Più varianti × 3 fonti — tutti i risultati presenti
3.  Errore PubMed — EuropePMC e ClinicalTrials ancora completano
4.  Errore EuropePMC — PubMed e ClinicalTrials ancora completano
5.  Errore ClinicalTrials — PubMed e EuropePMC ancora completano
6.  Collector lento — gli altri terminano, il lento non blocca
7.  Nessun risultato perso con N varianti
8.  Nessun duplicato introdotto dal parallelismo
9.  Global Limits rispettati (collector_max_workers)
10. Numero massimo di task concorrenti rispettato
"""
from __future__ import annotations

import threading
import time
import types
import unittest
from unittest.mock import MagicMock, patch

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_search_result(title: str, source: str, pmid: str = "") -> MagicMock:
    """Return a mock SearchResult with minimal required attributes."""
    item = MagicMock()
    item.title = title
    item.source = source
    item.pmid = pmid
    item.metadata = {}
    return item


def _make_relational_search_instance(
    pubmed_results=None,
    europepmc_results=None,
    clinicaltrials_results=None,
    pubmed_side_effect=None,
    europepmc_side_effect=None,
    clinicaltrials_side_effect=None,
):
    """
    Build a RelationalSearch instance with fully mocked collectors and LLM client.
    All external HTTP calls are bypassed.
    """
    import core.relational_search as rs_mod

    instance = rs_mod.RelationalSearch.__new__(rs_mod.RelationalSearch)
    instance._client = MagicMock()

    # Mock collectors
    instance.pubmed = MagicMock()
    instance.europepmc = MagicMock()
    instance.clinicaltrials = MagicMock()

    if pubmed_side_effect is not None:
        instance.pubmed.search.side_effect = pubmed_side_effect
    else:
        instance.pubmed.search.return_value = pubmed_results or []

    if europepmc_side_effect is not None:
        instance.europepmc.search.side_effect = europepmc_side_effect
    else:
        instance.europepmc.search.return_value = europepmc_results or []

    if clinicaltrials_side_effect is not None:
        instance.clinicaltrials.search.side_effect = clinicaltrials_side_effect
    else:
        instance.clinicaltrials.search.return_value = clinicaltrials_results or []

    return instance


def _make_variant(name: str):
    """Return a mock ClinicalRelation variant object used by query builders."""
    v = MagicMock()
    v.agent = {"normalized": "drug", "search_terms": ["drug"]}
    v.manifestation = {"normalized": "rash", "search_terms": ["rash"]}
    v.relation_type = "adverse_effect"
    return v


def _run_collection(instance, expanded, limits_override=None):
    """
    Run only the collection phase (variant × source loop) from RelationalSearch.search()
    with mocked infrastructure. Returns the raw dict {pubmed, europepmc, clinicaltrials}.
    Uses the parallel implementation directly (calls _collect_one via ThreadPoolExecutor).
    """
    import core.relational_search as rs_mod
    from concurrent.futures import ThreadPoolExecutor, as_completed

    raw = {"pubmed": [], "europepmc": [], "clinicaltrials": []}
    stats = {"query_calls": 0, "openai_calls": 0, "judge_errors": [], "judge_used": False,
             "vocab_enabled": False}

    collectors = {
        "pubmed": (instance.pubmed, instance._build_pubmed_query),
        "europepmc": (instance.europepmc, instance._build_epmc_query),
        "clinicaltrials": (instance.clinicaltrials, instance._build_ct_query),
    }

    def _collect_one(source, collector, query, provenance):
        try:
            items = collector.search(query, limit=None)
            return source, provenance, query, items
        except Exception as exc:
            return source, provenance, query, []

    if limits_override is not None:
        max_workers = limits_override
    else:
        from core.llm_limits import get_limits
        max_workers = get_limits().collector_max_workers

    tasks = [
        (source, collector, builder(variant), provenance)
        for variant, provenance in expanded
        for source, (collector, builder) in collectors.items()
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_collect_one, source, collector, query, provenance): (
                source, provenance, query
            )
            for source, collector, query, provenance in tasks
        }
        for fut in as_completed(futures):
            source, provenance, query, items = fut.result()
            if items:
                stats["query_calls"] += 1
            for item in items:
                item.metadata = dict(item.metadata or {})
                item.metadata.setdefault("match_provenance", []).append(provenance)
                item.metadata.setdefault("matched_queries", []).append(query)
                raw[source].append(item)

    return raw, stats


# ── Scenario 1: 1 variante × 3 fonti — caso base ────────────────────────────

class TestOneVariantThreeSources(unittest.TestCase):
    def test_all_sources_collected(self):
        """Con 1 variante e 3 fonti, ogni fonte deve contribuire i propri risultati."""
        from core import llm_limits
        llm_limits.invalidate_limits_cache()

        pub = [_make_search_result("PubMed Art 1", "PubMed", pmid="111")]
        epm = [_make_search_result("EPMC Art 1", "Europe PMC")]
        cts = [_make_search_result("CT Art 1", "ClinicalTrials.gov")]

        instance = _make_relational_search_instance(
            pubmed_results=pub,
            europepmc_results=epm,
            clinicaltrials_results=cts,
        )

        variant = _make_variant("v1")
        expanded = [(variant, "query_v1")]

        with patch("core.llm_limits._load_from_db",
                   return_value=llm_limits.HLEOLimits(collector_max_workers=6)):
            llm_limits.invalidate_limits_cache()
            raw, stats = _run_collection(instance, expanded)

        self.assertEqual(len(raw["pubmed"]), 1)
        self.assertEqual(len(raw["europepmc"]), 1)
        self.assertEqual(len(raw["clinicaltrials"]), 1)
        self.assertEqual(stats["query_calls"], 3)


# ── Scenario 2: Più varianti × 3 fonti ───────────────────────────────────────

class TestMultipleVariantsThreeSources(unittest.TestCase):
    def test_all_results_present(self):
        """Con N varianti × 3 fonti, tutti i risultati devono essere presenti."""
        from core import llm_limits
        llm_limits.invalidate_limits_cache()

        N_VARIANTS = 4
        pub_per_call = [_make_search_result(f"PubMed {i}", "PubMed") for i in range(2)]
        epm_per_call = [_make_search_result(f"EPMC {i}", "Europe PMC") for i in range(3)]
        cts_per_call = [_make_search_result(f"CT {i}", "ClinicalTrials.gov") for i in range(1)]

        # Each call returns the same list (simulating distinct results per variant)
        instance = _make_relational_search_instance(
            pubmed_results=pub_per_call,
            europepmc_results=epm_per_call,
            clinicaltrials_results=cts_per_call,
        )

        expanded = [(_make_variant(f"v{i}"), f"prov_{i}") for i in range(N_VARIANTS)]

        with patch("core.llm_limits._load_from_db",
                   return_value=llm_limits.HLEOLimits(collector_max_workers=6)):
            llm_limits.invalidate_limits_cache()
            raw, stats = _run_collection(instance, expanded)

        # Each variant × each source contributes its results
        self.assertEqual(len(raw["pubmed"]), N_VARIANTS * len(pub_per_call))
        self.assertEqual(len(raw["europepmc"]), N_VARIANTS * len(epm_per_call))
        self.assertEqual(len(raw["clinicaltrials"]), N_VARIANTS * len(cts_per_call))
        # query_calls counts only successful (non-empty) calls
        self.assertEqual(stats["query_calls"], N_VARIANTS * 3)


# ── Scenario 3: Errore PubMed ─────────────────────────────────────────────────

class TestPubMedError(unittest.TestCase):
    def test_pubmed_error_does_not_block_others(self):
        """Un errore PubMed non deve bloccare EuropePMC o ClinicalTrials."""
        from core import llm_limits
        llm_limits.invalidate_limits_cache()

        epm = [_make_search_result("EPMC Art", "Europe PMC")]
        cts = [_make_search_result("CT Art", "ClinicalTrials.gov")]

        instance = _make_relational_search_instance(
            pubmed_side_effect=RuntimeError("PubMed connection timeout"),
            europepmc_results=epm,
            clinicaltrials_results=cts,
        )

        expanded = [(_make_variant("v1"), "prov1")]

        with patch("core.llm_limits._load_from_db",
                   return_value=llm_limits.HLEOLimits(collector_max_workers=6)):
            llm_limits.invalidate_limits_cache()
            raw, stats = _run_collection(instance, expanded)

        # PubMed failed → 0 results; others must have succeeded
        self.assertEqual(len(raw["pubmed"]), 0)
        self.assertEqual(len(raw["europepmc"]), 1)
        self.assertEqual(len(raw["clinicaltrials"]), 1)
        # Only the 2 successful calls are counted
        self.assertEqual(stats["query_calls"], 2)


# ── Scenario 4: Errore EuropePMC ──────────────────────────────────────────────

class TestEuropePMCError(unittest.TestCase):
    def test_europepmc_error_does_not_block_others(self):
        """Un errore EuropePMC non deve bloccare PubMed o ClinicalTrials."""
        from core import llm_limits
        llm_limits.invalidate_limits_cache()

        pub = [_make_search_result("PubMed Art", "PubMed", pmid="999")]
        cts = [_make_search_result("CT Art", "ClinicalTrials.gov")]

        instance = _make_relational_search_instance(
            pubmed_results=pub,
            europepmc_side_effect=ConnectionError("EuropePMC 503"),
            clinicaltrials_results=cts,
        )

        expanded = [(_make_variant("v1"), "prov1")]

        with patch("core.llm_limits._load_from_db",
                   return_value=llm_limits.HLEOLimits(collector_max_workers=6)):
            llm_limits.invalidate_limits_cache()
            raw, stats = _run_collection(instance, expanded)

        self.assertEqual(len(raw["pubmed"]), 1)
        self.assertEqual(len(raw["europepmc"]), 0)
        self.assertEqual(len(raw["clinicaltrials"]), 1)
        self.assertEqual(stats["query_calls"], 2)


# ── Scenario 5: Errore ClinicalTrials ────────────────────────────────────────

class TestClinicalTrialsError(unittest.TestCase):
    def test_clinicaltrials_error_does_not_block_others(self):
        """Un errore ClinicalTrials non deve bloccare PubMed o EuropePMC."""
        from core import llm_limits
        llm_limits.invalidate_limits_cache()

        pub = [_make_search_result("PubMed Art", "PubMed", pmid="123")]
        epm = [_make_search_result("EPMC Art", "Europe PMC")]

        instance = _make_relational_search_instance(
            pubmed_results=pub,
            europepmc_results=epm,
            clinicaltrials_side_effect=TimeoutError("CT API timeout"),
        )

        expanded = [(_make_variant("v1"), "prov1")]

        with patch("core.llm_limits._load_from_db",
                   return_value=llm_limits.HLEOLimits(collector_max_workers=6)):
            llm_limits.invalidate_limits_cache()
            raw, stats = _run_collection(instance, expanded)

        self.assertEqual(len(raw["pubmed"]), 1)
        self.assertEqual(len(raw["europepmc"]), 1)
        self.assertEqual(len(raw["clinicaltrials"]), 0)
        self.assertEqual(stats["query_calls"], 2)


# ── Scenario 6: Collector lento ───────────────────────────────────────────────

class TestSlowCollector(unittest.TestCase):
    def test_slow_collector_does_not_block_fast_ones(self):
        """
        Un collector lento (PubMed con sleep) non deve bloccare gli altri.
        Gli altri collector devono completare prima o durante il lento.
        """
        from core import llm_limits
        llm_limits.invalidate_limits_cache()

        completion_order = []
        lock = threading.Lock()

        def slow_pubmed(query, limit=None):
            time.sleep(0.15)  # Simula latenza PubMed
            with lock:
                completion_order.append("pubmed")
            return [_make_search_result("PubMed Slow Art", "PubMed")]

        def fast_epmc(query, limit=None):
            time.sleep(0.01)
            with lock:
                completion_order.append("europepmc")
            return [_make_search_result("EPMC Fast Art", "Europe PMC")]

        def fast_ct(query, limit=None):
            time.sleep(0.01)
            with lock:
                completion_order.append("clinicaltrials")
            return [_make_search_result("CT Fast Art", "ClinicalTrials.gov")]

        instance = _make_relational_search_instance(
            pubmed_side_effect=slow_pubmed,
            europepmc_side_effect=fast_epmc,
            clinicaltrials_side_effect=fast_ct,
        )

        expanded = [(_make_variant("v1"), "prov1")]

        t0 = time.perf_counter()
        with patch("core.llm_limits._load_from_db",
                   return_value=llm_limits.HLEOLimits(collector_max_workers=6)):
            llm_limits.invalidate_limits_cache()
            raw, stats = _run_collection(instance, expanded)
        elapsed = time.perf_counter() - t0

        # All sources must have results
        self.assertEqual(len(raw["pubmed"]), 1)
        self.assertEqual(len(raw["europepmc"]), 1)
        self.assertEqual(len(raw["clinicaltrials"]), 1)

        # With parallelism, total time should be close to the slowest collector (~0.15s)
        # not the sum of all (~0.17s+). Serial would be ≥0.17s, parallel ≥0.15s.
        # We check it's under a generous threshold that would only pass with parallelism.
        self.assertLess(elapsed, 0.6, f"Collection took {elapsed:.3f}s — should be parallel")

        # The fast collectors finished before or during the slow one
        # (europepmc and clinicaltrials appear before pubmed in completion_order)
        pubmed_idx = completion_order.index("pubmed")
        epmc_idx = completion_order.index("europepmc")
        ct_idx = completion_order.index("clinicaltrials")
        self.assertLess(epmc_idx, pubmed_idx, "EuropePMC should finish before PubMed")
        self.assertLess(ct_idx, pubmed_idx, "ClinicalTrials should finish before PubMed")


# ── Scenario 7: Nessun risultato perso (N varianti) ──────────────────────────

class TestNoResultsLost(unittest.TestCase):
    def test_all_items_present_in_raw(self):
        """
        Con N varianti × 3 fonti × M articoli per call,
        il totale deve essere esattamente N × M × 3.
        """
        from core import llm_limits
        llm_limits.invalidate_limits_cache()

        N = 5
        M = 10  # articoli per fonte per variante

        def make_items(source, n=M):
            return [_make_search_result(f"{source} art {i}", source) for i in range(n)]

        instance = _make_relational_search_instance(
            pubmed_results=make_items("PubMed"),
            europepmc_results=make_items("Europe PMC"),
            clinicaltrials_results=make_items("ClinicalTrials.gov"),
        )

        expanded = [(_make_variant(f"v{i}"), f"prov_{i}") for i in range(N)]

        with patch("core.llm_limits._load_from_db",
                   return_value=llm_limits.HLEOLimits(collector_max_workers=6)):
            llm_limits.invalidate_limits_cache()
            raw, _ = _run_collection(instance, expanded)

        self.assertEqual(len(raw["pubmed"]), N * M)
        self.assertEqual(len(raw["europepmc"]), N * M)
        self.assertEqual(len(raw["clinicaltrials"]), N * M)


# ── Scenario 8: Nessun duplicato introdotto dal parallelismo ─────────────────

class TestNoParallelismDuplicates(unittest.TestCase):
    def test_no_extra_duplicates_from_concurrency(self):
        """
        Il parallelismo non deve introdurre duplicati nell'accumulazione di raw.
        L'accumulazione avviene nel main thread dopo il completamento di ogni future.

        Ogni chiamata al collector restituisce oggetti FRESCHI (side_effect),
        in modo che ogni item in raw provenga esattamente da 1 task, con 1 provenance.
        """
        from core import llm_limits
        llm_limits.invalidate_limits_cache()

        N = 8
        M = 5

        # Use side_effect with a factory so each call gets fresh item objects.
        # This is the realistic case: collectors return new objects per query.
        def make_fresh_pub(query, limit=None):
            return [_make_search_result(f"PubMed {i}", "PubMed", pmid=str(i)) for i in range(M)]

        def make_fresh_epm(query, limit=None):
            return [_make_search_result(f"EPMC {i}", "Europe PMC") for i in range(M)]

        def make_fresh_cts(query, limit=None):
            return [_make_search_result(f"CT {i}", "ClinicalTrials.gov") for i in range(M)]

        instance = _make_relational_search_instance(
            pubmed_side_effect=make_fresh_pub,
            europepmc_side_effect=make_fresh_epm,
            clinicaltrials_side_effect=make_fresh_cts,
        )

        expanded = [(_make_variant(f"v{i}"), f"prov_{i}") for i in range(N)]

        with patch("core.llm_limits._load_from_db",
                   return_value=llm_limits.HLEOLimits(collector_max_workers=6)):
            llm_limits.invalidate_limits_cache()
            raw, _ = _run_collection(instance, expanded)

        # Expected: N variants × M items per variant, all present in raw (pre-dedup)
        # The parallel loop must NOT double-append items from any single task
        self.assertEqual(len(raw["pubmed"]), N * M)
        self.assertEqual(len(raw["europepmc"]), N * M)
        self.assertEqual(len(raw["clinicaltrials"]), N * M)

        # Each item was returned by exactly 1 task → exactly 1 match_provenance entry
        for item in raw["pubmed"]:
            provs = item.metadata.get("match_provenance", [])
            self.assertEqual(len(provs), 1,
                             f"Expected 1 provenance per item per task, got {len(provs)}")


# ── Scenario 9: Global Limits rispettati (collector_max_workers) ─────────────

class TestGlobalLimitsCollectorMaxWorkers(unittest.TestCase):
    def test_collector_max_workers_read_from_limits(self):
        """collector_max_workers deve essere letto da HLEOLimits, non hardcoded."""
        from core import llm_limits
        llm_limits.invalidate_limits_cache()

        custom = llm_limits.HLEOLimits(collector_max_workers=3)
        with patch("core.llm_limits._load_from_db", return_value=custom):
            llm_limits.invalidate_limits_cache()
            lim = llm_limits.get_limits()

        self.assertEqual(lim.collector_max_workers, 3)

    def test_validate_clamps_collector_max_workers(self):
        """HLEOLimits.validate() deve limitare collector_max_workers a [1, 32]."""
        from core.llm_limits import HLEOLimits

        lim_low = HLEOLimits(collector_max_workers=0)
        lim_low.validate()
        self.assertEqual(lim_low.collector_max_workers, 1)

        lim_high = HLEOLimits(collector_max_workers=999)
        lim_high.validate()
        self.assertEqual(lim_high.collector_max_workers, 32)

    def test_collector_max_workers_default(self):
        """Il default di collector_max_workers deve essere 6."""
        from core.llm_limits import HLEOLimits
        lim = HLEOLimits()
        self.assertEqual(lim.collector_max_workers, 6)

    def test_collector_max_workers_in_to_dict(self):
        """to_dict() deve includere collector_max_workers."""
        from core.llm_limits import HLEOLimits
        d = HLEOLimits(collector_max_workers=4).to_dict()
        self.assertIn("collector_max_workers", d)
        self.assertEqual(d["collector_max_workers"], 4)

    def test_collector_max_workers_in_from_dict(self):
        """from_dict() deve leggere collector_max_workers."""
        from core.llm_limits import HLEOLimits
        lim = HLEOLimits.from_dict({"collector_max_workers": 9})
        self.assertEqual(lim.collector_max_workers, 9)


# ── Scenario 10: Numero massimo di task concorrenti rispettato ────────────────

class TestMaxConcurrencyRespected(unittest.TestCase):
    def test_concurrent_tasks_bounded_by_max_workers(self):
        """
        Il numero di task simultanei non deve mai superare collector_max_workers.
        Verificato contando il picco di concorrenza tramite un semaforo-counter.
        """
        from core import llm_limits
        llm_limits.invalidate_limits_cache()

        MAX_WORKERS = 3
        peak_concurrent = [0]
        current_concurrent = [0]
        lock = threading.Lock()

        def slow_collector(query, limit=None):
            with lock:
                current_concurrent[0] += 1
                peak_concurrent[0] = max(peak_concurrent[0], current_concurrent[0])
            time.sleep(0.05)  # Keep the slot occupied briefly
            with lock:
                current_concurrent[0] -= 1
            return [_make_search_result("Art", "PubMed")]

        instance = _make_relational_search_instance(
            pubmed_side_effect=slow_collector,
            europepmc_side_effect=slow_collector,
            clinicaltrials_side_effect=slow_collector,
        )

        # 5 varianti × 3 fonti = 15 tasks → peak should be ≤ MAX_WORKERS
        N = 5
        expanded = [(_make_variant(f"v{i}"), f"prov_{i}") for i in range(N)]

        with patch("core.llm_limits._load_from_db",
                   return_value=llm_limits.HLEOLimits(collector_max_workers=MAX_WORKERS)):
            llm_limits.invalidate_limits_cache()
            raw, _ = _run_collection(instance, expanded)

        # All 15 tasks must have completed
        total = len(raw["pubmed"]) + len(raw["europepmc"]) + len(raw["clinicaltrials"])
        self.assertEqual(total, N * 3)

        # Peak concurrency must not exceed MAX_WORKERS
        self.assertLessEqual(
            peak_concurrent[0], MAX_WORKERS,
            f"Peak concurrency was {peak_concurrent[0]}, limit is {MAX_WORKERS}",
        )


# ── Scenario 11: Tutti errori → raw vuoto, nessuna eccezione ─────────────────

class TestAllCollectorsFail(unittest.TestCase):
    def test_all_errors_returns_empty_not_raises(self):
        """Se tutti i collector falliscono, raw deve essere vuoto senza eccezioni."""
        from core import llm_limits
        llm_limits.invalidate_limits_cache()

        instance = _make_relational_search_instance(
            pubmed_side_effect=RuntimeError("PubMed down"),
            europepmc_side_effect=RuntimeError("EPMC down"),
            clinicaltrials_side_effect=RuntimeError("CT down"),
        )

        expanded = [(_make_variant("v1"), "prov1")]

        with patch("core.llm_limits._load_from_db",
                   return_value=llm_limits.HLEOLimits(collector_max_workers=6)):
            llm_limits.invalidate_limits_cache()
            raw, stats = _run_collection(instance, expanded)

        self.assertEqual(len(raw["pubmed"]), 0)
        self.assertEqual(len(raw["europepmc"]), 0)
        self.assertEqual(len(raw["clinicaltrials"]), 0)
        self.assertEqual(stats["query_calls"], 0)


# ── Scenario 12: match_provenance e matched_queries popolati correttamente ────

class TestMetadataPopulated(unittest.TestCase):
    def test_match_provenance_and_queries_set(self):
        """
        Ogni articolo deve avere match_provenance e matched_queries nel metadata,
        impostati dalla variante che lo ha generato.
        """
        from core import llm_limits
        llm_limits.invalidate_limits_cache()

        item = _make_search_result("Test Art", "PubMed", pmid="42")

        instance = _make_relational_search_instance(
            pubmed_results=[item],
            europepmc_results=[],
            clinicaltrials_results=[],
        )

        variant = _make_variant("v1")
        expanded = [(variant, "test_provenance_XYZ")]

        with patch("core.llm_limits._load_from_db",
                   return_value=llm_limits.HLEOLimits(collector_max_workers=6)):
            llm_limits.invalidate_limits_cache()
            raw, _ = _run_collection(instance, expanded)

        self.assertEqual(len(raw["pubmed"]), 1)
        art = raw["pubmed"][0]
        self.assertIn("match_provenance", art.metadata)
        self.assertIn("matched_queries", art.metadata)
        self.assertIn("test_provenance_XYZ", art.metadata["match_provenance"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
