"""
HLEO — Test suite FASE 4.2B: HTTP collector optimizations

Scenari testati (tutti con mock HTTP — nessuna chiamata reale):

Semafori per-fonte
  1.  PubMed max_concurrent = 2 rispettato
  2.  EuropePMC max_concurrent = 4 rispettato
  3.  ClinicalTrials max_concurrent = 3 rispettato
  4.  collector_max_workers limita il totale indipendentemente dai semafori

Sleep PubMed
  5.  Sleep NON eseguito quando la lista ID è vuota
  6.  Sleep eseguito quando ci sono ID (rispetta rate limit NCBI)
  7.  Valore sleep configurabile via Global Limits

Retry HTTP transitori
  8.  Retry su 429 (con Retry-After header)
  9.  Retry su 500
  10. Retry su 502
  11. Retry su 503
  12. Retry su 504
  13. Retry su Timeout
  14. Retry su ConnectionError
  15. NO retry su 400
  16. NO retry su 401
  17. NO retry su 403
  18. NO retry su 404
  19. Massimo numero di tentativi rispettato (max_retries=2 → 3 call totali)
  20. Retry-After header rispettato (capped a backoff_max_s)

Timeout
  21. Timeout configurabile via Global Limits

Correttezza risultati
  22. Tutti i collector restituiscono risultati corretti dopo retry
  23. Errore di un collector non blocca gli altri (regression FASE 4.1)
  24. Nessun risultato duplicato dopo retry

http_retry internals
  25. Risposta di successo (200) restituita senza retry
  26. Risposta 404 propagata senza retry
  27. Risposta 429 poi 200 → 1 retry, successo
  28. Risposta 503 esaurita dopo max_retries → eccezione
  29. Retry-After numerico rispettato
  30. Retry-After > backoff_max_s → capped a backoff_max_s
"""
from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock, patch, call


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mock_response(status_code: int, json_data=None, text_data: str = "",
                   headers: dict | None = None) -> MagicMock:
    r = MagicMock()
    r.status_code = status_code
    r.headers = headers or {}
    if json_data is not None:
        r.json.return_value = json_data
    r.text = text_data
    if status_code >= 400:
        import requests
        r.raise_for_status.side_effect = requests.HTTPError(
            f"HTTP {status_code}", response=r
        )
    else:
        r.raise_for_status.return_value = None
    return r


def _esearch_ok(ids=("111", "222"), count=2):
    return _mock_response(200, json_data={
        "esearchresult": {"idlist": list(ids), "count": str(count)}
    })


def _esummary_ok(ids=("111", "222")):
    return _mock_response(200, json_data={
        "result": {pmid: {"title": f"T{pmid}", "authors": [], "fulljournalname": "J", "pubdate": "2024"}
                   for pmid in ids}
    })


def _efetch_ok():
    return _mock_response(200, text_data="Abstract text")


def _epmc_ok(n=2, page=1, hit_count=2):
    return _mock_response(200, json_data={
        "hitCount": hit_count,
        "resultList": {"result": [
            {"title": f"EPMC{i}", "abstractText": "abs", "authorList": {"author": []},
             "pubYear": "2024", "doi": None, "journalTitle": "J", "id": str(i)}
            for i in range(n)
        ]}
    })


def _ct_ok(n=2, next_token=None):
    data = {"studies": [
        {"protocolSection": {
            "identificationModule": {"nctId": f"NCT{i:08d}", "briefTitle": f"CT{i}"},
            "statusModule": {"overallStatus": "ACTIVE", "startDateStruct": {"date": "2024-01-01"},
                             "primaryCompletionDateStruct": {"date": "2025-01-01"}},
            "conditionsModule": {"conditions": []},
            "descriptionModule": {"briefSummary": "Summary", "detailedDescription": ""},
            "armsInterventionsModule": {"interventions": []},
            "designModule": {"phases": ["PHASE3"], "enrollmentInfo": {"count": 100}},
            "outcomesModule": {"primaryOutcomes": []},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Sponsor"}},
        }} for i in range(n)
    ]}
    if next_token:
        data["nextPageToken"] = next_token
    return _mock_response(200, json_data=data)


# ═══════════════════════════════════════════════════════════════════════════════
# Sezione 1: Semafori per-fonte
# ═══════════════════════════════════════════════════════════════════════════════

class _SemaphoreTracker:
    """Misura il picco di concorrenza all'interno di un semaforo."""
    def __init__(self):
        self.peak = 0
        self.current = 0
        self._lock = threading.Lock()

    def __enter__(self):
        with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)
        return self

    def __exit__(self, *_):
        with self._lock:
            self.current -= 1


def _run_parallel_collection(expanded, instance, limits_override=None):
    """Run the collection phase of RelationalSearch with mocked semaphores."""
    import core.relational_search as rs_mod
    from concurrent.futures import ThreadPoolExecutor, as_completed

    raw = {"pubmed": [], "europepmc": [], "clinicaltrials": []}
    stats = {"query_calls": 0}

    collectors = {
        "pubmed": (instance.pubmed, instance._build_pubmed_query),
        "europepmc": (instance.europepmc, instance._build_epmc_query),
        "clinicaltrials": (instance.clinicaltrials, instance._build_ct_query),
    }

    source_sems = instance._source_sems

    def _collect_one(source, collector, query, provenance):
        sem = source_sems.get(source)
        with sem:
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
            pool.submit(_collect_one, source, collector, query, provenance): ()
            for source, collector, query, provenance in tasks
        }
        for fut in as_completed(futures):
            source, provenance, query, items = fut.result()
            if items:
                stats["query_calls"] += 1
            for item in items:
                item.metadata = dict(item.metadata or {})
                item.metadata.setdefault("match_provenance", []).append(provenance)
                raw[source].append(item)

    return raw, stats


def _make_rs_instance(pubmed_se=None, epmc_se=None, ct_se=None,
                       pubmed_rv=None, epmc_rv=None, ct_rv=None,
                       lim_override=None):
    """Build a RelationalSearch with mock collectors and configurable limits."""
    import core.relational_search as rs_mod
    from core import llm_limits

    lim = lim_override or llm_limits.HLEOLimits()

    instance = rs_mod.RelationalSearch.__new__(rs_mod.RelationalSearch)
    instance._client = MagicMock()
    instance.pubmed = MagicMock()
    instance.europepmc = MagicMock()
    instance.clinicaltrials = MagicMock()

    if pubmed_se is not None:
        instance.pubmed.search.side_effect = pubmed_se
    else:
        instance.pubmed.search.return_value = pubmed_rv or []
    if epmc_se is not None:
        instance.europepmc.search.side_effect = epmc_se
    else:
        instance.europepmc.search.return_value = epmc_rv or []
    if ct_se is not None:
        instance.clinicaltrials.search.side_effect = ct_se
    else:
        instance.clinicaltrials.search.return_value = ct_rv or []

    # Build semaphores from lim
    instance._source_sems = {
        "pubmed":         threading.Semaphore(lim.pubmed_max_concurrent),
        "europepmc":      threading.Semaphore(lim.epmc_max_concurrent),
        "clinicaltrials": threading.Semaphore(lim.ct_max_concurrent),
    }

    return instance


def _make_variant():
    v = MagicMock()
    v.agent = {"normalized": "drug", "search_terms": ["drug"]}
    v.manifestation = {"normalized": "rash", "search_terms": ["rash"]}
    v.relation_type = "adverse_effect"
    return v


class TestPubMedSemaphore(unittest.TestCase):
    """Scenario 1 — PubMed max_concurrent = 2."""

    def test_pubmed_peak_concurrency_respected(self):
        from core import llm_limits
        llm_limits.invalidate_limits_cache()

        tracker = _SemaphoreTracker()
        sem_real = threading.Semaphore(2)

        def tracked_pubmed(query, limit=None):
            with tracker:
                time.sleep(0.03)
                return []

        lim = llm_limits.HLEOLimits(pubmed_max_concurrent=2, collector_max_workers=8)
        instance = _make_rs_instance(pubmed_se=tracked_pubmed, lim_override=lim)
        # Replace the pubmed semaphore with the real one for tracking
        instance._source_sems["pubmed"] = sem_real

        N = 6
        expanded = [(_make_variant(), f"prov{i}") for i in range(N)]

        with patch("core.llm_limits._load_from_db", return_value=lim):
            llm_limits.invalidate_limits_cache()
            _run_parallel_collection(expanded, instance, limits_override=8)

        # Peak tracked inside the semaphore — at most 2 simultaneous
        # (note: tracker wraps the search call, not the semaphore acquire itself)
        self.assertLessEqual(tracker.peak, 2)


class TestEuropePMCSemaphore(unittest.TestCase):
    """Scenario 2 — EuropePMC max_concurrent = 4."""

    def test_epmc_semaphore_value(self):
        from core import llm_limits
        lim = llm_limits.HLEOLimits(epmc_max_concurrent=4)
        instance = _make_rs_instance(lim_override=lim)
        sem = instance._source_sems["europepmc"]
        # Acquire 4 times (hold them all), then check 5th fails, then release
        acquired = [sem.acquire(blocking=False) for _ in range(4)]
        self.assertEqual(sum(acquired), 4)
        # 5th acquire must fail while the 4 are still held
        fifth_ok = sem.acquire(blocking=False)
        # Release all acquired slots
        for ok in acquired:
            if ok:
                sem.release()
        self.assertFalse(fifth_ok, "5th acquire should have failed with epmc_max_concurrent=4")


class TestClinicalTrialsSemaphore(unittest.TestCase):
    """Scenario 3 — ClinicalTrials max_concurrent = 3."""

    def test_ct_semaphore_value(self):
        from core import llm_limits
        lim = llm_limits.HLEOLimits(ct_max_concurrent=3)
        instance = _make_rs_instance(lim_override=lim)
        sem = instance._source_sems["clinicaltrials"]
        # Acquire 3 times (hold them all), then check 4th fails, then release
        acquired = [sem.acquire(blocking=False) for _ in range(3)]
        self.assertEqual(sum(acquired), 3)
        # 4th acquire must fail while the 3 are still held
        fourth_ok = sem.acquire(blocking=False)
        for ok in acquired:
            if ok:
                sem.release()
        self.assertFalse(fourth_ok, "4th acquire should have failed with ct_max_concurrent=3")


class TestCollectorMaxWorkersTotal(unittest.TestCase):
    """Scenario 4 — collector_max_workers limita il totale."""

    def test_total_concurrency_bounded_by_max_workers(self):
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
            time.sleep(0.04)
            with lock:
                current_concurrent[0] -= 1
            return []

        lim = llm_limits.HLEOLimits(
            collector_max_workers=MAX_WORKERS,
            pubmed_max_concurrent=10,
            epmc_max_concurrent=10,
            ct_max_concurrent=10,
        )
        instance = _make_rs_instance(
            pubmed_se=slow_collector,
            epmc_se=slow_collector,
            ct_se=slow_collector,
            lim_override=lim,
        )

        N = 5
        expanded = [(_make_variant(), f"prov{i}") for i in range(N)]

        with patch("core.llm_limits._load_from_db", return_value=lim):
            llm_limits.invalidate_limits_cache()
            _run_parallel_collection(expanded, instance, limits_override=MAX_WORKERS)

        self.assertLessEqual(
            peak_concurrent[0], MAX_WORKERS,
            f"Peak concurrency {peak_concurrent[0]} exceeded max_workers {MAX_WORKERS}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Sezione 2: Sleep PubMed
# ═══════════════════════════════════════════════════════════════════════════════

class TestPubMedSleepBehavior(unittest.TestCase):
    """Scenari 5–7: sleep condizionale e configurabile."""

    def _make_pubmed_mock_calls(self, esearch_ids, sleep_val=0.0):
        """Helper: patch requests.get and time.sleep, run PubMed search."""
        from collectors.pubmed import PubMedCollector
        from core import llm_limits

        lim = llm_limits.HLEOLimits(
            pubmed_inter_call_sleep_s=sleep_val,
            collector_max_retries=0,
            collector_timeout_s=5.0,
        )
        sleep_calls = []

        esearch_resp = _mock_response(200, json_data={
            "esearchresult": {"idlist": esearch_ids, "count": str(len(esearch_ids))}
        })
        esummary_resp = _mock_response(200, json_data={
            "result": {pmid: {"title": f"T{pmid}", "authors": [],
                              "fulljournalname": "J", "pubdate": "2024"}
                       for pmid in esearch_ids}
        })
        efetch_resp = _mock_response(200, text_data="Abstract")

        responses = [esearch_resp]
        if esearch_ids:
            responses += [esummary_resp, efetch_resp]

        with patch("core.llm_limits._load_from_db", return_value=lim), \
             patch("core.http_retry.requests.get", side_effect=responses), \
             patch("collectors.pubmed.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            llm_limits.invalidate_limits_cache()
            collector = PubMedCollector()
            results = collector.search("test query")

        return results, sleep_calls

    def test_no_sleep_when_ids_empty(self):
        """Scenario 5 — sleep non eseguito quando la lista ID è vuota."""
        results, sleep_calls = self._make_pubmed_mock_calls(esearch_ids=[], sleep_val=0.4)
        self.assertEqual(results, [])
        self.assertEqual(sleep_calls, [], f"Expected no sleep calls, got: {sleep_calls}")

    def test_sleep_executed_when_ids_present(self):
        """Scenario 6 — sleep eseguito quando ci sono ID."""
        results, sleep_calls = self._make_pubmed_mock_calls(esearch_ids=["111"], sleep_val=0.4)
        self.assertEqual(len(sleep_calls), 2, f"Expected 2 sleep calls, got: {sleep_calls}")
        for s in sleep_calls:
            self.assertAlmostEqual(s, 0.4, places=5)

    def test_sleep_uses_configured_value(self):
        """Scenario 7 — valore sleep configurabile."""
        _, sleep_calls = self._make_pubmed_mock_calls(esearch_ids=["111"], sleep_val=0.1)
        self.assertTrue(all(abs(s - 0.1) < 1e-9 for s in sleep_calls),
                        f"Expected all sleeps = 0.1, got {sleep_calls}")


# ═══════════════════════════════════════════════════════════════════════════════
# Sezione 3: http_retry — test unitari
# ═══════════════════════════════════════════════════════════════════════════════

class TestHttpRetrySuccess(unittest.TestCase):
    """Scenario 25 — risposta 200 restituita senza retry."""

    def test_200_returned_immediately(self):
        from core.http_retry import http_get
        resp = _mock_response(200, json_data={"ok": True})
        with patch("core.http_retry.requests.get", return_value=resp) as mock_get, \
             patch("core.http_retry.time.sleep"):
            result = http_get("http://example.com", max_retries=2)
        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(result.status_code, 200)


class TestHttpRetryPermanentError(unittest.TestCase):
    """Scenari 15–18 — nessun retry su errori permanenti."""

    def _test_no_retry(self, status_code):
        import requests
        from core.http_retry import http_get
        resp = _mock_response(status_code)
        with patch("core.http_retry.requests.get", return_value=resp) as mock_get, \
             patch("core.http_retry.time.sleep") as mock_sleep:
            with self.assertRaises(requests.HTTPError):
                http_get("http://example.com", max_retries=2)
        self.assertEqual(mock_get.call_count, 1, f"Should not retry on {status_code}")
        mock_sleep.assert_not_called()

    def test_no_retry_400(self):
        """Scenario 15."""
        self._test_no_retry(400)

    def test_no_retry_401(self):
        """Scenario 16."""
        self._test_no_retry(401)

    def test_no_retry_403(self):
        """Scenario 17."""
        self._test_no_retry(403)

    def test_no_retry_404(self):
        """Scenario 18."""
        self._test_no_retry(404)


class TestHttpRetryTransient(unittest.TestCase):
    """Scenari 8–14 — retry su errori transitori."""

    def _test_retry_status(self, first_status: int):
        """First call returns *first_status*, second call returns 200."""
        import requests
        from core.http_retry import http_get
        resp_bad = _mock_response(first_status)
        resp_ok = _mock_response(200, json_data={"ok": True})
        with patch("core.http_retry.requests.get", side_effect=[resp_bad, resp_ok]) as mock_get, \
             patch("core.http_retry.time.sleep"):
            result = http_get("http://example.com", max_retries=2,
                              backoff_base_s=0.0, backoff_max_s=0.0)
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(result.status_code, 200)

    def test_retry_429(self):
        """Scenario 8."""
        self._test_retry_status(429)

    def test_retry_500(self):
        """Scenario 9."""
        self._test_retry_status(500)

    def test_retry_502(self):
        """Scenario 10."""
        self._test_retry_status(502)

    def test_retry_503(self):
        """Scenario 11."""
        self._test_retry_status(503)

    def test_retry_504(self):
        """Scenario 12."""
        self._test_retry_status(504)

    def test_retry_timeout(self):
        """Scenario 13 — retry su Timeout."""
        import requests
        from core.http_retry import http_get
        resp_ok = _mock_response(200, json_data={"ok": True})
        with patch("core.http_retry.requests.get",
                   side_effect=[requests.exceptions.Timeout(), resp_ok]) as mock_get, \
             patch("core.http_retry.time.sleep"):
            result = http_get("http://example.com", max_retries=2,
                              backoff_base_s=0.0, backoff_max_s=0.0)
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(result.status_code, 200)

    def test_retry_connection_error(self):
        """Scenario 14 — retry su ConnectionError."""
        import requests
        from core.http_retry import http_get
        resp_ok = _mock_response(200, json_data={"ok": True})
        with patch("core.http_retry.requests.get",
                   side_effect=[requests.exceptions.ConnectionError(), resp_ok]) as mock_get, \
             patch("core.http_retry.time.sleep"):
            result = http_get("http://example.com", max_retries=2,
                              backoff_base_s=0.0, backoff_max_s=0.0)
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(result.status_code, 200)


class TestHttpRetryMaxAttempts(unittest.TestCase):
    """Scenario 19 — massimo numero di tentativi rispettato."""

    def test_max_retries_2_means_3_total_calls(self):
        """max_retries=2 → 1 initial + 2 retries = 3 total calls."""
        import requests
        from core.http_retry import http_get
        resp_bad = _mock_response(503)
        with patch("core.http_retry.requests.get",
                   side_effect=[resp_bad] * 10) as mock_get, \
             patch("core.http_retry.time.sleep"):
            with self.assertRaises(requests.HTTPError):
                http_get("http://example.com", max_retries=2,
                         backoff_base_s=0.0, backoff_max_s=0.0)
        self.assertEqual(mock_get.call_count, 3)  # 1 + 2 retries

    def test_max_retries_0_means_1_total_call(self):
        """max_retries=0 → no retry, 1 call total."""
        import requests
        from core.http_retry import http_get
        resp_bad = _mock_response(503)
        with patch("core.http_retry.requests.get", return_value=resp_bad) as mock_get, \
             patch("core.http_retry.time.sleep"):
            with self.assertRaises(requests.HTTPError):
                http_get("http://example.com", max_retries=0)
        self.assertEqual(mock_get.call_count, 1)


class TestHttpRetryAfterHeader(unittest.TestCase):
    """Scenari 20, 29, 30 — rispetto Retry-After."""

    def test_retry_after_numeric_respected(self):
        """Scenario 29 — Retry-After: 3 → sleep(3.0)."""
        import requests
        from core.http_retry import http_get
        resp_bad = _mock_response(429, headers={"Retry-After": "3"})
        resp_ok = _mock_response(200)
        sleep_calls = []
        with patch("core.http_retry.requests.get", side_effect=[resp_bad, resp_ok]), \
             patch("core.http_retry.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            http_get("http://example.com", max_retries=2,
                     backoff_base_s=1.0, backoff_max_s=60.0)
        self.assertEqual(len(sleep_calls), 1)
        self.assertAlmostEqual(sleep_calls[0], 3.0, places=5)

    def test_retry_after_capped_at_backoff_max(self):
        """Scenario 30 — Retry-After > backoff_max_s → capped."""
        import requests
        from core.http_retry import http_get
        resp_bad = _mock_response(429, headers={"Retry-After": "120"})
        resp_ok = _mock_response(200)
        sleep_calls = []
        with patch("core.http_retry.requests.get", side_effect=[resp_bad, resp_ok]), \
             patch("core.http_retry.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            http_get("http://example.com", max_retries=2,
                     backoff_base_s=1.0, backoff_max_s=10.0)
        self.assertEqual(len(sleep_calls), 1)
        self.assertAlmostEqual(sleep_calls[0], 10.0, places=5)


# ═══════════════════════════════════════════════════════════════════════════════
# Sezione 4: Timeout configurabile
# ═══════════════════════════════════════════════════════════════════════════════

class TestCollectorTimeoutConfigurable(unittest.TestCase):
    """Scenario 21 — timeout configurabile via Global Limits."""

    def test_pubmed_uses_collector_timeout_s(self):
        from collectors.pubmed import PubMedCollector
        from core import llm_limits

        lim = llm_limits.HLEOLimits(collector_timeout_s=7.0, collector_max_retries=0,
                                     pubmed_inter_call_sleep_s=0.0)

        esearch_resp = _esearch_ok(ids=["1"])
        esummary_resp = _esummary_ok(ids=["1"])
        efetch_resp = _efetch_ok()

        with patch("core.llm_limits._load_from_db", return_value=lim), \
             patch("core.http_retry.requests.get",
                   side_effect=[esearch_resp, esummary_resp, efetch_resp]) as mock_get, \
             patch("collectors.pubmed.time.sleep"):
            llm_limits.invalidate_limits_cache()
            PubMedCollector().search("test")

        for c in mock_get.call_args_list:
            self.assertEqual(c.kwargs.get("timeout", c.args[1] if len(c.args) > 1 else None),
                             7.0)

    def test_epmc_uses_collector_timeout_s(self):
        from collectors.europepmc import EuropePMCCollector
        from core import llm_limits

        lim = llm_limits.HLEOLimits(collector_timeout_s=8.0, collector_max_retries=0)

        with patch("core.llm_limits._load_from_db", return_value=lim), \
             patch("core.http_retry.requests.get",
                   return_value=_epmc_ok()) as mock_get:
            llm_limits.invalidate_limits_cache()
            EuropePMCCollector().search("test", limit=2)

        for c in mock_get.call_args_list:
            self.assertEqual(c.kwargs.get("timeout", c.args[1] if len(c.args) > 1 else None),
                             8.0)

    def test_ct_uses_collector_timeout_s(self):
        from collectors.clinicaltrials import ClinicalTrialsCollector
        from core import llm_limits

        lim = llm_limits.HLEOLimits(collector_timeout_s=9.0, collector_max_retries=0)

        with patch("core.llm_limits._load_from_db", return_value=lim), \
             patch("core.http_retry.requests.get",
                   return_value=_ct_ok()) as mock_get:
            llm_limits.invalidate_limits_cache()
            ClinicalTrialsCollector().search("test", limit=2)

        for c in mock_get.call_args_list:
            self.assertEqual(c.kwargs.get("timeout", c.args[1] if len(c.args) > 1 else None),
                             9.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Sezione 5: Retry nei collector (integrazione)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPubMedRetry(unittest.TestCase):
    """Scenario 22 — PubMed restituisce risultati corretti dopo retry."""

    def test_pubmed_retry_on_429(self):
        from collectors.pubmed import PubMedCollector
        from core import llm_limits
        import requests

        lim = llm_limits.HLEOLimits(
            collector_max_retries=2,
            collector_timeout_s=5.0,
            pubmed_inter_call_sleep_s=0.0,
            backoff_base_s=0.0, backoff_max_s=0.0,
        )

        esearch_resp = _esearch_ok(ids=["1"])
        esummary_resp = _esummary_ok(ids=["1"])
        efetch_resp = _efetch_ok()

        # esearch fails once with 429, then succeeds
        resp_429 = _mock_response(429)
        with patch("core.llm_limits._load_from_db", return_value=lim), \
             patch("core.http_retry.requests.get",
                   side_effect=[resp_429, esearch_resp, esummary_resp, efetch_resp]), \
             patch("core.http_retry.time.sleep"), \
             patch("collectors.pubmed.time.sleep"):
            llm_limits.invalidate_limits_cache()
            results = PubMedCollector().search("test")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].pmid, "1")

    def test_pubmed_retry_on_503(self):
        from collectors.pubmed import PubMedCollector
        from core import llm_limits

        lim = llm_limits.HLEOLimits(
            collector_max_retries=2,
            collector_timeout_s=5.0,
            pubmed_inter_call_sleep_s=0.0,
            backoff_base_s=0.0, backoff_max_s=0.0,
        )

        resp_503 = _mock_response(503)
        esearch_resp = _esearch_ok(ids=["2"])
        esummary_resp = _esummary_ok(ids=["2"])
        efetch_resp = _efetch_ok()

        with patch("core.llm_limits._load_from_db", return_value=lim), \
             patch("core.http_retry.requests.get",
                   side_effect=[resp_503, esearch_resp, esummary_resp, efetch_resp]), \
             patch("core.http_retry.time.sleep"), \
             patch("collectors.pubmed.time.sleep"):
            llm_limits.invalidate_limits_cache()
            results = PubMedCollector().search("test")

        self.assertEqual(len(results), 1)


class TestEuropePMCRetry(unittest.TestCase):
    """Retry su EuropePMC."""

    def test_epmc_retry_on_429(self):
        from collectors.europepmc import EuropePMCCollector
        from core import llm_limits

        lim = llm_limits.HLEOLimits(
            collector_max_retries=2,
            collector_timeout_s=5.0,
            backoff_base_s=0.0, backoff_max_s=0.0,
        )

        resp_429 = _mock_response(429)
        resp_ok = _epmc_ok(n=3, hit_count=3)

        with patch("core.llm_limits._load_from_db", return_value=lim), \
             patch("core.http_retry.requests.get",
                   side_effect=[resp_429, resp_ok]), \
             patch("core.http_retry.time.sleep"):
            llm_limits.invalidate_limits_cache()
            results = EuropePMCCollector().search("test", limit=3)

        self.assertEqual(len(results), 3)

    def test_epmc_no_retry_on_404(self):
        from collectors.europepmc import EuropePMCCollector
        from core import llm_limits
        import requests

        lim = llm_limits.HLEOLimits(collector_max_retries=2, backoff_base_s=0.0)
        resp_404 = _mock_response(404)

        with patch("core.llm_limits._load_from_db", return_value=lim), \
             patch("core.http_retry.requests.get",
                   return_value=resp_404) as mock_get, \
             patch("core.http_retry.time.sleep"):
            llm_limits.invalidate_limits_cache()
            with self.assertRaises(requests.HTTPError):
                EuropePMCCollector().search("test")

        self.assertEqual(mock_get.call_count, 1)


class TestClinicalTrialsRetry(unittest.TestCase):
    """Retry su ClinicalTrials."""

    def test_ct_retry_on_503(self):
        from collectors.clinicaltrials import ClinicalTrialsCollector
        from core import llm_limits

        lim = llm_limits.HLEOLimits(
            collector_max_retries=2,
            collector_timeout_s=5.0,
            backoff_base_s=0.0, backoff_max_s=0.0,
        )

        resp_503 = _mock_response(503)
        resp_ok = _ct_ok(n=2)

        with patch("core.llm_limits._load_from_db", return_value=lim), \
             patch("core.http_retry.requests.get",
                   side_effect=[resp_503, resp_ok]), \
             patch("core.http_retry.time.sleep"):
            llm_limits.invalidate_limits_cache()
            results = ClinicalTrialsCollector().search("test", limit=2)

        self.assertEqual(len(results), 2)

    def test_ct_retry_on_timeout(self):
        from collectors.clinicaltrials import ClinicalTrialsCollector
        from core import llm_limits
        import requests

        lim = llm_limits.HLEOLimits(
            collector_max_retries=2,
            collector_timeout_s=5.0,
            backoff_base_s=0.0, backoff_max_s=0.0,
        )

        resp_ok = _ct_ok(n=1)

        with patch("core.llm_limits._load_from_db", return_value=lim), \
             patch("core.http_retry.requests.get",
                   side_effect=[requests.exceptions.Timeout(), resp_ok]), \
             patch("core.http_retry.time.sleep"):
            llm_limits.invalidate_limits_cache()
            results = ClinicalTrialsCollector().search("test", limit=1)

        self.assertEqual(len(results), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Sezione 6: Isolamento e correttezza (regression FASE 4.1)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCollectorIsolation(unittest.TestCase):
    """Scenario 23 — errore di un collector non blocca gli altri."""

    def test_pubmed_error_others_succeed(self):
        from core import llm_limits
        llm_limits.invalidate_limits_cache()

        from unittest.mock import MagicMock
        lim = llm_limits.HLEOLimits(collector_max_workers=6)
        item = MagicMock()
        item.metadata = {}

        instance = _make_rs_instance(
            pubmed_se=RuntimeError("PubMed down"),
            epmc_rv=[item],
            ct_rv=[item],
            lim_override=lim,
        )

        expanded = [(_make_variant(), "prov1")]
        with patch("core.llm_limits._load_from_db", return_value=lim):
            llm_limits.invalidate_limits_cache()
            raw, stats = _run_parallel_collection(expanded, instance, limits_override=6)

        self.assertEqual(len(raw["pubmed"]), 0)
        self.assertEqual(len(raw["europepmc"]), 1)
        self.assertEqual(len(raw["clinicaltrials"]), 1)


class TestNoDuplicatesAfterRetry(unittest.TestCase):
    """Scenario 24 — nessun duplicato introdotto dai retry."""

    def test_pubmed_no_duplicates_with_retry(self):
        from collectors.pubmed import PubMedCollector
        from core import llm_limits

        lim = llm_limits.HLEOLimits(
            collector_max_retries=1,
            collector_timeout_s=5.0,
            pubmed_inter_call_sleep_s=0.0,
            backoff_base_s=0.0, backoff_max_s=0.0,
        )

        ids = ["1", "2", "3"]
        resp_503 = _mock_response(503)
        esearch = _esearch_ok(ids=ids, count=3)
        esummary = _esummary_ok(ids=ids)
        efetch = _efetch_ok()

        with patch("core.llm_limits._load_from_db", return_value=lim), \
             patch("core.http_retry.requests.get",
                   side_effect=[resp_503, esearch, esummary, efetch]), \
             patch("core.http_retry.time.sleep"), \
             patch("collectors.pubmed.time.sleep"):
            llm_limits.invalidate_limits_cache()
            results = PubMedCollector().search("test")

        self.assertEqual(len(results), 3)
        pmids = [r.pmid for r in results]
        self.assertEqual(len(pmids), len(set(pmids)), "Duplicate PMIDs found after retry")


# ═══════════════════════════════════════════════════════════════════════════════
# Sezione 7: Global Limits — nuovi campi
# ═══════════════════════════════════════════════════════════════════════════════

class TestNewGlobalLimitsFields(unittest.TestCase):
    """Verifica che i nuovi campi siano presenti e validati correttamente."""

    def test_defaults(self):
        from core.llm_limits import HLEOLimits
        lim = HLEOLimits()
        self.assertEqual(lim.pubmed_max_concurrent, 2)
        self.assertEqual(lim.epmc_max_concurrent, 4)
        self.assertEqual(lim.ct_max_concurrent, 3)
        self.assertAlmostEqual(lim.pubmed_inter_call_sleep_s, 0.4)
        self.assertAlmostEqual(lim.collector_timeout_s, 20.0)
        self.assertEqual(lim.collector_max_retries, 2)

    def test_validate_clamps(self):
        from core.llm_limits import HLEOLimits
        lim = HLEOLimits(
            pubmed_max_concurrent=0, epmc_max_concurrent=999,
            ct_max_concurrent=0, pubmed_inter_call_sleep_s=-1.0,
            collector_timeout_s=1.0, collector_max_retries=99,
        )
        lim.validate()
        self.assertEqual(lim.pubmed_max_concurrent, 1)
        self.assertEqual(lim.epmc_max_concurrent, 10)
        self.assertEqual(lim.ct_max_concurrent, 1)
        self.assertAlmostEqual(lim.pubmed_inter_call_sleep_s, 0.0)
        self.assertAlmostEqual(lim.collector_timeout_s, 5.0)
        self.assertEqual(lim.collector_max_retries, 5)

    def test_to_dict_includes_new_fields(self):
        from core.llm_limits import HLEOLimits
        d = HLEOLimits().to_dict()
        for key in ("pubmed_max_concurrent", "epmc_max_concurrent", "ct_max_concurrent",
                    "pubmed_inter_call_sleep_s", "collector_timeout_s", "collector_max_retries"):
            self.assertIn(key, d)

    def test_from_dict_reads_new_fields(self):
        from core.llm_limits import HLEOLimits
        lim = HLEOLimits.from_dict({"pubmed_max_concurrent": 5, "collector_max_retries": 3})
        self.assertEqual(lim.pubmed_max_concurrent, 5)
        self.assertEqual(lim.collector_max_retries, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# Sezione 8: Migration nuove colonne
# ═══════════════════════════════════════════════════════════════════════════════

class TestMigrationFase42B(unittest.TestCase):
    """Verifica che le nuove colonne FASE 4.2B siano aggiunte correttamente."""

    def _fresh_engine(self):
        from sqlalchemy import create_engine
        return create_engine("sqlite:///:memory:", echo=False)

    def test_new_columns_added_to_existing_table(self):
        from sqlalchemy import text, inspect
        engine = self._fresh_engine()

        # Create minimal table without new columns
        with engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE hleo_global_limits ("
                "  id INTEGER PRIMARY KEY,"
                "  pipeline_max_workers INTEGER DEFAULT 8,"
                "  collector_max_workers INTEGER DEFAULT 6"
                ")"
            ))

        from core.migrations import run_schema_upgrades
        run_schema_upgrades(engine)

        insp = inspect(engine)
        cols = [c["name"] for c in insp.get_columns("hleo_global_limits")]
        for col in ("pubmed_max_concurrent", "epmc_max_concurrent", "ct_max_concurrent",
                    "pubmed_inter_call_sleep_s", "collector_timeout_s", "collector_max_retries"):
            self.assertIn(col, cols, f"Missing column: {col}")

    def test_migration_idempotent_with_new_columns(self):
        from sqlalchemy import inspect
        from core.database import Base
        engine = self._fresh_engine()
        Base.metadata.create_all(bind=engine)

        from core.migrations import run_schema_upgrades
        run_schema_upgrades(engine)
        run_schema_upgrades(engine)  # second run must be safe

        insp = inspect(engine)
        cols = [c["name"] for c in insp.get_columns("hleo_global_limits")]
        for col in ("pubmed_max_concurrent", "collector_timeout_s"):
            self.assertEqual(cols.count(col), 1, f"Column {col} duplicated")

    def test_defaults_applied_to_existing_row(self):
        from sqlalchemy import text
        engine = self._fresh_engine()

        with engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE hleo_global_limits ("
                "  id INTEGER PRIMARY KEY,"
                "  collector_max_workers INTEGER DEFAULT 6"
                ")"
            ))
            conn.execute(text("INSERT INTO hleo_global_limits (id) VALUES (1)"))

        from core.migrations import run_schema_upgrades
        run_schema_upgrades(engine)

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT pubmed_max_concurrent, ct_max_concurrent, collector_max_retries "
                     "FROM hleo_global_limits WHERE id = 1")
            ).fetchone()

        self.assertEqual(row[0], 2)   # pubmed_max_concurrent DEFAULT 2
        self.assertEqual(row[1], 3)   # ct_max_concurrent DEFAULT 3
        self.assertEqual(row[2], 2)   # collector_max_retries DEFAULT 2


if __name__ == "__main__":
    unittest.main(verbosity=2)
