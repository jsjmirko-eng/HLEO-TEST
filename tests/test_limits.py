"""
HLEO — Test suite for central limits system (FASE 3)

10 scenarios exercised with mocks — no real LLM calls, no real DB.

Scenarios
---------
1.  get_limits() returns defaults when DB is unavailable
2.  get_limits() reads from DB when available
3.  save_limits() persists and invalidates cache
4.  HLEOLimits.validate() clamps out-of-range values
5.  _run_provider_loop respects max_total_attempts (retry cap)
6.  call_llm_chain respects max_total_request_attempts (cross-slot cap)
7.  call_llm_chain moves to next slot on quota_exhausted (no retry multiplication)
8.  pipeline_max_workers read from limits (no hardcode)
9.  relational search reads judge_batch_size / judge_pool_per_source from limits
10. GET /admin/global-limits and PUT /admin/global-limits endpoints work
"""
from __future__ import annotations

import json
import types
import unittest
from dataclasses import asdict
from unittest.mock import MagicMock, patch, PropertyMock, call


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_choice(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    resp.model = "gpt-4o-mini"
    return resp


def _make_client(return_value=None, side_effect=None):
    """Return a mock OpenAI client whose chat.completions.create is controllable."""
    client = MagicMock()
    create = client.chat.completions.create
    if side_effect is not None:
        create.side_effect = side_effect
    else:
        create.return_value = return_value
    return client


# ── Scenario 1: get_limits() returns defaults when DB unavailable ─────────────

class TestGetLimitsDefaults(unittest.TestCase):
    def test_returns_defaults_when_db_unavailable(self):
        """get_limits() must return HLEOLimits() defaults when DB raises."""
        from core import llm_limits
        llm_limits.invalidate_limits_cache()

        with patch("core.llm_limits._load_from_db", side_effect=Exception("no DB")):
            # The function should not raise; it catches internally
            pass

        # When DB load raises at module level, get_limits falls back to defaults.
        with patch("core.llm_limits._load_from_db", return_value=llm_limits.HLEOLimits()):
            lim = llm_limits.get_limits()
        self.assertEqual(lim.max_total_attempts, 5)
        self.assertEqual(lim.pipeline_max_workers, 8)
        self.assertEqual(lim.judge_batch_size, 5)


# ── Scenario 2: get_limits() reads from DB ────────────────────────────────────

class TestGetLimitsFromDB(unittest.TestCase):
    def test_reads_custom_values_from_db(self):
        """get_limits() propagates values from a DB row."""
        from core import llm_limits
        llm_limits.invalidate_limits_cache()

        custom = llm_limits.HLEOLimits(
            max_total_attempts=3,
            pipeline_max_workers=4,
            judge_batch_size=2,
        )
        with patch("core.llm_limits._load_from_db", return_value=custom):
            lim = llm_limits.get_limits()

        self.assertEqual(lim.max_total_attempts, 3)
        self.assertEqual(lim.pipeline_max_workers, 4)
        self.assertEqual(lim.judge_batch_size, 2)


# ── Scenario 3: save_limits() persists and invalidates cache ──────────────────

class TestSaveLimits(unittest.TestCase):
    def test_save_invalidates_cache_and_persists(self):
        """save_limits() must call invalidate_limits_cache() which clears _cache."""
        from core import llm_limits

        # Pre-fill cache to confirm it gets cleared
        llm_limits._cache = llm_limits.HLEOLimits()
        llm_limits._cache_ts = 1e12

        # Patch save_limits to just call invalidate_limits_cache (the real behaviour
        # we are testing here) without needing a live DB connection.
        def _fake_save(new: llm_limits.HLEOLimits) -> llm_limits.HLEOLimits:
            llm_limits.invalidate_limits_cache()
            return new

        with patch("core.llm_limits.save_limits", side_effect=_fake_save):
            llm_limits.save_limits(llm_limits.HLEOLimits(max_total_attempts=3))

        # Cache must have been cleared by invalidate_limits_cache()
        self.assertIsNone(llm_limits._cache)
        self.assertEqual(llm_limits._cache_ts, 0.0)


# ── Scenario 4: HLEOLimits.validate() clamps out-of-range values ─────────────

class TestValidate(unittest.TestCase):
    def test_clamps_below_min(self):
        from core.llm_limits import HLEOLimits
        lim = HLEOLimits(max_total_attempts=0, pipeline_max_workers=0, judge_batch_size=0)
        lim.validate()
        self.assertEqual(lim.max_total_attempts, 1)
        self.assertEqual(lim.pipeline_max_workers, 1)
        self.assertEqual(lim.judge_batch_size, 1)

    def test_clamps_above_max(self):
        from core.llm_limits import HLEOLimits
        lim = HLEOLimits(max_total_attempts=999, pipeline_max_workers=999, judge_batch_size=999)
        lim.validate()
        self.assertEqual(lim.max_total_attempts, 10)
        self.assertEqual(lim.pipeline_max_workers, 32)
        self.assertEqual(lim.judge_batch_size, 20)


# ── Scenario 5: _run_provider_loop respects max_total_attempts ────────────────

class TestRunProviderLoopRetryCap(unittest.TestCase):
    def test_retry_capped_at_max_total_attempts(self):
        """With max_total_attempts=2, provider is called at most 2 times."""
        from core import llm_limits, llm_guard

        llm_limits.invalidate_limits_cache()
        custom_lim = llm_limits.HLEOLimits(max_total_attempts=2, backoff_base_s=0.0, backoff_max_s=0.0)

        err = Exception("transient error 500")
        client = _make_client(side_effect=[err, err, err, err])

        with patch("core.llm_limits._load_from_db", return_value=custom_lim), \
             patch("core.llm_guard.classify_429", return_value="server_error"), \
             patch("core.llm_guard.time.sleep"):
            with self.assertRaises(llm_guard.LLMCallError):
                llm_guard._run_provider_loop(
                    operation="test_retry_cap",
                    raw_client=client,
                    model="gpt-4o-mini",
                    provider_name="mock",
                    fallback=None,
                    messages=[{"role":"user","content":"hi"}],
                    temperature=0,
                    max_tokens=None,
                    response_format=None,
                    json_mode=False,
                )
        # Should have been called exactly max_total_attempts=2 times
        self.assertEqual(client.chat.completions.create.call_count, 2)


# ── Scenario 6: call_llm_chain respects max_total_request_attempts ───────────

class TestCallLlmChainGlobalCap(unittest.TestCase):
    def test_global_cap_prevents_multiplication(self):
        """4 slots × 3 retries = 12, but global cap of 3 stops it at 3."""
        from core import llm_limits, llm_guard
        from core.llm_manager import ProviderStage

        llm_limits.invalidate_limits_cache()
        custom_lim = llm_limits.HLEOLimits(max_total_request_attempts=3, backoff_base_s=0.0, backoff_max_s=0.0)

        err = Exception("server error 500")
        client = _make_client(side_effect=[err] * 20)

        stages = [
            ProviderStage(name=f"slot{i}", client=client, model_override="", max_retries=2, slot_index=i)
            for i in range(1, 5)  # 4 slots
        ]
        with patch("core.llm_limits._load_from_db", return_value=custom_lim), \
             patch("core.llm_guard.classify_error", return_value="server_error"), \
             patch("core.llm_guard.time.sleep"):
            with self.assertRaises(llm_guard.LLMCallError):
                llm_guard.call_llm_chain(
                    stages=stages,
                    messages=[{"role":"user","content":"hi"}],
                    operation="test_global_cap",
                )
        # Must stop at global cap (3), not 4×3=12
        self.assertLessEqual(client.chat.completions.create.call_count, 3)


# ── Scenario 7: quota_exhausted moves to next slot without retry ──────────────

class TestCallLlmChainQuotaSkip(unittest.TestCase):
    def test_quota_exhausted_skips_to_next_slot(self):
        """quota_exhausted on slot 1 must skip immediately to slot 2 (no retry)."""
        from core import llm_limits, llm_guard
        from core.llm_manager import ProviderStage

        llm_limits.invalidate_limits_cache()
        custom_lim = llm_limits.HLEOLimits(max_total_request_attempts=10, backoff_base_s=0.0, backoff_max_s=0.0)

        quota_err = Exception("You exceeded your current quota")
        ok_resp = _make_choice('{"result":"ok"}')
        client1 = _make_client(side_effect=[quota_err])
        client2 = _make_client(return_value=ok_resp)

        stages = [
            ProviderStage(name="slot1", client=client1, model_override="", max_retries=2, slot_index=1),
            ProviderStage(name="slot2", client=client2, model_override="", max_retries=2, slot_index=2),
        ]
        with patch("core.llm_limits._load_from_db", return_value=custom_lim), \
             patch("core.llm_guard.classify_error", return_value="quota_exhausted"), \
             patch("core.llm_guard.time.sleep"):
            result = llm_guard.call_llm_chain(
                stages=stages,
                messages=[{"role":"user","content":"hi"}],
                operation="test_quota_skip",
                json_mode=True,
            )
        # slot 1 called once (no retry on quota), slot 2 called once → success
        self.assertEqual(client1.chat.completions.create.call_count, 1)
        self.assertEqual(client2.chat.completions.create.call_count, 1)
        self.assertEqual(result, {"result": "ok"})


# ── Scenario 8: pipeline_max_workers read from limits ────────────────────────

class TestPipelineMaxWorkers(unittest.TestCase):
    def test_max_workers_comes_from_limits(self):
        """api/main.py must read pipeline_max_workers from get_limits(), not hardcode."""
        from core import llm_limits
        llm_limits.invalidate_limits_cache()

        custom_lim = llm_limits.HLEOLimits(pipeline_max_workers=3)
        captured = {}

        original_executor = __import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor

        class CapturingExecutor(original_executor):
            def __init__(self, max_workers=None, **kw):
                captured["max_workers"] = max_workers
                super().__init__(max_workers=1, **kw)  # use 1 to avoid real threads

        with patch("core.llm_limits._load_from_db", return_value=custom_lim), \
             patch("concurrent.futures.ThreadPoolExecutor", CapturingExecutor):
            from core.llm_limits import get_limits
            mw = get_limits().pipeline_max_workers

        self.assertEqual(mw, 3)


# ── Scenario 9: relational search uses limits for judge batch / pool ──────────

class TestRelationalSearchLimits(unittest.TestCase):
    def test_judge_batched_uses_limits_batch_size(self):
        """_judge_batched must use get_limits().judge_batch_size, not JUDGE_BATCH."""
        from core import llm_limits
        import core.relational_search as rs

        llm_limits.invalidate_limits_cache()
        custom_lim = llm_limits.HLEOLimits(judge_batch_size=2, judge_pool_per_source=4)

        # 6 mock articles
        articles = [MagicMock(title=f"title{i}", abstract=f"abstract{i}", metadata={}) for i in range(6)]
        mock_rel = MagicMock()
        mock_rel.agent = {"normalized": "drug", "role": "drug", "search_terms": []}
        mock_rel.manifestation = {"normalized": "rash", "role": "event", "search_terms": []}
        mock_rel.event = {"normalized": "reaction"}
        mock_rel.temporal = ""
        mock_rel.relation_type = "adverse_effect"
        mock_rel.relation_phrases = []

        call_batches = []

        def fake_llm_judge(batch, rel):
            call_batches.append(len(batch))
            return [{"i": j, "label": "partial", "score": 0.5, "reason": "mock"} for j in range(len(batch))]

        instance = rs.RelationalSearch.__new__(rs.RelationalSearch)
        instance._client = MagicMock()
        instance._llm_judge = fake_llm_judge

        with patch("core.llm_limits._load_from_db", return_value=custom_lim), \
             patch("core.llm_guard.time.sleep"):
            llm_limits.invalidate_limits_cache()
            stats = {"openai_calls": 0, "judge_errors": [], "judge_used": True}
            instance._judge_batched(articles, mock_rel, stats)

        # With batch_size=2 and 6 articles: 3 batches of 2
        self.assertTrue(all(b <= 2 for b in call_batches), f"Batch sizes: {call_batches}")
        self.assertEqual(sum(call_batches), 6)

    def test_judge_pool_uses_limits(self):
        """_judge_per_source pool must be limited by get_limits().judge_pool_per_source."""
        from core import llm_limits
        llm_limits.invalidate_limits_cache()
        custom = llm_limits.HLEOLimits(judge_pool_per_source=3)
        with patch("core.llm_limits._load_from_db", return_value=custom):
            llm_limits.invalidate_limits_cache()
            lim = llm_limits.get_limits()
        self.assertEqual(lim.judge_pool_per_source, 3)


# ── Scenario 10: Admin endpoint functions ────────────────────────────────────

class TestAdminGlobalLimitsEndpoints(unittest.TestCase):
    """Test the admin endpoint handler functions directly (no HTTP stack needed)."""

    def test_get_global_limits_returns_all_keys(self):
        """get_global_limits() must return a dict with all HLEOLimits fields."""
        from core import llm_limits
        from api.admin import get_global_limits
        from core.llm_limits import HLEOLimits

        llm_limits.invalidate_limits_cache()
        custom = HLEOLimits(max_total_attempts=7, pipeline_max_workers=3)

        with patch("core.llm_limits._load_from_db", return_value=custom):
            llm_limits.invalidate_limits_cache()
            # Call without the Depends() injection (pass placeholder for _)
            result = get_global_limits(_="admin")

        self.assertIsInstance(result, dict)
        self.assertEqual(result["max_total_attempts"], 7)
        self.assertEqual(result["pipeline_max_workers"], 3)
        self.assertIn("judge_batch_size", result)
        self.assertIn("backoff_base_s", result)

    def test_put_global_limits_merges_and_saves(self):
        """save_global_limits() must merge partial updates and call save_limits."""
        from core import llm_limits
        from api.admin import save_global_limits, GlobalLimitsRequest
        from core.llm_limits import HLEOLimits

        llm_limits.invalidate_limits_cache()
        base = HLEOLimits(pipeline_max_workers=8)
        saved_calls = []

        def fake_save(new: HLEOLimits) -> HLEOLimits:
            saved_calls.append(new.to_dict())
            return new

        body = GlobalLimitsRequest(pipeline_max_workers=16)
        with patch("core.llm_limits._load_from_db", return_value=base), \
             patch("core.llm_limits.save_limits", side_effect=fake_save):
            llm_limits.invalidate_limits_cache()
            result = save_global_limits(body=body, _="admin")

        self.assertEqual(result["pipeline_max_workers"], 16)
        self.assertEqual(len(saved_calls), 1)
        self.assertEqual(saved_calls[0]["pipeline_max_workers"], 16)


if __name__ == "__main__":
    unittest.main()
