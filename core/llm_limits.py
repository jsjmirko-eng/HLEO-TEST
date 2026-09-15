"""
HLEO — Central LLM & pipeline limits
======================================

Single source of truth for ALL tuneable limits in HLEO.

Every module that performs LLM calls, concurrent extraction, or relational
search MUST read its caps from get_limits() rather than hardcoding values.

Design rules
------------
- Limits are loaded from DB (hleo_global_limits, single row) at runtime.
- If the DB row or table is absent, the dataclass defaults are used.
- get_limits() is called at runtime, not at module import time, so Admin UI
  changes take effect on the next request without a server restart.
- A lightweight in-process cache (5-second TTL) prevents per-call DB queries
  while still picking up Admin changes quickly.
- External callers that imported the old module-level constants from
  llm_guard.py continue to see the default values (backward compat).

Limits exposed
--------------
LLM retry
  max_total_attempts            Per-provider retry cap (legacy single-provider path)
  max_total_request_attempts    Cross-slot cap for multi-slot chain (prevents
                                retry × slot multiplication)
  default_max_retries_per_stage Per-slot default retry budget when not set per-slot

Backoff
  backoff_base_s / backoff_max_s / backoff_jitter

Concurrency
  pipeline_max_workers          ThreadPoolExecutor size for /pipeline/run

Relational search
  judge_batch_size              Articles per judge LLM call
  judge_pool_per_source         Top-N candidates judged per source

Timeout
  slot_timeout_s                Default per-slot HTTP timeout (used when slot
                                has no individual timeout configured)

Results
  max_pipeline_results          Upper bound on articles sent to LLM extraction
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, fields, asdict
from typing import Optional

logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────────

@dataclass
class HLEOLimits:
    """All configurable limits for HLEO. Fields mirror hleo_global_limits columns."""

    # LLM retry — legacy single-provider path (call_llm / call_llm_json)
    max_total_attempts: int = 5           # 1 initial + 4 retries max

    # LLM retry — multi-slot chain (call_llm_chain); caps retry × slot product
    max_total_request_attempts: int = 10

    # Per-slot default retry budget (0 = no retry, max 4)
    default_max_retries_per_stage: int = 2

    # Backoff parameters for all retry loops
    backoff_base_s: float = 2.0           # initial sleep before first retry
    backoff_max_s: float = 30.0           # cap on a single sleep
    backoff_jitter: float = 0.15          # ±jitter fraction

    # Concurrency — ThreadPoolExecutor for /pipeline/run LLM extraction
    pipeline_max_workers: int = 8

    # Concurrency — bounded ThreadPoolExecutor for scientific collector loop
    # (variant × source tasks run in parallel up to this limit)
    collector_max_workers: int = 6

    # Per-source concurrency caps (semaphores inside the collector pool)
    # These constrain concurrency within collector_max_workers, not beyond it.
    pubmed_max_concurrent: int = 2     # NCBI rate limit: 3 req/s without API key
    epmc_max_concurrent: int = 4       # EuropePMC: no strict published limit
    ct_max_concurrent: int = 3         # ClinicalTrials: ~3–5 req/s documented

    # PubMed inter-call sleep (seconds) — respects NCBI E-utilities rate limit.
    # Applied only when IDs are found (not on empty results).
    pubmed_inter_call_sleep_s: float = 0.4

    # HTTP timeout for collector calls (seconds)
    collector_timeout_s: float = 20.0

    # Max additional HTTP retry attempts per single request (0 = no retry).
    # Applies to transient errors: 429, 5xx, Timeout, ConnectionError.
    collector_max_retries: int = 2

    # Relational search — LLM judge budget
    judge_batch_size: int = 5             # articles per judge call
    judge_pool_per_source: int = 10       # top-N judged per source (× 3 sources ≈ 6 calls)

    # Default per-slot HTTP timeout when not individually configured
    slot_timeout_s: float = 60.0

    # Max articles forwarded to LLM extraction per pipeline run
    max_pipeline_results: int = 50

    def validate(self) -> "HLEOLimits":
        """Clamp all values to safe ranges and return self."""
        self.max_total_attempts = max(1, min(10, self.max_total_attempts))
        self.max_total_request_attempts = max(1, min(40, self.max_total_request_attempts))
        self.default_max_retries_per_stage = max(0, min(4, self.default_max_retries_per_stage))
        self.backoff_base_s = max(0.1, min(60.0, self.backoff_base_s))
        self.backoff_max_s = max(self.backoff_base_s, min(300.0, self.backoff_max_s))
        self.backoff_jitter = max(0.0, min(0.5, self.backoff_jitter))
        self.pipeline_max_workers = max(1, min(32, self.pipeline_max_workers))
        self.collector_max_workers = max(1, min(32, self.collector_max_workers))
        self.pubmed_max_concurrent = max(1, min(10, self.pubmed_max_concurrent))
        self.epmc_max_concurrent = max(1, min(10, self.epmc_max_concurrent))
        self.ct_max_concurrent = max(1, min(10, self.ct_max_concurrent))
        self.pubmed_inter_call_sleep_s = max(0.0, min(5.0, self.pubmed_inter_call_sleep_s))
        self.collector_timeout_s = max(5.0, min(120.0, self.collector_timeout_s))
        self.collector_max_retries = max(0, min(5, self.collector_max_retries))
        self.judge_batch_size = max(1, min(20, self.judge_batch_size))
        self.judge_pool_per_source = max(1, min(50, self.judge_pool_per_source))
        self.slot_timeout_s = max(5.0, min(600.0, self.slot_timeout_s))
        self.max_pipeline_results = max(1, min(500, self.max_pipeline_results))
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HLEOLimits":
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields and v is not None}
        return cls(**filtered)


# ── In-process cache (5-second TTL) ──────────────────────────────────────────
_cache: Optional[HLEOLimits] = None
_cache_ts: float = 0.0
_CACHE_TTL = 5.0   # seconds; short enough to pick up Admin changes quickly


def invalidate_limits_cache() -> None:
    """Force the next get_limits() call to re-query the DB."""
    global _cache, _cache_ts
    _cache = None
    _cache_ts = 0.0


def get_limits() -> HLEOLimits:
    """Return the current global limits, loading from DB with a 5-second TTL cache.

    Falls back to defaults when the DB is unavailable or the table does not
    exist yet (e.g. fresh install before the first migration).
    """
    global _cache, _cache_ts
    now = time.monotonic()
    if _cache is not None and (now - _cache_ts) < _CACHE_TTL:
        return _cache

    lim = _load_from_db()
    _cache = lim
    _cache_ts = now
    return lim


def _load_from_db() -> HLEOLimits:
    """Query the DB for the global limits row. Returns defaults on any failure."""
    try:
        from core.database import SessionLocal
        from core.models import HLEOGlobalLimits
        from sqlalchemy import select

        db = SessionLocal()
        try:
            row = db.execute(select(HLEOGlobalLimits)).scalar_one_or_none()
            if row is None:
                return HLEOLimits()
            # Build dataclass from the DB row, skipping non-limit columns.
            field_names = {f.name for f in fields(HLEOLimits)}
            kwargs = {}
            for name in field_names:
                val = getattr(row, name, None)
                if val is not None:
                    kwargs[name] = val
            return HLEOLimits(**kwargs).validate()
        finally:
            db.close()
    except Exception as exc:
        logger.debug("HLEOLimits: DB unavailable, using defaults — %s", exc)
        return HLEOLimits()


def save_limits(new: HLEOLimits) -> HLEOLimits:
    """Persist limits to the DB (upsert). Invalidates cache. Returns saved limits."""
    from core.database import SessionLocal
    from core.models import HLEOGlobalLimits
    from sqlalchemy import select
    from datetime import datetime, timezone

    new.validate()
    db = SessionLocal()
    try:
        row = db.execute(select(HLEOGlobalLimits)).scalar_one_or_none()
        if row is None:
            row = HLEOGlobalLimits()
            db.add(row)
        for f in fields(new):
            setattr(row, f.name, getattr(new, f.name))
        row.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(row)
    finally:
        db.close()
    invalidate_limits_cache()
    return new
