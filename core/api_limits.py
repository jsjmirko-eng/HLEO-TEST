"""Small process-local protections for expensive API endpoints."""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque

from fastapi import HTTPException, Request
from sqlalchemy import text

from core.database import engine


_DEFAULT_WINDOW_SECONDS = 60.0
_DEFAULT_COSTLY_LIMIT = 60
_MAX_TRACKED_CLIENTS = 10_000


@dataclass
class _Bucket:
    timestamps: Deque[float]


class InMemoryRateLimiter:
    """Thread-safe fixed-window limiter suitable for one API process."""

    def __init__(self, max_clients: int = _MAX_TRACKED_CLIENTS) -> None:
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()
        self._max_clients = max(1, max_clients)

    def allow(self, key: str, limit: int, window_seconds: float) -> tuple[bool, int, float]:
        now = time.monotonic()
        limit = max(1, limit)
        window_seconds = max(1.0, window_seconds)
        cutoff = now - window_seconds
        with self._lock:
            bucket = self._buckets.setdefault(key, _Bucket(deque()))
            while bucket.timestamps and bucket.timestamps[0] <= cutoff:
                bucket.timestamps.popleft()
            if len(bucket.timestamps) >= limit:
                retry_after = max(0.0, bucket.timestamps[0] + window_seconds - now)
                return False, 0, retry_after
            bucket.timestamps.append(now)
            if len(self._buckets) > self._max_clients:
                oldest = min(
                    self._buckets,
                    key=lambda item: (
                        self._buckets[item].timestamps[-1]
                        if self._buckets[item].timestamps
                        else float("-inf")
                    ),
                )
                self._buckets.pop(oldest, None)
            return True, max(0, limit - len(bucket.timestamps)), window_seconds

    def clear(self) -> None:
        with self._lock:
            self._buckets.clear()


class DatabaseRateLimiter:
    """Fixed-window limiter shared by API workers through the application DB."""

    _TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS hleo_rate_limits (
            bucket_key VARCHAR(512) PRIMARY KEY,
            window_started DOUBLE PRECISION NOT NULL,
            hits INTEGER NOT NULL
        )
    """

    def __init__(self) -> None:
        self._fallback = InMemoryRateLimiter()
        self._ready = False
        self._lock = threading.Lock()

    def _ensure_table(self, connection) -> None:
        if not self._ready:
            connection.execute(text(self._TABLE_SQL))
            self._ready = True

    def allow(self, key: str, limit: int, window_seconds: float) -> tuple[bool, int, float]:
        limit = max(1, limit)
        window_seconds = max(1.0, window_seconds)
        now = time.time()
        try:
            with self._lock:
                with engine.begin() as connection:
                    self._ensure_table(connection)
                    dialect = connection.dialect.name
                    if dialect == "postgresql":
                        connection.execute(
                            text(
                                "INSERT INTO hleo_rate_limits "
                                "(bucket_key, window_started, hits) VALUES (:key, :started, 0) "
                                "ON CONFLICT (bucket_key) DO NOTHING"
                            ),
                            {"key": key, "started": now},
                        )
                        select_sql = (
                            "SELECT window_started, hits FROM hleo_rate_limits "
                            "WHERE bucket_key = :key FOR UPDATE"
                        )
                    elif dialect == "sqlite":
                        connection.execute(
                            text(
                                "INSERT OR IGNORE INTO hleo_rate_limits "
                                "(bucket_key, window_started, hits) VALUES (:key, :started, 0)"
                            ),
                            {"key": key, "started": now},
                        )
                        select_sql = (
                            "SELECT window_started, hits FROM hleo_rate_limits "
                            "WHERE bucket_key = :key"
                        )
                    else:
                        select_sql = (
                            "SELECT window_started, hits FROM hleo_rate_limits "
                            "WHERE bucket_key = :key"
                        )
                    row = connection.execute(text(select_sql), {"key": key}).first()
                    if row is None:
                        raise RuntimeError("rate-limit bucket could not be created")

                    started, hits = float(row[0]), int(row[1])
                    if now - started >= window_seconds:
                        connection.execute(
                            text(
                                "UPDATE hleo_rate_limits SET window_started = :started, hits = 1 "
                                "WHERE bucket_key = :key"
                            ),
                            {"key": key, "started": now},
                        )
                        return True, limit - 1, window_seconds
                    if hits >= limit:
                        return False, 0, max(0.0, started + window_seconds - now)

                    connection.execute(
                        text(
                            "UPDATE hleo_rate_limits SET hits = hits + 1 "
                            "WHERE bucket_key = :key"
                        ),
                        {"key": key},
                    )
                    return True, max(0, limit - hits - 1), window_seconds
        except Exception:
            # A degraded limiter must not make the API unavailable when the DB
            # itself is down; the local fallback still protects this process.
            return self._fallback.allow(key, limit, window_seconds)

    def clear(self) -> None:
        self._fallback.clear()
        try:
            with self._lock:
                with engine.begin() as connection:
                    self._ensure_table(connection)
                    connection.execute(text("DELETE FROM hleo_rate_limits"))
        except Exception:
            pass


rate_limiter = (
    InMemoryRateLimiter()
    if os.getenv("HLEO_RATE_LIMIT_BACKEND", "database").strip().lower() == "memory"
    else DatabaseRateLimiter()
)


def _float_env(name: str, default: float) -> float:
    try:
        return max(1.0, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def cors_origins_from_env() -> list[str]:
    """Return explicit CORS origins, never treating ``*`` as a valid origin."""
    raw = os.getenv("HLEO_CORS_ORIGINS", "")
    origins = [origin.strip().rstrip("/") for origin in raw.split(",")]
    return [origin for origin in origins if origin and origin != "*"]


def _client_key(request: Request) -> str:
    if os.getenv("HLEO_TRUST_PROXY_HEADERS", "0").strip().lower() in {"1", "true", "yes"}:
        forwarded = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
        if forwarded:
            return forwarded
    return request.client.host if request.client else "unknown"


def _limit_for_path(path: str) -> int:
    if path == "/pipeline/run":
        return _int_env("HLEO_API_RATE_LIMIT_PIPELINE", 20)
    if path in {"/search", "/rwe/search"}:
        return _int_env("HLEO_API_RATE_LIMIT_SEARCH", _DEFAULT_COSTLY_LIMIT)
    if path.startswith("/rwe/extract"):
        return _int_env("HLEO_API_RATE_LIMIT_RWE_EXTRACT", 30)
    if path.startswith("/assistant/") or path.startswith("/synthesis"):
        return _int_env("HLEO_API_RATE_LIMIT_LLM", 30)
    return _int_env("HLEO_API_RATE_LIMIT", _DEFAULT_COSTLY_LIMIT)


def costly_request_guard(request: Request) -> None:
    """Rate-limit a costly endpoint and raise an HTTP 429 when exceeded."""
    window = _float_env("HLEO_API_RATE_WINDOW_S", _DEFAULT_WINDOW_SECONDS)
    allowed, remaining, retry_after = rate_limiter.allow(
        f"{_client_key(request)}:{request.url.path}",
        _limit_for_path(request.url.path),
        window,
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded; retry later.",
            headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
        )
    request.state.rate_limit_remaining = remaining
