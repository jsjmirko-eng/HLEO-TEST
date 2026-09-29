from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from core.api_limits import (
    DatabaseRateLimiter,
    InMemoryRateLimiter,
    costly_request_guard,
    cors_origins_from_env,
    rate_limiter,
)


def test_cors_origins_are_explicit_and_wildcard_is_ignored(monkeypatch):
    monkeypatch.setenv(
        "HLEO_CORS_ORIGINS",
        " https://app.example.com/ , *, http://localhost:3000 ",
    )

    assert cors_origins_from_env() == [
        "https://app.example.com",
        "http://localhost:3000",
    ]


def test_empty_cors_configuration_is_same_origin_only(monkeypatch):
    monkeypatch.delenv("HLEO_CORS_ORIGINS", raising=False)

    assert cors_origins_from_env() == []


def test_rate_limiter_enforces_limit_across_threads():
    limiter = InMemoryRateLimiter()

    def attempt():
        return limiter.allow("client:/search", limit=3, window_seconds=60)[0]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: attempt(), range(8)))

    assert sum(results) == 3




def test_database_rate_limiter_shares_bucket_state():
    first = DatabaseRateLimiter()
    second = DatabaseRateLimiter()
    first.clear()

    assert first.allow("shared-client:/search", limit=1, window_seconds=60)[0]
    assert not second.allow("shared-client:/search", limit=1, window_seconds=60)[0]

    first.clear()

def test_costly_request_guard_returns_http_429(monkeypatch):
    monkeypatch.setenv("HLEO_API_RATE_LIMIT_SEARCH", "1")
    monkeypatch.setenv("HLEO_API_RATE_WINDOW_S", "60")
    rate_limiter.clear()
    request = Request({
        "type": "http",
        "method": "GET",
        "path": "/search",
        "headers": [],
        "client": ("127.0.0.1", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    })

    costly_request_guard(request)
    with pytest.raises(HTTPException) as raised:
        costly_request_guard(request)

    assert raised.value.status_code == 429
    assert "Retry-After" in raised.value.headers
