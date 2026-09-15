"""
HLEO — Centralized HTTP retry for collector HTTP calls
======================================================

Provides http_get(): a drop-in replacement for requests.get() with:

- Retry on transient errors: 429, 500, 502, 503, 504, Timeout, ConnectionError
- No retry on permanent errors: 400, 401, 403, 404, other 4xx
- Retry-After header support (capped at backoff_max_s)
- Exponential backoff with jitter
- Configurable max_retries, timeout, backoff

Design rules
------------
- Retry is per single HTTP request, not per collector.search().
- Parameters come from the caller (from Global Limits) — no hidden imports.
- Never raises on the first attempt if status is retryable; raises only when
  all retries are exhausted or the error is permanent.
- The caller decides whether to catch the final exception.
"""
from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)

# Status codes that warrant a retry (transient server-side issues)
_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# Status codes that are permanent failures — never retry
_PERMANENT_FAILURE_STATUS: frozenset[int] = frozenset({400, 401, 403, 404, 405, 410, 422})


def http_get(
    url: str,
    params=None,
    timeout: float = 20.0,
    max_retries: int = 2,
    backoff_base_s: float = 1.0,
    backoff_max_s: float = 10.0,
) -> requests.Response:
    """
    Execute GET *url* with automatic retry on transient errors.

    Parameters
    ----------
    url           : Target URL.
    params        : Query parameters (dict or list of tuples).
    timeout       : Per-request timeout in seconds.
    max_retries   : Maximum *additional* attempts after the first (0 = no retry).
    backoff_base_s: Initial sleep before the second attempt (doubles each retry).
    backoff_max_s : Cap on a single sleep interval.

    Returns
    -------
    requests.Response — always a successful (2xx) response.

    Raises
    ------
    requests.HTTPError       — on permanent failure or exhausted retries.
    requests.Timeout         — if the final attempt times out.
    requests.ConnectionError — if the final attempt cannot connect.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)

            # Permanent failure — never retry regardless of remaining budget
            if r.status_code in _PERMANENT_FAILURE_STATUS:
                r.raise_for_status()
                return r  # unreachable; raise_for_status always raises for 4xx

            # Success
            if r.status_code < 400:
                return r

            # Transient server error — retry if budget allows
            if r.status_code in _RETRYABLE_STATUS:
                if attempt >= max_retries:
                    r.raise_for_status()  # exhausted — propagate
                sleep_s = _retry_sleep(r, attempt, backoff_base_s, backoff_max_s)
                logger.warning(
                    "HTTP %s from %s (attempt %d/%d) — retrying in %.1fs",
                    r.status_code, url, attempt + 1, max_retries + 1, sleep_s,
                )
                time.sleep(sleep_s)
                continue

            # Any other 4xx/5xx not in our tables — treat as permanent
            r.raise_for_status()

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            if attempt >= max_retries:
                raise
            sleep_s = min(backoff_base_s * (2 ** attempt), backoff_max_s)
            logger.warning(
                "%s on %s (attempt %d/%d) — retrying in %.1fs",
                type(exc).__name__, url, attempt + 1, max_retries + 1, sleep_s,
            )
            time.sleep(sleep_s)

    # Should be unreachable — kept as safety net
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"http_get exhausted retries without result for {url}")


def _retry_sleep(
    response: requests.Response,
    attempt: int,
    backoff_base_s: float,
    backoff_max_s: float,
) -> float:
    """
    Compute sleep duration for a retry.

    Respects the Retry-After header when present (integer seconds or HTTP date).
    Falls back to exponential backoff, capped at backoff_max_s.
    """
    retry_after = response.headers.get("Retry-After", "").strip()
    if retry_after:
        try:
            suggested = float(retry_after)
            return min(suggested, backoff_max_s)
        except ValueError:
            # HTTP-date format — parse it
            try:
                from email.utils import parsedate_to_datetime
                import datetime
                target = parsedate_to_datetime(retry_after)
                delta = (target - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
                if delta > 0:
                    return min(delta, backoff_max_s)
            except Exception:
                pass

    return min(backoff_base_s * (2 ** attempt), backoff_max_s)
