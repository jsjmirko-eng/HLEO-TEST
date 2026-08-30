#!/usr/bin/env python3
"""
Temp-store abstraction for ephemeral search results.

Design goals:
- pluggable backend (Redis optional) with a fallback in-memory implementation
- TTL per-key, automatic cleanup
- simple API: set/get/delete/contains
- suitable for storing search-scoped results (search_id -> payload)

This module exposes a singleton `temp_store` instance obtained from get_temp_store().
"""
from __future__ import annotations
import os
import time
import threading
import json
from typing import Any, Dict, Optional

DEFAULT_TTL = int(os.getenv("TEMP_RESULTS_TTL", "3600"))  # default 1 hour (provisional)
_CLEANUP_INTERVAL = int(os.getenv("TEMP_STORE_CLEANUP_INTERVAL", "60"))  # seconds


class TempStoreBase:
    """Minimal interface for ephemeral store."""
    def set(self, key: str, value: Any, ttl: int = DEFAULT_TTL) -> None:
        raise NotImplementedError

    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def contains(self, key: str) -> bool:
        raise NotImplementedError


class InMemoryTempStore(TempStoreBase):
    """Thread-safe in-process temp store with TTL and periodic cleanup.

    Notes:
    - Simple, intended for development and single-process deployments.
    - Keys and values are kept in memory; will be lost on process restart.
    - TTL is enforced on get and by a background cleanup thread.
    """
    def __init__(self, cleanup_interval: int = _CLEANUP_INTERVAL):
        self._store: Dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self._cleanup_interval = cleanup_interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._thread.start()

    def set(self, key: str, value: Any, ttl: int = DEFAULT_TTL) -> None:
        expiry = time.time() + ttl
        with self._lock:
            # store JSON-serializable representation to avoid surprising references
            try:
                _ = json.dumps(value)
                store_value = value
            except Exception:
                # fallback: store repr if not JSON serializable
                store_value = {"__repr__": repr(value)}
            self._store[key] = (store_value, expiry)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            value, expiry = item
            if expiry < time.time():
                # expired
                del self._store[key]
                return None
            return value

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def contains(self, key: str) -> bool:
        return self.get(key) is not None

    def _cleanup_loop(self) -> None:
        while not self._stop.wait(self._cleanup_interval):
            now = time.time()
            with self._lock:
                expired = [k for k, (_, exp) in self._store.items() if exp <= now]
                for k in expired:
                    del self._store[k]

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


def get_temp_store() -> TempStoreBase:
    """Factory: returns a singleton TempStore instance.

    Behavior:
    - If REDIS_URL environment variable present and redis package importable:
      (optional) use Redis-backed implementation (not mandatory for this phase).
    - Otherwise use in-memory fallback.
    """
    # Minimal (safe) approach: prefer in-memory in this phase
    return InMemoryTempStore()


# Module-level singleton used by the app
temp_store: TempStoreBase = get_temp_store()

__all__ = ["temp_store", "TempStoreBase", "InMemoryTempStore", "get_temp_store", "DEFAULT_TTL"]
