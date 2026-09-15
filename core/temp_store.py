#!/usr/bin/env python3
"""
Temp-store abstraction for ephemeral search results.

Design goals:
- pluggable backend (Redis optional) with a fallback in-memory implementation
- TTL per-key, automatic cleanup
- simple API: set/get/delete/contains
- suitable for storing search-scoped results (search_id -> payload)

Single-tenant by design
-----------------------
This module exposes a singleton ``temp_store`` instance shared across the
entire process.  Isolation between callers is provided solely by the UUID-based
key (search_id).  There is no per-user or per-session namespace: any caller
that possesses a valid search_id can read its payload.  This is intentional for
a single-tenant deployment; multi-tenant deployments should add a namespace
prefix (e.g. ``<user_id>:<search_id>``) at the call site.

This module exposes a singleton `temp_store` instance obtained from get_temp_store().
"""
from __future__ import annotations
import collections
import os
import time
import threading
import json
from typing import Any, Dict, Optional

DEFAULT_TTL = int(os.getenv("TEMP_RESULTS_TTL", "300"))  # default 5 minutes
_CLEANUP_INTERVAL = int(os.getenv("TEMP_STORE_CLEANUP_INTERVAL", "60"))  # seconds

# Maximum number of keys retained simultaneously.  When the limit is reached
# the oldest-inserted key is evicted (FIFO) before the new one is written.
# Set to 0 to disable the limit (not recommended in production).
_MAX_KEYS = int(os.getenv("TEMP_STORE_MAX_KEYS", "1000"))


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
    """Thread-safe in-process temp store with TTL, periodic cleanup, and bounded size.

    Notes:
    - Simple, intended for development and single-process deployments.
    - Keys and values are kept in memory; will be lost on process restart.
    - TTL is enforced on get and by a background cleanup thread.
    - When ``maxsize > 0`` and the store is full, the oldest key (FIFO) is
      evicted before the new entry is written.  This prevents unbounded memory
      growth under sustained load (e.g. 1 req/s × 300 s TTL ≈ 300 live keys).

    Single-tenant by design: see module docstring for isolation guarantees.
    """
    def __init__(
        self,
        cleanup_interval: int = _CLEANUP_INTERVAL,
        maxsize: int = _MAX_KEYS,
    ):
        # OrderedDict preserves insertion order → O(1) FIFO eviction.
        self._store: collections.OrderedDict[str, tuple[Any, float]] = (
            collections.OrderedDict()
        )
        self._lock = threading.Lock()
        self._cleanup_interval = cleanup_interval
        self._maxsize = maxsize  # 0 = unlimited
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

            # If key already exists, remove it first so the new insertion
            # lands at the tail (preserving FIFO order correctly).
            if key in self._store:
                del self._store[key]

            # Evict oldest entries when at capacity.
            if self._maxsize > 0:
                while len(self._store) >= self._maxsize:
                    self._store.popitem(last=False)  # FIFO: remove oldest

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
