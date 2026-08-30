"""
Cache for vocabulary provider responses.

- In-memory dict with per-provider namespacing and TTL (same minimal pattern
  as core/orchestrator's process-lifetime cache).
- Optional JSON file persistence (HLEO_VOCAB_CACHE_PATH); default = memory
  only, nothing written to the repository.
- Keys never contain credentials; values store provider payloads only.
- Easily invalidated: HLEO_VOCAB_CACHE_DISABLE=1, invalidate(), or delete
  the cache file.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_TTL = 86400  # 24h


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


class VocabCache:
    def __init__(
        self,
        ttl: Optional[int] = None,
        path: Optional[str] = None,
    ) -> None:
        self.ttl = int(ttl if ttl is not None
                       else os.getenv("HLEO_VOCAB_CACHE_TTL", DEFAULT_TTL))
        self.disabled = _env_flag("HLEO_VOCAB_CACHE_DISABLE")
        self.path = Path(path) if (path or os.getenv("HLEO_VOCAB_CACHE_PATH")) else None
        self._store: dict = {}
        if self.path and self.path.exists():
            try:
                self._store = json.loads(self.path.read_text())
            except Exception:  # noqa: BLE001 — corrupt cache is just a miss
                logger.warning("vocab cache file unreadable, starting empty")
                self._store = {}

    @staticmethod
    def _key(provider: str, op: str, term: str, language: str = "") -> str:
        raw = f"{provider}|{op}|{term.lower().strip()}|{language.lower()}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, provider: str, op: str, term: str, language: str = ""):
        if self.disabled:
            return None
        k = self._key(provider, op, term, language)
        entry = self._store.get(k)
        if not entry:
            return None
        if time.time() - entry.get("ts", 0) > self.ttl:
            self._store.pop(k, None)
            return None
        return entry.get("value")

    def set(self, provider: str, op: str, term: str, value, language: str = "") -> None:
        if self.disabled:
            return
        k = self._key(provider, op, term, language)
        self._store[k] = {
            "provider": provider,
            "op": op,
            "ts": time.time(),
            "value": value,
        }
        self._flush()

    def invalidate(self, provider: Optional[str] = None) -> int:
        """Drop entries (one provider or all). Returns how many were removed."""
        if provider is None:
            n = len(self._store)
            self._store.clear()
        else:
            keys = [k for k, e in self._store.items()
                    if e.get("provider") == provider]
            for k in keys:
                self._store.pop(k, None)
            n = len(keys)
        self._flush()
        return n

    def _flush(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._store))
        except Exception:  # noqa: BLE001 — persistence is best-effort
            logger.warning("vocab cache flush failed", exc_info=True)

    def __len__(self) -> int:
        return len(self._store)
