"""
HLEO — Multi-provider LLM Manager
==================================

Central coordinator for the up-to-4 configurable provider slots.

Responsibilities
----------------
- Load enabled slots ordered by slot_index (priority 1 = highest).
- Expose a typed ProviderStage list consumed by llm_guard.
- Migrate a legacy single-row LLMConfig into slot 1 on first access.
- Keep backward compatibility: build_provider() still works unchanged.

Design rules
------------
- Provider names are free-form labels; never used for routing.
- All slots use the OpenAI SDK (OpenAI-compatible protocol).
- Empty slot (no api_key and no base_url) = slot disabled at runtime
  even if DB row says enabled=True.
- No hardcoded provider names, URLs, or model mappings here.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional

from sqlalchemy import select

logger = logging.getLogger(__name__)

# Maximum slots supported
MAX_SLOTS = 4


@dataclass
class ProviderStage:
    """One resolved provider stage ready for llm_guard to use."""
    name: str
    client: Any           # openai.OpenAI instance
    model_override: str   # "" = use call-site model
    max_retries: int      # per-stage retry budget (0 = no retry after 1st attempt)
    slot_index: int       # 1-4 for logging


def _build_client(api_key: str, base_url: str, timeout_s: float) -> Any:
    """Build an OpenAI-compatible client. Returns None on failure."""
    from openai import OpenAI
    kwargs: dict = {"api_key": api_key or "generic"}
    if base_url:
        kwargs["base_url"] = base_url
    if timeout_s and timeout_s > 0:
        kwargs["timeout"] = timeout_s
    return OpenAI(**kwargs)


def _slot_to_stage(row) -> Optional[ProviderStage]:
    """Convert a DB LLMProviderSlot row to a ProviderStage. Returns None if unusable."""
    from core.llm_provider import decrypt_secret
    api_key = decrypt_secret(getattr(row, "api_key_encrypted", "") or "").strip()
    base_url = (getattr(row, "base_url", "") or "").strip()
    if not api_key and not base_url:
        return None  # slot has no credentials → skip
    try:
        client = _build_client(api_key, base_url, float(getattr(row, "timeout_s", 60.0) or 60.0))
        return ProviderStage(
            name=(getattr(row, "name", "") or f"slot-{row.slot_index}").strip() or f"slot-{row.slot_index}",
            client=client,
            model_override=(getattr(row, "model", "") or "").strip(),
            max_retries=max(0, min(4, int(getattr(row, "max_retries", 2) or 2))),
            slot_index=int(row.slot_index),
        )
    except Exception as exc:
        logger.warning("LLMManager: slot %s init failed — %s", getattr(row, "slot_index", "?"), exc)
        return None


def _migrate_legacy_config(db) -> None:
    """If no slots exist but a legacy LLMConfig row does, copy it to slot 1."""
    try:
        from core.models import LLMConfig, LLMProviderSlot
        existing = db.execute(select(LLMProviderSlot)).scalars().first()
        if existing:
            return  # slots already seeded
        legacy = db.execute(select(LLMConfig).order_by(LLMConfig.id.desc())).scalar_one_or_none()
        if legacy is None or not getattr(legacy, "enabled", True):
            return
        # Seed slot 1 from the legacy config.
        slot = LLMProviderSlot(
            slot_index=1,
            enabled=True,
            name=(getattr(legacy, "provider", "") or "").strip() or "Provider 1",
            protocol=(getattr(legacy, "protocol", None) or "OpenAI-compatible"),
            api_key_encrypted=getattr(legacy, "api_key_encrypted", "") or "",
            base_url=(getattr(legacy, "base_url", "") or "").strip(),
            model=(getattr(legacy, "model", "") or "").strip(),
            timeout_s=60.0,
            max_retries=2,
        )
        # Ensure slots 2-4 exist as disabled placeholders.
        db.add(slot)
        for i in range(2, MAX_SLOTS + 1):
            db.add(LLMProviderSlot(slot_index=i, enabled=False, name=f"Provider {i}"))
        db.commit()
        logger.info("LLMManager: migrated legacy LLMConfig to slot 1.")
    except Exception as exc:
        logger.warning("LLMManager: legacy migration failed — %s", exc)
        try:
            db.rollback()
        except Exception:
            pass


def _ensure_slots_exist(db) -> None:
    """Ensure all 4 slot rows exist in the DB (idempotent)."""
    from core.models import LLMProviderSlot
    for i in range(1, MAX_SLOTS + 1):
        existing = db.execute(
            select(LLMProviderSlot).where(LLMProviderSlot.slot_index == i)
        ).scalar_one_or_none()
        if existing is None:
            db.add(LLMProviderSlot(slot_index=i, enabled=False, name=f"Provider {i}"))
    db.commit()


def get_provider_chain() -> List[ProviderStage]:
    """Return enabled provider stages in priority order (slot 1 first).

    Falls back to a legacy single-provider config if no slots are configured.
    Returns [] when no provider is available.
    """
    try:
        from core.database import SessionLocal
        from core.models import LLMProviderSlot
        db = SessionLocal()
        try:
            _migrate_legacy_config(db)
            _ensure_slots_exist(db)
            rows = db.execute(
                select(LLMProviderSlot)
                .where(LLMProviderSlot.enabled == True)  # noqa: E712
                .order_by(LLMProviderSlot.slot_index)
            ).scalars().all()
            chain: List[ProviderStage] = []
            for row in rows:
                stage = _slot_to_stage(row)
                if stage is not None:
                    chain.append(stage)
            if chain:
                return chain
        finally:
            db.close()
    except Exception as exc:
        logger.warning("LLMManager: DB unavailable — %s", exc)

    # DB unavailable or no slots: fall back to env vars.
    return _chain_from_env()


def _chain_from_env() -> List[ProviderStage]:
    """Build a single-stage chain from environment variables (legacy fallback)."""
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    if not api_key and not base_url:
        return []
    try:
        client = _build_client(api_key, base_url, 60.0)
        name = (os.getenv("HLEO_LLM_PROVIDER") or "openai").strip() or "openai"
        model = (os.getenv("HLEO_LLM_MODEL") or "").strip()
        return [ProviderStage(name=name, client=client, model_override=model,
                              max_retries=2, slot_index=0)]
    except Exception as exc:
        logger.warning("LLMManager: env chain build failed — %s", exc)
        return []


def get_slot_status() -> List[dict]:
    """Return a status dict for each of the 4 slots (for admin API)."""
    try:
        from core.database import SessionLocal
        from core.models import LLMProviderSlot
        from core.llm_provider import decrypt_secret
        db = SessionLocal()
        try:
            _ensure_slots_exist(db)
            rows = db.execute(
                select(LLMProviderSlot).order_by(LLMProviderSlot.slot_index)
            ).scalars().all()
            result = []
            for row in rows:
                api_key = decrypt_secret(row.api_key_encrypted or "")
                result.append({
                    "slot_index": row.slot_index,
                    "enabled": row.enabled,
                    "name": row.name or "",
                    "protocol": row.protocol or "OpenAI-compatible",
                    "api_key_configured": bool(api_key.strip()),
                    "base_url": row.base_url or "",
                    "model": row.model or "",
                    "timeout_s": row.timeout_s or 60.0,
                    "max_retries": row.max_retries if row.max_retries is not None else 2,
                    "rate_limit_rpm": row.rate_limit_rpm,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                })
            return result
        finally:
            db.close()
    except Exception as exc:
        logger.warning("LLMManager: get_slot_status failed — %s", exc)
        return [{"slot_index": i, "enabled": False, "name": f"Provider {i}",
                 "protocol": "OpenAI-compatible", "api_key_configured": False,
                 "base_url": "", "model": "", "timeout_s": 60.0,
                 "max_retries": 2, "rate_limit_rpm": None,
                 "created_at": None, "updated_at": None}
                for i in range(1, MAX_SLOTS + 1)]