"""
HLEO — generic LLM runtime configuration
=======================================

The runtime connection target is driven by the configured LLM settings only:
Protocol + Base URL + API Key + Model.

Provider names are free-form labels entered by the user and never used as a
hardcoded routing switch.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy import select

logger = logging.getLogger(__name__)


@dataclass
class LLMProvider:
    """General OpenAI-compatible endpoint wrapper."""
    name: str
    client: Any
    fallback: Optional["LLMProvider"] = None


def _secret_key() -> str:
    key = (os.getenv("HLEO_SECRET_KEY") or os.getenv("HLEO_ADMIN_PASSWORD_HASH") or "hleo-default-dev-secret").strip()
    return key or "hleo-default-dev-secret"


def encrypt_secret(plain: str) -> str:
    if not plain:
        return ""
    key = hashlib.sha256(_secret_key().encode()).digest()
    raw = plain.encode()
    enc = bytes(b ^ key[i % len(key)] for i, b in enumerate(raw))
    return base64.b64encode(enc).decode()


def decrypt_secret(payload: str) -> str:
    if not payload:
        return ""
    key = hashlib.sha256(_secret_key().encode()).digest()
    try:
        raw = base64.b64decode(payload.encode())
    except Exception:
        return ""
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(raw)).decode()


def _xor_encrypt(plain: str) -> str:
    return encrypt_secret(plain)


def _xor_decrypt(payload: str) -> str:
    return decrypt_secret(payload)


def _get_db_llm_config() -> Optional[dict]:
    try:
        from core.database import SessionLocal, engine, Base
        from core.models import LLMConfig
        # Ensure table exists (handles case where app started before LLMConfig was registered)
        Base.metadata.create_all(bind=engine, tables=[LLMConfig.__table__])
        db = SessionLocal()
        try:
            row = db.execute(select(LLMConfig).order_by(LLMConfig.id.desc())).scalar_one_or_none()
            if not row:
                logger.info("LLM config: no row found in hleo_llm_config table")
                return None
            if not row.enabled:
                logger.info("LLM config: row found but disabled")
                return None
            provider = (row.provider or "").strip()
            protocol = (getattr(row, "protocol", None) or "OpenAI-compatible").strip() or "OpenAI-compatible"
            api_key_encrypted = row.api_key_encrypted or ""
            api_key = _xor_decrypt(api_key_encrypted)
            logger.info(
                "LLM config loaded from DB: provider=%s, base_url=%s, model=%s, api_key_present=%s",
                provider, (row.base_url or "").strip()[:50], (row.model or "").strip(),
                bool(api_key_encrypted)
            )
            return {
                "provider": provider,
                "protocol": protocol,
                "base_url": (row.base_url or "").strip(),
                "model": (row.model or "").strip(),
                "api_key": api_key,
            }
        finally:
            db.close()
    except Exception as exc:
        logger.warning("LLM config: unable to load persisted Admin config — %s", exc)
        return None


def _env_llm_config() -> dict:
    return {
        "provider": (os.getenv("HLEO_LLM_PROVIDER", "") or "").strip(),
        "protocol": "OpenAI-compatible",
        "base_url": (os.getenv("OPENAI_BASE_URL") or "").strip(),
        "model": (os.getenv("HLEO_LLM_MODEL") or "").strip(),
        "api_key": (os.getenv("OPENAI_API_KEY") or "").strip(),
    }


def get_active_llm_settings() -> dict:
    db_cfg = _get_db_llm_config()
    if db_cfg:
        return {
            "provider": (db_cfg.get("provider") or "").strip(),
            "protocol": (db_cfg.get("protocol") or "OpenAI-compatible").strip() or "OpenAI-compatible",
            "base_url": (db_cfg.get("base_url") or "").strip(),
            "model": (db_cfg.get("model") or "").strip(),
            "api_key": (db_cfg.get("api_key") or "").strip(),
        }
    return _env_llm_config()


def configured_provider_name() -> str:
    return (get_active_llm_settings().get("provider") or "").strip()


def resolve_model(provider_name: str, requested: str) -> str:
    """Model selection is driven by the active runtime settings only.

    Free-form provider names are metadata. They never trigger a special model
    mapping or endpoint route.
    """
    requested = (requested or "").strip()
    active = get_active_llm_settings()
    if active.get("model"):
        return active["model"]
    return requested


def _build_default_openai(provider_name: str, api_key: str) -> Optional[LLMProvider]:
    if not api_key:
        return None
    try:
        from openai import OpenAI
        name = (provider_name or "openai").strip() or "openai"
        return LLMProvider(name=name, client=OpenAI(api_key=api_key))
    except Exception as exc:
        logger.warning("LLM provider: OpenAI init failed for %s — %s", provider_name, exc)
        return None


def _build_generic_openai_compatible(provider_name: str, api_key: str, base_url: str) -> Optional[LLMProvider]:
    if not base_url:
        raise ValueError(f"Custom OpenAI-compatible provider '{provider_name or 'OpenAI-compatible'}' requires a base_url.")
    try:
        from openai import OpenAI
        name = (provider_name or "openai-compatible").strip() or "openai-compatible"
        client = OpenAI(api_key=api_key or "generic", base_url=base_url)
        return LLMProvider(name=name, client=client)
    except Exception as exc:
        logger.warning("LLM provider: generic OpenAI-compatible init failed for %s — %s", provider_name, exc)
        return None


def build_provider(prefer: Optional[str] = None) -> Optional[LLMProvider]:
    """Build the configured LLM provider.

    Provider names are free-form labels only. The runtime connection target is
    determined by the configured Base URL, API Key, and Model; provider labels
    never route to a special endpoint or select a fallback.
    """
    active = get_active_llm_settings()
    env_cfg = _env_llm_config()

    api_key = (active.get("api_key") or env_cfg["api_key"]).strip()
    base_url = (active.get("base_url") or env_cfg["base_url"]).strip()
    provider_name = (prefer or active.get("provider") or env_cfg["provider"] or "").strip()

    logger.info(
        "build_provider: provider=%s, base_url=%s, model=%s, api_key_present=%s, env_base_url=%s",
        provider_name, (base_url or "").strip()[:50],
        (active.get("model") or env_cfg["model"] or "").strip(),
        bool(api_key), (env_cfg["base_url"] or "").strip()[:50]
    )

    if base_url:
        return _build_generic_openai_compatible(provider_name or "openai-compatible", api_key, base_url)

    if provider_name:
        raise ValueError(f"Custom OpenAI-compatible provider '{provider_name}' requires a base_url.")

    if api_key:
        return _build_default_openai(provider_name or "openai", api_key)

    return None


def llm_available() -> bool:
    """True when at least one LLM provider can be built from the active config."""
    return build_provider() is not None
