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
    """Fetch the latest persisted LLM configuration from the DB.

    On a fresh SQLite/Postgres database the hleo_llm_config table may not
    exist yet when this helper is called outside of the main FastAPI app
    (e.g. scripts/health_check.py). In that case we lazily create ONLY the
    LLMConfig table and retry once, instead of propagating a noisy
    "no such table" error.
    """
    try:
        from core.database import SessionLocal
        from core.models import LLMConfig
        from sqlalchemy.exc import OperationalError
    except Exception as exc:
        logger.warning("LLM config: DB stack unavailable — %s", exc)
        return None

    db = SessionLocal()
    try:
        try:
            row = db.execute(select(LLMConfig).order_by(LLMConfig.id.desc())).scalar_one_or_none()
        except OperationalError as exc:
            msg = str(exc).lower()
            # SQLite: "no such table"; Postgres: "does not exist" / "undefined table".
            if "hleo_llm_config" in msg and (
                "no such table" in msg or "does not exist" in msg or "undefined table" in msg
            ):
                try:
                    # Create only the LLMConfig table if it is missing.
                    LLMConfig.__table__.create(bind=db.bind, checkfirst=True)
                    db.commit()
                except Exception as create_exc:
                    logger.warning(
                        "LLM config: auto-create table failed — %s", create_exc
                    )
                    return None
                # Retry the query once after creating the table.
                try:
                    row = db.execute(select(LLMConfig).order_by(LLMConfig.id.desc())).scalar_one_or_none()
                except Exception as retry_exc:
                    logger.warning(
                        "LLM config: query after table create failed — %s", retry_exc
                    )
                    return None
            else:
                logger.warning(
                    "LLM config: DB query failed — %s", exc
                )
                return None

        if not row or not getattr(row, "enabled", True):
            return None

        provider = (getattr(row, "provider", "") or "").strip()
        protocol = (getattr(row, "protocol", None) or "OpenAI-compatible").strip() or "OpenAI-compatible"
        return {
            "provider": provider,
            "protocol": protocol,
            "base_url": (getattr(row, "base_url", "") or "").strip(),
            "model": (getattr(row, "model", "") or "").strip(),
            "api_key": _xor_decrypt(getattr(row, "api_key_encrypted", "") or ""),
        }
    except Exception as exc:
        logger.warning("LLM config: unable to load persisted Admin config — %s", exc)
        return None
    finally:
        try:
            db.close()
        except Exception:
            pass


def _get_db_llm_config_legacy() -> Optional[dict]:
    try:
        from core.database import SessionLocal
        from core.models import LLMConfig
        db = SessionLocal()
        try:
            row = db.execute(select(LLMConfig).order_by(LLMConfig.id.desc())).scalar_one_or_none()
            if not row or not row.enabled:
                return None
            provider = (row.provider or "").strip()
            protocol = (getattr(row, "protocol", None) or "OpenAI-compatible").strip() or "OpenAI-compatible"
            return {
                "provider": provider,
                "protocol": protocol,
                "base_url": (row.base_url or "").strip(),
                "model": (row.model or "").strip(),
                "api_key": _xor_decrypt(row.api_key_encrypted or ""),
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

    # When a Base URL is configured we always use the generic
    # OpenAI-compatible path. The provider name is metadata only.
    if base_url:
        return _build_generic_openai_compatible(provider_name or "openai-compatible", api_key, base_url)

    # No custom Base URL: fall back to the official OpenAI endpoint when an
    # API key is present. The provider label still flows into LLMProvider.name
    # but never changes routing behaviour.
    if api_key:
        return _build_default_openai(provider_name or "openai", api_key)

    return None


def llm_available() -> bool:
    """True when at least one LLM provider can be built from the active config."""
    return build_provider() is not None
