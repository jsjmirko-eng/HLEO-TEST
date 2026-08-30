"""
HLEO — LLM provider selection (OpenAI / Perplexity / local OpenAI-compatible)
=============================================================================

Single, env-driven factory for the LLM client used across the whole app.
This module does NOT implement any retry: the only retry boundary remains
``core.llm_guard`` (MAX_TOTAL_ATTEMPTS). Provider fallback is a linear,
one-way chain (perplexity → openai); never a loop, never duplicated retries.

Configuration (environment only — no code changes needed to switch provider):
    HLEO_LLM_PROVIDER   "auto" (default) | "openai" | "perplexity" | "local"
    PERPLEXITY_API_KEY  Perplexity Sonar API key (never logged or hardcoded)
    OPENAI_API_KEY      OpenAI API key
    HLEO_PERPLEXITY_MODEL       default "sonar-pro" (replaces gpt-4o-class calls)
    HLEO_PERPLEXITY_MODEL_MINI  default "sonar"     (replaces gpt-4o-mini-class calls)
    OPENAI_BASE_URL + HLEO_LLM_MODEL  local OpenAI-compatible endpoint
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

PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
DEFAULT_PERPLEXITY_MODEL = "sonar-pro"
DEFAULT_PERPLEXITY_MODEL_MINI = "sonar"


@dataclass
class LLMProvider:
    """An OpenAI-compatible chat-completions endpoint + one-way fallback."""
    name: str                      # "openai" | "perplexity" | "local"
    client: Any                    # openai.OpenAI instance
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
        from core.database import SessionLocal
        from core.models import LLMConfig
        db = SessionLocal()
        try:
            row = db.execute(select(LLMConfig).order_by(LLMConfig.id.desc())).scalar_one_or_none()
            if not row or not row.enabled:
                return None
            provider = (row.provider or "auto").strip() or "auto"
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
        "provider": (os.getenv("HLEO_LLM_PROVIDER", "auto") or "auto").strip() or "auto",
        "protocol": "OpenAI-compatible",
        "base_url": (os.getenv("OPENAI_BASE_URL") or "").strip(),
        "model": (os.getenv("HLEO_LLM_MODEL") or "").strip(),
        "api_key": (os.getenv("OPENAI_API_KEY") or "").strip(),
    }


def get_active_llm_settings() -> dict:
    db_cfg = _get_db_llm_config()
    if db_cfg:
        return {
            "provider": (db_cfg.get("provider") or "auto").strip() or "auto",
            "protocol": (db_cfg.get("protocol") or "OpenAI-compatible").strip() or "OpenAI-compatible",
            "base_url": (db_cfg.get("base_url") or "").strip(),
            "model": (db_cfg.get("model") or "").strip(),
            "api_key": (db_cfg.get("api_key") or "").strip(),
        }
    return _env_llm_config()


def configured_provider_name() -> str:
    return (get_active_llm_settings().get("provider", "auto") or "auto").strip() or "auto"


def resolve_model(provider_name: str, requested: str) -> str:
    """Map a caller-requested (OpenAI-style) model to the provider's model.

    The runtime keeps using OpenAI-style requests but a free-form provider name
    can be configured via Admin. In that case, the configured Base URL + Model
    take precedence, and there is no hardcoded provider-specific mapping.
    """
    requested = requested or ""
    active = get_active_llm_settings()
    pn = (provider_name or "").strip().lower()
    if pn == "perplexity":
        if active.get("model"):
            return active["model"]
        if "mini" in requested:
            return os.getenv("HLEO_PERPLEXITY_MODEL_MINI", DEFAULT_PERPLEXITY_MODEL_MINI)
        return os.getenv("HLEO_PERPLEXITY_MODEL", DEFAULT_PERPLEXITY_MODEL)
    if pn == "local":
        return (active.get("model") or os.getenv("HLEO_LLM_MODEL", "") or "").strip() or requested
    if pn and pn not in {"auto", "openai"}:
        return (active.get("model") or requested).strip() or requested
    return requested


def _build_openai(api_key: str) -> Optional[LLMProvider]:
    if not api_key:
        return None
    try:
        from openai import OpenAI
        return LLMProvider(name="openai", client=OpenAI(api_key=api_key))
    except Exception as exc:
        logger.warning("LLM provider: OpenAI init failed — %s", exc)
        return None


def _build_perplexity(api_key: str) -> Optional[LLMProvider]:
    if not api_key:
        return None
    try:
        from openai import OpenAI
        return LLMProvider(
            name="perplexity",
            client=OpenAI(api_key=api_key, base_url=PERPLEXITY_BASE_URL),
        )
    except Exception as exc:
        logger.warning("LLM provider: Perplexity init failed — %s", exc)
        return None


def _build_local(base_url: str, api_key: str) -> Optional[LLMProvider]:
    if not base_url:
        return None
    try:
        from openai import OpenAI
        return LLMProvider(
            name="local",
            client=OpenAI(api_key=api_key or "local", base_url=base_url),
        )
    except Exception as exc:
        logger.warning("LLM provider: local endpoint init failed — %s", exc)
        return None


def _build_generic_openai_compatible(provider_name: str, api_key: str, base_url: str) -> Optional[LLMProvider]:
    if not base_url:
        raise ValueError(f"Custom provider '{provider_name or 'OpenAI-compatible'}' requires a base_url for OpenAI-compatible endpoints.")
    if not (base_url or api_key):
        return None
    try:
        from openai import OpenAI
        name = (provider_name or "openai-compatible").strip() or "openai-compatible"
        client = OpenAI(api_key=api_key or "generic", base_url=base_url)
        return LLMProvider(name=name, client=client)
    except Exception as exc:
        logger.warning("LLM provider: generic OpenAI-compatible init failed for %s — %s", provider_name, exc)
        return None


def build_provider(prefer: Optional[str] = None) -> Optional[LLMProvider]:
    """Build the configured LLM provider. Returns None when nothing is
    configured (callers treat that as "LLM disabled", exactly like the
    previous "no OPENAI_API_KEY" behaviour).

    The Admin LLM configuration is the single source of truth. Provider names are
    treated as descriptive identifiers; the actual endpoint is determined by the
    configured Base URL and protocol. Legacy compatibility is kept only for the
    truly special providers (Perplexity and local endpoints), but no custom or
    OpenAI-compatible provider is silently redirected to api.openai.com.
    """
    active = get_active_llm_settings()
    env_cfg = _env_llm_config()

    pplx_key = (active.get("api_key") if active.get("provider") == "perplexity" else os.getenv("PERPLEXITY_API_KEY") or "").strip()
    if not pplx_key:
        pplx_key = (os.getenv("PERPLEXITY_API_KEY") or "").strip()
    oai_key = (active.get("api_key") or env_cfg["api_key"]).strip()
    local_base = (active.get("base_url") or env_cfg["base_url"]).strip()
    provider_name = (prefer or active.get("provider") or env_cfg["provider"] or "auto").strip()
    provider_key = provider_name.lower()

    def _pplx() -> Optional[LLMProvider]:
        p = _build_perplexity(pplx_key)
        if p is not None:
            p.fallback = _build_openai(oai_key)
        return p

    # Explicit special-case compatibility remains for legacy Perplexity and local
    # endpoints. For everything else, the actual endpoint always comes from the
    # configured Base URL and not from a hardcoded provider map.
    if provider_key == "perplexity":
        return _pplx()
    if provider_key == "local":
        if not local_base:
            raise ValueError("Local OpenAI-compatible provider requires a base_url.")
        return _build_local(local_base, oai_key)

    # If a non-empty Base URL is configured, always prefer the exact user-defined
    # OpenAI-compatible endpoint. This includes any provider name, including
    # "openai", "openrouter", or "groq", as long as the runtime is driven by
    # the configured Protocol + Base URL + API Key + Model.
    if local_base:
        provider_label = provider_name or "openai-compatible"
        if provider_label.lower() not in {"auto", "default", "none"}:
            return _build_generic_openai_compatible(provider_label, oai_key, local_base)
        if oai_key:
            return _build_generic_openai_compatible("openai-compatible", oai_key, local_base)

    # Free-form provider names without a Base URL are invalid for custom OpenAI-
    # compatible endpoints. Never silently fall back to api.openai.com.
    if provider_name and provider_name.lower() not in {"auto", "default", "none"}:
        raise ValueError(f"Custom OpenAI-compatible provider '{provider_name}' requires a base_url.")

    if pplx_key and provider_key in {"", "auto", "default", "none"}:
        return _pplx()
    if oai_key:
        return _build_openai(oai_key)
    return _build_local(local_base, oai_key)


def llm_available() -> bool:
    """True when at least one LLM provider can be built from the env."""
    return build_provider() is not None
