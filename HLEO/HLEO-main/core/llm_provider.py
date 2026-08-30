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

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

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


def configured_provider_name() -> str:
    return (os.getenv("HLEO_LLM_PROVIDER", "auto") or "auto").strip().lower() or "auto"


def resolve_model(provider_name: str, requested: str) -> str:
    """Map a caller-requested (OpenAI-style) model to the provider's model.

    Callers keep passing their usual "gpt-4o" / "gpt-4o-mini" labels; the
    mapping is applied only for non-OpenAI providers, so OpenAI behaviour is
    byte-identical to before.
    """
    requested = requested or ""
    if provider_name == "perplexity":
        if "mini" in requested:
            return os.getenv("HLEO_PERPLEXITY_MODEL_MINI", DEFAULT_PERPLEXITY_MODEL_MINI)
        return os.getenv("HLEO_PERPLEXITY_MODEL", DEFAULT_PERPLEXITY_MODEL)
    if provider_name == "local":
        return (os.getenv("HLEO_LLM_MODEL", "") or "").strip() or requested
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


def build_provider(prefer: Optional[str] = None) -> Optional[LLMProvider]:
    """Build the configured LLM provider. Returns None when nothing is
    configured (callers treat that as "LLM disabled", exactly like the
    previous "no OPENAI_API_KEY" behaviour).

    Fallback policy (linear, one-way): perplexity → openai. OpenAI and local
    never fall back, and no provider ever falls back TO Perplexity, so a
    Perplexity ↔ OpenAI loop is impossible by construction.
    """
    pref = (prefer or configured_provider_name()).lower()
    pplx_key = (os.getenv("PERPLEXITY_API_KEY") or "").strip()
    oai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    local_base = (os.getenv("OPENAI_BASE_URL") or "").strip()

    def _pplx() -> Optional[LLMProvider]:
        p = _build_perplexity(pplx_key)
        if p is not None:
            p.fallback = _build_openai(oai_key)
        return p

    if pref == "perplexity":
        return _pplx()
    if pref == "openai":
        return _build_openai(oai_key)
    if pref == "local":
        return _build_local(local_base, oai_key)
    # auto: Perplexity when its key is configured, else OpenAI, else local.
    if pplx_key:
        return _pplx()
    if oai_key:
        return _build_openai(oai_key)
    return _build_local(local_base, oai_key)


def llm_available() -> bool:
    """True when at least one LLM provider can be built from the env."""
    return build_provider() is not None
