"""
Tests for core/llm_provider.py + provider routing in core/llm_guard.py.

Covers:
  - provider selection via env (auto / openai / perplexity / local)
  - Perplexity client construction (official OpenAI-compatible endpoint,
    key read ONLY from PERPLEXITY_API_KEY, never hardcoded/logged)
  - model mapping (gpt-4o → sonar-pro, gpt-4o-mini → sonar, env overrides)
  - guard routing: provider success, JSON parsing reuse, json_object drop
  - one-way fallback Perplexity → OpenAI (quota + exhaustion), no loop back
  - OpenAI and local providers keep working unchanged
  - backward compatibility with raw SDK clients

All offline: fake SDK clients, no network, no real keys.
"""
from __future__ import annotations

import inspect
import json
from unittest.mock import MagicMock

import pytest

import core.llm_provider as lp
from core.llm_provider import (
    DEFAULT_PERPLEXITY_MODEL,
    DEFAULT_PERPLEXITY_MODEL_MINI,
    PERPLEXITY_BASE_URL,
    LLMProvider,
    build_provider,
    llm_available,
    resolve_model,
)
from core.llm_guard import (
    MAX_TOTAL_ATTEMPTS,
    LLMCallError,
    QuotaExhaustedError,
    call_llm,
    call_llm_json,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _choice(content):
    m = MagicMock()
    m.message.content = content
    return MagicMock(choices=[m])


def _raw_client(side_effects):
    client = MagicMock()
    client.chat.completions.create.side_effect = side_effects
    return client


def _patch_openai_ctor(monkeypatch):
    """Replace openai.OpenAI with a recorder; returns the kwargs list."""
    calls = []

    def fake_openai(**kwargs):
        calls.append(kwargs)
        return MagicMock(name=f"client-{len(calls)}")

    monkeypatch.setattr("openai.OpenAI", fake_openai)
    return calls


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch):
    for var in ("PERPLEXITY_API_KEY", "OPENAI_API_KEY", "HLEO_LLM_PROVIDER",
                "OPENAI_BASE_URL", "HLEO_LLM_MODEL", "HLEO_PERPLEXITY_MODEL",
                "HLEO_PERPLEXITY_MODEL_MINI"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("core.llm_guard.time.sleep", lambda d: None)


# ── Provider selection ───────────────────────────────────────────────────────

class TestProviderSelection:
    def test_auto_openai_only(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        calls = _patch_openai_ctor(monkeypatch)
        p = build_provider()
        assert p is not None and p.name == "openai"
        assert p.fallback is None
        assert calls == [{"api_key": "test-openai-key"}]

    def test_auto_perplexity_preferred_with_fallback(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-pplx-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        calls = _patch_openai_ctor(monkeypatch)
        p = build_provider()
        assert p.name == "perplexity"
        assert p.fallback is not None and p.fallback.name == "openai"
        assert p.fallback.fallback is None  # one-way chain, no loop
        assert calls[0] == {"api_key": "test-pplx-key", "base_url": PERPLEXITY_BASE_URL}
        assert calls[1] == {"api_key": "test-openai-key"}

    def test_auto_perplexity_without_openai_has_no_fallback(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-pplx-key")
        _patch_openai_ctor(monkeypatch)
        p = build_provider()
        assert p.name == "perplexity"
        assert p.fallback is None

    def test_explicit_openai_ignores_perplexity_key(self, monkeypatch):
        monkeypatch.setenv("HLEO_LLM_PROVIDER", "openai")
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-pplx-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        _patch_openai_ctor(monkeypatch)
        p = build_provider()
        assert p.name == "openai" and p.fallback is None

    def test_explicit_perplexity(self, monkeypatch):
        monkeypatch.setenv("HLEO_LLM_PROVIDER", "perplexity")
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-pplx-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        _patch_openai_ctor(monkeypatch)
        p = build_provider()
        assert p.name == "perplexity"
        assert p.fallback is not None and p.fallback.name == "openai"

    def test_explicit_perplexity_without_key_is_none(self, monkeypatch):
        monkeypatch.setenv("HLEO_LLM_PROVIDER", "perplexity")
        _patch_openai_ctor(monkeypatch)
        assert build_provider() is None
        assert not llm_available()

    def test_local_provider_via_base_url(self, monkeypatch):
        monkeypatch.setenv("HLEO_LLM_PROVIDER", "local")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
        calls = _patch_openai_ctor(monkeypatch)
        p = build_provider()
        assert p.name == "local" and p.fallback is None
        assert calls == [{"api_key": "local", "base_url": "http://localhost:1234/v1"}]

    def test_auto_local_when_only_base_url(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
        _patch_openai_ctor(monkeypatch)
        p = build_provider()
        assert p is not None and p.name == "local"

    def test_nothing_configured_returns_none(self, monkeypatch):
        _patch_openai_ctor(monkeypatch)
        assert build_provider() is None
        assert not llm_available()


# ── Model resolution ─────────────────────────────────────────────────────────

class TestModelResolution:
    def test_perplexity_full_model(self):
        assert resolve_model("perplexity", "gpt-4o") == DEFAULT_PERPLEXITY_MODEL

    def test_perplexity_mini_model(self):
        assert resolve_model("perplexity", "gpt-4o-mini") == DEFAULT_PERPLEXITY_MODEL_MINI

    def test_perplexity_model_env_override(self, monkeypatch):
        monkeypatch.setenv("HLEO_PERPLEXITY_MODEL", "sonar-reasoning-pro")
        monkeypatch.setenv("HLEO_PERPLEXITY_MODEL_MINI", "sonar-pro")
        assert resolve_model("perplexity", "gpt-4o") == "sonar-reasoning-pro"
        assert resolve_model("perplexity", "gpt-4o-mini") == "sonar-pro"

    def test_openai_passthrough(self):
        assert resolve_model("openai", "gpt-4o") == "gpt-4o"
        assert resolve_model("openai", "gpt-4o-mini") == "gpt-4o-mini"

    def test_local_model_override(self, monkeypatch):
        monkeypatch.setenv("HLEO_LLM_MODEL", "qwen2.5-7b-instruct")
        assert resolve_model("local", "gpt-4o") == "qwen2.5-7b-instruct"

    def test_local_without_override_passthrough(self):
        assert resolve_model("local", "gpt-4o-mini") == "gpt-4o-mini"


# ── Guard routing through a provider ─────────────────────────────────────────

class TestGuardProviderRouting:
    def test_perplexity_success_uses_mapped_model(self):
        raw = _raw_client([_choice("sonar answer")])
        provider = LLMProvider(name="perplexity", client=raw)
        out = call_llm(provider, messages=[{"role": "user", "content": "hi"}],
                       model="gpt-4o", operation="t")
        assert out == "sonar answer"
        kwargs = raw.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == DEFAULT_PERPLEXITY_MODEL

    def test_perplexity_mini_model_mapping(self):
        raw = _raw_client([_choice("ok")])
        provider = LLMProvider(name="perplexity", client=raw)
        call_llm(provider, messages=[{"role": "user", "content": "hi"}],
                 model="gpt-4o-mini", operation="t")
        kwargs = raw.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == DEFAULT_PERPLEXITY_MODEL_MINI

    def test_perplexity_json_object_format_dropped(self):
        """Sonar rejects OpenAI-style json_object; the guard must not send it
        (prompts already demand JSON; parsing is reused unchanged)."""
        raw = _raw_client([_choice(json.dumps({"ok": True}))])
        provider = LLMProvider(name="perplexity", client=raw)
        out = call_llm_json(provider, messages=[{"role": "user", "content": "hi"}],
                            model="gpt-4o", operation="t")
        assert out == {"ok": True}
        kwargs = raw.chat.completions.create.call_args.kwargs
        assert "response_format" not in kwargs

    def test_perplexity_json_fences_stripped_by_existing_parser(self):
        raw = _raw_client([_choice("```json\n{\"x\": 1}\n```")])
        provider = LLMProvider(name="perplexity", client=raw)
        out = call_llm_json(provider, messages=[{"role": "user", "content": "hi"}])
        assert out == {"x": 1}

    def test_openai_provider_keeps_json_object_format(self):
        raw = _raw_client([_choice(json.dumps({"ok": 1}))])
        provider = LLMProvider(name="openai", client=raw)
        call_llm_json(provider, messages=[{"role": "user", "content": "hi"}],
                      model="gpt-4o")
        kwargs = raw.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["model"] == "gpt-4o"

    def test_local_provider_routes_with_override_model(self, monkeypatch):
        monkeypatch.setenv("HLEO_LLM_MODEL", "local-model-x")
        raw = _raw_client([_choice("local ok")])
        provider = LLMProvider(name="local", client=raw)
        out = call_llm(provider, messages=[{"role": "user", "content": "hi"}],
                       model="gpt-4o", operation="t")
        assert out == "local ok"
        kwargs = raw.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "local-model-x"


# ── One-way fallback Perplexity → OpenAI ─────────────────────────────────────

class TestProviderFallback:
    def test_quota_exhausted_falls_back_to_openai(self):
        pplx_raw = _raw_client([Exception("insufficient_quota: no credit")])
        oai_raw = _raw_client([_choice("openai rescued")])
        provider = LLMProvider(
            name="perplexity", client=pplx_raw,
            fallback=LLMProvider(name="openai", client=oai_raw),
        )
        out = call_llm(provider, messages=[{"role": "user", "content": "hi"}],
                       model="gpt-4o", operation="t")
        assert out == "openai rescued"
        # quota → no retry on primary (1 call), exactly 1 fallback call
        assert pplx_raw.chat.completions.create.call_count == 1
        assert oai_raw.chat.completions.create.call_count == 1
        # fallback used the OpenAI model, not the Perplexity one
        assert oai_raw.chat.completions.create.call_args.kwargs["model"] == "gpt-4o"

    def test_exhausted_primary_then_fallback_success(self):
        pplx_raw = _raw_client([Exception("server hiccup")] * 10)
        oai_raw = _raw_client([_choice("rescued")])
        provider = LLMProvider(
            name="perplexity", client=pplx_raw,
            fallback=LLMProvider(name="openai", client=oai_raw),
        )
        out = call_llm(provider, messages=[{"role": "user", "content": "hi"}],
                       operation="t")
        assert out == "rescued"
        assert pplx_raw.chat.completions.create.call_count == MAX_TOTAL_ATTEMPTS
        assert oai_raw.chat.completions.create.call_count == 1

    def test_no_loop_back_to_perplexity(self):
        """When the fallback also fails, the error propagates — Perplexity is
        NEVER retried (linear chain, no Perplexity ↔ OpenAI loop)."""
        pplx_raw = _raw_client([Exception("pplx down")] * 100)
        oai_raw = _raw_client([Exception("oai down")] * 100)
        provider = LLMProvider(
            name="perplexity", client=pplx_raw,
            fallback=LLMProvider(name="openai", client=oai_raw),
        )
        with pytest.raises(LLMCallError):
            call_llm(provider, messages=[{"role": "user", "content": "hi"}],
                     operation="t")
        assert pplx_raw.chat.completions.create.call_count == MAX_TOTAL_ATTEMPTS
        assert oai_raw.chat.completions.create.call_count == MAX_TOTAL_ATTEMPTS

    def test_fallback_quota_propagates(self):
        pplx_raw = _raw_client([Exception("rate limit")] * 10)
        oai_raw = _raw_client([Exception("insufficient_quota")])
        provider = LLMProvider(
            name="perplexity", client=pplx_raw,
            fallback=LLMProvider(name="openai", client=oai_raw),
        )
        with pytest.raises(QuotaExhaustedError):
            call_llm(provider, messages=[{"role": "user", "content": "hi"}],
                     operation="t")

    def test_json_fallback_works(self):
        pplx_raw = _raw_client([Exception("insufficient_quota")])
        oai_raw = _raw_client([_choice(json.dumps({"via": "openai"}))])
        provider = LLMProvider(
            name="perplexity", client=pplx_raw,
            fallback=LLMProvider(name="openai", client=oai_raw),
        )
        out = call_llm_json(provider, messages=[{"role": "user", "content": "hi"}],
                            model="gpt-4o", operation="t")
        assert out == {"via": "openai"}

    def test_openai_provider_has_no_fallback(self):
        raw = _raw_client([Exception("insufficient_quota")])
        provider = LLMProvider(name="openai", client=raw)
        with pytest.raises(QuotaExhaustedError):
            call_llm(provider, messages=[{"role": "user", "content": "hi"}],
                     operation="t")
        assert raw.chat.completions.create.call_count == 1


# ── Backward compatibility (raw SDK client) ──────────────────────────────────

class TestRawClientBackwardCompat:
    def test_raw_client_model_passthrough(self):
        raw = _raw_client([_choice("ok")])
        out = call_llm(raw, messages=[{"role": "user", "content": "hi"}],
                       model="gpt-4o", operation="t")
        assert out == "ok"
        kwargs = raw.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "gpt-4o"

    def test_raw_client_json_mode_keeps_format(self):
        raw = _raw_client([_choice(json.dumps({"a": 2}))])
        out = call_llm_json(raw, messages=[{"role": "user", "content": "hi"}],
                            model="gpt-4o-mini")
        assert out == {"a": 2}
        kwargs = raw.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["model"] == "gpt-4o-mini"


# ── Key hygiene ──────────────────────────────────────────────────────────────

class TestKeyHygiene:
    def test_no_hardcoded_key_in_module(self):
        src = inspect.getsource(lp)
        assert "pplx-" not in src
        assert "sk-" not in src

    def test_key_read_only_from_env(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_KEY", "sentinel-pplx-key-000")
        calls = _patch_openai_ctor(monkeypatch)
        p = build_provider()
        assert p.name == "perplexity"
        assert calls[0]["api_key"] == "sentinel-pplx-key-000"
        # the key must not leak into the provider's repr/logs surface
        assert "sentinel-pplx-key-000" not in repr(p)
