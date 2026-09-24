"""
Tests for core/llm_provider.py + provider routing in core/llm_guard.py.

Provider-agnostic architecture:
  - Provider is a free-form metadata label, never used as a runtime switch.
  - Runtime uses exclusively: api_key, base_url, model from configured settings.
  - No hardcoded model mapping based on provider name.
  - No special routing based on provider name.

All offline: fake SDK clients, no network, no real keys.
"""
from __future__ import annotations

import inspect
import json
from unittest.mock import MagicMock

import pytest

import core.llm_provider as lp
from core.llm_provider import (
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
from core.database import SessionLocal, engine, Base
from core.models import LLMConfig, LLMProviderSlot


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


def _save_admin_slot(*, name="OpenAI", model="admin-model", base_url="https://api.openai.com/v1"):
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        db.query(LLMConfig).delete()
        db.query(LLMProviderSlot).delete()
        db.add(LLMProviderSlot(
            slot_index=1,
            enabled=True,
            name=name,
            protocol="OpenAI-compatible",
            api_key_encrypted=lp.encrypt_secret("admin-key"),
            base_url=base_url,
            model=model,
            timeout_s=60.0,
            max_retries=0,
        ))
        db.commit()



@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch):
    for var in ("OPENAI_API_KEY", "HLEO_LLM_PROVIDER",
                "OPENAI_BASE_URL", "HLEO_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("core.llm_guard.time.sleep", lambda d: None)


# ── Provider selection ────────────────────────────────────────────────────────

class TestProviderSelection:
    def test_openai_with_key_only(self, monkeypatch):
        """When only api_key is set, uses OpenAI official endpoint."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        calls = _patch_openai_ctor(monkeypatch)
        p = build_provider()
        assert p is not None
        assert p.name == "openai"
        assert p.fallback is None
        assert calls == [{"api_key": "test-openai-key"}]

    def test_custom_base_url_uses_generic_runtime_path(self, monkeypatch):
        """base_url presence selects generic OpenAI-compatible path."""
        monkeypatch.setenv("HLEO_LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
        calls = _patch_openai_ctor(monkeypatch)
        p = build_provider()
        assert p is not None and p.name == "openai"
        assert calls == [{"api_key": "test-openai-key", "base_url": "https://api.example.com/v1"}]

    def test_provider_name_is_freeform_metadata(self, monkeypatch):
        """Provider names are free-form labels - no special routing."""
        for provider_name, base_url in (
            ("openrouter", "https://openrouter.example.com/v1"),
            ("groq", "https://groq.example.com/v1"),
            ("my-custom-provider", "https://custom.example.com/v1"),
        ):
            monkeypatch.setenv("HLEO_LLM_PROVIDER", provider_name)
            monkeypatch.setenv("OPENAI_API_KEY", "test-key")
            monkeypatch.setenv("OPENAI_BASE_URL", base_url)
            calls = _patch_openai_ctor(monkeypatch)
            p = build_provider()
            assert p is not None and p.name == provider_name
            assert calls == [{"api_key": "test-key", "base_url": base_url}]

    def test_admin_slot_is_runtime_provider(self, monkeypatch):
        _save_admin_slot(name="Admin OpenAI", model="admin-model")
        calls = _patch_openai_ctor(monkeypatch)

        provider = build_provider()

        assert provider is not None
        assert provider.name == "Admin OpenAI"
        assert provider.model_override == "admin-model"
        assert provider.use_chain is True
        assert calls == [{
            "api_key": "admin-key",
            "base_url": "https://api.openai.com/v1",
            "timeout": 60.0,
        }]

    def test_local_provider_via_base_url(self, monkeypatch):
        """Local provider uses base_url for the endpoint."""
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
        calls = _patch_openai_ctor(monkeypatch)
        p = build_provider()
        assert p.name == "openai-compatible"
        assert p.fallback is None
        # api_key defaults to "generic" when not provided
        assert calls == [{"api_key": "generic", "base_url": "http://localhost:1234/v1"}]

    def test_nothing_configured_returns_none(self, monkeypatch):
        _patch_openai_ctor(monkeypatch)
        assert build_provider() is None
        assert not llm_available()

    def test_persisted_admin_llm_config_overrides_env(self, monkeypatch):
        """DB config takes precedence over environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        monkeypatch.setenv("HLEO_LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example.com/v1")
        Base.metadata.create_all(bind=engine)
        with SessionLocal() as db:
            db.query(LLMConfig).delete()
            db.add(LLMConfig(
                provider="my-provider",
                api_key="db-key",
                base_url="https://db.example.com/v1",
                model="gpt-4o-mini",
            ))
            db.commit()
        calls = _patch_openai_ctor(monkeypatch)
        p = build_provider()
        assert p is not None and p.name == "my-provider"
        assert calls == [{"api_key": "db-key", "base_url": "https://db.example.com/v1"}]


class TestAdminProviderWiring:
    def test_admin_slot_is_loaded_after_orchestrator_init(self, monkeypatch):
        from core.orchestrator import QueryOrchestrator
        QueryOrchestrator._cache.clear()
        orchestrator = QueryOrchestrator()
        assert orchestrator._client is None

        _save_admin_slot(model="late-admin-model")
        raw = _raw_client([_choice(json.dumps({"lang": "en", "query_en": "hair loss"}))])
        monkeypatch.setattr("openai.OpenAI", lambda **kwargs: raw)

        result = orchestrator.process("late provider hair loss")

        assert result.detected_language == "en"
        assert raw.chat.completions.create.call_args.kwargs["model"] == "late-admin-model"

    def test_admin_slot_reaches_scientific_orchestrator(self, monkeypatch):
        _save_admin_slot(model="scientific-admin-model")
        raw = _raw_client([_choice(json.dumps({"lang": "en", "query_en": "hair loss"}))])
        monkeypatch.setattr("openai.OpenAI", lambda **kwargs: raw)

        from core.orchestrator import QueryOrchestrator
        QueryOrchestrator._cache.clear()
        result = QueryOrchestrator().process("hair loss")

        assert result.detected_language == "en"
        assert raw.chat.completions.create.call_args.kwargs["model"] == "scientific-admin-model"

    def test_admin_slot_reaches_rwe_intent(self, monkeypatch):
        _save_admin_slot(model="rwe-admin-model")
        raw = _raw_client([_choice(json.dumps({
            "interventions": ["dutasteride"],
            "outcomes": ["hair loss"],
            "conditions": [],
            "synonyms": {},
            "relation_type": "side_effect",
        }))])
        monkeypatch.setattr("openai.OpenAI", lambda **kwargs: raw)

        from core.rwe.intent import extract_intent_llm
        intent = extract_intent_llm("dutasteride hair loss", "dutasteride hair loss")

        assert intent is not None
        assert intent.interventions == ["dutasteride"]
        assert raw.chat.completions.create.call_args.kwargs["model"] == "rwe-admin-model"


# ── Model resolution ──────────────────────────────────────────────────────────

class TestModelResolution:
    def test_db_config_model_takes_precedence(self, monkeypatch):
        """When DB has a model, it overrides the requested model."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        Base.metadata.create_all(bind=engine)
        with SessionLocal() as db:
            db.query(LLMConfig).delete()
            db.add(LLMConfig(
                provider="openai",
                api_key="test-key",
                model="gpt-4o-mini",
            ))
            db.commit()
        resolved = resolve_model("openai", "gpt-4o")
        assert resolved == "gpt-4o-mini"

    def test_requested_model_used_when_no_db_config(self):
        """Without DB config, uses the requested model."""
        resolved = resolve_model("openai", "gpt-4o")
        assert resolved == "gpt-4o"

    def test_provider_name_not_used_for_model_mapping(self, monkeypatch):
        """Provider name is metadata - not used for model selection."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        Base.metadata.create_all(bind=engine)
        with SessionLocal() as db:
            db.query(LLMConfig).delete()
            db.add(LLMConfig(
                provider="any-name",
                api_key="test-key",
                model="configured-model",
            ))
            db.commit()
        # Provider name doesn't affect model resolution
        assert resolve_model("any-name", "gpt-4o") == "configured-model"
        assert resolve_model("another-name", "gpt-4o") == "configured-model"


# ── Guard provider routing ────────────────────────────────────────────────────

class TestGuardProviderRouting:
    def test_freeform_provider_uses_configured_model(self, monkeypatch):
        """Freeform provider names work with configured model."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        Base.metadata.create_all(bind=engine)
        with SessionLocal() as db:
            db.query(LLMConfig).delete()
            db.add(LLMConfig(
                provider="my-custom-ai",
                api_key="test-key",
                base_url="https://api.example.com/v1",
                model="custom-model-v1",
            ))
            db.commit()
        p = build_provider()
        raw = _raw_client([_choice("ok")])
        provider = LLMProvider(name=p.name, client=raw)
        call_llm(provider, messages=[{"role": "user", "content": "hi"}],
                 model="gpt-4o", operation="t")
        kwargs = raw.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "custom-model-v1"

    def test_json_mode_applies_json_object_format(self):
        """JSON mode applies response_format for all providers."""
        raw = _raw_client([_choice(json.dumps({"ok": True}))])
        provider = LLMProvider(name="openai", client=raw)
        out = call_llm_json(provider, messages=[{"role": "user", "content": "hi"}],
                            model="gpt-4o", operation="t")
        assert out == {"ok": True}
        kwargs = raw.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_freeform_provider_json_mode(self):
        """JSON mode works for any provider name."""
        raw = _raw_client([_choice(json.dumps({"x": 1}))])
        provider = LLMProvider(name="openrouter", client=raw)
        out = call_llm_json(provider, messages=[{"role": "user", "content": "hi"}])
        assert out == {"x": 1}


# ── Fallback chain ──────────────────────────────────────────────────────────

class TestProviderFallback:
    def test_fallback_chain_uses_provider_names(self):
        """Fallback chain preserves provider names from LLMProvider."""
        primary_raw = _raw_client([Exception("server error")] * 10)
        fallback_raw = _raw_client([_choice("rescued")])
        provider = LLMProvider(
            name="my-primary", client=primary_raw,
            fallback=LLMProvider(name="my-fallback", client=fallback_raw),
        )
        out = call_llm(provider, messages=[{"role": "user", "content": "hi"}],
                       operation="t")
        assert out == "rescued"
        assert primary_raw.chat.completions.create.call_count == MAX_TOTAL_ATTEMPTS
        assert fallback_raw.chat.completions.create.call_count == 1

    def test_quota_exhausted_falls_back(self):
        """Quota exhaustion triggers fallback."""
        primary_raw = _raw_client([Exception("insufficient_quota: no credit")])
        fallback_raw = _raw_client([_choice("fallback ok")])
        provider = LLMProvider(
            name="primary", client=primary_raw,
            fallback=LLMProvider(name="fallback", client=fallback_raw),
        )
        out = call_llm(provider, messages=[{"role": "user", "content": "hi"}],
                       model="gpt-4o", operation="t")
        assert out == "fallback ok"
        assert primary_raw.chat.completions.create.call_count == 1
        assert fallback_raw.chat.completions.create.call_count == 1

    def test_fallback_quota_propagates(self):
        """When fallback also hits quota, propagates QuotaExhaustedError."""
        primary_raw = _raw_client([Exception("rate limit")] * 10)
        fallback_raw = _raw_client([Exception("insufficient_quota")])
        provider = LLMProvider(
            name="primary", client=primary_raw,
            fallback=LLMProvider(name="fallback", client=fallback_raw),
        )
        with pytest.raises(QuotaExhaustedError):
            call_llm(provider, messages=[{"role": "user", "content": "hi"}],
                     operation="t")

    def test_json_fallback_works(self):
        """JSON output works when fallback is used."""
        primary_raw = _raw_client([Exception("error")])
        fallback_raw = _raw_client([_choice(json.dumps({"via": "fallback"}))])
        provider = LLMProvider(
            name="primary", client=primary_raw,
            fallback=LLMProvider(name="fallback", client=fallback_raw),
        )
        out = call_llm_json(provider, messages=[{"role": "user", "content": "hi"}],
                            model="gpt-4o", operation="t")
        assert out == {"via": "fallback"}

    def test_no_fallback_raises_on_quota(self):
        """Provider without fallback raises QuotaExhaustedError immediately."""
        raw = _raw_client([Exception("insufficient_quota")])
        provider = LLMProvider(name="standalone", client=raw)
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
        assert "sk-" not in src
