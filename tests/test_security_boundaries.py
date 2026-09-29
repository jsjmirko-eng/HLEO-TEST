import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from api.main import (
    CardSynthesisRequest,
    ChatRequest,
    CompareRequest,
    RWEExtractRequest,
    RWEBatchExtractRequest,
    SynthesisRequest,
    TranslateRequest,
)
from core.input_limits import BodySizeLimitMiddleware, max_body_bytes
from core.logging_utils import query_fingerprint, redact_text


TEMPLATE = Path(__file__).parents[1] / "templates" / "index.html"


def test_admin_token_uses_session_storage_only():
    source = TEMPLATE.read_text()
    auth_block = source[source.index("const _ADMIN_TOKEN_KEY"):source.index("async function checkAdminAvailability")]

    assert "sessionStorage.getItem(_ADMIN_TOKEN_KEY)" in auth_block
    assert "sessionStorage.setItem(_ADMIN_TOKEN_KEY, token)" in auth_block
    assert "sessionStorage.removeItem(_ADMIN_TOKEN_KEY)" in auth_block
    assert "localStorage" not in auth_block


def test_rwe_batch_and_fields_are_bounded():
    with pytest.raises(ValidationError):
        RWEExtractRequest(text="x" * 20_001)
    with pytest.raises(ValidationError):
        RWEBatchExtractRequest(items=[{}] * 51)


def test_costly_payloads_are_bounded():
    with pytest.raises(ValidationError):
        ChatRequest(message="x" * 20_001)
    with pytest.raises(ValidationError):
        CompareRequest(query="x" * 501)
    with pytest.raises(ValidationError):
        SynthesisRequest(query="x" * 501)
    with pytest.raises(ValidationError):
        CardSynthesisRequest(query="x" * 501, title="title")
    with pytest.raises(ValidationError):
        TranslateRequest(text="x" * 20_001)


def test_body_limit_rejects_content_length(monkeypatch):
    monkeypatch.setenv("HLEO_MAX_BODY_BYTES", "5")
    sent = []

    async def endpoint(scope, receive, send):
        raise AssertionError("oversized body reached endpoint")

    async def send(message):
        sent.append(message)

    asyncio.run(
        BodySizeLimitMiddleware(endpoint)(
            {
                "type": "http",
                "method": "POST",
                "headers": [(b"content-length", b"6")],
            },
            lambda: None,
            send,
        )
    )

    assert sent[0]["status"] == 413
    assert max_body_bytes() == 5


def test_sensitive_log_values_are_redacted():
    value = "api_key=secret-value Authorization: Bearer admin-token"
    safe = redact_text(value)

    assert "secret-value" not in safe
    assert "admin-token" not in safe
    assert query_fingerprint("private patient query") != "private patient query"
