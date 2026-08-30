"""Tests for the RWE robust translation chain and openFDA query sanitisation.

All offline: fake LLM clients + mocked requests; no network, no real keys.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import core.llm_guard
import core.rwe.openfda_collector as fda
from core.rwe.openfda_collector import (
    STATUS_NO_RESULTS,
    STATUS_OK,
    STATUS_UNSUPPORTED_QUERY,
    OpenFDACollector,
    sanitize_fda_term,
)
from core.rwe.translation import _validate, translate_for_rwe


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """llm_guard retries with real backoff sleeps — neutralise them so the
    failing-LLM chain tests run in milliseconds."""
    monkeypatch.setattr(core.llm_guard.time, "sleep", lambda *_a, **_k: None)

IT_QUERY = ("dolore articolare e rigidità dopo l'uso di isotretinoina, "
            "esperienze dei pazienti")
EN_QUERY = "isotretinoin joint pain stiffness patient experiences"


class _FakeCompletions:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        msg = SimpleNamespace(content=out)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class _FakeClient:
    def __init__(self, outputs):
        self.chat = SimpleNamespace(completions=_FakeCompletions(outputs))


# ── Output validation ────────────────────────────────────────────────────────

def test_validate_accepts_plain_english():
    assert _validate(EN_QUERY, IT_QUERY) == EN_QUERY


def test_validate_rejects_echo_empty_and_rambling():
    assert _validate(IT_QUERY, IT_QUERY) == ""
    assert _validate("", IT_QUERY) == ""
    assert _validate("   ", IT_QUERY) == ""
    assert _validate("word " * 200, IT_QUERY) == ""
    assert _validate("12345", IT_QUERY) == ""


def test_validate_strips_quotes():
    assert _validate(f'"{EN_QUERY}"', IT_QUERY) == EN_QUERY


# ── Translation chain ────────────────────────────────────────────────────────

def test_llm_plaintext_success():
    client = _FakeClient([EN_QUERY])
    res = translate_for_rwe(IT_QUERY, "it", client=client)
    assert res.method == "llm"
    assert res.english_query == EN_QUERY


def test_llm_retry_after_first_failure():
    # call_llm retries internally up to 5 attempts per chain step: the full
    # prompt must exhaust them (5 failures) before the minimal prompt is tried.
    client = _FakeClient([RuntimeError("json_validate_failed")] * 5 + [EN_QUERY])
    res = translate_for_rwe(IT_QUERY, "it", client=client)
    assert res.method == "llm_retry"
    assert res.english_query == EN_QUERY


def test_llm_json_answer_tolerated():
    client = _FakeClient(['{"query_en": "isotretinoin joint pain stiffness"}'])
    res = translate_for_rwe(IT_QUERY, "it", client=client)
    assert res.english_query == "isotretinoin joint pain stiffness"


def test_deterministic_fallback_from_entities(monkeypatch):
    client = _FakeClient([RuntimeError("x")] * 10)  # both chain steps exhausted

    class _Rec:
        entities = [("drug", "isotretinoin", 0.9),
                    ("symptom", "joint pain", 0.8),
                    ("symptom", "stiffness", 0.8)]

    monkeypatch.setattr("core.vocab.entities.recognize",
                        lambda *a, **k: _Rec())
    monkeypatch.setattr("core.vocab.resolver.build_resolver_from_env",
                        lambda: object())
    res = translate_for_rwe(IT_QUERY, "it", client=client)
    assert res.method == "deterministic"
    assert res.english_query == "isotretinoin joint pain stiffness"


def test_total_failure_returns_original_marked_none(monkeypatch):
    client = _FakeClient([RuntimeError("x")] * 10)
    monkeypatch.setattr("core.vocab.resolver.build_resolver_from_env",
                        lambda: None)
    res = translate_for_rwe(IT_QUERY, "it", client=client)
    assert res.method == "none"
    assert res.english_query == IT_QUERY
    assert res.error == "translation_unavailable"


def test_english_query_not_translated():
    res = translate_for_rwe("finasteride side effects", "en", client=None)
    assert res.method == "none"
    assert res.english_query == "finasteride side effects"


# ── openFDA sanitisation ─────────────────────────────────────────────────────

def test_sanitize_apostrophes_and_unicode():
    q = "dolore articolare e rigidità dopo l’uso di isotretinoin, esperienze"
    out = sanitize_fda_term(q)
    assert "'" not in out and "’" not in out
    assert "rigidita" in out          # NFKD strips the accent
    assert "l uso" in out.replace("  ", " ") or "l uso" in out


def test_sanitize_brackets_and_cap():
    out = sanitize_fda_term(
        "isotretinoin 10 MG Oral Capsule [Myorisan] extra tokens here now")
    assert "[" not in out and "]" not in out
    assert len(out.split()) <= 8


def test_sanitize_empty():
    assert sanitize_fda_term("") == ""
    assert sanitize_fda_term("[]{}") == ""


# ── openFDA HTTP 400 handling ────────────────────────────────────────────────

class _Resp:
    def __init__(self, code, payload=None):
        self.status_code = code
        self._payload = payload or {}
        self.headers = {"content-type": "application/json"}

    def json(self):
        return self._payload


def _faers_payload(n=1):
    return {"results": [{
        "safetyreportid": f"R{i}",
        "patient": {
            "drug": [{"openfda": {"generic_name": ["ISOTRETINOIN"]},
                      "medicinalproduct": "ACCUTANE"}],
            "reaction": [{"reactionmeddrapt": "ARTHRALGIA"}],
        },
    } for i in range(n)]}


def test_openfda_400_retry_then_ok(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params["search"])
        if len(calls) == 1:
            return _Resp(400, {"error": {"message": "Search not supported"}})
        return _Resp(200, _faers_payload(2))

    monkeypatch.setattr(fda.requests, "get", fake_get)
    items, status, reason = OpenFDACollector().search_with_status(
        IT_QUERY, limit=5)
    assert status == STATUS_OK
    assert len(items) == 2
    assert len(calls) == 2                     # one retry happened
    assert "dopo l uso" in calls[0] or "l uso" in calls[0]


def test_openfda_double_400_unsupported_query(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        return _Resp(400, {"error": {"message": "Search not supported"}})

    monkeypatch.setattr(fda.requests, "get", fake_get)
    items, status, reason = OpenFDACollector().search_with_status(
        IT_QUERY, limit=5)
    assert status == STATUS_UNSUPPORTED_QUERY
    assert items == []
    assert "400" in reason


def test_openfda_404_still_no_results(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        return _Resp(404, {})

    monkeypatch.setattr(fda.requests, "get", fake_get)
    items, status, _reason = OpenFDACollector().search_with_status(
        "isotretinoin", limit=5)
    assert status == STATUS_NO_RESULTS
    assert items == []
