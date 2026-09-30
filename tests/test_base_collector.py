from types import SimpleNamespace

import pytest
import requests

from collectors.base_search import BASECollector
from core.search_page import SearchPage


class Response:
    status_code = 200
    headers = {}

    def __init__(self, payload=None, *, json_error=None):
        self.payload = payload
        self.json_error = json_error

    def json(self):
        if self.json_error:
            raise self.json_error
        return self.payload


def limits(monkeypatch):
    monkeypatch.setattr(
        "core.llm_limits.get_limits",
        lambda: SimpleNamespace(
            collector_timeout_s=1,
            collector_max_retries=0,
            backoff_base_s=0,
            backoff_max_s=0,
        ),
    )


def record(identifier="base-1", title="BASE study", **extra):
    return {
        "dcdocid": identifier,
        "dctitle": title,
        "dcdescription": "Study abstract",
        "dclink": "https://example.org/study",
        "dcdoi": "10.1000/base.1",
        "dcyear": "2024",
        "dccreator": ["A Author", "B Author"],
        "dclang": "eng",
        **extra,
    }


def test_base_maps_json_to_search_result_and_page(monkeypatch):
    limits(monkeypatch)
    monkeypatch.setenv("BASE_API_KEY", "test-key")
    payload = {"response": {"numFound": 1, "start": 0, "docs": [record()]}}
    calls = []

    def fake_get(_url, params=None, **_kwargs):
        calls.append(params)
        return Response(payload)

    monkeypatch.setattr("core.http_retry.http_get", fake_get)
    result = BASECollector().search_page("finasteride", limit=1)

    assert isinstance(result, SearchPage)
    item = result.items[0]
    assert item.title == "BASE study"
    assert item.abstract == "Study abstract"
    assert item.authors == ["A Author", "B Author"]
    assert item.year == 2024
    assert item.doi == "10.1000/base.1"
    assert item.url == "https://example.org/study"
    assert item.source == "BASE"
    assert item.source_id == "base-1"
    assert item.language == "eng"
    assert item.metadata["provenance"] == {"provider": "BASE"}
    assert result.total_available == 1
    assert result.has_more is False
    assert calls[0]["func"] == "PerformSearch"
    assert calls[0]["query"] == "finasteride"
    assert calls[0]["hits"] == 1
    assert calls[0]["offset"] == 0
    assert calls[0]["apikey"] == "test-key"


def test_base_offset_pagination_and_has_more(monkeypatch):
    limits(monkeypatch)
    monkeypatch.setenv("BASE_API_KEY", "test-key")
    responses = iter([
        Response({"response": {"numFound": 3, "start": 0, "docs": [record("base-1"), record("base-2")]}}),
        Response({"response": {"numFound": 3, "start": 2, "docs": [record("base-3")]}}),
    ])
    calls = []

    def fake_get(_url, params=None, **_kwargs):
        calls.append(params)
        return next(responses)

    monkeypatch.setattr("core.http_retry.http_get", fake_get)
    collector = BASECollector()
    first = collector.search_page("query", limit=3)
    second = collector.search_page("query", cursor=first.next_cursor, limit=3)

    assert first.has_more is True
    assert first.next_cursor == {"offset": 2, "collected": 2, "total": 3}
    assert second.has_more is False
    assert second.next_cursor is None
    assert calls[0]["offset"] == 0
    assert calls[1]["offset"] == 2
    assert [item.source_id for item in first.items + second.items] == ["base-1", "base-2", "base-3"]


def test_base_optional_metadata_and_doi_detection(monkeypatch):
    limits(monkeypatch)
    monkeypatch.setenv("BASE_API_KEY", "test-key")
    payload = {"response": {"numFound": "not-a-number", "docs": [{
        "identifier": "base-identifier",
        "dctitle": "No abstract",
        "dcidentifier": ["https://doi.org/10.5555/example"],
    }]}}
    monkeypatch.setattr("core.http_retry.http_get", lambda *args, **kwargs: Response(payload))

    result = BASECollector().search_page("query", limit=1)
    item = result.items[0]

    assert item.abstract == ""
    assert item.doi == "10.5555/example"
    assert item.url == "https://doi.org/10.5555/example"
    assert item.source_id == "base-identifier"
    assert item.language is None
    assert result.total_available is None
    assert result.has_more is False


def test_base_does_not_deduplicate_provider_results(monkeypatch):
    limits(monkeypatch)
    monkeypatch.setenv("BASE_API_KEY", "test-key")
    payload = {"response": {"numFound": 2, "docs": [record("one", "Same"), record("two", "Same")]}}
    monkeypatch.setattr("core.http_retry.http_get", lambda *args, **kwargs: Response(payload))

    result = BASECollector().search_page("query", limit=2)

    assert len(result.items) == 2
    assert [item.source_id for item in result.items] == ["one", "two"]


def test_base_api_error_is_reported(monkeypatch):
    limits(monkeypatch)
    monkeypatch.setenv("BASE_API_KEY", "test-key")
    monkeypatch.setattr("core.http_retry.http_get", lambda *args, **kwargs: Response({"error": "invalid key"}))

    with pytest.raises(RuntimeError, match="BASE API error: invalid key"):
        BASECollector().search_page("query", limit=1)


def test_base_malformed_json_is_reported(monkeypatch):
    limits(monkeypatch)
    monkeypatch.setenv("BASE_API_KEY", "test-key")
    monkeypatch.setattr("core.http_retry.http_get", lambda *args, **kwargs: Response(json_error=ValueError("bad json")))

    with pytest.raises(RuntimeError, match="malformed JSON"):
        BASECollector().search_page("query", limit=1)


def test_base_http_error_is_not_hidden(monkeypatch):
    limits(monkeypatch)
    monkeypatch.setenv("BASE_API_KEY", "test-key")
    error = requests.HTTPError("HTTP 503")
    monkeypatch.setattr("core.http_retry.http_get", lambda *args, **kwargs: (_ for _ in ()).throw(error))

    with pytest.raises(requests.HTTPError, match="HTTP 503"):
        BASECollector().search_page("query", limit=1)


def test_base_requires_environment_key(monkeypatch):
    limits(monkeypatch)
    monkeypatch.delenv("BASE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="BASE_API_KEY is not configured"):
        BASECollector().search_page("query", limit=1)
