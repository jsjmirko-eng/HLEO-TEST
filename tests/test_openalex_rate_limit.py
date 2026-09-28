from types import SimpleNamespace

import pytest
import requests

from collectors.openalex import OpenAlexCollector
from core.search_page import SearchPage


class Response:
    def __init__(self, status_code, payload=None, headers=None):
        self.status_code = status_code
        self.payload = payload or {}
        self.headers = headers or {}
        self.url = "https://api.openalex.org/works"

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


def _limits(monkeypatch, retries=2):
    monkeypatch.setattr(
        "core.llm_limits.get_limits",
        lambda: SimpleNamespace(
            collector_timeout_s=1,
            collector_max_retries=retries,
            backoff_base_s=1.0,
            backoff_max_s=10.0,
        ),
    )


def _payload(title="Cancer study", total=2, work_id="W1"):
    return {
        "meta": {"count": total},
        "results": [{
            "id": f"https://openalex.org/{work_id}",
            "title": title,
            "publication_year": 2024,
            "doi": "https://doi.org/10.1000/openalex",
            "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/123"},
            "abstract_inverted_index": {"study": [0], "abstract": [1]},
            "authorships": [{"author": {"display_name": "A Author"}}],
            "language": {"id": "en"},
            "primary_location": {"source": {"display_name": "Journal"}},
            "locations": [],
            "open_access": {"is_oa": False},
        }],
    }


def test_openalex_retries_429_using_retry_after_and_maps_result(monkeypatch):
    _limits(monkeypatch)
    responses = iter([
        Response(429, headers={"Retry-After": "3"}),
        Response(200, _payload()),
    ])
    calls = []
    sleeps = []

    def fake_get(url, params=None, **kwargs):
        calls.append((url, params, kwargs))
        return next(responses)

    monkeypatch.setattr("core.http_retry.requests.get", fake_get)
    monkeypatch.setattr("core.http_retry.time.sleep", sleeps.append)
    result = OpenAlexCollector().search_page("cancer", limit=1)

    assert isinstance(result, SearchPage)
    assert len(result.items) == 1
    assert result.items[0].title == "Cancer study"
    assert result.items[0].source == "OpenAlex"
    assert len(calls) == 2
    assert sleeps == [3.0]
    assert calls[0][1]["search"] == "cancer"


def test_openalex_429_retry_budget_is_bounded(monkeypatch):
    _limits(monkeypatch, retries=2)
    calls = []
    sleeps = []

    def fake_get(*_args, **_kwargs):
        calls.append(True)
        return Response(429, headers={"Retry-After": "1"})

    monkeypatch.setattr("core.http_retry.requests.get", fake_get)
    monkeypatch.setattr("core.http_retry.time.sleep", sleeps.append)

    with pytest.raises(requests.HTTPError):
        OpenAlexCollector().search_page("diabetes", limit=1)

    assert len(calls) == 3
    assert sleeps == [1.0, 1.0]


def test_openalex_paginates_without_duplicate_results(monkeypatch):
    _limits(monkeypatch, retries=0)
    payloads = iter([_payload("Page one", total=2, work_id="W1"), _payload("Page two", total=2, work_id="W2")])
    calls = []

    def fake_get(_url, params=None, **_kwargs):
        calls.append(params)
        return Response(200, next(payloads))

    monkeypatch.setattr("core.http_retry.requests.get", fake_get)
    collector = OpenAlexCollector()
    first = collector.search_page("finasteride", limit=2)
    second = collector.search_page("finasteride", cursor=first.next_cursor, limit=2)

    assert first.has_more is True
    assert first.next_cursor == {"page": 2, "collected": 1, "total": 2}
    assert second.has_more is False
    assert [item.title for item in first.items + second.items] == ["Page one", "Page two"]
    assert calls[0]["page"] == 1
    assert calls[1]["page"] == 2
