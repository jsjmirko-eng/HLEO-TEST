from types import SimpleNamespace

from core.search_page import SearchPage, collect_search_pages
from collectors.europepmc import EuropePMCCollector


class _Response:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 200

    def json(self):
        return self.payload


def test_collect_search_pages_keeps_cursor_opaque():
    cursors = []

    class Collector:
        def search_page(self, query, cursor=None, limit=None):
            cursors.append(cursor)
            if cursor is None:
                return SearchPage([1], {"provider_token": "opaque"}, True, 2)
            return SearchPage([2], None, False, 2)

    assert collect_search_pages(Collector(), "query") == [1, 2]
    assert cursors == [None, {"provider_token": "opaque"}]


def test_europepmc_search_page_uses_next_cursor_and_deduplicates(monkeypatch):
    limits = SimpleNamespace(
        collector_timeout_s=1,
        collector_max_retries=0,
        backoff_base_s=0,
        backoff_max_s=0,
    )
    monkeypatch.setattr("collectors.europepmc._lim", lambda: limits)
    calls = []
    payloads = [
        {
            "hitCount": 3,
            "nextCursorMark": "cursor-1",
            "resultList": {"result": [
                {"id": "one", "title": "One", "pubYear": "2024"},
                {"id": "two", "title": "Two", "pubYear": "2024"},
            ]},
        },
        {
            "hitCount": 3,
            "nextCursorMark": "cursor-2",
            "resultList": {"result": [
                {"id": "two", "title": "Two", "pubYear": "2024"},
                {"id": "three", "title": "Three", "pubYear": "2024"},
            ]},
        },
        {
            "hitCount": 3,
            "resultList": {"result": []},
        },
    ]

    def fake_http_get(url, params=None, **_kwargs):
        calls.append(params)
        return _Response(payloads[len(calls) - 1])

    monkeypatch.setattr("core.http_retry.http_get", fake_http_get)
    collector = EuropePMCCollector()
    first = collector.search_page("query")
    second = collector.search_page("query", cursor=first.next_cursor)

    assert [item.metadata["id"] for item in first.items] == ["one", "two"]
    assert [item.metadata["id"] for item in second.items] == ["three"]
    assert calls[0].get("cursorMark") is None
    assert calls[1]["cursorMark"] == "cursor-1"
    assert second.next_cursor["cursor_mark"] == "cursor-2"


def test_collect_search_pages_falls_back_for_legacy_collectors():
    class LegacyCollector:
        def search(self, query, limit=None):
            return [query, limit]

    assert collect_search_pages(LegacyCollector(), "query", limit=3) == ["query", 3]
