from types import SimpleNamespace

from collectors.hal import HALCollector
from core.search_page import SearchPage


class Response:
    status_code = 200
    headers = {}

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def _patch_limits(monkeypatch):
    monkeypatch.setattr(
        "core.llm_limits.get_limits",
        lambda: SimpleNamespace(
            collector_timeout_s=1,
            collector_max_retries=0,
            backoff_base_s=0,
            backoff_max_s=0,
        ),
    )


def _record(identifier="123", title="HAL title"):
    return {
        "docid": identifier,
        "title_s": [title],
        "abstract_s": ["<p>Abstract <b>text</b>.</p>"],
        "authFullName_s": ["A Author", "B Author"],
        "doiId_s": ["10.1000/hal"],
        "pmidId_s": ["42"],
        "pmcidId_s": ["PMC42"],
        "uri_s": "https://hal.science/hal-123",
        "fileMain_s": "https://hal.science/hal-123/document",
        "language_s": ["fr"],
        "producedDateY_i": 2021,
        "journalTitle_s": ["HAL Journal"],
        "openAccess_bool": True,
        "docType_s": ["ART"],
    }


def test_maps_search_page_metadata_and_preserves_query(monkeypatch):
    _patch_limits(monkeypatch)
    calls = []
    payload = {
        "response": {"numFound": 1, "docs": [_record()]},
        "nextCursorMark": "same-end",
    }

    def fake_get(url, params=None, **kwargs):
        calls.append((url, params, kwargs))
        return Response(payload)

    monkeypatch.setattr("core.http_retry.http_get", fake_get)
    result = HALCollector().search_page("finasteride hair regrowth", limit=1)

    assert isinstance(result, SearchPage)
    assert result.next_cursor is None
    assert result.has_more is False
    assert result.total_available == 1
    item = result.items[0]
    assert item.title == "HAL title"
    assert item.original_title == "HAL title"
    assert item.abstract == "Abstract text ."
    assert item.authors == ["A Author", "B Author"]
    assert item.year == 2021
    assert item.doi == "10.1000/hal"
    assert item.pmid == "42"
    assert item.pmcid == "PMC42"
    assert item.url.endswith("hal-123")
    assert item.full_text_url.endswith("document")
    assert item.language == "fr"
    assert item.source == "HAL"
    assert item.source_id == "123"
    assert item.full_text_available is True
    assert item.metadata["journal"] == "HAL Journal"
    assert calls[0][1]["q"] == "finasteride hair regrowth"
    assert calls[0][1]["cursorMark"] == "*"
    assert calls[0][1]["sort"] == "docid asc"


def test_keeps_hal_cursor_inside_search_page(monkeypatch):
    _patch_limits(monkeypatch)
    payloads = iter([
        {
            "response": {"numFound": 2, "docs": [_record()]},
            "nextCursorMark": "opaque-2",
        },
        {
            "response": {"numFound": 2, "docs": [_record("456", "Second")]},
            "nextCursorMark": "opaque-2",
        },
    ])
    calls = []

    def fake_get(_url, params=None, **_kwargs):
        calls.append(params)
        return Response(next(payloads))

    monkeypatch.setattr("core.http_retry.http_get", fake_get)
    collector = HALCollector()
    first = collector.search_page("query", limit=2)
    second = collector.search_page("query", cursor=first.next_cursor, limit=2)

    assert first.has_more is True
    assert first.next_cursor == {"hal_cursor": "opaque-2", "collected": 1, "total": 2}
    assert second.has_more is False
    assert calls[0]["cursorMark"] == "*"
    assert calls[1]["cursorMark"] == "opaque-2"
