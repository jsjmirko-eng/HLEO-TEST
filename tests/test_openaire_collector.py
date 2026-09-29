from types import SimpleNamespace

from collectors.openaire import OpenAIRECollector
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


def _record(identifier="record-1", title="OpenAIRE title"):
    return {
        "id": identifier,
        "mainTitle": title,
        "descriptions": ["Abstract text"],
        "publicationDate": "2021-05-06",
        "authors": [{"fullName": "A Author"}],
        "language": {"code": "eng", "label": "English"},
        "pids": [
            {"scheme": "doi", "value": "10.1000/openaire"},
            {"scheme": "pmid", "value": "42"},
            {"scheme": "pmc", "value": "PMC42"},
        ],
        "bestAccessRight": {"label": "OPEN"},
        "instances": [{"urls": ["https://repository.example/article"]}],
        "container": {"name": "Journal"},
        "collectedFrom": [{"value": "Crossref"}],
    }


def test_maps_search_page_metadata_and_preserves_query(monkeypatch):
    _patch_limits(monkeypatch)
    calls = []
    payload = {"header": {"numFound": 1}, "results": [_record()]}

    def fake_get(url, params=None, **kwargs):
        calls.append((url, params, kwargs))
        return Response(payload)

    monkeypatch.setattr("core.http_retry.http_get", fake_get)
    result = OpenAIRECollector().search_page("finasteride hair regrowth", limit=1)

    assert isinstance(result, SearchPage)
    assert result.next_cursor is None
    assert result.has_more is False
    assert result.total_available == 1
    item = result.items[0]
    assert item.title == "OpenAIRE title"
    assert item.original_title == "OpenAIRE title"
    assert item.abstract == "Abstract text"
    assert item.authors == ["A Author"]
    assert item.year == 2021
    assert item.doi == "10.1000/openaire"
    assert item.pmid == "42"
    assert item.pmcid == "PMC42"
    assert item.url.endswith("article")
    assert item.full_text_url.endswith("article")
    assert item.language == "en"
    assert item.source == "OpenAIRE"
    assert item.source_id == "record-1"
    assert item.full_text_available is True
    assert item.metadata["journal"] == "Journal"
    assert calls[0][1]["search"] == "finasteride hair regrowth"
    assert calls[0][1]["type"] == "publication"


def test_keeps_openaire_cursor_inside_search_page(monkeypatch):
    _patch_limits(monkeypatch)
    payloads = iter([
        {"header": {"numFound": 2, "nextCursor": "opaque-2"}, "results": [_record()]},
        {"header": {"numFound": 2}, "results": [_record("record-2", "Second")]},
    ])
    calls = []

    def fake_get(_url, params=None, **_kwargs):
        calls.append(params)
        return Response(next(payloads))

    monkeypatch.setattr("core.http_retry.http_get", fake_get)
    collector = OpenAIRECollector()
    first = collector.search_page("query", limit=2)
    second = collector.search_page("query", cursor=first.next_cursor, limit=2)

    assert first.has_more is True
    assert first.next_cursor == {"page": 2, "collected": 1, "total": 2, "openaire_cursor": "opaque-2"}
    assert second.has_more is False
    assert calls[0]["page"] == 1
    assert "cursor" not in calls[0]
    assert calls[1]["cursor"] == "opaque-2"
    assert "page" not in calls[1]
