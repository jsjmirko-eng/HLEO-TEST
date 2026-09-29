from types import SimpleNamespace

from collectors.cinii import CiNiiResearchCollector
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


def _record(identifier="https://cir.nii.ac.jp/crid/123", title="CiNii title"):
    return {
        "@id": identifier,
        "@type": "item",
        "title": title,
        "link": {"@id": identifier},
        "dc:creator": ["A Author", "B Author"],
        "dc:publisher": "Publisher",
        "dc:language": "en",
        "prism:publicationName": "CiNii Journal",
        "prism:publicationDate": "2022-03-10",
        "description": "<jats:p>Abstract text.</jats:p>",
        "dc:identifier": [
            {"@type": "cir:DOI", "@value": "10.1000/cinii"},
        ],
        "dc:source": [
            {"@id": "https://repository.example/fulltext.pdf"},
        ],
    }


def test_maps_search_page_metadata_and_preserves_query(monkeypatch):
    _patch_limits(monkeypatch)
    calls = []
    payload = {
        "opensearch:totalResults": 1,
        "opensearch:startIndex": 1,
        "opensearch:itemsPerPage": 1,
        "items": [_record()],
    }

    def fake_get(url, params=None, **kwargs):
        calls.append((url, params, kwargs))
        return Response(payload)

    monkeypatch.setattr("core.http_retry.http_get", fake_get)
    result = CiNiiResearchCollector().search_page("finasteride hair regrowth", limit=1)

    assert isinstance(result, SearchPage)
    assert result.next_cursor is None
    assert result.has_more is False
    assert result.total_available == 1
    item = result.items[0]
    assert item.title == "CiNii title"
    assert item.original_title == "CiNii title"
    assert item.abstract == "Abstract text."
    assert item.authors == ["A Author", "B Author"]
    assert item.metadata["journal"] == "CiNii Journal"
    assert item.year == 2022
    assert item.doi == "10.1000/cinii"
    assert item.pmid is None
    assert item.pmcid is None
    assert item.url.endswith("/123")
    assert item.full_text_url.endswith("fulltext.pdf")
    assert item.language == "en"
    assert item.source == "CiNii Research"
    assert item.source_id.endswith("/123")
    assert item.full_text_available is True
    assert item.metadata["provenance"]["provider"] == "CiNii Research"
    assert calls[0][1]["q"] == "finasteride hair regrowth"
    assert calls[0][1]["count"] == 1
    assert calls[0][1]["start"] == 1
    assert calls[0][1]["format"] == "json"


def test_keeps_cinii_start_cursor_inside_search_page(monkeypatch):
    _patch_limits(monkeypatch)
    payloads = iter([
        {
            "opensearch:totalResults": 2,
            "items": [_record()],
        },
        {
            "opensearch:totalResults": 2,
            "items": [_record("https://cir.nii.ac.jp/crid/456", "Second")],
        },
    ])
    calls = []

    def fake_get(_url, params=None, **_kwargs):
        calls.append(params)
        return Response(next(payloads))

    monkeypatch.setattr("core.http_retry.http_get", fake_get)
    collector = CiNiiResearchCollector()
    first = collector.search_page("query", limit=2)
    second = collector.search_page("query", cursor=first.next_cursor, limit=2)

    assert first.has_more is True
    assert first.next_cursor == {"start": 2, "collected": 1, "total": 2}
    assert second.has_more is False
    assert calls[0]["start"] == 1
    assert calls[0]["count"] == 2
    assert calls[1]["start"] == 2
