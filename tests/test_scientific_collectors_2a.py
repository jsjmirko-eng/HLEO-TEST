from types import SimpleNamespace

from aggregator import HLEOAggregator
from collectors.crossref import CrossrefCollector
from collectors.doaj import DOAJCollector
from collectors.openalex import OpenAlexCollector
from core.search_page import SearchPage
from core.search_result import SearchResult


class Response:
    status_code = 200
    headers = {}

    def __init__(self, payload):
        self.payload = payload

    def json(self):
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


def test_openalex_maps_metadata_and_page(monkeypatch):
    limits(monkeypatch)
    payload = {
        "meta": {"count": 2},
        "results": [{
            "id": "https://openalex.org/W1",
            "title": "Finasteride study",
            "publication_year": 2024,
            "doi": "https://doi.org/10.1000/test",
            "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/123", "pmcid": "PMC123"},
            "abstract_inverted_index": {"hair": [1], "regrowth": [0]},
            "authorships": [{"author": {"display_name": "A Author"}}],
            "language": {"id": "en"},
            "primary_location": {"source": {"display_name": "Journal"}, "landing_page_url": "https://example.org/a"},
            "locations": [{"pdf_url": "https://example.org/a.pdf"}],
            "open_access": {"is_oa": True},
        }],
    }
    monkeypatch.setattr("core.http_retry.http_get", lambda *args, **kwargs: Response(payload))
    result = OpenAlexCollector().search_page("finasteride", limit=1)
    assert isinstance(result, SearchPage)
    item = result.items[0]
    assert item.abstract == "regrowth hair"
    assert item.pmid == "123"
    assert item.pmcid == "PMC123"
    assert item.language == "en"
    assert item.full_text_url.endswith(".pdf")
    assert item.full_text_available is True
    assert result.total_available == 2
    assert result.has_more is False


def test_crossref_cursor_is_opaque_and_maps_pdf(monkeypatch):
    limits(monkeypatch)
    payloads = iter([
        {"message": {"total-results": 2, "next-cursor": "cursor-1", "items": [{
            "DOI": "10.1000/crossref", "title": ["Crossref title"], "abstract": "<jats:p>Abstract</jats:p>",
            "author": [{"given": "A", "family": "Author"}], "issued": {"date-parts": [[2023]]},
            "container-title": ["Journal"], "link": [{"content-type": "application/pdf", "URL": "https://x.test/a.pdf"}],
        }] }},
        {"message": {"total-results": 2, "next-cursor": "cursor-2", "items": [{
            "DOI": "10.1000/crossref-2", "title": ["Second"], "issued": {"date-parts": [[2024]]},
        }] }},
    ])
    calls = []

    def fake_get(_url, params=None, **_kwargs):
        calls.append(params)
        return Response(next(payloads))

    monkeypatch.setattr("core.http_retry.http_get", fake_get)
    collector = CrossrefCollector()
    first = collector.search_page("query", limit=2)
    second = collector.search_page("query", cursor=first.next_cursor, limit=2)
    assert calls[0]["cursor"] == "*"
    assert calls[1]["cursor"] == "cursor-1"
    assert first.items[0].abstract == "Abstract"
    assert first.items[0].full_text_available is True
    assert second.items[0].doi == "10.1000/crossref-2"


def test_doaj_preserves_language_and_fulltext(monkeypatch):
    limits(monkeypatch)
    payload = {
        "total": 1,
        "page": 1,
        "pageSize": 1,
        "results": [{
            "id": "doaj-id",
            "bibjson": {
                "title": "Titre français",
                "abstract": "Résumé",
                "year": "2022",
                "author": [{"name": "Auteur"}],
                "journal": {"title": "Revue", "language": ["FR"]},
                "identifier": [{"type": "doi", "id": "10.1000/doaj"}],
                "link": [{"type": "fulltext", "url": "https://example.org/full"}],
            },
        }],
    }
    monkeypatch.setattr("core.http_retry.http_get", lambda *args, **kwargs: Response(payload))
    item = DOAJCollector().search("finasteride", limit=1)[0]
    assert item.title == "Titre français"
    assert item.language == "FR"
    assert item.doi == "10.1000/doaj"
    assert item.full_text_available is True
    assert item.metadata["journal"] == "Revue"


def test_dedup_merges_sources_for_doi():
    first = SearchResult(title="Same", source="OpenAlex", doi="10.1000/same", abstract="long abstract", sources=["OpenAlex"])
    second = SearchResult(title="Same", source="Crossref", doi="10.1000/same", abstract="long abstract", sources=["Crossref"])
    cleaned, stats = HLEOAggregator().deduplicate_across_sources({"openalex": [first], "crossref": [second]})
    survivor = next(item for items in cleaned.values() for item in items if item)
    assert stats["unique"] == 1
    assert survivor.sources == ["OpenAlex", "Crossref"]
    assert survivor.metadata["sources"] == ["OpenAlex", "Crossref"]
