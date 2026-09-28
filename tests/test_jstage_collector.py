from types import SimpleNamespace

from collectors.jstage import JStageCollector
from core.search_page import SearchPage


class Response:
    status_code = 200
    headers = {}

    def __init__(self, content):
        self.content = content


ATOM = "http://www.w3.org/2005/Atom"
PRISM = "http://prismstandard.org/namespaces/basic/2.0/"
OPENSEARCH = "http://a9.com/-/spec/opensearch/1.1/"


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


def _feed(total, entries, status="0"):
    entries_xml = "".join(entries)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="{ATOM}" xmlns:prism="{PRISM}" xmlns:opensearch="{OPENSEARCH}">
      <result><status>{status}</status><message>{status}</message></result>
      <opensearch:totalResults>{total}</opensearch:totalResults>
      <opensearch:startIndex>1</opensearch:startIndex>
      <opensearch:itemsPerPage>{len(entries)}</opensearch:itemsPerPage>
      {entries_xml}
    </feed>""".encode()


def _entry(identifier="https://www.jstage.jst.go.jp/article/test/1/1/1_1/_article", title="J-STAGE title"):
    return f"""<entry xmlns="{ATOM}" xmlns:prism="{PRISM}">
      <article_title><en>{title}</en><ja>日本語タイトル</ja></article_title>
      <article_link><en>{identifier}</en><ja>{identifier}/-char/ja</ja></article_link>
      <author><en><name>A Author</name><name>B Author</name></en></author>
      <cdjournal>test</cdjournal>
      <material_title><en>J-STAGE Journal</en><ja>ジャーナル</ja></material_title>
      <pubyear>2021</pubyear>
      <prism:doi>10.1000/jstage</prism:doi>
      <systemname>J-STAGE</systemname>
      <id>{identifier}</id>
    </entry>"""


def test_maps_atom_metadata_and_preserves_query(monkeypatch):
    _patch_limits(monkeypatch)
    calls = []
    payload = _feed(1, [_entry()])

    def fake_get(url, params=None, **kwargs):
        calls.append((url, params, kwargs))
        return Response(payload)

    monkeypatch.setattr("core.http_retry.http_get", fake_get)
    result = JStageCollector().search_page("finasteride hair regrowth", limit=1)

    assert isinstance(result, SearchPage)
    assert result.next_cursor is None
    assert result.has_more is False
    assert result.total_available == 1
    item = result.items[0]
    assert item.title == "J-STAGE title"
    assert item.original_title == "J-STAGE title"
    assert item.abstract == ""
    assert item.authors == ["A Author", "B Author"]
    assert item.metadata["journal"] == "J-STAGE Journal"
    assert item.year == 2021
    assert item.doi == "10.1000/jstage"
    assert item.pmid is None
    assert item.pmcid is None
    assert item.url.endswith("/_article")
    assert item.full_text_url is None
    assert item.language is None
    assert item.source == "J-STAGE"
    assert item.source_id.endswith("/_article")
    assert item.full_text_available is False
    assert item.metadata["provenance"]["provider"] == "J-STAGE"
    assert calls[0][1]["service"] == 3
    assert calls[0][1]["text"] == "finasteride hair regrowth"
    assert calls[0][1]["start"] == 1
    assert calls[0][1]["count"] == 1


def test_keeps_jstage_offset_cursor_inside_search_page(monkeypatch):
    _patch_limits(monkeypatch)
    payloads = iter([
        _feed(2, [_entry()]),
        _feed(2, [_entry("https://www.jstage.jst.go.jp/article/test/1/2/1_2/_article", "Second")]),
    ])
    calls = []

    def fake_get(_url, params=None, **_kwargs):
        calls.append(params)
        return Response(next(payloads))

    monkeypatch.setattr("core.http_retry.http_get", fake_get)
    collector = JStageCollector()
    first = collector.search_page("query", limit=2)
    second = collector.search_page("query", cursor=first.next_cursor, limit=2)

    assert first.has_more is True
    assert first.next_cursor == {"start": 2, "collected": 1, "total": 2}
    assert second.has_more is False
    assert calls[0]["start"] == 1
    assert calls[0]["count"] == 2
    assert calls[1]["start"] == 2


def test_warn_002_parses_records_and_reports_total(monkeypatch):
    _patch_limits(monkeypatch)
    payload = _feed(235344, [_entry(title="Hormonal Effects of Z-350, Possessing <i>5&amp;alpha;-Reductase</i> Actions")], status="WARN_002")

    monkeypatch.setattr("core.http_retry.http_get", lambda *args, **kwargs: Response(payload))
    result = JStageCollector().search_page("cancer", limit=1)

    assert isinstance(result, SearchPage)
    assert result.total_available == 235344
    assert result.has_more is False
    assert result.next_cursor is None
    item = result.items[0]
    assert item.title == "Hormonal Effects of Z-350, Possessing 5α-Reductase Actions"
    assert item.doi == "10.1000/jstage"
    assert item.authors == ["A Author", "B Author"]
    assert item.year == 2021
    assert item.metadata["journal"] == "J-STAGE Journal"
    assert item.url.endswith("/_article")
    assert item.source_id.endswith("/_article")
    assert item.source == "J-STAGE"
