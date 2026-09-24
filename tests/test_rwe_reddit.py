"""Focused tests for the public Reddit RSS RWE source."""
from types import SimpleNamespace
from unittest.mock import patch

import requests

from core.rwe.reddit_adapter import RedditRWEAdapter, STATUS_NETWORK_ERROR, STATUS_OK


REDDIT_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>t3_abc123</id>
    <title>My finasteride experience</title>
    <link href="https://www.reddit.com/r/tressless/comments/abc123/my_experience/" />
    <updated>2025-03-04T12:30:00+00:00</updated>
    <content type="html"><![CDATA[<p>Finasteride reduced my shedding.</p>]]></content>
  </entry>
</feed>"""


def _response(content: str):
    response = SimpleNamespace(content=content.encode("utf-8"), status_code=200)
    response.raise_for_status = lambda: None
    return response


def test_reddit_rss_parses_to_rwe_item_with_provenance(monkeypatch):
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    adapter = RedditRWEAdapter()
    with patch("core.http_retry.requests.get", return_value=_response(REDDIT_ATOM)) as get:
        items, status, _ = adapter.search_with_status("finasteride hair loss", limit=5)

    assert status == STATUS_OK
    assert len(items) == 1
    item = items[0]
    assert item.source == "reddit"
    assert item.source_type == "community_forum"
    assert item.evidence_tier == "anecdotal"
    assert item.collection_method == "official_rss_feed"
    assert item.external_id == "t3_abc123"
    assert item.source_url.endswith("/abc123/my_experience/")
    assert item.metadata["subreddit"] == "tressless"
    assert item.date == "2025-03-04"
    assert item.text == "Finasteride reduced my shedding."
    assert item.topic == "finasteride hair loss"
    assert get.call_args.kwargs["headers"]["Accept"].startswith("application/atom+xml")
    assert "REDDIT_CLIENT_ID" not in get.call_args.kwargs
    assert "REDDIT_CLIENT_SECRET" not in get.call_args.kwargs


def test_reddit_rss_timeout_is_non_blocking():
    adapter = RedditRWEAdapter()
    with patch("core.http_retry.requests.get", side_effect=requests.Timeout("timed out")):
        items, status, reason = adapter.search_with_status("dutasteride")

    assert items == []
    assert status == STATUS_NETWORK_ERROR
    assert "network" in reason.lower()


def test_reddit_source_can_be_disabled_without_calling_adapter(monkeypatch):
    from core.rwe.pipeline import RWEPipeline

    pipe = RWEPipeline()
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("disabled Reddit source was called")

    with patch.object(pipe.reddit, "search_with_status", fail_if_called), \
         patch.object(pipe.openfda, "search_with_status",
                      return_value=([], "no_results", "none")), \
         patch.object(pipe._engine, "plan", return_value=SimpleNamespace(
             original_query="finasteride", translated_query="finasteride",
             canonical_query="", detected_language="en", translation_applied=False,
             expanded_queries=[], entities=[], vocabulary={}, intent=None,
             to_dict=lambda: {"expanded_queries": []},
         )):
        result = pipe.search("finasteride", sources=["openfda_faers"])

    assert called is False
    assert "reddit" not in result.source_status


def test_reddit_registry_declares_public_rss_without_credentials():
    from core.rwe.models import RWE_SOURCES

    meta = RWE_SOURCES["reddit"]
    assert meta["collection_method"] == "official_rss_feed"
    assert meta["source_type"] == "community_forum"
    assert meta["evidence_tier"] == "anecdotal"


def test_reddit_rss_requires_no_credentials(monkeypatch):
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    adapter = RedditRWEAdapter()
    with patch("core.http_retry.requests.get", return_value=_response(REDDIT_ATOM)) as get:
        items, status, _ = adapter.search_with_status("finasteride")

    assert status == STATUS_OK
    assert items
    assert "Authorization" not in get.call_args.kwargs["headers"]
