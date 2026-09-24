from __future__ import annotations

from types import SimpleNamespace

from collectors.pubmed import PubMedCollector


class _Response:
    def __init__(self, payload=None, text="", status_code=200):
        self._payload = payload or {}
        self.text = text
        self.status_code = status_code

    def json(self):
        return self._payload


def test_pubmed_summary_and_fetch_are_batched_and_partial_failures_survive(monkeypatch):
    ids = [str(i) for i in range(1, 251)]
    limits = SimpleNamespace(
        collector_timeout_s=1,
        collector_max_retries=0,
        backoff_base_s=0,
        backoff_max_s=0,
        pubmed_inter_call_sleep_s=0,
    )
    monkeypatch.setattr("collectors.pubmed._lim", lambda: limits)
    monkeypatch.setattr("collectors.pubmed.time.sleep", lambda *_: None)

    calls = {"search": 0, "summary": [], "fetch": []}

    def fake_http_get(url, params=None, **_kwargs):
        if url.endswith("esearch.fcgi"):
            start = calls["search"] * 100
            calls["search"] += 1
            batch = ids[start:start + 100]
            return _Response({"esearchresult": {
                "count": str(len(ids)), "idlist": batch,
            }})
        if url.endswith("esummary.fcgi"):
            batch = params["id"].split(",")
            calls["summary"].append(batch)
            if len(calls["summary"]) == 2:
                raise RuntimeError("summary batch failed")
            return _Response({"result": {
                pmid: {"title": f"Title {pmid}", "authors": []}
                for pmid in batch
            }})
        if url.endswith("efetch.fcgi"):
            batch = params["id"].split(",")
            calls["fetch"].append(batch)
            if len(calls["fetch"]) == 2:
                raise RuntimeError("fetch batch failed")
            return _Response(text="\n\n\n".join(f"Abstract {pmid}" for pmid in batch))
        raise AssertionError(url)

    monkeypatch.setattr("core.http_retry.http_get", fake_http_get)
    results = PubMedCollector().search("finasteride", limit=250)

    assert len(results) == len(ids)
    assert all(len(batch) <= 100 for batch in calls["summary"])
    assert all(len(batch) <= 100 for batch in calls["fetch"])
    assert results[0].title == "Title 1"
    assert results[100].title == ""
    assert results[200].title == "Title 201"
    assert results[0].abstract == "Abstract 1"
    assert results[100].abstract == ""
    assert results[200].abstract == "Abstract 201"
