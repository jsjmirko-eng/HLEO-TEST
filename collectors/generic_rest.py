"""
Generic REST API collector for HLEO

Supports basic configuration-driven ingestion via SourceRegistry.connection_spec.
- GET/POST
- base_url + path
- params or json body templates with simple placeholders {query}, {limit}, {offset}, {token}
- headers (supports ENV:VAR to pull from env)
- credentials_env_vars (checked by admin test)
- pagination: offset/limit or token (basic)
- items_path: dot-separated path into JSON response (e.g. data.items or items)
- field_mappings: map response fields to HLEO model fields (supports list extraction with [] syntax)

Implements both interfaces used in codebase:
- search(query, limit) -> list[core.search_result.SearchResult]
- search_with_status(query, limit) -> (List[core.rwe.models.RWEItem], status, reason)

This collector is intentionally permissive but conservative: it will not execute arbitrary code
from connection_spec and treats connection_spec as configuration only.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from core.search_result import SearchResult
from core.rwe.models import RWEItem

logger = logging.getLogger(__name__)


def _get_env_value(v: str) -> str:
    """Support header values like 'ENV:MY_API_KEY' which reads os.environ['MY_API_KEY']."""
    if isinstance(v, str) and v.startswith("ENV:"):
        return os.getenv(v.split("ENV:", 1)[1], "")
    return v


def _resolve_template(tpl: Any, mapping: Dict[str, Any]) -> Any:
    """Resolve simple templates in strings or in nested structures.
    Only supports replacing placeholders like {query} and {limit}.
    If tpl is a dict/list, recurse; otherwise if string, do .format_map with mapping.
    """
    if tpl is None:
        return None
    if isinstance(tpl, str):
        try:
            return tpl.format_map(mapping)
        except Exception:
            # fallback: return as-is
            return tpl
    if isinstance(tpl, dict):
        out = {}
        for k, v in tpl.items():
            out[k] = _resolve_template(v, mapping)
        return out
    if isinstance(tpl, list):
        return [_resolve_template(i, mapping) for i in tpl]
    return tpl


def _extract_path(obj: Any, path: str):
    """Simple dot-path extractor with limited array support.
    Example paths:
      - 'data.items' -> obj['data']['items']
      - 'items' -> obj['items']
      - 'authors[*].name' -> list of names
    Returns None if path not found.
    """
    if not path:
        return obj
    # allow leading $. trim it
    if path.startswith("$."):
        path = path[2:]
    parts = path.split('.')
    cur = obj
    for p in parts:
        if cur is None:
            return None
        # list wildcard
        if p.endswith('[*]'):
            key = p[:-3]
            cur = cur.get(key, []) if isinstance(cur, dict) else None
            if not isinstance(cur, list):
                return None
            # if last part return list itself
            # else map next accessor across elements
            # build remainder path
            remainder = '.'.join(parts[parts.index(p)+1:])
            if not remainder:
                return cur
            results = []
            for el in cur:
                val = _extract_path(el, remainder)
                if isinstance(val, list):
                    results.extend(val)
                elif val is not None:
                    results.append(val)
            return results
        else:
            if isinstance(cur, dict):
                cur = cur.get(p)
            else:
                return None
    return cur


class GenericRESTCollector:
    """Configuration-driven REST API collector.

    connection_spec keys supported (example):
    {
      "base_url": "https://api.example.org",
      "path": "/v1/search",
      "method": "GET",
      "params": {"q": "{query}", "limit": "{limit}"},
      "json": null or {"query":"{query}"},
      "headers": {"x-api-key": "ENV:MY_API_KEY"},
      "pagination": {"type":"offset","param":"offset","start":0,"size_param":"limit","page_size":100},
      "items_path": "data.items",
      "field_mappings": {"title":"title","abstract":"description","url":"link","authors":"authors[*].name","doi":"doi"}
    }
    """

    def __init__(self, connection_spec: Dict[str, Any], source_id: Optional[str] = None, category: Optional[str] = None):
        self.spec = connection_spec or {}
        self.base_url = self.spec.get('base_url', '').rstrip('/')
        self.path = self.spec.get('path', '')
        self.method = (self.spec.get('method', 'GET') or 'GET').upper()
        self.headers = self.spec.get('headers', {}) or {}
        self.params_template = self.spec.get('params')
        self.json_template = self.spec.get('json')
        self.data_template = self.spec.get('data')
        self.items_path = self.spec.get('items_path') or self.spec.get('itemsPath') or ''
        self.field_mappings = self.spec.get('field_mappings') or self.spec.get('fieldMappings') or {}
        self.pagination = self.spec.get('pagination') or {}
        # credentials_env_vars not fully used here; admin.test checks presence
        self.credentials_env_vars = self.spec.get('credentials_env_vars') or self.spec.get('credentialsEnvVars') or []
        self.source_id = source_id
        self.category = category or self.spec.get('category') or 'rwe'
        # resolve headers env tokens
        for hk, hv in list(self.headers.items()):
            self.headers[hk] = _get_env_value(hv)

    def _build_url(self):
        if self.path.startswith('http://') or self.path.startswith('https://'):
            return self.path
        return f"{self.base_url}{self.path}"

    def _call_api(self, query: str, limit: int = 10, extra: Optional[Dict[str, Any]] = None, pagination_state: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        url = self._build_url()
        mapping = {"query": query, "limit": limit}
        if extra:
            mapping.update(extra)
        # merge pagination_state into mapping
        if pagination_state:
            mapping.update(pagination_state)
        params = _resolve_template(self.params_template, mapping) if self.params_template else None
        json_body = _resolve_template(self.json_template, mapping) if self.json_template else None
        data_body = _resolve_template(self.data_template, mapping) if self.data_template else None
        headers = dict(self.headers or {})

        try:
            if self.method == 'GET':
                r = requests.get(url, params=params, headers=headers, timeout=15)
            else:
                # POST
                if json_body is not None:
                    r = requests.post(url, json=json_body, params=params, headers=headers, timeout=15)
                elif data_body is not None:
                    r = requests.post(url, data=data_body, params=params, headers=headers, timeout=15)
                else:
                    r = requests.post(url, params=params, headers=headers, timeout=15)
            r.raise_for_status()
            return r.json(), None
        except requests.exceptions.RequestException as exc:
            logger.warning("GenericRESTCollector network error for %s: %s", url, exc)
            return None, str(exc)
        except ValueError as exc:
            logger.warning("GenericRESTCollector JSON parse error for %s: %s", url, exc)
            return None, str(exc)

    def _extract_items(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        if payload is None:
            return []
        if not self.items_path:
            # try common keys
            if isinstance(payload, dict) and 'data' in payload and isinstance(payload['data'], list):
                return payload['data']
            if isinstance(payload, dict) and 'items' in payload and isinstance(payload['items'], list):
                return payload['items']
            # if response itself is list
            if isinstance(payload, list):
                return payload
            return []
        return _extract_path(payload, self.items_path) or []

    def _map_scientific(self, item: Dict[str, Any]) -> SearchResult:
        title = _extract_path(item, self.field_mappings.get('title') or 'title') or ''
        abstract = _extract_path(item, self.field_mappings.get('abstract') or 'abstract') or ''
        url = _extract_path(item, self.field_mappings.get('url') or 'url') or ''
        authors_raw = _extract_path(item, self.field_mappings.get('authors') or '') or []
        authors = []
        if isinstance(authors_raw, list):
            for a in authors_raw:
                if isinstance(a, str):
                    authors.append(a)
                elif isinstance(a, dict) and 'name' in a:
                    authors.append(a.get('name'))
        elif isinstance(authors_raw, str):
            authors = [authors_raw]
        year = _extract_path(item, self.field_mappings.get('year') or 'year')
        doi = _extract_path(item, self.field_mappings.get('doi') or 'doi')
        pmid = _extract_path(item, self.field_mappings.get('pmid') or 'pmid')
        metadata = {}
        # include whole item as metadata fallback
        metadata.update(item if isinstance(item, dict) else {})
        return SearchResult(
            title=title,
            source=self.source_id or (self.spec.get('source_id') or 'generic_rest'),
            url=url,
            abstract=abstract,
            authors=authors,
            year=int(year) if isinstance(year, (int, str)) and str(year).isdigit() else None,
            doi=(doi or None),
            pmid=(pmid or None),
            metadata=metadata,
        )

    def _map_rwe(self, item: Dict[str, Any]) -> RWEItem:
        # Build RWEItem fields from mapping
        title = _extract_path(item, self.field_mappings.get('title') or 'title') or ''
        text = _extract_path(item, self.field_mappings.get('text') or 'text') or ''
        external_id = str(_extract_path(item, self.field_mappings.get('external_id') or 'id') or '')
        source_url = _extract_path(item, self.field_mappings.get('source_url') or 'url') or ''
        treatment = _extract_path(item, self.field_mappings.get('treatment') or 'treatment')
        condition = _extract_path(item, self.field_mappings.get('condition') or 'condition')
        language = _extract_path(item, self.field_mappings.get('language') or 'language') or 'en'
        evidence_tier = _extract_path(item, self.field_mappings.get('evidence_tier') or 'evidence_tier') or 'anecdotal'
        topic = _extract_path(item, self.field_mappings.get('topic') or 'topic') or ''
        metadata = item if isinstance(item, dict) else {}
        return RWEItem(
            source=self.source_id or (self.spec.get('source_id') or 'generic_rest'),
            source_type=self.spec.get('source_type', 'api'),
            evidence_tier=evidence_tier,
            collection_method=self.spec.get('collection_method', 'official_api'),
            external_id=external_id,
            source_url=source_url,
            title=title,
            text=text,
            date=_extract_path(item, self.field_mappings.get('date') or 'date'),
            language=language,
            topic=topic,
            treatment=treatment,
            condition=condition,
            experience_type=_extract_path(item, self.field_mappings.get('experience_type') or 'experience_type') or 'discussion',
            metadata=metadata,
        )

    def _fetch_all(self, query: str, limit: int = 10) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Fetch pages and return flattened list of items and optional error reason."""
        items: List[Dict[str, Any]] = []
        pag = self.pagination or {}
        ptype = pag.get('type') or pag.get('mode') or 'none'

        if ptype in ('offset', 'page'):
            # offset-based pagination
            param = pag.get('param', 'offset')
            size_param = pag.get('size_param', 'limit')
            start = int(pag.get('start', 0))
            page_size = int(pag.get('page_size', 100))
            fetched = 0
            offset = start
            while fetched < limit:
                page_limit = min(page_size, limit - fetched)
                extra = {param: offset, 'limit': page_limit}
                payload, err = self._call_api(query, page_limit, extra=extra, pagination_state={'offset': offset})
                if err:
                    return items, err
                page_items = self._extract_items(payload)
                if not page_items:
                    break
                items.extend(page_items)
                fetched += len(page_items)
                if len(page_items) < page_limit:
                    break
                offset += page_limit
            return items[:limit], None

        if ptype == 'token':
            token_param = pag.get('token_param', 'token')
            token_path = pag.get('token_path') or pag.get('next_token_path')
            next_token = pag.get('start_token')
            fetched = 0
            while fetched < limit:
                extra = {token_param: next_token} if next_token else {}
                payload, err = self._call_api(query, limit - fetched, extra=extra, pagination_state={'token': next_token})
                if err:
                    return items, err
                page_items = self._extract_items(payload)
                if not page_items:
                    break
                items.extend(page_items)
                fetched += len(page_items)
                if not token_path:
                    break
                next_token = _extract_path(payload, token_path)
                if not next_token:
                    break
            return items[:limit], None

        # no pagination
        payload, err = self._call_api(query, limit, extra=None)
        if err:
            return [], err
        page_items = self._extract_items(payload)
        return (page_items or [])[:limit], None

    # Public interfaces -------------------------------------------------
    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        raw_items, err = self._fetch_all(query, limit)
        if err:
            logger.info("GenericRESTCollector.search failed: %s", err)
            return []
        results: List[SearchResult] = []
        for it in raw_items:
            try:
                results.append(self._map_scientific(it))
            except Exception:
                logger.exception("GenericRESTCollector: failed to map scientific item")
        return results

    def search_with_status(self, query: str, limit: int = 10) -> Tuple[List[RWEItem], str, str]:
        raw_items, err = self._fetch_all(query, limit)
        if err:
            return [], 'network_error', err
        items: List[RWEItem] = []
        try:
            for it in raw_items:
                items.append(self._map_rwe(it))
        except Exception as exc:
            logger.exception("GenericRESTCollector: mapping error %s", exc)
            return [], 'network_error', str(exc)
        if not items:
            return [], 'no_results', 'No items'
        return items, 'ok', f"Retrieved {len(items)} item(s)"
