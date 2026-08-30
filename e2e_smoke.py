import json
import time
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

from playwright.sync_api import TimeoutError, sync_playwright

BASE_URL = 'http://127.0.0.1:8000'
results = {
    'pipeline_run_json': None,
    'rwe_extract_json': None,
    'compare_payload': None,
    'compare_response_json': None,
    'active_ctx_after_pipeline': None,
    'active_ctx_after_rwe': None,
    'console': [],
    'page_errors': [],
}


def _rewrite_query(url: str, params: dict) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update({k: str(v) for k, v in params.items()})
    return urlunparse(parsed._replace(query=urlencode(query)))


def _request_post_data(request):
    try:
        return request.post_data
    except Exception:
        try:
            return request.post_data()
        except Exception:
            return None


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        context = browser.new_context()
        page = context.new_page()

        page.on('console', lambda msg: results['console'].append({'type': msg.type, 'text': msg.text}))
        page.on('pageerror', lambda exc: results['page_errors'].append(str(exc)))

        def _route_pipeline(route, request):
            route.continue_(url=_rewrite_query(request.url, {'max_results': 1}))

        def _route_rwe_search(route, request):
            route.continue_(url=_rewrite_query(request.url, {'limit': 1}))

        def _route_rwe_extract_batch(route, request):
            body = _request_post_data(request)
            payload = json.loads(body) if body else {}
            if isinstance(payload.get('items'), list):
                payload['items'] = payload['items'][:1]
            route.continue_(post_data=json.dumps(payload))

        page.route('**/pipeline/run**', _route_pipeline)
        page.route('**/rwe/search**', _route_rwe_search)
        page.route('**/rwe/extract-batch**', _route_rwe_extract_batch)

        page.goto(BASE_URL, wait_until='networkidle')

        # 1) Scientific search
        page.evaluate("setSearchMode('scientific')")
        page.fill('#searchInput', 'finasteride alopecia')
        page.evaluate("document.getElementById('searchForm').dispatchEvent(new Event('submit'))")
        page.wait_for_selector('#searchActionBar', timeout=30000)
        with page.expect_response(lambda r: '/pipeline/run' in r.url and r.request.method == 'POST', timeout=60000) as pipeline_resp_info:
            page.click('#runPipelineBtn')
        results['pipeline_run_json'] = pipeline_resp_info.value.json()

        results['active_ctx_after_pipeline'] = page.evaluate('typeof _activeSearchCtx !== "undefined" ? _activeSearchCtx : null')

        # 2) RWE search
        page.evaluate("setSearchMode('rwe')")
        page.fill('#searchInput', 'finasteride hair shedding')
        page.evaluate("document.getElementById('searchForm').dispatchEvent(new Event('submit'))")
        page.wait_for_selector('#searchActionBar', timeout=30000)
        with page.expect_response(lambda r: '/rwe/extract-batch' in r.url and r.request.method == 'POST', timeout=60000) as rwe_resp_info:
            page.click('#runExperiencesBtn')
        results['rwe_extract_json'] = rwe_resp_info.value.json()

        results['active_ctx_after_rwe'] = page.evaluate('typeof _activeSearchCtx !== "undefined" ? _activeSearchCtx : null')

        # 3) Compare
        page.evaluate("showPage('assistant')")
        page.wait_for_function("() => { const b = document.getElementById('compareBtn'); return b && !b.disabled && b.offsetParent !== null; }", timeout=30000)
        with page.expect_request(lambda r: '/assistant/compare' in r.url and r.method == 'POST', timeout=60000) as compare_req_info, page.expect_response(lambda r: '/assistant/compare' in r.url and r.request.method == 'POST', timeout=60000) as compare_resp_info:
            page.click('#compareBtn')
        results['compare_payload'] = compare_req_info.value.post_data
        results['compare_response_json'] = compare_resp_info.value.json()

        active_ctx = results['active_ctx_after_rwe'] or {}
        compare = json.loads(results['compare_payload']) if isinstance(results['compare_payload'], str) else (results['compare_payload'] or {})

        expected_clinical = set((results['pipeline_run_json'] or {}).get('episode_ids') or [])
        expected_rwe = set([r.get('episode_id') for r in ((results['rwe_extract_json'] or {}).get('results') or []) if isinstance(r, dict)])
        active_clinical = set(active_ctx.get('clinical_profile_episode_ids') or [])
        active_rwe = set(active_ctx.get('rwe_profile_episode_ids') or [])
        payload_clinical = set(compare.get('clinical_profile_episode_ids') or [])
        payload_rwe = set(compare.get('rwe_profile_episode_ids') or [])

        summary = {
            'pipeline_episode_ids': sorted(expected_clinical),
            'rwe_episode_ids': sorted(expected_rwe),
            'payload_clinical_episode_ids': sorted(payload_clinical),
            'payload_rwe_episode_ids': sorted(payload_rwe),
            'active_clinical_episode_ids': sorted(active_clinical),
            'active_rwe_episode_ids': sorted(active_rwe),
            'compare_matches_expected_clinical': payload_clinical == expected_clinical,
            'compare_matches_expected_rwe': payload_rwe == expected_rwe,
            'compare_response_keys': sorted((results['compare_response_json'] or {}).keys()),
            'console': results['console'][-10:],
            'page_errors': results['page_errors'],
        }
        print(json.dumps(summary, indent=2, default=str))
        browser.close()


if __name__ == '__main__':
    run()
