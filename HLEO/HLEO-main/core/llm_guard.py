"""
HLEO — LLM call guard (cost protection + bounded retry)
========================================================

Centralises EVERY OpenAI call so retry cannot multiply across layers.

Hard rules (enforced everywhere this module is used):
    MAX_TOTAL_ATTEMPTS = 5
        1 initial attempt + at most 4 retries = 5 LLM calls per operation.
        No caller/layer may add its own retry on top — doing so would breach
        the absolute cap. The guard is the ONLY retry boundary.

429 handling (per the master spec):
    - insufficient_quota / credit_balance_exhausted  → NO retry, raise
      QuotaExhaustedError immediately (credito esaurito).
    - rate_limit_exceeded (temporary)               → retry with backoff,
      up to MAX_TOTAL_ATTEMPTS.
    - other transient errors                       → retry, up to the cap.
    - JSON/schema validation errors                → retry, up to the cap
      (temperature is nudged on retries to break a stuck malformed response).

JSON output hardening (local-first support):
    - _extract_json() sanitises LLM output before parsing: markdown fences,
      prose wrapping (first balanced {...}/[...] block), invalid backslash
      escapes from local models. Never alters the parsed data.
    - Optional local-first routing via env (HLEO_LOCAL_LLM_URL …) sends
      intermediate operations to a local OpenAI-compatible server while
      "final" user-facing operations stay on OpenAI. With
      HLEO_LOCAL_LLM_FALLBACK=1 (default) the LAST attempt of the bounded
      retry loop falls back to the original OpenAI client if the local
      server keeps failing — the absolute 5-attempt cap still holds.
      Unset env → behaviour identical to plain OpenAI mode.

Optional Perplexity provider (HLEO_LLM_PROVIDER=perplexity):
    - OpenAI-compatible Sonar API becomes the primary provider; OpenAI stays
      as last-attempt fallback (HLEO_PERPLEXITY_FALLBACK=1, default).
    - Perplexity rejects response_format={"type": "json_object"} (400) → the
      guard drops it on Perplexity-routed calls and _extract_json() parses.
    - Intermediate ops default to HLEO_PERPLEXITY_MODEL (sonar) with web
      search disabled (HLEO_PERPLEXITY_DISABLE_SEARCH=1) — HLEO retrieval is
      PubMed/EuropePMC/CT.gov/RWE collectors, not the LLM. "Final" ops use
      HLEO_PERPLEXITY_MODEL_FINAL (sonar-pro) with search ON.
    - Every call (success or error) is recorded via _record_call():
      operation, provider, model, tokens, Perplexity-reported cost, latency,
      fallback flag. Dumped as JSONL at exit (HLEO_LLM_CALL_LOG).
      API keys are never logged.

Backoff: exponential, capped, with light jitter:
    delay = min(BASE_DELAY * 2**attempt, MAX_DELAY) * (1 ± 0.15)
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import re
import time
import random
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Absolute cap ─────────────────────────────────────────────────────────────
MAX_TOTAL_ATTEMPTS = 5      # 1 initial + 4 retries. No layer may exceed this.

# ── Backoff tunables ─────────────────────────────────────────────────────────
_BASE_DELAY = 2.0           # seconds; doubled each attempt
_MAX_DELAY = 30.0           # cap a single backoff sleep
_JITTER = 0.15              # ±15% jitter


class QuotaExhaustedError(RuntimeError):
    """Raised when the OpenAI account has no credit/quota — NOT retryable."""


class LLMCallError(RuntimeError):
    """Raised after MAX_TOTAL_ATTEMPTS is exhausted, or for non-429 hard errors."""


# ── 429 / quota classification ───────────────────────────────────────────────

# Substrings (case-insensitive) that mean the account credit is exhausted and
# retrying is pointless. Sourced from OpenAI's documented error shapes plus the
# specific codes named in the master spec.
_QUOTA_SIGNALS = (
    "insufficient_quota",
    "credit_balance_exhausted",
    "billing_hard_limit_reached",
    "exceeded your current quota",
    "you exceeded your current quota",
)

# Substrings indicating a temporary rate-limit that MAY resolve after backoff.
_RATE_LIMIT_SIGNALS = (
    "rate_limit_exceeded",
    "requests can be made",
    "too many requests",
    "rate limit",
)


def classify_429(message: str) -> str:
    """Classify a 429/error message into quota-exhausted vs rate-limited vs other.

    Returns one of:
        "quota_exhausted"  — NOT retryable (raise immediately)
        "rate_limit"        — retryable (backoff)
        "other"             — retryable (treated as transient)
    """
    msg = (message or "").lower()
    if any(s in msg for s in _QUOTA_SIGNALS):
        return "quota_exhausted"
    if "429" in msg or any(s in msg for s in _RATE_LIMIT_SIGNALS):
        return "rate_limit"
    return "other"


def _backoff_delay(attempt: int) -> float:
    raw = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
    return raw * (1.0 + random.uniform(-_JITTER, _JITTER))


def _extract_openai_message(exc: Exception) -> str:
    """OpenAI SDK errors carry a structured body; pull a readable string out."""
    # openai>=1.x: exc.response / exc.body / exc.message
    for attr in ("body", "message", "response"):
        val = getattr(exc, attr, None)
        if val:
            if isinstance(val, dict):
                inner = val.get("error") if "error" in val else val
                if isinstance(inner, dict):
                    return str(inner.get("message") or inner)
                return str(inner)
            return str(val)
    return str(exc)


# ── JSON output sanitization ─────────────────────────────────────────────────
# Local models (llama.cpp etc.) can wrap JSON in prose/fences or emit invalid
# escapes (e.g. Vicuna's LaTeX-style "\_"). Extraction below only reframes the
# transport; it never invents or rewrites parsed data.

_INVALID_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')


def _balanced_json_block(s: str) -> str:
    """Return the first balanced {...} or [...] block in s (string-aware)."""
    start = -1
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start == -1:
        raise json.JSONDecodeError("no JSON object/array in output", s, 0)
    stack: list[str] = []
    in_str = False
    esc = False
    pairs = {"}": "{", "]": "["}
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack and stack[-1] == pairs[ch]:
                stack.pop()
                if not stack:
                    return s[start:i + 1]
    # Unbalanced (e.g. truncated at max_tokens): return what we have.
    return s[start:]


def _extract_json(raw: str) -> Any:
    """Parse LLM output as JSON, tolerating framing noise.

    Order: markdown-fence strip → direct parse → first balanced block →
    drop invalid backslash escapes (a local-model quirk; only sequences
    that are not valid JSON escapes are touched). Raises JSONDecodeError
    if still unparseable.
    """
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    candidate = _balanced_json_block(s)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return json.loads(_INVALID_ESCAPE_RE.sub("", candidate))


# ── Optional local-first routing (llama.cpp / OpenAI-compatible server) ──────
# OFF by default: with HLEO_LOCAL_LLM_URL unset, behaviour is unchanged.
#
#   HLEO_LOCAL_LLM_URL       e.g. http://127.0.0.1:8081/v1
#   HLEO_LOCAL_LLM_MODEL     model name sent to the local server
#   HLEO_LOCAL_LLM_OPS       comma-separated operations routed locally;
#                            default: all EXCEPT the "final" ops below
#   HLEO_LOCAL_LLM_FALLBACK  "1" (default)/"0": if the local server keeps
#                            failing, the LAST of the MAX_TOTAL_ATTEMPTS
#                            attempts goes to the original (OpenAI) client.
#                            The absolute 5-attempt cap still holds.
_LOCAL_URL = os.getenv("HLEO_LOCAL_LLM_URL", "").strip().rstrip("/")
_LOCAL_MODEL = os.getenv("HLEO_LOCAL_LLM_MODEL", "local-model")
_LOCAL_OPS_ENV = os.getenv("HLEO_LOCAL_LLM_OPS", "")
_LOCAL_OPS = ({o.strip() for o in _LOCAL_OPS_ENV.split(",") if o.strip()}
              if _LOCAL_OPS_ENV.strip() else None)
_LOCAL_FALLBACK = os.getenv("HLEO_LOCAL_LLM_FALLBACK", "1") != "0"

# "Final" user-facing operations stay on OpenAI in local-first mode.
_FINAL_OPS = {
    "scientific_synthesis", "card_synthesis", "assistant_chat", "assistant_compare",
}

_local_client: Any = None


def _get_local_client() -> Any:
    global _local_client
    if _local_client is None:
        from openai import OpenAI
        _local_client = OpenAI(
            base_url=_LOCAL_URL,
            api_key=os.getenv("HLEO_LOCAL_LLM_API_KEY", "sk-local"),
            timeout=float(os.getenv("HLEO_LOCAL_LLM_TIMEOUT", "600")),
        )
    return _local_client


_ROUTING_STATS = {"local_calls": 0, "openai_calls": 0, "fallbacks": 0}


def get_routing_stats() -> dict:
    return dict(_ROUTING_STATS)


def reset_routing_stats() -> None:
    for k in _ROUTING_STATS:
        _ROUTING_STATS[k] = 0


def _route(operation: str, client: Any, model: str):
    """Return (client, model, fallback_client, fallback_model, route_name)."""
    if not _LOCAL_URL:
        return client, model, None, None, None
    # Guard: caller already points at the local server.
    if str(getattr(client, "base_url", "")).startswith(_LOCAL_URL):
        return client, model, None, None, None
    if _LOCAL_OPS is not None:
        go_local = operation in _LOCAL_OPS
    else:
        go_local = operation not in _FINAL_OPS
    if not go_local:
        return client, model, None, None, None
    if _LOCAL_FALLBACK:
        return _get_local_client(), _LOCAL_MODEL, client, model, "local"
    return _get_local_client(), _LOCAL_MODEL, None, None, "local"


# ── Optional Perplexity provider (OpenAI-compatible, fallback OpenAI) ────────
# OFF by default: HLEO_LLM_PROVIDER unset/≠"perplexity" → behaviour unchanged.
#
#   HLEO_LLM_PROVIDER="perplexity"     activate Perplexity as primary provider
#   PERPLEXITY_API_KEY                 required (env only, never logged)
#   HLEO_PERPLEXITY_MODEL              model for intermediate ops (default sonar)
#   HLEO_PERPLEXITY_MODEL_FINAL        model for "final" ops (default sonar-pro)
#   HLEO_PERPLEXITY_OPS                comma-separated ops to route; default: all
#   HLEO_PERPLEXITY_DISABLE_SEARCH     "1" (default): web search OFF for
#                                      intermediate ops (HLEO retrieval is
#                                      PubMed/EuropePMC/CT.gov, not the LLM);
#                                      "final" ops keep search ON.
#   HLEO_PERPLEXITY_FALLBACK           "1" (default): last attempt → OpenAI.
#
# Perplexity rejects response_format={"type":"json_object"} (400); when a call
# is routed to Perplexity the guard drops that field — _extract_json() then
# parses the plain-text output (verified live: relation extraction works).
_PPX_ENABLED = os.getenv("HLEO_LLM_PROVIDER", "").strip().lower() == "perplexity"
_PPX_API_KEY = os.getenv("PERPLEXITY_API_KEY", "").strip()
_PPX_BASE_URL = "https://api.perplexity.ai"
_PPX_MODEL = os.getenv("HLEO_PERPLEXITY_MODEL", "sonar")
_PPX_MODEL_FINAL = os.getenv("HLEO_PERPLEXITY_MODEL_FINAL", "sonar-pro")
_PPX_OPS_ENV = os.getenv("HLEO_PERPLEXITY_OPS", "")
_PPX_OPS = ({o.strip() for o in _PPX_OPS_ENV.split(",") if o.strip()}
            if _PPX_OPS_ENV.strip() else None)
_PPX_DISABLE_SEARCH = os.getenv("HLEO_PERPLEXITY_DISABLE_SEARCH", "1") != "0"
_PPX_FALLBACK = os.getenv("HLEO_PERPLEXITY_FALLBACK", "1") != "0"
# Perplexity Sonar emits longer structured outputs than gpt-4o at equal
# max_tokens (observed: scientific_synthesis truncated at ~10k chars with
# max_tokens=2200 → 4 consecutive JSON parse failures). Raise the floor for
# Perplexity-routed calls only; callers/OpenAI behaviour unchanged.
_PPX_MIN_MAX_TOKENS = int(os.getenv("HLEO_PERPLEXITY_MIN_MAX_TOKENS", "8000"))

_ppx_client: Any = None


def _get_ppx_client() -> Any:
    global _ppx_client
    if _ppx_client is None:
        from openai import OpenAI
        _ppx_client = OpenAI(
            base_url=_PPX_BASE_URL,
            api_key=_PPX_API_KEY,
            timeout=float(os.getenv("HLEO_PERPLEXITY_TIMEOUT", "180")),
        )
    return _ppx_client


def _route_ppx(operation: str, client: Any, model: str):
    """Return (client, model, fb_client, fb_model, route_name, is_final)."""
    if not (_PPX_ENABLED and _PPX_API_KEY):
        return client, model, None, None, None, False
    # Guard: caller already points at Perplexity.
    if "perplexity.ai" in str(getattr(client, "base_url", "")):
        return client, model, None, None, None, False
    if _PPX_OPS is not None and operation not in _PPX_OPS:
        return client, model, None, None, None, False
    is_final = operation in _FINAL_OPS
    ppx_model = _PPX_MODEL_FINAL if is_final else _PPX_MODEL
    if _PPX_FALLBACK:
        return _get_ppx_client(), ppx_model, client, model, "perplexity", is_final
    return _get_ppx_client(), ppx_model, None, None, "perplexity", is_final


# ── Per-call observability (provider/model/tokens/cost/latency/fallback) ────
# In-memory ring buffer + JSONL dump at exit (HLEO_LLM_CALL_LOG, default
# /tmp/hleo_llm_calls.jsonl). API keys are never recorded.
_CALL_LOG: list = []
_CALL_LOG_MAX = 5000
_CALL_LOG_PATH = os.getenv("HLEO_LLM_CALL_LOG", "/tmp/hleo_llm_calls.jsonl")


def _record_call(operation: str, provider: str, model: str, *,
                 latency_s: float, resp: Any = None, error: Optional[str] = None,
                 fallback: bool = False) -> None:
    try:
        usage = getattr(resp, "usage", None) if resp is not None else None
        cost = getattr(usage, "cost", None) if usage is not None else None
        if cost is not None and not isinstance(cost, (int, float)):
            try:
                cost = dict(cost)
            except Exception:
                cost = str(cost)
        entry = {
            "ts": round(time.time(), 3),
            "operation": operation,
            "provider": provider,
            "model": model,
            "latency_s": round(latency_s, 2),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "cost": cost,
            "fallback": fallback,
            "error": (str(error)[:200] if error else None),
        }
    except Exception:
        return
    _CALL_LOG.append(entry)
    if len(_CALL_LOG) > _CALL_LOG_MAX:
        del _CALL_LOG[: _CALL_LOG_MAX // 2]
    # Incremental flush: survives SIGKILL and lets cost monitoring watch the
    # file live. Failure to write never breaks the LLM path.
    try:
        with open(_CALL_LOG_PATH, "a") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass


def get_call_log() -> list:
    return list(_CALL_LOG)


def reset_call_log() -> None:
    _CALL_LOG.clear()


@atexit.register
def _dump_call_log() -> None:
    try:
        if _CALL_LOG:
            with open(_CALL_LOG_PATH, "a") as fh:
                for e in _CALL_LOG:
                    fh.write(json.dumps(e, default=str) + "\n")
    except Exception:
        pass


# ── Centralised LLM call ────────────────────────────────────────────────────

def _unwrap_provider(client: Any, model: str):
    """Return (raw_client, model, provider_name, fallback) when ``client`` is a
    core.llm_provider.LLMProvider; otherwise None (plain SDK client).

    Duck-typed (no import of core.llm_provider here) to avoid a module cycle
    and to keep plain OpenAI SDK clients — including the unittest MagicMock
    stand-ins used by legacy tests — on the legacy path: a raw client exposes
    ``.chat``, an LLMProvider exposes ``.client`` and has no ``.chat``.
    """
    if hasattr(client, "chat") or not hasattr(client, "client"):
        return None
    from core.llm_provider import resolve_model
    provider_name = getattr(client, "name", "openai") or "openai"
    resolved = resolve_model(provider_name, model)
    fb = getattr(client, "fallback", None)
    fallback = None
    if fb is not None:
        fallback = (fb.client, resolve_model(fb.name, model))
    return (client.client, resolved, provider_name, fallback)


def _provider_kwargs(provider_name: str, model: str, messages: list,
                     temperature: float, max_tokens: Optional[int],
                     response_format: Optional[dict], json_mode: bool) -> dict:
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if provider_name == "perplexity" and kwargs.get("max_tokens") is not None:
        kwargs["max_tokens"] = max(kwargs["max_tokens"], _PPX_MIN_MAX_TOKENS)
    # json_mode requests a bare JSON object (guard contract for call_llm_json);
    # Perplexity rejects response_format json_object → omitted there.
    if json_mode:
        if provider_name != "perplexity":
            kwargs["response_format"] = {"type": "json_object"}
    elif response_format is not None:
        if provider_name == "perplexity" and response_format.get("type") == "json_object":
            pass
        else:
            kwargs["response_format"] = response_format
    return kwargs


def _run_provider_loop(*, operation: str, raw_client: Any, model: str,
                       provider_name: str, fallback: Optional[tuple],
                       messages: list, temperature: float,
                       max_tokens: Optional[int], response_format: Optional[dict],
                       json_mode: bool):
    """Execute an LLMProvider call with the one-way fallback chain defined in
    core.llm_provider: primary → optional fallback, never back to primary.

    Each stage gets its own bounded retry budget (MAX_TOTAL_ATTEMPTS). A
    ``quota_exhausted`` error on the primary short-circuits to the fallback
    immediately; non-quota exhaustion consumes the stage budget before
    switching. Total attempts are bounded by stages × MAX_TOTAL_ATTEMPTS —
    the documented provider contract, not a duplicated retry layer.
    """
    stages: list = [(raw_client, model, provider_name)]
    if fallback is not None:
        fb_client, fb_model = fallback
        if fb_client is not None:
            stages.append((fb_client, fb_model, "openai"))

    last_exc: Optional[Exception] = None
    last_kind: str = "other"

    for stage_idx, (stage_client, stage_model, stage_provider) in enumerate(stages):
        is_last_stage = stage_idx == len(stages) - 1
        for attempt in range(MAX_TOTAL_ATTEMPTS):
            active_client, active_model, active_provider = stage_client, stage_model, stage_provider
            try:
                kwargs = _provider_kwargs(
                    active_provider, active_model, messages, temperature,
                    max_tokens, response_format, json_mode,
                )
                if active_provider == "local":
                    _ROUTING_STATS["local_calls"] += 1
                else:
                    _ROUTING_STATS["openai_calls"] += 1
                t0 = time.perf_counter()
                resp = active_client.chat.completions.create(**kwargs)
                _record_call(operation, active_provider, active_model,
                             latency_s=time.perf_counter() - t0, resp=resp,
                             fallback=stage_idx > 0)
                raw = resp.choices[0].message.content
                if raw is None:
                    raise ValueError("LLM returned null content.")
                if json_mode:
                    return _extract_json(raw)
                return raw

            except json.JSONDecodeError as exc:
                last_exc = exc
                _record_call(operation, active_provider, active_model,
                             latency_s=0.0, error=f"json_decode: {exc}",
                             fallback=stage_idx > 0)
                last_kind = "schema"
                remaining = MAX_TOTAL_ATTEMPTS - attempt - 1
                if remaining <= 0:
                    break
                delay = _backoff_delay(attempt)
                logger.warning(
                    "%s: JSON parse error on attempt %d/%d — retrying in %.1fs. %s",
                    operation, attempt + 1, MAX_TOTAL_ATTEMPTS, delay, str(exc)[:160],
                )
                time.sleep(delay)
                continue

            except Exception as exc:  # noqa: BLE001 — classify then decide
                last_exc = exc
                msg = _extract_openai_message(exc)
                if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    _record_call(operation, active_provider, active_model,
                                 latency_s=0.0, error=msg,
                                 fallback=stage_idx > 0)
                kind = classify_429(msg)
                if kind == "quota_exhausted":
                    if is_last_stage:
                        raise QuotaExhaustedError(
                            f"OpenAI credit/quota exhausted — API calls disabled. ({msg})"
                        ) from exc
                    # Primary quota → switch to fallback immediately, no retry.
                    logger.error(
                        "%s: %s quota exhausted — switching to fallback provider. %s",
                        operation, active_provider, msg,
                    )
                    _ROUTING_STATS["fallbacks"] += 1
                    break
                last_kind = kind
                remaining = MAX_TOTAL_ATTEMPTS - attempt - 1
                if remaining <= 0:
                    if not is_last_stage:
                        _ROUTING_STATS["fallbacks"] += 1
                        logger.warning(
                            "%s: %s exhausted after %d attempts — switching to fallback provider.",
                            operation, active_provider, MAX_TOTAL_ATTEMPTS,
                        )
                    break
                delay = _backoff_delay(attempt)
                logger.warning(
                    "%s: attempt %d/%d failed (%s) — retrying in %.1fs. %s",
                    operation, attempt + 1, MAX_TOTAL_ATTEMPTS, kind, delay,
                    msg[:160],
                )
                time.sleep(delay)

    raise LLMCallError(
        f"{operation} failed after {MAX_TOTAL_ATTEMPTS} attempts per provider "
        f"(last kind={last_kind}): {last_exc}"
    ) from last_exc


def call_llm(
    client: Any,
    *,
    messages: list[dict],
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    response_format: Optional[dict] = None,
    json_mode: bool = False,
    operation: str = "llm_call",
) -> str:
    """Call OpenAI chat.completions.create with a SINGLE, bounded retry policy.

    Returns the assistant message text. Raises QuotaExhaustedError (no retry)
    or LLMCallError (after MAX_TOTAL_ATTEMPTS) on failure.

    Callers MUST NOT add their own retry loop around this — that would breach
    the absolute cap. This is the only retry boundary in the whole project.
    """
    if json_mode and response_format is None:
        response_format = {"type": "json_object"}

    # LLMProvider (core.llm_provider) path: unwrap to the raw SDK client and
    # honour the one-way fallback chain + resolve_model mapping. A plain SDK
    # client (has .chat) stays on the legacy routing below unchanged.
    provider = _unwrap_provider(client, model)
    if provider is not None:
        raw_client, resolved_model, provider_name, fallback = provider
        return _run_provider_loop(
            operation=operation, raw_client=raw_client, model=resolved_model,
            provider_name=provider_name, fallback=fallback, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
            response_format=response_format, json_mode=json_mode,
        )

    client, model, fb_client, fb_model, route_name = _route(operation, client, model)
    is_final = False
    if route_name is None:
        client, model, fb_client, fb_model, route_name, is_final = _route_ppx(
            operation, client, model)

    last_exc: Optional[Exception] = None
    last_kind: str = "other"

    for attempt in range(MAX_TOTAL_ATTEMPTS):
        is_fallback = False
        # Last-resort: primary provider kept failing → final attempt on the
        # original (OpenAI) client. Absolute cap still holds.
        if fb_client is not None and attempt == MAX_TOTAL_ATTEMPTS - 1 and last_exc is not None:
            logger.warning(
                "%s: %s kept failing — final attempt falls back to OpenAI (%s)",
                operation, route_name or "provider", fb_model,
            )
            _ROUTING_STATS["fallbacks"] += 1
            active_client, active_model = fb_client, fb_model
            active_provider = "openai"
            is_fallback = True
        else:
            active_client, active_model = client, model
            active_provider = route_name or "openai"
        try:
            kwargs: dict = {
                "model": active_model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if active_provider == "perplexity" and kwargs.get("max_tokens") is not None:
                kwargs["max_tokens"] = max(kwargs["max_tokens"], _PPX_MIN_MAX_TOKENS)
            if response_format is not None and not is_fallback:
                # Perplexity rejects {"type": "json_object"} (400) → drop it;
                # _extract_json() parses the plain-text output downstream.
                if active_provider == "perplexity":
                    if response_format.get("type") == "json_object":
                        pass
                    else:
                        kwargs["response_format"] = response_format
                else:
                    kwargs["response_format"] = response_format
            if active_provider == "perplexity":
                disable = _PPX_DISABLE_SEARCH and not is_final
                if disable:
                    kwargs["extra_body"] = {"disable_search": True}

            if route_name == "local" and active_client is client:
                _ROUTING_STATS["local_calls"] += 1
            else:
                _ROUTING_STATS["openai_calls"] += 1
            t0 = time.perf_counter()
            resp = active_client.chat.completions.create(**kwargs)
            _record_call(operation, active_provider, active_model,
                         latency_s=time.perf_counter() - t0, resp=resp,
                         fallback=is_fallback)
            content = resp.choices[0].message.content
            if content is None:
                # Treat a null content as a transient malformed response.
                raise ValueError("LLM returned null content.")
            return content

        except Exception as exc:  # noqa: BLE001 — we classify, then decide
            last_exc = exc
            msg = _extract_openai_message(exc)
            if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
                _record_call(operation, active_provider, active_model,
                             latency_s=0.0, error=msg, fallback=is_fallback)
            kind = classify_429(msg)

            # Hard stop: account quota exhausted. No retry, ever.
            if kind == "quota_exhausted":
                logger.error(
                    "%s: OpenAI quota exhausted — not retrying. %s",
                    operation, msg,
                )
                raise QuotaExhaustedError(
                    f"OpenAI credit/quota exhausted — API calls disabled. ({msg})"
                ) from exc

            last_kind = kind
            remaining = MAX_TOTAL_ATTEMPTS - attempt - 1
            if remaining <= 0:
                break

            delay = _backoff_delay(attempt)
            logger.warning(
                "%s: attempt %d/%d failed (%s) — retrying in %.1fs. %s",
                operation, attempt + 1, MAX_TOTAL_ATTEMPTS, kind, delay,
                msg[:160],
            )
            time.sleep(delay)

    raise LLMCallError(
        f"{operation} failed after {MAX_TOTAL_ATTEMPTS} attempts "
        f"(last kind={last_kind}): {last_exc}"
    ) from last_exc


def call_llm_json(
    client: Any,
    *,
    messages: list[dict],
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    operation: str = "llm_json_call",
) -> dict:
    """Call OpenAI, parse JSON, with the same single bounded retry policy.

    JSON / schema validation failures ARE retryable (count toward the cap).
    On the final attempt the raw text is attached to the error so the caller
    can surface it.
    """
    client, model, fb_client, fb_model, route_name = _route(operation, client, model)
    is_final = False
    if route_name is None:
        client, model, fb_client, fb_model, route_name, is_final = _route_ppx(
            operation, client, model)

    # LLMProvider (core.llm_provider) path: unwrap to the raw SDK client and
    # honour the one-way fallback chain + resolve_model mapping. A plain SDK
    # client (has .chat) stays on the legacy routing below unchanged.
    provider = _unwrap_provider(client, model)
    if provider is not None:
        raw_client, resolved_model, provider_name, fallback = provider
        return _run_provider_loop(
            operation=operation, raw_client=raw_client, model=resolved_model,
            provider_name=provider_name, fallback=fallback, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
            response_format=None, json_mode=True,
        )

    last_exc: Optional[Exception] = None
    last_raw: str = ""
    last_kind: str = "other"

    for attempt in range(MAX_TOTAL_ATTEMPTS):
        is_fallback = False
        # Last-resort: primary provider kept failing → final attempt on OpenAI.
        if fb_client is not None and attempt == MAX_TOTAL_ATTEMPTS - 1 and last_exc is not None:
            logger.warning(
                "%s: %s returned unusable output — final attempt falls back to OpenAI (%s)",
                operation, route_name or "provider", fb_model,
            )
            _ROUTING_STATS["fallbacks"] += 1
            active_client, active_model = fb_client, fb_model
            active_provider = "openai"
            is_fallback = True
        else:
            active_client, active_model = client, model
            active_provider = route_name or "openai"
        try:
            kwargs: dict = {
                "model": active_model,
                "messages": messages,
                "temperature": temperature,
            }
            # json_object = constrained output on OpenAI/llama.cpp; Perplexity
            # rejects it (400) → dropped there, _extract_json() parses below.
            if active_provider != "perplexity":
                kwargs["response_format"] = {"type": "json_object"}
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if active_provider == "perplexity" and kwargs.get("max_tokens") is not None:
                kwargs["max_tokens"] = max(kwargs["max_tokens"], _PPX_MIN_MAX_TOKENS)
            if active_provider == "perplexity":
                disable = _PPX_DISABLE_SEARCH and not is_final
                if disable:
                    kwargs["extra_body"] = {"disable_search": True}

            if route_name == "local" and active_client is client:
                _ROUTING_STATS["local_calls"] += 1
            else:
                _ROUTING_STATS["openai_calls"] += 1
            t0 = time.perf_counter()
            resp = active_client.chat.completions.create(**kwargs)
            _record_call(operation, active_provider, active_model,
                         latency_s=time.perf_counter() - t0, resp=resp,
                         fallback=is_fallback)
            raw = resp.choices[0].message.content or ""
            last_raw = raw

            return _extract_json(raw)

        except json.JSONDecodeError as exc:
            # Schema/parse error — retryable, nudged by the cap.
            last_exc = exc
            _record_call(operation, active_provider, active_model,
                         latency_s=0.0, error=f"json_decode: {exc}",
                         fallback=is_fallback)
            last_kind = "schema"
            remaining = MAX_TOTAL_ATTEMPTS - attempt - 1
            if remaining <= 0:
                break
            delay = _backoff_delay(attempt)
            logger.warning(
                "%s: JSON parse error on attempt %d/%d — retrying in %.1fs. %s",
                operation, attempt + 1, MAX_TOTAL_ATTEMPTS, delay, str(exc)[:160],
            )
            time.sleep(delay)
            continue

        except Exception as exc:  # noqa: BLE001 — classify then decide
            last_exc = exc
            msg = _extract_openai_message(exc)
            _record_call(operation, active_provider, active_model,
                         latency_s=0.0, error=msg, fallback=is_fallback)
            kind = classify_429(msg)

            if kind == "quota_exhausted":
                logger.error(
                    "%s: OpenAI quota exhausted — not retrying. %s",
                    operation, msg,
                )
                raise QuotaExhaustedError(
                    f"OpenAI credit/quota exhausted — API calls disabled. ({msg})"
                ) from exc

            last_kind = kind
            remaining = MAX_TOTAL_ATTEMPTS - attempt - 1
            if remaining <= 0:
                break
            delay = _backoff_delay(attempt)
            logger.warning(
                "%s: attempt %d/%d failed (%s) — retrying in %.1fs. %s",
                operation, attempt + 1, MAX_TOTAL_ATTEMPTS, kind, delay,
                msg[:160],
            )
            time.sleep(delay)

    raise LLMCallError(
        f"{operation} failed after {MAX_TOTAL_ATTEMPTS} attempts "
        f"(last kind={last_kind}): {last_exc}"
    ) from last_exc
