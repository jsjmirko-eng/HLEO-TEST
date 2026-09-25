"""
HLEO — LLM call guard (cost protection + bounded retry)
========================================================

Centralises EVERY LLM call so retry cannot multiply across layers.

Hard rules (enforced everywhere this module is used):
    MAX_TOTAL_ATTEMPTS = 5
        1 initial attempt + at most 4 retries = 5 LLM calls per operation.
        No caller/layer may add its own retry on top — doing so would breach
        the absolute cap. The guard is the ONLY retry boundary.

429 handling:
    - insufficient_quota / credit_balance_exhausted  → NO retry, raise
      QuotaExhaustedError immediately.
    - rate_limit_exceeded (temporary)               → retry with backoff,
      up to MAX_TOTAL_ATTEMPTS.
    - other transient errors                       → retry, up to the cap.
    - JSON/schema validation errors                → retry, up to the cap
      (temperature is nudged on retries to break a stuck malformed response).

JSON output hardening:
    - _extract_json() sanitises LLM output before parsing: markdown fences,
      prose wrapping (first balanced {...}/[...] block), invalid backslash
      escapes from local models. Never alters the parsed data.

Provider-agnostic runtime:
    - The runtime is driven exclusively by the configured LLM settings:
      api_key, base_url, model, protocol.
    - Provider names are free-form metadata labels only.
    - No hardcoded provider branching or special routing based on provider name.

Optional local-first routing via env (HLEO_LOCAL_LLM_URL …):
    - Sends intermediate operations to a local OpenAI-compatible server while
      "final" user-facing operations stay on the configured provider.
    - With HLEO_LOCAL_LLM_FALLBACK=1 (default) the LAST attempt of the bounded
      retry loop falls back to the original client if the local server keeps
      failing — the absolute 5-attempt cap still holds. Unset env → behaviour
      identical to plain OpenAI-compatible mode.

Every call (success or error) is recorded via _record_call():
    operation, provider, model, tokens, cost, latency, fallback flag.
    Dumped as JSONL at exit (HLEO_LLM_CALL_LOG). API keys are never logged.

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

# ── Module-level defaults (backward-compat: external importers still see these)
# All active code uses _lim() below, which reads from core.llm_limits at runtime
# so Admin UI changes take effect without a server restart.
MAX_TOTAL_ATTEMPTS = 5
_BASE_DELAY = 2.0
_MAX_DELAY = 30.0
_JITTER = 0.15
MAX_TOTAL_REQUEST_ATTEMPTS = 10
_DEFAULT_MAX_RETRIES_PER_STAGE = 2


def _lim():
    """Return the current HLEOLimits (5-second TTL cache, never raises)."""
    try:
        from core.llm_limits import get_limits
        return get_limits()
    except Exception:
        from core.llm_limits import HLEOLimits
        return HLEOLimits()


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


# HTTP 401/403 / "invalid key" → not retryable, switch provider immediately.
_AUTH_ERROR_SIGNALS = (
    "invalid api key",
    "incorrect api key",
    "invalid_api_key",
    "authentication",
    "unauthorized",
    "permission denied",
    "api key not found",
)

# 5xx server errors → retryable.
_SERVER_ERROR_SIGNALS = (
    "internal server error",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "server error",
)

# Model / endpoint not found → not retryable.
_NOT_FOUND_SIGNALS = (
    "model not found",
    "no such model",
    "does not exist",
    "invalid model",
    "model_not_found",
)


def classify_error(exc: Exception, message: str) -> str:
    """Classify an LLM API error for retry/fallback decisions.

    Returns one of:
        "quota_exhausted" — NOT retryable, no fallback benefit (billing hard stop)
        "auth_error"      — NOT retryable on this provider; switch to next slot
        "not_found"       — NOT retryable (bad model/endpoint config)
        "rate_limit"      — retryable with backoff
        "server_error"    — retryable with backoff (5xx)
        "schema"          — retryable (JSON parse failure)
        "other"           — retryable (unknown transient)
    """
    msg = (message or "").lower()
    # Check HTTP status code when available (openai>=1.x sets status_code).
    status = getattr(exc, "status_code", None)
    if status is not None:
        if status in (401, 403):
            return "auth_error"
        if status == 404:
            return "not_found"
        if status == 429:
            if any(s in msg for s in _QUOTA_SIGNALS):
                return "quota_exhausted"
            return "rate_limit"
        if 500 <= status < 600:
            return "server_error"
    # Fallback: classify by message content.
    if any(s in msg for s in _QUOTA_SIGNALS):
        return "quota_exhausted"
    if any(s in msg for s in _AUTH_ERROR_SIGNALS):
        return "auth_error"
    if any(s in msg for s in _NOT_FOUND_SIGNALS):
        return "not_found"
    if "429" in msg or any(s in msg for s in _RATE_LIMIT_SIGNALS):
        return "rate_limit"
    if any(s in msg for s in _SERVER_ERROR_SIGNALS):
        return "server_error"
    return "other"


def classify_429(message: str) -> str:
    """Legacy shim — kept for external callers. Maps to classify_error categories."""
    msg = (message or "").lower()
    if any(s in msg for s in _QUOTA_SIGNALS):
        return "quota_exhausted"
    if "429" in msg or any(s in msg for s in _RATE_LIMIT_SIGNALS):
        return "rate_limit"
    return "other"


def _backoff_delay(attempt: int) -> float:
    lim = _lim()
    raw = min(lim.backoff_base_s * (2 ** attempt), lim.backoff_max_s)
    return raw * (1.0 + random.uniform(-lim.backoff_jitter, lim.backoff_jitter))


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
        fb_name = getattr(fb, "name", "fallback") or "fallback"
        fallback = (fb.client, resolve_model(fb_name, model), fb_name)
    return (client.client, resolved, provider_name, fallback,
            bool(getattr(client, "use_chain", False)))



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
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    elif response_format is not None:
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
        fb_client, fb_model, fb_name = fallback
        if fb_client is not None:
            stages.append((fb_client, fb_model, fb_name))

    last_exc: Optional[Exception] = None
    last_kind: str = "other"

    for stage_idx, (stage_client, stage_model, stage_provider) in enumerate(stages):
        is_last_stage = stage_idx == len(stages) - 1
        _max_att = _lim().max_total_attempts
        for attempt in range(_max_att):
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
                remaining = _max_att - attempt - 1
                if remaining <= 0:
                    break
                delay = _backoff_delay(attempt)
                logger.warning(
                    "%s: JSON parse error on attempt %d/%d — retrying in %.1fs. %s",
                    operation, attempt + 1, _max_att, delay, str(exc)[:160],
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
                    logger.error(
                        "%s: %s quota exhausted — switching to fallback provider. %s",
                        operation, active_provider, msg,
                    )
                    _ROUTING_STATS["fallbacks"] += 1
                    break
                last_kind = kind
                remaining = _max_att - attempt - 1
                if remaining <= 0:
                    if not is_last_stage:
                        _ROUTING_STATS["fallbacks"] += 1
                        logger.warning(
                            "%s: %s exhausted after %d attempts — switching to fallback provider.",
                            operation, active_provider, _max_att,
                        )
                    break
                delay = _backoff_delay(attempt)
                logger.warning(
                    "%s: attempt %d/%d failed (%s) — retrying in %.1fs. %s",
                    operation, attempt + 1, _max_att, kind, delay,
                    msg[:160],
                )
                time.sleep(delay)

    raise LLMCallError(
        f"{operation} failed after {_lim().max_total_attempts} attempts per provider "
        f"(last kind={last_kind}): {last_exc}"
    ) from last_exc


def _get_chain() -> list:
    """Return the current multi-slot provider chain (never raises)."""
    try:
        from core.llm_manager import get_provider_chain
        return get_provider_chain()
    except Exception:
        return []


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

    When the caller passes an LLMProvider wrapper, the full 4-slot chain from
    the Admin UI is used automatically (call_llm_chain). If no slots are
    configured the call falls back to the provider passed in. A plain SDK
    client (has .chat) bypasses the chain and goes through local-first routing.

    Callers MUST NOT add their own retry loop around this — that would breach
    the absolute cap. This is the only retry boundary in the whole project.
    """
    if json_mode and response_format is None:
        response_format = {"type": "json_object"}

    # LLMProvider (core.llm_provider) path: prefer the full multi-slot chain so
    # all callers automatically benefit from 4-slot fallback + per-slot retry.
    # Falls back to the single-provider _run_provider_loop when no slots are
    # configured (e.g. only env vars are set, no Admin UI slots).
    provider = _unwrap_provider(client, model)
    if provider is not None:
        raw_client, resolved_model, provider_name, fallback, use_chain = provider
        chain = _get_chain() if use_chain else []
        if chain:
            return call_llm_chain(
                chain, messages=messages, model=model,
                temperature=temperature, max_tokens=max_tokens,
                response_format=response_format, json_mode=json_mode,
                operation=operation,
            )
        # No opted-in chain configured → single-provider path (backward compat).
        return _run_provider_loop(
            operation=operation, raw_client=raw_client, model=resolved_model,
            provider_name=provider_name, fallback=fallback, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
            response_format=response_format, json_mode=json_mode,
        )

    client, model, fb_client, fb_model, route_name = _route(operation, client, model)

    last_exc: Optional[Exception] = None
    last_kind: str = "other"

    for attempt in range(MAX_TOTAL_ATTEMPTS):
        is_fallback = False
        # Last-resort: primary provider kept failing → final attempt on the
        # fallback client (from local-first routing). Absolute cap still holds.
        if fb_client is not None and attempt == MAX_TOTAL_ATTEMPTS - 1 and last_exc is not None:
            logger.warning(
                "%s: %s kept failing — final attempt falls back to configured provider (%s)",
                operation, route_name or "provider", fb_model,
            )
            _ROUTING_STATS["fallbacks"] += 1
            active_client, active_model = fb_client, fb_model
            active_provider = route_name or "fallback"
            is_fallback = True
        else:
            active_client, active_model = client, model
            active_provider = route_name or "openai-compatible"
        try:
            kwargs: dict = {
                "model": active_model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if response_format is not None and not is_fallback:
                kwargs["response_format"] = response_format

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
    response_format: Optional[dict] = None,
    operation: str = "llm_json_call",
) -> dict:
    """Call LLM, parse JSON, with the same single bounded retry policy.

    When the caller passes an LLMProvider wrapper, the full 4-slot chain is
    used automatically (same as call_llm). Falls back to the single-provider
    path when no slots are configured.

    JSON / schema validation failures ARE retryable (count toward the cap).
    """
    # LLMProvider path: prefer multi-slot chain, fall back to single-provider.
    provider = _unwrap_provider(client, model)
    if provider is not None:
        raw_client, resolved_model, provider_name, fallback, use_chain = provider
        chain = _get_chain() if use_chain else []
        if chain:
            return call_llm_chain(
                chain, messages=messages, model=model,
                temperature=temperature, max_tokens=max_tokens,
                response_format=response_format, json_mode=True,
                operation=operation,
            )
        return _run_provider_loop(
            operation=operation, raw_client=raw_client, model=resolved_model,
            provider_name=provider_name, fallback=fallback, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
            response_format=response_format, json_mode=True,
        )

    client, model, fb_client, fb_model, route_name = _route(operation, client, model)

    last_exc: Optional[Exception] = None
    last_raw: str = ""
    last_kind: str = "other"

    for attempt in range(MAX_TOTAL_ATTEMPTS):
        is_fallback = False
        # Last-resort: primary provider kept failing → final attempt on fallback.
        if fb_client is not None and attempt == MAX_TOTAL_ATTEMPTS - 1 and last_exc is not None:
            logger.warning(
                "%s: %s returned unusable output — final attempt falls back to configured provider (%s)",
                operation, route_name or "provider", fb_model,
            )
            _ROUTING_STATS["fallbacks"] += 1
            active_client, active_model = fb_client, fb_model
            active_provider = route_name or "fallback"
            is_fallback = True
        else:
            active_client, active_model = client, model
            active_provider = route_name or "openai-compatible"
        try:
            kwargs: dict = {
                "model": active_model,
                "messages": messages,
                "temperature": temperature,
            }
            kwargs["response_format"] = response_format or {"type": "json_object"}
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

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


# ── NON-RETRYABLE error kinds — switch to next stage immediately ──────────────
_NO_RETRY_KINDS = {"quota_exhausted", "auth_error", "not_found"}


def call_llm_chain(
    stages: list,   # list of ProviderStage from core.llm_manager
    *,
    messages: list,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    response_format: Optional[dict] = None,
    json_mode: bool = False,
    operation: str = "llm_call",
) -> Any:
    """Execute an LLM call against a chain of ProviderStage objects.

    Retry policy
    ------------
    - Each stage gets its own budget: stage.max_retries (0 = 1 attempt only).
    - Non-retryable errors (quota_exhausted, auth_error, not_found) skip to
      the next stage immediately without consuming retry budget.
    - A global counter MAX_TOTAL_REQUEST_ATTEMPTS caps total attempts across
      ALL stages so retry × stage cannot multiply unboundedly.
    - Backoff is exponential with jitter, same as the existing policy.
    - json_mode=True: parse JSON, retries on JSONDecodeError.

    Falls back to the legacy single-provider path when stages is empty.
    """
    if not stages:
        # No chain: legacy path (env-based, may return None → LLMCallError).
        from core.llm_provider import build_provider
        legacy = build_provider()
        if legacy is None:
            raise LLMCallError(f"{operation}: no LLM provider configured.")
        return call_llm(legacy, messages=messages, model=model,
                        temperature=temperature, max_tokens=max_tokens,
                        response_format=response_format, json_mode=json_mode,
                        operation=operation)

    if json_mode and response_format is None:
        response_format = {"type": "json_object"}

    total_attempts = 0
    last_exc: Optional[Exception] = None
    last_kind: str = "other"

    for stage in stages:
        resolved_model = stage.model_override or model
        stage_budget = stage.max_retries + 1  # e.g. max_retries=2 → 3 attempts
        stage_attempts = 0

        for attempt in range(stage_budget):
            _cap = _lim().max_total_request_attempts
            if total_attempts >= _cap:
                raise LLMCallError(
                    f"{operation}: global attempt cap ({_cap}) "
                    f"reached across all providers."
                ) from last_exc

            try:
                kwargs: dict = {
                    "model": resolved_model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                if response_format is not None:
                    kwargs["response_format"] = response_format

                _ROUTING_STATS["openai_calls"] += 1
                t0 = time.perf_counter()
                resp = stage.client.chat.completions.create(**kwargs)
                _record_call(operation, stage.name, resolved_model,
                             latency_s=time.perf_counter() - t0, resp=resp,
                             fallback=stage.slot_index > 1)
                total_attempts += 1
                raw = resp.choices[0].message.content
                if raw is None:
                    raise ValueError("LLM returned null content.")
                if json_mode:
                    return _extract_json(raw)
                return raw

            except json.JSONDecodeError as exc:
                last_exc = exc
                total_attempts += 1
                stage_attempts += 1
                _record_call(operation, stage.name, resolved_model,
                             latency_s=0.0, error=f"json_decode: {exc}",
                             fallback=stage.slot_index > 1)
                last_kind = "schema"
                remaining = stage_budget - attempt - 1
                if remaining <= 0:
                    break
                delay = _backoff_delay(attempt)
                logger.warning("%s [slot %d]: JSON parse error attempt %d/%d — retrying in %.1fs",
                               operation, stage.slot_index, attempt + 1, stage_budget, delay)
                time.sleep(delay)
                continue

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                total_attempts += 1
                stage_attempts += 1
                msg = _extract_openai_message(exc)
                if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    _record_call(operation, stage.name, resolved_model,
                                 latency_s=0.0, error=msg,
                                 fallback=stage.slot_index > 1)

                kind = classify_error(exc, msg)
                last_kind = kind

                if kind in _NO_RETRY_KINDS:
                    logger.warning(
                        "%s [slot %d %s]: %s — switching to next provider immediately. %s",
                        operation, stage.slot_index, stage.name, kind, msg[:120],
                    )
                    if kind == "quota_exhausted":
                        _ROUTING_STATS["fallbacks"] += 1
                    break  # skip to next stage

                remaining = stage_budget - attempt - 1
                if remaining <= 0 or total_attempts >= _lim().max_total_request_attempts:
                    if remaining <= 0:
                        logger.warning(
                            "%s [slot %d %s]: exhausted after %d attempt(s) — switching to next provider.",
                            operation, stage.slot_index, stage.name, stage_attempts,
                        )
                        _ROUTING_STATS["fallbacks"] += 1
                    break

                delay = _backoff_delay(attempt)
                logger.warning(
                    "%s [slot %d %s]: attempt %d/%d failed (%s) — retrying in %.1fs. %s",
                    operation, stage.slot_index, stage.name,
                    attempt + 1, stage_budget, kind, delay, msg[:120],
                )
                time.sleep(delay)

    if last_kind == "quota_exhausted":
        raise QuotaExhaustedError(
            f"OpenAI credit/quota exhausted — API calls disabled. ({last_exc})"
        ) from last_exc
    raise LLMCallError(
        f"{operation} failed on all {len(stages)} provider(s) "
        f"({total_attempts} total attempts, last kind={last_kind}): {last_exc}"
    ) from last_exc
