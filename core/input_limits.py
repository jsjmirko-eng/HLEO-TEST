"""Configurable HTTP input limits shared by the API boundary."""
from __future__ import annotations

import os

from starlette.types import ASGIApp, Message, Receive, Scope, Send


DEFAULT_MAX_BODY_BYTES = 1_048_576
DEFAULT_MAX_QUERY_CHARS = 500
DEFAULT_MAX_FIELD_CHARS = 20_000
DEFAULT_MAX_BATCH_ITEMS = 50
DEFAULT_MAX_CONTEXT_ITEMS = 50
DEFAULT_MAX_PIPELINE_RESULTS = 100


def _positive_int(name: str, default: int, maximum: int) -> int:
    try:
        return max(1, min(maximum, int(os.getenv(name, str(default)))))
    except (TypeError, ValueError):
        return default


def max_body_bytes() -> int:
    return _positive_int("HLEO_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES, 10 * 1024 * 1024)


MAX_QUERY_CHARS = _positive_int("HLEO_MAX_QUERY_CHARS", DEFAULT_MAX_QUERY_CHARS, 10_000)
MAX_FIELD_CHARS = _positive_int("HLEO_MAX_FIELD_CHARS", DEFAULT_MAX_FIELD_CHARS, 100_000)
MAX_BATCH_ITEMS = _positive_int("HLEO_MAX_BATCH_ITEMS", DEFAULT_MAX_BATCH_ITEMS, 500)
MAX_CONTEXT_ITEMS = _positive_int("HLEO_MAX_CONTEXT_ITEMS", DEFAULT_MAX_CONTEXT_ITEMS, 500)
MAX_PIPELINE_RESULTS = _positive_int(
    "HLEO_MAX_PIPELINE_RESULTS", DEFAULT_MAX_PIPELINE_RESULTS, 500
)


class BodyTooLarge(Exception):
    """Raised when a request body exceeds the configured byte limit."""


class BodySizeLimitMiddleware:
    """Reject oversized request bodies, including chunked bodies."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http" or scope.get("method") not in {
            "POST", "PUT", "PATCH"
        }:
            await self.app(scope, receive, send)
            return

        limit = max_body_bytes()
        content_length = _content_length(scope)
        if content_length is not None and content_length > limit:
            await _send_too_large(send, limit)
            return

        total = 0

        async def limited_receive() -> Message:
            nonlocal total
            message = await receive()
            if message.get("type") == "http.request":
                total += len(message.get("body", b""))
                if total > limit:
                    raise BodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except BodyTooLarge:
            await _send_too_large(send, limit)


def _content_length(scope: Scope) -> int | None:
    for key, value in scope.get("headers", []):
        if key.lower() == b"content-length":
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return None
    return None


async def _send_too_large(send: Send, limit: int) -> None:
    body = b'{"detail":"Request body exceeds the configured size limit."}'
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"retry-after", b"0"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
