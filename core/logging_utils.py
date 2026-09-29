"""Centralized redaction helpers for application logs and diagnostics."""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Any


_SECRET_PATTERNS = (
    re.compile(
        r"(?i)(api[_-]?key|password|passwd|secret|access[_-]?token|refresh[_-]?token)"
        r"(\s*[:=]\s*)(['\"]?)[^\s,'\"}]+"
    ),
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s,]+"),
    re.compile(r"(?i)([?&](?:apiKey|api_key|token|secret|password)=)[^&\s]+"),
)


def redact_text(value: Any, *, max_length: int | None = None) -> str:
    """Redact common credential forms and optionally bound the result."""
    text = str(value)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(r"\1\2\3[REDACTED]" if pattern.groups >= 3 else r"\1[REDACTED]", text)
    if max_length is not None:
        text = text[:max_length]
    return text


def query_fingerprint(query: Any) -> str:
    """Return non-reversible query metadata suitable for correlation logs."""
    text = str(query or "")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"sha256:{digest}/chars:{len(text)}"


class SecretRedactionFilter(logging.Filter):
    """Redact dynamic values after logging interpolation."""

    def filter(self, record: logging.LogRecord) -> bool:
        rendered = record.getMessage()
        record.msg = redact_text(rendered)
        record.args = ()
        return True


def configure_redaction() -> None:
    """Attach the redaction filter to all currently configured handlers."""
    redactor = SecretRedactionFilter()
    root = logging.getLogger()
    for handler in root.handlers:
        if not any(isinstance(item, SecretRedactionFilter) for item in handler.filters):
            handler.addFilter(redactor)
