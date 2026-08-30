"""
HLEO Admin authentication — secure HTTP Basic Auth + stateless login token.

No sessions, no users table, no hardcoded secrets. The admin identity is read
from environment variables:
  - HLEO_ADMIN_USERNAME      (default: none → admin disabled)
  - HLEO_ADMIN_PASSWORD_HASH  (bcrypt hash; the plaintext is NEVER stored)
  - HLEO_ADMIN_TOKEN_TTL      (optional, seconds; default 28800 = 8h)

If either HLEO_ADMIN_USERNAME / HLEO_ADMIN_PASSWORD_HASH is unset, the admin
section is completely disabled (require_admin returns 404 on every /admin/*
route — the section does not exist as far as an attacker can tell).

Two authentication mechanisms are supported on every protected /admin/* route:
  1. HTTP Basic Auth (Authorization: Basic ...)        — backward compatible
  2. Bearer token   (Authorization: Bearer <token>)   — for the web UI login

The Bearer token is a stateless HMAC-SHA256 signed payload (username + expiry).
The HMAC key is the bcrypt password hash itself, which never leaves the server.
Changing the admin password invalidates all outstanding tokens automatically.

POST /admin/login validates {username, password} against the same bcrypt hash
and returns {token, expires_at}. The frontend stores the token in
sessionStorage (cleared when the browser tab closes) and sends it as a Bearer
header on all subsequent /admin/* requests.

Generate a password hash with:
  python -c "import bcrypt,os; print(bcrypt.hashpw(os.environ['PWD_PLAIN'].encode(),bcrypt.gensalt()).decode())"
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.security.api_key import APIKeyHeader

_security = HTTPBasic(auto_error=False)
_bearer = APIKeyHeader(name="Authorization", auto_error=False, scheme_name="Bearer")

# Default token lifetime: 8 hours (a workday).
_DEFAULT_TOKEN_TTL = 28800


def _admin_config() -> tuple[Optional[str], Optional[str]]:
    """Return (username, bcrypt_hash) from env, or (None, None) if disabled."""
    user = os.getenv("HLEO_ADMIN_USERNAME", "").strip()
    pw_hash = os.getenv("HLEO_ADMIN_PASSWORD_HASH", "").strip()
    if not user or not pw_hash:
        return None, None
    return user, pw_hash


def is_admin_enabled() -> bool:
    u, h = _admin_config()
    return bool(u and h)


def _token_ttl() -> int:
    try:
        return max(60, int(os.getenv("HLEO_ADMIN_TOKEN_TTL", str(_DEFAULT_TOKEN_TTL))))
    except (ValueError, TypeError):
        return _DEFAULT_TOKEN_TTL


# ── Credential verification (shared by Basic Auth and /admin/login) ──────────

def _verify_credentials(username: str, password: str) -> bool:
    """Constant-time username compare + bcrypt password check."""
    configured_user, pw_hash = _admin_config()
    if not configured_user or not pw_hash:
        return False
    user_ok = hmac.compare_digest(username.encode(), configured_user.encode())
    if not user_ok:
        return False
    try:
        import bcrypt
        return bcrypt.checkpw(password.encode(), pw_hash.encode())
    except (ValueError, ImportError):
        return False


# ── Stateless HMAC token ─────────────────────────────────────────────────────

def _hmac_key() -> bytes:
    """HMAC signing key = the bcrypt password hash (never exposed to clients)."""
    _, pw_hash = _admin_config()
    return (pw_hash or "").encode()


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _ub64(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def create_admin_token(username: str) -> tuple[str, int]:
    """Return (token, expires_at_unix). Signed with the bcrypt hash."""
    exp = int(time.time()) + _token_ttl()
    payload = json.dumps({"u": username, "exp": exp}, separators=(",", ":")).encode()
    sig = hmac.new(_hmac_key(), payload, hashlib.sha256).digest()
    token = _b64(payload) + "." + _b64(sig)
    return token, exp


def verify_admin_token(token: str) -> Optional[str]:
    """Return the username if the token is valid and not expired, else None."""
    try:
        payload_b64, sig_b64 = token.split(".", 1)
    except ValueError:
        return None
    try:
        payload = _ub64(payload_b64)
        sig = _ub64(sig_b64)
    except Exception:
        return None
    expected = hmac.new(_hmac_key(), payload, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        data = json.loads(payload)
        username = data["u"]
        exp = int(data["exp"])
    except (ValueError, KeyError, TypeError):
        return None
    if exp < time.time():
        return None
    # Reject if admin config changed (username no longer matches).
    configured_user, _ = _admin_config()
    if not configured_user or not hmac.compare_digest(username.encode(), configured_user.encode()):
        return None
    return username


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip() or None
    return None


def require_admin(
    request: Request,
    credentials: Optional[HTTPBasicCredentials] = Depends(_security),
    authorization: Optional[str] = Depends(_bearer),
):
    """FastAPI dependency that gates every /admin/* route.

    - If admin is not configured → 404 (section does not exist).
    - Accepts Bearer token (web UI) OR Basic Auth (backward compat).
    - If neither valid → 401 with WWW-Authenticate (so Basic Auth still works).
    """
    username, pw_hash = _admin_config()
    if not username or not pw_hash:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")

    # 1) Bearer token (preferred for the web UI).
    token = _extract_bearer(authorization)
    if token:
        tok_user = verify_admin_token(token)
        if tok_user:
            return tok_user
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired admin token.",
            headers={"WWW-Authenticate": 'Bearer realm="HLEO Admin"'},
        )

    # 2) Basic Auth (backward compatible — API clients, curl).
    if credentials is not None:
        if _verify_credentials(credentials.username, credentials.password):
            return credentials.username
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials.",
            headers={"WWW-Authenticate": 'Basic realm="HLEO Admin"'},
        )

    # 3) No credentials at all.
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Admin credentials required.",
        headers={"WWW-Authenticate": 'Basic realm="HLEO Admin"'},
    )


def verify_password(plain: str) -> bool:
    """Helper for tests / CLI: verify a plaintext against the configured hash."""
    configured_user, pw_hash = _admin_config()
    if not configured_user or not pw_hash:
        return False
    return _verify_credentials(configured_user, plain)


def hash_password(plain: str) -> str:
    """Helper: produce a bcrypt hash for storing in HLEO_ADMIN_PASSWORD_HASH."""
    import bcrypt
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()
