"""Tests for the Admin authentication flow (token-based login + Basic Auth).

Covers the required scenarios:
  - unauthenticated user → /admin/status and /admin/sources return 401
  - wrong credentials → /admin/login returns 401
  - correct credentials → /admin/login returns 200 + token
  - /admin/status with Bearer token → 200
  - /admin/sources with Bearer token → 200 (sources load)
  - admin tab visibility is gated by /admin/ping (admin_enabled)
  - no credentials hardcoded in the client-side code
  - Basic Auth still works (backward compatibility)
  - expired/tampered token → 401
"""
import os
import time

import pytest

from core.admin_auth import create_admin_token, verify_admin_token


# ── Fixtures ─────────────────────────────────────────────────────────────────

_TEST_USER = "hleo-admin"
_TEST_PASSWORD = "test-admin-password-123"


@pytest.fixture()
def admin_enabled(monkeypatch):
    """Enable the admin section with a known bcrypt hash for the test."""
    from core.admin_auth import hash_password
    pw_hash = hash_password(_TEST_PASSWORD)
    monkeypatch.setenv("HLEO_ADMIN_USERNAME", _TEST_USER)
    monkeypatch.setenv("HLEO_ADMIN_PASSWORD_HASH", pw_hash)
    return pw_hash


@pytest.fixture()
def admin_disabled(monkeypatch):
    """Disable the admin section (no env vars)."""
    monkeypatch.delenv("HLEO_ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("HLEO_ADMIN_PASSWORD_HASH", raising=False)


@pytest.fixture()
def admin_token(admin_enabled):
    """Return a valid Bearer token (depends on admin_enabled)."""
    token, _ = create_admin_token(_TEST_USER)
    return token


# ── 1. Unauthenticated → Admin not accessible ───────────────────────────────

def test_unauthenticated_status_returns_401(client, admin_enabled):
    r = client.get("/admin/status")
    assert r.status_code == 401


def test_unauthenticated_sources_returns_401(client, admin_enabled):
    r = client.get("/admin/sources")
    assert r.status_code == 401


def test_ping_is_public(client, admin_enabled):
    """The ping endpoint must NOT require auth (frontend uses it to show the tab)."""
    r = client.get("/admin/ping")
    assert r.status_code == 200
    assert r.json() == {"admin_enabled": True}


# ── 2. Wrong credentials → rejected ─────────────────────────────────────────

def test_login_wrong_password_returns_401(client, admin_enabled):
    r = client.post("/admin/login", json={
        "username": _TEST_USER, "password": "wrong-password",
    })
    assert r.status_code == 401


def test_login_wrong_username_returns_401(client, admin_enabled):
    r = client.post("/admin/login", json={
        "username": "not-an-admin", "password": _TEST_PASSWORD,
    })
    assert r.status_code == 401


def test_login_empty_body_returns_422(client, admin_enabled):
    r = client.post("/admin/login", json={})
    assert r.status_code == 422


# ── 3. Correct credentials → login 200 + token ──────────────────────────────

def test_login_correct_returns_token(client, admin_enabled):
    r = client.post("/admin/login", json={
        "username": _TEST_USER, "password": _TEST_PASSWORD,
    })
    assert r.status_code == 200
    data = r.json()
    assert "token" in data
    assert data["token"]
    assert data["username"] == _TEST_USER
    assert "expires_at" in data
    # Token must NOT contain the password or hash.
    assert _TEST_PASSWORD not in data["token"]
    assert "$2b$" not in data["token"]


# ── 4. /admin/status with Bearer token → 200 ────────────────────────────────

def test_status_with_bearer_token_returns_200(client, admin_token):
    r = client.get("/admin/status", headers={
        "Authorization": f"Bearer {admin_token}",
    })
    assert r.status_code == 200
    assert r.json()["authenticated"] is True


# ── 5. /admin/sources with Bearer token → 200 (sources load) ────────────────

def test_sources_with_bearer_token_returns_200(client, admin_token):
    r = client.get("/admin/sources", headers={
        "Authorization": f"Bearer {admin_token}",
    })
    assert r.status_code == 200
    data = r.json()
    assert "sources" in data
    assert data["total"] >= 1  # runtime sources are seeded on first access


# ── 6. Admin tab visibility gated by /admin/ping ────────────────────────────

def test_ping_shows_disabled_when_admin_not_configured(client, admin_disabled):
    r = client.get("/admin/ping")
    assert r.status_code == 200
    assert r.json() == {"admin_enabled": False}


def test_admin_routes_404_when_disabled(client, admin_disabled):
    """When admin is not configured, routes return 404 (section hidden)."""
    assert client.get("/admin/status").status_code == 404
    assert client.get("/admin/sources").status_code == 404
    assert client.post("/admin/login", json={
        "username": "x", "password": "y",
    }).status_code == 404


# ── 7. No credentials hardcoded in client code ──────────────────────────────

def test_no_hardcoded_credentials_in_template():
    """The client-side HTML/JS must NOT contain any hardcoded admin password."""
    template_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "templates", "index.html",
    )
    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()
    # The test password must not appear anywhere in the template.
    assert _TEST_PASSWORD not in content
    # No bcrypt hash should be embedded in the client.
    assert "$2b$" not in content
    # The login form sends user input — no pre-filled credential values.
    assert 'value="' + _TEST_PASSWORD + '"' not in content


def test_no_hardcoded_credentials_in_backend():
    """Backend auth modules must NOT hardcode credentials — only read from env."""
    auth_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "core", "admin_auth.py",
    )
    with open(auth_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert _TEST_PASSWORD not in content
    # No literal password strings assigned.
    for forbidden in ['password = "', 'PASSWORD = "', "password = '"]:
        assert forbidden not in content, f"Found: {forbidden}"


# ── 8. Basic Auth backward compatibility ────────────────────────────────────

def test_basic_auth_still_works_for_status(client, admin_enabled):
    r = client.get("/admin/status", auth=(_TEST_USER, _TEST_PASSWORD))
    assert r.status_code == 200


def test_basic_auth_still_works_for_sources(client, admin_enabled):
    r = client.get("/admin/sources", auth=(_TEST_USER, _TEST_PASSWORD))
    assert r.status_code == 200


def test_basic_auth_wrong_password_returns_401(client, admin_enabled):
    r = client.get("/admin/status", auth=(_TEST_USER, "wrong"))
    assert r.status_code == 401


# ── 9. Token security: tampered / expired tokens rejected ───────────────────

def test_tampered_token_rejected(client, admin_token):
    """A token with a modified signature must be rejected."""
    parts = admin_token.split(".")
    tampered = parts[0] + "." + ("A" * len(parts[1]))
    r = client.get("/admin/status", headers={"Authorization": f"Bearer {tampered}"})
    assert r.status_code == 401


def test_garbage_token_rejected(client, admin_enabled):
    r = client.get("/admin/status", headers={"Authorization": "Bearer not-a-real-token"})
    assert r.status_code == 401


def test_empty_bearer_rejected(client, admin_enabled):
    r = client.get("/admin/status", headers={"Authorization": "Bearer "})
    assert r.status_code == 401


def test_expired_token_rejected(client, monkeypatch):
    """A token whose expiry is in the past must be rejected."""
    from core.admin_auth import hash_password
    pw_hash = hash_password(_TEST_PASSWORD)
    monkeypatch.setenv("HLEO_ADMIN_USERNAME", _TEST_USER)
    monkeypatch.setenv("HLEO_ADMIN_PASSWORD_HASH", pw_hash)
    monkeypatch.setenv("HLEO_ADMIN_TOKEN_TTL", "60")

    # Create a token, then move the clock past its expiry.
    token, _ = create_admin_token(_TEST_USER)
    assert verify_admin_token(token) == _TEST_USER
    import core.admin_auth as aa
    real_time = aa.time.time
    try:
        aa.time.time = lambda: real_time() + 9999
        assert verify_admin_token(token) is None
    finally:
        aa.time.time = real_time


def test_logout_clears_client_session(client, admin_token):
    """The logout endpoint accepts a valid token and returns 200."""
    r = client.post("/admin/logout", headers={
        "Authorization": f"Bearer {admin_token}",
    })
    assert r.status_code == 200
    assert r.json() == {"logged_out": True}


# ── 10. Token unit tests ────────────────────────────────────────────────────

def test_token_does_not_contain_password(admin_token):
    assert _TEST_PASSWORD not in admin_token


def test_token_does_not_contain_hash(admin_enabled, admin_token):
    pw_hash = admin_enabled
    assert pw_hash not in admin_token


def test_token_roundtrip(admin_enabled):
    token, exp = create_admin_token(_TEST_USER)
    assert verify_admin_token(token) == _TEST_USER
    assert exp > int(time.time())
