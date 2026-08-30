#!/usr/bin/env python3
"""
E2E test: Admin API saves free-form provider names (OpenRouter, Groq, etc.)
Verifies that:
1. Provider field accepts arbitrary name via Admin API
2. Config persists in DB
3. API returns config without exposing plaintext key
4. Multiple providers can be saved and retrieved
"""

import sqlite3
import time
from pathlib import Path

import requests

# Hardcoded test credentials (same as app defaults)
ADMIN_USER = "admin"
ADMIN_PASS = "admin"
BASE_URL = "http://localhost:8000"

def cleanup_db():
    """Clear LLMConfig from test database before test."""
    db_path = Path("hleo.db")
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("DELETE FROM hleo_llm_config")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Table doesn't exist yet
        finally:
            conn.close()

def test_freeform_provider_names():
    """Test multiple free-form provider names work end-to-end."""
    
    # Clean up
    cleanup_db()
    time.sleep(0.5)
    
    # Step 1: Login
    print("\n1️⃣  Admin login...")
    resp = requests.post(f"{BASE_URL}/admin/login", json={
        "username": ADMIN_USER,
        "password": ADMIN_PASS
    })
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    token = resp.json()["token"]
    print(f"   ✓ Authenticated as {ADMIN_USER}")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test providers
    test_cases = [
        {
            "name": "OpenRouter",
            "api_key": "sk-or-test-12345678",
            "base_url": "https://openrouter.ai/api/v1",
            "model": "openrouter/auto"
        },
        {
            "name": "Groq",
            "api_key": "gsk-test-key",
            "base_url": "https://api.groq.com/openai/v1",
            "model": "mixtral-8x7b-32768"
        },
        {
            "name": "Azure OpenAI",
            "api_key": "azure-key-test",
            "base_url": "https://myinstance.openai.azure.com/",
            "model": "gpt-4"
        },
        {
            "name": "CustomLLM",
            "api_key": "custom-key",
            "base_url": "http://localhost:5001/v1",
            "model": "local-model"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        provider_name = test_case["name"]
        print(f"\n{i}️⃣  Test provider: {provider_name}")
        
        # Save config
        payload = {
            "provider": provider_name,
            "api_key": test_case["api_key"],
            "base_url": test_case["base_url"],
            "model": test_case["model"]
        }
        resp = requests.put(f"{BASE_URL}/admin/llm-config", headers=headers, json=payload)
        assert resp.status_code == 200, f"Save failed: {resp.text}"
        saved = resp.json()
        
        # Verify response
        assert saved.get("provider").lower() == provider_name.lower(), \
            f"Expected '{provider_name}', got {saved.get('provider')}"
        assert saved.get("api_key_configured") is True, "API key should be configured"
        assert saved.get("base_url") == test_case["base_url"], "Base URL mismatch"
        assert saved.get("model") == test_case["model"], "Model mismatch"
        
        # CRITICAL: API must NOT expose plaintext key
        assert test_case["api_key"] not in str(saved), \
            f"API must not expose plaintext API key: {test_case['api_key']}"
        
        print(f"   ✓ Saved provider: {saved.get('provider')}")
        print(f"   ✓ API key hidden: {saved.get('api_key_configured')}")
        print(f"   ✓ Base URL: {saved.get('base_url')}")
        print(f"   ✓ Model: {saved.get('model')}")
    
    # Step 2: Verify database has last config (CustomLLM)
    # Note: The Admin LLM config maintains a SINGLE active configuration.
    # Each PUT overwrites the previous one (update, not insert).
    print("\n✅ Verify database persistence...")
    conn = sqlite3.connect("hleo.db")
    rows = conn.execute(
        "SELECT provider, base_url, model FROM hleo_llm_config ORDER BY id DESC LIMIT 5"
    ).fetchall()
    conn.close()
    
    # Should have at most 1 row (the latest config overwrites previous ones)
    last_row = rows[0] if rows else None
    assert last_row is not None, "No config found in database"
    
    last_provider, last_url, last_model = last_row
    expected_name = "CustomLLM"
    assert last_provider.lower() == expected_name.lower(), \
        f"Expected last provider '{expected_name}', got {last_provider}"
    
    print(f"   ✓ Latest config in DB: {last_provider}")
    print(f"   ✓ Base URL: {last_url}")
    print(f"   ✓ Model: {last_model}")
    
    # Step 3: Final refetch and verify last config
    print("\n✅ Verify final config refetch...")
    resp = requests.get(f"{BASE_URL}/admin/llm-config", headers=headers)
    assert resp.status_code == 200
    final = resp.json()
    
    assert final.get("provider").lower() == "customllm", \
        f"Expected final provider 'CustomLLM', got {final.get('provider')}"
    assert final.get("base_url") == "http://localhost:5001/v1"
    assert final.get("model") == "local-model"
    
    print(f"   ✓ Final provider: {final.get('provider')}")
    print(f"   ✓ Base URL: {final.get('base_url')}")
    print(f"   ✓ Model: {final.get('model')}")
    
    print("\n✅ All tests passed! Free-form provider names work end-to-end.")
    print("   - OpenRouter ✓")
    print("   - Groq ✓")
    print("   - Azure OpenAI ✓")
    print("   - CustomLLM ✓")

if __name__ == "__main__":
    try:
        test_freeform_provider_names()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
