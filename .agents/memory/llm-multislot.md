---
name: LLM Multi-slot System
description: Architettura multi-provider a 4 slot implementata in FASE 2. Decisioni di design, struttura, compatibilità.
---

## Sistema implementato

### Nuovi file
- `core/llm_manager.py` — ProviderManager: legge slot dal DB, costruisce catena, migra legacy config
- `core/models.py` — aggiunta classe `LLMProviderSlot` (tabella `hleo_llm_provider_slots`)

### File modificati (addizioni solo)
- `core/llm_guard.py` — aggiunti: `classify_error()`, `call_llm_chain()`, `_NO_RETRY_KINDS`, `MAX_TOTAL_REQUEST_ATTEMPTS=10`, `_DEFAULT_MAX_RETRIES_PER_STAGE=2`, segnali `_AUTH_ERROR_SIGNALS/_SERVER_ERROR_SIGNALS/_NOT_FOUND_SIGNALS`
- `core/llm_provider.py` — aggiunto `build_provider_chain()` come wrapper di `get_provider_chain()`
- `api/admin.py` — aggiunti: `LLMSlotRequest`, `_slot_to_dict`, GET/PUT `/admin/llm-slots`, POST `/admin/llm-slots/{n}/test`
- `templates/index.html` — pannello `adminLlmConfigPanel` sostituito con UI a 4 slot collassabili

### Backward compatibility
- `build_provider()` NON modificato — continua a leggere hleo_llm_config
- `call_llm()` e `call_llm_json()` NON modificati
- `classify_429()` resta come shim legacy
- `LLMConfig` (hleo_llm_config) non eliminato
- Migrazione automatica: al primo accesso admin, se hleo_llm_config ha una riga, viene copiata in slot 1

## Politica di retry

- Errori non-retryable (`_NO_RETRY_KINDS`): `quota_exhausted`, `auth_error`, `not_found` → switch immediato al provider successivo
- Errori retryable: `rate_limit`, `server_error`, `schema`, `other` → retry con backoff esponenziale
- Per-slot budget: `max_retries` (0-4, default 2 → 3 tentativi totali per slot)
- Budget globale: `MAX_TOTAL_REQUEST_ATTEMPTS=10` → cap assoluto cross-slot (evita retry×stage moltiplicazione)

**Why:** la moltiplicazione retry×slot (e.g. 4 slot × 5 retry = 20 tentativi) era il problema principale da risolvere.

**How to apply:** usare `call_llm_chain(stages, ...)` per nuovi flussi; i flussi esistenti usano ancora `call_llm()` che è retrocompatibile.

## Wiring dei caller esistenti

`call_llm()` e `call_llm_json()` ora intercettano la chiamata quando ricevono un `LLMProvider`:
1. `_unwrap_provider()` rileva che è un `LLMProvider`
2. `_get_chain()` interroga `get_provider_chain()` (DB → slot abilitati in ordine di priorità)
3. Se chain non vuota → `call_llm_chain()` con politica multi-slot
4. Se chain vuota (nessuno slot configurato) → `_run_provider_loop()` con il provider passato (backward compat)

**Why:** tutti i caller esistenti (`article_extractor`, `patient_extractor`, `relational_search`, `orchestrator`, `rwe/*`) chiamano `build_provider()` + `call_llm()` — non è stato necessario modificarli.

## Admin endpoints
- `GET /admin/llm-slots` — ritorna lista 4 slot
- `PUT /admin/llm-slots/{1-4}` — salva uno slot
- `POST /admin/llm-slots/{1-4}/test` — testa connessione (senza persistere)

## DB
- Tabella: `hleo_llm_provider_slots` (creata da `Base.metadata.create_all` al startup FastAPI)
- Campi: slot_index (1-4, unique), enabled, name, protocol, api_key_encrypted, base_url, model, timeout_s, max_retries, rate_limit_rpm
