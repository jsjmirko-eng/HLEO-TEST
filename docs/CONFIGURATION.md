# HLEO — Configuration

All configuration is done with environment variables. Copy the template and
fill in only what you need:

```bash
cp .env.example .env
```

`.env.example` documents every variable and contains **no secrets** (no keys,
tokens or passwords). `.env` is excluded from git.

## Categories

| Area | Variables | Required? |
|---|---|---|
| Database | `DATABASE_URL`, `POSTGRES_*`, `PG*` | No (SQLite default locally) |
| LLM provider | `HLEO_LLM_PROVIDER`, `OPENAI_API_KEY`, `PERPLEXITY_API_KEY`, `OPENAI_BASE_URL`, `HLEO_LLM_MODEL`, `HLEO_PERPLEXITY_*`, `HLEO_LOCAL_LLM_*` | Optional (LLM features return 503 when absent) |
| Admin auth | `HLEO_ADMIN_USERNAME`, `HLEO_ADMIN_PASSWORD_HASH`, `HLEO_ADMIN_TOKEN_TTL` | Optional (admin section disabled when unset) |
| Reddit | `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` | Optional |
| Vocabulary (Catena C) | `HLEO_VOCAB_ENABLED`, `HLEO_VOCAB_PROVIDERS`, `HLEO_VOCAB_CACHE_*`, `UMLS_API_KEY`, `HLEO_UMLS_API_KEY`, `HLEO_LOINC_*`, `HLEO_SNOMED_*` | Optional (defaults ON, keys only for UMLS/LOINC/SNOMED) |
| RWE | `HLEO_RWE_INTENT_SCORING`, `HLEO_OPENFDA_MAX_RESULTS` | Optional |
| HTTP input limits | `HLEO_MAX_BODY_BYTES`, `HLEO_MAX_QUERY_CHARS`, `HLEO_MAX_FIELD_CHARS`, `HLEO_MAX_BATCH_ITEMS`, `HLEO_MAX_CONTEXT_ITEMS`, `HLEO_MAX_PIPELINE_RESULTS` | Optional (safe defaults) |
| Rate limiting | `HLEO_RATE_LIMIT_BACKEND`, `HLEO_API_RATE_*` | Optional (`database` backend by default) |
| Temp store | `TEMP_RESULTS_TTL`, `TEMP_STORE_CLEANUP_INTERVAL` | Optional |

## Database precedence

`core/database.py` reads:

1. `DATABASE_URL` (used directly, with the `postgresql://` → `postgresql+psycopg2://`
   driver rewrite applied automatically);
2. otherwise it composes a PostgreSQL URL from `POSTGRES_*` / `PG*`
   (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST`,
   `POSTGRES_PORT`). Missing PostgreSQL components cause the local SQLite
   fallback (`sqlite:///./hleo.db`) instead of using an implicit password.

Local scripts default to SQLite (`DATABASE_URL=sqlite:///./hleo.db`) so a
fresh machine works without PostgreSQL.

## Admin password hash

Generate with:

```bash
python -c "import bcrypt,os; print(bcrypt.hashpw(os.environ['PWD_PLAIN'].encode(), bcrypt.gensalt()).decode())"
```

Then set `HLEO_ADMIN_USERNAME` and `HLEO_ADMIN_PASSWORD_HASH` in `.env`.

## HTTP input limits

The API applies configurable boundary limits before expensive work:

| Variable | Default | Applies to |
|---|---:|---|
| `HLEO_MAX_BODY_BYTES` | `1048576` | JSON bodies for `POST`, `PUT`, and `PATCH` |
| `HLEO_MAX_QUERY_CHARS` | `500` | Search and pipeline query strings |
| `HLEO_MAX_FIELD_CHARS` | `20000` | Text fields in extraction, Assistant, comparison, and synthesis payloads |
| `HLEO_MAX_BATCH_ITEMS` | `50` | `/rwe/extract-batch` items |
| `HLEO_MAX_CONTEXT_ITEMS` | `50` | Article/RWE/episode lists in Assistant and synthesis payloads |
| `HLEO_MAX_PIPELINE_RESULTS` | `100` | `/pipeline/run?max_results=` |

The process clamps these settings to safe upper bounds. Oversized HTTP bodies
return `413`; oversized fields, lists, and query parameters return `422`.

## Provider secret migration

Provider API keys are stored as authenticated Fernet ciphertext using
`HLEO_SECRET_KEY`. Legacy XOR/Base64 payloads are not decoded or accepted.
The startup migration reports such rows and an administrator must re-enter the
provider key in Admin settings; saving it writes the new Fernet format.

## Admin token lifecycle

The browser stores the stateless Admin Bearer token in `sessionStorage` under
`hleo_admin_token`, so it is cleared when the browser tab closes. It is sent
only through the `Authorization: Bearer` header. Server-side validation checks
the HMAC signature, configured username, and expiry.

## Logging and privacy

Application handlers apply centralized credential redaction to logs and the
LLM JSONL call log. User queries are represented by a non-reversible SHA-256
fingerprint in operational logs. The default database rate limiter shares
fixed-window buckets across API workers and replicas that use the same database.
Set `HLEO_RATE_LIMIT_BACKEND=memory` only for isolated single-process tests.
