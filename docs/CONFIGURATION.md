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
| Temp store | `TEMP_RESULTS_TTL`, `TEMP_STORE_CLEANUP_INTERVAL` | Optional |

## Database precedence

`core/database.py` reads:

1. `DATABASE_URL` (used directly, with the `postgresql://` → `postgresql+psycopg2://`
   driver rewrite applied automatically);
2. otherwise it composes a PostgreSQL URL from `POSTGRES_*` / `PG*`
   (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST`,
   `POSTGRES_PORT` with `hleo_admin` / `hleo_secure` / `hleo_db` /
   `localhost` / `5432` defaults).

Local scripts default to SQLite (`DATABASE_URL=sqlite:///./hleo.db`) so a
fresh machine works without PostgreSQL.

## Admin password hash

Generate with:

```bash
python -c "import bcrypt,os; print(bcrypt.hashpw(os.environ['PWD_PLAIN'].encode(), bcrypt.gensalt()).decode())"
```

Then set `HLEO_ADMIN_USERNAME` and `HLEO_ADMIN_PASSWORD_HASH` in `.env`.
