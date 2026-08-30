# HLEO — Troubleshooting

## Symptom → cause → fix

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'fastapi'` | Dependencies not installed | `bash scripts/install.sh` |
| `/health` returns 200 but scientific search returns 503 | No LLM provider key configured | Set `OPENAI_API_KEY` (or `PERPLEXITY_API_KEY`) in `.env` |
| `connection refused` on port 8000 | Backend not started | `bash scripts/start.sh`, then `bash scripts/health-check.sh` |
| Boot fails at Database stage | `DATABASE_URL` points to unreachable Postgres | For local use, set `DATABASE_URL=sqlite:///./hleo.db`; for Docker, ensure `db` service is healthy |
| `scripts/start.sh` says "Already running" | A previous instance is still alive | `bash scripts/stop.sh`, then `scripts/start.sh` |
| Stale pidfile but no process | Previous crash left the pidfile | `scripts/stop.sh` removes it, or delete `.hleo/hleo.pid` |
| Reddit collector returns `no_credentials` | Reddit OAuth2 not configured | Set `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET`; without them Reddit is skipped |
| Admin tab hidden | Admin not configured | Set `HLEO_ADMIN_USERNAME` + `HLEO_ADMIN_PASSWORD_HASH` |

## Health check statuses

`scripts/health-check.sh` prints one line per component:

- **PASS** — present and working
- **FAIL** — expected but unavailable/broken (non-zero exit)
- **NOT CONFIGURED** — optional and intentionally unset (not an error)

If any component is `FAIL`, the script exits non-zero and prints
`HLEO NOT READY`. Read the detail after the `—` to identify the failing
component.

## Docker-specific

- Port 5432 already in use → stop the local Postgres or change the compose
  port mapping.
- Build fails on `libpq-dev`/`gcc` → use a network with access to Debian
  repositories, or rely on the prebuilt `psycopg2-binary` wheel (the apt
  packages are a fallback).

## Notes

- `.env` is git-ignored; never commit it.
- `.env.example` intentionally contains no secrets.
- The obsolete backup archives (`HLEO-ultima-versione.zip`, `zipFile.zip`)
  are excluded by `.gitignore` and not part of the distribution.
