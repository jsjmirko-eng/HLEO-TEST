# HLEO — Dependencies

Exact runtime and library requirements, taken from the real files in this
repository. Nothing in this document is guessed.

## Runtime

| Component | Version / requirement | Source of truth |
|---|---|---|
| Python | **3.13** (CI) — Docker image uses **3.12-slim**; both are supported | `.github/workflows/umls-live.yml` (`3.13`), `Dockerfile` (`python:3.12-slim`) |
| Frontend | None — static HTML/CSS/JS served by FastAPI/Jinja2 | `templates/index.html`, `static/` |
| Node.js | **Not used** (no `package.json` tracked) | `git ls-files` |
| PostgreSQL | **15** (docker-compose), driver `psycopg2-binary` | `docker-compose.yml`, `requirements.txt` |
| SQLite | Optional local fallback (no service needed) | `core/database.py` (`DATABASE_URL`) |

## Python dependencies (`requirements.txt`, pinned)

```
fastapi==0.141.1
uvicorn[standard]==0.52.1
sqlalchemy==2.0.52
psycopg2-binary==2.9.12
pydantic==2.13.4
bcrypt==5.0.0
openai==3.0.0
python-dotenv==1.2.2
pytest==9.1.1
jinja2==3.1.6
requests==2.34.2
beautifulsoup4==4.15.0
praw==8.0.3
ddgs==9.14.4
httpx2==2.10.0
```

Every entry is used in code and is required by the runtime and tests.

## Package manager

- **pip** (`python -m pip install -r requirements.txt`)
- No `pyproject.toml`, `setup.py`, `Pipfile`, `poetry.lock` or `uv.lock` present.

## System dependencies

| Package | Why | Where |
|---|---|---|
| `libpq-dev` | PostgreSQL headers; build fallback for `psycopg2-binary` | `Dockerfile` |
| `gcc` | C compiler for any source builds | `Dockerfile` |

No other system packages are required (the DB is a separate Docker service).

## External services (network)

| Service | Purpose | Credential |
|---|---|---|
| PubMed (NCBI E-utilities) | Scientific search | none |
| Europe PMC REST | Scientific search | none |
| ClinicalTrials.gov API v2 | Scientific search | none |
| openFDA (`api.fda.gov`) | RWE pharmacovigilance (FAERS) | none |
| Calvizie.net RSS | RWE community (IT) | none |
| HairLossTalk.com RSS | RWE community (EN) | none |
| HairLossExperiences.com RSS | RWE community (EN) | none |
| MaladiesRaresInfo.org Atom | RWE community (FR) | none |
| Reddit (PRAW OAuth2) | RWE experiences | `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` |
| OpenAI API | LLM (default provider) | `OPENAI_API_KEY` |
| Perplexity Sonar API | LLM (optional provider) | `PERPLEXITY_API_KEY` |
| RxNav / MeSH / LOINC / ConceptNet / Wikidata / UMLS / SNOMED | Vocabulary layer (Catena C) | none except UMLS/LOINC/SNOMED keys |

All external providers degrade gracefully: when a credential is absent the
corresponding collector/provider is skipped and the endpoint reports it
(e.g. scientific search returns **503** without an LLM key — by design, no
silent fallback).

## Ports

| Port | Purpose |
|---|---|
| `8000` | HLEO API (uvicorn default in Docker / compose / scripts) |
| `5432` | PostgreSQL (docker-compose `db` service) |

## Start commands currently used

- Local runtime: `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000`
- Docker Compose: `docker compose up --build -d` (api on 8000)
- Docker image: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
