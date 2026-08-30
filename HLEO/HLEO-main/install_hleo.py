import os
import sys
import subprocess
import time
from pathlib import Path

# ==========================================
# 1. DEFINIZIONE DELL'ARCHITETTURA (PAYLOAD)
# ==========================================
HLEO_FILES = {
    "hleo_v1/docker-compose.yml": '''version: '3.8'

services:
  db:
    image: postgres:15-alpine
    container_name: hleo_db
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 5s
      timeout: 5s
      retries: 5

  api:
    build: .
    container_name: hleo_api
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    env_file:
      - .env
    depends_on:
      db:
        condition: service_healthy

volumes:
  postgres_data:
''',

    "hleo_v1/Dockerfile": '''FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \n    PYTHONUNBUFFERED=1 \n    PYTHONPATH=/app
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
''',

    "hleo_v1/requirements.txt": '''fastapi==0.141.1
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
''',

    "hleo_v1/core/database.py": '''import os
from collections.abc import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from dotenv import load_dotenv

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER", "hleo_admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "hleo_secure")
POSTGRES_DB = os.getenv("POSTGRES_DB", "hleo_db")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

SQLALCHEMY_DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
''',

    "hleo_v1/core/models.py": '''from sqlalchemy import String, Text, DateTime, Float, Boolean, Integer, JSON
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime, timezone
from core.database import Base, engine

class RawSource(Base):
    __tablename__ = "hleo_raw_sources"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    episode_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    source_type: Mapped[str] = mapped_column(String, nullable=False)
    platform: Mapped[str] = mapped_column(String, nullable=False)
    external_url: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    post_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class ClinicalProfile(Base):
    __tablename__ = "hleo_clinical_profiles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    episode_id: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    final_category: Mapped[str] = mapped_column(String, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    adjudication_required: Mapped[bool] = mapped_column(Boolean, default=False)
    extracted_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    validation_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class AuditLog(Base):
    __tablename__ = "hleo_audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    episode_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    action: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    details: Mapped[str] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

Base.metadata.create_all(bind=engine)
''',

    "hleo_v1/core/schemas.py": '''from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class BaselineEnum(str, Enum):
    ASSENTE = "assente"
    LIEVE = "lieve"
    MODERATA = "moderata"
    ELEVATA = "elevata"
    NON_DEDUCIBILE = "non_deducibile"

class PostTreatmentEnum(str, Enum):
    TORNATA_COME_PRIMA = "tornata_come_prima"
    INFERIORE = "inferiore_a_prima"
    SUPERIORE = "superiore_a_prima"
    NON_STABILIZZATA = "non_stabilizzata"
    NON_DEDUCIBILE = "non_deducibile"

class ClinicalCategory(str, Enum):
    CAT_A = "A"
    CAT_B = "B"
    CAT_C = "C"
    CAT_D = "D"
    CAT_E = "E"

class EvidenceQuote(BaseModel):
    verbatim_text: str
    source_url: str
    post_date: str

class ClinicalStatus(BaseModel):
    value: str
    supporting_quotes: List[EvidenceQuote] = Field(default_factory=list)
    support_strength: float = Field(ge=0.0, le=1.0)

class ExtractedClinicalProfile(BaseModel):
    episode_id: str
    user_id: str
    baseline_status: ClinicalStatus
    post_treatment_status: ClinicalStatus
    conflict_detected: bool

class ValidationItemResult(BaseModel):
    is_valid: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None

class ValidationReport(BaseModel):
    passed_validation: bool
    errors: List[ValidationItemResult] = Field(default_factory=list)

class JudgeResult(BaseModel):
    episode_id: str
    assigned_category: ClinicalCategory
    adjudication_required: bool
    final_confidence_score: float
''',

    "hleo_v1/core/extractor.py": '''import os
import json
from openai import OpenAI
from core.schemas import ExtractedClinicalProfile
import logging

logger = logging.getLogger(__name__)

class LLMExtractor:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"

    def extract(self, timeline_json: str) -> ExtractedClinicalProfile:
        logger.info("Avvio estrazione LLM.")
        system_prompt = "Sei un estrattore clinico in tricologia. Estrai il profilo clinico nel formato JSON richiesto."
        schema = ExtractedClinicalProfile.model_json_schema()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Timeline: {timeline_json}\\nSchema: {json.dumps(schema)}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return ExtractedClinicalProfile.model_validate_json(response.choices[0].message.content)
''',

    "hleo_v1/core/validator.py": '''import re
from datetime import datetime, timezone
from typing import Dict
from core.schemas import ExtractedClinicalProfile, ValidationReport, ValidationItemResult

class HLEOValidator:
    MAX_LOOKBACK = 120

    @classmethod
    def validate(cls, profile: ExtractedClinicalProfile, raw_texts: Dict[str, str], ep_start: datetime) -> ValidationReport:
        errors = []
        quotes = profile.baseline_status.supporting_quotes + profile.post_treatment_status.supporting_quotes
        for q in quotes:
            if not re.match(r'^https?:\\/\\/', q.source_url):
                errors.append(ValidationItemResult(is_valid=False, error_code="VAL_E02", error_message="URL non valido."))
            try:
                q_date = datetime.fromisoformat(q.post_date.replace('Z', '+00:00'))
                if (ep_start - q_date).days > cls.MAX_LOOKBACK:
                    errors.append(ValidationItemResult(is_valid=False, error_code="VAL_E04", error_message="Baseline fuori finestra."))
            except ValueError:
                errors.append(ValidationItemResult(is_valid=False, error_code="VAL_E04", error_message="Data malformata."))
            raw = raw_texts.get(q.source_url, "")
            clean_verb = re.sub(r'\\s+', ' ', q.verbatim_text.strip())
            clean_raw = re.sub(r'\\s+', ' ', raw.strip())
            if clean_verb not in clean_raw:
                errors.append(ValidationItemResult(is_valid=False, error_code="VAL_E01", error_message="Citazione allucinata."))
        return ValidationReport(passed_validation=len(errors) == 0, errors=errors)
''',

    "hleo_v1/core/judge.py": '''from core.schemas import ClinicalCategory, JudgeResult, BaselineEnum, PostTreatmentEnum

class HLEOJudge:
    @staticmethod
    def evaluate(base: str, post: str, valid: bool, strength: float, conflict: bool, ep_id: str) -> JudgeResult:
        if not valid or conflict:
            return JudgeResult(episode_id=ep_id, assigned_category=ClinicalCategory.CAT_E, adjudication_required=True, final_confidence_score=0.0)
        cat = ClinicalCategory.CAT_E
        if base in [BaselineEnum.ASSENTE.value, BaselineEnum.LIEVE.value]:
            if post == PostTreatmentEnum.TORNATA_COME_PRIMA.value: cat = ClinicalCategory.CAT_A
            elif post in [PostTreatmentEnum.SUPERIORE.value, PostTreatmentEnum.NON_STABILIZZATA.value]: cat = ClinicalCategory.CAT_D
        elif base == BaselineEnum.MODERATA.value:
            if post == PostTreatmentEnum.TORNATA_COME_PRIMA.value: cat = ClinicalCategory.CAT_B
            elif post in [PostTreatmentEnum.SUPERIORE.value, PostTreatmentEnum.NON_STABILIZZATA.value]: cat = ClinicalCategory.CAT_C
        elif base == BaselineEnum.ELEVATA.value:
            cat = ClinicalCategory.CAT_C
        adj = cat == ClinicalCategory.CAT_E
        return JudgeResult(episode_id=ep_id, assigned_category=cat, adjudication_required=adj, final_confidence_score=strength)
''',

    "hleo_v1/api/main.py": '''from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from core.database import get_db, engine, Base
from core.models import ClinicalProfile, RawSource
import logging

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="HLEO API", version="1.0.0")
Base.metadata.create_all(bind=engine)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head><title>HLEO v1.0</title><script src="https://cdn.tailwindcss.com"></script></head>
    <body class="bg-gray-100 p-10"><h1 class="text-3xl font-bold text-blue-800">HLEO v1.0 In Esecuzione</h1>
    <p class="mt-4">L'ambiente backend e database è operativo.</p></body></html>
    """

@app.get("/health")
def health_check():
    return {"status": "ok"}
''',

    "hleo_v1/tests/test_pipeline.py": '''import pytest
from datetime import datetime, timezone
from core.validator import HLEOValidator
from core.schemas import ExtractedClinicalProfile, ClinicalStatus, EvidenceQuote
from core.judge import HLEOJudge, ClinicalCategory

def test_validator_detects_hallucination():
    profile = ExtractedClinicalProfile(
        episode_id="T1", user_id="U1", conflict_detected=False,
        baseline_status=ClinicalStatus(value="moderata", support_strength=0.9, supporting_quotes=[
            EvidenceQuote(verbatim_text="capelli perfetti", source_url="http://x.com", post_date="2023-01-01T12:00:00Z")
        ]),
        post_treatment_status=ClinicalStatus(value="tornata_come_prima", support_strength=0.9, supporting_quotes=[])
    )
    raw = {"http://x.com": "ho perso i capelli"}
    report = HLEOValidator.validate(profile, raw, datetime.now(timezone.utc))
    assert not report.passed_validation
    assert report.errors[0].error_code == "VAL_E01"

def test_judge_logic_cat_b():
    res = HLEOJudge.evaluate("moderata", "tornata_come_prima", True, 0.8, False, "T1")
    assert res.assigned_category == ClinicalCategory.CAT_B
    assert not res.adjudication_required
'''
}

# ==========================================
# 2. FUNZIONI DI INSTALLAZIONE
# ==========================================

def run_cmd(cmd, cwd=None, capture=False):
    """Esegue un comando shell."""
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=capture, text=True)
    return result

def check_docker():
    """Verifica che Docker sia installato e in esecuzione."""
    print("[*] Controllo requisiti di sistema (Docker)...")
    res = run_cmd("docker info", capture=True)
    if res.returncode != 0:
        print("\n[ERRORE CRITICO] Docker non trovato o non in esecuzione.")
        print("Assicurati di aver installato Docker Desktop e di averlo avviato.")
        sys.exit(1)
    print("[OK] Docker è operativo.")

def setup_env():
    """Chiede all'utente la chiave API e genera il file .env."""
    print("\n" + "="*50)
    print("CONFIGURAZIONE AMBIENTE HLEO v1.0")
    print("="*50)
    
    api_key = input("Inserisci la tua OPENAI_API_KEY: ").strip()
    if not api_key:
        print("[ERRORE] La API Key è obbligatoria per HLEO.")
        sys.exit(1)

    env_content = f"""POSTGRES_USER=hleo_admin
POSTGRES_PASSWORD=hleo_secure
POSTGRES_DB=hleo_db
POSTGRES_HOST=db
POSTGRES_PORT=5432
OPENAI_API_KEY={api_key}
MAX_BASELINE_LOOKBACK_DAYS=120
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin
"""
    return env_content

def build_files(env_content):
    """Genera l'alberatura del progetto sul disco."""
    print("\n[*] Creazione dei file sorgente...")
    
    # Aggiungi il file .env alla lista dei file da generare
    HLEO_FILES["hleo_v1/.env"] = env_content
    
    for filepath, content in HLEO_FILES.items():
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")
        print(f"  -> Creato: {filepath}")
    
    print("[OK] Struttura del progetto creata con successo.")

def start_docker():
    """Avvia i container tramite docker compose."""
    target_dir = Path.cwd() / "hleo_v1"
    print("\n[*] Avvio dell'infrastruttura Docker in background...")
    
    res = run_cmd("docker compose up --build -d", cwd=target_dir)
    if res.returncode != 0:
        print("\n[ERRORE] Fallimento durante la build di Docker. Controlla i log qui sopra.")
        sys.exit(1)
    
    print("[*] Attesa che il database e l'API siano pronti (10 secondi)...")
    time.sleep(10)
    print("[OK] Servizi avviati.")

def run_tests():
    """Lancia la suite di test all'interno del container API."""
    print("\n[*] Esecuzione automatica dei Test (pytest)...")
    target_dir = Path.cwd() / "hleo_v1"
    
    res = run_cmd("docker exec -it hleo_api pytest tests/ -v", cwd=target_dir)
    
    if res.returncode == 0:
        print("\n" + "="*50)
        print("🎉 SUCCESSO! HLEO v1.0 E' STATO INSTALLATO E VALIDATO.")
        print("="*50)
        print("Tutti i test sono passati. L'applicazione è in esecuzione.")
        print("-> Apri la dashboard: http://localhost:8000")
        print("-> Per fermare il sistema vai in 'hleo_v1' ed esegui: docker compose down")
    else:
        print("\n" + "="*50)
        print("❌ ERRORE DURANTE I TEST")
        print("="*50)
        print("Copia l'errore rosso stampato qui sopra e incollalo nella chat per ricevere la correzione.")

# ==========================================
# 3. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    try:
        check_docker()
        env_config = setup_env()
        build_files(env_config)
        start_docker()
        run_tests()
    except KeyboardInterrupt:
        print("\n[INFO] Installazione annullata dall'utente.")
        sys.exit(0)
