import os
from collections.abc import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

load_dotenv()

# Prefer DATABASE_URL from the environment; fall back to component-specific values.
DATABASE_URL = os.getenv("DATABASE_URL") or (
    "postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}".format(
        user=os.getenv("PGUSER", os.getenv("POSTGRES_USER", "hleo_admin")),
        password=os.getenv("PGPASSWORD", os.getenv("POSTGRES_PASSWORD", "hleo_secure")),
        host=os.getenv("PGHOST", os.getenv("POSTGRES_HOST", "localhost")),
        port=os.getenv("PGPORT", os.getenv("POSTGRES_PORT", "5432")),
        db=os.getenv("PGDATABASE", os.getenv("POSTGRES_DB", "hleo_db")),
    )
)

# SQLAlchemy requires the psycopg2 driver prefix
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
