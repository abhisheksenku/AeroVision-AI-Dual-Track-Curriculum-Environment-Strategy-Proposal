"""
=============================================================================
AeroVision AI - Database Configuration & Session Factory
=============================================================================
SQLAlchemy SQLite configuration with automatic connection pooling and session lifecycle.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import NullPool

DB_DIR = "data"
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "aerovision.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# NullPool prevents thread-affinity lockups across FastAPI AnyIO worker threads
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=NullPool,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

def get_db():
    """FastAPI dependency for thread-safe database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
