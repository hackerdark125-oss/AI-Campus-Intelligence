"""
config.py
Central configuration, loaded from environment variables (with sane local defaults).
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Database: defaults to local SQLite for zero-setup dev.
# In Docker/production, docker-compose sets DATABASE_URL to Postgres.
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/campus.db")

# JWT auth
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))

# Face recognition
DATASET_DIR = os.getenv("DATASET_DIR", str(BASE_DIR / "dataset"))
TRAINER_DIR = os.getenv("TRAINER_DIR", str(BASE_DIR / "trainer"))
TRAINER_FILE = os.path.join(TRAINER_DIR, "trainer.yml")
ID_MAP_FILE = os.path.join(TRAINER_DIR, "id_map.txt")
FACE_CONFIDENCE_THRESHOLD = float(os.getenv("FACE_CONFIDENCE_THRESHOLD", "65"))

# RAG / chatbot
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# Analytics
ATTENDANCE_RISK_THRESHOLD = float(os.getenv("ATTENDANCE_RISK_THRESHOLD", "75.0"))
