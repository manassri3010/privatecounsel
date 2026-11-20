"""Connect to PostgreSQL with pgvector and create the tables."""

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = Path(__file__).with_name("schema.sql")


class Settings(BaseSettings):
    # Same .env as the API. Real environment variables win, so CI and Docker can override it.
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    database_url: str = "postgresql://privatecounsel:change-me@localhost:5432/privatecounsel"


def connect(url=None):
    """Open an autocommit connection (transactions are explicit) that understands the vector type."""
    conn = psycopg.connect(url or Settings().database_url, autocommit=True, connect_timeout=5)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def apply_schema(conn):
    conn.execute(SCHEMA.read_text())
