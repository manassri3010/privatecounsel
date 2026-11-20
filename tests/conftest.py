import os
import uuid

import psycopg
import pytest

from ingestion import db


@pytest.fixture
def conn():
    """A connection to real PostgreSQL + pgvector, working in a temporary schema that is dropped afterwards.

    Uses TEST_DATABASE_URL (set in CI) or DATABASE_URL from .env (the docker compose db locally).
    """
    try:
        conn = db.connect(os.environ.get("TEST_DATABASE_URL"))
    except psycopg.OperationalError as e:
        pytest.skip(f"PostgreSQL not reachable: {e}")
    schema = f"pytest_{uuid.uuid4().hex[:8]}"
    conn.execute(f"CREATE SCHEMA {schema}")
    conn.execute(f"SET search_path TO {schema}, public")  # public holds the vector type
    db.apply_schema(conn)
    yield conn
    conn.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.close()
