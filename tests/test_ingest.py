"""Storage tests against a real PostgreSQL + pgvector (docker compose db locally, a service container in CI).

Each test runs in its own temporary schema, dropped afterwards, so real data is never touched.
"""

import hashlib
import os
import uuid

import numpy as np
import psycopg
import pymupdf
import pytest

from ingestion import db, ingest
from ingestion.parse import Page


class FakeEmbedder:
    """Deterministic stand-in for BGE: one token per word, and a fixed unit vector per text."""

    name = "fake-model@1"

    def count_tokens(self, text):
        return len(text.split())

    def vector(self, text):
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")
        v = np.random.default_rng(seed).standard_normal(768)
        return v / np.linalg.norm(v)

    def embed_passages(self, passages):
        return np.array([self.vector(p) for p in passages])


@pytest.fixture
def conn():
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


def make_pdf(path, pages):
    with pymupdf.open() as doc:
        for text in pages:
            page = doc.new_page()
            page.insert_htmlbox(page.rect + (72, 72, -72, -72), text.replace("\n", "<br>"))  # Unicode font
        doc.save(path)
    return path


ACT_PAGES = [
    "CHAPTER II\n10. What agreements are contracts.—All agreements are contracts if they are made by the "
    "free consent of parties competent to contract.",
    "11. Who are competent to contract.—Every person is competent to contract who is of the age of majority.",
]


def store(conn, path, matter="matter-a", key="act", title="The Indian Contract Act, 1872"):
    return ingest.ingest_document(conn, FakeEmbedder(), matter_id=matter, source_key=key, title=title, path=path)


def test_schema_can_be_applied_twice(conn):
    db.apply_schema(conn)


def test_stores_document_and_chunks_with_pages_headings_and_vectors(conn, tmp_path):
    path = make_pdf(tmp_path / "act.pdf", ACT_PAGES)

    assert store(conn, path) == 1

    doc = conn.execute("SELECT matter_id, source_key, title, pages, embedding_model FROM documents").fetchone()
    assert doc == ("matter-a", "act", "The Indian Contract Act, 1872", 2, "fake-model@1")
    heading, page_start, page_end, ocr, embedding = conn.execute(
        "SELECT heading, page_start, page_end, ocr, embedding FROM chunks").fetchone()
    assert heading == "CHAPTER II > 10. What agreements are contracts."
    assert (page_start, page_end, ocr) == (1, 2, False)
    assert embedding.to_numpy().shape == (768,)


def test_unchanged_document_is_skipped_and_changed_one_replaced(conn, tmp_path):
    path = make_pdf(tmp_path / "act.pdf", ACT_PAGES)
    store(conn, path)
    first_id = conn.execute("SELECT id FROM documents").fetchone()[0]

    assert store(conn, path) is None  # same file, same model: nothing to do
    make_pdf(path, ACT_PAGES + ["73. Compensation for loss or damage caused by breach of contract.—When a "
                                "contract has been broken, the party who suffers is entitled to compensation."])
    assert store(conn, path) == 1

    assert conn.execute("SELECT count(*), max(pages), min(id) FROM documents").fetchone() == (1, 3, first_id + 1)
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1  # old chunks gone, not duplicated


def test_nearest_neighbour_search_respects_the_matter(conn, tmp_path):
    path = make_pdf(tmp_path / "act.pdf", ACT_PAGES)
    store(conn, path, matter="matter-a")
    store(conn, path, matter="matter-b")
    target = conn.execute("SELECT embedding FROM chunks WHERE matter_id = 'matter-b'").fetchone()[0]

    rows = conn.execute(
        "SELECT matter_id, embedding <=> %s AS distance FROM chunks WHERE matter_id = %s ORDER BY distance",
        (target, "matter-a")).fetchall()

    assert [matter for matter, _ in rows] == ["matter-a"]  # matter-b's identical chunk is never returned


def test_chunks_from_ocr_pages_are_flagged(conn, tmp_path, monkeypatch):
    path = make_pdf(tmp_path / "scan.pdf", ["placeholder"])
    pages = [Page(1, "HEADNOTES\n" + "Typed page. " * 30, ocr=False), Page(2, "Scanned page. " * 30, ocr=True)]
    monkeypatch.setattr(ingest, "parse_pdf", lambda p: pages)
    monkeypatch.setattr(ingest, "CHUNK_TOKENS", 70)  # one page per chunk

    assert store(conn, path) == 2

    assert conn.execute("SELECT page_start, ocr FROM chunks ORDER BY chunk_index").fetchall() == [
        (1, False), (2, True)]


def test_deleting_a_document_deletes_its_chunks(conn, tmp_path):
    store(conn, make_pdf(tmp_path / "act.pdf", ACT_PAGES))

    conn.execute("DELETE FROM documents")

    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
