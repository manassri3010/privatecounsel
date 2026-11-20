"""Vector search against real PostgreSQL + pgvector (see conftest.py)."""

import numpy as np
import pytest

from fakes import FakeEmbedder
from retrieval.search import search


def add_document(conn, matter, title, chunks, path="data/corpus/doc.pdf"):
    """Insert a document and its chunks directly; each chunk is (text, page, vector)."""
    document_id = conn.execute(
        """INSERT INTO documents (matter_id, source_key, title, file_path, sha256, pages, embedding_model)
           VALUES (%s, %s, %s, %s, 'x', 99, 'fake') RETURNING id""", (matter, title, title, path)).fetchone()[0]
    for i, (text, page, vector) in enumerate(chunks):
        conn.execute(
            """INSERT INTO chunks (document_id, matter_id, chunk_index, heading, text, page_start, page_end, ocr,
                                   embedding) VALUES (%s, %s, %s, 'HEADING', %s, %s, %s, false, %s)""",
            (document_id, matter, i, text, page, page, vector))


def nearby(vector, noise, seed):
    """A unit vector close to `vector`: the larger the noise, the further away."""
    v = vector + noise * np.random.default_rng(seed).standard_normal(768)
    return v / np.linalg.norm(v)


def test_returns_top_k_ordered_by_similarity_with_citation_details(conn):
    embedder = FakeEmbedder()
    question = "Can a minor enter into a contract?"
    target = embedder.embed_query(question)
    add_document(conn, "m", "The Indian Contract Act, 1872", [
        ("far", 1, nearby(target, 0.5, 1)),
        ("closest", 13, target),
        ("close", 14, nearby(target, 0.05, 2)),
    ], path="data/corpus/indian-contract-act-1872.pdf")

    hits = search(conn, embedder, question, "m", k=2)

    assert [h.text for h in hits] == ["closest", "close"]
    best = hits[0]
    assert best.score == pytest.approx(1.0, abs=1e-5)
    assert (best.title, best.heading, best.pages, best.ocr) == ("The Indian Contract Act, 1872", "HEADING", "p. 13", False)
    assert best.link == "data/corpus/indian-contract-act-1872.pdf#page=13"
    assert hits[0].score > hits[1].score


def test_never_returns_another_matters_chunks_even_when_closer(conn):
    embedder = FakeEmbedder()
    target = embedder.embed_query("question")
    add_document(conn, "other-client", "Their contract", [("exact match", 1, target)])
    add_document(conn, "our-client", "Our contract", [("weaker match", 1, nearby(target, 0.5, 3))])

    hits = search(conn, embedder, "question", "our-client")

    assert [h.text for h in hits] == ["weaker match"]


def test_filtered_index_search_still_returns_k_results(conn):
    """When Postgres searches through the HNSW index, a plain index scan stops after about 40 candidates
    from all matters and can return fewer than k for a small matter; the iterative scan keeps going.

    (For a small matter Postgres would normally use the matter_id index and an exact sort, so the test
    removes that option. HNSW is chosen in real use when a matter is a large share of the table.)"""
    embedder = FakeEmbedder()
    target = embedder.embed_query("question")
    # 400 chunks of another matter all sit closer to the question than our matter's 5.
    add_document(conn, "big-matter", "Big", [(f"big {i}", 1, nearby(target, 0.02, i)) for i in range(400)])
    add_document(conn, "small-matter", "Small", [(f"small {i}", 1, nearby(target, 0.2, 1000 + i)) for i in range(5)])
    conn.execute("DROP INDEX chunks_matter_id")  # only in this test's temporary schema
    conn.execute("ANALYZE chunks")
    conn.execute("SET enable_seqscan = off")  # leaves the HNSW index as the only way to answer

    hits = search(conn, embedder, "question", "small-matter", k=5)

    assert sorted(h.text for h in hits) == [f"small {i}" for i in range(5)]


def test_empty_matter_returns_nothing(conn):
    assert search(conn, FakeEmbedder(), "anything", "no-such-matter") == []
