"""Test doubles shared by the ingestion and retrieval tests."""

import hashlib

import numpy as np
import pymupdf


class FakeEmbedder:
    """Deterministic stand-in for BGE: one token per word, and a fixed unit vector per text.

    A query embeds to the same vector as an identical passage, so tests can aim at a chunk exactly.
    """

    name = "fake-model@1"

    def count_tokens(self, text):
        return len(text.split())

    def vector(self, text):
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")
        v = np.random.default_rng(seed).standard_normal(768)
        return v / np.linalg.norm(v)

    def embed_passages(self, passages):
        return np.array([self.vector(p) for p in passages])

    def embed_query(self, question):
        return self.vector(question)


def make_pdf(path, pages):
    with pymupdf.open() as doc:
        for text in pages:
            page = doc.new_page()
            page.insert_htmlbox(page.rect + (72, 72, -72, -72), text.replace("\n", "<br>"))  # Unicode font
        doc.save(path)
    return path


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
    return document_id


def nearby(vector, noise, seed):
    """A unit vector close to `vector`: the larger the noise, the further away."""
    v = vector + noise * np.random.default_rng(seed).standard_normal(768)
    return v / np.linalg.norm(v)
