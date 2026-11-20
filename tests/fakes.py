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
