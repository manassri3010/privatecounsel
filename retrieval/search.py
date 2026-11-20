"""Find the chunks closest in meaning to a question, within one matter.

Usage: python -m retrieval.search "Can a minor enter into a contract?" [--matter public-corpus] [-k 5]

This is the vector half of the hybrid search in #29; keyword (BM25) search and rank fusion come next.
"""

import argparse
import sys
import textwrap
from dataclasses import dataclass

from ingestion import db

# relaxed_order: with a matter filter, an HNSW index scan returns its ef_search (default 40) nearest
# chunks across all matters and only then filters, which can leave fewer than k. pgvector 0.8's
# iterative scan keeps searching until k matching rows are found; "relaxed" allows slightly
# out-of-order results, which the outer ORDER BY puts right.
SEARCH_SQL = """
WITH nearest AS MATERIALIZED (
    SELECT id, document_id, heading, text, page_start, page_end, ocr, embedding <=> %(query)s AS distance
    FROM chunks
    WHERE matter_id = %(matter)s
    ORDER BY embedding <=> %(query)s
    LIMIT %(k)s
)
SELECT d.title, d.file_path, n.heading, n.text, n.page_start, n.page_end, n.ocr, 1 - n.distance AS score
FROM nearest n JOIN documents d ON d.id = n.document_id
ORDER BY n.distance
"""


@dataclass(frozen=True)
class Hit:
    title: str
    file_path: str
    heading: str
    text: str
    page_start: int
    page_end: int
    ocr: bool
    score: float  # cosine similarity: 1 means the same direction, 0 unrelated

    @property
    def pages(self):
        return f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}"

    @property
    def link(self):
        """Opens the PDF at the first cited page in most PDF viewers and browsers."""
        return f"{self.file_path}#page={self.page_start}"


def search(conn, embedder, question, matter_id, k=5):
    query = embedder.embed_query(question)
    with conn.transaction():
        conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        rows = conn.execute(SEARCH_SQL, {"query": query, "matter": matter_id, "k": k}).fetchall()
    return [Hit(*row) for row in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--matter", default="public-corpus")
    parser.add_argument("-k", type=int, default=5, help="number of chunks to show")
    args = parser.parse_args(argv)

    from ingestion.embed import Embedder  # loads the model; kept out of module import for tests

    hits = search(db.connect(), Embedder(), args.question, args.matter, args.k)
    if not hits:
        print(f"No chunks in matter '{args.matter}'. Run python -m ingestion.ingest first.")
        return 1
    for rank, hit in enumerate(hits, 1):
        print(f"{rank}. [{hit.score:.3f}] {hit.title}, {hit.pages}{'  (OCR)' if hit.ocr else ''}")
        if hit.heading:
            print(f"   {hit.heading}")
        excerpt = " ".join(hit.text.split())
        print(textwrap.indent(textwrap.fill(textwrap.shorten(excerpt, 400), 100), "   "))
        print(f"   {hit.link}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
