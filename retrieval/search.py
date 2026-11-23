"""Find the chunks that best answer a question, within one matter: vector + keyword search (decision 006).

Usage: python -m retrieval.search "Can a minor enter into a contract?" [--matter public-corpus] [-k 5]
                                  [--mode hybrid|vector|keyword]

Vector search finds passages with the same meaning in other words; keyword search finds exact terms
("Order IX Rule 13", "section 19A", a party's name) that a vector can rank low. Each produces its top
CANDIDATES chunks, and reciprocal rank fusion merges the two lists by rank.
"""

import argparse
import sys
import textwrap
from dataclasses import dataclass

from ingestion import db

CANDIDATES = 20  # chunks taken from each search before fusing (and, later, reranking)
RRF_K = 60  # rank-fusion constant from Cormack et al. (2009); damps the gap between ranks 1, 2, 3...
MODES = {"hybrid": (1, 1), "vector": (1, 0), "keyword": (0, 1)}  # (vector weight, keyword weight)

# - vector: hnsw.iterative_scan (set per query) keeps the HNSW index scan going until it has enough
#   chunks of this matter; without it, the scan stops after ef_search (40) chunks from all matters
#   and can return too few. MATERIALIZED + LIMIT keeps the index in use.
# - keyword: the question's words are OR-ed ('minor' | 'enter' | 'contract'), so a chunk needn't contain
#   every word of a natural-language question; ts_rank_cd scores how many appear, how close together,
#   and whether in the heading (weight A). Normalisation 32 scales scores to 0..1.
# - fusion: score = sum over both lists of weight / (RRF_K + rank). A chunk near the top of both beats
#   one at the top of only one. Only ranks are used, so the two searches' different score scales
#   don't need calibrating.
SEARCH_SQL = """
WITH vector AS MATERIALIZED (
    SELECT id, embedding <=> %(embedding)s AS distance
    FROM chunks
    WHERE matter_id = %(matter)s
    ORDER BY embedding <=> %(embedding)s
    LIMIT %(candidates)s
),
vector_ranked AS (
    SELECT id, distance, row_number() OVER (ORDER BY distance, id) AS rank FROM vector
),
keyword_ranked AS (
    SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, q, 32) DESC, id) AS rank
    FROM chunks, (SELECT replace(plainto_tsquery('english', %(question)s)::text, '&', '|')::tsquery AS q) AS query
    WHERE matter_id = %(matter)s AND tsv @@ q
    ORDER BY rank
    LIMIT %(candidates)s
),
fused AS (
    SELECT coalesce(v.id, kw.id) AS id, v.rank AS vector_rank, kw.rank AS keyword_rank, v.distance,
           coalesce(%(vector_weight)s / (%(rrf_k)s + v.rank), 0)
           + coalesce(%(keyword_weight)s / (%(rrf_k)s + kw.rank), 0) AS score
    FROM vector_ranked v FULL OUTER JOIN keyword_ranked kw ON kw.id = v.id
)
SELECT d.title, d.file_path, c.heading, c.text, c.page_start, c.page_end, c.ocr,
       f.score, f.vector_rank, f.keyword_rank, 1 - f.distance AS similarity
FROM fused f
JOIN chunks c ON c.id = f.id
JOIN documents d ON d.id = c.document_id
WHERE f.score > 0
ORDER BY f.score DESC, f.vector_rank NULLS LAST, f.id
LIMIT %(k)s
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
    score: float  # fused score; only meaningful for ordering
    vector_rank: int | None  # position in each search's candidates, None if not among them
    keyword_rank: int | None
    similarity: float | None  # cosine similarity to the question, if found by vector search

    @property
    def pages(self):
        return f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}"

    @property
    def link(self):
        """Opens the PDF at the first cited page in most PDF viewers and browsers."""
        return f"{self.file_path}#page={self.page_start}"


def search(conn, embedder, question, matter_id, k=5, mode="hybrid"):
    vector_weight, keyword_weight = MODES[mode]
    params = {"embedding": embedder.embed_query(question), "question": question, "matter": matter_id,
              "candidates": max(CANDIDATES, k), "k": k, "rrf_k": RRF_K,
              "vector_weight": float(vector_weight), "keyword_weight": float(keyword_weight)}
    with conn.transaction():
        conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        rows = conn.execute(SEARCH_SQL, params).fetchall()
    return [Hit(*row) for row in rows]


def _found_by(hit):
    parts = [f"vector #{hit.vector_rank}" if hit.vector_rank else "", f"keyword #{hit.keyword_rank}" if hit.keyword_rank else ""]
    return ", ".join(p for p in parts if p)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--matter", default="public-corpus")
    parser.add_argument("-k", type=int, default=5, help="number of chunks to show")
    parser.add_argument("--mode", choices=MODES, default="hybrid", help="compare the searches on their own")
    args = parser.parse_args(argv)

    from ingestion.embed import Embedder  # loads the model; kept out of module import for tests

    hits = search(db.connect(), Embedder(), args.question, args.matter, args.k, args.mode)
    if not hits:
        print(f"No chunks found in matter '{args.matter}'. Run python -m ingestion.ingest first.")
        return 1
    for rank, hit in enumerate(hits, 1):
        print(f"{rank}. {hit.title}, {hit.pages}{'  (OCR)' if hit.ocr else ''}   [{_found_by(hit)}]")
        if hit.heading:
            print(f"   {hit.heading}")
        excerpt = " ".join(hit.text.split())
        print(textwrap.indent(textwrap.fill(textwrap.shorten(excerpt, 400), 100), "   "))
        print(f"   {hit.link}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
