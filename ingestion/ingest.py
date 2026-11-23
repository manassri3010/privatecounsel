"""Parse, chunk, embed and store documents in PostgreSQL + pgvector.

Usage: python -m ingestion.ingest [--matter public-corpus] [--kinds act,judgment]
                                  [--manifest data/corpus/manifest.json]

Reads the corpus manifest written by corpus/fetch_corpus.py. Unchanged documents are skipped, and a
changed one replaces its old chunks, so the command can be re-run safely.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from ingestion import db
from ingestion.chunk import CHUNKER_VERSION, chunk_pages
from ingestion.embed import CHUNK_TOKENS, Embedder, passage_text
from ingestion.parse import parse_pdf

DEFAULT_MANIFEST = db.ROOT / "data" / "corpus" / "manifest.json"


def ingest_document(conn, embedder, *, matter_id, source_key, title, path):
    """Store one PDF and its chunks. Returns the number of chunks, or None if it was unchanged.

    Unchanged means the same file, embedding model and chunking; a change to any of them re-ingests.
    """
    path = Path(path)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    chunking = f"v{CHUNKER_VERSION}/{CHUNK_TOKENS} tokens"
    stored = conn.execute(
        "SELECT sha256, embedding_model, chunking FROM documents WHERE matter_id = %s AND source_key = %s",
        (matter_id, source_key)).fetchone()
    if stored == (sha256, embedder.name, chunking):
        return None

    # The slow work (OCR, embedding) happens before the transaction, so it isn't held open for minutes.
    pages = parse_pdf(path)
    chunks = chunk_pages(pages, max_tokens=CHUNK_TOKENS, count_tokens=embedder.count_tokens)
    vectors = embedder.embed_passages([passage_text(title, c.heading, c.text) for c in chunks]) if chunks else []
    ocr_pages = {p.number for p in pages if p.ocr}

    # Replace the old version in one transaction: a search never sees half a document.
    with conn.transaction():
        conn.execute("DELETE FROM documents WHERE matter_id = %s AND source_key = %s", (matter_id, source_key))
        document_id = conn.execute(
            """INSERT INTO documents (matter_id, source_key, title, file_path, sha256, pages, embedding_model,
                                     chunking)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (matter_id, source_key, title, str(path), sha256, len(pages), embedder.name, chunking)).fetchone()[0]
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (document_id, matter_id, chunk_index, heading, text, page_start, page_end,
                                       ocr, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [(document_id, matter_id, i, c.heading, c.text, c.page_start, c.page_end,
                  any(n in ocr_pages for n in range(c.page_start, c.page_end + 1)), vector)
                 for i, (c, vector) in enumerate(zip(chunks, vectors))])
    return len(chunks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--matter", default="public-corpus", help="matter ID to store the documents under")
    parser.add_argument("--kinds", default="act,judgment", help="manifest kinds to ingest, e.g. 'scan'")
    args = parser.parse_args(argv)

    kinds = set(args.kinds.split(","))
    entries = [e for e in json.loads(args.manifest.read_text())["documents"] if e["kind"] in kinds]
    conn = db.connect()
    db.apply_schema(conn)
    embedder = Embedder()

    for entry in entries:
        started = time.time()
        count = ingest_document(conn, embedder, matter_id=args.matter, source_key=entry["id"],
                                title=entry["title"], path=args.manifest.parent / entry["file"])
        status = "unchanged" if count is None else f"{count:4} chunks"
        print(f"{status:>12}  {time.time() - started:5.1f}s  {entry['id']}")

    documents, chunks = conn.execute(
        """SELECT (SELECT count(*) FROM documents WHERE matter_id = %(m)s),
                  (SELECT count(*) FROM chunks WHERE matter_id = %(m)s)""", {"m": args.matter}).fetchone()
    print(f"\nmatter '{args.matter}': {documents} documents, {chunks} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
