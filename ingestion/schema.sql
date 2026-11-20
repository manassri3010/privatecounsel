-- Tables for ingested documents and their chunks. Safe to run repeatedly.
-- (No migration tool yet: while nothing is in production, changing this file and re-ingesting is simpler.)

CREATE TABLE IF NOT EXISTS documents (
    id              bigserial PRIMARY KEY,
    matter_id       text NOT NULL,          -- the case or client file the document belongs to
    source_key      text NOT NULL,          -- stable name within the matter, e.g. the corpus id
    title           text NOT NULL,
    file_path       text NOT NULL,
    sha256          text NOT NULL,          -- re-ingesting an unchanged file is skipped
    pages           integer NOT NULL,
    embedding_model text NOT NULL,          -- vectors from different models can't be compared
    chunking        text NOT NULL DEFAULT '',  -- chunker version and size, e.g. "v2/400 tokens"
    ingested_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (matter_id, source_key)
);

CREATE TABLE IF NOT EXISTS chunks (
    id          bigserial PRIMARY KEY,
    document_id bigint NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    matter_id   text NOT NULL,              -- copied from documents so search can filter without a join
    chunk_index integer NOT NULL,           -- order within the document
    heading     text NOT NULL,
    text        text NOT NULL,
    page_start  integer NOT NULL,           -- 1-based PDF pages, for citations (file.pdf#page=N)
    page_end    integer NOT NULL,
    ocr         boolean NOT NULL,           -- some of the text came from OCR, so may contain errors
    embedding   vector(768) NOT NULL,       -- bge-base-en-v1.5 (decision 005)
    UNIQUE (document_id, chunk_index)
);

-- Columns added after the first version, for databases created before them.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunking text NOT NULL DEFAULT '';

-- Approximate nearest-neighbour index for cosine distance (the <=> operator).
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_matter_id ON chunks (matter_id);
